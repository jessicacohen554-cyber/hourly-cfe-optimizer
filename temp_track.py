#!/usr/bin/env python3
"""
Temp track analysis: Compute newbuild + replace tracks incrementally.
Does NOT rerun baseline — preserves existing overprocure_results.json.

Track 1 (newbuild): hydro=0 mixes, uprates ON
  Source: expanded EF (27M), filtered to hydro=0 (~7.2M mixes)
  Purpose: What does hourly matching incentivize?

Track 2 (replace): all mixes (hydro≤existing), uprates OFF
  Source: original EF backup (8.6M mixes with floor filter)
  Purpose: Cost to replace existing clean generation

Checkpoint: Parquet-based. After each (iso, track) completes, results are
  appended to track_scenarios.parquet. On resume, completed (iso, track) pairs
  are read from the parquet header — no full data load needed.

Usage:
  python temp_track.py              # Medium-only (fast, ~30s)
  python temp_track.py --full       # All 5,832+ combos (hours)
  python temp_track.py --iso PJM    # Single ISO
"""

import os
import sys
import time
import argparse
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from step3_cost_optimization import (
    ISOS, OUTPUT_THRESHOLDS, REGIONAL_DEMAND_TWH, GRID_MIX_SHARES,
    WHOLESALE_PRICES, FUEL_ADJUSTMENTS, LEVEL_NAME, RESOURCE_TYPES,
    DEMAND_GROWTH_RATES, DEMAND_GROWTH_YEARS, DEMAND_GROWTH_LEVELS,
    precompute_base_year_coefficients, get_scenario_prices,
    eval_cost_fast, eval_and_argmin_all, build_winner_scenario,
    build_sensitivity_combos, medium_key, price_mix_batch,
    precompute_all_prices, batch_eval_and_argmin_all,
    TRACK_RESULTS_PATH, HAS_NUMBA,
)

from step3_cost_optimization import (
    _N_COEFFS, _COL_WHOLESALE, _COL_SOL_NEW, _COL_WND_NEW, _COL_CCS_NEW,
    _COL_UPRATE, _COL_GEO, _COL_REMAINING, _COL_BAT4, _COL_BAT8, _COL_LDES,
)

EF_EXPANDED_PATH = os.path.join(SCRIPT_DIR, 'data', 'pfs_post_ef.parquet')
EF_BACKUP_PATH = os.path.join(SCRIPT_DIR, 'data', 'pfs_post_ef_backup.parquet')

# Parquet paths — these ARE the checkpoints
PQ_SCENARIOS_PATH = os.path.join(SCRIPT_DIR, 'dashboard', 'track_scenarios.parquet')
PQ_DG_PATH = os.path.join(SCRIPT_DIR, 'dashboard', 'track_demand_growth.parquet')


def load_completed_tracks(pq_path):
    """Read parquet metadata to find completed (iso, track) pairs.

    Only reads the iso and track columns — O(rows) but tiny memory since
    it's just two string columns from a ~1MB file.
    """
    if not os.path.exists(pq_path):
        return set()
    import pandas as pd
    df = pd.read_parquet(pq_path, columns=['iso', 'track'])
    completed = set(zip(df['iso'], df['track']))
    print(f"  Parquet checkpoint: {len(completed)} (iso, track) pairs completed")
    for iso, track in sorted(completed):
        n = len(df[(df['iso'] == iso) & (df['track'] == track)])
        print(f"    {iso}/{track}: {n:,} scenarios")
    return completed


def flatten_track_rows(iso, track_name, result_dict):
    """Flatten nested track result dict into flat rows for parquet."""
    rows = []
    for thr_str, thr_val in result_dict.items():
        for sc_key, sc in thr_val.get('scenarios', {}).items():
            row = {
                'iso': iso,
                'track': track_name,
                'threshold': float(thr_str),
                'scenario': sc_key,
            }
            for k, v in sc.get('resource_mix', {}).items():
                row[f'mix_{k}'] = v
            row['procurement_pct'] = sc.get('procurement_pct')
            row['hourly_match_score'] = sc.get('hourly_match_score')
            row['battery_dispatch_pct'] = sc.get('battery_dispatch_pct')
            row['battery8_dispatch_pct'] = sc.get('battery8_dispatch_pct')
            row['ldes_dispatch_pct'] = sc.get('ldes_dispatch_pct')
            for k, v in sc.get('costs', {}).items():
                row[f'cost_{k}'] = v
            for k, v in sc.get('tranche_costs', {}).items():
                row[f'tranche_{k}'] = v
            for k, v in sc.get('gas_backup', {}).items():
                row[f'gas_{k}'] = v
            rows.append(row)
    return rows


def flatten_dg_rows(iso, track_name, dg_dict):
    """Flatten demand growth result dict into flat rows for parquet."""
    rows = []
    for thr_str, sc_dict in dg_dict.items():
        for sc_key, year_data in sc_dict.items():
            for year_str, growth_data in year_data.items():
                for g_level, vals in growth_data.items():
                    rows.append({
                        'iso': iso,
                        'track': track_name,
                        'threshold': float(thr_str),
                        'scenario': sc_key,
                        'year': int(year_str),
                        'growth_level': g_level,
                        'best_mix_idx': vals[0],
                        'total_cost': vals[1],
                        'effective_cost': vals[2],
                        'incremental_cost': vals[3],
                    })
    return rows


def append_to_parquet(new_rows, pq_path):
    """Append rows to parquet, replacing any existing data for the same (iso, track) pairs."""
    import pandas as pd

    df_new = pd.DataFrame(new_rows)
    if len(df_new) == 0:
        return 0

    if os.path.exists(pq_path):
        df_existing = pd.read_parquet(pq_path)
        new_keys = set(zip(df_new['iso'], df_new['track']))
        mask = ~pd.Series(list(zip(df_existing['iso'], df_existing['track']))).apply(
            lambda x: x in new_keys)
        df_keep = df_existing[mask.values]
        df = pd.concat([df_keep, df_new], ignore_index=True)
        print(f"    Parquet merge: kept {len(df_keep):,} + added {len(df_new):,} = {len(df):,} total")
    else:
        df = df_new

    # Atomic write
    tmp = pq_path + '.tmp'
    df.to_parquet(tmp, index=False, compression='zstd')
    os.replace(tmp, pq_path)
    size_mb = os.path.getsize(pq_path) / (1024 * 1024)
    print(f"    Saved: {pq_path} ({len(df):,} rows, {size_mb:.1f} MB)")
    return len(df)


def load_iso_arrays(table, iso):
    """Load numpy arrays for a single ISO from a pyarrow table."""
    mask = pc.equal(table.column('iso'), iso)
    sub = table.filter(mask)
    if sub.num_rows == 0:
        return None
    return {
        'clean_firm': sub.column('clean_firm').to_numpy(),
        'solar': sub.column('solar').to_numpy(),
        'wind': sub.column('wind').to_numpy(),
        'hydro': sub.column('hydro').to_numpy(),
        'procurement_pct': sub.column('procurement_pct').to_numpy(),
        'battery_dispatch_pct': sub.column('battery_dispatch_pct').to_numpy(),
        'battery8_dispatch_pct': (sub.column('battery8_dispatch_pct').to_numpy()
                                   if 'battery8_dispatch_pct' in sub.column_names
                                   else np.zeros(sub.num_rows, dtype=np.int64)),
        'ldes_dispatch_pct': sub.column('ldes_dispatch_pct').to_numpy(),
        'hourly_match_score': sub.column('hourly_match_score').to_numpy(),
    }


def pareto_prune_fast(coeff_matrix, constant, scores, thresholds):
    """Vectorized price-bound Pareto pruning per threshold.

    For each threshold t, among qualifying mixes (score >= t):
    1. Compute each mix's cost under lowest possible prices (min_cost)
       and under highest possible prices (max_cost)
    2. Find the best mix's max_cost (upper bound on best achievable)
    3. Prune any mix whose min_cost > that bound (can never win)

    Uses precomputed price ranges from sensitivity combos. O(N) per threshold.

    Returns: boolean mask of NON-dominated mixes (True = keep)
    """
    N = len(scores)
    if N < 100:
        return np.ones(N, dtype=bool)

    keep = np.ones(N, dtype=bool)

    # Price ranges across all ISOs and sensitivity combos (empirically measured)
    # Cols: wholesale, sol_new, wnd_new, ccs_new, uprate, geo, remaining, bat4, bat8, ldes
    min_prices = np.array([20, 40, 30, 52, 15, 0, 52, 69, 77, 116], dtype=np.float64)
    max_prices = np.array([50, 82, 83, 164, 15, 116, 84, 144, 179, 267], dtype=np.float64)

    for thr in thresholds:
        qual_mask = (scores >= thr) & keep
        qual_idx = np.where(qual_mask)[0]
        Q = len(qual_idx)
        if Q < 2:
            continue

        q_coeff = coeff_matrix[qual_idx]  # (Q, 10)
        q_const = constant[qual_idx]      # (Q,)

        # Min possible cost = q_coeff @ min_prices + q_const
        min_cost = q_coeff @ min_prices + q_const
        # Max possible cost = q_coeff @ max_prices + q_const
        max_cost = q_coeff @ max_prices + q_const

        # Best achievable upper bound: lowest max_cost among qualifying mixes
        best_max_cost = np.min(max_cost)

        # Prune: any mix whose min cost exceeds the best worst-case
        dominated = min_cost > best_max_cost
        n_pruned = dominated.sum()
        if n_pruned > 0:
            keep[qual_idx[dominated]] = False

    return keep


def run_track(track_name, iso, arrays, demand_twh, combos, uprate_cap_override=None,
              existing_override=None, apply_dominance_filter=True):
    """Run cost optimization for a track (newbuild or replace).

    Returns:
        track_data: dict of {threshold_str: {'scenarios': {key: scenario_dict}}}
        arch_set: set of winning mix indices (archetypes)
    """
    N = len(arrays['clean_firm'])
    if N == 0:
        return {}, set()

    scores = arrays['hourly_match_score'].astype(np.float64)

    # Pre-compute threshold indices
    thr_indices = {}
    for thr in OUTPUT_THRESHOLDS:
        idx = np.where(scores >= thr)[0]
        if len(idx) > 0:
            thr_indices[thr] = idx

    if not thr_indices:
        return {}, set()

    # Pre-compute coefficients
    coeff_matrix, constant, extras = precompute_base_year_coefficients(
        iso, arrays, demand_twh, uprate_cap_override=uprate_cap_override,
        existing_override=existing_override)

    # Apply Pareto dominance filter — prune mixes that can never win
    if apply_dominance_filter and N > 500:
        active_thr_list = sorted(thr_indices.keys())
        keep_mask = pareto_prune_fast(coeff_matrix, constant, scores, active_thr_list)
        n_pruned = N - keep_mask.sum()
        if n_pruned > 0:
            keep_idx = np.where(keep_mask)[0]
            arrays = {k: arrays[k][keep_idx] for k in arrays}
            coeff_matrix = coeff_matrix[keep_idx]
            constant = constant[keep_idx]
            scores = scores[keep_idx]
            extras = {k: (v[keep_idx] if isinstance(v, np.ndarray) and v.shape[0] == N else v)
                      for k, v in extras.items()}
            N_new = len(keep_idx)
            print(f"    {iso} {track_name}: Pareto pruned {n_pruned:,} / {N:,} "
                  f"({100*n_pruned/N:.1f}%) → {N_new:,} mixes remain")
            N = N_new
            # Recompute threshold indices on pruned arrays
            thr_indices = {}
            for thr in OUTPUT_THRESHOLDS:
                idx = np.where(scores >= thr)[0]
                if len(idx) > 0:
                    thr_indices[thr] = idx

    active_thresholds = sorted(thr_indices.keys())
    thresholds_desc = np.array(sorted(active_thresholds, reverse=True), dtype=np.float64)
    thr_pos = {float(thresholds_desc[k]): k for k in range(len(thresholds_desc))}

    thr_data = {thr: {'scenarios': {}} for thr in active_thresholds}
    arch_set = set()
    n_combos = len(combos)
    iso_start = time.time()

    # Pre-compute all price vectors at once
    price_matrix, wholesale_arr, nuclear_arr, ccs_arr = precompute_all_prices(iso, combos)
    price_time = time.time() - iso_start
    print(f"    {iso} {track_name}: prices pre-computed ({price_time:.1f}s), "
          f"launching batched eval ({N:,} mixes × {n_combos:,} combos)...")

    # Single batched Numba call — tiled for cache efficiency
    batch_start = time.time()
    all_best_idxs, all_best_vals = batch_eval_and_argmin_all(
        coeff_matrix, constant, price_matrix, scores, thresholds_desc)
    batch_elapsed = time.time() - batch_start
    print(f"    {iso} {track_name}: batched eval+argmin done in {batch_elapsed:.1f}s")

    # Build winner scenarios from batched results
    build_start = time.time()
    for j, (scenario_key, sens) in enumerate(combos):
        for thr in active_thresholds:
            k = thr_pos[float(thr)]
            if all_best_vals[j, k] == np.inf:
                continue
            best_idx = int(all_best_idxs[j, k])
            tc_val = float(all_best_vals[j, k])
            arch_set.add(best_idx)

            scenario = build_winner_scenario(
                arrays, extras, best_idx, sens, iso, demand_twh,
                tc_val, float(wholesale_arr[j]),
                float(nuclear_arr[j]), float(ccs_arr[j]))
            thr_data[thr]['scenarios'][scenario_key] = scenario
    build_elapsed = time.time() - build_start
    print(f"    {iso} {track_name}: winner scenarios built in {build_elapsed:.1f}s")

    result = {str(thr): thr_data[thr] for thr in active_thresholds}
    elapsed = time.time() - iso_start
    print(f"  {iso:>6} {track_name:>10}: {N:,} mixes, "
          f"{len(active_thresholds)} thresholds, {len(arch_set)} archetypes — {elapsed:.0f}s")
    return result, arch_set


def run_track_demand_growth(track_name, iso, arrays, arch_set, combos,
                             uprate_cap_override=None, existing_override=None):
    """Run demand growth sweep for track archetypes."""
    demand_twh = REGIONAL_DEMAND_TWH[iso]
    iso_rates = DEMAND_GROWTH_RATES[iso]

    arch_indices = sorted(arch_set)
    n_arch = len(arch_indices)
    if n_arch == 0:
        return {}

    arch_arrays = {k: arrays[k][arch_indices] for k in arrays}
    arch_scores = arch_arrays['hourly_match_score']

    arch_thr_mask = {}
    for thr in OUTPUT_THRESHOLDS:
        qualifying = np.where(arch_scores >= thr)[0]
        if len(qualifying) > 0:
            arch_thr_mask[thr] = qualifying

    thr_dg = {thr: {} for thr in arch_thr_mask}

    for scenario_key, sens in combos:
        wholesale = max(5, WHOLESALE_PRICES[iso] +
                        FUEL_ADJUSTMENTS[iso][LEVEL_NAME[sens['fuel']]])
        thr_year_results = {thr: {} for thr in arch_thr_mask}

        for year in DEMAND_GROWTH_YEARS:
            thr_growth_results = {thr: {} for thr in arch_thr_mask}
            for g_level in DEMAND_GROWTH_LEVELS:
                g_rate = iso_rates[g_level]
                tc, ec, _ = price_mix_batch(
                    iso, arch_arrays, sens, demand_twh,
                    target_year=year, growth_rate=g_rate,
                    uprate_cap_override=uprate_cap_override,
                    existing_override=existing_override
                )
                for thr in arch_thr_mask:
                    qual_idx = arch_thr_mask[thr]
                    best_local = int(qual_idx[np.argmin(tc[qual_idx])])
                    full_idx = arch_indices[best_local]
                    thr_growth_results[thr][g_level] = [
                        full_idx,
                        round(float(tc[best_local]), 2),
                        round(float(ec[best_local]), 2),
                        round(float(ec[best_local]) - wholesale, 2),
                    ]
            for thr in arch_thr_mask:
                thr_year_results[thr][str(year)] = thr_growth_results[thr]

        for thr in arch_thr_mask:
            thr_dg[thr][scenario_key] = thr_year_results[thr]

    result = {str(thr): thr_dg[thr] for thr in arch_thr_mask}
    print(f"  {iso:>6} {track_name:>10} DG: {n_arch} archetypes, "
          f"{len(arch_thr_mask)} thresholds")
    return result


def main():
    parser = argparse.ArgumentParser(description='Incremental track analysis')
    parser.add_argument('--full', action='store_true',
                        help='Run all sensitivity combos (hours). Default: Medium-only.')
    parser.add_argument('--fresh', action='store_true',
                        help='Ignore checkpoint and start from scratch.')
    parser.add_argument('--iso', type=str, default=None,
                        help='Run only this ISO (e.g., PJM). Default: all ISOs.')
    args = parser.parse_args()

    print("=" * 70)
    print("  INCREMENTAL TRACK ANALYSIS")
    print(f"  Mode: {'Full sweep (all combos)' if args.full else 'Medium-only (fast)'}")
    print(f"  Checkpoint: Parquet-based ({PQ_SCENARIOS_PATH})")
    print("=" * 70)
    total_start = time.time()

    # Load completed tracks from parquet checkpoint
    if args.fresh:
        print("  --fresh flag: ignoring existing parquet checkpoint")
        completed_tracks = set()
    else:
        completed_tracks = load_completed_tracks(PQ_SCENARIOS_PATH)

    # Warm up Numba JIT (batched function)
    if HAS_NUMBA:
        from step3_cost_optimization import _batch_eval_and_argmin, _eval_cost_numba, _argmin_bucketed, _N_COEFFS
        _dcm = np.zeros((2, _N_COEFFS))
        _dc = np.zeros(2)
        _dp = np.zeros(_N_COEFFS)
        _ds = np.array([50.0, 100.0])
        _dt = np.array([100.0, 50.0])
        _eval_cost_numba(_dcm, _dc, _dp)
        _argmin_bucketed(np.zeros(2), _ds, _dt)
        _dpm = np.zeros((2, _N_COEFFS))
        _batch_eval_and_argmin(_dcm, _dc, _dpm, _ds, _dt)
        print(f"  Numba JIT warmup complete (batched mode)")

    # Load EF sources
    print("\nLoading EF sources...")
    ef_expanded = pq.read_table(EF_EXPANDED_PATH)
    print(f"  Expanded EF: {ef_expanded.num_rows:,} rows")

    run_isos = [args.iso] if args.iso else ISOS
    for iso in run_isos:
        demand_twh = REGIONAL_DEMAND_TWH[iso]

        # Build combos
        if args.full:
            combos = build_sensitivity_combos(iso)
        else:
            mk = medium_key(iso)
            geo = 'M' if iso == 'CAISO' else None
            sens = {'ren': 'M', 'firm': 'M', 'batt': 'M', 'ldes_lvl': 'M',
                    'ccs': 'M', 'q45': '1', 'fuel': 'M', 'tx': 'M', 'geo': geo}
            combos = [(mk, sens)]

        # Greenfield existing override: all clean resources zeroed
        greenfield_all = {'clean_firm': 0, 'solar': 0, 'wind': 0, 'ccs_ccgt': 0, 'hydro': 0}
        # Replace override: hydro stays at existing floor, everything else zeroed
        existing_shares = GRID_MIX_SHARES[iso]
        greenfield_keep_hydro = {
            'clean_firm': 0, 'solar': 0, 'wind': 0, 'ccs_ccgt': 0,
            'hydro': existing_shares['hydro'],
        }

        # Track 1: newbuild (hydro=0, all existing zeroed, uprates ON)
        if (iso, 'newbuild') not in completed_tracks:
            all_arrays = load_iso_arrays(ef_expanded, iso)
            if all_arrays is not None:
                h0_mask = all_arrays['hydro'] == 0
                n_h0 = h0_mask.sum()
                if n_h0 > 0:
                    h0_idx = np.where(h0_mask)[0]
                    nb_arrays = {k: all_arrays[k][h0_idx] for k in all_arrays}

                    nb_data, nb_arch = run_track(
                        'newbuild', iso, nb_arrays, demand_twh, combos,
                        existing_override=greenfield_all)

                    # Save to parquet immediately
                    sc_rows = flatten_track_rows(iso, 'newbuild', nb_data)
                    append_to_parquet(sc_rows, PQ_SCENARIOS_PATH)

                    if nb_arch:
                        nb_dg = run_track_demand_growth(
                            'newbuild', iso, nb_arrays, nb_arch, combos,
                            existing_override=greenfield_all)
                        dg_rows = flatten_dg_rows(iso, 'newbuild', nb_dg)
                        if dg_rows:
                            append_to_parquet(dg_rows, PQ_DG_PATH)
                else:
                    print(f"  {iso:>6}   newbuild: no hydro=0 mixes")
        else:
            print(f"  {iso:>6}   newbuild: skipped (in parquet)")

        # Track 2: replace (hydro at existing floor, everything else zeroed, uprates OFF)
        if (iso, 'replace') not in completed_tracks:
            rp_arrays = load_iso_arrays(ef_expanded, iso)
            if rp_arrays is not None:
                # Compute 2050 high-demand hydro floor
                hydro_existing_share = GRID_MIX_SHARES[iso]['hydro']
                high_rate = DEMAND_GROWTH_RATES[iso].get('High',
                            DEMAND_GROWTH_RATES[iso].get('high', 0))
                target_year = max(DEMAND_GROWTH_YEARS) if DEMAND_GROWTH_YEARS else 2050
                years_of_growth = target_year - 2025
                demand_scale_2050 = (1 + high_rate) ** years_of_growth
                hydro_floor = hydro_existing_share / demand_scale_2050 if demand_scale_2050 > 0 else 0
                n_before = len(rp_arrays['hydro'])
                h_mask = rp_arrays['hydro'] >= hydro_floor
                n_pass = int(h_mask.sum())
                if n_pass > 0 and n_pass < n_before:
                    h_idx = np.where(h_mask)[0]
                    rp_arrays = {k: rp_arrays[k][h_idx] for k in rp_arrays}
                    print(f"    {iso} replace: hydro>={hydro_floor:.1f}% (2050 high-demand floor) "
                          f"{n_before:,} → {n_pass:,} ({100*(n_before-n_pass)/n_before:.1f}% pruned)")
                elif n_pass == 0:
                    print(f"    {iso} replace: no mixes with hydro>={hydro_floor:.1f}%, skipping")
                    continue

                rp_data, rp_arch = run_track(
                    'replace', iso, rp_arrays, demand_twh, combos,
                    uprate_cap_override=0, existing_override=greenfield_keep_hydro)

                # Save to parquet immediately
                sc_rows = flatten_track_rows(iso, 'replace', rp_data)
                append_to_parquet(sc_rows, PQ_SCENARIOS_PATH)

                if rp_arch:
                    rp_dg = run_track_demand_growth(
                        'replace', iso, rp_arrays, rp_arch, combos,
                        uprate_cap_override=0, existing_override=greenfield_keep_hydro)
                    dg_rows = flatten_dg_rows(iso, 'replace', rp_dg)
                    if dg_rows:
                        append_to_parquet(dg_rows, PQ_DG_PATH)
        else:
            print(f"  {iso:>6}    replace: skipped (in parquet)")

    total_elapsed = time.time() - total_start
    pq_size = os.path.getsize(PQ_SCENARIOS_PATH) / (1024 * 1024) if os.path.exists(PQ_SCENARIOS_PATH) else 0
    dg_size = os.path.getsize(PQ_DG_PATH) / (1024 * 1024) if os.path.exists(PQ_DG_PATH) else 0
    print(f"\n{'='*70}")
    print(f"  TRACK ANALYSIS COMPLETE in {total_elapsed:.0f}s")
    print(f"  Scenarios: {PQ_SCENARIOS_PATH} ({pq_size:.1f} MB)")
    print(f"  Demand Growth: {PQ_DG_PATH} ({dg_size:.1f} MB)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
