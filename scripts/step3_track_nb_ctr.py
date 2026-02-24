#!/usr/bin/env python3
"""
Track 2-3: New-Build (NB) + Cost-to-Replace (CTR) analysis.
Does NOT rerun baseline — preserves existing overprocure_results.json.

Track 2 (newbuild / NB): hydro=0 mixes, uprates ON
  Source: per-ISO EF parquets (data/step2-ef-parquets/step2_ef_{ISO}.parquet)
  Filtered to hydro=0 mixes.
  Purpose: What does hourly matching incentivize from scratch?

Track 3 (cost-to-replace / CTR): all mixes (hydro≤existing), uprates OFF
  Source: per-ISO EF parquets (data/step2-ef-parquets/step2_ef_{ISO}.parquet)
  Purpose: Cost to replace existing clean generation

Checkpoint: Parquet-based. After each (iso, track) completes, results are
  appended to track_scenarios.parquet. On resume, completed (iso, track) pairs
  are read from the parquet header — no full data load needed.

Usage:
  python step3_track_nb_ctr.py              # Medium-only (fast, ~30s)
  python step3_track_nb_ctr.py --full       # All 5,832+ combos (hours)
  python step3_track_nb_ctr.py --iso PJM    # Single ISO
"""

import os
import sys
import time
import argparse
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPT_DIR)

from step3_cost_optimization import (
    ISOS, OUTPUT_THRESHOLDS, REGIONAL_DEMAND_TWH, GRID_MIX_SHARES,
    WHOLESALE_PRICES, FUEL_ADJUSTMENTS, LEVEL_NAME, RESOURCE_TYPES,
    DEMAND_GROWTH_RATES, DEMAND_GROWTH_YEARS, DEMAND_GROWTH_LEVELS,
    THRESHOLD_TARGET_YEARS, DG_UNIQUE_YEARS, _DG_YEAR_TO_THRESHOLDS,
    precompute_base_year_coefficients, get_scenario_prices,
    eval_cost_fast, eval_and_argmin_all, build_winner_scenario,
    build_sensitivity_combos, medium_key, price_mix_batch,
    precompute_all_prices, batch_eval_and_argmin_all,
    HAS_NUMBA,
)

from step3_cost_optimization import (
    _N_COEFFS, _COL_WHOLESALE, _COL_SOL_NEW, _COL_WND_NEW, _COL_CCS_NEW,
    _COL_UPRATE, _COL_GEO, _COL_REMAINING, _COL_BAT4, _COL_BAT8, _COL_LDES,
)

# Per-ISO EF parquet directory (Step 2 output)
EF_ISO_DIR = os.path.join(SCRIPT_DIR, 'data', 'step2-ef-parquets')

# Parquet paths — these ARE the checkpoints
PQ_SCENARIOS_PATH = os.path.join(SCRIPT_DIR, 'dashboard', 'track_scenarios.parquet')
PQ_DG_PATH = os.path.join(SCRIPT_DIR, 'dashboard', 'track_demand_growth.parquet')


def load_completed_tracks(pq_path):
    """Read parquet metadata to find completed (iso, track) pairs.

    Only reads the iso and track columns — O(rows) but tiny memory since
    it's just two string columns from a ~1MB file.

    Also cleans up stale track names (e.g., 'replace' → 'cost_to_replace').
    """
    if not os.path.exists(pq_path):
        return set()
    table = pq.read_table(pq_path, columns=['iso', 'track'])
    iso_col = table.column('iso').to_pylist()
    track_col = table.column('track').to_pylist()

    # Clean up stale 'replace' track name from older runs
    VALID_TRACKS = {'newbuild', 'cost_to_replace'}
    stale_indices = [i for i, t in enumerate(track_col) if t not in VALID_TRACKS]
    if stale_indices:
        stale_names = list(set(track_col[i] for i in stale_indices))
        print(f"  Cleaning {len(stale_indices):,} stale rows with track names: {stale_names}")
        # Re-read full file, filter valid rows, rewrite
        full_table = pq.read_table(pq_path)
        full_track = full_table.column('track').to_pylist()
        keep_mask = pa.array([t in VALID_TRACKS for t in full_track])
        full_table = full_table.filter(keep_mask)
        tmp = pq_path + '.tmp'
        pq.write_table(full_table, tmp, compression='zstd')
        os.replace(tmp, pq_path)
        print(f"  Rewrote checkpoint: {full_table.num_rows:,} rows (removed {len(stale_indices):,} stale)")
        iso_col = full_table.column('iso').to_pylist()
        track_col = full_table.column('track').to_pylist()

    completed = set(zip(iso_col, track_col))
    print(f"  Parquet checkpoint: {len(completed)} (iso, track) pairs completed")
    # Count per pair
    pair_counts = {}
    for iso_val, track_val in zip(iso_col, track_col):
        pair_counts[(iso_val, track_val)] = pair_counts.get((iso_val, track_val), 0) + 1
    for (iso_val, track_val), n in sorted(pair_counts.items()):
        print(f"    {iso_val}/{track_val}: {n:,} scenarios")
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


def flatten_dg_rows(iso, track_name, dg_dict, arrays=None):
    """Flatten demand growth result dict into flat rows for parquet.

    If arrays (PFS arrays) are provided, resolves mix_idx to full resource mix
    for downstream pipeline compatibility.
    """
    rows = []
    for thr_str, sc_dict in dg_dict.items():
        for sc_key, year_data in sc_dict.items():
            for year_str, growth_data in year_data.items():
                for g_level, vals in growth_data.items():
                    mix_idx = vals[0]
                    row = {
                        'iso': iso,
                        'track': track_name,
                        'threshold': float(thr_str),
                        'scenario': sc_key,
                        'year': int(year_str),
                        'growth_level': g_level,
                        'growth_factor': vals[4] if len(vals) > 4 else None,
                        'annual_demand_mwh': vals[5] if len(vals) > 5 else None,
                        'cost_total_cost': vals[1],
                        'cost_effective_cost': vals[2],
                        'cost_incremental': vals[3],
                    }
                    # Resolve resource mix from PFS arrays if available
                    if arrays is not None:
                        cf = int(arrays['clean_firm'][mix_idx])
                        sol = int(arrays['solar'][mix_idx])
                        wnd = int(arrays['wind'][mix_idx])
                        hyd = int(arrays['hydro'][mix_idx])
                        ccs = max(0, 100 - (cf + sol + wnd + hyd))
                        row['mix_clean_firm'] = cf
                        row['mix_solar'] = sol
                        row['mix_wind'] = wnd
                        row['mix_ccs_ccgt'] = ccs
                        row['mix_hydro'] = hyd
                        row['procurement_pct'] = int(arrays['procurement'][mix_idx])
                        row['hourly_match_score'] = float(
                            arrays['hourly_match_score'][mix_idx])
                        row['battery_dispatch_pct'] = int(
                            arrays.get('battery_dispatch_pct', np.zeros(1))[mix_idx])
                        row['battery8_dispatch_pct'] = int(
                            arrays.get('battery8_dispatch_pct', np.zeros(1))[mix_idx])
                        row['ldes_dispatch_pct'] = int(
                            arrays.get('ldes_dispatch_pct', np.zeros(1))[mix_idx])
                    else:
                        row['best_mix_idx'] = mix_idx
                    rows.append(row)
    return rows


def append_to_parquet(new_rows, pq_path):
    """Append rows to parquet, replacing any existing data for the same (iso, track) pairs."""
    if not new_rows:
        return 0

    # Build pyarrow table from new rows
    all_keys = list(dict.fromkeys(k for r in new_rows for k in r))
    new_table = pa.table({k: pa.array([r.get(k) for r in new_rows]) for k in all_keys})

    if os.path.exists(pq_path):
        existing_table = pq.read_table(pq_path)
        # Normalize large_string → string to avoid schema merge failures
        def _normalize_schema(table):
            new_fields = []
            needs_cast = False
            for field in table.schema:
                if field.type == pa.large_string():
                    new_fields.append(pa.field(field.name, pa.string(), nullable=field.nullable))
                    needs_cast = True
                else:
                    new_fields.append(field)
            if needs_cast:
                table = table.cast(pa.schema(new_fields))
            return table
        existing_table = _normalize_schema(existing_table)
        new_table = _normalize_schema(new_table)
        # Find (iso, track) pairs in new data to replace
        new_iso = new_table.column('iso').to_pylist()
        new_track = new_table.column('track').to_pylist()
        new_keys = set(zip(new_iso, new_track))
        # Filter out existing rows that match new keys
        ex_iso = existing_table.column('iso').to_pylist()
        ex_track = existing_table.column('track').to_pylist()
        keep_mask = pa.array([(i, t) not in new_keys for i, t in zip(ex_iso, ex_track)])
        keep_table = existing_table.filter(keep_mask)
        # Unify schemas before concat
        combined_schema = pa.unify_schemas([keep_table.schema, new_table.schema])
        keep_table = keep_table.cast(combined_schema)
        new_table = new_table.cast(combined_schema)
        merged = pa.concat_tables([keep_table, new_table])
        print(f"    Parquet merge: kept {keep_table.num_rows:,} + added {new_table.num_rows:,} = {merged.num_rows:,} total")
    else:
        merged = new_table

    # Atomic write
    tmp = pq_path + '.tmp'
    pq.write_table(merged, tmp, compression='zstd')
    os.replace(tmp, pq_path)
    size_mb = os.path.getsize(pq_path) / (1024 * 1024)
    print(f"    Saved: {pq_path} ({merged.num_rows:,} rows, {size_mb:.1f} MB)")
    return merged.num_rows


def load_iso_ef_parquet(iso):
    """Load numpy arrays for a single ISO from its per-ISO EF parquet.

    Reads from data/step2-ef-parquets/step2_ef_{ISO}.parquet (Step 2 output).
    Returns None if the file doesn't exist or is empty.
    """
    path = os.path.join(EF_ISO_DIR, f'step2_ef_{iso}.parquet')
    if not os.path.exists(path):
        print(f"  WARNING: EF parquet not found for {iso}: {path}")
        return None
    sub = pq.read_table(path)
    if sub.num_rows == 0:
        return None
    print(f"  {iso}: loaded {sub.num_rows:,} EF mixes from {path}")
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
    """Run demand growth sweep for track archetypes (threshold-year paired).

    Each threshold is evaluated only at its SBTi-interpolated target year × L/M/H,
    producing 15 × 3 = 45 DG results per ISO per scenario combo.
    """
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

        # Threshold-year paired: each unique year evaluated once,
        # results stored only for matching thresholds
        for year in DG_UNIQUE_YEARS:
            matched_thresholds = [t for t in _DG_YEAR_TO_THRESHOLDS[year]
                                  if t in arch_thr_mask]
            if not matched_thresholds:
                continue

            for g_level in DEMAND_GROWTH_LEVELS:
                g_rate = iso_rates[g_level]
                gf = (1 + g_rate) ** (year - 2025)
                demand_grown_mwh = demand_twh * gf * 1e6
                tc, ec, _ = price_mix_batch(
                    iso, arch_arrays, sens, demand_twh,
                    target_year=year, growth_rate=g_rate,
                    uprate_cap_override=uprate_cap_override,
                    existing_override=existing_override
                )
                for thr in matched_thresholds:
                    qual_idx = arch_thr_mask[thr]
                    best_local = int(qual_idx[np.argmin(tc[qual_idx])])
                    full_idx = arch_indices[best_local]
                    if str(year) not in thr_year_results[thr]:
                        thr_year_results[thr][str(year)] = {}
                    thr_year_results[thr][str(year)][g_level] = [
                        full_idx,
                        round(float(tc[best_local]), 2),
                        round(float(ec[best_local]), 2),
                        round(float(ec[best_local]) - wholesale, 2),
                        round(gf, 6),
                        round(demand_grown_mwh, 0),
                    ]

        for thr in arch_thr_mask:
            thr_dg[thr][scenario_key] = thr_year_results[thr]

    n_dg_evals = len(DG_UNIQUE_YEARS) * len(DEMAND_GROWTH_LEVELS)
    result = {str(thr): thr_dg[thr] for thr in arch_thr_mask}
    print(f"  {iso:>6} {track_name:>10} DG: {n_arch} archetypes, "
          f"{len(arch_thr_mask)} thresholds, {n_dg_evals} paired evals")
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
    print("  TRACK 2-3: NEW-BUILD (NB) + COST-TO-REPLACE (CTR)")
    print(f"  Mode: {'Full sweep (all combos)' if args.full else 'Medium-only (fast)'}")
    print(f"  EF source: {EF_ISO_DIR}")
    print(f"  Checkpoint: Parquet-based ({PQ_SCENARIOS_PATH})")
    print("=" * 70)
    total_start = time.time()

    # Load completed tracks from parquet checkpoint
    if args.fresh:
        print("  --fresh flag: ignoring existing parquet checkpoint")
        completed_tracks = set()
        # Remove existing parquet files so append_to_parquet starts clean
        for _fp in (PQ_SCENARIOS_PATH, PQ_DG_PATH):
            if os.path.exists(_fp):
                os.remove(_fp)
                print(f"    Removed stale {_fp}")
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

    run_isos = [args.iso] if args.iso else ISOS
    for iso in run_isos:
        demand_twh = REGIONAL_DEMAND_TWH[iso]

        # Load per-ISO EF parquet (Step 2 output)
        iso_arrays = load_iso_ef_parquet(iso)
        if iso_arrays is None:
            print(f"  {iso:>6}: no EF data, skipping")
            continue

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
        # CTR override: hydro stays at existing floor, everything else zeroed
        existing_shares = GRID_MIX_SHARES[iso]
        greenfield_keep_hydro = {
            'clean_firm': 0, 'solar': 0, 'wind': 0, 'ccs_ccgt': 0,
            'hydro': existing_shares['hydro'],
        }

        # Track 2: newbuild / NB (hydro=0, all existing zeroed, uprates ON)
        if (iso, 'newbuild') not in completed_tracks:
            h0_mask = iso_arrays['hydro'] == 0
            n_h0 = h0_mask.sum()
            if n_h0 > 0:
                h0_idx = np.where(h0_mask)[0]
                nb_arrays = {k: iso_arrays[k][h0_idx] for k in iso_arrays}

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

        # Track 3: cost-to-replace / CTR (hydro at existing floor, everything else zeroed, uprates OFF)
        if (iso, 'cost_to_replace') not in completed_tracks:
            # Compute 2050 high-demand hydro floor
            hydro_existing_share = GRID_MIX_SHARES[iso]['hydro']
            high_rate = DEMAND_GROWTH_RATES[iso].get('High',
                        DEMAND_GROWTH_RATES[iso].get('high', 0))
            target_year = max(DEMAND_GROWTH_YEARS) if DEMAND_GROWTH_YEARS else 2050
            years_of_growth = target_year - 2025
            demand_scale_2050 = (1 + high_rate) ** years_of_growth
            hydro_floor_raw = hydro_existing_share / demand_scale_2050 if demand_scale_2050 > 0 else 0
            # Physics grid uses integer % steps — snap floor down to match
            hydro_floor = int(hydro_floor_raw)
            n_before = len(iso_arrays['hydro'])
            h_mask = iso_arrays['hydro'] >= hydro_floor
            n_pass = int(h_mask.sum())
            if hydro_floor > 0 and n_pass > 0 and n_pass < n_before:
                h_idx = np.where(h_mask)[0]
                ctr_arrays = {k: iso_arrays[k][h_idx] for k in iso_arrays}
                print(f"    {iso} cost_to_replace: hydro>={hydro_floor}% (2050 high-demand floor, "
                      f"raw={hydro_floor_raw:.2f}%) "
                      f"{n_before:,} → {n_pass:,} ({100*(n_before-n_pass)/n_before:.1f}% pruned)")
            elif hydro_floor > 0 and n_pass == 0:
                print(f"    {iso} cost_to_replace: no mixes with hydro>={hydro_floor}%, skipping")
                continue
            else:
                # hydro_floor == 0 or all mixes qualify — use all mixes
                if hydro_floor == 0:
                    print(f"    {iso} cost_to_replace: hydro floor rounds to 0% "
                          f"(raw={hydro_floor_raw:.2f}%) — using all {n_before:,} mixes")
                ctr_arrays = iso_arrays

            ctr_data, ctr_arch = run_track(
                'cost_to_replace', iso, ctr_arrays, demand_twh, combos,
                uprate_cap_override=0, existing_override=greenfield_keep_hydro)

            # Save to parquet immediately
            sc_rows = flatten_track_rows(iso, 'cost_to_replace', ctr_data)
            append_to_parquet(sc_rows, PQ_SCENARIOS_PATH)

            if ctr_arch:
                ctr_dg = run_track_demand_growth(
                    'cost_to_replace', iso, ctr_arrays, ctr_arch, combos,
                    uprate_cap_override=0, existing_override=greenfield_keep_hydro)
                dg_rows = flatten_dg_rows(iso, 'cost_to_replace', ctr_dg)
                if dg_rows:
                    append_to_parquet(dg_rows, PQ_DG_PATH)
        else:
            print(f"  {iso:>6}    cost_to_replace: skipped (in parquet)")

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
