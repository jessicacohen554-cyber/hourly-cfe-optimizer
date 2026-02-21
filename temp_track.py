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

Usage:
  python temp_track.py              # Medium-only (fast, ~30s)
  python temp_track.py --full       # All 5,832+ combos (hours)
"""

import json
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

EF_EXPANDED_PATH = os.path.join(SCRIPT_DIR, 'data', 'pfs_post_ef.parquet')
EF_BACKUP_PATH = os.path.join(SCRIPT_DIR, 'data', 'pfs_post_ef_backup.parquet')
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, 'data', 'track_checkpoint.json')

from step3_cost_optimization import (
    _N_COEFFS, _COL_WHOLESALE, _COL_SOL_NEW, _COL_WND_NEW, _COL_CCS_NEW,
    _COL_UPRATE, _COL_GEO, _COL_REMAINING, _COL_BAT4, _COL_BAT8, _COL_LDES,
)


def save_checkpoint(track_output, completed_steps):
    """Save progress checkpoint after each ISO/track completes."""
    ckpt = {
        'track_output': track_output,
        'completed_steps': completed_steps,
        'timestamp': time.time(),
    }
    tmp = CHECKPOINT_PATH + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(ckpt, f, separators=(',', ':'))
    os.replace(tmp, CHECKPOINT_PATH)
    print(f"    [checkpoint saved: {len(completed_steps)} steps completed]")


def load_checkpoint():
    """Load checkpoint if it exists. Returns (track_output, completed_steps) or (None, set())."""
    if not os.path.exists(CHECKPOINT_PATH):
        return None, set()
    try:
        with open(CHECKPOINT_PATH) as f:
            ckpt = json.load(f)
        completed = set(tuple(s) for s in ckpt['completed_steps'])
        age_min = (time.time() - ckpt.get('timestamp', 0)) / 60
        print(f"  Loaded checkpoint: {len(completed)} steps completed ({age_min:.0f} min ago)")
        return ckpt['track_output'], completed
    except Exception as e:
        print(f"  WARNING: Failed to load checkpoint: {e}")
        return None, set()


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
              existing_override=None, track_output=None, completed_steps=None,
              apply_dominance_filter=True):
    """Run cost optimization for a track (newbuild or replace).

    Checkpoints after every 500 combos (full sweep) by writing partial results
    to track_output and saving to disk.

    Args:
        apply_dominance_filter: if True, prune dominated mixes before evaluation.
            Dramatically reduces evaluation count (typically 50-80% reduction).

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

    # Check for existing partial results from a checkpoint
    existing_scenarios = set()
    if (track_output and iso in track_output.get('results', {})
            and track_name in track_output['results'][iso]):
        existing_data = track_output['results'][iso][track_name]
        for thr_str, thr_val in existing_data.items():
            for sk in thr_val.get('scenarios', {}):
                existing_scenarios.add(sk)
        if existing_scenarios:
            print(f"    {iso} {track_name}: resuming, {len(existing_scenarios)} scenarios cached")

    thr_data = {thr: {'scenarios': {}} for thr in active_thresholds}
    # Pre-populate from checkpoint
    if existing_scenarios and track_output:
        existing_data = track_output['results'][iso].get(track_name, {})
        for thr in active_thresholds:
            thr_str = str(thr)
            if thr_str in existing_data:
                thr_data[thr]['scenarios'] = dict(existing_data[thr_str].get('scenarios', {}))

    arch_set = set()
    n_combos = len(combos)
    iso_start = time.time()

    # Filter to uncomputed combos only
    remaining_combos = [(sk, sens) for sk, sens in combos if sk not in existing_scenarios]
    n_remaining = len(remaining_combos)
    n_skipped = n_combos - n_remaining

    if n_skipped > 0:
        print(f"    {iso} {track_name}: skipping {n_skipped} cached, evaluating {n_remaining}")

    if n_remaining > 0:
        # Pre-compute all price vectors at once
        price_matrix, wholesale_arr, nuclear_arr, ccs_arr = precompute_all_prices(iso, remaining_combos)
        price_time = time.time() - iso_start
        print(f"    {iso} {track_name}: prices pre-computed ({price_time:.1f}s), "
              f"launching batched eval ({N:,} mixes × {n_remaining:,} combos)...")

        # Single batched Numba call — tiled for cache efficiency
        batch_start = time.time()
        all_best_idxs, all_best_vals = batch_eval_and_argmin_all(
            coeff_matrix, constant, price_matrix, scores, thresholds_desc)
        batch_elapsed = time.time() - batch_start
        print(f"    {iso} {track_name}: batched eval+argmin done in {batch_elapsed:.1f}s")

        # Build winner scenarios from batched results
        build_start = time.time()
        for j, (scenario_key, sens) in enumerate(remaining_combos):
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
    args = parser.parse_args()

    mode = 'full' if args.full else 'medium_only'

    print("=" * 70)
    print("  INCREMENTAL TRACK ANALYSIS")
    print(f"  Mode: {'Full sweep (all combos)' if args.full else 'Medium-only (fast)'}")
    print("=" * 70)
    total_start = time.time()

    # Load checkpoint if available (and mode matches)
    if not args.fresh:
        ckpt_output, completed_steps = load_checkpoint()
        if ckpt_output and ckpt_output.get('meta', {}).get('mode') == mode:
            track_output = ckpt_output
            print(f"  Resuming from checkpoint — skipping {len(completed_steps)} completed steps")
        else:
            if ckpt_output:
                print(f"  Checkpoint mode mismatch (was {ckpt_output.get('meta',{}).get('mode')}, "
                      f"now {mode}) — starting fresh")
            track_output = None
            completed_steps = set()
    else:
        print("  --fresh flag: ignoring any existing checkpoint")
        track_output = None
        completed_steps = set()

    # Initialize output structure if no valid checkpoint
    if track_output is None:
        track_output = {
            'meta': {
                'tracks': {
                    'newbuild': {
                        'description': 'New-build requirement for hourly matching',
                        'hydro': 'excluded (hydro=0 mixes only)',
                        'existing_clean': 'zeroed (all existing CF/solar/wind/CCS = 0)',
                        'uprates': 'on (uprate tranche active as cheapest new-build)',
                        'purpose': 'What does hourly matching incentivize to BUILD from scratch?',
                    },
                    'replace': {
                        'description': 'Cost to replace all existing clean generation',
                        'hydro': 'included (existing floor, wholesale-priced)',
                        'existing_clean': 'zeroed (all existing CF/solar/wind/CCS = 0)',
                        'uprates': 'off (uprate_cap=0, no uprate tranche)',
                        'purpose': 'True greenfield cost — only hydro is existing, everything else new-build',
                    },
                },
                'mode': mode,
                'thresholds': OUTPUT_THRESHOLDS,
                'resource_types': RESOURCE_TYPES,
            },
            'results': {},
            'demand_growth': {'newbuild': {}, 'replace': {}},
        }
        completed_steps = set()

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

    # Replace track also uses expanded EF (dataset was expanded on purpose)
    # but filters to hydro > 0 since replace track needs existing hydro

    for iso in ISOS:
        demand_twh = REGIONAL_DEMAND_TWH[iso]
        if iso not in track_output['results']:
            track_output['results'][iso] = {}

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
        step_key_nb = (iso, 'newbuild')
        if step_key_nb not in completed_steps:
            all_arrays = load_iso_arrays(ef_expanded, iso)
            if all_arrays is not None:
                h0_mask = all_arrays['hydro'] == 0
                n_h0 = h0_mask.sum()
                if n_h0 > 0:
                    h0_idx = np.where(h0_mask)[0]
                    nb_arrays = {k: all_arrays[k][h0_idx] for k in all_arrays}

                    nb_data, nb_arch = run_track(
                        'newbuild', iso, nb_arrays, demand_twh, combos,
                        existing_override=greenfield_all,
                        track_output=track_output, completed_steps=completed_steps)
                    track_output['results'][iso]['newbuild'] = nb_data

                    if nb_arch:
                        nb_dg = run_track_demand_growth(
                            'newbuild', iso, nb_arrays, nb_arch, combos,
                            existing_override=greenfield_all)
                        track_output['demand_growth']['newbuild'][iso] = nb_dg
                else:
                    print(f"  {iso:>6}   newbuild: no hydro=0 mixes")

            completed_steps.add(step_key_nb)
            save_checkpoint(track_output, [list(s) for s in completed_steps])

            # Also save intermediate results to disk
            _save_results(track_output)
        else:
            print(f"  {iso:>6}   newbuild: skipped (checkpoint)")

        # Track 2: replace (hydro at existing floor, everything else zeroed, uprates OFF)
        # Filter: hydro >= 2050 high-demand floor (hydro can't go below this)
        step_key_rp = (iso, 'replace')
        if step_key_rp not in completed_steps:
            rp_arrays = load_iso_arrays(ef_expanded, iso)
            if rp_arrays is not None:
                # Compute 2050 high-demand hydro floor: existing share / max demand growth
                # Hydro is fixed capacity — as demand grows, its share shrinks.
                # At 2050 high growth, hydro share = existing / (1+rate)^25.
                # Mixes below this floor can never be optimal in replace track.
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
                    completed_steps.add(step_key_rp)
                    save_checkpoint(track_output, [list(s) for s in completed_steps])
                    continue

                rp_data, rp_arch = run_track(
                    'replace', iso, rp_arrays, demand_twh, combos,
                    uprate_cap_override=0, existing_override=greenfield_keep_hydro,
                    track_output=track_output, completed_steps=completed_steps)
                track_output['results'][iso]['replace'] = rp_data

                if rp_arch:
                    rp_dg = run_track_demand_growth(
                        'replace', iso, rp_arrays, rp_arch, combos,
                        uprate_cap_override=0, existing_override=greenfield_keep_hydro)
                    track_output['demand_growth']['replace'][iso] = rp_dg

            completed_steps.add(step_key_rp)
            save_checkpoint(track_output, [list(s) for s in completed_steps])

            # Save intermediate results
            _save_results(track_output)
        else:
            print(f"  {iso:>6}    replace: skipped (checkpoint)")

    # Final save
    _save_results(track_output)

    # Save parquet (compact, git-friendly)
    _save_parquet(track_output)

    # Clean up checkpoint on successful completion
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("  Checkpoint removed (run complete)")

    total_elapsed = time.time() - total_start
    out_path = str(TRACK_RESULTS_PATH)
    tr_size = os.path.getsize(out_path) / (1024 * 1024)
    pq_path = os.path.join(SCRIPT_DIR, 'dashboard', 'track_scenarios.parquet')
    pq_size = os.path.getsize(pq_path) / (1024 * 1024) if os.path.exists(pq_path) else 0
    print(f"\n{'='*70}")
    print(f"  TRACK ANALYSIS COMPLETE in {total_elapsed:.0f}s")
    print(f"  JSON:    {out_path} ({tr_size:.1f} MB)")
    print(f"  Parquet: {pq_path} ({pq_size:.1f} MB)")
    print(f"{'='*70}")

def _save_results(track_output):
    """Write track_output to disk."""
    out_path = str(TRACK_RESULTS_PATH)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(track_output, f, separators=(',', ':'))
    os.replace(tmp, out_path)


def _save_parquet(track_output):
    """Flatten track results into a compact parquet file for git storage."""
    import pandas as pd

    rows = []
    for iso, iso_data in track_output.get('results', {}).items():
        for track_name, thr_dict in iso_data.items():
            for thr_str, thr_val in thr_dict.items():
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

    if not rows:
        print("  No scenario data to write to parquet")
        return

    df = pd.DataFrame(rows)
    pq_path = os.path.join(SCRIPT_DIR, 'dashboard', 'track_scenarios.parquet')
    df.to_parquet(pq_path, index=False, compression='zstd')
    print(f"  track_scenarios.parquet: {len(df):,} rows, "
          f"{os.path.getsize(pq_path) / (1024*1024):.1f} MB")

    # Also save demand growth parquet
    dg_rows = []
    for track_name, iso_dict in track_output.get('demand_growth', {}).items():
        for iso, thr_dict in iso_dict.items():
            for thr_str, sc_dict in thr_dict.items():
                for sc_key, year_data in sc_dict.items():
                    for year_str, growth_data in year_data.items():
                        for g_level, vals in growth_data.items():
                            dg_rows.append({
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

    if dg_rows:
        df_dg = pd.DataFrame(dg_rows)
        dg_path = os.path.join(SCRIPT_DIR, 'dashboard', 'track_demand_growth.parquet')
        df_dg.to_parquet(dg_path, index=False, compression='zstd')
        print(f"  track_demand_growth.parquet: {len(df_dg):,} rows, "
              f"{os.path.getsize(dg_path) / (1024*1024):.1f} MB")


if __name__ == '__main__':
    main()
