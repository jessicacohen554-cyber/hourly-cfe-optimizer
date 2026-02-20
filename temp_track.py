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
    TRACK_RESULTS_PATH,
)

EF_EXPANDED_PATH = os.path.join(SCRIPT_DIR, 'data', 'pfs_post_ef.parquet')
EF_BACKUP_PATH = os.path.join(SCRIPT_DIR, 'data', 'pfs_post_ef_backup.parquet')
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, 'data', 'track_checkpoint.json')


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


def run_track(track_name, iso, arrays, demand_twh, combos, uprate_cap_override=None,
              existing_override=None, track_output=None, completed_steps=None):
    """Run cost optimization for a track (newbuild or replace).

    Checkpoints after every 500 combos (full sweep) by writing partial results
    to track_output and saving to disk.

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
    checkpoint_interval = 500 if n_combos > 100 else n_combos + 1
    skipped = 0

    for combo_i, (scenario_key, sens) in enumerate(combos):
        # Skip already-computed scenarios
        if scenario_key in existing_scenarios:
            skipped += 1
            continue

        prices, wholesale, nuclear_price, ccs_price = get_scenario_prices(iso, sens)
        best_idxs, best_vals = eval_and_argmin_all(
            coeff_matrix, constant, prices, scores, thresholds_desc)

        for thr in active_thresholds:
            k = thr_pos[float(thr)]
            if best_vals[k] == np.inf:
                continue
            best_idx = int(best_idxs[k])
            tc_val = float(best_vals[k])
            arch_set.add(best_idx)

            scenario = build_winner_scenario(
                arrays, extras, best_idx, sens, iso, demand_twh,
                tc_val, wholesale, nuclear_price, ccs_price)
            thr_data[thr]['scenarios'][scenario_key] = scenario

        computed = combo_i + 1 - skipped
        if n_combos > 100 and computed > 0 and computed % 1000 == 0:
            elapsed = time.time() - iso_start
            total_remaining = n_combos - combo_i - 1
            rate = computed / elapsed
            remaining = total_remaining / rate if rate > 0 else 0
            print(f"    {iso} {track_name} {combo_i+1}/{n_combos} "
                  f"({elapsed:.0f}s, ~{remaining:.0f}s remaining)")

        # Checkpoint every N combos
        if (track_output is not None and completed_steps is not None
                and computed > 0 and computed % checkpoint_interval == 0):
            result_partial = {str(thr): thr_data[thr] for thr in active_thresholds}
            track_output['results'][iso][track_name] = result_partial
            _save_results(track_output)
            save_checkpoint(track_output, [list(s) for s in completed_steps])

    if skipped > 0:
        print(f"    {iso} {track_name}: skipped {skipped} cached scenarios")

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

    # Load EF sources
    print("\nLoading EF sources...")
    ef_expanded = pq.read_table(EF_EXPANDED_PATH)
    print(f"  Expanded EF: {ef_expanded.num_rows:,} rows")

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
        step_key_rp = (iso, 'replace')
        if step_key_rp not in completed_steps:
            rp_arrays = load_iso_arrays(ef_expanded, iso)
            if rp_arrays is not None:
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

    # Clean up checkpoint on successful completion
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("  Checkpoint removed (run complete)")

    total_elapsed = time.time() - total_start
    out_path = str(TRACK_RESULTS_PATH)
    tr_size = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\n{'='*70}")
    print(f"  TRACK ANALYSIS COMPLETE in {total_elapsed:.0f}s")
    print(f"  Output: {out_path} ({tr_size:.1f} MB)")
    print(f"{'='*70}")

def _save_results(track_output):
    """Write track_output to disk."""
    out_path = str(TRACK_RESULTS_PATH)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(track_output, f, separators=(',', ':'))
    os.replace(tmp, out_path)


if __name__ == '__main__':
    main()
