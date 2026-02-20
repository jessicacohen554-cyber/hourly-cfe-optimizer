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


def run_track(track_name, iso, arrays, demand_twh, combos, uprate_cap_override=None):
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
        iso, arrays, demand_twh, uprate_cap_override=uprate_cap_override)

    active_thresholds = sorted(thr_indices.keys())
    thresholds_desc = np.array(sorted(active_thresholds, reverse=True), dtype=np.float64)
    thr_pos = {float(thresholds_desc[k]): k for k in range(len(thresholds_desc))}

    thr_data = {thr: {'scenarios': {}} for thr in active_thresholds}
    arch_set = set()

    n_combos = len(combos)
    iso_start = time.time()

    for combo_i, (scenario_key, sens) in enumerate(combos):
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

        if n_combos > 100 and (combo_i + 1) % 1000 == 0:
            elapsed = time.time() - iso_start
            rate = (combo_i + 1) / elapsed
            remaining = (n_combos - combo_i - 1) / rate
            print(f"    {iso} {track_name} {combo_i+1}/{n_combos} "
                  f"({elapsed:.0f}s, ~{remaining:.0f}s remaining)")

    result = {str(thr): thr_data[thr] for thr in active_thresholds}
    elapsed = time.time() - iso_start
    print(f"  {iso:>6} {track_name:>10}: {N:,} mixes, "
          f"{len(active_thresholds)} thresholds, {len(arch_set)} archetypes — {elapsed:.0f}s")
    return result, arch_set


def run_track_demand_growth(track_name, iso, arrays, arch_set, combos,
                             uprate_cap_override=None):
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
                    uprate_cap_override=uprate_cap_override
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
    args = parser.parse_args()

    print("=" * 70)
    print("  INCREMENTAL TRACK ANALYSIS")
    print(f"  Mode: {'Full sweep (all combos)' if args.full else 'Medium-only (fast)'}")
    print("=" * 70)
    total_start = time.time()

    # Load EF sources
    print("\nLoading EF sources...")
    ef_expanded = pq.read_table(EF_EXPANDED_PATH)
    print(f"  Expanded EF: {ef_expanded.num_rows:,} rows")

    if os.path.exists(EF_BACKUP_PATH):
        ef_backup = pq.read_table(EF_BACKUP_PATH)
        print(f"  Backup EF (original): {ef_backup.num_rows:,} rows")
    else:
        print(f"  WARNING: No backup EF found at {EF_BACKUP_PATH}")
        print(f"  Using expanded EF for Track 2 (will be slower)")
        ef_backup = ef_expanded

    track_output = {
        'meta': {
            'tracks': {
                'newbuild': {
                    'description': 'New-build requirement for hourly matching',
                    'hydro': 'excluded (hydro=0 mixes only)',
                    'uprates': 'on (uprate tranche active)',
                    'purpose': 'What resources does hourly matching incentivize vs what the grid needs?',
                },
                'replace': {
                    'description': 'Cost to replace existing clean generation',
                    'hydro': 'included (up to existing floor, wholesale-priced)',
                    'uprates': 'off (uprate_cap=0, all new CF at new-build prices)',
                    'purpose': 'True cost of replacing existing nuclear/firm with new-build',
                },
            },
            'mode': 'full' if args.full else 'medium_only',
            'thresholds': OUTPUT_THRESHOLDS,
            'resource_types': RESOURCE_TYPES,
        },
        'results': {},
        'demand_growth': {'newbuild': {}, 'replace': {}},
    }

    for iso in ISOS:
        demand_twh = REGIONAL_DEMAND_TWH[iso]
        track_output['results'][iso] = {}

        # Build combos
        if args.full:
            combos = build_sensitivity_combos(iso)
        else:
            # Medium-only: single combo
            mk = medium_key(iso)
            geo = 'M' if iso == 'CAISO' else None
            sens = {'ren': 'M', 'firm': 'M', 'batt': 'M', 'ldes_lvl': 'M',
                    'ccs': 'M', 'q45': '1', 'fuel': 'M', 'tx': 'M', 'geo': geo}
            combos = [(mk, sens)]

        # Track 1: newbuild (hydro=0 from expanded EF)
        all_arrays = load_iso_arrays(ef_expanded, iso)
        if all_arrays is not None:
            h0_mask = all_arrays['hydro'] == 0
            n_h0 = h0_mask.sum()
            if n_h0 > 0:
                h0_idx = np.where(h0_mask)[0]
                nb_arrays = {k: all_arrays[k][h0_idx] for k in all_arrays}

                nb_data, nb_arch = run_track(
                    'newbuild', iso, nb_arrays, demand_twh, combos)
                track_output['results'][iso]['newbuild'] = nb_data

                # Demand growth for newbuild archetypes
                if nb_arch:
                    nb_dg = run_track_demand_growth(
                        'newbuild', iso, nb_arrays, nb_arch, combos)
                    track_output['demand_growth']['newbuild'][iso] = nb_dg
            else:
                print(f"  {iso:>6}   newbuild: no hydro=0 mixes")

        # Track 2: replace (original EF, uprate_cap=0)
        rp_arrays = load_iso_arrays(ef_backup, iso)
        if rp_arrays is not None:
            rp_data, rp_arch = run_track(
                'replace', iso, rp_arrays, demand_twh, combos,
                uprate_cap_override=0)
            track_output['results'][iso]['replace'] = rp_data

            # Demand growth for replace archetypes
            if rp_arch:
                rp_dg = run_track_demand_growth(
                    'replace', iso, rp_arrays, rp_arch, combos,
                    uprate_cap_override=0)
                track_output['demand_growth']['replace'][iso] = rp_dg

    # Save
    out_path = str(TRACK_RESULTS_PATH)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(track_output, f, separators=(',', ':'))
    tr_size = os.path.getsize(out_path) / (1024 * 1024)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  TRACK ANALYSIS COMPLETE in {total_elapsed:.0f}s")
    print(f"  Output: {out_path} ({tr_size:.1f} MB)")
    print(f"{'='*70}")

    # Summary: Medium scenario comparison
    print("\nAll-Medium (45Q=ON) comparison — Baseline vs Tracks:")
    # Load baseline for comparison
    baseline_path = os.path.join(SCRIPT_DIR, 'dashboard', 'overprocure_results.json')
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)
    else:
        baseline = None

    for iso in ISOS:
        mk = medium_key(iso)
        print(f"\n  {iso} ({mk}):")
        print(f"  {'Thr':>6} | {'Track':>10} | {'CF':>3} {'Sol':>3} {'Wnd':>3} "
              f"{'CCS':>3} {'Hyd':>3} | {'Proc':>4} {'Eff$/MWh':>9} {'Match':>5}")
        print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*3}-{'-'*3}-{'-'*3}-{'-'*3}-{'-'*3}-+-"
              f"{'-'*4}-{'-'*9}-{'-'*5}")

        for thr in OUTPUT_THRESHOLDS:
            t_str = str(thr)

            # Baseline
            if baseline:
                sc = (baseline.get('results', {}).get(iso, {})
                      .get('thresholds', {}).get(t_str, {})
                      .get('scenarios', {}).get(mk))
                if sc:
                    rm = sc['resource_mix']
                    print(f"  {thr:>5}% | {'baseline':>10} | "
                          f"{rm.get('clean_firm',0):>3} {rm.get('solar',0):>3} "
                          f"{rm.get('wind',0):>3} {rm.get('ccs_ccgt',0):>3} "
                          f"{rm.get('hydro',0):>3} | "
                          f"{sc['procurement_pct']:>4} ${sc['costs']['effective_cost']:>7.1f} "
                          f"{sc['hourly_match_score']:>5.1f}")

            # Track results
            for track_name in ['newbuild', 'replace']:
                sc = (track_output.get('results', {}).get(iso, {})
                      .get(track_name, {}).get(t_str, {})
                      .get('scenarios', {}).get(mk))
                if sc:
                    rm = sc['resource_mix']
                    print(f"  {'':>6} | {track_name:>10} | "
                          f"{rm.get('clean_firm',0):>3} {rm.get('solar',0):>3} "
                          f"{rm.get('wind',0):>3} {rm.get('ccs_ccgt',0):>3} "
                          f"{rm.get('hydro',0):>3} | "
                          f"{sc['procurement_pct']:>4} ${sc['costs']['effective_cost']:>7.1f} "
                          f"{sc['hourly_match_score']:>5.1f}")


if __name__ == '__main__':
    main()
