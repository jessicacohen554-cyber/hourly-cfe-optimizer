#!/usr/bin/env python3
"""
Targeted 100% Hourly Match Search
===================================
Searches for feasible 100% hourly matching solutions by:
  1. Loading existing 99% archetypes from cache (46-70 per ISO)
  2. Evaluating each at procurement levels 100-400% (coarse 25% grid)
  3. Varying storage allocations (battery 0-20%, LDES 0-20%)
  4. Interpolating between grid points for smooth cost curves
  5. Reporting best achievable match and associated cost

This is a targeted search, NOT a full optimizer rerun. It tests
known-good mixes at higher procurement levels to find the 100% frontier.

Usage: python search_100pct.py [--iso CAISO,ERCOT,...]
"""

import json
import os
import sys
import time
import numpy as np

# Import data loading and scoring from optimizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optimize_overprocure import (
    load_data, fast_hourly_score, fast_score_with_battery,
    fast_score_with_both_storage,
    ISOS, H, RESOURCE_TYPES, HYDRO_CAPS,
    BATTERY_EFFICIENCY, BATTERY_DURATION_HOURS,
    LDES_EFFICIENCY, LDES_DURATION_HOURS, LDES_WINDOW_DAYS,
    NUCLEAR_SHARE_OF_CLEAN_FIRM, NUCLEAR_MONTHLY_CF,
    compute_costs_parameterized, FULL_LCOE_TABLES,
    FULL_TRANSMISSION_TABLES, WHOLESALE_PRICES,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(SCRIPT_DIR, 'data', 'optimizer_cache.json')
RESULTS_PATH = os.path.join(SCRIPT_DIR, 'dashboard', 'overprocure_results.json')

# Coarse procurement grid: 100% to 400% in 25% steps
PROCUREMENT_GRID = list(range(100, 425, 25))  # [100, 125, 150, ..., 400]

# Storage grid: test battery and LDES at these percentages
BATTERY_GRID = [0, 2, 4, 6, 8, 10, 14, 18]
LDES_GRID = [0, 2, 4, 6, 8, 10, 14, 18]


def build_supply_matrix(iso, demand_data, gen_profiles):
    """Build the 5×8760 supply matrix used by fast scoring functions."""
    demand_norm = np.array(demand_data[iso]['normalized'][:H], dtype=np.float64)

    # Build supply profiles matching optimizer logic
    supply_profiles = {}
    iso_gen = gen_profiles[iso]

    # Solar
    solar_key = 'solar' if 'solar' in iso_gen else 'solar_proxy'
    solar_raw = np.array(iso_gen[solar_key][:H], dtype=np.float64)
    if solar_raw.sum() > 0:
        solar_raw /= solar_raw.sum()
    supply_profiles['solar'] = solar_raw

    # Wind
    wind_raw = np.array(iso_gen['wind'][:H], dtype=np.float64)
    if wind_raw.sum() > 0:
        wind_raw /= wind_raw.sum()
    supply_profiles['wind'] = wind_raw

    # Clean firm (nuclear with seasonal derate)
    nuclear_share = NUCLEAR_SHARE_OF_CLEAN_FIRM.get(iso, 1.0)
    geo_share = 1.0 - nuclear_share
    cf_profile = np.zeros(H, dtype=np.float64)
    monthly_cf = NUCLEAR_MONTHLY_CF.get(iso, {})

    for h in range(H):
        day = h // 24
        month = 1
        days_in_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        acc = 0
        for m, d in enumerate(days_in_months, 1):
            acc += d
            if day < acc:
                month = m
                break

        nuc_cf = monthly_cf.get(month, 1.0)
        cf_profile[h] = nuclear_share * nuc_cf + geo_share

    cf_profile /= cf_profile.sum()
    supply_profiles['clean_firm'] = cf_profile

    # CCS-CCGT: flat baseload
    ccs_profile = np.full(H, 1.0 / H, dtype=np.float64)
    supply_profiles['ccs_ccgt'] = ccs_profile

    # Hydro
    if 'hydro' in iso_gen:
        hydro_raw = np.array(iso_gen['hydro'][:H], dtype=np.float64)
        if hydro_raw.sum() > 0:
            hydro_raw /= hydro_raw.sum()
    else:
        hydro_raw = np.full(H, 1.0 / H, dtype=np.float64)
    supply_profiles['hydro'] = hydro_raw

    # Build 5×H matrix
    supply_matrix = np.zeros((len(RESOURCE_TYPES), H), dtype=np.float64)
    for i, rtype in enumerate(RESOURCE_TYPES):
        supply_matrix[i] = supply_profiles[rtype]

    return demand_norm, supply_matrix, supply_profiles


def load_archetypes_from_cache(iso):
    """Load unique archetypes from 99% threshold (and 97.5% for diversity)."""
    with open(CACHE_PATH) as f:
        data = json.load(f)

    archetypes = []
    seen = set()

    for threshold_key in ['99', '97.5', '95']:
        scenarios = (data['results'].get(iso, {})
                     .get('thresholds', {})
                     .get(threshold_key, {})
                     .get('scenarios', {}))

        for sk, s in scenarios.items():
            mix = s.get('resource_mix', {})
            bat = s.get('battery_dispatch_pct', 0)
            ldes = s.get('ldes_dispatch_pct', 0)

            # Deduplicate by mix (ignore procurement/storage — we'll vary those)
            mk = tuple(mix.get(rt, 0) for rt in RESOURCE_TYPES)
            if mk in seen:
                continue
            seen.add(mk)

            archetypes.append({
                'resource_mix': mix,
                'original_battery': bat,
                'original_ldes': ldes,
                'original_procurement': s.get('procurement_pct', 0),
                'original_match': s.get('hourly_match_score', 0),
                'source_threshold': threshold_key,
            })

    return archetypes


def search_100pct(iso, demand_arr, supply_matrix, archetypes):
    """
    2D grid search: procurement × storage for each archetype.
    Returns best results sorted by match score.
    """
    hydro_cap = HYDRO_CAPS.get(iso, 0)
    results = []
    best_match = 0
    best_result = None
    evaluations = 0

    for arch_idx, arch in enumerate(archetypes):
        mix = arch['resource_mix']
        mix_fracs = np.array([mix.get(rt, 0) / 100.0 for rt in RESOURCE_TYPES],
                             dtype=np.float64)

        # Skip if hydro exceeds cap
        if mix.get('hydro', 0) > hydro_cap + 0.5:
            continue

        for proc_pct in PROCUREMENT_GRID:
            pf = proc_pct / 100.0

            # Quick pre-screen: score without storage
            base_score = fast_hourly_score(demand_arr, supply_matrix, mix_fracs, pf)
            base_match = base_score * 100.0
            evaluations += 1

            # If base score is below 95%, no amount of reasonable storage will hit 100%
            if base_match < 90.0:
                continue

            for bat_pct in BATTERY_GRID:
                if bat_pct > 0:
                    bat_score = fast_score_with_battery(
                        demand_arr, supply_matrix, mix_fracs, pf, bat_pct)
                    bat_match = bat_score * 100.0
                    evaluations += 1
                else:
                    bat_match = base_match

                # If battery alone gets us close, try LDES combinations
                if bat_match < 92.0 and bat_pct > 0:
                    continue

                for ldes_pct in LDES_GRID:
                    if ldes_pct == 0 and bat_pct == 0:
                        score = base_match
                    elif ldes_pct == 0:
                        score = bat_match
                    elif bat_pct == 0 and ldes_pct > 0:
                        s = fast_score_with_both_storage(
                            demand_arr, supply_matrix, mix_fracs, pf,
                            0, ldes_pct)
                        score = s * 100.0
                        evaluations += 1
                    else:
                        s = fast_score_with_both_storage(
                            demand_arr, supply_matrix, mix_fracs, pf,
                            bat_pct, ldes_pct)
                        score = s * 100.0
                        evaluations += 1

                    if score > best_match:
                        best_match = score
                        best_result = {
                            'resource_mix': mix,
                            'procurement_pct': proc_pct,
                            'battery_dispatch_pct': bat_pct,
                            'ldes_dispatch_pct': ldes_pct,
                            'hourly_match_score': round(score, 4),
                            'archetype_idx': arch_idx,
                            'source_threshold': arch['source_threshold'],
                        }

                    # If we hit 100%, record it
                    if score >= 99.995:
                        results.append({
                            'resource_mix': dict(mix),
                            'procurement_pct': proc_pct,
                            'battery_dispatch_pct': bat_pct,
                            'ldes_dispatch_pct': ldes_pct,
                            'hourly_match_score': round(score, 4),
                        })

    return results, best_result, best_match, evaluations


def compute_cost_for_result(iso, result):
    """Compute Medium-scenario cost for a 100% search result."""
    return compute_costs_parameterized(
        iso,
        result['resource_mix'],
        result['procurement_pct'],
        result['battery_dispatch_pct'],
        result['ldes_dispatch_pct'],
        result['hourly_match_score'],
        'Medium', 'Medium', 'Medium', 'Medium', 'Medium'
    )


def frontier_at_procurement(iso, demand_arr, supply_matrix, archetypes, procurement_pct):
    """Find best match score achievable at a specific procurement level."""
    hydro_cap = HYDRO_CAPS.get(iso, 0)
    best = 0
    best_config = None
    pf = procurement_pct / 100.0

    for arch in archetypes:
        mix = arch['resource_mix']
        if mix.get('hydro', 0) > hydro_cap + 0.5:
            continue
        mix_fracs = np.array([mix.get(rt, 0) / 100.0 for rt in RESOURCE_TYPES],
                             dtype=np.float64)

        for bat_pct in BATTERY_GRID:
            for ldes_pct in LDES_GRID:
                if bat_pct == 0 and ldes_pct == 0:
                    score = fast_hourly_score(demand_arr, supply_matrix, mix_fracs, pf) * 100
                elif ldes_pct == 0:
                    score = fast_score_with_battery(
                        demand_arr, supply_matrix, mix_fracs, pf, bat_pct) * 100
                else:
                    score = fast_score_with_both_storage(
                        demand_arr, supply_matrix, mix_fracs, pf,
                        bat_pct, ldes_pct) * 100

                if score > best:
                    best = score
                    best_config = {
                        'resource_mix': mix,
                        'procurement_pct': procurement_pct,
                        'battery_dispatch_pct': bat_pct,
                        'ldes_dispatch_pct': ldes_pct,
                        'hourly_match_score': round(score, 4),
                    }

    return best, best_config


def main():
    target_isos = None
    if '--iso' in sys.argv:
        idx = sys.argv.index('--iso')
        if idx + 1 < len(sys.argv):
            target_isos = [iso.strip() for iso in sys.argv[idx + 1].split(',')]

    print("=" * 80)
    print("  TARGETED 100% HOURLY MATCH SEARCH")
    print(f"  Procurement: {PROCUREMENT_GRID[0]}–{PROCUREMENT_GRID[-1]}% "
          f"(coarse {PROCUREMENT_GRID[1]-PROCUREMENT_GRID[0]}% grid)")
    print(f"  Battery: {BATTERY_GRID}")
    print(f"  LDES: {LDES_GRID}")
    print("=" * 80)

    start = time.time()
    demand_data, gen_profiles, emission_rates, fossil_mix = load_data()

    run_isos = target_isos or ISOS
    all_results = {}

    for iso in run_isos:
        print(f"\n{'─'*60}")
        print(f"  {iso}")
        iso_start = time.time()

        # Load archetypes
        archetypes = load_archetypes_from_cache(iso)
        print(f"    Loaded {len(archetypes)} unique archetypes from cache")

        # Build supply matrix
        demand_arr, supply_matrix, supply_profiles = build_supply_matrix(
            iso, demand_data, gen_profiles)

        # Run full 2D search
        feasible, best_result, best_match, evals = search_100pct(
            iso, demand_arr, supply_matrix, archetypes)

        # Frontier: best match at each procurement level
        print(f"\n    Procurement Frontier (best match at each level):")
        print(f"    {'Proc':>5}  {'Match':>8}  {'Bat':>4}  {'LDES':>4}  {'Mix (CF/Sol/Wnd/CCS/Hyd)':>30}")
        frontier = []
        for proc in PROCUREMENT_GRID:
            score, config = frontier_at_procurement(
                iso, demand_arr, supply_matrix, archetypes, proc)
            if config:
                mix = config['resource_mix']
                mix_str = f"{mix.get('clean_firm',0)}/{mix.get('solar',0)}/{mix.get('wind',0)}/{mix.get('ccs_ccgt',0)}/{mix.get('hydro',0)}"
                print(f"    {proc:>4}%  {score:>7.3f}%  {config['battery_dispatch_pct']:>3}%  "
                      f"{config['ldes_dispatch_pct']:>3}%  {mix_str:>30}")
                cost = compute_cost_for_result(iso, config)
                frontier.append({**config, 'costs': cost})

        elapsed = time.time() - iso_start
        print(f"\n    Best overall: {best_match:.4f}% match")
        if best_result:
            mix = best_result['resource_mix']
            print(f"    Config: proc={best_result['procurement_pct']}% "
                  f"bat={best_result['battery_dispatch_pct']}% "
                  f"ldes={best_result['ldes_dispatch_pct']}%")
            print(f"    Mix: CF={mix.get('clean_firm',0)} Sol={mix.get('solar',0)} "
                  f"Wnd={mix.get('wind',0)} CCS={mix.get('ccs_ccgt',0)} "
                  f"Hyd={mix.get('hydro',0)}")
        print(f"    Feasible 100% solutions found: {len(feasible)}")
        print(f"    Evaluations: {evals:,} in {elapsed:.1f}s")

        all_results[iso] = {
            'best_match': round(best_match, 4),
            'best_result': best_result,
            'feasible_100pct': feasible[:10],  # Top 10
            'frontier': frontier,
        }

    # Save results
    output = {
        'search_type': '100pct_targeted',
        'procurement_grid': PROCUREMENT_GRID,
        'battery_grid': BATTERY_GRID,
        'ldes_grid': LDES_GRID,
        'results': all_results,
        'runtime_seconds': round(time.time() - start, 1),
    }
    out_path = os.path.join(SCRIPT_DIR, 'data', 'search_100pct_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*80}")
    print(f"  COMPLETE in {time.time()-start:.0f}s")
    print(f"  Results: {out_path}")

    # Summary
    print(f"\n  Summary:")
    for iso in run_isos:
        r = all_results.get(iso, {})
        bm = r.get('best_match', 0)
        n = len(r.get('feasible_100pct', []))
        status = "100% ACHIEVED" if n > 0 else f"ceiling at {bm:.2f}%"
        print(f"    {iso}: {status}")
    print("=" * 80)


if __name__ == '__main__':
    main()
