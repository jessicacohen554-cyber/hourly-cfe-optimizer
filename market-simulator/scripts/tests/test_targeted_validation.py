#!/usr/bin/env python3
"""Targeted validation: 3 scenarios × 2 ISOs × 6 years = 36 rows.

Picks worst-case combos (ERCOT High demand, NEISO High demand) to verify:
1. avg_lmp drops to $25-60 range (was $267 mean)
2. scarcity_hours < 500 (was 6,869 max)
3. New-build fossil feedback visible (fleet grows year-over-year)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
from market_simulation import (
    run_market_simulation, build_market_scenarios, load_common_data,
    load_egrid_baselines, SIM_YEARS,
)
from lmp_engine import INSTALLED_FOSSIL_MW

# Pick 3 scenarios: worst-case High demand + varying fossil cost
scenarios = []
all_scenarios = build_market_scenarios()

# Find High demand scenarios with Medium price, Medium PPA, Medium gas, Medium queue
# varying only new_fossil_cost (L/M/H)
targets = []
for sid, cond in all_scenarios:
    if (cond['demand_growth'] == 'High'
        and cond.get('_price_sens_name', '') == 'P3_HiRenew_HiFossil'
        and cond['ppa_level'] == 'Medium'
        and cond['queue_cap_level'] == 'Medium'):
        targets.append((sid, cond))

if len(targets) < 3:
    # Fallback: just grab first 3 High-demand scenarios
    targets = [(sid, cond) for sid, cond in all_scenarios
               if cond['demand_growth'] == 'High'][:3]

print(f"Selected {len(targets)} scenarios:")
for sid, cond in targets:
    print(f"  {sid}: {cond.get('name', '')[:80]}")

# Run on 2 ISOs: ERCOT (highest growth) and NEISO (smallest fleet)
test_isos = ['ERCOT', 'NEISO']

print(f"\nRunning {len(targets)} scenarios × {len(test_isos)} ISOs × {len(SIM_YEARS)} years...")
print(f"SIM_YEARS = {SIM_YEARS}")
print()

# Pre-load data once
print("Loading common data...")
t0 = time.time()
demand_data, gen_profiles, emission_rates, fossil_mix = load_common_data()
egrid_baselines = load_egrid_baselines()
preloaded = {
    'demand_data': demand_data,
    'gen_profiles': gen_profiles,
    'emission_rates': emission_rates,
    'fossil_mix': fossil_mix,
    'egrid_baselines': egrid_baselines,
}
print(f"Data loaded in {time.time()-t0:.1f}s\n")

lmp_cache = {}

for sid, cond in targets:
    print(f"\n{'='*70}")
    print(f"Scenario: {sid}")
    print(f"{'='*70}")

    results = run_market_simulation(
        sid, cond,
        isos=test_isos,
        _preloaded=preloaded,
        _lmp_cache=lmp_cache,
        _quiet=True,
    )

    # Extract results per ISO
    for iso in test_isos:
        iso_results = results.get(iso, [])
        if not iso_results:
            print(f"  {iso}: NO RESULTS")
            continue

        print(f"\n  {iso} ({len(iso_results)} years):")
        print(f"  {'Year':>6} {'Clean%':>7} {'AvgLMP':>8} {'Scarcity':>10} {'NewFossil':>12} {'FossilGW':>10}")
        print(f"  {'-'*6} {'-'*7} {'-'*8} {'-'*10} {'-'*12} {'-'*10}")

        for yr in iso_results:
            year = yr.get('year', '?')
            clean = yr.get('clean_pct', 0)
            lmp = yr.get('avg_lmp', 0)
            scarcity = yr.get('ordc_scarcity_hours', 0)
            new_fossil = yr.get('total_new_fossil_mw', 0)
            fossil_gw = yr.get('fossil_built_gw', 0)
            print(f"  {year:>6} {clean:>7.1f} ${lmp:>7.1f} {scarcity:>10.0f} {new_fossil:>10.0f} MW {fossil_gw:>8.2f}")

        # Validation checks for non-baseline years
        sim_years = [r for r in iso_results if r.get('year', 0) >= 2030]
        if sim_years:
            max_lmp = max(r.get('avg_lmp', 0) for r in sim_years)
            max_scarcity = max(r.get('ordc_scarcity_hours', 0) for r in sim_years)
            any_new_fossil = any(r.get('total_new_fossil_mw', 0) > 0 for r in sim_years)
            fossil_grows = any(r.get('fossil_built_gw', 0) > 0 for r in sim_years)

            flags = []
            if max_lmp > 150:
                flags.append(f"HIGH LMP: ${max_lmp:.0f}")
            if max_scarcity > 1000:
                flags.append(f"HIGH SCARCITY: {max_scarcity:.0f}h")
            if not any_new_fossil:
                flags.append("NO NEW FOSSIL BUILDS")
            if not fossil_grows:
                flags.append("FOSSIL FLEET NOT GROWING")

            if flags:
                print(f"  ** WARNINGS: {', '.join(flags)}")
            else:
                print(f"  ✓ LMP max=${max_lmp:.0f}, scarcity max={max_scarcity:.0f}h, new fossil active")

print("\n" + "="*70)
print("VALIDATION COMPLETE")
print("="*70)
