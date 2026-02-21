#!/usr/bin/env python3
"""
Post-Processor: Recompute CO₂ Abatement with Dispatch-Stack Retirement Model
================================================================================
Uses merit-order fuel retirement to compute emission rates at each clean energy
threshold. As clean energy grows, coal retires first (dirtiest), then oil, then
gas. Above 70% clean, all coal and oil have retired — only gas CCGT remains.

This replaces the previous uniform hourly fossil mix model where coal/gas/oil
shares were constant regardless of clean energy percentage.

For each result:
  1. Determine which fuels have retired at the scenario's clean % (merit order)
  2. Compute the emission rate of the remaining fossil fleet
  3. CO₂_abated = Σ_h fossil_displaced[h] × emission_rate_at_threshold

Reads:  dashboard/overprocure_results.json
        data/egrid_emission_rates.json
        data/eia_fossil_mix.json
        data/eia_generation_profiles.json
        data/eia_demand_profiles.json
Writes: dashboard/overprocure_results.json (updated CO₂ fields)
        data/optimizer_cache.json (updated if exists)

Refactored to import shared dispatch logic from dispatch_utils.py.
"""

import json
import os
import sys
import numpy as np
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from dispatch_utils import (
    H, ISOS, RESOURCE_TYPES, CCS_RESIDUAL_EMISSION_RATE,
    GRID_MIX_SHARES, load_common_data,
    get_supply_profiles_simple as get_supply_profiles,
    compute_fossil_retirement as compute_dispatch_stack_emission_rate,
    reconstruct_hourly_dispatch,
)

DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
DATA_YEAR = '2025'
RESULTS_PATH = os.path.join(SCRIPT_DIR, 'dashboard', 'overprocure_results.json')
CACHE_PATH = os.path.join(DATA_DIR, 'optimizer_cache.json')


def load_data():
    """Load all required data files."""
    return load_common_data()


def get_emission_rate_for_threshold(iso, threshold_pct, emission_rates, fossil_mix):
    """Get the single emission rate (tCO₂/MWh) for a given clean energy threshold."""
    return compute_dispatch_stack_emission_rate(iso, threshold_pct, emission_rates, fossil_mix)


def compute_hourly_fossil_displacement(demand_norm, supply_profiles, resource_pcts,
                                        procurement_pct, battery_dispatch_pct, ldes_dispatch_pct):
    """Reconstruct hourly clean supply and compute fossil displacement at each hour.

    Thin wrapper around dispatch_utils.reconstruct_hourly_dispatch that preserves
    the original return signature (fossil_displaced, ccs_supply, curtailed).
    """
    # Original recompute_co2 didn't handle battery8 — pass 0 for backward compat
    result = reconstruct_hourly_dispatch(
        demand_norm, supply_profiles, resource_pcts,
        procurement_pct, battery_dispatch_pct,
        0,  # battery8 not used in original CO2 model
        ldes_dispatch_pct)

    return result['fossil_displaced'], result['ccs_supply'], result['curtailed']


def compute_co2_hourly(fossil_displaced, ccs_supply, emission_rate, demand_total_mwh,
                       retirement_info=None):
    """Compute CO₂ abated using dispatch-stack emission rate."""
    scale = demand_total_mwh

    non_ccs_displaced = np.maximum(0.0, fossil_displaced - np.minimum(fossil_displaced, ccs_supply))
    ccs_displaced = np.minimum(fossil_displaced, ccs_supply)

    co2_clean = np.sum(non_ccs_displaced) * emission_rate * scale
    ccs_credit = max(0.0, emission_rate - CCS_RESIDUAL_EMISSION_RATE)
    co2_ccs = np.sum(ccs_displaced) * ccs_credit * scale

    total_abated = co2_clean + co2_ccs
    matched_mwh = np.sum(fossil_displaced) * scale
    co2_rate = total_abated / matched_mwh if matched_mwh > 0 else 0

    result = {
        'total_co2_abated_tons': round(total_abated, 0),
        'co2_rate_per_mwh': round(co2_rate, 4),
        'matched_mwh': round(matched_mwh, 0),
        'emission_rate_tco2_mwh': round(emission_rate, 4),
        'methodology': 'dispatch_stack_retirement',
    }

    if retirement_info:
        result['retirement_info'] = retirement_info

    return result


def recompute_all_co2(results_data, demand_data, gen_profiles, emission_rates, fossil_mix):
    """Recompute CO₂ for all results using dispatch-stack retirement model."""
    start = time.time()

    for iso in ISOS:
        if iso not in results_data.get('results', {}):
            continue

        iso_data = results_data['results'][iso]
        iso_demand = demand_data.get(iso, {})
        year_demand = iso_demand.get(DATA_YEAR, iso_demand.get('2024', {}))
        if isinstance(year_demand, dict):
            demand_norm = year_demand.get('normalized', [0.0] * H)[:H]
            demand_total_mwh_fallback = year_demand.get('total_annual_mwh', 0)
        else:
            demand_norm = year_demand[:H] if isinstance(year_demand, list) else [0.0] * H
            demand_total_mwh_fallback = 0
        supply_profiles = get_supply_profiles(iso, gen_profiles)
        demand_total_mwh = iso_data.get('annual_demand_mwh', demand_total_mwh_fallback)

        baseline_clean = sum(GRID_MIX_SHARES.get(iso, {}).values())
        print(f"\n  {iso} (baseline clean: {baseline_clean:.1f}%):")
        print(f"    Dispatch-stack emission rates (tCO₂/MWh):")
        for t_pct in [50, 60, 70, 80, 90, 95, 100]:
            rate, info = get_emission_rate_for_threshold(iso, t_pct, emission_rates, fossil_mix)
            coal_r = info.get('coal_remaining_pct', 0)
            gas_only = info.get('forced_gas_only', False)
            label = " [gas-only]" if gas_only else f" [coal remaining: {coal_r}%]" if coal_r > 0.1 else ""
            print(f"      {t_pct:>3}% clean → {rate:.4f} tCO₂/MWh{label}")

        # Recompute sweep results
        if 'sweep' in iso_data:
            for i, sweep_result in enumerate(iso_data['sweep']):
                resource_mix = sweep_result.get('resource_mix', {})
                proc = sweep_result.get('procurement_pct', 100)
                batt = sweep_result.get('battery_dispatch_pct', 0)
                ldes = sweep_result.get('ldes_dispatch_pct', 0)
                match_score = sweep_result.get('hourly_match_score', 0)

                rate, info = get_emission_rate_for_threshold(
                    iso, match_score, emission_rates, fossil_mix)

                fossil_displaced, ccs_supply, curtailed = compute_hourly_fossil_displacement(
                    demand_norm, supply_profiles, resource_mix, proc, batt, ldes)

                co2 = compute_co2_hourly(
                    fossil_displaced, ccs_supply, rate, demand_total_mwh, info)
                sweep_result['co2_abated'] = co2
                sweep_result['curtailed_mwh'] = round(float(np.sum(curtailed)) * demand_total_mwh, 0)

        # Recompute threshold results
        thresholds_data = iso_data.get('thresholds', {})
        recomputed = 0

        for t_str, t_data in thresholds_data.items():
            threshold_pct = float(t_str)
            scenarios = t_data.get('scenarios', {})

            rate, info = get_emission_rate_for_threshold(
                iso, threshold_pct, emission_rates, fossil_mix)

            med_key = 'MMMM_M_M_M1_M' if iso == 'CAISO' else 'MMMM_M_M_M1_X'
            actual_med_key = None
            for mk_candidate in [med_key,
                                 'MMM_M_M_M1_M', 'MMM_M_M_M1_X',
                                 'MMM_M_M']:
                if mk_candidate in scenarios:
                    actual_med_key = mk_candidate
                    break
            if actual_med_key and actual_med_key in scenarios:
                result = scenarios[actual_med_key]
                resource_mix = result.get('resource_mix', {})
                proc = result.get('procurement_pct', 100)
                batt = result.get('battery_dispatch_pct', 0)
                ldes = result.get('ldes_dispatch_pct', 0)

                fossil_displaced, ccs_supply, curtailed = compute_hourly_fossil_displacement(
                    demand_norm, supply_profiles, resource_mix, proc, batt, ldes)

                co2 = compute_co2_hourly(
                    fossil_displaced, ccs_supply, rate, demand_total_mwh, info)
                result['co2_abated'] = co2
                result['curtailed_mwh'] = round(float(np.sum(curtailed)) * demand_total_mwh, 0)

                result['co2_abated_fuel_sensitivity'] = {
                    'Low': co2,
                    'Medium': co2,
                    'High': co2,
                }

                recomputed += 1

            for sk, result in scenarios.items():
                if sk == actual_med_key:
                    continue

                resource_mix = result.get('resource_mix', {})
                proc = result.get('procurement_pct', 100)
                batt = result.get('battery_dispatch_pct', 0)
                ldes = result.get('ldes_dispatch_pct', 0)

                fossil_displaced, ccs_supply, curtailed = compute_hourly_fossil_displacement(
                    demand_norm, supply_profiles, resource_mix, proc, batt, ldes)

                co2 = compute_co2_hourly(
                    fossil_displaced, ccs_supply, rate, demand_total_mwh, info)
                result['co2_abated'] = co2
                recomputed += 1

        print(f"    Recomputed: {recomputed} threshold scenarios + {len(iso_data.get('sweep', []))} sweep points")

    elapsed = time.time() - start
    print(f"\n  Total recompute time: {elapsed:.1f}s")

    return results_data


def main():
    print("=" * 70)
    print("  CO₂ RECOMPUTATION — Hourly Fossil-Fuel Emission Rates")
    print("=" * 70)

    print("\n  Loading data...")
    demand_data, gen_profiles, emission_rates, fossil_mix = load_data()

    if not os.path.exists(RESULTS_PATH):
        print(f"  ERROR: Results file not found: {RESULTS_PATH}")
        sys.exit(1)

    with open(RESULTS_PATH) as f:
        results_data = json.load(f)
    print(f"  Loaded results: {RESULTS_PATH}")

    isos_present = [iso for iso in ISOS if iso in results_data.get('results', {})]
    print(f"  ISOs in results: {isos_present}")

    results_data = recompute_all_co2(results_data, demand_data, gen_profiles, emission_rates, fossil_mix)

    with open(RESULTS_PATH, 'w') as f:
        json.dump(results_data, f)
    print(f"\n  Updated: {RESULTS_PATH} ({os.path.getsize(RESULTS_PATH) / 1024:.0f} KB)")

    print(f"  Cache ({CACHE_PATH}) is locked — skipped")

    print(f"\n{'='*70}")
    print("  CO₂ RECOMPUTATION COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
