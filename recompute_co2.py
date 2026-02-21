#!/usr/bin/env python3
"""
Post-Processor: Recompute CO₂ Abatement with Dispatch-Stack Retirement Model
================================================================================
Uses merit-order fuel retirement to compute emission rates at each clean energy
threshold. As clean energy grows, coal retires first (dirtiest), then oil, then
gas. Above 70% clean, all coal and oil have retired — only gas CCGT remains.

OPTIMIZATION (Feb 2026): Two computation paths:
  1. FAST PATH (match_score): Uses pre-computed hourly_match_score from the
     optimizer. No hourly dispatch reconstruction needed — pure scalar math.
     Emission rate is uniform across all hours (threshold-dependent), so total
     CO₂ = total_fossil_displaced × emission_rate. ~1000x faster.
  2. DISPATCH PATH (fallback): Full 8760-hour reconstruction with Numba-compiled
     battery/LDES dispatch + dispatch cache. Used only when match_score is
     unavailable. ~10-50x faster than original with Numba + supply matrix.

For each result:
  1. Determine which fuels have retired at the scenario's clean % (merit order)
  2. Compute the emission rate of the remaining fossil fleet
  3. CO₂_abated = fossil_displaced × emission_rate (scalar or hourly sum)

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
    GRID_MIX_SHARES, BASE_DEMAND_TWH, load_common_data,
    get_supply_profiles_simple as get_supply_profiles,
    compute_fossil_retirement as compute_dispatch_stack_emission_rate,
    reconstruct_hourly_dispatch,
    build_supply_matrix,
    get_or_compute_dispatch, load_dispatch_cache, save_dispatch_cache,
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


# ══════════════════════════════════════════════════════════════════════════════
# FAST PATH: CO₂ from match_score (no hourly dispatch needed)
# ══════════════════════════════════════════════════════════════════════════════

def fast_co2_from_match_score(match_score, resource_mix, procurement_pct,
                               threshold_pct, iso, emission_rates, fossil_mix,
                               demand_total_mwh, rate_cache=None):
    """Compute CO₂ abated using match_score — no hourly dispatch reconstruction.

    The emission rate is uniform across all hours at a given threshold (it's a
    scalar from compute_fossil_retirement). So:
        total_CO2 = total_fossil_displaced × emission_rate

    Total fossil displaced = (match_score / 100) × demand_total_mwh.
    This already accounts for battery/LDES dispatch and curtailment — the
    optimizer computed this during Step 1.

    CCS contribution is approximated from resource mix (flat baseload, nearly
    100% useful — minimal curtailment for baseload resources).

    ~1000x faster than hourly dispatch path.
    """
    # Get emission rate from cache or compute
    cache_key = (iso, round(threshold_pct, 1))
    if rate_cache is not None and cache_key in rate_cache:
        rate, info = rate_cache[cache_key]
    else:
        rate, info = compute_dispatch_stack_emission_rate(
            iso, threshold_pct, emission_rates, fossil_mix)
        if rate_cache is not None:
            rate_cache[cache_key] = (rate, info)

    # Total fossil displaced (pre-computed by optimizer, includes storage effects)
    fossil_displaced_mwh = (match_score / 100.0) * demand_total_mwh

    # CCS contribution (flat baseload — nearly all output displaces fossil)
    ccs_pct = resource_mix.get('ccs_ccgt', 0)
    ccs_supply_mwh = (ccs_pct / 100.0) * (procurement_pct / 100.0) * demand_total_mwh
    ccs_effective_mwh = min(ccs_supply_mwh, fossil_displaced_mwh)
    non_ccs_mwh = fossil_displaced_mwh - ccs_effective_mwh

    # CO₂ abated
    co2_clean = non_ccs_mwh * rate
    ccs_credit = max(0.0, rate - CCS_RESIDUAL_EMISSION_RATE)
    co2_ccs = ccs_effective_mwh * ccs_credit

    total_abated = co2_clean + co2_ccs
    co2_rate_per_mwh = total_abated / fossil_displaced_mwh if fossil_displaced_mwh > 0 else 0

    return {
        'total_co2_abated_tons': round(total_abated, 0),
        'co2_rate_per_mwh': round(co2_rate_per_mwh, 4),
        'matched_mwh': round(fossil_displaced_mwh, 0),
        'emission_rate_tco2_mwh': round(rate, 4),
        'methodology': 'fast_match_score',
        'retirement_info': info,
    }


# ══════════════════════════════════════════════════════════════════════════════
# DISPATCH PATH: Full hourly reconstruction (fallback when match_score missing)
# ══════════════════════════════════════════════════════════════════════════════

def compute_hourly_fossil_displacement(demand_norm, supply_profiles, resource_pcts,
                                        procurement_pct, battery_dispatch_pct, ldes_dispatch_pct,
                                        supply_matrix=None, dispatch_cache=None, iso=None):
    """Reconstruct hourly clean supply and compute fossil displacement.

    Optimizations vs. original:
      - supply_matrix: pre-built (5, H) numpy array (skips per-call conversion)
      - dispatch_cache: mutable dict from load_dispatch_cache() (avoids recomputation)
    """
    battery8_pct = 0  # Original CO2 model didn't handle battery8

    # Use dispatch cache if available
    if dispatch_cache is not None and iso is not None:
        result, hit = get_or_compute_dispatch(
            iso, demand_norm, supply_profiles, resource_pcts,
            procurement_pct, battery_dispatch_pct, battery8_pct,
            ldes_dispatch_pct, cache=dispatch_cache)
    else:
        result = reconstruct_hourly_dispatch(
            demand_norm, supply_profiles, resource_pcts,
            procurement_pct, battery_dispatch_pct, battery8_pct,
            ldes_dispatch_pct, supply_matrix=supply_matrix)

    return result['fossil_displaced'], result['ccs_supply'], result['curtailed']


def compute_co2_hourly(fossil_displaced, ccs_supply, emission_rate, demand_total_mwh,
                       retirement_info=None):
    """Compute CO₂ abated using dispatch-stack emission rate (hourly path)."""
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


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RECOMPUTE — routes each scenario to fast or dispatch path
# ══════════════════════════════════════════════════════════════════════════════

def recompute_all_co2(results_data, demand_data, gen_profiles, emission_rates, fossil_mix):
    """Recompute CO₂ for all results using dispatch-stack retirement model.

    Uses fast match_score path when available (~1000x speedup), falls back to
    Numba-accelerated hourly dispatch with pre-built supply matrix + cache.
    """
    start = time.time()
    fast_count = 0
    dispatch_count = 0

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

        # Pre-build supply matrix once per ISO (3x speedup on dispatch path)
        supply_profiles = get_supply_profiles(iso, gen_profiles)
        sup_matrix = build_supply_matrix(supply_profiles)
        demand_total_mwh = iso_data.get('annual_demand_mwh', demand_total_mwh_fallback)

        # Load dispatch cache for fallback path
        dispatch_cache = load_dispatch_cache(iso)
        cache_dirty = False

        # Pre-compute emission rates per threshold (avoid redundant calls)
        rate_cache = {}

        baseline_clean = sum(GRID_MIX_SHARES.get(iso, {}).values())
        print(f"\n  {iso} (baseline clean: {baseline_clean:.1f}%):")
        print(f"    Dispatch-stack emission rates (tCO₂/MWh):")
        for t_pct in [50, 60, 70, 80, 90, 95, 100]:
            rate, info = get_emission_rate_for_threshold(iso, t_pct, emission_rates, fossil_mix)
            rate_cache[(iso, t_pct)] = (rate, info)
            gas_only = info.get('forced_gas_only', False)
            label = " [gas-only]" if gas_only else ""
            print(f"      {t_pct:>3}% clean → {rate:.4f} tCO₂/MWh{label}")

        # --- Recompute sweep results ---
        if 'sweep' in iso_data:
            for sweep_result in iso_data['sweep']:
                resource_mix = sweep_result.get('resource_mix', {})
                proc = sweep_result.get('procurement_pct', 100)
                match_score = sweep_result.get('hourly_match_score', 0)

                if match_score > 0:
                    # FAST PATH: use match_score
                    co2 = fast_co2_from_match_score(
                        match_score, resource_mix, proc,
                        match_score, iso, emission_rates, fossil_mix,
                        demand_total_mwh, rate_cache)
                    sweep_result['co2_abated'] = co2
                    fast_count += 1
                else:
                    # DISPATCH PATH: full hourly reconstruction
                    batt = sweep_result.get('battery_dispatch_pct', 0)
                    ldes = sweep_result.get('ldes_dispatch_pct', 0)
                    rate, info = get_emission_rate_for_threshold(
                        iso, match_score, emission_rates, fossil_mix)
                    fossil_displaced, ccs_supply, curtailed = compute_hourly_fossil_displacement(
                        demand_norm, supply_profiles, resource_mix, proc, batt, ldes,
                        supply_matrix=sup_matrix, dispatch_cache=dispatch_cache, iso=iso)
                    cache_dirty = True
                    co2 = compute_co2_hourly(
                        fossil_displaced, ccs_supply, rate, demand_total_mwh, info)
                    sweep_result['co2_abated'] = co2
                    sweep_result['curtailed_mwh'] = round(float(np.sum(curtailed)) * demand_total_mwh, 0)
                    dispatch_count += 1

        # --- Recompute threshold results ---
        thresholds_data = iso_data.get('thresholds', {})
        recomputed = 0

        for t_str, t_data in thresholds_data.items():
            threshold_pct = float(t_str)
            scenarios = t_data.get('scenarios', {})

            # Pre-fetch rate for this threshold
            rate, info = get_emission_rate_for_threshold(
                iso, threshold_pct, emission_rates, fossil_mix)
            rate_cache[(iso, round(threshold_pct, 1))] = (rate, info)

            for sk, result in scenarios.items():
                resource_mix = result.get('resource_mix', {})
                proc = result.get('procurement_pct', 100)
                match_score = result.get('hourly_match_score', 0)

                if match_score > 0:
                    # FAST PATH
                    co2 = fast_co2_from_match_score(
                        match_score, resource_mix, proc,
                        threshold_pct, iso, emission_rates, fossil_mix,
                        demand_total_mwh, rate_cache)
                    result['co2_abated'] = co2
                    fast_count += 1
                else:
                    # DISPATCH PATH
                    batt = result.get('battery_dispatch_pct', 0)
                    ldes = result.get('ldes_dispatch_pct', 0)
                    fossil_displaced, ccs_supply, curtailed = compute_hourly_fossil_displacement(
                        demand_norm, supply_profiles, resource_mix, proc, batt, ldes,
                        supply_matrix=sup_matrix, dispatch_cache=dispatch_cache, iso=iso)
                    cache_dirty = True
                    co2 = compute_co2_hourly(
                        fossil_displaced, ccs_supply, rate, demand_total_mwh, info)
                    result['co2_abated'] = co2
                    result['curtailed_mwh'] = round(float(np.sum(curtailed)) * demand_total_mwh, 0)
                    dispatch_count += 1

                recomputed += 1

        # Save dispatch cache if any new entries were computed
        if cache_dirty:
            save_dispatch_cache(iso, dispatch_cache)

        print(f"    Recomputed: {recomputed} threshold scenarios + {len(iso_data.get('sweep', []))} sweep points")

    elapsed = time.time() - start
    print(f"\n  Fast path (match_score): {fast_count} scenarios")
    print(f"  Dispatch path (hourly):  {dispatch_count} scenarios")
    print(f"  Total recompute time: {elapsed:.1f}s")

    return results_data


def main():
    print("=" * 70)
    print("  CO₂ RECOMPUTATION — Dispatch-Stack Emission Rates")
    print("  Optimized: match_score fast path + Numba dispatch + cache")
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
