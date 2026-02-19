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
"""

import json
import os
import sys
import numpy as np
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
RESULTS_PATH = os.path.join(SCRIPT_DIR, 'dashboard', 'overprocure_results.json')
CACHE_PATH = os.path.join(DATA_DIR, 'optimizer_cache.json')

H = 8760
DATA_YEAR = '2025'

ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO']
RESOURCE_TYPES = ['clean_firm', 'solar', 'wind', 'ccs_ccgt', 'hydro']
CCS_RESIDUAL_EMISSION_RATE = 0.037  # tCO2/MWh after 90% capture

# Battery / LDES parameters (must match optimizer)
BATTERY_EFFICIENCY = 0.85
BATTERY_DURATION_HOURS = 4
LDES_EFFICIENCY = 0.50
LDES_DURATION_HOURS = 100
LDES_WINDOW_DAYS = 7

HYDRO_CAPS = {
    'CAISO': 30, 'ERCOT': 5, 'PJM': 15, 'NYISO': 40, 'NEISO': 30,
}

# Existing clean generation shares (% of total generation) — from step4 GRID_MIX_SHARES
GRID_MIX_SHARES = {
    'CAISO': {'clean_firm': 7.9, 'solar': 22.3, 'wind': 8.8, 'ccs_ccgt': 0, 'hydro': 9.5},
    'ERCOT': {'clean_firm': 8.6, 'solar': 13.8, 'wind': 23.6, 'ccs_ccgt': 0, 'hydro': 0.1},
    'PJM':   {'clean_firm': 32.1, 'solar': 2.9, 'wind': 3.8, 'ccs_ccgt': 0, 'hydro': 1.8},
    'NYISO': {'clean_firm': 18.4, 'solar': 0.0, 'wind': 4.7, 'ccs_ccgt': 0, 'hydro': 15.9},
    'NEISO': {'clean_firm': 23.8, 'solar': 1.4, 'wind': 3.9, 'ccs_ccgt': 0, 'hydro': 4.4},
}

# Threshold above which all coal and oil have retired — only gas CCGT remains
COAL_OIL_RETIREMENT_THRESHOLD = 70.0  # % clean energy


def compute_dispatch_stack_emission_rate(iso, clean_pct, emission_rates, fossil_mix):
    """
    Compute the emission rate of the remaining fossil fleet at a given clean energy %.

    Uses merit-order retirement: coal retires first (dirtiest), then oil, then gas.
    Above COAL_OIL_RETIREMENT_THRESHOLD (70%), forces gas-only.

    Args:
        iso: ISO region name
        clean_pct: Clean energy percentage (0-100) for this scenario
        emission_rates: eGRID per-fuel emission rates dict
        fossil_mix: EIA fossil mix data (annual average shares within fossil)

    Returns:
        emission_rate: tCO₂/MWh of the remaining fossil fleet at this clean %
        retirement_info: dict with coal/oil/gas displaced fractions for diagnostics
    """
    regional_data = emission_rates.get(iso, {})
    coal_rate = regional_data.get('coal_co2_lb_per_mwh', 0.0) / 2204.62  # lb → metric tons
    gas_rate = regional_data.get('gas_co2_lb_per_mwh', 0.0) / 2204.62
    oil_rate = regional_data.get('oil_co2_lb_per_mwh', 0.0) / 2204.62

    # Baseline clean % from existing grid mix
    baseline_clean = sum(GRID_MIX_SHARES.get(iso, {}).values())
    fossil_total = 100.0 - baseline_clean

    if fossil_total <= 0:
        return 0.0, {'coal_displaced': 0, 'oil_displaced': 0, 'gas_displaced': 0}

    # Get annual average fossil fuel shares within fossil generation
    iso_fossil = fossil_mix.get(iso, {})
    year_data = iso_fossil.get(DATA_YEAR, iso_fossil.get('2024', {}))
    coal_shares_arr = year_data.get('coal_share', [0.0] * H)
    gas_shares_arr = year_data.get('gas_share', [1.0] * H)
    oil_shares_arr = year_data.get('oil_share', [0.0] * H)

    # Annual average shares (within fossil)
    avg_coal_share = np.mean(coal_shares_arr[:H])
    avg_gas_share = np.mean(gas_shares_arr[:H])
    avg_oil_share = np.mean(oil_shares_arr[:H])

    # Convert to % of total generation
    coal_total_pct = avg_coal_share * fossil_total
    oil_total_pct = avg_oil_share * fossil_total
    gas_total_pct = avg_gas_share * fossil_total

    # Additional clean energy above baseline
    additional_clean = max(0, clean_pct - baseline_clean)

    # Above 70% clean: force gas-only (all coal and oil retired)
    if clean_pct >= COAL_OIL_RETIREMENT_THRESHOLD:
        gas_remaining = max(0, fossil_total - additional_clean)
        return gas_rate, {
            'coal_displaced': coal_total_pct,
            'oil_displaced': oil_total_pct,
            'gas_displaced': min(additional_clean - coal_total_pct - oil_total_pct, gas_total_pct),
            'forced_gas_only': True,
        }

    # Merit-order displacement: coal first, then oil, then gas
    coal_displaced = min(additional_clean, coal_total_pct)
    remaining = additional_clean - coal_displaced
    oil_displaced = min(remaining, oil_total_pct)
    remaining = remaining - oil_displaced
    gas_displaced = min(remaining, gas_total_pct)

    coal_remaining = coal_total_pct - coal_displaced
    oil_remaining = oil_total_pct - oil_displaced
    gas_remaining = gas_total_pct - gas_displaced
    fossil_remaining = coal_remaining + oil_remaining + gas_remaining

    if fossil_remaining <= 0.01:
        # Effectively 100% clean
        return gas_rate, {
            'coal_displaced': coal_displaced,
            'oil_displaced': oil_displaced,
            'gas_displaced': gas_displaced,
        }

    # Weighted average emission rate of remaining fossil fleet
    rate = (coal_remaining * coal_rate + oil_remaining * oil_rate + gas_remaining * gas_rate) / fossil_remaining

    return rate, {
        'coal_displaced': round(coal_displaced, 2),
        'oil_displaced': round(oil_displaced, 2),
        'gas_displaced': round(gas_displaced, 2),
        'coal_remaining_pct': round(coal_remaining, 2),
        'oil_remaining_pct': round(oil_remaining, 2),
        'gas_remaining_pct': round(gas_remaining, 2),
        'emission_rate_tco2_mwh': round(rate, 4),
    }


def load_data():
    """Load all required data files."""
    with open(os.path.join(DATA_DIR, 'eia_demand_profiles.json')) as f:
        demand_data = json.load(f)

    with open(os.path.join(DATA_DIR, 'eia_generation_profiles.json')) as f:
        gen_profiles = json.load(f)

    with open(os.path.join(DATA_DIR, 'egrid_emission_rates.json')) as f:
        emission_rates = json.load(f)

    with open(os.path.join(DATA_DIR, 'eia_fossil_mix.json')) as f:
        fossil_mix = json.load(f)

    return demand_data, gen_profiles, emission_rates, fossil_mix


def get_supply_profiles(iso, gen_profiles):
    """Get generation shape profiles (must match optimizer's function)."""
    profiles = {}
    profiles['clean_firm'] = [1.0 / H] * H

    if iso == 'NYISO':
        p = gen_profiles[iso][DATA_YEAR].get('solar_proxy')
        if not p:
            p = gen_profiles['NEISO'][DATA_YEAR].get('solar')
        profiles['solar'] = p[:H]
    else:
        profiles['solar'] = gen_profiles[iso][DATA_YEAR].get('solar', [0.0] * H)[:H]

    profiles['wind'] = gen_profiles[iso][DATA_YEAR].get('wind', [0.0] * H)[:H]
    profiles['ccs_ccgt'] = [1.0 / H] * H
    profiles['hydro'] = gen_profiles[iso][DATA_YEAR].get('hydro', [0.0] * H)[:H]

    for rtype in RESOURCE_TYPES:
        if len(profiles[rtype]) > H:
            profiles[rtype] = profiles[rtype][:H]
        elif len(profiles[rtype]) < H:
            profiles[rtype] = profiles[rtype] + [0.0] * (H - len(profiles[rtype]))

    return profiles


def get_emission_rate_for_threshold(iso, threshold_pct, emission_rates, fossil_mix):
    """
    Get the single emission rate (tCO₂/MWh) for a given clean energy threshold.

    Uses the dispatch-stack retirement model: coal retires first, then oil, then gas.
    Above 70% clean, emission rate = gas CCGT only.

    Returns: (emission_rate, retirement_info)
    """
    return compute_dispatch_stack_emission_rate(iso, threshold_pct, emission_rates, fossil_mix)


def compute_hourly_fossil_displacement(demand_norm, supply_profiles, resource_pcts,
                                        procurement_pct, battery_dispatch_pct, ldes_dispatch_pct):
    """
    Reconstruct hourly clean supply and compute fossil displacement at each hour.

    For each hour: fossil_displaced[h] = min(demand[h], total_clean_supply[h])
    where total_clean_supply includes battery/LDES dispatch.

    Returns:
        fossil_displaced: numpy array (H,) — MWh of fossil displaced at each hour (normalized)
        ccs_supply: numpy array (H,) — CCS-CCGT supply at each hour (for partial credit)
        curtailed: numpy array (H,) — curtailed energy at each hour
    """
    procurement_factor = procurement_pct / 100.0
    demand_arr = np.array(demand_norm[:H], dtype=np.float64)

    # Build total supply profile from resource mix
    supply_total = np.zeros(H, dtype=np.float64)
    ccs_supply = np.zeros(H, dtype=np.float64)

    for rtype in RESOURCE_TYPES:
        pct = resource_pcts.get(rtype, 0)
        if pct <= 0:
            continue
        profile = np.array(supply_profiles[rtype][:H], dtype=np.float64)
        contribution = procurement_factor * (pct / 100.0) * profile
        supply_total += contribution
        if rtype == 'ccs_ccgt':
            ccs_supply = contribution.copy()

    # Battery daily-cycle dispatch (simplified reconstruction)
    residual_surplus = np.maximum(0.0, supply_total - demand_arr)
    residual_gap = np.maximum(0.0, demand_arr - supply_total)

    battery_dispatch_profile = np.zeros(H, dtype=np.float64)
    if battery_dispatch_pct > 0:
        batt_dispatch_total = battery_dispatch_pct / 100.0
        num_days = H // 24
        daily_dispatch_target = batt_dispatch_total / num_days
        batt_power_rating = daily_dispatch_target / BATTERY_DURATION_HOURS

        for day in range(num_days):
            ds = day * 24
            de = ds + 24
            day_surplus = residual_surplus[ds:de].copy()
            day_gap = residual_gap[ds:de].copy()

            max_from_charge = day_surplus.sum() * BATTERY_EFFICIENCY
            actual_dispatch = min(daily_dispatch_target, max_from_charge, day_gap.sum())
            if actual_dispatch <= 0:
                continue

            required_charge = actual_dispatch / BATTERY_EFFICIENCY

            # Charge from largest surpluses
            sorted_idx = np.argsort(-day_surplus)
            remaining_charge = required_charge
            for idx in sorted_idx:
                if remaining_charge <= 0 or day_surplus[idx] <= 0:
                    break
                amt = min(float(day_surplus[idx]), batt_power_rating, remaining_charge)
                residual_surplus[ds + idx] -= amt
                remaining_charge -= amt

            # Discharge to largest gaps
            sorted_gap = np.argsort(-day_gap)
            remaining_dispatch = actual_dispatch
            for idx in sorted_gap:
                if remaining_dispatch <= 0 or day_gap[idx] <= 0:
                    break
                amt = min(float(day_gap[idx]), batt_power_rating, remaining_dispatch)
                battery_dispatch_profile[ds + idx] = amt
                residual_gap[ds + idx] -= amt
                remaining_dispatch -= amt

    # LDES dispatch (simplified reconstruction)
    ldes_dispatch_profile = np.zeros(H, dtype=np.float64)
    if ldes_dispatch_pct > 0:
        total_demand_energy = demand_arr.sum()
        ldes_energy_capacity = total_demand_energy * (24.0 / H)
        ldes_power_rating = ldes_energy_capacity / LDES_DURATION_HOURS
        state_of_charge = 0.0
        window_hours = LDES_WINDOW_DAYS * 24

        num_windows = (H + window_hours - 1) // window_hours
        for w in range(num_windows):
            w_start = w * window_hours
            w_end = min(w_start + window_hours, H)

            w_surplus = residual_surplus[w_start:w_end].copy()
            w_gap = residual_gap[w_start:w_end].copy()

            # Charge
            surplus_indices = np.argsort(-w_surplus)
            for idx in surplus_indices:
                if w_surplus[idx] <= 0:
                    break
                space = ldes_energy_capacity - state_of_charge
                if space <= 0:
                    break
                charge_amt = min(float(w_surplus[idx]), ldes_power_rating, space)
                if charge_amt > 0:
                    state_of_charge += charge_amt

            # Discharge
            gap_indices = np.argsort(-w_gap)
            for idx in gap_indices:
                if w_gap[idx] <= 0:
                    break
                avail = state_of_charge * LDES_EFFICIENCY
                if avail <= 0:
                    break
                dispatch_amt = min(float(w_gap[idx]), ldes_power_rating, avail)
                if dispatch_amt > 0:
                    ldes_dispatch_profile[w_start + idx] = dispatch_amt
                    state_of_charge -= dispatch_amt / LDES_EFFICIENCY
                    residual_gap[w_start + idx] -= dispatch_amt

    # Total clean supply including storage dispatch
    total_clean = supply_total + battery_dispatch_profile + ldes_dispatch_profile

    # Fossil displaced = demand met by clean energy at each hour
    fossil_displaced = np.minimum(demand_arr, total_clean)

    # Curtailed = clean supply exceeding demand
    curtailed = np.maximum(0.0, total_clean - demand_arr)

    return fossil_displaced, ccs_supply, curtailed


def compute_co2_hourly(fossil_displaced, ccs_supply, emission_rate, demand_total_mwh,
                       retirement_info=None):
    """
    Compute CO₂ abated using dispatch-stack emission rate.

    fossil_displaced: normalized hourly fossil displacement (sums to ~matched_fraction)
    ccs_supply: normalized hourly CCS-CCGT supply
    emission_rate: single tCO₂/MWh rate for the remaining fossil fleet at this threshold
    demand_total_mwh: total annual demand in MWh (to denormalize)
    retirement_info: optional dict with fuel retirement diagnostics

    For clean resources: each displaced MWh avoids emission_rate tons CO₂
    For CCS-CCGT: each MWh avoids (emission_rate - residual) tons CO₂
    """
    scale = demand_total_mwh

    # Non-CCS clean displacement (fossil_displaced minus CCS portion)
    non_ccs_displaced = np.maximum(0.0, fossil_displaced - np.minimum(fossil_displaced, ccs_supply))
    ccs_displaced = np.minimum(fossil_displaced, ccs_supply)

    # CO₂ from non-CCS clean: full credit at threshold emission rate
    co2_clean = np.sum(non_ccs_displaced) * emission_rate * scale

    # CO₂ from CCS: partial credit (emission_rate - residual)
    ccs_credit = max(0.0, emission_rate - CCS_RESIDUAL_EMISSION_RATE)
    co2_ccs = np.sum(ccs_displaced) * ccs_credit * scale

    total_abated = co2_clean + co2_ccs

    # Summary stats
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
    """
    Recompute CO₂ for all results using dispatch-stack retirement model.
    Emission rates are threshold-dependent: coal retires first, then oil, then gas.
    Above 70% clean, only gas CCGT remains.
    Updates results_data in-place.
    """
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

        # Log dispatch-stack emission rates at key thresholds
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

                # Use match score as the clean % for dispatch-stack
                rate, info = get_emission_rate_for_threshold(
                    iso, match_score, emission_rates, fossil_mix
                )

                fossil_displaced, ccs_supply, curtailed = compute_hourly_fossil_displacement(
                    demand_norm, supply_profiles, resource_mix, proc, batt, ldes
                )

                co2 = compute_co2_hourly(
                    fossil_displaced, ccs_supply, rate, demand_total_mwh, info
                )
                sweep_result['co2_abated'] = co2
                sweep_result['curtailed_mwh'] = round(float(np.sum(curtailed)) * demand_total_mwh, 0)

        # Recompute threshold results
        thresholds_data = iso_data.get('thresholds', {})
        recomputed = 0

        for t_str, t_data in thresholds_data.items():
            threshold_pct = float(t_str)
            scenarios = t_data.get('scenarios', {})

            # Get emission rate for this threshold (dispatch-stack model)
            rate, info = get_emission_rate_for_threshold(
                iso, threshold_pct, emission_rates, fossil_mix
            )

            # Always recompute Medium scenario
            medium_key = 'MMM_M_M'
            if medium_key in scenarios:
                result = scenarios[medium_key]
                resource_mix = result.get('resource_mix', {})
                proc = result.get('procurement_pct', 100)
                batt = result.get('battery_dispatch_pct', 0)
                ldes = result.get('ldes_dispatch_pct', 0)

                fossil_displaced, ccs_supply, curtailed = compute_hourly_fossil_displacement(
                    demand_norm, supply_profiles, resource_mix, proc, batt, ldes
                )

                co2 = compute_co2_hourly(
                    fossil_displaced, ccs_supply, rate, demand_total_mwh, info
                )
                result['co2_abated'] = co2
                result['curtailed_mwh'] = round(float(np.sum(curtailed)) * demand_total_mwh, 0)

                # Under dispatch-stack model, fuel price sensitivity no longer shifts
                # emission rates (coal/oil retired above 70%, no switching possible).
                # Below 70%, the stack determines composition, not fuel prices.
                # Store the single rate for all fuel scenarios.
                result['co2_abated_fuel_sensitivity'] = {
                    'Low': co2,
                    'Medium': co2,
                    'High': co2,
                }

                recomputed += 1

            # For non-Medium scenarios, same emission rate (threshold-dependent, not fuel-dependent)
            for sk, result in scenarios.items():
                if sk == medium_key:
                    continue

                resource_mix = result.get('resource_mix', {})
                proc = result.get('procurement_pct', 100)
                batt = result.get('battery_dispatch_pct', 0)
                ldes = result.get('ldes_dispatch_pct', 0)

                fossil_displaced, ccs_supply, curtailed = compute_hourly_fossil_displacement(
                    demand_norm, supply_profiles, resource_mix, proc, batt, ldes
                )

                co2 = compute_co2_hourly(
                    fossil_displaced, ccs_supply, rate, demand_total_mwh, info
                )
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

    # Load data
    print("\n  Loading data...")
    demand_data, gen_profiles, emission_rates, fossil_mix = load_data()

    # Load results
    if not os.path.exists(RESULTS_PATH):
        print(f"  ERROR: Results file not found: {RESULTS_PATH}")
        sys.exit(1)

    with open(RESULTS_PATH) as f:
        results_data = json.load(f)
    print(f"  Loaded results: {RESULTS_PATH}")

    # Check which ISOs are present
    isos_present = [iso for iso in ISOS if iso in results_data.get('results', {})]
    print(f"  ISOs in results: {isos_present}")

    # Recompute all CO₂
    results_data = recompute_all_co2(results_data, demand_data, gen_profiles, emission_rates, fossil_mix)

    # Save updated results
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results_data, f)
    print(f"\n  Updated: {RESULTS_PATH} ({os.path.getsize(RESULTS_PATH) / 1024:.0f} KB)")

    # Cache is LOCKED — do not modify
    # CO2 data is only written to the dashboard results file
    print(f"  Cache ({CACHE_PATH}) is locked — skipped")

    print(f"\n{'='*70}")
    print("  CO₂ RECOMPUTATION COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
