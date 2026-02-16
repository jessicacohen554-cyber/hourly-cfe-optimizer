#!/usr/bin/env python3
"""
Post-Processor: Recompute CO₂ Abatement with Hourly Fossil-Fuel Emission Rates
================================================================================
Replaces the flat regional emission rate with hourly variable rates built from:
  - eGRID 2023 per-fuel emission factors (coal, gas, oil) per region
  - EIA hourly fossil fuel mix shares per ISO

For each result, reconstructs hourly clean supply → fossil displacement, then:
  CO₂_abated[h] = fossil_displaced[h] × emission_rate[h]
  where emission_rate[h] = coal_share[h]×coal_rate + gas_share[h]×gas_rate + oil_share[h]×oil_rate

Also computes fuel-price-shifted emission rates (Low/Medium/High fossil fuel scenarios).

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

# Fuel-switching elasticity factors: how much the coal/gas share shifts
# under Low/High gas price scenarios. Positive = coal share increases.
# Based on SPEC.md Section 5.9.
# Format: {iso: {fuel_level: coal_share_shift}}
# At Low gas price, coal share decreases (more gas dispatch)
# At High gas price, coal share increases (gas-to-coal switching)
FUEL_SWITCHING_SHIFTS = {
    'CAISO': {'Low': -0.005, 'Medium': 0.0, 'High': 0.01},   # Very low elasticity (near-zero coal)
    'ERCOT': {'Low': -0.03,  'Medium': 0.0, 'High': 0.05},   # Low elasticity (limited coal)
    'PJM':   {'Low': -0.08,  'Medium': 0.0, 'High': 0.15},   # High elasticity (45GW coal)
    'NYISO': {'Low': -0.01,  'Medium': 0.0, 'High': 0.02},   # Low elasticity (minimal coal)
    'NEISO': {'Low': -0.01,  'Medium': 0.0, 'High': 0.02},   # Low elasticity (retiring coal)
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


def build_hourly_emission_rates(iso, emission_rates, fossil_mix, fuel_level='Medium'):
    """
    Build 8760-hour emission rate array from eGRID per-fuel rates × EIA hourly fossil mix.

    emission_rate[h] = coal_share[h] × coal_rate + gas_share[h] × gas_rate + oil_share[h] × oil_rate

    Under fuel price shifts, coal/gas shares are adjusted per fuel-switching elasticity.

    Returns: numpy array of shape (H,) with emission rates in metric tons CO₂/MWh
    """
    regional_data = emission_rates.get(iso, {})
    coal_rate = regional_data.get('coal_co2_lb_per_mwh', 0.0) / 2204.62  # lb → metric tons
    gas_rate = regional_data.get('gas_co2_lb_per_mwh', 0.0) / 2204.62
    oil_rate = regional_data.get('oil_co2_lb_per_mwh', 0.0) / 2204.62

    # Get hourly fossil mix shares
    iso_fossil = fossil_mix.get(iso, {})
    # Try DATA_YEAR first, fall back to most recent year
    year_data = iso_fossil.get(DATA_YEAR, iso_fossil.get('2024', {}))

    coal_shares = np.array(year_data.get('coal_share', [0.0] * H)[:H], dtype=np.float64)
    gas_shares = np.array(year_data.get('gas_share', [1.0] * H)[:H], dtype=np.float64)
    oil_shares = np.array(year_data.get('oil_share', [0.0] * H)[:H], dtype=np.float64)

    # Pad to H if shorter
    if len(coal_shares) < H:
        coal_shares = np.pad(coal_shares, (0, H - len(coal_shares)), mode='edge')
    if len(gas_shares) < H:
        gas_shares = np.pad(gas_shares, (0, H - len(gas_shares)), mode='edge')
    if len(oil_shares) < H:
        oil_shares = np.pad(oil_shares, (0, H - len(oil_shares)), mode='edge')

    # Apply fuel-switching shift
    shift = FUEL_SWITCHING_SHIFTS.get(iso, {}).get(fuel_level, 0.0)
    if shift != 0.0:
        # Shift coal share up/down, compensate with gas share
        coal_shares_shifted = np.clip(coal_shares + shift, 0.0, 1.0)
        gas_shares_shifted = np.clip(gas_shares - shift, 0.0, 1.0)

        # Renormalize so shares sum to 1.0
        total = coal_shares_shifted + gas_shares_shifted + oil_shares
        mask = total > 0
        coal_shares[mask] = coal_shares_shifted[mask] / total[mask]
        gas_shares[mask] = gas_shares_shifted[mask] / total[mask]
        oil_shares[mask] = oil_shares[mask] / total[mask]

    # Compute hourly emission rate (tons CO₂/MWh of fossil generation)
    hourly_rates = coal_shares * coal_rate + gas_shares * gas_rate + oil_shares * oil_rate

    return hourly_rates


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


def compute_co2_hourly(fossil_displaced, ccs_supply, hourly_emission_rates, demand_total_mwh):
    """
    Compute CO₂ abated using hourly emission rates.

    fossil_displaced: normalized hourly fossil displacement (sums to ~matched_fraction)
    ccs_supply: normalized hourly CCS-CCGT supply
    hourly_emission_rates: tons CO₂/MWh at each hour
    demand_total_mwh: total annual demand in MWh (to denormalize)

    For clean resources: each displaced MWh avoids emission_rate[h] tons CO₂
    For CCS-CCGT: each MWh avoids (emission_rate[h] - 0.037) tons CO₂
    """
    # Scale from normalized to actual MWh
    scale = demand_total_mwh

    # Non-CCS clean displacement (fossil_displaced minus CCS portion)
    non_ccs_displaced = np.maximum(0.0, fossil_displaced - np.minimum(fossil_displaced, ccs_supply))
    ccs_displaced = np.minimum(fossil_displaced, ccs_supply)

    # CO₂ from non-CCS clean: full credit at hourly rate
    co2_clean = np.sum(non_ccs_displaced * hourly_emission_rates) * scale

    # CO₂ from CCS: partial credit (emission_rate[h] - residual)
    ccs_credit = np.maximum(0.0, hourly_emission_rates - CCS_RESIDUAL_EMISSION_RATE)
    co2_ccs = np.sum(ccs_displaced * ccs_credit) * scale

    total_abated = co2_clean + co2_ccs

    # Summary stats
    matched_mwh = np.sum(fossil_displaced) * scale
    co2_rate = total_abated / matched_mwh if matched_mwh > 0 else 0

    # Hourly emission rate stats
    weighted_avg_rate = np.average(hourly_emission_rates, weights=fossil_displaced) \
        if np.sum(fossil_displaced) > 0 else np.mean(hourly_emission_rates)

    return {
        'total_co2_abated_tons': round(total_abated, 0),
        'co2_rate_per_mwh': round(co2_rate, 4),
        'matched_mwh': round(matched_mwh, 0),
        'hourly_emission_rate_avg_tons': round(np.mean(hourly_emission_rates), 4),
        'hourly_emission_rate_weighted_avg_tons': round(weighted_avg_rate, 4),
        'hourly_emission_rate_min_tons': round(np.min(hourly_emission_rates), 4),
        'hourly_emission_rate_max_tons': round(np.max(hourly_emission_rates), 4),
        'methodology': 'hourly_fossil_fuel_emission_rates',
    }


def recompute_all_co2(results_data, demand_data, gen_profiles, emission_rates, fossil_mix):
    """
    Recompute CO₂ for all results using hourly emission rates.
    Updates results_data in-place.
    """
    start = time.time()

    for iso in ISOS:
        if iso not in results_data.get('results', {}):
            continue

        iso_data = results_data['results'][iso]
        # Get demand from DATA_YEAR, falling back through years
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

        # Build hourly emission rates for L/M/H fuel scenarios
        emission_rate_by_fuel = {}
        for fuel_level in ['Low', 'Medium', 'High']:
            emission_rate_by_fuel[fuel_level] = build_hourly_emission_rates(
                iso, emission_rates, fossil_mix, fuel_level
            )

        # Log emission rate stats
        med_rates = emission_rate_by_fuel['Medium']
        print(f"\n  {iso}:")
        print(f"    Hourly emission rate (Medium): "
              f"avg={np.mean(med_rates):.4f}, min={np.min(med_rates):.4f}, "
              f"max={np.max(med_rates):.4f} tCO₂/MWh")

        old_flat_rate = emission_rates.get(iso, {}).get('fossil_co2_lb_per_mwh', 0) / 2204.62
        print(f"    Old flat rate: {old_flat_rate:.4f} tCO₂/MWh")
        print(f"    Difference: {((np.mean(med_rates) - old_flat_rate) / old_flat_rate * 100):+.1f}%")

        # Recompute sweep results
        if 'sweep' in iso_data:
            for i, sweep_result in enumerate(iso_data['sweep']):
                resource_mix = sweep_result.get('resource_mix', {})
                proc = sweep_result.get('procurement_pct', 100)
                batt = sweep_result.get('battery_dispatch_pct', 0)
                ldes = sweep_result.get('ldes_dispatch_pct', 0)

                fossil_displaced, ccs_supply, curtailed = compute_hourly_fossil_displacement(
                    demand_norm, supply_profiles, resource_mix, proc, batt, ldes
                )

                co2 = compute_co2_hourly(
                    fossil_displaced, ccs_supply,
                    emission_rate_by_fuel['Medium'],
                    demand_total_mwh
                )
                sweep_result['co2_abated'] = co2
                sweep_result['curtailed_mwh'] = round(float(np.sum(curtailed)) * demand_total_mwh, 0)

        # Recompute threshold results (Medium scenario)
        thresholds_data = iso_data.get('thresholds', {})
        recomputed = 0

        for t_str, t_data in thresholds_data.items():
            scenarios = t_data.get('scenarios', {})

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
                    fossil_displaced, ccs_supply,
                    emission_rate_by_fuel['Medium'],
                    demand_total_mwh
                )
                result['co2_abated'] = co2
                result['curtailed_mwh'] = round(float(np.sum(curtailed)) * demand_total_mwh, 0)

                # Also compute CO₂ under Low and High fuel scenarios for sensitivity
                co2_low = compute_co2_hourly(
                    fossil_displaced, ccs_supply,
                    emission_rate_by_fuel['Low'],
                    demand_total_mwh
                )
                co2_high = compute_co2_hourly(
                    fossil_displaced, ccs_supply,
                    emission_rate_by_fuel['High'],
                    demand_total_mwh
                )
                result['co2_abated_fuel_sensitivity'] = {
                    'Low': co2_low,
                    'Medium': co2,
                    'High': co2_high,
                }

                recomputed += 1

            # For non-Medium scenarios, compute CO₂ using the scenario's fuel level
            for sk, result in scenarios.items():
                if sk == medium_key:
                    continue  # Already done above

                resource_mix = result.get('resource_mix', {})
                proc = result.get('procurement_pct', 100)
                batt = result.get('battery_dispatch_pct', 0)
                ldes = result.get('ldes_dispatch_pct', 0)

                # Extract fuel level from scenario key
                # Key format: {renew}{firm}{storage}_{fuel}_{tx}
                # e.g., LMH_H_L → fuel=H → High
                parts = sk.split('_')
                fuel_level = 'Medium'
                if len(parts) >= 2:
                    fuel_char = parts[1]
                    fuel_level = {'L': 'Low', 'M': 'Medium', 'H': 'High'}.get(fuel_char, 'Medium')

                fossil_displaced, ccs_supply, curtailed = compute_hourly_fossil_displacement(
                    demand_norm, supply_profiles, resource_mix, proc, batt, ldes
                )

                hourly_rates = emission_rate_by_fuel.get(fuel_level, emission_rate_by_fuel['Medium'])
                co2 = compute_co2_hourly(
                    fossil_displaced, ccs_supply,
                    hourly_rates,
                    demand_total_mwh
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

    # Update cache if it exists
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            cache_data = json.load(f)
        cache_data['results'] = results_data['results']
        cache_data['metadata']['co2_recomputed'] = True
        cache_data['metadata']['co2_methodology'] = 'hourly_fossil_fuel_emission_rates'
        with open(CACHE_PATH, 'w') as f:
            json.dump(cache_data, f, indent=2)
        print(f"  Updated: {CACHE_PATH} ({os.path.getsize(CACHE_PATH) / 1024:.0f} KB)")

    print(f"\n{'='*70}")
    print("  CO₂ RECOMPUTATION COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
