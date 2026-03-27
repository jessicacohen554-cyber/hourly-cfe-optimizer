#!/usr/bin/env python3
"""
Step 7 — Act 4 PJM Grid Deployment Simulation
================================================
Simulates PJM's grid-level journey to 95% hourly clean energy under two
strategies, running full 8760 dispatch at each milestone to compute actual
Hourly Matching Scores (HMS).

Strategy 1B (Consequential / MAC queue):
  Walks PJM's MAC queue in marginal abatement cost order.
  Deploys cheapest-to-abate resources first (mostly VRE).
  Existing clean stays on grid; new-build adds on top.
  Floor ratchet: resources never decrease between milestones.

Strategy 2C (Hourly matching / co-optimized):
  Uses HOURLY_MIX_TEMPLATE from step5_2c which designs resource mixes
  specifically for hourly matching (firm + storage + VRE balanced).
  Deploys incrementally on top of existing + prior builds.
  Floor ratchet: same cumulative floor as 1B.

Gas calculation uses PEAK_CAPACITY_CREDITS (ELCC) matching step 2.2:
  - All resources get capacity credit per ELCC table
  - gas_needed = (RA_peak - clean_peak_mw) / GAS_AVAILABILITY_FACTOR
  - new_gas = max(0, gas_needed - existing_gas)

Output: dashboard/js/act4-pjm-data.js
"""

import os
import sys
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

import pandas as pd
from dispatch_utils import (
    load_common_data, get_demand_profile, get_supply_profiles,
    reconstruct_hourly_dispatch, build_supply_matrix,
    H, RESOURCE_TYPES,
)
from pipeline_config import (
    REGIONAL_DEMAND_TWH, PEAK_DEMAND_MW, EXISTING_GAS_CAPACITY_MW,
    RESOURCE_ADEQUACY_MARGIN, GRID_MIX_SHARES,
    DEMAND_GROWTH_RATES, THRESHOLD_TARGET_YEARS,
    RESOURCE_CAPACITY_FACTORS,
    PEAK_CAPACITY_CREDITS, GAS_AVAILABILITY_FACTOR,
    NEW_CCGT_COST_KW_YR,
)

ISO = 'PJM'
GROWTH_LEVEL = 'Medium'
PRICE_SENSITIVITY = 'all_med'

# HMS milestones we want to report
HMS_MILESTONES = [50, 60, 70, 80, 90, 95]

# Target years from pipeline_config
TARGET_YEARS = THRESHOLD_TARGET_YEARS

# HOURLY_MIX_TEMPLATE from step5_2c: co-optimized resource fractions for hourly matching
# These represent the INCREMENTAL new-build mix allocated at each threshold step
HOURLY_MIX_TEMPLATE = {
    50:    {'solar': 0.40, 'wind': 0.35, 'firm': 0.10, 'battery': 0.08, 'ldes': 0.02, 'uprate': 0.05},
    70:    {'solar': 0.35, 'wind': 0.30, 'firm': 0.15, 'battery': 0.10, 'ldes': 0.05, 'uprate': 0.05},
    85:    {'solar': 0.25, 'wind': 0.25, 'firm': 0.25, 'battery': 0.12, 'ldes': 0.08, 'uprate': 0.05},
    90:    {'solar': 0.22, 'wind': 0.22, 'firm': 0.28, 'battery': 0.13, 'ldes': 0.10, 'uprate': 0.05},
    95:    {'solar': 0.18, 'wind': 0.18, 'firm': 0.32, 'battery': 0.14, 'ldes': 0.13, 'uprate': 0.05},
    99:    {'solar': 0.15, 'wind': 0.15, 'firm': 0.35, 'battery': 0.15, 'ldes': 0.15, 'uprate': 0.05},
    99.5:  {'solar': 0.14, 'wind': 0.14, 'firm': 0.36, 'battery': 0.15, 'ldes': 0.16, 'uprate': 0.05},
    99.9:  {'solar': 0.13, 'wind': 0.13, 'firm': 0.37, 'battery': 0.15, 'ldes': 0.17, 'uprate': 0.05},
    99.99: {'solar': 0.12, 'wind': 0.12, 'firm': 0.38, 'battery': 0.15, 'ldes': 0.18, 'uprate': 0.05},
}

# Over-procurement ratios (from step5_2c — physics penalty of hourly matching)
OVER_PROCUREMENT_RATIO = {
    50: 1.05, 55: 1.07, 60: 1.10, 65: 1.13, 70: 1.16, 75: 1.20,
    80: 1.25, 85: 1.32, 87.5: 1.38, 90: 1.45, 92.5: 1.55, 95: 1.70,
    97.5: 1.90, 99: 2.20, 99.5: 2.50, 99.9: 2.80, 99.99: 3.00,
}


def get_hourly_mix_fractions(threshold):
    """Interpolate HOURLY_MIX_TEMPLATE for a given threshold."""
    thresholds = sorted(HOURLY_MIX_TEMPLATE.keys())
    resources = list(HOURLY_MIX_TEMPLATE[thresholds[0]].keys())
    if threshold <= thresholds[0]:
        return dict(HOURLY_MIX_TEMPLATE[thresholds[0]])
    if threshold >= thresholds[-1]:
        return dict(HOURLY_MIX_TEMPLATE[thresholds[-1]])
    for i in range(len(thresholds) - 1):
        if thresholds[i] <= threshold <= thresholds[i + 1]:
            frac = (threshold - thresholds[i]) / (thresholds[i + 1] - thresholds[i])
            low = HOURLY_MIX_TEMPLATE[thresholds[i]]
            high = HOURLY_MIX_TEMPLATE[thresholds[i + 1]]
            return {r: low[r] + frac * (high[r] - low[r]) for r in resources}
    return dict(HOURLY_MIX_TEMPLATE[thresholds[-1]])


def get_over_procurement_ratio(threshold):
    """Interpolate over-procurement ratio."""
    thresholds = sorted(OVER_PROCUREMENT_RATIO.keys())
    if threshold <= thresholds[0]:
        return OVER_PROCUREMENT_RATIO[thresholds[0]]
    if threshold >= thresholds[-1]:
        return OVER_PROCUREMENT_RATIO[thresholds[-1]]
    for i in range(len(thresholds) - 1):
        if thresholds[i] <= threshold <= thresholds[i + 1]:
            frac = (threshold - thresholds[i]) / (thresholds[i + 1] - thresholds[i])
            return OVER_PROCUREMENT_RATIO[thresholds[i]] + frac * (
                OVER_PROCUREMENT_RATIO[thresholds[i + 1]] - OVER_PROCUREMENT_RATIO[thresholds[i]])
    return OVER_PROCUREMENT_RATIO[thresholds[-1]]


def load_pjm_mac_queue():
    """Load PJM MAC queue data (Medium growth, all_med sensitivity)."""
    path = os.path.join(ROOT_DIR, 'data', 'step3-dispatch', 'mac_queue', 'mac_queue_PJM.parquet')
    df = pd.read_parquet(path)
    df = df[(df['demand_growth'] == GROWTH_LEVEL) & (df['price_sensitivity'] == PRICE_SENSITIVITY)]
    df = df.sort_values('threshold')
    return df


def compute_hms(demand_norm, supply_profiles, supply_matrix, resource_pcts,
                battery_pct=0, battery8_pct=0, ldes_pct=0, h2_pct=0):
    """Run 8760 dispatch and compute HMS = sum(min(clean[h], demand[h])) / sum(demand) * 100."""
    result = reconstruct_hourly_dispatch(
        demand_norm, supply_profiles, resource_pcts,
        procurement_pct=100,
        battery_dispatch_pct=battery_pct,
        battery8_dispatch_pct=battery8_pct,
        ldes_dispatch_pct=ldes_pct,
        h2_dispatch_pct=h2_pct,
        supply_matrix=supply_matrix,
        detailed=True,
    )
    total_clean = result['total_clean']
    demand_arr = np.array(demand_norm[:H], dtype=np.float64)
    matched = np.minimum(total_clean, demand_arr)
    hms = np.sum(matched) / np.sum(demand_arr) * 100.0
    curtailed = np.sum(result['curtailed'])

    # Per-resource curtailment (from detailed dispatch)
    curt_by_resource = {}
    for rtype in ['solar', 'wind', 'offshore_wind', 'clean_firm', 'hydro']:
        surplus_key = f'surplus_{rtype}'
        if surplus_key in result:
            curt_by_resource[rtype] = float(np.sum(result[surplus_key]))
        else:
            curt_by_resource[rtype] = 0.0

    return hms, float(curtailed), result, curt_by_resource


def compute_gas_dispatch_based(dispatch_result, demand_twh, baseline_gas_raw):
    """Compute gas needed from dispatch residual demand, matching step 2.2.

    Uses the actual peak residual demand hour from 8760 dispatch to size gas.
    This is the same method as scenario_common.py lines 414-448.

    Returns (new_gas_mw, total_gas_mw).
    """
    demand_mwh = demand_twh * 1e6
    gaf = GAS_AVAILABILITY_FACTOR[ISO]
    existing_gas = EXISTING_GAS_CAPACITY_MW[ISO]

    # Peak fossil gap hour from dispatch
    net_peak_norm = float(np.max(dispatch_result['residual_demand']))
    net_peak_mw = net_peak_norm * demand_mwh
    gas_raw = net_peak_mw * (1 + RESOURCE_ADEQUACY_MARGIN) / gaf

    # Delta calibration: how much gas changed vs baseline
    gas_delta = gas_raw - baseline_gas_raw
    gas_needed_mw = max(0, existing_gas + gas_delta)
    existing_gas_used = min(gas_needed_mw, existing_gas)
    new_gas_mw = max(0, gas_needed_mw - existing_gas_used)

    return new_gas_mw, gas_needed_mw


def compute_baseline_gas(demand_norm, supply_profiles, supply_matrix):
    """Compute baseline gas from existing resources only (for delta calibration)."""
    base_demand_mwh = REGIONAL_DEMAND_TWH[ISO] * 1e6
    gaf = GAS_AVAILABILITY_FACTOR[ISO]
    baseline_pcts = dict(GRID_MIX_SHARES[ISO])
    disp = reconstruct_hourly_dispatch(
        demand_norm, supply_profiles, baseline_pcts,
        procurement_pct=100, supply_matrix=supply_matrix,
    )
    baseline_peak = float(np.max(disp['residual_demand'])) * base_demand_mwh
    return baseline_peak * (1 + RESOURCE_ADEQUACY_MARGIN) / gaf


def compute_clean_gen_capex(new_build_twh):
    """Compute clean generation capex from new-build TWh."""
    capex = 0.0
    cf = RESOURCE_CAPACITY_FACTORS

    for res, occ_kw in [('solar', 1300), ('wind', 1500), ('offshore_wind', 4500), ('clean_firm', 8000)]:
        twh = new_build_twh.get(res, 0)
        if twh <= 0:
            continue
        res_cf = cf.get(res, {}).get(ISO, 0.25)
        mw = (twh * 1e6) / (8760 * res_cf)
        capex += mw * occ_kw * 1000  # MW * $/kW * 1000

    return capex


def build_entry(resource_pcts, battery_pct, battery8_pct, ldes_pct, h2_pct,
                demand_twh, target_year, threshold, existing_clean_twh,
                demand_norm, supply_profiles, supply_matrix, base_demand, existing,
                cum_new_build, baseline_gas_raw, mac_cost_total=None):
    """Build one result entry: run dispatch, compute HMS, gas, capex."""

    hms, curt_total, dispatch, curt_by_res = compute_hms(
        demand_norm, supply_profiles, supply_matrix, resource_pcts,
        battery_pct=battery_pct,
        battery8_pct=battery8_pct,
        ldes_pct=ldes_pct,
        h2_pct=h2_pct,
    )

    curt_twh = curt_total * demand_twh

    # Gas via dispatch-based residual demand (matching step 2.2)
    new_gas_mw, total_gas_mw = compute_gas_dispatch_based(
        dispatch, demand_twh, baseline_gas_raw)

    # Capex
    clean_capex = compute_clean_gen_capex(cum_new_build)
    gas_capex_b = new_gas_mw * NEW_CCGT_COST_KW_YR[ISO] * 1000 * 20 / 1e9  # 20yr levelized
    total_capex = clean_capex / 1e9 + gas_capex_b

    # Per-resource curtailment in TWh
    curt_solar_twh = curt_by_res.get('solar', 0) * demand_twh
    curt_wind_twh = (curt_by_res.get('wind', 0) + curt_by_res.get('offshore_wind', 0)) * demand_twh
    curt_firm_twh = curt_by_res.get('clean_firm', 0) * demand_twh

    entry = {
        'threshold': float(threshold),
        'year': target_year,
        'demandTwh': round(demand_twh, 1),
        'hms': round(hms, 1),
        'gasGw': round(new_gas_mw / 1000, 1),
        'totalGasGw': round(total_gas_mw / 1000, 1),
        'cleanCapexB': round(clean_capex / 1e9, 1),
        'totalCapexB': round(total_capex, 1),
        'curtTwh': round(curt_twh, 1),
        'curtSolarTwh': round(curt_solar_twh, 1),
        'curtWindTwh': round(curt_wind_twh, 1),
        'curtFirmTwh': round(curt_firm_twh, 1),
        # Bar chart data (TWh)
        'existingCleanTwh': round(existing_clean_twh, 1),
        'newSolarTwh': round(cum_new_build.get('solar', 0), 1),
        'newWindTwh': round(cum_new_build.get('wind', 0) + cum_new_build.get('offshore_wind', 0), 1),
        'newFirmTwh': round(cum_new_build.get('clean_firm', 0), 1),
        'batteryTwh': round(battery_pct / 100 * demand_twh, 1) if battery_pct > 0 else 0,
        'ldesTwh': round(ldes_pct / 100 * demand_twh, 1) if ldes_pct > 0 else 0,
        'newBuildCostTotal': round(mac_cost_total / 1e9, 1) if mac_cost_total else round(clean_capex / 1e9, 1),
    }

    return entry, hms


def simulate_strategy_1b(mac_queue, demand_norm, supply_profiles, supply_matrix, baseline_gas_raw):
    """Simulate Strategy 1B: walk PJM's MAC queue, compute HMS at each step.

    The MAC queue's `deployed_*_twh` columns are cumulative (floor ratchet built in).
    """
    print("\n=== Strategy 1B (Consequential / MAC Queue) ===")
    results = []
    base_demand = REGIONAL_DEMAND_TWH[ISO]
    existing = GRID_MIX_SHARES[ISO]
    existing_clean_twh = sum(v / 100 * base_demand for v in existing.values())

    for _, row in mac_queue.iterrows():
        threshold = float(row['threshold'])
        demand_twh = float(row['demand_twh'])
        target_year = int(row['target_year'])

        # MAC queue deployed_* columns are cumulative (already include floor ratchet)
        # Convert to % of demand for dispatch (dispatch works in % of base demand)
        resource_pcts = {
            'clean_firm': float(row['deployed_clean_firm_twh']) / base_demand * 100,
            'solar': float(row['deployed_solar_twh']) / base_demand * 100,
            'wind': float(row['deployed_wind_twh']) / base_demand * 100,
            'hydro': float(row['deployed_hydro_twh']) / base_demand * 100,
            'offshore_wind': float(row['deployed_offshore_wind_twh']) / base_demand * 100,
            'ccs_ccgt': 0,
        }

        battery_pct = float(row.get('deployed_battery_dispatch_pct', 0))
        battery8_pct = float(row.get('deployed_battery8_dispatch_pct', 0))
        ldes_pct = float(row.get('deployed_ldes_dispatch_pct', 0))
        h2_pct = float(row.get('deployed_h2_dispatch_pct', 0))

        # Cumulative new build = deployed - existing
        cum_new_build = {
            'clean_firm': max(0, float(row['deployed_clean_firm_twh']) - existing['clean_firm'] / 100 * base_demand),
            'solar': max(0, float(row['deployed_solar_twh']) - existing['solar'] / 100 * base_demand),
            'wind': max(0, float(row['deployed_wind_twh']) - existing['wind'] / 100 * base_demand),
            'offshore_wind': max(0, float(row['deployed_offshore_wind_twh']) - existing.get('offshore_wind', 0) / 100 * base_demand),
        }

        entry, hms = build_entry(
            resource_pcts, battery_pct, battery8_pct, ldes_pct, h2_pct,
            demand_twh, target_year, threshold, existing_clean_twh,
            demand_norm, supply_profiles, supply_matrix, base_demand, existing,
            cum_new_build, baseline_gas_raw, mac_cost_total=float(row['cumulative_new_build_cost']))

        print(f"  Threshold {threshold:5.1f}% → HMS {hms:5.1f}%, "
              f"New Gas {entry['gasGw']:.1f} GW, Curt {entry['curtTwh']:.0f} TWh, "
              f"Year {target_year}")

        results.append(entry)

    return results


def simulate_strategy_2c(demand_norm, supply_profiles, supply_matrix, baseline_gas_raw):
    """Simulate Strategy 2C: deploy co-optimized hourly matching portfolio.

    Uses HOURLY_MIX_TEMPLATE to determine the resource mix for new-build at
    each threshold. Each milestone adds incremental clean energy to hit the
    target, with floor ratchet (never unbuild).

    The key difference from 1B: early investment in firm + storage alongside
    VRE, not VRE-dominant. This costs more upfront but prevents gas lock-in.
    """
    print("\n=== Strategy 2C (Hourly Matching / Co-Optimized) ===")
    results = []
    base_demand = REGIONAL_DEMAND_TWH[ISO]
    existing = GRID_MIX_SHARES[ISO]
    existing_clean_twh = sum(v / 100 * base_demand for v in existing.values())

    # Use same thresholds as the milestones we target
    thresholds = [50, 55, 60, 65, 70, 75, 80, 85, 87.5, 90, 92.5, 95]

    # Floor ratchet: cumulative resources never decrease
    floor_solar_twh = 0.0
    floor_wind_twh = 0.0
    floor_firm_twh = 0.0
    floor_battery_pct = 0.0
    floor_ldes_pct = 0.0

    prev_hms = sum(existing.values())  # baseline HMS ~40%

    for threshold in thresholds:
        target_year = TARGET_YEARS.get(int(threshold), TARGET_YEARS.get(threshold, 2045))
        demand_twh = base_demand * (1 + DEMAND_GROWTH_RATES[ISO][GROWTH_LEVEL]) ** (target_year - 2025)

        # How much NEW clean energy do we need to go from current HMS to target?
        # The gap is (target - current) as % of demand, times over-procurement
        hms_gap = max(0, threshold - prev_hms)
        incremental_twh = hms_gap / 100 * demand_twh * get_over_procurement_ratio(threshold)

        # Allocate incremental new-build using HOURLY_MIX_TEMPLATE fractions
        mix = get_hourly_mix_fractions(threshold)
        gen_frac = mix['solar'] + mix['wind'] + mix['firm'] + mix['uprate']

        # Incremental adds on top of floor
        solar_twh = floor_solar_twh + incremental_twh * mix['solar'] / gen_frac
        wind_twh = floor_wind_twh + incremental_twh * mix['wind'] / gen_frac
        firm_twh = floor_firm_twh + incremental_twh * (mix['firm'] + mix['uprate']) / gen_frac

        # Storage: incremental capacity
        battery_pct = floor_battery_pct + mix['battery'] * incremental_twh / demand_twh * 100
        ldes_pct = floor_ldes_pct + mix['ldes'] * incremental_twh / demand_twh * 100

        # Floor ratchet: update floors
        floor_solar_twh = solar_twh
        floor_wind_twh = wind_twh
        floor_firm_twh = firm_twh
        floor_battery_pct = battery_pct
        floor_ldes_pct = ldes_pct

        # Total resources on grid = existing + new build (as % of base demand for dispatch)
        resource_pcts = {
            'clean_firm': (existing['clean_firm'] / 100 * base_demand + firm_twh) / base_demand * 100,
            'solar': (existing['solar'] / 100 * base_demand + solar_twh) / base_demand * 100,
            'wind': (existing['wind'] / 100 * base_demand + wind_twh) / base_demand * 100,
            'hydro': existing['hydro'],  # hydro stays at existing
            'offshore_wind': existing.get('offshore_wind', 0),
            'ccs_ccgt': 0,
        }

        cum_new_build = {
            'clean_firm': firm_twh,
            'solar': solar_twh,
            'wind': wind_twh,
            'offshore_wind': 0,
        }

        entry, hms = build_entry(
            resource_pcts, battery_pct, 0, ldes_pct, 0,
            demand_twh, target_year, threshold, existing_clean_twh,
            demand_norm, supply_profiles, supply_matrix, base_demand, existing,
            cum_new_build, baseline_gas_raw)

        # Update prev_hms for incremental gap calculation
        prev_hms = hms

        print(f"  Threshold {threshold:5.1f}% → HMS {hms:5.1f}%, "
              f"New Gas {entry['gasGw']:.1f} GW, Curt {entry['curtTwh']:.0f} TWh, "
              f"Year {target_year}, New: S={solar_twh:.0f} W={wind_twh:.0f} F={firm_twh:.0f} TWh")

        results.append(entry)

    return results


def interpolate_to_hms_milestones(entries, milestones):
    """Interpolate strategy results to exact HMS milestone values."""
    if not entries:
        return []

    sorted_entries = sorted(entries, key=lambda e: e['hms'])
    result = []

    for target_hms in milestones:
        below = None
        above = None
        for e in sorted_entries:
            if e['hms'] <= target_hms:
                below = e
            if e['hms'] >= target_hms and above is None:
                above = e

        if below is None and above is not None:
            result.append(dict(above, hms=round(above['hms'], 1)))
            continue
        if above is None and below is not None:
            result.append(dict(below, hms=round(below['hms'], 1)))
            continue
        if below is None and above is None:
            continue
        if below == above:
            result.append(dict(below))
            continue

        frac = (target_hms - below['hms']) / (above['hms'] - below['hms'])
        interp = {}
        for key in below:
            if isinstance(below[key], (int, float)):
                interp[key] = round(below[key] + frac * (above[key] - below[key]), 1)
            else:
                interp[key] = below[key] if frac < 0.5 else above[key]
        interp['hms'] = round(target_hms, 1)
        if 'year' in interp:
            interp['year'] = int(round(interp['year']))
        result.append(interp)

    return result


def main():
    print("Loading PJM common data...")
    common_data = load_common_data()
    demand_data, gen_profiles = common_data[0], common_data[1]

    demand_norm, total_mwh = get_demand_profile(ISO, demand_data)
    supply_profiles = get_supply_profiles(ISO, gen_profiles)
    supply_matrix = build_supply_matrix(supply_profiles)

    print(f"PJM base demand: {total_mwh/1e6:.1f} TWh")
    print(f"PJM peak: {PEAK_DEMAND_MW[ISO]} MW, RA target: {PEAK_DEMAND_MW[ISO] * (1 + RESOURCE_ADEQUACY_MARGIN):.0f} MW")
    print(f"PJM existing gas: {EXISTING_GAS_CAPACITY_MW[ISO]} MW")
    print(f"ELCC credits: {PEAK_CAPACITY_CREDITS}")

    # Compute baseline gas (existing resources only) for delta calibration
    baseline_gas_raw = compute_baseline_gas(demand_norm, supply_profiles, supply_matrix)
    print(f"Baseline gas (existing only): {baseline_gas_raw/1000:.1f} GW")

    # Load MAC queue for 1B
    mac_queue = load_pjm_mac_queue()
    print(f"MAC queue: {len(mac_queue)} entries")

    # Run simulations
    results_1b = simulate_strategy_1b(mac_queue, demand_norm, supply_profiles, supply_matrix, baseline_gas_raw)
    results_2c = simulate_strategy_2c(demand_norm, supply_profiles, supply_matrix, baseline_gas_raw)

    # Interpolate to HMS milestones
    milestones_1b = interpolate_to_hms_milestones(results_1b, HMS_MILESTONES)
    milestones_2c = interpolate_to_hms_milestones(results_2c, HMS_MILESTONES)

    print("\n=== HMS Milestone Summary ===")
    print(f"{'HMS':>6}  {'1B Year':>7} {'1B NewGas':>9} {'1B Curt':>8}  {'2C Year':>7} {'2C NewGas':>9} {'2C Curt':>8}")
    for m1, m2 in zip(milestones_1b, milestones_2c):
        print(f"{m1['hms']:5.0f}%  {m1['year']:>7} {m1['gasGw']:>8.1f} {m1['curtTwh']:>7.0f}  "
              f"{m2['year']:>7} {m2['gasGw']:>8.1f} {m2['curtTwh']:>7.0f}")

    # Write output
    output = {
        'iso': ISO,
        'baseDemandTwh': REGIONAL_DEMAND_TWH[ISO],
        'existingGasGw': EXISTING_GAS_CAPACITY_MW[ISO] / 1000,
        'growthLevel': GROWTH_LEVEL,
        'milestones': HMS_MILESTONES,
        'strategy1B': {
            'name': 'Consequential (MAC Queue)',
            'description': 'Deploy cheapest abatement first — VRE-heavy, drives gas lock-in',
            'raw': results_1b,
            'milestones': milestones_1b,
        },
        'strategy2C': {
            'name': 'Hourly Matching (Co-Optimized)',
            'description': 'Co-optimized portfolio — firm + storage + VRE balanced for hourly matching',
            'raw': results_2c,
            'milestones': milestones_2c,
        },
    }

    out_path = os.path.join(ROOT_DIR, 'dashboard', 'js', 'act4-pjm-data.js')
    with open(out_path, 'w') as f:
        f.write('// Auto-generated by step7_act4_pjm_deployment.py\n')
        f.write('// PJM grid-level deployment simulation with 8760 dispatch HMS\n')
        f.write(f'// Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        f.write('var ACT4_PJM_DATA = ')
        json.dump(output, f, indent=2)
        f.write(';\n')

    print(f"\nOutput written to {out_path}")
    print(f"  Strategy 1B: {len(results_1b)} raw entries → {len(milestones_1b)} milestones")
    print(f"  Strategy 2C: {len(results_2c)} raw entries → {len(milestones_2c)} milestones")


if __name__ == '__main__':
    main()
