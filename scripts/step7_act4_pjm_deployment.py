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

Strategy 2C (Hourly matching / EF-optimized):
  Uses physics-co-optimized resource mixes from the Efficient Frontier.
  At each threshold, deploys the EF winner mix that achieves the target HMS
  with firm + storage + VRE balanced for hourly matching.

For each milestone:
  - Runs reconstruct_hourly_dispatch() to get actual HMS
  - Computes new gas CCGT needed for resource adequacy
  - Tracks curtailment by resource type
  - Outputs clean generation capex and total capex

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
    LCOE_TABLES, TX_TABLES,
)

ISO = 'PJM'
GROWTH_LEVEL = 'Medium'
PRICE_SENSITIVITY = 'all_med'

# HMS milestones we want to report
HMS_MILESTONES = [50, 60, 70, 80, 90, 95]

# Target years from pipeline_config
TARGET_YEARS = THRESHOLD_TARGET_YEARS

# Gas CCGT cost assumptions ($/kW overnight, Medium scenario)
GAS_CCGT_CAPEX_PER_KW = 1200  # $/kW for new CCGT


def load_pjm_mac_queue():
    """Load PJM MAC queue data (Medium growth, all_med sensitivity)."""
    path = os.path.join(ROOT_DIR, 'data', 'step3-dispatch', 'mac_queue', 'mac_queue_PJM.parquet')
    df = pd.read_parquet(path)
    df = df[(df['demand_growth'] == GROWTH_LEVEL) & (df['price_sensitivity'] == PRICE_SENSITIVITY)]
    df = df.sort_values('threshold')
    return df


def load_ef_mixes():
    """Load EF winner mixes for PJM at each threshold."""
    import glob
    ef_dir = os.path.join(ROOT_DIR, 'data', 'step2.1-ef')
    results = {}
    for f in glob.glob(os.path.join(ef_dir, 'step_2_1_EF_PJM_*.parquet')):
        fname = os.path.basename(f)
        # Parse threshold from filename: step_2_1_EF_PJM_50.parquet
        thresh_str = fname.replace('step_2_1_EF_PJM_', '').replace('.parquet', '')
        try:
            thresh = float(thresh_str)
        except ValueError:
            continue
        df = pd.read_parquet(f)
        if df.empty:
            continue
        # Get the mix with highest HMS
        best = df.loc[df['hourly_match_score'].idxmax()]
        results[thresh] = {
            'clean_firm': float(best['clean_firm']),
            'solar': float(best['solar']),
            'wind': float(best['wind']),
            'hydro': float(best['hydro']),
            'offshore_wind': float(best['offshore_wind']),
            'battery_dispatch_pct': float(best['battery_dispatch_pct']),
            'battery8_dispatch_pct': float(best['battery8_dispatch_pct']),
            'ldes_dispatch_pct': float(best['ldes_dispatch_pct']),
            'h2_dispatch_pct': float(best.get('h2_dispatch_pct', 0)),
            'hourly_match_score': float(best['hourly_match_score']),
        }
    return results


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


def compute_gas_needed_mw(demand_twh, clean_firm_twh, year):
    """Compute new gas CCGT capacity needed for resource adequacy.

    Peak demand grows with demand_twh. Clean firm provides dispatchable capacity.
    Gas fills the gap to meet peak × (1 + RA margin).

    Returns (new_gas_mw, total_gas_mw).
    """
    base_demand = REGIONAL_DEMAND_TWH[ISO]
    base_peak = PEAK_DEMAND_MW[ISO]
    growth_factor = demand_twh / base_demand
    peak_mw = base_peak * growth_factor

    ra_target = peak_mw * (1 + RESOURCE_ADEQUACY_MARGIN)

    # Dispatchable clean capacity: clean firm at its capacity factor
    cf_firm = RESOURCE_CAPACITY_FACTORS['clean_firm'].get(ISO, 0.93)
    clean_firm_mw = (clean_firm_twh * 1e6) / (8760 * cf_firm)

    # Battery and LDES contribute some capacity credit but let's be conservative
    # and only count clean firm + existing gas
    existing_gas = EXISTING_GAS_CAPACITY_MW[ISO]

    total_dispatchable = clean_firm_mw + existing_gas
    gas_gap = max(0, ra_target - total_dispatchable)

    # new_gas = gap above existing gas capacity
    # But existing gas already exists, so new gas = max(0, gas_gap)
    # Actually: total gas needed = ra_target - clean_firm_mw
    # new gas = total gas needed - existing gas
    total_gas_needed = max(0, ra_target - clean_firm_mw)
    new_gas_mw = max(0, total_gas_needed - existing_gas)

    return new_gas_mw, total_gas_needed


def compute_clean_gen_capex(new_build_twh, demand_twh):
    """Compute clean generation capex (solar + wind + offshore wind + clean firm).

    Uses Medium LCOE × capacity factor to derive implied MW, then $/kW capex.
    Simplified: use new_build_cost_total from MAC queue when available,
    otherwise estimate from LCOE tables.
    """
    # We'll use a simplified capex model based on capacity implied by TWh
    capex = 0.0
    cf = RESOURCE_CAPACITY_FACTORS

    # Solar: $/kW × MW
    if 'solar' in new_build_twh and new_build_twh['solar'] > 0:
        solar_cf = cf['solar'].get(ISO, 0.17)
        solar_mw = (new_build_twh['solar'] * 1e6) / (8760 * solar_cf)
        capex += solar_mw * 1300 * 1000  # $1300/kW

    # Wind (onshore)
    if 'wind' in new_build_twh and new_build_twh['wind'] > 0:
        wind_cf = cf['wind'].get(ISO, 0.30)
        wind_mw = (new_build_twh['wind'] * 1e6) / (8760 * wind_cf)
        capex += wind_mw * 1500 * 1000  # $1500/kW

    # Offshore wind
    if 'offshore_wind' in new_build_twh and new_build_twh['offshore_wind'] > 0:
        osw_cf = cf.get('offshore_wind', {}).get(ISO, 0.48)
        osw_mw = (new_build_twh['offshore_wind'] * 1e6) / (8760 * osw_cf)
        capex += osw_mw * 4500 * 1000  # $4500/kW

    # Clean firm (nuclear)
    if 'clean_firm' in new_build_twh and new_build_twh['clean_firm'] > 0:
        firm_cf = cf['clean_firm'].get(ISO, 0.93)
        firm_mw = (new_build_twh['clean_firm'] * 1e6) / (8760 * firm_cf)
        capex += firm_mw * 8000 * 1000  # $8000/kW for nuclear

    return capex


def simulate_strategy_1b(mac_queue, demand_norm, supply_profiles, supply_matrix):
    """Simulate Strategy 1B: walk PJM's MAC queue, compute HMS at each step.

    The MAC queue's `deployed_*_twh` columns represent the cumulative resource
    portfolio on PJM's grid at each threshold. We convert these to % of demand
    and run 8760 dispatch to compute actual HMS.
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

        # Deployed resources as % of demand (for dispatch)
        resource_pcts = {
            'clean_firm': float(row['deployed_clean_firm_twh']) / demand_twh * 100,
            'solar': float(row['deployed_solar_twh']) / demand_twh * 100,
            'wind': float(row['deployed_wind_twh']) / demand_twh * 100,
            'hydro': float(row['deployed_hydro_twh']) / demand_twh * 100,
            'offshore_wind': float(row['deployed_offshore_wind_twh']) / demand_twh * 100,
            'ccs_ccgt': 0,
        }

        battery_pct = float(row.get('deployed_battery_dispatch_pct', 0))
        battery8_pct = float(row.get('deployed_battery8_dispatch_pct', 0))
        ldes_pct = float(row.get('deployed_ldes_dispatch_pct', 0))
        h2_pct = float(row.get('deployed_h2_dispatch_pct', 0))

        # Scale demand_norm to account for demand growth
        # The supply profiles are normalized to base demand, so we scale resource_pcts
        # to reflect the deployment relative to base demand
        growth_factor = demand_twh / base_demand
        scaled_pcts = {k: v * growth_factor for k, v in resource_pcts.items()}

        hms, curt_total, dispatch, curt_by_res = compute_hms(
            demand_norm, supply_profiles, supply_matrix, scaled_pcts,
            battery_pct=battery_pct * growth_factor,
            battery8_pct=battery8_pct * growth_factor,
            ldes_pct=ldes_pct * growth_factor,
            h2_pct=h2_pct * growth_factor,
        )

        # Curtailment in TWh (dispatch is normalized, multiply by demand_twh)
        curt_twh = curt_total * demand_twh

        # New-build TWh
        new_build = {
            'clean_firm': float(row['new_build_clean_firm_twh']),
            'solar': float(row['new_build_solar_twh']),
            'wind': float(row['new_build_wind_twh']),
            'offshore_wind': float(row['new_build_offshore_wind_twh']),
        }
        # Cumulative new build = deployed - existing
        cum_new_build = {
            'clean_firm': max(0, float(row['deployed_clean_firm_twh']) - existing['clean_firm'] / 100 * base_demand),
            'solar': max(0, float(row['deployed_solar_twh']) - existing['solar'] / 100 * base_demand),
            'wind': max(0, float(row['deployed_wind_twh']) - existing['wind'] / 100 * base_demand),
            'offshore_wind': max(0, float(row['deployed_offshore_wind_twh']) - existing.get('offshore_wind', 0) / 100 * base_demand),
        }

        # Gas capacity
        new_gas_mw, total_gas_mw = compute_gas_needed_mw(
            demand_twh, float(row['deployed_clean_firm_twh']), target_year
        )

        # Capex
        clean_capex = compute_clean_gen_capex(cum_new_build, demand_twh)
        gas_capex = new_gas_mw * GAS_CCGT_CAPEX_PER_KW * 1000  # MW * $/kW * 1000 = $
        total_capex = clean_capex + gas_capex

        # Per-resource curtailment in TWh
        curt_solar_twh = curt_by_res.get('solar', 0) * demand_twh
        curt_wind_twh = (curt_by_res.get('wind', 0) + curt_by_res.get('offshore_wind', 0)) * demand_twh
        curt_firm_twh = curt_by_res.get('clean_firm', 0) * demand_twh

        # Deployed TWh for the bar chart
        deployed = {
            'existing_clean_twh': existing_clean_twh,
            'solar_twh': max(0, float(row['deployed_solar_twh']) - existing['solar'] / 100 * base_demand),
            'wind_twh': max(0, float(row['deployed_wind_twh']) - existing['wind'] / 100 * base_demand +
                           float(row['deployed_offshore_wind_twh']) - existing.get('offshore_wind', 0) / 100 * base_demand),
            'clean_firm_twh': max(0, float(row['deployed_clean_firm_twh']) - existing['clean_firm'] / 100 * base_demand),
            'battery_twh': battery_pct / 100 * demand_twh * growth_factor if battery_pct > 0 else 0,
            'ldes_twh': ldes_pct / 100 * demand_twh * growth_factor if ldes_pct > 0 else 0,
        }

        entry = {
            'threshold': threshold,
            'year': target_year,
            'demandTwh': round(demand_twh, 1),
            'hms': round(hms, 1),
            'gasGw': round(new_gas_mw / 1000, 1),
            'totalGasGw': round(total_gas_mw / 1000, 1),
            'cleanCapexB': round(clean_capex / 1e9, 1),
            'totalCapexB': round(total_capex / 1e9, 1),
            'curtTwh': round(curt_twh, 1),
            'curtSolarTwh': round(curt_solar_twh, 1),
            'curtWindTwh': round(curt_wind_twh, 1),
            'curtFirmTwh': round(curt_firm_twh, 1),
            # Bar chart data (TWh)
            'existingCleanTwh': round(existing_clean_twh, 1),
            'newSolarTwh': round(deployed['solar_twh'], 1),
            'newWindTwh': round(deployed['wind_twh'], 1),
            'newFirmTwh': round(deployed['clean_firm_twh'], 1),
            'batteryTwh': round(deployed['battery_twh'], 1),
            'ldesTwh': round(deployed['ldes_twh'], 1),
            'newBuildCostTotal': round(float(row['cumulative_new_build_cost']) / 1e9, 1),
        }

        print(f"  Threshold {threshold:5.1f}% → HMS {hms:5.1f}%, "
              f"Gas {new_gas_mw/1000:.1f} GW, Curt {curt_twh:.0f} TWh, "
              f"Year {target_year}")

        results.append(entry)

    return results


def simulate_strategy_2c(ef_mixes, demand_norm, supply_profiles, supply_matrix):
    """Simulate Strategy 2C: deploy EF-optimized mixes onto PJM's grid.

    For each HMS milestone, find the EF mix closest to that target and deploy
    it onto PJM's grid with demand growth. The EF mix represents the total
    portfolio (as % of base demand) needed to achieve the target HMS.

    Key difference from 1B: the EF mixes are physics-co-optimized portfolios
    that include clean firm + storage. They achieve higher HMS with less
    curtailment because the resources are balanced for hourly matching.
    """
    print("\n=== Strategy 2C (Hourly Matching / EF-Optimized) ===")
    results = []
    base_demand = REGIONAL_DEMAND_TWH[ISO]
    existing = GRID_MIX_SHARES[ISO]
    existing_clean_twh = sum(v / 100 * base_demand for v in existing.values())

    # Use the same thresholds as the MAC queue for comparability
    target_thresholds = sorted([t for t in ef_mixes.keys() if 50 <= t <= 95])

    for threshold in target_thresholds:
        ef = ef_mixes[threshold]
        target_year = TARGET_YEARS.get(int(threshold), TARGET_YEARS.get(threshold, 2045))
        demand_twh = base_demand * (1 + DEMAND_GROWTH_RATES[ISO][GROWTH_LEVEL]) ** (target_year - 2025)

        # EF mix is in % of base demand. These are the TOTAL resources on-grid.
        # The EF was optimized at base demand, so the percentages represent
        # the correct resource ratios. We pass them directly — dispatch_utils
        # handles the normalization (resource_pcts * supply_profiles / demand).
        resource_pcts = {
            'clean_firm': ef['clean_firm'],
            'solar': ef['solar'],
            'wind': ef['wind'],
            'hydro': ef.get('hydro', 0),
            'offshore_wind': ef.get('offshore_wind', 0),
            'ccs_ccgt': 0,
        }

        # Ensure existing clean is at least maintained
        for res in ['clean_firm', 'solar', 'wind', 'hydro']:
            existing_pct = GRID_MIX_SHARES[ISO].get(res, 0)
            if resource_pcts[res] < existing_pct:
                resource_pcts[res] = existing_pct

        battery_pct = ef.get('battery_dispatch_pct', 0)
        battery8_pct = ef.get('battery8_dispatch_pct', 0)
        ldes_pct = ef.get('ldes_dispatch_pct', 0)
        h2_pct = ef.get('h2_dispatch_pct', 0)

        # Run dispatch at base demand ratios (EF was optimized for base)
        hms, curt_total, dispatch, curt_by_res = compute_hms(
            demand_norm, supply_profiles, supply_matrix, resource_pcts,
            battery_pct=battery_pct,
            battery8_pct=battery8_pct,
            ldes_pct=ldes_pct,
            h2_pct=h2_pct,
        )

        # Curtailment in TWh (using actual demand at year)
        curt_twh = curt_total * demand_twh

        # New build = total deployed - existing (in TWh at projected demand)
        new_firm_twh = max(0, resource_pcts['clean_firm'] / 100 * demand_twh - existing['clean_firm'] / 100 * base_demand)
        new_solar_twh = max(0, resource_pcts['solar'] / 100 * demand_twh - existing['solar'] / 100 * base_demand)
        new_wind_twh = max(0, resource_pcts['wind'] / 100 * demand_twh - existing['wind'] / 100 * base_demand)
        new_osw_twh = max(0, resource_pcts['offshore_wind'] / 100 * demand_twh)

        cum_new_build = {
            'clean_firm': new_firm_twh,
            'solar': new_solar_twh,
            'wind': new_wind_twh,
            'offshore_wind': new_osw_twh,
        }

        # Gas capacity
        total_firm_twh = resource_pcts['clean_firm'] / 100 * demand_twh
        new_gas_mw, total_gas_mw = compute_gas_needed_mw(demand_twh, total_firm_twh, target_year)

        # Capex
        clean_capex = compute_clean_gen_capex(cum_new_build, demand_twh)
        gas_capex = new_gas_mw * GAS_CCGT_CAPEX_PER_KW * 1000
        total_capex = clean_capex + gas_capex

        # Curtailment by resource
        curt_solar_twh = curt_by_res.get('solar', 0) * demand_twh
        curt_wind_twh = (curt_by_res.get('wind', 0) + curt_by_res.get('offshore_wind', 0)) * demand_twh
        curt_firm_twh = curt_by_res.get('clean_firm', 0) * demand_twh

        # Battery/LDES TWh dispatched
        batt_twh = battery_pct / 100 * demand_twh if battery_pct > 0 else 0
        ldes_twh_val = ldes_pct / 100 * demand_twh if ldes_pct > 0 else 0

        entry = {
            'threshold': float(threshold),
            'year': target_year,
            'demandTwh': round(demand_twh, 1),
            'hms': round(hms, 1),
            'gasGw': round(new_gas_mw / 1000, 1),
            'totalGasGw': round(total_gas_mw / 1000, 1),
            'cleanCapexB': round(clean_capex / 1e9, 1),
            'totalCapexB': round(total_capex / 1e9, 1),
            'curtTwh': round(curt_twh, 1),
            'curtSolarTwh': round(curt_solar_twh, 1),
            'curtWindTwh': round(curt_wind_twh, 1),
            'curtFirmTwh': round(curt_firm_twh, 1),
            'existingCleanTwh': round(existing_clean_twh, 1),
            'newSolarTwh': round(new_solar_twh, 1),
            'newWindTwh': round(new_wind_twh + new_osw_twh, 1),
            'newFirmTwh': round(new_firm_twh, 1),
            'batteryTwh': round(batt_twh, 1),
            'ldesTwh': round(ldes_twh_val, 1),
            'newBuildCostTotal': round(clean_capex / 1e9, 1),
        }

        print(f"  Threshold {threshold:5.1f}% → HMS {hms:5.1f}%, "
              f"Gas {new_gas_mw/1000:.1f} GW, Curt {curt_twh:.0f} TWh, "
              f"Year {target_year}")

        results.append(entry)

    return results


def interpolate_to_hms_milestones(entries, milestones):
    """Interpolate strategy results to exact HMS milestone values.

    Given a list of entries with 'hms' field, find (or interpolate between)
    entries that bracket each milestone HMS value.
    """
    if not entries:
        return []

    # Sort by HMS
    sorted_entries = sorted(entries, key=lambda e: e['hms'])
    result = []

    for target_hms in milestones:
        # Find bracketing entries
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

        # Linear interpolation
        frac = (target_hms - below['hms']) / (above['hms'] - below['hms'])
        interp = {}
        for key in below:
            if isinstance(below[key], (int, float)):
                interp[key] = round(below[key] + frac * (above[key] - below[key]), 1)
            else:
                interp[key] = below[key] if frac < 0.5 else above[key]
        interp['hms'] = round(target_hms, 1)
        # Year should be integer
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
    print(f"PJM demand norm sum: {sum(demand_norm):.4f}")

    # Load data sources
    mac_queue = load_pjm_mac_queue()
    print(f"MAC queue: {len(mac_queue)} entries")

    ef_mixes = load_ef_mixes()
    print(f"EF mixes: {len(ef_mixes)} thresholds")

    # Run simulations
    results_1b = simulate_strategy_1b(mac_queue, demand_norm, supply_profiles, supply_matrix)
    results_2c = simulate_strategy_2c(ef_mixes, demand_norm, supply_profiles, supply_matrix)

    # Interpolate to HMS milestones
    milestones_1b = interpolate_to_hms_milestones(results_1b, HMS_MILESTONES)
    milestones_2c = interpolate_to_hms_milestones(results_2c, HMS_MILESTONES)

    print("\n=== HMS Milestone Summary ===")
    print(f"{'HMS':>6}  {'1B Year':>7} {'1B Gas GW':>9} {'1B Curt':>8}  {'2C Year':>7} {'2C Gas GW':>9} {'2C Curt':>8}")
    for m1, m2 in zip(milestones_1b, milestones_2c):
        print(f"{m1['hms']:5.0f}%  {m1['year']:>7} {m1['gasGw']:>8.1f} {m1['curtTwh']:>7.0f}  "
              f"{m2['year']:>7} {m2['gasGw']:>8.1f} {m2['curtTwh']:>7.0f}")

    # Write output
    output = {
        'iso': ISO,
        'baseDemandTwh': REGIONAL_DEMAND_TWH[ISO],
        'growthLevel': GROWTH_LEVEL,
        'milestones': HMS_MILESTONES,
        'strategy1B': {
            'name': 'Consequential (MAC Queue)',
            'description': 'Deploy cheapest abatement first — VRE-heavy, drives gas lock-in',
            'raw': results_1b,
            'milestones': milestones_1b,
        },
        'strategy2C': {
            'name': 'Hourly Matching (EF-Optimized)',
            'description': 'Physics-co-optimized portfolio — firm + storage + VRE balanced',
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
