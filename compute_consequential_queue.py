#!/usr/bin/env python3
"""
Step 5 Post-Processing: Consequential Deployment Queue Analysis
================================================================
Computes the optimal cross-regional deployment path under consequential
accounting — where capital flows to whichever grid offers the cheapest
marginal $/tCO₂ abated at each step.

Uses dispatch-stack retirement model: coal retires first (highest marginal
cost + highest emissions), then oil, then gas. The emission rate of
DISPLACED fossil generation — not the remaining fleet average — determines
the consequential CO₂ impact at each step.

Reads: dashboard/overprocure_scenarios.parquet, dashboard/overprocure_meta.json,
       data/egrid_emission_rates.json, data/eia_fossil_mix.json
Writes: data/consequential_queue.json, dashboard/js/consequential-queue-data.js
"""

import json
import os
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

# ========== PATHS ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCENARIOS_PATH = os.path.join(BASE_DIR, 'dashboard', 'overprocure_scenarios.parquet')
META_PATH = os.path.join(BASE_DIR, 'dashboard', 'overprocure_meta.json')
EGRID_PATH = os.path.join(BASE_DIR, 'data', 'egrid_emission_rates.json')
FOSSIL_MIX_PATH = os.path.join(BASE_DIR, 'data', 'eia_fossil_mix.json')
OUTPUT_JSON = os.path.join(BASE_DIR, 'data', 'consequential_queue.json')
OUTPUT_JS = os.path.join(BASE_DIR, 'dashboard', 'js', 'consequential-queue-data.js')

# ========== CONSTANTS ==========
ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO']
THRESHOLDS = [50, 60, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 100]
RESOURCES = ['clean_firm', 'solar', 'wind', 'ccs_ccgt', 'hydro']

# Medium scenario keys (all-Medium cost assumptions, 45Q on)
MEDIUM_KEYS = {
    'CAISO': 'MMMM_M_M_M1_L',
    'ERCOT': 'MMMM_M_M_M1_X',
    'PJM': 'MMMM_M_M_M1_X',
    'NYISO': 'MMMM_M_M_M1_X',
    'NEISO': 'MMMM_M_M_M1_X',
}

# Marginal MAC zones (matching shared-data.js zone definitions)
ZONES = [
    {'label': '50→75%', 'start_thresh': 50, 'end_thresh': 75, 'start_idx': 0, 'end_idx': 3},
    {'label': '75→90%', 'start_thresh': 75, 'end_thresh': 90, 'start_idx': 3, 'end_idx': 7},
    {'label': '90→92.5%', 'start_thresh': 90, 'end_thresh': 92.5, 'start_idx': 7, 'end_idx': 8},
    {'label': '92.5→95%', 'start_thresh': 92.5, 'end_thresh': 95, 'start_idx': 8, 'end_idx': 9},
    {'label': '95→97.5%', 'start_thresh': 95, 'end_thresh': 97.5, 'start_idx': 9, 'end_idx': 10},
    {'label': '97.5→99%', 'start_thresh': 97.5, 'end_thresh': 99, 'start_idx': 10, 'end_idx': 11},
]

# Demand growth rates by ISO (medium scenario, %/yr)
GROWTH_RATES = {
    'CAISO': 1.8, 'ERCOT': 3.5, 'PJM': 2.4, 'NYISO': 1.2, 'NEISO': 1.0,
}

# SBTi milestone year mapping
SBTI_YEAR_MAP = {
    50: 2025, 60: 2027, 70: 2029, 75: 2030, 80: 2032,
    85: 2035, 87.5: 2036, 90: 2040, 92.5: 2042,
    95: 2045, 97.5: 2047, 99: 2049, 100: 2050,
}

LBS_PER_TON = 2204.62


def load_data():
    """Load all input data."""
    print("Loading optimizer scenarios...")
    df = pq.read_table(SCENARIOS_PATH).to_pandas()
    print(f"  {len(df):,} scenario rows loaded")

    with open(META_PATH) as f:
        meta = json.load(f)

    with open(EGRID_PATH) as f:
        egrid = json.load(f)

    with open(FOSSIL_MIX_PATH) as f:
        fossil_mix = json.load(f)

    return df, meta, egrid, fossil_mix


def build_dispatch_stack(egrid, fossil_mix):
    """
    Build the fossil dispatch stack for each ISO:
    - Fuel types ordered by retirement priority: coal first, then oil, then gas
    - Each fuel has: share of fossil generation, emission rate (tCO₂/MWh)
    - Coal retires first because highest marginal cost and highest emissions

    Returns: {iso: {'fuels': [{'type', 'share', 'emission_rate'}], 'total_fossil_share': float}}
    """
    stacks = {}
    for iso in ISOS:
        # Get fossil fuel shares (using latest available year)
        years = sorted(fossil_mix[iso].keys())
        latest_yr = years[-1]
        yr_data = fossil_mix[iso][latest_yr]

        # Average annual shares
        coal_share = np.mean(yr_data['coal_share'])
        gas_share = np.mean(yr_data['gas_share'])
        oil_share = np.mean(yr_data['oil_share'])

        # Emission rates per fuel (tCO₂/MWh) from eGRID
        rates = egrid[iso]
        coal_rate = rates['coal_co2_lb_per_mwh'] / LBS_PER_TON
        gas_rate = rates['gas_co2_lb_per_mwh'] / LBS_PER_TON
        oil_rate = rates['oil_co2_lb_per_mwh'] / LBS_PER_TON

        # Build stack in retirement order: coal → oil → gas
        fuels = []
        if coal_share > 0.001:
            fuels.append({'type': 'coal', 'share': round(coal_share, 4), 'emission_rate': round(coal_rate, 4)})
        if oil_share > 0.001:
            fuels.append({'type': 'oil', 'share': round(oil_share, 4), 'emission_rate': round(oil_rate, 4)})
        if gas_share > 0.001:
            fuels.append({'type': 'gas', 'share': round(gas_share, 4), 'emission_rate': round(gas_rate, 4)})

        stacks[iso] = {
            'fuels': fuels,
            'fuel_year': latest_yr,
        }

    return stacks


def compute_displaced_emission_rate(dispatch_stack, fossil_twh_start, fossil_twh_end, demand_twh):
    """
    Compute the weighted-average emission rate of the fossil generation
    displaced between two thresholds, using merit-order retirement.

    The fossil that retires first (coal) has the highest emission rate.
    Once coal is fully retired, subsequent displacement comes from oil/gas.

    Args:
        dispatch_stack: {'fuels': [{'type', 'share', 'emission_rate'}]}
        fossil_twh_start: Total fossil TWh at the starting threshold
        fossil_twh_end: Total fossil TWh at the ending threshold
        demand_twh: Annual demand in TWh

    Returns: (weighted_avg_rate, displacement_breakdown)
    """
    displaced_twh = fossil_twh_start - fossil_twh_end
    if displaced_twh <= 0.001:
        return 0.0, {}

    fuels = dispatch_stack['fuels']

    # Compute TWh for each fuel at the starting level
    # The stack is ordered by retirement priority (coal first)
    # At any fossil level, the fuel mix follows the observed shares
    # When fossil shrinks, coal retires first from the "top" of the stack

    # Total fuel TWh at start (using observed shares)
    fuel_twh_start = []
    for f in fuels:
        fuel_twh_start.append(f['share'] * fossil_twh_start)

    # Walk through displacement: retire coal first, then oil, then gas
    remaining_to_displace = displaced_twh
    displacement = {}
    total_co2 = 0

    for i, fuel in enumerate(fuels):
        fuel_available = fuel['share'] * fossil_twh_start

        # How much of this fuel was already retired by reaching fossil_twh_end?
        # We compute how much of each fuel remains at fossil_twh_end
        # Using merit-order: cheapest-to-retire fuel (coal) is removed first

        if remaining_to_displace <= 0:
            break

        displaced_from_this_fuel = min(fuel_available, remaining_to_displace)
        remaining_to_displace -= displaced_from_this_fuel

        co2_from_this = displaced_from_this_fuel * fuel['emission_rate'] * 1e6  # tons
        total_co2 += co2_from_this

        displacement[fuel['type']] = {
            'twh_displaced': round(displaced_from_this_fuel, 2),
            'emission_rate': fuel['emission_rate'],
            'co2_tons': round(co2_from_this, 0),
        }

    # Weighted average rate
    if displaced_twh > 0:
        weighted_rate = total_co2 / (displaced_twh * 1e6)
    else:
        weighted_rate = 0

    return weighted_rate, displacement


def extract_medium_scenarios(df):
    """Extract medium-cost scenario data for all ISOs and thresholds."""
    result = {}
    for iso in ISOS:
        med_key = MEDIUM_KEYS[iso]
        iso_df = df[(df['iso'] == iso) & (df['scenario'] == med_key)].copy()
        iso_df = iso_df.sort_values('threshold')

        result[iso] = {}
        for _, row in iso_df.iterrows():
            t = float(row['threshold'])
            demand_twh = row['annual_demand_mwh'] / 1e6
            proc = row['procurement_pct'] / 100

            res_twh = {}
            for res in RESOURCES:
                pct = row[f'mix_{res}'] / 100
                res_twh[res] = pct * proc * demand_twh

            bat_twh = row['battery_dispatch_pct'] / 100 * demand_twh
            ldes_twh = row['ldes_dispatch_pct'] / 100 * demand_twh

            result[iso][t] = {
                'demand_twh': demand_twh,
                'demand_mwh': row['annual_demand_mwh'],
                'procurement_pct': row['procurement_pct'],
                'match_score': row['hourly_match_score'],
                'eff_cost': row['cost_effective_cost'],
                'total_cost': row['cost_total_cost'],
                'incremental_cost': row['cost_incremental'],
                'wholesale': row['cost_wholesale'],
                'resource_twh': res_twh,
                'battery_twh': bat_twh,
                'ldes_twh': ldes_twh,
                'resource_pct': {res: float(row[f'mix_{res}']) for res in RESOURCES},
                'gas_backup_mw': float(row['gas_gas_backup_needed_mw']),
                'new_gas_mw': float(row['gas_new_gas_build_mw']),
                'gas_cost': float(row['gas_gas_cost_per_mwh']),
                'tranche_existing_twh': float(row['tranche_cf_existing_twh']),
                'tranche_uprate_twh': float(row['tranche_uprate_twh']),
                'tranche_geo_twh': float(row['tranche_geo_twh']),
                'tranche_nuclear_twh': float(row['tranche_nuclear_newbuild_twh']),
                'tranche_ccs_twh': float(row['tranche_ccs_tranche_twh']),
            }

    return result


def compute_zone_metrics(med_data, dispatch_stacks):
    """
    Compute metrics for each (ISO, zone) pair using dispatch-stack
    retirement emission rates.
    """
    zone_metrics = []

    for iso in ISOS:
        iso_data = med_data[iso]
        demand_twh = list(iso_data.values())[0]['demand_twh']
        demand_mwh = demand_twh * 1e6
        stack = dispatch_stacks[iso]

        for zone_idx, zone in enumerate(ZONES):
            t_start = float(zone['start_thresh'])
            t_end = float(zone['end_thresh'])

            if t_start not in iso_data or t_end not in iso_data:
                continue

            start = iso_data[t_start]
            end = iso_data[t_end]

            # Fossil TWh at each threshold
            fossil_twh_start = demand_twh * (1 - start['match_score'] / 100)
            fossil_twh_end = demand_twh * (1 - end['match_score'] / 100)
            delta_clean_twh = fossil_twh_start - fossil_twh_end

            # Dispatch-stack displacement: compute emission rate of DISPLACED fossil
            displaced_rate, displacement_breakdown = compute_displaced_emission_rate(
                stack, fossil_twh_start, fossil_twh_end, demand_twh
            )

            # CO₂ displaced using marginal (displaced) emission rate
            co2_displaced_mt = delta_clean_twh * displaced_rate  # TWh × tCO₂/MWh = MT

            # Cost change (system cost per MWh of demand)
            cost_start = start['eff_cost'] * start['procurement_pct'] / 100 + start['gas_cost']
            cost_end = end['eff_cost'] * end['procurement_pct'] / 100 + end['gas_cost']
            delta_cost_per_mwh = cost_end - cost_start
            delta_cost_total_bn = delta_cost_per_mwh * demand_mwh / 1e9

            # Marginal MAC ($/tCO₂)
            if co2_displaced_mt > 0.001:
                marginal_mac = (delta_cost_per_mwh * demand_mwh) / (co2_displaced_mt * 1e6)
            else:
                marginal_mac = float('inf')

            marginal_mac_display = min(marginal_mac, 1500)

            # Resource changes (delta TWh)
            delta_resources = {}
            for res in RESOURCES:
                delta_resources[res] = end['resource_twh'][res] - start['resource_twh'][res]
            delta_resources['battery'] = end['battery_twh'] - start['battery_twh']
            delta_resources['ldes'] = end['ldes_twh'] - start['ldes_twh']

            delta_gas_mw = end['new_gas_mw'] - start['new_gas_mw']

            year_start = SBTI_YEAR_MAP.get(t_start, 2025)
            year_end = SBTI_YEAR_MAP.get(t_end, 2050)
            midpoint_year = (year_start + year_end) / 2
            growth_rate = GROWTH_RATES[iso] / 100
            growth_factor = (1 + growth_rate) ** (midpoint_year - 2025)

            # Determine primary fuel displaced
            primary_displaced = 'gas'
            for fuel_info in displacement_breakdown.values():
                pass
            if displacement_breakdown:
                sorted_disp = sorted(displacement_breakdown.items(),
                                     key=lambda x: x[1]['twh_displaced'], reverse=True)
                primary_displaced = sorted_disp[0][0]

            zone_metrics.append({
                'iso': iso,
                'zone_idx': zone_idx,
                'zone_label': zone['label'],
                'threshold_start': t_start,
                'threshold_end': t_end,
                'year_start': year_start,
                'year_end': year_end,
                'marginal_mac': round(marginal_mac, 1),
                'marginal_mac_display': round(marginal_mac_display, 1),
                'co2_displaced_mt': round(co2_displaced_mt, 2),
                'displaced_emission_rate': round(displaced_rate, 4),
                'displacement_breakdown': displacement_breakdown,
                'primary_fuel_displaced': primary_displaced,
                'fossil_twh_start': round(fossil_twh_start, 1),
                'fossil_twh_end': round(fossil_twh_end, 1),
                'delta_clean_twh': round(delta_clean_twh, 1),
                'delta_cost_per_mwh': round(delta_cost_per_mwh, 2),
                'delta_cost_total_bn': round(delta_cost_total_bn, 2),
                'delta_resources': {k: round(v, 1) for k, v in delta_resources.items()},
                'end_resource_twh': {k: round(v, 1) for k, v in end['resource_twh'].items()},
                'end_procurement_pct': end['procurement_pct'],
                'gas_backup_mw_end': end['gas_backup_mw'],
                'delta_gas_mw': round(delta_gas_mw, 0),
                'demand_twh': demand_twh,
                'growth_factor': round(growth_factor, 3),
                'growth_adjusted_demand_twh': round(demand_twh * growth_factor, 1),
                'growth_adjusted_co2_mt': round(co2_displaced_mt * growth_factor, 2),
            })

    # Sort by marginal MAC (cheapest first) — the consequential deployment queue
    zone_metrics.sort(key=lambda x: (x['marginal_mac'], -x['co2_displaced_mt']))

    for i, step in enumerate(zone_metrics):
        step['queue_position'] = i + 1

    return zone_metrics


def compute_stranding_analysis(med_data):
    """
    For each ISO and resource, find peak TWh, final TWh, and stranding ratio.
    """
    stranding = {}
    for iso in ISOS:
        iso_data = med_data[iso]
        stranding[iso] = {}

        all_resources = RESOURCES + ['battery', 'ldes']
        for res in all_resources:
            peak_twh = 0
            peak_thresh = 50
            values = {}

            for t in THRESHOLDS:
                t_float = float(t)
                if t_float not in iso_data:
                    continue
                if res in ['battery', 'ldes']:
                    twh = iso_data[t_float].get(f'{res}_twh', 0)
                else:
                    twh = iso_data[t_float]['resource_twh'].get(res, 0)
                values[t] = round(twh, 1)
                if twh > peak_twh:
                    peak_twh = twh
                    peak_thresh = t

            final_thresh = 99 if 99.0 in iso_data else max(iso_data.keys())
            if res in ['battery', 'ldes']:
                final_twh = iso_data[final_thresh].get(f'{res}_twh', 0)
            else:
                final_twh = iso_data[final_thresh]['resource_twh'].get(res, 0)

            stranding_ratio = peak_twh / final_twh if final_twh > 0.1 else (999 if peak_twh > 0.1 else 0)

            stranding[iso][res] = {
                'peak_twh': round(peak_twh, 1),
                'peak_threshold': peak_thresh,
                'final_twh': round(final_twh, 1),
                'final_threshold': final_thresh,
                'stranding_ratio': round(stranding_ratio, 2),
                'stranded_twh': round(max(0, peak_twh - final_twh), 1),
                'values_by_threshold': values,
            }

    return stranding


def compute_cumulative_deployment(queue):
    """Follow the deployment queue and track cumulative metrics."""
    cumulative = []
    running_co2 = 0
    running_cost = 0
    running_twh = {}
    iso_progress = {iso: 50 for iso in ISOS}

    for step in queue:
        iso = step['iso']
        running_co2 += step['co2_displaced_mt']
        running_cost += step['delta_cost_total_bn']

        for res, delta in step['delta_resources'].items():
            running_twh[res] = running_twh.get(res, 0) + delta

        iso_progress[iso] = step['threshold_end']

        cumulative.append({
            'queue_position': step['queue_position'],
            'iso': iso,
            'zone_label': step['zone_label'],
            'marginal_mac': step['marginal_mac'],
            'cumulative_co2_mt': round(running_co2, 2),
            'cumulative_cost_bn': round(running_cost, 2),
            'cumulative_twh': {k: round(v, 1) for k, v in running_twh.items()},
            'iso_thresholds': dict(iso_progress),
        })

    return cumulative


def compute_resource_trajectories(med_data):
    """Full resource trajectory at each threshold for each ISO."""
    trajectories = {}
    for iso in ISOS:
        iso_data = med_data[iso]
        iso_traj = []

        for t in THRESHOLDS:
            t_float = float(t)
            if t_float not in iso_data:
                continue
            d = iso_data[t_float]
            row = {
                'threshold': t,
                'procurement_pct': d['procurement_pct'],
                'eff_cost': d['eff_cost'],
                'match_score': d['match_score'],
            }
            for res in RESOURCES:
                row[f'{res}_twh'] = round(d['resource_twh'][res], 1)
            row['battery_twh'] = round(d['battery_twh'], 1)
            row['ldes_twh'] = round(d['ldes_twh'], 1)
            row['gas_backup_mw'] = d['gas_backup_mw']
            row['new_gas_mw'] = d['new_gas_mw']
            iso_traj.append(row)

        trajectories[iso] = iso_traj

    return trajectories


def compute_demand_growth(med_data):
    """Demand at SBTi milestone years for each ISO."""
    projections = {}
    for iso in ISOS:
        demand_twh = list(med_data[iso].values())[0]['demand_twh']
        growth_rate = GROWTH_RATES[iso] / 100
        iso_proj = {}
        for year in [2025, 2030, 2035, 2040, 2045, 2050]:
            factor = (1 + growth_rate) ** (year - 2025)
            iso_proj[year] = {
                'demand_twh': round(demand_twh * factor, 1),
                'growth_factor': round(factor, 3),
                'growth_twh': round(demand_twh * (factor - 1), 1),
                'counterfactual_co2_mt': round(demand_twh * (factor - 1) * 0.35, 1),
            }
        projections[iso] = iso_proj
    return projections


def print_summary(queue, stranding, dispatch_stacks):
    """Print human-readable results."""
    # Dispatch stacks
    print("\n" + "=" * 80)
    print("FOSSIL DISPATCH STACKS (retirement order: coal → oil → gas)")
    print("=" * 80)
    for iso in ISOS:
        stack = dispatch_stacks[iso]
        fuels_str = ", ".join(
            f"{f['type']}: {f['share']*100:.1f}% @ {f['emission_rate']:.3f} tCO₂/MWh"
            for f in stack['fuels']
        )
        print(f"  {iso} ({stack['fuel_year']}): {fuels_str}")

    # Deployment queue
    print("\n" + "=" * 100)
    print("CONSEQUENTIAL DEPLOYMENT QUEUE (sorted by marginal $/tCO₂, dispatch-stack emission rates)")
    print("=" * 100)
    print(f"{'#':>3} {'ISO':<7} {'Zone':<12} {'MAC $/t':>9} {'Disp Rate':>10} {'Fuel':>6} "
          f"{'CO₂ MT':>8} {'ΔCost $B':>10} {'Primary Resource Change':>35}")
    print("-" * 100)

    for step in queue:
        deltas = step['delta_resources']
        sorted_d = sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True)
        top = sorted_d[0]
        top_str = f"{top[0]}: {'+' if top[1]>0 else ''}{top[1]:.0f} TWh"
        if len(sorted_d) > 1 and abs(sorted_d[1][1]) > 1:
            r2 = sorted_d[1]
            top_str += f", {r2[0]}: {'+' if r2[1]>0 else ''}{r2[1]:.0f}"

        mac_str = f"${step['marginal_mac']:,.0f}" if step['marginal_mac'] < 9999 else "$∞"

        print(f"{step['queue_position']:>3} {step['iso']:<7} {step['zone_label']:<12} "
              f"{mac_str:>9} {step['displaced_emission_rate']:>9.3f} {step['primary_fuel_displaced']:>6} "
              f"{step['co2_displaced_mt']:>7.1f} {step['delta_cost_total_bn']:>9.2f} {top_str:>35}")

    total_co2 = sum(s['co2_displaced_mt'] for s in queue)
    total_cost = sum(s['delta_cost_total_bn'] for s in queue)
    cheap = [s for s in queue if s['marginal_mac'] < 100]
    mid = [s for s in queue if 100 <= s['marginal_mac'] < 500]
    exp = [s for s in queue if s['marginal_mac'] >= 500]

    print("-" * 100)
    print(f"TOTAL: {total_co2:.1f} MT CO₂, ${total_cost:.1f}B annual cost")
    print(f"  Cheap (<$100/t): {len(cheap)} steps, {sum(s['co2_displaced_mt'] for s in cheap):.1f} MT")
    print(f"  Moderate ($100-500/t): {len(mid)} steps, {sum(s['co2_displaced_mt'] for s in mid):.1f} MT")
    print(f"  Expensive (>$500/t): {len(exp)} steps, {sum(s['co2_displaced_mt'] for s in exp):.1f} MT")

    # Stranding
    print("\n" + "=" * 80)
    print("STRANDING ANALYSIS (Peak TWh vs. Final at 99%)")
    print("=" * 80)
    for iso in ISOS:
        print(f"\n{iso}:")
        for res in RESOURCES + ['battery', 'ldes']:
            s = stranding[iso][res]
            if s['peak_twh'] < 0.5:
                continue
            flag = " ⚠ STRANDING" if s['stranding_ratio'] > 1.5 else ""
            print(f"  {res:>12}: peak {s['peak_twh']:6.1f} TWh @ {s['peak_threshold']}% → "
                  f"final {s['final_twh']:6.1f} TWh @ {s['final_threshold']}% "
                  f"(ratio: {s['stranding_ratio']:.1f}x){flag}")


def write_outputs(queue, cumulative, stranding, trajectories, projections, dispatch_stacks):
    """Write JSON and JS output files."""
    # Convert dispatch stacks for serialization
    stacks_serial = {}
    for iso, stack in dispatch_stacks.items():
        stacks_serial[iso] = {
            'fuels': stack['fuels'],
            'fuel_year': stack['fuel_year'],
        }

    output = {
        'metadata': {
            'description': 'Consequential deployment queue with dispatch-stack retirement emission rates',
            'methodology': 'Coal retires first (highest marginal cost), then oil, then gas. '
                           'Emission rate is for DISPLACED fossil, not remaining fleet average.',
            'zones': [z['label'] for z in ZONES],
            'thresholds': THRESHOLDS,
            'isos': ISOS,
            'medium_scenario_keys': MEDIUM_KEYS,
            'growth_rates_pct': GROWTH_RATES,
            'sbti_year_map': {str(k): v for k, v in SBTI_YEAR_MAP.items()},
        },
        'dispatch_stacks': stacks_serial,
        'deployment_queue': queue,
        'cumulative_deployment': cumulative,
        'stranding_analysis': stranding,
        'resource_trajectories': trajectories,
        'demand_growth_projections': projections,
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nJSON: {OUTPUT_JSON} ({os.path.getsize(OUTPUT_JSON) / 1024:.0f} KB)")

    os.makedirs(os.path.dirname(OUTPUT_JS), exist_ok=True)
    with open(OUTPUT_JS, 'w') as f:
        f.write("// Auto-generated by compute_consequential_queue.py\n")
        f.write("// Dispatch-stack retirement: coal→oil→gas merit order\n\n")
        f.write(f"const CQ_DATA = {json.dumps(output, indent=2, default=str)};\n")
    print(f"JS:   {OUTPUT_JS} ({os.path.getsize(OUTPUT_JS) / 1024:.0f} KB)")


def main():
    print("=" * 60)
    print("CONSEQUENTIAL DEPLOYMENT QUEUE ANALYSIS")
    print("  (Dispatch-Stack Retirement Emission Rates)")
    print("=" * 60)

    df, meta, egrid, fossil_mix = load_data()

    # Build dispatch stacks with merit-order retirement
    dispatch_stacks = build_dispatch_stack(egrid, fossil_mix)

    # Extract medium scenarios
    med_data = extract_medium_scenarios(df)

    # Compute zone metrics with dispatch-stack emission rates
    queue = compute_zone_metrics(med_data, dispatch_stacks)

    # Cumulative deployment path
    cumulative = compute_cumulative_deployment(queue)

    # Stranding analysis
    stranding = compute_stranding_analysis(med_data)

    # Resource trajectories
    trajectories = compute_resource_trajectories(med_data)

    # Demand growth
    projections = compute_demand_growth(med_data)

    # Print results
    print_summary(queue, stranding, dispatch_stacks)

    # Write outputs
    write_outputs(queue, cumulative, stranding, trajectories, projections, dispatch_stacks)

    print("\nDone.")


if __name__ == '__main__':
    main()
