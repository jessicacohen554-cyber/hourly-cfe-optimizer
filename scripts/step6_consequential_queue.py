#!/usr/bin/env python3
"""
Step 6 Post-Processing: Consequential Deployment Queue Analysis
================================================================
Computes the optimal cross-regional deployment path under consequential
accounting — where capital flows to whichever grid offers the cheapest
marginal $/tCO₂ abated at each step.

Uses the dispatch cache (step5_build_dispatch_cache) for hourly emission accounting. Each mix's
8760-hour fossil_displaced array determines exact CO₂ displacement using
merit-order fuel retirement (coal → oil → gas).

The MARGINAL emission rate between two thresholds is computed from the
difference in dispatch-based CO₂ at each zone boundary — capturing the
hourly shape of displacement (wind-heavy displaces different hours than
clean-firm-heavy).

Reads: data/step4-gas-ccs-parquets/step4_*.parquet (or step3 fallback),
       data/egrid_emission_rates.json, data/eia_fossil_mix.json
       data/step5-post-processing/dispatch_cache/{ISO}_dispatch_cache.npz
Writes: data/step5-post-processing/consequential_queue.json,
        dashboard/js/consequential-queue-data.js
"""

import json
import os
import sys
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

# Add project root to path for dispatch_utils import
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'scripts'))

from step3_cost_optimization import OUTPUT_THRESHOLDS as THRESHOLDS

from dispatch_utils import (
    compute_fossil_retirement,
    compute_co2_from_dispatch,
    COAL_CAP_TWH, OIL_CAP_TWH, COAL_OIL_RETIREMENT_THRESHOLD,
    BASE_DEMAND_TWH, GRID_MIX_SHARES, CCS_RESIDUAL_EMISSION_RATE,
    H, RESOURCE_TYPES,
    load_common_data, get_supply_profiles, get_demand_profile,
    build_supply_matrix, reconstruct_hourly_dispatch,
    _archetype_key, load_dispatch_cache, get_or_compute_dispatch,
)

# ========== PATHS ==========
STEP4_PARQUET_DIR = os.path.join(BASE_DIR, 'data', 'step4-gas-ccs-parquets')
STEP3_PARQUET_DIR = os.path.join(BASE_DIR, 'data', 'step3-cost-opt-parquets')
META_PATH = os.path.join(BASE_DIR, 'data', 'step4-gas-ccs-parquets', 'step4_meta.json')
EGRID_PATH = os.path.join(BASE_DIR, 'data', 'egrid_emission_rates.json')
FOSSIL_MIX_PATH = os.path.join(BASE_DIR, 'data', 'EIA 930 Data', 'eia_fossil_mix.json')  # legacy fallback
OUTPUT_JSON = os.path.join(BASE_DIR, 'data', 'step5-post-processing', 'consequential_queue.json')
OUTPUT_JS = os.path.join(BASE_DIR, 'dashboard', 'js', 'consequential-queue-data.js')

# ========== CONSTANTS ==========
ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']
RESOURCES = ['clean_firm', 'solar', 'wind', 'ccs_ccgt', 'hydro']

MEDIUM_KEYS = {
    'CAISO': 'MMMM_M_M_M1_L',
    'ERCOT': 'MMMM_M_M_M1_X',
    'PJM': 'MMMM_M_M_M1_X',
    'NYISO': 'MMMM_M_M_M1_X',
    'NEISO': 'MMMM_M_M_M1_X',
    'MISO': 'MMMM_M_M_M1_X',
    'SPP': 'MMMM_M_M_M1_X',
}

ZONES = [
    {'label': '50→75%', 'start_thresh': 50, 'end_thresh': 75},
    {'label': '75→90%', 'start_thresh': 75, 'end_thresh': 90},
    {'label': '90→92.5%', 'start_thresh': 90, 'end_thresh': 92.5},
    {'label': '92.5→95%', 'start_thresh': 92.5, 'end_thresh': 95},
    {'label': '95→97.5%', 'start_thresh': 95, 'end_thresh': 97.5},
    {'label': '97.5→99%', 'start_thresh': 97.5, 'end_thresh': 99},
    {'label': '99→99.5%', 'start_thresh': 99, 'end_thresh': 99.5},
    {'label': '99.5→99.9%', 'start_thresh': 99.5, 'end_thresh': 99.9},
    {'label': '99.9→100%', 'start_thresh': 99.9, 'end_thresh': 100},
]

GROWTH_RATES = {
    'CAISO': 1.8, 'ERCOT': 3.5, 'PJM': 2.4, 'NYISO': 1.2, 'NEISO': 1.0,
    'MISO': 2.2, 'SPP': 1.8,
}

# Pre-compute threshold-to-string mapping (avoids repeated str() calls in loops)
THRESHOLD_STRS = {t: str(int(t)) if t == int(t) else str(t) for t in THRESHOLDS}

# ========== MEMOIZATION ==========
_fossil_retirement_cache = {}


def cached_fossil_retirement(iso, threshold_pct, emission_rates, fossil_mix,
                              demand_growth_factor=1.0):
    """Memoized wrapper for compute_fossil_retirement (fallback when dispatch cache unavailable)."""
    cache_key = (iso, threshold_pct, demand_growth_factor)
    if cache_key not in _fossil_retirement_cache:
        _fossil_retirement_cache[cache_key] = compute_fossil_retirement(
            iso, threshold_pct, emission_rates, fossil_mix,
            demand_growth_factor=demand_growth_factor)
    return _fossil_retirement_cache[cache_key]

# Import canonical threshold-year mapping from Step 3
try:
    from step3_cost_optimization import THRESHOLD_TARGET_YEARS
    SBTI_YEAR_MAP = THRESHOLD_TARGET_YEARS
except ImportError:
    SBTI_YEAR_MAP = {
        50: 2030, 55: 2031, 60: 2033, 65: 2034, 70: 2035, 75: 2036, 80: 2037,
        85: 2038, 87.5: 2039, 90: 2040, 92.5: 2043,
        95: 2045, 97.5: 2048, 99: 2049, 99.5: 2049, 99.9: 2050, 100: 2050,
    }


# ========== DISPATCH CACHE CO₂ ACCOUNTING ==========
# Per-ISO dispatch cache + CO₂ results, populated in main()
_dispatch_caches = {}       # iso → {archetype_key: dispatch_result}
_co2_by_threshold = {}      # (iso, threshold) → co2_result dict


def get_dispatch_co2(iso, threshold, med_data, egrid, demand_data, gen_profiles):
    """Get CO₂ displacement for a medium-scenario mix using dispatch cache.

    Looks up the mix's dispatch in the per-ISO cache. If not cached, computes
    live dispatch and adds to cache. Returns the CO₂ result dict from
    compute_co2_from_dispatch().
    """
    cache_key = (iso, threshold)
    if cache_key in _co2_by_threshold:
        return _co2_by_threshold[cache_key]

    iso_data = med_data.get(iso, {})
    mix_info = iso_data.get(threshold, {})
    if not mix_info:
        return None

    resource_pcts = mix_info['resource_pct']
    bat_pct = mix_info.get('battery_dispatch_pct', 0)
    bat8_pct = mix_info.get('battery8_dispatch_pct', 0)
    ldes_pct = mix_info.get('ldes_dispatch_pct', 0)
    h2_pct = mix_info.get('h2_dispatch_pct', 0)
    demand_mwh = mix_info['demand_mwh']

    # Try dispatch cache first
    dispatch_cache = _dispatch_caches.get(iso, {})
    supply_profiles = get_supply_profiles(iso, gen_profiles)
    demand_norm, _ = get_demand_profile(iso, demand_data)

    dispatch_result, cache_hit = get_or_compute_dispatch(
        iso, demand_norm, supply_profiles, resource_pcts,
        100, bat_pct, bat8_pct, ldes_pct,
        cache=dispatch_cache)

    if not cache_hit:
        _dispatch_caches[iso] = dispatch_cache

    co2_result = compute_co2_from_dispatch(iso, dispatch_result, egrid, demand_mwh)
    _co2_by_threshold[cache_key] = co2_result
    return co2_result


def compute_marginal_displaced_rate_dispatch(iso, threshold_start, threshold_end,
                                              egrid, med_data, demand_data, gen_profiles):
    """Compute marginal emission rate between two thresholds using dispatch cache.

    Gets hourly-dispatch-based CO₂ at both thresholds, then:
      marginal_rate = (CO₂_end - CO₂_start) / (clean_TWh_end - clean_TWh_start)
    """
    co2_start = get_dispatch_co2(iso, threshold_start, med_data, egrid, demand_data, gen_profiles)
    co2_end = get_dispatch_co2(iso, threshold_end, med_data, egrid, demand_data, gen_profiles)

    if not co2_start or not co2_end:
        return 0, 0, {}

    delta_co2 = co2_end['total_co2_abated_tons'] - co2_start['total_co2_abated_tons']
    delta_co2_mt = delta_co2 / 1e6  # Convert tons → MT

    delta_displaced_twh = co2_end['displaced_twh'] - co2_start['displaced_twh']

    if delta_displaced_twh > 0.01:
        marginal_rate = delta_co2_mt / delta_displaced_twh
    else:
        marginal_rate = co2_end['weighted_emission_rate']

    # Marginal fuel displacement
    marginal_coal = co2_end['coal_displaced_twh'] - co2_start['coal_displaced_twh']
    marginal_oil = co2_end['oil_displaced_twh'] - co2_start['oil_displaced_twh']
    marginal_gas = co2_end['gas_displaced_twh'] - co2_start['gas_displaced_twh']

    fuels = {'coal': marginal_coal, 'oil': marginal_oil, 'gas': marginal_gas}
    primary_fuel = max(fuels, key=lambda k: fuels[k]) if any(v > 0.01 for v in fuels.values()) else 'gas'

    return marginal_rate, delta_co2_mt, {
        'marginal_coal_twh': round(marginal_coal, 2),
        'marginal_oil_twh': round(marginal_oil, 2),
        'marginal_gas_twh': round(marginal_gas, 2),
        'cumulative_coal_displaced_twh': round(co2_end['coal_displaced_twh'], 2),
        'cumulative_oil_displaced_twh': round(co2_end['oil_displaced_twh'], 2),
        'cumulative_gas_displaced_twh': round(co2_end['gas_displaced_twh'], 2),
        'forced_gas_only': threshold_end >= COAL_OIL_RETIREMENT_THRESHOLD,
        'primary_fuel': primary_fuel,
        'avg_rate_at_start': round(co2_start['weighted_emission_rate'], 4),
        'avg_rate_at_end': round(co2_end['weighted_emission_rate'], 4),
    }


def load_data():
    """Load all input data from Step 4 per-ISO parquets (Step 3 fallback)."""
    print("Loading optimizer scenarios from per-ISO parquets...")

    input_dir = STEP4_PARQUET_DIR if os.path.isdir(STEP4_PARQUET_DIR) else STEP3_PARQUET_DIR
    tables = []
    for iso in ISOS:
        for prefix in ['step4_', 'step3_co_']:
            path = os.path.join(input_dir, f'{prefix}{iso}.parquet')
            if os.path.exists(path):
                t = pq.read_table(path)
                tables.append(t)
                print(f"  Loaded {os.path.basename(path)}: {t.num_rows:,} rows")
                break
        else:
            # Try alternate directory
            alt_dir = STEP3_PARQUET_DIR if input_dir == STEP4_PARQUET_DIR else STEP4_PARQUET_DIR
            for prefix in ['step4_', 'step3_co_']:
                path = os.path.join(alt_dir, f'{prefix}{iso}.parquet')
                if os.path.exists(path):
                    t = pq.read_table(path)
                    tables.append(t)
                    print(f"  Loaded {os.path.basename(path)}: {t.num_rows:,} rows (fallback)")
                    break
            else:
                print(f"  WARNING: No parquet found for {iso} — skipping")

    if tables:
        dfs = [t.to_pandas() for t in tables]
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pq.read_table(
            os.path.join(BASE_DIR, 'dashboard', 'overprocure_scenarios.parquet')).to_pandas()
    print(f"  {len(df):,} total scenario rows loaded")

    meta = {}
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)

    with open(EGRID_PATH) as f:
        egrid = json.load(f)
    from eia_data_io import load_fossil_mix
    fossil_mix = load_fossil_mix()

    return df, meta, egrid, fossil_mix


def extract_medium_scenarios(df):
    """Extract medium-cost scenario data for all ISOs and thresholds.

    Also extracts battery/ldes dispatch percentages needed for dispatch cache lookup.
    """
    result = {}
    for iso in ISOS:
        med_key = MEDIUM_KEYS[iso]
        iso_df = df[(df['iso'] == iso) & (df['scenario'] == med_key)].copy()
        iso_df = iso_df.sort_values('threshold')

        result[iso] = {}
        for row in iso_df.itertuples(index=False):
            t = float(row.threshold)
            demand_twh = row.annual_demand_mwh / 1e6

            res_twh = {}
            for res in RESOURCES:
                pct = getattr(row, f'mix_{res}') / 100
                res_twh[res] = pct * demand_twh

            result[iso][t] = {
                'demand_twh': demand_twh,
                'demand_mwh': row.annual_demand_mwh,
                'match_score': row.hourly_match_score,
                'eff_cost': row.cost_effective_cost,
                'total_cost': row.cost_total_cost,
                'incremental_cost': row.cost_incremental,
                'wholesale': row.cost_wholesale,
                'resource_twh': res_twh,
                'battery_twh': row.battery_dispatch_pct / 100 * demand_twh,
                'ldes_twh': row.ldes_dispatch_pct / 100 * demand_twh,
                'resource_pct': {res: float(getattr(row, f'mix_{res}')) for res in RESOURCES},
                # Dispatch cache lookup parameters
                'battery_dispatch_pct': float(row.battery_dispatch_pct),
                'battery8_dispatch_pct': float(getattr(row, 'battery8_dispatch_pct', 0)),
                'ldes_dispatch_pct': float(row.ldes_dispatch_pct),
                'h2_dispatch_pct': float(getattr(row, 'h2_dispatch_pct', 0)),
                'gas_backup_mw': float(row.ra_gas_backup_needed_mw),
                'new_gas_mw': float(row.ra_new_gas_build_mw),
                'gas_cost': float(row.ra_gas_backup_cost_per_mwh),
                'tranche_existing_twh': float(row.tranche_cf_existing_twh),
                'tranche_uprate_twh': float(row.tranche_uprate_twh),
                'tranche_geo_twh': float(row.tranche_geo_twh),
                'tranche_nuclear_twh': float(row.tranche_nuclear_newbuild_twh),
                'tranche_ccs_twh': float(row.tranche_ccs_tranche_twh),
            }

    return result


def compute_zone_metrics(med_data, egrid, fossil_mix, demand_data, gen_profiles):
    """Compute metrics for each (ISO, zone) pair using dispatch-cache-based emission accounting.

    Uses hourly dispatch results from the dispatch cache to compute exact CO₂
    displacement, capturing the shape-dependent differences between wind-heavy
    and clean-firm-heavy portfolios.
    """
    zone_metrics = []

    for iso in ISOS:
        iso_data = med_data[iso]
        demand_twh = list(iso_data.values())[0]['demand_twh']
        demand_mwh = demand_twh * 1e6

        for zone_idx, zone in enumerate(ZONES):
            t_start = float(zone['start_thresh'])
            t_end = float(zone['end_thresh'])

            if t_start not in iso_data or t_end not in iso_data:
                continue

            start = iso_data[t_start]
            end = iso_data[t_end]

            # Compute marginal displaced emission rate using dispatch cache
            marginal_rate, delta_co2_mt, displacement = compute_marginal_displaced_rate_dispatch(
                iso, t_start, t_end, egrid, med_data, demand_data, gen_profiles
            )

            # Delta clean TWh (from optimizer match scores)
            delta_match_pct = (end['match_score'] - start['match_score']) / 100
            delta_clean_twh = delta_match_pct * demand_twh

            # CO₂ displaced from dispatch model
            co2_displaced_mt = delta_co2_mt

            # Clean procurement cost only (no gas backup — gas is a system cost, not abatement cost)
            cost_start = start['eff_cost']
            cost_end = end['eff_cost']
            delta_cost_per_mwh = cost_end - cost_start
            delta_cost_total_bn = delta_cost_per_mwh * demand_mwh / 1e9
            # Gas backup cost tracked separately for educational overlay
            delta_gas_cost_per_mwh = end['gas_cost'] - start['gas_cost']

            # Marginal MAC ($/tCO₂)
            if co2_displaced_mt > 0.001:
                marginal_mac = (delta_cost_per_mwh * demand_mwh) / (co2_displaced_mt * 1e6)
            else:
                marginal_mac = float('inf')

            # Resource changes
            delta_resources = {}
            for res in RESOURCES:
                delta_resources[res] = end['resource_twh'][res] - start['resource_twh'][res]
            delta_resources['battery'] = end['battery_twh'] - start['battery_twh']
            delta_resources['ldes'] = end['ldes_twh'] - start['ldes_twh']

            year_start = SBTI_YEAR_MAP.get(t_start, 2025)
            year_end = SBTI_YEAR_MAP.get(t_end, 2050)
            midpoint_year = (year_start + year_end) / 2
            growth_factor = (1 + GROWTH_RATES[iso] / 100) ** (midpoint_year - 2025)

            zone_metrics.append({
                'iso': iso,
                'zone_idx': zone_idx,
                'zone_label': zone['label'],
                'threshold_start': t_start,
                'threshold_end': t_end,
                'year_start': year_start,
                'year_end': year_end,
                'marginal_mac': round(marginal_mac, 1),
                'marginal_mac_display': round(min(marginal_mac, 1500), 1),
                'co2_displaced_mt': round(co2_displaced_mt, 2),
                'displaced_emission_rate': round(marginal_rate, 4),
                'displacement_detail': displacement,
                'primary_fuel_displaced': displacement.get('primary_fuel', 'gas'),
                'fossil_twh_start': round(demand_twh * (1 - start['match_score'] / 100), 1),
                'fossil_twh_end': round(demand_twh * (1 - end['match_score'] / 100), 1),
                'delta_clean_twh': round(delta_clean_twh, 1),
                'delta_cost_per_mwh': round(delta_cost_per_mwh, 2),
                'delta_cost_total_bn': round(delta_cost_total_bn, 2),
                'delta_resources': {k: round(v, 1) for k, v in delta_resources.items()},
                'end_resource_twh': {k: round(v, 1) for k, v in end['resource_twh'].items()},
                'gas_backup_mw_end': end['gas_backup_mw'],
                'gas_cost_per_mwh_end': round(end['gas_cost'], 2),
                'delta_gas_cost_per_mwh': round(delta_gas_cost_per_mwh, 2),
                'delta_gas_mw': round(end['new_gas_mw'] - start['new_gas_mw'], 0),
                'demand_twh': demand_twh,
                'growth_factor': round(growth_factor, 3),
                'growth_adjusted_demand_twh': round(demand_twh * growth_factor, 1),
                'growth_adjusted_co2_mt': round(co2_displaced_mt * growth_factor, 2),
            })

    # Sort by marginal MAC — the consequential deployment queue
    zone_metrics.sort(key=lambda x: (x['marginal_mac'], -x['co2_displaced_mt']))
    for i, step in enumerate(zone_metrics):
        step['queue_position'] = i + 1

    return zone_metrics


def compute_stranding_analysis(med_data):
    """For each ISO and resource, find peak TWh, final TWh, and stranding ratio."""
    stranding = {}
    for iso in ISOS:
        iso_data = med_data[iso]
        stranding[iso] = {}

        for res in RESOURCES + ['battery', 'ldes']:
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
    """Follow the queue and track cumulative metrics."""
    cumulative = []
    running_co2 = 0
    running_cost = 0
    running_twh = {}
    iso_progress = {iso: 50 for iso in ISOS}

    for step in queue:
        running_co2 += step['co2_displaced_mt']
        running_cost += step['delta_cost_total_bn']
        for res, delta in step['delta_resources'].items():
            running_twh[res] = running_twh.get(res, 0) + delta
        iso_progress[step['iso']] = step['threshold_end']

        cumulative.append({
            'queue_position': step['queue_position'],
            'iso': step['iso'],
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
    """Demand at SBTi milestone years."""
    projections = {}
    for iso in ISOS:
        demand_twh = list(med_data[iso].values())[0]['demand_twh']
        rate = GROWTH_RATES[iso] / 100
        iso_proj = {}
        for year in [2025, 2030, 2035, 2040, 2045, 2050]:
            factor = (1 + rate) ** (year - 2025)
            iso_proj[year] = {
                'demand_twh': round(demand_twh * factor, 1),
                'growth_factor': round(factor, 3),
                'growth_twh': round(demand_twh * (factor - 1), 1),
                'counterfactual_co2_mt': round(demand_twh * (factor - 1) * 0.35, 1),
            }
        projections[iso] = iso_proj
    return projections


def compute_emission_rate_trajectory(egrid, med_data, demand_data, gen_profiles):
    """Compute displaced emission rate at every threshold using dispatch cache."""
    rate_traj = {}
    for iso in ISOS:
        rates = []
        for t in THRESHOLDS:
            co2 = get_dispatch_co2(iso, t, med_data, egrid, demand_data, gen_profiles)
            if co2:
                rates.append({
                    'threshold': t,
                    'displaced_rate': co2['weighted_emission_rate'],
                    'coal_displaced_twh': co2['coal_displaced_twh'],
                    'oil_displaced_twh': co2['oil_displaced_twh'],
                    'gas_displaced_twh': co2['gas_displaced_twh'],
                    'forced_gas_only': t >= COAL_OIL_RETIREMENT_THRESHOLD,
                    'methodology': co2['methodology'],
                })
            else:
                # Fallback to analytical model
                rate, info = cached_fossil_retirement(iso, t, egrid, {})
                rates.append({
                    'threshold': t,
                    'displaced_rate': round(rate, 4),
                    'coal_displaced_twh': info['coal_displaced_twh'],
                    'oil_displaced_twh': info['oil_displaced_twh'],
                    'gas_displaced_twh': info['gas_displaced_twh'],
                    'forced_gas_only': info.get('forced_gas_only', False),
                    'methodology': 'analytical_fallback',
                })
        rate_traj[iso] = rates
    return rate_traj


def compute_sequencing_analysis(med_data, stranding, queue):
    """
    Analyze the cheap-first sequencing problem:
    - What gets built in the 50→75% zone when chasing cheapest $/tCO₂?
    - How much of it is stranded at deeper decarbonization (>90%)?
    - Does delaying clean firm cause gas capacity lock-in?
    - Compare gas trajectories of wind-heavy vs. clean-firm-heavy grids.

    Gracefully handles ISOs with incomplete threshold coverage — uses
    the highest available threshold as the "deep decarbonization" reference
    instead of requiring 99%.
    """
    analysis = {}

    def _safe_res_twh(d, res, default=0):
        """Safely extract resource TWh from a threshold dict (may be empty)."""
        if not d:
            return default
        if res in ('battery', 'ldes'):
            return d.get(f'{res}_twh', default)
        return d.get('resource_twh', {}).get(res, default)

    def _safe_get(d, key, default=0):
        """Safely get a key from a threshold dict (may be empty)."""
        if not d:
            return default
        return d.get(key, default)

    for iso in ISOS:
        iso_data = med_data[iso]
        if not iso_data:
            continue
        demand_twh = list(iso_data.values())[0]['demand_twh']

        # What gets built in 50→75% zone
        d50 = iso_data.get(50.0, {})
        d75 = iso_data.get(75.0, {})
        d90 = iso_data.get(90.0, {})
        d95 = iso_data.get(95.0, {})
        d99 = iso_data.get(99.0, {})

        # Find highest available threshold for "deep decarb" reference
        available_thresholds = sorted(iso_data.keys())
        t_max = available_thresholds[-1] if available_thresholds else 0
        d_deep = iso_data.get(t_max, {})  # best available deep-decarb data

        if not d50 or not d75:
            continue

        # Resource deltas in cheap zone
        cheap_zone_build = {}
        for res in RESOURCES:
            cheap_zone_build[res] = _safe_res_twh(d75, res) - _safe_res_twh(d50, res)
        cheap_zone_build['battery'] = _safe_get(d75, 'battery_twh') - _safe_get(d50, 'battery_twh')
        cheap_zone_build['ldes'] = _safe_get(d75, 'ldes_twh') - _safe_get(d50, 'ldes_twh')

        # Dominant resource in cheap zone
        dominant = max(cheap_zone_build.items(), key=lambda x: abs(x[1]))
        strategy = 'clean_firm_first' if dominant[0] == 'clean_firm' and dominant[1] > 10 else \
                   'wind_first' if cheap_zone_build.get('wind', 0) > 10 else \
                   'solar_first' if cheap_zone_build.get('solar', 0) > 10 else 'mixed'

        # Stranding: how much of what was built in 50→75% is NOT needed at deep decarb?
        # Use d_deep (highest available threshold) instead of requiring 99%
        stranding_from_cheap_zone = {}
        for res in ['solar', 'wind']:
            built_in_cheap = max(0, cheap_zone_build.get(res, 0))
            at_75 = _safe_res_twh(d75, res)
            at_deep = _safe_res_twh(d_deep, res)
            stranded = max(0, at_75 - at_deep)
            stranding_from_cheap_zone[res] = {
                'built_in_cheap_zone_twh': round(built_in_cheap, 1),
                'at_75pct_twh': round(at_75, 1),
                'at_deep_twh': round(at_deep, 1),
                'deep_threshold': t_max,
                'stranded_twh': round(stranded, 1),
                'stranding_pct': round(stranded / at_75 * 100, 1) if at_75 > 0.1 else 0,
            }

        # Gas capacity trajectory
        gas_trajectory = []
        for t in THRESHOLDS:
            t_f = float(t)
            if t_f not in iso_data:
                continue
            d = iso_data[t_f]
            gas_trajectory.append({
                'threshold': t,
                'gas_backup_mw': round(d['gas_backup_mw']),
                'new_gas_mw': round(d['new_gas_mw']),
                'clean_firm_twh': round(d['resource_twh'].get('clean_firm', 0), 1),
                'wind_twh': round(d['resource_twh'].get('wind', 0), 1),
                'solar_twh': round(d['resource_twh'].get('solar', 0), 1),
            })

        gas_at_50 = _safe_get(d50, 'new_gas_mw')
        gas_at_75 = _safe_get(d75, 'new_gas_mw')
        gas_at_90 = _safe_get(d90, 'new_gas_mw')
        gas_at_95 = _safe_get(d95, 'new_gas_mw')
        gas_at_deep = _safe_get(d_deep, 'new_gas_mw')

        cf_at_75 = _safe_res_twh(d75, 'clean_firm')
        cf_at_deep = _safe_res_twh(d_deep, 'clean_firm')
        cf_deficit_twh = cf_at_deep - cf_at_75
        cf_deficit_gw = cf_deficit_twh / (8.76 * 0.90) if cf_deficit_twh != 0 else 0

        wind_built = max(0, cheap_zone_build.get('wind', 0))
        wind_gw = wind_built / (8.76 * 0.35) if wind_built > 0 else 0
        wind_gas_offset_gw = wind_gw * 0.10
        cf_equivalent_twh = wind_built
        cf_equivalent_gw = cf_equivalent_twh / (8.76 * 0.90) if cf_equivalent_twh > 0 else 0
        cf_gas_offset_gw = cf_equivalent_gw * 0.85

        gas_lock_in = {
            'gas_at_50_mw': round(gas_at_50),
            'gas_at_75_mw': round(gas_at_75),
            'gas_at_90_mw': round(gas_at_90),
            'gas_at_deep_mw': round(gas_at_deep),
            'deep_threshold': t_max,
            'gas_delta_50_to_75_mw': round(gas_at_75 - gas_at_50),
            'gas_delta_75_to_deep_mw': round(gas_at_deep - gas_at_75),
            'clean_firm_at_75_twh': round(cf_at_75, 1),
            'clean_firm_at_deep_twh': round(cf_at_deep, 1),
            'clean_firm_deficit_twh': round(cf_deficit_twh, 1),
            'clean_firm_deficit_gw': round(cf_deficit_gw, 1),
            'wind_built_cheap_zone_gw': round(wind_gw, 1),
            'cf_equivalent_gw': round(cf_equivalent_gw, 1),
            'forgone_gas_reduction_gw': round(cf_gas_offset_gw - wind_gas_offset_gw, 1),
        }

        cheap_step = next((s for s in queue if s['iso'] == iso and s['zone_label'] == '50→75%'), None)
        mac_naive = cheap_step['marginal_mac'] if cheap_step else 0

        stranded_wind = stranding_from_cheap_zone.get('wind', {}).get('stranded_twh', 0)
        stranded_solar = stranding_from_cheap_zone.get('solar', {}).get('stranded_twh', 0)
        stranded_cost_bn = (stranded_wind * 30 + stranded_solar * 25) * 25 / 1e3
        excess_gas_gw = max(0, gas_lock_in['forgone_gas_reduction_gw'])
        gas_lockin_cost_bn = excess_gas_gw * 1000 * 30 * 25 / 1e9

        analysis[iso] = {
            'strategy': strategy,
            'cheap_zone_build': {k: round(v, 1) for k, v in cheap_zone_build.items()},
            'dominant_resource': dominant[0],
            'dominant_twh': round(dominant[1], 1),
            'stranding_from_cheap_zone': stranding_from_cheap_zone,
            'gas_trajectory': gas_trajectory,
            'gas_lock_in': gas_lock_in,
            'mac_naive': round(mac_naive, 1),
            'stranded_cost_bn': round(stranded_cost_bn, 1),
            'gas_lockin_cost_bn': round(gas_lockin_cost_bn, 1),
            'total_hidden_cost_bn': round(stranded_cost_bn + gas_lockin_cost_bn, 1),
            'deep_threshold': t_max,
        }

    return analysis


def print_sequencing_analysis(seq_analysis):
    """Print the sequencing / stranding / gas lock-in analysis."""
    print("\n" + "=" * 120)
    print("SEQUENCING ANALYSIS: CHEAP-FIRST vs. DEEP DECARBONIZATION")
    print("What gets built in 50→75% (chasing cheapest $/tCO₂) vs. what's needed for >90% clean")
    print("=" * 120)

    for iso in ISOS:
        if iso not in seq_analysis:
            continue
        a = seq_analysis[iso]
        deep_t = a.get('deep_threshold', 99)
        print(f"\n{'─' * 100}")
        print(f"  {iso} — Strategy: {a['strategy'].upper().replace('_', ' ')}")
        print(f"{'─' * 100}")

        build = a['cheap_zone_build']
        print(f"  Built in 50→75%: ", end="")
        parts = [(k, v) for k, v in sorted(build.items(), key=lambda x: -abs(x[1])) if abs(v) > 0.5]
        print(", ".join(f"{k}: {'+' if v > 0 else ''}{v:.0f} TWh" for k, v in parts))

        for res in ['wind', 'solar']:
            s = a['stranding_from_cheap_zone'][res]
            if s['built_in_cheap_zone_twh'] > 1:
                flag = " ⚠ STRANDED" if s['stranded_twh'] > 5 else ""
                print(f"  {res:>7} @ 75%: {s['at_75pct_twh']:.0f} TWh → @ {deep_t}%: {s['at_deep_twh']:.0f} TWh "
                      f"({s['stranded_twh']:.0f} TWh stranded, {s['stranding_pct']:.0f}%){flag}")

        gl = a['gas_lock_in']
        gas_delta = gl['gas_delta_50_to_75_mw']
        direction = "↑" if gas_delta > 0 else "↓"
        print(f"  Gas backup: {gl['gas_at_50_mw']:,} MW @ 50% → {gl['gas_at_75_mw']:,} MW @ 75% "
              f"({direction}{abs(gas_delta):,} MW) → {gl['gas_at_deep_mw']:,} MW @ {deep_t}%")
        print(f"  Clean firm deficit: {gl['clean_firm_at_75_twh']:.0f} TWh @ 75% → "
              f"{gl['clean_firm_at_deep_twh']:.0f} TWh needed @ {deep_t}% = "
              f"{gl['clean_firm_deficit_twh']:.0f} TWh / {gl['clean_firm_deficit_gw']:.0f} GW still to build")

        if gl['forgone_gas_reduction_gw'] > 0.5:
            print(f"  ⚠ Forgone gas reduction: Building wind instead of clean firm in cheap zone "
                  f"left {gl['forgone_gas_reduction_gw']:.0f} GW of gas backup that clean firm would have displaced")

        if a['stranded_cost_bn'] > 0.1 or a['gas_lockin_cost_bn'] > 0.1:
            print(f"  Hidden costs: ${a['stranded_cost_bn']:.1f}B stranded assets + "
                  f"${a['gas_lockin_cost_bn']:.1f}B gas lock-in = ${a['total_hidden_cost_bn']:.1f}B total")
            print(f"  Naive MAC: ${a['mac_naive']:.0f}/tCO₂ → True MAC (with stranding+gas): higher")

    print(f"\n{'=' * 120}")
    print("SUMMARY: WIND-FIRST vs CLEAN-FIRM-FIRST GRIDS")
    print(f"{'=' * 120}")
    print(f"{'ISO':<7} {'Strategy':<18} {'Wind Built':>12} {'Wind Stranded':>14} {'Gas @ 75%':>12} "
          f"{'Gas @ Deep':>12} {'Gas Δ':>10} {'CF Deficit':>12} {'Hidden $B':>10}")
    print("-" * 120)

    for iso in ISOS:
        if iso not in seq_analysis:
            continue
        a = seq_analysis[iso]
        gl = a['gas_lock_in']
        ws = a['stranding_from_cheap_zone']['wind']
        print(f"{iso:<7} {a['strategy']:<18} {ws['built_in_cheap_zone_twh']:>10.0f} TWh "
              f"{ws['stranded_twh']:>12.0f} TWh {gl['gas_at_75_mw']:>10,} MW "
              f"{gl['gas_at_deep_mw']:>10,} MW {gl['gas_delta_50_to_75_mw']:>+9,} MW "
              f"{gl['clean_firm_deficit_twh']:>10.0f} TWh {a['total_hidden_cost_bn']:>9.1f}")


def print_summary(queue, stranding, egrid, fossil_mix):
    """Print human-readable results."""
    print("\n" + "=" * 90)
    print("FOSSIL DISPATCH STACKS (merit-order: coal → oil → gas)")
    print("=" * 90)
    for iso in ISOS:
        coal_cap = COAL_CAP_TWH.get(iso, 0)
        oil_cap = OIL_CAP_TWH.get(iso, 0)
        baseline = sum(GRID_MIX_SHARES.get(iso, {}).values())
        demand = BASE_DEMAND_TWH[iso]
        fossil_twh = demand * (1 - baseline / 100)
        print(f"  {iso}: demand={demand:.0f} TWh, baseline_clean={baseline:.1f}%, "
              f"fossil={fossil_twh:.0f} TWh (coal={coal_cap:.1f}, oil={oil_cap:.1f})")

    print("\n" + "=" * 110)
    print("CONSEQUENTIAL DEPLOYMENT QUEUE (dispatch-cache hourly emission accounting)")
    print("=" * 110)
    print(f"{'#':>3} {'ISO':<7} {'Zone':<12} {'MAC $/t':>9} {'Marg Rate':>10} {'Fuel':>6} "
          f"{'CO₂ MT':>8} {'ΔCost $B':>10} {'Coal Left':>10} {'Primary Resource Change':>35}")
    print("-" * 110)

    for step in queue:
        deltas = step['delta_resources']
        sorted_d = sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True)
        top = sorted_d[0]
        top_str = f"{top[0]}: {'+' if top[1]>0 else ''}{top[1]:.0f} TWh"
        if len(sorted_d) > 1 and abs(sorted_d[1][1]) > 1:
            r2 = sorted_d[1]
            top_str += f", {r2[0]}: {'+' if r2[1]>0 else ''}{r2[1]:.0f}"

        mac_str = f"${step['marginal_mac']:,.0f}" if step['marginal_mac'] < 9999 else "$∞"
        coal_left = step['displacement_detail'].get('cumulative_coal_displaced_twh', 0)
        coal_cap = COAL_CAP_TWH.get(step['iso'], 0)
        coal_remaining = max(0, coal_cap - coal_left)

        print(f"{step['queue_position']:>3} {step['iso']:<7} {step['zone_label']:<12} "
              f"{mac_str:>9} {step['displaced_emission_rate']:>9.4f} {step['primary_fuel_displaced']:>6} "
              f"{step['co2_displaced_mt']:>7.1f} {step['delta_cost_total_bn']:>9.2f} "
              f"{coal_remaining:>9.1f} {top_str:>35}")

    total_co2 = sum(s['co2_displaced_mt'] for s in queue)
    total_cost = sum(s['delta_cost_total_bn'] for s in queue)
    cheap = [s for s in queue if s['marginal_mac'] < 100]
    mid = [s for s in queue if 100 <= s['marginal_mac'] < 500]
    exp = [s for s in queue if s['marginal_mac'] >= 500]

    print("-" * 110)
    print(f"TOTAL: {total_co2:.1f} MT CO₂, ${total_cost:.1f}B annual cost")
    if total_co2 > 0:
        print(f"  Cheap (<$100/t): {len(cheap)} steps, {sum(s['co2_displaced_mt'] for s in cheap):.1f} MT "
              f"({sum(s['co2_displaced_mt'] for s in cheap)/total_co2*100:.0f}% of CO₂)")
    print(f"  Moderate ($100-500/t): {len(mid)} steps, {sum(s['co2_displaced_mt'] for s in mid):.1f} MT")
    print(f"  Expensive (>$500/t): {len(exp)} steps, {sum(s['co2_displaced_mt'] for s in exp):.1f} MT")

    print("\n" + "=" * 80)
    print("STRANDING ANALYSIS (Peak vs Final @ 99%)")
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


def write_outputs(queue, cumulative, stranding, trajectories, projections,
                  rate_trajectory, sequencing_analysis, egrid, fossil_mix):
    """Write JSON and JS output files."""
    stack_summary = {}
    for iso in ISOS:
        coal_cap = COAL_CAP_TWH.get(iso, 0)
        oil_cap = OIL_CAP_TWH.get(iso, 0)
        baseline_clean = sum(GRID_MIX_SHARES.get(iso, {}).values())
        demand = BASE_DEMAND_TWH[iso]

        coal_rate = egrid[iso]['coal_co2_lb_per_mwh'] / 2204.62
        oil_rate = egrid[iso]['oil_co2_lb_per_mwh'] / 2204.62
        gas_rate = egrid[iso]['gas_co2_lb_per_mwh'] / 2204.62

        fuels = []
        if coal_cap > 0.01:
            fuels.append({'type': 'coal', 'cap_twh': round(coal_cap, 2), 'emission_rate': round(coal_rate, 4)})
        if oil_cap > 0.01:
            fuels.append({'type': 'oil', 'cap_twh': round(oil_cap, 2), 'emission_rate': round(oil_rate, 4)})
        fuels.append({'type': 'gas', 'cap_twh': round(demand * (1 - baseline_clean / 100) - coal_cap - oil_cap, 2),
                      'emission_rate': round(gas_rate, 4)})

        stack_summary[iso] = {
            'fuels': fuels,
            'baseline_clean_pct': round(baseline_clean, 1),
            'demand_twh': demand,
            'coal_oil_retirement_threshold': COAL_OIL_RETIREMENT_THRESHOLD,
        }

    output = {
        'metadata': {
            'description': 'Consequential deployment queue with dispatch-cache hourly emission accounting',
            'methodology': 'Uses 8760-hour dispatch cache from step5_build_dispatch_cache for exact fossil displacement. '
                           'Merit-order fuel retirement (coal → oil → gas) applied to hourly dispatch results. '
                           'Marginal emission rate = delta CO₂ / delta displaced TWh between zone boundaries.',
            'zones': [z['label'] for z in ZONES],
            'thresholds': THRESHOLDS,
            'isos': ISOS,
            'medium_scenario_keys': MEDIUM_KEYS,
            'growth_rates_pct': GROWTH_RATES,
            'sbti_year_map': {THRESHOLD_STRS.get(k, str(k)): v for k, v in SBTI_YEAR_MAP.items()},
        },
        'dispatch_stacks': stack_summary,
        'deployment_queue': queue,
        'cumulative_deployment': cumulative,
        'stranding_analysis': stranding,
        'resource_trajectories': trajectories,
        'demand_growth_projections': projections,
        'emission_rate_trajectory': rate_trajectory,
        'sequencing_analysis': sequencing_analysis,
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nJSON: {OUTPUT_JSON} ({os.path.getsize(OUTPUT_JSON) / 1024:.0f} KB)")

    os.makedirs(os.path.dirname(OUTPUT_JS), exist_ok=True)
    with open(OUTPUT_JS, 'w') as f:
        f.write("// Auto-generated by step6_consequential_queue.py\n")
        f.write("// Uses dispatch cache (step5_build_dispatch_cache) for hourly emission accounting\n")
        f.write("// Merit-order: coal→oil→gas; 8760-hour fossil displacement\n\n")
        f.write(f"const CQ_DATA = {json.dumps(output, indent=2, default=str)};\n")
    print(f"JS:   {OUTPUT_JS} ({os.path.getsize(OUTPUT_JS) / 1024:.0f} KB)")


def main():
    print("=" * 60)
    print("CONSEQUENTIAL DEPLOYMENT QUEUE ANALYSIS")
    print("  Dispatch-cache hourly emission accounting")
    print("  Merit-order coal/oil/gas displacement")
    print("=" * 60)

    df, meta, egrid, fossil_mix = load_data()

    # Load demand/gen profiles for dispatch cache lookups
    print("\n  Loading demand and generation profiles...")
    demand_data, gen_profiles, _, _ = load_common_data()

    # Pre-load dispatch caches for all ISOs
    print("  Loading dispatch caches...")
    for iso in ISOS:
        cache = load_dispatch_cache(iso)
        if cache:
            _dispatch_caches[iso] = cache
            print(f"    {iso}: {len(cache)} cached mixes")
        else:
            print(f"    {iso}: no cache (will compute live)")

    med_data = extract_medium_scenarios(df)

    # Compute with dispatch-cache-based emission model
    queue = compute_zone_metrics(med_data, egrid, fossil_mix, demand_data, gen_profiles)
    cumulative = compute_cumulative_deployment(queue)
    stranding = compute_stranding_analysis(med_data)
    trajectories = compute_resource_trajectories(med_data)
    projections = compute_demand_growth(med_data)
    rate_traj = compute_emission_rate_trajectory(egrid, med_data, demand_data, gen_profiles)

    seq_analysis = compute_sequencing_analysis(med_data, stranding, queue)

    print_summary(queue, stranding, egrid, fossil_mix)
    print_sequencing_analysis(seq_analysis)
    write_outputs(queue, cumulative, stranding, trajectories, projections,
                  rate_traj, seq_analysis, egrid, fossil_mix)

    print("\nDone.")


if __name__ == '__main__':
    main()
