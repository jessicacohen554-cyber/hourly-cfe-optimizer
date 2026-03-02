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
       data/egrid_emission_rates.json, data/eia-930/eia_fossil_mix.json
       data/step5-post-processing/dispatch_cache/{ISO}_dispatch_cache.parquet
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

from step3_cost_optimization import OUTPUT_THRESHOLDS as _ALL_THRESHOLDS

# Consequential queue only operates at >= 50% (below 50% is pre-SBTi baseline)
THRESHOLDS = [t for t in _ALL_THRESHOLDS if t >= 50]

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
FOSSIL_MIX_PATH = os.path.join(BASE_DIR, 'data', 'eia-930', 'eia_fossil_mix.json')  # legacy fallback
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

# Build consecutive threshold pair zones from THRESHOLDS (5% intervals)
# e.g., 50→55, 55→60, 60→65, ..., 99.9→99.99
ZONES = []
_zone_thresholds = [t for t in THRESHOLDS if t >= 50]
for _i in range(len(_zone_thresholds) - 1):
    _ts = _zone_thresholds[_i]
    _te = _zone_thresholds[_i + 1]
    _ts_str = str(int(_ts)) if _ts == int(_ts) else str(_ts)
    _te_str = str(int(_te)) if _te == int(_te) else str(_te)
    ZONES.append({
        'label': f'{_ts_str}→{_te_str}%',
        'start_thresh': _ts,
        'end_thresh': _te,
    })

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

# Pre-computed per-ISO data (populated in main() to avoid redundant recomputation)
_iso_supply_profiles = {}   # iso → supply_profiles dict
_iso_supply_matrices = {}   # iso → (5, H) numpy array
_iso_demand_norms = {}      # iso → (H,) numpy array


def get_dispatch_co2(iso, threshold, med_data, egrid, demand_data, gen_profiles):
    """Get CO₂ displacement for a medium-scenario mix using dispatch cache.

    Uses pre-built supply matrices and demand profiles from _iso_supply_matrices
    and _iso_demand_norms (populated in main()) to avoid redundant per-call
    list→array conversions. Falls back to on-the-fly computation if not pre-built.
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
    demand_mwh = mix_info['demand_mwh']

    dispatch_cache = _dispatch_caches.get(iso, {})

    # Use pre-built profiles/matrices if available (set up in main())
    supply_profiles = _iso_supply_profiles.get(iso)
    supply_matrix = _iso_supply_matrices.get(iso)
    demand_norm = _iso_demand_norms.get(iso)

    if supply_profiles is None:
        supply_profiles = get_supply_profiles(iso, gen_profiles)
    if demand_norm is None:
        demand_norm, _ = get_demand_profile(iso, demand_data)

    dispatch_result, cache_hit = get_or_compute_dispatch(
        iso, demand_norm, supply_profiles, resource_pcts,
        100, bat_pct, bat8_pct, ldes_pct,
        cache=dispatch_cache, supply_matrix=supply_matrix)

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
    """Load all input data from Step 4 per-ISO parquets (Step 3 fallback).

    Uses pyarrow push-down filters to load only medium-scenario rows,
    avoiding reading millions of non-medium rows into memory.
    """
    print("Loading optimizer scenarios from per-ISO parquets...")

    input_dir = STEP4_PARQUET_DIR if os.path.isdir(STEP4_PARQUET_DIR) else STEP3_PARQUET_DIR
    medium_scenarios = list(set(MEDIUM_KEYS.values()))
    pq_filters = [('scenario', 'in', medium_scenarios)]

    tables = []
    for iso in ISOS:
        for prefix in ['step4_', 'step3_co_']:
            path = os.path.join(input_dir, f'{prefix}{iso}.parquet')
            if os.path.exists(path):
                try:
                    t = pq.read_table(path, filters=pq_filters)
                except Exception:
                    t = pq.read_table(path)  # fallback if filter fails
                tables.append(t)
                print(f"  Loaded {os.path.basename(path)}: {t.num_rows:,} rows (filtered)")
                break
        else:
            # Try alternate directory
            alt_dir = STEP3_PARQUET_DIR if input_dir == STEP4_PARQUET_DIR else STEP4_PARQUET_DIR
            for prefix in ['step4_', 'step3_co_']:
                path = os.path.join(alt_dir, f'{prefix}{iso}.parquet')
                if os.path.exists(path):
                    try:
                        t = pq.read_table(path, filters=pq_filters)
                    except Exception:
                        t = pq.read_table(path)
                    tables.append(t)
                    print(f"  Loaded {os.path.basename(path)}: {t.num_rows:,} rows (fallback, filtered)")
                    break
            else:
                print(f"  WARNING: No parquet found for {iso} — skipping")

    if tables:
        dfs = [t.to_pandas() for t in tables]
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pq.read_table(
            os.path.join(BASE_DIR, 'dashboard', 'overprocure_scenarios.parquet')).to_pandas()
        df = df[df['scenario'].isin(medium_scenarios)]
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

    Uses vectorized pandas operations instead of row-by-row itertuples/getattr.
    Also extracts battery/ldes dispatch percentages needed for dispatch cache lookup.
    """
    result = {}

    # Pre-compute optional columns that may not exist
    has_battery8 = 'battery8_dispatch_pct' in df.columns
    has_h2 = 'h2_dispatch_pct' in df.columns

    for iso in ISOS:
        med_key = MEDIUM_KEYS[iso]
        iso_df = df[(df['iso'] == iso) & (df['scenario'] == med_key)].copy()
        iso_df = iso_df.sort_values('threshold')

        if iso_df.empty:
            result[iso] = {}
            continue

        # Vectorized column computations
        demand_twh = iso_df['annual_demand_mwh'].values / 1e6
        mix_cols = {res: iso_df[f'mix_{res}'].values for res in RESOURCES}
        bat_pct = iso_df['battery_dispatch_pct'].values
        bat8_pct = iso_df['battery8_dispatch_pct'].values if has_battery8 else np.zeros(len(iso_df))
        ldes_pct = iso_df['ldes_dispatch_pct'].values
        h2_pct = iso_df['h2_dispatch_pct'].values if has_h2 else np.zeros(len(iso_df))

        thresholds = iso_df['threshold'].values.astype(float)
        demand_mwh = iso_df['annual_demand_mwh'].values
        match_scores = iso_df['hourly_match_score'].values
        eff_costs = iso_df['cost_effective_cost'].values
        total_costs = iso_df['cost_total_cost'].values
        incr_costs = iso_df['cost_incremental'].values
        wholesale = iso_df['cost_wholesale'].values
        gas_backup = iso_df['ra_gas_backup_needed_mw'].values
        new_gas = iso_df['ra_new_gas_build_mw'].values
        gas_cost = iso_df['ra_gas_backup_cost_per_mwh'].values
        tr_existing = iso_df['tranche_cf_existing_twh'].values
        tr_uprate = iso_df['tranche_uprate_twh'].values
        tr_geo = iso_df['tranche_geo_twh'].values
        tr_nuclear = iso_df['tranche_nuclear_newbuild_twh'].values
        tr_ccs = iso_df['tranche_ccs_tranche_twh'].values

        # Build result dicts from pre-extracted arrays (vectorized field access)
        iso_result = {}
        # Pre-compute resource TWh arrays (vectorized multiply)
        resource_twh_arrays = {res: mix_cols[res] / 100.0 * demand_twh for res in RESOURCES}
        battery_twh_arr = bat_pct / 100.0 * demand_twh
        ldes_twh_arr = ldes_pct / 100.0 * demand_twh

        for i in range(len(iso_df)):
            t = float(thresholds[i])
            iso_result[t] = {
                'demand_twh': float(demand_twh[i]),
                'demand_mwh': float(demand_mwh[i]),
                'match_score': float(match_scores[i]),
                'eff_cost': float(eff_costs[i]),
                'total_cost': float(total_costs[i]),
                'incremental_cost': float(incr_costs[i]),
                'wholesale': float(wholesale[i]),
                'resource_twh': {res: float(resource_twh_arrays[res][i]) for res in RESOURCES},
                'battery_twh': float(battery_twh_arr[i]),
                'ldes_twh': float(ldes_twh_arr[i]),
                'resource_pct': {res: float(mix_cols[res][i]) for res in RESOURCES},
                'battery_dispatch_pct': float(bat_pct[i]),
                'battery8_dispatch_pct': float(bat8_pct[i]),
                'ldes_dispatch_pct': float(ldes_pct[i]),
                'h2_dispatch_pct': float(h2_pct[i]),
                'gas_backup_mw': float(gas_backup[i]),
                'new_gas_mw': float(new_gas[i]),
                'gas_cost': float(gas_cost[i]),
                'tranche_existing_twh': float(tr_existing[i]),
                'tranche_uprate_twh': float(tr_uprate[i]),
                'tranche_geo_twh': float(tr_geo[i]),
                'tranche_nuclear_twh': float(tr_nuclear[i]),
                'tranche_ccs_twh': float(tr_ccs[i]),
            }
        result[iso] = iso_result

    return result


def compute_baseline_co2_mt(iso, egrid):
    """Compute total fossil CO₂ at existing clean floor (MT).

    Baseline = hourly demand - hourly existing clean = fossil gap.
    Fossil gap dispatched in merit order: coal → oil → gas.
    Returns MT CO₂ from the entire fossil fleet at existing clean levels.
    """
    shares = GRID_MIX_SHARES.get(iso, {})
    existing_pct = sum(shares.values())
    demand_twh = BASE_DEMAND_TWH[iso]
    fossil_twh = demand_twh * (1 - existing_pct / 100)

    coal_cap = COAL_CAP_TWH.get(iso, 0)
    oil_cap = OIL_CAP_TWH.get(iso, 0)

    regional = egrid.get(iso, {})
    coal_rate = regional.get('coal_co2_lb_per_mwh', 0.0) / 2204.62
    oil_rate = regional.get('oil_co2_lb_per_mwh', 0.0) / 2204.62
    gas_rate = regional.get('gas_co2_lb_per_mwh', 0.0) / 2204.62

    coal_twh = min(coal_cap, fossil_twh)
    oil_twh = min(oil_cap, max(0, fossil_twh - coal_twh))
    gas_twh = max(0, fossil_twh - coal_twh - oil_twh)

    # TWh × tCO₂/MWh = MT CO₂ (since 1 TWh × 1 tCO₂/MWh = 1e6 MWh × 1 tCO₂/MWh = 1e6 t = 1 MT)
    return coal_twh * coal_rate + oil_twh * oil_rate + gas_twh * gas_rate


def compute_zone_metrics(med_data, egrid, fossil_mix, demand_data, gen_profiles):
    """Compute metrics for each (ISO, zone) pair using 5% threshold intervals.

    Implements the user's formula:
      Baseline emissions = hourly demand - hourly existing clean = fossil gap
      Reduced emissions  = hourly demand - (existing + new build) = target emissions
      Emissions reduced  = baseline - target
      MAC = cost of new resources / emissions reduced

    Existing clean resources = $0 cost. Only new-build incurs cost.

    Where the optimizer "overshoots" (identical results at consecutive thresholds),
    the initial deployment is distributed proportionally across 5% zones so the
    animation shows smooth progression.
    """
    zone_metrics = []

    for iso in ISOS:
        iso_data = med_data[iso]
        if not iso_data:
            continue
        demand_twh = list(iso_data.values())[0]['demand_twh']
        demand_mwh = demand_twh * 1e6

        # ---- Existing clean floor ----
        shares = GRID_MIX_SHARES.get(iso, {})
        existing_pct = sum(shares.values())
        existing_twh = {}
        for res in RESOURCES:
            existing_twh[res] = demand_twh * shares.get(res, 0) / 100
        existing_twh['battery'] = 0
        existing_twh['ldes'] = 0

        # Baseline CO₂ at existing clean floor (analytical: full fossil dispatch)
        baseline_co2_mt = compute_baseline_co2_mt(iso, egrid)

        # ---- Get dispatch-based CO₂ at each threshold ----
        co2_at_threshold = {}
        for t in _zone_thresholds:
            t_float = float(t)
            if t_float not in iso_data:
                continue
            co2_result = get_dispatch_co2(iso, t_float, med_data, egrid, demand_data, gen_profiles)
            if co2_result:
                abated_mt = co2_result['total_co2_abated_tons'] / 1e6
                co2_at_threshold[t_float] = max(0, baseline_co2_mt - abated_mt)
            else:
                co2_at_threshold[t_float] = baseline_co2_mt

        # ---- Build all zones for this ISO and classify them ----
        threshold_list = [float(t) for t in _zone_thresholds if float(t) in iso_data]
        if len(threshold_list) < 2:
            continue

        # For each zone, check if the optimizer result changed between start and end
        iso_zones = []
        for zone_idx, zone in enumerate(ZONES):
            zt_start = float(zone['start_thresh'])
            zt_end = float(zone['end_thresh'])
            if zt_start not in iso_data or zt_end not in iso_data:
                continue
            d_start = iso_data[zt_start]
            d_end = iso_data[zt_end]
            changed = abs(d_end['match_score'] - d_start['match_score']) > 0.01
            iso_zones.append({
                'zone_idx': zone_idx, 'zone': zone,
                't_start': zt_start, 't_end': zt_end,
                'data_start': d_start, 'data_end': d_end,
                'changed': changed,
            })

        if not iso_zones:
            continue

        # ---- Group consecutive UNCHANGED zones into "distribute" runs ----
        # Changed zones are "actual" (use real optimizer deltas)
        runs = []  # list of (type, [zone_info, ...])
        current_run = []
        for z in iso_zones:
            if z['changed']:
                if current_run:
                    runs.append(('distribute', current_run))
                    current_run = []
                runs.append(('actual', [z]))
            else:
                current_run.append(z)
        if current_run:
            runs.append(('distribute', current_run))

        # ---- Helper: get resource dict from optimizer data ----
        def _res_dict(d):
            r = {}
            for res in RESOURCES:
                r[res] = d['resource_twh'][res]
            r['battery'] = d['battery_twh']
            r['ldes'] = d['ldes_twh']
            return r

        # ---- Process each run ----
        for run_idx, (run_type, run_zones) in enumerate(runs):
            first_z = run_zones[0]
            last_z = run_zones[-1]

            # Determine the START state for this run
            if run_idx == 0 and run_type == 'distribute':
                # First run is distributed: start from EXISTING CLEAN FLOOR ($0 cost)
                start_res = dict(existing_twh)
                start_co2 = baseline_co2_mt
                start_cost_per_mwh = 0  # existing = $0
                start_gas_cost = iso_data[first_z['t_start']]['gas_cost']
                start_gas_mw = iso_data[first_z['t_start']]['new_gas_mw']
                start_match = existing_pct
            else:
                # Start from the optimizer at this zone's start threshold
                d = first_z['data_start']
                start_res = _res_dict(d)
                start_co2 = co2_at_threshold.get(first_z['t_start'], baseline_co2_mt)
                start_cost_per_mwh = d['incremental_cost'] - d['gas_cost']
                start_gas_cost = d['gas_cost']
                start_gas_mw = d['new_gas_mw']
                start_match = d['match_score']

            # Determine the END state for this run
            d_end = last_z['data_end']
            end_res = _res_dict(d_end)
            end_co2 = co2_at_threshold.get(last_z['t_end'], baseline_co2_mt)
            end_cost_per_mwh = d_end['incremental_cost'] - d_end['gas_cost']
            end_gas_cost = d_end['gas_cost']
            end_gas_mw = d_end['new_gas_mw']
            end_match = d_end['match_score']

            # Total deltas for the run
            total_delta_res = {}
            for res in list(RESOURCES) + ['battery', 'ldes']:
                total_delta_res[res] = end_res.get(res, 0) - start_res.get(res, 0)

            total_delta_co2 = max(0, start_co2 - end_co2)
            total_delta_cost_per_mwh = end_cost_per_mwh - start_cost_per_mwh
            total_delta_cost_bn = total_delta_cost_per_mwh * demand_mwh / 1e9
            total_delta_clean_twh = sum(max(0, v) for v in total_delta_res.values())

            # Get dispatch-based emission info
            _, _, displacement = compute_marginal_displaced_rate_dispatch(
                iso, first_z['t_start'], last_z['t_end'],
                egrid, med_data, demand_data, gen_profiles
            )
            ep_rate = displacement.get('avg_rate_at_end', 0) if displacement else 0
            if ep_rate < 0.001:
                regional = egrid.get(iso, {})
                ep_rate = regional.get('gas_co2_lb_per_mwh', 0.0) / 2204.62

            # Newbuild LCOE at run endpoint
            newbuild_lcoe = d_end['total_cost'] - d_end['gas_cost']
            match_frac = end_match / 100.0
            lcoe_per_cfe = newbuild_lcoe / match_frac if match_frac > 0.01 else float('inf')

            # MAC for the run
            if total_delta_co2 > 0.001:
                run_mac = abs(total_delta_cost_bn * 1e3) / total_delta_co2  # $/tCO₂
            elif ep_rate > 0.0001:
                run_mac = lcoe_per_cfe / ep_rate
            else:
                run_mac = float('inf')

            primary_fuel = displacement.get('primary_fuel', 'gas') if displacement else 'gas'

            # ---- Distribute across zones in this run ----
            n_zones = len(run_zones)
            for zi, z in enumerate(run_zones):
                frac = 1.0 / n_zones

                zone_delta_res = {res: total_delta_res.get(res, 0) * frac
                                  for res in list(RESOURCES) + ['battery', 'ldes']}
                zone_co2 = total_delta_co2 * frac
                zone_cost_bn = total_delta_cost_bn * frac
                zone_clean_twh = total_delta_clean_twh * frac
                zone_cost_per_mwh = total_delta_cost_per_mwh * frac
                gas_delta_mw = (end_gas_mw - start_gas_mw) * frac
                gas_cost_delta = (end_gas_cost - start_gas_cost) * frac

                year_start = SBTI_YEAR_MAP.get(z['t_start'], 2025)
                year_end = SBTI_YEAR_MAP.get(z['t_end'], 2050)
                midpoint_year = (year_start + year_end) / 2
                growth_factor = (1 + GROWTH_RATES[iso] / 100) ** (midpoint_year - 2025)

                # Interpolated match at zone end
                interp_match = start_match + (end_match - start_match) * (zi + 1) / n_zones

                zone_metrics.append({
                    'iso': iso,
                    'zone_idx': z['zone_idx'],
                    'zone_label': z['zone']['label'],
                    'threshold_start': z['t_start'],
                    'threshold_end': z['t_end'],
                    'year_start': year_start,
                    'year_end': year_end,
                    'marginal_mac': round(min(run_mac, 9999), 1),
                    'marginal_mac_display': round(min(run_mac, 1500), 1),
                    'newbuild_lcoe_per_cfe': round(lcoe_per_cfe, 1),
                    'endpoint_emission_rate': round(ep_rate, 4),
                    'co2_displaced_mt': round(zone_co2, 2),
                    'displaced_emission_rate': round(ep_rate, 4),
                    'displacement_detail': displacement if displacement else {},
                    'primary_fuel_displaced': primary_fuel,
                    'fossil_twh_start': round(demand_twh * (1 - start_match / 100), 1),
                    'fossil_twh_end': round(demand_twh * (1 - interp_match / 100), 1),
                    'delta_clean_twh': round(zone_clean_twh, 1),
                    'delta_cost_per_mwh': round(zone_cost_per_mwh, 2),
                    'delta_cost_total_bn': round(zone_cost_bn, 2),
                    'delta_resources': {k: round(v, 1) for k, v in zone_delta_res.items()},
                    'end_resource_twh': {k: round(v, 1) for k, v in end_res.items()},
                    'gas_backup_mw_end': d_end['gas_backup_mw'],
                    'gas_cost_per_mwh_end': round(end_gas_cost, 2),
                    'delta_gas_cost_per_mwh': round(gas_cost_delta, 2),
                    'delta_gas_mw': round(gas_delta_mw, 0),
                    'demand_twh': demand_twh,
                    'growth_factor': round(growth_factor, 3),
                    'growth_adjusted_demand_twh': round(demand_twh * growth_factor, 1),
                    'growth_adjusted_co2_mt': round(zone_co2 * growth_factor, 2),
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

        cheap_step = next((s for s in queue if s['iso'] == iso and s['threshold_start'] == 50.0), None)
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

    # ---- Derive keys that the page animation needs (server-side) ----
    # These used to be computed client-side in a fragile IIFE. Computing them here
    # ensures the data file arrives complete and the animation always works.

    # 1. existing_clean_floor — 2025 EIA clean generation per ISO
    existing_clean_floor = {}
    for iso in ISOS:
        shares = GRID_MIX_SHARES.get(iso, {})
        demand = BASE_DEMAND_TWH[iso]
        by_resource = {}
        total = 0
        for r in ['clean_firm', 'solar', 'wind', 'hydro']:
            twh = round(demand * shares.get(r, 0) / 100, 1)
            by_resource[r] = twh
            total += twh
        by_resource['ccs_ccgt'] = 0
        by_resource['battery'] = 0
        by_resource['ldes'] = 0
        existing_clean_floor[iso] = {
            'by_resource': by_resource,
            'total_twh': round(total, 1),
        }

    # 2. growth_adjusted_cumulative — alias for cumulative_deployment
    growth_adjusted_cumulative = cumulative

    # 3. fossil_baselines_2045 — fossil generation at 2045 demand
    fossil_baselines_2045 = {}
    for iso in ISOS:
        demand_2045 = projections[iso][2045]['demand_twh'] if 2045 in projections.get(iso, {}) else BASE_DEMAND_TWH[iso] * 1.43
        ecf_total = existing_clean_floor[iso]['total_twh']
        fossil_twh_total = max(0, demand_2045 - ecf_total)

        fuels_list = stack_summary[iso]['fuels']
        total_fuel = sum(max(0, f['cap_twh']) for f in fuels_list)

        coal_twh = oil_twh = gas_twh = 0.0
        coal_rate = oil_rate = gas_rate = 0.0
        for f in fuels_list:
            ratio = max(0, f['cap_twh']) / total_fuel if total_fuel > 0 else 0
            twh = fossil_twh_total * ratio
            if f['type'] == 'coal':
                coal_twh, coal_rate = twh, f['emission_rate']
            elif f['type'] == 'oil':
                oil_twh, oil_rate = twh, f['emission_rate']
            elif f['type'] == 'gas':
                gas_twh, gas_rate = twh, f['emission_rate']

        fossil_baselines_2045[iso] = {
            'demand_2045_twh': round(demand_2045, 1),
            'fossil_twh': round(fossil_twh_total, 1),
            'coal_twh': round(coal_twh, 1),
            'oil_twh': round(oil_twh, 1),
            'gas_twh': round(gas_twh, 1),
            'fossil_co2_mt': round(coal_twh * coal_rate + oil_twh * oil_rate + gas_twh * gas_rate, 2),
        }

    # 4. first_deployments — scan queue for first occurrence of each tech
    first_deployments = {'first_clean_firm': None, 'first_battery': None, 'first_ccs': None, 'first_ldes': None}
    for idx, qe in enumerate(queue):
        dr = qe.get('delta_resources', {})
        if not first_deployments['first_clean_firm'] and dr.get('clean_firm', 0) > 0.5:
            first_deployments['first_clean_firm'] = {
                'queue_position': idx, 'iso': qe['iso'], 'zone_label': qe['zone_label'],
                'year_end': qe.get('year_end'), 'clean_firm_twh': dr['clean_firm'],
            }
        if not first_deployments['first_battery'] and dr.get('battery', 0) > 0.5:
            first_deployments['first_battery'] = {
                'queue_position': idx, 'iso': qe['iso'], 'zone_label': qe['zone_label'],
                'year_end': qe.get('year_end'),
            }
        if not first_deployments['first_ccs'] and dr.get('ccs_ccgt', 0) > 0.5:
            first_deployments['first_ccs'] = {
                'queue_position': idx, 'iso': qe['iso'], 'zone_label': qe['zone_label'],
                'year_end': qe.get('year_end'),
            }
        if not first_deployments['first_ldes'] and dr.get('ldes', 0) > 0.5:
            first_deployments['first_ldes'] = {
                'queue_position': idx, 'iso': qe['iso'], 'zone_label': qe['zone_label'],
                'year_end': qe.get('year_end'),
            }

    # 5. iso_cfe_progress — CFE delta from resource trajectories
    iso_cfe_progress = {}
    for iso in ISOS:
        iso_traj = trajectories.get(iso, [])
        stack = stack_summary.get(iso)
        if not iso_traj or not stack:
            continue
        first = iso_traj[0]
        last = iso_traj[-1]
        start_cfe = stack.get('baseline_clean_pct', 50)
        end_cfe = last.get('match_score', start_cfe)
        iso_cfe_progress[iso] = {
            'start_cfe_pct': start_cfe,
            'end_cfe_pct': end_cfe,
            'delta_cfe_pct': end_cfe - start_cfe,
            'new_gas_mw_at_start': first.get('new_gas_mw', 0),
            'new_gas_mw_at_end': last.get('new_gas_mw', 0),
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
        'growth_adjusted_cumulative': growth_adjusted_cumulative,
        'existing_clean_floor': existing_clean_floor,
        'fossil_baselines_2045': fossil_baselines_2045,
        'first_deployments': first_deployments,
        'iso_cfe_progress': iso_cfe_progress,
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

    # Pre-build supply matrices and demand profiles per ISO (avoids redundant
    # list→array conversion on every dispatch call)
    print("  Pre-building supply matrices and demand profiles...")
    for iso in ISOS:
        profiles = get_supply_profiles(iso, gen_profiles)
        _iso_supply_profiles[iso] = profiles
        _iso_supply_matrices[iso] = build_supply_matrix(profiles)
        _iso_demand_norms[iso], _ = get_demand_profile(iso, demand_data)

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
