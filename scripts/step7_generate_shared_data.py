#!/usr/bin/env python3
"""
Generate complete shared-data.js from pipeline results + mac_stats.json.
Replaces the entire file with fresh data from the latest pipeline run.

Pipeline order: Step 1 (physics) → Step 2 (tranche) → postprocess → co2 → mac_stats → THIS

Input:  data/step5-post-processing/co2_results/ (CO2-enriched parquets, preferred)
        dashboard/overprocure_results.json      (monolithic JSON fallback)
        data/step5-post-processing/mac_stats.json
Output: dashboard/js/shared-data.js             (complete rewrite)
        data/step5-post-processing/shared_data.json (canonical JSON archive)
"""

import json
import os
import sys
from datetime import datetime

# Import cost tables directly from Step 3 (single source of truth)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step3_cost_optimization import (
    OUTPUT_THRESHOLDS as _OUTPUT_THRESHOLDS,
    GRID_MIX_SHARES as _GRID_MIX_SHARES,
    WHOLESALE_PRICES as _WHOLESALE_PRICES,
    FUEL_ADJUSTMENTS as _FUEL_ADJUSTMENTS,
    LCOE_TABLES as _LCOE_TABLES,
    TX_TABLES as _TX_TABLES,
    UPRATE_LCOE as _UPRATE_LCOE,
    NUCLEAR_NEWBUILD_LCOE as _NUCLEAR_NEWBUILD_LCOE,
    CCS_LCOE_45Q_ON as _CCS_LCOE_45Q_ON,
    CCS_LCOE_45Q_OFF as _CCS_LCOE_45Q_OFF,
    GEOTHERMAL_LCOE as _GEOTHERMAL_LCOE,
    GEO_CAP_TWH as _GEO_CAP_TWH,
    REGIONAL_DEMAND_TWH as _REGIONAL_DEMAND_TWH,
)

ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']
THRESHOLDS_NUM = _OUTPUT_THRESHOLDS
THRESHOLDS = [str(int(t)) if t == int(t) else str(t) for t in THRESHOLDS_NUM]
RESOURCES = ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro']
MATCHED_RESOURCES = ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro', 'battery', 'battery8', 'ldes', 'h2']

MAC_CAP = 1000  # Cap marginal MAC at $1000/ton


def medium_key(iso):
    """Return the all-Medium 9-dim scenario key for an ISO.
    Format: RFBL_FF_TX_CCSq45_GEO (R=ren, F=firm, B=batt, L=ldes)
    CAISO: MMMM_M_M_M1_M  (geothermal=M)
    Others: MMMM_M_M_M1_X  (no geothermal)
    """
    geo = 'M' if iso == 'CAISO' else 'X'
    return f'MMMM_M_M_M1_{geo}'


def get_scenario(iso_data, threshold, iso):
    """Get the medium scenario for an ISO/threshold, with backward compat fallback."""
    scenarios = iso_data.get('thresholds', {}).get(threshold, {}).get('scenarios', {})
    mk = medium_key(iso)
    return (scenarios.get(mk) or
            scenarios.get('MMM_M_M_M1_M') or scenarios.get('MMM_M_M_M1_X') or  # 8-dim legacy
            scenarios.get('MMM_M_M'))  # 5-dim legacy

# ============================================================================
# LOAD DATA
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STEP5_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'step5-post-processing')
MAC_STATS_PATH = os.path.join(STEP5_DIR, 'mac_stats.json')
# Fallback to legacy path if step5 dir doesn't have mac_stats yet
if not os.path.exists(MAC_STATS_PATH):
    MAC_STATS_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'mac_stats.json')

RESULTS_JSON = os.path.join(SCRIPT_DIR, '..', 'dashboard', 'overprocure_results.json')
CO2_BATCH_DIR = os.path.join(STEP5_DIR, 'co2_results')


def load_from_co2_batches(batch_dir, isos, thresholds):
    """Reassemble results dict from batched co2 result files.

    Reads _config.json, per-ISO meta files, and per-ISO/threshold files
    written by step6_recompute_co2.py.
    """
    import glob as _glob

    config_path = os.path.join(batch_dir, '_config.json')
    if not os.path.exists(config_path):
        return None

    with open(config_path) as f:
        data = json.load(f)

    data.setdefault('results', {})

    for iso in isos:
        iso_entry = {}

        # Load ISO-level metadata (sweep, annual_demand_mwh, etc.)
        meta_files = sorted(_glob.glob(os.path.join(batch_dir, f'{iso}_meta_*.json')))
        for mf in meta_files:
            with open(mf) as f:
                iso_entry.update(json.load(f))

        # Load per-threshold files
        iso_entry.setdefault('thresholds', {})
        for t in thresholds:
            pattern = os.path.join(batch_dir, f'{iso}_{t}_*.json')
            matches = sorted(_glob.glob(pattern))
            if matches:
                with open(matches[0]) as f:
                    iso_entry['thresholds'][t] = json.load(f)

        if iso_entry.get('thresholds') or 'annual_demand_mwh' in iso_entry:
            data['results'][iso] = iso_entry

    return data


print("Loading data...")

# Priority: (1) CO2 parquets, (2) monolithic JSON, (3) batched CO2 JSONs (legacy)
data = None

# Try CO2 parquets first (preferred — output of step6_recompute_co2.py)
if os.path.isdir(CO2_BATCH_DIR):
    co2_parquets = [f for f in os.listdir(CO2_BATCH_DIR) if f.endswith('.parquet')]
    if co2_parquets:
        sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))
        from parquet_io import load_from_parquets
        data = load_from_parquets(CO2_BATCH_DIR, ISOS)
        n_isos = len(data.get('results', {}))
        if n_isos > 0:
            print(f"  Results: loaded from CO2 parquets in {CO2_BATCH_DIR}/ ({n_isos} ISOs)")
        else:
            data = None  # Fall through to next option

# Monolithic JSON fallback
if data is None and os.path.exists(RESULTS_JSON):
    with open(RESULTS_JSON) as f:
        data = json.load(f)
    print(f"  Results: {os.path.getsize(RESULTS_JSON) / 1024:.0f} KB (monolithic JSON)")

# Legacy batched CO2 JSONs fallback
if data is None and os.path.isdir(CO2_BATCH_DIR):
    data = load_from_co2_batches(CO2_BATCH_DIR, ISOS, THRESHOLDS)
    if data is not None:
        n_isos = len(data.get('results', {}))
        print(f"  Results: reassembled from batched JSONs in {CO2_BATCH_DIR}/ ({n_isos} ISOs)")

if data is None:
    raise FileNotFoundError(
        f"No results found. Checked:\n  {CO2_BATCH_DIR}/ (parquets)\n  {RESULTS_JSON}\n  {CO2_BATCH_DIR}/ (JSONs)")

with open(MAC_STATS_PATH) as f:
    mac_stats = json.load(f)
print(f"  MAC stats: {os.path.getsize(MAC_STATS_PATH) / 1024:.0f} KB")

# Optimal targets (from step6_compute_optimal_targets.py) — optional
OPTIMAL_TARGETS_PATH = os.path.join(STEP5_DIR, 'optimal_targets.json')
optimal_targets = {}
if os.path.exists(OPTIMAL_TARGETS_PATH):
    with open(OPTIMAL_TARGETS_PATH) as f:
        optimal_targets = json.load(f)
    print(f"  Optimal targets: {os.path.getsize(OPTIMAL_TARGETS_PATH) / 1024:.0f} KB")
else:
    print("  Optimal targets: not found (step6_compute_optimal_targets.py not yet run)")

# Filter ISOS to only those present in BOTH results and mac_stats
results_isos = set(data.get('results', {}).keys())
mac_isos = set(mac_stats.get('envelope', {}).keys())
available_isos = [iso for iso in ISOS if iso in results_isos and iso in mac_isos]
if set(available_isos) != set(ISOS):
    missing = set(ISOS) - set(available_isos)
    print(f"  WARNING: Skipping ISOs not in both results & mac_stats: {missing}")
    ISOS = available_isos
print(f"  Processing ISOs: {ISOS}")

# ============================================================================
# EXTRACT MAC_DATA (average MAC — medium/low/high)
# ============================================================================

print("\nExtracting MAC_DATA...")
mac_data = {'medium': {}, 'low': {}, 'high': {}}
for iso in ISOS:
    env = mac_stats['envelope'][iso]['envelope']
    fan = mac_stats['fan_chart'][iso]

    # Medium = envelope (monotonic running-max of medium key (9-dim))
    mac_data['medium'][iso] = [round(v) if v is not None else None for v in env]
    # Low = P10 of fan chart
    mac_data['low'][iso] = [round(v) if v is not None else None for v in fan['p10']]
    # High = P90 of fan chart
    mac_data['high'][iso] = [round(v) if v is not None else None for v in fan['p90']]

    print(f"  {iso} medium: {mac_data['medium'][iso]}")

# ============================================================================
# EXTRACT MARGINAL_MAC_DATA (6-zone stepwise — medium/low/high)
# ============================================================================
# With 15 thresholds [50,55,60,65,70,75,80,85,87.5,90,92.5,95,97.5,99,≥99.99]:
#   stepwise_envelope indices: 0=None, 1=50→55, 2=55→60, 3=60→65,
#     4=65→70, 5=70→75, 6=75→80, 7=80→85, 8=85→87.5, 9=87.5→90,
#     10=90→92.5, 11=92.5→95, 12=95→97.5, 13=97.5→99, 14=99→≥99.99
# Zones:
#   Zone 0: 50→75%  (aggregate steps 1-5)
#   Zone 1: 75→90%  (aggregate steps 6-9)
#   Zone 2-5: 90→92.5, 92.5→95, 95→97.5, 97.5→99 (steps 10-13)

print("\nExtracting MARGINAL_MAC_DATA...")
marginal_mac_data = {'medium': {}, 'low': {}, 'high': {}}

def aggregate_zone(sw, start, end):
    """Average non-None stepwise MAC values in [start, end) range, capped at MAC_CAP."""
    steps = [sw[i] for i in range(start, end) if i < len(sw) and sw[i] is not None]
    if not steps:
        return None
    avg = round(sum(steps) / len(steps))
    return min(avg, MAC_CAP)

for iso in ISOS:
    # Medium: use stepwise_envelope (monotonic)
    sw_env = mac_stats['envelope'][iso]['stepwise_envelope']
    zone_entry = aggregate_zone(sw_env, 1, 6)   # 50→75% (steps 1-5)
    zone_backbone = aggregate_zone(sw_env, 6, 10)  # 75→90% (steps 6-9)
    zones = [zone_entry, zone_backbone]
    # Zones 2-5: granular steps 90→92.5, 92.5→95, 95→97.5, 97.5→99
    for step_idx in range(10, 14):
        v = sw_env[step_idx] if step_idx < len(sw_env) else None
        zones.append(min(round(v), MAC_CAP) if v is not None else None)
    marginal_mac_data['medium'][iso] = zones

    # Low: use stepwise_fan P10
    sw_lo = mac_stats['stepwise_fan'][iso]['p10']
    lo_zones = [aggregate_zone(sw_lo, 1, 6), aggregate_zone(sw_lo, 6, 10)]
    for step_idx in range(10, 14):
        v = sw_lo[step_idx] if step_idx < len(sw_lo) else None
        lo_zones.append(min(round(v), MAC_CAP) if v is not None else None)
    marginal_mac_data['low'][iso] = lo_zones

    # High: use stepwise_fan P90
    sw_hi = mac_stats['stepwise_fan'][iso]['p90']
    hi_zones = [aggregate_zone(sw_hi, 1, 6), aggregate_zone(sw_hi, 6, 10)]
    for step_idx in range(10, 14):
        v = sw_hi[step_idx] if step_idx < len(sw_hi) else None
        hi_zones.append(min(round(v), MAC_CAP) if v is not None else None)
    marginal_mac_data['high'][iso] = hi_zones

    print(f"  {iso} medium: {marginal_mac_data['medium'][iso]}")

# ============================================================================
# EXTRACT EFFECTIVE_COST_DATA
# ============================================================================

print("\nExtracting EFFECTIVE_COST_DATA...")
effective_cost_data = {}
for iso in ISOS:
    costs = []
    for t in THRESHOLDS:
        sc = data['results'][iso]['thresholds'].get(t, {}).get('scenarios', {}).get(medium_key(iso))
        if sc:
            costs.append(round(sc['costs']['effective_cost'], 1))
        else:
            costs.append(None)
    effective_cost_data[iso] = costs
    print(f"  {iso}: {costs}")

# Enforce monotonicity (lower thresholds <= higher thresholds)
for iso in ISOS:
    arr = effective_cost_data[iso]
    # Top-down ceiling enforcement
    for i in range(len(arr) - 2, -1, -1):
        if arr[i] is not None and arr[i + 1] is not None and arr[i] > arr[i + 1]:
            arr[i] = arr[i + 1]

# ============================================================================
# EXTRACT SYSTEM_COST_DATA (P10/Median/P90 of total_system_cost across 324 scenarios)
# ============================================================================

print("\nExtracting SYSTEM_COST_DATA (P10/Median/P90)...")
import numpy as np_sc

system_cost_data = {'medium': {}, 'low': {}, 'high': {}}
for iso in ISOS:
    med_line, lo_line, hi_line = [], [], []
    for t in THRESHOLDS:
        scenarios = data['results'][iso]['thresholds'].get(t, {}).get('scenarios', {})
        costs = []
        for sc_key, sc in scenarios.items():
            tsc = sc.get('gas_backup', {}).get('total_system_cost_per_mwh')
            if tsc is not None and tsc > 0:
                costs.append(tsc)
        if costs:
            med_line.append(round(float(np_sc.median(costs)), 2))
            lo_line.append(round(float(np_sc.percentile(costs, 10)), 2))
            hi_line.append(round(float(np_sc.percentile(costs, 90)), 2))
        else:
            med_line.append(None)
            lo_line.append(None)
            hi_line.append(None)
    system_cost_data['medium'][iso] = med_line
    system_cost_data['low'][iso] = lo_line
    system_cost_data['high'][iso] = hi_line
    print(f"  {iso} med: {med_line}")

# ============================================================================
# EXTRACT CLEAN_COST_DATA (P10/Median/P90 of effective_cost — NO gas backup)
# ============================================================================
# MAC and crossover analysis use clean procurement cost only.
# Gas backup is a system reliability cost, not an abatement cost.

print("\nExtracting CLEAN_COST_DATA (P10/Median/P90 of effective_cost)...")

clean_cost_data = {'medium': {}, 'low': {}, 'high': {}}
for iso in ISOS:
    med_line, lo_line, hi_line = [], [], []
    for t in THRESHOLDS:
        scenarios = data['results'][iso]['thresholds'].get(t, {}).get('scenarios', {})
        costs = []
        for sc_key, sc in scenarios.items():
            eff = sc.get('costs', {}).get('effective_cost')
            if eff is not None and eff > 0:
                costs.append(eff)
        if costs:
            med_line.append(round(float(np_sc.median(costs)), 2))
            lo_line.append(round(float(np_sc.percentile(costs, 10)), 2))
            hi_line.append(round(float(np_sc.percentile(costs, 90)), 2))
        else:
            med_line.append(None)
            lo_line.append(None)
            hi_line.append(None)
    clean_cost_data['medium'][iso] = med_line
    clean_cost_data['low'][iso] = lo_line
    clean_cost_data['high'][iso] = hi_line
    print(f"  {iso} clean_med: {med_line}")

# ============================================================================
# EXTRACT RESOURCE_MIX_DATA (from Step 3 demand-growth parquets, Medium growth)
# ============================================================================
# Each threshold maps to an SBTi target year via THRESHOLD_TARGET_YEARS.
# Demand grows at Medium rate from 2025 to that year, changing the cost-optimal
# mix. DG parquets (step3_dg_{ISO}_t{threshold}.parquet) contain the repriced
# results under demand growth — we extract Medium growth + Medium sensitivity.
# This replaces the old approach of reading static 2025-demand mixes from
# overprocure_results.json, which showed identical mixes up to ~80%.

print("\nExtracting RESOURCE_MIX_DATA from DG parquets (Medium growth × Medium sensitivity)...")
DG_PARQUET_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'step3-cost-opt-parquets')

try:
    import pandas as _pd
except ImportError:
    print("  WARNING: pandas not available — falling back to base results (no demand growth)")
    _pd = None

resource_mix_data = {}
demand_twh_by_threshold = {}  # {iso: [twh_at_threshold_0, twh_at_threshold_1, ...]}
hourly_match_scores = {}  # {iso: [score_at_threshold_0, score_at_threshold_1, ...]}

for iso in ISOS:
    iso_data = {r: [] for r in RESOURCES}
    iso_data['battery'] = []
    iso_data['battery8'] = []
    iso_data['ldes'] = []
    iso_data['h2'] = []
    iso_demand_twh = []
    iso_match_scores = []

    geo = 'M' if iso == 'CAISO' else 'X'
    med_scenario_key = f'MMMM_M_M_M1_{geo}'

    for t_idx, t in enumerate(THRESHOLDS):
        t_num = THRESHOLDS_NUM[t_idx]
        dg_file = os.path.join(DG_PARQUET_DIR, f'step3_dg_{iso}_t{t_num}.parquet')

        dg_row = None
        if _pd is not None and os.path.exists(dg_file):
            try:
                df = _pd.read_parquet(dg_file)
                med = df[(df['growth_level'] == 'Medium') & (df['scenario'] == med_scenario_key)]
                if len(med) > 0:
                    dg_row = med.iloc[0]
            except Exception as e:
                print(f"  WARNING: Error reading {dg_file}: {e}")

        if dg_row is not None:
            iso_data['clean_firm'].append(int(dg_row['mix_clean_firm']))
            iso_data['solar'].append(int(dg_row['mix_solar']))
            iso_data['wind'].append(int(dg_row['mix_wind']))
            iso_data['offshore_wind'].append(int(dg_row.get('mix_offshore_wind', 0)))
            iso_data['ccs_ccgt'].append(int(dg_row['mix_ccs_ccgt']))
            iso_data['hydro'].append(int(dg_row['mix_hydro']))
            iso_data['battery'].append(round(float(dg_row.get('battery_dispatch_pct', 0)), 4))
            iso_data['battery8'].append(round(float(dg_row.get('battery8_dispatch_pct', 0)), 4))
            iso_data['ldes'].append(round(float(dg_row.get('ldes_dispatch_pct', 0)), 4))
            iso_data['h2'].append(round(float(dg_row.get('h2_dispatch_pct', 0)), 4))
            iso_demand_twh.append(round(dg_row['annual_demand_mwh'] / 1e6, 2))
            iso_match_scores.append(round(float(dg_row.get('hourly_match_score', t_num)), 2))
        else:
            # Fallback to base results (no demand growth)
            sc = data['results'][iso]['thresholds'].get(t, {}).get('scenarios', {}).get(medium_key(iso))
            if sc:
                rm = sc.get('resource_mix', {})
                for res in RESOURCES:
                    iso_data[res].append(rm.get(res, 0))
                iso_data['battery'].append(sc.get('battery_dispatch_pct', 0))
                iso_data['battery8'].append(sc.get('battery8_dispatch_pct', 0))
                iso_data['ldes'].append(sc.get('ldes_dispatch_pct', 0))
                iso_data['h2'].append(sc.get('h2_dispatch_pct', 0))
            else:
                for res in RESOURCES:
                    iso_data[res].append(0)
                iso_data['battery'].append(0)
                iso_data['battery8'].append(0)
                iso_data['ldes'].append(0)
                iso_data['h2'].append(0)
            # Use base demand
            base_twh = _REGIONAL_DEMAND_TWH.get(iso, 100)
            iso_demand_twh.append(round(base_twh, 2))
            # Fallback: use threshold as score (conservative estimate)
            fallback_score = float(sc.get('hourly_match_score', t_num)) if sc else float(t_num)
            iso_match_scores.append(round(fallback_score, 2))

    resource_mix_data[iso] = iso_data
    demand_twh_by_threshold[iso] = iso_demand_twh
    hourly_match_scores[iso] = iso_match_scores
    print(f"  {iso} clean_firm: {iso_data['clean_firm']}")
    print(f"  {iso} demand_twh: {iso_demand_twh[:5]}...{iso_demand_twh[-3:]}")
    print(f"  {iso} match_scores: {iso_match_scores[:5]}...{iso_match_scores[-3:]}")

# ============================================================================
# EXTRACT COMPRESSED_DAY_DATA (from compressed_day_profiles.json)
# ============================================================================

print("\nExtracting COMPRESSED_DAY_DATA...")
cd_profiles_path = os.path.join(SCRIPT_DIR, '..', 'dashboard', 'compressed_day_profiles.json')
# Fallback to step5-post-processing dir
if not os.path.exists(cd_profiles_path):
    cd_profiles_path = os.path.join(STEP5_DIR, 'compressed_day_profiles.json')
cd_profiles = {}
if os.path.exists(cd_profiles_path):
    with open(cd_profiles_path) as f:
        cd_profiles = json.load(f)
    print(f"  Loaded compressed_day_profiles.json ({os.path.getsize(cd_profiles_path) / 1024 / 1024:.1f} MB)")
else:
    print("  WARNING: compressed_day_profiles.json not found — compressed day will be zeros")

compressed_day_data = {}
for iso in ISOS:
    iso_cd = {
        'demand': [],
        'matched': {r: [] for r in MATCHED_RESOURCES},
        'surplus': {r: [] for r in ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro']},
        'gap': [],
        'battery_charge': [],
        'battery8_charge': [],
        'ldes_charge': [],
        'h2_charge': [],
    }

    iso_profiles = cd_profiles.get(iso, {}).get('profiles', {})

    for t in THRESHOLDS:
        # Build the mix_key for Medium scenario at this threshold
        sc = data['results'][iso]['thresholds'].get(t, {}).get('scenarios', {}).get(medium_key(iso))
        profile = None
        if sc and iso_profiles:
            rm = sc.get('resource_mix', {})
            batt = sc.get('battery_dispatch_pct', 0)
            batt8 = sc.get('battery8_dispatch_pct', 0)
            ldes = sc.get('ldes_dispatch_pct', 0)
            h2 = sc.get('h2_dispatch_pct', 0)
            mk = f"{rm.get('clean_firm',0)}_{rm.get('solar',0)}_{rm.get('wind',0)}_{rm.get('offshore_wind',0)}_{rm.get('ccs_ccgt',0)}_{rm.get('hydro',0)}_100_{batt}_{batt8}_{ldes}_{h2}"
            profile = iso_profiles.get(mk)
            # Fallback to old mix_key format without offshore_wind
            if not profile:
                mk_v2 = f"{rm.get('clean_firm',0)}_{rm.get('solar',0)}_{rm.get('wind',0)}_{rm.get('ccs_ccgt',0)}_{rm.get('hydro',0)}_100_{batt}_{batt8}_{ldes}_{h2}"
                profile = iso_profiles.get(mk_v2)
            # Fallback to old mix_key format without h2
            if not profile:
                mk_old = f"{rm.get('clean_firm',0)}_{rm.get('solar',0)}_{rm.get('wind',0)}_{rm.get('ccs_ccgt',0)}_{rm.get('hydro',0)}_100_{batt}_{batt8}_{ldes}"
                profile = iso_profiles.get(mk_old)
            # Fallback to old mix_key format without battery8/h2
            if not profile:
                mk_old2 = f"{rm.get('clean_firm',0)}_{rm.get('solar',0)}_{rm.get('wind',0)}_{rm.get('ccs_ccgt',0)}_{rm.get('hydro',0)}_100_{batt}_{ldes}"
                profile = iso_profiles.get(mk_old2)

        if profile:
            iso_cd['demand'].append([round(v, 5) for v in profile['demand']])
            iso_cd['gap'].append([round(v, 5) for v in profile['gap']])
            iso_cd['battery_charge'].append([round(v, 5) for v in profile.get('battery_charge', [0]*24)])
            iso_cd['battery8_charge'].append([round(v, 5) for v in profile.get('battery8_charge', [0]*24)])
            iso_cd['ldes_charge'].append([round(v, 5) for v in profile.get('ldes_charge', [0]*24)])
            iso_cd['h2_charge'].append([round(v, 5) for v in profile.get('h2_charge', [0]*24)])
            for res in MATCHED_RESOURCES:
                vals = profile.get('matched', {}).get(res, [0]*24)
                iso_cd['matched'][res].append([round(v, 5) for v in vals])
            for res in ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro']:
                vals = profile.get('surplus', {}).get(res, [0]*24)
                iso_cd['surplus'][res].append([round(v, 5) for v in vals])
        else:
            iso_cd['demand'].append([0]*24)
            iso_cd['gap'].append([0]*24)
            iso_cd['battery_charge'].append([0]*24)
            iso_cd['battery8_charge'].append([0]*24)
            iso_cd['ldes_charge'].append([0]*24)
            iso_cd['h2_charge'].append([0]*24)
            for res in MATCHED_RESOURCES:
                iso_cd['matched'][res].append([0]*24)
            for res in ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro']:
                iso_cd['surplus'][res].append([0]*24)

    compressed_day_data[iso] = iso_cd
    filled = sum(1 for d in iso_cd['demand'] if any(v > 0 for v in d))
    print(f"  {iso}: {filled}/{len(iso_cd['demand'])} thresholds with profile data")

# ============================================================================
# EXTRACT CF_TRANCHE_DATA
# ============================================================================

print("\nExtracting CF_TRANCHE_DATA...")
cf_tranche_data = {}
for iso in ISOS:
    iso_tr = {k: [] for k in ['new_cf_twh', 'cf_existing_twh', 'uprate_twh',
                               'geo_twh', 'nuclear_newbuild_twh', 'ccs_tranche_twh',
                               'uprate_price', 'newbuild_price', 'effective_cf_lcoe']}
    for t in THRESHOLDS:
        sc = data['results'][iso]['thresholds'].get(t, {}).get('scenarios', {}).get(medium_key(iso))
        tc = sc.get('tranche_costs', {}) if sc else {}
        iso_tr['new_cf_twh'].append(round(tc.get('new_cf_twh', 0), 3))
        iso_tr['cf_existing_twh'].append(round(tc.get('cf_existing_twh', 0), 3))
        iso_tr['uprate_twh'].append(round(tc.get('uprate_twh', 0), 4))
        iso_tr['geo_twh'].append(round(tc.get('geo_twh', 0), 3))
        iso_tr['nuclear_newbuild_twh'].append(round(tc.get('nuclear_newbuild_twh', 0), 3))
        iso_tr['ccs_tranche_twh'].append(round(tc.get('ccs_tranche_twh', 0), 3))
        iso_tr['uprate_price'].append(tc.get('uprate_price', 0))
        iso_tr['newbuild_price'].append(round(tc.get('newbuild_price', 0), 1))
        iso_tr['effective_cf_lcoe'].append(round(tc.get('effective_new_cf_lcoe', 0), 1))
    cf_tranche_data[iso] = iso_tr

# ============================================================================
# EXTRACT WYN_RESOURCE_COSTS
# ============================================================================

print("\nExtracting WYN_RESOURCE_COSTS...")
WYN_RESOURCES = ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro', 'battery', 'battery8', 'ldes', 'h2']
wyn_resource_costs = {}
for iso in ISOS:
    iso_wyn = []
    for t in THRESHOLDS:
        sc = data['results'][iso]['thresholds'].get(t, {}).get('scenarios', {}).get(medium_key(iso))
        rc = sc.get('costs_detail', {}).get('resource_costs', {}) if sc else {}
        entry = {}
        for res in WYN_RESOURCES:
            rd = rc.get(res, {})
            if res in ('battery', 'battery8', 'ldes', 'h2'):
                entry[res] = {
                    'dispatch_pct': rd.get('dispatch_pct', 0),
                    'cost': round(rd.get('cost_per_demand_mwh', 0), 2),
                }
            else:
                entry[res] = {
                    'existing_pct': round(rd.get('existing_pct', rd.get('existing_share', 0)), 1),
                    'new_pct': round(rd.get('new_pct', rd.get('new_share', 0)), 1),
                    'cost': round(rd.get('cost_per_demand_mwh', 0), 2),
                }
        iso_wyn.append(entry)
    wyn_resource_costs[iso] = iso_wyn

# ============================================================================
# EXTRACT GAS_BACKUP_DATA (from step4 postprocess gas_backup fields)
# ============================================================================

print("\nExtracting GAS_BACKUP_DATA...")
gas_backup_data = {}
for iso in ISOS:
    iso_gb = {
        'total_system_cost': [],
        'incremental_with_new_gas': [],
        'gas_backup_mw': [],
        'new_gas_build_mw': [],
        'existing_gas_used_mw': [],
        'clean_coverage_pct': [],
        'ra_peak_mw': [],
        'gas_backup_cost': [],
        'new_gas_cost': [],
        'existing_gas_cost': [],
    }
    for t in THRESHOLDS:
        sc = data['results'][iso]['thresholds'].get(t, {}).get('scenarios', {}).get(medium_key(iso))
        if sc and 'gas_backup' in sc:
            gb = sc['gas_backup']
            iso_gb['total_system_cost'].append(round(gb.get('total_system_cost_per_mwh', 0), 2))
            iso_gb['incremental_with_new_gas'].append(round(gb.get('incremental_with_new_gas', 0), 2))
            iso_gb['gas_backup_mw'].append(gb.get('gas_backup_needed_mw', 0))
            iso_gb['new_gas_build_mw'].append(gb.get('new_gas_build_mw', 0))
            iso_gb['existing_gas_used_mw'].append(gb.get('existing_gas_used_mw', 0))
            iso_gb['clean_coverage_pct'].append(gb.get('clean_coverage_pct', 0))
            iso_gb['ra_peak_mw'].append(gb.get('ra_peak_mw', 0))
            iso_gb['gas_backup_cost'].append(round(gb.get('gas_backup_cost_per_mwh', 0), 2))
            iso_gb['new_gas_cost'].append(round(gb.get('new_gas_cost_per_mwh', 0), 2))
            iso_gb['existing_gas_cost'].append(round(gb.get('existing_gas_cost_per_mwh', 0), 2))
        else:
            for key in iso_gb:
                iso_gb[key].append(0)
    gas_backup_data[iso] = iso_gb
    print(f"  {iso}: total_sys_cost={iso_gb['total_system_cost'][:3]}... gas_mw={iso_gb['gas_backup_mw'][:3]}...")

# ============================================================================
# FORMAT AS JAVASCRIPT
# ============================================================================

print("\nGenerating shared-data.js...")

def fmt_array(arr):
    return '[' + ', '.join('null' if v is None else str(v) for v in arr) + ']'

def fmt_24h_array(arr):
    return '[' + ','.join(str(v) for v in arr) + ']'

lines = []

# HEADER
lines.append('// ============================================================================')
lines.append('// SHARED DATA MODULE — Single source of truth for all dashboard pages')
lines.append('// ============================================================================')
lines.append('// RULE: No data constants defined in HTML files. Change here, propagates everywhere.')
lines.append(f'// Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")} by generate_shared_data.py')
lines.append('// Source: overprocure_results.json (Step 2 tranche-repriced + postprocess + CO2)')
lines.append('// ============================================================================')
lines.append('')

# THRESHOLDS
lines.append(f'// --- Thresholds (from optimizer) ---')
lines.append(f'const THRESHOLDS = {fmt_array(THRESHOLDS_NUM)};')
lines.append('')

# MAC_DATA
lines.append('// --- Average MAC ($/ton CO2) ---')
lines.append('// Medium = monotonic envelope of medium scenario')
lines.append('// Low/High = P10/P90 from 324-scenario factorial experiment')
lines.append('const MAC_DATA = {')
for sens in ['medium', 'low', 'high']:
    lines.append(f'    {sens}: {{')
    for iso_idx, iso in enumerate(ISOS):
        comma = ',' if iso_idx < len(ISOS) - 1 else ''
        lines.append(f'        {iso}:  {fmt_array(mac_data[sens][iso])}{comma}')
    comma = ',' if sens != 'high' else ''
    lines.append(f'    }}{comma}')
lines.append('};')
lines.append('')

# REGION_COLORS
lines.append('// --- Region Colors (used by abatement pages) ---')
lines.append("const REGION_COLORS = {")
lines.append("    CAISO: '#F59E0B',")
lines.append("    ERCOT: '#22C55E',")
lines.append("    PJM:   '#4a90d9',")
lines.append("    NYISO: '#E91E63',")
lines.append("    NEISO: '#9C27B0',")
lines.append("    MISO:  '#06B6D4',")
lines.append("    SPP:   '#EF4444'")
lines.append("};")
lines.append('')

# MIX_RESOURCES, MIX_LABELS_MAP, MIX_COLORS
lines.append("// --- Resource Colors & Labels (used by dashboard, index, region_deepdive) ---")
lines.append("const MIX_RESOURCES = ['clean_firm', 'geothermal', 'hydro', 'ccs_ccgt', 'offshore_wind', 'wind', 'solar', 'battery', 'battery8', 'ldes', 'h2'];")
lines.append('')
lines.append('const MIX_LABELS_MAP = {')
lines.append("    clean_firm:    'Nuclear',")
lines.append("    geothermal:    'Geothermal',")
lines.append("    hydro:         'Hydro',")
lines.append("    ccs_ccgt:      'CCS-CCGT',")
lines.append("    offshore_wind: 'Offshore Wind',")
lines.append("    wind:          'Onshore Wind',")
lines.append("    solar:         'Solar',")
lines.append("    battery:       'Battery (4hr)',")
lines.append("    battery8:      'Battery (8hr)',")
lines.append("    ldes:          'LDES (100hr)',")
lines.append("    h2:            'Green H\\u2082'")
lines.append('};')
lines.append('')
lines.append('const MIX_COLORS = {')
lines.append("    clean_firm:    { fill: 'rgba(99,102,241,0.50)',  border: '#6366F1' },")
lines.append("    geothermal:    { fill: 'rgba(180,83,9,0.50)',    border: '#B45309' },")
lines.append("    hydro:         { fill: 'rgba(14,165,233,0.50)',  border: '#0EA5E9' },")
lines.append("    ccs_ccgt:      { fill: 'rgba(100,116,139,0.50)', border: '#64748B' },")
lines.append("    offshore_wind: { fill: 'rgba(0,150,136,0.50)',   border: '#009688' },")
lines.append("    wind:          { fill: 'rgba(34,197,94,0.50)',   border: '#22C55E' },")
lines.append("    solar:         { fill: 'rgba(245,158,11,0.50)',  border: '#F59E0B' },")
lines.append("    battery:       { fill: 'rgba(139,92,246,0.50)',  border: '#8B5CF6' },")
lines.append("    battery8:      { fill: 'rgba(167,139,250,0.50)', border: '#A78BFA' },")
lines.append("    ldes:          { fill: 'rgba(236,72,153,0.50)',  border: '#EC4899' },")
lines.append("    h2:            { fill: 'rgba(16,185,129,0.50)',  border: '#10B981' }")
lines.append('};')
lines.append('')

# RESOURCE_CAPS — TWh caps for supply-constrained resources
lines.append('// --- Resource TWh Caps (physics/geological constraints) ---')
lines.append('const RESOURCE_CAPS = {')
lines.append('    offshore_wind: { NYISO: 37, NEISO: 37, PJM: 30, CAISO: 20 },')
lines.append('    ccs_ccgt: { ERCOT: 85, PJM: 120, NYISO: 15, NEISO: 10, MISO: 95, SPP: 110 },')
lines.append('    geothermal: { CAISO: 39 }')
lines.append('};')
lines.append('')

# RESOURCE_CAPS — TWh caps by resource × ISO
lines.append('// --- Resource TWh Caps (physics-constrained maximum deployment) ---')
lines.append('const RESOURCE_CAPS = {')
lines.append("    offshore_wind: { NYISO: 37, NEISO: 37, PJM: 30, CAISO: 20 },")
lines.append("    ccs_ccgt: { ERCOT: 85, PJM: 120, NYISO: 15, NEISO: 10, MISO: 95, SPP: 110 },")
lines.append("    geothermal: { CAISO: 39 }")
lines.append('};')
lines.append('')

# BENCHMARKS_STATIC
lines.append('// --- Benchmark Data (static — researched L/M/H with sources) ---')
lines.append('const BENCHMARKS_STATIC = [')
lines.append("    { name: 'Energy Efficiency (Buildings)', short: 'Energy Efficiency', low: -100, mid: 0,   high: 60,   color: '#4CAF50', category: 'demand_reduction', trajectory: 'stable', confidence: 'high',")
lines.append("      sources: 'EDF/Evolved Energy MAC 2.0, World Bank, Gillingham & Stock' },")
lines.append("    { name: 'EU ETS Price',                  short: 'EU ETS', low: 65,   mid: 88,  high: 92,   color: '#2196F3', category: 'benchmark', trajectory: 'rising', confidence: 'high',")
lines.append("      sources: 'Trading Economics, Sandbag, BNEF' },")
lines.append("    { name: 'SCC \\u2014 EPA ($190/ton)',     short: 'SCC (EPA)', low: 140,  mid: 190, high: 380,  color: '#FF9800', category: 'benchmark', trajectory: 'rising', confidence: 'medium',")
lines.append("      sources: 'EPA Dec 2023 Report' },")
lines.append("    { name: 'SCC \\u2014 Rennert et al.',     short: 'SCC (Rennert)', low: 120,  mid: 185, high: 450,  color: '#E65100', category: 'benchmark', trajectory: 'rising', confidence: 'medium',")
lines.append("      sources: 'Rennert et al. (2022) Nature' },")
lines.append("    { name: 'Carbon Credits (Nature)',       short: 'Carbon Credits', low: 3,    mid: 15,  high: 35,   color: '#9E9E9E', category: 'voluntary', trajectory: 'rising', confidence: 'medium',")
lines.append("      sources: 'Sylvera 2026, Regreener, MSCI' }")
lines.append('];')
lines.append('')

# BENCHMARKS_DYNAMIC
lines.append('// --- Benchmark Data (dynamic — shift with user toggles) ---')
lines.append('const BENCHMARKS_DYNAMIC = {')
lines.append("    dac: {")
lines.append("        name: 'Direct Air Capture (DAC)', short: 'DAC', color: '#E91E63', category: 'carbon_removal', confidence: 'low',")
lines.append("        trajectory: 'declining_steep',")
lines.append("        Low:    { low: 100,  mid: 150,  high: 200 },")
lines.append("        Medium: { low: 150,  mid: 250,  high: 350 },")
lines.append("        High:   { low: 250,  mid: 400,  high: 550 },")
lines.append("        sources: 'DOE Liftoff NOAK, IEAGHG 2021, Fasihi et al. (J. Cleaner Prod. 2019), Sievert et al. (Joule 2024), Climeworks Gen 3 roadmap, DOE Carbon Negative Shot, Kanyako & Craig (Earth\\'s Future 2025)'")
lines.append("    },")
lines.append("    industrial: {")
lines.append("        name: 'Industrial Electrification', short: 'Ind. Electrification', color: '#8BC34A', category: 'industrial_decarb', confidence: 'medium',")
lines.append("        trajectory: 'declining',")
lines.append("        Low:    { low: -50, mid: 20,  high: 60 },")
lines.append("        Medium: { low: 0,   mid: 60,  high: 160 },")
lines.append("        High:   { low: 60,  mid: 120, high: 250 },")
lines.append("        sources: 'McKinsey, Thunder Said Energy, Rewiring America'")
lines.append("    },")
lines.append("    removal: {")
lines.append("        name: 'Carbon Removal (BECCS + Enhanced Weathering)', short: 'Carbon Removal\\u00B9', color: '#009688', category: 'carbon_removal', confidence: 'low',")
lines.append("        trajectory: 'declining',")
lines.append("        Low:    { low: 20,  mid: 75,  high: 150 },")
lines.append("        Medium: { low: 50,  mid: 150, high: 300 },")
lines.append("        High:   { low: 100, mid: 200, high: 350 },")
lines.append("        sources: 'ORNL, Nature 2024, Nature Comms 2025'")
lines.append("    }")
lines.append('};')
lines.append('')

# BENCHMARKS_EXTRA
lines.append('// --- Extra Benchmarks (not toggle-controlled) ---')
lines.append('const BENCHMARKS_EXTRA = [')
lines.append("    { name: 'Green Hydrogen (Industrial)',  short: 'Green H\\u2082', low: 150, mid: 500, high: 1250, color: '#00BCD4', category: 'industrial_decarb',")
lines.append("      trajectory: 'declining', confidence: 'low', sources: 'Shafiee & Schrag (Joule 2024), Belfer Center' },")
lines.append("    { name: 'Sustainable Aviation Fuel',    short: 'SAF', low: 136, mid: 300, high: 500,  color: '#FF5722', category: 'transport',")
lines.append("      trajectory: 'declining', confidence: 'medium', sources: 'NREL, RMI 2025, ICCT, WEF' },")
lines.append("    { name: 'CDR Credits (Engineered)',     short: 'CDR Credits', low: 177, mid: 320, high: 600,  color: '#7C4DFF', category: 'voluntary',")
lines.append("      trajectory: 'declining', confidence: 'medium', sources: 'Sylvera, CarbonCredits.com' },")
lines.append("    { name: 'EVs vs ICE (Fleet)',           short: 'EVs vs ICE', low: -50, mid: 250, high: 970,  color: '#FF6F00', category: 'transport',")
lines.append("      trajectory: 'declining_steep', confidence: 'medium', sources: 'Argonne Labs, Penn Wharton, RFF' }")
lines.append('];')
lines.append('')

# MARGINAL_MAC_LABELS + MARGINAL_MAC_DATA
lines.append("// --- Six-Zone Marginal MAC ($/ton CO2) ---")
lines.append("// Zone 0 (50→75%): entry-level aggregate MAC")
lines.append("// Zone 1 (75→90%): backbone aggregate MAC")
lines.append("// Zones 2-5 (90→99%): granular steps with monotonicity enforcement")
lines.append("// Cap: $1000/ton (NREL literature max)")
lines.append("const MARGINAL_MAC_LABELS = ['50→75%', '75→90%', '90→92.5%', '92.5→95%', '95→97.5%', '97.5→99%'];")
lines.append('')
lines.append('const MARGINAL_MAC_DATA = {')
for sens in ['medium', 'low', 'high']:
    lines.append(f'    {sens}: {{')
    for iso_idx, iso in enumerate(ISOS):
        comma = ',' if iso_idx < len(ISOS) - 1 else ''
        lines.append(f'        {iso}:  {fmt_array(marginal_mac_data[sens][iso])}{comma}')
    comma = ',' if sens != 'high' else ''
    lines.append(f'    }}{comma}')
lines.append('};')
lines.append('')

# EFFECTIVE_COST_DATA
lines.append('// --- Effective Cost per Useful MWh ($/MWh) ---')
lines.append('// Source: Step 2 tranche-repriced medium key (9-dim) + postprocess corrections')
lines.append('// Monotonicity enforced (lower thresholds <= higher thresholds)')
thresh_str = ', '.join(str(t) for t in THRESHOLDS_NUM)
lines.append(f'// Indices match THRESHOLDS array: [{thresh_str}]')
lines.append('const EFFECTIVE_COST_DATA = {')
for iso_idx, iso in enumerate(ISOS):
    comma = ',' if iso_idx < len(ISOS) - 1 else ''
    lines.append(f'    {iso}:  {fmt_array(effective_cost_data[iso])}{comma}')
lines.append('};')
lines.append('')

# SYSTEM_COST_DATA
lines.append('// --- Average System Cost ($/MWh) by threshold ---')
lines.append('// Source: P10/Median/P90 of total_system_cost_per_mwh across 324 sensitivity scenarios')
lines.append('// Includes clean procurement + gas backup (existing + new-build)')
lines.append(f'// Indices match THRESHOLDS array: [{thresh_str}]')
lines.append('const SYSTEM_COST_DATA = {')
for sens in ['medium', 'low', 'high']:
    lines.append(f'    {sens}: {{')
    for iso_idx, iso in enumerate(ISOS):
        comma = ',' if iso_idx < len(ISOS) - 1 else ''
        lines.append(f'        {iso}: {" " * (6-len(iso))}{fmt_array(system_cost_data[sens][iso])}{comma}')
    comma = ',' if sens != 'high' else ''
    lines.append(f'    }}{comma}')
lines.append('};')
lines.append('')

# CLEAN_COST_DATA — effective_cost only (NO gas backup)
lines.append('// --- Clean Procurement Cost ($/MWh) by threshold ---')
lines.append('// Source: P10/Median/P90 of effective_cost across sensitivity scenarios')
lines.append('// Does NOT include gas backup — used for MAC/crossover analysis')
lines.append(f'// Indices match THRESHOLDS array: [{thresh_str}]')
lines.append('const CLEAN_COST_DATA = {')
for sens in ['medium', 'low', 'high']:
    lines.append(f'    {sens}: {{')
    for iso_idx, iso in enumerate(ISOS):
        comma = ',' if iso_idx < len(ISOS) - 1 else ''
        lines.append(f'        {iso}: {" " * (6-len(iso))}{fmt_array(clean_cost_data[sens][iso])}{comma}')
    comma = ',' if sens != 'high' else ''
    lines.append(f'    }}{comma}')
lines.append('};')
lines.append('')

# UPRATE_CAPS_TWH
uprate_caps = data.get('config', {}).get('tranche_model', {}).get('uprate_caps_twh', {})
if not uprate_caps:
    uprate_caps = {'CAISO': 0.907, 'ERCOT': 1.064, 'PJM': 12.614, 'NYISO': 1.340, 'NEISO': 1.380}
lines.append("// --- Nuclear Uprate Caps (TWh/yr) — 5% of existing nuclear at 90% CF ---")
lines.append("const UPRATE_CAPS_TWH = {")
parts = [f'    {iso}: {uprate_caps.get(iso, 0)}' for iso in ISOS]
lines.append(',\n'.join(parts))
lines.append('};')
lines.append('')

# UTILITY FUNCTIONS
lines.append('// ============================================================================')
lines.append('// SHARED UTILITY FUNCTIONS')
lines.append('// ============================================================================')
lines.append('')
lines.append('function findCrossover(regionData, costLevel) {')
lines.append('    for (let i = 0; i < regionData.length; i++) {')
lines.append('        if (regionData[i] !== null && regionData[i] >= costLevel) return THRESHOLDS[i];')
lines.append('    }')
lines.append("    return '>99';")
lines.append('}')
lines.append('')
lines.append('const MARGINAL_THRESHOLDS = [50, 75, 90, 92.5, 95, 97.5];')
lines.append('function findMarginalCrossover(regionMarginals, costLevel) {')
lines.append('    for (let i = 0; i < regionMarginals.length; i++) {')
lines.append('        if (regionMarginals[i] !== null && regionMarginals[i] > costLevel) {')
lines.append('            return MARGINAL_THRESHOLDS[i];')
lines.append('        }')
lines.append('    }')
lines.append("    return '>99';")
lines.append('}')
lines.append('')
lines.append('function cellClass(val) {')
lines.append("    if (val === '>99' || val === '>100') return 'cell-green';")
lines.append("    if (val >= 97) return 'cell-green';")
lines.append("    if (val >= 95) return 'cell-yellow';")
lines.append("    if (val >= 92) return 'cell-orange';")
lines.append("    return 'cell-red';")
lines.append('}')
lines.append('')
lines.append('function getAllBenchmarks(state) {')
lines.append("    state = state || { dac: 'Medium', industrial: 'Medium', removal: 'Medium' };")
lines.append('    const dynamic = Object.keys(BENCHMARKS_DYNAMIC).map(key => {')
lines.append('        const b = BENCHMARKS_DYNAMIC[key];')
lines.append('        const costs = b[state[key]] || b.Medium;')
lines.append('        return { name: b.name, short: b.short || b.name, low: costs.low, mid: costs.mid, high: costs.high,')
lines.append('                 color: b.color, category: b.category, confidence: b.confidence, trajectory: b.trajectory, sources: b.sources };')
lines.append('    });')
lines.append('    return [...BENCHMARKS_STATIC, ...dynamic, ...BENCHMARKS_EXTRA]')
lines.append('        .filter(b => b.mid >= 0)')
lines.append('        .sort((a, b) => a.mid - b.mid);')
lines.append('}')
lines.append('')

# RESOURCE_MIX_DATA
lines.append('')
lines.append('// --- Resource Mix (% of demand) — Medium sensitivity × Medium demand growth ---')
lines.append('// Source: Step 3 demand-growth parquets (step3_dg_{ISO}_t{threshold}.parquet)')
lines.append('// Each threshold maps to an SBTi year; demand grows at Medium rate to that year.')
lines.append(f'// Indices match THRESHOLDS array: [{thresh_str}]')
lines.append('// battery/battery8/ldes/h2 = dispatch % of demand')
lines.append('const RESOURCE_MIX_DATA = {')
for iso_idx, iso in enumerate(ISOS):
    d = resource_mix_data[iso]
    lines.append(f'    {iso}: {{')
    for key in RESOURCES + ['battery', 'battery8', 'ldes', 'h2']:
        comma = ',' if key != 'h2' else ''
        padding = ' ' * max(0, 12 - len(key))
        lines.append(f'        {key}:{padding}{fmt_array(d[key])}{comma}')
    comma = ',' if iso_idx < len(ISOS) - 1 else ''
    lines.append(f'    }}{comma}')
lines.append('};')

# DEMAND_TWH_BY_THRESHOLD — demand TWh at each threshold's SBTi target year (Medium growth)
lines.append('')
lines.append('// --- Demand TWh at each threshold\'s SBTi year (Medium demand growth) ---')
lines.append('// Source: Step 3 demand-growth parquets (annual_demand_mwh / 1e6)')
lines.append('// Use in WYN panel: multiply resource % by this TWh instead of base 2025 demand.')
lines.append(f'// Indices match THRESHOLDS array: [{thresh_str}]')
lines.append('const DEMAND_TWH_BY_THRESHOLD = {')
for iso_idx, iso in enumerate(ISOS):
    dt = demand_twh_by_threshold[iso]
    comma = ',' if iso_idx < len(ISOS) - 1 else ''
    lines.append(f'    {iso}: {fmt_array(dt)}{comma}')
lines.append('};')

# HOURLY_MATCH_SCORES — actual hourly match score of the cost-optimal mix per threshold
lines.append('')
lines.append('// --- Hourly Match Scores (%) for cost-optimal mix at each threshold ---')
lines.append('// The cost-optimal mix may over-achieve the target threshold.')
lines.append('// Use this for accurate utilized/curtailed splits in the WYN chart.')
lines.append(f'// Indices match THRESHOLDS array: [{thresh_str}]')
lines.append('const HOURLY_MATCH_SCORES = {')
for iso_idx, iso in enumerate(ISOS):
    ms = hourly_match_scores.get(iso, [0] * len(THRESHOLDS_NUM))
    comma = ',' if iso_idx < len(ISOS) - 1 else ''
    lines.append(f'    {iso}: {fmt_array(ms)}{comma}')
lines.append('};')

# COMPRESSED_DAY_DATA
lines.append('')
lines.append('// --- Compressed Day Hourly Profiles (24h normalized) — medium scenario ---')
lines.append('// Source: overprocure_results.json compressed_day field')
lines.append(f'// Each sub-array is [24 hourly values] in UTC, one per threshold (matching THRESHOLDS)')
lines.append('const COMPRESSED_DAY_DATA = {')
for iso_idx, iso in enumerate(ISOS):
    cd = compressed_day_data[iso]
    lines.append(f'    {iso}: {{')

    # demand
    lines.append('        demand: [')
    for i, arr in enumerate(cd['demand']):
        comma = ',' if i < len(cd['demand']) - 1 else ''
        lines.append(f'            {fmt_24h_array(arr)}{comma}')
    lines.append('        ],')

    # matched
    lines.append('        matched: {')
    for res_idx, res in enumerate(MATCHED_RESOURCES):
        res_comma = ',' if res_idx < len(MATCHED_RESOURCES) - 1 else ''
        lines.append(f'            {res}: [')
        for i, arr in enumerate(cd['matched'][res]):
            comma = ',' if i < len(cd['matched'][res]) - 1 else ''
            lines.append(f'                {fmt_24h_array(arr)}{comma}')
        lines.append(f'            ]{res_comma}')
    lines.append('        },')

    # surplus
    surplus_resources = ['clean_firm', 'solar', 'wind', 'ccs_ccgt', 'hydro']
    lines.append('        surplus: {')
    for res_idx, res in enumerate(surplus_resources):
        res_comma = ',' if res_idx < len(surplus_resources) - 1 else ''
        lines.append(f'            {res}: [')
        for i, arr in enumerate(cd['surplus'][res]):
            comma = ',' if i < len(cd['surplus'][res]) - 1 else ''
            lines.append(f'                {fmt_24h_array(arr)}{comma}')
        lines.append(f'            ]{res_comma}')
    lines.append('        },')

    # gap
    lines.append('        gap: [')
    for i, arr in enumerate(cd['gap']):
        comma = ',' if i < len(cd['gap']) - 1 else ''
        lines.append(f'            {fmt_24h_array(arr)}{comma}')
    lines.append('        ],')

    # battery_charge
    lines.append('        battery_charge: [')
    for i, arr in enumerate(cd['battery_charge']):
        comma = ',' if i < len(cd['battery_charge']) - 1 else ''
        lines.append(f'            {fmt_24h_array(arr)}{comma}')
    lines.append('        ],')

    # battery8_charge
    lines.append('        battery8_charge: [')
    for i, arr in enumerate(cd['battery8_charge']):
        comma = ',' if i < len(cd['battery8_charge']) - 1 else ''
        lines.append(f'            {fmt_24h_array(arr)}{comma}')
    lines.append('        ],')

    # ldes_charge
    lines.append('        ldes_charge: [')
    for i, arr in enumerate(cd['ldes_charge']):
        comma = ',' if i < len(cd['ldes_charge']) - 1 else ''
        lines.append(f'            {fmt_24h_array(arr)}{comma}')
    lines.append('        ]')

    iso_comma = ',' if iso_idx < len(ISOS) - 1 else ''
    lines.append(f'    }}{iso_comma}')
lines.append('};')

# CF_TRANCHE_DATA
lines.append('')
lines.append('// --- CF Tranche Split (uprate vs new-build) — medium scenario ---')
lines.append('// Source: overprocure_results.json tranche_costs field')
lines.append(f'// Indices match THRESHOLDS array: [{thresh_str}]')
lines.append('const CF_TRANCHE_DATA = {')
for iso_idx, iso in enumerate(ISOS):
    tr = cf_tranche_data[iso]
    lines.append(f'    {iso}: {{')
    fields = ['new_cf_twh', 'cf_existing_twh', 'uprate_twh', 'geo_twh',
              'nuclear_newbuild_twh', 'ccs_tranche_twh',
              'uprate_price', 'newbuild_price', 'effective_cf_lcoe']
    for fi, field in enumerate(fields):
        comma = ',' if fi < len(fields) - 1 else ''
        padding = ' ' * max(0, 18 - len(field))
        lines.append(f'        {field}:{padding}{fmt_array(tr[field])}{comma}')
    iso_comma = ',' if iso_idx < len(ISOS) - 1 else ''
    lines.append(f'    }}{iso_comma}')
lines.append('};')

# WYN_RESOURCE_COSTS
lines.append('')
lines.append('// --- WYN Resource Costs (existing/new/cost per resource) — medium scenario ---')
lines.append('// Source: overprocure_results.json costs_detail.resource_costs')
lines.append(f'// Array of {len(THRESHOLDS)} objects per ISO (one per threshold in THRESHOLDS order)')
lines.append('const WYN_RESOURCE_COSTS = {')
for iso_idx, iso in enumerate(ISOS):
    lines.append(f'    {iso}: [')
    for ti, entry in enumerate(wyn_resource_costs[iso]):
        parts = []
        for res in WYN_RESOURCES:
            rd = entry[res]
            if 'dispatch_pct' in rd:
                parts.append(f'{res}:{{d:{rd["dispatch_pct"]},c:{rd["cost"]}}}')
            else:
                parts.append(f'{res}:{{e:{rd["existing_pct"]},n:{rd["new_pct"]},c:{rd["cost"]}}}')
        comma = ',' if ti < len(wyn_resource_costs[iso]) - 1 else ''
        lines.append(f'        {{{", ".join(parts)}}}{comma}')
    iso_comma = ',' if iso_idx < len(ISOS) - 1 else ''
    lines.append(f'    ]{iso_comma}')
lines.append('};')

# GAS_BACKUP_DATA
lines.append('')
lines.append('// --- Gas Capacity Backup & Resource Adequacy --- ')
lines.append('// Source: step4_postprocess.py compute_gas_capacity_and_ra()')
lines.append(f'// Indices match THRESHOLDS array: [{thresh_str}]')
lines.append('// total_system_cost = clean_cost + gas_backup_cost (existing+new)')
lines.append('// incremental_with_new_gas = clean_cost + new-build gas cost only')
lines.append('const GAS_BACKUP_DATA = {')
gb_fields = ['total_system_cost', 'incremental_with_new_gas', 'gas_backup_mw',
             'new_gas_build_mw', 'existing_gas_used_mw', 'clean_coverage_pct',
             'ra_peak_mw', 'gas_backup_cost', 'new_gas_cost', 'existing_gas_cost']
for iso_idx, iso in enumerate(ISOS):
    gb = gas_backup_data[iso]
    lines.append(f'    {iso}: {{')
    for fi, field in enumerate(gb_fields):
        comma = ',' if fi < len(gb_fields) - 1 else ''
        padding = ' ' * max(0, 26 - len(field))
        lines.append(f'        {field}:{padding}{fmt_array(gb[field])}{comma}')
    iso_comma = ',' if iso_idx < len(ISOS) - 1 else ''
    lines.append(f'    }}{iso_comma}')
lines.append('};')

# ============================================================================
# COST TABLES FOR CLIENT-SIDE REPRICING (new toggles: CCS/45Q/Geothermal)
# ============================================================================

print("\nExtracting cost tables for client-side repricing...")

tranche_model = data.get('config', {}).get('tranche_model', {})

lines.append('')
lines.append('// ============================================================================')
lines.append('// CLIENT-SIDE REPRICING — Cost tables for CCS/45Q/Geothermal toggles')
lines.append('// ============================================================================')
lines.append('// The dashboard uses these to reprice feasible mixes when new toggles change.')
lines.append('// This avoids pre-computing 40k+ sensitivity combos.')
lines.append('')

# Geothermal cap (alias used by some dashboard code)
lines.append(f'const GEOTHERMAL_CAP_TWH = {_GEO_CAP_TWH};')
lines.append('')

# Uprate LCOE (from Step 3 constants)
lines.append('// --- Nuclear Uprate LCOE ($/MWh) by Nuclear toggle ---')
lines.append(f'const UPRATE_LCOE = {{ L: {_UPRATE_LCOE.get("L", 15)}, M: {_UPRATE_LCOE.get("M", 25)}, H: {_UPRATE_LCOE.get("H", 40)} }};')
lines.append('')

# Wholesale prices (from Step 3 constants — single source of truth)
lines.append('// --- Wholesale Prices ($/MWh) ---')
parts = [f'{iso}: {_WHOLESALE_PRICES.get(iso, 0)}' for iso in ISOS]
lines.append(f'const WHOLESALE_PRICES = {{ {", ".join(parts)} }};')
lines.append('')

# Fuel adjustments
lines.append('// --- Fossil Fuel Price Adjustments ($/MWh delta from wholesale) ---')
lines.append('const FUEL_ADJUSTMENTS = {')
for iso_idx, iso in enumerate(ISOS):
    fa = _FUEL_ADJUSTMENTS.get(iso, {})
    parts = [f'{lev}: {fa.get(lev, 0)}' for lev in ['Low', 'Medium', 'High']]
    comma = ',' if iso_idx < len(ISOS) - 1 else ''
    lines.append(f'    {iso}: {{ {", ".join(parts)} }}{comma}')
lines.append('};')
lines.append('')

# LCOE tables (solar, wind, battery, battery8, LDES, H2 — from Step 3 constants)
lines.append('// --- LCOE Tables ($/MWh) for client-side repricing ---')
lines.append('const LCOE_TABLES = {')
for res in ['solar', 'wind', 'battery', 'battery8', 'ldes', 'h2']:
    rt = _LCOE_TABLES.get(res, {})
    lines.append(f'    {res}: {{')
    for lev_idx, lev in enumerate(['Low', 'Medium', 'High']):
        vals = rt.get(lev, {})
        parts = [f'{iso}: {vals.get(iso, 0)}' for iso in ISOS]
        comma = ',' if lev_idx < 2 else ''
        lines.append(f'        {lev}: {{ {", ".join(parts)} }}{comma}')
    res_comma = ',' if res != 'h2' else ''
    lines.append(f'    }}{res_comma}')
lines.append('};')
lines.append('')

# Battery 8hr storage
lines.append('// --- Battery 8hr storage flag ---')
lines.append('const HAS_BATTERY8 = true;')
lines.append('')

# Transmission tables (from Step 3 constants)
lines.append('// --- Transmission Adders ($/MWh) ---')
lines.append('const TX_TABLES = {')
for res_idx, res in enumerate(['solar', 'wind', 'clean_firm', 'ccs_ccgt', 'battery', 'battery8', 'ldes', 'h2']):
    rt = _TX_TABLES.get(res, {})
    lines.append(f'    {res}: {{')
    tx_levels = ['None', 'Low', 'Medium', 'High']
    for lev_idx, lev in enumerate(tx_levels):
        vals = rt.get(lev, {})
        if isinstance(vals, (int, float)):
            parts = [f'{iso}: {vals}' for iso in ISOS]
        else:
            parts = [f'{iso}: {vals.get(iso, 0)}' for iso in ISOS]
        comma = ',' if lev_idx < len(tx_levels) - 1 else ''
        lines.append(f'        {lev}: {{ {", ".join(parts)} }}{comma}')
    res_comma = ',' if res_idx < 7 else ''
    lines.append(f'    }}{res_comma}')
lines.append('};')
lines.append('')

# Nuclear new-build LCOE
lines.append('// --- Nuclear New-Build LCOE ($/MWh) by firm gen toggle ---')
lines.append('const NUCLEAR_NEWBUILD_LCOE = {')
for lev_idx, lev in enumerate(['L', 'M', 'H']):
    vals = _NUCLEAR_NEWBUILD_LCOE.get(lev, {})
    parts = [f'{iso}: {vals.get(iso, 0)}' for iso in ISOS]
    comma = ',' if lev_idx < 2 else ''
    lines.append(f'    {lev}: {{ {", ".join(parts)} }}{comma}')
lines.append('};')
lines.append('')

# CCS LCOE (with and without 45Q)
lines.append('// --- CCS-CCGT LCOE ($/MWh) with/without 45Q ---')
lines.append('const CCS_LCOE_45Q_ON = {')
for lev_idx, lev in enumerate(['L', 'M', 'H']):
    vals = _CCS_LCOE_45Q_ON.get(lev, {})
    parts = [f'{iso}: {vals.get(iso, 0)}' for iso in ISOS]
    comma = ',' if lev_idx < 2 else ''
    lines.append(f'    {lev}: {{ {", ".join(parts)} }}{comma}')
lines.append('};')
lines.append('const CCS_LCOE_45Q_OFF = {')
for lev_idx, lev in enumerate(['L', 'M', 'H']):
    vals = _CCS_LCOE_45Q_OFF.get(lev, {})
    parts = [f'{iso}: {vals.get(iso, 0)}' for iso in ISOS]
    comma = ',' if lev_idx < 2 else ''
    lines.append(f'    {lev}: {{ {", ".join(parts)} }}{comma}')
lines.append('};')
lines.append('')

# Geothermal (CAISO only)
lines.append('// --- Geothermal LCOE ($/MWh, CAISO only) ---')
lines.append(f'const GEOTHERMAL_LCOE = {{ L: {_GEOTHERMAL_LCOE["L"]}, M: {_GEOTHERMAL_LCOE["M"]}, H: {_GEOTHERMAL_LCOE["H"]} }};')
lines.append(f'const GEO_CAP_TWH = {_GEO_CAP_TWH};')
lines.append('')

# Grid mix shares (existing clean generation — from Step 3 constants)
lines.append('// --- Grid Mix Shares (% of demand — existing generation) ---')
lines.append('// This is the EXISTING clean energy floor: priced at wholesale, not LCOE.')
lines.append('const GRID_MIX_SHARES = {')
for iso_idx, iso in enumerate(ISOS):
    shares = _GRID_MIX_SHARES.get(iso, {})
    parts = [f'{res}: {shares.get(res, 0)}' for res in RESOURCES]
    comma = ',' if iso_idx < len(ISOS) - 1 else ''
    lines.append(f'    {iso}: {{ {", ".join(parts)} }}{comma}')
lines.append('};')
lines.append('')

# Regional demand (TWh — from Step 3 constants, falling back to co2 batch results)
lines.append('// --- Regional Annual Demand (TWh) ---')
lines.append('const REGIONAL_DEMAND_TWH = {')
for iso_idx, iso in enumerate(ISOS):
    # Prefer co2 batch value (has the actual year's demand), fall back to Step 3 constant
    demand_twh = round(data.get('results', {}).get(iso, {}).get('annual_demand_mwh', 0) / 1e6, 3)
    if demand_twh == 0:
        demand_twh = _REGIONAL_DEMAND_TWH.get(iso, 0)
    comma = ',' if iso_idx < len(ISOS) - 1 else ''
    lines.append(f'    {iso}: {demand_twh}{comma}')
lines.append('};')
lines.append('')

# ============================================================================
# FEASIBLE MIXES FOR CLIENT-SIDE REPRICING
# ============================================================================

# DEMAND_GROWTH_RATES (static — from ISO forecasts, EIA/FERC)
lines.append('// --- Demand Growth Rates (annual, ISO-specific) ---')
lines.append('// Sources: ISO forecasts, EIA/FERC data, regional load growth studies')
lines.append('const DEMAND_GROWTH_RATES = {')
lines.append("    CAISO:  { Low: 0.014, Medium: 0.019, High: 0.025, label: 'CA electrification-driven' },")
lines.append("    ERCOT:  { Low: 0.020, Medium: 0.035, High: 0.055, label: 'TX data centers + population' },")
lines.append("    PJM:    { Low: 0.015, Medium: 0.024, High: 0.036, label: 'Data center corridor' },")
lines.append("    NYISO:  { Low: 0.013, Medium: 0.020, High: 0.044, label: 'CLCPA mandate-driven' },")
lines.append("    NEISO:  { Low: 0.009, Medium: 0.018, High: 0.029, label: 'Heating electrification' }")
lines.append('};')
lines.append('')

# CLEAN_FIRM_ENERGY_SPLIT (static — from eGRID capacity factors)
lines.append('// --- Clean Firm Energy Split (nuclear vs geothermal) ---')
lines.append("// CAISO: 70% nuclear (avg CF 0.893) + 30% geothermal (flat CF 1.0).")
lines.append("// Energy fraction = capacity_share × CF / blended_CF.  Other ISOs: 100% nuclear.")
lines.append('const CLEAN_FIRM_ENERGY_SPLIT = {')
lines.append("    CAISO:  { nuclear: 0.676, geothermal: 0.324 },")
lines.append("    ERCOT:  { nuclear: 1.0,   geothermal: 0.0 },")
lines.append("    PJM:    { nuclear: 1.0,   geothermal: 0.0 },")
lines.append("    NYISO:  { nuclear: 1.0,   geothermal: 0.0 },")
lines.append("    NEISO:  { nuclear: 1.0,   geothermal: 0.0 },")
lines.append("    MISO:   { nuclear: 1.0,   geothermal: 0.0 },")
lines.append("    SPP:    { nuclear: 1.0,   geothermal: 0.0 }")
lines.append('};')
lines.append('')

# ============================================================================
# SBTi TIMELINE + DAC LEARNING CURVE
# ============================================================================

print("\nGenerating SBTi timeline + DAC trajectory data...")

# Canonical threshold → target year mapping (SBTi-interpolated, from Step 3)
# Each of 15 thresholds maps to the year when it's expected to be achieved
try:
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    from step3_cost_optimization import THRESHOLD_TARGET_YEARS
except ImportError:
    THRESHOLD_TARGET_YEARS = {
        50: 2030, 55: 2031, 60: 2033, 65: 2034, 70: 2035, 75: 2036, 80: 2037,
        85: 2038, 87.5: 2039, 90: 2040, 92.5: 2043,
        95: 2045, 97.5: 2048, 99: 2049, 99.99: 2050,
    }

# SBTi milestone anchors (sparse, for dashboard display)
SBTI_MILESTONES = [
    {'year': 2025, 'threshold': 0,   'label': 'Today'},
    {'year': 2030, 'threshold': 50,  'label': 'SBTi 50%'},
    {'year': 2035, 'threshold': 70,  'label': 'SBTi ~70%'},
    {'year': 2040, 'threshold': 90,  'label': 'SBTi 90%'},
    {'year': 2045, 'threshold': 95,  'label': 'SBTi ~95%'},
    {'year': 2050, 'threshold': 99.99, 'label': 'Net-Zero'},
]

# DAC cost projections: anchored to 2025 actuals ($600-$1,500/tCO₂)
# $/ton CO₂ net DACCS (capture + transport + storage + MRV), 2024 USD
# Sources: Climeworks, IEAGHG, Belfer Center (Harvard), Sievert et al. (Joule 2024)
DAC_TRAJECTORY = {
    'optimistic':   {2025: 600, 2030: 350, 2035: 230, 2040: 175, 2045: 130, 2050: 100},
    'central':      {2025: 800, 2030: 500, 2035: 375, 2040: 300, 2045: 250, 2050: 200},
    'conservative': {2025: 1100, 2030: 750, 2035: 550, 2040: 450, 2045: 375, 2050: 300},
}

lines.append('// ============================================================================')
lines.append('// SBTi TIMELINE + DAC LEARNING CURVE (SPEC.md §7.3)')
lines.append('// ============================================================================')
lines.append('// Maps clean energy thresholds to SBTi target years, with projected DAC costs')
lines.append('// declining along literature-sourced learning curves.')
lines.append('// Sources: DOE Liftoff (2023), Sievert et al. (Joule 2024), IEA (2022/2024),')
lines.append('//   Fasihi et al. (2019), IEAGHG (2021/2024), Climeworks Gen 3, DOE Carbon')
lines.append('//   Negative Shot, Kanyako & Craig (2025), NAS (2019), Young et al. (2023),')
lines.append('//   Keith et al. (2018), Shayegh et al. (2021), Belfer Center (2023).')
lines.append('')
lines.append('const SBTI_MILESTONES = [')
for ms in SBTI_MILESTONES:
    lines.append(f"    {{ year: {ms['year']}, threshold: {ms['threshold']}, label: '{ms['label']}' }},")
lines.append('];')
lines.append('')

# Full threshold → year mapping (all 15 thresholds, SBTi-interpolated)
lines.append('// Full threshold-year mapping: each of 15 thresholds paired with its target year')
lines.append('// Interpolated between SBTi anchors (50%→2030, 70%→2035, 90%→2040, 95%→2045, ≥99.99%→2050)')
lines.append('const THRESHOLD_TARGET_YEARS = {')
for t in THRESHOLDS_NUM:
    year = THRESHOLD_TARGET_YEARS.get(t, 2050)
    lines.append(f'    {t}: {year},')
lines.append('};')
lines.append('')
lines.append('// DAC cost trajectories ($/ton CO₂ net DACCS, 2024 USD)')
lines.append('// Optimistic: 15-20% learning rate, R&D breakthroughs, <$20/MWh renewables')
lines.append('// Central: 10-12% learning rate, moderate policy, $30-40/MWh renewables')
lines.append('// Conservative: 5-8% learning rate, limited policy, $40-60/MWh renewables')
lines.append('const DAC_TRAJECTORY = {')
for traj_name, traj_data in DAC_TRAJECTORY.items():
    years = sorted(traj_data.keys())
    year_parts = [f'{y}: {traj_data[y]}' for y in years]
    comma = ',' if traj_name != 'conservative' else ''
    lines.append(f'    {traj_name}: {{ {", ".join(year_parts)} }}{comma}')
lines.append('};')
lines.append('')

# Interpolation helper: given a threshold, find the corresponding SBTi year and DAC cost
# For thresholds between milestones, linearly interpolate year, then lookup DAC cost
lines.append('// Interpolate DAC cost for any threshold via SBTi year mapping')
lines.append('function dacCostAtThreshold(threshold, trajectory) {')
lines.append('    trajectory = trajectory || "central";')
lines.append('    const traj = DAC_TRAJECTORY[trajectory];')
lines.append('    const ms = SBTI_MILESTONES;')
lines.append('    // Find bounding milestones')
lines.append('    let year;')
lines.append('    if (threshold <= ms[0].threshold) year = ms[0].year;')
lines.append('    else if (threshold >= ms[ms.length-1].threshold) year = ms[ms.length-1].year;')
lines.append('    else {')
lines.append('        for (let i = 0; i < ms.length - 1; i++) {')
lines.append('            if (threshold >= ms[i].threshold && threshold <= ms[i+1].threshold) {')
lines.append('                const frac = (threshold - ms[i].threshold) / (ms[i+1].threshold - ms[i].threshold);')
lines.append('                year = ms[i].year + frac * (ms[i+1].year - ms[i].year);')
lines.append('                break;')
lines.append('            }')
lines.append('        }')
lines.append('    }')
lines.append('    // Interpolate DAC cost at that year')
lines.append('    const years = Object.keys(traj).map(Number).sort((a,b) => a-b);')
lines.append('    if (year <= years[0]) return traj[years[0]];')
lines.append('    if (year >= years[years.length-1]) return traj[years[years.length-1]];')
lines.append('    for (let i = 0; i < years.length - 1; i++) {')
lines.append('        if (year >= years[i] && year <= years[i+1]) {')
lines.append('            const frac = (year - years[i]) / (years[i+1] - years[i]);')
lines.append('            return Math.round(traj[years[i]] + frac * (traj[years[i+1]] - traj[years[i]]));')
lines.append('        }')
lines.append('    }')
lines.append('    return traj[years[years.length-1]];')
lines.append('}')
lines.append('')

# ============================================================================
# DEMAND GROWTH COUNTERFACTUAL MAC (SPEC.md §7.2)
# ============================================================================

print("\nComputing demand growth counterfactual MAC...")

# Regional demand (TWh) and growth rates (from step3)
REGIONAL_DEMAND_TWH_PY = {
    'CAISO': 224.039, 'ERCOT': 488.020, 'PJM': 843.331,
    'NYISO': 151.599, 'NEISO': 115.336,
}
DEMAND_GROWTH_RATES_PY = {
    'CAISO':  {'Low': 0.014, 'Medium': 0.019, 'High': 0.025},
    'ERCOT':  {'Low': 0.020, 'Medium': 0.035, 'High': 0.055},
    'PJM':    {'Low': 0.015, 'Medium': 0.024, 'High': 0.036},
    'NYISO':  {'Low': 0.013, 'Medium': 0.020, 'High': 0.044},
    'NEISO':  {'Low': 0.009, 'Medium': 0.018, 'High': 0.029},
}
NEW_GAS_EMISSION_RATE = 0.35  # tCO₂/MWh (new CCGT counterfactual)

# For each threshold/target-year pair, compute growth MWh and counterfactual emissions
# This replaces the sparse SBTi-only calculation with all 15 thresholds
growth_counterfactual = {}
for iso in ISOS:
    iso_cf = {}
    base_twh = REGIONAL_DEMAND_TWH_PY.get(iso, 0)
    for t in THRESHOLDS_NUM:
        year = THRESHOLD_TARGET_YEARS.get(t, 2050)
        level_data = {}
        for level in ['Low', 'Medium', 'High']:
            rate = DEMAND_GROWTH_RATES_PY.get(iso, {}).get(level, 0.02)
            years_from_base = year - 2025
            if years_from_base <= 0:
                growth_twh = 0
                gf = 1.0
            else:
                gf = (1 + rate) ** years_from_base
                grown_twh = base_twh * gf
                growth_twh = grown_twh - base_twh
            growth_mwh = growth_twh * 1e6
            cf_tons = growth_mwh * NEW_GAS_EMISSION_RATE
            level_data[level] = {
                'growth_twh': round(growth_twh, 1),
                'counterfactual_mt': round(cf_tons / 1e6, 2),
                'growth_factor': round(gf, 4),
                'demand_mwh': round(base_twh * gf * 1e6, 0),
            }
        iso_cf[str(t)] = {'year': year, **level_data}
    growth_counterfactual[iso] = iso_cf
    t90_data = iso_cf.get('90', {}).get('Medium', {})
    print(f"  {iso}: 90% (2040) Medium growth={t90_data.get('growth_twh', 0)} TWh, "
          f"counterfactual={t90_data.get('counterfactual_mt', 0)} MtCO₂")

lines.append('// ============================================================================')
lines.append('// DEMAND GROWTH COUNTERFACTUAL (threshold-year paired)')
lines.append('// ============================================================================')
lines.append('// Each threshold paired with its SBTi target year. Without clean procurement,')
lines.append('// demand growth MWh would be met by new gas at 350 kg/MWh (0.35 tCO₂/MWh).')
lines.append('// growth_twh = baseTWh × ((1 + rate)^(year-2025) - 1)')
lines.append('// Keyed by threshold (not year) for easy lookup alongside cost/mix data.')
lines.append(f'const NEW_GAS_EMISSION_RATE = {NEW_GAS_EMISSION_RATE}; // tCO₂/MWh')
lines.append('')
lines.append('const GROWTH_COUNTERFACTUAL = {')
for iso_idx, iso in enumerate(ISOS):
    lines.append(f'    {iso}: {{')
    for t_idx, t in enumerate(THRESHOLDS_NUM):
        t_str = str(t) if t != int(t) else str(int(t))
        td = growth_counterfactual[iso][str(t)]
        year = td['year']
        t_comma = ',' if t_idx < len(THRESHOLDS_NUM) - 1 else ''
        lines.append(
            f'        "{t_str}": {{ year: {year}, '
            f'Low: {{ twh: {td["Low"]["growth_twh"]}, mt: {td["Low"]["counterfactual_mt"]}, gf: {td["Low"]["growth_factor"]} }}, '
            f'Medium: {{ twh: {td["Medium"]["growth_twh"]}, mt: {td["Medium"]["counterfactual_mt"]}, gf: {td["Medium"]["growth_factor"]} }}, '
            f'High: {{ twh: {td["High"]["growth_twh"]}, mt: {td["High"]["counterfactual_mt"]}, gf: {td["High"]["growth_factor"]} }} }}{t_comma}')
    iso_comma = ',' if iso_idx < len(ISOS) - 1 else ''
    lines.append(f'    }}{iso_comma}')
lines.append('};')
lines.append('')

# DG MAC data from mac_stats.json (if available)
dg_mac = mac_stats.get('demand_growth_mac', {})
if dg_mac:
    print("Adding demand growth MAC data to shared-data.js...")
    lines.append('// ============================================================================')
    lines.append('// DEMAND GROWTH MAC (threshold-year paired, P10/P50/P90)')
    lines.append('// ============================================================================')
    lines.append('// MAC percentiles for demand growth scenarios at each threshold/year.')
    lines.append('// Keyed: ISO → threshold → growth_level → {mac_p10, mac_p50, mac_p90, year, gf}')
    lines.append(f'const DG_MAC = {json.dumps(dg_mac)};')
    lines.append('')
else:
    print("  No DG MAC data in mac_stats.json — skipping")

# ============================================================================
# OPTIMAL TARGETS (from step6_compute_optimal_targets.py)
# ============================================================================
if optimal_targets:
    print("Adding optimal targets data to shared-data.js...")
    lines.append('// ============================================================================')
    lines.append('// OPTIMAL CFE TARGETS — MAC × DAC crossover analysis')
    lines.append('// Source: step6_compute_optimal_targets.py → optimal_targets.json')
    lines.append('// Crossover range = 3 grid cost tiers × 3 DAC scenarios = 9 combos')
    lines.append('// No-regrets: minimum resource floor across crossover range × L/M/H demand growth')
    lines.append('// ============================================================================')
    lines.append('')
    lines.append('const OPTIMAL_TARGETS = {')
    for iso_idx, iso in enumerate(ISOS):
        iso_data = optimal_targets.get(iso, {})
        if not iso_data:
            continue
        lines.append(f'    {iso}: {{')

        # Crossover range
        rng = iso_data.get('crossover_range', {})
        lines.append(f'        crossover_range: {json.dumps(rng)},')

        # Central crossover (medium grid × central DAC)
        cen = iso_data.get('crossovers', {}).get('medium_grid__central_dac', {})
        lines.append(f'        crossover_central: {json.dumps(cen)},')

        # All 9 crossovers
        lines.append(f'        crossovers: {json.dumps(iso_data.get("crossovers", {}))},')

        # Smooth MAC curves for charting
        sc = iso_data.get('smooth_curve', {})
        lines.append(f'        smooth_thresholds: {json.dumps(sc.get("thresholds", []))},')
        lines.append(f'        smooth_marginal_mac: {json.dumps(sc.get("marginal_mac", []))},')

        # L/M/H sensitivity band
        lmh = iso_data.get('smooth_curves_lmh', {})
        for tier in ['low', 'medium', 'high']:
            tc = lmh.get(tier, {})
            lines.append(f'        smooth_mac_{tier}: {json.dumps(tc.get("marginal_mac", []))},')

        # Discrete data points
        dp = iso_data.get('discrete_points', {})
        lines.append(f'        discrete_thresholds: {json.dumps(dp.get("thresholds", []))},')
        lines.append(f'        discrete_mac: {json.dumps(dp.get("mac_at_thresholds", []))},')

        # No-regrets investments
        nr = iso_data.get('no_regrets', {})
        if isinstance(nr, dict) and 'resources' in nr:
            nr_summary = {}
            for res in ['clean_firm', 'solar', 'wind', 'ccs_ccgt', 'hydro', 'battery', 'battery8', 'ldes', 'h2']:
                stats = nr.get('resources', {}).get(res, {})
                if stats.get('max_pct', 0) == 0:
                    continue
                nr_summary[res] = {
                    'floor_pct': stats.get('floor_pct'),
                    'avg_pct': stats.get('avg_pct'),
                    'max_pct': stats.get('max_pct'),
                    'is_consensus': stats.get('is_consensus'),
                    'twh_by_growth': stats.get('twh_by_growth'),
                }
            lines.append(f'        no_regrets: {json.dumps(nr_summary)},')
            lines.append(f'        no_regrets_thresholds: {json.dumps(nr.get("range_thresholds", []))},')
            lines.append(f'        total_clean_by_growth: {json.dumps(nr.get("total_clean_by_growth", {}))},')
        else:
            lines.append(f'        no_regrets: null,')

        # Demand growth rates
        lines.append(f'        demand_growth_rates: {json.dumps(iso_data.get("demand_growth_rates", {}))},')

        comma = ',' if iso_idx < len(ISOS) - 1 else ''
        lines.append(f'    }}{comma}')

    lines.append('};')
    lines.append('')
    print(f"  Added OPTIMAL_TARGETS for {len([i for i in ISOS if i in optimal_targets])} ISOs")
else:
    print("  Skipping OPTIMAL_TARGETS — not yet computed")

print("Extracting FEASIBLE_MIXES from Step 3 feasible-mix parquets...")
# Strategy: Step 3 already generates step3_feasible_{ISO}.parquet files containing
# up to 500 EF-sampled mixes per threshold plus all archetype winners.  These are
# the candidate set the dashboard needs for client-side repricing.
#
# For t=99.99: practically unreachable — we use mixes from the feasible set
# that score >=99.5% (matching Step 3's effective_gate logic).

import numpy as np
import pandas as pd

STEP3_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'step3-cost-opt-parquets')
MIX_FIELDS = ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro',
              'hourly_match_score',
              'battery_dispatch_pct', 'battery8_dispatch_pct', 'ldes_dispatch_pct',
              'h2_dispatch_pct']

lines.append('// --- Feasible Mixes per (ISO, threshold) for client-side repricing ---')
lines.append('// Each mix: [clean_firm%, solar%, wind%, offshore_wind%, ccs_ccgt%, hydro%, match%, battery%, battery8%, ldes%, h2%]')
lines.append('// Source: step3_feasible_{ISO}.parquet — ~500 EF-sampled mixes + archetype winners per threshold.')
lines.append('// t=99.99 uses mixes scoring >=99.5% from the feasible set (effective gate).')
lines.append('const FEASIBLE_MIXES = {')
total_fm = 0
for iso_idx, iso in enumerate(ISOS):
    lines.append(f'    {iso}: {{')

    # Load the feasible parquet for this ISO (all thresholds in one file)
    feasible_path = os.path.join(STEP3_DIR, f'step3_feasible_{iso}.parquet')
    if os.path.exists(feasible_path):
        feas_df = pd.read_parquet(feasible_path)
        # Ensure all expected columns exist
        for col in MIX_FIELDS:
            if col not in feas_df.columns:
                feas_df[col] = 0
        print(f"  {iso}: loaded {len(feas_df)} feasible mixes from step3_feasible_{iso}.parquet")
    else:
        feas_df = pd.DataFrame()
        print(f"  WARNING: step3_feasible_{iso}.parquet not found — {iso} will have empty FEASIBLE_MIXES")

    for t_idx, t in enumerate(THRESHOLDS):
        t_num = float(t)

        if t_num == 99.99:
            # t=99.99: use mixes from t=99 feasible set that score >=99.5%
            if len(feas_df) > 0:
                # Combined filter: threshold==99 AND score>=99.5 in one pass
                sub = feas_df[(feas_df['threshold'] == 99.0) & (feas_df['hourly_match_score'] >= 99.5)]
                if sub.empty:
                    # Fallback: highest-scoring mixes from ANY threshold
                    sub = feas_df[feas_df['hourly_match_score'] >= 99.5]
            else:
                sub = pd.DataFrame()
        else:
            sub = feas_df[feas_df['threshold'] == t_num] if len(feas_df) > 0 else pd.DataFrame()

        mix_rows = []
        if len(sub) > 0:
            # Vectorized: extract columns as numpy arrays, round, then convert to list of lists
            mix_cols = ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro',
                        'hourly_match_score', 'battery_dispatch_pct',
                        'battery8_dispatch_pct', 'ldes_dispatch_pct', 'h2_dispatch_pct']
            # Backward compat: add offshore_wind column if missing
            if 'offshore_wind' not in sub.columns:
                sub = sub.copy()
                sub['offshore_wind'] = 0
            arr = sub[mix_cols].to_numpy(copy=False)
            arr[:, 6] = np.round(arr[:, 6], 1)  # round hourly_match_score (col 6 now, was col 5)
            mix_rows = arr.tolist()

        total_fm += len(mix_rows)
        lines.append(f'        "{t}": [')
        for m_idx, arr in enumerate(mix_rows):
            comma = ',' if m_idx < len(mix_rows) - 1 else ''
            lines.append(f'            [{",".join(str(v) for v in arr)}]{comma}')
        t_comma = ',' if t_idx < len(THRESHOLDS) - 1 else ''
        lines.append(f'        ]{t_comma}')
    iso_comma = ',' if iso_idx < len(ISOS) - 1 else ''
    lines.append(f'    }}{iso_comma}')
lines.append('};')
lines.append('')

print(f"  Feasible mixes: {total_fm} total across {len(ISOS)} ISOs × {len(THRESHOLDS)} thresholds")

# ============================================================================
# WRITE OUTPUT
# ============================================================================

js_content = '\n'.join(lines) + '\n'
output_path = os.path.join(SCRIPT_DIR, '..', 'dashboard', 'js', 'shared-data.js')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    f.write(js_content)

file_size = os.path.getsize(output_path)
print(f"\nWrote {output_path}")
print(f"  Lines: {len(lines)}")
print(f"  Size: {file_size:,} bytes ({file_size/1024:.1f} KB)")

# Archive canonical JSON to step5 results directory
os.makedirs(STEP5_DIR, exist_ok=True)
shared_json_path = os.path.join(STEP5_DIR, 'shared_data.json')
shared_json = {
    'thresholds': THRESHOLDS_NUM,
    'isos': ISOS,
    'mac_data': mac_data,
    'marginal_mac_data': marginal_mac_data,
    'effective_cost_data': effective_cost_data,
    'system_cost_data': system_cost_data,
    'clean_cost_data': clean_cost_data,
    'resource_mix_data': resource_mix_data,
    'gas_backup_data': gas_backup_data,
    'cf_tranche_data': cf_tranche_data,
    'growth_counterfactual': growth_counterfactual,
}
with open(shared_json_path, 'w') as f:
    json.dump(shared_json, f, indent=2)
shared_size = os.path.getsize(shared_json_path)
print(f"Archived: {shared_json_path} ({shared_size:,} bytes, {shared_size/1024:.1f} KB)")

# ============================================================================
# VERIFICATION
# ============================================================================

print("\nVerification:")
print(f"  Thresholds: {len(THRESHOLDS_NUM)} ({THRESHOLDS_NUM[0]}% - {THRESHOLDS_NUM[-1]}%)")

for iso in ISOS:
    rm = resource_mix_data[iso]
    for i in range(len(THRESHOLDS)):
        total = sum(rm[r][i] for r in RESOURCES)
        if total != 100:
            print(f"  WARNING: {iso} threshold {THRESHOLDS[i]} resource mix sums to {total}")

    # Check effective cost monotonicity
    ec = effective_cost_data[iso]
    for i in range(len(ec) - 1):
        if ec[i] is not None and ec[i+1] is not None and ec[i] > ec[i+1]:
            print(f"  WARNING: {iso} effective cost NOT monotonic at {THRESHOLDS[i]}: {ec[i]} > {ec[i+1]}")

    # Check compressed_day data exists for all thresholds
    cd = compressed_day_data[iso]
    zero_count = sum(1 for d in cd['demand'] if all(v == 0 for v in d))
    if zero_count > 0:
        print(f"  WARNING: {iso} has {zero_count} empty compressed_day entries")

print("\nDone.")
