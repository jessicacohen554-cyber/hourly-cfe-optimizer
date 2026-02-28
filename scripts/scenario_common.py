#!/usr/bin/env python3
"""
Shared constants, cost functions, and utilities for scenario comparison scripts.

Used by:
  - step6_scenario_a.py (Scenario A: Pure Consequential)
  - step6_scenario_b.py (Scenario B: Hourly Matching)
  - step6_scenario_compare.py (comparison and output)
  - step2_5_expand_ef_for_floors.py (EF expansion)
"""

import json
import os
import re
import sys
import datetime
import numpy as np
import pandas as pd
from pathlib import Path

# Import dispatch utilities
sys.path.insert(0, str(Path(__file__).parent))
from dispatch_utils import (
    compute_fossil_retirement, compute_co2_from_dispatch,
    BASE_DEMAND_TWH, GRID_MIX_SHARES,
    COAL_CAP_TWH, OIL_CAP_TWH, COAL_OIL_RETIREMENT_THRESHOLD,
    H, load_common_data, get_supply_profiles, get_demand_profile,
    build_supply_matrix, reconstruct_hourly_dispatch,
    _archetype_key, load_dispatch_cache, get_or_compute_dispatch,
)

# ============================================================================
# CONSTANTS (from step3_cost_optimization.py)
# ============================================================================

from step3_cost_optimization import OUTPUT_THRESHOLDS as THRESHOLDS

ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']
RESOURCES = ['clean_firm', 'solar', 'wind', 'ccs_ccgt', 'hydro']

HYDRO_CAP_TWH = {iso: GRID_MIX_SHARES[iso].get('hydro', 0) / 100.0 * BASE_DEMAND_TWH[iso]
                 for iso in ISOS}

WHOLESALE_PRICES = {'CAISO': 30, 'ERCOT': 27, 'PJM': 34, 'NYISO': 42, 'NEISO': 41, 'MISO': 30, 'SPP': 25}
FUEL_ADJUSTMENTS = {
    'CAISO': {'Low': -5, 'Medium': 0, 'High': 10},
    'ERCOT': {'Low': -7, 'Medium': 0, 'High': 12},
    'PJM':   {'Low': -6, 'Medium': 0, 'High': 11},
    'NYISO': {'Low': -4, 'Medium': 0, 'High': 8},
    'NEISO': {'Low': -4, 'Medium': 0, 'High': 8},
    'MISO':  {'Low': -6, 'Medium': 0, 'High': 11},
    'SPP':   {'Low': -7, 'Medium': 0, 'High': 12},
}

LCOE_TABLES = {
    'solar': {
        'Low':    {'CAISO': 45, 'ERCOT': 40, 'PJM': 50, 'NYISO': 70, 'NEISO': 62, 'MISO': 48, 'SPP': 43},
        'Medium': {'CAISO': 60, 'ERCOT': 54, 'PJM': 65, 'NYISO': 92, 'NEISO': 82, 'MISO': 62, 'SPP': 57},
        'High':   {'CAISO': 78, 'ERCOT': 70, 'PJM': 85, 'NYISO': 120, 'NEISO': 107, 'MISO': 82, 'SPP': 74},
    },
    'wind': {
        'Low':    {'CAISO': 55, 'ERCOT': 30, 'PJM': 47, 'NYISO': 61, 'NEISO': 55, 'MISO': 33, 'SPP': 28},
        'Medium': {'CAISO': 73, 'ERCOT': 40, 'PJM': 62, 'NYISO': 81, 'NEISO': 73, 'MISO': 43, 'SPP': 37},
        'High':   {'CAISO': 95, 'ERCOT': 52, 'PJM': 81, 'NYISO': 105, 'NEISO': 95, 'MISO': 56, 'SPP': 48},
    },
    'battery': {
        'Low':    {'CAISO': 77, 'ERCOT': 69, 'PJM': 74, 'NYISO': 81, 'NEISO': 79, 'MISO': 72, 'SPP': 70},
        'Medium': {'CAISO': 102, 'ERCOT': 92, 'PJM': 98, 'NYISO': 108, 'NEISO': 105, 'MISO': 96, 'SPP': 93},
        'High':   {'CAISO': 133, 'ERCOT': 120, 'PJM': 127, 'NYISO': 140, 'NEISO': 137, 'MISO': 125, 'SPP': 121},
    },
    'battery8': {
        'Low':    {'CAISO': 85, 'ERCOT': 77, 'PJM': 82, 'NYISO': 90, 'NEISO': 88, 'MISO': 80, 'SPP': 78},
        'Medium': {'CAISO': 125, 'ERCOT': 113, 'PJM': 120, 'NYISO': 132, 'NEISO': 129, 'MISO': 117, 'SPP': 115},
        'High':   {'CAISO': 165, 'ERCOT': 149, 'PJM': 159, 'NYISO': 175, 'NEISO': 170, 'MISO': 155, 'SPP': 152},
    },
    'ldes': {
        'Low':    {'CAISO': 135, 'ERCOT': 116, 'PJM': 128, 'NYISO': 150, 'NEISO': 143, 'MISO': 122, 'SPP': 118},
        'Medium': {'CAISO': 180, 'ERCOT': 155, 'PJM': 170, 'NYISO': 200, 'NEISO': 190, 'MISO': 162, 'SPP': 158},
        'High':   {'CAISO': 234, 'ERCOT': 202, 'PJM': 221, 'NYISO': 260, 'NEISO': 247, 'MISO': 211, 'SPP': 205},
    },
}

TX_TABLES = {
    'wind':       {'None': 0, 'Low': {'CAISO': 4, 'ERCOT': 3, 'PJM': 5, 'NYISO': 7, 'NEISO': 6, 'MISO': 5, 'SPP': 4},
                   'Medium': {'CAISO': 8, 'ERCOT': 6, 'PJM': 10, 'NYISO': 14, 'NEISO': 12, 'MISO': 9, 'SPP': 7},
                   'High': {'CAISO': 14, 'ERCOT': 10, 'PJM': 18, 'NYISO': 22, 'NEISO': 20, 'MISO': 16, 'SPP': 12}},
    'solar':      {'None': 0, 'Low': {'CAISO': 1, 'ERCOT': 1, 'PJM': 2, 'NYISO': 3, 'NEISO': 3, 'MISO': 2, 'SPP': 1},
                   'Medium': {'CAISO': 3, 'ERCOT': 3, 'PJM': 5, 'NYISO': 7, 'NEISO': 6, 'MISO': 4, 'SPP': 3},
                   'High': {'CAISO': 6, 'ERCOT': 5, 'PJM': 9, 'NYISO': 12, 'NEISO': 10, 'MISO': 8, 'SPP': 6}},
    'clean_firm': {'None': 0, 'Low': {'CAISO': 1, 'ERCOT': 1, 'PJM': 1, 'NYISO': 2, 'NEISO': 2, 'MISO': 1, 'SPP': 1},
                   'Medium': {'CAISO': 3, 'ERCOT': 2, 'PJM': 3, 'NYISO': 5, 'NEISO': 4, 'MISO': 3, 'SPP': 2},
                   'High': {'CAISO': 6, 'ERCOT': 4, 'PJM': 6, 'NYISO': 9, 'NEISO': 7, 'MISO': 5, 'SPP': 4}},
    'ccs_ccgt':   {'None': 0, 'Low': {'CAISO': 1, 'ERCOT': 1, 'PJM': 1, 'NYISO': 2, 'NEISO': 2, 'MISO': 1, 'SPP': 1},
                   'Medium': {'CAISO': 2, 'ERCOT': 2, 'PJM': 3, 'NYISO': 4, 'NEISO': 3, 'MISO': 2, 'SPP': 2},
                   'High': {'CAISO': 4, 'ERCOT': 3, 'PJM': 5, 'NYISO': 7, 'NEISO': 6, 'MISO': 4, 'SPP': 3}},
    'battery':    {'None': 0, 'Low': {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 1, 'NEISO': 1, 'MISO': 0, 'SPP': 0},
                   'Medium': {'CAISO': 1, 'ERCOT': 1, 'PJM': 1, 'NYISO': 2, 'NEISO': 2, 'MISO': 1, 'SPP': 1},
                   'High': {'CAISO': 2, 'ERCOT': 2, 'PJM': 3, 'NYISO': 4, 'NEISO': 3, 'MISO': 2, 'SPP': 2}},
    'battery8':   {'None': 0, 'Low': {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 1, 'NEISO': 1, 'MISO': 0, 'SPP': 0},
                   'Medium': {'CAISO': 1, 'ERCOT': 1, 'PJM': 1, 'NYISO': 2, 'NEISO': 2, 'MISO': 1, 'SPP': 1},
                   'High': {'CAISO': 2, 'ERCOT': 2, 'PJM': 3, 'NYISO': 4, 'NEISO': 3, 'MISO': 2, 'SPP': 2}},
    'ldes':       {'None': 0, 'Low': {'CAISO': 1, 'ERCOT': 1, 'PJM': 1, 'NYISO': 2, 'NEISO': 2, 'MISO': 1, 'SPP': 1},
                   'Medium': {'CAISO': 2, 'ERCOT': 2, 'PJM': 3, 'NYISO': 4, 'NEISO': 3, 'MISO': 2, 'SPP': 2},
                   'High': {'CAISO': 4, 'ERCOT': 3, 'PJM': 5, 'NYISO': 7, 'NEISO': 6, 'MISO': 4, 'SPP': 3}},
    'hydro':      {'None': 0, 'Low': 0, 'Medium': 0, 'High': 0},
}

UPRATE_LCOE = {'L': 15, 'M': 25, 'H': 40}
NUCLEAR_NEWBUILD_LCOE = {
    'L': {'CAISO': 70, 'ERCOT': 68, 'PJM': 72, 'NYISO': 75, 'NEISO': 73, 'MISO': 70, 'SPP': 68},
    'M': {'CAISO': 95, 'ERCOT': 90, 'PJM': 105, 'NYISO': 110, 'NEISO': 108, 'MISO': 100, 'SPP': 92},
    'H': {'CAISO': 140, 'ERCOT': 135, 'PJM': 160, 'NYISO': 170, 'NEISO': 165, 'MISO': 155, 'SPP': 140},
}
GEOTHERMAL_LCOE = {'L': 63, 'M': 88, 'H': 110}
GEO_CAP_TWH = 39.0
CCS_LCOE_45Q_ON = {
    'L': {'CAISO': 58, 'ERCOT': 52, 'PJM': 62, 'NYISO': 78, 'NEISO': 75, 'MISO': 55, 'SPP': 50},
    'M': {'CAISO': 86, 'ERCOT': 71, 'PJM': 79, 'NYISO': 99, 'NEISO': 96, 'MISO': 74, 'SPP': 68},
    'H': {'CAISO': 115, 'ERCOT': 92, 'PJM': 102, 'NYISO': 128, 'NEISO': 122, 'MISO': 96, 'SPP': 88},
}

EXISTING_NUCLEAR_GW = {'CAISO': 2.3, 'ERCOT': 2.7, 'PJM': 32.0, 'NYISO': 3.4, 'NEISO': 3.5, 'MISO': 12.0, 'SPP': 1.2}
UPRATE_CAP_TWH = {iso: round(gw * 0.08 * 0.90 * 8760 / 1e3, 3)
                  for iso, gw in EXISTING_NUCLEAR_GW.items()}

PEAK_DEMAND_MW = {'CAISO': 43860, 'ERCOT': 83597, 'PJM': 160560, 'NYISO': 31857, 'NEISO': 25898, 'MISO': 127125, 'SPP': 54368}
EXISTING_GAS_CAPACITY_MW = {'CAISO': 37000, 'ERCOT': 55000, 'PJM': 75000, 'NYISO': 18000, 'NEISO': 14000, 'MISO': 68000, 'SPP': 32000}
NEW_CCGT_COST_KW_YR = {'CAISO': 112, 'ERCOT': 89, 'PJM': 99, 'NYISO': 114, 'NEISO': 105, 'MISO': 95, 'SPP': 88}
EXISTING_GAS_FOM_KW_YR = {'CAISO': 16, 'ERCOT': 13, 'PJM': 14, 'NYISO': 17, 'NEISO': 15, 'MISO': 14, 'SPP': 13}
PEAK_CAPACITY_CREDITS = {
    'clean_firm': 1.0, 'solar': 0.30, 'wind': 0.10, 'ccs_ccgt': 0.90,
    'hydro': 0.50, 'battery': 0.95, 'battery8': 0.95, 'ldes': 0.90,
}
GAS_AVAILABILITY_FACTOR = {
    'CAISO': 0.88, 'ERCOT': 0.83, 'PJM': 0.82, 'NYISO': 0.82, 'NEISO': 0.85,
    'MISO': 0.84, 'SPP': 0.84,
}
RESOURCE_ADEQUACY_MARGIN = 0.15

LEVEL_NAME = {'L': 'Low', 'M': 'Medium', 'H': 'High'}

# SBTi timeline mapping
try:
    from step3_cost_optimization import THRESHOLD_TARGET_YEARS
    SBTI_YEAR_MAP = THRESHOLD_TARGET_YEARS
except ImportError:
    SBTI_YEAR_MAP = {
        50: 2030, 55: 2031, 60: 2033, 65: 2034, 70: 2035, 75: 2036, 80: 2037,
        85: 2038, 87.5: 2039, 90: 2040, 92.5: 2043,
        95: 2045, 97.5: 2048, 99: 2049, 99.5: 2049, 99.9: 2050, 100: 2050,
    }

DEMAND_GROWTH_RATES = {
    'CAISO': 0.019, 'ERCOT': 0.035, 'PJM': 0.024,
    'NYISO': 0.020, 'NEISO': 0.018,
    'MISO': 0.022, 'SPP': 0.018,
}
BASE_YEAR = 2025

ZONES = [
    {'label': '50→55%', 'start': 50, 'end': 55},
    {'label': '55→60%', 'start': 55, 'end': 60},
    {'label': '60→65%', 'start': 60, 'end': 65},
    {'label': '65→70%', 'start': 65, 'end': 70},
    {'label': '70→75%', 'start': 70, 'end': 75},
    {'label': '75→90%', 'start': 75, 'end': 90},
    {'label': '90→95%', 'start': 90, 'end': 95},
    {'label': '95→97.5%', 'start': 95, 'end': 97.5},
    {'label': '97.5→99%', 'start': 97.5, 'end': 99},
    {'label': '99→99.5%', 'start': 99, 'end': 99.5},
    {'label': '99.5→99.9%', 'start': 99.5, 'end': 99.9},
    {'label': '99.9→100%', 'start': 99.9, 'end': 100},
]

# ============================================================================
# SCENARIO DEFINITIONS
# ============================================================================

SCENARIO_A = {
    'name': 'Pure Consequential',
    'short': 'pure_consequential',
    'description': 'Sequential procurement chasing cheapest $/tCO₂ — no clean firm investment means FOAK prices when firm is finally needed',
    'toggles': {
        'ren': 'L', 'firm': 'H', 'batt': 'L', 'ldes_lvl': 'H',
        'fuel': 'M', 'tx': 'M', 'ccs': 'H', 'q45': '1', 'geo': 'H',
    },
    'uprate_level': 'M',
}

SCENARIO_B = {
    'name': 'Hourly Matching',
    'short': 'hourly_matching',
    'description': 'Forward-looking: optimal 2045 endpoint defines deployment trajectory. Nuclear/LDES FOAK(H)→NOAK(L). CCS FOAK(H)→NOAK(M) with 45Q. Early firm investment yields lower long-term system cost.',
    'toggles': {
        'ren': 'L', 'firm': 'H', 'batt': 'L', 'ldes_lvl': 'H',
        'fuel': 'M', 'tx': 'M', 'ccs': 'H', 'q45': '1', 'geo': 'H',
    },
    'uprate_level': 'M',
}

SCENARIOS = [SCENARIO_A, SCENARIO_B]

# Max PFS mixes to evaluate per fallback pass (prevents timeout on large ISOs)
MAX_PFS_EVAL = 500


# ============================================================================
# DEMAND & LEARNING CURVE
# ============================================================================

def get_demand_twh(iso, threshold):
    """Get demand TWh for a given ISO and threshold, accounting for medium demand growth."""
    year = SBTI_YEAR_MAP.get(threshold, 2050)
    years = max(0, year - BASE_YEAR)
    rate = DEMAND_GROWTH_RATES[iso]
    return BASE_DEMAND_TWH[iso] * (1 + rate) ** years


def get_demand_growth_factor(iso, threshold):
    """Get the demand growth factor relative to BASE_YEAR."""
    year = SBTI_YEAR_MAP.get(threshold, 2050)
    years = max(0, year - BASE_YEAR)
    rate = DEMAND_GROWTH_RATES[iso]
    return (1 + rate) ** years


def learning_fraction(threshold, scenario='B'):
    """Map CFE threshold to FOAK→NOAK learning curve fraction [0, 1]."""
    year = SBTI_YEAR_MAP.get(threshold, 2050)
    if scenario == 'B':
        foak_start, noak_year = 2030, 2040
    else:
        foak_start, noak_year = 2035, 2047
    if year < foak_start:
        return 0.0
    if year >= noak_year:
        return 1.0
    active = (year - foak_start) / (noak_year - foak_start)
    return active ** 0.6


# ============================================================================
# PARSE FEASIBLE MIXES
# ============================================================================

def _load_feasible_from_parquet(iso, step3_dir='data/step3-cost-opt-parquets'):
    """Load feasible mixes for a single ISO from the step3 parquet file."""
    feasible_path = os.path.join(step3_dir, f'step3_feasible_{iso}.parquet')
    if not os.path.exists(feasible_path):
        return {}
    df = pd.read_parquet(feasible_path)
    mix_fields = ['clean_firm', 'solar', 'wind', 'ccs_ccgt', 'hydro',
                  'hourly_match_score',
                  'battery_dispatch_pct', 'battery8_dispatch_pct',
                  'ldes_dispatch_pct', 'h2_dispatch_pct']
    for col in mix_fields:
        if col not in df.columns:
            df[col] = 0
    result = {}
    for t_val, grp in df.groupby('threshold'):
        t_str = str(int(t_val)) if t_val == int(t_val) else str(t_val)
        rows = []
        for _, row in grp.iterrows():
            rows.append([
                row['clean_firm'], row['solar'], row['wind'],
                row['ccs_ccgt'], row['hydro'],
                round(row['hourly_match_score'], 1),
                row['battery_dispatch_pct'], row['battery8_dispatch_pct'],
                row['ldes_dispatch_pct'], row['h2_dispatch_pct'],
            ])
        result[t_str] = rows
    return result


def parse_feasible_mixes(js_path='dashboard/js/shared-data.js'):
    """Parse FEASIBLE_MIXES from shared-data.js, with parquet fallback."""
    with open(js_path) as f:
        content = f.read()
    start = content.find('const FEASIBLE_MIXES = {')
    if start < 0:
        raise ValueError("FEASIBLE_MIXES not found in shared-data.js")
    brace_count = 0
    i = content.index('{', start)
    for j in range(i, len(content)):
        if content[j] == '{':
            brace_count += 1
        elif content[j] == '}':
            brace_count -= 1
            if brace_count == 0:
                end = j + 1
                break
    block = content[i:end]
    block = re.sub(r'(\b[A-Z]+\b)\s*:', r'"\1":', block)
    block = re.sub(r',\s*([}\]])', r'\1', block)
    mixes = json.loads(block)
    expected_thresholds = {str(int(t)) if t == int(t) else str(t) for t in THRESHOLDS}
    for iso in ISOS:
        iso_data = mixes.get(iso, {})
        has_mixes = any(len(v) > 0 for v in iso_data.values()) if iso_data else False
        if not has_mixes:
            parquet_mixes = _load_feasible_from_parquet(iso)
            if parquet_mixes:
                mixes[iso] = parquet_mixes
                total = sum(len(v) for v in parquet_mixes.values())
                print(f"  Loaded {iso} feasible mixes from parquet fallback: {total} mixes")
            else:
                print(f"  WARNING: No feasible mixes for {iso} in shared-data.js or parquets")
        else:
            present_thresholds = set(iso_data.keys())
            missing = expected_thresholds - present_thresholds
            if missing:
                parquet_mixes = _load_feasible_from_parquet(iso)
                if parquet_mixes:
                    filled = []
                    for t_str in sorted(missing):
                        if t_str in parquet_mixes and parquet_mixes[t_str]:
                            mixes[iso][t_str] = parquet_mixes[t_str]
                            filled.append(t_str)
                    if filled:
                        print(f"  Backfilled {iso} thresholds from parquet: {', '.join(filled)}")
    return mixes


# ============================================================================
# COST FUNCTIONS
# ============================================================================

def get_tx(rtype, tx_level, iso):
    val = TX_TABLES[rtype][tx_level]
    if isinstance(val, dict):
        return val[iso]
    return val


def _resource_new_build_lcoe(res, sens, iso):
    """Get new-build LCOE+transmission for a resource under given sensitivity toggles."""
    ren_name = LEVEL_NAME[sens['ren']]
    batt_name = LEVEL_NAME[sens['batt']]
    ldes_name = LEVEL_NAME[sens['ldes_lvl']]
    tx_name = LEVEL_NAME[sens['tx']]
    firm_lev = sens['firm']
    ccs_lev = sens['ccs']
    q45 = sens['q45']
    if res == 'solar':
        return LCOE_TABLES['solar'][ren_name][iso] + get_tx('solar', tx_name, iso)
    elif res == 'wind':
        return LCOE_TABLES['wind'][ren_name][iso] + get_tx('wind', tx_name, iso)
    elif res == 'clean_firm':
        return NUCLEAR_NEWBUILD_LCOE[firm_lev][iso] + get_tx('clean_firm', tx_name, iso)
    elif res == 'ccs_ccgt':
        ccs_table = CCS_LCOE_45Q_ON if q45 == '1' else None
        return ccs_table[ccs_lev][iso] + get_tx('ccs_ccgt', tx_name, iso)
    elif res == 'hydro':
        return 0
    elif res == 'battery':
        return LCOE_TABLES['battery'][batt_name][iso] + get_tx('battery', tx_name, iso)
    elif res == 'ldes':
        return LCOE_TABLES['ldes'][ldes_name][iso] + get_tx('ldes', tx_name, iso)
    return 0


def compute_mix_cost(mix, sens, iso, demand_twh, overrides=None, growth_factor=1.0):
    """Compute total system cost per MWh for a single mix under a sensitivity scenario."""
    cf_pct, sol_pct, wnd_pct, ccs_pct, hyd_pct = mix[0], mix[1], mix[2], mix[3], mix[4]
    match_score = mix[5]
    bat4_pct, bat8_pct, ldes_pct = mix[6], mix[7], mix[8]
    h2_pct = mix[9] if len(mix) > 9 else 0
    match_frac = match_score / 100.0
    ren_name = LEVEL_NAME[sens['ren']]
    batt_name = LEVEL_NAME[sens['batt']]
    ldes_name = LEVEL_NAME[sens['ldes_lvl']]
    fuel_name = LEVEL_NAME[sens['fuel']]
    tx_name = LEVEL_NAME[sens['tx']]
    firm_lev = sens['firm']
    ccs_lev = sens['ccs']
    q45 = sens['q45']
    geo_lev = sens.get('geo')
    existing = {k: v / growth_factor for k, v in GRID_MIX_SHARES[iso].items()}
    wholesale = max(5, WHOLESALE_PRICES[iso] + FUEL_ADJUSTMENTS[iso][fuel_name])
    if overrides and 'ccs_lcoe' in overrides:
        ccs_lcoe = overrides['ccs_lcoe']
    else:
        ccs_table = CCS_LCOE_45Q_ON if q45 == '1' else None
        ccs_lcoe = ccs_table[ccs_lev][iso]
    ccs_tx = get_tx('ccs_ccgt', tx_name, iso)
    ccs_price = ccs_lcoe + ccs_tx
    if overrides and 'nuclear_lcoe' in overrides:
        nuclear_price = overrides['nuclear_lcoe'] + get_tx('clean_firm', tx_name, iso)
    else:
        nuclear_price = NUCLEAR_NEWBUILD_LCOE[firm_lev][iso] + get_tx('clean_firm', tx_name, iso)
    remaining_price = min(nuclear_price, ccs_price)
    geo_price = 0.0
    if iso == 'CAISO' and geo_lev:
        if overrides and 'geo_lcoe' in overrides:
            geo_price = overrides['geo_lcoe'] + get_tx('clean_firm', tx_name, iso)
        else:
            geo_price = GEOTHERMAL_LCOE[geo_lev] + get_tx('clean_firm', tx_name, iso)
    sol_demand = sol_pct
    wnd_demand = wnd_pct
    ccs_demand = ccs_pct
    cf_demand = cf_pct
    hydro_twh_raw = hyd_pct / 100.0 * demand_twh
    hydro_twh_capped = min(hydro_twh_raw, HYDRO_CAP_TWH[iso])
    hyd_demand = hydro_twh_capped / demand_twh * 100.0
    sol_existing = min(sol_demand, existing['solar'])
    sol_new = max(0, sol_demand - existing['solar'])
    wnd_existing = min(wnd_demand, existing['wind'])
    wnd_new = max(0, wnd_demand - existing['wind'])
    ccs_existing = min(ccs_demand, existing.get('ccs_ccgt', 0))
    ccs_new = max(0, ccs_demand - existing.get('ccs_ccgt', 0))
    cf_existing = min(cf_demand, existing['clean_firm'])
    cf_new = max(0, cf_demand - existing['clean_firm'])
    new_cf_twh = cf_new / 100.0 * demand_twh
    uprate_twh = min(new_cf_twh, UPRATE_CAP_TWH[iso])
    remaining_after_uprate = max(0, new_cf_twh - uprate_twh)
    geo_twh = 0.0
    remaining_after_geo = remaining_after_uprate
    if iso == 'CAISO':
        geo_twh = min(remaining_after_uprate, GEO_CAP_TWH)
        remaining_after_geo = max(0, remaining_after_uprate - geo_twh)
    demand_mwh = demand_twh * 1e6
    avg_demand_mw = demand_mwh / 8760
    peak_mw = PEAK_DEMAND_MW[iso] * growth_factor
    ra_peak_mw = peak_mw * (1 + RESOURCE_ADEQUACY_MARGIN)
    hydro_avg_mw = hydro_twh_capped * 1e6 / 8760
    clean_peak_mw = (
        cf_pct / 100 * avg_demand_mw * PEAK_CAPACITY_CREDITS['clean_firm'] +
        sol_pct / 100 * avg_demand_mw * PEAK_CAPACITY_CREDITS['solar'] +
        wnd_pct / 100 * avg_demand_mw * PEAK_CAPACITY_CREDITS['wind'] +
        ccs_pct / 100 * avg_demand_mw * PEAK_CAPACITY_CREDITS['ccs_ccgt'] +
        hydro_avg_mw * PEAK_CAPACITY_CREDITS['hydro'] +
        bat4_pct / 100 * avg_demand_mw * PEAK_CAPACITY_CREDITS['battery'] +
        bat8_pct / 100 * avg_demand_mw * PEAK_CAPACITY_CREDITS['battery8'] +
        ldes_pct / 100 * avg_demand_mw * PEAK_CAPACITY_CREDITS['ldes']
    )
    gaf = GAS_AVAILABILITY_FACTOR[iso]
    gas_needed_mw = max(0, ra_peak_mw - clean_peak_mw) / gaf
    existing_gas_mw = EXISTING_GAS_CAPACITY_MW[iso]
    existing_gas_used_mw = min(gas_needed_mw, existing_gas_mw)
    new_gas_mw = max(0, gas_needed_mw - existing_gas_used_mw)
    gas_cost = (
        existing_gas_used_mw * EXISTING_GAS_FOM_KW_YR[iso] * 1000 +
        new_gas_mw * NEW_CCGT_COST_KW_YR[iso] * 1000
    ) / demand_mwh
    uprate_price = overrides.get('uprate_lcoe', UPRATE_LCOE[firm_lev]) if overrides else UPRATE_LCOE[firm_lev]
    if overrides and 'ldes_lcoe' in overrides:
        ldes_price = overrides['ldes_lcoe'] + get_tx('ldes', tx_name, iso)
    else:
        ldes_price = LCOE_TABLES['ldes'][ldes_name][iso] + get_tx('ldes', tx_name, iso)
    total_cost = (
        sol_new / 100.0 * (LCOE_TABLES['solar'][ren_name][iso] + get_tx('solar', tx_name, iso)) +
        wnd_new / 100.0 * (LCOE_TABLES['wind'][ren_name][iso] + get_tx('wind', tx_name, iso)) +
        ccs_new / 100.0 * ccs_price +
        uprate_twh / demand_twh * uprate_price +
        geo_twh / demand_twh * geo_price +
        remaining_after_geo / demand_twh * remaining_price +
        bat4_pct / 100.0 * (LCOE_TABLES['battery'][batt_name][iso] + get_tx('battery', tx_name, iso)) +
        bat8_pct / 100.0 * (LCOE_TABLES['battery8'][batt_name][iso] + get_tx('battery8', tx_name, iso)) +
        ldes_pct / 100.0 * ldes_price +
        gas_cost
    )
    new_build_per_mwh = total_cost - gas_cost
    new_gen_twh = (sol_new + wnd_new + ccs_new + cf_new) / 100.0 * demand_twh
    new_build_cost_total = new_build_per_mwh * demand_mwh
    if new_gen_twh > 0.01:
        blended_new_lcoe = new_build_per_mwh * demand_twh / new_gen_twh
    else:
        blended_new_lcoe = 0.0
    eff_cost = total_cost / match_frac if match_frac > 0 else 0
    incremental = eff_cost - wholesale
    resource_twh = {}
    for res, pct in zip(RESOURCES, [cf_pct, sol_pct, wnd_pct, ccs_pct, hyd_pct]):
        if res == 'hydro':
            resource_twh[res] = hydro_twh_capped
        else:
            resource_twh[res] = pct / 100.0 * demand_twh
    return {
        'total_cost': round(total_cost, 2),
        'effective_cost': round(eff_cost, 2),
        'incremental': round(incremental, 2),
        'wholesale': wholesale,
        'resource_twh': resource_twh,
        'resource_pct': {'clean_firm': cf_pct, 'solar': sol_pct, 'wind': wnd_pct,
                         'ccs_ccgt': ccs_pct, 'hydro': hyd_pct},
        'match_score': match_score,
        'battery_twh': bat4_pct / 100.0 * demand_twh,
        'battery8_twh': bat8_pct / 100.0 * demand_twh,
        'ldes_twh': ldes_pct / 100.0 * demand_twh,
        'h2_twh': h2_pct / 100.0 * demand_twh,
        'gas_backup_mw': round(gas_needed_mw),
        'new_gas_mw': round(new_gas_mw),
        'existing_gas_used_mw': round(existing_gas_used_mw),
        'clean_peak_mw': round(clean_peak_mw),
        'new_build_cost_total': new_build_cost_total,
        'new_gen_twh': round(new_gen_twh, 3),
        'blended_new_lcoe': round(blended_new_lcoe, 2),
    }


# ============================================================================
# MIX UTILITIES
# ============================================================================

def _mix_resource_twh(mix, demand_twh, iso=None):
    """Extract deployed resource TWh from a raw mix vector."""
    cf, sol, wnd, ccs, hyd = mix[0], mix[1], mix[2], mix[3], mix[4]
    bat4, bat8, ldes = mix[6], mix[7], mix[8]
    h2 = mix[9] if len(mix) > 9 else 0
    hydro_twh = min(hyd / 100.0 * demand_twh,
                    HYDRO_CAP_TWH[iso]) if iso else hyd / 100.0 * demand_twh
    return {
        'clean_firm': cf / 100.0 * demand_twh,
        'solar':      sol / 100.0 * demand_twh,
        'wind':       wnd / 100.0 * demand_twh,
        'ccs_ccgt':   ccs / 100.0 * demand_twh,
        'hydro':      hydro_twh,
        'battery':    (bat4 + bat8) / 100.0 * demand_twh,
        'ldes':       ldes / 100.0 * demand_twh,
    }


# ============================================================================
# LEARNING CURVE OVERRIDES
# ============================================================================

def _build_learning_overrides(iso, frac):
    """Build LCOE overrides interpolated between High (FOAK) and Low (NOAK)."""
    overrides = {}
    nuc_h = NUCLEAR_NEWBUILD_LCOE['H'][iso]
    nuc_l = NUCLEAR_NEWBUILD_LCOE['L'][iso]
    overrides['nuclear_lcoe'] = nuc_h + frac * (nuc_l - nuc_h)
    overrides['uprate_lcoe'] = UPRATE_LCOE['M']
    ccs_h = CCS_LCOE_45Q_ON['H'][iso]
    ccs_l = CCS_LCOE_45Q_ON['L'][iso]
    overrides['ccs_lcoe'] = ccs_h + frac * (ccs_l - ccs_h)
    if iso == 'CAISO':
        geo_h = GEOTHERMAL_LCOE['H']
        geo_l = GEOTHERMAL_LCOE['L']
        overrides['geo_lcoe'] = geo_h + frac * (geo_l - geo_h)
    ldes_h = LCOE_TABLES['ldes']['High'][iso]
    ldes_l = LCOE_TABLES['ldes']['Low'][iso]
    overrides['ldes_lcoe'] = ldes_h + frac * (ldes_l - ldes_h)
    return overrides


def _build_learning_overrides_b(iso, frac):
    """Build LCOE overrides for Scenario B with differentiated learning curves.
    Nuclear/LDES: H→L. CCS: H→M (starts lower, structural cost floor)."""
    overrides = {}
    nuc_h = NUCLEAR_NEWBUILD_LCOE['H'][iso]
    nuc_l = NUCLEAR_NEWBUILD_LCOE['L'][iso]
    overrides['nuclear_lcoe'] = nuc_h + frac * (nuc_l - nuc_h)
    overrides['uprate_lcoe'] = UPRATE_LCOE['M']
    ccs_h = CCS_LCOE_45Q_ON['H'][iso]
    ccs_m = CCS_LCOE_45Q_ON['M'][iso]
    overrides['ccs_lcoe'] = ccs_h + frac * (ccs_m - ccs_h)
    if iso == 'CAISO':
        geo_h = GEOTHERMAL_LCOE['H']
        geo_l = GEOTHERMAL_LCOE['L']
        overrides['geo_lcoe'] = geo_h + frac * (geo_l - geo_h)
    ldes_h = LCOE_TABLES['ldes']['High'][iso]
    ldes_l = LCOE_TABLES['ldes']['Low'][iso]
    overrides['ldes_lcoe'] = ldes_h + frac * (ldes_l - ldes_h)
    return overrides


def _get_excess_lcoe(res, sens, iso, overrides):
    """Get LCOE for pricing floor excess resources, using scenario-specific overrides."""
    tx_name = LEVEL_NAME[sens['tx']]
    if res == 'clean_firm':
        if overrides and 'nuclear_lcoe' in overrides:
            return overrides['nuclear_lcoe'] + get_tx('clean_firm', tx_name, iso)
        return NUCLEAR_NEWBUILD_LCOE[sens['firm']][iso] + get_tx('clean_firm', tx_name, iso)
    elif res == 'ccs_ccgt':
        if overrides and 'ccs_lcoe' in overrides:
            return overrides['ccs_lcoe'] + get_tx('ccs_ccgt', tx_name, iso)
        ccs_table = CCS_LCOE_45Q_ON if sens.get('q45') == '1' else None
        if ccs_table:
            return ccs_table[sens['ccs']][iso] + get_tx('ccs_ccgt', tx_name, iso)
        return 0
    elif res == 'ldes':
        if overrides and 'ldes_lcoe' in overrides:
            return overrides['ldes_lcoe'] + get_tx('ldes', tx_name, iso)
        ldes_name = LEVEL_NAME[sens['ldes_lvl']]
        return LCOE_TABLES['ldes'][ldes_name][iso] + get_tx('ldes', tx_name, iso)
    elif res == 'solar':
        ren_name = LEVEL_NAME[sens['ren']]
        return LCOE_TABLES['solar'][ren_name][iso] + get_tx('solar', tx_name, iso)
    elif res == 'wind':
        ren_name = LEVEL_NAME[sens['ren']]
        return LCOE_TABLES['wind'][ren_name][iso] + get_tx('wind', tx_name, iso)
    elif res == 'battery':
        batt_name = LEVEL_NAME[sens['batt']]
        return LCOE_TABLES['battery'][batt_name][iso] + get_tx('battery', tx_name, iso)
    elif res == 'hydro':
        return 0
    return 0


# ============================================================================
# PFS LOADER & FLOOR FILTERING
# ============================================================================

def _load_pfs_mixes(iso, threshold):
    """Load PFS mixes for a single ISO/threshold from step1 parquets."""
    t_str = str(int(threshold)) if threshold == int(threshold) else str(threshold)
    pfs_path = Path(f'data/step1-pfs-parquets/{iso}_t{t_str}_raw_pfs.parquet')
    if not pfs_path.exists():
        return []
    df = pd.read_parquet(pfs_path)
    mixes = []
    for _, row in df.iterrows():
        mixes.append([
            row.get('clean_firm', 0), row.get('solar', 0), row.get('wind', 0),
            0, row.get('hydro', 0),
            round(row.get('hourly_match_score', 0), 1),
            row.get('battery_dispatch_pct', 0), row.get('battery8_dispatch_pct', 0),
            row.get('ldes_dispatch_pct', 0), row.get('h2_dispatch_pct', 0),
        ])
    return mixes


def _filter_mixes_by_floor(mixes, floor_twh, demand_twh, iso):
    """Filter mixes to those meeting per-resource TWh floors."""
    floor_pct = {}
    for res in ['clean_firm', 'solar', 'wind', 'ccs_ccgt']:
        floor_pct[res] = floor_twh.get(res, 0) / demand_twh * 100.0 if demand_twh > 0 else 0
    hydro_floor_twh = min(floor_twh.get('hydro', 0), HYDRO_CAP_TWH[iso])
    floor_pct['hydro'] = hydro_floor_twh / demand_twh * 100.0 if demand_twh > 0 else 0
    floor_pct['battery'] = floor_twh.get('battery', 0) / demand_twh * 100.0 if demand_twh > 0 else 0
    floor_pct['ldes'] = floor_twh.get('ldes', 0) / demand_twh * 100.0 if demand_twh > 0 else 0
    passed = []
    for mix in mixes:
        cf, sol, wnd, ccs, hyd = mix[0], mix[1], mix[2], mix[3], mix[4]
        bat4, bat8, ldes = mix[6], mix[7], mix[8]
        if cf < floor_pct['clean_firm'] - 0.01: continue
        if sol < floor_pct['solar'] - 0.01: continue
        if wnd < floor_pct['wind'] - 0.01: continue
        if ccs < floor_pct['ccs_ccgt'] - 0.01: continue
        hyd_twh = min(hyd / 100.0 * demand_twh, HYDRO_CAP_TWH[iso])
        if hyd_twh < hydro_floor_twh - 0.01: continue
        if (bat4 + bat8) < floor_pct['battery'] - 0.01: continue
        if ldes < floor_pct['ldes'] - 0.01: continue
        passed.append(mix)
    return passed


def _filter_pfs_by_floor_window(mixes, floor_twh, demand_twh, iso, max_pct_above=10.0):
    """Filter PFS mixes within a search window above the per-resource floor."""
    floor_pct = {}
    for res in ['clean_firm', 'solar', 'wind']:
        floor_pct[res] = floor_twh.get(res, 0) / demand_twh * 100.0 if demand_twh > 0 else 0
    floor_pct['ccs_ccgt'] = 0
    hydro_floor_twh = min(floor_twh.get('hydro', 0), HYDRO_CAP_TWH[iso])
    floor_pct['hydro'] = hydro_floor_twh / demand_twh * 100.0 if demand_twh > 0 else 0
    floor_pct['battery'] = floor_twh.get('battery', 0) / demand_twh * 100.0 if demand_twh > 0 else 0
    floor_pct['ldes'] = floor_twh.get('ldes', 0) / demand_twh * 100.0 if demand_twh > 0 else 0
    passed = []
    for mix in mixes:
        cf, sol, wnd, ccs, hyd = mix[0], mix[1], mix[2], mix[3], mix[4]
        bat4, bat8, ldes = mix[6], mix[7], mix[8]
        if cf < floor_pct['clean_firm'] - 0.01 or cf > floor_pct['clean_firm'] + max_pct_above + 0.01: continue
        if sol < floor_pct['solar'] - 0.01 or sol > floor_pct['solar'] + max_pct_above + 0.01: continue
        if wnd < floor_pct['wind'] - 0.01 or wnd > floor_pct['wind'] + max_pct_above + 0.01: continue
        hyd_twh = min(hyd / 100.0 * demand_twh, HYDRO_CAP_TWH[iso])
        if hyd_twh < hydro_floor_twh - 0.01: continue
        if (bat4 + bat8) < floor_pct['battery'] - 0.01: continue
        if ldes < floor_pct['ldes'] - 0.01: continue
        passed.append(mix)
    return passed


def _rank_and_cap_pfs(mixes, floor_twh, demand_twh, iso, max_eval=MAX_PFS_EVAL):
    """Rank PFS mixes by Manhattan distance from floor and cap at max_eval.
    Prevents timeout when PFS fallback returns millions of mixes."""
    if len(mixes) <= max_eval:
        return mixes
    floor_pct = {}
    for res in ['clean_firm', 'solar', 'wind']:
        floor_pct[res] = floor_twh.get(res, 0) / demand_twh * 100.0 if demand_twh > 0 else 0
    scored = []
    for mix in mixes:
        cf, sol, wnd = mix[0], mix[1], mix[2]
        dist = (max(0, cf - floor_pct['clean_firm']) +
                max(0, sol - floor_pct['solar']) +
                max(0, wnd - floor_pct['wind']))
        scored.append((dist, mix))
    scored.sort(key=lambda x: x[0])
    capped = [m for _, m in scored[:max_eval]]
    print(f"    ⚡ Capped PFS from {len(mixes)} → {len(capped)} mixes (ranked by floor proximity)")
    return capped


# ============================================================================
# EXISTING MATCH HELPERS
# ============================================================================

def _existing_match_pct(iso):
    """Estimate the hourly CFE match % that 2025 existing resources achieve."""
    shares = GRID_MIX_SHARES[iso]
    cf_frac  = shares.get('clean_firm', 0) / 100.0
    hyd_frac = shares.get('hydro',      0) / 100.0
    sol_frac = shares.get('solar',      0) / 100.0
    wnd_frac = shares.get('wind',       0) / 100.0
    return min((cf_frac + hyd_frac) * 90.0 + (sol_frac + wnd_frac) * 60.0, 95.0)


def _build_existing_only_entry(iso, threshold, demand_twh, gf, existing_twh, sens):
    """Build a result entry representing 2025 existing resources with zero new build."""
    demand_mwh = demand_twh * 1e6
    base_demand = BASE_DEMAND_TWH[iso]
    resource_twh = {}
    for res, pct in GRID_MIX_SHARES[iso].items():
        twh = pct / 100.0 * base_demand
        if res == 'hydro':
            twh = min(twh, HYDRO_CAP_TWH[iso])
        resource_twh[res] = twh
    clean_peak_mw = 0.0
    for res, twh in resource_twh.items():
        pcc = PEAK_CAPACITY_CREDITS.get(res, 0)
        if pcc > 0:
            clean_peak_mw += (twh * 1e6 / 8760) * pcc
    ra_peak_mw = PEAK_DEMAND_MW[iso] * gf * (1 + RESOURCE_ADEQUACY_MARGIN)
    gaf = GAS_AVAILABILITY_FACTOR[iso]
    gas_needed_mw = max(0, ra_peak_mw - clean_peak_mw) / gaf
    existing_gas_mw = EXISTING_GAS_CAPACITY_MW[iso]
    new_gas_mw = max(0, gas_needed_mw - existing_gas_mw)
    fuel_name = LEVEL_NAME[sens['fuel']]
    wholesale = max(5, WHOLESALE_PRICES[iso] + FUEL_ADJUSTMENTS[iso][fuel_name])
    est_match = _existing_match_pct(iso)
    return {
        'threshold': threshold,
        'demand_twh': demand_twh,
        'effective_cost': round(wholesale, 2),
        'total_cost': round(wholesale, 2),
        'incremental': 0.0,
        'wholesale': wholesale,
        'match_score': est_match,
        'resource_twh': resource_twh,
        'resource_pct': dict(GRID_MIX_SHARES[iso]),
        'battery_twh': 0.0, 'battery8_twh': 0.0, 'ldes_twh': 0.0,
        'battery_dispatch_pct': 0.0, 'battery8_dispatch_pct': 0.0,
        'ldes_dispatch_pct': 0.0, 'h2_dispatch_pct': 0.0,
        'gas_backup_mw': round(gas_needed_mw),
        'new_gas_mw': round(new_gas_mw),
        'existing_gas_used_mw': round(min(gas_needed_mw, existing_gas_mw)),
        'clean_peak_mw': round(clean_peak_mw),
        'new_build_cost_total': 0.0, 'new_gen_twh': 0.0, 'blended_new_lcoe': 0.0,
        'demand_mwh': demand_mwh,
    }


# ============================================================================
# DISPATCH CO2 HELPER
# ============================================================================

def _get_dispatch_co2_for_mix(iso, mix_result, egrid, demand_data, gen_profiles,
                              dispatch_caches):
    """Get CO₂ displacement for a scenario mix result using dispatch cache."""
    resource_pcts = mix_result.get('resource_pct', {})
    if not resource_pcts:
        return None
    bat_pct = mix_result.get('battery_dispatch_pct', 0)
    bat8_pct = mix_result.get('battery8_dispatch_pct', 0)
    ldes_pct = mix_result.get('ldes_dispatch_pct', 0)
    h2_pct = mix_result.get('h2_dispatch_pct', 0)
    supply_profiles = get_supply_profiles(iso, gen_profiles)
    demand_norm, _ = get_demand_profile(iso, demand_data)
    cache = dispatch_caches.get(iso, {})
    dispatch_result, _ = get_or_compute_dispatch(
        iso, demand_norm, supply_profiles, resource_pcts,
        battery_dispatch_pct=bat_pct, battery8_dispatch_pct=bat8_pct,
        ldes_dispatch_pct=ldes_pct, cache=cache)
    dispatch_caches[iso] = cache
    demand_mwh = mix_result.get('demand_mwh', 0)
    if demand_mwh <= 0:
        demand_twh = sum(mix_result.get('resource_twh', {}).values())
        demand_mwh = demand_twh * 1e6 if demand_twh > 0 else BASE_DEMAND_TWH[iso] * 1e6
    return compute_co2_from_dispatch(iso, dispatch_result, egrid, demand_mwh)


# ============================================================================
# CONSEQUENTIAL QUEUE BUILDER
# ============================================================================

def build_consequential_queue(scenario_results, egrid, fossil_mix, cfr_fn=None,
                               demand_data=None, gen_profiles=None, dispatch_caches=None):
    """Build the consequential deployment queue for a scenario."""
    use_dispatch = demand_data is not None and gen_profiles is not None
    if cfr_fn is None:
        cfr_fn = compute_fossil_retirement
    if dispatch_caches is None:
        dispatch_caches = {}
    _co2_cache = {}
    if use_dispatch:
        for iso in ISOS:
            iso_data = scenario_results.get(iso, {})
            for t in THRESHOLDS:
                if t not in iso_data:
                    continue
                mix_result = iso_data[t]
                co2 = _get_dispatch_co2_for_mix(iso, mix_result, egrid,
                                                 demand_data, gen_profiles, dispatch_caches)
                if co2:
                    _co2_cache[(iso, t)] = co2
    zone_metrics = []
    for iso in ISOS:
        iso_data = scenario_results.get(iso, {})
        if not iso_data:
            continue
        baseline_clean = sum(GRID_MIX_SHARES[iso].values())
        for zone in ZONES:
            t_start, t_end = zone['start'], zone['end']
            start_data = iso_data.get(t_start, {})
            end_data = iso_data.get(t_end, {})
            if not start_data or not end_data:
                continue
            cost_start = start_data['effective_cost']
            cost_end = end_data['effective_cost']
            delta_cost = cost_end - cost_start
            gas_start = start_data['gas_backup_mw']
            gas_end = end_data['gas_backup_mw']
            delta_gas = gas_end - gas_start
            new_gas_end = end_data.get('new_gas_mw', 0)
            co2_start = _co2_cache.get((iso, t_start))
            co2_end = _co2_cache.get((iso, t_end))
            if co2_start and co2_end:
                co2_mt = max(0, co2_end['total_co2_abated_tons'] -
                             co2_start['total_co2_abated_tons']) / 1e6
            else:
                demand_twh = get_demand_twh(iso, t_end)
                gf_end = get_demand_growth_factor(iso, t_end)
                rate_end, _ = cfr_fn(iso, t_end, egrid, fossil_mix, gf_end)
                gf_start = get_demand_growth_factor(iso, t_start)
                rate_start, _ = cfr_fn(iso, t_start, egrid, fossil_mix, gf_start)
                new_gen_end = end_data.get('new_gen_twh', 0)
                new_gen_start = start_data.get('new_gen_twh', 0)
                delta_gen = max(0, new_gen_end - new_gen_start)
                avg_rate = (rate_start + rate_end) / 2 if (rate_start + rate_end) > 0 else 0.4
                co2_mt = delta_gen * 1e6 * avg_rate / 1e6
            if co2_mt < 0.001:
                co2_mt = 0.001
            marginal_mac = abs(delta_cost * BASE_DEMAND_TWH[iso] * 1e6 / (co2_mt * 1e6)) if co2_mt > 0.001 else 9999
            delta_resources = {}
            for res in RESOURCES:
                r_start = start_data['resource_twh'].get(res, 0)
                r_end = end_data['resource_twh'].get(res, 0)
                if abs(r_end - r_start) > 0.5:
                    delta_resources[res] = r_end - r_start
            for stor in ['battery', 'ldes']:
                s_key = f'{stor}_twh' if stor != 'battery' else 'battery_twh'
                s_start = start_data.get(s_key, 0)
                if stor == 'battery':
                    s_start += start_data.get('battery8_twh', 0)
                s_end = end_data.get(s_key, 0)
                if stor == 'battery':
                    s_end += end_data.get('battery8_twh', 0)
                if abs(s_end - s_start) > 0.5:
                    delta_resources[stor] = s_end - s_start
            zone_metrics.append({
                'iso': iso,
                'zone_label': zone['label'],
                'threshold_start': t_start,
                'threshold_end': t_end,
                'delta_cost_per_mwh': round(delta_cost, 2),
                'co2_displaced_mt': round(co2_mt, 3),
                'marginal_mac': round(min(marginal_mac, 9999), 1),
                'gas_backup_mw_start': gas_start,
                'gas_backup_mw_end': gas_end,
                'delta_gas_mw': delta_gas,
                'new_gas_mw_end': new_gas_end,
                'delta_resources': delta_resources,
            })
    ordered = sorted(zone_metrics, key=lambda x: x['marginal_mac'])
    iso_next_zone = {iso: 0 for iso in ISOS}
    final_ordered = []
    remaining = list(ordered)
    while remaining:
        placed = False
        for item in remaining:
            iso = item['iso']
            zone_idx = next((i for i, z in enumerate(ZONES)
                            if z['start'] == item['threshold_start']
                            and z['end'] == item['threshold_end']), -1)
            if zone_idx <= iso_next_zone[iso]:
                final_ordered.append(item)
                remaining.remove(item)
                iso_next_zone[iso] = zone_idx + 1
                placed = True
                break
        if not placed:
            final_ordered.append(remaining.pop(0))
    for i, step in enumerate(final_ordered):
        step['queue_position'] = i + 1
    return final_ordered


# ============================================================================
# STRANDING ANALYSIS
# ============================================================================

def compute_stranding(scenario_results):
    """Compute stranding analysis for each ISO: peak vs final resource levels."""
    stranding = {}
    for iso in ISOS:
        iso_data = scenario_results.get(iso, {})
        if not iso_data:
            continue
        iso_stranding = {}
        for res in RESOURCES + ['battery', 'ldes']:
            values = []
            for t in THRESHOLDS:
                if t not in iso_data:
                    continue
                if res in RESOURCES:
                    val = iso_data[t]['resource_twh'].get(res, 0)
                elif res == 'battery':
                    val = iso_data[t].get('battery_twh', 0) + iso_data[t].get('battery8_twh', 0)
                else:
                    val = iso_data[t].get(f'{res}_twh', 0)
                values.append((t, val))
            if not values:
                continue
            peak_t, peak_val = max(values, key=lambda x: x[1])
            final_candidates = [(t, v) for t, v in values if t >= 99]
            if final_candidates:
                final_t, final_val = final_candidates[0]
            else:
                final_t, final_val = values[-1]
            ratio = peak_val / final_val if final_val > 0.1 else 1.0
            iso_stranding[res] = {
                'peak_twh': round(peak_val, 1),
                'peak_threshold': peak_t,
                'final_twh': round(final_val, 1),
                'final_threshold': final_t,
                'stranding_ratio': round(ratio, 2),
                'values_by_threshold': {t: round(v, 1) for t, v in values},
            }
        stranding[iso] = iso_stranding
    return stranding


# ============================================================================
# CLI & I/O HELPERS
# ============================================================================

def parse_iso_args():
    """Parse --iso and --max-pfs-eval CLI arguments. Returns (iso_list, max_pfs_eval)."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--iso', type=str, default='ALL',
                        help='ISO to process (CAISO, ERCOT, PJM, NYISO, NEISO, MISO, SPP, or ALL)')
    parser.add_argument('--max-pfs-eval', type=int, default=MAX_PFS_EVAL,
                        help=f'Max PFS mixes to evaluate per fallback (default: {MAX_PFS_EVAL})')
    args, _ = parser.parse_known_args()
    if args.iso == 'ALL':
        return ISOS, args.max_pfs_eval
    elif args.iso in ISOS:
        return [args.iso], args.max_pfs_eval
    else:
        print(f"ERROR: Unknown ISO: {args.iso}. Valid: {', '.join(ISOS)} or ALL")
        sys.exit(1)


def save_scenario_results(results, scenario_label, isos_processed):
    """Save scenario results to intermediate JSON for the comparison script."""
    output = {
        'metadata': {
            'scenario': scenario_label,
            'isos_processed': isos_processed,
            'timestamp': datetime.datetime.utcnow().isoformat(),
            'thresholds': list(THRESHOLDS),
        },
        'results': {}
    }
    for iso in isos_processed:
        iso_data = results.get(iso, {})
        output['results'][iso] = {}
        for t, d in iso_data.items():
            t_key = str(int(t)) if t == int(t) else str(t)
            entry = {}
            for k, v in d.items():
                if isinstance(v, (np.integer,)):
                    entry[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    entry[k] = float(v)
                elif isinstance(v, np.ndarray):
                    entry[k] = v.tolist()
                else:
                    entry[k] = v
            output['results'][iso][t_key] = entry
    out_path = f'data/step5-post-processing/scenario_{scenario_label.lower()}_results.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved {scenario_label} results: {out_path} ({os.path.getsize(out_path) / 1024:.0f} KB)")
    return out_path


def load_scenario_results(scenario_label):
    """Load scenario results from intermediate JSON. Returns (results_dict, metadata)."""
    path = f'data/step5-post-processing/scenario_{scenario_label.lower()}_results.json'
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scenario results not found: {path}")
    with open(path) as f:
        data = json.load(f)
    results = {}
    for iso, iso_data in data['results'].items():
        results[iso] = {}
        for t_str, d in iso_data.items():
            t = float(t_str)
            results[iso][t] = d
    return results, data['metadata']
