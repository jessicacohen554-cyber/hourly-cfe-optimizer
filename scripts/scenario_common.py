#!/usr/bin/env python3
"""
Shared constants, cost functions, and utilities for scenario comparison scripts.

Used by:
  - step5_scenario_a_consequential.py (Scenario A: Pure Consequential)
  - step5_scenario_b_hourly.py (Scenario B: Hourly Matching)
  - step5_scenario_comparison.py (comparison and output)
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

# Import shared constants from pipeline_config (single source of truth)
sys.path.insert(0, str(Path(__file__).parent))
from pipeline_config import (
    ISOS, OFFSHORE_ISOS,
    OFFSHORE_WIND_CAP_TWH, CCS_CAP_TWH, GEOTHERMAL_CAP_TWH,
    GRID_MIX_SHARES, REGIONAL_DEMAND_TWH,
    WHOLESALE_PRICES, FUEL_ADJUSTMENTS,
    NEISO_CCS_GAS_ADDER, NEISO_WHOLESALE_ADDER,
    H,
)

# Import dispatch utilities
from dispatch_utils import (
    compute_fossil_retirement, compute_co2_from_dispatch,
    COAL_CAP_TWH, OIL_CAP_TWH, COAL_OIL_RETIREMENT_THRESHOLD,
    load_common_data, get_supply_profiles, get_demand_profile,
    build_supply_matrix, reconstruct_hourly_dispatch,
    _archetype_key, load_dispatch_cache, get_or_compute_dispatch,
)

# ============================================================================
# CONSTANTS
# ============================================================================

from step3_cost_optimization import OUTPUT_THRESHOLDS as THRESHOLDS

RESOURCES = ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro']

# Backward compatibility alias
BASE_DEMAND_TWH = REGIONAL_DEMAND_TWH

HYDRO_CAP_TWH = {iso: GRID_MIX_SHARES[iso].get('hydro', 0) / 100.0 * REGIONAL_DEMAND_TWH[iso]
                 for iso in ISOS}

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
    'offshore_wind': {
        'Low':    {'CAISO': 110, 'ERCOT': 0, 'PJM': 65, 'NYISO': 72, 'NEISO': 68, 'MISO': 0, 'SPP': 0},
        'Medium': {'CAISO': 150, 'ERCOT': 0, 'PJM': 85, 'NYISO': 95, 'NEISO': 90, 'MISO': 0, 'SPP': 0},
        'High':   {'CAISO': 200, 'ERCOT': 0, 'PJM': 112, 'NYISO': 125, 'NEISO': 118, 'MISO': 0, 'SPP': 0},
    },
    # Storage: annualized capacity cost ($/MWh-cap), NOT LCOS.
    # NREL ATB 2024 component model: Energy($/kWh) + Power($/kW)/Duration.
    # 4hr→8hr ~14% cheaper (power spread over 2× energy). Regional variation baked in; TX=0.
    'battery': {
        'Low':    {'CAISO': 3.86, 'ERCOT': 3.48, 'PJM': 3.70, 'NYISO': 4.08, 'NEISO': 3.97, 'MISO': 3.62, 'SPP': 3.52},
        'Medium': {'CAISO': 4.75, 'ERCOT': 4.27, 'PJM': 4.55, 'NYISO': 5.02, 'NEISO': 4.87, 'MISO': 4.46, 'SPP': 4.33},
        'High':   {'CAISO': 6.03, 'ERCOT': 5.43, 'PJM': 5.78, 'NYISO': 6.38, 'NEISO': 6.20, 'MISO': 5.66, 'SPP': 5.50},
    },
    'battery8': {
        'Low':    {'CAISO': 3.30, 'ERCOT': 2.97, 'PJM': 3.16, 'NYISO': 3.49, 'NEISO': 3.39, 'MISO': 3.10, 'SPP': 3.01},
        'Medium': {'CAISO': 4.06, 'ERCOT': 3.66, 'PJM': 3.89, 'NYISO': 4.30, 'NEISO': 4.17, 'MISO': 3.81, 'SPP': 3.70},
        'High':   {'CAISO': 5.19, 'ERCOT': 4.67, 'PJM': 4.97, 'NYISO': 5.49, 'NEISO': 5.33, 'MISO': 4.87, 'SPP': 4.73},
    },
    'ldes': {
        'Low':    {'CAISO': 0.38, 'ERCOT': 0.33, 'PJM': 0.36, 'NYISO': 0.42, 'NEISO': 0.40, 'MISO': 0.34, 'SPP': 0.33},
        'Medium': {'CAISO': 0.63, 'ERCOT': 0.54, 'PJM': 0.59, 'NYISO': 0.70, 'NEISO': 0.66, 'MISO': 0.56, 'SPP': 0.55},
        'High':   {'CAISO': 1.00, 'ERCOT': 0.86, 'PJM': 0.94, 'NYISO': 1.11, 'NEISO': 1.06, 'MISO': 0.90, 'SPP': 0.88},
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
    'offshore_wind': {'None': 0, 'Low': {'CAISO': 10, 'ERCOT': 0, 'PJM': 6, 'NYISO': 8, 'NEISO': 7, 'MISO': 0, 'SPP': 0},
                      'Medium': {'CAISO': 20, 'ERCOT': 0, 'PJM': 11, 'NYISO': 15, 'NEISO': 13, 'MISO': 0, 'SPP': 0},
                      'High': {'CAISO': 35, 'ERCOT': 0, 'PJM': 18, 'NYISO': 25, 'NEISO': 22, 'MISO': 0, 'SPP': 0}},
    # Storage TX = 0: regional variation baked into annualized capacity costs
    'battery':    {'None': 0, 'Low': 0, 'Medium': 0, 'High': 0},
    'battery8':   {'None': 0, 'Low': 0, 'Medium': 0, 'High': 0},
    'ldes':       {'None': 0, 'Low': 0, 'Medium': 0, 'High': 0},
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
    'L': {'CAISO': 59.5, 'ERCOT': 53.5, 'PJM': 63.5, 'NYISO': 79.5, 'NEISO': 76.5, 'MISO': 56.5, 'SPP': 51.5},
    'M': {'CAISO': 87.5, 'ERCOT': 72.5, 'PJM': 80.5, 'NYISO': 100.5, 'NEISO': 97.5, 'MISO': 75.5, 'SPP': 69.5},
    'H': {'CAISO': 116.5, 'ERCOT': 93.5, 'PJM': 103.5, 'NYISO': 129.5, 'NEISO': 123.5, 'MISO': 97.5, 'SPP': 89.5},
}

NEISO_CCS_GAS_ADDER = 13.13  # $/MWh — winter gas pipeline constraint (Algonquin Citygates)

EXISTING_NUCLEAR_GW = {'CAISO': 2.3, 'ERCOT': 2.7, 'PJM': 32.0, 'NYISO': 3.4, 'NEISO': 3.5, 'MISO': 12.0, 'SPP': 1.2}
UPRATE_CAP_TWH = {iso: round(gw * 0.08 * 0.90 * 8760 / 1e3, 3)
                  for iso, gw in EXISTING_NUCLEAR_GW.items()}

PEAK_DEMAND_MW = {'CAISO': 43860, 'ERCOT': 83597, 'PJM': 160560, 'NYISO': 31857, 'NEISO': 25898, 'MISO': 127125, 'SPP': 54368}
EXISTING_GAS_CAPACITY_MW = {'CAISO': 37000, 'ERCOT': 55000, 'PJM': 75000, 'NYISO': 18000, 'NEISO': 14000, 'MISO': 68000, 'SPP': 32000}
NEW_CCGT_COST_KW_YR = {'CAISO': 112, 'ERCOT': 89, 'PJM': 99, 'NYISO': 114, 'NEISO': 105, 'MISO': 95, 'SPP': 88}
EXISTING_GAS_FOM_KW_YR = {'CAISO': 16, 'ERCOT': 13, 'PJM': 14, 'NYISO': 17, 'NEISO': 15, 'MISO': 14, 'SPP': 13}
PEAK_CAPACITY_CREDITS = {
    'clean_firm': 1.0, 'solar': 0.30, 'wind': 0.10, 'offshore_wind': 0.25,
    'ccs_ccgt': 0.90, 'hydro': 0.50, 'battery': 0.95, 'battery8': 0.95, 'ldes': 0.90,
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
    {'label': '99.9→≥99.99%', 'start': 99.9, 'end': 99.99},
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

# Max mixes to evaluate per threshold (prevents timeout on large ISOs like PJM/NEISO/MISO)
MAX_PFS_EVAL = 500    # PFS fallback cap
MAX_EF_EVAL  = 2000   # EF cap — EF mixes are pre-filtered so higher is fine


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
        foak_start, noak_year = 2036, 2048
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
    """Load feasible mixes for a single ISO from the step3 parquet file.
    Uses vectorized column extraction — no iterrows().
    """
    feasible_path = os.path.join(step3_dir, f'step3_feasible_{iso}.parquet')
    if not os.path.exists(feasible_path):
        return {}
    df = pd.read_parquet(feasible_path)
    mix_fields = ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro',
                  'hourly_match_score',
                  'battery_dispatch_pct', 'battery8_dispatch_pct',
                  'ldes_dispatch_pct', 'h2_dispatch_pct']
    for col in mix_fields:
        if col not in df.columns:
            df[col] = 0
    # Vectorized: build (N, 11) array, then split by threshold
    vals = df[mix_fields].values.astype(np.float64)
    vals[:, 6] = np.round(vals[:, 6], 1)  # round match score
    result = {}
    for t_val, grp_idx in df.groupby('threshold').groups.items():
        t_str = str(int(t_val)) if t_val == int(t_val) else str(t_val)
        result[t_str] = vals[grp_idx].tolist()
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
    elif res == 'offshore_wind':
        osw_lcoe = LCOE_TABLES['offshore_wind'][ren_name].get(iso, 0)
        return osw_lcoe + get_tx('offshore_wind', tx_name, iso) if osw_lcoe > 0 else 0
    elif res == 'clean_firm':
        return NUCLEAR_NEWBUILD_LCOE[firm_lev][iso] + get_tx('clean_firm', tx_name, iso)
    elif res == 'ccs_ccgt':
        ccs_table = CCS_LCOE_45Q_ON if q45 == '1' else None
        ccs_lcoe = ccs_table[ccs_lev][iso]
        if iso == 'NEISO':
            ccs_lcoe += NEISO_CCS_GAS_ADDER
        return ccs_lcoe + get_tx('ccs_ccgt', tx_name, iso)
    elif res == 'hydro':
        return 0
    elif res == 'battery':
        return LCOE_TABLES['battery'][batt_name][iso] + get_tx('battery', tx_name, iso)
    elif res == 'ldes':
        return LCOE_TABLES['ldes'][ldes_name][iso] + get_tx('ldes', tx_name, iso)
    return 0


def compute_mix_cost(mix, sens, iso, demand_twh, overrides=None, growth_factor=1.0):
    """Compute total system cost per MWh for a single mix under a sensitivity scenario.

    Mix format: [cf, sol, wnd, osw, ccs, hyd, score, bat4, bat8, ldes, h2] (11 elements)
    Backward compat: 10-element mixes treated as [cf, sol, wnd, ccs, hyd, score, bat4, bat8, ldes, h2]
    """
    if len(mix) >= 11:
        cf_pct, sol_pct, wnd_pct, osw_pct = mix[0], mix[1], mix[2], mix[3]
        ccs_pct, hyd_pct = mix[4], mix[5]
        match_score = mix[6]
        bat4_pct, bat8_pct, ldes_pct = mix[7], mix[8], mix[9]
        h2_pct = mix[10] if len(mix) > 10 else 0
    else:
        # Legacy 10-element format (no offshore wind)
        cf_pct, sol_pct, wnd_pct, ccs_pct, hyd_pct = mix[0], mix[1], mix[2], mix[3], mix[4]
        osw_pct = 0
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
    if iso == 'NEISO':
        ccs_lcoe += NEISO_CCS_GAS_ADDER
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
    osw_demand = osw_pct
    ccs_demand = ccs_pct
    cf_demand = cf_pct
    hydro_twh_raw = hyd_pct / 100.0 * demand_twh
    hydro_twh_capped = min(hydro_twh_raw, HYDRO_CAP_TWH[iso])
    hyd_demand = hydro_twh_capped / demand_twh * 100.0
    sol_existing = min(sol_demand, existing['solar'])
    sol_new = max(0, sol_demand - existing['solar'])
    wnd_existing = min(wnd_demand, existing['wind'])
    wnd_new = max(0, wnd_demand - existing['wind'])
    osw_existing = min(osw_demand, existing.get('offshore_wind', 0))
    osw_new = max(0, osw_demand - existing.get('offshore_wind', 0))
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
        osw_pct / 100 * avg_demand_mw * PEAK_CAPACITY_CREDITS['offshore_wind'] +
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
    # Offshore wind cost (0 for non-offshore ISOs since osw_new=0 and LCOE=0)
    osw_lcoe = LCOE_TABLES['offshore_wind'][ren_name].get(iso, 0)
    osw_tx = get_tx('offshore_wind', tx_name, iso)
    total_cost = (
        sol_new / 100.0 * (LCOE_TABLES['solar'][ren_name][iso] + get_tx('solar', tx_name, iso)) +
        wnd_new / 100.0 * (LCOE_TABLES['wind'][ren_name][iso] + get_tx('wind', tx_name, iso)) +
        osw_new / 100.0 * (osw_lcoe + osw_tx) +
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
    new_gen_twh = (sol_new + wnd_new + osw_new + ccs_new + cf_new) / 100.0 * demand_twh
    new_build_cost_total = new_build_per_mwh * demand_mwh
    if new_gen_twh > 0.01:
        blended_new_lcoe = new_build_per_mwh * demand_twh / new_gen_twh
    else:
        blended_new_lcoe = 0.0
    eff_cost = total_cost / match_frac if match_frac > 0 else 0
    incremental = eff_cost - wholesale
    resource_twh = {}
    for res, pct in zip(RESOURCES, [cf_pct, sol_pct, wnd_pct, osw_pct, ccs_pct, hyd_pct]):
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
                         'offshore_wind': osw_pct, 'ccs_ccgt': ccs_pct, 'hydro': hyd_pct},
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
    """Extract deployed resource TWh from a raw mix vector.

    Mix format: 11 elements [cf, sol, wnd, osw, ccs, hyd, score, bat4, bat8, ldes, h2]
    or legacy 10 elements [cf, sol, wnd, ccs, hyd, score, bat4, bat8, ldes, h2]
    """
    if len(mix) >= 11:
        cf, sol, wnd, osw, ccs, hyd = mix[0], mix[1], mix[2], mix[3], mix[4], mix[5]
        bat4, bat8, ldes = mix[7], mix[8], mix[9]
        h2 = mix[10] if len(mix) > 10 else 0
    else:
        cf, sol, wnd, ccs, hyd = mix[0], mix[1], mix[2], mix[3], mix[4]
        osw = 0
        bat4, bat8, ldes = mix[6], mix[7], mix[8]
        h2 = mix[9] if len(mix) > 9 else 0
    hydro_twh = min(hyd / 100.0 * demand_twh,
                    HYDRO_CAP_TWH[iso]) if iso else hyd / 100.0 * demand_twh
    return {
        'clean_firm': cf / 100.0 * demand_twh,
        'solar':      sol / 100.0 * demand_twh,
        'wind':       wnd / 100.0 * demand_twh,
        'offshore_wind': osw / 100.0 * demand_twh,
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
    elif res == 'offshore_wind':
        ren_name = LEVEL_NAME[sens['ren']]
        osw_lcoe = LCOE_TABLES['offshore_wind'][ren_name].get(iso, 0)
        return osw_lcoe + get_tx('offshore_wind', tx_name, iso) if osw_lcoe > 0 else 0
    elif res == 'ccs_ccgt':
        neiso_adder = NEISO_CCS_GAS_ADDER if iso == 'NEISO' else 0
        if overrides and 'ccs_lcoe' in overrides:
            return overrides['ccs_lcoe'] + neiso_adder + get_tx('ccs_ccgt', tx_name, iso)
        ccs_table = CCS_LCOE_45Q_ON if sens.get('q45') == '1' else None
        if ccs_table:
            return ccs_table[sens['ccs']][iso] + neiso_adder + get_tx('ccs_ccgt', tx_name, iso)
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
# VECTORIZED BATCH OPERATIONS (numpy + Numba)
# ============================================================================
# These replace per-mix Python loops for filter + cost evaluation.
# Speedup: ~1000× on large EF sets (7M mixes in <1s vs 20+ min).

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

def _to_mix_array(mixes):
    """Convert list-of-lists mixes to (N, 11) float64 array.

    Mix format: [cf, sol, wnd, osw, ccs, hyd, score, bat4, bat8, ldes, h2]
    Backward compat: 10-element mixes (no osw) auto-padded with osw=0 at index 3.
    """
    if isinstance(mixes, np.ndarray):
        if mixes.ndim == 2 and mixes.shape[1] >= 11:
            return mixes.astype(np.float64, copy=False)
        if mixes.ndim == 2 and mixes.shape[1] == 10:
            # Old 10-col format: [cf, sol, wnd, ccs, hyd, score, bat4, bat8, ldes, h2]
            # Insert osw=0 at index 3
            padded = np.zeros((mixes.shape[0], 11), dtype=np.float64)
            padded[:, :3] = mixes[:, :3]       # cf, sol, wnd
            padded[:, 3] = 0.0                   # osw = 0
            padded[:, 4:] = mixes[:, 3:]         # ccs, hyd, score, bat4, bat8, ldes, h2
            return padded
    arr = np.zeros((len(mixes), 11), dtype=np.float64)
    for i, m in enumerate(mixes):
        if len(m) == 10:
            # Old format — insert osw=0 at index 3
            arr[i, :3] = m[:3]
            arr[i, 3] = 0.0
            arr[i, 4:] = m[3:]
        else:
            n = min(len(m), 11)
            for j in range(n):
                arr[i, j] = m[j]
    return arr


def _precompute_cost_params(sens, iso, demand_twh, overrides=None, growth_factor=1.0):
    """Pack all scalar cost parameters into a flat float64 array for the batch kernel.

    Returns (params, is_caiso) where params is float64[36].
    Column layout documented in _batch_costs_numpy / _batch_costs_numba.
    """
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

    # CCS price
    if overrides and 'ccs_lcoe' in overrides:
        ccs_lcoe = overrides['ccs_lcoe']
    else:
        ccs_table = CCS_LCOE_45Q_ON if q45 == '1' else None
        ccs_lcoe = ccs_table[ccs_lev][iso]
    if iso == 'NEISO':
        ccs_lcoe += NEISO_CCS_GAS_ADDER
    ccs_price = ccs_lcoe + get_tx('ccs_ccgt', tx_name, iso)

    # Nuclear / remaining price
    if overrides and 'nuclear_lcoe' in overrides:
        nuclear_price = overrides['nuclear_lcoe'] + get_tx('clean_firm', tx_name, iso)
    else:
        nuclear_price = NUCLEAR_NEWBUILD_LCOE[firm_lev][iso] + get_tx('clean_firm', tx_name, iso)
    remaining_price = min(nuclear_price, ccs_price)

    # Geothermal
    geo_cap = 0.0
    geo_price = 0.0
    if iso == 'CAISO' and geo_lev:
        if overrides and 'geo_lcoe' in overrides:
            geo_price = overrides['geo_lcoe'] + get_tx('clean_firm', tx_name, iso)
        else:
            geo_price = GEOTHERMAL_LCOE[geo_lev] + get_tx('clean_firm', tx_name, iso)
        geo_cap = GEO_CAP_TWH

    # Uprate
    uprate_price = overrides.get('uprate_lcoe', UPRATE_LCOE[firm_lev]) if overrides else UPRATE_LCOE[firm_lev]

    # LDES
    if overrides and 'ldes_lcoe' in overrides:
        ldes_price = overrides['ldes_lcoe'] + get_tx('ldes', tx_name, iso)
    else:
        ldes_price = LCOE_TABLES['ldes'][ldes_name][iso] + get_tx('ldes', tx_name, iso)

    # Offshore wind price
    if iso in OFFSHORE_ISOS:
        osw_ren_name = ren_name  # Same sensitivity level as onshore
        if overrides and 'osw_lcoe' in overrides:
            osw_price = overrides['osw_lcoe'] + get_tx('offshore_wind', tx_name, iso)
        else:
            osw_price = LCOE_TABLES['offshore_wind'][osw_ren_name][iso] + get_tx('offshore_wind', tx_name, iso)
    else:
        osw_price = 0.0  # Non-offshore ISOs: no offshore wind available

    demand_mwh = demand_twh * 1e6
    avg_demand_mw = demand_mwh / 8760.0
    peak_mw = PEAK_DEMAND_MW[iso] * growth_factor
    ra_peak_mw = peak_mw * (1 + RESOURCE_ADEQUACY_MARGIN)

    pcc = PEAK_CAPACITY_CREDITS
    params = np.array([
        existing.get('clean_firm', 0),       # 0: exist_cf_pct
        existing.get('solar', 0),            # 1: exist_sol_pct
        existing.get('wind', 0),             # 2: exist_wnd_pct
        existing.get('ccs_ccgt', 0),         # 3: exist_ccs_pct
        demand_twh,                          # 4
        HYDRO_CAP_TWH[iso],                  # 5
        LCOE_TABLES['solar'][ren_name][iso] + get_tx('solar', tx_name, iso),  # 6: sol_price
        LCOE_TABLES['wind'][ren_name][iso] + get_tx('wind', tx_name, iso),    # 7: wnd_price
        ccs_price,                           # 8
        UPRATE_CAP_TWH[iso],                 # 9
        uprate_price,                        # 10
        geo_cap,                             # 11
        geo_price,                           # 12
        remaining_price,                     # 13
        LCOE_TABLES['battery'][batt_name][iso] + get_tx('battery', tx_name, iso),   # 14: bat4_price
        LCOE_TABLES['battery8'][batt_name][iso] + get_tx('battery8', tx_name, iso), # 15: bat8_price
        ldes_price,                          # 16
        demand_mwh,                          # 17
        avg_demand_mw,                       # 18
        ra_peak_mw,                          # 19
        pcc.get('clean_firm', 0),            # 20
        pcc.get('solar', 0),                 # 21
        pcc.get('wind', 0),                  # 22
        pcc.get('ccs_ccgt', 0),              # 23
        pcc.get('hydro', 0),                 # 24
        pcc.get('battery', 0),               # 25
        pcc.get('battery8', 0),              # 26
        pcc.get('ldes', 0),                  # 27
        GAS_AVAILABILITY_FACTOR[iso],        # 28
        EXISTING_GAS_CAPACITY_MW[iso],       # 29
        EXISTING_GAS_FOM_KW_YR[iso] * 1000,  # 30: gas_fom per MW
        NEW_CCGT_COST_KW_YR[iso] * 1000,     # 31: new ccgt per MW
        wholesale,                           # 32
        existing.get('offshore_wind', 0),    # 33: exist_osw_pct
        osw_price,                           # 34: offshore wind price (LCOE+TX)
        pcc.get('offshore_wind', 0),         # 35: offshore wind capacity credit
    ], dtype=np.float64)

    return params


def _batch_costs_numpy(mixes, params):
    """Vectorized cost computation using pure numpy.

    Args:
        mixes: (N, 11) float64 array — [cf, sol, wnd, osw, ccs, hyd, match, bat4, bat8, ldes, h2]
        params: float64[36] from _precompute_cost_params

    Returns: (N,) array of total_cost per MWh
    """
    cf  = mixes[:, 0]
    sol = mixes[:, 1]
    wnd = mixes[:, 2]
    osw = mixes[:, 3]
    ccs = mixes[:, 4]
    hyd = mixes[:, 5]
    bat4 = mixes[:, 7]
    bat8 = mixes[:, 8]
    ldes = mixes[:, 9]

    p = params  # alias
    dt = p[4]   # demand_twh

    # New build fractions (% of demand)
    sol_new = np.maximum(0.0, sol - p[1])
    wnd_new = np.maximum(0.0, wnd - p[2])
    ccs_new = np.maximum(0.0, ccs - p[3])
    cf_new  = np.maximum(0.0, cf  - p[0])
    osw_new = np.maximum(0.0, osw - p[33])

    # Clean firm merit order: uprate → geo → nuclear/CCS
    new_cf_twh = cf_new / 100.0 * dt
    uprate_twh = np.minimum(new_cf_twh, p[9])
    remaining_twh = np.maximum(0.0, new_cf_twh - uprate_twh)
    geo_twh = np.minimum(remaining_twh, p[11])       # 0 for non-CAISO
    remaining_twh = np.maximum(0.0, remaining_twh - geo_twh)

    # Gas backup
    hydro_twh_capped = np.minimum(hyd / 100.0 * dt, p[5])
    hydro_avg_mw = hydro_twh_capped * 1e6 / 8760.0

    clean_peak_mw = (
        cf   / 100.0 * p[18] * p[20] +
        sol  / 100.0 * p[18] * p[21] +
        wnd  / 100.0 * p[18] * p[22] +
        osw  / 100.0 * p[18] * p[35] +
        ccs  / 100.0 * p[18] * p[23] +
        hydro_avg_mw * p[24] +
        bat4 / 100.0 * p[18] * p[25] +
        bat8 / 100.0 * p[18] * p[26] +
        ldes / 100.0 * p[18] * p[27]
    )

    gas_needed_mw = np.maximum(0.0, p[19] - clean_peak_mw) / p[28]
    existing_gas_used = np.minimum(gas_needed_mw, p[29])
    new_gas = np.maximum(0.0, gas_needed_mw - p[29])

    gas_cost = (existing_gas_used * p[30] + new_gas * p[31]) / p[17]

    # Total system cost $/MWh
    total_cost = (
        sol_new / 100.0 * p[6] +
        wnd_new / 100.0 * p[7] +
        osw_new / 100.0 * p[34] +
        ccs_new / 100.0 * p[8] +
        uprate_twh / dt * p[10] +
        geo_twh / dt * p[12] +
        remaining_twh / dt * p[13] +
        bat4 / 100.0 * p[14] +
        bat8 / 100.0 * p[15] +
        ldes / 100.0 * p[16] +
        gas_cost
    )

    return total_cost


def _make_numba_kernel():
    """Create Numba-JIT compiled cost kernel (called once at import if Numba available)."""
    @njit(cache=True)
    def _kernel(mixes, params):
        N = mixes.shape[0]
        costs = np.empty(N)
        for i in range(N):
            cf   = mixes[i, 0]
            sol  = mixes[i, 1]
            wnd  = mixes[i, 2]
            osw  = mixes[i, 3]
            ccs  = mixes[i, 4]
            hyd  = mixes[i, 5]
            bat4 = mixes[i, 7]
            bat8 = mixes[i, 8]
            ld   = mixes[i, 9]

            dt = params[4]

            sol_new = max(0.0, sol - params[1])
            wnd_new = max(0.0, wnd - params[2])
            ccs_new = max(0.0, ccs - params[3])
            cf_new  = max(0.0, cf  - params[0])
            osw_new = max(0.0, osw - params[33])

            new_cf_twh = cf_new / 100.0 * dt
            uprate_twh = min(new_cf_twh, params[9])
            rem = max(0.0, new_cf_twh - uprate_twh)
            geo_twh = min(rem, params[11])
            rem = max(0.0, rem - geo_twh)

            hyd_twh_cap = min(hyd / 100.0 * dt, params[5])
            hyd_avg_mw = hyd_twh_cap * 1e6 / 8760.0

            clean_peak = (
                cf   / 100.0 * params[18] * params[20] +
                sol  / 100.0 * params[18] * params[21] +
                wnd  / 100.0 * params[18] * params[22] +
                osw  / 100.0 * params[18] * params[35] +
                ccs  / 100.0 * params[18] * params[23] +
                hyd_avg_mw * params[24] +
                bat4 / 100.0 * params[18] * params[25] +
                bat8 / 100.0 * params[18] * params[26] +
                ld   / 100.0 * params[18] * params[27]
            )

            gas_needed = max(0.0, params[19] - clean_peak) / params[28]
            ex_gas = min(gas_needed, params[29])
            new_gas = max(0.0, gas_needed - params[29])

            gas_cost = (ex_gas * params[30] + new_gas * params[31]) / params[17]

            costs[i] = (
                sol_new / 100.0 * params[6] +
                wnd_new / 100.0 * params[7] +
                osw_new / 100.0 * params[34] +
                ccs_new / 100.0 * params[8] +
                uprate_twh / dt * params[10] +
                geo_twh / dt * params[12] +
                rem / dt * params[13] +
                bat4 / 100.0 * params[14] +
                bat8 / 100.0 * params[15] +
                ld / 100.0 * params[16] +
                gas_cost
            )
        return costs
    return _kernel

# Compile Numba kernel at import time (if available)
_batch_costs_numba = _make_numba_kernel() if _HAS_NUMBA else None


def batch_compute_total_costs(mixes_arr, params):
    """Compute total_cost for all mixes. Uses Numba if available, else numpy.

    Args:
        mixes_arr: (N, 11) float64 array — [cf, sol, wnd, osw, ccs, hyd, match, bat4, bat8, ldes, h2]
        params: float64[36] from _precompute_cost_params

    Returns: (N,) float64 array of total_cost per MWh
    """
    if _batch_costs_numba is not None:
        return _batch_costs_numba(mixes_arr, params)
    return _batch_costs_numpy(mixes_arr, params)


def batch_effective_costs(mixes_arr, params):
    """Compute effective_cost (= total_cost / match_frac) for all mixes.

    Returns: (N,) float64 array of effective_cost per MWh.
    Mixes with match_score == 0 get cost = inf.
    """
    total = batch_compute_total_costs(mixes_arr, params)
    match_frac = mixes_arr[:, 6] / 100.0  # col 6 = match_score (was col 5 in 10-elem format)
    eff = np.where(match_frac > 0, total / match_frac, np.inf)
    return eff


def batch_filter_floor(mixes_arr, floor_twh, demand_twh, iso):
    """Vectorized floor filter. Returns boolean mask (True = passes floor).

    Args:
        mixes_arr: (N, 11) float64 array — [cf, sol, wnd, osw, ccs, hyd, match, bat4, bat8, ldes, h2]
        floor_twh: dict with per-resource floor TWh values
        demand_twh: scalar
        iso: ISO string (for hydro cap lookup)

    Returns: (N,) boolean array
    """
    if demand_twh <= 0:
        return np.ones(mixes_arr.shape[0], dtype=bool)

    floor_cf  = floor_twh.get('clean_firm', 0) / demand_twh * 100.0
    floor_sol = floor_twh.get('solar', 0) / demand_twh * 100.0
    floor_wnd = floor_twh.get('wind', 0) / demand_twh * 100.0
    floor_osw = floor_twh.get('offshore_wind', 0) / demand_twh * 100.0
    floor_ccs = floor_twh.get('ccs_ccgt', 0) / demand_twh * 100.0
    floor_hyd_twh = min(floor_twh.get('hydro', 0), HYDRO_CAP_TWH[iso])
    floor_bat = floor_twh.get('battery', 0) / demand_twh * 100.0
    floor_ldes = floor_twh.get('ldes', 0) / demand_twh * 100.0

    eps = 0.01
    mask = (
        (mixes_arr[:, 0] >= floor_cf  - eps) &
        (mixes_arr[:, 1] >= floor_sol - eps) &
        (mixes_arr[:, 2] >= floor_wnd - eps) &
        (mixes_arr[:, 3] >= floor_osw - eps) &
        (mixes_arr[:, 4] >= floor_ccs - eps) &
        (np.minimum(mixes_arr[:, 5] / 100.0 * demand_twh, HYDRO_CAP_TWH[iso])
            >= floor_hyd_twh - eps) &
        ((mixes_arr[:, 7] + mixes_arr[:, 8]) >= floor_bat - eps) &
        (mixes_arr[:, 9] >= floor_ldes - eps)
    )
    return mask


def batch_floor_excess(mixes_arr, floor_twh, existing_twh, demand_twh, iso,
                       excess_lcoes):
    """Vectorized floor-excess penalty for mixes that go under floor.

    Args:
        mixes_arr: (N, 10) float64
        floor_twh: dict of per-resource floor TWh
        existing_twh: dict of per-resource existing TWh (priced at $0)
        demand_twh: scalar
        iso: ISO string
        excess_lcoes: dict of per-resource LCOE for pricing shortfall
            (precomputed via _get_excess_lcoe for each resource)

    Returns: (N,) float64 array of excess $/MWh penalty
    """
    N = mixes_arr.shape[0]
    excess = np.zeros(N, dtype=np.float64)

    # Map resources to mix columns and their TWh
    # 11-element: cf=0, sol=1, wnd=2, osw=3, ccs=4, hyd=5, score=6, bat4=7, bat8=8, ldes=9, h2=10
    resources = ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro', 'battery', 'ldes']
    cols = {
        'clean_firm':    lambda m: m[:, 0] / 100.0 * demand_twh,
        'solar':         lambda m: m[:, 1] / 100.0 * demand_twh,
        'wind':          lambda m: m[:, 2] / 100.0 * demand_twh,
        'offshore_wind': lambda m: m[:, 3] / 100.0 * demand_twh,
        'ccs_ccgt':      lambda m: m[:, 4] / 100.0 * demand_twh,
        'hydro':         lambda m: np.minimum(m[:, 5] / 100.0 * demand_twh, HYDRO_CAP_TWH[iso]),
        'battery':       lambda m: (m[:, 7] + m[:, 8]) / 100.0 * demand_twh,
        'ldes':          lambda m: m[:, 9] / 100.0 * demand_twh,
    }

    for res in resources:
        fl = floor_twh.get(res, 0)
        if fl < 0.01:
            continue
        lcoe = excess_lcoes.get(res, 0)
        if lcoe <= 0:
            continue

        deployed = cols[res](mixes_arr)
        shortfall = np.maximum(0.0, fl - deployed)

        # Split shortfall into existing (free) and newbuild (priced)
        exist_twh = existing_twh.get(res, 0)
        existing_gap = np.maximum(0.0, np.minimum(shortfall, exist_twh - deployed))
        newbuild_gap = shortfall - np.maximum(0.0, existing_gap)

        excess += np.where(newbuild_gap > 0.01,
                           newbuild_gap / demand_twh * lcoe, 0.0)

    return excess


def precompute_excess_lcoes(sens, iso, overrides=None):
    """Precompute per-resource excess LCOEs for batch_floor_excess."""
    return {res: _get_excess_lcoe(res, sens, iso, overrides)
            for res in ['clean_firm', 'solar', 'wind', 'ccs_ccgt',
                        'hydro', 'battery', 'ldes']}


def find_cheapest_mix(mixes, floor_twh, existing_twh, demand_twh, iso,
                      sens, overrides=None, growth_factor=1.0,
                      use_effective_cost=False):
    """Find the cheapest mix from a list using vectorized batch evaluation.

    This replaces the per-mix Python loop pattern:
        for mix in filtered:
            result = compute_mix_cost(mix, ...)
            if result['total_cost'] < best: ...

    Args:
        mixes: list of mix arrays or (N, 10) ndarray
        floor_twh: per-resource floor TWh dict (for excess penalty)
        existing_twh: per-resource existing TWh dict
        demand_twh: scalar
        iso: ISO string
        sens: sensitivity dict
        overrides: optional LCOE overrides dict
        growth_factor: demand growth factor
        use_effective_cost: if True, minimize effective_cost (total/match_frac)
            instead of total_cost + excess. Used by Scenario B target finding.

    Returns: (best_idx, best_total_with_excess, costs_arr, mixes_arr)
        best_idx: index into mixes_arr of cheapest mix
        best_total_with_excess: total_cost + excess_penalty of cheapest
        costs_arr: (N,) array of total_cost per MWh
        mixes_arr: (N, 10) float64 array
    """
    if not mixes:
        return -1, float('inf'), np.array([]), np.array([]).reshape(0, 10)

    mixes_arr = _to_mix_array(mixes)
    params = _precompute_cost_params(sens, iso, demand_twh, overrides, growth_factor)

    if use_effective_cost:
        eff = batch_effective_costs(mixes_arr, params)
        best_idx = int(np.argmin(eff))
        return best_idx, eff[best_idx], eff, mixes_arr

    costs = batch_compute_total_costs(mixes_arr, params)

    # Add floor excess penalty
    excess_lcoes = precompute_excess_lcoes(sens, iso, overrides)
    excess = batch_floor_excess(mixes_arr, floor_twh, existing_twh,
                                demand_twh, iso, excess_lcoes)
    total_aug = costs + excess
    best_idx = int(np.argmin(total_aug))

    return best_idx, total_aug[best_idx], costs, mixes_arr


# ============================================================================
# PFS LOADER & FLOOR FILTERING
# ============================================================================

def _load_pfs_mixes(iso, threshold):
    """Load PFS mixes for a single ISO/threshold from step1 parquets.

    Returns (N, 11) numpy array: [cf, sol, wnd, osw, ccs, hyd, match, bat4, bat8, ldes, h2].
    Uses vectorized column extraction — no iterrows().
    """
    t_str = str(int(threshold)) if threshold == int(threshold) else str(threshold)
    pfs_path = Path(f'data/step1-pfs-parquets/{iso}_t{t_str}_raw_pfs.parquet')
    if not pfs_path.exists():
        return np.empty((0, 11), dtype=np.float64)
    df = pd.read_parquet(pfs_path)
    if df.empty:
        return np.empty((0, 11), dtype=np.float64)
    N = len(df)
    arr = np.zeros((N, 11), dtype=np.float64)
    arr[:, 0] = df['clean_firm'].values if 'clean_firm' in df.columns else 0
    arr[:, 1] = df['solar'].values if 'solar' in df.columns else 0
    arr[:, 2] = df['wind'].values if 'wind' in df.columns else 0
    arr[:, 3] = df['offshore_wind'].values if 'offshore_wind' in df.columns else 0
    arr[:, 4] = 0  # ccs_ccgt not in PFS
    arr[:, 5] = df['hydro'].values if 'hydro' in df.columns else 0
    arr[:, 6] = np.round(df['hourly_match_score'].values, 1) if 'hourly_match_score' in df.columns else 0
    arr[:, 7] = df['battery_dispatch_pct'].values if 'battery_dispatch_pct' in df.columns else 0
    arr[:, 8] = df['battery8_dispatch_pct'].values if 'battery8_dispatch_pct' in df.columns else 0
    arr[:, 9] = df['ldes_dispatch_pct'].values if 'ldes_dispatch_pct' in df.columns else 0
    arr[:, 10] = df['h2_dispatch_pct'].values if 'h2_dispatch_pct' in df.columns else 0
    return arr


def _filter_mixes_by_floor(mixes, floor_twh, demand_twh, iso):
    """Filter mixes to those meeting per-resource TWh floors."""
    floor_pct = {}
    for res in ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt']:
        floor_pct[res] = floor_twh.get(res, 0) / demand_twh * 100.0 if demand_twh > 0 else 0
    hydro_floor_twh = min(floor_twh.get('hydro', 0), HYDRO_CAP_TWH[iso])
    floor_pct['hydro'] = hydro_floor_twh / demand_twh * 100.0 if demand_twh > 0 else 0
    floor_pct['battery'] = floor_twh.get('battery', 0) / demand_twh * 100.0 if demand_twh > 0 else 0
    floor_pct['ldes'] = floor_twh.get('ldes', 0) / demand_twh * 100.0 if demand_twh > 0 else 0
    passed = []
    for mix in mixes:
        if len(mix) >= 11:
            cf, sol, wnd, osw, ccs, hyd = mix[0], mix[1], mix[2], mix[3], mix[4], mix[5]
            bat4, bat8, ldes = mix[7], mix[8], mix[9]
        else:
            cf, sol, wnd, ccs, hyd = mix[0], mix[1], mix[2], mix[3], mix[4]
            osw = 0
            bat4, bat8, ldes = mix[6], mix[7], mix[8]
        if cf < floor_pct['clean_firm'] - 0.01: continue
        if sol < floor_pct['solar'] - 0.01: continue
        if wnd < floor_pct['wind'] - 0.01: continue
        if osw < floor_pct.get('offshore_wind', 0) - 0.01: continue
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
    for res in ['clean_firm', 'solar', 'wind', 'offshore_wind']:
        floor_pct[res] = floor_twh.get(res, 0) / demand_twh * 100.0 if demand_twh > 0 else 0
    floor_pct['ccs_ccgt'] = 0
    hydro_floor_twh = min(floor_twh.get('hydro', 0), HYDRO_CAP_TWH[iso])
    floor_pct['hydro'] = hydro_floor_twh / demand_twh * 100.0 if demand_twh > 0 else 0
    floor_pct['battery'] = floor_twh.get('battery', 0) / demand_twh * 100.0 if demand_twh > 0 else 0
    floor_pct['ldes'] = floor_twh.get('ldes', 0) / demand_twh * 100.0 if demand_twh > 0 else 0
    passed = []
    for mix in mixes:
        if len(mix) >= 11:
            cf, sol, wnd, osw, ccs, hyd = mix[0], mix[1], mix[2], mix[3], mix[4], mix[5]
            bat4, bat8, ldes = mix[7], mix[8], mix[9]
        else:
            cf, sol, wnd, ccs, hyd = mix[0], mix[1], mix[2], mix[3], mix[4]
            osw = 0
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


def _rank_and_cap_mixes(mixes, floor_twh, demand_twh, iso, max_eval, label=''):
    """Rank mixes by Manhattan distance from floor and cap at max_eval.
    Works for both EF and PFS mixes. Prevents timeout on large ISOs."""
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
    tag = label or 'mixes'
    print(f"    ⚡ Capped {tag} from {len(mixes):,} → {len(capped):,} (ranked by floor proximity)")
    return capped


def _rank_and_cap_pfs(mixes, floor_twh, demand_twh, iso, max_eval=MAX_PFS_EVAL):
    """Rank PFS mixes by floor proximity and cap. Wrapper for backward compat."""
    return _rank_and_cap_mixes(mixes, floor_twh, demand_twh, iso, max_eval, label='PFS')


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
# SHARED HELPERS (gas backup, augmented results)
# ============================================================================

def recompute_gas_backup(iso, resource_twh, battery_twh, ldes_twh, gf):
    """Recompute gas backup MW from augmented clean capacity.

    Used after floor-ratchet augmentation to update gas needs.
    Returns (gas_backup_mw, new_gas_mw, existing_gas_used_mw, clean_peak_mw).
    """
    clean_peak_mw = 0
    for r, twh in resource_twh.items():
        pcc = PEAK_CAPACITY_CREDITS.get(r, 0)
        if pcc > 0 and twh > 0:
            clean_peak_mw += (twh * 1e6 / 8760) * pcc
    clean_peak_mw += (battery_twh * 1e6 / 8760) * PEAK_CAPACITY_CREDITS.get('battery', 0.95)
    clean_peak_mw += (ldes_twh * 1e6 / 8760) * PEAK_CAPACITY_CREDITS.get('ldes', 0.90)

    ra_peak_mw = PEAK_DEMAND_MW[iso] * gf * (1 + RESOURCE_ADEQUACY_MARGIN)
    gaf = GAS_AVAILABILITY_FACTOR[iso]
    gas_needed_mw = max(0, ra_peak_mw - clean_peak_mw) / gaf
    existing_gas_mw = EXISTING_GAS_CAPACITY_MW[iso]

    return (
        round(gas_needed_mw),
        round(max(0, gas_needed_mw - existing_gas_mw)),
        round(min(gas_needed_mw, existing_gas_mw)),
        round(clean_peak_mw),
    )


def build_augmented_result(best_result, best_mix, floor_twh, deployed,
                           best_total_cost, best_excess_per_mwh,
                           iso, demand_twh, gf, source='EF', extra_fields=None):
    """Build a floor-ratcheted augmented result dict from a best-mix selection.

    Handles: augmented resources, effective cost recalculation, gas backup,
    new-build cost tracking, dispatch parameters.

    Args:
        best_result: dict from compute_mix_cost()
        best_mix: raw mix vector (list or array)
        floor_twh: current floor dict {resource: TWh}
        deployed: dict from _mix_resource_twh()
        best_total_cost: total cost including floor excess
        best_excess_per_mwh: excess cost penalty $/MWh
        iso: ISO string
        demand_twh: demand in TWh for this threshold
        gf: demand growth factor
        source: mix source label ('EF', 'PFS+10', etc.)
        extra_fields: optional dict of additional fields to set on the result

    Returns: (aug_result_dict, augmented_dict, excess_twh_dict, total_excess)
    """
    demand_mwh = demand_twh * 1e6

    # Compute augmented = max(floor, deployed) per resource
    augmented = {}
    excess_twh_dict = {}
    for res in floor_twh:
        ef_val = deployed.get(res, 0)
        floor_val = floor_twh.get(res, 0)
        augmented[res] = max(ef_val, floor_val)
        excess_twh_dict[res] = max(0, floor_val - ef_val)

    total_excess = sum(excess_twh_dict.values())

    # Build result dict
    aug = dict(best_result)
    aug['total_cost'] = round(best_total_cost, 2)
    match_frac = best_result['match_score'] / 100.0
    aug['effective_cost'] = round(
        aug['total_cost'] / match_frac if match_frac > 0 else 0, 2)
    aug['incremental'] = round(
        aug['effective_cost'] - best_result['wholesale'], 2)

    # Augmented resource TWh
    aug['resource_twh'] = {res: augmented.get(res, 0) for res in RESOURCES}
    aug['battery_twh'] = augmented.get('battery', 0)
    aug['battery8_twh'] = 0
    aug['ldes_twh'] = augmented.get('ldes', 0)
    aug['demand_twh'] = demand_twh
    aug['mix_raw'] = list(best_mix)
    aug['mix_source'] = source

    # Recompute gas backup from augmented clean capacity
    gas_mw, new_gas_mw, existing_gas_used_mw, clean_peak_mw = recompute_gas_backup(
        iso, aug['resource_twh'], augmented.get('battery', 0),
        augmented.get('ldes', 0), gf)
    aug['gas_backup_mw'] = gas_mw
    aug['new_gas_mw'] = new_gas_mw
    aug['existing_gas_used_mw'] = existing_gas_used_mw
    aug['clean_peak_mw'] = clean_peak_mw

    # New-build cost tracking (for MAC calculation)
    aug['new_build_cost_total'] = (
        best_result['new_build_cost_total'] +
        best_excess_per_mwh * demand_mwh)
    aug['new_gen_twh'] = round(
        best_result['new_gen_twh'] +
        sum(v for k, v in excess_twh_dict.items() if k != 'hydro'), 3)

    # Dispatch parameters (11-element mix: score at [6], storage at [7..10])
    if len(best_mix) >= 11:
        aug['battery_dispatch_pct'] = float(best_mix[7])
        aug['battery8_dispatch_pct'] = float(best_mix[8])
        aug['ldes_dispatch_pct'] = float(best_mix[9])
        aug['h2_dispatch_pct'] = float(best_mix[10]) if len(best_mix) > 10 else 0.0
    else:
        aug['battery_dispatch_pct'] = float(best_mix[6])
        aug['battery8_dispatch_pct'] = float(best_mix[7])
        aug['ldes_dispatch_pct'] = float(best_mix[8])
        aug['h2_dispatch_pct'] = float(best_mix[9]) if len(best_mix) > 9 else 0.0
    aug['demand_mwh'] = demand_mwh

    if extra_fields:
        aug.update(extra_fields)

    return aug, augmented, excess_twh_dict, total_excess


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
    """Parse --iso and --max-pfs-eval CLI arguments. Returns (iso_list, max_pfs_eval).

    Supports multiple ISOs:
      --iso CAISO ERCOT PJM   (multiple positional after --iso)
      --iso CAISO --iso ERCOT  (repeated flag)
      --iso ALL                (all ISOs)
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--iso', type=str, nargs='+', default=['ALL'],
                        help='ISO(s) to process (CAISO, ERCOT, PJM, NYISO, NEISO, MISO, SPP, or ALL)')
    parser.add_argument('--max-pfs-eval', type=int, default=MAX_PFS_EVAL,
                        help=f'Max PFS mixes to evaluate per fallback (default: {MAX_PFS_EVAL})')
    args, _ = parser.parse_known_args()
    iso_list = args.iso
    if 'ALL' in iso_list:
        return ISOS, args.max_pfs_eval
    for iso in iso_list:
        if iso not in ISOS:
            print(f"ERROR: Unknown ISO: {iso}. Valid: {', '.join(ISOS)} or ALL")
            sys.exit(1)
    return iso_list, args.max_pfs_eval


def _serialize_result_entry(d):
    """Convert numpy types to JSON-serializable Python types."""
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
    return entry


def save_scenario_results(results, scenario_label, isos_processed):
    """Save scenario results as per-ISO JSON files for parallel-safe operation.

    Each ISO gets its own file: scenario_{label}_{iso}.json
    This allows parallel matrix jobs (one per ISO) to write without clobbering.
    """
    out_dir = 'data/step5-post-processing'
    os.makedirs(out_dir, exist_ok=True)

    for iso in isos_processed:
        iso_data = results.get(iso, {})
        output = {
            'metadata': {
                'scenario': scenario_label,
                'iso': iso,
                'timestamp': datetime.datetime.utcnow().isoformat(),
                'thresholds': list(THRESHOLDS),
            },
            'results': {}
        }
        for t, d in iso_data.items():
            t_key = str(int(t)) if t == int(t) else str(t)
            output['results'][t_key] = _serialize_result_entry(d)

        out_path = os.path.join(out_dir, f'scenario_{scenario_label.lower()}_{iso}.json')
        with open(out_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"  Saved {scenario_label} {iso}: {out_path} ({os.path.getsize(out_path) / 1024:.0f} KB)")


def load_scenario_results(scenario_label, isos=None):
    """Load scenario results from per-ISO JSON files. Merges all available ISOs.

    Args:
        scenario_label: 'A' or 'B'
        isos: optional list of ISOs to load (defaults to all found files)

    Returns: (results_dict, metadata) or (None, None) if no files found.
    """
    out_dir = 'data/step5-post-processing'
    label_lower = scenario_label.lower()
    pattern = f'scenario_{label_lower}_'

    results = {}
    metadata = None

    # Discover per-ISO files
    if not os.path.isdir(out_dir):
        return None, None

    for fname in sorted(os.listdir(out_dir)):
        if not fname.startswith(pattern) or not fname.endswith('.json'):
            continue
        # Extract ISO from filename: scenario_a_CAISO.json -> CAISO
        iso_part = fname[len(pattern):-len('.json')]
        if iso_part == 'results':
            # Legacy single-file format — load and merge
            path = os.path.join(out_dir, fname)
            with open(path) as f:
                data = json.load(f)
            for iso, iso_data in data['results'].items():
                if isos and iso not in isos:
                    continue
                results[iso] = {}
                for t_str, d in iso_data.items():
                    results[iso][float(t_str)] = d
            if metadata is None:
                metadata = data['metadata']
            continue

        # Per-ISO file
        if isos and iso_part not in isos:
            continue

        path = os.path.join(out_dir, fname)
        with open(path) as f:
            data = json.load(f)

        results[iso_part] = {}
        for t_str, d in data['results'].items():
            results[iso_part][float(t_str)] = d
        if metadata is None:
            metadata = data['metadata']

    if not results:
        return None, None

    return results, metadata
