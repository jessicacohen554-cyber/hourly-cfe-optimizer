#!/usr/bin/env python3
"""
Dual-scenario comparison: Pure Consequential vs Hourly Matching.

Compares two procurement strategies:
  Scenario A (Pure Consequential): chase cheapest $/tCO₂ sequentially,
    locking in prior resource commitments at each threshold step.
  Scenario B (Hourly Matching): target a high CFE threshold directly,
    co-optimizing the full resource mix with clean firm deployed along
    the FOAK→NOAK learning curve.

Uses feasible mixes from shared-data.js pipeline output + cost tables
from step3, then traces through dispatch_utils for emission rates.

Output: JSON + JS data files for the comparison dashboard page.
"""

import json
import os
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Import dispatch utilities for emission rates
sys.path.insert(0, str(Path(__file__).parent))
from dispatch_utils import (
    compute_fossil_retirement, BASE_DEMAND_TWH, GRID_MIX_SHARES,
    COAL_CAP_TWH, OIL_CAP_TWH,
)

# ============================================================================
# CONSTANTS (from step3_cost_optimization.py)
# ============================================================================

ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO']
THRESHOLDS = [50, 55, 60, 65, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 100]
RESOURCES = ['clean_firm', 'solar', 'wind', 'ccs_ccgt', 'hydro']

WHOLESALE_PRICES = {'CAISO': 30, 'ERCOT': 27, 'PJM': 34, 'NYISO': 42, 'NEISO': 41}
FUEL_ADJUSTMENTS = {
    'CAISO': {'Low': -5, 'Medium': 0, 'High': 10},
    'ERCOT': {'Low': -7, 'Medium': 0, 'High': 12},
    'PJM':   {'Low': -6, 'Medium': 0, 'High': 11},
    'NYISO': {'Low': -4, 'Medium': 0, 'High': 8},
    'NEISO': {'Low': -4, 'Medium': 0, 'High': 8},
}

LCOE_TABLES = {
    'solar': {
        'Low':    {'CAISO': 45, 'ERCOT': 40, 'PJM': 50, 'NYISO': 70, 'NEISO': 62},
        'Medium': {'CAISO': 60, 'ERCOT': 54, 'PJM': 65, 'NYISO': 92, 'NEISO': 82},
        'High':   {'CAISO': 78, 'ERCOT': 70, 'PJM': 85, 'NYISO': 120, 'NEISO': 107},
    },
    'wind': {
        'Low':    {'CAISO': 55, 'ERCOT': 30, 'PJM': 47, 'NYISO': 61, 'NEISO': 55},
        'Medium': {'CAISO': 73, 'ERCOT': 40, 'PJM': 62, 'NYISO': 81, 'NEISO': 73},
        'High':   {'CAISO': 95, 'ERCOT': 52, 'PJM': 81, 'NYISO': 105, 'NEISO': 95},
    },
    'battery': {
        'Low':    {'CAISO': 77, 'ERCOT': 69, 'PJM': 74, 'NYISO': 81, 'NEISO': 79},
        'Medium': {'CAISO': 102, 'ERCOT': 92, 'PJM': 98, 'NYISO': 108, 'NEISO': 105},
        'High':   {'CAISO': 133, 'ERCOT': 120, 'PJM': 127, 'NYISO': 140, 'NEISO': 137},
    },
    'battery8': {
        'Low':    {'CAISO': 85, 'ERCOT': 77, 'PJM': 82, 'NYISO': 90, 'NEISO': 88},
        'Medium': {'CAISO': 125, 'ERCOT': 113, 'PJM': 120, 'NYISO': 132, 'NEISO': 129},
        'High':   {'CAISO': 165, 'ERCOT': 149, 'PJM': 159, 'NYISO': 175, 'NEISO': 170},
    },
    'ldes': {
        'Low':    {'CAISO': 135, 'ERCOT': 116, 'PJM': 128, 'NYISO': 150, 'NEISO': 143},
        'Medium': {'CAISO': 180, 'ERCOT': 155, 'PJM': 170, 'NYISO': 200, 'NEISO': 190},
        'High':   {'CAISO': 234, 'ERCOT': 202, 'PJM': 221, 'NYISO': 260, 'NEISO': 247},
    },
}

TX_TABLES = {
    'wind':       {'None': 0, 'Low': {'CAISO': 4, 'ERCOT': 3, 'PJM': 5, 'NYISO': 7, 'NEISO': 6},
                   'Medium': {'CAISO': 8, 'ERCOT': 6, 'PJM': 10, 'NYISO': 14, 'NEISO': 12},
                   'High': {'CAISO': 14, 'ERCOT': 10, 'PJM': 18, 'NYISO': 22, 'NEISO': 20}},
    'solar':      {'None': 0, 'Low': {'CAISO': 1, 'ERCOT': 1, 'PJM': 2, 'NYISO': 3, 'NEISO': 3},
                   'Medium': {'CAISO': 3, 'ERCOT': 3, 'PJM': 5, 'NYISO': 7, 'NEISO': 6},
                   'High': {'CAISO': 6, 'ERCOT': 5, 'PJM': 9, 'NYISO': 12, 'NEISO': 10}},
    'clean_firm': {'None': 0, 'Low': {'CAISO': 1, 'ERCOT': 1, 'PJM': 1, 'NYISO': 2, 'NEISO': 2},
                   'Medium': {'CAISO': 3, 'ERCOT': 2, 'PJM': 3, 'NYISO': 5, 'NEISO': 4},
                   'High': {'CAISO': 6, 'ERCOT': 4, 'PJM': 6, 'NYISO': 9, 'NEISO': 7}},
    'ccs_ccgt':   {'None': 0, 'Low': {'CAISO': 1, 'ERCOT': 1, 'PJM': 1, 'NYISO': 2, 'NEISO': 2},
                   'Medium': {'CAISO': 2, 'ERCOT': 2, 'PJM': 3, 'NYISO': 4, 'NEISO': 3},
                   'High': {'CAISO': 4, 'ERCOT': 3, 'PJM': 5, 'NYISO': 7, 'NEISO': 6}},
    'battery':    {'None': 0, 'Low': {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 1, 'NEISO': 1},
                   'Medium': {'CAISO': 1, 'ERCOT': 1, 'PJM': 1, 'NYISO': 2, 'NEISO': 2},
                   'High': {'CAISO': 2, 'ERCOT': 2, 'PJM': 3, 'NYISO': 4, 'NEISO': 3}},
    'battery8':   {'None': 0, 'Low': {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 1, 'NEISO': 1},
                   'Medium': {'CAISO': 1, 'ERCOT': 1, 'PJM': 1, 'NYISO': 2, 'NEISO': 2},
                   'High': {'CAISO': 2, 'ERCOT': 2, 'PJM': 3, 'NYISO': 4, 'NEISO': 3}},
    'ldes':       {'None': 0, 'Low': {'CAISO': 1, 'ERCOT': 1, 'PJM': 1, 'NYISO': 2, 'NEISO': 2},
                   'Medium': {'CAISO': 2, 'ERCOT': 2, 'PJM': 3, 'NYISO': 4, 'NEISO': 3},
                   'High': {'CAISO': 4, 'ERCOT': 3, 'PJM': 5, 'NYISO': 7, 'NEISO': 6}},
    'hydro':      {'None': 0, 'Low': 0, 'Medium': 0, 'High': 0},
}

UPRATE_LCOE = {'L': 15, 'M': 25, 'H': 40}
NUCLEAR_NEWBUILD_LCOE = {
    'L': {'CAISO': 70, 'ERCOT': 68, 'PJM': 72, 'NYISO': 75, 'NEISO': 73},
    'M': {'CAISO': 95, 'ERCOT': 90, 'PJM': 105, 'NYISO': 110, 'NEISO': 108},
    'H': {'CAISO': 140, 'ERCOT': 135, 'PJM': 160, 'NYISO': 170, 'NEISO': 165},
}
GEOTHERMAL_LCOE = {'L': 63, 'M': 88, 'H': 110}
GEO_CAP_TWH = 39.0
CCS_LCOE_45Q_ON = {
    'L': {'CAISO': 58, 'ERCOT': 52, 'PJM': 62, 'NYISO': 78, 'NEISO': 75},
    'M': {'CAISO': 86, 'ERCOT': 71, 'PJM': 79, 'NYISO': 99, 'NEISO': 96},
    'H': {'CAISO': 115, 'ERCOT': 92, 'PJM': 102, 'NYISO': 128, 'NEISO': 122},
}

EXISTING_NUCLEAR_GW = {'CAISO': 2.3, 'ERCOT': 2.7, 'PJM': 32.0, 'NYISO': 3.4, 'NEISO': 3.5}
UPRATE_CAP_TWH = {iso: round(gw * 0.05 * 0.90 * 8760 / 1e3, 3)
                  for iso, gw in EXISTING_NUCLEAR_GW.items()}

PEAK_DEMAND_MW = {'CAISO': 43860, 'ERCOT': 83597, 'PJM': 160560, 'NYISO': 31857, 'NEISO': 25898}
EXISTING_GAS_CAPACITY_MW = {'CAISO': 37000, 'ERCOT': 55000, 'PJM': 75000, 'NYISO': 18000, 'NEISO': 14000}
NEW_CCGT_COST_KW_YR = {'CAISO': 112, 'ERCOT': 89, 'PJM': 99, 'NYISO': 114, 'NEISO': 105}
EXISTING_GAS_FOM_KW_YR = {'CAISO': 16, 'ERCOT': 13, 'PJM': 14, 'NYISO': 17, 'NEISO': 15}
PEAK_CAPACITY_CREDITS = {
    'clean_firm': 1.0, 'solar': 0.30, 'wind': 0.10, 'ccs_ccgt': 0.90,
    'hydro': 0.50, 'battery': 0.95, 'battery8': 0.95, 'ldes': 0.90,
}
GAS_AVAILABILITY_FACTOR = {
    'CAISO': 0.88, 'ERCOT': 0.83, 'PJM': 0.82, 'NYISO': 0.82, 'NEISO': 0.85,
}
RESOURCE_ADEQUACY_MARGIN = 0.15

LEVEL_NAME = {'L': 'Low', 'M': 'Medium', 'H': 'High'}

# SBTi timeline mapping
# Canonical threshold → target year mapping (SBTi-interpolated)
# Imported from step3 when available; hardcoded fallback for standalone use
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from step3_cost_optimization import THRESHOLD_TARGET_YEARS
    SBTI_YEAR_MAP = THRESHOLD_TARGET_YEARS
except ImportError:
    SBTI_YEAR_MAP = {
        50: 2030, 55: 2031, 60: 2033, 65: 2034, 70: 2035, 75: 2036, 80: 2037,
        85: 2038, 87.5: 2039, 90: 2040, 92.5: 2043,
        95: 2045, 97.5: 2048, 99: 2049, 100: 2050,
    }

# Demand growth rates (decimal) — medium growth scenario for both A and B
DEMAND_GROWTH_RATES = {
    'CAISO': 0.019, 'ERCOT': 0.035, 'PJM': 0.024,
    'NYISO': 0.020, 'NEISO': 0.018,
}
BASE_YEAR = 2025


def get_demand_twh(iso, threshold):
    """Get demand TWh for a given ISO and threshold, accounting for medium demand growth.

    Each threshold maps to a year via SBTI_YEAR_MAP. Demand grows at
    the ISO's medium growth rate compounded annually from 2025.
    """
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

# Zone definitions for consequential queue
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
    {'label': '99→100%', 'start': 99, 'end': 100},
]


# ============================================================================
# SCENARIO DEFINITIONS
# ============================================================================

# Scenario A: "Pure Consequential"
# Sequential path-dependent procurement — chase cheapest $/tCO₂ at each step
SCENARIO_A = {
    'name': 'Pure Consequential',
    'short': 'pure_consequential',
    'description': 'Sequential procurement chasing cheapest $/tCO₂ — resources lock in at each threshold step',
    'toggles': {
        'ren': 'L',        # Low renewable (cheap solar/wind)
        'firm': 'H',       # High firm gen (expensive nuclear)
        'batt': 'L',       # Low battery
        'ldes_lvl': 'H',   # High LDES (expensive)
        'fuel': 'M',       # Medium fossil fuel
        'tx': 'M',         # Medium transmission
        'ccs': 'H',        # High CCS (expensive)
        'q45': '1',        # 45Q on
        'geo': 'H',        # High geothermal (CAISO only)
    },
}

# Scenario B: "Hourly Matching"
# Graduated clean firm deployment along FOAK→NOAK learning curve.
# Early thresholds see higher clean firm costs (FOAK), later thresholds
# reach NOAK pricing as cumulative deployment drives learning.
# Mature tech (solar, wind, battery) stays at Low cost throughout.
SCENARIO_B = {
    'name': 'Hourly Matching',
    'short': 'hourly_matching',
    'description': 'Hourly matching with graduated clean firm deployment — FOAK→NOAK learning curve drives costs from High to Low over the SBTi timeline, yielding increasing clean firm investment at each threshold',
    'toggles': {
        'ren': 'L',        # Low renewable (mature tech, already cheap)
        'firm': 'L',       # Target NOAK (learning curve interpolates H→L per threshold)
        'batt': 'L',       # Low battery (mature tech)
        'ldes_lvl': 'L',   # Target NOAK (learning curve interpolates H→L)
        'fuel': 'M',       # Medium fossil fuel
        'tx': 'M',         # Medium transmission
        'ccs': 'L',        # Target NOAK (learning curve interpolates H→L)
        'q45': '1',        # 45Q on
        'geo': 'L',        # Target NOAK (CAISO only, learning curve interpolates H→L)
    },
}

SCENARIOS = [SCENARIO_A, SCENARIO_B]


# ============================================================================
# LEARNING CURVE FRACTION
# ============================================================================

def learning_fraction(threshold):
    """Map CFE threshold to FOAK→NOAK learning curve fraction [0, 1].

    0 = pure FOAK (High cost), 1 = full NOAK (Low cost).
    Uses a slightly convex ramp with a 5% floor — ensures some early
    investment to seed the learning curve, with accelerating cost reduction
    as cumulative deployment drives Wright's Law doublings.

    Resulting nuclear LCOE trajectory (PJM example, H=$160 → L=$72):
      50% (2025): $156/MWh — FOAK, expensive but invested
      70% (2029): $140/MWh — early learning
      80% (2032): $121/MWh — significant cost reduction
      90% (2040): $98/MWh  — approaching NOAK
      95% (2045): $86/MWh  — near NOAK
      100%(2050): $72/MWh  — full NOAK
    """
    t_norm = (threshold - 50) / 50  # 50%→0, 100%→1
    # Floor of 5% ensures some cost reduction even at earliest thresholds,
    # reflecting minimum FOAK investment to begin the learning curve.
    # Exponent 1.8 creates a convex ramp: slow start, accelerating through
    # mid-range (80-95%), converging on NOAK at high thresholds.
    return 0.05 + 0.95 * (t_norm ** 1.8)


# ============================================================================
# PARSE FEASIBLE MIXES FROM SHARED-DATA.JS
# ============================================================================

def parse_feasible_mixes(js_path='dashboard/js/shared-data.js'):
    """Parse FEASIBLE_MIXES from the shared-data.js file."""
    with open(js_path) as f:
        content = f.read()

    # Find the FEASIBLE_MIXES block
    start = content.find('const FEASIBLE_MIXES = {')
    if start < 0:
        raise ValueError("FEASIBLE_MIXES not found in shared-data.js")

    # Find matching closing brace
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

    # Convert JS object to valid JSON
    # Replace unquoted keys (CAISO: → "CAISO":)
    block = re.sub(r'(\b[A-Z]+\b)\s*:', r'"\1":', block)
    # Remove trailing commas before } or ]
    block = re.sub(r',\s*([}\]])', r'\1', block)

    mixes = json.loads(block)
    # mixes[iso][threshold_str] = list of [cf%, sol%, wnd%, ccs%, hyd%, proc%, match%, bat4%, bat8%, ldes%]
    return mixes




# ============================================================================
# COST FUNCTION (simplified from step3)
# ============================================================================

def get_tx(rtype, tx_level, iso):
    val = TX_TABLES[rtype][tx_level]
    if isinstance(val, dict):
        return val[iso]
    return val


def _resource_new_build_lcoe(res, sens, iso):
    """Get new-build LCOE+transmission for a resource under given sensitivity toggles.

    Used to price locked-in excess resources from prior path-dependent steps.
    Returns $/MWh delivered.
    """
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
        return 0  # Existing-only, wholesale-priced
    elif res == 'battery':
        return LCOE_TABLES['battery'][batt_name][iso] + get_tx('battery', tx_name, iso)
    elif res == 'ldes':
        return LCOE_TABLES['ldes'][ldes_name][iso] + get_tx('ldes', tx_name, iso)
    return 0


def compute_mix_cost(mix, sens, iso, demand_twh, overrides=None, growth_factor=1.0):
    """
    Compute total system cost per MWh for a single mix under a sensitivity scenario.

    mix: [cf%, sol%, wnd%, ccs%, hyd%, proc%, match%, bat4%, bat8%, ldes%]
    overrides: optional dict with explicit LCOE values (bypasses toggle lookups):
        nuclear_lcoe, ccs_lcoe, geo_lcoe, ldes_lcoe, uprate_lcoe
    growth_factor: demand growth multiplier (>1 means demand has grown from base year).
        Existing resource percentages are scaled DOWN by this factor so that existing
        absolute TWh stays fixed as demand grows.
    Returns: dict with cost details + gas backup
    """
    cf_pct, sol_pct, wnd_pct, ccs_pct, hyd_pct = mix[0], mix[1], mix[2], mix[3], mix[4]
    proc_pct, match_score = mix[5], mix[6]
    bat4_pct, bat8_pct, ldes_pct = mix[7], mix[8], mix[9]

    proc = proc_pct / 100.0
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

    # Existing resource percentages scaled for demand growth.
    # Existing infrastructure produces fixed absolute TWh (2025 levels).
    # As demand grows, existing covers a smaller percentage.
    existing = {k: v / growth_factor for k, v in GRID_MIX_SHARES[iso].items()}
    wholesale = max(5, WHOLESALE_PRICES[iso] + FUEL_ADJUSTMENTS[iso][fuel_name])

    # CCS price
    if overrides and 'ccs_lcoe' in overrides:
        ccs_lcoe = overrides['ccs_lcoe']
    else:
        ccs_table = CCS_LCOE_45Q_ON if q45 == '1' else None
        ccs_lcoe = ccs_table[ccs_lev][iso]
    ccs_tx = get_tx('ccs_ccgt', tx_name, iso)
    ccs_price = ccs_lcoe + ccs_tx

    # Nuclear price
    if overrides and 'nuclear_lcoe' in overrides:
        nuclear_price = overrides['nuclear_lcoe'] + get_tx('clean_firm', tx_name, iso)
    else:
        nuclear_price = NUCLEAR_NEWBUILD_LCOE[firm_lev][iso] + get_tx('clean_firm', tx_name, iso)
    remaining_price = min(nuclear_price, ccs_price)

    # Geothermal price
    geo_price = 0.0
    if iso == 'CAISO' and geo_lev:
        if overrides and 'geo_lcoe' in overrides:
            geo_price = overrides['geo_lcoe'] + get_tx('clean_firm', tx_name, iso)
        else:
            geo_price = GEOTHERMAL_LCOE[geo_lev] + get_tx('clean_firm', tx_name, iso)

    # Demand-weighted percentages
    sol_demand = proc * sol_pct
    wnd_demand = proc * wnd_pct
    hyd_demand = proc * hyd_pct
    ccs_demand = proc * ccs_pct
    cf_demand = proc * cf_pct

    # Existing/new splits
    sol_existing = min(sol_demand, existing['solar'])
    sol_new = max(0, sol_demand - existing['solar'])
    wnd_existing = min(wnd_demand, existing['wind'])
    wnd_new = max(0, wnd_demand - existing['wind'])
    ccs_existing = min(ccs_demand, existing.get('ccs_ccgt', 0))
    ccs_new = max(0, ccs_demand - existing.get('ccs_ccgt', 0))
    cf_existing = min(cf_demand, existing['clean_firm'])
    cf_new = max(0, cf_demand - existing['clean_firm'])

    # Clean firm tranche allocation
    new_cf_twh = cf_new / 100.0 * demand_twh
    uprate_twh = min(new_cf_twh, UPRATE_CAP_TWH[iso])
    remaining_after_uprate = max(0, new_cf_twh - uprate_twh)

    geo_twh = 0.0
    remaining_after_geo = remaining_after_uprate
    if iso == 'CAISO':
        geo_twh = min(remaining_after_uprate, GEO_CAP_TWH)
        remaining_after_geo = max(0, remaining_after_uprate - geo_twh)

    # Gas backup (scenario-invariant given the mix)
    demand_mwh = demand_twh * 1e6
    avg_demand_mw = demand_mwh / 8760
    # Peak demand grows proportionally with annual demand
    peak_mw = PEAK_DEMAND_MW[iso] * growth_factor
    ra_peak_mw = peak_mw * (1 + RESOURCE_ADEQUACY_MARGIN)

    clean_peak_mw = (
        proc * cf_pct / 100 * avg_demand_mw * PEAK_CAPACITY_CREDITS['clean_firm'] +
        proc * sol_pct / 100 * avg_demand_mw * PEAK_CAPACITY_CREDITS['solar'] +
        proc * wnd_pct / 100 * avg_demand_mw * PEAK_CAPACITY_CREDITS['wind'] +
        proc * ccs_pct / 100 * avg_demand_mw * PEAK_CAPACITY_CREDITS['ccs_ccgt'] +
        proc * hyd_pct / 100 * avg_demand_mw * PEAK_CAPACITY_CREDITS['hydro'] +
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

    # Uprate and LDES prices (with optional override support)
    uprate_price = overrides.get('uprate_lcoe', UPRATE_LCOE[firm_lev]) if overrides else UPRATE_LCOE[firm_lev]
    if overrides and 'ldes_lcoe' in overrides:
        ldes_price = overrides['ldes_lcoe'] + get_tx('ldes', tx_name, iso)
    else:
        ldes_price = LCOE_TABLES['ldes'][ldes_name][iso] + get_tx('ldes', tx_name, iso)

    # Total cost = Σ(coefficient × price) + gas_cost
    # Coefficients: fraction of demand priced at each source
    total_cost = (
        (sol_existing + wnd_existing + hyd_demand + ccs_existing + cf_existing) / 100.0 * wholesale +
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

    # New-build cost tracking (for MAC = LCOE / displaced_rate)
    existing_cost_per_mwh = (sol_existing + wnd_existing + hyd_demand + ccs_existing + cf_existing) / 100.0 * wholesale
    new_build_per_mwh = total_cost - existing_cost_per_mwh - gas_cost
    new_gen_twh = (sol_new + wnd_new + ccs_new + cf_new) / 100.0 * demand_twh
    new_build_cost_total = new_build_per_mwh * demand_mwh
    if new_gen_twh > 0.01:
        blended_new_lcoe = new_build_per_mwh * demand_twh / new_gen_twh
    else:
        blended_new_lcoe = 0.0

    eff_cost = total_cost / match_frac if match_frac > 0 else 0
    incremental = eff_cost - wholesale

    # Resource TWh
    resource_twh = {}
    for res, pct in zip(RESOURCES, [cf_pct, sol_pct, wnd_pct, ccs_pct, hyd_pct]):
        resource_twh[res] = pct / 100.0 * proc * demand_twh

    return {
        'total_cost': round(total_cost, 2),
        'effective_cost': round(eff_cost, 2),
        'incremental': round(incremental, 2),
        'wholesale': wholesale,
        'resource_twh': resource_twh,
        'resource_pct': {'clean_firm': cf_pct, 'solar': sol_pct, 'wind': wnd_pct,
                         'ccs_ccgt': ccs_pct, 'hydro': hyd_pct},
        'procurement_pct': proc_pct,
        'match_score': match_score,
        'battery_twh': bat4_pct / 100.0 * demand_twh,
        'battery8_twh': bat8_pct / 100.0 * demand_twh,
        'ldes_twh': ldes_pct / 100.0 * demand_twh,
        'gas_backup_mw': round(gas_needed_mw),
        'new_gas_mw': round(new_gas_mw),
        'existing_gas_used_mw': round(existing_gas_used_mw),
        'clean_peak_mw': round(clean_peak_mw),
        'new_build_cost_total': new_build_cost_total,
        'new_gen_twh': round(new_gen_twh, 3),
        'blended_new_lcoe': round(blended_new_lcoe, 2),
    }


# ============================================================================
# FIND OPTIMAL MIX PER (ISO, THRESHOLD, SCENARIO)
# ============================================================================

def find_optimal_mixes(feasible_mixes, scenario, demand_twh_map):
    """For each ISO × threshold, find the cheapest mix under this scenario's costs."""
    results = {}
    sens = scenario['toggles']

    for iso in ISOS:
        iso_results = {}
        iso_sens = dict(sens)
        if iso != 'CAISO':
            iso_sens['geo'] = None

        for t in THRESHOLDS:
            t_str = str(int(t)) if t == int(t) else str(t)
            mixes = feasible_mixes.get(iso, {}).get(t_str, [])
            if not mixes:
                continue

            best_cost = float('inf')
            best_result = None

            for mix in mixes:
                result = compute_mix_cost(mix, iso_sens, iso, demand_twh_map[iso])
                if result['effective_cost'] < best_cost:
                    best_cost = result['effective_cost']
                    best_result = result

            if best_result:
                iso_results[t] = best_result

        results[iso] = iso_results
    return results


def _mix_resource_twh(mix, demand_twh):
    """Extract deployed resource TWh from a raw mix vector for floor comparison.

    mix = [cf%, sol%, wnd%, ccs%, hyd%, proc%, match%, bat4%, bat8%, ldes%]
    Returns dict of resource → TWh deployed.
    """
    cf, sol, wnd, ccs, hyd, proc_pct = mix[0], mix[1], mix[2], mix[3], mix[4], mix[5]
    bat4, bat8, ldes = mix[7], mix[8], mix[9]
    proc = proc_pct / 100.0
    return {
        'clean_firm': proc * cf / 100.0 * demand_twh,
        'solar':      proc * sol / 100.0 * demand_twh,
        'wind':       proc * wnd / 100.0 * demand_twh,
        'ccs_ccgt':   proc * ccs / 100.0 * demand_twh,
        'hydro':      proc * hyd / 100.0 * demand_twh,
        'battery':    (bat4 + bat8) / 100.0 * demand_twh,
        'ldes':       ldes / 100.0 * demand_twh,
    }


def find_optimal_mixes_sequential(feasible_mixes, scenario, demand_twh_map):
    """Path-dependent sequential optimization for the consequential scenario.

    Both scenarios target 95% clean as the SBTi destination. Scenario A reaches
    it via sequential path-dependent procurement — chasing cheapest $/tCO₂ at each
    step, locking in prior resource commitments (floor ratchet).

    Algorithm:
      1. Find cost-optimal EF mix at 95% as the deployment target ("north star").
      2. At each threshold 50→100%, find the cheapest EF mix that doesn't overshoot
         any resource beyond the 95% target + 20% slack (prevents wasteful overbuilding).
      3. Overlay locked-in excess from prior steps: augmented = max(floor, EF_target).
      4. Price excess at LCOE — the cost penalty of path-dependent overbuilding.
      5. Floor ratchets up to augmented level.

    Demand grows year-over-year per SBTI timeline (medium growth rate).
    Existing resource floors are fixed at 2025 absolute TWh levels.
    """
    results = {}
    sens = scenario['toggles']

    for iso in ISOS:
        iso_results = {}
        iso_sens = dict(sens)
        if iso != 'CAISO':
            iso_sens['geo'] = None

        base_demand_twh = demand_twh_map[iso]

        # ------------------------------------------------------------------
        # Step 1: Find cost-optimal mix at 95% as the deployment target
        # ------------------------------------------------------------------
        t95_str = '95'
        t95_mixes = feasible_mixes.get(iso, {}).get(t95_str, [])
        gf_95 = get_demand_growth_factor(iso, 95)
        demand_95 = base_demand_twh * gf_95

        best_95_cost = float('inf')
        best_95_mix = None
        for mix in t95_mixes:
            result = compute_mix_cost(mix, iso_sens, iso, demand_95, growth_factor=gf_95)
            if result['effective_cost'] < best_95_cost:
                best_95_cost = result['effective_cost']
                best_95_mix = mix

        # Resource caps from 95% target (with 20% slack for thresholds > 95%)
        target_95_deployed = _mix_resource_twh(best_95_mix, demand_95) if best_95_mix else {}
        if target_95_deployed:
            print(f"  {iso} 95% target: CF={target_95_deployed['clean_firm']:.0f} TWh, "
                  f"Sol={target_95_deployed['solar']:.0f}, Wnd={target_95_deployed['wind']:.0f}, "
                  f"CCS={target_95_deployed['ccs_ccgt']:.0f} (demand={demand_95:.0f} TWh)")

        # ------------------------------------------------------------------
        # Step 2: Sequential optimization with floor ratchet
        # ------------------------------------------------------------------
        # Floor starts at existing resource levels in ABSOLUTE TWh (2025 base year).
        existing = GRID_MIX_SHARES[iso]
        floor = {
            'clean_firm': existing.get('clean_firm', 0) / 100.0 * base_demand_twh,
            'solar': existing.get('solar', 0) / 100.0 * base_demand_twh,
            'wind': existing.get('wind', 0) / 100.0 * base_demand_twh,
            'ccs_ccgt': existing.get('ccs_ccgt', 0) / 100.0 * base_demand_twh,
            'hydro': existing.get('hydro', 0) / 100.0 * base_demand_twh,
            'battery': 0, 'ldes': 0,
        }

        for t in THRESHOLDS:
            t_str = str(int(t)) if t == int(t) else str(t)
            mixes = feasible_mixes.get(iso, {}).get(t_str, [])
            if not mixes:
                continue

            # Year-specific demand with growth
            gf = get_demand_growth_factor(iso, t)
            demand_twh = base_demand_twh * gf
            demand_mwh = demand_twh * 1e6

            # 2a. Filter EF mixes: prefer those that don't massively overshoot the
            #     95% target. Scale the 95% target to this threshold's demand level.
            #     Allow 20% slack for thresholds above 95% that may need more resources.
            demand_ratio = demand_twh / demand_95 if demand_95 > 0 else 1.0
            slack = 1.2 if t <= 95 else 1.5  # more slack above 95%

            if target_95_deployed:
                def _within_target(mix):
                    deployed = _mix_resource_twh(mix, demand_twh)
                    for res in ['clean_firm', 'solar', 'wind', 'ccs_ccgt']:
                        cap = target_95_deployed.get(res, 0) * demand_ratio * slack
                        if cap > 1.0 and deployed.get(res, 0) > cap:
                            return False
                    return True

                target_passing = [m for m in mixes if _within_target(m)]
            else:
                target_passing = list(mixes)

            # Fall back to all mixes if target filter is too restrictive
            if not target_passing:
                target_passing = list(mixes)

            # 2b. Score ALL mixes by augmented cost (EF cost + floor excess penalty).
            #     This ensures we pick the mix that maximizes use of locked-in resources
            #     from prior steps, not just the cheapest mix in isolation.
            best_augmented_cost = float('inf')
            best_result = None
            best_mix = None
            best_excess_twh = None
            best_excess_per_mwh = 0

            for mix in target_passing:
                result = compute_mix_cost(mix, iso_sens, iso, demand_twh, growth_factor=gf)

                # Compute augmented cost: how expensive is this mix INCLUDING floor excess?
                ef_deployed = _mix_resource_twh(mix, demand_twh)
                mix_excess_per_mwh = 0.0
                for res in floor:
                    excess = max(0, floor.get(res, 0) - ef_deployed.get(res, 0))
                    if excess > 0.01:
                        lcoe = _resource_new_build_lcoe(res, iso_sens, iso)
                        mix_excess_per_mwh += excess / demand_twh * lcoe

                augmented_eff = (result['total_cost'] + mix_excess_per_mwh) / (result['match_score'] / 100) if result['match_score'] > 0 else float('inf')

                if augmented_eff < best_augmented_cost:
                    best_augmented_cost = augmented_eff
                    best_result = result
                    best_mix = mix
                    best_excess_per_mwh = mix_excess_per_mwh

            if not best_result:
                continue

            # 3. Compute augmented resources = max(floor, EF_target) per resource
            ef_deployed = _mix_resource_twh(best_mix, demand_twh)
            augmented = {}
            excess_twh = {}
            for res in floor:
                ef_val = ef_deployed.get(res, 0)
                floor_val = floor.get(res, 0)
                augmented[res] = max(ef_val, floor_val)
                excess_twh[res] = max(0, floor_val - ef_val)

            total_excess = sum(excess_twh.values())
            excess_cost_per_mwh = best_excess_per_mwh

            # 4. Build augmented result — EF cost + excess cost penalty
            augmented_result = dict(best_result)
            augmented_result['total_cost'] = round(best_result['total_cost'] + excess_cost_per_mwh, 2)
            match_frac = best_result['match_score'] / 100
            augmented_result['effective_cost'] = round(
                augmented_result['total_cost'] / match_frac if match_frac > 0 else 0, 2)
            augmented_result['incremental'] = round(
                augmented_result['effective_cost'] - best_result['wholesale'], 2)

            # Override resource TWh with augmented values (floor + EF target)
            augmented_result['resource_twh'] = {
                res: augmented.get(res, 0) for res in RESOURCES}
            augmented_result['battery_twh'] = augmented.get('battery', 0)
            augmented_result['ldes_twh'] = augmented.get('ldes', 0)

            # 5. Recompute gas backup from augmented clean capacity
            clean_peak_mw = 0
            for r, twh in augmented_result['resource_twh'].items():
                pcc = PEAK_CAPACITY_CREDITS.get(r, 0)
                if pcc > 0 and twh > 0:
                    clean_peak_mw += (twh * 1e6 / 8760) * pcc
            clean_peak_mw += (augmented.get('battery', 0) * 1e6 / 8760) * PEAK_CAPACITY_CREDITS.get('battery', 0.95)
            clean_peak_mw += (augmented.get('ldes', 0) * 1e6 / 8760) * PEAK_CAPACITY_CREDITS.get('ldes', 0.90)

            ra_peak_mw = PEAK_DEMAND_MW[iso] * gf * (1 + RESOURCE_ADEQUACY_MARGIN)
            gaf = GAS_AVAILABILITY_FACTOR[iso]
            gas_needed_mw = max(0, ra_peak_mw - clean_peak_mw) / gaf
            augmented_result['gas_backup_mw'] = round(gas_needed_mw)
            augmented_result['new_gas_mw'] = round(max(0, gas_needed_mw - EXISTING_GAS_CAPACITY_MW[iso]))
            augmented_result['clean_peak_mw'] = round(clean_peak_mw)

            # Track new-build cost (for MAC calculation)
            augmented_result['new_build_cost_total'] = (
                best_result['new_build_cost_total'] + excess_cost_per_mwh * demand_mwh)
            augmented_result['new_gen_twh'] = round(
                best_result['new_gen_twh'] + sum(
                    v for k, v in excess_twh.items() if k != 'hydro'), 3)

            if total_excess > 1.0:
                print(f"  ↗ {iso} {t}%: {total_excess:.0f} TWh excess locked in "
                      f"(+${excess_cost_per_mwh:.1f}/MWh penalty)")

            iso_results[t] = augmented_result

            # 6. Update floor: augmented resources become the new locked-in floor
            floor = dict(augmented)

        results[iso] = iso_results
    return results


def _build_learning_overrides(iso, frac):
    """Build LCOE overrides interpolated between High (FOAK) and Low (NOAK)."""
    overrides = {}
    nuc_h = NUCLEAR_NEWBUILD_LCOE['H'][iso]
    nuc_l = NUCLEAR_NEWBUILD_LCOE['L'][iso]
    overrides['nuclear_lcoe'] = nuc_h + frac * (nuc_l - nuc_h)
    overrides['uprate_lcoe'] = UPRATE_LCOE['H'] + frac * (UPRATE_LCOE['L'] - UPRATE_LCOE['H'])
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


def find_optimal_mixes_learning_curve(feasible_mixes, scenario, demand_twh_map,
                                     target_threshold=95):
    """Scenario B: Forward-looking hourly matching with graduated clean firm deployment.

    Strategy: select cost-optimal mix at the target threshold (near-NOAK pricing) as
    the deployment "north star", then work backwards to deploy resources gradually:

      - Clean firm / CCS: deployed along FOAK→NOAK learning curve, starting low
        and accelerating as costs decline. At each threshold, the target clean firm
        is proportional to the target mix × learning_fraction(t)/learning_fraction(target).
      - Wind / solar / uprates: deployed per pure cost optimization (cheapest $/tCO₂).
        These are mature technologies at Low cost from day one.
      - LDES: also on learning curve (emerging technology).

    No path-dependency floor — each threshold optimized independently.
    No fallback — physics-feasible EF mixes scored by cost + alignment with
    the graduated clean firm deployment target.

    Demand grows year-over-year per SBTI timeline (medium growth rate).

    Parameters:
        target_threshold: the CFE% target to work backwards from (default 95).
    """
    results = {}
    sens = scenario['toggles']
    FRAC_TARGET = learning_fraction(target_threshold)
    CF_ALIGNMENT_WEIGHT = 5.0

    for iso in ISOS:
        iso_results = {}
        iso_sens = dict(sens)
        if iso != 'CAISO':
            iso_sens['geo'] = None

        base_demand_twh = demand_twh_map[iso]

        # ------------------------------------------------------------------
        # Step 1: Find cost-optimal mix at target threshold as deployment target
        # ------------------------------------------------------------------
        t_str = str(int(target_threshold)) if target_threshold == int(target_threshold) else str(target_threshold)
        t_mixes = feasible_mixes.get(iso, {}).get(t_str, [])
        gf_target = get_demand_growth_factor(iso, target_threshold)
        demand_target = base_demand_twh * gf_target
        overrides_target = _build_learning_overrides(iso, FRAC_TARGET)

        best_target_cost = float('inf')
        best_target_mix = None
        for mix in t_mixes:
            result = compute_mix_cost(mix, iso_sens, iso, demand_target,
                                      overrides=overrides_target, growth_factor=gf_target)
            if result['effective_cost'] < best_target_cost:
                best_target_cost = result['effective_cost']
                best_target_mix = mix

        if not best_target_mix:
            results[iso] = {}
            continue

        target_deployed = _mix_resource_twh(best_target_mix, demand_target)

        # ------------------------------------------------------------------
        # Step 2: At each threshold, optimize with learning-curve costs
        # and alignment scoring toward the target
        # ------------------------------------------------------------------
        for t in THRESHOLDS:
            t_str = str(int(t)) if t == int(t) else str(t)
            mixes = feasible_mixes.get(iso, {}).get(t_str, [])
            if not mixes:
                continue

            gf = get_demand_growth_factor(iso, t)
            demand_twh = base_demand_twh * gf
            frac = learning_fraction(t)

            # Learning-curve LCOE overrides
            overrides = _build_learning_overrides(iso, frac)

            # Target clean firm deployment at this threshold:
            # Proportional to target mix × learning curve position.
            cf_scale = frac / FRAC_TARGET  # 0→1 over 50→target, >1 above target
            target_cf_twh = target_deployed['clean_firm'] * cf_scale * (demand_twh / demand_target)
            target_ccs_twh = target_deployed['ccs_ccgt'] * cf_scale * (demand_twh / demand_target)
            target_firm_twh = target_cf_twh + target_ccs_twh

            # Score each mix: cost + clean firm alignment penalty
            best_score = float('inf')
            best_result = None

            for mix in mixes:
                result = compute_mix_cost(mix, iso_sens, iso, demand_twh,
                                          overrides=overrides, growth_factor=gf)

                # Clean firm alignment: penalize deviation from graduated target
                actual_cf_twh = result['resource_twh'].get('clean_firm', 0)
                actual_ccs_twh = result['resource_twh'].get('ccs_ccgt', 0)
                actual_firm_twh = actual_cf_twh + actual_ccs_twh

                if target_firm_twh > 1.0:
                    # Relative deviation from target (0 = perfect alignment)
                    deviation = abs(actual_firm_twh - target_firm_twh) / target_firm_twh
                else:
                    deviation = 0.0

                alignment_penalty = deviation * CF_ALIGNMENT_WEIGHT
                score = result['effective_cost'] + alignment_penalty

                if score < best_score:
                    best_score = score
                    best_result = result

            if best_result:
                iso_results[t] = best_result

        results[iso] = iso_results
    return results


# ============================================================================
# CONSEQUENTIAL QUEUE BUILDER
# ============================================================================

def build_consequential_queue(scenario_results, egrid, fossil_mix):
    """Build the consequential deployment queue for a scenario."""
    zone_metrics = []

    for iso in ISOS:
        iso_data = scenario_results[iso]
        baseline_clean = sum(GRID_MIX_SHARES[iso].values())

        for zone in ZONES:
            t_start, t_end = zone['start'], zone['end']
            if t_start not in iso_data or t_end not in iso_data:
                continue

            start = iso_data[t_start]
            end = iso_data[t_end]

            # Use year-specific demand for each endpoint
            demand_twh_start = get_demand_twh(iso, t_start)
            demand_twh_end = get_demand_twh(iso, t_end)

            # Cost delta (for reporting only — NOT used in MAC)
            delta_cost_per_mwh = end['effective_cost'] - start['effective_cost']

            # CO₂ displaced using merit-order dispatch (coal → oil → gas)
            clean_twh_start = max(0, (t_start - baseline_clean) / 100.0 * demand_twh_start)
            clean_twh_end = max(0, (t_end - baseline_clean) / 100.0 * demand_twh_end)
            delta_clean_twh = clean_twh_end - clean_twh_start

            rate_start, _ = compute_fossil_retirement(iso, t_start, egrid, fossil_mix)
            rate_end, _ = compute_fossil_retirement(iso, t_end, egrid, fossil_mix)
            avg_rate = (rate_start + rate_end) / 2

            co2_displaced_mt = delta_clean_twh * avg_rate

            # MAC = LCOE / displaced_emission_rate
            # LCOE = marginal new-build cost per MWh of new clean generation
            delta_new_cost = end['new_build_cost_total'] - start['new_build_cost_total']
            delta_new_gen = end['new_gen_twh'] - start['new_gen_twh']
            if delta_new_gen > 0.01:
                marginal_lcoe = delta_new_cost / (delta_new_gen * 1e6)
            else:
                marginal_lcoe = end.get('blended_new_lcoe', 0)

            if avg_rate > 0.001 and marginal_lcoe > 0:
                marginal_mac = marginal_lcoe / avg_rate
            else:
                marginal_mac = float('inf')

            # Resource deltas
            delta_resources = {}
            for res in RESOURCES:
                delta_resources[res] = end['resource_twh'][res] - start['resource_twh'][res]
            delta_resources['battery'] = (end['battery_twh'] + end.get('battery8_twh', 0)) - \
                                         (start['battery_twh'] + start.get('battery8_twh', 0))
            delta_resources['ldes'] = end['ldes_twh'] - start['ldes_twh']

            year_start = SBTI_YEAR_MAP.get(t_start, 2025)
            year_end = SBTI_YEAR_MAP.get(t_end, 2050)

            zone_metrics.append({
                'iso': iso,
                'zone_label': zone['label'],
                'threshold_start': t_start,
                'threshold_end': t_end,
                'year_start': year_start,
                'year_end': year_end,
                'marginal_mac': round(marginal_mac, 1) if marginal_mac < 9999 else 9999,
                'co2_displaced_mt': round(co2_displaced_mt, 2),
                'delta_cost_per_mwh': round(delta_cost_per_mwh, 2),
                'delta_resources': {k: round(v, 1) for k, v in delta_resources.items()},
                'gas_backup_mw_start': start['gas_backup_mw'],
                'gas_backup_mw_end': end['gas_backup_mw'],
                'delta_gas_mw': end['gas_backup_mw'] - start['gas_backup_mw'],
                'new_gas_mw_end': end['new_gas_mw'],
                'eff_cost_start': start['effective_cost'],
                'eff_cost_end': end['effective_cost'],
            })

    # Sort by marginal MAC (cheapest first)
    zone_metrics.sort(key=lambda x: x['marginal_mac'])

    # Assign queue positions
    for i, step in enumerate(zone_metrics):
        step['queue_position'] = i + 1

    return zone_metrics


# ============================================================================
# STRANDING ANALYSIS
# ============================================================================

def compute_stranding(scenario_results):
    """Compute stranding analysis for each ISO: peak vs final resource levels."""
    stranding = {}
    for iso in ISOS:
        iso_data = scenario_results[iso]
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
            # Final = value at 99% (or highest available)
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
# DOMINO SEQUENCE — SHOW WHICH ISOs FALL FIRST AT EACH MAC LEVEL
# ============================================================================

def compute_domino_sequence(queue_a, queue_b):
    """
    Build a domino sequence showing step-by-step what happens as you follow
    the cheapest $/tCO₂ path (A) vs clean firm path (B).

    Shows cumulative: CO₂ displaced, cost, gas capacity, stranded risk.
    """
    for scenario_label, queue in [('consequential', queue_a), ('hourly_matching', queue_b)]:
        running_co2 = 0
        running_cost = 0
        iso_thresholds = {iso: 50 for iso in ISOS}
        total_gas = sum(queue[0]['gas_backup_mw_start'] for q in [queue]
                        for s in [q[0]] if s['threshold_start'] == 50 and s['iso'] == s['iso'])

        for step in queue:
            running_co2 += step['co2_displaced_mt']
            iso_thresholds[step['iso']] = step['threshold_end']


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("DUAL-SCENARIO COMPARISON: PURE CONSEQUENTIAL vs HOURLY MATCHING")
    print("  A: Pure Consequential — sequential cheapest $/tCO₂ procurement")
    print("  B: Hourly Matching — direct hourly matching with clean firm on FOAK→NOAK curve")
    print("=" * 80)

    # Load egrid and fossil mix data
    with open('data/egrid_emission_rates.json') as f:
        egrid = json.load(f)
    with open('data/EIA 930 Data/eia_fossil_mix.json') as f:
        fossil_mix = json.load(f)

    # Parse feasible mixes
    print("\nParsing feasible mixes from shared-data.js...")
    feasible_mixes = parse_feasible_mixes()
    total_mixes = sum(len(mixes) for iso_data in feasible_mixes.values()
                      for mixes in iso_data.values())
    print(f"  Loaded {total_mixes:,} feasible mixes across {len(ISOS)} ISOs × {len(THRESHOLDS)} thresholds")

    # Scenario A: Pure Consequential — path-dependent sequential optimization
    print("\nScenario A (Pure Consequential): path-dependent sequential optimization...")
    results_a = find_optimal_mixes_sequential(feasible_mixes, SCENARIO_A, BASE_DEMAND_TWH)

    # Scenario B: Hourly Matching — compute for each possible target threshold
    # This is parametric: the slider in the dashboard lets users pick the target,
    # and Scenario B recalculates its deployment path working backwards from that target.
    print("\nScenario B (Hourly Matching): pre-computing for all target thresholds...")
    print("  Learning curve: clean firm/CCS/LDES interpolated High→Low per threshold")
    print("  Mature tech (solar, wind, battery) at Low cost throughout")

    results_b_by_target = {}
    for target_t in THRESHOLDS:
        print(f"  Target {target_t}%...", end='')
        results_b_by_target[target_t] = find_optimal_mixes_learning_curve(
            feasible_mixes, SCENARIO_B, BASE_DEMAND_TWH, target_threshold=target_t)
        print(" done")

    # Default Scenario B at 95% for backward compatibility
    results_b = results_b_by_target[95]

    # Build consequential queues
    print("\nBuilding consequential queues...")
    queue_a = build_consequential_queue(results_a, egrid, fossil_mix)
    queue_b = build_consequential_queue(results_b, egrid, fossil_mix)

    # Stranding analysis
    print("\nComputing stranding analysis...")
    stranding_a = compute_stranding(results_a)
    stranding_b = compute_stranding(results_b)

    # ========================================================================
    # PRINT COMPARISON
    # ========================================================================

    for scenario, results, queue, stranding, label in [
        (SCENARIO_A, results_a, queue_a, stranding_a, 'A'),
        (SCENARIO_B, results_b, queue_b, stranding_b, 'B'),
    ]:
        print(f"\n{'=' * 120}")
        print(f"SCENARIO {label}: {scenario['name'].upper()}")
        print(f"  {scenario['description']}")
        print(f"  Toggles: ren={scenario['toggles']['ren']}, firm={scenario['toggles']['firm']}, "
              f"batt={scenario['toggles']['batt']}, ldes={scenario['toggles']['ldes_lvl']}, "
              f"ccs={scenario['toggles']['ccs']}, fuel={scenario['toggles']['fuel']}, "
              f"tx={scenario['toggles']['tx']}")
        print(f"{'=' * 120}")

        # Queue
        print(f"\n{'#':>3} {'ISO':<7} {'Zone':<12} {'MAC $/t':>10} {'CO₂ MT':>8} {'ΔCost':>8} "
              f"{'Gas End':>10} {'ΔGas':>10} {'Primary Resource':>40}")
        print("-" * 120)

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
                  f"{mac_str:>10} {step['co2_displaced_mt']:>7.1f} "
                  f"${step['delta_cost_per_mwh']:>6.1f} "
                  f"{step['gas_backup_mw_end']:>9,} {step['delta_gas_mw']:>+9,} "
                  f"{top_str:>40}")

        total_co2 = sum(s['co2_displaced_mt'] for s in queue)
        total_cost_delta = sum(s['delta_cost_per_mwh'] * BASE_DEMAND_TWH[s['iso']] * 1e6
                               for s in queue) / 1e9
        print(f"\nTotal CO₂ displaced: {total_co2:.1f} MT")
        print(f"Total cost delta: ${total_cost_delta:.1f}B")

        # Stranding summary
        print(f"\nStranding flags (peak/final ratio > 1.5):")
        for iso in ISOS:
            for res in ['wind', 'solar', 'clean_firm']:
                s = stranding[iso].get(res, {})
                if s.get('stranding_ratio', 0) > 1.5 and s.get('peak_twh', 0) > 1:
                    print(f"  ⚠ {iso} {res}: peak {s['peak_twh']:.0f} TWh @ {s['peak_threshold']}% → "
                          f"final {s['final_twh']:.0f} TWh @ {s['final_threshold']}% "
                          f"(ratio: {s['stranding_ratio']:.1f}x)")

    # ========================================================================
    # SIDE-BY-SIDE COMPARISON
    # ========================================================================

    print(f"\n{'=' * 150}")
    print("SIDE-BY-SIDE: RESOURCE MIX + GAS + COST AT EACH THRESHOLD (demand grows per SBTI year)")
    print(f"{'=' * 150}")

    for iso in ISOS:
        print(f"\n{'─' * 130}")
        print(f"  {iso} (base demand: {BASE_DEMAND_TWH[iso]:.1f} TWh, growth: {DEMAND_GROWTH_RATES[iso]*100:.1f}%/yr)")
        print(f"{'─' * 130}")
        print(f"{'Thr':>5} {'Year':>5} {'Dem':>6} │ {'CF_A':>6} {'Sol_A':>6} {'Wnd_A':>6} {'Gas_A':>8} {'$/MWh_A':>8} │ "
              f"{'CF_B':>6} {'Sol_B':>6} {'Wnd_B':>6} {'Gas_B':>8} {'$/MWh_B':>8} │ "
              f"{'ΔCost':>7} {'ΔGas':>8}")
        print("-" * 130)

        for t in THRESHOLDS:
            year = SBTI_YEAR_MAP.get(t, '?')
            demand_twh = get_demand_twh(iso, t)
            a = results_a.get(iso, {}).get(t, {})
            b = results_b.get(iso, {}).get(t, {})
            if not a or not b:
                continue

            a_rt = a['resource_twh']
            b_rt = b['resource_twh']

            delta_cost = b['effective_cost'] - a['effective_cost']
            delta_gas = b['gas_backup_mw'] - a['gas_backup_mw']

            print(f"{t:>5} {year:>5} {demand_twh:>5.0f} │ "
                  f"{a_rt['clean_firm']:>6.0f} {a_rt['solar']:>6.0f} {a_rt['wind']:>6.0f} "
                  f"{a['gas_backup_mw']:>8,} {a['effective_cost']:>7.1f} │ "
                  f"{b_rt['clean_firm']:>6.0f} {b_rt['solar']:>6.0f} {b_rt['wind']:>6.0f} "
                  f"{b['gas_backup_mw']:>8,} {b['effective_cost']:>7.1f} │ "
                  f"{delta_cost:>+7.1f} {delta_gas:>+8,}")

    # ========================================================================
    # DOMINO SEQUENCE — MAC ESCALATION
    # ========================================================================

    print(f"\n{'=' * 140}")
    print("DOMINO SEQUENCE: CUMULATIVE CO₂ + GAS + COST AS YOU CLIMB THE MAC LADDER")
    print(f"{'=' * 140}")

    for scenario, queue, label in [(SCENARIO_A, queue_a, 'A'), (SCENARIO_B, queue_b, 'B')]:
        print(f"\nScenario {label}: {scenario['name']}")
        print(f"{'#':>3} {'ISO':<7} {'Zone':<12} {'MAC':>10} {'Cum CO₂':>9} {'Cum $B':>8} "
              f"{'Gas(end)':>10} {'New Gas':>10} {'ISO Progress':>50}")
        print("-" * 130)

        cum_co2 = 0
        cum_cost = 0
        iso_progress = {iso: 50 for iso in ISOS}

        for step in queue:
            cum_co2 += step['co2_displaced_mt']
            cum_cost += step['delta_cost_per_mwh'] * BASE_DEMAND_TWH[step['iso']] * 1e6 / 1e9
            iso_progress[step['iso']] = step['threshold_end']

            prog_str = " | ".join(f"{iso}:{iso_progress[iso]:.0f}%" for iso in ISOS)
            mac_str = f"${step['marginal_mac']:,.0f}" if step['marginal_mac'] < 9999 else "$∞"

            print(f"{step['queue_position']:>3} {step['iso']:<7} {step['zone_label']:<12} "
                  f"{mac_str:>10} {cum_co2:>8.1f} {cum_cost:>+7.1f} "
                  f"{step['gas_backup_mw_end']:>10,} {step['new_gas_mw_end']:>10,} "
                  f"{prog_str:>50}")

    # ========================================================================
    # WRITE OUTPUT FILES
    # ========================================================================

    # Build resource trajectories with per-threshold stepwise MAC
    def _build_trajectory(results, enforce_monotonic=False):
        """Build trajectory dicts from optimization results."""
        traj = {}
        for iso in ISOS:
            iso_traj = []
            base_demand_twh = BASE_DEMAND_TWH[iso]
            prev_t = None
            for t in THRESHOLDS:
                d = results.get(iso, {}).get(t, {})
                if not d:
                    continue
                demand_twh = get_demand_twh(iso, t)
                stepwise_mac = None
                if prev_t is not None and prev_t in results.get(iso, {}):
                    prev_d = results[iso][prev_t]
                    delta_new_cost = d['new_build_cost_total'] - prev_d['new_build_cost_total']
                    delta_new_gen = d['new_gen_twh'] - prev_d['new_gen_twh']
                    if delta_new_gen > 0.01:
                        marginal_lcoe = delta_new_cost / (delta_new_gen * 1e6)
                    else:
                        marginal_lcoe = d.get('blended_new_lcoe', 0)
                    rate_cur, _ = compute_fossil_retirement(iso, t, egrid, fossil_mix)
                    if rate_cur > 0.001 and marginal_lcoe > 0:
                        stepwise_mac = round(marginal_lcoe / rate_cur, 1)
                    else:
                        stepwise_mac = 9999
                cf_twh = d['resource_twh'].get('clean_firm', 0)
                ccs_twh = d['resource_twh'].get('ccs_ccgt', 0)
                existing_cf_twh = GRID_MIX_SHARES[iso].get('clean_firm', 0) / 100.0 * base_demand_twh
                firm_total_twh = max(0, cf_twh - existing_cf_twh) + ccs_twh
                iso_traj.append({
                    'threshold': t,
                    'year': SBTI_YEAR_MAP.get(t, 2050),
                    'demand_twh': round(demand_twh, 1),
                    'effective_cost': d['effective_cost'],
                    'total_cost': d['total_cost'],
                    'incremental': d['incremental'],
                    'resource_twh': d['resource_twh'],
                    'battery_twh': d.get('battery_twh', 0) + d.get('battery8_twh', 0),
                    'ldes_twh': d.get('ldes_twh', 0),
                    'gas_backup_mw': d['gas_backup_mw'],
                    'new_gas_mw': d['new_gas_mw'],
                    'existing_gas_used_mw': d.get('existing_gas_used_mw', 0),
                    'clean_peak_mw': d.get('clean_peak_mw', 0),
                    'firm_total_twh': round(firm_total_twh, 1),
                    'procurement_pct': d.get('procurement_pct', 100),
                    'stepwise_mac': stepwise_mac,
                    'blended_new_lcoe': d.get('blended_new_lcoe', 0),
                    'new_gen_twh': d.get('new_gen_twh', 0),
                    'new_build_cost_total': d.get('new_build_cost_total', 0),
                })
                prev_t = t
            traj[iso] = iso_traj

        if enforce_monotonic:
            for iso in ISOS:
                if iso not in traj:
                    continue
                running_max = {}
                for entry in traj[iso]:
                    for res in list(entry['resource_twh'].keys()):
                        val = entry['resource_twh'][res]
                        if res in running_max:
                            entry['resource_twh'][res] = max(running_max[res], val)
                        running_max[res] = entry['resource_twh'][res]
                    for field in ['battery_twh', 'ldes_twh']:
                        val = entry.get(field, 0)
                        old_max = running_max.get(field, 0)
                        entry[field] = max(old_max, val)
                        running_max[field] = entry[field]
                    clean_peak_mw = 0
                    for r, twh in entry['resource_twh'].items():
                        pcc = PEAK_CAPACITY_CREDITS.get(r, 0)
                        if pcc > 0 and twh > 0:
                            clean_peak_mw += (twh * 1e6 / 8760) * pcc
                    clean_peak_mw += (entry.get('battery_twh', 0) * 1e6 / 8760) * PEAK_CAPACITY_CREDITS.get('battery', 0.95)
                    clean_peak_mw += (entry.get('ldes_twh', 0) * 1e6 / 8760) * PEAK_CAPACITY_CREDITS.get('ldes', 0.90)
                    gf = get_demand_growth_factor(iso, entry['threshold'])
                    ra_peak_mw = PEAK_DEMAND_MW[iso] * gf * (1 + RESOURCE_ADEQUACY_MARGIN)
                    gaf = GAS_AVAILABILITY_FACTOR[iso]
                    gas_needed_mw = max(0, ra_peak_mw - clean_peak_mw) / gaf
                    entry['gas_backup_mw'] = round(gas_needed_mw)
                    entry['new_gas_mw'] = round(max(0, gas_needed_mw - EXISTING_GAS_CAPACITY_MW[iso]))
                    entry['clean_peak_mw'] = round(clean_peak_mw)
                    cf_twh = entry['resource_twh'].get('clean_firm', 0)
                    ccs_twh = entry['resource_twh'].get('ccs_ccgt', 0)
                    existing_cf_twh = GRID_MIX_SHARES[iso].get('clean_firm', 0) / 100.0 * BASE_DEMAND_TWH[iso]
                    entry['firm_total_twh'] = round(max(0, cf_twh - existing_cf_twh) + ccs_twh, 1)
        return traj

    trajectories = {}
    trajectories['pure_consequential'] = _build_trajectory(results_a, enforce_monotonic=True)
    trajectories['hourly_matching'] = _build_trajectory(results_b, enforce_monotonic=False)

    # Build per-target Scenario B trajectories for the dashboard slider
    trajectories_b_by_target = {}
    for target_t in THRESHOLDS:
        t_key = str(int(target_t)) if target_t == int(target_t) else str(target_t)
        trajectories_b_by_target[t_key] = _build_trajectory(
            results_b_by_target[target_t], enforce_monotonic=False)

    output = {
        'metadata': {
            'description': 'Dual-scenario comparison: Pure Consequential vs Hourly Matching',
            'scenario_a': {
                'name': SCENARIO_A['name'],
                'description': SCENARIO_A['description'],
                'toggles': SCENARIO_A['toggles'],
                'method': 'path_dependent_sequential',
                'method_description': 'Pure consequential: sequential optimization where resources deployed at each threshold become the floor for the next. Cheapest $/tCO2 at each step, locked into prior commitments.',
            },
            'scenario_b': {
                'name': SCENARIO_B['name'],
                'description': SCENARIO_B['description'],
                'toggles': SCENARIO_B['toggles'],
                'method': 'learning_curve_graduated',
                'method_description': 'Hourly matching with graduated FOAK→NOAK deployment. At each threshold, clean firm/CCS/LDES costs are interpolated between High (FOAK) and Low (NOAK) based on SBTi timeline position. Early thresholds see less clean firm (expensive); deployment increases as learning drives costs down. Mature tech (solar, wind, battery) at Low cost throughout.',
                'learning_curve': 'convex ramp: 0.05 + 0.95 * ((t-50)/50)^1.8',
            },
            'sbti_year_map': {str(k): v for k, v in SBTI_YEAR_MAP.items()},
            'thresholds': THRESHOLDS,
            'isos': ISOS,
            'grid_mix_shares': {iso: dict(GRID_MIX_SHARES[iso]) for iso in ISOS},
            'base_demand_twh': dict(BASE_DEMAND_TWH),
            'demand_growth_rates': DEMAND_GROWTH_RATES,
            'demand_growth_level': 'Medium',
        },
        'queue_a': queue_a,
        'queue_b': queue_b,
        'trajectories': trajectories,
        'trajectories_b_by_target': trajectories_b_by_target,
        'stranding_a': stranding_a,
        'stranding_b': stranding_b,
    }

    out_json = 'data/step5-post-processing/scenario_comparison.json'
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nJSON: {out_json} ({os.path.getsize(out_json) / 1024:.0f} KB)")

    out_js = 'dashboard/js/scenario-comparison-data.js'
    os.makedirs(os.path.dirname(out_js), exist_ok=True)
    with open(out_js, 'w') as f:
        f.write("// Auto-generated by step5_PP3_scenario_comparison.py\n")
        f.write("// Dual-scenario comparison: Pure Consequential vs Hourly Matching\n")
        f.write("// A: Pure Consequential  B: Hourly Matching\n\n")
        f.write(f"const SCENARIO_COMPARISON = {json.dumps(output, indent=2, default=str)};\n")
    print(f"JS:   {out_js} ({os.path.getsize(out_js) / 1024:.0f} KB)")

    print("\nDone.")


if __name__ == '__main__':
    main()
