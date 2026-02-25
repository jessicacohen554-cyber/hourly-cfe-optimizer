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

GROWTH_RATES = {'CAISO': 1.9, 'ERCOT': 3.5, 'PJM': 2.4, 'NYISO': 2.0, 'NEISO': 1.8}

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


def compute_mix_cost(mix, sens, iso, demand_twh, overrides=None):
    """
    Compute total system cost per MWh for a single mix under a sensitivity scenario.

    mix: [cf%, sol%, wnd%, ccs%, hyd%, proc%, match%, bat4%, bat8%, ldes%]
    overrides: optional dict with explicit LCOE values (bypasses toggle lookups):
        nuclear_lcoe, ccs_lcoe, geo_lcoe, ldes_lcoe, uprate_lcoe
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

    existing = GRID_MIX_SHARES[iso]
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
    peak_mw = PEAK_DEMAND_MW[iso]
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


def _floor_violation_score(mix, floor, demand_twh):
    """Compute how much a mix violates the floor (0 = perfect compliance).

    Returns the sum of shortfalls (in TWh) across all resources.
    Only counts resources where floor > 1 TWh (ignore negligible floors).
    """
    deployed = _mix_resource_twh(mix, demand_twh)
    violation = 0.0
    for res, floor_val in floor.items():
        if floor_val < 1.0:
            continue
        shortfall = max(0, floor_val - deployed.get(res, 0))
        violation += shortfall
    return violation


def find_optimal_mixes_sequential(feasible_mixes, scenario, demand_twh_map):
    """Path-dependent sequential optimization for the consequential scenario.

    At each threshold step, resources deployed in prior steps form a floor —
    the optimizer can only ADD on top, never shrink.  This models the
    consequential procurement strategy where buyers chase cheapest $/tCO₂
    at each increment, locking in prior resource commitments.

    Resources remain at LCOE pricing (not wholesale) — committed capacity
    retains its original cost, it doesn't become "existing" at wholesale.

    When the feasible mix space is too sparse for strict monotonicity,
    the optimizer uses a soft-floor approach: score each mix by its total
    floor violation (TWh shortfall), then select the cheapest mix among
    those with the minimum violation. This preserves path dependency
    while handling discrete feasible-set gaps.
    """
    results = {}
    sens = scenario['toggles']

    for iso in ISOS:
        iso_results = {}
        iso_sens = dict(sens)
        if iso != 'CAISO':
            iso_sens['geo'] = None

        demand_twh = demand_twh_map[iso]
        # Floor starts at existing resource levels — cannot shrink below current grid.
        # These are absolute TWh floors locked in from existing infrastructure.
        existing = GRID_MIX_SHARES[iso]
        floor = {
            'clean_firm': existing.get('clean_firm', 0) / 100.0 * demand_twh,
            'solar': existing.get('solar', 0) / 100.0 * demand_twh,
            'wind': existing.get('wind', 0) / 100.0 * demand_twh,
            'ccs_ccgt': existing.get('ccs_ccgt', 0) / 100.0 * demand_twh,
            'hydro': existing.get('hydro', 0) / 100.0 * demand_twh,
            'battery': 0, 'ldes': 0,
        }

        for t in THRESHOLDS:
            t_str = str(int(t)) if t == int(t) else str(t)
            mixes = feasible_mixes.get(iso, {}).get(t_str, [])
            if not mixes:
                continue

            # Score all mixes by (floor_violation, cost) — lexicographic sort
            # This picks the mix with minimum floor violation; among ties, cheapest
            candidates = []
            for mix in mixes:
                violation = _floor_violation_score(mix, floor, demand_twh)
                result = compute_mix_cost(mix, iso_sens, iso, demand_twh)
                candidates.append((violation, result['effective_cost'], result, mix))

            # Sort: minimum violation first, then cheapest
            candidates.sort(key=lambda x: (x[0], x[1]))

            if candidates:
                violation, cost, best_result, best_mix = candidates[0]
                if violation > 0.1:
                    print(f"  ⚠ {iso} {t}%: soft floor violation {violation:.0f} TWh "
                          f"(best of {len(candidates)} mixes)")
                iso_results[t] = best_result
                # Update floor: take element-wise MAX of old floor and new deployment
                # This ensures the ratchet only goes up, even if soft constraint was used
                new_deployed = _mix_resource_twh(best_mix, demand_twh)
                floor = {res: max(floor[res], new_deployed.get(res, 0))
                         for res in floor}

        results[iso] = iso_results
    return results


def find_optimal_mixes_learning_curve(feasible_mixes, scenario, demand_twh_map):
    """Hourly Matching scenario: graduated clean firm deployment along FOAK→NOAK curve.

    At each threshold, clean firm / CCS / LDES / geothermal costs are
    interpolated between High (FOAK) and Low (NOAK) based on position
    on the SBTi timeline via learning_fraction().  Mature technologies
    (solar, wind, battery) stay at Low cost throughout.

    This produces:
      - Less clean firm at early thresholds (expensive FOAK pricing)
      - Gradually increasing clean firm as learning drives costs down
      - Near-full NOAK deployment at high thresholds (95-100%)

    Each threshold is optimized independently (no path-dependency floor).
    The increasing deployment emerges naturally from declining costs +
    increasing physics requirements for firm generation at higher CFE.
    """
    results = {}
    sens = scenario['toggles']

    for iso in ISOS:
        iso_results = {}
        iso_sens = dict(sens)
        if iso != 'CAISO':
            iso_sens['geo'] = None

        demand_twh = demand_twh_map[iso]

        # Existing resource floor — cannot shrink below current grid (absolute TWh)
        existing = GRID_MIX_SHARES[iso]
        existing_floor = {
            'clean_firm': existing.get('clean_firm', 0) / 100.0 * demand_twh,
            'solar': existing.get('solar', 0) / 100.0 * demand_twh,
            'wind': existing.get('wind', 0) / 100.0 * demand_twh,
            'ccs_ccgt': existing.get('ccs_ccgt', 0) / 100.0 * demand_twh,
            'hydro': existing.get('hydro', 0) / 100.0 * demand_twh,
            'battery': 0, 'ldes': 0,
        }

        for t in THRESHOLDS:
            t_str = str(int(t)) if t == int(t) else str(t)
            mixes = feasible_mixes.get(iso, {}).get(t_str, [])
            if not mixes:
                continue

            # Learning curve fraction: how far along FOAK→NOAK
            frac = learning_fraction(t)

            # Build LCOE overrides: interpolate between High and Low
            overrides = {}

            # Nuclear new-build
            nuc_h = NUCLEAR_NEWBUILD_LCOE['H'][iso]
            nuc_l = NUCLEAR_NEWBUILD_LCOE['L'][iso]
            overrides['nuclear_lcoe'] = nuc_h + frac * (nuc_l - nuc_h)

            # Nuclear uprate
            overrides['uprate_lcoe'] = UPRATE_LCOE['H'] + frac * (UPRATE_LCOE['L'] - UPRATE_LCOE['H'])

            # CCS-CCGT (with 45Q)
            ccs_h = CCS_LCOE_45Q_ON['H'][iso]
            ccs_l = CCS_LCOE_45Q_ON['L'][iso]
            overrides['ccs_lcoe'] = ccs_h + frac * (ccs_l - ccs_h)

            # Geothermal (CAISO only)
            if iso == 'CAISO':
                geo_h = GEOTHERMAL_LCOE['H']
                geo_l = GEOTHERMAL_LCOE['L']
                overrides['geo_lcoe'] = geo_h + frac * (geo_l - geo_h)

            # LDES (100hr iron-air — emerging tech, also on learning curve)
            ldes_h = LCOE_TABLES['ldes']['High'][iso]
            ldes_l = LCOE_TABLES['ldes']['Low'][iso]
            overrides['ldes_lcoe'] = ldes_h + frac * (ldes_l - ldes_h)

            # Filter mixes that respect existing resource floor
            floor_passing = []
            for mix in mixes:
                violation = _floor_violation_score(mix, existing_floor, demand_twh)
                if violation < 1.0:
                    floor_passing.append(mix)
            if not floor_passing:
                floor_passing = mixes  # Fall back if none pass

            # Find cheapest mix at this threshold with interpolated costs
            best_cost = float('inf')
            best_result = None

            for mix in floor_passing:
                result = compute_mix_cost(mix, iso_sens, iso, demand_twh, overrides=overrides)
                if result['effective_cost'] < best_cost:
                    best_cost = result['effective_cost']
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
        demand_twh = BASE_DEMAND_TWH[iso]
        demand_mwh = demand_twh * 1e6
        baseline_clean = sum(GRID_MIX_SHARES[iso].values())

        for zone in ZONES:
            t_start, t_end = zone['start'], zone['end']
            if t_start not in iso_data or t_end not in iso_data:
                continue

            start = iso_data[t_start]
            end = iso_data[t_end]

            # Cost delta (for reporting only — NOT used in MAC)
            delta_cost_per_mwh = end['effective_cost'] - start['effective_cost']

            # CO₂ displaced using merit-order dispatch (coal → oil → gas)
            clean_twh_start = max(0, (t_start - baseline_clean) / 100.0 * demand_twh)
            clean_twh_end = max(0, (t_end - baseline_clean) / 100.0 * demand_twh)
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

    # Scenario B: Hourly Matching — graduated FOAK→NOAK learning curve deployment
    print("\nScenario B (Hourly Matching): graduated FOAK→NOAK learning curve optimization...")
    print("  Learning curve: clean firm/CCS/LDES interpolated High→Low per threshold")
    print("  Mature tech (solar, wind, battery) at Low cost throughout")
    results_b = find_optimal_mixes_learning_curve(feasible_mixes, SCENARIO_B, BASE_DEMAND_TWH)
    print("  Done.")

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

    print(f"\n{'=' * 140}")
    print("SIDE-BY-SIDE: RESOURCE MIX + GAS + COST AT EACH THRESHOLD")
    print(f"{'=' * 140}")

    for iso in ISOS:
        print(f"\n{'─' * 120}")
        print(f"  {iso}")
        print(f"{'─' * 120}")
        print(f"{'Thr':>5} {'Year':>5} │ {'CF_A':>6} {'Sol_A':>6} {'Wnd_A':>6} {'Gas_A':>8} {'$/MWh_A':>8} │ "
              f"{'CF_B':>6} {'Sol_B':>6} {'Wnd_B':>6} {'Gas_B':>8} {'$/MWh_B':>8} │ "
              f"{'ΔCost':>7} {'ΔGas':>8}")
        print(f"{'':>5} {'':>5} │ {'(TWh)':>6} {'(TWh)':>6} {'(TWh)':>6} {'(MW)':>8} {'':>8} │ "
              f"{'(TWh)':>6} {'(TWh)':>6} {'(TWh)':>6} {'(MW)':>8} {'':>8} │ "
              f"{'($/MWh)':>7} {'(MW)':>8}")
        print("-" * 120)

        for t in THRESHOLDS:
            year = SBTI_YEAR_MAP.get(t, '?')
            a = results_a.get(iso, {}).get(t, {})
            b = results_b.get(iso, {}).get(t, {})
            if not a or not b:
                continue

            a_rt = a['resource_twh']
            b_rt = b['resource_twh']

            delta_cost = b['effective_cost'] - a['effective_cost']
            delta_gas = b['gas_backup_mw'] - a['gas_backup_mw']

            print(f"{t:>5} {year:>5} │ "
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
    trajectories = {}
    for scenario, results, label in [(SCENARIO_A, results_a, 'pure_consequential'),
                                      (SCENARIO_B, results_b, 'hourly_matching')]:
        traj = {}
        for iso in ISOS:
            iso_traj = []
            demand_twh = BASE_DEMAND_TWH[iso]
            demand_mwh = demand_twh * 1e6
            baseline_clean = sum(GRID_MIX_SHARES[iso].values())
            prev_t = None
            for t in THRESHOLDS:
                d = results.get(iso, {}).get(t, {})
                if not d:
                    continue

                # Stepwise MAC = marginal LCOE / displaced emission rate
                stepwise_mac = None
                if prev_t is not None and prev_t in results.get(iso, {}):
                    prev_d = results[iso][prev_t]
                    # Marginal LCOE of incremental new-build resources
                    delta_new_cost = d['new_build_cost_total'] - prev_d['new_build_cost_total']
                    delta_new_gen = d['new_gen_twh'] - prev_d['new_gen_twh']
                    if delta_new_gen > 0.01:
                        marginal_lcoe = delta_new_cost / (delta_new_gen * 1e6)
                    else:
                        marginal_lcoe = d.get('blended_new_lcoe', 0)
                    # Displaced emission rate from merit-order dispatch
                    rate_cur, _ = compute_fossil_retirement(iso, t, egrid, fossil_mix)
                    if rate_cur > 0.001 and marginal_lcoe > 0:
                        stepwise_mac = round(marginal_lcoe / rate_cur, 1)
                    else:
                        stepwise_mac = 9999

                # Clean firm TWh — NEW BUILD ONLY (subtract existing clean firm)
                cf_twh = d['resource_twh'].get('clean_firm', 0)
                ccs_twh = d['resource_twh'].get('ccs_ccgt', 0)
                existing_cf_twh = GRID_MIX_SHARES[iso].get('clean_firm', 0) / 100.0 * BASE_DEMAND_TWH[iso]
                firm_total_twh = max(0, cf_twh - existing_cf_twh) + ccs_twh

                iso_traj.append({
                    'threshold': t,
                    'year': SBTI_YEAR_MAP.get(t, 2050),
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

        # Post-process: enforce strict monotonicity for Scenario A
        # Resources can only go UP (path-dependent lock-in). If the feasible set forced
        # a soft-floor violation, the prior locked-in resources still physically exist.
        # Recompute gas backup from the floor-enforced resource levels.
        if label == 'pure_consequential':
            for iso in ISOS:
                if iso not in traj:
                    continue
                running_max = {}
                for entry in traj[iso]:
                    # Enforce monotonic resource_twh
                    for res in list(entry['resource_twh'].keys()):
                        val = entry['resource_twh'][res]
                        if res in running_max:
                            entry['resource_twh'][res] = max(running_max[res], val)
                        running_max[res] = entry['resource_twh'][res]

                    # Enforce monotonic battery/LDES
                    for field in ['battery_twh', 'ldes_twh']:
                        val = entry.get(field, 0)
                        old_max = running_max.get(field, 0)
                        entry[field] = max(old_max, val)
                        running_max[field] = entry[field]

                    # Recompute gas backup from floor-enforced resources
                    clean_peak_mw = 0
                    for r, twh in entry['resource_twh'].items():
                        pcc = PEAK_CAPACITY_CREDITS.get(r, 0)
                        if pcc > 0 and twh > 0:
                            clean_peak_mw += (twh * 1e6 / 8760) * pcc
                    clean_peak_mw += (entry.get('battery_twh', 0) * 1e6 / 8760) * PEAK_CAPACITY_CREDITS.get('battery', 0.95)
                    clean_peak_mw += (entry.get('ldes_twh', 0) * 1e6 / 8760) * PEAK_CAPACITY_CREDITS.get('ldes', 0.90)

                    ra_peak_mw = PEAK_DEMAND_MW[iso] * (1 + RESOURCE_ADEQUACY_MARGIN)
                    gaf = GAS_AVAILABILITY_FACTOR[iso]
                    gas_needed_mw = max(0, ra_peak_mw - clean_peak_mw) / gaf
                    entry['gas_backup_mw'] = round(gas_needed_mw)
                    entry['new_gas_mw'] = round(max(0, gas_needed_mw - EXISTING_GAS_CAPACITY_MW[iso]))
                    entry['clean_peak_mw'] = round(clean_peak_mw)

                    # Recompute firm_total_twh from floor-enforced values
                    cf_twh = entry['resource_twh'].get('clean_firm', 0)
                    ccs_twh = entry['resource_twh'].get('ccs_ccgt', 0)
                    existing_cf_twh = GRID_MIX_SHARES[iso].get('clean_firm', 0) / 100.0 * BASE_DEMAND_TWH[iso]
                    entry['firm_total_twh'] = round(max(0, cf_twh - existing_cf_twh) + ccs_twh, 1)

        trajectories[label] = traj

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
        },
        'queue_a': queue_a,
        'queue_b': queue_b,
        'trajectories': trajectories,
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
