#!/usr/bin/env python3
"""
Dual-scenario consequential accounting comparison.

Compares two cost scenarios to test whether chasing cheapest $/tCO₂
(Scenario A: cheap renewables / expensive clean firm) leads to worse
outcomes than investing in clean firm learning curves (Scenario B:
everything cheap from FOAK investment).

Uses feasible mixes from shared-data.js pipeline output + cost tables
from step3, then traces through dispatch_utils for emission rates.

Output: JSON + JS data files for the comparison dashboard page.
"""

import json
import os
import re
import sys
import numpy as np
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
THRESHOLDS = [50, 60, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 100]
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
SBTI_YEAR_MAP = {
    50: 2025, 60: 2027, 70: 2029, 75: 2030, 80: 2032,
    85: 2035, 87.5: 2037, 90: 2040, 92.5: 2042,
    95: 2045, 97.5: 2047, 99: 2049, 100: 2050,
}

GROWTH_RATES = {'CAISO': 1.9, 'ERCOT': 3.5, 'PJM': 2.4, 'NYISO': 2.0, 'NEISO': 1.8}

# Zone definitions for consequential queue
ZONES = [
    {'label': '50→75%', 'start': 50, 'end': 75},
    {'label': '75→90%', 'start': 75, 'end': 90},
    {'label': '90→95%', 'start': 90, 'end': 95},
    {'label': '95→97.5%', 'start': 95, 'end': 97.5},
    {'label': '97.5→99%', 'start': 97.5, 'end': 99},
    {'label': '99→100%', 'start': 99, 'end': 100},
]


# ============================================================================
# SCENARIO DEFINITIONS
# ============================================================================

# Scenario A: "Cheap Renewables / Expensive Clean Firm"
# Pure consequential — just chase cheapest $/tCO₂ with no clean firm investment signal
SCENARIO_A = {
    'name': 'Cheap Renewables Only',
    'short': 'cheap_ren',
    'description': 'Low renewable costs, high clean firm costs — no investment signal for FOAK',
    'toggles': {
        'ren': 'L',        # Low renewable (cheap solar/wind)
        'firm': 'H',       # High firm gen (expensive nuclear)
        'batt': 'M',       # Medium battery
        'ldes_lvl': 'H',   # High LDES (expensive)
        'fuel': 'M',       # Medium fossil fuel
        'tx': 'M',         # Medium transmission
        'ccs': 'H',        # High CCS (expensive)
        'q45': '1',        # 45Q on
        'geo': 'H',        # High geothermal (CAISO only)
    },
}

# Scenario B: "Clean Firm Learning Curve / Everything Cheap"
# FOAK investment pays off — all clean technologies are cheap
SCENARIO_B = {
    'name': 'Clean Firm Investment',
    'short': 'clean_firm_invest',
    'description': 'Low costs across all clean technologies — FOAK learning curve success',
    'toggles': {
        'ren': 'L',        # Low renewable (still cheap)
        'firm': 'L',       # Low firm gen (cheap nuclear from learning curve)
        'batt': 'L',       # Low battery
        'ldes_lvl': 'L',   # Low LDES
        'fuel': 'M',       # Medium fossil fuel
        'tx': 'M',         # Medium transmission
        'ccs': 'L',        # Low CCS
        'q45': '1',        # 45Q on
        'geo': 'L',        # Low geothermal (CAISO only)
    },
}

SCENARIOS = [SCENARIO_A, SCENARIO_B]


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


def compute_mix_cost(mix, sens, iso, demand_twh):
    """
    Compute total system cost per MWh for a single mix under a sensitivity scenario.

    mix: [cf%, sol%, wnd%, ccs%, hyd%, proc%, match%, bat4%, bat8%, ldes%]
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
    ccs_table = CCS_LCOE_45Q_ON if q45 == '1' else None
    ccs_lcoe = ccs_table[ccs_lev][iso]
    ccs_tx = get_tx('ccs_ccgt', tx_name, iso)
    ccs_price = ccs_lcoe + ccs_tx

    # Nuclear price
    nuclear_price = NUCLEAR_NEWBUILD_LCOE[firm_lev][iso] + get_tx('clean_firm', tx_name, iso)
    remaining_price = min(nuclear_price, ccs_price)

    # Geothermal price
    geo_price = 0.0
    if iso == 'CAISO' and geo_lev:
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

    # Total cost = Σ(coefficient × price) + gas_cost
    # Coefficients: fraction of demand priced at each source
    total_cost = (
        (sol_existing + wnd_existing + hyd_demand + ccs_existing + cf_existing) / 100.0 * wholesale +
        sol_new / 100.0 * (LCOE_TABLES['solar'][ren_name][iso] + get_tx('solar', tx_name, iso)) +
        wnd_new / 100.0 * (LCOE_TABLES['wind'][ren_name][iso] + get_tx('wind', tx_name, iso)) +
        ccs_new / 100.0 * ccs_price +
        uprate_twh / demand_twh * UPRATE_LCOE[firm_lev] +
        geo_twh / demand_twh * geo_price +
        remaining_after_geo / demand_twh * remaining_price +
        bat4_pct / 100.0 * (LCOE_TABLES['battery'][batt_name][iso] + get_tx('battery', tx_name, iso)) +
        bat8_pct / 100.0 * (LCOE_TABLES['battery8'][batt_name][iso] + get_tx('battery8', tx_name, iso)) +
        ldes_pct / 100.0 * (LCOE_TABLES['ldes'][ldes_name][iso] + get_tx('ldes', tx_name, iso)) +
        gas_cost
    )

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
        # Floor starts at zero — existing grid resources are already accounted
        # for in the cost function (wholesale pricing for existing share).
        floor = {
            'clean_firm': 0, 'solar': 0, 'wind': 0, 'ccs_ccgt': 0,
            'hydro': 0, 'battery': 0, 'ldes': 0,
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


def find_optimal_mix_at_target(feasible_mixes, scenario, demand_twh_map, target_threshold=95):
    """FOAK scenario: optimize at a single target threshold.

    Everyone targets the same high CFE threshold simultaneously.
    The optimal mix at that threshold (with Low clean firm costs) IS
    the portfolio.  Returns results dict with the target mix replicated
    at every threshold for charting, but the 'target' key holds the
    canonical result.
    """
    results = {}
    sens = scenario['toggles']

    for iso in ISOS:
        iso_results = {}
        iso_sens = dict(sens)
        if iso != 'CAISO':
            iso_sens['geo'] = None

        demand_twh = demand_twh_map[iso]
        t_str = str(int(target_threshold)) if target_threshold == int(target_threshold) else str(target_threshold)
        mixes = feasible_mixes.get(iso, {}).get(t_str, [])

        # Find cheapest mix at the target threshold
        best_cost = float('inf')
        best_result = None

        for mix in mixes:
            result = compute_mix_cost(mix, iso_sens, iso, demand_twh)
            if result['effective_cost'] < best_cost:
                best_cost = result['effective_cost']
                best_result = result

        if best_result:
            # The target mix IS the portfolio at every threshold for display
            for t in THRESHOLDS:
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

            # Cost delta
            delta_cost_per_mwh = end['effective_cost'] - start['effective_cost']

            # CO₂ displaced (using dispatch_utils)
            clean_twh_start = max(0, (t_start - baseline_clean) / 100.0 * demand_twh)
            clean_twh_end = max(0, (t_end - baseline_clean) / 100.0 * demand_twh)
            delta_clean_twh = clean_twh_end - clean_twh_start

            # Get marginal emission rate via dispatch_utils
            rate_start, _ = compute_fossil_retirement(iso, t_start, egrid, fossil_mix)
            rate_end, _ = compute_fossil_retirement(iso, t_end, egrid, fossil_mix)
            avg_rate = (rate_start + rate_end) / 2

            co2_displaced_mt = delta_clean_twh * avg_rate

            # Marginal MAC
            if co2_displaced_mt > 0.01:
                marginal_mac = (delta_cost_per_mwh * demand_mwh) / (co2_displaced_mt * 1e6)
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
    for scenario_label, queue in [('cheap_ren', queue_a), ('clean_firm_invest', queue_b)]:
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
    print("DUAL-SCENARIO CONSEQUENTIAL ACCOUNTING COMPARISON")
    print("  A: Cheap Renewables / Expensive Clean Firm (pure consequential)")
    print("  B: Clean Firm Investment / Everything Cheap (FOAK learning curve)")
    print("=" * 80)

    # Load egrid and fossil mix data
    with open('data/egrid_emission_rates.json') as f:
        egrid = json.load(f)
    with open('data/eia_fossil_mix.json') as f:
        fossil_mix = json.load(f)

    # Parse feasible mixes
    print("\nParsing feasible mixes from shared-data.js...")
    feasible_mixes = parse_feasible_mixes()
    total_mixes = sum(len(mixes) for iso_data in feasible_mixes.values()
                      for mixes in iso_data.values())
    print(f"  Loaded {total_mixes:,} feasible mixes across {len(ISOS)} ISOs × {len(THRESHOLDS)} thresholds")

    # Scenario A: path-dependent sequential optimization
    # At each threshold, resources from prior steps are locked in as floor
    print("\nScenario A: path-dependent sequential optimization...")
    results_a = find_optimal_mixes_sequential(feasible_mixes, SCENARIO_A, BASE_DEMAND_TWH)

    # Scenario B: everyone targets 95% CFE simultaneously
    # The optimal mix at 95% with cheap clean firm IS the portfolio
    print("\nScenario B: target 95% optimization...")
    results_b = find_optimal_mix_at_target(feasible_mixes, SCENARIO_B, BASE_DEMAND_TWH, target_threshold=95)
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
    for scenario, results, label in [(SCENARIO_A, results_a, 'cheap_ren'),
                                      (SCENARIO_B, results_b, 'clean_firm_invest')]:
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

                # Stepwise MAC from previous threshold
                stepwise_mac = None
                if prev_t is not None and prev_t in results.get(iso, {}):
                    prev_d = results[iso][prev_t]
                    delta_cost = d['effective_cost'] - prev_d['effective_cost']
                    # CO2 displaced in this step
                    clean_twh_prev = max(0, (prev_t - baseline_clean) / 100.0 * demand_twh)
                    clean_twh_cur = max(0, (t - baseline_clean) / 100.0 * demand_twh)
                    delta_clean = clean_twh_cur - clean_twh_prev
                    rate_prev, _ = compute_fossil_retirement(iso, prev_t, egrid, fossil_mix)
                    rate_cur, _ = compute_fossil_retirement(iso, t, egrid, fossil_mix)
                    avg_rate = (rate_prev + rate_cur) / 2
                    co2_mt = delta_clean * avg_rate
                    if co2_mt > 0.001:
                        stepwise_mac = round((delta_cost * demand_mwh) / (co2_mt * 1e6), 1)
                    else:
                        stepwise_mac = 9999

                # Clean firm TWh (nuclear + geothermal + CCS combined)
                cf_twh = d['resource_twh'].get('clean_firm', 0)
                ccs_twh = d['resource_twh'].get('ccs_ccgt', 0)
                firm_total_twh = cf_twh + ccs_twh

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
                })
                prev_t = t
            traj[iso] = iso_traj
        trajectories[label] = traj

    output = {
        'metadata': {
            'description': 'Dual-scenario consequential accounting comparison',
            'scenario_a': {
                'name': SCENARIO_A['name'],
                'description': SCENARIO_A['description'],
                'toggles': SCENARIO_A['toggles'],
                'method': 'path_dependent_sequential',
                'method_description': 'Sequential optimization: resources deployed at each threshold become the floor for the next. Cheapest $/tCO2 at each step, but locked into prior commitments.',
            },
            'scenario_b': {
                'name': SCENARIO_B['name'],
                'description': SCENARIO_B['description'],
                'toggles': SCENARIO_B['toggles'],
                'method': 'target_95_collective',
                'method_description': 'All buyers target 95% CFE simultaneously. Optimal mix at 95% with Low clean firm costs is the portfolio. FOAK investment drives learning curves to NOAK pricing.',
                'target_threshold': 95,
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

    out_json = 'data/scenario_comparison.json'
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nJSON: {out_json} ({os.path.getsize(out_json) / 1024:.0f} KB)")

    out_js = 'dashboard/js/scenario-comparison-data.js'
    os.makedirs(os.path.dirname(out_js), exist_ok=True)
    with open(out_js, 'w') as f:
        f.write("// Auto-generated by compute_scenario_comparison.py\n")
        f.write("// Dual-scenario consequential accounting comparison\n")
        f.write("// A: Cheap Renewables Only  B: Clean Firm Investment\n\n")
        f.write(f"const SCENARIO_COMPARISON = {json.dumps(output, indent=2, default=str)};\n")
    print(f"JS:   {out_js} ({os.path.getsize(out_js) / 1024:.0f} KB)")

    print("\nDone.")


if __name__ == '__main__':
    main()
