#!/usr/bin/env python3
"""
Step 3: Cost Optimization — Vectorized Cross-Evaluation
========================================================
For each (region, threshold):
  1. Load physically feasible mixes from PFS post-EF (Step 2 output)
  2. For each sensitivity combo: vectorized evaluation of ALL mixes
  3. Select cheapest mix → that becomes the scenario result
  4. Extract archetypes (unique winning mixes) for demand growth sweep
  5. For each (year, growth): evaluate archetypes, select cheapest

Pipeline position: Step 3 of 4
  Step 1 — PFS Generator (step1_pfs_generator.py)
  Step 2 — Efficient Frontier extraction (step2_efficient_frontier.py)
  Step 3 — Cost optimization (this file)
  Step 4 — Post-processing (step4_postprocess.py)

Input:  data/pfs_post_ef.parquet       (from Step 2)
Output: dashboard/overprocure_results.json  (full 9-dim factorial keys + feasible mixes)
        dashboard/demand_growth_results.json (full factorial: all combos × years × growth)
        data/cf_split_table.json            (tranche breakdown)

Key format: RFS_FF_TX_CCSq45_GEO (e.g., MMM_M_M_M1_M for CAISO all-Medium)
  CAISO: 17,496 combos per threshold. Non-CAISO: 5,832 combos per threshold.
"""

import json
import os
import time
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from itertools import product

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# ============================================================================
# COST TABLES
# ============================================================================

ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO']
RESOURCE_TYPES = ['clean_firm', 'solar', 'wind', 'ccs_ccgt', 'hydro']
LMH = ['L', 'M', 'H']
LEVEL_NAME = {'L': 'Low', 'M': 'Medium', 'H': 'High', 'N': 'None'}

# Regional demand (TWh, 2025 base year)
REGIONAL_DEMAND_TWH = {
    'CAISO': 224.039, 'ERCOT': 488.020, 'PJM': 843.331,
    'NYISO': 151.599, 'NEISO': 115.336,
}

# Existing clean generation as % of demand
GRID_MIX_SHARES = {
    'CAISO': {'clean_firm': 7.9, 'solar': 22.3, 'wind': 8.8, 'ccs_ccgt': 0, 'hydro': 9.5},
    'ERCOT': {'clean_firm': 8.6, 'solar': 13.8, 'wind': 23.6, 'ccs_ccgt': 0, 'hydro': 0.1},
    'PJM':   {'clean_firm': 32.1, 'solar': 2.9, 'wind': 3.8, 'ccs_ccgt': 0, 'hydro': 1.8},
    'NYISO': {'clean_firm': 18.4, 'solar': 0.0, 'wind': 4.7, 'ccs_ccgt': 0, 'hydro': 15.9},
    'NEISO': {'clean_firm': 23.8, 'solar': 1.4, 'wind': 3.9, 'ccs_ccgt': 0, 'hydro': 4.4},
}

# Wholesale electricity prices ($/MWh, base)
WHOLESALE_PRICES = {'CAISO': 30, 'ERCOT': 27, 'PJM': 34, 'NYISO': 42, 'NEISO': 41}

# Fossil fuel price adjustments ($/MWh delta from base wholesale)
FUEL_ADJUSTMENTS = {
    'CAISO': {'Low': -5, 'Medium': 0, 'High': 10},
    'ERCOT': {'Low': -7, 'Medium': 0, 'High': 12},
    'PJM':   {'Low': -6, 'Medium': 0, 'High': 11},
    'NYISO': {'Low': -4, 'Medium': 0, 'High': 8},
    'NEISO': {'Low': -4, 'Medium': 0, 'High': 8},
}

# LCOE tables by resource type × sensitivity × ISO ($/MWh)
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

# Transmission adders ($/MWh) by resource × tx level × ISO
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

# Nuclear uprate LCOE ($/MWh)
UPRATE_LCOE = {'L': 15, 'M': 25, 'H': 40}

# Nuclear new-build LCOE ($/MWh)
NUCLEAR_NEWBUILD_LCOE = {
    'L': {'CAISO': 70, 'ERCOT': 68, 'PJM': 72, 'NYISO': 75, 'NEISO': 73},
    'M': {'CAISO': 95, 'ERCOT': 90, 'PJM': 105, 'NYISO': 110, 'NEISO': 108},
    'H': {'CAISO': 140, 'ERCOT': 135, 'PJM': 160, 'NYISO': 170, 'NEISO': 165},
}

# Geothermal (CAISO only)
GEOTHERMAL_LCOE = {'L': 63, 'M': 88, 'H': 110}
GEO_CAP_TWH = 39.0

# CCS-CCGT LCOE with/without 45Q
CCS_LCOE_45Q_ON = {
    'L': {'CAISO': 58, 'ERCOT': 52, 'PJM': 62, 'NYISO': 78, 'NEISO': 75},
    'M': {'CAISO': 86, 'ERCOT': 71, 'PJM': 79, 'NYISO': 99, 'NEISO': 96},
    'H': {'CAISO': 115, 'ERCOT': 92, 'PJM': 102, 'NYISO': 128, 'NEISO': 122},
}
CCS_LCOE_45Q_OFF = {
    'L': {'CAISO': 87, 'ERCOT': 81, 'PJM': 91, 'NYISO': 107, 'NEISO': 104},
    'M': {'CAISO': 115, 'ERCOT': 100, 'PJM': 108, 'NYISO': 128, 'NEISO': 125},
    'H': {'CAISO': 144, 'ERCOT': 121, 'PJM': 131, 'NYISO': 157, 'NEISO': 151},
}

# Uprate cap: 5% of existing nuclear × 90% CF → TWh/yr
EXISTING_NUCLEAR_GW = {'CAISO': 2.3, 'ERCOT': 2.7, 'PJM': 32.0, 'NYISO': 3.4, 'NEISO': 3.5}
UPRATE_CAP_TWH = {iso: round(gw * 0.05 * 0.90 * 8760 / 1e3, 3) for iso, gw in EXISTING_NUCLEAR_GW.items()}

# Demand growth
DEMAND_GROWTH_RATES = {
    'CAISO':  {'Low': 0.014, 'Medium': 0.019, 'High': 0.025},
    'ERCOT':  {'Low': 0.020, 'Medium': 0.035, 'High': 0.055},
    'PJM':    {'Low': 0.015, 'Medium': 0.024, 'High': 0.036},
    'NYISO':  {'Low': 0.013, 'Medium': 0.020, 'High': 0.044},
    'NEISO':  {'Low': 0.009, 'Medium': 0.018, 'High': 0.029},
}
DEMAND_GROWTH_YEARS = list(range(2026, 2051))
DEMAND_GROWTH_LEVELS = ['Low', 'Medium', 'High']


# ── Gas Capacity Backup (resource adequacy) ──
RESOURCE_ADEQUACY_MARGIN = 0.15  # 15% reserve margin

PEAK_DEMAND_MW = {
    'CAISO': 43860, 'ERCOT': 83597, 'PJM': 160560, 'NYISO': 31857, 'NEISO': 25898,
}
EXISTING_GAS_CAPACITY_MW = {
    'CAISO': 37000, 'ERCOT': 55000, 'PJM': 75000, 'NYISO': 18000, 'NEISO': 14000,
}
# Lazard v16.0 CCGT annualized capacity cost ($/kW-yr)
NEW_CCGT_COST_KW_YR = {
    'CAISO': 112, 'ERCOT': 89, 'PJM': 99, 'NYISO': 114, 'NEISO': 105,
}
# Existing gas fixed O&M ($/kW-yr)
EXISTING_GAS_FOM_KW_YR = {
    'CAISO': 16, 'ERCOT': 13, 'PJM': 14, 'NYISO': 17, 'NEISO': 15,
}
# Capacity credits at system peak
PEAK_CAPACITY_CREDITS = {
    'clean_firm': 1.0, 'solar': 0.30, 'wind': 0.10, 'ccs_ccgt': 0.90,
    'hydro': 0.50, 'battery': 0.95, 'battery8': 0.95, 'ldes': 0.90,
}


def get_tx(rtype, tx_name, iso):
    """Lookup transmission adder for a resource type."""
    entry = TX_TABLES.get(rtype, {}).get(tx_name, 0)
    if isinstance(entry, dict):
        return entry.get(iso, 0)
    return entry


# ============================================================================
# VECTORIZED COST FUNCTION
# ============================================================================

def price_mix_batch(iso, arrays, sens, demand_twh, target_year=None, growth_rate=None):
    """
    Vectorized pricing of N mixes under a single sensitivity combo.

    Args:
        iso: region string
        arrays: dict with numpy arrays (shape N) for each mix dimension:
            'clean_firm', 'solar', 'wind', 'hydro' (% allocation),
            'procurement_pct', 'battery_dispatch_pct', 'battery8_dispatch_pct',
            'ldes_dispatch_pct', 'hourly_match_score'
        sens: dict with scalar sensitivity parameters:
            'ren', 'firm', 'batt', 'ldes_lvl', 'ccs', 'q45', 'fuel', 'tx', 'geo'
        demand_twh: base year demand (scalar)
        target_year, growth_rate: demand growth params (scalars, optional)

    Returns:
        numpy array of total_cost (shape N), and effective_cost (shape N)
    """
    N = len(arrays['clean_firm'])
    if N == 0:
        return np.array([]), np.array([])

    # Demand growth
    years = max(0, (target_year or 2025) - 2025)
    gf = (1 + (growth_rate or 0)) ** years if years > 0 else 1.0
    demand = demand_twh * gf
    existing_scale = 1.0 / gf

    # Sensitivity lookups (scalar)
    ren_name = LEVEL_NAME[sens['ren']]
    batt_name = LEVEL_NAME[sens['batt']]
    ldes_name = LEVEL_NAME[sens['ldes_lvl']]
    fuel_name = LEVEL_NAME[sens['fuel']]
    tx_name = LEVEL_NAME[sens['tx']]
    firm_lev = sens['firm']
    ccs_lev = sens['ccs']
    q45 = sens['q45']
    geo_lev = sens.get('geo')

    wholesale = max(5, WHOLESALE_PRICES[iso] + FUEL_ADJUSTMENTS[iso][fuel_name])
    existing = GRID_MIX_SHARES[iso]

    proc = arrays['procurement_pct'].astype(np.float64) / 100.0
    match_frac = arrays['hourly_match_score'].astype(np.float64) / 100.0

    total_cost = np.zeros(N, dtype=np.float64)

    # CCS pct = 100 - (cf + sol + wnd + hyd) -- implicit 5th resource
    cf_pct = arrays['clean_firm'].astype(np.float64)
    sol_pct = arrays['solar'].astype(np.float64)
    wnd_pct = arrays['wind'].astype(np.float64)
    hyd_pct = arrays['hydro'].astype(np.float64)
    ccs_pct = 100.0 - (cf_pct + sol_pct + wnd_pct + hyd_pct)
    ccs_pct = np.maximum(ccs_pct, 0.0)

    bat_pct = arrays['battery_dispatch_pct'].astype(np.float64)
    bat8_pct = arrays.get('battery8_dispatch_pct', np.zeros(N)).astype(np.float64)
    ldes_pct = arrays['ldes_dispatch_pct'].astype(np.float64)

    # --- Solar ---
    sol_demand_pct = proc * sol_pct
    sol_existing = min(existing['solar'] * existing_scale, 100.0)
    sol_existing_pct = np.minimum(sol_demand_pct, sol_existing)
    sol_new_pct = np.maximum(0, sol_demand_pct - sol_existing)
    sol_lcoe = LCOE_TABLES['solar'][ren_name][iso]
    sol_tx = get_tx('solar', tx_name, iso)
    total_cost += sol_existing_pct / 100.0 * wholesale + sol_new_pct / 100.0 * (sol_lcoe + sol_tx)

    # --- Wind ---
    wnd_demand_pct = proc * wnd_pct
    wnd_existing = min(existing['wind'] * existing_scale, 100.0)
    wnd_existing_pct = np.minimum(wnd_demand_pct, wnd_existing)
    wnd_new_pct = np.maximum(0, wnd_demand_pct - wnd_existing)
    wnd_lcoe = LCOE_TABLES['wind'][ren_name][iso]
    wnd_tx = get_tx('wind', tx_name, iso)
    total_cost += wnd_existing_pct / 100.0 * wholesale + wnd_new_pct / 100.0 * (wnd_lcoe + wnd_tx)

    # --- Hydro (always existing, wholesale-priced, $0 tx) ---
    hyd_demand_pct = proc * hyd_pct
    total_cost += hyd_demand_pct / 100.0 * wholesale

    # --- CCS-CCGT ---
    ccs_demand_pct = proc * ccs_pct
    ccs_existing = min(existing.get('ccs_ccgt', 0) * existing_scale, 100.0)
    ccs_existing_pct = np.minimum(ccs_demand_pct, ccs_existing)
    ccs_new_pct = np.maximum(0, ccs_demand_pct - ccs_existing)
    ccs_table = CCS_LCOE_45Q_ON if q45 == '1' else CCS_LCOE_45Q_OFF
    ccs_lcoe = ccs_table[ccs_lev][iso]
    ccs_tx = get_tx('ccs_ccgt', tx_name, iso)
    total_cost += ccs_existing_pct / 100.0 * wholesale + ccs_new_pct / 100.0 * (ccs_lcoe + ccs_tx)

    # --- Clean Firm (merit-order tranche pricing) ---
    cf_demand_pct = proc * cf_pct
    cf_existing = min(existing['clean_firm'] * existing_scale, 100.0)
    cf_existing_pct = np.minimum(cf_demand_pct, cf_existing)
    cf_new_pct = np.maximum(0, cf_demand_pct - cf_existing)

    existing_cost = cf_existing_pct / 100.0 * wholesale

    # New CF in TWh for tranche pricing
    new_cf_twh = cf_new_pct / 100.0 * demand

    # Transmission adders
    tx_cf = get_tx('clean_firm', tx_name, iso)
    tx_ccs_cf = get_tx('ccs_ccgt', tx_name, iso)

    # Tranche 1: Nuclear uprates (capped, no tx beyond grid connection)
    uprate_cap = UPRATE_CAP_TWH[iso]
    uprate_twh = np.minimum(new_cf_twh, uprate_cap)
    uprate_cost = uprate_twh * UPRATE_LCOE[firm_lev]
    remaining = np.maximum(0, new_cf_twh - uprate_twh)

    # Tranche 2: Geothermal (CAISO only, capped)
    geo_cost = np.zeros(N)
    if iso == 'CAISO' and geo_lev:
        geo_price = GEOTHERMAL_LCOE[geo_lev] + tx_cf
        geo_twh = np.minimum(remaining, GEO_CAP_TWH)
        geo_cost = geo_twh * geo_price
        remaining = np.maximum(0, remaining - geo_twh)

    # Tranche 3: Cheapest of nuclear new-build vs CCS
    nuclear_price = NUCLEAR_NEWBUILD_LCOE[firm_lev][iso] + tx_cf
    ccs_tranche_price = ccs_table[ccs_lev][iso] + tx_ccs_cf
    tranche3_is_nuclear = nuclear_price <= ccs_tranche_price
    tranche3_price = min(nuclear_price, ccs_tranche_price)
    tranche3_cost = remaining * tranche3_price

    # Tranche 3 split: nuclear new-build vs CCS-CCGT
    nuclear_newbuild_twh = remaining if tranche3_is_nuclear else np.zeros(N)
    ccs_tranche_twh = np.zeros(N) if tranche3_is_nuclear else remaining

    cf_total_new_cost = uprate_cost + geo_cost + tranche3_cost
    cf_cost_per_demand = cf_total_new_cost / demand
    total_cost += existing_cost + cf_cost_per_demand

    # --- Storage (battery toggle = 4hr + 8hr paired; LDES toggle = independent) ---
    bat4_lcoe = LCOE_TABLES['battery'][batt_name][iso] + get_tx('battery', tx_name, iso)
    bat8_lcoe = LCOE_TABLES['battery8'][batt_name][iso] + get_tx('battery8', tx_name, iso)
    ldes_lcoe = LCOE_TABLES['ldes'][ldes_name][iso] + get_tx('ldes', tx_name, iso)
    total_cost += (bat_pct / 100.0 * bat4_lcoe +
                   bat8_pct / 100.0 * bat8_lcoe +
                   ldes_pct / 100.0 * ldes_lcoe)

    # --- Gas Capacity Backup (resource adequacy) ---
    # Compute clean peak capacity contribution, then gas backup needed, then cost
    peak_mw = PEAK_DEMAND_MW[iso]
    ra_peak_mw = peak_mw * (1 + RESOURCE_ADEQUACY_MARGIN)
    demand_mwh = demand * 1e6  # TWh → MWh
    avg_demand_mw = demand_mwh / 8760

    # Clean peak capacity from each resource (vectorized)
    clean_peak_mw = (
        proc * cf_pct / 100.0 * avg_demand_mw * PEAK_CAPACITY_CREDITS['clean_firm'] +
        proc * sol_pct / 100.0 * avg_demand_mw * PEAK_CAPACITY_CREDITS['solar'] +
        proc * wnd_pct / 100.0 * avg_demand_mw * PEAK_CAPACITY_CREDITS['wind'] +
        proc * ccs_pct / 100.0 * avg_demand_mw * PEAK_CAPACITY_CREDITS['ccs_ccgt'] +
        proc * hyd_pct / 100.0 * avg_demand_mw * PEAK_CAPACITY_CREDITS['hydro'] +
        bat_pct / 100.0 * avg_demand_mw * PEAK_CAPACITY_CREDITS['battery'] +
        bat8_pct / 100.0 * avg_demand_mw * PEAK_CAPACITY_CREDITS['battery8'] +
        ldes_pct / 100.0 * avg_demand_mw * PEAK_CAPACITY_CREDITS['ldes']
    )

    gas_needed_mw = np.maximum(0, ra_peak_mw - clean_peak_mw)
    existing_gas_mw = EXISTING_GAS_CAPACITY_MW[iso]
    existing_gas_used_mw = np.minimum(gas_needed_mw, existing_gas_mw)
    new_gas_mw = np.maximum(0, gas_needed_mw - existing_gas_used_mw)

    # Annualized capacity cost: existing gas FOM + new CCGT full cost ($/yr)
    # Convert to $/MWh of demand: (MW × $/kW-yr × 1000) / demand_mwh
    gas_cost_per_mwh = (
        existing_gas_used_mw * EXISTING_GAS_FOM_KW_YR[iso] * 1000 +
        new_gas_mw * NEW_CCGT_COST_KW_YR[iso] * 1000
    ) / demand_mwh

    total_cost += gas_cost_per_mwh

    # Effective cost (total cost ÷ match fraction)
    effective_cost = np.where(match_frac > 0, total_cost / match_frac, 0)

    # Build tranche breakdown (per-mix arrays, shape N)
    tranche_data = {
        'cf_existing_twh': cf_existing_pct / 100.0 * demand,
        'uprate_twh': uprate_twh,
        'geo_twh': geo_twh if (iso == 'CAISO' and geo_lev) else np.zeros(N),
        'nuclear_newbuild_twh': nuclear_newbuild_twh,
        'ccs_tranche_twh': ccs_tranche_twh,
        'new_cf_twh': new_cf_twh,
        # Gas backup fields
        'gas_backup_mw': gas_needed_mw,
        'existing_gas_used_mw': existing_gas_used_mw,
        'new_gas_build_mw': new_gas_mw,
        'gas_cost_per_mwh': gas_cost_per_mwh,
        'clean_peak_mw': clean_peak_mw,
        'ra_peak_mw': np.full(N, ra_peak_mw),
    }

    return total_cost, effective_cost, tranche_data


# ============================================================================
# PRE-COMPUTED COEFFICIENT MODEL (Phase 1 acceleration)
# ============================================================================
# For base year (no growth), total_cost decomposes into:
#   tc[i] = Σ(coeff_matrix[i,k] × prices[k]) + constant[i]
# where coefficients are scenario-invariant and prices are 10 scalars.
# This avoids recomputing existing/new splits, tranche allocations, and
# gas backup for every scenario — just 10 scalar-vector multiplies.

# Coefficient column indices
_COL_WHOLESALE = 0  # existing generation priced at wholesale
_COL_SOL_NEW   = 1  # new solar (LCOE + tx)
_COL_WND_NEW   = 2  # new wind (LCOE + tx)
_COL_CCS_NEW   = 3  # new CCS-CCGT standalone (LCOE + tx)
_COL_UPRATE    = 4  # nuclear uprate tranche
_COL_GEO       = 5  # geothermal tranche (CAISO only, 0 elsewhere)
_COL_REMAINING = 6  # tranche 3: min(nuclear, CCS) new-build
_COL_BAT4      = 7  # 4hr battery dispatch
_COL_BAT8      = 8  # 8hr battery dispatch
_COL_LDES      = 9  # LDES dispatch
_N_COEFFS = 10


def precompute_base_year_coefficients(iso, arrays, demand_twh):
    """Pre-compute scenario-invariant coefficient arrays for base year.

    Returns:
        coeff_matrix: (N, 10) float64 — multiply by scenario prices
        constant: (N,) float64 — gas backup cost (scenario-invariant)
        extras: dict with per-element data needed for winner detail extraction
    """
    N = len(arrays['clean_firm'])

    proc = arrays['procurement_pct'].astype(np.float64) / 100.0
    match_frac = arrays['hourly_match_score'].astype(np.float64) / 100.0

    cf_pct = arrays['clean_firm'].astype(np.float64)
    sol_pct = arrays['solar'].astype(np.float64)
    wnd_pct = arrays['wind'].astype(np.float64)
    hyd_pct = arrays['hydro'].astype(np.float64)
    ccs_pct = np.maximum(0.0, 100.0 - (cf_pct + sol_pct + wnd_pct + hyd_pct))

    bat_pct = arrays['battery_dispatch_pct'].astype(np.float64)
    bat8_pct = arrays.get('battery8_dispatch_pct', np.zeros(N)).astype(np.float64)
    ldes_pct = arrays['ldes_dispatch_pct'].astype(np.float64)

    existing = GRID_MIX_SHARES[iso]

    # Demand pcts (proc × alloc) — scenario-invariant
    sol_demand_pct = proc * sol_pct
    wnd_demand_pct = proc * wnd_pct
    hyd_demand_pct = proc * hyd_pct
    ccs_demand_pct = proc * ccs_pct
    cf_demand_pct = proc * cf_pct

    # Existing/new splits (base year, existing_scale=1.0)
    sol_existing_pct = np.minimum(sol_demand_pct, existing['solar'])
    sol_new_pct = np.maximum(0, sol_demand_pct - existing['solar'])

    wnd_existing_pct = np.minimum(wnd_demand_pct, existing['wind'])
    wnd_new_pct = np.maximum(0, wnd_demand_pct - existing['wind'])

    ccs_ex = existing.get('ccs_ccgt', 0)
    ccs_existing_pct = np.minimum(ccs_demand_pct, ccs_ex)
    ccs_new_pct = np.maximum(0, ccs_demand_pct - ccs_ex)

    cf_existing_pct = np.minimum(cf_demand_pct, existing['clean_firm'])
    cf_new_pct = np.maximum(0, cf_demand_pct - existing['clean_firm'])

    # Clean firm tranche allocation (scenario-invariant quantities)
    new_cf_twh = cf_new_pct / 100.0 * demand_twh
    uprate_cap = UPRATE_CAP_TWH[iso]
    uprate_twh = np.minimum(new_cf_twh, uprate_cap)
    remaining_after_uprate = np.maximum(0, new_cf_twh - uprate_twh)

    geo_twh = np.zeros(N)
    remaining_after_geo = remaining_after_uprate
    if iso == 'CAISO':
        geo_twh = np.minimum(remaining_after_uprate, GEO_CAP_TWH)
        remaining_after_geo = np.maximum(0, remaining_after_uprate - geo_twh)

    # Gas backup (entirely scenario-invariant for base year)
    peak_mw = PEAK_DEMAND_MW[iso]
    ra_peak_mw = peak_mw * (1 + RESOURCE_ADEQUACY_MARGIN)
    demand_mwh = demand_twh * 1e6
    avg_demand_mw = demand_mwh / 8760

    clean_peak_mw = (
        proc * cf_pct / 100.0 * avg_demand_mw * PEAK_CAPACITY_CREDITS['clean_firm'] +
        proc * sol_pct / 100.0 * avg_demand_mw * PEAK_CAPACITY_CREDITS['solar'] +
        proc * wnd_pct / 100.0 * avg_demand_mw * PEAK_CAPACITY_CREDITS['wind'] +
        proc * ccs_pct / 100.0 * avg_demand_mw * PEAK_CAPACITY_CREDITS['ccs_ccgt'] +
        proc * hyd_pct / 100.0 * avg_demand_mw * PEAK_CAPACITY_CREDITS['hydro'] +
        bat_pct / 100.0 * avg_demand_mw * PEAK_CAPACITY_CREDITS['battery'] +
        bat8_pct / 100.0 * avg_demand_mw * PEAK_CAPACITY_CREDITS['battery8'] +
        ldes_pct / 100.0 * avg_demand_mw * PEAK_CAPACITY_CREDITS['ldes']
    )

    gas_needed_mw = np.maximum(0, ra_peak_mw - clean_peak_mw)
    existing_gas_mw = EXISTING_GAS_CAPACITY_MW[iso]
    existing_gas_used_mw = np.minimum(gas_needed_mw, existing_gas_mw)
    new_gas_mw = np.maximum(0, gas_needed_mw - existing_gas_used_mw)

    constant = (
        existing_gas_used_mw * EXISTING_GAS_FOM_KW_YR[iso] * 1000 +
        new_gas_mw * NEW_CCGT_COST_KW_YR[iso] * 1000
    ) / demand_mwh

    # Build coefficient matrix (N, 10)
    coeff_matrix = np.empty((N, _N_COEFFS), dtype=np.float64)
    coeff_matrix[:, _COL_WHOLESALE] = (sol_existing_pct + wnd_existing_pct +
                                        hyd_demand_pct + ccs_existing_pct +
                                        cf_existing_pct) / 100.0
    coeff_matrix[:, _COL_SOL_NEW] = sol_new_pct / 100.0
    coeff_matrix[:, _COL_WND_NEW] = wnd_new_pct / 100.0
    coeff_matrix[:, _COL_CCS_NEW] = ccs_new_pct / 100.0
    coeff_matrix[:, _COL_UPRATE] = uprate_twh / demand_twh
    coeff_matrix[:, _COL_GEO] = geo_twh / demand_twh
    coeff_matrix[:, _COL_REMAINING] = remaining_after_geo / demand_twh
    coeff_matrix[:, _COL_BAT4] = bat_pct / 100.0
    coeff_matrix[:, _COL_BAT8] = bat8_pct / 100.0
    coeff_matrix[:, _COL_LDES] = ldes_pct / 100.0

    extras = {
        'match_frac': match_frac,
        'cf_existing_twh': cf_existing_pct / 100.0 * demand_twh,
        'new_cf_twh': new_cf_twh,
        'uprate_twh': uprate_twh,
        'geo_twh': geo_twh,
        'remaining_twh': remaining_after_geo,
        'gas_needed_mw': gas_needed_mw,
        'existing_gas_used_mw': existing_gas_used_mw,
        'new_gas_mw': new_gas_mw,
        'clean_peak_mw': clean_peak_mw,
        'ra_peak_mw': ra_peak_mw,
    }

    return coeff_matrix, constant, extras


def get_scenario_prices(iso, sens):
    """Look up 10 scenario-dependent price scalars from sensitivity toggles.

    Returns:
        prices: (10,) float64 array matching coefficient column order
        wholesale: scalar for incremental cost calculation
        nuclear_price: scalar for tranche detail extraction
        ccs_tranche_price: scalar for tranche detail extraction
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

    wholesale = max(5, WHOLESALE_PRICES[iso] + FUEL_ADJUSTMENTS[iso][fuel_name])
    ccs_table = CCS_LCOE_45Q_ON if q45 == '1' else CCS_LCOE_45Q_OFF
    ccs_lcoe = ccs_table[ccs_lev][iso]
    ccs_tx = get_tx('ccs_ccgt', tx_name, iso)
    ccs_price = ccs_lcoe + ccs_tx

    nuclear_price = NUCLEAR_NEWBUILD_LCOE[firm_lev][iso] + get_tx('clean_firm', tx_name, iso)
    remaining_price = min(nuclear_price, ccs_price)

    geo_price = 0.0
    if iso == 'CAISO' and geo_lev:
        geo_price = GEOTHERMAL_LCOE[geo_lev] + get_tx('clean_firm', tx_name, iso)

    prices = np.array([
        wholesale,
        LCOE_TABLES['solar'][ren_name][iso] + get_tx('solar', tx_name, iso),
        LCOE_TABLES['wind'][ren_name][iso] + get_tx('wind', tx_name, iso),
        ccs_price,
        UPRATE_LCOE[firm_lev],
        geo_price,
        remaining_price,
        LCOE_TABLES['battery'][batt_name][iso] + get_tx('battery', tx_name, iso),
        LCOE_TABLES['battery8'][batt_name][iso] + get_tx('battery8', tx_name, iso),
        LCOE_TABLES['ldes'][ldes_name][iso] + get_tx('ldes', tx_name, iso),
    ], dtype=np.float64)

    return prices, wholesale, nuclear_price, ccs_price


# Numba-accelerated cost evaluation
if HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _eval_cost_numba(coeff_matrix, constant, prices):
        """Fused multiply-add: tc[i] = Σ(coeff[i,k] × prices[k]) + constant[i]"""
        N = coeff_matrix.shape[0]
        K = coeff_matrix.shape[1]
        tc = np.empty(N, dtype=np.float64)
        for i in prange(N):
            s = constant[i]
            for k in range(K):
                s += coeff_matrix[i, k] * prices[k]
            tc[i] = s
        return tc

    @njit(cache=True)
    def _argmin_indexed(tc, idx):
        """Argmin over a subset of tc without allocating a temporary array."""
        best_val = tc[idx[0]]
        best_pos = 0
        for i in range(1, len(idx)):
            v = tc[idx[i]]
            if v < best_val:
                best_val = v
                best_pos = i
        return idx[best_pos], best_val

    @njit(cache=True)
    def _argmin_bucketed(tc, scores, thresholds_desc):
        """O(N) multi-threshold argmin via bucket accumulation.

        Assigns each element to its highest qualifying threshold bucket
        (single pass), then takes cumulative min across buckets.
        Total work: N × ~7 comparisons + N × 1 comparison = O(N).
        vs naive 13 × argmin = O(13N).

        Thresholds must be sorted descending (100, 99, 97.5, ..., 50).
        """
        N = len(tc)
        n_thr = len(thresholds_desc)
        bucket_mins = np.full(n_thr, np.inf)
        bucket_min_idxs = np.zeros(n_thr, dtype=np.int64)

        # Pass 1: assign each element to its highest qualifying bucket
        for i in range(N):
            s = scores[i]
            t = tc[i]
            for k in range(n_thr):
                if s >= thresholds_desc[k]:
                    if t < bucket_mins[k]:
                        bucket_mins[k] = t
                        bucket_min_idxs[k] = i
                    break

        # Pass 2: cumulative min from highest to lowest threshold
        best_vals = np.full(n_thr, np.inf)
        best_idxs = np.zeros(n_thr, dtype=np.int64)
        running_min = np.inf
        running_idx = np.int64(0)
        for k in range(n_thr):
            if bucket_mins[k] < running_min:
                running_min = bucket_mins[k]
                running_idx = bucket_min_idxs[k]
            best_vals[k] = running_min
            best_idxs[k] = running_idx

        return best_idxs, best_vals


def eval_cost_fast(coeff_matrix, constant, prices):
    """Evaluate total cost using Numba if available, else numpy."""
    if HAS_NUMBA:
        return _eval_cost_numba(coeff_matrix, constant, prices)
    return coeff_matrix @ prices + constant


def argmin_indexed(tc, idx):
    """Find argmin of tc[idx] without allocating a copy."""
    if HAS_NUMBA:
        return _argmin_indexed(tc, idx)
    local_best = int(np.argmin(tc[idx]))
    return int(idx[local_best]), float(tc[idx[local_best]])


def eval_and_argmin_all(coeff_matrix, constant, prices, scores, thresholds_desc):
    """Parallel cost eval + bucketed multi-threshold argmin."""
    tc = eval_cost_fast(coeff_matrix, constant, prices)
    if HAS_NUMBA:
        return _argmin_bucketed(tc, scores, thresholds_desc)
    # Numpy fallback
    n_thr = len(thresholds_desc)
    best_idxs = np.zeros(n_thr, dtype=np.int64)
    best_vals = np.full(n_thr, np.inf)
    for k in range(n_thr):
        qualifying = np.where(scores >= thresholds_desc[k])[0]
        if len(qualifying) > 0:
            local_best = int(np.argmin(tc[qualifying]))
            best_idxs[k] = qualifying[local_best]
            best_vals[k] = tc[qualifying[local_best]]
    return best_idxs, best_vals


def build_winner_scenario(arrays, extras, best_idx, sens, iso, demand_twh,
                          tc_val, wholesale, nuclear_price, ccs_price):
    """Build scenario result dict for a single winning mix."""
    match_frac = extras['match_frac'][best_idx]
    ec_val = tc_val / match_frac if match_frac > 0 else 0.0

    best_mix = arrays_to_mix_dict(arrays, best_idx)

    # Tranche detail (scenario-dependent: which is cheaper, nuclear or CCS?)
    tranche3_is_nuclear = nuclear_price <= ccs_price
    remaining_twh = float(extras['remaining_twh'][best_idx])

    return {
        'resource_mix': best_mix['resource_mix'],
        'procurement_pct': best_mix['procurement_pct'],
        'hourly_match_score': best_mix['hourly_match_score'],
        'battery_dispatch_pct': best_mix['battery_dispatch_pct'],
        'battery8_dispatch_pct': best_mix['battery8_dispatch_pct'],
        'ldes_dispatch_pct': best_mix['ldes_dispatch_pct'],
        'costs': {
            'total_cost': round(tc_val, 2),
            'effective_cost': round(ec_val, 2),
            'incremental': round(ec_val - wholesale, 2),
            'wholesale': wholesale,
        },
        'tranche_costs': {
            'cf_existing_twh': round(float(extras['cf_existing_twh'][best_idx]), 3),
            'uprate_twh': round(float(extras['uprate_twh'][best_idx]), 3),
            'geo_twh': round(float(extras['geo_twh'][best_idx]), 3),
            'nuclear_newbuild_twh': round(remaining_twh if tranche3_is_nuclear else 0.0, 3),
            'ccs_tranche_twh': round(0.0 if tranche3_is_nuclear else remaining_twh, 3),
            'new_cf_twh': round(float(extras['new_cf_twh'][best_idx]), 3),
        },
        'gas_backup': {
            'gas_backup_needed_mw': round(float(extras['gas_needed_mw'][best_idx])),
            'existing_gas_used_mw': round(float(extras['existing_gas_used_mw'][best_idx])),
            'new_gas_build_mw': round(float(extras['new_gas_mw'][best_idx])),
            'gas_cost_per_mwh': round(float(
                (extras['existing_gas_used_mw'][best_idx] * EXISTING_GAS_FOM_KW_YR[iso] * 1000 +
                 extras['new_gas_mw'][best_idx] * NEW_CCGT_COST_KW_YR[iso] * 1000)
                / (demand_twh * 1e6)), 2),
            'clean_peak_capacity_mw': round(float(extras['clean_peak_mw'][best_idx])),
            'ra_peak_mw': round(float(extras['ra_peak_mw'])),
        },
    }


# ============================================================================
# SENSITIVITY KEY HELPERS
# ============================================================================

def make_scenario_key(r, f, batt, ldes, ff, tx, ccs, q45, geo):
    """Build 9-dim scenario key.
    Format: {Ren}{Firm}{Batt}{LDES}_{Fuel}_{Tx}_{CCS}{45Q}_{Geo}
    Example: MMMM_M_M_M1_M (all-Medium, 45Q on, CAISO)
    Geo='X' for non-CAISO ISOs (no geothermal resource).
    """
    geo_code = geo if geo else 'X'
    return f"{r}{f}{batt}{ldes}_{ff}_{tx}_{ccs}{q45}_{geo_code}"


def medium_key(iso):
    """Return the all-Medium scenario key for an ISO.
    CAISO: MMMM_M_M_M1_M  (geothermal=M)
    Others: MMMM_M_M_M1_X  (no geothermal)
    """
    geo = 'M' if iso == 'CAISO' else 'X'
    return f'MMMM_M_M_M1_{geo}'


def make_old_key(r, f, s, ff, tx):
    """Build old 5-dim scenario key like MMM_M_M (backward compat)."""
    return f"{r}{f}{s}_{ff}_{tx}"


def build_sensitivity_combos(iso):
    """Build all sensitivity combos for an ISO.
    Returns list of (scenario_key, sens_dict) tuples.

    9 toggles: Ren(3) × Firm(3) × Battery(3) × LDES(3) × CCS(3) × 45Q(2) × Fuel(3) × Tx(4) × Geo(3|1)
    CAISO:     3^6 × 2 × 4 × 3 = 17,496 combos per threshold.
    Non-CAISO: 3^6 × 2 × 4 × 1 = 5,832 combos per threshold.
    """
    combos = []
    tx_levels = ['N', 'L', 'M', 'H']
    ccs_levels = LMH
    q45_states = ['1', '0']
    geo_levels = LMH if iso == 'CAISO' else [None]

    for r, f, batt, ldes in product(LMH, LMH, LMH, LMH):
        for ff in LMH:
            for tx in tx_levels:
                for ccs in ccs_levels:
                    for q45 in q45_states:
                        for geo in geo_levels:
                            key = make_scenario_key(r, f, batt, ldes, ff, tx, ccs, q45, geo)
                            sens = {
                                'ren': r, 'firm': f,
                                'batt': batt, 'ldes_lvl': ldes,
                                'ccs': ccs, 'q45': q45,
                                'fuel': ff, 'tx': tx, 'geo': geo,
                            }
                            combos.append((key, sens))
    return combos


# ============================================================================
# LOAD PFS POST-EF
# ============================================================================

PFS_POST_EF_PATH = Path('data/pfs_post_ef.parquet')
RESULTS_PATH = Path('dashboard/overprocure_results.json')
DG_RESULTS_PATH = Path('dashboard/demand_growth_results.json')

# Thresholds to include in backward-compat output
OUTPUT_THRESHOLDS = [50, 60, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 100]


def load_pfs_post_ef():
    """Load PFS post-EF (threshold-free) and organize by ISO.

    The post-EF data has no threshold column — each unique mix is stored once
    with its actual match score. Step 3 filters by score >= threshold at
    evaluation time, enabling cross-threshold picking.

    Returns:
        pfs: dict keyed by ISO → numpy arrays of all mixes
        thr_indices: dict keyed by (ISO, threshold) → index array of qualifying mixes
    """
    if not PFS_POST_EF_PATH.exists():
        raise FileNotFoundError(f"Run Step 2 first: {PFS_POST_EF_PATH}")

    table = pq.read_table(PFS_POST_EF_PATH)
    print(f"Loaded PFS post-EF: {table.num_rows:,} rows")

    import pyarrow.compute as pc

    pfs = {}
    thr_indices = {}

    for iso in ISOS:
        mask = pc.equal(table.column('iso'), iso)
        sub = table.filter(mask)
        if sub.num_rows == 0:
            continue

        arrays = {
            'clean_firm': sub.column('clean_firm').to_numpy(),
            'solar': sub.column('solar').to_numpy(),
            'wind': sub.column('wind').to_numpy(),
            'hydro': sub.column('hydro').to_numpy(),
            'procurement_pct': sub.column('procurement_pct').to_numpy(),
            'battery_dispatch_pct': sub.column('battery_dispatch_pct').to_numpy(),
            'battery8_dispatch_pct': (sub.column('battery8_dispatch_pct').to_numpy()
                                      if 'battery8_dispatch_pct' in sub.column_names
                                      else np.zeros(sub.num_rows, dtype=np.int64)),
            'ldes_dispatch_pct': sub.column('ldes_dispatch_pct').to_numpy(),
            'hourly_match_score': sub.column('hourly_match_score').to_numpy(),
        }
        pfs[iso] = arrays

        # Pre-compute threshold index arrays (score >= threshold)
        scores = arrays['hourly_match_score']
        for thr in OUTPUT_THRESHOLDS:
            idx = np.where(scores >= thr)[0]
            if len(idx) > 0:
                thr_indices[(iso, thr)] = idx

        print(f"  {iso}: {sub.num_rows:,} mixes, "
              f"thresholds with data: {sum(1 for t in OUTPUT_THRESHOLDS if (iso, t) in thr_indices)}")

    return pfs, thr_indices


def arrays_to_mix_dict(arrays, idx):
    """Extract a single mix from arrays as a dict."""
    ccs_pct = max(0, 100 - (int(arrays['clean_firm'][idx]) + int(arrays['solar'][idx]) +
                             int(arrays['wind'][idx]) + int(arrays['hydro'][idx])))
    bat8 = arrays.get('battery8_dispatch_pct')
    return {
        'resource_mix': {
            'clean_firm': int(arrays['clean_firm'][idx]),
            'solar': int(arrays['solar'][idx]),
            'wind': int(arrays['wind'][idx]),
            'ccs_ccgt': ccs_pct,
            'hydro': int(arrays['hydro'][idx]),
        },
        'procurement_pct': int(arrays['procurement_pct'][idx]),
        'hourly_match_score': round(float(arrays['hourly_match_score'][idx]), 4),
        'battery_dispatch_pct': int(arrays['battery_dispatch_pct'][idx]),
        'battery8_dispatch_pct': int(bat8[idx]) if bat8 is not None else 0,
        'ldes_dispatch_pct': int(arrays['ldes_dispatch_pct'][idx]),
    }


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("=" * 70)
    print("  STEP 3: COST OPTIMIZATION (v4 — vectorized, threshold-free PFS)")
    print("=" * 70)

    total_start = time.time()

    # Load PFS post-EF (threshold-free: one set of mixes per ISO)
    pfs, thr_indices = load_pfs_post_ef()
    print(f"  ISOs loaded: {len(pfs)}")

    # Build output structure
    output = {
        'config': {
            'data_year': 2025,
            'thresholds': OUTPUT_THRESHOLDS,
            'resource_types': RESOURCE_TYPES,
            'wholesale_prices': WHOLESALE_PRICES,
            'grid_mix_shares': GRID_MIX_SHARES,
            'lcoe_tables': LCOE_TABLES,
            'transmission_tables': TX_TABLES,
            'fuel_prices': FUEL_ADJUSTMENTS,
            'tranche_model': {
                'uprate_caps_twh': UPRATE_CAP_TWH,
                'uprate_lcoe': UPRATE_LCOE,
                'nuclear_newbuild_lcoe': NUCLEAR_NEWBUILD_LCOE,
                'geothermal_lcoe': {'L': {'CAISO': 63}, 'M': {'CAISO': 88}, 'H': {'CAISO': 110}},
                'geothermal_cap_twh': GEO_CAP_TWH,
                'ccs_lcoe_45q_on': CCS_LCOE_45Q_ON,
                'ccs_lcoe_45q_off': CCS_LCOE_45Q_OFF,
                'existing_nuclear_gw': EXISTING_NUCLEAR_GW,
            },
            'demand_growth_rates': DEMAND_GROWTH_RATES,
        },
        'results': {},
    }
    cf_split_table = []

    # ================================================================
    # PHASE 1: Base year cross-evaluation (pre-computed + Numba)
    # ================================================================
    # Pre-compute scenario-invariant coefficient arrays per ISO, then
    # evaluate total_cost as a fused multiply-add (10 coefficients × 10 prices).
    # Numba prange parallelizes across CPU cores; no temporary arrays.
    print("\n--- PHASE 1: Base year cross-evaluation (pre-computed + Numba) ---")
    print(f"  Numba JIT: {'enabled' if HAS_NUMBA else 'DISABLED (numpy fallback)'}")
    phase1_start = time.time()

    # Warm up Numba JIT on first call (compile cost)
    if HAS_NUMBA:
        _dummy_cm = np.zeros((2, _N_COEFFS))
        _dummy_c = np.zeros(2)
        _dummy_p = np.zeros(_N_COEFFS)
        _dummy_s = np.array([50.0, 100.0])
        _dummy_t = np.array([100.0, 50.0])
        _dummy_tc = _eval_cost_numba(_dummy_cm, _dummy_c, _dummy_p)
        _argmin_bucketed(_dummy_tc, _dummy_s, _dummy_t)
        print("  Numba JIT warmup complete")

    # Archetypes per ISO (union of winners across all thresholds) for Phase 2
    archetypes = {}

    for iso in ISOS:
        if iso not in pfs:
            continue

        arrays = pfs[iso]
        N = len(arrays['clean_firm'])
        demand_twh = REGIONAL_DEMAND_TWH[iso]

        output['results'][iso] = {
            'annual_demand_mwh': demand_twh * 1e6,
            'thresholds': {},
        }

        # Pre-compute coefficient matrix + constant (one-time per ISO)
        coeff_matrix, constant, extras = precompute_base_year_coefficients(
            iso, arrays, demand_twh)
        scores = arrays['hourly_match_score'].astype(np.float64)

        all_combos = build_sensitivity_combos(iso)
        n_combos = len(all_combos)

        active_thresholds = [thr for thr in OUTPUT_THRESHOLDS
                             if (iso, thr) in thr_indices]
        # Sorted descending for fused eval+argmin (exploits nested structure)
        thresholds_desc = np.array(sorted(active_thresholds, reverse=True),
                                   dtype=np.float64)
        # Map from position in thresholds_desc back to threshold value
        thr_pos = {float(thresholds_desc[k]): k for k in range(len(thresholds_desc))}

        thr_data = {}
        thr_arch_sets = {}
        for thr in active_thresholds:
            thr_data[thr] = {'scenarios': {}, 'feasible_mixes': {}}
            thr_arch_sets[thr] = set()

        iso_arch_set = set()
        iso_start = time.time()

        # Parallel eval (Numba prange) + bucketed argmin (O(N) single pass)
        for combo_i, (scenario_key, sens) in enumerate(all_combos):
            prices, wholesale, nuclear_price, ccs_price = get_scenario_prices(iso, sens)
            best_idxs, best_vals = eval_and_argmin_all(
                coeff_matrix, constant, prices, scores, thresholds_desc)

            for thr in active_thresholds:
                k = thr_pos[float(thr)]
                if best_vals[k] == np.inf:
                    continue
                best_idx = int(best_idxs[k])
                tc_val = float(best_vals[k])

                thr_arch_sets[thr].add(best_idx)
                iso_arch_set.add(best_idx)

                scenario = build_winner_scenario(
                    arrays, extras, best_idx, sens, iso, demand_twh,
                    tc_val, wholesale, nuclear_price, ccs_price)

                thr_data[thr]['scenarios'][scenario_key] = scenario

            if (combo_i + 1) % 1000 == 0:
                elapsed = time.time() - iso_start
                rate = (combo_i + 1) / elapsed
                remaining = (n_combos - combo_i - 1) / rate
                print(f"    {iso} {combo_i+1}/{n_combos} "
                      f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

        # Store feasible mixes per threshold
        for thr in active_thresholds:
            idx = thr_indices[(iso, thr)]
            N_thr = len(idx)

            max_feasible = min(N_thr, 500)
            step = max(1, N_thr // max_feasible)
            sample_full = list(idx[::step])
            for aidx in thr_arch_sets[thr]:
                if aidx not in sample_full:
                    sample_full.append(aidx)
            sample_full.sort()

            idx_arr = np.array(sample_full)
            ccs_pct = np.maximum(0, 100 - (arrays['clean_firm'][idx_arr].astype(int) +
                                            arrays['solar'][idx_arr].astype(int) +
                                            arrays['wind'][idx_arr].astype(int) +
                                            arrays['hydro'][idx_arr].astype(int)))
            bat8 = arrays.get('battery8_dispatch_pct', np.zeros(N, dtype=np.int64))
            thr_data[thr]['feasible_mixes'] = {
                'clean_firm': arrays['clean_firm'][idx_arr].astype(int).tolist(),
                'solar': arrays['solar'][idx_arr].astype(int).tolist(),
                'wind': arrays['wind'][idx_arr].astype(int).tolist(),
                'ccs_ccgt': ccs_pct.tolist(),
                'hydro': arrays['hydro'][idx_arr].astype(int).tolist(),
                'procurement_pct': arrays['procurement_pct'][idx_arr].astype(int).tolist(),
                'hourly_match_score': np.round(arrays['hourly_match_score'][idx_arr], 4).tolist(),
                'battery_dispatch_pct': arrays['battery_dispatch_pct'][idx_arr].astype(int).tolist(),
                'battery8_dispatch_pct': bat8[idx_arr].astype(int).tolist(),
                'ldes_dispatch_pct': arrays['ldes_dispatch_pct'][idx_arr].astype(int).tolist(),
            }

            t_str = str(thr)
            output['results'][iso]['thresholds'][t_str] = thr_data[thr]

        archetypes[iso] = iso_arch_set
        iso_elapsed = time.time() - iso_start

        print(f"  {iso:>6}: {N:>7,} mixes, {n_combos} scenarios, "
              f"{len(active_thresholds)} thresholds, "
              f"{len(iso_arch_set)} archetypes — {iso_elapsed:.0f}s")

    phase1_elapsed = time.time() - phase1_start
    print(f"\nPhase 1 complete: {phase1_elapsed:.0f}s")

    # ================================================================
    # PHASE 2: Demand growth sweep on archetypes (vectorized)
    # ================================================================
    # Archetypes are the union of all winning mixes per ISO across all thresholds.
    # Evaluate all archetypes once per (scenario, year, growth), then pick best
    # per threshold using score-based filtering.
    print("\n--- PHASE 2: Demand growth sweep (vectorized, cross-threshold) ---")
    phase2_start = time.time()

    dg_output = {
        'meta': {
            'years': DEMAND_GROWTH_YEARS,
            'growth_levels': DEMAND_GROWTH_LEVELS,
            'growth_rates': DEMAND_GROWTH_RATES,
            'base_year': 2025,
            'fields': ['mix_idx', 'total_cost', 'effective_cost', 'incremental'],
            'note': 'Full 9-dim factorial: RFBL_FF_TX_CCSq45_GEO keys (R=ren,F=firm,B=batt,L=ldes). '
                    'CAISO: 17,496 combos. Non-CAISO: 5,832 combos.',
        },
        'results': {},
    }

    for iso in ISOS:
        if iso not in pfs or iso not in archetypes:
            continue

        dg_output['results'][iso] = {}
        arrays = pfs[iso]
        demand_twh = REGIONAL_DEMAND_TWH[iso]
        iso_rates = DEMAND_GROWTH_RATES[iso]
        all_combos = build_sensitivity_combos(iso)

        arch_indices = sorted(archetypes[iso])
        n_arch = len(arch_indices)
        if n_arch == 0:
            continue

        # Extract archetype sub-arrays (evaluate only these in Phase 2)
        arch_arrays = {k: arrays[k][arch_indices] for k in arrays}

        # Pre-compute which archetypes qualify for each threshold
        arch_scores = arch_arrays['hourly_match_score']
        arch_thr_mask = {}  # thr → indices into arch_arrays that qualify
        for thr in OUTPUT_THRESHOLDS:
            qualifying = np.where(arch_scores >= thr)[0]
            if len(qualifying) > 0:
                arch_thr_mask[thr] = qualifying

        # Initialize per-threshold result dicts
        thr_dg = {thr: {} for thr in arch_thr_mask}

        # Full 9-dim factorial sweep — evaluate once per (scenario, year, growth)
        for scenario_key, sens in all_combos:
            wholesale = max(5, WHOLESALE_PRICES[iso] +
                            FUEL_ADJUSTMENTS[iso][LEVEL_NAME[sens['fuel']]])

            thr_year_results = {thr: {} for thr in arch_thr_mask}

            for year in DEMAND_GROWTH_YEARS:
                thr_growth_results = {thr: {} for thr in arch_thr_mask}

                for g_level in DEMAND_GROWTH_LEVELS:
                    g_rate = iso_rates[g_level]
                    tc, ec, _ = price_mix_batch(
                        iso, arch_arrays, sens, demand_twh,
                        target_year=year, growth_rate=g_rate
                    )

                    # Pick best per threshold using pre-computed score masks
                    for thr in arch_thr_mask:
                        qual_idx = arch_thr_mask[thr]
                        best_local = int(qual_idx[np.argmin(tc[qual_idx])])
                        full_idx = arch_indices[best_local]
                        thr_growth_results[thr][g_level] = [
                            full_idx,
                            round(float(tc[best_local]), 2),
                            round(float(ec[best_local]), 2),
                            round(float(ec[best_local]) - wholesale, 2),
                        ]

                for thr in arch_thr_mask:
                    thr_year_results[thr][str(year)] = thr_growth_results[thr]

            for thr in arch_thr_mask:
                thr_dg[thr][scenario_key] = thr_year_results[thr]

        for thr in arch_thr_mask:
            dg_output['results'][iso][str(thr)] = thr_dg[thr]

        print(f"  {iso:>6}: {n_arch} archetypes, "
              f"{len(arch_thr_mask)} thresholds, "
              f"{len(all_combos)} scenarios × "
              f"{len(DEMAND_GROWTH_YEARS)} years × "
              f"{len(DEMAND_GROWTH_LEVELS)} growth")

    phase2_elapsed = time.time() - phase2_start
    print(f"\nPhase 2 complete: {phase2_elapsed:.0f}s")

    # ================================================================
    # SAVE OUTPUTS
    # ================================================================
    print("\n--- Saving outputs ---")

    # Results JSON
    os.makedirs(RESULTS_PATH.parent, exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, separators=(',', ':'))
    r_size = os.path.getsize(RESULTS_PATH) / (1024 * 1024)
    print(f"  {RESULTS_PATH} ({r_size:.1f} MB)")

    # Demand growth results
    with open(DG_RESULTS_PATH, 'w') as f:
        json.dump(dg_output, f, separators=(',', ':'))
    dg_size = os.path.getsize(DG_RESULTS_PATH) / (1024 * 1024)
    print(f"  {DG_RESULTS_PATH} ({dg_size:.1f} MB)")

    # CF split table
    with open('data/cf_split_table.json', 'w') as f:
        json.dump(cf_split_table, f, indent=2)

    # ================================================================
    # SAVE PARQUET OUTPUTS (compact, git-friendly)
    # ================================================================
    print("\n--- Saving parquet outputs ---")
    import pandas as pd

    # 1. Scenarios parquet: flatten ISO × threshold × scenario → rows
    sc_rows = []
    for iso, iso_data in output['results'].items():
        annual = iso_data.get('annual_demand_mwh', 0)
        for t_str, thr_data in iso_data.get('thresholds', {}).items():
            for sc_key, sc in thr_data.get('scenarios', {}).items():
                row = {'iso': iso, 'threshold': float(t_str),
                       'scenario': sc_key, 'annual_demand_mwh': annual}
                for k, v in sc.get('resource_mix', {}).items():
                    row[f'mix_{k}'] = v
                row['procurement_pct'] = sc.get('procurement_pct')
                row['hourly_match_score'] = sc.get('hourly_match_score')
                row['battery_dispatch_pct'] = sc.get('battery_dispatch_pct')
                row['battery8_dispatch_pct'] = sc.get('battery8_dispatch_pct')
                row['ldes_dispatch_pct'] = sc.get('ldes_dispatch_pct')
                for k, v in sc.get('costs', {}).items():
                    row[f'cost_{k}'] = v
                for k, v in sc.get('tranche_costs', {}).items():
                    row[f'tranche_{k}'] = v
                for k, v in sc.get('gas_backup', {}).items():
                    row[f'gas_{k}'] = v
                sc_rows.append(row)
    df_sc = pd.DataFrame(sc_rows)
    df_sc.to_parquet('dashboard/overprocure_scenarios.parquet',
                     index=False, compression='zstd')
    print(f"  overprocure_scenarios.parquet: {len(df_sc)} rows, "
          f"{os.path.getsize('dashboard/overprocure_scenarios.parquet') / 1e6:.1f} MB")

    # 2. Feasible mixes parquet
    mix_rows = []
    for iso, iso_data in output['results'].items():
        for t_str, thr_data in iso_data.get('thresholds', {}).items():
            fm = thr_data.get('feasible_mixes', {})
            if not fm:
                continue
            n = len(list(fm.values())[0])
            for i in range(n):
                row = {'iso': iso, 'threshold': float(t_str)}
                for k, vals in fm.items():
                    row[k] = vals[i]
                mix_rows.append(row)
    df_mix = pd.DataFrame(mix_rows)
    df_mix.to_parquet('dashboard/overprocure_feasible_mixes.parquet',
                      index=False, compression='zstd')
    print(f"  overprocure_feasible_mixes.parquet: {len(df_mix)} rows, "
          f"{os.path.getsize('dashboard/overprocure_feasible_mixes.parquet') / 1e6:.1f} MB")

    # 3. Overprocure meta (config + postprocessing, small JSON)
    meta = {k: output[k] for k in output if k != 'results'}
    with open('dashboard/overprocure_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  overprocure_meta.json: "
          f"{os.path.getsize('dashboard/overprocure_meta.json') / 1e3:.1f} KB")

    # 4. Demand growth parquet
    dg_rows = []
    for iso, iso_thrs in dg_output.get('results', {}).items():
        for t_str, thr_scenarios in iso_thrs.items():
            for sc_key, year_data in thr_scenarios.items():
                for year_str, growth_data in year_data.items():
                    for g_level, vals in growth_data.items():
                        dg_rows.append({
                            'iso': iso,
                            'threshold': float(t_str),
                            'scenario': sc_key,
                            'year': int(year_str),
                            'growth_level': g_level,
                            'mix_idx': vals[0],
                            'total_cost': vals[1],
                            'effective_cost': vals[2],
                            'incremental': vals[3],
                        })
    df_dg = pd.DataFrame(dg_rows)
    df_dg.to_parquet('dashboard/demand_growth_results.parquet',
                     index=False, compression='zstd')
    print(f"  demand_growth_results.parquet: {len(df_dg)} rows, "
          f"{os.path.getsize('dashboard/demand_growth_results.parquet') / 1e6:.1f} MB")

    # 5. Demand growth meta
    dg_meta = dg_output.get('meta', {})
    with open('dashboard/demand_growth_meta.json', 'w') as f:
        json.dump(dg_meta, f, indent=2)

    # ================================================================
    # SUMMARY
    # ================================================================
    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  STEP 3 COMPLETE in {total_elapsed:.0f}s")
    print(f"{'='*70}")

    # Print Medium scenario summary (ISO-aware medium key)
    print("\nAll-Medium (45Q=ON) summary:")
    for iso in ISOS:
        mk = medium_key(iso)
        print(f"\n  {iso} (key={mk}):")
        for thr in OUTPUT_THRESHOLDS:
            t_str = str(thr)
            sc = output['results'][iso].get('thresholds', {}).get(t_str, {}).get('scenarios', {}).get(mk)
            if not sc:
                continue
            rm = sc['resource_mix']
            eff = sc['costs']['effective_cost']
            match = sc['hourly_match_score']
            proc = sc['procurement_pct']
            print(f"    {thr:>5}%: CF={rm.get('clean_firm',0):>2} Sol={rm.get('solar',0):>2} "
                  f"Wnd={rm.get('wind',0):>2} CCS={rm.get('ccs_ccgt',0):>2} Hyd={rm.get('hydro',0):>2} | "
                  f"proc={proc}% eff=${eff:.1f}/MWh match={match}%")


if __name__ == '__main__':
    main()
