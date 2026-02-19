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
Output: dashboard/overprocure_results.json  (backward-compat 324-key + feasible mixes)
        dashboard/demand_growth_results.json (full factorial: all combos × years × growth)
        data/cf_split_table.json            (tranche breakdown)
"""

import json
import os
import time
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from itertools import product

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
            'procurement_pct', 'battery_dispatch_pct', 'ldes_dispatch_pct',
            'hourly_match_score'
        sens: dict with scalar sensitivity parameters:
            'ren', 'firm', 'stor', 'ccs', 'q45', 'fuel', 'tx', 'geo'
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
    stor_name = LEVEL_NAME[sens['stor']]
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

    # --- Storage ---
    bat_lcoe = LCOE_TABLES['battery'][stor_name][iso] + get_tx('battery', tx_name, iso)
    ldes_lcoe = LCOE_TABLES['ldes'][stor_name][iso] + get_tx('ldes', tx_name, iso)
    total_cost += bat_pct / 100.0 * bat_lcoe + ldes_pct / 100.0 * ldes_lcoe

    # Effective cost
    effective_cost = np.where(match_frac > 0, total_cost / match_frac, 0)

    # Build tranche breakdown (per-mix arrays, shape N)
    tranche_data = {
        'cf_existing_twh': cf_existing_pct / 100.0 * demand,
        'uprate_twh': uprate_twh,
        'geo_twh': geo_twh if (iso == 'CAISO' and geo_lev) else np.zeros(N),
        'nuclear_newbuild_twh': nuclear_newbuild_twh,
        'ccs_tranche_twh': ccs_tranche_twh,
        'new_cf_twh': new_cf_twh,
    }

    return total_cost, effective_cost, tranche_data


# ============================================================================
# SENSITIVITY KEY HELPERS
# ============================================================================

def make_old_key(r, f, s, ff, tx):
    """Build old-format scenario key like MMM_M_M."""
    return f"{r}{f}{s}_{ff}_{tx}"


def build_sensitivity_combos(iso):
    """Build all sensitivity combos for an ISO.
    Returns list of (old_key, sens_dict) tuples."""
    combos = []
    tx_levels = ['N', 'L', 'M', 'H']
    ccs_levels = LMH
    q45_states = ['1', '0']
    geo_levels = LMH if iso == 'CAISO' else [None]

    for r, f, s in product(LMH, LMH, LMH):
        for ff in LMH:
            for tx in tx_levels:
                old_key = make_old_key(r, f, s, ff, tx)
                for ccs in ccs_levels:
                    for q45 in q45_states:
                        for geo in geo_levels:
                            sens = {
                                'ren': r, 'firm': f, 'stor': s,
                                'ccs': ccs, 'q45': q45,
                                'fuel': ff, 'tx': tx, 'geo': geo,
                            }
                            combos.append((old_key, sens))
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
    """Load PFS post-EF and organize by (ISO, threshold)."""
    if not PFS_POST_EF_PATH.exists():
        raise FileNotFoundError(f"Run Step 2.1 first: {PFS_POST_EF_PATH}")

    table = pq.read_table(PFS_POST_EF_PATH)
    print(f"Loaded PFS post-EF: {table.num_rows:,} rows")

    # Organize into dict of numpy arrays per (iso, threshold)
    data = {}
    for iso in ISOS:
        for thr in OUTPUT_THRESHOLDS:
            import pyarrow.compute as pc
            mask = pc.and_(
                pc.equal(table.column('iso'), iso),
                pc.equal(table.column('threshold'), float(thr))
            )
            sub = table.filter(mask)
            if sub.num_rows == 0:
                continue
            data[(iso, thr)] = {
                'clean_firm': sub.column('clean_firm').to_numpy(),
                'solar': sub.column('solar').to_numpy(),
                'wind': sub.column('wind').to_numpy(),
                'hydro': sub.column('hydro').to_numpy(),
                'procurement_pct': sub.column('procurement_pct').to_numpy(),
                'battery_dispatch_pct': sub.column('battery_dispatch_pct').to_numpy(),
                'ldes_dispatch_pct': sub.column('ldes_dispatch_pct').to_numpy(),
                'hourly_match_score': sub.column('hourly_match_score').to_numpy(),
            }
    return data


def arrays_to_mix_dict(arrays, idx):
    """Extract a single mix from arrays as a dict."""
    ccs_pct = max(0, 100 - (int(arrays['clean_firm'][idx]) + int(arrays['solar'][idx]) +
                             int(arrays['wind'][idx]) + int(arrays['hydro'][idx])))
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
        'ldes_dispatch_pct': int(arrays['ldes_dispatch_pct'][idx]),
    }


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("=" * 70)
    print("  STEP 2: COST OPTIMIZATION (v3 — vectorized, PFS post-EF)")
    print("=" * 70)

    total_start = time.time()

    # Load PFS post-EF
    pfs = load_pfs_post_ef()
    print(f"  Groups: {len(pfs)} (ISO × threshold)")

    # Build backward-compatible output structure
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
    # PHASE 1: Base year cross-evaluation (all combos, no growth)
    # ================================================================
    print("\n--- PHASE 1: Base year cross-evaluation ---")
    phase1_start = time.time()

    # We'll store archetypes per (iso, threshold) for phase 2
    archetypes = {}  # {(iso, thr): set of mix indices}

    for iso in ISOS:
        demand_twh = REGIONAL_DEMAND_TWH[iso]
        output['results'][iso] = {
            'annual_demand_mwh': demand_twh * 1e6,
            'thresholds': {},
        }

        # Build all sensitivity combos for this ISO
        all_combos = build_sensitivity_combos(iso)
        # Group by old_key for backward-compat output
        old_key_combos = {}
        for old_key, sens in all_combos:
            if old_key not in old_key_combos:
                old_key_combos[old_key] = []
            old_key_combos[old_key].append(sens)

        # Default sensitivity: Medium everything, 45Q=ON, CCS=M, Geo=M
        default_sens = {
            'ren': 'M', 'firm': 'M', 'stor': 'M', 'ccs': 'M',
            'q45': '1', 'fuel': 'M', 'tx': 'M',
            'geo': 'M' if iso == 'CAISO' else None,
        }

        for thr in OUTPUT_THRESHOLDS:
            key = (iso, thr)
            if key not in pfs:
                continue
            arrays = pfs[key]
            N = len(arrays['clean_firm'])

            t_str = str(thr)
            threshold_data = {'scenarios': {}, 'feasible_mixes': []}
            arch_set = set()

            # For each old-format scenario key, find the cheapest mix
            # using the default CCS/45Q/Geo mapping (firm level = CCS level)
            for old_key in old_key_combos:
                parts_rfs = old_key.split('_')[0]
                firm_lev = parts_rfs[1]  # F from RFS
                fuel_code = old_key.split('_')[1]
                tx_code = old_key.split('_')[2]

                # Default mapping: CCS = firm level, 45Q = ON, Geo = M
                sens = {
                    'ren': parts_rfs[0], 'firm': firm_lev, 'stor': parts_rfs[2],
                    'ccs': firm_lev, 'q45': '1',
                    'fuel': fuel_code, 'tx': tx_code,
                    'geo': 'M' if iso == 'CAISO' else None,
                }

                tc, ec, tranche = price_mix_batch(iso, arrays, sens, demand_twh)
                best_idx = int(np.argmin(tc))
                arch_set.add(best_idx)

                best_mix = arrays_to_mix_dict(arrays, best_idx)
                wholesale = max(5, WHOLESALE_PRICES[iso] +
                                FUEL_ADJUSTMENTS[iso][LEVEL_NAME[fuel_code]])

                scenario = {
                    'resource_mix': best_mix['resource_mix'],
                    'procurement_pct': best_mix['procurement_pct'],
                    'hourly_match_score': best_mix['hourly_match_score'],
                    'battery_dispatch_pct': best_mix['battery_dispatch_pct'],
                    'ldes_dispatch_pct': best_mix['ldes_dispatch_pct'],
                    'costs': {
                        'total_cost': round(float(tc[best_idx]), 2),
                        'effective_cost': round(float(ec[best_idx]), 2),
                        'incremental': round(float(ec[best_idx]) - wholesale, 2),
                        'wholesale': wholesale,
                    },
                    'tranche_costs': {
                        'cf_existing_twh': round(float(tranche['cf_existing_twh'][best_idx]), 3),
                        'uprate_twh': round(float(tranche['uprate_twh'][best_idx]), 3),
                        'geo_twh': round(float(tranche['geo_twh'][best_idx]), 3),
                        'nuclear_newbuild_twh': round(float(tranche['nuclear_newbuild_twh'][best_idx]), 3),
                        'ccs_tranche_twh': round(float(tranche['ccs_tranche_twh'][best_idx]), 3),
                        'new_cf_twh': round(float(tranche['new_cf_twh'][best_idx]), 3),
                    },
                }

                # Also compute no-45Q costs
                no45q_sens = dict(sens)
                no45q_sens['q45'] = '0'
                tc_no45q, ec_no45q, _ = price_mix_batch(iso, arrays, no45q_sens, demand_twh)
                scenario['no_45q_costs'] = {
                    'total_cost': round(float(tc_no45q[best_idx]), 2),
                    'effective_cost': round(float(ec_no45q[best_idx]), 2),
                    'incremental': round(float(ec_no45q[best_idx]) - wholesale, 2),
                    'wholesale': wholesale,
                }

                threshold_data['scenarios'][old_key] = scenario

            # Store feasible mixes for client-side repricing
            # (limit to a reasonable number — all unique allocations)
            max_feasible = min(N, 500)
            step = max(1, N // max_feasible)
            for i in range(0, N, step):
                threshold_data['feasible_mixes'].append(arrays_to_mix_dict(arrays, i))
            # Ensure all archetypes are in feasible_mixes
            for aidx in arch_set:
                mix_d = arrays_to_mix_dict(arrays, aidx)
                if mix_d not in threshold_data['feasible_mixes']:
                    threshold_data['feasible_mixes'].append(mix_d)

            archetypes[key] = arch_set
            output['results'][iso]['thresholds'][t_str] = threshold_data

            print(f"  {iso:>6} {thr:>5}%: {N:>6,} mixes, "
                  f"{len(old_key_combos)} scenarios, "
                  f"{len(arch_set)} archetypes")

    phase1_elapsed = time.time() - phase1_start
    print(f"\nPhase 1 complete: {phase1_elapsed:.0f}s")

    # ================================================================
    # PHASE 2: Demand growth sweep on archetypes
    # ================================================================
    # Sweep 324 base sensitivity keys × 25 years × 3 growth levels.
    # CCS/45Q/Geo toggles are handled by client-side repricing.
    # This tests whether the optimal mix changes under demand growth.
    print("\n--- PHASE 2: Demand growth sweep (324 keys × years × growth) ---")
    phase2_start = time.time()

    dg_output = {
        'meta': {
            'years': DEMAND_GROWTH_YEARS,
            'growth_levels': DEMAND_GROWTH_LEVELS,
            'growth_rates': DEMAND_GROWTH_RATES,
            'base_year': 2025,
            'fields': ['mix_idx', 'total_cost', 'effective_cost', 'incremental'],
            'note': 'CCS/45Q/Geo toggles use default mapping (CCS=firm, 45Q=ON, Geo=M). '
                    'Client-side repricing handles full factorial.',
        },
        'results': {},
    }

    for iso in ISOS:
        dg_output['results'][iso] = {}
        demand_twh = REGIONAL_DEMAND_TWH[iso]
        iso_rates = DEMAND_GROWTH_RATES[iso]

        for thr in OUTPUT_THRESHOLDS:
            key = (iso, thr)
            if key not in pfs or key not in archetypes:
                continue

            arrays = pfs[key]
            arch_indices = sorted(archetypes[key])
            n_arch = len(arch_indices)

            if n_arch == 0:
                continue

            arch_arrays = {k: arrays[k][arch_indices] for k in arrays}

            t_str = str(thr)
            threshold_dg = {}

            # 324 base sensitivity keys (CCS defaults to firm level, 45Q=ON, Geo=M)
            for r, f, s in product(LMH, LMH, LMH):
                for ff in LMH:
                    for tx in ['N', 'L', 'M', 'H']:
                        old_key = make_old_key(r, f, s, ff, tx)
                        sens = {
                            'ren': r, 'firm': f, 'stor': s,
                            'ccs': f, 'q45': '1',
                            'fuel': ff, 'tx': tx,
                            'geo': 'M' if iso == 'CAISO' else None,
                        }
                        wholesale = max(5, WHOLESALE_PRICES[iso] +
                                        FUEL_ADJUSTMENTS[iso][LEVEL_NAME[ff]])

                        year_results = {}
                        for year in DEMAND_GROWTH_YEARS:
                            growth_results = {}
                            for g_level in DEMAND_GROWTH_LEVELS:
                                g_rate = iso_rates[g_level]
                                tc, ec, _ = price_mix_batch(
                                    iso, arch_arrays, sens, demand_twh,
                                    target_year=year, growth_rate=g_rate
                                )
                                best_local = int(np.argmin(tc))
                                growth_results[g_level] = [
                                    arch_indices[best_local],
                                    round(float(tc[best_local]), 2),
                                    round(float(ec[best_local]), 2),
                                    round(float(ec[best_local]) - wholesale, 2),
                                ]
                            year_results[str(year)] = growth_results

                        threshold_dg[old_key] = year_results

            dg_output['results'][iso][t_str] = threshold_dg
            print(f"  {iso:>6} {thr:>5}%: {len(threshold_dg)} keys × "
                  f"{len(DEMAND_GROWTH_YEARS)} years × "
                  f"{len(DEMAND_GROWTH_LEVELS)} growth ({n_arch} archetypes)")

    phase2_elapsed = time.time() - phase2_start
    print(f"\nPhase 2 complete: {phase2_elapsed:.0f}s")

    # ================================================================
    # SAVE OUTPUTS
    # ================================================================
    print("\n--- Saving outputs ---")

    # Backward-compat results JSON
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
    # SUMMARY
    # ================================================================
    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  STEP 2 COMPLETE in {total_elapsed:.0f}s")
    print(f"{'='*70}")

    # Print Medium scenario summary
    print("\nMMM_M_M (all-Medium, 45Q=ON) summary:")
    for iso in ISOS:
        print(f"\n  {iso}:")
        for thr in OUTPUT_THRESHOLDS:
            t_str = str(thr)
            sc = output['results'][iso].get('thresholds', {}).get(t_str, {}).get('scenarios', {}).get('MMM_M_M')
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
