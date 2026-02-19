#!/usr/bin/env python3
"""
Step 2: Cross-Evaluation Tranche Repricing (v2)
=================================================
For each (region, threshold):
  1. Collect ALL physically feasible mixes from Step 1 (old 324-key or PFS)
  2. For each sensitivity combo: price EVERY feasible mix with that combo's
     cost assumptions + tranche CF pricing
  3. Select the cheapest mix → that becomes the scenario result

v2 changes:
  - Separate CCS toggle (L/M/H) from Firm Gen toggle
  - 45Q credit as binary On/Off switch
  - Geothermal tranche for CAISO (capped at 5 GW / ~39 TWh/yr)
  - Nuclear new-build Low target = $70/MWh (nth-of-a-kind)
  - Merit order: existing → uprates → geothermal (CAISO) → cheapest of nuclear/CCS
  - Output: backward-compat 324-key results + cost tables for client-side repricing

Tranche model for clean firm:
  - Existing CF: priced at wholesale (already on grid)
  - Tranche 1: nuclear uprates (capped at 5% of existing fleet)
  - Tranche 2 (CAISO only): geothermal (capped at 5 GW / ~39 TWh/yr)
  - Tranche 3: cheapest of nuclear new-build vs CCS-CCGT (toggle-dependent)

Input:  data/optimizer_cache.json  (LOCKED — read-only, never modified)
Output: dashboard/overprocure_results.json  (cross-evaluated repriced copy)
        data/cf_split_table.json  (uprate vs new-build breakdown)
"""

import json
import copy
import os
import time
from pathlib import Path
from itertools import product

# ============================================================================
# COST TABLES — NEW ARCHITECTURE (Feb 2026)
# ============================================================================

# Nuclear uprate LCOE by firm gen sensitivity level ($/MWh)
UPRATE_LCOE = {'L': 15, 'M': 25, 'H': 40}

# Nuclear new-build LCOE by firm gen sensitivity level ($/MWh)
# Low = nth-of-a-kind SMR target ($70/MWh)
NUCLEAR_NEWBUILD_LCOE = {
    'L': {'CAISO': 70, 'ERCOT': 68, 'PJM': 72, 'NYISO': 75, 'NEISO': 73},
    'M': {'CAISO': 95, 'ERCOT': 90, 'PJM': 105, 'NYISO': 110, 'NEISO': 108},
    'H': {'CAISO': 140, 'ERCOT': 135, 'PJM': 160, 'NYISO': 170, 'NEISO': 165},
}

# Geothermal LCOE (CAISO only) by geothermal sensitivity level ($/MWh)
GEOTHERMAL_LCOE = {
    'L': {'CAISO': 63},
    'M': {'CAISO': 88},
    'H': {'CAISO': 110},
}

# Geothermal cap: 5 GW at 90% CF = ~39 TWh/yr (CAISO only)
GEO_CAP_TWH = 39.0

# CCS-CCGT LCOE by CCS sensitivity level ($/MWh)
# 45Q ON: $29/MWh offset baked in (current tables)
CCS_LCOE_45Q_ON = {
    'L': {'CAISO': 58, 'ERCOT': 52, 'PJM': 62, 'NYISO': 78, 'NEISO': 75},
    'M': {'CAISO': 86, 'ERCOT': 71, 'PJM': 79, 'NYISO': 99, 'NEISO': 96},
    'H': {'CAISO': 115, 'ERCOT': 92, 'PJM': 102, 'NYISO': 128, 'NEISO': 122},
}
# 45Q OFF: add back $29/MWh
CCS_LCOE_45Q_OFF = {
    'L': {'CAISO': 87, 'ERCOT': 81, 'PJM': 91, 'NYISO': 107, 'NEISO': 104},
    'M': {'CAISO': 115, 'ERCOT': 100, 'PJM': 108, 'NYISO': 128, 'NEISO': 125},
    'H': {'CAISO': 144, 'ERCOT': 121, 'PJM': 131, 'NYISO': 157, 'NEISO': 151},
}

# Uprate cap: 5% of existing nuclear × 90% CF → TWh/yr
EXISTING_NUCLEAR_GW = {
    'CAISO': 2.3, 'ERCOT': 2.7, 'PJM': 32.0, 'NYISO': 3.4, 'NEISO': 3.5
}
UPRATE_CAP_TWH = {
    iso: round(gw * 0.05 * 0.90 * 8760 / 1e3, 3)
    for iso, gw in EXISTING_NUCLEAR_GW.items()
}

ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO']

# Demand growth rates by ISO (annual, from EIA/NREL projections)
DEMAND_GROWTH_RATES = {
    'CAISO':  {'Low': 0.014, 'Medium': 0.019, 'High': 0.025},
    'ERCOT':  {'Low': 0.020, 'Medium': 0.035, 'High': 0.055},
    'PJM':    {'Low': 0.015, 'Medium': 0.024, 'High': 0.036},
    'NYISO':  {'Low': 0.013, 'Medium': 0.020, 'High': 0.044},
    'NEISO':  {'Low': 0.009, 'Medium': 0.018, 'High': 0.029},
}

# Full year range for demand growth scenarios (2026-2050)
DEMAND_GROWTH_YEARS = list(range(2026, 2051))
DEMAND_GROWTH_LEVELS = ['Low', 'Medium', 'High']
RESOURCE_TYPES = ['clean_firm', 'solar', 'wind', 'ccs_ccgt', 'hydro']
LEVEL_NAME = {'L': 'Low', 'M': 'Medium', 'H': 'High', 'N': 'None'}
LEVEL_CODE = {'Low': 'L', 'Medium': 'M', 'High': 'H', 'None': 'N'}
LMH = ['L', 'M', 'H']

print("Uprate caps (TWh/yr):", UPRATE_CAP_TWH)
print(f"Geothermal cap (CAISO): {GEO_CAP_TWH} TWh/yr")

# ============================================================================
# SENSITIVITY KEY ARCHITECTURE
# ============================================================================
# Old format: RFS_FF_TX (e.g., MMM_M_M) — 324 combos
# New format: RFSC_QFF_TX[_G] — 5,832 (non-CAISO) / 17,496 (CAISO)
#   R = Renewable L/M/H
#   F = Firm Gen (nuclear) L/M/H
#   S = Storage L/M/H
#   C = CCS L/M/H
#   Q = 45Q 1/0
#   FF = Fossil Fuel L/M/H
#   TX = Transmission N/L/M/H
#   G = Geothermal L/M/H (CAISO only)

def old_to_new_key(old_key, ccs_level='M', q45='1', geo_level=None):
    """Map old RFS_FF_TX key to new RFSC_QFF_TX[_G] format."""
    parts = old_key.split('_')
    r, f, s = parts[0][0], parts[0][1], parts[0][2]
    ff, tx = parts[1], parts[2]
    new_key = f"{r}{f}{s}{ccs_level}_{q45}{ff}_{tx}"
    if geo_level is not None:
        new_key += f"_{geo_level}"
    return new_key


def parse_new_key(new_key):
    """Parse new-format sensitivity key into component levels."""
    parts = new_key.split('_')
    rfsc = parts[0]  # 4 chars: R, F, S, C
    qff = parts[1]   # Q + FF (e.g., '1M')
    tx = parts[2]    # TX level

    result = {
        'ren_level': rfsc[0],
        'firm_level': rfsc[1],
        'stor_level': rfsc[2],
        'ccs_level': rfsc[3],
        'q45': qff[0],           # '1' or '0'
        'fuel_level': qff[1],
        'tx_level': tx,
    }

    if len(parts) > 3:
        result['geo_level'] = parts[3]
    else:
        result['geo_level'] = None

    return result


def new_to_old_key(new_key):
    """Map new-format key back to old RFS_FF_TX format (dropping CCS/45Q/geo)."""
    p = parse_new_key(new_key)
    return f"{p['ren_level']}{p['firm_level']}{p['stor_level']}_{p['fuel_level']}_{p['tx_level']}"


# ============================================================================
# LOAD CACHE (read-only) → deep copy for output
# ============================================================================

CACHE_PATH = Path('data/optimizer_cache.json')
if not CACHE_PATH.exists():
    raise FileNotFoundError(f"Locked cache not found: {CACHE_PATH}")

print(f"Reading (read-only): {CACHE_PATH}")
with open(CACHE_PATH) as f:
    cache = json.load(f)

# Work entirely on a copy — cache file is never modified
data = copy.deepcopy(cache)
config = data['config']
grid_mix = config['grid_mix_shares']
lcoe_tables = config['lcoe_tables']
tx_tables = config['transmission_tables']
wholesale_prices = config['wholesale_prices']
fuel_adjustments = config.get('fuel_prices', {})

# ============================================================================
# COST FUNCTION: price any mix under any sensitivity with tranche CF
# ============================================================================

def price_mix(iso, mix_data, new_sens_key, demand_twh, target_year=None, growth_rate=None):
    """
    Price a physical mix under a given sensitivity's cost assumptions.
    Uses tranche pricing for clean firm with merit-order dispatch:
      1. Existing CF at wholesale
      2. Nuclear uprates (capped)
      3. Geothermal (CAISO only, capped at 39 TWh/yr)
      4. Cheapest of nuclear new-build vs CCS (toggle-dependent)

    Args:
        iso: region
        mix_data: dict with resource_mix, procurement_pct, hourly_match_score,
                  battery_dispatch_pct, ldes_dispatch_pct
        new_sens_key: new-format key (e.g., 'MMMM_1M_M' or 'MMMM_1M_M_M')
        demand_twh: regional annual demand (base year 2025)
        target_year: optional target year for demand growth (e.g. 2035)
        growth_rate: optional annual demand growth rate (e.g. 0.019)

    Returns:
        dict with total_cost, effective_cost, incremental, wholesale,
             tranche_costs, resource_costs_detail
    """
    p = parse_new_key(new_sens_key)
    ren_level = p['ren_level']
    firm_level = p['firm_level']
    stor_level = p['stor_level']
    ccs_level = p['ccs_level']
    q45 = p['q45']
    fuel_level = p['fuel_level']
    tx_level = p['tx_level']
    geo_level = p['geo_level']

    # Demand growth: existing generation stays flat in absolute TWh,
    # its share of grown demand shrinks, requiring more new-build
    years = max(0, (target_year or 2025) - 2025)
    growth_factor = (1 + (growth_rate or 0)) ** years if years > 0 else 1.0
    demand_twh = demand_twh * growth_factor
    existing_scale = 1.0 / growth_factor  # scale existing shares down

    ren_name = LEVEL_NAME[ren_level]
    stor_name = LEVEL_NAME[stor_level]
    fuel_name = LEVEL_NAME[fuel_level]
    tx_name = LEVEL_NAME[tx_level]

    rm = mix_data['resource_mix']
    proc = mix_data['procurement_pct'] / 100.0
    match_frac = mix_data['hourly_match_score'] / 100.0
    bat_pct = mix_data.get('battery_dispatch_pct', 0)
    ldes_pct = mix_data.get('ldes_dispatch_pct', 0)

    existing = grid_mix[iso]
    wholesale = wholesale_prices[iso]
    fuel_adj = fuel_adjustments.get(iso, {}).get(fuel_name, 0)
    wholesale += fuel_adj
    wholesale = max(5, wholesale)

    total_cost = 0.0
    resource_costs = {}

    for rtype in RESOURCE_TYPES:
        pct = rm.get(rtype, 0)
        if pct <= 0:
            resource_costs[rtype] = {
                'existing_pct': 0, 'new_pct': 0,
                'cost_per_demand_mwh': 0,
            }
            continue

        resource_frac = proc * (pct / 100.0)
        resource_pct_of_demand = resource_frac * 100.0
        # Existing stays flat in absolute TWh; scale share for grown demand
        existing_share = existing.get(rtype, 0) * existing_scale
        existing_pct = min(resource_pct_of_demand, existing_share)
        new_pct = max(0, resource_pct_of_demand - existing_share)

        if rtype == 'clean_firm':
            # MERIT-ORDER TRANCHE PRICING for clean firm
            new_cf_twh = new_pct / 100.0 * demand_twh
            existing_cost = existing_pct / 100.0 * wholesale

            uprate_twh = 0
            geo_twh = 0
            nuclear_newbuild_twh = 0
            ccs_newbuild_twh = 0
            uprate_cost_m = 0
            geo_cost_m = 0
            nuclear_cost_m = 0
            ccs_cost_m = 0

            if new_cf_twh > 0:
                uprate_cap = UPRATE_CAP_TWH[iso]
                tx_add_cf = tx_tables.get('clean_firm', {}).get(tx_name, {}).get(iso, 0)
                tx_add_ccs = tx_tables.get('ccs_ccgt', {}).get(tx_name, {}).get(iso, 0)

                # Tranche 1: Nuclear uprates (no tx, capped)
                uprate_twh = min(new_cf_twh, uprate_cap)
                uprate_cost_m = uprate_twh * UPRATE_LCOE[firm_level]
                remaining = max(0, new_cf_twh - uprate_twh)

                # Tranche 2: Geothermal (CAISO only, capped)
                if iso == 'CAISO' and geo_level and remaining > 0:
                    geo_price = GEOTHERMAL_LCOE[geo_level]['CAISO'] + tx_add_cf
                    geo_twh = min(remaining, GEO_CAP_TWH)
                    geo_cost_m = geo_twh * geo_price
                    remaining = max(0, remaining - geo_twh)

                # Tranche 3: Cheapest of nuclear new-build vs CCS
                if remaining > 0:
                    nuclear_price = NUCLEAR_NEWBUILD_LCOE[firm_level][iso] + tx_add_cf
                    ccs_table = CCS_LCOE_45Q_ON if q45 == '1' else CCS_LCOE_45Q_OFF
                    ccs_price = ccs_table[ccs_level][iso] + tx_add_ccs

                    if nuclear_price <= ccs_price:
                        nuclear_newbuild_twh = remaining
                        nuclear_cost_m = nuclear_newbuild_twh * nuclear_price
                    else:
                        ccs_newbuild_twh = remaining
                        ccs_cost_m = ccs_newbuild_twh * ccs_price

                tranche_total_m = uprate_cost_m + geo_cost_m + nuclear_cost_m + ccs_cost_m
                new_cf_cost_per_demand = tranche_total_m / demand_twh
                effective_new_lcoe = tranche_total_m / new_cf_twh if new_cf_twh > 0 else 0
            else:
                tranche_total_m = 0
                new_cf_cost_per_demand = 0
                effective_new_lcoe = 0

            cost_per_demand = existing_cost + new_cf_cost_per_demand

            # Build tranche detail
            ccs_table_ref = CCS_LCOE_45Q_ON if q45 == '1' else CCS_LCOE_45Q_OFF
            resource_costs[rtype] = {
                'existing_pct': round(existing_pct, 1),
                'new_pct': round(new_pct, 1),
                'cost_per_demand_mwh': round(cost_per_demand, 2),
                'new_cf_twh': round(new_cf_twh, 3),
                'uprate_twh': round(uprate_twh, 4),
                'uprate_price': UPRATE_LCOE[firm_level],
                'geo_twh': round(geo_twh, 3),
                'geo_price': GEOTHERMAL_LCOE.get(geo_level or 'M', {}).get(iso, 0) + tx_tables.get('clean_firm', {}).get(tx_name, {}).get(iso, 0) if iso == 'CAISO' and geo_level else 0,
                'nuclear_newbuild_twh': round(nuclear_newbuild_twh, 3),
                'nuclear_price': NUCLEAR_NEWBUILD_LCOE[firm_level][iso] + tx_tables.get('clean_firm', {}).get(tx_name, {}).get(iso, 0),
                'ccs_newbuild_twh': round(ccs_newbuild_twh, 3),
                'ccs_price': ccs_table_ref[ccs_level][iso] + tx_tables.get('ccs_ccgt', {}).get(tx_name, {}).get(iso, 0),
                'effective_new_lcoe': round(effective_new_lcoe, 1),
            }

        elif rtype == 'hydro':
            # Hydro: always existing, wholesale-priced, $0 tx
            cost_per_demand = resource_pct_of_demand / 100.0 * wholesale
            resource_costs[rtype] = {
                'existing_pct': round(existing_pct, 1),
                'new_pct': round(new_pct, 1),
                'cost_per_demand_mwh': round(cost_per_demand, 2),
            }

        elif rtype == 'ccs_ccgt':
            # CCS as a separate resource type in the mix
            # Use CCS toggle level + 45Q switch for pricing
            ccs_table = CCS_LCOE_45Q_ON if q45 == '1' else CCS_LCOE_45Q_OFF
            lcoe = ccs_table[ccs_level][iso]
            tx_add = tx_tables.get(rtype, {}).get(tx_name, {}).get(iso, 0)
            new_build_cost = lcoe + tx_add
            cost_per_demand = (existing_pct / 100.0 * wholesale) + \
                              (new_pct / 100.0 * new_build_cost)
            resource_costs[rtype] = {
                'existing_pct': round(existing_pct, 1),
                'new_pct': round(new_pct, 1),
                'cost_per_demand_mwh': round(cost_per_demand, 2),
            }

        else:
            # Solar, wind: standard LCOE + tx pricing
            lcoe = lcoe_tables[rtype][ren_name][iso]
            tx_add = tx_tables.get(rtype, {}).get(tx_name, {}).get(iso, 0)
            new_build_cost = lcoe + tx_add
            cost_per_demand = (existing_pct / 100.0 * wholesale) + \
                              (new_pct / 100.0 * new_build_cost)
            resource_costs[rtype] = {
                'existing_pct': round(existing_pct, 1),
                'new_pct': round(new_pct, 1),
                'cost_per_demand_mwh': round(cost_per_demand, 2),
            }

        total_cost += cost_per_demand

    # Storage costs (dispatch-based, not in resource_mix)
    bat_cost_lcoe = lcoe_tables['battery'][stor_name][iso]
    bat_tx = tx_tables.get('battery', {}).get(tx_name, {}).get(iso, 0)
    bat_cost = (bat_pct / 100.0) * (bat_cost_lcoe + bat_tx)
    resource_costs['battery'] = {
        'dispatch_pct': bat_pct,
        'cost_per_demand_mwh': round(bat_cost, 2),
    }
    total_cost += bat_cost

    ldes_cost_lcoe = lcoe_tables['ldes'][stor_name][iso]
    ldes_tx = tx_tables.get('ldes', {}).get(tx_name, {}).get(iso, 0)
    ldes_cost = (ldes_pct / 100.0) * (ldes_cost_lcoe + ldes_tx)
    resource_costs['ldes'] = {
        'dispatch_pct': ldes_pct,
        'cost_per_demand_mwh': round(ldes_cost, 2),
    }
    total_cost += ldes_cost

    effective_cost = total_cost / match_frac if match_frac > 0 else 0

    # Build tranche_costs summary
    cf_rc = resource_costs.get('clean_firm', {})
    tranche_costs = {
        'new_cf_twh': cf_rc.get('new_cf_twh', 0),
        'uprate_twh': cf_rc.get('uprate_twh', 0),
        'uprate_price': cf_rc.get('uprate_price', 0),
        'geo_twh': cf_rc.get('geo_twh', 0),
        'geo_price': cf_rc.get('geo_price', 0),
        'nuclear_newbuild_twh': cf_rc.get('nuclear_newbuild_twh', 0),
        'nuclear_price': cf_rc.get('nuclear_price', 0),
        'ccs_newbuild_twh': cf_rc.get('ccs_newbuild_twh', 0),
        'ccs_price': cf_rc.get('ccs_price', 0),
        'effective_new_cf_lcoe': cf_rc.get('effective_new_lcoe', 0),
    }

    return {
        'total_cost': round(total_cost, 2),
        'effective_cost': round(effective_cost, 2),
        'incremental': round(effective_cost - wholesale, 2),
        'wholesale': wholesale,
        'tranche_costs': tranche_costs,
        'resource_costs': resource_costs,
    }


# ============================================================================
# EXTRACT UNIQUE PHYSICS MIXES (from old 324-key cache or future PFS)
# ============================================================================

def extract_unique_mixes(scenarios):
    """Extract all unique physical mixes from a set of scenarios.
    Works with both old 324-key format and future PFS format."""
    unique_mixes = {}
    for sk, sc in scenarios.items():
        rm = sc['resource_mix']
        # Include storage dispatch in mix key since it affects costs
        mix_key = (
            tuple(sorted(rm.items())),
            sc.get('procurement_pct', 100),
            sc.get('battery_dispatch_pct', 0),
            sc.get('ldes_dispatch_pct', 0),
        )
        if mix_key not in unique_mixes:
            unique_mixes[mix_key] = {
                'resource_mix': rm,
                'procurement_pct': sc['procurement_pct'],
                'hourly_match_score': sc['hourly_match_score'],
                'battery_dispatch_pct': sc.get('battery_dispatch_pct', 0),
                'ldes_dispatch_pct': sc.get('ldes_dispatch_pct', 0),
                'source_key': sk,
            }
    return list(unique_mixes.values())


# ============================================================================
# CROSS-EVALUATE: all mixes × all sensitivities per (region, threshold)
# ============================================================================

print("\nCross-evaluating all mixes × all sensitivities with tranche pricing...")
print("  New architecture: separate CCS/45Q/geothermal toggles")
start_time = time.time()

cf_split_table = []
stats = {'total_evals': 0, 'mix_swaps': 0, 'scenarios_updated': 0}

# For backward compat, we reprice the original 324 scenario keys
# using the new cost model. CCS defaults to same level as firm gen,
# 45Q defaults to ON, geo defaults to Medium (for CAISO).
# This ensures MMM_M_M and all other old keys get repriced correctly.

for iso in ISOS:
    demand_twh = data['results'][iso]['annual_demand_mwh'] / 1e6

    for t_key, t_data in data['results'][iso]['thresholds'].items():
        scenarios = t_data['scenarios']

        # 1. Collect all unique physically feasible mixes
        mix_list = extract_unique_mixes(scenarios)
        n_mixes = len(mix_list)

        # 2. For each old scenario key, reprice using new cost model
        best_key_for_thresh = None
        best_cost_for_thresh = float('inf')

        for sens_key, sens_sc in scenarios.items():
            # Map old key to new format:
            # CCS level = same as firm gen level (backward compat)
            # 45Q = ON, Geo = Medium (for CAISO)
            parts = sens_key.split('_')
            firm_lev = parts[0][1]  # F from RFS
            geo = 'M' if iso == 'CAISO' else None
            new_key = old_to_new_key(sens_key, ccs_level=firm_lev, q45='1', geo_level=geo)

            best_mix = None
            best_cost = float('inf')
            best_result = None

            for mix_data in mix_list:
                result = price_mix(iso, mix_data, new_key, demand_twh)
                stats['total_evals'] += 1

                if result['total_cost'] < best_cost:
                    best_cost = result['total_cost']
                    best_mix = mix_data
                    best_result = result

            # Check for mix swap
            own_mix_key = tuple(sorted(sens_sc['resource_mix'].items()))
            winning_mix_key = tuple(sorted(best_mix['resource_mix'].items()))
            if own_mix_key != winning_mix_key:
                stats['mix_swaps'] += 1

            # 3. Update scenario with winning mix + new costs
            sens_sc['resource_mix'] = best_mix['resource_mix']
            sens_sc['procurement_pct'] = best_mix['procurement_pct']
            sens_sc['hourly_match_score'] = best_mix['hourly_match_score']
            sens_sc['battery_dispatch_pct'] = best_mix.get('battery_dispatch_pct', 0)
            sens_sc['ldes_dispatch_pct'] = best_mix.get('ldes_dispatch_pct', 0)
            sens_sc['mix_source'] = best_mix['source_key']
            sens_sc['new_sens_key'] = new_key  # Track the new-format key

            # Copy compressed_day from source
            source_sc = scenarios.get(best_mix['source_key'])
            if source_sc and 'compressed_day' in source_sc:
                sens_sc['compressed_day'] = source_sc['compressed_day']

            # Update costs
            sens_sc['costs'] = {
                'total_cost': best_result['total_cost'],
                'effective_cost': best_result['effective_cost'],
                'incremental': best_result['incremental'],
                'wholesale': best_result['wholesale'],
            }
            sens_sc['tranche_costs'] = best_result['tranche_costs']

            # Update costs_detail
            sens_sc['costs_detail'] = {
                'total_cost_per_demand_mwh': best_result['total_cost'],
                'effective_cost_per_useful_mwh': best_result['effective_cost'],
                'incremental_above_baseline': best_result['incremental'],
                'baseline_wholesale_cost': best_result['wholesale'],
                'resource_costs': best_result['resource_costs'],
            }

            # Also compute no-45Q costs using same winning mix + tranche pricing
            no45q_key = old_to_new_key(sens_key, ccs_level=firm_lev, q45='0', geo_level=geo)
            no45q_result = price_mix(iso, best_mix, no45q_key, demand_twh)
            sens_sc['no_45q_costs'] = {
                'total_cost': no45q_result['total_cost'],
                'effective_cost': no45q_result['effective_cost'],
                'incremental': no45q_result['incremental'],
                'wholesale': no45q_result['wholesale'],
            }
            sens_sc['no_45q_tranche_costs'] = no45q_result['tranche_costs']

            stats['scenarios_updated'] += 1

            # Track global optimum
            if best_result['total_cost'] < best_cost_for_thresh:
                best_cost_for_thresh = best_result['total_cost']
                best_key_for_thresh = sens_key

            # CF split table
            tc = best_result['tranche_costs']
            if tc.get('new_cf_twh', 0) > 0:
                cf_split_table.append({
                    'iso': iso, 'threshold': t_key, 'scenario': sens_key,
                    'new_sens_key': new_key,
                    'cf_pct': best_mix['resource_mix'].get('clean_firm', 0),
                    'new_cf_twh': tc['new_cf_twh'],
                    'uprate_twh': tc['uprate_twh'],
                    'geo_twh': tc.get('geo_twh', 0),
                    'nuclear_newbuild_twh': tc.get('nuclear_newbuild_twh', 0),
                    'ccs_newbuild_twh': tc.get('ccs_newbuild_twh', 0),
                    'effective_new_cf_lcoe': tc['effective_new_cf_lcoe'],
                })

        # Store feasible mixes for client-side repricing
        t_data['feasible_mixes'] = [
            {
                'resource_mix': m['resource_mix'],
                'procurement_pct': m['procurement_pct'],
                'hourly_match_score': m['hourly_match_score'],
                'battery_dispatch_pct': m['battery_dispatch_pct'],
                'ldes_dispatch_pct': m['ldes_dispatch_pct'],
            }
            for m in mix_list
        ]

        # Mark global optimum
        if best_key_for_thresh:
            t_data['global_optimal'] = best_key_for_thresh
            t_data['global_optimal_cost'] = round(best_cost_for_thresh, 2)

        print(f"  {iso} {t_key:>5}%: {n_mixes} mixes × {len(scenarios)} sens = "
              f"{n_mixes * len(scenarios)} evals")

elapsed = time.time() - start_time
print(f"\nCross-evaluation complete in {elapsed:.1f}s")
print(f"  Total evaluations: {stats['total_evals']:,}")
print(f"  Mix swaps (scenario got a different mix): {stats['mix_swaps']:,}")

# ============================================================================
# FULL FACTORIAL DEMAND GROWTH SWEEP
# ============================================================================
# For each (ISO, threshold), evaluate all feasible mixes under the full
# sensitivity factorial (324 × 3 CCS × 2 45Q × [3 geo CAISO]) × 3 demand
# growth levels × 25 years (2026-2050). Output: compact separate JSON file.

print("\nRunning full factorial demand growth sweep (2026-2050)...")
print(f"  Years: {len(DEMAND_GROWTH_YEARS)} ({DEMAND_GROWTH_YEARS[0]}-{DEMAND_GROWTH_YEARS[-1]})")
print(f"  Growth levels: {DEMAND_GROWTH_LEVELS}")
dg_start = time.time()
dg_stats = {'total_evals': 0}

# Compact output: {iso: {threshold: {sens_key: {year: {growth: [mix_idx, tc, ec, ic]}}}}}
demand_growth_data = {
    'meta': {
        'years': DEMAND_GROWTH_YEARS,
        'growth_levels': DEMAND_GROWTH_LEVELS,
        'growth_rates': DEMAND_GROWTH_RATES,
        'fields': ['mix_idx', 'total_cost', 'effective_cost', 'incremental'],
        'base_year': 2025,
    },
    'results': {},
}

for iso in ISOS:
    demand_growth_data['results'][iso] = {}
    base_demand_twh = data['results'][iso]['annual_demand_mwh'] / 1e6
    iso_rates = DEMAND_GROWTH_RATES[iso]

    # Build full sensitivity key set for this ISO
    # CCS levels × 45Q states × [geo levels for CAISO]
    ccs_levels = LMH
    q45_states = ['1', '0']
    geo_levels = LMH if iso == 'CAISO' else [None]

    for t_key, t_data in data['results'][iso]['thresholds'].items():
        scenarios = t_data['scenarios']
        mix_list = extract_unique_mixes(scenarios)
        if not mix_list:
            continue

        threshold_results = {}

        for sens_key in scenarios:
            parts = sens_key.split('_')
            firm_lev = parts[0][1]

            # Full factorial over CCS × 45Q × geo
            for ccs_lev in ccs_levels:
                for q45 in q45_states:
                    for geo_lev in geo_levels:
                        new_key = old_to_new_key(sens_key, ccs_level=ccs_lev,
                                                  q45=q45, geo_level=geo_lev)
                        # Build compound key for storage
                        extra = f"_C{ccs_lev}_Q{q45}"
                        if geo_lev:
                            extra += f"_G{geo_lev}"
                        compound_key = sens_key + extra

                        year_results = {}
                        for year in DEMAND_GROWTH_YEARS:
                            growth_results = {}
                            for g_level in DEMAND_GROWTH_LEVELS:
                                g_rate = iso_rates[g_level]

                                best_idx = 0
                                best_cost = float('inf')
                                best_result = None

                                for idx, mx in enumerate(mix_list):
                                    r = price_mix(iso, mx, new_key, base_demand_twh,
                                                  target_year=year, growth_rate=g_rate)
                                    dg_stats['total_evals'] += 1

                                    if r['total_cost'] < best_cost:
                                        best_cost = r['total_cost']
                                        best_idx = idx
                                        best_result = r

                                # Store compact: [mix_idx, total_cost, effective_cost, incremental]
                                growth_results[g_level] = [
                                    best_idx,
                                    round(best_result['total_cost'], 2),
                                    round(best_result['effective_cost'], 2),
                                    round(best_result['incremental'], 2),
                                ]

                            year_results[str(year)] = growth_results

                        threshold_results[compound_key] = year_results

        demand_growth_data['results'][iso][str(t_key)] = threshold_results
        n_keys = len(threshold_results)
        print(f"  {iso} {t_key:>5}%: {n_keys} sensitivity combos × {len(DEMAND_GROWTH_YEARS)} years × "
              f"{len(DEMAND_GROWTH_LEVELS)} growth = {n_keys * len(DEMAND_GROWTH_YEARS) * len(DEMAND_GROWTH_LEVELS)} entries")

dg_elapsed = time.time() - dg_start
print(f"\nDemand growth sweep complete in {dg_elapsed:.1f}s")
print(f"  Total evaluations: {dg_stats['total_evals']:,}")

# Save demand growth results separately (compact JSON)
DG_RESULTS_PATH = Path('dashboard/demand_growth_results.json')
with open(DG_RESULTS_PATH, 'w') as f:
    json.dump(demand_growth_data, f, separators=(',', ':'))
dg_size_mb = os.path.getsize(DG_RESULTS_PATH) / (1024 * 1024)
print(f"  Saved {DG_RESULTS_PATH} ({dg_size_mb:.1f} MB)")

# ============================================================================
# EMBED COST TABLES FOR CLIENT-SIDE REPRICING
# ============================================================================

data['config']['tranche_model'] = {
    'uprate_caps_twh': UPRATE_CAP_TWH,
    'uprate_lcoe': UPRATE_LCOE,
    'nuclear_newbuild_lcoe': NUCLEAR_NEWBUILD_LCOE,
    'geothermal_lcoe': GEOTHERMAL_LCOE,
    'geothermal_cap_twh': GEO_CAP_TWH,
    'ccs_lcoe_45q_on': CCS_LCOE_45Q_ON,
    'ccs_lcoe_45q_off': CCS_LCOE_45Q_OFF,
    'existing_nuclear_gw': EXISTING_NUCLEAR_GW,
    'method': 'cross_evaluation_v2',
    'description': (
        'v2: Separate CCS/45Q/geothermal toggles. '
        'Merit order: existing → uprates → geothermal (CAISO) → cheapest of nuclear/CCS. '
        'Backward-compat 324-key output + feasible_mixes for client-side repricing.'
    ),
    'sensitivity_key_format': {
        'old': 'RFS_FF_TX (324 combos)',
        'new': 'RFSC_QFF_TX[_G] (5832 non-CAISO / 17496 CAISO)',
        'default_new': 'MMMM_1M_M' + ('_M' if True else ''),
    },
}

# ============================================================================
# SAVE
# ============================================================================

RESULTS_PATH = Path('dashboard/overprocure_results.json')
with open(RESULTS_PATH, 'w') as f:
    json.dump(data, f, separators=(',', ':'))

with open('data/cf_split_table.json', 'w') as f:
    json.dump(cf_split_table, f, indent=2)

file_size_mb = os.path.getsize(RESULTS_PATH) / (1024 * 1024)
print(f"\nSaved {RESULTS_PATH} ({file_size_mb:.1f} MB)")
print(f"Saved data/cf_split_table.json ({len(cf_split_table)} rows)")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("MMM_M_M (all-Medium toggles — dashboard default)")
print("New key: MMMM_1M_M (CCS=M, 45Q=ON) / MMMM_1M_M_M (CAISO + geo=M)")
print("=" * 70)
for iso in ISOS:
    print(f"\n{iso} (existing CF: {grid_mix[iso]['clean_firm']}% of demand):")
    for t in sorted(data['results'][iso]['thresholds'].keys(), key=float):
        sc = data['results'][iso]['thresholds'][t]['scenarios'].get('MMM_M_M')
        if sc:
            cf = sc['resource_mix'].get('clean_firm', 0)
            proc = sc['procurement_pct']
            cf_demand = proc / 100 * cf
            eff = sc['costs']['effective_cost']
            match = sc['hourly_match_score']
            src = sc.get('mix_source', 'self')
            tc = sc.get('tranche_costs', {})
            new_cf = tc.get('new_cf_twh', 0)
            geo = tc.get('geo_twh', 0)
            nuc = tc.get('nuclear_newbuild_twh', 0)
            ccs = tc.get('ccs_newbuild_twh', 0)
            print(f"  {t:>5}%: CF={cf:>2}% (of demand:{cf_demand:>5.1f}%) "
                  f"eff=${eff:.1f}/MWh match={match}% "
                  f"new_cf={new_cf:.1f}TWh geo={geo:.1f} nuc={nuc:.1f} ccs={ccs:.1f}")

print("\n" + "=" * 70)
print("GLOBAL OPTIMA (cheapest scenario per region+threshold)")
print("=" * 70)
for iso in ISOS:
    print(f"\n{iso}:")
    for t in sorted(data['results'][iso]['thresholds'].keys(), key=float):
        td = data['results'][iso]['thresholds'][t]
        opt = td.get('global_optimal')
        if not opt:
            continue
        cost = td['global_optimal_cost']
        sc = td['scenarios'][opt]
        cf = sc['resource_mix'].get('clean_firm', 0)
        eff = sc['costs']['effective_cost']
        print(f"  {t:>5}%: {opt:>9} — total ${cost:.1f}, eff ${eff:.1f}/MWh, CF={cf}%")

print("\nStep 2 complete.")
