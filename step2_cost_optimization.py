#!/usr/bin/env python3
"""
Step 2: Cross-Evaluation Tranche Repricing
============================================
For each (region, threshold):
  1. Collect ALL physically feasible mixes from Step 1's 324 scenarios
  2. For each sensitivity combo: price EVERY feasible mix with that combo's
     cost assumptions + tranche CF pricing
  3. Select the cheapest mix → that becomes the scenario result

This is an N-mixes × N-sensitivities cross-evaluation. With tranche pricing,
mixes with less new CF become much cheaper (existing CF at wholesale vs
new CF at $105+/MWh), so the optimal mix shifts dramatically from Step 1.

Tranche model for clean firm:
  - Existing CF: priced at wholesale (already on grid)
  - New CF tranche 1: nuclear uprates (capped at 5% of existing fleet)
  - New CF tranche 2: regional new-build (geothermal CAISO, SMR elsewhere)

Input:  data/optimizer_cache.json  (LOCKED — read-only, never modified)
Output: dashboard/overprocure_results.json  (cross-evaluated repriced copy)
        data/cf_split_table.json  (uprate vs new-build breakdown)
"""

import json
import copy
import os
import time
from pathlib import Path

# ============================================================================
# TRANCHE PARAMETERS
# ============================================================================

# Nuclear uprate LCOE by firm gen sensitivity level ($/MWh)
UPRATE_LCOE = {'L': 15, 'M': 25, 'H': 40}

# Regional new-build LCOE by firm gen sensitivity level ($/MWh)
# CAISO = geothermal; all others = SMR
NEWBUILD_LCOE = {
    'L': {'CAISO': 65, 'ERCOT': 70, 'PJM': 80, 'NYISO': 85, 'NEISO': 82},
    'M': {'CAISO': 88, 'ERCOT': 95, 'PJM': 105, 'NYISO': 110, 'NEISO': 108},
    'H': {'CAISO': 125, 'ERCOT': 135, 'PJM': 160, 'NYISO': 170, 'NEISO': 165},
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
RESOURCE_TYPES = ['clean_firm', 'solar', 'wind', 'ccs_ccgt', 'hydro']
LEVEL_NAME = {'L': 'Low', 'M': 'Medium', 'H': 'High', 'N': 'None'}
LEVEL_CODE = {'Low': 'L', 'Medium': 'M', 'High': 'H', 'None': 'N'}

print("Uprate caps (TWh/yr):", UPRATE_CAP_TWH)

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

def price_mix(iso, mix_data, sens_key, demand_twh):
    """
    Price a physical mix under a given sensitivity's cost assumptions.
    Uses tranche pricing for clean firm (existing at wholesale, uprate, newbuild).

    Args:
        iso: region
        mix_data: dict with resource_mix, procurement_pct, hourly_match_score,
                  battery_dispatch_pct, ldes_dispatch_pct
        sens_key: e.g. 'MMM_M_M' — determines cost assumptions
        demand_twh: regional annual demand

    Returns:
        dict with total_cost, effective_cost, incremental, wholesale,
             tranche_costs, resource_costs_detail
    """
    # Parse sensitivity key
    parts = sens_key.split('_')
    ren_level = parts[0][0]   # R in RFS
    firm_level = parts[0][1]  # F in RFS
    stor_level = parts[0][2]  # S in RFS
    fuel_level = parts[1]     # FF
    tx_level = parts[2]       # TX

    ren_name = LEVEL_NAME[ren_level]
    firm_name = LEVEL_NAME[firm_level]
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
        existing_share = existing.get(rtype, 0)
        existing_pct = min(resource_pct_of_demand, existing_share)
        new_pct = max(0, resource_pct_of_demand - existing_share)

        # Get LCOE for this resource under this sensitivity
        if rtype == 'clean_firm':
            # TRANCHE PRICING — don't use blended LCOE
            new_cf_twh = new_pct / 100.0 * demand_twh
            existing_cf_twh = existing_pct / 100.0 * demand_twh

            # Existing CF at wholesale
            existing_cost = existing_pct / 100.0 * wholesale

            if new_cf_twh > 0:
                uprate_cap = UPRATE_CAP_TWH[iso]
                tx_add = tx_tables.get('clean_firm', {}).get(tx_name, {}).get(iso, 0)

                # Tranche 1: uprate (no tx, capped)
                uprate_twh = min(new_cf_twh, uprate_cap)
                uprate_cost_m = uprate_twh * UPRATE_LCOE[firm_level]

                # Tranche 2: newbuild (with tx, uncapped)
                newbuild_twh = max(0, new_cf_twh - uprate_twh)
                newbuild_cost_m = newbuild_twh * (NEWBUILD_LCOE[firm_level][iso] + tx_add)

                tranche_total_m = uprate_cost_m + newbuild_cost_m
                new_cf_cost_per_demand = tranche_total_m / demand_twh
                effective_new_lcoe = tranche_total_m / new_cf_twh
            else:
                uprate_twh = 0
                newbuild_twh = 0
                uprate_cost_m = 0
                newbuild_cost_m = 0
                tranche_total_m = 0
                new_cf_cost_per_demand = 0
                effective_new_lcoe = 0

            cost_per_demand = existing_cost + new_cf_cost_per_demand
            resource_costs[rtype] = {
                'existing_pct': round(existing_pct, 1),
                'new_pct': round(new_pct, 1),
                'cost_per_demand_mwh': round(cost_per_demand, 2),
                'new_cf_twh': round(new_cf_twh, 3),
                'uprate_twh': round(uprate_twh, 4),
                'newbuild_twh': round(newbuild_twh, 3),
                'uprate_price': UPRATE_LCOE[firm_level],
                'newbuild_price': NEWBUILD_LCOE[firm_level][iso] + tx_tables.get('clean_firm', {}).get(tx_name, {}).get(iso, 0),
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
        else:
            # Solar, wind, CCS: standard LCOE + tx pricing
            if rtype in ('solar', 'wind'):
                lcoe = lcoe_tables[rtype][ren_name][iso]
            elif rtype == 'ccs_ccgt':
                lcoe = lcoe_tables['ccs_ccgt'][firm_name][iso]
            else:
                lcoe = 0

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
        'newbuild_twh': cf_rc.get('newbuild_twh', 0),
        'newbuild_price': cf_rc.get('newbuild_price', 0),
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
# CROSS-EVALUATE: all mixes × all sensitivities per (region, threshold)
# ============================================================================

print("\nCross-evaluating all mixes × all sensitivities with tranche pricing...")
start_time = time.time()

cf_split_table = []
stats = {'total_evals': 0, 'mix_swaps': 0, 'scenarios_updated': 0}

for iso in ISOS:
    demand_twh = data['results'][iso]['annual_demand_mwh'] / 1e6

    for t_key, t_data in data['results'][iso]['thresholds'].items():
        scenarios = t_data['scenarios']

        # 1. Collect all unique physically feasible mixes at this threshold
        unique_mixes = {}  # mix_key → mix_data (from one representative scenario)
        for sk, sc in scenarios.items():
            rm = sc['resource_mix']
            mix_key = tuple(sorted(rm.items()))
            if mix_key not in unique_mixes:
                unique_mixes[mix_key] = {
                    'resource_mix': rm,
                    'procurement_pct': sc['procurement_pct'],
                    'hourly_match_score': sc['hourly_match_score'],
                    'battery_dispatch_pct': sc.get('battery_dispatch_pct', 0),
                    'ldes_dispatch_pct': sc.get('ldes_dispatch_pct', 0),
                    'source_key': sk,  # track which scenario this mix came from
                }

        mix_list = list(unique_mixes.values())
        n_mixes = len(mix_list)

        # 2. For each sensitivity combo, price every mix and pick cheapest
        best_key_for_thresh = None
        best_cost_for_thresh = float('inf')

        for sens_key, sens_sc in scenarios.items():
            best_mix = None
            best_cost = float('inf')
            best_result = None

            for mix_data in mix_list:
                result = price_mix(iso, mix_data, sens_key, demand_twh)
                stats['total_evals'] += 1

                if result['total_cost'] < best_cost:
                    best_cost = result['total_cost']
                    best_mix = mix_data
                    best_result = result

            # Check if the winning mix is different from the scenario's own mix
            own_mix_key = tuple(sorted(sens_sc['resource_mix'].items()))
            winning_mix_key = tuple(sorted(best_mix['resource_mix'].items()))
            if own_mix_key != winning_mix_key:
                stats['mix_swaps'] += 1

            # 3. Update this scenario with the winning mix + new costs
            # Replace resource_mix with the winning mix
            sens_sc['resource_mix'] = best_mix['resource_mix']
            sens_sc['procurement_pct'] = best_mix['procurement_pct']
            sens_sc['hourly_match_score'] = best_mix['hourly_match_score']
            sens_sc['battery_dispatch_pct'] = best_mix.get('battery_dispatch_pct', 0)
            sens_sc['ldes_dispatch_pct'] = best_mix.get('ldes_dispatch_pct', 0)
            sens_sc['mix_source'] = best_mix['source_key']

            # Copy compressed_day from the source scenario if available
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

            stats['scenarios_updated'] += 1

            # Track global optimum for this threshold
            if best_result['total_cost'] < best_cost_for_thresh:
                best_cost_for_thresh = best_result['total_cost']
                best_key_for_thresh = sens_key

            # CF split table
            tc = best_result['tranche_costs']
            if tc.get('new_cf_twh', 0) > 0:
                cf_split_table.append({
                    'iso': iso, 'threshold': t_key, 'scenario': sens_key,
                    'cf_pct': best_mix['resource_mix'].get('clean_firm', 0),
                    'new_cf_twh': tc['new_cf_twh'],
                    'uprate_twh': tc['uprate_twh'],
                    'newbuild_twh': tc['newbuild_twh'],
                    'effective_new_cf_lcoe': tc['effective_new_cf_lcoe'],
                })

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
# SAVE
# ============================================================================

data['config']['tranche_model'] = {
    'uprate_caps_twh': UPRATE_CAP_TWH,
    'uprate_lcoe': UPRATE_LCOE,
    'newbuild_lcoe': NEWBUILD_LCOE,
    'existing_nuclear_gw': EXISTING_NUCLEAR_GW,
    'method': 'cross_evaluation',
    'description': 'Cross-evaluated: all physically feasible mixes priced under each '
                   'sensitivity combo with tranche CF pricing. Cheapest mix wins.',
}

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
            print(f"  {t:>5}%: CF={cf:>2}% (of demand:{cf_demand:>5.1f}%) "
                  f"eff=${eff:.1f}/MWh match={match}% "
                  f"new_cf={new_cf:.1f}TWh src={src}")

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
