#!/usr/bin/env python3
"""
Step 2: CF Tranche Repricing
=============================
Only reprices clean firm using two-step tranche model.
Everything else stays exactly as Step 1 computed it.

Input:  data/optimizer_cache.json  (LOCKED — read-only, never modified)
Output: dashboard/overprocure_results.json  (repriced copy)
        data/cf_split_table.json  (uprate vs new-build breakdown for WYN panel)

Tranche model:
  Step A: Nuclear uprates — capped at 5% of existing fleet, cheapest $/MWh
  Step B: Regional new-build — geothermal (CAISO) or SMR (elsewhere), uncapped
"""

import json
import copy
import os
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
LEVEL_NAME = {'L': 'Low', 'M': 'Medium', 'H': 'High', 'N': 'None'}

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
old_cf_lcoes = config['lcoe_tables']['clean_firm']
tx_tables = config['transmission_tables']

# ============================================================================
# REPRICE: only clean firm costs change (delta approach)
# ============================================================================

print("\nRepricing clean firm across all scenarios...")

cf_split_table = []  # For WYN panel
stats = {'total': 0, 'cf_repriced': 0}

for iso in ISOS:
    demand_twh = data['results'][iso]['annual_demand_mwh'] / 1e6
    existing_cf_pct = grid_mix[iso]['clean_firm']
    uprate_cap = UPRATE_CAP_TWH[iso]

    for t_key, t_data in data['results'][iso]['thresholds'].items():
        best_key = None
        best_cost = float('inf')

        for sk, sc in t_data['scenarios'].items():
            stats['total'] += 1

            # Parse scenario key: RFS_FF_TX
            parts = sk.split('_')
            firm_level = parts[0][1]   # F in RFS
            tx_level = parts[2]        # TX

            rm = sc.get('resource_mix', {})
            cf_pct = rm.get('clean_firm', 0)
            proc = sc.get('procurement_pct', 100) / 100
            match = sc.get('hourly_match_score', 0) / 100
            old_total = sc['costs']['total_cost']
            old_wholesale = sc['costs']['wholesale']

            # 3-tranche CF cost build
            # Tranche 1: Existing CF at wholesale (already in Step 1's old_total)
            # Tranche 2: Nuclear uprate (capped, cheapest)
            # Tranche 3: New-build — geothermal/SMR (uncapped, remainder)
            cf_pct_of_demand = proc * (cf_pct / 100) * 100
            existing_cf_used = min(cf_pct_of_demand, existing_cf_pct)
            new_cf_pct = max(0, cf_pct_of_demand - existing_cf_pct)
            new_cf_twh = new_cf_pct / 100 * demand_twh
            existing_cf_twh = existing_cf_used / 100 * demand_twh

            firm_name = LEVEL_NAME[firm_level]
            tx_name = LEVEL_NAME[tx_level]
            tx_add = tx_tables.get('clean_firm', {}).get(tx_name, {}).get(iso, 0)

            if new_cf_twh > 0:
                stats['cf_repriced'] += 1

                # What Step 1 charged for new CF ($/MWh-demand basis)
                old_cf_lcoe = old_cf_lcoes[firm_name][iso]
                old_new_cf_per_demand = (new_cf_pct / 100) * (old_cf_lcoe + tx_add)

                # Tranche 1: Uprate (no tx — existing sites, capped)
                uprate_twh = min(new_cf_twh, uprate_cap)
                uprate_cost_m = uprate_twh * UPRATE_LCOE[firm_level]

                # Tranche 2: New-build (with tx, uncapped)
                newbuild_twh = max(0, new_cf_twh - uprate_twh)
                newbuild_cost_m = newbuild_twh * (NEWBUILD_LCOE[firm_level][iso] + tx_add)

                tranche_total = uprate_cost_m + newbuild_cost_m
                new_cf_per_demand = tranche_total / demand_twh

                # Delta: existing CF at wholesale is unchanged (stays in old_total)
                delta = new_cf_per_demand - old_new_cf_per_demand
                new_total = old_total + delta

                effective_new_cf_lcoe = tranche_total / new_cf_twh

                # Store 2-tranche split (existing CF at wholesale is implicit in base cost)
                sc['tranche_costs'] = {
                    'new_cf_twh': round(new_cf_twh, 4),
                    'uprate_twh': round(uprate_twh, 4),
                    'uprate_cost_m': round(uprate_cost_m, 2),
                    'uprate_price': UPRATE_LCOE[firm_level],
                    'newbuild_twh': round(newbuild_twh, 4),
                    'newbuild_cost_m': round(newbuild_cost_m, 2),
                    'newbuild_price': NEWBUILD_LCOE[firm_level][iso] + tx_add,
                    'effective_new_cf_lcoe': round(effective_new_cf_lcoe, 2),
                    'delta_per_mwh': round(delta, 2),
                }

                # CF split table row (for WYN panel)
                cf_split_table.append({
                    'iso': iso, 'threshold': t_key, 'scenario': sk,
                    'cf_pct': cf_pct, 'new_cf_twh': round(new_cf_twh, 3),
                    'uprate_twh': round(uprate_twh, 4),
                    'newbuild_twh': round(newbuild_twh, 4),
                    'effective_new_cf_lcoe': round(effective_new_cf_lcoe, 2),
                })
            else:
                new_total = old_total

            # Update costs
            new_effective = new_total / match if match > 0 else 0
            sc['costs'] = {
                'total_cost': round(new_total, 2),
                'effective_cost': round(new_effective, 2),
                'incremental': round(new_effective - old_wholesale, 2),
                'wholesale': old_wholesale,
            }

            # Update costs_detail to match tranche-repriced values
            cd = sc.get('costs_detail')
            if cd and new_cf_twh > 0:
                # Update clean_firm cost in resource_costs breakdown
                rc = cd.get('resource_costs', {})
                cf_rc = rc.get('clean_firm')
                if cf_rc:
                    # Existing CF stays at wholesale; new CF uses tranche blend
                    existing_cf_cost = existing_cf_twh * old_wholesale
                    new_cf_cost = tranche_total  # uprate + newbuild
                    total_cf_cost = existing_cf_cost + new_cf_cost
                    cf_rc['cost_per_demand_mwh'] = round(total_cf_cost / demand_twh, 2)

                # Recompute totals from all resource costs
                total_resource_cost = 0
                for res_name, res_data in rc.items():
                    c = res_data.get('cost_per_demand_mwh', res_data.get('cost', 0))
                    total_resource_cost += c
                cd['total_cost_per_demand_mwh'] = round(total_resource_cost, 2)
                cd['effective_cost_per_useful_mwh'] = round(total_resource_cost / match, 2) if match > 0 else 0
                cd['incremental_above_baseline'] = round((total_resource_cost / match if match > 0 else 0) - old_wholesale, 2)

            # Track cheapest for this region+threshold
            if new_total < best_cost:
                best_cost = new_total
                best_key = sk

        # Mark global optimum for this region+threshold
        if best_key is not None:
            t_data['global_optimal'] = best_key
            t_data['global_optimal_cost'] = round(best_cost, 2)

# ============================================================================
# SAVE
# ============================================================================

# Store tranche model metadata
data['config']['tranche_model'] = {
    'uprate_caps_twh': UPRATE_CAP_TWH,
    'uprate_lcoe': UPRATE_LCOE,
    'newbuild_lcoe': NEWBUILD_LCOE,
    'existing_nuclear_gw': EXISTING_NUCLEAR_GW,
    'description': 'Two-step CF tranche: uprate (capped at 5% existing nuclear) then regional new-build.',
}

RESULTS_PATH = Path('dashboard/overprocure_results.json')
with open(RESULTS_PATH, 'w') as f:
    json.dump(data, f, separators=(',', ':'))

with open('data/cf_split_table.json', 'w') as f:
    json.dump(cf_split_table, f, indent=2)

file_size_mb = os.path.getsize(RESULTS_PATH) / (1024 * 1024)
print(f"\nSaved {RESULTS_PATH} ({file_size_mb:.1f} MB)")
print(f"Saved data/cf_split_table.json ({len(cf_split_table)} rows)")
print(f"Total scenarios: {stats['total']}, CF repriced: {stats['cf_repriced']}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("GLOBAL OPTIMA (cheapest scenario per region+threshold)")
print("=" * 70)

for iso in ISOS:
    print(f"\n{iso} (demand: {data['results'][iso]['annual_demand_mwh']/1e6:.1f} TWh, "
          f"uprate cap: {UPRATE_CAP_TWH[iso]} TWh):")
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

# Also show MMM_M_M for comparison
print("\n" + "=" * 70)
print("MMM_M_M (all-Medium toggles — dashboard default)")
print("=" * 70)
for iso in ISOS:
    parts = []
    for t in sorted(data['results'][iso]['thresholds'].keys(), key=float):
        sc = data['results'][iso]['thresholds'][t]['scenarios'].get('MMM_M_M')
        if sc:
            parts.append(f"${sc['costs']['effective_cost']:.1f}")
    print(f"  {iso}: [{', '.join(parts)}]")

print("\nStep 2 complete.")
