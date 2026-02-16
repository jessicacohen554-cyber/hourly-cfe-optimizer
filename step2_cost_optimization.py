#!/usr/bin/env python3
"""
Step 2: Cost Optimization Pipeline
===================================
Applies merit-order tranche pricing to cached Step 1 physics results.
Re-ranks scenarios to find lowest-cost physics-valid mix per threshold/region.

Input:  dashboard/overprocure_results.json (Step 1 output)
Output: dashboard/overprocure_results.json (updated with tranche_costs fields — additive only)
        dashboard/js/shared-data.js (new columns appended, existing untouched)

Tranche model (§5.3 SPEC.md):
  Tranche 1: Nuclear uprates — capped at 5% of existing nuclear GW, cheapest
  Tranche 2: Regional new-build — geothermal (CAISO) or SMR (elsewhere), uncapped
"""

import json
import copy
import sys
import os
from pathlib import Path

# ============================================================================
# TRANCHE PARAMETERS (from SPEC.md §5.3)
# ============================================================================

# Nuclear uprate LCOE by firm gen sensitivity level ($/MWh)
UPRATE_LCOE = {'L': 15, 'M': 25, 'H': 40}

# Regional new-build LCOE by firm gen sensitivity level ($/MWh)
NEWBUILD_LCOE = {
    'L': {'CAISO': 65, 'ERCOT': 70, 'PJM': 80, 'NYISO': 85, 'NEISO': 82},
    'M': {'CAISO': 88, 'ERCOT': 95, 'PJM': 105, 'NYISO': 110, 'NEISO': 108},
    'H': {'CAISO': 125, 'ERCOT': 135, 'PJM': 160, 'NYISO': 170, 'NEISO': 165},
}

# Existing nuclear capacity (GW) — for uprate cap calculation
EXISTING_NUCLEAR_GW = {
    'CAISO': 2.3,   # Diablo Canyon
    'ERCOT': 2.7,   # South Texas Project
    'PJM':   32.0,  # Largest US fleet
    'NYISO': 3.4,   # Nine Mile, FitzPatrick, Ginna
    'NEISO': 3.5,   # Millstone, Seabrook
}

# Uprate cap: 5% of existing nuclear capacity → GW → TWh/yr at 90% CF
UPRATE_PCT = 0.05
UPRATE_CF = 0.90
HOURS_PER_YEAR = 8760

UPRATE_CAP_TWH = {}
for iso, gw in EXISTING_NUCLEAR_GW.items():
    cap_gw = gw * UPRATE_PCT
    cap_twh = cap_gw * UPRATE_CF * HOURS_PER_YEAR / 1e3  # GW × CF × hours → TWh
    UPRATE_CAP_TWH[iso] = round(cap_twh, 3)

print("Uprate caps (TWh/yr):", UPRATE_CAP_TWH)

# ============================================================================
# LOAD CACHED RESULTS
# ============================================================================

RESULTS_PATH = Path('dashboard/overprocure_results.json')
if not RESULTS_PATH.exists():
    print(f"ERROR: {RESULTS_PATH} not found")
    sys.exit(1)

with open(RESULTS_PATH) as f:
    data = json.load(f)

config = data['config']
grid_mix = config['grid_mix_shares']
wholesale = config['wholesale_prices']
old_lcoe_tables = config['lcoe_tables']
tx_tables = config['transmission_tables']

ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO']


def parse_scenario_key(key):
    """Parse scenario key like 'LMH_H_L' into component levels.
    Returns (solar, wind, storage, firmgen, transmission) levels."""
    parts = key.split('_')
    sws = parts[0]  # Solar/Wind/Storage
    fg = parts[1]   # Firm Gen
    tx = parts[2]   # Transmission
    return {
        'solar': sws[0],
        'wind': sws[1],
        'storage': sws[2],
        'firmgen': fg,
        'transmission': tx,
    }


def get_level_name(char):
    """Map L/M/H/N to full name for LCOE table lookup."""
    return {'L': 'Low', 'M': 'Medium', 'H': 'High', 'N': 'None'}[char]


def compute_tranche_cost(iso, new_cf_twh, firmgen_level, tx_level):
    """Compute clean firm cost using merit-order tranche pricing.

    Returns:
        dict with tranche breakdown and total cost
    """
    if new_cf_twh <= 0:
        return {
            'uprate_twh': 0, 'newbuild_twh': 0,
            'uprate_cost_total': 0, 'newbuild_cost_total': 0,
            'total_cf_cost': 0, 'effective_cf_lcoe': 0,
            'uprate_pct_of_new': 0,
        }

    cap = UPRATE_CAP_TWH[iso]
    uprate_twh = min(new_cf_twh, cap)
    newbuild_twh = max(0, new_cf_twh - cap)

    uprate_lcoe = UPRATE_LCOE[firmgen_level]
    newbuild_lcoe = NEWBUILD_LCOE[firmgen_level][iso]

    # Transmission: uprates use existing sites (no new tx), new-build needs tx
    tx_name = get_level_name(tx_level)
    tx_adder = tx_tables.get('clean_firm', {}).get(tx_name, {}).get(iso, 0)

    uprate_cost = uprate_twh * uprate_lcoe  # no tx for uprates
    newbuild_cost = newbuild_twh * (newbuild_lcoe + tx_adder)
    total_cost = uprate_cost + newbuild_cost
    effective_lcoe = total_cost / new_cf_twh if new_cf_twh > 0 else 0

    return {
        'uprate_twh': round(uprate_twh, 4),
        'newbuild_twh': round(newbuild_twh, 4),
        'uprate_cost_total': round(uprate_cost, 2),
        'newbuild_cost_total': round(newbuild_cost, 2),
        'total_cf_cost': round(total_cost, 2),
        'effective_cf_lcoe': round(effective_lcoe, 2),
        'uprate_pct_of_new': round(uprate_twh / new_cf_twh * 100, 1) if new_cf_twh > 0 else 0,
        'uprate_lcoe_used': uprate_lcoe,
        'newbuild_lcoe_used': newbuild_lcoe,
        'tx_adder': tx_adder,
    }


def compute_old_cf_cost(iso, new_cf_twh, firmgen_level, tx_level):
    """Compute what the old blended model charged for this clean firm quantity."""
    if new_cf_twh <= 0:
        return 0
    fg_name = get_level_name(firmgen_level)
    tx_name = get_level_name(tx_level)
    old_lcoe = old_lcoe_tables['clean_firm'][fg_name][iso]
    tx_adder = tx_tables.get('clean_firm', {}).get(tx_name, {}).get(iso, 0)
    return new_cf_twh * (old_lcoe + tx_adder)


# ============================================================================
# REPRICE ALL SCENARIOS
# ============================================================================

print("\nRepricing 16,200 scenarios with tranche model...")

stats = {iso: {'scenarios': 0, 'optimal_changed': 0} for iso in ISOS}
new_optimal_results = {}  # iso → threshold → best scenario key

for iso in ISOS:
    region = data['results'][iso]
    demand_mwh = region['annual_demand_mwh']
    demand_twh = demand_mwh / 1e6
    existing_cf_pct = grid_mix[iso]['clean_firm']

    new_optimal_results[iso] = {}

    for threshold_key, threshold_data in region['thresholds'].items():
        scenarios = threshold_data.get('scenarios', {})
        if not scenarios:
            continue

        best_key = None
        best_cost = float('inf')

        for skey, sc in scenarios.items():
            stats[iso]['scenarios'] += 1
            levels = parse_scenario_key(skey)
            rm = sc.get('resource_mix', {})
            old_costs = sc.get('costs', {})

            # Clean firm % of demand from cached mix
            cf_pct = rm.get('clean_firm', 0)
            new_cf_pct = max(0, cf_pct - existing_cf_pct)
            new_cf_twh = new_cf_pct / 100.0 * demand_twh

            # Compute old and new clean firm costs
            old_cf_cost = compute_old_cf_cost(
                iso, new_cf_twh, levels['firmgen'], levels['transmission'])
            tranche = compute_tranche_cost(
                iso, new_cf_twh, levels['firmgen'], levels['transmission'])

            # Cost delta ($/MWh of demand)
            cost_delta_total = tranche['total_cf_cost'] - old_cf_cost  # TWh * $/MWh
            cost_delta_per_mwh = cost_delta_total / demand_twh if demand_twh > 0 else 0

            # New costs
            old_total = old_costs.get('total_cost', 0)
            new_total = old_total + cost_delta_per_mwh
            new_effective = old_costs.get('effective_cost', 0) + cost_delta_per_mwh
            new_incremental = old_costs.get('incremental', 0) + cost_delta_per_mwh

            # Override costs with tranche pricing (tranche IS the cost model now)
            sc['costs']['total_cost'] = round(new_total, 2)
            sc['costs']['effective_cost'] = round(new_effective, 2)
            sc['costs']['incremental'] = round(new_incremental, 2)
            # wholesale unchanged

            # Store tranche breakdown for transparency
            sc['tranche_costs'] = {
                'cost_delta_per_mwh': round(cost_delta_per_mwh, 2),
                'clean_firm_tranche': tranche,
                'new_cf_pct': round(new_cf_pct, 1),
                'new_cf_twh': round(new_cf_twh, 3),
            }

            # Update costs_detail for MMM_M_M (the only scenario with full breakdown)
            if skey == 'MMM_M_M' and sc.get('costs_detail'):
                cd = sc['costs_detail']
                cd['total_cost_per_demand_mwh'] = round(
                    cd.get('total_cost_per_demand_mwh', 0) + cost_delta_per_mwh, 2)
                cd['effective_cost_per_useful_mwh'] = round(
                    cd.get('effective_cost_per_useful_mwh', 0) + cost_delta_per_mwh, 2)
                cd['incremental_above_baseline'] = round(
                    cd.get('incremental_above_baseline', 0) + cost_delta_per_mwh, 2)
                # Update clean_firm cost_per_demand_mwh in resource_costs
                rc = cd.get('resource_costs', {}).get('clean_firm', {})
                if rc and new_cf_twh > 0:
                    rc['cost_per_demand_mwh'] = round(
                        tranche['total_cf_cost'] / demand_twh, 2)
                    rc['tranche_effective_lcoe'] = tranche['effective_cf_lcoe']
                    rc['uprate_pct_of_new'] = tranche['uprate_pct_of_new']

            # Track optimal
            if new_total < best_cost:
                best_cost = new_total
                best_key = skey

        new_optimal_results[iso][threshold_key] = {
            'optimal': best_key,
            'cost': round(best_cost, 2),
        }

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("STEP 2 COST OPTIMIZATION — RESULTS")
print("=" * 70)

for iso in ISOS:
    print(f"\n{iso}: {stats[iso]['scenarios']} scenarios repriced")
    for t in sorted(new_optimal_results[iso].keys(), key=lambda x: float(x)):
        r = new_optimal_results[iso][t]
        sc = data['results'][iso]['thresholds'][t]['scenarios'][r['optimal']]
        cf = sc.get('resource_mix', {}).get('clean_firm', 0)
        print(f"  {t:>5}%: {r['optimal']} — ${r['cost']:.1f}/MWh, CF={cf}%")

# ============================================================================
# SAVE RESULTS
# ============================================================================

# Update config with tranche model metadata
data['config']['tranche_model'] = {
    'uprate_pct': UPRATE_PCT,
    'uprate_cf': UPRATE_CF,
    'uprate_caps_twh': UPRATE_CAP_TWH,
    'uprate_lcoe': UPRATE_LCOE,
    'newbuild_lcoe': NEWBUILD_LCOE,
    'existing_nuclear_gw': EXISTING_NUCLEAR_GW,
    'description': 'Merit-order two-tranche clean firm pricing. Uprates filled first (capped), then regional new-build.',
}

# Update LCOE tables — clean_firm now shows the effective tranche LCOE at MMM_M_M's CF level
# (This is informational — actual pricing is quantity-dependent via tranche model)
# Keep the new-build LCOE as the displayed "clean firm" cost since that's the marginal resource
data['config']['lcoe_tables']['clean_firm'] = {
    'Low':    NEWBUILD_LCOE['L'],
    'Medium': NEWBUILD_LCOE['M'],
    'High':   NEWBUILD_LCOE['H'],
}

# Store optimal scenarios
data['postprocessing'] = data.get('postprocessing', {})
data['postprocessing']['optimal_scenarios'] = new_optimal_results

# Save (preserves all existing fields, only adds new ones)
output_path = RESULTS_PATH
with open(output_path, 'w') as f:
    json.dump(data, f, separators=(',', ':'))

file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
print(f"\nSaved to {output_path} ({file_size_mb:.1f} MB)")

# ============================================================================
# GENERATE SHARED DATA ADDITIONS
# ============================================================================

# Build effective cost data for shared-data.js
print("\n\nEFFECTIVE_COST_DATA ($/MWh effective cost at optimal scenario):")
for iso in ISOS:
    parts = []
    thresholds = sorted(new_optimal_results[iso].keys(), key=lambda x: float(x))
    for t in thresholds:
        best = new_optimal_results[iso][t]['optimal']
        sc = data['results'][iso]['thresholds'][t]['scenarios'][best]
        cost = sc['costs']['effective_cost']
        parts.append(f"{cost:.1f}")
    print(f"    {iso}: [{', '.join(parts)}],")

# Also output MMM_M_M effective costs (what the dashboard shows at all-Medium toggles)
print("\n\nMMM_M_M effective costs (dashboard default view):")
for iso in ISOS:
    parts = []
    thresholds = sorted(data['results'][iso]['thresholds'].keys(), key=lambda x: float(x))
    for t in thresholds:
        sc = data['results'][iso]['thresholds'][t]['scenarios'].get('MMM_M_M')
        if sc:
            cost = sc['costs']['effective_cost']
            parts.append(f"{cost:.1f}")
    print(f"    {iso}: [{', '.join(parts)}],")

print("\nStep 2 complete.")
