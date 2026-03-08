#!/usr/bin/env python3
"""
Step 9D: Compute grid-level dispatch metrics for deployment visualization.

For each strategy × ISO × participation × threshold, runs the procurement
resource mix through hourly dispatch physics to compute:
- Hourly match score (% of hours where clean supply >= demand)
- Curtailment (TWh of excess clean energy)
- Gas backup needed (GW)
- Total grid clean %

Also computes grid baseline (existing clean shares per ISO at 0% participation).

Output: Augments dashboard/js/deployment-data.js with dispatch fields.
"""

import os
import sys
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from dispatch_utils import (
    load_common_data, get_demand_profile, get_supply_profiles,
    build_supply_matrix, reconstruct_hourly_dispatch, H,
)
from pipeline_config import (
    GRID_MIX_SHARES, REGIONAL_DEMAND_TWH,
    PEAK_DEMAND_MW, EXISTING_GAS_CAPACITY_MW,
)

ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']
JS_DIR = os.path.join(ROOT_DIR, 'dashboard', 'js')
DEPLOYMENT_JS = os.path.join(JS_DIR, 'deployment-data.js')

# Resource adequacy margin (15%) for gas backup calc
RA_MARGIN = 0.15

# Approximate capacity factors for converting TWh → peak MW contribution
CAPACITY_FACTORS = {
    'solar': 0.25, 'wind': 0.35, 'offshore_wind': 0.45,
    'clean_firm': 0.90, 'nuclear_uprate': 0.90,
    'ccs': 0.85, 'ccs_ccgt': 0.85,
    'hydro': 0.40, 'battery': 0.0, 'ldes': 0.0,
    'storage': 0.0, 'geothermal': 0.90, 'green_h2': 0.0,
}
# Peak credit (fraction of nameplate counted toward RA)
PEAK_CREDITS = {
    'solar': 0.30, 'wind': 0.15, 'offshore_wind': 0.30,
    'clean_firm': 0.95, 'nuclear_uprate': 0.95,
    'ccs': 0.90, 'ccs_ccgt': 0.90,
    'hydro': 0.50, 'battery': 0.80, 'ldes': 0.80,
    'storage': 0.80, 'geothermal': 0.95, 'green_h2': 0.0,
}


def compute_grid_baseline():
    """Compute grid baseline clean shares for each ISO."""
    baseline = {}
    for iso in ISOS:
        shares = GRID_MIX_SHARES.get(iso, {})
        total_pct = sum(shares.values())
        baseline[iso] = {
            'solar': shares.get('solar', 0),
            'wind': shares.get('wind', 0),
            'offshore_wind': shares.get('offshore_wind', 0),
            'hydro': shares.get('hydro', 0),
            'clean_firm': shares.get('clean_firm', 0),
            'ccs_ccgt': shares.get('ccs_ccgt', 0),
            'totalPct': round(total_pct, 1),
        }
    return baseline


def compute_dispatch_metrics(iso, resource_pcts, demand_norm, supply_profiles,
                              supply_matrix, total_mwh):
    """Run dispatch and compute grid-level metrics for a resource mix.

    Args:
        iso: ISO name
        resource_pcts: dict {resource: % of demand} — grid baseline + procurement
        demand_norm: (8760,) normalized demand profile
        supply_profiles: dict of supply shape profiles
        supply_matrix: pre-built (N_RESOURCES, 8760) matrix
        total_mwh: annual demand in MWh

    Returns:
        dict with hms, curtTwh, gasGw, gridCleanPct
    """
    # Extract storage from resource_pcts (separate from generation resources)
    battery_pct = resource_pcts.pop('battery', 0)
    ldes_pct = resource_pcts.pop('ldes', 0)
    storage_pct = resource_pcts.pop('storage', 0)
    battery_pct += storage_pct  # storage → battery4

    result = reconstruct_hourly_dispatch(
        demand_norm=demand_norm,
        supply_profiles=supply_profiles,
        resource_pcts=resource_pcts,
        procurement_pct=100,
        battery_dispatch_pct=battery_pct,
        ldes_dispatch_pct=ldes_pct,
        supply_matrix=supply_matrix,
    )

    total_clean = result['total_clean']  # (8760,) normalized

    # Hourly match score: % of hours where clean >= demand
    demand_arr = np.array(demand_norm[:H], dtype=np.float64)
    matched_hours = np.sum(total_clean >= demand_arr * 0.999)  # 0.1% tolerance
    hms = matched_hours / H * 100

    # Curtailment
    curtailed = result.get('curtailed', np.zeros(H))
    curt_twh = float(np.sum(curtailed) * total_mwh / 1e6)  # MWh → TWh

    # Grid clean %
    total_clean_energy = float(np.sum(total_clean))
    total_demand_energy = float(np.sum(demand_arr))
    grid_clean_pct = (total_clean_energy / total_demand_energy * 100) if total_demand_energy > 0 else 0

    # Gas backup calculation
    peak_mw = PEAK_DEMAND_MW.get(iso, 50000)
    ra_peak = peak_mw * (1 + RA_MARGIN)
    avg_demand_mw = total_mwh / H

    # Clean peak contribution
    clean_peak_mw = 0
    all_pcts = dict(resource_pcts)  # generation resources
    all_pcts['battery'] = battery_pct
    all_pcts['ldes'] = ldes_pct
    for res, pct in all_pcts.items():
        if pct <= 0:
            continue
        cf = CAPACITY_FACTORS.get(res, 0.30)
        pk = PEAK_CREDITS.get(res, 0.20)
        if cf > 0:
            nameplate_mw = (pct / 100.0 * avg_demand_mw) / cf
            clean_peak_mw += nameplate_mw * pk

    gas_needed_gw = max(0, (ra_peak - clean_peak_mw) / 1000)

    return {
        'hms': round(hms, 1),
        'curtTwh': round(curt_twh, 2),
        'gasGw': round(gas_needed_gw, 1),
        'gridCleanPct': round(grid_clean_pct, 1),
    }


def build_grid_resource_pcts(iso, record):
    """Build resource_pcts dict from grid baseline + NEW procurement only.

    Existing claims (record['e']) are already part of the grid baseline
    (GRID_MIX_SHARES) — they represent accounting claims on existing capacity,
    not new physical capacity. Only new-build ('n') and cross-ISO ('x')
    resources add capacity beyond the baseline.

    The 4 pools: SSS, Corporate-Contracted, Merchant Clean are all subsets
    of existing grid clean. Only Pool 4 (New-Build) adds new capacity.
    """
    demand_twh = REGIONAL_DEMAND_TWH.get(iso, 300)

    # Start with grid baseline (existing clean as % of demand)
    pcts = dict(GRID_MIX_SHARES.get(iso, {}))

    # Only add NEW-BUILD procurement on top of grid baseline
    # record['e'] is NOT added — it's already in the grid baseline
    if record.get('n'):
        for res, twh in record['n'].items():
            if twh > 0:
                mapped = _map_resource(res)
                pcts[mapped] = pcts.get(mapped, 0) + (twh / demand_twh * 100)

    if record.get('x'):
        for src_iso, resources in record['x'].items():
            for res, twh in resources.items():
                if twh > 0:
                    mapped = _map_resource(res)
                    pcts[mapped] = pcts.get(mapped, 0) + (twh / demand_twh * 100)

    return pcts


def _map_resource(res):
    """Map deployment-data resource names to dispatch_utils resource names."""
    mapping = {
        'nuclear_uprate': 'clean_firm',
        'ccs': 'ccs_ccgt',
        'new_vre': 'wind',  # approximate
        'existing_vre': 'solar',  # approximate
        'existing_nuclear': 'clean_firm',
        'grid_clean': 'clean_firm',
        'existing_recs': 'solar',  # RECs from existing VRE
        'cross_regional_recs': 'wind',
        'new_build_vre': 'wind',
        'green_h2': 'clean_firm',  # approximate for dispatch
    }
    return mapping.get(res, res)


def main():
    print("=" * 70)
    print("Step 9D: Dispatch Integration for Deployment Visualization")
    print("=" * 70)

    # Load deployment data
    if not os.path.exists(DEPLOYMENT_JS):
        print(f"ERROR: {DEPLOYMENT_JS} not found. Run step9c first.")
        sys.exit(1)

    with open(DEPLOYMENT_JS) as f:
        content = f.read()
    # Strip JS wrapper
    json_str = content.split('const DEPLOYMENT_DATA = ', 1)[1].rsplit(';', 1)[0]
    deployment = json.loads(json_str)

    # Load dispatch data
    print("\nLoading demand/supply profiles...")
    demand_data, gen_profiles, emission_rates, fossil_mix = load_common_data()

    # Pre-compute per-ISO dispatch data
    iso_dispatch = {}
    for iso in ISOS:
        demand_norm, total_mwh = get_demand_profile(iso, demand_data)
        supply_profiles = get_supply_profiles(iso, gen_profiles)
        supply_matrix = build_supply_matrix(supply_profiles)
        iso_dispatch[iso] = {
            'demand_norm': demand_norm,
            'total_mwh': total_mwh,
            'supply_profiles': supply_profiles,
            'supply_matrix': supply_matrix,
        }
        print(f"  {iso}: demand={total_mwh/1e6:.1f} TWh")

    # Compute grid baseline
    grid_baseline = compute_grid_baseline()
    deployment['gridBaseline'] = grid_baseline
    print(f"\nGrid baselines computed:")
    for iso, bl in grid_baseline.items():
        print(f"  {iso}: {bl['totalPct']:.1f}% clean")

    # Process each strategy × ISO × participation × threshold
    total_processed = 0
    total_skipped = 0
    strategies = deployment.get('data', {})

    for strat_id, strat_data in strategies.items():
        print(f"\nProcessing {strat_id}...")
        strat_count = 0

        for iso, iso_data in strat_data.items():
            if iso not in iso_dispatch:
                continue
            dd = iso_dispatch[iso]

            for part_key, part_data in iso_data.items():
                for thr_key, record in part_data.items():
                    if not isinstance(record, dict):
                        continue

                    # Build combined grid mix
                    resource_pcts = build_grid_resource_pcts(iso, record)

                    try:
                        metrics = compute_dispatch_metrics(
                            iso, resource_pcts,
                            dd['demand_norm'], dd['supply_profiles'],
                            dd['supply_matrix'], dd['total_mwh'],
                        )
                        record.update(metrics)
                        strat_count += 1
                    except Exception as e:
                        total_skipped += 1

        total_processed += strat_count
        print(f"  {strat_id}: {strat_count} records augmented")

    print(f"\n{'=' * 70}")
    print(f"Total: {total_processed} records augmented, {total_skipped} skipped")

    # Write augmented JS file
    with open(DEPLOYMENT_JS, 'w') as f:
        f.write('// Auto-generated by step9c + step9d (dispatch-augmented)\n')
        f.write(f'// Generated: {__import__("datetime").datetime.now().isoformat()}\n')
        f.write(f'// Entries: {total_processed} (with dispatch metrics)\n\n')
        f.write('const DEPLOYMENT_DATA = ')
        json.dump(deployment, f, separators=(',', ':'))
        f.write(';\n')

    size_kb = os.path.getsize(DEPLOYMENT_JS) / 1024
    print(f"\nWrote {DEPLOYMENT_JS}")
    print(f"Size: {size_kb:.0f} KB")
    print("Done.")


if __name__ == '__main__':
    main()
