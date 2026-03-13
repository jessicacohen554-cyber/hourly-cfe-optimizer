#!/usr/bin/env python3
"""
Step 7.1E: Compute grid-level dispatch metrics for deployment visualization.

For each strategy × ISO × participation × threshold, runs the procurement
resource mix through hourly dispatch physics to compute:
- Hourly match score (CFE score: fraction of demand met by clean energy hourly)
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
    compute_co2_from_dispatch, COAL_CAP_TWH, OIL_CAP_TWH,
)
from pipeline_config import (
    ISOS,
    GRID_MIX_SHARES, REGIONAL_DEMAND_TWH,
    PEAK_DEMAND_MW, EXISTING_GAS_CAPACITY_MW,
)
JS_DIR = os.path.join(ROOT_DIR, 'dashboard', 'js')
DEPLOYMENT_JS = os.path.join(JS_DIR, 'deployment-data.js')

# Resource adequacy margin (15%) for gas backup calc
RA_MARGIN = 0.15

# Approximate capacity factors for converting TWh → peak MW contribution
CAPACITY_FACTORS = {
    'solar': 0.25, 'wind': 0.35, 'offshore_wind': 0.45,
    'clean_firm': 0.90, 'nuclear_uprate': 0.90,
    'ccs': 0.85, 'ccs_ccgt': 0.85,
    'hydro': 0.40,
    'battery': 0.15,   # 4hr daily cycle: 4/24 * 0.85 eff ≈ 0.14
    'ldes': 0.10,      # 100hr, 7-day window, less frequent cycling
    'storage': 0.15,   # generic storage → battery assumption
    'geothermal': 0.90, 'green_h2': 0.0,
}
# Peak credit (fraction of nameplate counted toward RA)
PEAK_CREDITS = {
    'solar': 0.30, 'wind': 0.15, 'offshore_wind': 0.30,
    'clean_firm': 0.95, 'nuclear_uprate': 0.95,
    'ccs': 0.90, 'ccs_ccgt': 0.90,
    'hydro': 0.50, 'battery': 0.80, 'ldes': 0.80,
    'storage': 0.80, 'geothermal': 0.95, 'green_h2': 0.0,
}


def compute_grid_baseline(iso_dispatch=None, emission_rates=None):
    """Compute grid baseline clean shares + dispatch HMS + CO2 for each ISO.

    If iso_dispatch is provided, runs dispatch on baseline grid mix to get
    the true hourly matching score and baseline fossil CO2.
    """
    baseline = {}
    for iso in ISOS:
        shares = GRID_MIX_SHARES.get(iso, {})
        total_pct = sum(shares.values())
        bl = {
            'solar': shares.get('solar', 0),
            'wind': shares.get('wind', 0),
            'offshore_wind': shares.get('offshore_wind', 0),
            'hydro': shares.get('hydro', 0),
            'clean_firm': shares.get('clean_firm', 0),
            'ccs_ccgt': shares.get('ccs_ccgt', 0),
            'totalPct': round(total_pct, 1),
        }

        # Compute baseline HMS via dispatch (no procurement, just existing grid)
        if iso_dispatch and iso in iso_dispatch:
            dd = iso_dispatch[iso]
            resource_pcts = dict(shares)  # copy
            try:
                metrics = compute_dispatch_metrics(
                    iso, resource_pcts,
                    dd['demand_norm'], dd['supply_profiles'],
                    dd['supply_matrix'], dd['total_mwh'],
                )
                bl['baselineHms'] = metrics['hms']

                # Compute baseline fossil CO2 (how much CO2 the grid emits today)
                if emission_rates:
                    # Residual demand = fossil generation
                    baseline_pcts = dict(shares)
                    result = reconstruct_hourly_dispatch(
                        demand_norm=dd['demand_norm'],
                        supply_profiles=dd['supply_profiles'],
                        resource_pcts=baseline_pcts,
                        procurement_pct=100,
                        battery_dispatch_pct=0,
                        ldes_dispatch_pct=0,
                        supply_matrix=dd['supply_matrix'],
                    )
                    # Fossil CO2 = residual demand × emission rate
                    residual = result.get('residual_demand', np.zeros(H))
                    fossil_mwh = np.sum(np.maximum(0, residual)) * dd['total_mwh']
                    regional = emission_rates.get(iso, {})
                    # Use weighted fossil rate (gas dominates most grids)
                    gas_rate = regional.get('gas_co2_lb_per_mwh', 800) / 2204.62
                    coal_rate = regional.get('coal_co2_lb_per_mwh', 2000) / 2204.62
                    oil_rate = regional.get('oil_co2_lb_per_mwh', 1600) / 2204.62
                    # Simple weighted average based on typical fossil mix
                    # COAL_CAP_TWH, OIL_CAP_TWH imported from dispatch_utils at top
                    coal_cap = COAL_CAP_TWH.get(iso, 0) * 1e6
                    oil_cap = OIL_CAP_TWH.get(iso, 0) * 1e6
                    fossil_total = fossil_mwh
                    coal_mwh = min(fossil_total, coal_cap)
                    remaining = fossil_total - coal_mwh
                    oil_mwh = min(remaining, oil_cap)
                    gas_mwh = max(0, remaining - oil_mwh)
                    baseline_co2_mt = (
                        coal_mwh * coal_rate + oil_mwh * oil_rate + gas_mwh * gas_rate
                    ) / 1e6  # tons → Mt
                    bl['baselineCo2Mt'] = round(baseline_co2_mt, 1)
            except Exception as e:
                print(f"    WARNING: baseline dispatch failed for {iso}: {e}")

        baseline[iso] = bl
    return baseline


def compute_dispatch_metrics(iso, resource_pcts, demand_norm, supply_profiles,
                              supply_matrix, total_mwh, emission_rates=None):
    """Run dispatch and compute grid-level metrics for a resource mix.

    Args:
        iso: ISO name
        resource_pcts: dict {resource: % of demand} — grid baseline + procurement
        demand_norm: (8760,) normalized demand profile
        supply_profiles: dict of supply shape profiles
        supply_matrix: pre-built (N_RESOURCES, 8760) matrix
        total_mwh: annual demand in MWh
        emission_rates: optional eGRID emission rates dict for fossil CO2 calc

    Returns:
        dict with hms, curtTwh, gasGw, gridCleanPct, and optionally fossilCo2Mt
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
    demand_arr = np.array(demand_norm[:H], dtype=np.float64)

    # Grid-level clean % (physical reality — always includes baseline)
    total_clean_energy = float(np.sum(total_clean))
    total_demand_energy = float(np.sum(demand_arr))
    grid_clean_pct = (total_clean_energy / total_demand_energy * 100) if total_demand_energy > 0 else 0

    # Curtailment (physical — grid level)
    curtailed = result.get('curtailed', np.zeros(H))
    curt_twh = float(np.sum(curtailed) * total_mwh / 1e6)  # MWh → TWh

    # === Grid-level CFE score (hourly matching) ===
    # CFE = sum(min(clean_h, demand_h)) / sum(demand_h)
    # Shows what fraction of demand is physically met by clean energy
    # when accounting for hourly supply/demand mismatch.
    # Includes grid baseline + any new procurement from the strategy.
    hourly_matched = np.minimum(total_clean, demand_arr)
    total_demand_sum = float(np.sum(demand_arr))
    hms = float(np.sum(hourly_matched) / total_demand_sum * 100) if total_demand_sum > 0 else 0

    # Gas backup calculation
    peak_mw = PEAK_DEMAND_MW.get(iso, 50000)
    ra_peak = peak_mw * (1 + RA_MARGIN)
    avg_demand_mw = total_mwh / H

    # Clean peak contribution (generation + storage)
    clean_peak_mw = 0
    all_pcts = dict(resource_pcts)
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

    out = {
        'hms': round(hms, 1),
        'curtTwh': round(curt_twh, 2),
        'gasGw': round(gas_needed_gw, 1),
        'gridCleanPct': round(grid_clean_pct, 1),
    }

    # Compute fossil CO2 from residual demand (actual grid emissions)
    if emission_rates:
        residual = result.get('residual_demand', np.zeros(H))
        fossil_mwh_total = float(np.sum(np.maximum(0, residual))) * total_mwh
        regional = emission_rates.get(iso, {})
        coal_rate = regional.get('coal_co2_lb_per_mwh', 2000) / 2204.62
        oil_rate = regional.get('oil_co2_lb_per_mwh', 1600) / 2204.62
        gas_rate = regional.get('gas_co2_lb_per_mwh', 800) / 2204.62
        coal_cap = COAL_CAP_TWH.get(iso, 0) * 1e6
        oil_cap = OIL_CAP_TWH.get(iso, 0) * 1e6
        coal_mwh = min(fossil_mwh_total, coal_cap)
        remaining_mwh = fossil_mwh_total - coal_mwh
        oil_mwh = min(remaining_mwh, oil_cap)
        gas_mwh = max(0, remaining_mwh - oil_mwh)
        fossil_co2_mt = (coal_mwh * coal_rate + oil_mwh * oil_rate + gas_mwh * gas_rate) / 1e6
        out['fossilCo2Mt'] = round(fossil_co2_mt, 1)

    return out


def build_grid_resource_pcts(iso, record):
    """Build resource_pcts dict from grid baseline + NEW same-ISO procurement only.

    Existing claims (record['e']) are already part of the grid baseline
    (GRID_MIX_SHARES) — they represent accounting claims on existing capacity,
    not new physical capacity. Only new-build ('n') adds physical capacity.

    Cross-ISO resources (record['x']) are excluded — they are accounting
    claims (RECs/contracts) not physical capacity on the buyer's grid.
    Grid physics (dispatch, peak credit, gas backup) reflect only what's
    physically built on each ISO's grid.
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

    # Cross-ISO resources (record['x']) are NOT added to grid physics.
    # They are accounting claims (RECs/contracts) — not physical capacity
    # on the buyer ISO's grid. Only resources physically built on the grid
    # affect dispatch, peak credit, and gas backup requirements.

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
    print("Step 7.1E: Dispatch Integration for Deployment Visualization")
    print("=" * 70)

    # Load deployment data
    if not os.path.exists(DEPLOYMENT_JS):
        print(f"ERROR: {DEPLOYMENT_JS} not found. Run step7_1b first.")
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

    # Compute grid baseline (with dispatch HMS + CO2)
    grid_baseline = compute_grid_baseline(iso_dispatch, emission_rates)
    deployment['gridBaseline'] = grid_baseline
    print(f"\nGrid baselines computed:")
    for iso, bl in grid_baseline.items():
        hms_str = f", HMS={bl['baselineHms']}%" if 'baselineHms' in bl else ''
        co2_str = f", CO2={bl.get('baselineCo2Mt', '?')} Mt" if 'baselineCo2Mt' in bl else ''
        print(f"  {iso}: {bl['totalPct']:.1f}% clean{hms_str}{co2_str}")

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

                    # Build combined grid mix (grid baseline + new-build)
                    resource_pcts = build_grid_resource_pcts(iso, record)

                    try:
                        metrics = compute_dispatch_metrics(
                            iso, resource_pcts,
                            dd['demand_norm'], dd['supply_profiles'],
                            dd['supply_matrix'], dd['total_mwh'],
                            emission_rates=emission_rates,
                        )
                        # Compute dispatch-based CO2 reduction (baseline - with-procurement)
                        bl_co2 = grid_baseline.get(iso, {}).get('baselineCo2Mt')
                        if bl_co2 is not None and 'fossilCo2Mt' in metrics:
                            metrics['dispCo2'] = round(bl_co2 - metrics['fossilCo2Mt'], 1)
                        record.update(metrics)
                        strat_count += 1
                    except Exception as e:
                        total_skipped += 1

        total_processed += strat_count
        print(f"  {strat_id}: {strat_count} records augmented")

    print(f"\n{'=' * 70}")
    print(f"Total: {total_processed} records augmented, {total_skipped} skipped")

    # ══════════════════════════════════════════════════════════════════════
    # PASS 2: Grid-centric HMS — aggregate all buyers' cross-ISO flows
    # into each destination grid and compute the combined grid HMS.
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("Pass 2: Grid-centric HMS aggregation")
    print("=" * 70)

    grid_hms_count = 0
    for strat_id, strat_data in strategies.items():
        print(f"\n  Grid HMS for {strat_id}...")

        # Collect all (participation, threshold) combinations present in data
        pt_combos = set()
        for iso_data in strat_data.values():
            if not isinstance(iso_data, dict):
                continue
            for pk, pk_data in iso_data.items():
                if not isinstance(pk_data, dict):
                    continue
                for tk in pk_data.keys():
                    pt_combos.add((pk, tk))

        for pk, tk in sorted(pt_combos):
            # For each grid ISO, aggregate new-build from ALL buyers
            for grid_iso in ISOS:
                if grid_iso not in iso_dispatch:
                    continue

                # Start with grid baseline
                agg_pcts = dict(GRID_MIX_SHARES.get(grid_iso, {}))
                grid_demand_twh = REGIONAL_DEMAND_TWH.get(grid_iso, 300)

                # Collect new-build deployed TO this grid from all buyer records
                for buyer_iso in ISOS:
                    buyer_data = strat_data.get(buyer_iso, {})
                    if not isinstance(buyer_data, dict):
                        continue
                    pk_data = buyer_data.get(pk, {})
                    if not isinstance(pk_data, dict):
                        continue
                    record = pk_data.get(tk)
                    if not record or not isinstance(record, dict):
                        continue

                    # Same-ISO new-build (buyer == grid)
                    if buyer_iso == grid_iso and record.get('n'):
                        for res, twh in record['n'].items():
                            if isinstance(twh, (int, float)) and twh > 0:
                                mapped = _map_resource(res)
                                agg_pcts[mapped] = agg_pcts.get(mapped, 0) + (twh / grid_demand_twh * 100)

                    # Cross-ISO flows where source == this grid
                    if record.get('x'):
                        for src_iso, resources in record['x'].items():
                            if src_iso != grid_iso:
                                continue
                            for res, twh in resources.items():
                                if isinstance(twh, (int, float)) and twh > 0:
                                    mapped = _map_resource(res)
                                    agg_pcts[mapped] = agg_pcts.get(mapped, 0) + (twh / grid_demand_twh * 100)

                # If no new capacity beyond baseline, use baseline HMS
                baseline_total = sum(GRID_MIX_SHARES.get(grid_iso, {}).values())
                agg_total = sum(agg_pcts.values())
                if agg_total <= baseline_total + 0.1:
                    grid_record = strat_data.get(grid_iso, {}).get(pk, {}).get(tk)
                    if grid_record and isinstance(grid_record, dict):
                        bl = grid_baseline.get(grid_iso, {})
                        grid_record.setdefault('gridHms', bl.get('baselineHms', 0))
                        grid_record.setdefault('gridAggCleanPct', bl.get('baselineCleanPct', 0))
                    continue

                try:
                    dd = iso_dispatch[grid_iso]
                    metrics = compute_dispatch_metrics(
                        grid_iso, dict(agg_pcts),
                        dd['demand_norm'], dd['supply_profiles'],
                        dd['supply_matrix'], dd['total_mwh'],
                        emission_rates=emission_rates,
                    )

                    # Store grid-centric HMS in the grid ISO's record
                    grid_record = strat_data.get(grid_iso, {}).get(pk, {}).get(tk)
                    if grid_record and isinstance(grid_record, dict):
                        grid_record['gridHms'] = metrics['hms']
                        grid_record['gridCurtTwh'] = metrics['curtTwh']
                        grid_record['gridGasGw'] = metrics['gasGw']
                        grid_record['gridAggCleanPct'] = metrics['gridCleanPct']
                        # Grid-centric dispatch CO2 reduction
                        bl_co2 = grid_baseline.get(grid_iso, {}).get('baselineCo2Mt')
                        if bl_co2 is not None and 'fossilCo2Mt' in metrics:
                            grid_record['gridDispCo2'] = round(bl_co2 - metrics['fossilCo2Mt'], 1)
                        grid_hms_count += 1
                except Exception:
                    pass

    print(f"\n  Grid HMS: {grid_hms_count} records computed")

    # Write augmented JS file
    with open(DEPLOYMENT_JS, 'w') as f:
        f.write('// Auto-generated by step7_1b + step7_1e (dispatch-augmented)\n')
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
