#!/usr/bin/env python3
"""
Step 2.5: Expand Efficient Frontier with floor-compatible mixes from PFS.

Problem: Scenario A's sequential floor ratchet requires mixes that satisfy ALL
per-resource floors simultaneously. The standard EF (~500 cost-optimal mixes per
threshold) often lacks this diversity — especially for ISOs like PJM where
existing nuclear is high and the floor ratchet accumulates aggressively.

Solution: Simulate Scenario A's floor progression, identify the floor constraints
at each threshold, then extract PFS mixes satisfying those floors and append them
to the EF parquets. This way step5_scenario_a_consequential.py never needs PFS fallback.

Inputs:
  - data/step1-pfs-parquets/{ISO}_t{threshold}_raw_pfs.parquet (PFS mixes)
  - data/step3-cost-opt-parquets/step3_feasible_{ISO}.parquet (current EF)
  - dashboard/js/shared-data.js (current EF, alternative source)

Output:
  - data/step3-cost-opt-parquets/step3_feasible_{ISO}.parquet (augmented)
  - data/step5-post-processing/ef_expansion_log.json (audit trail)

Does NOT require a Step 1 re-run — reads existing PFS parquets.
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from scenario_common import (
    ISOS, RESOURCES, THRESHOLDS, SCENARIO_A,
    HYDRO_CAP_TWH, GRID_MIX_SHARES, BASE_DEMAND_TWH,
    SBTI_YEAR_MAP, DEMAND_GROWTH_RATES, LEVEL_NAME,
    get_demand_growth_factor,
    compute_mix_cost, _mix_resource_twh,
    _load_pfs_mixes, _filter_mixes_by_floor,
    parse_feasible_mixes, _load_feasible_from_parquet,
)


# How many floor-compatible PFS mixes to add per threshold
TARGET_FLOOR_MIXES = 200

# Maximum PFS mixes to scan per threshold (memory guard)
MAX_PFS_SCAN = 500_000


def simulate_floor_progression(feasible_mixes, iso, scenario_toggles):
    """Simulate Scenario A's floor ratchet to predict floor constraints per threshold.

    Returns dict: threshold → floor_twh dict
    """
    base_demand = BASE_DEMAND_TWH[iso]
    existing = GRID_MIX_SHARES[iso]

    # Starting floor = existing resource levels in absolute TWh
    floor_twh = {
        'clean_firm': existing.get('clean_firm', 0) / 100.0 * base_demand,
        'solar': existing.get('solar', 0) / 100.0 * base_demand,
        'wind': existing.get('wind', 0) / 100.0 * base_demand,
        'ccs_ccgt': existing.get('ccs_ccgt', 0) / 100.0 * base_demand,
        'hydro': existing.get('hydro', 0) / 100.0 * base_demand,
        'battery': 0, 'ldes': 0,
    }

    sens = dict(scenario_toggles)
    if iso != 'CAISO':
        sens['geo'] = None

    floors_by_threshold = {}
    scenario_thresholds = [t for t in THRESHOLDS if t >= 50]

    for t in scenario_thresholds:
        t_str = str(int(t)) if t == int(t) else str(t)
        gf = get_demand_growth_factor(iso, t)
        demand_twh = base_demand * gf

        # Record the floor BEFORE this threshold
        floors_by_threshold[t] = dict(floor_twh)

        # Find cheapest EF mix that passes floor
        ef_mixes = feasible_mixes.get(iso, {}).get(t_str, [])
        filtered = _filter_mixes_by_floor(ef_mixes, floor_twh, demand_twh, iso)

        if not filtered:
            # Floor blocks all EF mixes — this is where we need expansion
            continue

        # Pick cheapest (using compute_mix_cost)
        best_cost = float('inf')
        best_mix = None
        for mix in filtered:
            result = compute_mix_cost(mix, sens, iso, demand_twh, growth_factor=gf)
            if result['effective_cost'] < best_cost:
                best_cost = result['effective_cost']
                best_mix = mix

        if best_mix:
            deployed = _mix_resource_twh(best_mix, demand_twh, iso)
            for res in floor_twh:
                floor_twh[res] = max(floor_twh[res], deployed.get(res, 0))

    return floors_by_threshold


def find_floor_compatible_pfs(iso, threshold, floor_twh, demand_twh,
                               max_scan=MAX_PFS_SCAN, target_count=TARGET_FLOOR_MIXES):
    """Find PFS mixes that satisfy the floor constraints for a given threshold.

    Returns list of mix vectors in v5.0 format.
    """
    pfs_mixes = _load_pfs_mixes(iso, threshold)
    if not pfs_mixes:
        return []

    # Cap scan for memory
    if len(pfs_mixes) > max_scan:
        # Random subsample
        rng = np.random.default_rng(42)
        indices = rng.choice(len(pfs_mixes), max_scan, replace=False)
        pfs_mixes = [pfs_mixes[i] for i in indices]

    # Filter by floor
    filtered = _filter_mixes_by_floor(pfs_mixes, floor_twh, demand_twh, iso)

    if not filtered:
        return []

    # If too many, rank by diversity (spread in resource space)
    if len(filtered) > target_count:
        # Score by total resource percentage (diverse spread)
        # Take a mix of low-cost-proxy and high-diversity mixes
        scored = []
        for mix in filtered:
            total_pct = mix[0] + mix[1] + mix[2] + mix[3]
            scored.append((total_pct, mix))
        scored.sort(key=lambda x: x[0])

        # Take bottom half (cheapest) and top half (most diverse)
        half = target_count // 2
        selected = [m for _, m in scored[:half]]
        selected += [m for _, m in scored[-half:]]
        return selected[:target_count]

    return filtered


def expand_ef_for_iso(iso, feasible_mixes):
    """Expand the EF for a single ISO with floor-compatible PFS mixes.

    Returns dict: threshold_str → list of new mixes added
    """
    print(f"\n{'='*60}")
    print(f"  {iso}: Simulating Scenario A floor progression...")

    floors = simulate_floor_progression(feasible_mixes, iso, SCENARIO_A['toggles'])

    base_demand = BASE_DEMAND_TWH[iso]
    additions = {}
    total_added = 0

    for t in THRESHOLDS:
        if t < 50:
            continue

        t_str = str(int(t)) if t == int(t) else str(t)
        gf = get_demand_growth_factor(iso, t)
        demand_twh = base_demand * gf

        floor_twh = floors.get(t, {})
        if not floor_twh:
            continue

        # Check how many existing EF mixes pass the floor
        ef_mixes = feasible_mixes.get(iso, {}).get(t_str, [])
        ef_passing = _filter_mixes_by_floor(ef_mixes, floor_twh, demand_twh, iso)

        if len(ef_passing) >= 50:
            # Enough EF mixes pass — no expansion needed
            continue

        # Find floor-compatible PFS mixes
        pfs_compatible = find_floor_compatible_pfs(
            iso, t, floor_twh, demand_twh)

        if not pfs_compatible:
            print(f"  {iso} {t}%: {len(ef_passing)} EF pass floor, "
                  f"NO PFS mixes found either")
            continue

        # Deduplicate against existing EF mixes
        existing_sigs = set()
        for mix in ef_mixes:
            sig = (round(mix[0], 1), round(mix[1], 1),
                   round(mix[2], 1), round(mix[3], 1))
            existing_sigs.add(sig)

        new_mixes = []
        for mix in pfs_compatible:
            sig = (round(mix[0], 1), round(mix[1], 1),
                   round(mix[2], 1), round(mix[3], 1))
            if sig not in existing_sigs:
                new_mixes.append(mix)
                existing_sigs.add(sig)

        if new_mixes:
            additions[t_str] = new_mixes
            total_added += len(new_mixes)
            print(f"  {iso} {t}%: {len(ef_passing)} EF pass floor → "
                  f"adding {len(new_mixes)} PFS mixes "
                  f"(from {len(pfs_compatible)} candidates)")

    print(f"  {iso}: Total new mixes added: {total_added}")
    return additions


def augment_ef_parquet(iso, additions):
    """Append new mixes to the step3 feasible parquet for this ISO."""
    parquet_path = f'data/step3-cost-opt-parquets/step3_feasible_{iso}.parquet'

    if not os.path.exists(parquet_path):
        print(f"  WARNING: {parquet_path} not found, cannot augment")
        return 0

    df = pd.read_parquet(parquet_path)
    original_count = len(df)

    new_rows = []
    for t_str, mixes in additions.items():
        t_val = float(t_str)
        for mix in mixes:
            row = {
                'threshold': t_val,
                'clean_firm': mix[0],
                'solar': mix[1],
                'wind': mix[2],
                'ccs_ccgt': mix[3],
                'hydro': mix[4],
                'hourly_match_score': mix[5],
                'battery_dispatch_pct': mix[6],
                'battery8_dispatch_pct': mix[7],
                'ldes_dispatch_pct': mix[8],
                'h2_dispatch_pct': mix[9] if len(mix) > 9 else 0,
                'source': 'pfs_floor_expansion',
            }
            new_rows.append(row)

    if not new_rows:
        return 0

    new_df = pd.DataFrame(new_rows)
    # Add 'source' column to original if missing
    if 'source' not in df.columns:
        df['source'] = 'ef_original'

    combined = pd.concat([df, new_df], ignore_index=True)
    combined.to_parquet(parquet_path, index=False)

    added = len(combined) - original_count
    print(f"  {iso}: Augmented {parquet_path}: {original_count} → {len(combined)} "
          f"(+{added} mixes)")
    return added


def main():
    parser = argparse.ArgumentParser(
        description='Step 2.5: Expand EF with floor-compatible PFS mixes')
    parser.add_argument('--iso', type=str, default='ALL',
                        help='ISO to process (or ALL)')
    parser.add_argument('--target-mixes', type=int, default=TARGET_FLOOR_MIXES,
                        help=f'Target floor mixes per threshold (default: {TARGET_FLOOR_MIXES})')
    parser.add_argument('--dry-run', action='store_true',
                        help='Simulate only, do not modify EF parquets')
    args = parser.parse_args()

    isos_to_run = ISOS if args.iso == 'ALL' else [args.iso]

    print("=" * 60)
    print("STEP 2.5: EXPAND EF WITH FLOOR-COMPATIBLE PFS MIXES")
    print(f"  ISOs: {', '.join(isos_to_run)}")
    print(f"  Target mixes per threshold: {args.target_mixes}")
    print(f"  Dry run: {args.dry_run}")
    print("=" * 60)

    # Load current EF mixes
    print("\nLoading current EF mixes...")
    feasible_mixes = parse_feasible_mixes()

    log = {'isos': {}, 'total_added': 0}

    for iso in isos_to_run:
        additions = expand_ef_for_iso(iso, feasible_mixes)

        if additions and not args.dry_run:
            added = augment_ef_parquet(iso, additions)
            log['total_added'] += added
            log['isos'][iso] = {
                'thresholds_expanded': list(additions.keys()),
                'mixes_added': {t: len(m) for t, m in additions.items()},
                'total_added': added,
            }
        elif additions:
            total = sum(len(m) for m in additions.values())
            log['isos'][iso] = {
                'thresholds_expanded': list(additions.keys()),
                'mixes_added': {t: len(m) for t, m in additions.items()},
                'total_would_add': total,
                'dry_run': True,
            }

    # Save log
    log_path = 'data/step5-post-processing/ef_expansion_log.json'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"\nExpansion log: {log_path}")

    if args.dry_run:
        print("\nDRY RUN — no parquets were modified.")
        print("Run without --dry-run to apply changes.")
    else:
        print(f"\nTotal mixes added across all ISOs: {log['total_added']}")
        print("\nIMPORTANT: If you modified parquets, re-run step6_generate_shared_data.py")
        print("to update dashboard/js/shared-data.js with the expanded EF.")

    print("\nDone.")


if __name__ == '__main__':
    main()
