#!/usr/bin/env python3
"""One-time script to refilter existing near-miss parquets.

Applies threshold-crossing filter + 100K stratified cap to existing
near-miss files, replacing the bloated union files with compact ones.

Usage:
  python scripts/temp_refilter_near_miss.py --iso PJM       # single ISO
  python scripts/temp_refilter_near_miss.py --iso ALL        # all ISOs
  python scripts/temp_refilter_near_miss.py --iso PJM --dry  # dry run (no overwrite)
"""

import argparse
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
import step1_pfs_generator as s1
from step1c_zone_search import (
    _compute_max_storage_scores,
    _stratified_sample,
    ACTIVE_THRESHOLDS,
    MAX_NEAR_MISS,
)

ALL_ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']


def refilter_iso(iso, demand_arr, supply_matrix, dry_run=False):
    """Refilter a single ISO's near-miss parquet."""
    rtypes = s1.get_resource_types(iso)
    path = os.path.join(s1.STEP1_RAW_PFS_PARQUET_DIR,
                        f'{iso}_near_miss.parquet')

    if not os.path.exists(path):
        print(f"  {iso}: near-miss file not found — skipping")
        return

    # Load existing near-miss
    table = pq.read_table(path)
    combos = np.column_stack([table.column(rt).to_numpy() for rt in rtypes])
    scores = table.column('base_score').to_numpy()
    old_size = os.path.getsize(path) / (1024 * 1024)
    n_before = len(combos)

    print(f"  {iso}: {n_before:,} mixes ({old_size:.1f} MB)")

    # Check if max_storage_score already exists
    if 'max_storage_score' in table.column_names:
        max_storage_scores = table.column('max_storage_score').to_numpy()
        print(f"    Using existing max_storage_score column")
    else:
        print(f"    Computing max_storage_score ({n_before:,} mixes)...")
        t0 = time.time()
        max_storage_scores = _compute_max_storage_scores(
            combos, demand_arr, supply_matrix)
        elapsed = time.time() - t0
        print(f"    Max-storage screen done in {elapsed:.1f}s")

    # Threshold-crossing filter
    targets = np.array([t / 100.0 for t in ACTIVE_THRESHOLDS])
    crosses_any = np.zeros(len(combos), dtype=bool)
    for target in targets:
        crosses_any |= (scores < target) & (max_storage_scores >= target)

    combos = combos[crosses_any]
    scores = scores[crosses_any]
    max_storage_scores = max_storage_scores[crosses_any]
    n_after_cross = len(combos)
    print(f"    Threshold-crossing: {n_before:,} → {n_after_cross:,} "
          f"({n_before - n_after_cross:,} pruned, {(n_before - n_after_cross)/n_before*100:.1f}%)")

    # Per-threshold coverage report
    for th in ACTIVE_THRESHOLDS:
        target = th / 100.0
        eligible = (scores < target) & (max_storage_scores >= target)
        n = int(eligible.sum())
        if n > 0:
            print(f"      t{th}%: {n:,} eligible")

    # Stratified cap
    if len(combos) > MAX_NEAR_MISS:
        idx = _stratified_sample(combos, scores, max_storage_scores,
                                 rtypes, MAX_NEAR_MISS)
        combos = combos[idx]
        scores = scores[idx]
        print(f"    Stratified cap: {n_after_cross:,} → {len(combos):,}")

    if dry_run:
        est_mb = len(combos) * len(rtypes) * 8 / (1024 * 1024) * 0.35
        print(f"    [DRY RUN] Would save {len(combos):,} mixes (~{est_mb:.1f} MB)")
        return

    # Save (no max_storage_score column)
    data = {}
    for i, rt in enumerate(rtypes):
        data[rt] = combos[:, i].astype(np.float64)
    data['base_score'] = scores

    table_out = pa.table(data)
    pq.write_table(table_out, path, compression='snappy')
    new_size = os.path.getsize(path) / (1024 * 1024)
    print(f"    Saved: {len(combos):,} mixes → {new_size:.1f} MB "
          f"(was {old_size:.1f} MB, {(1 - new_size/old_size)*100:.0f}% reduction)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iso', default='ALL',
                        help='ISO to refilter (or ALL)')
    parser.add_argument('--dry', action='store_true',
                        help='Dry run — report sizes without overwriting')
    args = parser.parse_args()

    isos = ALL_ISOS if args.iso == 'ALL' else [args.iso]

    # Load profile data (needed for max_storage computation)
    print("Loading EIA data...")
    demand_data, gen_profiles, _, _ = s1.load_data()

    for iso in isos:
        print(f"\n{'='*60}")
        print(f"  Refiltering {iso}")
        print(f"{'='*60}")

        demand_norm = demand_data[iso]['normalized']
        supply_profiles = s1.get_supply_profiles(iso, gen_profiles)
        demand_arr, supply_matrix = s1.prepare_numpy_profiles(
            iso, demand_norm, supply_profiles)

        # JIT warmup on first ISO
        if iso == isos[0] and s1.HAS_NUMBA:
            print("  Warming up Numba JIT...")
            rtypes = s1.get_resource_types(iso)
            warmup_combos = np.array([[50] * len(rtypes), [30] * len(rtypes)])
            fracs = warmup_combos.astype(np.float64) / 100.0
            supply_warmup = fracs @ supply_matrix
            s1._batch_mixes_storage_screen(
                demand_arr, supply_warmup, 1.0, 2,
                np.array([0.5]), np.array([1.0]), np.array([5.0]),
                1, 1, 1,
                s1.BATTERY_EFFICIENCY, s1.BATTERY8_EFFICIENCY, s1.LDES_EFFICIENCY,
                s1.BATTERY_DURATION_HOURS, s1.BATTERY8_DURATION_HOURS,
                s1.LDES_DURATION_HOURS,
                s1.LDES_WINDOW_DAYS * 24, 48,
                np.array([25.0]), 1, s1.H2_EFFICIENCY,
                float(s1.H2_DURATION_HOURS), s1.H2_WINDOW_DAYS * 24)
            print("  JIT warmup done")

        refilter_iso(iso, demand_arr, supply_matrix, dry_run=args.dry)

    print("\nDone.")


if __name__ == '__main__':
    main()
