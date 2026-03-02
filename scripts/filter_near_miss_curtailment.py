#!/usr/bin/env python3
"""Filter near-miss parquets: remove mixes with zero curtailment.

Reads each {ISO}_near_miss.parquet, computes hourly supply vs demand,
and removes mixes that never produce surplus (no hour with supply > demand).
These mixes cannot benefit from storage dispatch and are dead weight.

Usage:
  python scripts/filter_near_miss_curtailment.py          # all ISOs
  python scripts/filter_near_miss_curtailment.py PJM MISO # specific ISOs
"""
import os
import sys
import time
import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

import pyarrow.parquet as pq
import step1_pfs_generator as s1


def filter_iso(iso, demand_arr, supply_matrix):
    """Filter one ISO's near-miss parquet in-place. Returns (before, after) counts."""
    rtypes = s1.get_resource_types(iso)
    path = os.path.join(s1.STEP1_RAW_PFS_PARQUET_DIR, f'{iso}_near_miss.parquet')

    if not os.path.exists(path):
        print(f"  {iso}: no near-miss parquet — skipping")
        return 0, 0

    table = pq.read_table(path)
    n_before = table.num_rows
    if n_before == 0:
        print(f"  {iso}: empty parquet — skipping")
        return 0, 0

    # Extract combos as numpy
    combos = np.column_stack([
        table.column(rt).to_numpy() for rt in rtypes
    ])

    # Chunked curtailment check (same as step1c._has_curtailment_mask)
    chunk_size = 10000
    mask = np.empty(n_before, dtype=np.bool_)
    for cs in range(0, n_before, chunk_size):
        ce = min(cs + chunk_size, n_before)
        fracs = combos[cs:ce].astype(np.float64) / 100.0
        supply = fracs @ supply_matrix
        surplus = supply - demand_arr[np.newaxis, :]
        mask[cs:ce] = np.any(surplus > 0, axis=1)

    n_after = int(mask.sum())
    n_removed = n_before - n_after

    if n_removed == 0:
        print(f"  {iso}: {n_before:,} mixes, all have curtailment — no change")
        return n_before, n_after

    # Filter and overwrite
    keep_indices = np.where(mask)[0].tolist()
    filtered = table.take(keep_indices)
    pq.write_table(filtered, path, compression='snappy')

    pct = n_removed / n_before * 100
    print(f"  {iso}: {n_before:,} → {n_after:,} ({n_removed:,} removed, {pct:.1f}%)")
    return n_before, n_after


def main():
    target_isos = [a.upper() for a in sys.argv[1:] if a.upper() in s1.ISOS]
    if not target_isos:
        target_isos = list(s1.ISOS)

    print("=" * 60)
    print("  Near-miss curtailment filter")
    print(f"  ISOs: {target_isos}")
    print("=" * 60)

    t0 = time.time()
    demand_data, gen_profiles, _, _ = s1.load_data()

    total_before = 0
    total_after = 0
    for iso in target_isos:
        supply_profiles = s1.get_supply_profiles(iso, gen_profiles)
        demand_arr, supply_matrix = s1.prepare_numpy_profiles(
            iso, demand_data[iso]['normalized'], supply_profiles)
        b, a = filter_iso(iso, demand_arr, supply_matrix)
        total_before += b
        total_after += a

    total_removed = total_before - total_after
    print(f"\n  Total: {total_before:,} → {total_after:,} "
          f"({total_removed:,} removed) in {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
