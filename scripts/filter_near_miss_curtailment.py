#!/usr/bin/env python3
"""Filter & compress PFS parquets to reduce file size and row count.

Near-miss filters (two passes):
  1. Curtailment filter — remove mixes with zero surplus hours (no hour
     where supply > demand). These can never charge storage.
  2. Bridge filter — remove mixes where base_score + surplus_pct < 0.50
     (the lowest storage threshold). Even if they have some curtailment,
     the total surplus energy is too small to bridge them to any threshold
     that step1d targets.  This is the same filter step1d applies at
     runtime; doing it upfront shrinks the parquets permanently.

Compression pass (all parquet types):
  - Downcast float64 → int8 (integer percentages 0-100) or float32 (scores)
  - Rewrite with zstd compression (replaces snappy, ~57% smaller)
  - Applied to: near_miss, coarse_cache, mixes, raw_pfs, step1d storage

Usage:
  python scripts/filter_near_miss_curtailment.py          # all ISOs
  python scripts/filter_near_miss_curtailment.py PJM MISO # specific ISOs
  python scripts/filter_near_miss_curtailment.py --compress-only  # skip row filters, just recompress
"""
import os
import sys
import time
import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

import pyarrow as pa
import pyarrow.parquet as pq
import step1_pfs_generator as s1

# Must match step1d_fine_storage.py
STORAGE_SWEEP_FLOOR = 0.50

# Compression: zstd gives ~57% smaller files than snappy with no speed penalty
PARQUET_COMPRESSION = 'zstd'


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

    # Extract combos and base scores as numpy
    combos = np.column_stack([
        table.column(rt).to_numpy() for rt in rtypes
    ])
    base_scores = table.column('base_score').to_numpy().astype(np.float64) / 100.0

    # float32 throughout — halves memory, plenty of precision for 0-100% scores
    supply_f32 = supply_matrix.astype(np.float32)
    demand_f32 = demand_arr.astype(np.float32)
    demand_total = float(demand_f32.sum())

    # Chunked: compute curtailment mask AND total surplus in one pass
    # 50K chunks → ~1.7 GB peak per chunk (float32), fits 7GB CI runner
    chunk_size = 50_000
    n_chunks = (n_before + chunk_size - 1) // chunk_size
    has_curtailment = np.empty(n_before, dtype=np.bool_)
    total_surplus = np.empty(n_before, dtype=np.float32)

    for ci, cs in enumerate(range(0, n_before, chunk_size)):
        ce = min(cs + chunk_size, n_before)
        fracs = combos[cs:ce].astype(np.float32) * np.float32(0.01)
        buf = fracs @ supply_f32              # (chunk, 8760) float32
        buf -= demand_f32                     # in-place: surplus = supply - demand
        has_curtailment[cs:ce] = np.any(buf > 0, axis=1)
        np.maximum(buf, 0.0, out=buf)        # in-place: clamp negatives
        total_surplus[cs:ce] = buf.sum(axis=1)
        if (ci + 1) % 20 == 0 or ce == n_before:
            print(f"    chunk {ci+1}/{n_chunks} ({ce:,}/{n_before:,} rows)")

    surplus_pct = total_surplus / demand_total

    # Filter 1: must have at least one surplus hour
    n_no_curtailment = int((~has_curtailment).sum())

    # Filter 2: base_score + surplus_pct must reach the lowest threshold (50%)
    # This is the bridge filter from step1d — if a mix can't even reach 50%
    # with all its surplus, no storage config will push it past any threshold.
    can_bridge = (base_scores + surplus_pct) >= STORAGE_SWEEP_FLOOR

    # Combined mask
    mask = has_curtailment & can_bridge

    n_after = int(mask.sum())
    n_removed = n_before - n_after
    n_bridge_only = int(has_curtailment.sum()) - n_after

    # Filter rows (or keep all if nothing removed)
    if n_removed > 0:
        keep_indices = np.where(mask)[0].tolist()
        filtered = table.take(keep_indices)
    else:
        filtered = table

    # Downcast to smallest sufficient dtypes:
    # - Resource columns are integers 0-100 → int8 (max 127)
    # - base_score is float → float32 (0.01 precision is plenty)
    compact_cols = {}
    for col_name in filtered.column_names:
        col = filtered.column(col_name)
        arr = col.to_numpy()
        if col_name == 'base_score':
            compact_cols[col_name] = pa.array(arr.astype(np.float32))
        else:
            # Resource percentages: int8 handles 0-100
            compact_cols[col_name] = pa.array(arr.astype(np.int8))

    compact_table = pa.table(compact_cols)
    size_before = os.path.getsize(path)
    pq.write_table(compact_table, path, compression=PARQUET_COMPRESSION)
    size_after = os.path.getsize(path)

    pct_rows = n_removed / n_before * 100 if n_before else 0
    pct_size = (1 - size_after / size_before) * 100 if size_before else 0
    print(f"  {iso}: {n_before:,} → {n_after:,} rows "
          f"({n_removed:,} removed, {pct_rows:.1f}%)")
    print(f"    zero-curtailment: {n_no_curtailment:,}, "
          f"bridge-fail: {n_bridge_only:,}")
    print(f"    file: {size_before/(1024*1024):.1f} MB → "
          f"{size_after/(1024*1024):.1f} MB "
          f"({pct_size:.0f}% smaller)")
    return n_before, n_after


# ══════════════════════════════════════════════════════════════════════════════
# DTYPE + COMPRESSION PASS (all parquet types)
# ══════════════════════════════════════════════════════════════════════════════

# Columns that hold float scores (keep as float32); everything else → int8
FLOAT_SCORE_COLS = {'base_score', 'score', 'hourly_match_score'}
# Columns that are strings (keep as-is)
STRING_COLS = {'iso'}
# Columns that hold threshold values like 87.5, 92.5 (keep as float32)
FLOAT_THRESHOLD_COLS = {'threshold'}


def _compact_dtype(col_name, arr):
    """Return a pa.array with the smallest dtype that preserves the data."""
    if col_name in STRING_COLS:
        return pa.array(arr)
    if col_name in FLOAT_SCORE_COLS or col_name in FLOAT_THRESHOLD_COLS:
        return pa.array(arr.astype(np.float32))
    # dispatch_pct columns can be 0-100 float → int8 if integer-valued
    # Resource percentages are always integers 0-100
    if np.issubdtype(arr.dtype, np.floating):
        # Check if all values are integer-representable and in int8 range
        if np.all(arr == np.round(arr)) and arr.min() >= -128 and arr.max() <= 127:
            return pa.array(arr.astype(np.int8))
        else:
            return pa.array(arr.astype(np.float32))
    if np.issubdtype(arr.dtype, np.integer):
        if arr.min() >= -128 and arr.max() <= 127:
            return pa.array(arr.astype(np.int8))
        elif arr.min() >= -32768 and arr.max() <= 32767:
            return pa.array(arr.astype(np.int16))
    return pa.array(arr)


def compress_parquets(target_isos):
    """Rewrite all PFS parquets with compact dtypes + zstd compression.

    Handles: coarse_cache, mixes, raw_pfs (step1), storage (step1d).
    Near-miss parquets are handled by filter_iso (which also compresses).
    """
    import glob

    dirs = [
        s1.STEP1_RAW_PFS_PARQUET_DIR,
        os.path.join(os.path.dirname(s1.STEP1_RAW_PFS_PARQUET_DIR),
                     'step1d-storage-parquets'),
    ]

    total_before = 0
    total_after = 0
    n_files = 0

    for d in dirs:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith('.parquet'):
                continue
            # Skip near-miss (handled separately with row filters)
            if '_near_miss' in f:
                continue
            # Check ISO prefix
            iso_prefix = f.split('_')[0]
            if iso_prefix not in target_isos:
                continue

            path = os.path.join(d, f)
            size_before = os.path.getsize(path)

            try:
                table = pq.read_table(path)
            except Exception as e:
                print(f"    WARN: can't read {f}: {e}")
                continue

            if table.num_rows == 0:
                continue

            # Check if already compressed (zstd + compact dtypes)
            meta = pq.read_metadata(path)
            already_zstd = False
            if meta.num_row_groups > 0:
                rg = meta.row_group(0)
                if rg.num_columns > 0:
                    col_meta = rg.column(0)
                    already_zstd = (str(col_meta.compression).upper()
                                    in ('ZSTD', 'ZSTANDARD'))

            # Compact dtypes
            compact_cols = {}
            for col_name in table.column_names:
                arr = table.column(col_name).to_numpy()
                compact_cols[col_name] = _compact_dtype(col_name, arr)

            compact_table = pa.table(compact_cols)
            pq.write_table(compact_table, path, compression=PARQUET_COMPRESSION)
            size_after = os.path.getsize(path)

            total_before += size_before
            total_after += size_after
            n_files += 1

            if size_before != size_after:
                pct = (1 - size_after / size_before) * 100
                print(f"    {f}: {size_before/1024:.0f} KB → "
                      f"{size_after/1024:.0f} KB ({pct:.0f}% smaller)")

    if n_files:
        pct = (1 - total_after / total_before) * 100 if total_before else 0
        print(f"\n  Compression: {n_files} files, "
              f"{total_before/(1024*1024):.1f} MB → "
              f"{total_after/(1024*1024):.1f} MB ({pct:.0f}% smaller)")
    return total_before, total_after


def main():
    compress_only = '--compress-only' in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    target_isos = [a.upper() for a in args if a.upper() in s1.ISOS]
    if not target_isos:
        target_isos = list(s1.ISOS)

    print("=" * 60)
    if compress_only:
        print("  Parquet compression pass (dtype + zstd)")
    else:
        print("  Near-miss filter + parquet compression")
    print(f"  ISOs: {target_isos}")
    print(f"  Bridge floor: {STORAGE_SWEEP_FLOOR}")
    print("=" * 60)

    t0 = time.time()

    if not compress_only:
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
        pct = total_removed / total_before * 100 if total_before else 0
        print(f"\n  Near-miss: {total_before:,} → {total_after:,} rows "
              f"({total_removed:,} removed, {pct:.1f}%)")

    print(f"\n  Compressing all PFS parquets...")
    compress_parquets(target_isos)

    print(f"\n  Done in {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
