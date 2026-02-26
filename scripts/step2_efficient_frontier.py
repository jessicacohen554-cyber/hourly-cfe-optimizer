#!/usr/bin/env python3
"""
Step 2: Efficient Frontier (EF) Extraction from Physics Feasible Space (PFS)
=============================================================================
Reads the PFS parquets and applies a 2-phase reduction:

  Phase 1: Threshold gate — keep only rows whose scores fall in target ranges
  Phase 2: Global deduplication. Drop the threshold column. For each unique
           allocation (ISO/CF/Sol/Wnd/Hyd/Geo/Bat/Bat8/LDES), keep only the
           row with the highest hourly_match_score. Each unique physical
           configuration is stored ONCE — Step 3 handles threshold selection
           by filtering to mixes with score >= target threshold.

Note: No dominance removal across different resource mixes is performed.
Different resource mixes at the same storage/score can have very different
costs under different LCOE assumptions — removing them risks losing true
cost optimums. Cost-based selection happens in Step 3.

Pipeline position: Step 2 of 4
  Step 1 — PFS Generator (step1_pfs_generator.py)
  Step 2 — Efficient Frontier extraction (this file)
  Step 3 — Cost optimization (step3_cost_optimization.py)
  Step 4 — Post-processing (step4_postprocess.py)

Input:  data/step1-pfs-parquets/{ISO}_t{XX}_raw_pfs.parquet (from Step 1)
Output: data/step2-ef-parquets/step2_ef_{ISO}.parquet (per-ISO, all thresholds combined)

The output preserves all mixes that could be optimal under ANY cost assumption
at ANY threshold, ensuring no true optimum is lost during Step 3.
"""

import os
import time
import re
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PFS_DIR = os.path.join(SCRIPT_DIR, 'data')
STEP1_RAW_DIR = os.path.join(PFS_DIR, 'step1-pfs-parquets')
STEP2_EF_OUTPUT_DIR = os.path.join(PFS_DIR, 'step2-ef-parquets')

# Target thresholds — 10-40 added for Track 2/3 greenfield, 50-100 for all tracks
TARGET_THRESHOLDS = [10.0, 20.0, 30.0, 40.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 87.5, 90.0, 92.5, 95.0, 97.5, 99.0, 100.0]
TARGET_THRESHOLD_SET = set(TARGET_THRESHOLDS)

ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']

# Resource columns per ISO — CAISO has 5D (geothermal), others have 4D
RESOURCE_COLS_BASE = ['clean_firm', 'solar', 'wind', 'hydro']
RESOURCE_COLS_CAISO = ['clean_firm', 'solar', 'wind', 'hydro', 'geothermal']

# Storage dispatch columns (always present)
STORAGE_COLS = ['battery_dispatch_pct', 'battery8_dispatch_pct', 'ldes_dispatch_pct']

# Common columns in output (resource cols are ISO-dependent)
COMMON_COLS = ['iso', 'hourly_match_score', 'pareto_type']


def get_resource_cols(iso):
    """Return resource column names for the given ISO."""
    return RESOURCE_COLS_CAISO if iso == 'CAISO' else RESOURCE_COLS_BASE


def normalize_table(t, iso):
    """Cast a table to have consistent types, filling missing columns with defaults."""
    cols = {}
    resource_cols = get_resource_cols(iso)

    for name in ['iso']:
        if name in t.column_names:
            col = t.column(name)
            cols[name] = col if col.type == pa.string() else pc.cast(col, pa.string())
        else:
            raise ValueError(f"Missing required column '{name}' in {t.column_names}")

    for name in ['threshold']:
        if name in t.column_names:
            col = t.column(name)
            cols[name] = col if col.type == pa.float64() else pc.cast(col, pa.float64())
        else:
            cols[name] = pa.array(np.zeros(t.num_rows, dtype=np.float64))

    for name in resource_cols:
        if name in t.column_names:
            col = t.column(name)
            cols[name] = col if col.type == pa.int16() else pc.cast(col, pa.int16())
        elif name == 'geothermal':
            cols[name] = pa.array(np.zeros(t.num_rows, dtype=np.int16))
        else:
            raise ValueError(f"Missing required column '{name}' in {t.column_names}")

    for name in STORAGE_COLS:
        if name in t.column_names:
            col = t.column(name)
            cols[name] = col if col.type == pa.float64() else pc.cast(col, pa.float64())
        else:
            cols[name] = pa.array(np.zeros(t.num_rows, dtype=np.float64))

    for name in ['hourly_match_score']:
        if name in t.column_names:
            col = t.column(name)
            cols[name] = col if col.type == pa.float64() else pc.cast(col, pa.float64())
        else:
            raise ValueError(f"Missing required column '{name}' in {t.column_names}")

    if 'pareto_type' in t.column_names:
        col = t.column('pareto_type')
        cols['pareto_type'] = col if col.type == pa.string() else pc.cast(col, pa.string())
    else:
        cols['pareto_type'] = pa.array([''] * t.num_rows, type=pa.string())

    return pa.table(cols)


def load_iso_tables():
    """Load PFS data and return a dict of {iso: pyarrow.Table}.

    Reads from data/step1-pfs-parquets/{ISO}_t{XX}_raw_pfs.parquet.
    Groups files by ISO prefix, concatenates all threshold files per ISO.

    Returns dict keyed by ISO name with per-ISO tables (already schema-normalized).
    """
    iso_tables = {}

    if not os.path.isdir(STEP1_RAW_DIR):
        raise FileNotFoundError(
            f"Step 1 output directory not found: {STEP1_RAW_DIR}\n"
            f"Run step1_pfs_generator.py first to produce "
            f"{{ISO}}_t{{XX}}_raw_pfs.parquet files."
        )

    parquet_files = sorted(
        f for f in os.listdir(STEP1_RAW_DIR) if f.endswith('.parquet')
    )
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found in {STEP1_RAW_DIR}.\n"
            f"Run step1_pfs_generator.py first."
        )

    print(f"Scanning Step 1 raw parquets in {STEP1_RAW_DIR}")

    # Group files by ISO prefix (e.g., "ERCOT_t100_raw_pfs.parquet" -> "ERCOT")
    # Supports both canonical files ({ISO}_t{XX}_raw_pfs.parquet) and
    # NYISO batch files ({ISO}_t{XX}_raw_pfs_b{N}.parquet). Both naming
    # patterns share the same ISO prefix and are concatenated together.
    #
    # Dedup guard: if both a canonical file and batch files exist for the
    # same ISO/threshold, prefer batch files to avoid double-counting.
    files_by_iso = {}
    for fname in parquet_files:
        iso_prefix = fname.split('_')[0] if '_' in fname else None
        if iso_prefix and iso_prefix in ISOS:
            files_by_iso.setdefault(iso_prefix, []).append(fname)

    # For each ISO, detect canonical vs batch file overlap and prefer batch
    for iso, fnames in list(files_by_iso.items()):
        # Identify batch files per threshold: {ISO}_t{XX}_raw_pfs_b{N}.parquet
        batch_thresholds = set()
        for f in fnames:
            m = re.match(rf'^{re.escape(iso)}_t([\d.]+)_raw_pfs_b\d+\.parquet$', f)
            if m:
                batch_thresholds.add(m.group(1))
        # If batch files exist for a threshold, drop the canonical file for that threshold
        if batch_thresholds:
            filtered = []
            for f in fnames:
                m_canon = re.match(rf'^{re.escape(iso)}_t([\d.]+)_raw_pfs\.parquet$', f)
                if m_canon and m_canon.group(1) in batch_thresholds:
                    print(f"  Skipping canonical {f} (batch files exist for threshold {m_canon.group(1)}%)")
                    continue
                filtered.append(f)
            files_by_iso[iso] = filtered

    # Batched read: load all threshold files per ISO, concat once
    for iso, fnames in files_by_iso.items():
        iso_subtables = []
        total_rows = 0
        for fname in fnames:
            path = os.path.join(STEP1_RAW_DIR, fname)
            t = pq.read_table(path)
            if 'clean_firm' in t.column_names and 'hourly_match_score' in t.column_names:
                iso_subtables.append(t)
                total_rows += t.num_rows
        if iso_subtables:
            combined = pa.concat_tables(iso_subtables) if len(iso_subtables) > 1 else iso_subtables[0]
            iso_tables[iso] = normalize_table(combined, iso)
            print(f"  {iso}: {total_rows:>10,} rows from {len(fnames)} threshold files")

    if not iso_tables:
        raise FileNotFoundError(
            f"No valid Step 1 parquet files found in {STEP1_RAW_DIR}.\n"
            f"Files must contain 'clean_firm' and 'hourly_match_score' columns."
        )

    found = sorted(iso_tables.keys())
    missing = [iso for iso in ISOS if iso not in iso_tables]
    print(f"\nLoaded {len(found)} ISOs: {', '.join(found)}")
    if missing:
        print(f"  WARNING: Missing ISOs: {', '.join(missing)}")

    return iso_tables


def threshold_gate(table):
    """Keep only rows matching target thresholds."""
    threshold_col = table.column('threshold')
    target_set = pa.array(TARGET_THRESHOLDS, type=pa.float64())
    mask = pc.is_in(threshold_col, value_set=target_set)
    return table.filter(mask)


def deduplicate_mixes(arrays, resource_cols):
    """
    For each unique allocation (resource_cols + storage_cols), keep only
    the row with the highest hourly_match_score.

    With procurement removed, each unique physical configuration
    (CF/Sol/Wnd/Hyd[/Geo]/Bat4/Bat8/LDES) maps to a single score.
    Duplicates arise from the same mix appearing at multiple thresholds.

    Storage dispatch columns (bat, bat8, ldes) are float64 with 0.05%
    granularity from Step 1. They are scaled by 20x (0.05% -> 1) to produce
    exact integer keys.

    Returns indices into the original arrays of rows to keep.
    """
    n = len(arrays[resource_cols[0]])
    if n == 0:
        return np.array([], dtype=np.int64)

    score = arrays['hourly_match_score']

    # Scale storage dispatch to integer keys at 0.05% resolution.
    STORAGE_SCALE = 20
    bat_key = np.round(arrays['battery_dispatch_pct'] * STORAGE_SCALE).astype(np.int64)
    bat8_key = np.round(arrays['battery8_dispatch_pct'] * STORAGE_SCALE).astype(np.int64)
    ldes_key = np.round(arrays['ldes_dispatch_pct'] * STORAGE_SCALE).astype(np.int64)

    # Pack allocation into a single int64 key.
    # Resource columns are int16 0-100 -> base 101.
    # Storage keys: max ~100*20=2000, base 2001.
    STORAGE_BASE = 2001
    n_res = len(resource_cols)

    # Build group key from resource columns
    group_key = np.zeros(n, dtype=np.int64)
    for i, col in enumerate(resource_cols):
        multiplier = (101 ** (n_res - 1 - i)) * (STORAGE_BASE ** 3)
        group_key += arrays[col].astype(np.int64) * multiplier

    group_key += bat_key * (STORAGE_BASE ** 2) + bat8_key * STORAGE_BASE + ldes_key

    # Sort by (group_key ascending, score descending)
    sort_idx = np.lexsort((-score, group_key))
    sk = group_key[sort_idx]

    # Keep first row per group (highest score due to descending sort)
    is_first = np.empty(n, dtype=np.bool_)
    is_first[0] = True
    is_first[1:] = sk[1:] != sk[:-1]

    return sort_idx[is_first]


def process_iso_table(iso, table):
    """Process a single ISO's table: threshold gate + deduplication."""
    n_input = table.num_rows
    if n_input == 0:
        return None, 0, 0

    # Threshold gate
    table = threshold_gate(table)
    n_gated = table.num_rows

    if n_gated == 0:
        return None, n_input, 0

    resource_cols = get_resource_cols(iso)

    # Extract numpy arrays for deduplication
    arrays = {}
    for col in resource_cols:
        arrays[col] = table.column(col).to_numpy()
    for col in STORAGE_COLS:
        arrays[col] = table.column(col).to_numpy()
    arrays['hourly_match_score'] = table.column('hourly_match_score').to_numpy()

    dedup_idx = deduplicate_mixes(arrays, resource_cols)
    n_dedup = len(dedup_idx)

    # Build result without threshold column
    result_cols = ['iso'] + resource_cols + STORAGE_COLS + ['hourly_match_score']
    if 'pareto_type' in table.column_names:
        result_cols.append('pareto_type')

    result_arrays = {}
    for col_name in result_cols:
        if col_name in table.column_names:
            result_arrays[col_name] = table.column(col_name).take(dedup_idx)

    result = pa.table(result_arrays)
    return result, n_gated, n_dedup


def write_per_iso_outputs(results_by_iso):
    """Write per-ISO Step 2 EF outputs to data/step2-ef-parquets."""
    os.makedirs(STEP2_EF_OUTPUT_DIR, exist_ok=True)
    written = []

    for iso, table in results_by_iso.items():
        path = os.path.join(STEP2_EF_OUTPUT_DIR, f'step2_ef_{iso}.parquet')
        pq.write_table(table, path, compression='snappy')
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  Per-ISO output: {path} ({table.num_rows:,} rows, {size_mb:.1f} MB)")
        written.append(path)

    return written


def main():
    print("=" * 70)
    print("  STEP 2: EFFICIENT FRONTIER (EF) EXTRACTION")
    print("  PFS -> PFS post-EF (threshold-free, deduplicated)")
    print("=" * 70)

    total_start = time.time()

    # Load data as per-ISO tables (avoids concat-then-split overhead)
    iso_tables = load_iso_tables()

    # Process each ISO: threshold gate + deduplication
    print("\nStep 2: Threshold gate + deduplication (threshold-free)")
    print(f"  {'ISO':>6}  {'Input':>9}  {'Gated':>9}  {'Dedup':>9}  {'Time':>6}")
    print("  " + "-" * 50)

    results_by_iso = {}
    total_gated = 0
    total_dedup = 0

    for iso in ISOS:
        if iso not in iso_tables:
            continue

        t0 = time.time()
        result, n_gated, n_dedup = process_iso_table(iso, iso_tables[iso])
        elapsed = time.time() - t0

        n_input = iso_tables[iso].num_rows
        if result is not None and result.num_rows > 0:
            results_by_iso[iso] = result
            total_gated += n_gated
            total_dedup += n_dedup
            print(f"  {iso:>6}  {n_input:>8,}  {n_gated:>8,}  {n_dedup:>8,}  {elapsed:>5.1f}s")

    total_input = sum(t.num_rows for t in iso_tables.values())
    print(f"\n  Total: {total_input:,} -> {total_gated:,} (gated) -> {total_dedup:,} (dedup)")
    if total_gated > 0:
        print(f"  Reduction: {(1 - total_dedup/total_gated)*100:.1f}%")

    if not results_by_iso:
        raise RuntimeError('Step 2 produced no ISO outputs to write.')

    # Write per-ISO EF parquets
    write_per_iso_outputs(results_by_iso)

    elapsed_total = time.time() - total_start

    # Score distribution summary
    for iso in ISOS:
        if iso not in results_by_iso:
            continue
        iso_scores = results_by_iso[iso].column('hourly_match_score').to_numpy()
        if len(iso_scores) > 0:
            avail = []
            for thr in TARGET_THRESHOLDS:
                n = (iso_scores >= thr).sum()
                avail.append(f"{thr:.0f}%:{n:,}")
            print(f"  {iso} mixes per threshold: {', '.join(avail[:6])}...")

    print(f"\n  Total rows: {total_input:,} -> {total_dedup:,} (EF)")
    print(f"  Total time: {elapsed_total:.0f}s")
    print("\n" + "=" * 70)
    print("  STEP 2 COMPLETE — per-ISO EF parquets ready in data/step2-ef-parquets/")
    print("  Step 3 reads from that directory; no merged output file is written.")
    print("=" * 70)


if __name__ == '__main__':
    main()
