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

import argparse
import gc
import os
import time
import re
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PFS_DIR = os.path.join(SCRIPT_DIR, 'data')
STEP1_RAW_DIR = os.path.join(PFS_DIR, 'step1-pfs-parquets')
STEP1D_DIR = os.path.join(PFS_DIR, 'step1d-storage-parquets')
STEP2_EF_OUTPUT_DIR = os.path.join(PFS_DIR, 'step2-ef-parquets')

# Target thresholds — 10-40 added for Track 2/3 greenfield, 50-100 for all tracks
# Must match Step 1 THRESHOLDS and Step 3 OUTPUT_THRESHOLDS
TARGET_THRESHOLDS = [10.0, 20.0, 30.0, 40.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 87.5, 90.0, 92.5, 95.0, 97.5, 99.0, 99.5, 99.9, 99.99]
TARGET_THRESHOLD_SET = set(TARGET_THRESHOLDS)

ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']

# Resource columns per ISO — CAISO has 5D (geothermal), others have 4D
RESOURCE_COLS_BASE = ['clean_firm', 'solar', 'wind', 'hydro']
RESOURCE_COLS_CAISO = ['clean_firm', 'solar', 'wind', 'hydro', 'geothermal']

# Storage dispatch columns (always present)
STORAGE_COLS = ['battery_dispatch_pct', 'battery8_dispatch_pct', 'ldes_dispatch_pct', 'h2_dispatch_pct']

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


def scan_iso_files(target_isos=None):
    """Scan Step 1 parquet directory and return {iso: [filenames]} mapping.

    Groups files by ISO prefix, handles canonical vs batch file dedup.
    Does NOT load any data — just discovers and validates file paths.

    Args:
        target_isos: List of ISO names to scan for. None = all ISOs.

    Returns dict keyed by ISO name with lists of filenames to process.
    """
    iso_filter = set(target_isos) if target_isos else set(ISOS)

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
        if iso_prefix and iso_prefix in iso_filter:
            files_by_iso.setdefault(iso_prefix, []).append(fname)

    # For each ISO, detect canonical vs batch file overlap and prefer batch
    for iso, fnames in list(files_by_iso.items()):
        batch_thresholds = set()
        for f in fnames:
            m = re.match(rf'^{re.escape(iso)}_t([\d.]+)_raw_pfs_b\d+\.parquet$', f)
            if m:
                batch_thresholds.add(m.group(1))
        if batch_thresholds:
            filtered = []
            for f in fnames:
                m_canon = re.match(rf'^{re.escape(iso)}_t([\d.]+)_raw_pfs\.parquet$', f)
                if m_canon and m_canon.group(1) in batch_thresholds:
                    print(f"  Skipping canonical {f} (batch files exist for threshold {m_canon.group(1)}%)")
                    continue
                filtered.append(f)
            files_by_iso[iso] = filtered

    # Also scan Step 1D storage refinement parquets
    if os.path.isdir(STEP1D_DIR):
        step1d_files = sorted(
            f for f in os.listdir(STEP1D_DIR) if f.endswith('.parquet')
        )
        if step1d_files:
            n_1d = 0
            for fname in step1d_files:
                iso_prefix = fname.split('_')[0] if '_' in fname else None
                if iso_prefix and iso_prefix in iso_filter:
                    # Tag with directory prefix so load_and_process_iso knows
                    # to read from STEP1D_DIR instead of STEP1_RAW_DIR
                    files_by_iso.setdefault(iso_prefix, []).append(f'1d/{fname}')
                    n_1d += 1
            if n_1d > 0:
                print(f"  Found {n_1d} Step 1D storage-refined parquets")

    missing = [iso for iso in iso_filter if iso not in files_by_iso]
    if missing:
        print(f"  WARNING: Missing ISOs: {', '.join(missing)}")

    return files_by_iso


# Maximum accumulated rows before triggering an incremental dedup pass.
# Keeps peak memory under ~2GB during dedup (groupby on ~5M rows).
INCREMENTAL_DEDUP_THRESHOLD = 5_000_000


def load_and_process_iso(iso, fnames):
    """Load, gate, and deduplicate a single ISO's threshold files incrementally.

    Streams parquet files row-group by row-group (~1M rows each) instead of
    loading entire files at once. This bounds peak memory to ~1-2GB even for
    ISOs with 23M-row files (e.g., SPP_t99.5), preventing OOM kills on
    GitHub Actions runners (7GB RAM).

    Dedup ("keep max score per unique mix") is associative — deduplicating
    partial results then merging more rows and deduplicating again produces
    the same result as a single global dedup.

    Incremental dedup triggers on NEW rows added since last pass (not total
    accumulator size), preventing redundant dedup passes when small 1D
    storage-refined parquets are appended to an already-deduped accumulator.

    Args:
        iso: ISO name (e.g., 'PJM')
        fnames: List of parquet filenames to process

    Returns:
        (result_table, n_input, n_gated, n_dedup) or (None, n_input, 0, 0)
    """
    resource_cols = get_resource_cols(iso)
    result_col_names = ['iso'] + resource_cols + STORAGE_COLS + ['hourly_match_score', 'pareto_type']

    accumulated = None
    total_input = 0
    total_gated = 0
    n_incremental_dedups = 0
    rows_since_last_dedup = 0  # Track new rows added since last dedup pass

    for fname in fnames:
        # Step 1D files are prefixed with '1d/' to indicate the directory
        if fname.startswith('1d/'):
            path = os.path.join(STEP1D_DIR, fname[3:])
        else:
            path = os.path.join(STEP1_RAW_DIR, fname)

        pf = pq.ParquetFile(path)
        file_meta = pf.metadata
        file_rows = file_meta.num_rows
        total_input += file_rows

        # Check schema once (applies to all row groups in the file)
        schema_names = set(pf.schema_arrow.names)
        if 'clean_firm' not in schema_names or 'hourly_match_score' not in schema_names:
            del pf
            continue

        # Stream row groups to bound peak memory. Large PFS files
        # (SPP_t99.5: 23M rows) require ~4GB to load + dedup at once,
        # which OOMs GitHub Actions runners (7GB RAM). Row groups are
        # ~1M rows each, keeping peak memory under 2GB.
        for rg_idx in range(file_meta.num_row_groups):
            rg = pf.read_row_group(rg_idx)

            rg = normalize_table(rg, iso)
            rg = threshold_gate(rg)
            total_gated += rg.num_rows

            if rg.num_rows == 0:
                del rg
                continue

            # Drop threshold column — not needed after gating
            keep_cols = [c for c in result_col_names if c in rg.column_names]
            rg = rg.select(keep_cols)

            # Accumulate
            if accumulated is None:
                accumulated = rg
            else:
                accumulated = pa.concat_tables([accumulated, rg], promote_options='permissive')
            rows_since_last_dedup += rg.num_rows
            del rg

            # Incremental dedup when enough NEW rows have been added since
            # last pass. Using rows_since_last_dedup (not total accumulator
            # size) prevents redundant dedup passes when small files (e.g.,
            # 1D storage-refined parquets at 200K-1.2M rows) are appended
            # to an already-deduped accumulator of 36M+ rows.
            if rows_since_last_dedup > INCREMENTAL_DEDUP_THRESHOLD:
                pre = accumulated.num_rows
                arrays = {}
                for col in resource_cols + STORAGE_COLS + ['hourly_match_score']:
                    arrays[col] = accumulated.column(col).to_numpy()
                keep_idx = deduplicate_mixes(arrays, resource_cols)
                accumulated = accumulated.take(keep_idx)
                del arrays
                gc.collect()
                n_incremental_dedups += 1
                rows_since_last_dedup = 0
                print(f"    Incremental dedup: {pre:,} -> {accumulated.num_rows:,}", flush=True)

        del pf
        gc.collect()

    if accumulated is None or accumulated.num_rows == 0:
        return None, total_input, total_gated, 0

    # Final dedup pass
    arrays = {}
    for col in resource_cols + STORAGE_COLS + ['hourly_match_score']:
        arrays[col] = accumulated.column(col).to_numpy()
    keep_idx = deduplicate_mixes(arrays, resource_cols)
    result = accumulated.take(keep_idx)
    n_dedup = result.num_rows
    del accumulated, arrays
    gc.collect()

    if n_incremental_dedups > 0:
        print(f"    ({n_incremental_dedups} incremental dedup passes used)", flush=True)

    return result, total_input, total_gated, n_dedup


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
    (CF/Sol/Wnd/Hyd[/Geo]/Bat4/Bat8/LDES/H2) maps to a single score.
    Duplicates arise from the same mix appearing at multiple thresholds.

    Storage dispatch columns (bat, bat8, ldes, h2) are float64 with 0.05%
    granularity from Step 1. They are scaled by 20x (0.05% -> 1) to produce
    exact integer keys.

    Uses pandas hash-based groupby (O(n) average) instead of np.lexsort
    (O(n log n)) for ~3x less peak memory and faster execution on large
    datasets (18M+ rows). The previous lexsort approach created sorted
    copies of all 9 key columns (~2.8 GB for 18M rows), causing OOM on
    GitHub Actions runners (7 GB RAM).

    Returns indices into the original arrays of rows to keep.
    """
    n = len(arrays[resource_cols[0]])
    if n == 0:
        return np.array([], dtype=np.int64)

    # Scale storage dispatch to integer keys at 0.05% resolution.
    # Use int32 (not int64) — max resource ~200, max storage key ~2000,
    # both fit in int32 and halve memory vs int64.
    STORAGE_SCALE = 20

    data = {}
    for col in resource_cols:
        data[col] = arrays[col].astype(np.int32)
    data['_bat'] = np.round(arrays['battery_dispatch_pct'] * STORAGE_SCALE).astype(np.int32)
    data['_bat8'] = np.round(arrays['battery8_dispatch_pct'] * STORAGE_SCALE).astype(np.int32)
    data['_ldes'] = np.round(arrays['ldes_dispatch_pct'] * STORAGE_SCALE).astype(np.int32)
    data['_h2'] = np.round(arrays['h2_dispatch_pct'] * STORAGE_SCALE).astype(np.int32)
    data['_score'] = arrays['hourly_match_score']

    df = pd.DataFrame(data)
    del data

    group_cols = list(resource_cols) + ['_bat', '_bat8', '_ldes', '_h2']

    # Hash-based groupby: O(n) average, returns index of max-score row per group
    keep_idx = df.groupby(group_cols, sort=False)['_score'].idxmax().values
    del df

    return keep_idx.astype(np.int64)



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


def parse_args():
    parser = argparse.ArgumentParser(description='Step 2: Efficient Frontier extraction')
    parser.add_argument('--iso', type=str, default=None,
                        help='Single ISO to process (e.g. CAISO). Default: all ISOs.')
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine which ISOs to process
    if args.iso:
        iso_arg = args.iso.upper()
        if iso_arg not in ISOS:
            raise ValueError(f"Unknown ISO '{iso_arg}'. Must be one of {ISOS}")
        target_isos = [iso_arg]
    else:
        target_isos = ISOS

    print("=" * 70, flush=True)
    print("  STEP 2: EFFICIENT FRONTIER (EF) EXTRACTION")
    print(f"  ISOs: {', '.join(target_isos)}")
    print("  PFS -> PFS post-EF (threshold-free, deduplicated)")
    print("=" * 70, flush=True)

    total_start = time.time()

    # Scan files (no data loaded yet)
    print(f"\nScanning Step 1 raw parquets in {STEP1_RAW_DIR} + {STEP1D_DIR}")
    files_by_iso = scan_iso_files(target_isos)

    found = sorted(files_by_iso.keys())
    print(f"Found {len(found)} ISOs: {', '.join(found)}")

    # Process each ISO incrementally: load → gate → dedup per file
    # This streams data instead of loading all rows at once, keeping
    # peak memory bounded even for large ISOs (PJM: 63M+ rows).
    print("\nStep 2: Threshold gate + deduplication (threshold-free)")
    print(f"  {'ISO':>6}  {'Input':>9}  {'Gated':>9}  {'Dedup':>9}  {'Time':>6}")
    print("  " + "-" * 50)

    results_by_iso = {}
    total_input = 0
    total_gated = 0
    total_dedup = 0

    for iso in target_isos:
        if iso not in files_by_iso:
            continue

        t0 = time.time()
        result, n_input, n_gated, n_dedup = load_and_process_iso(iso, files_by_iso[iso])
        elapsed = time.time() - t0

        total_input += n_input
        if result is not None and result.num_rows > 0:
            results_by_iso[iso] = result
            total_gated += n_gated
            total_dedup += n_dedup
            print(f"  {iso:>6}  {n_input:>8,}  {n_gated:>8,}  {n_dedup:>8,}  {elapsed:>5.1f}s", flush=True)
    print(f"\n  Total: {total_input:,} -> {total_gated:,} (gated) -> {total_dedup:,} (dedup)")
    if total_gated > 0:
        print(f"  Reduction: {(1 - total_dedup/total_gated)*100:.1f}%")

    if not results_by_iso:
        raise RuntimeError('Step 2 produced no ISO outputs to write.')

    # Write per-ISO EF parquets
    write_per_iso_outputs(results_by_iso)

    elapsed_total = time.time() - total_start

    # Score distribution summary (only read scores from the already-in-memory result tables)
    for iso in target_isos:
        if iso not in results_by_iso:
            continue
        iso_scores = results_by_iso[iso].column('hourly_match_score').to_numpy()
        if len(iso_scores) > 0:
            avail = []
            for thr in TARGET_THRESHOLDS:
                n_above = int((iso_scores >= thr).sum())
                avail.append(f"{thr:.0f}%:{n_above:,}")
            print(f"  {iso} mixes per threshold: {', '.join(avail[:6])}...")

    print(f"\n  Total rows: {total_input:,} -> {total_dedup:,} (EF)")
    print(f"  Total time: {elapsed_total:.0f}s")
    print("\n" + "=" * 70)
    print("  STEP 2 COMPLETE — per-ISO EF parquets ready in data/step2-ef-parquets/")
    print("  Step 3 reads from that directory; no merged output file is written.")
    print("=" * 70)


if __name__ == '__main__':
    main()
