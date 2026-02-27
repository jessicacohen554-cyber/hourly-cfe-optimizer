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
# Must match Step 1 THRESHOLDS and Step 3 OUTPUT_THRESHOLDS
TARGET_THRESHOLDS = [10.0, 20.0, 30.0, 40.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 87.5, 90.0, 92.5, 95.0, 97.5, 99.0, 99.5, 99.9, 100.0]
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

    missing = [iso for iso in iso_filter if iso not in files_by_iso]
    if missing:
        print(f"  WARNING: Missing ISOs: {', '.join(missing)}")

    return files_by_iso


# Maximum accumulated rows before triggering an incremental dedup pass.
# Keeps peak memory under ~3GB during dedup (lexsort + sorted views).
INCREMENTAL_DEDUP_THRESHOLD = 5_000_000


def load_and_process_iso(iso, fnames):
    """Load, gate, and deduplicate a single ISO's threshold files incrementally.

    Instead of loading all files into memory at once (which OOMs on large ISOs
    like PJM with 63M+ rows), this streams each file: normalize → gate → drop
    threshold column → accumulate → periodically dedup to bound memory.

    The incremental dedup is correct because "keep max score per unique mix"
    is associative — deduplicating a partial accumulation then merging more
    rows and deduplicating again produces the same result as a single global
    dedup.

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

    for fname in fnames:
        path = os.path.join(STEP1_RAW_DIR, fname)
        t = pq.read_table(path)

        if 'clean_firm' not in t.column_names or 'hourly_match_score' not in t.column_names:
            del t
            continue

        total_input += t.num_rows

        # Normalize schema, then gate to target thresholds immediately
        t = normalize_table(t, iso)
        t = threshold_gate(t)
        total_gated += t.num_rows

        if t.num_rows == 0:
            del t
            continue

        # Drop threshold column — not needed after gating
        keep_cols = [c for c in result_col_names if c in t.column_names]
        t = t.select(keep_cols)

        # Accumulate
        if accumulated is None:
            accumulated = t
        else:
            accumulated = pa.concat_tables([accumulated, t], promote_options='permissive')
        del t

        # Incremental dedup when accumulator gets large
        if accumulated.num_rows > INCREMENTAL_DEDUP_THRESHOLD:
            arrays = {}
            for col in resource_cols + STORAGE_COLS + ['hourly_match_score']:
                arrays[col] = accumulated.column(col).to_numpy()
            keep_idx = deduplicate_mixes(arrays, resource_cols)
            accumulated = accumulated.take(keep_idx)
            del arrays
            n_incremental_dedups += 1

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

    if n_incremental_dedups > 0:
        print(f"    ({n_incremental_dedups} incremental dedup passes used)")

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

    Uses multi-column lexsort + boundary detection instead of a composite int64
    key to avoid overflow for CAISO (5 resources: 101^4 × 2001^3 × 100 > int64 max).

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
    h2_key = np.round(arrays['h2_dispatch_pct'] * STORAGE_SCALE).astype(np.int64)

    # Build integer arrays for each resource column
    res_keys = [arrays[col].astype(np.int64) for col in resource_cols]

    # Multi-column lexsort: sort by all group columns (ascending), then score (descending).
    # lexsort sorts by LAST key first → resource_cols[0] is primary sort key.
    sort_keys = [-score, h2_key, ldes_key, bat8_key, bat_key]
    for rk in reversed(res_keys):
        sort_keys.append(rk)
    sort_idx = np.lexsort(sort_keys)

    # Detect group boundaries: new group starts when ANY group column differs.
    # Build sorted column views for boundary detection.
    sorted_group_cols = [rk[sort_idx] for rk in res_keys]
    sorted_group_cols.append(bat_key[sort_idx])
    sorted_group_cols.append(bat8_key[sort_idx])
    sorted_group_cols.append(ldes_key[sort_idx])
    sorted_group_cols.append(h2_key[sort_idx])

    is_first = np.empty(n, dtype=np.bool_)
    is_first[0] = True
    # OR across all columns: any change → new group
    changed = sorted_group_cols[0][1:] != sorted_group_cols[0][:-1]
    for sc in sorted_group_cols[1:]:
        changed |= (sc[1:] != sc[:-1])
    is_first[1:] = changed

    return sort_idx[is_first]



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

    print("=" * 70)
    print("  STEP 2: EFFICIENT FRONTIER (EF) EXTRACTION")
    print(f"  ISOs: {', '.join(target_isos)}")
    print("  PFS -> PFS post-EF (threshold-free, deduplicated)")
    print("=" * 70)

    total_start = time.time()

    # Scan files (no data loaded yet)
    print(f"\nScanning Step 1 raw parquets in {STEP1_RAW_DIR}")
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
            print(f"  {iso:>6}  {n_input:>8,}  {n_gated:>8,}  {n_dedup:>8,}  {elapsed:>5.1f}s")
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
