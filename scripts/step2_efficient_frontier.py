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
Output: data/step2-ef-parquets/step2_ef_{ISO}_t{XX}.parquet (per-ISO per-threshold band)

After dedup, mixes are partitioned into non-overlapping threshold bands based
on their score. Each mix appears in exactly one file — the band whose threshold
is the highest T where T <= score. Step 3 loads bands >= its target threshold
to reconstruct the full qualifying set for any target.

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

# Resource columns per ISO — dimensionality depends on available resources
# Interior ISOs (ERCOT, MISO, SPP): 4D
# Offshore ISOs (NYISO, NEISO, PJM): 5D (+ offshore_wind)
# CAISO: 6D (+ offshore_wind + geothermal)
RESOURCE_COLS_BASE = ['clean_firm', 'solar', 'wind', 'hydro']
RESOURCE_COLS_OFFSHORE = ['clean_firm', 'solar', 'wind', 'hydro', 'offshore_wind']
RESOURCE_COLS_CAISO = ['clean_firm', 'solar', 'wind', 'hydro', 'offshore_wind', 'geothermal']

# ISOs with offshore wind resource (must match step1_pfs_generator.py)
OFFSHORE_ISOS = ['NYISO', 'NEISO', 'PJM', 'CAISO']

# Storage dispatch columns (always present)
STORAGE_COLS = ['battery_dispatch_pct', 'battery8_dispatch_pct', 'ldes_dispatch_pct', 'h2_dispatch_pct']

# Common columns in output (resource cols are ISO-dependent)
COMMON_COLS = ['iso', 'hourly_match_score', 'pareto_type']

# Resource caps — must match Step 1 (step1_pfs_generator.py)
SOLAR_CAP = 100       # Max solar as % of demand
TOTAL_PROCUREMENT_CAP = 350  # Max sum of all resources as % of demand


def get_resource_cols(iso):
    """Return resource column names for the given ISO."""
    if iso == 'CAISO':
        return RESOURCE_COLS_CAISO
    elif iso in OFFSHORE_ISOS:
        return RESOURCE_COLS_OFFSHORE
    else:
        return RESOURCE_COLS_BASE


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
        elif name in ('geothermal', 'offshore_wind'):
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


def _extract_threshold_from_filename(fname):
    """Extract threshold value from a PFS filename. Returns float or None."""
    # Handles: {ISO}_t{XX}_raw_pfs.parquet, {ISO}_t{XX}_raw_pfs_b{N}.parquet
    # Also handles 1d/ prefixed files: 1d/{ISO}_t{XX}_storage.parquet
    base = fname[3:] if fname.startswith('1d/') else fname
    m = re.match(r'^[A-Z]+_t([\d.]+)_', base)
    return float(m.group(1)) if m else None


def scan_iso_files(target_isos=None, threshold_filter=None):
    """Scan Step 1 parquet directory and return {iso: [filenames]} mapping.

    Groups files by ISO prefix, handles canonical vs batch file dedup.
    Does NOT load any data — just discovers and validates file paths.

    Args:
        target_isos: List of ISO names to scan for. None = all ISOs.
        threshold_filter: Optional set of threshold values. If provided, only
            include PFS files whose filename threshold is in this set. This
            pre-filters at the file level for batched threshold processing,
            avoiding loading large files that won't pass the threshold gate.

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
            # Apply threshold filter at file level if provided
            if threshold_filter is not None:
                file_thr = _extract_threshold_from_filename(fname)
                if file_thr is not None and file_thr not in threshold_filter:
                    continue
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
                    # Apply threshold filter for 1D files too
                    if threshold_filter is not None:
                        file_thr = _extract_threshold_from_filename(fname)
                        if file_thr is not None and file_thr not in threshold_filter:
                            continue
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


# Active thresholds (50+) that require Step 1D storage refinement data
STEP1D_REQUIRED_THRESHOLDS = [t for t in TARGET_THRESHOLDS if t >= 50.0]


def validate_step1d_coverage(files_by_iso, target_isos, threshold_filter=None):
    """Validate that Step 1D storage-refined parquets exist for active thresholds.

    Step 1D fills storage dispatch gaps from Step 1C's coarse grid. Without
    these files, the EF will miss high-quality storage-optimized mixes,
    leading to suboptimal results at thresholds >= 50%.

    Args:
        files_by_iso: Dict from scan_iso_files()
        target_isos: List of ISOs being processed
        threshold_filter: Optional threshold filter set

    Returns:
        True if all required 1D files are present, False if any are missing.
        Always prints warnings for missing coverage.
    """
    required_thresholds = STEP1D_REQUIRED_THRESHOLDS
    if threshold_filter is not None:
        required_thresholds = [t for t in required_thresholds if t in threshold_filter]

    if not required_thresholds:
        return True  # No active thresholds need 1D data

    if not os.path.isdir(STEP1D_DIR):
        print(f"\n  WARNING: Step 1D directory missing: {STEP1D_DIR}")
        print(f"  Run step1d_storage_refinement.py to generate storage-refined parquets.")
        print(f"  Proceeding without 1D data — results may have suboptimal storage dispatch.\n")
        return False

    all_ok = True
    for iso in target_isos:
        fnames = files_by_iso.get(iso, [])
        # Extract thresholds covered by 1D files
        step1d_thresholds = set()
        for f in fnames:
            if f.startswith('1d/'):
                thr = _extract_threshold_from_filename(f[3:])  # Strip '1d/' prefix
                if thr is not None:
                    step1d_thresholds.add(thr)

        missing = [t for t in required_thresholds if t not in step1d_thresholds]
        if missing:
            all_ok = False
            print(f"  WARNING: {iso} missing Step 1D data for thresholds: "
                  f"{', '.join(f'{t}%' for t in missing)}")

    if not all_ok:
        print(f"  Run step1d_storage_refinement.py to fill gaps.")
        print(f"  Proceeding with available data — storage dispatch may be suboptimal.\n")
    else:
        print(f"  Step 1D validation: all {len(target_isos)} ISOs have "
              f"storage-refined data for {len(required_thresholds)} active thresholds")

    return all_ok


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
            rg = resource_cap_filter(rg, iso)
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
    """Keep only rows matching target thresholds (respects batch filtering)."""
    threshold_col = table.column('threshold')
    target_set = pa.array(sorted(TARGET_THRESHOLD_SET), type=pa.float64())
    mask = pc.is_in(threshold_col, value_set=target_set)
    return table.filter(mask)


def resource_cap_filter(table, iso):
    """Enforce solar cap and total procurement cap (matching Step 1 constraints).

    Removes mixes where solar > SOLAR_CAP or total resources > TOTAL_PROCUREMENT_CAP.
    """
    if table.num_rows == 0:
        return table

    resource_cols = get_resource_cols(iso)
    solar = table.column('solar').to_numpy()
    mask = solar <= SOLAR_CAP

    total = np.zeros(len(solar))
    for col in resource_cols:
        total += table.column(col).to_numpy()
    mask &= total <= TOTAL_PROCUREMENT_CAP

    if mask.all():
        return table
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


def format_threshold(thr):
    """Format threshold for filenames (e.g., 87.5 → '87.5', 10.0 → '10')."""
    return f'{thr:g}'


def partition_by_threshold_band(table, thresholds):
    """Partition a deduped table into non-overlapping threshold bands.

    Each mix is assigned to the highest threshold T where T <= mix's score.
    Mixes scoring below the lowest threshold go into the lowest band.

    Returns dict: threshold (float) → pyarrow.Table of mixes in that band.
    """
    scores = table.column('hourly_match_score').to_numpy()
    sorted_thrs = sorted(thresholds)

    # searchsorted(side='right') gives first index where threshold > score.
    # Subtract 1 → highest threshold <= score.
    band_indices = np.searchsorted(sorted_thrs, scores, side='right') - 1
    # Scores below lowest threshold go to band 0
    band_indices = np.clip(band_indices, 0, len(sorted_thrs) - 1)

    result = {}
    for i, thr in enumerate(sorted_thrs):
        mask = band_indices == i
        n = int(mask.sum())
        if n > 0:
            indices = np.where(mask)[0]
            result[thr] = table.take(indices)

    return result


def write_per_iso_threshold_outputs(results_by_iso, thresholds, batch_label=None):
    """Write per-ISO/threshold Step 2 EF outputs to data/step2-ef-parquets.

    Partitions each ISO's deduped mixes into non-overlapping threshold bands
    and writes one parquet file per band.

    Args:
        results_by_iso: dict of {iso: pyarrow.Table}
        thresholds: list of threshold values to partition into bands
        batch_label: If set, write to step2_ef_{ISO}_t{T}_batch_{label}.parquet
                     instead of the final step2_ef_{ISO}_t{T}.parquet.
    """
    os.makedirs(STEP2_EF_OUTPUT_DIR, exist_ok=True)
    written = []
    total_files = 0

    for iso, table in results_by_iso.items():
        bands = partition_by_threshold_band(table, thresholds)
        iso_files = 0
        iso_rows = 0

        for thr in sorted(bands.keys()):
            band_table = bands[thr]
            thr_str = format_threshold(thr)
            if batch_label:
                fname = f'step2_ef_{iso}_t{thr_str}_batch_{batch_label}.parquet'
            else:
                fname = f'step2_ef_{iso}_t{thr_str}.parquet'
            path = os.path.join(STEP2_EF_OUTPUT_DIR, fname)
            pq.write_table(band_table, path, compression='snappy')
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"    {fname} ({band_table.num_rows:,} rows, {size_mb:.1f} MB)")
            written.append(path)
            iso_files += 1
            iso_rows += band_table.num_rows

        print(f"  {iso}: {iso_files} threshold files, {iso_rows:,} total rows")
        total_files += iso_files

    print(f"  Total: {total_files} files written")
    return written


def partitioned_dedup(table, iso):
    """Deduplicate a large table by partitioning on clean_firm to bound peak memory.

    Each partition (one clean_firm value at a time) is deduped independently.
    Since the dedup key includes clean_firm, partitioning by it is lossless —
    no cross-partition duplicates can exist.

    This keeps peak memory proportional to the largest single clean_firm
    partition (~1-3M rows) instead of the full table (60-70M rows), preventing
    OOM on GitHub Actions runners (7 GB RAM).
    """
    resource_cols = get_resource_cols(iso)
    cf_col = table.column('clean_firm').to_numpy()
    unique_cf = np.unique(cf_col)

    print(f"    Partitioned dedup: {table.num_rows:,} rows across {len(unique_cf)} clean_firm values", flush=True)

    kept_indices = []
    for cf_val in unique_cf:
        mask = cf_col == cf_val
        partition_indices = np.where(mask)[0]

        if len(partition_indices) <= 1:
            kept_indices.append(partition_indices)
            continue

        partition = table.take(partition_indices)
        arrays = {}
        for col in resource_cols + STORAGE_COLS + ['hourly_match_score']:
            arrays[col] = partition.column(col).to_numpy()
        local_keep = deduplicate_mixes(arrays, resource_cols)
        kept_indices.append(partition_indices[local_keep])
        del partition, arrays

    all_keep = np.concatenate(kept_indices)
    result = table.take(all_keep)
    del cf_col, kept_indices
    gc.collect()

    print(f"    Partitioned dedup result: {result.num_rows:,} rows", flush=True)
    return result


def _batch_labels_cover_all_thresholds(batch_files):
    """Check if batch files cover all 3 predefined threshold batches (low/mid/high).

    When all 3 batches are present they span every threshold in
    TARGET_THRESHOLDS, so the existing final EF is fully superseded and
    does not need to be loaded during merge.
    """
    labels = set()
    for bf in batch_files:
        base = os.path.basename(bf)
        # step2_ef_{ISO}_t{T}_batch_{label}.parquet
        m = re.match(r'^step2_ef_[A-Z]+_t[\d.]+_batch_(.+)\.parquet$', base)
        if m:
            labels.add(m.group(1))
    return labels >= {'low', 'mid', 'high'}


def merge_batch_outputs(iso):
    """Merge all per-threshold batch files for an ISO into final per-threshold EF files.

    Memory-efficient streaming merge: reads batch files one at a time,
    deduplicating incrementally so peak memory never exceeds ~2x the
    largest single batch (instead of the sum of all inputs).

    After merging and dedup, partitions into per-threshold band files:
    step2_ef_{ISO}_t{T}.parquet.

    When all 3 predefined batches (low/mid/high) are present, the
    existing final EF files are skipped since batches already cover
    every threshold.

    Returns (final_rows, n_batches) or (0, 0) if no batch files found.
    """
    import glob as globmod

    pattern = os.path.join(STEP2_EF_OUTPUT_DIR, f'step2_ef_{iso}_t*_batch_*.parquet')
    batch_files = sorted(globmod.glob(pattern))

    if not batch_files:
        print(f"  {iso}: No batch files found to merge")
        return 0, 0

    print(f"\n  {iso}: Merging {len(batch_files)} batch files...")

    # Determine whether to include the existing final EF files.
    all_batches_present = _batch_labels_cover_all_thresholds(batch_files)

    # Discover existing final per-threshold files
    existing_thr_files = sorted(globmod.glob(
        os.path.join(STEP2_EF_OUTPUT_DIR, f'step2_ef_{iso}_t*.parquet')
    ))
    # Exclude batch files from existing list
    existing_thr_files = [f for f in existing_thr_files if '_batch_' not in os.path.basename(f)]

    include_existing = (not all_batches_present) and len(existing_thr_files) > 0

    if all_batches_present:
        print(f"    All 3 batches present — skipping existing EF (fully superseded)")

    # Build the list of files to stream through
    files_to_merge = list(batch_files)
    if include_existing:
        files_to_merge.extend(existing_thr_files)

    # ── Streaming merge with incremental dedup ────────────────────────
    accumulated = None
    total_rows_in = 0
    n_sources = 0

    for fpath in files_to_merge:
        t = pq.read_table(fpath)
        total_rows_in += t.num_rows
        n_sources += 1
        basename = os.path.basename(fpath)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        is_existing = fpath in existing_thr_files
        label = f"Existing {basename}" if is_existing else basename
        print(f"    {label}: {t.num_rows:,} rows ({size_mb:.1f} MB)")

        if accumulated is None:
            accumulated = t
        else:
            merged = pa.concat_tables([accumulated, t], promote_options='permissive')
            del accumulated, t
            gc.collect()

            pre = merged.num_rows
            accumulated = partitioned_dedup(merged, iso)
            del merged
            gc.collect()
            print(f"    Streaming dedup: {pre:,} -> {accumulated.num_rows:,}", flush=True)

    if accumulated is None or accumulated.num_rows == 0:
        return 0, 0

    print(f"  Combined: {total_rows_in:,} input rows -> {accumulated.num_rows:,} after streaming dedup")

    # Partition into per-threshold band files
    print(f"  Writing per-threshold band files...")
    bands = partition_by_threshold_band(accumulated, TARGET_THRESHOLDS)
    final_rows = accumulated.num_rows
    del accumulated
    gc.collect()

    for thr in sorted(bands.keys()):
        band_table = bands[thr]
        thr_str = format_threshold(thr)
        fname = f'step2_ef_{iso}_t{thr_str}.parquet'
        path = os.path.join(STEP2_EF_OUTPUT_DIR, fname)
        pq.write_table(band_table, path, compression='snappy')
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"    {fname} ({band_table.num_rows:,} rows, {size_mb:.1f} MB)")

    del bands
    gc.collect()

    # Clean up batch files after successful merge
    for bf in batch_files:
        os.remove(bf)
        print(f"    Removed: {os.path.basename(bf)}")

    # Remove old existing per-threshold files that were superseded
    if include_existing:
        for ef in existing_thr_files:
            if os.path.exists(ef):
                os.remove(ef)
                print(f"    Removed old: {os.path.basename(ef)}")

    return final_rows, len(batch_files)


# Predefined threshold batches for large ISOs. Each batch processes a subset
# of thresholds to keep peak memory under ~4 GB during dedup. The highest
# thresholds (99+) produce the most unique mixes and go in their own batch.
THRESHOLD_BATCHES = {
    'low':  [10.0, 20.0, 30.0, 40.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0],
    'mid':  [80.0, 85.0, 87.5, 90.0, 92.5, 95.0],
    'high': [97.5, 99.0, 99.5, 99.9, 99.99],
}


def parse_args():
    parser = argparse.ArgumentParser(description='Step 2: Efficient Frontier extraction')
    parser.add_argument('--iso', type=str, default=None,
                        help='Single ISO to process (e.g. CAISO). Default: all ISOs.')
    parser.add_argument('--thresholds', type=str, default=None,
                        help='Comma-separated thresholds to process (e.g. "50,55,60,65,70"). '
                             'Only PFS files matching these thresholds will be loaded. '
                             'Default: all thresholds.')
    parser.add_argument('--batch', type=str, default=None,
                        choices=list(THRESHOLD_BATCHES.keys()),
                        help='Use a predefined threshold batch (low/mid/high). '
                             'Mutually exclusive with --thresholds.')
    parser.add_argument('--batch-label', type=str, default=None,
                        help='Label for batch output file (e.g. "low", "mid", "high"). '
                             'Output: step2_ef_{ISO}_batch_{label}.parquet. '
                             'Auto-set when --batch is used.')
    parser.add_argument('--merge', action='store_true',
                        help='Merge mode: combine batch parquet files into final EF. '
                             'Reads step2_ef_{ISO}_batch_*.parquet, deduplicates, '
                             'writes step2_ef_{ISO}.parquet.')
    parser.add_argument('--clear', action='store_true',
                        help='Clear existing step2-ef-parquets for the target ISO(s) '
                             'before running. Removes stale EF files from previous runs.')
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

    # ── Clear existing EF outputs for target ISOs ─────────────────────
    if args.clear and os.path.isdir(STEP2_EF_OUTPUT_DIR):
        import glob as globmod
        removed = 0
        for iso in target_isos:
            pattern = os.path.join(STEP2_EF_OUTPUT_DIR, f'step2_ef_{iso}_t*.parquet')
            for f in globmod.glob(pattern):
                os.remove(f)
                removed += 1
        if removed:
            print(f"  Cleared {removed} existing EF file(s) for {', '.join(target_isos)}")

    # ── Merge mode ─────────────────────────────────────────────────────
    if args.merge:
        print("=" * 70, flush=True)
        print("  STEP 2: MERGE BATCH EF OUTPUTS")
        print(f"  ISOs: {', '.join(target_isos)}")
        print("=" * 70, flush=True)

        total_start = time.time()
        for iso in target_isos:
            t0 = time.time()
            n_rows, n_batches = merge_batch_outputs(iso)
            elapsed = time.time() - t0
            if n_batches > 0:
                print(f"  {iso}: {n_rows:,} final rows from {n_batches} batches ({elapsed:.0f}s)")

        print(f"\n  Total time: {time.time() - total_start:.0f}s")
        print("=" * 70)
        print("  MERGE COMPLETE")
        print("=" * 70)
        return

    # ── Batch / threshold configuration ────────────────────────────────
    threshold_filter = None
    batch_label = args.batch_label
    active_thresholds = TARGET_THRESHOLDS

    if args.batch and args.thresholds:
        raise ValueError("Cannot use both --batch and --thresholds. Pick one.")

    if args.batch:
        active_thresholds = THRESHOLD_BATCHES[args.batch]
        threshold_filter = set(active_thresholds)
        batch_label = batch_label or args.batch
        print(f"  Batch '{args.batch}': thresholds {active_thresholds}")

    elif args.thresholds:
        active_thresholds = [float(t.strip()) for t in args.thresholds.split(',')]
        threshold_filter = set(active_thresholds)
        if not batch_label:
            # Auto-generate label from threshold range
            batch_label = f"t{active_thresholds[0]:.0f}-{active_thresholds[-1]:.0f}"
        print(f"  Custom thresholds: {active_thresholds}")

    mode_str = f"batch={batch_label}" if batch_label else "full"

    print("=" * 70, flush=True)
    print("  STEP 2: EFFICIENT FRONTIER (EF) EXTRACTION")
    print(f"  ISOs: {', '.join(target_isos)} | Mode: {mode_str}")
    if threshold_filter:
        print(f"  Thresholds: {sorted(threshold_filter)}")
    print("  PFS -> PFS post-EF (threshold-free, deduplicated)")
    print("=" * 70, flush=True)

    total_start = time.time()

    # Override TARGET_THRESHOLDS for gate function if batch mode
    global TARGET_THRESHOLD_SET
    if threshold_filter:
        TARGET_THRESHOLD_SET = threshold_filter

    # Scan files (no data loaded yet) — with threshold pre-filter
    print(f"\nScanning Step 1 raw parquets in {STEP1_RAW_DIR} + {STEP1D_DIR}")
    files_by_iso = scan_iso_files(target_isos, threshold_filter=threshold_filter)

    found = sorted(files_by_iso.keys())
    print(f"Found {len(found)} ISOs: {', '.join(found)}")
    for iso in found:
        n_files = len(files_by_iso[iso])
        print(f"  {iso}: {n_files} files to process")

    # Validate Step 1D coverage for active thresholds (50%+)
    validate_step1d_coverage(files_by_iso, target_isos, threshold_filter=threshold_filter)

    # Process each ISO incrementally: load → gate → dedup per file
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

    # Write outputs — partition into per-threshold band files
    write_per_iso_threshold_outputs(results_by_iso, active_thresholds, batch_label=batch_label)

    elapsed_total = time.time() - total_start

    # Score distribution summary (only read scores from the already-in-memory result tables)
    for iso in target_isos:
        if iso not in results_by_iso:
            continue
        iso_scores = results_by_iso[iso].column('hourly_match_score').to_numpy()
        if len(iso_scores) > 0:
            avail = []
            for thr in sorted(active_thresholds):
                n_above = int((iso_scores >= thr).sum())
                avail.append(f"{thr}%:{n_above:,}")
            print(f"  {iso} mixes per threshold: {', '.join(avail[:6])}...")

    print(f"\n  Total rows: {total_input:,} -> {total_dedup:,} (EF)")
    print(f"  Total time: {elapsed_total:.0f}s")
    print("\n" + "=" * 70)
    if batch_label:
        print(f"  STEP 2 BATCH '{batch_label}' COMPLETE")
        print(f"  Batch outputs in data/step2-ef-parquets/step2_ef_*_t*_batch_{batch_label}.parquet")
        print(f"  Run with --merge to combine all batches into final per-threshold EF files.")
    else:
        print("  STEP 2 COMPLETE — per-ISO/threshold EF parquets ready in data/step2-ef-parquets/")
        print("  Output: step2_ef_{ISO}_t{T}.parquet (one file per threshold band)")
    print("=" * 70)


if __name__ == '__main__':
    main()
