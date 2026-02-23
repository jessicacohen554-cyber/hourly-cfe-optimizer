#!/usr/bin/env python3
"""
Step 2: Efficient Frontier (EF) Extraction from Physics Feasible Space (PFS)
=============================================================================
Reads the PFS parquet and applies a 3-phase reduction:

  Phase 0: Existing generation filter — remove mixes that don't fully utilize
           existing clean generation (even at 2050 high demand growth). Two
           sub-filters:
           (a) Demand-fraction check: procurement × allocation must cover
               the existing share at 2050 high growth.
           (b) Allocation floor: direct floor on each resource's allocation
               (existing_share / growth_factor), ensuring no mix under-allocates
               relative to what the grid already has.
  Phase 1: Threshold gate — keep only rows whose scores fall in target ranges
  Phase 2: Global deduplication and Pareto-optimal procurement selection.
           Drop the threshold column. For each unique allocation
           (ISO/CF/Sol/Wnd/Hyd/Bat/Bat8/LDES), keep only the Pareto front
           on (minimize procurement, maximize score). Each unique physical
           configuration is stored ONCE — Step 3 handles threshold selection
           by filtering to mixes with score >= target threshold, enabling
           cross-threshold picking (a cheap mix that overachieves can win).

Note: No dominance removal across different resource mixes is performed.
Different resource mixes at the same procurement/storage/score can have very
different costs under different LCOE assumptions — removing them risks losing
true cost optimums. Cost-based selection happens in Step 3.

Pipeline position: Step 2 of 4
  Step 1 — PFS Generator (step1_pfs_generator.py)
  Step 2 — Efficient Frontier extraction (this file)
  Step 3 — Cost optimization (step3_cost_optimization.py)
  Step 4 — Post-processing (step4_postprocess.py)

Input:  data/step1_raw_pfs_parquets/{ISO}_step1_pfs_t{threshold}.parquet  (primary, from Step 1)
        Falls back to data/physics_cache_v4_{ISO}.parquet or legacy merged file
Output: data/step-2-EF-parquets/step2_ef_{ISO}.parquet (per-ISO, threshold-free)
        data/pfs_post_ef.parquet (merged copy for compatibility)

The output preserves all mixes that could be optimal under ANY cost assumption
at ANY threshold, ensuring no true optimum is lost during Step 3.
"""

import os
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PFS_DIR = os.path.join(SCRIPT_DIR, 'data')
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'data', 'pfs_post_ef.parquet')
STEP1_RAW_DIR = os.path.join(PFS_DIR, 'step1_raw_pfs_parquets')
STEP2_EF_OUTPUT_DIR = os.path.join(PFS_DIR, 'step-2-EF-parquets')

# Per-ISO PFS files (from Step 1 two-phase adaptive sweep)
# Falls back to legacy single-file if per-ISO files don't exist
LEGACY_PFS_PATH = os.path.join(PFS_DIR, 'physics_cache_v4.parquet')

# Target thresholds — all 13 from v4 PFS (50-100%)
TARGET_THRESHOLDS = [50.0, 60.0, 70.0, 75.0, 80.0, 85.0, 87.5, 90.0, 92.5, 95.0, 97.5, 99.0, 100.0]

ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']

# Existing clean generation as % of 2025 demand (from eGRID/EIA)
GRID_MIX_SHARES = {
    'CAISO': {'clean_firm': 7.9, 'solar': 22.3, 'wind': 8.8, 'ccs_ccgt': 0, 'hydro': 9.5},
    'ERCOT': {'clean_firm': 8.6, 'solar': 13.8, 'wind': 23.6, 'ccs_ccgt': 0, 'hydro': 0.1},
    'PJM':   {'clean_firm': 32.1, 'solar': 2.9, 'wind': 3.8, 'ccs_ccgt': 0, 'hydro': 1.8},
    'NYISO': {'clean_firm': 18.4, 'solar': 0.0, 'wind': 4.7, 'ccs_ccgt': 0, 'hydro': 15.9},
    'NEISO': {'clean_firm': 23.8, 'solar': 1.4, 'wind': 3.9, 'ccs_ccgt': 0, 'hydro': 4.4},
    'MISO':  {'clean_firm': 13.1, 'solar': 2.1, 'wind': 14.5, 'ccs_ccgt': 0, 'hydro': 1.6},
    'SPP':   {'clean_firm': 5.2, 'solar': 0.4, 'wind': 37.1, 'ccs_ccgt': 0, 'hydro': 4.3},
}

# High demand growth rates (annual) — most aggressive scenario
# At 2050 high growth, existing share is smallest → tightest filter
DEMAND_GROWTH_HIGH = {
    'CAISO': 0.025, 'ERCOT': 0.055, 'PJM': 0.036, 'NYISO': 0.044, 'NEISO': 0.029,
    'MISO': 0.038, 'SPP': 0.030,
}


def compute_existing_min_shares():
    """Compute minimum existing shares at 2050 high growth for each ISO.
    Returns dict of {iso: {rtype: min_share_as_pct_of_demand}}."""
    result = {}
    for iso in ISOS:
        growth_factor = (1 + DEMAND_GROWTH_HIGH[iso]) ** 25  # 2050 - 2025 = 25 years
        result[iso] = {}
        for rtype, share in GRID_MIX_SHARES[iso].items():
            result[iso][rtype] = share / growth_factor  # as % of grown demand
    return result


def load_pfs():
    """Load the v4 Physics Feasible Space from per-ISO parquet files.

    Reads data/physics_cache_v4_{ISO}.parquet for each ISO, concatenates
    into a single table. Falls back to legacy single-file if per-ISO
    files are not available.
    """
    # Try per-ISO files first
    per_iso_tables = []
    found_isos = []
    for iso in ISOS:
        iso_path = os.path.join(PFS_DIR, f'physics_cache_v4_{iso}.parquet')
        if os.path.exists(iso_path):
            t = pq.read_table(iso_path)
            per_iso_tables.append(t)
            found_isos.append(iso)
            size_mb = os.path.getsize(iso_path) / (1024 * 1024)
            print(f"  {iso}: {t.num_rows:>10,} rows  ({size_mb:.1f} MB)")

    if per_iso_tables:
        print(f"\nLoaded {len(found_isos)} per-ISO PFS files: {', '.join(found_isos)}")
        table = pa.concat_tables(per_iso_tables)
        print(f"  Combined: {table.num_rows:,} rows")
        missing = [iso for iso in ISOS if iso not in found_isos]
        if missing:
            print(f"  WARNING: Missing ISOs: {', '.join(missing)}")
        return table

    # Fallback to legacy single file
    if os.path.exists(LEGACY_PFS_PATH):
        print(f"Loading legacy PFS: {LEGACY_PFS_PATH}")
        table = pq.read_table(LEGACY_PFS_PATH)
        print(f"  Total rows: {table.num_rows:,}")
        return table

    raise FileNotFoundError(
        f"No PFS files found. Expected per-ISO files in {PFS_DIR}/ "
        f"(physics_cache_v4_{{ISO}}.parquet) or legacy {LEGACY_PFS_PATH}"
    )


def load_step1_raw_pfs():
    """Load Step 1 raw parquet files from step1_raw_pfs_parquets/ directory.

    Handles two file schemas:
      1. Step 1 native (from step1_pfs_generator.py): flat columns including
         clean_firm, solar, wind, hydro, procurement_pct, battery_dispatch_pct,
         battery8_dispatch_pct, ldes_dispatch_pct, hourly_match_score, pareto_type.
         Filename pattern: {ISO}_step1_pfs_t{threshold}.parquet
      2. Step 1.5 legacy (from step1_5_convert_checkpoints_to_parquet.py):
         list-based 'mix' column with tuple-decoded fields.
         Filename pattern: {ISO}_t{threshold}_raw_pfs.parquet

    Returns:
        pyarrow.Table if raw files are present, otherwise None.
    """
    if not os.path.isdir(STEP1_RAW_DIR):
        print(f"Step 1 raw input directory missing: {STEP1_RAW_DIR}")
        return None

    parquet_files = sorted(
        os.path.join(STEP1_RAW_DIR, f)
        for f in os.listdir(STEP1_RAW_DIR)
        if f.endswith('.parquet')
    )
    if not parquet_files:
        print(f"No parquet files found in Step 1 raw input directory: {STEP1_RAW_DIR}")
        return None

    print(f"Loading Step 1 raw parquet files from {STEP1_RAW_DIR}")
    native_tables = []
    legacy_tables = []

    for path in parquet_files:
        t = pq.read_table(path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        fname = os.path.basename(path)

        # Detect schema: native files have 'clean_firm' column, legacy have 'mix' column
        if 'clean_firm' in t.column_names and 'hourly_match_score' in t.column_names:
            # Step 1 native format — already has the right columns
            print(f"  {fname}: {t.num_rows:>10,} rows ({size_mb:.1f} MB) [native]")
            native_tables.append(t)

        elif 'mix' in t.column_names:
            # Step 1.5 legacy format — needs normalization
            mix = t.column('mix')
            valid_mask = pc.and_(
                pc.equal(pc.list_value_length(mix), 5),
                pc.and_(
                    pc.is_valid(t.column('threshold')),
                    pc.is_valid(t.column('lcoe')),
                ),
            )

            filtered = t.filter(valid_mask)
            n = filtered.num_rows
            if n == 0:
                continue

            mix = filtered.column('mix')
            proc = pc.cast(filtered.column('threshold'), pa.float64())
            score = pc.cast(filtered.column('lcoe'), pa.float64())

            proc_pct = pc.if_else(pc.less_equal(proc, 5.0), pc.multiply(proc, 100.0), proc)
            score_pct = pc.if_else(pc.less_equal(score, 1.5), pc.multiply(score, 100.0), score)

            source_threshold = pc.fill_null(filtered.column('source_threshold'), pa.scalar(0.0))
            dispatch_mode = pc.fill_null(filtered.column('dispatch_mode'), pa.scalar(''))

            legacy_tables.append(pa.table({
                'iso': filtered.column('iso'),
                'threshold': pc.cast(source_threshold, pa.float64()),
                'clean_firm': pc.cast(pc.round(pc.multiply(pc.list_element(mix, 0), 100.0)), pa.int16()),
                'solar': pc.cast(pc.round(pc.multiply(pc.list_element(mix, 1), 100.0)), pa.int16()),
                'wind': pc.cast(pc.round(pc.multiply(pc.list_element(mix, 2), 100.0)), pa.int16()),
                'hydro': pc.cast(pc.round(pc.multiply(pc.list_element(mix, 3), 100.0)), pa.int16()),
                'procurement_pct': pc.cast(pc.round(proc_pct), pa.int16()),
                'battery_dispatch_pct': pc.cast(pc.multiply(pc.list_element(mix, 4), 100.0), pa.float64()),
                'battery8_dispatch_pct': pa.array(np.zeros(n, dtype=np.float64)),
                'ldes_dispatch_pct': pa.array(np.zeros(n, dtype=np.float64)),
                'hourly_match_score': pc.cast(score_pct, pa.float64()),
                'pareto_type': dispatch_mode,
            }))

            print(f"  {fname}: {n:>10,} rows ({size_mb:.1f} MB) [legacy]")

        else:
            print(f"  {fname}: SKIPPED (unrecognized schema: {t.column_names[:5]}...)")

    all_tables = native_tables + legacy_tables
    if not all_tables:
        return None

    # Ensure schema compatibility: cast native tables to match expected types
    # Native tables may have different dtypes (e.g., float64 vs int16 for resource columns)
    target_schema = pa.schema([
        ('iso', pa.string()),
        ('threshold', pa.float64()),
        ('clean_firm', pa.int16()),
        ('solar', pa.int16()),
        ('wind', pa.int16()),
        ('hydro', pa.int16()),
        ('procurement_pct', pa.int16()),
        ('battery_dispatch_pct', pa.float64()),
        ('battery8_dispatch_pct', pa.float64()),
        ('ldes_dispatch_pct', pa.float64()),
        ('hourly_match_score', pa.float64()),
        ('pareto_type', pa.string()),
    ])

    compatible_tables = []
    target_cols = [f.name for f in target_schema]
    for t in all_tables:
        cols = {}
        for field in target_schema:
            name = field.name
            if name in t.column_names:
                col = t.column(name)
                if col.type != field.type:
                    col = pc.cast(col, field.type)
                cols[name] = col
            elif name == 'battery8_dispatch_pct':
                cols[name] = pa.array(np.zeros(t.num_rows, dtype=np.float64))
            elif name == 'pareto_type':
                cols[name] = pa.array([''] * t.num_rows, type=pa.string())
            elif name == 'threshold':
                cols[name] = pa.array(np.zeros(t.num_rows, dtype=np.float64))
            else:
                raise ValueError(f"Missing required column '{name}' in {t.column_names}")
        compatible_tables.append(pa.table(cols))

    table = pa.concat_tables(compatible_tables)
    n_native = sum(t.num_rows for t in native_tables)
    n_legacy = sum(t.num_rows for t in legacy_tables)
    print(f"  Combined: {table.num_rows:,} rows ({n_native:,} native + {n_legacy:,} legacy)")
    return table


def step0_existing_generation_filter(table):
    """Step 0: Remove mixes that don't fully utilize existing clean generation.

    A mix is filtered if, for any resource type, the mix's allocation
    (procurement × resource_pct) is less than the existing generation's share
    of demand at 2050 high growth. Such mixes waste free/cheap existing
    generation and will never be cost-optimal under any sensitivity.

    Only filters on resources with >0 existing share.
    """
    print("\nStep 0: Existing generation utilization filter")
    min_shares = compute_existing_min_shares()

    cf = table.column('clean_firm').to_numpy()
    sol = table.column('solar').to_numpy()
    wnd = table.column('wind').to_numpy()
    hyd = table.column('hydro').to_numpy()
    proc = table.column('procurement_pct').to_numpy()
    iso_col = table.column('iso').to_pylist()

    # Convert ISO strings to indices for vectorized lookup
    iso_to_idx = {iso: i for i, iso in enumerate(ISOS)}
    iso_indices = np.array([iso_to_idx.get(s, 0) for s in iso_col])

    # Build min-share arrays indexed by ISO
    min_cf = np.array([min_shares[iso]['clean_firm'] for iso in ISOS])
    min_sol = np.array([min_shares[iso]['solar'] for iso in ISOS])
    min_wnd = np.array([min_shares[iso]['wind'] for iso in ISOS])
    min_hyd = np.array([min_shares[iso]['hydro'] for iso in ISOS])

    # For each mix: demand_fraction = procurement/100 * pct/100 * 100 = procurement * pct / 100
    # Min share is in % of demand. So check: procurement * pct / 100 >= min_share
    # Or: procurement * pct >= min_share * 100
    keep = np.ones(len(cf), dtype=np.bool_)

    # Materiality threshold: only filter on resources with >1% min existing share
    # This avoids filtering on negligible existing generation (e.g., ERCOT hydro 0.03%)
    MATERIALITY_THRESHOLD = 1.0  # % of demand at 2050 high growth

    for r_pct, r_min in [(cf, min_cf), (sol, min_sol), (wnd, min_wnd), (hyd, min_hyd)]:
        # Look up min share for each row's ISO
        row_min = r_min[iso_indices]  # min share as % of demand
        # Mix's demand fraction in %: proc * pct / 100
        mix_demand_pct = proc * r_pct / 100.0
        # Only filter if min share is material (>1% of demand at 2050 high)
        has_material_existing = row_min > MATERIALITY_THRESHOLD
        fails = has_material_existing & (mix_demand_pct < row_min)
        keep &= ~fails

    # Resource allocation floors: existing generation scaled to 2050 high demand growth.
    # Any mix allocating less than the floor for a resource wastes cheap existing generation.
    # Applied to all four resource types where the floor exceeds the materiality threshold.
    resource_cols = {'clean_firm': cf, 'solar': sol, 'wind': wnd, 'hydro': hyd}
    print("  Resource allocation floors (existing / 2050 high growth factor):")
    alloc_floor_removed = 0
    for rtype, r_col in resource_cols.items():
        floor_arr = np.zeros(len(ISOS), dtype=np.float64)
        for i, iso in enumerate(ISOS):
            existing = GRID_MIX_SHARES[iso].get(rtype, 0)
            growth_factor = (1 + DEMAND_GROWTH_HIGH[iso]) ** 25
            floor_val = existing / growth_factor
            if floor_val > MATERIALITY_THRESHOLD:
                floor_arr[i] = floor_val
        row_floor = floor_arr[iso_indices]
        fails = (row_floor > 0) & (r_col < row_floor)
        n_removed = (keep & fails).sum()
        if n_removed > 0:
            keep &= ~fails
            alloc_floor_removed += n_removed
        # Print summary per resource
        active_isos = [iso for i, iso in enumerate(ISOS) if floor_arr[i] > 0]
        if active_isos:
            floors_str = ", ".join(f"{iso}≥{floor_arr[ISOS.index(iso)]:.1f}" for iso in active_isos)
            print(f"    {rtype:>10}: {floors_str} (removed {n_removed:,})")
    if alloc_floor_removed > 0:
        print(f"  Allocation floor total: removed {alloc_floor_removed:,} additional rows")

    filtered = table.filter(pa.array(keep))
    removed = table.num_rows - filtered.num_rows

    # Per-ISO stats
    for iso in ISOS:
        iso_mask = np.array(iso_col) == iso
        iso_total = iso_mask.sum()
        iso_kept = (iso_mask & keep).sum()
        print(f"  {iso:>6}: {iso_total:>9,} → {iso_kept:>9,} "
              f"(removed {iso_total - iso_kept:>7,}, {(iso_total - iso_kept)/iso_total*100:.1f}%)")

    print(f"  Total: {table.num_rows:,} → {filtered.num_rows:,} (removed {removed:,})")
    return filtered


def step1_threshold_gate(table):
    """Step 1: Keep only target thresholds."""
    print("\nStep 1: Threshold gate")

    # Build OR filter for target thresholds
    threshold_col = table.column('threshold')
    mask = None
    for thr in TARGET_THRESHOLDS:
        eq = pc.equal(threshold_col, thr)
        mask = eq if mask is None else pc.or_(mask, eq)

    filtered = table.filter(mask)
    print(f"  {table.num_rows:,} → {filtered.num_rows:,} "
          f"(kept {len(TARGET_THRESHOLDS)} of {len(pc.unique(threshold_col).to_pylist())} thresholds)")
    return filtered


def step2_pareto_procurement(arrays):
    """
    For each unique allocation (CF/Sol/Wnd/Hyd/Bat4/Bat8/LDES), keep only
    the Pareto-optimal (procurement, score) pairs: rows where no other row
    with the same allocation has <= procurement AND >= score.

    Within each allocation group sorted by ascending procurement, this means
    keeping only rows where the score strictly increases (the running max).
    """
    n = len(arrays['clean_firm'])
    if n == 0:
        return np.array([], dtype=np.int64)

    cf = arrays['clean_firm']
    sol = arrays['solar']
    wnd = arrays['wind']
    hyd = arrays['hydro']
    bat = arrays['battery_dispatch_pct']
    bat8 = arrays['battery8_dispatch_pct']
    ldes = arrays['ldes_dispatch_pct']
    proc = arrays['procurement_pct']
    score = arrays['hourly_match_score']

    # Pack allocation into a single key
    group_key = (cf.astype(np.int64) * (101**6) +
                 sol.astype(np.int64) * (101**5) +
                 wnd.astype(np.int64) * (101**4) +
                 hyd.astype(np.int64) * (101**3) +
                 bat.astype(np.int64) * (101**2) +
                 bat8.astype(np.int64) * 101 +
                 ldes.astype(np.int64))

    # Sort by (allocation, procurement ascending, score descending)
    # Within same (allocation, proc), keep highest score; across proc levels,
    # keep only where score increases (Pareto front).
    sort_idx = np.lexsort((-score, proc, group_key))
    sorted_keys = group_key[sort_idx]
    sorted_proc = proc[sort_idx]
    sorted_score = score[sort_idx]

    keep_mask = np.zeros(n, dtype=np.bool_)

    # Walk through sorted rows; within each allocation group, track running max score
    i = 0
    while i < n:
        # Find end of this allocation group
        j = i + 1
        while j < n and sorted_keys[j] == sorted_keys[i]:
            j += 1

        # Within this group (sorted by proc asc, score desc):
        # For each unique proc level, keep the first row (highest score).
        # Across proc levels, keep only if score exceeds running max.
        running_max = -1.0
        prev_proc = -1
        for k in range(i, j):
            p = sorted_proc[k]
            s = sorted_score[k]
            # Skip duplicate proc levels (already took highest score)
            if p == prev_proc:
                continue
            prev_proc = p
            # Pareto check: keep only if score exceeds all lower-proc points
            if s > running_max:
                keep_mask[k] = True
                running_max = s

        i = j

    return sort_idx[keep_mask]


def process_iso(table, iso):
    """Process all rows for an ISO: global Pareto-optimal procurement selection."""
    mask = pc.equal(table.column('iso'), iso)
    subtable = table.filter(mask)
    n_raw = subtable.num_rows

    if n_raw == 0:
        return None, 0, 0

    arrays = {
        'clean_firm': subtable.column('clean_firm').to_numpy(),
        'solar': subtable.column('solar').to_numpy(),
        'wind': subtable.column('wind').to_numpy(),
        'hydro': subtable.column('hydro').to_numpy(),
        'procurement_pct': subtable.column('procurement_pct').to_numpy(),
        'battery_dispatch_pct': subtable.column('battery_dispatch_pct').to_numpy(),
        'battery8_dispatch_pct': (subtable.column('battery8_dispatch_pct').to_numpy()
                                   if 'battery8_dispatch_pct' in subtable.column_names
                                   else np.zeros(n_raw, dtype=np.int64)),
        'ldes_dispatch_pct': subtable.column('ldes_dispatch_pct').to_numpy(),
        'hourly_match_score': subtable.column('hourly_match_score').to_numpy(),
    }

    pareto_idx = step2_pareto_procurement(arrays)
    n_pareto = len(pareto_idx)

    # Build result without threshold column
    result_cols = ['iso', 'clean_firm', 'solar', 'wind', 'hydro',
                   'procurement_pct', 'battery_dispatch_pct',
                   'battery8_dispatch_pct', 'ldes_dispatch_pct',
                   'hourly_match_score']
    if 'pareto_type' in subtable.column_names:
        result_cols.append('pareto_type')

    result_arrays = []
    for col_name in result_cols:
        if col_name in subtable.column_names:
            result_arrays.append(subtable.column(col_name).take(pareto_idx))
        elif col_name == 'battery8_dispatch_pct':
            result_arrays.append(pa.array(np.zeros(n_pareto, dtype=np.int64)))

    result = pa.table(result_arrays, names=[c for c in result_cols if c in subtable.column_names or c == 'battery8_dispatch_pct'])
    return result, n_raw, n_pareto




def write_per_iso_outputs(results_by_iso):
    """Write per-ISO Step 2 EF outputs to data/step-2-EF-parquets."""
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
    print("  PFS → PFS post-EF (threshold-free)")
    print("=" * 70)

    total_start = time.time()
    table = load_step1_raw_pfs()
    if table is None:
        print("No Step 1 raw parquets found, falling back to per-ISO PFS files...")
        table = load_pfs()
    else:
        print("Using Step 1 raw parquet directory as Step 2 input source")

    # Step 0: Existing generation utilization filter — DISABLED (Feb 20, 2026)
    # Removed to allow below-floor mixes (hydro=0, low clean_firm) into the EF
    # for Track 1 (new-build hourly matching) and Track 2 (cost to replace existing).
    # The filter function is preserved above for reference but no longer called.
    # table = step0_existing_generation_filter(table)

    # Step 1: Threshold gate (keep only rows from target threshold ranges)
    table = step1_threshold_gate(table)

    # Step 2: Global Pareto-optimal procurement per ISO
    # Drop threshold column, deduplicate, keep Pareto front on (proc, score)
    # per allocation. Step 3 will filter by score >= target threshold.
    print("\nStep 2: Global Pareto-optimal procurement (threshold-free)")
    print(f"  {'ISO':>6}  {'Raw':>9}  {'Pareto':>9}  {'Time':>6}")
    print("  " + "-" * 40)

    results = []
    results_by_iso = {}
    total_raw = 0
    total_pareto = 0

    for iso in ISOS:
        t0 = time.time()
        result, n_raw, n_pareto = process_iso(table, iso)
        elapsed = time.time() - t0

        if result is not None and result.num_rows > 0:
            results.append(result)
            results_by_iso[iso] = result
            total_raw += n_raw
            total_pareto += n_pareto
            print(f"  {iso:>6}  {n_raw:>8,}  {n_pareto:>8,}  {elapsed:>5.1f}s")

    print(f"\n  Total: {total_raw:,} → {total_pareto:,}")
    if total_raw > 0:
        print(f"  Reduction: {(1 - total_pareto/total_raw)*100:.1f}%")

    if not results:
        raise RuntimeError('Step 2 produced no ISO outputs to write.')

    combined = pa.concat_tables(results)

    # Save per-ISO outputs first
    write_per_iso_outputs(results_by_iso)

    # Save merged compatibility output
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    pq.write_table(combined, OUTPUT_PATH, compression='snappy')
    file_size = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)

    elapsed_total = time.time() - total_start
    print(f"\n  Output: {OUTPUT_PATH}")
    print(f"  Size: {file_size:.1f} MB ({combined.num_rows:,} rows)")
    print(f"  Columns: {combined.column_names}")
    print(f"  Total time: {elapsed_total:.0f}s")

    # Score distribution summary
    scores = combined.column('hourly_match_score').to_numpy()
    for iso in ISOS:
        iso_mask = np.array(combined.column('iso').to_pylist()) == iso
        iso_scores = scores[iso_mask]
        if len(iso_scores) > 0:
            avail = []
            for thr in TARGET_THRESHOLDS:
                n = (iso_scores >= thr).sum()
                avail.append(f"{thr:.0f}%:{n:,}")
            print(f"  {iso} mixes per threshold: {', '.join(avail[:6])}...")

    print("\n" + "=" * 70)
    print("  STEP 2 COMPLETE — PFS post-EF ready for Step 3")
    print("  Step 3 filters by score >= threshold for cross-threshold picking")
    print("=" * 70)


if __name__ == '__main__':
    main()
