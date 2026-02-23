#!/usr/bin/env python3
"""
Step 2: Efficient Frontier (EF) Extraction from Physics Feasible Space (PFS)
=============================================================================
Reads the PFS parquet and applies a 2-phase reduction:

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

Input:  data/step1_raw_pfs_parquets/{ISO}_step1_pfs_t{threshold}.parquet (primary, from Step 1)
        Falls back to data/physics_cache_v4_{ISO}.parquet or legacy merged file
Output: data/step-2-EF-parquets/step2_ef_{ISO}.parquet (per-ISO, threshold-free)

The output preserves all mixes that could be optimal under ANY cost assumption
at ANY threshold, ensuring no true optimum is lost during Step 3.
"""

import os
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

try:
    from numba import njit
except ImportError:  # pragma: no cover - workflow installs numba; local fallback
    njit = None

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PFS_DIR = os.path.join(SCRIPT_DIR, 'data')
STEP1_RAW_DIR = os.path.join(PFS_DIR, 'step1_raw_pfs_parquets')
STEP2_EF_OUTPUT_DIR = os.path.join(PFS_DIR, 'step-2-EF-parquets')

# Per-ISO PFS files (from Step 1 two-phase adaptive sweep)
# Falls back to legacy single-file if per-ISO files don't exist
LEGACY_PFS_PATH = os.path.join(PFS_DIR, 'physics_cache_v4.parquet')

# Target thresholds — all 13 from v4 PFS (50-100%)
TARGET_THRESHOLDS = [50.0, 60.0, 70.0, 75.0, 80.0, 85.0, 87.5, 90.0, 92.5, 95.0, 97.5, 99.0, 100.0]
TARGET_THRESHOLD_SET = set(TARGET_THRESHOLDS)

ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']

# Target schema for Step 2 output columns
TARGET_SCHEMA = pa.schema([
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


def normalize_table(t):
    """Cast a table to match TARGET_SCHEMA, filling missing columns with defaults."""
    cols = {}
    for field in TARGET_SCHEMA:
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
    return pa.table(cols)

def normalize_legacy_step1_raw_table(t):
    """Convert legacy Step 1 raw parquet schema to TARGET_SCHEMA columns."""
    required = {'iso', 'source_threshold', 'mix', 'threshold', 'dispatch_mode', 'battery_hours', 'ldes_hours', 'lcoe'}
    if not required.issubset(set(t.column_names)):
        return None

    mix_vals = t.column('mix').to_pylist()
    mix_np = np.asarray(mix_vals, dtype=np.float64)
    if mix_np.ndim != 2 or mix_np.shape[1] != 5:
        raise ValueError(f"Unsupported legacy mix shape: {mix_np.shape}")

    # Legacy mixes are [clean_firm_base, solar, wind, hydro, clean_firm_extra],
    # all as fractions summing to 1.0. Merge base+extra into clean_firm.
    clean_firm = np.rint((mix_np[:, 0] + mix_np[:, 4]) * 100.0).astype(np.int16)
    solar = np.rint(mix_np[:, 1] * 100.0).astype(np.int16)
    wind = np.rint(mix_np[:, 2] * 100.0).astype(np.int16)
    hydro = np.rint(mix_np[:, 3] * 100.0).astype(np.int16)

    # Legacy `threshold` is procurement ratio (0.75-2.00), while
    # `source_threshold` is the target score threshold (50-100).
    procurement_pct = np.rint(t.column('threshold').to_numpy() * 100.0).astype(np.int16)
    threshold = t.column('source_threshold').to_numpy().astype(np.float64)

    dispatch_mode = np.asarray(t.column('dispatch_mode').to_pylist())
    battery_hours = np.nan_to_num(t.column('battery_hours').to_numpy(), nan=0.0)
    ldes_hours = np.nan_to_num(t.column('ldes_hours').to_numpy(), nan=0.0)

    battery_dispatch_pct = np.where(np.isin(dispatch_mode, ['b', 'bl']), battery_hours, 0.0).astype(np.float64)
    ldes_dispatch_pct = np.where(np.isin(dispatch_mode, ['l', 'bl']), ldes_hours, 0.0).astype(np.float64)

    cols = {
        'iso': pc.cast(t.column('iso'), pa.string()),
        'threshold': pa.array(threshold, type=pa.float64()),
        'clean_firm': pa.array(clean_firm, type=pa.int16()),
        'solar': pa.array(solar, type=pa.int16()),
        'wind': pa.array(wind, type=pa.int16()),
        'hydro': pa.array(hydro, type=pa.int16()),
        'procurement_pct': pa.array(procurement_pct, type=pa.int16()),
        'battery_dispatch_pct': pa.array(battery_dispatch_pct, type=pa.float64()),
        'battery8_dispatch_pct': pa.array(np.zeros(t.num_rows, dtype=np.float64), type=pa.float64()),
        'ldes_dispatch_pct': pa.array(ldes_dispatch_pct, type=pa.float64()),
        # Legacy `lcoe` is stored as ratio, convert to percent score scale.
        'hourly_match_score': pa.array(t.column('lcoe').to_numpy() * 100.0, type=pa.float64()),
        'pareto_type': pa.array([''] * t.num_rows, type=pa.string()),
    }
    return pa.table(cols)


def load_iso_tables():
    """Load PFS data and return a dict of {iso: pyarrow.Table}.

    Tries three sources in order:
      1. data/step1_raw_pfs_parquets/{ISO}_step1_pfs_t{threshold}.parquet (native Step 1)
      2. data/physics_cache_v4_{ISO}.parquet (per-ISO cache)
      3. data/physics_cache_v4.parquet (legacy single merged file, split by ISO)

    Returns dict keyed by ISO name with per-ISO tables (already schema-normalized).
    """
    iso_tables = {}

    # --- Source 1: Step 1 raw parquets (native format only) ---
    if os.path.isdir(STEP1_RAW_DIR):
        parquet_files = sorted(
            os.path.join(STEP1_RAW_DIR, f)
            for f in os.listdir(STEP1_RAW_DIR)
            if f.endswith('.parquet')
        )
        if parquet_files:
            print(f"Scanning Step 1 raw parquets in {STEP1_RAW_DIR}")
            native_by_iso = {}
            for path in parquet_files:
                t = pq.read_table(path)
                fname = os.path.basename(path)
                normalized = None
                if 'clean_firm' in t.column_names and 'hourly_match_score' in t.column_names:
                    normalized = normalize_table(t)
                elif 'source_threshold' in t.column_names and 'mix' in t.column_names:
                    normalized = normalize_legacy_step1_raw_table(t)

                if normalized is None:
                    print(f"  {fname}: SKIPPED (unrecognized Step 1 schema)")
                    continue

                iso_vals = pc.unique(normalized.column('iso')).to_pylist()
                for iso_val in iso_vals:
                    iso_sub = normalized.filter(pc.equal(normalized.column('iso'), iso_val))
                    native_by_iso.setdefault(iso_val, []).append(iso_sub)

                size_mb = os.path.getsize(path) / (1024 * 1024)
                fmt = 'native' if 'clean_firm' in t.column_names else 'legacy-raw'
                print(f"  {fname}: {t.num_rows:>10,} rows ({size_mb:.1f} MB), format={fmt}, isos={len(iso_vals)}")

            for iso, tables in native_by_iso.items():
                combined = pa.concat_tables(tables)
                iso_tables[iso] = normalize_table(combined)
                print(f"  {iso}: {iso_tables[iso].num_rows:,} rows from Step 1 raw")

    # --- Source 2: Per-ISO cache files ---
    for iso in ISOS:
        if iso in iso_tables:
            continue
        iso_path = os.path.join(PFS_DIR, f'physics_cache_v4_{iso}.parquet')
        if os.path.exists(iso_path):
            t = pq.read_table(iso_path)
            iso_tables[iso] = normalize_table(t)
            size_mb = os.path.getsize(iso_path) / (1024 * 1024)
            print(f"  {iso}: {t.num_rows:>10,} rows from per-ISO cache ({size_mb:.1f} MB)")

    # --- Source 3: Legacy single merged file ---
    missing = [iso for iso in ISOS if iso not in iso_tables]
    if missing and os.path.exists(LEGACY_PFS_PATH):
        print(f"Loading legacy PFS for missing ISOs: {', '.join(missing)}")
        legacy = pq.read_table(LEGACY_PFS_PATH)
        for iso in missing:
            mask = pc.equal(legacy.column('iso'), iso)
            subtable = legacy.filter(mask)
            if subtable.num_rows > 0:
                iso_tables[iso] = normalize_table(subtable)
                print(f"  {iso}: {subtable.num_rows:,} rows from legacy cache")

    if not iso_tables:
        raise FileNotFoundError(
            f"No PFS files found. Expected per-ISO files in {PFS_DIR}/ "
            f"(physics_cache_v4_{{ISO}}.parquet) or {STEP1_RAW_DIR}/ "
            f"or legacy {LEGACY_PFS_PATH}"
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


def pareto_procurement(arrays):
    """
    For each unique allocation (CF/Sol/Wnd/Hyd/Bat4/Bat8/LDES), keep only
    the Pareto-optimal (procurement, score) pairs: rows where no other row
    with the same allocation has <= procurement AND >= score.

    Within each allocation group sorted by ascending procurement, this means
    keeping only rows where the score strictly increases (the running max).

    Storage dispatch columns (bat, bat8, ldes) are float64 with 0.05%
    granularity from Step 1. They are scaled by 20x (0.05% -> 1) to produce
    exact integer keys, avoiding truncation that would merge distinct configs.

    Returns indices into the original arrays of rows to keep.
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

    # Scale storage dispatch to integer keys at 0.05% resolution.
    STORAGE_SCALE = 20
    STORAGE_BASE = 2001
    bat_key = np.round(bat * STORAGE_SCALE).astype(np.int64)
    bat8_key = np.round(bat8 * STORAGE_SCALE).astype(np.int64)
    ldes_key = np.round(ldes * STORAGE_SCALE).astype(np.int64)

    # Pack allocation into a single int64 key.
    # Resource columns (cf/sol/wnd/hyd) are int16 0-100 -> base 101.
    # Max key ~ 100 * 101^3 * 2001^3 ~ 8.25e17, fits int64 (max 9.22e18).
    group_key = (cf.astype(np.int64) * (101**3 * STORAGE_BASE**3) +
                 sol.astype(np.int64) * (101**2 * STORAGE_BASE**3) +
                 wnd.astype(np.int64) * (101 * STORAGE_BASE**3) +
                 hyd.astype(np.int64) * (STORAGE_BASE**3) +
                 bat_key * (STORAGE_BASE**2) +
                 bat8_key * STORAGE_BASE +
                 ldes_key)

    # Sort by (allocation, procurement ascending, score descending)
    sort_idx = np.lexsort((-score, proc, group_key))
    sk = group_key[sort_idx]
    sp = proc[sort_idx]
    ss = score[sort_idx]

    # --- Vectorized dedup: keep first row per (group_key, proc) ---
    # After sorting by (group asc, proc asc, score desc), the first row
    # at each (group, proc) has the highest score.
    is_first = np.empty(n, dtype=np.bool_)
    is_first[0] = True
    is_first[1:] = (sk[1:] != sk[:-1]) | (sp[1:] != sp[:-1])

    # Extract the first-at-proc subset
    fap_pos = np.where(is_first)[0]
    fap_scores = ss[fap_pos]
    fap_keys = sk[fap_pos]
    m = len(fap_pos)

    # --- Vectorized Pareto front within each group ---
    # Detect group starts in the first-at-proc array
    gs = np.empty(m, dtype=np.bool_)
    gs[0] = True
    gs[1:] = fap_keys[1:] != fap_keys[:-1]

    keep = pareto_keep_mask(fap_scores, gs)

    # Map back to original indices
    kept_sorted_pos = fap_pos[keep]
    return sort_idx[kept_sorted_pos]


def _pareto_keep_mask_numpy(scores, group_starts):
    """Numpy fallback for environments without numba."""
    m = len(scores)
    keep = np.empty(m, dtype=np.bool_)
    running = -1.0
    for i in range(m):
        if group_starts[i]:
            running = -1.0
        keep_i = scores[i] > running
        keep[i] = keep_i
        if keep_i:
            running = scores[i]
    return keep


if njit is not None:
    @njit(cache=True)
    def pareto_keep_mask(scores, group_starts):
        """JIT-compiled segmented strict-cummax mask for Pareto filtering."""
        m = len(scores)
        keep = np.empty(m, dtype=np.bool_)
        running = -1.0
        for i in range(m):
            if group_starts[i]:
                running = -1.0
            keep_i = scores[i] > running
            keep[i] = keep_i
            if keep_i:
                running = scores[i]
        return keep
else:
    pareto_keep_mask = _pareto_keep_mask_numpy


def process_iso_table(iso, table):
    """Process a single ISO's table: threshold gate + Pareto-optimal procurement."""
    n_input = table.num_rows
    if n_input == 0:
        return None, 0, 0

    # Threshold gate
    table = threshold_gate(table)
    n_gated = table.num_rows

    if n_gated == 0:
        return None, n_input, 0

    # Extract numpy arrays for Pareto computation
    arrays = {
        'clean_firm': table.column('clean_firm').to_numpy(),
        'solar': table.column('solar').to_numpy(),
        'wind': table.column('wind').to_numpy(),
        'hydro': table.column('hydro').to_numpy(),
        'procurement_pct': table.column('procurement_pct').to_numpy(),
        'battery_dispatch_pct': table.column('battery_dispatch_pct').to_numpy(),
        'battery8_dispatch_pct': table.column('battery8_dispatch_pct').to_numpy(),
        'ldes_dispatch_pct': table.column('ldes_dispatch_pct').to_numpy(),
        'hourly_match_score': table.column('hourly_match_score').to_numpy(),
    }

    pareto_idx = pareto_procurement(arrays)
    n_pareto = len(pareto_idx)

    # Build result without threshold column
    result_cols = ['iso', 'clean_firm', 'solar', 'wind', 'hydro',
                   'procurement_pct', 'battery_dispatch_pct',
                   'battery8_dispatch_pct', 'ldes_dispatch_pct',
                   'hourly_match_score']
    if 'pareto_type' in table.column_names:
        result_cols.append('pareto_type')

    result_arrays = {}
    for col_name in result_cols:
        if col_name in table.column_names:
            result_arrays[col_name] = table.column(col_name).take(pareto_idx)

    result = pa.table(result_arrays)
    return result, n_gated, n_pareto


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
    print("  PFS -> PFS post-EF (threshold-free)")
    print("=" * 70)

    total_start = time.time()

    # Load data as per-ISO tables (avoids concat-then-split overhead)
    iso_tables = load_iso_tables()

    # Process each ISO: threshold gate + Pareto-optimal procurement
    print("\nStep 2: Threshold gate + Pareto-optimal procurement (threshold-free)")
    print(f"  {'ISO':>6}  {'Input':>9}  {'Gated':>9}  {'Pareto':>9}  {'Time':>6}")
    print("  " + "-" * 50)

    results_by_iso = {}
    total_gated = 0
    total_pareto = 0

    for iso in ISOS:
        if iso not in iso_tables:
            continue

        t0 = time.time()
        result, n_gated, n_pareto = process_iso_table(iso, iso_tables[iso])
        elapsed = time.time() - t0

        n_input = iso_tables[iso].num_rows
        if result is not None and result.num_rows > 0:
            results_by_iso[iso] = result
            total_gated += n_gated
            total_pareto += n_pareto
            print(f"  {iso:>6}  {n_input:>8,}  {n_gated:>8,}  {n_pareto:>8,}  {elapsed:>5.1f}s")

    total_input = sum(t.num_rows for t in iso_tables.values())
    print(f"\n  Total: {total_input:,} -> {total_gated:,} (gated) -> {total_pareto:,} (Pareto)")
    if total_gated > 0:
        print(f"  Reduction: {(1 - total_pareto/total_gated)*100:.1f}%")

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

    print(f"\n  Total rows: {total_input:,} -> {total_pareto:,} (EF)")
    print(f"  Total time: {elapsed_total:.0f}s")
    print("\n" + "=" * 70)
    print("  STEP 2 COMPLETE — per-ISO EF parquets ready in data/step-2-EF-parquets/")
    print("  Step 3 reads from that directory; no merged output file is written.")
    print("=" * 70)


if __name__ == '__main__':
    main()
