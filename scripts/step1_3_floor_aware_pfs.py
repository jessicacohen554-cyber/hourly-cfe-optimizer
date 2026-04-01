#!/usr/bin/env python3
"""
Step 1B2: Floor-Aware PFS Generator for 50-70% Threshold Range
================================================================
Generates physics-feasible mixes that start from the EXISTING clean resource
floor, adding INCREMENTAL resources to reach target thresholds. This produces
mixes with minimal new-build above existing clean, which is critical for
accurate marginal abatement cost (MAC) calculations.

The standard PFS (Step 1B) generates mixes from a neutral starting point,
producing full portfolios that often include large resource allocations
unrelated to the existing grid. For MAC optimization, we need mixes that
represent the cheapest DELTA from what already exists.

Algorithm:
  For each ISO:
    1. Load existing clean floor from GRID_MIX_SHARES
    2. Generate a fine grid (2% step) of resource ADDITIONS above the floor
    3. Score each floor + addition combo using batch_hourly_scores
    4. Assign scored mixes to thresholds 50-70% (with near-miss for higher)
    5. Save to PFS parquet directory for MAC queue consumption

Grid generation:
  - Solar additions: 0 to 80% above existing (2% step)
  - Wind additions: 0 to 80% above existing (2% step)
  - Clean firm additions: 0 to 40% above existing (2% step)
  - Hydro: fixed at existing (capped, $0)
  - Offshore wind additions: 0 to 30% (5% step, only for offshore ISOs)
  - Geothermal additions: 0 to 20% (5% step, CAISO only)

No storage at this level — just resource dispatch shapes.

Input:  EIA-930 demand/generation profiles
Output: data/step1-pfs/{ISO}_t{T}_floor_pfs.parquet (per threshold)

Usage:
  python scripts/step1b2_floor_aware_pfs.py --iso CAISO
  python scripts/step1b2_floor_aware_pfs.py --iso ALL
"""

import argparse
import gc
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from pipeline_config import (
    ISOS, OFFSHORE_ISOS, GEOTHERMAL_ISOS,
    GRID_MIX_SHARES, REGIONAL_DEMAND_TWH, HYDRO_CAP_PCT,
    OFFSHORE_WIND_CAP_TWH, GEOTHERMAL_CAP_TWH,
    PATHS, H,
)
from dispatch_utils import (
    load_common_data,
    get_demand_profile,
    get_supply_profiles,
    build_supply_matrix,
    RESOURCE_TYPES,
)
import step1_pfs_generator as s1
from parquet_utils import write_parquet_chunked, read_parquet_parts

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

# ============================================================================
# CONSTANTS
# ============================================================================

PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'step1-pfs')

# Target thresholds for floor-aware search (50-70% + near-miss into 75-80%)
FLOOR_THRESHOLDS = [50, 55, 60, 65, 70]
NEAR_MISS_THRESHOLDS = [75, 80]  # Also capture mixes that nearly reach these

# Near-miss cache constants (matching step1b_zone_search conventions)
MAX_NEAR_MISS = 100_000  # Max near-miss mixes per ISO

# Grid parameters
RESOURCE_STEP = 2     # 2% step for main resources
FIRM_STEP = 2         # 2% step for clean firm
OSW_STEP = 5          # 5% step for offshore wind
GEO_STEP = 5          # 5% step for geothermal

# Max additions above existing (% of demand)
MAX_SOLAR_ADD = 80
MAX_WIND_ADD = 80
MAX_CF_ADD = 40
MAX_OSW_ADD = 30
MAX_GEO_ADD = 20

# Hybrid resource parameters
HYBRID_STEP = 10        # 10% step for hybrids in floor-aware (coarser to control explosion)
MAX_HYBRID_ADD = 40     # Each hybrid type ≤ 40% (matches HYBRID_MAX_PER_TYPE)

# Memory safety: max rows per generate-score-filter chunk.
# 4M rows × 10 cols × 8 bytes = 320 MB — well within CI's 7 GB RAM.
MAX_CHUNK_ROWS = 4_000_000

# Scoring
CHUNK_SIZE = 20000    # Batch size for hourly scoring


# ============================================================================
# GRID GENERATION
# ============================================================================

def _build_base_grid(iso):
    """Build the base (non-hybrid) resource grid as flat arrays.

    Returns (cf, sol, wnd, osw, geo, hydro_val, has_geo) where each array
    is the absolute resource allocation (floor + addition) after filtering.
    """
    existing = GRID_MIX_SHARES[iso]
    floor_cf = existing.get('clean_firm', 0)
    floor_sol = existing.get('solar', 0)
    floor_wnd = existing.get('wind', 0)
    floor_hyd = existing.get('hydro', 0)
    hydro_val = min(floor_hyd, HYDRO_CAP_PCT.get(iso, 50))

    cf_add = np.arange(0, MAX_CF_ADD + FIRM_STEP, FIRM_STEP, dtype=np.float64)
    sol_add = np.arange(0, MAX_SOLAR_ADD + RESOURCE_STEP, RESOURCE_STEP, dtype=np.float64)
    wnd_add = np.arange(0, MAX_WIND_ADD + RESOURCE_STEP, RESOURCE_STEP, dtype=np.float64)

    if iso in OFFSHORE_ISOS:
        osw_cap_pct = OFFSHORE_WIND_CAP_TWH.get(iso, 0) / REGIONAL_DEMAND_TWH[iso] * 100
        osw_max = min(MAX_OSW_ADD, osw_cap_pct + 5)
        osw_add = np.arange(0, osw_max + OSW_STEP, OSW_STEP, dtype=np.float64)
    else:
        osw_add = np.array([0.0])

    if iso in GEOTHERMAL_ISOS:
        geo_cap_pct = GEOTHERMAL_CAP_TWH / REGIONAL_DEMAND_TWH[iso] * 100
        geo_max = min(MAX_GEO_ADD, geo_cap_pct + 2)
        geo_add = np.arange(0, geo_max + GEO_STEP, GEO_STEP, dtype=np.float64)
    else:
        geo_add = np.array([0.0])

    n_base = len(cf_add) * len(sol_add) * len(wnd_add) * len(osw_add) * len(geo_add)
    print(f"  {iso}: {n_base:,} base combos "
          f"({len(cf_add)} CF × {len(sol_add)} sol × {len(wnd_add)} wnd "
          f"× {len(osw_add)} osw × {len(geo_add)} geo)")

    grids = np.meshgrid(cf_add, sol_add, wnd_add, osw_add, geo_add, indexing='ij')
    flat = [g.ravel() for g in grids]
    cf = floor_cf + flat[0]
    sol = floor_sol + flat[1]
    wnd = floor_wnd + flat[2]
    osw = flat[3]
    geo = flat[4]

    total = cf + sol + wnd + hydro_val + osw + geo
    mask = (total <= 350.0) & (total > 0.1)
    return cf[mask], sol[mask], wnd[mask], osw[mask], geo[mask], hydro_val, iso in GEOTHERMAL_ISOS


def generate_floor_aware_grid(iso, include_hybrids=False):
    """Generate resource mixes starting from existing clean floor.

    Non-hybrid mode uses the original single meshgrid approach.
    Hybrid mode is handled by _process_hybrid_chunked() instead — this
    function is only called for the non-hybrid path.

    Returns:
        mix_batch: numpy array (N, n_resources) of resource allocations (% of demand)
        resource_names: list of resource name strings matching columns
        raw_components: dict of numpy arrays for each resource's original values
    """
    cf_vals, sol_vals, wnd_vals, osw_vals, geo_vals, hydro_val, has_geo = _build_base_grid(iso)

    explicit_sum = cf_vals + sol_vals + wnd_vals + hydro_val + osw_vals + geo_vals
    ccs_vals = np.maximum(0, 100.0 - explicit_sum)

    N = len(cf_vals)
    print(f"  → {N:,} mixes after filtering")

    resource_names = list(RESOURCE_TYPES)
    mix_batch = np.zeros((N, len(resource_names)), dtype=np.float64)
    mix_batch[:, 0] = cf_vals
    mix_batch[:, 1] = sol_vals
    mix_batch[:, 2] = wnd_vals
    mix_batch[:, 3] = osw_vals
    mix_batch[:, 4] = ccs_vals
    mix_batch[:, 5] = hydro_val

    if has_geo and geo_vals.max() > 0:
        mix_batch[:, 0] = cf_vals + geo_vals

    return mix_batch, resource_names, {
        'clean_firm': cf_vals.copy(), 'solar': sol_vals, 'wind': wnd_vals,
        'hydro': np.full(N, hydro_val), 'offshore_wind': osw_vals,
        'geothermal': geo_vals.copy(), 'ccs_ccgt': ccs_vals,
    }


def _process_hybrid_chunked(iso, demand_arr, supply_matrix):
    """Generate, score, and filter hybrid mixes in memory-bounded chunks.

    Instead of building the full 9D Cartesian grid (which would OOM on CI for
    most ISOs), this function:
      1. Builds the base 5D grid as usual (~247K combos)
      2. Pre-computes + pre-filters the hybrid combo grid
      3. For each chunk of base mixes, expands with hybrid combos, applies
         family-budget pruning, scores, and keeps only threshold-relevant results
      4. Concatenates only the small set of kept results at the end

    Peak memory: MAX_CHUNK_ROWS (4M) × 10 cols × 8 bytes ≈ 320 MB for grid +
    scoring intermediates. Well under CI's 7 GB limit.

    Returns (scores, raw_components) matching the assign_and_save interface.
    """
    cf_base, sol_base, wnd_base, osw_base, geo_base, hydro_val, has_geo = _build_base_grid(iso)
    n_base = len(cf_base)
    print(f"  → {n_base:,} base mixes, expanding with hybrids in chunks...")

    sol_cap = s1.SOLAR_FAMILY_CAP.get(iso, 100)
    wnd_cap = s1.WIND_FAMILY_CAP.get(iso, 100)
    hybrid_range = np.arange(0, MAX_HYBRID_ADD + HYBRID_STEP, HYBRID_STEP, dtype=np.float64)
    n_hv = len(hybrid_range)

    # Pre-compute hybrid combo grid (sb4 × sb8 × wb4 × wb8) and pre-filter
    hg = np.meshgrid(hybrid_range, hybrid_range, hybrid_range, hybrid_range, indexing='ij')
    h_sb4, h_sb8, h_wb4, h_wb8 = hg[0].ravel(), hg[1].ravel(), hg[2].ravel(), hg[3].ravel()
    h_mask = ((h_sb4 + h_sb8) <= sol_cap) & ((h_wb4 + h_wb8) <= wnd_cap)
    h_sb4, h_sb8, h_wb4, h_wb8 = h_sb4[h_mask], h_sb8[h_mask], h_wb4[h_mask], h_wb8[h_mask]
    n_h = len(h_sb4)
    print(f"    Hybrid grid: {n_hv}^4 = {n_hv**4:,} → {n_h:,} after pre-filter")

    resource_names = list(RESOURCE_TYPES) + list(s1.HYBRID_TYPES)
    col_map = {rt: i for i, rt in enumerate(resource_names)}
    n_cols = len(resource_names)

    all_thresholds = FLOOR_THRESHOLDS + NEAR_MISS_THRESHOLDS
    min_score = min(all_thresholds) - 2.0
    max_score = max(all_thresholds) + 5.0

    # Accumulators — only threshold-relevant results are kept
    comp_keys = ['clean_firm', 'solar', 'wind', 'hydro', 'offshore_wind',
                 'geothermal', 'ccs_ccgt', 'solar_batt4', 'solar_batt8',
                 'wind_batt4', 'wind_batt8']
    kept = {k: [] for k in comp_keys}
    kept_scores = []
    total_expanded = 0
    total_kept = 0

    base_chunk = max(1, MAX_CHUNK_ROWS // max(1, n_h))

    for cs in range(0, n_base, base_chunk):
        ce = min(cs + base_chunk, n_base)
        cn = ce - cs

        # Expand: each base mix × each hybrid combo via repeat/tile
        c_cf  = np.repeat(cf_base[cs:ce], n_h)
        c_sol = np.repeat(sol_base[cs:ce], n_h)
        c_wnd = np.repeat(wnd_base[cs:ce], n_h)
        c_osw = np.repeat(osw_base[cs:ce], n_h)
        c_geo = np.repeat(geo_base[cs:ce], n_h)
        c_sb4 = np.tile(h_sb4, cn)
        c_sb8 = np.tile(h_sb8, cn)
        c_wb4 = np.tile(h_wb4, cn)
        c_wb8 = np.tile(h_wb8, cn)

        # Family-budget pruning + procurement cap
        mask = ((c_sol + c_sb4 + c_sb8) <= sol_cap)
        mask &= ((c_wnd + c_wb4 + c_wb8) <= wnd_cap)
        total = c_cf + c_sol + c_wnd + c_osw + c_geo + hydro_val + c_sb4 + c_sb8 + c_wb4 + c_wb8
        mask &= (total <= 350.0) & (total > 0.1)

        if mask.sum() == 0:
            continue

        cf = c_cf[mask]; sol = c_sol[mask]; wnd = c_wnd[mask]
        osw = c_osw[mask]; geo = c_geo[mask]
        sb4 = c_sb4[mask]; sb8 = c_sb8[mask]
        wb4 = c_wb4[mask]; wb8 = c_wb8[mask]

        explicit = cf + sol + wnd + hydro_val + osw + geo + sb4 + sb8 + wb4 + wb8
        ccs = np.maximum(0, 100.0 - explicit)

        N = len(cf)
        mix_batch = np.zeros((N, n_cols), dtype=np.float64)
        mix_batch[:, col_map['clean_firm']] = (cf + geo) if has_geo else cf
        mix_batch[:, col_map['solar']] = sol
        mix_batch[:, col_map['wind']] = wnd
        mix_batch[:, col_map['offshore_wind']] = osw
        mix_batch[:, col_map['ccs_ccgt']] = ccs
        mix_batch[:, col_map['hydro']] = hydro_val
        mix_batch[:, col_map['solar_batt4']] = sb4
        mix_batch[:, col_map['solar_batt8']] = sb8
        mix_batch[:, col_map['wind_batt4']] = wb4
        mix_batch[:, col_map['wind_batt8']] = wb8

        scores = score_mixes(mix_batch, demand_arr, supply_matrix)
        total_expanded += N

        keep = (scores >= min_score) & (scores <= max_score)
        nk = keep.sum()
        if nk > 0:
            kept['clean_firm'].append(cf[keep])
            kept['solar'].append(sol[keep])
            kept['wind'].append(wnd[keep])
            kept['hydro'].append(np.full(nk, hydro_val))
            kept['offshore_wind'].append(osw[keep])
            kept['geothermal'].append(geo[keep])
            kept['ccs_ccgt'].append(ccs[keep])
            kept['solar_batt4'].append(sb4[keep])
            kept['solar_batt8'].append(sb8[keep])
            kept['wind_batt4'].append(wb4[keep])
            kept['wind_batt8'].append(wb8[keep])
            kept_scores.append(scores[keep])
            total_kept += nk

        if cs == 0 or (cs // base_chunk) % 5 == 0:
            print(f"    Chunk {cs:,}-{ce:,}/{n_base:,}: "
                  f"{N:,} valid → {nk:,} in threshold range (total kept: {total_kept:,})")

        del mix_batch, scores
        gc.collect()

    print(f"  Expanded {total_expanded:,} → {total_kept:,} in threshold range")

    if total_kept == 0:
        return np.array([]), {}

    raw = {k: np.concatenate(v) for k, v in kept.items() if v}
    return np.concatenate(kept_scores), raw


# ============================================================================
# SCORING & THRESHOLD ASSIGNMENT
# ============================================================================

def score_mixes(mix_batch, demand_arr, supply_matrix):
    """Score all mixes using batch_hourly_scores from step1_pfs_generator."""
    scores = s1.batch_hourly_scores(demand_arr, supply_matrix, mix_batch, CHUNK_SIZE)
    # Normalize: scores are sum(matched) / sum(demand) but returned as raw sums
    # Need to divide by total demand
    total_demand = demand_arr.sum()
    if total_demand > 0:
        scores = scores / total_demand
    return scores * 100.0  # Convert to percentage


def assign_and_save(iso, scores, raw_components, output_dir, include_hybrids=False,
                    thresholds_filter=None):
    """Assign scored mixes to thresholds and save parquets."""
    N = len(scores)
    all_thresholds = FLOOR_THRESHOLDS + NEAR_MISS_THRESHOLDS
    if thresholds_filter:
        all_thresholds = [t for t in all_thresholds if t in thresholds_filter]

    # Output columns: base resource types (excluding ccs_ccgt which is implicit)
    # plus geothermal and hybrids as applicable
    output_cols = ['clean_firm', 'solar', 'wind', 'hydro', 'offshore_wind']
    if iso in GEOTHERMAL_ISOS:
        output_cols.append('geothermal')
    if include_hybrids:
        output_cols.extend(s1.HYBRID_TYPES)

    saved_count = 0
    for threshold in all_thresholds:
        # Feasible: score >= threshold (with small tolerance)
        # Near-miss: score >= threshold - 2
        if threshold in FLOOR_THRESHOLDS:
            # Strict: within [threshold - 0.5, threshold + 5]
            mask = (scores >= threshold - 0.5) & (scores <= threshold + 5.0)
        else:
            # Near-miss for higher thresholds
            mask = (scores >= threshold - 2.0) & (scores <= threshold + 3.0)

        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue

        # Build output structure dynamically
        rows = {
            'iso': [iso] * len(indices),
            'threshold': [float(threshold)] * len(indices),
        }
        for rt in output_cols:
            if rt in raw_components:
                rows[rt] = raw_components[rt][indices]
            else:
                rows[rt] = np.zeros(len(indices), dtype=np.float64)

        # Storage dispatch columns (all zeros at this step)
        rows['battery_dispatch_pct'] = np.zeros(len(indices))
        rows['battery8_dispatch_pct'] = np.zeros(len(indices))
        rows['ldes_dispatch_pct'] = np.zeros(len(indices))
        rows['h2_dispatch_pct'] = np.zeros(len(indices))
        rows['hourly_match_score'] = scores[indices]

        # Save as parquet (chunked if >45 MB)
        if HAS_PYARROW:
            arrays = {k: pa.array(v) for k, v in rows.items()}
            table = pa.table(arrays)
            out_path = os.path.join(output_dir, f'{iso}_t{threshold:g}_floor_pfs.parquet')
            written = write_parquet_chunked(table, out_path, max_mb=45,
                                            compression='snappy')
            print(f"    t{threshold:g}: {len(indices):,} mixes → {len(written)} file(s)")
            saved_count += len(indices)
        else:
            print(f"    t{threshold:g}: {len(indices):,} mixes (pyarrow not available, skipping save)")

    return saved_count


def save_near_miss_cache(iso, scores, raw_components):
    """Append floor-aware near-miss mixes to the shared near-miss cache.

    Near-miss mixes fall below a threshold but are close enough that storage
    dispatch might push them into feasibility. These are especially valuable
    because floor-aware mixes represent minimal new-build paths.

    Schema matches step1b_zone_search: resource columns + base_score.
    """
    if not HAS_PYARROW:
        print("  Near-miss cache: pyarrow not available, skipping")
        return

    # Collect near-miss: score < threshold but >= threshold - width
    # Use same width logic as step1b_zone_search
    all_thresholds = FLOOR_THRESHOLDS + NEAR_MISS_THRESHOLDS
    nm_mask = np.zeros(len(scores), dtype=bool)
    scores_frac = scores / 100.0  # Convert to 0-1 scale for width comparison

    for threshold in all_thresholds:
        t_frac = threshold / 100.0
        # Width: 0.25 for <85%, 0.20 for >=85% (matching zone search)
        width = 0.25 if threshold < 85 else 0.20
        lower = t_frac - width
        # Near-miss = below threshold but within width
        nm_mask |= (scores_frac >= lower) & (scores_frac < t_frac)

    nm_indices = np.where(nm_mask)[0]
    if len(nm_indices) == 0:
        print("  Near-miss cache: 0 mixes in near-miss range")
        return

    # Build near-miss data in zone-search schema
    rtypes = s1.get_resource_types(iso)
    data = {}
    for rt in rtypes:
        if rt in raw_components:
            data[rt] = raw_components[rt][nm_indices].astype(np.float64)
        else:
            data[rt] = np.zeros(len(nm_indices), dtype=np.float64)
    data['base_score'] = scores_frac[nm_indices]  # Store as 0-1 fraction

    # Load existing near-miss and append
    out_path = os.path.join(OUTPUT_DIR, f'{iso}_near_miss.parquet')
    new_table = pa.table(data)

    existing_table = read_parquet_parts(out_path)
    if existing_table is not None:
        existing = existing_table
        # Align columns: new table may have different columns than existing
        # Use existing columns as reference, add missing as zeros
        aligned_arrays = {}
        for col in existing.column_names:
            if col in new_table.column_names:
                aligned_arrays[col] = pa.concat_arrays(
                    [existing.column(col).combine_chunks(), new_table.column(col).combine_chunks()])
            else:
                aligned_arrays[col] = pa.concat_arrays(
                    [existing.column(col).combine_chunks(),
                     pa.array(np.zeros(len(nm_indices), dtype=np.float64))])
        # Add any new columns not in existing
        n_existing = len(existing)
        for col in new_table.column_names:
            if col not in aligned_arrays:
                aligned_arrays[col] = pa.concat_arrays(
                    [pa.array(np.zeros(n_existing, dtype=np.float64)),
                     new_table.column(col).combine_chunks()])
        combined = pa.table(aligned_arrays)
        print(f"  Near-miss cache: {len(nm_indices):,} new + {n_existing:,} existing")
    else:
        combined = new_table
        print(f"  Near-miss cache: {len(nm_indices):,} new mixes (no existing cache)")

    # Deduplicate by resource columns (round to int for matching)
    res_cols = [c for c in combined.column_names if c != 'base_score']
    if len(combined) > 0:
        # Simple dedup: convert to numpy, round, get unique rows
        res_data = np.column_stack([combined.column(c).to_numpy() for c in res_cols])
        rounded = np.round(res_data).astype(np.int64)
        n_res = rounded.shape[1]
        # Use structured row view for dedup (handles any number of columns)
        row_view = np.ascontiguousarray(rounded).view(
            np.dtype((np.void, rounded.dtype.itemsize * n_res))
        ).ravel()
        _, unique_idx = np.unique(row_view, return_index=True)
        unique_idx.sort()

        if len(unique_idx) < len(combined):
            combined = combined.take(unique_idx)
            print(f"  Near-miss cache: deduplicated to {len(combined):,} mixes")

    # Cap at MAX_NEAR_MISS
    if len(combined) > MAX_NEAR_MISS:
        # Keep the highest-scoring near-miss mixes
        base_scores = combined.column('base_score').to_numpy()
        top_idx = np.argsort(base_scores)[-MAX_NEAR_MISS:]
        combined = combined.take(top_idx)
        print(f"  Near-miss cache: capped at {MAX_NEAR_MISS:,}")

    written = write_parquet_chunked(combined, out_path, max_mb=45,
                                    compression='snappy')
    total_mb = sum(os.path.getsize(p) / (1024*1024) for p in written)
    print(f"  Near-miss cache: {len(combined):,} total → {len(written)} file(s) ({total_mb:.1f} MB)")


# ============================================================================
# MAIN
# ============================================================================

def process_iso(iso, demand_data, gen_profiles, include_hybrids=False,
                thresholds_filter=None):
    """Run floor-aware PFS generation for a single ISO."""
    # Auto-detect hybrid mode from coarse cache (supports multi-part files)
    coarse_schema = s1.read_coarse_cache_schema(iso)
    if not include_hybrids and coarse_schema is not None:
        if 'solar_batt4' in coarse_schema.names:
            include_hybrids = True
            print(f"  Auto-detected hybrid columns in coarse cache")

    hybrid_str = " [HYBRID]" if include_hybrids else ""
    print(f"\n{'='*60}")
    print(f"  {iso}: Floor-aware PFS (50-70%){hybrid_str}")
    print(f"{'='*60}")

    # Load profiles
    demand_norm, total_mwh = get_demand_profile(iso, demand_data)
    supply_profiles = get_supply_profiles(iso, gen_profiles)

    # Build supply matrix — base 6D (RESOURCE_TYPES including ccs_ccgt)
    supply_matrix = build_supply_matrix(supply_profiles)

    # Extend supply matrix with hybrid profiles if needed
    if include_hybrids:
        hybrid_profiles = s1.load_hybrid_profiles(iso)
        print(f"  Loaded hybrid profiles: {list(hybrid_profiles.keys())}")
        hybrid_rows = np.stack([
            np.asarray(hybrid_profiles[ht][:H], dtype=np.float64)
            for ht in s1.HYBRID_TYPES
        ])  # (4, 8760)
        supply_matrix = np.vstack([supply_matrix, hybrid_rows])

    demand_arr = demand_norm
    t0 = time.time()

    if include_hybrids:
        # Chunked hybrid path: generate → score → filter per chunk to bound memory
        scores, raw_components = _process_hybrid_chunked(
            iso, demand_arr, supply_matrix)

        if len(scores) == 0:
            print(f"  No mixes in threshold range for {iso}")
            return 0
    else:
        # Original non-hybrid path: single meshgrid
        mix_batch, resource_names, raw_components = generate_floor_aware_grid(iso)
        print(f"  Scoring {len(mix_batch):,} mixes...")
        scores = score_mixes(mix_batch, demand_arr, supply_matrix)

    score_time = time.time() - t0
    print(f"  Scored in {score_time:.1f}s")
    print(f"  Score range: {scores.min():.1f}% - {scores.max():.1f}%")

    for t in FLOOR_THRESHOLDS:
        in_range = ((scores >= t - 0.5) & (scores <= t + 5.0)).sum()
        print(f"    t{t:g}: {in_range:,} feasible")

    saved = assign_and_save(iso, scores, raw_components, OUTPUT_DIR,
                            include_hybrids=include_hybrids,
                            thresholds_filter=thresholds_filter)
    print(f"  Total saved: {saved:,} mixes")

    save_near_miss_cache(iso, scores, raw_components)

    gc.collect()
    return saved


def _parse_thresholds(raw):
    """Parse comma-separated threshold list, return list of floats or None."""
    if not raw or raw.strip().upper() in ('', 'ALL'):
        return None
    parts = [p.strip() for p in raw.split(',') if p.strip()]
    result = []
    for p in parts:
        try:
            result.append(float(p))
        except ValueError:
            print(f"WARNING: Ignoring invalid threshold '{p}'")
    return sorted(set(result)) if result else None


def main():
    parser = argparse.ArgumentParser(description='Floor-Aware PFS Generator (50-80%)')
    parser.add_argument('--iso', type=str, default='ALL',
                        help='ISO to process (default: ALL)')
    parser.add_argument('--hybrid', action='store_true',
                        help='Enable hybrid resource types (solar+batt, wind+batt)')
    parser.add_argument('--thresholds', type=str, default='',
                        help='Comma-separated thresholds to process '
                             '(e.g. "55,60,65"). Default: all (50-80).')
    args = parser.parse_args()

    isos = ISOS if args.iso == 'ALL' else [args.iso]

    thresholds_filter = _parse_thresholds(args.thresholds)
    if thresholds_filter:
        valid = set(FLOOR_THRESHOLDS + NEAR_MISS_THRESHOLDS)
        bad = [t for t in thresholds_filter if t not in valid]
        if bad:
            print(f"WARNING: Thresholds {bad} not in {sorted(valid)} — ignoring")
            thresholds_filter = [t for t in thresholds_filter if t in valid]
        print(f"Threshold filter: {thresholds_filter}")

    if args.hybrid:
        print("Hybrid mode: enabled (CLI flag)")

    print("Loading common data...")
    demand_data, gen_profiles, _, _ = load_common_data()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    t0 = time.time()
    total_saved = 0
    for iso in isos:
        saved = process_iso(iso, demand_data, gen_profiles,
                            include_hybrids=args.hybrid,
                            thresholds_filter=thresholds_filter)
        total_saved += saved

    elapsed = time.time() - t0
    print(f"\nDone. {total_saved:,} total mixes saved in {elapsed:.1f}s")


if __name__ == '__main__':
    main()
