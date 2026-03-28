#!/usr/bin/env python3
"""
Step 1B3: Fine-Grid PFS Generator for 40-70% Threshold Range
================================================================
Generates physics-feasible mixes on a 1% fine grid for mid-range thresholds
(40-70%). This fills the gap in the standard PFS which was originally designed
for 70%+ and has coarse coverage in the 40-70% range.

Uses a floor-aware approach: starts from existing clean resources and adds
incremental new-build. The finer 1% grid produces more diverse mixes than
step1b2's 2% grid, improving MAC optimization at lower thresholds.

Algorithm:
  For each ISO:
    1. Load existing clean floor from GRID_MIX_SHARES
    2. Generate a 1% fine grid of resource ADDITIONS above the floor
    3. Score each combo using batch_hourly_scores
    4. Assign scored mixes to thresholds 40-70% (with near-miss for 75%)
    5. Save to PFS parquet directory

Grid generation:
  - Solar additions: 0 to 60% above existing (1% step)
  - Wind additions: 0 to 60% above existing (1% step)
  - Clean firm additions: 0 to 30% above existing (1% step)
  - Hydro: fixed at existing (capped, $0)
  - Offshore wind additions: 0 to 20% (2% step, only for offshore ISOs)
  - Geothermal additions: 0 to 15% (3% step, CAISO only)

No storage at this level — just resource dispatch shapes.
Storage is handled by step1c at 50%+ thresholds.

Input:  EIA-930 demand/generation profiles
Output: data/step1-pfs/{ISO}_t{T}_fine_pfs.parquet (per threshold)

Usage:
  python scripts/step1b3_fine_grid_pfs.py --iso CAISO
  python scripts/step1b3_fine_grid_pfs.py --iso ALL
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

# Target thresholds for fine-grid search
FINE_THRESHOLDS = [40, 45, 50, 55, 60, 65, 70]
NEAR_MISS_THRESHOLDS = [75]  # Also capture mixes that nearly reach 75%

# Near-miss cache constants (matching step1b_zone_search conventions)
MAX_NEAR_MISS = 100_000  # Max near-miss mixes per ISO

# Grid parameters — 1% step for main resources
RESOURCE_STEP = 1     # 1% step for solar/wind
FIRM_STEP = 1         # 1% step for clean firm
OSW_STEP = 2          # 2% step for offshore wind
GEO_STEP = 3          # 3% step for geothermal

# Max additions above existing (% of demand)
MAX_SOLAR_ADD = 60
MAX_WIND_ADD = 60
MAX_CF_ADD = 30
MAX_OSW_ADD = 20
MAX_GEO_ADD = 15

# Hybrid resource parameters
HYBRID_STEP = 5         # 5% step for hybrids in fine grid
MAX_HYBRID_ADD = 30     # Each hybrid type ≤ 30% (tighter than floor-aware)

# Scoring
CHUNK_SIZE = 20000    # Batch size for hourly scoring


# ============================================================================
# GRID GENERATION
# ============================================================================

def generate_fine_grid(iso, include_hybrids=False):
    """Generate resource mixes on 1% fine grid starting from existing clean floor.

    Returns:
        mix_batch: numpy array (N, n_resources) of resource allocations (% of demand)
        resource_names: list of resource name strings matching columns
        raw_components: dict of numpy arrays for each resource's original values
    """
    existing = GRID_MIX_SHARES[iso]

    # Fixed floor values
    floor_cf = existing.get('clean_firm', 0)
    floor_sol = existing.get('solar', 0)
    floor_wnd = existing.get('wind', 0)
    floor_hyd = existing.get('hydro', 0)

    # Hydro is always existing only (capped)
    hydro_val = min(floor_hyd, HYDRO_CAP_PCT.get(iso, 50))

    # Resource addition ranges
    cf_additions = np.arange(0, MAX_CF_ADD + FIRM_STEP, FIRM_STEP, dtype=np.float64)
    sol_additions = np.arange(0, MAX_SOLAR_ADD + RESOURCE_STEP, RESOURCE_STEP, dtype=np.float64)
    wnd_additions = np.arange(0, MAX_WIND_ADD + RESOURCE_STEP, RESOURCE_STEP, dtype=np.float64)

    # Offshore wind (only for ISOs with offshore resource)
    if iso in OFFSHORE_ISOS:
        osw_cap_pct = OFFSHORE_WIND_CAP_TWH.get(iso, 0) / REGIONAL_DEMAND_TWH[iso] * 100
        osw_max = min(MAX_OSW_ADD, osw_cap_pct + 5)
        osw_additions = np.arange(0, osw_max + OSW_STEP, OSW_STEP, dtype=np.float64)
    else:
        osw_additions = np.array([0.0])

    # Geothermal (CAISO only)
    if iso in GEOTHERMAL_ISOS:
        geo_cap_pct = GEOTHERMAL_CAP_TWH / REGIONAL_DEMAND_TWH[iso] * 100
        geo_max = min(MAX_GEO_ADD, geo_cap_pct + 2)
        geo_additions = np.arange(0, geo_max + GEO_STEP, GEO_STEP, dtype=np.float64)
    else:
        geo_additions = np.array([0.0])

    # Hybrid resource additions
    if include_hybrids:
        sb4_additions = np.arange(0, MAX_HYBRID_ADD + HYBRID_STEP, HYBRID_STEP, dtype=np.float64)
        sb8_additions = np.arange(0, MAX_HYBRID_ADD + HYBRID_STEP, HYBRID_STEP, dtype=np.float64)
        wb4_additions = np.arange(0, MAX_HYBRID_ADD + HYBRID_STEP, HYBRID_STEP, dtype=np.float64)
        wb8_additions = np.arange(0, MAX_HYBRID_ADD + HYBRID_STEP, HYBRID_STEP, dtype=np.float64)
    else:
        sb4_additions = np.array([0.0])
        sb8_additions = np.array([0.0])
        wb4_additions = np.array([0.0])
        wb8_additions = np.array([0.0])

    # Count combos
    all_additions = [cf_additions, sol_additions, wnd_additions, osw_additions, geo_additions,
                     sb4_additions, sb8_additions, wb4_additions, wb8_additions]
    n_combos = 1
    for a in all_additions:
        n_combos *= len(a)

    hybrid_str = ""
    if include_hybrids:
        hybrid_str = (f" × {len(sb4_additions)} sb4 × {len(sb8_additions)} sb8"
                      f" × {len(wb4_additions)} wb4 × {len(wb8_additions)} wb8")
    print(f"  {iso}: {n_combos:,} fine-grid combos "
          f"({len(cf_additions)} CF × {len(sol_additions)} sol × {len(wnd_additions)} wnd "
          f"× {len(osw_additions)} osw × {len(geo_additions)} geo{hybrid_str})")

    # OOM protection: increase step sizes if grid too large
    MAX_BATCH = 50_000_000  # 50M max at once to avoid OOM
    if n_combos > MAX_BATCH:
        print(f"  WARNING: {n_combos:,} combos exceeds {MAX_BATCH:,} limit. "
              f"Increasing step sizes.")
        # Fall back to 2% grid for base resources
        sol_additions = np.arange(0, MAX_SOLAR_ADD + 2, 2, dtype=np.float64)
        wnd_additions = np.arange(0, MAX_WIND_ADD + 2, 2, dtype=np.float64)
        cf_additions = np.arange(0, MAX_CF_ADD + 2, 2, dtype=np.float64)
        if include_hybrids:
            # Increase hybrid step to 10%
            sb4_additions = np.arange(0, MAX_HYBRID_ADD + 10, 10, dtype=np.float64)
            sb8_additions = np.arange(0, MAX_HYBRID_ADD + 10, 10, dtype=np.float64)
            wb4_additions = np.arange(0, MAX_HYBRID_ADD + 10, 10, dtype=np.float64)
            wb8_additions = np.arange(0, MAX_HYBRID_ADD + 10, 10, dtype=np.float64)
        all_additions = [cf_additions, sol_additions, wnd_additions, osw_additions, geo_additions,
                         sb4_additions, sb8_additions, wb4_additions, wb8_additions]
        n_combos = 1
        for a in all_additions:
            n_combos *= len(a)
        print(f"  → Reduced to {n_combos:,} combos with increased step")

    # Build resource names matching the supply_matrix order:
    # base 6D RESOURCE_TYPES + hybrid extensions
    resource_names = list(RESOURCE_TYPES)  # ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro']
    if include_hybrids:
        resource_names.extend(s1.HYBRID_TYPES)  # + ['solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8']

    has_geo = iso in GEOTHERMAL_ISOS

    # Generate grid via meshgrid
    grids = np.meshgrid(*all_additions, indexing='ij')
    flat = [g.ravel() for g in grids]
    cf_vals = floor_cf + flat[0]
    sol_vals = floor_sol + flat[1]
    wnd_vals = floor_wnd + flat[2]
    osw_vals = flat[3]  # offshore wind is all new-build (no existing)
    geo_vals = flat[4]  # geothermal is all new-build

    # Hybrid values
    sb4_vals = flat[5] if include_hybrids else np.zeros_like(cf_vals)
    sb8_vals = flat[6] if include_hybrids else np.zeros_like(cf_vals)
    wb4_vals = flat[7] if include_hybrids else np.zeros_like(cf_vals)
    wb8_vals = flat[8] if include_hybrids else np.zeros_like(cf_vals)

    # CCS residual: implicit (100 - sum of explicit resources including hybrids)
    explicit_sum = (cf_vals + sol_vals + wnd_vals + hydro_val + osw_vals + geo_vals +
                    sb4_vals + sb8_vals + wb4_vals + wb8_vals)
    ccs_vals = np.maximum(0, 100.0 - explicit_sum)

    # Filter: total procurement < 350% (same as step1a)
    total = explicit_sum
    mask = total <= 350.0
    # Also filter: at least some generation
    mask &= total > 0.1

    # Family cap constraints for hybrids
    if include_hybrids:
        solar_fam_cap = s1.SOLAR_FAMILY_CAP.get(iso, 100)
        wind_fam_cap = s1.WIND_FAMILY_CAP.get(iso, 100)
        solar_family = sol_vals + sb4_vals + sb8_vals
        wind_family = wnd_vals + wb4_vals + wb8_vals
        mask &= solar_family <= solar_fam_cap
        mask &= wind_family <= wind_fam_cap

    cf_vals = cf_vals[mask]
    sol_vals = sol_vals[mask]
    wnd_vals = wnd_vals[mask]
    osw_vals = osw_vals[mask]
    geo_vals = geo_vals[mask]
    ccs_vals = ccs_vals[mask]
    sb4_vals = sb4_vals[mask]
    sb8_vals = sb8_vals[mask]
    wb4_vals = wb4_vals[mask]
    wb8_vals = wb8_vals[mask]

    N = len(cf_vals)
    print(f"  → {N:,} mixes after filtering")

    # Build mix_batch matching resource_names order from get_resource_types()
    mix_batch = np.zeros((N, len(resource_names)), dtype=np.float64)
    res_vals = {
        'clean_firm': cf_vals, 'solar': sol_vals, 'wind': wnd_vals,
        'offshore_wind': osw_vals, 'ccs_ccgt': ccs_vals, 'hydro': np.full(N, hydro_val),
        'geothermal': geo_vals,
        'solar_batt4': sb4_vals, 'solar_batt8': sb8_vals,
        'wind_batt4': wb4_vals, 'wind_batt8': wb8_vals,
    }
    for i, rt in enumerate(resource_names):
        if rt in res_vals:
            mix_batch[:, i] = res_vals[rt]

    # For geothermal scoring: add to clean_firm (both flat baseload profiles)
    if has_geo and geo_vals.max() > 0:
        cf_idx = resource_names.index('clean_firm')
        mix_batch[:, cf_idx] = cf_vals + geo_vals

    # Store original component values for output
    raw_components = {
        'clean_firm': cf_vals.copy(),
        'solar': sol_vals,
        'wind': wnd_vals,
        'hydro': np.full(N, hydro_val),
        'offshore_wind': osw_vals,
        'geothermal': geo_vals.copy(),
        'ccs_ccgt': ccs_vals,
    }
    if include_hybrids:
        raw_components['solar_batt4'] = sb4_vals
        raw_components['solar_batt8'] = sb8_vals
        raw_components['wind_batt4'] = wb4_vals
        raw_components['wind_batt8'] = wb8_vals

    return mix_batch, resource_names, raw_components


# ============================================================================
# SCORING & THRESHOLD ASSIGNMENT
# ============================================================================

def score_mixes(mix_batch, demand_arr, supply_matrix):
    """Score all mixes using batch_hourly_scores from step1_pfs_generator."""
    scores = s1.batch_hourly_scores(demand_arr, supply_matrix, mix_batch, CHUNK_SIZE)
    total_demand = demand_arr.sum()
    if total_demand > 0:
        scores = scores / total_demand
    return scores * 100.0  # Convert to percentage


def assign_and_save(iso, scores, raw_components, output_dir, include_hybrids=False):
    """Assign scored mixes to thresholds and save parquets."""
    N = len(scores)
    all_thresholds = FINE_THRESHOLDS + NEAR_MISS_THRESHOLDS

    # Output columns: base resource types (excluding ccs_ccgt which is implicit)
    # plus geothermal and hybrids as applicable
    output_cols = ['clean_firm', 'solar', 'wind', 'hydro', 'offshore_wind']
    if iso in GEOTHERMAL_ISOS:
        output_cols.append('geothermal')
    if include_hybrids:
        output_cols.extend(s1.HYBRID_TYPES)

    saved_count = 0
    for threshold in all_thresholds:
        # Feasible: within [threshold - 0.5, threshold + 5]
        if threshold in FINE_THRESHOLDS:
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

        # Geothermal if applicable (and not already in output_cols)
        if iso in GEOTHERMAL_ISOS and 'geothermal' not in rows:
            rows['geothermal'] = raw_components['geothermal'][indices]

        # Save as parquet
        if HAS_PYARROW:
            arrays = {k: pa.array(v) for k, v in rows.items()}
            table = pa.table(arrays)
            out_path = os.path.join(output_dir, f'{iso}_t{threshold:g}_fine_pfs.parquet')
            pq.write_table(table, out_path)
            print(f"    t{threshold:g}: {len(indices):,} mixes → {out_path}")
            saved_count += len(indices)
        else:
            print(f"    t{threshold:g}: {len(indices):,} mixes (pyarrow not available, skipping save)")

    return saved_count


def save_near_miss_cache(iso, scores, raw_components):
    """Append fine-grid near-miss mixes to the shared near-miss cache.

    Near-miss mixes fall below a threshold but are close enough that storage
    dispatch might push them into feasibility. Fine-grid mixes at 1% resolution
    provide dense near-miss coverage that improves step1c storage refinement.

    Schema matches step1b_zone_search: resource columns + base_score.
    """
    if not HAS_PYARROW:
        print("  Near-miss cache: pyarrow not available, skipping")
        return

    # Collect near-miss: score < threshold but >= threshold - width
    all_thresholds = FINE_THRESHOLDS + NEAR_MISS_THRESHOLDS
    nm_mask = np.zeros(len(scores), dtype=bool)
    scores_frac = scores / 100.0  # Convert to 0-1 scale

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

    if os.path.exists(out_path):
        existing = pq.read_table(out_path)
        # Align columns: new table may have different columns than existing
        aligned_arrays = {}
        for col in existing.column_names:
            if col in new_table.column_names:
                aligned_arrays[col] = pa.concat_arrays(
                    [existing.column(col), new_table.column(col)])
            else:
                aligned_arrays[col] = pa.concat_arrays(
                    [existing.column(col),
                     pa.array(np.zeros(len(nm_indices), dtype=np.float64))])
        n_existing = len(existing)
        for col in new_table.column_names:
            if col not in aligned_arrays:
                aligned_arrays[col] = pa.concat_arrays(
                    [pa.array(np.zeros(n_existing, dtype=np.float64)),
                     new_table.column(col)])
        combined = pa.table(aligned_arrays)
        print(f"  Near-miss cache: {len(nm_indices):,} new + {n_existing:,} existing")
    else:
        combined = new_table
        print(f"  Near-miss cache: {len(nm_indices):,} new mixes (no existing cache)")

    # Deduplicate by resource columns
    res_cols = [c for c in combined.column_names if c != 'base_score']
    if len(combined) > 0:
        res_data = np.column_stack([combined.column(c).to_numpy() for c in res_cols])
        rounded = np.round(res_data).astype(np.int64)
        n_res = rounded.shape[1]
        multipliers = np.array([301**i for i in range(n_res)], dtype=np.int64)
        keys = (rounded * multipliers).sum(axis=1)
        _, unique_idx = np.unique(keys, return_index=True)
        unique_idx.sort()

        if len(unique_idx) < len(combined):
            combined = combined.take(unique_idx)
            print(f"  Near-miss cache: deduplicated to {len(combined):,} mixes")

    # Cap at MAX_NEAR_MISS
    if len(combined) > MAX_NEAR_MISS:
        base_scores = combined.column('base_score').to_numpy()
        top_idx = np.argsort(base_scores)[-MAX_NEAR_MISS:]
        combined = combined.take(top_idx)
        print(f"  Near-miss cache: capped at {MAX_NEAR_MISS:,}")

    pq.write_table(combined, out_path, compression='snappy')
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  Near-miss cache: {len(combined):,} total → {out_path} ({size_mb:.1f} MB)")


# ============================================================================
# MAIN
# ============================================================================

def process_iso(iso, demand_data, gen_profiles, include_hybrids=False):
    """Run fine-grid PFS generation for a single ISO."""
    # Auto-detect hybrid mode from coarse cache
    coarse_path = os.path.join(OUTPUT_DIR, f'{iso}_coarse_cache.parquet')
    if not include_hybrids and HAS_PYARROW and os.path.exists(coarse_path):
        coarse_schema = pq.read_schema(coarse_path)
        if 'solar_batt4' in coarse_schema.names:
            include_hybrids = True
            print(f"  Auto-detected hybrid columns in coarse cache")

    hybrid_str = " [HYBRID]" if include_hybrids else ""
    print(f"\n{'='*60}")
    print(f"  {iso}: Fine-grid PFS (40-70%){hybrid_str}")
    print(f"{'='*60}")

    # Load profiles
    demand_norm, total_mwh = get_demand_profile(iso, demand_data)
    supply_profiles = get_supply_profiles(iso, gen_profiles)

    # Build supply matrix — base 6D (RESOURCE_TYPES including ccs_ccgt)
    supply_matrix = build_supply_matrix(supply_profiles)

    # Extend supply matrix with hybrid profiles if needed
    hybrid_profiles = None
    if include_hybrids:
        hybrid_profiles = s1.load_hybrid_profiles(iso)
        print(f"  Loaded hybrid profiles: {list(hybrid_profiles.keys())}")
        # Append hybrid profile rows to supply matrix
        hybrid_rows = np.stack([
            np.asarray(hybrid_profiles[ht][:H], dtype=np.float64)
            for ht in s1.HYBRID_TYPES
        ])  # (4, 8760)
        supply_matrix = np.vstack([supply_matrix, hybrid_rows])

    demand_arr = demand_norm

    # Generate fine grid
    t0 = time.time()
    mix_batch, resource_names, raw_components = generate_fine_grid(
        iso, include_hybrids=include_hybrids)

    # Score all mixes
    print(f"  Scoring {len(mix_batch):,} mixes...")
    scores = score_mixes(mix_batch, demand_arr, supply_matrix)

    score_time = time.time() - t0
    print(f"  Scored in {score_time:.1f}s")
    print(f"  Score range: {scores.min():.1f}% - {scores.max():.1f}%")
    print(f"  Score mean: {scores.mean():.1f}%, median: {np.median(scores):.1f}%")

    # Distribution across thresholds
    for t in FINE_THRESHOLDS:
        in_range = ((scores >= t - 0.5) & (scores <= t + 5.0)).sum()
        print(f"    t{t:g}: {in_range:,} feasible")

    # Assign to thresholds and save PFS parquets
    saved = assign_and_save(iso, scores, raw_components, OUTPUT_DIR,
                            include_hybrids=include_hybrids)
    print(f"  Total saved: {saved:,} mixes")

    # Write near-miss mixes to shared cache for step1c
    save_near_miss_cache(iso, scores, raw_components)

    gc.collect()
    return saved


def main():
    parser = argparse.ArgumentParser(description='Fine-Grid PFS Generator (40-70%)')
    parser.add_argument('--iso', type=str, default='ALL',
                        help='ISO to process (default: ALL)')
    parser.add_argument('--hybrid', action='store_true',
                        help='Enable hybrid resource types (solar+batt, wind+batt)')
    args = parser.parse_args()

    isos = ISOS if args.iso == 'ALL' else [args.iso]

    if args.hybrid:
        print("Hybrid mode: enabled (CLI flag)")

    print("Loading common data...")
    demand_data, gen_profiles, _, _ = load_common_data()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    t0 = time.time()
    total_saved = 0
    for iso in isos:
        saved = process_iso(iso, demand_data, gen_profiles,
                            include_hybrids=args.hybrid)
        total_saved += saved

    elapsed = time.time() - t0
    print(f"\nDone. {total_saved:,} total mixes saved in {elapsed:.1f}s")


if __name__ == '__main__':
    main()
