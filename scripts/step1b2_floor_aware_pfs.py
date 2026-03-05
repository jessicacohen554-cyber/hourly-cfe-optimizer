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
Output: data/step1-pfs-parquets/{ISO}_t{T}_floor_pfs.parquet (per threshold)

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
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'step1-pfs-parquets')

# Target thresholds for floor-aware search (50-70% + near-miss into 75-80%)
FLOOR_THRESHOLDS = [50, 55, 60, 65, 70]
NEAR_MISS_THRESHOLDS = [75, 80]  # Also capture mixes that nearly reach these

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

# Scoring
CHUNK_SIZE = 20000    # Batch size for hourly scoring


# ============================================================================
# GRID GENERATION
# ============================================================================

def generate_floor_aware_grid(iso):
    """Generate resource mixes starting from existing clean floor.

    Returns:
        mix_batch: numpy array (N, n_resources) of resource allocations (% of demand)
        resource_names: list of resource name strings matching columns
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

    # Count combos
    n_combos = len(cf_additions) * len(sol_additions) * len(wnd_additions) * len(osw_additions) * len(geo_additions)
    print(f"  {iso}: {n_combos:,} floor-aware combos "
          f"({len(cf_additions)} CF × {len(sol_additions)} sol × {len(wnd_additions)} wnd "
          f"× {len(osw_additions)} osw × {len(geo_additions)} geo)")

    # Build resource names matching the supply_matrix order
    # RESOURCE_TYPES = ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro']
    resource_names = list(RESOURCE_TYPES)  # ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro']

    # Add geothermal dimension if applicable
    has_geo = iso in GEOTHERMAL_ISOS

    # Generate grid via meshgrid
    grids = np.meshgrid(cf_additions, sol_additions, wnd_additions, osw_additions, geo_additions,
                        indexing='ij')
    flat = [g.ravel() for g in grids]
    cf_vals = floor_cf + flat[0]
    sol_vals = floor_sol + flat[1]
    wnd_vals = floor_wnd + flat[2]
    osw_vals = flat[3]  # offshore wind is all new-build (no existing)
    geo_vals = flat[4]  # geothermal is all new-build

    # CCS residual: implicit (100 - sum of explicit resources)
    # For scoring purposes, CCS = max(0, 100 - cf - sol - wnd - hyd - osw - geo)
    explicit_sum = cf_vals + sol_vals + wnd_vals + hydro_val + osw_vals + geo_vals
    ccs_vals = np.maximum(0, 100.0 - explicit_sum)

    # Filter: total procurement < 350% (same as step1a)
    total = cf_vals + sol_vals + wnd_vals + hydro_val + osw_vals + geo_vals
    mask = total <= 350.0
    # Also filter: at least some generation
    mask &= total > 0.1

    cf_vals = cf_vals[mask]
    sol_vals = sol_vals[mask]
    wnd_vals = wnd_vals[mask]
    osw_vals = osw_vals[mask]
    geo_vals = geo_vals[mask]
    ccs_vals = ccs_vals[mask]

    N = len(cf_vals)
    print(f"  → {N:,} mixes after filtering")

    # Build mix_batch matching RESOURCE_TYPES order:
    # ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro']
    mix_batch = np.zeros((N, len(resource_names)), dtype=np.float64)
    mix_batch[:, 0] = cf_vals       # clean_firm
    mix_batch[:, 1] = sol_vals      # solar
    mix_batch[:, 2] = wnd_vals      # wind
    mix_batch[:, 3] = osw_vals      # offshore_wind
    mix_batch[:, 4] = ccs_vals      # ccs_ccgt (implicit residual)
    mix_batch[:, 5] = hydro_val     # hydro (constant)

    # For geothermal: need to pass as additional resource in supply matrix
    # Since geothermal is part of clean_firm in the supply profiles,
    # we handle it by adding geo to clean_firm for scoring purposes
    if has_geo and geo_vals.max() > 0:
        # Geothermal adds to the clean_firm profile (flat baseload)
        # In step1, geothermal is treated as a separate flat baseload
        # For scoring, we add it to clean_firm
        mix_batch[:, 0] = cf_vals + geo_vals  # clean_firm includes geothermal for scoring

    # Store original component values for output
    raw_cf = cf_vals.copy()
    raw_geo = geo_vals.copy()

    return mix_batch, resource_names, {
        'clean_firm': raw_cf,
        'solar': sol_vals,
        'wind': wnd_vals,
        'hydro': np.full(N, hydro_val),
        'offshore_wind': osw_vals,
        'geothermal': raw_geo,
        'ccs_ccgt': ccs_vals,
    }


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


def assign_and_save(iso, scores, raw_components, output_dir):
    """Assign scored mixes to thresholds and save parquets."""
    N = len(scores)
    all_thresholds = FLOOR_THRESHOLDS + NEAR_MISS_THRESHOLDS

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

        # Build output DataFrame-like structure
        rows = {
            'iso': [iso] * len(indices),
            'threshold': [float(threshold)] * len(indices),
            'clean_firm': raw_components['clean_firm'][indices],
            'solar': raw_components['solar'][indices],
            'wind': raw_components['wind'][indices],
            'hydro': raw_components['hydro'][indices],
            'offshore_wind': raw_components['offshore_wind'][indices],
            'battery_dispatch_pct': np.zeros(len(indices)),
            'battery8_dispatch_pct': np.zeros(len(indices)),
            'ldes_dispatch_pct': np.zeros(len(indices)),
            'h2_dispatch_pct': np.zeros(len(indices)),
            'hourly_match_score': scores[indices],
        }

        if iso in GEOTHERMAL_ISOS:
            rows['geothermal'] = raw_components['geothermal'][indices]

        # Save as parquet
        if HAS_PYARROW:
            arrays = {k: pa.array(v) for k, v in rows.items()}
            table = pa.table(arrays)
            out_path = os.path.join(output_dir, f'{iso}_t{threshold:g}_floor_pfs.parquet')
            pq.write_table(table, out_path)
            print(f"    t{threshold:g}: {len(indices):,} mixes → {out_path}")
            saved_count += len(indices)
        else:
            print(f"    t{threshold:g}: {len(indices):,} mixes (pyarrow not available, skipping save)")

    return saved_count


# ============================================================================
# MAIN
# ============================================================================

def process_iso(iso, demand_data, gen_profiles):
    """Run floor-aware PFS generation for a single ISO."""
    print(f"\n{'='*60}")
    print(f"  {iso}: Floor-aware PFS (50-70%)")
    print(f"{'='*60}")

    # Load profiles
    demand_norm, total_mwh = get_demand_profile(iso, demand_data)
    supply_profiles = get_supply_profiles(iso, gen_profiles)
    supply_matrix = build_supply_matrix(supply_profiles)

    # Convert demand to matching format
    demand_arr = demand_norm  # Already normalized

    # Generate floor-aware grid
    t0 = time.time()
    mix_batch, resource_names, raw_components = generate_floor_aware_grid(iso)

    # Score all mixes
    print(f"  Scoring {len(mix_batch):,} mixes...")
    scores = score_mixes(mix_batch, demand_arr, supply_matrix)

    score_time = time.time() - t0
    print(f"  Scored in {score_time:.1f}s")
    print(f"  Score range: {scores.min():.1f}% - {scores.max():.1f}%")
    print(f"  Score mean: {scores.mean():.1f}%, median: {np.median(scores):.1f}%")

    # Distribution across thresholds
    for t in FLOOR_THRESHOLDS:
        in_range = ((scores >= t - 0.5) & (scores <= t + 5.0)).sum()
        print(f"    t{t:g}: {in_range:,} feasible")

    # Assign to thresholds and save
    saved = assign_and_save(iso, scores, raw_components, OUTPUT_DIR)
    print(f"  Total saved: {saved:,} mixes")

    gc.collect()
    return saved


def main():
    parser = argparse.ArgumentParser(description='Floor-Aware PFS Generator (50-70%)')
    parser.add_argument('--iso', type=str, default='ALL',
                        help='ISO to process (default: ALL)')
    args = parser.parse_args()

    isos = ISOS if args.iso == 'ALL' else [args.iso]

    print("Loading common data...")
    demand_data, gen_profiles, _, _ = load_common_data()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    t0 = time.time()
    total_saved = 0
    for iso in isos:
        saved = process_iso(iso, demand_data, gen_profiles)
        total_saved += saved

    elapsed = time.time() - t0
    print(f"\nDone. {total_saved:,} total mixes saved in {elapsed:.1f}s")


if __name__ == '__main__':
    main()
