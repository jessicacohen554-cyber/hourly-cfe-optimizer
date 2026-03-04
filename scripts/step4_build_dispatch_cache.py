#!/usr/bin/env python3
"""
Step 5: Build Dispatch Cache
=============================
Pre-computes full 8760-hour dispatch for every unique resource mix across all
ISOs and thresholds. Downstream modules (step5c_compress_day_profiles, step5a_compute_co2,
step5b_compute_lmp_prices) read from cache instead of recomputing independently.

Uses dispatch_utils.reconstruct_hourly_dispatch(detailed=True) to produce
per-resource matched/surplus breakdowns and storage charge profiles needed
by step5c_compress_day_profiles's compressed day profiles.

Pipeline position:
  Step 1 (PFS) → Step 2 (EF) → Step 3 (Cost) → Step 4 (Gas/CCS)
                                                      ↓
                                          step4_build_dispatch_cache.py  ← THIS
                                                      ↓
                            data/step5-post-processing/dispatch_cache/{ISO}_dispatch_cache.parquet
                                                      ↓
                                  +--------+----------+----------+
                                  |        |          |          |
                              step6_cd  step6_sc  step6_co2  step6_lmp

Input:  data/step3-cost-opt-parquets/
Output: data/step5-post-processing/dispatch_cache/{ISO}_dispatch_cache.parquet

Usage:
  python step4_build_dispatch_cache.py                    # All ISOs
  python step4_build_dispatch_cache.py --iso PJM          # Single ISO
  python step4_build_dispatch_cache.py --force            # Rebuild from scratch
  python step4_build_dispatch_cache.py --input-dir PATH   # Custom input
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPT_DIR)

from dispatch_utils import (
    ISOS, RESOURCE_TYPES, CACHE_VERSION, H,
    load_common_data, get_supply_profiles, get_demand_profile,
    build_supply_matrix, reconstruct_hourly_dispatch,
    _archetype_key, load_dispatch_cache, save_dispatch_cache,
)
from parquet_io import find_input_dir, find_parquet, ALL_ISOS


MIX_COLUMNS = [
    'mix_clean_firm', 'mix_solar', 'mix_wind', 'mix_offshore_wind', 'mix_ccs_ccgt', 'mix_hydro',
    'battery_dispatch_pct', 'battery8_dispatch_pct',
    'ldes_dispatch_pct', 'h2_dispatch_pct',
]


def extract_unique_mixes(iso, input_dir):
    """Read step4/step3 parquet for an ISO and extract unique mix tuples.

    Returns list of dicts with keys: resource_pcts,
    battery_dispatch_pct, battery8_dispatch_pct, ldes_dispatch_pct, h2_dispatch_pct.

    Uses vectorized numpy column extraction instead of iterrows() for ~50-100×
    speedup on large DataFrames (iterrows is notoriously slow due to per-row
    Series construction).
    """
    path = find_parquet(input_dir, iso)
    if not path:
        return []

    # Read available columns (h2_dispatch_pct may be absent in older parquets)
    avail_cols = pd.read_parquet(path, columns=[]).columns.tolist()
    read_cols = [c for c in MIX_COLUMNS if c in avail_cols]
    df = pd.read_parquet(path, columns=read_cols)
    # Fill missing columns with 0
    for c in MIX_COLUMNS:
        if c not in df.columns:
            df[c] = 0
    unique = df.drop_duplicates()

    # Vectorized extraction: pull columns as numpy arrays, iterate indices
    cf = unique['mix_clean_firm'].to_numpy(dtype=np.float64)
    sol = unique['mix_solar'].to_numpy(dtype=np.float64)
    wnd = unique['mix_wind'].to_numpy(dtype=np.float64)
    osw = unique['mix_offshore_wind'].to_numpy(dtype=np.float64)
    ccs = unique['mix_ccs_ccgt'].to_numpy(dtype=np.float64)
    hyd = unique['mix_hydro'].to_numpy(dtype=np.float64)
    bat = unique['battery_dispatch_pct'].to_numpy(dtype=np.float64)
    bat8 = unique['battery8_dispatch_pct'].to_numpy(dtype=np.float64)
    ldes = unique['ldes_dispatch_pct'].to_numpy(dtype=np.float64)
    h2 = unique['h2_dispatch_pct'].to_numpy(dtype=np.float64)

    n = len(unique)
    mixes = [None] * n
    for i in range(n):
        mixes[i] = {
            'resource_pcts': {
                'clean_firm': cf[i], 'solar': sol[i], 'wind': wnd[i],
                'offshore_wind': osw[i], 'ccs_ccgt': ccs[i], 'hydro': hyd[i],
            },
            'battery_dispatch_pct': bat[i],
            'battery8_dispatch_pct': bat8[i],
            'ldes_dispatch_pct': ldes[i],
            'h2_dispatch_pct': h2[i],
        }

    return mixes


def build_cache_for_iso(iso, unique_mixes, demand_data, gen_profiles,
                         existing_cache=None, force=False):
    """Compute dispatch for all unique mixes and return updated cache dict.

    Args:
        existing_cache: loaded cache dict. Mixes already in cache are skipped
            unless force=True.
        force: rebuild all mixes even if already cached.

    Returns:
        cache: dict {archetype_key: {field: array}}
        computed: number of newly computed mixes
        skipped: number of mixes already in cache
    """
    supply_profiles = get_supply_profiles(iso, gen_profiles)
    supply_matrix = build_supply_matrix(supply_profiles)
    demand_norm, total_mwh = get_demand_profile(iso, demand_data)

    cache = {} if force else (existing_cache or {})
    computed = 0
    skipped = 0

    for mix_info in unique_mixes:
        rp = mix_info['resource_pcts']
        key = _archetype_key(
            iso, rp,
            100,  # procurement_pct removed (always 100 in v5.0)
            mix_info['battery_dispatch_pct'],
            mix_info['battery8_dispatch_pct'],
            mix_info['ldes_dispatch_pct'],
        )

        if not force and key in cache:
            skipped += 1
            continue

        result = reconstruct_hourly_dispatch(
            demand_norm, supply_profiles, rp,
            100,  # procurement_pct removed (always 100 in v5.0)
            mix_info['battery_dispatch_pct'],
            mix_info['battery8_dispatch_pct'],
            mix_info['ldes_dispatch_pct'],
            supply_matrix=supply_matrix,
            detailed=True,
            h2_dispatch_pct=mix_info['h2_dispatch_pct'],
        )

        cache[key] = {k: v for k, v in result.items()}
        computed += 1

    return cache, computed, skipped


def main():
    parser = argparse.ArgumentParser(
        description='Build comprehensive dispatch cache for all unique mixes.',
    )
    parser.add_argument('--iso', dest='isos', action='append', choices=ALL_ISOS,
                        metavar='ISO',
                        help=f'ISO to process (repeatable). Default: all available.')
    parser.add_argument('--input-dir', type=str, default=None,
                        help='Directory containing step3/step4 parquets.')
    parser.add_argument('--force', action='store_true',
                        help='Rebuild cache from scratch (ignore existing).')
    args = parser.parse_args()

    print("=" * 70)
    print("  Step 5: Build Dispatch Cache")
    print("  Pre-computes 8760-hour dispatch for all unique resource mixes")
    print(f"  Cache version: {CACHE_VERSION}")
    print("=" * 70)

    t0 = time.time()

    # Load common data
    print("\n  Loading demand and generation profiles...")
    demand_data, gen_profiles, _, _ = load_common_data()

    # Find input parquets
    run_isos = args.isos or ALL_ISOS
    input_dir = args.input_dir or find_input_dir(run_isos)
    if not input_dir:
        print("  ERROR: No parquets found. Run Step 3/4 first.")
        sys.exit(1)
    print(f"  Input: {input_dir}")

    total_computed = 0
    total_skipped = 0
    total_mixes = 0

    for iso in run_isos:
        if not find_parquet(input_dir, iso):
            continue

        print(f"\n  {iso}:")

        # Extract unique mixes
        unique_mixes = extract_unique_mixes(iso, input_dir)
        if not unique_mixes:
            print(f"    No mixes found, skipping")
            continue
        total_mixes += len(unique_mixes)
        print(f"    {len(unique_mixes)} unique mixes from parquets")

        # Load existing cache (check version)
        if args.force:
            existing = {}
        else:
            existing = load_dispatch_cache(iso, require_version=CACHE_VERSION)
            if existing:
                print(f"    Existing cache: {len(existing)} archetypes (v{CACHE_VERSION})")

        # Build cache
        iso_t0 = time.time()
        cache, computed, skipped = build_cache_for_iso(
            iso, unique_mixes, demand_data, gen_profiles,
            existing_cache=existing, force=args.force)

        # Save
        save_dispatch_cache(iso, cache, version=CACHE_VERSION)
        iso_elapsed = time.time() - iso_t0

        print(f"    Computed: {computed}, skipped (cached): {skipped}")
        print(f"    Total archetypes: {len(cache)}, time: {iso_elapsed:.1f}s")

        total_computed += computed
        total_skipped += skipped

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  Step 5 Dispatch Cache Complete")
    print(f"  {total_mixes} unique mixes across {len(run_isos)} ISOs")
    print(f"  Computed: {total_computed}, skipped: {total_skipped}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
