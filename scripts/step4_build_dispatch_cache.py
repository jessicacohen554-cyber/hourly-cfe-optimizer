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
                            data/step4-dispatch-cache/{ISO}_dispatch_cache.parquet
                                                      ↓
                                  +--------+----------+----------+
                                  |        |          |          |
                              step6_cd  step6_sc  step6_co2  step6_lmp

Input:  data/step3-cost-opt-parquets/
Output: data/step4-dispatch-cache/{ISO}_dispatch_cache.parquet

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
import pyarrow.parquet as pq

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPT_DIR)

from dispatch_utils import (
    ISOS, RESOURCE_TYPES, CACHE_VERSION, H,
    DISPATCH_CACHE_DIR,
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
    avail_cols = pq.read_schema(path).names
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

    # Always start fresh — never carry forward stale archetypes from prior runs.
    # Downstream scripts must only see current winners.
    cache = {}
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


def enrich_parquets_with_dispatch_shares(iso, input_dir, cache):
    """Add actual battery/LDES dispatch share columns to step3 parquets.

    For each row in the step3 parquet, looks up the dispatch cache entry,
    sums the 8760-hour dispatch profiles, and computes actual energy dispatched
    as % of annual demand. Writes new columns alongside the existing capacity
    parameters.

    The dispatch profiles in the cache are in demand-normalized units where
    demand sums to 1.0. So sum(profile) * 100 gives % of demand met.
    """
    path = find_parquet(input_dir, iso)
    if not path:
        return 0

    df = pd.read_parquet(path)
    n = len(df)

    # Pre-fill new columns with 0
    batt4_shares = np.zeros(n, dtype=np.float64)
    batt8_shares = np.zeros(n, dtype=np.float64)
    ldes_shares = np.zeros(n, dtype=np.float64)
    h2_shares = np.zeros(n, dtype=np.float64)

    # Extract arrays for archetype key computation
    cf = df['mix_clean_firm'].to_numpy(dtype=np.float64)
    sol = df['mix_solar'].to_numpy(dtype=np.float64)
    wnd = df['mix_wind'].to_numpy(dtype=np.float64)
    osw = df['mix_offshore_wind'].to_numpy(dtype=np.float64) if 'mix_offshore_wind' in df.columns else np.zeros(n)
    ccs = df['mix_ccs_ccgt'].to_numpy(dtype=np.float64)
    hyd = df['mix_hydro'].to_numpy(dtype=np.float64)
    bat = df['battery_dispatch_pct'].to_numpy(dtype=np.float64)
    bat8 = df['battery8_dispatch_pct'].to_numpy(dtype=np.float64)
    ldes_arr = df['ldes_dispatch_pct'].to_numpy(dtype=np.float64)

    # Build unique key → dispatch share mapping (avoid redundant sums)
    key_cache = {}
    hits = 0
    misses = 0

    for i in range(n):
        rp = {
            'clean_firm': cf[i], 'solar': sol[i], 'wind': wnd[i],
            'offshore_wind': osw[i], 'ccs_ccgt': ccs[i], 'hydro': hyd[i],
        }
        key = _archetype_key(iso, rp, 100, bat[i], bat8[i], ldes_arr[i])

        if key in key_cache:
            b4, b8, ld, h2v = key_cache[key]
            batt4_shares[i] = b4
            batt8_shares[i] = b8
            ldes_shares[i] = ld
            h2_shares[i] = h2v
            hits += 1
            continue

        entry = cache.get(key)
        if entry is not None:
            b4 = float(np.sum(entry['battery4_profile'])) * 100
            b8 = float(np.sum(entry['battery8_profile'])) * 100
            ld = float(np.sum(entry['ldes_profile'])) * 100
            h2v = float(np.sum(entry.get('h2_profile', np.zeros(H)))) * 100
            key_cache[key] = (b4, b8, ld, h2v)
            batt4_shares[i] = b4
            batt8_shares[i] = b8
            ldes_shares[i] = ld
            h2_shares[i] = h2v
            hits += 1
        else:
            misses += 1

    # Add columns to DataFrame
    df['battery_dispatch_share'] = np.round(batt4_shares, 2)
    df['battery8_dispatch_share'] = np.round(batt8_shares, 2)
    df['ldes_dispatch_share'] = np.round(ldes_shares, 2)
    df['h2_dispatch_share'] = np.round(h2_shares, 2)

    # Write back
    df.to_parquet(path, index=False, compression='zstd')
    print(f"    Enriched {os.path.basename(path)}: {hits} cache hits, {misses} misses, "
          f"batt4 share max={batt4_shares.max():.2f}%, ldes max={ldes_shares.max():.2f}%")

    # Also enrich the feasible parquet (used by FEASIBLE_MIXES in shared-data.js)
    feas_path = os.path.join(input_dir, f'step3_feasible_{iso}.parquet')
    if os.path.exists(feas_path):
        try:
            _enrich_single_parquet(feas_path, iso, cache)
        except Exception as e:
            print(f"    WARNING: Could not enrich {os.path.basename(feas_path)}: {e}")

    return hits


def _enrich_single_parquet(path, iso, cache):
    """Add dispatch share columns to a single parquet file using the dispatch cache."""
    df = pd.read_parquet(path)
    n = len(df)
    if n == 0:
        return

    batt4_shares = np.zeros(n, dtype=np.float64)
    batt8_shares = np.zeros(n, dtype=np.float64)
    ldes_shares = np.zeros(n, dtype=np.float64)
    h2_shares = np.zeros(n, dtype=np.float64)

    cf = df['clean_firm'].to_numpy(dtype=np.float64) if 'clean_firm' in df.columns else df.get('mix_clean_firm', pd.Series(dtype=float)).to_numpy(dtype=np.float64)
    sol = df['solar'].to_numpy(dtype=np.float64) if 'solar' in df.columns else df.get('mix_solar', pd.Series(dtype=float)).to_numpy(dtype=np.float64)
    wnd = df['wind'].to_numpy(dtype=np.float64) if 'wind' in df.columns else df.get('mix_wind', pd.Series(dtype=float)).to_numpy(dtype=np.float64)
    osw_col = 'offshore_wind' if 'offshore_wind' in df.columns else 'mix_offshore_wind'
    osw = df[osw_col].to_numpy(dtype=np.float64) if osw_col in df.columns else np.zeros(n)
    ccs = df['ccs_ccgt'].to_numpy(dtype=np.float64) if 'ccs_ccgt' in df.columns else df.get('mix_ccs_ccgt', pd.Series(dtype=float)).to_numpy(dtype=np.float64)
    hyd = df['hydro'].to_numpy(dtype=np.float64) if 'hydro' in df.columns else df.get('mix_hydro', pd.Series(dtype=float)).to_numpy(dtype=np.float64)
    bat = df['battery_dispatch_pct'].to_numpy(dtype=np.float64)
    bat8 = df['battery8_dispatch_pct'].to_numpy(dtype=np.float64)
    ldes_arr = df['ldes_dispatch_pct'].to_numpy(dtype=np.float64)

    key_cache_local = {}
    hits = 0

    for i in range(n):
        rp = {
            'clean_firm': cf[i], 'solar': sol[i], 'wind': wnd[i],
            'offshore_wind': osw[i], 'ccs_ccgt': ccs[i], 'hydro': hyd[i],
        }
        key = _archetype_key(iso, rp, 100, bat[i], bat8[i], ldes_arr[i])

        if key in key_cache_local:
            b4, b8, ld, h2v = key_cache_local[key]
        else:
            entry = cache.get(key)
            if entry is not None:
                b4 = float(np.sum(entry['battery4_profile'])) * 100
                b8 = float(np.sum(entry['battery8_profile'])) * 100
                ld = float(np.sum(entry['ldes_profile'])) * 100
                h2v = float(np.sum(entry.get('h2_profile', np.zeros(H)))) * 100
                key_cache_local[key] = (b4, b8, ld, h2v)
                hits += 1
            else:
                b4 = b8 = ld = h2v = 0.0

        batt4_shares[i] = b4
        batt8_shares[i] = b8
        ldes_shares[i] = ld
        h2_shares[i] = h2v

    df['battery_dispatch_share'] = np.round(batt4_shares, 4)
    df['battery8_dispatch_share'] = np.round(batt8_shares, 4)
    df['ldes_dispatch_share'] = np.round(ldes_shares, 4)
    df['h2_dispatch_share'] = np.round(h2_shares, 4)

    df.to_parquet(path, index=False, compression='zstd')
    print(f"    Enriched {os.path.basename(path)}: {n} rows, {hits} unique cache hits")


def build_annual_manifest(iso, unique_mixes, cache):
    """Build annual manifest parquet from dispatch cache.

    Sums 8760h profiles into annual percentages (% of demand).
    One row per unique archetype. This is the single source of truth for
    actual dispatch values — downstream steps (5A/B, 5C, 9A) read from here.

    Capacity columns come from the mix inputs (Step 3 values).
    Dispatch columns are computed from the 8760h cache profiles.
    """
    rows = []

    for mix_info in unique_mixes:
        rp = mix_info['resource_pcts']
        key = _archetype_key(
            iso, rp, 100,
            mix_info['battery_dispatch_pct'],
            mix_info['battery8_dispatch_pct'],
            mix_info['ldes_dispatch_pct'],
        )

        entry = cache.get(key)
        if entry is None:
            continue

        row = {'archetype_key': key}

        # Mix identity — capacity values from Step 3
        row['mix_clean_firm'] = rp.get('clean_firm', 0)
        row['mix_solar'] = rp.get('solar', 0)
        row['mix_wind'] = rp.get('wind', 0)
        row['mix_offshore_wind'] = rp.get('offshore_wind', 0)
        row['mix_ccs_ccgt'] = rp.get('ccs_ccgt', 0)
        row['mix_hydro'] = rp.get('hydro', 0)

        # Storage CAPACITY values (from Step 3 — these are input parameters, not dispatch)
        row['battery_capacity_pct'] = mix_info['battery_dispatch_pct']
        row['battery8_capacity_pct'] = mix_info['battery8_dispatch_pct']
        row['ldes_capacity_pct'] = mix_info['ldes_dispatch_pct']
        row['h2_capacity_pct'] = mix_info.get('h2_dispatch_pct', 0)

        # Demand in cache is normalized to sum=1.0 across 8760 hours.
        # So sum(profile) * 100 gives % of annual demand met.

        # Dispatch (sum of 8760h matched profiles * 100 = % of demand met)
        for resource in ['clean_firm', 'ccs_ccgt', 'solar', 'wind', 'offshore_wind', 'hydro']:
            matched = entry.get(f'matched_{resource}', np.zeros(H))
            surplus = entry.get(f'surplus_{resource}', np.zeros(H))
            row[f'{resource}_dispatch_pct'] = float(np.sum(matched)) * 100
            row[f'{resource}_surplus_pct'] = float(np.sum(surplus)) * 100

        # Storage DISPATCH and CHARGE (actual cycling, not capacity)
        for st, disp_key, charge_key in [
            ('battery', 'battery4_profile', 'battery4_charge'),
            ('battery8', 'battery8_profile', 'battery8_charge'),
            ('ldes', 'ldes_profile', 'ldes_charge'),
            ('h2', 'h2_profile', 'h2_charge'),
        ]:
            disp_profile = entry.get(disp_key, np.zeros(H))
            charge_profile = entry.get(charge_key, np.zeros(H))
            row[f'{st}_dispatch_pct'] = float(np.sum(disp_profile)) * 100
            row[f'{st}_charge_pct'] = float(np.sum(np.abs(charge_profile))) * 100

        # Re-score hourly match from 8760h profiles
        # fossil_displaced = demand matched by clean energy (capped at demand each hour)
        fossil_displaced = entry.get('fossil_displaced', np.zeros(H))
        row['hourly_match_score'] = float(np.sum(fossil_displaced)) * 100

        # Aggregate metrics
        total_clean = entry.get('total_clean', np.zeros(H))
        row['total_clean_dispatch_pct'] = float(np.sum(total_clean)) * 100
        row['total_curtailment_pct'] = float(np.sum(entry.get('curtailed', np.zeros(H)))) * 100
        row['gap_pct'] = float(np.sum(entry.get('residual_demand', np.zeros(H)))) * 100

        rows.append(row)

    if not rows:
        print(f"    WARNING: No manifest rows for {iso}")
        return None

    df = pd.DataFrame(rows)

    # QA/QC checks
    _run_manifest_qa(iso, df)

    # Save
    out_path = os.path.join(DISPATCH_CACHE_DIR, f'{iso}_annual_manifest.parquet')
    df.to_parquet(out_path, index=False, compression='zstd')
    print(f"    Annual manifest: {len(df)} archetypes → {os.path.basename(out_path)}")

    return df


def _run_manifest_qa(iso, df):
    """Run QA/QC checks on the annual manifest."""
    warnings = []

    # 1. Energy balance: gap + match ≈ 100% (demand is normalized)
    # total_clean_dispatch includes surplus, hourly_match_score is capped at demand
    # So: hourly_match_score + gap ≈ 100
    balance = df['hourly_match_score'] + df['gap_pct']
    balance_err = np.abs(balance - 100.0)
    bad_balance = balance_err > 1.0  # 1 percentage point tolerance
    if bad_balance.any():
        n_bad = bad_balance.sum()
        max_err = balance_err.max()
        warnings.append(f"Energy balance: {n_bad} archetypes off by >{1}pp (max {max_err:.2f}pp)")

    # 2. Battery charge × RTE ≈ dispatch (round-trip efficiency check)
    for st, rte in [('battery', 0.85), ('battery8', 0.85), ('ldes', 0.50), ('h2', 0.35)]:
        charge_col = f'{st}_charge_pct'
        disp_col = f'{st}_dispatch_pct'
        if charge_col in df.columns and disp_col in df.columns:
            has_storage = df[charge_col] > 0.001
            if has_storage.any():
                expected_dispatch = df.loc[has_storage, charge_col] * rte
                actual_dispatch = df.loc[has_storage, disp_col]
                ratio = actual_dispatch / expected_dispatch.clip(lower=1e-6)
                # Allow 50% tolerance — window constraints, partial cycles, etc.
                bad_rte = (ratio < 0.3) | (ratio > 2.0)
                if bad_rte.any():
                    n_bad = bad_rte.sum()
                    warnings.append(
                        f"{st} RTE check: {n_bad}/{has_storage.sum()} archetypes "
                        f"outside 0.3-2.0× expected (RTE={rte})"
                    )

    # 3. Dispatch should never exceed capacity × theoretical max cycles
    # 4hr battery: max ~365 cycles/year → dispatch ≤ capacity * 365 * 4/8760 ≈ capacity * 0.167
    # But we just check dispatch > 0 when capacity > 0 (sanity)
    for st in ['battery', 'battery8', 'ldes', 'h2']:
        cap_col = f'{st}_capacity_pct'
        disp_col = f'{st}_dispatch_pct'
        if cap_col in df.columns and disp_col in df.columns:
            has_cap = df[cap_col] > 0
            no_disp = df[disp_col] == 0
            cap_no_disp = has_cap & no_disp
            if cap_no_disp.any():
                n = cap_no_disp.sum()
                if n > len(df) * 0.5:  # Only warn if >50% have this issue
                    warnings.append(f"{st}: {n}/{has_cap.sum()} archetypes have capacity but 0 dispatch")

    # 4. Total dispatch should be reasonable
    max_dispatch = df['total_clean_dispatch_pct'].max()
    if max_dispatch > 200:
        warnings.append(f"Suspicious total clean dispatch: max={max_dispatch:.1f}% of demand")

    # Print results
    if warnings:
        print(f"    QA/QC warnings ({len(warnings)}):")
        for w in warnings:
            print(f"      ⚠ {w}")
    else:
        print(f"    QA/QC passed: {len(df)} archetypes, energy balance OK, RTE checks OK")


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

        # Build cache fresh (no accumulation from prior runs)
        iso_t0 = time.time()
        cache, computed, skipped = build_cache_for_iso(
            iso, unique_mixes, demand_data, gen_profiles,
            force=True)

        # Save (fresh cache, only current winners)
        save_dispatch_cache(iso, cache, version=CACHE_VERSION)
        iso_elapsed = time.time() - iso_t0

        print(f"    Computed: {computed}, skipped (cached): {skipped}")
        print(f"    Total archetypes: {len(cache)}, time: {iso_elapsed:.1f}s")

        # Enrich step3 parquets with actual dispatch shares
        enrich_parquets_with_dispatch_shares(iso, input_dir, cache)

        # Build annual manifest (single source of truth for dispatch values)
        build_annual_manifest(iso, unique_mixes, cache)

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
