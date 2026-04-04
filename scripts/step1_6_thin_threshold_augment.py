#!/usr/bin/env python3
"""
Step 1.6: Thin-Threshold Augmentation
======================================
Lightweight script that adds ~250 EF-quality mixes for any ISO/threshold
band where the current efficient frontier has <100 mixes.

Strategy:
  1. Scan data/step2.1-ef/ to find thin bands (<100 mixes)
  2. For each thin band, generate a 1% fine grid of resource mixes
     targeted at the narrow score window for that threshold
  3. Score mixes using the same model as Step 1 (no CCS backfill)
  4. Save as {ISO}_t{T}_augment.parquet — picked up by Step 2.1 on next run

This is a tack-on script: it writes new PFS parquets that the existing
Step 2.1 pipeline will ingest alongside the existing PFS files. No need
to rerun Steps 1.1–1.5.

Usage:
  python scripts/step1_6_thin_threshold_augment.py              # all thin bands
  python scripts/step1_6_thin_threshold_augment.py --iso NYISO  # one ISO
  python scripts/step1_6_thin_threshold_augment.py --min-target 500  # raise target
"""

import argparse
import gc
import glob
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
    SOLAR_FAMILY_CAP, WIND_FAMILY_CAP, HYBRID_MAX_PER_TYPE,
    H,
)
from dispatch_utils import (
    load_common_data,
    get_demand_profile,
    get_supply_profiles,
)
import step1_pfs_generator as s1

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("ERROR: pyarrow required. pip install pyarrow")
    sys.exit(1)

from parquet_utils import write_parquet_chunked

PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PFS_DIR = os.path.join(PROJECT_ROOT, 'data', 'step1-pfs')
EF_DIR = os.path.join(PROJECT_ROOT, 'data', 'step2.1-ef')

THIN_THRESHOLD = 100   # Augment bands with fewer than this many EF mixes
MIX_TARGET = 250        # Target mix count per band

ALL_THRESHOLDS = [10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 87.5,
                  90, 92.5, 95, 97.5, 99, 99.5, 99.9]

HYBRID_TYPES = s1.HYBRID_TYPES  # ['solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8']

CHUNK_SIZE = 20000


def find_thin_bands(target_isos=None, min_ef=THIN_THRESHOLD):
    """Scan EF directory and return [(iso, threshold, current_count), ...].

    Also detects completely missing thresholds (0 mixes, no EF file).
    """
    thin = []
    existing = set()
    for f in sorted(glob.glob(os.path.join(EF_DIR, 'step_2_1_EF_*.parquet'))):
        if '_batch_' in os.path.basename(f):
            continue
        bn = os.path.basename(f).replace('step_2_1_EF_', '').replace('.parquet', '')
        parts = bn.rsplit('_', 1)
        if len(parts) != 2:
            continue
        iso, thresh_str = parts
        if target_isos and iso not in target_isos:
            continue
        try:
            thresh = float(thresh_str)
        except ValueError:
            continue
        existing.add((iso, thresh))
        n = pq.read_metadata(f).num_rows
        if n < min_ef:
            thin.append((iso, thresh, n))

    # Detect completely missing thresholds (0 mixes = no EF file)
    check_isos = target_isos if target_isos else set(ISOS)
    for iso in check_isos:
        for thresh in ALL_THRESHOLDS:
            if (iso, thresh) not in existing:
                thin.append((iso, thresh, 0))

    return sorted(thin)


def score_mixes(mix_batch, demand_arr, supply_matrix):
    """Score mixes → percentage match (0-100).

    Uses the same scoring as Step 1: no CCS backfill. The mix_batch columns
    must match the supply_matrix columns (from s1.get_resource_types + hybrids).
    """
    scores = s1.batch_hourly_scores(demand_arr, supply_matrix, mix_batch, CHUNK_SIZE)
    total_demand = demand_arr.sum()
    if total_demand > 0:
        scores = scores / total_demand
    return scores * 100.0


def _get_band_window(target_thresh):
    """Return (band_floor, band_ceil) for the target threshold band."""
    sorted_thresholds = sorted(ALL_THRESHOLDS)
    idx = sorted_thresholds.index(target_thresh) if target_thresh in sorted_thresholds else -1
    if idx >= 0 and idx < len(sorted_thresholds) - 1:
        band_ceil = sorted_thresholds[idx + 1]
    else:
        band_ceil = target_thresh + 5.0
    band_floor = target_thresh - 0.5
    return band_floor, band_ceil + 0.5


def generate_augmentation_mixes(iso, target_thresh, demand_arr, supply_matrix, rtypes):
    """Generate a fine grid of resource mixes targeted at a specific threshold band.

    Uses 1% steps for base resources and 5% steps for hybrids. Builds mix_batch
    to match the supply_matrix column ordering (rtypes), which excludes CCS — matching
    how Step 1 scores mixes.

    Returns (scores, raw_components, output_col_names) for mixes in the target band.
    """
    existing = GRID_MIX_SHARES[iso]
    floor_cf = existing.get('clean_firm', 0)
    floor_sol = existing.get('solar', 0)
    floor_wnd = existing.get('wind', 0)
    floor_hyd = existing.get('hydro', 0)
    hydro_val = min(floor_hyd, HYDRO_CAP_PCT.get(iso, 50))
    has_geo = iso in GEOTHERMAL_ISOS
    has_osw = iso in OFFSHORE_ISOS
    include_hybrids = any(ht in rtypes for ht in HYBRID_TYPES)

    # Column index map for the supply matrix
    col_map = {rt: i for i, rt in enumerate(rtypes)}
    n_cols = len(rtypes)

    # Determine grid parameters based on threshold
    if target_thresh <= 55:
        step, firm_step = 1, 1
        max_sol, max_wnd, max_cf = 30, 30, 20
    elif target_thresh <= 70:
        step, firm_step = 1, 1
        max_sol, max_wnd, max_cf = 50, 50, 25
    else:
        step, firm_step = 1, 1
        max_sol, max_wnd, max_cf = 60, 60, 30

    cf_add = np.arange(0, max_cf + firm_step, firm_step, dtype=np.float64)
    sol_add = np.arange(0, max_sol + step, step, dtype=np.float64)
    wnd_add = np.arange(0, max_wnd + step, step, dtype=np.float64)

    # Hydro: sweep from 0 to existing cap (allows sub-floor mixes at low thresholds)
    hydro_step = max(1, int(hydro_val / 8)) if hydro_val > 2 else 1
    if hydro_val < 1:
        hydro_range = np.array([0.0, hydro_val]) if hydro_val > 0 else np.array([0.0])
    else:
        hydro_range = np.arange(0, hydro_val + hydro_step, hydro_step, dtype=np.float64)
        hydro_range = np.clip(hydro_range, 0, hydro_val)
        hydro_range = np.unique(hydro_range)

    if has_osw:
        osw_cap_pct = OFFSHORE_WIND_CAP_TWH.get(iso, 0) / REGIONAL_DEMAND_TWH[iso] * 100
        osw_max = min(20, osw_cap_pct + 5)
        osw_step = 2 if target_thresh > 55 else 3
        osw_add = np.arange(0, osw_max + osw_step, osw_step, dtype=np.float64)
    else:
        osw_add = np.array([0.0])

    if has_geo:
        geo_cap_pct = GEOTHERMAL_CAP_TWH / REGIONAL_DEMAND_TWH[iso] * 100
        geo_max = min(15, geo_cap_pct + 2)
        geo_add = np.arange(0, geo_max + 3, 3, dtype=np.float64)
    else:
        geo_add = np.array([0.0])

    n_base = len(cf_add) * len(sol_add) * len(wnd_add) * len(hydro_range) * len(osw_add) * len(geo_add)

    # Safety: increase step sizes if grid too large
    MAX_BASE = 20_000_000
    while n_base > MAX_BASE and step < 5:
        step += 1
        sol_add = np.arange(0, max_sol + step, step, dtype=np.float64)
        wnd_add = np.arange(0, max_wnd + step, step, dtype=np.float64)
        n_base = len(cf_add) * len(sol_add) * len(wnd_add) * len(hydro_range) * len(osw_add) * len(geo_add)

    print(f"    Base grid: {n_base:,} combos "
          f"({len(cf_add)} CF × {len(sol_add)} sol × {len(wnd_add)} wnd "
          f"× {len(hydro_range)} hyd × {len(osw_add)} osw × {len(geo_add)} geo)")

    # Build meshgrid — sweep CF from 0 to floor+max_cf, all others from 0
    grids = np.meshgrid(cf_add, sol_add, wnd_add, hydro_range, osw_add, geo_add, indexing='ij')
    flat = [g.ravel() for g in grids]
    cf = flat[0]       # clean firm additions (0 to max_cf)
    sol = flat[1]      # solar (0 to max)
    wnd = flat[2]      # wind (0 to max)
    hyd = flat[3]      # hydro (0 to existing cap)
    osw = flat[4]      # offshore wind
    geo = flat[5]      # geothermal

    total = cf + sol + wnd + hyd + osw + geo
    mask = (total <= 350.0) & (total > 0.1)
    cf, sol, wnd, hyd, osw, geo = cf[mask], sol[mask], wnd[mask], hyd[mask], osw[mask], geo[mask]

    N = len(cf)
    print(f"    {N:,} mixes after cap filter")

    # Build mix_batch matching supply_matrix column order (no CCS!)
    mix_batch = np.zeros((N, n_cols), dtype=np.float64)
    mix_batch[:, col_map['clean_firm']] = (cf + geo) if has_geo else cf
    mix_batch[:, col_map['solar']] = sol
    mix_batch[:, col_map['wind']] = wnd
    mix_batch[:, col_map['hydro']] = hyd
    if 'offshore_wind' in col_map:
        mix_batch[:, col_map['offshore_wind']] = osw
    if 'geothermal' in col_map:
        mix_batch[:, col_map['geothermal']] = geo

    scores = score_mixes(mix_batch, demand_arr, supply_matrix)

    band_floor, band_ceil = _get_band_window(target_thresh)
    band_mask = (scores >= band_floor) & (scores < band_ceil)
    n_in_band = band_mask.sum()
    print(f"    {n_in_band:,} base mixes in band [{band_floor:.1f}, {band_ceil:.1f})")

    raw = {
        'clean_firm': cf, 'solar': sol, 'wind': wnd,
        'hydro': hyd, 'offshore_wind': osw, 'geothermal': geo,
    }

    # If base mixes are sufficient, return early
    if n_in_band >= MIX_TARGET * 3:
        return _filter_results(scores, band_mask, raw, {})

    # Expand with hybrids for more diversity
    if not include_hybrids:
        return _filter_results(scores, band_mask, raw, {})

    print(f"    Expanding with hybrids...")
    hybrid_step = 5
    max_hybrid = min(HYBRID_MAX_PER_TYPE, 30)
    h_range = np.arange(0, max_hybrid + hybrid_step, hybrid_step, dtype=np.float64)

    sol_cap = SOLAR_FAMILY_CAP.get(iso, 100)
    wnd_cap = WIND_FAMILY_CAP.get(iso, 100)

    hg = np.meshgrid(h_range, h_range, h_range, h_range, indexing='ij')
    h_sb4, h_sb8, h_wb4, h_wb8 = [g.ravel() for g in hg]
    h_mask = ((h_sb4 + h_sb8) <= sol_cap) & ((h_wb4 + h_wb8) <= wnd_cap)
    h_sb4, h_sb8, h_wb4, h_wb8 = h_sb4[h_mask], h_sb8[h_mask], h_wb4[h_mask], h_wb8[h_mask]
    n_h = len(h_sb4)
    print(f"    Hybrid grid: {n_h:,} combos")

    # Only expand base mixes near the target band
    near_mask = (scores >= target_thresh - 15) & (scores < band_ceil + 10)
    near_idx = np.where(near_mask)[0]
    if len(near_idx) == 0:
        return _filter_results(scores, band_mask, raw, {})

    MAX_EXPAND = 10_000_000
    if len(near_idx) * n_h > MAX_EXPAND:
        rng = np.random.default_rng(42)
        near_idx = rng.choice(near_idx, size=MAX_EXPAND // n_h, replace=False)
        near_idx.sort()

    print(f"    Expanding {len(near_idx):,} near-band × {n_h:,} hybrid combos")

    all_scores = [scores[band_mask]]
    all_raw = {k: [v[band_mask]] for k, v in raw.items()}
    for ht in HYBRID_TYPES:
        all_raw[ht] = [np.zeros(int(band_mask.sum()))]

    CHUNK = max(1, MAX_EXPAND // n_h)
    for cs in range(0, len(near_idx), CHUNK):
        ce = min(cs + CHUNK, len(near_idx))
        idx_chunk = near_idx[cs:ce]
        cn = len(idx_chunk)

        c_cf = np.repeat(cf[idx_chunk], n_h)
        c_sol = np.repeat(sol[idx_chunk], n_h)
        c_wnd = np.repeat(wnd[idx_chunk], n_h)
        c_hyd = np.repeat(hyd[idx_chunk], n_h)
        c_osw = np.repeat(osw[idx_chunk], n_h)
        c_geo = np.repeat(geo[idx_chunk], n_h)
        c_sb4 = np.tile(h_sb4, cn)
        c_sb8 = np.tile(h_sb8, cn)
        c_wb4 = np.tile(h_wb4, cn)
        c_wb8 = np.tile(h_wb8, cn)

        fmask = ((c_sol + c_sb4 + c_sb8) <= sol_cap)
        fmask &= ((c_wnd + c_wb4 + c_wb8) <= wnd_cap)
        total_h = c_cf + c_sol + c_wnd + c_hyd + c_osw + c_geo + c_sb4 + c_sb8 + c_wb4 + c_wb8
        fmask &= (total_h <= 350.0)

        if fmask.sum() == 0:
            continue

        c_cf = c_cf[fmask]; c_sol = c_sol[fmask]; c_wnd = c_wnd[fmask]
        c_hyd = c_hyd[fmask]; c_osw = c_osw[fmask]; c_geo = c_geo[fmask]
        c_sb4 = c_sb4[fmask]; c_sb8 = c_sb8[fmask]
        c_wb4 = c_wb4[fmask]; c_wb8 = c_wb8[fmask]

        M = len(c_cf)
        mb = np.zeros((M, n_cols), dtype=np.float64)
        mb[:, col_map['clean_firm']] = (c_cf + c_geo) if has_geo else c_cf
        mb[:, col_map['solar']] = c_sol
        mb[:, col_map['wind']] = c_wnd
        mb[:, col_map['hydro']] = c_hyd
        if 'offshore_wind' in col_map:
            mb[:, col_map['offshore_wind']] = c_osw
        if 'geothermal' in col_map:
            mb[:, col_map['geothermal']] = c_geo
        for ht_col, arr in [('solar_batt4', c_sb4), ('solar_batt8', c_sb8),
                             ('wind_batt4', c_wb4), ('wind_batt8', c_wb8)]:
            if ht_col in col_map:
                mb[:, col_map[ht_col]] = arr

        h_scores = score_mixes(mb, demand_arr, supply_matrix)
        h_band = (h_scores >= band_floor) & (h_scores < band_ceil)
        nk = h_band.sum()

        if nk > 0:
            all_scores.append(h_scores[h_band])
            all_raw['clean_firm'].append(c_cf[h_band])
            all_raw['solar'].append(c_sol[h_band])
            all_raw['wind'].append(c_wnd[h_band])
            all_raw['hydro'].append(c_hyd[h_band])
            all_raw['offshore_wind'].append(c_osw[h_band])
            all_raw['geothermal'].append(c_geo[h_band])
            all_raw['solar_batt4'].append(c_sb4[h_band])
            all_raw['solar_batt8'].append(c_sb8[h_band])
            all_raw['wind_batt4'].append(c_wb4[h_band])
            all_raw['wind_batt8'].append(c_wb8[h_band])

        del mb, h_scores
        gc.collect()

    final_scores = np.concatenate(all_scores)
    final_raw = {k: np.concatenate(v) for k, v in all_raw.items()}
    print(f"    Total mixes in band: {len(final_scores):,}")
    return final_scores, final_raw


def _filter_results(scores, mask, raw, hybrid_raw):
    """Filter base results to band and add zero hybrid columns."""
    filtered = {k: v[mask] for k, v in raw.items()}
    n = int(mask.sum())
    for ht in HYBRID_TYPES:
        filtered[ht] = hybrid_raw.get(ht, np.zeros(n))
    return scores[mask], filtered


def save_augment_parquet(iso, thresh, scores, raw):
    """Save augmentation mixes to {ISO}_t{T}_augment.parquet."""
    os.makedirs(PFS_DIR, exist_ok=True)
    N = len(scores)

    output_cols = ['clean_firm', 'solar', 'wind', 'hydro', 'offshore_wind']
    if iso in GEOTHERMAL_ISOS:
        output_cols.append('geothermal')
    output_cols.extend(HYBRID_TYPES)

    rows = {
        'iso': [iso] * N,
        'threshold': [float(thresh)] * N,
    }
    for col in output_cols:
        rows[col] = raw.get(col, np.zeros(N))
    rows['battery_dispatch_pct'] = np.zeros(N)
    rows['battery8_dispatch_pct'] = np.zeros(N)
    rows['ldes_dispatch_pct'] = np.zeros(N)
    rows['h2_dispatch_pct'] = np.zeros(N)
    rows['hourly_match_score'] = scores

    table = pa.table({k: pa.array(v) for k, v in rows.items()})

    thresh_str = f'{thresh:g}'
    out_path = os.path.join(PFS_DIR, f'{iso}_t{thresh_str}_augment.parquet')
    written = write_parquet_chunked(table, out_path, max_mb=45, compression='snappy')
    total_kb = sum(os.path.getsize(f) for f in written) / 1024
    if len(written) == 1:
        print(f"    Saved {out_path} ({N:,} mixes, {total_kb:.0f} KB)")
    else:
        print(f"    Saved {len(written)} parts ({N:,} mixes, {total_kb:.0f} KB total)")
    return written


def main():
    parser = argparse.ArgumentParser(description='Augment thin EF threshold bands')
    parser.add_argument('--iso', type=str, default=None,
                        help='Target ISO (default: all thin bands)')
    parser.add_argument('--min-ef', type=int, default=THIN_THRESHOLD,
                        help=f'Augment bands with fewer than this many EF mixes (default: {THIN_THRESHOLD})')
    parser.add_argument('--min-target', type=int, default=MIX_TARGET,
                        help=f'Target mix count per band (default: {MIX_TARGET})')
    parser.add_argument('--min-thresh', type=float, default=0,
                        help='Skip thresholds below this value (default: 0)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Just list thin bands, don\'t generate')
    args = parser.parse_args()

    target_isos = {args.iso} if args.iso else None
    thin_bands = find_thin_bands(target_isos, args.min_ef)
    if args.min_thresh > 0:
        thin_bands = [(iso, t, n) for iso, t, n in thin_bands if t >= args.min_thresh]

    if not thin_bands:
        print("No thin bands found — all ISO/threshold pairs have >= "
              f"{args.min_ef} EF mixes.")
        return

    print(f"Found {len(thin_bands)} thin bands (<{args.min_ef} EF mixes):")
    for iso, thresh, n in thin_bands:
        print(f"  {iso:6s} t={thresh:5.1f}: {n:5d} current EF mixes")

    if args.dry_run:
        return

    print("\nLoading demand/generation profiles...")
    demand_data, gen_profiles, _, _ = load_common_data()

    from collections import defaultdict
    by_iso = defaultdict(list)
    for iso, thresh, n in thin_bands:
        by_iso[iso].append((thresh, n))

    total_saved = 0
    for iso in sorted(by_iso):
        bands = by_iso[iso]
        print(f"\n{'='*60}")
        print(f"Processing {iso} ({len(bands)} thin bands)")
        print(f"{'='*60}")

        # Use the same resource types + supply matrix as Step 1 (no CCS!)
        rtypes = s1.get_resource_types(iso, include_hybrids=True)
        demand_norm = get_demand_profile(iso, demand_data)
        supply_profiles = get_supply_profiles(iso, gen_profiles, include_hybrids=True)

        # CAISO: add geothermal as flat year-round profile (matches step1_pfs_generator)
        if iso == 'CAISO':
            supply_profiles['geothermal'] = np.full(H, 1.0 / H, dtype=np.float64)

        # prepare_numpy_profiles returns (demand_arr, supply_matrix)
        # matching rtypes column ordering
        demand_arr, supply_matrix = s1.prepare_numpy_profiles(
            iso, demand_norm[0], supply_profiles,
            include_hybrids=True,
            hybrid_profiles=s1.load_hybrid_profiles(iso))

        for thresh, current_n in sorted(bands):
            print(f"\n  --- {iso} t={thresh} (current: {current_n} EF mixes) ---")
            t0 = time.time()

            scores, raw = generate_augmentation_mixes(
                iso, thresh, demand_arr, supply_matrix, rtypes)

            if len(scores) == 0:
                print(f"    No mixes found in target band — skipping")
                continue

            save_augment_parquet(iso, thresh, scores, raw)
            total_saved += len(scores)
            print(f"    Done in {time.time() - t0:.1f}s")

        gc.collect()

    print(f"\n{'='*60}")
    print(f"Augmentation complete: {total_saved:,} total mixes across "
          f"{len(thin_bands)} bands")
    print(f"Next: rerun Step 2.1 to incorporate augmented mixes into the EF")


if __name__ == '__main__':
    main()
