#!/usr/bin/env python3
"""
Step 0: Fetch offshore wind profiles from NREL NOW-23 dataset.

Uses the NREL Wind Toolkit API to download hourly wind speeds at 140m and 160m
from the NOW-23 dataset, interpolates to 150m hub height, applies the IEA 15 MW
reference turbine power curve, and generates normalized 8760 capacity factor
profiles per ISO.

Data source: NREL NOW-23 (2km WRF, hourly, 2000-2020)
Turbine: IEA 15 MW (150m hub, 240m rotor, 3-25 m/s operating range)

Requires: NREL API key (set NREL_API_KEY env var or pass --api-key)
Sign up: https://developer.nrel.gov/signup/

Output: data/offshore_wind_profiles.json
  Format: {iso: {year_str: {"offshore_wind": [8760 normalized floats]}}}
  Normalization: sum = 1.0 (consistent with all other generation profiles)
"""

import argparse
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

# ─── Configuration ───────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
OUTPUT_PATH = os.path.join(DATA_DIR, 'offshore_wind_profiles.json')

# ISOs with offshore wind resource (others get zero)
OFFSHORE_ISOS = ['NYISO', 'NEISO', 'PJM', 'CAISO']

# Representative lease area coordinates (lon, lat) — longitude first for WKT
# These are centroids of the primary BOEM lease areas per ISO, verified to be
# well offshore (in the NOW-23 ocean grid domain, not on land).
LEASE_AREA_COORDS = {
    'NYISO': (-72.75, 40.08),   # NY Bight — Empire Wind / Sunrise Wind area (~20mi S of Long Island)
    'NEISO': (-70.60, 41.05),   # Vineyard Wind 1 area (S of Martha's Vineyard)
    'PJM':   (-73.40, 39.10),   # NJ Atlantic Shores / Ocean Wind (~20mi E of Atlantic City)
    'CAISO': (-121.60, 35.50),  # Morro Bay WEA (floating, ~30mi offshore)
}

# NOW-23 regional API endpoint for each ISO
API_ENDPOINTS = {
    'NYISO': 'offshore-mid-atlantic-download',
    'NEISO': 'offshore-north-atlantic-download',
    'PJM':   'offshore-mid-atlantic-download',
    'CAISO': 'offshore-ca-download',
}

BASE_URL = 'https://developer.nrel.gov/api/wind-toolkit/v2/wind'

# Years to fetch (5-year average, matching EIA onshore methodology: 2016-2020)
FETCH_YEARS = [2016, 2017, 2018, 2019, 2020]

# Loss stack (multiplicative)
WAKE_LOSS = 0.90          # 10% wake losses
ELECTRICAL_LOSS = 0.975   # 2.5% electrical losses
AVAILABILITY = 0.95       # 5% availability losses
NET_LOSS_FACTOR = WAKE_LOSS * ELECTRICAL_LOSS * AVAILABILITY  # ~0.834

# ─── IEA 15 MW Reference Turbine Power Curve ────────────────────────────────
# Source: NREL turbine-models package (Offshore/IEA_Reference_15MW_240.csv)
# 59 data points from 3-25 m/s. Power in kW, rated = 15000 kW.
# Below cut-in (3 m/s): 0 kW. Above cut-out (25 m/s): 0 kW.

IEA_15MW_WIND_SPEEDS = np.array([
    3.0, 3.5, 4.0, 4.5, 4.75, 5.0, 5.25, 6.0, 6.2, 6.4,
    6.5, 6.55, 6.6, 6.7, 6.8, 6.9, 6.92, 6.93, 6.94, 6.95,
    6.96, 6.97, 6.98, 6.99, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5,
    10.0, 10.25, 10.5, 10.6, 10.7, 10.72, 10.74, 10.76, 10.78, 10.784,
    10.786, 10.787, 10.788, 10.789, 10.7895, 10.8, 10.9, 11.0, 11.25, 11.5,
    11.75, 12.0, 13.0, 14.0, 15.0, 17.5, 20.0, 22.5, 25.0
], dtype=np.float64)

IEA_15MW_POWER_KW = np.array([
    70.02, 301.99, 595.09, 964.89, 1185.08, 1429.22, 1695.25, 2656.26,
    2957.22, 3275.74, 3442.67, 3528.65, 3614.99, 3791.16, 3971.97,
    4155.59, 4192.39, 4210.77, 4228.84, 4247.19, 4265.46, 4283.90,
    4301.95, 4320.29, 4339.30, 5338.82, 6481.12, 7774.57, 9229.23,
    10855.04, 12661.25, 13638.15, 14660.66, 14994.85, 14994.65, 14994.61,
    14994.55, 14994.52, 14994.55, 14994.53, 14994.53, 14994.53, 14994.52,
    14994.52, 14994.59, 14994.54, 14994.43, 14994.27, 14994.02, 14994.11,
    14994.18, 14994.17, 14994.76, 14994.76, 14994.76, 14994.83, 14994.83,
    14996.27, 14997.63
], dtype=np.float64)

RATED_POWER_KW = 15000.0
CUT_IN = 3.0
CUT_OUT = 25.0


def wind_speed_to_cf(wind_speeds: np.ndarray) -> np.ndarray:
    """Convert wind speed array to capacity factor using IEA 15MW power curve.

    Args:
        wind_speeds: Array of wind speeds in m/s (8760,)

    Returns:
        Capacity factor array (0-1) with loss stack applied (8760,)
    """
    # Interpolate power from the curve
    power_kw = np.interp(wind_speeds, IEA_15MW_WIND_SPEEDS, IEA_15MW_POWER_KW,
                         left=0.0, right=0.0)

    # Zero outside operating range
    power_kw = np.where(wind_speeds < CUT_IN, 0.0, power_kw)
    power_kw = np.where(wind_speeds > CUT_OUT, 0.0, power_kw)

    # Convert to capacity factor and apply loss stack
    cf = (power_kw / RATED_POWER_KW) * NET_LOSS_FACTOR

    return cf


def interpolate_hub_height(ws_140m: np.ndarray, ws_160m: np.ndarray,
                           target_height: float = 150.0) -> np.ndarray:
    """Linear interpolation between 140m and 160m to target hub height.

    For the small 20m span, linear interpolation is sufficiently accurate.
    Log-law would give virtually identical results over this range.
    """
    fraction = (target_height - 140.0) / (160.0 - 140.0)  # 0.5 for 150m
    return ws_140m + fraction * (ws_160m - ws_140m)


def fetch_year(iso: str, year: int, api_key: str) -> pd.DataFrame:
    """Fetch hourly wind speeds at 140m and 160m for one ISO/year.

    Uses CSV direct download (single point, single year).
    Returns DataFrame with columns: windspeed_140m, windspeed_160m
    """
    endpoint = API_ENDPOINTS[iso]
    lon, lat = LEASE_AREA_COORDS[iso]
    url = f"{BASE_URL}/{endpoint}.csv"

    params = {
        'api_key': api_key,
        'wkt': f'POINT ({lon} {lat})',
        'attributes': 'windspeed_140m,windspeed_160m',
        'names': str(year),
        'utc': 'true',
        'leap_day': 'true',
        'interval': '60',
        'email': os.environ.get('NREL_API_EMAIL', 'cfe.optimizer@gmail.com'),
    }

    # Retry with exponential backoff on transient network errors only.
    # 400/422 errors are not transient — log and raise immediately.
    for attempt in range(4):
        try:
            response = requests.get(url, params=params, timeout=120)
            if response.status_code in (400, 422):
                # Log response body for diagnosis — NREL returns specific error details
                body = response.text[:1000] if response.text else "(empty)"
                raise RuntimeError(
                    f"NREL API {response.status_code} for {iso}/{year}. "
                    f"URL: {response.url}\n"
                    f"Response body: {body}"
                )
            response.raise_for_status()
            break
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt == 3:
                raise RuntimeError(f"Failed after 4 attempts for {iso}/{year}: {e}")
            wait = 2 ** (attempt + 1)
            print(f"  Retry {attempt + 1}/4 for {iso}/{year} in {wait}s...")
            time.sleep(wait)

    # Parse CSV response — NREL CSVs have metadata header rows
    lines = response.text.strip().split('\n')

    # Find the actual data header (contains 'windspeed' or 'Year')
    header_idx = 0
    for i, line in enumerate(lines):
        if 'windspeed' in line.lower() or 'wind speed' in line.lower():
            header_idx = i
            break
        if line.startswith('Year') or line.startswith('year'):
            header_idx = i
            break

    csv_text = '\n'.join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(csv_text))

    # Standardize column names
    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if '140' in cl and 'speed' in cl:
            col_map[col] = 'windspeed_140m'
        elif '160' in cl and 'speed' in cl:
            col_map[col] = 'windspeed_160m'
    df = df.rename(columns=col_map)

    if 'windspeed_140m' not in df.columns or 'windspeed_160m' not in df.columns:
        raise ValueError(
            f"Expected windspeed_140m and windspeed_160m columns for {iso}/{year}. "
            f"Got: {list(df.columns)}"
        )

    return df[['windspeed_140m', 'windspeed_160m']]


def process_to_8760(df: pd.DataFrame) -> np.ndarray:
    """Extract exactly 8760 hourly wind speed values from a possibly longer DataFrame.

    Handles leap year (8784 hours) by removing Feb 29.
    """
    ws_140 = df['windspeed_140m'].values.astype(np.float64)
    ws_160 = df['windspeed_160m'].values.astype(np.float64)

    # If leap year (8784 rows), remove Feb 29 (hours 1416-1439, 0-indexed)
    if len(ws_140) > 8760:
        # Feb 29 in UTC is hours 1416-1439 (day 59, 0-indexed from Jan 1 00:00)
        leap_start = 59 * 24  # hour 1416
        leap_end = 60 * 24    # hour 1440
        ws_140 = np.delete(ws_140, slice(leap_start, leap_end))
        ws_160 = np.delete(ws_160, slice(leap_start, leap_end))

    # Truncate/pad to exactly 8760
    ws_140 = ws_140[:8760]
    ws_160 = ws_160[:8760]
    if len(ws_140) < 8760:
        ws_140 = np.pad(ws_140, (0, 8760 - len(ws_140)), mode='edge')
        ws_160 = np.pad(ws_160, (0, 8760 - len(ws_160)), mode='edge')

    # Interpolate to 150m hub height
    ws_150 = interpolate_hub_height(ws_140, ws_160)

    return ws_150


def build_profile(iso: str, api_key: str) -> dict:
    """Build multi-year offshore wind profiles for one ISO.

    Returns: {year_str: {"offshore_wind": [8760 normalized floats]}}
    """
    result = {}
    for year in FETCH_YEARS:
        print(f"  Fetching {iso} {year}...")
        df = fetch_year(iso, year, api_key)
        ws_150 = process_to_8760(df)

        # Apply power curve → capacity factor with losses
        cf = wind_speed_to_cf(ws_150)

        # Normalize to sum = 1.0 (consistent with all gen profiles)
        total = cf.sum()
        if total > 0:
            profile = (cf / total).tolist()
        else:
            profile = [0.0] * 8760

        result[str(year)] = {'offshore_wind': profile}

        # Stats
        annual_cf = cf.mean()
        zero_hours = np.sum(cf == 0)
        print(f"    Annual CF: {annual_cf:.3f}, Zero-gen hours: {zero_hours} ({zero_hours/87.6:.1f}%)")

        # Rate limit: 1 request/second
        time.sleep(1.1)

    return result


def build_profiles_from_cache(iso: str, cache_dir: str) -> dict:
    """Build profiles from pre-downloaded CSV files (for offline/CI use).

    Expects files named: {cache_dir}/{iso}_{year}.csv
    Each CSV should have windspeed_140m and windspeed_160m columns.
    """
    result = {}
    for year in FETCH_YEARS:
        csv_path = os.path.join(cache_dir, f"{iso}_{year}.csv")
        if not os.path.exists(csv_path):
            print(f"  SKIP {iso} {year} — no cached file at {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        ws_150 = process_to_8760(df)
        cf = wind_speed_to_cf(ws_150)

        total = cf.sum()
        if total > 0:
            profile = (cf / total).tolist()
        else:
            profile = [0.0] * 8760

        result[str(year)] = {'offshore_wind': profile}
        annual_cf = cf.mean()
        print(f"  {iso} {year}: CF={annual_cf:.3f}")

    return result


def validate_profiles(all_profiles: dict):
    """Validate offshore wind profiles against known benchmarks."""
    print("\n--- Validation ---")
    for iso, years in all_profiles.items():
        yearly_cfs = []
        for year_str, fuels in years.items():
            profile = np.array(fuels['offshore_wind'])
            # Denormalize: profile × 8760 gives relative CF per hour
            cf_hourly = profile * 8760
            annual_cf = cf_hourly.mean()
            yearly_cfs.append(annual_cf)

        avg_cf = np.mean(yearly_cfs)
        print(f"  {iso}: avg CF = {avg_cf:.3f} (target: 0.40-0.52)")

        # Validation checks
        if avg_cf < 0.30 or avg_cf > 0.60:
            print(f"    WARNING: CF outside expected range for offshore wind!")
        if iso in ['NYISO', 'NEISO', 'PJM'] and avg_cf < 0.40:
            print(f"    WARNING: Atlantic offshore CF unexpectedly low (expected 0.44-0.52)")
        if iso == 'CAISO' and avg_cf < 0.35:
            print(f"    WARNING: Pacific offshore CF unexpectedly low (expected 0.40-0.48)")

    # Compare with South Fork Wind actual (46.4% CF in first year)
    print(f"\n  Benchmark: South Fork Wind actual = 46.4% CF")
    print(f"  Benchmark: NREL ATB 2024 offshore = ~49% CF")


def main():
    parser = argparse.ArgumentParser(description='Fetch offshore wind profiles from NREL NOW-23')
    parser.add_argument('--api-key', type=str, default=None,
                        help='NREL API key (or set NREL_API_KEY env var)')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Directory with pre-downloaded CSV files (offline mode)')
    parser.add_argument('--isos', type=str, nargs='+', default=OFFSHORE_ISOS,
                        help='ISOs to fetch (default: all offshore ISOs)')
    parser.add_argument('--output', type=str, default=OUTPUT_PATH,
                        help=f'Output JSON path (default: {OUTPUT_PATH})')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be fetched without making API calls')
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get('NREL_API_KEY')

    if args.dry_run:
        print("DRY RUN — would fetch:")
        for iso in args.isos:
            lon, lat = LEASE_AREA_COORDS[iso]
            endpoint = API_ENDPOINTS[iso]
            for year in FETCH_YEARS:
                print(f"  {iso} {year}: {endpoint} @ POINT({lon}, {lat})")
        print(f"\nTotal API calls: {len(args.isos) * len(FETCH_YEARS)}")
        return

    if not api_key and not args.cache_dir:
        print("ERROR: Must provide --api-key, set NREL_API_KEY env var, or use --cache-dir")
        print("Sign up for free API key at: https://developer.nrel.gov/signup/")
        sys.exit(1)

    # Load existing profiles if any (for incremental updates)
    all_profiles = {}
    if os.path.exists(args.output):
        with open(args.output) as f:
            all_profiles = json.load(f)
        print(f"Loaded existing profiles: {list(all_profiles.keys())}")

    print(f"\nFetching offshore wind profiles for: {args.isos}")
    print(f"Years: {FETCH_YEARS}")
    print(f"Loss factor: {NET_LOSS_FACTOR:.4f} (wake={WAKE_LOSS}, elec={ELECTRICAL_LOSS}, avail={AVAILABILITY})")
    print()

    for iso in args.isos:
        if iso not in OFFSHORE_ISOS:
            print(f"SKIP {iso} — no offshore wind resource")
            continue

        print(f"Processing {iso}...")
        if args.cache_dir:
            iso_profiles = build_profiles_from_cache(iso, args.cache_dir)
        else:
            iso_profiles = build_profile(iso, api_key)

        all_profiles[iso] = iso_profiles

    # Validate
    validate_profiles(all_profiles)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_profiles, f)
    print(f"\nSaved to {args.output}")

    # Summary
    print("\n--- Summary ---")
    for iso in all_profiles:
        years = list(all_profiles[iso].keys())
        print(f"  {iso}: {len(years)} years ({', '.join(years)})")


if __name__ == '__main__':
    main()
