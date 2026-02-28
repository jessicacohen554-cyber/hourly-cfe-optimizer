#!/usr/bin/env python3
"""
Fetch actual hourly LMP data from all 7 ISOs via gridstatus library.
================================================================================
Downloads day-ahead hourly LMPs for calibration against our synthetic LMP model.
Saves per-ISO JSON files compatible with calibrate_lmp_model.py.

ISOs: CAISO, ERCOT, PJM, NYISO, ISO-NE (NEISO), MISO, SPP
Data: Configurable year (2024 or 2025), hourly granularity, zone/hub-level

Usage:
  python step0_fetch_lmp_2025.py                    # Fetch all ISOs, 2025
  python step0_fetch_lmp_2025.py --year 2024        # Fetch all ISOs, 2024
  python step0_fetch_lmp_2025.py --iso PJM          # Fetch single ISO, 2025
  python step0_fetch_lmp_2025.py --iso ALL --year 2024  # All 7 ISOs, 2024
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

try:
    import gridstatus
except ImportError:
    print("ERROR: gridstatus not installed. Run: pip install gridstatus")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
LMP_DIR = os.path.join(DATA_DIR, 'lmp')
os.makedirs(LMP_DIR, exist_ok=True)

# Map our ISO names to gridstatus classes and representative nodes
ISO_CONFIG = {
    'CAISO': {
        'class': gridstatus.CAISO,
        'market': 'DAY_AHEAD_HOURLY',
        'location_type': 'ALL',  # Will filter to hub/zone after
        'hub_name': 'TH_SP15_GEN-APND',  # SP15 trading hub
        'zone_names': ['SP15', 'NP15', 'ZP26'],
    },
    'ERCOT': {
        'class': gridstatus.Ercot,
        'market': 'DAY_AHEAD_HOURLY',
        'location_type': 'ALL',
        'hub_name': 'HB_HOUSTON',  # Houston hub (largest load zone)
        'zone_names': ['HB_HOUSTON', 'HB_NORTH', 'HB_SOUTH', 'HB_WEST'],
    },
    'PJM': {
        'class': gridstatus.PJM,
        'market': 'DAY_AHEAD_HOURLY',
        'location_type': 'ZONE',
        'hub_name': 'WESTERN HUB',
        'zone_names': None,  # All zones
    },
    'NYISO': {
        'class': gridstatus.NYISO,
        'market': 'DAY_AHEAD_HOURLY',
        'location_type': 'ALL',
        'hub_name': None,  # Use Zone J (NYC) as reference
        'zone_names': None,
    },
    'NEISO': {
        'class': gridstatus.ISONE,
        'market': 'DAY_AHEAD_HOURLY',
        'location_type': 'ALL',
        'hub_name': '.I.HUB',  # Internal Hub
        'zone_names': None,
    },
    'MISO': {
        'class': gridstatus.MISO,
        'market': 'DAY_AHEAD_HOURLY',
        'location_type': 'ALL',
        'hub_name': 'INDIANA.HUB',  # Indiana Hub (central reference)
        'zone_names': None,
    },
    'SPP': {
        'class': gridstatus.SPP,
        'market': 'DAY_AHEAD_HOURLY',
        'location_type': 'ALL',
        'hub_name': 'SPS',  # SPS hub (representative)
        'zone_names': None,
    },
}

# Local timezones for peak/off-peak classification
ISO_TIMEZONE = {
    'CAISO': ZoneInfo('America/Los_Angeles'),
    'ERCOT': ZoneInfo('America/Chicago'),
    'PJM': ZoneInfo('America/New_York'),
    'NYISO': ZoneInfo('America/New_York'),
    'NEISO': ZoneInfo('America/New_York'),
    'MISO': ZoneInfo('America/Chicago'),
    'SPP': ZoneInfo('America/Chicago'),
}

def get_monthly_chunks(year):
    """Generate monthly date range chunks for any year."""
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    chunks = []
    for i in range(12):
        start_month = month_names[i]
        if i < 11:
            end_month = month_names[i + 1]
            end_year = year
        else:
            end_month = 'Jan'
            end_year = year + 1
        chunks.append((f'{start_month} 1, {year}', f'{end_month} 1, {end_year}'))
    return chunks


def fetch_iso_lmp(iso_name, year=2025, retry_max=3):
    """Fetch DA hourly LMPs for a single ISO and year."""
    config = ISO_CONFIG[iso_name]
    iso_cls = config['class']
    iso = iso_cls()

    months = get_monthly_chunks(year)

    print(f"\n{'='*60}")
    print(f"  Fetching {iso_name} {year} DA Hourly LMPs")
    print(f"{'='*60}")

    all_dfs = []
    for i, (start, end) in enumerate(months):
        month_label = pd.Timestamp(start).strftime('%b %Y')
        print(f"  [{i+1}/12] {month_label}...", end='', flush=True)

        for attempt in range(retry_max):
            try:
                kwargs = {'date': start, 'end': end}

                # ISO-specific API parameters
                if iso_name == 'ERCOT':
                    # ERCOT doesn't take 'market' param — returns settlement point prices
                    kwargs['location_type'] = 'Settlement Point'
                elif iso_name == 'CAISO':
                    kwargs['market'] = 'DAY_AHEAD_HOURLY'
                    kwargs['locations'] = config.get('zone_names')
                elif iso_name == 'PJM':
                    kwargs['market'] = config['market']
                    kwargs['location_type'] = config.get('location_type', 'ZONE')
                elif iso_name in ('NYISO', 'NEISO'):
                    kwargs['market'] = 'DAY_AHEAD_HOURLY'
                elif iso_name == 'MISO':
                    kwargs['market'] = 'DAY_AHEAD_HOURLY'
                elif iso_name == 'SPP':
                    kwargs['market'] = 'DAY_AHEAD_HOURLY'

                df = iso.get_lmp(**kwargs)
                all_dfs.append(df)
                print(f" {len(df)} rows", flush=True)
                break
            except Exception as e:
                wait = 2 ** (attempt + 1)
                print(f" ERROR: {e}", flush=True)
                if attempt < retry_max - 1:
                    print(f"    Retrying in {wait}s...", flush=True)
                    time.sleep(wait)
                else:
                    print(f"    FAILED after {retry_max} attempts", flush=True)

        time.sleep(1)  # Rate limiting

    if not all_dfs:
        print(f"  No data fetched for {iso_name}")
        return None

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n  Total rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Date range: {df.iloc[:, 0].min()} to {df.iloc[:, 0].max()}")

    return df


def extract_hourly_hub_lmp(df, iso_name):
    """Extract hourly hub/zone-level LMP from raw gridstatus output."""
    config = ISO_CONFIG[iso_name]

    # Identify the LMP column
    lmp_col = None
    for col in df.columns:
        col_lower = str(col).lower()
        if 'lmp' in col_lower and ('total' in col_lower or col_lower.endswith('lmp')):
            lmp_col = col
            break
    if lmp_col is None:
        # Fallback: look for any column with 'lmp'
        for col in df.columns:
            if 'lmp' in str(col).lower():
                lmp_col = col
                break
    if lmp_col is None:
        print(f"  WARNING: No LMP column found. Columns: {list(df.columns)}")
        return None

    print(f"  Using LMP column: {lmp_col}")

    # Identify the time column
    time_col = None
    for col in df.columns:
        if 'time' in str(col).lower() or 'interval' in str(col).lower():
            time_col = col
            break
    if time_col is None:
        time_col = df.columns[0]
    print(f"  Using time column: {time_col}")

    # Identify the location column
    loc_col = None
    for col in df.columns:
        col_lower = str(col).lower()
        if 'location' in col_lower or 'node' in col_lower or 'pnode' in col_lower or 'zone' in col_lower:
            loc_col = col
            break
    if loc_col is not None:
        print(f"  Using location column: {loc_col}")
        print(f"  Unique locations (sample): {list(df[loc_col].unique()[:20])}")

    # Filter to hub/reference node
    hub_name = config.get('hub_name')
    if hub_name and loc_col:
        # Try exact match first, then partial
        hub_mask = df[loc_col].astype(str).str.contains(hub_name, case=False, na=False)
        hub_df = df[hub_mask].copy()
        if len(hub_df) == 0:
            print(f"  WARNING: Hub '{hub_name}' not found. Using system average.")
            hub_df = df.copy()
        else:
            print(f"  Filtered to hub '{hub_name}': {len(hub_df)} rows")
    else:
        hub_df = df.copy()

    # Convert to hourly average
    hub_df[time_col] = pd.to_datetime(hub_df[time_col], utc=True)
    hub_df = hub_df.set_index(time_col)
    hub_df[lmp_col] = pd.to_numeric(hub_df[lmp_col], errors='coerce')

    # Resample to hourly if sub-hourly
    hourly = hub_df[lmp_col].resample('h').mean().dropna()
    print(f"  Hourly observations: {len(hourly)}")

    return hourly


def compute_lmp_summary(hourly_lmp, iso_name, year=2025):
    """Compute summary statistics from hourly LMP series."""
    lmp = hourly_lmp.values
    lmp = lmp[~np.isnan(lmp)]

    stats = {
        'iso': iso_name,
        'year': year,
        'n_hours': len(lmp),
        'avg_lmp': float(np.mean(lmp)),
        'median_lmp': float(np.median(lmp)),
        'std_lmp': float(np.std(lmp)),
        'min_lmp': float(np.min(lmp)),
        'max_lmp': float(np.max(lmp)),
        'p10': float(np.percentile(lmp, 10)),
        'p25': float(np.percentile(lmp, 25)),
        'p50': float(np.percentile(lmp, 50)),
        'p75': float(np.percentile(lmp, 75)),
        'p90': float(np.percentile(lmp, 90)),
        'p95': float(np.percentile(lmp, 95)),
        'p99': float(np.percentile(lmp, 99)),
        'negative_hours': int(np.sum(lmp < 0)),
        'scarcity_hours': int(np.sum(lmp > 200)),
        'zero_hours': int(np.sum(lmp == 0)),
    }

    # Peak/off-peak (peak = hours 7-22 LOCAL time)
    tz = ISO_TIMEZONE.get(iso_name)
    idx = hourly_lmp.index
    if tz:
        local_hours = idx.tz_convert(tz).hour
    else:
        local_hours = idx.hour
    peak_mask = (local_hours >= 7) & (local_hours < 22)
    if peak_mask.any():
        stats['peak_avg'] = float(np.mean(lmp[peak_mask]))
        stats['offpeak_avg'] = float(np.mean(lmp[~peak_mask]))
    else:
        stats['peak_avg'] = stats['avg_lmp']
        stats['offpeak_avg'] = stats['avg_lmp']

    return stats


def save_results(hourly_lmp, stats, iso_name, year=2025):
    """Save hourly LMP data and summary stats."""
    # Save hourly as JSON (for calibrate_lmp_model.py compatibility)
    hourly_data = []
    for ts, val in hourly_lmp.items():
        hourly_data.append({
            'timestamp': ts.isoformat(),
            'lmp': float(val) if not np.isnan(val) else None,
        })

    hourly_path = os.path.join(LMP_DIR, f'{iso_name}_actual_lmp_{year}.json')
    with open(hourly_path, 'w') as f:
        json.dump(hourly_data, f, indent=1)
    print(f"  Saved hourly LMP: {hourly_path}")

    # Save stats
    stats_path = os.path.join(LMP_DIR, f'{iso_name}_actual_lmp_stats_{year}.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved stats: {stats_path}")

    return hourly_path, stats_path


def print_summary_table(all_stats, year=2025):
    """Print cross-ISO comparison table."""
    print(f"\n{'='*80}")
    print(f"  {year} ACTUAL LMP SUMMARY — ALL ISOs")
    print(f"{'='*80}")
    print(f"  {'ISO':<8} {'Avg':>8} {'P10':>8} {'P50':>8} {'P90':>8} "
          f"{'Peak':>8} {'OffPk':>8} {'Neg hrs':>8} {'Scar':>8} {'Hours':>8}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} "
          f"{'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for s in all_stats:
        print(f"  {s['iso']:<8} "
              f"${s['avg_lmp']:>6.1f} "
              f"${s['p10']:>6.1f} "
              f"${s['p50']:>6.1f} "
              f"${s['p90']:>6.1f} "
              f"${s['peak_avg']:>6.1f} "
              f"${s['offpeak_avg']:>6.1f} "
              f"{s['negative_hours']:>8} "
              f"{s['scarcity_hours']:>8} "
              f"{s['n_hours']:>8}")

    # Save combined summary
    summary_path = os.path.join(LMP_DIR, f'actual_lmp_summary_{year}.json')
    with open(summary_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n  Combined summary: {summary_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Fetch actual hourly LMP data from ISOs')
    parser.add_argument('--iso', type=str, default=None,
                        help='ISO to fetch (CAISO, ERCOT, PJM, NYISO, NEISO, MISO, SPP, or ALL)')
    parser.add_argument('--year', type=int, default=2025,
                        help='Year to fetch (default: 2025)')
    parser.add_argument('--force', action='store_true',
                        help='Re-fetch even if data already exists')
    args = parser.parse_args()

    year = args.year
    if args.iso == 'ALL':
        isos = list(ISO_CONFIG.keys())
    elif args.iso:
        isos = [args.iso]
    else:
        isos = list(ISO_CONFIG.keys())

    all_stats = []

    for iso_name in isos:
        try:
            # Check if already fetched
            existing = os.path.join(LMP_DIR, f'{iso_name}_actual_lmp_{year}.json')
            if os.path.exists(existing) and not args.force:
                print(f"\n  {iso_name}: Already fetched ({existing}). Use --force to re-fetch.")
                stats_path = os.path.join(LMP_DIR, f'{iso_name}_actual_lmp_stats_{year}.json')
                if os.path.exists(stats_path):
                    with open(stats_path) as f:
                        all_stats.append(json.load(f))
                continue

            df = fetch_iso_lmp(iso_name, year=year)
            if df is None:
                continue

            hourly = extract_hourly_hub_lmp(df, iso_name)
            if hourly is None:
                continue

            stats = compute_lmp_summary(hourly, iso_name, year=year)
            save_results(hourly, stats, iso_name, year=year)
            all_stats.append(stats)

            # Print individual ISO summary
            print(f"\n  {iso_name} {year} LMP Summary:")
            print(f"    Avg: ${stats['avg_lmp']:.2f}/MWh")
            print(f"    P10/P50/P90: ${stats['p10']:.1f} / ${stats['p50']:.1f} / ${stats['p90']:.1f}")
            print(f"    Peak/Off-peak: ${stats['peak_avg']:.1f} / ${stats['offpeak_avg']:.1f}")
            print(f"    Negative hours: {stats['negative_hours']}")
            print(f"    Scarcity hours (>$200): {stats['scarcity_hours']}")

        except Exception as e:
            print(f"\n  ERROR fetching {iso_name}: {e}")
            import traceback
            traceback.print_exc()

    if all_stats:
        print_summary_table(all_stats, year=year)


if __name__ == '__main__':
    main()
