#!/usr/bin/env python3
"""
Fix UTC→local timezone conversion in EIA demand and generation profiles.
================================================================================
The original fetch_all_data.py saved hourly profiles indexed by UTC hour-of-year.
This script re-indexes them to local prevailing time so that hour 0 = midnight local,
hour 12 = noon local, etc.

This is critical for:
  - LMP diurnal pricing (peak/off-peak classification)
  - Any hour-of-day analysis

Note: The optimizer's hourly matching is unaffected because both demand and generation
are UTC-indexed consistently — their relative alignment is correct. Only analyses that
depend on local time-of-day labeling (LMP, peak/off-peak) are impacted.

Usage:
    python fix_utc_profiles.py              # Fix all profiles
    python fix_utc_profiles.py --verify     # Just check current alignment
"""

import json
import os
import sys
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
H = 8760

# UTC offsets (standard time hours behind UTC; DST adds 1)
# We use a simplified approach: apply fixed offset per ISO
# For a more precise fix, we'd need to handle DST transitions hour-by-hour
ISO_TIMEZONES = {
    'CAISO': ZoneInfo('America/Los_Angeles'),   # UTC-8 (PST) / UTC-7 (PDT)
    'ERCOT': ZoneInfo('America/Chicago'),       # UTC-6 (CST) / UTC-5 (CDT)
    'PJM':   ZoneInfo('America/New_York'),      # UTC-5 (EST) / UTC-4 (EDT)
    'NYISO': ZoneInfo('America/New_York'),      # UTC-5 (EST) / UTC-4 (EDT)
    'NEISO': ZoneInfo('America/New_York'),      # UTC-5 (EST) / UTC-4 (EDT)
}

# Standard (winter) UTC offsets in hours
STD_OFFSETS = {
    'CAISO': -8,
    'ERCOT': -6,
    'PJM':   -5,
    'NYISO': -5,
    'NEISO': -5,
}


def get_utc_to_local_mapping(year, tz):
    """Build a mapping from UTC hour-of-year to local hour-of-year.

    Handles DST transitions properly:
    - Spring forward: one UTC hour maps to no local hour (skip)
    - Fall back: two UTC hours map to the same local hour (average)

    Returns: dict mapping local_hoy → list of utc_hoys
    """
    local_to_utc = defaultdict(list)

    jan1_utc = datetime(year, 1, 1, 0, 0, 0, tzinfo=ZoneInfo('UTC'))
    jan1_local = datetime(year, 1, 1, 0, 0, 0, tzinfo=tz)

    for utc_hoy in range(H + 48):  # extra buffer for timezone offset
        utc_dt = jan1_utc + timedelta(hours=utc_hoy)
        local_dt = utc_dt.astimezone(tz)

        # Only include if it maps to the target year
        if local_dt.year != year:
            continue

        local_jan1 = datetime(year, 1, 1, 0, 0, 0, tzinfo=tz)
        delta = local_dt - local_jan1
        local_hoy = int(delta.total_seconds() // 3600)

        if 0 <= local_hoy < H and 0 <= utc_hoy < H + 48:
            local_to_utc[local_hoy].append(utc_hoy)

    return local_to_utc


def reindex_array(utc_array, local_to_utc, n_utc):
    """Re-index a UTC-indexed array to local time.

    For fall-back hours (2 UTC hours → 1 local hour): average values
    For spring-forward gaps (no UTC hour → 1 local hour): interpolate
    """
    result = np.zeros(H)

    for local_hoy in range(H):
        utc_hoys = local_to_utc.get(local_hoy, [])
        # Filter to valid UTC indices (must be within source array bounds)
        valid = [u for u in utc_hoys if 0 <= u < min(n_utc, H)]

        if valid:
            result[local_hoy] = np.mean([utc_array[u] for u in valid])
        else:
            # Interpolate from neighbors
            prev_val = None
            next_val = None
            for offset in range(1, 24):
                if prev_val is None and local_hoy - offset >= 0:
                    prev_utcs = local_to_utc.get(local_hoy - offset, [])
                    prev_valid = [u for u in prev_utcs if 0 <= u < min(n_utc, H)]
                    if prev_valid:
                        prev_val = np.mean([utc_array[u] for u in prev_valid])
                if next_val is None and local_hoy + offset < H:
                    next_utcs = local_to_utc.get(local_hoy + offset, [])
                    next_valid = [u for u in next_utcs if 0 <= u < min(n_utc, H)]
                    if next_valid:
                        next_val = np.mean([utc_array[u] for u in next_valid])
                if prev_val is not None and next_val is not None:
                    break

            if prev_val is not None and next_val is not None:
                result[local_hoy] = (prev_val + next_val) / 2
            elif prev_val is not None:
                result[local_hoy] = prev_val
            elif next_val is not None:
                result[local_hoy] = next_val

    return result


def verify_alignment(iso, demand_raw, label=''):
    """Check if demand profile looks like local time (peak at local noon-6PM)."""
    arr = np.array(demand_raw[:H])

    # Compute average demand by hour of day
    n_days = H // 24
    hourly_avg = np.zeros(24)
    for h in range(24):
        hourly_avg[h] = np.mean(arr[h::24])

    peak_hour = np.argmax(hourly_avg)
    valley_hour = np.argmin(hourly_avg)

    # Local time: peak should be 14-19 (2PM-7PM), valley should be 2-6 (2AM-6AM)
    peak_ok = 12 <= peak_hour <= 20
    valley_ok = 0 <= valley_hour <= 8

    status = "OK" if (peak_ok and valley_ok) else "MISALIGNED"

    print(f"  {iso} {label}: peak hour={peak_hour:02d} (avg {hourly_avg[peak_hour]:,.0f} MW), "
          f"valley hour={valley_hour:02d} (avg {hourly_avg[valley_hour]:,.0f} MW) → {status}")

    if not (peak_ok and valley_ok):
        print(f"    Expected peak 12-20, valley 0-8. Likely UTC-indexed.")

    return peak_ok and valley_ok


def fix_profiles():
    """Fix both demand and generation profiles from UTC to local time."""

    # Load current profiles
    demand_path = os.path.join(DATA_DIR, 'eia_demand_profiles.json')
    gen_path = os.path.join(DATA_DIR, 'eia_generation_profiles.json')

    with open(demand_path) as f:
        demand_profiles = json.load(f)
    with open(gen_path) as f:
        gen_profiles = json.load(f)

    print("=" * 70)
    print("  BEFORE FIX — Verifying current alignment")
    print("=" * 70)

    for iso in ISO_TIMEZONES:
        if iso in demand_profiles:
            for year in sorted(demand_profiles[iso].keys()):
                raw = demand_profiles[iso][year].get('raw_mw', [])
                if raw:
                    verify_alignment(iso, raw, f'{year} demand')

    # Backup originals
    backup_demand = os.path.join(DATA_DIR, 'eia_demand_profiles_utc_backup.json')
    backup_gen = os.path.join(DATA_DIR, 'eia_generation_profiles_utc_backup.json')

    if not os.path.exists(backup_demand):
        with open(backup_demand, 'w') as f:
            json.dump(demand_profiles, f)
        print(f"\n  Backed up demand profiles to {backup_demand}")

    if not os.path.exists(backup_gen):
        with open(backup_gen, 'w') as f:
            json.dump(gen_profiles, f)
        print(f"  Backed up generation profiles to {backup_gen}")

    print(f"\n{'='*70}")
    print(f"  FIXING UTC → LOCAL TIME")
    print(f"{'='*70}")

    # Fix demand profiles
    for iso, tz in ISO_TIMEZONES.items():
        if iso not in demand_profiles:
            continue

        for year_str in sorted(demand_profiles[iso].keys()):
            year = int(year_str)
            local_to_utc = get_utc_to_local_mapping(year, tz)

            raw_mw = demand_profiles[iso][year_str].get('raw_mw', [])
            if not raw_mw:
                continue

            utc_arr = np.array(raw_mw[:H], dtype=np.float64)
            local_arr = reindex_array(utc_arr, local_to_utc, len(raw_mw))

            # Update raw_mw
            demand_profiles[iso][year_str]['raw_mw'] = [round(float(v), 1) for v in local_arr]

            # Recompute normalized
            total = demand_profiles[iso][year_str].get('total_annual_mwh', np.sum(local_arr))
            if total > 0:
                demand_profiles[iso][year_str]['normalized'] = [float(v / total) for v in local_arr]

            # Recompute peak/min/avg
            demand_profiles[iso][year_str]['peak_mw'] = float(np.max(local_arr))
            demand_profiles[iso][year_str]['min_mw'] = float(np.min(local_arr[local_arr > 0])) if np.any(local_arr > 0) else 0.0
            demand_profiles[iso][year_str]['avg_mw'] = float(np.mean(local_arr))

            print(f"  Fixed {iso} {year_str} demand: {len(raw_mw)} UTC hours → {H} local hours")

    # Fix generation profiles
    for iso, tz in ISO_TIMEZONES.items():
        if iso not in gen_profiles:
            continue

        for year_str in sorted(gen_profiles[iso].keys()):
            year = int(year_str)
            local_to_utc = get_utc_to_local_mapping(year, tz)

            for resource in ['nuclear', 'solar', 'wind', 'hydro', 'geothermal']:
                arr = gen_profiles[iso][year_str].get(resource)
                if arr is None:
                    continue

                utc_arr = np.array(arr[:H], dtype=np.float64)
                local_arr = reindex_array(utc_arr, local_to_utc, len(arr))
                gen_profiles[iso][year_str][resource] = [round(float(v), 6) for v in local_arr]

            print(f"  Fixed {iso} {year_str} generation profiles")

    # Save fixed profiles
    with open(demand_path, 'w') as f:
        json.dump(demand_profiles, f)
    print(f"\n  Saved fixed demand profiles: {demand_path}")

    with open(gen_path, 'w') as f:
        json.dump(gen_profiles, f)
    print(f"  Saved fixed generation profiles: {gen_path}")

    # Also fix per-ISO raw files
    print(f"\n{'='*70}")
    print(f"  FIXING PER-ISO RAW FILES")
    print(f"{'='*70}")

    for iso, tz in ISO_TIMEZONES.items():
        for year in [2024, 2025]:
            # Fix demand file
            dem_file = os.path.join(DATA_DIR, f'eia_demand_{iso}_{year}.json')
            if os.path.exists(dem_file):
                with open(dem_file) as f:
                    dem_data = json.load(f)

                utc_vals = np.array([h.get('demand_mw', 0) or 0 for h in dem_data[:H]], dtype=np.float64)
                local_to_utc = get_utc_to_local_mapping(year, tz)
                local_vals = reindex_array(utc_vals, local_to_utc, len(dem_data))

                # Rebuild with local timestamps
                local_data = []
                jan1 = datetime(year, 1, 1, 0, 0, 0)
                for h in range(H):
                    ts = jan1 + timedelta(hours=h)
                    local_data.append({
                        'period': ts.strftime('%Y-%m-%dT%H'),
                        'demand_mw': round(float(local_vals[h]), 1)
                    })

                with open(dem_file, 'w') as f:
                    json.dump(local_data, f, indent=1)
                print(f"  Fixed {dem_file}")

            # Fix hourly gen file
            gen_file = os.path.join(DATA_DIR, f'eia_hourly_{iso}_{year}.json')
            if os.path.exists(gen_file):
                with open(gen_file) as f:
                    gen_data = json.load(f)

                # Extract all fuel columns
                fuel_keys = [k for k in gen_data[0].keys() if k != 'period']
                utc_arrays = {}
                for key in fuel_keys:
                    utc_arrays[key] = np.array([h.get(key, 0) or 0 for h in gen_data[:H]], dtype=np.float64)

                local_to_utc = get_utc_to_local_mapping(year, tz)
                local_arrays = {}
                for key in fuel_keys:
                    local_arrays[key] = reindex_array(utc_arrays[key], local_to_utc, len(gen_data))

                # Rebuild with local timestamps
                local_data = []
                jan1 = datetime(year, 1, 1, 0, 0, 0)
                for h in range(H):
                    ts = jan1 + timedelta(hours=h)
                    entry = {'period': ts.strftime('%Y-%m-%dT%H')}
                    for key in fuel_keys:
                        val = local_arrays[key][h]
                        # Round appropriately
                        if 'share' in key:
                            entry[key] = round(float(val), 6)
                        else:
                            entry[key] = round(float(val), 2)
                    local_data.append(entry)

                with open(gen_file, 'w') as f:
                    json.dump(local_data, f, indent=1)
                print(f"  Fixed {gen_file}")

    # Verify after fix
    print(f"\n{'='*70}")
    print(f"  AFTER FIX — Verifying alignment")
    print(f"{'='*70}")

    # Reload
    with open(demand_path) as f:
        demand_profiles = json.load(f)

    for iso in ISO_TIMEZONES:
        if iso in demand_profiles:
            for year in sorted(demand_profiles[iso].keys()):
                raw = demand_profiles[iso][year].get('raw_mw', [])
                if raw:
                    verify_alignment(iso, raw, f'{year} demand')


def verify_only():
    """Just check current alignment without fixing."""
    demand_path = os.path.join(DATA_DIR, 'eia_demand_profiles.json')
    with open(demand_path) as f:
        demand_profiles = json.load(f)

    print("=" * 70)
    print("  DEMAND PROFILE ALIGNMENT CHECK")
    print("=" * 70)

    all_ok = True
    for iso in ISO_TIMEZONES:
        if iso in demand_profiles:
            for year in sorted(demand_profiles[iso].keys()):
                raw = demand_profiles[iso][year].get('raw_mw', [])
                if raw:
                    ok = verify_alignment(iso, raw, f'{year} demand')
                    if not ok:
                        all_ok = False

    if all_ok:
        print("\n  All profiles correctly aligned to local time.")
    else:
        print("\n  Some profiles are UTC-indexed. Run without --verify to fix.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true', help='Just check alignment')
    args = parser.parse_args()

    if args.verify:
        verify_only()
    else:
        fix_profiles()
