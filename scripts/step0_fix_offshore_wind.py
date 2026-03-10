#!/usr/bin/env python3
"""
Convert offshore wind profiles from UTC to local prevailing time and write
to the EIA-format generation parquet.

The NREL Wind Toolkit API returns data in UTC (utc=true). All other
generation profiles (solar, wind, nuclear, hydro) in the EIA parquet are
indexed in local prevailing time. This script:

  1. Reads data/offshore_wind_profiles.json  (UTC-indexed, sum-normalized per year)
  2. Converts each 8760-hour profile from UTC → local prevailing time
     using DST-aware mapping (same logic as step0_fix_utc_profiles.py)
  3. Appends 'offshore_wind' rows to data/eia-930/eia_generation_profiles.parquet
     matching schema: (iso, year, fuel, hour, value)
  4. Also writes a standalone parquet for convenience:
     data/offshore_wind_profiles.parquet

Usage:
    python step0_fix_offshore_wind.py              # Convert + append to EIA parquet
    python step0_fix_offshore_wind.py --verify     # Check alignment without writing
"""

import json
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
EIA_DIR = os.path.join(DATA_DIR, 'eia-930')
H = 8760

# Source file (UTC-indexed, from step0_fetch_offshore_wind.py)
OFFSHORE_JSON = os.path.join(DATA_DIR, 'offshore_wind_profiles.json')

# Output files
OFFSHORE_PARQUET = os.path.join(DATA_DIR, 'offshore_wind_profiles.parquet')
EIA_GEN_PARQUET = os.path.join(EIA_DIR, 'eia_generation_profiles.parquet')

# ISO → IANA timezone (same as step0_fix_utc_profiles.py + MISO/SPP)
ISO_TIMEZONES = {
    'CAISO': ZoneInfo('America/Los_Angeles'),
    'ERCOT': ZoneInfo('America/Chicago'),
    'PJM':   ZoneInfo('America/New_York'),
    'NYISO': ZoneInfo('America/New_York'),
    'NEISO': ZoneInfo('America/New_York'),
    'MISO':  ZoneInfo('America/Chicago'),
    'SPP':   ZoneInfo('America/Chicago'),
}

# Years used in EIA generation profiles (we map offshore wind years → these)
# EIA profiles: 2021-2025.  Offshore wind: 5 years per ISO.
# We average across all offshore years to produce one profile per EIA year,
# since the offshore wind resource is synthetic (no year-matched EIA data).
# The 5-year average smooths interannual variability.
EIA_YEARS = [2021, 2022, 2023, 2024, 2025]


def get_utc_to_local_mapping(year, tz):
    """Build mapping from local hour-of-year → list of UTC hour-of-year indices.

    Handles DST transitions:
    - Spring forward: no UTC hour maps to one local hour → interpolate
    - Fall back: two UTC hours map to same local hour → average
    """
    local_to_utc = defaultdict(list)

    jan1_utc = datetime(year, 1, 1, 0, 0, 0, tzinfo=ZoneInfo('UTC'))

    for utc_hoy in range(H + 48):  # buffer for timezone offset
        utc_dt = jan1_utc + timedelta(hours=utc_hoy)
        local_dt = utc_dt.astimezone(tz)

        if local_dt.year != year:
            continue

        local_jan1 = datetime(year, 1, 1, 0, 0, 0, tzinfo=tz)
        delta = local_dt - local_jan1
        local_hoy = int(delta.total_seconds() // 3600)

        if 0 <= local_hoy < H and 0 <= utc_hoy < H + 48:
            local_to_utc[local_hoy].append(utc_hoy)

    return local_to_utc


def reindex_utc_to_local(utc_array, local_to_utc, n_utc):
    """Re-index a UTC-indexed array to local prevailing time.

    Fall-back hours (2 UTC → 1 local): average.
    Spring-forward gaps (0 UTC → 1 local): interpolate from neighbors.
    """
    result = np.zeros(H, dtype=np.float64)

    for local_hoy in range(H):
        utc_hoys = local_to_utc.get(local_hoy, [])
        valid = [u for u in utc_hoys if 0 <= u < n_utc]

        if valid:
            result[local_hoy] = np.mean([utc_array[u] for u in valid])
        else:
            # Interpolate from neighbors
            prev_val = next_val = None
            for offset in range(1, 24):
                if prev_val is None and local_hoy - offset >= 0:
                    prev_utcs = local_to_utc.get(local_hoy - offset, [])
                    pv = [u for u in prev_utcs if 0 <= u < n_utc]
                    if pv:
                        prev_val = np.mean([utc_array[u] for u in pv])
                if next_val is None and local_hoy + offset < H:
                    next_utcs = local_to_utc.get(local_hoy + offset, [])
                    nv = [u for u in next_utcs if 0 <= u < n_utc]
                    if nv:
                        next_val = np.mean([utc_array[u] for u in nv])
                if prev_val is not None and next_val is not None:
                    break

            if prev_val is not None and next_val is not None:
                result[local_hoy] = (prev_val + next_val) / 2
            elif prev_val is not None:
                result[local_hoy] = prev_val
            elif next_val is not None:
                result[local_hoy] = next_val

    return result


def load_offshore_json():
    """Load offshore wind JSON → {iso: {year_str: {offshore_wind: [8760], capacity_factor: float}}}."""
    if not os.path.exists(OFFSHORE_JSON):
        print(f"ERROR: {OFFSHORE_JSON} not found. Run step0_fetch_offshore_wind.py first.")
        sys.exit(1)

    with open(OFFSHORE_JSON) as f:
        data = json.load(f)

    print(f"Loaded offshore wind JSON: {list(data.keys())}")
    for iso in data:
        years = sorted(data[iso].keys())
        cfs = [data[iso][y]['capacity_factor'] for y in years]
        print(f"  {iso}: years={years}, CFs={[f'{c:.3f}' for c in cfs]}")

    return data


def convert_and_build_parquet(data):
    """Convert UTC profiles → local time, average across years, write parquet.

    Returns DataFrame with (iso, year, fuel, hour, value) matching EIA schema.
    """
    rows = []

    for iso, iso_data in data.items():
        tz = ISO_TIMEZONES.get(iso)
        if tz is None:
            print(f"  SKIP {iso} — no timezone mapping (not an offshore ISO)")
            continue

        # Collect all year profiles (UTC, sum-normalized)
        yearly_utc = []
        for year_str in sorted(iso_data.keys()):
            profile = iso_data[year_str]['offshore_wind']
            assert len(profile) == H, f"{iso}/{year_str}: expected {H} hours, got {len(profile)}"
            yearly_utc.append(np.array(profile, dtype=np.float64))

        if not yearly_utc:
            continue

        # Average across offshore wind years → single representative UTC profile
        avg_utc = np.mean(yearly_utc, axis=0)

        # Convert UTC → local prevailing time
        # Use a representative year for DST mapping (2024 — non-leap, recent)
        ref_year = 2024
        local_to_utc = get_utc_to_local_mapping(ref_year, tz)
        avg_local = reindex_utc_to_local(avg_utc, local_to_utc, H)

        # Re-normalize so profile sums to 1.0 (timezone shift can slightly change sum)
        total = avg_local.sum()
        if total > 0:
            avg_local = avg_local / total

        local_cf = avg_local.mean() * H  # unnormalized mean
        raw_cfs = [iso_data[y]['capacity_factor'] for y in sorted(iso_data.keys())]
        avg_cf = np.mean(raw_cfs)

        print(f"  {iso}: avg CF={avg_cf:.3f}, profile sum={avg_local.sum():.6f}, "
              f"local max_hour={np.argmax(avg_local)}, years_averaged={len(yearly_utc)}")

        # Write one row per (iso, year, hour) for each EIA year
        # All EIA years get the same averaged offshore wind profile
        # (no year-matched NREL data — offshore wind is a resource potential, not historical)
        for year in EIA_YEARS:
            for h in range(H):
                rows.append({
                    'iso': iso,
                    'year': year,
                    'fuel': 'offshore_wind',
                    'hour': h,
                    'value': float(avg_local[h]),
                })

    df = pd.DataFrame(rows)
    print(f"\nBuilt DataFrame: {len(df)} rows, {df['iso'].nunique()} ISOs, "
          f"{df['year'].nunique()} years")

    return df


def verify_alignment(df):
    """Verify offshore wind profiles look reasonable in local time.

    Wind should NOT have a strong diurnal peak (unlike solar), but it should
    not show a UTC-offset artifact either.
    """
    print("\n" + "=" * 70)
    print("  OFFSHORE WIND LOCAL-TIME ALIGNMENT CHECK")
    print("=" * 70)

    for iso in df['iso'].unique():
        iso_df = df[(df['iso'] == iso) & (df['year'] == EIA_YEARS[0])]
        values = iso_df.sort_values('hour')['value'].values

        # Compute hourly average by hour-of-day
        n_days = H // 24
        hourly_avg = np.zeros(24)
        for h in range(24):
            hourly_avg[h] = np.mean(values[h::24])

        peak_hour = np.argmax(hourly_avg)
        valley_hour = np.argmin(hourly_avg)
        diurnal_range = (hourly_avg.max() - hourly_avg.min()) / hourly_avg.mean() * 100

        print(f"  {iso}: peak={peak_hour:02d}:00, valley={valley_hour:02d}:00, "
              f"diurnal_range={diurnal_range:.1f}% of mean")

        # Wind shouldn't have >50% diurnal range — if it does, likely still UTC-shifted
        if diurnal_range > 50:
            print(f"    WARNING: Very high diurnal range — check timezone conversion")


def append_to_eia_parquet(offshore_df):
    """Append offshore_wind rows to the EIA generation profiles parquet.

    Removes any existing offshore_wind rows first (idempotent).
    """
    if not os.path.exists(EIA_GEN_PARQUET):
        print(f"WARNING: {EIA_GEN_PARQUET} not found — writing offshore-only parquet")
        offshore_df.to_parquet(EIA_GEN_PARQUET, index=False)
        return

    existing = pd.read_parquet(EIA_GEN_PARQUET)
    print(f"\nExisting EIA parquet: {len(existing)} rows, fuels={existing['fuel'].unique().tolist()}")

    # Remove any previous offshore_wind rows
    before = len(existing)
    existing = existing[existing['fuel'] != 'offshore_wind']
    removed = before - len(existing)
    if removed > 0:
        print(f"  Removed {removed} existing offshore_wind rows")

    # Append
    combined = pd.concat([existing, offshore_df], ignore_index=True)
    combined = combined.sort_values(['iso', 'year', 'fuel', 'hour']).reset_index(drop=True)

    combined.to_parquet(EIA_GEN_PARQUET, index=False)
    print(f"  Written {len(combined)} rows to {EIA_GEN_PARQUET}")
    print(f"  Fuels now: {combined['fuel'].unique().tolist()}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert offshore wind profiles UTC → local and write parquet")
    parser.add_argument('--verify', action='store_true', help='Just check alignment without writing')
    args = parser.parse_args()

    data = load_offshore_json()
    offshore_df = convert_and_build_parquet(data)

    verify_alignment(offshore_df)

    if args.verify:
        print("\n  --verify mode: no files written.")
        return

    # Write standalone offshore wind parquet
    offshore_df.to_parquet(OFFSHORE_PARQUET, index=False)
    print(f"\nWrote standalone: {OFFSHORE_PARQUET} ({len(offshore_df)} rows)")

    # Append to EIA generation parquet
    append_to_eia_parquet(offshore_df)

    print("\nDone. Offshore wind profiles are now in local prevailing time, "
          "matching all other EIA generation profiles.")


if __name__ == '__main__':
    main()
