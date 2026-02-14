#!/usr/bin/env python3
"""
DST Fix: Convert existing EIA profiles from UTC-ordered to local-time-ordered.
============================================================================
Uses the optimizer results from the current run to identify globally optimal
solutions, then creates corrected profiles for a Phase 3 refinement run.

This is a lightweight fix that:
1. Reads existing EIA data files (already in data/)
2. Reorders hourly arrays from UTC to local prevailing time
3. Handles DST transitions (interpolate spring-forward, average fall-back)
4. Saves corrected profiles that can be used by optimize_phase3_only.py

No EIA API calls needed — works with existing data files.
"""

import json
import os
import sys
from datetime import datetime, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

H = 8760

# IANA timezones for each ISO
ISO_TIMEZONES = {
    'CAISO': 'America/Los_Angeles',
    'ERCOT': 'America/Chicago',
    'PJM':   'America/New_York',
    'NYISO': 'America/New_York',
    'NEISO': 'America/New_York',
}

# UTC offsets: standard time offset from UTC for each timezone
# During DST, offset is 1 hour less (closer to UTC)
STANDARD_OFFSETS = {
    'America/Los_Angeles': -8,
    'America/Chicago':     -6,
    'America/New_York':    -5,
}

# 2025 DST transitions (US): Spring forward March 9 at 2am, Fall back Nov 2 at 2am
# These are in LOCAL standard time
DST_START_2025 = datetime(2025, 3, 9, 2, 0)   # Spring forward: 2am → 3am
DST_END_2025 = datetime(2025, 11, 2, 2, 0)     # Fall back: 2am → 1am (repeat)


def get_utc_start_hour(tz_name):
    """Get the UTC hour that corresponds to midnight local standard time."""
    offset = STANDARD_OFFSETS[tz_name]
    return -offset  # e.g., PST is UTC-8, so midnight PST = 08:00 UTC


def utc_index_to_local_hour_of_year(utc_idx, tz_name, year=2025):
    """
    Convert a UTC array index (0-8759) to local hour-of-year (0-8759).

    The EIA data starts at local midnight in UTC, so:
    - CAISO data[0] = Jan 1 00:00 PST = Jan 1 08:00 UTC
    - CAISO data[8] = Jan 1 08:00 PST = Jan 1 16:00 UTC

    During DST (March 9 - Nov 2 for 2025):
    - Local clocks spring forward: 2am → 3am (hour 2 doesn't exist locally)
    - UTC index continues linearly but local hour shifts by 1
    """
    utc_start = get_utc_start_hour(tz_name)
    std_offset = STANDARD_OFFSETS[tz_name]

    # UTC datetime for this index
    # Data starts at local midnight = UTC hour utc_start on Jan 1
    base_utc = datetime(year, 1, 1, utc_start, 0)
    utc_dt = base_utc + timedelta(hours=utc_idx)

    # Determine if this UTC time falls in DST period
    # Convert DST boundaries to UTC
    dst_start_utc = DST_START_2025 - timedelta(hours=std_offset)  # 2am local to UTC
    dst_end_utc = DST_END_2025 - timedelta(hours=std_offset + 1)  # 2am local (DST) to UTC

    if dst_start_utc <= utc_dt < dst_end_utc:
        # During DST: offset is 1 hour closer to UTC
        local_dt = utc_dt + timedelta(hours=std_offset + 1)
    else:
        # Standard time
        local_dt = utc_dt + timedelta(hours=std_offset)

    # Convert local datetime to hour-of-year
    year_start = datetime(year, 1, 1, 0, 0)
    local_hoy = int((local_dt - year_start).total_seconds() / 3600)

    return local_hoy, local_dt


def reorder_utc_to_local(utc_array, tz_name, year=2025):
    """
    Reorder a UTC-ordered array to local-time ordering.

    Handles:
    - Spring forward: UTC hour that maps to skipped local hour → interpolate
    - Fall back: Two UTC hours map to same local hour → average
    - Produces exactly 8760 values indexed by local hour-of-year
    """
    n = min(len(utc_array), H)

    # Map each UTC index to its local hour-of-year
    local_buckets = {}  # local_hoy → list of (utc_idx, value)

    for utc_idx in range(n):
        local_hoy, _ = utc_index_to_local_hour_of_year(utc_idx, tz_name, year)
        if 0 <= local_hoy < H:
            if local_hoy not in local_buckets:
                local_buckets[local_hoy] = []
            local_buckets[local_hoy].append(utc_array[utc_idx])

    # Build output array
    result = [None] * H

    for hoy in range(H):
        if hoy in local_buckets:
            values = local_buckets[hoy]
            result[hoy] = sum(values) / len(values)  # Average for fall-back duplicates

    # Interpolate any missing hours (spring-forward gaps)
    for hoy in range(H):
        if result[hoy] is None:
            # Find nearest non-None neighbors
            prev_val = None
            next_val = None
            for offset in range(1, 25):
                if hoy - offset >= 0 and result[hoy - offset] is not None:
                    prev_val = result[hoy - offset]
                    break
            for offset in range(1, 25):
                if hoy + offset < H and result[hoy + offset] is not None:
                    next_val = result[hoy + offset]
                    break

            if prev_val is not None and next_val is not None:
                result[hoy] = (prev_val + next_val) / 2.0
            elif prev_val is not None:
                result[hoy] = prev_val
            elif next_val is not None:
                result[hoy] = next_val
            else:
                result[hoy] = 0.0

    return result


def normalize_profile(profile):
    """Normalize profile to sum to 1.0."""
    total = sum(profile)
    if total > 0:
        return [v / total for v in profile]
    return profile


def fix_generation_profiles():
    """Fix generation profiles: UTC → local time."""
    input_path = os.path.join(DATA_DIR, 'eia_generation_profiles.json')
    output_path = os.path.join(DATA_DIR, 'eia_generation_profiles_local.json')

    with open(input_path) as f:
        data = json.load(f)

    print("  Fixing generation profiles (UTC → local time)...")

    for iso in ISO_TIMEZONES:
        if iso not in data:
            continue

        tz = ISO_TIMEZONES[iso]

        for year in data[iso]:
            year_int = int(year)
            fuel_types = [k for k in data[iso][year] if k != 'solar_proxy']

            for fuel in fuel_types:
                profile = data[iso][year][fuel]
                if not profile or len(profile) < 100:
                    continue

                reordered = reorder_utc_to_local(profile, tz, year_int)
                # Re-normalize (sum should stay ~1.0 but reordering may cause tiny drift)
                data[iso][year][fuel] = normalize_profile(reordered)

            # Handle solar_proxy for NYISO
            if 'solar_proxy' in data[iso][year]:
                proxy = data[iso][year]['solar_proxy']
                reordered = reorder_utc_to_local(proxy, tz, year_int)
                data[iso][year]['solar_proxy'] = normalize_profile(reordered)

            print(f"    {iso} {year}: {len(fuel_types)} fuel types reordered")

    with open(output_path, 'w') as f:
        json.dump(data, f)
    print(f"  Saved: {output_path} ({os.path.getsize(output_path) / 1024:.0f} KB)")

    return output_path


def fix_demand_profiles():
    """Fix demand profiles: UTC → local time."""
    input_path = os.path.join(DATA_DIR, 'eia_demand_profiles.json')
    output_path = os.path.join(DATA_DIR, 'eia_demand_profiles_local.json')

    with open(input_path) as f:
        data = json.load(f)

    print("  Fixing demand profiles (UTC → local time)...")

    for iso in ISO_TIMEZONES:
        if iso not in data:
            continue

        tz = ISO_TIMEZONES[iso]

        # Fix normalized demand profile
        if 'normalized' in data[iso]:
            profile = data[iso]['normalized']
            reordered = reorder_utc_to_local(profile, tz)
            data[iso]['normalized'] = normalize_profile(reordered)

        # Fix raw MWh profile if present
        if 'raw_mwh' in data[iso]:
            profile = data[iso]['raw_mwh']
            reordered = reorder_utc_to_local(profile, tz)
            # Don't normalize raw MWh — preserve total
            total_orig = sum(profile[:H])
            total_new = sum(reordered)
            if total_new > 0:
                scale = total_orig / total_new
                data[iso]['raw_mwh'] = [v * scale for v in reordered]
            else:
                data[iso]['raw_mwh'] = reordered

        print(f"    {iso}: demand profile reordered")

    with open(output_path, 'w') as f:
        json.dump(data, f)
    print(f"  Saved: {output_path} ({os.path.getsize(output_path) / 1024:.0f} KB)")

    return output_path


def fix_fossil_mix():
    """Fix fossil mix profiles: UTC → local time."""
    input_path = os.path.join(DATA_DIR, 'eia_fossil_mix.json')
    output_path = os.path.join(DATA_DIR, 'eia_fossil_mix_local.json')

    with open(input_path) as f:
        data = json.load(f)

    print("  Fixing fossil mix profiles (UTC → local time)...")

    for iso in ISO_TIMEZONES:
        if iso not in data:
            continue

        tz = ISO_TIMEZONES[iso]

        for year in data[iso]:
            year_int = int(year)

            for fuel_share in ['coal_share', 'gas_share', 'oil_share']:
                if fuel_share in data[iso][year]:
                    profile = data[iso][year][fuel_share]
                    reordered = reorder_utc_to_local(profile, tz, year_int)
                    data[iso][year][fuel_share] = reordered

            # Re-normalize shares to sum to 1.0 at each hour
            coal = data[iso][year].get('coal_share', [0] * H)
            gas = data[iso][year].get('gas_share', [0] * H)
            oil = data[iso][year].get('oil_share', [0] * H)

            for h in range(min(H, len(coal), len(gas), len(oil))):
                total = coal[h] + gas[h] + oil[h]
                if total > 0:
                    coal[h] /= total
                    gas[h] /= total
                    oil[h] /= total

            # Update timestamps if present
            if 'hours' in data[iso][year]:
                # Regenerate local timestamps
                local_hours = []
                for h in range(H):
                    dt = datetime(year_int, 1, 1) + timedelta(hours=h)
                    local_hours.append(dt.strftime('%Y-%m-%dT%H'))
                data[iso][year]['hours'] = local_hours

            print(f"    {iso} {year}: fossil mix reordered + renormalized")

    with open(output_path, 'w') as f:
        json.dump(data, f)
    print(f"  Saved: {output_path} ({os.path.getsize(output_path) / 1024:.0f} KB)")

    return output_path


def main():
    print("=" * 70)
    print("  DST FIX: Convert EIA Profiles from UTC to Local Time")
    print("=" * 70)
    print()

    gen_path = fix_generation_profiles()
    demand_path = fix_demand_profiles()
    fossil_path = fix_fossil_mix()

    print()
    print("  Output files (use --profiles-dir flag in Phase 3 optimizer):")
    print(f"    {gen_path}")
    print(f"    {demand_path}")
    print(f"    {fossil_path}")
    print()
    print("  To use with Phase 3 re-optimizer:")
    print("    python3 optimize_phase3_only.py --profiles-dir data/")
    print("    (after renaming _local files to replace originals)")
    print()
    print("=" * 70)
    print("  DST FIX COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
