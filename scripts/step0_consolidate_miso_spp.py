#!/usr/bin/env python3
"""
Consolidate MISO and SPP raw per-year JSON files into the aggregated
eia-930 profiles used by Step 1.

The raw per-year files (eia_hourly_{ISO}_{YEAR}.json, eia_demand_{ISO}_{YEAR}.json)
were fetched by step0_fetch_all_data.py but never merged into the consolidated
eia_generation_profiles.json, eia_demand_profiles.json, and eia_fossil_mix.json.

This script reads those raw files and appends MISO/SPP entries to the consolidated
JSONs, also copying the per-year files into the eia-930 directory.
"""

import json
import os
import shutil

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
EIA_930_DIR = os.path.join(DATA_DIR, 'eia-930')

NEW_ISOS = ['MISO', 'SPP']
YEARS = [2021, 2022, 2023, 2024, 2025]
H = 8760


def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_json(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, separators=(',', ':'))
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  Saved: {os.path.basename(filepath)} ({size_mb:.2f} MB)")


def remove_leap_day(arr):
    """Remove Feb 29 (hours 1416–1439) from leap-year 8784-hour array."""
    if len(arr) <= H:
        return arr[:H]
    return arr[:1416] + arr[1440:]


def main():
    print("=" * 70)
    print("Consolidating MISO & SPP into eia-930 profiles")
    print("=" * 70)

    # Load existing consolidated files
    gen_path = os.path.join(EIA_930_DIR, 'eia_generation_profiles.json')
    demand_path = os.path.join(EIA_930_DIR, 'eia_demand_profiles.json')
    fossil_path = os.path.join(EIA_930_DIR, 'eia_fossil_mix.json')

    gen_profiles = load_json(gen_path)
    demand_profiles = load_json(demand_path)
    fossil_mix = load_json(fossil_path)

    for iso in NEW_ISOS:
        print(f"\n--- Processing {iso} ---")

        # Generation profiles & fossil mix
        gen_profiles[iso] = {}
        fossil_mix[iso] = {}
        demand_profiles[iso] = {}

        for year in YEARS:
            year_str = str(year)

            # Load hourly generation
            hourly_path = os.path.join(DATA_DIR, f'eia_hourly_{iso}_{year}.json')
            if not os.path.exists(hourly_path):
                print(f"  WARNING: {hourly_path} not found, skipping year {year}")
                continue

            hourly = load_json(hourly_path)
            if len(hourly) > H:
                hourly = remove_leap_day(hourly)
            else:
                hourly = hourly[:H]

            # Build generation profiles (normalized by fuel)
            profile_fuels = {
                'solar': 'solar',
                'wind': 'wind',
                'nuclear': 'nuclear',
                'hydro': 'hydro',
            }

            fuel_lists = {}
            for profile_name, source_key in profile_fuels.items():
                values = [h.get(source_key, 0.0) or 0.0 for h in hourly]
                annual_total = sum(values)
                if annual_total > 0:
                    normalized = [round(v / annual_total, 10) for v in values]
                else:
                    nv = len(values)
                    normalized = [round(1.0 / nv, 10)] * nv
                fuel_lists[profile_name] = normalized

            gen_profiles[iso][year_str] = fuel_lists
            print(f"  {iso}/{year}: generation profiles built "
                  f"(solar={sum(h.get('solar',0) or 0 for h in hourly):.0f} MWh total, "
                  f"wind={sum(h.get('wind',0) or 0 for h in hourly):.0f} MWh total)")

            # Build fossil mix
            fossil_mix[iso][year_str] = {
                'hours': [h['period'] for h in hourly],
                'coal_share': [h.get('coal_share', 0.0) or 0.0 for h in hourly],
                'gas_share': [h.get('gas_share', 0.0) or 0.0 for h in hourly],
                'oil_share': [h.get('oil_share', 0.0) or 0.0 for h in hourly],
            }

            # Load demand
            demand_path_year = os.path.join(DATA_DIR, f'eia_demand_{iso}_{year}.json')
            if not os.path.exists(demand_path_year):
                print(f"  WARNING: {demand_path_year} not found, skipping demand year {year}")
                continue

            demand_raw = load_json(demand_path_year)
            if len(demand_raw) > H:
                demand_raw = remove_leap_day(demand_raw)
            else:
                demand_raw = demand_raw[:H]

            values = [h.get('demand_mw', 0.0) or 0.0 for h in demand_raw]
            total = sum(values)
            if total > 0:
                normalized = [v / total for v in values]
            else:
                normalized = [1.0 / len(values)] * len(values)

            demand_profiles[iso][year_str] = {
                'raw_mw': [round(v, 2) for v in values],
                'normalized': normalized,
                'total_annual_mwh': round(total),
                'peak_mw': round(max(values)),
                'min_mw': round(min(values)),
                'avg_mw': round(total / len(values)),
            }
            print(f"  {iso}/{year}: demand profile built "
                  f"(total={total/1e6:.1f} TWh, peak={max(values):.0f} MW)")

        # Copy raw files into eia-930 directory for consistency
        for year in YEARS:
            for prefix in ['eia_hourly', 'eia_demand']:
                src = os.path.join(DATA_DIR, f'{prefix}_{iso}_{year}.json')
                dst = os.path.join(EIA_930_DIR, f'{prefix}_{iso}_{year}.json')
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    print(f"  Copied {os.path.basename(src)} → eia-930/")

    # Save updated consolidated files
    print("\n--- Saving consolidated files ---")
    save_json(gen_path, gen_profiles)
    save_json(fossil_path, fossil_mix)
    save_json(demand_path, demand_profiles)

    # Verify
    print("\n--- Verification ---")
    for name, data in [('generation', gen_profiles), ('demand', demand_profiles), ('fossil_mix', fossil_mix)]:
        isos = sorted(data.keys())
        print(f"  {name}: {len(isos)} ISOs — {', '.join(isos)}")
        for iso in NEW_ISOS:
            if iso in data:
                years = sorted(data[iso].keys())
                print(f"    {iso}: years {', '.join(years)}")

    print("\nDone! MISO and SPP consolidated into eia-930 profiles.")
    print("Step 1 PFS Generator can now load all 7 ISOs.")


if __name__ == '__main__':
    main()
