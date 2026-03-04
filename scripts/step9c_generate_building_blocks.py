#!/usr/bin/env python3
"""Step 9C: Generate building-blocks-data.js for the building blocks dashboard page.

Reads EIA hourly profiles from parquet and creates 24-hour representative day
shapes for each resource type per ISO. Nuclear, geothermal, and CCS-CCGT use
the modeled flat-baseload profiles (matching Step 1 PFS assumptions), not raw
EIA generation data.

MISO and SPP profiles are rotated from UTC to Central time (UTC-6) for display.

Output: dashboard/js/building-blocks-data.js
"""

import json
import os
import sys
import numpy as np

# Add scripts dir to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
ROOT = os.path.dirname(SCRIPT_DIR)

try:
    import pyarrow.parquet as pq
except ImportError:
    print("ERROR: pyarrow required. pip install pyarrow")
    sys.exit(1)

H = 8760

# UTC offset (standard time) for each ISO — used to rotate UTC profiles to local display time
UTC_OFFSETS = {
    'CAISO': 8,   # Pacific
    'ERCOT': 6,   # Central
    'PJM':   5,   # Eastern
    'NYISO': 5,   # Eastern
    'NEISO': 5,   # Eastern
    'MISO':  6,   # Central
    'SPP':   6,   # Central
}

# ISOs already in local time (fixed by step0_fix_utc_profiles.py)
LOCAL_TIME_ISOS = {'CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO'}

# ISOs that need UTC → local rotation for display
UTC_ISOS = {'MISO', 'SPP'}

# Which ISOs have offshore wind
OFFSHORE_ISOS = {'CAISO', 'NYISO', 'NEISO', 'PJM'}

ALL_ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']

# Synthetic storage dispatch shapes (same across all ISOs, in local time)
# Battery 4hr: charge midday (solar surplus), discharge evening peak
BATTERY4_SHAPE = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -1.5, -1.5, -1.5, -1.5, -1.5,
    0, 0,
    1.8, 1.8, 1.8, 1.8,
    0, 0, 0
]

# Battery 8hr: charge midday (wider window), discharge evening (wider window)
BATTERY8_SHAPE = [
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8,
    0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7
]

# LDES: discharge overnight/morning, charge midday
LDES_SHAPE = [
    0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
    -0.8, -0.8, -0.8, -0.8, -0.8,
    0.3, 0.3,
    0.8, 0.8, 0.8, 0.8, 0.8,
    0.3, 0.3
]


def load_profiles_from_parquet():
    """Load EIA generation and demand profiles from parquet files."""
    gen_path = os.path.join(ROOT, 'data', 'eia-930', 'eia_generation_profiles.parquet')
    dem_path = os.path.join(ROOT, 'data', 'eia-930', 'eia_demand_profiles.parquet')

    gen_table = pq.read_table(gen_path)
    dem_table = pq.read_table(dem_path)

    # Parse generation profiles: iso, fuel, hour, value
    gen_iso = np.array(gen_table.column('iso').to_pylist())
    gen_fuel = np.array(gen_table.column('fuel').to_pylist())
    gen_hour = gen_table.column('hour').to_numpy()
    gen_value = gen_table.column('value').to_numpy()

    # Parse demand profiles: iso, hour, normalized
    dem_iso = np.array(dem_table.column('iso').to_pylist())
    dem_hour = dem_table.column('hour').to_numpy()
    dem_norm = dem_table.column('normalized').to_numpy()

    # Check if demand has year column for seasonal splits
    dem_cols = dem_table.column_names
    dem_year = dem_table.column('year').to_numpy() if 'year' in dem_cols else None

    return (gen_iso, gen_fuel, gen_hour, gen_value,
            dem_iso, dem_hour, dem_norm, dem_year)


def compute_24h_avg(hours, values):
    """Average hourly values into a 24-hour representative day profile.
    Returns normalized array where mean = 1.0."""
    hod = hours % 24
    avg = np.array([values[hod == h].mean() if np.any(hod == h) else 0.0 for h in range(24)])
    mean = avg.mean()
    if mean > 0:
        avg = avg / mean
    return avg


def compute_seasonal_avg(hours, values, year_arr=None):
    """Compute summer (Jun-Aug) and winter (Dec-Feb) 24-hour profiles.
    If year_arr is None, derive season from hour-of-year position."""
    # Summer: hours 3624-5832 (Jun 1 to Aug 31, days 152-243)
    # Winter: hours 0-1416 (Jan 1 to Feb 28) + hours 8016-8760 (Dec 1 to Dec 31)
    hod = hours % 24
    doy = (hours % H) // 24

    summer_mask = (doy >= 152) & (doy < 244)
    winter_mask = (doy < 59) | (doy >= 335)

    summer_hours = hod[summer_mask]
    summer_vals = values[summer_mask]
    winter_hours = hod[winter_mask]
    winter_vals = values[winter_mask]

    summer_avg = np.array([summer_vals[summer_hours == h].mean()
                           if np.any(summer_hours == h) else 0.0 for h in range(24)])
    winter_avg = np.array([winter_vals[winter_hours == h].mean()
                           if np.any(winter_hours == h) else 0.0 for h in range(24)])

    # Normalize each to mean = 1.0
    sm = summer_avg.mean()
    wm = winter_avg.mean()
    if sm > 0:
        summer_avg = summer_avg / sm
    if wm > 0:
        winter_avg = winter_avg / wm

    return summer_avg, winter_avg


def rotate_utc_to_local(profile, offset):
    """Rotate a 24-hour UTC profile to local time by shifting right by offset hours.
    E.g., offset=6 for Central time: UTC hour 18 becomes local hour 12."""
    arr = np.array(profile)
    return np.roll(arr, -offset).tolist()


def round_list(arr, decimals=4):
    """Round array values for compact JSON output."""
    return [round(float(v), decimals) for v in arr]


def main():
    print("Loading EIA profiles from parquet...")
    (gen_iso, gen_fuel, gen_hour, gen_value,
     dem_iso, dem_hour, dem_norm, dem_year) = load_profiles_from_parquet()

    hourly_profiles = {}
    seasonal_profiles = {}

    for iso in ALL_ISOS:
        print(f"  Processing {iso}...")
        needs_rotation = iso in UTC_ISOS
        offset = UTC_OFFSETS[iso] if needs_rotation else 0

        iso_data = {}

        # --- Demand ---
        dem_mask = dem_iso == iso
        d_hours = dem_hour[dem_mask]
        d_values = dem_norm[dem_mask]
        demand_24h = compute_24h_avg(d_hours, d_values)
        if needs_rotation:
            demand_24h = np.array(rotate_utc_to_local(demand_24h, offset))
        iso_data['demand'] = round_list(demand_24h)

        # --- Solar ---
        # Use solar or solar_proxy
        sol_mask = (gen_iso == iso) & (gen_fuel == 'solar')
        if not np.any(sol_mask):
            sol_mask = (gen_iso == iso) & (gen_fuel == 'solar_proxy')
        if np.any(sol_mask):
            s_hours = gen_hour[sol_mask]
            s_values = gen_value[sol_mask]
            solar_24h = compute_24h_avg(s_hours, s_values)
            if needs_rotation:
                solar_24h = np.array(rotate_utc_to_local(solar_24h, offset))
            iso_data['solar'] = round_list(solar_24h)

        # --- Wind ---
        wnd_mask = (gen_iso == iso) & (gen_fuel == 'wind')
        if np.any(wnd_mask):
            w_hours = gen_hour[wnd_mask]
            w_values = gen_value[wnd_mask]
            wind_24h = compute_24h_avg(w_hours, w_values)
            if needs_rotation:
                wind_24h = np.array(rotate_utc_to_local(wind_24h, offset))
            iso_data['wind'] = round_list(wind_24h)

        # --- Nuclear: FLAT BASELOAD (Step 1 modeled profile) ---
        # Step 1 uses NUCLEAR_MONTHLY_CF with seasonal derate, but 24-hour
        # average is near-constant. Display as flat to match the model.
        iso_data['nuclear'] = [1.0] * 24

        # --- Hydro ---
        hyd_mask = (gen_iso == iso) & (gen_fuel == 'hydro')
        if np.any(hyd_mask):
            h_hours = gen_hour[hyd_mask]
            h_values = gen_value[hyd_mask]
            hydro_24h = compute_24h_avg(h_hours, h_values)
            if needs_rotation:
                hydro_24h = np.array(rotate_utc_to_local(hydro_24h, offset))
            iso_data['hydro'] = round_list(hydro_24h)

        # --- Geothermal (CAISO only): FLAT BASELOAD ---
        # Step 1 uses np.full(H, 1.0/H) — perfectly flat 24/7
        if iso == 'CAISO':
            iso_data['geothermal'] = [1.0] * 24

        # --- Offshore Wind ---
        if iso in OFFSHORE_ISOS:
            osw_mask = (gen_iso == iso) & (gen_fuel == 'offshore_wind')
            if np.any(osw_mask):
                o_hours = gen_hour[osw_mask]
                o_values = gen_value[osw_mask]
                osw_24h = compute_24h_avg(o_hours, o_values)
                if needs_rotation:
                    osw_24h = np.array(rotate_utc_to_local(osw_24h, offset))
                iso_data['offshore_wind'] = round_list(osw_24h)

        # --- CCS-CCGT: FLAT BASELOAD ---
        iso_data['ccs_ccgt'] = [1.0] * 24

        # --- Storage shapes (synthetic, same for all ISOs) ---
        iso_data['battery4_shape'] = list(BATTERY4_SHAPE)
        iso_data['battery8_shape'] = list(BATTERY8_SHAPE)
        iso_data['ldes_shape'] = list(LDES_SHAPE)

        hourly_profiles[iso] = iso_data

        # --- Seasonal profiles ---
        seasonal = {}
        # Demand seasonal
        if len(d_hours) > 0:
            d_summer, d_winter = compute_seasonal_avg(d_hours, d_values)
            if needs_rotation:
                d_summer = np.array(rotate_utc_to_local(d_summer, offset))
                d_winter = np.array(rotate_utc_to_local(d_winter, offset))
            seasonal['demand_summer'] = round_list(d_summer)
            seasonal['demand_winter'] = round_list(d_winter)

        # Solar seasonal
        if 'solar' in iso_data and np.any(sol_mask):
            s_summer, s_winter = compute_seasonal_avg(
                gen_hour[sol_mask], gen_value[sol_mask])
            if needs_rotation:
                s_summer = np.array(rotate_utc_to_local(s_summer, offset))
                s_winter = np.array(rotate_utc_to_local(s_winter, offset))
            seasonal['solar_summer'] = round_list(s_summer)
            seasonal['solar_winter'] = round_list(s_winter)

        seasonal_profiles[iso] = seasonal

    # Write output
    out_path = os.path.join(ROOT, 'dashboard', 'js', 'building-blocks-data.js')
    print(f"\nWriting {out_path}...")

    with open(out_path, 'w') as f:
        f.write("// Auto-generated by step9c_generate_building_blocks.py\n")
        f.write("// Nuclear, geothermal, and CCS-CCGT use modeled flat-baseload profiles\n")
        f.write("// (matching Step 1 PFS assumptions), not raw EIA generation data.\n")
        f.write("// MISO and SPP profiles are rotated from UTC to Central time for display.\n\n")
        f.write("const HOURLY_PROFILES = ")
        f.write(json.dumps(hourly_profiles, indent=2))
        f.write(";\n\n")
        f.write("const SEASONAL_PROFILES = ")
        f.write(json.dumps(seasonal_profiles, indent=2))
        f.write(";\n")

    print("Done!")


if __name__ == '__main__':
    main()
