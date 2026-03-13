#!/usr/bin/env python3
"""Step 4.1F — Extract building blocks data for the Building Blocks dashboard page.

Uses Step 1's EXACT generation profiles (via step1_pfs_generator.load_data +
get_supply_profiles) to ensure the 24-hour shapes shown on the dashboard match
the profiles used in the physics feasible space optimizer. This includes:
  - Nuclear monthly capacity factor derate
  - Solar DST-aware nighttime zeroing
  - Multi-year averaging (2021-2025)
  - Geothermal flat baseload (CAISO only)

Reads Step 3 cost-optimization parquets for resource mix data.

Outputs:
  1. HOURLY_PROFILES — normalized 24-hour generation shapes per ISO per resource
  2. SEASONAL_PROFILES — summer/winter 24-hour profiles per ISO
  3. BUILDING_BLOCKS_BASELINE — resource mix & storage at each threshold (Track 1)
  4. BUILDING_BLOCKS_DG — resource mix & storage at each threshold (Track 2)

Output file: dashboard/js/building-blocks-data.js

Dependencies:
  - step1_pfs_generator.py (load_data, get_supply_profiles)
  - data/eia-930/eia_generation_profiles.parquet
  - data/eia-930/eia_demand_profiles.parquet
  - data/step2.2-cost/step3_co_{ISO}.parquet (baseline)
  - data/step2.2-cost/step3_dg_{ISO}_t{T}.parquet (demand growth)
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
STEP3_DIR = DATA_DIR / "step2.2-cost"
OUT_DIR = DATA_DIR / "step4-analysis" / "building-blocks"
JS_PATH = ROOT / "dashboard" / "js" / "building-blocks-data.js"

# Add scripts dir to sys.path for Step 1 imports
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_config import ISOS
ALL_THRESHOLDS = [10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 87.5,
                  90, 92.5, 95, 97.5, 99, 99.5, 99.9, 99.99]

H = 8760  # hours in a standard year

# Resources and column mapping
RESOURCE_COLS = {
    "clean_firm": "mix_clean_firm",
    "solar": "mix_solar",
    "wind": "mix_wind",
    "offshore_wind": "mix_offshore_wind",
    "ccs_ccgt": "mix_ccs_ccgt",
    "hydro": "mix_hydro",
}
STORAGE_COLS = {
    "battery4": "battery_dispatch_pct",
    "battery8": "battery8_dispatch_pct",
    "ldes": "ldes_dispatch_pct",
    "h2": "h2_dispatch_pct",
}

# Base demand for GW conversion (TWh → GW = TWh / 8.76)
BASE_DEMAND_TWH = {
    "CAISO": 224.0, "ERCOT": 405.0, "PJM": 732.0,
    "NYISO": 149.0, "NEISO": 112.0, "MISO": 571.0, "SPP": 245.0,
}

# Synthetic storage dispatch shapes (same across all ISOs, in local time)
BATTERY4_SHAPE = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -1.5, -1.5, -1.5, -1.5, -1.5,
    0, 0,
    1.8, 1.8, 1.8, 1.8,
    0, 0, 0
]
BATTERY8_SHAPE = [
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8,
    0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7
]
LDES_SHAPE = [
    0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
    -0.8, -0.8, -0.8, -0.8, -0.8,
    0.3, 0.3,
    0.8, 0.8, 0.8, 0.8, 0.8,
    0.3, 0.3
]


# ---------------------------------------------------------------------------
# Profile helpers — uses EXACT Step 1 profiles (nuclear derate, solar DST, etc.)
# ---------------------------------------------------------------------------

def round_list(arr, decimals=4):
    """Round array values for compact JSON output."""
    return [round(float(v), decimals) for v in arr]


def profile_to_24h(arr_8760):
    """Collapse an 8760-hour profile into a 24-hour representative day.
    Returns normalized array where mean = 1.0. All values clamped >= 0."""
    arr = np.array(arr_8760[:H], dtype=np.float64)
    np.maximum(arr, 0.0, out=arr)
    hours = np.arange(H) % 24
    avg = np.array([arr[hours == h].mean() for h in range(24)])
    mean = avg.mean()
    if mean > 0:
        avg = avg / mean
    return avg


def profile_to_seasonal(arr_8760):
    """Collapse an 8760-hour profile into summer/winter 24-hour profiles.
    Returns (summer_24h, winter_24h) both normalized to mean=1.0."""
    arr = np.array(arr_8760[:H], dtype=np.float64)
    np.maximum(arr, 0.0, out=arr)
    hours = np.arange(H)
    hod = hours % 24
    doy = hours // 24
    summer_mask = (doy >= 152) & (doy < 244)
    winter_mask = (doy < 59) | (doy >= 335)

    def avg_by_hour(mask):
        h = hod[mask]
        v = arr[mask]
        a = np.array([v[h == hr].mean() if np.any(h == hr) else 0.0
                       for hr in range(24)])
        m = a.mean()
        return a / m if m > 0 else a

    return avg_by_hour(summer_mask), avg_by_hour(winter_mask)


def load_step1_profiles():
    """Load profiles using Step 1's exact functions (load_data + get_supply_profiles).

    Returns: (hourly_profiles, seasonal_profiles)
      hourly_profiles: {ISO: {resource: [24 values]}}
      seasonal_profiles: {ISO: {demand_summer: [24], ...}}
    """
    from step1_pfs_generator import load_data, get_supply_profiles

    print("  Loading Step 1 data (multi-year averaged EIA profiles)...")
    demand_data, gen_profiles, _, _ = load_data()

    hourly_profiles = {}
    seasonal_profiles = {}

    for iso in ISOS:
        print(f"  {iso}: computing 24-hour shapes from Step 1 profiles...")
        supply = get_supply_profiles(iso, gen_profiles)
        demand_norm = demand_data[iso]["normalized"]

        iso_data = {}
        iso_seasonal = {}

        # Demand (from Step 1's multi-year averaged, normalized demand)
        iso_data["demand"] = round_list(profile_to_24h(demand_norm))
        d_summer, d_winter = profile_to_seasonal(demand_norm)
        iso_seasonal["demand_summer"] = round_list(d_summer)
        iso_seasonal["demand_winter"] = round_list(d_winter)

        # Solar (Step 1 applies DST-aware nighttime zeroing)
        if "solar" in supply:
            iso_data["solar"] = round_list(profile_to_24h(supply["solar"]))
            s_summer, s_winter = profile_to_seasonal(supply["solar"])
            iso_seasonal["solar_summer"] = round_list(s_summer)
            iso_seasonal["solar_winter"] = round_list(s_winter)

        # Wind
        if "wind" in supply:
            iso_data["wind"] = round_list(profile_to_24h(supply["wind"]))

        # Clean firm / Nuclear (Step 1 applies monthly capacity factor derate)
        if "clean_firm" in supply:
            iso_data["nuclear"] = round_list(profile_to_24h(supply["clean_firm"]))

        # Hydro
        if "hydro" in supply:
            iso_data["hydro"] = round_list(profile_to_24h(supply["hydro"]))

        # Geothermal (CAISO only — flat baseload, no negatives)
        if "geothermal" in supply:
            iso_data["geothermal"] = [1.0] * 24

        # Offshore wind
        if "offshore_wind" in supply:
            iso_data["offshore_wind"] = round_list(profile_to_24h(supply["offshore_wind"]))

        # CCS-CCGT — flat baseload
        iso_data["ccs_ccgt"] = [1.0] * 24

        # Storage shapes (synthetic, same for all ISOs)
        iso_data["battery4_shape"] = list(BATTERY4_SHAPE)
        iso_data["battery8_shape"] = list(BATTERY8_SHAPE)
        iso_data["ldes_shape"] = list(LDES_SHAPE)

        hourly_profiles[iso] = iso_data
        seasonal_profiles[iso] = iso_seasonal

    return hourly_profiles, seasonal_profiles


def extract_mix_data(track="baseline"):
    """Extract resource mix + storage + cost at each threshold for all ISOs.

    track: "baseline" reads step3_co_{ISO}.parquet (medium scenario)
           "dg" reads step3_dg_{ISO}_t{T}.parquet (medium scenario, medium growth)

    Returns: {ISO: {thresholds: [...], mixes: [{resource GW, storage %, cost}]}}
    """
    result = {}

    for iso in ISOS:
        demand_twh = BASE_DEMAND_TWH.get(iso, 200)
        demand_gw = demand_twh / 8.76  # approximate average GW

        geo = "M" if iso == "CAISO" else "X"
        med_scenario = f"MMMM_M_M_M1_{geo}"

        thresholds_found = []
        mixes = []

        if track == "baseline":
            # Baseline: single parquet per ISO with all thresholds
            co_path = STEP3_DIR / f"step_2_2a_CO_{iso}.parquet"
            if not co_path.exists():
                co_path = STEP3_DIR / f"step3_co_{iso}.parquet"
            if not co_path.exists():
                print(f"  {iso}: baseline parquet not found, skipping")
                continue

            df = pd.read_parquet(co_path)
            df_med = df[df["scenario"] == med_scenario]

            for t in ALL_THRESHOLDS:
                row = df_med[df_med["threshold"] == t]
                if len(row) == 0:
                    continue
                row = row.iloc[0]
                mix = extract_row(row, demand_gw)
                thresholds_found.append(t)
                mixes.append(mix)

        else:
            # Demand growth: per-threshold parquets
            for t in ALL_THRESHOLDS:
                t_str = str(int(t)) if t == int(t) else str(t)
                dg_path = STEP3_DIR / f"step_2_2a_DG_{iso}_{t_str}.parquet"
                if not dg_path.exists():
                    dg_path = STEP3_DIR / f"step3_dg_{iso}_t{t_str}.parquet"
                if not dg_path.exists():
                    continue

                df = pd.read_parquet(dg_path)
                df_med = df[(df["scenario"] == med_scenario) &
                            (df["growth_level"] == "Medium")]

                if len(df_med) == 0:
                    continue

                row = df_med.iloc[0]
                mix = extract_row(row, demand_gw)
                thresholds_found.append(t)
                mixes.append(mix)

        if thresholds_found:
            result[iso] = {"thresholds": thresholds_found, "mixes": mixes}
            print(f"  {iso} ({track}): {len(thresholds_found)} thresholds")

    return result


def extract_row(row, demand_gw):
    """Extract a single row into a mix dict with GW values."""
    mix = {}
    for res, col in RESOURCE_COLS.items():
        pct = float(row.get(col, 0) or 0)
        mix[res] = round(pct / 100 * demand_gw, 3)

    # Storage as % of demand
    mix["battery4_pct"] = round(float(row.get("battery_dispatch_pct", 0) or 0), 2)
    mix["battery8_pct"] = round(float(row.get("battery8_dispatch_pct", 0) or 0), 2)
    mix["ldes_pct"] = round(float(row.get("ldes_dispatch_pct", 0) or 0), 2)
    mix["h2_pct"] = round(float(row.get("h2_dispatch_pct", 0) or 0), 2)

    # Cost
    mix["cost"] = round(float(row.get("cost_effective_cost", 0) or 0), 2)

    return mix


def main():
    print("Step 4.1F: Extracting building blocks data")
    print("=" * 60)

    # 1-2. Load hourly + seasonal profiles from Step 1's exact generation shapes
    print("\n1-2. Loading profiles from Step 1 (nuclear derate, solar DST, multi-year avg)...")
    hourly, seasonal = load_step1_profiles()
    for iso in ISOS:
        resources = [k for k in hourly.get(iso, {}).keys()
                     if k not in ("battery4_shape", "battery8_shape", "ldes_shape")]
        print(f"  {iso}: {len(resources)} profiles — {resources}")
        s_keys = list(seasonal.get(iso, {}).keys())
        print(f"  {iso}: {len(s_keys)} seasonal profiles")

    # 3. Baseline mix data (Track 1)
    print("\n3. Extracting baseline mix data (Track 1)...")
    baseline = extract_mix_data("baseline")

    # 4. Demand growth mix data (Track 2)
    print("\n4. Extracting demand growth mix data (Track 2)...")
    dg = extract_mix_data("dg")

    # 5. Write JS output
    print("\n5. Writing JS output...")
    js_parts = [
        f"// Auto-generated by step6d_extract_building_blocks.py",
        f"// Building blocks data: hourly profiles, seasonal, baseline + DG mixes",
        f"// Generated: {datetime.now().isoformat()}",
        f"",
        f"const HOURLY_PROFILES = {json.dumps(hourly, separators=(',', ':'))};",
        f"",
        f"const SEASONAL_PROFILES = {json.dumps(seasonal, separators=(',', ':'))};",
        f"",
        f"const BUILDING_BLOCKS_BASELINE = {json.dumps(baseline, separators=(',', ':'))};",
        f"",
        f"const BUILDING_BLOCKS_DG = {json.dumps(dg, separators=(',', ':'))};",
    ]
    js_content = "\n".join(js_parts) + "\n"

    JS_PATH.parent.mkdir(parents=True, exist_ok=True)
    JS_PATH.write_text(js_content)
    print(f"  Wrote {JS_PATH} ({len(js_content):,} bytes)")

    # 6. Also save JSON archive to step6 output dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    archive = {
        "hourly_profiles": hourly,
        "seasonal_profiles": seasonal,
        "baseline": baseline,
        "demand_growth": dg,
        "generated": datetime.now().isoformat(),
    }
    archive_path = OUT_DIR / "building_blocks.json"
    archive_path.write_text(json.dumps(archive, indent=2))
    print(f"  Wrote {archive_path} ({os.path.getsize(archive_path):,} bytes)")

    print("\nDone!")


if __name__ == "__main__":
    main()
