#!/usr/bin/env python3
"""Step 6D — Extract building blocks data for the Building Blocks dashboard page.

Reads Step 3 cost-optimization parquets (both baseline and DG) and EIA hourly
generation profiles to produce:
  1. HOURLY_PROFILES — normalized 24-hour generation shapes per ISO per resource
  2. SEASONAL_PROFILES — monthly capacity factor patterns per ISO per resource
  3. BUILDING_BLOCKS_BASELINE — resource mix & storage at each threshold (Track 1)
  4. BUILDING_BLOCKS_DG — resource mix & storage at each threshold (Track 2)

Output: dashboard/js/building-blocks-data.js

Dependencies:
  - data/step3-cost-opt-parquets/step3_co_{ISO}.parquet (baseline)
  - data/step3-cost-opt-parquets/step3_dg_{ISO}_t{T}.parquet (demand growth)
  - EIA hourly generation profiles in data/ (for hourly shapes)
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
STEP3_DIR = DATA_DIR / "step3-cost-opt-parquets"
OUT_DIR = DATA_DIR / "step6-derived-analytics" / "building-blocks"
JS_PATH = ROOT / "dashboard" / "js" / "building-blocks-data.js"

sys.path.insert(0, str(ROOT))

ISOS = ["CAISO", "ERCOT", "PJM", "NYISO", "NEISO", "MISO", "SPP"]
ALL_THRESHOLDS = [10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 87.5,
                  90, 92.5, 95, 97.5, 99, 99.5, 99.9, 99.99]

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
# These are approximate 2025 annual demands from pipeline_config
BASE_DEMAND_TWH = {
    "CAISO": 224.0, "ERCOT": 405.0, "PJM": 732.0,
    "NYISO": 149.0, "NEISO": 112.0, "MISO": 571.0, "SPP": 245.0,
}


def load_eia_profiles():
    """Load EIA hourly generation profiles and compute 24-hour average shapes.

    Returns: {ISO: {resource: [24 values normalized around 1.0]}}
    """
    profiles = {}
    eia_dir = DATA_DIR / "eia-hourly"

    for iso in ISOS:
        iso_profiles = {}

        # Try loading from EIA multi-year parquet files
        for resource, filename_pattern in [
            ("demand", f"{iso}_demand"),
            ("solar", f"{iso}_solar"),
            ("wind", f"{iso}_wind"),
            ("nuclear", f"{iso}_nuclear"),
            ("hydro", f"{iso}_hydro"),
        ]:
            found = False
            for ext in [".parquet", ".csv"]:
                fpath = eia_dir / f"{filename_pattern}{ext}"
                if fpath.exists():
                    try:
                        if ext == ".parquet":
                            df = pd.read_parquet(fpath)
                        else:
                            df = pd.read_csv(fpath)
                        # Expect column with hourly values; compute 24-hour average
                        val_col = [c for c in df.columns if c not in ("datetime", "timestamp", "hour", "date")]
                        if val_col:
                            vals = df[val_col[0]].values
                            if len(vals) >= 8760:
                                hourly_avg = np.zeros(24)
                                for h in range(24):
                                    hourly_avg[h] = np.mean(vals[h::24])
                                # Normalize around 1.0
                                mean_val = np.mean(hourly_avg)
                                if mean_val > 0:
                                    iso_profiles[resource] = (hourly_avg / mean_val).tolist()
                                    found = True
                    except Exception as e:
                        print(f"  WARNING: Could not load {fpath}: {e}")
                if found:
                    break

        # Offshore wind (CAISO, PJM, NYISO, NEISO only)
        if iso in ["CAISO", "PJM", "NYISO", "NEISO"]:
            osw_path = DATA_DIR / "offshore-wind" / f"{iso}_offshore_wind.parquet"
            if not osw_path.exists():
                osw_path = DATA_DIR / "offshore-wind" / f"{iso}_offshore_wind.csv"
            if osw_path.exists():
                try:
                    df = pd.read_parquet(osw_path) if str(osw_path).endswith(".parquet") else pd.read_csv(osw_path)
                    val_col = [c for c in df.columns if c not in ("datetime", "timestamp", "hour", "date")]
                    if val_col:
                        vals = df[val_col[0]].values
                        if len(vals) >= 8760:
                            hourly_avg = np.zeros(24)
                            for h in range(24):
                                hourly_avg[h] = np.mean(vals[h::24])
                            mean_val = np.mean(hourly_avg)
                            if mean_val > 0:
                                iso_profiles["offshore_wind"] = (hourly_avg / mean_val).tolist()
                except Exception:
                    pass

        # Geothermal (CAISO only) — flat profile
        if iso == "CAISO":
            iso_profiles["geothermal"] = [1.0] * 24

        # CCS-CCGT — flat baseload
        iso_profiles["ccs_ccgt"] = [1.0] * 24

        # Storage shapes (illustrative daily patterns)
        # Battery 4hr: charge midday (solar peak), discharge evening
        bat4 = [0]*6 + [-0.3,-0.5,-0.8,-1.0,-0.8,-0.5,  # charge 6-11
                -0.2,0.0,0.0,0.0, 0.3,0.6,0.9,1.0,0.8,0.5,0.2,0.0]
        iso_profiles["battery4_shape"] = bat4

        # Battery 8hr: wider charge/discharge
        bat8 = [0]*5 + [-0.2,-0.4,-0.6,-0.8,-1.0,-0.8,-0.6,
                -0.3,0.0,0.1,0.3,0.5,0.7,0.9,1.0,0.8,0.5,0.2,0.0]
        iso_profiles["battery8_shape"] = bat8

        # LDES: multi-day (show as broader charge/discharge pattern)
        ldes = [-0.3,-0.3,-0.2,-0.2,-0.1,-0.1,
                -0.4,-0.5,-0.7,-0.8,-0.6,-0.4,
                -0.2,0.1,0.2,0.3,0.5,0.7,0.9,1.0,0.8,0.5,0.2,0.0]
        iso_profiles["ldes_shape"] = ldes

        profiles[iso] = iso_profiles

    return profiles


def load_seasonal_profiles():
    """Compute monthly capacity factor patterns from EIA data.

    Returns: {ISO: {resource: [12 monthly values normalized around 1.0]}}
    """
    seasonal = {}
    eia_dir = DATA_DIR / "eia-hourly"

    for iso in ISOS:
        iso_seasonal = {}

        for resource, filename_pattern in [
            ("demand", f"{iso}_demand"),
            ("solar", f"{iso}_solar"),
            ("wind", f"{iso}_wind"),
        ]:
            for ext in [".parquet", ".csv"]:
                fpath = eia_dir / f"{filename_pattern}{ext}"
                if fpath.exists():
                    try:
                        df = pd.read_parquet(fpath) if ext == ".parquet" else pd.read_csv(fpath)
                        val_col = [c for c in df.columns if c not in ("datetime", "timestamp", "hour", "date")]
                        if val_col and len(df) >= 8760:
                            vals = df[val_col[0]].values[:8760]
                            # Group by month (assuming 8760 hours, standard year)
                            hours_per_month = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
                            monthly = []
                            offset = 0
                            for hrs in hours_per_month:
                                monthly.append(np.mean(vals[offset:offset+hrs]))
                                offset += hrs
                            mean_val = np.mean(monthly)
                            if mean_val > 0:
                                iso_seasonal[resource] = [round(v / mean_val, 4) for v in monthly]
                    except Exception:
                        pass
                    break

        seasonal[iso] = iso_seasonal

    return seasonal


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
    print("Step 6D: Extracting building blocks data")
    print("=" * 60)

    # 1. Hourly profiles
    print("\n1. Loading hourly generation profiles...")
    hourly = load_eia_profiles()
    for iso in ISOS:
        resources = list(hourly.get(iso, {}).keys())
        print(f"  {iso}: {len(resources)} profiles — {resources}")

    # 2. Seasonal profiles
    print("\n2. Loading seasonal profiles...")
    seasonal = load_seasonal_profiles()
    for iso in ISOS:
        resources = list(seasonal.get(iso, {}).keys())
        print(f"  {iso}: {len(resources)} monthly profiles")

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
