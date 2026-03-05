#!/usr/bin/env python3
"""Step 6E — Extract per-scenario resource density data for the abatement dashboard.

Reads Step 3 DG parquets for thresholds in each ISO's optimal crossover range,
computes new-build TWh per resource per cost scenario, and outputs a JS data file
for the resource density strip plot on the CO2 abatement page.

Output:
  - dashboard/js/resource-density-data.js
  - data/step6-derived-analytics/resource-density/resource_density.json

Dependencies: Step 3 DG parquets, optimal_targets.json (for crossover ranges)
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
DG_DIR = ROOT / "data" / "step3-cost-opt-parquets"
OT_PATH = ROOT / "data" / "step5-post-processing" / "optimal_targets.json"
OUT_PATH = ROOT / "dashboard" / "js" / "resource-density-data.js"
ARCHIVE_DIR = ROOT / "data" / "step6-derived-analytics" / "resource-density"

sys.path.insert(0, str(ROOT / "scripts"))
from pipeline_config import ACTIVE_THRESHOLDS, THRESHOLD_TARGET_YEARS

ISOS = ["CAISO", "ERCOT", "PJM", "NYISO", "NEISO", "MISO", "SPP"]
# Single source of truth: pipeline_config
ALL_THRESHOLDS = ACTIVE_THRESHOLDS

# Existing clean resources as % of demand (2025 baseline)
EXISTING_PCT = {
    "CAISO":  {"nuclear": 7.9, "geothermal": 0, "solar": 22.3, "wind": 8.8, "ccs_ccgt": 0},
    "ERCOT":  {"nuclear": 8.6, "geothermal": 0, "solar": 13.8, "wind": 23.6, "ccs_ccgt": 0},
    "PJM":    {"nuclear": 32.1, "geothermal": 0, "solar": 2.9, "wind": 3.8, "ccs_ccgt": 0},
    "NYISO":  {"nuclear": 18.4, "geothermal": 0, "solar": 0.0, "wind": 4.7, "ccs_ccgt": 0},
    "NEISO":  {"nuclear": 23.8, "geothermal": 0, "solar": 1.4, "wind": 3.9, "ccs_ccgt": 0},
    "MISO":   {"nuclear": 13.1, "geothermal": 0, "solar": 2.1, "wind": 14.5, "ccs_ccgt": 0},
    "SPP":    {"nuclear": 5.2, "geothermal": 0, "solar": 0.4, "wind": 37.1, "ccs_ccgt": 0},
}

# Scenario key format: {Ren}{Firm}{Batt}{LDES}_{Fuel}_{Tx}_{CCS}{45Q}_{Geo}
# First 4 chars = Renewable, Firm, Battery, LDES levels (L/M/H)
# Then Fuel, Tx, CCS levels, 45Q (0/1), Geo (L/M/H/X)
TOGGLE_NAMES = {
    0: "Renewable Gen",
    1: "Firm Gen",
    2: "Battery Storage",
    3: "LDES Storage",
    "fuel": "Fossil Fuel",
    "tx": "Transmission",
    "ccs": "CCS",
    "q45": "45Q Credit",
    "geo": "Geothermal",
}
LEVEL_MAP = {"L": "Low", "M": "Medium", "H": "High", "N": "None",
             "0": "Off", "1": "On", "X": "N/A"}


def parse_scenario(sc_key):
    """Parse scenario key into toggle dict."""
    parts = sc_key.split("_")
    gen = parts[0]  # RFBL (4 chars)
    fuel = parts[1]
    tx = parts[2]
    ccs_q45 = parts[3]  # e.g. M1
    geo = parts[4] if len(parts) > 4 else "X"
    return {
        "ren": gen[0], "firm": gen[1], "batt": gen[2], "ldes": gen[3],
        "fuel": fuel, "tx": tx, "ccs": ccs_q45[0], "q45": ccs_q45[1],
        "geo": geo,
    }


def get_crossover_thresholds(ot_data, iso):
    """Get thresholds within the ISO's optimal crossover range."""
    iso_data = ot_data.get(iso, {})

    # Try crossover_scenarios for P10-P90 range
    scenarios = iso_data.get("crossover_scenarios", {})
    thresholds = set()
    for sc_name, sc_data in scenarios.items():
        t = sc_data.get("threshold")
        if t is not None:
            thresholds.add(float(t))

    if thresholds:
        lower = min(thresholds)
        upper = max(thresholds)
    else:
        # Fallback to crossover_central
        cc = iso_data.get("crossover_central", {})
        central = cc.get("threshold", 90)
        lower = max(50, central - 15)
        upper = min(99.99, central + 5)

    # Expand to include a few thresholds around the range for context
    lower = max(50, lower - 5)
    upper = min(99.99, upper + 2.5)

    return [t for t in ALL_THRESHOLDS if lower <= t <= upper]


def analyze_resource_drivers(df_resource, toggle_col_map):
    """Identify which cost toggles most influence a resource's presence/level.

    Returns a short description of the key price conditions.
    """
    if len(df_resource) < 10:
        return ""

    twh_col = "twh"
    results = []

    for toggle_name, col_func in toggle_col_map.items():
        vals = df_resource.apply(col_func, axis=1)
        groups = df_resource.groupby(vals)[twh_col]
        means = groups.mean()
        if len(means) < 2:
            continue
        spread = means.max() - means.min()
        if spread < 0.5:
            continue
        best_level = means.idxmax()
        results.append((spread, toggle_name, LEVEL_MAP.get(best_level, best_level)))

    results.sort(reverse=True)
    if not results:
        return "Appears across most cost assumptions"

    top = results[:2]
    parts = []
    for _, toggle, level in top:
        cost_desc = "low" if level == "Low" else ("high" if level == "High" else "medium")
        parts.append(f"{cost_desc}-cost {toggle.lower()}")
    return "Favored under " + " and ".join(parts) + " assumptions"


def process_iso(iso, ot_data):
    """Process a single ISO and return density data dict."""
    thresholds = get_crossover_thresholds(ot_data, iso)
    if not thresholds:
        print(f"  {iso}: no crossover thresholds found, skipping")
        return None

    print(f"  {iso}: thresholds {thresholds}")

    frames = []
    for t in thresholds:
        t_str = str(int(t)) if t == int(t) else str(t)
        path = DG_DIR / f"step3_dg_{iso}_t{t_str}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        # Filter to medium growth only
        df = df[df["growth_level"] == "Medium"].copy()
        frames.append(df)

    if not frames:
        print(f"  {iso}: no DG parquets found")
        return None

    df = pd.concat(frames, ignore_index=True)
    print(f"  {iso}: {len(df)} scenario rows across {len(thresholds)} thresholds")

    exist = EXISTING_PCT.get(iso, {})
    demand_twh = df["annual_demand_mwh"].iloc[0] / 1e6  # MWh → TWh

    # Compute new-build TWh per resource
    resources = {}

    # Solar (new-build only)
    solar_total = df["mix_solar"] / 100 * demand_twh
    solar_exist = exist.get("solar", 0) / 100 * demand_twh
    resources["solar"] = np.maximum(0, solar_total - solar_exist)

    # Onshore Wind (new-build only)
    wind_total = df["mix_wind"] / 100 * demand_twh
    wind_exist = exist.get("wind", 0) / 100 * demand_twh
    resources["wind"] = np.maximum(0, wind_total - wind_exist)

    # Nuclear (from tranche data)
    resources["nuclear"] = np.maximum(0,
        df["tranche_nuclear_newbuild_twh"].fillna(0) + df["tranche_uprate_twh"].fillna(0))

    # Geothermal (from tranche data, CAISO only)
    resources["geothermal"] = np.maximum(0, df["tranche_geo_twh"].fillna(0))

    # CCS-CCGT
    resources["ccs_ccgt"] = np.maximum(0, df["tranche_ccs_tranche_twh"].fillna(0))

    # Battery 4hr
    resources["battery"] = np.maximum(0, df["battery_dispatch_pct"].fillna(0) / 100 * demand_twh)

    # Battery 8hr
    resources["battery8"] = np.maximum(0, df["battery8_dispatch_pct"].fillna(0) / 100 * demand_twh)

    # LDES
    resources["ldes"] = np.maximum(0, df["ldes_dispatch_pct"].fillna(0) / 100 * demand_twh)

    # Parse scenario toggles for driver analysis
    toggle_extractors = {
        "Renewable Gen": lambda row: parse_scenario(row["scenario"])["ren"],
        "Firm Gen": lambda row: parse_scenario(row["scenario"])["firm"],
        "Fossil Fuel": lambda row: parse_scenario(row["scenario"])["fuel"],
        "Transmission": lambda row: parse_scenario(row["scenario"])["tx"],
        "CCS": lambda row: parse_scenario(row["scenario"])["ccs"],
        "45Q Credit": lambda row: parse_scenario(row["scenario"])["q45"],
    }
    if iso == "CAISO":
        toggle_extractors["Geothermal"] = lambda row: parse_scenario(row["scenario"])["geo"]

    # Build output for each resource
    result = {"thresholds": thresholds, "totalScenarios": len(df), "resources": {}}

    for res_key, twh_series in resources.items():
        twh_arr = twh_series.values
        nonzero_mask = twh_arr > 0.1

        presence_rate = float(nonzero_mask.sum() / len(twh_arr)) if len(twh_arr) > 0 else 0
        if nonzero_mask.sum() == 0:
            # Resource not present
            result["resources"][res_key] = {
                "presenceRate": 0,
                "stats": None,
                "points": [],
                "driver": "Not in optimal mix for this region"
            }
            continue

        nz_vals = twh_arr[nonzero_mask]
        stats = {
            "p10": float(np.percentile(nz_vals, 10)),
            "p25": float(np.percentile(nz_vals, 25)),
            "median": float(np.median(nz_vals)),
            "p75": float(np.percentile(nz_vals, 75)),
            "p90": float(np.percentile(nz_vals, 90)),
            "mean": float(np.mean(nz_vals)),
            "std": float(np.std(nz_vals)),
            "min": float(np.min(nz_vals)),
            "max": float(np.max(nz_vals)),
        }

        # Sample up to 2000 points per resource for strip plot
        # At 8% opacity, 2000 dots produces excellent density visualization
        max_points = 2000
        if len(twh_arr) <= max_points:
            sample_vals = twh_arr.tolist()
        else:
            indices = np.random.RandomState(42).choice(len(twh_arr), size=max_points, replace=False)
            sample_vals = twh_arr[indices].tolist()

        # Round to 1 decimal place for compact JS output
        sample_vals = [round(v, 1) for v in sample_vals]

        # Driver analysis
        df_tmp = df.copy()
        df_tmp["twh"] = twh_arr
        driver = analyze_resource_drivers(df_tmp[nonzero_mask], toggle_extractors)

        result["resources"][res_key] = {
            "presenceRate": round(presence_rate, 4),
            "stats": {k: round(v, 2) for k, v in stats.items()},
            "points": sample_vals,
            "driver": driver,
        }

    return result


def main():
    print("Step 6E: Extracting resource density data for abatement dashboard")

    # Load optimal targets for crossover ranges
    if OT_PATH.exists():
        with open(OT_PATH) as f:
            ot_data = json.load(f)
        print(f"Loaded optimal_targets.json ({len(ot_data)} ISOs)")
    else:
        print("WARNING: optimal_targets.json not found, using default ranges")
        ot_data = {}

    all_data = {}
    for iso in ISOS:
        data = process_iso(iso, ot_data)
        if data:
            all_data[iso] = data

    # Write JS output
    js_content = (
        f"// Auto-generated by step6e_extract_resource_density.py\n"
        f"// Per-scenario resource density data for strip plots\n"
        f"// Generated: {datetime.now().isoformat()}\n"
        f"\n"
        f"const RESOURCE_DENSITY_DATA = {json.dumps(all_data, separators=(',', ':'))};\n"
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(js_content)
    print(f"\nWrote {OUT_PATH} ({len(js_content):,} bytes)")

    # Also write JSON archive to step6 output directory
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = ARCHIVE_DIR / "resource_density.json"
    archive_path.write_text(json.dumps(all_data, indent=2))
    print(f"Wrote {archive_path} ({os.path.getsize(archive_path):,} bytes)")

    print(f"ISOs: {list(all_data.keys())}")
    for iso, data in all_data.items():
        active = [k for k, v in data["resources"].items() if v["presenceRate"] > 0]
        print(f"  {iso}: {data['totalScenarios']} scenarios, active resources: {active}")


if __name__ == "__main__":
    main()
