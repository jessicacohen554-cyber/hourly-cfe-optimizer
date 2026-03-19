#!/usr/bin/env python3
"""
Export sweep dispatch data for client-side fleet recalculation.

Reads sweep_405_flat.parquet and extracts per-ISO, per-fuel CF and margin arrays
into a compact JSON that the browser can use for real-time fleet dispatch.

Usage:
    python scripts/export_sweep_dispatch_data.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
ROOT = SCRIPT_DIR.parent
SWEEP_PATH = ROOT / "results" / "sweep_405" / "sweep_405_flat.parquet"
OUTPUT_PATH = ROOT / "frontend" / "data" / "sweep_dispatch_data.json"

FUEL_TYPES = ["gas_ccgt", "gas_ct", "coal_steam", "oil_ct"]


def main():
    if not SWEEP_PATH.exists():
        print(f"ERROR: {SWEEP_PATH} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {SWEEP_PATH}")
    df = pd.read_parquet(SWEEP_PATH)
    print(f"  {len(df)} rows, {df['scenario'].nunique()} scenarios, "
          f"{df['iso'].nunique()} ISOs, {df['year'].nunique()} years")

    scenarios = sorted(df["scenario"].unique())
    years = sorted(df["year"].unique())
    isos = sorted(df["iso"].unique())

    scenario_idx = {s: i for i, s in enumerate(scenarios)}
    year_idx = {y: i for i, y in enumerate(years)}

    n_scenarios = len(scenarios)
    n_years = len(years)

    data = {}
    for iso in isos:
        iso_df = df[df["iso"] == iso].copy()
        si = iso_df["scenario"].map(scenario_idx).values.astype(int)
        yi = iso_df["year"].map(year_idx).values.astype(int)

        iso_data = {}
        for fuel in FUEL_TYPES:
            cf_col = f"ge_{fuel}_cf"
            margin_col = f"ge_{fuel}_margin_mwh"

            # Build (n_scenarios, n_years) arrays
            cf_arr = np.full((n_scenarios, n_years), 0.0)
            margin_arr = np.full((n_scenarios, n_years), 0.0)

            if cf_col in iso_df.columns:
                vals = iso_df[cf_col].fillna(0).values
                cf_arr[si, yi] = vals
            if margin_col in iso_df.columns:
                vals = iso_df[margin_col].fillna(0).values
                margin_arr[si, yi] = vals

            # Round to 4 decimal places to reduce JSON size
            iso_data[f"{fuel}_cf"] = np.round(cf_arr, 4).tolist()
            iso_data[f"{fuel}_margin"] = np.round(margin_arr, 2).tolist()

        data[iso] = iso_data

    output = {
        "scenarios": [str(s) for s in scenarios],
        "years": [int(y) for y in years],
        "isos": [str(i) for i in isos],
        "fuel_types": FUEL_TYPES,
        "n_scenarios": n_scenarios,
        "n_years": n_years,
        "data": data,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"Exported {OUTPUT_PATH} ({size_mb:.1f} MB)")
    print(f"  {n_scenarios} scenarios × {len(isos)} ISOs × {n_years} years × {len(FUEL_TYPES)} fuels")


if __name__ == "__main__":
    main()
