#!/usr/bin/env python3
"""
Export sweep dispatch data for client-side fleet recalculation.

Reads per-ISO parquets (sweep_1215_ERCOT.parquet, sweep_1215_PJM.parquet, etc.)
OR a combined sweep_1215_flat.parquet, and extracts per-ISO, per-fuel CF and
margin arrays into a compact JSON for browser-side fleet dispatch.

Usage:
    python scripts/export_sweep_dispatch_data.py
    python scripts/export_sweep_dispatch_data.py --sweep-dir ../results/sweep_1215
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
ROOT = SCRIPT_DIR.parent
DEFAULT_SWEEP_DIR = ROOT / "results" / "sweep_1215"
OUTPUT_PATH = ROOT / "frontend" / "data" / "sweep_dispatch_data.json"

# Also write to dashboard/data/ for the main repo
DASHBOARD_OUTPUT_PATH = ROOT.parent / "dashboard" / "data" / "sweep_dispatch_data.json"

FUEL_TYPES = ["gas_ccgt", "gas_ct", "coal_steam", "oil_ct"]
NEW_FOSSIL_COLS = ["total_new_fossil_mw", "gas_built_gw", "fossil_built_gw",
                   "nb_gas_ccgt_mw", "nb_gas_ct_mw", "nb_coal_mw"]

# 5-tier heat-rate bins for per-plant dispatch granularity
HEAT_RATE_TIERS = ["very_low", "low", "medium", "high", "very_high"]
# Fuels that have per-tier columns in the sweep parquet.
# Only export gas_ccgt tiers for now — Constellation fleet focus.
# gas_ct and coal_steam tiers available but excluded to keep JSON < 30 MB.
TIERED_FUELS = ["gas_ccgt"]

ALL_ISOS = ["CAISO", "ERCOT", "MISO", "NEISO", "NYISO", "PJM", "SPP"]


def load_sweep_data(sweep_dir: Path) -> pd.DataFrame:
    """Load sweep data from per-ISO parquets or combined parquet.

    Prefers per-ISO parquets (sweep_1215_<ISO>.parquet) so that ISOs can be
    run independently without overwriting each other. Falls back to the
    combined sweep_1215_flat.parquet for backward compatibility.
    """
    # Try per-ISO parquets first
    iso_parquets = sorted(sweep_dir.glob("sweep_1215_[A-Z]*.parquet"))
    if iso_parquets:
        frames = []
        for p in iso_parquets:
            iso_name = p.stem.replace("sweep_1215_", "")
            if iso_name in ALL_ISOS:
                df_iso = pd.read_parquet(p)
                frames.append(df_iso)
                print(f"  Loaded {p.name}: {len(df_iso)} rows ({iso_name})")
        if frames:
            df = pd.concat(frames, ignore_index=True)
            print(f"  Merged {len(frames)} per-ISO parquets: {len(df)} total rows")
            return df

    # Fall back to combined parquet
    combined = sweep_dir / "sweep_1215_flat.parquet"
    if combined.exists():
        print(f"  Loading combined parquet: {combined}")
        df = pd.read_parquet(combined)
        return df

    print(f"ERROR: No sweep parquets found in {sweep_dir}", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Export sweep dispatch data to JSON')
    parser.add_argument('--sweep-dir', type=Path, default=DEFAULT_SWEEP_DIR,
                        help='Directory containing sweep parquets')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output JSON path (default: frontend/data/sweep_dispatch_data.json)')
    args = parser.parse_args()

    sweep_dir = args.sweep_dir
    output_path = args.output or OUTPUT_PATH

    print(f"Reading sweep data from {sweep_dir}")
    df = load_sweep_data(sweep_dir)
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

        # Per-tier CF and margin arrays (e.g., gas_ccgt_very_low_cf)
        for fuel in TIERED_FUELS:
            for tier in HEAT_RATE_TIERS:
                tier_cf_col = f"ge_{fuel}_{tier}_cf"
                tier_margin_col = f"ge_{fuel}_{tier}_margin_mwh"

                if tier_cf_col in iso_df.columns:
                    cf_arr = np.full((n_scenarios, n_years), 0.0)
                    vals = iso_df[tier_cf_col].fillna(0).values
                    cf_arr[si, yi] = vals
                    iso_data[f"{fuel}_{tier}_cf"] = np.round(cf_arr, 4).tolist()

                if tier_margin_col in iso_df.columns:
                    margin_arr = np.full((n_scenarios, n_years), 0.0)
                    vals = iso_df[tier_margin_col].fillna(0).values
                    margin_arr[si, yi] = vals
                    iso_data[f"{fuel}_{tier}_margin"] = np.round(margin_arr, 2).tolist()

        # New fossil build columns (per-ISO, per-scenario, per-year)
        nf_cols_present = [c for c in NEW_FOSSIL_COLS if c in iso_df.columns]
        for col in nf_cols_present:
            arr = np.full((n_scenarios, n_years), 0.0)
            vals = iso_df[col].fillna(0).values
            arr[si, yi] = vals
            iso_data[col] = np.round(arr, 2).tolist()

        data[iso] = iso_data

    # Collect tier columns that were actually exported
    tier_fuel_types = []
    for fuel in TIERED_FUELS:
        for tier in HEAT_RATE_TIERS:
            col = f"ge_{fuel}_{tier}_cf"
            if col in df.columns:
                tier_fuel_types.append(f"{fuel}_{tier}")

    output = {
        "scenarios": [str(s) for s in scenarios],
        "years": [int(y) for y in years],
        "isos": [str(i) for i in isos],
        "fuel_types": FUEL_TYPES,
        "tier_fuel_types": tier_fuel_types,
        "new_fossil_cols": [c for c in NEW_FOSSIL_COLS if c in df.columns],
        "n_scenarios": n_scenarios,
        "n_years": n_years,
        "data": data,
    }

    # Write to primary output path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Exported {output_path} ({size_mb:.1f} MB)")
    print(f"  {n_scenarios} scenarios × {len(isos)} ISOs × {n_years} years × {len(FUEL_TYPES)} fuels")

    # Also write to dashboard/data/ if it exists
    if DASHBOARD_OUTPUT_PATH.parent.exists() and output_path != DASHBOARD_OUTPUT_PATH:
        with open(DASHBOARD_OUTPUT_PATH, "w") as f:
            json.dump(output, f, separators=(",", ":"))
        print(f"  Also exported to {DASHBOARD_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
