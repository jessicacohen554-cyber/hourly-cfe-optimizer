#!/usr/bin/env python3
"""
Fetch eGRID plant data for ERCOT — lat/lon, fuel type, capacity, emissions.

Uses the existing eGRID 2023 Excel file already in the repo.
Outputs a clean CSV with one row per plant for map visualization.
"""

import os
import sys
import pandas as pd

# Column indices in eGRID PLNT23 sheet (0-indexed)
EGRID_COLS = {
    "SEQPLT":   0,   # Sequence number
    "YEAR":     1,   # Data year
    "PSTATABB": 2,   # State abbreviation
    "PNAME":    3,   # Plant name
    "ORISPL":   4,   # ORIS plant code (unique ID)
    "OPRNAME":  5,   # Operator name
    "UTLSRVNM": 6,  # Utility name
    "SECTOR":   9,   # Sector
    "BANAME":   10,  # Balancing authority name
    "BACODE":   11,  # Balancing authority code
    "NERC":     12,  # NERC region
    "SUBRGN":   13,  # eGRID subregion code
    "FIPS":     16,  # FIPS state code
    "COUNTY":   17,  # County name (check)
    "LAT":      19,  # Latitude
    "LON":      20,  # Longitude
    "PLPRMFL":  24,  # Primary fuel code
    "PLFUELCT": 25,  # Primary fuel category
    "CAPFAC":   27,  # Capacity factor
    "NAMEPCAP": 28,  # Nameplate capacity (MW)
    "PLNGENAN": 40,  # Annual net generation (MWh)
    "CO2_TONS": 46,  # Annual CO2 emissions (tons)
    "CO2_RATE": 54,  # CO2 output emission rate (lb/MWh)
}


def extract_ercot_plants(egrid_path: str, output_path: str, ba_code: str = "ERCO") -> pd.DataFrame:
    """Extract all plants for a given balancing authority from eGRID."""

    print(f"Reading eGRID data from {egrid_path}...")
    df_raw = pd.read_excel(
        egrid_path,
        sheet_name="PLNT23",
        header=0,      # First row is descriptive headers
        skiprows=[1],   # Second row is column codes — skip it
    )

    # The actual column names are the descriptive headers from row 0
    # Map them to our short names using position
    cols = list(df_raw.columns)

    # Build rename map from position
    rename_map = {}
    for short_name, idx in EGRID_COLS.items():
        if idx < len(cols):
            rename_map[cols[idx]] = short_name

    df = df_raw.rename(columns=rename_map)

    # Filter to target BA
    print(f"Filtering for BA code = {ba_code}...")
    mask = df["BACODE"] == ba_code
    df_ba = df.loc[mask].copy()
    print(f"  Found {len(df_ba)} plants")

    # Select and clean columns
    keep_cols = [
        "ORISPL", "PNAME", "OPRNAME", "PSTATABB", "SECTOR",
        "LAT", "LON", "PLPRMFL", "PLFUELCT",
        "NAMEPCAP", "CAPFAC", "PLNGENAN", "CO2_TONS", "CO2_RATE",
        "NERC", "SUBRGN",
    ]
    # Only keep columns that exist
    keep_cols = [c for c in keep_cols if c in df_ba.columns]
    df_out = df_ba[keep_cols].copy()

    # Clean up types
    for col in ["LAT", "LON", "NAMEPCAP", "CAPFAC", "PLNGENAN", "CO2_TONS", "CO2_RATE"]:
        if col in df_out.columns:
            df_out[col] = pd.to_numeric(df_out[col], errors="coerce")

    df_out["ORISPL"] = df_out["ORISPL"].astype(int)

    # Drop plants with no coordinates
    n_before = len(df_out)
    df_out = df_out.dropna(subset=["LAT", "LON"])
    n_dropped = n_before - len(df_out)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} plants with missing coordinates")

    # Add display-friendly fuel label
    fuel_labels = {
        "GAS": "Natural Gas",
        "COAL": "Coal",
        "OIL": "Oil",
        "NUCLEAR": "Nuclear",
        "WIND": "Wind",
        "SOLAR": "Solar",
        "HYDRO": "Hydro",
        "BIOMASS": "Biomass",
        "OTHF": "Other",
        "OFSL": "Other Fossil",
        "GEOTHERMAL": "Geothermal",
    }
    df_out["FUEL_LABEL"] = df_out["PLFUELCT"].map(fuel_labels).fillna("Other")

    # Sort by capacity descending
    df_out = df_out.sort_values("NAMEPCAP", ascending=False).reset_index(drop=True)

    # Summary
    print(f"\n=== ERCOT Plant Summary ===")
    print(f"Total plants: {len(df_out)}")
    print(f"Total capacity: {df_out['NAMEPCAP'].sum():,.0f} MW")
    print(f"\nBy fuel type:")
    fuel_summary = df_out.groupby("PLFUELCT").agg(
        count=("ORISPL", "count"),
        capacity_mw=("NAMEPCAP", "sum"),
        gen_mwh=("PLNGENAN", "sum"),
    ).sort_values("capacity_mw", ascending=False)
    for fuel, row in fuel_summary.iterrows():
        print(f"  {fuel:12s}: {int(row['count']):4d} plants, {row['capacity_mw']:10,.0f} MW, {row['gen_mwh']:14,.0f} MWh")

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    return df_out


if __name__ == "__main__":
    egrid_path = "data/egrid2023_data_rev2 2.xlsx"
    output_path = "grid_viz/data/egrid/ercot_plants.csv"

    if not os.path.exists(egrid_path):
        print(f"ERROR: eGRID file not found at {egrid_path}")
        sys.exit(1)

    df = extract_ercot_plants(egrid_path, output_path)
