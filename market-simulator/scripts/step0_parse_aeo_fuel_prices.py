#!/usr/bin/env python3
"""
Step 0: Parse EIA AEO 2025 Fuel Price Projections
==================================================
Parses Table 1 CSVs from EIA's Annual Energy Outlook 2025 Data Browser
for Reference, Low Oil Price, and High Oil Price cases.

Extracts annual fuel price projections (2023-2050) for:
  - Natural Gas (Henry Hub, real 2024 $/MMBtu)
  - Coal (Delivered to power plants, real 2024 $/MMBtu)
  - Oil (WTI crude, converted from $/barrel to $/MMBtu via 5.8 MMBtu/bbl)

Output: aeo2025_fuel_price_projections.json

Usage:
    python step0_parse_aeo_fuel_prices.py

Requires 3 CSV files in data/eia-natgas-prices/:
    - Table_1._Total_Energy_Supply_Disposition_and_Price_Summary.csv  (Reference)
    - Table_1_Low_Oil_Price.csv
    - Table_1_High_Oil_Price.csv
"""

import csv
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'data', 'eia-natgas-prices')

# CSV file mapping: case label -> filename
CASE_FILES = {
    'Medium': 'Table_1._Total_Energy_Supply_Disposition_and_Price_Summary.csv',
    'Low': 'Table_1_Low_Oil_Price.csv',
    'High': 'Table_1_High_Oil_Price.csv',
}

# Row identifiers (partial match on the API key column)
# These identify the specific price series we need from Table 1
SERIES_IDENTIFIERS = {
    'gas': 'PRCE_RLP_TEN_NA_HHP_NA_USA_Y13DLRPMMBTU',    # Henry Hub real $/MMBtu
    'coal': 'PRCE_RLP_TEN_NA_DCL_NA_USA_Y13DLRPMMBTU',    # Delivered coal real $/MMBtu
    'oil': 'PRCE_RLP_TEN_NA_WTI_NA_USA_Y13DLRPBBL',       # WTI real $/barrel
}

# Oil conversion: $/barrel -> $/MMBtu
# Crude oil energy content: ~5.8 MMBtu per barrel (EIA standard)
OIL_MMBTU_PER_BARREL = 5.8

OUTPUT_FILE = os.path.join(DATA_DIR, 'aeo2025_fuel_price_projections.json')


def parse_table1_csv(filepath: str) -> dict:
    """Parse an AEO Table 1 CSV and extract fuel price time series.

    Returns:
        dict: {fuel_type: {year_str: price_mmbtu}} for gas, coal, oil
    """
    if not os.path.exists(filepath):
        print(f"  WARNING: File not found: {filepath}")
        return {}

    results = {}

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Find the header row with year columns (row index 4, 0-indexed)
    # Format: "", full_name, api_key, units, "2023", "2024", ..., "2050", "Growth..."
    header_row = None
    year_cols = {}  # year -> column index

    for i, row in enumerate(rows):
        if len(row) > 5 and row[3].strip().lower() == 'units':
            header_row = i
            for j in range(4, len(row)):
                val = row[j].strip().strip('"')
                try:
                    yr = int(val)
                    if 2023 <= yr <= 2050:
                        year_cols[yr] = j
                except ValueError:
                    pass
            break

    if header_row is None:
        print(f"  WARNING: Could not find header row in {filepath}")
        return {}

    print(f"  Found {len(year_cols)} year columns: {min(year_cols)}-{max(year_cols)}")

    # In the header, the API key column is at index 2.
    # Some data rows have commas in the first column (e.g., "Coal, Delivered")
    # which shifts all subsequent columns. We detect this by finding the API key
    # column in each row, then computing the offset from the expected position (2).
    EXPECTED_API_KEY_COL = 2

    for fuel_type, api_key_fragment in SERIES_IDENTIFIERS.items():
        for row in rows[header_row + 1:]:
            if len(row) < 4:
                continue
            # Find which column contains the API key
            api_key_col = None
            for ci, cell in enumerate(row):
                if api_key_fragment in cell:
                    api_key_col = ci
                    break
            if api_key_col is None:
                continue

            # Column offset: how many extra columns this row has vs the header
            col_offset = api_key_col - EXPECTED_API_KEY_COL

            prices = {}
            for yr, col_idx in sorted(year_cols.items()):
                shifted_idx = col_idx + col_offset
                if 0 <= shifted_idx < len(row):
                    val_str = row[shifted_idx].strip().strip('"')
                    try:
                        price = float(val_str)
                        if fuel_type == 'oil':
                            price = round(price / OIL_MMBTU_PER_BARREL, 4)
                        else:
                            price = round(price, 4)
                        prices[str(yr)] = price
                    except (ValueError, TypeError):
                        pass
            if prices:
                results[fuel_type] = prices
                sample_yr = '2030' if '2030' in prices else list(prices.keys())[len(prices) // 2]
                print(f"  {fuel_type}: {len(prices)} years, "
                      f"2025={prices.get('2025', 'N/A')}, "
                      f"{sample_yr}={prices.get(sample_yr, 'N/A')} $/MMBtu"
                      f"{' (col_offset=' + str(col_offset) + ')' if col_offset else ''}")
            break

    return results


def main():
    print("=" * 70)
    print("EIA AEO 2025 Fuel Price Projections Parser")
    print("=" * 70)

    output = {
        'source': 'EIA Annual Energy Outlook 2025',
        'units': '2024 $/MMBtu (real)',
        'oil_conversion': f'WTI $/barrel ÷ {OIL_MMBTU_PER_BARREL} MMBtu/bbl',
        'cases': {},
    }

    missing_files = []
    for case_label, filename in CASE_FILES.items():
        filepath = os.path.join(DATA_DIR, filename)
        print(f"\nParsing {case_label} case: {filename}")

        if not os.path.exists(filepath):
            missing_files.append((case_label, filename))
            print(f"  SKIPPED — file not found")
            continue

        prices = parse_table1_csv(filepath)
        if prices:
            output['cases'][case_label] = prices
        else:
            print(f"  WARNING: No price data extracted")

    if missing_files:
        print(f"\n{'=' * 70}")
        print("MISSING FILES — download from EIA AEO 2025 Data Browser:")
        print("  https://www.eia.gov/outlooks/aeo/data/browser/")
        for case_label, filename in missing_files:
            print(f"  - {case_label}: save as {filename}")
        print(f"  Into: {DATA_DIR}")

    if output['cases']:
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nWrote {OUTPUT_FILE}")
        print(f"Cases: {list(output['cases'].keys())}")
    else:
        print("\nERROR: No data parsed. Cannot write output.")
        sys.exit(1)

    # Print summary table
    print(f"\n{'=' * 70}")
    print("Summary: Fuel Price Projections (2024 $/MMBtu)")
    print(f"{'Year':<8}", end='')
    for case in ['Low', 'Medium', 'High']:
        if case in output['cases']:
            print(f"{'Gas-' + case:<12}{'Coal-' + case:<12}{'Oil-' + case:<12}", end='')
    print()
    print("-" * 70)

    for yr in ['2025', '2030', '2035', '2040', '2045', '2050']:
        print(f"{yr:<8}", end='')
        for case in ['Low', 'Medium', 'High']:
            if case in output['cases']:
                c = output['cases'][case]
                g = c.get('gas', {}).get(yr, '-')
                cl = c.get('coal', {}).get(yr, '-')
                o = c.get('oil', {}).get(yr, '-')
                print(f"{g:<12}{cl:<12}{o:<12}", end='')
        print()


if __name__ == '__main__':
    main()
