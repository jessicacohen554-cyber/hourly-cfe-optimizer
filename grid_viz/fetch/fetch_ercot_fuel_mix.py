#!/usr/bin/env python3
"""
Fetch ERCOT hourly generation by fuel type.

Sources (in preference order):
1. EIA API — Hourly generation by fuel type for ERCOT (balancing authority)
2. Existing repo data — eia_generation_profiles_multiyear.json has 2019-2025
3. ERCOT public reports — Fuel mix reports (backup)

For plant-level renewable/nuclear dispatch, we allocate the hourly aggregate
to individual plants proportionally to their nameplate capacity. This is a
reasonable approximation because:
- Wind output is strongly correlated across a region (weather-driven)
- Solar output is almost perfectly correlated (sun angle)
- Nuclear runs at ~90% CF baseload (nearly constant)
- Hydro is small in ERCOT (~0.4% of capacity)

The CEMS data gives us actual plant-level dispatch for fossil plants, so
this proportional allocation only applies to renewables + nuclear.
"""

import os
import sys
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SCENARIO_DAYS, ERCOT_DIR

# EIA API v2
EIA_API_BASE = "https://api.eia.gov/v2"
EIA_API_KEY = os.environ.get("EIA_API_KEY", "")

# Fuel type mapping from EIA codes to our categories
EIA_FUEL_MAP = {
    "SUN": "SOLAR",
    "WND": "WIND",
    "NG":  "GAS",
    "NUC": "NUCLEAR",
    "COL": "COAL",
    "WAT": "HYDRO",
    "OTH": "OTHER",
    "OIL": "OIL",    # petroleum
    "PET": "OIL",
}


def load_existing_profiles() -> dict | None:
    """Try to load existing EIA generation profiles from repo."""
    path = "data/eia_generation_profiles_multiyear.json"
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        print(f"Loaded existing profiles from {path}")
        return data
    return None


def extract_scenario_from_profiles(
    profiles: dict,
    scenario_id: str,
    scenario_info: dict,
) -> pd.DataFrame | None:
    """
    Extract hourly fuel mix for a scenario date range from existing profiles.

    The existing profiles have structure:
    {
      "ERCOT": {
        "wind": {"2021": [8760 hourly values]},
        "solar": {"2021": [8760 values]},
        ...
      }
    }
    """
    ercot = profiles.get("ERCOT", {})
    if not ercot:
        return None

    begin_str, end_str = scenario_info["date_range"]
    begin_date = datetime.strptime(begin_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_str, "%Y-%m-%d")
    year = str(begin_date.year)

    # Calculate hour-of-year indices
    year_start = datetime(begin_date.year, 1, 1)
    start_hour = int((begin_date - year_start).total_seconds() / 3600)
    end_hour = int((end_date - year_start).total_seconds() / 3600) + 24  # Include full end day

    records = []
    fuel_types_found = []

    for fuel_key, year_data in ercot.items():
        if year not in year_data:
            continue

        hourly = year_data[year]
        if not hourly or len(hourly) < end_hour:
            continue

        fuel_upper = fuel_key.upper()
        fuel_types_found.append(fuel_upper)

        for h_idx in range(start_hour, min(end_hour, len(hourly))):
            hour_of_day = h_idx % 24
            day_offset = (h_idx - start_hour) // 24
            date = begin_date + pd.Timedelta(days=day_offset)

            records.append({
                "date": date.strftime("%Y-%m-%d"),
                "hour": hour_of_day,
                "fuel": fuel_upper,
                "mw": hourly[h_idx],  # EIA data is in MW
            })

    if not records:
        return None

    df = pd.DataFrame(records)
    print(f"  Extracted {len(df)} records for {scenario_id} ({', '.join(fuel_types_found)})")
    return df


def fetch_eia_hourly(
    begin_date: str,
    end_date: str,
    respondent: str = "ERCO",
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    Fetch hourly generation by fuel type from EIA API v2.

    Requires EIA_API_KEY environment variable.
    Falls back to existing repo data if API key not available.
    """
    if not EIA_API_KEY:
        print("  No EIA_API_KEY set. Using existing repo data...")
        profiles = load_existing_profiles()
        if profiles:
            # Find matching scenario
            for sid, info in SCENARIO_DAYS.items():
                if info["date_range"][0] == begin_date:
                    df = extract_scenario_from_profiles(profiles, sid, info)
                    if df is not None and output_path:
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        df.to_csv(output_path, index=False)
                    return df if df is not None else pd.DataFrame()
        return pd.DataFrame()

    # EIA API v2 endpoint for electricity generation
    url = f"{EIA_API_BASE}/electricity/rto/fuel-type-data/data/"
    params = {
        "api_key": EIA_API_KEY,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": respondent,
        "start": begin_date,
        "end": end_date,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
    }

    print(f"  Fetching EIA hourly data: {respondent}, {begin_date} to {end_date}...")
    try:
        resp = requests.get(url, params=params, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            records = data.get("response", {}).get("data", [])
            if records:
                df = pd.DataFrame(records)
                df = df.rename(columns={
                    "period": "datetime",
                    "fueltype": "fuel_code",
                    "value": "mw",
                    "respondent": "iso",
                })
                df["fuel"] = df["fuel_code"].map(EIA_FUEL_MAP).fillna("OTHER")
                df["date"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d")
                df["hour"] = pd.to_datetime(df["datetime"]).dt.hour

                if output_path:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    df.to_csv(output_path, index=False)
                return df
        else:
            print(f"  EIA API returned {resp.status_code}")
    except Exception as e:
        print(f"  EIA API error: {e}")

    return pd.DataFrame()


def fetch_all_scenarios() -> dict[str, pd.DataFrame]:
    """Fetch ERCOT fuel mix data for all scenario days."""

    results = {}

    # First try to use existing repo data
    profiles = load_existing_profiles()

    for scenario_id, info in SCENARIO_DAYS.items():
        begin, end = info["date_range"]
        output_path = os.path.join(ERCOT_DIR, f"fuel_mix_{scenario_id}.csv")

        if os.path.exists(output_path):
            print(f"  [{scenario_id}] Already cached at {output_path}")
            results[scenario_id] = pd.read_csv(output_path)
            continue

        # Try existing profiles first
        if profiles:
            df = extract_scenario_from_profiles(profiles, scenario_id, info)
            if df is not None and len(df) > 0:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                df.to_csv(output_path, index=False)
                results[scenario_id] = df
                continue

        # Fall back to EIA API
        df = fetch_eia_hourly(begin, end, output_path=output_path)
        results[scenario_id] = df

    return results


if __name__ == "__main__":
    os.makedirs(ERCOT_DIR, exist_ok=True)
    results = fetch_all_scenarios()

    print("\n=== Fuel Mix Summary ===")
    for scenario_id, df in results.items():
        if len(df) > 0:
            fuels = df["fuel"].unique() if "fuel" in df.columns else []
            print(f"  {scenario_id}: {len(df)} records, fuels: {', '.join(sorted(fuels))}")
        else:
            print(f"  {scenario_id}: No data")
