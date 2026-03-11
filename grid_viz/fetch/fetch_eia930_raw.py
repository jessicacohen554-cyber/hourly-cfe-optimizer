#!/usr/bin/env python3
"""
Fetch RAW (not averaged) hourly EIA-930 generation by fuel type for specific date ranges.

Unlike the main pipeline's step0_fetch_eia_multiyear.py which averages profiles
across years, this fetches actual historical hourly MW for each fuel type on
specific dates — critical for event-driven visualizations like Winter Storm Uri
where the actual hour-by-hour generation collapse matters.

Uses EIA API v2: https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/

Outputs CSV with columns: datetime_utc, datetime_local, fuel, mw
One file per scenario: grid_viz/data/ercot/raw_generation_{scenario_id}.csv

Usage:
    export EIA_API_KEY=your_key
    python grid_viz/fetch/fetch_eia930_raw.py
    python grid_viz/fetch/fetch_eia930_raw.py --scenario uri_2021
"""

import argparse
import os
import sys
import time
import json
import requests
import pandas as pd
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SCENARIO_DAYS, ERCOT_DIR

# ── Constants ──

EIA_BASE_URL = "https://api.eia.gov/v2/electricity/rto/"
FUEL_TYPE_ENDPOINT = "fuel-type-data/data/"
DEMAND_ENDPOINT = "region-data/data/"

BA_CODE = "ERCO"  # ERCOT balancing authority
LOCAL_TZ = ZoneInfo("America/Chicago")

# EIA fuel type codes → our display categories
FUEL_MAP = {
    "SUN": "SOLAR",
    "WND": "WIND",
    "NUC": "NUCLEAR",
    "WAT": "HYDRO",
    "NG":  "GAS",
    "COL": "COAL",
    "OIL": "OIL",
    "OTH": "OTHER",
    "GEO": "GEOTHERMAL",
}

MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0


def fetch_eia_page(url: str, params: dict, api_key: str) -> dict | None:
    """Fetch one page from EIA API with retry."""
    params = dict(params)
    params["api_key"] = api_key

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                if "response" in data and "data" in data["response"]:
                    return data["response"]
                elif "error" in data:
                    print(f"  API error: {data['error']}")
                    return None
            elif resp.status_code == 429:
                backoff = INITIAL_BACKOFF * (2 ** attempt)
                print(f"  Rate limited. Retrying in {backoff:.0f}s...")
                time.sleep(backoff)
                continue
            elif resp.status_code >= 500:
                backoff = INITIAL_BACKOFF * (2 ** attempt)
                print(f"  Server error ({resp.status_code}). Retrying in {backoff:.0f}s...")
                time.sleep(backoff)
                continue
            else:
                print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
                return None
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            backoff = INITIAL_BACKOFF * (2 ** attempt)
            print(f"  {type(e).__name__}. Retrying in {backoff:.0f}s...")
            time.sleep(backoff)

    print(f"  FAILED after {MAX_RETRIES} attempts")
    return None


def fetch_all_pages(url: str, params: dict, api_key: str) -> list:
    """Paginate through all results."""
    all_rows = []
    offset = 0
    page_size = 5000

    while True:
        page_params = dict(params)
        page_params["offset"] = offset
        page_params["length"] = page_size

        response = fetch_eia_page(url, page_params, api_key)
        if response is None:
            break

        rows = response.get("data", [])
        total = int(response.get("total", 0))
        all_rows.extend(rows)

        if len(all_rows) >= total or len(rows) == 0:
            break
        offset += page_size
        time.sleep(0.2)  # Rate limiting

    return all_rows


def fetch_scenario_generation(
    scenario_id: str,
    begin_date: str,
    end_date: str,
    api_key: str,
) -> pd.DataFrame:
    """
    Fetch actual hourly generation by fuel type for a date range.
    Returns DataFrame with datetime_utc, datetime_local, fuel, mw columns.
    """
    url = EIA_BASE_URL + FUEL_TYPE_ENDPOINT

    # EIA uses UTC timestamps in format YYYY-MM-DDTHH
    params = {
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": BA_CODE,
        "start": f"{begin_date}T00",
        "end": f"{end_date}T23",
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
    }

    print(f"  Fetching generation: {BA_CODE}, {begin_date} to {end_date}...")
    rows = fetch_all_pages(url, params, api_key)
    print(f"    Got {len(rows)} records")

    if not rows:
        return pd.DataFrame()

    records = []
    for row in rows:
        period = row.get("period", "")
        fuel_code = row.get("fueltype", row.get("fueltypeid", ""))
        value = row.get("value")

        if value is None or fuel_code not in FUEL_MAP:
            continue

        # Parse UTC timestamp
        try:
            if len(period) == 13:  # "2021-02-15T08"
                utc_dt = datetime(
                    int(period[:4]), int(period[5:7]), int(period[8:10]),
                    int(period[11:13]), tzinfo=timezone.utc,
                )
            else:
                utc_dt = datetime.strptime(period[:16], "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
        except (ValueError, IndexError):
            continue

        local_dt = utc_dt.astimezone(LOCAL_TZ)

        records.append({
            "datetime_utc": utc_dt.strftime("%Y-%m-%d %H:%M"),
            "datetime_local": local_dt.strftime("%Y-%m-%d %H:%M"),
            "date_local": local_dt.strftime("%Y-%m-%d"),
            "hour_local": local_dt.hour,
            "fuel": FUEL_MAP[fuel_code],
            "fuel_code": fuel_code,
            "mw": float(value),
        })

    df = pd.DataFrame(records)
    if len(df) > 0:
        print(f"    Parsed {len(df)} gen records")
        print(f"    Fuels: {sorted(df['fuel'].unique())}")
        print(f"    Date range: {df['datetime_local'].min()} to {df['datetime_local'].max()}")

        # Quick summary
        pivot = df.groupby("fuel")["mw"].agg(["mean", "max"]).sort_values("max", ascending=False)
        for fuel, row in pivot.iterrows():
            print(f"      {fuel:10s}: avg {row['mean']:8,.0f} MW, peak {row['max']:8,.0f} MW")

    return df


def fetch_scenario_demand(
    scenario_id: str,
    begin_date: str,
    end_date: str,
    api_key: str,
) -> pd.DataFrame:
    """Fetch actual hourly demand for a date range."""
    url = EIA_BASE_URL + DEMAND_ENDPOINT

    params = {
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": BA_CODE,
        "facets[type][]": "D",
        "start": f"{begin_date}T00",
        "end": f"{end_date}T23",
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
    }

    print(f"  Fetching demand: {BA_CODE}, {begin_date} to {end_date}...")
    rows = fetch_all_pages(url, params, api_key)
    print(f"    Got {len(rows)} demand records")

    if not rows:
        return pd.DataFrame()

    records = []
    for row in rows:
        period = row.get("period", "")
        value = row.get("value")
        if value is None:
            continue

        try:
            if len(period) == 13:
                utc_dt = datetime(
                    int(period[:4]), int(period[5:7]), int(period[8:10]),
                    int(period[11:13]), tzinfo=timezone.utc,
                )
            else:
                utc_dt = datetime.strptime(period[:16], "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
        except (ValueError, IndexError):
            continue

        local_dt = utc_dt.astimezone(LOCAL_TZ)

        records.append({
            "datetime_utc": utc_dt.strftime("%Y-%m-%d %H:%M"),
            "datetime_local": local_dt.strftime("%Y-%m-%d %H:%M"),
            "date_local": local_dt.strftime("%Y-%m-%d"),
            "hour_local": local_dt.hour,
            "demand_mw": float(value),
        })

    df = pd.DataFrame(records)
    if len(df) > 0:
        print(f"    Demand range: {df['demand_mw'].min():.0f} - {df['demand_mw'].max():.0f} MW")

    return df


def main():
    parser = argparse.ArgumentParser(description="Fetch raw EIA-930 hourly data for grid animation")
    parser.add_argument("--api-key", default=os.environ.get("EIA_API_KEY", ""),
                       help="EIA API key (or set EIA_API_KEY env var)")
    parser.add_argument("--scenario", default="all",
                       choices=["all"] + list(SCENARIO_DAYS.keys()),
                       help="Which scenario to fetch")
    args = parser.parse_args()

    api_key = args.api_key
    if not api_key:
        print("ERROR: No EIA API key. Set EIA_API_KEY env var or pass --api-key")
        print("  Get a free key at: https://www.eia.gov/opendata/register.php")
        sys.exit(1)

    os.makedirs(ERCOT_DIR, exist_ok=True)

    scenarios = SCENARIO_DAYS if args.scenario == "all" else {args.scenario: SCENARIO_DAYS[args.scenario]}

    for scenario_id, info in scenarios.items():
        begin, end = info["date_range"]
        gen_path = os.path.join(ERCOT_DIR, f"raw_generation_{scenario_id}.csv")
        demand_path = os.path.join(ERCOT_DIR, f"raw_demand_{scenario_id}.csv")

        print(f"\n{'='*60}")
        print(f"  {info['name']} ({begin} to {end})")
        print(f"{'='*60}")

        # Generation by fuel type
        if os.path.exists(gen_path):
            print(f"  Generation already cached: {gen_path}")
        else:
            gen_df = fetch_scenario_generation(scenario_id, begin, end, api_key)
            if len(gen_df) > 0:
                gen_df.to_csv(gen_path, index=False)
                print(f"  Saved generation to {gen_path}")

        # Demand
        if os.path.exists(demand_path):
            print(f"  Demand already cached: {demand_path}")
        else:
            demand_df = fetch_scenario_demand(scenario_id, begin, end, api_key)
            if len(demand_df) > 0:
                demand_df.to_csv(demand_path, index=False)
                print(f"  Saved demand to {demand_path}")

        time.sleep(1)  # Be polite between scenarios


if __name__ == "__main__":
    main()
