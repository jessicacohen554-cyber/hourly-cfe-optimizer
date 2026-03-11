#!/usr/bin/env python3
"""
Fetch hourly CEMS (Continuous Emissions Monitoring) data from EPA CAMD API
or Catalyst Cooperative's PUDL bulk Parquet files.

Data sources (in priority order):
1. PUDL Parquet bulk files (fastest, no API key needed)
   - Pre-packaged by Catalyst Cooperative from EPA CEMS data
   - Available on AWS Open Data / Zenodo
   - Partitioned by year and state
2. CAMD Streaming API (api.epa.gov/easey)
   - Requires free api.data.gov API key
   - Returns CSV for large queries
   - Good for targeted date ranges

This fetches plant-level hourly generation and emissions for ERCOT fossil plants
on specific scenario days (Uri, summer heat, normal spring).
"""

import os
import sys
import time
import json
import requests
import pandas as pd
from datetime import datetime, timedelta

# Add parent for config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import CAMD_API_BASE, SCENARIO_DAYS, CEMS_DIR


# CAMD Streaming API endpoints
# The streaming/apportioned/hourly endpoint allows larger result sets
HOURLY_URL = f"{CAMD_API_BASE}/streaming-services/emissions/apportioned/hourly"

# Fallback: standard paginated endpoint
HOURLY_PAGINATED_URL = f"{CAMD_API_BASE}/emissions/apportioned/hourly"


def fetch_cems_hourly(
    state: str = "TX",
    begin_date: str = "2021-02-10",
    end_date: str = "2021-02-20",
    output_path: str | None = None,
    max_retries: int = 4,
) -> pd.DataFrame:
    """
    Fetch hourly CEMS data for all plants in a state for a date range.

    Uses the CAMD streaming API which returns CSV for large result sets.
    Falls back to paginated JSON endpoint if streaming fails.

    Parameters
    ----------
    state : str
        State abbreviation (e.g., "TX")
    begin_date : str
        Start date YYYY-MM-DD
    end_date : str
        End date YYYY-MM-DD
    output_path : str, optional
        Path to save CSV output
    max_retries : int
        Number of retry attempts with exponential backoff

    Returns
    -------
    pd.DataFrame
        Hourly CEMS data with columns:
        - facilityId (ORIS code)
        - facilityName
        - unitId
        - date, hour
        - grossLoad (MW)
        - steamLoad (1000 lb/hr)
        - so2Mass (lbs), co2Mass (tons), noxMass (lbs)
        - heatInput (MMBtu)
        - state, latitude, longitude
    """
    params = {
        "stateCode": state,
        "beginDate": begin_date,
        "endDate": end_date,
        "operatingHoursOnly": "true",
    }

    headers = {
        "Accept": "text/csv",  # Request CSV for streaming
    }

    print(f"Fetching CAMD CEMS hourly data: {state}, {begin_date} to {end_date}...")

    # Try streaming endpoint first (handles large result sets)
    for attempt in range(max_retries):
        try:
            resp = requests.get(
                HOURLY_URL,
                params=params,
                headers=headers,
                timeout=300,
                stream=True,
            )

            if resp.status_code == 200:
                # Stream CSV to file, then read into DataFrame
                temp_path = output_path or f"/tmp/cems_{state}_{begin_date}_{end_date}.csv"
                os.makedirs(os.path.dirname(temp_path) if os.path.dirname(temp_path) else ".", exist_ok=True)

                total_bytes = 0
                with open(temp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                        total_bytes += len(chunk)

                print(f"  Downloaded {total_bytes / 1024:.0f} KB")

                df = pd.read_csv(temp_path)
                print(f"  Parsed {len(df)} hourly records for {df['Facility ID'].nunique() if 'Facility ID' in df.columns else '?'} facilities")

                # Standardize column names
                col_map = {
                    "Facility ID": "facilityId",
                    "Facility Name": "facilityName",
                    "Unit ID": "unitId",
                    "Date": "date",
                    "Hour": "hour",
                    "Gross Load (MW)": "grossLoad",
                    "Steam Load (1000 lb/hr)": "steamLoad",
                    "SO2 Mass (lbs)": "so2Mass",
                    "CO2 Mass (short tons)": "co2Mass",
                    "NOx Mass (lbs)": "noxMass",
                    "Heat Input (mmBtu)": "heatInput",
                    "State": "state",
                    "Latitude": "latitude",
                    "Longitude": "longitude",
                    "Primary Fuel Type": "primaryFuel",
                    "Secondary Fuel Type": "secondaryFuel",
                    "Unit Type": "unitType",
                    "SO2 Controls": "so2Controls",
                    "NOx Controls": "noxControls",
                    "PM Controls": "pmControls",
                    "Hg Controls": "hgControls",
                    "Program Code": "programCode",
                    "EPA Region": "epaRegion",
                    "NERC Region": "nercRegion",
                    "County": "county",
                    "Operating Time": "operatingTime",
                }
                df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

                # Save final output
                if output_path:
                    df.to_csv(output_path, index=False)
                    print(f"  Saved to {output_path}")

                return df

            elif resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                print(f"  Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            else:
                print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"  Retrying in {wait}s...")
                    time.sleep(wait)

        except requests.exceptions.Timeout:
            wait = 2 ** (attempt + 1)
            print(f"  Timeout. Retrying in {wait}s...")
            time.sleep(wait)
        except requests.exceptions.ConnectionError as e:
            wait = 2 ** (attempt + 1)
            print(f"  Connection error: {e}. Retrying in {wait}s...")
            time.sleep(wait)

    print(f"  FAILED after {max_retries} attempts")
    return pd.DataFrame()


def fetch_pudl_cems(
    year: int,
    state: str = "TX",
    output_dir: str | None = None,
) -> pd.DataFrame:
    """
    Download CEMS data from PUDL (Catalyst Cooperative) bulk Parquet files.

    PUDL repackages all EPA CEMS data as clean Parquet. Much faster than the API
    for full-year or multi-month queries. No API key needed.

    The PUDL dataset is hosted on AWS Open Data and Zenodo.
    """
    # PUDL S3 bucket (public, no auth needed)
    pudl_url = f"https://s3.us-west-2.amazonaws.com/pudl.catalyst.coop/v2024.11.0/out/epacems/year={year}/state={state.lower()}/part0.parquet"

    # Alternative: direct from nightly builds
    pudl_nightly = f"https://s3.us-west-2.amazonaws.com/pudl.catalyst.coop/nightly/epacems/year={year}/state={state.lower()}/part0.parquet"

    print(f"  Downloading PUDL CEMS data: {state} {year}...")

    for url in [pudl_url, pudl_nightly]:
        try:
            df = pd.read_parquet(url)
            print(f"  Downloaded {len(df)} records from PUDL")

            # PUDL columns: plant_id_eia, datetime_utc, gross_load_mw, co2_mass_tons,
            # nox_mass_lbs, so2_mass_lbs, heat_content_mmbtu, operating_time_hours, etc.
            col_map = {
                "plant_id_eia": "facilityId",
                "emissions_unit_id_epa": "unitId",
                "datetime_utc": "datetime_utc",
                "gross_load_mw": "grossLoad",
                "steam_load_1000_lbs": "steamLoad",
                "so2_mass_lbs": "so2Mass",
                "co2_mass_tons": "co2Mass",
                "nox_mass_lbs": "noxMass",
                "heat_content_mmbtu": "heatInput",
                "operating_time_hours": "operatingTime",
                "state": "state",
                "facility_name": "facilityName",
                "primary_fuel": "primaryFuel",
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

            # Extract date/hour from datetime
            if "datetime_utc" in df.columns:
                df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
                df["date"] = df["datetime_utc"].dt.strftime("%Y-%m-%d")
                df["hour"] = df["datetime_utc"].dt.hour

            if output_dir:
                out_path = os.path.join(output_dir, f"cems_pudl_{state}_{year}.parquet")
                os.makedirs(output_dir, exist_ok=True)
                df.to_parquet(out_path, index=False)
                print(f"  Saved to {out_path}")

            return df

        except Exception as e:
            print(f"  Failed from {url}: {e}")
            continue

    print(f"  PUDL download failed for {state} {year}")
    return pd.DataFrame()


def extract_scenario_from_pudl(
    pudl_df: pd.DataFrame,
    scenario_id: str,
    scenario_info: dict,
    ercot_oris_codes: set | None = None,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Extract scenario date range from a full-year PUDL CEMS DataFrame."""
    begin_str, end_str = scenario_info["date_range"]

    # Filter to date range
    mask = (pudl_df["date"] >= begin_str) & (pudl_df["date"] <= end_str)
    df = pudl_df[mask].copy()

    # Filter to ERCOT plants if we have the ORIS code set
    if ercot_oris_codes and "facilityId" in df.columns:
        df = df[df["facilityId"].isin(ercot_oris_codes)]

    print(f"  [{scenario_id}] Extracted {len(df)} records for {begin_str} to {end_str}")

    if output_path and len(df) > 0:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"  Saved to {output_path}")

    return df


def load_ercot_oris_codes() -> set:
    """Load ERCOT plant ORIS codes from eGRID extract."""
    path = "grid_viz/data/egrid/ercot_plants.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        codes = set(df["ORISPL"].astype(int).values)
        print(f"  Loaded {len(codes)} ERCOT plant ORIS codes")
        return codes
    return set()


def fetch_all_scenarios(state: str = "TX", use_pudl: bool = True) -> dict[str, pd.DataFrame]:
    """Fetch CEMS data for all configured scenario days.

    Tries PUDL bulk download first (faster, no API key), falls back to CAMD API.
    """
    results = {}

    # Group scenarios by year for efficient PUDL downloads
    scenarios_by_year = {}
    for scenario_id, info in SCENARIO_DAYS.items():
        begin_str = info["date_range"][0]
        year = int(begin_str[:4])
        if year not in scenarios_by_year:
            scenarios_by_year[year] = []
        scenarios_by_year[year].append((scenario_id, info))

    ercot_oris = load_ercot_oris_codes()

    for year, scenarios in scenarios_by_year.items():
        pudl_df = None

        # Check if all scenarios for this year are already cached
        all_cached = all(
            os.path.exists(os.path.join(CEMS_DIR, f"cems_{sid}.csv"))
            for sid, _ in scenarios
        )
        if all_cached:
            for sid, _ in scenarios:
                path = os.path.join(CEMS_DIR, f"cems_{sid}.csv")
                print(f"  [{sid}] Already cached at {path}")
                results[sid] = pd.read_csv(path)
            continue

        # Try PUDL first
        if use_pudl:
            pudl_df = fetch_pudl_cems(year, state, output_dir=CEMS_DIR)

        for scenario_id, info in scenarios:
            output_path = os.path.join(CEMS_DIR, f"cems_{scenario_id}.csv")

            if os.path.exists(output_path):
                print(f"  [{scenario_id}] Already cached at {output_path}")
                results[scenario_id] = pd.read_csv(output_path)
                continue

            if pudl_df is not None and len(pudl_df) > 0:
                df = extract_scenario_from_pudl(pudl_df, scenario_id, info, ercot_oris, output_path)
                results[scenario_id] = df
            else:
                # Fall back to CAMD API
                begin, end = info["date_range"]
                df = fetch_cems_hourly(state=state, begin_date=begin, end_date=end, output_path=output_path)
                results[scenario_id] = df
                time.sleep(2)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch CEMS hourly data for grid animation")
    parser.add_argument("--no-pudl", action="store_true", help="Skip PUDL, use CAMD API only")
    parser.add_argument("--api-key", default=os.environ.get("CAMD_API_KEY", ""),
                       help="CAMD/data.gov API key (for API mode)")
    args = parser.parse_args()

    os.makedirs(CEMS_DIR, exist_ok=True)
    results = fetch_all_scenarios(use_pudl=not args.no_pudl)

    print("\n=== Fetch Summary ===")
    for scenario_id, df in results.items():
        if len(df) > 0:
            fac_col = "facilityId" if "facilityId" in df.columns else None
            n_fac = df[fac_col].nunique() if fac_col else "?"
            print(f"  {scenario_id}: {len(df)} records, {n_fac} facilities")
        else:
            print(f"  {scenario_id}: FAILED or empty")
