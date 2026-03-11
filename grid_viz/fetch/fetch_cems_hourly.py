#!/usr/bin/env python3
"""
Fetch hourly CEMS (Continuous Emissions Monitoring) data from EPA CAMD API.

CAMD API: https://api.epa.gov/easey
- Public, no API key required
- Returns hourly emissions + generation for fossil fuel plants
- Covers: CO2, NOx, SO2, heat input, gross/steam load
- Available historically back to 1995

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


def fetch_all_scenarios(state: str = "TX") -> dict[str, pd.DataFrame]:
    """Fetch CEMS data for all configured scenario days."""

    results = {}
    for scenario_id, info in SCENARIO_DAYS.items():
        begin, end = info["date_range"]
        output_path = os.path.join(CEMS_DIR, f"cems_{scenario_id}.csv")

        if os.path.exists(output_path):
            print(f"  [{scenario_id}] Already cached at {output_path}")
            results[scenario_id] = pd.read_csv(output_path)
            continue

        df = fetch_cems_hourly(
            state=state,
            begin_date=begin,
            end_date=end,
            output_path=output_path,
        )
        results[scenario_id] = df

        # Be polite to the API between large requests
        time.sleep(2)

    return results


if __name__ == "__main__":
    os.makedirs(CEMS_DIR, exist_ok=True)
    results = fetch_all_scenarios()

    print("\n=== Fetch Summary ===")
    for scenario_id, df in results.items():
        if len(df) > 0:
            print(f"  {scenario_id}: {len(df)} records, {df['facilityId'].nunique() if 'facilityId' in df.columns else '?'} facilities")
        else:
            print(f"  {scenario_id}: FAILED or empty")
