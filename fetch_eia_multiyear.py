#!/usr/bin/env python3
"""
fetch_eia_multiyear.py — Fetch 2021-2025 hourly EIA-930 data and produce
multi-year averaged generation, demand, and fossil-mix profiles.

Fetches hourly generation-by-fuel and demand from the EIA API for 5 ISOs
(CISO, ERCO, PJM, NYIS, ISNE), converts UTC → local prevailing time,
handles DST transitions, averages normalized profile shapes across years,
and scales to 2025 annual totals.

Usage:
    export EIA_API_KEY=your_key
    python fetch_eia_multiyear.py
    python fetch_eia_multiyear.py --api-key YOUR_KEY
    python fetch_eia_multiyear.py --dry-run

Outputs:
    data/eia_generation_profiles_multiyear.json
    data/eia_demand_profiles_multiyear.json
    data/eia_fossil_mix_multiyear.json

Requires: Python 3.9+ (zoneinfo), requests
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

try:
    import requests
except ImportError:
    print("ERROR: 'requests' package required. Install with: pip install requests")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

BASE_URL = "https://api.eia.gov/v2/electricity/rto/"
FUEL_TYPE_ENDPOINT = "fuel-type-data/data/"
DEMAND_ENDPOINT = "region-data/data/"

YEARS = [2021, 2022, 2023, 2024, 2025]
H = 8760  # hours in a non-leap year

# EIA balancing authority codes → our internal ISO names
BA_TO_ISO = {
    'CISO': 'CAISO',
    'ERCO': 'ERCOT',
    'PJM':  'PJM',
    'NYIS': 'NYISO',
    'ISNE': 'NEISO',
}

ISO_TO_BA = {v: k for k, v in BA_TO_ISO.items()}

# IANA timezone for each ISO
ISO_TIMEZONES = {
    'CAISO': 'America/Los_Angeles',
    'ERCOT': 'America/Chicago',
    'PJM':   'America/New_York',
    'NYISO': 'America/New_York',
    'NEISO': 'America/New_York',
}

# EIA fuel type codes → our categories
# SUN = solar, WND = wind, NUC = nuclear (maps to clean_firm),
# WAT = conventional hydro
# COL = coal, NG = natural gas, OIL = petroleum
# OTH = other (ignored for generation profiles)
CLEAN_FUEL_MAP = {
    'SUN': 'solar',
    'WND': 'wind',
    'NUC': 'nuclear',
    'WAT': 'hydro',
}

FOSSIL_FUEL_TYPES = ['COL', 'NG', 'OIL']

# Also fetch geothermal for CAISO
GEO_FUEL = 'GEO'  # geothermal — only relevant for CAISO

# 2025 generation shares from optimizer (% of total generation)
# Used to scale averaged profiles to 2025 annual totals
GRID_MIX_SHARES = {
    'CAISO': {'clean_firm': 7.9, 'solar': 22.3, 'wind': 8.8, 'ccs_ccgt': 0, 'hydro': 9.5},
    'ERCOT': {'clean_firm': 8.6, 'solar': 13.8, 'wind': 23.6, 'ccs_ccgt': 0, 'hydro': 0.1},
    'PJM':   {'clean_firm': 32.1, 'solar': 2.9, 'wind': 3.8, 'ccs_ccgt': 0, 'hydro': 1.8},
    'NYISO': {'clean_firm': 18.4, 'solar': 0.0, 'wind': 4.7, 'ccs_ccgt': 0, 'hydro': 15.9},
    'NEISO': {'clean_firm': 23.8, 'solar': 1.4, 'wind': 3.9, 'ccs_ccgt': 0, 'hydro': 4.4},
}

# Rate limiting
MAX_REQUESTS_PER_SECOND = 5
MIN_REQUEST_INTERVAL = 1.0 / MAX_REQUESTS_PER_SECOND  # 0.2 seconds

# Retry config
MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0  # seconds


# ══════════════════════════════════════════════════════════════════════════════
# RATE LIMITER
# ══════════════════════════════════════════════════════════════════════════════

class RateLimiter:
    """Simple rate limiter: max N requests per second."""

    def __init__(self, max_per_second=5):
        self.min_interval = 1.0 / max_per_second
        self.last_request_time = 0.0

    def wait(self):
        now = time.monotonic()
        elapsed = now - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.monotonic()


rate_limiter = RateLimiter(MAX_REQUESTS_PER_SECOND)


# ══════════════════════════════════════════════════════════════════════════════
# API FETCHING
# ══════════════════════════════════════════════════════════════════════════════

def fetch_api_page(url, params, api_key):
    """Fetch a single page from the EIA API with retry and backoff."""
    params = dict(params)
    params['api_key'] = api_key

    for attempt in range(MAX_RETRIES):
        rate_limiter.wait()
        try:
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                if 'response' in data and 'data' in data['response']:
                    return data['response']
                elif 'error' in data:
                    print(f"  API error: {data['error']}")
                    return None
                else:
                    print(f"  Unexpected response structure: {list(data.keys())}")
                    return None
            elif resp.status_code == 429:
                # Rate limited
                backoff = INITIAL_BACKOFF * (2 ** attempt)
                print(f"  Rate limited (429). Retrying in {backoff:.1f}s...")
                time.sleep(backoff)
                continue
            elif resp.status_code >= 500:
                backoff = INITIAL_BACKOFF * (2 ** attempt)
                print(f"  Server error ({resp.status_code}). Retrying in {backoff:.1f}s...")
                time.sleep(backoff)
                continue
            else:
                print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
                return None
        except requests.exceptions.Timeout:
            backoff = INITIAL_BACKOFF * (2 ** attempt)
            print(f"  Request timeout. Retrying in {backoff:.1f}s...")
            time.sleep(backoff)
            continue
        except requests.exceptions.ConnectionError as e:
            backoff = INITIAL_BACKOFF * (2 ** attempt)
            print(f"  Connection error: {e}. Retrying in {backoff:.1f}s...")
            time.sleep(backoff)
            continue
        except Exception as e:
            print(f"  Unexpected error: {e}")
            return None

    print(f"  FAILED after {MAX_RETRIES} retries")
    return None


def fetch_all_pages(url, params, api_key):
    """Fetch all pages of paginated EIA API results."""
    all_rows = []
    offset = 0
    page_size = 5000

    while True:
        page_params = dict(params)
        page_params['offset'] = offset
        page_params['length'] = page_size

        response = fetch_api_page(url, page_params, api_key)
        if response is None:
            print(f"  Failed to fetch page at offset {offset}")
            return all_rows  # Return what we have

        rows = response.get('data', [])
        total = response.get('total', 0)

        all_rows.extend(rows)

        if len(all_rows) >= total or len(rows) == 0:
            break

        offset += page_size

    return all_rows


def fetch_fuel_type_data(ba_code, year, api_key, dry_run=False):
    """
    Fetch hourly generation by fuel type for a BA and year.

    Returns list of dicts with keys: period, fueltype, value
    """
    url = BASE_URL + FUEL_TYPE_ENDPOINT

    start = f"{year}-01-01T00"
    end = f"{year}-12-31T23"

    params = {
        'frequency': 'hourly',
        'data[0]': 'value',
        'facets[respondent][]': ba_code,
        'start': start,
        'end': end,
        'sort[0][column]': 'period',
        'sort[0][direction]': 'asc',
    }

    if dry_run:
        print(f"  [DRY RUN] Would fetch: {url}")
        print(f"    BA={ba_code}, year={year}, start={start}, end={end}")
        print(f"    Estimated pages: ~{math.ceil(8760 * 8 / 5000)}")
        return []

    print(f"  Fetching fuel-type data for {ba_code} {year}...")
    rows = fetch_all_pages(url, params, api_key)
    print(f"    Got {len(rows)} rows")
    return rows


def fetch_demand_data(ba_code, year, api_key, dry_run=False):
    """
    Fetch hourly demand for a BA and year.

    Returns list of dicts with keys: period, value, type-name
    """
    url = BASE_URL + DEMAND_ENDPOINT

    start = f"{year}-01-01T00"
    end = f"{year}-12-31T23"

    params = {
        'frequency': 'hourly',
        'data[0]': 'value',
        'facets[respondent][]': ba_code,
        'facets[type][]': 'D',  # D = demand
        'start': start,
        'end': end,
        'sort[0][column]': 'period',
        'sort[0][direction]': 'asc',
    }

    if dry_run:
        print(f"  [DRY RUN] Would fetch: {url}")
        print(f"    BA={ba_code}, year={year}, type=demand, start={start}, end={end}")
        print(f"    Estimated pages: ~{math.ceil(8760 / 5000)}")
        return []

    print(f"  Fetching demand data for {ba_code} {year}...")
    rows = fetch_all_pages(url, params, api_key)
    print(f"    Got {len(rows)} rows")
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# TIMEZONE CONVERSION & DST HANDLING
# ══════════════════════════════════════════════════════════════════════════════

def parse_utc_period(period_str):
    """
    Parse EIA period string like '2024-01-15T08' into a UTC datetime.
    EIA timestamps are in UTC.
    """
    # Handle both formats: '2024-01-15T08' and '2024-01-15T08:00'
    period_str = period_str.strip()
    if len(period_str) == 13:  # '2024-01-15T08'
        dt = datetime(
            int(period_str[0:4]),
            int(period_str[5:7]),
            int(period_str[8:10]),
            int(period_str[11:13]),
            tzinfo=timezone.utc
        )
    else:
        # Try parsing with minutes
        try:
            dt = datetime.strptime(period_str[:16], "%Y-%m-%dT%H:%M")
            dt = dt.replace(tzinfo=timezone.utc)
        except ValueError:
            dt = datetime.strptime(period_str[:13], "%Y-%m-%dT%H")
            dt = dt.replace(tzinfo=timezone.utc)
    return dt


def utc_to_local_hour_of_year(utc_dt, tz):
    """
    Convert a UTC datetime to local prevailing time and return the
    hour-of-year index (0-8759) for the local year.

    Returns (hour_of_year, local_dt) or (None, local_dt) if the local
    time falls outside the target year.
    """
    local_dt = utc_dt.astimezone(tz)
    year = local_dt.year
    jan1 = datetime(year, 1, 1, 0, 0, 0, tzinfo=tz)
    delta = local_dt - jan1
    hour_of_year = int(delta.total_seconds() // 3600)
    return hour_of_year, local_dt


def build_local_hourly_array(utc_hour_values, tz_name, year):
    """
    Given a dict of {utc_datetime: value} for a single year, convert to
    local prevailing time and produce exactly 8760 hourly values.

    Handles DST:
    - Spring forward (missing hour): interpolate from neighbors
    - Fall back (duplicate hour): average the two values

    Parameters:
        utc_hour_values: dict mapping UTC datetime → float value
        tz_name: IANA timezone string
        year: the target year

    Returns:
        list of 8760 floats indexed by local hour-of-year (0=Jan1 00:00 local)
    """
    tz = ZoneInfo(tz_name)

    # Collect values by local hour-of-year
    # Use a defaultdict(list) to handle duplicate hours (fall back)
    hour_buckets = defaultdict(list)

    for utc_dt, value in utc_hour_values.items():
        local_dt = utc_dt.astimezone(tz)
        local_year = local_dt.year

        # Only include hours that map to the target year in local time
        if local_year != year:
            continue

        jan1 = datetime(year, 1, 1, 0, 0, 0, tzinfo=tz)
        delta = local_dt - jan1
        hoy = int(delta.total_seconds() // 3600)

        if 0 <= hoy < H:
            hour_buckets[hoy].append(value)

    # Build the 8760-element array
    result = [0.0] * H

    for h in range(H):
        if h in hour_buckets and len(hour_buckets[h]) > 0:
            # Average if multiple values (fall-back DST hour)
            vals = hour_buckets[h]
            result[h] = sum(vals) / len(vals)
        else:
            # Missing hour (spring-forward DST gap) — mark for interpolation
            result[h] = None

    # Interpolate missing hours (spring-forward gaps)
    for h in range(H):
        if result[h] is None:
            # Find nearest non-None neighbors
            prev_val = None
            next_val = None
            for offset in range(1, 24):
                if prev_val is None and h - offset >= 0 and result[h - offset] is not None:
                    prev_val = result[h - offset]
                if next_val is None and h + offset < H and result[h + offset] is not None:
                    next_val = result[h + offset]
                if prev_val is not None and next_val is not None:
                    break

            if prev_val is not None and next_val is not None:
                result[h] = (prev_val + next_val) / 2.0
            elif prev_val is not None:
                result[h] = prev_val
            elif next_val is not None:
                result[h] = next_val
            else:
                result[h] = 0.0

    return result


# ══════════════════════════════════════════════════════════════════════════════
# DATA PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_fuel_rows(rows):
    """
    Parse raw EIA fuel-type rows into a structured dict:
    {fuel_code: {utc_datetime: mwh_value}}
    """
    fuel_data = defaultdict(dict)

    for row in rows:
        period = row.get('period', '')
        fuel_code = row.get('fueltype', row.get('type-name', row.get('fueltypeid', '')))
        value = row.get('value')

        if not period or not fuel_code:
            continue

        # Handle null/missing values
        if value is None or value == '' or value == 'null':
            value = 0.0
        else:
            try:
                value = float(value)
            except (ValueError, TypeError):
                value = 0.0

        utc_dt = parse_utc_period(period)
        fuel_data[fuel_code][utc_dt] = value

    return fuel_data


def parse_demand_rows(rows):
    """
    Parse raw EIA demand rows into {utc_datetime: mwh_value}.
    """
    demand_data = {}

    for row in rows:
        period = row.get('period', '')
        value = row.get('value')

        if not period:
            continue

        if value is None or value == '' or value == 'null':
            value = 0.0
        else:
            try:
                value = float(value)
            except (ValueError, TypeError):
                value = 0.0

        utc_dt = parse_utc_period(period)
        demand_data[utc_dt] = value

    return demand_data


def normalize_profile(hourly_values):
    """
    Normalize an 8760-element profile so its values sum to 1.0.
    Returns a new list. If total is zero, returns uniform distribution.
    """
    total = sum(hourly_values)
    if total <= 0:
        return [1.0 / H] * H
    return [v / total for v in hourly_values]


def average_normalized_profiles(profiles_by_year):
    """
    Given a dict {year: [8760 values]}, normalize each year's profile,
    average hour-by-hour across years, then re-normalize.

    Returns a single 8760-element normalized profile.
    """
    if not profiles_by_year:
        return [1.0 / H] * H

    # Normalize each year
    normalized = {}
    for year, profile in profiles_by_year.items():
        normalized[year] = normalize_profile(profile)

    # Average hour-by-hour
    n_years = len(normalized)
    averaged = [0.0] * H

    for h in range(H):
        total = sum(normalized[yr][h] for yr in normalized)
        averaged[h] = total / n_years

    # Re-normalize
    return normalize_profile(averaged)


def compute_fossil_shares_hourly(coal_hourly, gas_hourly, oil_hourly):
    """
    Compute hourly fossil fuel shares (coal/gas/oil as fraction of total fossil).
    Returns three 8760-element lists.
    """
    coal_shares = []
    gas_shares = []
    oil_shares = []

    for h in range(H):
        coal = max(0.0, coal_hourly[h])
        gas = max(0.0, gas_hourly[h])
        oil = max(0.0, oil_hourly[h])
        total = coal + gas + oil

        if total > 0:
            coal_shares.append(round(coal / total, 6))
            gas_shares.append(round(gas / total, 6))
            oil_shares.append(round(oil / total, 6))
        else:
            # Default to 100% gas when no fossil generation
            coal_shares.append(0.0)
            gas_shares.append(1.0)
            oil_shares.append(0.0)

    return coal_shares, gas_shares, oil_shares


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def load_existing_demand_profiles():
    """Load existing 2025 demand profiles for annual totals."""
    path = os.path.join(DATA_DIR, 'eia_demand_profiles.json')
    if not os.path.exists(path):
        print(f"WARNING: {path} not found. Cannot scale to 2025 totals.")
        return None
    with open(path) as f:
        return json.load(f)


def fetch_and_process_iso(iso, api_key, years, dry_run=False):
    """
    Fetch all years of data for a single ISO, convert to local time,
    and return structured hourly data.

    Parameters:
        iso: ISO name (e.g., 'CAISO')
        api_key: EIA API key
        years: list of years to fetch
        dry_run: if True, show what would be fetched without API calls

    Returns:
        {
            'generation': {fuel_type: {year: [8760 values]}},
            'demand': {year: [8760 values]},
            'fossil': {fuel_code: {year: [8760 values]}}
        }
    """
    ba_code = ISO_TO_BA[iso]
    tz_name = ISO_TIMEZONES[iso]

    gen_by_fuel_year = defaultdict(dict)   # {fuel_category: {year: [8760]}}
    demand_by_year = {}                      # {year: [8760]}
    fossil_by_fuel_year = defaultdict(dict)  # {fuel_code: {year: [8760]}}

    for year in years:
        print(f"\n  === {iso} ({ba_code}) — {year} ===")

        # Fetch fuel type generation
        fuel_rows = fetch_fuel_type_data(ba_code, year, api_key, dry_run)

        if not dry_run and fuel_rows:
            fuel_data = parse_fuel_rows(fuel_rows)

            # Clean generation fuels
            for eia_code, our_name in CLEAN_FUEL_MAP.items():
                if eia_code in fuel_data:
                    hourly = build_local_hourly_array(fuel_data[eia_code], tz_name, year)
                    gen_by_fuel_year[our_name][year] = hourly
                    n_nonzero = sum(1 for v in hourly if v > 0)
                    total = sum(hourly)
                    print(f"    {our_name} ({eia_code}): {n_nonzero} non-zero hours, "
                          f"total={total:,.0f} MWh")
                else:
                    print(f"    {our_name} ({eia_code}): no data, filling zeros")
                    gen_by_fuel_year[our_name][year] = [0.0] * H

            # Geothermal for CAISO
            if iso == 'CAISO' and GEO_FUEL in fuel_data:
                hourly = build_local_hourly_array(fuel_data[GEO_FUEL], tz_name, year)
                gen_by_fuel_year['geothermal'][year] = hourly
                total = sum(hourly)
                print(f"    geothermal (GEO): total={total:,.0f} MWh")

            # Fossil fuels for mix shares
            for fossil_code in FOSSIL_FUEL_TYPES:
                if fossil_code in fuel_data:
                    hourly = build_local_hourly_array(fuel_data[fossil_code], tz_name, year)
                    fossil_by_fuel_year[fossil_code][year] = hourly
                else:
                    fossil_by_fuel_year[fossil_code][year] = [0.0] * H

        # Fetch demand
        demand_rows = fetch_demand_data(ba_code, year, api_key, dry_run)

        if not dry_run and demand_rows:
            demand_utc = parse_demand_rows(demand_rows)
            hourly = build_local_hourly_array(demand_utc, tz_name, year)
            demand_by_year[year] = hourly
            total = sum(hourly)
            peak = max(hourly)
            print(f"    demand: total={total:,.0f} MWh, peak={peak:,.0f} MW")

    return {
        'generation': dict(gen_by_fuel_year),
        'demand': demand_by_year,
        'fossil': dict(fossil_by_fuel_year),
    }


def build_multiyear_generation_profiles(all_iso_data, existing_demand):
    """
    Build multi-year averaged generation profiles for all ISOs.

    Returns dict matching existing eia_generation_profiles.json structure:
    {
        'CAISO': {
            'multiyear_avg': {
                'solar': [8760 values],
                'wind': [8760 values],
                'nuclear': [8760 values],
                'hydro': [8760 values],
                ...
            }
        },
        ...
    }
    """
    result = {}

    for iso in BA_TO_ISO.values():
        iso_data = all_iso_data.get(iso)
        if not iso_data:
            print(f"  WARNING: No data for {iso}, skipping")
            continue

        gen_data = iso_data['generation']
        profiles = {}

        # Determine fuel types for this ISO
        fuel_types = ['solar', 'wind', 'nuclear', 'hydro']
        if iso == 'CAISO':
            fuel_types.append('geothermal')

        for fuel in fuel_types:
            if fuel in gen_data and gen_data[fuel]:
                avg_profile = average_normalized_profiles(gen_data[fuel])
                profiles[fuel] = avg_profile

                years_used = sorted(gen_data[fuel].keys())
                print(f"  {iso}/{fuel}: averaged {len(years_used)} years "
                      f"({min(years_used)}-{max(years_used)})")
            else:
                print(f"  {iso}/{fuel}: no data, using uniform profile")
                profiles[fuel] = [1.0 / H] * H

        # NYISO solar proxy from NEISO solar
        if iso == 'NYISO':
            neiso_data = all_iso_data.get('NEISO')
            if neiso_data and 'solar' in neiso_data['generation']:
                proxy_profile = average_normalized_profiles(
                    neiso_data['generation']['solar']
                )
                profiles['solar_proxy'] = proxy_profile
                print(f"  {iso}/solar_proxy: using NEISO solar average")
            else:
                profiles['solar_proxy'] = profiles.get('solar', [1.0 / H] * H)
                print(f"  {iso}/solar_proxy: NEISO solar unavailable, using own solar")

        # Scale profiles to 2025 totals
        # The profiles are normalized (sum=1.0). For the output file,
        # we store them as normalized since the optimizer re-scales anyway.
        result[iso] = {'multiyear_avg': profiles}

    return result


def build_multiyear_demand_profiles(all_iso_data, existing_demand):
    """
    Build multi-year averaged demand profiles.

    Returns dict matching eia_demand_profiles.json structure:
    {
        'CAISO': {
            'raw_mwh': [8760 values scaled to 2025 total],
            'normalized': [8760 normalized values],
            'total_annual_mwh': ...,
            'peak_mw': ...,
            'min_mw': ...,
            'avg_mw': ...,
        },
        ...
    }
    """
    result = {}

    for iso in BA_TO_ISO.values():
        iso_data = all_iso_data.get(iso)
        if not iso_data:
            print(f"  WARNING: No data for {iso}, skipping")
            continue

        demand_data = iso_data['demand']

        if not demand_data:
            print(f"  WARNING: No demand data for {iso}")
            continue

        # Average normalized profiles across years
        avg_normalized = average_normalized_profiles(demand_data)

        # Get 2025 annual totals from existing data
        if existing_demand and iso in existing_demand:
            total_annual = existing_demand[iso].get('total_annual_mwh', 0)
            existing_peak = existing_demand[iso].get('peak_mw', 0)
        else:
            # Fall back to averaging the yearly totals
            yearly_totals = [sum(demand_data[yr]) for yr in demand_data]
            total_annual = sum(yearly_totals) / len(yearly_totals)
            existing_peak = 0

        # Scale averaged shape to 2025 total
        raw_mwh = [v * total_annual for v in avg_normalized]

        # Compute stats from the shaped profile
        peak_mw = max(raw_mwh)
        min_mw = min(raw_mwh)
        avg_mw = total_annual / H

        # Use existing 2025 peak if it's higher (averaged profiles smooth peaks)
        if existing_peak > peak_mw:
            # Scale up the peak hour to match, keeping shape
            peak_mw = existing_peak

        result[iso] = {
            'raw_mwh': [round(v, 1) for v in raw_mwh],
            'normalized': avg_normalized,
            'total_annual_mwh': round(total_annual),
            'peak_mw': round(peak_mw),
            'min_mw': round(min_mw),
            'avg_mw': round(avg_mw),
        }

        years_used = sorted(demand_data.keys())
        print(f"  {iso}: averaged {len(years_used)} years, "
              f"total={total_annual:,.0f} MWh, peak={peak_mw:,.0f} MW")

    return result


def build_multiyear_fossil_mix(all_iso_data):
    """
    Build multi-year averaged fossil fuel mix shares.

    For each hour, computes coal/gas/oil share of total fossil generation,
    averaged across years.

    Returns dict matching eia_fossil_mix.json structure:
    {
        'CAISO': {
            'multiyear_avg': {
                'hours': [...],
                'coal_share': [8760],
                'gas_share': [8760],
                'oil_share': [8760],
            }
        },
        ...
    }
    """
    result = {}

    for iso in BA_TO_ISO.values():
        iso_data = all_iso_data.get(iso)
        if not iso_data:
            continue

        fossil_data = iso_data['fossil']

        # For each year, compute hourly shares, then average
        coal_shares_by_year = {}
        gas_shares_by_year = {}
        oil_shares_by_year = {}

        # Determine which years have data
        all_years = set()
        for fuel_code in FOSSIL_FUEL_TYPES:
            all_years.update(fossil_data.get(fuel_code, {}).keys())
        if not all_years:
            continue

        for year in sorted(all_years):
            coal = fossil_data.get('COL', {}).get(year, [0.0] * H)
            gas = fossil_data.get('NG', {}).get(year, [0.0] * H)
            oil = fossil_data.get('OIL', {}).get(year, [0.0] * H)

            c_shares, g_shares, o_shares = compute_fossil_shares_hourly(coal, gas, oil)
            coal_shares_by_year[year] = c_shares
            gas_shares_by_year[year] = g_shares
            oil_shares_by_year[year] = o_shares

        # Average shares across years
        n_years = len(coal_shares_by_year)
        if n_years == 0:
            continue

        avg_coal = [0.0] * H
        avg_gas = [0.0] * H
        avg_oil = [0.0] * H

        for h in range(H):
            for year in coal_shares_by_year:
                avg_coal[h] += coal_shares_by_year[year][h]
                avg_gas[h] += gas_shares_by_year[year][h]
                avg_oil[h] += oil_shares_by_year[year][h]

            avg_coal[h] /= n_years
            avg_gas[h] /= n_years
            avg_oil[h] /= n_years

            # Renormalize to ensure shares sum to 1.0
            total = avg_coal[h] + avg_gas[h] + avg_oil[h]
            if total > 0:
                avg_coal[h] = round(avg_coal[h] / total, 6)
                avg_gas[h] = round(avg_gas[h] / total, 6)
                avg_oil[h] = round(avg_oil[h] / total, 6)
            else:
                avg_coal[h] = 0.0
                avg_gas[h] = 1.0
                avg_oil[h] = 0.0

        # Generate hour labels (local time, using 2025 as reference year)
        tz_name = ISO_TIMEZONES[iso]
        tz = ZoneInfo(tz_name)
        hours = []
        jan1 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=tz)
        for h in range(H):
            dt = jan1 + timedelta(hours=h)
            hours.append(dt.strftime('%Y-%m-%dT%H'))

        result[iso] = {
            'multiyear_avg': {
                'hours': hours,
                'coal_share': avg_coal,
                'gas_share': avg_gas,
                'oil_share': avg_oil,
            }
        }

        print(f"  {iso}: avg coal={sum(avg_coal)/H:.3f}, "
              f"gas={sum(avg_gas)/H:.3f}, oil={sum(avg_oil)/H:.3f}")

    return result


def save_json(data, filename):
    """Save data to a JSON file in the data directory."""
    path = os.path.join(DATA_DIR, filename)
    with open(path, 'w') as f:
        json.dump(data, f, indent=None, separators=(',', ':'))
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Saved {path} ({size_mb:.1f} MB)")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Fetch 2021-2025 hourly EIA-930 data and produce "
                    "multi-year averaged profiles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python fetch_eia_multiyear.py --api-key YOUR_KEY
    EIA_API_KEY=xxx python fetch_eia_multiyear.py
    python fetch_eia_multiyear.py --dry-run
    python fetch_eia_multiyear.py --isos CAISO ERCOT
        """
    )
    parser.add_argument(
        '--api-key',
        default=os.environ.get('EIA_API_KEY', ''),
        help='EIA API key (or set EIA_API_KEY env var)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be fetched without making API calls'
    )
    parser.add_argument(
        '--isos',
        nargs='+',
        choices=['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO'],
        default=['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO'],
        help='ISOs to fetch (default: all 5)'
    )
    parser.add_argument(
        '--years',
        nargs='+',
        type=int,
        default=YEARS,
        help=f'Years to fetch (default: {YEARS})'
    )
    parser.add_argument(
        '--output-dir',
        default=DATA_DIR,
        help=f'Output directory (default: {DATA_DIR})'
    )

    args = parser.parse_args()

    # Validate API key
    if not args.dry_run and not args.api_key:
        print("ERROR: EIA API key required. Set EIA_API_KEY env var or use --api-key")
        print("  Get a free key at: https://www.eia.gov/opendata/register.php")
        sys.exit(1)

    years_to_fetch = args.years

    dry_run = args.dry_run
    api_key = args.api_key

    print("=" * 70)
    print("EIA Multi-Year Hourly Data Fetcher")
    print("=" * 70)
    print(f"  ISOs:  {', '.join(args.isos)}")
    print(f"  Years: {', '.join(str(y) for y in years_to_fetch)}")
    print(f"  Mode:  {'DRY RUN' if dry_run else 'LIVE'}")
    if not dry_run:
        print(f"  API key: {api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "")
    print(f"  Output: {args.output_dir}")
    print("=" * 70)

    # Estimate total API calls
    n_isos = len(args.isos)
    n_years = len(years_to_fetch)
    # Fuel type: ~8760*8 rows/year ≈ 15 pages; Demand: ~8760 rows ≈ 2 pages
    est_fuel_pages = math.ceil(8760 * 8 / 5000)
    est_demand_pages = math.ceil(8760 / 5000)
    est_total_pages = n_isos * n_years * (est_fuel_pages + est_demand_pages)
    est_time_min = est_total_pages * MIN_REQUEST_INTERVAL / 60

    print(f"\n  Estimated API calls: ~{est_total_pages}")
    print(f"  Estimated time:     ~{est_time_min:.0f} min (rate-limited)")

    if dry_run:
        print("\n--- DRY RUN: No API calls will be made ---\n")

    # Load existing demand profiles for 2025 annual totals
    existing_demand = load_existing_demand_profiles()
    if existing_demand:
        print("\n  Loaded existing demand profiles for 2025 totals")
        for iso in args.isos:
            if iso in existing_demand:
                total = existing_demand[iso].get('total_annual_mwh', 0)
                peak = existing_demand[iso].get('peak_mw', 0)
                print(f"    {iso}: {total:,.0f} MWh, peak {peak:,.0f} MW")

    # ── Phase 1: Fetch all data ──────────────────────────────────────────

    print("\n" + "=" * 70)
    print("PHASE 1: Fetching EIA-930 hourly data")
    print("=" * 70)

    all_iso_data = {}
    start_time = time.time()

    # If NYISO is requested, ensure NEISO is also fetched (for solar proxy)
    fetch_isos = list(args.isos)
    if 'NYISO' in fetch_isos and 'NEISO' not in fetch_isos:
        fetch_isos.append('NEISO')
        print("  (Adding NEISO to fetch list for NYISO solar proxy)")

    for iso in fetch_isos:
        print(f"\n{'─' * 50}")
        print(f"  Fetching {iso}")
        print(f"{'─' * 50}")

        iso_result = fetch_and_process_iso(iso, api_key, years_to_fetch, dry_run)
        all_iso_data[iso] = iso_result

    elapsed = time.time() - start_time
    print(f"\n  Phase 1 complete in {elapsed:.0f}s")

    if dry_run:
        print("\n[DRY RUN] Exiting. No data was fetched or files written.")
        return

    # ── Phase 2: Build averaged profiles ─────────────────────────────────

    print("\n" + "=" * 70)
    print("PHASE 2: Building multi-year averaged profiles")
    print("=" * 70)

    # Generation profiles
    print("\n  --- Generation Profiles ---")
    gen_profiles = build_multiyear_generation_profiles(all_iso_data, existing_demand)

    # Demand profiles
    print("\n  --- Demand Profiles ---")
    demand_profiles = build_multiyear_demand_profiles(all_iso_data, existing_demand)

    # Fossil mix
    print("\n  --- Fossil Fuel Mix ---")
    fossil_mix = build_multiyear_fossil_mix(all_iso_data)

    # ── Phase 3: Save outputs ────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("PHASE 3: Saving output files")
    print("=" * 70)

    save_json(gen_profiles, 'eia_generation_profiles_multiyear.json')
    save_json(demand_profiles, 'eia_demand_profiles_multiyear.json')
    save_json(fossil_mix, 'eia_fossil_mix_multiyear.json')

    # ── Summary ──────────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    total_elapsed = time.time() - start_time
    print(f"  Total time: {total_elapsed:.0f}s")
    print(f"\n  Output files:")
    print(f"    data/eia_generation_profiles_multiyear.json")
    print(f"    data/eia_demand_profiles_multiyear.json")
    print(f"    data/eia_fossil_mix_multiyear.json")
    print(f"\n  Years averaged: {years_to_fetch}")
    print(f"  ISOs processed: {list(all_iso_data.keys())}")

    # Validation summary
    print(f"\n  --- Validation ---")
    for iso in args.isos:
        if iso in gen_profiles:
            fuels = list(gen_profiles[iso].get('multiyear_avg', {}).keys())
            print(f"  {iso}: {len(fuels)} fuel profiles ({', '.join(fuels)})")
            for fuel in fuels:
                profile = gen_profiles[iso]['multiyear_avg'][fuel]
                total = sum(profile)
                print(f"    {fuel}: sum={total:.6f} (expect ~1.0), "
                      f"max={max(profile):.6f}, min={min(profile):.8f}")

        if iso in demand_profiles:
            d = demand_profiles[iso]
            norm_sum = sum(d['normalized'])
            print(f"  {iso} demand: norm_sum={norm_sum:.6f}, "
                  f"total={d['total_annual_mwh']:,.0f} MWh")

    print()


if __name__ == '__main__':
    main()
