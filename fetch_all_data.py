#!/usr/bin/env python3
"""
Fetch hourly generation mix and demand data from EIA API for 7 ISOs, 2021-2025.
Saves per-ISO-year JSON files plus aggregated profiles.

Usage:
  python fetch_all_data.py                  # Fetch all 7 ISOs
  python fetch_all_data.py MISO SWPP        # Fetch only MISO and SPP
  python fetch_all_data.py CISO ERCO        # Fetch only CAISO and ERCOT
"""
import json
import os
import sys
import time
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timedelta

API_KEY = "rCMyAk40PIFypAygazPXuEI9cYCddmhBdRKaoBNJ"
GEN_URL = "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"
DEMAND_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"

RESPONDENTS = ["CISO", "ERCO", "PJM", "NYIS", "ISNE", "MISO", "SWPP"]
YEARS = [2021, 2022, 2023, 2024, 2025]

FUEL_MAP = {
    "COL": "coal",
    "NG":  "gas",
    "NUC": "nuclear",
    "OIL": "oil",
    "OTH": "other",
    "SUN": "solar",
    "WAT": "hydro",
    "WND": "wind",
}

REGION_MAP = {
    "CISO": "CAISO",
    "ERCO": "ERCOT",
    "PJM":  "PJM",
    "NYIS": "NYISO",
    "ISNE": "NEISO",
    "MISO": "MISO",
    "SWPP": "SPP",
}

PAGE_SIZE = 5000
MAX_RETRIES = 3
CALL_DELAY = 0.5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")


def hours_in_year(year):
    start = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1)
    return int((end - start).total_seconds() / 3600)


def build_hour_index(year):
    n = hours_in_year(year)
    base = datetime(year, 1, 1)
    return [(base + timedelta(hours=h)).strftime("%Y-%m-%dT%H") for h in range(n)]


def api_get(base_url, params, retry=0):
    params_copy = dict(params)
    params_copy["api_key"] = API_KEY
    url = base_url + "?" + urllib.parse.urlencode(params_copy, doseq=True)
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")
            data = json.loads(body)
            return data
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError) as exc:
        if retry < MAX_RETRIES:
            wait = 2 ** retry
            print(f"    [RETRY {retry+1}/{MAX_RETRIES}] {exc} -- waiting {wait}s")
            time.sleep(wait)
            return api_get(base_url, params, retry + 1)
        else:
            print(f"    [FAILED] Giving up after {MAX_RETRIES} retries: {exc}")
            raise


def fetch_paged(base_url, params, label, call_counter):
    all_records = []
    offset = 0
    while True:
        call_counter += 1
        print(f"  API call #{call_counter}: {label}, offset={offset}")
        params_copy = dict(params)
        params_copy["offset"] = str(offset)
        params_copy["length"] = str(PAGE_SIZE)
        resp = api_get(base_url, params_copy)
        time.sleep(CALL_DELAY)
        data_block = resp.get("response", {}).get("data", [])
        total = int(resp.get("response", {}).get("total", 0))
        if not data_block:
            print(f"    No data returned (total reported: {total})")
            break
        all_records.extend(data_block)
        print(f"    Got {len(data_block)} records (cumulative: {len(all_records)} / {total})")
        offset += PAGE_SIZE
        if offset >= total:
            break
    return all_records, call_counter


# ---- Generation processing ----

def records_to_hourly_gen(records, year):
    hourly = {}
    for rec in records:
        period = rec.get("period", "")
        fueltype = rec.get("fueltype", "")
        raw_val = rec.get("value")
        key = FUEL_MAP.get(fueltype)
        if key is None:
            continue
        try:
            val = float(raw_val) if raw_val is not None else 0.0
        except (ValueError, TypeError):
            val = 0.0
        if period not in hourly:
            hourly[period] = {k: 0.0 for k in FUEL_MAP.values()}
        hourly[period][key] = val
    return hourly


def build_full_gen_array(hourly_dict, year):
    n_hours = hours_in_year(year)
    all_hours = build_hour_index(year)
    fuel_keys = list(FUEL_MAP.values())

    raw = []
    for h_str in all_hours:
        raw.append(hourly_dict.get(h_str))

    missing_count = sum(1 for r in raw if r is None)
    if missing_count > 0:
        print(f"  Missing gen hours: {missing_count} / {n_hours} -- will interpolate")

    filled = []
    for i in range(n_hours):
        if raw[i] is not None:
            filled.append(dict(raw[i]))
        else:
            prev_idx = next_idx = None
            for p in range(i - 1, -1, -1):
                if raw[p] is not None:
                    prev_idx = p
                    break
            for nn in range(i + 1, n_hours):
                if raw[nn] is not None:
                    next_idx = nn
                    break
            if prev_idx is not None and next_idx is not None:
                span = next_idx - prev_idx
                frac = (i - prev_idx) / span
                interp = {}
                for k in fuel_keys:
                    interp[k] = raw[prev_idx][k] + frac * (raw[next_idx][k] - raw[prev_idx][k])
                filled.append(interp)
            elif prev_idx is not None:
                filled.append(dict(raw[prev_idx]))
            elif next_idx is not None:
                filled.append(dict(raw[next_idx]))
            else:
                filled.append({k: 0.0 for k in fuel_keys})

    result = []
    for i, h_str in enumerate(all_hours):
        entry = {"period": h_str}
        fuels = filled[i]
        for k in fuel_keys:
            entry[k] = round(fuels[k], 2)
        total_fossil = fuels["coal"] + fuels["gas"] + fuels["oil"]
        entry["total_fossil"] = round(total_fossil, 2)
        if total_fossil > 0:
            entry["coal_share"] = round(fuels["coal"] / total_fossil, 6)
            entry["gas_share"] = round(fuels["gas"] / total_fossil, 6)
            entry["oil_share"] = round(fuels["oil"] / total_fossil, 6)
        else:
            entry["coal_share"] = 0.0
            entry["gas_share"] = 0.0
            entry["oil_share"] = 0.0
        result.append(entry)
    return result


# ---- Demand processing ----

def records_to_hourly_demand(records):
    hourly = {}
    for rec in records:
        period = rec.get("period", "")
        raw_val = rec.get("value")
        try:
            val = float(raw_val) if raw_val is not None else 0.0
        except (ValueError, TypeError):
            val = 0.0
        hourly[period] = val
    return hourly


def build_full_demand_array(hourly_dict, year):
    n_hours = hours_in_year(year)
    all_hours = build_hour_index(year)

    raw = []
    for h_str in all_hours:
        raw.append(hourly_dict.get(h_str))

    missing_count = sum(1 for r in raw if r is None)
    if missing_count > 0:
        print(f"  Missing demand hours: {missing_count} / {n_hours} -- will interpolate")

    filled = []
    for i in range(n_hours):
        if raw[i] is not None:
            filled.append(raw[i])
        else:
            prev_idx = next_idx = None
            for p in range(i - 1, -1, -1):
                if raw[p] is not None:
                    prev_idx = p
                    break
            for nn in range(i + 1, n_hours):
                if raw[nn] is not None:
                    next_idx = nn
                    break
            if prev_idx is not None and next_idx is not None:
                span = next_idx - prev_idx
                frac = (i - prev_idx) / span
                filled.append(raw[prev_idx] + frac * (raw[next_idx] - raw[prev_idx]))
            elif prev_idx is not None:
                filled.append(raw[prev_idx])
            elif next_idx is not None:
                filled.append(raw[next_idx])
            else:
                filled.append(0.0)

    return [{"period": all_hours[i], "demand_mw": round(filled[i], 2)} for i in range(n_hours)]


# ---- Aggregate builders ----

def build_generation_profiles(all_gen):
    profiles = {}
    for (respondent, year), hourly_array in all_gen.items():
        region = REGION_MAP[respondent]
        year_str = str(year)
        if region not in profiles:
            profiles[region] = {}
        if region == "CAISO":
            profile_fuels = {
                "solar": "solar", "wind": "wind", "nuclear": "nuclear",
                "hydro": "hydro", "geothermal": "other"
            }
        else:
            profile_fuels = {
                "solar": "solar", "wind": "wind", "nuclear": "nuclear",
                "hydro": "hydro"
            }
        fuel_lists = {}
        for profile_name, source_key in profile_fuels.items():
            values = [h[source_key] for h in hourly_array]
            annual_total = sum(values)
            if annual_total > 0:
                normalized = [round(v / annual_total, 10) for v in values]
            else:
                nv = len(values)
                normalized = [round(1.0 / nv, 10)] * nv
            fuel_lists[profile_name] = normalized
        profiles[region][year_str] = fuel_lists
    for year_str in profiles.get("NYISO", {}):
        if "NEISO" in profiles and year_str in profiles["NEISO"]:
            profiles["NYISO"][year_str]["solar_proxy"] = list(
                profiles["NEISO"][year_str].get("solar", [])
            )
    return profiles


def build_fossil_mix(all_gen):
    fossil_mix = {}
    for (respondent, year), hourly_array in all_gen.items():
        region = REGION_MAP[respondent]
        year_str = str(year)
        if region not in fossil_mix:
            fossil_mix[region] = {}
        fossil_mix[region][year_str] = {
            "hours": [h["period"] for h in hourly_array],
            "coal_share": [h["coal_share"] for h in hourly_array],
            "gas_share": [h["gas_share"] for h in hourly_array],
            "oil_share": [h["oil_share"] for h in hourly_array],
        }
    return fossil_mix


def build_demand_profiles(all_demand):
    profiles = {}
    for (respondent, year), hourly_array in all_demand.items():
        region = REGION_MAP[respondent]
        year_str = str(year)
        values = [h["demand_mw"] for h in hourly_array]
        total = sum(values)
        if total > 0:
            normalized = [v / total for v in values]
        else:
            normalized = [1.0 / len(values)] * len(values)

        if region not in profiles:
            profiles[region] = {}
        profiles[region][year_str] = {
            "raw_mw": [round(v, 2) for v in values],
            "normalized": normalized,
            "total_annual_mwh": round(total),
            "peak_mw": round(max(values)),
            "min_mw": round(min(values)),
            "avg_mw": round(total / len(values)),
        }
    return profiles


def save_json(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  Saved: {os.path.basename(filepath)} ({size_mb:.2f} MB)")


def load_json(filepath):
    """Load existing JSON file, return empty dict if missing."""
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def main():
    # CLI arg filtering: specify BA codes to fetch a subset
    if len(sys.argv) > 1:
        requested = [a.upper() for a in sys.argv[1:]]
        run_respondents = [r for r in RESPONDENTS if r in requested]
        if not run_respondents:
            print(f"Error: none of {requested} found in {RESPONDENTS}")
            sys.exit(1)
    else:
        run_respondents = RESPONDENTS

    print("=" * 70)
    print("EIA Hourly Generation + Demand Fetcher")
    print(f"Regions: {', '.join(REGION_MAP[r] for r in run_respondents)}")
    print(f"Years: {', '.join(str(y) for y in YEARS)}")
    print(f"Output dir: {DATA_DIR}")
    print("=" * 70)
    os.makedirs(DATA_DIR, exist_ok=True)

    call_counter = 0

    # ---- PHASE 1: Fetch generation data ----
    print()
    print("=== PHASE 1: Fetching hourly generation data ===")
    print()
    raw_gen = {}
    for respondent in run_respondents:
        for year in YEARS:
            region = REGION_MAP[respondent]
            start_str = f"{year}-01-01T00"
            end_str = f"{year}-12-31T23"
            params = {
                "frequency": "hourly",
                "data[0]": "value",
                "facets[respondent][]": respondent,
                "start": start_str,
                "end": end_str,
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
            }
            label = f"GEN {region} {year}"
            print(f"Fetching {label}:")
            records, call_counter = fetch_paged(GEN_URL, params, label, call_counter)
            raw_gen[(respondent, year)] = records
            print(f"  Total records: {len(records)}")
            print()

    # ---- PHASE 2: Fetch demand data ----
    print("=== PHASE 2: Fetching hourly demand data ===")
    print()
    raw_demand = {}
    for respondent in run_respondents:
        for year in YEARS:
            region = REGION_MAP[respondent]
            start_str = f"{year}-01-01T00"
            end_str = f"{year}-12-31T23"
            params = {
                "frequency": "hourly",
                "data[0]": "value",
                "facets[respondent][]": respondent,
                "facets[type][]": "D",  # D = Demand
                "start": start_str,
                "end": end_str,
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
            }
            label = f"DEMAND {region} {year}"
            print(f"Fetching {label}:")
            records, call_counter = fetch_paged(DEMAND_URL, params, label, call_counter)
            raw_demand[(respondent, year)] = records
            print(f"  Total records: {len(records)}")
            print()

    print(f"Total API calls made: {call_counter}")
    print()

    # ---- PHASE 3: Process generation ----
    print("=== PHASE 3: Processing generation data ===")
    print()
    all_gen = {}
    for respondent in run_respondents:
        for year in YEARS:
            region = REGION_MAP[respondent]
            print(f"Processing GEN {region} {year}...")
            records = raw_gen[(respondent, year)]
            hourly_dict = records_to_hourly_gen(records, year)
            n_hours = hours_in_year(year)
            print(f"  Unique hours with data: {len(hourly_dict)} / {n_hours}")
            full_array = build_full_gen_array(hourly_dict, year)
            all_gen[(respondent, year)] = full_array
            total_gen = sum(
                h["gas"] + h["coal"] + h["nuclear"] + h["solar"] +
                h["wind"] + h["hydro"] + h["oil"] + h["other"]
                for h in full_array
            )
            print(f"  Total generation: {total_gen:,.0f} MWh")
            print()

    # ---- PHASE 4: Process demand ----
    print("=== PHASE 4: Processing demand data ===")
    print()
    all_demand = {}
    for respondent in run_respondents:
        for year in YEARS:
            region = REGION_MAP[respondent]
            print(f"Processing DEMAND {region} {year}...")
            records = raw_demand[(respondent, year)]
            hourly_dict = records_to_hourly_demand(records)
            n_hours = hours_in_year(year)
            print(f"  Unique hours with data: {len(hourly_dict)} / {n_hours}")
            full_array = build_full_demand_array(hourly_dict, year)
            all_demand[(respondent, year)] = full_array
            total_demand = sum(h["demand_mw"] for h in full_array)
            print(f"  Total demand: {total_demand:,.0f} MWh")
            print()

    # ---- PHASE 5: Save output files ----
    print("=== PHASE 5: Saving output files ===")
    print()

    # Per-ISO-year generation files
    for (respondent, year), hourly_array in all_gen.items():
        region = REGION_MAP[respondent]
        filename = f"eia_hourly_{region}_{year}.json"
        filepath = os.path.join(DATA_DIR, filename)
        save_json(filepath, hourly_array)

    # Per-ISO-year demand files
    for (respondent, year), hourly_array in all_demand.items():
        region = REGION_MAP[respondent]
        filename = f"eia_demand_{region}_{year}.json"
        filepath = os.path.join(DATA_DIR, filename)
        save_json(filepath, hourly_array)

    # Aggregated profiles — merge with existing data (preserves ISOs not fetched)
    print()
    gen_path = os.path.join(DATA_DIR, "eia_generation_profiles.json")
    fossil_path = os.path.join(DATA_DIR, "eia_fossil_mix.json")
    demand_path = os.path.join(DATA_DIR, "eia_demand_profiles.json")

    print("Building generation profiles...")
    new_profiles = build_generation_profiles(all_gen)
    existing_profiles = load_json(gen_path)
    existing_profiles.update(new_profiles)
    save_json(gen_path, existing_profiles)

    print("Building fossil mix...")
    new_fossil = build_fossil_mix(all_gen)
    existing_fossil = load_json(fossil_path)
    existing_fossil.update(new_fossil)
    save_json(fossil_path, existing_fossil)

    print("Building demand profiles...")
    new_demand = build_demand_profiles(all_demand)
    existing_demand = load_json(demand_path)
    existing_demand.update(new_demand)
    save_json(demand_path, existing_demand)

    print()
    print("=" * 70)
    print("DONE! Output files:")
    print("=" * 70)
    for fname in sorted(os.listdir(DATA_DIR)):
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"  {fname} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
