#!/usr/bin/env python3
"""
One-time data export: convert JSON/parquet from the main repo into
portable CSVs inside CEG-IPP-target/data/.

Run from the repo root:
    python CEG-IPP-target/export_data.py

After running, the CEG-IPP-target/ directory is fully self-contained.
"""

import json
import sys
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
OUT_DATA = SCRIPT_DIR / "data"
OUT_DATA.mkdir(parents=True, exist_ok=True)

FLEET_JSON = REPO_ROOT / "data" / "ipp_fleet_config.json"
SWEEP_PARQUET = REPO_ROOT / "data" / "step6-smartargets" / "ipp_sweep_results.parquet"
SMARTARGETS_DIR = REPO_ROOT / "data" / "step6-smartargets"


# ══════════════════════════════════════════════════════════════════════
# 1. Fleet Config → ipp_fleet_config.csv (one row per plant)
#                  + ipp_fleet_companies.csv (one row per company)
# ══════════════════════════════════════════════════════════════════════

def export_fleet():
    """Flatten fleet JSON into per-plant and per-company CSVs."""
    with open(FLEET_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    plant_rows = []
    company_rows = []

    for co in data["companies"]:
        cid = co["id"]
        cname = co["name"]
        short = co["shortName"]

        # Per-fuel aggregation for company-level CSV
        fuel_mw = {}
        iso_set = set()

        for p in co["plants"]:
            plant_rows.append({
                "company_id": cid,
                "company_name": cname,
                "short_name": short,
                "co2_2023_mt": co.get("co2_2023_mt"),
                "co2_2024_mt": co.get("co2_2024_mt"),
                "intensity_kg": co.get("intensity_kg"),
                "company_gen_twh": co.get("gen_twh"),
                "company_cap_gw": co.get("cap_gw"),
                "target": co.get("target", ""),
                "plant_name": p["name"],
                "iso": p["iso"],
                "fuel": p["fuel"],
                "cap_mw": p["cap_mw"],
                "gen_twh": p["gen_twh"],
                "co2_mt": p["co2_mt"],
                "heat_rate": p.get("hr"),
                "retire_by": p.get("retire_by"),
            })

            fuel = p["fuel"]
            fuel_mw[fuel] = fuel_mw.get(fuel, 0) + p["cap_mw"]
            iso_set.add(p["iso"])

        primary_iso = max(
            iso_set,
            key=lambda iso: sum(
                p["cap_mw"] for p in co["plants"] if p["iso"] == iso
            ),
        )

        company_rows.append({
            "company_id": cid,
            "company_name": cname,
            "short_name": short,
            "co2_2023_mt": co.get("co2_2023_mt"),
            "co2_2024_mt": co.get("co2_2024_mt"),
            "intensity_kg": co.get("intensity_kg"),
            "gen_twh": co.get("gen_twh"),
            "cap_gw": co.get("cap_gw"),
            "target": co.get("target", ""),
            "nuclear_mw": fuel_mw.get("nuclear", 0),
            "coal_mw": fuel_mw.get("coal", 0),
            "gas_ccgt_mw": fuel_mw.get("gas_ccgt", 0),
            "gas_peaker_mw": fuel_mw.get("gas_peaker", 0),
            "solar_mw": fuel_mw.get("solar", 0),
            "wind_mw": fuel_mw.get("wind", 0),
            "battery_mw": fuel_mw.get("battery", 0),
            "hydro_mw": fuel_mw.get("hydro", 0),
            "geothermal_mw": fuel_mw.get("geothermal", 0),
            "oil_mw": fuel_mw.get("oil", 0),
            "iso_count": len(iso_set),
            "primary_iso": primary_iso,
        })

    df_plants = pd.DataFrame(plant_rows)
    df_companies = pd.DataFrame(company_rows)

    df_plants.to_csv(OUT_DATA / "ipp_fleet_config.csv", index=False)
    df_companies.to_csv(OUT_DATA / "ipp_fleet_companies.csv", index=False)

    print(f"  ipp_fleet_config.csv    : {len(df_plants)} rows (plants)")
    print(f"  ipp_fleet_companies.csv : {len(df_companies)} rows (companies)")


# ══════════════════════════════════════════════════════════════════════
# 2. Sweep Results parquet → ipp_sweep_results.csv
# ══════════════════════════════════════════════════════════════════════

def export_sweep():
    """Convert parquet sweep results to CSV."""
    if not SWEEP_PARQUET.exists():
        print(f"  WARNING: {SWEEP_PARQUET} not found, skipping sweep export.")
        return

    df = pd.read_parquet(SWEEP_PARQUET)
    df.to_csv(OUT_DATA / "ipp_sweep_results.csv", index=False)
    print(f"  ipp_sweep_results.csv   : {len(df)} rows")


# ══════════════════════════════════════════════════════════════════════
# 3. Pipeline constants → pipeline_constants.csv
# ══════════════════════════════════════════════════════════════════════

def export_constants():
    """Extract key constants from pipeline_config.py as key-value CSV."""
    # Import pipeline_config dynamically
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import pipeline_config as pc
    except ImportError:
        print("  WARNING: Could not import pipeline_config, skipping constants export.")
        return

    rows = []

    # Wholesale prices
    if hasattr(pc, "WHOLESALE_PRICES"):
        for iso, price in pc.WHOLESALE_PRICES.items():
            rows.append({
                "key": f"wholesale_price_{iso}",
                "value": price,
                "unit": "$/MWh",
                "source": "pipeline_config.WHOLESALE_PRICES",
            })

    # Capacity market prices
    if hasattr(pc, "CAPACITY_MARKET_PRICES"):
        for iso, price in pc.CAPACITY_MARKET_PRICES.items():
            rows.append({
                "key": f"capacity_price_{iso}",
                "value": price,
                "unit": "$/kW-yr",
                "source": "pipeline_config.CAPACITY_MARKET_PRICES",
            })

    # Gas prices
    if hasattr(pc, "GAS_PRICES"):
        for level, price in pc.GAS_PRICES.items():
            rows.append({
                "key": f"gas_price_{level}",
                "value": price,
                "unit": "$/MMBtu",
                "source": "pipeline_config.GAS_PRICES",
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DATA / "pipeline_constants.csv", index=False)
    print(f"  pipeline_constants.csv  : {len(df)} rows")


# ══════════════════════════════════════════════════════════════════════
# 4. SMARTargets sweep summaries → smartargets_sweep_summary.csv
# ══════════════════════════════════════════════════════════════════════

def export_smartargets_summaries():
    """Merge the 3 sweep summary JSONs into one CSV."""
    scenarios = ["reference", "power_nz", "economy_nz"]
    all_rows = []

    for scenario in scenarios:
        path = SMARTARGETS_DIR / f"smartargets_sweep_{scenario}_summary.json"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping.")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for row in data:
                row["scenario_type"] = scenario
                all_rows.append(row)
        elif isinstance(data, dict):
            data["scenario_type"] = scenario
            all_rows.append(data)

    if all_rows:
        df = pd.json_normalize(all_rows)
        df.to_csv(OUT_DATA / "smartargets_sweep_summary.csv", index=False)
        print(f"  smartargets_sweep_summary.csv : {len(df)} rows")
    else:
        print("  smartargets_sweep_summary.csv : no data found")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("Exporting data to CEG-IPP-target/data/ ...")
    print()
    export_fleet()
    export_sweep()
    export_constants()
    export_smartargets_summaries()
    print()
    print("Done. The CEG-IPP-target/ directory is now self-contained.")
    print("You can copy it to any machine with Python + pandas and regenerate all outputs.")


if __name__ == "__main__":
    main()
