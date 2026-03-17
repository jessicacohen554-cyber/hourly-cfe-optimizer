"""
Generate plant-specific heat rates from EIA 923 or EPA CAMPD data.

Priority chain:
  1. EIA 923: heat_rate = total-consumption-btu / gross-generation  (MMBtu/MWh)
  2. EPA CAMPD: heat_rate = mmbtu / gross_load  (MMBtu/MWh)
  3. Synthetic defaults by fuel type

Output: data/plant_heat_rates.json
  { "<plantCode>": { "heat_rate": float, "source": str, "plant_name": str, "state": str } }
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_FILE = DATA_DIR / "plant_heat_rates.json"

# Default heat rates (MMBtu/MWh) by fuel type
DEFAULTS = {
    "coal": 10.0,
    "gas": 7.2,
    "oil": 10.5,
    "nuclear": 10.4,
    "other": 9.0,
}


def load_eia923_heat_rates():
    """Compute heat rates from EIA 923 monthly generation/fuel data."""
    eia923_dir = DATA_DIR / "eia-923"
    if not eia923_dir.exists():
        return {}

    # Accumulate annual totals per plant
    plant_btu = defaultdict(float)
    plant_gen = defaultdict(float)
    plant_meta = {}

    for state_dir in sorted(eia923_dir.iterdir()):
        if not state_dir.is_dir():
            continue
        for json_file in sorted(state_dir.glob("*.json")):
            try:
                with open(json_file) as f:
                    records = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            if not isinstance(records, list):
                continue

            for rec in records:
                plant_code = str(rec.get("plantCode", ""))
                if not plant_code:
                    continue

                # Skip aggregate rows (fuel2002 == "ALL" / primeMover == "ALL")
                # We want fuel-specific rows to sum up
                gross_gen = rec.get("gross-generation")
                total_btu = rec.get("total-consumption-btu")

                if gross_gen is None or total_btu is None:
                    continue

                try:
                    gen_mwh = float(gross_gen)
                    btu = float(total_btu)
                except (TypeError, ValueError):
                    continue

                if gen_mwh <= 0 or btu <= 0:
                    continue

                plant_btu[plant_code] += btu
                plant_gen[plant_code] += gen_mwh

                if plant_code not in plant_meta:
                    plant_meta[plant_code] = {
                        "plant_name": rec.get("plantName", ""),
                        "state": rec.get("state", ""),
                    }

    # Compute heat rates
    results = {}
    for plant_code in plant_btu:
        if plant_gen[plant_code] > 0:
            hr = plant_btu[plant_code] / plant_gen[plant_code]
            # Sanity check: reasonable range 5-20 MMBtu/MWh
            if 5.0 <= hr <= 20.0:
                results[plant_code] = {
                    "heat_rate": round(hr, 2),
                    "source": "eia923",
                    **plant_meta.get(plant_code, {}),
                }

    return results


def load_campd_heat_rates():
    """Compute heat rates from EPA CAMPD hourly data."""
    campd_dir = DATA_DIR / "epa-campd"
    if not campd_dir.exists():
        return {}

    plant_btu = defaultdict(float)
    plant_gen = defaultdict(float)
    plant_meta = {}

    for state_dir in sorted(campd_dir.iterdir()):
        if not state_dir.is_dir():
            continue
        for json_file in sorted(state_dir.glob("*.json")):
            try:
                with open(json_file) as f:
                    records = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            if not isinstance(records, list):
                continue

            for rec in records:
                # CAMPD uses different field names — check common variants
                plant_id = str(rec.get("facilityId", rec.get("orisCode", rec.get("plantCode", ""))))
                if not plant_id:
                    continue

                mmbtu = rec.get("heatInput", rec.get("mmbtu", 0))
                gross_load = rec.get("grossLoad", rec.get("gross_load", rec.get("grossGeneration", 0)))

                try:
                    mmbtu = float(mmbtu or 0)
                    gross_load = float(gross_load or 0)
                except (TypeError, ValueError):
                    continue

                if mmbtu <= 0 or gross_load <= 0:
                    continue

                plant_btu[plant_id] += mmbtu
                plant_gen[plant_id] += gross_load

                if plant_id not in plant_meta:
                    plant_meta[plant_id] = {
                        "plant_name": rec.get("facilityName", rec.get("plantName", "")),
                        "state": rec.get("state", rec.get("stateCode", "")),
                    }

    results = {}
    for plant_id in plant_btu:
        if plant_gen[plant_id] > 0:
            hr = plant_btu[plant_id] / plant_gen[plant_id]
            if 5.0 <= hr <= 20.0:
                results[plant_id] = {
                    "heat_rate": round(hr, 2),
                    "source": "campd",
                    **plant_meta.get(plant_id, {}),
                }

    return results


def main():
    print("Generating plant heat rates...")

    # Priority 1: EIA 923
    heat_rates = load_eia923_heat_rates()
    eia923_count = len(heat_rates)
    print(f"  EIA 923: {eia923_count} plants with valid heat rates")

    # Priority 2: EPA CAMPD (fill gaps only)
    campd_rates = load_campd_heat_rates()
    campd_added = 0
    for plant_id, data in campd_rates.items():
        if plant_id not in heat_rates:
            heat_rates[plant_id] = data
            campd_added += 1
    print(f"  EPA CAMPD: {campd_added} additional plants (total: {len(heat_rates)})")

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(heat_rates, f, indent=2)
    print(f"  Saved to {OUTPUT_FILE} ({len(heat_rates)} plants)")


if __name__ == "__main__":
    main()
