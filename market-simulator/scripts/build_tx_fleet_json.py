#!/usr/bin/env python3
"""
Build a comprehensive TX ERCOT fleet JSON for the client-side IPP portfolio tool.
Merges EIA 860 (inventory) + EIA 923 (generation/fuel) for ALL generator types:
fossil, nuclear, wind, solar, batteries, hydro.

Outputs owner-level portfolio data with CCGT heat rate binning for stranding analysis.

Output: dashboard/js/tx-fleet-data.js
"""
import json
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "frontend" / "js" / "tx-fleet-data.js"

# ── Fuel code → category mapping ──
FUEL_CATEGORIES = {
    # Fossil
    "BIT": "coal", "SUB": "coal", "LIG": "coal", "RC": "coal", "WC": "coal", "SC": "coal", "COL": "coal",
    "NG": "gas", "OG": "gas", "BFG": "gas", "SG": "gas",
    "DFO": "oil", "RFO": "oil", "JF": "oil", "KER": "oil", "PC": "oil", "WO": "oil", "SGP": "oil",
    # Clean
    "NUC": "nuclear",
    "WND": "wind",
    "SUN": "solar",
    "WAT": "hydro",
    "MWH": "battery",
    # Other
    "LFG": "biomass", "WDS": "biomass", "BLQ": "biomass", "OBS": "biomass",
    "WH": "other", "PUR": "other", "OTH": "other",
}

# CO2 emission rates (tons CO2 / MMBtu)
CO2_RATES = {
    "coal": 0.0953,
    "gas": 0.0531,
    "oil": 0.0733,
}

# VOM defaults ($/MWh)
VOM_DEFAULTS = {
    "coal_steam": 5.50,
    "gas_ccgt": 3.50,
    "gas_ct": 5.00,
    "gas_steam": 4.00,
    "oil_ct": 6.00,
    "nuclear": 2.50,
    "wind": 0.00,
    "solar": 0.00,
    "battery": 0.50,
    "hydro": 1.50,
}

# CCGT heat rate bins for stranding analysis
CCGT_HR_BINS = [
    {"label": "Efficient CCGT", "tag": "ccgt_eff", "min": 0, "max": 6.8},
    {"label": "Modern CCGT", "tag": "ccgt_mod", "min": 6.8, "max": 7.3},
    {"label": "Average CCGT", "tag": "ccgt_avg", "min": 7.3, "max": 8.0},
    {"label": "Old CCGT", "tag": "ccgt_old", "min": 8.0, "max": 9.5},
    {"label": "Inefficient CCGT", "tag": "ccgt_ineff", "min": 9.5, "max": 99},
]


def classify_unit(prime_mover, fuel_code):
    """Classify a generator into detailed unit type + fuel category."""
    fuel_code = (fuel_code or "").upper().strip()
    pm = (prime_mover or "").upper().strip()
    fuel_cat = FUEL_CATEGORIES.get(fuel_code)

    if not fuel_cat:
        return None, None

    # Nuclear
    if fuel_cat == "nuclear":
        return "nuclear", "nuclear"

    # Renewables
    if fuel_cat == "wind":
        return "wind", "wind"
    if fuel_cat == "solar":
        return "solar", "solar"
    if fuel_cat == "hydro":
        return "hydro", "hydro"
    if fuel_cat == "battery":
        return "battery", "battery"
    if fuel_cat == "biomass":
        return "biomass", "biomass"

    # Coal
    if fuel_cat == "coal" and pm == "ST":
        return "coal_steam", "coal"

    # Gas
    if fuel_cat == "gas":
        if pm in ("CA", "CS", "CC"):
            return "gas_ccgt", "gas"
        elif pm == "CT":
            return "gas_ct", "gas"
        elif pm == "ST":
            return "gas_steam", "gas"
        elif pm == "GT":
            return "gas_ct", "gas"
        elif pm == "IC":
            return "gas_ic", "gas"
        return "gas_other", "gas"

    # Oil
    if fuel_cat == "oil":
        if pm in ("GT", "CT", "IC"):
            return "oil_ct", "oil"
        elif pm == "ST":
            return "oil_steam", "oil"
        return "oil_other", "oil"

    return None, None


def get_ccgt_hr_bin(heat_rate):
    """Assign a CCGT to a heat rate bin for stranding analysis."""
    for b in CCGT_HR_BINS:
        if b["min"] <= heat_rate < b["max"]:
            return b["tag"], b["label"]
    return "ccgt_other", "Other CCGT"


def load_860():
    """Load ALL EIA 860 TX operating generators."""
    path = DATA / "eia-860" / "TX" / "eia860_TX.json"
    with open(path) as f:
        raw = json.load(f)

    gens = {}
    for r in raw:
        status = (r.get("status") or "").upper()
        if status != "OP":
            continue

        plant_id = str(r.get("plantid", ""))
        gen_id = str(r.get("generatorid", ""))
        key = f"{plant_id}_{gen_id}"

        pm = r.get("prime_mover_code", "")
        fuel = r.get("energy_source_code", "")
        unit_type, fuel_cat = classify_unit(pm, fuel)
        if not unit_type:
            continue

        cap = float(r.get("net-summer-capacity-mw") or r.get("nameplate-capacity-mw") or 0)
        if cap <= 0:
            continue

        op_year = r.get("operating-year-month", "")
        year = int(op_year[:4]) if op_year and len(op_year) >= 4 else 2000

        ba = r.get("balancing_authority_code", "")
        entity = r.get("entityName", "")
        sector = r.get("sector", "")
        technology = r.get("technology", "")

        gens[key] = {
            "plant_id": plant_id,
            "gen_id": gen_id,
            "plant_name": r.get("plantName", ""),
            "entity": entity,
            "sector": sector,
            "technology": technology,
            "county": r.get("county", ""),
            "lat": float(r.get("latitude") or 0),
            "lon": float(r.get("longitude") or 0),
            "ba": ba,
            "unit_type": unit_type,
            "fuel_cat": fuel_cat,
            "prime_mover": pm,
            "fuel_code": fuel,
            "capacity_mw": cap,
            "online_year": year,
            "age": 2025 - year,
            "retirement": r.get("planned-retirement-year-month"),
        }

    return gens


def load_923():
    """Load EIA 923 TX generation/fuel data for 2024 (full year)."""
    plant_fuel = defaultdict(lambda: {"gen_mwh": 0, "fuel_mmbtu": 0})

    for month in range(1, 13):
        path = DATA / "eia-923" / "TX" / f"eia923_TX_2024-{month:02d}.json"
        if not path.exists():
            continue

        with open(path) as f:
            raw = json.load(f)

        for r in raw:
            fuel = (r.get("fuel2002") or "").upper()
            pm = (r.get("primeMover") or "").upper()

            if fuel == "ALL" or pm == "ALL":
                continue

            plant_id = str(r.get("plantCode", ""))
            gen_mwh = float(r.get("generation") or 0)
            fuel_btu = float(r.get("total-consumption-btu") or 0)

            key = f"{plant_id}_{pm}_{fuel}"
            plant_fuel[key]["gen_mwh"] += gen_mwh
            plant_fuel[key]["fuel_mmbtu"] += fuel_btu
            plant_fuel[key]["plant_id"] = plant_id
            plant_fuel[key]["pm"] = pm
            plant_fuel[key]["fuel"] = fuel

    return plant_fuel


def compute_operational_data(gens, fuel_data):
    """Compute heat rates, capacity factors, and generation from 923 data."""
    # Aggregate fuel data by plant_id + prime_mover combo
    plant_pm_fuel = defaultdict(lambda: {"gen": 0, "fuel": 0})
    # Also aggregate by plant_id only (for nuclear/renewables which don't vary by PM)
    plant_fuel_only = defaultdict(lambda: {"gen": 0, "fuel": 0})

    for k, v in fuel_data.items():
        plant_id = v.get("plant_id", "")
        pm = v.get("pm", "")
        fuel = v.get("fuel", "")

        pk = f"{plant_id}_{pm}"
        plant_pm_fuel[pk]["gen"] += v["gen_mwh"]
        plant_pm_fuel[pk]["fuel"] += v["fuel_mmbtu"]

        pk2 = f"{plant_id}_{fuel}"
        plant_fuel_only[pk2]["gen"] += v["gen_mwh"]
        plant_fuel_only[pk2]["fuel"] += v["fuel_mmbtu"]

    # Default heat rates by unit type
    default_hr = {
        "coal_steam": 10.0,
        "gas_ccgt": 7.0,
        "gas_ct": 10.5,
        "gas_steam": 9.0,
        "gas_ic": 9.5,
        "gas_other": 9.5,
        "oil_ct": 10.5,
        "oil_steam": 11.0,
        "oil_other": 11.0,
    }

    for key, gen in gens.items():
        fuel_cat = gen["fuel_cat"]
        is_thermal = fuel_cat in ("coal", "gas", "oil")

        # Try plant_id + prime_mover match first (fossil)
        pk = f"{gen['plant_id']}_{gen['prime_mover']}"
        data = plant_pm_fuel.get(pk)

        # Fallback to plant_id + fuel code for renewables/nuclear
        if not data or data["gen"] < 100:
            pk2 = f"{gen['plant_id']}_{gen['fuel_code']}"
            data2 = plant_fuel_only.get(pk2)
            if data2 and data2["gen"] > data.get("gen", 0) if data else True:
                data = data2

        if data and data["gen"] > 100:
            gen["annual_gen_mwh"] = round(data["gen"])
            gen["capacity_factor"] = round(
                data["gen"] / (gen["capacity_mw"] * 8760), 3
            ) if gen["capacity_mw"] > 0 else 0

            # Compute heat rate for thermal units
            if is_thermal and data["fuel"] > 0 and data["gen"] > 1000:
                hr = data["fuel"] / data["gen"]
                if 4.0 < hr < 20.0:
                    gen["heat_rate"] = round(hr, 2)
                    gen["heat_rate_source"] = "revealed"
                    gen["annual_fuel_mmbtu"] = round(data["fuel"])
                    continue

        # Defaults for units without good 923 data
        if is_thermal:
            gen["heat_rate"] = default_hr.get(gen["unit_type"], 10.0)
            gen["heat_rate_source"] = "default"
        else:
            gen["heat_rate"] = 0
            gen["heat_rate_source"] = "n/a"

        gen.setdefault("annual_gen_mwh", 0)
        gen.setdefault("capacity_factor", 0)


def build_output(gens):
    """Build comprehensive output with portfolio analysis."""
    # Filter to ERCOT only
    ercot_gens = [g for g in gens.values() if g["ba"] == "ERCO"]

    # Sort: fossil by unit_type + heat_rate, clean by type + capacity
    def sort_key(g):
        type_order = {
            "nuclear": 0, "wind": 1, "solar": 2, "hydro": 3, "battery": 4,
            "coal_steam": 10, "gas_ccgt": 11, "gas_steam": 12, "gas_ct": 13,
            "gas_ic": 14, "gas_other": 15, "oil_ct": 16, "oil_steam": 17,
            "oil_other": 18, "biomass": 19,
        }
        return (type_order.get(g["unit_type"], 99), g.get("heat_rate", 0), -g["capacity_mw"])

    ercot_gens.sort(key=sort_key)

    # Build generator records
    generators = []
    for g in ercot_gens:
        fuel_cat = g["fuel_cat"]
        is_thermal = fuel_cat in ("coal", "gas", "oil")
        hr = g.get("heat_rate", 0)

        rec = {
            "id": f"{g['plant_id']}_{g['gen_id']}",
            "plant": g["plant_name"],
            "owner": g["entity"],
            "sector": g["sector"],
            "tech": g["technology"],
            "county": g["county"],
            "type": g["unit_type"],
            "fuel": fuel_cat,
            "pm": g["prime_mover"],
            "mw": g["capacity_mw"],
            "age": g["age"],
            "year": g["online_year"],
            "cf": g.get("capacity_factor", 0),
            "gen_mwh": g.get("annual_gen_mwh", 0),
        }

        if is_thermal:
            rec["hr"] = hr
            rec["hr_src"] = g.get("heat_rate_source", "default")
            rec["co2_rate"] = round(CO2_RATES.get(fuel_cat, 0.053) * hr, 3)
            rec["vom"] = VOM_DEFAULTS.get(g["unit_type"], 4.0)

            # CCGT heat rate bin
            if g["unit_type"] == "gas_ccgt":
                bin_tag, bin_label = get_ccgt_hr_bin(hr)
                rec["hr_bin"] = bin_tag
                rec["hr_bin_label"] = bin_label
        else:
            rec["vom"] = VOM_DEFAULTS.get(g["unit_type"], 0)

        if g.get("retirement"):
            rec["retirement"] = g["retirement"]

        generators.append(rec)

    # ── Summary stats by type ──
    by_type = defaultdict(lambda: {"count": 0, "mw": 0, "gen_mwh": 0})
    for g in generators:
        by_type[g["type"]]["count"] += 1
        by_type[g["type"]]["mw"] += g["mw"]
        by_type[g["type"]]["gen_mwh"] += g["gen_mwh"]

    # ── CCGT heat rate distribution ──
    ccgt_bins = defaultdict(lambda: {"count": 0, "mw": 0, "gen_mwh": 0, "avg_hr": 0, "hrs_sum": 0})
    for g in generators:
        if g.get("hr_bin"):
            b = ccgt_bins[g["hr_bin"]]
            b["count"] += 1
            b["mw"] += g["mw"]
            b["gen_mwh"] += g["gen_mwh"]
            b["hrs_sum"] += g.get("hr", 0) * g["mw"]  # MW-weighted

    for tag, b in ccgt_bins.items():
        if b["mw"] > 0:
            b["avg_hr"] = round(b["hrs_sum"] / b["mw"], 2)
        del b["hrs_sum"]
        b["mw"] = round(b["mw"])
        b["gen_mwh"] = round(b["gen_mwh"])
        # Find the label
        for hbin in CCGT_HR_BINS:
            if hbin["tag"] == tag:
                b["label"] = hbin["label"]
                b["hr_range"] = f"{hbin['min']}-{hbin['max']}"
                break

    # ── Owner portfolios ──
    owner_data = defaultdict(lambda: {
        "fossil_mw": 0, "nuclear_mw": 0, "wind_mw": 0, "solar_mw": 0,
        "battery_mw": 0, "hydro_mw": 0, "other_mw": 0, "total_mw": 0,
        "fossil_gen": 0, "nuclear_gen": 0, "renewable_gen": 0, "total_gen": 0,
        "units": [], "types": set(),
    })

    for g in generators:
        o = owner_data[g["owner"]]
        o["total_mw"] += g["mw"]
        o["total_gen"] += g["gen_mwh"]
        o["types"].add(g["type"])

        fuel = g["fuel"]
        if fuel in ("coal", "gas", "oil"):
            o["fossil_mw"] += g["mw"]
            o["fossil_gen"] += g["gen_mwh"]
        elif fuel == "nuclear":
            o["nuclear_mw"] += g["mw"]
            o["nuclear_gen"] += g["gen_mwh"]
        elif fuel in ("wind", "solar"):
            o[f"{fuel}_mw"] += g["mw"]
            o["renewable_gen"] += g["gen_mwh"]
        elif fuel == "battery":
            o["battery_mw"] += g["mw"]
        elif fuel == "hydro":
            o["hydro_mw"] += g["mw"]
            o["renewable_gen"] += g["gen_mwh"]
        else:
            o["other_mw"] += g["mw"]

        # Track unit-level for the owner (compact)
        unit_rec = {"type": g["type"], "mw": g["mw"], "plant": g["plant"]}
        if g.get("hr") and g["hr"] > 0:
            unit_rec["hr"] = g["hr"]
        if g.get("cf") and g["cf"] > 0:
            unit_rec["cf"] = g["cf"]
        o["units"].append(unit_rec)

    # Build top owners sorted by total MW
    top_owners = sorted(owner_data.items(), key=lambda x: -x[1]["total_mw"])[:30]
    owners = []
    for name, data in top_owners:
        owners.append({
            "name": name,
            "total_mw": round(data["total_mw"]),
            "fossil_mw": round(data["fossil_mw"]),
            "nuclear_mw": round(data["nuclear_mw"]),
            "wind_mw": round(data["wind_mw"]),
            "solar_mw": round(data["solar_mw"]),
            "battery_mw": round(data["battery_mw"]),
            "hydro_mw": round(data["hydro_mw"]),
            "total_gen_gwh": round(data["total_gen"] / 1000),
            "fossil_gen_gwh": round(data["fossil_gen"] / 1000),
            "nuclear_gen_gwh": round(data["nuclear_gen"] / 1000),
            "renewable_gen_gwh": round(data["renewable_gen"] / 1000),
            "types": sorted(list(data["types"])),
            "unit_count": len(data["units"]),
        })

    # ── Fleet totals ──
    total_fossil_mw = sum(g["mw"] for g in generators if g["fuel"] in ("coal", "gas", "oil"))
    total_nuclear_mw = sum(g["mw"] for g in generators if g["fuel"] == "nuclear")
    total_wind_mw = sum(g["mw"] for g in generators if g["fuel"] == "wind")
    total_solar_mw = sum(g["mw"] for g in generators if g["fuel"] == "solar")
    total_battery_mw = sum(g["mw"] for g in generators if g["fuel"] == "battery")
    total_hydro_mw = sum(g["mw"] for g in generators if g["fuel"] == "hydro")

    summary = {
        "total_generators": len(generators),
        "total_mw": round(sum(g["mw"] for g in generators)),
        "total_gen_twh": round(sum(g["gen_mwh"] for g in generators) / 1e6, 1),
        "fossil_mw": round(total_fossil_mw),
        "nuclear_mw": round(total_nuclear_mw),
        "wind_mw": round(total_wind_mw),
        "solar_mw": round(total_solar_mw),
        "battery_mw": round(total_battery_mw),
        "hydro_mw": round(total_hydro_mw),
        "by_type": {
            k: {"count": v["count"], "mw": round(v["mw"]),
                "gen_twh": round(v["gen_mwh"] / 1e6, 1)}
            for k, v in by_type.items()
        },
        "ccgt_hr_bins": dict(ccgt_bins),
    }

    return {
        "meta": {
            "iso": "ERCOT",
            "state": "TX",
            "data_year": 2024,
            "source": "EIA 860 + EIA 923",
            "generated": "2026-03-13",
        },
        "summary": summary,
        "owners": owners,
        "generators": generators,
    }


def main():
    print("Loading EIA 860 (all TX generators)...")
    gens = load_860()
    print(f"  {len(gens)} operating generators in TX")

    print("Loading EIA 923 (2024 generation/fuel)...")
    fuel_data = load_923()
    print(f"  {len(fuel_data)} plant/fuel/PM combos")

    print("Computing operational data (heat rates, CFs)...")
    compute_operational_data(gens, fuel_data)

    revealed = sum(1 for g in gens.values() if g.get("heat_rate_source") == "revealed")
    thermal = sum(1 for g in gens.values() if g["fuel_cat"] in ("coal", "gas", "oil"))
    print(f"  {revealed}/{thermal} thermal generators with revealed heat rates")

    print("Building output...")
    output = build_output(gens)
    s = output["summary"]
    print(f"  {s['total_generators']} ERCOT generators, {s['total_mw']} MW total")
    print(f"  Fossil: {s['fossil_mw']} MW | Nuclear: {s['nuclear_mw']} MW | "
          f"Wind: {s['wind_mw']} MW | Solar: {s['solar_mw']} MW | "
          f"Battery: {s['battery_mw']} MW")

    print("\n  By type:")
    for t, st in sorted(s["by_type"].items(), key=lambda x: -x[1]["mw"]):
        print(f"    {t}: {st['count']} units, {st['mw']} MW, {st['gen_twh']} TWh")

    print("\n  CCGT Heat Rate Bins:")
    for tag, b in s["ccgt_hr_bins"].items():
        print(f"    {b.get('label', tag)}: {b['count']} units, {b['mw']} MW, avg HR={b['avg_hr']}")

    print(f"\n  Top 10 Owners:")
    for o in output["owners"][:10]:
        parts = []
        if o["fossil_mw"]: parts.append(f"fossil={o['fossil_mw']}")
        if o["nuclear_mw"]: parts.append(f"nuc={o['nuclear_mw']}")
        if o["wind_mw"]: parts.append(f"wind={o['wind_mw']}")
        if o["solar_mw"]: parts.append(f"solar={o['solar_mw']}")
        if o["battery_mw"]: parts.append(f"batt={o['battery_mw']}")
        print(f"    {o['name']}: {o['total_mw']} MW ({', '.join(parts)})")

    # Write as JS module
    os.makedirs(OUT.parent, exist_ok=True)
    js_content = (
        "// TX ERCOT Full Fleet — EIA 860 + 923 (2024)\n"
        "// Auto-generated by build_tx_fleet_json.py\n"
        f"const TX_FLEET = {json.dumps(output, indent=None, separators=(',', ':'))};\n"
    )

    with open(OUT, "w") as f:
        f.write(js_content)
    print(f"\nWrote {OUT} ({os.path.getsize(OUT) / 1024:.0f} KB)")

    # Debug JSON
    debug_dir = ROOT / "data" / "results"
    os.makedirs(debug_dir, exist_ok=True)
    debug_path = debug_dir / "tx_fleet_debug.json"
    with open(debug_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote debug JSON: {debug_path}")


if __name__ == "__main__":
    main()
