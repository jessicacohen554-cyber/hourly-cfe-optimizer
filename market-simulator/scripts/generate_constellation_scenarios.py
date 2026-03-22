#!/usr/bin/env python3
"""
Generate constellation_scenarios.json from CEG_fleet_rosetta.csv.

Parses the FULL Constellation fleet CSV (all plant types — nuclear, fossil,
renewable, geothermal, storage), maps fuel/plant types to standardized
categories, and produces the scenario JSON with base_fleet + scenario variants.

Single source of truth: market-simulator/data/CEG_fleet_rosetta.csv
"""

import csv
import json
import hashlib
import os

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'CEG_fleet_rosetta.csv')
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fleet_scenarios', 'constellation_scenarios.json')

# --- Constants ---
HEAT_RATES = {'gas_ccgt': 7.0, 'gas_ct': 10.5, 'oil_ct': 10.5, 'gas_oil_ct': 10.5}
CO2_RATES = {'gas_ccgt': 0.37, 'gas_ct': 0.55, 'oil_ct': 0.65, 'gas_oil_ct': 0.58}

# Capacity defaults by fuel type (MW, nameplate estimate per unit)
CAPACITY_DEFAULTS = {
    'gas_ccgt': 800, 'gas_ct': 200, 'oil_ct': 150, 'gas_oil_ct': 200,
    'nuclear': 2000, 'geothermal': 250, 'wind': 150, 'solar': 100,
    'hydro': 500, 'battery': 100, 'pumped_storage': 800,
}

# Generation estimates (TWh, equity-weighted, 2023 baseline)
# Based on EIA/eGRID cross-reference + user guidance
GENERATION_ESTIMATES_TWH = {
    'nuclear': 190.0,       # 12 stations, equity shares 43-100%, ~90% CF
    'geothermal': 9.0,      # Geysers dominant, ~1.1 GW nameplate
    'wind': 3.5,            # 38 wind farms, mostly 51% equity
    'solar': 0.8,           # 6 solar plants, 51-100% equity
    'hydro': 0.5,           # Conowingo
    'gas_ccgt': 50.0,       # ~13 GW nameplate, ~44% CF equity-weighted
    'gas_ct': 1.5,          # Peakers, low CF
    'oil_ct': 0.3,          # Oil peakers, very low CF
    'gas_oil_ct': 0.8,      # Dual-fuel peakers
    'battery': 0.0,         # Storage, net zero gen
    'pumped_storage': 0.0,  # Pumped storage, net zero gen
}

# Plant category for frontend grouping
CATEGORY_MAP = {
    'nuclear': 'nuclear',
    'geothermal': 'renewable',
    'wind': 'renewable',
    'solar': 'renewable',
    'hydro': 'renewable',
    'battery': 'storage',
    'pumped_storage': 'storage',
    'gas_ccgt': 'fossil',
    'gas_ct': 'fossil',
    'oil_ct': 'fossil',
    'gas_oil_ct': 'fossil',
}

ISO_MAP = {
    'New England': 'NEISO',
    'New York': 'NYISO',
    'Alberta': 'NA',
    'Ontario': 'NA',
}


def normalize_iso(raw_iso):
    """Normalize ISO name from CSV."""
    raw = raw_iso.strip()
    if ',' in raw:
        raw = raw.split(',')[0].strip()
    if raw in ISO_MAP:
        return ISO_MAP[raw]
    if 'Yucat' in raw:
        return 'NA'
    return raw


def classify_fuel_type(fuel_csv, plant_type_csv, stat_type):
    """Map CSV fuel type + plant type + stat type to standardized fuel_type."""
    fuel = fuel_csv.strip()
    pt = plant_type_csv.strip().lower()
    stat = stat_type.strip()

    # Non-fossil types
    if fuel == 'Nuclear':
        return 'nuclear'
    if fuel == 'Geothermal':
        return 'geothermal'
    if fuel == 'Wind':
        return 'wind'
    if fuel == 'Solar':
        return 'solar'
    if fuel == 'Water':
        if 'pumped' in pt:
            return 'pumped_storage'
        return 'hydro'
    if fuel == 'Battery':
        return 'battery'

    # Storage from stat type
    if stat == 'Storage':
        return 'battery'

    # Fossil fuel types
    fuel_lower = fuel.lower()
    if fuel_lower == 'gas/oil':
        return 'gas_oil_ct'
    if fuel_lower == 'oil':
        return 'oil_ct'
    if fuel_lower == 'gas':
        if 'combined cycle' in pt:
            return 'gas_ccgt'
        else:
            return 'gas_ct'

    return None  # Unrecognized


def stable_hash_id(name, state, unit_detail):
    """Generate a stable 6-digit hash-based ORISPL for plants without CAMPD IDs."""
    key = f"{name}|{state}|{unit_detail}"
    h = hashlib.sha256(key.encode()).hexdigest()
    return 900000 + (int(h[:6], 16) % 100000)


def main():
    plants = []
    seen_keys = set()

    with open(CSV_PATH, 'r', encoding='latin-1') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Read plant status from Rosetta (Operating / Retired)
            row_status = row.get('Status', '').strip().lower()

            stat_type = row.get('Stat Type', '').strip()
            fuel_csv = row.get('Fuel Type', '').strip()
            plant_type_csv = row.get('Plant Type', '').strip()

            ft = classify_fuel_type(fuel_csv, plant_type_csv, stat_type)
            if ft is None:
                continue

            # Use SP Name (Persefoni name) as canonical name
            name = row.get('SP Name', '').strip()
            if not name:
                name = row.get('Persefoni Organization Name', '').strip()
            short_name = row.get('Persefoni Organization Name', '').strip()

            iso = normalize_iso(row.get('ISO', ''))
            state = row.get('State', '').strip()

            # CAMPD Facility ID
            campd_id = row.get('CAMPD Facility ID', '').strip()
            if campd_id and campd_id != 'N/A' and campd_id != '':
                try:
                    orispl = int(campd_id)
                except ValueError:
                    orispl = stable_hash_id(name, state, row.get('Unit Detail', ''))
            else:
                orispl = stable_hash_id(name, state, row.get('Unit Detail', ''))

            # Equity share
            eq_raw = row.get('Equity Share %', '100%').strip().replace('%', '')
            try:
                equity_share = round(float(eq_raw) / 100.0, 2)
            except ValueError:
                equity_share = 1.0

            # CCS eligible: set in the sidebar UI by the user, default false
            ccs_eligible = False

            # Year built
            year_str = row.get('First in Service Year', '').strip()
            try:
                year_built = int(year_str) if year_str else None
            except ValueError:
                year_built = None

            # Dedup
            unit_detail = row.get('Unit Detail', '').strip()
            dedup_key = (name, iso, ft, unit_detail)
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            category = CATEGORY_MAP.get(ft, 'other')

            # Use actual capacity from CSV, fall back to defaults only if missing
            cap_raw = row.get('Constellation Owned Capacity (MW)', '').strip()
            try:
                capacity_mw = float(cap_raw) if cap_raw else CAPACITY_DEFAULTS.get(ft, 200)
            except ValueError:
                capacity_mw = CAPACITY_DEFAULTS.get(ft, 200)

            # Use actual CCS capacity from CSV if available
            ccs_cap_raw = row.get('Available CCS Capacity', '').strip()
            try:
                ccs_capacity_mw = float(ccs_cap_raw) if ccs_cap_raw else None
            except ValueError:
                ccs_capacity_mw = None

            # Set status from Rosetta
            plant_status = 'retired' if row_status == 'retired' else 'operating'

            plant = {
                'orispl': orispl,
                'name': short_name or name,
                'full_name': name,
                'iso': iso,
                'state': state,
                'capacity_mw': capacity_mw,
                'fuel_type': ft,
                'plant_category': category,
                'equity_share': equity_share,
                'status': plant_status,
            }

            # Add fossil-specific fields
            if category == 'fossil':
                plant['heat_rate_mmbtu_mwh'] = HEAT_RATES.get(ft, 10.0)
                plant['co2_rate_t_mwh'] = CO2_RATES.get(ft, 0.37)
                if ccs_capacity_mw is not None:
                    plant['ccs_capacity_mw'] = ccs_capacity_mw

            # Add year built if available
            if year_built:
                plant['year_built'] = year_built

            plants.append(plant)

    # Sort: nuclear first, then renewable/geothermal, then fossil by type, then storage
    cat_order = {'nuclear': 0, 'renewable': 1, 'fossil': 2, 'storage': 3}
    type_order = {'nuclear': 0, 'geothermal': 1, 'wind': 2, 'solar': 3, 'hydro': 4,
                  'gas_ccgt': 5, 'gas_ct': 6, 'gas_oil_ct': 7, 'oil_ct': 8,
                  'battery': 9, 'pumped_storage': 10}
    plants.sort(key=lambda p: (cat_order.get(p['plant_category'], 99),
                                type_order.get(p['fuel_type'], 99),
                                p['name']))

    print(f"Total plants extracted: {len(plants)}")
    by_category = {}
    for p in plants:
        by_category.setdefault(p['plant_category'], []).append(p)
    for cat, ps in sorted(by_category.items()):
        print(f"  {cat}: {len(ps)} plants")
        by_type = {}
        for p in ps:
            by_type.setdefault(p['fuel_type'], []).append(p)
        for ft, fps in sorted(by_type.items()):
            print(f"    {ft}: {len(fps)}")

    by_iso = {}
    for p in plants:
        by_iso.setdefault(p['iso'], []).append(p)
    for iso, ps in sorted(by_iso.items()):
        print(f"  {iso}: {len(ps)}")

    # Hardcoded top CCS-eligible CCGT orispls for scenario definitions
    # CCS eligibility is managed by the user in the sidebar UI, not derived from data
    TOP_CCS_ORISPLS = [997153, 55327, 55327, 50292, 55172, 55299]

    # Build output JSON
    output = {
        "$schema_version": "2.2",
        "$schema_description": "Full fleet scenario schema — includes all plant types (nuclear, fossil, renewable, geothermal, storage). Scenarios apply modifications only to fossil plants.",
        "metadata": {
            "company": "Constellation Energy",
            "as_of_date": "2025-01-01",
            "source": "CEG_fleet_rosetta.csv",
            "notes": f"Full fleet of {len(plants)} plants across all fuel types. Includes CEG legacy fleet and CPN (Calpine) acquisition. Nuclear ~190 TWh, Geothermal ~9 TWh, Wind/Solar/Hydro ~4-5 TWh.",
            "generation_estimates_twh": GENERATION_ESTIMATES_TWH,
        },
        "new_build_interaction": {
            "rule": "additive",
            "description": "Fleet-level add_plant actions are additive to grid-level new fossil builds.",
        },
        "baseline_mt": 23.16,
        "targets": {
            "sbti_15": {
                "type": "sbti_15c",
                "label": "SBTi 1.5\u00b0C (Power Sector v2)",
                "description": "SBTi Power Sector v2 draft guidance: net zero by 2040 for power sector companies.",
                "base_year": 2023,
                "baseline_mt": 23.16,
                "milestones": {
                    "2023": 23.16,
                    "2025": 19.7,
                    "2030": 11.6,
                    "2035": 5.8,
                    "2040": 0.0,
                    "2045": 0.0,
                    "2050": 0.0
                }
            },
            "at_power_nz": {
                "type": "at_power_nz",
                "label": "AT Power NZ",
                "description": "Constellation committed target: 95% reduction by 2040, net zero by 2045.",
                "milestones": {
                    "2023": 23.16,
                    "2025": 20.4,
                    "2030": 15.1,
                    "2035": 8.1,
                    "2040": 1.2,
                    "2045": 0.0,
                    "2050": 0.0
                }
            }
        },
        "base_fleet": plants,
        "scenarios": {
            "baseline": {
                "description": "Status quo \u2014 no CCS, no new gas, no early retirements. Economic retirements only.",
                "modifications": []
            },
            "ccs_top_emitters": {
                "description": "CCS retrofit on the top 6 largest CCS-eligible CCGTs by 2030-2032.",
                "modifications": [
                    {"orispl": TOP_CCS_ORISPLS[i], "action": "ccs_retrofit", "year_online": 2030 + (i // 2)}
                    for i in range(len(TOP_CCS_ORISPLS))
                ]
            },
            "ccs_plus_new_gas": {
                "description": "CCS on top emitters + 1,200 MW new efficient CCGT in PJM.",
                "modifications": [
                    {"orispl": TOP_CCS_ORISPLS[0], "action": "ccs_retrofit", "year_online": 2030},
                    {"orispl": TOP_CCS_ORISPLS[1], "action": "ccs_retrofit", "year_online": 2030},
                    {"orispl": TOP_CCS_ORISPLS[2], "action": "ccs_retrofit", "year_online": 2031},
                    {
                        "action": "add_plant",
                        "name": "New PJM CCGT East",
                        "iso": "PJM",
                        "capacity_mw": 600,
                        "fuel_type": "gas_ccgt",
                        "heat_rate_mmbtu_mwh": 6.3,
                        "equity_share": 1.0,
                        "year_online": 2028
                    },
                    {
                        "action": "add_plant",
                        "name": "New PJM CCGT West",
                        "iso": "PJM",
                        "capacity_mw": 600,
                        "fuel_type": "gas_ccgt",
                        "heat_rate_mmbtu_mwh": 6.3,
                        "equity_share": 1.0,
                        "year_online": 2029
                    }
                ]
            },
            "retire_peakers_ccs_baseload": {
                "description": "Retire all oil CTs, gas/oil CTs, and gas CTs by 2030, CCS retrofit on all remaining gas CCGTs by 2035.",
                "modifications": [
                    {"fuel_type": "oil_ct", "action": "retire", "year_online": 2030},
                    {"fuel_type": "gas_oil_ct", "action": "retire", "year_online": 2030},
                    {"fuel_type": "gas_ct", "action": "retire", "year_online": 2030},
                    {"fuel_type": "gas_ccgt", "action": "ccs_retrofit", "year_online": 2035}
                ]
            }
        }
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {OUT_PATH}")
    print(f"base_fleet entries: {len(output['base_fleet'])}")


if __name__ == '__main__':
    main()
