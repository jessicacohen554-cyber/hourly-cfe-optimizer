#!/usr/bin/env python3
"""
Generate constellation_scenarios.json from CEG_fleet_rosetta.csv.

Parses the full Constellation fleet CSV, extracts all fossil fuel plants,
maps fuel/plant types to standardized categories, and produces the scenario JSON.
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
CAPACITY_DEFAULTS = {'gas_ccgt': 800, 'gas_ct': 200, 'oil_ct': 150, 'gas_oil_ct': 200}

ISO_MAP = {
    'New England': 'NEISO',
    'New York': 'NYISO',
    'Alberta': 'NA',
    'Ontario': 'NA',
}


def normalize_iso(raw_iso):
    """Normalize ISO name from CSV."""
    raw = raw_iso.strip()
    # Handle comma-separated ISOs - take the first
    if ',' in raw:
        raw = raw.split(',')[0].strip()
    # Check explicit map
    if raw in ISO_MAP:
        return ISO_MAP[raw]
    # Handle Yucatan with special character
    if 'Yucat' in raw:
        return 'NA'
    # Already standard (CAISO, ERCOT, PJM, MISO, SPP, NA, etc.)
    return raw


def classify_fuel_type(fuel, plant_type):
    """Map CSV fuel type + plant type to our standardized fuel_type."""
    fuel = fuel.strip().lower()
    pt = plant_type.strip().lower()

    if fuel == 'gas/oil':
        return 'gas_oil_ct'
    if fuel == 'oil':
        return 'oil_ct'
    if fuel == 'gas':
        if 'combined cycle' in pt:
            return 'gas_ccgt'
        else:
            # Combustion Turbine, Gas Turbine, boiler, etc.
            return 'gas_ct'
    return None  # Not a fossil fuel type we handle


def stable_hash_id(name, state, unit_detail):
    """Generate a stable 6-digit hash-based ORISPL for plants without CAMPD IDs."""
    key = f"{name}|{state}|{unit_detail}"
    h = hashlib.sha256(key.encode()).hexdigest()
    # Use first 6 hex digits -> decimal, ensure 6 digits (900000-999999 range)
    return 900000 + (int(h[:6], 16) % 100000)


def main():
    plants = []
    seen_keys = set()  # Track to deduplicate rows with same name+ISO+fuel_type

    with open(CSV_PATH, 'r', encoding='latin-1') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stat_type = row.get('Stat Type', '').strip()
            if stat_type != 'Fossil Generation':
                continue

            fuel = row.get('Fuel Type', '').strip()
            plant_type = row.get('Plant Type', '').strip()
            ft = classify_fuel_type(fuel, plant_type)
            if ft is None:
                continue

            # Use SP Name (Persefoni name) as canonical name
            name = row.get('SP Name', '').strip()
            if not name:
                name = row.get('Persefoni Organization Name', '').strip()

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

            # CCS eligible
            ccs_raw = row.get('CCS Eligible?', '').strip()
            ccs_eligible = ccs_raw.lower() in ('yes', '1', 'true')

            # Each CSV row is a distinct unit/plant entry — keep them all.
            # Disambiguate rows with same name+ISO+fuel_type by appending unit detail.
            unit_detail = row.get('Unit Detail', '').strip()
            dedup_key = (name, iso, ft, unit_detail)
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            plant = {
                'orispl': orispl,
                'name': name,
                'iso': iso,
                'capacity_mw': CAPACITY_DEFAULTS[ft],
                'fuel_type': ft,
                'heat_rate_mmbtu_mwh': HEAT_RATES[ft],
                'co2_rate_t_mwh': CO2_RATES[ft],
                'equity_share': equity_share,
                'ccs_eligible': ccs_eligible,
                'status': 'operating'
            }
            plants.append(plant)

    # Sort: CCGTs first (by name), then CTs, then dual, then oil
    type_order = {'gas_ccgt': 0, 'gas_ct': 1, 'gas_oil_ct': 2, 'oil_ct': 3}
    plants.sort(key=lambda p: (type_order.get(p['fuel_type'], 99), p['name']))

    print(f"Total fossil plants extracted: {len(plants)}")
    by_type = {}
    for p in plants:
        by_type.setdefault(p['fuel_type'], []).append(p)
    for ft, ps in sorted(by_type.items()):
        print(f"  {ft}: {len(ps)}")
    by_iso = {}
    for p in plants:
        by_iso.setdefault(p['iso'], []).append(p)
    for iso, ps in sorted(by_iso.items()):
        print(f"  {iso}: {len(ps)}")

    # Identify top 6 CCS-eligible CCGTs for scenario modifications
    ccgt_ccs = [p for p in plants if p['fuel_type'] == 'gas_ccgt' and p['ccs_eligible']]
    top6 = ccgt_ccs[:6] if len(ccgt_ccs) >= 6 else ccgt_ccs

    # Build output JSON
    output = {
        "$schema_version": "2.1",
        "$schema_description": "Fleet scenario schema \u2014 extends Phase 2.1 with named scenario variants and new-build interaction rules. Each scenario applies modifications to the base_fleet: CCS retrofits, retirements, or new plant additions. Fleet-level add_plant actions interact with grid-level new fossil builds per the new_build_interaction rule.",
        "metadata": {
            "company": "Constellation Energy",
            "as_of_date": "2025-01-01",
            "source": "CEG_fleet_rosetta.csv",
            "notes": f"Fossil fleet of {len(plants)} plants across multiple ISOs with 4 scenario variants for fleet-level emissions modeling. Plants include CEG legacy fleet and CPN (Calpine) acquisition."
        },
        "new_build_interaction": {
            "rule": "additive",
            "description": "Fleet-level add_plant actions are additive to grid-level new fossil builds from apply_economic_new_build(). Fleet additions represent company-specific decisions; grid builds represent market-driven capacity. Both can coexist.",
            "reporting": "Both sources reported separately in output: grid_new_fossil_mw (market) + fleet_new_fossil_mw (company) = total_new_fossil_mw"
        },
        "schema": {
            "base_fleet": {
                "description": "Array of Phase 2.1 plant objects. All plants start in 'operating' status. This is the default fleet state before any scenario modifications.",
                "items": "See Phase 2.1 schema (sample_fleet.json) for per-plant fields."
            },
            "scenarios": {
                "description": "Named scenario variants. Each scenario applies an ordered list of modifications to a copy of the base_fleet.",
                "value_schema": {
                    "description": {
                        "type": "string",
                        "required": True,
                        "description": "Human-readable scenario description."
                    },
                    "modifications": {
                        "type": "array",
                        "required": True,
                        "description": "Ordered list of fleet modifications. Applied sequentially to a copy of base_fleet.",
                        "item_types": {
                            "ccs_retrofit": {
                                "description": "Retrofit an existing plant with CCS. Targeted by orispl (single plant) or fuel_type (all plants of that type).",
                                "fields": {
                                    "action": "ccs_retrofit",
                                    "orispl": {"type": "integer", "required": False, "description": "Target a specific plant by ORIS ID."},
                                    "fuel_type": {"type": "string", "required": False, "description": "Target all plants of this fuel type. Mutually exclusive with orispl."},
                                    "year_online": {"type": "integer", "required": True, "description": "Year CCS becomes operational."},
                                    "ccs_capture_rate": {"type": "number", "default": 0.95, "description": "CO2 capture rate (0-1)."},
                                    "ccs_heat_rate_penalty": {"type": "number", "default": 1.14, "description": "Heat rate multiplier for CCS parasitic load."}
                                }
                            },
                            "retire": {
                                "description": "Retire a plant or all plants of a fuel type. Zero generation and emissions from year_online onward.",
                                "fields": {
                                    "action": "retire",
                                    "orispl": {"type": "integer", "required": False, "description": "Target a specific plant."},
                                    "fuel_type": {"type": "string", "required": False, "description": "Target all plants of this fuel type."},
                                    "year_online": {"type": "integer", "required": True, "description": "Year retirement takes effect."}
                                }
                            },
                            "add_plant": {
                                "description": "Add a new plant to the fleet (e.g., new-build CCGT).",
                                "fields": {
                                    "action": "add_plant",
                                    "name": {"type": "string", "required": True},
                                    "iso": {"type": "string", "required": True},
                                    "capacity_mw": {"type": "number", "required": True},
                                    "fuel_type": {"type": "string", "required": True},
                                    "heat_rate_mmbtu_mwh": {"type": "number", "required": True},
                                    "co2_rate_t_mwh": {"type": "number", "required": False, "description": "Derived from heat_rate x emission_factor if omitted."},
                                    "equity_share": {"type": "number", "default": 1.0},
                                    "year_online": {"type": "integer", "required": True, "description": "Year the plant enters service."}
                                }
                            }
                        }
                    }
                }
            }
        },
        "baseline_mt": 23.16,
        "targets": {
            "sbti_15c": {
                "type": "sbti_15c",
                "description": "SBTi 1.5\u00b0C pathway: net zero by 2040.",
                "base_year": 2023,
                "baseline_mt": 23.16,
                "milestones": {
                    "2023": 23.16,
                    "2030": 11.6,
                    "2035": 5.8,
                    "2040": 0.0,
                    "2045": 0.0,
                    "2050": 0.0
                }
            },
            "at_power_nz": {
                "type": "at_power_nz",
                "description": "Constellation committed target: AT Power net-zero pathway.",
                "milestones": {
                    "2023": 23.16,
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
                "description": "Status quo \u2014 no CCS, no new gas, no early retirements. Economic retirements only (driven by market conditions in each sweep scenario).",
                "modifications": []
            },
            "ccs_top_emitters": {
                "description": "CCS retrofit on the top 6 largest CCS-eligible CCGTs by 2030-2032.",
                "modifications": [
                    {"orispl": top6[0]['orispl'], "action": "ccs_retrofit", "year_online": 2030},
                    {"orispl": top6[1]['orispl'], "action": "ccs_retrofit", "year_online": 2030},
                    {"orispl": top6[2]['orispl'], "action": "ccs_retrofit", "year_online": 2031},
                    {"orispl": top6[3]['orispl'], "action": "ccs_retrofit", "year_online": 2031},
                    {"orispl": top6[4]['orispl'], "action": "ccs_retrofit", "year_online": 2032},
                    {"orispl": top6[5]['orispl'], "action": "ccs_retrofit", "year_online": 2032},
                ]
            },
            "ccs_plus_new_gas": {
                "description": "CCS on top emitters + 1,200 MW new efficient CCGT in PJM to replace aging peakers. Fleet-level gas additions are additive to grid-level new fossil builds.",
                "modifications": [
                    {"orispl": top6[0]['orispl'], "action": "ccs_retrofit", "year_online": 2030},
                    {"orispl": top6[1]['orispl'], "action": "ccs_retrofit", "year_online": 2030},
                    {"orispl": top6[2]['orispl'], "action": "ccs_retrofit", "year_online": 2031},
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
                "description": "Retire all oil CTs, gas/oil CTs, and gas CTs by 2030, CCS retrofit on all remaining gas CCGTs by 2035. Aggressive decarbonization \u2014 eliminates peaker fleet entirely and captures residual CCGT emissions.",
                "modifications": [
                    {"fuel_type": "oil_ct", "action": "retire", "year_online": 2030},
                    {"fuel_type": "gas_oil_ct", "action": "retire", "year_online": 2030},
                    {"fuel_type": "gas_ct", "action": "retire", "year_online": 2030},
                    {"fuel_type": "gas_ccgt", "action": "ccs_retrofit", "year_online": 2035}
                ]
            }
        }
    }

    with open(OUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {OUT_PATH}")
    print(f"base_fleet entries: {len(output['base_fleet'])}")


if __name__ == '__main__':
    main()
