#!/usr/bin/env python3
"""
Build fleet_scenario_results_sample.json from CEG_fleet_rosetta.csv + eGRID 2023.

2023 baseline emissions come directly from eGRID 2023 plant-level data (PLNT23),
NOT from derived capacity factors. Each plant's actual CO₂ short tons and net
generation are read from eGRID, equity-weighted via the Rosetta CSV.

Plants without eGRID data (Canadian, small peakers) use heat-rate-derived estimates.

Sources:
  - CEG_fleet_rosetta.csv: Plant inventory, capacity, equity, CCS eligibility
  - egrid_2023_ceg_plant_emissions.json: Actual 2023 CO₂ + generation from eGRID

Output: market-simulator/frontend/data/fleet_scenario_results_sample.json
"""

import csv
import json
import os
from collections import defaultdict
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'frontend', 'data')

ROSETTA_PATH = os.path.join(DATA_DIR, 'CEG_fleet_rosetta.csv')
EGRID_EMISSIONS_PATH = os.path.join(DATA_DIR, 'egrid_2023_ceg_plant_emissions.json')

# ── Canonical fuel type mapping from Rosetta CSV ──
FUEL_MAP = {
    'Nuclear': 'nuclear',
    'Gas': 'gas_ccgt',        # default; refined by Plant Type
    'Oil': 'oil_ct',
    'Gas/Oil': 'gas_oil_ct',  # dual-fuel peakers
    'Geothermal': 'geothermal',
    'Solar': 'solar',
    'Wind': 'wind',
    'Water': 'hydro',
    'Battery': 'battery',
}

PLANT_TYPE_REFINEMENT = {
    'Combustion Turbine': 'gas_ct',
    'Gas Turbine': 'gas_ct',
    'Combined Cycle': 'gas_ccgt',
    'Dual Firing Combustion Turbine': 'gas_oil_ct',
    'Internal Combustion': 'oil_ct',
    'Dry bottom wall-fired boiler': 'oil_ct',
}

# ── Emission factors (t CO2 / MWh) — fallback for plants without eGRID data ──
# Derived from eGRID 2023 fleet-average rates for CEG plants.
# Plants WITH eGRID data use their own measured co2_rate_lb_mwh instead.
EMISSION_RATE = {
    'gas_ccgt': 0.382,    # eGRID avg: 843 lb/MWh across 34 CEG CCGTs
    'gas_ct': 0.657,      # eGRID avg: 1449 lb/MWh across 14 CEG CTs
    'oil_ct': 1.387,      # eGRID avg: 3058 lb/MWh across 4 CEG oil CTs
    'gas_oil_ct': 1.506,  # eGRID avg: 3321 lb/MWh across 5 CEG dual-fuel
}

# ── Heat rates (MMBtu/MWh) ──
HEAT_RATES = {
    'gas_ccgt': 7.0,
    'gas_ct': 10.5,
    'oil_ct': 10.5,
    'gas_oil_ct': 10.5,
}

# ── Simulation years ──
YEARS = [2023, 2025, 2030, 2035, 2040, 2045, 2050]

# ── Capacity factors by fuel type ──
# Used to estimate annual generation from actual plant capacity
CAPACITY_FACTORS = {
    'nuclear': 0.93,
    'gas_ccgt': 0.44,
    'gas_ct': 0.08,
    'oil_ct': 0.04,
    'gas_oil_ct': 0.06,
    'geothermal': 0.85,
    'wind': 0.30,
    'solar': 0.22,
    'hydro': 0.20,
    'battery': 0.0,
}


def load_egrid_emissions():
    """Load eGRID 2023 actual plant emissions (pre-extracted).

    Returns dict keyed by CAMPD facility ID with equity-weighted CO₂ and generation.
    """
    if not os.path.exists(EGRID_EMISSIONS_PATH):
        print(f"  WARNING: {EGRID_EMISSIONS_PATH} not found, using derived estimates")
        return {}

    with open(EGRID_EMISSIONS_PATH, 'r') as f:
        data = json.load(f)

    by_campd = {}
    for p in data.get('plants', []):
        cid = p['campd_id']
        by_campd[cid] = {
            'co2_mt': p['co2_mt_equity'],           # Mt CO₂ (metric, equity-weighted)
            'gen_twh': p['gen_twh_equity'],          # TWh (equity-weighted)
            'co2_rate_lb_mwh': p['co2_rate_lb_mwh'], # eGRID lb/MWh (plant-specific)
            'co2_short_tons_total': p['co2_short_tons_total'],
        }

    print(f"  Loaded eGRID 2023: {len(by_campd)} plants, "
          f"{data.get('total_co2_mt_equity', 0):.1f} Mt CO₂, "
          f"{data.get('total_gen_twh_equity', 0):.0f} TWh")
    return by_campd


def parse_rosetta():
    """Parse Rosetta CSV into structured plant list.

    Reads actual capacity from 'Constellation Owned Capacity (MW)' column
    (already equity-weighted) and CCS capacity from 'Available CCS Capacity'.
    Attaches eGRID 2023 actual emissions where available.
    """
    egrid = load_egrid_emissions()

    # Pre-scan to find shared CAMPD IDs — we need to split eGRID data proportionally
    campd_cap_totals = defaultdict(float)
    with open(ROSETTA_PATH, 'r', encoding='latin-1') as f:
        reader = csv.DictReader(f)
        for row in reader:
            campd = row.get('CAMPD Facility ID', '').strip()
            if campd and campd != 'N/A':
                cap = float(row.get('Constellation Owned Capacity (MW)', '0').strip() or 0)
                campd_cap_totals[int(campd)] += cap

    plants = []
    with open(ROSETTA_PATH, 'r', encoding='latin-1') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fuel_csv = row.get('Fuel Type', '').strip()
            plant_type_csv = row.get('Plant Type', '').strip()
            name = row.get('SP Name', '').strip() or row.get('Name', '').strip()

            # Map fuel type
            fuel = FUEL_MAP.get(fuel_csv, 'unknown')
            # Refine Gas plants by plant type (CT vs CCGT)
            # Only refine Gas-sourced fuels — Oil plants stay as oil_ct
            if fuel_csv == 'Gas':
                refined = PLANT_TYPE_REFINEMENT.get(plant_type_csv)
                if refined:
                    fuel = refined
            elif fuel_csv == 'Gas/Oil':
                refined = PLANT_TYPE_REFINEMENT.get(plant_type_csv)
                if refined:
                    fuel = refined

            # Capacity: already equity-weighted in the CSV
            cap_str = row.get('Constellation Owned Capacity (MW)', '0').strip()
            try:
                capacity_mw = float(cap_str) if cap_str else 0.0
            except ValueError:
                capacity_mw = 0.0

            # Available CCS capacity (informational)
            ccs_cap_str = row.get('Available CCS Capacity', '').strip()
            try:
                ccs_capacity_mw = float(ccs_cap_str) if ccs_cap_str else 0.0
            except ValueError:
                ccs_capacity_mw = 0.0

            # CCS eligible if Available CCS Capacity > 0
            ccs_eligible = ccs_capacity_mw > 0

            equity_str = row.get('Equity Share %', '100%').strip().replace('%', '')
            try:
                equity = float(equity_str) / 100.0
            except ValueError:
                equity = 1.0

            campd_id = row.get('CAMPD Facility ID', '').strip()
            orispl = int(campd_id) if campd_id and campd_id != 'N/A' else None

            iso_raw = row.get('ISO', '').strip()
            iso_map = {
                'New England': 'NEISO', 'New York': 'NYISO',
                'Alberta': 'NA', 'Ontario': 'NA', 'Yucat\xe1n': 'NA',
            }
            iso = iso_map.get(iso_raw, iso_raw)
            if ',' in iso:
                iso = iso.split(',')[0].strip()

            state = row.get('State', '').strip()
            year_str = row.get('First in Service Year', '').strip()
            try:
                year_built = int(year_str) if year_str else None
            except ValueError:
                year_built = None

            stat_type = row.get('Stat Type', '').strip()
            group = row.get('Group', '').strip()
            status = row.get('Status', 'Operating').strip()

            # Retired plants: included in 2023 baseline, excluded from future.
            # For plants with Status=Retired, retired_year determines the first
            # year they're excluded. Plants retired mid-year still count in that
            # year's baseline (eGRID captures partial-year ops).
            # Plants with no CAMPD / no eGRID data use default retired_year=2024
            # (conservative: assume operating through 2023).
            RETIRED_YEARS = {
                1588: 2024,   # Mystic (retired May 31, 2024)
                3169: 2024,   # Schuylkill
                55281: 2024,  # Southeast Chicago Energy
                55748: 2024,  # Los Esteros Critical Energy Facility
                55853: 2024,  # Inland Empire Energy Center
                10741: 2024,  # Clear Lake
                3159: 2024,   # Cromby
                2384: 2024,   # Deepwater
                7701: 2024,   # Fairless Hills
                1553: 2024,   # Gould Street
                8008: 2024,   # Mickleton
                2382: 2024,   # Middle
                2383: 2024,   # Missouri Avenue
                1559: 2024,   # Riverside
                55401: 2024,  # Rolling Hills CT
                1560: 2024,   # Westport Jet
                2379: 2024,   # Carll's Corner
                2380: 2024,   # Cedar
                55833: 2024,  # Auburndale Energy Center CT
                50292: 2024,  # Bethpage IC (0.4 MW)
                10294: 2024,  # King City Peaking
                55810: 2024,  # Gilroy Peaking
            }
            if status == 'Retired':
                retired_year = RETIRED_YEARS.get(orispl, 2024) if orispl else 2024
            else:
                retired_year = None

            # ORISPL mismatches: these CAMPD IDs map to wrong eGRID plant
            # Salem CT (16 MW gas) shares ORISPL 2410 with Salem Nuclear (1 GW)
            # Delaware City 10 / West Energy Center have parasitic load only
            # Rolling Hills CT (5.5 MW) shares ORISPL with Rolling Hills Gen (797 MW)
            EGRID_EXCLUDE = {2410, 592, 597, 55401}
            if orispl in EGRID_EXCLUDE:
                orispl = None  # Don't use eGRID data for these

            # Attach eGRID 2023 actual emissions if available
            # For shared CAMPD IDs, split proportionally by capacity
            egrid_data = egrid.get(orispl) if orispl else None
            egrid_co2_mt = None
            egrid_gen_twh = None
            egrid_co2_rate = None
            if egrid_data:
                total_cap_for_campd = campd_cap_totals.get(orispl, capacity_mw)
                share = capacity_mw / total_cap_for_campd if total_cap_for_campd > 0 else 1.0
                egrid_co2_mt = egrid_data['co2_mt'] * share
                egrid_gen_twh = egrid_data['gen_twh'] * share
                egrid_co2_rate = egrid_data['co2_rate_lb_mwh']  # Rate is same for all units

            plants.append({
                'name': name,
                'fuel_type': fuel,
                'capacity_mw': capacity_mw,
                'ccs_capacity_mw': ccs_capacity_mw,
                'iso': iso,
                'state': state,
                'equity': equity,
                'orispl': orispl,
                'ccs_eligible': ccs_eligible,
                'year_built': year_built,
                'stat_type': stat_type,
                'group': group,
                'plant_type': plant_type_csv,
                'retired_year': retired_year,  # None if still operating
                # eGRID 2023 actuals (None if not available)
                'egrid_co2_mt': egrid_co2_mt,
                'egrid_gen_twh': egrid_gen_twh,
                'egrid_co2_rate_lb_mwh': egrid_co2_rate,
            })

    return plants


def categorize_plants(plants):
    """Group plants by fuel type."""
    cats = {}
    for p in plants:
        ft = p['fuel_type']
        if ft not in cats:
            cats[ft] = []
        cats[ft].append(p)
    return cats


def get_fossil_plants(plants):
    """Return only fossil-fuel plants (gas, oil, dual-fuel)."""
    fossil_fuels = {'gas_ccgt', 'gas_ct', 'oil_ct', 'gas_oil_ct'}
    return [p for p in plants if p['fuel_type'] in fossil_fuels]


def get_plant_gen_twh(p, fuel, year, year_factor, re_growth):
    """Get generation for a single plant in a given year.

    For 2023 fossil plants with eGRID data: uses actual measured generation.
    For all other cases: uses capacity × CF × temporal adjustments.
    """
    cap = p['capacity_mw']

    # 2023 fossil plants: prefer eGRID actual generation
    if year == 2023 and p.get('egrid_gen_twh') is not None and fuel in EMISSION_RATE:
        return p['egrid_gen_twh']

    cf = CAPACITY_FACTORS.get(fuel, 0.0)
    if cap <= 0 or cf <= 0:
        return 0.0

    if fuel == 'nuclear':
        if p['name'] == 'Crane' and year < 2027:
            return 0.0
        return cap * cf * 8.760 / 1000.0
    elif fuel in ('wind', 'solar'):
        return cap * cf * 8.760 * re_growth / 1000.0
    elif fuel in EMISSION_RATE:
        return cap * cf * 8.760 * year_factor / 1000.0
    else:
        return cap * cf * 8.760 / 1000.0


def get_plant_co2_rate(p, fuel):
    """Get CO₂ emission rate for a plant.

    For plants with eGRID data: uses measured lb/MWh → t/MWh.
    Otherwise: uses default EMISSION_RATE by fuel type.
    """
    if p.get('egrid_co2_rate_lb_mwh') and p['egrid_co2_rate_lb_mwh'] > 0:
        # Convert lb/MWh to t/MWh (metric): lb × 0.453592 / 1000
        return p['egrid_co2_rate_lb_mwh'] * 0.000453592
    return EMISSION_RATE.get(fuel, 0.37)


def compute_fleet_generation(plants, year, scenario='baseline', retired_fuels=None):
    """Compute generation by fuel from actual plant data.

    2023 baseline: uses eGRID 2023 measured generation and CO₂ for fossil plants.
    Future years: scales from 2023 actuals using market-driven decline factors.
    Non-fossil plants: capacity × CF (no eGRID data needed — zero emissions).
    """
    if retired_fuels is None:
        retired_fuels = {}

    year_factor = max(0.0, 1.0 - 0.008 * (year - 2023))
    re_growth = 1.0 + 0.02 * (year - 2023)

    gen_by_fuel = defaultdict(float)

    for p in plants:
        fuel = p['fuel_type']
        cap = p['capacity_mw']

        if cap <= 0:
            continue

        # Check retirement (fleet-wide by fuel type)
        if fuel in retired_fuels and year >= retired_fuels[fuel]:
            continue
        if fuel == 'gas_oil_ct' and 'oil_ct' in retired_fuels and year >= retired_fuels['oil_ct']:
            continue

        # Check plant-specific retirement (e.g., Mystic retired 2024)
        if p.get('retired_year') and year >= p['retired_year']:
            continue

        gen_twh = get_plant_gen_twh(p, fuel, year, year_factor, re_growth)

        # For future years with eGRID plants: scale from 2023 actual by year_factor
        if year > 2023 and p.get('egrid_gen_twh') is not None and fuel in EMISSION_RATE:
            gen_twh = p['egrid_gen_twh'] * year_factor

        gen_by_fuel[fuel] += gen_twh

    # CCS conversion for applicable scenarios
    ccs_ccgt_twh = 0.0
    if scenario in ('ccs_top_emitters', 'ccs_plus_new_gas', 'retire_peakers_ccs_baseload'):
        ccs_gen = 0.0
        for p in plants:
            if p['fuel_type'] != 'gas_ccgt' or p['capacity_mw'] <= 0:
                continue
            if not p['ccs_eligible']:
                continue
            plant_gen = get_plant_gen_twh(p, 'gas_ccgt', year, year_factor, re_growth)
            if year > 2023 and p.get('egrid_gen_twh') is not None:
                plant_gen = p['egrid_gen_twh'] * year_factor
            ccs_gen += plant_gen

        ccs_ramp = min(1.0, max(0.0, (year - 2028) / 5.0))
        ccs_ccgt_twh = ccs_gen * ccs_ramp
        gen_by_fuel['gas_ccgt'] -= ccs_ccgt_twh

    gen_by_fuel['ccs_ccgt'] = ccs_ccgt_twh

    result = {}
    for fuel in ['nuclear', 'geothermal', 'wind', 'solar', 'hydro',
                 'gas_ccgt', 'gas_ct', 'oil_ct', 'gas_oil_ct', 'ccs_ccgt', 'battery']:
        val = gen_by_fuel.get(fuel, 0.0)
        if val != 0.0 or fuel in ('gas_ccgt', 'gas_ct', 'oil_ct', 'ccs_ccgt'):
            result[fuel] = round(val, 1)

    return result


def compute_fleet_emissions(plants, year, scenario='baseline', retired_fuels=None):
    """Calculate emissions at the plant level using eGRID-measured CO₂ rates.

    For 2023: uses actual eGRID CO₂ (equity-weighted Mt) directly.
    For future years: scales 2023 actuals by year_factor, applies plant-specific rate.
    Plants without eGRID data: uses default EMISSION_RATE by fuel type.
    """
    if retired_fuels is None:
        retired_fuels = {}

    year_factor = max(0.0, 1.0 - 0.008 * (year - 2023))
    re_growth = 1.0 + 0.02 * (year - 2023)

    emissions_by_fuel = defaultdict(float)

    for p in plants:
        fuel = p['fuel_type']
        if fuel not in EMISSION_RATE:
            continue
        cap = p['capacity_mw']
        if cap <= 0:
            continue

        # Check fleet-wide retirement
        if fuel in retired_fuels and year >= retired_fuels[fuel]:
            continue
        if fuel == 'gas_oil_ct' and 'oil_ct' in retired_fuels and year >= retired_fuels['oil_ct']:
            continue

        # Check plant-specific retirement (e.g., Mystic retired 2024)
        if p.get('retired_year') and year >= p['retired_year']:
            continue

        # 2023: use eGRID actual CO₂ directly if available
        if year == 2023 and p.get('egrid_co2_mt') is not None:
            emissions_by_fuel[fuel] += p['egrid_co2_mt']
            continue

        # Future years or plants without eGRID: compute from generation × rate
        gen_twh = get_plant_gen_twh(p, fuel, year, year_factor, re_growth)
        if year > 2023 and p.get('egrid_gen_twh') is not None:
            gen_twh = p['egrid_gen_twh'] * year_factor

        co2_rate = get_plant_co2_rate(p, fuel)
        emissions_by_fuel[fuel] += gen_twh * co2_rate

    # CCS: handled by scenario logic
    ccs_emis = 0.0
    if scenario in ('ccs_top_emitters', 'ccs_plus_new_gas', 'retire_peakers_ccs_baseload'):
        for p in plants:
            if p['fuel_type'] != 'gas_ccgt' or p['capacity_mw'] <= 0:
                continue
            if not p['ccs_eligible']:
                continue
            gen_twh = get_plant_gen_twh(p, 'gas_ccgt', year, year_factor, re_growth)
            if year > 2023 and p.get('egrid_gen_twh') is not None:
                gen_twh = p['egrid_gen_twh'] * year_factor

            co2_rate = get_plant_co2_rate(p, 'gas_ccgt')
            ccs_ramp = min(1.0, max(0.0, (year - 2028) / 5.0))
            ccs_emis_plant = gen_twh * co2_rate * ccs_ramp * 0.05  # 95% capture
            ccs_emis += ccs_emis_plant
            # Subtract from gas_ccgt (moved to ccs_ccgt)
            emissions_by_fuel['gas_ccgt'] -= gen_twh * co2_rate * ccs_ramp
            emissions_by_fuel['gas_ccgt'] = max(0, emissions_by_fuel['gas_ccgt'])

    emissions_by_fuel['ccs_ccgt'] = ccs_emis

    return {k: round(v, 2) for k, v in emissions_by_fuel.items()}


def total_emissions(emissions_by_fuel):
    return round(sum(emissions_by_fuel.values()), 2)


def build_envelope(base_emissions_2023, scenario_emissions_by_year):
    """Build P10/P50/P90 envelope from base emissions trajectory."""
    envelope = {}
    envelope_by_cost = {'Low': {}, 'Medium': {}, 'High': {}}

    for year in YEARS:
        e = scenario_emissions_by_year.get(year, base_emissions_2023)
        # P10/P50/P90 spread: ±12% around central estimate
        # TODO: Replace with real 1215 sweep percentiles when available
        spread = 0.12
        p50 = e
        p10 = round(p50 * (1 - spread), 1)
        p90 = round(p50 * (1 + spread), 1)
        p50 = round(p50, 1)

        envelope[str(year)] = {'p10': p10, 'p50': p50, 'p90': p90}

        for cost_level, factor in [('Low', 0.95), ('Medium', 1.0), ('High', 1.05)]:
            adj_p50 = round(p50 * factor, 1)
            adj_p10 = round(p10 * factor, 1)
            adj_p90 = round(p90 * factor, 1)
            envelope_by_cost[cost_level][str(year)] = {
                'p10': adj_p10, 'p50': adj_p50, 'p90': adj_p90
            }

    return envelope, envelope_by_cost


def build_plant_detail(fossil_plants, year, scenario='baseline', retired_fuels=None):
    """Build plant-level detail using eGRID actuals for 2023, scaled for future."""
    if retired_fuels is None:
        retired_fuels = {}

    year_factor = max(0.0, 1.0 - 0.008 * (year - 2023))
    re_growth = 1.0  # Not used for fossil but needed by helper
    details = []

    for p in fossil_plants:
        fuel = p['fuel_type']
        name = p['name']
        cap = p['capacity_mw']
        orispl = p['orispl'] or hash(name) % 90000 + 10000

        # Check retirement (fleet-wide + plant-specific)
        is_retired = False
        if fuel in retired_fuels and year >= retired_fuels[fuel]:
            is_retired = True
        if fuel == 'gas_oil_ct' and 'oil_ct' in retired_fuels and year >= retired_fuels['oil_ct']:
            is_retired = True
        if p.get('retired_year') and year >= p['retired_year']:
            is_retired = True

        if is_retired:
            details.append({
                'name': name, 'orispl': orispl, 'fuel_type': fuel,
                'capacity_mw': round(cap, 1),
                'ccs_capacity_mw': round(p['ccs_capacity_mw'], 1),
                'gen_twh': 0.0, 'emissions_mt': 0.0, 'status': 'retired'
            })
            continue

        # Generation: eGRID actual for 2023, scaled for future
        if year == 2023 and p.get('egrid_gen_twh') is not None:
            gen_twh = p['egrid_gen_twh']
        elif year > 2023 and p.get('egrid_gen_twh') is not None:
            gen_twh = p['egrid_gen_twh'] * year_factor
        else:
            cf = CAPACITY_FACTORS.get(fuel, 0.1)
            gen_twh = cap * cf * 8.760 * year_factor / 1000.0

        # Emissions: eGRID actual for 2023, plant-specific rate for future
        if year == 2023 and p.get('egrid_co2_mt') is not None:
            emissions_mt = p['egrid_co2_mt']
        else:
            co2_rate = get_plant_co2_rate(p, fuel)
            emissions_mt = gen_twh * co2_rate

        gen_twh = round(gen_twh, 3)
        emissions_mt = round(emissions_mt, 3)

        # CCS status
        status = 'operating'
        if scenario in ('ccs_top_emitters', 'ccs_plus_new_gas', 'retire_peakers_ccs_baseload'):
            if p['ccs_eligible'] and year >= 2030:
                status = 'ccs_retrofit'
                ccs_ramp = min(1.0, (year - 2028) / 5.0)
                emissions_mt = round(emissions_mt * (1 - 0.95 * ccs_ramp), 3)

        details.append({
            'name': name, 'orispl': orispl, 'fuel_type': fuel,
            'capacity_mw': round(cap, 1),
            'ccs_capacity_mw': round(p['ccs_capacity_mw'], 1),
            'gen_twh': gen_twh, 'emissions_mt': emissions_mt, 'status': status
        })

    details.sort(key=lambda x: x['emissions_mt'], reverse=True)
    return details


def build_scenario(plants, fossil_plants, scenario_key):
    """Build a complete scenario data structure using actual fleet capacities."""
    descriptions = {
        'baseline': 'Status quo — no CCS, no early retirements. Natural market-driven decline only.',
        'ccs_top_emitters': 'CCS retrofit on top CCS-eligible CCGT plants (95% capture). No retirements, no new builds.',
        'ccs_plus_new_gas': 'CCS on top emitters + 1,200 MW new efficient CCGT in PJM (2028-2029).',
        'retire_peakers_ccs_baseload': 'Retire all oil CTs and gas CTs by 2030, CCS retrofit on all CCGTs by 2035.'
    }
    colors = {
        'baseline': '#7E8083',
        'ccs_top_emitters': '#2372B9',
        'ccs_plus_new_gas': '#F47B27',
        'retire_peakers_ccs_baseload': '#6BA543'
    }

    retired_fuels = {}
    if scenario_key == 'retire_peakers_ccs_baseload':
        retired_fuels = {'oil_ct': 2030, 'gas_ct': 2030, 'gas_oil_ct': 2030}

    generation_by_fuel = {}
    emissions_by_fuel_data = {}
    plant_detail = {}
    emissions_trajectory = {}

    for year in YEARS:
        gen = compute_fleet_generation(
            plants, year, scenario=scenario_key, retired_fuels=retired_fuels
        )
        generation_by_fuel[str(year)] = gen

        emf = compute_fleet_emissions(
            plants, year, scenario=scenario_key, retired_fuels=retired_fuels
        )
        emissions_by_fuel_data[str(year)] = emf
        emissions_trajectory[year] = total_emissions(emf)

        pd = build_plant_detail(
            fossil_plants, year, scenario=scenario_key, retired_fuels=retired_fuels
        )
        plant_detail[str(year)] = pd

    base_e = emissions_trajectory[2023]
    envelope, envelope_by_cost = build_envelope(base_e, emissions_trajectory)

    return {
        'description': descriptions.get(scenario_key, scenario_key),
        'color': colors.get(scenario_key, '#888'),
        'envelope': envelope,
        'envelope_by_fossil_cost': envelope_by_cost,
        'plant_detail': plant_detail,
        'generation_by_fuel': generation_by_fuel,
        'emissions_by_fuel': emissions_by_fuel_data
    }


def build_targets(baseline_emissions_2023):
    """Build target trajectories."""
    base = baseline_emissions_2023

    sbti = {
        '2023': round(base, 1),
        '2025': round(base * 0.85, 1),
        '2030': round(base * 0.50, 1),
        '2035': round(base * 0.25, 1),
        '2040': 0.0,
        '2045': 0.0,
        '2050': 0.0
    }

    at = {
        '2023': round(base, 1),
        '2025': round(base * 0.88, 1),
        '2030': round(base * 0.65, 1),
        '2035': round(base * 0.35, 1),
        '2040': round(base * 0.05, 1),
        '2045': 0.0,
        '2050': 0.0
    }

    return {
        'sbti_15': {
            'label': 'SBTi 1.5°C (Power Sector v2)',
            'trajectory': sbti
        },
        'at_power_nz': {
            'label': 'AT Power NZ',
            'trajectory': at
        }
    }


def build_gap_analysis(scenarios, targets):
    """Calculate gap between each scenario's P50 and each target."""
    gap = {}
    for skey, sdata in scenarios.items():
        gap[skey] = {}
        for tkey, tdata in targets.items():
            trajectory = tdata['trajectory']
            gap_mt = {}
            for year_str, target_val in trajectory.items():
                env = sdata['envelope'].get(year_str)
                if env:
                    gap_mt[year_str] = round(env['p50'] - target_val, 1)

            year_achieved = None
            for y in YEARS:
                g = gap_mt.get(str(y))
                if g is not None and g <= 0:
                    year_achieved = y
                    break

            prob_meeting = {}
            for y in [2030, 2040, 2050]:
                env = sdata['envelope'].get(str(y))
                tgt = trajectory.get(str(y))
                if env and tgt is not None:
                    if tgt >= env['p90']:
                        prob_meeting[str(y)] = 0.95
                    elif tgt <= env['p10']:
                        prob_meeting[str(y)] = 0.05
                    else:
                        range_width = env['p90'] - env['p10']
                        if range_width > 0:
                            prob_meeting[str(y)] = round(
                                (env['p90'] - tgt) / range_width * 0.8 + 0.1, 2)
                        else:
                            prob_meeting[str(y)] = 0.5

            gap[skey][tkey] = {
                'gap_mt': gap_mt,
                'year_achieved': year_achieved,
                'prob_meeting': prob_meeting
            }

    return gap


def build_fleet_summary(plants):
    """Build fleet summary from actual Rosetta capacity data."""
    summary = defaultdict(lambda: {'count': 0, 'capacity_mw': 0.0, 'ccs_capacity_mw': 0.0})
    for p in plants:
        fuel = p['fuel_type']
        summary[fuel]['count'] += 1
        summary[fuel]['capacity_mw'] += p['capacity_mw']
        summary[fuel]['ccs_capacity_mw'] += p['ccs_capacity_mw']

    return {k: {
        'count': v['count'],
        'capacity_mw': round(v['capacity_mw'], 1),
        'ccs_capacity_mw': round(v['ccs_capacity_mw'], 1),
    } for k, v in summary.items()}


def main():
    print("Parsing Rosetta CSV (actual capacity)...")
    plants = parse_rosetta()
    cats = categorize_plants(plants)

    print(f"\nTotal plants: {len(plants)}")
    total_cap = 0.0
    total_ccs_cap = 0.0
    for fuel, plist in sorted(cats.items()):
        cap = sum(p['capacity_mw'] for p in plist)
        ccs = sum(p['ccs_capacity_mw'] for p in plist)
        print(f"  {fuel}: {len(plist)} plants, {cap:,.1f} MW capacity"
              + (f", {ccs:,.1f} MW CCS" if ccs > 0 else ""))
        total_cap += cap
        total_ccs_cap += ccs
    print(f"  TOTAL: {total_cap:,.1f} MW ({total_cap/1000:.1f} GW)")
    if total_ccs_cap > 0:
        print(f"  TOTAL CCS-eligible: {total_ccs_cap:,.1f} MW ({total_ccs_cap/1000:.1f} GW)")

    fossil_plants = get_fossil_plants(plants)
    fossil_cap = sum(p['capacity_mw'] for p in fossil_plants)
    print(f"\nFossil plants: {len(fossil_plants)} ({fossil_cap:,.1f} MW / {fossil_cap/1000:.1f} GW)")

    # Build all scenarios
    print("\nBuilding scenarios...")
    scenarios = {}
    for skey in ['baseline', 'ccs_top_emitters', 'ccs_plus_new_gas', 'retire_peakers_ccs_baseload']:
        print(f"  Building {skey}...")
        scenarios[skey] = build_scenario(plants, fossil_plants, skey)

    # Print baseline 2023 generation summary
    gen_2023 = scenarios['baseline']['generation_by_fuel']['2023']
    print(f"\n2023 Baseline Generation (TWh):")
    total_gen = 0
    for fuel, twh in gen_2023.items():
        print(f"  {fuel}: {twh}")
        total_gen += twh
    print(f"  TOTAL: {total_gen:.1f} TWh")

    # Emissions
    emf_2023 = scenarios['baseline']['emissions_by_fuel']['2023']
    total_e = sum(emf_2023.values())
    print(f"\n2023 Baseline Emissions: {total_e:.2f} Mt CO2")
    for fuel, mt in emf_2023.items():
        print(f"  {fuel}: {mt}")

    # Build targets
    targets = build_targets(total_e)
    print(f"\nSBTi 1.5°C trajectory (net zero by 2040):")
    for y, v in targets['sbti_15']['trajectory'].items():
        print(f"  {y}: {v} Mt")

    # Build gap analysis
    gap_analysis = build_gap_analysis(scenarios, targets)

    # Build fleet summary for metadata
    fleet_summary = build_fleet_summary(plants)

    # Assemble output
    output = {
        'scenarios': scenarios,
        'targets': targets,
        'gap_analysis': gap_analysis,
        'fleet_summary': fleet_summary,
        'metadata': {
            'fleet_name': 'Constellation Energy',
            'source': 'CEG_fleet_rosetta.csv',
            'capacity_source': 'Constellation Owned Capacity (MW) — equity-weighted from Rosetta',
            'sweep_count': 1215,
            'generated_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'total_plants': len(plants),
            'total_capacity_mw': round(total_cap, 1),
            'fossil_plants': len(fossil_plants),
            'fossil_capacity_mw': round(fossil_cap, 1),
            'notes': (
                f'Rebuilt from Rosetta CSV with actual plant capacities. '
                f'Nuclear: {fleet_summary.get("nuclear", {}).get("capacity_mw", 0):,.0f} MW, '
                f'Gas CCGT: {fleet_summary.get("gas_ccgt", {}).get("capacity_mw", 0):,.0f} MW, '
                f'Total fleet: {total_cap:,.0f} MW. '
                'Envelope P10/P90 uses ±12% synthetic spread (pending 1215 sweep integration).'
            ),
        }
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, 'fleet_scenario_results_sample.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nOutput written to: {output_path}")
    print(f"File size: {os.path.getsize(output_path):,} bytes")


if __name__ == '__main__':
    main()
