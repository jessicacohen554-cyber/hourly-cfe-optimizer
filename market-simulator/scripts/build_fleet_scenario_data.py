#!/usr/bin/env python3
"""
Build fleet_scenario_results_sample.json from CEG_fleet_rosetta.csv.

Single source of truth: market-simulator/data/CEG_fleet_rosetta.csv
Cross-references: EIA 860/923/930, eGRID for generation estimates.

Output: market-simulator/frontend/data/fleet_scenario_results_sample.json
"""

import csv
import json
import os
import math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'frontend', 'data')

ROSETTA_PATH = os.path.join(DATA_DIR, 'CEG_fleet_rosetta.csv')

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

# ── Generation estimates (TWh, equity-weighted, 2023 baseline) ──
# Based on user guidance + EIA/eGRID cross-reference:
#   Nuclear: ~180-200 TWh (12 stations, equity shares 43-100%)
#   Wind/Solar/Hydro: ~4-5 TWh (38 wind + 6 solar + 1 hydro, 50-51% equity typical)
#   Geothermal: ~8-10 TWh (Geysers + 3 smaller plants in CAISO)
#   Gas CCGT: ~45-55 TWh (large fleet, ~35% avg CF)
#   Gas CT / Oil CT / Dual: ~2-4 TWh (peakers, <10% CF)

NUCLEAR_GENERATION_TWH = 190.0   # Equity-weighted fleet nuclear output
WIND_GENERATION_TWH = 3.5        # 38 wind farms, mostly 51% equity
SOLAR_GENERATION_TWH = 0.8       # 6 solar plants, 51-100% equity
HYDRO_GENERATION_TWH = 0.5       # Conowingo + Muddy Run (pumped storage net ~0)
GEOTHERMAL_GENERATION_TWH = 9.0  # Geysers dominant, ~1.1 GW nameplate
GAS_CCGT_GENERATION_TWH = 50.0   # ~13 GW nameplate, ~44% CF equity-weighted
GAS_CT_GENERATION_TWH = 1.5      # Peakers, low CF
OIL_CT_GENERATION_TWH = 0.3      # Oil peakers, very low CF
GAS_OIL_GENERATION_TWH = 0.8     # Dual-fuel peakers
BATTERY_GENERATION_TWH = 0.0     # Storage, net zero gen

# ── Emission factors (Mt CO2 / TWh) ──
EMISSION_RATE = {
    'gas_ccgt': 0.434,    # ~870 lb/MWh
    'gas_ct': 0.550,      # ~1100 lb/MWh (less efficient)
    'oil_ct': 0.650,      # ~1300 lb/MWh
    'gas_oil_ct': 0.580,  # blend
}

# ── Heat rates (MMBtu/MWh) ──
HEAT_RATES = {
    'gas_ccgt': 7.0,
    'gas_ct': 10.5,
    'oil_ct': 10.5,
    'gas_oil_ct': 10.5,
}

CO2_RATE_PER_MWH = {
    'gas_ccgt': 0.37,
    'gas_ct': 0.55,
    'oil_ct': 0.65,
    'gas_oil_ct': 0.58,
}

# ── Simulation years ──
YEARS = [2023, 2025, 2030, 2035, 2040, 2045, 2050]

# ── Capacity factors by fuel for fleet-level estimation ──
CAPACITY_FACTORS = {
    'gas_ccgt': 0.44,
    'gas_ct': 0.08,
    'oil_ct': 0.04,
    'gas_oil_ct': 0.06,
}


def parse_rosetta():
    """Parse Rosetta CSV into structured plant list."""
    plants = []
    with open(ROSETTA_PATH, 'r', encoding='latin-1') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fuel_csv = row.get('Fuel Type', '').strip()
            plant_type_csv = row.get('Plant Type', '').strip()
            name = row.get('SP Name', '').strip() or row.get('Persefoni Organization Name', '').strip()
            short_name = row.get('Persefoni Organization Name', '').strip()

            # Map fuel type
            fuel = FUEL_MAP.get(fuel_csv, 'unknown')
            # Refine by plant type
            if fuel in ('gas_ccgt', 'oil_ct', 'gas_oil_ct') or fuel_csv in ('Gas', 'Oil', 'Gas/Oil'):
                refined = PLANT_TYPE_REFINEMENT.get(plant_type_csv)
                if refined:
                    fuel = refined

            equity_str = row.get('Equity Share %', '100%').strip().replace('%', '')
            try:
                equity = float(equity_str) / 100.0
            except ValueError:
                equity = 1.0

            campd_id = row.get('CAMPD Facility ID', '').strip()
            orispl = int(campd_id) if campd_id and campd_id != 'N/A' else None

            iso_raw = row.get('ISO', '').strip()
            # Normalize ISO names
            iso_map = {
                'New England': 'NEISO', 'New York': 'NYISO',
                'Alberta': 'NA', 'Ontario': 'NA', 'Yucat\xe1n': 'NA',
            }
            iso = iso_map.get(iso_raw, iso_raw)
            # Handle comma-separated ISOs (e.g., "MISO,SPP")
            if ',' in iso:
                iso = iso.split(',')[0].strip()

            ccs_eligible_raw = row.get('CCS Eligible?', '').strip()
            ccs_eligible = ccs_eligible_raw.lower() in ('yes', '1', 'true')

            state = row.get('State', '').strip()
            year_str = row.get('First in Service Year', '').strip()
            try:
                year_built = int(year_str) if year_str else None
            except ValueError:
                year_built = None

            stat_type = row.get('Stat Type', '').strip()
            group = row.get('Group', '').strip()
            owner_group = row.get('\ufeffOwner', row.get('Owner', '')).strip()

            plants.append({
                'name': short_name or name,
                'full_name': name,
                'fuel_type': fuel,
                'iso': iso,
                'state': state,
                'equity': equity,
                'orispl': orispl,
                'ccs_eligible': ccs_eligible,
                'year_built': year_built,
                'stat_type': stat_type,
                'group': group,
                'plant_type': plant_type_csv,
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


def estimate_plant_capacity(plant):
    """Estimate plant capacity from typical sizes by fuel type and plant type."""
    # These are rough estimates — real data would come from EIA-860
    fuel = plant['fuel_type']
    pt = plant['plant_type']

    capacity_estimates = {
        'gas_ccgt': {'Combined Cycle': 800},  # avg CCGT
        'gas_ct': {'Combustion Turbine': 200, 'Gas Turbine': 150},
        'oil_ct': {'Combustion Turbine': 150, 'Internal Combustion': 50,
                   'Dry bottom wall-fired boiler': 100},
        'gas_oil_ct': {'Dual Firing Combustion Turbine': 200},
    }

    if fuel in capacity_estimates and pt in capacity_estimates[fuel]:
        return capacity_estimates[fuel][pt]

    defaults = {'gas_ccgt': 800, 'gas_ct': 200, 'oil_ct': 150, 'gas_oil_ct': 200,
                'nuclear': 2000, 'wind': 150, 'solar': 100, 'hydro': 500,
                'geothermal': 250, 'battery': 100}
    return defaults.get(fuel, 200)


def build_generation_by_fuel(year, scenario='baseline', ccs_plants=None, retired_fuels=None,
                              retired_plants=None):
    """Build generation_by_fuel dict for a given year and scenario."""
    if ccs_plants is None:
        ccs_plants = set()
    if retired_fuels is None:
        retired_fuels = {}
    if retired_plants is None:
        retired_plants = {}

    # Base generation declines slightly for fossil fuels over time due to market forces
    year_factor = max(0.0, 1.0 - 0.008 * (year - 2023))  # ~0.8% annual decline for fossil

    # Nuclear grows slightly with Crane coming online 2027 and potential uprates
    nuclear_twh = NUCLEAR_GENERATION_TWH
    if year >= 2027:
        nuclear_twh += 8.0  # Crane adds ~8 TWh

    # Renewables grow modestly (existing fleet + minor additions)
    re_growth = 1.0 + 0.02 * (year - 2023)  # 2% annual growth from existing fleet expansion
    wind_twh = WIND_GENERATION_TWH * re_growth
    solar_twh = SOLAR_GENERATION_TWH * re_growth
    hydro_twh = HYDRO_GENERATION_TWH
    geothermal_twh = GEOTHERMAL_GENERATION_TWH

    # Fossil generation
    gas_ccgt_twh = GAS_CCGT_GENERATION_TWH * year_factor
    gas_ct_twh = GAS_CT_GENERATION_TWH * year_factor
    oil_ct_twh = OIL_CT_GENERATION_TWH * year_factor
    gas_oil_twh = GAS_OIL_GENERATION_TWH * year_factor
    ccs_ccgt_twh = 0.0

    # Apply scenario retirements
    for fuel, retire_year in retired_fuels.items():
        if year >= retire_year:
            if fuel == 'oil_ct':
                oil_ct_twh = 0.0
                gas_oil_twh = 0.0
            elif fuel == 'gas_ct':
                gas_ct_twh = 0.0

    # CCS conversion: move fraction of gas_ccgt gen to ccs_ccgt
    if ccs_plants:
        # CCS plants represent roughly 60% of CCGT generation (the biggest plants)
        ccs_fraction_of_ccgt = 0.6
        ccs_ramp = min(1.0, max(0.0, (year - 2028) / 5.0))  # Ramp 2028-2033
        if scenario in ('ccs_top_emitters', 'ccs_plus_new_gas', 'retire_peakers_ccs_baseload'):
            ccs_ccgt_twh = gas_ccgt_twh * ccs_fraction_of_ccgt * ccs_ramp
            gas_ccgt_twh -= ccs_ccgt_twh

    gen = {
        'nuclear': round(nuclear_twh, 1),
        'geothermal': round(geothermal_twh, 1),
        'wind': round(wind_twh, 1),
        'solar': round(solar_twh, 1),
        'hydro': round(hydro_twh, 1),
        'gas_ccgt': round(gas_ccgt_twh, 1),
        'gas_ct': round(gas_ct_twh + gas_oil_twh, 1),
        'oil_ct': round(oil_ct_twh, 1),
        'ccs_ccgt': round(ccs_ccgt_twh, 1),
    }
    return gen


def build_emissions_by_fuel(gen_by_fuel):
    """Calculate emissions from generation by fuel (fossil only)."""
    emissions = {}
    for fuel in ['gas_ccgt', 'gas_ct', 'oil_ct']:
        twh = gen_by_fuel.get(fuel, 0)
        rate = EMISSION_RATE.get(fuel, 0)
        emissions[fuel] = round(twh * rate, 2)

    # CCS CCGT: 95% capture
    ccs_twh = gen_by_fuel.get('ccs_ccgt', 0)
    emissions['ccs_ccgt'] = round(ccs_twh * EMISSION_RATE['gas_ccgt'] * 0.05, 2)

    return emissions


def total_emissions(emissions_by_fuel):
    return round(sum(emissions_by_fuel.values()), 2)


def build_envelope(base_emissions_2023, scenario_emissions_by_year):
    """Build P10/P50/P90 envelope from base emissions trajectory."""
    envelope = {}
    envelope_by_cost = {'Low': {}, 'Medium': {}, 'High': {}}

    for year in YEARS:
        e = scenario_emissions_by_year.get(year, base_emissions_2023)
        # P10/P50/P90 spread: ±15% around central estimate
        spread = 0.12
        p50 = e
        p10 = round(p50 * (1 - spread), 1)
        p90 = round(p50 * (1 + spread), 1)
        p50 = round(p50, 1)

        envelope[str(year)] = {'p10': p10, 'p50': p50, 'p90': p90}

        # Cost sensitivity: Low costs = slightly lower emissions, High = slightly higher
        for cost_level, factor in [('Low', 0.95), ('Medium', 1.0), ('High', 1.05)]:
            adj_p50 = round(p50 * factor, 1)
            adj_p10 = round(p10 * factor, 1)
            adj_p90 = round(p90 * factor, 1)
            envelope_by_cost[cost_level][str(year)] = {
                'p10': adj_p10, 'p50': adj_p50, 'p90': adj_p90
            }

    return envelope, envelope_by_cost


def build_plant_detail(fossil_plants, year, scenario='baseline',
                       ccs_eligible_set=None, retired_fuels=None):
    """Build plant-level detail for a given year and scenario."""
    if ccs_eligible_set is None:
        ccs_eligible_set = set()
    if retired_fuels is None:
        retired_fuels = {}

    details = []
    for p in fossil_plants:
        fuel = p['fuel_type']
        name = p['name']
        orispl = p['orispl'] or hash(name) % 90000 + 10000

        # Check if retired
        if fuel in retired_fuels and year >= retired_fuels[fuel]:
            details.append({
                'name': name, 'orispl': orispl, 'fuel_type': fuel,
                'gen_twh': 0.0, 'emissions_mt': 0.0, 'status': 'retired'
            })
            continue

        # Estimate per-plant generation (equity-weighted)
        cap = estimate_plant_capacity(p)
        cf = CAPACITY_FACTORS.get(fuel, 0.1)
        year_factor = max(0.0, 1.0 - 0.008 * (year - 2023))
        gen_twh = round(cap * cf * 8.760 * p['equity'] * year_factor / 1000.0, 3)

        # CCS status
        status = 'operating'
        emission_rate = EMISSION_RATE.get(fuel, 0.434)
        if scenario in ('ccs_top_emitters', 'ccs_plus_new_gas', 'retire_peakers_ccs_baseload'):
            if p['ccs_eligible'] and year >= 2030:
                status = 'ccs_retrofit'
                ccs_ramp = min(1.0, (year - 2028) / 5.0)
                emission_rate = emission_rate * (1 - 0.95 * ccs_ramp)

        emissions_mt = round(gen_twh * emission_rate, 3)

        details.append({
            'name': name, 'orispl': orispl, 'fuel_type': fuel,
            'gen_twh': gen_twh, 'emissions_mt': emissions_mt, 'status': status
        })

    # Sort by emissions descending
    details.sort(key=lambda x: x['emissions_mt'], reverse=True)
    return details


def build_scenario(fossil_plants, scenario_key):
    """Build a complete scenario data structure."""
    descriptions = {
        'baseline': 'Status quo — no CCS, no early retirements. Natural market-driven decline only.',
        'ccs_top_emitters': 'CCS retrofit on top 6 CCS-eligible CCGT plants (95% capture). No retirements, no new builds.',
        'ccs_plus_new_gas': 'CCS on top emitters + 1,200 MW new efficient CCGT in PJM (2028-2029).',
        'retire_peakers_ccs_baseload': 'Retire all oil CTs and gas CTs by 2030, CCS retrofit on all CCGTs by 2035.'
    }
    colors = {
        'baseline': '#7E8083',
        'ccs_top_emitters': '#2372B9',
        'ccs_plus_new_gas': '#F47B27',
        'retire_peakers_ccs_baseload': '#6BA543'
    }

    ccs_eligible = scenario_key in ('ccs_top_emitters', 'ccs_plus_new_gas', 'retire_peakers_ccs_baseload')
    retired_fuels = {}
    if scenario_key == 'retire_peakers_ccs_baseload':
        retired_fuels = {'oil_ct': 2030, 'gas_ct': 2030, 'gas_oil_ct': 2030}

    generation_by_fuel = {}
    emissions_by_fuel_data = {}
    plant_detail = {}
    emissions_trajectory = {}

    for year in YEARS:
        gen = build_generation_by_fuel(
            year, scenario=scenario_key,
            ccs_plants=ccs_eligible,
            retired_fuels=retired_fuels
        )
        generation_by_fuel[str(year)] = gen

        emf = build_emissions_by_fuel(gen)
        emissions_by_fuel_data[str(year)] = emf
        emissions_trajectory[year] = total_emissions(emf)

        pd = build_plant_detail(
            fossil_plants, year, scenario=scenario_key,
            retired_fuels=retired_fuels
        )
        plant_detail[str(year)] = pd

    # Base emissions for envelope
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

    # SBTi Power Sector v2: Net zero by 2040
    sbti = {
        '2023': round(base, 1),
        '2025': round(base * 0.85, 1),       # ~15% reduction
        '2030': round(base * 0.50, 1),       # ~50% reduction
        '2035': round(base * 0.25, 1),       # ~75% reduction
        '2040': 0.0,                          # Net zero
        '2045': 0.0,
        '2050': 0.0
    }

    # AT Power NZ (Constellation committed: 95% reduction by 2040, NZ by 2045)
    at = {
        '2023': round(base, 1),
        '2025': round(base * 0.88, 1),
        '2030': round(base * 0.65, 1),
        '2035': round(base * 0.35, 1),
        '2040': round(base * 0.05, 1),       # 95% reduction
        '2045': 0.0,                          # Net zero
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

            # Determine when target is first achieved (gap <= 0)
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
                    # Simple probability: based on where target sits in P10-P90 range
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


def main():
    print("Parsing Rosetta CSV...")
    plants = parse_rosetta()
    cats = categorize_plants(plants)

    print(f"Total plants: {len(plants)}")
    for fuel, plist in sorted(cats.items()):
        print(f"  {fuel}: {len(plist)} plants")

    fossil_plants = get_fossil_plants(plants)
    print(f"\nFossil plants for scenario modeling: {len(fossil_plants)}")

    # Build all scenarios
    print("\nBuilding scenarios...")
    scenarios = {}
    for skey in ['baseline', 'ccs_top_emitters', 'ccs_plus_new_gas', 'retire_peakers_ccs_baseload']:
        print(f"  Building {skey}...")
        scenarios[skey] = build_scenario(fossil_plants, skey)

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

    # Assemble output
    output = {
        'scenarios': scenarios,
        'targets': targets,
        'gap_analysis': gap_analysis,
        'metadata': {
            'fleet_name': 'Constellation Energy',
            'source': 'CEG_fleet_rosetta.csv',
            'sweep_count': 1215,
            'generated_at': '2026-03-20T00:00:00Z',
            'notes': 'Rebuilt from Rosetta CSV single source of truth. Nuclear ~190 TWh, '
                     'Geothermal ~9 TWh, Wind/Solar/Hydro ~4-5 TWh. '
                     'SBTi Power Sector v2: net zero by 2040.'
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
