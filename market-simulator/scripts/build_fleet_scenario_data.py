#!/usr/bin/env python3
"""
Build fleet_scenario_results_sample.json from CEG_fleet_rosetta.csv + CAMPD 2025.

Uses three years of actual emissions/generation data:
  - 2023/2024: Equity-weighted actuals read directly from Rosetta CSV columns
  - 2025: CAMPD facility-level data aggregated from unit-level reports
  - 2030+: Projected forward from 2024 actuals using year_factor scaling

Plants without actuals (Canadian, small peakers) use heat-rate-derived estimates.

Sources:
  - CEG_fleet_rosetta.csv: Plant inventory, capacity, equity, 2023/2024 actuals
  - 2025_annual_campd_emissions.csv: CAMPD 2025 unit-level emissions/generation

Output: market-simulator/frontend/data/fleet_scenario_results_sample.json
"""

import csv
import json
import os
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'frontend', 'data')

ROSETTA_PATH = os.path.join(DATA_DIR, 'CEG_fleet_rosetta.csv')
CAMPD_2025_PATH = os.path.join(DATA_DIR, '2025_annual_campd_emissions.csv')
OVERRIDES_PATH = os.path.join(DATA_DIR, 'plant_emissions_overrides.json')

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
YEARS = list(range(2023, 2051))  # Annual: 2023–2050 (28 years)

# ── Capacity factors by fuel type ──
# Used to estimate annual generation from actual plant capacity
SWEEP_PARQUET_PATH = os.path.join(
    SCRIPT_DIR, '..', 'results', 'sweep_1215', 'sweep_1215_flat.parquet'
)

# Mapping from fleet fuel types to sweep CF column names
SWEEP_CF_COLS = {
    'gas_ccgt': 'ge_gas_ccgt_cf',
    'gas_ct': 'ge_gas_ct_cf',
    'oil_ct': 'ge_oil_ct_cf',
}

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


def load_overrides():
    """Load user-supplied plant emissions overrides by year.

    Returns dict: {year_str: {campd_id: {co2_short_tons, net_gen_mwh, ...}}}
    """
    if not os.path.exists(OVERRIDES_PATH):
        return {}

    with open(OVERRIDES_PATH, 'r') as f:
        data = json.load(f)

    overrides = {}
    for year_str, plants in data.items():
        if year_str.startswith('_'):
            continue
        by_id = {}
        for campd_str, vals in plants.items():
            try:
                cid = int(campd_str)
            except ValueError:
                continue
            by_id[cid] = {
                'co2_short_tons': vals.get('co2_short_tons', 0),
                'net_gen_mwh': vals.get('net_gen_mwh', 0),
                'heat_input_mmbtu': vals.get('heat_input_mmbtu'),
            }
        if by_id:
            overrides[year_str] = by_id

    if overrides:
        total_plants = sum(len(v) for v in overrides.values())
        print(f"  Loaded overrides: {total_plants} plant entries across years {list(overrides.keys())}")
    return overrides


# Module-level overrides cache (loaded once, used by all functions)
_OVERRIDES = None


def get_overrides():
    global _OVERRIDES
    if _OVERRIDES is None:
        _OVERRIDES = load_overrides()
    return _OVERRIDES


# ── Sweep percentile cache ──
_SWEEP_CF_PERCENTILES = None


def load_sweep_cf_percentiles():
    """Load 1,215-scenario sweep and compute CF percentiles per ISO/year/fuel.

    Returns dict: {iso: {year: {fuel: {p10, p50, p90}}}}
    Falls back to None if sweep parquet not available.
    """
    global _SWEEP_CF_PERCENTILES
    if _SWEEP_CF_PERCENTILES is not None:
        return _SWEEP_CF_PERCENTILES

    if not HAS_PANDAS or not os.path.exists(SWEEP_PARQUET_PATH):
        print("  WARNING: Sweep parquet not found — using synthetic ±12% spread")
        _SWEEP_CF_PERCENTILES = {}
        return _SWEEP_CF_PERCENTILES

    print("  Loading 1,215-scenario sweep for real CF percentiles...")
    df = pd.read_parquet(SWEEP_PARQUET_PATH)

    result = {}
    for iso in df['iso'].unique():
        result[iso] = {}
        iso_df = df[df['iso'] == iso]
        for year in iso_df['year'].unique():
            yr_df = iso_df[iso_df['year'] == year]
            year_int = int(year)
            result[iso][year_int] = {}
            for fuel, col in SWEEP_CF_COLS.items():
                vals = yr_df[col].dropna()
                if len(vals) > 10:
                    result[iso][year_int][fuel] = {
                        'p10': float(vals.quantile(0.10)),
                        'p50': float(vals.quantile(0.50)),
                        'p90': float(vals.quantile(0.90)),
                    }

    sweep_years = set()
    for iso_data in result.values():
        sweep_years.update(iso_data.keys())
    print(f"  Sweep loaded: {len(result)} ISOs, years {min(sweep_years)}-{max(sweep_years)}")
    _SWEEP_CF_PERCENTILES = result
    return result


def compute_fleet_emissions_sweep(plants, year, scenario='baseline',
                                  retired_fuels=None,
                                  base_emissions_mt=None):
    """Compute fleet emissions at P10/P50/P90 using sweep CF variation.

    Uses the sweep's P10/P50/P90 capacity factors as *relative* multipliers
    around the fleet's central (P50) projection. For each ISO/fuel, computes
    the ratio P10_cf/P50_cf and P90_cf/P50_cf, then applies those ratios to
    the plant's base projected emissions.

    For years with actuals (2023-2025), emissions are known — no uncertainty.
    P10/P50/P90 are all equal to the actual value.

    Returns: dict {p10: float, p50: float, p90: float} in Mt CO2
    """
    if retired_fuels is None:
        retired_fuels = {}

    sweep = load_sweep_cf_percentiles()
    if not sweep:
        return None  # Caller falls back to synthetic spread

    year_factor = max(0.0, 1.0 - 0.008 * (year - 2024))
    re_growth = 1.0 + 0.02 * (year - 2024)

    # For years with actual data, emissions are known — no sweep modulation
    # Check if all fossil plants have actuals for this year
    has_actuals_year = year <= 2025

    if has_actuals_year and base_emissions_mt is not None:
        return {'p10': base_emissions_mt, 'p50': base_emissions_mt,
                'p90': base_emissions_mt}

    # Separate fixed emissions (non-fossil, actuals, overrides) from
    # variable emissions (modeled fossil, subject to market conditions)
    fixed_emissions = 0.0
    variable_by_iso_fuel = defaultdict(float)  # (iso, fuel) -> base emissions Mt

    for p in plants:
        fuel = p['fuel_type']
        actual_co2 = p.get('actual_co2_mt', {})
        has_actual_emissions = any(v > 0 for k, v in actual_co2.items()
                                   if isinstance(k, int))

        if fuel not in EMISSION_RATE and not has_actual_emissions:
            continue
        cap = p['capacity_mw']
        if cap <= 0:
            continue

        has_actual_for_year = year in actual_co2

        if not has_actual_for_year:
            if fuel in retired_fuels and year >= retired_fuels[fuel]:
                continue
            if fuel == 'gas_oil_ct' and 'oil_ct' in retired_fuels \
                    and year >= retired_fuels['oil_ct']:
                continue
            if p.get('retired_year') and year >= p['retired_year']:
                continue

        orispl = p.get('orispl')
        overrides = get_overrides()
        year_overrides = overrides.get(str(year), {})
        if orispl and orispl in year_overrides:
            equity = p.get('equity', 1.0)
            co2_mt = year_overrides[orispl]['co2_short_tons'] * equity \
                * 0.907185 / 1e6
            fixed_emissions += co2_mt
            continue

        if has_actual_for_year:
            fixed_emissions += actual_co2[year] / 1e6
            continue

        # Modeled emissions — variable with market conditions
        base_gen_twh = get_plant_gen_twh(p, fuel, year, year_factor, re_growth)
        co2_rate = get_plant_co2_rate(p, fuel)
        base_em = base_gen_twh * co2_rate
        iso = p.get('iso', '')
        variable_by_iso_fuel[(iso, fuel)] += base_em

    # Apply sweep CF percentiles to variable emissions.
    # Use the sweep's absolute P50 CF trajectory (normalized to 2024 reference)
    # instead of the linear year_factor, so the P50 line reflects actual market
    # dynamics (e.g., the 2034-35 inflection from renewables displacement).
    totals = {'p10': fixed_emissions, 'p50': fixed_emissions,
              'p90': fixed_emissions}

    for (iso, fuel), base_em in variable_by_iso_fuel.items():
        sweep_fuel = fuel if fuel != 'gas_oil_ct' else 'gas_ct'
        iso_year = sweep.get(iso, {}).get(year, {})
        fuel_pcts = iso_year.get(sweep_fuel)

        # Get 2024 reference CF to normalize the sweep trajectory
        ref_pcts = sweep.get(iso, {}).get(2024, {}).get(sweep_fuel)

        if (fuel_pcts and fuel_pcts['p50'] > 0.001
                and ref_pcts and ref_pcts['p50'] > 0.001
                and year_factor > 0.001):
            # Scale base_em from linear model → sweep-based trajectory.
            # base_em was computed with linear year_factor; replace it with
            # the sweep's CF ratio relative to 2024 reference.
            # sweep_scale = (sweep_cf[year] / sweep_cf[2024]) / year_factor
            inv_linear = 1.0 / year_factor
            ref_cf = ref_pcts['p50']
            totals['p10'] += base_em * inv_linear * fuel_pcts['p10'] / ref_cf
            totals['p50'] += base_em * inv_linear * fuel_pcts['p50'] / ref_cf
            totals['p90'] += base_em * inv_linear * fuel_pcts['p90'] / ref_cf
        else:
            # No sweep data — add base emissions unchanged
            for pct in ('p10', 'p50', 'p90'):
                totals[pct] += base_em

    # CCS adjustment — applied uniformly to all percentiles
    if scenario in ('ccs_top_emitters', 'ccs_plus_new_gas',
                     'retire_peakers_ccs_baseload'):
        for p in plants:
            if p['fuel_type'] != 'gas_ccgt' or p['capacity_mw'] <= 0:
                continue
            if not p['ccs_eligible']:
                continue
            if p.get('retired_year') and year >= p['retired_year']:
                continue
            gen_twh = get_plant_gen_twh(p, 'gas_ccgt', year, year_factor,
                                        re_growth)
            co2_rate = get_plant_co2_rate(p, 'gas_ccgt')
            ccs_ramp = min(1.0, max(0.0, (year - 2028) / 5.0))
            reduction = gen_twh * co2_rate * ccs_ramp * 0.95
            for pct in ('p10', 'p50', 'p90'):
                totals[pct] -= reduction
                totals[pct] = max(0.0, totals[pct])

    return {k: round(v, 2) for k, v in totals.items()}


def parse_number(s):
    """Parse a number that may have commas, spaces, or be a dash/empty for zero."""
    if not s:
        return 0.0
    s = s.strip()
    if s in ('-', 'N/A', ''):
        return 0.0
    try:
        return float(s.replace(',', ''))
    except ValueError:
        return 0.0


def load_campd_2025():
    """Load 2025 CAMPD emissions/generation, aggregated from unit to facility level.

    Returns dict: {facility_id: {'co2_mt': float, 'gen_mwh': float}}
    CO2 is converted from short tons to metric tons (×0.907185).
    """
    if not os.path.exists(CAMPD_2025_PATH):
        print(f"  WARNING: {CAMPD_2025_PATH} not found, no 2025 CAMPD data")
        return {}

    by_facility = defaultdict(lambda: {'co2_st': 0.0, 'gen_mwh': 0.0})
    with open(CAMPD_2025_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid_str = row.get('Facility ID', '').strip()
            if not fid_str:
                continue
            try:
                fid = int(fid_str)
            except ValueError:
                continue
            co2_st = parse_number(row.get('CO2 Mass (short tons)', ''))
            gen_mwh = parse_number(row.get('Gross Load (MWh)', ''))
            by_facility[fid]['co2_st'] += co2_st
            by_facility[fid]['gen_mwh'] += gen_mwh

    result = {}
    for fid, vals in by_facility.items():
        result[fid] = {
            'co2_mt': vals['co2_st'] * 0.907185,  # short tons → metric tons
            'gen_mwh': vals['gen_mwh'],
        }

    total_co2 = sum(v['co2_mt'] for v in result.values())
    total_gen = sum(v['gen_mwh'] for v in result.values())
    print(f"  Loaded CAMPD 2025: {len(result)} facilities, "
          f"{total_co2/1e6:.2f} MMt CO₂, {total_gen/1e6:.1f} TWh")
    return result


def parse_rosetta():
    """Parse Rosetta CSV into structured plant list.

    Reads actual capacity from 'Constellation Owned Capacity (MW)' column
    (already equity-weighted) and CCS capacity from 'Available CCS Capacity'.
    Reads 2023/2024 actual emissions and generation directly from Rosetta columns.
    Integrates 2025 CAMPD data for fossil plants.
    """
    campd_2025 = load_campd_2025()

    # Pre-scan to find shared CAMPD IDs and their 2023/2024 emission totals
    # Used to split 2025 CAMPD data proportionally
    campd_emission_totals = defaultdict(float)  # campd_id → sum of 2023+2024 emissions
    campd_plant_emissions = {}  # (campd_id, plant_name) → 2023+2024 emissions

    with open(ROSETTA_PATH, 'r', encoding='latin-1') as f:
        reader = csv.DictReader(f)
        for row in reader:
            campd = row.get('CAMPD Facility ID', '').strip()
            if campd and campd != 'N/A':
                cid = int(campd)
                name = row.get('SP Name', '').strip() or row.get('Name', '').strip()
                em23 = parse_number(row.get('2023_equity_emissions_mtco2', ''))
                em24 = parse_number(row.get('2024_equity_emissions_mtco2', ''))
                total = em23 + em24
                campd_emission_totals[cid] += total
                campd_plant_emissions[(cid, name)] = total

    plants = []
    with open(ROSETTA_PATH, 'r', encoding='latin-1') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fuel_csv = row.get('Fuel Type', '').strip()
            plant_type_csv = row.get('Plant Type', '').strip()
            name = row.get('SP Name', '').strip() or row.get('Name', '').strip()

            # Skip pumped storage — not a generating asset for fleet modeling
            if 'Muddy Run' in name or 'Pumped Storage' in plant_type_csv:
                continue

            # Map fuel type
            fuel = FUEL_MAP.get(fuel_csv, 'unknown')
            # Refine Gas plants by plant type (CT vs CCGT)
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

            # Available CCS capacity
            ccs_cap_str = row.get('Available CCS Capacity', '').strip()
            try:
                ccs_capacity_mw = float(ccs_cap_str) if ccs_cap_str else 0.0
            except ValueError:
                ccs_capacity_mw = 0.0

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

            # Retired plants
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

            # ORISPL mismatches: these CAMPD IDs map to wrong plant
            CAMPD_EXCLUDE = {2410, 592, 597, 55401}
            if orispl in CAMPD_EXCLUDE:
                orispl = None

            # ── Read 2023/2024 actuals directly from Rosetta ──
            # These are already equity-weighted — no additional multiplication needed
            em_2023 = parse_number(row.get('2023_equity_emissions_mtco2', ''))
            gen_2023 = parse_number(row.get('2023_equity_netgen_MWh', ''))
            em_2024 = parse_number(row.get('2024_equity_emissions_mtco2', ''))
            gen_2024 = parse_number(row.get('2024_equity_netgen_MWh', ''))

            # Units: emissions in metric tons, generation in MWh
            actual_co2_mt = {2023: em_2023, 2024: em_2024}
            actual_gen_mwh = {2023: gen_2023, 2024: gen_2024}

            # Solar exception: 2024 solar gen is anomalous (0.01 TWh vs 0.57 TWh in 2023)
            # Use 2023 solar generation as the projection base
            if fuel == 'solar' and gen_2023 > 0 and gen_2024 < gen_2023 * 0.1:
                actual_gen_mwh['projection_base'] = gen_2023
            else:
                actual_gen_mwh['projection_base'] = gen_2024 if gen_2024 > 0 else gen_2023

            # ── Derive CO2 rate from actuals (t CO2 / MWh) for forward projection ──
            co2_rate_t_per_mwh = None
            if gen_2023 > 0 and em_2023 > 0:
                co2_rate_t_per_mwh = em_2023 / gen_2023  # metric tons / MWh
            elif gen_2024 > 0 and em_2024 > 0:
                co2_rate_t_per_mwh = em_2024 / gen_2024

            # ── 2025 CAMPD data ──
            # CAMPD reports total facility emissions/generation — must apply equity.
            # Only assign 2025 CAMPD data to plants with prior emissions (2023 or 2024)
            # to avoid inflating totals with new acquisitions or data anomalies.
            # Also skip plants retired before 2025.
            em_2025 = None
            gen_2025 = None
            has_prior_emissions = em_2023 > 0 or em_2024 > 0
            is_retired_before_2025 = retired_year is not None and retired_year < 2025

            if orispl and orispl in campd_2025 and has_prior_emissions and not is_retired_before_2025:
                campd_data = campd_2025[orispl]
                # For shared CAMPD IDs, split proportionally using 2023/2024 emission ratios
                total_em_for_campd = campd_emission_totals.get(orispl, 0)
                plant_em = campd_plant_emissions.get((orispl, name), 0)
                if total_em_for_campd > 0 and plant_em > 0:
                    share = plant_em / total_em_for_campd
                else:
                    share = 1.0
                em_2025 = campd_data['co2_mt'] * share * equity
                gen_2025 = campd_data['gen_mwh'] * share * equity
            elif fuel in EMISSION_RATE and has_prior_emissions and not is_retired_before_2025:
                # Plants not in CAMPD 2025 but with prior emissions:
                # use average of 2023/2024 Rosetta values
                vals = [v for v in [em_2023, em_2024] if v > 0]
                em_2025 = sum(vals) / len(vals) if vals else 0.0
                gen_vals = [v for v in [gen_2023, gen_2024] if v > 0]
                gen_2025 = sum(gen_vals) / len(gen_vals) if gen_vals else 0.0

            if em_2025 is not None:
                actual_co2_mt[2025] = em_2025
            if gen_2025 is not None:
                actual_gen_mwh[2025] = gen_2025

            # Plants with zero emissions in both 2023 AND 2024 should not generate
            # phantom modeled emissions — they're either not operated by CEG or non-emitting
            if not has_prior_emissions and fuel in EMISSION_RATE and 2025 not in actual_co2_mt:
                actual_co2_mt[2025] = 0.0
                actual_gen_mwh[2025] = 0.0
                actual_gen_mwh['projection_base'] = 0.0

            # For renewables/nuclear not in CAMPD: use 2024 generation as 2025 proxy
            # Solar: use 2023 generation (2024 is anomalous)
            if fuel not in EMISSION_RATE and 2025 not in actual_gen_mwh:
                if fuel == 'solar':
                    actual_gen_mwh[2025] = gen_2023
                else:
                    actual_gen_mwh[2025] = gen_2024 if gen_2024 > 0 else gen_2023
                # Geothermal emits CO2 — use average of 2023/2024 for 2025
                if fuel == 'geothermal' and (em_2023 > 0 or em_2024 > 0):
                    em_vals = [v for v in [em_2023, em_2024] if v > 0]
                    actual_co2_mt[2025] = sum(em_vals) / len(em_vals)
                else:
                    actual_co2_mt[2025] = 0.0  # Non-emitting

            # STP nuclear: bought mid-2023, so 2023 is partial year
            # Use 2024 (full year) for forward projections
            # Rosetta already has correct equity-weighted values for both years
            # projection_base is already set to gen_2024 above for non-solar

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
                'retired_year': retired_year,
                # Actual emissions by year (metric tons)
                'actual_co2_mt': actual_co2_mt,
                # Actual generation by year (MWh) + 'projection_base' key
                'actual_gen_mwh': actual_gen_mwh,
                # Derived CO2 rate (t CO2 / MWh) for forward projection
                'co2_rate_t_per_mwh': co2_rate_t_per_mwh,
            })

    # Print summary
    total_em_23 = sum(p['actual_co2_mt'].get(2023, 0) for p in plants)
    total_em_24 = sum(p['actual_co2_mt'].get(2024, 0) for p in plants)
    total_gen_23 = sum(p['actual_gen_mwh'].get(2023, 0) for p in plants)
    total_gen_24 = sum(p['actual_gen_mwh'].get(2024, 0) for p in plants)
    print(f"  Rosetta actuals: 2023={total_em_23/1e6:.2f} MMt CO₂, {total_gen_23/1e6:.1f} TWh")
    print(f"                   2024={total_em_24/1e6:.2f} MMt CO₂, {total_gen_24/1e6:.1f} TWh")
    campd_count = sum(1 for p in plants if 2025 in p['actual_co2_mt'])
    print(f"  2025 data: {campd_count} plants with CAMPD/interpolated actuals")

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

    Priority chain:
      1. User override (plant_emissions_overrides.json) — highest priority
      2. Actual data (Rosetta 2023/2024, CAMPD 2025) — for years with actuals
      3. Projected from 2024 actuals using year_factor/re_growth — for 2030+
      4. Capacity × CF × temporal adjustments (fallback for plants without actuals)
    """
    cap = p['capacity_mw']
    orispl = p.get('orispl')
    actual_gen = p.get('actual_gen_mwh', {})

    # Check user overrides first
    if orispl and fuel in EMISSION_RATE:
        overrides = get_overrides()
        year_overrides = overrides.get(str(year), {})
        if orispl in year_overrides:
            equity = p.get('equity', 1.0)
            return year_overrides[orispl]['net_gen_mwh'] * equity / 1e6  # MWh → TWh

    # Years with actuals: use directly (including 0 — plants may not have generated)
    if year in actual_gen and year != 'projection_base':
        if actual_gen[year] > 0:
            return actual_gen[year] / 1e6  # MWh → TWh
        # Actual data says 0 generation for this year — respect it
        return 0.0

    # Future years (2030+): project from projection_base (2024 actuals, or 2023 for solar)
    proj_base_mwh = actual_gen.get('projection_base', 0)
    if year > 2025:
        # Nuclear special case: Crane Clean Energy Center comes online 2027
        # Use CF-based generation for nuclear plants with 0 projection_base
        # (new builds or plants acquired after baseline years)
        if fuel == 'nuclear':
            if proj_base_mwh > 0:
                return proj_base_mwh / 1e6  # Constant from 2024 actuals
            # Crane or other future nuclear: use CF-based gen from online year
            online_year = p.get('online_year')
            if online_year and year >= online_year:
                cf = CAPACITY_FACTORS.get('nuclear', 0.93)
                return cap * cf * 8.760 / 1000.0
            elif not online_year and p['name'] == 'Crane':
                # Legacy fallback: Crane online 2027
                if year >= 2027:
                    cf = CAPACITY_FACTORS.get('nuclear', 0.93)
                    return cap * cf * 8.760 / 1000.0
            return 0.0

        if proj_base_mwh > 0:
            if fuel in ('wind', 'solar'):
                return proj_base_mwh / 1e6 * re_growth
            elif fuel in EMISSION_RATE:
                return proj_base_mwh / 1e6 * year_factor
            else:
                return proj_base_mwh / 1e6
        # If projection_base is explicitly 0 (plant has Rosetta data but zero gen),
        # don't fall through to CF fallback — the plant isn't generating
        if 'projection_base' in actual_gen:
            return 0.0

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
    """Get CO₂ emission rate for a plant (t CO₂ / MWh).

    For plants with actuals: uses rate derived from actual emissions/generation.
    Otherwise: uses default EMISSION_RATE by fuel type.
    """
    rate = p.get('co2_rate_t_per_mwh')
    if rate and rate > 0:
        return rate
    return EMISSION_RATE.get(fuel, 0.37)


def compute_fleet_generation(plants, year, scenario='baseline', retired_fuels=None,
                              use_sweep=True):
    """Compute generation by fuel from actual plant data.

    2023/2024: Rosetta actuals. 2025: CAMPD actuals.
    2030+: Projected from 2024 actuals, scaled by sweep P50 CFs when available
    (replaces the old linear 0.8% annual decay with actual market dynamics).
    """
    if retired_fuels is None:
        retired_fuels = {}

    # Rebase year_factor to 2024 for forward projections
    year_factor = max(0.0, 1.0 - 0.008 * (year - 2024))
    re_growth = 1.0 + 0.02 * (year - 2024)

    # Load sweep CFs for market-based fossil generation scaling
    sweep = load_sweep_cf_percentiles() if use_sweep else {}

    # Pre-compute per-ISO/fuel sweep scale factors:
    # sweep_scale = (sweep_p50[year] / sweep_p50[2024]) / year_factor
    # This replaces the linear decay with the sweep's actual CF trajectory.
    sweep_scales = {}
    if sweep and year > 2025 and year_factor > 0.001:
        for iso_key, iso_data in sweep.items():
            year_data = iso_data.get(year, {})
            ref_data = iso_data.get(2024, {})
            for fuel_key, pcts in year_data.items():
                ref_pcts = ref_data.get(fuel_key)
                if (pcts and ref_pcts
                        and pcts['p50'] > 0.001 and ref_pcts['p50'] > 0.001):
                    sweep_scales[(iso_key, fuel_key)] = (
                        pcts['p50'] / ref_pcts['p50'] / year_factor
                    )

    gen_by_fuel = defaultdict(float)
    # Track fossil gen by ISO/fuel for sweep scaling
    gen_by_iso_fuel = defaultdict(float)

    for p in plants:
        fuel = p['fuel_type']
        cap = p['capacity_mw']

        if cap <= 0:
            continue

        # For years with actuals, skip retirement check — actuals already
        # reflect partial-year operations
        actual_gen = p.get('actual_gen_mwh', {})
        has_actual_for_year = year in actual_gen and actual_gen[year] > 0

        if not has_actual_for_year:
            # Check retirement (fleet-wide by fuel type)
            if fuel in retired_fuels and year >= retired_fuels[fuel]:
                continue
            if fuel == 'gas_oil_ct' and 'oil_ct' in retired_fuels and year >= retired_fuels['oil_ct']:
                continue
            # Check plant-specific retirement
            if p.get('retired_year') and year >= p['retired_year']:
                continue

        gen_twh = get_plant_gen_twh(p, fuel, year, year_factor, re_growth)

        # Apply sweep scaling to fossil fuels (non-actual years only)
        if not has_actual_for_year and fuel in EMISSION_RATE and sweep_scales:
            iso = p.get('iso', '')
            sweep_fuel = fuel if fuel != 'gas_oil_ct' else 'gas_ct'
            scale = sweep_scales.get((iso, sweep_fuel))
            if scale is not None:
                gen_twh *= scale

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
            if p.get('retired_year') and year >= p['retired_year']:
                continue
            plant_gen = get_plant_gen_twh(p, 'gas_ccgt', year, year_factor, re_growth)
            # Apply same sweep scaling
            if sweep_scales:
                iso = p.get('iso', '')
                scale = sweep_scales.get((iso, 'gas_ccgt'))
                if scale is not None:
                    plant_gen *= scale
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


def compute_fleet_emissions(plants, year, scenario='baseline', retired_fuels=None,
                             use_sweep=True):
    """Calculate emissions at the plant level using actual data.

    2023/2024: Rosetta actuals (equity-weighted metric tons).
    2025: CAMPD actuals where available, 2023/2024 average otherwise.
    2030+: Projected from 2024 actuals, scaled by sweep P50 CFs when available
    (replaces linear decay with actual market dynamics from 1,215-scenario sweep).
    """
    if retired_fuels is None:
        retired_fuels = {}

    # Rebase year_factor to 2024 for forward projections
    year_factor = max(0.0, 1.0 - 0.008 * (year - 2024))
    re_growth = 1.0 + 0.02 * (year - 2024)

    # Load sweep CFs for market-based fossil scaling
    sweep = load_sweep_cf_percentiles() if use_sweep else {}
    sweep_scales = {}
    if sweep and year > 2025 and year_factor > 0.001:
        for iso_key, iso_data in sweep.items():
            year_data = iso_data.get(year, {})
            ref_data = iso_data.get(2024, {})
            for fuel_key, pcts in year_data.items():
                ref_pcts = ref_data.get(fuel_key)
                if (pcts and ref_pcts
                        and pcts['p50'] > 0.001 and ref_pcts['p50'] > 0.001):
                    sweep_scales[(iso_key, fuel_key)] = (
                        pcts['p50'] / ref_pcts['p50'] / year_factor
                    )

    emissions_by_fuel = defaultdict(float)

    for p in plants:
        fuel = p['fuel_type']
        actual_co2 = p.get('actual_co2_mt', {})
        has_actual_emissions = any(v > 0 for k, v in actual_co2.items() if isinstance(k, int))

        # Skip fuels with no emissions (nuclear, wind, solar, hydro, battery)
        # unless they have actual emissions data (e.g., geothermal)
        if fuel not in EMISSION_RATE and not has_actual_emissions:
            continue
        cap = p['capacity_mw']
        if cap <= 0:
            continue

        # For years with actuals, skip retirement check — actuals already
        # reflect partial-year operations (e.g., Mystic retired May 2024
        # but has actual 2024 emissions for the months it operated)
        has_actual_for_year = year in actual_co2

        if not has_actual_for_year:
            # Check fleet-wide retirement
            if fuel in retired_fuels and year >= retired_fuels[fuel]:
                continue
            if fuel == 'gas_oil_ct' and 'oil_ct' in retired_fuels and year >= retired_fuels['oil_ct']:
                continue
            # Check plant-specific retirement
            if p.get('retired_year') and year >= p['retired_year']:
                continue

        orispl = p.get('orispl')

        # Priority 1: user override
        overrides = get_overrides()
        year_overrides = overrides.get(str(year), {})
        if orispl and orispl in year_overrides:
            equity = p.get('equity', 1.0)
            co2_st = year_overrides[orispl]['co2_short_tons'] * equity
            emissions_by_fuel[fuel] += co2_st * 0.907185 / 1e6  # short tons → Mt metric
            continue

        # Priority 2: actual emissions for years with data
        if has_actual_for_year:
            emissions_by_fuel[fuel] += actual_co2[year] / 1e6  # metric tons → MMt
            continue

        # Priority 3: modeled from generation × emission rate (2030+)
        gen_twh = get_plant_gen_twh(p, fuel, year, year_factor, re_growth)
        co2_rate = get_plant_co2_rate(p, fuel)

        # Apply sweep scaling to fossil fuels
        em = gen_twh * co2_rate
        if fuel in EMISSION_RATE and sweep_scales:
            iso = p.get('iso', '')
            sweep_fuel = fuel if fuel != 'gas_oil_ct' else 'gas_ct'
            scale = sweep_scales.get((iso, sweep_fuel))
            if scale is not None:
                em *= scale
        emissions_by_fuel[fuel] += em

    # CCS: handled by scenario logic
    ccs_emis = 0.0
    if scenario in ('ccs_top_emitters', 'ccs_plus_new_gas', 'retire_peakers_ccs_baseload'):
        for p in plants:
            if p['fuel_type'] != 'gas_ccgt' or p['capacity_mw'] <= 0:
                continue
            if not p['ccs_eligible']:
                continue
            if p.get('retired_year') and year >= p['retired_year']:
                continue
            gen_twh = get_plant_gen_twh(p, 'gas_ccgt', year, year_factor, re_growth)
            # Apply sweep scaling
            if sweep_scales:
                iso = p.get('iso', '')
                scale = sweep_scales.get((iso, 'gas_ccgt'))
                if scale is not None:
                    gen_twh *= scale
            co2_rate = get_plant_co2_rate(p, 'gas_ccgt')
            ccs_ramp = min(1.0, max(0.0, (year - 2028) / 5.0))
            ccs_emis_plant = gen_twh * co2_rate * ccs_ramp * 0.05  # 95% capture
            ccs_emis += ccs_emis_plant
            emissions_by_fuel['gas_ccgt'] -= gen_twh * co2_rate * ccs_ramp
            emissions_by_fuel['gas_ccgt'] = max(0, emissions_by_fuel['gas_ccgt'])

    emissions_by_fuel['ccs_ccgt'] = ccs_emis

    return {k: round(v, 2) for k, v in emissions_by_fuel.items()}


def total_emissions(emissions_by_fuel):
    return round(sum(emissions_by_fuel.values()), 2)


def build_envelope(base_emissions_2023, scenario_emissions_by_year,
                   sweep_emissions_by_year=None):
    """Build P10/P50/P90 envelope from sweep-derived percentiles.

    If sweep_emissions_by_year is provided (from compute_fleet_emissions_sweep),
    uses real P10/P50/P90 from 1,215 market scenarios. Otherwise falls back to
    synthetic ±12% spread.
    """
    envelope = {}
    envelope_by_cost = {'Low': {}, 'Medium': {}, 'High': {}}

    for year in YEARS:
        e = scenario_emissions_by_year.get(year, base_emissions_2023)

        if sweep_emissions_by_year and year in sweep_emissions_by_year:
            sw = sweep_emissions_by_year[year]
            p10 = round(sw['p10'], 1)
            p50 = round(sw['p50'], 1)
            p90 = round(sw['p90'], 1)
        else:
            # Fallback: synthetic ±12% for years outside sweep range
            spread = 0.12
            p50 = round(e, 1)
            p10 = round(p50 * (1 - spread), 1)
            p90 = round(p50 * (1 + spread), 1)

        envelope[str(year)] = {'p10': p10, 'p50': p50, 'p90': p90}

        for cost_level, factor in [('Low', 0.95), ('Medium', 1.0), ('High', 1.05)]:
            adj_p50 = round(p50 * factor, 1)
            adj_p10 = round(p10 * factor, 1)
            adj_p90 = round(p90 * factor, 1)
            envelope_by_cost[cost_level][str(year)] = {
                'p10': adj_p10, 'p50': adj_p50, 'p90': adj_p90
            }

    return envelope, envelope_by_cost


def build_plant_detail(fossil_plants, year, scenario='baseline', retired_fuels=None,
                       all_plants=None):
    """Build plant-level detail using actuals for 2023-2025, projected for future.

    Includes fossil plants (with emissions) AND nuclear/geothermal plants
    (for generation drill-down debugging).
    """
    if retired_fuels is None:
        retired_fuels = {}

    year_factor = max(0.0, 1.0 - 0.008 * (year - 2024))
    re_growth = 1.0 + 0.02 * (year - 2024)
    details = []

    # ── Nuclear & geothermal plants (non-fossil with generation) ──
    if all_plants:
        for p in all_plants:
            fuel = p['fuel_type']
            if fuel not in ('nuclear', 'geothermal'):
                continue
            name = p['name']
            cap = p['capacity_mw']
            orispl = p.get('orispl') or hash(name) % 90000 + 10000
            if cap <= 0:
                continue

            gen_twh = get_plant_gen_twh(p, fuel, year, year_factor, re_growth)
            # Geothermal has small emissions; nuclear has 0
            actual_co2 = p.get('actual_co2_mt', {})
            if year in actual_co2:
                emissions_mt = actual_co2[year] / 1e6
            else:
                emissions_mt = 0.0

            details.append({
                'name': name, 'orispl': orispl, 'fuel_type': fuel,
                'capacity_mw': round(cap, 1),
                'ccs_capacity_mw': 0.0,
                'gen_twh': round(gen_twh, 3),
                'emissions_mt': round(emissions_mt, 3),
                'status': 'operating'
            })

    # ── Fossil plants ──
    for p in fossil_plants:
        fuel = p['fuel_type']
        name = p['name']
        cap = p['capacity_mw']
        orispl = p['orispl'] or hash(name) % 90000 + 10000

        # Check retirement — skip for years with actuals (partial-year ops captured)
        actual_co2 = p.get('actual_co2_mt', {})
        has_actual_for_year = year in actual_co2

        is_retired = False
        if not has_actual_for_year:
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

        # Priority 1: user override
        overrides = get_overrides()
        year_overrides = overrides.get(str(year), {})
        if orispl and orispl in year_overrides:
            equity = p.get('equity', 1.0)
            ov = year_overrides[orispl]
            gen_twh = ov['net_gen_mwh'] * equity / 1e6
            emissions_mt = ov['co2_short_tons'] * equity * 0.907185 / 1e6
        else:
            # Use get_plant_gen_twh which handles actuals + projection
            gen_twh = get_plant_gen_twh(p, fuel, year, year_factor, re_growth)
            # Emissions: use actuals if available, otherwise derive from gen × rate
            actual_co2 = p.get('actual_co2_mt', {})
            if year in actual_co2:
                emissions_mt = actual_co2[year] / 1e6  # metric tons → MMt
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


def build_plant_percentiles(plant_detail):
    """Build per-plant P10/P50/P90 bands for drill-down charts.

    Uses ±12% spread around point estimates (same as fleet-level envelope).
    Output keyed by orispl → {name, fuel_type, capacity_mw, gen_gwh, emissions_t}
    where gen_gwh[year] and emissions_t[year] each have {p10, p50, p90}.
    Units: GWh for generation, metric tons for emissions.
    """
    SPREAD = 0.12

    # Collect all unique plants across years
    plant_map = {}  # orispl → {name, fuel_type, ...}
    for year_str, plants_list in plant_detail.items():
        for p in plants_list:
            oid = p['orispl']
            if oid not in plant_map:
                plant_map[oid] = {
                    'name': p['name'],
                    'fuel_type': p['fuel_type'],
                    'capacity_mw': p['capacity_mw'],
                    'gen_gwh': {},
                    'emissions_t': {}
                }
            # gen_twh → GWh (×1000), emissions_mt → metric tons (×1e6)
            gen_gwh = p['gen_twh'] * 1000
            emis_t = p['emissions_mt'] * 1e6
            plant_map[oid]['gen_gwh'][year_str] = {
                'p10': round(gen_gwh * (1 - SPREAD), 1),
                'p50': round(gen_gwh, 1),
                'p90': round(gen_gwh * (1 + SPREAD), 1)
            }
            plant_map[oid]['emissions_t'][year_str] = {
                'p10': round(emis_t * (1 - SPREAD)),
                'p50': round(emis_t),
                'p90': round(emis_t * (1 + SPREAD))
            }

    return plant_map


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
            fossil_plants, year, scenario=scenario_key, retired_fuels=retired_fuels,
            all_plants=plants
        )
        plant_detail[str(year)] = pd

    # Compute sweep-derived P10/P50/P90 emissions per year
    sweep_emissions = {}
    for year in YEARS:
        sw = compute_fleet_emissions_sweep(
            plants, year, scenario=scenario_key, retired_fuels=retired_fuels,
            base_emissions_mt=emissions_trajectory.get(year)
        )
        if sw is not None:
            sweep_emissions[year] = sw

    base_e = emissions_trajectory[2023]
    envelope, envelope_by_cost = build_envelope(
        base_e, emissions_trajectory, sweep_emissions_by_year=sweep_emissions
    )

    # ── Build per-plant percentile bands for drill-down charts ──
    # Uses ±12% spread (same as fleet-level envelope) applied per-plant
    plant_percentiles = build_plant_percentiles(plant_detail)

    return {
        'description': descriptions.get(scenario_key, scenario_key),
        'color': colors.get(scenario_key, '#888'),
        'envelope': envelope,
        'envelope_by_fossil_cost': envelope_by_cost,
        'plant_detail': plant_detail,
        'plant_percentiles': plant_percentiles,
        'generation_by_fuel': generation_by_fuel,
        'emissions_by_fuel': emissions_by_fuel_data
    }


def build_targets(baseline_emissions_2023):
    """Build target trajectories."""
    base = baseline_emissions_2023

    sbti = {
        '2023': round(base, 1),
        '2024': round(base * 0.925, 1),
        '2025': round(base * 0.85, 1),
        '2030': round(base * 0.50, 1),
        '2035': round(base * 0.25, 1),
        '2040': 0.0,
        '2045': 0.0,
        '2050': 0.0
    }

    at = {
        '2023': round(base, 1),
        '2024': round(base * 0.94, 1),
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
                f'Rebuilt from Rosetta CSV (2023/2024 actuals) + CAMPD 2025. '
                f'Nuclear: {fleet_summary.get("nuclear", {}).get("capacity_mw", 0):,.0f} MW, '
                f'Gas CCGT: {fleet_summary.get("gas_ccgt", {}).get("capacity_mw", 0):,.0f} MW, '
                f'Total fleet: {total_cap:,.0f} MW. '
                'Envelope P10/P90 derived from 1,215-scenario market sweep CF percentiles.'
            ),
        }
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, 'fleet_scenario_results_sample.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nOutput written to: {output_path}")
    print(f"File size: {os.path.getsize(output_path):,} bytes")

    # Also copy to dashboard/data/ for fleet_scenarios.html
    dashboard_data_dir = os.path.join(SCRIPT_DIR, '..', '..', 'dashboard', 'data')
    if os.path.isdir(dashboard_data_dir):
        dashboard_path = os.path.join(dashboard_data_dir,
                                       'fleet_scenario_results_sample.json')
        with open(dashboard_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Dashboard copy: {dashboard_path}")
    else:
        print(f"  WARNING: dashboard/data/ not found — skipping dashboard copy")


if __name__ == '__main__':
    main()
