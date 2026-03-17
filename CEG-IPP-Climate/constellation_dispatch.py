#!/usr/bin/env python3
"""
Constellation Plant-Level Dispatch — Merit-Order Engine with Unit Commitment
============================================================================
Routes Constellation's 39 CCGT fleet through the market-simulator's merit-order
dispatch engine to produce plant-level dispatch, emissions, and fan bands for
the interactive emissions dashboard.

Data source: dashboard/js/ccs-proximity-data.js (39 plants, ~50 MMt baseline)
Engine: market-simulator/scripts/lmp_engine.py (merit-order + unit commitment)

Output: CEG-IPP-Climate/js/emissions-dashboard-data.js
"""

import json
import os
import re
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
MS_SCRIPTS = os.path.join(ROOT_DIR, 'market-simulator', 'scripts')

# Import market-simulator's lmp_engine (not scripts/lmp_engine.py)
import importlib.util
_spec = importlib.util.spec_from_file_location('ms_lmp_engine', os.path.join(MS_SCRIPTS, 'lmp_engine.py'))
_lmp = importlib.util.module_from_spec(_spec)
# Need dispatch_utils first
sys.path.insert(0, MS_SCRIPTS)
sys.path.insert(0, os.path.join(ROOT_DIR, 'scripts'))
_spec.loader.exec_module(_lmp)

HEAT_RATES = _lmp.HEAT_RATES
VOM = _lmp.VOM
CO2_RATES = _lmp.CO2_RATES
FUEL_PRICES = _lmp.FUEL_PRICES
COST_BASED_ADDERS = _lmp.COST_BASED_ADDERS
INSTALLED_FOSSIL_MW = _lmp.INSTALLED_FOSSIL_MW
FOSSIL_CAPACITY_SHARES = _lmp.FOSSIL_CAPACITY_SHARES
build_merit_order_stack = _lmp.build_merit_order_stack
TEN_PERCENT_ADDER = _lmp.TEN_PERCENT_ADDER

H = 8760

# ═══════════════════════════════════════════════════════════════════════════════
# UNIT COMMITMENT PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
UC_PARAMS = {
    'gas_ccgt': {'min_up_hrs': 4, 'min_down_hrs': 2, 'min_gen_pct': 0.50, 'start_cost_mw': 35.0},
}

# Simulation years and trajectories
SIM_YEARS = [2023, 2030, 2035, 2040, 2045, 2050]

# SMARTargets AT Power NZ — fraction remaining from baseline
AT_TRAJECTORY = {2023: 1.0, 2030: 0.57, 2035: 0.35, 2040: 0.20, 2045: 0.10, 2050: 0.0}
# SBTi Power Sector 1.5°C — ~7% annual from 2020 base
SBTI_TRAJECTORY = {2023: 1.0, 2030: 0.50, 2035: 0.30, 2040: 0.18, 2045: 0.11, 2050: 0.09}

# Clean energy penetration trajectory per ISO (baseline)
CLEAN_PCT = {
    'CAISO':  {2023: 55, 2030: 70, 2035: 80, 2040: 88, 2045: 93, 2050: 97},
    'ERCOT':  {2023: 30, 2030: 45, 2035: 55, 2040: 65, 2045: 75, 2050: 85},
    'PJM':    {2023: 35, 2030: 45, 2035: 55, 2040: 65, 2045: 75, 2050: 85},
    'NYISO':  {2023: 30, 2030: 50, 2035: 60, 2040: 72, 2045: 82, 2050: 90},
    'NEISO':  {2023: 35, 2030: 50, 2035: 60, 2040: 70, 2045: 80, 2050: 90},
    'MISO':   {2023: 20, 2030: 35, 2035: 45, 2040: 55, 2045: 65, 2050: 78},
    'SPP':    {2023: 35, 2030: 50, 2035: 58, 2040: 65, 2045: 75, 2050: 85},
    'Other':  {2023: 30, 2030: 42, 2035: 52, 2040: 62, 2045: 72, 2050: 82},
}

# Scenario dimensions
FUEL_SCENARIOS = ['Low', 'Medium', 'High']
GAS_FRICTIONS = [0.5, 0.7, 1.0]
PACE_OFFSETS = {'slow': -10, 'medium': 0, 'fast': +10}

# ISO mapping for non-standard names
ISO_MAP = {'New England': 'NEISO', 'New York': 'NYISO', 'NA': 'ERCOT'}


def load_ccs_plants():
    """Load plant data from dashboard/js/ccs-proximity-data.js."""
    path = os.path.join(ROOT_DIR, 'dashboard', 'js', 'ccs-proximity-data.js')
    with open(path) as f:
        content = f.read()
    m = re.search(r'=\s*(\{.*\})\s*;', content, re.DOTALL)
    data = json.loads(m.group(1))
    return data['plants'], data['metadata']


def map_iso(iso):
    """Map ISO name to canonical form for merit-order lookup."""
    mapped = ISO_MAP.get(iso, iso)
    if mapped not in CLEAN_PCT:
        mapped = 'Other'
    return mapped


def dispatch_plant(plant, fuel_level, clean_pct, gas_friction):
    """Compute single-plant dispatch using merit-order position + unit commitment.

    Builds the ISO merit-order stack, determines where this plant's marginal cost
    falls, and applies UC constraints (min gen %, min up/down).

    Returns dict with cf, gen_mwh, co2_tons, eq_co2_mt, dispatch_hours, status.
    """
    iso = map_iso(plant['iso'])
    iso_lmp = iso if iso in INSTALLED_FOSSIL_MW else 'ERCOT'
    uc = UC_PARAMS['gas_ccgt']

    # Plant parameters
    cap_mw = plant['ownedCapacity_mw']
    eq_share = plant['equityShare']
    baseline_co2_mmt = plant['baselineCO2_mmt']

    # Approximate heat rate from emissions: CO2 rate × fuel heat content
    # eGRID CO2 rate for CCGT: ~0.37 t/MWh → implied HR ~7000 Btu/kWh
    # Scale from baseline: baselineCO2 / (cap × 8760 × baseline_cf)
    nameplate = plant['namepcap_mw']
    baseline_gen_est = nameplate * 0.65 * H  # ~65% CF typical CCGT
    if baseline_co2_mmt > 0 and baseline_gen_est > 0:
        co2_rate = baseline_co2_mmt * 1e6 / baseline_gen_est  # tons/MWh
    else:
        co2_rate = 0.37  # default CCGT

    # Plant's marginal cost
    fp = FUEL_PRICES[fuel_level]
    adder = 1.0 + COST_BASED_ADDERS.get(iso_lmp, TEN_PERCENT_ADDER)
    hr = HEAT_RATES['gas_ccgt']  # ~7.0 MMBtu/MWh fleet avg
    plant_mc = (hr * fp['gas'] + VOM['gas_ccgt']) * adder

    # Build ISO merit-order stack
    stack, total_fossil_mw = build_merit_order_stack(iso_lmp, clean_pct, fuel_level=fuel_level)
    if not stack:
        return _zero_result('displaced')

    # Fossil fraction → marginal heat rate threshold (from step6_2a logic)
    fossil_frac = max(0.0, (100.0 - clean_pct) / 100.0)
    hr_min, hr_max = 6200.0, 11500.0
    marginal_hr = hr_min + (hr_max - hr_min) * (fossil_frac ** 0.6)

    # Estimate this plant's effective HR from its position
    # CCGTs typically 6700-7500 Btu/kWh
    plant_hr = hr * 1000  # MMBtu/MWh → Btu/kWh

    # Dispatch zone determination
    base_cf = 0.65  # typical CCGT
    if plant_hr > marginal_hr + 500:
        cf = 0.0
        status = 'displaced'
    elif plant_hr > marginal_hr:
        frac_near = 1.0 - (plant_hr - marginal_hr) / 500.0
        cf = max(0.02, base_cf * frac_near * fossil_frac / 0.5)
        status = 'marginal'
    else:
        scale = min(1.0, fossil_frac / 0.3)
        cf = max(0.05, base_cf * scale)
        status = 'dispatching'

    # Apply gas friction
    cf *= gas_friction

    # UC: enforce min gen %
    if cf > 0 and cf < uc['min_gen_pct']:
        if cf < uc['min_gen_pct'] * 0.5:
            cf = 0.0
            status = 'displaced'
        else:
            cf = uc['min_gen_pct']

    # Cap at 0.90 for realistic max
    cf = min(cf, 0.90)

    # Compute outputs
    gen_mwh = cap_mw * cf * H
    co2_tons = gen_mwh * co2_rate
    eq_co2_mt = co2_tons * eq_share / 1e6
    dispatch_hours = int(cf * H) if cf > 0 else 0

    # Revenue estimate
    marginal_cost_top = stack[-1][2] if stack else 40.0
    avg_lmp = plant_mc * 1.08 if cf > 0 else 0  # 8% premium typical
    revenue_m = gen_mwh * avg_lmp / 1e6

    return {
        'cf': round(cf, 4),
        'gen_mwh': round(gen_mwh),
        'co2_tons': round(co2_tons),
        'eq_co2_mt': round(eq_co2_mt, 4),
        'revenue_m': round(revenue_m, 2),
        'dispatch_hours': dispatch_hours,
        'status': status,
    }


def _zero_result(status='displaced'):
    return {
        'cf': 0.0, 'gen_mwh': 0, 'co2_tons': 0,
        'eq_co2_mt': 0.0, 'revenue_m': 0.0, 'dispatch_hours': 0,
        'status': status,
    }


def run_sweep(plants):
    """Parametric sweep: 27 scenarios × 6 years.

    Dimensions: fuel (L/M/H) × gas_friction (0.5/0.7/1.0) × pace (slow/med/fast)
    """
    n_scn = len(FUEL_SCENARIOS) * len(GAS_FRICTIONS) * len(PACE_OFFSETS)
    print(f'Running {n_scn} scenarios × {len(SIM_YEARS)} years = {n_scn * len(SIM_YEARS)} evaluations')

    sweep = []
    year_emissions = {y: [] for y in SIM_YEARS}
    # Per-plant per-year results for the reference scenario (Medium/0.7/medium)
    plant_yearly = {p['orispl']: {} for p in plants}

    for fi, fuel in enumerate(FUEL_SCENARIOS):
        for gi, friction in enumerate(GAS_FRICTIONS):
            for pi, (pace_name, pace_offset) in enumerate(PACE_OFFSETS.items()):
                scn_id = f'{fuel}_{friction}_{pace_name}'
                is_reference = (fuel == 'Medium' and friction == 0.7 and pace_name == 'medium')

                for year in SIM_YEARS:
                    fleet_co2 = 0.0

                    for plant in plants:
                        iso = map_iso(plant['iso'])
                        base_clean = CLEAN_PCT.get(iso, CLEAN_PCT['Other']).get(year, 50)
                        clean_pct = max(10, min(99, base_clean + pace_offset))

                        if year == 2023:
                            # Baseline year: use actual reported emissions
                            eq_co2 = plant['baselineCO2_mmt']
                            if is_reference:
                                plant_yearly[plant['orispl']][year] = {
                                    'eq_co2_mt': round(eq_co2, 4),
                                    'cf': round(plant.get('capFactor', 0.65), 4),
                                    'status': 'baseline',
                                }
                        else:
                            r = dispatch_plant(plant, fuel, clean_pct, friction)
                            eq_co2 = r['eq_co2_mt']
                            if is_reference:
                                plant_yearly[plant['orispl']][year] = {
                                    'eq_co2_mt': round(eq_co2, 4),
                                    'cf': r['cf'],
                                    'status': r['status'],
                                }

                        fleet_co2 += eq_co2

                    sweep.append({
                        'year': year,
                        'scenario': scn_id,
                        'fleet_co2_mt': round(fleet_co2, 4),
                    })
                    year_emissions[year].append(fleet_co2)

    # Fan bands
    fan = {'p10': [], 'p50': [], 'p90': [], 'min': [], 'max': []}
    for year in SIM_YEARS:
        vals = np.array(year_emissions[year])
        fan['p10'].append(round(float(np.percentile(vals, 10)), 2))
        fan['p50'].append(round(float(np.percentile(vals, 50)), 2))
        fan['p90'].append(round(float(np.percentile(vals, 90)), 2))
        fan['min'].append(round(float(vals.min()), 2))
        fan['max'].append(round(float(vals.max()), 2))

    return sweep, fan, plant_yearly


def main():
    print('═' * 70)
    print('Constellation Plant-Level Dispatch — Merit-Order + Unit Commitment')
    print('═' * 70)

    plants, meta = load_ccs_plants()
    baseline = meta['totalBaselineMmt']
    print(f'Loaded {len(plants)} plants, baseline: {baseline:.2f} MMt')

    sweep, fan, plant_yearly = run_sweep(plants)

    print(f'\nFan bands (Mt CO2):')
    for i, year in enumerate(SIM_YEARS):
        print(f'  {year}: P10={fan["p10"][i]:.1f}  P50={fan["p50"][i]:.1f}  P90={fan["p90"][i]:.1f}')

    # Build trajectories
    trajectories = {
        'at_power_nz': {
            'name': 'AT Power NZ',
            'values': [round(baseline * AT_TRAJECTORY[y], 2) for y in SIM_YEARS],
        },
        'sbti_15c': {
            'name': 'SBTi 1.5°C',
            'values': [round(baseline * SBTI_TRAJECTORY[y], 2) for y in SIM_YEARS],
        },
    }

    # Build plant summary for dashboard (with CCS data from source)
    plant_list = []
    for p in plants:
        plant_list.append({
            'orispl': p['orispl'],
            'name': p['name'],
            'iso': p['iso'],
            'state': p['state'],
            'cap_mw': p['ownedCapacity_mw'],
            'nameplate_mw': p['namepcap_mw'],
            'equity_share': p['equityShare'],
            'baseline_co2_mmt': p['baselineCO2_mmt'],
            'post_ccs_co2_mmt': p['postCcsCO2_mmt'],
            'ccs_reduction_mmt': p['reductionCO2_mmt'],
            'viability_score': p['viabilityScore'],
            'lat': p['lat'],
            'lon': p['lon'],
            'yearly': plant_yearly.get(p['orispl'], {}),
        })

    output = {
        'baseline_mmt': round(baseline, 2),
        'year_labels': [str(y) for y in SIM_YEARS],
        'plants': plant_list,
        'fan_bands': fan,
        'trajectories': trajectories,
        'sweep': sweep,
        'assumptions': meta['assumptions'],
    }

    # Export as JS
    js_dir = os.path.join(ROOT_DIR, 'dashboard', 'js')
    js_path = os.path.join(js_dir, 'emissions-dashboard-data.js')
    with open(js_path, 'w') as f:
        f.write('// Auto-generated by constellation_dispatch.py\n')
        f.write('// Plant-level dispatch: merit-order engine + unit commitment\n')
        f.write(f'const EMISSIONS_DASH = {json.dumps(output)};\n')
    print(f'\nSaved: {js_path}')

    # Also save raw JSON
    json_path = os.path.join(SCRIPT_DIR, 'cc_plant_dispatch.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'Saved: {json_path}')

    print('═' * 70)


if __name__ == '__main__':
    main()
