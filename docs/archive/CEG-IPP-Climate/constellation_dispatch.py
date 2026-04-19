#!/usr/bin/env python3
"""
Constellation Plant-Level Dispatch — Merit-Order Engine with Unit Commitment
============================================================================
Routes Constellation's 39 CCGT fleet through the market-simulator's merit-order
dispatch engine to produce plant-level dispatch, emissions, and fan bands.

CRITICAL: Fan bands use REFERENCE sweep only (not power_nz or economy_nz).
The reference sweep represents market-driven decarbonization without net-zero
policy mandates — this is the qualified target basis.

Data sources:
  - dashboard/js/ccs-proximity-data.js (39 plants, ~50 MMt baseline)
  - data/step6-smartargets/sweep_reference_{ISO}.parquet (540 scenarios × 7 ISOs)
  - market-simulator/scripts/lmp_engine.py (merit-order dispatch engine)

Output: dashboard/js/emissions-dashboard-data.js
"""

import json
import os
import re
import sys
import glob
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
MS_SCRIPTS = os.path.join(ROOT_DIR, 'market-simulator', 'scripts')

# Import market-simulator's lmp_engine (not scripts/lmp_engine.py)
import importlib.util
_spec = importlib.util.spec_from_file_location(
    'ms_lmp_engine', os.path.join(MS_SCRIPTS, 'lmp_engine.py'))
_lmp = importlib.util.module_from_spec(_spec)
sys.path.insert(0, MS_SCRIPTS)
sys.path.insert(0, os.path.join(ROOT_DIR, 'scripts'))
_spec.loader.exec_module(_lmp)

HEAT_RATES = _lmp.HEAT_RATES
VOM = _lmp.VOM
FUEL_PRICES = _lmp.FUEL_PRICES
COST_BASED_ADDERS = _lmp.COST_BASED_ADDERS
INSTALLED_FOSSIL_MW = _lmp.INSTALLED_FOSSIL_MW
build_merit_order_stack = _lmp.build_merit_order_stack
TEN_PERCENT_ADDER = _lmp.TEN_PERCENT_ADDER

H = 8760

# ═══════════════════════════════════════════════════════════════════════════════
# UNIT COMMITMENT PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
UC_MIN_GEN_PCT = 0.50   # CCGT minimum stable generation
UC_MIN_UP_HRS = 4
UC_MIN_DOWN_HRS = 2

# Simulation years
SIM_YEARS = [2023, 2030, 2035, 2040, 2045, 2050]

# SMARTargets trajectory fractions remaining from baseline
AT_TRAJECTORY = {2023: 1.0, 2030: 0.57, 2035: 0.35, 2040: 0.20, 2045: 0.10, 2050: 0.0}
SBTI_TRAJECTORY = {2023: 1.0, 2030: 0.50, 2035: 0.30, 2040: 0.18, 2045: 0.11, 2050: 0.09}

# ISO mapping for CCS proximity data → sweep data ISOs
ISO_MAP = {
    'Other': None,  # Plants in non-ISO territory — use ERCOT as proxy
}


def load_ccs_plants():
    """Load plant data from dashboard/js/ccs-proximity-data.js."""
    path = os.path.join(ROOT_DIR, 'dashboard', 'js', 'ccs-proximity-data.js')
    with open(path) as f:
        content = f.read()
    m = re.search(r'=\s*(\{.*\})\s*;', content, re.DOTALL)
    data = json.loads(m.group(1))
    return data['plants'], data['metadata']


def load_reference_sweep():
    """Load ONLY the reference sweep parquets (not power_nz or economy_nz).

    Returns DataFrame with columns: scenario, iso, year, clean_pct, avg_lmp,
    gas_friction, conditions, demand_growth, price_sens, ...
    """
    sweep_dir = os.path.join(ROOT_DIR, 'data', 'step6-smartargets')
    ref_files = sorted(glob.glob(os.path.join(sweep_dir, 'sweep_reference_*.parquet')))

    if not ref_files:
        print('ERROR: No reference sweep parquets found')
        sys.exit(1)

    frames = [pd.read_parquet(f) for f in ref_files]
    df = pd.concat(frames, ignore_index=True)
    print(f'  Loaded reference sweep: {len(df)} rows, '
          f'{df["scenario"].nunique()} scenarios, '
          f'{sorted(df["iso"].unique())} ISOs')
    return df


def map_plant_iso(plant_iso):
    """Map plant ISO name to sweep ISO name."""
    if plant_iso in ('CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP'):
        return plant_iso
    return 'ERCOT'  # Proxy for non-ISO territory


def dispatch_plant(plant, clean_pct, avg_lmp, gas_friction, fuel_level='Medium'):
    """Compute single-plant dispatch using merit-order position + unit commitment.

    Args:
        plant: dict from CCS proximity data
        clean_pct: clean energy % for this plant's ISO in this scenario/year
        avg_lmp: average LMP ($/MWh) for this ISO in this scenario/year
        gas_friction: gas permitting friction factor (0.3-1.0)
        fuel_level: fuel price scenario for marginal cost calc

    Returns dict with cf, gen_mwh, co2_tons, eq_co2_mt, status
    """
    iso_lmp = map_plant_iso(plant['iso'])
    if iso_lmp not in INSTALLED_FOSSIL_MW:
        iso_lmp = 'ERCOT'

    cap_mw = plant['ownedCapacity_mw']
    eq_share = plant['equityShare']
    baseline_co2_mmt = plant['baselineCO2_mmt']

    # Implied CO2 rate from eGRID baseline
    nameplate = plant['namepcap_mw']
    baseline_gen_est = nameplate * 0.65 * H  # ~65% CF typical CCGT
    if baseline_co2_mmt > 0 and baseline_gen_est > 0:
        co2_rate = baseline_co2_mmt * 1e6 / baseline_gen_est  # tons/MWh
    else:
        co2_rate = 0.37

    # Build ISO merit-order stack to get marginal cost context
    stack, total_fossil_mw = build_merit_order_stack(
        iso_lmp, clean_pct, fuel_level=fuel_level)

    if not stack:
        return _zero('displaced')

    # Fossil fraction → marginal heat rate threshold
    fossil_frac = max(0.0, (100.0 - clean_pct) / 100.0)
    hr_min, hr_max = 6200.0, 11500.0
    marginal_hr = hr_min + (hr_max - hr_min) * (fossil_frac ** 0.6)

    # CCGT typical HR: 7000 Btu/kWh
    plant_hr = HEAT_RATES['gas_ccgt'] * 1000  # 7000 Btu/kWh
    base_cf = 0.65

    # Dispatch zone
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
    if 0 < cf < UC_MIN_GEN_PCT:
        if cf < UC_MIN_GEN_PCT * 0.5:
            cf = 0.0
            status = 'displaced'
        else:
            cf = UC_MIN_GEN_PCT

    cf = min(cf, 0.90)

    gen_mwh = cap_mw * cf * H
    co2_tons = gen_mwh * co2_rate
    eq_co2_mt = co2_tons * eq_share / 1e6
    revenue_m = gen_mwh * avg_lmp / 1e6 if cf > 0 else 0.0

    return {
        'cf': round(cf, 4),
        'gen_mwh': round(gen_mwh),
        'co2_tons': round(co2_tons),
        'eq_co2_mt': round(eq_co2_mt, 4),
        'revenue_m': round(revenue_m, 2),
        'dispatch_hours': int(cf * H) if cf > 0 else 0,
        'status': status,
    }


def _zero(status='displaced'):
    return {
        'cf': 0.0, 'gen_mwh': 0, 'co2_tons': 0,
        'eq_co2_mt': 0.0, 'revenue_m': 0.0, 'dispatch_hours': 0,
        'status': status,
    }


def run_reference_sweep(plants, sweep_df):
    """Run all 39 plants through the 540 reference scenarios × 6 years.

    For each (scenario, year):
      1. Look up clean_pct and avg_lmp per ISO from sweep_df
      2. Dispatch each plant through merit-order with UC constraints
      3. Sum fleet equity-adjusted CO2

    Returns fan bands computed from REFERENCE sweep only.
    """
    scenarios = sweep_df['scenario'].unique()
    n_scn = len(scenarios)
    print(f'  Dispatching {len(plants)} plants × {n_scn} scenarios × {len(SIM_YEARS)} years')

    # Build lookup: (scenario, iso, year) → {clean_pct, avg_lmp, gas_friction}
    lookup = {}
    for _, row in sweep_df.iterrows():
        key = (row['scenario'], row['iso'], int(row['year']))
        lookup[key] = {
            'clean_pct': float(row['clean_pct']),
            'avg_lmp': float(row.get('avg_lmp', 30.0)),
            'gas_friction': float(row.get('gas_friction', 0.7)),
        }

    year_emissions = {y: [] for y in SIM_YEARS}
    # Per-plant per-year emissions across ALL scenarios (for percentile computation)
    # plant_year_emissions[orispl][year_idx] = list of eq_co2_mt across scenarios
    plant_year_emissions = {
        p['orispl']: {yi: [] for yi in range(len(SIM_YEARS))}
        for p in plants
    }

    for si, scenario in enumerate(scenarios):
        if si % 100 == 0:
            print(f'    Scenario {si+1}/{n_scn}...')

        for yi, year in enumerate(SIM_YEARS):
            fleet_co2 = 0.0

            for plant in plants:
                plant_iso = map_plant_iso(plant['iso'])

                if year == 2023:
                    eq_co2 = plant['baselineCO2_mmt']
                else:
                    scn_data = lookup.get((scenario, plant_iso, year))
                    if scn_data is None:
                        eq_co2 = plant['baselineCO2_mmt'] * 0.8
                    else:
                        r = dispatch_plant(
                            plant,
                            clean_pct=scn_data['clean_pct'],
                            avg_lmp=scn_data['avg_lmp'],
                            gas_friction=scn_data['gas_friction'],
                        )
                        eq_co2 = r['eq_co2_mt']

                fleet_co2 += eq_co2
                plant_year_emissions[plant['orispl']][yi].append(eq_co2)

            year_emissions[year].append(fleet_co2)

    # Fan bands from reference sweep ONLY
    fan = {'p10': [], 'p50': [], 'p90': [], 'min': [], 'max': []}
    for year in SIM_YEARS:
        vals = np.array(year_emissions[year])
        fan['p10'].append(round(float(np.percentile(vals, 10)), 2))
        fan['p50'].append(round(float(np.percentile(vals, 50)), 2))
        fan['p90'].append(round(float(np.percentile(vals, 90)), 2))
        fan['min'].append(round(float(vals.min()), 2))
        fan['max'].append(round(float(vals.max()), 2))

    # Per-plant per-year P10/P50/P90 emissions (for CCS delta calculation)
    plant_fan = {}
    for p in plants:
        orispl = p['orispl']
        pf = {'p10': [], 'p50': [], 'p90': []}
        for yi in range(len(SIM_YEARS)):
            vals = np.array(plant_year_emissions[orispl][yi])
            pf['p10'].append(round(float(np.percentile(vals, 10)), 4))
            pf['p50'].append(round(float(np.percentile(vals, 50)), 4))
            pf['p90'].append(round(float(np.percentile(vals, 90)), 4))
        plant_fan[orispl] = pf

    return fan, plant_fan


def main():
    print('═' * 70)
    print('Constellation Plant-Level Dispatch — Reference Sweep Only')
    print('═' * 70)

    plants, meta = load_ccs_plants()
    baseline = meta['totalBaselineMmt']
    print(f'\n  {len(plants)} plants, baseline: {baseline:.2f} MMt')

    print('\nLoading reference sweep parquets...')
    sweep_df = load_reference_sweep()

    print('\nRunning dispatch...')
    fan, plant_fan = run_reference_sweep(plants, sweep_df)

    print(f'\nFan bands — REFERENCE SWEEP ONLY (Mt CO2):')
    for i, year in enumerate(SIM_YEARS):
        pct_chg = (fan['p50'][i] - baseline) / baseline * 100
        print(f'  {year}: P10={fan["p10"][i]:6.1f}  P50={fan["p50"][i]:6.1f}  '
              f'P90={fan["p90"][i]:6.1f}  (P50 Δ={pct_chg:+.0f}%)')

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

    # Build plant list with per-year sweep emissions + CCS parameters
    # CCS delta is computed client-side:
    #   delta[year] = sweep_emissions[year] - ccs_emissions
    #   ccs_emissions = cap_mw * 0.80 * 8760 * co2_rate_t_per_mwh * 1.14 * 0.05 * equity / 1e6
    plant_list = []
    for p in plants:
        orispl = p['orispl']
        cap = p['ownedCapacity_mw']
        nameplate = p['namepcap_mw']
        eq = p['equityShare']
        baseline_mmt = p['baselineCO2_mmt']

        # Compute CO2 rate (t/MWh) from eGRID baseline for CCS math
        baseline_gen_est = nameplate * 0.65 * H
        co2_rate = baseline_mmt * 1e6 / baseline_gen_est if baseline_gen_est > 0 else 0.37

        # CCS emissions at 80% CF, 95% capture, +14% HR penalty (equity-adjusted, MMt)
        ccs_co2_mmt = cap * 0.80 * H * co2_rate * 1.14 * 0.05 * eq / 1e6

        plant_list.append({
            'orispl': orispl,
            'name': p['name'],
            'iso': p['iso'],
            'state': p['state'],
            'lat': p['lat'],
            'lon': p['lon'],
            'cap_mw': cap,
            'nameplate_mw': nameplate,
            'equity_share': eq,
            'baseline_co2_mmt': baseline_mmt,
            'co2_rate_t_mwh': round(co2_rate, 5),
            'ccs_co2_mmt': round(ccs_co2_mmt, 4),
            'viability_score': p['viabilityScore'],
            # Per-year sweep emissions (P10/P50/P90 across 540 scenarios)
            'sweep_p10': plant_fan[orispl]['p10'],
            'sweep_p50': plant_fan[orispl]['p50'],
            'sweep_p90': plant_fan[orispl]['p90'],
        })

    output = {
        'baseline_mmt': round(baseline, 2),
        'year_labels': [str(y) for y in SIM_YEARS],
        'plants': plant_list,
        'fan_bands': fan,
        'trajectories': trajectories,
        'assumptions': {
            **meta['assumptions'],
            'ccs_note': 'CCS delta = plant sweep emissions - CCS emissions. '
                        'CCS: 80% CF, 95% capture, +14% HR penalty on lb CO2/MWh.',
        },
        'sweep_type': 'reference',
        'sweep_note': 'Fan bands from 540 reference market scenarios. '
                      'Per-plant sweep_p50 = dispatch emissions at each year.',
    }

    # Export as JS
    js_path = os.path.join(ROOT_DIR, 'dashboard', 'js', 'emissions-dashboard-data.js')
    with open(js_path, 'w') as f:
        f.write('// Auto-generated by constellation_dispatch.py\n')
        f.write('// Fan bands: REFERENCE sweep only (540 scenarios, market-driven)\n')
        f.write('// NOT including power_nz or economy_nz sweeps\n')
        f.write(f'const EMISSIONS_DASH = {json.dumps(output)};\n')
    print(f'\nSaved: {js_path}')

    # Also save raw JSON
    json_path = os.path.join(SCRIPT_DIR, 'cc_plant_dispatch.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'Saved: {json_path}')

    print('\n' + '═' * 70)


if __name__ == '__main__':
    main()
