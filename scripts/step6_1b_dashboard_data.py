#!/usr/bin/env python3
"""Extract Step 10 SMARTargets parquet data into a JS-embeddable data file for the dashboard."""

import json
import os
import sys
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'step10-smartargets')
OUT_PATH = os.path.join(SCRIPT_DIR, '..', 'dashboard', 'js', 'smartargets-data.js')

sys.path.insert(0, SCRIPT_DIR)
from pipeline_config import ISOS
BASE_SCENARIOS = ['R1', 'R2', 'AT1', 'AT2', 'AT3', 'AT4']
QT_SCENARIOS = ['QT1', 'QT2', 'QT3', 'QT4']
MIX_COLS = ['mix_clean_firm_twh', 'mix_solar_twh', 'mix_wind_twh', 'mix_offshore_wind_twh',
            'mix_ccs_ccgt_twh', 'mix_hydro_twh', 'mix_battery_twh', 'mix_ldes_twh']
MIX_KEYS = ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro', 'battery', 'ldes']

# Scenario metadata for the dashboard
SCENARIO_META = {
    'R1': {'name': 'Reference: Facilitating', 'group': 'reference', 'color': '#22C55E',
           'desc': 'Market-driven, no emission constraints. Facilitating conditions (low LCOE, fast learning, reformed queues).'},
    'R2': {'name': 'Reference: Challenging', 'group': 'reference', 'color': '#EF4444',
           'desc': 'Market-driven, no emission constraints. Challenging conditions (high LCOE, slow learning, constrained queues).'},
    'AT1': {'name': 'Power NZ: Facilitating', 'group': 'power_nz', 'color': '#6366F1',
            'desc': 'Power sector net-zero by 2050. Emission cap trajectory under facilitating conditions.'},
    'AT2': {'name': 'Power NZ: Challenging', 'group': 'power_nz', 'color': '#EC4899',
            'desc': 'Power sector net-zero by 2050. Emission cap under challenging conditions.'},
    'AT3': {'name': 'Economy NZ: Facilitating', 'group': 'economy_nz', 'color': '#0EA5E9',
            'desc': 'Economy-wide net-zero by 2050. Tighter power sector caps. Facilitating conditions.'},
    'AT4': {'name': 'Economy NZ: Challenging', 'group': 'economy_nz', 'color': '#F97316',
            'desc': 'Economy-wide net-zero by 2050. Tighter caps, challenging conditions.'},
    'QT1': {'name': 'QT Medium Demand: Facilitating', 'group': 'qualified', 'color': '#22C55E',
            'desc': 'Parametric sweep (5-95% reduction). Medium demand, facilitating conditions.'},
    'QT2': {'name': 'QT Medium Demand: Challenging', 'group': 'qualified', 'color': '#EF4444',
            'desc': 'Parametric sweep (5-95% reduction). Medium demand, challenging conditions.'},
    'QT3': {'name': 'QT High Demand: Facilitating', 'group': 'qualified', 'color': '#6366F1',
            'desc': 'Parametric sweep (5-95% reduction). High demand, facilitating conditions.'},
    'QT4': {'name': 'QT High Demand: Challenging', 'group': 'qualified', 'color': '#F97316',
            'desc': 'Parametric sweep (5-95% reduction). High demand, challenging conditions.'},
}


def round_list(arr, decimals=2):
    return [round(float(x), decimals) if x is not None and not (isinstance(x, float) and np.isnan(x)) else None for x in arr]


def extract_base_scenario(name):
    """Extract single-trajectory scenario (R1/R2/AT1-4)."""
    path = os.path.join(DATA_DIR, f'smartargets_{name}.parquet')
    if not os.path.exists(path):
        print(f'  WARNING: {path} not found, skipping {name}')
        return {}
    df = pd.read_parquet(path)
    result = {}
    for iso in ISOS:
        sub = df[df.iso == iso].sort_values('year')
        if sub.empty:
            continue
        entry = {
            'years': sub['year'].tolist(),
            'emissions_mt': round_list(sub['emissions_mt']),
            'gross_emissions_mt': round_list(sub['gross_emissions_mt']) if 'gross_emissions_mt' in sub.columns else round_list(sub['emissions_mt']),
            'clean_pct': round_list(sub['clean_pct']),
            'cost_per_mwh': round_list(sub['cost_per_mwh']),
            'demand_twh': round_list(sub['demand_twh']),
            'emission_rate': round_list(sub.get('emission_rate', pd.Series([0]*len(sub)))),
            'emission_cap_mt': [round(float(x), 2) if pd.notna(x) else None for x in sub['emission_cap_mt']] if 'emission_cap_mt' in sub.columns else [None]*len(sub),
            'dac_offset_mt': round_list(sub['dac_offset_mt']) if 'dac_offset_mt' in sub.columns else [0]*len(sub),
            'dac_cost_million': round_list(sub['dac_cost_million']) if 'dac_cost_million' in sub.columns else [0]*len(sub),
            'dac_cost_per_ton': round_list(sub['dac_cost_per_ton']) if 'dac_cost_per_ton' in sub.columns else [0]*len(sub),
            'carbon_shadow_price': round_list(sub['carbon_shadow_price']) if 'carbon_shadow_price' in sub.columns else [0]*len(sub),
            'mandated_subsidy_mwh': round_list(sub['mandated_subsidy_mwh']) if 'mandated_subsidy_mwh' in sub.columns else [0]*len(sub),
            'avg_lmp': round_list(sub['avg_lmp']) if 'avg_lmp' in sub.columns else [0]*len(sub),
            'gas_built_gw': round_list(sub['gas_built_gw']) if 'gas_built_gw' in sub.columns else [0]*len(sub),
            'total_gas_gw': round_list(sub['total_gas_gw']) if 'total_gas_gw' in sub.columns else [0]*len(sub),
            'mix': {},
        }
        for col, key in zip(MIX_COLS, MIX_KEYS):
            if col in sub.columns:
                entry['mix'][key] = round_list(sub[col])
            else:
                entry['mix'][key] = [0.0] * len(sub)
        # Compute curtailment: total mix TWh - demand * clean_pct / 100
        total_mix = np.zeros(len(sub))
        for col in MIX_COLS:
            if col in sub.columns:
                total_mix += sub[col].values
        utilized = sub['demand_twh'].values * sub['clean_pct'].values / 100
        curtailment = np.maximum(0, total_mix - utilized)
        entry['curtailment_twh'] = round_list(curtailment)
        result[iso] = entry
    return result


def extract_qt_scenario(name):
    """Extract QT parametric sweep scenario (19 reduction targets × 6 years × 7 ISOs)."""
    path = os.path.join(DATA_DIR, f'smartargets_{name}.parquet')
    if not os.path.exists(path):
        print(f'  WARNING: {path} not found, skipping {name}')
        return {}
    df = pd.read_parquet(path)
    result = {}
    for iso in ISOS:
        sub = df[df.iso == iso].copy()
        if sub.empty:
            continue
        targets = sorted(sub['reduction_target'].unique())
        years = sorted(sub['year'].unique())
        entry = {
            'reduction_targets': round_list(targets),
            'years': [int(y) for y in years],
            'emissions': {},
            'gross_emissions': {},
            'clean_pct': {},
            'cost_per_mwh': {},
            'demand_twh': {},
            'dac_offset_mt': {},
            'carbon_shadow_price': {},
            'mandated_subsidy_mwh': {},
            'mix': {},
            'curtailment_twh': {},
        }
        for t in targets:
            tkey = f'{t:.2f}'
            tsub = sub[sub.reduction_target == t].sort_values('year')
            entry['emissions'][tkey] = round_list(tsub['emissions_mt'])
            entry['gross_emissions'][tkey] = round_list(tsub['gross_emissions_mt']) if 'gross_emissions_mt' in tsub.columns else round_list(tsub['emissions_mt'])
            entry['clean_pct'][tkey] = round_list(tsub['clean_pct'])
            entry['cost_per_mwh'][tkey] = round_list(tsub['cost_per_mwh'])
            entry['demand_twh'][tkey] = round_list(tsub['demand_twh'])
            entry['dac_offset_mt'][tkey] = round_list(tsub['dac_offset_mt']) if 'dac_offset_mt' in tsub.columns else [0]*len(tsub)
            entry['carbon_shadow_price'][tkey] = round_list(tsub['carbon_shadow_price']) if 'carbon_shadow_price' in tsub.columns else [0]*len(tsub)
            entry['mandated_subsidy_mwh'][tkey] = round_list(tsub['mandated_subsidy_mwh']) if 'mandated_subsidy_mwh' in tsub.columns else [0]*len(tsub)
            mix_dict = {}
            total_mix = np.zeros(len(tsub))
            for col, key in zip(MIX_COLS, MIX_KEYS):
                vals = tsub[col].values if col in tsub.columns else np.zeros(len(tsub))
                mix_dict[key] = round_list(vals)
                total_mix += vals
            entry['mix'][tkey] = mix_dict
            utilized = tsub['demand_twh'].values * tsub['clean_pct'].values / 100
            curtailment = np.maximum(0, total_mix - utilized)
            entry['curtailment_twh'][tkey] = round_list(curtailment)

        # Compute fan bands (P5, P25, P50, P75, P95) across reduction targets per year
        fan = {'p5': [], 'p25': [], 'p50': [], 'p75': [], 'p95': [], 'min': [], 'max': []}
        for yi, y in enumerate(years):
            vals = []
            for t in targets:
                tkey = f'{t:.2f}'
                if yi < len(entry['emissions'][tkey]):
                    vals.append(entry['emissions'][tkey][yi])
            if vals:
                arr = np.array(vals)
                fan['p5'].append(round(float(np.percentile(arr, 5)), 2))
                fan['p25'].append(round(float(np.percentile(arr, 25)), 2))
                fan['p50'].append(round(float(np.percentile(arr, 50)), 2))
                fan['p75'].append(round(float(np.percentile(arr, 75)), 2))
                fan['p95'].append(round(float(np.percentile(arr, 95)), 2))
                fan['min'].append(round(float(arr.min()), 2))
                fan['max'].append(round(float(arr.max()), 2))
        entry['fan'] = fan
        result[iso] = entry
    return result


def extract_iso_trajectories(base_data, qt_data):
    """Build per-ISO emission trajectories with all scenarios + QT fan bands for overlay charts."""
    result = {}
    years = [2023, 2030, 2035, 2040, 2045, 2050]
    for iso in ISOS:
        iso_traj = {'years': years, 'scenarios': {}, 'qt_fans': {}}

        # Base scenarios (R1, R2, AT1-4) — extract emissions_mt and clean_pct per year
        for scn_name in BASE_SCENARIOS:
            scn_data = base_data.get(scn_name, {})
            iso_data = scn_data.get(iso)
            if iso_data:
                iso_traj['scenarios'][scn_name] = {
                    'emissions_mt': iso_data['emissions_mt'],
                    'clean_pct': iso_data['clean_pct'],
                }

        # QT fan bands — percentiles across reduction targets per year
        for qt_name in QT_SCENARIOS:
            qt_iso = qt_data.get(qt_name, {}).get(iso)
            if not qt_iso:
                continue
            qt_years = qt_iso['years']
            targets = qt_iso['reduction_targets']
            fan = {'p10': [], 'p25': [], 'p50': [], 'p75': [], 'p90': [], 'min': [], 'max': []}
            for yi in range(len(qt_years)):
                vals = []
                for t in targets:
                    tkey = f'{t:.2f}'
                    em = qt_iso['emissions'].get(tkey, [])
                    if yi < len(em):
                        vals.append(em[yi])
                if vals:
                    arr = np.array(vals)
                    fan['p10'].append(round(float(np.percentile(arr, 10)), 2))
                    fan['p25'].append(round(float(np.percentile(arr, 25)), 2))
                    fan['p50'].append(round(float(np.percentile(arr, 50)), 2))
                    fan['p75'].append(round(float(np.percentile(arr, 75)), 2))
                    fan['p90'].append(round(float(np.percentile(arr, 90)), 2))
                    fan['min'].append(round(float(arr.min()), 2))
                    fan['max'].append(round(float(arr.max()), 2))
            iso_traj['qt_fans'][qt_name] = fan

        result[iso] = iso_traj
        n_scn = len(iso_traj['scenarios'])
        n_qt = len(iso_traj['qt_fans'])
        print(f'    {iso}: {n_scn} scenarios + {n_qt} QT fans')
    return result


def main():
    print('Extracting Step 10 SMARTargets data for dashboard...')
    output = {
        'scenario_meta': SCENARIO_META,
        'scenarios': {},
        'qt_scenarios': {},
    }

    # Base scenarios (R1/R2/AT1-4)
    for name in BASE_SCENARIOS:
        print(f'  Extracting {name}...')
        output['scenarios'][name] = extract_base_scenario(name)

    # QT scenarios
    for name in QT_SCENARIOS:
        print(f'  Extracting {name}...')
        output['qt_scenarios'][name] = extract_qt_scenario(name)

    # Per-ISO emission trajectories (all scenarios on one chart per ISO)
    print('  Building per-ISO emission trajectories...')
    output['iso_trajectories'] = extract_iso_trajectories(output['scenarios'], output['qt_scenarios'])

    # Write JS file
    json_str = json.dumps(output, separators=(',', ':'))
    js_content = f'// Auto-generated by step10_dashboard_data.py — do not edit manually\nconst SMARTARGETS_DATA = {json_str};\n'

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        f.write(js_content)

    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f'  Written {OUT_PATH} ({size_kb:.0f} KB)')
    print('Done.')


if __name__ == '__main__':
    main()
