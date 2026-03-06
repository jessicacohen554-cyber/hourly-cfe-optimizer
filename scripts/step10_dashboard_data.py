#!/usr/bin/env python3
"""Extract Step 10 SMARTargets parquet data into a JS-embeddable data file for the dashboard."""

import json
import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'step10-smartargets')
OUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'dashboard', 'js', 'smartargets-data.js')

ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']
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
    return [round(float(x), decimals) for x in arr]


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
            'clean_pct': round_list(sub['clean_pct']),
            'cost_per_mwh': round_list(sub['cost_per_mwh']),
            'demand_twh': round_list(sub['demand_twh']),
            'emission_rate': round_list(sub.get('emission_rate', pd.Series([0]*len(sub)))),
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
            'clean_pct': {},
            'cost_per_mwh': {},
            'demand_twh': {},
            'mix': {},
            'curtailment_twh': {},
        }
        for t in targets:
            tkey = f'{t:.2f}'
            tsub = sub[sub.reduction_target == t].sort_values('year')
            entry['emissions'][tkey] = round_list(tsub['emissions_mt'])
            entry['clean_pct'][tkey] = round_list(tsub['clean_pct'])
            entry['cost_per_mwh'][tkey] = round_list(tsub['cost_per_mwh'])
            entry['demand_twh'][tkey] = round_list(tsub['demand_twh'])
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
