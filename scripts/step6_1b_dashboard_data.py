#!/usr/bin/env python3
"""Extract Step 6.1 SMARTargets scenario data into JS-embeddable dashboard format.

Loads:
- Reference: Market trajectory fan from reference sweep (all conditions combined)
- AT1/AT2/AT3/AT4: SMARTargets aspirational targets (policy scenarios)
"""

import json
import os
import sys
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'step6-smartargets')
OUT_PATH = os.path.join(SCRIPT_DIR, '..', 'dashboard', 'js', 'smartargets-data.js')

sys.path.insert(0, SCRIPT_DIR)
from pipeline_config import ISOS


def extract_scenario_data(parquet_path, scenario_code):
    """Extract full scenario time series from a dedicated AT/QT parquet file."""
    if not os.path.exists(parquet_path):
        print(f'    ERROR: {parquet_path} not found')
        return {}

    df = pd.read_parquet(parquet_path)
    result = {}

    for iso in ISOS:
        iso_df = df[df['iso'] == iso].copy()
        if iso_df.empty:
            continue

        iso_df = iso_df.sort_values('year')
        years = iso_df['year'].tolist()

        # Build time series arrays for all relevant metrics
        data = {
            'years': [int(y) for y in years],
            'emissions_mt': iso_df['emissions_mt'].round(2).tolist(),
            'clean_pct': iso_df['clean_pct'].round(1).tolist(),
            'carbon_shadow_price': iso_df['carbon_shadow_price'].round(1).tolist(),
            'dac_offset_mt': iso_df['dac_offset_mt'].round(2).tolist(),
            'dac_cost_per_ton': iso_df.get('dac_cost_per_ton', [0]*len(years)).fillna(0).round(2).tolist() if 'dac_cost_per_ton' in iso_df.columns else [0]*len(years),
            'mandated_subsidy_mwh': iso_df['mandated_subsidy_mwh'].round(2).tolist(),
        }

        # Add resource mix data
        mix_cols = ['mix_clean_firm_twh', 'mix_solar_twh', 'mix_wind_twh', 'mix_offshore_wind_twh',
                    'mix_ccs_ccgt_twh', 'mix_hydro_twh', 'mix_battery_twh', 'mix_ldes_twh']
        for col in mix_cols:
            if col in iso_df.columns:
                data[col] = iso_df[col].round(2).tolist()

        result[iso] = data

    return result


def extract_at_from_sweep(sweep_type, condition_filter):
    """Extract AT-equivalent P50 trajectory from a parametric sweep.

    AT1 = power_nz + Facilitating (F)
    AT2 = power_nz + Challenging (C)
    AT3 = economy_nz + Facilitating (F)
    AT4 = economy_nz + Challenging (C)

    Returns P50 (median) across all scenarios matching the condition filter,
    plus P10/P90 fan for uncertainty visualization.
    """
    result = {}

    for iso in ISOS:
        path = os.path.join(DATA_DIR, f'sweep_{sweep_type}_{iso}.parquet')
        if not os.path.exists(path):
            continue

        df = pd.read_parquet(path)
        if 'conditions' in df.columns:
            df = df[df['conditions'] == condition_filter]

        if df.empty:
            continue

        years = sorted(df['year'].unique())

        data = {'years': [int(y) for y in years]}

        # Compute P50 for key metrics, P10/P90 for emissions fan
        metric_cols = {
            'emissions_mt': 2,
            'clean_pct': 1,
            'carbon_shadow_price': 1,
            'dac_offset_mt': 2,
            'mandated_subsidy_mwh': 2,
        }

        for col, decimals in metric_cols.items():
            if col not in df.columns:
                data[col] = [0] * len(years)
                continue
            vals = []
            for yr in years:
                yr_vals = df[df['year'] == yr][col].dropna()
                vals.append(round(float(yr_vals.median()), decimals) if len(yr_vals) > 0 else 0)
            data[col] = vals

        # Emissions fan (P10/P90)
        data['emissions_fan'] = {'p10': [], 'p50': [], 'p90': []}
        for yr in years:
            yr_vals = df[df['year'] == yr]['emissions_mt'].values
            data['emissions_fan']['p10'].append(round(float(np.percentile(yr_vals, 10)), 2))
            data['emissions_fan']['p50'].append(round(float(np.percentile(yr_vals, 50)), 2))
            data['emissions_fan']['p90'].append(round(float(np.percentile(yr_vals, 90)), 2))

        # DAC cost per ton: compute from dac_cost_million / dac_offset_mt
        dac_cost_per_ton = []
        for yr in years:
            yr_df = df[df['year'] == yr]
            cost = yr_df['dac_cost_million'].median() if 'dac_cost_million' in yr_df.columns else 0
            offset = yr_df['dac_offset_mt'].median() if 'dac_offset_mt' in yr_df.columns else 0
            if offset > 0 and cost > 0:
                dac_cost_per_ton.append(round(float(cost / offset), 2))
            else:
                dac_cost_per_ton.append(0)
        data['dac_cost_per_ton'] = dac_cost_per_ton

        # Resource mix (P50)
        mix_cols = ['mix_clean_firm_twh', 'mix_solar_twh', 'mix_wind_twh', 'mix_offshore_wind_twh',
                    'mix_ccs_ccgt_twh', 'mix_hydro_twh', 'mix_battery_twh', 'mix_ldes_twh']
        for col in mix_cols:
            if col in df.columns:
                vals = []
                for yr in years:
                    yr_vals = df[df['year'] == yr][col].dropna()
                    vals.append(round(float(yr_vals.median()), 2) if len(yr_vals) > 0 else 0)
                data[col] = vals

        result[iso] = data

    return result


def extract_market_trajectory():
    """Extract market trajectory fan from reference sweep (all conditions combined).
    
    Returns p10/p25/p50/p75/p90 bands across all parameterizations.
    """
    results = {}
    
    for iso in ISOS:
        path = os.path.join(DATA_DIR, f'sweep_reference_{iso}.parquet')
        if not os.path.exists(path):
            continue
        
        df = pd.read_parquet(path)
        
        if df.empty:
            continue
        
        years = sorted(df['year'].unique())
        
        iso_data = {
            'years': [int(y) for y in years],
            'fan': {
                'p10': [], 'p25': [], 'p50': [], 'p75': [], 'p90': [],
                'min': [], 'max': []
            },
        }
        
        clean_pct_vals = {'p10': [], 'p25': [], 'p50': [], 'p75': [], 'p90': []}
        
        for year in years:
            year_df = df[df['year'] == year]
            emissions = year_df['emissions_mt'].values
            
            iso_data['fan']['p10'].append(round(float(np.percentile(emissions, 10)), 2))
            iso_data['fan']['p25'].append(round(float(np.percentile(emissions, 25)), 2))
            iso_data['fan']['p50'].append(round(float(np.percentile(emissions, 50)), 2))
            iso_data['fan']['p75'].append(round(float(np.percentile(emissions, 75)), 2))
            iso_data['fan']['p90'].append(round(float(np.percentile(emissions, 90)), 2))
            iso_data['fan']['min'].append(round(float(emissions.min()), 2))
            iso_data['fan']['max'].append(round(float(emissions.max()), 2))
            
            # Clean percentage fan (if column exists)
            if 'clean_pct' in year_df.columns:
                cpct = year_df['clean_pct'].values
                for p, q in [('p10', 10), ('p25', 25), ('p50', 50), ('p75', 75), ('p90', 90)]:
                    clean_pct_vals[p].append(round(float(np.percentile(cpct, q)), 1))
        
        # Add P50 aliases for backward compatibility with code expecting AT-format fields
        iso_data['emissions_mt'] = iso_data['fan']['p50']
        if clean_pct_vals['p50']:
            iso_data['clean_pct'] = clean_pct_vals['p50']
            iso_data['clean_pct_fan'] = clean_pct_vals
        
        results[iso] = iso_data
    
    return results


def main():
    print('Extracting Step 6.1 SMARTargets scenario data for dashboard...')
    output = {
        'scenarios': {},
    }
    
    # Extract Reference market trajectory from reference sweep (all conditions combined)
    print('  Extracting Reference (Market Trajectory) from sweep...')
    ref_data = extract_market_trajectory()
    output['scenarios']['Reference'] = ref_data
    for iso in ref_data:
        print(f'    {iso}: {len(ref_data[iso]["years"])} years')
    
    # Extract AT1-AT4 from sweep parquets (preferred) or legacy AT parquets (fallback)
    at_sweep_mapping = {
        'AT1': ('power_nz', 'F'),      # Facilitating
        'AT2': ('power_nz', 'C'),      # Challenging
        'AT3': ('economy_nz', 'F'),    # Facilitating
        'AT4': ('economy_nz', 'C'),    # Challenging
    }

    for code, (sweep_type, cond_filter) in at_sweep_mapping.items():
        # Check if sweep parquets exist
        sweep_exists = any(
            os.path.exists(os.path.join(DATA_DIR, f'sweep_{sweep_type}_{iso}.parquet'))
            for iso in ISOS
        )

        if sweep_exists:
            print(f'  Extracting {code} from {sweep_type} sweep (conditions={cond_filter})...')
            data = extract_at_from_sweep(sweep_type, cond_filter)
        else:
            # Fallback to legacy AT parquet
            legacy_path = os.path.join(DATA_DIR, f'smartargets_{code}.parquet')
            print(f'  Extracting {code} from legacy parquet...')
            data = extract_scenario_data(legacy_path, code)

        output['scenarios'][code] = data
        for iso in data:
            print(f'    {iso}: {len(data[iso]["years"])} years')
    
    # Write JS file
    json_str = json.dumps(output, separators=(',', ':'))
    js_content = f'// Auto-generated by step6_1b_dashboard_data.py — do not edit manually\nconst SMARTARGETS_DATA = {json_str};\n'

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        f.write(js_content)

    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f'  Written {OUT_PATH} ({size_kb:.0f} KB)')
    print('Done.')


if __name__ == '__main__':
    main()


