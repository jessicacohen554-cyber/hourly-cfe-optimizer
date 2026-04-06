#!/usr/bin/env python3
"""Step 7.1H — Extract hybrid analysis data for dashboard.

Reads step 2.2a cost-optimization parquets and computes aggregated
statistics on solar+battery and wind+battery hybrid resource adoption
across ISOs, thresholds, and cost scenarios. Writes results to
dashboard/js/hybrid-analysis-data.js for the scrollytell hybrid page.

Scenario key format: {Ren}{Firm}{Batt}{LDES}_{Fuel}_{Tx}_{CCS}{45Q}_{Geo}
  - Position 0: Renewable cost (L/M/H)
  - Position 1: Firm cost (L/M/H)
  - Position 2: Battery cost (L/M/H)
  - Position 3: LDES cost (L/M/H)
  - Field 1: Fossil fuel cost (L/M/H)
  - Field 2: Transmission (N/L/M/H)
  - Field 3[0]: CCS cost (L/M/H), [1]: 45Q (0/1)
  - Field 4: Geothermal (L/M/H/X)
"""

import json
import os
import sys
import numpy as np
import pandas as pd

ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']
HYBRID_COLS = ['mix_solar_batt4', 'mix_solar_batt8', 'mix_wind_batt4', 'mix_wind_batt8']
MIX_COLS = ['mix_clean_firm', 'mix_solar', 'mix_wind', 'mix_offshore_wind',
            'mix_geothermal', 'mix_ccs_ccgt', 'mix_hydro'] + HYBRID_COLS

DATA_DIR = 'data/step2.2-cost'
OUT_PATH = 'dashboard/js/hybrid-analysis-data.js'


def load_all_co():
    """Load all step 2.2a CO parquets into a single DataFrame."""
    frames = []
    for iso in ISOS:
        path = os.path.join(DATA_DIR, f'step_2_2a_CO_{iso}.parquet')
        if os.path.exists(path):
            df = pd.read_parquet(path)
            frames.append(df)
            print(f'  Loaded {iso}: {len(df):,} rows')
        else:
            print(f'  WARNING: {path} not found, skipping {iso}')
    if not frames:
        print('ERROR: No CO parquets found.')
        sys.exit(1)
    all_df = pd.concat(frames, ignore_index=True)

    # Derived columns
    all_df['hybrid_total'] = all_df[HYBRID_COLS].sum(axis=1)
    all_df['has_hybrid'] = all_df['hybrid_total'] > 0
    all_df['solar_hybrid'] = all_df['mix_solar_batt4'] + all_df['mix_solar_batt8']
    all_df['wind_hybrid'] = all_df['mix_wind_batt4'] + all_df['mix_wind_batt8']

    # Parse scenario components
    all_df['ren_cost'] = all_df['scenario'].str[0]
    all_df['firm_cost'] = all_df['scenario'].str[1]
    all_df['batt_cost'] = all_df['scenario'].str[2]
    all_df['ldes_cost'] = all_df['scenario'].str[3]

    return all_df


def compute_prevalence(df):
    """Hybrid prevalence by threshold and ISO."""
    thresholds = sorted(df['threshold'].unique())
    result = {'thresholds': [float(t) for t in thresholds], 'byISO': {}}

    for iso in ISOS:
        iso_df = df[df['iso'] == iso]
        if iso_df.empty:
            continue
        pct_list = []
        avg_list = []
        for t in thresholds:
            sub = iso_df[iso_df['threshold'] == t]
            if sub.empty:
                pct_list.append(0)
                avg_list.append(0)
                continue
            pct_list.append(round(float(sub['has_hybrid'].mean() * 100), 1))
            avg_list.append(round(float(sub['hybrid_total'].mean()), 1))
        result['byISO'][iso] = {
            'pct_with_hybrid': pct_list,
            'avg_hybrid_share': avg_list
        }
    return result


def compute_type_breakdown(df):
    """Average share of each hybrid type by threshold and ISO."""
    thresholds = sorted(df['threshold'].unique())
    result = {'thresholds': [float(t) for t in thresholds], 'byISO': {}}

    for iso in ISOS:
        iso_df = df[df['iso'] == iso]
        if iso_df.empty:
            continue
        iso_data = {}
        for col in HYBRID_COLS:
            key = col.replace('mix_', '')
            vals = []
            for t in thresholds:
                sub = iso_df[iso_df['threshold'] == t]
                vals.append(round(float(sub[col].mean()), 2) if not sub.empty else 0)
            iso_data[key] = vals
        result['byISO'][iso] = iso_data
    return result


def compute_substitution(df):
    """With-hybrid vs without-hybrid comparison at high thresholds."""
    thresholds = [75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 99.5, 99.9]
    result = {'thresholds': thresholds, 'byISO': {}}

    for iso in ISOS:
        iso_df = df[df['iso'] == iso]
        if iso_df.empty:
            continue
        w, wo = {}, {}
        for key, metric in [
            ('standalone_sw', lambda s: (s['mix_solar'] + s['mix_wind']).mean()),
            ('total_sw', lambda s: (s['mix_solar'] + s['mix_wind'] + s[HYBRID_COLS].sum(axis=1)).mean()),
            ('clean_firm', lambda s: s['mix_clean_firm'].mean()),
            ('cost', lambda s: s['cost_total_cost'].mean()),
        ]:
            w_vals, wo_vals = [], []
            for t in thresholds:
                sub = iso_df[iso_df['threshold'] == t]
                wh = sub[sub['has_hybrid']]
                woh = sub[~sub['has_hybrid']]
                w_vals.append(round(float(metric(wh)), 1) if len(wh) > 0 else None)
                wo_vals.append(round(float(metric(woh)), 1) if len(woh) > 0 else None)
            w[key] = w_vals
            wo[key] = wo_vals
        result['byISO'][iso] = {'with_hybrid': w, 'without_hybrid': wo}
    return result


def compute_cost_sensitivity(df):
    """Hybrid share grouped by battery/LDES/renewable/firm cost levels."""
    thresholds = [85, 87.5, 90, 92.5, 95, 97.5, 99, 99.5, 99.9]
    levels = ['L', 'M', 'H']
    result = {'thresholds': [float(t) for t in thresholds], 'byISO': {}}

    for iso in ISOS:
        iso_df = df[df['iso'] == iso]
        if iso_df.empty:
            continue
        iso_data = {}
        for dimension, col_name in [
            ('byBatteryCost', 'batt_cost'),
            ('byLDESCost', 'ldes_cost'),
            ('byRenewableCost', 'ren_cost'),
            ('byFirmCost', 'firm_cost'),
        ]:
            dim_data = {}
            for lvl in levels:
                vals = []
                for t in thresholds:
                    sub = iso_df[(iso_df['threshold'] == t) & (iso_df[col_name] == lvl)]
                    vals.append(round(float(sub['hybrid_total'].mean()), 1) if not sub.empty else 0)
                dim_data[lvl] = vals
            iso_data[dimension] = dim_data
        result['byISO'][iso] = iso_data
    return result


def compute_medium_mix(df):
    """Full mix composition at medium costs for narrative."""
    # Medium scenarios: first 4 chars = 'MMMM'
    med = df[df['scenario'].str[:4] == 'MMMM']
    thresholds = [50, 75, 90, 95, 97.5, 99, 99.9]
    cols_to_export = ['mix_solar', 'mix_wind', 'mix_solar_batt4', 'mix_solar_batt8',
                      'mix_wind_batt4', 'mix_wind_batt8', 'mix_clean_firm',
                      'mix_hydro', 'mix_ccs_ccgt', 'mix_geothermal', 'mix_offshore_wind']
    result = {'thresholds': thresholds, 'byISO': {}}

    for iso in ISOS:
        iso_df = med[med['iso'] == iso]
        if iso_df.empty:
            continue
        iso_data = {}
        for col in cols_to_export:
            key = col.replace('mix_', '')
            vals = []
            for t in thresholds:
                sub = iso_df[iso_df['threshold'] == t]
                vals.append(round(float(sub[col].mean()), 1) if not sub.empty else 0)
            iso_data[key] = vals
        result['byISO'][iso] = iso_data
    return result


def compute_peak_hybrid(df):
    """Peak hybrid penetration per ISO."""
    result = {}
    for iso in ISOS:
        iso_df = df[df['iso'] == iso]
        if iso_df.empty:
            continue
        idx = iso_df['hybrid_total'].idxmax()
        row = iso_df.loc[idx]
        result[iso] = {
            'max_pct': int(row['hybrid_total']),
            'threshold': float(row['threshold']),
            'solar_batt4': int(row['mix_solar_batt4']),
            'solar_batt8': int(row['mix_solar_batt8']),
            'wind_batt4': int(row['mix_wind_batt4']),
            'wind_batt8': int(row['mix_wind_batt8']),
            'standalone_solar': int(row['mix_solar']),
            'standalone_wind': int(row['mix_wind']),
        }
    return result


def compute_duration_preference(df):
    """4hr vs 8hr battery pairing preference at high thresholds."""
    high = df[df['threshold'] >= 90]
    result = {}
    for iso in ISOS:
        sub = high[high['iso'] == iso]
        if sub.empty:
            continue
        result[iso] = {
            'solar_4hr_freq': round(float((sub['mix_solar_batt4'] > 0).mean() * 100), 1),
            'solar_8hr_freq': round(float((sub['mix_solar_batt8'] > 0).mean() * 100), 1),
            'wind_4hr_freq': round(float((sub['mix_wind_batt4'] > 0).mean() * 100), 1),
            'wind_8hr_freq': round(float((sub['mix_wind_batt8'] > 0).mean() * 100), 1),
            'solar_4hr_avg': round(float(sub.loc[sub['mix_solar_batt4'] > 0, 'mix_solar_batt4'].mean()), 1)
                if (sub['mix_solar_batt4'] > 0).any() else 0,
            'solar_8hr_avg': round(float(sub.loc[sub['mix_solar_batt8'] > 0, 'mix_solar_batt8'].mean()), 1)
                if (sub['mix_solar_batt8'] > 0).any() else 0,
            'wind_4hr_avg': round(float(sub.loc[sub['mix_wind_batt4'] > 0, 'mix_wind_batt4'].mean()), 1)
                if (sub['mix_wind_batt4'] > 0).any() else 0,
            'wind_8hr_avg': round(float(sub.loc[sub['mix_wind_batt8'] > 0, 'mix_wind_batt8'].mean()), 1)
                if (sub['mix_wind_batt8'] > 0).any() else 0,
        }
    return result


def main():
    print('Step 7.1H — Extracting hybrid analysis data')
    print('=' * 60)

    print('\nLoading CO parquets...')
    df = load_all_co()
    print(f'Total: {len(df):,} rows across {df["iso"].nunique()} ISOs\n')

    print('Computing prevalence...')
    prevalence = compute_prevalence(df)

    print('Computing type breakdown...')
    type_breakdown = compute_type_breakdown(df)

    print('Computing substitution analysis...')
    substitution = compute_substitution(df)

    print('Computing cost sensitivity...')
    cost_sensitivity = compute_cost_sensitivity(df)

    print('Computing medium-scenario mixes...')
    medium_mix = compute_medium_mix(df)

    print('Computing peak hybrid stats...')
    peak_hybrid = compute_peak_hybrid(df)

    print('Computing duration preference...')
    duration_preference = compute_duration_preference(df)

    # Assemble final data object
    hybrid_data = {
        'prevalence': prevalence,
        'typeBreakdown': type_breakdown,
        'substitution': substitution,
        'costSensitivity': cost_sensitivity,
        'mediumMix': medium_mix,
        'peakHybrid': peak_hybrid,
        'durationPreference': duration_preference,
    }

    # Write JS file
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        f.write('// Auto-generated by step7_1h_extract_hybrid_data.py\n')
        f.write('// Do not edit manually.\n\n')
        f.write('const HYBRID_DATA = ')
        json.dump(hybrid_data, f, indent=2)
        f.write(';\n')

    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f'\nWrote {OUT_PATH} ({size_kb:.1f} KB)')
    print('Done.')


if __name__ == '__main__':
    main()
