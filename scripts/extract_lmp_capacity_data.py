#!/usr/bin/env python3
"""Extract capacity market revenue data from augmented LMP parquets for dashboard.

Generates dashboard/js/lmp-capacity-data.js with:
1. Per-ISO capacity revenue curves by threshold
2. Nuclear total market revenue (LMP + capacity) for viability analysis
3. Reference case market price fans from Step 10 sweep (if available)
"""
import os
import sys
import json
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LMP_DIR = os.path.join(ROOT_DIR, 'data', 'step5-post-processing', 'lmp')
STEP10_DIR = os.path.join(ROOT_DIR, 'data', 'step10-smartargets')
OUTPUT_JS = os.path.join(ROOT_DIR, 'dashboard', 'js', 'lmp-capacity-data.js')

ALL_ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']

# Active thresholds matching lmp-data.js
ACTIVE_THRESHOLDS = [50, 55, 60, 65, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 99.99]


def extract_capacity_data():
    """Extract capacity market revenue stats from augmented LMP parquets."""
    all_data = {}

    for iso in ALL_ISOS:
        path = os.path.join(LMP_DIR, f'{iso}_lmp.parquet')
        if not os.path.exists(path):
            print(f"  {iso}: no LMP parquet, skipping")
            continue

        df = pd.read_parquet(path)

        # Check if capacity columns exist
        if 'capacity_rev_system_mwh' not in df.columns:
            print(f"  {iso}: no capacity revenue columns, skipping")
            continue

        # Filter to active thresholds and Medium fuel
        df_med = df[df['fuel_level'] == 'Medium'] if 'fuel_level' in df.columns else df
        if df_med.empty:
            df_med = df

        iso_data = {'thresholds': ACTIVE_THRESHOLDS}
        metrics = [
            'capacity_rev_system_mwh', 'capacity_rev_clean_firm_mwh',
            'capacity_rev_vre_mwh', 'capacity_rev_storage_mwh', 'capacity_rev_ccs_mwh',
            'total_market_rev_mwh', 'total_rev_clean_firm_mwh',
            'total_rev_vre_mwh', 'total_rev_storage_mwh', 'total_rev_ccs_mwh',
        ]

        for metric in metrics:
            if metric not in df_med.columns:
                continue

            p10_vals, p50_vals, p90_vals = [], [], []
            for t in ACTIVE_THRESHOLDS:
                subset = df_med[df_med['threshold'] == t][metric]
                if len(subset) > 0:
                    p10_vals.append(round(float(subset.quantile(0.10)), 2))
                    p50_vals.append(round(float(subset.quantile(0.50)), 2))
                    p90_vals.append(round(float(subset.quantile(0.90)), 2))
                else:
                    p10_vals.append(0)
                    p50_vals.append(0)
                    p90_vals.append(0)

            iso_data[metric] = {'p10': p10_vals, 'p50': p50_vals, 'p90': p90_vals}

        # Also extract avg_lmp for reference
        p50_lmp = []
        for t in ACTIVE_THRESHOLDS:
            subset = df_med[df_med['threshold'] == t]['avg_lmp']
            p50_lmp.append(round(float(subset.quantile(0.50)), 2) if len(subset) > 0 else 0)
        iso_data['avg_lmp_p50'] = p50_lmp

        # Nuclear viability: total_rev_clean_firm at Low/Medium/High fuel
        nuc_by_fuel = {}
        for fuel in ['Low', 'Medium', 'High']:
            df_fuel = df[df['fuel_level'] == fuel] if 'fuel_level' in df.columns else df
            vals = []
            for t in ACTIVE_THRESHOLDS:
                subset = df_fuel[df_fuel['threshold'] == t]['total_rev_clean_firm_mwh']
                vals.append(round(float(subset.quantile(0.50)), 2) if len(subset) > 0 else 0)
            nuc_by_fuel[fuel] = vals
        iso_data['nuclear_total_rev_by_fuel'] = nuc_by_fuel

        all_data[iso] = iso_data
        print(f"  {iso}: extracted capacity data ({len(ACTIVE_THRESHOLDS)} thresholds)")

    return all_data


def _extract_clean_price_map(df_2050, min_n=3):
    """Bin 2050 data by clean% (5% bins) and compute P10/P50/P90 price stats."""
    if len(df_2050) == 0 or 'clean_pct' not in df_2050.columns:
        return []

    df_2050 = df_2050.copy()
    df_2050['clean_bin'] = (df_2050['clean_pct'] / 5).round() * 5
    bins = sorted(df_2050['clean_bin'].unique())

    result = []
    for b in bins:
        subset = df_2050[df_2050['clean_bin'] == b]
        if len(subset) < min_n or b < 50 or b > 100:
            continue

        entry = {
            'clean_pct': round(float(b), 0),
            'n': len(subset),
            'avg_lmp_p50': round(float(subset['avg_lmp'].quantile(0.50)), 1),
        }

        # Power market revenue = energy + capacity (no RECs)
        if 'energy_rev_mwh' in subset.columns and 'capacity_rev_mwh' in subset.columns:
            pm = subset['energy_rev_mwh'] + subset['capacity_rev_mwh']
            entry['power_market_p10'] = round(float(pm.quantile(0.10)), 1)
            entry['power_market_p50'] = round(float(pm.quantile(0.50)), 1)
            entry['power_market_p90'] = round(float(pm.quantile(0.90)), 1)

        result.append(entry)

    return result


def extract_reference_case_data():
    """Extract Step 10 reference case price fans for dashboard overlay.

    Loads three sweep families (reference, power_nz, economy_nz) and outputs
    per-family clean_price_map with P10/P50/P90 confidence intervals.
    """
    ref_data = {}

    SWEEP_FAMILIES = {
        'reference': 'Reference (Market)',
        'power_nz': 'AT Power NZ',
        'economy_nz': 'AT Economy NZ',
    }

    for iso in ALL_ISOS:
        # Load reference sweep (always required)
        ref_path = os.path.join(STEP10_DIR, f'sweep_reference_{iso}.parquet')
        if not os.path.exists(ref_path):
            continue

        df_ref = pd.read_parquet(ref_path)
        if 'energy_rev_mwh' not in df_ref.columns and 'avg_lmp' not in df_ref.columns:
            continue

        iso_ref = {'years': [2023, 2030, 2035, 2040, 2045, 2050]}

        # Price fans by year (aggregated across all reference scenarios)
        for metric in ['avg_lmp', 'revenue_per_mwh', 'energy_rev_mwh',
                        'capacity_rev_mwh', 'rec_rev_mwh', 'clean_pct',
                        'cost_per_mwh', 'emissions_mt']:
            if metric not in df_ref.columns:
                continue

            p10, p25, p50, p75, p90 = [], [], [], [], []
            for yr in iso_ref['years']:
                subset = df_ref[df_ref['year'] == yr][metric].dropna()
                if len(subset) > 0:
                    p10.append(round(float(subset.quantile(0.10)), 2))
                    p25.append(round(float(subset.quantile(0.25)), 2))
                    p50.append(round(float(subset.quantile(0.50)), 2))
                    p75.append(round(float(subset.quantile(0.75)), 2))
                    p90.append(round(float(subset.quantile(0.90)), 2))
                else:
                    p10.append(0); p25.append(0); p50.append(0)
                    p75.append(0); p90.append(0)

            iso_ref[metric] = {'p10': p10, 'p25': p25, 'p50': p50, 'p75': p75, 'p90': p90}

        # Legacy: aggregated clean_price_map_2050 (backward compat)
        df_2050 = df_ref[df_ref['year'] == 2050]
        iso_ref['clean_price_map_2050'] = _extract_clean_price_map(df_2050)

        # Per-family clean_price_map with P10/P50/P90 CIs
        families = {}
        for sweep_key, sweep_label in SWEEP_FAMILIES.items():
            sweep_path = os.path.join(STEP10_DIR, f'sweep_{sweep_key}_{iso}.parquet')
            if not os.path.exists(sweep_path):
                continue

            df_sweep = pd.read_parquet(sweep_path)
            df_sweep_2050 = df_sweep[df_sweep['year'] == 2050]
            price_map = _extract_clean_price_map(df_sweep_2050)

            if price_map:
                families[sweep_key] = {
                    'label': sweep_label,
                    'n_scenarios': int(df_sweep['scenario'].nunique()),
                    'clean_price_map_2050': price_map,
                }
                print(f"    {iso}/{sweep_key}: {len(price_map)} bins, "
                      f"{df_sweep['scenario'].nunique()} scenarios")

        iso_ref['families'] = families

        ref_data[iso] = iso_ref
        n_families = len(families)
        print(f"  {iso}: extracted reference case data ({n_families} families)")

    return ref_data


def main():
    print("=" * 70)
    print("  Extracting LMP + Capacity Market Data for Dashboard")
    print("=" * 70)

    print("\n--- Step 5B Capacity Revenue ---")
    cap_data = extract_capacity_data()

    print("\n--- Step 10 Reference Case ---")
    ref_data = extract_reference_case_data()

    # Write JS file
    js_content = "// Auto-generated by extract_lmp_capacity_data.py\n"
    js_content += "// Capacity market revenue data (from Step 5B augmented LMP parquets)\n"
    js_content += "// Reference case market price fans (from Step 10 sweep)\n\n"

    js_content += f"var LMP_CAPACITY = {json.dumps(cap_data, indent=2)};\n\n"
    js_content += f"var REFERENCE_CASE = {json.dumps(ref_data, indent=2)};\n"

    os.makedirs(os.path.dirname(OUTPUT_JS), exist_ok=True)
    with open(OUTPUT_JS, 'w') as f:
        f.write(js_content)

    print(f"\n→ Wrote {OUTPUT_JS}")
    print(f"  Capacity data: {len(cap_data)} ISOs")
    print(f"  Reference case: {len(ref_data)} ISOs")


if __name__ == '__main__':
    main()
