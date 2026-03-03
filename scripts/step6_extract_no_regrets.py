#!/usr/bin/env python3
"""
Step 7 Post-Processor: Extract No-Regrets Investment Data from Step 3 DG Parquets

Reads the demand-growth parquets (step3_dg_{ISO}_t{T}.parquet) for thresholds
within each ISO's MAC-DAC crossover range and computes no-regrets resource
investment statistics across all 5,832+ cost scenarios × 3 growth levels.

Output: dashboard/js/no-regrets-data.js — JavaScript constants consumed by
        abatement_dashboard.html

Usage:
    python scripts/step6_extract_no_regrets.py

Depends on:
    - data/step3-cost-opt-parquets/step3_dg_{ISO}_t{T}.parquet
    - dashboard/js/shared-data.js (for MAC data, DAC trajectories — read inline below)
    - dashboard/js/mac-stats-data.js (for MAC_STEPWISE_FAN — crossover computation)
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================
ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']
THRESHOLDS = [50, 55, 60, 65, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 99.5, 99.9, 99.99]

# Threshold-to-file key mapping (DG parquets use integer or decimal keys)
THRESHOLD_FILE_KEYS = {
    50: '50', 55: '55', 60: '60', 65: '65', 70: '70', 75: '75',
    80: '80', 85: '85', 87.5: '87.5', 90: '90', 92.5: '92.5',
    95: '95', 97.5: '97.5', 99: '99', 99.99: '99.9'  # 99.99 may be stored as 99.9
}

DG_DIR = Path('data/step3-cost-opt-parquets')
OUTPUT_JS = Path('dashboard/js/no-regrets-data.js')
OUTPUT_JSON = Path('data/step5-post-processing/no_regrets_analysis.json')

# Resources to track
MIX_RESOURCES = ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro']
STORAGE_RESOURCES = ['battery', 'battery8', 'ldes']
ALL_RESOURCES = MIX_RESOURCES + STORAGE_RESOURCES

# DAC cost trajectories (from SPEC.md §7.3)
DAC_TRAJECTORY = {
    'optimistic':   {2025: 600, 2030: 350, 2035: 230, 2040: 175, 2045: 130, 2050: 100},
    'central':      {2025: 800, 2030: 500, 2035: 375, 2040: 300, 2045: 250, 2050: 200},
    'conservative': {2025: 1100, 2030: 750, 2035: 550, 2040: 450, 2045: 375, 2050: 300},
}

# SBTi milestone threshold-to-year mapping
THRESHOLD_TARGET_YEARS = {
    50: 2030, 55: 2031, 60: 2033, 65: 2034, 70: 2035,
    75: 2036, 80: 2037, 85: 2038, 87.5: 2039, 90: 2040,
    92.5: 2043, 95: 2045, 97.5: 2048, 99: 2050, 99.99: 2050
}

ISO_LABELS = {
    'CAISO': 'California', 'ERCOT': 'Texas', 'PJM': 'Mid-Atlantic',
    'NYISO': 'New York', 'NEISO': 'New England', 'MISO': 'Midwest', 'SPP': 'Central'
}


def interpolate_dac(year, scenario='central'):
    """Interpolate DAC cost at a given year."""
    traj = DAC_TRAJECTORY[scenario]
    years = sorted(traj.keys())
    if year <= years[0]:
        return traj[years[0]]
    if year >= years[-1]:
        return traj[years[-1]]
    for i in range(len(years) - 1):
        if years[i] <= year <= years[i + 1]:
            frac = (year - years[i]) / (years[i + 1] - years[i])
            return traj[years[i]] + frac * (traj[years[i + 1]] - traj[years[i]])
    return traj[years[-1]]


def dac_at_threshold(threshold, scenario='central'):
    """Get DAC cost at a given threshold via SBTi year mapping."""
    year = THRESHOLD_TARGET_YEARS.get(threshold)
    if year is None:
        # Interpolate year from nearest thresholds
        sorted_t = sorted(THRESHOLD_TARGET_YEARS.keys())
        for i in range(len(sorted_t) - 1):
            if sorted_t[i] <= threshold <= sorted_t[i + 1]:
                frac = (threshold - sorted_t[i]) / (sorted_t[i + 1] - sorted_t[i])
                year = THRESHOLD_TARGET_YEARS[sorted_t[i]] + frac * (THRESHOLD_TARGET_YEARS[sorted_t[i + 1]] - THRESHOLD_TARGET_YEARS[sorted_t[i]])
                break
        if year is None:
            year = 2050
    return interpolate_dac(year, scenario)


def load_mac_data():
    """Load MAC stepwise fan data from mac-stats-data.js."""
    mac_js_path = Path('dashboard/js/mac-stats-data.js')
    if not mac_js_path.exists():
        print(f"  WARNING: {mac_js_path} not found, using hardcoded crossover ranges")
        return None

    content = mac_js_path.read_text()
    # Extract MAC_STEPWISE_FAN object
    start = content.find('const MAC_STEPWISE_FAN')
    if start < 0:
        start = content.find('var MAC_STEPWISE_FAN')
    if start < 0:
        print("  WARNING: MAC_STEPWISE_FAN not found in mac-stats-data.js")
        return None

    # Find the object boundaries
    brace_start = content.find('{', start)
    depth = 0
    pos = brace_start
    while pos < len(content):
        if content[pos] == '{':
            depth += 1
        elif content[pos] == '}':
            depth -= 1
            if depth == 0:
                break
        pos += 1

    json_str = content[brace_start:pos + 1]
    # Clean up JS → JSON: handle trailing commas, single quotes, etc.
    json_str = json_str.replace("'", '"')
    # Remove trailing commas before } or ]
    import re
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    # Handle null values
    json_str = json_str.replace('null', 'null')

    try:
        mac_data = json.loads(json_str)
        return mac_data
    except json.JSONDecodeError as e:
        print(f"  WARNING: Could not parse MAC_STEPWISE_FAN: {e}")
        return None


def compute_crossover_range(iso, mac_data):
    """Compute the MAC-DAC crossover range for an ISO using 3×3 grid."""
    if mac_data is None or iso not in mac_data:
        return None

    fan = mac_data[iso]
    mac_keys = ['p10', 'p50', 'p90']
    dac_scenarios = ['optimistic', 'central', 'conservative']

    crossovers = []
    for mk in mac_keys:
        mac_arr = fan.get(mk, [])
        for ds in dac_scenarios:
            for i, t in enumerate(THRESHOLDS):
                if i >= len(mac_arr):
                    break
                mac_val = mac_arr[i]
                if mac_val is None:
                    continue
                dac_val = dac_at_threshold(t, ds)
                if mac_val > dac_val:
                    crossovers.append(t)
                    break

    if not crossovers:
        return {'lower': THRESHOLDS[-2], 'upper': THRESHOLDS[-1], 'central': THRESHOLDS[-1],
                'grid_always_wins': True}

    # Central crossover: p50 × central
    central = None
    p50_arr = fan.get('p50', [])
    for i, t in enumerate(THRESHOLDS):
        if i >= len(p50_arr):
            break
        if p50_arr[i] is not None and p50_arr[i] > dac_at_threshold(t, 'central'):
            central = t
            break

    return {
        'lower': min(crossovers),
        'upper': max(crossovers),
        'central': central or min(crossovers),
        'grid_always_wins': False
    }


def find_dg_file(iso, threshold):
    """Find the DG parquet file for a given ISO and threshold."""
    # Try different key formats
    for key_fmt in [str(int(threshold)) if threshold == int(threshold) else str(threshold),
                    str(threshold), f"{threshold:.1f}"]:
        path = DG_DIR / f'step3_dg_{iso}_t{key_fmt}.parquet'
        if path.exists():
            return path

    # Also try without decimal for integer thresholds
    if threshold == int(threshold):
        path = DG_DIR / f'step3_dg_{iso}_t{int(threshold)}.parquet'
        if path.exists():
            return path

    return None


def analyze_no_regrets(iso, crossover_range):
    """Analyze no-regrets resources from DG parquets within the crossover range."""
    lower = crossover_range['lower']
    upper = crossover_range['upper']

    # Include thresholds in range plus one below and above for context
    range_thresholds = [t for t in THRESHOLDS if lower <= t <= upper]
    if not range_thresholds:
        range_thresholds = [THRESHOLDS[-2]]

    print(f"  {iso}: Analyzing thresholds {range_thresholds} (crossover {lower}-{upper}%)")

    all_records = []
    thresholds_loaded = []

    for t in range_thresholds:
        fpath = find_dg_file(iso, t)
        if fpath is None:
            print(f"    WARNING: No DG file for {iso} t={t}")
            continue

        df = pd.read_parquet(fpath)
        thresholds_loaded.append(t)

        # Vectorized: build records from DataFrame columns without iterrows
        n_rows = len(df)
        # Pre-extract columns as arrays for fast access
        scenarios = df['scenario'].values if 'scenario' in df.columns else [''] * n_rows
        growth_levels = df['growth_level'].values if 'growth_level' in df.columns else ['Medium'] * n_rows

        mix_vals = {}
        for res in MIX_RESOURCES:
            col = f'mix_{res}'
            mix_vals[res] = df[col].fillna(0).values if col in df.columns else np.zeros(n_rows)

        storage_vals = {}
        for res in STORAGE_RESOURCES:
            col = 'battery8_dispatch_pct' if res == 'battery8' else f'{res}_dispatch_pct'
            storage_vals[res] = df[col].fillna(0).values if col in df.columns else np.zeros(n_rows)

        eff_costs = df['cost_effective_cost'].fillna(0).values if 'cost_effective_cost' in df.columns else np.zeros(n_rows)
        total_costs = df['cost_total_cost'].fillna(0).values if 'cost_total_cost' in df.columns else np.zeros(n_rows)

        for i in range(n_rows):
            record = {
                'threshold': t,
                'scenario': scenarios[i],
                'growth_level': growth_levels[i],
            }
            for res in MIX_RESOURCES:
                record[res] = float(mix_vals[res][i])
            for res in STORAGE_RESOURCES:
                record[res] = float(storage_vals[res][i])
            record['effective_cost'] = float(eff_costs[i])
            record['total_cost'] = float(total_costs[i])
            all_records.append(record)

    if not all_records:
        return None

    print(f"    Loaded {len(all_records)} scenario-threshold-growth combos across {len(thresholds_loaded)} thresholds")

    # Compute no-regrets statistics
    total_combos = len(all_records)
    analysis = {}

    for res in ALL_RESOURCES:
        # Vectorized: numpy array + boolean mask instead of list comprehensions
        arr = np.array([r[res] for r in all_records])
        non_zero = arr[arr > 0.5]

        presence_rate = len(non_zero) / total_combos if total_combos > 0 else 0
        floor_val = float(non_zero.min()) if len(non_zero) > 0 else 0
        avg_val = float(non_zero.mean()) if len(non_zero) > 0 else 0
        ceiling_val = float(non_zero.max()) if len(non_zero) > 0 else 0
        p25_val = float(np.percentile(non_zero, 25)) if len(non_zero) > 0 else 0
        p75_val = float(np.percentile(non_zero, 75)) if len(non_zero) > 0 else 0

        analysis[res] = {
            'presenceRate': round(presence_rate, 4),
            'floor': round(floor_val, 1),
            'average': round(avg_val, 1),
            'ceiling': round(ceiling_val, 1),
            'p25': round(p25_val, 1),
            'p75': round(p75_val, 1),
            'isNoRegrets': presence_rate >= 0.80,
            'isConsensus': presence_rate >= 0.95,
            'totalCombos': total_combos,
            'nonZeroCount': len(non_zero)
        }

    return {
        'analysis': analysis,
        'rangeThresholds': thresholds_loaded,
        'totalCombos': total_combos,
        'crossover': crossover_range,
        'source': 'step3_dg_parquets'
    }


def main():
    print("=" * 70)
    print("Step 7: Extract No-Regrets Investment Data")
    print("=" * 70)

    # Load MAC data for crossover computation
    print("\nLoading MAC data...")
    mac_data = load_mac_data()

    results = {}

    for iso in ISOS:
        print(f"\nProcessing {iso} ({ISO_LABELS[iso]})...")

        # Check if DG files exist for this ISO
        has_dg = any(find_dg_file(iso, t) is not None for t in THRESHOLDS)
        if not has_dg:
            print(f"  SKIP: No DG parquet files found for {iso}")
            continue

        # Compute crossover range
        crossover = compute_crossover_range(iso, mac_data)
        if crossover is None:
            print(f"  SKIP: Could not compute crossover range for {iso}")
            continue

        print(f"  Crossover range: {crossover['lower']}-{crossover['upper']}% "
              f"(central: {crossover['central']}%)")

        # Analyze no-regrets from DG parquets
        nr_data = analyze_no_regrets(iso, crossover)
        if nr_data is None:
            print(f"  SKIP: No data loaded for {iso}")
            continue

        results[iso] = nr_data

        # Print summary
        nr_resources = [res for res, a in nr_data['analysis'].items() if a['isNoRegrets']]
        print(f"  No-regrets resources: {', '.join(nr_resources) if nr_resources else 'none'}")

    if not results:
        print("\nERROR: No results computed. Check that DG parquet files exist.")
        sys.exit(1)

    # Write output JS file
    print(f"\nWriting {OUTPUT_JS}...")
    js_content = "// Auto-generated by step6_extract_no_regrets.py\n"
    js_content += "// No-regrets resource investment data from Step 3 DG parquets\n"
    js_content += f"// Generated: {pd.Timestamp.now().isoformat()}\n\n"
    js_content += "const NO_REGRETS_DATA = " + json.dumps(results, indent=2) + ";\n"

    OUTPUT_JS.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JS.write_text(js_content)

    # Write JSON output
    print(f"Writing {OUTPUT_JSON}...")
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(results, indent=2))

    print(f"\nDone. {len(results)} ISOs processed.")
    for iso, data in results.items():
        nr = [r for r, a in data['analysis'].items() if a['isNoRegrets']]
        print(f"  {iso}: {data['crossover']['lower']}-{data['crossover']['upper']}% "
              f"| {len(nr)} no-regrets | {data['totalCombos']} combos")


if __name__ == '__main__':
    main()
