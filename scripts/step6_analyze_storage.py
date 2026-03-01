#!/usr/bin/env python3
"""
Step 6: Battery & Hydrogen Storage Analysis
=============================================
Extracts storage dispatch metrics from the dispatch cache and compressed day
profiles, producing a standalone JS data file for the storage analysis dashboard.

Analyzes:
  - Battery (4hr/8hr) dispatch patterns: charge/discharge timing, utilization, cycling
  - LDES (100hr iron-air) multi-day bridging: window utilization, seasonal patterns
  - Green H2 (1000hr) seasonal storage: when it appears, feasibility thresholds
  - Storage contribution to CFE matching across thresholds
  - Storage sizing progression (how much storage is needed at each threshold)
  - Regional variation in storage need (solar-dominant vs wind-dominant ISOs)

Input:
  - data/step5-post-processing/dispatch_cache/{ISO}_dispatch_cache.parquet
  - dashboard/js/shared-data.js (FEASIBLE_MIXES for storage allocations)
  - dashboard/compressed_day_profiles.json (24hr compressed profiles)

Output:
  - dashboard/js/storage-analysis-data.js
"""

import json
import os
import sys
import re
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
DASHBOARD_DIR = os.path.join(ROOT_DIR, 'dashboard')

sys.path.insert(0, ROOT_DIR)

from scripts.dispatch_utils import (
    ISOS, H, RESOURCE_TYPES,
    BATTERY_EFFICIENCY, BATTERY_DURATION_HOURS,
    BATTERY8_EFFICIENCY, BATTERY8_DURATION_HOURS,
    LDES_EFFICIENCY, LDES_DURATION_HOURS, LDES_WINDOW_DAYS,
    H2_EFFICIENCY, H2_DURATION_HOURS, H2_WINDOW_DAYS,
    BASE_DEMAND_TWH,
)

THRESHOLDS = [50, 55, 60, 65, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 99.5, 99.9, 99.99]
THRESHOLD_LABELS = ['50', '55', '60', '65', '70', '75', '80', '85', '87.5', '90', '92.5', '95', '97.5', '99', '99.5', '99.9', '≥99.99']

# Shared-data.js threshold array (superset — includes sub-50 thresholds)
SD_THRESHOLDS = [10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 99.5, 99.9, 99.99]

# Storage cost assumptions (NREL ATB 2024 aligned)
STORAGE_COSTS = {
    'battery4': {'low': 115, 'medium': 140, 'high': 165},      # $/kWh capacity, 4hr duration → $/MWh = $/kWh / duration
    'battery8': {'low': 95, 'medium': 120, 'high': 145},       # $/kWh capacity, 8hr
    'ldes':     {'low': 30, 'medium': 50, 'high': 80},         # $/kWh, 100hr iron-air
    'h2':       {'low': 150, 'medium': 250, 'high': 400},      # $/MWh discharged (electrolysis + storage + turbine)
}


def parse_shared_data():
    """Extract FEASIBLE_MIXES from shared-data.js."""
    path = os.path.join(DASHBOARD_DIR, 'js', 'shared-data.js')
    with open(path) as f:
        content = f.read()

    # Find FEASIBLE_MIXES block
    match = re.search(r'const\s+FEASIBLE_MIXES\s*=\s*\{', content)
    if not match:
        print("WARNING: FEASIBLE_MIXES not found in shared-data.js")
        return {}

    # Extract by finding matching braces
    start = match.start()
    brace_count = 0
    end = start
    for i in range(match.end() - 1, len(content)):
        if content[i] == '{':
            brace_count += 1
        elif content[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                end = i + 1
                break

    block = content[start:end]
    # Extract the object assignment
    eq_pos = block.index('=')
    obj_str = block[eq_pos+1:].strip().rstrip(';')

    # Convert JS object to valid JSON
    # Replace unquoted keys and trailing commas
    obj_str = re.sub(r'(\w+)\s*:', r'"\1":', obj_str)
    obj_str = re.sub(r',\s*}', '}', obj_str)
    obj_str = re.sub(r',\s*]', ']', obj_str)

    try:
        return json.loads(obj_str)
    except json.JSONDecodeError as e:
        print(f"WARNING: Failed to parse FEASIBLE_MIXES: {e}")
        return {}


def load_compressed_profiles():
    """Load compressed 24-hour day profiles."""
    path = os.path.join(DASHBOARD_DIR, 'compressed_day_profiles.json')
    if not os.path.exists(path):
        print("WARNING: compressed_day_profiles.json not found")
        return {}
    with open(path) as f:
        return json.load(f)


def extract_storage_by_threshold(feasible_mixes):
    """Extract storage allocation (battery/ldes/h2) by ISO and threshold.

    Returns dict: {iso: {threshold_idx: {battery, battery8, ldes, h2}}}
    """
    result = {}
    for iso in ISOS:
        iso_data = feasible_mixes.get(iso, {})
        result[iso] = {}

        for i, th in enumerate(SD_THRESHOLDS):
            storage = {
                'battery': 0, 'battery8': 0, 'ldes': 0, 'h2': 0,
                'clean_firm': 0, 'solar': 0, 'wind': 0, 'ccs_ccgt': 0, 'hydro': 0,
            }
            for res in storage:
                arr = iso_data.get(res, [])
                if i < len(arr) and arr[i] is not None:
                    storage[res] = arr[i]
            result[iso][i] = storage

    return result


def compute_storage_metrics(storage_by_threshold):
    """Compute derived storage metrics for each ISO across thresholds.

    Returns structured data for dashboard visualization.
    """
    metrics = {}

    for iso in ISOS:
        iso_metrics = {
            'thresholds': [],
            'battery4_pct': [],
            'battery8_pct': [],
            'ldes_pct': [],
            'h2_pct': [],
            'total_storage_pct': [],
            'storage_share_of_clean': [],
            'battery_twh': [],
            'ldes_twh': [],
            'h2_twh': [],
            'total_storage_twh': [],
            'resource_mix': {r: [] for r in ['clean_firm', 'solar', 'wind', 'ccs_ccgt', 'hydro']},
        }

        demand_twh = BASE_DEMAND_TWH.get(iso, 0)

        for i, th in enumerate(SD_THRESHOLDS):
            if th < 50:
                continue  # Skip below 50% — not relevant for storage analysis

            s = storage_by_threshold[iso].get(i, {})
            bat4 = s.get('battery', 0)
            bat8 = s.get('battery8', 0)
            ldes = s.get('ldes', 0)
            h2 = s.get('h2', 0)
            total_storage = bat4 + bat8 + ldes + h2

            # Resource mix percentages
            total_clean = sum(s.get(r, 0) for r in ['clean_firm', 'solar', 'wind', 'ccs_ccgt', 'hydro'])

            # Storage as share of clean procurement
            storage_share = (total_storage / (total_clean + total_storage) * 100) if (total_clean + total_storage) > 0 else 0

            # Convert dispatch % to TWh (dispatch_pct is % of annual demand dispatched by storage)
            bat_twh = bat4 * demand_twh / 100.0
            ldes_twh_val = ldes * demand_twh / 100.0
            h2_twh_val = h2 * demand_twh / 100.0

            iso_metrics['thresholds'].append(th)
            iso_metrics['battery4_pct'].append(round(bat4, 2))
            iso_metrics['battery8_pct'].append(round(bat8, 2))
            iso_metrics['ldes_pct'].append(round(ldes, 2))
            iso_metrics['h2_pct'].append(round(h2, 2))
            iso_metrics['total_storage_pct'].append(round(total_storage, 2))
            iso_metrics['storage_share_of_clean'].append(round(storage_share, 1))
            iso_metrics['battery_twh'].append(round(bat_twh, 2))
            iso_metrics['ldes_twh'].append(round(ldes_twh_val, 2))
            iso_metrics['h2_twh'].append(round(h2_twh_val, 2))
            iso_metrics['total_storage_twh'].append(round(bat_twh + ldes_twh_val + h2_twh_val, 2))

            for r in ['clean_firm', 'solar', 'wind', 'ccs_ccgt', 'hydro']:
                iso_metrics['resource_mix'][r].append(round(s.get(r, 0), 1))

        metrics[iso] = iso_metrics

    return metrics


def extract_compressed_day_storage(compressed_profiles):
    """Extract battery and LDES charge/discharge 24hr profiles from compressed day data.

    Returns per-ISO profiles showing when storage charges and discharges.
    """
    result = {}

    for iso in ISOS:
        iso_data = compressed_profiles.get(iso, {})
        profiles = iso_data.get('profiles', {})

        # Pick a few representative mixes at key thresholds
        # Mix key format: "CF_solar_wind_CCS_hydro_bat_bat8_ldes"
        iso_profiles = {
            'mix_keys': [],
            'battery_charge': [],
            'ldes_charge': [],
            'matched_by_resource': [],
            'gap': [],
            'demand': [],
        }

        for mix_key, mix_data in profiles.items():
            parts = mix_key.split('_')
            if len(parts) < 6:
                continue

            bat_pct = float(parts[5]) if len(parts) > 5 else 0
            ldes_pct = float(parts[7]) if len(parts) > 7 else 0

            # Include mixes with any storage
            if bat_pct > 0 or ldes_pct > 0:
                battery_charge = mix_data.get('battery_charge', [0]*24)
                ldes_charge = mix_data.get('ldes_charge', [0]*24)
                matched = mix_data.get('matched', {})
                gap = mix_data.get('gap', [0]*24)
                demand = mix_data.get('demand', [0]*24)

                iso_profiles['mix_keys'].append(mix_key)
                iso_profiles['battery_charge'].append([round(v, 6) for v in battery_charge])
                iso_profiles['ldes_charge'].append([round(v, 6) for v in ldes_charge])

                # Extract battery/LDES dispatch from matched
                bat_matched = matched.get('battery', [0]*24)
                ldes_matched = matched.get('ldes', [0]*24)
                iso_profiles['matched_by_resource'].append({
                    'battery': [round(v, 6) for v in bat_matched],
                    'ldes': [round(v, 6) for v in ldes_matched],
                    'clean_firm': [round(v, 6) for v in matched.get('clean_firm', [0]*24)],
                    'solar': [round(v, 6) for v in matched.get('solar', [0]*24)],
                    'wind': [round(v, 6) for v in matched.get('wind', [0]*24)],
                })
                iso_profiles['gap'].append([round(v, 6) for v in gap])
                iso_profiles['demand'].append([round(v, 6) for v in demand])

        result[iso] = iso_profiles

    return result


def compute_storage_value_metrics(storage_by_threshold):
    """Compute storage value: how much gap would exist without storage at each threshold.

    Shows the "value add" of storage — the gap between resource-only supply and demand
    that storage fills.
    """
    result = {}
    for iso in ISOS:
        iso_vals = {
            'thresholds': [],
            'gap_without_storage_pct': [],
            'gap_with_storage_pct': [],
            'storage_fills_pct': [],
        }

        for i, th in enumerate(SD_THRESHOLDS):
            if th < 50:
                continue
            s = storage_by_threshold[iso].get(i, {})
            total_storage = s.get('battery', 0) + s.get('battery8', 0) + s.get('ldes', 0) + s.get('h2', 0)

            # Without storage: gap = threshold target - (clean resources only)
            total_clean_gen = sum(s.get(r, 0) for r in ['clean_firm', 'solar', 'wind', 'ccs_ccgt', 'hydro'])

            # Storage fills the temporal mismatch — it doesn't add new energy,
            # it shifts surplus from high-gen hours to deficit hours
            # The value is approximately total_storage dispatch %
            iso_vals['thresholds'].append(th)
            iso_vals['gap_without_storage_pct'].append(round(total_storage, 2))
            iso_vals['gap_with_storage_pct'].append(0)
            iso_vals['storage_fills_pct'].append(round(total_storage, 2))

        result[iso] = iso_vals
    return result


def compute_h2_feasibility():
    """Analyze when hydrogen storage becomes part of optimal mixes.

    H2 is only included at ≥95% thresholds in the PFS generator.
    This documents the constraint and shows where H2 would be needed.
    """
    return {
        'constraint_note': 'Green H2 storage only evaluated at ≥95% CFE thresholds',
        'h2_params': {
            'efficiency_pct': round(H2_EFFICIENCY * 100, 1),
            'duration_hours': H2_DURATION_HOURS,
            'window_days': H2_WINDOW_DAYS,
            'description': 'Electrolysis (70%) × Storage (95%) × Turbine (55%) = 35% round-trip',
        },
        'comparison': {
            'technologies': ['Battery 4hr', 'Battery 8hr', 'LDES 100hr', 'Green H2 1000hr'],
            'efficiency_pct': [85, 85, 50, 35],
            'duration_hours': [4, 8, 100, 1000],
            'use_case': [
                'Daily solar shifting (day→evening)',
                'Extended daily shifting (day→night)',
                'Multi-day weather bridging (3-7 day lulls)',
                'Seasonal storage (weeks-to-months)'
            ],
            'cost_low': [STORAGE_COSTS['battery4']['low'],
                        STORAGE_COSTS['battery8']['low'],
                        STORAGE_COSTS['ldes']['low'],
                        STORAGE_COSTS['h2']['low']],
            'cost_med': [STORAGE_COSTS['battery4']['medium'],
                        STORAGE_COSTS['battery8']['medium'],
                        STORAGE_COSTS['ldes']['medium'],
                        STORAGE_COSTS['h2']['medium']],
            'cost_high': [STORAGE_COSTS['battery4']['high'],
                         STORAGE_COSTS['battery8']['high'],
                         STORAGE_COSTS['ldes']['high'],
                         STORAGE_COSTS['h2']['high']],
            'cost_unit': ['$/kWh', '$/kWh', '$/kWh', '$/MWh discharged'],
        },
    }


def compute_regional_storage_drivers():
    """Explain why different ISOs need different storage mixes.

    Solar-dominated ISOs (CAISO, ERCOT) need daily shifting (battery).
    Wind-dominated ISOs (SPP, MISO) need multi-day bridging (LDES).
    Nuclear-heavy ISOs (PJM, NEISO) need less storage overall.
    """
    drivers = {}
    for iso in ISOS:
        solar_share = 0
        wind_share = 0
        firm_share = 0
        from scripts.dispatch_utils import GRID_MIX_SHARES
        mix = GRID_MIX_SHARES.get(iso, {})
        total = sum(mix.values()) or 1
        solar_share = mix.get('solar', 0) / total * 100
        wind_share = mix.get('wind', 0) / total * 100
        firm_share = mix.get('clean_firm', 0) / total * 100

        if solar_share > 30:
            primary_driver = 'solar_dominant'
            storage_need = 'Battery (daily shifting) primary, LDES secondary'
        elif wind_share > 25:
            primary_driver = 'wind_dominant'
            storage_need = 'LDES (multi-day) primary, battery secondary'
        elif firm_share > 25:
            primary_driver = 'firm_dominant'
            storage_need = 'Minimal storage needed — baseload covers most hours'
        else:
            primary_driver = 'mixed'
            storage_need = 'Balanced battery + LDES depending on threshold'

        drivers[iso] = {
            'solar_share': round(solar_share, 1),
            'wind_share': round(wind_share, 1),
            'firm_share': round(firm_share, 1),
            'primary_driver': primary_driver,
            'storage_need': storage_need,
            'demand_twh': BASE_DEMAND_TWH.get(iso, 0),
        }
    return drivers


def write_js_data(metrics, compressed_storage, h2_feasibility, storage_value, regional_drivers, parquet_stats=None):
    """Write all storage analysis data to a standalone JS file."""
    output = {
        'storage_metrics': metrics,
        'compressed_day_storage': compressed_storage,
        'h2_feasibility': h2_feasibility,
        'storage_value': storage_value,
        'regional_drivers': regional_drivers,
        'storage_params': {
            'battery4': {
                'efficiency': BATTERY_EFFICIENCY,
                'duration_hours': BATTERY_DURATION_HOURS,
                'description': '4hr Li-ion, 85% RTE, daily cycle',
            },
            'battery8': {
                'efficiency': BATTERY8_EFFICIENCY,
                'duration_hours': BATTERY8_DURATION_HOURS,
                'description': '8hr Li-ion, 85% RTE, daily cycle',
            },
            'ldes': {
                'efficiency': LDES_EFFICIENCY,
                'duration_hours': LDES_DURATION_HOURS,
                'window_days': LDES_WINDOW_DAYS,
                'description': '100hr iron-air, 50% RTE, 7-day window',
            },
            'h2': {
                'efficiency': H2_EFFICIENCY,
                'duration_hours': H2_DURATION_HOURS,
                'window_days': H2_WINDOW_DAYS,
                'description': '1000hr green H2, 35% RTE, 30-day window',
            },
        },
        'thresholds': [th for th in SD_THRESHOLDS if th >= 50],
        'threshold_labels': THRESHOLD_LABELS,
        'isos': ISOS,
        'parquet_stats': parquet_stats or {},
    }

    out_path = os.path.join(DASHBOARD_DIR, 'js', 'storage-analysis-data.js')
    with open(out_path, 'w') as f:
        f.write('// Auto-generated by step6_analyze_storage.py\n')
        f.write('// Storage dispatch analysis data for battery, LDES, and hydrogen\n')
        f.write(f'const STORAGE_ANALYSIS = {json.dumps(output, separators=(",", ":"))};\n')

    print(f"Wrote {out_path} ({os.path.getsize(out_path):,} bytes)")
    return out_path


def extract_parquet_storage_stats():
    """Extract storage statistics directly from Step 4 parquets.

    Provides richer data than shared-data.js: distribution of storage across
    ALL feasible mixes (not just the single optimal), showing storage landscape.
    """
    try:
        import pandas as pd
    except ImportError:
        print("  pandas not available — skipping parquet analysis")
        return {}

    pdir = os.path.join(DATA_DIR, 'step4-gas-ccs-parquets')
    if not os.path.exists(pdir):
        pdir = os.path.join(DATA_DIR, 'step3-cost-opt-parquets')
    if not os.path.exists(pdir):
        return {}

    result = {}
    for iso in ISOS:
        fpath = os.path.join(pdir, f'step4_{iso}.parquet')
        if not os.path.exists(fpath):
            fpath = os.path.join(pdir, f'step3_{iso}.parquet')
        if not os.path.exists(fpath):
            continue

        try:
            import pyarrow.parquet as pq
            schema = pq.read_schema(fpath)
            cols = schema.names
            read_cols = ['threshold']
            for c in ['battery_dispatch_pct', 'battery8_dispatch_pct',
                       'ldes_dispatch_pct', 'h2_dispatch_pct',
                       'cost_total_cost', 'score']:
                if c in cols:
                    read_cols.append(c)
            df = pd.read_parquet(fpath, columns=read_cols)

            iso_stats = {
                'thresholds_with_battery': [],
                'thresholds_with_ldes': [],
                'thresholds_with_h2': [],
                'battery_onset_threshold': None,
                'ldes_onset_threshold': None,
                'h2_onset_threshold': None,
                'storage_by_threshold': {},
            }

            for th in sorted(df['threshold'].unique()):
                tdf = df[df['threshold'] == th]
                bat = tdf['battery_dispatch_pct'] if 'battery_dispatch_pct' in tdf.columns else pd.Series(0, index=tdf.index)
                ldes = tdf['ldes_dispatch_pct'] if 'ldes_dispatch_pct' in tdf.columns else pd.Series(0, index=tdf.index)
                h2 = tdf['h2_dispatch_pct'] if 'h2_dispatch_pct' in tdf.columns else pd.Series(0, index=tdf.index)

                has_bat = (bat > 0).sum()
                has_ldes = (ldes > 0).sum()
                has_h2 = (h2 > 0).sum()

                if has_bat > 0:
                    iso_stats['thresholds_with_battery'].append(float(th))
                    if iso_stats['battery_onset_threshold'] is None:
                        iso_stats['battery_onset_threshold'] = float(th)
                if has_ldes > 0:
                    iso_stats['thresholds_with_ldes'].append(float(th))
                    if iso_stats['ldes_onset_threshold'] is None:
                        iso_stats['ldes_onset_threshold'] = float(th)
                if has_h2 > 0:
                    iso_stats['thresholds_with_h2'].append(float(th))
                    if iso_stats['h2_onset_threshold'] is None:
                        iso_stats['h2_onset_threshold'] = float(th)

                # Stats for mixes WITH storage
                has_storage = (bat > 0) | (ldes > 0) | (h2 > 0)
                iso_stats['storage_by_threshold'][str(float(th))] = {
                    'total_mixes': int(len(tdf)),
                    'mixes_with_storage': int(has_storage.sum()),
                    'pct_with_storage': round(has_storage.mean() * 100, 1),
                    'battery_max': round(float(bat.max()), 2),
                    'ldes_max': round(float(ldes.max()), 2),
                    'h2_max': round(float(h2.max()), 2),
                    'battery_mean_nonzero': round(float(bat[bat > 0].mean()), 2) if has_bat > 0 else 0,
                    'ldes_mean_nonzero': round(float(ldes[ldes > 0].mean()), 2) if has_ldes > 0 else 0,
                }

            result[iso] = iso_stats
            print(f"  {iso}: battery@{iso_stats['battery_onset_threshold']}, ldes@{iso_stats['ldes_onset_threshold']}")

        except Exception as e:
            print(f"  {iso}: ERROR reading parquet: {e}")

    return result


def main():
    print("=" * 70)
    print("Step 6: Battery & Hydrogen Storage Analysis")
    print("=" * 70)

    # 1. Parse FEASIBLE_MIXES from shared-data.js
    print("\n[1/6] Parsing FEASIBLE_MIXES from shared-data.js...")
    feasible_mixes = parse_shared_data()
    if not feasible_mixes:
        print("ERROR: Could not parse FEASIBLE_MIXES. Aborting.")
        sys.exit(1)
    print(f"  Found data for {len(feasible_mixes)} ISOs")

    # 2. Extract storage allocations by threshold
    print("\n[2/6] Extracting storage allocations by threshold...")
    storage_by_threshold = extract_storage_by_threshold(feasible_mixes)
    for iso in ISOS:
        n_with_storage = sum(
            1 for i in storage_by_threshold[iso].values()
            if i.get('battery', 0) > 0 or i.get('ldes', 0) > 0 or i.get('h2', 0) > 0
        )
        print(f"  {iso}: {n_with_storage} thresholds with storage")

    # 3. Compute derived metrics
    print("\n[3/6] Computing storage metrics...")
    metrics = compute_storage_metrics(storage_by_threshold)
    storage_value = compute_storage_value_metrics(storage_by_threshold)

    # 4. Load compressed day profiles for dispatch visualization
    print("\n[4/6] Loading compressed day profiles...")
    compressed = load_compressed_profiles()
    compressed_storage = extract_compressed_day_storage(compressed)
    for iso in ISOS:
        n = len(compressed_storage[iso]['mix_keys'])
        print(f"  {iso}: {n} mixes with storage dispatch profiles")

    # 5. Extract parquet-level storage statistics
    print("\n[5/6] Extracting storage statistics from parquets...")
    parquet_stats = extract_parquet_storage_stats()

    # 6. Compute additional analysis
    print("\n[6/6] Computing H2 feasibility & regional drivers...")
    h2_feasibility = compute_h2_feasibility()
    regional_drivers = compute_regional_storage_drivers()

    # Write output
    print("\nWriting storage-analysis-data.js...")
    out_path = write_js_data(metrics, compressed_storage, h2_feasibility,
                              storage_value, regional_drivers,
                              parquet_stats=parquet_stats)

    print("\nDone!")
    return out_path


if __name__ == '__main__':
    main()
