#!/usr/bin/env python3
"""
Shared Parquet I/O — per-ISO parquet ↔ nested dict conversion
==============================================================
Used by post-processing scripts (Steps 4-5) to read Step 3 cost optimization
parquets into the nested dict format expected by processing functions.
"""

import os
import sys

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas required. Install: pip install pandas pyarrow")
    sys.exit(1)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ALL_ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']
RESOURCE_TYPES = ['clean_firm', 'solar', 'wind', 'ccs_ccgt', 'hydro']

# Fallback demand (MWh) for ISOs without annual_demand_mwh in parquet
REGIONAL_DEMAND_MWH = {
    'CAISO': 224.039e6, 'ERCOT': 488.020e6, 'PJM': 843.331e6,
    'NYISO': 151.599e6, 'NEISO': 115.336e6, 'MISO': 660.000e6, 'SPP': 296.000e6,
}

# Default parquet directory for cost optimization results
DEFAULT_INPUT_DIRS = [
    os.path.join(SCRIPT_DIR, 'data', 'step3-cost-opt-parquets'),
]


def find_parquet(input_dir, iso):
    """Find parquet file for an ISO in the given directory."""
    for prefix in ['step3_co_', 'co2_']:
        path = os.path.join(input_dir, f'{prefix}{iso}.parquet')
        if os.path.exists(path):
            return path
    return None


def find_input_dir(isos=None, preferred_dirs=None):
    """Find the first directory containing parquet files for the given ISOs."""
    dirs = preferred_dirs or DEFAULT_INPUT_DIRS
    for d in dirs:
        if not os.path.isdir(d):
            continue
        try:
            parquets = [f for f in os.listdir(d) if f.endswith('.parquet')]
        except OSError:
            continue
        if parquets:
            if isos:
                for iso in isos:
                    if find_parquet(d, iso):
                        return d
            else:
                return d
    return None


def load_from_parquets(input_dir, isos):
    """
    Read per-ISO parquets and reconstruct nested dict format.

    Reads step3_co_<ISO>.parquet files from cost optimization output.

    Returns:
        data: {
            'results': {
                iso: {
                    'annual_demand_mwh': float,
                    'thresholds': {
                        t_str: {
                            'scenarios': {
                                scenario_key: {
                                    'resource_mix': {...},
                                    'costs': {...},
                                    'co2_abated': {...},       # if present
                                    'no_45q_costs': {...},     # if present (step4)
                                    'gas_backup': {...},       # if present (step4)
                                    'tranche_costs': {...},    # if present
                                    'hourly_match_score': float,
                                    'battery_dispatch_pct': int,
                                    'battery8_dispatch_pct': int,
                                    'ldes_dispatch_pct': int,
                                    'h2_dispatch_pct': int,
                                }
                            }
                        }
                    }
                }
            }
        }
    """
    data = {'results': {}}
    loaded_count = 0

    for iso in isos:
        parquet_path = find_parquet(input_dir, iso)
        if parquet_path is None:
            print(f"  WARNING: No parquet found for {iso} in {input_dir} — skipping")
            continue

        df = pd.read_parquet(parquet_path)
        print(f"  Loaded {iso}: {len(df):,} rows from {os.path.basename(parquet_path)} "
              f"({os.path.getsize(parquet_path) / 1024:.0f} KB)")

        annual_demand = (float(df['annual_demand_mwh'].iloc[0])
                         if 'annual_demand_mwh' in df.columns
                         else REGIONAL_DEMAND_MWH.get(iso, 0))

        iso_data = {'annual_demand_mwh': annual_demand, 'thresholds': {}}

        for threshold, thr_group in df.groupby('threshold'):
            if isinstance(threshold, float) and threshold != int(threshold):
                t_str = str(threshold)
            else:
                t_str = str(int(threshold))

            scenarios = {}
            for _, row in thr_group.iterrows():
                sc_key = row['scenario']

                # Resource mix
                resource_mix = {}
                for rtype in RESOURCE_TYPES:
                    col = f'mix_{rtype}'
                    resource_mix[rtype] = int(row[col]) if col in row.index else 0

                # Costs
                costs = {}
                for ckey in ['total_cost', 'effective_cost', 'incremental', 'wholesale']:
                    col = f'cost_{ckey}'
                    costs[ckey] = float(row[col]) if col in row.index else 0.0

                scenario = {
                    'resource_mix': resource_mix,
                    'costs': costs,
                    'hourly_match_score': float(row['hourly_match_score']) if 'hourly_match_score' in row.index else 0.0,
                    'battery_dispatch_pct': int(row['battery_dispatch_pct']) if 'battery_dispatch_pct' in row.index else 0,
                    'battery8_dispatch_pct': int(row['battery8_dispatch_pct']) if 'battery8_dispatch_pct' in row.index else 0,
                    'ldes_dispatch_pct': int(row['ldes_dispatch_pct']) if 'ldes_dispatch_pct' in row.index else 0,
                    'h2_dispatch_pct': int(row['h2_dispatch_pct']) if 'h2_dispatch_pct' in row.index else 0,
                }

                # CO2 data (if present from previous recompute)
                co2_cols = [c for c in row.index if c.startswith('co2_')]
                if co2_cols:
                    co2 = {}
                    for col in co2_cols:
                        key = col[len('co2_'):]
                        val = row[col]
                        if pd.notna(val):
                            try:
                                co2[key] = float(val)
                            except (ValueError, TypeError):
                                co2[key] = val
                    if co2:
                        scenario['co2_abated'] = co2

                # Step 4: no_45q_costs
                no45q = {}
                for ckey in ['total_cost', 'effective_cost', 'incremental', 'wholesale']:
                    col = f'no45q_{ckey}'
                    if col in row.index and pd.notna(row[col]):
                        no45q[ckey] = float(row[col])
                if no45q:
                    scenario['no_45q_costs'] = no45q

                # Step 4: gas_backup (RA fields)
                gb = {}
                ra_cols = [c for c in row.index if c.startswith('ra_')]
                for col in ra_cols:
                    key = col[len('ra_'):]
                    val = row[col]
                    if pd.notna(val):
                        gb[key] = float(val)
                if gb:
                    scenario['gas_backup'] = gb

                # Tranche costs (passthrough)
                tranche = {}
                tranche_cols = [c for c in row.index if c.startswith('tranche_')]
                for col in tranche_cols:
                    key = col[len('tranche_'):]
                    val = row[col]
                    if pd.notna(val):
                        tranche[key] = float(val)
                if tranche:
                    scenario['tranche_costs'] = tranche

                scenarios[sc_key] = scenario

            iso_data['thresholds'][t_str] = {'scenarios': scenarios}

        data['results'][iso] = iso_data
        loaded_count += 1

    print(f"  Loaded {loaded_count} ISOs from parquet files")
    return data


def save_to_parquets(data, output_dir, isos, file_prefix='step4_'):
    """Flatten nested dict back to per-ISO parquets.

    Preserves all fields: step 3 base columns, step 4 additions, and
    any CO2/MAC columns added by post-processors.
    """
    os.makedirs(output_dir, exist_ok=True)

    for iso in isos:
        if iso not in data.get('results', {}):
            continue

        iso_data = data['results'][iso]
        annual_demand = iso_data.get('annual_demand_mwh', 0)
        rows = []

        for t_str, t_data in iso_data.get('thresholds', {}).items():
            for sc_key, sc in t_data.get('scenarios', {}).items():
                row = {
                    'iso': iso,
                    'threshold': float(t_str),
                    'scenario': sc_key,
                    'annual_demand_mwh': annual_demand,
                }

                # Resource mix
                mix = sc.get('resource_mix', {})
                for rtype in RESOURCE_TYPES:
                    row[f'mix_{rtype}'] = mix.get(rtype, 0)

                # Physics
                row['hourly_match_score'] = sc.get('hourly_match_score', 0.0)
                row['battery_dispatch_pct'] = sc.get('battery_dispatch_pct', 0)
                row['battery8_dispatch_pct'] = sc.get('battery8_dispatch_pct', 0)
                row['ldes_dispatch_pct'] = sc.get('ldes_dispatch_pct', 0)
                row['h2_dispatch_pct'] = sc.get('h2_dispatch_pct', 0)

                # Costs
                costs = sc.get('costs', {})
                for ckey in ['total_cost', 'effective_cost', 'incremental', 'wholesale']:
                    row[f'cost_{ckey}'] = costs.get(ckey, 0.0)

                # Tranche costs
                for k, v in sc.get('tranche_costs', {}).items():
                    row[f'tranche_{k}'] = v

                # CO2 abated
                co2 = sc.get('co2_abated', {})
                for k, v in co2.items():
                    row[f'co2_{k}'] = v

                # no_45q_costs (passthrough from step 4)
                no45q = sc.get('no_45q_costs', {})
                for ckey in ['total_cost', 'effective_cost', 'incremental', 'wholesale']:
                    if ckey in no45q:
                        row[f'no45q_{ckey}'] = no45q[ckey]

                # gas_backup RA (passthrough from step 4)
                gb = sc.get('gas_backup', {})
                for k, v in gb.items():
                    if isinstance(v, dict):
                        for sk, sv in v.items():
                            row[f'ra_{k}_{sk}'] = sv
                    else:
                        row[f'ra_{k}'] = v

                rows.append(row)

        if not rows:
            continue

        df_out = pd.DataFrame(rows)
        out_path = os.path.join(output_dir, f'{file_prefix}{iso}.parquet')
        df_out.to_parquet(out_path, index=False, compression='zstd')
        print(f"  {os.path.basename(out_path)}: {len(df_out):,} rows, "
              f"{os.path.getsize(out_path) / 1e6:.1f} MB")


# ══════════════════════════════════════════════════════════════════════════════
# DEMAND GROWTH PARQUET I/O
# ══════════════════════════════════════════════════════════════════════════════

def find_dg_parquets(input_dir, iso, prefix='step4_dg_'):
    """Find all DG parquet files for an ISO in the given directory.

    Tries step4_dg_ first (post-processed), then step3_dg_ (raw).
    Returns list of file paths sorted by threshold.
    """
    import glob as glob_mod
    for pfx in [prefix, 'step4_dg_', 'step3_dg_']:
        pattern = os.path.join(input_dir, f'{pfx}{iso}_t*.parquet')
        files = sorted(glob_mod.glob(pattern))
        if files:
            return files
    return []


def load_dg_from_parquets(input_dir, isos, prefix='step4_dg_'):
    """Load demand growth parquets into nested dict format.

    Tries step4 DG files first, falls back to step3 DG files.

    Returns:
        dg_data: {
            iso: {
                threshold_str: {
                    (year, growth_level): {
                        scenario_key: {
                            'resource_mix': {...},
                            'costs': {...},
                            'no_45q_costs': {...},      # if step4
                            'gas_backup': {...},          # if step4
                            'hourly_match_score': float,
                            'growth_factor': float,
                            'annual_demand_mwh': float,
                            'year': int,
                            'growth_level': str,
                        }
                    }
                }
            }
        }
    """
    dg_data = {}

    for iso in isos:
        files = find_dg_parquets(input_dir, iso, prefix)
        if not files:
            # Also check step3 input directory
            for alt_dir in DEFAULT_INPUT_DIRS:
                files = find_dg_parquets(alt_dir, iso, 'step3_dg_')
                if files:
                    break
        if not files:
            continue

        iso_dg = {}
        total_rows = 0

        for fpath in files:
            df = pd.read_parquet(fpath)
            total_rows += len(df)

            for threshold, thr_group in df.groupby('threshold'):
                t_str = str(threshold) if threshold != int(threshold) else str(int(threshold))
                if t_str not in iso_dg:
                    iso_dg[t_str] = {}

                for _, row in thr_group.iterrows():
                    year = int(row['year']) if 'year' in row.index else 2025
                    g_level = str(row['growth_level']) if 'growth_level' in row.index else 'Medium'
                    sc_key = row['scenario']
                    dg_key = (year, g_level)

                    if dg_key not in iso_dg[t_str]:
                        iso_dg[t_str][dg_key] = {}

                    # Build scenario dict
                    resource_mix = {}
                    for rtype in RESOURCE_TYPES:
                        col = f'mix_{rtype}'
                        resource_mix[rtype] = int(row[col]) if col in row.index else 0

                    costs = {}
                    for ckey in ['total_cost', 'effective_cost', 'incremental', 'wholesale']:
                        col = f'cost_{ckey}'
                        costs[ckey] = float(row[col]) if col in row.index and pd.notna(row[col]) else 0.0

                    scenario = {
                        'resource_mix': resource_mix,
                        'costs': costs,
                        'hourly_match_score': float(row['hourly_match_score']) if 'hourly_match_score' in row.index else 0.0,
                        'growth_factor': float(row['growth_factor']) if 'growth_factor' in row.index and pd.notna(row['growth_factor']) else 1.0,
                        'annual_demand_mwh': float(row['annual_demand_mwh']) if 'annual_demand_mwh' in row.index and pd.notna(row['annual_demand_mwh']) else 0,
                        'year': year,
                        'growth_level': g_level,
                    }

                    # Step 4 additions (if present)
                    no45q = {}
                    for ckey in ['total_cost', 'effective_cost', 'incremental', 'wholesale']:
                        col = f'no45q_{ckey}'
                        if col in row.index and pd.notna(row[col]):
                            no45q[ckey] = float(row[col])
                    if no45q:
                        scenario['no_45q_costs'] = no45q

                    gb = {}
                    ra_cols = [c for c in row.index if c.startswith('ra_')]
                    for col in ra_cols:
                        key = col[len('ra_'):]
                        val = row[col]
                        if pd.notna(val):
                            gb[key] = float(val)
                    if gb:
                        scenario['gas_backup'] = gb

                    iso_dg[t_str][dg_key][sc_key] = scenario

        if iso_dg:
            dg_data[iso] = iso_dg
            n_dg_combos = sum(len(v) for v in iso_dg.values())
            print(f"  DG loaded {iso}: {total_rows:,} rows, "
                  f"{len(iso_dg)} thresholds, {n_dg_combos} (year,growth) combos")

    return dg_data
