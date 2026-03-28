#!/usr/bin/env python3
"""
Extract empirical resource caps from Step 2.2 cost optimization results.
=========================================================================

Scans ALL parquets in data/step2.2-cost/ to find the maximum resource
percentage that appears in any winning mix, across all ISOs, thresholds,
and cost scenarios. Adds a configurable buffer (default +10pp) and rounds
up to the nearest step size.

These caps define the search windows for Step 1 mix generation — no need
to explore resource allocations that never win under any scenario.

Output: data/step2.2-cost/empirical_resource_caps.json
        (consumed by step1_1a_generate_mixes.py)

Usage:
  python scripts/extract_empirical_caps.py [--buffer 10] [--step 5]
"""

import argparse
import glob
import json
import math
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'data')
COST_DIR = os.path.join(DATA_DIR, 'step2.2-cost')

ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']

RESOURCE_COLS = [
    'mix_clean_firm', 'mix_solar', 'mix_wind', 'mix_offshore_wind',
    'mix_geothermal', 'mix_ccs_ccgt', 'mix_hydro'
]


def scan_parquets(cost_dir, exclude_tx_none=False):
    """Scan all parquets for max resource values per ISO.

    If exclude_tx_none=True, filters out scenarios with transmission='N' (None).
    Scenario format: {Ren}{Firm}{Batt}{LDES}_{Fuel}_{Tx}_{CCS}{45Q}_{Geo}
    Transmission is parts[2] after splitting on '_'.
    """
    all_files = sorted(glob.glob(os.path.join(cost_dir, '*.parquet')))
    print(f"Scanning {len(all_files)} parquet files in {cost_dir}/")
    if exclude_tx_none:
        print("  Excluding scenarios with transmission = None")

    iso_maxes = {iso: {} for iso in ISOS}

    for fpath in all_files:
        try:
            df = pd.read_parquet(fpath)
        except Exception as e:
            print(f"  SKIP {os.path.basename(fpath)}: {e}")
            continue

        valid_cols = [c for c in RESOURCE_COLS if c in df.columns]
        if not valid_cols or 'iso' not in df.columns:
            continue

        # Filter out transmission=None scenarios if requested
        if exclude_tx_none and 'scenario' in df.columns:
            before = len(df)
            # Transmission is the 3rd underscore-delimited part (index 2)
            tx_level = df['scenario'].str.split('_').str[2]
            df = df[tx_level != 'N']
            if len(df) == 0:
                continue

        df['total_procurement'] = df[valid_cols].sum(axis=1)

        for iso in df['iso'].unique():
            if iso not in ISOS:
                continue
            idf = df[df['iso'] == iso]
            m = iso_maxes[iso]
            m['total'] = max(m.get('total', 0), float(idf['total_procurement'].max()))
            for c in valid_cols:
                key = c.replace('mix_', '')
                m[key] = max(m.get(key, 0), float(idf[c].max()))

    return iso_maxes


def apply_buffer(iso_maxes, buffer_pp, step_size):
    """Add buffer and round up to nearest step size."""
    caps = {}
    for iso in ISOS:
        m = iso_maxes.get(iso, {})
        caps[iso] = {}
        for key, val in m.items():
            capped = int(math.ceil((val + buffer_pp) / step_size) * step_size)
            caps[iso][key] = capped
    return caps


def main():
    parser = argparse.ArgumentParser(description='Extract empirical resource caps')
    parser.add_argument('--buffer', type=int, default=10, help='Buffer in percentage points (default: 10)')
    parser.add_argument('--step', type=int, default=5, help='Round up to nearest step (default: 5)')
    parser.add_argument('--exclude-tx-none', action='store_true', default=True,
                        help='Exclude scenarios with transmission=None (default: True)')
    parser.add_argument('--include-tx-none', action='store_true',
                        help='Include scenarios with transmission=None')
    args = parser.parse_args()

    exclude_tx = args.exclude_tx_none and not args.include_tx_none
    raw_maxes = scan_parquets(COST_DIR, exclude_tx_none=exclude_tx)

    # Print raw maxes
    resources = ['total', 'clean_firm', 'solar', 'wind', 'offshore_wind', 'hydro', 'geothermal', 'ccs_ccgt']
    short = {'total': 'Total', 'clean_firm': 'CF', 'solar': 'Solar', 'wind': 'Wind',
             'offshore_wind': 'OSW', 'hydro': 'Hydro', 'geothermal': 'Geo', 'ccs_ccgt': 'CCS'}

    print(f"\nRAW MAX (across all parquets, all scenarios, all thresholds)")
    hdr = f"{'ISO':<8}" + "".join(f"{short[r]:>8}" for r in resources)
    print(hdr)
    print("-" * len(hdr))
    for iso in ISOS:
        m = raw_maxes[iso]
        vals = "".join(f"{m.get(r, 0):>8.0f}" for r in resources)
        print(f"{iso:<8}{vals}")

    # Apply buffer
    caps = apply_buffer(raw_maxes, args.buffer, args.step)

    print(f"\nWITH +{args.buffer}pp BUFFER (rounded to nearest {args.step})")
    print(hdr)
    print("-" * len(hdr))
    for iso in ISOS:
        c = caps[iso]
        vals = "".join(f"{c.get(r, 0):>8}" for r in resources)
        print(f"{iso:<8}{vals}")

    # Compute grid size reduction
    print(f"\nGRID SIZE REDUCTION vs current (0-350 total, 0-100 per resource at 5% steps)")
    for iso in ISOS:
        c = caps[iso]
        # Current: product of (cap/step + 1) for each resource
        current_dims = {
            'clean_firm': 100, 'solar': 100, 'wind': 100,
            'offshore_wind': max(c.get('offshore_wind', 0), 0),
            'hydro': max(c.get('hydro', 0), 0),
            'geothermal': max(c.get('geothermal', 0), 0),
        }
        old_total = 1
        new_total = 1
        for r in ['clean_firm', 'solar', 'wind', 'offshore_wind', 'hydro', 'geothermal']:
            old_levels = (100 // args.step) + 1 if r in ['clean_firm', 'solar', 'wind'] else (current_dims[r] // args.step) + 1
            new_cap = c.get(r, 0)
            new_levels = max((new_cap // args.step) + 1, 1)
            if old_levels > 1:
                old_total *= old_levels
            if new_levels > 1:
                new_total *= new_levels
        reduction = (1 - new_total / old_total) * 100 if old_total > 0 else 0
        print(f"  {iso}: {old_total:>12,} → {new_total:>12,} grid points ({reduction:.0f}% reduction)")

    # Save
    output = {
        'buffer_pp': args.buffer,
        'step_size': args.step,
        'raw_maxes': raw_maxes,
        'caps': caps,
    }
    out_path = os.path.join(COST_DIR, 'empirical_resource_caps.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
