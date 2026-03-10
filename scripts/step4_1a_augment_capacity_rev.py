#!/usr/bin/env python3
"""Augment existing Step 5B LMP parquets with capacity market revenue columns.

Post-hoc computation: reads existing LMP parquets + step3 resource mixes,
computes capacity market revenue per resource category, and overwrites
the LMP parquets with new columns added.

This avoids needing the dispatch cache (Step 4) — capacity revenue depends
only on ISO, threshold (clean%), and resource mix, not hourly profiles.
"""
import os
import sys
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'scripts'))

from pipeline_config import (
    CAPACITY_MARKET_PRICES, CAPACITY_DEGRADATION_ALPHA,
    PEAK_CAPACITY_CREDITS, RESOURCE_CAPACITY_FACTORS,
)

LMP_DIR = os.path.join(ROOT_DIR, 'data', 'step4-analysis', 'lmp')
STEP3_DIR = os.path.join(ROOT_DIR, 'data', 'step2.2-cost')

ALL_ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']

# Resource → category mapping
CLEAN_FIRM_RES = ['clean_firm', 'geothermal']
VRE_RES = ['solar', 'wind', 'offshore_wind']
STORAGE_RES = ['battery', 'battery8', 'ldes', 'h2']
CCS_RES = ['ccs_ccgt']


def compute_per_resource_cap_rev(iso, threshold):
    """Compute capacity revenue ($/MWh) per resource at given ISO/threshold."""
    base_price = CAPACITY_MARKET_PRICES.get(iso, 0)
    alpha = CAPACITY_DEGRADATION_ALPHA.get(iso, 0)
    degraded = base_price * max(0, 1.0 - alpha * threshold / 100.0)

    revs = {}
    for res in (CLEAN_FIRM_RES + VRE_RES + STORAGE_RES + CCS_RES + ['hydro']):
        elcc = PEAK_CAPACITY_CREDITS.get(res, 0)
        cf = RESOURCE_CAPACITY_FACTORS.get(res, {}).get(iso, 0.30)
        if cf > 0 and elcc > 0 and degraded > 0:
            revs[res] = degraded * elcc / (cf * 8.760)
        else:
            revs[res] = 0.0
    return revs


def weighted_category_rev(per_res_rev, mix_row, cat_resources):
    """Weighted average cap rev for a category, weighted by resource share."""
    total_share = 0
    weighted = 0
    for r in cat_resources:
        col = f'mix_{r}'
        share = mix_row.get(col, 0)
        if share > 0:
            weighted += per_res_rev.get(r, 0) * share
            total_share += share
    if total_share > 0:
        return weighted / total_share
    # If no resources present, return the first resource's rate (theoretical)
    return per_res_rev.get(cat_resources[0], 0)


def process_iso(iso):
    """Augment LMP parquet for one ISO with capacity revenue columns."""
    lmp_path = os.path.join(LMP_DIR, f'{iso}_lmp.parquet')
    s3_path = os.path.join(STEP3_DIR, f'step3_co_{iso}.parquet')

    if not os.path.exists(lmp_path):
        print(f"  {iso}: No LMP parquet, skipping")
        return
    if not os.path.exists(s3_path):
        print(f"  {iso}: No step3 parquet, skipping")
        return

    lmp = pd.read_parquet(lmp_path)
    s3 = pd.read_parquet(s3_path)

    # Get resource mix columns from step3
    mix_cols = [c for c in s3.columns if c.startswith('mix_')]
    storage_cols = ['battery_dispatch_pct', 'battery8_dispatch_pct',
                    'ldes_dispatch_pct', 'h2_dispatch_pct']
    keep_cols = ['iso', 'threshold', 'scenario'] + mix_cols
    # Keep only cols that exist
    keep_cols = [c for c in keep_cols if c in s3.columns]
    storage_cols = [c for c in storage_cols if c in s3.columns]
    keep_cols += storage_cols

    s3_slim = s3[keep_cols].drop_duplicates(subset=['iso', 'threshold', 'scenario'])

    # Merge on iso/threshold/scenario
    merged = lmp.merge(s3_slim, on=['iso', 'threshold', 'scenario'], how='left')

    # Vectorized capacity revenue computation
    n = len(merged)
    cap_sys = np.zeros(n)
    cap_cf = np.zeros(n)  # clean firm
    cap_vre = np.zeros(n)
    cap_stor = np.zeros(n)
    cap_ccs = np.zeros(n)

    # Pre-compute per-resource cap rev for each unique threshold
    unique_thresholds = merged['threshold'].unique()
    per_res_by_thresh = {}
    for t in unique_thresholds:
        per_res_by_thresh[t] = compute_per_resource_cap_rev(iso, t)

    for idx, row in merged.iterrows():
        t = row['threshold']
        pr = per_res_by_thresh[t]
        row_dict = row.to_dict()

        cap_cf[idx] = weighted_category_rev(pr, row_dict, CLEAN_FIRM_RES)
        cap_vre[idx] = weighted_category_rev(pr, row_dict, VRE_RES)
        cap_stor[idx] = weighted_category_rev(pr, row_dict, STORAGE_RES)
        cap_ccs[idx] = weighted_category_rev(pr, row_dict, CCS_RES)

        # System average: weighted across all resources
        all_res = CLEAN_FIRM_RES + VRE_RES + STORAGE_RES + CCS_RES + ['hydro']
        total_share = 0
        weighted_rev = 0
        for r in all_res:
            share = row_dict.get(f'mix_{r}', 0)
            if share > 0:
                weighted_rev += pr.get(r, 0) * share
                total_share += share
        cap_sys[idx] = weighted_rev / total_share if total_share > 0 else 0

    # Add new columns
    avg_lmp = merged['avg_lmp'].values

    merged['capacity_rev_system_mwh'] = np.round(cap_sys, 2)
    merged['capacity_rev_clean_firm_mwh'] = np.round(cap_cf, 2)
    merged['capacity_rev_vre_mwh'] = np.round(cap_vre, 2)
    merged['capacity_rev_storage_mwh'] = np.round(cap_stor, 2)
    merged['capacity_rev_ccs_mwh'] = np.round(cap_ccs, 2)

    merged['total_market_rev_mwh'] = np.round(avg_lmp + cap_sys, 2)
    merged['total_rev_clean_firm_mwh'] = np.round(avg_lmp + cap_cf, 2)
    merged['total_rev_vre_mwh'] = np.round(avg_lmp + cap_vre, 2)
    merged['total_rev_storage_mwh'] = np.round(avg_lmp + cap_stor, 2)
    merged['total_rev_ccs_mwh'] = np.round(avg_lmp + cap_ccs, 2)

    # Drop the mix columns (they came from step3 merge, not part of LMP schema)
    drop_cols = [c for c in merged.columns
                 if c.startswith('mix_') or c.endswith('_dispatch_pct')]
    merged = merged.drop(columns=drop_cols, errors='ignore')

    # Save back
    merged.to_parquet(lmp_path, index=False, compression='zstd')
    print(f"  {iso}: {len(merged)} rows augmented with capacity revenue")

    # Summary stats
    cap_isos = merged.groupby('threshold').agg({
        'capacity_rev_system_mwh': 'mean',
        'capacity_rev_clean_firm_mwh': 'mean',
        'total_market_rev_mwh': 'mean',
        'avg_lmp': 'mean',
    }).round(2)
    print(f"    Sample (avg across scenarios):")
    for t in [50, 75, 90, 95]:
        if t in cap_isos.index:
            r = cap_isos.loc[t]
            print(f"      {t}%: LMP=${r['avg_lmp']:.1f} + CapRev=${r['capacity_rev_system_mwh']:.1f}"
                  f" = Total=${r['total_market_rev_mwh']:.1f}"
                  f"  |  Nuclear rev=${r['capacity_rev_clean_firm_mwh']:.1f}/MWh")


if __name__ == '__main__':
    print("=" * 70)
    print("  Step 5B Augmentation: Adding Capacity Market Revenue")
    print("=" * 70)

    for iso in ALL_ISOS:
        process_iso(iso)

    print("\nDone.")
