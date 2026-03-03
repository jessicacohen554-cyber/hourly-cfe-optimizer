#!/usr/bin/env python3
"""
Export track parquets to track_results.json for dashboard consumption.

Reads:
  - dashboard/track_scenarios.parquet (NB + CTR scenario results)

Writes:
  - dashboard/track_results.json

Strategy: For ISOs with full sweep data (many scenarios per threshold), we
precompute P10/P50/P90 percentiles server-side and only include full detail
for the medium scenario. For medium-only ISOs, the single scenario IS the
medium scenario with full detail.

This keeps the JSON small (~1-5 MB) instead of 196 MB with all scenarios.

Output JSON structure:
  {
    "meta": { "mode": "full"|"medium", "thresholds": [...], "isos": [...] },
    "results": {
      "<ISO>": {
        "newbuild": {
          "<threshold>": {
            "scenarios": {
              "<MEDIUM_KEY>": { full scenario detail }
            },
            "sweep": {  // only present when multiple scenarios exist
              "n": 5832,
              "p10": { "effective_cost": N },
              "p50": { "effective_cost": N },
              "p90": { "effective_cost": N }
            }
          }
        },
        "cost_to_replace": { ... same structure ... }
      }
    }
  }

The frontend code in cost_to_replace.html and newbuild_requirement.html
handles both formats: per-scenario iteration AND precomputed sweep percentiles.
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PQ_SCENARIOS = os.path.join(SCRIPT_DIR, 'dashboard', 'track_scenarios.parquet')
OUT_PATH = os.path.join(SCRIPT_DIR, 'dashboard', 'track_results.json')
STEP5_DIR = os.path.join(SCRIPT_DIR, 'data', 'step5-post-processing')

# Known medium scenario keys (non-CAISO and CAISO variants)
MEDIUM_KEYS = {'MMMM_M_M_M1_X', 'MMMM_M_M_M1_M'}


def build_scenario_dict(row):
    """Convert a parquet row to the nested scenario dict the frontend expects."""
    return {
        'resource_mix': {
            'clean_firm': int(row['mix_clean_firm']),
            'solar': int(row['mix_solar']),
            'wind': int(row['mix_wind']),
            'offshore_wind': int(row.get('mix_offshore_wind', 0)),
            'ccs_ccgt': int(row['mix_ccs_ccgt']),
            'hydro': int(row['mix_hydro']),
        },
        'costs': {
            'total_cost': round(float(row['cost_total_cost']), 2),
            'effective_cost': round(float(row['cost_effective_cost']), 2),
            'incremental': round(float(row['cost_incremental']), 2),
            'wholesale': round(float(row['cost_wholesale']), 2),
        },
        'hourly_match_score': round(float(row['hourly_match_score']), 2),
        'battery_dispatch_pct': round(float(row['battery_dispatch_pct']), 4),
        'battery8_dispatch_pct': round(float(row.get('battery8_dispatch_pct', 0)), 4),
        'ldes_dispatch_pct': round(float(row['ldes_dispatch_pct']), 4),
        'h2_dispatch_pct': round(float(row.get('h2_dispatch_pct', 0)), 4),
        'tranche_costs': {
            'cf_existing_twh': round(float(row.get('tranche_cf_existing_twh', 0)), 3),
            'uprate_twh': round(float(row.get('tranche_uprate_twh', 0)), 3),
            'geo_twh': round(float(row.get('tranche_geo_twh', 0)), 3),
            'nuclear_newbuild_twh': round(float(row.get('tranche_nuclear_newbuild_twh', 0)), 3),
            'ccs_tranche_twh': round(float(row.get('tranche_ccs_tranche_twh', 0)), 3),
            'new_cf_twh': round(float(row.get('tranche_new_cf_twh', 0)), 3),
        },
        'gas': {
            'gas_backup_needed_mw': int(row.get('gas_gas_backup_needed_mw', 0)),
            'existing_gas_used_mw': int(row.get('gas_existing_gas_used_mw', 0)),
            'new_gas_build_mw': int(row.get('gas_new_gas_build_mw', 0)),
            'gas_cost_per_mwh': round(float(row.get('gas_gas_cost_per_mwh', 0)), 2),
            'clean_peak_capacity_mw': int(row.get('gas_clean_peak_capacity_mw', 0)),
            'ra_peak_mw': int(row.get('gas_ra_peak_mw', 0)),
        },
    }


def format_threshold_key(threshold_float):
    """50.0 → '50', 87.5 → '87.5', 100.0 → '100'"""
    if threshold_float == int(threshold_float):
        return str(int(threshold_float))
    return str(threshold_float)


def compute_sweep_percentiles(thr_df):
    """Compute P10/P50/P90 effective cost from all scenarios at a threshold."""
    costs = thr_df['cost_effective_cost'].values
    if len(costs) < 2:
        return None
    return {
        'n': int(len(costs)),
        'p10': {'effective_cost': round(float(np.percentile(costs, 10)), 2)},
        'p50': {'effective_cost': round(float(np.percentile(costs, 50)), 2)},
        'p90': {'effective_cost': round(float(np.percentile(costs, 90)), 2)},
    }


def main():
    t0 = time.time()
    print("=" * 60)
    print("Export Track Results: parquet → track_results.json")
    print("=" * 60)

    if not os.path.exists(PQ_SCENARIOS):
        print(f"ERROR: {PQ_SCENARIOS} not found")
        sys.exit(1)

    df = pd.read_parquet(PQ_SCENARIOS)
    print(f"Loaded {len(df):,} scenario rows")

    isos = sorted(df['iso'].unique())
    tracks = sorted(df['track'].unique())
    thresholds = sorted(df['threshold'].unique())

    max_scenarios_per_group = df.groupby(['iso', 'track', 'threshold'])['scenario'].nunique().max()
    mode = 'full' if max_scenarios_per_group > 1 else 'medium'
    print(f"Mode: {mode} (max {max_scenarios_per_group} scenarios per group)")
    print(f"ISOs: {isos}")
    print(f"Tracks: {tracks}")
    print(f"Thresholds: {[format_threshold_key(t) for t in thresholds]}")

    results = {}

    for iso in isos:
        iso_df = df[df['iso'] == iso]
        results[iso] = {}

        for track in tracks:
            track_df = iso_df[iso_df['track'] == track]
            if track_df.empty:
                continue

            track_out = {}
            for thr in sorted(track_df['threshold'].unique()):
                thr_df = track_df[track_df['threshold'] == thr]
                thr_key = format_threshold_key(thr)
                n_scenarios = len(thr_df)

                # Always include medium scenario(s) with full detail
                medium_rows = thr_df[thr_df['scenario'].isin(MEDIUM_KEYS)]
                scenarios = {
                    row['scenario']: build_scenario_dict(row)
                    for row in medium_rows.to_dict('records')
                }

                # If only 1 scenario total and it wasn't in MEDIUM_KEYS, include it
                if not scenarios and n_scenarios > 0:
                    row = thr_df.iloc[0]
                    scenarios[row['scenario']] = build_scenario_dict(row)

                thr_out = {'scenarios': scenarios}

                # For full-sweep groups, precompute percentiles instead of
                # dumping all 5K+ scenarios
                if n_scenarios > 1:
                    sweep = compute_sweep_percentiles(thr_df)
                    if sweep:
                        thr_out['sweep'] = sweep

                track_out[thr_key] = thr_out

            results[iso][track] = track_out
            n_thr = len(track_out)
            n_sc = sum(len(v['scenarios']) for v in track_out.values())
            has_sweep = sum(1 for v in track_out.values() if 'sweep' in v)
            print(f"  {iso}/{track}: {n_thr} thresholds, {n_sc} medium scenarios, {has_sweep} with sweep percentiles")

    output = {
        'meta': {
            'mode': mode,
            'thresholds': [format_threshold_key(t) for t in thresholds],
            'isos': isos,
            'tracks': tracks,
        },
        'results': results,
    }

    print(f"\nWriting {OUT_PATH}...")
    with open(OUT_PATH, 'w') as f:
        json.dump(output, f, separators=(',', ':'))

    # Archive canonical copy to step5 results directory
    os.makedirs(STEP5_DIR, exist_ok=True)
    step5_out = os.path.join(STEP5_DIR, 'track_results.json')
    with open(step5_out, 'w') as f:
        json.dump(output, f, separators=(',', ':'))

    file_size = os.path.getsize(OUT_PATH) / 1024 / 1024
    elapsed = time.time() - t0
    print(f"Done: {file_size:.2f} MB, {elapsed:.1f}s")
    print(f"Archived: {step5_out}")


if __name__ == '__main__':
    main()
