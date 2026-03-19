#!/usr/bin/env python3
"""Run the full parametric sweep and save as JSON + parquet.

The sweep covers 3 demand × 5 price × 3 PPA × 3 gas friction × 3 queue
× 3 new-build fossil cost = 1,215 scenarios. Each scenario is simulated
across all ISOs and years.

Usage:
    cd market-simulator/scripts
    python run_sweep_405.py [--isos CAISO PJM] [--output-dir ../results/sweep_1215]

Produces:
    sweep_1215/market_simulation_results.json   — full nested JSON (all fields)
    sweep_1215/sweep_1215_flat.parquet           — flat table for analysis
    sweep_1215/sweep_1215_aggregates.json        — P10/P50/P90 percentile bands
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

# Ensure scripts/ is on path
sys.path.insert(0, os.path.dirname(__file__))

from market_simulation import (
    build_market_scenarios,
    run_full_sweep,
    save_results,
    aggregate_sweep_percentiles,
    SIM_YEARS,
)

# Fields to extract into the flat parquet table
SCALAR_FIELDS = [
    'iso', 'scenario', 'year',
    'clean_pct', 'demand_twh', 'emissions_mt', 'emission_rate_tco2_mwh',
    'cost_per_mwh', 'revenue_per_mwh',
    'energy_rev_mwh', 'capacity_rev_mwh', 'rec_rev_mwh',
    'avg_lmp', 'lmp_p90',
    'gas_built_gw', 'fossil_built_gw', 'total_gas_gw',
    'total_new_fossil_mw',
    'market_stop', 'nuclear_retired',
    'total_economic_retirement_mw',
    'rps_mandated_pct', 'rps_eligible_pct', 'rps_shortfall_pct',
    'acp_cost_million', 'cumulative_acp_million',
    'ordc_scarcity_hours', 'data_source',
]


def flatten_year_result(scenario_id, yr):
    """Convert a single year_result dict into a flat row dict."""
    row = {'scenario_id': scenario_id}

    # Scalar fields
    for f in SCALAR_FIELDS:
        row[f] = yr.get(f)

    # resource_mix_twh → prefixed columns
    rmix = yr.get('resource_mix_twh', {}) or {}
    for res, twh in rmix.items():
        row[f'mix_{res}_twh'] = twh

    # generator_economics → prefixed columns (capacity_mw, cf, margin_mwh per fuel)
    gen_econ = yr.get('generator_economics', {}) or {}
    for fuel, metrics in gen_econ.items():
        if isinstance(metrics, dict):
            row[f'ge_{fuel}_capacity_mw'] = metrics.get('capacity_mw')
            row[f'ge_{fuel}_cf'] = metrics.get('cf')
            row[f'ge_{fuel}_margin_mwh'] = metrics.get('margin_mwh')

    # economic_retirements_mw → prefixed columns
    econ_ret = yr.get('economic_retirements_mw', {}) or {}
    for fuel, mw in econ_ret.items():
        row[f'ret_{fuel}_mw'] = mw

    # new_fossil_builds_mw → prefixed columns
    nfb = yr.get('new_fossil_builds_mw', {}) or {}
    for fuel, mw in nfb.items():
        row[f'nb_{fuel}_mw'] = mw

    # emissions_by_fuel → prefixed columns
    ebf = yr.get('emissions_by_fuel', {}) or {}
    for fuel, mt in ebf.items():
        row[f'emis_{fuel}_mt'] = mt

    # nuclear_revenue → prefixed columns
    nuc_rev = yr.get('nuclear_revenue', {}) or {}
    for k, v in nuc_rev.items():
        if isinstance(v, (int, float)):
            row[f'nuc_{k}'] = v

    # ccs_breakeven → prefixed columns
    ccs_be = yr.get('ccs_breakeven', {}) or {}
    for k, v in ccs_be.items():
        if isinstance(v, (int, float)):
            row[f'ccs_be_{k}'] = v

    return row


def results_to_dataframe(all_results):
    """Convert full sweep results dict to a flat DataFrame."""
    rows = []
    for scenario_id, iso_results in all_results.items():
        if scenario_id.startswith('_'):
            continue  # skip _aggregates
        for iso, year_results in iso_results.items():
            for yr in year_results:
                rows.append(flatten_year_result(scenario_id, yr))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='Run 1,215-scenario sweep → JSON + parquet')
    parser.add_argument('--isos', nargs='+', default=None,
                        help='ISOs to simulate (default: all 7)')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: ../results/sweep_1215)')
    parser.add_argument('--nuclear-retirement', type=float, default=None,
                        help='Nuclear retirement threshold $/MWh')
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), '..', 'results', 'sweep_1215')
    os.makedirs(output_dir, exist_ok=True)

    # Verify scenario count
    scenarios = build_market_scenarios()
    print(f"Sweep configuration: {len(scenarios)} scenarios × "
          f"{len(SIM_YEARS)} years × {len(args.isos or ['CAISO','ERCOT','PJM','NYISO','NEISO','MISO','SPP'])} ISOs")
    print(f"Years: {SIM_YEARS}")
    print(f"Output: {os.path.abspath(output_dir)}")

    # ── Run the sweep ──
    t0 = time.time()
    all_results = run_full_sweep(
        isos=args.isos,
        nuclear_retirement_threshold=args.nuclear_retirement,
    )
    elapsed = time.time() - t0
    print(f"\nSweep completed in {elapsed:.1f}s")

    # ── Save JSON (full nested structure) ──
    json_path = os.path.join(output_dir, 'market_simulation_results.json')
    save_results(all_results, output_dir)
    print(f"JSON saved: {json_path}")

    # ── Save flat parquet ──
    print("Converting to flat parquet...")
    df = results_to_dataframe(all_results)
    parquet_path = os.path.join(output_dir, 'sweep_1215_flat.parquet')
    df.to_parquet(parquet_path, index=False, engine='pyarrow')
    print(f"Parquet saved: {parquet_path}")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # Expected: 1,215 scenarios × 7 ISOs × 6 years = 51,030
    n_scenarios = len([k for k in all_results if not k.startswith('_')])
    n_isos = len(set(df['iso'].dropna()))
    n_years = len(set(df['year'].dropna()))
    print(f"  Scenarios: {n_scenarios}, ISOs: {n_isos}, Years: {n_years}")
    expected = n_scenarios * n_isos * n_years
    actual = len(df)
    if actual == expected:
        print(f"  ✓ Row count matches expected: {actual}")
    else:
        print(f"  ⚠ Row count {actual} ≠ expected {expected}")

    # ── Save aggregates separately ──
    print("Computing aggregates...")
    aggregates = aggregate_sweep_percentiles(all_results)
    agg_path = os.path.join(output_dir, 'sweep_1215_aggregates.json')
    with open(agg_path, 'w') as f:
        json.dump(aggregates, f, indent=2, default=str)
    print(f"Aggregates saved: {agg_path}")

    print(f"\nDone. Total time: {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
