#!/usr/bin/env python3
"""Run the full parametric sweep and save as parquet only (no large JSON).

Default: 3 demand × 5 price × 3 PPA × 3 gas friction × 3 queue × 3 new-build
fossil cost = 1,215 scenarios. Each scenario is simulated across all ISOs/years.

The parameter space can be customized via CSV templates:
  --params-csv <path>      Override sweep axes (demand growth rates, gas friction,
                           PPA, queue, new fossil cost levels)
  --price-sens-csv <path>  Override price sensitivity bundles (9-dim toggle vectors)
  --export-defaults <dir>  Export current defaults as editable CSV templates

Usage:
    cd market-simulator/scripts
    python run_sweep_1215.py [--isos CAISO PJM] [--output-dir ../results/sweep_1215]
    python run_sweep_1215.py --export-defaults ../configs/
    python run_sweep_1215.py --params-csv ../configs/sweep_params_default.csv \\
                             --price-sens-csv ../configs/sweep_price_sensitivities_default.csv

Produces:
    sweep_1215/sweep_1215_flat.parquet           — flat table for analysis (~10 MB)
    sweep_1215/sweep_1215_aggregates.json        — P10/P50/P90 percentile bands (~1 MB)
"""

import argparse
import json
import os
import sys
import time

import math

import numpy as np
import pandas as pd

# Ensure scripts/ is on path
sys.path.insert(0, os.path.dirname(__file__))

from market_simulation import (
    build_market_scenarios,
    build_sim_years,
    run_full_sweep,
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


def _sanitize(v):
    """Replace inf/-inf with None so pyarrow can serialize to parquet."""
    if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
        return None
    return v


def flatten_year_result(scenario_id, yr):
    """Convert a single year_result dict into a flat row dict."""
    row = {'scenario_id': scenario_id}

    # Scalar fields
    for f in SCALAR_FIELDS:
        row[f] = _sanitize(yr.get(f))

    # resource_mix_twh → prefixed columns
    rmix = yr.get('resource_mix_twh', {}) or {}
    for res, twh in rmix.items():
        row[f'mix_{res}_twh'] = _sanitize(twh)

    # generator_economics → prefixed columns (capacity_mw, cf, margin_mwh per fuel)
    gen_econ = yr.get('generator_economics', {}) or {}
    for fuel, metrics in gen_econ.items():
        if isinstance(metrics, dict):
            row[f'ge_{fuel}_capacity_mw'] = _sanitize(metrics.get('capacity_mw'))
            row[f'ge_{fuel}_cf'] = _sanitize(metrics.get('cf'))
            row[f'ge_{fuel}_margin_mwh'] = _sanitize(metrics.get('margin_mwh'))

    # economic_retirements_mw → prefixed columns
    econ_ret = yr.get('economic_retirements_mw', {}) or {}
    for fuel, mw in econ_ret.items():
        row[f'ret_{fuel}_mw'] = _sanitize(mw)

    # new_fossil_builds_mw → prefixed columns
    nfb = yr.get('new_fossil_builds_mw', {}) or {}
    for fuel, mw in nfb.items():
        row[f'nb_{fuel}_mw'] = _sanitize(mw)

    # emissions_by_fuel → prefixed columns
    ebf = yr.get('emissions_by_fuel', {}) or {}
    for fuel, mt in ebf.items():
        row[f'emis_{fuel}_mt'] = _sanitize(mt)

    # nuclear_revenue → prefixed columns
    nuc_rev = yr.get('nuclear_revenue', {}) or {}
    for k, v in nuc_rev.items():
        if isinstance(v, (int, float)):
            row[f'nuc_{k}'] = _sanitize(v)

    # ccs_breakeven → prefixed columns
    ccs_be = yr.get('ccs_breakeven', {}) or {}
    for k, v in ccs_be.items():
        if isinstance(v, (int, float)):
            row[f'ccs_be_{k}'] = _sanitize(v)

    return row


def results_to_dataframe(all_results):
    """Convert full sweep results dict to a flat DataFrame."""
    rows = []
    for scenario_id, iso_results in all_results.items():
        if scenario_id.startswith('_'):
            continue  # skip _aggregates
        for iso, year_results in iso_results.items():
            if iso.startswith('_'):
                continue  # skip per-scenario _provenance metadata
            for yr in year_results:
                rows.append(flatten_year_result(scenario_id, yr))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='Run parametric market sweep → parquet')
    parser.add_argument('--isos', nargs='+', default=None,
                        help='ISOs to simulate (default: all 7)')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: ../results/sweep_1215)')
    parser.add_argument('--nuclear-retirement', type=float, default=None,
                        help='Nuclear retirement threshold $/MWh')
    parser.add_argument('--start-year', type=int, default=2023,
                        help='First simulation year (default: 2023, includes eGRID baseline)')
    parser.add_argument('--end-year', type=int, default=2050,
                        help='Last simulation year (default: 2050)')
    parser.add_argument('--year-step', type=int, default=1,
                        help='Year step size (default: 1 = annual)')
    parser.add_argument('--params-csv', default=None,
                        help='CSV file defining sweep parameter axes '
                             '(demand growth, gas friction, PPA, queue, '
                             'new fossil cost). Overrides hardcoded defaults.')
    parser.add_argument('--price-sens-csv', default=None,
                        help='CSV file defining price sensitivity bundles '
                             '(9-dim toggle vectors). Overrides hardcoded defaults.')
    parser.add_argument('--export-defaults', metavar='DIR', default=None,
                        help='Export current hardcoded defaults as CSV templates '
                             'to the specified directory, then exit.')
    parser.add_argument('--shard', type=str, default=None,
                        help='Scenario shard: "K/N" runs shard K of N '
                             '(1-indexed). E.g., --shard 1/3 runs first third.')
    args = parser.parse_args()

    # Handle --export-defaults
    if args.export_defaults:
        from sweep_params_io import export_default_params
        export_default_params(args.export_defaults)
        return

    sim_years = build_sim_years(args.start_year, args.end_year, args.year_step)

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), '..', 'results', 'sweep_1215')
    os.makedirs(output_dir, exist_ok=True)

    # Build scenarios — from CSV overrides or hardcoded defaults
    scenarios = build_market_scenarios(
        params_csv=args.params_csv,
        price_sens_csv=args.price_sens_csv,
    )
    n_isos_planned = len(args.isos or ['CAISO','ERCOT','PJM','NYISO','NEISO','MISO','SPP'])
    print(f"Sweep configuration: {len(scenarios)} scenarios × "
          f"{len(sim_years)} years × {n_isos_planned} ISOs")
    if args.params_csv:
        print(f"  Parameters CSV: {os.path.abspath(args.params_csv)}")
    if args.price_sens_csv:
        print(f"  Price sensitivities CSV: {os.path.abspath(args.price_sens_csv)}")
    print(f"Years: {sim_years[0]}–{sim_years[-1]} (step={args.year_step}, n={len(sim_years)})")
    print(f"Output: {os.path.abspath(output_dir)}")

    # Parse shard argument
    shard = None
    if args.shard:
        parts = args.shard.split('/')
        shard = (int(parts[0]), int(parts[1]))

    # ── Run the sweep ──
    t0 = time.time()
    all_results = run_full_sweep(
        isos=args.isos,
        nuclear_retirement_threshold=args.nuclear_retirement,
        sim_years=sim_years,
        params_csv=args.params_csv,
        price_sens_csv=args.price_sens_csv,
        shard=shard,
    )
    elapsed = time.time() - t0
    print(f"\nSweep completed in {elapsed:.1f}s")

    # ── Save flat parquet (primary output) ──
    print("Converting to flat parquet...")
    df = results_to_dataframe(all_results)
    # Replace any remaining inf/-inf with NaN (pyarrow cannot serialize inf)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Write per-ISO parquets so each ISO can be run independently
    # without overwriting results from other ISOs
    isos_in_df = sorted(df['iso'].dropna().unique())
    for iso_name in isos_in_df:
        iso_df = df[df['iso'] == iso_name]
        iso_parquet_path = os.path.join(output_dir, f'sweep_1215_{iso_name}.parquet')
        iso_df.to_parquet(iso_parquet_path, index=False, engine='pyarrow')
        print(f"  Per-ISO parquet: {iso_parquet_path} ({len(iso_df)} rows)")

    # Also write combined parquet for backward compatibility
    parquet_path = os.path.join(output_dir, 'sweep_1215_flat.parquet')
    df.to_parquet(parquet_path, index=False, engine='pyarrow')
    print(f"Combined parquet saved: {parquet_path}")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # Expected: 1,215 scenarios × N ISOs × N years
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

    # ── Save aggregates (small JSON, ~1 MB) ──
    print("Computing aggregates...")
    aggregates = aggregate_sweep_percentiles(all_results)
    agg_path = os.path.join(output_dir, 'sweep_1215_aggregates.json')
    with open(agg_path, 'w') as f:
        json.dump(aggregates, f, indent=2, default=str)
    print(f"Aggregates saved: {agg_path}")

    print(f"\nDone. Total time: {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
