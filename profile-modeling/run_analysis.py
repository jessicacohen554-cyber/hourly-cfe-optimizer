#!/usr/bin/env python3
"""
Main entry point for profile-modeling cost optimization.

Usage:
    # Wind + solar only, ERCOT, all cost levels
    python run_analysis.py --iso ERCOT --resources wind solar

    # Wind + solar + 4hr battery
    python run_analysis.py --iso ERCOT --resources wind solar battery4

    # Use a preset portfolio
    python run_analysis.py --iso ERCOT --resources wind_solar_firm

    # Compare two portfolios
    python run_analysis.py --iso ERCOT --compare wind_solar wind_solar_bat4

    # Custom targets
    python run_analysis.py --iso ERCOT --resources wind solar battery4 --targets 0.90 0.95 0.99

Presets:
    wind_solar            wind, solar
    wind_solar_bat4       wind, solar, battery4
    wind_solar_bat4_bat8  wind, solar, battery4, battery8
    wind_solar_storage    wind, solar, battery4, ldes
    wind_solar_firm       wind, solar, battery4, clean_firm
    wind_solar_ccs        wind, solar, battery4, ccs_ccgt
    full_re_storage       wind, solar, battery4, battery8, ldes
    full_portfolio        wind, solar, clean_firm, battery4, ldes
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ISOS, THRESHOLDS, COST_LEVELS, WHOLESALE_PRICES, H
from load_profiles import get_load_profile
from supply import load_supply_shapes
from dispatch import warmup
from optimizer import optimize_sweep, resolve_resources, PRESETS


def print_results_table(results, iso, wholesale, label=''):
    """Print a formatted results table."""
    if not results:
        return

    cost_level = results[0]['cost_level']
    res_list = results[0].get('resource_list', [])
    res_label = ', '.join(res_list)

    print(f"\n{'='*100}")
    if label:
        print(f"  {label}")
    print(f"  {iso} — {cost_level} Cost — Resources: [{res_label}]")
    print(f"  Wholesale: ${wholesale}/MWh")
    print(f"{'='*100}")

    # Determine columns from resources
    gen_cols = results[0].get('gen_resources', [])
    stor_cols = results[0].get('stor_resources', [])

    # Header
    hdr = f"  {'Target':>7}  {'Score':>7}  {'Cost':>8}  {'Prem':>7}  {'GenTot':>6}"
    for g in gen_cols:
        hdr += f"  {g[:7]:>7}"
    for s in stor_cols:
        hdr += f"  {s[:7]:>7}"
    print(hdr)

    sep = f"  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*6}"
    for _ in gen_cols + stor_cols:
        sep += f"  {'-'*7}"
    print(sep)

    for r in results:
        t = f"{r['target']*100:.1f}%"
        sc = f"{r['score']*100:.1f}%"
        co = f"${r['total_cost']:.1f}"
        pr = f"+${r['total_cost'] - wholesale:.0f}"
        gt = f"{r['total_gen_pct']:.0f}%"

        row = f"  {t:>7}  {sc:>7}  {co:>8}  {pr:>7}  {gt:>6}"
        for g in gen_cols:
            res = r['resources'].get(g, {})
            pct = res.get('alloc_pct', 0)
            row += f"  {pct:6.1f}%" if pct > 0.05 else f"  {'—':>7}"
        for s in stor_cols:
            res = r['resources'].get(s, {})
            pct = res.get('cap_pct', 0)
            row += f"  {pct:5.3f}%" if pct > 0.001 else f"  {'—':>7}"
        print(row)


def print_comparison(all_results_by_portfolio, iso, wholesale):
    """Print side-by-side comparison of portfolios at each target."""
    if not all_results_by_portfolio:
        return

    # Collect all targets across portfolios
    all_targets = sorted(set(
        r['target'] for results in all_results_by_portfolio.values()
        for r in results
    ))

    print(f"\n{'='*80}")
    print(f"  COMPARISON — {iso} (wholesale: ${wholesale}/MWh)")
    print(f"{'='*80}")

    # Header with portfolio names
    hdr = f"  {'Target':>7}"
    for name in all_results_by_portfolio:
        hdr += f"  {name:>20}"
    print(hdr)
    print(f"  {'-'*7}" + f"  {'-'*20}" * len(all_results_by_portfolio))

    for target in all_targets:
        row = f"  {target*100:.1f}%"
        for name, results in all_results_by_portfolio.items():
            match = [r for r in results if abs(r['target'] - target) < 0.001]
            if match:
                r = match[0]
                if r['score'] >= target - 0.001:
                    row += f"  ${r['total_cost']:>17.1f}/MWh"
                else:
                    row += f"  {'INFEASIBLE':>20}"
            else:
                row += f"  {'—':>20}"
        print(row)


def save_results(all_results, output_dir, iso, profile_name, resource_tag):
    """Save results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{iso}_{profile_name}_{resource_tag}.json"
    filepath = os.path.join(output_dir, filename)

    serializable = []
    for r in all_results:
        entry = {k: v for k, v in r.items()
                 if k not in ('gen_resources', 'stor_resources')}
        serializable.append(entry)

    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)

    print(f"\n  Saved: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(
        description='Profile modeling — cost-optimal clean energy mix',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Presets: " + ", ".join(PRESETS.keys()))

    parser.add_argument('--iso', default='ERCOT')
    parser.add_argument('--resources', nargs='+', default=['wind', 'solar', 'battery4'],
                        help='Resources to include (names or preset)')
    parser.add_argument('--compare', nargs='+', default=None,
                        help='Compare multiple presets (e.g., --compare wind_solar wind_solar_bat4)')
    parser.add_argument('--cost-levels', nargs='+', default=['Medium'],
                        choices=COST_LEVELS)
    parser.add_argument('--profile', default='flat')
    parser.add_argument('--targets', nargs='+', type=float, default=None)
    parser.add_argument('--maxiter', type=int, default=300)
    parser.add_argument('--popsize', type=int, default=30)
    parser.add_argument('--output-dir', default=None)

    args = parser.parse_args()

    isos = ISOS if args.iso.lower() == 'all' else [args.iso.upper()]
    targets = args.targets if args.targets else THRESHOLDS
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'results')

    # Determine portfolios to run
    if args.compare:
        portfolios = {name: resolve_resources(name) for name in args.compare}
    else:
        # Single resource list (could be a preset name)
        if len(args.resources) == 1 and args.resources[0] in PRESETS:
            res = resolve_resources(args.resources[0])
            portfolios = {args.resources[0]: res}
        else:
            res = args.resources
            tag = '+'.join(res)
            portfolios = {tag: res}

    print("Profile Modeling — Cost-Optimal Clean Energy Mix")
    print(f"  ISOs: {isos}")
    print(f"  Cost levels: {args.cost_levels}")
    print(f"  Load profile: {args.profile}")
    print(f"  Targets: {[f'{t*100:.1f}%' for t in targets]}")
    for name, res in portfolios.items():
        print(f"  Portfolio '{name}': {res}")

    print("\nWarming up Numba JIT...")
    warmup()
    print("JIT ready.")

    for iso in isos:
        print(f"\n{'#'*80}")
        print(f"  ISO: {iso}")
        print(f"{'#'*80}")

        supply_shapes = load_supply_shapes(iso)
        demand = get_load_profile(args.profile, iso=iso)
        wholesale = WHOLESALE_PRICES.get(iso, 30)

        comparison_results = {}

        for pname, resources in portfolios.items():
            for cost_level in args.cost_levels:
                label = f"{pname} — {cost_level}"
                print(f"\n  ── {label} ──")

                results = optimize_sweep(
                    iso, cost_level, targets, resources, supply_shapes, demand,
                    maxiter=args.maxiter, popsize=args.popsize, verbose=True,
                )

                print_results_table(results, iso, wholesale, label)

                resource_tag = pname.replace('+', '_')
                save_results(results, output_dir, iso, args.profile,
                             f"{resource_tag}_{cost_level}")

                # For comparison (Medium only)
                if cost_level == 'Medium':
                    comparison_results[pname] = results

        if len(portfolios) > 1:
            print_comparison(comparison_results, iso, wholesale)

    print("\nDone.")


if __name__ == '__main__':
    main()
