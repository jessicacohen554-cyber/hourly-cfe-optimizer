#!/usr/bin/env python3
"""
LMP Model Calibration — Validate synthetic LMP against PJM actual data
========================================================================
Compares compute_lmp_prices.py output against published PJM market statistics
from the Independent Market Monitor (Monitoring Analytics) State of the Market
reports and EIA wholesale data.

Calibration approach:
  1. Run synthetic LMP for PJM at 2025 baseline (~40.6% clean), Medium fuel
  2. Compare key metrics against published actuals
  3. Report delta and suggest parameter adjustments
  4. Optionally load raw hourly PJM LMP data (if available) for distribution matching

Reference data sources:
  - Monitoring Analytics 2024 SOM: https://monitoringanalytics.com/reports/PJM_State_of_the_Market/2024/
  - EIA Wholesale Markets: https://www.eia.gov/electricity/wholesalemarkets/data.php?rto=pjm
  - PJM Data Miner 2: https://dataminer2.pjm.com/feed/da_hrl_lmps

Usage:
  python calibrate_lmp_model.py                  # Run calibration with embedded targets
  python calibrate_lmp_model.py --hourly FILE    # Also compare against hourly CSV
"""

import json
import os
import sys
import time
import argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from dispatch_utils import (
    H, GRID_MIX_SHARES, BASE_DEMAND_TWH,
    load_common_data, get_demand_profile, get_supply_profiles,
    reconstruct_hourly_dispatch,
)
from compute_lmp_prices import (
    build_merit_order_stack, get_price_model,
    compute_hourly_lmp_vectorized, compute_lmp_stats,
    load_scenarios,
)

LMP_DIR = os.path.join(SCRIPT_DIR, 'data', 'lmp')

# ══════════════════════════════════════════════════════════════════════════════
# PUBLISHED PJM REFERENCE DATA (Monitoring Analytics SOM / EIA)
# ══════════════════════════════════════════════════════════════════════════════

# System-wide RT load-weighted avg LMP ($/MWh) — IMM State of the Market
PJM_ACTUAL_LMP = {
    2021: {'rt_lw_avg': 39.78, 'total_wholesale_cost': None},
    2022: {'rt_lw_avg': 80.14, 'total_wholesale_cost': None},
    2023: {'rt_lw_avg': 31.08, 'total_wholesale_cost': 53.08},
    2024: {'rt_lw_avg': 33.74, 'total_wholesale_cost': 55.54},
}

# DA LMP is typically $1-3 higher than RT in PJM (DA risk premium)
DA_RT_SPREAD = 2.0  # $/MWh average DA premium over RT

# Western Hub vs system-wide spread (Western Hub slightly lower than LW avg)
WESTERN_HUB_SYSTEM_SPREAD = -1.0  # $/MWh

# Estimated DA Western Hub average for 2024
PJM_DA_WESTERN_HUB_2024_EST = (
    PJM_ACTUAL_LMP[2024]['rt_lw_avg'] + DA_RT_SPREAD + WESTERN_HUB_SYSTEM_SPREAD
)  # ~$34.74/MWh

# PJM 2024 market structure stats (from IMM SOM)
PJM_2024_STATS = {
    'avg_hourly_load_mw': 94_787,
    'peak_load_mw': 154_045,
    'installed_capacity_mw': 179_656,
    'negative_rt_price_intervals': None,  # not publicly broken out in search results
    # Typical PJM: 100-300 negative price hours/year (DA), most at night
    'negative_da_price_hours_est': 200,  # order-of-magnitude estimate
    # Scarcity (> $200/MWh): 50-150 hours/year in normal years
    'scarcity_hours_est': 100,
}

# PJM price distribution targets (estimated from published percentiles)
# Based on typical PJM DA LMP distribution at Western Hub
PJM_2024_DISTRIBUTION = {
    'avg': 34.7,       # DA WH estimated
    'peak_avg': 42.0,  # peak hours typically 20-25% above avg
    'offpeak_avg': 28.0,  # off-peak typically 15-20% below avg
    'p10': 18.0,       # low-load shoulder hours
    'p25': 23.0,       # baseload gas CCGT marginal
    'p50': 30.0,       # median — gas CCGT marginal
    'p75': 42.0,       # moderate load — gas CT marginal
    'p90': 55.0,       # high load — gas CT + congestion
    'volatility': 25.0,  # std dev — moderate in 2024
}


# ══════════════════════════════════════════════════════════════════════════════
# CALIBRATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def run_synthetic_lmp_for_calibration(iso='PJM', fuel_level='Medium'):
    """Run synthetic LMP at current grid mix and return stats for comparison.

    For calibration, we use the ACTUAL 2024 grid mix (GRID_MIX_SHARES) with
    100% procurement — this represents the real grid, not the optimizer's
    optimal hourly-matching mix. The optimizer output is used for forward
    projections, not baseline calibration.
    """
    demand_data, gen_profiles, emission_rates, fossil_mix = load_common_data()
    demand_norm, total_mwh = get_demand_profile(iso, demand_data)
    supply_profiles = get_supply_profiles(iso, gen_profiles)
    demand_mw_profile = demand_norm * total_mwh

    baseline_clean = sum(GRID_MIX_SHARES.get(iso, {}).values())
    print(f"  {iso} baseline clean: {baseline_clean:.1f}%")

    # For calibration: use actual grid mix at 100% procurement
    # This represents PJM's 2024 operating reality, not an optimizer scenario
    resource_mix = GRID_MIX_SHARES.get(iso, {})
    procurement_pct = 100  # actual grid delivers its full mix

    print(f"  Calibration mix (actual grid): {resource_mix}")
    print(f"  Procurement: {procurement_pct}%")
    closest_threshold = baseline_clean

    # Dispatch — actual grid with no storage (2024 PJM has minimal grid storage)
    dispatch = reconstruct_hourly_dispatch(
        demand_norm, supply_profiles, resource_mix,
        procurement_pct,
        0,  # battery
        0,  # battery8
        0)  # ldes

    # Merit-order stack (RA + GAF aware)
    stack, fossil_mw = build_merit_order_stack(
        iso, closest_threshold, fuel_level,
        resource_mix=resource_mix, procurement_pct=procurement_pct)
    print(f"  Fossil stack: {fossil_mw:,.0f} MW")

    # Price model
    price_model = get_price_model(iso, fuel_level)

    # LMP
    hourly_lmp, hourly_mu = compute_hourly_lmp_vectorized(
        dispatch, demand_mw_profile, stack, price_model, iso)

    stats = compute_lmp_stats(hourly_lmp, hourly_mu, demand_mw_profile, dispatch)

    return {
        'stats': stats,
        'hourly_lmp': hourly_lmp,
        'hourly_mu': hourly_mu,
        'demand_mw': demand_mw_profile,
        'residual_mw': dispatch['residual_demand'] * total_mwh,
        'resource_mix': resource_mix,
        'threshold': closest_threshold,
        'stack': stack,
        'fossil_mw': fossil_mw,
    }


def compare_metrics(synthetic, targets, label=''):
    """Compare synthetic LMP stats against calibration targets."""
    print(f"\n  {'='*60}")
    print(f"  {label or 'CALIBRATION COMPARISON'}")
    print(f"  {'='*60}")
    print(f"  {'Metric':<30} {'Synthetic':>12} {'Target':>12} {'Delta':>12} {'Status':>10}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

    results = []
    for key, target in targets.items():
        if target is None:
            continue
        synth_val = synthetic.get(key)
        if synth_val is None:
            continue
        delta = synth_val - target
        pct = (delta / target * 100) if target != 0 else float('inf')

        # Status thresholds
        if abs(pct) < 10:
            status = 'GOOD'
        elif abs(pct) < 25:
            status = 'FAIR'
        elif abs(pct) < 50:
            status = 'ADJUST'
        else:
            status = 'FAIL'

        print(f"  {key:<30} {synth_val:>12.2f} {target:>12.2f} "
              f"{delta:>+12.2f} {status:>10}")
        results.append({'metric': key, 'synthetic': synth_val, 'target': target,
                        'delta': delta, 'pct_delta': pct, 'status': status})

    return results


def suggest_adjustments(comparison_results, synthetic_result):
    """Analyze calibration gaps and suggest parameter adjustments."""
    print(f"\n  {'='*60}")
    print(f"  SUGGESTED ADJUSTMENTS")
    print(f"  {'='*60}")

    adjustments = []

    for r in comparison_results:
        if r['status'] in ('GOOD', 'FAIR'):
            continue

        metric = r['metric']
        delta = r['delta']
        pct = r['pct_delta']

        if metric == 'avg' or metric == 'avg_lmp':
            if delta > 0:
                adj = (f"  avg LMP too HIGH by {abs(pct):.0f}%: "
                       f"Consider reducing scarcity_threshold or scarcity_cap")
            else:
                adj = (f"  avg LMP too LOW by {abs(pct):.0f}%: "
                       f"Consider increasing fuel prices or tightening stack capacity")
            adjustments.append(adj)

        elif 'volatility' in metric:
            if delta > 0:
                adj = f"  Volatility too HIGH: Reduce scarcity_cap or surplus_decay"
            else:
                adj = f"  Volatility too LOW: Increase scarcity sensitivity"
            adjustments.append(adj)

        elif 'negative' in metric:
            if delta > 0:
                adj = f"  Too many negative hours: Reduce surplus_decay parameter"
            else:
                adj = f"  Too few negative hours: Increase surplus_decay or lower floor"
            adjustments.append(adj)

        elif 'scarcity' in metric:
            if delta > 0:
                adj = f"  Too many scarcity hours: Increase stack capacity or lower scarcity_threshold"
            else:
                adj = f"  Too few scarcity hours: Reduce stack capacity or raise scarcity_threshold"
            adjustments.append(adj)

        elif 'p10' in metric or 'p25' in metric:
            if delta < -5:
                adj = f"  Low percentiles too LOW: Too many surplus hours driving floor prices"
            elif delta > 5:
                adj = f"  Low percentiles too HIGH: Price floor needs to be lower or surplus pricing steeper"
            else:
                continue
            adjustments.append(adj)

        elif 'p90' in metric:
            if delta > 20:
                adj = f"  P90 too HIGH: Scarcity pricing too aggressive"
            elif delta < -10:
                adj = f"  P90 too LOW: Stack may be oversized"
            adjustments.append(adj)

    if not adjustments:
        print("  No major adjustments needed — model within tolerance.")
    else:
        for adj in adjustments:
            print(adj)

    return adjustments


def analyze_hourly_distribution(synthetic_lmp, actual_lmp=None):
    """Analyze hourly LMP distribution shape."""
    print(f"\n  {'='*60}")
    print(f"  PRICE DISTRIBUTION ANALYSIS")
    print(f"  {'='*60}")

    # Synthetic distribution
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    synth_pcts = np.percentile(synthetic_lmp, percentiles)

    print(f"\n  Synthetic LMP Distribution:")
    for p, v in zip(percentiles, synth_pcts):
        print(f"    P{p:02d}: ${v:>8.2f}/MWh")

    # Price buckets
    buckets = [
        ('< $0', synthetic_lmp < 0),
        ('$0-10', (synthetic_lmp >= 0) & (synthetic_lmp < 10)),
        ('$10-25', (synthetic_lmp >= 10) & (synthetic_lmp < 25)),
        ('$25-50', (synthetic_lmp >= 25) & (synthetic_lmp < 50)),
        ('$50-100', (synthetic_lmp >= 50) & (synthetic_lmp < 100)),
        ('$100-200', (synthetic_lmp >= 100) & (synthetic_lmp < 200)),
        ('$200-500', (synthetic_lmp >= 200) & (synthetic_lmp < 500)),
        ('$500+', synthetic_lmp >= 500),
    ]

    print(f"\n  Price Bucket Distribution:")
    print(f"  {'Bucket':<15} {'Hours':>8} {'Pct':>8}")
    print(f"  {'-'*15} {'-'*8} {'-'*8}")
    for label, mask in buckets:
        hours = int(mask.sum())
        pct = hours / H * 100
        bar = '█' * int(pct / 2)
        print(f"  {label:<15} {hours:>8} {pct:>7.1f}% {bar}")

    # Diurnal pattern
    print(f"\n  Diurnal Pattern (avg by hour of day):")
    hourly_avg = np.zeros(24)
    for hod in range(24):
        hourly_avg[hod] = np.mean(synthetic_lmp[hod::24])
    peak_hour = np.argmax(hourly_avg)
    valley_hour = np.argmin(hourly_avg)
    print(f"    Peak hour: {peak_hour:02d}:00 (${hourly_avg[peak_hour]:.2f}/MWh)")
    print(f"    Valley hour: {valley_hour:02d}:00 (${hourly_avg[valley_hour]:.2f}/MWh)")
    print(f"    Peak/valley ratio: {hourly_avg[peak_hour]/max(0.01, hourly_avg[valley_hour]):.2f}x")

    if actual_lmp is not None:
        print(f"\n  Actual vs Synthetic comparison available — {len(actual_lmp)} hours")
        # TODO: implement when raw hourly data is available


def load_actual_hourly_csv(filepath):
    """Load PJM hourly LMP CSV (from Data Miner 2 export).

    Expected columns: datetime_beginning_ept, total_lmp_da
    """
    import pandas as pd
    df = pd.read_csv(filepath)
    # Try various column name formats
    lmp_col = None
    for col in ['total_lmp_da', 'Total LMP', 'LMP', 'total_lmp', 'Price']:
        if col in df.columns:
            lmp_col = col
            break
    if lmp_col is None:
        print(f"  Available columns: {df.columns.tolist()}")
        raise ValueError("Cannot find LMP column in CSV")

    return df[lmp_col].values


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Calibrate LMP model against PJM actuals')
    parser.add_argument('--hourly', type=str, default=None,
                        help='Path to hourly PJM DA LMP CSV (from Data Miner 2)')
    parser.add_argument('--iso', type=str, default='PJM')
    args = parser.parse_args()

    print("=" * 70)
    print("  LMP MODEL CALIBRATION")
    print("=" * 70)

    # Run synthetic LMP
    print(f"\n  Running synthetic LMP for {args.iso}...")
    result = run_synthetic_lmp_for_calibration(args.iso)
    if result is None:
        return

    stats = result['stats']

    # Build comparison targets
    targets = {
        'avg_lmp': PJM_2024_DISTRIBUTION['avg'],
        'peak_avg_lmp': PJM_2024_DISTRIBUTION['peak_avg'],
        'offpeak_avg_lmp': PJM_2024_DISTRIBUTION['offpeak_avg'],
        'lmp_p10': PJM_2024_DISTRIBUTION['p10'],
        'lmp_p25': PJM_2024_DISTRIBUTION['p25'],
        'lmp_p50': PJM_2024_DISTRIBUTION['p50'],
        'lmp_p75': PJM_2024_DISTRIBUTION['p75'],
        'lmp_p90': PJM_2024_DISTRIBUTION['p90'],
        'price_volatility': PJM_2024_DISTRIBUTION['volatility'],
        'negative_price_hours': PJM_2024_STATS['negative_da_price_hours_est'],
        'scarcity_hours': PJM_2024_STATS['scarcity_hours_est'],
    }

    # Compare
    comparison = compare_metrics(stats, targets, f'{args.iso} — 2024 Baseline Calibration')

    # Suggest adjustments
    suggest_adjustments(comparison, result)

    # Distribution analysis
    analyze_hourly_distribution(result['hourly_lmp'])

    # Load actual data if provided
    if args.hourly:
        print(f"\n  Loading actual hourly data: {args.hourly}")
        actual_lmp = load_actual_hourly_csv(args.hourly)
        analyze_hourly_distribution(result['hourly_lmp'], actual_lmp)

    # Save calibration report
    os.makedirs(LMP_DIR, exist_ok=True)
    report = {
        'iso': args.iso,
        'calibration_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'synthetic_stats': stats,
        'targets': targets,
        'comparison': [{k: v for k, v in r.items()} for r in comparison],
        'distribution': {
            'percentiles': {
                f'p{p}': round(float(v), 2)
                for p, v in zip([1, 5, 10, 25, 50, 75, 90, 95, 99],
                                np.percentile(result['hourly_lmp'],
                                              [1, 5, 10, 25, 50, 75, 90, 95, 99]))
            },
            'negative_hours': int((result['hourly_lmp'] < 0).sum()),
            'zero_hours': int((result['hourly_lmp'] <= 0).sum()),
            'scarcity_hours': int((result['hourly_lmp'] > 200).sum()),
        },
    }
    report_path = os.path.join(LMP_DIR, f'{args.iso}_calibration.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  Calibration report saved: {report_path}")

    print(f"\n{'='*70}")
    print(f"  CALIBRATION COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
