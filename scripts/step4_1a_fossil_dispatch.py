#!/usr/bin/env python3
"""
Step 4.1A: Merged Fossil Dispatch — CO₂ + LMP from Single Merit-Order Computation
==================================================================================
Replaces separate CO₂ and LMP computation scripts.

Both CO₂ and LMP require the same underlying computation:
  1. Read clean dispatch from Step 3A dispatch cache (8760h profiles)
  2. Compute residual_demand[h] = max(0, demand[h] - total_clean[h])
  3. Fill residual with merit-order fossil stack (coal → oil → gas)
  4. CO₂ = sum of fossil_unit_dispatch[h] × emission_rate[unit]
  5. LMP = marginal cost of last fossil unit dispatched at each hour

Running them separately either duplicates work or risks divergence (different
retirement thresholds, different fossil fleet assumptions). This module
guarantees consistency.

Pipeline position:
  Step 3A (Dispatch Cache) → THIS
                                ├→ CO₂ results
                                └→ LMP results

Input:  data/step3-dispatch/{ISO}_dispatch_cache.parquet
        data/step2.2-cost/ (scenario metadata)
Output: data/step4-analysis/co2_results/
        data/step4-analysis/lmp/

Usage:
  python step4_1a_fossil_dispatch.py                    # All ISOs, Medium fuel
  python step4_1a_fossil_dispatch.py --iso PJM          # Single ISO
  python step4_1a_fossil_dispatch.py --fuel-level High   # Specific fuel level
"""

import argparse
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'scripts'))

from pipeline_config import (
    OUTPUT_THRESHOLDS as ALL_THRESHOLDS,
    CAPACITY_MARKET_PRICES, CAPACITY_DEGRADATION_ALPHA,
    PEAK_CAPACITY_CREDITS, RESOURCE_CAPACITY_FACTORS,
)

from dispatch_utils import (
    H, ISOS, RESOURCE_TYPES, CCS_RESIDUAL_EMISSION_RATE,
    GRID_MIX_SHARES, BASE_DEMAND_TWH,
    load_common_data, get_supply_profiles, get_demand_profile,
    build_supply_matrix,
    _archetype_key, load_dispatch_cache, DISPATCH_CACHE_DIR,
    compute_fossil_retirement,
)

from parquet_io import (
    find_input_dir, find_parquet, ALL_ISOS,
)

# Import LMP engine
from lmp_engine import (
    build_merit_order_stack,
    compute_hourly_lmp_vectorized,
    compute_lmp_stats,
    get_price_model,
    compute_marginal_costs,
)

DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
STEP5_DIR = os.path.join(DATA_DIR, 'step4-analysis')
CO2_DIR = os.path.join(STEP5_DIR, 'co2_results')
LMP_DIR = os.path.join(STEP5_DIR, 'lmp')


def compute_capacity_market_revenue(iso, threshold, resource_mix):
    """Compute capacity market revenue by resource category.

    Uses same formula as Step 10: (degraded_price × ELCC) / (CF × 8760).
    Degraded price falls as clean share (threshold) increases.

    Returns dict with per-category and system-level capacity revenue ($/MWh).
    """
    base_price = CAPACITY_MARKET_PRICES.get(iso, 0)
    alpha = CAPACITY_DEGRADATION_ALPHA.get(iso, 0)
    degraded_price = base_price * max(0, 1.0 - alpha * threshold / 100.0)

    if degraded_price <= 0:
        return {
            'capacity_rev_system_mwh': 0.0,
            'capacity_rev_clean_firm_mwh': 0.0,
            'capacity_rev_vre_mwh': 0.0,
            'capacity_rev_storage_mwh': 0.0,
            'capacity_rev_ccs_mwh': 0.0,
        }

    # Per-resource capacity revenue ($/MWh of that resource's generation)
    per_res = {}
    for res in ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt',
                'hydro', 'battery', 'battery8', 'ldes', 'h2', 'geothermal']:
        elcc = PEAK_CAPACITY_CREDITS.get(res, 0)
        cf = RESOURCE_CAPACITY_FACTORS.get(res, {}).get(iso, 0.30)
        if cf > 0 and elcc > 0:
            per_res[res] = degraded_price * elcc / (cf * 8.760)  # $/kW-yr → $/MWh
        else:
            per_res[res] = 0.0

    # Category aggregation (weighted by resource share within category)
    clean_firm_res = ['clean_firm', 'geothermal']
    vre_res = ['solar', 'wind', 'offshore_wind']
    storage_res = ['battery', 'battery8', 'ldes', 'h2']
    ccs_res = ['ccs_ccgt']

    def weighted_cat_rev(cat_resources):
        """Weighted average capacity revenue for resources in a category."""
        total_share = 0
        weighted_rev = 0
        for r in cat_resources:
            share = resource_mix.get(r, resource_mix.get('mix_' + r, 0))
            if share > 0:
                weighted_rev += per_res.get(r, 0) * share
                total_share += share
        return weighted_rev / total_share if total_share > 0 else per_res.get(cat_resources[0], 0)

    # System-wide: weighted across ALL resources (clean + fossil get capacity payments)
    # For fossil, use hydro CF as proxy for dispatchable capacity value
    all_shares = sum(resource_mix.get(k, 0) for k in resource_mix
                     if k.startswith('mix_') or k in per_res)
    system_rev = 0
    if all_shares > 0:
        for res, rev in per_res.items():
            share = resource_mix.get(res, resource_mix.get('mix_' + res, 0))
            system_rev += rev * share
        system_rev /= max(all_shares, 1e-9)

    return {
        'capacity_rev_system_mwh': round(system_rev, 2),
        'capacity_rev_clean_firm_mwh': round(weighted_cat_rev(clean_firm_res), 2),
        'capacity_rev_vre_mwh': round(weighted_cat_rev(vre_res), 2),
        'capacity_rev_storage_mwh': round(weighted_cat_rev(storage_res), 2),
        'capacity_rev_ccs_mwh': round(weighted_cat_rev(ccs_res), 2),
    }


def compute_hourly_co2(dispatch_result, iso, threshold, emission_rates, fossil_mix,
                        total_mwh):
    """Compute CO₂ abated from hourly dispatch profiles.

    Uses dispatch-stack emission rate (varies by threshold via merit-order
    retirement). CCS gets special treatment: its dispatch displaces fossil
    but has residual emissions.

    Args:
        dispatch_result: dict with 'fossil_displaced', 'ccs_supply', etc.
        iso: ISO region
        threshold: clean energy threshold (determines emission rate)
        emission_rates: regional emission rates
        fossil_mix: regional fossil fuel mix
        total_mwh: total annual demand in MWh

    Returns:
        dict with CO₂ abatement metrics
    """
    rate, info = compute_fossil_retirement(iso, threshold, emission_rates, fossil_mix)

    fossil_displaced = dispatch_result['fossil_displaced']
    ccs_supply = dispatch_result.get('ccs_supply', np.zeros(H))

    # Split fossil displacement into CCS and non-CCS components
    non_ccs_displaced = np.maximum(0.0, fossil_displaced - np.minimum(fossil_displaced, ccs_supply))
    ccs_displaced = np.minimum(fossil_displaced, ccs_supply)

    # CO₂ calculation
    co2_clean = float(np.sum(non_ccs_displaced)) * rate * total_mwh
    ccs_credit = max(0.0, rate - CCS_RESIDUAL_EMISSION_RATE)
    co2_ccs = float(np.sum(ccs_displaced)) * ccs_credit * total_mwh

    total_abated = co2_clean + co2_ccs
    matched_mwh = float(np.sum(fossil_displaced)) * total_mwh
    co2_rate = total_abated / matched_mwh if matched_mwh > 0 else 0

    return {
        'total_co2_abated_tons': round(total_abated, 0),
        'co2_rate_per_mwh': round(co2_rate, 4),
        'matched_mwh': round(matched_mwh, 0),
        'emission_rate_tco2_mwh': round(rate, 4),
        'methodology': 'hourly_fossil_dispatch',
        'retirement_info': info,
    }


def run_fossil_dispatch_for_iso(iso, demand_data, gen_profiles, emission_rates,
                                 fossil_mix, fuel_level='Medium',
                                 dispatch_cache=None, thresholds=None):
    """Compute both CO₂ and LMP for all cached archetypes of an ISO.

    Single-pass computation: for each archetype, builds the merit-order fossil
    stack, fills residual demand, and produces both CO₂ and LMP outputs.

    Args:
        iso: ISO region
        demand_data: demand profiles
        gen_profiles: generation profiles
        emission_rates: regional emission rates
        fossil_mix: regional fossil fuel mix
        fuel_level: 'Low', 'Medium', 'High'
        dispatch_cache: pre-loaded dispatch cache dict
        thresholds: optional list of thresholds to process

    Returns:
        co2_results: list of CO₂ result dicts
        lmp_results: list of LMP result dicts
    """
    if dispatch_cache is None:
        dispatch_cache = load_dispatch_cache(iso)
    if not dispatch_cache:
        print(f"    WARNING: No dispatch cache for {iso}. Run Step 4 first.")
        return [], []

    demand_norm, total_mwh = get_demand_profile(iso, demand_data)
    demand_mw_profile = demand_norm * total_mwh
    price_model = get_price_model(iso, fuel_level)

    # Load scenarios from step3 parquets for metadata
    import pandas as pd
    input_dir = find_input_dir([iso])
    co_path = find_parquet(input_dir, iso) if input_dir else None

    scenarios = []
    if co_path:
        df = pd.read_parquet(co_path)
        if thresholds:
            df = df[df['threshold'].isin(thresholds)]
        # Group by unique mix to avoid redundant computation
        mix_cols = ['mix_clean_firm', 'mix_solar', 'mix_wind', 'mix_offshore_wind',
                    'mix_ccs_ccgt', 'mix_hydro', 'battery_dispatch_pct',
                    'battery8_dispatch_pct', 'ldes_dispatch_pct']
        h2_col = 'h2_dispatch_pct' if 'h2_dispatch_pct' in df.columns else None

        for _, row in df.drop_duplicates(subset=mix_cols + ['threshold', 'scenario']).iterrows():
            rp = {
                'clean_firm': float(row['mix_clean_firm']),
                'solar': float(row['mix_solar']),
                'wind': float(row['mix_wind']),
                'offshore_wind': float(row.get('mix_offshore_wind', 0)),
                'ccs_ccgt': float(row['mix_ccs_ccgt']),
                'hydro': float(row['mix_hydro']),
            }
            scenarios.append({
                'resource_mix': rp,
                'threshold': float(row['threshold']),
                'scenario': row.get('scenario', ''),
                'battery_dispatch_pct': float(row['battery_dispatch_pct']),
                'battery8_dispatch_pct': float(row['battery8_dispatch_pct']),
                'ldes_dispatch_pct': float(row['ldes_dispatch_pct']),
                'h2_dispatch_pct': float(row[h2_col]) if h2_col else 0,
            })

    co2_results = []
    lmp_results = []
    seen_archetypes = {}

    for sc in scenarios:
        rp = sc['resource_mix']
        threshold = sc['threshold']
        bat = sc['battery_dispatch_pct']
        bat8 = sc['battery8_dispatch_pct']
        ldes = sc['ldes_dispatch_pct']

        # Look up dispatch cache
        key = _archetype_key(iso, rp, 100, bat, bat8, ldes)
        entry = dispatch_cache.get(key)
        if entry is None:
            continue

        # Archetype dedup key includes threshold and fuel (affects fossil stack)
        dedup_key = f"{key}_{threshold}_{fuel_level}"
        if dedup_key in seen_archetypes:
            co2, lmp_stats = seen_archetypes[dedup_key]
        else:
            # CO₂ from hourly dispatch
            co2 = compute_hourly_co2(
                entry, iso, threshold, emission_rates, fossil_mix, total_mwh)

            # LMP from merit-order fossil stack
            stack, total_fossil_mw = build_merit_order_stack(
                iso, threshold, fuel_level,
                resource_mix=rp,
                battery_pct=bat, battery8_pct=bat8, ldes_pct=ldes,
                h2_pct=sc.get('h2_dispatch_pct', 0))

            hourly_lmp, hourly_mu = compute_hourly_lmp_vectorized(
                entry, demand_mw_profile, stack, price_model, iso,
                vre_penetration=threshold / 100.0 if threshold else None)

            lmp_stats = compute_lmp_stats(hourly_lmp, hourly_mu, demand_mw_profile, entry)

            # Capacity market revenue by resource category
            cap_rev = compute_capacity_market_revenue(iso, threshold, rp)
            lmp_stats.update(cap_rev)
            lmp_stats['total_market_rev_mwh'] = round(
                lmp_stats['avg_lmp'] + cap_rev['capacity_rev_system_mwh'], 2)
            # Per-category total (energy + capacity)
            lmp_stats['total_rev_clean_firm_mwh'] = round(
                lmp_stats['avg_lmp'] + cap_rev['capacity_rev_clean_firm_mwh'], 2)
            lmp_stats['total_rev_vre_mwh'] = round(
                lmp_stats['avg_lmp'] + cap_rev['capacity_rev_vre_mwh'], 2)
            lmp_stats['total_rev_storage_mwh'] = round(
                lmp_stats['avg_lmp'] + cap_rev['capacity_rev_storage_mwh'], 2)
            lmp_stats['total_rev_ccs_mwh'] = round(
                lmp_stats['avg_lmp'] + cap_rev['capacity_rev_ccs_mwh'], 2)

            seen_archetypes[dedup_key] = (co2, lmp_stats)

        co2_results.append({
            'iso': iso,
            'threshold': threshold,
            'scenario': sc['scenario'],
            'archetype_key': key,
            **co2,
        })

        lmp_results.append({
            'iso': iso,
            'threshold': threshold,
            'scenario': sc['scenario'],
            'archetype_key': key,
            'fuel_level': fuel_level,
            **lmp_stats,
        })

    print(f"    {iso}: {len(co2_results)} scenarios, "
          f"{len(seen_archetypes)} unique archetypes")
    return co2_results, lmp_results


def save_results(iso, co2_results, lmp_results):
    """Save CO₂ and LMP results as parquets."""
    import pandas as pd

    os.makedirs(CO2_DIR, exist_ok=True)
    os.makedirs(LMP_DIR, exist_ok=True)

    if co2_results:
        df = pd.DataFrame(co2_results)
        # Drop non-serializable retirement_info
        if 'retirement_info' in df.columns:
            df = df.drop(columns=['retirement_info'])
        path = os.path.join(CO2_DIR, f'co2_{iso}.parquet')
        df.to_parquet(path, index=False, compression='zstd')
        print(f"    → {os.path.basename(path)}: {len(df)} rows")

    if lmp_results:
        df = pd.DataFrame(lmp_results)
        path = os.path.join(LMP_DIR, f'{iso}_lmp.parquet')
        df.to_parquet(path, index=False, compression='zstd')
        print(f"    → {os.path.basename(path)}: {len(df)} rows")


def main():
    parser = argparse.ArgumentParser(
        description='Merged CO₂ + LMP computation from single fossil dispatch.',
    )
    parser.add_argument('--iso', dest='isos', action='append', choices=ALL_ISOS,
                        metavar='ISO',
                        help='ISO to process (repeatable). Default: all.')
    parser.add_argument('--fuel-level', type=str, default='Medium',
                        choices=['Low', 'Medium', 'High'],
                        help='Fossil fuel price level. Default: Medium.')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Single threshold to process. Default: all.')
    args = parser.parse_args()

    print("=" * 70)
    print("  Step 4.1A: Merged Fossil Dispatch — CO₂ + LMP")
    print("  Single-pass merit-order computation for both outputs")
    print("=" * 70)

    t0 = time.time()

    print("\n  Loading common data...")
    demand_data, gen_profiles, emission_rates, fossil_mix = load_common_data()

    run_isos = args.isos or ALL_ISOS
    thresholds = [args.threshold] if args.threshold else None

    total_co2 = 0
    total_lmp = 0

    for iso in run_isos:
        print(f"\n  {iso}:")

        # Load dispatch cache
        dispatch_cache = load_dispatch_cache(iso)
        if not dispatch_cache:
            print(f"    No dispatch cache. Skipping.")
            continue

        co2_results, lmp_results = run_fossil_dispatch_for_iso(
            iso, demand_data, gen_profiles, emission_rates, fossil_mix,
            fuel_level=args.fuel_level,
            dispatch_cache=dispatch_cache,
            thresholds=thresholds,
        )

        save_results(iso, co2_results, lmp_results)
        total_co2 += len(co2_results)
        total_lmp += len(lmp_results)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  Step 4.1A Complete")
    print(f"  CO₂: {total_co2} scenarios, LMP: {total_lmp} scenarios")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
