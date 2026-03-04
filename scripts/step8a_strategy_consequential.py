#!/usr/bin/env python3
"""
Step 6.5 — Strategy 1: Consequential Cross-Regional Netting
============================================================
Models three variants of consequential accounting:
  1A: Grid-average emission baseline
  1B: Fossil-average emission baseline
  1C: Marginal emission baseline

Buyers purchase cheapest $/tCO₂ clean energy anywhere in the US to "net"
against location-based carbon emissions. No ISO boundary constraint —
cross-regional procurement allowed.

Key characteristics:
- Chases cheapest $/tCO₂ → heavy VRE at low thresholds
- Cross-regional: can procure from cheapest ISO (SPP wind, ERCOT solar)
- No temporal constraint (annual/consequential, not hourly)
- Learning curve: Scenario A (delayed) — FOAK until 2035
- At high thresholds, faces FOAK firm costs (never invested in firm clean)

Output: data/step5-post-processing/strategy1_consequential.json
"""

import os
import sys
import json
import argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from procurement_utils import (
    ISOS, THRESHOLDS, TIMELINE_YEARS, BASE_YEAR,
    PARTICIPATION_LEVELS,
    CI_SHARE, BASE_DEMAND_TWH,
    GRID_MIX_SHARES,
    get_demand_twh_at_year, get_ci_demand_twh, get_buyer_demand_twh,
    get_threshold_for_year,
    get_lcoe, get_ppa_price, get_learning_adjusted_lcoe, get_learning_adjusted_ppa,
    get_wholesale_price, estimate_lmp_at_clean_pct,
    get_emission_rate, compute_co2_abated,
    get_existing_clean_twh, get_sss_twh,
    make_strategy_result, build_25yr_trajectory,
    save_results_json, save_js_data,
    UPRATE_CAP_TWH,
)

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD CONSEQUENTIAL QUEUE (SINGLE SOURCE OF TRUTH FOR DEPLOYMENT ORDERING)
# ═══════════════════════════════════════════════════════════════════════════════

_QUEUE_CACHE = None

QUEUE_JSON_PATH = os.path.join(BASE_DIR, 'data', 'step5-post-processing', 'consequential_queue.json')

# Map queue resource names → PPA resource names
_RESOURCE_TO_PPA = {
    'solar': 'solar',
    'wind': 'wind',
    'clean_firm': 'nuclear_newbuild',
    'ccs_ccgt': 'ccs_45q_on',
    'offshore_wind': 'wind',
    'hydro': None,         # Existing-only, no PPA
    'battery': None,       # Storage, no generation PPA
    'ldes': None,          # Storage, no generation PPA
}


def _load_queue():
    """Load the consequential deployment queue from JSON (cached)."""
    global _QUEUE_CACHE
    if _QUEUE_CACHE is not None:
        return _QUEUE_CACHE
    if not os.path.exists(QUEUE_JSON_PATH):
        print(f"  WARNING: Queue file not found at {QUEUE_JSON_PATH} — "
              f"falling back to price-based ordering")
        _QUEUE_CACHE = []
        return _QUEUE_CACHE
    with open(QUEUE_JSON_PATH) as f:
        data = json.load(f)
    _QUEUE_CACHE = data.get('queue', data.get('deployment_queue', []))
    print(f"  Loaded consequential queue: {len(_QUEUE_CACHE)} steps")
    return _QUEUE_CACHE

# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-REGIONAL CHEAPEST $/tCO₂ PROCUREMENT
# ═══════════════════════════════════════════════════════════════════════════════

# For consequential, buyer can procure from ANY ISO. Find cheapest sources.
# Source priority: cheapest new-build VRE > nuclear uprates > new-build firm


def find_cheapest_clean_sources(threshold, scenario='A', level='Medium', ppa_level='Medium'):
    """Find clean energy sources in deployment queue order (cheapest $/tCO2 first).

    Uses the consequential deployment queue as single source of truth for
    deployment ordering. Falls back to price-based ordering if queue unavailable.

    Returns list of (iso, resource, $/MWh_ppa, available_twh) in queue order.
    Cross-regional: no boundary constraint.
    """
    queue = _load_queue()

    if not queue:
        # Fallback: original price-based ordering when queue not available
        return _find_sources_by_price(threshold, scenario, level, ppa_level)

    # Walk queue steps in order (already sorted by marginal_mac).
    # For each step, extract the resource deltas and create source entries
    # with PPA-based pricing (buyer perspective) but queue-based ordering.
    sources = []
    for step in queue:
        iso = step['iso']
        delta_res = step.get('delta_resources', {})

        for resource, delta_twh in delta_res.items():
            if delta_twh < 0.5:
                continue
            ppa_resource = _RESOURCE_TO_PPA.get(resource)
            if ppa_resource is None:
                continue  # Storage, hydro — no generation PPA

            # PPA price for this resource in this ISO (buyer's cost)
            ppa_price = get_learning_adjusted_ppa(
                ppa_resource, iso, threshold, scenario, level, ppa_level)
            sources.append((iso, ppa_resource, ppa_price, delta_twh))

    # If queue didn't provide enough sources, supplement with price-based fallback
    if len(sources) < 5:
        sources.extend(_find_sources_by_price(threshold, scenario, level, ppa_level))

    return sources


def _find_sources_by_price(threshold, scenario='A', level='Medium', ppa_level='Medium'):
    """Fallback: find sources sorted by $/MWh PPA price (original approach)."""
    sources = []
    for iso in ISOS:
        solar_ppa = get_learning_adjusted_ppa('solar', iso, threshold, scenario, level, ppa_level)
        solar_cap = BASE_DEMAND_TWH[iso] * 0.30
        sources.append((iso, 'solar', solar_ppa, solar_cap))

        wind_ppa = get_learning_adjusted_ppa('wind', iso, threshold, scenario, level, ppa_level)
        wind_cap = BASE_DEMAND_TWH[iso] * 0.25
        sources.append((iso, 'wind', wind_ppa, wind_cap))

        uprate_twh = UPRATE_CAP_TWH.get(iso, 0)
        if uprate_twh > 0:
            uprate_ppa = get_learning_adjusted_ppa('uprate', iso, threshold, scenario, level, ppa_level)
            sources.append((iso, 'uprate', uprate_ppa, uprate_twh))

        firm_ppa = get_learning_adjusted_ppa('nuclear_newbuild', iso, threshold, scenario, level, ppa_level)
        firm_cap = BASE_DEMAND_TWH[iso] * 0.15
        sources.append((iso, 'nuclear_newbuild', firm_ppa, firm_cap))

        ccs_ppa = get_learning_adjusted_ppa('ccs_45q_on', iso, threshold, scenario, level, ppa_level)
        ccs_cap = BASE_DEMAND_TWH[iso] * 0.10
        sources.append((iso, 'ccs', ccs_ppa, ccs_cap))
    sources.sort(key=lambda s: s[2])
    return sources


def compute_consequential_cost(buyer_iso, year, threshold, participation_pct,
                                baseline_type='grid_average',
                                growth_level='Medium', scenario='A',
                                level='Medium', ppa_level='Medium'):
    """Compute procurement cost for a consequential strategy variant.

    The buyer needs to NET their emissions by purchasing clean energy credits.
    Total clean procurement needed = buyer_demand_twh × emission_rate × (threshold/100)
    ... expressed as: buyer must cover (threshold%) of their load with clean energy,
    where each MWh of clean energy "abates" emission_rate tCO₂.

    For consequential, the accounting is: buyer procures cheapest $/tCO₂ from
    any ISO to offset their emissions up to the CFE target.

    Returns a strategy result dict.
    """
    buyer_demand = get_buyer_demand_twh(buyer_iso, year, participation_pct, growth_level)
    if buyer_demand <= 0 or threshold <= 0:
        return make_strategy_result(
            strategy_id=f'1{_baseline_suffix(baseline_type)}',
            iso=buyer_iso, year=year, threshold=threshold,
            participation_pct=participation_pct,
            cost_per_mwh=0, total_cost_m=0, co2_abated_mmt=0, mac_per_ton=0,
        )

    # Clean energy needed (TWh)
    target_fraction = min(threshold / 100.0, 1.0)
    clean_needed_twh = buyer_demand * target_fraction

    # Emission rate for this ISO/baseline type
    emission_rate = get_emission_rate(buyer_iso, baseline_type)

    # Find cheapest cross-regional sources
    sources = find_cheapest_clean_sources(threshold, scenario, level, ppa_level)

    # Walk up the source stack until clean_needed is satisfied
    total_cost = 0.0  # $M
    total_procured = 0.0  # TWh
    resource_mix = {}

    for src_iso, resource, price_mwh, available_twh in sources:
        if total_procured >= clean_needed_twh:
            break

        remaining = clean_needed_twh - total_procured
        procure = min(remaining, available_twh)

        total_cost += procure * price_mwh * 1e6 / 1e6  # TWh × $/MWh = $M
        total_procured += procure

        key = f"{src_iso}_{resource}"
        resource_mix[key] = resource_mix.get(key, 0) + round(procure, 2)

    # Cost metrics
    cost_per_mwh = total_cost / total_procured if total_procured > 0 else 0
    # CO2 abated: use queue dispatch-based emission rate if available
    co2_abated = total_procured * emission_rate / 1e3  # TWh × tCO₂/MWh / 1000 = MtCO₂
    # MAC = total PPA cost ($M) / CO2 abated (Mt) → $/tCO₂
    mac_per_ton = (total_cost * 1e6) / (co2_abated * 1e6) if co2_abated > 0 else None  # $/tCO₂

    # Wholesale price impact (for the buyer's ISO)
    total_demand = get_demand_twh_at_year(buyer_iso, year, growth_level)
    existing_clean = get_existing_clean_twh(buyer_iso)
    current_clean_pct = (existing_clean / total_demand * 100) if total_demand > 0 else 0
    lmp_impact = estimate_lmp_at_clean_pct(buyer_iso, current_clean_pct + threshold * 0.3)

    return make_strategy_result(
        strategy_id=f'1{_baseline_suffix(baseline_type)}',
        iso=buyer_iso,
        year=year,
        threshold=threshold,
        participation_pct=participation_pct,
        cost_per_mwh=cost_per_mwh,
        total_cost_m=total_cost,
        co2_abated_mmt=co2_abated,
        mac_per_ton=mac_per_ton,
        resource_mix=resource_mix,
        metadata={
            'baseline_type': baseline_type,
            'emission_rate': emission_rate,
            'clean_procured_twh': round(total_procured, 2),
            'clean_needed_twh': round(clean_needed_twh, 2),
            'scenario': scenario,
            'lmp_impact': round(lmp_impact, 2),
        },
    )


def _baseline_suffix(baseline_type):
    """Map baseline type to strategy variant suffix."""
    return {'grid_average': 'A', 'fossil_average': 'B', 'marginal': 'C'}.get(baseline_type, 'A')


# ═══════════════════════════════════════════════════════════════════════════════
# 25-YEAR TRAJECTORY FOR STRATEGY 1
# ═══════════════════════════════════════════════════════════════════════════════


def compute_strategy1_trajectory(iso, participation_pct=0.10,
                                  baseline_type='grid_average',
                                  growth_level='Medium', scenario='A',
                                  level='Medium', ppa_level='Medium'):
    """Compute 25-year cost trajectory for Strategy 1 (consequential).

    Returns list of annual results from 2025→2050.
    """
    def strategy_fn(iso, year, threshold, participation_pct, **kwargs):
        return compute_consequential_cost(
            buyer_iso=iso, year=year, threshold=threshold,
            participation_pct=participation_pct,
            baseline_type=baseline_type,
            growth_level=kwargs.get('growth_level', 'Medium'),
            scenario=scenario,
            level=kwargs.get('level', 'Medium'),
            ppa_level=kwargs.get('ppa_level', 'Medium'),
        )

    return build_25yr_trajectory(
        iso=iso,
        strategy_fn=strategy_fn,
        participation_pct=participation_pct,
        growth_level=growth_level,
        scenario=scenario,
        level=level,
        ppa_level=ppa_level,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FULL SWEEP: ALL ISOs × PARTICIPATION × BASELINES
# ═══════════════════════════════════════════════════════════════════════════════


def run_full_sweep(isos=None, participation_levels=None, growth_levels=None):
    """Run Strategy 1 for all ISOs × participation levels × baseline types × growth levels.

    Returns dict with structure:
    {
        'strategy1A': {iso: {growth: {part: [trajectory]}}},
        'strategy1B': {iso: {growth: {part: [trajectory]}}},
        'strategy1C': {iso: {growth: {part: [trajectory]}}},
    }
    """
    if isos is None:
        isos = ISOS
    if participation_levels is None:
        participation_levels = PARTICIPATION_LEVELS
    if growth_levels is None:
        growth_levels = ['Low', 'Medium', 'High']

    baselines = [
        ('grid_average', '1A'),
        ('fossil_average', '1B'),
        ('marginal', '1C'),
    ]

    all_results = {}
    total = len(baselines) * len(isos) * len(growth_levels) * len(participation_levels)
    count = 0

    for baseline_type, strategy_id in baselines:
        strategy_key = f'strategy{strategy_id}'
        all_results[strategy_key] = {}

        for iso in isos:
            all_results[strategy_key][iso] = {}

            for growth in growth_levels:
                all_results[strategy_key][iso][growth] = {}

                for part in participation_levels:
                    count += 1
                    if count % 50 == 0:
                        print(f"  [{count}/{total}] {strategy_id} {iso} {growth} part={part*100:.0f}%")

                    trajectory = compute_strategy1_trajectory(
                        iso=iso,
                        participation_pct=part,
                        baseline_type=baseline_type,
                        growth_level=growth,
                        scenario='A',  # Consequential always uses Scenario A (delayed learning)
                    )

                    part_key = f'{part*100:.0f}pct'
                    all_results[strategy_key][iso][growth][part_key] = trajectory

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description='Step 6.5 — Strategy 1: Consequential Cross-Regional Netting'
    )
    parser.add_argument('--iso', type=str, default='ALL',
                        help='ISO to compute (default ALL)')
    parser.add_argument('--participation', type=float, nargs='+', default=None,
                        help='Participation levels (e.g., 0.05 0.10 0.20)')
    parser.add_argument('--growth', type=str, nargs='+', default=['Low', 'Medium', 'High'],
                        help='Demand growth levels')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: Medium growth only, fewer participation levels')
    args = parser.parse_args()

    isos = ISOS if args.iso == 'ALL' else [args.iso]

    if args.quick:
        participation = [0.05, 0.10, 0.20, 0.40]
        growth_levels = ['Medium']
    else:
        participation = args.participation or PARTICIPATION_LEVELS
        growth_levels = args.growth

    print("=" * 70)
    print("Step 6.5 — Strategy 1: Consequential Cross-Regional Netting")
    print("=" * 70)
    print(f"ISOs: {isos}")
    print(f"Participation levels: {[f'{p*100:.0f}%' for p in participation]}")
    print(f"Growth levels: {growth_levels}")
    print(f"Variants: 1A (grid-avg), 1B (fossil-avg), 1C (marginal)")
    print(f"Scenario: A (delayed learning curve)")
    print()

    results = run_full_sweep(
        isos=isos,
        participation_levels=participation,
        growth_levels=growth_levels,
    )

    # Save results
    save_results_json(results, 'strategy1_consequential.json')

    # Print summary for Medium growth, 10% participation
    print("\n" + "=" * 70)
    print("SUMMARY: Strategy 1 at 10% participation, Medium growth")
    print("=" * 70)
    for variant in ['strategy1A', 'strategy1B', 'strategy1C']:
        if variant not in results:
            continue
        print(f"\n--- {variant.upper()} ---")
        for iso in isos:
            data = results[variant].get(iso, {}).get('Medium', {}).get('10pct', [])
            if not data:
                continue
            # Show key milestone years
            milestones = {r['year']: r for r in data if r['year'] in [2030, 2035, 2040, 2050]}
            for yr, r in sorted(milestones.items()):
                print(f"  {iso} {yr}: ${r['cost_per_mwh']:.1f}/MWh, "
                      f"{r['co2_abated_mmt']:.3f} MtCO₂, "
                      f"threshold={r['threshold']:.0f}%")

    print("\nDone.")


if __name__ == '__main__':
    main()
