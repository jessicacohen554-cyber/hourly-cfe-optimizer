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

# Try mac_queue subdirectory first (step5d_deployment_queue output), fall back to step5 root
_MAC_QUEUE_PATH = os.path.join(BASE_DIR, 'data', 'step5-post-processing', 'mac_queue', 'consequential_queue.json')
_LEGACY_PATH = os.path.join(BASE_DIR, 'data', 'step5-post-processing', 'consequential_queue.json')
QUEUE_JSON_PATH = _MAC_QUEUE_PATH if os.path.isfile(_MAC_QUEUE_PATH) else _LEGACY_PATH

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


def _reorder_queue_with_iso_monotonicity(queue):
    """Reorder queue: cheapest $/mtCO2 first, but within each ISO steps must
    follow physical buildout order (ascending threshold_start).

    Can't access 75→80% in MISO until 23→75% is built. Excludes entries
    with no delta_resources (missing data intervals).
    """
    from collections import defaultdict

    # Filter out entries with empty delta_resources (missing data)
    valid = [e for e in queue if e.get('delta_resources')]

    # Group by ISO, sort by threshold_start within each ISO
    iso_steps = defaultdict(list)
    for entry in valid:
        iso_steps[entry['iso']].append(entry)
    for iso in iso_steps:
        iso_steps[iso].sort(key=lambda e: e['threshold_start'])

    # Greedy: pick cheapest available entry where all ISO prerequisites are done
    iso_idx = {iso: 0 for iso in iso_steps}
    ordered = []
    total = sum(len(v) for v in iso_steps.values())

    while len(ordered) < total:
        best = None
        best_mac = float('inf')
        best_iso = None

        for iso, steps in iso_steps.items():
            idx = iso_idx[iso]
            if idx >= len(steps):
                continue
            entry = steps[idx]
            mac = entry.get('marginal_mac', float('inf'))
            if mac < best_mac:
                best_mac = mac
                best = entry
                best_iso = iso

        if best is None:
            break

        ordered.append(best)
        iso_idx[best_iso] += 1

    return ordered


def _load_queue():
    """Load the consequential deployment queue from JSON (cached).

    Applies within-ISO monotonic ordering: queue is sorted by cheapest
    $/mtCO2 globally, but within each ISO steps must follow physical
    buildout order (can't skip threshold intervals).
    """
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
    raw_queue = data.get('queue', data.get('deployment_queue', []))
    _QUEUE_CACHE = _reorder_queue_with_iso_monotonicity(raw_queue)
    print(f"  Loaded consequential queue: {len(raw_queue)} raw → "
          f"{len(_QUEUE_CACHE)} valid steps (within-ISO monotonic)")
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


def compute_all_isos_consequential(year, threshold, participation_pct,
                                    baseline_type='grid_average',
                                    growth_level='Medium', scenario='A',
                                    level='Medium', ppa_level='Medium',
                                    isos=None):
    """Compute consequential procurement for ALL buyer ISOs simultaneously.

    Shared queue depletion: all buyer ISOs walk the SAME queue. When a queue
    step's capacity is consumed by one buyer, it's unavailable to others.
    Allocation is pro-rata by demand share when multiple buyers need the same step.

    Returns dict {iso: strategy_result_dict}.
    """
    if isos is None:
        isos = ISOS

    # Compute buyer demands and CO2 targets for each ISO
    buyers = {}
    for iso in isos:
        demand = get_buyer_demand_twh(iso, year, participation_pct, growth_level)
        if demand <= 0 or threshold <= 0:
            buyers[iso] = {
                'demand': 0, 'co2_target': 0, 'co2_displaced': 0,
                'total_cost': 0, 'total_procured': 0, 'resource_mix': {},
                'emission_rate': 0, 'baseline_co2_mt': 0,
            }
            continue
        emission_rate = get_emission_rate(iso, baseline_type)
        baseline_co2_mt = demand * emission_rate
        target_fraction = min(threshold / 100.0, 1.0)
        co2_target = baseline_co2_mt * target_fraction
        buyers[iso] = {
            'demand': demand, 'co2_target': co2_target,
            'co2_displaced': 0.0, 'total_cost': 0.0, 'total_procured': 0.0,
            'resource_mix': {}, 'emission_rate': emission_rate,
            'baseline_co2_mt': baseline_co2_mt,
        }

    # Walk the shared queue with local-first allocation:
    # Phase 1: Each queue step's resources are consumed by LOCAL buyers first.
    # Phase 2: Remaining capacity goes cross-regionally, prioritized by
    #          highest local MAC (most expensive ISOs get dibs on cheap sources).
    queue = _load_queue()

    # Pre-compute local MAC for cross-regional priority ordering.
    # ISOs with highest local MAC get first dibs on cheap cross-regional sources.
    # Use the average marginal MAC across that ISO's queue entries as proxy.
    _iso_mac = {}
    for iso in isos:
        iso_macs = [e.get('marginal_mac', 0) for e in queue
                    if e.get('iso') == iso and e.get('marginal_mac', 0) > 0]
        _iso_mac[iso] = sum(iso_macs) / len(iso_macs) if iso_macs else 500.0

    def _allocate_to_buyer(b, src_iso, delta_res, co2_amount, fraction):
        """Allocate a fraction of a queue step to a buyer."""
        for resource, delta_twh in delta_res.items():
            if delta_twh < 0.5:
                continue
            ppa_resource = _RESOURCE_TO_PPA.get(resource)
            if ppa_resource is None:
                continue
            procure_twh = delta_twh * fraction
            ppa_price = get_learning_adjusted_ppa(
                ppa_resource, src_iso, threshold, scenario, level, ppa_level)
            b['total_cost'] += procure_twh * ppa_price
            b['total_procured'] += procure_twh
            key = f"{src_iso}_{ppa_resource}"
            b['resource_mix'][key] = b['resource_mix'].get(key, 0) + procure_twh
        b['co2_displaced'] += co2_amount

    for step in queue:
        # Find buyers that still need CO2 abatement
        active = {iso: b for iso, b in buyers.items()
                  if b['co2_target'] > 0 and b['co2_displaced'] < b['co2_target']}
        if not active:
            break

        src_iso = step['iso']
        delta_res = step.get('delta_resources', {})
        step_co2 = step.get('co2_displaced_mt', 0)
        if step_co2 <= 0:
            continue

        remaining_step_co2 = step_co2

        # --- PHASE 1: Local buyer gets first dibs ---
        if src_iso in active:
            b = active[src_iso]
            remaining_co2 = b['co2_target'] - b['co2_displaced']
            local_co2 = min(remaining_step_co2, remaining_co2)
            if local_co2 > 0 and remaining_step_co2 > 0:
                frac = local_co2 / step_co2
                _allocate_to_buyer(b, src_iso, delta_res, local_co2, frac)
                remaining_step_co2 -= local_co2

        # --- PHASE 2: Cross-regional allocation of remainder ---
        # Highest local MAC ISOs get priority (most expensive local options → most
        # incentive to seek cheap cross-regional sources).
        if remaining_step_co2 > 0.001:
            cross_buyers = [(iso, b) for iso, b in active.items()
                            if iso != src_iso and b['co2_displaced'] < b['co2_target']]
            # Sort by descending local MAC (most expensive gets first dibs)
            cross_buyers.sort(key=lambda x: _iso_mac.get(x[0], 0), reverse=True)

            for iso, b in cross_buyers:
                if remaining_step_co2 <= 0.001:
                    break
                remaining_co2 = b['co2_target'] - b['co2_displaced']
                alloc_co2 = min(remaining_step_co2, remaining_co2)
                if alloc_co2 > 0:
                    frac = alloc_co2 / step_co2
                    _allocate_to_buyer(b, src_iso, delta_res, alloc_co2, frac)
                    remaining_step_co2 -= alloc_co2

    # Fallback for buyers that didn't reach their target
    for iso, b in buyers.items():
        if b['co2_target'] <= 0:
            continue
        if b['co2_displaced'] < b['co2_target'] * 0.99 and b['total_procured'] > 0:
            fallback_sources = _find_sources_by_price(threshold, scenario, level, ppa_level)
            remaining_co2 = b['co2_target'] - b['co2_displaced']
            for src_iso, resource, price_mwh, available_twh in fallback_sources:
                if remaining_co2 <= 0:
                    break
                src_emission_rate = get_emission_rate(src_iso, 'fossil_average')
                co2_per_twh = src_emission_rate
                twh_needed = remaining_co2 / co2_per_twh if co2_per_twh > 0 else 0
                procure = min(twh_needed, available_twh)
                b['total_cost'] += procure * price_mwh
                b['total_procured'] += procure
                b['co2_displaced'] += procure * co2_per_twh
                remaining_co2 -= procure * co2_per_twh
                key = f"{src_iso}_{resource}"
                b['resource_mix'][key] = b['resource_mix'].get(key, 0) + procure

    # Build result dicts
    results = {}
    for iso, b in buyers.items():
        cost_per_mwh = b['total_cost'] / b['total_procured'] if b['total_procured'] > 0 else 0
        co2_abated = b['co2_displaced']
        mac_per_ton = (b['total_cost'] * 1e6) / (co2_abated * 1e6) if co2_abated > 0 else None
        co2_reduction_pct = (b['co2_displaced'] / b['baseline_co2_mt'] * 100) if b['baseline_co2_mt'] > 0 else 0

        total_demand = get_demand_twh_at_year(iso, year, growth_level)
        existing_clean = get_existing_clean_twh(iso)
        current_clean_pct = (existing_clean / total_demand * 100) if total_demand > 0 else 0
        lmp_impact = estimate_lmp_at_clean_pct(iso, current_clean_pct + threshold * 0.3)

        results[iso] = make_strategy_result(
            strategy_id=f'1{_baseline_suffix(baseline_type)}',
            iso=iso, year=year, threshold=threshold,
            participation_pct=participation_pct,
            cost_per_mwh=cost_per_mwh,
            total_cost_m=b['total_cost'],
            co2_abated_mmt=co2_abated,
            mac_per_ton=mac_per_ton,
            resource_mix=b['resource_mix'],
            metadata={
                'baseline_type': baseline_type,
                'emission_rate': b['emission_rate'],
                'baseline_co2_mt': round(b['baseline_co2_mt'], 4),
                'co2_reduction_pct': round(co2_reduction_pct, 2),
                'clean_procured_twh': round(b['total_procured'], 2),
                'scenario': scenario,
                'lmp_impact': round(lmp_impact, 2),
            },
        )

    return results


def compute_consequential_cost(buyer_iso, year, threshold, participation_pct,
                                baseline_type='grid_average',
                                growth_level='Medium', scenario='A',
                                level='Medium', ppa_level='Medium'):
    """Compute procurement cost for a consequential strategy variant.

    Uses shared queue depletion via compute_all_isos_consequential() —
    all buyer ISOs walk the SAME queue simultaneously, with pro-rata
    allocation by demand share. This prevents the same queue step from
    being consumed at full capacity by every buyer independently.

    Returns a strategy result dict.
    """
    results = compute_all_isos_consequential(
        year=year, threshold=threshold,
        participation_pct=participation_pct,
        baseline_type=baseline_type,
        growth_level=growth_level, scenario=scenario,
        level=level, ppa_level=ppa_level,
    )
    return results.get(buyer_iso, make_strategy_result(
        strategy_id=f'1{_baseline_suffix(baseline_type)}',
        iso=buyer_iso, year=year, threshold=threshold,
        participation_pct=participation_pct,
        cost_per_mwh=0, total_cost_m=0, co2_abated_mmt=0, mac_per_ton=0,
    ))


def _baseline_suffix(baseline_type):
    """Map baseline type to strategy variant suffix."""
    return {'grid_average': 'A', 'fossil_average': 'B', 'marginal': 'C'}.get(baseline_type, 'A')


# ═══════════════════════════════════════════════════════════════════════════════
# FULL SWEEP: ALL ISOs × PARTICIPATION × BASELINES
# ═══════════════════════════════════════════════════════════════════════════════


def run_full_sweep(isos=None, participation_levels=None, growth_levels=None):
    """Run Strategy 1 for all ISOs × participation levels × baseline types × growth levels.

    Uses shared queue depletion: all ISOs are computed simultaneously per
    (year, threshold, participation) combo via compute_all_isos_consequential().

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
    total = len(baselines) * len(growth_levels) * len(participation_levels)
    count = 0

    for baseline_type, strategy_id in baselines:
        strategy_key = f'strategy{strategy_id}'
        all_results[strategy_key] = {}
        for iso in isos:
            all_results[strategy_key][iso] = {}
            for growth in growth_levels:
                all_results[strategy_key][iso][growth] = {}

    for baseline_type, strategy_id in baselines:
        strategy_key = f'strategy{strategy_id}'

        for growth in growth_levels:
            for part in participation_levels:
                count += 1
                if count % 10 == 0:
                    print(f"  [{count}/{total}] {strategy_id} {growth} part={part*100:.0f}%")

                # Build trajectory for all ISOs simultaneously (shared queue)
                for year in TIMELINE_YEARS:
                    threshold = get_threshold_for_year(year)
                    batch = compute_all_isos_consequential(
                        year=year, threshold=threshold,
                        participation_pct=part,
                        baseline_type=baseline_type,
                        growth_level=growth, scenario='A',
                        isos=isos,
                    )
                    for iso in isos:
                        part_key = f'{part*100:.0f}pct'
                        if part_key not in all_results[strategy_key][iso][growth]:
                            all_results[strategy_key][iso][growth][part_key] = []
                        all_results[strategy_key][iso][growth][part_key].append(batch[iso])

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
