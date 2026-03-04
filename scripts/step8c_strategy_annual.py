#!/usr/bin/env python3
"""
Step 6.5 — Strategy 3: Annual Matching
========================================
Models four variants of annual volumetric matching:
  3A: Same-ISO, new build required
  3B: Cross-regional, new build required
  3C: Same-ISO, no additionality (existing clean counts)
  3D: Cross-regional, no additionality (cheapest — status quo)

Key characteristics:
- Annual volumetric: match total MWh over the year, no hourly constraint
- No temporal premium (no over-procurement for hourly matching)
- 3A/3C: Same-ISO boundary; 3B/3D: cross-regional
- 3A/3B: Only new-build counts (additionality); 3C/3D: existing + new
- Learning curve: Scenario A (delayed) — annual flexibility defers firm clean
- 3D is the "status quo" — cheapest unbundled RECs from anywhere

Output: data/step5-post-processing/strategy3_annual.json
"""

import os
import sys
import json
import argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
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
    get_emission_rate,
    get_existing_clean_twh, get_sss_twh,
    make_strategy_result, build_25yr_trajectory,
    save_results_json,
    UPRATE_CAP_TWH, EXISTING_EAC_PRICE,
)

# ═══════════════════════════════════════════════════════════════════════════════
# ANNUAL MATCHING: NO OVER-PROCUREMENT PENALTY
# ═══════════════════════════════════════════════════════════════════════════════
# Annual matching only requires total MWh to match over the year.
# No hourly constraint → no over-procurement penalty (ratio = 1.0).
# However, at very high thresholds, even annual matching needs some slack
# because the "last few %" of demand are expensive to cover with new clean.

ANNUAL_SLACK = {
    # threshold: slight procurement buffer for annual matching
    50:    1.00,
    70:    1.00,
    85:    1.02,
    90:    1.03,
    95:    1.05,
    99:    1.08,
    99.99: 1.10,
}


def get_annual_slack(threshold):
    """Get annual matching procurement slack factor."""
    thresholds = sorted(ANNUAL_SLACK.keys())
    values = [ANNUAL_SLACK[t] for t in thresholds]

    if threshold <= thresholds[0]:
        return values[0]
    if threshold >= thresholds[-1]:
        return values[-1]

    for i in range(len(thresholds) - 1):
        if thresholds[i] <= threshold <= thresholds[i + 1]:
            frac = (threshold - thresholds[i]) / (thresholds[i + 1] - thresholds[i])
            return values[i] + frac * (values[i + 1] - values[i])
    return values[-1]


# ═══════════════════════════════════════════════════════════════════════════════
# UNBUNDLED REC PRICES (for non-additionality strategies 3C/3D)
# ═══════════════════════════════════════════════════════════════════════════════
# Existing clean generators selling unbundled RECs at low prices.
# Cross-regional RECs are cheaper (larger pool, more competition).

UNBUNDLED_REC_PRICE = {
    # Same-ISO REC prices ($/MWh)
    'same_iso': {
        'CAISO': 5.0,   # CA compliance market — higher REC value
        'ERCOT': 2.5,   # Abundant wind, low-value RECs
        'PJM':   4.0,   # PJM-GATS compliance value
        'NYISO': 6.0,   # NY Tier 4 local premium
        'NEISO': 5.5,   # NE-GIS, MA/CT SREC remnants
        'MISO':  3.0,   # Low-cost wind RECs
        'SPP':   2.0,   # Cheapest — massive wind surplus
    },
    # Cross-regional (buyer gets cheapest from any ISO)
    'cross_regional': 2.0,  # $/MWh — national floor from SPP/ERCOT surplus
}

# At high participation, REC prices escalate (scarcity)
REC_SCARCITY_MULTIPLIER = {
    # participation → multiplier on base REC price
    0.02: 1.0,
    0.05: 1.0,
    0.10: 1.1,
    0.20: 1.3,
    0.30: 1.6,
    0.40: 2.0,
    0.50: 2.5,
    0.60: 3.2,
    0.80: 5.0,
}


def get_rec_scarcity_mult(participation):
    """Get REC scarcity multiplier at a given participation level."""
    levels = sorted(REC_SCARCITY_MULTIPLIER.keys())
    values = [REC_SCARCITY_MULTIPLIER[l] for l in levels]

    if participation <= levels[0]:
        return values[0]
    if participation >= levels[-1]:
        return values[-1]

    for i in range(len(levels) - 1):
        if levels[i] <= participation <= levels[i + 1]:
            frac = (participation - levels[i]) / (levels[i + 1] - levels[i])
            return values[i] + frac * (values[i + 1] - values[i])
    return values[-1]


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 3A: SAME-ISO, NEW BUILD REQUIRED
# ═══════════════════════════════════════════════════════════════════════════════


def compute_strategy_3a(iso, year, threshold, participation_pct,
                         growth_level='Medium', scenario='A',
                         level='Medium', ppa_level='Medium', **kwargs):
    """Strategy 3A: Annual matching, same-ISO, new build only.

    Annual version of 2A — same resource constraint but no hourly penalty.
    Cheapest annual new-build: VRE-heavy at low thresholds (no temporal constraint
    means solar/wind surplus isn't wasted).
    """
    buyer_demand = get_buyer_demand_twh(iso, year, participation_pct, growth_level)
    if buyer_demand <= 0 or threshold <= 0:
        return make_strategy_result('3A', iso, year, threshold, participation_pct,
                                     0, 0, 0, 0)

    target_fraction = min(threshold / 100.0, 1.0)
    slack = get_annual_slack(threshold)
    clean_needed = buyer_demand * target_fraction * slack

    # Annual matching is VRE-dominated (no hourly constraint makes VRE surplus useful)
    # At higher thresholds, some firm is needed even annually (base coverage)
    if threshold <= 70:
        mix = {'solar': 0.50, 'wind': 0.45, 'uprate': 0.05}
    elif threshold <= 90:
        mix = {'solar': 0.40, 'wind': 0.35, 'firm': 0.15, 'uprate': 0.05, 'storage': 0.05}
    else:
        mix = {'solar': 0.30, 'wind': 0.28, 'firm': 0.22, 'uprate': 0.05, 'storage': 0.15}

    total_cost = 0.0
    resource_breakdown = {}

    for resource, frac in mix.items():
        twh = clean_needed * frac

        if resource == 'solar':
            price = get_learning_adjusted_ppa('solar', iso, threshold, scenario, level, ppa_level)
        elif resource == 'wind':
            price = get_learning_adjusted_ppa('wind', iso, threshold, scenario, level, ppa_level)
        elif resource == 'firm':
            nuc = get_learning_adjusted_ppa('nuclear_newbuild', iso, threshold, scenario, level, ppa_level)
            ccs = get_learning_adjusted_ppa('ccs_45q_on', iso, threshold, scenario, level, ppa_level)
            price = min(nuc, ccs)
        elif resource == 'storage':
            price = get_ppa_price('battery', iso, level, ppa_level)
        elif resource == 'uprate':
            twh = min(twh, UPRATE_CAP_TWH.get(iso, 0))
            price = get_learning_adjusted_ppa('uprate', iso, threshold, scenario, level, ppa_level)
        else:
            price = 50

        cost = twh * price
        total_cost += cost
        resource_breakdown[resource] = {'twh': round(twh, 2), 'price': round(price, 1), 'cost_m': round(cost, 2)}

    cost_per_mwh = total_cost / clean_needed if clean_needed > 0 else 0
    emission_rate = get_emission_rate(iso, 'grid_average')  # Annual matching, grid-avg baseline
    co2_abated = buyer_demand * target_fraction * emission_rate / 1e3
    mac = (total_cost * 1e6) / (co2_abated * 1e6) if co2_abated > 0 else None

    return make_strategy_result(
        '3A', iso, year, threshold, participation_pct,
        cost_per_mwh, total_cost, co2_abated, mac,
        resource_mix=resource_breakdown,
        metadata={'annual_slack': round(slack, 3), 'scenario': scenario},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 3B: CROSS-REGIONAL, NEW BUILD REQUIRED
# ═══════════════════════════════════════════════════════════════════════════════


def compute_strategy_3b(iso, year, threshold, participation_pct,
                         growth_level='Medium', scenario='A',
                         level='Medium', ppa_level='Medium', **kwargs):
    """Strategy 3B: Annual matching, cross-regional, new build only.

    Like Strategy 1 (consequential) but annual volumetric rather than emission netting.
    Can procure cheapest new-build VRE from any ISO (SPP wind, ERCOT solar).
    """
    buyer_demand = get_buyer_demand_twh(iso, year, participation_pct, growth_level)
    if buyer_demand <= 0 or threshold <= 0:
        return make_strategy_result('3B', iso, year, threshold, participation_pct,
                                     0, 0, 0, 0)

    target_fraction = min(threshold / 100.0, 1.0)
    slack = get_annual_slack(threshold)
    clean_needed = buyer_demand * target_fraction * slack

    # Find cheapest new-build sources across all ISOs
    sources = []
    for src_iso in ISOS:
        solar_ppa = get_learning_adjusted_ppa('solar', src_iso, threshold, scenario, level, ppa_level)
        sources.append((src_iso, 'solar', solar_ppa, BASE_DEMAND_TWH[src_iso] * 0.30))

        wind_ppa = get_learning_adjusted_ppa('wind', src_iso, threshold, scenario, level, ppa_level)
        sources.append((src_iso, 'wind', wind_ppa, BASE_DEMAND_TWH[src_iso] * 0.25))

        uprate_twh = UPRATE_CAP_TWH.get(src_iso, 0)
        if uprate_twh > 0:
            uprate_ppa = get_learning_adjusted_ppa('uprate', src_iso, threshold, scenario, level, ppa_level)
            sources.append((src_iso, 'uprate', uprate_ppa, uprate_twh))

        if threshold >= 85:  # Only need firm at high thresholds
            nuc = get_learning_adjusted_ppa('nuclear_newbuild', src_iso, threshold, scenario, level, ppa_level)
            sources.append((src_iso, 'firm', nuc, BASE_DEMAND_TWH[src_iso] * 0.10))

    sources.sort(key=lambda s: s[2])

    total_cost = 0.0
    total_procured = 0.0
    resource_breakdown = {}

    for src_iso, resource, price, available in sources:
        if total_procured >= clean_needed:
            break
        remaining = clean_needed - total_procured
        procure = min(remaining, available)

        cost = procure * price
        total_cost += cost
        total_procured += procure

        key = f"{src_iso}_{resource}"
        resource_breakdown[key] = {'twh': round(procure, 2), 'price': round(price, 1), 'cost_m': round(cost, 2)}

    cost_per_mwh = total_cost / total_procured if total_procured > 0 else 0
    emission_rate = get_emission_rate(iso, 'grid_average')
    co2_abated = buyer_demand * target_fraction * emission_rate / 1e3
    mac = (total_cost * 1e6) / (co2_abated * 1e6) if co2_abated > 0 else None

    return make_strategy_result(
        '3B', iso, year, threshold, participation_pct,
        cost_per_mwh, total_cost, co2_abated, mac,
        resource_mix=resource_breakdown,
        metadata={'annual_slack': round(slack, 3), 'scenario': scenario,
                  'cross_regional': True},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 3C: SAME-ISO, NO ADDITIONALITY
# ═══════════════════════════════════════════════════════════════════════════════


def compute_strategy_3c(iso, year, threshold, participation_pct,
                         growth_level='Medium', scenario='A',
                         level='Medium', ppa_level='Medium', **kwargs):
    """Strategy 3C: Annual matching, same-ISO, no additionality.

    Existing clean counts. Can purchase unbundled RECs from same-ISO generators.
    Cheapest same-ISO option but no environmental signal for new build.
    """
    buyer_demand = get_buyer_demand_twh(iso, year, participation_pct, growth_level)
    if buyer_demand <= 0 or threshold <= 0:
        return make_strategy_result('3C', iso, year, threshold, participation_pct,
                                     0, 0, 0, 0)

    total_demand = get_demand_twh_at_year(iso, year, growth_level)
    target_fraction = min(threshold / 100.0, 1.0)
    clean_needed = buyer_demand * target_fraction

    # Available existing clean for REC purchase (total existing minus SSS)
    existing_clean = get_existing_clean_twh(iso)
    sss_twh = get_sss_twh(iso, year, growth_level)
    available_existing_recs = max(0, existing_clean - sss_twh)

    # Buyer's share of available RECs
    buyer_share = buyer_demand / total_demand if total_demand > 0 else 0
    buyer_available_recs = available_existing_recs * buyer_share

    # REC pricing with scarcity
    rec_base = UNBUNDLED_REC_PRICE['same_iso'][iso]
    rec_scarcity = get_rec_scarcity_mult(participation_pct)
    rec_price = rec_base * rec_scarcity

    total_cost = 0.0
    resource_breakdown = {}

    # First: buy existing RECs (cheap)
    recs_purchased = min(clean_needed, buyer_available_recs)
    if recs_purchased > 0:
        cost = recs_purchased * rec_price
        total_cost += cost
        resource_breakdown['existing_recs'] = {
            'twh': round(recs_purchased, 2),
            'price': round(rec_price, 1),
            'cost_m': round(cost, 2),
        }

    # Remaining: need new-build (even without additionality requirement,
    # if existing RECs are exhausted, new build is needed)
    remaining = max(0, clean_needed - recs_purchased)
    if remaining > 0:
        slack = get_annual_slack(threshold)
        new_build_twh = remaining * slack

        solar_ppa = get_learning_adjusted_ppa('solar', iso, threshold, scenario, level, ppa_level)
        wind_ppa = get_learning_adjusted_ppa('wind', iso, threshold, scenario, level, ppa_level)
        avg_vre = min(solar_ppa, wind_ppa)

        cost = new_build_twh * avg_vre
        total_cost += cost
        resource_breakdown['new_build_vre'] = {
            'twh': round(new_build_twh, 2),
            'price': round(avg_vre, 1),
            'cost_m': round(cost, 2),
        }

    total_procured = recs_purchased + remaining
    cost_per_mwh = total_cost / total_procured if total_procured > 0 else 0
    emission_rate = get_emission_rate(iso, 'grid_average')
    co2_abated = buyer_demand * target_fraction * emission_rate / 1e3
    mac = (total_cost * 1e6) / (co2_abated * 1e6) if co2_abated > 0 else None

    return make_strategy_result(
        '3C', iso, year, threshold, participation_pct,
        cost_per_mwh, total_cost, co2_abated, mac,
        resource_mix=resource_breakdown,
        metadata={'rec_price': round(rec_price, 2), 'scarcity_mult': round(rec_scarcity, 2),
                  'scenario': scenario},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 3D: CROSS-REGIONAL, NO ADDITIONALITY (STATUS QUO)
# ═══════════════════════════════════════════════════════════════════════════════


def compute_strategy_3d(iso, year, threshold, participation_pct,
                         growth_level='Medium', scenario='A',
                         level='Medium', ppa_level='Medium', **kwargs):
    """Strategy 3D: Annual matching, cross-regional, no additionality.

    Cheapest option — unbundled RECs from anywhere in the US.
    This is the current "status quo" for most corporate procurement.
    Very cheap but near-zero environmental impact (existing generators
    already running, no signal for new build).
    """
    buyer_demand = get_buyer_demand_twh(iso, year, participation_pct, growth_level)
    if buyer_demand <= 0 or threshold <= 0:
        return make_strategy_result('3D', iso, year, threshold, participation_pct,
                                     0, 0, 0, 0)

    target_fraction = min(threshold / 100.0, 1.0)
    clean_needed = buyer_demand * target_fraction

    # Cross-regional RECs at national floor price
    rec_base = UNBUNDLED_REC_PRICE['cross_regional']
    rec_scarcity = get_rec_scarcity_mult(participation_pct)
    rec_price = rec_base * rec_scarcity

    # Total available cross-regional RECs (all ISOs' existing clean minus SSS)
    total_available_recs = 0
    for src_iso in ISOS:
        ec = get_existing_clean_twh(src_iso)
        sss = get_sss_twh(src_iso, year, growth_level)
        total_available_recs += max(0, ec - sss)

    # All buyers compete for same REC pool
    # Simplified: assume buyer can get their share
    total_cost = 0.0
    resource_breakdown = {}

    recs_purchased = min(clean_needed, total_available_recs * 0.05)  # Assume max 5% of pool
    if recs_purchased > 0:
        cost = recs_purchased * rec_price
        total_cost += cost
        resource_breakdown['cross_regional_recs'] = {
            'twh': round(recs_purchased, 2),
            'price': round(rec_price, 1),
            'cost_m': round(cost, 2),
        }

    # If RECs exhausted, spillover to cheapest new-build anywhere
    remaining = max(0, clean_needed - recs_purchased)
    if remaining > 0:
        # Cheapest cross-regional VRE (SPP wind is typically cheapest)
        cheapest_price = float('inf')
        cheapest_src = 'SPP'
        for src_iso in ISOS:
            wind = get_learning_adjusted_ppa('wind', src_iso, threshold, scenario, level, ppa_level)
            solar = get_learning_adjusted_ppa('solar', src_iso, threshold, scenario, level, ppa_level)
            p = min(wind, solar)
            if p < cheapest_price:
                cheapest_price = p
                cheapest_src = src_iso

        cost = remaining * cheapest_price
        total_cost += cost
        resource_breakdown[f'{cheapest_src}_new_vre'] = {
            'twh': round(remaining, 2),
            'price': round(cheapest_price, 1),
            'cost_m': round(cost, 2),
        }

    total_procured = clean_needed
    cost_per_mwh = total_cost / total_procured if total_procured > 0 else 0

    # CO₂ abated is questionable for status quo (existing generators already running)
    # Use reduced emission rate to reflect marginal impact
    emission_rate = get_emission_rate(iso, 'grid_average')
    # Discount: only 10-30% of REC purchases lead to actual emission reduction
    additionality_discount = 0.15 if recs_purchased > 0 else 0.80
    co2_abated = buyer_demand * target_fraction * emission_rate * additionality_discount / 1e3
    mac = (total_cost * 1e6) / (co2_abated * 1e6) if co2_abated > 0 else None

    return make_strategy_result(
        '3D', iso, year, threshold, participation_pct,
        cost_per_mwh, total_cost, co2_abated, mac,
        resource_mix=resource_breakdown,
        metadata={
            'rec_price': round(rec_price, 2),
            'scarcity_mult': round(rec_scarcity, 2),
            'additionality_discount': additionality_discount,
            'cross_regional': True,
            'scenario': scenario,
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FULL SWEEP
# ═══════════════════════════════════════════════════════════════════════════════

STRATEGY_3_FUNCS = {
    '3A': compute_strategy_3a,
    '3B': compute_strategy_3b,
    '3C': compute_strategy_3c,
    '3D': compute_strategy_3d,
}


def run_full_sweep(isos=None, participation_levels=None, growth_levels=None):
    """Run Strategy 3 (all variants) for all ISOs × participation × growth."""
    if isos is None:
        isos = ISOS
    if participation_levels is None:
        participation_levels = PARTICIPATION_LEVELS
    if growth_levels is None:
        growth_levels = ['Low', 'Medium', 'High']

    all_results = {}
    total = len(STRATEGY_3_FUNCS) * len(isos) * len(growth_levels) * len(participation_levels)
    count = 0

    for variant_id, compute_fn in STRATEGY_3_FUNCS.items():
        strategy_key = f'strategy{variant_id}'
        all_results[strategy_key] = {}

        for iso in isos:
            all_results[strategy_key][iso] = {}

            for growth in growth_levels:
                all_results[strategy_key][iso][growth] = {}

                for part in participation_levels:
                    count += 1
                    if count % 50 == 0:
                        print(f"  [{count}/{total}] {variant_id} {iso} {growth} part={part*100:.0f}%")

                    trajectory = []
                    for year in TIMELINE_YEARS:
                        threshold = get_threshold_for_year(year)
                        result = compute_fn(
                            iso=iso, year=year, threshold=threshold,
                            participation_pct=part,
                            growth_level=growth,
                            scenario='A',  # Annual always uses Scenario A
                            level='Medium',
                            ppa_level='Medium',
                        )
                        result['year'] = year
                        result['demand_twh'] = round(get_demand_twh_at_year(iso, year, growth), 2)
                        result['buyer_demand_twh'] = round(get_buyer_demand_twh(iso, year, part, growth), 2)
                        result['threshold'] = round(threshold, 2)
                        trajectory.append(result)

                    part_key = f'{part*100:.0f}pct'
                    all_results[strategy_key][iso][growth][part_key] = trajectory

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description='Step 6.5 — Strategy 3: Annual Matching'
    )
    parser.add_argument('--iso', type=str, default='ALL',
                        help='ISO to compute (default ALL)')
    parser.add_argument('--participation', type=float, nargs='+', default=None,
                        help='Participation levels')
    parser.add_argument('--growth', type=str, nargs='+', default=['Low', 'Medium', 'High'],
                        help='Demand growth levels')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: Medium growth, fewer participation levels')
    args = parser.parse_args()

    isos = ISOS if args.iso == 'ALL' else [args.iso]

    if args.quick:
        participation = [0.05, 0.10, 0.20, 0.40]
        growth_levels = ['Medium']
    else:
        participation = args.participation or PARTICIPATION_LEVELS
        growth_levels = args.growth

    print("=" * 70)
    print("Step 6.5 — Strategy 3: Annual Matching")
    print("=" * 70)
    print(f"ISOs: {isos}")
    print(f"Participation levels: {[f'{p*100:.0f}%' for p in participation]}")
    print(f"Growth levels: {growth_levels}")
    print(f"Variants: 3A (same-ISO new), 3B (cross new), 3C (same-ISO any), 3D (cross any)")
    print(f"Scenario: A (delayed learning curve)")
    print()

    results = run_full_sweep(
        isos=isos,
        participation_levels=participation,
        growth_levels=growth_levels,
    )

    save_results_json(results, 'strategy3_annual.json')

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Strategy 3 at 10% participation, Medium growth")
    print("=" * 70)
    for variant in ['strategy3A', 'strategy3B', 'strategy3C', 'strategy3D']:
        if variant not in results:
            continue
        print(f"\n--- {variant.upper()} ---")
        for iso in isos:
            data = results[variant].get(iso, {}).get('Medium', {}).get('10pct', [])
            if not data:
                continue
            milestones = {r['year']: r for r in data if r['year'] in [2030, 2035, 2040, 2050]}
            for yr, r in sorted(milestones.items()):
                print(f"  {iso} {yr}: ${r['cost_per_mwh']:.1f}/MWh, "
                      f"{r['co2_abated_mmt']:.3f} MtCO₂, "
                      f"threshold={r['threshold']:.0f}%")

    print("\nDone.")


if __name__ == '__main__':
    main()
