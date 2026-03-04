#!/usr/bin/env python3
"""
Step 6.5 — Strategy 2: Hourly Matching (Same-ISO)
===================================================
Models three variants of hourly CFE matching:
  2A: 100% new build — no existing clean credit
  2B: Grid baseline — existing clean at 8760 shape is free
  2C: SSS allocation + premium tranches — pro-rata SSS + existing at premium + new-build

Key characteristics:
- Same-ISO: buyer must procure within their own ISO
- Hourly: every hour must be matched (CFE threshold applied per-hour)
- Highest verifiability, highest cost
- Learning curve: Scenario B (accelerated) — FOAK→NOAK by 2040
- Forces early clean firm + storage investment → drives learning curve

The hourly matching premium is captured by the over-procurement ratio:
higher CFE targets require more over-procurement because VRE doesn't
match demand hour-by-hour. This comes from the existing optimizer results.

Output: data/step5-post-processing/strategy2_hourly.json
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
    CI_SHARE, BASE_DEMAND_TWH, H,
    GRID_MIX_SHARES,
    get_demand_twh_at_year, get_ci_demand_twh, get_buyer_demand_twh,
    get_threshold_for_year,
    get_lcoe, get_ppa_price, get_learning_adjusted_lcoe, get_learning_adjusted_ppa,
    get_wholesale_price, estimate_lmp_at_clean_pct,
    get_emission_rate, compute_co2_abated,
    get_existing_clean_twh, get_sss_twh,
    get_sss_hourly_shape, get_existing_clean_hourly_shape,
    build_procurement_tranches,
    make_strategy_result, build_25yr_trajectory,
    save_results_json,
    UPRATE_CAP_TWH, EXISTING_EAC_PRICE,
)

# ═══════════════════════════════════════════════════════════════════════════════
# OVER-PROCUREMENT RATIOS (from optimizer physics)
# ═══════════════════════════════════════════════════════════════════════════════
# At higher CFE thresholds, hourly matching requires more over-procurement
# because VRE generation doesn't match demand shape hour-by-hour.
# These ratios are derived from Step 1 PFS physics (actual optimizer results).

OVER_PROCUREMENT_RATIO = {
    # threshold: ratio (procured TWh / matched TWh)
    # Represents the physics penalty of hourly matching vs annual
    50:    1.05,   # Minimal — mostly baseload hours
    55:    1.07,
    60:    1.10,
    65:    1.13,
    70:    1.16,
    75:    1.20,   # Increasing — VRE surplus in daytime, gap at night
    80:    1.25,
    85:    1.32,
    87.5:  1.38,
    90:    1.45,   # Steep — need storage + firm for nighttime
    92.5:  1.55,
    95:    1.70,   # Very steep — every hour matters
    97.5:  1.90,
    99:    2.20,   # Extreme — last few % require massive overbuild
    99.99: 3.00,   # Physical limit — need 3× capacity to cover every hour
}


def get_over_procurement_ratio(threshold):
    """Interpolate over-procurement ratio for a given threshold."""
    thresholds = sorted(OVER_PROCUREMENT_RATIO.keys())
    ratios = [OVER_PROCUREMENT_RATIO[t] for t in thresholds]

    if threshold <= thresholds[0]:
        return ratios[0]
    if threshold >= thresholds[-1]:
        return ratios[-1]

    for i in range(len(thresholds) - 1):
        if thresholds[i] <= threshold <= thresholds[i + 1]:
            frac = (threshold - thresholds[i]) / (thresholds[i + 1] - thresholds[i])
            return ratios[i] + frac * (ratios[i + 1] - ratios[i])
    return ratios[-1]


# ═══════════════════════════════════════════════════════════════════════════════
# RESOURCE MIX TEMPLATES (from optimizer results at Medium costs)
# ═══════════════════════════════════════════════════════════════════════════════
# Approximate resource mix fractions at each threshold for hourly matching.
# These capture the physics: at low thresholds VRE dominates, at high thresholds
# firm + storage become essential.

HOURLY_MIX_TEMPLATE = {
    # threshold: {resource: fraction of total procurement}
    50:    {'solar': 0.40, 'wind': 0.35, 'firm': 0.10, 'storage': 0.10, 'uprate': 0.05},
    70:    {'solar': 0.35, 'wind': 0.30, 'firm': 0.15, 'storage': 0.15, 'uprate': 0.05},
    85:    {'solar': 0.25, 'wind': 0.25, 'firm': 0.25, 'storage': 0.20, 'uprate': 0.05},
    90:    {'solar': 0.22, 'wind': 0.22, 'firm': 0.28, 'storage': 0.23, 'uprate': 0.05},
    95:    {'solar': 0.18, 'wind': 0.18, 'firm': 0.32, 'storage': 0.27, 'uprate': 0.05},
    99:    {'solar': 0.15, 'wind': 0.15, 'firm': 0.35, 'storage': 0.30, 'uprate': 0.05},
    99.99: {'solar': 0.12, 'wind': 0.12, 'firm': 0.38, 'storage': 0.33, 'uprate': 0.05},
}


def get_resource_mix_fractions(threshold):
    """Interpolate resource mix fractions for a given threshold."""
    thresholds = sorted(HOURLY_MIX_TEMPLATE.keys())
    resources = list(HOURLY_MIX_TEMPLATE[thresholds[0]].keys())

    if threshold <= thresholds[0]:
        return dict(HOURLY_MIX_TEMPLATE[thresholds[0]])
    if threshold >= thresholds[-1]:
        return dict(HOURLY_MIX_TEMPLATE[thresholds[-1]])

    # Find bracketing thresholds
    for i in range(len(thresholds) - 1):
        if thresholds[i] <= threshold <= thresholds[i + 1]:
            frac = (threshold - thresholds[i]) / (thresholds[i + 1] - thresholds[i])
            low = HOURLY_MIX_TEMPLATE[thresholds[i]]
            high = HOURLY_MIX_TEMPLATE[thresholds[i + 1]]
            return {r: low[r] + frac * (high[r] - low[r]) for r in resources}

    return dict(HOURLY_MIX_TEMPLATE[thresholds[-1]])


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 2A: ALL NEW BUILD (NO EXISTING CLEAN CREDIT)
# ═══════════════════════════════════════════════════════════════════════════════


def compute_strategy_2a(iso, year, threshold, participation_pct,
                         growth_level='Medium', scenario='B',
                         level='Medium', ppa_level='Medium', **kwargs):
    """Strategy 2A: 100% new-build hourly matching. No existing clean credit.

    Maximum additionality. Equivalent to existing Track 2 NB analysis.
    Most expensive but strongest signal for new clean investment.
    """
    buyer_demand = get_buyer_demand_twh(iso, year, participation_pct, growth_level)
    if buyer_demand <= 0 or threshold <= 0:
        return make_strategy_result('2A', iso, year, threshold, participation_pct,
                                     0, 0, 0, 0)

    target_fraction = min(threshold / 100.0, 1.0)
    op_ratio = get_over_procurement_ratio(threshold)
    clean_needed_twh = buyer_demand * target_fraction * op_ratio

    # Resource mix at this threshold
    mix_fracs = get_resource_mix_fractions(threshold)

    # Price each resource
    total_cost = 0.0
    resource_breakdown = {}

    for resource, frac in mix_fracs.items():
        twh = clean_needed_twh * frac

        if resource == 'solar':
            price = get_learning_adjusted_ppa('solar', iso, threshold, scenario, level, ppa_level)
        elif resource == 'wind':
            price = get_learning_adjusted_ppa('wind', iso, threshold, scenario, level, ppa_level)
        elif resource == 'firm':
            # Cheapest firm: nuclear newbuild or CCS
            nuc = get_learning_adjusted_ppa('nuclear_newbuild', iso, threshold, scenario, level, ppa_level)
            ccs = get_learning_adjusted_ppa('ccs_45q_on', iso, threshold, scenario, level, ppa_level)
            price = min(nuc, ccs)
        elif resource == 'storage':
            # Weighted: 60% battery (4hr), 40% LDES (for high thresholds)
            batt = get_ppa_price('battery', iso, level, ppa_level)
            ldes = get_learning_adjusted_ppa('ldes', iso, threshold, scenario, level, ppa_level)
            storage_ldes_frac = min(0.6, max(0.2, (threshold - 70) / 60))  # More LDES at higher thresholds
            price = batt * (1 - storage_ldes_frac) + ldes * storage_ldes_frac
        elif resource == 'uprate':
            uprate_cap = UPRATE_CAP_TWH.get(iso, 0)
            twh = min(twh, uprate_cap)
            price = get_learning_adjusted_ppa('uprate', iso, threshold, scenario, level, ppa_level)
        else:
            price = 50  # Fallback

        cost = twh * price  # TWh × $/MWh = $M
        total_cost += cost
        resource_breakdown[resource] = {
            'twh': round(twh, 2),
            'price': round(price, 1),
            'cost_m': round(cost, 2),
        }

    # Metrics
    cost_per_mwh = total_cost / clean_needed_twh if clean_needed_twh > 0 else 0
    emission_rate = get_emission_rate(iso, 'fossil_average')  # Hourly displaces fossil
    co2_abated = buyer_demand * target_fraction * emission_rate / 1e3
    mac = (total_cost * 1e6) / (co2_abated * 1e6) if co2_abated > 0 else None

    return make_strategy_result(
        '2A', iso, year, threshold, participation_pct,
        cost_per_mwh, total_cost, co2_abated, mac,
        resource_mix=resource_breakdown,
        metadata={
            'over_procurement_ratio': round(op_ratio, 3),
            'clean_procured_twh': round(clean_needed_twh, 2),
            'scenario': scenario,
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 2B: GRID BASELINE (EXISTING CLEAN FREE)
# ═══════════════════════════════════════════════════════════════════════════════


def compute_strategy_2b(iso, year, threshold, participation_pct,
                         growth_level='Medium', scenario='B',
                         level='Medium', ppa_level='Medium', **kwargs):
    """Strategy 2B: Grid baseline — existing clean at 8760 shape is free.

    Buyer takes credit for existing clean grid mix. Only procures new-build
    above the existing baseline. Cheapest hourly variant but creates
    stranding risk (no revenue signal for existing clean).
    """
    buyer_demand = get_buyer_demand_twh(iso, year, participation_pct, growth_level)
    if buyer_demand <= 0 or threshold <= 0:
        return make_strategy_result('2B', iso, year, threshold, participation_pct,
                                     0, 0, 0, 0)

    total_demand = get_demand_twh_at_year(iso, year, growth_level)
    target_fraction = min(threshold / 100.0, 1.0)

    # Existing clean covers some of the target for free
    existing_clean = get_existing_clean_twh(iso)
    existing_clean_pct = existing_clean / total_demand if total_demand > 0 else 0

    # Buyer's pro-rata share of existing clean (hourly shape is free)
    buyer_existing_clean_twh = buyer_demand * existing_clean_pct

    # Remaining needed above existing baseline
    needed_twh = buyer_demand * target_fraction - buyer_existing_clean_twh
    needed_twh = max(0, needed_twh)

    # Apply over-procurement ratio to the new-build portion
    op_ratio = get_over_procurement_ratio(threshold)
    new_build_twh = needed_twh * op_ratio

    if new_build_twh <= 0:
        # Existing clean is sufficient — no new procurement needed
        return make_strategy_result('2B', iso, year, threshold, participation_pct,
                                     0, 0,
                                     buyer_demand * target_fraction * get_emission_rate(iso, 'fossil_average') / 1e3,
                                     0,
                                     metadata={'existing_covers_target': True})

    # Price the new-build procurement (same mix logic as 2A)
    mix_fracs = get_resource_mix_fractions(threshold)
    total_cost = 0.0
    resource_breakdown = {}

    for resource, frac in mix_fracs.items():
        twh = new_build_twh * frac

        if resource == 'solar':
            price = get_learning_adjusted_ppa('solar', iso, threshold, scenario, level, ppa_level)
        elif resource == 'wind':
            price = get_learning_adjusted_ppa('wind', iso, threshold, scenario, level, ppa_level)
        elif resource == 'firm':
            nuc = get_learning_adjusted_ppa('nuclear_newbuild', iso, threshold, scenario, level, ppa_level)
            ccs = get_learning_adjusted_ppa('ccs_45q_on', iso, threshold, scenario, level, ppa_level)
            price = min(nuc, ccs)
        elif resource == 'storage':
            batt = get_ppa_price('battery', iso, level, ppa_level)
            ldes = get_learning_adjusted_ppa('ldes', iso, threshold, scenario, level, ppa_level)
            storage_ldes_frac = min(0.6, max(0.2, (threshold - 70) / 60))
            price = batt * (1 - storage_ldes_frac) + ldes * storage_ldes_frac
        elif resource == 'uprate':
            uprate_cap = UPRATE_CAP_TWH.get(iso, 0)
            twh = min(twh, uprate_cap)
            price = get_learning_adjusted_ppa('uprate', iso, threshold, scenario, level, ppa_level)
        else:
            price = 50

        cost = twh * price
        total_cost += cost
        resource_breakdown[resource] = {
            'twh': round(twh, 2),
            'price': round(price, 1),
            'cost_m': round(cost, 2),
        }

    # Add existing clean as zero-cost entry
    resource_breakdown['existing_clean_free'] = {
        'twh': round(buyer_existing_clean_twh, 2),
        'price': 0.0,
        'cost_m': 0.0,
    }

    effective_procured = new_build_twh + buyer_existing_clean_twh
    cost_per_mwh = total_cost / effective_procured if effective_procured > 0 else 0
    emission_rate = get_emission_rate(iso, 'fossil_average')
    co2_abated = buyer_demand * target_fraction * emission_rate / 1e3
    mac = (total_cost * 1e6) / (co2_abated * 1e6) if co2_abated > 0 else None

    return make_strategy_result(
        '2B', iso, year, threshold, participation_pct,
        cost_per_mwh, total_cost, co2_abated, mac,
        resource_mix=resource_breakdown,
        metadata={
            'existing_clean_pct': round(existing_clean_pct * 100, 1),
            'new_build_twh': round(new_build_twh, 2),
            'over_procurement_ratio': round(op_ratio, 3),
            'scenario': scenario,
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 2C: SSS ALLOCATION + PREMIUM TRANCHES
# ═══════════════════════════════════════════════════════════════════════════════


def compute_strategy_2c(iso, year, threshold, participation_pct,
                         growth_level='Medium', scenario='B',
                         level='Medium', ppa_level='Medium',
                         use_45u=True, use_ctr=False, ctr_value=None, **kwargs):
    """Strategy 2C: SSS allocation + premium tranches for hourly matching.

    Layer 1: SSS allocation (free, follows SSS 8760 shape)
    Layer 2: Existing clean beyond SSS (at premium prices via tranches)
    Layer 3: New-build (at PPA prices)

    The tranche merit-order determines cost (§15.14.4).
    """
    buyer_demand = get_buyer_demand_twh(iso, year, participation_pct, growth_level)
    if buyer_demand <= 0 or threshold <= 0:
        return make_strategy_result('2C', iso, year, threshold, participation_pct,
                                     0, 0, 0, 0)

    total_demand = get_demand_twh_at_year(iso, year, growth_level)
    target_fraction = min(threshold / 100.0, 1.0)

    # Layer 1: SSS allocation (free)
    sss_twh = get_sss_twh(iso, year, growth_level)
    buyer_sss_share = (buyer_demand / total_demand) * sss_twh if total_demand > 0 else 0

    # Total clean needed (buyer's share of target)
    total_clean_needed = buyer_demand * target_fraction

    # Remaining above SSS
    remaining = max(0, total_clean_needed - buyer_sss_share)

    # Apply over-procurement ratio to remaining
    op_ratio = get_over_procurement_ratio(threshold)
    procurement_twh = remaining * op_ratio

    if procurement_twh <= 0:
        return make_strategy_result('2C', iso, year, threshold, participation_pct,
                                     0, 0,
                                     total_clean_needed * get_emission_rate(iso, 'fossil_average') / 1e3,
                                     0,
                                     metadata={'sss_covers_target': True,
                                              'sss_share_twh': round(buyer_sss_share, 2)})

    # Build procurement tranches and walk up the merit order
    tranches = build_procurement_tranches(
        iso, threshold, year, scenario, level, ppa_level,
        use_45u, use_ctr, ctr_value, growth_level,
    )

    total_cost = 0.0
    total_procured = 0.0
    resource_breakdown = {}

    # Walk tranches
    for tranche in tranches:
        if total_procured >= procurement_twh:
            break

        remaining_need = procurement_twh - total_procured
        # Scale available by buyer's share of market
        buyer_share_of_available = (buyer_demand / total_demand) if total_demand > 0 else 0.01
        available = tranche['available_twh'] * buyer_share_of_available
        procure = min(remaining_need, available)

        cost = procure * tranche['price']
        total_cost += cost
        total_procured += procure

        resource_breakdown[tranche['source']] = {
            'twh': round(procure, 2),
            'price': round(tranche['price'], 1),
            'cost_m': round(cost, 2),
            'category': tranche['category'],
        }

    # If tranches exhausted, fill with new-build VRE + firm at market prices
    if total_procured < procurement_twh:
        shortfall = procurement_twh - total_procured
        # Use hourly mix template for the shortfall
        mix_fracs = get_resource_mix_fractions(threshold)
        for resource, frac in mix_fracs.items():
            twh = shortfall * frac
            if resource == 'solar':
                price = get_learning_adjusted_ppa('solar', iso, threshold, scenario, level, ppa_level)
            elif resource == 'wind':
                price = get_learning_adjusted_ppa('wind', iso, threshold, scenario, level, ppa_level)
            elif resource == 'firm':
                nuc = get_learning_adjusted_ppa('nuclear_newbuild', iso, threshold, scenario, level, ppa_level)
                ccs = get_learning_adjusted_ppa('ccs_45q_on', iso, threshold, scenario, level, ppa_level)
                price = min(nuc, ccs)
            elif resource == 'storage':
                batt = get_ppa_price('battery', iso, level, ppa_level)
                ldes = get_learning_adjusted_ppa('ldes', iso, threshold, scenario, level, ppa_level)
                price = batt * 0.5 + ldes * 0.5
            else:
                price = get_learning_adjusted_ppa('uprate', iso, threshold, scenario, level, ppa_level)
                twh = min(twh, UPRATE_CAP_TWH.get(iso, 0))

            cost = twh * price
            total_cost += cost
            total_procured += twh

            key = f'new_build_{resource}'
            resource_breakdown[key] = {
                'twh': round(twh, 2),
                'price': round(price, 1),
                'cost_m': round(cost, 2),
                'category': 'new_build',
            }

    # SSS as zero-cost entry
    resource_breakdown['sss_allocation'] = {
        'twh': round(buyer_sss_share, 2),
        'price': 0.0,
        'cost_m': 0.0,
        'category': 'sss',
    }

    effective_procured = total_procured + buyer_sss_share
    cost_per_mwh = total_cost / effective_procured if effective_procured > 0 else 0
    emission_rate = get_emission_rate(iso, 'fossil_average')
    co2_abated = total_clean_needed * emission_rate / 1e3
    mac = (total_cost * 1e6) / (co2_abated * 1e6) if co2_abated > 0 else None

    return make_strategy_result(
        '2C', iso, year, threshold, participation_pct,
        cost_per_mwh, total_cost, co2_abated, mac,
        resource_mix=resource_breakdown,
        metadata={
            'sss_share_twh': round(buyer_sss_share, 2),
            'procurement_twh': round(procurement_twh, 2),
            'over_procurement_ratio': round(op_ratio, 3),
            'scenario': scenario,
            'use_45u': use_45u,
            'use_ctr': use_ctr,
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FULL SWEEP
# ═══════════════════════════════════════════════════════════════════════════════


STRATEGY_2_FUNCS = {
    '2A': compute_strategy_2a,
    '2B': compute_strategy_2b,
    '2C': compute_strategy_2c,
}


def run_full_sweep(isos=None, participation_levels=None, growth_levels=None):
    """Run Strategy 2 (all variants) for all ISOs × participation × growth.

    Returns dict:
    {
        'strategy2A': {iso: {growth: {part: [trajectory]}}},
        'strategy2B': {iso: {growth: {part: [trajectory]}}},
        'strategy2C': {iso: {growth: {part: [trajectory]}}},
    }
    """
    if isos is None:
        isos = ISOS
    if participation_levels is None:
        participation_levels = PARTICIPATION_LEVELS
    if growth_levels is None:
        growth_levels = ['Low', 'Medium', 'High']

    all_results = {}
    total = len(STRATEGY_2_FUNCS) * len(isos) * len(growth_levels) * len(participation_levels)
    count = 0

    for variant_id, compute_fn in STRATEGY_2_FUNCS.items():
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

                    # Build 25-year trajectory
                    trajectory = []
                    for year in TIMELINE_YEARS:
                        threshold = get_threshold_for_year(year)
                        result = compute_fn(
                            iso=iso, year=year, threshold=threshold,
                            participation_pct=part,
                            growth_level=growth,
                            scenario='B',  # Hourly always uses Scenario B
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
        description='Step 6.5 — Strategy 2: Hourly Matching (Same-ISO)'
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
    print("Step 6.5 — Strategy 2: Hourly Matching (Same-ISO)")
    print("=" * 70)
    print(f"ISOs: {isos}")
    print(f"Participation levels: {[f'{p*100:.0f}%' for p in participation]}")
    print(f"Growth levels: {growth_levels}")
    print(f"Variants: 2A (all new), 2B (grid baseline), 2C (SSS + tranches)")
    print(f"Scenario: B (accelerated learning curve)")
    print()

    results = run_full_sweep(
        isos=isos,
        participation_levels=participation,
        growth_levels=growth_levels,
    )

    save_results_json(results, 'strategy2_hourly.json')

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Strategy 2 at 10% participation, Medium growth")
    print("=" * 70)
    for variant in ['strategy2A', 'strategy2B', 'strategy2C']:
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
