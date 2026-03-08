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
    build_procurement_tranches, build_newbuild_only_tranches,
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
    # All 17 active thresholds from pipeline_config
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
    99.5:  2.50,   # Last-mile: exponential capacity growth
    99.9:  2.80,   # Near-physical limit
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
    99.5:  {'solar': 0.14, 'wind': 0.14, 'firm': 0.36, 'storage': 0.31, 'uprate': 0.05},
    99.9:  {'solar': 0.13, 'wind': 0.13, 'firm': 0.37, 'storage': 0.32, 'uprate': 0.05},
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

    Uses same tranche architecture as 2C but with ONLY new-build tranches
    (nuclear uprate + new VRE + new firm). No SSS, no existing clean credit.
    Maximum additionality — strongest signal for new clean investment.
    """
    buyer_demand = get_buyer_demand_twh(iso, year, participation_pct, growth_level)
    if buyer_demand <= 0 or threshold <= 0:
        return make_strategy_result('2A', iso, year, threshold, participation_pct,
                                     0, 0, 0, 0)

    total_demand = get_demand_twh_at_year(iso, year, growth_level)
    target_fraction = min(threshold / 100.0, 1.0)

    # No free baseline — all procurement is new-build
    total_clean_needed = buyer_demand * target_fraction
    op_ratio = get_over_procurement_ratio(threshold)
    procurement_twh = total_clean_needed * op_ratio

    # Build new-build-only tranches and walk merit order
    tranches = build_newbuild_only_tranches(
        iso, threshold, year, scenario, level, ppa_level, growth_level,
    )

    total_cost = 0.0
    total_procured = 0.0
    resource_breakdown = {}

    for tranche in tranches:
        if total_procured >= procurement_twh:
            break

        remaining_need = procurement_twh - total_procured
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

    # Fallback if tranches exhausted
    if total_procured < procurement_twh:
        shortfall = procurement_twh - total_procured
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

    # CO2 accounting
    cost_per_mwh = total_cost / total_procured if total_procured > 0 else 0
    emission_rate = get_emission_rate(iso, 'fossil_average')
    baseline_co2_mt = buyer_demand * emission_rate
    co2_abated = total_clean_needed * emission_rate
    mac = (total_cost * 1e6) / (co2_abated * 1e6) if co2_abated > 0 else None
    co2_reduction_pct = (co2_abated / baseline_co2_mt * 100) if baseline_co2_mt > 0 else 0

    return make_strategy_result(
        '2A', iso, year, threshold, participation_pct,
        cost_per_mwh, total_cost, co2_abated, mac,
        resource_mix=resource_breakdown,
        metadata={
            'over_procurement_ratio': round(op_ratio, 3),
            'procurement_twh': round(procurement_twh, 2),
            'scenario': scenario,
            'baseline_co2_mt': round(baseline_co2_mt, 4),
            'co2_reduction_pct': round(co2_reduction_pct, 2),
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 2B: GRID BASELINE (EXISTING CLEAN FREE)
# ═══════════════════════════════════════════════════════════════════════════════


def compute_strategy_2b(iso, year, threshold, participation_pct,
                         growth_level='Medium', scenario='B',
                         level='Medium', ppa_level='Medium', **kwargs):
    """Strategy 2B: Grid mix clean baseline + new-build procurement.

    Uses same tranche architecture as 2C but with grid mix clean as the free
    baseline (instead of SSS). Existing clean on the grid counts as "already
    matched" — buyer only procures new-build above the grid mix baseline.
    Only new-build tranches available (no existing nuclear/VRE tranches,
    since those are already counted in grid mix).
    """
    buyer_demand = get_buyer_demand_twh(iso, year, participation_pct, growth_level)
    if buyer_demand <= 0 or threshold <= 0:
        return make_strategy_result('2B', iso, year, threshold, participation_pct,
                                     0, 0, 0, 0)

    total_demand = get_demand_twh_at_year(iso, year, growth_level)
    target_fraction = min(threshold / 100.0, 1.0)

    # Layer 1: Grid mix clean (free — buyer's pro-rata share of all existing clean)
    existing_clean = get_existing_clean_twh(iso)
    existing_clean_pct = existing_clean / total_demand if total_demand > 0 else 0
    buyer_grid_share = buyer_demand * existing_clean_pct

    # Remaining above grid mix baseline
    total_clean_needed = buyer_demand * target_fraction
    remaining = max(0, total_clean_needed - buyer_grid_share)

    # Apply over-procurement ratio to remaining
    op_ratio = get_over_procurement_ratio(threshold)
    procurement_twh = remaining * op_ratio

    if procurement_twh <= 0:
        return make_strategy_result('2B', iso, year, threshold, participation_pct,
                                     0, 0,
                                     total_clean_needed * get_emission_rate(iso, 'fossil_average'),
                                     0,
                                     metadata={'grid_mix_covers_target': True,
                                              'grid_mix_share_twh': round(buyer_grid_share, 2)})

    # Build new-build-only tranches and walk merit order
    tranches = build_newbuild_only_tranches(
        iso, threshold, year, scenario, level, ppa_level, growth_level,
    )

    total_cost = 0.0
    total_procured = 0.0
    resource_breakdown = {}

    for tranche in tranches:
        if total_procured >= procurement_twh:
            break

        remaining_need = procurement_twh - total_procured
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

    # Fallback if tranches exhausted
    if total_procured < procurement_twh:
        shortfall = procurement_twh - total_procured
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

    # Grid mix as zero-cost entry
    resource_breakdown['grid_mix_allocation'] = {
        'twh': round(buyer_grid_share, 2),
        'price': 0.0,
        'cost_m': 0.0,
        'category': 'grid_mix',
    }

    # CO2 accounting
    effective_procured = total_procured + buyer_grid_share
    cost_per_mwh = total_cost / effective_procured if effective_procured > 0 else 0
    emission_rate = get_emission_rate(iso, 'fossil_average')
    baseline_co2_mt = buyer_demand * emission_rate
    co2_abated = total_clean_needed * emission_rate
    mac = (total_cost * 1e6) / (co2_abated * 1e6) if co2_abated > 0 else None
    co2_reduction_pct = (co2_abated / baseline_co2_mt * 100) if baseline_co2_mt > 0 else 0

    return make_strategy_result(
        '2B', iso, year, threshold, participation_pct,
        cost_per_mwh, total_cost, co2_abated, mac,
        resource_mix=resource_breakdown,
        metadata={
            'grid_mix_share_twh': round(buyer_grid_share, 2),
            'existing_clean_pct': round(existing_clean_pct * 100, 1),
            'procurement_twh': round(procurement_twh, 2),
            'over_procurement_ratio': round(op_ratio, 3),
            'scenario': scenario,
            'baseline_co2_mt': round(baseline_co2_mt, 4),
            'co2_reduction_pct': round(co2_reduction_pct, 2),
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
                                     total_clean_needed * get_emission_rate(iso, 'fossil_average'),
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

    # CO2 accounting: (hourly MWh - hourly match) × fossil avg
    effective_procured = total_procured + buyer_sss_share
    cost_per_mwh = total_cost / effective_procured if effective_procured > 0 else 0
    emission_rate = get_emission_rate(iso, 'fossil_average')
    baseline_co2_mt = buyer_demand * emission_rate
    co2_abated = total_clean_needed * emission_rate
    mac = (total_cost * 1e6) / (co2_abated * 1e6) if co2_abated > 0 else None
    co2_reduction_pct = (co2_abated / baseline_co2_mt * 100) if baseline_co2_mt > 0 else 0

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
            'baseline_co2_mt': round(baseline_co2_mt, 4),
            'co2_reduction_pct': round(co2_reduction_pct, 2),
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


def _apply_participation_ratchet(result, floor_twh):
    """Enforce floor ratchet: each resource TWh >= floor from prior participation.

    If any resource in the result's resource_mix dropped below the floor,
    bump it up to the floor and add the extra cost. Returns updated result
    and new floor_twh dict.
    """
    resource_mix = result.get('resource_mix', {})
    if not resource_mix:
        return result, floor_twh

    new_floor = dict(floor_twh)
    extra_cost = 0.0
    extra_twh = 0.0

    for source, entry in resource_mix.items():
        if not isinstance(entry, dict):
            continue
        # Skip free allocations (SSS, grid_mix) — they scale with participation naturally
        category = entry.get('category', '')
        if category in ('sss', 'grid_mix'):
            continue

        twh = entry.get('twh', 0)
        floor_val = floor_twh.get(source, 0)

        if twh < floor_val and floor_val > 0:
            # Bump up to floor
            added = floor_val - twh
            price = entry.get('price', 0)
            added_cost = added * price
            entry['twh'] = round(floor_val, 2)
            entry['cost_m'] = round(entry.get('cost_m', 0) + added_cost, 2)
            extra_cost += added_cost
            extra_twh += added

        # Update floor
        new_floor[source] = max(new_floor.get(source, 0), entry.get('twh', 0))

    # Update result totals if ratchet added resources
    if extra_cost > 0:
        result['total_cost_m'] = round(result.get('total_cost_m', 0) + extra_cost, 2)
        total_twh = sum(
            e.get('twh', 0) for e in resource_mix.values()
            if isinstance(e, dict)
        )
        result['cost_per_mwh'] = round(
            result['total_cost_m'] / total_twh if total_twh > 0 else 0, 2)
        # Recompute MAC
        co2 = result.get('co2_abated_mmt', 0)
        if co2 > 0:
            result['mac_per_ton'] = round(
                (result['total_cost_m'] * 1e6) / (co2 * 1e6), 2)

    return result, new_floor


def run_full_sweep(isos=None, participation_levels=None, growth_levels=None):
    """Run Strategy 2 (all variants) for all ISOs × participation × growth.

    For 2A and 2B, applies a participation-level floor ratchet: resources
    deployed at lower participation levels are locked in at higher levels.
    Participation levels are processed in ascending order.

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

    # Sort participation levels ascending for ratchet
    sorted_parts = sorted(participation_levels)

    all_results = {}
    total = len(STRATEGY_2_FUNCS) * len(isos) * len(growth_levels) * len(sorted_parts)
    count = 0

    for variant_id, compute_fn in STRATEGY_2_FUNCS.items():
        strategy_key = f'strategy{variant_id}'
        all_results[strategy_key] = {}
        use_ratchet = variant_id in ('2A', '2B', '2C')

        for iso in isos:
            all_results[strategy_key][iso] = {}

            for growth in growth_levels:
                all_results[strategy_key][iso][growth] = {}

                # Per-year floor ratchet state (resets per ISO/growth)
                # Maps year → {source: floor_twh}
                year_floors = {}

                for part in sorted_parts:
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

                        # Apply participation ratchet for 2A/2B
                        if use_ratchet:
                            floor = year_floors.get(year, {})
                            result, new_floor = _apply_participation_ratchet(result, floor)
                            year_floors[year] = new_floor

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

    # ── Dispatch-based 8760 hourly match scoring ──
    # Post-processing: compute actual hourly match scores using EF storage params
    print("\nComputing 8760 dispatch-based hourly match scores...")
    try:
        import pandas as pd
        from dispatch_utils import (
            load_common_data, get_demand_profile, get_supply_profiles,
            reconstruct_hourly_dispatch, build_supply_matrix
        )
        common_data = load_common_data()
        demand_data, gen_profiles = common_data[0], common_data[1]

        # Load storage dispatch params from Track 2 NB efficient frontier
        track_parquet = os.path.join(os.path.dirname(SCRIPT_DIR), 'dashboard', 'track_scenarios.parquet')
        ef_storage = {}  # {(iso, threshold): {battery, battery8, ldes, h2, resource_mix}}
        if os.path.exists(track_parquet):
            tdf = pd.read_parquet(track_parquet)
            nb = tdf[tdf['track'] == 'newbuild']
            for _, row in nb.iterrows():
                key = (row['iso'], float(row['threshold']))
                ef_storage[key] = {
                    'battery_dispatch_pct': float(row.get('battery_dispatch_pct', 0) or 0),
                    'battery8_dispatch_pct': float(row.get('battery8_dispatch_pct', 0) or 0),
                    'ldes_dispatch_pct': float(row.get('ldes_dispatch_pct', 0) or 0),
                    'h2_dispatch_pct': float(row.get('h2_dispatch_pct', 0) or 0),
                    # EF resource mix (% of demand) for dispatch
                    'mix_clean_firm': float(row.get('mix_clean_firm', 0) or 0),
                    'mix_solar': float(row.get('mix_solar', 0) or 0),
                    'mix_wind': float(row.get('mix_wind', 0) or 0),
                    'mix_offshore_wind': float(row.get('mix_offshore_wind', 0) or 0),
                    'mix_ccs_ccgt': float(row.get('mix_ccs_ccgt', 0) or 0),
                    'mix_hydro': float(row.get('mix_hydro', 0) or 0),
                    'hourly_match_score': float(row.get('hourly_match_score', 0) or 0),
                }
            print(f"  Loaded {len(ef_storage)} EF storage entries from track_scenarios.parquet")
        else:
            print(f"  WARNING: {track_parquet} not found — running without storage")

        def _find_closest_ef(iso, threshold):
            """Find closest EF entry for an ISO/threshold combo."""
            exact = ef_storage.get((iso, threshold))
            if exact:
                return exact
            # Find closest threshold
            best_key, best_dist = None, float('inf')
            for k in ef_storage:
                if k[0] == iso:
                    dist = abs(k[1] - threshold)
                    if dist < best_dist:
                        best_dist = dist
                        best_key = k
            return ef_storage.get(best_key) if best_key else None

        # Map tranche source names → dispatch resource types
        TRANCHE_TO_DISPATCH = {
            'new_build_vre': 'solar',
            'new_build_firm': 'clean_firm',
            'nuclear_uprate': 'clean_firm',
            'existing_nuclear': 'clean_firm',
            'existing_vre_hydro': 'hydro',
            'new_build_solar': 'solar',
            'new_build_wind': 'wind',
            'new_build_uprate': 'clean_firm',
        }

        dispatch_count = 0
        for variant in ['strategy2A', 'strategy2B', 'strategy2C']:
            if variant not in results:
                continue
            for iso in isos:
                growth_data = results[variant].get(iso, {}).get('Medium', {})
                if not growth_data:
                    continue

                # Pre-build supply matrix for this ISO
                demand_norm, total_mwh = get_demand_profile(iso, demand_data)
                supply_profiles = get_supply_profiles(iso, gen_profiles)
                supply_matrix = build_supply_matrix(supply_profiles)

                for part_key, trajectory in growth_data.items():
                    if not isinstance(trajectory, list):
                        continue
                    for entry in trajectory:
                        rm = entry.get('resource_mix', {})
                        if not rm:
                            continue

                        threshold = entry.get('threshold', 0)
                        buyer_twh = entry.get('buyer_demand_twh', 0)
                        if buyer_twh <= 0 or threshold <= 0:
                            continue

                        # Get EF storage params for this ISO/threshold
                        ef = _find_closest_ef(iso, threshold)

                        # Use EF resource mix directly for dispatch (it's the physics-optimal mix)
                        # This gives the "what should the hourly score be" answer
                        if ef:
                            resource_pcts = {
                                'clean_firm': ef['mix_clean_firm'],
                                'solar': ef['mix_solar'],
                                'wind': ef['mix_wind'],
                                'offshore_wind': ef['mix_offshore_wind'],
                                'ccs_ccgt': ef['mix_ccs_ccgt'],
                                'hydro': ef['mix_hydro'],
                            }
                            batt_pct = ef['battery_dispatch_pct']
                            batt8_pct = ef['battery8_dispatch_pct']
                            ldes_pct = ef['ldes_dispatch_pct']
                            h2_pct = ef['h2_dispatch_pct']
                        else:
                            # Fallback: convert tranche TWh → resource_pcts
                            resource_pcts = {}
                            for source, info in rm.items():
                                if not isinstance(info, dict):
                                    continue
                                twh = info.get('twh', 0)
                                if twh <= 0:
                                    continue
                                cat = info.get('category', '')
                                if cat in ('grid_mix', 'sss'):
                                    continue
                                dispatch_type = TRANCHE_TO_DISPATCH.get(source, 'solar')
                                pct = (twh / buyer_twh) * 100
                                resource_pcts[dispatch_type] = resource_pcts.get(dispatch_type, 0) + pct
                            batt_pct = batt8_pct = ldes_pct = h2_pct = 0

                        if not any(v > 0 for v in resource_pcts.values()):
                            continue

                        # For 2B/2C: add grid mix / SSS contribution to resource_pcts
                        # Grid clean contributes to hourly matching in proportion to its share
                        meta = entry.get('metadata', {})
                        if variant in ('strategy2B', 'strategy2C'):
                            grid_twh = 0
                            for source, info in rm.items():
                                if isinstance(info, dict) and info.get('category') in ('grid_mix', 'sss'):
                                    grid_twh += info.get('twh', 0)
                            if grid_twh > 0:
                                # Add grid clean as a mix of the ISO's existing resources
                                gm = GRID_MIX_SHARES[iso]
                                total_existing_pct = sum(
                                    gm.get(r, 0) for r in ['clean_firm', 'solar', 'wind', 'hydro']
                                )
                                if total_existing_pct > 0:
                                    grid_pct_of_buyer = (grid_twh / buyer_twh) * 100
                                    for r in ['clean_firm', 'solar', 'wind', 'hydro']:
                                        share = gm.get(r, 0) / total_existing_pct
                                        resource_pcts[r] = resource_pcts.get(r, 0) + grid_pct_of_buyer * share

                        try:
                            result = reconstruct_hourly_dispatch(
                                demand_norm, supply_profiles, resource_pcts,
                                procurement_pct=100,
                                battery_dispatch_pct=batt_pct,
                                battery8_dispatch_pct=batt8_pct,
                                ldes_dispatch_pct=ldes_pct,
                                supply_matrix=supply_matrix,
                                h2_dispatch_pct=h2_pct,
                            )
                            demand_arr = np.array(demand_norm[:8760])
                            total_clean = result['total_clean']
                            matched = np.minimum(total_clean, demand_arr)
                            curtailed = result['curtailed']
                            fossil_displaced = result['fossil_displaced']

                            score = float(np.sum(matched) / np.sum(demand_arr) * 100)
                            curtail_pct = float(np.sum(curtailed) / np.sum(total_clean) * 100) if np.sum(total_clean) > 0 else 0
                            fossil_disp_twh = float(np.sum(fossil_displaced) / np.sum(demand_arr)) * buyer_twh

                            # Consequential CO₂: only new-build displaces fossil
                            # Grid mix / SSS were already running — no additional displacement
                            new_twh = 0
                            grid_twh_local = 0
                            for source, info in rm.items():
                                if not isinstance(info, dict):
                                    continue
                                twh_val = info.get('twh', 0)
                                cat = info.get('category', '')
                                if cat in ('grid_mix', 'sss'):
                                    grid_twh_local += twh_val
                                elif cat == 'new_build':
                                    new_twh += twh_val

                            total_procured = new_twh + grid_twh_local
                            new_frac = new_twh / total_procured if total_procured > 0 else 1.0

                            fossil_rate = get_emission_rate(iso, 'fossil_average')
                            consequential_co2 = fossil_disp_twh * new_frac * fossil_rate

                            md = entry.setdefault('metadata', {})
                            md['hourly_match_score'] = round(score, 2)
                            md['curtailment_pct'] = round(curtail_pct, 2)
                            md['fossil_displaced_twh'] = round(fossil_disp_twh, 4)
                            md['new_build_twh'] = round(new_twh, 2)
                            md['grid_baseline_twh'] = round(grid_twh_local, 2)
                            md['new_build_fraction'] = round(new_frac, 4)

                            # Overwrite paper CO₂ with consequential CO₂
                            entry['co2_abated_mmt'] = round(consequential_co2, 4)
                            md['paper_co2_mmt'] = round(buyer_twh * (threshold / 100) * fossil_rate, 4)

                            dispatch_count += 1
                        except Exception:
                            pass

        print(f"  Computed {dispatch_count} dispatch scores")

        # Re-save with dispatch scores
        save_results_json(results, 'strategy2_hourly.json')
    except Exception as e:
        print(f"  WARNING: Dispatch scoring failed: {e}")
        import traceback; traceback.print_exc()
        print("  (Continuing without dispatch scores)")

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
