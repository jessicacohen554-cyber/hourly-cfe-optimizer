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
    get_existing_clean_twh, get_sss_twh, get_merchant_clean_twh,
    get_sss_hourly_shape, get_existing_clean_hourly_shape,
    build_procurement_tranches, build_newbuild_only_tranches,
    load_ef_resource_mix, get_resource_ppa_price,
    make_strategy_result, build_25yr_trajectory,
    save_results_json, compute_dispatch_hms,
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
    # Storage split into battery (daily cycling) and LDES (multi-day weather events).
    # LDES share grows at higher thresholds where multi-day coverage is essential.
    50:    {'solar': 0.40, 'wind': 0.35, 'firm': 0.10, 'battery': 0.08, 'ldes': 0.02, 'uprate': 0.05},
    70:    {'solar': 0.35, 'wind': 0.30, 'firm': 0.15, 'battery': 0.10, 'ldes': 0.05, 'uprate': 0.05},
    85:    {'solar': 0.25, 'wind': 0.25, 'firm': 0.25, 'battery': 0.12, 'ldes': 0.08, 'uprate': 0.05},
    90:    {'solar': 0.22, 'wind': 0.22, 'firm': 0.28, 'battery': 0.13, 'ldes': 0.10, 'uprate': 0.05},
    95:    {'solar': 0.18, 'wind': 0.18, 'firm': 0.32, 'battery': 0.14, 'ldes': 0.13, 'uprate': 0.05},
    99:    {'solar': 0.15, 'wind': 0.15, 'firm': 0.35, 'battery': 0.15, 'ldes': 0.15, 'uprate': 0.05},
    99.5:  {'solar': 0.14, 'wind': 0.14, 'firm': 0.36, 'battery': 0.15, 'ldes': 0.16, 'uprate': 0.05},
    99.9:  {'solar': 0.13, 'wind': 0.13, 'firm': 0.37, 'battery': 0.15, 'ldes': 0.17, 'uprate': 0.05},
    99.99: {'solar': 0.12, 'wind': 0.12, 'firm': 0.38, 'battery': 0.15, 'ldes': 0.18, 'uprate': 0.05},
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

    Uses EF-based physics template to determine resource mix (solar, wind,
    firm, storage) that actually achieves the target hourly matching score.
    No SSS, no existing clean credit — maximum additionality.
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

    # Use EF physics-optimized mix to determine resource allocation
    ef_mix = load_ef_resource_mix(iso, threshold)
    if not ef_mix:
        # Fallback to template if EF not available
        ef_mix = get_resource_mix_fractions(threshold)
    else:
        # EF data may have zero LDES/H2 even at high thresholds where they're
        # physically essential. Supplement missing storage from template.
        template = get_resource_mix_fractions(threshold)
        for stype in ('battery', 'ldes'):
            if stype not in ef_mix or ef_mix.get(stype, 0) <= 0:
                tmpl_val = template.get(stype, 0)
                if tmpl_val > 0:
                    # Scale template fraction to match EF total generation scale
                    gen_total = sum(v for k, v in ef_mix.items()
                                   if k not in ('battery', 'battery8', 'ldes', 'h2'))
                    ef_mix[stype] = tmpl_val * gen_total if gen_total > 0 else tmpl_val

    # Allocate procurement across resources using EF proportions
    total_pct = sum(ef_mix.values())
    if total_pct <= 0:
        total_pct = 1.0

    total_cost = 0.0
    total_procured = 0.0
    resource_breakdown = {}

    for resource, pct in ef_mix.items():
        if pct <= 0:
            continue
        frac = pct / total_pct
        twh = procurement_twh * frac

        # Cap uprate at available capacity
        if resource == 'nuclear_uprate':
            twh = min(twh, UPRATE_CAP_TWH.get(iso, 0) * (buyer_demand / total_demand))

        price = get_resource_ppa_price(resource, iso, threshold, year, scenario, level, ppa_level)
        cost = twh * price
        total_cost += cost
        total_procured += twh

        resource_breakdown[resource] = {
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
            'ef_based': True,
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 2B: GRID BASELINE (EXISTING CLEAN FREE)
# ═══════════════════════════════════════════════════════════════════════════════


def compute_strategy_2b(iso, year, threshold, participation_pct,
                         growth_level='Medium', scenario='B',
                         level='Medium', ppa_level='Medium', **kwargs):
    """Strategy 2B: Grid mix clean baseline + new-build procurement.

    Grid mix clean is free — buyer only procures new-build above grid baseline.
    Uses EF-based physics template for the new-build portion to ensure correct
    resource mix (firm + storage + VRE) for hourly matching.
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

    # Use EF physics-optimized mix for the new-build portion
    ef_mix = load_ef_resource_mix(iso, threshold)
    if not ef_mix:
        ef_mix = get_resource_mix_fractions(threshold)
    else:
        # Supplement missing storage from template at high thresholds
        template = get_resource_mix_fractions(threshold)
        for stype in ('battery', 'ldes'):
            if stype not in ef_mix or ef_mix.get(stype, 0) <= 0:
                tmpl_val = template.get(stype, 0)
                if tmpl_val > 0:
                    gen_total = sum(v for k, v in ef_mix.items()
                                   if k not in ('battery', 'battery8', 'ldes', 'h2'))
                    ef_mix[stype] = tmpl_val * gen_total if gen_total > 0 else tmpl_val

    total_pct = sum(ef_mix.values())
    if total_pct <= 0:
        total_pct = 1.0

    total_cost = 0.0
    total_procured = 0.0
    resource_breakdown = {}

    for resource, pct in ef_mix.items():
        if pct <= 0:
            continue
        frac = pct / total_pct
        twh = procurement_twh * frac

        if resource == 'nuclear_uprate':
            twh = min(twh, UPRATE_CAP_TWH.get(iso, 0) * (buyer_demand / total_demand))

        price = get_resource_ppa_price(resource, iso, threshold, year, scenario, level, ppa_level)
        cost = twh * price
        total_cost += cost
        total_procured += twh

        resource_breakdown[resource] = {
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

    # CO2 accounting — only new-build displaces fossil (grid baseline already running)
    effective_procured = total_procured + buyer_grid_share
    cost_per_mwh = total_cost / effective_procured if effective_procured > 0 else 0
    emission_rate = get_emission_rate(iso, 'fossil_average')
    baseline_co2_mt = buyer_demand * emission_rate
    new_build_twh = sum(info['twh'] for info in resource_breakdown.values()
                        if isinstance(info, dict) and info.get('category') == 'new_build')
    co2_abated = new_build_twh * emission_rate
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
                         use_45u=True, use_ctr=False, ctr_value=None,
                         nuclear_policy='stable', **kwargs):
    """Strategy 2C: SSS allocation + premium tranches for hourly matching.

    Layer 1: SSS allocation (free, follows SSS 8760 shape)
    Layer 2: Existing clean beyond SSS (at premium prices via tranches)
    Layer 3: New-build (at PPA prices)

    The tranche merit-order determines cost (§15.14.4).

    Args:
        nuclear_policy: 'stable' or 'rolloff' — controls ZEC/CMC expiration modeling.
    """
    buyer_demand = get_buyer_demand_twh(iso, year, participation_pct, growth_level)
    if buyer_demand <= 0 or threshold <= 0:
        return make_strategy_result('2C', iso, year, threshold, participation_pct,
                                     0, 0, 0, 0)

    total_demand = get_demand_twh_at_year(iso, year, growth_level)
    target_fraction = min(threshold / 100.0, 1.0)

    # Layer 1: SSS allocation (free)
    sss_twh = get_sss_twh(iso, year, growth_level, nuclear_policy=nuclear_policy)
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

    # 4-pool exhaustion model:
    # Pool 2: Non-SSS existing merchant clean (competitive — exhaust before new-build)
    # Pool 3: Nuclear uprate
    # Pool 4: New-build (EF-templated for hourly matching physics)

    total_cost = 0.0
    total_procured = 0.0
    resource_breakdown = {}
    remaining_need = procurement_twh

    # Pool 2: Existing merchant clean (competitive exhaustion)
    merchant_pool = get_merchant_clean_twh(iso, year, growth_level, nuclear_policy=nuclear_policy)
    buyer_share = (buyer_demand / total_demand) if total_demand > 0 else 0.01
    merchant_available = merchant_pool * buyer_share
    merchant_procured = min(remaining_need, merchant_available)

    if merchant_procured > 0:
        # Price at EAC market proxy
        eac_price = EXISTING_EAC_PRICE.get(level, 5.0)
        cost = merchant_procured * eac_price
        total_cost += cost
        total_procured += merchant_procured
        remaining_need -= merchant_procured

        resource_breakdown['existing_merchant'] = {
            'twh': round(merchant_procured, 2),
            'price': round(eac_price, 1),
            'cost_m': round(cost, 2),
            'category': 'existing',
        }

    # Pool 3: Nuclear uprate
    if remaining_need > 0:
        uprate_twh = UPRATE_CAP_TWH.get(iso, 0) * buyer_share
        uprate_procured = min(remaining_need, uprate_twh)
        if uprate_procured > 0:
            price = get_resource_ppa_price('nuclear_uprate', iso, threshold, year, scenario, level, ppa_level)
            cost = uprate_procured * price
            total_cost += cost
            total_procured += uprate_procured
            remaining_need -= uprate_procured

            resource_breakdown['nuclear_uprate'] = {
                'twh': round(uprate_procured, 2),
                'price': round(price, 1),
                'cost_m': round(cost, 2),
                'category': 'new_build',
            }

    # Pool 4: New-build (EF-templated for hourly matching)
    if remaining_need > 0:
        ef_mix = load_ef_resource_mix(iso, threshold)
        if not ef_mix:
            ef_mix = get_resource_mix_fractions(threshold)
        else:
            # Supplement missing storage from template at high thresholds
            template = get_resource_mix_fractions(threshold)
            for stype in ('battery', 'ldes'):
                if stype not in ef_mix or ef_mix.get(stype, 0) <= 0:
                    tmpl_val = template.get(stype, 0)
                    if tmpl_val > 0:
                        gen_total = sum(v for k, v in ef_mix.items()
                                       if k not in ('battery', 'battery8', 'ldes', 'h2'))
                        ef_mix[stype] = tmpl_val * gen_total if gen_total > 0 else tmpl_val

        total_pct = sum(ef_mix.values())
        if total_pct <= 0:
            total_pct = 1.0

        for resource, pct in ef_mix.items():
            if pct <= 0 or resource == 'nuclear_uprate':
                continue
            frac = pct / total_pct
            twh = remaining_need * frac

            price = get_resource_ppa_price(resource, iso, threshold, year, scenario, level, ppa_level)
            cost = twh * price
            total_cost += cost
            total_procured += twh

            resource_breakdown[resource] = {
                'twh': round(twh, 2),
                'price': round(price, 1),
                'cost_m': round(cost, 2),
                'category': 'new_build',
            }

    # SSS as zero-cost entry (Pool 1)
    resource_breakdown['sss_allocation'] = {
        'twh': round(buyer_sss_share, 2),
        'price': 0.0,
        'cost_m': 0.0,
        'category': 'sss',
    }

    # CO2 accounting — only new-build + uprate displaces fossil (SSS + existing already running)
    effective_procured = total_procured + buyer_sss_share
    cost_per_mwh = total_cost / effective_procured if effective_procured > 0 else 0
    emission_rate = get_emission_rate(iso, 'fossil_average')
    baseline_co2_mt = buyer_demand * emission_rate
    new_build_twh = sum(info['twh'] for info in resource_breakdown.values()
                        if isinstance(info, dict) and info.get('category') in ('new_build', 'uprate'))
    co2_abated = new_build_twh * emission_rate
    mac = (total_cost * 1e6) / (co2_abated * 1e6) if co2_abated > 0 else None
    co2_reduction_pct = (co2_abated / baseline_co2_mt * 100) if baseline_co2_mt > 0 else 0

    return make_strategy_result(
        '2C', iso, year, threshold, participation_pct,
        cost_per_mwh, total_cost, co2_abated, mac,
        resource_mix=resource_breakdown,
        metadata={
            'sss_share_twh': round(buyer_sss_share, 2),
            'merchant_clean_twh': round(merchant_procured, 2),
            'procurement_twh': round(procurement_twh, 2),
            'over_procurement_ratio': round(op_ratio, 3),
            'scenario': scenario,
            'use_45u': use_45u,
            'use_ctr': use_ctr,
            'baseline_co2_mt': round(baseline_co2_mt, 4),
            'co2_reduction_pct': round(co2_reduction_pct, 2),
            'ef_based': True,
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


def run_full_sweep(isos=None, participation_levels=None, growth_levels=None,
                   nuclear_policy='stable'):
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

    Args:
        nuclear_policy: 'stable' (default) or 'rolloff'. Passed to 2C/3D compute functions.
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
                            nuclear_policy=nuclear_policy,
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

    # ── Run stable policy (default) ──
    print("\n── Nuclear Policy: STABLE ──")
    results = run_full_sweep(
        isos=isos,
        participation_levels=participation,
        growth_levels=growth_levels,
        nuclear_policy='stable',
    )

    save_results_json(results, 'strategy2_hourly.json')

    # ── Dispatch-based 8760 hourly match scoring ──
    try:
        compute_dispatch_hms(results, ['strategy2A', 'strategy2B', 'strategy2C'], isos)
        # Re-save with dispatch scores
        save_results_json(results, 'strategy2_hourly.json')
    except Exception as e:
        print(f"  WARNING: Dispatch scoring failed: {e}")
        import traceback; traceback.print_exc()
        print("  (Continuing without dispatch scores)")

    # ── Run rolloff policy (ZEC/CMC expiration sensitivity) ──
    # Only 2C uses SSS, so only 2C results change. Run full sweep but only
    # keep the 2C variant under a separate key.
    print("\n── Nuclear Policy: ROLLOFF (ZEC/CMC expiration) ──")
    rolloff_results = run_full_sweep(
        isos=isos,
        participation_levels=participation,
        growth_levels=growth_levels,
        nuclear_policy='rolloff',
    )

    # Merge rolloff 2C into results under 'strategy2C_rolloff'
    results['strategy2C_rolloff'] = rolloff_results.get('strategy2C', {})

    # Re-save with rolloff variant included
    save_results_json(results, 'strategy2_hourly.json')

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Strategy 2 at 10% participation, Medium growth")
    print("=" * 70)
    for variant in ['strategy2A', 'strategy2B', 'strategy2C', 'strategy2C_rolloff']:
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
