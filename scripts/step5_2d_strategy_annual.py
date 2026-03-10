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

Output: data/step5-scenarios/strategy3_annual.json
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
    get_existing_clean_twh, get_sss_twh, get_merchant_clean_twh,
    build_newbuild_only_tranches, get_resource_ppa_price,
    make_strategy_result, build_25yr_trajectory,
    save_results_json, compute_dispatch_hms,
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
    """Strategy 3A: Annual matching, same-ISO, ALL NEW BUILD.

    No existing clean credit. Uses new-build-only tranches (uprate → VRE → firm)
    with annual volumetric matching (no hourly over-procurement penalty).
    """
    buyer_demand = get_buyer_demand_twh(iso, year, participation_pct, growth_level)
    if buyer_demand <= 0 or threshold <= 0:
        return make_strategy_result('3A', iso, year, threshold, participation_pct,
                                     0, 0, 0, 0)

    total_demand = get_demand_twh_at_year(iso, year, growth_level)
    target_fraction = min(threshold / 100.0, 1.0)

    # No free baseline — all procurement is new-build
    total_clean_needed = buyer_demand * target_fraction
    slack = get_annual_slack(threshold)
    procurement_twh = total_clean_needed * slack

    if procurement_twh <= 0:
        return make_strategy_result('3A', iso, year, threshold, participation_pct,
                                     0, 0, 0, 0)

    # New-build-only tranches (no existing nuclear/VRE)
    tranches = build_newbuild_only_tranches(
        iso, threshold, year, scenario, level, ppa_level, growth_level,
    )

    total_cost = 0.0
    total_procured = 0.0
    resource_breakdown = {}

    buyer_share = (buyer_demand / total_demand) if total_demand > 0 else 0.01
    for tranche in tranches:
        if total_procured >= procurement_twh:
            break
        remaining_need = procurement_twh - total_procured
        available = tranche['available_twh'] * buyer_share
        procure = min(remaining_need, available)

        cost = procure * tranche['price']
        total_cost += cost
        total_procured += procure

        resource_breakdown[tranche['source']] = {
            'twh': round(procure, 2),
            'price': round(tranche['price'], 1),
            'cost_m': round(cost, 2),
            'category': 'new_build',
        }

    # CO2 accounting — all new-build, all displaces fossil
    cost_per_mwh = total_cost / total_procured if total_procured > 0 else 0
    emission_rate = get_emission_rate(iso, 'fossil_average')
    baseline_co2_mt = buyer_demand * emission_rate
    co2_abated = total_procured * emission_rate  # all new-build → all displaces fossil
    mac = (total_cost * 1e6) / (co2_abated * 1e6) if co2_abated > 0 else None
    co2_reduction_pct = (co2_abated / baseline_co2_mt * 100) if baseline_co2_mt > 0 else 0

    return make_strategy_result(
        '3A', iso, year, threshold, participation_pct,
        cost_per_mwh, total_cost, co2_abated, mac,
        resource_mix=resource_breakdown,
        metadata={'annual_slack': round(slack, 3), 'scenario': scenario,
                  'procurement_twh': round(procurement_twh, 2),
                  'baseline_co2_mt': round(baseline_co2_mt, 4),
                  'co2_reduction_pct': round(co2_reduction_pct, 2)},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 3B: CROSS-REGIONAL, NEW BUILD REQUIRED
# ═══════════════════════════════════════════════════════════════════════════════


def compute_strategy_3b(iso, year, threshold, participation_pct,
                         growth_level='Medium', scenario='A',
                         level='Medium', ppa_level='Medium', **kwargs):
    """Strategy 3B: Annual matching, cross-regional, new build.

    Local cheap clean is consumed first (same-ISO resources at local prices),
    then buyers procure the cheapest $/MWh from any other ISO. Resources that
    are cheaply available locally stay in-ISO before going cross-regional.
    """
    from procurement_utils import get_demand_twh_at_year

    buyer_demand = get_buyer_demand_twh(iso, year, participation_pct, growth_level)
    if buyer_demand <= 0 or threshold <= 0:
        return make_strategy_result('3B', iso, year, threshold, participation_pct,
                                     0, 0, 0, 0)

    target_fraction = min(threshold / 100.0, 1.0)
    slack = get_annual_slack(threshold)
    clean_needed = buyer_demand * target_fraction * slack

    # --- Phase 1: Local (same-ISO) cheap resources consumed first ---
    local_sources = []
    total_demand = get_demand_twh_at_year(iso, year, growth_level)
    buyer_share = (buyer_demand / total_demand) if total_demand > 0 else 0.01

    solar_ppa = get_learning_adjusted_ppa('solar', iso, threshold, scenario, level, ppa_level)
    local_sources.append((iso, 'solar', solar_ppa, BASE_DEMAND_TWH[iso] * 0.30 * buyer_share))

    wind_ppa = get_learning_adjusted_ppa('wind', iso, threshold, scenario, level, ppa_level)
    local_sources.append((iso, 'wind', wind_ppa, BASE_DEMAND_TWH[iso] * 0.25 * buyer_share))

    uprate_twh = UPRATE_CAP_TWH.get(iso, 0)
    if uprate_twh > 0:
        uprate_ppa = get_learning_adjusted_ppa('uprate', iso, threshold, scenario, level, ppa_level)
        local_sources.append((iso, 'uprate', uprate_ppa, uprate_twh * buyer_share))

    if threshold >= 85:
        nuc = get_learning_adjusted_ppa('nuclear_newbuild', iso, threshold, scenario, level, ppa_level)
        local_sources.append((iso, 'firm', nuc, BASE_DEMAND_TWH[iso] * 0.10 * buyer_share))

    local_sources.sort(key=lambda s: s[2])

    total_cost = 0.0
    total_procured = 0.0
    resource_breakdown = {}

    # Consume local first
    for src_iso, resource, price, available in local_sources:
        if total_procured >= clean_needed:
            break
        remaining = clean_needed - total_procured
        procure = min(remaining, available)
        cost = procure * price
        total_cost += cost
        total_procured += procure
        key = f"{src_iso}_{resource}"
        resource_breakdown[key] = {'twh': round(procure, 2), 'price': round(price, 1), 'cost_m': round(cost, 2)}

    # --- Phase 2: Cross-ISO cheapest $/MWh for remainder ---
    if total_procured < clean_needed:
        cross_sources = []
        for src_iso in ISOS:
            if src_iso == iso:
                continue  # Already consumed local
            src_demand = get_demand_twh_at_year(src_iso, year, growth_level)
            src_buyer_share = buyer_share  # buyer's proportional claim on remote pool

            s_ppa = get_learning_adjusted_ppa('solar', src_iso, threshold, scenario, level, ppa_level)
            cross_sources.append((src_iso, 'solar', s_ppa, BASE_DEMAND_TWH[src_iso] * 0.15 * src_buyer_share))

            w_ppa = get_learning_adjusted_ppa('wind', src_iso, threshold, scenario, level, ppa_level)
            cross_sources.append((src_iso, 'wind', w_ppa, BASE_DEMAND_TWH[src_iso] * 0.12 * src_buyer_share))

            u_twh = UPRATE_CAP_TWH.get(src_iso, 0)
            if u_twh > 0:
                u_ppa = get_learning_adjusted_ppa('uprate', src_iso, threshold, scenario, level, ppa_level)
                cross_sources.append((src_iso, 'uprate', u_ppa, u_twh * src_buyer_share))

            if threshold >= 85:
                n_ppa = get_learning_adjusted_ppa('nuclear_newbuild', src_iso, threshold, scenario, level, ppa_level)
                cross_sources.append((src_iso, 'firm', n_ppa, BASE_DEMAND_TWH[src_iso] * 0.05 * src_buyer_share))

        cross_sources.sort(key=lambda s: s[2])

        for src_iso, resource, price, available in cross_sources:
            if total_procured >= clean_needed:
                break
            remaining = clean_needed - total_procured
            procure = min(remaining, available)
            cost = procure * price
            total_cost += cost
            total_procured += procure
            key = f"{src_iso}_{resource}"
            if key in resource_breakdown:
                resource_breakdown[key]['twh'] = round(resource_breakdown[key]['twh'] + procure, 2)
                resource_breakdown[key]['cost_m'] = round(resource_breakdown[key]['cost_m'] + cost, 2)
            else:
                resource_breakdown[key] = {'twh': round(procure, 2), 'price': round(price, 1), 'cost_m': round(cost, 2)}

    # CO2 accounting — all new-build, all displaces fossil
    cost_per_mwh = total_cost / total_procured if total_procured > 0 else 0
    emission_rate = get_emission_rate(iso, 'fossil_average')
    baseline_co2_mt = buyer_demand * emission_rate
    co2_abated = total_procured * emission_rate  # all new-build → all displaces fossil
    mac = (total_cost * 1e6) / (co2_abated * 1e6) if co2_abated > 0 else None
    co2_reduction_pct = (co2_abated / baseline_co2_mt * 100) if baseline_co2_mt > 0 else 0

    return make_strategy_result(
        '3B', iso, year, threshold, participation_pct,
        cost_per_mwh, total_cost, co2_abated, mac,
        resource_mix=resource_breakdown,
        metadata={'annual_slack': round(slack, 3), 'scenario': scenario,
                  'cross_regional': True,
                  'baseline_co2_mt': round(baseline_co2_mt, 4),
                  'co2_reduction_pct': round(co2_reduction_pct, 2)},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 3C: SAME-ISO, NO ADDITIONALITY
# ═══════════════════════════════════════════════════════════════════════════════


def compute_strategy_3c(iso, year, threshold, participation_pct,
                         growth_level='Medium', scenario='A',
                         level='Medium', ppa_level='Medium',
                         nuclear_policy='stable', **kwargs):
    """Strategy 3C: Annual matching, same-ISO, 4-pool model (same pools as 2C).

    Pool 1: SSS (free, pro-rata)
    Pool 2: Non-SSS existing merchant clean (competitive exhaustion)
    Pool 3: Nuclear uprate
    Pool 4: New-build (tranche walk for remaining)

    Args:
        nuclear_policy: 'stable' or 'rolloff' — controls ZEC/CMC expiration modeling.
    """
    buyer_demand = get_buyer_demand_twh(iso, year, participation_pct, growth_level)
    if buyer_demand <= 0 or threshold <= 0:
        return make_strategy_result('3C', iso, year, threshold, participation_pct,
                                     0, 0, 0, 0)

    total_demand = get_demand_twh_at_year(iso, year, growth_level)
    target_fraction = min(threshold / 100.0, 1.0)
    buyer_share = (buyer_demand / total_demand) if total_demand > 0 else 0.01

    # Pool 1: SSS (free)
    sss_twh = get_sss_twh(iso, year, growth_level, nuclear_policy=nuclear_policy)
    buyer_sss_share = sss_twh * buyer_share

    total_clean_needed = buyer_demand * target_fraction
    remaining_need = max(0, total_clean_needed - buyer_sss_share)
    slack = get_annual_slack(threshold)
    procurement_twh = remaining_need * slack

    if procurement_twh <= 0:
        return make_strategy_result('3C', iso, year, threshold, participation_pct,
                                     0, 0,
                                     total_clean_needed * get_emission_rate(iso, 'grid_average'),
                                     0,
                                     metadata={'sss_covers_target': True,
                                              'sss_share_twh': round(buyer_sss_share, 2)})

    total_cost = 0.0
    total_procured = 0.0
    resource_breakdown = {}
    left = procurement_twh

    # Pool 2: Non-SSS existing merchant clean (competitive exhaustion)
    # Merchant pool is first-come-first-served, NOT pro-rata like SSS.
    # At low participation, few buyers share the full pool → plenty of cheap clean.
    # At high participation, pool exhausts → new-build required → Wright's Law kicks in.
    merchant_pool = get_merchant_clean_twh(iso, year, growth_level, nuclear_policy=nuclear_policy)
    merchant_available = merchant_pool  # entire pool available to participating buyers
    merchant_procured = min(left, merchant_available)

    if merchant_procured > 0:
        eac_price = EXISTING_EAC_PRICE.get(level, 5.0)
        cost = merchant_procured * eac_price
        total_cost += cost
        total_procured += merchant_procured
        left -= merchant_procured
        resource_breakdown['existing_merchant'] = {
            'twh': round(merchant_procured, 2),
            'price': round(eac_price, 1),
            'cost_m': round(cost, 2),
            'category': 'existing',
        }

    # Pool 3: Nuclear uprate
    if left > 0:
        uprate_twh = UPRATE_CAP_TWH.get(iso, 0) * buyer_share
        uprate_procured = min(left, uprate_twh)
        if uprate_procured > 0:
            price = get_resource_ppa_price('nuclear_uprate', iso, threshold, year, scenario, level, ppa_level)
            cost = uprate_procured * price
            total_cost += cost
            total_procured += uprate_procured
            left -= uprate_procured
            resource_breakdown['nuclear_uprate'] = {
                'twh': round(uprate_procured, 2),
                'price': round(price, 1),
                'cost_m': round(cost, 2),
                'category': 'new_build',
            }

    # Pool 4: New-build (tranche walk — VRE + firm)
    if left > 0:
        tranches = build_newbuild_only_tranches(
            iso, threshold, year, scenario, level, ppa_level, growth_level,
        )
        for tranche in tranches:
            if total_procured >= procurement_twh:
                break
            remaining = procurement_twh - total_procured
            available = tranche['available_twh'] * buyer_share
            procure = min(remaining, available)
            cost = procure * tranche['price']
            total_cost += cost
            total_procured += procure

            resource_breakdown[tranche['source']] = {
                'twh': round(procure, 2),
                'price': round(tranche['price'], 1),
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
        '3C', iso, year, threshold, participation_pct,
        cost_per_mwh, total_cost, co2_abated, mac,
        resource_mix=resource_breakdown,
        metadata={'sss_share_twh': round(buyer_sss_share, 2),
                  'merchant_clean_twh': round(merchant_procured, 2),
                  'scenario': scenario,
                  'baseline_co2_mt': round(baseline_co2_mt, 4),
                  'co2_reduction_pct': round(co2_reduction_pct, 2)},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 3D: CROSS-REGIONAL, NO ADDITIONALITY (STATUS QUO)
# ═══════════════════════════════════════════════════════════════════════════════


def compute_strategy_3d(iso, year, threshold, participation_pct,
                         growth_level='Medium', scenario='A',
                         level='Medium', ppa_level='Medium',
                         nuclear_policy='stable', **kwargs):
    """Strategy 3D: Annual matching, cross-regional, 4-pool model.

    Same 4-pool structure as 2C/3C but cross-regional:
    Pool 1: SSS (same-ISO, free)
    Pool 2: Cross-ISO merchant clean (competitive exhaustion, cheapest first)
    Pool 3: Cross-ISO uprates
    Pool 4: Cross-ISO new-build (cheapest $/MWh across regions)

    Args:
        nuclear_policy: 'stable' or 'rolloff' — controls ZEC/CMC expiration modeling.
    """
    buyer_demand = get_buyer_demand_twh(iso, year, participation_pct, growth_level)
    if buyer_demand <= 0 or threshold <= 0:
        return make_strategy_result('3D', iso, year, threshold, participation_pct,
                                     0, 0, 0, 0)

    total_demand = get_demand_twh_at_year(iso, year, growth_level)
    target_fraction = min(threshold / 100.0, 1.0)
    buyer_share = (buyer_demand / total_demand) if total_demand > 0 else 0.01

    # Pool 1: SSS (same-ISO, free)
    sss_twh = get_sss_twh(iso, year, growth_level, nuclear_policy=nuclear_policy)
    buyer_sss_share = sss_twh * buyer_share

    total_clean_needed = buyer_demand * target_fraction
    remaining_need = max(0, total_clean_needed - buyer_sss_share)
    slack = get_annual_slack(threshold)
    procurement_twh = remaining_need * slack

    if procurement_twh <= 0:
        return make_strategy_result('3D', iso, year, threshold, participation_pct,
                                     0, 0,
                                     total_clean_needed * get_emission_rate(iso, 'grid_average'),
                                     0,
                                     metadata={'sss_covers_target': True,
                                              'sss_share_twh': round(buyer_sss_share, 2)})

    total_cost = 0.0
    total_procured = 0.0
    resource_breakdown = {}
    left = procurement_twh

    # Pool 2: Cross-ISO merchant clean (cheapest across all ISOs)
    merchant_sources = []
    for src_iso in ISOS:
        m_twh = get_merchant_clean_twh(src_iso, year, growth_level, nuclear_policy=nuclear_policy)
        src_demand = get_demand_twh_at_year(src_iso, year, growth_level)
        m_available = m_twh  # entire pool available (first-come-first-served)
        eac_price = EXISTING_EAC_PRICE.get(level, 5.0)
        if m_available > 0:
            merchant_sources.append((src_iso, m_available, eac_price))

    merchant_sources.sort(key=lambda s: s[2])  # cheapest first

    for src_iso, available, price in merchant_sources:
        if left <= 0:
            break
        procure = min(left, available)
        cost = procure * price
        total_cost += cost
        total_procured += procure
        left -= procure

        key = f'{src_iso}_existing_merchant'
        resource_breakdown[key] = {
            'twh': round(procure, 2),
            'price': round(price, 1),
            'cost_m': round(cost, 2),
            'category': 'existing',
        }

    # Pool 3: Cross-ISO uprates
    if left > 0:
        for src_iso in ISOS:
            if left <= 0:
                break
            uprate_twh = UPRATE_CAP_TWH.get(src_iso, 0) * buyer_share
            if uprate_twh <= 0:
                continue
            procure = min(left, uprate_twh)
            price = get_resource_ppa_price('nuclear_uprate', src_iso, threshold, year, scenario, level, ppa_level)
            cost = procure * price
            total_cost += cost
            total_procured += procure
            left -= procure

            key = f'{src_iso}_nuclear_uprate'
            resource_breakdown[key] = {
                'twh': round(procure, 2),
                'price': round(price, 1),
                'cost_m': round(cost, 2),
                'category': 'new_build',
            }

    # Pool 4: Cross-ISO new-build (cheapest $/MWh across regions)
    if left > 0:
        nb_sources = []
        for src_iso in ISOS:
            solar = get_learning_adjusted_ppa('solar', src_iso, threshold, scenario, level, ppa_level)
            wind = get_learning_adjusted_ppa('wind', src_iso, threshold, scenario, level, ppa_level)
            src_demand = BASE_DEMAND_TWH[src_iso]
            nb_sources.append((src_iso, 'solar', solar, src_demand * 0.30 * buyer_share))
            nb_sources.append((src_iso, 'wind', wind, src_demand * 0.25 * buyer_share))

            if threshold >= 85:
                firm_price = get_resource_ppa_price('clean_firm', src_iso, threshold, year, scenario, level, ppa_level)
                nb_sources.append((src_iso, 'clean_firm', firm_price, src_demand * 0.10 * buyer_share))

        nb_sources.sort(key=lambda s: s[2])

        for src_iso, resource, price, available in nb_sources:
            if left <= 0:
                break
            procure = min(left, available)
            cost = procure * price
            total_cost += cost
            total_procured += procure
            left -= procure

            key = f'{src_iso}_{resource}'
            if key in resource_breakdown:
                resource_breakdown[key]['twh'] = round(resource_breakdown[key]['twh'] + procure, 2)
                resource_breakdown[key]['cost_m'] = round(resource_breakdown[key]['cost_m'] + cost, 2)
            else:
                resource_breakdown[key] = {
                    'twh': round(procure, 2),
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
        '3D', iso, year, threshold, participation_pct,
        cost_per_mwh, total_cost, co2_abated, mac,
        resource_mix=resource_breakdown,
        metadata={
            'sss_share_twh': round(buyer_sss_share, 2),
            'cross_regional': True,
            'scenario': scenario,
            'baseline_co2_mt': round(baseline_co2_mt, 4),
            'co2_reduction_pct': round(co2_reduction_pct, 2),
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


def run_full_sweep(isos=None, participation_levels=None, growth_levels=None,
                   nuclear_policy='stable'):
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
                            nuclear_policy=nuclear_policy,
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

    # ── Run stable policy (default) ──
    print("\n── Nuclear Policy: STABLE ──")
    results = run_full_sweep(
        isos=isos,
        participation_levels=participation,
        growth_levels=growth_levels,
        nuclear_policy='stable',
    )

    save_results_json(results, 'strategy3_annual.json')

    # ── Dispatch-based 8760 hourly match scoring ──
    try:
        compute_dispatch_hms(results, ['strategy3A', 'strategy3B', 'strategy3C', 'strategy3D'], isos)
        # Re-save with dispatch scores
        save_results_json(results, 'strategy3_annual.json')
    except Exception as e:
        print(f"  WARNING: Dispatch scoring failed: {e}")
        import traceback; traceback.print_exc()
        print("  (Continuing without dispatch scores)")

    # ── Run rolloff policy (ZEC/CMC expiration sensitivity) ──
    # Only 3C/3D use SSS, so only those results change.
    print("\n── Nuclear Policy: ROLLOFF (ZEC/CMC expiration) ──")
    rolloff_results = run_full_sweep(
        isos=isos,
        participation_levels=participation,
        growth_levels=growth_levels,
        nuclear_policy='rolloff',
    )

    # Merge rolloff variants into results
    results['strategy3C_rolloff'] = rolloff_results.get('strategy3C', {})
    results['strategy3D_rolloff'] = rolloff_results.get('strategy3D', {})

    # Re-save with rolloff variants included
    save_results_json(results, 'strategy3_annual.json')

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Strategy 3 at 10% participation, Medium growth")
    print("=" * 70)
    for variant in ['strategy3A', 'strategy3B', 'strategy3C', 'strategy3C_rolloff',
                     'strategy3D', 'strategy3D_rolloff']:
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
