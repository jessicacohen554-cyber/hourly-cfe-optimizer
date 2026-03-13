#!/usr/bin/env python3
"""Scenario B: GHG Protocol Hourly Matching — Four-Pool Incentive-Driven Grid Buildout.

Models the systemic incentive structure created by hourly Scope 2 accounting
with Standard Supply Service (SSS) as proposed by GHG Protocol revisions.

Four supply pools:
  Pool 1 (SSS): Policy-supported clean (nuclear ZECs, public hydro, RPS mandates).
                 Free to all ratepayers — embedded in utility rates.
  Pool 2 (Contracted): Nuclear locked via hyperscaler PPAs (Susquehanna→Amazon,
                        Clinton/Duane Arnold→Meta). Unavailable for voluntary market.
  Pool 3 (Existing Merchant): Merchant nuclear, existing solar/wind on grid.
                               Available via EAC procurement at $3-5/MWh premium.
  Pool 4 (New-Build): What hourly matching incentivizes above Pools 1-3.
                       Hourly constraint forces firm clean + storage for night/winter.
                       Learning curves (FOAK→NOAK, 2030-2040) apply here.

Algorithm:
  For each ISO × threshold (mapped to SBTi year):
  1. Compute SSS (Pool 1) TWh + 8760 hourly shape at that year
  2. Subtract contracted (Pool 2) — unavailable
  3. Compute merchant clean (Pool 3) = total existing - SSS - contracted
  4. Compute hourly coverage from Pools 1+3
  5. Compute residual hourly gap = target% × demand - available (per hour)
  6. Evaluate EF mixes: feasibility from cached physics, pool-aware cost metric
  7. Apply learning curves (2030-2040) to Pool 4 (new-build) costs
  8. Select cheapest mix by pool-aware incremental cost
  9. Floor ratchet — committed resources don't un-deploy

Usage:
  python step5_1_scenario_hourly.py             # Run all ISOs
  python step5_1_scenario_hourly.py --iso CAISO  # Run single ISO
  python step5_1_scenario_hourly.py --iso CAISO ERCOT  # Run multiple ISOs
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from scenario_common import (
    ISOS, RESOURCES, THRESHOLDS, SCENARIO_A, SCENARIO_B,
    HYDRO_CAP_TWH, WHOLESALE_PRICES, FUEL_ADJUSTMENTS, CCS_CAP_TWH,
    GRID_MIX_SHARES, BASE_DEMAND_TWH, SBTI_YEAR_MAP,
    PEAK_DEMAND_MW, EXISTING_GAS_CAPACITY_MW, EXISTING_GAS_FOM_KW_YR,
    NEW_CCGT_COST_KW_YR, PEAK_CAPACITY_CREDITS, GAS_AVAILABILITY_FACTOR,
    RESOURCE_ADEQUACY_MARGIN, EXISTING_NUCLEAR_GW, UPRATE_CAP_TWH,
    UPRATE_LCOE, NUCLEAR_NEWBUILD_LCOE, CCS_LCOE_45Q_ON,
    GEOTHERMAL_LCOE, GEO_CAP_TWH, LCOE_TABLES, TX_TABLES, LEVEL_NAME,
    DEMAND_GROWTH_RATES,
    get_demand_twh, get_demand_growth_factor,
    learning_fraction,
    parse_feasible_mixes,
    compute_mix_cost, get_tx,
    _mix_resource_twh, _get_excess_lcoe,
    _existing_match_pct, _build_existing_only_entry,
    _build_learning_overrides_b,
    build_augmented_result,
    parse_iso_args, save_scenario_results,
    # Vectorized batch operations
    _to_mix_array, _precompute_cost_params,
    batch_compute_total_costs, batch_effective_costs,
    batch_filter_floor, batch_floor_excess, precompute_excess_lcoes,
)
from dispatch_utils import (
    load_common_data, get_demand_profile, get_supply_profiles,
    build_supply_matrix, H,
)
from procurement_utils import (
    get_sss_twh, get_sss_hourly_shape, get_merchant_clean_twh,
    get_merchant_hourly_shape, get_existing_clean_twh,
    get_contracted_clean_twh, CONTRACTED_CLEAN_TWH,
    SSS_FIXED_FLEET_TWH, EXISTING_EAC_PRICE,
)


# ============================================================================
# PER-ISO OPTIMAL TARGET FROM MAC/DAC CROSSOVER
# ============================================================================

DEFAULT_TARGET_THRESHOLD = 95.0

def _load_optimal_targets():
    """Load per-ISO optimal target thresholds from MAC/DAC crossover analysis."""
    targets_path = Path(__file__).parent.parent / 'data' / 'step4-analysis' / 'optimal_targets.json'
    if not targets_path.exists():
        print(f"  WARNING: {targets_path} not found, using default {DEFAULT_TARGET_THRESHOLD}% for all ISOs")
        return {iso: DEFAULT_TARGET_THRESHOLD for iso in ISOS}

    with open(targets_path) as f:
        data = json.load(f)

    targets = {}
    for iso in ISOS:
        iso_data = data.get(iso, {})
        crossovers = iso_data.get('crossovers', {})
        central = crossovers.get('medium_grid__central_dac', {})
        raw_threshold = central.get('threshold')

        if raw_threshold is None:
            targets[iso] = DEFAULT_TARGET_THRESHOLD
            print(f"  {iso}: No MAC/DAC crossover, using default {DEFAULT_TARGET_THRESHOLD}%")
        else:
            nearest = min(THRESHOLDS, key=lambda t: abs(t - raw_threshold))
            targets[iso] = nearest
            print(f"  {iso}: MAC/DAC crossover at {raw_threshold}% -> target {nearest}%")

    return targets


# ============================================================================
# FOUR-POOL SUPPLY MODEL HELPERS
# ============================================================================

def _compute_pool_coverage(iso, year, demand_twh, demand_norm, gen_profiles,
                           growth_level='Medium'):
    """Compute 8760 hourly coverage from Pools 1 (SSS) + 3 (Merchant).

    Returns:
        dict with:
          sss_twh: Total SSS clean TWh at year
          contracted_twh: Locked corporate-contracted TWh
          merchant_twh: Available merchant clean TWh
          sss_8760_mwh: 8760 array of SSS MWh per hour
          merchant_8760_mwh: 8760 array of merchant clean MWh per hour
          available_8760_mwh: SSS + merchant combined
          demand_8760_mwh: Demand MWh per hour
          coverage_frac: Per-hour coverage fraction (capped at 1.0)
          avg_coverage_pct: Average hourly coverage %
          min_coverage_pct: Minimum hourly coverage (worst hour)
          max_coverage_pct: Maximum hourly coverage (best hour)
    """
    # Pool 1: SSS
    sss_twh = get_sss_twh(iso, year, growth_level)
    sss_shape = get_sss_hourly_shape(iso, gen_profiles)
    sss_8760_mwh = sss_shape * sss_twh * 1e6

    # Pool 2: Contracted (locked — excluded)
    contracted_twh = get_contracted_clean_twh(iso)

    # Pool 3: Merchant clean
    merchant_twh = get_merchant_clean_twh(iso, year, growth_level)
    merchant_shape = get_merchant_hourly_shape(iso, gen_profiles)
    merchant_8760_mwh = merchant_shape * merchant_twh * 1e6

    # Combined available (Pools 1+3, Pool 2 excluded)
    available_8760_mwh = sss_8760_mwh + merchant_8760_mwh

    # Demand 8760
    demand_8760_mwh = demand_norm * demand_twh * 1e6

    # Coverage fraction per hour (cap at 1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        coverage_frac = np.where(
            demand_8760_mwh > 0,
            np.minimum(available_8760_mwh / demand_8760_mwh, 1.0),
            1.0
        )

    avg_pct = float(np.mean(coverage_frac) * 100)
    min_pct = float(np.min(coverage_frac) * 100)
    max_pct = float(np.max(coverage_frac) * 100)

    return {
        'sss_twh': sss_twh,
        'contracted_twh': contracted_twh,
        'merchant_twh': merchant_twh,
        'sss_8760_mwh': sss_8760_mwh,
        'merchant_8760_mwh': merchant_8760_mwh,
        'available_8760_mwh': available_8760_mwh,
        'demand_8760_mwh': demand_8760_mwh,
        'coverage_frac': coverage_frac,
        'avg_coverage_pct': avg_pct,
        'min_coverage_pct': min_pct,
        'max_coverage_pct': max_pct,
    }


def _compute_residual_gap(demand_8760_mwh, available_8760_mwh, target_frac):
    """Compute the residual hourly gap that Pool 4 (new-build) must fill.

    Args:
        demand_8760_mwh: 8760 demand profile in MWh
        available_8760_mwh: SSS + merchant combined hourly MWh
        target_frac: Target hourly CFE fraction (e.g. 0.90 for 90%)

    Returns:
        dict with gap characterization metrics
    """
    # Gap at each hour: what target requires minus what's available
    target_8760_mwh = demand_8760_mwh * target_frac
    gap_8760_mwh = np.maximum(target_8760_mwh - available_8760_mwh, 0.0)

    residual_gap_twh = float(np.sum(gap_8760_mwh) / 1e6)
    gap_hours = int(np.sum(gap_8760_mwh > 0))

    # Characterize gap by time of day to identify firm/storage needs
    # Night hours (20:00-06:00): indices where solar is zero → firm/storage needed
    night_mask = np.zeros(H, dtype=bool)
    for h in range(H):
        hour_of_day = h % 24
        if hour_of_day >= 20 or hour_of_day < 6:
            night_mask[h] = True

    night_gap_mwh = gap_8760_mwh[night_mask]
    firm_demand_hours = int(np.sum(night_gap_mwh > 0))  # Hours needing firm/storage

    # Storage opportunity: hours where available > target (surplus for charging)
    surplus_8760_mwh = np.maximum(available_8760_mwh - target_8760_mwh, 0.0)
    storage_opportunity_hours = int(np.sum(surplus_8760_mwh > 0))
    surplus_twh = float(np.sum(surplus_8760_mwh) / 1e6)

    return {
        'gap_8760_mwh': gap_8760_mwh,
        'residual_gap_twh': residual_gap_twh,
        'gap_hours': gap_hours,
        'firm_demand_hours': firm_demand_hours,
        'storage_opportunity_hours': storage_opportunity_hours,
        'surplus_twh': surplus_twh,
    }


def _compute_pool_aware_cost(mix_total_cost, mix_resource_twh, demand_twh,
                              sss_twh, merchant_twh, eac_price=4.0):
    """Compute pool-aware incremental cost for a mix.

    Pool 1 (SSS): $0 incremental — embedded in utility rates.
    Pool 3 (Merchant): EAC premium price (typically $3-5/MWh).
    Pool 4 (New-Build): Full LCOE cost (captured in mix_total_cost).

    The mix_total_cost from the optimizer prices everything at LCOE.
    We adjust by crediting SSS resources at $0 and merchant at EAC.
    """
    if demand_twh <= 0:
        return mix_total_cost

    # Total clean TWh in this mix
    total_clean = sum(mix_resource_twh.get(r, 0) for r in
                      ['clean_firm', 'solar', 'wind', 'offshore_wind',
                       'hydro', 'ccs_ccgt'])

    if total_clean <= 0:
        return mix_total_cost

    # SSS covers up to sss_twh of the mix's clean_firm + hydro at $0
    # (SSS is nuclear+hydro, so it offsets the most expensive baseload resources)
    sss_covered = min(sss_twh, total_clean)

    # Merchant covers the next tranche at EAC premium
    remaining_after_sss = total_clean - sss_covered
    merchant_covered = min(merchant_twh, remaining_after_sss)

    # Pool 4 (new-build) = total_clean - sss_covered - merchant_covered
    # This portion pays full LCOE (already in mix_total_cost)

    # Cost adjustment: SSS portion is priced at wholesale (already accounted
    # for in the base cost). The key insight is that SSS resources reduce
    # the incremental $/MWh the buyer pays.
    # Merchant EAC cost replaces LCOE for existing merchant resources.
    # We compute the blended incremental cost.

    # The mix_total_cost is $/MWh of total demand. We adjust it:
    # 1. SSS portion: reduce cost by (LCOE - 0) * sss_twh / demand_twh
    #    But SSS covers nuclear+hydro which are wholesale-priced in the optimizer
    #    So the adjustment is smaller — SSS resources are already cheap.
    #    The key value of SSS is they're "free" (no procurement cost).
    # 2. Merchant EAC: flat EAC premium replaces LCOE for merchant portion

    # For simplicity and accuracy: the pool-aware cost is
    # (total_mix_cost * demand_twh - sss_savings - merchant_savings) / demand_twh

    # SSS savings: the optimizer prices SSS nuclear at wholesale cost.
    # Under pool model, SSS is $0 incremental, so savings = wholesale * sss_covered
    wholesale_avg = 32.0  # Rough average across ISOs
    sss_savings_per_mwh = wholesale_avg * sss_covered / demand_twh

    # Merchant savings: optimizer prices at LCOE, but merchant is just EAC premium
    # The savings = (avg_merchant_lcoe - eac_price) * merchant_covered / demand_twh
    # Approximate merchant LCOE from solar/wind at ~$35/MWh average
    avg_merchant_lcoe = 35.0
    merchant_savings_per_mwh = max(0, avg_merchant_lcoe - eac_price) * merchant_covered / demand_twh

    adjusted_cost = mix_total_cost - sss_savings_per_mwh - merchant_savings_per_mwh
    return max(0.0, adjusted_cost)


# ============================================================================
# CFE ZONE CLASSIFICATION
# ============================================================================

def _classify_zone(threshold):
    """Classify threshold into CFE curve zone for comparison framework."""
    if threshold <= 65:
        return 'early'
    elif threshold <= 90:
        return 'inflection'
    else:
        return 'last_mile'


# ============================================================================
# SCENARIO B: FOUR-POOL HOURLY MATCHING GRID BUILDOUT
# ============================================================================

def find_scenario_b_mixes(feasible_mixes, isos=None):
    """Scenario B: GHG Protocol hourly matching incentive-driven grid buildout.

    Models the systemic incentive structure created by hourly Scope 2 accounting
    with Standard Supply Service (SSS):

    1. SSS layer (Pool 1): Policy-driven clean (nuclear ZECs, public hydro, RPS
       mandates) grows per state trajectories. All consumers get this at $0.
    2. Contracted layer (Pool 2): Nuclear locked via hyperscaler PPAs — excluded.
    3. Merchant layer (Pool 3): Existing solar/wind/merchant nuclear available
       at EAC premium. Huge in ERCOT/SPP, small in PJM/NYISO.
    4. New-build layer (Pool 4): What hourly matching incentivizes above Pools 1-3.
       Hourly constraint forces firm clean + storage for night/winter coverage.

    At each threshold, finds the optimal resource mix to fill the temporal gap
    between Pools 1+3 hourly coverage and the hourly matching target.

    Args:
        feasible_mixes: Dict of {iso: {threshold_str: [mixes]}} from parse_feasible_mixes.
        isos: List of ISOs to process. Defaults to ISOS (all 7).
    """
    if isos is None:
        isos = ISOS

    results = {}

    # Load dispatch data for profiles and gas backup
    print("\n  Loading dispatch data for pool coverage calculation...")
    demand_data, gen_profiles, _, _ = load_common_data()

    # Load per-ISO optimal target thresholds from MAC/DAC crossover
    print("\n  Loading per-ISO optimal targets from MAC/DAC crossover analysis...")
    iso_targets = _load_optimal_targets()

    for iso in isos:
        iso_sens = dict(SCENARIO_B['toggles'])
        if iso != 'CAISO':
            iso_sens['geo'] = None

        base_demand = BASE_DEMAND_TWH[iso]
        existing = GRID_MIX_SHARES[iso]

        # Load dispatch profiles for this ISO
        iso_demand_norm, _ = get_demand_profile(iso, demand_data)
        iso_supply_profiles = get_supply_profiles(iso, gen_profiles)
        iso_supply_matrix = build_supply_matrix(iso_supply_profiles)

        # Per-ISO target threshold from MAC/DAC crossover
        target_t = iso_targets.get(iso, DEFAULT_TARGET_THRESHOLD)

        # Print pool overview for this ISO
        existing_clean = get_existing_clean_twh(iso)
        sss_2025 = get_sss_twh(iso, 2025)
        contracted = get_contracted_clean_twh(iso)
        merchant = get_merchant_clean_twh(iso, 2025)

        print(f"\n  {iso} Four-Pool Supply Model (2025 baseline):")
        print(f"    Total existing clean: {existing_clean:.1f} TWh")
        print(f"    Pool 1 (SSS):         {sss_2025:.1f} TWh")
        print(f"    Pool 2 (Contracted):  {contracted:.1f} TWh")
        print(f"    Pool 3 (Merchant):    {merchant:.1f} TWh")
        print(f"    Target threshold:     {target_t}%")

        # Setup for floor ratcheting
        existing_twh_b = {
            'clean_firm': existing.get('clean_firm', 0) / 100.0 * base_demand,
            'solar': existing.get('solar', 0) / 100.0 * base_demand,
            'wind': existing.get('wind', 0) / 100.0 * base_demand,
            'ccs_ccgt': existing.get('ccs_ccgt', 0) / 100.0 * base_demand,
            'hydro': existing.get('hydro', 0) / 100.0 * base_demand,
            'battery': 0, 'ldes': 0,
        }
        floor = dict(existing_twh_b)

        exist_match = _existing_match_pct(iso)
        iso_results = {}

        for t in THRESHOLDS:
            gf = get_demand_growth_factor(iso, t)
            demand_twh = base_demand * gf

            # Map threshold to SBTi year for SSS/demand growth calc
            year = SBTI_YEAR_MAP.get(t, 2025)
            if year is None:
                year = 2025

            if t <= exist_match:
                entry = _build_existing_only_entry(
                    iso, t, demand_twh, gf, existing_twh_b, iso_sens)
                # Add pool metadata even for existing-only entries
                pool = _compute_pool_coverage(iso, year, demand_twh,
                                              iso_demand_norm, gen_profiles)
                entry['pool1_sss_twh'] = round(pool['sss_twh'], 1)
                entry['pool2_contracted_twh'] = round(pool['contracted_twh'], 1)
                entry['pool3_merchant_twh'] = round(pool['merchant_twh'], 1)
                entry['pool4_newbuild_twh'] = 0.0
                entry['available_hourly_avg_pct'] = round(pool['avg_coverage_pct'], 1)
                entry['sss_hourly_min_coverage_pct'] = round(pool['min_coverage_pct'], 1)
                entry['residual_gap_twh'] = 0.0
                entry['firm_demand_hours'] = 0
                entry['zone'] = _classify_zone(t)
                iso_results[t] = entry
                floor = dict(existing_twh_b)
                continue

            t_str = str(int(t)) if t == int(t) else str(t)
            mixes = feasible_mixes.get(iso, {}).get(t_str, [])
            if not mixes:
                continue

            # ── Pool coverage at this threshold/year ──
            pool = _compute_pool_coverage(iso, year, demand_twh,
                                          iso_demand_norm, gen_profiles)
            target_frac = t / 100.0
            gap = _compute_residual_gap(pool['demand_8760_mwh'],
                                        pool['available_8760_mwh'],
                                        target_frac)

            # ── Learning-curve overrides ──
            frac = learning_fraction(t, scenario='B')
            overrides = _build_learning_overrides_b(iso, frac)

            # ── EAC price for merchant clean (Pool 3) ──
            eac_price = EXISTING_EAC_PRICE.get('Medium', 4.0)

            # ── Vectorized mix evaluation ──
            mixes_arr = _to_mix_array(mixes)

            # CCS cap filter: exclude mixes exceeding ISO's geologic storage cap
            ccs_cap = CCS_CAP_TWH.get(iso, 9999.0)
            if ccs_cap < 9999.0:
                ccs_twh = mixes_arr[:, 4] / 100.0 * demand_twh
                ccs_ok = ccs_twh <= ccs_cap + 0.01  # small tolerance
                if np.any(ccs_ok):
                    mixes_arr = mixes_arr[ccs_ok]

            # Match score ceiling filter: score <= threshold + 2
            scores = mixes_arr[:, 6]
            score_ceiling = min(t + 2.0, 100.0)
            score_ok = scores <= score_ceiling
            if np.any(score_ok):
                mixes_arr = mixes_arr[score_ok]

            params = _precompute_cost_params(iso_sens, iso, demand_twh,
                                             overrides, gf)
            costs = batch_compute_total_costs(mixes_arr, params)

            # Floor excess penalty (vectorized)
            excess_lcoes = precompute_excess_lcoes(iso_sens, iso, overrides)
            excess = batch_floor_excess(mixes_arr, floor, existing_twh_b,
                                        demand_twh, iso, excess_lcoes)

            # ── Pool-aware cost adjustment (vectorized) ──
            # Compute SSS and merchant savings for each mix
            # SSS covers baseload (clean_firm + hydro) at $0
            # Merchant covers existing VRE at EAC premium
            pool_sss_twh = pool['sss_twh']
            pool_merchant_twh = pool['merchant_twh']

            # Per-mix clean TWh columns: cf(0), sol(1), wnd(2), osw(3), ccs(4), hyd(5)
            cf_pct = mixes_arr[:, 0]
            sol_pct = mixes_arr[:, 1]
            wnd_pct = mixes_arr[:, 2]
            osw_pct = mixes_arr[:, 3]
            ccs_pct = mixes_arr[:, 4]
            hyd_pct = mixes_arr[:, 5]
            total_clean_pct = cf_pct + sol_pct + wnd_pct + osw_pct + ccs_pct + hyd_pct
            total_clean_twh = total_clean_pct / 100.0 * demand_twh

            # SSS covers nuclear+hydro first (cheapest/baseload)
            sss_covered = np.minimum(pool_sss_twh, total_clean_twh)
            remaining = total_clean_twh - sss_covered

            # Merchant covers next tranche
            merchant_covered = np.minimum(pool_merchant_twh, np.maximum(remaining, 0))

            # Pool 4 = remainder
            pool4_twh = np.maximum(total_clean_twh - sss_covered - merchant_covered, 0)

            # SSS savings: SSS resources are free (vs wholesale-priced in optimizer)
            wholesale = WHOLESALE_PRICES.get(iso, 32)
            sss_savings = wholesale * sss_covered / demand_twh

            # Merchant savings: EAC premium replaces LCOE
            avg_merchant_lcoe = 35.0  # Blended solar/wind LCOE
            merchant_savings = np.maximum(avg_merchant_lcoe - eac_price, 0) * merchant_covered / demand_twh

            # Pool-aware total cost
            pool_adjusted_costs = costs - sss_savings - merchant_savings
            pool_adjusted_costs = np.maximum(pool_adjusted_costs, 0)

            total_aug = pool_adjusted_costs + excess

            # ── Tier classification for firm/storage pacing ──
            # Under hourly matching, firm investment is driven by the gap analysis
            # rather than S-curve pacing. Use gap characterization to set firm targets.
            gap_firm_signal = gap['residual_gap_twh']

            # Firm TWh deployed
            firm_twh = (cf_pct + ccs_pct) / 100.0 * demand_twh
            ldes_twh = mixes_arr[:, 8] / 100.0 * demand_twh

            # At inflection/last-mile, prefer mixes with firm clean matching the gap
            if t >= 80 and gap_firm_signal > 0:
                # Boost mixes that have adequate firm for night/winter coverage
                # Firm demand ≈ firm_demand_hours / 8760 * gap_TWh
                min_firm_twh = gap['firm_demand_hours'] / 8760 * gap_firm_signal * 0.5
                firm_adequate = firm_twh >= min_firm_twh
                if np.any(firm_adequate):
                    # Penalty for insufficient firm
                    firm_penalty = np.where(firm_adequate, 0.0, 5.0)  # $/MWh penalty
                    total_aug = total_aug + firm_penalty

            # Select cheapest
            best_idx = int(np.argmin(total_aug))
            if total_aug[best_idx] == np.inf:
                continue

            sel_cost_aug = float(total_aug[best_idx])
            sel_excess = float(excess[best_idx])
            sel_mix = mixes_arr[best_idx].tolist()
            sel_pool4_twh = float(pool4_twh[best_idx])

            # Build full result dict only for the winner (scalar call)
            sel_result = compute_mix_cost(sel_mix, iso_sens, iso, demand_twh,
                                          overrides=overrides, growth_factor=gf,
                                          demand_norm=iso_demand_norm,
                                          supply_profiles=iso_supply_profiles,
                                          supply_matrix=iso_supply_matrix)
            sel_deployed = _mix_resource_twh(sel_mix, demand_twh, iso)

            # ── Build augmented result with floor ratchet ──
            extra = {
                'learning_fraction': round(frac, 3),
                'learning_nuclear_lcoe': round(overrides['nuclear_lcoe'], 1),
                'learning_ccs_lcoe': round(overrides['ccs_lcoe'], 1),
                'learning_ldes_lcoe': round(overrides['ldes_lcoe'], 1),
                'target_threshold': target_t,
                # Four-pool metadata
                'pool1_sss_twh': round(pool['sss_twh'], 1),
                'pool2_contracted_twh': round(pool['contracted_twh'], 1),
                'pool3_merchant_twh': round(pool['merchant_twh'], 1),
                'pool4_newbuild_twh': round(sel_pool4_twh, 1),
                'sss_hourly_min_coverage_pct': round(pool['min_coverage_pct'], 1),
                'sss_hourly_max_coverage_pct': round(pool['max_coverage_pct'], 1),
                'available_hourly_avg_pct': round(pool['avg_coverage_pct'], 1),
                'residual_gap_twh': round(gap['residual_gap_twh'], 1),
                'firm_demand_hours': gap['firm_demand_hours'],
                'storage_opportunity_hours': gap['storage_opportunity_hours'],
                'surplus_twh': round(gap['surplus_twh'], 1),
                'gap_hours': gap['gap_hours'],
                'zone': _classify_zone(t),
                # Pool-aware cost breakdown
                'pool_adjusted_cost': round(float(pool_adjusted_costs[best_idx]), 2),
                'sss_savings_per_mwh': round(float(sss_savings[best_idx]), 2),
                'merchant_savings_per_mwh': round(float(merchant_savings[best_idx]), 2),
                'eac_price': eac_price,
            }
            aug, augmented, excess_twh, total_excess = build_augmented_result(
                sel_result, sel_mix, floor, sel_deployed,
                sel_cost_aug, sel_excess,
                iso, demand_twh, gf, source='EF', extra_fields=extra)

            if total_excess > 1.0:
                zone = _classify_zone(t)
                print(f"  ↗ {iso} {t}% [B-{zone}]: {total_excess:.0f} TWh excess "
                      f"(+${sel_excess:.1f}/MWh)")

            iso_results[t] = aug

            # Floor ratchet: lock in deployed + augmented resources
            for res in floor:
                floor[res] = max(floor[res], augmented.get(res, 0))

        # Print trajectory summary with pool breakdown
        print(f"\n  {iso} Scenario B Trajectory (Four-Pool Model):")
        print(f"    {'Thr':>5s}  {'Zone':>10s}  {'SSS':>5s}  {'Mrch':>5s}  {'P4':>5s}  "
              f"{'Gap':>5s}  {'FirmH':>5s}  {'LF':>4s}  {'$/MWh':>6s}")
        for t in sorted(iso_results.keys()):
            r = iso_results[t]
            rt = r['resource_twh']
            zone = r.get('zone', '?')
            p1 = r.get('pool1_sss_twh', 0)
            p3 = r.get('pool3_merchant_twh', 0)
            p4 = r.get('pool4_newbuild_twh', 0)
            gap_twh = r.get('residual_gap_twh', 0)
            firm_hrs = r.get('firm_demand_hours', 0)
            lf = r.get('learning_fraction', 0)
            cost = r.get('effective_cost', 0)
            print(f"    {t:5.1f}%  {zone:>10s}  {p1:5.0f}  {p3:5.0f}  {p4:5.0f}  "
                  f"{gap_twh:5.0f}  {firm_hrs:5d}  {lf:.2f}  ${cost:5.0f}")

        results[iso] = iso_results

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("SCENARIO B: GHG PROTOCOL HOURLY MATCHING — FOUR-POOL GRID BUILDOUT")
    print("  Pool 1 (SSS): Policy-supported clean at $0")
    print("  Pool 2 (Contracted): Locked via hyperscaler PPAs")
    print("  Pool 3 (Merchant): Existing clean at EAC premium")
    print("  Pool 4 (New-Build): Hourly matching investment signal")
    print("  Learning curve: FOAK(H)→NOAK(L), 2030-2040")
    print("=" * 80)

    # Parse ISO arguments
    isos_to_run, _max_pfs_eval = parse_iso_args()
    print(f"\nISOs to process: {', '.join(isos_to_run)}")

    # Load feasible mixes from shared-data.js (physics-validated EF mixes)
    print("\nLoading feasible mixes from shared-data.js...")
    feasible_mixes = parse_feasible_mixes()

    # Filter to requested ISOs
    feasible_mixes = {iso: data for iso, data in feasible_mixes.items()
                      if iso in isos_to_run}

    total_mixes = sum(len(v) for iso_data in feasible_mixes.values()
                      for v in iso_data.values())
    print(f"  Loaded {total_mixes} feasible mixes across "
          f"{len(feasible_mixes)} ISOs")

    # Run Scenario B
    print("\nRunning Scenario B (GHG Protocol Four-Pool Hourly Matching)...")
    results_b = find_scenario_b_mixes(feasible_mixes, isos=isos_to_run)

    # Save results
    isos_processed = [iso for iso in isos_to_run if iso in results_b
                      and results_b[iso]]
    save_scenario_results(results_b, 'B', isos_processed)

    # Print summary
    print("\n" + "=" * 80)
    print("SCENARIO B SUMMARY — FOUR-POOL MODEL")
    print("=" * 80)
    for iso in isos_processed:
        iso_results = results_b[iso]
        thresholds_done = sorted(iso_results.keys())
        if not thresholds_done:
            print(f"  {iso}: No results")
            continue
        min_t = min(thresholds_done)
        max_t = max(thresholds_done)
        min_cost = iso_results[min_t].get('effective_cost', 0)
        max_cost = iso_results[max_t].get('effective_cost', 0)

        # Pool summary at highest threshold
        top = iso_results[max_t]
        p1 = top.get('pool1_sss_twh', 0)
        p3 = top.get('pool3_merchant_twh', 0)
        p4 = top.get('pool4_newbuild_twh', 0)
        gap = top.get('residual_gap_twh', 0)
        firm_hrs = top.get('firm_demand_hours', 0)

        print(f"  {iso}: {len(thresholds_done)} thresholds "
              f"({min_t}%-{max_t}%), "
              f"cost ${min_cost:.0f}-${max_cost:.0f}/MWh")
        print(f"    At {max_t}%: SSS={p1:.0f} Merchant={p3:.0f} "
              f"NewBuild={p4:.0f} TWh, Gap={gap:.0f} TWh, "
              f"FirmHours={firm_hrs}")

    print(f"\nScenario B complete: {len(isos_processed)} ISOs processed.")


if __name__ == '__main__':
    main()
