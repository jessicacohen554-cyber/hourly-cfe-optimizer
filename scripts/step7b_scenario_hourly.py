#!/usr/bin/env python3
"""Scenario B: Hourly Matching — forward-looking endpoint-optimal deployment with S-curve pacing.

Extracted from step6_scenario_comparison.py for standalone execution.

Strategy:
  1. Load the MAC/DAC crossover target per ISO from optimal_targets.json.
     The target threshold is the nearest analyzed threshold to the
     medium_grid__central_dac crossover (where medium DAC cost crosses
     the medium grid cost curve). Defaults to 95% if no crossover exists.
  2. Find the NOAK-optimal mix at the target threshold with all costs Low
     (CCS at Medium+45Q as NOAK endpoint — CCS learning curve is H→M, not H→L).
     This is the "north star" — what the grid looks like in 2045.
  3. At each threshold, deploy resources toward this target using an S-curve
     ramp that frontloads firm/storage investment.
  4. Nuclear/LDES: accelerated FOAK(H)→NOAK(L) learning curve.
     CCS: FOAK(H)→NOAK(M) learning curve with 45Q.
  5. Select from feasible mixes those best matching the deployment target
     at learning-curve-adjusted prices. Strict pacing enforcement.
  6. Floor ratchet prevents un-deploying committed resources.

Usage:
  python step7b_scenario_hourly.py             # Run all ISOs
  python step7b_scenario_hourly.py --iso CAISO  # Run single ISO
  python step7b_scenario_hourly.py --iso CAISO ERCOT  # Run multiple ISOs
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from scenario_common import (
    ISOS, RESOURCES, THRESHOLDS, SCENARIO_A, SCENARIO_B,
    HYDRO_CAP_TWH, WHOLESALE_PRICES, FUEL_ADJUSTMENTS,
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


# ============================================================================
# COST ADJUSTMENT WITH LEARNING CURVE
# ============================================================================

def _adjust_costs_with_learning(results, scenario):
    """Scenario B: Apply learning curve — FOAK starts 2029, NOAK by 2038.

    At each threshold, compute learning_fraction(scenario='B') → interpolate:
      Nuclear/LDES: H→L (FOAK→NOAK), CCS: H→M (FOAK→Medium NOAK) with 45Q.
    Step 3 costs were computed at full FOAK (firm='H', ldes='H', ccs='H').
    We compute the cost delta from FOAK to the learning-curve-adjusted price
    for each resource tranche and apply it.

    10-year learning period (2030-2040), concave ramp (exponent 0.6).
    Sources: INL SOAK data, NEA learning ranges, DOE Liftoff.
    """
    for iso in results:
        for t in results[iso]:
            r = results[iso][t]
            demand_twh = r['demand_twh']
            if demand_twh <= 0:
                continue

            frac = learning_fraction(t, scenario='B')
            overrides = _build_learning_overrides_b(iso, frac)

            total_delta = 0.0

            # 1. Uprate: step3 used H ($40), target is M ($25)
            uprate_twh = r.get('tranche_uprate_twh', 0)
            if uprate_twh > 0:
                uprate_delta = UPRATE_LCOE['M'] - UPRATE_LCOE['H']
                total_delta += uprate_delta * uprate_twh / demand_twh

            # 2. Nuclear newbuild: step3 used H, learning curve gives interpolated
            nuc_twh = r.get('tranche_nuclear_newbuild_twh', 0)
            if nuc_twh > 0:
                nuc_h = NUCLEAR_NEWBUILD_LCOE['H'][iso]
                nuc_delta = overrides['nuclear_lcoe'] - nuc_h
                total_delta += nuc_delta * nuc_twh / demand_twh

            # 3. CCS new-build
            # CCS TWh from mix minus existing
            existing_ccs_frac = GRID_MIX_SHARES[iso].get('ccs_ccgt', 0) / 100.0
            gf = demand_twh / BASE_DEMAND_TWH[iso]
            existing_ccs_twh = existing_ccs_frac * BASE_DEMAND_TWH[iso]
            # v5.0: procurement baked into resource percentages
            total_ccs_twh = r['resource_twh'].get('ccs_ccgt', 0)
            ccs_new_twh = max(0, total_ccs_twh - existing_ccs_twh)
            if ccs_new_twh > 0:
                ccs_h = CCS_LCOE_45Q_ON['H'][iso]
                ccs_delta = overrides['ccs_lcoe'] - ccs_h
                total_delta += ccs_delta * ccs_new_twh / demand_twh

            # 4. Geothermal (CAISO only)
            if iso == 'CAISO' and 'geo_lcoe' in overrides:
                geo_twh = r.get('tranche_geo_twh', 0)
                if geo_twh > 0:
                    geo_h = GEOTHERMAL_LCOE['H']
                    geo_delta = overrides['geo_lcoe'] - geo_h
                    total_delta += geo_delta * geo_twh / demand_twh

            # 5. LDES (all new-build)
            ldes_twh = r.get('ldes_twh', 0)
            if ldes_twh > 0:
                ldes_h = LCOE_TABLES['ldes']['High'][iso]
                ldes_delta = overrides['ldes_lcoe'] - ldes_h
                total_delta += ldes_delta * ldes_twh / demand_twh

            # Apply total delta
            if abs(total_delta) > 0.001:
                r['total_cost'] = round(r['total_cost'] + total_delta, 2)
                match_frac = r['match_score'] / 100
                r['effective_cost'] = round(
                    r['total_cost'] / match_frac if match_frac > 0 else 0, 2)
                r['incremental'] = round(
                    r['effective_cost'] - r.get('wholesale', WHOLESALE_PRICES[iso]), 2)
                # Update new-build cost tracking
                r['new_build_cost_total'] = r.get('new_build_cost_total', 0) + \
                    total_delta * demand_twh * 1e6

            # Store learning curve metadata
            r['learning_fraction'] = round(frac, 3)
            r['learning_nuclear_lcoe'] = round(overrides['nuclear_lcoe'], 1)
            r['learning_ccs_lcoe'] = round(overrides['ccs_lcoe'], 1)
            r['learning_ldes_lcoe'] = round(overrides['ldes_lcoe'], 1)

    return results


# ============================================================================
# PER-ISO OPTIMAL TARGET FROM MAC/DAC CROSSOVER
# ============================================================================

DEFAULT_TARGET_THRESHOLD = 95.0

def _load_optimal_targets():
    """Load per-ISO optimal target thresholds from MAC/DAC crossover analysis.

    Reads optimal_targets.json and extracts the medium_grid__central_dac
    crossover threshold for each ISO, then snaps to the nearest analyzed
    threshold in THRESHOLDS.

    Returns:
        dict[str, float]: {iso: target_threshold}
    """
    targets_path = Path(__file__).parent.parent / 'data' / 'step5-post-processing' / 'optimal_targets.json'
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
            # No crossover (grid always cheaper than DAC) — default to 95%
            targets[iso] = DEFAULT_TARGET_THRESHOLD
            print(f"  {iso}: No MAC/DAC crossover, using default {DEFAULT_TARGET_THRESHOLD}%")
        else:
            # Snap to nearest analyzed threshold
            nearest = min(THRESHOLDS, key=lambda t: abs(t - raw_threshold))
            targets[iso] = nearest
            print(f"  {iso}: MAC/DAC crossover at {raw_threshold}% -> target {nearest}%")

    return targets


# ============================================================================
# SCENARIO B: ENDPOINT-OPTIMAL DEPLOYMENT WITH S-CURVE PACING
# ============================================================================

def find_scenario_b_mixes(feasible_mixes, isos=None):
    """Scenario B: Forward-looking endpoint-optimal deployment.

    Fundamentally different from Scenario A's greedy sequential approach.

    Strategy:
      1. Load per-ISO optimal target threshold from MAC/DAC crossover analysis
         (medium_grid__central_dac from optimal_targets.json, snapped to
         nearest analyzed threshold). Defaults to 95% if no crossover.
      2. Find the NOAK-optimal mix at the target threshold with all costs Low
         (CCS at Medium+45Q as NOAK endpoint — CCS learning curve is H→M,
         not H→L). This is the "north star" — what the grid looks like in 2045.
      3. At each threshold, deploy resources toward this target using an S-curve
         ramp that frontloads firm/storage investment.
      4. Nuclear/LDES: accelerated FOAK(H)→NOAK(L) learning curve.
         CCS: FOAK(H)→NOAK(M) learning curve with 45Q.
      5. Select from feasible mixes those best matching the deployment target
         at learning-curve-adjusted prices. Strict pacing enforcement.
      6. Floor ratchet prevents un-deploying committed resources.

    This produces drastically different trajectories from Scenario A because:
      - A chases cheap carbon greedily → renewable-heavy early, FOAK cliff at 90%+
      - B works backward from optimal 2045 endpoint → balanced deployment from
        the start, with firm/CCS/storage allocated from early thresholds
      - CCS starts on a learning curve from H→M (not stuck at FOAK forever)
      - By 80-90%, B has driven nuclear/LDES toward NOAK via cumulative deployment
      - At 95%+, B benefits from NOAK firm and learned CCS; A faces cost cliff

    Args:
        feasible_mixes: Dict of {iso: {threshold_str: [mixes]}} from parse_feasible_mixes.
        isos: List of ISOs to process. Defaults to ISOS (all 7).
    """
    if isos is None:
        isos = ISOS

    results = {}

    # Load per-ISO optimal target thresholds from MAC/DAC crossover
    print("\n  Loading per-ISO optimal targets from MAC/DAC crossover analysis...")
    iso_targets = _load_optimal_targets()

    # NOAK target sensitivities: all at Low except CCS at Medium+45Q
    # (CCS NOAK endpoint is Medium, not Low — structural cost floor)
    NOAK_SENS = {
        'ren': 'L', 'firm': 'L', 'batt': 'L', 'ldes_lvl': 'L',
        'fuel': 'M', 'tx': 'M', 'ccs': 'M', 'q45': '1', 'geo': 'L',
    }

    for iso in isos:
        iso_sens = dict(SCENARIO_B['toggles'])
        if iso != 'CAISO':
            iso_sens['geo'] = None

        noak_sens = dict(NOAK_SENS)
        if iso != 'CAISO':
            noak_sens['geo'] = None

        base_demand = BASE_DEMAND_TWH[iso]
        existing = GRID_MIX_SHARES[iso]

        # Per-ISO target threshold from MAC/DAC crossover
        target_t = iso_targets.get(iso, DEFAULT_TARGET_THRESHOLD)
        target_t_str = str(int(target_t)) if target_t == int(target_t) else str(target_t)

        # ==================================================================
        # Step 1: Find NOAK-optimal mix at target threshold — the "north star"
        # All costs Low, CCS at Medium+45Q (its NOAK endpoint).
        # This is what the grid looks like in 2045 when learning investments
        # have paid off — the endpoint we're building toward.
        # ==================================================================
        gf_target = get_demand_growth_factor(iso, target_t)
        demand_target = base_demand * gf_target
        mixes_target = feasible_mixes.get(iso, {}).get(target_t_str, [])

        # Use full NOAK overrides for target finding
        noak_overrides = _build_learning_overrides_b(iso, 1.0)  # frac=1 = full NOAK

        # Vectorized: find cheapest effective_cost at target threshold
        if mixes_target:
            arr_target = _to_mix_array(mixes_target)
            params_target = _precompute_cost_params(noak_sens, iso, demand_target,
                                                     noak_overrides, gf_target)
            eff_target = batch_effective_costs(arr_target, params_target)
            best_idx_target = int(np.argmin(eff_target))
            best_cost_target = float(eff_target[best_idx_target])
            best_mix_target = arr_target[best_idx_target].tolist()
        else:
            best_mix_target = None

        if not best_mix_target:
            print(f"  WARNING {iso}: No feasible mixes at target {target_t}%, skipping")
            results[iso] = {}
            continue

        target_deployed = _mix_resource_twh(best_mix_target, demand_target, iso)

        # Target resource levels at NOAK target threshold
        target_cf_twh = target_deployed.get('clean_firm', 0)
        target_ccs_twh = target_deployed.get('ccs_ccgt', 0)
        target_firm_twh = target_cf_twh + target_ccs_twh
        target_ldes_twh = target_deployed.get('ldes', 0)
        target_batt_twh = (best_mix_target[6] + best_mix_target[7]) / 100.0 * demand_target
        target_sol_twh = target_deployed.get('solar', 0)
        target_wnd_twh = target_deployed.get('wind', 0)

        # Existing levels (2025 baseline)
        existing_cf_twh = existing.get('clean_firm', 0) / 100.0 * base_demand
        existing_ccs_twh = existing.get('ccs_ccgt', 0) / 100.0 * base_demand
        existing_firm_twh = existing_cf_twh + existing_ccs_twh
        existing_sol_twh = existing.get('solar', 0) / 100.0 * base_demand
        existing_wnd_twh = existing.get('wind', 0) / 100.0 * base_demand

        print(f"\n  {iso} {target_t}% NOAK target: firm={target_firm_twh:.0f} TWh "
              f"(CF={target_cf_twh:.0f}, CCS={target_ccs_twh:.0f}), "
              f"LDES={target_ldes_twh:.0f} TWh, Batt={target_batt_twh:.0f} TWh, "
              f"Sol={target_sol_twh:.0f}, Wnd={target_wnd_twh:.0f}, "
              f"existing firm={existing_firm_twh:.0f} TWh")

        # ==================================================================
        # Step 2: Backward-from-endpoint deployment with S-curve pacing
        # ==================================================================
        existing_twh_b = {
            'clean_firm': existing_cf_twh,
            'solar': existing_sol_twh,
            'wind': existing_wnd_twh,
            'ccs_ccgt': existing_ccs_twh,
            'hydro': existing.get('hydro', 0) / 100.0 * base_demand,
            'battery': 0, 'ldes': 0,
        }
        floor = dict(existing_twh_b)

        exist_match = _existing_match_pct(iso)
        iso_results = {}

        for t in THRESHOLDS:
            gf = get_demand_growth_factor(iso, t)
            demand_twh = base_demand * gf
            demand_mwh = demand_twh * 1e6

            if t <= exist_match:
                iso_results[t] = _build_existing_only_entry(
                    iso, t, demand_twh, gf, existing_twh_b, iso_sens)
                floor = dict(existing_twh_b)
                continue

            t_str = str(int(t)) if t == int(t) else str(t)
            mixes = feasible_mixes.get(iso, {}).get(t_str, [])
            if not mixes:
                continue

            # Learning-curve overrides: nuclear/LDES H→L, CCS H→M
            frac = learning_fraction(t, scenario='B')
            overrides = _build_learning_overrides_b(iso, frac)

            # S-curve deployment fraction toward target threshold.
            # Sigmoid frontloads firm investment vs linear:
            #   T=50%: ~5%, T=target: 100%
            if t <= 50:
                deploy_frac = 0.05
            elif t >= target_t:
                deploy_frac = 1.0
            else:
                x = (t - 50) / (target_t - 50)  # 0→1
                raw = 1.0 / (1.0 + math.exp(-6 * (x - 0.4)))
                f0 = 1.0 / (1.0 + math.exp(-6 * (0 - 0.4)))
                f1 = 1.0 / (1.0 + math.exp(-6 * (1 - 0.4)))
                deploy_frac = 0.05 + 0.95 * (raw - f0) / (f1 - f0)

            # Paced targets for firm, CCS, LDES at this threshold
            demand_ratio = demand_twh / demand_target if demand_target > 0 else 1.0
            paced_cf_twh = existing_cf_twh + deploy_frac * (
                target_cf_twh * demand_ratio - existing_cf_twh)
            paced_ccs_twh = existing_ccs_twh + deploy_frac * (
                target_ccs_twh * demand_ratio - existing_ccs_twh)
            paced_firm_twh = paced_cf_twh + paced_ccs_twh
            paced_ldes_twh = deploy_frac * target_ldes_twh * demand_ratio
            paced_batt_twh = deploy_frac * target_batt_twh * demand_ratio

            # ==================================================================
            # Step 3: Vectorized cost + tier categorization
            # Three-tier selection:
            #   Tier 1: Strict — firm >= 85% of pace, CCS >= 85%, LDES >= 50%
            #   Tier 2: Relaxed — firm >= 50% of pace
            #   Tier 3: Any feasible mix (fallback)
            # Within each tier, pick cheapest at learning-curve prices.
            # ==================================================================
            mixes_arr = _to_mix_array(mixes)
            params = _precompute_cost_params(iso_sens, iso, demand_twh,
                                             overrides, gf)
            costs = batch_compute_total_costs(mixes_arr, params)

            # Floor excess penalty (vectorized)
            excess_lcoes = precompute_excess_lcoes(iso_sens, iso, overrides)
            excess = batch_floor_excess(mixes_arr, floor, existing_twh_b,
                                        demand_twh, iso, excess_lcoes)
            total_aug = costs + excess

            # Compute deployed TWh for tier classification (vectorized)
            hydro_cap = HYDRO_CAP_TWH[iso]
            cf_twh = mixes_arr[:, 0] / 100.0 * demand_twh
            ccs_twh = mixes_arr[:, 3] / 100.0 * demand_twh
            firm_twh = cf_twh + ccs_twh
            ldes_twh = mixes_arr[:, 8] / 100.0 * demand_twh

            # Tier masks (vectorized boolean)
            firm_ok_strict = (firm_twh >= paced_firm_twh * 0.85) | (paced_firm_twh < 1)
            ccs_ok = (ccs_twh >= paced_ccs_twh * 0.85) | (paced_ccs_twh < 1)
            ldes_ok = (ldes_twh >= paced_ldes_twh * 0.50) | (paced_ldes_twh < 1)
            tier1_mask = firm_ok_strict & ccs_ok & ldes_ok

            firm_ok_relaxed = (firm_twh >= paced_firm_twh * 0.50) | (paced_firm_twh < 1)
            tier2_mask = firm_ok_relaxed

            # Select from best available tier
            if np.any(tier1_mask):
                tier_costs = np.where(tier1_mask, total_aug, np.inf)
                source_flag = ''
            elif np.any(tier2_mask):
                tier_costs = np.where(tier2_mask, total_aug, np.inf)
                source_flag = ' (relaxed pace)'
            else:
                tier_costs = total_aug
                source_flag = ' (no pace-compliant mix)'

            best_idx = int(np.argmin(tier_costs))
            if tier_costs[best_idx] == np.inf:
                continue

            sel_cost_aug = float(total_aug[best_idx])
            sel_excess = float(excess[best_idx])
            sel_mix = mixes_arr[best_idx].tolist()

            # Build full result dict only for the winner (scalar call)
            sel_result = compute_mix_cost(sel_mix, iso_sens, iso, demand_twh,
                                          overrides=overrides, growth_factor=gf)
            sel_deployed = _mix_resource_twh(sel_mix, demand_twh, iso)

            # Build augmented result with floor ratchet
            extra = {
                'learning_fraction': round(frac, 3),
                'learning_nuclear_lcoe': round(overrides['nuclear_lcoe'], 1),
                'learning_ccs_lcoe': round(overrides['ccs_lcoe'], 1),
                'learning_ldes_lcoe': round(overrides['ldes_lcoe'], 1),
                'paced_firm_target_twh': round(paced_firm_twh, 1),
                'paced_ccs_target_twh': round(paced_ccs_twh, 1),
                'deploy_fraction': round(deploy_frac, 3),
                'target_threshold': target_t,
            }
            aug, augmented, excess_twh, total_excess = build_augmented_result(
                sel_result, sel_mix, floor, sel_deployed,
                sel_cost_aug, sel_excess,
                iso, demand_twh, gf, source='EF', extra_fields=extra)

            if total_excess > 1.0:
                print(f"  ↗ {iso} {t}% [B]: {total_excess:.0f} TWh excess "
                      f"(+${sel_excess:.1f}/MWh){source_flag}")

            iso_results[t] = aug

            # Floor ratchet: lock in deployed + augmented resources
            for res in floor:
                floor[res] = max(floor[res], augmented.get(res, 0))

        # Print trajectory summary
        for t in sorted(iso_results.keys()):
            r = iso_results[t]
            rt = r['resource_twh']
            lf = r.get('learning_fraction', 0)
            paced = r.get('paced_firm_target_twh', 0)
            actual_firm = rt.get('clean_firm', 0) + rt.get('ccs_ccgt', 0)
            df = r.get('deploy_fraction', 0)
            print(f"    {t:5.1f}%: CF={rt['clean_firm']:7.0f} Sol={rt['solar']:6.0f} "
                  f"Wnd={rt['wind']:6.0f} CCS={rt.get('ccs_ccgt', 0):6.0f} "
                  f"Firm={actual_firm:6.0f}/{paced:5.0f} pace "
                  f"DF={df:.2f} LF={lf:.2f} ${r['effective_cost']:.0f}/MWh [B]")

        results[iso] = iso_results

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("SCENARIO B: HOURLY MATCHING — ENDPOINT-OPTIMAL DEPLOYMENT")
    print("  Forward-looking endpoint-optimal deployment with S-curve pacing")
    print("  Nuclear/LDES: FOAK(H)→NOAK(L) learning curve")
    print("  CCS: FOAK(H)→NOAK(M) learning curve with 45Q")
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
    print("\nRunning Scenario B (Hourly Matching): "
          "Endpoint-optimal, backward from per-ISO MAC/DAC crossover target...")
    results_b = find_scenario_b_mixes(feasible_mixes, isos=isos_to_run)

    # Save results
    isos_processed = [iso for iso in isos_to_run if iso in results_b
                      and results_b[iso]]
    save_scenario_results(results_b, 'B', isos_processed)

    # Print summary
    print("\n" + "=" * 80)
    print("SCENARIO B SUMMARY")
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
        print(f"  {iso}: {len(thresholds_done)} thresholds "
              f"({min_t}%-{max_t}%), "
              f"cost ${min_cost:.0f}-${max_cost:.0f}/MWh")

    print(f"\nScenario B complete: {len(isos_processed)} ISOs processed.")


if __name__ == '__main__':
    main()
