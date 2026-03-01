#!/usr/bin/env python3
"""Scenario A: Pure Consequential — forward-stepping with static FOAK firm costs.

At each threshold, evaluates ALL feasible EF mixes under static cost assumptions
and picks the cheapest augmented effective cost. Floor ratchets.

With expensive firm (FOAK forever), the optimizer gravitates toward renewable-heavy
mixes at early thresholds. At late thresholds (90%+), firm becomes unavoidable
but stays at FOAK prices — creating the cost cliff, renewable overbuilding, and
asset stranding that characterizes the "chase cheap carbon" strategy.

Cost: Low renewables, Medium batteries/uprates/tx, High (static FOAK) firm/CCS/LDES/geo.
"""

import sys
from pathlib import Path

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
    MAX_PFS_EVAL,
    get_demand_twh, get_demand_growth_factor,
    learning_fraction,
    parse_feasible_mixes,
    compute_mix_cost, get_tx,
    _mix_resource_twh, _get_excess_lcoe,
    _filter_mixes_by_floor, _filter_pfs_by_floor_window,
    _load_pfs_mixes, _rank_and_cap_pfs,
    _existing_match_pct, _build_existing_only_entry,
    _build_learning_overrides,
    parse_iso_args, save_scenario_results,
)


# ============================================================================
# CORE FORWARD-STEPPING OPTIMIZER WITH FLOOR RATCHET
# ============================================================================

def _forward_step_optimization(feasible_mixes, sens, get_overrides_fn, label,
                               isos=None, max_pfs_eval=MAX_PFS_EVAL):
    """Core forward-stepping optimizer with per-resource TWh floor filter.

    Algorithm:
      1. At each threshold, convert prior-step deployed TWh into a per-resource
         floor percentage (floor_twh / next_demand_twh).
      2. Filter EF mixes to only those meeting ALL per-resource floors.
      3. Price filtered mixes under scenario cost assumptions.
      4. Pick the cheapest that achieves the threshold.
      5. If the floor eliminates all EF mixes, fall back to PFS:
         a. First pass: floor to floor+10% per resource
         b. Second pass: floor to floor+250% per resource (5% increments implicit in PFS grid)
         c. If a PFS mix goes under on a resource, carry that resource's cost forward.
      6. Update floor = max(prior_floor, deployed) for each resource.

    After each PFS fallback filter step, if the number of surviving mixes
    exceeds max_pfs_eval, _rank_and_cap_pfs is called to cap the evaluated
    set. This prevents the 8.8M-mix timeout that kills PJM.

    Args:
        feasible_mixes: from parse_feasible_mixes()
        sens: sensitivity toggle dict (defines base cost lookups)
        get_overrides_fn: fn(iso, threshold) -> dict of LCOE overrides
        label: 'A' or 'B' for logging
        isos: list of ISOs to process (defaults to ISOS)
        max_pfs_eval: max PFS mixes to evaluate per fallback step (default MAX_PFS_EVAL)

    Returns: dict[iso][threshold] -> result dict
    """
    if isos is None:
        isos = ISOS

    results = {}

    for iso in isos:
        iso_sens = dict(sens)
        if iso != 'CAISO':
            iso_sens['geo'] = None

        base_demand = BASE_DEMAND_TWH[iso]
        existing = GRID_MIX_SHARES[iso]

        # Existing resource levels in absolute TWh (2025 baseline) -- priced at $0
        existing_twh = {
            'clean_firm': existing.get('clean_firm', 0) / 100.0 * base_demand,
            'solar': existing.get('solar', 0) / 100.0 * base_demand,
            'wind': existing.get('wind', 0) / 100.0 * base_demand,
            'ccs_ccgt': existing.get('ccs_ccgt', 0) / 100.0 * base_demand,
            'hydro': existing.get('hydro', 0) / 100.0 * base_demand,
            'battery': 0, 'ldes': 0,
        }

        # Floor = 2025 existing clean resource levels in absolute TWh
        floor_twh = dict(existing_twh)

        iso_results = {}

        # Inject existing-only anchor entries for thresholds the 2025 fleet already meets.
        # This populates the pre-50% portion of Scenario A's trajectory with the correct
        # baseline (2025 existing clean mix, zero new build) and ensures the chart's
        # "Existing Clean" reference line aligns with the starting bars.
        exist_match = _existing_match_pct(iso)
        for t_pre in [t2 for t2 in THRESHOLDS if t2 < 50]:
            if t_pre <= exist_match:
                gf_pre = get_demand_growth_factor(iso, t_pre)
                demand_pre = base_demand * gf_pre
                iso_results[t_pre] = _build_existing_only_entry(
                    iso, t_pre, demand_pre, gf_pre, existing_twh, iso_sens)

        # Build threshold list starting at 50%
        scenario_thresholds = [t for t in THRESHOLDS if t >= 50]

        for t_idx, t in enumerate(scenario_thresholds):
            t_str = str(int(t)) if t == int(t) else str(t)

            gf = get_demand_growth_factor(iso, t)
            demand_twh = base_demand * gf
            demand_mwh = demand_twh * 1e6

            overrides = get_overrides_fn(iso, t)

            # --- Step 1: Filter EF mixes by per-resource floor ---
            ef_mixes = feasible_mixes.get(iso, {}).get(t_str, [])
            filtered = _filter_mixes_by_floor(ef_mixes, floor_twh, demand_twh, iso)
            source = 'EF'

            # --- Step 2: If no EF mixes survive, fall back to PFS ---
            if not filtered:
                pfs_mixes = _load_pfs_mixes(iso, t)
                if pfs_mixes:
                    # First pass: floor to floor+10%
                    filtered = _filter_pfs_by_floor_window(
                        pfs_mixes, floor_twh, demand_twh, iso, max_pct_above=10.0)
                    source = 'PFS+10'

                    # Cap if too many mixes survived
                    if len(filtered) > max_pfs_eval:
                        filtered = _rank_and_cap_pfs(
                            filtered, floor_twh, demand_twh, iso, max_pfs_eval)

                    if not filtered:
                        # Second pass: floor to floor+250%
                        filtered = _filter_pfs_by_floor_window(
                            pfs_mixes, floor_twh, demand_twh, iso, max_pct_above=250.0)
                        source = 'PFS+250'

                        # Cap if too many mixes survived
                        if len(filtered) > max_pfs_eval:
                            filtered = _rank_and_cap_pfs(
                                filtered, floor_twh, demand_twh, iso, max_pfs_eval)

                    if not filtered:
                        # Final fallback: take ALL PFS mixes that meet the floor
                        # as a minimum (no upper bound)
                        filtered = _filter_mixes_by_floor(
                            pfs_mixes, floor_twh, demand_twh, iso)
                        source = 'PFS-all'

                        # Cap if too many mixes survived
                        if len(filtered) > max_pfs_eval:
                            filtered = _rank_and_cap_pfs(
                                filtered, floor_twh, demand_twh, iso, max_pfs_eval)

                    if not filtered:
                        # Absolute last resort: relax floor, take any PFS mix,
                        # but carry under-floor resource costs forward
                        filtered = pfs_mixes
                        source = 'PFS-relaxed'

                        # Cap if too many mixes survived
                        if len(filtered) > max_pfs_eval:
                            filtered = _rank_and_cap_pfs(
                                filtered, floor_twh, demand_twh, iso, max_pfs_eval)

                if not filtered:
                    print(f"  WARNING {iso} {t}% [{label}]: No mixes found (EF or PFS)")
                    continue

            if source != 'EF':
                print(f"  -> {iso} {t}% [{label}]: EF empty after floor filter, "
                      f"using {source} ({len(filtered)} mixes)")

            # --- Step 3: Price filtered mixes, pick cheapest ---
            best_total_cost = float('inf')
            best_result = None
            best_mix = None
            best_excess_per_mwh = 0.0

            for mix in filtered:
                result = compute_mix_cost(mix, iso_sens, iso, demand_twh,
                                         overrides=overrides, growth_factor=gf)

                deployed = _mix_resource_twh(mix, demand_twh, iso)

                # If this mix goes UNDER on a resource vs floor (only possible
                # in PFS-relaxed fallback), price the gap as carried cost
                excess_per_mwh = 0.0
                for res in floor_twh:
                    shortfall = max(0, floor_twh.get(res, 0) - deployed.get(res, 0))
                    if shortfall > 0.01:
                        # Split into existing ($0) and committed-new (newbuild)
                        existing_gap = max(0, min(shortfall,
                                                  existing_twh.get(res, 0) - deployed.get(res, 0)))
                        newbuild_gap = shortfall - max(0, existing_gap)
                        if newbuild_gap > 0.01:
                            lcoe = _get_excess_lcoe(res, iso_sens, iso, overrides)
                            excess_per_mwh += newbuild_gap / demand_twh * lcoe

                total_with_excess = result['total_cost'] + excess_per_mwh

                if total_with_excess < best_total_cost:
                    best_total_cost = total_with_excess
                    best_result = result
                    best_mix = mix
                    best_excess_per_mwh = excess_per_mwh

            if not best_result or not best_mix:
                continue

            # --- Step 4: Build result with augmented resources ---
            deployed = _mix_resource_twh(best_mix, demand_twh, iso)
            augmented = {}
            excess_twh_dict = {}
            for res in floor_twh:
                ef_val = deployed.get(res, 0)
                floor_val = floor_twh.get(res, 0)
                augmented[res] = max(ef_val, floor_val)
                excess_twh_dict[res] = max(0, floor_val - ef_val)

            total_excess = sum(excess_twh_dict.values())

            # Build result dict
            aug = dict(best_result)
            aug['total_cost'] = round(best_total_cost, 2)
            match_frac = best_result['match_score'] / 100.0
            aug['effective_cost'] = round(
                aug['total_cost'] / match_frac if match_frac > 0 else 0, 2)
            aug['incremental'] = round(
                aug['effective_cost'] - best_result['wholesale'], 2)

            # Augmented resource TWh (locked-in floor + new EF deployment)
            aug['resource_twh'] = {res: augmented.get(res, 0) for res in RESOURCES}
            aug['battery_twh'] = augmented.get('battery', 0)
            aug['battery8_twh'] = 0  # Not tracked separately post-augmentation
            aug['ldes_twh'] = augmented.get('ldes', 0)
            aug['demand_twh'] = demand_twh
            aug['mix_raw'] = list(best_mix)
            aug['mix_source'] = source

            # Recompute gas backup from augmented clean capacity
            clean_peak_mw = 0
            for r, twh in aug['resource_twh'].items():
                pcc = PEAK_CAPACITY_CREDITS.get(r, 0)
                if pcc > 0 and twh > 0:
                    clean_peak_mw += (twh * 1e6 / 8760) * pcc
            clean_peak_mw += (augmented.get('battery', 0) * 1e6 / 8760) * \
                PEAK_CAPACITY_CREDITS.get('battery', 0.95)
            clean_peak_mw += (augmented.get('ldes', 0) * 1e6 / 8760) * \
                PEAK_CAPACITY_CREDITS.get('ldes', 0.90)

            ra_peak_mw = PEAK_DEMAND_MW[iso] * gf * (1 + RESOURCE_ADEQUACY_MARGIN)
            gaf = GAS_AVAILABILITY_FACTOR[iso]
            gas_needed_mw = max(0, ra_peak_mw - clean_peak_mw) / gaf
            aug['gas_backup_mw'] = round(gas_needed_mw)
            aug['new_gas_mw'] = round(max(0, gas_needed_mw - EXISTING_GAS_CAPACITY_MW[iso]))
            aug['clean_peak_mw'] = round(clean_peak_mw)
            aug['existing_gas_used_mw'] = round(min(gas_needed_mw, EXISTING_GAS_CAPACITY_MW[iso]))

            # New-build cost tracking (for MAC calculation)
            aug['new_build_cost_total'] = (
                best_result['new_build_cost_total'] +
                best_excess_per_mwh * demand_mwh)
            aug['new_gen_twh'] = round(
                best_result['new_gen_twh'] +
                sum(v for k, v in excess_twh_dict.items() if k != 'hydro'), 3)

            if total_excess > 1.0:
                print(f"  -> {iso} {t}% [{label}]: {total_excess:.0f} TWh floor excess "
                      f"(+${best_excess_per_mwh:.1f}/MWh) [{source}]")

            iso_results[t] = aug

            # --- Step 5: Update floor = max(prior, deployed) for each resource ---
            for res in floor_twh:
                floor_twh[res] = max(floor_twh[res], deployed.get(res, 0))

        # Print trajectory summary
        for t in sorted(iso_results.keys()):
            r = iso_results[t]
            rt = r['resource_twh']
            src = r.get('mix_source', '?')
            print(f"    {t:5.1f}%: CF={rt['clean_firm']:7.0f} Sol={rt['solar']:6.0f} "
                  f"Wnd={rt['wind']:6.0f} CCS={rt.get('ccs_ccgt', 0):6.0f} "
                  f"${r['effective_cost']:.0f}/MWh [{label}] ({src})")

        results[iso] = iso_results

    return results


# ============================================================================
# DELAYED LEARNING CURVE (Scenario A post-adjustment)
# ============================================================================

def _adjust_costs_no_learning(results, scenario):
    """Scenario A: Delayed learning curve -- FOAK until 2036, NOAK by 2048.

    Same interpolation machinery as Scenario B, but with delayed start (2036)
    and stretched timeline (12 years vs 10). Less deployment -> slower learning.

    Sources: INL SOAK data, NEA learning ranges, DOE Liftoff projections.
    In a lower-investment scenario, first SMRs don't deploy until ~2036,
    and fewer units per year means the learning curve takes 12 years to NOAK.
    """
    for iso in results:
        for t in results[iso]:
            r = results[iso][t]
            demand_twh = r['demand_twh']
            if demand_twh <= 0:
                continue

            frac = learning_fraction(t, scenario='A')
            overrides = _build_learning_overrides(iso, frac)

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
            existing_ccs_frac = GRID_MIX_SHARES[iso].get('ccs_ccgt', 0) / 100.0
            existing_ccs_twh = existing_ccs_frac * BASE_DEMAND_TWH[iso]
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

            # Store learning curve metadata
            r['learning_fraction'] = round(frac, 3)
            r['learning_nuclear_lcoe'] = round(overrides['nuclear_lcoe'], 1)
            r['learning_ccs_lcoe'] = round(overrides['ccs_lcoe'], 1)
            r['learning_ldes_lcoe'] = round(overrides['ldes_lcoe'], 1)
    return results


# ============================================================================
# SCENARIO A ENTRY POINT
# ============================================================================

def find_scenario_a_mixes(feasible_mixes, isos=None, max_pfs_eval=MAX_PFS_EVAL):
    """Scenario A: Pure Consequential -- forward-stepping with static FOAK firm costs.

    At each threshold, evaluates ALL feasible EF mixes under static cost assumptions
    and picks the cheapest augmented effective cost. Floor ratchets.

    With expensive firm (FOAK forever), the optimizer gravitates toward renewable-heavy
    mixes at early thresholds. At late thresholds (90%+), firm becomes unavoidable
    but stays at FOAK prices -- creating the cost cliff, renewable overbuilding, and
    asset stranding that characterizes the "chase cheap carbon" strategy.

    Cost: Low renewables, Medium batteries/uprates/tx, High (static FOAK) firm/CCS/LDES/geo.

    Args:
        feasible_mixes: from parse_feasible_mixes()
        isos: list of ISOs to process (defaults to all ISOS)
        max_pfs_eval: max PFS mixes to evaluate per fallback step (default MAX_PFS_EVAL)

    Returns: dict[iso][threshold] -> result dict
    """
    def get_overrides_a(iso, threshold):
        # Only override: uprate at Medium ($25/MWh) -- existing plants
        # Firm/CCS/LDES/geo stay at FOAK (High) at ALL thresholds -- no learning
        return {'uprate_lcoe': UPRATE_LCOE['M']}

    print(f"  Cost: ren=Low, batt=Med, tx=Med, uprate=Med, firm/CCS/LDES/geo=High (static FOAK)")
    return _forward_step_optimization(
        feasible_mixes, SCENARIO_A['toggles'], get_overrides_a, 'A',
        isos=isos, max_pfs_eval=max_pfs_eval)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run Scenario A (Pure Consequential) optimization."""
    isos_to_run, max_pfs_eval = parse_iso_args()

    print("=" * 80)
    print("SCENARIO A: PURE CONSEQUENTIAL")
    print("  Forward-stepping, FOAK firm costs (no learning), cheap $/tCO2 first")
    print(f"  ISOs: {', '.join(isos_to_run)}")
    print(f"  Max PFS eval: {max_pfs_eval}")
    print("=" * 80)

    # Load feasible mixes from shared-data.js (physics-validated EF mixes)
    print("\nLoading feasible mixes from shared-data.js...")
    feasible_mixes = parse_feasible_mixes()
    total_mixes = sum(len(v) for iso_data in feasible_mixes.values()
                      for v in iso_data.values())
    print(f"  Loaded {total_mixes} feasible mixes across {len(feasible_mixes)} ISOs")

    # Run Scenario A
    print("\nRunning Scenario A (Pure Consequential)...")
    results_a = find_scenario_a_mixes(
        feasible_mixes, isos=isos_to_run, max_pfs_eval=max_pfs_eval)

    # Save results
    save_scenario_results(results_a, 'A', isos_to_run)

    # Print summary
    print("\n" + "=" * 80)
    print("SCENARIO A SUMMARY")
    print("=" * 80)
    for iso in isos_to_run:
        iso_data = results_a.get(iso, {})
        if not iso_data:
            print(f"  {iso}: No results")
            continue
        thresholds = sorted(iso_data.keys())
        t_min, t_max = thresholds[0], thresholds[-1]
        cost_min = iso_data[t_min]['effective_cost']
        cost_max = iso_data[t_max]['effective_cost']
        print(f"  {iso}: {len(thresholds)} thresholds, "
              f"${cost_min:.0f}/MWh @ {t_min}% -> ${cost_max:.0f}/MWh @ {t_max}%")

    print("\nDone.")


if __name__ == '__main__':
    main()
