#!/usr/bin/env python3
"""
Phase 3 Refinement Script — Fine-grained search around cached optimizer results
================================================================================

Runs ONLY the Phase 3 fine-refinement step of the 3-phase optimizer, searching
a ±N% neighborhood around each cached winning resource mix. Designed for use
after correcting data (e.g., DST/multi-year profile fixes) without re-running
the full 3-hour optimizer.

For each ISO × threshold × scenario in the cached results:
  1. Extract the winning resource mix and procurement level
  2. Search ±5% (configurable) around each resource dimension at 1% steps
  3. Search ±4% around procurement at 2% steps
  4. Re-optimize storage dispatch for each improved candidate
  5. Keep the better result (cached or newly found)

Expected runtime: 10-20 minutes (vs ~3 hours for full optimizer)

Usage:
  python optimize_phase3_only.py
  python optimize_phase3_only.py --neighborhood 3 --workers 5
  python optimize_phase3_only.py --results-file path/to/cache.json
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from multiprocessing import Pool

# Import constants and functions from the main optimizer
from optimize_overprocure import (
    RESOURCE_TYPES, THRESHOLDS, ISOS, H, ISO_LABELS,
    HYDRO_CAPS, PROCUREMENT_BOUNDS,
    WHOLESALE_PRICES, REGIONAL_LCOE, GRID_MIX_SHARES,
    TRANSMISSION_ADDERS, CCS_RESIDUAL_EMISSION_RATE,
    FULL_LCOE_TABLES, FULL_TRANSMISSION_TABLES, FUEL_PRICES,
    WHOLESALE_FUEL_ADJUSTMENTS, PAIRED_TOGGLE_GROUPS,
    BATTERY_EFFICIENCY, BATTERY_DURATION_HOURS,
    LDES_EFFICIENCY, LDES_DURATION_HOURS, LDES_WINDOW_DAYS,
    DATA_DIR, DATA_YEAR,
    load_data, get_supply_profiles, find_anomaly_hours,
    prepare_numpy_profiles,
    fast_hourly_score, fast_score_with_battery,
    fast_score_with_both_storage, find_optimal_storage,
    compute_costs, compute_costs_parameterized,
    compute_co2_abatement, compute_peak_gap, compute_compressed_day,
    generate_combinations_around,
    generate_all_cost_scenarios, ALL_COST_SCENARIOS, COST_SCENARIO_MAP,
)

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS FORMAT DETECTION AND NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def detect_results_format(results_data):
    """
    Detect whether results are in old format (single result per threshold)
    or new format (324 scenarios per threshold).

    Old format: thresholds.{T} = {procurement_pct, resource_mix, storage_dispatch_pct, ...}
    New format: thresholds.{T} = {scenarios: {scenario_key: {procurement_pct, resource_mix, ...}}, ...}
    """
    for iso in ISOS:
        if iso not in results_data:
            continue
        iso_data = results_data[iso]
        if 'thresholds' not in iso_data:
            continue
        for t_str, t_data in iso_data['thresholds'].items():
            if 'scenarios' in t_data:
                return 'new'
            elif 'resource_mix' in t_data:
                return 'old'
    return 'unknown'


def normalize_result(result, fmt='old'):
    """
    Normalize a single result dict to ensure it has all required fields.
    Handles old format (storage_dispatch_pct) and new format (battery_dispatch_pct + ldes_dispatch_pct).
    """
    normalized = dict(result)

    # Ensure resource_mix has all 5 resource types
    mix = normalized.get('resource_mix', {})
    for rt in RESOURCE_TYPES:
        if rt not in mix:
            mix[rt] = 0
    normalized['resource_mix'] = mix

    # Normalize storage fields
    if 'battery_dispatch_pct' not in normalized:
        # Old format: single storage_dispatch_pct maps to battery only
        sdp = normalized.get('storage_dispatch_pct', 0)
        normalized['battery_dispatch_pct'] = sdp
        normalized['ldes_dispatch_pct'] = 0
    if 'ldes_dispatch_pct' not in normalized:
        normalized['ldes_dispatch_pct'] = 0

    return normalized


def iterate_results(results_data, fmt):
    """
    Iterate over all (iso, threshold, scenario_key, result) tuples in the results.
    For old format, scenario_key is 'MMM_M_M' (Medium everything).
    For new format, iterates over all scenarios.

    Yields: (iso, threshold_str, scenario_key, result_dict)
    """
    for iso in ISOS:
        if iso not in results_data:
            continue
        iso_data = results_data[iso]
        if 'thresholds' not in iso_data:
            continue

        for t_str, t_data in iso_data['thresholds'].items():
            if fmt == 'new' and 'scenarios' in t_data:
                for scenario_key, result in t_data['scenarios'].items():
                    yield iso, t_str, scenario_key, normalize_result(result, fmt)
            elif fmt == 'old' and 'resource_mix' in t_data:
                yield iso, t_str, 'MMM_M_M', normalize_result(t_data, fmt)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 REFINEMENT FOR A SINGLE ISO
# ══════════════════════════════════════════════════════════════════════════════

def refine_single_result(demand_arr, supply_matrix, hydro_cap,
                         result, threshold, neighborhood,
                         iso, cost_levels, demand_norm, supply_profiles):
    """
    Run Phase 3 fine-grained refinement around a single cached result.

    Searches a ±neighborhood% area around each resource dimension at 1% steps
    and ±4% around procurement at 2% steps. Re-optimizes storage for improved
    candidates.

    Returns: (improved_result, improved_flag)
    """
    target = threshold / 100.0
    mix = result['resource_mix']
    proc = result['procurement_pct']
    batt_pct = result.get('battery_dispatch_pct', 0)
    ldes_pct = result.get('ldes_dispatch_pct', 0)
    orig_score = result.get('hourly_match_score', 0) / 100.0

    # Compute original cost for comparison
    if cost_levels:
        r_gen, f_gen, stor, fuel, tx = cost_levels
        orig_cost_data = compute_costs_parameterized(
            iso, mix, proc, batt_pct, ldes_pct, orig_score * 100,
            r_gen, f_gen, stor, fuel, tx
        )
        orig_cost = orig_cost_data['effective_cost']
    else:
        orig_cost_data = compute_costs(
            iso, mix, proc, batt_pct, ldes_pct, orig_score * 100,
            demand_norm, supply_profiles
        )
        orig_cost = orig_cost_data['effective_cost_per_useful_mwh']

    best_result = dict(result)
    best_cost = orig_cost

    # Procurement bounds for this threshold
    proc_min, proc_max = PROCUREMENT_BOUNDS.get(threshold, (70, 310))

    # Cost evaluation helper
    def eval_cost(combo, p, bp, lp, score):
        if cost_levels:
            r_gen, f_gen, stor, fuel, tx = cost_levels
            cd = compute_costs_parameterized(
                iso, combo, p, bp, lp, score * 100,
                r_gen, f_gen, stor, fuel, tx
            )
            return cd['effective_cost']
        else:
            cd = compute_costs(
                iso, combo, p, bp, lp, score * 100,
                demand_norm, supply_profiles
            )
            return cd['effective_cost_per_useful_mwh']

    # Generate neighborhood around the winning mix
    fine_combos = generate_combinations_around(mix, hydro_cap, step=1, radius=neighborhood)

    # Sweep procurement in ±4% range at 2% steps
    proc_range = list(range(max(proc_min, proc - 4), min(proc_max, proc + 5), 2))
    if proc not in proc_range:
        proc_range.append(proc)
    proc_range.sort()

    for p in proc_range:
        pf = p / 100.0
        for combo in fine_combos:
            mix_fracs = np.array([combo[rt] / 100.0 for rt in RESOURCE_TYPES])

            # Base score (no storage)
            score = fast_hourly_score(demand_arr, supply_matrix, mix_fracs, pf)
            best_bp = 0
            best_lp = 0
            best_score_here = score

            # Try storage if base score doesn't meet threshold
            if score >= target:
                # Already meets threshold — check storage for improvement
                for bp in range(0, min(batt_pct + 8, 32), 2):
                    if bp == 0 and ldes_pct == 0:
                        continue
                    for lp in range(0, min(ldes_pct + 8, 22), 2):
                        if bp == 0 and lp == 0:
                            continue
                        score_ws = fast_score_with_both_storage(
                            demand_arr, supply_matrix, mix_fracs, pf, bp, lp
                        )
                        if score_ws > best_score_here:
                            best_score_here = score_ws
                            best_bp = bp
                            best_lp = lp
                if best_score_here == score:
                    # No storage helps — use base score
                    best_score_here = score
            else:
                # Need storage to meet threshold
                # Try battery
                for bp in range(2, 30, 2):
                    score_ws = fast_score_with_battery(
                        demand_arr, supply_matrix, mix_fracs, pf, bp
                    )
                    if score_ws > best_score_here:
                        best_score_here = score_ws
                        best_bp = bp
                    if score_ws >= target:
                        break

                if best_score_here < target:
                    # Try LDES
                    for lp in range(2, 22, 2):
                        score_ws = fast_score_with_both_storage(
                            demand_arr, supply_matrix, mix_fracs, pf, 0, lp
                        )
                        if score_ws > best_score_here:
                            best_score_here = score_ws
                            best_bp = 0
                            best_lp = lp
                        if score_ws >= target:
                            break

                if best_score_here < target:
                    # Try both battery + LDES
                    for bp in range(2, 22, 2):
                        for lp in range(2, 22, 2):
                            score_ws = fast_score_with_both_storage(
                                demand_arr, supply_matrix, mix_fracs, pf, bp, lp
                            )
                            if score_ws > best_score_here:
                                best_score_here = score_ws
                                best_bp = bp
                                best_lp = lp
                            if score_ws >= target:
                                break
                        if best_score_here >= target:
                            break

            if best_score_here < target:
                continue

            # Evaluate cost
            cost = eval_cost(combo, p, best_bp, best_lp, best_score_here)
            if cost < best_cost:
                best_cost = cost
                best_result = {
                    'procurement_pct': p,
                    'resource_mix': dict(combo),
                    'battery_dispatch_pct': round(best_bp, 1),
                    'ldes_dispatch_pct': round(best_lp, 1),
                    'hourly_match_score': round(best_score_here * 100, 1),
                }

    improved = best_cost < orig_cost - 0.01
    return best_result, improved, orig_cost, best_cost


def process_iso_phase3(args):
    """
    Process Phase 3 refinement for a single ISO.
    Called by multiprocessing pool.
    """
    (iso, cached_results, demand_data, gen_profiles, emission_rates,
     fossil_mix, fmt, neighborhood) = args

    print(f"\n{'='*70}")
    print(f"  Phase 3 Refinement: {ISO_LABELS[iso]}")
    print(f"{'='*70}")

    demand_norm = demand_data[iso]['normalized'][:H]
    supply_profiles = get_supply_profiles(iso, gen_profiles)
    hydro_cap = HYDRO_CAPS[iso]
    anomaly_hours = find_anomaly_hours(iso, gen_profiles)
    demand_arr, supply_arrs, supply_matrix = prepare_numpy_profiles(demand_norm, supply_profiles)
    demand_total_mwh = demand_data[iso]['total_annual_mwh']

    iso_data = cached_results[iso]

    total_refined = 0
    total_improved = 0
    total_cost_savings = 0.0
    improvements_by_threshold = {}

    # Process each threshold
    for t_str in sorted(iso_data.get('thresholds', {}).keys(), key=lambda x: float(x)):
        threshold = float(t_str)
        t_data = iso_data['thresholds'][t_str]
        t_start = time.time()
        t_improved = 0
        t_refined = 0

        if fmt == 'new' and 'scenarios' in t_data:
            # New format: iterate over all scenarios
            scenarios = t_data['scenarios']
            for scenario_key, result in list(scenarios.items()):
                result = normalize_result(result, fmt)
                cost_levels = COST_SCENARIO_MAP.get(scenario_key)

                refined, improved, old_cost, new_cost = refine_single_result(
                    demand_arr, supply_matrix, hydro_cap,
                    result, threshold, neighborhood,
                    iso, cost_levels, demand_norm, supply_profiles
                )
                t_refined += 1

                if improved:
                    # Recompute cost data for the refined result
                    if cost_levels:
                        r_gen, f_gen, stor, fuel, tx = cost_levels
                        cost_data = compute_costs_parameterized(
                            iso, refined['resource_mix'], refined['procurement_pct'],
                            refined['battery_dispatch_pct'], refined['ldes_dispatch_pct'],
                            refined['hourly_match_score'],
                            r_gen, f_gen, stor, fuel, tx
                        )
                        refined['costs'] = cost_data
                    scenarios[scenario_key] = refined
                    t_improved += 1
                    total_cost_savings += (old_cost - new_cost)

            # Refresh Medium scenario detail if improved
            medium_key = 'MMM_M_M'
            if medium_key in scenarios:
                med = scenarios[medium_key]
                peak_gap = compute_peak_gap(
                    demand_norm, supply_profiles, med['resource_mix'],
                    med['procurement_pct'], med['battery_dispatch_pct'],
                    med['ldes_dispatch_pct'], anomaly_hours
                )
                med['peak_gap_pct'] = peak_gap
                full_costs = compute_costs(
                    iso, med['resource_mix'], med['procurement_pct'],
                    med['battery_dispatch_pct'], med['ldes_dispatch_pct'],
                    med['hourly_match_score'],
                    demand_norm, supply_profiles
                )
                med['costs_detail'] = full_costs
                co2 = compute_co2_abatement(
                    iso, med['resource_mix'], med['procurement_pct'],
                    med['hourly_match_score'],
                    med['battery_dispatch_pct'], med['ldes_dispatch_pct'],
                    emission_rates, demand_total_mwh,
                    demand_norm=demand_norm, supply_profiles=supply_profiles,
                    fossil_mix=fossil_mix
                )
                med['co2_abated'] = co2
                cdp = compute_compressed_day(
                    demand_norm, supply_profiles, med['resource_mix'],
                    med['procurement_pct'],
                    med['battery_dispatch_pct'], med['ldes_dispatch_pct']
                )
                med['compressed_day'] = cdp

            t_data['scenarios'] = scenarios
            t_data['scenario_count'] = len(scenarios)

        elif fmt == 'old' and 'resource_mix' in t_data:
            # Old format: single result per threshold
            result = normalize_result(t_data, fmt)

            refined, improved, old_cost, new_cost = refine_single_result(
                demand_arr, supply_matrix, hydro_cap,
                result, threshold, neighborhood,
                iso, None, demand_norm, supply_profiles
            )
            t_refined += 1

            if improved:
                # Update the threshold data in-place
                for k, v in refined.items():
                    t_data[k] = v

                # Recompute full costs, peak gap, CO2, compressed day
                peak_gap = compute_peak_gap(
                    demand_norm, supply_profiles, refined['resource_mix'],
                    refined['procurement_pct'], refined['battery_dispatch_pct'],
                    refined['ldes_dispatch_pct'], anomaly_hours
                )
                t_data['peak_gap_pct'] = peak_gap

                full_costs = compute_costs(
                    iso, refined['resource_mix'], refined['procurement_pct'],
                    refined['battery_dispatch_pct'], refined['ldes_dispatch_pct'],
                    refined['hourly_match_score'],
                    demand_norm, supply_profiles
                )
                t_data['costs'] = full_costs

                co2 = compute_co2_abatement(
                    iso, refined['resource_mix'], refined['procurement_pct'],
                    refined['hourly_match_score'],
                    refined['battery_dispatch_pct'], refined['ldes_dispatch_pct'],
                    emission_rates, demand_total_mwh,
                    demand_norm=demand_norm, supply_profiles=supply_profiles,
                    fossil_mix=fossil_mix
                )
                t_data['co2_abated'] = co2

                cdp = compute_compressed_day(
                    demand_norm, supply_profiles, refined['resource_mix'],
                    refined['procurement_pct'],
                    refined['battery_dispatch_pct'], refined['ldes_dispatch_pct']
                )
                t_data['compressed_day'] = cdp

                t_improved += 1
                total_cost_savings += (old_cost - new_cost)

        t_elapsed = time.time() - t_start
        total_refined += t_refined
        total_improved += t_improved
        improvements_by_threshold[t_str] = t_improved

        print(f"    {t_str}%: refined {t_refined} scenarios, "
              f"{t_improved} improved, {t_elapsed:.1f}s")

    # ── Monotonicity enforcement ──
    sorted_thresholds = sorted(
        [float(t) for t in iso_data.get('thresholds', {}).keys()]
    )

    if fmt == 'new':
        mono_fixes = enforce_monotonicity_new_format(
            iso, iso_data, sorted_thresholds, demand_arr, supply_matrix,
            hydro_cap, demand_norm, supply_profiles,
            emission_rates, demand_total_mwh, fossil_mix, anomaly_hours,
            neighborhood
        )
    else:
        mono_fixes = enforce_monotonicity_old_format(
            iso, iso_data, sorted_thresholds, demand_arr, supply_matrix,
            hydro_cap, demand_norm, supply_profiles,
            emission_rates, demand_total_mwh, fossil_mix, anomaly_hours,
            neighborhood
        )

    total_improved += mono_fixes

    # ── Re-sweep the sweep chart data ──
    if 'sweep' in iso_data:
        print(f"  Re-computing sweep chart data...")
        for i, r in enumerate(iso_data['sweep']):
            r = normalize_result(r, fmt)
            proc = r['procurement_pct']
            mix = r['resource_mix']
            batt_pct = r.get('battery_dispatch_pct', r.get('storage_dispatch_pct', 0))
            ldes_pct = r.get('ldes_dispatch_pct', 0)

            # Re-evaluate matching score with corrected profiles
            mix_fracs = np.array([mix[rt] / 100.0 for rt in RESOURCE_TYPES])
            pf = proc / 100.0

            if batt_pct > 0 or ldes_pct > 0:
                new_score = fast_score_with_both_storage(
                    demand_arr, supply_matrix, mix_fracs, pf, batt_pct, ldes_pct
                )
            else:
                new_score = fast_hourly_score(demand_arr, supply_matrix, mix_fracs, pf)

            r['hourly_match_score'] = round(new_score * 100, 1)

            # Recompute costs
            costs = compute_costs(
                iso, mix, proc, batt_pct, ldes_pct,
                r['hourly_match_score'], demand_norm, supply_profiles
            )
            r['costs'] = costs

            # Recompute peak gap
            peak_gap = compute_peak_gap(
                demand_norm, supply_profiles, mix, proc, batt_pct, ldes_pct,
                anomaly_hours
            )
            r['peak_gap_pct'] = peak_gap

            # Recompute CO2
            co2 = compute_co2_abatement(
                iso, mix, proc, r['hourly_match_score'],
                batt_pct, ldes_pct,
                emission_rates, demand_total_mwh,
                demand_norm=demand_norm, supply_profiles=supply_profiles,
                fossil_mix=fossil_mix
            )
            r['co2_abated'] = co2
            iso_data['sweep'][i] = r

    print(f"\n  {iso} Summary: {total_improved}/{total_refined} improved, "
          f"avg savings: ${total_cost_savings/max(total_improved,1):.2f}/MWh")

    return iso, iso_data, total_improved, total_refined, total_cost_savings


# ══════════════════════════════════════════════════════════════════════════════
# MONOTONICITY ENFORCEMENT
# ══════════════════════════════════════════════════════════════════════════════

def enforce_monotonicity_new_format(iso, iso_data, sorted_thresholds,
                                    demand_arr, supply_matrix, hydro_cap,
                                    demand_norm, supply_profiles,
                                    emission_rates, demand_total_mwh,
                                    fossil_mix, anomaly_hours, neighborhood):
    """
    Enforce cost monotonicity for new format (324 scenarios per threshold).
    Cost must be non-decreasing across thresholds for each scenario.
    If violated, re-refine the lower threshold using the higher threshold's mix as seed.
    """
    MAX_ROUNDS = 2
    total_fixes = 0

    # Collect all scenario keys
    all_scenario_keys = set()
    for t in sorted_thresholds:
        t_str = str(t) if t == int(t) else str(t)
        t_data = iso_data['thresholds'].get(t_str, {})
        if 'scenarios' in t_data:
            all_scenario_keys.update(t_data['scenarios'].keys())

    for round_num in range(MAX_ROUNDS):
        violations = {}  # {threshold: {scenario_key: better_threshold}}
        total_violations = 0

        for scenario_key in all_scenario_keys:
            prev_cost = None
            prev_t = None
            for threshold in sorted_thresholds:
                t_str = str(threshold) if threshold == int(threshold) else str(threshold)
                t_data = iso_data['thresholds'].get(t_str, {})
                scenarios = t_data.get('scenarios', {})
                if scenario_key not in scenarios:
                    continue
                result = scenarios[scenario_key]
                if 'costs' not in result:
                    continue
                cost = result['costs']['effective_cost']
                if prev_cost is not None and cost < prev_cost - 0.01:
                    total_violations += 1
                    if prev_t not in violations:
                        violations[prev_t] = {}
                    violations[prev_t][scenario_key] = threshold
                prev_cost = cost
                prev_t = threshold

        if not violations:
            if round_num == 0:
                print(f"  Monotonicity check passed for all scenarios")
            else:
                print(f"  Monotonicity round {round_num}: all violations resolved")
            break

        print(f"  Monotonicity round {round_num + 1}: {total_violations} violations "
              f"across {len(violations)} threshold(s)")

        for viol_threshold in sorted(violations.keys()):
            violated_scenarios = violations[viol_threshold]
            t_str = str(viol_threshold) if viol_threshold == int(viol_threshold) else str(viol_threshold)
            scenarios = iso_data['thresholds'][t_str].get('scenarios', {})

            for sk, better_t in violated_scenarios.items():
                # Use the better threshold's mix as seed for re-refinement
                better_t_str = str(better_t) if better_t == int(better_t) else str(better_t)
                better_result = iso_data['thresholds'].get(better_t_str, {}).get('scenarios', {}).get(sk)

                if not better_result or 'resource_mix' not in better_result:
                    continue

                cost_levels = COST_SCENARIO_MAP.get(sk)
                seed_result = normalize_result(better_result, 'new')

                # Re-refine at the violated threshold using the seed mix
                refined, improved, old_cost, new_cost = refine_single_result(
                    demand_arr, supply_matrix, hydro_cap,
                    seed_result, viol_threshold, neighborhood,
                    iso, cost_levels, demand_norm, supply_profiles
                )

                # Also check: just directly using the higher threshold's result
                # (it already achieves >= higher threshold, so it meets lower too)
                if cost_levels:
                    r_gen, f_gen, stor, fuel, tx = cost_levels
                    seed_cost_data = compute_costs_parameterized(
                        iso, seed_result['resource_mix'], seed_result['procurement_pct'],
                        seed_result['battery_dispatch_pct'], seed_result['ldes_dispatch_pct'],
                        seed_result['hourly_match_score'],
                        r_gen, f_gen, stor, fuel, tx
                    )
                    seed_cost = seed_cost_data['effective_cost']
                else:
                    seed_cost = float('inf')

                # Pick the cheaper of: (a) refined result, (b) seed from higher threshold
                current = scenarios.get(sk)
                current_cost = current['costs']['effective_cost'] if current and 'costs' in current else float('inf')

                if cost_levels:
                    r_gen, f_gen, stor, fuel, tx = cost_levels
                    refined_cost_data = compute_costs_parameterized(
                        iso, refined['resource_mix'], refined['procurement_pct'],
                        refined['battery_dispatch_pct'], refined['ldes_dispatch_pct'],
                        refined['hourly_match_score'],
                        r_gen, f_gen, stor, fuel, tx
                    )
                    refined_cost = refined_cost_data['effective_cost']
                else:
                    refined_cost = float('inf')

                # Use whichever is cheapest
                if seed_cost < refined_cost and seed_cost < current_cost:
                    seed_result['costs'] = seed_cost_data
                    scenarios[sk] = seed_result
                    total_fixes += 1
                elif refined_cost < current_cost:
                    refined['costs'] = refined_cost_data
                    scenarios[sk] = refined
                    total_fixes += 1

            print(f"    Re-refined {t_str}%: {len(violated_scenarios)} scenarios checked")
    else:
        # Exhausted MAX_ROUNDS
        remaining = 0
        for scenario_key in all_scenario_keys:
            prev_cost = None
            for threshold in sorted_thresholds:
                t_str = str(threshold) if threshold == int(threshold) else str(threshold)
                t_data = iso_data['thresholds'].get(t_str, {})
                scenarios = t_data.get('scenarios', {})
                if scenario_key not in scenarios:
                    continue
                result = scenarios[scenario_key]
                if 'costs' not in result:
                    continue
                cost = result['costs']['effective_cost']
                if prev_cost is not None and cost < prev_cost - 0.01:
                    remaining += 1
                prev_cost = cost
        if remaining > 0:
            print(f"  WARNING: {remaining} monotonicity violations remain after {MAX_ROUNDS} rounds")
        else:
            print(f"  All monotonicity violations resolved after {MAX_ROUNDS} rounds")

    return total_fixes


def enforce_monotonicity_old_format(iso, iso_data, sorted_thresholds,
                                    demand_arr, supply_matrix, hydro_cap,
                                    demand_norm, supply_profiles,
                                    emission_rates, demand_total_mwh,
                                    fossil_mix, anomaly_hours, neighborhood):
    """
    Enforce cost monotonicity for old format (single result per threshold).
    Cost must be non-decreasing across thresholds.
    """
    total_fixes = 0

    # Build cost sequence
    costs = []
    for threshold in sorted_thresholds:
        t_str = str(threshold) if threshold == int(threshold) else str(threshold)
        t_data = iso_data['thresholds'].get(t_str, {})
        cost = None
        if 'costs' in t_data:
            cost = t_data['costs'].get('effective_cost_per_useful_mwh',
                                        t_data['costs'].get('effective_cost'))
        costs.append((threshold, t_str, cost))

    # Find violations
    for i in range(len(costs) - 1):
        t_lower, t_str_lower, cost_lower = costs[i]
        t_higher, t_str_higher, cost_higher = costs[i + 1]

        if cost_lower is None or cost_higher is None:
            continue

        if cost_higher < cost_lower - 0.01:
            # Violation: lower threshold costs more than higher
            # Use higher threshold's mix as seed to re-refine the lower threshold
            higher_data = iso_data['thresholds'].get(t_str_higher, {})
            higher_result = normalize_result(higher_data, 'old')

            refined, improved, old_cost, new_cost = refine_single_result(
                demand_arr, supply_matrix, hydro_cap,
                higher_result, t_lower, neighborhood,
                iso, None, demand_norm, supply_profiles
            )

            # If refinement found something cheaper than current, use it
            lower_data = iso_data['thresholds'].get(t_str_lower, {})
            current_cost = lower_data.get('costs', {}).get('effective_cost_per_useful_mwh', float('inf'))

            if new_cost < current_cost - 0.01:
                for k, v in refined.items():
                    lower_data[k] = v

                # Recompute full costs, peak gap, CO2, compressed day
                peak_gap = compute_peak_gap(
                    demand_norm, supply_profiles, refined['resource_mix'],
                    refined['procurement_pct'], refined['battery_dispatch_pct'],
                    refined['ldes_dispatch_pct'], anomaly_hours
                )
                lower_data['peak_gap_pct'] = peak_gap
                full_costs = compute_costs(
                    iso, refined['resource_mix'], refined['procurement_pct'],
                    refined['battery_dispatch_pct'], refined['ldes_dispatch_pct'],
                    refined['hourly_match_score'],
                    demand_norm, supply_profiles
                )
                lower_data['costs'] = full_costs
                co2 = compute_co2_abatement(
                    iso, refined['resource_mix'], refined['procurement_pct'],
                    refined['hourly_match_score'],
                    refined['battery_dispatch_pct'], refined['ldes_dispatch_pct'],
                    emission_rates, demand_total_mwh,
                    demand_norm=demand_norm, supply_profiles=supply_profiles,
                    fossil_mix=fossil_mix
                )
                lower_data['co2_abated'] = co2
                cdp = compute_compressed_day(
                    demand_norm, supply_profiles, refined['resource_mix'],
                    refined['procurement_pct'],
                    refined['battery_dispatch_pct'], refined['ldes_dispatch_pct']
                )
                lower_data['compressed_day'] = cdp

                total_fixes += 1
                print(f"    Monotonicity fix: {t_str_lower}% cost reduced "
                      f"${current_cost:.2f} -> ${new_cost:.2f}")

    return total_fixes


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Phase 3 fine-grained refinement around cached optimizer results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python optimize_phase3_only.py
  python optimize_phase3_only.py --neighborhood 3 --workers 3
  python optimize_phase3_only.py --results-file data/optimizer_cache.json
        """
    )
    parser.add_argument('--results-file', type=str, default=None,
                        help='Path to cached results JSON (default: auto-detect '
                             'data/optimizer_cache.json or dashboard/overprocure_results.json)')
    parser.add_argument('--profiles-dir', type=str, default=DATA_DIR,
                        help=f'Directory containing corrected profile files (default: {DATA_DIR})')
    parser.add_argument('--workers', type=int, default=5,
                        help='Number of parallel workers (default: 5)')
    parser.add_argument('--neighborhood', type=int, default=5,
                        help='Search radius in %% points around each resource (default: 5)')
    args = parser.parse_args()

    start_time = time.time()
    neighborhood = args.neighborhood

    # ── Load cached results ──
    results_file = args.results_file
    if results_file is None:
        # Auto-detect: prefer optimizer_cache.json, fall back to overprocure_results.json
        cache_path = os.path.join(SCRIPT_DIR, 'data', 'optimizer_cache.json')
        dashboard_path = os.path.join(SCRIPT_DIR, 'dashboard', 'overprocure_results.json')
        if os.path.exists(cache_path):
            results_file = cache_path
        elif os.path.exists(dashboard_path):
            results_file = dashboard_path
        else:
            print("ERROR: No cached results found. Run the full optimizer first.")
            print(f"  Looked for: {cache_path}")
            print(f"           : {dashboard_path}")
            sys.exit(1)

    print(f"Loading cached results from: {results_file}")
    with open(results_file) as f:
        cached_data = json.load(f)

    # Extract results (handle both cache format and dashboard format)
    if 'results' in cached_data:
        results = cached_data['results']
        config = cached_data.get('config', {})
    else:
        print("ERROR: Unexpected results file format (no 'results' key)")
        sys.exit(1)

    # Detect format
    fmt = detect_results_format(results)
    print(f"Results format: {fmt}")
    if fmt == 'unknown':
        print("WARNING: Could not detect results format. Attempting 'old' format handling.")
        fmt = 'old'

    # Count total scenarios to process
    total_scenarios = 0
    for iso in ISOS:
        if iso not in results:
            continue
        for t_str, t_data in results[iso].get('thresholds', {}).items():
            if fmt == 'new' and 'scenarios' in t_data:
                total_scenarios += len(t_data['scenarios'])
            elif fmt == 'old' and 'resource_mix' in t_data:
                total_scenarios += 1

    print(f"Total scenarios to refine: {total_scenarios}")
    print(f"Neighborhood: ±{neighborhood}% (1% step)")
    print(f"Workers: {args.workers}")

    # ── Load corrected data ──
    print(f"\nLoading corrected profiles from: {args.profiles_dir}")
    demand_data, gen_profiles, emission_rates, fossil_mix = load_data()

    # ── Run Phase 3 refinement for all ISOs ──
    worker_args = [
        (iso, results, demand_data, gen_profiles, emission_rates,
         fossil_mix, fmt, neighborhood)
        for iso in ISOS if iso in results
    ]

    all_improved = 0
    all_refined = 0
    all_savings = 0.0

    try:
        with Pool(processes=min(args.workers, len(worker_args))) as pool:
            results_list = pool.map(process_iso_phase3, worker_args)

        for iso, iso_data, n_improved, n_refined, savings in results_list:
            results[iso] = iso_data
            all_improved += n_improved
            all_refined += n_refined
            all_savings += savings
    except Exception as e:
        print(f"\nMultiprocessing failed ({e}), falling back to sequential...")
        for worker_arg in worker_args:
            iso, iso_data, n_improved, n_refined, savings = process_iso_phase3(worker_arg)
            results[iso] = iso_data
            all_improved += n_improved
            all_refined += n_refined
            all_savings += savings

    # ── Save results ──
    # Rebuild the full output structure
    output_data = {
        'config': config,
        'results': results,
    }

    # Save dashboard results
    dashboard_path = os.path.join(SCRIPT_DIR, 'dashboard', 'overprocure_results.json')
    os.makedirs(os.path.dirname(dashboard_path), exist_ok=True)
    with open(dashboard_path, 'w') as f:
        json.dump(output_data, f)
    print(f"\nSaved dashboard results: {dashboard_path} "
          f"({os.path.getsize(dashboard_path) / 1024:.0f} KB)")

    # Save optimizer cache
    try:
        import subprocess
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=SCRIPT_DIR, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = 'unknown'

    from datetime import datetime, timezone
    cache_data = {
        'metadata': {
            'created_at': datetime.now(timezone.utc).isoformat(),
            'optimizer_version': '3.2-phase3-refinement',
            'git_commit': git_hash,
            'runtime_seconds': round(time.time() - start_time, 1),
            'source_file': os.path.basename(results_file),
            'description': f'Phase 3 refinement (±{neighborhood}%) of cached optimizer results',
            'thresholds': THRESHOLDS,
            'isos': ISOS,
            'total_refined': all_refined,
            'total_improved': all_improved,
        },
        'config': config,
        'results': results,
    }
    cache_path = os.path.join(SCRIPT_DIR, 'data', 'optimizer_cache.json')
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f, indent=2)
    print(f"Saved optimizer cache: {cache_path} "
          f"({os.path.getsize(cache_path) / 1024:.0f} KB)")

    # ── Summary ──
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  Phase 3 Refinement Complete")
    print(f"{'='*70}")
    print(f"  Runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Scenarios refined: {all_refined}")
    print(f"  Improvements found: {all_improved} "
          f"({all_improved/max(all_refined,1)*100:.1f}%)")
    if all_improved > 0:
        print(f"  Average cost reduction: ${all_savings/all_improved:.2f}/MWh")
        print(f"  Total cost savings: ${all_savings:.2f}/MWh (summed)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
