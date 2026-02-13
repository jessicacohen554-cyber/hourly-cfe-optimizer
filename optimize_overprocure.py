#!/usr/bin/env python3
"""
Over-Procurement Optimization Engine
======================================
For each ISO region (using EIA demand shape), sweep over-procurement levels
(100%, 120%, 140%...) and find the optimal mix of 4 resource types
(clean firm, solar, wind, hydro) + uncapped storage to maximize hourly matching.

Three-phase refinement: 20% coarse → 5% medium → 1% fine

Resource types:
  - Clean Firm: flat baseload (1/8760 per hour) — represents nuclear/geothermal
  - Solar: EIA regional profile
  - Wind: EIA regional profile
  - Hydro: EIA regional profile (capped by region)

Storage: 4h Li-ion, 85% RTE, dispatch target optimized (uncapped)
"""

import json
import os
import sys
import time
import numpy as np
from itertools import product

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
DATA_YEAR = '2025'
H = 8760

STORAGE_EFFICIENCY = 0.85
STORAGE_DURATION_HOURS = 4

# Hydro caps (same as before)
HYDRO_CAPS = {
    'CAISO': 30,
    'ERCOT': 5,
    'PJM': 15,
    'NYISO': 40,
    'NEISO': 30,
}

ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO']

ISO_LABELS = {
    'CAISO': 'CAISO (California)',
    'ERCOT': 'ERCOT (Texas)',
    'PJM': 'PJM (Mid-Atlantic)',
    'NYISO': 'NYISO (New York)',
    'NEISO': 'NEISO (New England)',
}

# Resource types (order for optimization)
RESOURCE_TYPES = ['clean_firm', 'solar', 'wind', 'hydro']

# Target thresholds to find
THRESHOLDS = [75, 80, 85, 90, 95, 99, 100]

# ══════════════════════════════════════════════════════════════════════════════
# COST LAYER — Regional LCOE and Wholesale Market Prices
# ══════════════════════════════════════════════════════════════════════════════

# Wholesale electricity prices ($/MWh) — 2023-2024 averages from FERC/ISO market reports
WHOLESALE_PRICES = {
    'CAISO': 30,
    'ERCOT': 27,
    'PJM':   34,
    'NYISO': 42,
    'NEISO': 41,
}

# New-build LCOE by resource type and ISO ($/MWh)
# Sources: LBNL Utility-Scale Solar 2024, LBNL Wind Market Report 2024
# Wind scaled +30% from 2024 PPA prices to reflect 2025 cost trends (matching solar YoY trajectory)
# Clean firm: $90/MWh nationally (nuclear/geothermal new-build blended estimate)
# Storage: $100/MWh (4h Li-ion LCOS estimate, 2025)
REGIONAL_LCOE = {
    'CAISO': {'clean_firm': 90, 'solar': 60, 'wind': 73, 'hydro': 0, 'storage': 100},
    'ERCOT': {'clean_firm': 90, 'solar': 54, 'wind': 40, 'hydro': 0, 'storage': 100},
    'PJM':   {'clean_firm': 90, 'solar': 65, 'wind': 62, 'hydro': 0, 'storage': 100},
    'NYISO': {'clean_firm': 90, 'solar': 92, 'wind': 81, 'hydro': 0, 'storage': 100},
    'NEISO': {'clean_firm': 90, 'solar': 82, 'wind': 73, 'hydro': 0, 'storage': 100},
}

# Existing clean grid mix shares (% of total generation) from EIA-930 2025 data
# Resources up to these shares priced at wholesale; above = new-build LCOE
GRID_MIX_SHARES = {
    'CAISO': {'clean_firm': 7.9, 'solar': 22.3, 'wind': 8.8, 'hydro': 9.5},
    'ERCOT': {'clean_firm': 8.6, 'solar': 13.8, 'wind': 23.6, 'hydro': 0.1},
    'PJM':   {'clean_firm': 32.1, 'solar': 2.9, 'wind': 3.8, 'hydro': 1.8},
    'NYISO': {'clean_firm': 18.4, 'solar': 0.0, 'wind': 4.7, 'hydro': 15.9},
    'NEISO': {'clean_firm': 23.8, 'solar': 1.4, 'wind': 3.9, 'hydro': 4.4},
}


def load_data():
    """Load demand profiles and generation profiles."""
    print("Loading data...")

    with open(os.path.join(DATA_DIR, 'eia_demand_profiles.json')) as f:
        demand_data = json.load(f)

    with open(os.path.join(DATA_DIR, 'eia_generation_profiles.json')) as f:
        gen_profiles = json.load(f)

    print("  Data loaded.")
    return demand_data, gen_profiles


def get_supply_profiles(iso, gen_profiles):
    """Get generation shape profiles for the 4 resource types."""
    profiles = {}

    # Clean firm = flat baseload
    profiles['clean_firm'] = [1.0 / H] * H

    # Solar
    if iso == 'NYISO':
        p = gen_profiles[iso][DATA_YEAR].get('solar_proxy')
        if not p:
            p = gen_profiles['NEISO'][DATA_YEAR].get('solar')
        profiles['solar'] = p
    else:
        profiles['solar'] = gen_profiles[iso][DATA_YEAR].get('solar')

    # Wind
    profiles['wind'] = gen_profiles[iso][DATA_YEAR].get('wind')

    # Hydro
    profiles['hydro'] = gen_profiles[iso][DATA_YEAR].get('hydro')

    return profiles


def find_anomaly_hours(iso, gen_profiles):
    """Find hours where all gen types report zero (EIA data gaps)."""
    types = [t for t in gen_profiles[iso][DATA_YEAR].keys() if t != 'solar_proxy']
    anomalies = set()
    for h in range(H):
        if all(gen_profiles[iso][DATA_YEAR][t][h] == 0.0 for t in types):
            anomalies.add(h)
    return anomalies


def prepare_numpy_profiles(demand_norm, supply_profiles):
    """Convert profiles to numpy arrays for fast vectorized computation."""
    demand_arr = np.array(demand_norm, dtype=np.float64)
    supply_arrs = {}
    for rtype in RESOURCE_TYPES:
        supply_arrs[rtype] = np.array(supply_profiles[rtype], dtype=np.float64)
    # Pre-build supply matrix: shape (4, 8760) for [cf, solar, wind, hydro]
    supply_matrix = np.stack([supply_arrs[rt] for rt in RESOURCE_TYPES])  # (4, 8760)
    return demand_arr, supply_arrs, supply_matrix


def fast_hourly_score(demand_arr, supply_matrix, mix_fractions, procurement_factor):
    """
    Ultra-fast hourly matching score using numpy vectorized ops.
    mix_fractions: array of [cf, solar, wind, hydro] as fractions (sum to 1.0)
    Returns: matching score (0-1)
    """
    # Weighted supply = procurement_factor * sum(fraction_i * profile_i)
    supply = procurement_factor * np.dot(mix_fractions, supply_matrix)
    matched = np.minimum(demand_arr, supply)
    return matched.sum()  # demand sums to 1.0, so this is the score


def fast_score_with_storage(demand_arr, supply_matrix, mix_fractions, procurement_factor,
                            storage_dispatch_pct):
    """
    Fast scoring with storage (numpy-accelerated).
    Returns matching score (0-1).
    """
    supply = procurement_factor * np.dot(mix_fractions, supply_matrix)
    surplus = np.maximum(0.0, supply - demand_arr)
    gap = np.maximum(0.0, demand_arr - supply)
    base_matched = np.minimum(demand_arr, supply)

    if storage_dispatch_pct <= 0:
        return base_matched.sum()

    # Storage dispatch simulation
    storage_dispatch_total = storage_dispatch_pct / 100.0
    num_days = H // 24
    daily_dispatch_target = storage_dispatch_total / num_days
    power_rating = daily_dispatch_target / STORAGE_DURATION_HOURS

    total_dispatched = 0.0
    for day in range(num_days):
        ds = day * 24
        de = ds + 24

        day_surplus = surplus[ds:de]
        day_gap = gap[ds:de]

        total_surplus = day_surplus.sum()
        total_gap = day_gap.sum()

        max_from_charge = total_surplus * STORAGE_EFFICIENCY
        actual_dispatch = min(daily_dispatch_target, max_from_charge, total_gap)
        if actual_dispatch <= 0:
            continue

        required_charge = actual_dispatch / STORAGE_EFFICIENCY

        # Distribute charge (greedily, largest surplus first)
        sorted_idx = np.argsort(-day_surplus)
        remaining_charge = required_charge
        for idx in sorted_idx:
            if remaining_charge <= 0 or day_surplus[idx] <= 0:
                break
            amt = min(day_surplus[idx], power_rating, remaining_charge)
            remaining_charge -= amt

        actual_charge = required_charge - remaining_charge
        ach_dispatch = min(actual_dispatch, actual_charge * STORAGE_EFFICIENCY)

        # Distribute dispatch (greedily, largest gap first)
        sorted_idx = np.argsort(-day_gap)
        remaining_dispatch = ach_dispatch
        for idx in sorted_idx:
            if remaining_dispatch <= 0 or day_gap[idx] <= 0:
                break
            amt = min(day_gap[idx], power_rating, remaining_dispatch)
            total_dispatched += amt
            remaining_dispatch -= amt

    return base_matched.sum() + total_dispatched


def compute_hourly_matching(demand_norm, supply_profiles, resource_pcts, procurement_pct,
                            storage_dispatch_pct=0):
    """
    Compute hourly matching score.
    resource_pcts: dict of resource -> % of PROCURED amount (sums to 100)
    procurement_pct: total procurement as % of annual load (e.g., 100, 120, 150)
    storage_dispatch_pct: % of annual load to dispatch from storage (0 = no storage)

    Returns: (score, hourly_detail, dispatch_profile, charge_profile)
    """
    # Build hourly demand and supply
    # Use normalized profiles — demand_norm sums to 1.0, supply profiles sum to ~1.0
    # procurement_pct/100 is the multiplier for total supply vs total demand

    procurement_factor = procurement_pct / 100.0

    # Compute hourly detail (surplus/gap before storage)
    hourly_detail = []
    for h in range(H):
        demand_h = demand_norm[h]  # normalized, sums to 1.0

        supply_h = 0.0
        for rtype, pct in resource_pcts.items():
            if pct <= 0:
                continue
            # This resource's share of procured amount × its hourly shape
            supply_h += procurement_factor * (pct / 100.0) * supply_profiles[rtype][h]

        matched_h = min(demand_h, supply_h)
        surplus_h = max(0.0, supply_h - demand_h)
        gap_h = max(0.0, demand_h - supply_h)

        hourly_detail.append({
            'demand': demand_h,
            'supply': supply_h,
            'matched': matched_h,
            'surplus': surplus_h,
            'gap': gap_h,
        })

    # Apply storage if dispatch_pct > 0
    dispatch_profile = [0.0] * H
    charge_profile = [0.0] * H

    if storage_dispatch_pct > 0:
        # Storage dispatch target as fraction of total demand (which sums to 1.0)
        storage_dispatch_total = storage_dispatch_pct / 100.0
        num_days = H // 24
        daily_dispatch_target = storage_dispatch_total / num_days
        power_rating = daily_dispatch_target / STORAGE_DURATION_HOURS

        for day in range(num_days):
            ds = day * 24
            de = min(ds + 24, H)

            surplus_hours = []
            gap_hours = []
            for h_idx in range(ds, de):
                d = hourly_detail[h_idx]
                if d['surplus'] > 0:
                    surplus_hours.append((h_idx, d['surplus']))
                if d['gap'] > 0:
                    gap_hours.append((h_idx, d['gap']))

            total_surplus = sum(s for _, s in surplus_hours)
            total_gap = sum(g for _, g in gap_hours)

            max_from_charge = total_surplus * STORAGE_EFFICIENCY
            actual_dispatch = min(daily_dispatch_target, max_from_charge, total_gap)
            if actual_dispatch <= 0:
                continue

            required_charge = actual_dispatch / STORAGE_EFFICIENCY

            # Distribute charge
            remaining_charge = required_charge
            surplus_hours.sort(key=lambda x: -x[1])
            for h_idx, surplus in surplus_hours:
                if remaining_charge <= 0:
                    break
                amt = min(surplus, power_rating, remaining_charge)
                charge_profile[h_idx] = amt
                remaining_charge -= amt

            actual_charge = required_charge - remaining_charge
            ach_dispatch = min(actual_dispatch, actual_charge * STORAGE_EFFICIENCY)

            # Distribute dispatch
            remaining_dispatch = ach_dispatch
            gap_hours.sort(key=lambda x: -x[1])
            for h_idx, gap in gap_hours:
                if remaining_dispatch <= 0:
                    break
                amt = min(gap, power_rating, remaining_dispatch)
                dispatch_profile[h_idx] = amt
                remaining_dispatch -= amt

    # Compute final score
    total_matched = 0.0
    total_demand = 0.0
    for h in range(H):
        d = hourly_detail[h]
        disp = dispatch_profile[h]
        new_matched = d['matched'] + min(d['gap'], disp)
        total_matched += new_matched
        total_demand += d['demand']

    score = total_matched / total_demand if total_demand > 0 else 0.0
    return score, hourly_detail, dispatch_profile, charge_profile


def generate_combinations(hydro_cap, step=5):
    """
    Generate all valid resource mix combinations that sum to 100%.
    Resources: clean_firm (0-100), solar (0-100), wind (0-100), hydro (0-hydro_cap)
    """
    combos = []
    for cf in range(0, 101, step):
        for sol in range(0, 101 - cf, step):
            for wnd in range(0, 101 - cf - sol, step):
                hyd = 100 - cf - sol - wnd
                if hyd <= hydro_cap and hyd >= 0:
                    combos.append({
                        'clean_firm': cf, 'solar': sol, 'wind': wnd, 'hydro': hyd
                    })
    return combos


def find_optimal_storage(demand_norm, supply_profiles, resource_pcts, procurement_pct):
    """
    Find the optimal storage dispatch % that maximizes hourly matching.
    Sweep from 0 to 50% of annual load in steps of 5%, then refine to 1%.
    """
    best_score = 0
    best_dispatch_pct = 0

    # Coarse sweep 0-50% in 5% steps
    for dp in range(0, 55, 5):
        score, _, _, _ = compute_hourly_matching(
            demand_norm, supply_profiles, resource_pcts, procurement_pct, dp
        )
        if score > best_score:
            best_score = score
            best_dispatch_pct = dp

    # Fine sweep around best
    for dp_10 in range(max(0, (best_dispatch_pct - 4) * 10), (best_dispatch_pct + 5) * 10, 10):
        dp = dp_10 / 10.0
        score, _, _, _ = compute_hourly_matching(
            demand_norm, supply_profiles, resource_pcts, procurement_pct, dp
        )
        if score > best_score:
            best_score = score
            best_dispatch_pct = dp

    return best_score, best_dispatch_pct


def optimize_at_procurement_level(iso, demand_norm, supply_profiles, procurement_pct, hydro_cap,
                                   np_profiles=None):
    """
    Find the resource mix that maximizes hourly matching at a given procurement level.
    Used for sweep visualization. Two-phase: 5% coarse → 1% fine, then storage.
    Uses numpy for the coarse phase if np_profiles provided.
    """
    pf = procurement_pct / 100.0

    if np_profiles:
        demand_arr, _, supply_matrix = np_profiles
        # Fast coarse search
        combos = generate_combinations(hydro_cap, step=5)
        best_score = -1
        best_combo = None

        for combo in combos:
            mix_fracs = np.array([combo[rt] / 100.0 for rt in RESOURCE_TYPES])
            score = fast_hourly_score(demand_arr, supply_matrix, mix_fracs, pf)
            if score > best_score:
                best_score = score
                best_combo = combo

        # 1% refinement (also numpy)
        refined_ranges = {}
        for rtype in RESOURCE_TYPES:
            base = best_combo[rtype]
            cap = hydro_cap if rtype == 'hydro' else 100
            refined_ranges[rtype] = list(range(max(0, base - 4), min(cap, base + 4) + 1))

        for cf in refined_ranges['clean_firm']:
            for sol in refined_ranges['solar']:
                for wnd in refined_ranges['wind']:
                    hyd = 100 - cf - sol - wnd
                    if hyd < 0 or hyd > hydro_cap:
                        continue
                    combo = {'clean_firm': cf, 'solar': sol, 'wind': wnd, 'hydro': hyd}
                    mix_fracs = np.array([combo[rt] / 100.0 for rt in RESOURCE_TYPES])
                    score = fast_hourly_score(demand_arr, supply_matrix, mix_fracs, pf)
                    if score > best_score:
                        best_score = score
                        best_combo = combo
    else:
        combos = generate_combinations(hydro_cap, step=5)
        best_score = -1
        best_combo = None

        for combo in combos:
            score, _, _, _ = compute_hourly_matching(
                demand_norm, supply_profiles, combo, procurement_pct, 0
            )
            if score > best_score:
                best_score = score
                best_combo = combo

        # 1% refinement
        refined_ranges = {}
        for rtype in RESOURCE_TYPES:
            base = best_combo[rtype]
            cap = hydro_cap if rtype == 'hydro' else 100
            refined_ranges[rtype] = list(range(max(0, base - 4), min(cap, base + 4) + 1))

        for cf in refined_ranges['clean_firm']:
            for sol in refined_ranges['solar']:
                for wnd in refined_ranges['wind']:
                    hyd = 100 - cf - sol - wnd
                    if hyd < 0 or hyd > hydro_cap:
                        continue
                    combo = {'clean_firm': cf, 'solar': sol, 'wind': wnd, 'hydro': hyd}
                    score, _, _, _ = compute_hourly_matching(
                        demand_norm, supply_profiles, combo, procurement_pct, 0
                    )
                    if score > best_score:
                        best_score = score
                        best_combo = combo

    # Optimize storage (uses original functions — only called once per level)
    best_score_ws, best_storage_pct = find_optimal_storage(
        demand_norm, supply_profiles, best_combo, procurement_pct
    )

    return {
        'procurement_pct': procurement_pct,
        'resource_mix': best_combo,
        'storage_dispatch_pct': round(best_storage_pct, 1),
        'hourly_match_score': round(best_score_ws * 100, 1),
    }


def optimize_for_threshold(iso, demand_norm, supply_profiles, threshold, hydro_cap):
    """
    CO-OPTIMIZE cost and matching simultaneously.
    For a given threshold target (e.g., 75%), search across procurement levels
    AND resource mixes to find the CHEAPEST combination that meets or exceeds it.

    Uses numpy-accelerated scoring for speed.

    Performance-optimized 3-phase approach:
      Phase 1: Coarse scan (10% mix × 10% procurement × fast no-storage scoring)
      Phase 2: Refine top candidates (5% mix × 5% procurement × storage check)
      Phase 3: Fine-tune best result (1% mix × 1% procurement × optimized storage)
    """
    target = threshold / 100.0
    max_proc = 500 if target >= 0.995 else 310  # Higher range for 100% target
    # Allow near-misses: combos within 5% of target get storage check
    storage_threshold = max(0.5, target - 0.05)
    demand_arr, supply_arrs, supply_matrix = prepare_numpy_profiles(demand_norm, supply_profiles)

    best_result = None
    best_cost = float('inf')

    def update_best(combo, proc, sp, score):
        """Helper to update best result if this is the cheapest so far."""
        nonlocal best_result, best_cost
        cost_data = compute_costs(iso, combo, proc, sp, score * 100, demand_norm, supply_profiles)
        cost = cost_data['effective_cost_per_useful_mwh']
        if cost < best_cost:
            best_cost = cost
            best_result = {
                'procurement_pct': proc,
                'resource_mix': dict(combo),
                'storage_dispatch_pct': round(sp, 1),
                'hourly_match_score': round(score * 100, 1),
            }
        return cost

    # ── Phase 1: Scan with numpy fast scoring ──
    # 5% mix grid × procurement levels in 10% steps
    combos_5 = generate_combinations(hydro_cap, step=5)
    candidates = []

    for procurement_pct in range(70, max_proc, 10):
        pf = procurement_pct / 100.0
        for combo in combos_5:
            mix_fracs = np.array([combo[rt] / 100.0 for rt in RESOURCE_TYPES])
            # Fast no-storage score
            score = fast_hourly_score(demand_arr, supply_matrix, mix_fracs, pf)
            if score >= target:
                cost = update_best(combo, procurement_pct, 0, score)
                candidates.append((cost, combo, score, 0, procurement_pct))
            elif score >= storage_threshold:
                # Near-miss: try with storage at a few levels
                for sp in [5, 10, 15, 20]:
                    score_ws = fast_score_with_storage(
                        demand_arr, supply_matrix, mix_fracs, pf, sp
                    )
                    if score_ws >= target:
                        cost = update_best(combo, procurement_pct, sp, score_ws)
                        candidates.append((cost, combo, score_ws, sp, procurement_pct))
                        break  # Found a working storage level

    if not candidates:
        return None

    # Keep top candidates within 15% of best cost
    candidates.sort(key=lambda x: x[0])
    top = [c for c in candidates if c[0] <= best_cost * 1.15][:8]

    # ── Phase 2: 5% refinement around top candidates ──
    phase2 = []
    seen = set()
    for _, combo, _, sp_base, proc in top:
        for p_d in [-5, 0, 5]:
            p = proc + p_d
            if p < 70 or p > max_proc:
                continue
            pf = p / 100.0

            for cf_d in [-5, 0, 5]:
                for sol_d in [-5, 0, 5]:
                    for wnd_d in [-5, 0, 5]:
                        cf = combo['clean_firm'] + cf_d
                        sol = combo['solar'] + sol_d
                        wnd = combo['wind'] + wnd_d
                        hyd = 100 - cf - sol - wnd
                        if cf < 0 or sol < 0 or wnd < 0 or hyd < 0 or hyd > hydro_cap:
                            continue
                        if cf > 100 or sol > 100 or wnd > 100:
                            continue

                        key = (cf, sol, wnd, p)
                        if key in seen:
                            continue
                        seen.add(key)

                        rcombo = {'clean_firm': cf, 'solar': sol, 'wind': wnd, 'hydro': hyd}
                        mix_fracs = np.array([rcombo[rt] / 100.0 for rt in RESOURCE_TYPES])

                        # Try without storage first
                        score = fast_hourly_score(demand_arr, supply_matrix, mix_fracs, pf)
                        best_sp = 0

                        if score < target:
                            # Try storage levels
                            for sp in [5, 10, 15, 20]:
                                score_ws = fast_score_with_storage(
                                    demand_arr, supply_matrix, mix_fracs, pf, sp
                                )
                                if score_ws >= target:
                                    score = score_ws
                                    best_sp = sp
                                    break

                        if score >= target:
                            cost = update_best(rcombo, p, best_sp, score)
                            phase2.append((cost, rcombo, score, best_sp, p))

    # ── Phase 3: Fine-tune (1% mix, ±2% procurement, refined storage) ──
    all_phase2 = phase2 if phase2 else top
    all_phase2.sort(key=lambda x: x[0])
    finalists = [c for c in all_phase2 if c[0] <= best_cost * 1.05][:5]

    seen2 = set()
    for _, combo, _, sp_base, proc in finalists:
        for p_d in range(-2, 3):
            p = proc + p_d
            if p < 70 or p > max_proc:
                continue
            pf = p / 100.0

            for cf_d in range(-2, 3):
                for sol_d in range(-2, 3):
                    for wnd_d in range(-2, 3):
                        cf = combo['clean_firm'] + cf_d
                        sol = combo['solar'] + sol_d
                        wnd = combo['wind'] + wnd_d
                        hyd = 100 - cf - sol - wnd
                        if cf < 0 or sol < 0 or wnd < 0 or hyd < 0 or hyd > hydro_cap:
                            continue
                        if cf > 100 or sol > 100 or wnd > 100:
                            continue

                        key = (cf, sol, wnd, p)
                        if key in seen2:
                            continue
                        seen2.add(key)

                        rcombo = {'clean_firm': cf, 'solar': sol, 'wind': wnd, 'hydro': hyd}
                        mix_fracs = np.array([rcombo[rt] / 100.0 for rt in RESOURCE_TYPES])

                        # Try no-storage first
                        score = fast_hourly_score(demand_arr, supply_matrix, mix_fracs, pf)
                        best_sp = 0
                        best_score = score

                        # Try storage in 2% steps for finer tuning
                        for sp in range(2, 22, 2):
                            score_ws = fast_score_with_storage(
                                demand_arr, supply_matrix, mix_fracs, pf, sp
                            )
                            if score_ws > best_score:
                                best_score = score_ws
                                best_sp = sp
                            if score_ws >= target and best_sp == sp:
                                break  # Got enough, stop adding storage cost

                        if best_score >= target:
                            update_best(rcombo, p, best_sp, best_score)

    return best_result


def compute_peak_gap(demand_norm, supply_profiles, resource_pcts, procurement_pct,
                     storage_dispatch_pct, anomaly_hours):
    """Compute peak single-hour gap % after storage."""
    score, hourly_detail, dispatch_profile, _ = compute_hourly_matching(
        demand_norm, supply_profiles, resource_pcts, procurement_pct, storage_dispatch_pct
    )
    peak_gap = 0.0
    for h in range(H):
        if h in anomaly_hours:
            continue
        d = hourly_detail[h]
        disp = dispatch_profile[h]
        residual_gap = max(0, d['gap'] - disp)
        if d['demand'] > 0:
            gap_pct = (residual_gap / d['demand']) * 100
            if gap_pct > peak_gap:
                peak_gap = gap_pct
    return round(peak_gap, 1)


def compute_compressed_day(demand_norm, supply_profiles, resource_pcts, procurement_pct,
                           storage_dispatch_pct):
    """Build compressed day profile for visualization."""
    procurement_factor = procurement_pct / 100.0

    _, hourly_detail, dispatch_profile, charge_profile = compute_hourly_matching(
        demand_norm, supply_profiles, resource_pcts, procurement_pct, storage_dispatch_pct
    )

    demand_sums = [0.0] * 24
    supply_by_type = {rt: [0.0]*24 for rt in RESOURCE_TYPES}
    gap_sums = [0.0] * 24
    surplus_sums = [0.0] * 24
    storage_dispatch_sums = [0.0] * 24
    storage_charge_sums = [0.0] * 24

    for h in range(H):
        hod = h % 24
        d = hourly_detail[h]
        demand_sums[hod] += d['demand']

        # Supply by type
        for rtype, pct in resource_pcts.items():
            if pct <= 0:
                continue
            type_supply = procurement_factor * (pct / 100.0) * supply_profiles[rtype][h]
            supply_by_type[rtype][hod] += type_supply

        storage_dispatch_sums[hod] += dispatch_profile[h]
        storage_charge_sums[hod] += charge_profile[h]

    # Match supply to demand per hour-of-day
    # Match order: clean_firm, hydro, wind, solar (same baseload-first concept)
    match_order = ['clean_firm', 'hydro', 'wind', 'solar']
    cut_order = list(reversed(match_order))  # solar cut first

    matched_by_type = {rt: [0.0]*24 for rt in RESOURCE_TYPES}
    surplus_by_type = {rt: [0.0]*24 for rt in RESOURCE_TYPES}
    matched_by_type['storage'] = [0.0]*24

    for hod in range(24):
        remaining = demand_sums[hod]
        for rtype in match_order:
            avail = supply_by_type[rtype][hod]
            matched = min(remaining, avail)
            matched_by_type[rtype][hod] = matched
            surplus_by_type[rtype][hod] = avail - matched
            remaining -= matched

        # Storage dispatch fills remaining gap
        disp = min(remaining, storage_dispatch_sums[hod])
        matched_by_type['storage'][hod] = disp
        remaining -= disp

        gap_sums[hod] = max(0, remaining)

        # Reduce surplus by charging
        rem_charge = storage_charge_sums[hod]
        for rtype in cut_order:
            if rem_charge <= 0:
                break
            absorb = min(surplus_by_type[rtype][hod], rem_charge)
            surplus_by_type[rtype][hod] -= absorb
            rem_charge -= absorb

    # Total surplus per hour-of-day
    for hod in range(24):
        surplus_sums[hod] = sum(surplus_by_type[rt][hod] for rt in RESOURCE_TYPES)

    return {
        'demand': [round(v, 4) for v in demand_sums],
        'matched': {rt: [round(v, 4) for v in matched_by_type[rt]] for rt in list(RESOURCE_TYPES) + ['storage']},
        'surplus': {rt: [round(v, 4) for v in surplus_by_type[rt]] for rt in RESOURCE_TYPES},
        'gap': [round(v, 4) for v in gap_sums],
        'storage_charge': [round(v, 4) for v in storage_charge_sums],
        'total_surplus': [round(v, 4) for v in surplus_sums],
    }


def compute_costs(iso, resource_pcts, procurement_pct, storage_dispatch_pct,
                   hourly_match_score, demand_norm, supply_profiles):
    """
    Compute blended cost of energy and incremental cost above baseline.

    Cost model:
      - Resources up to the existing grid mix share → priced at wholesale market rate
      - Resources above the existing grid mix share → priced at new-build LCOE
      - Storage → priced at storage LCOS for dispatched MWh
      - Curtailment inflates effective cost: we pay for ALL procured MWh but only
        get credit for USEFUL (matched) MWh, so over-procurement raises $/useful-MWh

    Returns dict with cost metrics in $/MWh (per MWh of demand served).
    """
    wholesale = WHOLESALE_PRICES[iso]
    lcoe = REGIONAL_LCOE[iso]
    grid_shares = GRID_MIX_SHARES[iso]

    procurement_factor = procurement_pct / 100.0

    # Total procured MWh by resource (as % of annual demand)
    # resource_pcts sum to 100 and represent share of PROCURED amount
    # So actual MWh as fraction of demand = procurement_factor × (pct/100)
    resource_costs = {}
    total_cost_per_demand = 0.0  # Total cost expressed as $/MWh-of-demand

    for rtype in RESOURCE_TYPES:
        pct = resource_pcts.get(rtype, 0)
        if pct <= 0:
            resource_costs[rtype] = {'existing_share': 0, 'new_share': 0, 'cost': 0}
            continue

        # This resource's total MWh as fraction of annual demand
        resource_fraction = procurement_factor * (pct / 100.0)
        # In percentage terms (of annual demand)
        resource_pct_of_demand = resource_fraction * 100.0

        # Existing share is priced at wholesale
        existing_share = grid_shares.get(rtype, 0)
        existing_pct = min(resource_pct_of_demand, existing_share)
        new_pct = max(0, resource_pct_of_demand - existing_share)

        # Cost in $/MWh-of-demand for this resource
        # (existing_pct/100 × wholesale) + (new_pct/100 × lcoe)
        cost_per_demand = (existing_pct / 100.0 * wholesale) + (new_pct / 100.0 * lcoe.get(rtype, 0))

        # Hydro: always at wholesale (existing resource, no new-build)
        if rtype == 'hydro':
            cost_per_demand = resource_pct_of_demand / 100.0 * wholesale

        resource_costs[rtype] = {
            'total_pct_of_demand': round(resource_pct_of_demand, 1),
            'existing_pct': round(existing_pct, 1),
            'new_pct': round(new_pct, 1),
            'cost_per_demand_mwh': round(cost_per_demand, 2),
        }
        total_cost_per_demand += cost_per_demand

    # Storage cost: only pay for dispatched MWh
    # storage_dispatch_pct is % of annual demand dispatched
    storage_cost_per_demand = (storage_dispatch_pct / 100.0) * lcoe['storage']
    resource_costs['storage'] = {
        'dispatch_pct': round(storage_dispatch_pct, 1),
        'cost_per_demand_mwh': round(storage_cost_per_demand, 2),
    }
    total_cost_per_demand += storage_cost_per_demand

    # Effective cost per USEFUL MWh (accounting for curtailment)
    # hourly_match_score is the % of demand actually matched
    # But we're buying procurement_factor × demand worth of energy
    # Effective cost = total_cost / (matched fraction of demand)
    matched_fraction = hourly_match_score / 100.0 if hourly_match_score > 0 else 1.0
    effective_cost_per_useful_mwh = total_cost_per_demand / matched_fraction

    # Baseline cost: 100% wholesale (what you'd pay for grid power)
    baseline_cost = wholesale

    # Incremental cost above baseline
    incremental = effective_cost_per_useful_mwh - baseline_cost

    return {
        'resource_costs': resource_costs,
        'total_cost_per_demand_mwh': round(total_cost_per_demand, 2),
        'effective_cost_per_useful_mwh': round(effective_cost_per_useful_mwh, 2),
        'baseline_wholesale_cost': wholesale,
        'incremental_above_baseline': round(incremental, 2),
        'curtailment_pct': round((procurement_factor - matched_fraction) / procurement_factor * 100, 1)
            if procurement_factor > 0 else 0,
    }


def main():
    start_time = time.time()
    demand_data, gen_profiles = load_data()

    all_results = {
        'config': {
            'data_year': DATA_YEAR,
            'storage_duration': '4h',
            'storage_efficiency': STORAGE_EFFICIENCY,
            'hydro_caps': HYDRO_CAPS,
            'resource_types': RESOURCE_TYPES,
            'thresholds': THRESHOLDS,
            'wholesale_prices': WHOLESALE_PRICES,
            'regional_lcoe': REGIONAL_LCOE,
            'grid_mix_shares': GRID_MIX_SHARES,
        },
        'results': {},
    }

    for iso in ISOS:
        print(f"\n{'='*70}")
        print(f"  {ISO_LABELS[iso]}")
        print(f"{'='*70}")

        demand_norm = demand_data[iso]['normalized']
        supply_profiles = get_supply_profiles(iso, gen_profiles)
        hydro_cap = HYDRO_CAPS[iso]
        anomaly_hours = find_anomaly_hours(iso, gen_profiles)
        np_profiles = prepare_numpy_profiles(demand_norm, supply_profiles)

        iso_results = {
            'iso': iso,
            'label': ISO_LABELS[iso],
            'annual_demand_mwh': demand_data[iso]['total_annual_mwh'],
            'peak_demand_mw': demand_data[iso]['peak_mw'],
            'sweep': [],  # All procurement levels tested
            'thresholds': {},  # Results at each target threshold
        }

        # ── SWEEP: max-matching runs for the sweep chart visualization ──
        print(f"  Sweep: max-matching at key procurement levels...")
        sweep_results = {}
        for proc_pct in list(range(70, 130, 10)) + list(range(140, 520, 20)):
            result = optimize_at_procurement_level(
                iso, demand_norm, supply_profiles, proc_pct, hydro_cap,
                np_profiles=np_profiles
            )
            sweep_results[proc_pct] = result
            score = result['hourly_match_score']
            print(f"    {proc_pct}%: {score}% match")
            if score >= 99.95 and proc_pct >= 140:
                break  # No need to go further

        # Build sweep with costs and peak gap
        for p in sorted(sweep_results.keys()):
            r = sweep_results[p]
            peak_gap = compute_peak_gap(
                demand_norm, supply_profiles, r['resource_mix'],
                r['procurement_pct'], r['storage_dispatch_pct'], anomaly_hours
            )
            r['peak_gap_pct'] = peak_gap
            costs = compute_costs(
                iso, r['resource_mix'], r['procurement_pct'],
                r['storage_dispatch_pct'], r['hourly_match_score'],
                demand_norm, supply_profiles
            )
            r['costs'] = costs
            iso_results['sweep'].append(r)

        # ── THRESHOLDS: co-optimize cost + matching for each target ──
        print(f"\n  Cost-optimizing thresholds...")
        for threshold in THRESHOLDS:
            print(f"    Target {threshold}%...")
            result = optimize_for_threshold(
                iso, demand_norm, supply_profiles, threshold, hydro_cap
            )
            if result:
                # Compute peak gap and costs
                peak_gap = compute_peak_gap(
                    demand_norm, supply_profiles, result['resource_mix'],
                    result['procurement_pct'], result['storage_dispatch_pct'], anomaly_hours
                )
                result['peak_gap_pct'] = peak_gap
                costs = compute_costs(
                    iso, result['resource_mix'], result['procurement_pct'],
                    result['storage_dispatch_pct'], result['hourly_match_score'],
                    demand_norm, supply_profiles
                )
                result['costs'] = costs

                # Generate compressed day
                cdp = compute_compressed_day(
                    demand_norm, supply_profiles, result['resource_mix'],
                    result['procurement_pct'], result['storage_dispatch_pct']
                )
                iso_results['thresholds'][str(threshold)] = {
                    **result,
                    'compressed_day': cdp,
                }
                cost_info = result['costs']
                mix = result['resource_mix']
                print(f"      => procurement={result['procurement_pct']}%, "
                      f"score={result['hourly_match_score']}%, "
                      f"mix=CF{mix['clean_firm']}/Sol{mix['solar']}/Wnd{mix['wind']}/Hyd{mix['hydro']} "
                      f"cost=${cost_info['effective_cost_per_useful_mwh']}/MWh "
                      f"(+${cost_info['incremental_above_baseline']}/MWh)")
            else:
                print(f"      => No feasible solution found for {threshold}%")

        all_results['results'][iso] = iso_results

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), 'dashboard', 'overprocure_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f)

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  Complete in {elapsed:.0f}s. Saved to: {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1024:.0f} KB")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
