#!/usr/bin/env python3
"""
Step 1: Physics Feasible Space (PFS) Generator
===============================================
Generates the Physical Feasibility Space (PFS) for hourly CFE matching.
Physics only — no cost model. Cost sensitivities applied in Step 3.

Pipeline position: Step 1 of 4
  Step 1 — PFS Generator (this file)
  Step 2 — Efficient Frontier (EF) extraction
  Step 3 — Cost optimization
  Step 4 — Post-processing

Key features:
  - 4D resource space: clean_firm (absorbs CCS), solar, wind, hydro
  - 13 thresholds: 50, 60, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 100
  - Adaptive grid search: 5% → 1% refinement
  - Pareto frontier: 3-5 points per threshold×ISO (procurement/storage tradeoff)
  - Numba JIT-compiled scoring functions
  - Parallel ISO execution (multiprocessing)
  - Vectorized batch mix evaluation

Output: data/physics_cache_v4.parquet  (21.4M rows — the PFS)

Resource types (4D optimization):
  - Clean Firm: nuclear (seasonal-derated) + CCS-CCGT (flat baseload)
    Sub-allocation determined by cost model in Step 3.
    Physics uses nuclear-derated profile (conservative — CCS only improves matching).
  - Solar: EIA 2021-2025 averaged hourly profile (DST-aware)
  - Wind: EIA 2021-2025 averaged hourly profile
  - Hydro: EIA 2021-2025 averaged hourly profile (capped by region, existing only)

Storage (not part of mix %, swept as separate dimensions):
  - Battery (4hr): Li-ion, 85% RTE, daily-cycle dispatch
  - Battery (8hr): Li-ion, 85% RTE, daily-cycle dispatch (power = cap/8hr)
  - LDES: 100hr iron-air, 50% RTE, 7-day rolling window dispatch
"""

import json
import os
import sys
import time
import numpy as np
from multiprocessing import Pool, cpu_count
from datetime import datetime, timezone

# Numba JIT — try import, fall back to pure NumPy
try:
    from numba import njit, prange
    HAS_NUMBA = True
    print("  Numba available — JIT-compiled scoring enabled")
except ImportError:
    HAS_NUMBA = False
    print("  Numba not available — using NumPy fallback")
    def njit(*args, **kwargs):
        """No-op decorator when Numba unavailable."""
        def wrapper(fn):
            return fn
        if args and callable(args[0]):
            return args[0]
        return wrapper
    prange = range

# PyArrow for Parquet output
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
DATA_YEAR = '2025'
PROFILE_YEARS = ['2021', '2022', '2023', '2024', '2025']
H = 8760
LEAP_FEB29_START = 1416

# Storage
BATTERY_EFFICIENCY = 0.85
BATTERY_DURATION_HOURS = 4
BATTERY8_EFFICIENCY = 0.85
BATTERY8_DURATION_HOURS = 8
LDES_EFFICIENCY = 0.50
LDES_DURATION_HOURS = 100
LDES_WINDOW_DAYS = 7

# 4D resource types (CCS merged into clean_firm)
RESOURCE_TYPES = ['clean_firm', 'solar', 'wind', 'hydro']
N_RESOURCES = len(RESOURCE_TYPES)

# Regions
ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']
ISO_LABELS = {
    'CAISO': 'CAISO (California)',
    'ERCOT': 'ERCOT (Texas)',
    'PJM': 'PJM (Mid-Atlantic)',
    'NYISO': 'NYISO (New York)',
    'NEISO': 'NEISO (New England)',
    'MISO': 'MISO (Midwest)',
    'SPP': 'SPP (Central)',
}

HYDRO_CAPS = {
    'CAISO': 9.5, 'ERCOT': 0.1, 'PJM': 1.8, 'NYISO': 15.9, 'NEISO': 4.4,
    'MISO': 1.6, 'SPP': 4.3,
}

# 13 thresholds (v4.0: added 50, 60, 70)
THRESHOLDS = [50, 60, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 100]

# Threshold-adaptive procurement bounds (Decision 3C, expanded)
# 90-99%: capped at 250% per user direction (high enough for extreme renewables,
# but not wastefully wide). 100%: pushed to 500% for perfect hourly matching.
PROCUREMENT_BOUNDS = {
    50:   (50, 150),
    60:   (60, 150),
    70:   (70, 175),
    75:   (75, 200),
    80:   (80, 200),
    85:   (85, 225),
    87.5: (87, 250),
    90:   (90, 250),
    92.5: (92, 250),
    95:   (95, 250),
    97.5: (100, 250),
    99:   (100, 250),
    100:  (100, 350),
}

# Nuclear seasonal derate
NUCLEAR_SHARE_OF_CLEAN_FIRM = {
    'CAISO': 0.70, 'ERCOT': 1.0, 'PJM': 1.0, 'NYISO': 1.0, 'NEISO': 1.0,
    'MISO': 1.0, 'SPP': 1.0,
}

NUCLEAR_MONTHLY_CF = {
    'CAISO': {1: 0.94, 2: 0.94, 3: 0.85, 4: 0.75, 5: 0.80, 6: 0.99,
              7: 1.0, 8: 1.0, 9: 0.90, 10: 0.78, 11: 0.82, 12: 0.94},
    'ERCOT': {1: 1.0, 2: 1.0, 3: 0.90, 4: 0.80, 5: 0.89, 6: 0.97,
              7: 0.97, 8: 0.96, 9: 0.88, 10: 0.79, 11: 0.85, 12: 1.0},
    'PJM':   {1: 1.0, 2: 1.0, 3: 0.92, 4: 0.85, 5: 0.87, 6: 0.98,
              7: 0.99, 8: 0.97, 9: 0.93, 10: 0.89, 11: 0.91, 12: 1.0},
    'NYISO': {1: 1.0, 2: 1.0, 3: 0.88, 4: 0.78, 5: 0.81, 6: 0.95,
              7: 0.96, 8: 0.94, 9: 0.85, 10: 0.75, 11: 0.79, 12: 1.0},
    'NEISO': {1: 1.0, 2: 0.99, 3: 0.92, 4: 0.83, 5: 0.88, 6: 0.96,
              7: 0.97, 8: 0.95, 9: 0.88, 10: 0.82, 11: 0.85, 12: 1.0},
    # MISO: large inland fleet (11 plants), spring refueling pattern similar to PJM
    'MISO':  {1: 1.0, 2: 1.0, 3: 0.91, 4: 0.83, 5: 0.86, 6: 0.97,
              7: 0.98, 8: 0.96, 9: 0.91, 10: 0.85, 11: 0.88, 12: 1.0},
    # SPP: single plant (Wolf Creek), higher refueling impact
    'SPP':   {1: 1.0, 2: 1.0, 3: 0.88, 4: 0.78, 5: 0.85, 6: 0.96,
              7: 0.97, 8: 0.95, 9: 0.87, 10: 0.80, 11: 0.84, 12: 1.0},
}

# Checkpoint directory
CHECKPOINT_DIR = os.path.join(DATA_DIR, 'checkpoints_v4')

# Mix-level checkpoint interval: save progress every N outer-loop mixes
# within a single threshold. Protects against mid-threshold crashes.
MIX_CHECKPOINT_INTERVAL = 500

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING (preserved from v3.x with minimal changes)
# ══════════════════════════════════════════════════════════════════════════════

def _remove_leap_day(profile_8784):
    if len(profile_8784) <= H:
        return list(profile_8784[:H])
    return list(profile_8784[:LEAP_FEB29_START]) + list(profile_8784[LEAP_FEB29_START + 24:H + 24])


def _average_profiles(yearly_profiles):
    if not yearly_profiles:
        return [0.0] * H
    stacked = np.array(yearly_profiles, dtype=np.float64)
    return np.mean(stacked, axis=0).tolist()


def _validate_demand_profile(iso, year, profile):
    arr = np.array(profile[:H])
    for hod in range(24):
        vals = arr[hod::24]
        median = np.median(vals)
        if median > 0 and vals.max() > 100 * median:
            print(f"  WARNING: {iso} {year} demand excluded — outlier detected")
            return False
    return True


def _qa_qc_profiles(demand_data, gen_profiles):
    print("\n  QA/QC: Validating profile shapes...")
    issues = []
    for iso in ISOS:
        norm = demand_data[iso]['normalized']
        arr = np.array(norm[:H])
        total = arr.sum()
        if abs(total - 1.0) > 0.01:
            issues.append(f"  FAIL: {iso} demand sum = {total:.4f} (expected ~1.0)")
        if arr.min() < 0:
            issues.append(f"  FAIL: {iso} demand has negative values")
    if issues:
        fatal = [i for i in issues if 'FAIL' in i]
        if fatal:
            raise ValueError(f"QA/QC failed with {len(fatal)} fatal issues")
        for issue in issues:
            print(f"    {issue}")
    else:
        print("    All profiles passed QA/QC")
    print()


def load_data():
    """Load demand profiles, generation profiles, emission rates, and fossil mix."""
    print("Loading data...")

    with open(os.path.join(DATA_DIR, 'eia_demand_profiles.json')) as f:
        demand_raw = json.load(f)
    with open(os.path.join(DATA_DIR, 'eia_generation_profiles.json')) as f:
        gen_raw = json.load(f)
    with open(os.path.join(DATA_DIR, 'egrid_emission_rates.json')) as f:
        emission_rates = json.load(f)
    with open(os.path.join(DATA_DIR, 'eia_fossil_mix.json')) as f:
        fossil_mix = json.load(f)

    # Average generation profiles across PROFILE_YEARS
    gen_profiles = {}
    for iso in ISOS:
        iso_raw = gen_raw.get(iso, {})
        available_years = [y for y in PROFILE_YEARS if y in iso_raw]
        if not available_years:
            raise ValueError(f"No generation profile years found for {iso}")
        all_rtypes = set()
        for y in available_years:
            all_rtypes.update(iso_raw[y].keys())
        gen_profiles[iso] = {}
        for rtype in all_rtypes:
            yearly = []
            for y in available_years:
                raw = iso_raw[y].get(rtype)
                if raw is None:
                    continue
                if len(raw) > H:
                    raw = _remove_leap_day(raw)
                else:
                    raw = list(raw[:H])
                yearly.append(raw)
            if yearly:
                gen_profiles[iso][rtype] = _average_profiles(yearly)
        print(f"  {iso}: gen profiles averaged over {len(available_years)} years")

    # Average demand profiles; use 2025 actuals for scalars
    demand_data = {}
    for iso in ISOS:
        iso_data = demand_raw.get(iso, {})
        year_keys = [k for k in iso_data.keys() if k.isdigit()]
        if not year_keys and 'normalized' in iso_data:
            demand_data[iso] = iso_data
            continue

        available_years = [y for y in PROFILE_YEARS if y in iso_data]
        if not available_years:
            raise ValueError(f"No demand data years found for {iso}")
        yearly_norms = []
        valid_years = []
        for y in available_years:
            raw = iso_data[y].get('normalized', [])
            if len(raw) > H:
                raw = _remove_leap_day(raw)
            else:
                raw = list(raw[:H])
            if _validate_demand_profile(iso, y, raw):
                yearly_norms.append(raw)
                valid_years.append(y)
        if not yearly_norms:
            raise ValueError(f"All demand data years excluded for {iso}")

        avg_norm = _average_profiles(yearly_norms)
        actuals_year = DATA_YEAR if DATA_YEAR in iso_data else available_years[-1]
        demand_data[iso] = {
            'normalized': avg_norm,
            'total_annual_mwh': iso_data[actuals_year]['total_annual_mwh'],
            'peak_mw': iso_data[actuals_year]['peak_mw'],
        }
        print(f"  {iso}: demand shape averaged over {len(valid_years)} years, scalars from {actuals_year}")

    print("  Data loaded.")
    _qa_qc_profiles(demand_data, gen_profiles)
    return demand_data, gen_profiles, emission_rates, fossil_mix


def get_supply_profiles(iso, gen_profiles):
    """Get generation profiles for 4D resource types (v4.0: no separate CCS).

    Returns dict mapping resource type to list of H floats (normalized shape profiles).
    Uses numpy for vectorized DST correction and profile construction.
    """
    profiles = {}

    # Clean firm = nuclear seasonal-derated baseload
    # CCS sub-allocation handled by cost model; physics uses nuclear profile (conservative)
    nuc_share = NUCLEAR_SHARE_OF_CLEAN_FIRM.get(iso, 1.0)
    geo_share = 1.0 - nuc_share
    monthly_cf = NUCLEAR_MONTHLY_CF.get(iso, {m: 1.0 for m in range(1, 13)})
    month_hours = np.array([744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744])
    month_cfs = np.array([nuc_share * monthly_cf.get(m + 1, 1.0) + geo_share
                          for m in range(12)])
    cf_profile = np.repeat(month_cfs, month_hours)[:H] / H
    if len(cf_profile) < H:
        cf_profile = np.pad(cf_profile, (0, H - len(cf_profile)),
                            constant_values=1.0 / H)
    profiles['clean_firm'] = cf_profile.tolist()

    # Solar (with vectorized DST-aware nighttime correction)
    if iso == 'NYISO':
        p = gen_profiles[iso].get('solar_proxy')
        if not p:
            p = gen_profiles['NEISO'].get('solar')
        solar_arr = np.array(p[:H], dtype=np.float64)
    else:
        solar_arr = np.array(gen_profiles[iso].get('solar', [0.0] * H)[:H],
                             dtype=np.float64)

    STD_UTC_OFFSETS = {'CAISO': 8, 'ERCOT': 6, 'PJM': 5, 'NYISO': 5, 'NEISO': 5,
                       'MISO': 6, 'SPP': 6}
    DST_START_DAY, DST_END_DAY = 69, 307
    local_start, local_end = 6, 19
    std_off = STD_UTC_OFFSETS.get(iso, 5)

    # Vectorized: build daylight mask for all 8760 hours at once
    hours = np.arange(H)
    days = hours // 24
    hour_of_day = hours % 24
    is_dst = (days >= DST_START_DAY) & (days < DST_END_DAY)
    utc_off = std_off - is_dst.astype(int)
    utc_start = (local_start + utc_off) % 24
    utc_end = (local_end + utc_off) % 24
    # Normal case: utc_start <= utc_end → daylight when start <= h <= end
    # Wrap case: utc_start > utc_end → daylight when h >= start OR h <= end
    normal_mask = utc_start <= utc_end
    is_daylight = np.where(
        normal_mask,
        (hour_of_day >= utc_start) & (hour_of_day <= utc_end),
        (hour_of_day >= utc_start) | (hour_of_day <= utc_end)
    )
    solar_arr[~is_daylight] = 0.0
    if len(solar_arr) < H:
        solar_arr = np.pad(solar_arr, (0, H - len(solar_arr)))
    profiles['solar'] = solar_arr.tolist()

    # Wind
    profiles['wind'] = list(gen_profiles[iso].get('wind', [0.0] * H)[:H])

    # Hydro
    profiles['hydro'] = list(gen_profiles[iso].get('hydro', [0.0] * H)[:H])

    # Ensure all profiles are exactly H hours, no negatives (vectorized)
    for rtype in RESOURCE_TYPES:
        arr = np.array(profiles[rtype], dtype=np.float64)
        if len(arr) > H:
            arr = arr[:H]
        elif len(arr) < H:
            arr = np.pad(arr, (0, H - len(arr)))
        np.maximum(arr, 0.0, out=arr)
        profiles[rtype] = arr.tolist()

    return profiles


def prepare_numpy_profiles(demand_norm, supply_profiles):
    """Convert to numpy arrays + build supply matrix (4, 8760)."""
    demand_arr = np.array(demand_norm[:H], dtype=np.float64)
    supply_matrix = np.stack([
        np.array(supply_profiles[rt][:H], dtype=np.float64)
        for rt in RESOURCE_TYPES
    ])  # shape (4, 8760)
    return demand_arr, supply_matrix


# ══════════════════════════════════════════════════════════════════════════════
# SCORING FUNCTIONS — Numba JIT compiled
# ══════════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def _compute_max_daily_battery_soc(demand, supply_row, procurement,
                                    batt_eff, batt_duration):
    """Compute max daily SOC a battery would reach with unlimited capacity.

    This is the SOC screening formula: for a given (mix, procurement), find the
    absolute maximum SOC any battery (4hr or 8hr) would reach on any day of the
    year. Storage levels above this are pure waste — the battery would never fill.

    Also returns total annual dispatch at unlimited capacity (useful for checking
    if any dispatch is possible at all).

    Returns: (max_soc, total_dispatch)
        max_soc: maximum state of charge reached on any day (normalized units)
        total_dispatch: total energy dispatched over the year (normalized units)
    """
    max_soc = 0.0
    total_dispatch = 0.0
    for day in range(365):
        ds = day * 24
        stored = 0.0
        # Charge phase: accumulate all surplus (no capacity limit)
        for h in range(24):
            s = procurement * supply_row[ds + h] - demand[ds + h]
            if s > 0:
                stored += s  # No power limit — we want max possible SOC
        if stored > max_soc:
            max_soc = stored
        # Discharge: all stored × RTE into gaps (no power limit)
        avail = stored * batt_eff
        for h in range(24):
            g = demand[ds + h] - procurement * supply_row[ds + h]
            if g > 0 and avail > 0:
                d = g if g < avail else avail
                total_dispatch += d
                avail -= d
    return max_soc, total_dispatch


@njit(cache=True)
def _compute_max_ldes_soc(demand, supply_row, procurement,
                           ldes_eff, ldes_window_hours):
    """Compute max SOC LDES would reach with unlimited capacity.

    Returns: (max_soc, total_dispatch)
    """
    max_soc = 0.0
    total_dispatch = 0.0
    soc = 0.0
    n_windows = (8760 + ldes_window_hours - 1) // ldes_window_hours
    for w in range(n_windows):
        ws = w * ldes_window_hours
        we = ws + ldes_window_hours
        if we > 8760:
            we = 8760
        # Charge phase
        for h in range(ws, we):
            s = procurement * supply_row[h] - demand[h]
            if s > 0:
                soc += s
        if soc > max_soc:
            max_soc = soc
        # Discharge phase
        for h in range(ws, we):
            g = demand[h] - procurement * supply_row[h]
            if g > 0 and soc > 0:
                ae = soc * ldes_eff
                d = g if g < ae else ae
                total_dispatch += d
                soc -= d / ldes_eff
    return max_soc, total_dispatch


@njit(cache=True)
def _score_with_all_storage(demand, supply_row, procurement,
                            batt_capacity, batt_power, batt_eff,
                            batt8_capacity, batt8_power, batt8_eff,
                            ldes_capacity, ldes_power, ldes_eff,
                            ldes_window_hours):
    """Hourly score + battery4 (daily) + battery8 (daily) + LDES (multi-day).

    Dispatch order: battery4 first (cheapest short-duration), then battery8,
    then LDES on post-battery residual. Each phase updates residual surplus/gap.
    """
    supply = np.empty(8760)
    surplus = np.empty(8760)
    gap = np.empty(8760)
    for h in range(8760):
        s = procurement * supply_row[h]
        supply[h] = s
        d = demand[h]
        if s > d:
            surplus[h] = s - d
            gap[h] = 0.0
        else:
            surplus[h] = 0.0
            gap[h] = d - s

    base_matched = 0.0
    for h in range(8760):
        base_matched += min(demand[h], supply[h])

    # Phase 1: Battery 4hr daily cycle on residual surplus/gap
    batt_dispatched = 0.0
    residual_surplus = np.copy(surplus)
    residual_gap = np.copy(gap)

    if batt_capacity > 0:
        for day in range(365):
            ds = day * 24
            stored = 0.0
            for h in range(24):
                s = residual_surplus[ds + h]
                if s > 0 and stored < batt_capacity:
                    charge = s
                    if charge > batt_power:
                        charge = batt_power
                    remaining = batt_capacity - stored
                    if charge > remaining:
                        charge = remaining
                    stored += charge
                    residual_surplus[ds + h] -= charge
            available = stored * batt_eff
            for h in range(24):
                g = residual_gap[ds + h]
                if g > 0 and available > 0:
                    discharge = g
                    if discharge > batt_power:
                        discharge = batt_power
                    if discharge > available:
                        discharge = available
                    batt_dispatched += discharge
                    available -= discharge
                    residual_gap[ds + h] -= discharge

    # Phase 2: Battery 8hr daily cycle on post-4hr residual
    batt8_dispatched = 0.0
    if batt8_capacity > 0:
        for day in range(365):
            ds = day * 24
            stored = 0.0
            for h in range(24):
                s = residual_surplus[ds + h]
                if s > 0 and stored < batt8_capacity:
                    charge = s
                    if charge > batt8_power:
                        charge = batt8_power
                    remaining = batt8_capacity - stored
                    if charge > remaining:
                        charge = remaining
                    stored += charge
                    residual_surplus[ds + h] -= charge
            available = stored * batt8_eff
            for h in range(24):
                g = residual_gap[ds + h]
                if g > 0 and available > 0:
                    discharge = g
                    if discharge > batt8_power:
                        discharge = batt8_power
                    if discharge > available:
                        discharge = available
                    batt8_dispatched += discharge
                    available -= discharge
                    residual_gap[ds + h] -= discharge

    # Phase 3: LDES multi-day rolling window on post-battery residual
    ldes_dispatched = 0.0
    if ldes_capacity > 0:
        soc = 0.0
        n_windows = (8760 + ldes_window_hours - 1) // ldes_window_hours
        for w in range(n_windows):
            ws = w * ldes_window_hours
            we = ws + ldes_window_hours
            if we > 8760:
                we = 8760
            for h in range(ws, we):
                s = residual_surplus[h]
                if s > 0 and soc < ldes_capacity:
                    charge = s
                    if charge > ldes_power:
                        charge = ldes_power
                    remaining = ldes_capacity - soc
                    if charge > remaining:
                        charge = remaining
                    soc += charge
            for h in range(ws, we):
                g = residual_gap[h]
                if g > 0 and soc > 0:
                    available_e = soc * ldes_eff
                    discharge = g
                    if discharge > ldes_power:
                        discharge = ldes_power
                    if discharge > available_e:
                        discharge = available_e
                    ldes_dispatched += discharge
                    soc -= discharge / ldes_eff

    return base_matched + batt_dispatched + batt8_dispatched + ldes_dispatched


# ══════════════════════════════════════════════════════════════════════════════
# BATCH EVALUATION — vectorized for grid search + Numba parallel for storage
# ══════════════════════════════════════════════════════════════════════════════

def batch_hourly_scores(demand_arr, supply_matrix, mix_batch, procurement):
    """Evaluate N mixes at once. mix_batch shape (N, 4), returns (N,) scores.

    Uses total matched energy metric: sum(min(demand, supply)) for consistency
    with storage scoring functions.
    """
    # (N, 4) @ (4, 8760) = (N, 8760) — all mixes in one matrix multiply
    supply_batch = procurement * (mix_batch @ supply_matrix)
    matched = np.minimum(demand_arr, supply_batch)
    return matched.sum(axis=1)


@njit(cache=True)
def _batch_score_no_storage(demand, supply_rows, procurement, N):
    """Score N mixes without storage, using Numba parallel if available.

    Uses sum(min(demand, supply)) — total matched energy, consistent with
    _score_with_all_storage's base_matched computation.
    """
    scores = np.empty(N, dtype=np.float64)
    for i in prange(N):
        total = 0.0
        for h in range(8760):
            s = procurement * supply_rows[i, h]
            d = demand[h]
            if s < d:
                total += s
            else:
                total += d
        scores[i] = total
    return scores


@njit(cache=True)
def _batch_score_storage(demand, supply_rows, procurement, N,
                         batt_cap, batt_pow, batt_eff,
                         batt8_cap, batt8_pow, batt8_eff,
                         ldes_cap, ldes_pow, ldes_eff,
                         ldes_window_hours):
    """Evaluate N mixes in parallel at a single (procurement, storage) config.

    Each mix is independent — Numba prange distributes across CPU cores.
    """
    scores = np.empty(N, dtype=np.float64)
    for i in prange(N):
        scores[i] = _score_with_all_storage(
            demand, supply_rows[i], procurement,
            batt_cap, batt_pow, batt_eff,
            batt8_cap, batt8_pow, batt8_eff,
            ldes_cap, ldes_pow, ldes_eff,
            ldes_window_hours)
    return scores


# ══════════════════════════════════════════════════════════════════════════════
# GRID SEARCH — 4D adaptive (Decision 1C)
# ══════════════════════════════════════════════════════════════════════════════

def generate_4d_combos(hydro_cap, step=5, max_single=100):
    """Generate all 4D resource mixes summing to 100% with constraints.

    max_single=100 allows any single resource up to 100% of the mix,
    enabling extreme solar-only or wind-only portfolios (which at high
    procurement can reach 200%+ of demand).

    Returns numpy array of shape (N, 4) where columns are
    [clean_firm, solar, wind, hydro] as percentages.

    Uses vectorized numpy meshgrid instead of nested Python loops.
    """
    hydro_max = min(int(hydro_cap), max_single)
    hyd_vals = np.arange(0, hydro_max + 1, step)
    cf_vals = np.arange(0, max_single + 1, step)
    sol_vals = np.arange(0, max_single + 1, step)

    # Build all (cf, sol, hyd) triples via meshgrid
    cf_grid, sol_grid, hyd_grid = np.meshgrid(cf_vals, sol_vals, hyd_vals, indexing='ij')
    cf_flat = cf_grid.ravel()
    sol_flat = sol_grid.ravel()
    hyd_flat = hyd_grid.ravel()
    wnd_flat = 100 - cf_flat - sol_flat - hyd_flat

    # Filter: wind must be in [0, max_single]
    valid = (wnd_flat >= 0) & (wnd_flat <= max_single)
    combos = np.column_stack([cf_flat[valid], sol_flat[valid],
                              wnd_flat[valid], hyd_flat[valid]])
    return combos.astype(np.float64)


def generate_4d_combos_around(base_combo, hydro_cap, step=1, radius=2, max_single=100):
    """Generate 4D combos in neighborhood of base_combo."""
    combos = []
    seen = set()
    hydro_max = min(int(hydro_cap), max_single)

    base = [int(base_combo[i]) for i in range(4)]
    ranges = []
    for i, val in enumerate(base):
        cap = hydro_max if i == 3 else max_single
        lo = max(0, val - radius * step)
        hi = min(cap, val + radius * step)
        ranges.append(list(range(lo, hi + 1, step)))

    for cf in ranges[0]:
        for sol in ranges[1]:
            for wnd in ranges[2]:
                for hyd in ranges[3]:
                    if cf + sol + wnd + hyd == 100:
                        key = (cf, sol, wnd, hyd)
                        if key not in seen and all(v <= max_single for v in key):
                            seen.add(key)
                            combos.append([cf, sol, wnd, hyd])

    return np.array(combos, dtype=np.float64) if combos else np.empty((0, 4))


# Edge case seed mixes (4D: clean_firm, solar, wind, hydro)
# Includes extreme single-resource mixes to capture solar+storage, wind+storage outcomes
EDGE_CASE_SEEDS = [
    # Extreme solar (at high procurement, 100% solar = 200%+ of demand from solar alone)
    [0, 100, 0, 0],    # Pure solar
    [0, 95, 5, 0],     # Near-pure solar + wind
    [0, 95, 0, 5],     # Near-pure solar + hydro
    [0, 90, 10, 0],
    [0, 90, 5, 5],
    [5, 90, 5, 0],
    [5, 85, 10, 0],
    # Extreme wind
    [0, 0, 100, 0],    # Pure wind
    [0, 5, 95, 0],     # Near-pure wind + solar
    [0, 0, 95, 5],     # Near-pure wind + hydro
    [0, 10, 90, 0],
    [5, 5, 90, 0],
    [5, 10, 85, 0],
    # Solar-dominant
    [5, 70, 20, 5],
    [5, 75, 15, 5],
    [10, 80, 10, 0],
    # Wind-dominant
    [5, 20, 70, 5],
    [5, 15, 75, 5],
    [10, 10, 80, 0],
    # Balanced renewable + hydro
    [5, 40, 40, 15],
    [10, 45, 45, 0],
    # Clean firm dominant
    [60, 15, 15, 10],
    [70, 10, 10, 10],
    [80, 10, 10, 0],
    [90, 5, 5, 0],
    [100, 0, 0, 0],    # Pure clean firm
    [95, 5, 0, 0],
    # Moderate firm + renewables
    [50, 25, 25, 0],
    [40, 30, 30, 0],
    [30, 35, 35, 0],
    # Minimal firm
    [20, 40, 40, 0],
    [10, 45, 45, 0],
    [0, 50, 50, 0],    # Zero firm — pure renewables
]


def get_seed_combos(hydro_cap):
    """Return edge case seeds valid for this region's hydro cap."""
    valid = []
    seen = set()
    for seed in EDGE_CASE_SEEDS:
        if seed[3] > hydro_cap:
            continue
        key = tuple(seed)
        if key not in seen and sum(seed) == 100:
            seen.add(key)
            valid.append(seed)
    return np.array(valid, dtype=np.float64) if valid else np.empty((0, 4))


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZER CORE — per-threshold sweep
# ══════════════════════════════════════════════════════════════════════════════

def optimize_threshold(iso, threshold, demand_arr, supply_matrix, hydro_cap,
                       prev_pruning=None, cross_feasible_mixes=None,
                       flush_callback=None):
    """Find ALL feasible solutions for a single threshold × ISO.

    Uses cross-threshold pruning: mixes that were infeasible at a lower threshold
    are skipped. Each mix's min-feasible procurement from the previous threshold
    becomes the floor for the procurement sweep.

    Cross-threshold pollination: mixes proven feasible from cached results are
    excluded from the Phase 1b storage sweep (already known to work without
    needing the expensive storage search). Phase 1a still evaluates them to
    find the right procurement level.

    flush_callback: optional callable(candidates_so_far) called every 1M solutions
                    to save interim progress within a single threshold.

    Args:
        prev_pruning: dict from previous threshold with:
            - 'feasible_mixes': set of mix tuples that were feasible
            - 'min_proc': dict mapping mix_tuple -> minimum procurement that worked
            - 'all_mixes': set of all mix tuples tested
            If None, no pruning (first threshold).
        cross_feasible_mixes: set of mix tuples (cf, sol, wnd, hyd) already proven
            to meet this threshold from cached/lower-threshold solutions.

    Returns:
        (candidates, pruning_info) where pruning_info can be passed to next threshold
    """
    target = threshold / 100.0
    proc_min, proc_max = PROCUREMENT_BOUNDS.get(threshold, (70, 200))

    # Storage constants
    batt_eff = BATTERY_EFFICIENCY
    batt8_eff = BATTERY8_EFFICIENCY
    ldes_eff = LDES_EFFICIENCY
    ldes_window_hours = LDES_WINDOW_DAYS * 24

    # Storage levels to sweep — sub-percent granularity below saturation threshold.
    # Battery4/8 saturate at ~1.2% of annual demand (PJM binding case). LDES never
    # saturates (multi-day accumulation). Fine granularity finds right-sized capacity
    # to maximize utilization and minimize LCOS.
    batt_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                   1.0, 1.1, 1.2, 1.5, 2.0, 2.5, 5, 10, 15, 20]
    batt8_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                    1.0, 1.1, 1.2, 1.5, 2.0, 2.5, 5, 10, 15, 20]
    ldes_levels = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 5, 8, 10, 15, 20]

    # ── Phase 1: Coarse grid at 5% step ──
    combos_5 = generate_4d_combos(hydro_cap, step=5)
    seeds = get_seed_combos(hydro_cap)
    if len(seeds) > 0:
        combos_5 = np.vstack([combos_5, seeds])
        combos_5 = np.unique(combos_5, axis=0)

    # Cross-threshold pruning: eliminate mixes that were infeasible at previous threshold
    if prev_pruning is not None:
        feasible_prev = prev_pruning.get('feasible_mixes', set())
        all_prev = prev_pruning.get('all_mixes', set())
        if feasible_prev and all_prev:
            # Only keep mixes that were either feasible before or weren't tested
            keep_mask = []
            for row in combos_5:
                key = (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
                # Keep if: was feasible at lower threshold, or is a new mix we haven't tested
                if key in feasible_prev or key not in all_prev:
                    keep_mask.append(True)
                else:
                    keep_mask.append(False)
            keep_mask = np.array(keep_mask)
            n_before = len(combos_5)
            combos_5 = combos_5[keep_mask]
            n_pruned = n_before - len(combos_5)
            if n_pruned > 0:
                print(f"      Pruned {n_pruned} infeasible mixes from previous threshold")

    n_combos = len(combos_5)
    mix_fracs = combos_5 / 100.0

    # Pre-compute supply_row for each mix: (N, 8760)
    supply_rows = mix_fracs @ supply_matrix

    # Procurement levels — adaptive step based on range width
    proc_range = proc_max - proc_min
    if proc_range > 200:
        proc_step = 10  # Very wide range (300%+): coarse sweep
    elif proc_range > 100:
        proc_step = 5   # Wide range (100-200%): medium sweep
    else:
        proc_step = 2   # Narrow range: fine sweep
    proc_levels = list(range(proc_min, proc_max + 1, proc_step))
    if proc_max not in proc_levels:
        proc_levels.append(proc_max)

    # ── Mix checkpoint: load or initialize state ──
    chk = _load_mix_progress(iso, threshold)
    if chk is not None:
        candidates = chk['candidates']
        resume_phase = chk['phase']
        resume_cursor = chk['mix_cursor']
        near_miss_mixes = chk['near_miss_mixes'] or {}
        mix_min_proc = chk['mix_min_proc'] or {}
        # Rebuild seen set from candidates
        seen = set()
        for c in candidates:
            mk = (c['resource_mix']['clean_firm'], c['resource_mix']['solar'],
                   c['resource_mix']['wind'], c['resource_mix']['hydro'])
            key = (mk, c['procurement_pct'], c['battery_dispatch_pct'],
                   c.get('battery8_dispatch_pct', 0), c['ldes_dispatch_pct'])
            seen.add(key)
        # Rebuild feasible_mix_keys from candidates
        feasible_mix_keys = set()
        for c in candidates:
            mk = (c['resource_mix']['clean_firm'], c['resource_mix']['solar'],
                   c['resource_mix']['wind'], c['resource_mix']['hydro'])
            feasible_mix_keys.add(mk)
        # Rebuild all_mix_keys from combos evaluated so far
        all_mix_keys = set()
        cursor_1a = resume_cursor if resume_phase == '1a' else n_combos
        for idx in range(min(cursor_1a, n_combos)):
            all_mix_keys.add((int(combos_5[idx][0]), int(combos_5[idx][1]),
                              int(combos_5[idx][2]), int(combos_5[idx][3])))
        print(f"      Resuming from phase={resume_phase}, cursor={resume_cursor}, "
              f"{len(candidates):,} candidates loaded")
    else:
        candidates = []
        seen = set()
        near_miss_mixes = {}
        mix_min_proc = {}
        all_mix_keys = set()
        feasible_mix_keys = set()
        resume_phase = '1a'
        resume_cursor = 0

    # ── Cross-threshold pollination: skip set for proven mixes ──
    cross_skip = cross_feasible_mixes if cross_feasible_mixes else set()
    if cross_skip:
        feasible_mix_keys.update(cross_skip)
        if resume_phase == '1a' and resume_cursor == 0:
            print(f"      {len(cross_skip):,} mixes known feasible from cache (skip near-miss storage)")

    def add_candidate(mix_arr, proc, bp, b8p, lp, score):
        """Add candidate if not already seen."""
        mix_key = (int(mix_arr[0]), int(mix_arr[1]), int(mix_arr[2]), int(mix_arr[3]))
        key = (mix_key, proc, bp, b8p, lp)
        if key not in seen:
            seen.add(key)
            candidates.append({
                'resource_mix': {rt: mix_key[j] for j, rt in enumerate(RESOURCE_TYPES)},
                'procurement_pct': proc,
                'battery_dispatch_pct': bp,
                'battery8_dispatch_pct': b8p,
                'ldes_dispatch_pct': lp,
                'hourly_match_score': round(score * 100, 2),
            })

    # Build per-mix procurement floors from previous threshold
    prev_min_proc = {}
    if prev_pruning is not None:
        prev_min_proc = prev_pruning.get('min_proc', {})

    # Phase 1a: No-storage sweep — vectorized batch + binary search procurement
    # Batch-scores all mixes at max procurement in one matrix multiply, then
    # binary searches for minimum feasible procurement per feasible mix.
    # Near-miss window: 25% (wider than original 15% per SPEC item 3).
    phase1a_start = resume_cursor if resume_phase == '1a' else n_combos

    if phase1a_start < n_combos:
        # Vectorized: score ALL mixes at max procurement in one shot
        max_pf = proc_levels[-1] / 100.0
        all_max_scores = batch_hourly_scores(demand_arr, supply_matrix, mix_fracs, max_pf)

        for i in range(phase1a_start, n_combos):
            mix_key = (int(combos_5[i][0]), int(combos_5[i][1]),
                       int(combos_5[i][2]), int(combos_5[i][3]))
            all_mix_keys.add(mix_key)

            max_score = all_max_scores[i]
            if max_score >= target:
                # Binary search for minimum feasible procurement
                floor = max(prev_min_proc.get(mix_key, proc_min), proc_min)
                valid_procs = [p for p in proc_levels if p >= floor]
                if valid_procs:
                    lo, hi = 0, len(valid_procs) - 1
                    while lo < hi:
                        mid = (lo + hi) // 2
                        pf = valid_procs[mid] / 100.0
                        score = np.sum(np.minimum(demand_arr, supply_rows[i] * pf))
                        if score >= target:
                            hi = mid
                        else:
                            lo = mid + 1
                    min_proc = valid_procs[lo]
                    score = np.sum(np.minimum(demand_arr, supply_rows[i] * (min_proc / 100.0)))
                    add_candidate(combos_5[i], min_proc, 0, 0, 0, score)
                    mix_min_proc[mix_key] = min(mix_min_proc.get(mix_key, 9999), min_proc)
                    feasible_mix_keys.add(mix_key)
            elif max_score >= target - 0.25:
                # Near-miss: candidate for storage sweep in Phase 1b
                if mix_key not in cross_skip:
                    near_miss_mixes[i] = max(near_miss_mixes.get(i, 0), max_score)

            # Nth-mix checkpoint within Phase 1a
            mixes_done = i + 1 - phase1a_start
            if mixes_done > 0 and mixes_done % MIX_CHECKPOINT_INTERVAL == 0:
                _save_mix_progress(iso, threshold, candidates, '1a', i + 1,
                                   near_miss_data=near_miss_mixes,
                                   mix_min_proc_data=mix_min_proc)

    # Phase 1a→1b transition checkpoint
    if resume_phase == '1a' and phase1a_start < n_combos:
        _save_mix_progress(iso, threshold, candidates, '1b', 0,
                           near_miss_data=near_miss_mixes,
                           mix_min_proc_data=mix_min_proc)

    # Phase 1b: Storage sweep on near-miss mixes
    # OPTIMIZED with three key efficiency strategies:
    #
    # 1. SOC PRE-SCREEN: For each (mix, max_procurement), compute the max daily
    #    SOC that bat4/bat8/LDES would ever reach with unlimited capacity. Storage
    #    levels above max_soc are pure waste — skip them entirely. This eliminates
    #    80-95% of bat4/bat8 combos since they saturate below ~1.2%.
    #
    # 2. DOMINANCE PRUNING: Within each storage type's sweep, once a level achieves
    #    the target, higher levels are dominated (same target met with more idle
    #    capacity = higher LCOS). Stop searching higher levels.
    #    Exception: we still record the first-feasible as a solution, since Step 3
    #    needs all feasible configs to find cost-optimal ones.
    #
    # 3. CURTAILMENT GUARD: Only run storage sweep if the mix actually has curtailed
    #    clean energy to harness. If surplus at max procurement is zero, storage
    #    can't help regardless of capacity.
    #
    # Dispatch order remains: battery4 → battery8 → LDES (cheapest first).
    near_miss_list = sorted(near_miss_mixes.keys())
    phase1b_start = resume_cursor if resume_phase == '1b' else 0
    if resume_phase not in ('1a', '1b'):
        phase1b_start = len(near_miss_list)  # Skip entirely if past Phase 1b

    for nm_idx in range(phase1b_start, len(near_miss_list)):
        i = near_miss_list[nm_idx]
        supply_row = supply_rows[i]
        mix = combos_5[i]
        mix_key = (int(mix[0]), int(mix[1]), int(mix[2]), int(mix[3]))

        if mix_key in cross_skip:
            continue

        max_pf = proc_levels[-1] / 100.0

        # ── SOC PRE-SCREEN ──
        # Compute max useful SOC for each storage type at max procurement.
        # This is fast (no storage capacity constraint) and gives us the ceiling.
        bat4_max_soc, bat4_max_disp = _compute_max_daily_battery_soc(
            demand_arr, supply_row, max_pf, batt_eff, BATTERY_DURATION_HOURS)
        bat8_max_soc, bat8_max_disp = _compute_max_daily_battery_soc(
            demand_arr, supply_row, max_pf, batt8_eff, BATTERY8_DURATION_HOURS)
        ldes_max_soc, ldes_max_disp = _compute_max_ldes_soc(
            demand_arr, supply_row, max_pf, ldes_eff, ldes_window_hours)

        # Skip mix entirely if no curtailment to harness
        if bat4_max_disp <= 0 and bat8_max_disp <= 0 and ldes_max_disp <= 0:
            continue

        # Filter storage levels to only those below max useful SOC (+ 10% margin)
        bat4_max_pct = bat4_max_soc * 100.0 * 1.1  # 10% margin
        bat8_max_pct = bat8_max_soc * 100.0 * 1.1
        ldes_max_pct = ldes_max_soc * 100.0 * 1.1

        eff_batt_levels = [bl for bl in batt_levels if bl <= bat4_max_pct or bl == 0]
        eff_batt8_levels = [bl for bl in batt8_levels if bl <= bat8_max_pct or bl == 0]
        eff_ldes_levels = [ll for ll in ldes_levels if ll <= ldes_max_pct or ll == 0]

        # Ensure at least one non-zero level per type if there's any dispatch potential
        if bat4_max_disp > 0 and len(eff_batt_levels) == 1:
            eff_batt_levels = [0, batt_levels[1]]  # Add smallest non-zero
        if bat8_max_disp > 0 and len(eff_batt8_levels) == 1:
            eff_batt8_levels = [0, batt8_levels[1]]
        if ldes_max_disp > 0 and len(eff_ldes_levels) == 1:
            eff_ldes_levels = [0, ldes_levels[1]]

        # ── STORAGE SWEEP WITH DOMINANCE PRUNING ──
        # For bat4: sweep ascending. Once a level makes the combo feasible at
        # min procurement, record it but also test the NEXT level (it might achieve
        # feasibility at even lower procurement). Stop once two consecutive levels
        # produce the same min_proc — higher levels are dominated.
        for bp in eff_batt_levels:
            batt_cap = bp / 100.0
            batt_pow = batt_cap / BATTERY_DURATION_HOURS if batt_cap > 0 else 0
            bat4_hit_target = False
            bat4_best_proc = 9999

            for b8p in eff_batt8_levels:
                batt8_cap = b8p / 100.0
                batt8_pow = batt8_cap / BATTERY8_DURATION_HOURS if batt8_cap > 0 else 0
                bat8_hit_target = False
                bat8_best_proc = 9999

                for lp in eff_ldes_levels:
                    if bp == 0 and b8p == 0 and lp == 0:
                        continue
                    ldes_cap = lp / 100.0
                    ldes_pow = ldes_cap / LDES_DURATION_HOURS if ldes_cap > 0 else 0

                    # Check feasibility at max procurement first
                    max_score = _score_with_all_storage(
                        demand_arr, supply_row, max_pf,
                        batt_cap, batt_pow, batt_eff,
                        batt8_cap, batt8_pow, batt8_eff,
                        ldes_cap, ldes_pow, ldes_eff, ldes_window_hours)
                    if max_score < target:
                        continue  # Infeasible even at max procurement

                    # Binary search for minimum feasible procurement
                    lo, hi = 0, len(proc_levels) - 1
                    while lo < hi:
                        mid = (lo + hi) // 2
                        pf = proc_levels[mid] / 100.0
                        score = _score_with_all_storage(
                            demand_arr, supply_row, pf,
                            batt_cap, batt_pow, batt_eff,
                            batt8_cap, batt8_pow, batt8_eff,
                            ldes_cap, ldes_pow, ldes_eff, ldes_window_hours)
                        if score >= target:
                            hi = mid
                        else:
                            lo = mid + 1
                    min_proc = proc_levels[lo]
                    score = _score_with_all_storage(
                        demand_arr, supply_row, min_proc / 100.0,
                        batt_cap, batt_pow, batt_eff,
                        batt8_cap, batt8_pow, batt8_eff,
                        ldes_cap, ldes_pow, ldes_eff, ldes_window_hours)
                    add_candidate(mix, min_proc, bp, b8p, lp, score)
                    mix_min_proc[mix_key] = min(mix_min_proc.get(mix_key, 9999), min_proc)
                    feasible_mix_keys.add(mix_key)

                    # LDES dominance: if adding more LDES doesn't lower min_proc,
                    # higher LDES levels are dominated for this (bat4, bat8) pair
                    if min_proc >= bat8_best_proc and bat8_hit_target:
                        break  # No improvement — stop LDES sweep
                    bat8_hit_target = True
                    bat8_best_proc = min(bat8_best_proc, min_proc)

                # bat8 dominance check
                if bat8_best_proc <= bat4_best_proc and bat4_hit_target and b8p > 0:
                    # bat8 at this level didn't improve over previous — check if stuck
                    pass  # Don't break bat8 loop — LDES combos may differ
                bat4_hit_target = bat4_hit_target or bat8_hit_target
                bat4_best_proc = min(bat4_best_proc, bat8_best_proc)

        # Nth-mix checkpoint within Phase 1b
        mixes_done = nm_idx + 1 - phase1b_start
        if mixes_done > 0 and mixes_done % MIX_CHECKPOINT_INTERVAL == 0:
            _save_mix_progress(iso, threshold, candidates, '1b', nm_idx + 1,
                               near_miss_data=near_miss_mixes,
                               mix_min_proc_data=mix_min_proc)

    # ── Phase 2: Refine feasible archetypes to 1% resolution ──
    # Same early-stop logic: per-mix, stop procurement once target is met
    if candidates:
        mix_archetypes = set()
        for c in candidates:
            m = tuple(c['resource_mix'][rt] for rt in RESOURCE_TYPES)
            mix_archetypes.add(m)

        for mix_tuple in mix_archetypes:
            base = np.array(mix_tuple, dtype=np.float64)
            fine_combos = generate_4d_combos_around(base, hydro_cap, step=1, radius=3)
            if len(fine_combos) == 0:
                continue

            fine_fracs = fine_combos / 100.0
            fine_supply = fine_fracs @ supply_matrix
            n_fine = len(fine_combos)

            # No-storage: batch eval at max proc + binary search for feasible
            # Near-miss window: 15% for refinement (wider than original 10%, SPEC item 3)
            max_pf = proc_levels[-1] / 100.0
            fine_max_scores = batch_hourly_scores(demand_arr, supply_matrix, fine_fracs, max_pf)

            for j in range(n_fine):
                max_score_j = fine_max_scores[j]
                if max_score_j >= target:
                    # Binary search for min feasible procurement (no storage)
                    lo, hi = 0, len(proc_levels) - 1
                    while lo < hi:
                        mid = (lo + hi) // 2
                        pf = proc_levels[mid] / 100.0
                        score = np.sum(np.minimum(demand_arr, fine_supply[j] * pf))
                        if score >= target:
                            hi = mid
                        else:
                            lo = mid + 1
                    min_proc = proc_levels[lo]
                    score = np.sum(np.minimum(demand_arr, fine_supply[j] * (min_proc / 100.0)))
                    add_candidate(fine_combos[j], min_proc, 0, 0, 0, score)
                elif max_score_j >= target - 0.15:
                    # Near-miss refinement: storage sweep with SOC screen + dominance
                    supply_row_j = fine_supply[j]

                    # SOC pre-screen for this fine mix
                    b4ms, b4md = _compute_max_daily_battery_soc(
                        demand_arr, supply_row_j, max_pf, batt_eff, BATTERY_DURATION_HOURS)
                    b8ms, b8md = _compute_max_daily_battery_soc(
                        demand_arr, supply_row_j, max_pf, batt8_eff, BATTERY8_DURATION_HOURS)
                    lms, lmd = _compute_max_ldes_soc(
                        demand_arr, supply_row_j, max_pf, ldes_eff, ldes_window_hours)

                    if b4md <= 0 and b8md <= 0 and lmd <= 0:
                        continue  # No curtailment to harness

                    b4_ceil = b4ms * 100.0 * 1.1
                    b8_ceil = b8ms * 100.0 * 1.1
                    l_ceil = lms * 100.0 * 1.1
                    e_bl = [bl for bl in batt_levels if bl <= b4_ceil or bl == 0]
                    e_b8l = [bl for bl in batt8_levels if bl <= b8_ceil or bl == 0]
                    e_ll = [ll for ll in ldes_levels if ll <= l_ceil or ll == 0]

                    for bp in e_bl:
                        batt_cap = bp / 100.0
                        batt_pow = batt_cap / BATTERY_DURATION_HOURS if batt_cap > 0 else 0
                        for b8p in e_b8l:
                            batt8_cap = b8p / 100.0
                            batt8_pow = batt8_cap / BATTERY8_DURATION_HOURS if batt8_cap > 0 else 0
                            ldes_best_proc = 9999
                            for lp in e_ll:
                                if bp == 0 and b8p == 0 and lp == 0:
                                    continue
                                ldes_cap = lp / 100.0
                                ldes_pow = ldes_cap / LDES_DURATION_HOURS if ldes_cap > 0 else 0
                                max_sc = _score_with_all_storage(
                                    demand_arr, supply_row_j, max_pf,
                                    batt_cap, batt_pow, batt_eff,
                                    batt8_cap, batt8_pow, batt8_eff,
                                    ldes_cap, ldes_pow, ldes_eff,
                                    ldes_window_hours)
                                if max_sc < target:
                                    continue
                                lo, hi = 0, len(proc_levels) - 1
                                while lo < hi:
                                    mid = (lo + hi) // 2
                                    pf = proc_levels[mid] / 100.0
                                    sc = _score_with_all_storage(
                                        demand_arr, supply_row_j, pf,
                                        batt_cap, batt_pow, batt_eff,
                                        batt8_cap, batt8_pow, batt8_eff,
                                        ldes_cap, ldes_pow, ldes_eff,
                                        ldes_window_hours)
                                    if sc >= target:
                                        hi = mid
                                    else:
                                        lo = mid + 1
                                min_proc = proc_levels[lo]
                                sc = _score_with_all_storage(
                                    demand_arr, supply_row_j, min_proc / 100.0,
                                    batt_cap, batt_pow, batt_eff,
                                    batt8_cap, batt8_pow, batt8_eff,
                                    ldes_cap, ldes_pow, ldes_eff,
                                    ldes_window_hours)
                                add_candidate(fine_combos[j], min_proc, bp, b8p, lp, sc)
                                # LDES dominance pruning
                                if min_proc >= ldes_best_proc and ldes_best_proc < 9999:
                                    break
                                ldes_best_proc = min(ldes_best_proc, min_proc)

    # Build pruning info for next threshold
    # Phase 2 fine combos also contribute to feasibility (update from candidates)
    for c in candidates:
        mk = tuple(c['resource_mix'][rt] for rt in RESOURCE_TYPES)
        feasible_mix_keys.add(mk)
        if mk not in mix_min_proc or c['procurement_pct'] < mix_min_proc[mk]:
            mix_min_proc[mk] = c['procurement_pct']

    pruning_info = {
        'feasible_mixes': feasible_mix_keys,
        'min_proc': mix_min_proc,
        'all_mixes': all_mix_keys,
    }

    return candidates, pruning_info


def extract_pareto(candidates, target):
    """Extract Pareto-optimal candidates along procurement/storage tradeoff.

    Returns 3-5 representative points:
    1. Minimum procurement (with whatever storage needed)
    2. Minimum storage (with higher procurement)
    3. Minimum total (procurement + storage balanced)
    4-5. Diverse mix archetypes if available
    """
    if not candidates:
        return []

    # Deduplicate
    seen = set()
    unique = []
    for c in candidates:
        m = tuple(c['resource_mix'][rt] for rt in RESOURCE_TYPES)
        key = (m, c['procurement_pct'], c['battery_dispatch_pct'],
               c.get('battery8_dispatch_pct', 0), c['ldes_dispatch_pct'])
        if key not in seen:
            seen.add(key)
            unique.append(c)

    if not unique:
        return []

    # Metrics for Pareto
    for c in unique:
        c['total_storage'] = (c['battery_dispatch_pct'] +
                              c.get('battery8_dispatch_pct', 0) +
                              c['ldes_dispatch_pct'])
        c['total_resource'] = c['procurement_pct'] + c['total_storage']

    # Sort by different criteria
    by_procurement = sorted(unique, key=lambda c: (c['procurement_pct'], c['total_storage']))
    by_storage = sorted(unique, key=lambda c: (c['total_storage'], c['procurement_pct']))
    by_total = sorted(unique, key=lambda c: c['total_resource'])

    pareto = []
    pareto_keys = set()

    def add_if_new(candidate, ptype):
        m = tuple(candidate['resource_mix'][rt] for rt in RESOURCE_TYPES)
        key = (m, candidate['procurement_pct'], candidate['battery_dispatch_pct'],
               candidate.get('battery8_dispatch_pct', 0), candidate['ldes_dispatch_pct'])
        if key not in pareto_keys:
            pareto_keys.add(key)
            c = dict(candidate)
            c['pareto_type'] = ptype
            del c['total_storage']
            del c['total_resource']
            pareto.append(c)

    # 1. Minimum procurement
    add_if_new(by_procurement[0], 'min_procurement')

    # 2. Minimum storage
    add_if_new(by_storage[0], 'min_storage')

    # 3. Minimum total
    add_if_new(by_total[0], 'min_total')

    # 4-5. Diverse mix archetypes (different from above)
    # Find mixes with highest solar, highest wind, highest firm
    by_solar = sorted(unique, key=lambda c: -c['resource_mix']['solar'])
    by_wind = sorted(unique, key=lambda c: -c['resource_mix']['wind'])
    by_firm = sorted(unique, key=lambda c: -c['resource_mix']['clean_firm'])

    for cand, ptype in [(by_solar[0], 'high_solar'), (by_wind[0], 'high_wind'),
                        (by_firm[0], 'high_firm')]:
        if len(pareto) >= 5:
            break
        add_if_new(cand, ptype)

    return pareto


# ══════════════════════════════════════════════════════════════════════════════
# PER-ISO PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def process_iso(args):
    """Process all thresholds for a single ISO. Designed for multiprocessing.

    Uses per-ISO/threshold parquet checkpoints for both threshold-level and
    intra-threshold (Nth-mix) resume. Falls back to legacy JSON checkpoints
    for backward compatibility with older runs.
    """
    iso, demand_data, gen_profiles = args
    iso_start = time.time()

    demand_norm = demand_data[iso]['normalized']
    supply_profiles = get_supply_profiles(iso, gen_profiles)
    demand_arr, supply_matrix = prepare_numpy_profiles(demand_norm, supply_profiles)
    hydro_cap = HYDRO_CAPS[iso]

    print(f"\n  {iso}: Starting optimization ({len(THRESHOLDS)} thresholds, "
          f"hydro_cap={hydro_cap}%)")

    # Check which thresholds are done: new parquet-based + legacy JSON fallback
    legacy_checkpoint = load_checkpoint(iso)
    legacy_completed = set(legacy_checkpoint.get('completed', []))

    iso_results = {
        'iso': iso,
        'label': ISO_LABELS.get(iso, iso),
        'annual_demand_mwh': demand_data[iso]['total_annual_mwh'],
        'peak_demand_mw': demand_data[iso]['peak_mw'],
        'hydro_cap': hydro_cap,
        'thresholds': {},
    }

    # ── Cross-threshold pollination: mix→max_score map ──
    mix_max_score = {}

    # Seed from done parquets (completed thresholds)
    for threshold in THRESHOLDS:
        t_str = str(threshold)
        if _is_threshold_done(iso, threshold):
            # Load from done parquet
            done_path = _threshold_done_path(iso, threshold)
            try:
                done_table = pq.read_table(done_path)
                n = done_table.num_rows
                if n > 0:
                    cf = done_table.column('clean_firm').to_pylist()
                    sol = done_table.column('solar').to_pylist()
                    wnd = done_table.column('wind').to_pylist()
                    hyd = done_table.column('hydro').to_pylist()
                    scores = done_table.column('hourly_match_score').to_pylist()
                    for j in range(n):
                        mk = (cf[j], sol[j], wnd[j], hyd[j])
                        if scores[j] > mix_max_score.get(mk, 0):
                            mix_max_score[mk] = scores[j]
            except Exception:
                pass

    # Also seed from interim parquet cache (legacy)
    interim_path = os.path.join(CHECKPOINT_DIR, f'{iso}_v4_interim.parquet')
    if os.path.exists(interim_path) and HAS_PARQUET:
        try:
            cached = pq.read_table(interim_path)
            if cached.num_rows > 0:
                import pandas as pd
                df = cached.select(['clean_firm', 'solar', 'wind', 'hydro',
                                     'hourly_match_score']).to_pandas()
                grouped = df.groupby(['clean_firm', 'solar', 'wind', 'hydro']
                                      )['hourly_match_score'].max()
                for k, v in grouped.items():
                    if v > mix_max_score.get(k, 0):
                        mix_max_score[k] = v
                del df, grouped, cached
        except Exception:
            pass

    if mix_max_score:
        print(f"    {iso}: {len(mix_max_score):,} unique mixes from cache/completed thresholds")

    # ── Cross-threshold solution seeding accumulator ──
    # Collects full solutions from completed thresholds so qualifying ones
    # can be injected into higher thresholds without re-computation.
    all_found_solutions = []

    prev_pruning = None
    for threshold in THRESHOLDS:
        t_str = str(threshold)

        # Skip if done (parquet exists or legacy JSON says so)
        if _is_threshold_done(iso, threshold) or t_str in legacy_completed:
            print(f"    {iso} {threshold}%: done — skipping")
            # Build pruning info + load full solutions for cross-threshold seeding
            done_path = _threshold_done_path(iso, threshold)
            if os.path.exists(done_path):
                try:
                    done_table = pq.read_table(done_path)
                    n = done_table.num_rows
                    if n > 0:
                        cf = done_table.column('clean_firm').to_pylist()
                        sol = done_table.column('solar').to_pylist()
                        wnd = done_table.column('wind').to_pylist()
                        hyd = done_table.column('hydro').to_pylist()
                        proc = done_table.column('procurement_pct').to_pylist()
                        bat = done_table.column('battery_dispatch_pct').to_pylist()
                        bat8 = (done_table.column('battery8_dispatch_pct').to_pylist()
                                if 'battery8_dispatch_pct' in done_table.column_names
                                else [0] * n)
                        ldes_col = done_table.column('ldes_dispatch_pct').to_pylist()
                        scores = done_table.column('hourly_match_score').to_pylist()
                        feasible_keys = set()
                        min_proc = {}
                        for j in range(n):
                            mk = (cf[j], sol[j], wnd[j], hyd[j])
                            feasible_keys.add(mk)
                            if mk not in min_proc or proc[j] < min_proc[mk]:
                                min_proc[mk] = proc[j]
                            all_found_solutions.append({
                                'resource_mix': {'clean_firm': cf[j], 'solar': sol[j],
                                                 'wind': wnd[j], 'hydro': hyd[j]},
                                'procurement_pct': proc[j],
                                'battery_dispatch_pct': bat[j],
                                'battery8_dispatch_pct': bat8[j],
                                'ldes_dispatch_pct': ldes_col[j],
                                'hourly_match_score': scores[j],
                            })
                        prev_pruning = {
                            'feasible_mixes': feasible_keys,
                            'min_proc': min_proc,
                            'all_mixes': feasible_keys,  # Approximate
                        }
                except Exception:
                    pass
            continue

        # Build cross-feasible set from mix_max_score
        cross_feasible_mixes = set()
        if mix_max_score:
            cross_feasible_mixes = {mk for mk, s in mix_max_score.items() if s >= threshold}

        t_start = time.time()
        feasible, pruning_info = optimize_threshold(
            iso, threshold, demand_arr, supply_matrix, hydro_cap,
            prev_pruning=prev_pruning, cross_feasible_mixes=cross_feasible_mixes)
        prev_pruning = pruning_info

        # ── Cross-threshold seeding: inject qualifying solutions from
        #    lower thresholds. Mixes that scored >= current threshold at a
        #    lower threshold are injected directly (their Phase 1b was
        #    skipped via cross_skip, so the known result is carried forward).
        if all_found_solutions:
            existing_keys = set()
            for c in feasible:
                mk = (c['resource_mix']['clean_firm'], c['resource_mix']['solar'],
                       c['resource_mix']['wind'], c['resource_mix']['hydro'])
                existing_keys.add((mk, c['procurement_pct'], c['battery_dispatch_pct'],
                                    c.get('battery8_dispatch_pct', 0), c['ldes_dispatch_pct']))
            seeded = 0
            for c in all_found_solutions:
                if c['hourly_match_score'] >= threshold:
                    mk = (c['resource_mix']['clean_firm'], c['resource_mix']['solar'],
                           c['resource_mix']['wind'], c['resource_mix']['hydro'])
                    key = (mk, c['procurement_pct'], c['battery_dispatch_pct'],
                            c.get('battery8_dispatch_pct', 0), c['ldes_dispatch_pct'])
                    if key not in existing_keys:
                        feasible.append(c)
                        existing_keys.add(key)
                        seeded += 1
            if seeded > 0:
                print(f"      Seeded {seeded:,} solutions from lower thresholds")

        # Accumulate new solutions into mix_max_score + seeding pool
        for c in feasible:
            mk = (c['resource_mix']['clean_firm'], c['resource_mix']['solar'],
                   c['resource_mix']['wind'], c['resource_mix']['hydro'])
            s = c['hourly_match_score']
            if s > mix_max_score.get(mk, 0):
                mix_max_score[mk] = s
        all_found_solutions.extend(feasible)
        t_elapsed = time.time() - t_start

        archetypes = set()
        for c in feasible:
            archetypes.add(tuple(c['resource_mix'][rt] for rt in RESOURCE_TYPES))

        iso_results['thresholds'][t_str] = {
            'candidates': feasible,
            'candidate_count': len(feasible),
            'mix_archetypes': len(archetypes),
            'elapsed_seconds': round(t_elapsed, 2),
        }
        print(f"    {iso} {threshold}%: {len(feasible)} solutions "
              f"({len(archetypes)} mix archetypes), {t_elapsed:.1f}s")

        # Save done parquet (also cleans up progress file)
        _save_threshold_done(iso, threshold, feasible)

        # Append to interim cache + legacy JSON checkpoint for backward compat
        append_threshold_to_cache(iso, threshold, feasible)
        save_checkpoint(iso, iso_results, t_str)

    iso_elapsed = time.time() - iso_start
    print(f"  {iso} completed in {iso_elapsed:.1f}s")
    return iso, iso_results


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINTING
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(iso, iso_results, completed_threshold):
    """Save checkpoint after each threshold."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f'{iso}_v4_checkpoint.json')

    # Read existing to preserve completed list
    completed = set()
    if os.path.exists(path):
        try:
            with open(path) as f:
                existing = json.load(f)
            completed = set(existing.get('completed', []))
        except (json.JSONDecodeError, IOError):
            pass

    completed.add(completed_threshold)
    checkpoint = {
        'completed': sorted(completed),
        'thresholds': iso_results['thresholds'],
    }
    with open(path, 'w') as f:
        json.dump(checkpoint, f)


def append_threshold_to_cache(iso, threshold, candidates):
    """Append a single threshold's solutions to a per-ISO interim Parquet file.

    Called after each threshold completes so solutions are persisted immediately.
    Each ISO gets its own interim file to avoid write conflicts in parallel mode.
    These get merged into the main cache at the end of the run.
    """
    if not HAS_PARQUET or not candidates:
        return

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    interim_path = os.path.join(CHECKPOINT_DIR, f'{iso}_v4_interim.parquet')

    # Build rows for this threshold
    rows = []
    for c in candidates:
        rows.append({
            'iso': iso,
            'threshold': float(threshold),
            'clean_firm': c['resource_mix']['clean_firm'],
            'solar': c['resource_mix']['solar'],
            'wind': c['resource_mix']['wind'],
            'hydro': c['resource_mix']['hydro'],
            'procurement_pct': c['procurement_pct'],
            'battery_dispatch_pct': c['battery_dispatch_pct'],
            'battery8_dispatch_pct': c.get('battery8_dispatch_pct', 0),
            'ldes_dispatch_pct': c['ldes_dispatch_pct'],
            'hourly_match_score': c['hourly_match_score'],
            'pareto_type': c.get('pareto_type', ''),
        })

    new_table = _rows_to_table(rows)
    if new_table is None:
        return

    # Append to existing interim file if it exists
    if os.path.exists(interim_path):
        try:
            existing = pq.read_table(interim_path)
            new_table = pa.concat_tables([existing, new_table])
        except Exception:
            pass  # If corrupt, just overwrite with new data

    pq.write_table(new_table, interim_path, compression='snappy')


def load_checkpoint(iso):
    """Load checkpoint if exists."""
    path = os.path.join(CHECKPOINT_DIR, f'{iso}_v4_checkpoint.json')
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


# ── Per-ISO/threshold Parquet mix checkpoints ──

def _mix_progress_path(iso, threshold):
    """Path for in-progress mix checkpoint parquet."""
    return os.path.join(CHECKPOINT_DIR, f'{iso}_v4_t{threshold}_progress.parquet')


def _threshold_done_path(iso, threshold):
    """Path for completed threshold results parquet."""
    return os.path.join(CHECKPOINT_DIR, f'{iso}_v4_t{threshold}_done.parquet')


def _is_threshold_done(iso, threshold):
    """Check if a threshold has a completed parquet."""
    return os.path.exists(_threshold_done_path(iso, threshold))


def _save_mix_progress(iso, threshold, candidates, phase, mix_cursor,
                       near_miss_data=None, mix_min_proc_data=None):
    """Save mix-level progress as per-ISO/threshold parquet with cursor metadata.

    Writes atomically (temp file + rename) to avoid corruption on crash.
    Metadata stored in parquet schema: phase, mix_cursor, near_miss, mix_min_proc.
    """
    if not HAS_PARQUET:
        return
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = _mix_progress_path(iso, threshold)

    rows = []
    for c in candidates:
        rows.append({
            'iso': iso,
            'threshold': float(threshold),
            'clean_firm': c['resource_mix']['clean_firm'],
            'solar': c['resource_mix']['solar'],
            'wind': c['resource_mix']['wind'],
            'hydro': c['resource_mix']['hydro'],
            'procurement_pct': c['procurement_pct'],
            'battery_dispatch_pct': c['battery_dispatch_pct'],
            'battery8_dispatch_pct': c.get('battery8_dispatch_pct', 0),
            'ldes_dispatch_pct': c['ldes_dispatch_pct'],
            'hourly_match_score': c['hourly_match_score'],
            'pareto_type': c.get('pareto_type', ''),
        })

    if rows:
        table = _rows_to_table(rows)
    else:
        table = pa.table({
            'iso': pa.array([], type=pa.string()),
            'threshold': pa.array([], type=pa.float64()),
            'clean_firm': pa.array([], type=pa.float64()),
            'solar': pa.array([], type=pa.float64()),
            'wind': pa.array([], type=pa.float64()),
            'hydro': pa.array([], type=pa.float64()),
            'procurement_pct': pa.array([], type=pa.int64()),
            'battery_dispatch_pct': pa.array([], type=pa.float64()),
            'battery8_dispatch_pct': pa.array([], type=pa.float64()),
            'ldes_dispatch_pct': pa.array([], type=pa.float64()),
            'hourly_match_score': pa.array([], type=pa.float64()),
            'pareto_type': pa.array([], type=pa.string()),
        })

    # Encode cursor state in schema metadata
    meta = {
        b'phase': phase.encode(),
        b'mix_cursor': str(mix_cursor).encode(),
        b'timestamp': datetime.now(timezone.utc).isoformat().encode(),
    }
    if near_miss_data is not None:
        meta[b'near_miss_json'] = json.dumps(
            {str(k): v for k, v in near_miss_data.items()}
        ).encode()
    if mix_min_proc_data is not None:
        meta[b'mix_min_proc_json'] = json.dumps(
            {','.join(str(x) for x in k): v for k, v in mix_min_proc_data.items()}
        ).encode()

    existing_meta = table.schema.metadata or {}
    existing_meta.update(meta)
    table = table.replace_schema_metadata(existing_meta)

    # Atomic write: temp file + rename
    import tempfile
    fd, tmp_path = tempfile.mkstemp(dir=CHECKPOINT_DIR, suffix='.parquet.tmp')
    os.close(fd)
    try:
        pq.write_table(table, tmp_path, compression='snappy')
        os.replace(tmp_path, path)  # Atomic on POSIX
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    n = len(candidates)
    print(f"      [mix checkpoint] {iso} {threshold}%: {n:,} solutions, "
          f"phase={phase}, cursor={mix_cursor}")


def _load_mix_progress(iso, threshold):
    """Load mix-level progress checkpoint from per-ISO/threshold parquet.

    Returns dict with candidates, phase, mix_cursor, near_miss_mixes, mix_min_proc.
    Returns None if no checkpoint exists or it's unreadable.
    """
    if not HAS_PARQUET:
        return None
    path = _mix_progress_path(iso, threshold)
    if not os.path.exists(path):
        return None

    try:
        table = pq.read_table(path)
        meta = table.schema.metadata or {}

        phase = meta.get(b'phase', b'1a').decode()
        mix_cursor = int(meta.get(b'mix_cursor', b'0').decode())

        # Decode near_miss_mixes: {int_index: float_score}
        near_miss_raw = meta.get(b'near_miss_json')
        near_miss_mixes = None
        if near_miss_raw:
            raw = json.loads(near_miss_raw.decode())
            near_miss_mixes = {int(k): v for k, v in raw.items()}

        # Decode mix_min_proc: {(int,...): int}
        min_proc_raw = meta.get(b'mix_min_proc_json')
        mix_min_proc = None
        if min_proc_raw:
            raw = json.loads(min_proc_raw.decode())
            mix_min_proc = {
                tuple(int(x) for x in k.split(',')): v
                for k, v in raw.items()
            }

        # Reconstruct candidates from parquet rows
        candidates = []
        n = table.num_rows
        if n > 0:
            cf = table.column('clean_firm').to_pylist()
            sol = table.column('solar').to_pylist()
            wnd = table.column('wind').to_pylist()
            hyd = table.column('hydro').to_pylist()
            proc = table.column('procurement_pct').to_pylist()
            bat = table.column('battery_dispatch_pct').to_pylist()
            bat8 = (table.column('battery8_dispatch_pct').to_pylist()
                    if 'battery8_dispatch_pct' in table.column_names
                    else [0] * n)
            ldes = table.column('ldes_dispatch_pct').to_pylist()
            score = table.column('hourly_match_score').to_pylist()
            pareto = table.column('pareto_type').to_pylist()

            for i in range(n):
                candidates.append({
                    'resource_mix': {
                        'clean_firm': cf[i], 'solar': sol[i],
                        'wind': wnd[i], 'hydro': hyd[i],
                    },
                    'procurement_pct': proc[i],
                    'battery_dispatch_pct': bat[i],
                    'battery8_dispatch_pct': bat8[i],
                    'ldes_dispatch_pct': ldes[i],
                    'hourly_match_score': score[i],
                    'pareto_type': pareto[i],
                })

        print(f"      [mix checkpoint loaded] {iso} {threshold}%: "
              f"{n:,} solutions, phase={phase}, cursor={mix_cursor}")

        return {
            'candidates': candidates,
            'phase': phase,
            'mix_cursor': mix_cursor,
            'near_miss_mixes': near_miss_mixes,
            'mix_min_proc': mix_min_proc,
        }
    except Exception as e:
        print(f"      [mix checkpoint] Could not load {path}: {e}")
        return None


def _save_threshold_done(iso, threshold, candidates):
    """Write finalized threshold parquet and clean up progress file."""
    if not HAS_PARQUET or not candidates:
        return
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    rows = []
    for c in candidates:
        rows.append({
            'iso': iso,
            'threshold': float(threshold),
            'clean_firm': c['resource_mix']['clean_firm'],
            'solar': c['resource_mix']['solar'],
            'wind': c['resource_mix']['wind'],
            'hydro': c['resource_mix']['hydro'],
            'procurement_pct': c['procurement_pct'],
            'battery_dispatch_pct': c['battery_dispatch_pct'],
            'battery8_dispatch_pct': c.get('battery8_dispatch_pct', 0),
            'ldes_dispatch_pct': c['ldes_dispatch_pct'],
            'hourly_match_score': c['hourly_match_score'],
            'pareto_type': c.get('pareto_type', ''),
        })

    table = _rows_to_table(rows)
    if table is None:
        return

    done_path = _threshold_done_path(iso, threshold)
    pq.write_table(table, done_path, compression='snappy')

    # Clean up progress file
    progress_path = _mix_progress_path(iso, threshold)
    if os.path.exists(progress_path):
        os.remove(progress_path)


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT — JSON + Parquet
# ══════════════════════════════════════════════════════════════════════════════

def _results_to_rows(all_results):
    """Convert nested results dict to flat list of row dicts for Parquet."""
    rows = []
    for iso, iso_data in all_results.items():
        for t_str, t_data in iso_data.get('thresholds', {}).items():
            for candidate in t_data.get('candidates', []):
                rows.append({
                    'iso': iso,
                    'threshold': float(t_str),
                    'clean_firm': candidate['resource_mix']['clean_firm'],
                    'solar': candidate['resource_mix']['solar'],
                    'wind': candidate['resource_mix']['wind'],
                    'hydro': candidate['resource_mix']['hydro'],
                    'procurement_pct': candidate['procurement_pct'],
                    'battery_dispatch_pct': candidate['battery_dispatch_pct'],
                    'battery8_dispatch_pct': candidate.get('battery8_dispatch_pct', 0),
                    'ldes_dispatch_pct': candidate['ldes_dispatch_pct'],
                    'hourly_match_score': candidate['hourly_match_score'],
                    'pareto_type': candidate.get('pareto_type', ''),
                })
    return rows


def _rows_to_table(rows):
    """Convert list of row dicts to a PyArrow table."""
    if not rows:
        return None
    return pa.table({
        col: [r[col] for r in rows]
        for col in rows[0].keys()
    })


def save_results_parquet(all_results, output_path):
    """Save results as Parquet."""
    if not HAS_PARQUET:
        print("  Parquet skipped — pyarrow not installed")
        return

    rows = _results_to_rows(all_results)
    if not rows:
        print("  Parquet skipped — no results")
        return

    table = _rows_to_table(rows)
    pq.write_table(table, output_path, compression='snappy')
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Parquet saved: {output_path} ({size_mb:.1f} MB)")


# ══════════════════════════════════════════════════════════════════════════════
# PERSISTENT SOLUTION CACHE — never lose feasible solutions across runs
# ══════════════════════════════════════════════════════════════════════════════

def merge_with_persistent_cache(new_results, cache_path):
    """Merge new results with Parquet cache, keeping ALL unique solutions.

    Each run may explore different parameter spaces. The persistent cache
    accumulates ALL feasible solutions ever found, so refining parameters
    never loses previous work. Uses Parquet for compact storage.
    """
    if not HAS_PARQUET:
        print("\n  Warning: pyarrow not available — cache merge skipped")
        return new_results

    new_rows = _results_to_rows(new_results)
    new_count = len(new_rows)

    # Load existing cache
    existing_table = None
    if os.path.exists(cache_path):
        try:
            existing_table = pq.read_table(cache_path)
            print(f"\n  Persistent cache loaded: {cache_path} ({existing_table.num_rows:,} solutions)")
        except Exception:
            print(f"\n  Warning: Could not read cache at {cache_path}, starting fresh")

    if existing_table is None or existing_table.num_rows == 0:
        print(f"  No previous cache — all {new_count:,} solutions are new")
        return new_results

    # Convert existing Parquet rows to dicts for dedup
    existing_count = existing_table.num_rows
    cols = existing_table.column_names

    # Backfill battery8_dispatch_pct if missing from old cache
    if 'battery8_dispatch_pct' not in existing_table.column_names:
        zeros = [0] * existing_count
        existing_table = existing_table.append_column('battery8_dispatch_pct',
                                                       pa.array(zeros, type=pa.int64()))
        print("  Backfilled battery8_dispatch_pct=0 for old cache entries")

    # Build set of existing keys for dedup
    # Key = (iso, threshold, clean_firm, solar, wind, hydro, procurement_pct, battery_dispatch_pct, battery8_dispatch_pct, ldes_dispatch_pct)
    existing_iso = existing_table.column('iso').to_pylist()
    existing_threshold = existing_table.column('threshold').to_pylist()
    existing_cf = existing_table.column('clean_firm').to_pylist()
    existing_sol = existing_table.column('solar').to_pylist()
    existing_wnd = existing_table.column('wind').to_pylist()
    existing_hyd = existing_table.column('hydro').to_pylist()
    existing_proc = existing_table.column('procurement_pct').to_pylist()
    existing_bat = existing_table.column('battery_dispatch_pct').to_pylist()
    existing_bat8 = existing_table.column('battery8_dispatch_pct').to_pylist()
    existing_ldes = existing_table.column('ldes_dispatch_pct').to_pylist()

    existing_keys = set()
    for i in range(existing_count):
        existing_keys.add((
            existing_iso[i], existing_threshold[i],
            existing_cf[i], existing_sol[i], existing_wnd[i], existing_hyd[i],
            existing_proc[i], existing_bat[i], existing_bat8[i], existing_ldes[i]
        ))

    # Filter new rows to only those not already in cache
    truly_new = []
    for r in new_rows:
        key = (r['iso'], r['threshold'],
               r['clean_firm'], r['solar'], r['wind'], r['hydro'],
               r['procurement_pct'], r['battery_dispatch_pct'],
               r.get('battery8_dispatch_pct', 0), r['ldes_dispatch_pct'])
        if key not in existing_keys:
            truly_new.append(r)

    if truly_new:
        new_table = _rows_to_table(truly_new)
        merged_table = pa.concat_tables([existing_table, new_table])
    else:
        merged_table = existing_table

    merged_count = merged_table.num_rows
    net_new = merged_count - existing_count
    print(f"  Cache merge: {existing_count:,} existing + {new_count:,} new run → "
          f"{merged_count:,} merged ({net_new:+,} net new solutions)")

    # Save merged cache
    pq.write_table(merged_table, cache_path, compression='snappy')
    size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    print(f"  Cache saved: {cache_path} ({size_mb:.1f} MB)")

    # Convert merged table back to results dict for downstream use
    return _table_to_results(merged_table)


def _table_to_results(table):
    """Convert a Parquet table back to the nested results dict format."""
    import pyarrow.compute as pc

    results = {}
    iso_col = table.column('iso')
    unique_isos = pc.unique(iso_col).to_pylist()

    for iso in unique_isos:
        iso_mask = pc.equal(iso_col, iso)
        iso_table = table.filter(iso_mask)

        threshold_col = iso_table.column('threshold')
        unique_thresholds = pc.unique(threshold_col).to_pylist()

        iso_results = {'thresholds': {}}
        for threshold in unique_thresholds:
            t_mask = pc.equal(threshold_col, threshold)
            t_table = iso_table.filter(t_mask)

            candidates = []
            n = t_table.num_rows
            cf = t_table.column('clean_firm').to_pylist()
            sol = t_table.column('solar').to_pylist()
            wnd = t_table.column('wind').to_pylist()
            hyd = t_table.column('hydro').to_pylist()
            proc = t_table.column('procurement_pct').to_pylist()
            bat = t_table.column('battery_dispatch_pct').to_pylist()
            bat8 = (t_table.column('battery8_dispatch_pct').to_pylist()
                    if 'battery8_dispatch_pct' in t_table.column_names
                    else [0] * n)
            ldes = t_table.column('ldes_dispatch_pct').to_pylist()
            score = t_table.column('hourly_match_score').to_pylist()
            pareto = t_table.column('pareto_type').to_pylist()

            archetypes = set()
            for i in range(n):
                candidates.append({
                    'resource_mix': {
                        'clean_firm': cf[i], 'solar': sol[i],
                        'wind': wnd[i], 'hydro': hyd[i],
                    },
                    'procurement_pct': proc[i],
                    'battery_dispatch_pct': bat[i],
                    'battery8_dispatch_pct': bat8[i],
                    'ldes_dispatch_pct': ldes[i],
                    'hourly_match_score': score[i],
                    'pareto_type': pareto[i],
                })
                archetypes.add((cf[i], sol[i], wnd[i], hyd[i]))

            t_str = str(threshold)
            iso_results['thresholds'][t_str] = {
                'candidates': candidates,
                'candidate_count': n,
                'mix_archetypes': len(archetypes),
            }

        results[iso] = iso_results

    return results


def _dedup_parquet_table(table):
    """Remove duplicate rows from a Parquet table based on solution key columns."""
    import pyarrow.compute as pc

    # Build composite key for dedup
    key_cols = ['iso', 'threshold', 'clean_firm', 'solar', 'wind', 'hydro',
                'procurement_pct', 'battery_dispatch_pct', 'battery8_dispatch_pct',
                'ldes_dispatch_pct']

    n = table.num_rows
    cols = {col: table.column(col).to_pylist() for col in key_cols}

    seen = set()
    keep = []
    for i in range(n):
        key = tuple(cols[col][i] for col in key_cols)
        if key not in seen:
            seen.add(key)
            keep.append(i)

    if len(keep) == n:
        return table  # No duplicates

    deduped = table.take(keep)
    removed = n - len(keep)
    if removed > 0:
        print(f"  Dedup: removed {removed:,} duplicates ({n:,} → {deduped.num_rows:,})")
    return deduped


def _count_solutions(results):
    """Count total solutions across all ISOs × thresholds."""
    total = 0
    for iso_data in results.values():
        for t_data in iso_data.get('thresholds', {}).values():
            total += len(t_data.get('candidates', []))
    return total


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    start_time = time.time()
    print("=" * 70)
    print("  v4.0 Physics Optimizer — Fresh Rebuild")
    print(f"  Numba: {'enabled' if HAS_NUMBA else 'disabled (NumPy fallback)'}")
    print(f"  Thresholds: {len(THRESHOLDS)} ({THRESHOLDS[0]}% - {THRESHOLDS[-1]}%)")
    print(f"  Resources: {N_RESOURCES}D ({', '.join(RESOURCE_TYPES)})")
    print(f"  ISOs: {len(ISOS)} ({', '.join(ISOS)})")
    print(f"  CPUs: {cpu_count()}")
    print("=" * 70)

    # Load data
    demand_data, gen_profiles, emission_rates, fossil_mix = load_data()

    # Warm up Numba JIT (first call compiles — includes batch kernels)
    if HAS_NUMBA:
        print("  Warming up Numba JIT...")
        dummy_demand = np.ones(H) / H
        dummy_supply = np.ones(H) / H
        dummy_supply_2d = np.ones((2, H)) / H
        _score_with_all_storage(dummy_demand, dummy_supply, 1.0,
                                0.01, 0.0025, 0.85,
                                0.01, 0.00125, 0.85,
                                0.01, 0.0001, 0.50, 168)
        _batch_score_no_storage(dummy_demand, dummy_supply_2d, 1.0, 2)
        _batch_score_storage(dummy_demand, dummy_supply_2d, 1.0, 2,
                             0.01, 0.0025, 0.85, 0.01, 0.00125, 0.85,
                             0.01, 0.0001, 0.50, 168)
        print("  JIT compilation complete")

    # Parse CLI args for target ISOs
    target_isos = None
    if len(sys.argv) > 1:
        target_isos = [a.upper() for a in sys.argv[1:] if a.upper() in ISOS]
        if target_isos:
            print(f"  Target ISOs: {target_isos}")

    run_isos = target_isos if target_isos else ISOS

    # Run ISOs in parallel
    n_workers = min(len(run_isos), cpu_count(), 5)
    worker_args = [(iso, demand_data, gen_profiles) for iso in run_isos]

    all_results = {}
    if n_workers > 1 and len(run_isos) > 1:
        print(f"\n  Running {len(run_isos)} ISOs in parallel ({n_workers} workers)...")
        with Pool(n_workers) as pool:
            results = pool.map(process_iso, worker_args)
        for iso_name, iso_results in results:
            all_results[iso_name] = iso_results
    else:
        # Sequential for single ISO or debugging
        for args in worker_args:
            iso_name, iso_results = process_iso(args)
            all_results[iso_name] = iso_results

    # Merge interim per-ISO Parquet files with persistent cache
    elapsed = time.time() - start_time
    cache_path = os.path.join(DATA_DIR, 'physics_cache_v4.parquet')

    # Collect interim files from this run + done parquets as fallback
    interim_tables = []
    for iso in run_isos:
        interim_path = os.path.join(CHECKPOINT_DIR, f'{iso}_v4_interim.parquet')
        if os.path.exists(interim_path):
            try:
                interim_tables.append(pq.read_table(interim_path))
            except Exception:
                pass
        # Fallback: also collect from per-threshold done parquets
        for threshold in THRESHOLDS:
            done_path = _threshold_done_path(iso, threshold)
            if os.path.exists(done_path):
                try:
                    interim_tables.append(pq.read_table(done_path))
                except Exception:
                    pass

    if interim_tables:
        # Merge all interim tables into one
        run_table = pa.concat_tables(interim_tables)
        print(f"\n  Interim cache: {run_table.num_rows:,} solutions from {len(interim_tables)} ISOs")

        # Backfill battery8 in interim tables if needed
        if 'battery8_dispatch_pct' not in run_table.column_names:
            zeros = [0] * run_table.num_rows
            run_table = run_table.append_column('battery8_dispatch_pct',
                                                 pa.array(zeros, type=pa.int64()))

        # Merge with persistent cache
        if os.path.exists(cache_path):
            try:
                existing = pq.read_table(cache_path)
                # Backfill battery8 in old cache if missing
                if 'battery8_dispatch_pct' not in existing.column_names:
                    zeros = [0] * existing.num_rows
                    existing = existing.append_column('battery8_dispatch_pct',
                                                       pa.array(zeros, type=pa.int64()))
                    print("  Backfilled battery8_dispatch_pct=0 for old cache entries")
                print(f"  Persistent cache: {existing.num_rows:,} existing solutions")
                merged = pa.concat_tables([existing, run_table])
            except Exception:
                merged = run_table
        else:
            merged = run_table

        # Dedup the merged table
        merged = _dedup_parquet_table(merged)
        pq.write_table(merged, cache_path, compression='snappy')
        size_mb = os.path.getsize(cache_path) / (1024 * 1024)
        print(f"  Cache saved: {cache_path} ({merged.num_rows:,} solutions, {size_mb:.1f} MB)")

        # Convert back to results dict for dashboard output
        all_results = _table_to_results(merged)

        # Clean up interim files
        for iso in run_isos:
            interim_path = os.path.join(CHECKPOINT_DIR, f'{iso}_v4_interim.parquet')
            if os.path.exists(interim_path):
                os.remove(interim_path)
    else:
        # No interim files — just merge in-memory results with cache
        all_results = merge_with_persistent_cache(all_results, cache_path)

    # Save dashboard output as Parquet
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard')
    os.makedirs(output_dir, exist_ok=True)

    parquet_path = os.path.join(output_dir, 'physics_results_v4.parquet')
    save_results_parquet(all_results, parquet_path)

    print(f"\n{'=' * 70}")
    print(f"  Complete in {elapsed:.1f}s")
    print(f"  Total candidates: {sum(len(t.get('candidates', [])) for iso in all_results.values() for t in iso.get('thresholds', {}).values())}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
