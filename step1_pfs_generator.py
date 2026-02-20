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
    from numba import njit
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
ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO']
ISO_LABELS = {
    'CAISO': 'CAISO (California)',
    'ERCOT': 'ERCOT (Texas)',
    'PJM': 'PJM (Mid-Atlantic)',
    'NYISO': 'NYISO (New York)',
    'NEISO': 'NEISO (New England)',
}

HYDRO_CAPS = {
    'CAISO': 9.5, 'ERCOT': 0.1, 'PJM': 1.8, 'NYISO': 15.9, 'NEISO': 4.4,
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
    100:  (100, 500),
}

# Nuclear seasonal derate
NUCLEAR_SHARE_OF_CLEAN_FIRM = {
    'CAISO': 0.70, 'ERCOT': 1.0, 'PJM': 1.0, 'NYISO': 1.0, 'NEISO': 1.0,
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
}

# Checkpoint directory
CHECKPOINT_DIR = os.path.join(DATA_DIR, 'checkpoints_v4')

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
    n = len(yearly_profiles)
    avg = [0.0] * H
    for profile in yearly_profiles:
        for h in range(H):
            avg[h] += profile[h]
    for h in range(H):
        avg[h] /= n
    return avg


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
    """Get generation profiles for 4D resource types (v4.0: no separate CCS)."""
    profiles = {}

    # Clean firm = nuclear seasonal-derated baseload
    # CCS sub-allocation handled by cost model; physics uses nuclear profile (conservative)
    nuc_share = NUCLEAR_SHARE_OF_CLEAN_FIRM.get(iso, 1.0)
    geo_share = 1.0 - nuc_share
    monthly_cf = NUCLEAR_MONTHLY_CF.get(iso, {m: 1.0 for m in range(1, 13)})
    cf_profile = []
    month_hours = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
    hour = 0
    for month_idx, hours_in_month in enumerate(month_hours):
        month_num = month_idx + 1
        nuc_cf = monthly_cf.get(month_num, 1.0)
        blended = nuc_share * nuc_cf + geo_share * 1.0
        for _ in range(hours_in_month):
            if hour < H:
                cf_profile.append(blended / H)
                hour += 1
    while len(cf_profile) < H:
        cf_profile.append(1.0 / H)
    profiles['clean_firm'] = cf_profile[:H]

    # Solar (with DST-aware nighttime correction)
    if iso == 'NYISO':
        p = gen_profiles[iso].get('solar_proxy')
        if not p:
            p = gen_profiles['NEISO'].get('solar')
        solar_raw = list(p[:H])
    else:
        solar_raw = list(gen_profiles[iso].get('solar', [0.0] * H)[:H])

    STD_UTC_OFFSETS = {'CAISO': 8, 'ERCOT': 6, 'PJM': 5, 'NYISO': 5, 'NEISO': 5}
    DST_START_DAY, DST_END_DAY = 69, 307
    local_start, local_end = 6, 19
    std_off = STD_UTC_OFFSETS.get(iso, 5)
    for day in range(H // 24):
        ds = day * 24
        is_dst = DST_START_DAY <= day < DST_END_DAY
        utc_off = std_off - (1 if is_dst else 0)
        utc_start = (local_start + utc_off) % 24
        utc_end = (local_end + utc_off) % 24
        for h_utc in range(24):
            idx = ds + h_utc
            if idx < len(solar_raw):
                if utc_start <= utc_end:
                    is_daylight = utc_start <= h_utc <= utc_end
                else:
                    is_daylight = h_utc >= utc_start or h_utc <= utc_end
                if not is_daylight:
                    solar_raw[idx] = 0.0
    profiles['solar'] = solar_raw

    # Wind
    profiles['wind'] = gen_profiles[iso].get('wind', [0.0] * H)[:H]

    # Hydro
    profiles['hydro'] = gen_profiles[iso].get('hydro', [0.0] * H)[:H]

    # Ensure all profiles are exactly H hours, no negatives
    for rtype in RESOURCE_TYPES:
        p = profiles[rtype]
        if len(p) > H:
            p = p[:H]
        elif len(p) < H:
            p = p + [0.0] * (H - len(p))
        profiles[rtype] = [max(0.0, v) for v in p]

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
def _score_hourly(demand, supply_row, procurement):
    """Base hourly matching score. supply_row is already mix-weighted (8760,)."""
    total = 0.0
    for h in range(8760):
        s = procurement * supply_row[h]
        total += min(demand[h], s)
    return total


@njit(cache=True)
def _score_with_battery(demand, supply_row, procurement,
                        batt_capacity, batt_power, batt_eff):
    """Hourly score + battery daily-cycle dispatch with power limits."""
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

    if batt_capacity <= 0:
        return base_matched

    total_dispatched = 0.0
    for day in range(365):
        ds = day * 24
        # Charge phase
        stored = 0.0
        for h in range(24):
            s = surplus[ds + h]
            if s > 0 and stored < batt_capacity:
                charge = s
                if charge > batt_power:
                    charge = batt_power
                remaining = batt_capacity - stored
                if charge > remaining:
                    charge = remaining
                stored += charge
        # Discharge phase
        available = stored * batt_eff
        for h in range(24):
            g = gap[ds + h]
            if g > 0 and available > 0:
                discharge = g
                if discharge > batt_power:
                    discharge = batt_power
                if discharge > available:
                    discharge = available
                total_dispatched += discharge
                available -= discharge

    return base_matched + total_dispatched


@njit(cache=True)
def _score_with_both_storage(demand, supply_row, procurement,
                             batt_capacity, batt_power, batt_eff,
                             ldes_capacity, ldes_power, ldes_eff,
                             ldes_window_hours):
    """Hourly score + battery (daily) + LDES (multi-day rolling window)."""
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

    # Phase 1: Battery daily cycle on residual surplus/gap
    batt_dispatched = 0.0
    residual_surplus = np.copy(surplus)
    residual_gap = np.copy(gap)

    if batt_capacity > 0:
        for day in range(365):
            ds = day * 24
            stored = 0.0
            # Charge
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
            # Discharge
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

    # Phase 2: LDES multi-day rolling window on post-battery residual
    ldes_dispatched = 0.0
    if ldes_capacity > 0:
        soc = 0.0
        n_windows = (8760 + ldes_window_hours - 1) // ldes_window_hours
        for w in range(n_windows):
            ws = w * ldes_window_hours
            we = ws + ldes_window_hours
            if we > 8760:
                we = 8760
            # Charge phase
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
            # Discharge phase
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

    return base_matched + batt_dispatched + ldes_dispatched


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
# BATCH EVALUATION — vectorized for grid search
# ══════════════════════════════════════════════════════════════════════════════

def batch_hourly_scores(demand_arr, supply_matrix, mix_batch, procurement):
    """Evaluate N mixes at once. mix_batch shape (N, 4), returns (N,) scores."""
    # (N, 4) @ (4, 8760) = (N, 8760) — all mixes in one matrix multiply
    supply_batch = procurement * (mix_batch @ supply_matrix)
    matched = np.minimum(demand_arr, supply_batch)
    return matched.sum(axis=1)


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
    """
    combos = []
    hydro_max = min(int(hydro_cap), max_single)
    for hyd in range(0, hydro_max + 1, step):
        remainder = 100 - hyd
        for cf in range(0, min(max_single + 1, remainder + 1), step):
            for sol in range(0, min(max_single + 1, remainder - cf + 1), step):
                wnd = remainder - cf - sol
                if 0 <= wnd <= max_single:
                    combos.append([cf, sol, wnd, hyd])
    return np.array(combos, dtype=np.float64)


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
                       prev_pruning=None, cross_solutions=None):
    """Find ALL feasible solutions for a single threshold × ISO.

    Uses cross-threshold pruning: mixes that were infeasible at a lower threshold
    are skipped. Each mix's min-feasible procurement from the previous threshold
    becomes the floor for the procurement sweep.

    Cross-threshold pollination: solutions from lower thresholds that over-achieved
    (hourly_match_score >= this threshold) are pre-seeded as candidates. Mixes
    proven feasible without storage skip the expensive Phase 1b storage sweep.
    Specific (mix, storage) combos already proven are also skipped.

    Args:
        prev_pruning: dict from previous threshold with:
            - 'feasible_mixes': set of mix tuples that were feasible
            - 'min_proc': dict mapping mix_tuple -> minimum procurement that worked
            - 'all_mixes': set of all mix tuples tested
            If None, no pruning (first threshold).
        cross_solutions: list of candidate dicts from lower thresholds where
            hourly_match_score >= this threshold. Pre-seeded as candidates.

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

    # Storage levels to sweep (all thresholds get full sweep — no pruning)
    batt_levels = [0, 2, 5, 8, 10, 15, 20]
    batt8_levels = [0, 2, 5, 8, 10, 15, 20]
    ldes_levels = [0, 2, 5, 8, 10, 15, 20]

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

    candidates = []
    seen = set()  # Dedup key: (mix_tuple, proc, bp, lp)

    # ── Cross-threshold pollination: pre-seed from over-achieving lower-threshold solutions ──
    cross_no_storage_feasible = set()  # Mix keys feasible WITHOUT storage (skip Phase 1b entirely)
    cross_storage_known = set()        # (mix_key, bp, b8p, lp) combos already proven (skip in Phase 1b)
    n_cross_seeded = 0

    if cross_solutions:
        for cs in cross_solutions:
            proc = cs['procurement_pct']
            # Only add if procurement is within this threshold's bounds
            if proc < proc_min or proc > proc_max:
                continue
            mix = cs['resource_mix']
            mix_key = (mix['clean_firm'], mix['solar'], mix['wind'], mix['hydro'])
            bp = cs['battery_dispatch_pct']
            b8p = cs.get('battery8_dispatch_pct', 0)
            lp = cs['ldes_dispatch_pct']
            score_val = cs['hourly_match_score'] / 100.0

            if score_val >= target:
                # Pre-seed as candidate
                mix_arr = np.array([mix['clean_firm'], mix['solar'], mix['wind'], mix['hydro']], dtype=np.float64)
                key = (mix_key, proc, bp, b8p, lp)
                if key not in seen:
                    seen.add(key)
                    candidates.append({
                        'resource_mix': {rt: mix_key[j] for j, rt in enumerate(RESOURCE_TYPES)},
                        'procurement_pct': proc,
                        'battery_dispatch_pct': bp,
                        'battery8_dispatch_pct': b8p,
                        'ldes_dispatch_pct': lp,
                        'hourly_match_score': round(score_val * 100, 2),
                    })
                    n_cross_seeded += 1

                    # Track for skip logic
                    if bp == 0 and b8p == 0 and lp == 0:
                        cross_no_storage_feasible.add(mix_key)
                    else:
                        cross_storage_known.add((mix_key, bp, b8p, lp))

        if n_cross_seeded > 0:
            print(f"      Cross-pollinated {n_cross_seeded} solutions "
                  f"({len(cross_no_storage_feasible)} no-storage mixes, "
                  f"{len(cross_storage_known)} storage combos)")

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

    # Phase 1a: No-storage sweep — per-mix procurement early stopping
    # For each mix, sweep procurement upward from the floor (previous threshold's
    # min-feasible procurement). Once target is met, stop.
    near_miss_mixes = {}  # mix_index -> best_score (for storage sweep)
    mix_min_proc = {}     # mix_tuple -> min procurement that achieved this threshold
    all_mix_keys = set()  # All mixes tested (for cross-threshold pruning)
    feasible_mix_keys = set()  # Mixes that achieved this threshold

    # Inherit cross-pollinated feasibility into tracking structures
    for mk in cross_no_storage_feasible:
        feasible_mix_keys.add(mk)

    for i in range(n_combos):
        mix_key = (int(combos_5[i][0]), int(combos_5[i][1]),
                   int(combos_5[i][2]), int(combos_5[i][3]))
        all_mix_keys.add(mix_key)

        # Start procurement at previous threshold's min-feasible (if known)
        floor = prev_min_proc.get(mix_key, proc_min)
        floor = max(floor, proc_min)  # Never go below this threshold's min

        for proc in proc_levels:
            if proc < floor:
                continue  # Skip below floor
            pf = proc / 100.0
            supply_scaled = supply_rows[i] * pf
            score = np.sum(np.minimum(supply_scaled / demand_arr, 1.0)) / H
            if score >= target:
                add_candidate(combos_5[i], proc, 0, 0, 0, score)
                mix_min_proc[mix_key] = min(mix_min_proc.get(mix_key, 9999), proc)
                feasible_mix_keys.add(mix_key)
                break  # Early stop: higher procurement only adds cost
            elif score >= target - 0.15:
                # Skip near-miss if mix is already proven feasible from cross-pollination
                if mix_key not in cross_no_storage_feasible:
                    near_miss_mixes[i] = max(near_miss_mixes.get(i, 0), score)

    # Phase 1b: Storage sweep on near-miss mixes
    # For each (mix, storage_config), sweep procurement upward with early stopping.
    # Triple-nested: battery4 × battery8 × LDES (dispatch order: 4hr → 8hr → LDES).
    for i in near_miss_mixes:
        supply_row = supply_rows[i]
        mix = combos_5[i]
        mix_key = (int(mix[0]), int(mix[1]), int(mix[2]), int(mix[3]))

        # Battery4 only — for each battery level, sweep procurement with early stop
        for bp in batt_levels:
            if bp == 0:
                continue
            batt_cap = bp / 100.0
            batt_pow = batt_cap / BATTERY_DURATION_HOURS
            for proc in proc_levels:
                pf = proc / 100.0
                score = _score_with_battery(demand_arr, supply_row, pf,
                                            batt_cap, batt_pow, batt_eff)
                if score >= target:
                    add_candidate(mix, proc, bp, 0, 0, score)
                    mix_min_proc[mix_key] = min(mix_min_proc.get(mix_key, 9999), proc)
                    feasible_mix_keys.add(mix_key)
                    break  # Early stop per (mix, battery_level)

        # Full storage combos: battery4 × battery8 × LDES
        # Early stop per (mix, batt4, batt8, ldes) combo
        for bp in batt_levels:
            batt_cap = bp / 100.0
            batt_pow = batt_cap / BATTERY_DURATION_HOURS
            for b8p in batt8_levels:
                batt8_cap = b8p / 100.0
                batt8_pow = batt8_cap / BATTERY8_DURATION_HOURS
                for lp in ldes_levels:
                    # Skip no-storage (already covered) and battery4-only (covered above)
                    if bp == 0 and b8p == 0 and lp == 0:
                        continue
                    if b8p == 0 and lp == 0:
                        continue  # Battery4-only already covered above
                    # Skip combos already proven from cross-pollination
                    if (mix_key, bp, b8p, lp) in cross_storage_known:
                        continue
                    ldes_cap = lp / 100.0
                    ldes_pow = ldes_cap / LDES_DURATION_HOURS
                    for proc in proc_levels:
                        pf = proc / 100.0
                        score = _score_with_all_storage(
                            demand_arr, supply_row, pf,
                            batt_cap, batt_pow, batt_eff,
                            batt8_cap, batt8_pow, batt8_eff,
                            ldes_cap, ldes_pow, ldes_eff, ldes_window_hours)
                        if score >= target:
                            add_candidate(mix, proc, bp, b8p, lp, score)
                            mix_min_proc[mix_key] = min(mix_min_proc.get(mix_key, 9999), proc)
                            feasible_mix_keys.add(mix_key)
                            break  # Early stop per (mix, batt4, batt8, ldes)

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

            # No-storage: per-mix procurement early stop
            for j in range(n_fine):
                for proc in proc_levels:
                    pf = proc / 100.0
                    supply_scaled = fine_supply[j] * pf
                    score = np.sum(np.minimum(supply_scaled / demand_arr, 1.0)) / H
                    if score >= target:
                        add_candidate(fine_combos[j], proc, 0, 0, 0, score)
                        break  # Early stop
                    elif score >= target - 0.10:
                        # Near-miss: try storage with early stop per (mix, batt4)
                        supply_row_j = fine_supply[j]
                        for bp in [2, 5, 10]:
                            batt_cap = bp / 100.0
                            batt_pow = batt_cap / BATTERY_DURATION_HOURS
                            sc = _score_with_battery(demand_arr, supply_row_j, pf,
                                                     batt_cap, batt_pow, batt_eff)
                            if sc >= target:
                                add_candidate(fine_combos[j], proc, bp, 0, 0, sc)
                        # Also try battery8 alone in refinement
                        for b8p in [2, 5, 10]:
                            batt8_cap = b8p / 100.0
                            batt8_pow = batt8_cap / BATTERY8_DURATION_HOURS
                            sc = _score_with_all_storage(
                                demand_arr, supply_row_j, pf,
                                0.0, 0.0, batt_eff,
                                batt8_cap, batt8_pow, batt8_eff,
                                0.0, 0.0, ldes_eff, ldes_window_hours)
                            if sc >= target:
                                add_candidate(fine_combos[j], proc, 0, b8p, 0, sc)

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
    """Process all thresholds for a single ISO. Designed for multiprocessing."""
    iso, demand_data, gen_profiles = args
    iso_start = time.time()

    demand_norm = demand_data[iso]['normalized']
    supply_profiles = get_supply_profiles(iso, gen_profiles)
    demand_arr, supply_matrix = prepare_numpy_profiles(demand_norm, supply_profiles)
    hydro_cap = HYDRO_CAPS[iso]

    print(f"\n  {iso}: Starting optimization ({len(THRESHOLDS)} thresholds, "
          f"hydro_cap={hydro_cap}%)")

    # Check for checkpoint
    checkpoint = load_checkpoint(iso)
    completed_thresholds = set(checkpoint.get('completed', []))

    iso_results = {
        'iso': iso,
        'label': ISO_LABELS.get(iso, iso),
        'annual_demand_mwh': demand_data[iso]['total_annual_mwh'],
        'peak_demand_mw': demand_data[iso]['peak_mw'],
        'hydro_cap': hydro_cap,
        'thresholds': checkpoint.get('thresholds', {}),
    }

    # ── Cross-threshold pollination: mine existing cache + accumulate solutions ──
    all_solutions = []  # Accumulator: list of candidate dicts with actual scores

    # Mine the interim parquet cache from previous/current run
    interim_path = os.path.join(CHECKPOINT_DIR, f'{iso}_v4_interim.parquet')
    if os.path.exists(interim_path) and HAS_PARQUET:
        try:
            cached = pq.read_table(interim_path)
            n_cached = cached.num_rows
            if n_cached > 0:
                # Extract all solutions for cross-pollination mining
                c_cf = cached.column('clean_firm').to_pylist()
                c_sol = cached.column('solar').to_pylist()
                c_wnd = cached.column('wind').to_pylist()
                c_hyd = cached.column('hydro').to_pylist()
                c_proc = cached.column('procurement_pct').to_pylist()
                c_bat = cached.column('battery_dispatch_pct').to_pylist()
                c_bat8 = (cached.column('battery8_dispatch_pct').to_pylist()
                          if 'battery8_dispatch_pct' in cached.column_names
                          else [0] * n_cached)
                c_ldes = cached.column('ldes_dispatch_pct').to_pylist()
                c_score = cached.column('hourly_match_score').to_pylist()

                for i in range(n_cached):
                    all_solutions.append({
                        'resource_mix': {
                            'clean_firm': c_cf[i], 'solar': c_sol[i],
                            'wind': c_wnd[i], 'hydro': c_hyd[i],
                        },
                        'procurement_pct': c_proc[i],
                        'battery_dispatch_pct': c_bat[i],
                        'battery8_dispatch_pct': c_bat8[i],
                        'ldes_dispatch_pct': c_ldes[i],
                        'hourly_match_score': c_score[i],
                    })
                print(f"    {iso}: Mined {n_cached:,} cached solutions for cross-pollination")
        except Exception as e:
            print(f"    {iso}: Could not read interim cache: {e}")

    prev_pruning = None  # Cross-threshold pruning state
    for threshold in THRESHOLDS:
        t_str = str(threshold)
        if t_str in completed_thresholds:
            print(f"    {iso} {threshold}%: loaded from checkpoint — skipping")
            continue

        # Build cross-solutions: all previously found solutions that qualify for this threshold
        cross_solutions = [s for s in all_solutions if s['hourly_match_score'] >= threshold]

        t_start = time.time()
        feasible, pruning_info = optimize_threshold(
            iso, threshold, demand_arr, supply_matrix, hydro_cap,
            prev_pruning=prev_pruning, cross_solutions=cross_solutions if cross_solutions else None)
        prev_pruning = pruning_info  # Pass to next threshold

        # Accumulate new solutions for cross-pollination to higher thresholds
        for c in feasible:
            all_solutions.append(c)
        t_elapsed = time.time() - t_start

        # Count unique mix archetypes
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

        # Checkpoint after each threshold + append to interim Parquet cache
        save_checkpoint(iso, iso_results, t_str)
        append_threshold_to_cache(iso, threshold, feasible)

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

    # Warm up Numba JIT (first call compiles)
    if HAS_NUMBA:
        print("  Warming up Numba JIT...")
        dummy_demand = np.ones(H) / H
        dummy_supply = np.ones(H) / H
        _score_hourly(dummy_demand, dummy_supply, 1.0)
        _score_with_battery(dummy_demand, dummy_supply, 1.0, 0.01, 0.0025, 0.85)
        _score_with_both_storage(dummy_demand, dummy_supply, 1.0,
                                 0.01, 0.0025, 0.85, 0.01, 0.0001, 0.50, 168)
        _score_with_all_storage(dummy_demand, dummy_supply, 1.0,
                                0.01, 0.0025, 0.85,
                                0.01, 0.00125, 0.85,
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

    # Collect interim files from this run
    interim_tables = []
    for iso in run_isos:
        interim_path = os.path.join(CHECKPOINT_DIR, f'{iso}_v4_interim.parquet')
        if os.path.exists(interim_path):
            try:
                interim_tables.append(pq.read_table(interim_path))
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
