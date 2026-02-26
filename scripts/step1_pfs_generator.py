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
  - 15 thresholds: 50, 55, 60, 65, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 100
  - Adaptive grid search: 5% → 1% refinement (±4% radius)
  - Pareto frontier: 3-5 points per threshold×ISO (procurement/storage tradeoff)
  - Numba JIT-compiled scoring functions
  - Parallel ISO execution (multiprocessing)
  - Vectorized batch mix evaluation

Output: data/step1-pfs-parquets/{ISO}_t{XX}_raw_pfs.parquet (per ISO/threshold)

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

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
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

# 15 thresholds (v4.1: added 55, 65 for finer low-range granularity)
THRESHOLDS = [50, 55, 60, 65, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 99.5, 99.9, 100]

# Threshold-adaptive procurement bounds (Decision 3C, expanded)
# 90-99%: capped at 250% per user direction (high enough for extreme renewables,
# but not wastefully wide). 100%: pushed to 500% for perfect hourly matching.
# Procurement bounds — raised for lower thresholds to allow higher RE saturation
# with storage. A pure solar mix at 250% procurement + right-sized battery can
# hit 80%+ targets — the old caps (150-200%) blocked this solution space.
PROCUREMENT_BOUNDS = {
    50:   (50, 200),
    55:   (55, 212),
    60:   (60, 225),
    65:   (65, 237),
    70:   (70, 250),
    75:   (75, 250),
    80:   (80, 250),
    85:   (85, 250),
    87.5: (87, 250),
    90:   (90, 250),
    92.5: (92, 250),
    95:   (95, 250),
    97.5: (100, 250),
    99:   (100, 250),
    99.5: (100, 300),
    99.9: (100, 350),
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
STEP1_RAW_PFS_PARQUET_DIR = os.path.join(DATA_DIR, 'step1-pfs-parquets')

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

    with open(os.path.join(DATA_DIR, 'EIA 930 Data', 'eia_demand_profiles.json')) as f:
        demand_raw = json.load(f)
    with open(os.path.join(DATA_DIR, 'EIA 930 Data', 'eia_generation_profiles.json')) as f:
        gen_raw = json.load(f)
    with open(os.path.join(DATA_DIR, 'egrid_emission_rates.json')) as f:
        emission_rates = json.load(f)
    with open(os.path.join(DATA_DIR, 'EIA 930 Data', 'eia_fossil_mix.json')) as f:
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
    profiles['clean_firm'] = cf_profile

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
    profiles['solar'] = solar_arr

    # Wind
    profiles['wind'] = np.array(gen_profiles[iso].get('wind', [0.0] * H)[:H], dtype=np.float64)

    # Hydro
    profiles['hydro'] = np.array(gen_profiles[iso].get('hydro', [0.0] * H)[:H], dtype=np.float64)

    # Ensure all profiles are exactly H hours, no negatives (kept as numpy arrays)
    for rtype in RESOURCE_TYPES:
        arr = np.asarray(profiles[rtype], dtype=np.float64)
        if len(arr) > H:
            arr = arr[:H].copy()
        elif len(arr) < H:
            arr = np.pad(arr, (0, H - len(arr)))
        np.maximum(arr, 0.0, out=arr)
        profiles[rtype] = arr

    return profiles


def prepare_numpy_profiles(demand_norm, supply_profiles):
    """Convert to numpy arrays + build supply matrix (4, 8760).

    Re-normalizes demand_arr to sum to exactly 1.0 to fix a scaling issue
    where leap-year profiles (2024) were normalized over 8784 hours then
    truncated to 8760, causing the average to sum to ~0.9995 instead of 1.0.
    Without this fix, hourly_match_score caps at 99.95% even with perfect
    hourly matching.
    """
    demand_arr = np.array(demand_norm[:H], dtype=np.float64)
    dsum = demand_arr.sum()
    if dsum > 0 and abs(dsum - 1.0) > 1e-9:
        demand_arr *= (1.0 / dsum)
    supply_matrix = np.stack([
        np.asarray(supply_profiles[rt][:H], dtype=np.float64)
        for rt in RESOURCE_TYPES
    ])  # shape (4, 8760)
    return demand_arr, supply_matrix


# ══════════════════════════════════════════════════════════════════════════════
# SCORING FUNCTIONS — Numba JIT compiled
# ══════════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def _compute_storage_caps(demand, supply_row, procurement,
                          batt4_duration, batt8_duration, ldes_duration):
    """Compute max useful capacity and curtailment frequency for storage screening.

    Energy-based ceiling: the max useful battery capacity equals the total surplus
    energy available in its charge window (1 day for bat4, 2 days for bat8,
    7 days for LDES). A battery can charge over multiple surplus hours — the
    ceiling is the total window surplus, not max_hourly × duration.

    A battery larger than max_window_surplus has idle capacity that can never fill
    in any single charge cycle. Power = cap/duration may exceed max_hourly_surplus,
    but that's OK — the surplus rate is the binding constraint, and the battery
    charges at min(surplus, power) per hour.

    Curtailment frequency: counts days with non-trivial surplus. Daily-cycle
    batteries need FREQUENT curtailment (150+ days) to justify capacity.

    Returns: (bat4_cap, bat8_cap, ldes_cap, has_curtailment, surplus_days)
        bat4_cap: max useful bat4 capacity = max daily surplus (normalized)
        bat8_cap: max useful bat8 capacity = max 2-day surplus (normalized)
        ldes_cap: max useful LDES capacity = max 7-day surplus (normalized)
        has_curtailment: True if any hour has surplus > 0
        surplus_days: number of days with surplus > 0.1% of daily demand
    """
    has_curtailment = False
    surplus_days = 0
    daily_demand_avg = 1.0 / 365.0
    surplus_threshold = daily_demand_avg * 0.001

    # Pass 1: compute daily surplus totals and surplus_days count
    daily_surpluses = np.empty(365)
    for day in range(365):
        ds = day * 24
        day_surplus = 0.0
        for h in range(24):
            s = procurement * supply_row[ds + h] - demand[ds + h]
            if s > 0:
                day_surplus += s
                has_curtailment = True
        daily_surpluses[day] = day_surplus
        if day_surplus > surplus_threshold:
            surplus_days += 1

    # Bat4: max daily surplus (battery charges over full day, discharges same day)
    max_daily = 0.0
    for day in range(365):
        if daily_surpluses[day] > max_daily:
            max_daily = daily_surpluses[day]
    bat4_cap = max_daily

    # Bat8: max 2-day surplus (48hr charge window)
    max_2day = 0.0
    for w in range(0, 365, 2):
        window = daily_surpluses[w]
        if w + 1 < 365:
            window += daily_surpluses[w + 1]
        if window > max_2day:
            max_2day = window
    bat8_cap = max_2day

    # LDES: max 7-day surplus (168hr charge window)
    max_7day = 0.0
    for w in range(0, 365, 7):
        window = 0.0
        for d in range(w, min(w + 7, 365)):
            window += daily_surpluses[d]
        if window > max_7day:
            max_7day = window
    ldes_cap = max_7day

    return bat4_cap, bat8_cap, ldes_cap, has_curtailment, surplus_days


@njit(cache=True, parallel=True)
def _batch_compute_storage_caps(demand, supply_rows, procurement, N,
                                 batt4_duration, batt8_duration, ldes_duration):
    """Compute storage caps for N mixes in parallel using Numba prange.

    Replaces per-mix Python loop with a single parallel kernel call.
    Each mix is independent — Numba distributes across CPU cores.

    Returns: (bat4_caps, bat8_caps, ldes_caps, has_curtailment, surplus_days)
        Each is a 1D array of length N.
    """
    bat4_caps = np.empty(N, dtype=np.float64)
    bat8_caps = np.empty(N, dtype=np.float64)
    ldes_caps = np.empty(N, dtype=np.float64)
    has_curtailment_arr = np.empty(N, dtype=np.int64)
    surplus_days_arr = np.empty(N, dtype=np.int64)

    for i in prange(N):
        b4, b8, lc, hc, sd = _compute_storage_caps(
            demand, supply_rows[i], procurement,
            batt4_duration, batt8_duration, ldes_duration)
        bat4_caps[i] = b4
        bat8_caps[i] = b8
        ldes_caps[i] = lc
        has_curtailment_arr[i] = 1 if hc else 0
        surplus_days_arr[i] = sd

    return bat4_caps, bat8_caps, ldes_caps, has_curtailment_arr, surplus_days_arr


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

    # Phase 2: Battery 8hr on 2-day (48hr) window cycle on post-4hr residual
    # 8hr batteries cycle ~200×/year (~every 1.8 days), not daily.
    # Using 2-day windows allows accumulating surplus across 2 days before
    # discharging, better reflecting actual 8hr battery operations.
    batt8_dispatched = 0.0
    if batt8_capacity > 0:
        batt8_window = 48  # 2-day window
        n_win8 = (8760 + batt8_window - 1) // batt8_window
        for w in range(n_win8):
            ws = w * batt8_window
            we = ws + batt8_window
            if we > 8760:
                we = 8760
            stored = 0.0
            for h in range(ws, we):
                s = residual_surplus[h]
                if s > 0 and stored < batt8_capacity:
                    charge = s
                    if charge > batt8_power:
                        charge = batt8_power
                    remaining = batt8_capacity - stored
                    if charge > remaining:
                        charge = remaining
                    stored += charge
                    residual_surplus[h] -= charge
            available = stored * batt8_eff
            for h in range(ws, we):
                g = residual_gap[h]
                if g > 0 and available > 0:
                    discharge = g
                    if discharge > batt8_power:
                        discharge = batt8_power
                    if discharge > available:
                        discharge = available
                    batt8_dispatched += discharge
                    available -= discharge
                    residual_gap[h] -= discharge

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


@njit(cache=True)
def _batch_storage_scores(demand, supply_row, procurement,
                          bat4_levels, bat8_levels, ldes_levels_arr,
                          n_b4, n_b8, n_ldes,
                          batt_eff, batt8_eff, ldes_eff,
                          batt4_dur, batt8_dur, ldes_dur,
                          ldes_window_hours, batt8_window):
    """Evaluate ALL storage combos for a single (mix, procurement) in one call.

    Instead of calling _score_with_all_storage N times, this function:
    1. Computes base supply/surplus/gap once
    2. For each bat4 level, dispatches bat4 → gets residual (reused for all bat8)
    3. For each bat8 level, dispatches bat8 → gets residual (reused for all LDES)
    4. For each LDES level, dispatches LDES → computes score

    Returns flat array of scores with shape (n_b4 * n_b8 * n_ldes,).
    Index mapping: scores[b4_idx * n_b8 * n_ldes + b8_idx * n_ldes + l_idx]
    """
    scores = np.full(n_b4 * n_b8 * n_ldes, -1.0)  # -1 = not evaluated

    # Base supply and matching
    supply = np.empty(8760)
    base_surplus = np.empty(8760)
    base_gap = np.empty(8760)
    base_matched = 0.0
    for h in range(8760):
        s = procurement * supply_row[h]
        supply[h] = s
        d = demand[h]
        base_matched += min(d, s)
        if s > d:
            base_surplus[h] = s - d
            base_gap[h] = 0.0
        else:
            base_surplus[h] = 0.0
            base_gap[h] = d - s

    for b4_idx in range(n_b4):
        bp = bat4_levels[b4_idx]
        batt_cap = bp / 100.0
        batt_pow = batt_cap / batt4_dur if batt_cap > 0 else 0.0

        # Dispatch bat4 and get residual
        res_surplus4 = base_surplus.copy()
        res_gap4 = base_gap.copy()
        batt_dispatched = 0.0

        if batt_cap > 0:
            for day in range(365):
                ds = day * 24
                stored = 0.0
                for h in range(24):
                    s = res_surplus4[ds + h]
                    if s > 0 and stored < batt_cap:
                        charge = min(s, batt_pow, batt_cap - stored)
                        stored += charge
                        res_surplus4[ds + h] -= charge
                available = stored * batt_eff
                for h in range(24):
                    g = res_gap4[ds + h]
                    if g > 0 and available > 0:
                        discharge = min(g, batt_pow, available)
                        batt_dispatched += discharge
                        available -= discharge
                        res_gap4[ds + h] -= discharge

        for b8_idx in range(n_b8):
            b8p = bat8_levels[b8_idx]
            batt8_cap = b8p / 100.0
            batt8_pow = batt8_cap / batt8_dur if batt8_cap > 0 else 0.0

            # Dispatch bat8 on post-bat4 residual (2-day window)
            res_surplus8 = res_surplus4.copy()
            res_gap8 = res_gap4.copy()
            batt8_dispatched = 0.0

            if batt8_cap > 0:
                n_win8 = (8760 + batt8_window - 1) // batt8_window
                for w in range(n_win8):
                    ws = w * batt8_window
                    we = ws + batt8_window
                    if we > 8760:
                        we = 8760
                    stored = 0.0
                    for h in range(ws, we):
                        s = res_surplus8[h]
                        if s > 0 and stored < batt8_cap:
                            charge = min(s, batt8_pow, batt8_cap - stored)
                            stored += charge
                            res_surplus8[h] -= charge
                    available = stored * batt8_eff
                    for h in range(ws, we):
                        g = res_gap8[h]
                        if g > 0 and available > 0:
                            discharge = min(g, batt8_pow, available)
                            batt8_dispatched += discharge
                            available -= discharge
                            res_gap8[h] -= discharge

            for l_idx in range(n_ldes):
                lp = ldes_levels_arr[l_idx]
                if bp == 0 and b8p == 0 and lp == 0:
                    continue
                ldes_cap = lp / 100.0
                ldes_pow = ldes_cap / ldes_dur if ldes_cap > 0 else 0.0

                # Dispatch LDES on post-battery residual
                ldes_dispatched = 0.0
                if ldes_cap > 0:
                    soc = 0.0
                    n_windows = (8760 + ldes_window_hours - 1) // ldes_window_hours
                    for w in range(n_windows):
                        ws = w * ldes_window_hours
                        we = ws + ldes_window_hours
                        if we > 8760:
                            we = 8760
                        for h in range(ws, we):
                            s = res_surplus8[h]
                            if s > 0 and soc < ldes_cap:
                                charge = min(s, ldes_pow, ldes_cap - soc)
                                soc += charge
                        for h in range(ws, we):
                            g = res_gap8[h]
                            if g > 0 and soc > 0:
                                ae = soc * ldes_eff
                                discharge = min(g, ldes_pow, ae)
                                ldes_dispatched += discharge
                                soc -= discharge / ldes_eff

                idx = b4_idx * n_b8 * n_ldes + b8_idx * n_ldes + l_idx
                scores[idx] = base_matched + batt_dispatched + batt8_dispatched + ldes_dispatched

    return scores


@njit(cache=True, parallel=True)
def _batch_mixes_storage_screen(demand, supply_rows_batch, procurement, n_mixes,
                                 bat4_levels, bat8_levels, ldes_levels,
                                 n_b4, n_b8, n_ldes,
                                 batt_eff, batt8_eff, ldes_eff,
                                 batt4_dur, batt8_dur, ldes_dur,
                                 ldes_window_hours, batt8_window):
    """Screen N mixes × all storage combos at max procurement, in parallel.

    Uses Numba prange to distribute mixes across CPU cores. Each mix evaluates
    all bat4×bat8×LDES combos, reusing dispatch intermediates via
    _batch_storage_scores.

    Args:
        supply_rows_batch: (n_mixes, 8760) array of pre-mixed supply profiles
        Other args: same as _batch_storage_scores

    Returns: (n_mixes, n_combos) array of scores at max procurement. -1 = skipped.
    """
    n_combos = n_b4 * n_b8 * n_ldes
    all_scores = np.full((n_mixes, n_combos), -1.0)

    for i in prange(n_mixes):
        mix_scores = _batch_storage_scores(
            demand, supply_rows_batch[i], procurement,
            bat4_levels, bat8_levels, ldes_levels,
            n_b4, n_b8, n_ldes,
            batt_eff, batt8_eff, ldes_eff,
            batt4_dur, batt8_dur, ldes_dur,
            ldes_window_hours, batt8_window)
        for j in range(n_combos):
            all_scores[i, j] = mix_scores[j]

    return all_scores


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


def vectorized_procurement_sweep(demand_arr, supply_rows, target, proc_levels, floors=None):
    """Find minimum feasible procurement for each mix via batch level sweep.

    Instead of per-mix binary search (N × log(P) individual numpy ops), evaluates
    ALL mixes at each procurement level in a single vectorized numpy operation
    (P batch ops total). For typical N=500 mixes and P=20 levels, this is ~50x
    fewer Python-level operations.

    Args:
        demand_arr: (H,) normalized demand profile
        supply_rows: (N, H) pre-computed supply profiles for each mix
        target: feasibility threshold (e.g., 0.90 for 90%)
        proc_levels: list of procurement levels to sweep (low → high)
        floors: (N,) optional per-mix procurement floor (from prev threshold)

    Returns:
        min_procs: (N,) minimum feasible procurement level per mix
        final_scores: (N,) hourly match score at minimum procurement
    """
    n_mixes = len(supply_rows)
    max_proc = proc_levels[-1]
    min_procs = np.full(n_mixes, max_proc, dtype=np.float64)
    if floors is None:
        floors = np.full(n_mixes, float(proc_levels[0]))

    # Sweep procurement levels low → high; record first feasible per mix
    for proc in proc_levels:
        pf = proc / 100.0
        # One vectorized numpy op evaluates all mixes at this procurement
        scores = np.minimum(demand_arr, supply_rows * pf).sum(axis=1)
        newly_feasible = (
            (scores >= target) &
            (min_procs == max_proc) &
            (float(proc) >= floors)
        )
        min_procs[newly_feasible] = proc

    # Compute final scores grouped by unique procurement level
    unique_procs = np.unique(min_procs)
    final_scores = np.empty(n_mixes)
    for up in unique_procs:
        mask = min_procs == up
        pf = up / 100.0
        final_scores[mask] = np.minimum(
            demand_arr, supply_rows[mask] * pf
        ).sum(axis=1)

    return min_procs.astype(int), final_scores


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
    """Generate 4D combos in neighborhood of base_combo.

    Uses vectorized meshgrid instead of 4 nested Python loops.
    Wind is derived from the sum=100 constraint, then filtered to be
    within its valid range. np.unique handles deduplication.
    """
    hydro_max = min(int(hydro_cap), max_single)
    base = [int(base_combo[i]) for i in range(4)]

    # Per-dimension ranges (cf, sol, wnd, hyd)
    caps = [max_single, max_single, max_single, hydro_max]
    lo = [max(0, base[i] - radius * step) for i in range(4)]
    hi = [min(caps[i], base[i] + radius * step) for i in range(4)]

    # Meshgrid over cf, sol, hyd — wind derived from sum=100
    cf_vals = np.arange(lo[0], hi[0] + 1, step)
    sol_vals = np.arange(lo[1], hi[1] + 1, step)
    hyd_vals = np.arange(lo[3], hi[3] + 1, step)

    cf_grid, sol_grid, hyd_grid = np.meshgrid(cf_vals, sol_vals, hyd_vals, indexing='ij')
    cf_flat = cf_grid.ravel()
    sol_flat = sol_grid.ravel()
    hyd_flat = hyd_grid.ravel()
    wnd_flat = 100 - cf_flat - sol_flat - hyd_flat

    # Filter: wind must be in valid neighborhood range and within bounds
    valid = (
        (wnd_flat >= 0) & (wnd_flat <= max_single) &
        (wnd_flat >= lo[2]) & (wnd_flat <= hi[2])
    )

    if not np.any(valid):
        return np.empty((0, 4))

    combos = np.column_stack([cf_flat[valid], sol_flat[valid],
                              wnd_flat[valid], hyd_flat[valid]])
    return np.unique(combos, axis=0).astype(np.float64)


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
                       flush_callback=None, storage_levels=None,
                       max_runtime_seconds=None,
                       max_mixes_per_run=None):
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
    run_start = time.time()
    timed_out = False
    mixes_processed = 0

    def _time_limit_hit():
        return max_runtime_seconds is not None and (time.time() - run_start) >= max_runtime_seconds

    def _checkpoint_and_timeout(phase, cursor):
        nonlocal timed_out
        _save_mix_progress(iso, threshold, candidates, phase, cursor,
                           near_miss_data=near_miss_mixes,
                           mix_min_proc_data=mix_min_proc)
        timed_out = True
        print(f"      Runtime cap reached for {iso} {threshold}% ({max_runtime_seconds:.0f}s); checkpoint saved.")

    def _mix_limit_hit():
        return max_mixes_per_run is not None and mixes_processed >= max_mixes_per_run

    def _checkpoint_and_mix_cap(phase, cursor):
        nonlocal timed_out
        _save_mix_progress(iso, threshold, candidates, phase, cursor,
                           near_miss_data=near_miss_mixes,
                           mix_min_proc_data=mix_min_proc)
        timed_out = True
        print(
            f"      Mix cap reached for {iso} {threshold}% "
            f"({max_mixes_per_run} mixes); checkpoint saved."
        )

    # With demand_arr re-normalized to sum to exactly 1.0 (see
    # prepare_numpy_profiles), a score of 1.0 is now reachable when every
    # hour is fully matched.  No cap needed.
    target = threshold / 100.0
    proc_min, proc_max = PROCUREMENT_BOUNDS.get(threshold, (70, 200))

    # Storage constants
    batt_eff = BATTERY_EFFICIENCY
    batt8_eff = BATTERY8_EFFICIENCY
    ldes_eff = LDES_EFFICIENCY
    ldes_window_hours = LDES_WINDOW_DAYS * 24

    # Storage levels — either from two-phase adaptive sweep (storage_levels param)
    # or default coarse 0.25% for initial pass.
    if storage_levels is not None:
        batt_levels = storage_levels['bat4']
        batt8_levels = storage_levels['bat8']
        ldes_levels = storage_levels['ldes']
    else:
        # Default: coarse 0.25% sweep (Phase 1 of two-phase adaptive approach)
        batt_levels = [0, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 2.5]
        batt8_levels = [0, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
        ldes_levels = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 5, 8, 10]

    # Curtailment frequency threshold: daily-cycle batteries need 150+ surplus days
    MIN_SURPLUS_DAYS_FOR_BATTERY = 150
    # Bat8 dispatch window: 2-day (48hr) for ~200 cycles/year
    batt8_window = 48

    # ── Phase 1: Coarse grid at 5% step ──
    combos_5 = generate_4d_combos(hydro_cap, step=5)
    seeds = get_seed_combos(hydro_cap)
    if len(seeds) > 0:
        combos_5 = np.vstack([combos_5, seeds])
        combos_5 = np.unique(combos_5, axis=0)

    # Cross-threshold pruning: eliminate mixes that were infeasible at previous threshold
    # Vectorized: build mask from set lookups in a single pass
    if prev_pruning is not None:
        feasible_prev = prev_pruning.get('feasible_mixes', set())
        all_prev = prev_pruning.get('all_mixes', set())
        if feasible_prev and all_prev:
            infeasible_prev = all_prev - feasible_prev
            combos_int = combos_5.astype(np.int64)
            keep_mask = np.array([
                (int(r[0]), int(r[1]), int(r[2]), int(r[3])) not in infeasible_prev
                for r in combos_int
            ], dtype=bool)
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

    # Phase 1a: No-storage sweep — FULLY VECTORIZED batch procurement sweep
    # Instead of per-mix binary search (N × log(P) individual numpy ops),
    # evaluates ALL mixes at each procurement level in one vectorized numpy op.
    # ~20 batch calls replaces thousands of individual evaluations.
    phase1a_start = resume_cursor if resume_phase == '1a' else n_combos

    if phase1a_start < n_combos:
        if _time_limit_hit():
            _checkpoint_and_timeout('1a', phase1a_start)

    if not timed_out and phase1a_start < n_combos:
        # Score ALL mixes at max procurement in one shot
        max_pf = proc_levels[-1] / 100.0
        all_max_scores = batch_hourly_scores(demand_arr, supply_matrix, mix_fracs, max_pf)

        # Track all mix keys (batch)
        for i in range(phase1a_start, n_combos):
            all_mix_keys.add((int(combos_5[i][0]), int(combos_5[i][1]),
                              int(combos_5[i][2]), int(combos_5[i][3])))
        mixes_processed += n_combos - phase1a_start

        # Identify feasible and near-miss mixes (vectorized boolean masks)
        feasible_mask = all_max_scores >= target
        near_miss_mask = (~feasible_mask) & (all_max_scores >= target - 0.40)

        # --- Batch procurement sweep for feasible mixes ---
        feasible_indices = np.where(feasible_mask)[0]
        feasible_indices = feasible_indices[feasible_indices >= phase1a_start]

        if len(feasible_indices) > 0:
            n_feas = len(feasible_indices)
            feasible_supply = supply_rows[feasible_indices]

            # Build per-mix procurement floors from previous threshold
            floors = np.full(n_feas, float(proc_min))
            for j in range(n_feas):
                fi = feasible_indices[j]
                mk = (int(combos_5[fi][0]), int(combos_5[fi][1]),
                      int(combos_5[fi][2]), int(combos_5[fi][3]))
                floors[j] = float(max(prev_min_proc.get(mk, proc_min), proc_min))

            # Vectorized procurement sweep: find min feasible level per mix
            min_procs, final_scores = vectorized_procurement_sweep(
                demand_arr, feasible_supply, target, proc_levels, floors)

            # Add all feasible candidates (cheap dict ops)
            for j in range(n_feas):
                fi = feasible_indices[j]
                proc = int(min_procs[j])
                add_candidate(combos_5[fi], proc, 0, 0, 0, final_scores[j])
                mk = (int(combos_5[fi][0]), int(combos_5[fi][1]),
                      int(combos_5[fi][2]), int(combos_5[fi][3]))
                mix_min_proc[mk] = min(mix_min_proc.get(mk, 9999), proc)
                feasible_mix_keys.add(mk)

        # --- Near-miss recording (vectorized) ---
        near_miss_indices = np.where(near_miss_mask)[0]
        near_miss_indices = near_miss_indices[near_miss_indices >= phase1a_start]
        for i in near_miss_indices:
            mk = (int(combos_5[i][0]), int(combos_5[i][1]),
                  int(combos_5[i][2]), int(combos_5[i][3]))
            if mk not in cross_skip:
                near_miss_mixes[int(i)] = max(
                    near_miss_mixes.get(int(i), 0), float(all_max_scores[i]))

    if timed_out:
        pruning_info = {
            'feasible_mixes': feasible_mix_keys,
            'min_proc': mix_min_proc,
            'all_mixes': all_mix_keys,
            'completed': False,
        }
        return candidates, pruning_info

    # Phase 1a→1b transition checkpoint
    if resume_phase == '1a' and phase1a_start < n_combos:
        _save_mix_progress(iso, threshold, candidates, '1b', 0,
                           near_miss_data=near_miss_mixes,
                           mix_min_proc_data=mix_min_proc)

    # Phase 1b: Storage sweep on near-miss mixes — BATCHED + PARALLEL
    # Process mixes in batches of MAX_MIX_BATCH using Numba prange to
    # parallelize across mixes. Each mix evaluates all storage combos in a
    # single _batch_storage_scores call (reusing bat4/bat8 intermediates).
    # This prevents stalling on large ISOs (NYISO, PJM) with many near-misses.
    #
    # Key optimizations:
    # 1. BATCH DISPATCH: _batch_mixes_storage_screen evaluates N mixes × all
    #    storage combos at max procurement in one Numba prange call.
    # 2. CURTAILMENT-MW CAP: Per-mix caps filter which batch results to use.
    # 3. INFEASIBLE SCREENING: Combos infeasible at max procurement skipped.
    # 4. DOMINANCE PRUNING: LDES levels that don't reduce min_proc → stop.
    # 5. CURTAILMENT GUARD: No surplus → skip mix entirely.
    MAX_MIX_BATCH = 100

    near_miss_list = sorted(near_miss_mixes.keys())
    phase1b_start = resume_cursor if resume_phase == '1b' else 0
    if resume_phase not in ('1a', '1b'):
        phase1b_start = len(near_miss_list)  # Skip entirely if past Phase 1b

    # Storage level arrays for batch dispatch (full lists — per-mix caps
    # applied in post-processing to filter which results to use)
    b4_arr = np.array(batt_levels, dtype=np.float64)
    b8_arr = np.array(batt8_levels, dtype=np.float64)
    l_arr = np.array(ldes_levels, dtype=np.float64)
    n_b4, n_b8, n_l = len(b4_arr), len(b8_arr), len(l_arr)

    # Pre-compute curtailment caps for all near-miss mixes using batch Numba kernel
    # (parallel prange across mixes instead of sequential Python loop)
    max_pf = proc_levels[-1] / 100.0

    # Pre-filter: skip cross_skip mixes before expensive storage cap computation
    nm_candidates = []
    for nm_idx in range(phase1b_start, len(near_miss_list)):
        i = near_miss_list[nm_idx]
        mix_key = (int(combos_5[i][0]), int(combos_5[i][1]),
                   int(combos_5[i][2]), int(combos_5[i][3]))
        if mix_key not in cross_skip:
            nm_candidates.append((nm_idx, i, mix_key))

    nm_valid = []
    if nm_candidates:
        # Gather supply rows for batch computation
        nm_supply = np.empty((len(nm_candidates), H), dtype=np.float64)
        for ci, (nm_idx, i, mk) in enumerate(nm_candidates):
            nm_supply[ci] = supply_rows[i]

        # BATCH: compute storage caps for all near-miss mixes in parallel
        b4_caps, b8_caps, l_caps, hc_arr, sd_arr = _batch_compute_storage_caps(
            demand_arr, nm_supply, max_pf, len(nm_candidates),
            BATTERY_DURATION_HOURS, BATTERY8_DURATION_HOURS, LDES_DURATION_HOURS)

        for ci, (nm_idx, i, mix_key) in enumerate(nm_candidates):
            if hc_arr[ci]:
                nm_valid.append((nm_idx, i, mix_key,
                                 b4_caps[ci] * 100.0 * 1.1,
                                 b8_caps[ci] * 100.0 * 1.1,
                                 l_caps[ci] * 100.0 * 1.1,
                                 int(sd_arr[ci])))

    # Process mixes in batches of MAX_MIX_BATCH
    for batch_start in range(0, len(nm_valid), MAX_MIX_BATCH):
        if _time_limit_hit():
            _checkpoint_and_timeout('1b', batch_start + phase1b_start)
            break
        if _mix_limit_hit():
            _checkpoint_and_mix_cap('1b', batch_start + phase1b_start)
            break

        batch_end = min(batch_start + MAX_MIX_BATCH, len(nm_valid))
        batch = nm_valid[batch_start:batch_end]
        n_batch = len(batch)

        # Gather supply rows for this batch
        batch_supply = np.empty((n_batch, H), dtype=np.float64)
        for bi in range(n_batch):
            batch_supply[bi] = supply_rows[batch[bi][1]]

        # BATCH: screen all mixes × all storage combos at max procurement
        # Numba prange distributes mixes across CPU cores
        batch_scores = _batch_mixes_storage_screen(
            demand_arr, batch_supply, max_pf, n_batch,
            b4_arr, b8_arr, l_arr, n_b4, n_b8, n_l,
            batt_eff, batt8_eff, ldes_eff,
            BATTERY_DURATION_HOURS, BATTERY8_DURATION_HOURS, LDES_DURATION_HOURS,
            ldes_window_hours, batt8_window)

        # Process results per mix: apply per-mix caps, skip infeasible, binary-search
        for bi in range(n_batch):
            nm_idx, i, mix_key, bat4_max_pct, bat8_max_pct, ldes_max_pct, n_sd = batch[bi]
            mixes_processed += 1
            mix = combos_5[i]
            supply_row = supply_rows[i]
            scores = batch_scores[bi]

            for b4_idx in range(n_b4):
                bp = b4_arr[b4_idx]
                # Cap: skip battery above curtailment ceiling or insufficient surplus days
                if bp > 0 and (n_sd < MIN_SURPLUS_DAYS_FOR_BATTERY or bp > bat4_max_pct):
                    continue
                batt_cap = bp / 100.0
                batt_pow = batt_cap / BATTERY_DURATION_HOURS if batt_cap > 0 else 0

                for b8_idx in range(n_b8):
                    b8p = b8_arr[b8_idx]
                    if b8p > 0 and (n_sd < MIN_SURPLUS_DAYS_FOR_BATTERY or b8p > bat8_max_pct):
                        continue
                    batt8_cap = b8p / 100.0
                    batt8_pow = batt8_cap / BATTERY8_DURATION_HOURS if batt8_cap > 0 else 0
                    prev_ldes_proc = 9999

                    for l_idx in range(n_l):
                        lp = l_arr[l_idx]
                        if bp == 0 and b8p == 0 and lp == 0:
                            continue
                        if lp > 0 and lp > ldes_max_pct:
                            continue

                        idx = b4_idx * n_b8 * n_l + b8_idx * n_l + l_idx
                        max_score = scores[idx]
                        if max_score < 0 or max_score < target:
                            continue  # Infeasible at max procurement — skip

                        # Binary search for minimum feasible procurement
                        ldes_cap = lp / 100.0
                        ldes_pow = ldes_cap / LDES_DURATION_HOURS if ldes_cap > 0 else 0
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

                        # LDES dominance: if increasing LDES doesn't reduce min_proc, stop
                        if min_proc >= prev_ldes_proc and prev_ldes_proc < 9999:
                            break
                        prev_ldes_proc = min_proc

        # Batch checkpoint
        last_nm_idx = batch[n_batch - 1][0]
        _save_mix_progress(iso, threshold, candidates, '1b', last_nm_idx + 1,
                           near_miss_data=near_miss_mixes,
                           mix_min_proc_data=mix_min_proc)

    if timed_out:
        pruning_info = {
            'feasible_mixes': feasible_mix_keys,
            'min_proc': mix_min_proc,
            'all_mixes': all_mix_keys,
            'completed': False,
        }
        return candidates, pruning_info

    # ── Phase 2: Refine feasible archetypes to 1% resolution — BATCHED ──
    # Collects all fine combos across ALL archetypes, deduplicates, then batch-
    # evaluates in a single pass. Eliminates per-archetype overhead and avoids
    # re-evaluating overlapping neighborhoods.
    if candidates and not timed_out:
        mix_archetypes = set()
        for c in candidates:
            m = tuple(c['resource_mix'][rt] for rt in RESOURCE_TYPES)
            mix_archetypes.add(m)

        # Generate fine combos for all archetypes, concatenate and deduplicate
        all_fine_parts = []
        for mix_tuple in mix_archetypes:
            base = np.array(mix_tuple, dtype=np.float64)
            fine = generate_4d_combos_around(base, hydro_cap, step=1, radius=4)
            if len(fine) > 0:
                all_fine_parts.append(fine)

        if all_fine_parts:
            all_fine = np.unique(np.vstack(all_fine_parts), axis=0)
            all_fine_fracs = all_fine / 100.0
            all_fine_supply = all_fine_fracs @ supply_matrix
            n_all_fine = len(all_fine)

            max_pf = proc_levels[-1] / 100.0
            all_fine_max_scores = batch_hourly_scores(
                demand_arr, supply_matrix, all_fine_fracs, max_pf)

            # --- Batch procurement sweep for feasible fine combos ---
            fine_feas_mask = all_fine_max_scores >= target
            fine_feas_idx = np.where(fine_feas_mask)[0]

            if len(fine_feas_idx) > 0:
                feas_supply = all_fine_supply[fine_feas_idx]
                min_procs_fine, final_scores_fine = vectorized_procurement_sweep(
                    demand_arr, feas_supply, target, proc_levels)

                for j in range(len(fine_feas_idx)):
                    fi = fine_feas_idx[j]
                    add_candidate(all_fine[fi], int(min_procs_fine[j]),
                                  0, 0, 0, final_scores_fine[j])

            # --- Batch storage screen for near-miss fine combos ---
            fine_nm_mask = (~fine_feas_mask) & (all_fine_max_scores >= target - 0.30)
            fine_nm_idx = np.where(fine_nm_mask)[0]

            if len(fine_nm_idx) > 0:
                nm_fine_supply = all_fine_supply[fine_nm_idx]
                n_nm_fine = len(fine_nm_idx)

                # Batch compute storage caps (parallel Numba kernel)
                p2_b4c, p2_b8c, p2_lc, p2_hc, p2_sd = _batch_compute_storage_caps(
                    demand_arr, nm_fine_supply, max_pf, n_nm_fine,
                    BATTERY_DURATION_HOURS, BATTERY8_DURATION_HOURS, LDES_DURATION_HOURS)

                # Filter to mixes with curtailment and build batch list
                nm_fine_valid = []
                for ci in range(n_nm_fine):
                    if p2_hc[ci]:
                        nm_fine_valid.append((
                            ci, fine_nm_idx[ci],
                            p2_b4c[ci] * 100.0 * 1.1,
                            p2_b8c[ci] * 100.0 * 1.1,
                            p2_lc[ci] * 100.0 * 1.1,
                            int(p2_sd[ci]),
                        ))

                # Storage sweep in batches using _batch_mixes_storage_screen
                p2_b4 = np.array(batt_levels, dtype=np.float64)
                p2_b8 = np.array(batt8_levels, dtype=np.float64)
                p2_l = np.array(ldes_levels, dtype=np.float64)
                p2_nb4, p2_nb8, p2_nl = len(p2_b4), len(p2_b8), len(p2_l)

                for batch_start in range(0, len(nm_fine_valid), MAX_MIX_BATCH):
                    batch_end = min(batch_start + MAX_MIX_BATCH, len(nm_fine_valid))
                    batch_p2 = nm_fine_valid[batch_start:batch_end]
                    n_bp2 = len(batch_p2)

                    batch_supply_p2 = np.empty((n_bp2, H), dtype=np.float64)
                    for bi in range(n_bp2):
                        batch_supply_p2[bi] = all_fine_supply[batch_p2[bi][1]]

                    batch_scores_p2 = _batch_mixes_storage_screen(
                        demand_arr, batch_supply_p2, max_pf, n_bp2,
                        p2_b4, p2_b8, p2_l, p2_nb4, p2_nb8, p2_nl,
                        batt_eff, batt8_eff, ldes_eff,
                        BATTERY_DURATION_HOURS, BATTERY8_DURATION_HOURS,
                        LDES_DURATION_HOURS, ldes_window_hours, batt8_window)

                    for bi in range(n_bp2):
                        ci, orig_idx, b4_ceil, b8_ceil, l_ceil, n_sd = batch_p2[bi]
                        scores = batch_scores_p2[bi]
                        supply_row_j = all_fine_supply[orig_idx]

                        for bi4 in range(p2_nb4):
                            bp = p2_b4[bi4]
                            if bp > 0 and (n_sd < MIN_SURPLUS_DAYS_FOR_BATTERY or bp > b4_ceil):
                                continue
                            batt_cap = bp / 100.0
                            batt_pow = batt_cap / BATTERY_DURATION_HOURS if batt_cap > 0 else 0
                            for bi8 in range(p2_nb8):
                                b8p = p2_b8[bi8]
                                if b8p > 0 and (n_sd < MIN_SURPLUS_DAYS_FOR_BATTERY or b8p > b8_ceil):
                                    continue
                                batt8_cap = b8p / 100.0
                                batt8_pow = batt8_cap / BATTERY8_DURATION_HOURS if batt8_cap > 0 else 0
                                ldes_best_proc = 9999
                                for li in range(p2_nl):
                                    lp = p2_l[li]
                                    if bp == 0 and b8p == 0 and lp == 0:
                                        continue
                                    if lp > 0 and lp > l_ceil:
                                        continue
                                    sidx = bi4 * p2_nb8 * p2_nl + bi8 * p2_nl + li
                                    max_sc = scores[sidx]
                                    if max_sc < 0 or max_sc < target:
                                        continue
                                    ldes_cap = lp / 100.0
                                    ldes_pow = ldes_cap / LDES_DURATION_HOURS if ldes_cap > 0 else 0
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
                                    add_candidate(all_fine[orig_idx], min_proc,
                                                  bp, b8p, lp, sc)
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
        'completed': not timed_out,
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
    intra-threshold (Nth-mix) resume.
    """
    iso, demand_data, gen_profiles = args
    iso_start = time.time()

    demand_norm = demand_data[iso]['normalized']
    supply_profiles = get_supply_profiles(iso, gen_profiles)
    demand_arr, supply_matrix = prepare_numpy_profiles(demand_norm, supply_profiles)
    hydro_cap = HYDRO_CAPS[iso]

    print(f"\n  {iso}: Starting optimization ({len(THRESHOLDS)} thresholds, "
          f"hydro_cap={hydro_cap}%)")

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

    if mix_max_score:
        print(f"    {iso}: {len(mix_max_score):,} unique mixes from cache/completed thresholds")

    # ── Cross-threshold solution seeding accumulator ──
    # Collects full solutions from completed thresholds so qualifying ones
    # can be injected into higher thresholds without re-computation.
    all_found_solutions = []

    # ════════════════════════════════════════════════════════════════
    # TWO-PHASE ADAPTIVE STORAGE SWEEP
    # Phase 1: Coarse 0.25% sweep → identify saturation range
    # Phase 2: Fine 0.05% sweep within identified range → merge
    # ════════════════════════════════════════════════════════════════

    def _run_threshold_loop(phase_label, storage_levels_override=None):
        """Run all thresholds with given storage levels. Returns per-threshold results."""
        nonlocal mix_max_score
        phase_results = {}  # threshold -> list of candidates
        prev_prun = None
        found_solutions = list(all_found_solutions)  # copy for cross-threshold seeding

        for threshold in THRESHOLDS:
            t_str = str(threshold)

            # Build cross-feasible set from mix_max_score
            cross_feas = set()
            if mix_max_score:
                cross_feas = {mk for mk, s in mix_max_score.items() if s >= threshold}

            t_start = time.time()
            feasible, prun_info = optimize_threshold(
                iso, threshold, demand_arr, supply_matrix, hydro_cap,
                prev_pruning=prev_prun, cross_feasible_mixes=cross_feas,
                storage_levels=storage_levels_override)
            prev_prun = prun_info

            # Cross-threshold seeding
            if found_solutions:
                existing_keys = set()
                for c in feasible:
                    mk = (c['resource_mix']['clean_firm'], c['resource_mix']['solar'],
                           c['resource_mix']['wind'], c['resource_mix']['hydro'])
                    existing_keys.add((mk, c['procurement_pct'], c['battery_dispatch_pct'],
                                        c.get('battery8_dispatch_pct', 0), c['ldes_dispatch_pct']))
                seeded = 0
                for c in found_solutions:
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

            # Accumulate
            for c in feasible:
                mk = (c['resource_mix']['clean_firm'], c['resource_mix']['solar'],
                       c['resource_mix']['wind'], c['resource_mix']['hydro'])
                s = c['hourly_match_score']
                if s > mix_max_score.get(mk, 0):
                    mix_max_score[mk] = s
            found_solutions.extend(feasible)

            t_elapsed = time.time() - t_start
            archetypes = set()
            for c in feasible:
                archetypes.add(tuple(c['resource_mix'][rt] for rt in RESOURCE_TYPES))
            phase_results[threshold] = feasible
            print(f"    {iso} {threshold}%: {len(feasible)} solutions "
                  f"({len(archetypes)} mix archetypes), {t_elapsed:.1f}s")

            # Incremental save: persist results immediately after each threshold
            # (prevents data loss if process is interrupted during sweep)
            _save_threshold_done(iso, threshold, feasible)

        return phase_results, found_solutions

    def _clear_iso_checkpoints():
        """Clear all checkpoint files for this ISO to start a fresh phase."""
        for threshold in THRESHOLDS:
            for path in [_threshold_done_path(iso, threshold),
                         _mix_progress_path(iso, threshold)]:
                if os.path.exists(path):
                    os.remove(path)

    # ── Phase 1: Coarse sweep ──
    print(f"    {iso}: Phase 1 — Coarse 0.25% storage sweep")
    _clear_iso_checkpoints()

    coarse_results, coarse_solutions = _run_threshold_loop("coarse")

    # Analyze saturation from coarse results
    max_bat4 = 0.0
    max_bat8 = 0.0
    max_ldes = 0.0
    for results_list in coarse_results.values():
        for c in results_list:
            b4 = c.get('battery_dispatch_pct', 0) or 0
            b8 = c.get('battery8_dispatch_pct', 0) or 0
            ld = c.get('ldes_dispatch_pct', 0) or 0
            if b4 > max_bat4:
                max_bat4 = b4
            if b8 > max_bat8:
                max_bat8 = b8
            if ld > max_ldes:
                max_ldes = ld

    print(f"    {iso}: Coarse saturation — bat4={max_bat4:.2f}%, bat8={max_bat8:.2f}%, ldes={max_ldes:.1f}%")

    # ── Phase 2: Fine sweep within saturation range ──
    # Generate 0.05% steps from 0 to (max + 0.25% margin)
    fine_bat4_max = max_bat4 + 0.25
    fine_bat8_max = max_bat8 + 0.25
    fine_bat4 = [0] + [round(x * 0.05, 2) for x in range(1, int(fine_bat4_max / 0.05) + 1)]
    fine_bat8 = [0] + [round(x * 0.05, 2) for x in range(1, int(fine_bat8_max / 0.05) + 1)]
    # LDES: keep coarse (0.5% steps) — less sensitive to granularity
    fine_ldes = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 5, 8, 10]

    fine_levels = {'bat4': fine_bat4, 'bat8': fine_bat8, 'ldes': fine_ldes}
    print(f"    {iso}: Phase 2 — Fine sweep: bat4={len(fine_bat4)} levels (0-{fine_bat4[-1]:.2f}%), "
          f"bat8={len(fine_bat8)} levels (0-{fine_bat8[-1]:.2f}%)")

    # Clear ALL checkpoints for fresh fine sweep
    _clear_iso_checkpoints()

    # Seed with coarse solutions
    all_found_solutions.extend(coarse_solutions)

    fine_results, fine_solutions = _run_threshold_loop("fine", fine_levels)

    total_solutions = sum(len(r) for r in fine_results.values())
    print(f"    {iso}: Total: {total_solutions:,} solutions")

    iso_elapsed = time.time() - iso_start
    print(f"  {iso} completed in {iso_elapsed:.1f}s")
    return iso, iso_results


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINTING
# ══════════════════════════════════════════════════════════════════════════════

# ── Per-ISO/threshold Parquet mix checkpoints ──

def _normalize_threshold_str(threshold):
    """Normalize threshold to consistent string: int-like floats become ints (99.0→'99')."""
    t = float(threshold)
    return str(int(t)) if t == int(t) else str(t)


def _mix_progress_path(iso, threshold):
    """Path for in-progress mix checkpoint parquet."""
    t_str = _normalize_threshold_str(threshold)
    return os.path.join(CHECKPOINT_DIR, f'{iso}_v4_t{t_str}_progress.parquet')


def _threshold_done_path(iso, threshold):
    """Path for completed threshold results parquet: {ISO}_t{XX}_raw_pfs.parquet."""
    t_str = _normalize_threshold_str(threshold)
    return os.path.join(STEP1_RAW_PFS_PARQUET_DIR, f'{iso}_t{t_str}_raw_pfs.parquet')


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
    os.makedirs(STEP1_RAW_PFS_PARQUET_DIR, exist_ok=True)

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


def _rows_to_table(rows):
    """Convert list of row dicts to a PyArrow table."""
    if not rows:
        return None
    return pa.table({
        col: [r[col] for r in rows]
        for col in rows[0].keys()
    })


def _parse_cli_args(argv):
    """Parse CLI args for ISO and threshold filters.

    Supported threshold forms:
      --threshold=95
      --threshold 95
      --threshold 95,97.5
    """
    target_isos = []
    threshold_tokens = []
    i = 0

    while i < len(argv):
        arg = argv[i]

        if arg.startswith('--threshold='):
            threshold_tokens.extend(part.strip() for part in arg.split('=', 1)[1].split(','))
            i += 1
            continue

        if arg == '--threshold':
            if i + 1 >= len(argv):
                raise ValueError("--threshold requires a value")
            threshold_tokens.extend(part.strip() for part in argv[i + 1].split(','))
            i += 2
            continue

        if arg.startswith('-'):
            raise ValueError(f"Unknown argument: {arg}")

        iso = arg.upper()
        if iso in ISOS:
            target_isos.append(iso)
        else:
            raise ValueError(f"Unknown ISO argument: {arg}")
        i += 1

    # Preserve order while deduplicating
    target_isos = list(dict.fromkeys(target_isos))

    target_thresholds = []
    if threshold_tokens:
        for token in threshold_tokens:
            if not token:
                continue
            try:
                value = float(token)
            except ValueError as exc:
                raise ValueError(f"Invalid threshold value: {token}") from exc

            matched = None
            for threshold in THRESHOLDS:
                if abs(float(threshold) - value) < 1e-9:
                    matched = threshold
                    break

            if matched is None:
                raise ValueError(
                    f"Unsupported threshold {token}. Allowed values: {', '.join(str(t) for t in THRESHOLDS)}"
                )

            target_thresholds.append(matched)

    target_thresholds = list(dict.fromkeys(target_thresholds))
    return target_isos, target_thresholds


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global THRESHOLDS
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
        _compute_storage_caps(dummy_demand, dummy_supply, 1.0, 4, 8, 100)
        _batch_compute_storage_caps(dummy_demand, dummy_supply_2d, 1.0, 2, 4, 8, 100)
        dummy_levels = np.array([0.0, 0.1], dtype=np.float64)
        _batch_storage_scores(dummy_demand, dummy_supply, 1.0,
                              dummy_levels, dummy_levels, dummy_levels,
                              2, 2, 2, 0.85, 0.85, 0.50, 4, 8, 100, 168, 48)
        _batch_mixes_storage_screen(dummy_demand, dummy_supply_2d, 1.0, 2,
                                     dummy_levels, dummy_levels, dummy_levels,
                                     2, 2, 2, 0.85, 0.85, 0.50, 4, 8, 100, 168, 48)
        print("  JIT compilation complete")

    target_isos, target_thresholds = _parse_cli_args(sys.argv[1:])

    if target_thresholds:
        THRESHOLDS = target_thresholds
        print(f"  Target thresholds: {THRESHOLDS}")

    if target_isos:
        print(f"  Target ISOs: {target_isos}")

    run_isos = target_isos if target_isos else ISOS

    os.makedirs(STEP1_RAW_PFS_PARQUET_DIR, exist_ok=True)

    # Run ISOs sequentially and write per-ISO/threshold parquet outputs only
    total_solutions = 0
    for iso in run_isos:
        args = (iso, demand_data, gen_profiles)
        iso_name, iso_results = process_iso(args)

        iso_total = 0
        for threshold in THRESHOLDS:
            path = _threshold_done_path(iso_name, threshold)
            if not os.path.exists(path):
                continue
            try:
                iso_total += pq.read_table(path).num_rows
            except Exception:
                pass

        total_solutions += iso_total
        print(f"  Raw parquet outputs ready for {iso_name}: {iso_total:,} solutions")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"  Complete in {elapsed:.1f}s")
    print(f"  Total candidates: {total_solutions:,}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
