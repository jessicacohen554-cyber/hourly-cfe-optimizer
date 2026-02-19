#!/usr/bin/env python3
"""
v4.0 Physics Optimizer — Fresh Rebuild
=======================================
Generates the feasible solution space for hourly CFE matching.
Physics only — no cost model. Cost sensitivities applied in Step 2.

Key changes from v3.x:
  - 4D resource space: clean_firm (absorbs CCS), solar, wind, hydro
  - 13 thresholds: 50, 60, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 100
  - Adaptive grid search: 5% → 1% refinement
  - Pareto frontier: 3-5 points per threshold×ISO (procurement/storage tradeoff)
  - Numba JIT-compiled scoring functions
  - Parallel ISO execution (multiprocessing)
  - Vectorized batch mix evaluation
  - Output: JSON + Parquet

Resource types (4D optimization):
  - Clean Firm: nuclear (seasonal-derated) + CCS-CCGT (flat baseload)
    Sub-allocation determined by cost model in Step 2.
    Physics uses nuclear-derated profile (conservative — CCS only improves matching).
  - Solar: EIA 2021-2025 averaged hourly profile (DST-aware)
  - Wind: EIA 2021-2025 averaged hourly profile
  - Hydro: EIA 2021-2025 averaged hourly profile (capped by region, existing only)

Storage (not part of mix %, swept as separate dimensions):
  - Battery: 4hr Li-ion, 85% RTE, daily-cycle dispatch
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

# Threshold-adaptive procurement bounds (Decision 3C)
PROCUREMENT_BOUNDS = {
    50:   (50, 105),
    60:   (60, 105),
    70:   (70, 110),
    75:   (75, 110),
    80:   (80, 115),
    85:   (85, 120),
    87.5: (87, 130),
    90:   (90, 140),
    92.5: (92, 150),
    95:   (95, 170),
    97.5: (100, 200),
    99:   (100, 200),
    100:  (100, 200),
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

def generate_4d_combos(hydro_cap, step=5, max_single=80):
    """Generate all 4D resource mixes summing to 100% with constraints.

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


def generate_4d_combos_around(base_combo, hydro_cap, step=1, radius=2, max_single=80):
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
EDGE_CASE_SEEDS = [
    [5, 70, 20, 5],    # Solar-dominant
    [5, 75, 15, 5],
    [5, 20, 70, 5],    # Wind-dominant
    [5, 15, 75, 5],
    [5, 40, 40, 15],   # Balanced renewable + hydro
    [10, 45, 45, 0],   # Balanced renewable
    [60, 15, 15, 10],  # Clean firm dominant
    [70, 10, 10, 10],
    [80, 10, 10, 0],
    [50, 25, 25, 0],   # Moderate firm + renewables
    [40, 30, 30, 0],
    [30, 35, 35, 0],
    [20, 40, 40, 0],   # Minimal firm
    [10, 45, 45, 0],
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

def optimize_threshold(iso, threshold, demand_arr, supply_matrix, hydro_cap):
    """Find feasible solutions for a single threshold × ISO.

    Returns list of candidate dicts, each with:
      resource_mix, procurement_pct, battery_dispatch_pct, ldes_dispatch_pct,
      hourly_match_score, pareto_type
    """
    target = threshold / 100.0
    proc_min, proc_max = PROCUREMENT_BOUNDS.get(threshold, (70, 200))

    # Storage constants
    batt_eff = BATTERY_EFFICIENCY
    ldes_eff = LDES_EFFICIENCY
    ldes_window_hours = LDES_WINDOW_DAYS * 24

    # Adaptive storage levels based on threshold difficulty
    if threshold <= 70:
        batt_levels = [2, 5, 10]
        ldes_levels = [2, 5]
        storage_proc_step = 3  # Only sweep every 3rd procurement for storage
    elif threshold <= 90:
        batt_levels = [2, 5, 10, 15]
        ldes_levels = [2, 5, 10]
        storage_proc_step = 2
    else:
        batt_levels = [2, 5, 8, 10, 15, 20]
        ldes_levels = [2, 5, 8, 10, 15, 20]
        storage_proc_step = 1

    # ── Phase 1: Coarse grid at 5% step ──
    combos_5 = generate_4d_combos(hydro_cap, step=5)
    seeds = get_seed_combos(hydro_cap)
    if len(seeds) > 0:
        combos_5 = np.vstack([combos_5, seeds])
        combos_5 = np.unique(combos_5, axis=0)

    n_combos = len(combos_5)
    mix_fracs = combos_5 / 100.0

    # Pre-compute supply_row for each mix: (N, 8760)
    supply_rows = mix_fracs @ supply_matrix

    # Procurement levels
    proc_step = 5 if (proc_max - proc_min) > 30 else 2
    proc_levels = list(range(proc_min, proc_max + 1, proc_step))
    if proc_max not in proc_levels:
        proc_levels.append(proc_max)

    candidates = []
    passed_indices = set()    # Mixes that achieved target WITHOUT storage
    near_indices = set()      # Mixes that are close but need storage

    # Phase 1a: Batch hourly scores (no storage) — fast vectorized scan
    for proc in proc_levels:
        pf = proc / 100.0
        scores = batch_hourly_scores(demand_arr, supply_matrix, mix_fracs, pf)
        for i in range(n_combos):
            if scores[i] >= target:
                mix = combos_5[i]
                candidates.append({
                    'resource_mix': {rt: int(mix[j]) for j, rt in enumerate(RESOURCE_TYPES)},
                    'procurement_pct': proc,
                    'battery_dispatch_pct': 0,
                    'ldes_dispatch_pct': 0,
                    'hourly_match_score': round(scores[i] * 100, 2),
                })
                passed_indices.add(i)
            elif scores[i] >= target - 0.15:
                near_indices.add(i)

    # Phase 1b: Storage sweep — ONLY on near-miss mixes (not ones that already passed)
    needs_storage = near_indices - passed_indices
    for i in needs_storage:
        supply_row = supply_rows[i]
        mix = combos_5[i]

        # Only sweep a subset of procurement levels for storage
        storage_procs = proc_levels[::storage_proc_step]
        if proc_levels[-1] not in storage_procs:
            storage_procs.append(proc_levels[-1])

        for proc in storage_procs:
            pf = proc / 100.0

            # Battery only — find minimum battery that works
            for bp in batt_levels:
                batt_cap = bp / 100.0
                batt_pow = batt_cap / BATTERY_DURATION_HOURS
                score = _score_with_battery(demand_arr, supply_row, pf,
                                            batt_cap, batt_pow, batt_eff)
                if score >= target:
                    candidates.append({
                        'resource_mix': {rt: int(mix[j]) for j, rt in enumerate(RESOURCE_TYPES)},
                        'procurement_pct': proc,
                        'battery_dispatch_pct': bp,
                        'ldes_dispatch_pct': 0,
                        'hourly_match_score': round(score * 100, 2),
                    })
                    break

            # Battery + LDES combo — try one efficient combo
            if threshold >= 90:
                for bp in [5, 10]:
                    batt_cap = bp / 100.0
                    batt_pow = batt_cap / BATTERY_DURATION_HOURS
                    for lp in [5, 10]:
                        ldes_cap = lp / 100.0
                        ldes_pow = ldes_cap / LDES_DURATION_HOURS
                        score = _score_with_both_storage(
                            demand_arr, supply_row, pf,
                            batt_cap, batt_pow, batt_eff,
                            ldes_cap, ldes_pow, ldes_eff, ldes_window_hours)
                        if score >= target:
                            candidates.append({
                                'resource_mix': {rt: int(mix[j]) for j, rt in enumerate(RESOURCE_TYPES)},
                                'procurement_pct': proc,
                                'battery_dispatch_pct': bp,
                                'ldes_dispatch_pct': lp,
                                'hourly_match_score': round(score * 100, 2),
                            })
                            break
                    else:
                        continue
                    break

    # ── Phase 2: Refine top candidates to 1% resolution ──
    if candidates:
        # Group by mix archetype (rounded to 5%), find best
        mix_groups = {}
        for c in candidates:
            m = tuple(c['resource_mix'][rt] for rt in RESOURCE_TYPES)
            total = c['procurement_pct'] + c['battery_dispatch_pct'] + c['ldes_dispatch_pct']
            if m not in mix_groups or total < mix_groups[m]['total']:
                mix_groups[m] = {'candidate': c, 'total': total}

        # Refine top 15 archetypes
        sorted_mixes = sorted(mix_groups.items(), key=lambda x: x[1]['total'])[:15]

        for mix_tuple, info in sorted_mixes:
            base = np.array(mix_tuple, dtype=np.float64)
            fine_combos = generate_4d_combos_around(base, hydro_cap, step=1, radius=3)
            if len(fine_combos) == 0:
                continue

            fine_fracs = fine_combos / 100.0
            best_proc = info['candidate']['procurement_pct']

            # Sweep narrow procurement window around best
            for proc_d in range(-3, 4):
                proc = best_proc + proc_d
                if proc < proc_min or proc > proc_max:
                    continue
                pf = proc / 100.0
                scores = batch_hourly_scores(demand_arr, supply_matrix, fine_fracs, pf)
                for j in range(len(fine_combos)):
                    if scores[j] >= target:
                        mix = fine_combos[j]
                        candidates.append({
                            'resource_mix': {rt: int(mix[k]) for k, rt in enumerate(RESOURCE_TYPES)},
                            'procurement_pct': proc,
                            'battery_dispatch_pct': 0,
                            'ldes_dispatch_pct': 0,
                            'hourly_match_score': round(scores[j] * 100, 2),
                        })

    # ── Extract Pareto frontier (3-5 points) ──
    pareto = extract_pareto(candidates, target)

    return pareto


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
        key = (m, c['procurement_pct'], c['battery_dispatch_pct'], c['ldes_dispatch_pct'])
        if key not in seen:
            seen.add(key)
            unique.append(c)

    if not unique:
        return []

    # Metrics for Pareto
    for c in unique:
        c['total_storage'] = c['battery_dispatch_pct'] + c['ldes_dispatch_pct']
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
               candidate['ldes_dispatch_pct'])
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

    for threshold in THRESHOLDS:
        t_str = str(threshold)
        if t_str in completed_thresholds:
            print(f"    {iso} {threshold}%: loaded from checkpoint — skipping")
            continue

        t_start = time.time()
        pareto = optimize_threshold(iso, threshold, demand_arr, supply_matrix, hydro_cap)
        t_elapsed = time.time() - t_start

        iso_results['thresholds'][t_str] = {
            'candidates': pareto,
            'candidate_count': len(pareto),
            'elapsed_seconds': round(t_elapsed, 2),
        }
        print(f"    {iso} {threshold}%: {len(pareto)} Pareto points, {t_elapsed:.1f}s")

        # Checkpoint after each threshold
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

def save_results_json(all_results, output_path, runtime_seconds):
    """Save results as JSON."""
    try:
        import subprocess
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = 'unknown'

    output = {
        'metadata': {
            'version': '4.0-physics',
            'created_at': datetime.now(timezone.utc).isoformat(),
            'git_commit': git_hash,
            'runtime_seconds': round(runtime_seconds, 1),
            'description': 'v4.0 physics feasible solution space: '
                           '13 thresholds x 5 ISOs, Pareto frontier per threshold',
            'thresholds': THRESHOLDS,
            'isos': ISOS,
            'resource_types': RESOURCE_TYPES,
            'numba_enabled': HAS_NUMBA,
        },
        'config': {
            'data_year': DATA_YEAR,
            'profile_years': PROFILE_YEARS,
            'battery': {
                'duration_hours': BATTERY_DURATION_HOURS,
                'efficiency': BATTERY_EFFICIENCY,
            },
            'ldes': {
                'duration_hours': LDES_DURATION_HOURS,
                'efficiency': LDES_EFFICIENCY,
                'window_days': LDES_WINDOW_DAYS,
            },
            'hydro_caps': HYDRO_CAPS,
            'procurement_bounds': {str(k): list(v) for k, v in PROCUREMENT_BOUNDS.items()},
        },
        'results': all_results,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"  JSON saved: {output_path} ({size_kb:.0f} KB)")


def save_results_parquet(all_results, output_path):
    """Save results as Parquet for analytics."""
    if not HAS_PARQUET:
        print("  Parquet skipped — pyarrow not installed")
        return

    rows = []
    for iso, iso_data in all_results.items():
        for t_str, t_data in iso_data.get('thresholds', {}).items():
            for candidate in t_data.get('candidates', []):
                row = {
                    'iso': iso,
                    'threshold': float(t_str),
                    'clean_firm': candidate['resource_mix']['clean_firm'],
                    'solar': candidate['resource_mix']['solar'],
                    'wind': candidate['resource_mix']['wind'],
                    'hydro': candidate['resource_mix']['hydro'],
                    'procurement_pct': candidate['procurement_pct'],
                    'battery_dispatch_pct': candidate['battery_dispatch_pct'],
                    'ldes_dispatch_pct': candidate['ldes_dispatch_pct'],
                    'hourly_match_score': candidate['hourly_match_score'],
                    'pareto_type': candidate.get('pareto_type', ''),
                }
                rows.append(row)

    if not rows:
        print("  Parquet skipped — no results")
        return

    table = pa.table({
        col: [r[col] for r in rows]
        for col in rows[0].keys()
    })
    pq.write_table(table, output_path)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"  Parquet saved: {output_path} ({size_kb:.0f} KB)")


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

    # Save results
    elapsed = time.time() - start_time
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard')
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, 'physics_results_v4.json')
    save_results_json(all_results, json_path, elapsed)

    parquet_path = os.path.join(output_dir, 'physics_results_v4.parquet')
    save_results_parquet(all_results, parquet_path)

    # Also save to data/ as cache
    cache_path = os.path.join(DATA_DIR, 'physics_cache_v4.json')
    save_results_json(all_results, cache_path, elapsed)

    print(f"\n{'=' * 70}")
    print(f"  Complete in {elapsed:.1f}s")
    print(f"  Total candidates: {sum(len(t.get('candidates', [])) for iso in all_results.values() for t in iso.get('thresholds', {}).values())}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
