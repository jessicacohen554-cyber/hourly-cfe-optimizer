#!/usr/bin/env python3
"""
Step 1: Physics Feasible Space (PFS) Generator
===============================================
Generates the Physical Feasibility Space (PFS) for hourly CFE matching.
Physics only — no cost model. Cost sensitivities applied in Step 3.

v5.0 Architecture: Direct resource fractions as % of demand.
  - No procurement multiplier — each resource independently varies from 0% to its cap.
  - One-time coarse sweep per ISO (cached), reused across all thresholds.
  - Per-threshold: filter cache → storage sweep → fine 1% refinement.

Pipeline position: Step 1 of 7
  Step 1 — PFS Generator (this file)
  Step 2 — Efficient Frontier (EF) extraction
  Step 3 — Cost optimization
  Steps 4-7 — Post-processing, dispatch, dashboard

Key features:
  - 4D non-CAISO / 5D CAISO (with geothermal): independent resource fractions
  - 17 thresholds: 50..100 with finer resolution at inflection zone
  - Coarse 5% → Fine 1% adaptive grid (±4% radius around archetypes)
  - Numba JIT-compiled scoring functions
  - Inter-ISO parallel execution via multiprocessing.Pool (--workers N)
  - Vectorized batch mix evaluation

Output:
  data/step1-pfs-parquets/{ISO}_coarse_cache.parquet (one-time coarse sweep)
  data/step1-pfs-parquets/{ISO}_t{XX}_raw_pfs.parquet (per ISO/threshold)

Resource types:
  - Clean Firm (0-120%): nuclear (seasonal-derated) + CCS-CCGT
  - Solar (0-250%): EIA 2021-2025 averaged hourly profile (DST-aware)
  - Wind (0-250%): EIA 2021-2025 averaged hourly profile
  - Hydro (0-cap+10%): EIA 2021-2025 averaged hourly (capped by region)
  - Geothermal (CAISO only, 0-20%): Flat year-round, no seasonal derate

Storage (swept as separate dimensions after resource fractions):
  - Battery (4hr): Li-ion, 85% RTE, daily-cycle dispatch
  - Battery (8hr): Li-ion, 85% RTE, daily-cycle dispatch (power = cap/8hr)
  - LDES: 100hr iron-air, 50% RTE, 7-day rolling window dispatch
  - Green H2: 1000hr salt cavern, 35% RTE, 30-day rolling window (≥95% only)
"""

import json
import os
import sys
import time
import itertools
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
H2_EFFICIENCY = 0.35         # Green H2: electrolysis (70%) × storage (95%) × turbine (55%)
H2_DURATION_HOURS = 1000     # ~42 days at full power (salt cavern)
H2_WINDOW_DAYS = 30          # 30-day rolling window for multi-week weather bridging

# Resource types — 4D for non-CAISO, 5D for CAISO (geothermal)
RESOURCE_TYPES_BASE = ['clean_firm', 'solar', 'wind', 'hydro']
RESOURCE_TYPES_CAISO = ['clean_firm', 'solar', 'wind', 'hydro', 'geothermal']

def get_resource_types(iso):
    """Return resource type list for this ISO."""
    return RESOURCE_TYPES_CAISO if iso == 'CAISO' else RESOURCE_TYPES_BASE

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

# Hydro caps (% of demand, existing only)
HYDRO_CAPS = {
    'CAISO': 9.5, 'ERCOT': 0.1, 'PJM': 1.8, 'NYISO': 15.9, 'NEISO': 4.4,
    'MISO': 1.6, 'SPP': 4.3,
}
# Physics-only hydro adder: allow +10% above existing cap for run-of-river
# innovation potential. NOT priced in Step 3 — just explored in physics cache.
HYDRO_ADDER_PCT = 10

# CAISO geothermal cap: (existing_geo_TWh + potential) / demand
# = (5.31 + 39.0) / 224.039 = 19.8% → cap at 20%
GEO_CAP_PCT = 20

# Resource caps: max % of demand for each resource in coarse grid search
# Each resource varies independently — no sum-to-100 constraint.
# Caps informed by step 3 cost-optimal analysis across all ISOs/scenarios:
#   - CF: 120% hit by MISO/PJM at 99.99% threshold (keep)
#   - Solar: max observed 81% (MISO 99.9%); cap at 100% (+23% buffer)
#   - Wind: max observed 242% (MISO 99.9%); keep at 250% (+3% buffer)
# Additionally, per-ISO caps applied in generate_resource_combos() for
# ISOs where observed maxima are much lower.
RESOURCE_CAPS = {
    'clean_firm': 120,   # Nuclear/CCS flat profile; >100% = surplus for storage
    'solar':      100,   # Observed max 81% (MISO); buffer to 100%
    'wind':       250,   # Observed max 242% (MISO); wind-heavy ISOs need this
    # hydro and geothermal caps are per-ISO (HYDRO_CAPS + HYDRO_ADDER_PCT, GEO_CAP_PCT)
}

# Total procurement cap: max sum of all resources + storage as % of demand.
# Observed max across all step 3 winners: 281% (MISO at 99.9%).
# Cap at 350% to provide generous buffer while filtering combinatorial blowup.
TOTAL_PROCUREMENT_CAP = 350

# 21 thresholds (10-40 added for Track 2/3 greenfield analysis)
THRESHOLDS = [10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 99.5, 99.9, 99.99]

# Nuclear seasonal derate — CAISO now pure nuclear (geothermal is separate 5th dim)
NUCLEAR_SHARE_OF_CLEAN_FIRM = {
    'CAISO': 1.0, 'ERCOT': 1.0, 'PJM': 1.0, 'NYISO': 1.0, 'NEISO': 1.0,
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

# Chunked saving — flush candidates to disk every N batches to survive OOM/crashes.
# Each chunk is saved as {ISO}_t{XX}_chunk{N}.parquet and merged at threshold end.
CHUNK_SAVE_INTERVAL = 5000   # batches between disk flushes
CHUNK_CANDIDATE_LIMIT = 500_000  # also flush if candidate list exceeds this count

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

    from eia_data_io import load_generation_profiles, load_demand_profiles, load_fossil_mix
    demand_raw = load_demand_profiles()
    gen_raw = load_generation_profiles()
    with open(os.path.join(DATA_DIR, 'egrid_emission_rates.json')) as f:
        emission_rates = json.load(f)
    fossil_mix = load_fossil_mix()

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
    """Get generation profiles for resource types (4D non-CAISO, 5D CAISO).

    Returns dict mapping resource type to numpy array of H floats.
    Each profile represents the hourly generation shape per unit of annual demand
    (i.e., if a resource is at 100% of demand, profile[h] * demand_TWh gives MWh
    in hour h). Profiles are normalized so sum(profile) = 1.0 / H for flat resources.

    CAISO includes geothermal (flat, no seasonal derate) as 5th resource.
    """
    profiles = {}

    # Clean firm = nuclear seasonal-derated baseload (+ CCS in shoulder months)
    # CAISO: pure nuclear (geothermal split to separate dimension)
    # Non-CAISO: pure nuclear (no geothermal)
    monthly_cf = NUCLEAR_MONTHLY_CF.get(iso, {m: 1.0 for m in range(1, 13)})
    month_hours = np.array([744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744])
    month_cfs = np.array([monthly_cf.get(m + 1, 1.0) for m in range(12)])
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

    # CAISO: add geothermal as flat year-round profile (no seasonal derate)
    if iso == 'CAISO':
        profiles['geothermal'] = np.full(H, 1.0 / H, dtype=np.float64)

    # Ensure all profiles are exactly H hours, no negatives (kept as numpy arrays)
    for rtype in get_resource_types(iso):
        arr = np.asarray(profiles[rtype], dtype=np.float64)
        if len(arr) > H:
            arr = arr[:H].copy()
        elif len(arr) < H:
            arr = np.pad(arr, (0, H - len(arr)))
        np.maximum(arr, 0.0, out=arr)
        profiles[rtype] = arr

    return profiles


def prepare_numpy_profiles(iso, demand_norm, supply_profiles):
    """Convert to numpy arrays + build supply matrix (N_resources, 8760).

    Returns (demand_arr, supply_matrix) where supply_matrix shape is
    (4, 8760) for non-CAISO or (5, 8760) for CAISO (includes geothermal).

    Re-normalizes demand_arr to sum to exactly 1.0 to fix a scaling issue
    where leap-year profiles (2024) were normalized over 8784 hours then
    truncated to 8760, causing the average to sum to ~0.9995 instead of 1.0.
    """
    demand_arr = np.array(demand_norm[:H], dtype=np.float64)
    dsum = demand_arr.sum()
    if dsum > 0 and abs(dsum - 1.0) > 1e-9:
        demand_arr *= (1.0 / dsum)
    rtypes = get_resource_types(iso)
    supply_matrix = np.stack([
        np.asarray(supply_profiles[rt][:H], dtype=np.float64)
        for rt in rtypes
    ])  # shape (n_resources, 8760)
    return demand_arr, supply_matrix


# ══════════════════════════════════════════════════════════════════════════════
# SCORING FUNCTIONS — Numba JIT compiled
# ══════════════════════════════════════════════════════════════════════════════

@njit(cache=True, fastmath=True)
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


@njit(cache=True, parallel=True, fastmath=True)
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


@njit(cache=True, fastmath=True)
def _score_with_all_storage(demand, supply_row, procurement,
                            batt_capacity, batt_power, batt_eff,
                            batt8_capacity, batt8_power, batt8_eff,
                            ldes_capacity, ldes_power, ldes_eff,
                            ldes_window_hours,
                            h2_capacity=0.0, h2_power=0.0, h2_eff=0.0,
                            h2_window_hours=720):
    """Hourly score + battery4 (daily) + battery8 (daily) + LDES (multi-day) + H2 (seasonal).

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

    # Phase 4: Green H2 seasonal storage on post-LDES residual
    h2_dispatched = 0.0
    if h2_capacity > 0:
        h2_soc = 0.0
        n_h2_windows = (8760 + h2_window_hours - 1) // h2_window_hours
        for w in range(n_h2_windows):
            ws = w * h2_window_hours
            we = ws + h2_window_hours
            if we > 8760:
                we = 8760
            for h in range(ws, we):
                s = residual_surplus[h]
                if s > 0 and h2_soc < h2_capacity:
                    charge = s
                    if charge > h2_power:
                        charge = h2_power
                    remaining = h2_capacity - h2_soc
                    if charge > remaining:
                        charge = remaining
                    h2_soc += charge
                    residual_surplus[h] -= charge
            for h in range(ws, we):
                g = residual_gap[h]
                if g > 0 and h2_soc > 0:
                    available_e = h2_soc * h2_eff
                    discharge = g
                    if discharge > h2_power:
                        discharge = h2_power
                    if discharge > available_e:
                        discharge = available_e
                    h2_dispatched += discharge
                    h2_soc -= discharge / h2_eff
                    residual_gap[h] -= discharge

    return base_matched + batt_dispatched + batt8_dispatched + ldes_dispatched + h2_dispatched


@njit(cache=True, fastmath=True)
def _batch_storage_scores(demand, supply_row, procurement,
                          bat4_levels, bat8_levels, ldes_levels_arr,
                          n_b4, n_b8, n_ldes,
                          batt_eff, batt8_eff, ldes_eff,
                          batt4_dur, batt8_dur, ldes_dur,
                          ldes_window_hours, batt8_window,
                          h2_levels_arr=np.zeros(1), n_h2=1,
                          h2_eff=0.35, h2_dur=1000.0, h2_window_hours=720):
    """Evaluate ALL storage combos for a single (mix, procurement) in one call.

    Instead of calling _score_with_all_storage N times, this function:
    1. Computes base supply/surplus/gap once
    2. For each bat4 level, dispatches bat4 → gets residual (reused for all bat8)
    3. For each bat8 level, dispatches bat8 → gets residual (reused for all LDES)
    4. For each LDES level, dispatches LDES → gets residual (reused for all H2)
    5. For each H2 level, dispatches H2 → computes score

    Pre-allocates scratch arrays once and copies into them to avoid ~2,400
    heap allocations per mix from .copy() calls in the inner loops.

    Returns flat array of scores with shape (n_b4 * n_b8 * n_ldes * n_h2,).
    Index mapping: scores[b4_idx * n_b8 * n_ldes * n_h2 + b8_idx * n_ldes * n_h2 + l_idx * n_h2 + h2_idx]
    """
    scores = np.full(n_b4 * n_b8 * n_ldes * n_h2, -1.0)  # -1 = not evaluated

    # Base supply and matching
    base_surplus = np.empty(8760)
    base_gap = np.empty(8760)
    base_matched = 0.0
    for h in range(8760):
        s = procurement * supply_row[h]
        d = demand[h]
        base_matched += min(d, s)
        if s > d:
            base_surplus[h] = s - d
            base_gap[h] = 0.0
        else:
            base_surplus[h] = 0.0
            base_gap[h] = d - s

    # Pre-allocate scratch arrays once — reuse via copy-into instead of .copy()
    res_surplus4 = np.empty(8760)
    res_gap4 = np.empty(8760)
    res_surplus8 = np.empty(8760)
    res_gap8 = np.empty(8760)
    res_surplus_l = np.empty(8760)
    res_gap_l = np.empty(8760)

    for b4_idx in range(n_b4):
        bp = bat4_levels[b4_idx]
        batt_cap = bp / 100.0
        batt_pow = batt_cap / batt4_dur if batt_cap > 0 else 0.0

        # Copy base into bat4 scratch
        for h in range(8760):
            res_surplus4[h] = base_surplus[h]
            res_gap4[h] = base_gap[h]
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

            # Copy bat4 residual into bat8 scratch
            for h in range(8760):
                res_surplus8[h] = res_surplus4[h]
                res_gap8[h] = res_gap4[h]
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
                if bp == 0 and b8p == 0 and lp == 0 and (n_h2 <= 1 or h2_levels_arr[0] == 0):
                    continue
                ldes_cap = lp / 100.0
                ldes_pow = ldes_cap / ldes_dur if ldes_cap > 0 else 0.0

                # Copy bat8 residual into LDES scratch
                for h in range(8760):
                    res_surplus_l[h] = res_surplus8[h]
                    res_gap_l[h] = res_gap8[h]
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
                            s = res_surplus_l[h]
                            if s > 0 and soc < ldes_cap:
                                charge = min(s, ldes_pow, ldes_cap - soc)
                                soc += charge
                                res_surplus_l[h] -= charge
                        for h in range(ws, we):
                            g = res_gap_l[h]
                            if g > 0 and soc > 0:
                                ae = soc * ldes_eff
                                discharge = min(g, ldes_pow, ae)
                                ldes_dispatched += discharge
                                soc -= discharge / ldes_eff
                                res_gap_l[h] -= discharge

                # Phase 4: H2 seasonal storage on post-LDES residual
                for h2_idx in range(n_h2):
                    h2p = h2_levels_arr[h2_idx]
                    if bp == 0 and b8p == 0 and lp == 0 and h2p == 0:
                        continue
                    h2_cap = h2p / 100.0
                    h2_pow = h2_cap / h2_dur if h2_cap > 0 else 0.0

                    h2_dispatched = 0.0
                    if h2_cap > 0:
                        h2_soc = 0.0
                        n_h2_win = (8760 + h2_window_hours - 1) // h2_window_hours
                        for w in range(n_h2_win):
                            ws = w * h2_window_hours
                            we = ws + h2_window_hours
                            if we > 8760:
                                we = 8760
                            for h in range(ws, we):
                                s = res_surplus_l[h]
                                if s > 0 and h2_soc < h2_cap:
                                    charge = min(s, h2_pow, h2_cap - h2_soc)
                                    h2_soc += charge
                            for h in range(ws, we):
                                g = res_gap_l[h]
                                if g > 0 and h2_soc > 0:
                                    ae = h2_soc * h2_eff
                                    discharge = min(g, h2_pow, ae)
                                    h2_dispatched += discharge
                                    h2_soc -= discharge / h2_eff

                    idx = b4_idx * n_b8 * n_ldes * n_h2 + b8_idx * n_ldes * n_h2 + l_idx * n_h2 + h2_idx
                    scores[idx] = base_matched + batt_dispatched + batt8_dispatched + ldes_dispatched + h2_dispatched

    return scores


@njit(cache=True, parallel=True, fastmath=True)
def _batch_mixes_storage_screen(demand, supply_rows_batch, procurement, n_mixes,
                                 bat4_levels, bat8_levels, ldes_levels,
                                 n_b4, n_b8, n_ldes,
                                 batt_eff, batt8_eff, ldes_eff,
                                 batt4_dur, batt8_dur, ldes_dur,
                                 ldes_window_hours, batt8_window,
                                 h2_levels=np.zeros(1), n_h2=1,
                                 h2_eff=0.35, h2_dur=1000.0, h2_window_hours=720):
    """Screen N mixes × all storage combos at max procurement, in parallel.

    Uses Numba prange to distribute mixes across CPU cores. Each mix evaluates
    all bat4×bat8×LDES×H2 combos, reusing dispatch intermediates via
    _batch_storage_scores.

    Args:
        supply_rows_batch: (n_mixes, 8760) array of pre-mixed supply profiles
        Other args: same as _batch_storage_scores

    Returns: (n_mixes, n_combos) array of scores at max procurement. -1 = skipped.
    """
    n_combos = n_b4 * n_b8 * n_ldes * n_h2
    all_scores = np.full((n_mixes, n_combos), -1.0)

    for i in prange(n_mixes):
        mix_scores = _batch_storage_scores(
            demand, supply_rows_batch[i], procurement,
            bat4_levels, bat8_levels, ldes_levels,
            n_b4, n_b8, n_ldes,
            batt_eff, batt8_eff, ldes_eff,
            batt4_dur, batt8_dur, ldes_dur,
            ldes_window_hours, batt8_window,
            h2_levels, n_h2, h2_eff, h2_dur, h2_window_hours)
        for j in range(n_combos):
            all_scores[i, j] = mix_scores[j]

    return all_scores


# ══════════════════════════════════════════════════════════════════════════════
# BATCH EVALUATION — vectorized for grid search + Numba parallel for storage
# ══════════════════════════════════════════════════════════════════════════════

def batch_hourly_scores(demand_arr, supply_matrix, mix_batch, chunk_size=10000):
    """Evaluate N mixes at once. mix_batch shape (N, n_resources), returns (N,) scores.

    Resource fractions are direct % of demand (no procurement multiplier).
    Uses total matched energy metric: sum(min(demand, supply)).

    Processes in chunks to avoid OOM on large combo sets (e.g. CAISO 5D
    produces 1.6M combos × 8760 hours = 106 GiB at float64 if done at once).
    chunk_size=10000 keeps the intermediate array under ~0.7 GiB (safe for
    GitHub Actions runners with 7 GB RAM).
    """
    N = len(mix_batch)
    if N <= chunk_size:
        supply_batch = (mix_batch / 100.0) @ supply_matrix
        matched = np.minimum(demand_arr, supply_batch)
        return matched.sum(axis=1)

    scores = np.empty(N, dtype=np.float64)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        supply_chunk = (mix_batch[start:end] / 100.0) @ supply_matrix
        np.minimum(demand_arr, supply_chunk, out=supply_chunk)
        scores[start:end] = supply_chunk.sum(axis=1)
    return scores


@njit(cache=True, fastmath=True)
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


@njit(cache=True, fastmath=True)
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

def generate_resource_combos(iso, step=5):
    """Generate all resource fraction combos as independent % of demand.

    Each resource varies independently (no sum-to-100 constraint).
    Resource caps defined in RESOURCE_CAPS + per-ISO hydro/geo caps.

    Returns numpy array of shape (N, n_resources) where n_resources is
    4 (non-CAISO) or 5 (CAISO with geothermal). Values are percentages
    of annual demand (e.g., solar=150 means 150% of demand from solar).

    Uses itertools.product for clean Cartesian product generation.
    """
    hydro_cap = HYDRO_CAPS[iso] + HYDRO_ADDER_PCT
    cf_max = RESOURCE_CAPS['clean_firm']
    sol_max = RESOURCE_CAPS['solar']
    wnd_max = RESOURCE_CAPS['wind']

    cf_vals = list(range(0, cf_max + 1, step))
    sol_vals = list(range(0, sol_max + 1, step))
    wnd_vals = list(range(0, wnd_max + 1, step))
    hyd_vals = list(range(0, int(hydro_cap) + 1, step))
    if int(hydro_cap) not in hyd_vals and hydro_cap > 0:
        hyd_vals.append(int(hydro_cap))

    if iso == 'CAISO':
        geo_vals = list(range(0, GEO_CAP_PCT + 1, step))
        combos = np.array(list(itertools.product(
            cf_vals, sol_vals, wnd_vals, hyd_vals, geo_vals
        )), dtype=np.float64)
    else:
        combos = np.array(list(itertools.product(
            cf_vals, sol_vals, wnd_vals, hyd_vals
        )), dtype=np.float64)

    # Filter: at least some generation (sum > 0)
    row_sums = combos.sum(axis=1)
    combos = combos[row_sums > 0]

    # Filter: total procurement within observed cost-optimal range + buffer.
    # Max observed across all step 3 winners: 281% (MISO 99.9%). Cap at 350%.
    if len(combos) > 0:
        total = combos.sum(axis=1)
        combos = combos[total <= TOTAL_PROCUREMENT_CAP]

    return combos


def generate_resource_combos_around(base_combo, iso, step=1, radius=4):
    """Generate fine combos in neighborhood of base_combo.

    Each resource varies ±radius*step from its base value, clamped to
    valid range. Returns deduplicated array.
    """
    hydro_cap = HYDRO_CAPS[iso] + HYDRO_ADDER_PCT
    rtypes = get_resource_types(iso)
    n_res = len(rtypes)

    caps = []
    for rt in rtypes:
        if rt == 'hydro':
            caps.append(int(hydro_cap))
        elif rt == 'geothermal':
            caps.append(GEO_CAP_PCT)
        else:
            caps.append(RESOURCE_CAPS[rt])

    ranges = []
    for i in range(n_res):
        base_val = int(base_combo[i])
        lo = max(0, base_val - radius * step)
        hi = min(caps[i], base_val + radius * step)
        ranges.append(list(range(lo, hi + 1, step)))

    combos = np.array(list(itertools.product(*ranges)), dtype=np.float64)

    # Filter: at least some generation + total procurement cap
    if len(combos) > 0:
        row_sums = combos.sum(axis=1)
        combos = combos[(row_sums > 0) & (row_sums <= TOTAL_PROCUREMENT_CAP)]
        combos = np.unique(combos, axis=0)

    return combos


# Edge case seed combos: direct % of demand (not mix fractions)
# These guarantee extreme-but-potentially-optimal strategies are evaluated.
# Updated to reflect tightened solar cap (100%) and total cap (350%).
EDGE_CASE_SEEDS_BASE = [
    # (clean_firm, solar, wind, hydro) as % of demand
    # High solar + storage strategies
    [0, 100, 0, 0],
    [0, 100, 50, 0],
    [0, 80, 0, 0],
    [0, 80, 50, 0],
    [10, 100, 0, 0],
    # High wind + storage strategies
    [0, 0, 200, 0],
    [0, 50, 200, 0],
    [0, 0, 150, 0],
    [0, 50, 150, 0],
    [10, 0, 200, 0],
    # Balanced high renewable
    [0, 80, 100, 0],
    [0, 80, 80, 0],
    [10, 80, 80, 0],
    # Clean firm dominant
    [100, 0, 0, 0],
    [100, 50, 0, 0],
    [100, 0, 50, 0],
    [80, 50, 50, 0],
    [120, 0, 0, 0],
    # Firm + moderate renewables
    [60, 80, 20, 0],
    [60, 20, 80, 0],
    [40, 80, 50, 0],
    [40, 50, 100, 0],
    # Minimal firm, heavy renewables
    [10, 80, 100, 0],
    [0, 80, 100, 0],
    [0, 80, 150, 0],
    # Pure renewables
    [0, 80, 80, 0],
]

# Extra CAISO seeds with geothermal (5th column = geothermal %)
EDGE_CASE_SEEDS_CAISO_GEO = [
    # (clean_firm, solar, wind, hydro, geothermal) as % of demand
    [0, 80, 0, 0, 20],
    [0, 80, 50, 0, 20],
    [0, 0, 150, 0, 20],
    [50, 80, 0, 0, 20],
    [80, 50, 0, 0, 20],
    [0, 80, 80, 0, 10],
    [20, 100, 100, 0, 15],
]


def get_seed_combos(iso):
    """Return edge case seeds valid for this ISO's resource caps."""
    hydro_cap = HYDRO_CAPS[iso] + HYDRO_ADDER_PCT
    rtypes = get_resource_types(iso)
    n_res = len(rtypes)

    seeds = []
    if iso == 'CAISO':
        # Add base seeds with geo=0, then CAISO-specific geo seeds
        for seed in EDGE_CASE_SEEDS_BASE:
            seeds.append(seed + [0])  # append geo=0
        seeds.extend(EDGE_CASE_SEEDS_CAISO_GEO)
    else:
        seeds = list(EDGE_CASE_SEEDS_BASE)

    valid = []
    seen = set()
    for seed in seeds:
        if len(seed) != n_res:
            continue
        # Check resource caps
        if seed[3] > hydro_cap:
            continue
        if n_res > 4 and seed[4] > GEO_CAP_PCT:
            continue
        if seed[0] > RESOURCE_CAPS['clean_firm']:
            continue
        if seed[1] > RESOURCE_CAPS['solar']:
            continue
        if seed[2] > RESOURCE_CAPS['wind']:
            continue
        if sum(seed) == 0:
            continue
        key = tuple(seed)
        if key not in seen:
            seen.add(key)
            valid.append(seed)

    return np.array(valid, dtype=np.float64) if valid else np.empty((0, n_res))


# ══════════════════════════════════════════════════════════════════════════════
# COARSE CACHE — one-time per-ISO sweep
# ══════════════════════════════════════════════════════════════════════════════

def _coarse_cache_path(iso):
    """Path for cached coarse sweep results."""
    return os.path.join(STEP1_RAW_PFS_PARQUET_DIR, f'{iso}_coarse_cache.parquet')


def build_coarse_cache(iso, demand_arr, supply_matrix):
    """Two-stage adaptive coarse sweep: 10% recon → targeted 5% fill.

    Stage 1: Score all combos at 10% step (~15× fewer than uniform 5%).
    Stage 2: Generate 5% sub-combos only around 10% cells that scored
             above a viability floor, eliminating dead space.

    The viability floor (10% hourly match) is conservative: the lowest
    optimized threshold is 50% and storage can add ~40pp, so any combo
    scoring below 10% is unreachable even with maximum storage.

    Returns (combos, scores) where combos is (N, n_res) and scores is (N,).
    """
    rtypes = get_resource_types(iso)
    n_res = len(rtypes)

    # ── Stage 1: 10% reconnaissance sweep ──
    recon_combos = generate_resource_combos(iso, step=10)
    n_recon = len(recon_combos)
    print(f"    {iso}: Stage 1 — 10% recon sweep ({n_recon:,} combos, {n_res}D)")

    recon_scores = batch_hourly_scores(demand_arr, supply_matrix, recon_combos)

    # Viability floor: lowest threshold is now 10%, and storage adds up
    # to ~40pp uplift. A combo must score at least ~1% to be reachable
    # for any threshold. We use 0.01 as a conservative floor — this
    # only eliminates combos with near-zero generation (effectively empty).
    VIABILITY_FLOOR = 0.01
    interesting_mask = recon_scores >= VIABILITY_FLOOR
    n_interesting = int(interesting_mask.sum())
    pct_interesting = n_interesting / n_recon * 100 if n_recon > 0 else 0
    print(f"    {iso}: {n_interesting:,}/{n_recon:,} cells above viability floor "
          f"({pct_interesting:.0f}%)")

    # ── Stage 2: Targeted 5% fill around interesting cells ──
    targeted_parts = [recon_combos]  # Always include the full 10% grid

    for combo in recon_combos[interesting_mask]:
        fine = generate_resource_combos_around(combo, iso, step=5, radius=1)
        targeted_parts.append(fine)

    # Add seed combos (known-interesting mixes)
    seeds = get_seed_combos(iso)
    if len(seeds) > 0:
        targeted_parts.append(seeds)

    combos = np.unique(np.vstack(targeted_parts), axis=0)
    # Filter: at least some generation (sum > 0)
    row_sums = combos.sum(axis=1)
    combos = combos[row_sums > 0]

    n_combos = len(combos)
    print(f"    {iso}: Stage 2 — {n_combos:,} targeted 5% combos")

    # Score all targeted combos
    scores = batch_hourly_scores(demand_arr, supply_matrix, combos)

    # Save to parquet cache
    if HAS_PARQUET:
        os.makedirs(STEP1_RAW_PFS_PARQUET_DIR, exist_ok=True)
        data = {}
        for i, rt in enumerate(rtypes):
            data[rt] = combos[:, i]
        data['score'] = scores
        table = pa.table(data)
        pq.write_table(table, _coarse_cache_path(iso), compression='snappy')
        print(f"    {iso}: Coarse cache saved ({n_combos:,} combos)")

    return combos, scores


def load_coarse_cache(iso):
    """Load cached coarse sweep results. Returns (combos, scores) or None."""
    path = _coarse_cache_path(iso)
    if not HAS_PARQUET or not os.path.exists(path):
        return None
    try:
        table = pq.read_table(path)
        rtypes = get_resource_types(iso)
        combos = np.column_stack([
            table.column(rt).to_numpy() for rt in rtypes
        ])
        scores = table.column('score').to_numpy()
        print(f"    {iso}: Coarse cache loaded ({len(combos):,} combos)")
        return combos, scores
    except Exception as e:
        print(f"    {iso}: Could not load coarse cache: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZER CORE — per-threshold (reads from coarse cache)
# ══════════════════════════════════════════════════════════════════════════════

def optimize_threshold(iso, threshold, demand_arr, supply_matrix,
                       coarse_combos, coarse_scores,
                       storage_levels=None):
    """Find feasible solutions for a single threshold × ISO.

    Flow:
      1. Procurement window pre-filter (lower bound = target, scaled upper)
      2. Score all coarse generation mixes (no storage)
      3. Feasible (score >= target) → keep as-is
      4. Near-miss (within 40pp) with curtailment → fixed storage sweep
      5. Fine refinement at 1% step around feasible archetypes

    Returns list of candidate dicts.
    """
    rtypes = get_resource_types(iso)
    n_res = len(rtypes)
    target = threshold / 100.0

    # Storage constants
    batt_eff = BATTERY_EFFICIENCY
    batt8_eff = BATTERY8_EFFICIENCY
    ldes_eff = LDES_EFFICIENCY
    ldes_window_hours = LDES_WINDOW_DAYS * 24
    h2_eff = H2_EFFICIENCY
    h2_dur = H2_DURATION_HOURS
    h2_window_hours = H2_WINDOW_DAYS * 24

    # Storage levels — fixed coarse sweep with threshold-dependent expansion
    if storage_levels is not None:
        batt_levels = storage_levels['bat4']
        batt8_levels = storage_levels['bat8']
        ldes_levels = storage_levels['ldes']
        h2_levels = storage_levels.get('h2', [0])
    elif threshold >= 95:
        batt_levels = [0, 1, 3, 5]
        batt8_levels = [0, 2, 4, 6]
        ldes_levels = [0, 5, 10, 20]
        h2_levels = [0, 5, 10, 20]
    else:
        batt_levels = [0, 1, 3]
        batt8_levels = [0, 2, 4]
        ldes_levels = [0, 5, 10]
        h2_levels = [0]

    MIN_SURPLUS_DAYS_FOR_BATTERY = 150
    NM_CHUNK = 10000       # ~0.7 GiB per chunk at float64 × 8760 (reduced from 20K for GH Actions 7GB runners)
    MAX_MIX_BATCH = 100    # max mixes per storage-sweep batch
    batt8_window = 48

    candidates = []
    seen = set()
    chunk_num = [0]        # mutable counter for chunk saves (list for closure access)
    total_batches = [0]    # total batches processed across all phases

    def _flush_candidates_to_chunk():
        """Save current candidates to a chunk file and clear the list."""
        if candidates:
            chunk_num[0] = _save_chunk(iso, threshold, candidates, chunk_num[0])
            candidates.clear()

    def _maybe_flush(force=False):
        """Flush candidates if over the limit or if forced."""
        if force or len(candidates) >= CHUNK_CANDIDATE_LIMIT:
            _flush_candidates_to_chunk()

    def add_candidate(mix_arr, bp, b8p, lp, score, h2p=0):
        """Add candidate if not already seen."""
        mix_key = tuple(int(mix_arr[i]) for i in range(n_res))
        key = (mix_key, bp, b8p, lp, h2p)
        if key not in seen:
            seen.add(key)
            cand = {
                'resource_mix': {rt: mix_key[j] for j, rt in enumerate(rtypes)},
                'battery_dispatch_pct': bp,
                'battery8_dispatch_pct': b8p,
                'ldes_dispatch_pct': lp,
                'h2_dispatch_pct': h2p,
                'hourly_match_score': round(score * 100, 2),
            }
            candidates.append(cand)

    # ── Pre-filter: procurement window ──
    # Lower bound: never search mixes with less total procurement than target.
    # Upper bound scales with threshold to exclude over-procured mixes:
    #   <50%:   target to max(target+30, 50)  (generous for low targets)
    #   50-75%: target to 2× target
    #   80-90%: target to 200%
    #   >90%:   target to 250% (was unlimited — capped to prevent OOM on
    #           GitHub Actions runners; mixes >250% are heavily over-procured
    #           and dominated by leaner alternatives at every threshold)
    total_procurement = coarse_combos.sum(axis=1)
    proc_lower = threshold
    if threshold < 50:
        proc_upper = max(threshold + 30, 50)
    elif threshold <= 75:
        proc_upper = 2.0 * threshold
    elif threshold <= 90:
        proc_upper = 200.0
    else:
        proc_upper = 250.0

    n_before = len(coarse_combos)
    proc_mask = total_procurement >= proc_lower
    if proc_upper is not None:
        proc_mask &= total_procurement <= proc_upper
    coarse_combos = coarse_combos[proc_mask]
    coarse_scores = coarse_scores[proc_mask]
    window_str = f"[{proc_lower}%, {proc_upper}%]" if proc_upper else f"[{proc_lower}%, unlimited]"
    print(f"      {iso} {threshold}%: Procurement window {window_str} → "
          f"{len(coarse_combos):,} of {n_before:,} mixes retained")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: GENERATION SCORING (coarse + fine, NO storage)
    # Score all generation mixes first, then run storage on the combined
    # set. This ensures storage sweep sees all hydro levels including
    # fine-grid values at exact hydro caps (e.g., hydro=2 for PJM).
    # ══════════════════════════════════════════════════════════════════

    # ── Phase 1a: Score coarse cache (no storage) ──
    feasible_mask = coarse_scores >= target
    feasible_idx = np.where(feasible_mask)[0]

    for i in feasible_idx:
        add_candidate(coarse_combos[i], 0, 0, 0, coarse_scores[i])
    _maybe_flush()

    print(f"      {iso} {threshold}%: {len(feasible_idx):,} coarse feasible (no storage)")

    # ── Phase 1b: Fine 1% refinement around boundary archetypes (no storage) ──
    FINE_REFINEMENT_BAND = 0.03  # 3pp above target
    FINE_REFINEMENT_MIN_THRESHOLD = 50
    FINE_RADIUS = 2 if n_res >= 5 else 4
    MAX_FINE_ARCHETYPES = 500 if n_res >= 5 else 5000

    # Collect all fine-grid mixes and their scores for the storage sweep
    all_fine_combos_for_storage = []
    all_fine_scores_for_storage = []

    has_any_candidates = len(candidates) > 0 or chunk_num[0] > 0
    fine_total_feasible = 0
    fine_total_combos = 0

    if has_any_candidates and threshold >= FINE_REFINEMENT_MIN_THRESHOLD:
        boundary_upper = target + FINE_REFINEMENT_BAND
        boundary_mask = (coarse_scores >= target) & (coarse_scores <= boundary_upper)
        boundary_idx = np.where(boundary_mask)[0]
        mix_archetypes = set()
        for i in boundary_idx:
            m = tuple(int(coarse_combos[i][j]) for j in range(n_res))
            mix_archetypes.add(m)

        if len(mix_archetypes) > MAX_FINE_ARCHETYPES:
            print(f"      {iso} {threshold}%: Capping fine archetypes "
                  f"{len(mix_archetypes):,} → {MAX_FINE_ARCHETYPES} (5D safety)")
            archetype_list = sorted(mix_archetypes, key=lambda m: sum(m))
            step_size = max(1, len(archetype_list) // MAX_FINE_ARCHETYPES)
            mix_archetypes = set(archetype_list[::step_size][:MAX_FINE_ARCHETYPES])

        print(f"      {iso} {threshold}%: Phase 1b — {len(mix_archetypes):,} boundary archetypes, "
              f"radius={FINE_RADIUS}, ~{len(mix_archetypes) * (2*FINE_RADIUS+1)**n_res:,} max fine combos")

        ARCHETYPE_BATCH_SIZE = 100 if n_res >= 5 else 500
        archetype_list = list(mix_archetypes)
        n_arch = len(archetype_list)

        for arch_start in range(0, n_arch, ARCHETYPE_BATCH_SIZE):
            arch_end = min(arch_start + ARCHETYPE_BATCH_SIZE, n_arch)
            arch_batch = archetype_list[arch_start:arch_end]

            batch_fine_parts = []
            for mix_tuple in arch_batch:
                base = np.array(mix_tuple, dtype=np.float64)
                fine = generate_resource_combos_around(base, iso, step=1, radius=FINE_RADIUS)
                if len(fine) > 0:
                    batch_fine_parts.append(fine)

            if not batch_fine_parts:
                continue

            all_fine = np.unique(np.vstack(batch_fine_parts), axis=0)
            fine_proc_sums = all_fine.sum(axis=1)
            all_fine = all_fine[fine_proc_sums >= threshold]
            n_all_fine = len(all_fine)
            fine_total_combos += n_all_fine

            if n_all_fine == 0:
                continue

            all_fine_scores_arr = batch_hourly_scores(demand_arr, supply_matrix, all_fine)

            # Feasible fine combos (no storage)
            fine_feas_mask = all_fine_scores_arr >= target
            fine_feas_idx = np.where(fine_feas_mask)[0]
            for fi in fine_feas_idx:
                add_candidate(all_fine[fi], 0, 0, 0, all_fine_scores_arr[fi])
            fine_total_feasible += len(fine_feas_idx)

            # Collect ALL fine combos + scores for storage sweep later
            all_fine_combos_for_storage.append(all_fine)
            all_fine_scores_for_storage.append(all_fine_scores_arr)

            _maybe_flush()

            print(f"\r      {iso} {threshold}%: Fine arch batch "
                  f"{min(arch_end, n_arch)}/{n_arch}, "
                  f"{fine_total_combos:,} combos, "
                  f"{fine_total_feasible:,} feasible so far", end="", flush=True)

        print(f"\n      {iso} {threshold}%: Phase 1b complete — {fine_total_combos:,} fine combos, "
              f"{fine_total_feasible:,} feasible (no storage)")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: STORAGE SWEEP on combined coarse + fine near-miss mixes
    # Runs on ALL generation mixes from Phase 1a+1b, so storage sees
    # every hydro level including fine-grid values.
    # ══════════════════════════════════════════════════════════════════

    STORAGE_SWEEP_FLOOR = 0.50
    # Tiered near-miss window: wider at lower thresholds where storage has more headroom
    if threshold >= 95:
        nm_width = 0.15   # 15pp
    elif threshold >= 85:
        nm_width = 0.30   # 30pp
    else:
        nm_width = 0.40   # 40pp
    near_miss_lower = max(target - nm_width, STORAGE_SWEEP_FLOOR)
    storage_feasible = 0

    # Build combined near-miss array: coarse + fine mixes
    # Coarse near-miss
    coarse_nm_mask = (~feasible_mask) & (coarse_scores >= near_miss_lower)
    coarse_nm_idx = np.where(coarse_nm_mask)[0]
    coarse_nm_combos = coarse_combos[coarse_nm_idx]

    # Fine near-miss
    fine_nm_combos_list = []
    if all_fine_combos_for_storage:
        for fc, fs in zip(all_fine_combos_for_storage, all_fine_scores_for_storage):
            fine_nm_lower = max(target - 0.30, STORAGE_SWEEP_FLOOR)
            fine_feas = fs >= target
            fine_nm = (~fine_feas) & (fs >= fine_nm_lower)
            if fine_nm.any():
                fine_nm_combos_list.append(fc[fine_nm])

    # Combine
    if fine_nm_combos_list:
        fine_nm_combos = np.vstack(fine_nm_combos_list)
        combined_nm = np.vstack([coarse_nm_combos, fine_nm_combos]) if len(coarse_nm_combos) > 0 else fine_nm_combos
        # Track which source each mix came from (for supply computation)
        combined_source = np.concatenate([
            np.zeros(len(coarse_nm_combos), dtype=np.int8),
            np.ones(len(fine_nm_combos), dtype=np.int8)])
    elif len(coarse_nm_combos) > 0:
        combined_nm = coarse_nm_combos
        combined_source = np.zeros(len(coarse_nm_combos), dtype=np.int8)
    else:
        combined_nm = np.empty((0, n_res), dtype=coarse_combos.dtype)
        combined_source = np.empty(0, dtype=np.int8)

    n_combined_nm = len(combined_nm)
    n_coarse_nm = len(coarse_nm_combos)
    n_fine_nm = n_combined_nm - n_coarse_nm

    if n_combined_nm > 0:
        print(f"      {iso} {threshold}%: Storage sweep — {n_combined_nm:,} near-miss "
              f"({n_coarse_nm:,} coarse + {n_fine_nm:,} fine)")

        # Chunk-wise storage cap computation + total surplus for curtailment gate
        b4_caps = np.empty(n_combined_nm, dtype=np.float64)
        b8_caps = np.empty(n_combined_nm, dtype=np.float64)
        l_caps = np.empty(n_combined_nm, dtype=np.float64)
        hc_arr = np.empty(n_combined_nm, dtype=np.int64)
        sd_arr = np.empty(n_combined_nm, dtype=np.int64)
        total_surplus = np.empty(n_combined_nm, dtype=np.float64)
        demand_total = demand_arr.sum()
        n_cap_chunks = (n_combined_nm + NM_CHUNK - 1) // NM_CHUNK

        for cs in range(0, n_combined_nm, NM_CHUNK):
            ce = min(cs + NM_CHUNK, n_combined_nm)
            cap_chunk_idx = cs // NM_CHUNK + 1
            if n_cap_chunks > 1:
                print(f"\r        Cap computation: chunk {cap_chunk_idx}/{n_cap_chunks} "
                      f"({cs:,}/{n_combined_nm:,} mixes)", end="", flush=True)
            chunk_fracs = combined_nm[cs:ce].astype(np.float64) / 100.0
            chunk_supply = chunk_fracs @ supply_matrix
            chunk_n = ce - cs
            cb4, cb8, cl, chc, csd = _batch_compute_storage_caps(
                demand_arr, chunk_supply, 1.0, chunk_n,
                BATTERY_DURATION_HOURS, BATTERY8_DURATION_HOURS, LDES_DURATION_HOURS)
            b4_caps[cs:ce] = cb4
            b8_caps[cs:ce] = cb8
            l_caps[cs:ce] = cl
            hc_arr[cs:ce] = chc
            sd_arr[cs:ce] = csd
            # Total annual surplus for curtailment-magnitude gate
            chunk_surplus = np.maximum(chunk_supply - demand_arr[np.newaxis, :], 0.0)
            total_surplus[cs:ce] = chunk_surplus.sum(axis=1)
        if n_cap_chunks > 1:
            print()

        # Curtailment-magnitude gate: if total surplus × best efficiency < gap,
        # storage physically cannot bridge the gap — skip the mix.
        has_curtailment = hc_arr.astype(bool)
        n_with_curtailment = has_curtailment.sum()
        best_eff = max(BATTERY_EFFICIENCY, BATTERY8_EFFICIENCY, LDES_EFFICIENCY)

        # Score each near-miss mix — coarse mixes have scores, fine mixes need recompute
        nm_scores = np.empty(n_combined_nm, dtype=np.float64)
        nm_scores[:n_coarse_nm] = coarse_scores[coarse_nm_idx]
        if n_fine_nm > 0:
            # Fine mix scores were computed in Phase 1b; recompute cheaply
            fine_nm_supply = (combined_nm[n_coarse_nm:].astype(np.float64) / 100.0) @ supply_matrix
            nm_scores[n_coarse_nm:] = _batch_score_no_storage(
                demand_arr, fine_nm_supply, 1.0, n_fine_nm)

        score_gap = target - nm_scores
        max_score_lift = total_surplus * best_eff / demand_total
        can_bridge = max_score_lift >= score_gap

        # Combine curtailment + bridge filters
        valid_mask = has_curtailment & can_bridge
        valid_ci = np.where(valid_mask)[0]
        n_gated = int(n_with_curtailment) - len(valid_ci)
        print(f"      {iso} {threshold}%: {len(valid_ci):,} viable for storage sweep "
              f"(curtailment: {n_with_curtailment:,}, gated: {n_gated:,} insufficient surplus)")

        # Identify force_1d candidates: mixes with curtailment & surplus headroom
        # that are outside step1d's normal near-miss window. These get flagged so
        # step1d includes them even if they'd normally be excluded.
        # force_1d = has curtailment AND can bridge AND score < (target - 2×nm_width)
        # (i.e., outside 2× the tiered near-miss window that step1d uses)
        force_1d_width = nm_width * 2.0
        force_1d_mask = (has_curtailment & can_bridge &
                         (nm_scores < (target - force_1d_width)))
        force_1d_indices = np.where(force_1d_mask)[0]

        nm_valid = [(int(ci), ci,  # ci IS the index into combined_nm
                     b4_caps[ci] * 100.0 * 1.1,
                     b8_caps[ci] * 100.0 * 1.1,
                     l_caps[ci] * 100.0 * 1.1,
                     int(sd_arr[ci])) for ci in valid_ci]

        b4_arr = np.array(batt_levels, dtype=np.float64)
        b8_arr = np.array(batt8_levels, dtype=np.float64)
        l_arr = np.array(ldes_levels, dtype=np.float64)
        h2_arr = np.array(h2_levels, dtype=np.float64)
        n_b4, n_b8, n_l, n_h2 = len(b4_arr), len(b8_arr), len(l_arr), len(h2_arr)

        n_total_batches = (len(nm_valid) + MAX_MIX_BATCH - 1) // MAX_MIX_BATCH
        batch_num = 0

        for batch_start in range(0, len(nm_valid), MAX_MIX_BATCH):
            batch_end = min(batch_start + MAX_MIX_BATCH, len(nm_valid))
            batch = nm_valid[batch_start:batch_end]
            n_batch = len(batch)
            batch_num += 1

            print(f"\r        Storage sweep: batch {batch_num}/{n_total_batches} "
                  f"({batch_start}/{len(nm_valid)} mixes, "
                  f"{storage_feasible:,} feasible so far)", end="", flush=True)

            batch_orig_idx = np.array([batch[bi][1] for bi in range(n_batch)])
            batch_supply = (combined_nm[batch_orig_idx].astype(np.float64) / 100.0) @ supply_matrix

            batch_scores = _batch_mixes_storage_screen(
                demand_arr, batch_supply, 1.0, n_batch,
                b4_arr, b8_arr, l_arr, n_b4, n_b8, n_l,
                batt_eff, batt8_eff, ldes_eff,
                BATTERY_DURATION_HOURS, BATTERY8_DURATION_HOURS, LDES_DURATION_HOURS,
                ldes_window_hours, batt8_window,
                h2_arr, n_h2, h2_eff, float(h2_dur), h2_window_hours)

            for bi in range(n_batch):
                ci, orig_idx, bat4_max_pct, bat8_max_pct, ldes_max_pct, n_sd = batch[bi]
                mix = combined_nm[orig_idx]
                scores = batch_scores[bi]

                for b4_idx in range(n_b4):
                    bp = b4_arr[b4_idx]
                    if bp > 0 and (n_sd < MIN_SURPLUS_DAYS_FOR_BATTERY or bp > bat4_max_pct):
                        continue
                    for b8_idx in range(n_b8):
                        b8p = b8_arr[b8_idx]
                        if b8p > 0 and (n_sd < MIN_SURPLUS_DAYS_FOR_BATTERY or b8p > bat8_max_pct):
                            continue
                        for l_idx in range(n_l):
                            lp = l_arr[l_idx]
                            if lp > 0 and lp > ldes_max_pct:
                                continue
                            for h2_idx in range(n_h2):
                                h2p = h2_arr[h2_idx]
                                if bp == 0 and b8p == 0 and lp == 0 and h2p == 0:
                                    continue
                                idx = (b4_idx * n_b8 * n_l * n_h2 +
                                       b8_idx * n_l * n_h2 +
                                       l_idx * n_h2 + h2_idx)
                                score = scores[idx]
                                if score < 0 or score < target:
                                    continue
                                add_candidate(mix, bp, b8p, lp, score, h2p)
                                storage_feasible += 1

            total_batches[0] += 1
            if total_batches[0] % CHUNK_SAVE_INTERVAL == 0:
                _maybe_flush(force=True)
            else:
                _maybe_flush()

        print(f"\r        Storage sweep: {n_total_batches}/{n_total_batches} batches done"
              f"                                        ")
        print(f"      {iso} {threshold}%: +{storage_feasible:,} with storage "
              f"({len(nm_valid):,} curtailing / {n_combined_nm:,} near-miss)")
        _maybe_flush()
        # Write force_1d candidates: mixes with surplus headroom but outside
        # step1d's normal near-miss window. Step1d can read this to expand its pool.
        if len(force_1d_indices) > 0:
            _write_force_1d_candidates(
                iso, threshold, combined_nm, force_1d_indices,
                nm_scores, total_surplus, max_score_lift, score_gap, rtypes)
    else:
        print(f"      {iso} {threshold}%: 0 near-miss for storage sweep")
        force_1d_indices = np.empty(0, dtype=np.int64)

    # Free fine-grid arrays
    del all_fine_combos_for_storage, all_fine_scores_for_storage

    # Final flush of any remaining candidates
    _maybe_flush(force=True)

    # Return empty list — all candidates are in chunk files now
    # The caller should use _merge_chunks_and_finalize to produce the final parquet
    return candidates


def dominance_filter_candidates(candidates, rtypes):
    """Remove candidates whose resource mix is dominated by a leaner mix.

    Candidate C is dominated if another candidate D has all resources <= C
    and is not identical (D achieves the same target with fewer resources).
    Only compares resource mix dimensions; storage configs are independent.

    Uses an incremental Pareto front algorithm sorted by total resources
    ascending. Each candidate is checked only against the current front
    (typically hundreds–low thousands), not all candidates, giving
    O(n × front_size × dims) instead of O(n² × dims).

    Returns filtered list of non-dominated candidates.
    """
    if len(candidates) <= 1:
        return candidates

    # Group by unique resource mix to avoid processing duplicate mixes
    mix_to_candidates = {}
    for c in candidates:
        mix_key = tuple(c['resource_mix'][rt] for rt in rtypes)
        if mix_key not in mix_to_candidates:
            mix_to_candidates[mix_key] = []
        mix_to_candidates[mix_key].append(c)

    unique_mixes = list(mix_to_candidates.keys())
    n_unique = len(unique_mixes)
    if n_unique <= 1:
        return candidates

    mix_arr = np.array(unique_mixes, dtype=np.float64)
    ndim = mix_arr.shape[1]

    # Sort by total procurement ascending — leanest mixes first.
    # Leanest mixes are most likely non-dominated, so the front builds
    # quickly and later (larger) mixes are efficiently pruned.
    order = np.argsort(mix_arr.sum(axis=1))

    # Pre-allocate Pareto front array (trimmed at end)
    front_cap = min(n_unique, 50000)
    front = np.empty((front_cap, ndim), dtype=np.float64)
    front_size = 0
    pareto_indices = []  # indices into unique_mixes (original ordering)

    for idx in order:
        mix_i = mix_arr[idx]

        if front_size > 0:
            fslice = front[:front_size]
            # Check if any front member dominates mix_i:
            # front[j] dominates mix_i iff all(front[j] <= mix_i) and !=
            leq = np.all(fslice <= mix_i, axis=1)
            if np.any(leq):
                eq = np.all(fslice == mix_i, axis=1)
                if np.any(leq & ~eq):
                    continue  # dominated — skip

            # Check if mix_i dominates any existing front members
            geq = np.all(fslice >= mix_i, axis=1)
            if np.any(geq):
                eq = np.all(fslice == mix_i, axis=1)
                dominated_by_new = geq & ~eq
                if np.any(dominated_by_new):
                    keep = ~dominated_by_new
                    new_size = int(keep.sum())
                    front[:new_size] = fslice[keep]
                    pareto_indices = [p for p, k in
                                     zip(pareto_indices, keep) if k]
                    front_size = new_size

        # Add to front (grow if needed)
        if front_size >= front_cap:
            extra = np.empty((10000, ndim), dtype=np.float64)
            front = np.vstack([front, extra])
            front_cap = len(front)
        front[front_size] = mix_i
        front_size += 1
        pareto_indices.append(int(idx))

    kept_mixes = {unique_mixes[i] for i in pareto_indices}
    filtered = [c for c in candidates
                if tuple(c['resource_mix'][rt] for rt in rtypes) in kept_mixes]
    return filtered


def extract_pareto(candidates, iso):
    """Extract Pareto-optimal candidates along generation/storage tradeoff.

    Returns 3-5 representative points:
    1. Minimum total generation (sum of resource fractions)
    2. Minimum storage
    3. Minimum total (generation + storage balanced)
    4-5. Diverse mix archetypes if available
    """
    if not candidates:
        return []

    rtypes = get_resource_types(iso)

    # Deduplicate
    seen = set()
    unique = []
    for c in candidates:
        m = tuple(c['resource_mix'][rt] for rt in rtypes)
        key = (m, c['battery_dispatch_pct'],
               c.get('battery8_dispatch_pct', 0), c['ldes_dispatch_pct'],
               c.get('h2_dispatch_pct', 0))
        if key not in seen:
            seen.add(key)
            unique.append(c)

    if not unique:
        return []

    # Metrics for Pareto
    for c in unique:
        c['total_storage'] = (c['battery_dispatch_pct'] +
                              c.get('battery8_dispatch_pct', 0) +
                              c['ldes_dispatch_pct'] +
                              c.get('h2_dispatch_pct', 0))
        c['total_generation'] = sum(c['resource_mix'].get(rt, 0) for rt in rtypes)
        c['total_resource'] = c['total_generation'] + c['total_storage']

    by_generation = sorted(unique, key=lambda c: (c['total_generation'], c['total_storage']))
    by_storage = sorted(unique, key=lambda c: (c['total_storage'], c['total_generation']))
    by_total = sorted(unique, key=lambda c: c['total_resource'])

    pareto = []
    pareto_keys = set()

    def add_if_new(candidate, ptype):
        m = tuple(candidate['resource_mix'][rt] for rt in rtypes)
        key = (m, candidate['battery_dispatch_pct'],
               candidate.get('battery8_dispatch_pct', 0), candidate['ldes_dispatch_pct'],
               candidate.get('h2_dispatch_pct', 0))
        if key not in pareto_keys:
            pareto_keys.add(key)
            c = dict(candidate)
            c['pareto_type'] = ptype
            for k in ('total_storage', 'total_generation', 'total_resource'):
                c.pop(k, None)
            pareto.append(c)

    add_if_new(by_generation[0], 'min_generation')
    add_if_new(by_storage[0], 'min_storage')
    add_if_new(by_total[0], 'min_total')

    by_solar = sorted(unique, key=lambda c: -c['resource_mix'].get('solar', 0))
    by_wind = sorted(unique, key=lambda c: -c['resource_mix'].get('wind', 0))
    by_firm = sorted(unique, key=lambda c: -c['resource_mix'].get('clean_firm', 0))

    for cand, ptype in [(by_solar[0], 'high_solar'), (by_wind[0], 'high_wind'),
                        (by_firm[0], 'high_firm')]:
        if len(pareto) >= 5:
            break
        add_if_new(cand, ptype)

    return pareto



def build_fine_storage_levels(max_bat4, max_bat8, max_ldes, max_h2):
    """Build refined storage levels from Phase 1 coarse saturation analysis.

    Phase 2 refinement: 0.5 steps for batteries, 1.0 for LDES, 2.0 for H2,
    from 0 up to max_used + headroom. Returns None if no storage was used.

    Combo counts (worst-case ≥95%): ~12×8×12×6 = 5,760/mix.
    """
    if max_bat4 == 0 and max_bat8 == 0 and max_ldes == 0 and max_h2 == 0:
        return None

    def _range(step, ceiling):
        """Generate [0, step, 2*step, ..., ceiling] inclusive."""
        levels = [0]
        v = step
        while v <= ceiling + 0.001:
            levels.append(round(v, 2))
            v += step
        return levels

    # bat4: 0.5 steps, up to max+0.5 (cap 6)
    fine_bat4 = _range(0.5, min(max_bat4 + 0.5, 6.0)) if max_bat4 > 0 else [0]
    # bat8: 1.0 steps, up to max+1 (cap 8)
    fine_bat8 = _range(1.0, min(max_bat8 + 1.0, 8.0)) if max_bat8 > 0 else [0]
    # LDES: 1.0 steps, up to max+1 (cap 25)
    fine_ldes = _range(1.0, min(max_ldes + 1.0, 25.0)) if max_ldes > 0 else [0]
    # H2: 2.0 steps, up to max+2 (cap 25)
    fine_h2 = _range(2.0, min(max_h2 + 2.0, 25.0)) if max_h2 > 0 else [0]

    return {'bat4': fine_bat4, 'bat8': fine_bat8, 'ldes': fine_ldes, 'h2': fine_h2}


# ══════════════════════════════════════════════════════════════════════════════
# PER-ISO PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def process_iso(args):
    """Process all thresholds for a single ISO.

    Architecture (v5.1):
    1. Build coarse cache (one-time): score all resource fraction combos at 5% step
    2. For each threshold: score pure generation mixes → feasible kept as-is →
       near-miss with curtailment get fixed storage sweep → fine mix refinement
    3. Single-pass storage with fixed coarse levels (3-4 per type, expanded ≥95%)
    """
    iso, demand_data, gen_profiles = args
    iso_start = time.time()
    rtypes = get_resource_types(iso)

    demand_norm = demand_data[iso]['normalized']
    supply_profiles = get_supply_profiles(iso, gen_profiles)
    demand_arr, supply_matrix = prepare_numpy_profiles(iso, demand_norm, supply_profiles)
    hydro_cap = HYDRO_CAPS[iso]

    print(f"\n  {iso}: Starting optimization ({len(THRESHOLDS)} thresholds, "
          f"{len(rtypes)}D, hydro_cap={hydro_cap}%)")

    iso_results = {
        'iso': iso,
        'label': ISO_LABELS.get(iso, iso),
        'annual_demand_mwh': demand_data[iso]['total_annual_mwh'],
        'peak_demand_mw': demand_data[iso]['peak_mw'],
        'hydro_cap': hydro_cap,
        'thresholds': {},
    }

    # ── Step 1: Build or load coarse cache ──
    cached = load_coarse_cache(iso)
    if cached is not None:
        coarse_combos, coarse_scores = cached
    else:
        coarse_combos, coarse_scores = build_coarse_cache(iso, demand_arr, supply_matrix)

    # ── Step 2: Single-pass storage sweep across thresholds ──

    def _run_threshold_loop(phase_label, storage_levels_override=None):
        """Run all thresholds. Returns per-threshold results dict."""
        phase_results = {}

        for threshold in THRESHOLDS:
            # Skip already-completed thresholds (enables incremental runs)
            if _is_threshold_done(iso, threshold):
                print(f"    {iso} {threshold}%: Already done — skipping ({phase_label})")
                continue

            t_start = time.time()

            feasible = optimize_threshold(
                iso, threshold, demand_arr, supply_matrix,
                coarse_combos, coarse_scores,
                storage_levels=storage_levels_override)

            t_elapsed = time.time() - t_start
            archetypes = set()
            for c in feasible:
                archetypes.add(tuple(c['resource_mix'][rt] for rt in rtypes))
            phase_results[threshold] = feasible

            # Merge any chunk files + remaining candidates into final parquet
            _merge_chunks_and_finalize(iso, threshold, feasible)

            # Count from the final parquet (includes chunks)
            done_path = _threshold_done_path(iso, threshold)
            if os.path.exists(done_path):
                try:
                    final_table = pq.read_table(done_path)
                    n_final = final_table.num_rows
                except Exception:
                    n_final = len(feasible)
            else:
                n_final = len(feasible)

            print(f"    {iso} {threshold}%: {n_final:,} solutions "
                  f"({len(archetypes)} in-memory archetypes), {t_elapsed:.1f}s")

        return phase_results

    # Phase 1: Coarse storage sweep with fixed levels
    print(f"    {iso}: Phase 1 — Coarse storage sweep (fixed levels, curtailment-gated)")
    coarse_results = _run_threshold_loop("coarse")

    # Analyze storage saturation from Phase 1 results (read from parquets since
    # candidates may have been flushed to chunks during processing)
    max_bat4 = max_bat8 = max_ldes = max_h2 = 0.0
    for threshold in THRESHOLDS:
        done_path = _threshold_done_path(iso, threshold)
        if os.path.exists(done_path):
            try:
                t_df = pq.read_table(done_path).to_pandas()
                if 'battery_dispatch_pct' in t_df.columns:
                    max_bat4 = max(max_bat4, t_df['battery_dispatch_pct'].max())
                if 'battery8_dispatch_pct' in t_df.columns:
                    max_bat8 = max(max_bat8, t_df['battery8_dispatch_pct'].max())
                if 'ldes_dispatch_pct' in t_df.columns:
                    max_ldes = max(max_ldes, t_df['ldes_dispatch_pct'].max())
                if 'h2_dispatch_pct' in t_df.columns:
                    max_h2 = max(max_h2, t_df['h2_dispatch_pct'].max())
            except Exception:
                pass
        # Also check in-memory results
        for c in coarse_results.get(threshold, []):
            max_bat4 = max(max_bat4, c.get('battery_dispatch_pct', 0) or 0)
            max_bat8 = max(max_bat8, c.get('battery8_dispatch_pct', 0) or 0)
            max_ldes = max(max_ldes, c.get('ldes_dispatch_pct', 0) or 0)
            max_h2 = max(max_h2, c.get('h2_dispatch_pct', 0) or 0)

    print(f"    {iso}: Phase 1 saturation — bat4={max_bat4:.1f}%, bat8={max_bat8:.1f}%, "
          f"ldes={max_ldes:.1f}%, h2={max_h2:.1f}%")

    # Phase 2: Refined storage sweep around Phase 1 winners
    fine_levels = build_fine_storage_levels(max_bat4, max_bat8, max_ldes, max_h2)

    if fine_levels is not None:
        n_combos = len(fine_levels['bat4']) * len(fine_levels['bat8']) * \
                   len(fine_levels['ldes']) * len(fine_levels['h2'])
        print(f"    {iso}: Phase 2 — Refined sweep: bat4={len(fine_levels['bat4'])} levels, "
              f"bat8={len(fine_levels['bat8'])} levels, ldes={len(fine_levels['ldes'])} levels, "
              f"h2={len(fine_levels['h2'])} levels ({n_combos:,} combos/mix)")
        fine_results = _run_threshold_loop("fine", fine_levels)
    else:
        print(f"    {iso}: No storage used in Phase 1 — skipping Phase 2")
        fine_results = coarse_results

    # Count total solutions from parquet files (authoritative, includes chunks)
    total_solutions = 0
    for threshold in THRESHOLDS:
        done_path = _threshold_done_path(iso, threshold)
        if os.path.exists(done_path):
            try:
                total_solutions += pq.read_table(done_path).num_rows
            except Exception:
                pass
    print(f"    {iso}: Total: {total_solutions:,} solutions (from parquets)")

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


def _force_1d_path(iso):
    """Path for force_1d candidates: {ISO}_force_1d_candidates.parquet."""
    return os.path.join(STEP1_RAW_PFS_PARQUET_DIR, f'{iso}_force_1d_candidates.parquet')


def _write_force_1d_candidates(iso, threshold, combined_nm, force_indices,
                                nm_scores, total_surplus, max_lift, score_gap, rtypes):
    """Write force_1d candidate mixes to a parquet file for step1d to pick up.

    These are mixes that have sufficient curtailment surplus to theoretically
    bridge the gap but are outside step1d's normal near-miss window. Step1d
    can read this file and include them in its candidate pool.

    Appends to existing file (across thresholds within an ISO).
    """
    if not HAS_PARQUET or len(force_indices) == 0:
        return

    n = len(force_indices)
    rows = {
        'iso': [iso] * n,
        'threshold': [float(threshold)] * n,
    }
    for j, rt in enumerate(rtypes):
        rows[rt] = [int(combined_nm[fi][j]) for fi in force_indices]
    rows['base_score'] = [float(nm_scores[fi]) for fi in force_indices]
    rows['total_surplus'] = [float(total_surplus[fi]) for fi in force_indices]
    rows['max_score_lift'] = [float(max_lift[fi]) for fi in force_indices]
    rows['score_gap'] = [float(score_gap[fi]) for fi in force_indices]

    new_table = pa.table(rows)

    # Append to existing file if it exists
    fpath = _force_1d_path(iso)
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    if os.path.exists(fpath):
        try:
            existing = pq.read_table(fpath)
            new_table = pa.concat_tables([existing, new_table], promote_options='permissive')
        except Exception:
            pass  # overwrite if existing is corrupt

    pq.write_table(new_table, fpath, compression='snappy')
    print(f"      {iso} {threshold}%: {n:,} force_1d candidates written → "
          f"{os.path.basename(fpath)} ({new_table.num_rows:,} total)")


def _candidate_to_row(iso, threshold, c):
    """Convert a candidate dict to a flat row dict for parquet output."""
    rtypes = get_resource_types(iso)
    row = {
        'iso': iso,
        'threshold': float(threshold),
    }
    for rt in rtypes:
        row[rt] = c['resource_mix'].get(rt, 0)
    row['battery_dispatch_pct'] = float(c['battery_dispatch_pct'])
    row['battery8_dispatch_pct'] = float(c.get('battery8_dispatch_pct', 0))
    row['ldes_dispatch_pct'] = float(c['ldes_dispatch_pct'])
    row['h2_dispatch_pct'] = float(c.get('h2_dispatch_pct', 0))
    row['hourly_match_score'] = c['hourly_match_score']
    row['pareto_type'] = c.get('pareto_type', '')
    return row


def _save_threshold_done(iso, threshold, candidates):
    """Write finalized threshold parquet and clean up progress file."""
    if not HAS_PARQUET or not candidates:
        return
    os.makedirs(STEP1_RAW_PFS_PARQUET_DIR, exist_ok=True)

    rows = [_candidate_to_row(iso, threshold, c) for c in candidates]

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


def _chunk_path(iso, threshold, chunk_num):
    """Path for an intermediate chunk parquet: {ISO}_t{XX}_chunk{N}.parquet."""
    t_str = _normalize_threshold_str(threshold)
    return os.path.join(STEP1_RAW_PFS_PARQUET_DIR, f'{iso}_t{t_str}_chunk{chunk_num}.parquet')


def _save_chunk(iso, threshold, candidates, chunk_num):
    """Write a chunk of candidates to a numbered chunk parquet. Returns chunk_num+1."""
    if not HAS_PARQUET or not candidates:
        return chunk_num
    os.makedirs(STEP1_RAW_PFS_PARQUET_DIR, exist_ok=True)
    rows = [_candidate_to_row(iso, threshold, c) for c in candidates]
    table = _rows_to_table(rows)
    if table is None:
        return chunk_num
    path = _chunk_path(iso, threshold, chunk_num)
    pq.write_table(table, path, compression='snappy')
    print(f"      {iso} {threshold}%: Saved chunk {chunk_num} ({len(candidates):,} candidates) → {os.path.basename(path)}")
    return chunk_num + 1


def _merge_chunks_and_finalize(iso, threshold, remaining_candidates):
    """Merge all chunk parquets + remaining candidates into the final threshold parquet.

    Reads any existing chunk files, concatenates with remaining in-memory candidates,
    writes the final {ISO}_t{XX}_raw_pfs.parquet, and cleans up chunk files.
    """
    if not HAS_PARQUET:
        return

    os.makedirs(STEP1_RAW_PFS_PARQUET_DIR, exist_ok=True)
    t_str = _normalize_threshold_str(threshold)

    # Find all chunk files for this ISO/threshold
    chunk_tables = []
    chunk_files = []
    chunk_num = 0
    while True:
        cp = _chunk_path(iso, threshold, chunk_num)
        if os.path.exists(cp):
            try:
                chunk_tables.append(pq.read_table(cp))
                chunk_files.append(cp)
            except Exception as e:
                print(f"      Warning: Failed to read chunk {cp}: {e}")
            chunk_num += 1
        else:
            break

    # Convert remaining in-memory candidates
    if remaining_candidates:
        rows = [_candidate_to_row(iso, threshold, c) for c in remaining_candidates]
        remaining_table = _rows_to_table(rows)
        if remaining_table is not None:
            chunk_tables.append(remaining_table)

    if not chunk_tables:
        print(f"      {iso} {threshold}%: No solutions to save (0 chunks, 0 remaining)")
        return

    # Concatenate all tables (promote int64→float64 if chunks have mixed types)
    merged = pa.concat_tables(chunk_tables, promote_options='permissive')
    done_path = _threshold_done_path(iso, threshold)
    pq.write_table(merged, done_path, compression='snappy')
    print(f"      {iso} {threshold}%: Merged {len(chunk_files)} chunks + remaining → "
          f"{merged.num_rows:,} total solutions → {os.path.basename(done_path)}")

    # Clean up chunk files
    for cp in chunk_files:
        try:
            os.remove(cp)
        except OSError:
            pass

    # Clean up progress file
    progress_path = _mix_progress_path(iso, threshold)
    if os.path.exists(progress_path):
        os.remove(progress_path)


def _parse_cli_args(argv):
    """Parse CLI args for ISO, threshold, and worker count.

    Supported forms:
      CAISO ERCOT           — positional ISO names
      --threshold=95        — single threshold
      --threshold 95,97.5   — comma-separated thresholds
      --workers=4           — number of parallel ISO workers (default 1)
      --workers 0           — auto-detect from CPU count
    """
    target_isos = []
    threshold_tokens = []
    n_workers = 1
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

        if arg.startswith('--workers='):
            n_workers = int(arg.split('=', 1)[1])
            i += 1
            continue

        if arg == '--workers':
            if i + 1 >= len(argv):
                raise ValueError("--workers requires a value")
            n_workers = int(argv[i + 1])
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

    # Workers: 0 = auto (cpu_count), negative = 1
    if n_workers <= 0:
        n_workers = cpu_count()
    n_workers = max(1, n_workers)

    return target_isos, target_thresholds, n_workers


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global THRESHOLDS
    start_time = time.time()
    print("=" * 70)
    print("  v5.0 Physics Optimizer — Direct Fractions (no procurement)")
    print(f"  Numba: {'enabled' if HAS_NUMBA else 'disabled (NumPy fallback)'}")
    print(f"  Thresholds: {len(THRESHOLDS)} ({THRESHOLDS[0]}% - {THRESHOLDS[-1]}%)")
    print(f"  Resources: 4D base + 5D CAISO (geothermal)")
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

    target_isos, target_thresholds, n_workers = _parse_cli_args(sys.argv[1:])

    if target_thresholds:
        THRESHOLDS = target_thresholds
        print(f"  Target thresholds: {THRESHOLDS}")

    if target_isos:
        print(f"  Target ISOs: {target_isos}")

    run_isos = target_isos if target_isos else ISOS

    os.makedirs(STEP1_RAW_PFS_PARQUET_DIR, exist_ok=True)

    # Parallel or sequential ISO execution
    n_workers = min(n_workers, len(run_isos))
    iso_args = [(iso, demand_data, gen_profiles) for iso in run_isos]

    if n_workers > 1:
        print(f"\n  Parallel execution: {n_workers} workers for {len(run_isos)} ISOs")
        with Pool(processes=n_workers) as pool:
            results = pool.map(process_iso, iso_args)
    else:
        print(f"\n  Sequential execution: {len(run_isos)} ISOs")
        results = [process_iso(a) for a in iso_args]

    # Count solutions from parquet outputs
    total_solutions = 0
    for iso_name, iso_results in results:
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
