#!/usr/bin/env python3
"""
Shared Dispatch Utilities — Single source of truth for hourly dispatch reconstruction.
======================================================================================
Extracted from step4_1a_fossil_dispatch.py to avoid duplicating dispatch logic between the CO2
model and the LMP pricing module. Both import from here.

Provides:
  - Constants (battery/LDES params, hydro caps, grid mix, coal/oil caps, base demand)
  - get_supply_profiles(iso, gen_profiles) — generation shape profiles (Step 1 version)
  - reconstruct_hourly_dispatch(mix, demand, profiles, ...) — battery + LDES + battery8
  - compute_fossil_retirement(iso, clean_pct, ...) — remaining capacity at threshold
  - load_common_data() — demand, gen profiles, emission rates, fossil mix
  - Hourly dispatch cache: save/load/append cached 8760 profiles per archetype

The dispatch cache persists computed hourly profiles so downstream modules (CO2, LMP)
don't recompute dispatch for mixes already evaluated.
"""

import json
import os
import hashlib
import warnings
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix, lil_matrix

# Numba JIT — 10-50x speedup on battery/LDES dispatch loops
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        """Fallback no-op decorator when Numba is not installed."""
        def decorator(f):
            return f
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

MODULE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(MODULE_ROOT, 'data')
# Search local data directory and profiles subdirectory
_DATA_SEARCH = [DATA_DIR, os.path.join(DATA_DIR, 'profiles')]
SCRIPT_DIR = MODULE_ROOT  # backward compat

# Import shared constants from pipeline_config (single source of truth)
from pipeline_config import (
    ISOS, OFFSHORE_ISOS,
    OFFSHORE_WIND_CAP_TWH, CCS_CAP_TWH, GEOTHERMAL_CAP_TWH,
    BATTERY_EFFICIENCY, BATTERY_DURATION_HOURS,
    BATTERY8_EFFICIENCY, BATTERY8_DURATION_HOURS,
    LDES_EFFICIENCY, LDES_DURATION_HOURS, LDES_WINDOW_DAYS,
    H2_EFFICIENCY, H2_DURATION_HOURS, H2_WINDOW_DAYS,
    GRID_MIX_SHARES, REGIONAL_DEMAND_TWH,
    WHOLESALE_PRICES, FUEL_ADJUSTMENTS,
    CCS_RESIDUAL_EMISSION_RATE, COAL_OIL_RETIREMENT_THRESHOLD,
    H,
    # Dispatch-specific constants (migrated to pipeline_config)
    HYDRO_CAPS, COAL_CAP_TWH, OIL_CAP_TWH,
    NUCLEAR_SHARE_OF_CLEAN_FIRM, NUCLEAR_MONTHLY_CF,
    STORAGE_DISPATCH_MODE,
)

DATA_YEAR = '2025'
# Available weather years for sensitivity analysis (EIA-930 parquet data)
# Years 2021-2025 provide 5 distinct weather patterns for demand and renewable
# generation profiles. Using individual years instead of multi-year average
# captures extreme events (e.g., Winter Storm Uri 2021 in ERCOT, 2023 CAISO
# atmospheric rivers) that drive capacity adequacy decisions.
AVAILABLE_WEATHER_YEARS = ['2021', '2022', '2023', '2024', '2025']
RESOURCE_TYPES = ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro']

# Alias for backward compatibility
BASE_DEMAND_TWH = REGIONAL_DEMAND_TWH

# Dispatch-specific constants now imported from pipeline_config above:
# HYDRO_CAPS, COAL_CAP_TWH, OIL_CAP_TWH, NUCLEAR_SHARE_OF_CLEAN_FIRM, NUCLEAR_MONTHLY_CF

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_common_data():
    """Load all shared data files: demand, gen profiles, emission rates, fossil mix."""
    from eia_data_io import load_generation_profiles, load_demand_profiles, load_fossil_mix
    demand_data = load_demand_profiles()
    gen_profiles = load_generation_profiles()
    # Search multiple directories for emission rates
    emission_rates = None
    for search_dir in _DATA_SEARCH:
        epath = os.path.join(search_dir, 'egrid_emission_rates.json')
        if os.path.exists(epath):
            with open(epath) as f:
                emission_rates = json.load(f)
            break
    if emission_rates is None:
        raise FileNotFoundError("egrid_emission_rates.json not found in any data directory")
    fossil_mix = load_fossil_mix()
    return demand_data, gen_profiles, emission_rates, fossil_mix


def get_demand_profile(iso, demand_data, weather_year=None):
    """Extract normalized 8760 demand profile and total MWh for an ISO.

    Args:
        iso: ISO identifier string
        demand_data: Nested dict from load_demand_profiles()
        weather_year: Optional year string ('2021'-'2025') for weather-year
            sensitivity analysis. If None, uses DATA_YEAR (default '2025').
            Different weather years capture distinct demand patterns driven
            by temperature extremes (e.g., 2021 Winter Storm Uri in ERCOT).
    """
    year = weather_year or DATA_YEAR
    iso_demand = demand_data.get(iso, {})
    year_demand = iso_demand.get(year, iso_demand.get(DATA_YEAR, iso_demand.get('2024', {})))
    if isinstance(year_demand, dict):
        demand_norm = year_demand.get('normalized', [0.0] * H)[:H]
        total_mwh = year_demand.get('total_annual_mwh', 0)
    else:
        demand_norm = year_demand[:H] if isinstance(year_demand, list) else [0.0] * H
        total_mwh = 0
    return np.array(demand_norm[:H], dtype=np.float64), total_mwh


# ══════════════════════════════════════════════════════════════════════════════
# SUPPLY PROFILES — Step 1 version (nuclear seasonal derate, DST correction)
# ══════════════════════════════════════════════════════════════════════════════

def get_supply_profiles(iso, gen_profiles, weather_year=None):
    """Get generation shape profiles — Step 1 version with nuclear seasonal derate.

    Args:
        iso: ISO identifier string
        gen_profiles: Nested dict from load_generation_profiles()
        weather_year: Optional year string ('2021'-'2025') for weather-year
            sensitivity. Affects solar and wind profiles (different irradiance
            and wind patterns per year). Clean_firm/CCS are flat baseload
            and are not affected by weather year.

    This is the authoritative version. step5a_compute_co2.py's simpler version (flat
    clean_firm) is preserved for backward compatibility but new code should use this.
    """
    profiles = {}

    # Clean firm = nuclear seasonal-derated baseload (blended with geothermal for CAISO)
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

    _wy = weather_year or DATA_YEAR

    # Solar (with DST-aware nighttime correction)
    if iso == 'NYISO':
        p = gen_profiles.get(iso, {}).get(_wy, gen_profiles.get(iso, {}).get(DATA_YEAR, gen_profiles.get(iso, {}))).get('solar_proxy')
        if not p:
            neiso_data = gen_profiles.get('NEISO', {})
            neiso_year = neiso_data.get(_wy, neiso_data.get(DATA_YEAR, neiso_data))
            p = neiso_year.get('solar') if isinstance(neiso_year, dict) else None
        if not p:
            p = [0.0] * H
        solar_raw = list(p[:H])
    else:
        iso_data = gen_profiles.get(iso, {})
        year_data = iso_data.get(_wy, iso_data.get(DATA_YEAR, iso_data))
        if isinstance(year_data, dict):
            solar_raw = list(year_data.get('solar', [0.0] * H)[:H])
        else:
            solar_raw = [0.0] * H

    STD_UTC_OFFSETS = {'CAISO': 8, 'ERCOT': 6, 'PJM': 5, 'NYISO': 5, 'NEISO': 5, 'MISO': 6, 'SPP': 6}
    DST_START_DAY, DST_END_DAY = 69, 307
    local_start, local_end = 6, 19
    std_off = STD_UTC_OFFSETS.get(iso, 5)

    # Vectorized DST correction — replaces 8760-iteration double for-loop
    solar_arr = np.array(solar_raw[:H], dtype=np.float64)
    hour_indices = np.arange(H)
    days = hour_indices // 24
    hours = hour_indices % 24
    is_dst = (days >= DST_START_DAY) & (days < DST_END_DAY)
    utc_off = std_off - is_dst.astype(np.int64)
    utc_start = (local_start + utc_off) % 24
    utc_end = (local_end + utc_off) % 24
    # When utc_start <= utc_end: daylight = utc_start <= hour <= utc_end
    # When utc_start > utc_end (wraps midnight): daylight = hour >= utc_start OR hour <= utc_end
    no_wrap = utc_start <= utc_end
    is_daylight = np.where(no_wrap,
                           (hours >= utc_start) & (hours <= utc_end),
                           (hours >= utc_start) | (hours <= utc_end))
    solar_arr[~is_daylight] = 0.0
    profiles['solar'] = solar_arr.tolist()

    # Wind
    iso_data = gen_profiles.get(iso, {})
    year_data = iso_data.get(_wy, iso_data.get(DATA_YEAR, iso_data))
    if isinstance(year_data, dict):
        profiles['wind'] = year_data.get('wind', [0.0] * H)[:H]
    else:
        profiles['wind'] = [0.0] * H

    # Offshore wind (NYISO, NEISO, PJM, CAISO only — zeros for other ISOs)
    if iso in OFFSHORE_ISOS:
        iso_data_osw = gen_profiles.get(iso, {})
        year_data_osw = iso_data_osw.get(_wy, iso_data_osw.get(DATA_YEAR, iso_data_osw))
        if isinstance(year_data_osw, dict):
            profiles['offshore_wind'] = year_data_osw.get('offshore_wind', [0.0] * H)[:H]
        else:
            profiles['offshore_wind'] = [0.0] * H
    else:
        profiles['offshore_wind'] = [0.0] * H

    # CCS-CCGT: flat baseload (same as clean firm but separate for cost tracking)
    profiles['ccs_ccgt'] = [1.0 / H] * H

    # Hydro
    if isinstance(year_data, dict):
        profiles['hydro'] = year_data.get('hydro', [0.0] * H)[:H]
    else:
        profiles['hydro'] = [0.0] * H

    # Ensure all profiles exactly H hours, no negatives
    for rtype in RESOURCE_TYPES:
        p = profiles[rtype]
        if len(p) > H:
            p = p[:H]
        elif len(p) < H:
            p = list(p) + [0.0] * (H - len(p))
        profiles[rtype] = [max(0.0, v) for v in p]

    return profiles


# ══════════════════════════════════════════════════════════════════════════════
# HOURLY DISPATCH RECONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

# --- Numba-accelerated inner loops (10-50x faster than pure Python) ---

DISPATCH_ORDER = ['clean_firm', 'ccs_ccgt', 'hydro', 'offshore_wind', 'wind', 'solar']

# Pre-computed dispatch order as array indices into RESOURCE_TYPES for @njit
# RESOURCE_TYPES = ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro']
# DISPATCH_ORDER = ['clean_firm', 'ccs_ccgt', 'hydro', 'offshore_wind', 'wind', 'solar']
_DISPATCH_ORDER_INDICES = np.array([RESOURCE_TYPES.index(r) for r in DISPATCH_ORDER],
                                    dtype=np.int64)

# Cache version — bump when dispatch algorithm or field set changes
CACHE_VERSION = 4  # v4 = sequential dispatch aligned with step1_pfs_generator


@njit(cache=True)
def _battery_loop(residual_surplus, residual_gap, dispatch_profile,
                  capacity, power_rating, efficiency, window_hours, total_hours):
    """Inner loop for battery dispatch — sequential hour-by-hour, matching step1_pfs_generator.

    Args:
        capacity: energy capacity as fraction of annual demand
        power_rating: max charge/discharge power per hour (capacity / duration_hours)
        efficiency: round-trip efficiency
        window_hours: 24 for 4hr battery, 48 for 8hr battery
        total_hours: 8760
    """
    n_windows = (total_hours + window_hours - 1) // window_hours
    for w in range(n_windows):
        ws = w * window_hours
        we = ws + window_hours
        if we > total_hours:
            we = total_hours
        # Charge pass: sequential hour-by-hour
        stored = 0.0
        for h in range(ws, we):
            s = residual_surplus[h]
            if s > 0.0 and stored < capacity:
                charge = s
                if charge > power_rating:
                    charge = power_rating
                remaining = capacity - stored
                if charge > remaining:
                    charge = remaining
                stored += charge
                residual_surplus[h] -= charge
        # Discharge pass: sequential hour-by-hour
        available = stored * efficiency
        for h in range(ws, we):
            g = residual_gap[h]
            if g > 0.0 and available > 0.0:
                discharge = g
                if discharge > power_rating:
                    discharge = power_rating
                if discharge > available:
                    discharge = available
                dispatch_profile[h] = discharge
                residual_gap[h] -= discharge
                available -= discharge
    return dispatch_profile


@njit(cache=True)
def _ldes_loop(residual_surplus, residual_gap, dispatch_profile,
               energy_capacity, power_rating, ldes_efficiency,
               window_hours, total_hours):
    """Inner loop for LDES/H2 multi-day dispatch — sequential hour-by-hour.

    Charge phase subtracts from residual_surplus to prevent double-counting
    between LDES and downstream storage (H2). Discharge subtracts from
    residual_gap. SOC carries over between windows.

    BUG FIX (Mar 2026): Previously did NOT subtract charge from residual_surplus,
    allowing the same MWh to be consumed by both battery and LDES. Now consistent
    with battery dispatch behavior. Existing Step 1 caches still have the old
    behavior — defer re-run until next Step 1 is needed.
    """
    state_of_charge = 0.0
    num_windows = (total_hours + window_hours - 1) // window_hours

    for w in range(num_windows):
        w_start = w * window_hours
        w_end = w_start + window_hours
        if w_end > total_hours:
            w_end = total_hours

        # Charge pass: sequential hour-by-hour
        for h in range(w_start, w_end):
            s = residual_surplus[h]
            if s > 0.0 and state_of_charge < energy_capacity:
                charge_amt = s
                if charge_amt > power_rating:
                    charge_amt = power_rating
                space = energy_capacity - state_of_charge
                if charge_amt > space:
                    charge_amt = space
                state_of_charge += charge_amt
                residual_surplus[h] -= charge_amt

        # Discharge pass: sequential hour-by-hour
        for h in range(w_start, w_end):
            g = residual_gap[h]
            if g > 0.0 and state_of_charge > 0.0:
                avail = state_of_charge * ldes_efficiency
                dispatch_amt = g
                if dispatch_amt > power_rating:
                    dispatch_amt = power_rating
                if dispatch_amt > avail:
                    dispatch_amt = avail
                dispatch_profile[h] = dispatch_amt
                state_of_charge -= dispatch_amt / ldes_efficiency
                residual_gap[h] -= dispatch_amt

    return dispatch_profile


@njit(cache=True)
def _battery_loop_detailed(residual_surplus, residual_gap, dispatch_profile,
                           charge_profile, capacity, power_rating, efficiency,
                           window_hours, total_hours):
    """Battery dispatch with charge tracking — sequential hour-by-hour, matching step1."""
    n_windows = (total_hours + window_hours - 1) // window_hours
    for w in range(n_windows):
        ws = w * window_hours
        we = ws + window_hours
        if we > total_hours:
            we = total_hours
        # Charge pass: sequential hour-by-hour
        stored = 0.0
        for h in range(ws, we):
            s = residual_surplus[h]
            if s > 0.0 and stored < capacity:
                charge = s
                if charge > power_rating:
                    charge = power_rating
                remaining = capacity - stored
                if charge > remaining:
                    charge = remaining
                stored += charge
                residual_surplus[h] -= charge
                charge_profile[h] += charge
        # Discharge pass: sequential hour-by-hour
        available = stored * efficiency
        for h in range(ws, we):
            g = residual_gap[h]
            if g > 0.0 and available > 0.0:
                discharge = g
                if discharge > power_rating:
                    discharge = power_rating
                if discharge > available:
                    discharge = available
                dispatch_profile[h] = discharge
                residual_gap[h] -= discharge
                available -= discharge
    return dispatch_profile, charge_profile


@njit(cache=True)
def _ldes_loop_detailed(residual_surplus, residual_gap, dispatch_profile,
                        charge_profile, energy_capacity, power_rating,
                        ldes_efficiency, window_hours, total_hours):
    """LDES/H2 dispatch with charge tracking — sequential hour-by-hour.
    BUG FIX (Mar 2026): Now subtracts charge from residual_surplus."""
    state_of_charge = 0.0
    num_windows = (total_hours + window_hours - 1) // window_hours

    for w in range(num_windows):
        w_start = w * window_hours
        w_end = w_start + window_hours
        if w_end > total_hours:
            w_end = total_hours

        # Charge pass: sequential hour-by-hour
        for h in range(w_start, w_end):
            s = residual_surplus[h]
            if s > 0.0 and state_of_charge < energy_capacity:
                charge_amt = s
                if charge_amt > power_rating:
                    charge_amt = power_rating
                space = energy_capacity - state_of_charge
                if charge_amt > space:
                    charge_amt = space
                state_of_charge += charge_amt
                charge_profile[h] += charge_amt
                residual_surplus[h] -= charge_amt

        # Discharge pass: sequential hour-by-hour
        for h in range(w_start, w_end):
            g = residual_gap[h]
            if g > 0.0 and state_of_charge > 0.0:
                avail = state_of_charge * ldes_efficiency
                dispatch_amt = g
                if dispatch_amt > power_rating:
                    dispatch_amt = power_rating
                if dispatch_amt > avail:
                    dispatch_amt = avail
                dispatch_profile[h] = dispatch_amt
                state_of_charge -= dispatch_amt / ldes_efficiency
                residual_gap[h] -= dispatch_amt

    return dispatch_profile, charge_profile


def _dispatch_battery_detailed(residual_surplus, residual_gap, dispatch_pct,
                                duration_hours, efficiency, window_hours=24):
    """Battery dispatch with charge tracking. Modifies residual arrays in-place.

    Args:
        window_hours: 24 for 4hr battery (daily), 48 for 8hr battery (2-day)
    """
    dispatch_profile = np.zeros(H, dtype=np.float64)
    charge_profile = np.zeros(H, dtype=np.float64)
    if dispatch_pct <= 0:
        return dispatch_profile, charge_profile

    capacity = dispatch_pct / 100.0
    power_rating = capacity / duration_hours

    return _battery_loop_detailed(residual_surplus, residual_gap, dispatch_profile,
                                   charge_profile, capacity, power_rating,
                                   efficiency, window_hours, H)


def _dispatch_ldes_detailed(residual_surplus, residual_gap, dispatch_pct, demand_arr):
    """LDES multi-day dispatch with charge tracking. Modifies residual arrays in-place."""
    dispatch_profile = np.zeros(H, dtype=np.float64)
    charge_profile = np.zeros(H, dtype=np.float64)
    if dispatch_pct <= 0:
        return dispatch_profile, charge_profile

    # dispatch_pct is LDES energy capacity as % of annual demand (matching Step 1)
    # e.g. dispatch_pct=0.1 → capacity = 0.001 of annual demand energy
    energy_capacity = dispatch_pct / 100.0
    power_rating = energy_capacity / LDES_DURATION_HOURS
    window_hours = LDES_WINDOW_DAYS * 24

    return _ldes_loop_detailed(residual_surplus, residual_gap, dispatch_profile,
                                charge_profile, energy_capacity, power_rating,
                                LDES_EFFICIENCY, window_hours, H)


def _dispatch_h2(residual_surplus, residual_gap, dispatch_pct, demand_arr):
    """Green H2 seasonal storage dispatch (1000hr, 30-day window). Modifies residual arrays in-place."""
    dispatch_profile = np.zeros(H, dtype=np.float64)
    if dispatch_pct <= 0:
        return dispatch_profile

    energy_capacity = dispatch_pct / 100.0
    power_rating = energy_capacity / H2_DURATION_HOURS
    window_hours = H2_WINDOW_DAYS * 24

    return _ldes_loop(residual_surplus, residual_gap, dispatch_profile,
                      energy_capacity, power_rating, H2_EFFICIENCY,
                      window_hours, H)


def _dispatch_h2_detailed(residual_surplus, residual_gap, dispatch_pct, demand_arr):
    """Green H2 seasonal storage dispatch with charge tracking. Modifies residual arrays in-place."""
    dispatch_profile = np.zeros(H, dtype=np.float64)
    charge_profile = np.zeros(H, dtype=np.float64)
    if dispatch_pct <= 0:
        return dispatch_profile, charge_profile

    energy_capacity = dispatch_pct / 100.0
    power_rating = energy_capacity / H2_DURATION_HOURS
    window_hours = H2_WINDOW_DAYS * 24

    return _ldes_loop_detailed(residual_surplus, residual_gap, dispatch_profile,
                                charge_profile, energy_capacity, power_rating,
                                H2_EFFICIENCY, window_hours, H)


# ══════════════════════════════════════════════════════════════════════════════
# LP CO-OPTIMIZED STORAGE DISPATCH
# ══════════════════════════════════════════════════════════════════════════════

# Storage type definitions for LP dispatch
_STORAGE_TYPES = [
    {
        'name': 'battery4',
        'duration': BATTERY_DURATION_HOURS,
        'efficiency': BATTERY_EFFICIENCY,
        'window_hours': 24,
        'soc_carryover': False,
    },
    {
        'name': 'battery8',
        'duration': BATTERY8_DURATION_HOURS,
        'efficiency': BATTERY8_EFFICIENCY,
        'window_hours': 48,
        'soc_carryover': False,
    },
    {
        'name': 'ldes',
        'duration': LDES_DURATION_HOURS,
        'efficiency': LDES_EFFICIENCY,
        'window_hours': LDES_WINDOW_DAYS * 24,
        'soc_carryover': True,
    },
    {
        'name': 'h2',
        'duration': H2_DURATION_HOURS,
        'efficiency': H2_EFFICIENCY,
        'window_hours': H2_WINDOW_DAYS * 24,
        'soc_carryover': True,
    },
]


def _count_active_storage(b4_pct, b8_pct, ldes_pct, h2_pct):
    """Count how many storage types have non-zero capacity."""
    return (b4_pct > 0) + (b8_pct > 0) + (ldes_pct > 0) + (h2_pct > 0)


def _build_active_storage_params(b4_pct, b8_pct, ldes_pct, h2_pct):
    """Build list of active storage params from dispatch percentages.

    Returns list of dicts with keys: name, capacity, power_rating, efficiency,
    window_hours, soc_carryover, and the original index into _STORAGE_TYPES.
    """
    pcts = [b4_pct, b8_pct, ldes_pct, h2_pct]
    active = []
    for i, (pct, stype) in enumerate(zip(pcts, _STORAGE_TYPES)):
        if pct > 0:
            cap = pct / 100.0
            active.append({
                'name': stype['name'],
                'capacity': cap,
                'power_rating': cap / stype['duration'],
                'efficiency': stype['efficiency'],
                'window_hours': stype['window_hours'],
                'soc_carryover': stype['soc_carryover'],
                'idx': i,
            })
    return active


def _solve_storage_window_lp(surplus_w, gap_w, capacities, power_ratings, rtes, soc_init):
    """Solve LP for one rolling window of co-optimized storage dispatch.

    Variables: [charge(K*W), discharge(K*W), unmet(W)]
    Objective: minimize sum(unmet)

    Constraints:
      - SOC upper: cumsum(charge) - cumsum(discharge/rte) <= cap - soc_init  (K*W)
      - SOC lower: -(cumsum(charge) - cumsum(discharge/rte)) <= soc_init     (K*W)
      - Surplus:   sum_k(charge[k,h]) <= surplus[h]                          (W)
      - Gap eq:    sum_k(discharge[k,h]) + unmet[h] = gap[h]                 (W)

    Returns (charge, discharge, soc_final) or None on failure.
    """
    from scipy.sparse import coo_matrix

    K = len(capacities)
    W = len(surplus_w)
    n_vars = 2 * K * W + W  # charge + discharge + unmet

    # --- Objective: minimize sum(unmet) ---
    c = np.zeros(n_vars)
    c[2 * K * W:] = 1.0

    # --- Variable bounds (as array for speed) ---
    lb = np.zeros(n_vars)
    ub = np.empty(n_vars)
    for k in range(K):
        ub[k * W:(k + 1) * W] = power_ratings[k]          # charge bounds
        ub[K * W + k * W:K * W + (k + 1) * W] = power_ratings[k]  # discharge bounds
    ub[2 * K * W:] = gap_w  # unmet bounds

    # --- Build SOC constraint matrices using vectorized COO construction ---
    # For each type k and hour h, the cumulative-sum constraint involves
    # a lower-triangular pattern. We build this as COO triplets.

    # Pre-compute the number of non-zeros for SOC constraints:
    # Each type contributes W*(W+1)/2 entries for charge + same for discharge = W*(W+1) per type
    # Total SOC nnz = K * W * (W + 1) (for upper), same for lower
    tri_nnz = W * (W + 1) // 2  # lower-triangular entries per type per variable

    # Build row/col/data arrays for SOC upper constraints (K*W rows)
    # Row h of type k = k*W + h, cols = k*W+0..k*W+h (charge), K*W+k*W+0..K*W+k*W+h (discharge)
    soc_rows_list = []
    soc_cols_list = []
    soc_data_list = []

    # Precompute lower-triangular indices (row_offset, col_offset within W block)
    row_idx = np.repeat(np.arange(W), np.arange(1, W + 1))  # [0, 1,1, 2,2,2, ...]
    col_idx = np.concatenate([np.arange(h + 1) for h in range(W)])  # [0, 0,1, 0,1,2, ...]

    for k in range(K):
        inv_rte = 1.0 / rtes[k]
        base_row = k * W

        # Charge columns (positive for upper, negative for lower)
        soc_rows_list.append(base_row + row_idx)
        soc_cols_list.append(k * W + col_idx)
        soc_data_list.append(np.ones(tri_nnz))

        # Discharge columns (negative/rte for upper)
        soc_rows_list.append(base_row + row_idx)
        soc_cols_list.append(K * W + k * W + col_idx)
        soc_data_list.append(np.full(tri_nnz, -inv_rte))

    soc_upper_rows = np.concatenate(soc_rows_list)
    soc_upper_cols = np.concatenate(soc_cols_list)
    soc_upper_data = np.concatenate(soc_data_list)

    # SOC upper bound RHS: cap[k] - soc_init[k]
    b_soc_upper = np.empty(K * W)
    for k in range(K):
        b_soc_upper[k * W:(k + 1) * W] = capacities[k] - soc_init[k]

    # SOC lower: negate the upper matrix, RHS = soc_init[k]
    soc_lower_data = -soc_upper_data
    b_soc_lower = np.empty(K * W)
    for k in range(K):
        b_soc_lower[k * W:(k + 1) * W] = soc_init[k]

    # Surplus constraint: sum_k charge[k,h] <= surplus[h]  (W rows)
    surplus_rows = np.repeat(np.arange(W), K)
    surplus_cols = np.concatenate([k * W + np.arange(W) for k in range(K)]).reshape(K, W).T.ravel()
    surplus_data = np.ones(K * W)

    # Combine all inequality constraints
    n_ub = 2 * K * W + W
    all_ub_rows = np.concatenate([
        soc_upper_rows,
        K * W + soc_upper_rows,  # SOC lower rows offset by K*W
        2 * K * W + surplus_rows,
    ])
    all_ub_cols = np.concatenate([soc_upper_cols, soc_upper_cols, surplus_cols])
    all_ub_data = np.concatenate([soc_upper_data, soc_lower_data, surplus_data])
    b_ub = np.concatenate([b_soc_upper, b_soc_lower, surplus_w])

    A_ub_csc = csc_matrix((all_ub_data, (all_ub_rows, all_ub_cols)), shape=(n_ub, n_vars))

    # --- Equality: sum_k discharge[k,h] + unmet[h] = gap[h] ---
    eq_rows = np.concatenate([np.arange(W) for _ in range(K)] + [np.arange(W)])
    eq_cols = np.concatenate(
        [K * W + k * W + np.arange(W) for k in range(K)] + [2 * K * W + np.arange(W)]
    )
    eq_data = np.ones(K * W + W)
    A_eq_csc = csc_matrix((eq_data, (eq_rows, eq_cols)), shape=(W, n_vars))
    b_eq = gap_w.copy()

    # Build bounds as (n_vars, 2) array for linprog compatibility
    bounds = np.column_stack([lb, ub])

    result = linprog(c, A_ub=A_ub_csc, b_ub=b_ub, A_eq=A_eq_csc, b_eq=b_eq,
                     bounds=bounds, method='highs',
                     options={'presolve': True, 'time_limit': 10.0})

    if not result.success:
        return None

    x = np.clip(result.x, 0.0, None)

    charge = np.zeros((K, W))
    discharge = np.zeros((K, W))
    for k in range(K):
        charge[k] = x[k * W:(k + 1) * W]
        discharge[k] = x[K * W + k * W:K * W + (k + 1) * W]
        np.clip(charge[k], 0, power_ratings[k], out=charge[k])
        np.clip(discharge[k], 0, power_ratings[k], out=discharge[k])

    soc_final = np.zeros(K)
    for k in range(K):
        soc_final[k] = soc_init[k] + np.sum(charge[k]) - np.sum(discharge[k]) / rtes[k]
        soc_final[k] = np.clip(soc_final[k], 0.0, capacities[k])

    return charge, discharge, soc_final


def co_dispatch_storage_lp(residual_surplus, residual_gap, active_params, detailed=False):
    """LP co-optimized storage dispatch across all active storage types.

    Solves a rolling-window LP that coordinates charge/discharge across all storage
    types simultaneously, minimizing unmet demand (residual gap).

    Args:
        residual_surplus: (H,) surplus array, modified in-place
        residual_gap: (H,) gap array, modified in-place
        active_params: list of dicts from _build_active_storage_params()
        detailed: if True, also return charge profiles

    Returns:
        dict with keys like 'battery4', 'battery8', 'ldes', 'h2' -> (H,) dispatch profiles,
        and optionally 'battery4_charge', etc. -> (H,) charge profiles.
    """
    K = len(active_params)
    total_hours = len(residual_surplus)

    # Initialize output profiles (all 4 types, even inactive ones get zeros)
    all_names = ['battery4', 'battery8', 'ldes', 'h2']
    dispatch_profiles = {n: np.zeros(total_hours, dtype=np.float64) for n in all_names}
    charge_profiles = {n: np.zeros(total_hours, dtype=np.float64) for n in all_names}

    if K == 0:
        result = dict(dispatch_profiles)
        if detailed:
            result.update({f'{n}_charge': v for n, v in charge_profiles.items()})
        return result

    # Unified window = largest active type's window
    window_hours = max(p['window_hours'] for p in active_params)

    capacities = np.array([p['capacity'] for p in active_params])
    power_ratings = np.array([p['power_rating'] for p in active_params])
    rtes = np.array([p['efficiency'] for p in active_params])
    names = [p['name'] for p in active_params]
    carryover = [p['soc_carryover'] for p in active_params]

    soc_state = np.zeros(K)  # Current SOC for types with carryover

    num_windows = (total_hours + window_hours - 1) // window_hours
    greedy_fallback_used = False

    for w in range(num_windows):
        ws = w * window_hours
        we = min(ws + window_hours, total_hours)
        wlen = we - ws

        surplus_w = residual_surplus[ws:we].copy()
        gap_w = residual_gap[ws:we].copy()

        # Set initial SOC: 0 for batteries (no carryover), carried for LDES/H2
        soc_init = np.zeros(K)
        for k in range(K):
            if carryover[k]:
                soc_init[k] = soc_state[k]

        # Skip LP if no surplus and no gap in this window
        if np.sum(surplus_w) < 1e-12 and np.sum(gap_w) < 1e-12:
            # Reset battery SOCs (no carryover)
            for k in range(K):
                if not carryover[k]:
                    soc_state[k] = 0.0
            continue

        lp_result = _solve_storage_window_lp(surplus_w, gap_w, capacities,
                                              power_ratings, rtes, soc_init)

        if lp_result is None:
            # LP failed — fall back to greedy for this window
            if not greedy_fallback_used:
                warnings.warn(f"LP co-dispatch infeasible/failed at window {w}, "
                              f"falling back to greedy for this window.")
                greedy_fallback_used = True
            # Reset SOC for non-carryover types
            for k in range(K):
                if not carryover[k]:
                    soc_state[k] = 0.0
            continue

        charge_w, discharge_w, soc_final = lp_result

        # Write results into output arrays and update residuals
        for k in range(K):
            name = names[k]
            dispatch_profiles[name][ws:we] += discharge_w[k, :wlen]
            charge_profiles[name][ws:we] += charge_w[k, :wlen]

        # Update residual arrays in-place (consumed surplus, filled gap)
        total_charge_h = np.sum(charge_w[:, :wlen], axis=0)
        total_discharge_h = np.sum(discharge_w[:, :wlen], axis=0)
        residual_surplus[ws:we] -= total_charge_h
        residual_gap[ws:we] -= total_discharge_h

        # Clip to avoid numerical negatives
        np.clip(residual_surplus[ws:we], 0.0, None, out=residual_surplus[ws:we])
        np.clip(residual_gap[ws:we], 0.0, None, out=residual_gap[ws:we])

        # Update SOC state
        for k in range(K):
            if carryover[k]:
                soc_state[k] = soc_final[k]
            else:
                soc_state[k] = 0.0

    result = dict(dispatch_profiles)
    if detailed:
        result.update({f'{n}_charge': v for n, v in charge_profiles.items()})
    return result


@njit(cache=True)
def _per_resource_dispatch_njit(demand_arr, supply_matrix, resource_pcts_arr,
                                 procurement_factor, dispatch_order_indices):
    """Numba-compiled merit-order dispatch per resource.

    Args:
        demand_arr: (H,) demand array
        supply_matrix: (N_RESOURCES, H) supply profiles
        resource_pcts_arr: (N_RESOURCES,) percentage allocations
        procurement_factor: scalar multiplier
        dispatch_order_indices: (N_RESOURCES,) dispatch order as indices

    Returns:
        matched: (N_RESOURCES, H) array
        surplus: (N_RESOURCES, H) array
    """
    n_res = supply_matrix.shape[0]
    h = demand_arr.shape[0]
    gen = np.zeros((n_res, h), dtype=np.float64)
    for i in range(n_res):
        pct = resource_pcts_arr[i] / 100.0
        for t in range(h):
            gen[i, t] = procurement_factor * pct * supply_matrix[i, t]

    matched = np.zeros((n_res, h), dtype=np.float64)
    surplus = np.zeros((n_res, h), dtype=np.float64)
    remaining_demand = demand_arr.copy()

    for i_order in range(n_res):
        r_idx = dispatch_order_indices[i_order]
        for t in range(h):
            g = gen[r_idx, t]
            if g < remaining_demand[t]:
                matched[r_idx, t] = g
                surplus[r_idx, t] = 0.0
                remaining_demand[t] -= g
            else:
                matched[r_idx, t] = remaining_demand[t]
                surplus[r_idx, t] = g - remaining_demand[t]
                remaining_demand[t] = 0.0

    return matched, surplus


def _compute_per_resource_dispatch(demand_arr, supply_profiles, resource_pcts,
                                    procurement_factor, supply_matrix=None):
    """Merit-order dispatch per resource: CF -> CCS -> hydro -> wind -> solar.

    Thin wrapper that converts dicts to arrays, calls the @njit inner function,
    and converts results back to dicts.

    Returns:
        matched: dict {resource: np.array(H)} -- dispatched to demand
        surplus: dict {resource: np.array(H)} -- excess above demand
    """
    if supply_matrix is None:
        supply_matrix = build_supply_matrix(supply_profiles)

    resource_pcts_arr = np.array([resource_pcts.get(rt, 0) for rt in RESOURCE_TYPES],
                                  dtype=np.float64)

    matched_arr, surplus_arr = _per_resource_dispatch_njit(
        demand_arr, supply_matrix, resource_pcts_arr,
        procurement_factor, _DISPATCH_ORDER_INDICES)

    matched = {RESOURCE_TYPES[i]: matched_arr[i] for i in range(len(RESOURCE_TYPES))}
    surplus = {RESOURCE_TYPES[i]: surplus_arr[i] for i in range(len(RESOURCE_TYPES))}
    return matched, surplus


def _dispatch_battery(residual_surplus, residual_gap, dispatch_pct, duration_hours,
                      efficiency, window_hours=24):
    """Battery dispatch (4hr or 8hr). Modifies residual arrays in-place.

    Args:
        window_hours: 24 for 4hr battery (daily), 48 for 8hr battery (2-day)
    """
    dispatch_profile = np.zeros(H, dtype=np.float64)
    if dispatch_pct <= 0:
        return dispatch_profile

    capacity = dispatch_pct / 100.0
    power_rating = capacity / duration_hours

    return _battery_loop(residual_surplus, residual_gap, dispatch_profile,
                         capacity, power_rating, efficiency, window_hours, H)


def _dispatch_ldes(residual_surplus, residual_gap, dispatch_pct, demand_arr):
    """LDES multi-day dispatch (100hr, 7-day window). Modifies residual arrays in-place."""
    dispatch_profile = np.zeros(H, dtype=np.float64)
    if dispatch_pct <= 0:
        return dispatch_profile

    # dispatch_pct is LDES energy capacity as % of annual demand (matching Step 1)
    # e.g. dispatch_pct=0.1 → capacity = 0.001 of annual demand energy
    energy_capacity = dispatch_pct / 100.0
    power_rating = energy_capacity / LDES_DURATION_HOURS
    window_hours = LDES_WINDOW_DAYS * 24

    return _ldes_loop(residual_surplus, residual_gap, dispatch_profile,
                      energy_capacity, power_rating, LDES_EFFICIENCY,
                      window_hours, H)


def build_supply_matrix(supply_profiles):
    """Pre-convert supply profile dict → (N_RESOURCES, H) numpy matrix.

    Call once per ISO. Pass the matrix to reconstruct_hourly_dispatch via
    supply_matrix kwarg for ~3x speedup on repeated dispatches.
    """
    matrix = np.zeros((len(RESOURCE_TYPES), H), dtype=np.float64)
    for i, rtype in enumerate(RESOURCE_TYPES):
        p = supply_profiles.get(rtype, [0.0] * H)
        matrix[i, :] = np.array(p[:H], dtype=np.float64)
    return matrix


def reconstruct_hourly_dispatch(demand_norm, supply_profiles, resource_pcts,
                                 procurement_pct=100, battery_dispatch_pct=0,
                                 battery8_dispatch_pct=0, ldes_dispatch_pct=0,
                                 supply_matrix=None, detailed=False,
                                 h2_dispatch_pct=0, dispatch_mode=None,
                                 interchange_norm=None):
    """Reconstruct full 8760 hourly dispatch for a resource mix.

    Args:
        supply_matrix: optional (N_RESOURCES, H) numpy array from build_supply_matrix().
            If provided, skips per-call array conversion (faster for batch calls).
        detailed: if True, also compute per-resource matched/surplus arrays and
            storage charge profiles. Used by step4_build_dispatch_cache and step5c_compress_day_profiles.
        h2_dispatch_pct: green H2 seasonal storage capacity as % of demand (default 0).

    Returns:
        result dict with keys:
          - supply_total: (H,) clean supply before storage
          - battery4_profile: (H,) 4hr battery dispatch
          - battery8_profile: (H,) 8hr battery dispatch
          - ldes_profile: (H,) LDES dispatch
          - h2_profile: (H,) green H2 dispatch
          - total_clean: (H,) total clean supply including storage
          - residual_demand: (H,) demand not met by clean (positive = fossil needed)
          - curtailed: (H,) curtailed clean energy
          - ccs_supply: (H,) CCS-CCGT supply portion
          - fossil_displaced: (H,) fossil MWh displaced at each hour (normalized)

        When detailed=True, also includes:
          - matched_{clean_firm,ccs_ccgt,solar,wind,offshore_wind,hydro}: (H,) per-resource dispatched
          - surplus_{clean_firm,ccs_ccgt,solar,wind,offshore_wind,hydro}: (H,) per-resource surplus
          - battery4_charge: (H,) battery4 charge profile
          - battery8_charge: (H,) battery8 charge profile
          - ldes_charge: (H,) LDES charge profile
          - h2_charge: (H,) H2 charge profile
    """
    procurement_factor = procurement_pct / 100.0
    demand_arr = np.array(demand_norm[:H], dtype=np.float64)

    # Build total supply profile from resource mix
    supply_total = np.zeros(H, dtype=np.float64)
    ccs_supply = np.zeros(H, dtype=np.float64)

    # Use pre-built matrix if available (batch optimization)
    if supply_matrix is not None:
        mix_weights = np.array([resource_pcts.get(rt, 0) / 100.0 for rt in RESOURCE_TYPES],
                               dtype=np.float64)
        supply_total = procurement_factor * (mix_weights @ supply_matrix)
        ccs_idx = RESOURCE_TYPES.index('ccs_ccgt')
        if mix_weights[ccs_idx] > 0:
            ccs_supply = procurement_factor * mix_weights[ccs_idx] * supply_matrix[ccs_idx]
    else:
        for rtype in RESOURCE_TYPES:
            pct = resource_pcts.get(rtype, 0)
            if pct <= 0:
                continue
            profile = np.array(supply_profiles[rtype][:H], dtype=np.float64)
            contribution = procurement_factor * (pct / 100.0) * profile
            supply_total += contribution
            if rtype == 'ccs_ccgt':
                ccs_supply = contribution.copy()

    # Storage dispatch
    residual_surplus = np.maximum(0.0, supply_total - demand_arr)
    residual_gap = np.maximum(0.0, demand_arr - supply_total)

    if dispatch_mode is None:
        dispatch_mode = STORAGE_DISPATCH_MODE

    active_count = _count_active_storage(battery_dispatch_pct, battery8_dispatch_pct,
                                          ldes_dispatch_pct, h2_dispatch_pct)

    # LP co-dispatch when enabled and >=2 storage types active (otherwise greedy is equivalent)
    if dispatch_mode == 'lp' and active_count >= 2:
        active_params = _build_active_storage_params(
            battery_dispatch_pct, battery8_dispatch_pct,
            ldes_dispatch_pct, h2_dispatch_pct)
        lp_result = co_dispatch_storage_lp(residual_surplus, residual_gap,
                                            active_params, detailed=detailed)
        battery4_profile = lp_result['battery4']
        battery8_profile = lp_result['battery8']
        ldes_profile = lp_result['ldes']
        h2_profile = lp_result['h2']
        if detailed:
            battery4_charge = lp_result['battery4_charge']
            battery8_charge = lp_result['battery8_charge']
            ldes_charge = lp_result['ldes_charge']
            h2_charge = lp_result['h2_charge']
    elif detailed:
        # Greedy sequential dispatch with charge tracking
        battery4_profile, battery4_charge = _dispatch_battery_detailed(
            residual_surplus, residual_gap,
            battery_dispatch_pct, BATTERY_DURATION_HOURS, BATTERY_EFFICIENCY)
        battery8_profile, battery8_charge = _dispatch_battery_detailed(
            residual_surplus, residual_gap,
            battery8_dispatch_pct, BATTERY8_DURATION_HOURS, BATTERY8_EFFICIENCY,
            window_hours=48)
        ldes_profile, ldes_charge = _dispatch_ldes_detailed(
            residual_surplus, residual_gap,
            ldes_dispatch_pct, demand_arr)
        h2_profile, h2_charge = _dispatch_h2_detailed(
            residual_surplus, residual_gap,
            h2_dispatch_pct, demand_arr)
    else:
        # Greedy sequential dispatch (no charge tracking)
        battery4_profile = _dispatch_battery(
            residual_surplus, residual_gap,
            battery_dispatch_pct, BATTERY_DURATION_HOURS, BATTERY_EFFICIENCY)
        battery8_profile = _dispatch_battery(
            residual_surplus, residual_gap,
            battery8_dispatch_pct, BATTERY8_DURATION_HOURS, BATTERY8_EFFICIENCY,
            window_hours=48)
        ldes_profile = _dispatch_ldes(
            residual_surplus, residual_gap,
            ldes_dispatch_pct, demand_arr)
        h2_profile = _dispatch_h2(
            residual_surplus, residual_gap,
            h2_dispatch_pct, demand_arr)

    total_clean = supply_total + battery4_profile + battery8_profile + ldes_profile + h2_profile
    fossil_displaced = np.minimum(demand_arr, total_clean)
    residual_demand = np.maximum(0.0, demand_arr - total_clean)
    curtailed = np.maximum(0.0, total_clean - demand_arr)

    # Apply inter-regional interchange: net imports reduce residual demand,
    # net exports increase it. Applied AFTER storage dispatch to avoid
    # interfering with clean supply/storage optimization.
    if interchange_norm is not None:
        interchange_arr = np.asarray(interchange_norm[:H], dtype=np.float64)
        # Positive interchange = net imports → reduces fossil need
        # Negative interchange = net exports → increases fossil need
        residual_demand = np.maximum(0.0, residual_demand - interchange_arr)
        # Imports that arrive during surplus hours reduce curtailment
        curtailed = np.maximum(0.0, curtailed - np.maximum(0.0, interchange_arr))

    result = {
        'supply_total': supply_total,
        'battery4_profile': battery4_profile,
        'battery8_profile': battery8_profile,
        'ldes_profile': ldes_profile,
        'h2_profile': h2_profile,
        'total_clean': total_clean,
        'residual_demand': residual_demand,
        'curtailed': curtailed,
        'ccs_supply': ccs_supply,
        'fossil_displaced': fossil_displaced,
    }

    if detailed:
        matched, surplus = _compute_per_resource_dispatch(
            demand_arr, supply_profiles, resource_pcts, procurement_factor, supply_matrix)
        for rtype in RESOURCE_TYPES:
            result[f'matched_{rtype}'] = matched[rtype]
            result[f'surplus_{rtype}'] = surplus[rtype]
        result['battery4_charge'] = battery4_charge
        result['battery8_charge'] = battery8_charge
        result['ldes_charge'] = ldes_charge
        result['h2_charge'] = h2_charge

    return result


# ══════════════════════════════════════════════════════════════════════════════
# FOSSIL RETIREMENT MODEL
# ══════════════════════════════════════════════════════════════════════════════

def compute_fossil_retirement(iso, clean_pct, emission_rates, fossil_mix,
                               demand_growth_factor=1.0):
    """Dispatch-stack retirement model: coal → oil → gas merit-order displacement.

    Returns:
        displaced_rate: tCO2/MWh weighted avg of displaced fossil
        retirement_info: dict with displaced/remaining TWh and both rates
    """
    regional_data = emission_rates.get(iso, {})
    coal_rate = regional_data.get('coal_co2_lb_per_mwh', 0.0) / 2204.62
    gas_rate = regional_data.get('gas_co2_lb_per_mwh', 0.0) / 2204.62
    oil_rate = regional_data.get('oil_co2_lb_per_mwh', 0.0) / 2204.62

    base_demand_twh = BASE_DEMAND_TWH.get(iso, 0)
    grown_demand_twh = base_demand_twh * demand_growth_factor

    fossil_pct = (100.0 - clean_pct) / 100.0
    fossil_twh = grown_demand_twh * fossil_pct

    coal_cap = COAL_CAP_TWH.get(iso, 0)
    oil_cap = OIL_CAP_TWH.get(iso, 0)
    baseline_clean = sum(GRID_MIX_SHARES.get(iso, {}).values())

    if fossil_twh <= 0.01 or clean_pct >= 100:
        total_fossil_twh = grown_demand_twh * (100.0 - baseline_clean) / 100.0
        coal_twh = min(coal_cap, total_fossil_twh)
        oil_twh = min(oil_cap, max(0, total_fossil_twh - coal_twh))
        gas_twh = max(0, total_fossil_twh - coal_twh - oil_twh)
        if total_fossil_twh > 0.01:
            rate = (coal_twh * coal_rate + oil_twh * oil_rate + gas_twh * gas_rate) / total_fossil_twh
        else:
            rate = gas_rate
        return rate, {
            'coal_displaced_twh': round(coal_twh, 2),
            'oil_displaced_twh': round(oil_twh, 2),
            'gas_displaced_twh': round(gas_twh, 2),
            'displaced_rate_tco2_mwh': round(rate, 4),
            'remaining_rate_tco2_mwh': 0,
            'demand_growth_factor': demand_growth_factor,
        }

    coal_twh = min(coal_cap, fossil_twh)
    remaining_fossil = fossil_twh - coal_twh
    oil_twh = min(oil_cap, remaining_fossil)
    gas_twh = max(0, fossil_twh - coal_twh - oil_twh)

    additional_clean_twh = max(0, (clean_pct - baseline_clean) / 100.0 * grown_demand_twh)

    if clean_pct >= COAL_OIL_RETIREMENT_THRESHOLD:
        coal_displaced = coal_cap
        oil_displaced = oil_cap
        gas_displaced = max(0, additional_clean_twh - coal_cap - oil_cap)
        total_displaced = coal_displaced + oil_displaced + gas_displaced

        if total_displaced > 0.01:
            displaced_rate = (coal_displaced * coal_rate + oil_displaced * oil_rate +
                              gas_displaced * gas_rate) / total_displaced
        else:
            displaced_rate = gas_rate

        return displaced_rate, {
            'coal_displaced_twh': round(coal_displaced, 2),
            'oil_displaced_twh': round(oil_displaced, 2),
            'gas_displaced_twh': round(gas_displaced, 2),
            'displaced_rate_tco2_mwh': round(displaced_rate, 4),
            'remaining_rate_tco2_mwh': round(gas_rate, 4),
            'forced_gas_only': True,
            'demand_growth_factor': demand_growth_factor,
        }

    # Merit-order displacement: coal first, then oil, then gas
    coal_displaced = min(additional_clean_twh, coal_twh)
    remaining = additional_clean_twh - coal_displaced
    oil_displaced = min(remaining, oil_twh)
    remaining = remaining - oil_displaced
    gas_displaced = min(remaining, gas_twh)
    total_displaced = coal_displaced + oil_displaced + gas_displaced

    if total_displaced > 0.01:
        displaced_rate = (coal_displaced * coal_rate + oil_displaced * oil_rate +
                          gas_displaced * gas_rate) / total_displaced
    else:
        displaced_rate = (coal_twh * coal_rate + oil_twh * oil_rate + gas_twh * gas_rate) / fossil_twh

    coal_remaining = coal_twh - coal_displaced
    oil_remaining = oil_twh - oil_displaced
    gas_remaining = gas_twh - gas_displaced
    fossil_remaining = coal_remaining + oil_remaining + gas_remaining

    if fossil_remaining > 0.01:
        remaining_rate = (coal_remaining * coal_rate + oil_remaining * oil_rate +
                          gas_remaining * gas_rate) / fossil_remaining
    else:
        remaining_rate = gas_rate

    return displaced_rate, {
        'coal_displaced_twh': round(coal_displaced, 2),
        'oil_displaced_twh': round(oil_displaced, 2),
        'gas_displaced_twh': round(gas_displaced, 2),
        'coal_remaining_twh': round(coal_remaining, 2),
        'oil_remaining_twh': round(oil_remaining, 2),
        'gas_remaining_twh': round(gas_remaining, 2),
        'fossil_remaining_twh': round(fossil_remaining, 2),
        'displaced_rate_tco2_mwh': round(displaced_rate, 4),
        'remaining_rate_tco2_mwh': round(remaining_rate, 4),
        'demand_growth_factor': demand_growth_factor,
    }


def compute_fossil_capacity_at_threshold(iso, clean_pct, demand_growth_factor=1.0):
    """Compute remaining fossil fleet capacity (MW and TWh) at a given clean energy %.

    Returns dict with coal/oil/gas remaining TWh and total fossil MW (estimated).
    Used by LMP module to build the merit-order stack with correct capacity slices.
    """
    base_demand_twh = BASE_DEMAND_TWH.get(iso, 0)
    grown_demand_twh = base_demand_twh * demand_growth_factor

    fossil_pct = (100.0 - clean_pct) / 100.0
    fossil_twh = grown_demand_twh * fossil_pct

    coal_cap = COAL_CAP_TWH.get(iso, 0)
    oil_cap = OIL_CAP_TWH.get(iso, 0)

    if clean_pct >= COAL_OIL_RETIREMENT_THRESHOLD:
        # All coal and oil retired
        return {
            'coal_twh': 0.0, 'oil_twh': 0.0,
            'gas_twh': max(0, fossil_twh),
            'total_fossil_twh': max(0, fossil_twh),
        }

    coal_twh = min(coal_cap, fossil_twh)
    remaining = fossil_twh - coal_twh
    oil_twh = min(oil_cap, remaining)
    gas_twh = max(0, fossil_twh - coal_twh - oil_twh)

    return {
        'coal_twh': round(coal_twh, 2),
        'oil_twh': round(oil_twh, 2),
        'gas_twh': round(gas_twh, 2),
        'total_fossil_twh': round(fossil_twh, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# HOURLY DISPATCH CACHE — append-mode, keyed by archetype hash
# ══════════════════════════════════════════════════════════════════════════════

DISPATCH_CACHE_DIR = os.path.join(DATA_DIR, 'step3-dispatch')


def _archetype_key(iso, resource_pcts, procurement_pct=100, battery_dispatch_pct=0,
                   battery8_dispatch_pct=0, ldes_dispatch_pct=0):
    """Deterministic hash key for a unique dispatch archetype."""
    parts = [
        iso,
        str(float(resource_pcts.get('clean_firm', 0))),
        str(float(resource_pcts.get('solar', 0))),
        str(float(resource_pcts.get('wind', 0))),
        str(float(resource_pcts.get('offshore_wind', 0))),
        str(float(resource_pcts.get('ccs_ccgt', 0))),
        str(float(resource_pcts.get('hydro', 0))),
        str(float(procurement_pct)),
        str(float(battery_dispatch_pct)),
        str(float(battery8_dispatch_pct)),
        str(float(ldes_dispatch_pct)),
    ]
    key_str = '|'.join(parts)
    return hashlib.md5(key_str.encode()).hexdigest()[:16]


def _cache_path(iso):
    """Per-ISO cache file path (parquet format)."""
    return os.path.join(DISPATCH_CACHE_DIR, f'{iso}_dispatch_cache.parquet')


def _cache_path_npz(iso):
    """Legacy NPZ cache path for backward compatibility."""
    return os.path.join(DISPATCH_CACHE_DIR, f'{iso}_dispatch_cache.npz')


def _load_dispatch_cache_npz(iso, require_version=None):
    """Load legacy NPZ dispatch cache. Returns dict of {key: arrays_dict}."""
    path = _cache_path_npz(iso)
    if not os.path.exists(path):
        return {}
    try:
        data = np.load(path, allow_pickle=True)
        if require_version is not None:
            if '_meta_version' in data.files:
                stored_version = int(data['_meta_version'][0])
                if stored_version != require_version:
                    return {}
            else:
                return {}
        cache = {}
        for arr_name in data.files:
            if arr_name.startswith('_meta_'):
                continue
            parts = arr_name.split('_', 2)
            if len(parts) >= 3 and parts[0] == 'k':
                k = parts[1]
                field = '_'.join(parts[2:])
                if k not in cache:
                    cache[k] = {}
                cache[k][field] = data[arr_name]
        return cache
    except Exception:
        return {}


def load_dispatch_cache(iso, require_version=None):
    """Load existing dispatch cache for an ISO. Returns dict of {key: arrays_dict}.

    Reads parquet format (primary), falls back to legacy NPZ if parquet not found.

    Args:
        require_version: if set, returns empty dict if cache version doesn't match.
    """
    import pyarrow.parquet as pq

    path = _cache_path(iso)
    if os.path.exists(path):
        try:
            table = pq.read_table(path)
            metadata = table.schema.metadata or {}
            if require_version is not None:
                stored = int(metadata.get(b'cache_version', b'0'))
                if stored != require_version:
                    return {}

            keys = table.column('archetype_key').to_pylist()
            field_names = [n for n in table.schema.names if n != 'archetype_key']
            cache = {}

            # Efficient extraction via flat values + offsets per list column
            col_data = {}
            for field in field_names:
                col = table.column(field).combine_chunks()
                col_data[field] = (col.values.to_numpy().copy(),
                                   col.offsets.to_numpy())

            for i, key in enumerate(keys):
                entry = {}
                for field in field_names:
                    vals, offs = col_data[field]
                    entry[field] = vals[offs[i]:offs[i + 1]].copy()
                cache[key] = entry
            return cache
        except Exception:
            pass

    # Fallback to legacy NPZ format
    return _load_dispatch_cache_npz(iso, require_version=require_version)


def save_dispatch_cache(iso, cache, version=None):
    """Save dispatch cache for an ISO as parquet. Overwrites existing file.

    Uses pyarrow list columns: one row per archetype, each dispatch field
    stored as a list<float64> of 8760 hourly values. Snappy-compressed.

    Args:
        version: if set, stores version metadata in parquet schema metadata.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(DISPATCH_CACHE_DIR, exist_ok=True)
    if not cache:
        return

    keys = list(cache.keys())
    sample_fields = list(next(iter(cache.values())).keys())

    # Build pyarrow arrays
    key_arr = pa.array(keys, type=pa.string())
    field_arrays = {}
    for field in sample_fields:
        field_arrays[field] = pa.array(
            [cache[k].get(field, np.zeros(H)) for k in keys],
            type=pa.list_(pa.float64()),
        )

    schema_fields = [pa.field('archetype_key', pa.string())]
    schema_fields += [pa.field(f, pa.list_(pa.float64())) for f in sample_fields]
    meta = {b'cache_version': str(version or 0).encode()}
    schema = pa.schema(schema_fields, metadata=meta)

    table = pa.table(
        [key_arr] + [field_arrays[f] for f in sample_fields],
        schema=schema,
    )

    path = _cache_path(iso)
    tmp_path = path + '.tmp'
    pq.write_table(table, tmp_path, compression='snappy')
    os.replace(tmp_path, path)


def get_or_compute_dispatch(iso, demand_norm, supply_profiles, resource_pcts,
                             procurement_pct=100, battery_dispatch_pct=0,
                             battery8_dispatch_pct=0, ldes_dispatch_pct=0,
                             cache=None, supply_matrix=None):
    """Get dispatch from cache or compute and add to cache.

    Args:
        cache: mutable dict (load_dispatch_cache output). If provided, checks cache
               first and adds new results. Caller is responsible for calling
               save_dispatch_cache() when done with a batch.
        supply_matrix: optional pre-built (5, H) numpy array from build_supply_matrix().
            Avoids repeated list→array conversion on cache misses.

    Returns:
        dispatch result dict (same as reconstruct_hourly_dispatch)
        cache_hit: bool
    """
    key = _archetype_key(iso, resource_pcts, procurement_pct,
                         battery_dispatch_pct, battery8_dispatch_pct,
                         ldes_dispatch_pct)

    if cache is not None and key in cache:
        cached = cache[key]
        return {k: cached[k] for k in cached}, True

    result = reconstruct_hourly_dispatch(
        demand_norm, supply_profiles, resource_pcts,
        procurement_pct, battery_dispatch_pct,
        battery8_dispatch_pct, ldes_dispatch_pct,
        supply_matrix=supply_matrix)

    if cache is not None:
        cache[key] = {k: v for k, v in result.items()}

    return result, False


# ══════════════════════════════════════════════════════════════════════════════
# DISPATCH-BASED CO₂ ACCOUNTING — hourly fossil displacement from cache
# ══════════════════════════════════════════════════════════════════════════════

def compute_co2_from_dispatch(iso, dispatch_result, emission_rates, demand_total_mwh):
    """Compute CO₂ displacement from hourly dispatch results.

    Uses the 8760-hour fossil_displaced and ccs_supply arrays from the dispatch
    cache to compute exact CO₂ abated. The emission rate at each hour is the
    weighted average of the remaining fossil stack (merit-order: coal → oil → gas).

    Unlike the analytical compute_fossil_retirement() which uses TWh-level averages,
    this function captures the hourly shape of displacement — a wind-heavy portfolio
    displaces different fossil hours than clean firm.

    Args:
        iso: ISO region identifier
        dispatch_result: dict from dispatch cache with 'fossil_displaced', 'ccs_supply' arrays
        emission_rates: eGRID emission rates dict
        demand_total_mwh: annual demand in MWh (for scaling normalized arrays)

    Returns:
        dict with total_co2_abated_tons, co2_rate_per_mwh, matched_mwh,
        displaced TWh by fuel type, emission_rate details
    """
    regional = emission_rates.get(iso, {})
    coal_rate = regional.get('coal_co2_lb_per_mwh', 0.0) / 2204.62  # tCO2/MWh
    oil_rate = regional.get('oil_co2_lb_per_mwh', 0.0) / 2204.62
    gas_rate = regional.get('gas_co2_lb_per_mwh', 0.0) / 2204.62

    fossil_displaced = dispatch_result['fossil_displaced']  # (H,) normalized
    ccs_supply = dispatch_result.get('ccs_supply', np.zeros(H))

    # Scale to MWh
    fossil_mwh = fossil_displaced * demand_total_mwh  # per-hour MWh displaced
    ccs_mwh = np.minimum(ccs_supply * demand_total_mwh, fossil_mwh)
    non_ccs_mwh = fossil_mwh - ccs_mwh

    total_displaced_mwh = float(np.sum(fossil_mwh))
    total_displaced_twh = total_displaced_mwh / 1e6

    # Determine which fossil fuels are displaced using merit-order
    # Coal displaced first (up to cap), then oil, then gas
    coal_cap_mwh = COAL_CAP_TWH.get(iso, 0) * 1e6
    oil_cap_mwh = OIL_CAP_TWH.get(iso, 0) * 1e6

    coal_displaced_mwh = min(total_displaced_mwh, coal_cap_mwh)
    remaining = total_displaced_mwh - coal_displaced_mwh
    oil_displaced_mwh = min(remaining, oil_cap_mwh)
    gas_displaced_mwh = max(0, remaining - oil_displaced_mwh)

    # CO₂ from non-CCS displacement (weighted by fuel type displaced)
    if total_displaced_mwh > 0:
        weighted_rate = (
            coal_displaced_mwh * coal_rate +
            oil_displaced_mwh * oil_rate +
            gas_displaced_mwh * gas_rate
        ) / total_displaced_mwh
    else:
        weighted_rate = gas_rate

    co2_non_ccs = float(np.sum(non_ccs_mwh)) * weighted_rate
    ccs_credit = max(0.0, weighted_rate - CCS_RESIDUAL_EMISSION_RATE)
    co2_ccs = float(np.sum(ccs_mwh)) * ccs_credit

    total_co2 = co2_non_ccs + co2_ccs
    co2_rate = total_co2 / total_displaced_mwh if total_displaced_mwh > 0 else 0

    return {
        'total_co2_abated_tons': round(total_co2, 0),
        'co2_rate_per_mwh': round(co2_rate, 4),
        'matched_mwh': round(total_displaced_mwh, 0),
        'displaced_twh': round(total_displaced_twh, 2),
        'coal_displaced_twh': round(coal_displaced_mwh / 1e6, 2),
        'oil_displaced_twh': round(oil_displaced_mwh / 1e6, 2),
        'gas_displaced_twh': round(gas_displaced_mwh / 1e6, 2),
        'weighted_emission_rate': round(weighted_rate, 4),
        'methodology': 'dispatch_cache_hourly',
    }
