#!/usr/bin/env python3
"""
Shared Dispatch Utilities — Single source of truth for hourly dispatch reconstruction.
======================================================================================
Extracted from step4_1a_fossil_dispatch.py to avoid duplicating dispatch logic between the CO₂
model and the LMP pricing module. Both import from here.

Provides:
  - Constants imported from pipeline_config (battery/LDES params, hydro caps,
    grid mix, coal/oil caps, base demand, cache version)
  - get_supply_profiles(iso, gen_profiles) — generation shape profiles
  - reconstruct_hourly_dispatch(mix, demand, profiles, ...) — full 4-phase
    storage dispatch: battery4 → battery8 → LDES → H2
  - compute_fossil_retirement(iso, clean_pct, ...) — remaining fossil capacity
  - load_common_data() — demand, gen profiles, emission rates, fossil mix
  - Hourly dispatch cache: save/load/append cached 8760 profiles per archetype

Dispatch model:
  All storage types carry SOC across window boundaries and apply round-trip
  efficiency per discharge event (SOC -= discharge / eff). This enables
  multi-day energy shifting while maintaining correct round-trip losses.
  Matches step1_pfs_generator.py dispatch algorithm exactly.

Cache version (DISPATCH_CACHE_VERSION) is imported from pipeline_config —
single source of truth. Downstream modules compare cache version to detect
stale caches that need rebuilding.
"""

import json
import os
import hashlib
import numpy as np

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

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

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
    COAL_PHASE_OUT_START, COAL_PHASE_OUT_END,
    CCS_CAPTURE_RATES, CCS_UNABATED_GAS_RATE,
    ANNOUNCED_COAL_RETIREMENTS, get_announced_coal_retired_gw,
    H,
    DISPATCH_CACHE_VERSION,
    # Dispatch-specific constants (migrated to pipeline_config)
    HYDRO_CAPS, COAL_CAP_TWH, OIL_CAP_TWH,
    NUCLEAR_SHARE_OF_CLEAN_FIRM, NUCLEAR_MONTHLY_CF,
)

DATA_YEAR = '2025'
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
    with open(os.path.join(DATA_DIR, 'egrid_emission_rates.json')) as f:
        emission_rates = json.load(f)
    fossil_mix = load_fossil_mix()
    return demand_data, gen_profiles, emission_rates, fossil_mix


def get_demand_profile(iso, demand_data):
    """Extract normalized 8760 demand profile and total MWh for an ISO."""
    iso_demand = demand_data.get(iso, {})
    year_demand = iso_demand.get(DATA_YEAR, iso_demand.get('2024', {}))
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

def get_supply_profiles(iso, gen_profiles):
    """Get generation shape profiles with nuclear seasonal derate.

    This is the authoritative version used by all pipeline stages.
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

    # Solar (with DST-aware nighttime correction)
    if iso == 'NYISO':
        p = gen_profiles.get(iso, {}).get(DATA_YEAR, gen_profiles.get(iso, {})).get('solar_proxy')
        if not p:
            neiso_data = gen_profiles.get('NEISO', {})
            neiso_year = neiso_data.get(DATA_YEAR, neiso_data)
            p = neiso_year.get('solar') if isinstance(neiso_year, dict) else None
        if not p:
            p = [0.0] * H
        solar_raw = list(p[:H])
    else:
        iso_data = gen_profiles.get(iso, {})
        year_data = iso_data.get(DATA_YEAR, iso_data)
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
    year_data = iso_data.get(DATA_YEAR, iso_data)
    if isinstance(year_data, dict):
        profiles['wind'] = year_data.get('wind', [0.0] * H)[:H]
    else:
        profiles['wind'] = [0.0] * H

    # Offshore wind (NYISO, NEISO, PJM, CAISO only — zeros for other ISOs)
    if iso in OFFSHORE_ISOS:
        iso_data_osw = gen_profiles.get(iso, {})
        year_data_osw = iso_data_osw.get(DATA_YEAR, iso_data_osw)
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

# Cache version — imported from pipeline_config (single source of truth).
# Aliased here for backward compatibility with code referencing CACHE_VERSION.
CACHE_VERSION = DISPATCH_CACHE_VERSION


@njit(cache=True)
def _battery_loop(residual_surplus, residual_gap, dispatch_profile,
                  capacity, power_rating, efficiency, window_hours, total_hours):
    """Inner loop for battery dispatch — sequential hour-by-hour, matching step1_pfs_generator.

    SOC carries across window boundaries so energy stored late in one window
    can be discharged early in the next. RTE is applied per discharge event
    (not once per window). This matches the LDES dispatch pattern.

    Args:
        capacity: energy capacity as fraction of annual demand
        power_rating: max charge/discharge power per hour (capacity / duration_hours)
        efficiency: round-trip efficiency
        window_hours: 24 for 4hr battery, 48 for 8hr battery
        total_hours: 8760
    """
    n_windows = (total_hours + window_hours - 1) // window_hours
    soc = 0.0
    for w in range(n_windows):
        ws = w * window_hours
        we = ws + window_hours
        if we > total_hours:
            we = total_hours
        # Charge pass: sequential hour-by-hour
        for h in range(ws, we):
            s = residual_surplus[h]
            if s > 0.0 and soc < capacity:
                charge = s
                if charge > power_rating:
                    charge = power_rating
                remaining = capacity - soc
                if charge > remaining:
                    charge = remaining
                soc += charge
                residual_surplus[h] -= charge
        # Discharge pass: sequential hour-by-hour
        for h in range(ws, we):
            g = residual_gap[h]
            if g > 0.0 and soc > 0.0:
                available = soc * efficiency
                discharge = g
                if discharge > power_rating:
                    discharge = power_rating
                if discharge > available:
                    discharge = available
                dispatch_profile[h] = discharge
                soc -= discharge / efficiency
                residual_gap[h] -= discharge
    return dispatch_profile


@njit(cache=True)
def _ldes_loop(residual_surplus, residual_gap, dispatch_profile,
               energy_capacity, power_rating, ldes_efficiency,
               window_hours, total_hours):
    """Inner loop for LDES/H2 multi-day dispatch — sequential hour-by-hour.

    Charge phase subtracts from residual_surplus to prevent double-counting
    between LDES and downstream storage (H2). Discharge subtracts from
    residual_gap so downstream phases (H2) see the post-LDES residual.
    SOC carries across window boundaries. RTE applied per discharge event.
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
    """Battery dispatch with charge tracking — sequential hour-by-hour, matching step1.
    SOC carries across window boundaries. RTE applied per discharge event."""
    n_windows = (total_hours + window_hours - 1) // window_hours
    soc = 0.0
    for w in range(n_windows):
        ws = w * window_hours
        we = ws + window_hours
        if we > total_hours:
            we = total_hours
        # Charge pass: sequential hour-by-hour
        for h in range(ws, we):
            s = residual_surplus[h]
            if s > 0.0 and soc < capacity:
                charge = s
                if charge > power_rating:
                    charge = power_rating
                remaining = capacity - soc
                if charge > remaining:
                    charge = remaining
                soc += charge
                residual_surplus[h] -= charge
                charge_profile[h] += charge
        # Discharge pass: sequential hour-by-hour
        for h in range(ws, we):
            g = residual_gap[h]
            if g > 0.0 and soc > 0.0:
                available = soc * efficiency
                discharge = g
                if discharge > power_rating:
                    discharge = power_rating
                if discharge > available:
                    discharge = available
                dispatch_profile[h] = discharge
                soc -= discharge / efficiency
                residual_gap[h] -= discharge
    return dispatch_profile, charge_profile


@njit(cache=True)
def _ldes_loop_detailed(residual_surplus, residual_gap, dispatch_profile,
                        charge_profile, energy_capacity, power_rating,
                        ldes_efficiency, window_hours, total_hours):
    """LDES/H2 dispatch with charge tracking — sequential hour-by-hour.
    SOC carries across window boundaries. RTE applied per discharge event.
    Discharge updates residual_gap for downstream storage phases."""
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
                                 h2_dispatch_pct=0):
    """Reconstruct full 8760 hourly dispatch for a resource mix.

    Args:
        supply_matrix: optional (N_RESOURCES, H) numpy array from build_supply_matrix().
            If provided, skips per-call array conversion (faster for batch calls).
        detailed: if True, also compute per-resource matched/surplus arrays and
            storage charge profiles. Used by step3a_build_dispatch_cache and step4_1b_compress_day_profiles.
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

    # Storage dispatch: battery4 → battery8 → LDES (sequential, each reduces residuals)
    residual_surplus = np.maximum(0.0, supply_total - demand_arr)
    residual_gap = np.maximum(0.0, demand_arr - supply_total)

    if detailed:
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
# SIGMOID COAL PHASE-OUT MODEL
# ══════════════════════════════════════════════════════════════════════════════

def coal_fraction_at_clean_pct(clean_pct):
    """Sigmoid coal phase-out: returns fraction of coal capacity remaining (0-1).

    Replaces the previous cliff model (100% → 0% at a single threshold).
    Uses a smooth quadratic ramp: coal declines from 100% at COAL_PHASE_OUT_START
    to 0% at COAL_PHASE_OUT_END.

    This better reflects real-world coal retirement dynamics driven by economics,
    regulation (EPA 111(d)), and plant age rather than a clean-energy threshold.
    Literature: Grubert 2020 (ES&T), Gillingham & Stock 2018 (JEP).
    """
    if clean_pct <= COAL_PHASE_OUT_START:
        return 1.0
    if clean_pct >= COAL_PHASE_OUT_END:
        return 0.0
    # Smooth quadratic: f(x) = 1 - ((x - start) / (end - start))^2
    # Concave: faster decline at beginning, slower at end (realistic: easy retirements first)
    t = (clean_pct - COAL_PHASE_OUT_START) / (COAL_PHASE_OUT_END - COAL_PHASE_OUT_START)
    return max(0.0, 1.0 - t * t)


# ══════════════════════════════════════════════════════════════════════════════
# FOSSIL RETIREMENT MODEL
# ══════════════════════════════════════════════════════════════════════════════

def compute_fossil_retirement(iso, clean_pct, emission_rates, fossil_mix,
                               demand_growth_factor=1.0, year=None):
    """Dispatch-stack retirement model: coal → oil → gas merit-order displacement.

    Hybrid model:
    - Near-term (through 2035): Uses EIA 860 announced retirement schedule to reduce
      coal capacity before applying merit-order displacement.
    - Long-term (2035+ or year=None): Pure threshold-based retirement (backward compat).

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

    # Apply announced coal retirements (near-term through 2035)
    if year is not None and year <= 2035 and coal_cap > 0:
        retired_gw = get_announced_coal_retired_gw(iso, year)
        retired_twh = retired_gw * 4.38  # ~50% capacity factor
        coal_cap = max(0, coal_cap - retired_twh)

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

    # Sigmoid coal phase-out: coal capacity declines smoothly from 50% to 85% clean
    # Oil follows coal at same fraction (both retire together in merit order)
    cf = coal_fraction_at_clean_pct(clean_pct)
    effective_coal_cap = coal_cap * cf
    effective_oil_cap = oil_cap * cf

    coal_twh = min(effective_coal_cap, fossil_twh)
    remaining_fossil = fossil_twh - coal_twh
    oil_twh = min(effective_oil_cap, remaining_fossil)
    gas_twh = max(0, fossil_twh - coal_twh - oil_twh)

    additional_clean_twh = max(0, (clean_pct - baseline_clean) / 100.0 * grown_demand_twh)

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

    # Sigmoid coal/oil phase-out
    cf = coal_fraction_at_clean_pct(clean_pct)
    effective_coal_cap = coal_cap * cf
    effective_oil_cap = oil_cap * cf

    coal_twh = min(effective_coal_cap, fossil_twh)
    remaining = fossil_twh - coal_twh
    oil_twh = min(effective_oil_cap, remaining)
    gas_twh = max(0, fossil_twh - coal_twh - oil_twh)

    return {
        'coal_twh': round(coal_twh, 2),
        'oil_twh': round(oil_twh, 2),
        'gas_twh': round(gas_twh, 2),
        'total_fossil_twh': round(fossil_twh, 2),
        'coal_fraction': round(cf, 4),
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

def compute_co2_from_dispatch(iso, dispatch_result, emission_rates, demand_total_mwh,
                              clean_pct=None, ccs_capture_level='Medium'):
    """Compute CO₂ displacement from hourly dispatch results with hour-varying rates.

    Uses the 8760-hour fossil_displaced and ccs_supply arrays from the dispatch
    cache. Emission rate varies per-hour based on merit-order dispatch: in hours
    with high residual demand, coal/oil are marginal (high rate); in low-demand
    hours, gas is marginal (lower rate). This captures the temporal specificity
    that static annual rates miss (Siler-Evans et al. 2012, ES&T).

    Args:
        iso: ISO region identifier
        dispatch_result: dict from dispatch cache with 'fossil_displaced', 'ccs_supply' arrays
        emission_rates: eGRID emission rates dict
        demand_total_mwh: annual demand in MWh (for scaling normalized arrays)
        clean_pct: clean energy percentage (for sigmoid coal phase-out)
        ccs_capture_level: 'Low'/'Medium'/'High' for CCS capture rate sensitivity

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

    # Sigmoid coal/oil phase-out
    cf = coal_fraction_at_clean_pct(clean_pct) if clean_pct is not None else 1.0
    coal_cap_mwh = COAL_CAP_TWH.get(iso, 0) * 1e6 * cf
    oil_cap_mwh = OIL_CAP_TWH.get(iso, 0) * 1e6 * cf

    # ── Hour-varying emission rates via merit-order dispatch ──
    # Each hour: determine which fossil fuel is marginal based on cumulative
    # displacement. Hours with high displacement exhaust coal cap → marginal = gas.
    # Hours with low displacement → marginal = coal (highest emitting).
    # This captures the key insight from Siler-Evans et al. (2012) that marginal
    # rates vary 2-3x within a day.

    # Sort hours by displacement magnitude to build cumulative merit-order
    hourly_co2 = np.zeros(H, dtype=np.float64)

    # Vectorized merit-order attribution per hour
    # For each hour, fossil_mwh[h] is allocated: coal first, then oil, then gas
    # But we need to respect *fleet-wide* caps across all hours
    # Approach: allocate coal/oil/gas proportionally based on fleet-level caps
    if total_displaced_mwh > 0:
        coal_displaced_mwh = min(total_displaced_mwh, coal_cap_mwh)
        remaining = total_displaced_mwh - coal_displaced_mwh
        oil_displaced_mwh = min(remaining, oil_cap_mwh)
        gas_displaced_mwh = max(0, remaining - oil_displaced_mwh)

        # Compute fleet-level weighted rate
        weighted_rate = (
            coal_displaced_mwh * coal_rate +
            oil_displaced_mwh * oil_rate +
            gas_displaced_mwh * gas_rate
        ) / total_displaced_mwh

        # Hour-varying rate: hours with higher displacement are more likely to be
        # displacing gas (lower rate); hours with lower displacement displace coal first
        # Use a linear interpolation based on per-hour displacement rank
        sort_idx = np.argsort(fossil_mwh)  # ascending: low displacement → high
        cumulative_mwh = np.cumsum(fossil_mwh[sort_idx])

        # Per-hour marginal rate based on cumulative position in merit order
        hourly_rate = np.full(H, gas_rate, dtype=np.float64)
        coal_mask = cumulative_mwh <= coal_cap_mwh
        oil_mask = (~coal_mask) & (cumulative_mwh <= coal_cap_mwh + oil_cap_mwh)
        gas_mask = ~coal_mask & ~oil_mask

        hourly_rate_sorted = np.full(H, gas_rate, dtype=np.float64)
        hourly_rate_sorted[coal_mask] = coal_rate
        hourly_rate_sorted[oil_mask] = oil_rate
        hourly_rate_sorted[gas_mask] = gas_rate

        # Unsort back to original hour order
        hourly_rate[sort_idx] = hourly_rate_sorted

        # CO₂ from non-CCS displacement (hour-varying)
        hourly_co2 = non_ccs_mwh * hourly_rate

        # CCS credit with sensitivity
        capture_rate = CCS_CAPTURE_RATES.get(ccs_capture_level, 0.90)
        ccs_residual = CCS_UNABATED_GAS_RATE * (1 - capture_rate)
        ccs_credit_per_hour = np.maximum(0.0, hourly_rate - ccs_residual)
        hourly_co2 += ccs_mwh * ccs_credit_per_hour
    else:
        weighted_rate = gas_rate
        coal_displaced_mwh = 0.0
        oil_displaced_mwh = 0.0
        gas_displaced_mwh = 0.0

    total_co2 = float(np.sum(hourly_co2))
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
        'coal_fraction': round(cf, 4),
        'ccs_capture_level': ccs_capture_level,
        'methodology': 'dispatch_cache_hourly_varying',
    }
