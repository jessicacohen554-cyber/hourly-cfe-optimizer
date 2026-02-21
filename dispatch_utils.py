#!/usr/bin/env python3
"""
Shared Dispatch Utilities — Single source of truth for hourly dispatch reconstruction.
======================================================================================
Extracted from recompute_co2.py to avoid duplicating dispatch logic between the CO2
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

H = 8760
DATA_YEAR = '2025'

ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO']
RESOURCE_TYPES = ['clean_firm', 'solar', 'wind', 'ccs_ccgt', 'hydro']

# ══════════════════════════════════════════════════════════════════════════════
# STORAGE PARAMETERS (must match Step 1 / optimizer)
# ══════════════════════════════════════════════════════════════════════════════

BATTERY_EFFICIENCY = 0.85
BATTERY_DURATION_HOURS = 4
BATTERY8_EFFICIENCY = 0.85
BATTERY8_DURATION_HOURS = 8
LDES_EFFICIENCY = 0.50
LDES_DURATION_HOURS = 100
LDES_WINDOW_DAYS = 7

# ══════════════════════════════════════════════════════════════════════════════
# REGIONAL CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

HYDRO_CAPS = {
    'CAISO': 30, 'ERCOT': 5, 'PJM': 15, 'NYISO': 40, 'NEISO': 30,
}

GRID_MIX_SHARES = {
    'CAISO': {'clean_firm': 7.9, 'solar': 22.3, 'wind': 8.8, 'ccs_ccgt': 0, 'hydro': 9.5},
    'ERCOT': {'clean_firm': 8.6, 'solar': 13.8, 'wind': 23.6, 'ccs_ccgt': 0, 'hydro': 0.1},
    'PJM':   {'clean_firm': 32.1, 'solar': 2.9, 'wind': 3.8, 'ccs_ccgt': 0, 'hydro': 1.8},
    'NYISO': {'clean_firm': 18.4, 'solar': 0.0, 'wind': 4.7, 'ccs_ccgt': 0, 'hydro': 15.9},
    'NEISO': {'clean_firm': 23.8, 'solar': 1.4, 'wind': 3.9, 'ccs_ccgt': 0, 'hydro': 4.4},
}

BASE_DEMAND_TWH = {
    'CAISO': 224.0, 'ERCOT': 488.0, 'PJM': 843.3, 'NYISO': 151.6, 'NEISO': 115.3,
}

COAL_OIL_RETIREMENT_THRESHOLD = 70.0

COAL_CAP_TWH = {
    'CAISO': 0.00, 'ERCOT': 67.58, 'PJM': 139.09, 'NYISO': 0.00, 'NEISO': 0.31,
}
OIL_CAP_TWH = {
    'CAISO': 0.60, 'ERCOT': 0.00, 'PJM': 4.59, 'NYISO': 0.15, 'NEISO': 1.29,
}

CCS_RESIDUAL_EMISSION_RATE = 0.037  # tCO2/MWh after 90% capture

# Nuclear seasonal derate (from Step 1)
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

# Wholesale prices and fuel adjustments (from Step 3)
WHOLESALE_PRICES = {'CAISO': 30, 'ERCOT': 27, 'PJM': 34, 'NYISO': 42, 'NEISO': 41}
FUEL_ADJUSTMENTS = {
    'CAISO': {'Low': -5, 'Medium': 0, 'High': 10},
    'ERCOT': {'Low': -7, 'Medium': 0, 'High': 12},
    'PJM':   {'Low': -6, 'Medium': 0, 'High': 11},
    'NYISO': {'Low': -4, 'Medium': 0, 'High': 8},
    'NEISO': {'Low': -4, 'Medium': 0, 'High': 8},
}

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_common_data():
    """Load all shared data files: demand, gen profiles, emission rates, fossil mix."""
    with open(os.path.join(DATA_DIR, 'eia_demand_profiles.json')) as f:
        demand_data = json.load(f)
    with open(os.path.join(DATA_DIR, 'eia_generation_profiles.json')) as f:
        gen_profiles = json.load(f)
    with open(os.path.join(DATA_DIR, 'egrid_emission_rates.json')) as f:
        emission_rates = json.load(f)
    with open(os.path.join(DATA_DIR, 'eia_fossil_mix.json')) as f:
        fossil_mix = json.load(f)
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
    """Get generation shape profiles — Step 1 version with nuclear seasonal derate.

    This is the authoritative version. recompute_co2.py's simpler version (flat
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
    iso_data = gen_profiles.get(iso, {})
    year_data = iso_data.get(DATA_YEAR, iso_data)
    if isinstance(year_data, dict):
        profiles['wind'] = year_data.get('wind', [0.0] * H)[:H]
    else:
        profiles['wind'] = [0.0] * H

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


def get_supply_profiles_simple(iso, gen_profiles):
    """Simplified supply profiles (flat clean_firm, no DST correction).

    Backward-compatible with recompute_co2.py's original implementation.
    Use get_supply_profiles() for new code.
    """
    profiles = {}
    profiles['clean_firm'] = [1.0 / H] * H

    iso_data = gen_profiles.get(iso, {})
    year_data = iso_data.get(DATA_YEAR, iso_data)

    if iso == 'NYISO':
        if isinstance(year_data, dict):
            p = year_data.get('solar_proxy')
        else:
            p = None
        if not p:
            neiso_data = gen_profiles.get('NEISO', {})
            neiso_year = neiso_data.get(DATA_YEAR, neiso_data)
            p = neiso_year.get('solar') if isinstance(neiso_year, dict) else None
        if not p:
            p = [0.0] * H
        profiles['solar'] = list(p[:H])
    else:
        if isinstance(year_data, dict):
            profiles['solar'] = year_data.get('solar', [0.0] * H)[:H]
        else:
            profiles['solar'] = [0.0] * H

    if isinstance(year_data, dict):
        profiles['wind'] = year_data.get('wind', [0.0] * H)[:H]
    else:
        profiles['wind'] = [0.0] * H

    profiles['ccs_ccgt'] = [1.0 / H] * H

    if isinstance(year_data, dict):
        profiles['hydro'] = year_data.get('hydro', [0.0] * H)[:H]
    else:
        profiles['hydro'] = [0.0] * H

    for rtype in RESOURCE_TYPES:
        if len(profiles[rtype]) > H:
            profiles[rtype] = profiles[rtype][:H]
        elif len(profiles[rtype]) < H:
            profiles[rtype] = list(profiles[rtype]) + [0.0] * (H - len(profiles[rtype]))

    return profiles


# ══════════════════════════════════════════════════════════════════════════════
# HOURLY DISPATCH RECONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

# --- Numba-accelerated inner loops (10-50x faster than pure Python) ---

@njit(cache=True)
def _battery_loop(residual_surplus, residual_gap, dispatch_profile,
                  num_days, daily_target, power_rating, efficiency):
    """Inner loop for daily battery dispatch — Numba-compiled."""
    for day in range(num_days):
        ds = day * 24
        de = ds + 24
        day_surplus = residual_surplus[ds:de].copy()
        day_gap = residual_gap[ds:de].copy()

        max_from_charge = 0.0
        gap_sum = 0.0
        for i in range(24):
            if day_surplus[i] > 0:
                max_from_charge += day_surplus[i]
            if day_gap[i] > 0:
                gap_sum += day_gap[i]
        max_from_charge *= efficiency

        actual_dispatch = daily_target
        if max_from_charge < actual_dispatch:
            actual_dispatch = max_from_charge
        if gap_sum < actual_dispatch:
            actual_dispatch = gap_sum
        if actual_dispatch <= 0:
            continue

        required_charge = actual_dispatch / efficiency

        # Charge from largest surpluses
        sorted_idx = np.argsort(-day_surplus)
        remaining_charge = required_charge
        for j in range(24):
            idx = sorted_idx[j]
            if remaining_charge <= 0 or day_surplus[idx] <= 0:
                break
            amt = day_surplus[idx]
            if power_rating < amt:
                amt = power_rating
            if remaining_charge < amt:
                amt = remaining_charge
            residual_surplus[ds + idx] -= amt
            remaining_charge -= amt

        # Discharge to largest gaps
        sorted_gap = np.argsort(-day_gap)
        remaining_dispatch = actual_dispatch
        for j in range(24):
            idx = sorted_gap[j]
            if remaining_dispatch <= 0 or day_gap[idx] <= 0:
                break
            amt = day_gap[idx]
            if power_rating < amt:
                amt = power_rating
            if remaining_dispatch < amt:
                amt = remaining_dispatch
            dispatch_profile[ds + idx] = amt
            residual_gap[ds + idx] -= amt
            remaining_dispatch -= amt
    return dispatch_profile


@njit(cache=True)
def _ldes_loop(residual_surplus, residual_gap, dispatch_profile,
               energy_capacity, power_rating, ldes_efficiency,
               window_hours, total_hours):
    """Inner loop for LDES multi-day dispatch — Numba-compiled."""
    state_of_charge = 0.0
    num_windows = (total_hours + window_hours - 1) // window_hours

    for w in range(num_windows):
        w_start = w * window_hours
        w_end = w_start + window_hours
        if w_end > total_hours:
            w_end = total_hours
        w_len = w_end - w_start

        w_surplus = residual_surplus[w_start:w_end].copy()
        w_gap = residual_gap[w_start:w_end].copy()

        # Charge from surplus
        surplus_indices = np.argsort(-w_surplus)
        for j in range(w_len):
            idx = surplus_indices[j]
            if w_surplus[idx] <= 0:
                break
            space = energy_capacity - state_of_charge
            if space <= 0:
                break
            charge_amt = w_surplus[idx]
            if power_rating < charge_amt:
                charge_amt = power_rating
            if space < charge_amt:
                charge_amt = space
            if charge_amt > 0:
                state_of_charge += charge_amt

        # Discharge to gaps
        gap_indices = np.argsort(-w_gap)
        for j in range(w_len):
            idx = gap_indices[j]
            if w_gap[idx] <= 0:
                break
            avail = state_of_charge * ldes_efficiency
            if avail <= 0:
                break
            dispatch_amt = w_gap[idx]
            if power_rating < dispatch_amt:
                dispatch_amt = power_rating
            if avail < dispatch_amt:
                dispatch_amt = avail
            if dispatch_amt > 0:
                dispatch_profile[w_start + idx] = dispatch_amt
                state_of_charge -= dispatch_amt / ldes_efficiency
                residual_gap[w_start + idx] -= dispatch_amt

    return dispatch_profile


def _dispatch_battery(residual_surplus, residual_gap, dispatch_pct, duration_hours,
                      efficiency):
    """Daily-cycle battery dispatch (4hr or 8hr). Modifies residual arrays in-place."""
    dispatch_profile = np.zeros(H, dtype=np.float64)
    if dispatch_pct <= 0:
        return dispatch_profile

    total_dispatch = dispatch_pct / 100.0
    num_days = H // 24
    daily_target = total_dispatch / num_days
    power_rating = daily_target / duration_hours

    return _battery_loop(residual_surplus, residual_gap, dispatch_profile,
                         num_days, daily_target, power_rating, efficiency)


def _dispatch_ldes(residual_surplus, residual_gap, dispatch_pct, demand_arr):
    """LDES multi-day dispatch (100hr, 7-day window). Modifies residual arrays in-place."""
    dispatch_profile = np.zeros(H, dtype=np.float64)
    if dispatch_pct <= 0:
        return dispatch_profile

    total_demand_energy = demand_arr.sum()
    energy_capacity = total_demand_energy * (24.0 / H)
    power_rating = energy_capacity / LDES_DURATION_HOURS
    window_hours = LDES_WINDOW_DAYS * 24

    return _ldes_loop(residual_surplus, residual_gap, dispatch_profile,
                      energy_capacity, power_rating, LDES_EFFICIENCY,
                      window_hours, H)


def build_supply_matrix(supply_profiles):
    """Pre-convert supply profile dict → (5, H) numpy matrix.

    Call once per ISO. Pass the matrix to reconstruct_hourly_dispatch via
    supply_matrix kwarg for ~3x speedup on repeated dispatches.
    """
    matrix = np.zeros((len(RESOURCE_TYPES), H), dtype=np.float64)
    for i, rtype in enumerate(RESOURCE_TYPES):
        p = supply_profiles.get(rtype, [0.0] * H)
        matrix[i, :] = np.array(p[:H], dtype=np.float64)
    return matrix


def reconstruct_hourly_dispatch(demand_norm, supply_profiles, resource_pcts,
                                 procurement_pct, battery_dispatch_pct,
                                 battery8_dispatch_pct, ldes_dispatch_pct,
                                 supply_matrix=None):
    """Reconstruct full 8760 hourly dispatch for a resource mix.

    Args:
        supply_matrix: optional (5, H) numpy array from build_supply_matrix().
            If provided, skips per-call array conversion (faster for batch calls).

    Returns:
        result dict with keys:
          - supply_total: (H,) clean supply before storage
          - battery4_profile: (H,) 4hr battery dispatch
          - battery8_profile: (H,) 8hr battery dispatch
          - ldes_profile: (H,) LDES dispatch
          - total_clean: (H,) total clean supply including storage
          - residual_demand: (H,) demand not met by clean (positive = fossil needed)
          - curtailed: (H,) curtailed clean energy
          - ccs_supply: (H,) CCS-CCGT supply portion
          - fossil_displaced: (H,) fossil MWh displaced at each hour (normalized)
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

    battery4_profile = _dispatch_battery(
        residual_surplus, residual_gap,
        battery_dispatch_pct, BATTERY_DURATION_HOURS, BATTERY_EFFICIENCY)

    battery8_profile = _dispatch_battery(
        residual_surplus, residual_gap,
        battery8_dispatch_pct, BATTERY8_DURATION_HOURS, BATTERY8_EFFICIENCY)

    ldes_profile = _dispatch_ldes(
        residual_surplus, residual_gap,
        ldes_dispatch_pct, demand_arr)

    total_clean = supply_total + battery4_profile + battery8_profile + ldes_profile
    fossil_displaced = np.minimum(demand_arr, total_clean)
    residual_demand = np.maximum(0.0, demand_arr - total_clean)
    curtailed = np.maximum(0.0, total_clean - demand_arr)

    return {
        'supply_total': supply_total,
        'battery4_profile': battery4_profile,
        'battery8_profile': battery8_profile,
        'ldes_profile': ldes_profile,
        'total_clean': total_clean,
        'residual_demand': residual_demand,
        'curtailed': curtailed,
        'ccs_supply': ccs_supply,
        'fossil_displaced': fossil_displaced,
    }


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

DISPATCH_CACHE_DIR = os.path.join(DATA_DIR, 'dispatch_cache')


def _archetype_key(iso, resource_pcts, procurement_pct, battery_dispatch_pct,
                   battery8_dispatch_pct, ldes_dispatch_pct):
    """Deterministic hash key for a unique dispatch archetype."""
    parts = [
        iso,
        str(resource_pcts.get('clean_firm', 0)),
        str(resource_pcts.get('solar', 0)),
        str(resource_pcts.get('wind', 0)),
        str(resource_pcts.get('ccs_ccgt', 0)),
        str(resource_pcts.get('hydro', 0)),
        str(procurement_pct),
        str(battery_dispatch_pct),
        str(battery8_dispatch_pct),
        str(ldes_dispatch_pct),
    ]
    key_str = '|'.join(parts)
    return hashlib.md5(key_str.encode()).hexdigest()[:16]


def _cache_path(iso):
    """Per-ISO cache file path."""
    return os.path.join(DISPATCH_CACHE_DIR, f'{iso}_dispatch_cache.npz')


def load_dispatch_cache(iso):
    """Load existing dispatch cache for an ISO. Returns dict of {key: arrays_dict}."""
    path = _cache_path(iso)
    if not os.path.exists(path):
        return {}
    try:
        data = np.load(path, allow_pickle=True)
        cache = {}
        # Keys stored as 'key_{hash}_{field}' → reconstruct
        keys_seen = set()
        for arr_name in data.files:
            parts = arr_name.split('_', 2)
            if len(parts) >= 3 and parts[0] == 'k':
                k = parts[1]
                field = '_'.join(parts[2:])
                if k not in cache:
                    cache[k] = {}
                cache[k][field] = data[arr_name]
                keys_seen.add(k)
        return cache
    except Exception:
        return {}


def save_dispatch_cache(iso, cache):
    """Save dispatch cache for an ISO. Overwrites existing file."""
    os.makedirs(DISPATCH_CACHE_DIR, exist_ok=True)
    arrays = {}
    for k, fields in cache.items():
        for field, arr in fields.items():
            arrays[f'k_{k}_{field}'] = arr
    path = _cache_path(iso)
    # np.savez_compressed auto-appends .npz, so use a stem without extension
    tmp_stem = path.replace('.npz', '') + '_tmp'
    np.savez_compressed(tmp_stem, **arrays)
    tmp_file = tmp_stem + '.npz'
    os.replace(tmp_file, path)


def get_or_compute_dispatch(iso, demand_norm, supply_profiles, resource_pcts,
                             procurement_pct, battery_dispatch_pct,
                             battery8_dispatch_pct, ldes_dispatch_pct,
                             cache=None):
    """Get dispatch from cache or compute and add to cache.

    Args:
        cache: mutable dict (load_dispatch_cache output). If provided, checks cache
               first and adds new results. Caller is responsible for calling
               save_dispatch_cache() when done with a batch.

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
        battery8_dispatch_pct, ldes_dispatch_pct)

    if cache is not None:
        cache[key] = {k: v for k, v in result.items()}

    return result, False
