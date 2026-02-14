#!/usr/bin/env python3
"""
Over-Procurement Optimization Engine — Advanced Sensitivity Model
==================================================================
For each ISO region (CAISO, ERCOT, PJM, NYISO, NEISO), sweep over-procurement
levels and find the optimal mix of 5 resource types (clean_firm, solar, wind,
ccs_ccgt, hydro) + 2 storage types (battery 4hr Li-ion + LDES 100hr iron-air)
to maximize hourly matching at minimum cost.

Three-phase refinement: coarse -> medium -> fine (adapted for 5D)

Resource types:
  - Clean Firm: flat baseload (1/8760 per hour) — nuclear/geothermal
  - Solar: EIA 2025 hourly regional profile
  - Wind: EIA 2025 hourly regional profile
  - CCS-CCGT: flat baseload (1/8760 per hour) — dispatchable, 90% capture
  - Hydro: EIA 2025 hourly regional profile (capped by region, existing only)

Storage:
  - Battery: 4hr Li-ion, 85% RTE, daily-cycle greedy dispatch
  - LDES: 100hr iron-air, 50% RTE, 7-day rolling window multi-day dispatch
"""

import json
import os
import sys
import time
import numpy as np
from multiprocessing import Pool

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
DATA_YEAR = '2025'
H = 8760

# ══════════════════════════════════════════════════════════════════════════════
# STORAGE CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

BATTERY_EFFICIENCY = 0.85
BATTERY_DURATION_HOURS = 4

LDES_EFFICIENCY = 0.50
LDES_DURATION_HOURS = 100
LDES_WINDOW_DAYS = 7

# ══════════════════════════════════════════════════════════════════════════════
# REGIONAL CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

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

# 5 optimization resource types (sum to 100%)
RESOURCE_TYPES = ['clean_firm', 'solar', 'wind', 'ccs_ccgt', 'hydro']

# 10 target thresholds — key inflection points with 2.5% granularity in steep zone
THRESHOLDS = [75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 100]

# Adaptive procurement bounds by threshold — avoids wasting compute searching
# procurement levels that can't possibly be optimal for a given threshold
PROCUREMENT_BOUNDS = {
    75:   (75, 105),
    80:   (75, 110),
    85:   (80, 110),
    87.5: (87, 130),
    90:   (90, 140),
    92.5: (92, 150),
    95:   (95, 175),
    97.5: (100, 200),
    99:   (100, 220),
    100:  (100, 200),
}

# ══════════════════════════════════════════════════════════════════════════════
# COST TABLES — Medium values used by optimizer; full L/M/H output for dashboard
# ══════════════════════════════════════════════════════════════════════════════

# Wholesale electricity prices ($/MWh) — 2025 averages from FERC/ISO market reports
WHOLESALE_PRICES = {
    'CAISO': 30,
    'ERCOT': 27,
    'PJM':   34,
    'NYISO': 42,
    'NEISO': 41,
}

# Medium LCOE values used by the optimizer for mix optimization
REGIONAL_LCOE = {
    'CAISO': {'clean_firm': 78, 'solar': 60, 'wind': 73, 'ccs_ccgt': 86, 'hydro': 0, 'battery': 102, 'ldes': 180},
    'ERCOT': {'clean_firm': 85, 'solar': 54, 'wind': 40, 'ccs_ccgt': 71, 'hydro': 0, 'battery': 92, 'ldes': 155},
    'PJM':   {'clean_firm': 93, 'solar': 65, 'wind': 62, 'ccs_ccgt': 79, 'hydro': 0, 'battery': 98, 'ldes': 170},
    'NYISO': {'clean_firm': 98, 'solar': 92, 'wind': 81, 'ccs_ccgt': 99, 'hydro': 0, 'battery': 108, 'ldes': 200},
    'NEISO': {'clean_firm': 96, 'solar': 82, 'wind': 73, 'ccs_ccgt': 96, 'hydro': 0, 'battery': 105, 'ldes': 190},
}

# Transmission adders at Medium level (used by optimizer)
TRANSMISSION_ADDERS = {
    'CAISO': {'wind': 8, 'solar': 3, 'clean_firm': 3, 'ccs_ccgt': 2, 'battery': 1, 'ldes': 2, 'hydro': 0},
    'ERCOT': {'wind': 6, 'solar': 3, 'clean_firm': 2, 'ccs_ccgt': 2, 'battery': 1, 'ldes': 2, 'hydro': 0},
    'PJM':   {'wind': 10, 'solar': 5, 'clean_firm': 3, 'ccs_ccgt': 3, 'battery': 1, 'ldes': 3, 'hydro': 0},
    'NYISO': {'wind': 14, 'solar': 7, 'clean_firm': 5, 'ccs_ccgt': 4, 'battery': 2, 'ldes': 4, 'hydro': 0},
    'NEISO': {'wind': 12, 'solar': 6, 'clean_firm': 4, 'ccs_ccgt': 3, 'battery': 2, 'ldes': 3, 'hydro': 0},
}

# Full L/M/H cost tables for dashboard output
FULL_LCOE_TABLES = {
    'solar': {
        'Low':    {'CAISO': 45, 'ERCOT': 40, 'PJM': 50, 'NYISO': 70, 'NEISO': 62},
        'Medium': {'CAISO': 60, 'ERCOT': 54, 'PJM': 65, 'NYISO': 92, 'NEISO': 82},
        'High':   {'CAISO': 78, 'ERCOT': 70, 'PJM': 85, 'NYISO': 120, 'NEISO': 107},
    },
    'wind': {
        'Low':    {'CAISO': 55, 'ERCOT': 30, 'PJM': 47, 'NYISO': 61, 'NEISO': 55},
        'Medium': {'CAISO': 73, 'ERCOT': 40, 'PJM': 62, 'NYISO': 81, 'NEISO': 73},
        'High':   {'CAISO': 95, 'ERCOT': 52, 'PJM': 81, 'NYISO': 105, 'NEISO': 95},
    },
    'clean_firm': {
        'Low':    {'CAISO': 58, 'ERCOT': 63, 'PJM': 72, 'NYISO': 75, 'NEISO': 73},
        'Medium': {'CAISO': 78, 'ERCOT': 85, 'PJM': 93, 'NYISO': 98, 'NEISO': 96},
        'High':   {'CAISO': 110, 'ERCOT': 120, 'PJM': 140, 'NYISO': 150, 'NEISO': 145},
    },
    'ccs_ccgt': {
        'Low':    {'CAISO': 58, 'ERCOT': 52, 'PJM': 62, 'NYISO': 78, 'NEISO': 75},
        'Medium': {'CAISO': 86, 'ERCOT': 71, 'PJM': 79, 'NYISO': 99, 'NEISO': 96},
        'High':   {'CAISO': 115, 'ERCOT': 92, 'PJM': 102, 'NYISO': 128, 'NEISO': 122},
    },
    'battery': {
        'Low':    {'CAISO': 77, 'ERCOT': 69, 'PJM': 74, 'NYISO': 81, 'NEISO': 79},
        'Medium': {'CAISO': 102, 'ERCOT': 92, 'PJM': 98, 'NYISO': 108, 'NEISO': 105},
        'High':   {'CAISO': 133, 'ERCOT': 120, 'PJM': 127, 'NYISO': 140, 'NEISO': 137},
    },
    'ldes': {
        'Low':    {'CAISO': 135, 'ERCOT': 116, 'PJM': 128, 'NYISO': 150, 'NEISO': 143},
        'Medium': {'CAISO': 180, 'ERCOT': 155, 'PJM': 170, 'NYISO': 200, 'NEISO': 190},
        'High':   {'CAISO': 234, 'ERCOT': 202, 'PJM': 221, 'NYISO': 260, 'NEISO': 247},
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# PAIRED TOGGLE GROUPS — 5 groups replacing 10 individual toggles
# Dashboard toggles move all member variables in lockstep (all L, all M, or all H)
# ══════════════════════════════════════════════════════════════════════════════

PAIRED_TOGGLE_GROUPS = {
    'renewable_generation': {
        'label': 'Renewable Generation Cost',
        'options': ['Low', 'Medium', 'High'],
        'members': ['solar', 'wind'],
        'description': 'Solar and wind LCOE move together — driven by similar manufacturing scale, supply chain, and installation labor factors',
    },
    'firm_generation': {
        'label': 'Firm Generation Cost',
        'options': ['Low', 'Medium', 'High'],
        'members': ['clean_firm', 'ccs_ccgt'],
        'description': 'Clean firm (nuclear/geothermal) and CCS-CCGT share capital-intensive, long-lead-time cost structures',
    },
    'storage': {
        'label': 'Storage Cost',
        'options': ['Low', 'Medium', 'High'],
        'members': ['battery', 'ldes'],
        'description': 'Battery and LDES costs share manufacturing/materials cost drivers across storage technologies',
    },
    'fossil_fuel': {
        'label': 'Fossil Fuel Price',
        'options': ['Low', 'Medium', 'High'],
        'members': ['natural_gas', 'coal', 'oil'],
        'description': 'Gas, coal, and oil prices are correlated through energy commodity markets and macro conditions',
    },
    'transmission': {
        'label': 'Transmission Cost',
        'options': ['None', 'Low', 'Medium', 'High'],
        'members': ['wind', 'solar', 'clean_firm', 'ccs_ccgt', 'battery', 'ldes'],
        'description': 'Grid interconnection and transmission infrastructure costs affect all new-build resources within a region',
    },
}

# Full transmission adder tables (None/Low/Medium/High) for dashboard
FULL_TRANSMISSION_TABLES = {
    'wind': {
        'None':   {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 0, 'NEISO': 0},
        'Low':    {'CAISO': 4, 'ERCOT': 3, 'PJM': 5, 'NYISO': 7, 'NEISO': 6},
        'Medium': {'CAISO': 8, 'ERCOT': 6, 'PJM': 10, 'NYISO': 14, 'NEISO': 12},
        'High':   {'CAISO': 14, 'ERCOT': 10, 'PJM': 18, 'NYISO': 22, 'NEISO': 20},
    },
    'solar': {
        'None':   {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 0, 'NEISO': 0},
        'Low':    {'CAISO': 1, 'ERCOT': 1, 'PJM': 2, 'NYISO': 3, 'NEISO': 3},
        'Medium': {'CAISO': 3, 'ERCOT': 3, 'PJM': 5, 'NYISO': 7, 'NEISO': 6},
        'High':   {'CAISO': 6, 'ERCOT': 5, 'PJM': 9, 'NYISO': 12, 'NEISO': 10},
    },
    'clean_firm': {
        'None':   {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 0, 'NEISO': 0},
        'Low':    {'CAISO': 1, 'ERCOT': 1, 'PJM': 1, 'NYISO': 2, 'NEISO': 2},
        'Medium': {'CAISO': 3, 'ERCOT': 2, 'PJM': 3, 'NYISO': 5, 'NEISO': 4},
        'High':   {'CAISO': 6, 'ERCOT': 4, 'PJM': 6, 'NYISO': 9, 'NEISO': 7},
    },
    'ccs_ccgt': {
        'None':   {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 0, 'NEISO': 0},
        'Low':    {'CAISO': 1, 'ERCOT': 1, 'PJM': 1, 'NYISO': 2, 'NEISO': 2},
        'Medium': {'CAISO': 2, 'ERCOT': 2, 'PJM': 3, 'NYISO': 4, 'NEISO': 3},
        'High':   {'CAISO': 4, 'ERCOT': 3, 'PJM': 5, 'NYISO': 7, 'NEISO': 6},
    },
    'battery': {
        'None':   {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 0, 'NEISO': 0},
        'Low':    {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 1, 'NEISO': 1},
        'Medium': {'CAISO': 1, 'ERCOT': 1, 'PJM': 1, 'NYISO': 2, 'NEISO': 2},
        'High':   {'CAISO': 2, 'ERCOT': 2, 'PJM': 3, 'NYISO': 4, 'NEISO': 3},
    },
    'ldes': {
        'None':   {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 0, 'NEISO': 0},
        'Low':    {'CAISO': 1, 'ERCOT': 1, 'PJM': 1, 'NYISO': 2, 'NEISO': 2},
        'Medium': {'CAISO': 2, 'ERCOT': 2, 'PJM': 3, 'NYISO': 4, 'NEISO': 3},
        'High':   {'CAISO': 4, 'ERCOT': 3, 'PJM': 5, 'NYISO': 7, 'NEISO': 6},
    },
    'hydro': {
        'None':   {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 0, 'NEISO': 0},
        'Low':    {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 0, 'NEISO': 0},
        'Medium': {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 0, 'NEISO': 0},
        'High':   {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 0, 'NEISO': 0},
    },
}

# Fuel prices (L/M/H) for dashboard
FUEL_PRICES = {
    'natural_gas': {'Low': 2.00, 'Medium': 3.50, 'High': 6.00},
    'coal':        {'Low': 1.80, 'Medium': 2.50, 'High': 4.00},
    'oil':         {'Low': 55,   'Medium': 75,   'High': 110},
}

# Existing clean grid mix shares (% of total generation) from EIA-930 2025 data
# Resources up to these shares priced at wholesale; above = new-build LCOE
# CCS-CCGT has no existing share (entirely new-build)
GRID_MIX_SHARES = {
    'CAISO': {'clean_firm': 7.9, 'solar': 22.3, 'wind': 8.8, 'ccs_ccgt': 0, 'hydro': 9.5},
    'ERCOT': {'clean_firm': 8.6, 'solar': 13.8, 'wind': 23.6, 'ccs_ccgt': 0, 'hydro': 0.1},
    'PJM':   {'clean_firm': 32.1, 'solar': 2.9, 'wind': 3.8, 'ccs_ccgt': 0, 'hydro': 1.8},
    'NYISO': {'clean_firm': 18.4, 'solar': 0.0, 'wind': 4.7, 'ccs_ccgt': 0, 'hydro': 15.9},
    'NEISO': {'clean_firm': 23.8, 'solar': 1.4, 'wind': 3.9, 'ccs_ccgt': 0, 'hydro': 4.4},
}

# CCS-CCGT residual emission rate (tCO2/MWh) after 90% capture
CCS_RESIDUAL_EMISSION_RATE = 0.037


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    """Load demand profiles, generation profiles, emission rates, and fossil mix."""
    print("Loading data...")

    with open(os.path.join(DATA_DIR, 'eia_demand_profiles.json')) as f:
        demand_data = json.load(f)

    with open(os.path.join(DATA_DIR, 'eia_generation_profiles.json')) as f:
        gen_profiles = json.load(f)

    with open(os.path.join(DATA_DIR, 'egrid_emission_rates.json')) as f:
        emission_rates = json.load(f)

    with open(os.path.join(DATA_DIR, 'eia_fossil_mix.json')) as f:
        fossil_mix = json.load(f)

    print("  Data loaded.")
    return demand_data, gen_profiles, emission_rates, fossil_mix


def get_supply_profiles(iso, gen_profiles):
    """Get generation shape profiles for the 5 resource types."""
    profiles = {}

    # Clean firm = flat baseload
    profiles['clean_firm'] = [1.0 / H] * H

    # Solar
    if iso == 'NYISO':
        p = gen_profiles[iso][DATA_YEAR].get('solar_proxy')
        if not p:
            p = gen_profiles['NEISO'][DATA_YEAR].get('solar')
        profiles['solar'] = p[:H]
    else:
        profiles['solar'] = gen_profiles[iso][DATA_YEAR].get('solar', [0.0] * H)[:H]

    # Wind
    profiles['wind'] = gen_profiles[iso][DATA_YEAR].get('wind', [0.0] * H)[:H]

    # CCS-CCGT = flat baseload (same shape as clean firm)
    profiles['ccs_ccgt'] = [1.0 / H] * H

    # Hydro
    profiles['hydro'] = gen_profiles[iso][DATA_YEAR].get('hydro', [0.0] * H)[:H]

    # Ensure all profiles are exactly H hours
    for rtype in RESOURCE_TYPES:
        if len(profiles[rtype]) > H:
            profiles[rtype] = profiles[rtype][:H]
        elif len(profiles[rtype]) < H:
            profiles[rtype] = profiles[rtype] + [0.0] * (H - len(profiles[rtype]))

    return profiles


def find_anomaly_hours(iso, gen_profiles):
    """Find hours where all gen types report zero (EIA data gaps)."""
    types = [t for t in gen_profiles[iso][DATA_YEAR].keys() if t != 'solar_proxy']
    anomalies = set()
    for h in range(H):
        if all(gen_profiles[iso][DATA_YEAR][t][h] == 0.0 for t in types):
            anomalies.add(h)
    return anomalies


# ══════════════════════════════════════════════════════════════════════════════
# NUMPY PROFILE PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def prepare_numpy_profiles(demand_norm, supply_profiles):
    """Convert profiles to numpy arrays for fast vectorized computation."""
    demand_arr = np.array(demand_norm[:H], dtype=np.float64)
    supply_arrs = {}
    for rtype in RESOURCE_TYPES:
        supply_arrs[rtype] = np.array(supply_profiles[rtype][:H], dtype=np.float64)
    # Pre-build supply matrix: shape (5, 8760) for [cf, solar, wind, ccs_ccgt, hydro]
    supply_matrix = np.stack([supply_arrs[rt] for rt in RESOURCE_TYPES])  # (5, 8760)
    return demand_arr, supply_arrs, supply_matrix


# ══════════════════════════════════════════════════════════════════════════════
# FAST SCORING FUNCTIONS (numpy-accelerated)
# ══════════════════════════════════════════════════════════════════════════════

def fast_hourly_score(demand_arr, supply_matrix, mix_fractions, procurement_factor):
    """
    Ultra-fast hourly matching score using numpy vectorized ops.
    mix_fractions: array of [cf, solar, wind, ccs_ccgt, hydro] as fractions (sum to 1.0)
    Returns: matching score (0-1), where demand sums to 1.0
    """
    supply = procurement_factor * np.dot(mix_fractions, supply_matrix)
    matched = np.minimum(demand_arr, supply)
    return matched.sum()


def fast_score_with_battery(demand_arr, supply_matrix, mix_fractions, procurement_factor,
                            battery_dispatch_pct):
    """
    Fast scoring with battery storage (4hr Li-ion, 85% RTE, daily-cycle greedy dispatch).
    Preserves the original daily-cycle algorithm exactly.
    Returns matching score (0-1).
    """
    supply = procurement_factor * np.dot(mix_fractions, supply_matrix)
    surplus = np.maximum(0.0, supply - demand_arr)
    gap = np.maximum(0.0, demand_arr - supply)
    base_matched = np.minimum(demand_arr, supply)

    if battery_dispatch_pct <= 0:
        return base_matched.sum()

    storage_dispatch_total = battery_dispatch_pct / 100.0
    num_days = H // 24
    daily_dispatch_target = storage_dispatch_total / num_days
    power_rating = daily_dispatch_target / BATTERY_DURATION_HOURS

    total_dispatched = 0.0
    for day in range(num_days):
        ds = day * 24
        de = ds + 24

        day_surplus = surplus[ds:de]
        day_gap = gap[ds:de]

        total_surplus = day_surplus.sum()
        total_gap = day_gap.sum()

        max_from_charge = total_surplus * BATTERY_EFFICIENCY
        actual_dispatch = min(daily_dispatch_target, max_from_charge, total_gap)
        if actual_dispatch <= 0:
            continue

        required_charge = actual_dispatch / BATTERY_EFFICIENCY

        # Distribute charge (greedily, largest surplus first)
        sorted_idx = np.argsort(-day_surplus)
        remaining_charge = required_charge
        for idx in sorted_idx:
            if remaining_charge <= 0 or day_surplus[idx] <= 0:
                break
            amt = min(float(day_surplus[idx]), power_rating, remaining_charge)
            remaining_charge -= amt

        actual_charge = required_charge - remaining_charge
        ach_dispatch = min(actual_dispatch, actual_charge * BATTERY_EFFICIENCY)

        # Distribute dispatch (greedily, largest gap first)
        sorted_idx = np.argsort(-day_gap)
        remaining_dispatch = ach_dispatch
        for idx in sorted_idx:
            if remaining_dispatch <= 0 or day_gap[idx] <= 0:
                break
            amt = min(float(day_gap[idx]), power_rating, remaining_dispatch)
            total_dispatched += amt
            remaining_dispatch -= amt

    return base_matched.sum() + total_dispatched


def compute_ldes_dispatch(demand_arr, supply_arr_total):
    """
    LDES dispatch algorithm: 100hr iron-air, 50% RTE, 7-day rolling window.

    Charges during multi-day surplus periods, discharges during multi-day deficit periods.
    Power rating = capacity / 100hr (very low power, huge energy).

    Args:
        demand_arr: numpy array (H,) of normalized demand
        supply_arr_total: numpy array (H,) of total supply after resource mix

    Returns:
        ldes_dispatch: numpy array (H,) of LDES dispatch amounts (added to matched)
        ldes_charge: numpy array (H,) of LDES charge amounts (absorbed from surplus)
        total_dispatched: float total energy dispatched
    """
    surplus = np.maximum(0.0, supply_arr_total - demand_arr)
    gap = np.maximum(0.0, demand_arr - supply_arr_total)

    ldes_dispatch = np.zeros(H, dtype=np.float64)
    ldes_charge = np.zeros(H, dtype=np.float64)

    # LDES energy capacity: use a capacity that scales with total demand
    # Set capacity as fraction of total demand energy
    total_demand_energy = demand_arr.sum()  # sums to ~1.0 for normalized
    # Capacity sized relative to demand: enough to store ~1 day of average demand
    ldes_energy_capacity = total_demand_energy * (24.0 / H)  # ~1 day of energy
    ldes_power_rating = ldes_energy_capacity / LDES_DURATION_HOURS

    state_of_charge = 0.0
    window_hours = LDES_WINDOW_DAYS * 24

    # Process in 7-day rolling windows
    num_windows = (H + window_hours - 1) // window_hours
    for w in range(num_windows):
        w_start = w * window_hours
        w_end = min(w_start + window_hours, H)
        window_len = w_end - w_start

        # Identify surplus and deficit hours in this window
        w_surplus = surplus[w_start:w_end].copy()
        w_gap = gap[w_start:w_end].copy()

        # Phase 1: Charge during surplus hours (largest surpluses first)
        surplus_indices = np.argsort(-w_surplus)
        for idx in surplus_indices:
            if w_surplus[idx] <= 0:
                break
            space = ldes_energy_capacity - state_of_charge
            if space <= 0:
                break
            charge_amt = min(float(w_surplus[idx]), ldes_power_rating, space)
            if charge_amt > 0:
                ldes_charge[w_start + idx] = charge_amt
                state_of_charge += charge_amt

        # Phase 2: Discharge during deficit hours (largest gaps first)
        gap_indices = np.argsort(-w_gap)
        for idx in gap_indices:
            if w_gap[idx] <= 0:
                break
            available = state_of_charge * LDES_EFFICIENCY
            if available <= 1e-12:
                break
            dispatch_amt = min(float(w_gap[idx]), ldes_power_rating, available)
            if dispatch_amt > 0:
                ldes_dispatch[w_start + idx] = dispatch_amt
                # Energy drawn from storage = dispatch / efficiency
                state_of_charge -= dispatch_amt / LDES_EFFICIENCY
                state_of_charge = max(0.0, state_of_charge)

    total_dispatched = ldes_dispatch.sum()
    return ldes_dispatch, ldes_charge, total_dispatched


def fast_score_with_ldes(demand_arr, supply_matrix, mix_fractions, procurement_factor,
                         ldes_dispatch_pct):
    """
    Fast scoring with LDES only (no battery). Used in sweep/optimization.
    ldes_dispatch_pct is a scaling hint for how much LDES capacity is available.
    Returns matching score (0-1).
    """
    supply = procurement_factor * np.dot(mix_fractions, supply_matrix)

    if ldes_dispatch_pct <= 0:
        matched = np.minimum(demand_arr, supply)
        return matched.sum()

    base_matched = np.minimum(demand_arr, supply)
    ldes_dispatch_arr, _, total_dispatched = compute_ldes_dispatch(demand_arr, supply)

    # Scale the dispatch by the ldes_dispatch_pct factor
    scale = ldes_dispatch_pct / 10.0  # normalize: 10% = 1x capacity
    gap = np.maximum(0.0, demand_arr - supply)
    scaled_dispatch = np.minimum(ldes_dispatch_arr * scale, gap)

    return base_matched.sum() + scaled_dispatch.sum()


def fast_score_with_both_storage(demand_arr, supply_matrix, mix_fractions, procurement_factor,
                                 battery_dispatch_pct, ldes_dispatch_pct):
    """
    Fast scoring with both battery (daily) and LDES (multi-day).
    Battery runs first on daily cycle, LDES fills remaining multi-day gaps.
    Returns matching score (0-1).
    """
    supply = procurement_factor * np.dot(mix_fractions, supply_matrix)
    surplus = np.maximum(0.0, supply - demand_arr)
    gap = np.maximum(0.0, demand_arr - supply)
    base_matched = np.minimum(demand_arr, supply)

    total_dispatched = 0.0
    residual_gap = gap.copy()
    residual_surplus = surplus.copy()

    # Phase 1: Battery daily-cycle dispatch
    if battery_dispatch_pct > 0:
        batt_dispatch_total = battery_dispatch_pct / 100.0
        num_days = H // 24
        daily_dispatch_target = batt_dispatch_total / num_days
        batt_power_rating = daily_dispatch_target / BATTERY_DURATION_HOURS

        for day in range(num_days):
            ds = day * 24
            de = ds + 24

            day_surplus = residual_surplus[ds:de]
            day_gap = residual_gap[ds:de]

            total_surplus_day = day_surplus.sum()
            total_gap_day = day_gap.sum()

            max_from_charge = total_surplus_day * BATTERY_EFFICIENCY
            actual_dispatch = min(daily_dispatch_target, max_from_charge, total_gap_day)
            if actual_dispatch <= 0:
                continue

            required_charge = actual_dispatch / BATTERY_EFFICIENCY

            # Charge from largest surpluses
            sorted_idx = np.argsort(-day_surplus)
            remaining_charge = required_charge
            for idx in sorted_idx:
                if remaining_charge <= 0 or day_surplus[idx] <= 0:
                    break
                amt = min(float(day_surplus[idx]), batt_power_rating, remaining_charge)
                residual_surplus[ds + idx] -= amt
                remaining_charge -= amt

            actual_charge = required_charge - remaining_charge
            ach_dispatch = min(actual_dispatch, actual_charge * BATTERY_EFFICIENCY)

            # Dispatch to largest gaps
            sorted_idx = np.argsort(-day_gap)
            remaining_dispatch = ach_dispatch
            for idx in sorted_idx:
                if remaining_dispatch <= 0 or day_gap[idx] <= 0:
                    break
                amt = min(float(day_gap[idx]), batt_power_rating, remaining_dispatch)
                residual_gap[ds + idx] -= amt
                total_dispatched += amt
                remaining_dispatch -= amt

    # Phase 2: LDES multi-day dispatch on remaining gaps
    if ldes_dispatch_pct > 0:
        total_demand_energy = demand_arr.sum()
        ldes_energy_capacity = total_demand_energy * (24.0 / H)
        ldes_power_rating = ldes_energy_capacity / LDES_DURATION_HOURS
        scale = ldes_dispatch_pct / 10.0

        state_of_charge = 0.0
        window_hours = LDES_WINDOW_DAYS * 24

        num_windows = (H + window_hours - 1) // window_hours
        for w in range(num_windows):
            w_start = w * window_hours
            w_end = min(w_start + window_hours, H)

            w_surplus = residual_surplus[w_start:w_end]
            w_gap = residual_gap[w_start:w_end]

            # Charge
            surplus_indices = np.argsort(-w_surplus)
            for idx in surplus_indices:
                if w_surplus[idx] <= 0:
                    break
                space = (ldes_energy_capacity * scale) - state_of_charge
                if space <= 0:
                    break
                charge_amt = min(float(w_surplus[idx]), ldes_power_rating * scale, space)
                if charge_amt > 0:
                    state_of_charge += charge_amt

            # Discharge
            gap_indices = np.argsort(-w_gap)
            for idx in gap_indices:
                if w_gap[idx] <= 0:
                    break
                available = state_of_charge * LDES_EFFICIENCY
                if available <= 1e-12:
                    break
                dispatch_amt = min(float(w_gap[idx]), ldes_power_rating * scale, available)
                if dispatch_amt > 0:
                    total_dispatched += dispatch_amt
                    state_of_charge -= dispatch_amt / LDES_EFFICIENCY
                    state_of_charge = max(0.0, state_of_charge)

    return base_matched.sum() + total_dispatched


# ══════════════════════════════════════════════════════════════════════════════
# DETAILED DISPATCH COMPUTATION (for final results)
# ══════════════════════════════════════════════════════════════════════════════

def compute_hourly_matching_detailed(demand_norm, supply_profiles, resource_pcts, procurement_pct,
                                     battery_dispatch_pct=0, ldes_dispatch_pct=0):
    """
    Compute detailed hourly matching with both storage types.
    Returns: (score, hourly_detail, battery_dispatch_profile, battery_charge_profile,
              ldes_dispatch_profile, ldes_charge_profile)
    """
    procurement_factor = procurement_pct / 100.0

    hourly_detail = []
    supply_total = np.zeros(H, dtype=np.float64)

    for h in range(H):
        demand_h = demand_norm[h]
        supply_h = 0.0
        for rtype, pct in resource_pcts.items():
            if pct <= 0:
                continue
            supply_h += procurement_factor * (pct / 100.0) * supply_profiles[rtype][h]

        supply_total[h] = supply_h
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

    demand_arr = np.array([d['demand'] for d in hourly_detail], dtype=np.float64)
    surplus_arr = np.array([d['surplus'] for d in hourly_detail], dtype=np.float64)
    gap_arr = np.array([d['gap'] for d in hourly_detail], dtype=np.float64)

    battery_dispatch_profile = np.zeros(H, dtype=np.float64)
    battery_charge_profile = np.zeros(H, dtype=np.float64)
    ldes_dispatch_profile = np.zeros(H, dtype=np.float64)
    ldes_charge_profile = np.zeros(H, dtype=np.float64)

    residual_surplus = surplus_arr.copy()
    residual_gap = gap_arr.copy()

    # Phase 1: Battery daily-cycle dispatch
    if battery_dispatch_pct > 0:
        batt_dispatch_total = battery_dispatch_pct / 100.0
        num_days = H // 24
        daily_dispatch_target = batt_dispatch_total / num_days
        batt_power_rating = daily_dispatch_target / BATTERY_DURATION_HOURS

        for day in range(num_days):
            ds = day * 24
            de = ds + 24

            day_surplus = residual_surplus[ds:de]
            day_gap = residual_gap[ds:de]

            total_surplus_day = day_surplus.sum()
            total_gap_day = day_gap.sum()

            max_from_charge = total_surplus_day * BATTERY_EFFICIENCY
            actual_dispatch = min(daily_dispatch_target, max_from_charge, total_gap_day)
            if actual_dispatch <= 0:
                continue

            required_charge = actual_dispatch / BATTERY_EFFICIENCY

            # Charge from largest surpluses
            sorted_idx = np.argsort(-day_surplus)
            remaining_charge = required_charge
            for idx in sorted_idx:
                if remaining_charge <= 0 or day_surplus[idx] <= 0:
                    break
                amt = min(float(day_surplus[idx]), batt_power_rating, remaining_charge)
                battery_charge_profile[ds + idx] = amt
                residual_surplus[ds + idx] -= amt
                remaining_charge -= amt

            actual_charge = required_charge - remaining_charge
            ach_dispatch = min(actual_dispatch, actual_charge * BATTERY_EFFICIENCY)

            # Dispatch to largest gaps
            sorted_idx = np.argsort(-day_gap)
            remaining_dispatch = ach_dispatch
            for idx in sorted_idx:
                if remaining_dispatch <= 0 or day_gap[idx] <= 0:
                    break
                amt = min(float(day_gap[idx]), batt_power_rating, remaining_dispatch)
                battery_dispatch_profile[ds + idx] = amt
                residual_gap[ds + idx] -= amt
                remaining_dispatch -= amt

    # Phase 2: LDES multi-day dispatch on remaining gaps
    if ldes_dispatch_pct > 0:
        total_demand_energy = demand_arr.sum()
        ldes_energy_capacity = total_demand_energy * (24.0 / H)
        ldes_power_rating = ldes_energy_capacity / LDES_DURATION_HOURS
        scale = ldes_dispatch_pct / 10.0

        state_of_charge = 0.0
        window_hours = LDES_WINDOW_DAYS * 24

        num_windows = (H + window_hours - 1) // window_hours
        for w in range(num_windows):
            w_start = w * window_hours
            w_end = min(w_start + window_hours, H)

            w_surplus = residual_surplus[w_start:w_end]
            w_gap = residual_gap[w_start:w_end]

            # Charge from surplus
            surplus_indices = np.argsort(-w_surplus)
            for idx in surplus_indices:
                if w_surplus[idx] <= 0:
                    break
                space = (ldes_energy_capacity * scale) - state_of_charge
                if space <= 0:
                    break
                charge_amt = min(float(w_surplus[idx]), ldes_power_rating * scale, space)
                if charge_amt > 0:
                    ldes_charge_profile[w_start + idx] = charge_amt
                    residual_surplus[w_start + idx] -= charge_amt
                    state_of_charge += charge_amt

            # Discharge to gaps
            gap_indices = np.argsort(-w_gap)
            for idx in gap_indices:
                if w_gap[idx] <= 0:
                    break
                available = state_of_charge * LDES_EFFICIENCY
                if available <= 1e-12:
                    break
                dispatch_amt = min(float(w_gap[idx]), ldes_power_rating * scale, available)
                if dispatch_amt > 0:
                    ldes_dispatch_profile[w_start + idx] = dispatch_amt
                    residual_gap[w_start + idx] -= dispatch_amt
                    state_of_charge -= dispatch_amt / LDES_EFFICIENCY
                    state_of_charge = max(0.0, state_of_charge)

    # Compute final score
    total_matched = 0.0
    total_demand = 0.0
    for h in range(H):
        d = hourly_detail[h]
        batt_disp = battery_dispatch_profile[h]
        ldes_disp = ldes_dispatch_profile[h]
        new_matched = d['matched'] + min(d['gap'], batt_disp + ldes_disp)
        total_matched += new_matched
        total_demand += d['demand']

    score = total_matched / total_demand if total_demand > 0 else 0.0
    return (score, hourly_detail,
            battery_dispatch_profile, battery_charge_profile,
            ldes_dispatch_profile, ldes_charge_profile)


# ══════════════════════════════════════════════════════════════════════════════
# 5D COMBINATION GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

# Edge case seed mixes — injected into Phase 1 to ensure extreme but potentially
# optimal mixes survive coarse pruning. Under low-cost renewables, a solar-dominant
# mix with heavy storage may be globally cheapest for <90% targets. Under low-cost
# firm generation, clean firm or CCS-dominant mixes may win. The coarse grid
# generates these combos but may not rank them highly enough to survive to Phase 2/3
# under all 324 cost scenarios. Seeds guarantee they're always evaluated.
#
# Format: {clean_firm, solar, wind, ccs_ccgt, hydro} — must sum to 100%
# Hydro-dependent seeds filtered at runtime by regional hydro cap.
EDGE_CASE_SEEDS = [
    # Low-cost renewable: solar-dominant
    {'clean_firm': 10, 'solar': 70, 'wind': 10, 'ccs_ccgt': 0, 'hydro': 10},
    {'clean_firm': 5,  'solar': 75, 'wind': 10, 'ccs_ccgt': 0, 'hydro': 10},
    {'clean_firm': 10, 'solar': 60, 'wind': 20, 'ccs_ccgt': 0, 'hydro': 10},
    # Low-cost renewable: wind-dominant
    {'clean_firm': 10, 'solar': 10, 'wind': 70, 'ccs_ccgt': 0, 'hydro': 10},
    {'clean_firm': 5,  'solar': 10, 'wind': 75, 'ccs_ccgt': 0, 'hydro': 10},
    {'clean_firm': 10, 'solar': 20, 'wind': 60, 'ccs_ccgt': 0, 'hydro': 10},
    # Low-cost renewable: balanced solar+wind
    {'clean_firm': 10, 'solar': 40, 'wind': 40, 'ccs_ccgt': 0, 'hydro': 10},
    {'clean_firm': 5,  'solar': 45, 'wind': 45, 'ccs_ccgt': 0, 'hydro': 5},
    # Low-cost clean firm: nuclear/geothermal dominant
    {'clean_firm': 70, 'solar': 10, 'wind': 10, 'ccs_ccgt': 0, 'hydro': 10},
    {'clean_firm': 80, 'solar': 10, 'wind': 10, 'ccs_ccgt': 0, 'hydro': 0},
    {'clean_firm': 60, 'solar': 15, 'wind': 15, 'ccs_ccgt': 0, 'hydro': 10},
    # Low-cost firm: combined clean firm + CCS
    {'clean_firm': 40, 'solar': 10, 'wind': 10, 'ccs_ccgt': 30, 'hydro': 10},
    {'clean_firm': 30, 'solar': 10, 'wind': 10, 'ccs_ccgt': 40, 'hydro': 10},
    # CCS-dominant (cheap firm gen + favorable geology)
    {'clean_firm': 20, 'solar': 15, 'wind': 15, 'ccs_ccgt': 50, 'hydro': 0},
    {'clean_firm': 10, 'solar': 10, 'wind': 10, 'ccs_ccgt': 60, 'hydro': 10},
    # High-hydro regions (NYISO, CAISO, NEISO) — hydro + firm
    {'clean_firm': 30, 'solar': 10, 'wind': 10, 'ccs_ccgt': 10, 'hydro': 40},
    {'clean_firm': 20, 'solar': 20, 'wind': 10, 'ccs_ccgt': 10, 'hydro': 40},
]


def generate_combinations(hydro_cap, step=5, max_single=80):
    """
    Generate all valid resource mix combinations that sum to 100%.
    Resources: clean_firm, solar, wind, ccs_ccgt, hydro
    Hydro capped by region. No single resource exceeds max_single%.
    """
    combos = []
    for cf in range(0, min(max_single + 1, 101), step):
        for sol in range(0, min(max_single + 1, 101 - cf), step):
            for wnd in range(0, min(max_single + 1, 101 - cf - sol), step):
                for ccs in range(0, min(max_single + 1, 101 - cf - sol - wnd), step):
                    hyd = 100 - cf - sol - wnd - ccs
                    if hyd >= 0 and hyd <= hydro_cap and hyd <= max_single:
                        combos.append({
                            'clean_firm': cf, 'solar': sol, 'wind': wnd,
                            'ccs_ccgt': ccs, 'hydro': hyd
                        })
    return combos


def get_seed_combos(hydro_cap):
    """
    Return edge case seed mixes valid for this region's hydro cap.
    Filters out seeds where hydro exceeds regional cap.
    """
    valid = []
    seen = set()
    for seed in EDGE_CASE_SEEDS:
        if seed['hydro'] > hydro_cap:
            continue
        key = tuple(seed[rt] for rt in RESOURCE_TYPES)
        if key not in seen:
            seen.add(key)
            valid.append(dict(seed))
    return valid


def generate_combinations_around(base_combo, hydro_cap, step=1, radius=2):
    """
    Generate combinations in a neighborhood around base_combo with given step and radius.
    """
    combos = []
    seen = set()
    ranges = {}
    for rtype in RESOURCE_TYPES:
        base = base_combo[rtype]
        cap = hydro_cap if rtype == 'hydro' else 100
        low = max(0, base - radius * step)
        high = min(cap, base + radius * step)
        ranges[rtype] = list(range(low, high + 1, step))

    for cf in ranges['clean_firm']:
        for sol in ranges['solar']:
            for wnd in ranges['wind']:
                for ccs in ranges['ccs_ccgt']:
                    hyd = 100 - cf - sol - wnd - ccs
                    if hyd < 0 or hyd > hydro_cap:
                        continue
                    key = (cf, sol, wnd, ccs, hyd)
                    if key in seen:
                        continue
                    seen.add(key)
                    combos.append({
                        'clean_firm': cf, 'solar': sol, 'wind': wnd,
                        'ccs_ccgt': ccs, 'hydro': hyd
                    })
    return combos


# ══════════════════════════════════════════════════════════════════════════════
# STORAGE OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════════

def find_optimal_storage(demand_arr, supply_matrix, mix_fractions, procurement_factor, hydro_cap):
    """
    Find the optimal battery + LDES dispatch percentages that maximize hourly matching.
    Sweep battery from 0-30% and LDES from 0-20%, then refine.
    """
    best_score = fast_hourly_score(demand_arr, supply_matrix, mix_fractions, procurement_factor)
    best_batt = 0
    best_ldes = 0

    # Coarse sweep
    for bp in range(0, 35, 5):
        for lp in range(0, 25, 5):
            if bp == 0 and lp == 0:
                continue
            score = fast_score_with_both_storage(
                demand_arr, supply_matrix, mix_fractions, procurement_factor, bp, lp
            )
            if score > best_score:
                best_score = score
                best_batt = bp
                best_ldes = lp

    # Fine sweep around best
    fine_best_score = best_score
    fine_best_batt = best_batt
    fine_best_ldes = best_ldes
    for bp in range(max(0, best_batt - 4), best_batt + 5, 2):
        for lp in range(max(0, best_ldes - 4), best_ldes + 5, 2):
            score = fast_score_with_both_storage(
                demand_arr, supply_matrix, mix_fractions, procurement_factor, bp, lp
            )
            if score > fine_best_score:
                fine_best_score = score
                fine_best_batt = bp
                fine_best_ldes = lp

    return fine_best_score, fine_best_batt, fine_best_ldes


# ══════════════════════════════════════════════════════════════════════════════
# CO2 ABATEMENT
# ══════════════════════════════════════════════════════════════════════════════

def compute_co2_abatement(iso, resource_pcts, procurement_pct, hourly_match_score,
                          battery_dispatch_pct, ldes_dispatch_pct,
                          emission_rates, demand_total_mwh,
                          demand_norm=None, supply_profiles=None, fossil_mix=None):
    """
    Compute CO2 abatement using hourly fossil-fuel emission rates.

    Methodology:
      1. Build hourly emission rate from eGRID per-fuel rates × EIA hourly fossil mix
         emission_rate[h] = coal_share[h]×coal_rate + gas_share[h]×gas_rate + oil_share[h]×oil_rate
      2. Reconstruct hourly clean supply → fossil displacement at each hour
      3. CO2_abated = Σ_h fossil_displaced[h] × emission_rate[h]
      4. CCS-CCGT: partial credit (rate[h] - 0.037 tCO2/MWh residual)

    Falls back to flat regional rate if hourly data not provided (backward compat).
    """
    regional_data = emission_rates.get(iso, {})

    # Per-fuel emission rates from eGRID (lb/MWh → metric tons/MWh)
    coal_rate = regional_data.get('coal_co2_lb_per_mwh', 0.0) / 2204.62
    gas_rate = regional_data.get('gas_co2_lb_per_mwh', 0.0) / 2204.62
    oil_rate = regional_data.get('oil_co2_lb_per_mwh', 0.0) / 2204.62

    # ── Hourly emission rate calculation ──
    use_hourly = (demand_norm is not None and supply_profiles is not None
                  and fossil_mix is not None)

    if use_hourly:
        # Build hourly emission rates from fossil mix shares
        iso_fossil = fossil_mix.get(iso, {})
        year_data = iso_fossil.get(DATA_YEAR, iso_fossil.get('2024', {}))

        coal_shares = np.array(year_data.get('coal_share', [0.0] * H)[:H], dtype=np.float64)
        gas_shares = np.array(year_data.get('gas_share', [1.0] * H)[:H], dtype=np.float64)
        oil_shares = np.array(year_data.get('oil_share', [0.0] * H)[:H], dtype=np.float64)

        # Pad to H if shorter
        for arr_name in ['coal_shares', 'gas_shares', 'oil_shares']:
            arr = locals()[arr_name]
            if len(arr) < H:
                locals()[arr_name] = np.pad(arr, (0, H - len(arr)), mode='edge')

        hourly_rates = coal_shares * coal_rate + gas_shares * gas_rate + oil_shares * oil_rate

        # Reconstruct hourly clean supply and fossil displacement
        procurement_factor = procurement_pct / 100.0
        demand_arr = np.array(demand_norm[:H], dtype=np.float64)
        supply_total = np.zeros(H, dtype=np.float64)
        ccs_supply = np.zeros(H, dtype=np.float64)

        for rtype in RESOURCE_TYPES:
            pct = resource_pcts.get(rtype, 0)
            if pct <= 0:
                continue
            profile = np.array(supply_profiles[rtype][:H], dtype=np.float64)
            contribution = procurement_factor * (pct / 100.0) * profile
            supply_total += contribution
            if rtype == 'ccs_ccgt':
                ccs_supply = contribution.copy()

        # Simplified storage effect (proportional adjustment)
        storage_boost = (battery_dispatch_pct + ldes_dispatch_pct) / 100.0
        # Storage shifts supply from surplus to gap hours; approximate by
        # adding storage fraction uniformly to matched fraction
        total_clean = supply_total  # Storage is already accounted in match score

        fossil_displaced = np.minimum(demand_arr, total_clean)
        matched_mwh = np.sum(fossil_displaced) * demand_total_mwh

        # Non-CCS clean displacement
        ccs_matched = np.minimum(fossil_displaced, ccs_supply)
        non_ccs_matched = fossil_displaced - ccs_matched

        # CO₂ from non-CCS: full credit at hourly rate
        co2_clean = np.sum(non_ccs_matched * hourly_rates) * demand_total_mwh
        # CO₂ from CCS: partial credit
        ccs_credit = np.maximum(0.0, hourly_rates - CCS_RESIDUAL_EMISSION_RATE)
        co2_ccs = np.sum(ccs_matched * ccs_credit) * demand_total_mwh

        # Storage displacement (shifts surplus→gap, displaces fossil at gap hours)
        storage_mwh = demand_total_mwh * storage_boost
        # Weighted avg rate at gap hours (approximation)
        gap_mask = demand_arr > total_clean
        if np.any(gap_mask):
            gap_weighted_rate = np.average(hourly_rates[gap_mask],
                                           weights=np.maximum(0, demand_arr[gap_mask] - total_clean[gap_mask]))
        else:
            gap_weighted_rate = np.mean(hourly_rates)
        co2_storage = storage_mwh * gap_weighted_rate

        total_abated = co2_clean + co2_ccs + co2_storage
        matched_mwh_total = matched_mwh + storage_mwh
        co2_rate = total_abated / matched_mwh_total if matched_mwh_total > 0 else 0
        weighted_avg_rate = np.average(hourly_rates, weights=fossil_displaced) \
            if np.sum(fossil_displaced) > 0 else np.mean(hourly_rates)

        return {
            'marginal_emission_rate_tons_per_mwh': round(float(weighted_avg_rate), 4),
            'hourly_emission_rate_avg': round(float(np.mean(hourly_rates)), 4),
            'hourly_emission_rate_min': round(float(np.min(hourly_rates)), 4),
            'hourly_emission_rate_max': round(float(np.max(hourly_rates)), 4),
            'total_co2_abated_tons': round(float(total_abated), 0),
            'co2_rate_per_mwh': round(float(co2_rate), 4),
            'methodology': 'hourly_fossil_fuel_emission_rates',
        }

    # ── Fallback: flat regional rate (backward compatibility) ──
    fossil_co2_lb_per_mwh = regional_data.get('fossil_co2_lb_per_mwh', 900.0)
    marginal_rate_tons = fossil_co2_lb_per_mwh / 2204.62

    procurement_factor = procurement_pct / 100.0
    matched_fraction = hourly_match_score / 100.0
    matched_mwh = demand_total_mwh * matched_fraction

    resource_co2 = {}
    total_abated = 0.0

    for rtype in RESOURCE_TYPES:
        pct = resource_pcts.get(rtype, 0)
        if pct <= 0:
            resource_co2[rtype] = {'mwh': 0, 'co2_abated_tons': 0}
            continue

        resource_mwh = matched_mwh * (pct / 100.0)

        if rtype == 'ccs_ccgt':
            abated = resource_mwh * (marginal_rate_tons - CCS_RESIDUAL_EMISSION_RATE)
        else:
            abated = resource_mwh * marginal_rate_tons

        abated = max(0, abated)
        resource_co2[rtype] = {
            'mwh': round(resource_mwh, 0),
            'co2_abated_tons': round(abated, 0),
        }
        total_abated += abated

    storage_mwh = demand_total_mwh * ((battery_dispatch_pct + ldes_dispatch_pct) / 100.0)
    storage_abated = storage_mwh * marginal_rate_tons
    resource_co2['battery'] = {
        'mwh': round(demand_total_mwh * battery_dispatch_pct / 100.0, 0),
        'co2_abated_tons': round(demand_total_mwh * battery_dispatch_pct / 100.0 * marginal_rate_tons, 0),
    }
    resource_co2['ldes'] = {
        'mwh': round(demand_total_mwh * ldes_dispatch_pct / 100.0, 0),
        'co2_abated_tons': round(demand_total_mwh * ldes_dispatch_pct / 100.0 * marginal_rate_tons, 0),
    }
    total_abated += storage_abated

    co2_rate = total_abated / matched_mwh if matched_mwh > 0 else 0

    return {
        'marginal_emission_rate_tons_per_mwh': round(marginal_rate_tons, 4),
        'total_co2_abated_tons': round(total_abated, 0),
        'co2_rate_per_mwh': round(co2_rate, 4),
        'resource_breakdown': resource_co2,
        'methodology': 'flat_regional_rate',
    }


# ══════════════════════════════════════════════════════════════════════════════
# COST COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_costs(iso, resource_pcts, procurement_pct, battery_dispatch_pct, ldes_dispatch_pct,
                  hourly_match_score, demand_norm, supply_profiles):
    """
    Compute blended cost of energy and incremental cost above baseline.

    Cost model:
      - Resources up to existing grid mix share -> priced at wholesale market rate
      - Resources above grid mix share -> priced at new-build LCOE + transmission adder
      - CCS-CCGT: no existing share, all new-build priced
      - Hydro: always at wholesale (existing resource, no new-build)
      - Battery storage -> priced at battery LCOS + tx adder for dispatched MWh
      - LDES storage -> priced at LDES LCOS + tx adder for dispatched MWh

    Returns dict with cost metrics in $/MWh (per MWh of demand served).
    """
    wholesale = WHOLESALE_PRICES[iso]
    lcoe = REGIONAL_LCOE[iso]
    tx = TRANSMISSION_ADDERS[iso]
    grid_shares = GRID_MIX_SHARES[iso]

    procurement_factor = procurement_pct / 100.0

    resource_costs = {}
    total_cost_per_demand = 0.0

    for rtype in RESOURCE_TYPES:
        pct = resource_pcts.get(rtype, 0)
        if pct <= 0:
            resource_costs[rtype] = {'existing_share': 0, 'new_share': 0, 'cost': 0}
            continue

        resource_fraction = procurement_factor * (pct / 100.0)
        resource_pct_of_demand = resource_fraction * 100.0

        existing_share = grid_shares.get(rtype, 0)
        existing_pct = min(resource_pct_of_demand, existing_share)
        new_pct = max(0, resource_pct_of_demand - existing_share)

        if rtype == 'hydro':
            # Hydro: always at wholesale, no new-build, no transmission adder
            cost_per_demand = resource_pct_of_demand / 100.0 * wholesale
        else:
            # Existing portion at wholesale, new-build at LCOE + transmission
            new_build_cost = lcoe.get(rtype, 0) + tx.get(rtype, 0)
            cost_per_demand = (existing_pct / 100.0 * wholesale) + (new_pct / 100.0 * new_build_cost)

        resource_costs[rtype] = {
            'total_pct_of_demand': round(resource_pct_of_demand, 1),
            'existing_pct': round(existing_pct, 1),
            'new_pct': round(new_pct, 1),
            'cost_per_demand_mwh': round(cost_per_demand, 2),
        }
        total_cost_per_demand += cost_per_demand

    # Battery storage cost: pay for dispatched MWh at battery LCOS + tx
    battery_cost_rate = lcoe['battery'] + tx.get('battery', 0)
    battery_cost_per_demand = (battery_dispatch_pct / 100.0) * battery_cost_rate
    resource_costs['battery'] = {
        'dispatch_pct': round(battery_dispatch_pct, 1),
        'cost_per_demand_mwh': round(battery_cost_per_demand, 2),
    }
    total_cost_per_demand += battery_cost_per_demand

    # LDES storage cost: pay for dispatched MWh at LDES LCOS + tx
    ldes_cost_rate = lcoe['ldes'] + tx.get('ldes', 0)
    ldes_cost_per_demand = (ldes_dispatch_pct / 100.0) * ldes_cost_rate
    resource_costs['ldes'] = {
        'dispatch_pct': round(ldes_dispatch_pct, 1),
        'cost_per_demand_mwh': round(ldes_cost_per_demand, 2),
    }
    total_cost_per_demand += ldes_cost_per_demand

    # Effective cost per useful MWh (accounting for curtailment)
    matched_fraction = hourly_match_score / 100.0 if hourly_match_score > 0 else 1.0
    effective_cost_per_useful_mwh = total_cost_per_demand / matched_fraction

    baseline_cost = wholesale
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


# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERIZED COST COMPUTATION (for paired toggle sensitivity scenarios)
# ══════════════════════════════════════════════════════════════════════════════

# Wholesale price adjustments by region and fossil fuel price level
# Based on regional fossil fuel generation share and fuel price sensitivity
# Gas heat rate ~7 MMBtu/MWh × price delta, weighted by regional fossil share
WHOLESALE_FUEL_ADJUSTMENTS = {
    'CAISO': {'Low': -5, 'Medium': 0, 'High': 10},   # ~40% gas generation
    'ERCOT': {'Low': -7, 'Medium': 0, 'High': 12},   # ~50% gas, most sensitive
    'PJM':   {'Low': -6, 'Medium': 0, 'High': 11},   # ~40% gas + coal mix
    'NYISO': {'Low': -4, 'Medium': 0, 'High': 8},    # ~35% gas, more nuclear
    'NEISO': {'Low': -4, 'Medium': 0, 'High': 8},    # ~35% gas, more nuclear
}


def compute_costs_parameterized(iso, resource_pcts, procurement_pct, battery_dispatch_pct,
                                 ldes_dispatch_pct, hourly_match_score,
                                 renewable_gen_level, firm_gen_level, storage_level,
                                 fossil_fuel_level, transmission_level):
    """
    Compute costs for a specific paired toggle scenario on a cached resource mix.

    Instead of reading from global Medium constants, uses the full L/M/H cost tables
    mapped through the paired toggle groups.
    """
    # Map paired toggle levels to individual resource LCOEs
    lcoe_map = {
        'solar': FULL_LCOE_TABLES['solar'][renewable_gen_level][iso],
        'wind': FULL_LCOE_TABLES['wind'][renewable_gen_level][iso],
        'clean_firm': FULL_LCOE_TABLES['clean_firm'][firm_gen_level][iso],
        'ccs_ccgt': FULL_LCOE_TABLES['ccs_ccgt'][firm_gen_level][iso],
        'battery': FULL_LCOE_TABLES['battery'][storage_level][iso],
        'ldes': FULL_LCOE_TABLES['ldes'][storage_level][iso],
        'hydro': 0,
    }

    # Map transmission level to per-resource adders
    tx_map = {rtype: FULL_TRANSMISSION_TABLES[rtype][transmission_level][iso]
              for rtype in FULL_TRANSMISSION_TABLES}

    # Adjust wholesale price based on fossil fuel level
    wholesale = WHOLESALE_PRICES[iso] + WHOLESALE_FUEL_ADJUSTMENTS[iso][fossil_fuel_level]
    wholesale = max(5, wholesale)  # Floor at $5/MWh

    grid_shares = GRID_MIX_SHARES[iso]
    procurement_factor = procurement_pct / 100.0

    total_cost_per_demand = 0.0

    for rtype in RESOURCE_TYPES:
        pct = resource_pcts.get(rtype, 0)
        if pct <= 0:
            continue

        resource_fraction = procurement_factor * (pct / 100.0)
        resource_pct_of_demand = resource_fraction * 100.0

        existing_share = grid_shares.get(rtype, 0)
        existing_pct = min(resource_pct_of_demand, existing_share)
        new_pct = max(0, resource_pct_of_demand - existing_share)

        if rtype == 'hydro':
            cost_per_demand = resource_pct_of_demand / 100.0 * wholesale
        else:
            new_build_cost = lcoe_map.get(rtype, 0) + tx_map.get(rtype, 0)
            cost_per_demand = (existing_pct / 100.0 * wholesale) + (new_pct / 100.0 * new_build_cost)

        total_cost_per_demand += cost_per_demand

    # Battery storage cost
    battery_cost_rate = lcoe_map['battery'] + tx_map.get('battery', 0)
    total_cost_per_demand += (battery_dispatch_pct / 100.0) * battery_cost_rate

    # LDES storage cost
    ldes_cost_rate = lcoe_map['ldes'] + tx_map.get('ldes', 0)
    total_cost_per_demand += (ldes_dispatch_pct / 100.0) * ldes_cost_rate

    # Effective cost per useful MWh
    matched_fraction = hourly_match_score / 100.0 if hourly_match_score > 0 else 1.0
    effective_cost = total_cost_per_demand / matched_fraction
    incremental = effective_cost - wholesale

    return {
        'effective_cost': round(effective_cost, 2),
        'incremental': round(incremental, 2),
        'total_cost': round(total_cost_per_demand, 2),
        'wholesale': round(wholesale, 2),
    }


def precompute_sensitivity_costs(iso, result, emission_rates, demand_total_mwh):
    """
    Pre-compute costs for all 324 paired toggle combinations on a cached resource mix.

    Returns dict keyed by scenario string: "{renew}_{firm}_{storage}_{fuel}_{tx}"
    where each level is L/M/H (or N for None on transmission).
    """
    resource_pcts = result['resource_mix']
    procurement_pct = result['procurement_pct']
    battery_pct = result['battery_dispatch_pct']
    ldes_pct = result['ldes_dispatch_pct']
    match_score = result['hourly_match_score']

    level_keys = {'Low': 'L', 'Medium': 'M', 'High': 'H', 'None': 'N'}
    gen_levels = ['Low', 'Medium', 'High']
    tx_levels = ['None', 'Low', 'Medium', 'High']

    scenarios = {}

    for renew in gen_levels:
        for firm in gen_levels:
            for storage in gen_levels:
                for fuel in gen_levels:
                    for tx in tx_levels:
                        key = f"{level_keys[renew]}{level_keys[firm]}{level_keys[storage]}_{level_keys[fuel]}_{level_keys[tx]}"
                        cost_data = compute_costs_parameterized(
                            iso, resource_pcts, procurement_pct,
                            battery_pct, ldes_pct, match_score,
                            renew, firm, storage, fuel, tx
                        )
                        scenarios[key] = cost_data

    return scenarios


# ══════════════════════════════════════════════════════════════════════════════
# PEAK GAP COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_peak_gap(demand_norm, supply_profiles, resource_pcts, procurement_pct,
                     battery_dispatch_pct, ldes_dispatch_pct, anomaly_hours):
    """Compute peak single-hour gap % after both storage types."""
    (score, hourly_detail,
     batt_dispatch, batt_charge,
     ldes_dispatch, ldes_charge) = compute_hourly_matching_detailed(
        demand_norm, supply_profiles, resource_pcts, procurement_pct,
        battery_dispatch_pct, ldes_dispatch_pct
    )
    peak_gap = 0.0
    for h in range(H):
        if h in anomaly_hours:
            continue
        d = hourly_detail[h]
        disp = batt_dispatch[h] + ldes_dispatch[h]
        residual_gap = max(0, d['gap'] - disp)
        if d['demand'] > 0:
            gap_pct = (residual_gap / d['demand']) * 100
            if gap_pct > peak_gap:
                peak_gap = gap_pct
    return round(peak_gap, 1)


# ══════════════════════════════════════════════════════════════════════════════
# COMPRESSED DAY PROFILE
# ══════════════════════════════════════════════════════════════════════════════

def compute_compressed_day(demand_norm, supply_profiles, resource_pcts, procurement_pct,
                           battery_dispatch_pct, ldes_dispatch_pct):
    """Build compressed day profile for visualization with all 7 resource types."""
    procurement_factor = procurement_pct / 100.0

    (_, hourly_detail,
     batt_dispatch_profile, batt_charge_profile,
     ldes_dispatch_profile, ldes_charge_profile) = compute_hourly_matching_detailed(
        demand_norm, supply_profiles, resource_pcts, procurement_pct,
        battery_dispatch_pct, ldes_dispatch_pct
    )

    demand_sums = [0.0] * 24
    supply_by_type = {rt: [0.0]*24 for rt in RESOURCE_TYPES}
    gap_sums = [0.0] * 24
    surplus_sums = [0.0] * 24
    batt_dispatch_sums = [0.0] * 24
    batt_charge_sums = [0.0] * 24
    ldes_dispatch_sums = [0.0] * 24
    ldes_charge_sums = [0.0] * 24

    for h in range(H):
        hod = h % 24
        d = hourly_detail[h]
        demand_sums[hod] += d['demand']

        for rtype, pct in resource_pcts.items():
            if pct <= 0:
                continue
            type_supply = procurement_factor * (pct / 100.0) * supply_profiles[rtype][h]
            supply_by_type[rtype][hod] += type_supply

        batt_dispatch_sums[hod] += batt_dispatch_profile[h]
        batt_charge_sums[hod] += batt_charge_profile[h]
        ldes_dispatch_sums[hod] += ldes_dispatch_profile[h]
        ldes_charge_sums[hod] += ldes_charge_profile[h]

    # Match supply to demand per hour-of-day
    # Match order: clean_firm, ccs_ccgt, hydro, wind, solar (baseload first, then variable)
    match_order = ['clean_firm', 'ccs_ccgt', 'hydro', 'wind', 'solar']
    cut_order = list(reversed(match_order))  # solar curtailed first

    matched_by_type = {rt: [0.0]*24 for rt in RESOURCE_TYPES}
    surplus_by_type = {rt: [0.0]*24 for rt in RESOURCE_TYPES}
    matched_by_type['battery'] = [0.0]*24
    matched_by_type['ldes'] = [0.0]*24

    for hod in range(24):
        remaining = demand_sums[hod]

        # Match generation resources
        for rtype in match_order:
            avail = supply_by_type[rtype][hod]
            matched = min(remaining, avail)
            matched_by_type[rtype][hod] = matched
            surplus_by_type[rtype][hod] = avail - matched
            remaining -= matched

        # Battery dispatch fills remaining gap
        batt_disp = min(remaining, batt_dispatch_sums[hod])
        matched_by_type['battery'][hod] = batt_disp
        remaining -= batt_disp

        # LDES dispatch fills further remaining gap
        ldes_disp = min(remaining, ldes_dispatch_sums[hod])
        matched_by_type['ldes'][hod] = ldes_disp
        remaining -= ldes_disp

        gap_sums[hod] = max(0, remaining)

        # Reduce surplus by battery charging
        rem_charge = batt_charge_sums[hod]
        for rtype in cut_order:
            if rem_charge <= 0:
                break
            absorb = min(surplus_by_type[rtype][hod], rem_charge)
            surplus_by_type[rtype][hod] -= absorb
            rem_charge -= absorb

        # Reduce surplus by LDES charging
        rem_charge = ldes_charge_sums[hod]
        for rtype in cut_order:
            if rem_charge <= 0:
                break
            absorb = min(surplus_by_type[rtype][hod], rem_charge)
            surplus_by_type[rtype][hod] -= absorb
            rem_charge -= absorb

    # Total surplus per hour-of-day
    for hod in range(24):
        surplus_sums[hod] = sum(surplus_by_type[rt][hod] for rt in RESOURCE_TYPES)

    all_match_types = list(RESOURCE_TYPES) + ['battery', 'ldes']

    return {
        'demand': [round(v, 4) for v in demand_sums],
        'matched': {rt: [round(v, 4) for v in matched_by_type[rt]] for rt in all_match_types},
        'surplus': {rt: [round(v, 4) for v in surplus_by_type[rt]] for rt in RESOURCE_TYPES},
        'gap': [round(v, 4) for v in gap_sums],
        'battery_charge': [round(v, 4) for v in batt_charge_sums],
        'ldes_charge': [round(v, 4) for v in ldes_charge_sums],
        'total_surplus': [round(v, 4) for v in surplus_sums],
    }


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZATION AT PROCUREMENT LEVEL (for sweep chart)
# ══════════════════════════════════════════════════════════════════════════════

def optimize_at_procurement_level(iso, demand_norm, supply_profiles, procurement_pct, hydro_cap,
                                  np_profiles=None):
    """
    Find the resource mix that maximizes hourly matching at a given procurement level.
    Two-phase: coarse (5%) -> fine (1%), then optimize both storage types.
    """
    pf = procurement_pct / 100.0

    if np_profiles:
        demand_arr, _, supply_matrix = np_profiles

        # Coarse search with 10% step for 5D
        combos = generate_combinations(hydro_cap, step=10)
        best_score = -1
        best_combo = None

        for combo in combos:
            mix_fracs = np.array([combo[rt] / 100.0 for rt in RESOURCE_TYPES])
            score = fast_hourly_score(demand_arr, supply_matrix, mix_fracs, pf)
            if score > best_score:
                best_score = score
                best_combo = combo

        # Medium refinement (5% step, radius 10)
        if best_combo:
            combos_med = generate_combinations_around(best_combo, hydro_cap, step=5, radius=2)
            for combo in combos_med:
                mix_fracs = np.array([combo[rt] / 100.0 for rt in RESOURCE_TYPES])
                score = fast_hourly_score(demand_arr, supply_matrix, mix_fracs, pf)
                if score > best_score:
                    best_score = score
                    best_combo = combo

        # Fine refinement (1% step, radius 3)
        if best_combo:
            combos_fine = generate_combinations_around(best_combo, hydro_cap, step=1, radius=3)
            for combo in combos_fine:
                mix_fracs = np.array([combo[rt] / 100.0 for rt in RESOURCE_TYPES])
                score = fast_hourly_score(demand_arr, supply_matrix, mix_fracs, pf)
                if score > best_score:
                    best_score = score
                    best_combo = combo

        # Optimize storage
        mix_fracs = np.array([best_combo[rt] / 100.0 for rt in RESOURCE_TYPES])
        best_score_ws, best_batt, best_ldes = find_optimal_storage(
            demand_arr, supply_matrix, mix_fracs, pf, hydro_cap
        )
    else:
        combos = generate_combinations(hydro_cap, step=10)
        best_score = -1
        best_combo = None

        demand_arr = np.array(demand_norm[:H], dtype=np.float64)
        supply_matrix = np.stack([np.array(supply_profiles[rt][:H], dtype=np.float64) for rt in RESOURCE_TYPES])

        for combo in combos:
            mix_fracs = np.array([combo[rt] / 100.0 for rt in RESOURCE_TYPES])
            score = fast_hourly_score(demand_arr, supply_matrix, mix_fracs, pf)
            if score > best_score:
                best_score = score
                best_combo = combo

        if best_combo:
            combos_fine = generate_combinations_around(best_combo, hydro_cap, step=1, radius=3)
            for combo in combos_fine:
                mix_fracs = np.array([combo[rt] / 100.0 for rt in RESOURCE_TYPES])
                score = fast_hourly_score(demand_arr, supply_matrix, mix_fracs, pf)
                if score > best_score:
                    best_score = score
                    best_combo = combo

        mix_fracs = np.array([best_combo[rt] / 100.0 for rt in RESOURCE_TYPES])
        best_score_ws, best_batt, best_ldes = find_optimal_storage(
            demand_arr, supply_matrix, mix_fracs, pf, hydro_cap
        )

    return {
        'procurement_pct': procurement_pct,
        'resource_mix': best_combo,
        'battery_dispatch_pct': round(best_batt, 1),
        'ldes_dispatch_pct': round(best_ldes, 1),
        'hourly_match_score': round(best_score_ws * 100, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# THRESHOLD OPTIMIZATION (3-phase: coarse -> medium -> fine)
# ══════════════════════════════════════════════════════════════════════════════

def optimize_for_threshold(iso, demand_norm, supply_profiles, threshold, hydro_cap,
                           emission_rates, demand_total_mwh,
                           cost_levels=None, score_cache=None,
                           resweep=False, seed_mixes=None,
                           procurement_bounds_override=None):
    """
    CO-OPTIMIZE cost and matching simultaneously for a given threshold target.
    Search across procurement levels AND resource mixes to find the CHEAPEST
    combination that meets or exceeds the threshold.

    3-phase approach adapted for 5D resource space:
      Phase 1: Coarse scan (10% mix steps, 10% procurement steps)
      Phase 2: Medium refinement (5% mix steps, 5% procurement steps)
      Phase 3: Fine-tune (1% mix, 2% procurement, refined storage)

    Args:
        cost_levels: Optional tuple (renewable_gen_level, firm_gen_level, storage_level,
                     fossil_fuel_level, transmission_level). If None, uses Medium costs.
        score_cache: Optional dict for caching matching scores across cost scenarios.
                     Matching scores are physics-based and cost-independent, so caching
                     them across cost scenarios is scientifically correct (not a shortcut).
        resweep: If True, use broader search parameters (finer Phase 1 step, more
                 Phase 2/3 candidates). Triggered by monotonicity violations.
        seed_mixes: Optional list of resource mix dicts to inject as seeds in Phase 1.
                    Used during re-sweep to seed with mixes that achieved better cost
                    at a higher threshold.
        procurement_bounds_override: Optional (min, max) tuple to override default
                                     procurement bounds. Used during re-sweep to expand
                                     the search range.

    Returns: best_result dict or None
    """
    target = threshold / 100.0
    # Adaptive procurement bounds per threshold (use override during re-sweep)
    if procurement_bounds_override:
        proc_min, proc_max = procurement_bounds_override
    else:
        proc_min, proc_max = PROCUREMENT_BOUNDS.get(threshold, (70, 310))
    storage_threshold = max(0.5, target - 0.10)  # Wider net to catch storage-dependent optima
    demand_arr, supply_arrs, supply_matrix = prepare_numpy_profiles(demand_norm, supply_profiles)

    if score_cache is None:
        score_cache = {}

    # ---- Cached scoring wrappers ----
    # Matching scores are physics (hourly profile alignment) — cost-independent.
    # Caching them across cost scenarios avoids redundant computation without
    # compromising scientific rigor.

    def c_hourly_score(mix_fracs_tuple, pf):
        key = ('h', mix_fracs_tuple, round(pf, 4))
        if key not in score_cache:
            score_cache[key] = fast_hourly_score(demand_arr, supply_matrix,
                                                  np.array(mix_fracs_tuple), pf)
        return score_cache[key]

    def c_battery_score(mix_fracs_tuple, pf, bp):
        key = ('b', mix_fracs_tuple, round(pf, 4), bp)
        if key not in score_cache:
            score_cache[key] = fast_score_with_battery(demand_arr, supply_matrix,
                                                        np.array(mix_fracs_tuple), pf, bp)
        return score_cache[key]

    def c_both_score(mix_fracs_tuple, pf, bp, lp):
        key = ('bl', mix_fracs_tuple, round(pf, 4), bp, lp)
        if key not in score_cache:
            score_cache[key] = fast_score_with_both_storage(demand_arr, supply_matrix,
                                                              np.array(mix_fracs_tuple), pf, bp, lp)
        return score_cache[key]

    # ---- Cost function: uses scenario-specific costs ----
    best_result = None
    best_cost = float('inf')

    def eval_cost(combo, proc, bp, lp, score):
        """Evaluate cost using the scenario's cost function. Cost drives mix selection."""
        if cost_levels:
            r_gen, f_gen, stor, fuel, tx = cost_levels
            cost_data = compute_costs_parameterized(
                iso, combo, proc, bp, lp, score * 100,
                r_gen, f_gen, stor, fuel, tx
            )
            return cost_data['effective_cost']
        else:
            cost_data = compute_costs(iso, combo, proc, bp, lp, score * 100,
                                       demand_norm, supply_profiles)
            return cost_data['effective_cost_per_useful_mwh']

    def update_best(combo, proc, bp, lp, score):
        """Update best result if this is the cheapest so far under current cost scenario."""
        nonlocal best_result, best_cost
        cost = eval_cost(combo, proc, bp, lp, score)
        if cost < best_cost:
            best_cost = cost
            best_result = {
                'procurement_pct': proc,
                'resource_mix': dict(combo),
                'battery_dispatch_pct': round(bp, 1),
                'ldes_dispatch_pct': round(lp, 1),
                'hourly_match_score': round(score * 100, 1),
            }
        return cost

    # ---- Phase 1: Coarse scan + edge case seeds ----
    # Re-sweep uses 5% step for finer exploration; normal uses 10%
    phase1_step = 5 if resweep else 10
    combos_10 = generate_combinations(hydro_cap, step=phase1_step)
    # Inject edge case seeds to guarantee extreme mixes survive pruning
    seeds = get_seed_combos(hydro_cap)
    seed_set = set(tuple(s[rt] for rt in RESOURCE_TYPES) for s in seeds)
    existing_set = set(tuple(c[rt] for rt in RESOURCE_TYPES) for c in combos_10)
    for seed in seeds:
        if tuple(seed[rt] for rt in RESOURCE_TYPES) not in existing_set:
            combos_10.append(seed)
    # Inject re-sweep seed mixes (from higher thresholds that achieved better cost)
    if seed_mixes:
        for smix in seed_mixes:
            key = tuple(smix[rt] for rt in RESOURCE_TYPES)
            if key not in existing_set:
                existing_set.add(key)
                combos_10.append(dict(smix))
    candidates = []

    for procurement_pct in range(proc_min, proc_max + 1, 10):
        pf = procurement_pct / 100.0
        for combo in combos_10:
            mix_fracs = tuple(combo[rt] / 100.0 for rt in RESOURCE_TYPES)
            score = c_hourly_score(mix_fracs, pf)

            if score >= target:
                cost = update_best(combo, procurement_pct, 0, 0, score)
                candidates.append((cost, combo, score, 0, 0, procurement_pct))
            elif score >= storage_threshold:
                # Battery only
                for bp in [5, 10, 15, 20, 25]:
                    score_ws = c_battery_score(mix_fracs, pf, bp)
                    if score_ws >= target:
                        cost = update_best(combo, procurement_pct, bp, 0, score_ws)
                        candidates.append((cost, combo, score_ws, bp, 0, procurement_pct))
                        break
                # LDES only (wind-heavy mixes benefit from multi-day shifting)
                for lp in [5, 10, 15, 20]:
                    score_ws = c_both_score(mix_fracs, pf, 0, lp)
                    if score_ws >= target:
                        cost = update_best(combo, procurement_pct, 0, lp, score_ws)
                        candidates.append((cost, combo, score_ws, 0, lp, procurement_pct))
                        break
                # Combined battery + LDES
                for bp in [5, 10, 15, 20]:
                    for lp in [5, 10, 15, 20]:
                        score_ws = c_both_score(mix_fracs, pf, bp, lp)
                        if score_ws >= target:
                            cost = update_best(combo, procurement_pct, bp, lp, score_ws)
                            candidates.append((cost, combo, score_ws, bp, lp, procurement_pct))
                            break
                    else:
                        continue
                    break

    if not candidates:
        return None

    # Select top candidates for refinement — ranked by THIS scenario's cost function
    # Re-sweep uses wider cost filter and more candidates for broader exploration
    phase2_cost_mult = 2.00 if resweep else 1.50
    phase2_top_n = 30 if resweep else 20
    candidates.sort(key=lambda x: x[0])
    top = [c for c in candidates if c[0] <= best_cost * phase2_cost_mult][:phase2_top_n]

    # ---- Phase 2: 5% refinement around top candidates ----
    phase2 = []
    seen = set()
    for _, combo, _, bp_base, lp_base, proc in top:
        for p_d in [-5, 0, 5]:
            p = proc + p_d
            if p < proc_min or p > proc_max:
                continue
            pf = p / 100.0

            neighborhood = generate_combinations_around(combo, hydro_cap, step=5, radius=1)
            for rcombo in neighborhood:
                key = (rcombo['clean_firm'], rcombo['solar'], rcombo['wind'],
                       rcombo['ccs_ccgt'], rcombo['hydro'], p)
                if key in seen:
                    continue
                seen.add(key)

                mix_fracs = tuple(rcombo[rt] / 100.0 for rt in RESOURCE_TYPES)

                score = c_hourly_score(mix_fracs, pf)
                best_bp = 0
                best_lp = 0

                if score < target:
                    for bp in [5, 10, 15, 20, 25]:
                        score_ws = c_battery_score(mix_fracs, pf, bp)
                        if score_ws >= target:
                            score = score_ws
                            best_bp = bp
                            break

                if score < target:
                    # Try LDES only
                    for lp in [5, 10, 15, 20]:
                        score_ws = c_both_score(mix_fracs, pf, 0, lp)
                        if score_ws >= target:
                            score = score_ws
                            best_bp = 0
                            best_lp = lp
                            break

                if score < target:
                    for bp in [5, 10, 15, 20]:
                        for lp in [5, 10, 15, 20]:
                            score_ws = c_both_score(mix_fracs, pf, bp, lp)
                            if score_ws >= target:
                                score = score_ws
                                best_bp = bp
                                best_lp = lp
                                break
                        else:
                            continue
                        break

                if score >= target:
                    cost = update_best(rcombo, p, best_bp, best_lp, score)
                    phase2.append((cost, rcombo, score, best_bp, best_lp, p))

    # ---- Phase 3: Fine-tune (1% mix, 2% procurement, refined storage) ----
    # Re-sweep uses wider filter and more finalists to avoid pruning the true optimum
    phase3_cost_mult = 1.20 if resweep else 1.10
    phase3_top_n = 15 if resweep else 8
    all_phase2 = phase2 if phase2 else top
    all_phase2.sort(key=lambda x: x[0])
    finalists = [c for c in all_phase2 if c[0] <= best_cost * phase3_cost_mult][:phase3_top_n]

    seen2 = set()
    for _, combo, _, bp_base, lp_base, proc in finalists:
        for p_d in range(-2, 3):
            p = proc + p_d
            if p < proc_min or p > proc_max:
                continue
            pf = p / 100.0

            fine_combos = generate_combinations_around(combo, hydro_cap, step=1, radius=2)
            for rcombo in fine_combos:
                key = (rcombo['clean_firm'], rcombo['solar'], rcombo['wind'],
                       rcombo['ccs_ccgt'], rcombo['hydro'], p)
                if key in seen2:
                    continue
                seen2.add(key)

                mix_fracs = tuple(rcombo[rt] / 100.0 for rt in RESOURCE_TYPES)

                score = c_hourly_score(mix_fracs, pf)
                best_bp = 0
                best_lp = 0
                best_score_here = score

                for bp in range(2, 22, 2):
                    score_ws = c_battery_score(mix_fracs, pf, bp)
                    if score_ws > best_score_here:
                        best_score_here = score_ws
                        best_bp = bp
                    if score_ws >= target and best_bp == bp:
                        break

                if best_score_here < target:
                    for lp in range(2, 22, 2):
                        score_ws = c_both_score(mix_fracs, pf, bp, lp)
                        if score_ws > best_score_here:
                            best_score_here = score_ws
                            best_lp = lp
                        if score_ws >= target and best_lp == lp:
                            break
                elif best_score_here >= target:
                    for lp in range(2, 12, 2):
                        score_ws = c_both_score(mix_fracs, pf, best_bp, lp)
                        if score_ws > best_score_here:
                            best_score_here = score_ws
                            best_lp = lp

                if best_score_here >= target:
                    update_best(rcombo, p, best_bp, best_lp, best_score_here)

    return best_result


# ══════════════════════════════════════════════════════════════════════════════
# PER-ISO WORKER (for multiprocessing)
# ══════════════════════════════════════════════════════════════════════════════

def generate_all_cost_scenarios():
    """
    Generate all 324 paired toggle cost scenarios.
    Returns list of (scenario_key, cost_levels_tuple) pairs.
    """
    level_keys = {'Low': 'L', 'Medium': 'M', 'High': 'H', 'None': 'N'}
    gen_levels = ['Low', 'Medium', 'High']
    tx_levels = ['None', 'Low', 'Medium', 'High']

    scenarios = []
    for renew in gen_levels:
        for firm in gen_levels:
            for storage in gen_levels:
                for fuel in gen_levels:
                    for tx in tx_levels:
                        key = f"{level_keys[renew]}{level_keys[firm]}{level_keys[storage]}_{level_keys[fuel]}_{level_keys[tx]}"
                        levels = (renew, firm, storage, fuel, tx)
                        scenarios.append((key, levels))
    return scenarios


ALL_COST_SCENARIOS = generate_all_cost_scenarios()
# Dict for O(1) lookup of cost_levels by scenario key (used during re-sweep)
COST_SCENARIO_MAP = {key: levels for key, levels in ALL_COST_SCENARIOS}


CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'checkpoints')


def save_checkpoint(iso, iso_results, phase='threshold', partial_threshold=None):
    """Save incremental checkpoint for an ISO's results.

    partial_threshold: optional dict with {threshold, scenarios} for mid-threshold saves.
    This allows resuming from within a threshold's 324-scenario loop.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f'{iso}_checkpoint.json')
    payload = {'iso': iso, 'phase': phase, 'results': iso_results}
    if partial_threshold:
        payload['partial_threshold'] = partial_threshold
    with open(ckpt_path, 'w') as f:
        json.dump(payload, f)
    print(f"    [checkpoint] Saved {iso} ({phase}) → {os.path.getsize(ckpt_path)/1024:.0f} KB")


def load_checkpoint(iso):
    """Load existing checkpoint for an ISO.

    Returns (iso_results, completed_thresholds, partial_threshold) or None.
    partial_threshold is a dict {threshold, scenarios} if a threshold was mid-progress.
    """
    ckpt_path = os.path.join(CHECKPOINT_DIR, f'{iso}_checkpoint.json')
    if not os.path.exists(ckpt_path):
        return None
    try:
        with open(ckpt_path) as f:
            data = json.load(f)
        iso_results = data['results']
        completed = set(iso_results.get('thresholds', {}).keys())
        phase = data.get('phase', 'unknown')
        partial = data.get('partial_threshold', None)
        partial_msg = ''
        if partial:
            partial_msg = f", partial {partial['threshold']}% ({len(partial['scenarios'])}/324 scenarios)"
        print(f"    [checkpoint] Resuming {iso} from {phase}: "
              f"{len(completed)} thresholds done{partial_msg}")
        return iso_results, completed, partial
    except Exception as e:
        print(f"    [checkpoint] Failed to load {iso}: {e}")
        return None


def clear_checkpoint(iso):
    """Remove checkpoint after ISO is fully complete."""
    ckpt_path = os.path.join(CHECKPOINT_DIR, f'{iso}_checkpoint.json')
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)


def process_iso(args):
    """
    Process a single ISO region. Called sequentially.

    For each threshold:
      1. Run full 3-phase co-optimization for all 324 cost scenarios
      2. Each scenario uses its own cost function → different costs produce different optimal mixes
      3. Matching cache shared across scenarios within a threshold (physics reuse only)
      4. Phase 2/3 refinement neighborhoods are cost-driven (different per scenario)
      5. Post-hoc monotonicity re-sweep: if cost(T_lower) > cost(T_higher), re-sweep
         T_lower with broader parameters (5% Phase 1 step, expanded procurement range,
         seed mixes from T_higher) to find the true optimum — up to 2 rounds

    Saves checkpoint after each threshold so progress is never lost.
    Resumes from checkpoint if one exists for this ISO.
    """
    # Unpack args — supports both old (4-tuple) and new (5-tuple) format
    if len(args) == 5:
        iso, demand_data, gen_profiles, emission_rates, fossil_mix = args
    else:
        iso, demand_data, gen_profiles, emission_rates = args
        fossil_mix = None

    print(f"\n{'='*70}")
    print(f"  {ISO_LABELS[iso]}")
    print(f"{'='*70}")

    demand_norm = demand_data[iso]['normalized'][:H]
    supply_profiles = get_supply_profiles(iso, gen_profiles)
    hydro_cap = HYDRO_CAPS[iso]
    anomaly_hours = find_anomaly_hours(iso, gen_profiles)
    np_profiles = prepare_numpy_profiles(demand_norm, supply_profiles)
    demand_total_mwh = demand_data[iso]['total_annual_mwh']

    # ── Check for existing checkpoint to resume from ──
    checkpoint = load_checkpoint(iso)
    completed_thresholds = set()
    partial_threshold_data = None
    if checkpoint:
        iso_results, completed_thresholds, partial_threshold_data = checkpoint
        # Re-derive sweep_results dict from stored sweep list (if any)
        if iso_results.get('sweep'):
            print(f"  Resuming from checkpoint: sweep done, "
                  f"{len(completed_thresholds)} thresholds complete")
        else:
            print(f"  Checkpoint found but no sweep — starting fresh")
            checkpoint = None

    if not checkpoint:
        iso_results = {
            'iso': iso,
            'label': ISO_LABELS[iso],
            'annual_demand_mwh': demand_total_mwh,
            'peak_demand_mw': demand_data[iso]['peak_mw'],
            'sweep': [],
            'thresholds': {},
        }

    # ---- SWEEP: max-matching runs for the sweep chart visualization ----
    if not iso_results.get('sweep'):
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
                break
    else:
        print(f"  Sweep: loaded from checkpoint ({len(iso_results['sweep'])} points)")
        sweep_results = None  # Already in iso_results['sweep']

    # Build sweep with costs, peak gap, and CO2 (skip if loaded from checkpoint)
    if sweep_results is not None:
        for p in sorted(sweep_results.keys()):
            r = sweep_results[p]
            peak_gap = compute_peak_gap(
                demand_norm, supply_profiles, r['resource_mix'],
                r['procurement_pct'], r['battery_dispatch_pct'], r['ldes_dispatch_pct'],
                anomaly_hours
            )
            r['peak_gap_pct'] = peak_gap
            costs = compute_costs(
                iso, r['resource_mix'], r['procurement_pct'],
                r['battery_dispatch_pct'], r['ldes_dispatch_pct'],
                r['hourly_match_score'],
                demand_norm, supply_profiles
            )
            r['costs'] = costs
            co2 = compute_co2_abatement(
                iso, r['resource_mix'], r['procurement_pct'], r['hourly_match_score'],
                r['battery_dispatch_pct'], r['ldes_dispatch_pct'],
                emission_rates, demand_total_mwh,
                demand_norm=demand_norm, supply_profiles=supply_profiles,
                fossil_mix=fossil_mix
            )
            r['co2_abated'] = co2
            iso_results['sweep'].append(r)
        save_checkpoint(iso, iso_results, phase='sweep')

    # ---- THRESHOLDS: co-optimize cost + matching for ALL 324 cost scenarios ----
    remaining = [t for t in THRESHOLDS if str(t) not in completed_thresholds]
    print(f"\n  Cost-optimizing thresholds (324 scenarios each)... "
          f"[{len(THRESHOLDS) - len(remaining)} done, {len(remaining)} remaining]")

    # Each threshold optimized independently; monotonicity enforced via
    # post-hoc re-sweep with broader parameters (not by result replacement)

    INTRA_THRESHOLD_CHECKPOINT_INTERVAL = 50  # Save every 50 scenarios (~5 min max loss)

    for threshold in THRESHOLDS:
        t_str = str(threshold)
        if t_str in completed_thresholds:
            print(f"    {threshold}%: loaded from checkpoint — skipping")
            continue

        t_start = time.time()
        # Fresh matching cache per threshold — no cross-threshold contamination
        score_cache = {}
        threshold_scenarios = {}
        medium_result = None

        # Resume from partial checkpoint if available for this threshold
        partial_done = set()
        if partial_threshold_data and str(partial_threshold_data.get('threshold')) == t_str:
            threshold_scenarios = partial_threshold_data.get('scenarios', {})
            partial_done = set(threshold_scenarios.keys())
            print(f"    {threshold}%: resuming from partial checkpoint "
                  f"({len(partial_done)}/324 scenarios done)")
            # Clear partial data so it's not reused for next threshold
            partial_threshold_data = None

        for s_idx, (scenario_key, cost_levels) in enumerate(ALL_COST_SCENARIOS):
            if scenario_key in partial_done:
                continue  # Already completed in previous session

            result = optimize_for_threshold(
                iso, demand_norm, supply_profiles, threshold, hydro_cap,
                emission_rates, demand_total_mwh,
                cost_levels=cost_levels, score_cache=score_cache
            )

            if result:
                # Compute cost for this scenario
                r_gen, f_gen, stor, fuel, tx = cost_levels
                cost_data = compute_costs_parameterized(
                    iso, result['resource_mix'], result['procurement_pct'],
                    result['battery_dispatch_pct'], result['ldes_dispatch_pct'],
                    result['hourly_match_score'],
                    r_gen, f_gen, stor, fuel, tx
                )
                result['costs'] = cost_data

                threshold_scenarios[scenario_key] = result

                # Track Medium scenario for logging
                if scenario_key == 'MMM_M_M':
                    medium_result = result

            # Intra-threshold checkpoint: save every N scenarios
            new_count = s_idx + 1 - len(partial_done)
            if new_count > 0 and new_count % INTRA_THRESHOLD_CHECKPOINT_INTERVAL == 0:
                save_checkpoint(iso, iso_results,
                    phase=f'threshold-{threshold}-partial-{len(threshold_scenarios)}of324',
                    partial_threshold={'threshold': threshold, 'scenarios': threshold_scenarios})

        # Log Medium scenario progress
        t_elapsed = time.time() - t_start
        cache_size = len(score_cache)
        if medium_result:
            mix = medium_result['resource_mix']
            print(f"    {threshold}%: {len(threshold_scenarios)}/324 scenarios, "
                  f"cache={cache_size}, {t_elapsed:.1f}s | "
                  f"Medium: CF{mix['clean_firm']}/Sol{mix['solar']}/Wnd{mix['wind']}"
                  f"/CCS{mix['ccs_ccgt']}/Hyd{mix['hydro']} "
                  f"batt={medium_result['battery_dispatch_pct']}% "
                  f"ldes={medium_result['ldes_dispatch_pct']}%")
        else:
            print(f"    {threshold}%: {len(threshold_scenarios)}/324 scenarios, "
                  f"cache={cache_size}, {t_elapsed:.1f}s")

        # ── Cross-pollination: evaluate all discovered mixes for all scenarios ──
        # A mix found optimal for one cost scenario may be cheaper for another
        unique_results = []
        seen_mix_keys = set()
        for sk, res in threshold_scenarios.items():
            mix = res['resource_mix']
            mk = (mix['clean_firm'], mix['solar'], mix['wind'],
                  mix['ccs_ccgt'], mix['hydro'],
                  res['procurement_pct'],
                  res['battery_dispatch_pct'],
                  res['ldes_dispatch_pct'])
            if mk not in seen_mix_keys:
                seen_mix_keys.add(mk)
                unique_results.append(res)

        cross_fixes = 0
        for scenario_key, cost_levels in ALL_COST_SCENARIOS:
            r_gen, f_gen, stor, fuel, tx = cost_levels
            current_cost = float('inf')
            if scenario_key in threshold_scenarios:
                current = threshold_scenarios[scenario_key]
                if 'costs' in current:
                    current_cost = current['costs']['effective_cost']

            for candidate in unique_results:
                if candidate['hourly_match_score'] < threshold:
                    continue
                cand_cost_data = compute_costs_parameterized(
                    iso, candidate['resource_mix'], candidate['procurement_pct'],
                    candidate['battery_dispatch_pct'], candidate['ldes_dispatch_pct'],
                    candidate['hourly_match_score'],
                    r_gen, f_gen, stor, fuel, tx
                )
                if cand_cost_data['effective_cost'] < current_cost:
                    threshold_scenarios[scenario_key] = dict(candidate)
                    threshold_scenarios[scenario_key]['costs'] = cand_cost_data
                    current_cost = cand_cost_data['effective_cost']
                    cross_fixes += 1

        if cross_fixes > 0:
            print(f"      Cross-pollination: {cross_fixes} improvements, "
                  f"{len(unique_results)} unique mixes evaluated")

        # Store all scenarios for this threshold
        # Medium result gets full detail (compressed day, peak gap, CO2)
        medium_key = 'MMM_M_M'
        if medium_key in threshold_scenarios:
            med = threshold_scenarios[medium_key]
            peak_gap = compute_peak_gap(
                demand_norm, supply_profiles, med['resource_mix'],
                med['procurement_pct'], med['battery_dispatch_pct'],
                med['ldes_dispatch_pct'], anomaly_hours
            )
            med['peak_gap_pct'] = peak_gap

            # Full costs at Medium
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

        iso_results['thresholds'][str(threshold)] = {
            'scenarios': threshold_scenarios,
            'scenario_count': len(threshold_scenarios),
        }
        # Checkpoint after each threshold — never lose more than one threshold's work
        save_checkpoint(iso, iso_results, phase=f'threshold-{threshold}')

    # ---- MONOTONICITY RE-SWEEP ----
    # For each cost scenario, cost must be non-decreasing across thresholds.
    # If cost(T_lower) > cost(T_higher), the search missed a better solution at T_lower.
    # Instead of replacing with the higher threshold's result, re-sweep T_lower
    # with broader parameters + seeds from the higher threshold's winning mix.
    MAX_RESWEEP_ROUNDS = 2
    sorted_thresholds = sorted(THRESHOLDS)

    # Build set of all scenario keys present across thresholds
    checked_scenarios = set()
    for t_idx in range(len(sorted_thresholds)):
        t_str = str(sorted_thresholds[t_idx])
        if t_str not in iso_results['thresholds']:
            continue
        for sk in iso_results['thresholds'][t_str]['scenarios']:
            checked_scenarios.add(sk)

    for resweep_round in range(MAX_RESWEEP_ROUNDS):
        # Detect monotonicity violations: {threshold: {scenario_key: better_threshold}}
        violations = {}
        total_violations = 0
        for scenario_key in checked_scenarios:
            prev_cost = None
            prev_t = None
            for threshold in sorted_thresholds:
                t_str = str(threshold)
                if t_str not in iso_results['thresholds']:
                    continue
                scenarios = iso_results['thresholds'][t_str]['scenarios']
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
            if resweep_round == 0:
                print(f"  Monotonicity check passed for all scenarios")
            else:
                print(f"  Monotonicity re-sweep round {resweep_round}: all violations resolved")
            break

        print(f"  Monotonicity round {resweep_round + 1}: {total_violations} violations "
              f"across {len(violations)} threshold(s) — triggering re-sweep")

        # Re-sweep each violated threshold with broader parameters
        for viol_threshold in sorted(violations.keys()):
            violated_scenarios = violations[viol_threshold]
            t_str = str(viol_threshold)

            # Collect seed mixes from the thresholds that achieved better cost
            seed_mixes_for_resweep = []
            seen_seeds = set()
            for sk, better_t in violated_scenarios.items():
                better_result = iso_results['thresholds'][str(better_t)]['scenarios'].get(sk)
                if better_result and 'resource_mix' in better_result:
                    key = tuple(better_result['resource_mix'][rt] for rt in RESOURCE_TYPES)
                    if key not in seen_seeds:
                        seen_seeds.add(key)
                        seed_mixes_for_resweep.append(dict(better_result['resource_mix']))

            # Expand procurement bounds for broader search
            default_min, default_max = PROCUREMENT_BOUNDS.get(viol_threshold, (70, 310))
            expanded_min = max(50, default_min - 20)
            expanded_max = min(500, default_max + 30)

            print(f"    Re-sweeping {viol_threshold}%: {len(violated_scenarios)} scenarios, "
                  f"{len(seed_mixes_for_resweep)} seed mixes, "
                  f"procurement [{expanded_min}-{expanded_max}%]")

            # Fresh score cache for re-sweep (shared across re-swept scenarios)
            resweep_cache = {}
            resweep_fixes = 0
            threshold_scenarios = iso_results['thresholds'][t_str]['scenarios']

            for scenario_key in violated_scenarios:
                cost_levels = COST_SCENARIO_MAP[scenario_key]
                result = optimize_for_threshold(
                    iso, demand_norm, supply_profiles, viol_threshold, hydro_cap,
                    emission_rates, demand_total_mwh,
                    cost_levels=cost_levels, score_cache=resweep_cache,
                    resweep=True, seed_mixes=seed_mixes_for_resweep,
                    procurement_bounds_override=(expanded_min, expanded_max)
                )

                if result:
                    r_gen, f_gen, stor, fuel, tx = cost_levels
                    cost_data = compute_costs_parameterized(
                        iso, result['resource_mix'], result['procurement_pct'],
                        result['battery_dispatch_pct'], result['ldes_dispatch_pct'],
                        result['hourly_match_score'],
                        r_gen, f_gen, stor, fuel, tx
                    )
                    result['costs'] = cost_data

                    # Replace only if the re-sweep found a cheaper solution
                    current = threshold_scenarios.get(scenario_key)
                    if current is None or cost_data['effective_cost'] < current['costs']['effective_cost']:
                        threshold_scenarios[scenario_key] = result
                        resweep_fixes += 1

            # Cross-pollination within re-swept threshold
            unique_results = []
            seen_mix_keys = set()
            for sk, res in threshold_scenarios.items():
                mix = res['resource_mix']
                mk = (mix['clean_firm'], mix['solar'], mix['wind'],
                      mix['ccs_ccgt'], mix['hydro'],
                      res['procurement_pct'],
                      res['battery_dispatch_pct'],
                      res['ldes_dispatch_pct'])
                if mk not in seen_mix_keys:
                    seen_mix_keys.add(mk)
                    unique_results.append(res)

            cross_fixes = 0
            for scenario_key in violated_scenarios:
                cost_levels = COST_SCENARIO_MAP[scenario_key]
                r_gen, f_gen, stor, fuel, tx = cost_levels
                current = threshold_scenarios.get(scenario_key)
                current_cost = current['costs']['effective_cost'] if current and 'costs' in current else float('inf')

                for candidate in unique_results:
                    if candidate['hourly_match_score'] < viol_threshold:
                        continue
                    cand_cost_data = compute_costs_parameterized(
                        iso, candidate['resource_mix'], candidate['procurement_pct'],
                        candidate['battery_dispatch_pct'], candidate['ldes_dispatch_pct'],
                        candidate['hourly_match_score'],
                        r_gen, f_gen, stor, fuel, tx
                    )
                    if cand_cost_data['effective_cost'] < current_cost:
                        threshold_scenarios[scenario_key] = dict(candidate)
                        threshold_scenarios[scenario_key]['costs'] = cand_cost_data
                        current_cost = cand_cost_data['effective_cost']
                        cross_fixes += 1

            # Update Medium scenario detail if it was re-swept
            medium_key = 'MMM_M_M'
            if medium_key in violated_scenarios and medium_key in threshold_scenarios:
                med = threshold_scenarios[medium_key]
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

            print(f"      Fixed {resweep_fixes} via re-sweep, "
                  f"{cross_fixes} via cross-pollination")

    else:
        # Exhausted MAX_RESWEEP_ROUNDS — report remaining violations
        remaining = 0
        for scenario_key in checked_scenarios:
            prev_cost = None
            prev_t = None
            for threshold in sorted_thresholds:
                t_str = str(threshold)
                if t_str not in iso_results['thresholds']:
                    continue
                scenarios = iso_results['thresholds'][t_str]['scenarios']
                if scenario_key not in scenarios:
                    continue
                result = scenarios[scenario_key]
                if 'costs' not in result:
                    continue
                cost = result['costs']['effective_cost']
                if prev_cost is not None and cost < prev_cost - 0.01:
                    remaining += 1
                prev_cost = cost
                prev_t = threshold
        if remaining > 0:
            print(f"  WARNING: {remaining} monotonicity violations remain after "
                  f"{MAX_RESWEEP_ROUNDS} re-sweep rounds (search space exhausted)")
        else:
            print(f"  All monotonicity violations resolved after {MAX_RESWEEP_ROUNDS} rounds")

    # ISO complete — clear checkpoint (full results saved in main())
    clear_checkpoint(iso)
    return iso, iso_results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    start_time = time.time()
    demand_data, gen_profiles, emission_rates, fossil_mix = load_data()

    # Build config section with ALL cost tables for dashboard recalculation
    config = {
        'data_year': DATA_YEAR,
        'battery_duration': '4h',
        'battery_efficiency': BATTERY_EFFICIENCY,
        'ldes_duration': '100h',
        'ldes_efficiency': LDES_EFFICIENCY,
        'ldes_window_days': LDES_WINDOW_DAYS,
        'hydro_caps': HYDRO_CAPS,
        'resource_types': RESOURCE_TYPES,
        'thresholds': THRESHOLDS,
        'procurement_bounds': {str(k): list(v) for k, v in PROCUREMENT_BOUNDS.items()},
        'total_scenarios_per_threshold': len(ALL_COST_SCENARIOS),
        'wholesale_prices': WHOLESALE_PRICES,
        'regional_lcoe': REGIONAL_LCOE,
        'grid_mix_shares': GRID_MIX_SHARES,
        'ccs_residual_emission_rate': CCS_RESIDUAL_EMISSION_RATE,
        # Paired toggle group definitions for dashboard controls
        'paired_toggle_groups': PAIRED_TOGGLE_GROUPS,
        # Full L/M/H tables for dashboard cost recalculation
        'lcoe_tables': FULL_LCOE_TABLES,
        'transmission_tables': FULL_TRANSMISSION_TABLES,
        'fuel_prices': FUEL_PRICES,
        # Emission rates from eGRID (for CO2 calculations)
        'emission_rates': {
            iso: {
                'fossil_co2_lb_per_mwh': emission_rates[iso]['fossil_co2_lb_per_mwh'],
                'gas_co2_lb_per_mwh': emission_rates[iso]['gas_co2_lb_per_mwh'],
                'coal_co2_lb_per_mwh': emission_rates[iso]['coal_co2_lb_per_mwh'],
                'oil_co2_lb_per_mwh': emission_rates[iso]['oil_co2_lb_per_mwh'],
                'total_co2_lb_per_mwh': emission_rates[iso]['total_co2_lb_per_mwh'],
            }
            for iso in ISOS
        },
        # Grid mix shares for baseline display
        'grid_mix_shares_display': GRID_MIX_SHARES,
        # Regional fossil fuel mix shares (for fuel price sensitivity)
        'fossil_mix': {
            iso: {
                'coal_share': fossil_mix[iso][DATA_YEAR]['coal_share'][:24] if DATA_YEAR in fossil_mix.get(iso, {}) else [],
                'gas_share': fossil_mix[iso][DATA_YEAR]['gas_share'][:24] if DATA_YEAR in fossil_mix.get(iso, {}) else [],
                'oil_share': fossil_mix[iso][DATA_YEAR]['oil_share'][:24] if DATA_YEAR in fossil_mix.get(iso, {}) else [],
            }
            for iso in ISOS
        },
    }

    all_results = {
        'config': config,
        'results': {},
    }

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard', 'overprocure_results.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Run ISOs sequentially to avoid memory pressure and enable incremental saves.
    # Each ISO does 10 thresholds × 324 scenarios — heavy compute per ISO.
    for iso in ISOS:
        iso_start = time.time()
        args = (iso, demand_data, gen_profiles, emission_rates, fossil_mix)
        iso_name, iso_results = process_iso(args)
        all_results['results'][iso_name] = iso_results
        iso_elapsed = time.time() - iso_start
        print(f"\n  {iso_name} completed in {iso_elapsed:.0f}s")

        # Incremental save after each ISO — never lose progress
        with open(output_path, 'w') as f:
            json.dump(all_results, f)
        print(f"  Saved incrementally: {os.path.getsize(output_path) / 1024:.0f} KB "
              f"({len(all_results['results'])}/{len(ISOS)} ISOs)")

    # Save cached results file (reusable input for future projects)
    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'optimizer_cache.json')
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    try:
        import subprocess
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                           cwd=os.path.dirname(os.path.abspath(__file__)),
                                           stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_hash = 'unknown'

    from datetime import datetime, timezone
    cache_data = {
        'metadata': {
            'created_at': datetime.now(timezone.utc).isoformat(),
            'optimizer_version': '3.1-paired-toggles-resweep',
            'git_commit': git_hash,
            'runtime_seconds': round(time.time() - start_time, 1),
            'description': 'Full co-optimized results: 10 thresholds x 324 paired-toggle scenarios x 5 ISOs',
            'thresholds': THRESHOLDS,
            'isos': ISOS,
            'scenarios_per_threshold': len(ALL_COST_SCENARIOS),
            'total_optimizations': len(THRESHOLDS) * len(ALL_COST_SCENARIOS) * len(ISOS),
        },
        'config': config,
        'results': all_results['results'],
    }
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f, indent=2)
    print(f"  Cached results: {cache_path} ({os.path.getsize(cache_path) / 1024:.0f} KB)")

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  Complete in {elapsed:.0f}s. Saved to: {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1024:.0f} KB")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
