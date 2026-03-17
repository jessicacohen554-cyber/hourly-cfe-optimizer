#!/usr/bin/env python3
"""
Market Simulator — What happens to generators under different market conditions?
================================================================================
Profit-driven market simulation: deploys clean resources where profitable,
stops when profit ≤ 0. Clean energy level is an OUTPUT (emerges from
profitability), not a target.

Leverages the pre-computed physics cache (Step 1 PFS + Step 2 EF/cost parquets)
and re-evaluates market economics on top of it. Generator-level economics
(per heat-rate bin) are tracked through dispatch.

Adapted from step6_1_smartargets.py R1/R2 reference sweep, with all emission-
constraint / mandated-deployment / DAC logic removed.

Usage:
  python market_simulation.py                           # Full 270-scenario sweep
  python market_simulation.py --isos CAISO ERCOT        # Subset ISOs
  python market_simulation.py --single                  # Single scenario (Medium defaults)
  python market_simulation.py --snapshot                # Single-year snapshot mode
  python market_simulation.py --carbon-price 50         # Override carbon price
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product as cartesian

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_ROOT = os.path.dirname(SCRIPT_DIR)
# Add scripts dir to path for shared utilities
sys.path.insert(0, SCRIPT_DIR)

from pipeline_config import (
    ISOS, REGIONAL_DEMAND_TWH, DEMAND_GROWTH_RATES,
    GRID_MIX_SHARES, WHOLESALE_PRICES, THRESHOLDS,
    CAPACITY_MARKET_PRICES, CAPACITY_DEGRADATION_ALPHA, CAPACITY_DEGRADATION_PARAMS,
    PEAK_CAPACITY_CREDITS, RESOURCE_CAPACITY_FACTORS,
    LCOE_TABLES, TX_TABLES, get_tx,
    NUCLEAR_NEWBUILD_LCOE, CCS_LCOE_45Q_ON, CCS_LCOE_45Q_OFF, GEOTHERMAL_LCOE,
    FOAK_NUCLEAR_NEWBUILD, FOAK_CCS_45Q_ON, FOAK_CCS_45Q_OFF, FOAK_GEOTHERMAL,
    FOAK_LDES, FOAK_H2, FOAK_OFFSHORE_WIND,
    EXISTING_GAS_FOM_KW_YR,
    NEW_GAS_CCGT_LCOE, NEW_GAS_CT_LCOE,
    compute_storage_revenue_credit,
    OFFSHORE_ISOS, CCS_CAP_TWH, GEOTHERMAL_CAP_TWH,
    H, NUCLEAR_OFFTAKE_CONTRACTS,
    get_rps_floor,
)
from dispatch_utils import (
    load_common_data, get_demand_profile, get_supply_profiles,
    reconstruct_hourly_dispatch, compute_fossil_retirement,
    COAL_CAP_TWH, OIL_CAP_TWH,
)
from lmp_engine import (
    build_merit_order_stack, compute_hourly_lmp_vectorized, PriceModel,
    HEAT_RATES, VOM, CO2_RATES, FUEL_PRICES,
    INSTALLED_FOSSIL_MW, FOSSIL_CAPACITY_SHARES,
)
from procurement_utils import get_rps_target_at_year, PPA_PREMIUMS

OUTPUT_DIR = os.path.join(MODULE_ROOT, 'data', 'results')

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Interconnection queue caps (GW/yr new build per ISO)
# Sources: LBNL "Queued Up 2024" (Rand et al., 2024) — queue completion rate analysis
#   https://emp.lbl.gov/queues
# Methodology: Completion rates by ISO derived from 2014-2023 historical data.
# High: 50th-percentile completion speed (FERC Order 2023 reforms). ~40-50% of queue COD in 7yr.
# Low: 20th-percentile (status quo permitting). ~15-25% of queue COD.
# Medium: Geometric mean of Low/High.
QUEUE_CAP_GW = {
    'High': {
        'CAISO': 6, 'ERCOT': 12, 'PJM': 7, 'NYISO': 5,
        'NEISO': 5, 'MISO': 7, 'SPP': 6,
    },
    'Low': {
        'CAISO': 3, 'ERCOT': 6, 'PJM': 3, 'NYISO': 2,
        'NEISO': 2, 'MISO': 3, 'SPP': 3,
    },
    'Medium': {
        'CAISO': 4.5, 'ERCOT': 9, 'PJM': 5, 'NYISO': 3.5,
        'NEISO': 3.5, 'MISO': 5, 'SPP': 4.5,
    },
}

# Wright's Law — deployment-based learning
WRIGHT_CUMULATIVE_GW_2025 = {
    'nuclear': 2.0, 'ccs': 0.3, 'ldes': 0.01, 'h2': 0.1,
    'geothermal': 0.05, 'battery': 50.0, 'battery8': 50.0,
    'solar': 150.0, 'wind': 150.0, 'offshore_wind': 5.0,
}

WRIGHT_LEARNING_RATE = {
    'nuclear':       {'Fast': 0.15, 'Slow': 0.10},
    'ccs':           {'Fast': 0.12, 'Slow': 0.10},
    'ldes':          {'Fast': 0.20, 'Slow': 0.15},
    'h2':            {'Fast': 0.18, 'Slow': 0.12},
    'geothermal':    {'Fast': 0.20, 'Slow': 0.15},
    'battery':       {'Fast': 0.20, 'Slow': 0.18},
    'battery8':      {'Fast': 0.20, 'Slow': 0.18},
    'solar':         {'Fast': 0.0, 'Slow': 0.0},
    'wind':          {'Fast': 0.0, 'Slow': 0.0},
    'offshore_wind': {'Fast': 0.12, 'Slow': 0.08},
}

WRIGHT_BACKGROUND_GW = {
    'nuclear':       {'Fast': (30, 150), 'Slow': (5, 20)},
    'ccs':           {'Fast': (10, 50),  'Slow': (2, 10)},
    'ldes':          {'Fast': (5, 30),   'Slow': (0.5, 5)},
    'h2':            {'Fast': (3, 20),   'Slow': (0.5, 3)},
    'geothermal':    {'Fast': (3, 15),   'Slow': (0.5, 3)},
    'battery':       {'Fast': (200, 800),'Slow': (80, 300)},
    'battery8':      {'Fast': (50, 200), 'Slow': (20, 80)},
    'solar':         {'Fast': (500, 2000), 'Slow': (200, 800)},
    'wind':          {'Fast': (300, 1200), 'Slow': (100, 500)},
    'offshore_wind': {'Fast': (40, 150),  'Slow': (10, 50)},
}

PTC_45Y_NEW_NUCLEAR = 26.0  # $/MWh
PTC_45U_VALUE = 15.0        # $/MWh existing nuclear PTC
PTC_45U_SUNSET_YEAR = 2032

# Nuclear retirement threshold — default operating cost
NUCLEAR_FOM_PER_MWH = 30.0  # $/MWh equivalent at 93% CF

# Simulation years — default sparse set (legacy); annual mode uses build_sim_years()
SIM_YEARS = [2023, 2030, 2035, 2040, 2045, 2050]


def build_sim_years(start=2025, end=2060, step=1):
    """Build simulation year list from user-specified range and step.

    Returns a list like [2025, 2026, ..., 2060] for step=1
    or [2025, 2030, 2035, ..., 2060] for step=5.
    Always includes the end year if not already present.
    """
    years = list(range(start, end + 1, step))
    if end not in years:
        years.append(end)
    return years

# 2023 eGRID actual clean energy share (%)
EGRID_2023_CLEAN_PCT = {
    'CAISO': 48.5, 'ERCOT': 40.2, 'PJM': 36.8, 'NYISO': 33.0,
    'NEISO': 29.5, 'MISO': 25.8, 'SPP': 42.0,
}

EGRID_2023_LMP = {
    'CAISO': 36.0, 'ERCOT': 28.5, 'PJM': 33.0, 'NYISO': 38.0,
    'NEISO': 42.0, 'MISO': 27.0, 'SPP': 24.0,
}

RESOURCE_TO_TECH = {
    'clean_firm': 'nuclear', 'solar': 'solar', 'wind': 'wind',
    'offshore_wind': 'offshore_wind', 'ccs_ccgt': 'ccs', 'hydro': 'hydro',
    'battery': 'battery', 'battery8': 'battery8', 'ldes': 'ldes', 'h2': 'h2',
    'geothermal': 'geothermal',
}

# Gas friction levels
GAS_FRICTION_LEVELS = {'Low': 0.3, 'Medium': 0.7, 'High': 1.0}

# PPA levels
PPA_LEVELS = ['Low', 'Medium', 'High']

# Queue cap level → learning speed mapping (for sweep mode)
QUEUE_LEARNING_MAP = {
    'High': 'Fast',
    'Medium': 'Medium',
    'Low': 'Slow',
}

DEMAND_GROWTH_LEVELS = ['Low', 'Medium', 'High']

# Price sensitivities (9-dim cost toggles)
PRICE_SENSITIVITIES = {
    'all_low': {
        'ren': 'Low', 'firm': 'L', 'batt': 'Low', 'ldes_lvl': 'Low',
        'ccs': 'L', 'q45': '1', 'fuel': 'Low', 'tx': 'Low', 'geo': 'L',
    },
    'all_med': {
        'ren': 'Medium', 'firm': 'M', 'batt': 'Medium', 'ldes_lvl': 'Medium',
        'ccs': 'M', 'q45': '1', 'fuel': 'Medium', 'tx': 'Medium', 'geo': 'M',
    },
    'all_high': {
        'ren': 'High', 'firm': 'H', 'batt': 'High', 'ldes_lvl': 'High',
        'ccs': 'H', 'q45': '1', 'fuel': 'High', 'tx': 'High', 'geo': 'H',
    },
    'high_vre_low_firm': {
        'ren': 'High', 'firm': 'L', 'batt': 'High', 'ldes_lvl': 'High',
        'ccs': 'L', 'q45': '1', 'fuel': 'Medium', 'tx': 'Medium', 'geo': 'L',
    },
    'high_firm_low_vre': {
        'ren': 'Low', 'firm': 'H', 'batt': 'Low', 'ldes_lvl': 'Low',
        'ccs': 'H', 'q45': '1', 'fuel': 'Medium', 'tx': 'Medium', 'geo': 'H',
    },
}

# RPS/REC Compliance model constants
ACP_RATES = {
    'CAISO': 50.0, 'ERCOT': 0.0, 'PJM': 45.0, 'NYISO': 42.5,
    'NEISO': 45.0, 'MISO': 15.0, 'SPP': 10.0,
}
REC_COMPLIANCE_PRICE_2025 = {
    'CAISO': 34.0, 'ERCOT': 0.5, 'PJM': 38.0, 'NYISO': 25.0,
    'NEISO': 40.0, 'MISO': 8.0, 'SPP': 3.0,
}
VOLUNTARY_REC_FLOOR = {
    'CAISO': 3.0, 'ERCOT': 0.5, 'PJM': 2.0, 'NYISO': 2.0,
    'NEISO': 2.5, 'MISO': 1.5, 'SPP': 2.0,
}
VOLUNTARY_DEMAND_ADDER = {
    'CAISO': 0.00, 'ERCOT': 0.02, 'PJM': 0.00, 'NYISO': 0.04,
    'NEISO': 0.035, 'MISO': 0.06, 'SPP': 0.01,
}
REC_SCARCITY_K = {
    'CAISO': 0.10, 'ERCOT': 0.10, 'PJM': 0.29, 'NYISO': 0.15,
    'NEISO': 0.29, 'MISO': 0.12, 'SPP': 0.10,
}
REC_SURPLUS_DECAY_K = 0.20
CES_ISOS = {'NYISO', 'NEISO', 'CAISO'}
REC_ELIGIBLE = {'solar', 'wind', 'offshore_wind', 'hydro', 'geothermal'}
CES_ELIGIBLE = REC_ELIGIBLE | {'clean_firm', 'ccs_ccgt'}
CES_DISCOUNT_FACTOR = 0.60

# ACP recycling parameters (matching step6_1_smartargets.py)
ACP_FUND_EFFICIENCY = 0.65   # Fraction of ACP payments that fund renewable dev
AVG_COST_PER_GW = 1200       # $M/GW for utility-scale solar/wind

# PPA market depth by ISO (for PPA discount scaling)
PPA_MARKET_DEPTH = {
    'CAISO': 0.90, 'ERCOT': 1.00, 'PJM': 0.85, 'NYISO': 0.70,
    'NEISO': 0.65, 'MISO': 0.60, 'SPP': 0.50,
}

# Unit commitment parameters per unit type
# Sources: NREL 2024 ATB, EIA Form 860 operational data, FERC Form 714
# min_up_hrs / min_down_hrs: thermal cycling constraints
# start_cost_per_mw: $/MW cold-start cost (hot start ~40% of this)
# min_gen_pct: minimum stable generation as fraction of nameplate
#   For gas_ccgt, min_gen varies by vintage — newer single-shaft F/H-class
#   turbines turn down to ~35%, older 2x1 multi-shaft units need ~50%.
UNIT_COMMITMENT = {
    'coal_steam':  {'min_up_hrs': 24, 'min_down_hrs': 12, 'start_cost_per_mw': 150.0, 'min_gen_pct': 0.40},
    'gas_ccgt':    {'min_up_hrs': 4,  'min_down_hrs': 2,  'start_cost_per_mw': 35.0,  'min_gen_pct': 0.50},
    'gas_ct':      {'min_up_hrs': 1,  'min_down_hrs': 1,  'start_cost_per_mw': 15.0,  'min_gen_pct': 0.20},
    'oil_ct':      {'min_up_hrs': 1,  'min_down_hrs': 1,  'start_cost_per_mw': 20.0,  'min_gen_pct': 0.20},
}


def get_unit_commitment_params(unit_type, online_year=None):
    """Get UC parameters adjusted for plant vintage.

    Newer CCGTs (online 2010+) use F/H-class single-shaft turbines with
    better turndown (~35% min gen). Older 2x1 multi-shaft units need ~50%.
    Start costs also lower for newer units due to faster ramp rates.

    Args:
        unit_type: 'coal_steam', 'gas_ccgt', 'gas_ct', 'oil_ct'
        online_year: year plant came online (from EIA 860). None = use defaults.

    Returns:
        dict with min_up_hrs, min_down_hrs, start_cost_per_mw, min_gen_pct
    """
    base = UNIT_COMMITMENT.get(unit_type, {})
    if not base:
        return base

    params = dict(base)  # copy

    if unit_type == 'gas_ccgt' and online_year is not None:
        try:
            yr = int(online_year)
        except (ValueError, TypeError):
            return params

        if yr >= 2015:
            # Latest H-class (GE 7HA, Siemens 9000HL): 30-35% min stable
            params['min_gen_pct'] = 0.30
            params['start_cost_per_mw'] = 25.0
            params['min_up_hrs'] = 3
            params['min_down_hrs'] = 1
        elif yr >= 2005:
            # F-class era (GE 7FA, Siemens V94.3A): 35-40% min stable
            params['min_gen_pct'] = 0.38
            params['start_cost_per_mw'] = 30.0
            params['min_up_hrs'] = 4
            params['min_down_hrs'] = 2
        # else: pre-2005, keep defaults (50% min gen, older multi-shaft)

    return params


# ═══════════════════════════════════════════════════════════════════════════════
# EGRID BASELINES
# ═══════════════════════════════════════════════════════════════════════════════

def load_egrid_baselines():
    """Load 2023 eGRID absolute emission baselines."""
    egrid_path = os.path.join(MODULE_ROOT, 'data', 'egrid_2023_baseline_emissions.json')
    if os.path.exists(egrid_path):
        with open(egrid_path) as f:
            data = json.load(f)
        return {iso: d['co2_metric_tons'] for iso, d in data.items() if iso in ISOS}
    # Fallback hardcoded from eGRID 2023 BA23 sheet
    return {
        'CAISO':  31_376_504, 'ERCOT': 157_460_286, 'PJM':   267_318_273,
        'NYISO':  28_193_964, 'NEISO':  25_081_328, 'MISO':  290_402_405,
        'SPP':   110_506_609,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# WRIGHT'S LAW LEARNING CURVES
# ═══════════════════════════════════════════════════════════════════════════════

def wright_cost(foak_cost, noak_floor, cumulative_gw, reference_gw, learning_rate):
    """Deployment-based Wright's Law cost at current cumulative GW."""
    if cumulative_gw <= reference_gw or learning_rate <= 0:
        return foak_cost
    exponent = -np.log2(1.0 - learning_rate)
    cost = foak_cost * (cumulative_gw / reference_gw) ** (-exponent)
    return max(noak_floor, cost)


def get_background_gw(tech, learning_speed, year):
    """Interpolate exogenous rest-of-world cumulative GW at a given year."""
    bg = WRIGHT_BACKGROUND_GW.get(tech, {}).get(learning_speed, (0, 0))
    gw_2035, gw_2050 = bg
    if year <= 2025:
        return 0.0
    if year >= 2050:
        return gw_2050
    if year <= 2035:
        return gw_2035 * (year - 2025) / 10.0
    return gw_2035 + (gw_2050 - gw_2035) * (year - 2035) / 15.0


def get_effective_cumulative_gw(tech, model_deployed_gw, learning_speed, year):
    """Total cumulative = 2025 baseline + model deployed + background."""
    base = WRIGHT_CUMULATIVE_GW_2025.get(tech, 1.0)
    bg = get_background_gw(tech, learning_speed, year)
    return base + model_deployed_gw + bg


# ═══════════════════════════════════════════════════════════════════════════════
# REVENUE MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def compute_lmp_at_threshold(iso, clean_pct, fuel_level, demand_norm,
                              demand_mw_profile, supply_profiles, resource_pcts,
                              battery_pct=0, battery8_pct=0, ldes_pct=0, h2_pct=0,
                              carbon_price=0, nox_price=0.0, sox_price=0.0,
                              nox_limit=None, sox_limit=None,
                              custom_fuel_prices=None, custom_co2_price=None,
                              custom_heat_rates=None, custom_vom=None):
    """Compute 8760-hour LMP at a given clean percentage.

    Returns (hourly_lmp_array, avg_lmp, lmp_p90, generator_economics).
    generator_economics is a dict of per-unit-type dispatch metrics.
    """
    stack, total_fossil_mw = build_merit_order_stack(
        iso, clean_pct, fuel_level=fuel_level,
        resource_mix=resource_pcts,
        battery_pct=battery_pct, battery8_pct=battery8_pct,
        ldes_pct=ldes_pct, h2_pct=h2_pct,
        nox_price=nox_price, sox_price=sox_price,
        nox_limit=nox_limit, sox_limit=sox_limit,
        custom_fuel_prices=custom_fuel_prices,
        custom_co2_price=custom_co2_price,
        custom_heat_rates=custom_heat_rates,
        custom_vom=custom_vom,
    )

    dispatch = reconstruct_hourly_dispatch(
        demand_norm, supply_profiles, resource_pcts,
        procurement_pct=clean_pct,
        battery_dispatch_pct=battery_pct,
        battery8_dispatch_pct=battery8_pct,
        ldes_dispatch_pct=ldes_pct,
        h2_dispatch_pct=h2_pct,
    )

    price_model = PriceModel(iso, fuel_level)
    hourly_lmp, unit_idx = compute_hourly_lmp_vectorized(
        dispatch, demand_mw_profile, stack, price_model, iso=iso,
        vre_penetration=clean_pct / 100.0 if clean_pct is not None else None,
    )

    avg_lmp = float(np.mean(hourly_lmp))
    p90_lmp = float(np.percentile(hourly_lmp, 90))

    # --- GENERATOR-LEVEL ECONOMICS ---
    gen_econ = compute_generator_economics(
        stack, hourly_lmp, unit_idx, dispatch, demand_mw_profile,
        total_fossil_mw, iso, clean_pct, fuel_level, carbon_price,
    )

    return hourly_lmp, avg_lmp, p90_lmp, gen_econ


def apply_unit_commitment(dispatched, mw_dispatched, cap_mw, uc_params):
    """Apply unit commitment constraints to a single unit's dispatch profile.

    Post-processes raw merit-order dispatch with minimum up/down times,
    minimum stable generation, and start-up cost tracking.

    Args:
        dispatched: bool array (H,) — hours where merit-order says unit should run
        mw_dispatched: float array (H,) — raw MW output per hour
        cap_mw: nameplate capacity MW
        uc_params: dict with min_up_hrs, min_down_hrs, min_gen_pct, start_cost_per_mw

    Returns:
        committed: bool array — UC-adjusted commitment schedule
        mw_output: float array — UC-adjusted MW output
        n_starts: int — number of start events
    """
    H_len = len(dispatched)
    min_up = uc_params.get('min_up_hrs', 1)
    min_down = uc_params.get('min_down_hrs', 1)
    min_gen = uc_params.get('min_gen_pct', 0.0) * cap_mw

    committed = np.zeros(H_len, dtype=bool)
    n_starts = 0
    hours_in_state = 0  # hours since last state change
    is_on = False

    for h in range(H_len):
        want_on = bool(dispatched[h])

        if is_on:
            hours_in_state += 1
            if not want_on and hours_in_state >= min_up:
                # Can shut down — check min_down feasibility
                is_on = False
                hours_in_state = 0
            # else: stay on (either want_on=True, or min_up not met yet)
            committed[h] = is_on if not want_on else True
            if not is_on:
                committed[h] = False
            else:
                committed[h] = True
        else:
            hours_in_state += 1
            if want_on and hours_in_state >= min_down:
                # Can start up
                is_on = True
                hours_in_state = 0
                n_starts += 1
                committed[h] = True
            else:
                committed[h] = False
                if want_on and hours_in_state < min_down:
                    pass  # Can't start yet, min_down not satisfied

    # Apply minimum generation floor and cap
    mw_output = np.where(committed, np.maximum(mw_dispatched, min_gen), 0.0)
    mw_output = np.minimum(mw_output, cap_mw)

    return committed, mw_output, n_starts


def compute_generator_economics(stack, hourly_lmp, unit_idx, dispatch,
                                 demand_mw_profile, total_fossil_mw,
                                 iso, clean_pct, fuel_level, carbon_price=0):
    """Compute per-generator-class dispatch economics.

    Tracks: dispatch hours, capacity factor, average revenue, variable cost,
    and net margin for each unit type in the merit-order stack.

    Returns dict: {unit_type: {dispatch_hours, cf, avg_rev_mwh, var_cost_mwh,
                                margin_mwh, capacity_mw}}
    """
    gen_econ = {}

    if len(stack) == 0:
        return gen_econ

    # Build cumulative capacity array from stack
    cum_cap = np.zeros(len(stack) + 1)
    unit_types = []
    unit_costs = []
    unit_caps = []
    for i, (utype, cap_mw, mc) in enumerate(stack):
        cum_cap[i + 1] = cum_cap[i] + cap_mw
        unit_types.append(utype)
        unit_costs.append(mc)
        unit_caps.append(cap_mw)

    # Residual demand that fossil must serve (MW per hour)
    residual = dispatch.get('residual_demand', np.zeros(H))
    if isinstance(residual, (list, tuple)):
        residual = np.array(residual, dtype=np.float64)
    # residual_demand is in same normalized units as demand_norm (sum ≈ 1.0).
    # Convert to MW: residual_mw = residual_norm × total_annual_mwh
    # (Same conversion used in lmp_engine.compute_hourly_lmp_vectorized)
    total_annual_mwh = float(np.sum(demand_mw_profile))
    fossil_demand_mw = residual * total_annual_mwh

    # Compute carbon adder per unit
    carbon_adder = {}
    for utype in set(unit_types):
        co2_rate = CO2_RATES.get(utype, 0.5)
        carbon_adder[utype] = co2_rate * carbon_price

    # Track per-unit economics
    for i, (utype, cap_mw, mc) in enumerate(stack):
        low = cum_cap[i]
        high = cum_cap[i + 1]

        # Hours where this unit is dispatched (demand exceeds lower cumulative)
        dispatched = fossil_demand_mw > low
        # MW dispatched per hour (capped at unit capacity)
        mw_dispatched = np.clip(fossil_demand_mw - low, 0, cap_mw) * dispatched

        # Apply unit commitment constraints (min up/down, min gen)
        uc = UNIT_COMMITMENT.get(utype, {})
        start_cost_total = 0.0
        if uc:
            dispatched, mw_dispatched, n_starts = apply_unit_commitment(
                dispatched, mw_dispatched, cap_mw, uc)
            start_cost_total = n_starts * uc.get('start_cost_per_mw', 0) * cap_mw

        dispatch_hours = int(np.sum(dispatched))
        total_mwh = float(np.sum(mw_dispatched))
        cf = total_mwh / (cap_mw * H) if cap_mw > 0 else 0

        # Revenue: LMP in dispatched hours, weighted by MW
        if total_mwh > 0:
            avg_rev = float(np.sum(hourly_lmp * mw_dispatched)) / total_mwh
        else:
            avg_rev = 0.0

        # Variable cost includes carbon + amortized start-up costs
        var_cost = mc + carbon_adder.get(utype, 0)
        start_cost_mwh = start_cost_total / total_mwh if total_mwh > 0 else 0.0

        margin = avg_rev - var_cost - start_cost_mwh

        key = utype
        if key in gen_econ:
            # Aggregate multiple units of same type
            existing = gen_econ[key]
            total_existing_mwh = existing['_total_mwh']
            combined_mwh = total_existing_mwh + total_mwh
            if combined_mwh > 0:
                existing['avg_rev_mwh'] = (
                    existing['avg_rev_mwh'] * total_existing_mwh +
                    avg_rev * total_mwh
                ) / combined_mwh
                existing['var_cost_mwh'] = (
                    existing['var_cost_mwh'] * total_existing_mwh +
                    var_cost * total_mwh
                ) / combined_mwh
            existing['dispatch_hours'] = max(existing['dispatch_hours'], dispatch_hours)
            existing['cf'] = (total_existing_mwh + total_mwh) / ((existing['capacity_mw'] + cap_mw) * H)
            existing['capacity_mw'] += cap_mw
            existing['_total_mwh'] = combined_mwh
            existing['margin_mwh'] = existing['avg_rev_mwh'] - existing['var_cost_mwh']
        else:
            gen_econ[key] = {
                'dispatch_hours': dispatch_hours,
                'cf': round(cf, 4),
                'avg_rev_mwh': round(avg_rev, 2),
                'var_cost_mwh': round(var_cost, 2),
                'margin_mwh': round(margin, 2),
                'capacity_mw': cap_mw,
                '_total_mwh': total_mwh,
            }

    # Clean up internal fields and round
    for key in gen_econ:
        del gen_econ[key]['_total_mwh']
        for field in ('cf', 'avg_rev_mwh', 'var_cost_mwh', 'margin_mwh'):
            gen_econ[key][field] = round(gen_econ[key][field], 2)

    return gen_econ


def compute_plant_level_economics(plant_stack, hourly_lmp, dispatch,
                                   demand_mw_profile, fuel_prices, carbon_price,
                                   year=2025):
    """Compute per-plant dispatch economics using plant-level merit order.

    Each plant's position in the merit-order stack determines when it dispatches.
    Returns list of dicts with full per-plant economics.
    """
    if not plant_stack:
        return []

    H_val = len(hourly_lmp)

    # Build cumulative capacity array
    cum_cap = np.zeros(len(plant_stack) + 1)
    for i, plant in enumerate(plant_stack):
        cum_cap[i + 1] = cum_cap[i] + plant['capacity_mw']

    # Residual demand (MW per hour) — fossil must serve
    residual = dispatch.get('residual_demand', np.zeros(H_val))
    if isinstance(residual, (list, tuple)):
        residual = np.array(residual, dtype=np.float64)
    # residual_demand is in same normalized units as demand_norm (sum ≈ 1.0).
    # Convert to MW: residual_mw = residual_norm × total_annual_mwh
    # (Same conversion used in lmp_engine.compute_hourly_lmp_vectorized)
    total_annual_mwh = float(np.sum(demand_mw_profile))
    fossil_demand_mw = residual * total_annual_mwh

    results = []
    for i, plant in enumerate(plant_stack):
        low = cum_cap[i]
        cap_mw = plant['capacity_mw']

        # Hours where this plant dispatches (raw merit-order)
        dispatched = fossil_demand_mw > low
        mw_dispatched = np.clip(fossil_demand_mw - low, 0, cap_mw) * dispatched

        # Apply unit commitment constraints (min up/down times, min gen)
        # Use vintage-aware params for CCGTs (newer units have better turndown)
        uc = get_unit_commitment_params(plant['unit_type'], plant.get('online_year'))
        start_cost_total = 0.0
        n_starts = 0
        if uc:
            dispatched, mw_dispatched, n_starts = apply_unit_commitment(
                dispatched, mw_dispatched, cap_mw, uc)
            start_cost_total = n_starts * uc.get('start_cost_per_mw', 0) * cap_mw

        dispatch_hours = int(np.sum(dispatched))
        total_mwh = float(np.sum(mw_dispatched))
        cf = total_mwh / (cap_mw * H_val) if cap_mw > 0 else 0

        # Revenue
        if total_mwh > 0:
            avg_rev = float(np.sum(hourly_lmp * mw_dispatched)) / total_mwh
        else:
            avg_rev = 0.0

        # Costs
        fuel_key = {'coal_steam': 'coal', 'gas_ccgt': 'gas', 'gas_ct': 'gas', 'oil_ct': 'oil'}.get(plant['unit_type'], 'gas')
        fuel_price = fuel_prices.get(fuel_key, 3.5)
        hr = plant['heat_rate']
        vom = VOM.get(plant['unit_type'], 4.0)
        fuel_cost_mwh = hr * fuel_price
        carbon_cost_mwh = plant['co2_rate'] * carbon_price
        start_cost_mwh = start_cost_total / total_mwh if total_mwh > 0 else 0.0
        total_cost_mwh = vom + fuel_cost_mwh + carbon_cost_mwh + start_cost_mwh
        profit_mwh = avg_rev - total_cost_mwh

        # Emissions
        co2_tons = total_mwh * plant['co2_rate']
        nox_lbs = total_mwh * plant['nox_rate']
        sox_lbs = total_mwh * plant['sox_rate']
        fuel_consumed = total_mwh * hr

        # Status determination
        if profit_mwh < -5:
            status = 'stranded'
        elif profit_mwh <= 2 or cf < 0.10:
            status = 'at_risk'
        else:
            status = 'operating'

        results.append({
            'entity': plant.get('entity_name', ''),
            'plant_name': plant.get('plant_name', ''),
            'plant_id': plant.get('plant_id', ''),
            'generator_id': plant.get('gen_id', ''),
            'state': plant.get('state', ''),
            'county': plant.get('county', ''),
            'latitude': plant.get('latitude'),
            'longitude': plant.get('longitude'),
            'capacity_mw': round(cap_mw, 1),
            'heat_rate_mmbtu_mwh': round(hr, 2),
            'fuel_type': plant.get('fuel_type', ''),
            'prime_mover': plant.get('prime_mover', ''),
            'online_year': plant.get('online_year'),
            'age_years': plant.get('age_years'),
            'capacity_factor': round(cf, 4),
            'mwh_generated': round(total_mwh, 0),
            'fuel_consumed_mmbtu': round(fuel_consumed, 0),
            'co2_tons': round(co2_tons, 1),
            'nox_lbs': round(nox_lbs, 1),
            'sox_lbs': round(sox_lbs, 1),
            'revenue_per_mwh': round(avg_rev, 2),
            'vom_per_mwh': round(vom, 2),
            'fuel_cost_per_mwh': round(fuel_cost_mwh, 2),
            'profit_per_mwh': round(profit_mwh, 2),
            'start_cost_per_mwh': round(start_cost_mwh, 2),
            'n_starts': n_starts,
            'total_revenue_million': round(avg_rev * total_mwh / 1e6, 2),
            'total_cost_million': round(total_cost_mwh * total_mwh / 1e6, 2),
            'total_profit_million': round(profit_mwh * total_mwh / 1e6, 2),
            'status': status,
        })

    return results


def compute_capacity_degradation(iso, clean_pct):
    """Compute capacity price degradation factor using S-curve (sigmoid) model.

    Returns a multiplier in [floor, 1.0] applied to the base capacity price.
    At low clean shares, price stays near base. Through the transition zone
    (midpoint ± ~15%), prices drop steeply. At high clean shares, prices
    approach floor × base_price.

    S-curve matches real RPM/ICAP auction behavior better than linear alpha:
    - PJM RPM clearing prices show sticky behavior below 40% clean, then steep
      decline through 50-70%, flattening above 80%.
    - Linear model over-degrades at low clean shares and under-degrades mid-range.
    """
    params = CAPACITY_DEGRADATION_PARAMS.get(iso, {})
    max_degrade = params.get('max_degrade', 0.0)
    midpoint = params.get('midpoint', 0.50)
    k = params.get('k', 8)
    floor = params.get('floor', 0.0)

    if max_degrade <= 0:
        return 1.0  # No degradation (energy-only or weak capacity market)

    x = clean_pct / 100.0
    sigmoid = 1.0 / (1.0 + np.exp(-k * (x - midpoint)))
    factor = 1.0 - max_degrade * sigmoid
    return max(floor, factor)


def compute_nuclear_revenue(iso, clean_pct, hourly_lmp, year, conditions=None):
    """Compute nuclear plant revenue stack by ISO.

    Returns dict with energy_rev, capacity_rev, ptc, total (all $/MWh).
    Uses 45U contract-for-difference floor mechanism when conditions provided.
    """
    nuclear_cf = 0.93
    nuclear_gen = np.ones(H) * nuclear_cf  # Flat baseload

    # Energy revenue: LMP × generation
    energy_rev = float(np.mean(hourly_lmp * nuclear_gen)) / nuclear_cf

    # Capacity revenue — S-curve degradation
    base_price = CAPACITY_MARKET_PRICES.get(iso, 0)
    if conditions and conditions.get('capacity_market_price') is not None:
        base_price = conditions['capacity_market_price']
    degrade_factor = compute_capacity_degradation(iso, clean_pct)
    degraded_price = base_price * degrade_factor
    cap_rev = degraded_price * 1.0 / (nuclear_cf * 8.760)  # ELCC=1.0 for nuclear

    # PTC 45U — contract-for-difference floor mechanism
    ptc_max = conditions.get('ptc_45u_max', PTC_45U_VALUE) if conditions else PTC_45U_VALUE
    floor_base = conditions.get('ptc_45u_floor', 40.0) if conditions else 40.0
    escalation = conditions.get('ptc_45u_floor_escalation', 0.0) if conditions else 0.0
    sunset_year = conditions.get('ptc_45u_sunset_year', PTC_45U_SUNSET_YEAR) if conditions else PTC_45U_SUNSET_YEAR

    if year <= sunset_year:
        # Escalate floor from base year (2024)
        years_elapsed = max(0, year - 2024)
        floor_price = floor_base * (1 + escalation / 100.0) ** years_elapsed
        # All-in revenue before PTC
        all_in_rev = energy_rev + cap_rev
        # CfD: credit fills gap between floor and revenue, capped at ptc_max
        ptc = max(0.0, min(ptc_max, floor_price - all_in_rev))
    else:
        ptc = 0.0

    total = energy_rev + cap_rev + ptc

    return {
        'energy_rev_mwh': round(energy_rev, 2),
        'capacity_rev_mwh': round(cap_rev, 2),
        'ptc_mwh': round(ptc, 2),
        'total_mwh': round(total, 2),
    }


def compute_energy_revenue_by_resource(hourly_lmp, supply_profiles, resource_pcts,
                                        demand_total_mwh):
    """Compute energy revenue ($/MWh) per resource from hourly LMP × generation."""
    revenues = {}
    for res, pct in resource_pcts.items():
        if pct <= 0:
            continue
        profile = supply_profiles.get(res)
        if profile is None:
            continue
        gen_profile = np.array(profile, dtype=np.float64) * (pct / 100.0)
        gen_mwh = float(np.sum(gen_profile)) * demand_total_mwh / H
        if gen_mwh > 0:
            revenue_weighted = float(np.sum(gen_profile * hourly_lmp))
            gen_total = float(np.sum(gen_profile))
            revenues[res] = revenue_weighted / gen_total if gen_total > 0 else 0
    return revenues


def compute_capacity_revenue(iso, clean_pct, resource_pcts, conditions=None):
    """Compute capacity market revenue ($/MWh) per resource."""
    base_price = CAPACITY_MARKET_PRICES.get(iso, 0)
    if conditions and conditions.get('capacity_market_price') is not None:
        base_price = conditions['capacity_market_price']
    degrade_factor = compute_capacity_degradation(iso, clean_pct)
    degraded_price = base_price * degrade_factor

    cap_revs = {}
    for res, pct in resource_pcts.items():
        if pct <= 0:
            continue
        elcc = PEAK_CAPACITY_CREDITS.get(res, 0)
        cf = RESOURCE_CAPACITY_FACTORS.get(res, {}).get(iso, 0.30)
        if cf > 0:
            cap_revs[res] = degraded_price * elcc / (cf * 8.760)
        else:
            cap_revs[res] = 0
    return cap_revs


def get_compliance_eligible_pct(resource_pcts, iso):
    """Clean % counting only compliance-eligible resources."""
    eligible = CES_ELIGIBLE if iso in CES_ISOS else REC_ELIGIBLE
    return sum(pct for res, pct in resource_pcts.items()
               if res in eligible and pct > 0)


def compute_rec_price(iso, eligible_pct, year, rec_price_override=None):
    """Scarcity-driven compliance REC price ($/MWh)."""
    if rec_price_override is not None:
        return rec_price_override
    acp = ACP_RATES.get(iso, 0)
    if acp <= 0:
        return VOLUNTARY_REC_FLOOR.get(iso, 0)

    rps_target_frac = get_rps_target_at_year(iso, year)
    vol_adder = VOLUNTARY_DEMAND_ADDER.get(iso, 0)
    eff_target_pct = (rps_target_frac + vol_adder) * 100.0
    gap = eff_target_pct - eligible_pct

    floor = VOLUNTARY_REC_FLOOR.get(iso, 1.0)
    k_scarcity = REC_SCARCITY_K.get(iso, 0.15)
    compliance_2025 = REC_COMPLIANCE_PRICE_2025.get(iso, 5.0)

    if gap > 0:
        price = acp * (1.0 - np.exp(-k_scarcity * gap))
    else:
        price = floor + (compliance_2025 - floor) * np.exp(REC_SURPLUS_DECAY_K * gap)

    return max(floor, min(acp, price))


def compute_rec_revenue(iso, resource_pcts, clean_pct, year, rec_price_override=None):
    """REC/CES revenue for eligible resources ($/MWh)."""
    eligible_pct = get_compliance_eligible_pct(resource_pcts, iso)
    rec_price = compute_rec_price(iso, eligible_pct, year, rec_price_override=rec_price_override)

    result = {}
    for res, pct in resource_pcts.items():
        if pct <= 0:
            continue
        if res in REC_ELIGIBLE:
            result[res] = rec_price
        elif res in CES_ELIGIBLE and iso in CES_ISOS:
            result[res] = rec_price * CES_DISCOUNT_FACTOR
        else:
            result[res] = 0
    return result


def compute_zone_revenue(iso, clean_pct, resource_pcts, hourly_lmp,
                          supply_profiles, demand_total_mwh, year,
                          rec_price_override=None, conditions=None):
    """Total blended revenue ($/MWh) for resources at this zone."""
    energy_revs = compute_energy_revenue_by_resource(
        hourly_lmp, supply_profiles, resource_pcts, demand_total_mwh)
    cap_revs = compute_capacity_revenue(iso, clean_pct, resource_pcts, conditions=conditions)
    rec_revs = compute_rec_revenue(iso, resource_pcts, clean_pct, year,
                                    rec_price_override=rec_price_override)

    total_pct = sum(pct for pct in resource_pcts.values() if pct > 0)
    if total_pct <= 0:
        return 0, {}, {'energy_rev_mwh': 0, 'capacity_rev_mwh': 0, 'rec_rev_mwh': 0}

    per_resource_rev = {}
    blended = 0
    blended_energy = blended_cap = blended_rec = 0
    for res, pct in resource_pcts.items():
        if pct <= 0:
            continue
        e_rev = energy_revs.get(res, 0)
        c_rev = cap_revs.get(res, 0)
        r_rev = rec_revs.get(res, 0)
        rev = e_rev + c_rev + r_rev
        per_resource_rev[res] = round(rev, 2)
        weight = pct / total_pct
        blended += rev * weight
        blended_energy += e_rev * weight
        blended_cap += c_rev * weight
        blended_rec += r_rev * weight

    breakdown = {
        'energy_rev_mwh': round(blended_energy, 2),
        'capacity_rev_mwh': round(blended_cap, 2),
        'rec_rev_mwh': round(blended_rec, 2),
    }
    return round(blended, 2), per_resource_rev, breakdown


# ═══════════════════════════════════════════════════════════════════════════════
# COST MODEL
# ═══════════════════════════════════════════════════════════════════════════════

# Baseline PTC/ITC assumptions baked into LCOE tables (for delta approach)
BASELINE_PTC_IN_LCOE = {'solar': 26.0, 'wind': 26.0}
BASELINE_ITC_PCT = 30.0  # ITC already in offshore wind / storage LCOE tables


def get_resource_lcoe(res, iso, lcoe_level, cumulative_gw, learning_speed, year,
                       conditions=None):
    """Get effective LCOE after Wright's Law learning.

    When conditions dict includes custom_lcoes, those override the entire LCOE
    pipeline (no Wright's Law, no PTC subtraction). PTC/ITC delta adjustments
    only apply when using the default LCOE tables.
    """
    # Phase 6: Custom LCOE overrides replace everything
    if conditions:
        custom_lcoes = conditions.get('custom_lcoes')
        if custom_lcoes:
            lcoe_key_map = {'clean_firm': 'nuclear', 'ccs_ccgt': 'ccs_ccgt', 'geothermal': 'geothermal'}
            key = lcoe_key_map.get(res, res)
            if key in custom_lcoes and custom_lcoes[key] is not None:
                return custom_lcoes[key]

    tech = RESOURCE_TO_TECH.get(res, res)

    if res == 'clean_firm':
        base_lcoe = NUCLEAR_NEWBUILD_LCOE.get(lcoe_level, {}).get(iso, 100)
        foak = FOAK_NUCLEAR_NEWBUILD.get(iso, 175)
        noak = NUCLEAR_NEWBUILD_LCOE.get('Low', {}).get(iso, 68)
        lr = WRIGHT_LEARNING_RATE.get('nuclear', {}).get(learning_speed, 0.12)
        ref_gw = WRIGHT_CUMULATIVE_GW_2025.get('nuclear', 2.0)
        eff_gw = get_effective_cumulative_gw('nuclear', cumulative_gw.get('nuclear', 0),
                                              learning_speed, year)
        cost = wright_cost(foak, noak, eff_gw, ref_gw, lr)
        # Phase 2A: Parameterized 45Y PTC for new nuclear
        ptc_new_nuc = conditions.get('ptc_nuclear_new', PTC_45Y_NEW_NUCLEAR) if conditions else PTC_45Y_NEW_NUCLEAR
        cost = max(noak, cost - ptc_new_nuc)
        return cost

    elif res == 'ccs_ccgt':
        # 3-tier logic: custom CCS credit override > 45Q toggle > default ON
        ccs_credit_override = conditions.get('ccs_credit_override') if conditions else None

        if ccs_credit_override is not None:
            # User specified exact $/MWh credit — apply as offset from 45Q-OFF table
            ccs_table = CCS_LCOE_45Q_OFF
            base_lcoe = ccs_table.get(lcoe_level, {}).get(iso, 120) - ccs_credit_override
            foak = FOAK_CCS_45Q_OFF.get(iso, 130) - ccs_credit_override
            noak = ccs_table.get('Low', {}).get(iso, 80) - ccs_credit_override
        else:
            q45_on = conditions.get('q45', True) if conditions else True
            if q45_on:
                ccs_table = CCS_LCOE_45Q_ON
                foak = FOAK_CCS_45Q_ON.get(iso, 130)
            else:
                ccs_table = CCS_LCOE_45Q_OFF
                foak = FOAK_CCS_45Q_OFF.get(iso, 130)
            base_lcoe = ccs_table.get(lcoe_level, {}).get(iso, 120)
            noak = ccs_table.get('Low', {}).get(iso, 80)

        lr = WRIGHT_LEARNING_RATE.get('ccs', {}).get(learning_speed, 0.10)
        ref_gw = WRIGHT_CUMULATIVE_GW_2025.get('ccs', 0.3)
        eff_gw = get_effective_cumulative_gw('ccs', cumulative_gw.get('ccs', 0),
                                              learning_speed, year)
        return wright_cost(foak, noak, eff_gw, ref_gw, lr)

    elif res == 'geothermal':
        base_lcoe = GEOTHERMAL_LCOE.get(lcoe_level, 88)
        foak = FOAK_GEOTHERMAL if isinstance(FOAK_GEOTHERMAL, (int, float)) else 150
        noak = GEOTHERMAL_LCOE.get('Low', 63)
        lr = WRIGHT_LEARNING_RATE.get('geothermal', {}).get(learning_speed, 0.15)
        ref_gw = WRIGHT_CUMULATIVE_GW_2025.get('geothermal', 0.05)
        eff_gw = get_effective_cumulative_gw('geothermal', cumulative_gw.get('geothermal', 0),
                                              learning_speed, year)
        return wright_cost(foak, noak, eff_gw, ref_gw, lr)

    elif res in ('solar', 'wind', 'offshore_wind'):
        tables = LCOE_TABLES.get(res, {})
        base = tables.get(lcoe_level, {}).get(iso, 50)

        if conditions:
            # Phase 2C: Solar/Wind PTC delta adjustment
            if res in ('solar', 'wind'):
                user_ptc = conditions.get(f'ptc_{res}', BASELINE_PTC_IN_LCOE.get(res, 26.0))
                delta = user_ptc - BASELINE_PTC_IN_LCOE.get(res, 26.0)
                base -= delta  # More credit → lower LCOE; less credit → higher LCOE

            # Phase 3: ITC delta adjustment for offshore wind
            elif res == 'offshore_wind':
                user_itc = conditions.get('itc_pct', BASELINE_ITC_PCT)
                if user_itc != BASELINE_ITC_PCT:
                    capital_fraction = 0.80  # ~80% of offshore wind LCOE is capital
                    itc_delta = (user_itc - BASELINE_ITC_PCT) / 100.0
                    base *= (1 - capital_fraction * itc_delta / (1 - BASELINE_ITC_PCT / 100))

        return max(0, base)

    elif res == 'hydro':
        # Phase 5A: Wholesale price override for hydro
        if conditions and conditions.get('wholesale_price_override') is not None:
            return conditions['wholesale_price_override']
        return WHOLESALE_PRICES.get(iso, 30)

    else:
        tables = LCOE_TABLES.get(res, {})
        base = tables.get(lcoe_level, {}).get(iso, 50)
        if conditions and res in ('battery', 'battery8', 'ldes'):
            # Custom storage cost override (converted from $/kW-yr to $/MWh in main.py)
            custom_storage = conditions.get('custom_storage_lcoe')
            if custom_storage and custom_storage.get(res) is not None:
                base = custom_storage[res]
            # ITC delta adjustment for storage
            user_itc = conditions.get('itc_pct', BASELINE_ITC_PCT)
            if user_itc != BASELINE_ITC_PCT:
                capital_fraction = 0.85  # ~85% of storage LCOE is capital
                itc_delta = (user_itc - BASELINE_ITC_PCT) / 100.0
                base *= (1 - capital_fraction * itc_delta / (1 - BASELINE_ITC_PCT / 100))
        return max(0, base)


def _get_ppa_discount(res, ppa_level, iso=None):
    """PPA risk premium discount, scaled by regional market depth."""
    category = 'VRE' if res in ('solar', 'wind', 'offshore_wind') else 'Firm'
    base_discount = PPA_PREMIUMS.get(category, {}).get(ppa_level, 0)
    depth = PPA_MARKET_DEPTH.get(iso, 0.75) if iso else 1.0
    return base_discount * depth


def compute_zone_cost(iso, delta_resources, lcoe_level, cumulative_gw,
                       learning_speed, year, tx_level='Medium', ppa_level=None,
                       conditions=None):
    """Compute blended LCOE ($/MWh) for incremental resources."""
    per_resource_cost = {}
    total_twh = 0
    weighted_cost = 0

    for res, delta_twh in delta_resources.items():
        if delta_twh <= 0:
            continue
        lcoe = get_resource_lcoe(res, iso, lcoe_level, cumulative_gw,
                                  learning_speed, year, conditions=conditions)

        if res in ('solar', 'wind', 'clean_firm', 'offshore_wind'):
            tx = get_tx(res if res != 'clean_firm' else 'clean_firm', tx_level, iso)
            lcoe += tx

        if ppa_level is not None:
            discount = _get_ppa_discount(res, ppa_level, iso)
            lcoe *= (1 - discount)

        per_resource_cost[res] = round(lcoe, 2)
        weighted_cost += lcoe * delta_twh
        total_twh += delta_twh

    blended = weighted_cost / total_twh if total_twh > 0 else 0
    return round(blended, 2), per_resource_cost


# ═══════════════════════════════════════════════════════════════════════════════
# CCS RETROFIT BREAKEVEN
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ccs_retrofit_breakeven(iso, fuel_level='Medium', conditions=None):
    """Find carbon price where CCS retrofit beats continued unabated CCGT operation.

    Returns dict with breakeven carbon prices for:
    - ccs_vs_existing: When CCS retrofit is cheaper than running existing CCGT
    - new_gas_vs_old: When new efficient gas beats old inefficient gas
    """
    gas_hr = HEAT_RATES.get('gas_ccgt', 7.0)
    gas_vom = VOM.get('gas_ccgt', 3.5)
    gas_co2 = CO2_RATES.get('gas_ccgt', 0.37)
    fuel_price = FUEL_PRICES.get(fuel_level, {}).get('gas', 3.50)

    # Existing CCGT variable cost (rises with carbon price)
    # MC_existing = HR × fuel + VOM + CO2_rate × carbon_price
    existing_fixed_cost = gas_hr * fuel_price + gas_vom

    # CCS CCGT: ~90% capture, higher fixed cost, lower variable emissions
    # Select CCS LCOE based on 45Q toggle / credit override
    ccs_credit_override = conditions.get('ccs_credit_override') if conditions else None
    if ccs_credit_override is not None:
        ccs_lcoe = CCS_LCOE_45Q_OFF.get('Medium', {}).get(iso, 100) - ccs_credit_override
    else:
        q45_on = conditions.get('q45', True) if conditions else True
        ccs_table = CCS_LCOE_45Q_ON if q45_on else CCS_LCOE_45Q_OFF
        ccs_lcoe = ccs_table.get('Medium', {}).get(iso, 100)
    ccs_co2_rate = gas_co2 * 0.10  # 90% capture → 10% residual

    # Breakeven: existing_cost + gas_co2 × Cp = ccs_lcoe + ccs_co2 × Cp
    # Cp × (gas_co2 - ccs_co2) = ccs_lcoe - existing_fixed_cost
    co2_diff = gas_co2 - ccs_co2_rate
    if co2_diff > 0:
        breakeven_ccs = (ccs_lcoe - existing_fixed_cost) / co2_diff
    else:
        breakeven_ccs = float('inf')

    # New efficient gas (HR 6.2) vs old gas (HR 8.0+)
    new_hr = 6.2
    old_hr = 8.5
    new_cost = new_hr * fuel_price + 3.0  # Lower VOM for new
    old_cost = old_hr * fuel_price + 5.0  # Higher VOM for old
    new_co2 = 0.053 * new_hr  # ~0.053 tCO2/MMBtu × HR
    old_co2 = 0.053 * old_hr

    co2_diff_gas = old_co2 - new_co2
    cost_diff_gas = new_cost - old_cost  # New is cheaper on fuel, but consider CAPEX
    new_gas_capex_adder = 8.0  # $/MWh annualized CAPEX for new build

    if co2_diff_gas > 0:
        breakeven_new_gas = (new_gas_capex_adder + cost_diff_gas) / co2_diff_gas
    else:
        breakeven_new_gas = float('inf')

    return {
        'ccs_vs_existing_carbon_price': round(max(0, breakeven_ccs), 1),
        'new_gas_vs_old_carbon_price': round(max(0, breakeven_new_gas), 1),
        'existing_ccgt_var_cost': round(existing_fixed_cost, 2),
        'ccs_lcoe': round(ccs_lcoe, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LCOE MERIT-ORDER DEPLOYMENT MODEL
# ═══════════════════════════════════════════════════════════════════════════════

# Deployable clean resource types and their typical capacity factors by ISO
DEPLOYABLE_RESOURCES = [
    'solar', 'wind', 'offshore_wind', 'clean_firm', 'ccs_ccgt', 'geothermal',
]

# Maximum capacity per resource type (TWh/yr per ISO) — physical/permitting limits
RESOURCE_CAP_TWH = {
    'geothermal': {'CAISO': 39.0},  # Only available in CAISO
    'offshore_wind': {  # Lease area constraints
        'CAISO': 20.0, 'NYISO': 25.0, 'NEISO': 20.0, 'PJM': 15.0,
    },
    'ccs_ccgt': {  # CO2 transport/storage pipeline constraints
        'CAISO': 30.0, 'ERCOT': 50.0, 'PJM': 40.0, 'NYISO': 15.0,
        'NEISO': 10.0, 'MISO': 45.0, 'SPP': 35.0,
    },
}


def compute_market_deployment(iso, year, demand_twh, current_clean_pct,
                               conditions, cumulative_gw, queue_remaining_gw,
                               hourly_lmp, avg_lmp, p90_lmp,
                               supply_profiles_iso, demand_total_mwh,
                               gen_econ, state):
    """Pure economics-driven resource deployment via LCOE merit order.

    Ranks all available clean resources by net LCOE (after incentives, learning
    curves, PPA discounts, transmission). Deploys cheapest first as long as
    revenue > cost. Stops when no more profitable resources or queue cap hit.

    Clean energy percentage is purely an OUTPUT of this function, not an input.

    Args:
        iso: ISO region string
        year: Simulation year
        demand_twh: Total demand in TWh for this year
        current_clean_pct: Current clean energy percentage (from prior years)
        conditions: Dict with lcoe_level, fuel_level, tx_level, etc.
        cumulative_gw: Dict tracking cumulative GW deployed per tech
        queue_remaining_gw: Remaining GW that can be interconnected this period
        hourly_lmp: 8760 array of hourly LMP values
        avg_lmp: Average LMP ($/MWh)
        p90_lmp: P90 LMP
        supply_profiles_iso: Generation profile dict for this ISO
        demand_total_mwh: Total demand in MWh
        gen_econ: Generator economics dict from LMP calculation
        state: Mutable iso_state dict

    Returns:
        (new_clean_pct, deployed_resources, zone_results, rev_breakdown)
        where deployed_resources is {resource: twh_deployed}
    """
    lcoe_level = conditions.get('lcoe_level', 'Medium')
    fuel_level = conditions.get('fuel_level', 'Medium')
    tx_level = conditions.get('tx_level', 'Medium')
    learning_speed = conditions.get('learning_speed', 'Medium')
    ppa_level = conditions.get('ppa_level', 'Medium')

    # Estimate revenue available for clean resources
    # Revenue = energy price (avg LMP serves as proxy) + capacity + REC
    base_energy_rev = avg_lmp  # $/MWh energy revenue

    # Compute per-resource LCOE and rank by net cost
    resource_economics = []
    for res in DEPLOYABLE_RESOURCES:
        # Skip resources not available in this ISO
        if res == 'geothermal' and iso != 'CAISO':
            continue
        if res == 'offshore_wind' and iso not in ('CAISO', 'NYISO', 'NEISO', 'PJM'):
            continue

        # Get LCOE after learning curves and incentives
        lcoe = get_resource_lcoe(res, iso, lcoe_level, cumulative_gw,
                                  learning_speed, year, conditions=conditions)

        # Add transmission cost
        if res in ('solar', 'wind', 'clean_firm', 'offshore_wind', 'ccs_ccgt', 'geothermal'):
            tx = get_tx(res if res != 'clean_firm' else 'clean_firm', tx_level, iso)
            lcoe += tx

        # Apply PPA discount
        if ppa_level is not None:
            discount = _get_ppa_discount(res, ppa_level, iso)
            lcoe *= (1 - discount)

        # Estimate capacity factor
        cf = RESOURCE_CAPACITY_FACTORS.get(res, {}).get(iso, 0.25)
        if res == 'clean_firm':
            cf = 0.93  # Nuclear CF
        elif res == 'ccs_ccgt':
            cf = 0.85  # CCS baseload
        elif res == 'geothermal':
            cf = 0.90

        # Resource-specific revenue adjustments
        capacity_rev = 0
        rec_rev = 0
        base_cap_price = conditions.get('capacity_market_price') or CAPACITY_MARKET_PRICES.get(iso, 0)
        if base_cap_price > 0:
            # Capacity credit varies by resource
            cap_credit = PEAK_CAPACITY_CREDITS.get(res, 0)
            # Convert $/kW-yr to $/MWh
            capacity_rev = base_cap_price * cap_credit / (cf * 8.760) if cf > 0 else 0

        # REC revenue for eligible resources
        if res in REC_ELIGIBLE:
            rec_rev = compute_rec_revenue(iso, {res: 100}, current_clean_pct, year,
                                           rec_price_override=conditions.get('rec_price_override'))
            rec_rev = rec_rev.get(res, 0)

        total_revenue = base_energy_rev + capacity_rev + rec_rev
        net_profit = total_revenue - lcoe

        # Max TWh deployable for this resource (physical limits)
        max_twh = RESOURCE_CAP_TWH.get(res, {}).get(iso, 999)
        # Already deployed — subtract from cap
        already_deployed_gw = cumulative_gw.get(RESOURCE_TO_TECH.get(res, res), 0)
        already_deployed_twh = already_deployed_gw * cf * 8.760 if cf > 0 else 0

        resource_economics.append({
            'resource': res,
            'lcoe': round(lcoe, 2),
            'revenue': round(total_revenue, 2),
            'profit': round(net_profit, 2),
            'cf': cf,
            'max_twh': max_twh - already_deployed_twh,
            'capacity_rev': round(capacity_rev, 2),
            'rec_rev': round(rec_rev, 2),
        })

    # Sort by LCOE ascending (cheapest first) — deploy profitable ones
    resource_economics.sort(key=lambda x: x['lcoe'])

    deployed = {}
    total_deployed_twh = 0
    zone_results = []
    remaining_gw = queue_remaining_gw
    clean_pct = current_clean_pct

    for entry in resource_economics:
        if remaining_gw <= 0:
            break
        if entry['profit'] <= 0:
            continue  # Not profitable
        if entry['max_twh'] <= 0:
            continue  # Resource cap reached

        res = entry['resource']
        cf = entry['cf']

        # How much can we deploy given queue cap?
        max_deploy_gw = remaining_gw
        max_deploy_twh = max_deploy_gw * cf * 8.760 if cf > 0 else 0

        # Constrain by resource cap and remaining demand headroom
        max_clean_headroom_twh = (99.99 - clean_pct) / 100.0 * demand_twh
        deploy_twh = min(max_deploy_twh, entry['max_twh'], max_clean_headroom_twh)

        if deploy_twh <= 0:
            continue

        # Convert back to GW
        deploy_gw = deploy_twh / (cf * 8.760) if cf > 0 else 0

        # Deploy
        deployed[res] = deploy_twh
        total_deployed_twh += deploy_twh
        remaining_gw -= deploy_gw

        # Update cumulative GW
        tech = RESOURCE_TO_TECH.get(res, res)
        cumulative_gw[tech] = cumulative_gw.get(tech, 0) + deploy_gw

        # Update clean%
        clean_increase_pct = deploy_twh / demand_twh * 100
        clean_pct += clean_increase_pct

        zone_results.append({
            'resource': res,
            'threshold': round(clean_pct, 1),
            'twh': round(deploy_twh, 2),
            'gw': round(deploy_gw, 2),
            'new_gw': round(deploy_gw, 2),
            'lcoe': entry['lcoe'],
            'cost': entry['lcoe'],
            'revenue': entry['revenue'],
            'profit': entry['profit'],
            'avg_lmp': round(avg_lmp, 1),
            'energy_rev_mwh': round(base_energy_rev, 2),
            'capacity_rev_mwh': round(entry['capacity_rev'], 2),
            'rec_rev_mwh': round(entry['rec_rev'], 2),
        })

    # Build revenue breakdown from deployed mix
    total_energy = 0
    total_cap = 0
    total_rec = 0
    if total_deployed_twh > 0:
        for entry in resource_economics:
            twh = deployed.get(entry['resource'], 0)
            if twh > 0:
                weight = twh / total_deployed_twh
                total_energy += base_energy_rev * weight
                total_cap += entry['capacity_rev'] * weight
                total_rec += entry['rec_rev'] * weight

    rev_breakdown = {
        'energy_rev_mwh': round(total_energy, 2),
        'capacity_rev_mwh': round(total_cap, 2),
        'rec_rev_mwh': round(total_rec, 2),
    }

    blended_cost = 0
    blended_revenue = 0
    if total_deployed_twh > 0:
        for entry in resource_economics:
            twh = deployed.get(entry['resource'], 0)
            if twh > 0:
                weight = twh / total_deployed_twh
                blended_cost += entry['lcoe'] * weight
                blended_revenue += entry['revenue'] * weight

    return (
        round(clean_pct, 2),
        deployed,
        zone_results,
        rev_breakdown,
        round(blended_cost, 2),
        round(blended_revenue, 2),
        remaining_gw,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_step3_data():
    """Load step2.2 cost optimization results for all ISOs.

    Searches for parquet files in priority order:
      1. step_2_2a_CO_{ISO}.parquet  (current naming convention)
      2. step3_co_{ISO}.parquet       (legacy naming)

    Searches in directories:
      1. {MODULE_ROOT}/data/step2.2-cost/     (local market-simulator data)
    Returns {iso: {threshold_float: result_dict}}.
    """
    search_dirs = [
        os.path.join(MODULE_ROOT, 'data', 'step2.2-cost'),
    ]

    all_data = {}
    for iso in ISOS:
        path = None
        # Search for parquet files in priority order across directories
        for d in search_dirs:
            if not os.path.isdir(d):
                continue
            for pattern in [f'step_2_2a_CO_{iso}.parquet', f'step3_co_{iso}.parquet']:
                candidate = os.path.join(d, pattern)
                if os.path.exists(candidate):
                    path = candidate
                    break
            if path:
                break

        if path is None:
            print(f"  WARNING: No cost parquet for {iso}, skipping")
            continue

        print(f"  Loaded {os.path.basename(path)} for {iso}")
        df = pd.read_parquet(path)
        iso_data = {}
        for t_val, grp in df.groupby('threshold'):
            t = float(t_val)
            best_idx = grp['cost_total_cost'].idxmin()
            row = grp.loc[best_idx]

            resource_pcts = {
                'clean_firm': float(row.get('mix_clean_firm', 0)),
                'solar': float(row.get('mix_solar', 0)),
                'wind': float(row.get('mix_wind', 0)),
                'offshore_wind': float(row.get('mix_offshore_wind', 0)),
                'ccs_ccgt': float(row.get('mix_ccs_ccgt', 0)),
                'hydro': float(row.get('mix_hydro', 0)),
            }

            iso_data[t] = {
                'resource_pcts': resource_pcts,
                'total_cost': float(row.get('cost_total_cost', 0)),
                'hourly_match_score': float(row.get('hourly_match_score', 0)),
                'battery_pct': float(row.get('battery_dispatch_pct', 0)),
                'battery8_pct': float(row.get('battery8_dispatch_pct', 0)),
                'ldes_pct': float(row.get('ldes_dispatch_pct', 0)),
                'h2_pct': float(row.get('h2_dispatch_pct', 0)),
                'gas_backup_mw': float(row.get('gas_gas_backup_needed_mw', 0)),
                'new_gas_mw': float(row.get('gas_new_gas_build_mw', 0)),
                'gas_cost_per_mwh': float(row.get('gas_gas_cost_per_mwh', 0)),
            }

        all_data[iso] = iso_data

    # If no parquet data found, generate synthetic threshold data
    if not all_data:
        print("  INFO: No step2.2 parquets found — generating synthetic threshold data")
        all_data = _generate_synthetic_step3_data()

    return all_data


def _generate_synthetic_step3_data():
    """Generate synthetic resource mix data for each ISO when parquets are absent.

    Produces a reasonable set of threshold → resource_mix mappings based on
    known grid characteristics. This enables the tool to run standalone for
    screening-level analysis without requiring the full pipeline.
    """
    from pipeline_config import ISOS, GRID_MIX_SHARES

    # Typical resource ramp patterns per ISO
    iso_profiles = {
        'CAISO': {'solar_max': 35, 'wind_max': 15, 'firm_max': 12, 'offshore_max': 0, 'hydro': 10, 'ccs_max': 5},
        'ERCOT': {'solar_max': 25, 'wind_max': 30, 'firm_max': 8, 'offshore_max': 3, 'hydro': 1, 'ccs_max': 5},
        'PJM':   {'solar_max': 20, 'wind_max': 20, 'firm_max': 15, 'offshore_max': 8, 'hydro': 2, 'ccs_max': 8},
        'NYISO': {'solar_max': 15, 'wind_max': 15, 'firm_max': 10, 'offshore_max': 10, 'hydro': 15, 'ccs_max': 5},
        'NEISO': {'solar_max': 12, 'wind_max': 18, 'firm_max': 8, 'offshore_max': 15, 'hydro': 8, 'ccs_max': 5},
        'MISO':  {'solar_max': 20, 'wind_max': 30, 'firm_max': 10, 'offshore_max': 0, 'hydro': 3, 'ccs_max': 8},
        'SPP':   {'solar_max': 20, 'wind_max': 35, 'firm_max': 5, 'offshore_max': 0, 'hydro': 3, 'ccs_max': 3},
    }

    thresholds = [50, 55, 60, 65, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 99.9]
    all_data = {}

    for iso in ISOS:
        profile = iso_profiles.get(iso, iso_profiles['PJM'])
        existing_clean = sum(GRID_MIX_SHARES.get(iso, {}).values())
        iso_data = {}

        for t in thresholds:
            if t <= existing_clean:
                continue

            # Linear ramp from existing clean to target, allocating resources
            progress = min(1.0, (t - existing_clean) / (99.9 - existing_clean))

            solar = profile['solar_max'] * progress
            wind = profile['wind_max'] * progress
            firm = profile['firm_max'] * progress * min(1.0, progress * 1.5)  # firm ramps faster at high targets
            offshore = profile['offshore_max'] * progress
            ccs = profile['ccs_max'] * max(0, progress - 0.3) / 0.7 if progress > 0.3 else 0
            hydro = profile['hydro']  # hydro is existing, doesn't change

            # Scale to hit target
            total_new = solar + wind + firm + offshore + ccs
            remaining = t - existing_clean - hydro
            if total_new > 0 and remaining > 0:
                scale = remaining / total_new
                solar *= scale
                wind *= scale
                firm *= scale
                offshore *= scale
                ccs *= scale

            resource_pcts = {
                'clean_firm': round(firm + GRID_MIX_SHARES.get(iso, {}).get('clean_firm', 0), 2),
                'solar': round(solar + GRID_MIX_SHARES.get(iso, {}).get('solar', 0), 2),
                'wind': round(wind + GRID_MIX_SHARES.get(iso, {}).get('wind', 0), 2),
                'offshore_wind': round(offshore, 2),
                'ccs_ccgt': round(ccs, 2),
                'hydro': round(hydro, 2),
            }

            # Storage ramps with clean % (more needed at high targets)
            bat_pct = min(15, progress * 12)
            bat8_pct = min(8, max(0, progress - 0.3) * 10)
            ldes_pct = min(5, max(0, progress - 0.5) * 8)
            h2_pct = min(3, max(0, progress - 0.8) * 10) if t >= 95 else 0

            iso_data[t] = {
                'resource_pcts': resource_pcts,
                'total_cost': 0,  # no cost data without parquets
                'hourly_match_score': t / 100.0,
                'battery_pct': round(bat_pct, 1),
                'battery8_pct': round(bat8_pct, 1),
                'ldes_pct': round(ldes_pct, 1),
                'h2_pct': round(h2_pct, 1),
                'gas_backup_mw': 0,
                'new_gas_mw': 0,
                'gas_cost_per_mwh': 0,
            }

        all_data[iso] = iso_data

    return all_data


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def _map_price_sens_to_lcoe_fuel_tx(sens):
    """Map 9-dim price sensitivity dict to lcoe_level/fuel_level/tx_level."""
    level_map = {'L': 'Low', 'M': 'Medium', 'H': 'High'}
    return {
        'lcoe_level': level_map.get(sens['firm'], sens['firm']),
        'fuel_level': sens['fuel'],
        'tx_level': sens['tx'],
        '_price_sens': dict(sens),
    }


def build_market_scenarios():
    """Generate all parametric sweep scenarios.

    3 demand × 5 price × 3 PPA × 3 gas friction × 3 queue = 405 scenarios.
    No emission constraints, no NZ targets — purely market-driven reference trajectory.

    Returns list of (scenario_id_str, conditions_dict).
    """
    combos = []
    demand_keys = DEMAND_GROWTH_LEVELS
    price_keys = list(PRICE_SENSITIVITIES.keys())
    ppa_keys = PPA_LEVELS
    gas_keys = list(GAS_FRICTION_LEVELS.keys())
    queue_keys = ['Low', 'Medium', 'High']

    for demand, price_name, ppa, gas_name, queue in cartesian(
            demand_keys, price_keys, ppa_keys, gas_keys, queue_keys):

        price_mapping = _map_price_sens_to_lcoe_fuel_tx(PRICE_SENSITIVITIES[price_name])

        demand_code = demand[0]
        ppa_code = ppa[0]
        gas_code = gas_name[0]
        queue_code = queue[0]
        scenario_id = f"MKT_{demand_code}_{price_name}_{ppa_code}_{gas_code}_{queue_code}"

        conditions = {
            'name': f"Market: {demand} demand | {price_name} | PPA={ppa} | Gas={gas_name} | Queue={queue}",
            'demand_growth': demand,
            'lcoe_level': price_mapping['lcoe_level'],
            'learning_speed': QUEUE_LEARNING_MAP.get(queue, 'Medium'),
            'queue_cap_level': queue,
            'gas_friction': GAS_FRICTION_LEVELS[gas_name],
            'carbon_price': 0,
            'fuel_level': price_mapping['fuel_level'],
            'tx_level': price_mapping['tx_level'],
            'ppa_level': ppa,
            '_price_sens_name': price_name,
            '_price_sens': price_mapping.get('_price_sens', {}),
        }

        combos.append((scenario_id, conditions))

    return combos


def build_single_scenario(overrides=None):
    """Build a single scenario with user overrides.

    Returns (scenario_id, conditions_dict).
    """
    defaults = {
        'name': 'Custom Market Scenario',
        'demand_growth': 'Medium',
        'lcoe_level': 'Medium',
        'learning_speed': 'Medium',
        'queue_cap_level': 'Medium',
        'gas_friction': 0.7,
        'carbon_price': 0,
        'fuel_level': 'Medium',
        'tx_level': 'Medium',
        'ppa_level': 'Medium',
    }
    if overrides:
        defaults.update(overrides)
    return ('MKT_CUSTOM', defaults)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_demand_at_year(iso, year, growth_level):
    """Project demand TWh at a future year from 2023 baseline."""
    base = REGIONAL_DEMAND_TWH[iso]
    rate = DEMAND_GROWTH_RATES[iso][growth_level]
    return base * (1 + rate) ** (year - 2023)


def estimate_new_gw_from_delta(delta_resources_twh, iso):
    """Estimate GW of new capacity from TWh delta using capacity factors."""
    gw = {}
    for res, twh in delta_resources_twh.items():
        if twh <= 0:
            continue
        tech = RESOURCE_TO_TECH.get(res, res)
        cf = RESOURCE_CAPACITY_FACTORS.get(res, {}).get(iso, 0.25)
        if cf > 0:
            gw[tech] = twh / (cf * 8.760)
        else:
            gw[tech] = 0
    return gw


# ═══════════════════════════════════════════════════════════════════════════════
# CORE MARKET SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_market_simulation(scenario_id, conditions, isos=None,
                           nuclear_retirement_threshold=None,
                           snapshot_mode=False,
                           sim_years=None,
                           _preloaded=None, _lmp_cache=None, _quiet=False,
                           weather_year=None):
    """Run purely profit-driven market simulation via LCOE merit-order deployment.

    No emission constraints, no mandated deployment, no DAC, no clean% targets.
    Deploy clean resources where profitable (LCOE < revenue), stop when not.
    Clean energy level is an OUTPUT that emerges from market economics.

    Args:
        scenario_id: Identifier string for this scenario.
        conditions: Dict with demand_growth, lcoe_level, fuel_level, etc.
        isos: List of ISOs to simulate (default: all 7).
        nuclear_retirement_threshold: $/MWh — if nuclear total revenue falls
            below this, nuclear retires and model re-dispatches. None = no retirement.
        snapshot_mode: Deprecated, ignored. All modes are trajectory-based.
        sim_years: Optional explicit list of years to simulate. Use build_sim_years()
            to generate from start/end/step.
        _preloaded: Pre-loaded data dict to avoid re-reading.
        _lmp_cache: Shared LMP cache across scenarios.
        _quiet: Suppress per-zone print output.
        weather_year: Optional year string ('2021'-'2025') for weather-year
            sensitivity. Uses historical demand/generation shapes from the
            specified year instead of the default (2025). This captures
            interannual variability in renewable generation and demand.

    Returns {iso: [year_result_dict, ...]}.
    """
    if isos is None:
        isos = list(ISOS)

    _log = (lambda *a, **kw: None) if _quiet else print

    _log(f"\n{'='*70}")
    _log(f"Market Simulation — Scenario {scenario_id}: {conditions['name']}")
    _log(f"{'='*70}")

    # Load data
    if _preloaded is not None:
        demand_data = _preloaded['demand_data']
        gen_profiles = _preloaded['gen_profiles']
        emission_rates = _preloaded['emission_rates']
        fossil_mix = _preloaded['fossil_mix']
        egrid_baselines = _preloaded['egrid_baselines']
    else:
        t0 = time.time()
        _log("Loading common data...")
        demand_data, gen_profiles, emission_rates, fossil_mix = load_common_data()
        _log(f"  Common data loaded in {time.time()-t0:.1f}s")
        egrid_baselines = load_egrid_baselines()

    cumulative_gw = dict(WRIGHT_CUMULATIVE_GW_2025)
    results = {iso: [] for iso in isos}

    # Per-ISO state
    iso_state = {}
    for iso in isos:
        existing_clean = sum(GRID_MIX_SHARES.get(iso, {}).values())
        baseline_rps_eligible = sum(
            v for k, v in GRID_MIX_SHARES.get(iso, {}).items()
            if k in REC_ELIGIBLE
        )
        iso_state[iso] = {
            'clean_pct': existing_clean,
            'rps_eligible_pct': baseline_rps_eligible,
            'rps_eligible_twh_floor': baseline_rps_eligible / 100.0 * REGIONAL_DEMAND_TWH[iso],
            'market_stopped': False,
            'gas_built_gw': 0,
            'nuclear_retired': False,
            'acp_bonus_queue_gw': 0,
            'cumulative_acp_million': 0,
        }

    if sim_years is not None:
        _sim_years = sim_years
    elif snapshot_mode:
        _sim_years = [2025]
    else:
        _sim_years = SIM_YEARS

    for year in _sim_years:
        _log(f"\n--- Year {year} ---")

        # 2023 baseline: inject actual eGRID data
        if year == 2023:
            for iso in isos:
                baseline_co2_tons = egrid_baselines.get(iso, 100_000_000)
                baseline_co2_mt = baseline_co2_tons / 1e6
                baseline_demand = REGIONAL_DEMAND_TWH[iso]
                baseline_clean = EGRID_2023_CLEAN_PCT.get(iso, 40.0)
                baseline_lmp = EGRID_2023_LMP.get(iso, 30.0)
                fossil_twh = (1 - baseline_clean / 100.0) * baseline_demand
                baseline_er = baseline_co2_tons / (fossil_twh * 1e6) if fossil_twh > 0 else 0.5

                year_result = {
                    'iso': iso,
                    'scenario': scenario_id,
                    'year': 2023,
                    'clean_pct': round(baseline_clean, 1),
                    'demand_twh': round(baseline_demand, 1),
                    'emissions_mt': round(baseline_co2_mt, 2),
                    'emission_rate_tco2_mwh': round(baseline_er, 4),
                    'cost_per_mwh': 0,
                    'revenue_per_mwh': 0,
                    'energy_rev_mwh': 0,
                    'capacity_rev_mwh': 0,
                    'rec_rev_mwh': 0,
                    'avg_lmp': round(baseline_lmp, 1),
                    'lmp_p90': round(baseline_lmp * 1.5, 1),
                    'gas_built_gw': 0,
                    'total_gas_gw': 0,
                    'market_stop': False,
                    'resource_mix_twh': {},
                    'cumulative_gw': dict(WRIGHT_CUMULATIVE_GW_2025),
                    'zones_deployed': [],
                    'generator_economics': {},
                    'nuclear_revenue': {},
                    'nuclear_retired': False,
                    'ccs_breakeven': {},
                    # RPS compliance tracking
                    'rps_mandated_pct': 0,
                    'rps_eligible_pct': round(iso_state[iso].get('rps_eligible_pct', 0), 1),
                    'rps_shortfall_pct': 0,
                    'acp_cost_million': 0,
                    'cumulative_acp_million': 0,
                }
                results[iso].append(year_result)
                _log(f"  {iso}: 2023 baseline — {baseline_co2_mt:.1f} Mt, "
                     f"{baseline_clean:.1f}% clean, LMP=${baseline_lmp:.0f}")
            continue

        for iso in isos:
            state = iso_state[iso]

            demand_twh = get_demand_at_year(iso, year, conditions['demand_growth'])
            demand_total_mwh = demand_twh * 1e6

            demand_norm, total_mwh_base = get_demand_profile(iso, demand_data, weather_year=weather_year)
            supply_profiles_iso = get_supply_profiles(iso, gen_profiles, weather_year=weather_year)

            growth_factor = demand_twh / REGIONAL_DEMAND_TWH[iso]
            demand_mw_profile = np.array(demand_norm, dtype=np.float64) * total_mwh_base * growth_factor

            current_pct = state['clean_pct']

            # Enforce TWh ratchet: installed renewable TWh doesn't shrink with demand growth
            rps_elig_floor_pct = state['rps_eligible_twh_floor'] / demand_twh * 100
            if state['rps_eligible_pct'] < rps_elig_floor_pct:
                state['rps_eligible_pct'] = rps_elig_floor_pct

            # Annual queue budget — from queue_cap_level or override
            queue_cap_level = conditions.get('queue_cap_level', 'Medium')
            queue_budget_gw = conditions.get('queue_cap_override_gw')
            if queue_budget_gw is None:
                queue_budget_gw = QUEUE_CAP_GW.get(queue_cap_level, {}).get(iso, 5)
            years_in_period = 7 if year == 2030 else 5
            queue_remaining_gw = queue_budget_gw * years_in_period

            # ACP recycling: prior-period ACP payments boost queue (capped at 20% of base)
            acp_bonus = min(state.get('acp_bonus_queue_gw', 0),
                           queue_budget_gw * years_in_period * 0.20)
            if acp_bonus > 0:
                queue_remaining_gw += acp_bonus
                _log(f"  {iso} ACP recycling: +{acp_bonus:.2f} GW bonus queue")
                state['acp_bonus_queue_gw'] = 0

            # --- LMP + GENERATOR ECONOMICS at current clean% ---
            carbon_price = conditions.get('carbon_price', 0)
            resource_pcts = {r: 0 for r in DEPLOYABLE_RESOURCES}
            # Estimate current resource pcts from grid mix shares
            for r, pct in GRID_MIX_SHARES.get(iso, {}).items():
                if r in resource_pcts:
                    resource_pcts[r] = pct

            _lmp_key = (iso, current_pct, conditions['fuel_level'],
                        conditions['demand_growth'], year, carbon_price)
            if _lmp_cache is not None and _lmp_key in _lmp_cache:
                hourly_lmp, avg_lmp, p90_lmp, gen_econ = _lmp_cache[_lmp_key]
            else:
                hourly_lmp, avg_lmp, p90_lmp, gen_econ = compute_lmp_at_threshold(
                    iso, current_pct, conditions['fuel_level'],
                    demand_norm, demand_mw_profile,
                    supply_profiles_iso, resource_pcts,
                    battery_pct=0, battery8_pct=0, ldes_pct=0, h2_pct=0,
                    carbon_price=carbon_price,
                    nox_price=conditions.get('nox_price', 0.0),
                    sox_price=conditions.get('sox_price', 0.0),
                    nox_limit=conditions.get('nox_limit'),
                    sox_limit=conditions.get('sox_limit'),
                    custom_fuel_prices=conditions.get('custom_fuel_prices'),
                    custom_co2_price=conditions.get('custom_co2_price'),
                    custom_heat_rates=conditions.get('custom_heat_rates'),
                    custom_vom=conditions.get('custom_vom'),
                )
                if _lmp_cache is not None:
                    _lmp_cache[_lmp_key] = (hourly_lmp, avg_lmp, p90_lmp, gen_econ)

            # --- LCOE MERIT-ORDER DEPLOYMENT ---
            # Deploy cheapest profitable clean resources until queue cap or no more profitable
            (new_clean_pct, deployed, zone_results, rev_breakdown,
             blended_cost, blended_revenue, remaining_gw) = compute_market_deployment(
                iso, year, demand_twh, current_pct,
                conditions, cumulative_gw, queue_remaining_gw,
                hourly_lmp, avg_lmp, p90_lmp,
                supply_profiles_iso, demand_total_mwh,
                gen_econ, state,
            )

            # Update state
            old_pct = current_pct
            current_pct = new_clean_pct
            state['clean_pct'] = current_pct
            state['market_stopped'] = (current_pct <= old_pct + 0.01)

            # Track RPS-eligible deployment
            for res, twh in deployed.items():
                if res in REC_ELIGIBLE:
                    rps_delta_pct = twh / demand_twh * 100
                    state['rps_eligible_pct'] += rps_delta_pct

            _log(f"  {iso} → {current_pct:.1f}% clean (was {old_pct:.1f}%): "
                 f"LMP avg=${avg_lmp:.1f}, deployed {sum(deployed.values()):.1f} TWh "
                 f"across {len(deployed)} resources")

            # Nuclear revenue at current threshold
            nuclear_rev = compute_nuclear_revenue(iso, current_pct, hourly_lmp, year, conditions=conditions)

            # --- NUCLEAR RETIREMENT CHECK ---
            offtake = NUCLEAR_OFFTAKE_CONTRACTS.get(iso)
            contract_protected = (
                offtake is not None and
                year <= offtake.get('contract_end_year', 0)
            )
            if (nuclear_retirement_threshold is not None and
                    not state['nuclear_retired'] and
                    not contract_protected and
                    nuclear_rev['total_mwh'] < nuclear_retirement_threshold):
                state['nuclear_retired'] = True
                _log(f"  {iso} NUCLEAR RETIRES — "
                     f"revenue ${nuclear_rev['total_mwh']:.1f}/MWh "
                     f"< threshold ${nuclear_retirement_threshold:.1f}/MWh")

            # --- RPS/CES FLOOR ENFORCEMENT ---
            # After profit-driven deployment, check if state RPS mandates are met.
            # CES ISOs (NYISO, NEISO, CAISO) count all clean energy.
            # Non-CES ISOs (PJM, MISO, SPP) count only REC_ELIGIBLE resources.
            rps_mandated_pct = 0
            acp_cost_million = 0
            rps_shortfall_pct = 0
            rps_floor = get_rps_floor(iso, year)
            if rps_floor > 0:
                if iso in CES_ISOS:
                    compliance_pct = current_pct
                else:
                    compliance_pct = state['rps_eligible_pct']

                if compliance_pct < rps_floor:
                    rps_gap = rps_floor - compliance_pct
                    _log(f"  {iso} RPS floor {rps_floor:.0f}% vs compliance "
                         f"{compliance_pct:.1f}% "
                         f"({'CES' if iso in CES_ISOS else 'RPS-eligible only'}): "
                         f"gap = {rps_gap:.1f}pp")

                    # Estimate GW needed using blended solar/wind CF
                    rps_twh_needed = rps_gap / 100.0 * demand_twh
                    solar_cf = RESOURCE_CAPACITY_FACTORS.get('solar', {}).get(iso, 0.20)
                    wind_cf = RESOURCE_CAPACITY_FACTORS.get('wind', {}).get(iso, 0.30)
                    blended_cf = (solar_cf + wind_cf) / 2.0
                    rps_gw_needed = rps_twh_needed / (blended_cf * 8.760) if blended_cf > 0 else 0

                    if rps_gw_needed > queue_remaining_gw and queue_remaining_gw > 0:
                        # Queue-constrained: deploy what's possible, ACP the rest
                        scale = queue_remaining_gw / rps_gw_needed
                        achievable_pct = rps_gap * scale
                        current_pct += achievable_pct
                        state['rps_eligible_pct'] += achievable_pct
                        rps_mandated_pct = achievable_pct
                        rps_shortfall_pct = rps_gap - achievable_pct
                        shortfall_twh = rps_shortfall_pct / 100.0 * demand_twh
                        acp_rate = ACP_RATES.get(iso, 0)
                        acp_cost_million = shortfall_twh * 1e3 * acp_rate / 1e6
                        queue_remaining_gw = 0
                        _log(f"    Queue-constrained: +{achievable_pct:.1f}pp deployed, "
                             f"{rps_shortfall_pct:.1f}pp shortfall → "
                             f"ACP ${acp_cost_million:.0f}M")
                    elif queue_remaining_gw <= 0:
                        # Queue exhausted — entire gap covered by ACP
                        rps_shortfall_pct = rps_gap
                        shortfall_twh = rps_shortfall_pct / 100.0 * demand_twh
                        acp_rate = ACP_RATES.get(iso, 0)
                        acp_cost_million = shortfall_twh * 1e3 * acp_rate / 1e6
                        _log(f"    Queue exhausted: {rps_shortfall_pct:.1f}pp shortfall → "
                             f"ACP ${acp_cost_million:.0f}M")
                    else:
                        # Queue has capacity — deploy fully to RPS floor
                        current_pct += rps_gap
                        state['rps_eligible_pct'] += rps_gap
                        rps_mandated_pct = rps_gap
                        queue_remaining_gw -= rps_gw_needed
                        _log(f"    RPS mandate: +{rps_gap:.1f}pp deployed to reach "
                             f"{rps_floor:.0f}% floor")

                    # ACP recycling: payments fund future renewable dev
                    if acp_cost_million > 0:
                        bonus_gw = (acp_cost_million * ACP_FUND_EFFICIENCY) / AVG_COST_PER_GW
                        state['acp_bonus_queue_gw'] += bonus_gw
                        state['cumulative_acp_million'] += acp_cost_million
                        _log(f"    ACP recycling: ${acp_cost_million:.0f}M → "
                             f"+{bonus_gw:.2f} GW future queue bonus")

                    state['clean_pct'] = current_pct

            # --- EMISSION ACCOUNTING ---
            gf = demand_twh / REGIONAL_DEMAND_TWH[iso]
            _, retirement_info = compute_fossil_retirement(
                iso, current_pct, emission_rates, fossil_mix, gf)
            # Use remaining fleet emission rate (not displaced rate) for actual emissions
            er = retirement_info.get('remaining_rate_tco2_mwh', 0)
            fossil_twh = (1 - current_pct / 100.0) * demand_twh
            emissions_mt = fossil_twh * 1e6 * er / 1e6

            # Per-fuel-type emissions breakdown (Mt CO2) from generator economics
            emissions_by_fuel = {}
            for utype, econ in gen_econ.items():
                cap_mw = econ.get('capacity_mw', 0)
                cf = econ.get('cf', 0)
                co2_rate = CO2_RATES.get(utype, 0.5)  # tons CO2/MWh
                gen_mwh = cap_mw * cf * H
                fuel_co2_mt = gen_mwh * co2_rate / 1e6
                emissions_by_fuel[utype] = round(fuel_co2_mt, 3)

            # Update TWh ratchet floor after all deployment
            current_rps_twh = state['rps_eligible_pct'] / 100.0 * demand_twh
            state['rps_eligible_twh_floor'] = max(state['rps_eligible_twh_floor'], current_rps_twh)

            # CCS breakeven at this ISO/fuel level
            ccs_breakeven = compute_ccs_retrofit_breakeven(iso, conditions['fuel_level'])

            # Resource mix in TWh — built from deployed resources + existing mix
            existing_mix_twh = {r: p / 100.0 * demand_twh
                                for r, p in GRID_MIX_SHARES.get(iso, {}).items()}
            resource_mix_twh = dict(existing_mix_twh)
            for res, twh in deployed.items():
                resource_mix_twh[res] = resource_mix_twh.get(res, 0) + twh

            year_result = {
                'iso': iso,
                'scenario': scenario_id,
                'year': year,
                'clean_pct': round(current_pct, 1),
                'demand_twh': round(demand_twh, 1),
                'emissions_mt': round(emissions_mt, 2),
                'emission_rate_tco2_mwh': round(er, 4),
                'cost_per_mwh': round(blended_cost, 2),
                'revenue_per_mwh': round(blended_revenue, 2),
                'energy_rev_mwh': rev_breakdown['energy_rev_mwh'],
                'capacity_rev_mwh': rev_breakdown['capacity_rev_mwh'],
                'rec_rev_mwh': rev_breakdown['rec_rev_mwh'],
                'avg_lmp': round(avg_lmp, 1),
                'lmp_p90': round(p90_lmp, 1),
                'gas_built_gw': round(state['gas_built_gw'], 2),
                'total_gas_gw': 0,
                'market_stop': state['market_stopped'],
                'resource_mix_twh': {k: round(v, 2) for k, v in resource_mix_twh.items()},
                'cumulative_gw': {k: round(v, 2) for k, v in cumulative_gw.items()},
                'zones_deployed': [z['resource'] for z in zone_results],
                'zone_details': zone_results,
                'generator_economics': gen_econ,
                'emissions_by_fuel': emissions_by_fuel,
                'nuclear_revenue': nuclear_rev,
                'nuclear_retired': state['nuclear_retired'],
                'ccs_breakeven': ccs_breakeven,
                # RPS compliance tracking
                'rps_mandated_pct': round(rps_mandated_pct, 1),
                'rps_eligible_pct': round(state['rps_eligible_pct'], 1),
                'rps_shortfall_pct': round(rps_shortfall_pct, 1),
                'acp_cost_million': round(acp_cost_million, 1),
                'cumulative_acp_million': round(state['cumulative_acp_million'], 1),
            }
            results[iso].append(year_result)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SWEEP RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_sweep(isos=None, nuclear_retirement_threshold=None,
                    snapshot_mode=False, weather_years=None):
    """Run full 270-scenario market sweep.

    Pre-loads data once, shares LMP cache across scenarios.

    Args:
        weather_years: Optional list of year strings (e.g., ['2021', '2023', '2025'])
            to run weather-year sensitivity. Each year runs the full sweep
            independently, then results are combined. If None, uses default
            year only (1× sweep = 270 scenarios). With 5 years, runs 5× sweep
            = 1,350 scenarios total.
    """
    t0 = time.time()
    print("Loading common data...")
    demand_data, gen_profiles, emission_rates, fossil_mix = load_common_data()
    print(f"  Common data loaded in {time.time()-t0:.1f}s")
    egrid_baselines = load_egrid_baselines()

    preloaded = {
        'demand_data': demand_data,
        'gen_profiles': gen_profiles,
        'emission_rates': emission_rates,
        'fossil_mix': fossil_mix,
        'egrid_baselines': egrid_baselines,
    }

    scenarios = build_market_scenarios()

    # Weather-year iteration: run each weather year as a separate sweep pass
    wy_list = weather_years or [None]  # None = default year (DATA_YEAR)
    total_runs = len(scenarios) * len(wy_list)
    print(f"\nRunning {len(scenarios)} market scenarios × {len(wy_list)} weather year(s) = {total_runs} total...")

    lmp_cache = {}
    all_results = {}
    run_count = 0

    for wy in wy_list:
        wy_label = f"_WY{wy}" if wy else ""
        if wy:
            print(f"\n--- Weather Year {wy} ---")
            lmp_cache.clear()  # Each weather year has different profiles

        for i, (scenario_id, conditions) in enumerate(scenarios):
            run_count += 1
            full_id = f"{scenario_id}{wy_label}"
            print(f"\n[{run_count}/{total_runs}] {full_id}")
            results = run_market_simulation(
                full_id, conditions, isos=isos,
                nuclear_retirement_threshold=nuclear_retirement_threshold,
                snapshot_mode=snapshot_mode,
                _preloaded=preloaded,
                _lmp_cache=lmp_cache,
                _quiet=True,
                weather_year=wy,
            )
            all_results[full_id] = results

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Sweep complete: {total_runs} scenarios in {elapsed:.1f}s")
    print(f"LMP cache entries: {len(lmp_cache)} unique threshold/fuel/demand combos")

    return all_results


def aggregate_sweep_percentiles(all_results):
    """Compute P10/P50/P90 uncertainty bands across all sweep scenarios.

    Groups year_results by (iso, year), computes percentiles across 270 scenarios
    for all scalar metrics and per-resource breakdowns. This is the primary
    uncertainty quantification output — it shows how sensitive each outcome is
    to parametric uncertainty (fuel prices, demand growth, grid conditions, etc.).

    Args:
        all_results: dict[scenario_id, dict[iso, list[year_result]]]

    Returns:
        dict[iso, dict[year, dict[metric, dict[p10/p50/p90, float]]]]
    """
    # Scalar metrics to aggregate
    SCALAR_METRICS = [
        'clean_pct', 'demand_twh', 'emissions_mt', 'emission_rate_tco2_mwh',
        'cost_per_mwh', 'revenue_per_mwh', 'energy_rev_mwh', 'capacity_rev_mwh',
        'rec_rev_mwh', 'avg_lmp', 'lmp_p90', 'gas_built_gw',
    ]
    # Boolean metrics: report % of scenarios where condition is True
    BOOL_METRICS = ['market_stop', 'nuclear_retired']

    # Collect values by (iso, year)
    from collections import defaultdict
    grouped = defaultdict(lambda: defaultdict(list))  # (iso, year) → [year_result, ...]

    for scenario_id, iso_results in all_results.items():
        for iso, year_results in iso_results.items():
            for yr in year_results:
                key = (iso, yr.get('year', 0))
                grouped[key]['_results'].append(yr)

    aggregates = {}
    for (iso, year), data in grouped.items():
        results_list = data['_results']
        n = len(results_list)
        if n < 3:
            continue  # Need at least 3 scenarios for meaningful percentiles

        if iso not in aggregates:
            aggregates[iso] = {}

        year_agg = {}

        # Scalar metrics
        for metric in SCALAR_METRICS:
            values = [r.get(metric, 0) or 0 for r in results_list]
            values = [v for v in values if isinstance(v, (int, float))]
            if values:
                arr = np.array(values, dtype=np.float64)
                year_agg[metric] = {
                    'p10': round(float(np.percentile(arr, 10)), 3),
                    'p50': round(float(np.percentile(arr, 50)), 3),
                    'p90': round(float(np.percentile(arr, 90)), 3),
                    'mean': round(float(np.mean(arr)), 3),
                    'std': round(float(np.std(arr)), 3),
                    'n': len(values),
                }

        # Boolean metrics: fraction of scenarios where True
        for metric in BOOL_METRICS:
            values = [1 if r.get(metric) else 0 for r in results_list]
            year_agg[metric] = {
                'pct_true': round(100 * sum(values) / n, 1),
                'n': n,
            }

        # Per-resource mix breakdown
        resource_agg = {}
        all_resources = set()
        for r in results_list:
            rmix = r.get('resource_mix_twh', {})
            if isinstance(rmix, dict):
                all_resources.update(rmix.keys())

        for resource in sorted(all_resources):
            values = []
            for r in results_list:
                rmix = r.get('resource_mix_twh', {})
                if isinstance(rmix, dict):
                    values.append(rmix.get(resource, 0) or 0)
            if values:
                arr = np.array(values, dtype=np.float64)
                resource_agg[resource] = {
                    'p10': round(float(np.percentile(arr, 10)), 3),
                    'p50': round(float(np.percentile(arr, 50)), 3),
                    'p90': round(float(np.percentile(arr, 90)), 3),
                }

        if resource_agg:
            year_agg['resource_mix_twh'] = resource_agg

        # Nuclear revenue aggregation
        nuc_rev_values = []
        for r in results_list:
            nr = r.get('nuclear_revenue', {})
            if isinstance(nr, dict) and 'total_mwh' in nr:
                nuc_rev_values.append(nr['total_mwh'])
        if nuc_rev_values:
            arr = np.array(nuc_rev_values, dtype=np.float64)
            year_agg['nuclear_revenue_mwh'] = {
                'p10': round(float(np.percentile(arr, 10)), 3),
                'p50': round(float(np.percentile(arr, 50)), 3),
                'p90': round(float(np.percentile(arr, 90)), 3),
            }

        # Zone deployment count
        zone_counts = [len(r.get('zones_deployed', [])) for r in results_list]
        if zone_counts:
            arr = np.array(zone_counts, dtype=np.float64)
            year_agg['zones_deployed_count'] = {
                'p10': round(float(np.percentile(arr, 10)), 1),
                'p50': round(float(np.percentile(arr, 50)), 1),
                'p90': round(float(np.percentile(arr, 90)), 1),
            }

        aggregates[iso][str(year)] = year_agg

    return aggregates


def save_results(all_results, output_dir=None):
    """Save results as JSON, including P10/P50/P90 uncertainty bands."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'market_simulation_results.json')
    # Convert to serializable format
    serializable = {}
    for scenario_id, iso_results in all_results.items():
        serializable[scenario_id] = {}
        for iso, year_results in iso_results.items():
            serializable[scenario_id][iso] = year_results

    # Compute P10/P50/P90 uncertainty bands across all scenarios
    if len(all_results) > 1:
        print("Computing P10/P50/P90 uncertainty bands across sweep scenarios...")
        aggregates = aggregate_sweep_percentiles(all_results)
        serializable['_aggregates'] = aggregates
        n_isos = len(aggregates)
        n_years = sum(len(yrs) for yrs in aggregates.values())
        print(f"  Aggregated {n_isos} ISOs × {n_years} year-groups "
              f"from {len(all_results)} scenarios")

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)

    print(f"Results saved to {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Market Simulator — Generator Economics Under Market Conditions')
    parser.add_argument('--isos', nargs='+', default=None,
                        help='ISOs to simulate (default: all 7)')
    parser.add_argument('--single', action='store_true',
                        help='Run single scenario instead of full 270 sweep')
    parser.add_argument('--snapshot', action='store_true',
                        help='Single-year snapshot mode (no year progression)')
    parser.add_argument('--carbon-price', type=float, default=0,
                        help='Carbon price $/ton (default: 0)')
    parser.add_argument('--fuel-level', default='Medium',
                        choices=['Low', 'Medium', 'High'],
                        help='Fuel price level')
    parser.add_argument('--lcoe-level', default='Medium',
                        choices=['Low', 'Medium', 'High'],
                        help='Clean resource LCOE level')
    parser.add_argument('--nuclear-retirement', type=float, default=None,
                        help='Nuclear retirement threshold $/MWh (default: None = no retirement)')
    parser.add_argument('--weather-year', default=None,
                        help='Weather year for demand/gen profiles (2021-2025). '
                             'For sweep: comma-separated (e.g., "2021,2023,2025") '
                             'or "all" for all 5 years.')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for results')

    args = parser.parse_args()

    # Parse weather-year argument
    weather_years = None
    if args.weather_year:
        if args.weather_year == 'all':
            from dispatch_utils import AVAILABLE_WEATHER_YEARS
            weather_years = AVAILABLE_WEATHER_YEARS
        else:
            weather_years = [y.strip() for y in args.weather_year.split(',')]

    if args.single:
        overrides = {
            'carbon_price': args.carbon_price,
            'fuel_level': args.fuel_level,
            'lcoe_level': args.lcoe_level,
        }
        scenario_id, conditions = build_single_scenario(overrides)
        wy = weather_years[0] if weather_years else None
        results = run_market_simulation(
            scenario_id, conditions, isos=args.isos,
            nuclear_retirement_threshold=args.nuclear_retirement,
            snapshot_mode=args.snapshot,
            weather_year=wy,
        )
        all_results = {scenario_id: results}
    else:
        all_results = run_full_sweep(
            isos=args.isos,
            nuclear_retirement_threshold=args.nuclear_retirement,
            snapshot_mode=args.snapshot,
            weather_years=weather_years,
        )

    save_results(all_results, args.output_dir)


if __name__ == '__main__':
    main()
