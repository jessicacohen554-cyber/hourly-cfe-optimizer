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
  python market_simulation.py                           # Full 1,215-scenario sweep
  python market_simulation.py --isos CAISO ERCOT        # Subset ISOs
  python market_simulation.py --single                  # Single scenario (Medium defaults)
  python market_simulation.py --snapshot                # Single-year snapshot mode
  python market_simulation.py --carbon-price 50         # Override carbon price
"""

import argparse
import datetime
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product as cartesian

# Numba JIT — speedup on unit commitment state machine loop
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
MODULE_ROOT = os.path.dirname(SCRIPT_DIR)
# Add scripts dir to path for shared utilities
sys.path.insert(0, SCRIPT_DIR)

from pipeline_config import (
    PIPELINE_VERSION,
    ISOS, REGIONAL_DEMAND_TWH, DEMAND_GROWTH_RATES,
    GRID_MIX_SHARES, WHOLESALE_PRICES, THRESHOLDS,
    CAPACITY_MARKET_PRICES, CAPACITY_DEGRADATION_ALPHA, CAPACITY_DEGRADATION_PARAMS,
    CAPACITY_SCARCITY_PARAMS, compute_capacity_price,
    PEAK_CAPACITY_CREDITS, RESOURCE_CAPACITY_FACTORS,
    LCOE_TABLES, TX_TABLES, get_tx,
    NUCLEAR_NEWBUILD_LCOE, CCS_LCOE_45Q_ON, CCS_LCOE_45Q_OFF, GEOTHERMAL_LCOE,
    FOAK_NUCLEAR_NEWBUILD, FOAK_CCS_45Q_ON, FOAK_CCS_45Q_OFF, FOAK_GEOTHERMAL,
    FOAK_LDES, FOAK_H2, FOAK_OFFSHORE_WIND,
    NOAK_BATTERY, NOAK_BATTERY8,
    EXISTING_GAS_FOM_KW_YR,
    NEW_GAS_CCGT_LCOE, NEW_GAS_CT_LCOE, NEW_COAL_LCOE,
    NEW_BUILD_CAPEX_KW_YR, NEW_BUILD_HEAT_RATES, NEW_BUILD_VOM,
    NEW_BUILD_CO2_RATES, NEW_BUILD_MIN_CF, NEW_BUILD_MAX_GW_YR,
    PEAK_DEMAND_MW, EXISTING_GAS_CAPACITY_MW, RESOURCE_ADEQUACY_MARGIN,
    compute_storage_revenue_credit,
    STORAGE_ANCILLARY_PRODUCT, ANCILLARY_SERVICE_RATES, ANCILLARY_HOURS,
    REVENUE_STACKING_FACTOR, STORAGE_MAX, H2_MIN_THRESHOLD,
    OFFSHORE_ISOS, CCS_CAP_TWH, GEOTHERMAL_CAP_TWH,
    H, NUCLEAR_OFFTAKE_CONTRACTS, EXISTING_NUCLEAR_GW,
    get_rps_floor,
    FIRM_IMPORT_MW,
    CANNIBALIZATION_ENABLED,
    SYNTHETIC_DATA_MODE,
    SCARCITY_MODE,
    VRE_PRIMARY_ZONE,
    ENDOGENOUS_LEARNING,
    CORRELATED_SCENARIOS,
)
from dispatch_utils import (
    load_common_data, get_demand_profile, get_supply_profiles,
    reconstruct_hourly_dispatch, compute_fossil_retirement,
    COAL_CAP_TWH, OIL_CAP_TWH,
    RESOURCE_TYPES, H,
)
from lmp_engine import (
    build_merit_order_stack, build_plant_level_merit_order,
    compute_hourly_lmp_vectorized, PriceModel,
    HEAT_RATES, VOM, CO2_RATES, FUEL_PRICES,
    INSTALLED_FOSSIL_MW, FOSSIL_CAPACITY_SHARES,
)
from procurement_utils import get_rps_target_at_year, PPA_PREMIUMS

# Add backend dir to path for model imports
sys.path.insert(0, os.path.join(MODULE_ROOT, 'backend'))
from models import ProvenanceMetadata

OUTPUT_DIR = os.path.join(MODULE_ROOT, 'data', 'results')

logger = logging.getLogger(__name__)

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

# Per-technology queue caps (GW/yr per ISO per tech)
# Sources: LBNL "Queued Up 2024" (Rand et al., 2024) — queue completion rate by technology
# Solar completes at ~8 GW/yr nationally, wind ~5 GW/yr, nuclear/CCS <0.5 GW/yr.
# Per-tech caps sum to approximately the uniform QUEUE_CAP_GW for backward compatibility.
TECH_QUEUE_CAP_GW = {
    'Medium': {
        'CAISO':  {'solar': 2.5, 'wind': 0.8, 'offshore_wind': 0.3, 'clean_firm': 0.2, 'ccs_ccgt': 0.4, 'geothermal': 0.3},
        'ERCOT':  {'solar': 4.0, 'wind': 2.5, 'offshore_wind': 0.2, 'clean_firm': 0.2, 'ccs_ccgt': 0.5, 'geothermal': 0.0},
        'PJM':    {'solar': 2.5, 'wind': 1.2, 'offshore_wind': 0.5, 'clean_firm': 0.3, 'ccs_ccgt': 0.5, 'geothermal': 0.0},
        'NYISO':  {'solar': 1.0, 'wind': 0.6, 'offshore_wind': 0.5, 'clean_firm': 0.2, 'ccs_ccgt': 0.2, 'geothermal': 0.0},
        'NEISO':  {'solar': 0.8, 'wind': 0.8, 'offshore_wind': 0.7, 'clean_firm': 0.2, 'ccs_ccgt': 0.3, 'geothermal': 0.0},
        'MISO':   {'solar': 2.0, 'wind': 1.5, 'offshore_wind': 0.0, 'clean_firm': 0.2, 'ccs_ccgt': 0.5, 'geothermal': 0.0},
        'SPP':    {'solar': 1.8, 'wind': 1.5, 'offshore_wind': 0.0, 'clean_firm': 0.1, 'ccs_ccgt': 0.3, 'geothermal': 0.0},
    },
    'Low': {  # ~50% of Medium (status quo permitting)
        'CAISO':  {'solar': 1.3, 'wind': 0.4, 'offshore_wind': 0.15, 'clean_firm': 0.1, 'ccs_ccgt': 0.2, 'geothermal': 0.15},
        'ERCOT':  {'solar': 2.0, 'wind': 1.3, 'offshore_wind': 0.1, 'clean_firm': 0.1, 'ccs_ccgt': 0.25, 'geothermal': 0.0},
        'PJM':    {'solar': 1.3, 'wind': 0.6, 'offshore_wind': 0.25, 'clean_firm': 0.15, 'ccs_ccgt': 0.25, 'geothermal': 0.0},
        'NYISO':  {'solar': 0.5, 'wind': 0.3, 'offshore_wind': 0.25, 'clean_firm': 0.1, 'ccs_ccgt': 0.1, 'geothermal': 0.0},
        'NEISO':  {'solar': 0.4, 'wind': 0.4, 'offshore_wind': 0.35, 'clean_firm': 0.1, 'ccs_ccgt': 0.15, 'geothermal': 0.0},
        'MISO':   {'solar': 1.0, 'wind': 0.8, 'offshore_wind': 0.0, 'clean_firm': 0.1, 'ccs_ccgt': 0.25, 'geothermal': 0.0},
        'SPP':    {'solar': 0.9, 'wind': 0.8, 'offshore_wind': 0.0, 'clean_firm': 0.05, 'ccs_ccgt': 0.15, 'geothermal': 0.0},
    },
    'High': {  # ~133% of Medium (FERC Order 2023 reforms)
        'CAISO':  {'solar': 3.3, 'wind': 1.1, 'offshore_wind': 0.4, 'clean_firm': 0.3, 'ccs_ccgt': 0.5, 'geothermal': 0.4},
        'ERCOT':  {'solar': 5.3, 'wind': 3.3, 'offshore_wind': 0.3, 'clean_firm': 0.3, 'ccs_ccgt': 0.7, 'geothermal': 0.0},
        'PJM':    {'solar': 3.3, 'wind': 1.6, 'offshore_wind': 0.7, 'clean_firm': 0.4, 'ccs_ccgt': 0.7, 'geothermal': 0.0},
        'NYISO':  {'solar': 1.3, 'wind': 0.8, 'offshore_wind': 0.7, 'clean_firm': 0.3, 'ccs_ccgt': 0.3, 'geothermal': 0.0},
        'NEISO':  {'solar': 1.1, 'wind': 1.1, 'offshore_wind': 0.9, 'clean_firm': 0.3, 'ccs_ccgt': 0.4, 'geothermal': 0.0},
        'MISO':   {'solar': 2.7, 'wind': 2.0, 'offshore_wind': 0.0, 'clean_firm': 0.3, 'ccs_ccgt': 0.7, 'geothermal': 0.0},
        'SPP':    {'solar': 2.4, 'wind': 2.0, 'offshore_wind': 0.0, 'clean_firm': 0.15, 'ccs_ccgt': 0.4, 'geothermal': 0.0},
    },
}
TECH_DIFFERENTIATED_QUEUE = True   # Set False to use legacy uniform QUEUE_CAP_GW
QUEUE_FLEX_FRACTION = 0.20         # 20% of total cap available as flex pool across techs

# Validate per-tech caps sum to within 50% of the uniform cap (catch config errors).
# Tech caps are intentionally lower than uniform — LBNL completion rates reflect real
# bottlenecks. The flex pool (QUEUE_FLEX_FRACTION) compensates for the difference.
for _level in TECH_QUEUE_CAP_GW:
    for _iso in TECH_QUEUE_CAP_GW[_level]:
        _tech_sum = sum(TECH_QUEUE_CAP_GW[_level][_iso].values())
        _uniform = QUEUE_CAP_GW.get(_level, {}).get(_iso, 0)
        if _uniform > 0 and abs(_tech_sum - _uniform) / _uniform > 0.50:
            import warnings as _w
            _w.warn(f"TECH_QUEUE_CAP_GW[{_level}][{_iso}] sums to {_tech_sum:.1f} GW "
                    f"but uniform QUEUE_CAP_GW is {_uniform:.1f} GW (>50% difference)")

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


def _split_stack_by_zone(stack, zone_config):
    """Split a system-level merit-order stack into per-zone stacks.

    When real plant-to-zone mapping isn't available (e.g., synthetic stack
    in sweep mode), splits each unit's capacity proportionally by zone
    demand share. This is an approximation — real zonal stacks from
    FleetModel.build_zonal_merit_order_stacks() are more accurate.

    Args:
        stack: list of (unit_type, cap_mw, mc) tuples
        zone_config: dict with 'zones' and 'demand_share'

    Returns:
        dict {zone_name: [(unit_type, cap_mw, mc), ...]}
    """
    zone_stacks = {}
    for zname in zone_config['zones']:
        share = zone_config['demand_share'][zname]
        zone_stacks[zname] = [
            (utype, cap * share, mc)
            for utype, cap, mc in stack
            if cap * share > 0.1  # Skip negligible slices
        ]
    return zone_stacks


def compute_lmp_at_threshold(iso, clean_pct, fuel_level, demand_norm,
                              demand_mw_profile, supply_profiles, resource_pcts,
                              battery_pct=0, battery8_pct=0, ldes_pct=0, h2_pct=0,
                              carbon_price=0, nox_price=0.0, sox_price=0.0,
                              nox_limit=None, sox_limit=None,
                              custom_fuel_prices=None, custom_co2_price=None,
                              custom_heat_rates=None, custom_vom=None,
                              interchange_norm=None, firm_import_mw=0,
                              dr_level='Off',
                              demand_growth_factor=1.0,
                              new_fossil_builds=None):
    """Compute 8760-hour LMP at a given clean percentage.

    Returns (hourly_lmp_array, avg_lmp, lmp_p90, generator_economics,
             dr_metrics, zonal_stats_or_None).
    generator_economics is a dict of per-unit-type dispatch metrics.
    zonal_stats is a dict with per-zone LMP stats and congestion data
    (None if copper-plate mode was used).
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
        firm_import_mw=firm_import_mw,
        demand_growth_factor=demand_growth_factor,
        new_fossil_builds=new_fossil_builds,
    )

    # resource_pcts already represents actual % of demand each resource serves
    # (baseline + cumulative deployed), so procurement_pct=100 avoids
    # double-attenuation. Previously procurement_pct=clean_pct caused
    # supply = (clean_pct/100) × (resource_pcts/100) which under-counted
    # clean energy and over-dispatched fossil.
    dispatch = reconstruct_hourly_dispatch(
        demand_norm, supply_profiles, resource_pcts,
        procurement_pct=100,
        battery_dispatch_pct=battery_pct,
        battery8_dispatch_pct=battery8_pct,
        ldes_dispatch_pct=ldes_pct,
        h2_dispatch_pct=h2_pct,
        interchange_norm=interchange_norm,
    )

    price_model = PriceModel(iso, fuel_level)
    vre_pen = clean_pct / 100.0 if clean_pct is not None else None

    # Adjust demand profile for interchange (net imports reduce effective demand)
    # Applied before LMP computation so both zonal and copper-plate paths see
    # the interchange-adjusted demand signal.
    effective_demand_mw = demand_mw_profile.copy()
    if interchange_norm is not None:
        total_annual_mwh = demand_mw_profile.sum()
        interchange_mw = np.asarray(interchange_norm[:H], dtype=np.float64) * total_annual_mwh
        # Cap at firm import MW
        if firm_import_mw > 0:
            interchange_mw = np.clip(interchange_mw, -firm_import_mw, firm_import_mw)
        effective_demand_mw = np.maximum(0.0, demand_mw_profile - interchange_mw)

    # Try zonal LMP if zone config available
    zonal_lmp_matrix = None
    zonal_zone_names = None
    zonal_stats = None
    try:
        from pipeline_config import ZONE_CONFIG
        if iso in ZONE_CONFIG:
            from zonal_lmp import compute_zonal_lmp_hourly
            zone_config = ZONE_CONFIG[iso]
            # Split synthetic stack by zone demand share (proportional approximation)
            zone_stacks = _split_stack_by_zone(stack, zone_config)
            # BUG FIX: Pass residual demand (after clean dispatch) to zonal LP,
            # not full demand. The zone stacks contain only fossil units, so the
            # LP demand signal must be the fossil residual — otherwise hours where
            # total demand > fossil capacity are infeasible ($500 scarcity cap)
            # even though clean energy covers 40-75% of load.
            total_annual_mwh = effective_demand_mw.sum()
            residual_demand_mw = np.asarray(
                dispatch['residual_demand'], dtype=np.float64) * total_annual_mwh
            # Floor at zero — negative residual = clean surplus (no fossil needed)
            residual_demand_mw = np.maximum(residual_demand_mw, 0.0)
            zonal_lmp_matrix, system_lmp, _, zonal_stats = compute_zonal_lmp_hourly(
                iso=iso, zone_config=zone_config, zone_stacks=zone_stacks,
                demand_mw_profile=residual_demand_mw,
                price_model=price_model, vre_penetration=vre_pen,
                full_demand_mw_profile=effective_demand_mw,
            )
            zonal_zone_names = zone_config['zones']
            hourly_lmp = system_lmp
            unit_idx = np.full(len(hourly_lmp), -1, dtype=np.int8)
    except Exception:
        # Fall back to copper-plate
        pass

    if zonal_lmp_matrix is None:
        hourly_lmp, unit_idx, _dr_unused = compute_hourly_lmp_vectorized(
            dispatch, effective_demand_mw, stack, price_model, iso=iso,
            vre_penetration=vre_pen,
        )

    # --- DEMAND RESPONSE POST-PROCESSING ---
    # Applied after LMP is computed (whether zonal or copper-plate) so DR
    # responds to the fully-formed price signal regardless of LMP method.
    from pipeline_config import DEMAND_RESPONSE, DR_LEVELS, SCARCITY_MODE, ORDC_PARAMS
    dr_curtailed_mw = np.zeros(H, dtype=np.float64)
    dr_activation_mode = 'off'
    if dr_level != 'Off' and iso in DEMAND_RESPONSE:
        dr_params = DEMAND_RESPONSE[iso]
        dr_lvl = DR_LEVELS.get(dr_level, DR_LEVELS['Off'])
        effective_participation = dr_params['participation'] * dr_lvl['participation_mult']
        fixed_trigger = dr_params['trigger_price'] * dr_lvl['trigger_mult']

        # Dynamic DR-ORDC trigger: when ORDC active and dr_ordc_link enabled,
        # trigger = max(fixed_trigger, VOLL * 0.05) — DR activates when ORDC
        # adder exceeds 5% of VOLL (genuine reserve stress).
        use_ordc_link = dr_params.get('dr_ordc_link', True) and SCARCITY_MODE == 'ordc'
        if use_ordc_link and iso in ORDC_PARAMS:
            ordc_dynamic_trigger = ORDC_PARAMS[iso]['voll'] * 0.05
            effective_trigger = max(fixed_trigger, ordc_dynamic_trigger)
            dr_activation_mode = 'ordc_dynamic'
        else:
            effective_trigger = fixed_trigger
            dr_activation_mode = 'fixed'

        max_dr_mw = dr_params['max_dr_gw'] * 1000 * effective_participation

        if max_dr_mw > 0:
            dr_mask = hourly_lmp > effective_trigger
            if dr_mask.any():
                # Cap at 15% of hourly demand
                dr_potential = np.minimum(max_dr_mw, demand_mw_profile[dr_mask] * 0.15)
                # Linear ramp: 0% at trigger, 100% at 2× trigger
                price_ratio = np.clip(
                    (hourly_lmp[dr_mask] - effective_trigger) / effective_trigger, 0, 1)
                dr_curtailed_mw[dr_mask] = dr_potential * price_ratio

                # Dampen LMP for DR-active hours: reduced demand shifts marginal unit
                # Vectorized: supply elasticity (~3× leverage), capped at 50% reduction
                dr_active_mask = dr_mask & (demand_mw_profile > 0) & (dr_curtailed_mw > 0)
                if dr_active_mask.any():
                    demand_reduction_pct = dr_curtailed_mw[dr_active_mask] / demand_mw_profile[dr_active_mask]
                    price_reduction = np.minimum(demand_reduction_pct * 3.0, 0.5)
                    hourly_lmp[dr_active_mask] *= (1.0 - price_reduction)
                    # Floor at 95% of trigger price
                    hourly_lmp[dr_active_mask] = np.maximum(
                        hourly_lmp[dr_active_mask], effective_trigger * 0.95)

    # --- CURTAILMENT RATE (R10: VRE economics feedback) ---
    # Compute system-level curtailment rate = curtailed_mwh / total_vre_mwh.
    # Used downstream to penalize marginal VRE LCOE as curtailment rises,
    # creating a natural saturation point that prevents unrealistic overdeployment.
    curtailment_rate = 0.0
    curtailed_arr = dispatch.get('curtailed')
    if curtailed_arr is not None:
        curtailed_total = float(np.sum(curtailed_arr))
        # VRE = solar + wind + offshore_wind portion of supply_total
        vre_resources = ('solar', 'wind', 'offshore_wind')
        total_vre = 0.0
        for vr in vre_resources:
            pct = resource_pcts.get(vr, 0)
            if pct > 0 and vr in supply_profiles:
                profile = np.array(supply_profiles[vr][:H], dtype=np.float64)
                total_vre += float(np.sum(profile * (pct / 100.0)))
        if total_vre > 0:
            curtailment_rate = min(curtailed_total / total_vre, 1.0)

    avg_lmp = float(np.mean(hourly_lmp))
    p90_lmp = float(np.percentile(hourly_lmp, 90))

    # --- GENERATOR-LEVEL ECONOMICS ---
    gen_econ = compute_generator_economics(
        stack, hourly_lmp, unit_idx, dispatch, demand_mw_profile,
        total_fossil_mw, iso, clean_pct, fuel_level, carbon_price,
    )

    # --- DR METRICS ---
    dr_metrics = {}
    if dr_level != 'Off':
        dr_active = dr_curtailed_mw > 0
        dr_metrics = {
            'dr_curtailed_gwh': round(float(dr_curtailed_mw.sum()) / 1e3, 1),
            'dr_peak_gw': round(float(dr_curtailed_mw.max()) / 1e3, 2),
            'dr_hours': int(dr_active.sum()),
            'dr_avg_price': round(float(hourly_lmp[dr_active].mean()), 1) if dr_active.any() else 0,
            'dr_activation_mode': dr_activation_mode,
            'dr_effective_trigger': round(float(effective_trigger), 1),
        }

    # --- ORDC SCARCITY HOURS FRACTION ---
    # Fraction of hours where ORDC adder > $50/MWh — used to floor
    # VRE cannibalization depression (scarcity hours keep prices high
    # even at high VRE penetration).
    scarcity_hours_fraction = 0.0
    if SCARCITY_MODE == 'ordc':
        # Reconstruct residual MW from dispatch to compute reserves
        total_annual_mwh = demand_mw_profile.sum()
        residual_norm = dispatch['residual_demand']
        _residual_mw = residual_norm * total_annual_mwh
        _reserves_mw = np.maximum(total_fossil_mw - _residual_mw, 0.0)
        _ordc_adder = price_model.compute_ordc_adder(_reserves_mw)
        scarcity_hours_fraction = float(np.sum(_ordc_adder > 50.0)) / H

    return hourly_lmp, avg_lmp, p90_lmp, gen_econ, dr_metrics, zonal_stats, scarcity_hours_fraction, zonal_lmp_matrix, zonal_zone_names, curtailment_rate


@njit(cache=True)
def _uc_loop(dispatched_i8, min_up, min_down):
    """Numba-accelerated unit commitment state machine.

    Args:
        dispatched_i8: int8 array (H,) — 1 where merit-order says unit should run
        min_up: minimum hours a unit must stay on after starting
        min_down: minimum hours a unit must stay off after shutting down

    Returns:
        committed: int8 array (H,) — UC-adjusted commitment (1=on, 0=off)
        n_starts: number of start events
    """
    H_len = dispatched_i8.shape[0]
    committed = np.zeros(H_len, dtype=np.int8)
    n_starts = 0
    hours_in_state = 0
    is_on = 0  # 0=off, 1=on

    for h in range(H_len):
        want_on = dispatched_i8[h]

        if is_on == 1:
            hours_in_state += 1
            if want_on == 0 and hours_in_state >= min_up:
                is_on = 0
                hours_in_state = 0
                committed[h] = 0
            else:
                committed[h] = 1
        else:
            hours_in_state += 1
            if want_on == 1 and hours_in_state >= min_down:
                is_on = 1
                hours_in_state = 0
                n_starts += 1
                committed[h] = 1
            else:
                committed[h] = 0

    return committed, n_starts


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
    min_up = uc_params.get('min_up_hrs', 1)
    min_down = uc_params.get('min_down_hrs', 1)
    min_gen = uc_params.get('min_gen_pct', 0.0) * cap_mw

    # Run state machine via Numba kernel (20-50x faster than Python loop)
    dispatched_i8 = np.asarray(dispatched, dtype=np.int8)
    committed_i8, n_starts = _uc_loop(dispatched_i8, min_up, min_down)
    committed = committed_i8.astype(bool)

    # Apply minimum generation floor and cap (already vectorized)
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
                                   year=2025, zonal_lmp=None, plant_zones=None,
                                   zone_names=None):
    """Compute per-plant dispatch economics using plant-level merit order.

    Each plant's position in the merit-order stack determines when it dispatches.
    Returns list of dicts with full per-plant economics.

    If zonal_lmp is provided (n_zones × H matrix), each plant uses its
    zone-specific LMP for revenue instead of system-average hourly_lmp.
    plant_zones is a list of zone name strings parallel to plant_stack.
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

        # Revenue — use zonal LMP if available for this plant
        plant_lmp = hourly_lmp  # default: system-average
        if zonal_lmp is not None and plant_zones is not None and zone_names is not None:
            pzone = plant_zones[i] if i < len(plant_zones) else None
            if pzone and pzone in zone_names:
                z_idx = zone_names.index(pzone)
                plant_lmp = zonal_lmp[z_idx]
        if total_mwh > 0:
            avg_rev = float(np.sum(plant_lmp * mw_dispatched)) / total_mwh
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
            'heat_rate_source': plant.get('heat_rate_source', 'default'),
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
            'zone': plant_zones[i] if plant_zones and i < len(plant_zones) else None,
        })

    return results


def apply_economic_retirement(gen_econ, iso, year, state, _log=print,
                              plant_economics=None, demand_twh=None):
    """Retire fossil capacity using plant-level economics (G1).

    When plant_economics is provided (from compute_plant_level_economics()),
    retires individual plants sorted by margin (worst-first). Each plant with
    margin < -$5/MWh is retired individually, tracked by plant ID for inter-year
    persistence. A zonal reliability floor (15% reserve margin on peak demand)
    prevents over-retirement.

    Nuclear plants are evaluated individually using NUCLEAR_OFFTAKE_CONTRACTS:
    plants with below-market contracts face higher stranding risk than those
    with market-rate or regulated-rate revenue.

    Falls back to legacy fleet-fraction retirement if plant_economics is None.

    Returns:
        adjusted_gen_econ: dict with retired capacity removed
        retired_capacity: dict of {unit_type: retired_mw}
        total_retired_mw: float
        plant_retirement_list: list of retired plant dicts
    """
    if not gen_econ:
        return gen_econ, {}, 0.0, []

    # --- PLANT-LEVEL PATH (G1) ---
    if plant_economics is not None:
        return _apply_plant_level_retirement(
            gen_econ, iso, year, state, plant_economics, demand_twh, _log)

    # --- LEGACY FLEET-FRACTION PATH (fallback) ---
    adjusted, retired_capacity, total_retired_mw = _apply_fleet_fraction_retirement(
        gen_econ, iso, year, state, _log)
    return adjusted, retired_capacity, total_retired_mw, []


def _apply_plant_level_retirement(gen_econ, iso, year, state, plant_economics,
                                  demand_twh, _log=print):
    """Plant-level retirement engine: retire individual plants by margin.

    1. Sorts all plants by margin (worst first)
    2. Retires plants with margin < -$5/MWh
    3. Tracks retired plant IDs in state['retired_plants'] for persistence
    4. Enforces 15% zonal reserve margin floor on peak demand
    5. Evaluates nuclear plants individually using contract economics

    Returns: (adjusted_gen_econ, retired_by_type, total_retired_mw, plant_retirement_list)
    """
    RETIREMENT_THRESHOLD = -5.0  # $/MWh — matches existing stranded classification
    RESERVE_MARGIN_FLOOR = 0.15  # 15% minimum reserve margin per zone

    # Already-retired plant IDs from prior years
    prior_retired = set(state.get('retired_plants', []))

    # Filter out already-retired plants from economics
    active_plants = [p for p in plant_economics
                     if p.get('plant_id') not in prior_retired
                     and p.get('generator_id') not in prior_retired]

    # Sort by margin — worst (most negative) first
    active_plants.sort(key=lambda p: p.get('profit_per_mwh', 0))

    # Compute zonal peak demand for reliability floor
    # Use demand_twh to derive peak; _PEAK_TO_AVG_RATIO = 1.5
    peak_to_avg = _PEAK_TO_AVG_RATIO
    if demand_twh and demand_twh > 0:
        avg_demand_mw = demand_twh * 1e6 / 8760
        peak_demand_mw = avg_demand_mw * peak_to_avg
    else:
        # Fallback: use pipeline_config PEAK_DEMAND_MW if available
        peak_demand_mw = PEAK_DEMAND_MW.get(iso, 50000)

    # Total available capacity (all active plants + clean capacity)
    total_fossil_mw = sum(p.get('capacity_mw', 0) for p in active_plants)
    clean_gw = sum(state.get('cumulative_gw', {}).values()) if 'cumulative_gw' in state else 0
    clean_mw = clean_gw * 1000
    total_supply_mw = total_fossil_mw + clean_mw

    # Minimum fossil capacity to maintain reserve margin
    min_total_supply = peak_demand_mw * (1 + RESERVE_MARGIN_FLOOR)
    max_retirable_mw = max(0, total_supply_mw - min_total_supply)

    # --- Nuclear plant-level evaluation ---
    # Build set of nuclear plant IDs with their contract status
    nuclear_contract = _evaluate_nuclear_contracts(iso, year, active_plants)

    retired_plants = []
    retired_by_type = {}
    total_retired_mw = 0.0
    cumulative_retired_this_year = 0.0

    for plant in active_plants:
        plant_id = plant.get('plant_id', plant.get('generator_id', ''))
        margin = plant.get('profit_per_mwh', 0)
        cap_mw = plant.get('capacity_mw', 0)
        utype = plant.get('fuel_type', '') or plant.get('prime_mover', '')
        zone = plant.get('zone')

        # Skip nuclear plants — handled separately below
        if _is_nuclear_plant(plant):
            continue

        # Retirement decision based on individual margin
        if margin >= RETIREMENT_THRESHOLD:
            # Profitable or marginally economic — no retirement
            continue

        # Check reliability floor: can we retire this plant?
        if cumulative_retired_this_year + cap_mw > max_retirable_mw:
            _log(f"    {iso} reliability floor: cannot retire {plant.get('plant_name', plant_id)} "
                 f"({cap_mw:.0f} MW) — would breach 15% reserve margin")
            continue

        # Retire this plant
        cumulative_retired_this_year += cap_mw
        total_retired_mw += cap_mw
        unit_type_key = plant.get('unit_type', utype) or 'unknown'
        retired_by_type[unit_type_key] = retired_by_type.get(unit_type_key, 0) + cap_mw

        retired_plants.append({
            'plant_id': plant_id,
            'generator_id': plant.get('generator_id', ''),
            'plant_name': plant.get('plant_name', ''),
            'capacity_mw': round(cap_mw, 1),
            'unit_type': unit_type_key,
            'margin': round(margin, 2),
            'iso': iso,
            'zone': zone,
            'year_retired': year,
        })

        _log(f"    {iso} retire {plant.get('plant_name', plant_id)} "
             f"({unit_type_key}, {cap_mw:.0f} MW, margin ${margin:.1f}/MWh)")

    # --- Nuclear plant-level retirement ---
    nuclear_retired_plants = _retire_nuclear_plants(
        iso, year, state, active_plants, nuclear_contract,
        cumulative_retired_this_year, max_retirable_mw, _log)
    for nplant in nuclear_retired_plants:
        total_retired_mw += nplant['capacity_mw']
        retired_by_type['nuclear'] = retired_by_type.get('nuclear', 0) + nplant['capacity_mw']
        retired_plants.append(nplant)

    # Persist retired plant IDs in state
    new_retired_ids = prior_retired | {p['plant_id'] for p in retired_plants}
    state['retired_plants'] = list(new_retired_ids)

    # Also maintain legacy economic_retirements dict for backward compatibility
    prior_econ_retirements = state.get('economic_retirements', {})
    for utype, mw in retired_by_type.items():
        prior_econ_retirements[utype] = prior_econ_retirements.get(utype, 0) + mw
    state['economic_retirements'] = prior_econ_retirements

    # Build adjusted gen_econ: reduce capacity per unit type by retired amount
    adjusted = {}
    for utype, econ in gen_econ.items():
        adj = dict(econ)
        retired_mw = retired_by_type.get(utype, 0)
        if retired_mw > 0 and adj.get('capacity_mw', 0) > 0:
            adj['capacity_mw'] = max(0, adj['capacity_mw'] - retired_mw)
        adjusted[utype] = adj

    if total_retired_mw > 0:
        _log(f"    {iso} total plant-level retirement: {total_retired_mw:.0f} MW "
             f"({len(retired_plants)} plants)")

    return adjusted, retired_by_type, total_retired_mw, retired_plants


def _is_nuclear_plant(plant):
    """Check if a plant dict represents a nuclear unit."""
    fuel = str(plant.get('fuel_type', '')).upper()
    mover = str(plant.get('prime_mover', '')).upper()
    utype = str(plant.get('unit_type', '')).lower()
    return ('NUC' in fuel or 'UR' in fuel or  # NUC, NUCLEAR, URANIUM
            'ST' == mover and 'NUC' in fuel or
            utype == 'nuclear')


def _evaluate_nuclear_contracts(iso, year, active_plants):
    """Evaluate per-plant nuclear contract economics.

    Returns dict of {plant_id: {contract_protected: bool, contract_revenue_mwh: float}}
    """
    offtake = NUCLEAR_OFFTAKE_CONTRACTS.get(iso)
    result = {}

    nuclear_plants = [p for p in active_plants if _is_nuclear_plant(p)]
    for plant in nuclear_plants:
        pid = plant.get('plant_id', plant.get('generator_id', ''))
        if offtake and year <= offtake.get('contract_end_year', 0):
            # Plant has explicit contract protection
            result[pid] = {
                'contract_protected': True,
                'contract_revenue_mwh': offtake['contract_floor_mwh'],
                'contract_name': offtake.get('name', ''),
            }
        else:
            # No explicit contract — relies on capacity market or merchant revenue
            # ISOs with capacity markets (PJM, NYISO, NEISO, MISO) provide
            # going-forward cost recovery separate from energy revenue
            has_capacity_market = iso in ('PJM', 'NYISO', 'NEISO', 'MISO')
            cap_price = CAPACITY_MARKET_PRICES.get(iso, 0) if has_capacity_market else 0
            result[pid] = {
                'contract_protected': False,
                'capacity_market_revenue_kw_yr': cap_price,
                'has_capacity_market': has_capacity_market,
            }
    return result


def _retire_nuclear_plants(iso, year, state, active_plants, nuclear_contract,
                           cumulative_retired_mw, max_retirable_mw, _log=print):
    """Evaluate and retire individual nuclear plants based on contract economics.

    Plants with below-market offtake contracts are at higher stranding risk.
    Plants protected by PPA floors or capacity market revenue are more resilient.
    Nuclear retirement is individual, not fleet-wide.

    Returns list of retired nuclear plant dicts.
    """
    NUCLEAR_OPERATING_COST_MWH = 30.0  # $/MWh typical nuclear OPEX (fuel + O&M)

    nuclear_plants = [p for p in active_plants if _is_nuclear_plant(p)]
    if not nuclear_plants:
        return []

    retired = []
    for plant in nuclear_plants:
        pid = plant.get('plant_id', plant.get('generator_id', ''))
        cap_mw = plant.get('capacity_mw', 0)
        energy_rev = plant.get('revenue_per_mwh', 0) or plant.get('profit_per_mwh', 0) + NUCLEAR_OPERATING_COST_MWH

        contract_info = nuclear_contract.get(pid, {})

        # Compute effective revenue including contract/capacity market support
        if contract_info.get('contract_protected'):
            # PPA floor guarantees minimum revenue
            effective_rev = max(energy_rev, contract_info['contract_revenue_mwh'])
        elif contract_info.get('has_capacity_market'):
            # Capacity market revenue supplements energy revenue
            cap_rev_mwh = contract_info.get('capacity_market_revenue_kw_yr', 0) * 1000 / 8760
            effective_rev = energy_rev + cap_rev_mwh
        else:
            effective_rev = energy_rev

        # Retirement decision: does effective revenue cover operating costs?
        margin = effective_rev - NUCLEAR_OPERATING_COST_MWH

        if margin >= -5:
            continue  # Plant is economic — survives

        # Check reliability floor
        if cumulative_retired_mw + cap_mw > max_retirable_mw:
            _log(f"    {iso} reliability floor: cannot retire nuclear plant {pid} "
                 f"({cap_mw:.0f} MW)")
            continue

        cumulative_retired_mw += cap_mw
        retired.append({
            'plant_id': pid,
            'generator_id': plant.get('generator_id', ''),
            'plant_name': plant.get('plant_name', ''),
            'capacity_mw': round(cap_mw, 1),
            'unit_type': 'nuclear',
            'margin': round(margin, 2),
            'iso': iso,
            'zone': plant.get('zone'),
            'year_retired': year,
            'contract_protected': contract_info.get('contract_protected', False),
        })

        _log(f"    {iso} retire nuclear {plant.get('plant_name', pid)} "
             f"({cap_mw:.0f} MW, margin ${margin:.1f}/MWh"
             f"{', contract-protected' if contract_info.get('contract_protected') else ''})")

    return retired


def _apply_fleet_fraction_retirement(gen_econ, iso, year, state, _log=print):
    """Legacy fleet-fraction retirement (fallback when plant data unavailable).

    Preserved for backward compatibility when plant_economics is not provided.
    """
    prior_retirements = state.get('economic_retirements', {})
    retired_capacity = {}
    total_retired_mw = 0.0

    adjusted = {}
    for utype, econ in gen_econ.items():
        margin = econ.get('margin_mwh', 0)
        cap_mw = econ.get('capacity_mw', 0)
        prior_retired_mw = prior_retirements.get(utype, 0)

        if margin < -5:
            loss_depth = min(abs(margin), 30)
            retire_frac = 0.20 + 0.70 * ((loss_depth - 5) / 25)
            retire_frac = min(0.90, retire_frac)
            cumulative_frac = min(0.95, retire_frac + prior_retired_mw / cap_mw if cap_mw > 0 else 0)
            retire_mw = cap_mw * cumulative_frac
            remaining_mw = cap_mw - retire_mw
            min_mw = cap_mw * 0.05
            remaining_mw = max(remaining_mw, min_mw)
            retire_mw = cap_mw - remaining_mw

            retired_capacity[utype] = retire_mw
            total_retired_mw += retire_mw
            adj = dict(econ)
            if cap_mw > 0:
                adj['capacity_mw'] = remaining_mw
            adjusted[utype] = adj

            _log(f"    {iso} {utype}: margin ${margin:.1f}/MWh → "
                 f"retire {retire_mw:.0f} MW ({cumulative_frac*100:.0f}%), "
                 f"keep {remaining_mw:.0f} MW")

        elif margin < 2:
            at_risk_frac = 0.10
            prior_frac = prior_retired_mw / cap_mw if cap_mw > 0 else 0
            cumulative_frac = min(0.50, at_risk_frac + prior_frac)
            retire_mw = cap_mw * cumulative_frac
            remaining_mw = max(cap_mw * 0.10, cap_mw - retire_mw)
            retire_mw = cap_mw - remaining_mw

            retired_capacity[utype] = retire_mw
            total_retired_mw += retire_mw
            adj = dict(econ)
            adj['capacity_mw'] = remaining_mw
            adjusted[utype] = adj
        else:
            adjusted[utype] = dict(econ)

    updated_retirements = dict(prior_retirements)
    for utype, mw in retired_capacity.items():
        updated_retirements[utype] = updated_retirements.get(utype, 0) + mw
    state['economic_retirements'] = updated_retirements

    if total_retired_mw > 0:
        _log(f"    {iso} total economic retirement: {total_retired_mw:.0f} MW")

    return adjusted, retired_capacity, total_retired_mw


def apply_economic_new_build(gen_econ, iso, year, state, conditions,
                              demand_twh, hourly_lmp, _log=print,
                              reserve_margin_pct=None):
    """Evaluate and apply economic new-build fossil capacity (CCGT, CT, coal).

    New fossil capacity is built when:
    1. RA trigger: reserve margin falls below target after retirements + clean
       deployment, requiring new dispatchable capacity.
    2. Economic trigger: LMP levels support positive margins for new plants
       above their minimum CF threshold.

    The cheapest viable option is built first. Capacity is capped at the
    per-ISO annual build rate × years_in_period.

    New-build fossil cost level is a sweep parameter ('Low'/'Medium'/'High')
    in the 1,215 parametric sweep, and a free-form input in single trajectory
    runs (via conditions['new_fossil_cost_level'] or per-type overrides).

    Args:
        gen_econ: Current generator economics dict from dispatch model.
        iso: ISO region string.
        year: Simulation year.
        state: Per-ISO mutable state dict (tracks cumulative builds).
        conditions: Scenario conditions dict. Keys used:
            - new_fossil_cost_level: 'Low'/'Medium'/'High' (default 'Medium')
            - new_fossil_capex_override: {type: $/kW-yr} overrides per type
            - new_fossil_min_cf_override: {type: fraction} overrides per type
            - new_fossil_enabled: bool (default True)
        demand_twh: Demand in TWh for this ISO/year.
        hourly_lmp: Array of 8760 hourly LMP values ($/MWh).
        _log: Logging function.

    Returns:
        new_builds: dict of {unit_type: new_capacity_mw}
        total_new_mw: float — total MW of new fossil built
        new_build_details: dict with per-type economics for output
    """
    if not conditions.get('new_fossil_enabled', True):
        return {}, 0.0, {}

    cost_level = conditions.get('new_fossil_cost_level', 'Medium')
    capex_override = conditions.get('new_fossil_capex_override', {})
    min_cf_override = conditions.get('new_fossil_min_cf_override', {})
    carbon_price = conditions.get('carbon_price', 0)

    # Get fuel prices for variable cost calculation
    fuel_level = conditions.get('fuel_level', 'Medium')
    fp = FUEL_PRICES.get(fuel_level, FUEL_PRICES['Medium'])

    # --- Compute RA gap ---
    # Total existing fossil capacity (post-retirement)
    existing_fossil_mw = sum(e.get('capacity_mw', 0) for e in gen_econ.values())
    # Prior new-build capacity already online
    prior_new_builds = state.get('new_fossil_builds', {})
    prior_new_mw = sum(prior_new_builds.values())
    # Clean capacity from cumulative deployments
    cumulative_gw = state.get('cumulative_gw', {})
    clean_cap_mw = sum(v * 1000 for v in cumulative_gw.values())

    total_supply_mw = existing_fossil_mw + prior_new_mw + clean_cap_mw
    avg_demand_mw = demand_twh * 1e6 / 8760
    peak_demand_mw = PEAK_DEMAND_MW.get(iso, avg_demand_mw * 1.5)
    # Scale peak demand by growth factor
    growth_factor = demand_twh / REGIONAL_DEMAND_TWH.get(iso, demand_twh)
    peak_demand_mw *= growth_factor

    target_reserve = RESOURCE_ADEQUACY_MARGIN  # 0.15
    required_supply_mw = peak_demand_mw * (1 + target_reserve)
    ra_gap_mw = max(0, required_supply_mw - total_supply_mw)

    # --- Annual build cap ---
    # How many years this period covers (matches queue budget logic)
    years_in_period = 7 if year == 2030 else 5
    max_build_mw = NEW_BUILD_MAX_GW_YR.get(iso, 3.0) * 1000 * years_in_period

    # --- Evaluate each fossil type ---
    FOSSIL_TYPES = ['gas_ccgt', 'gas_ct', 'coal']
    FUEL_KEY_MAP = {'gas_ccgt': 'gas', 'gas_ct': 'gas', 'coal': 'coal'}

    candidates = []
    build_details = {}

    for ftype in FOSSIL_TYPES:
        # CAPEX: override or from L/M/H table
        if ftype in capex_override:
            capex_kw_yr = capex_override[ftype]
        else:
            capex_table = NEW_BUILD_CAPEX_KW_YR.get(cost_level, {}).get(ftype, {})
            capex_kw_yr = capex_table.get(iso, 999)

        # Skip if effectively blocked (999 = not buildable in this ISO)
        if capex_kw_yr >= 900:
            continue

        # Min CF threshold: override or default
        min_cf = min_cf_override.get(ftype, NEW_BUILD_MIN_CF.get(ftype, 0.30))

        # Variable cost: fuel + VOM + carbon
        hr = NEW_BUILD_HEAT_RATES.get(ftype, HEAT_RATES.get(ftype, 7.0))
        vom = NEW_BUILD_VOM.get(ftype, VOM.get(ftype, 3.5))
        co2_rate = NEW_BUILD_CO2_RATES.get(ftype, CO2_RATES.get(ftype, 0.37))
        fuel_key = FUEL_KEY_MAP[ftype]
        fuel_price = fp.get(fuel_key, 3.50)

        var_cost = hr * fuel_price + vom + co2_rate * carbon_price

        # All-in cost at min CF: CAPEX annuity / (CF × 8760) + var_cost
        # CAPEX is $/kW-yr → $/MWh = CAPEX / (CF × 8.760)
        capex_per_mwh = capex_kw_yr / (min_cf * 8.760) if min_cf > 0 else float('inf')
        all_in_cost = capex_per_mwh + var_cost

        # Revenue estimate: what would this unit earn at its dispatch position?
        # For new builds, they dispatch when LMP > var_cost
        dispatch_hours = np.sum(hourly_lmp > var_cost)
        if dispatch_hours > 0:
            dispatch_mask = hourly_lmp > var_cost
            avg_rev_when_dispatched = float(np.mean(hourly_lmp[dispatch_mask]))
            expected_cf = dispatch_hours / len(hourly_lmp)
        else:
            avg_rev_when_dispatched = 0.0
            expected_cf = 0.0

        # Net margin including CAPEX at expected CF
        if expected_cf > 0:
            capex_at_actual_cf = capex_kw_yr / (expected_cf * 8.760)
            net_margin = avg_rev_when_dispatched - var_cost - capex_at_actual_cf
        else:
            net_margin = -999

        # Capacity market revenue offset — endogenous pricing
        _rm = reserve_margin_pct if reserve_margin_pct is not None else 100.0
        _cp = state.get('clean_pct', 0)
        cap_mkt_price = compute_capacity_price(iso, _rm, _cp)
        if cap_mkt_price > 0:
            # New builds get full ELCC (1.0 for dispatchable)
            cap_rev_per_mwh = cap_mkt_price / (expected_cf * 8.760) if expected_cf > 0 else 0
            net_margin += cap_rev_per_mwh

        detail = {
            'capex_kw_yr': round(capex_kw_yr, 1),
            'var_cost': round(var_cost, 2),
            'all_in_at_min_cf': round(all_in_cost, 2),
            'expected_cf': round(expected_cf, 4),
            'avg_rev': round(avg_rev_when_dispatched, 2),
            'net_margin': round(net_margin, 2),
            'dispatch_hours': int(dispatch_hours),
            'viable': expected_cf >= min_cf and net_margin > 0,
        }
        build_details[ftype] = detail

        # Two paths to build: RA need or economic viability
        if expected_cf >= min_cf and net_margin > 0:
            candidates.append((ftype, net_margin, var_cost, capex_kw_yr))

    # --- Build decision ---
    new_builds = {}
    total_new_mw = 0.0
    remaining_build_cap = max_build_mw

    # Sort candidates by net margin (most profitable first)
    candidates.sort(key=lambda x: -x[1])

    if ra_gap_mw > 0 and candidates:
        # RA-driven: fill the gap with cheapest viable option
        # For RA, prefer CT (faster to build, lower CAPEX) unless CCGT is more profitable
        ra_candidates = sorted(candidates, key=lambda x: x[3])  # sort by CAPEX
        best_ra = ra_candidates[0]
        ftype = best_ra[0]
        build_mw = min(ra_gap_mw, remaining_build_cap)
        new_builds[ftype] = new_builds.get(ftype, 0) + build_mw
        total_new_mw += build_mw
        remaining_build_cap -= build_mw
        _log(f"    {iso} RA gap {ra_gap_mw:.0f} MW → new-build {ftype} "
             f"{build_mw:.0f} MW (CAPEX ${best_ra[3]}/kW-yr)")

    # Economic-driven: build additional if profitable and room remains
    for ftype, margin, var_cost, capex in candidates:
        if remaining_build_cap <= 0:
            break
        if ftype in new_builds:
            continue  # Already built for RA

        # Economic builds: size to capture scarcity/margin opportunity
        # Build enough to serve ~50% of hours where the unit would dispatch
        # above min CF, capped at build rate
        detail = build_details[ftype]
        if not detail['viable']:
            continue

        # Scale build size by margin attractiveness
        # Higher margins → larger build (up to build cap)
        margin_scale = min(1.0, margin / 20.0)  # Normalize: $20/MWh margin = full build
        econ_build_mw = remaining_build_cap * margin_scale * 0.5  # conservative: 50% of available
        econ_build_mw = max(100, econ_build_mw)  # minimum 100 MW unit
        econ_build_mw = min(econ_build_mw, remaining_build_cap)

        new_builds[ftype] = new_builds.get(ftype, 0) + econ_build_mw
        total_new_mw += econ_build_mw
        remaining_build_cap -= econ_build_mw
        _log(f"    {iso} economic new-build {ftype} {econ_build_mw:.0f} MW "
             f"(margin ${margin:.1f}/MWh, CF {detail['expected_cf']:.1%})")

    # Persist cumulative builds in state
    if total_new_mw > 0:
        if 'new_fossil_builds' not in state:
            state['new_fossil_builds'] = {}
        for ftype, mw in new_builds.items():
            state['new_fossil_builds'][ftype] = state['new_fossil_builds'].get(ftype, 0) + mw
        state['gas_built_gw'] = state.get('gas_built_gw', 0) + sum(
            mw for ft, mw in new_builds.items() if ft.startswith('gas')) / 1000.0
        state['fossil_built_gw'] = state.get('fossil_built_gw', 0) + total_new_mw / 1000.0
        _log(f"    {iso} total new fossil: {total_new_mw:.0f} MW "
             f"(cumulative: {state['fossil_built_gw']:.2f} GW)")

    return new_builds, total_new_mw, build_details


def compute_reserve_margin(gen_econ, cumulative_gw, demand_twh):
    """Compute reserve margin percentage from current fleet state.

    Args:
        gen_econ: Dict of generator economics {unit_id: {capacity_mw, ...}}
        cumulative_gw: Dict of cumulative deployed clean GW {tech: GW}
        demand_twh: Total annual demand in TWh

    Returns:
        Reserve margin as percentage (e.g., 15.0 means 15%)
    """
    total_cap_mw = sum(e.get('capacity_mw', 0) for e in gen_econ.values())
    clean_cap_mw = sum(v * 1000 for v in cumulative_gw.values())
    total_supply_mw = total_cap_mw + clean_cap_mw
    avg_demand_mw = demand_twh * 1e6 / 8760
    peak_demand_mw = avg_demand_mw * _PEAK_TO_AVG_RATIO
    if peak_demand_mw <= 0:
        return 100.0  # Default to high reserve margin if no demand
    return (total_supply_mw - peak_demand_mw) / peak_demand_mw * 100


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


# ═══════════════════════════════════════════════════════════════════════════════
# ECONOMICS-DRIVEN STORAGE DEPLOYMENT (R1)
# ═══════════════════════════════════════════════════════════════════════════════

# Storage technology parameters for arbitrage calculation
_STORAGE_TECH_PARAMS = {
    'battery':  {'duration': 4,    'efficiency': 0.85, 'cycles_per_year': 365, 'window_days': 1},
    'battery8': {'duration': 8,    'efficiency': 0.85, 'cycles_per_year': 365, 'window_days': 2},
    'ldes':     {'duration': 100,  'efficiency': 0.50, 'cycles_per_year': 52,  'window_days': 7},
    'h2':       {'duration': 1000, 'efficiency': 0.35, 'cycles_per_year': 12,  'window_days': 30},
}


def compute_storage_arbitrage_from_lmp(hourly_lmp, iso=None):
    """Compute realized arbitrage revenue ($/kW-yr) for each storage tech.

    Uses price-taking dispatch: charge during lowest-LMP hours, discharge
    during highest-LMP hours per cycle window. Vectorized numpy — no Numba
    needed for 8760 elements.

    Args:
        hourly_lmp: 8760 array of hourly LMP ($/MWh)
        iso: ISO string (unused currently, reserved for future regional adjustments)

    Returns:
        dict mapping tech name to realized arbitrage revenue in $/kW-yr
    """
    lmp = np.asarray(hourly_lmp, dtype=np.float64)
    results = {}

    for tech, params in _STORAGE_TECH_PARAMS.items():
        duration = params['duration']
        rte = params['efficiency']
        cycles = params['cycles_per_year']
        window_hours = params['window_days'] * 24

        # Number of windows in the year
        n_windows = min(cycles, H // window_hours) if window_hours > 0 else 0
        if n_windows <= 0 or duration <= 0:
            results[tech] = 0.0
            continue

        # Per-window arbitrage: sort LMP within each window, charge at bottom,
        # discharge at top. This captures realistic temporal constraints
        # (can't charge in January and discharge in June for a 4hr battery).
        #
        # For a 1 kW power-rated system with `duration` hours of storage:
        #   - Charges at 1 kW for `duration` hours → stores `duration` kWh
        #   - Discharges at 1 kW for `duration` hours → delivers `duration * RTE` kWh
        #   - Per-cycle revenue = avg(discharge_lmp) * duration * RTE - avg(charge_lmp) * duration
        #   - Units: $/MWh × hours = $/MW per cycle → ÷ 1000 for $/kW per cycle
        total_revenue_dollar_per_mw = 0.0
        charge_hours_per_window = min(duration, window_hours // 2)
        discharge_hours_per_window = charge_hours_per_window  # symmetric

        for w in range(n_windows):
            w_start = w * window_hours
            w_end = min(w_start + window_hours, H)
            if w_end - w_start < 2 * charge_hours_per_window:
                continue  # window too short

            window_lmp = lmp[w_start:w_end]
            sorted_idx = np.argsort(window_lmp)

            # Charge at cheapest hours, discharge at most expensive
            # Each hour: 1 MW power → 1 MWh energy, cost/revenue = LMP × 1 MWh
            charge_cost = np.sum(window_lmp[sorted_idx[:charge_hours_per_window]])
            discharge_rev = np.sum(window_lmp[sorted_idx[-discharge_hours_per_window:]])

            # Net revenue per cycle in $/MW (1 MW power capacity assumed)
            cycle_revenue = discharge_rev * rte - charge_cost
            total_revenue_dollar_per_mw += cycle_revenue

        # Convert $/MW-yr → $/kW-yr (÷ 1000)
        results[tech] = max(0.0, total_revenue_dollar_per_mw / 1000.0)

    return results


def compute_nuclear_revenue(iso, clean_pct, hourly_lmp, year, conditions=None,
                             reserve_margin_pct=None):
    """Compute nuclear plant revenue stack by ISO.

    Returns dict with energy_rev, capacity_rev, ptc, total (all $/MWh).
    Uses 45U contract-for-difference floor mechanism when conditions provided.
    Capacity price is endogenous — responds to reserve margin and clean share.
    """
    nuclear_cf = 0.93
    nuclear_gen = np.ones(H) * nuclear_cf  # Flat baseload

    # Energy revenue: LMP × generation
    energy_rev = float(np.mean(hourly_lmp * nuclear_gen)) / nuclear_cf

    # Capacity revenue — endogenous pricing (scarcity + clean degradation)
    _rm = reserve_margin_pct if reserve_margin_pct is not None else 100.0
    if conditions and conditions.get('capacity_market_price') is not None:
        degraded_price = conditions['capacity_market_price']
    else:
        degraded_price = compute_capacity_price(iso, _rm, clean_pct)
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
                                        demand_total_mwh, iso=None,
                                        zonal_lmp_matrix=None,
                                        zonal_zone_names=None):
    """Compute energy revenue ($/MWh) per resource from hourly LMP × generation.

    When zonal LMP data is available, VRE resources (solar, wind, offshore_wind)
    use the LMP of their primary zone (from VRE_PRIMARY_ZONE) instead of the
    system-average LMP. Non-VRE resources and VRE resources without a zone
    mapping fall back to system-average LMP.

    Returns:
        dict {resource: $/MWh} — profile-weighted average energy revenue
    """
    # Build zone name → row index lookup for zonal LMP
    _zone_idx = {}
    if zonal_lmp_matrix is not None and zonal_zone_names is not None:
        _zone_idx = {z: i for i, z in enumerate(zonal_zone_names)}

    vre_zone_map = VRE_PRIMARY_ZONE.get(iso, {}) if iso else {}

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
            # Use zonal LMP for VRE resources when available
            effective_lmp = hourly_lmp
            if _zone_idx and res in vre_zone_map:
                zone_name = vre_zone_map[res]
                z_idx = _zone_idx.get(zone_name)
                if z_idx is not None:
                    effective_lmp = zonal_lmp_matrix[z_idx]
            revenue_weighted = float(np.sum(gen_profile * effective_lmp))
            gen_total = float(np.sum(gen_profile))
            revenues[res] = revenue_weighted / gen_total if gen_total > 0 else 0
    return revenues


def compute_capacity_revenue(iso, clean_pct, resource_pcts, conditions=None,
                              reserve_margin_pct=None):
    """Compute capacity market revenue ($/MWh) per resource."""
    _rm = reserve_margin_pct if reserve_margin_pct is not None else 100.0
    if conditions and conditions.get('capacity_market_price') is not None:
        degraded_price = conditions['capacity_market_price']
    else:
        degraded_price = compute_capacity_price(iso, _rm, clean_pct)

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
                          rec_price_override=None, conditions=None,
                          zonal_lmp_matrix=None, zonal_zone_names=None,
                          reserve_margin_pct=None):
    """Total blended revenue ($/MWh) for resources at this zone.

    When zonal_lmp_matrix is provided, VRE resources use zone-specific LMP
    for energy revenue. The basis_differential per resource ($/MWh difference
    between zone LMP and system LMP) is included in the breakdown.
    """
    energy_revs = compute_energy_revenue_by_resource(
        hourly_lmp, supply_profiles, resource_pcts, demand_total_mwh,
        iso=iso, zonal_lmp_matrix=zonal_lmp_matrix,
        zonal_zone_names=zonal_zone_names)
    cap_revs = compute_capacity_revenue(iso, clean_pct, resource_pcts,
                                         conditions=conditions,
                                         reserve_margin_pct=reserve_margin_pct)
    rec_revs = compute_rec_revenue(iso, resource_pcts, clean_pct, year,
                                    rec_price_override=rec_price_override)

    # Compute basis differentials: zone LMP − system LMP for VRE resources
    basis_diffs = {}
    if zonal_lmp_matrix is not None and zonal_zone_names is not None and iso:
        _zone_idx = {z: i for i, z in enumerate(zonal_zone_names)}
        vre_zone_map = VRE_PRIMARY_ZONE.get(iso, {})
        system_avg = float(np.mean(hourly_lmp))
        for res in resource_pcts:
            if res in vre_zone_map:
                zone_name = vre_zone_map[res]
                z_idx = _zone_idx.get(zone_name)
                if z_idx is not None:
                    zone_avg = float(np.mean(zonal_lmp_matrix[z_idx]))
                    basis_diffs[res] = round(zone_avg - system_avg, 2)

    total_pct = sum(pct for pct in resource_pcts.values() if pct > 0)
    if total_pct <= 0:
        return 0, {}, {'energy_rev_mwh': 0, 'capacity_rev_mwh': 0, 'rec_rev_mwh': 0,
                        'basis_differentials': {}}

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
        'basis_differentials': basis_diffs,
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

    # Learning curves toggle — when Off, use base LCOE at selected level (no Wright's Law)
    learning_enabled = conditions.get('learning_curves_enabled', True) if conditions else True

    tech = RESOURCE_TO_TECH.get(res, res)

    if res == 'clean_firm':
        base_lcoe = NUCLEAR_NEWBUILD_LCOE.get(lcoe_level, {}).get(iso, 100)
        ptc_new_nuc = conditions.get('ptc_nuclear_new', PTC_45Y_NEW_NUCLEAR) if conditions else PTC_45Y_NEW_NUCLEAR
        if not learning_enabled:
            return max(0, base_lcoe - ptc_new_nuc)
        foak = FOAK_NUCLEAR_NEWBUILD.get(iso, 175)
        noak = NUCLEAR_NEWBUILD_LCOE.get('Low', {}).get(iso, 68)
        lr = WRIGHT_LEARNING_RATE.get('nuclear', {}).get(learning_speed, 0.12)
        ref_gw = WRIGHT_CUMULATIVE_GW_2025.get('nuclear', 2.0)
        eff_gw = get_effective_cumulative_gw('nuclear', cumulative_gw.get('nuclear', 0),
                                              learning_speed, year)
        cost = wright_cost(foak, noak, eff_gw, ref_gw, lr)
        # Phase 2A: Parameterized 45Y PTC for new nuclear
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

        if not learning_enabled:
            return base_lcoe
        lr = WRIGHT_LEARNING_RATE.get('ccs', {}).get(learning_speed, 0.10)
        ref_gw = WRIGHT_CUMULATIVE_GW_2025.get('ccs', 0.3)
        eff_gw = get_effective_cumulative_gw('ccs', cumulative_gw.get('ccs', 0),
                                              learning_speed, year)
        return wright_cost(foak, noak, eff_gw, ref_gw, lr)

    elif res == 'geothermal':
        base_lcoe = GEOTHERMAL_LCOE.get(lcoe_level, 88)
        if not learning_enabled:
            return base_lcoe
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

        # R2: Apply Wright's Law learning for storage techs with nonzero learning rates
        if learning_enabled and res in ('battery', 'battery8', 'ldes', 'h2'):
            tech = RESOURCE_TO_TECH.get(res, res)
            lr = WRIGHT_LEARNING_RATE.get(tech, {}).get(learning_speed, 0.0)
            if lr > 0:
                noak_tables = {
                    'battery': NOAK_BATTERY, 'battery8': NOAK_BATTERY8,
                }.get(tech)
                if noak_tables:
                    noak = noak_tables.get(lcoe_level, {}).get(iso, base * 0.5)
                else:
                    noak = base * 0.5  # fallback for ldes/h2
                foak = base  # Current LCOE_TABLES value is the starting (FOAK) cost
                ref_gw = WRIGHT_CUMULATIVE_GW_2025.get(tech, 1.0)
                eff_gw = get_effective_cumulative_gw(tech, cumulative_gw.get(tech, 0),
                                                      learning_speed, year)
                base = wright_cost(foak, noak, eff_gw, ref_gw, lr)

        return max(0, base)


def compute_lcoe_snapshot(iso, cumulative_gw, lcoe_level, learning_speed, year,
                          conditions=None):
    """Compute current LCOE for all deployable techs at given cumulative GW.

    Returns a dict {tech: lcoe_$/MWh} reflecting the Wright's Law cost
    reduction achieved so far.  Used to populate lcoe_trajectory in YearResult.
    """
    snapshot = {}
    for res in DEPLOYABLE_RESOURCES:
        if res == 'geothermal' and iso != 'CAISO':
            continue
        if res == 'offshore_wind' and iso not in ('CAISO', 'NYISO', 'NEISO', 'PJM'):
            continue
        lcoe = get_resource_lcoe(res, iso, lcoe_level, cumulative_gw,
                                  learning_speed, year, conditions=conditions)
        tech = RESOURCE_TO_TECH.get(res, res)
        snapshot[tech] = round(lcoe, 2)
    # Also include storage techs that have learning curves
    for storage_res in ('battery', 'battery8', 'ldes'):
        lcoe = get_resource_lcoe(storage_res, iso, lcoe_level, cumulative_gw,
                                  learning_speed, year, conditions=conditions)
        snapshot[storage_res] = round(lcoe, 2)
    return snapshot


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

    tx_overrides = (conditions or {}).get('tx_overrides') or {}

    for res, delta_twh in delta_resources.items():
        if delta_twh <= 0:
            continue
        lcoe = get_resource_lcoe(res, iso, lcoe_level, cumulative_gw,
                                  learning_speed, year, conditions=conditions)

        if res in ('solar', 'wind', 'clean_firm', 'offshore_wind'):
            res_key = res if res != 'clean_firm' else 'nuclear'
            tx_override = tx_overrides.get(res_key)
            tx = tx_override if tx_override is not None else get_tx(res if res != 'clean_firm' else 'clean_firm', tx_level, iso)
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
        ccs_lcoe = CCS_LCOE_45Q_OFF.get('M', {}).get(iso, 100) - ccs_credit_override
    else:
        q45_on = conditions.get('q45', True) if conditions else True
        ccs_table = CCS_LCOE_45Q_ON if q45_on else CCS_LCOE_45Q_OFF
        ccs_lcoe = ccs_table.get('M', {}).get(iso, 100)
    ccs_co2_rate = gas_co2 * 0.10  # 90% capture → 10% residual

    # Breakeven: existing_cost + gas_co2 × Cp = ccs_lcoe + ccs_co2 × Cp
    # Cp × (gas_co2 - ccs_co2) = ccs_lcoe - existing_fixed_cost
    co2_diff = gas_co2 - ccs_co2_rate
    if co2_diff > 0:
        breakeven_ccs = (ccs_lcoe - existing_fixed_cost) / co2_diff
    else:
        breakeven_ccs = float('inf')

    # New efficient gas vs old gas (HR 8.0+)
    new_hr = conditions.get('custom_heat_rates', {}).get('new_gas_ccgt', 6.2) if conditions else 6.2
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


# ═══════════════════════════════════════════════════════════════════════════════
# IPM TRIGGER INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════
# Flags when simulation results cross thresholds where the screening model's
# approximations break down, recommending production-model validation.

# Nuclear retirement threshold used by trigger check ($/MWh).
_NUCLEAR_RETIREMENT_THRESHOLD_DEFAULT = 30.0

# VRE resources to sum for penetration check
_VRE_RESOURCES = {'solar', 'wind', 'offshore_wind'}

# Storage resources to sum for dominance check
_STORAGE_RESOURCES = {'battery_4hr', 'battery_8hr', 'ldes', 'green_h2'}

# Peak-to-average demand ratio (proxy for reserve-margin calc)
_PEAK_TO_AVG_RATIO = 1.5


def compute_ipm_triggers(iso, year, year_result, gen_econ, state, conditions,
                         nuclear_retirement_threshold=None,
                         zonal_congestion_data=None):
    """Evaluate IPM trigger conditions for this year's results.

    Pure threshold checks — negligible overhead (<0.01ms per ISO-year).

    Args:
        zonal_congestion_data: Optional dict from zonal LP solver containing
            '_congestion' key with inter-zonal LMP spreads and flow utilization.
            If None, falls back to VRE-ratio proxy for HIGH_CONGESTION trigger.

    Returns:
        list[dict]: Each dict matches IPMTrigger schema:
            trigger_id, severity, explanation, metric_value, threshold,
            recommended_model.
    """
    triggers = []
    demand_twh = year_result.get('demand_twh', 0)
    resource_mix = year_result.get('resource_mix_twh', {})
    if demand_twh <= 0:
        return triggers

    # ── VRE_CANNIBALIZATION ──────────────────────────────────────────────
    vre_twh = sum(resource_mix.get(r, 0) for r in _VRE_RESOURCES)
    vre_pct = vre_twh / demand_twh * 100
    if vre_pct > 40:
        severity = 'high' if vre_pct > 60 else 'medium'
        triggers.append({
            'trigger_id': 'VRE_CANNIBALIZATION',
            'severity': severity,
            'explanation': (
                "VRE penetration above 40% causes significant price cannibalization "
                "effects. A production dispatch model with hourly granularity and "
                "curtailment modeling would better quantify revenue erosion and "
                "optimal storage sizing."
            ),
            'metric_value': round(vre_pct, 1),
            'threshold': 60.0 if severity == 'high' else 40.0,
            'recommended_model': 'Production dispatch model (PLEXOS, GenX)',
        })

    # ── TIGHT_RA_MARGIN ─────────────────────────────────────────────────
    # Reserve margin = (total capacity - peak demand) / peak demand
    total_cap_mw = sum(e.get('capacity_mw', 0) for e in gen_econ.values())
    # Add deployed clean capacity (cumulative_gw from state)
    cumulative_gw = year_result.get('cumulative_gw', {})
    clean_cap_mw = sum(v * 1000 for v in cumulative_gw.values())
    total_supply_mw = total_cap_mw + clean_cap_mw
    avg_demand_mw = demand_twh * 1e6 / 8760
    peak_demand_mw = avg_demand_mw * _PEAK_TO_AVG_RATIO
    if peak_demand_mw > 0:
        reserve_margin_pct = (total_supply_mw - peak_demand_mw) / peak_demand_mw * 100
        if reserve_margin_pct < 10:
            severity = 'high' if reserve_margin_pct < 5 else 'medium'
            triggers.append({
                'trigger_id': 'TIGHT_RA_MARGIN',
                'severity': severity,
                'explanation': (
                    "Reserve margins are tight enough that unit commitment constraints "
                    "(ramp rates, minimum generation, start-up costs) materially affect "
                    "price formation and reliability. A UC-constrained dispatch model "
                    "is recommended."
                ),
                'metric_value': round(reserve_margin_pct, 1),
                'threshold': 5.0 if severity == 'high' else 10.0,
                'recommended_model': 'UC-constrained dispatch model (IPM, PLEXOS)',
            })

    # ── HIGH_CONGESTION ──────────────────────────────────────────────────
    # Use actual zonal congestion data when available; fall back to VRE-ratio proxy
    _congestion_triggered = False
    cong = None
    if zonal_congestion_data and '_congestion' in zonal_congestion_data:
        cong = zonal_congestion_data['_congestion']
    elif year_result.get('zonal_congestion'):
        cong = year_result['zonal_congestion']

    if cong is not None:
        max_spread = cong.get('max_spread_p50', 0)
        spread_pair = cong.get('max_spread_pair', 'unknown')
        # Check flow utilization across all interfaces
        max_hours_70 = 0
        max_hours_95 = 0
        worst_iface_70 = ''
        worst_iface_95 = ''
        for iface in cong.get('interfaces', []):
            h70 = iface.get('hours_above_70pct', 0)
            h95 = iface.get('hours_above_95pct', 0)
            iface_label = f"{iface.get('zone_a', '?')}-{iface.get('zone_b', '?')}"
            if h70 > max_hours_70:
                max_hours_70 = h70
                worst_iface_70 = iface_label
            if h95 > max_hours_95:
                max_hours_95 = h95
                worst_iface_95 = iface_label

        # High: spread P50 > $25/MWh OR any interface at 95%+ for > 500 hours
        # Medium: spread P50 > $15/MWh OR any interface at 70%+ for > 1000 hours
        high_spread = max_spread > 25
        high_flow = max_hours_95 > 500
        med_spread = max_spread > 15
        med_flow = max_hours_70 > 1000

        if high_spread or high_flow:
            _congestion_triggered = True
            detail_parts = []
            if high_spread:
                detail_parts.append(
                    f"Max zonal LMP spread (P50) ${max_spread:.1f}/MWh "
                    f"on {spread_pair} interface")
            if high_flow:
                detail_parts.append(
                    f"{worst_iface_95} interface at 95%+ utilization "
                    f"for {max_hours_95} hours/year")
            triggers.append({
                'trigger_id': 'HIGH_CONGESTION',
                'severity': 'high',
                'explanation': (
                    f"Transmission congestion is binding. {'; '.join(detail_parts)}. "
                    "A nodal or detailed zonal dispatch model would better capture "
                    "locational price signals and congestion rent allocation."
                ),
                'metric_value': round(max_spread, 1),
                'threshold': 25.0,
                'recommended_model': 'Zonal/nodal dispatch model (PLEXOS, nodal IPM)',
            })
        elif med_spread or med_flow:
            _congestion_triggered = True
            detail_parts = []
            if med_spread:
                detail_parts.append(
                    f"Max zonal LMP spread (P50) ${max_spread:.1f}/MWh "
                    f"on {spread_pair} interface")
            if med_flow:
                detail_parts.append(
                    f"{worst_iface_70} interface at 70%+ utilization "
                    f"for {max_hours_70} hours/year")
            triggers.append({
                'trigger_id': 'HIGH_CONGESTION',
                'severity': 'medium',
                'explanation': (
                    f"Moderate transmission congestion detected. {'; '.join(detail_parts)}. "
                    "Zonal dispatch modeling would improve locational price accuracy "
                    "and resource siting decisions."
                ),
                'metric_value': round(max_spread, 1),
                'threshold': 15.0,
                'recommended_model': 'Zonal/nodal dispatch model (PLEXOS, nodal IPM)',
            })

    # Fallback: VRE-ratio proxy when zonal data unavailable (copper-plate mode)
    if not _congestion_triggered and cong is None:
        vre_gw = sum(cumulative_gw.get(r, 0) for r in _VRE_RESOURCES)
        years_elapsed = max(1, year - 2025)
        queue_cap = QUEUE_CAP_GW.get('Medium', {}).get(iso, 5)
        expected_gw = queue_cap * years_elapsed
        if expected_gw > 0:
            deploy_ratio = vre_gw / expected_gw
            if deploy_ratio > 2.0:
                severity = 'high' if deploy_ratio > 3.0 else 'medium'
                triggers.append({
                    'trigger_id': 'HIGH_CONGESTION',
                    'severity': severity,
                    'explanation': (
                        "Transmission congestion is material (proxy: VRE deployment "
                        "significantly exceeds historical interconnection queue "
                        "completion rates). Zonal or nodal dispatch modeling would "
                        "better capture locational price signals and their impact "
                        "on resource siting decisions."
                    ),
                    'metric_value': round(deploy_ratio, 2),
                    'threshold': 3.0 if severity == 'high' else 2.0,
                    'recommended_model': 'Zonal/nodal dispatch model (PLEXOS, nodal IPM)',
                })

    # ── STORAGE_DOMINANCE ────────────────────────────────────────────────
    storage_twh = sum(resource_mix.get(r, 0) for r in _STORAGE_RESOURCES)
    storage_pct = storage_twh / demand_twh * 100
    if storage_pct > 15:
        severity = 'high' if storage_pct > 25 else 'medium'
        triggers.append({
            'trigger_id': 'STORAGE_DOMINANCE',
            'severity': severity,
            'explanation': (
                "Storage is a major contributor to supply. Co-optimized storage "
                "dispatch (jointly with generation and unit commitment) would "
                "materially change utilization patterns and economics."
            ),
            'metric_value': round(storage_pct, 1),
            'threshold': 25.0 if severity == 'high' else 15.0,
            'recommended_model': 'Co-optimized storage dispatch (GenX, PLEXOS)',
        })

    # ── RETIREMENT_CASCADE ───────────────────────────────────────────────
    econ_retired_mw = year_result.get('total_economic_retirement_mw', 0)
    total_fossil_cap = sum(e.get('capacity_mw', 0) for e in gen_econ.values())
    if total_fossil_cap > 0:
        retired_pct = econ_retired_mw / total_fossil_cap * 100
        if retired_pct > 20:
            severity = 'high' if retired_pct > 35 else 'medium'
            triggers.append({
                'trigger_id': 'RETIREMENT_CASCADE',
                'severity': severity,
                'explanation': (
                    "Large-scale fossil retirement is occurring. Binary plant-level "
                    "retirement decisions, reliability-must-run contracts, and "
                    "regulatory backstop interventions would significantly alter "
                    "this trajectory. Plant-level modeling (EIA 860 fleet) is "
                    "recommended."
                ),
                'metric_value': round(retired_pct, 1),
                'threshold': 35.0 if severity == 'high' else 20.0,
                'recommended_model': 'Plant-level retirement model (EIA 860, IPM)',
            })

    # ── NUCLEAR_AT_RISK ──────────────────────────────────────────────────
    nuc_rev = year_result.get('nuclear_revenue', {})
    nuc_rev_mwh = nuc_rev.get('total_mwh', 0) if isinstance(nuc_rev, dict) else 0
    nuc_threshold = nuclear_retirement_threshold or _NUCLEAR_RETIREMENT_THRESHOLD_DEFAULT
    if (not state.get('nuclear_retired', False) and
            nuc_rev_mwh > 0 and
            abs(nuc_rev_mwh - nuc_threshold) <= 5.0):
        triggers.append({
            'trigger_id': 'NUCLEAR_AT_RISK',
            'severity': 'high',
            'explanation': (
                "Nuclear plant revenue is near the retirement cliff. Small changes "
                "in LMP assumptions could flip the retirement decision. Detailed "
                "plant-level economics with contract-specific data is recommended "
                "before acting on this result."
            ),
            'metric_value': round(nuc_rev_mwh, 2),
            'threshold': nuc_threshold,
            'recommended_model': 'Plant-level nuclear economics (contract-specific)',
        })

    return triggers


def _compute_zone_capture_adjustments(iso, system_vre_penetration, zonal_stats):
    """Compute zone-aware capture rate adjustments for VRE resources.

    When zonal LMP data is available, solar/wind in high-curtailment zones
    face worse capture rates than the system average, while resources in
    load-center zones may fare better.

    Args:
        iso: ISO region string
        system_vre_penetration: System-wide VRE penetration fraction
        zonal_stats: Dict from _compute_zonal_stats() — keys are zone names
            with sub-dict containing 'avg_lmp', plus '_congestion' key.

    Returns dict {resource: adjustment_factor} where factor > 1.0 means
    the zone-level capture is better than system average, < 1.0 means worse.
    Factor of 1.0 = no adjustment (fallback).
    """
    if not zonal_stats:
        return {}

    # Extract zone-level avg LMPs from zonal_stats structure
    # zonal_stats has zone names as keys (e.g., 'SP15', 'NP15') with
    # sub-dicts containing 'avg_lmp'. Skip special keys like '_congestion'.
    zone_avg_lmps = {}
    for key, val in zonal_stats.items():
        if isinstance(val, dict) and 'avg_lmp' in val:
            zone_avg_lmps[key] = val['avg_lmp']

    if not zone_avg_lmps:
        return {}

    # Compute system average from zone LMPs
    system_avg_lmp = sum(zone_avg_lmps.values()) / len(zone_avg_lmps)
    if system_avg_lmp <= 0:
        return {}

    vre_zones = VRE_PRIMARY_ZONE.get(iso, {})
    adjustments = {}

    for vre_res, canonical in [('solar', 'solar'), ('wind', 'wind'),
                                ('offshore_wind', 'offshore_wind')]:
        zone_name = vre_zones.get(canonical)
        if zone_name is None and vre_res == 'offshore_wind':
            # Fallback: use wind zone for offshore_wind if no dedicated zone
            zone_name = vre_zones.get('wind')
        if zone_name and zone_name in zone_avg_lmps:
            zone_lmp = zone_avg_lmps[zone_name]
            # Ratio of zone LMP to system LMP — zones with lower average
            # LMP (due to VRE surplus) get a capture penalty
            adjustments[vre_res] = zone_lmp / system_avg_lmp
        # else: no adjustment (factor defaults to 1.0 in caller)

    return adjustments


def compute_storage_deployment(iso, year, hourly_lmp, demand_twh,
                                demand_total_mwh, current_clean_pct,
                                conditions, cumulative_gw, state):
    """Economics-driven storage deployment via revenue vs. cost comparison.

    For each storage technology, computes total revenue (arbitrage + capacity +
    ancillary) and compares against annualized LCOE. Deploys capacity where
    revenue exceeds cost, subject to STORAGE_MAX caps.

    Args:
        iso: ISO region string
        year: Simulation year
        hourly_lmp: 8760 array of hourly LMP ($/MWh)
        demand_twh: Total demand in TWh
        demand_total_mwh: Total demand in MWh
        current_clean_pct: Current clean energy percentage
        conditions: Dict with lcoe_level, fuel_level, etc.
        cumulative_gw: Dict tracking cumulative GW deployed per tech
        state: Mutable iso_state dict

    Returns:
        dict with:
            deployed_pcts: {tech: pct_of_demand} for battery, battery8, ldes, h2
            storage_details: {tech: {revenue, cost, margin, deployed}} per tech
            total_storage_cost_mwh: Blended storage cost in $/MWh of total demand
    """
    lcoe_level = conditions.get('lcoe_level', 'Medium')
    learning_speed = conditions.get('learning_speed', 'Medium')

    # Compute LMP-based arbitrage revenue for each tech
    arb_revenue = compute_storage_arbitrage_from_lmp(hourly_lmp, iso)

    # Capacity market revenue with degradation at high clean shares
    base_cap_price = CAPACITY_MARKET_PRICES.get(iso, 0)
    cap_degrade = compute_capacity_degradation(iso, current_clean_pct)
    degraded_cap_price = base_cap_price * cap_degrade

    deployed_pcts = {}
    storage_details = {}
    total_storage_cost = 0.0

    for tech in ['battery', 'battery8', 'ldes', 'h2']:
        # H2 only at ≥95% thresholds
        if tech == 'h2' and current_clean_pct < H2_MIN_THRESHOLD:
            deployed_pcts[tech] = 0.0
            storage_details[tech] = {
                'revenue_kw_yr': 0, 'arb_kw_yr': 0, 'cap_kw_yr': 0,
                'anc_kw_yr': 0, 'revenue_lcoe': 0, 'cost_lcoe': 0,
                'margin': 0, 'deployed_pct': 0, 'reason': 'below_threshold',
            }
            continue

        params = _STORAGE_TECH_PARAMS[tech]
        duration = params['duration']

        # --- Revenue stack ($/kW-yr) ---
        # 1. Arbitrage revenue from LMP dispatch
        arb_kw_yr = arb_revenue.get(tech, 0)

        # 2. Capacity revenue (availability-based, always earned)
        # Storage gets full capacity credit (dispatchable resource)
        cap_kw_yr = degraded_cap_price

        # 3. Ancillary services revenue
        product = STORAGE_ANCILLARY_PRODUCT[tech]
        anc_rate = ANCILLARY_SERVICE_RATES[product].get(iso, 0)
        anc_hours = ANCILLARY_HOURS[product]
        anc_kw_yr = anc_rate * anc_hours / 1000.0  # $/MW-hr × hr/yr → $/kW-yr

        # Capacity is always earned; arbitrage + ancillary compete for same hours
        total_rev_kw_yr = cap_kw_yr + (arb_kw_yr + anc_kw_yr) * REVENUE_STACKING_FACTOR

        # Convert to LCOE-comparable units (same as LCOE_TABLES storage entries)
        # LCOE_TABLES storage: annualized cost per % of annual demand
        # Revenue credit: 1000 × $/kW-yr ÷ duration_hours
        revenue_lcoe = 1000.0 * total_rev_kw_yr / duration if duration > 0 else 0

        # --- Cost (annualized LCOE from LCOE_TABLES) ---
        cost_lcoe = get_resource_lcoe(tech, iso, lcoe_level, cumulative_gw,
                                       learning_speed, year, conditions=conditions)

        # --- Deploy decision ---
        margin = revenue_lcoe - cost_lcoe
        max_pct = STORAGE_MAX.get(tech, 0)

        if margin > 0:
            # Profitable — deploy to max cap
            deploy_pct = max_pct
        else:
            deploy_pct = 0.0

        deployed_pcts[tech] = round(deploy_pct, 4)
        storage_details[tech] = {
            'revenue_kw_yr': round(total_rev_kw_yr, 1),
            'arb_kw_yr': round(arb_kw_yr, 1),
            'cap_kw_yr': round(cap_kw_yr, 1),
            'anc_kw_yr': round(anc_kw_yr, 1),
            'revenue_lcoe': round(revenue_lcoe, 1),
            'cost_lcoe': round(cost_lcoe, 1),
            'margin': round(margin, 1),
            'deployed_pct': round(deploy_pct, 4),
        }

        # Storage cost contribution to total demand
        if deploy_pct > 0:
            # Cost in $/MWh of demand: cost_lcoe × deploy_pct / 100
            total_storage_cost += cost_lcoe * deploy_pct / 100.0

    state['storage_deployed'] = deployed_pcts
    state['storage_details'] = storage_details

    return {
        'deployed_pcts': deployed_pcts,
        'storage_details': storage_details,
        'total_storage_cost_mwh': round(total_storage_cost, 2),
    }


def compute_market_deployment(iso, year, demand_twh, current_clean_pct,
                               conditions, cumulative_gw, queue_remaining_gw,
                               hourly_lmp, avg_lmp, p90_lmp,
                               supply_profiles_iso, demand_total_mwh,
                               gen_econ, state, tech_queue_budget=None,
                               per_resource_energy_rev=None,
                               scarcity_hours_fraction=0.0,
                               zonal_stats=None,
                               zonal_lmp_matrix=None,
                               zonal_zone_names=None,
                               reserve_margin_pct=None,
                               curtailment_rate=0.0):
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
            (used as total cap when tech_queue_budget is None)
        hourly_lmp: 8760 array of hourly LMP values
        avg_lmp: Average LMP ($/MWh)
        p90_lmp: P90 LMP
        supply_profiles_iso: Generation profile dict for this ISO
        demand_total_mwh: Total demand in MWh
        gen_econ: Generator economics dict from LMP calculation
        state: Mutable iso_state dict
        tech_queue_budget: Optional dict {resource: remaining_gw} for per-tech
            queue caps. When provided, each resource is constrained by its own
            tech-specific budget plus a shared flex pool.

    Returns:
        (new_clean_pct, deployed_resources, zone_results, rev_breakdown,
         blended_cost, blended_revenue, remaining_gw)
        where deployed_resources is {resource: twh_deployed}
    """
    lcoe_level = conditions.get('lcoe_level', 'Medium')
    fuel_level = conditions.get('fuel_level', 'Medium')
    tx_level = conditions.get('tx_level', 'Medium')
    learning_speed = conditions.get('learning_speed', 'Medium')
    ppa_level = conditions.get('ppa_level', 'Medium')
    tx_overrides = conditions.get('tx_overrides') or {}

    # Per-resource energy revenue (temporal value) or fallback to flat avg_lmp
    per_res_rev = dict(per_resource_energy_rev) if per_resource_energy_rev else {}

    # Compute per-resource basis differentials (zone LMP − system LMP)
    _basis_diffs = {}
    if zonal_lmp_matrix is not None and zonal_zone_names is not None:
        _zone_idx = {z: i for i, z in enumerate(zonal_zone_names)}
        _vre_zone_map = VRE_PRIMARY_ZONE.get(iso, {})
        _sys_avg = float(np.mean(hourly_lmp))
        for res_key, zone_name in _vre_zone_map.items():
            z_idx = _zone_idx.get(zone_name)
            if z_idx is not None:
                _basis_diffs[res_key] = round(
                    float(np.mean(zonal_lmp_matrix[z_idx])) - _sys_avg, 2)

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

        # Add transmission cost (per-resource override takes priority over master L/M/H)
        if res in ('solar', 'wind', 'clean_firm', 'offshore_wind', 'ccs_ccgt', 'geothermal'):
            res_key = res if res != 'clean_firm' else 'nuclear'
            tx_override = tx_overrides.get(res_key)
            if tx_override is not None:
                tx = tx_override
            else:
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

        # R10: Curtailment feedback — reduce effective CF for VRE resources
        # based on system curtailment rate.  As curtailment rises, marginal
        # VRE produces less usable energy → effective LCOE increases →
        # deployment naturally saturates.
        effective_lcoe = lcoe
        if res in ('solar', 'wind', 'offshore_wind') and curtailment_rate > 0:
            effective_cf = cf * (1.0 - curtailment_rate)
            if effective_cf > 0:
                # Scale LCOE inversely with usable fraction: same capex
                # spread over fewer productive MWh.
                effective_lcoe = lcoe / (1.0 - curtailment_rate)
            else:
                effective_lcoe = float('inf')
            cf = effective_cf  # Use derated CF for capacity calcs too

        # Resource-specific revenue adjustments
        capacity_rev = 0
        rec_rev = 0
        # Endogenous capacity price: scarcity (reserve margin) × clean degradation
        _rm = reserve_margin_pct if reserve_margin_pct is not None else 100.0
        if conditions.get('capacity_market_price') is not None:
            base_cap_price = conditions['capacity_market_price']
        else:
            base_cap_price = compute_capacity_price(iso, _rm, current_clean_pct)
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

        # Use temporal revenue for this resource; fallback to avg_lmp
        res_energy_rev = per_res_rev.get(res, avg_lmp)
        total_revenue = res_energy_rev + capacity_rev + rec_rev
        net_profit = total_revenue - effective_lcoe

        # Max TWh deployable for this resource (physical limits)
        max_twh = RESOURCE_CAP_TWH.get(res, {}).get(iso, 999)
        # Already deployed — subtract from cap
        already_deployed_gw = cumulative_gw.get(RESOURCE_TO_TECH.get(res, res), 0)
        already_deployed_twh = already_deployed_gw * cf * 8.760 if cf > 0 else 0

        resource_economics.append({
            'resource': res,
            'lcoe': round(effective_lcoe, 2),
            'lcoe_base': round(lcoe, 2),
            'energy_rev': round(res_energy_rev, 2),
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

    # Tech-differentiated queue: compute flex pool from total budget
    use_tech_queue = tech_queue_budget is not None
    flex_pool_gw = 0.0
    if use_tech_queue:
        total_tech_budget = sum(tech_queue_budget.values())
        flex_pool_gw = total_tech_budget * QUEUE_FLEX_FRACTION
        # Reduce each tech's dedicated budget proportionally to fund the flex pool
        tech_budget = {res: cap * (1.0 - QUEUE_FLEX_FRACTION)
                       for res, cap in tech_queue_budget.items()}
    else:
        tech_budget = None

    for entry in resource_economics:
        if use_tech_queue:
            # Check if any budget remains (dedicated + flex)
            res_budget = tech_budget.get(entry['resource'], 0)
            if res_budget <= 0 and flex_pool_gw <= 0:
                continue
        else:
            if remaining_gw <= 0:
                break
        if entry['profit'] <= 0:
            continue  # Not profitable
        if entry['max_twh'] <= 0:
            continue  # Resource cap reached

        res = entry['resource']
        cf = entry['cf']

        # How much can we deploy given queue cap?
        if use_tech_queue:
            # Dedicated budget for this tech + flex pool overflow
            res_dedicated = max(0, tech_budget.get(res, 0))
            max_deploy_gw = res_dedicated + flex_pool_gw
        else:
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

        if use_tech_queue:
            # Deduct from dedicated budget first, overflow to flex pool
            res_dedicated = max(0, tech_budget.get(res, 0))
            if deploy_gw <= res_dedicated:
                tech_budget[res] = res_dedicated - deploy_gw
            else:
                # Exhaust dedicated, draw remainder from flex pool
                flex_draw = deploy_gw - res_dedicated
                tech_budget[res] = 0
                flex_pool_gw = max(0, flex_pool_gw - flex_draw)
            # Also decrement the total remaining_gw for backward-compat return
            remaining_gw -= deploy_gw
        else:
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
            'energy_rev_mwh': round(entry['energy_rev'], 2),
            'capture_rate': round(entry['energy_rev'] / avg_lmp, 3) if avg_lmp > 0 else 1.0,
            'capacity_rev_mwh': round(entry['capacity_rev'], 2),
            'rec_rev_mwh': round(entry['rec_rev'], 2),
            'basis_differential': _basis_diffs.get(res, 0.0),
        })

        # Intra-deployment cannibalization: depress VRE energy revenue for
        # subsequent tranches as solar/wind penetration increases.
        # ORDC-aware: scarcity hours create a floor on revenue depression.
        # Zone-aware: use zone-level VRE penetration when zonal data available.
        if per_resource_energy_rev is not None and res in ('solar', 'wind', 'offshore_wind'):
            cumulative_vre_twh = sum(
                deployed.get(r, 0) for r in ('solar', 'wind', 'offshore_wind')
            )
            vre_penetration = cumulative_vre_twh / demand_twh if demand_twh > 0 else 0
            # Sigmoid depression matching procurement_utils.py pattern
            depression = 0.55 * (1.0 / (1.0 + np.exp(-8.0 * (vre_penetration - 0.6))))

            # ORDC floor: scarcity hours keep prices high even at high VRE.
            # If 10% of hours have ORDC > $50, max depression capped at 97%.
            if scarcity_hours_fraction > 0 and SCARCITY_MODE == 'ordc':
                scarcity_floor = scarcity_hours_fraction * 0.3
                depression = min(depression, 1.0 - scarcity_floor)

            # Zone-aware: adjust per-resource depression by zone-level penetration
            # when zonal LMP data is available
            _zone_capture = _compute_zone_capture_adjustments(
                iso, vre_penetration, zonal_stats) if zonal_stats else {}

            for vre_res in ('solar', 'wind', 'offshore_wind'):
                base_rev = per_resource_energy_rev.get(vre_res, avg_lmp)
                zone_adj = _zone_capture.get(vre_res, 1.0)
                per_res_rev[vre_res] = base_rev * (1.0 - depression) * zone_adj

    # Build revenue breakdown from deployed mix
    total_energy = 0
    total_cap = 0
    total_rec = 0
    if total_deployed_twh > 0:
        for entry in resource_economics:
            twh = deployed.get(entry['resource'], 0)
            if twh > 0:
                weight = twh / total_deployed_twh
                total_energy += entry.get('energy_rev', avg_lmp) * weight
                total_cap += entry['capacity_rev'] * weight
                total_rec += entry['rec_rev'] * weight

    rev_breakdown = {
        'energy_rev_mwh': round(total_energy, 2),
        'capacity_rev_mwh': round(total_cap, 2),
        'rec_rev_mwh': round(total_rec, 2),
        'basis_differentials': _basis_diffs,
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

    # Compute capture rates for deployed resources
    capture_rates = {}
    energy_rev_by_res = {}
    for res in deployed:
        rev = per_res_rev.get(res, avg_lmp)
        energy_rev_by_res[res] = round(rev, 2)
        capture_rates[res] = round(rev / avg_lmp, 3) if avg_lmp > 0 else 1.0

    return (
        round(clean_pct, 2),
        deployed,
        zone_results,
        rev_breakdown,
        round(blended_cost, 2),
        round(blended_revenue, 2),
        remaining_gw,
        energy_rev_by_res,
        capture_rates,
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

    # If no parquet data found, handle based on SYNTHETIC_DATA_MODE
    if not all_data:
        missing_isos = list(ISOS)

        if SYNTHETIC_DATA_MODE == "error":
            raise RuntimeError(
                f"SYNTHETIC_DATA_MODE='error' but no step2.2 parquets found. "
                f"Missing ISOs: {missing_isos}. Run the full pipeline (Steps 1-2) "
                f"or set SYNTHETIC_DATA_MODE='warn' to use synthetic fallback."
            )
        elif SYNTHETIC_DATA_MODE == "warn":
            logger.warning(
                "No step2.2 parquets found for ISOs: %s. "
                "Generating synthetic threshold data. Results are ILLUSTRATIVE ONLY.",
                missing_isos,
            )
        else:  # "silent"
            logger.info("No step2.2 parquets found — generating synthetic threshold data")

        all_data = _generate_synthetic_step3_data()

    return all_data


def check_data_sources():
    """Check which ISOs have real parquet data vs. requiring synthetic fallback.

    Returns dict with two keys:
      - 'simple': {iso: 'parquet' | 'synthetic'} — backward-compatible single-tier
      - 'tiers':  {iso: {resource_mix, zonal_config, interchange, fleet_data, dr_params}}
    """
    from pipeline_config import ZONE_CONFIG, DEMAND_RESPONSE

    search_dirs = [os.path.join(MODULE_ROOT, 'data', 'step2.2-cost')]
    simple = {}
    tiers = {}

    # Check interchange data availability (shared across ISOs)
    interchange_file = os.path.join(MODULE_ROOT, 'data', 'profiles',
                                    'eia_interchange_profiles.json')
    has_interchange = os.path.isfile(interchange_file)

    # Check EIA-860 plant-level data
    eia860_dir = os.path.join(MODULE_ROOT, 'data', 'eia-860')
    has_eia860 = os.path.isdir(eia860_dir) and bool(os.listdir(eia860_dir))

    for iso in ISOS:
        # Tier 1: resource mix (parquet vs synthetic)
        found_parquet = False
        for d in search_dirs:
            if not os.path.isdir(d):
                continue
            for pattern in [f'step_2_2a_CO_{iso}.parquet',
                            f'step3_co_{iso}.parquet']:
                if os.path.exists(os.path.join(d, pattern)):
                    found_parquet = True
                    break
            if found_parquet:
                break
        resource_mix = 'parquet' if found_parquet else 'synthetic'
        simple[iso] = resource_mix

        # Tier 2: zonal configuration
        zonal_config = 'validated' if iso in ZONE_CONFIG else 'hardcoded'

        # Tier 3: interchange
        interchange = 'eia_930' if has_interchange else 'none'

        # Tier 4: fleet data
        fleet_data = 'plant_level' if has_eia860 else 'aggregated'

        # Tier 5: demand response parameters
        dr_params = 'calibrated' if iso in DEMAND_RESPONSE else 'default'

        tiers[iso] = {
            'resource_mix': resource_mix,
            'zonal_config': zonal_config,
            'interchange': interchange,
            'fleet_data': fleet_data,
            'dr_params': dr_params,
        }

    return {'simple': simple, 'tiers': tiers}


def _build_data_quality(iso: str, data_sources: dict) -> dict:
    """Build structured data_quality metadata for a given ISO.

    Returns dict with:
      - synthetic_backed: True if resource mix uses synthetic fallback
      - missing_sources: list of missing data source identifiers
      - mode: current SYNTHETIC_DATA_MODE value
    """
    ds_simple = data_sources.get('simple', data_sources)
    ds_tiers = data_sources.get('tiers', {}).get(iso, {})
    is_synthetic = ds_simple.get(iso, 'synthetic') == 'synthetic'

    missing = []
    if is_synthetic:
        missing.append('step2.2_parquet')
    if ds_tiers.get('interchange') == 'none':
        missing.append('eia_interchange')
    if ds_tiers.get('fleet_data') == 'aggregated':
        missing.append('eia_860_plant_data')

    return {
        'synthetic_backed': is_synthetic,
        'missing_sources': missing,
        'mode': SYNTHETIC_DATA_MODE,
    }


def _generate_synthetic_step3_data():
    """Generate synthetic resource mix data for each ISO when parquets are absent.

    Produces a reasonable set of threshold → resource_mix mappings based on
    known grid characteristics. This enables the tool to run standalone for
    screening-level analysis without requiring the full pipeline.

    R1: Now emits an explicit warning instead of silently generating ramp data.
    """
    import warnings
    warnings.warn(
        "Parquet data not found — using synthetic storage ramps. "
        "Results are approximate. Run the full pipeline (Steps 1-2) "
        "for production-quality results.",
        UserWarning, stacklevel=2,
    )
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


NEW_FOSSIL_COST_LEVELS = ['Low', 'Medium', 'High']


def build_market_scenarios():
    """Generate all parametric sweep scenarios.

    3 demand × 5 price × 3 PPA × 3 gas friction × 3 queue × 3 new fossil cost
    = 1,215 scenarios.
    No emission constraints, no NZ targets — purely market-driven reference trajectory.

    Returns list of (scenario_id_str, conditions_dict).
    """
    combos = []
    demand_keys = DEMAND_GROWTH_LEVELS
    price_keys = list(PRICE_SENSITIVITIES.keys())
    ppa_keys = PPA_LEVELS
    gas_keys = list(GAS_FRICTION_LEVELS.keys())
    queue_keys = ['Low', 'Medium', 'High']
    nfc_keys = NEW_FOSSIL_COST_LEVELS

    for demand, price_name, ppa, gas_name, queue, nfc in cartesian(
            demand_keys, price_keys, ppa_keys, gas_keys, queue_keys, nfc_keys):

        price_mapping = _map_price_sens_to_lcoe_fuel_tx(PRICE_SENSITIVITIES[price_name])

        demand_code = demand[0]
        ppa_code = ppa[0]
        gas_code = gas_name[0]
        queue_code = queue[0]
        nfc_code = nfc[0]
        scenario_id = f"MKT_{demand_code}_{price_name}_{ppa_code}_{gas_code}_{queue_code}_{nfc_code}"

        conditions = {
            'name': (f"Market: {demand} demand | {price_name} | PPA={ppa} | "
                     f"Gas={gas_name} | Queue={queue} | NewFossil={nfc}"),
            'demand_growth': demand,
            'lcoe_level': price_mapping['lcoe_level'],
            'learning_speed': QUEUE_LEARNING_MAP.get(queue, 'Medium'),
            'queue_cap_level': queue,
            'gas_friction': GAS_FRICTION_LEVELS[gas_name],
            'carbon_price': 0,
            'fuel_level': price_mapping['fuel_level'],
            'tx_level': price_mapping['tx_level'],
            'ppa_level': ppa,
            'new_fossil_cost_level': nfc,
            '_price_sens_name': price_name,
            '_price_sens': price_mapping.get('_price_sens', {}),
        }

        combos.append((scenario_id, conditions))

    return combos


def build_single_scenario(overrides=None):
    """Build a single scenario with user overrides.

    New-build fossil parameters (all optional, override via 'overrides' dict):
        new_fossil_cost_level: 'Low'/'Medium'/'High' — selects CAPEX table
        new_fossil_capex_override: {type: $/kW-yr} — per-type CAPEX override
        new_fossil_min_cf_override: {type: fraction} — per-type min CF override
        new_fossil_enabled: bool — disable new fossil builds entirely (default True)

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
        'dr_level': 'Off',
        'new_fossil_cost_level': 'Medium',
        'new_fossil_enabled': True,
        # User-overridable per-type: new_fossil_capex_override, new_fossil_min_cf_override
    }
    if overrides:
        defaults.update(overrides)
    return ('MKT_CUSTOM', defaults)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_demand_at_year(iso, year, growth_level):
    """Project demand TWh at a future year from 2023 baseline.

    growth_level can be 'Low'/'Medium'/'High' (looks up DEMAND_GROWTH_RATES)
    or a numeric value (interpreted as annual % growth rate, capped at 7.5%).
    """
    base = REGIONAL_DEMAND_TWH[iso]
    if isinstance(growth_level, (int, float)):
        # Numeric growth rate — interpret as percentage, cap at 7.5%
        rate = min(float(growth_level) / 100.0, 0.075)
    else:
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
# PROVENANCE METADATA
# ═══════════════════════════════════════════════════════════════════════════════

_PIPELINE_CONFIG_PATH = os.path.join(SCRIPT_DIR, 'pipeline_config.py')


def build_provenance_metadata(input_params: dict) -> ProvenanceMetadata:
    """Build provenance metadata capturing code version, config, and inputs.

    Args:
        input_params: The input request parameters (scenario conditions, ISOs, etc.)

    Returns:
        ProvenanceMetadata instance with git SHA, branch, config hash, etc.
    """
    # Git SHA (short)
    try:
        git_sha = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=SCRIPT_DIR, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        git_sha = 'unknown'

    # Git branch
    try:
        git_branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            cwd=SCRIPT_DIR, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        git_branch = 'unknown'

    # SHA-256 of pipeline_config.py
    try:
        with open(_PIPELINE_CONFIG_PATH, 'rb') as f:
            config_hash = hashlib.sha256(f.read()).hexdigest()
    except Exception:
        config_hash = 'unknown'

    return ProvenanceMetadata(
        model_version=PIPELINE_VERSION,
        git_sha=git_sha,
        git_branch=git_branch,
        config_hash=config_hash,
        run_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        python_version=sys.version,
        input_snapshot=input_params,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CORE MARKET SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_market_simulation(scenario_id, conditions, isos=None,
                           nuclear_retirement_threshold=None,
                           snapshot_mode=False,
                           sim_years=None,
                           _preloaded=None, _lmp_cache=None, _quiet=False,
                           weather_year=None,
                           _data_sources=None):
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

    # Build provenance metadata
    provenance = build_provenance_metadata({
        'scenario_id': scenario_id,
        'conditions': conditions,
        'isos': isos,
        'nuclear_retirement_threshold': nuclear_retirement_threshold,
        'snapshot_mode': snapshot_mode,
        'sim_years': sim_years,
        'weather_year': weather_year,
    })

    # Load data
    if _preloaded is not None:
        demand_data = _preloaded['demand_data']
        gen_profiles = _preloaded['gen_profiles']
        emission_rates = _preloaded['emission_rates']
        fossil_mix = _preloaded['fossil_mix']
        egrid_baselines = _preloaded['egrid_baselines']
        interchange_data = _preloaded.get('interchange_data', {})
    else:
        t0 = time.time()
        _log("Loading common data...")
        demand_data, gen_profiles, emission_rates, fossil_mix = load_common_data()
        _log(f"  Common data loaded in {time.time()-t0:.1f}s")
        egrid_baselines = load_egrid_baselines()
        # Load interchange profiles (empty dict if unavailable → copper-plate fallback)
        try:
            from eia_data_io import load_interchange_profiles
            interchange_data = load_interchange_profiles()
        except Exception:
            interchange_data = {}

    # Interchange enabled flag from conditions (default: True)
    interchange_enabled = conditions.get('interchange_enabled', True)

    # Determine data source per ISO (parquet vs synthetic)
    if _data_sources is None:
        _full_data_sources = check_data_sources()
        _data_sources = _full_data_sources.get('simple', _full_data_sources) if isinstance(_full_data_sources, dict) and 'simple' in _full_data_sources else _full_data_sources
    else:
        _full_data_sources = {'simple': _data_sources}

    cumulative_gw = dict(WRIGHT_CUMULATIVE_GW_2025)
    # Endogenous learning: allow conditions to override the global toggle
    _endogenous = conditions.get('endogenous_learning', ENDOGENOUS_LEARNING)
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
            'fossil_built_gw': 0,
            'new_fossil_builds': {},  # cumulative MW built by unit type
            'nuclear_retired': False,
            'acp_bonus_queue_gw': 0,
            'cumulative_acp_million': 0,
            'economic_retirements': {},  # cumulative MW retired by unit type
            'retired_plants': [],  # G1: plant IDs retired across years (inter-year persistence)
            'deployed_twh': {},  # cumulative TWh deployed by resource (for dispatch consistency)
            'storage_deployed': {},  # R1: {tech: pct_of_demand} from economics-driven deployment
            'storage_details': {},   # R1: per-tech revenue/cost breakdown
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
                    'fossil_built_gw': 0,
                    'total_gas_gw': 0,
                    'market_stop': False,
                    'resource_mix_twh': {},
                    'cumulative_gw': dict(WRIGHT_CUMULATIVE_GW_2025),
                    'zones_deployed': [],
                    'generator_economics': {},
                    'nuclear_revenue': {},
                    'nuclear_retired': False,
                    'ccs_breakeven': {},
                    'new_fossil_builds_mw': {},
                    'total_new_fossil_mw': 0,
                    'new_fossil_details': {},
                    # RPS compliance tracking
                    'rps_mandated_pct': 0,
                    'rps_eligible_pct': round(iso_state[iso].get('rps_eligible_pct', 0), 1),
                    'rps_shortfall_pct': 0,
                    'acp_cost_million': 0,
                    'cumulative_acp_million': 0,
                    'data_source': _data_sources.get(iso, 'synthetic'),
                    'data_quality': _build_data_quality(iso, _full_data_sources),
                    # Baseline capacity pricing (assume healthy reserves in 2023)
                    'reserve_margin_pct': 20.0,
                    'capacity_price_kw_yr': round(CAPACITY_MARKET_PRICES.get(iso, 0), 2),
                }
                results[iso].append(year_result)
                _log(f"  {iso}: 2023 baseline — {baseline_co2_mt:.1f} Mt, "
                     f"{baseline_clean:.1f}% clean, LMP=${baseline_lmp:.0f}")
            continue

        # Static learning mode: freeze cumulative GW at 2025 baseline each year
        if not _endogenous:
            cumulative_gw = dict(WRIGHT_CUMULATIVE_GW_2025)

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

            # Build per-tech queue budget if tech-differentiated mode is active
            use_tech_queue = conditions.get('tech_differentiated_queue', TECH_DIFFERENTIATED_QUEUE)
            tech_queue_budget = None
            if use_tech_queue and queue_budget_gw == QUEUE_CAP_GW.get(queue_cap_level, {}).get(iso, 5):
                # Use per-tech caps (only when no manual override)
                iso_tech_caps = TECH_QUEUE_CAP_GW.get(queue_cap_level, {}).get(iso, {})
                tech_queue_budget = {res: cap * years_in_period
                                     for res, cap in iso_tech_caps.items()}

            # ACP recycling: prior-period ACP payments boost queue (capped at 20% of base)
            acp_bonus = min(state.get('acp_bonus_queue_gw', 0),
                           queue_budget_gw * years_in_period * 0.20)
            if acp_bonus > 0:
                queue_remaining_gw += acp_bonus
                # When tech-differentiated, distribute ACP bonus proportionally
                if tech_queue_budget is not None:
                    total_tech = sum(tech_queue_budget.values())
                    if total_tech > 0:
                        for res in tech_queue_budget:
                            tech_queue_budget[res] += acp_bonus * (tech_queue_budget[res] / total_tech)
                _log(f"  {iso} ACP recycling: +{acp_bonus:.2f} GW bonus queue")
                state['acp_bonus_queue_gw'] = 0

            # --- LMP + GENERATOR ECONOMICS at current clean% ---
            carbon_price = conditions.get('carbon_price', 0)
            # Build resource_pcts from baseline + cumulative deployed resources.
            # resource_pcts must represent the ACTUAL share of total demand each
            # resource serves, so the dispatch model's residual demand curve
            # is consistent with the tracked clean_pct.
            resource_pcts = {r: 0 for r in RESOURCE_TYPES}
            # Start with baseline grid mix (% of total demand) — includes hydro
            for r, pct in GRID_MIX_SHARES.get(iso, {}).items():
                if r in resource_pcts:
                    resource_pcts[r] = pct
            # Add cumulative deployed resources (converted from TWh to % of current demand)
            for r, twh in state.get('deployed_twh', {}).items():
                if r in resource_pcts and demand_twh > 0:
                    resource_pcts[r] += (twh / demand_twh) * 100.0

            # --- Inter-regional interchange ---
            # Retrieve hourly net import profile for this ISO (normalized units).
            # In trajectory mode, scale by demand growth but cap at firm import MW.
            iso_ic_data = interchange_data.get(iso, {}).get('2024', {})
            if interchange_enabled and iso_ic_data.get('net_import_norm'):
                ic_norm_base = np.array(iso_ic_data['net_import_norm'][:H], dtype=np.float64)
                # Scale imports by demand growth (more demand → more imports needed)
                ic_norm = ic_norm_base * growth_factor
                # Cap at FIRM_IMPORT_MW in normalized units
                firm_cap = FIRM_IMPORT_MW.get(iso, 0)
                if firm_cap > 0 and demand_total_mwh > 0:
                    cap_norm = firm_cap / demand_total_mwh
                    ic_norm = np.clip(ic_norm, -cap_norm, cap_norm)
                ic_firm_mw = FIRM_IMPORT_MW.get(iso, 0)
            else:
                ic_norm = None
                ic_firm_mw = 0

            dr_level = conditions.get('dr_level', 'Off')
            # Include new-build fossil total in cache key — fleet state affects LMP.
            # Round to nearest 5 GW to improve cache hit rate: scenarios with
            # similar fossil fleets (e.g., 42 GW vs 43 GW) produce nearly
            # identical LMPs, so bucketing avoids redundant dispatch solves.
            _nb_total = sum(state.get('new_fossil_builds', {}).values())
            _nb_bucket = round(_nb_total / 5000) * 5000  # 5 GW buckets
            # R1: Include storage state in LMP cache key
            _prev_stor = state.get('storage_deployed', {})
            _stor_key = tuple(sorted(_prev_stor.items())) if _prev_stor else ()
            _lmp_key = (iso, current_pct, conditions['fuel_level'],
                        conditions['demand_growth'], year, carbon_price,
                        interchange_enabled, dr_level, _nb_bucket, _stor_key)
            zonal_congestion_data = None
            scarcity_hours_frac = 0.0
            curtailment_rate = 0.0
            if _lmp_cache is not None and _lmp_key in _lmp_cache:
                hourly_lmp, avg_lmp, p90_lmp, gen_econ, dr_metrics, zonal_congestion_data, scarcity_hours_frac, _zonal_lmp_matrix, _zonal_zone_names, curtailment_rate = _lmp_cache[_lmp_key]
            else:
                # R1: Use previously deployed storage in LMP calculation (Pass 1)
                hourly_lmp, avg_lmp, p90_lmp, gen_econ, dr_metrics, zonal_congestion_data, scarcity_hours_frac, _zonal_lmp_matrix, _zonal_zone_names, curtailment_rate = compute_lmp_at_threshold(
                    iso, current_pct, conditions['fuel_level'],
                    demand_norm, demand_mw_profile,
                    supply_profiles_iso, resource_pcts,
                    battery_pct=_prev_stor.get('battery', 0),
                    battery8_pct=_prev_stor.get('battery8', 0),
                    ldes_pct=_prev_stor.get('ldes', 0),
                    h2_pct=_prev_stor.get('h2', 0),
                    carbon_price=carbon_price,
                    nox_price=conditions.get('nox_price', 0.0),
                    sox_price=conditions.get('sox_price', 0.0),
                    nox_limit=conditions.get('nox_limit'),
                    sox_limit=conditions.get('sox_limit'),
                    custom_fuel_prices=conditions.get('custom_fuel_prices'),
                    custom_co2_price=conditions.get('custom_co2_price'),
                    custom_heat_rates=conditions.get('custom_heat_rates'),
                    custom_vom=conditions.get('custom_vom'),
                    interchange_norm=ic_norm,
                    firm_import_mw=ic_firm_mw,
                    dr_level=dr_level,
                    demand_growth_factor=growth_factor,
                    new_fossil_builds=state.get('new_fossil_builds'),
                )
                if _lmp_cache is not None:
                    _lmp_cache[_lmp_key] = (hourly_lmp, avg_lmp, p90_lmp, gen_econ, dr_metrics, zonal_congestion_data, scarcity_hours_frac, _zonal_lmp_matrix, _zonal_zone_names, curtailment_rate)

            # --- RESERVE MARGIN (for endogenous capacity pricing) ---
            reserve_margin_pct = compute_reserve_margin(
                gen_econ, cumulative_gw, demand_twh)

            # --- R1: ECONOMICS-DRIVEN STORAGE DEPLOYMENT ---
            storage_result = compute_storage_deployment(
                iso, year, hourly_lmp, demand_twh, demand_total_mwh,
                current_pct, conditions, cumulative_gw, state)
            _new_stor = storage_result['deployed_pcts']

            # Check if storage allocation changed materially (>0.1pp on any tech)
            _storage_changed = any(
                abs(_new_stor.get(t, 0) - _prev_stor.get(t, 0)) > 0.001
                for t in ['battery', 'battery8', 'ldes', 'h2']
            )

            if _storage_changed:
                # LMP Pass 2: recompute with new storage deployment
                _stor_key2 = tuple(sorted(_new_stor.items()))
                _lmp_key2 = (iso, current_pct, conditions['fuel_level'],
                             conditions['demand_growth'], year, carbon_price,
                             interchange_enabled, dr_level, _nb_bucket, _stor_key2)
                if _lmp_cache is not None and _lmp_key2 in _lmp_cache:
                    hourly_lmp, avg_lmp, p90_lmp, gen_econ, dr_metrics, zonal_congestion_data, scarcity_hours_frac, _zonal_lmp_matrix, _zonal_zone_names, curtailment_rate = _lmp_cache[_lmp_key2]
                else:
                    hourly_lmp, avg_lmp, p90_lmp, gen_econ, dr_metrics, zonal_congestion_data, scarcity_hours_frac, _zonal_lmp_matrix, _zonal_zone_names, curtailment_rate = compute_lmp_at_threshold(
                        iso, current_pct, conditions['fuel_level'],
                        demand_norm, demand_mw_profile,
                        supply_profiles_iso, resource_pcts,
                        battery_pct=_new_stor.get('battery', 0),
                        battery8_pct=_new_stor.get('battery8', 0),
                        ldes_pct=_new_stor.get('ldes', 0),
                        h2_pct=_new_stor.get('h2', 0),
                        carbon_price=carbon_price,
                        nox_price=conditions.get('nox_price', 0.0),
                        sox_price=conditions.get('sox_price', 0.0),
                        nox_limit=conditions.get('nox_limit'),
                        sox_limit=conditions.get('sox_limit'),
                        custom_fuel_prices=conditions.get('custom_fuel_prices'),
                        custom_co2_price=conditions.get('custom_co2_price'),
                        custom_heat_rates=conditions.get('custom_heat_rates'),
                        custom_vom=conditions.get('custom_vom'),
                        interchange_norm=ic_norm,
                        firm_import_mw=ic_firm_mw,
                        dr_level=dr_level,
                        demand_growth_factor=growth_factor,
                        new_fossil_builds=state.get('new_fossil_builds'),
                    )
                    if _lmp_cache is not None:
                        _lmp_cache[_lmp_key2] = (hourly_lmp, avg_lmp, p90_lmp, gen_econ, dr_metrics, zonal_congestion_data, scarcity_hours_frac, _zonal_lmp_matrix, _zonal_zone_names, curtailment_rate)

                _log(f"  {iso} R1 storage: "
                     + ", ".join(f"{t}={_new_stor.get(t, 0):.4f}%"
                                for t in ['battery', 'battery8', 'ldes', 'h2']
                                if _new_stor.get(t, 0) > 0)
                     + f" (LMP pass 2: avg=${avg_lmp:.1f})")

            # --- LCOE MERIT-ORDER DEPLOYMENT ---
            # Compute per-resource temporal energy revenue for deployment economics
            if CANNIBALIZATION_ENABLED:
                _per_res_rev = compute_energy_revenue_by_resource(
                    hourly_lmp, supply_profiles_iso, resource_pcts, demand_total_mwh,
                    iso=iso, zonal_lmp_matrix=_zonal_lmp_matrix,
                    zonal_zone_names=_zonal_zone_names)
            else:
                _per_res_rev = None

            # Deploy cheapest profitable clean resources until queue cap or no more profitable
            (new_clean_pct, deployed, zone_results, rev_breakdown,
             blended_cost, blended_revenue, remaining_gw,
             energy_rev_by_resource, capture_rates) = compute_market_deployment(
                iso, year, demand_twh, current_pct,
                conditions, cumulative_gw, queue_remaining_gw,
                hourly_lmp, avg_lmp, p90_lmp,
                supply_profiles_iso, demand_total_mwh,
                gen_econ, state,
                tech_queue_budget=tech_queue_budget,
                per_resource_energy_rev=_per_res_rev,
                scarcity_hours_fraction=scarcity_hours_frac,
                zonal_stats=zonal_congestion_data,
                zonal_lmp_matrix=_zonal_lmp_matrix,
                zonal_zone_names=_zonal_zone_names,
                reserve_margin_pct=reserve_margin_pct,
                curtailment_rate=curtailment_rate,
            )

            # Sync queue budget after deployment
            queue_remaining_gw = remaining_gw

            # Update state
            old_pct = current_pct
            current_pct = new_clean_pct
            state['clean_pct'] = current_pct
            state['market_stopped'] = (current_pct <= old_pct + 0.01)

            # Track cumulative deployed TWh for dispatch consistency
            for res, twh in deployed.items():
                state['deployed_twh'][res] = state['deployed_twh'].get(res, 0) + twh

            # Track RPS-eligible deployment
            for res, twh in deployed.items():
                if res in REC_ELIGIBLE:
                    rps_delta_pct = twh / demand_twh * 100
                    state['rps_eligible_pct'] += rps_delta_pct

            _log(f"  {iso} → {current_pct:.1f}% clean (was {old_pct:.1f}%): "
                 f"LMP avg=${avg_lmp:.1f}, deployed {sum(deployed.values()):.1f} TWh "
                 f"across {len(deployed)} resources")

            # Nuclear revenue at current threshold
            nuclear_rev = compute_nuclear_revenue(iso, current_pct, hourly_lmp, year,
                                                   conditions=conditions,
                                                   reserve_margin_pct=reserve_margin_pct)

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

            # --- ECONOMIC RETIREMENT (G1: Plant-Level) ---
            # Build plant-level merit order and compute per-plant economics.
            # Retire individual plants by margin (worst-first) with zonal reliability floor.
            plant_economics = None
            plant_retirement_list = []
            try:
                fuel_level = conditions.get('fuel_level', 'Medium')
                _fuel_prices = conditions.get('custom_fuel_prices') or FUEL_PRICES.get(fuel_level, FUEL_PRICES['Medium'])
                plant_stack, _plant_total_mw = build_plant_level_merit_order(
                    iso, current_pct, fuel_level=fuel_level,
                    carbon_price=carbon_price,
                    custom_fuel_prices=_fuel_prices,
                    custom_vom=conditions.get('custom_vom'),
                )
                if plant_stack:
                    # Filter out previously retired plants
                    prior_retired_ids = set(state.get('retired_plants', []))
                    plant_stack = [p for p in plant_stack
                                   if p.get('plant_id') not in prior_retired_ids]
                    # Reconstruct dispatch for plant-level economics
                    _stor = state.get('storage_deployed', {})
                    _dispatch = reconstruct_hourly_dispatch(
                        demand_norm, supply_profiles_iso, resource_pcts,
                        procurement_pct=100,
                        battery_dispatch_pct=_stor.get('battery', 0),
                        battery8_dispatch_pct=_stor.get('battery8', 0),
                        ldes_dispatch_pct=_stor.get('ldes', 0),
                        h2_dispatch_pct=_stor.get('h2', 0),
                    )
                    plant_economics = compute_plant_level_economics(
                        plant_stack, hourly_lmp, _dispatch,
                        demand_mw_profile, _fuel_prices, carbon_price,
                        year=year,
                    )
            except Exception as _pe:
                _log(f"    {iso} plant-level economics unavailable ({_pe}), "
                     f"falling back to fleet-fraction retirement")
                plant_economics = None

            # Pass cumulative_gw to state so reliability floor can account for clean capacity
            state['cumulative_gw'] = cumulative_gw

            adjusted_gen_econ, econ_retired, econ_retired_mw, plant_retirement_list = apply_economic_retirement(
                gen_econ, iso, year, state, _log=_log,
                plant_economics=plant_economics, demand_twh=demand_twh)

            # --- ECONOMIC NEW-BUILD FOSSIL ---
            # After retirements, evaluate whether new fossil capacity should be
            # built based on RA needs and/or economic viability (positive margins).
            # Recompute reserve margin after retirements (before new-build decision)
            reserve_margin_pct_post_retire = compute_reserve_margin(
                adjusted_gen_econ, cumulative_gw, demand_twh)
            new_fossil_builds, new_fossil_mw, new_fossil_details = apply_economic_new_build(
                adjusted_gen_econ, iso, year, state, conditions,
                demand_twh, hourly_lmp, _log=_log,
                reserve_margin_pct=reserve_margin_pct_post_retire)

            # Add new-build capacity to adjusted gen_econ for emission accounting.
            # New units have better heat rates than fleet average, so they
            # enter at their own cost/emission characteristics.
            for ftype, new_mw in new_fossil_builds.items():
                fleet_key = ftype if ftype != 'coal' else 'coal_steam'
                if fleet_key in adjusted_gen_econ:
                    adjusted_gen_econ[fleet_key]['capacity_mw'] += new_mw
                else:
                    # New type entering the fleet
                    hr = NEW_BUILD_HEAT_RATES.get(ftype, HEAT_RATES.get(fleet_key, 7.0))
                    vom = NEW_BUILD_VOM.get(ftype, VOM.get(fleet_key, 3.5))
                    co2 = NEW_BUILD_CO2_RATES.get(ftype, CO2_RATES.get(fleet_key, 0.37))
                    adjusted_gen_econ[fleet_key] = {
                        'capacity_mw': new_mw,
                        'cf': 0.50,
                        'avg_rev_mwh': float(np.mean(hourly_lmp)),
                        'var_cost_mwh': hr * FUEL_PRICES.get(
                            conditions.get('fuel_level', 'Medium'), {}).get(
                            'gas' if 'gas' in ftype else 'coal', 3.5) + vom,
                        'margin_mwh': 0,
                        'dispatch_hours': 4380,
                    }

            # --- EMISSION ACCOUNTING ---
            gf = demand_twh / REGIONAL_DEMAND_TWH[iso]
            _, retirement_info = compute_fossil_retirement(
                iso, current_pct, emission_rates, fossil_mix, gf)
            # Use remaining fleet emission rate (not displaced rate) for actual emissions
            er = retirement_info.get('remaining_rate_tco2_mwh', 0)
            fossil_twh = (1 - current_pct / 100.0) * demand_twh

            # Adjust fossil generation downward for economic retirements.
            # Economically retired capacity can't generate — reduce fossil TWh
            # proportionally to retired fraction of total fossil fleet.
            total_fossil_cap = sum(e.get('capacity_mw', 0) for e in gen_econ.values())
            if total_fossil_cap > 0 and econ_retired_mw > 0:
                surviving_frac = max(0.05, 1.0 - econ_retired_mw / total_fossil_cap)
                fossil_twh *= surviving_frac
                _log(f"    {iso} fossil TWh adjusted: ×{surviving_frac:.2f} "
                     f"({econ_retired_mw:.0f} MW retired of {total_fossil_cap:.0f} MW)")

            # Add generation from new-build fossil capacity.
            # New builds generate at their expected CF with their own emission rate.
            new_build_twh = 0.0
            new_build_emissions_mt = 0.0
            for ftype, new_mw in new_fossil_builds.items():
                detail = new_fossil_details.get(ftype, {})
                nb_cf = detail.get('expected_cf', 0.30)
                nb_twh = new_mw * nb_cf * 8.760 / 1000.0  # MW → GW → TWh
                nb_co2_rate = NEW_BUILD_CO2_RATES.get(ftype, 0.37)
                nb_emissions = nb_twh * nb_co2_rate  # Mt CO2
                new_build_twh += nb_twh
                new_build_emissions_mt += nb_emissions

            emissions_mt = fossil_twh * 1e6 * er / 1e6 + new_build_emissions_mt

            # Per-fuel-type emissions breakdown (Mt CO2) derived from fossil
            # retirement model — consistent with the emissions_mt total.
            # Previously used gen_econ capacity factors which ran a separate
            # dispatch disconnected from clean_pct, producing phantom emissions.
            regional_er = emission_rates.get(iso, {})
            coal_co2_rate = regional_er.get('coal_co2_lb_per_mwh', 0.0) / 2204.62
            gas_co2_rate = regional_er.get('gas_co2_lb_per_mwh', 0.0) / 2204.62
            oil_co2_rate = regional_er.get('oil_co2_lb_per_mwh', 0.0) / 2204.62

            # Derive per-fuel remaining TWh from retirement model
            emissions_by_fuel = {}
            if retirement_info.get('remaining_rate_tco2_mwh', 0) == 0:
                # 100% clean or zero fossil — no emissions by fuel
                pass
            elif retirement_info.get('forced_gas_only'):
                # Coal/oil fully retired — all remaining fossil is gas
                emissions_by_fuel['gas_ccgt'] = round(fossil_twh * gas_co2_rate, 3)
            elif 'coal_remaining_twh' in retirement_info:
                # Merit-order path — use remaining TWh from retirement model,
                # scaled by economic retirement surviving fraction
                econ_scale = 1.0
                if total_fossil_cap > 0 and econ_retired_mw > 0:
                    econ_scale = max(0.05, 1.0 - econ_retired_mw / total_fossil_cap)
                coal_rem = retirement_info['coal_remaining_twh'] * econ_scale
                oil_rem = retirement_info['oil_remaining_twh'] * econ_scale
                gas_rem = retirement_info['gas_remaining_twh'] * econ_scale
                if coal_rem > 0:
                    emissions_by_fuel['coal'] = round(coal_rem * coal_co2_rate, 3)
                if oil_rem > 0:
                    emissions_by_fuel['oil'] = round(oil_rem * oil_co2_rate, 3)
                if gas_rem > 0:
                    emissions_by_fuel['gas_ccgt'] = round(gas_rem * gas_co2_rate, 3)
            else:
                # Fallback: use total emissions_mt with blended rate
                emissions_by_fuel['fossil'] = round(emissions_mt, 3)

            # Update TWh ratchet floor after all deployment
            current_rps_twh = state['rps_eligible_pct'] / 100.0 * demand_twh
            state['rps_eligible_twh_floor'] = max(state['rps_eligible_twh_floor'], current_rps_twh)

            # CCS breakeven at this ISO/fuel level
            ccs_breakeven = compute_ccs_retrofit_breakeven(iso, conditions['fuel_level'], conditions=conditions)

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
                'basis_differentials': rev_breakdown.get('basis_differentials', {}),
                'avg_lmp': round(avg_lmp, 1),
                'lmp_p90': round(p90_lmp, 1),
                'gas_built_gw': round(state.get('gas_built_gw', 0), 2),
                'fossil_built_gw': round(state.get('fossil_built_gw', 0), 2),
                'total_gas_gw': round(
                    sum(e.get('capacity_mw', 0) for k, e in adjusted_gen_econ.items()
                        if k.startswith('gas')) / 1000.0, 2),
                'market_stop': state['market_stopped'],
                'resource_mix_twh': {k: round(v, 2) for k, v in resource_mix_twh.items()},
                'cumulative_gw': {k: round(v, 2) for k, v in cumulative_gw.items()},
                'zones_deployed': [z['resource'] for z in zone_results],
                'zone_details': zone_results,
                'energy_rev_by_resource': energy_rev_by_resource,
                'capture_rates': capture_rates,
                'generator_economics': gen_econ,
                'adjusted_generator_economics': adjusted_gen_econ,
                'economic_retirements_mw': {k: round(v, 0) for k, v in econ_retired.items()},
                'total_economic_retirement_mw': round(econ_retired_mw, 0),
                'plant_retirements': plant_retirement_list,
                'new_fossil_builds_mw': {k: round(v, 0) for k, v in new_fossil_builds.items()},
                'total_new_fossil_mw': round(new_fossil_mw, 0),
                'new_fossil_details': new_fossil_details,
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
                # Demand response metrics
                'dr_curtailed_gwh': dr_metrics.get('dr_curtailed_gwh', 0),
                'dr_peak_gw': dr_metrics.get('dr_peak_gw', 0),
                'dr_hours': dr_metrics.get('dr_hours', 0),
                'dr_avg_price': dr_metrics.get('dr_avg_price', 0),
                'data_source': _data_sources.get(iso, 'synthetic'),
                'data_quality': _build_data_quality(iso, _full_data_sources),
                # ORDC scarcity metrics
                'ordc_scarcity_hours': round(scarcity_hours_frac * H),
                'ordc_scarcity_hours_fraction': round(scarcity_hours_frac, 4),
                # Endogenous capacity market pricing
                'reserve_margin_pct': round(reserve_margin_pct, 1),
                'capacity_price_kw_yr': round(
                    compute_capacity_price(iso, reserve_margin_pct, current_pct), 2),
                # VRE curtailment feedback (R10)
                'curtailment_rate': round(curtailment_rate, 4),
                # R1: Economics-driven storage deployment
                'battery_pct': round(state.get('storage_deployed', {}).get('battery', 0), 4),
                'battery8_pct': round(state.get('storage_deployed', {}).get('battery8', 0), 4),
                'ldes_pct': round(state.get('storage_deployed', {}).get('ldes', 0), 4),
                'h2_pct': round(state.get('storage_deployed', {}).get('h2', 0), 4),
                'storage_revenue_details': state.get('storage_details', {}),
                'storage_cost_per_mwh': round(storage_result.get('total_storage_cost_mwh', 0), 2),
            }

            # ── LCOE trajectory (endogenous Wright's Law) ────────────
            lcoe_level = conditions.get('lcoe_level', 'Medium')
            learning_speed = conditions.get('learning_speed', 'Medium')
            year_result['lcoe_trajectory'] = compute_lcoe_snapshot(
                iso, cumulative_gw, lcoe_level, learning_speed, year,
                conditions=conditions)

            # ── Zonal congestion data (from zonal LP solver) ────────────
            if zonal_congestion_data and '_congestion' in zonal_congestion_data:
                cong = zonal_congestion_data['_congestion']
                year_result['zonal_lmp_spread'] = cong.get('max_spread_p50', 0)
                year_result['zonal_spread_pair'] = cong.get('max_spread_pair')
                year_result['congested_hours'] = cong.get('max_congested_hours', 0)
                year_result['zonal_congestion'] = cong

            # IPM trigger evaluation — flag when screening-model limits are binding
            ipm_triggers = compute_ipm_triggers(
                iso, year, year_result, gen_econ, state, conditions,
                nuclear_retirement_threshold=nuclear_retirement_threshold,
                zonal_congestion_data=zonal_congestion_data)
            year_result['ipm_triggers'] = ipm_triggers

            results[iso].append(year_result)

    results['_provenance'] = provenance.model_dump()
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SWEEP RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_sweep(isos=None, nuclear_retirement_threshold=None,
                    snapshot_mode=False, weather_years=None):
    """Run full 1,215-scenario market sweep.

    Pre-loads data once, shares LMP cache across scenarios.

    Args:
        weather_years: Optional list of year strings (e.g., ['2021', '2023', '2025'])
            to run weather-year sensitivity. Each year runs the full sweep
            independently, then results are combined. If None, uses default
            year only (1× sweep = 1,215 scenarios). With 5 years, runs 5× sweep
            = 6,075 scenarios total.
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

    # Attach sweep-level provenance
    sweep_provenance = build_provenance_metadata({
        'mode': 'full_sweep',
        'isos': isos,
        'nuclear_retirement_threshold': nuclear_retirement_threshold,
        'snapshot_mode': snapshot_mode,
        'weather_years': weather_years,
        'total_scenarios': total_runs,
    })
    all_results['_provenance'] = sweep_provenance.model_dump()

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# CORRELATED SCENARIO RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def _correlated_scenario_to_conditions(name, spec):
    """Map a CORRELATED_SCENARIOS entry to the conditions dict used by run_market_simulation.

    The correlated scenario spec uses high-level parameter names (gas_price,
    renewable_lcoe, learning_rate, 45q).  This function translates them into
    the internal conditions keys (fuel_level, lcoe_level, learning_speed, etc.).
    """
    # Map gas_price L/M/H → fuel_level and gas_friction
    gas_map = {'Low': ('Low', 0.3), 'Medium': ('Medium', 0.7), 'High': ('High', 1.0)}
    fuel_level, gas_friction = gas_map[spec['gas_price']]

    # Map learning_rate Slow/Medium/Fast → learning_speed
    learning_map = {'Slow': 'Slow', 'Medium': 'Medium', 'Fast': 'Fast'}
    learning_speed = learning_map[spec['learning_rate']]

    return {
        'name': f"Correlated: {name} — {spec['description']}",
        'demand_growth': spec['demand_growth'],
        'lcoe_level': spec['renewable_lcoe'],
        'learning_speed': learning_speed,
        'queue_cap_level': 'Medium',
        'gas_friction': gas_friction,
        'carbon_price': spec['carbon_price'],
        'fuel_level': fuel_level,
        'tx_level': 'Medium',
        'ppa_level': 'Medium',
        'new_fossil_cost_level': 'Medium',
        '_correlated_scenario': name,
    }


def run_correlated_scenarios(iso, scenario_names=None,
                              nuclear_retirement_threshold=None,
                              snapshot_mode=False):
    """Run IEA-aligned correlated scenario bundles for a single ISO.

    Unlike the independent Cartesian sweep, these scenarios represent
    internally-consistent macro futures where parameters are correlated
    as they would be in reality.

    Args:
        iso: ISO region to simulate (e.g. 'PJM').
        scenario_names: List of scenario keys from CORRELATED_SCENARIOS.
            If None, runs all 5 scenarios.
        nuclear_retirement_threshold: Optional nuclear retirement override.
        snapshot_mode: If True, run snapshot (single-year) mode.

    Returns:
        dict keyed by scenario name → simulation results dict.
    """
    if scenario_names is None:
        scenario_names = list(CORRELATED_SCENARIOS.keys())

    # Validate requested scenario names
    invalid = [s for s in scenario_names if s not in CORRELATED_SCENARIOS]
    if invalid:
        raise ValueError(
            f"Unknown correlated scenario(s): {invalid}. "
            f"Valid: {list(CORRELATED_SCENARIOS.keys())}"
        )

    t0 = time.time()
    print(f"Loading common data for correlated scenarios ({iso})...")
    demand_data, gen_profiles, emission_rates, fossil_mix = load_common_data()
    egrid_baselines = load_egrid_baselines()

    preloaded = {
        'demand_data': demand_data,
        'gen_profiles': gen_profiles,
        'emission_rates': emission_rates,
        'fossil_mix': fossil_mix,
        'egrid_baselines': egrid_baselines,
    }

    lmp_cache = {}
    results = {}

    for i, name in enumerate(scenario_names):
        spec = CORRELATED_SCENARIOS[name]
        conditions = _correlated_scenario_to_conditions(name, spec)
        scenario_id = f"CORR_{name}"

        print(f"\n[{i+1}/{len(scenario_names)}] {scenario_id}: {spec['description']}")
        sim_result = run_market_simulation(
            scenario_id, conditions, isos=[iso],
            nuclear_retirement_threshold=nuclear_retirement_threshold,
            snapshot_mode=snapshot_mode,
            _preloaded=preloaded,
            _lmp_cache=lmp_cache,
            _quiet=True,
        )
        results[name] = {
            'scenario_id': scenario_id,
            'description': spec['description'],
            'parameters': dict(spec),
            'results': sim_result,
        }

    elapsed = time.time() - t0
    print(f"\nCorrelated scenarios complete: {len(scenario_names)} scenarios in {elapsed:.1f}s")

    # Attach provenance
    provenance = build_provenance_metadata({
        'mode': 'correlated_scenarios',
        'iso': iso,
        'scenarios': scenario_names,
        'snapshot_mode': snapshot_mode,
    })
    results['_provenance'] = provenance.model_dump()

    return results


def _compute_weighted_percentiles(values, weights, percentiles):
    """Compute weighted percentiles using linear interpolation.

    Args:
        values: 1-D numpy array of metric values.
        weights: 1-D numpy array of scenario weights (same length as values).
        percentiles: list of percentiles in [0, 100].

    Returns:
        list of weighted percentile values (same order as *percentiles*).
    """
    sort_idx = np.argsort(values)
    sorted_vals = values[sort_idx]
    sorted_wts = weights[sort_idx]

    # Cumulative weight, normalised to [0, 100]
    cum_wt = np.cumsum(sorted_wts)
    cum_wt = 100.0 * (cum_wt - 0.5 * sorted_wts) / cum_wt[-1]

    return [float(np.interp(p, cum_wt, sorted_vals)) for p in percentiles]


def _scenario_weight(scenario_id):
    """Compute combined probability weight for a scenario from its ID.

    Scenario IDs follow the pattern:
        MKT_{demand_code}_{price_name}_{ppa_code}_{gas_code}_{queue_code}_{nfc_code}

    Combined weight = product of individual dimension weights from
    ``pipeline_config.SCENARIO_WEIGHTS``.  Falls back to 1.0 for any
    unrecognised dimension value.
    """
    try:
        from scripts.pipeline_config import SCENARIO_WEIGHTS
    except ImportError:
        try:
            from pipeline_config import SCENARIO_WEIGHTS
        except ImportError:
            return 1.0

    parts = scenario_id.split('_')
    # Expected: MKT, demand_code, price_name, ppa_code, gas_code, queue_code, nfc_code
    # But price_name can itself contain underscores (e.g. "high_vre_low_firm")
    # so we parse from both ends.
    if len(parts) < 7 or parts[0] != 'MKT':
        return 1.0

    # Decode single-character codes back to level names
    code_to_level = {'L': 'Low', 'M': 'Medium', 'H': 'High'}

    demand_code = parts[1]
    ppa_code = parts[-3]
    gas_code = parts[-2]
    # The queue_code and nfc_code are the last two? Re-check format:
    # MKT_{D}_{price_name}_{P}_{G}_{Q}_{N}  → 7 minimum tokens
    queue_code = parts[-2]
    nfc_code = parts[-1]

    # Actually re-parse: parts = [MKT, D, ...price_name..., P, G, Q, N]
    # Last 4 single-char codes: ppa, gas, queue, nfc
    nfc_code = parts[-1]
    queue_code = parts[-2]
    gas_code = parts[-3]
    ppa_code = parts[-4]
    # price_name is everything between index 2 and -4
    price_name = '_'.join(parts[2:-4])

    demand = code_to_level.get(demand_code, demand_code)
    ppa = code_to_level.get(ppa_code, ppa_code)
    gas = code_to_level.get(gas_code, gas_code)
    queue = code_to_level.get(queue_code, queue_code)
    nfc = code_to_level.get(nfc_code, nfc_code)

    w = 1.0
    w *= SCENARIO_WEIGHTS.get('demand', {}).get(demand, 1.0)
    w *= SCENARIO_WEIGHTS.get('price', {}).get(price_name, 1.0)
    w *= SCENARIO_WEIGHTS.get('ppa', {}).get(ppa, 1.0)
    w *= SCENARIO_WEIGHTS.get('gas_friction', {}).get(gas, 1.0)
    w *= SCENARIO_WEIGHTS.get('queue_cap', {}).get(queue, 1.0)
    w *= SCENARIO_WEIGHTS.get('new_fossil_cost', {}).get(nfc, 1.0)

    return w


def _percentile_dict(arr, weights=None):
    """Return dict with P10/P25/P50/P75/P90/mean/std, plus weighted variants."""
    result = {
        'p10': round(float(np.percentile(arr, 10)), 3),
        'p25': round(float(np.percentile(arr, 25)), 3),
        'p50': round(float(np.percentile(arr, 50)), 3),
        'p75': round(float(np.percentile(arr, 75)), 3),
        'p90': round(float(np.percentile(arr, 90)), 3),
        'mean': round(float(np.mean(arr)), 3),
        'std': round(float(np.std(arr)), 3),
        'n': len(arr),
    }

    if weights is not None and len(weights) == len(arr):
        wp = _compute_weighted_percentiles(arr, weights, [10, 25, 50, 75, 90])
        result['wp10'] = round(wp[0], 3)
        result['wp25'] = round(wp[1], 3)
        result['wp50'] = round(wp[2], 3)
        result['wp75'] = round(wp[3], 3)
        result['wp90'] = round(wp[4], 3)

    return result


def aggregate_sweep_percentiles(all_results):
    """Compute P10/P25/P50/P75/P90 uncertainty bands across all sweep scenarios.

    Groups year_results by (iso, year), computes unweighted and weighted
    percentiles across 1,215 scenarios for all scalar metrics and per-resource
    breakdowns.  Weighted percentiles use prior probability weights from
    ``pipeline_config.SCENARIO_WEIGHTS``.

    Args:
        all_results: dict[scenario_id, dict[iso, list[year_result]]]

    Returns:
        dict[iso, dict[year, dict[metric, dict[p10/p25/p50/p75/p90/wp10/.., float]]]]
    """
    # Scalar metrics to aggregate
    SCALAR_METRICS = [
        'clean_pct', 'demand_twh', 'emissions_mt', 'emission_rate_tco2_mwh',
        'cost_per_mwh', 'revenue_per_mwh', 'energy_rev_mwh', 'capacity_rev_mwh',
        'rec_rev_mwh', 'avg_lmp', 'lmp_p90', 'gas_built_gw', 'fossil_built_gw',
        'total_new_fossil_mw', 'reserve_margin_pct', 'capacity_price_kw_yr',
    ]
    # Boolean metrics: report % of scenarios where condition is True
    BOOL_METRICS = ['market_stop', 'nuclear_retired']

    # Pre-compute scenario weights
    scenario_weights = {}
    for scenario_id in all_results:
        scenario_weights[scenario_id] = _scenario_weight(scenario_id)

    # Collect values by (iso, year)
    from collections import defaultdict
    grouped = defaultdict(lambda: defaultdict(list))  # (iso, year) → [year_result, ...]

    for scenario_id, iso_results in all_results.items():
        for iso, year_results in iso_results.items():
            for yr in year_results:
                key = (iso, yr.get('year', 0))
                grouped[key]['_results'].append((scenario_id, yr))

    aggregates = {}
    for (iso, year), data in grouped.items():
        results_list = data['_results']  # list of (scenario_id, year_result)
        n = len(results_list)
        if n < 3:
            continue  # Need at least 3 scenarios for meaningful percentiles

        if iso not in aggregates:
            aggregates[iso] = {}

        year_agg = {}

        # Build weight array aligned with results
        wt_arr = np.array([scenario_weights.get(sid, 1.0)
                           for sid, _ in results_list], dtype=np.float64)

        # Scalar metrics
        for metric in SCALAR_METRICS:
            values = []
            valid_wts = []
            for i, (sid, r) in enumerate(results_list):
                v = r.get(metric, 0) or 0
                if isinstance(v, (int, float)):
                    values.append(v)
                    valid_wts.append(wt_arr[i])
            if values:
                arr = np.array(values, dtype=np.float64)
                w = np.array(valid_wts, dtype=np.float64)
                year_agg[metric] = _percentile_dict(arr, w)

        # Boolean metrics: fraction of scenarios where True (unweighted + weighted)
        for metric in BOOL_METRICS:
            bool_vals = np.array([1.0 if r.get(metric) else 0.0
                                  for _, r in results_list], dtype=np.float64)
            pct_true = round(100.0 * float(np.mean(bool_vals)), 1)
            weighted_pct = round(
                100.0 * float(np.average(bool_vals, weights=wt_arr)), 1
            )
            year_agg[metric] = {
                'pct_true': pct_true,
                'weighted_pct_true': weighted_pct,
                'n': n,
            }

        # Per-resource mix breakdown
        resource_agg = {}
        all_resources = set()
        for _, r in results_list:
            rmix = r.get('resource_mix_twh', {})
            if isinstance(rmix, dict):
                all_resources.update(rmix.keys())

        for resource in sorted(all_resources):
            values = []
            valid_wts = []
            for i, (sid, r) in enumerate(results_list):
                rmix = r.get('resource_mix_twh', {})
                if isinstance(rmix, dict):
                    values.append(rmix.get(resource, 0) or 0)
                    valid_wts.append(wt_arr[i])
            if values:
                arr = np.array(values, dtype=np.float64)
                w = np.array(valid_wts, dtype=np.float64)
                resource_agg[resource] = _percentile_dict(arr, w)

        if resource_agg:
            year_agg['resource_mix_twh'] = resource_agg

        # Nuclear revenue aggregation
        nuc_rev_values = []
        nuc_rev_wts = []
        for i, (sid, r) in enumerate(results_list):
            nr = r.get('nuclear_revenue', {})
            if isinstance(nr, dict) and 'total_mwh' in nr:
                nuc_rev_values.append(nr['total_mwh'])
                nuc_rev_wts.append(wt_arr[i])
        if nuc_rev_values:
            arr = np.array(nuc_rev_values, dtype=np.float64)
            w = np.array(nuc_rev_wts, dtype=np.float64)
            year_agg['nuclear_revenue_mwh'] = _percentile_dict(arr, w)

        # Zone deployment count
        zone_counts = np.array([len(r.get('zones_deployed', []))
                                for _, r in results_list], dtype=np.float64)
        if len(zone_counts):
            year_agg['zones_deployed_count'] = _percentile_dict(zone_counts, wt_arr)

        aggregates[iso][str(year)] = year_agg

    return aggregates


def compute_sweep_uncertainty(all_results):
    """High-level uncertainty quantification entry point.

    Wraps ``aggregate_sweep_percentiles`` and returns both the raw aggregates
    dict and a structured list of ``UncertaintyBands`` dicts suitable for the
    API response.

    Args:
        all_results: dict[scenario_id, dict[iso, list[year_result]]]

    Returns:
        (aggregates_dict, uncertainty_list) where *uncertainty_list* is a list
        of ``{iso, year, bands: [{metric, p10, p25, p50, p75, p90, ...}]}``
        dicts matching the ``SweepUncertainty`` schema.
    """
    aggregates = aggregate_sweep_percentiles(all_results)

    # KEY_METRICS are the headline metrics surfaced in UncertaintyBands
    KEY_METRICS = [
        'clean_pct', 'total_cost_per_mwh', 'cost_per_mwh',
        'emissions_mt', 'avg_lmp', 'demand_twh',
        'gas_built_gw', 'fossil_built_gw',
    ]

    uncertainty_list = []
    for iso, years in aggregates.items():
        for year_str, metrics in years.items():
            bands = []
            for metric in KEY_METRICS:
                m = metrics.get(metric)
                if m and isinstance(m, dict) and 'p50' in m:
                    bands.append({
                        'metric': metric,
                        'p10': m.get('p10', 0),
                        'p25': m.get('p25', 0),
                        'p50': m.get('p50', 0),
                        'p75': m.get('p75', 0),
                        'p90': m.get('p90', 0),
                        'mean': m.get('mean', 0),
                        'std': m.get('std', 0),
                        'n': m.get('n', 0),
                        'wp10': m.get('wp10'),
                        'wp25': m.get('wp25'),
                        'wp50': m.get('wp50'),
                        'wp75': m.get('wp75'),
                        'wp90': m.get('wp90'),
                    })
            if bands:
                uncertainty_list.append({
                    'iso': iso,
                    'year': int(year_str),
                    'bands': bands,
                })

    return aggregates, uncertainty_list


def save_results(all_results, output_dir=None):
    """Save results as JSON, including P10/P25/P50/P75/P90 uncertainty bands."""
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

    # Compute uncertainty bands across all scenarios
    if len(all_results) > 1:
        print("Computing P10/P25/P50/P75/P90 uncertainty bands across sweep scenarios...")
        aggregates, uncertainty_list = compute_sweep_uncertainty(all_results)
        serializable['_aggregates'] = aggregates
        serializable['_uncertainty_bands'] = uncertainty_list
        n_isos = len(aggregates)
        n_years = sum(len(yrs) for yrs in aggregates.values())
        print(f"  Aggregated {n_isos} ISOs × {n_years} year-groups "
              f"from {len(all_results)} scenarios")
        print(f"  {len(uncertainty_list)} ISO×year uncertainty band entries "
              f"(weighted + unweighted)")

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
                        help='Run single scenario instead of full 1,215 sweep')
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
    parser.add_argument('--dr-level', default='Off',
                        choices=['Off', 'Low', 'Medium', 'High'],
                        help='Demand response level (default: Off)')
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
            'dr_level': args.dr_level,
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
