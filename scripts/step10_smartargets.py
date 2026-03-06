#!/usr/bin/env python3
"""
SMARTargets V1 — Market Simulation of Clean Energy Deployment
=============================================================
Market-driven simulation: deploys resources where profitable, stops when profit ≤ 0.
CFE level is an OUTPUT (emerges from profitability), not an input.

V1 Scope: R1-R2 (Reference) + AT1-AT4 (Ambitious Transition) + QT1-QT4 (Quick Transition).
- Per-ISO (no cross-regional capital flow)
- 5% zone steps
- Merchant revenue (no PPA discount)
- RPS/CES compliance modeled via scarcity-driven REC pricing (not mandate floors)

Architecture: Built on scenario_common.build_zones() zone-delta structure.
Revenue model: step5b LMP engine + capacity market + storage + REC.
Cost model: pipeline_config LCOE tables + Wright's Law learning curves.

Usage:
  python step10_smartargets.py                          # R1 + R2, all ISOs
  python step10_smartargets.py --scenarios R1            # R1 only
  python step10_smartargets.py --isos ERCOT CAISO        # subset ISOs
  python step10_smartargets.py --scenarios R1 --isos ERCOT  # single combo
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

from pipeline_config import (
    ISOS, REGIONAL_DEMAND_TWH, DEMAND_GROWTH_RATES,
    GRID_MIX_SHARES, WHOLESALE_PRICES,
    CAPACITY_MARKET_PRICES, PEAK_CAPACITY_CREDITS, RESOURCE_CAPACITY_FACTORS,
    LCOE_TABLES, TX_TABLES, get_tx,
    NUCLEAR_NEWBUILD_LCOE, CCS_LCOE_45Q_ON, GEOTHERMAL_LCOE,
    FOAK_NUCLEAR_NEWBUILD, FOAK_CCS_45Q_ON, FOAK_GEOTHERMAL,
    FOAK_LDES, FOAK_H2, FOAK_OFFSHORE_WIND,
    EXISTING_GAS_FOM_KW_YR,
    NEW_GAS_CCGT_LCOE, NEW_GAS_CT_LCOE,
    compute_storage_revenue_credit,
    OFFSHORE_ISOS, CCS_CAP_TWH, GEOTHERMAL_CAP_TWH,
    H,
)
from dispatch_utils import (
    load_common_data, get_demand_profile, get_supply_profiles,
    reconstruct_hourly_dispatch, compute_fossil_retirement,
    COAL_CAP_TWH, OIL_CAP_TWH,
)
from scenario_common import (
    build_zones, THRESHOLDS,
    _load_feasible_from_parquet,
)
from procurement_utils import get_rps_target_at_year

# Import LMP engine from step5b
from step5b_compute_lmp_prices import (
    build_merit_order_stack, compute_hourly_lmp_vectorized, PriceModel,
)

OUTPUT_DIR = os.path.join(ROOT_DIR, 'data', 'step10-smartargets')

# ═══════════════════════════════════════════════════════════════════════════════
# SMARTARGETS-SPECIFIC CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Capacity market price degradation (alpha) — price falls as clean share rises
# cap_price(t) = base_price × max(0, 1 - alpha × clean_share)
# Energy-only markets (ERCOT, SPP, MISO) already $0, alpha irrelevant
CAPACITY_DEGRADATION_ALPHA = {
    'CAISO': 0.40, 'ERCOT': 0.0, 'PJM': 0.35, 'NYISO': 0.40,
    'NEISO': 0.35, 'MISO': 0.0, 'SPP': 0.0,
}

# Interconnection queue caps (GW/yr new build per ISO)
QUEUE_CAP_GW = {
    'Facilitating': {
        'CAISO': 6, 'ERCOT': 8, 'PJM': 7, 'NYISO': 5,
        'NEISO': 5, 'MISO': 7, 'SPP': 6,
    },
    'Challenging': {
        'CAISO': 3, 'ERCOT': 4, 'PJM': 3, 'NYISO': 2,
        'NEISO': 2, 'MISO': 3, 'SPP': 3,
    },
}

# Wright's Law — deployment-based learning (supersedes time-based for SMARTargets)
# Starting cumulative GW baselines (2025) per SMARTARGETS.md
WRIGHT_CUMULATIVE_GW_2025 = {
    'nuclear': 2.0, 'ccs': 0.3, 'ldes': 0.01, 'h2': 0.1,
    'geothermal': 0.05, 'battery': 50.0, 'battery8': 50.0,
    'solar': 150.0, 'wind': 150.0, 'offshore_wind': 5.0,
}

# Learning rate = cost reduction per doubling of cumulative capacity
WRIGHT_LEARNING_RATE = {
    'nuclear':       {'Fast': 0.15, 'Slow': 0.10},
    'ccs':           {'Fast': 0.12, 'Slow': 0.10},
    'ldes':          {'Fast': 0.20, 'Slow': 0.15},
    'h2':            {'Fast': 0.18, 'Slow': 0.12},
    'geothermal':    {'Fast': 0.20, 'Slow': 0.15},
    'battery':       {'Fast': 0.20, 'Slow': 0.18},
    'battery8':      {'Fast': 0.20, 'Slow': 0.18},
    'solar':         {'Fast': 0.0, 'Slow': 0.0},   # Mature tech — no further learning
    'wind':          {'Fast': 0.0, 'Slow': 0.0},   # Mature tech — no further learning
    'offshore_wind': {'Fast': 0.12, 'Slow': 0.08},
}

# Background learning (exogenous rest-of-world GW by 2035 and 2050)
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

# 45U existing nuclear PTC
PTC_45U_VALUE = 15.0     # $/MWh
PTC_45U_SUNSET_YEAR = 2032

# 45Y new nuclear LCOE reduction (PTC for new-build nuclear only)
PTC_45Y_NEW_NUCLEAR = 26.0  # $/MWh

# Fixed O&M thresholds for retirement
NUCLEAR_FOM_PER_MWH = 30.0   # $/MWh equivalent at 93% CF
COAL_FOM_KW_YR = 45.0        # $/kW-yr

# ─── RPS/REC Compliance Model ─────────────────────────────────────────────
# Scarcity-driven REC pricing: price depends on gap between RPS-eligible
# clean% and RPS target. Voluntary corporate buyers compete for same supply,
# amplifying scarcity beyond just the RPS gap.
#
# Sources: LBNL 2024 RPS Status Update (Barbose), Power Advisory "REC-ord
# High Price", OPIS PJM REC analysis, ICE futures, S&P Global.

# Alternative Compliance Payment caps ($/MWh) — statutory penalty ceiling.
# Weighted average across major load-serving entities within each ISO.
# If ACP < REC market price in a state, utilities pay ACP instead → no build.
ACP_RATES = {
    'CAISO': 50.0,   # CA RPS TREC price cap ($50)
    'ERCOT': 0.0,    # No RPS → no ACP
    'PJM':   45.0,   # Weighted: PA $45, VA $45, NJ $50, MD $30 → ~$45
    'NYISO': 42.5,   # NY Tier 1 ACP (~$42.50)
    'NEISO': 45.0,   # Weighted: MA $67, CT $40, ME/RI ~$35 → ~$45
    'MISO':  15.0,   # Weighted: MN $25, IL $20, MI $10 → ~$15
    'SPP':   10.0,   # Weighted: KS $15, OK $0, NM $10 → ~$10
}

# 2025 observed compliance REC prices ($/MWh) — used as calibration anchors.
# These are COMPLIANCE-grade RECs, not voluntary. Voluntary trade at $1-5.
REC_COMPLIANCE_PRICE_2025 = {
    'CAISO': 34.0,   # WREGIS compliance RECs
    'ERCOT': 0.5,    # No mandate → voluntary only
    'PJM':   38.0,   # Near PA/VA ACP cap ($45). 12× increase since 2018.
    'NYISO': 25.0,   # NY Tier 1, below ACP cap
    'NEISO': 40.0,   # MA Class I, near ACP cap ($45)
    'MISO':  8.0,    # Modest state RPS, some surplus
    'SPP':   3.0,    # Minimal RPS, abundant wind
}

# Voluntary REC floor ($/MWh) — when clean% exceeds RPS target
VOLUNTARY_REC_FLOOR = {
    'CAISO': 3.0, 'ERCOT': 0.5, 'PJM': 2.0, 'NYISO': 2.0,
    'NEISO': 2.5, 'MISO': 1.5, 'SPP': 2.0,
}

# Voluntary corporate demand adder (fraction of load).
# Corporate buyers (PPAs, unbundled RECs) compete with RPS for the same
# clean supply. This additional demand creates scarcity even when aggregate
# clean% slightly exceeds the RPS target. Calibrated so that the scarcity
# model reproduces observed 2025 compliance REC prices.
# Key insight: In NEISO/NYISO, there's basically no clean energy available
# for voluntary buyers — the RPS + corporate demand consumes everything.
VOLUNTARY_DEMAND_ADDER = {
    'CAISO': 0.00,   # Already captured by k calibration — large PPA market
    'ERCOT': 0.02,   # Some corporate demand, but massive surplus
    'PJM':   0.00,   # Already captured by k calibration — binding RPS
    'NYISO': 0.04,   # 4% — NYC corporate demand lifts effective target to ~39%
    'NEISO': 0.035,  # 3.5% — MA/CT voluntary market. Tight. $40/MWh RECs.
    'MISO':  0.06,   # 6% — fills gap between 12% RPS and 18% clean
    'SPP':   0.01,   # 1% — minimal corporate demand
}

# Scarcity model: price = ACP × (1 - exp(-k × gap%))
# k calibrated per ISO against 2025 observed prices.
# Higher k = steeper scarcity response (price ramps toward ACP faster).
# PJM/NEISO high because RPS is binding with interconnection queue bottleneck.
# CAISO lower because larger, more liquid market with PPA-based procurement.
REC_SCARCITY_K = {
    'CAISO': 0.10,   # 11.5% gap → $34 (calibrated to WREGIS observed)
    'ERCOT': 0.10,   # N/A — massive surplus, scarcity never triggers
    'PJM':   0.29,   # 6.5% gap → $38 (calibrated to PJM-GATS observed)
    'NYISO': 0.15,   # Moderate — CES broader than RPS
    'NEISO': 0.29,   # ~$40/MWh at binding — similar scarcity as PJM
    'MISO':  0.12,   # Low — fragmented state RPS, abundant wind
    'SPP':   0.10,   # Low — minimal RPS, massive surplus
}
REC_SURPLUS_DECAY_K = 0.20  # How fast prices collapse when in surplus

# ISOs with Clean Energy Standards (nuclear/CCS count toward compliance)
# vs traditional RPS (only renewables + hydro count)
CES_ISOS = {'NYISO', 'NEISO', 'CAISO'}
# NYISO: CLCPA Tier 3 ZECs for nuclear
# NEISO: CT CCEF for Millstone
# CAISO: SB 100 counts all zero-carbon

REC_ELIGIBLE = {'solar', 'wind', 'offshore_wind', 'hydro', 'geothermal'}
CES_ELIGIBLE = REC_ELIGIBLE | {'clean_firm', 'ccs_ccgt'}
CES_DISCOUNT_FACTOR = 0.60  # ZEC/Tier 3 = ~60% of Tier 1 REC price

# Simulation years (AT trajectory milestones)
SIM_YEARS = [2025, 2030, 2035, 2040, 2045, 2050]

# Map resources in zone delta_resources to tech categories for Wright's Law
RESOURCE_TO_TECH = {
    'clean_firm': 'nuclear', 'solar': 'solar', 'wind': 'wind',
    'offshore_wind': 'offshore_wind', 'ccs_ccgt': 'ccs', 'hydro': 'hydro',
    'battery': 'battery', 'battery8': 'battery8', 'ldes': 'ldes', 'h2': 'h2',
    'geothermal': 'geothermal',
}


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

SCENARIOS = {
    'R1': {
        'name': 'Reference: Facilitating',
        'demand_growth': 'Medium',
        'lcoe_level': 'Low',
        'learning_speed': 'Fast',
        'queue_type': 'Facilitating',
        'gas_friction': 0.7,       # ESG/siting friction on new gas
        'carbon_price': 0,
        'fuel_level': 'Medium',
        'tx_level': 'Medium',
    },
    'R2': {
        'name': 'Reference: Challenging',
        'demand_growth': 'High',
        'lcoe_level': 'High',
        'learning_speed': 'Slow',
        'queue_type': 'Challenging',
        'gas_friction': 1.0,       # Gas builds freely
        'carbon_price': 0,
        'fuel_level': 'Medium',
        'tx_level': 'Medium',
    },

    # ─── Ambitious Transition: Power Sector Net-Zero ─────────────────────
    # Emission CONSTRAINT as % of 2023 eGRID absolute emissions per ISO.
    # Power NZ trajectory: 2030=43%, 2035=18%, 2040=12%, 2045=6%, 2050=0%.
    # e.g., PJM 2045 = 6% of 267 Mt = 16 Mt allowed (94% absolute reduction).
    # Market-profitable build happens first, then mandated build fills the
    # gap to meet the constraint (policy subsidy covers the loss).
    'AT1': {
        'name': 'AT Power NZ: Facilitating',
        'demand_growth': 'Medium',
        'lcoe_level': 'Low',
        'learning_speed': 'Fast',
        'queue_type': 'Facilitating',
        'gas_friction': 0.5,       # Strong ESG/policy friction on new gas
        'carbon_price': 0,
        'fuel_level': 'Medium',
        'tx_level': 'Medium',
        'emission_constraint': 'power_nz',
    },
    'AT2': {
        'name': 'AT Power NZ: Challenging',
        'demand_growth': 'High',
        'lcoe_level': 'High',
        'learning_speed': 'Slow',
        'queue_type': 'Challenging',
        'gas_friction': 0.7,
        'carbon_price': 0,
        'fuel_level': 'Medium',
        'tx_level': 'Medium',
        'emission_constraint': 'power_nz',
    },

    # ─── Ambitious Transition: Economy-Wide Net-Zero ─────────────────────
    # Tighter constraint — economy-wide decarbonization forces power sector
    # to overachieve since it's cheapest to abate.
    # Economy NZ trajectory: 2030=35%, 2035=15%, 2040=8%, 2045=3%, 2050=0%.
    # Electrification drives demand growth (High).
    'AT3': {
        'name': 'AT Economy NZ: Facilitating',
        'demand_growth': 'High',   # Electrification drives demand
        'lcoe_level': 'Low',
        'learning_speed': 'Fast',
        'queue_type': 'Facilitating',
        'gas_friction': 0.3,       # Near-moratorium on new gas
        'carbon_price': 0,
        'fuel_level': 'High',      # Fossil fuel prices rise
        'tx_level': 'Low',         # Accelerated tx buildout
        'emission_constraint': 'economy_nz',
    },
    'AT4': {
        'name': 'AT Economy NZ: Challenging',
        'demand_growth': 'High',
        'lcoe_level': 'High',
        'learning_speed': 'Slow',
        'queue_type': 'Challenging',
        'gas_friction': 0.5,
        'carbon_price': 0,
        'fuel_level': 'High',
        'tx_level': 'Medium',
        'emission_constraint': 'economy_nz',
    },

    # ─── Quick Transition ────────────────────────────────────────────────
    # Parametric sweep: 19 emission reduction targets (5%, 10%, ..., 95%)
    # by 2050. Each target follows the AT temporal shape scaled to that
    # final reduction level. QT1/QT2 use power NZ shape, QT3/QT4 economy NZ.
    # The 'qt_reduction_grid' flag triggers the sweep in run_market_simulation.
    'QT1': {
        'name': 'QT Power NZ: Facilitating',
        'demand_growth': 'Medium',
        'lcoe_level': 'Low',
        'learning_speed': 'Fast',
        'queue_type': 'Facilitating',
        'gas_friction': 0.3,
        'carbon_price': 0,
        'fuel_level': 'Low',
        'tx_level': 'Low',
        'emission_constraint': 'power_nz',
        'qt_reduction_grid': True,
    },
    'QT2': {
        'name': 'QT Power NZ: Challenging',
        'demand_growth': 'Medium',
        'lcoe_level': 'High',
        'learning_speed': 'Slow',
        'queue_type': 'Challenging',
        'gas_friction': 0.7,
        'carbon_price': 0,
        'fuel_level': 'Medium',
        'tx_level': 'Medium',
        'emission_constraint': 'power_nz',
        'qt_reduction_grid': True,
    },
    'QT3': {
        'name': 'QT Economy NZ: Facilitating',
        'demand_growth': 'High',
        'lcoe_level': 'Low',
        'learning_speed': 'Fast',
        'queue_type': 'Facilitating',
        'gas_friction': 0.3,
        'carbon_price': 0,
        'fuel_level': 'Low',
        'tx_level': 'Low',
        'emission_constraint': 'economy_nz',
        'qt_reduction_grid': True,
    },
    'QT4': {
        'name': 'QT Economy NZ: Challenging',
        'demand_growth': 'High',
        'lcoe_level': 'High',
        'learning_speed': 'Slow',
        'queue_type': 'Challenging',
        'gas_friction': 0.5,
        'carbon_price': 0,
        'fuel_level': 'High',
        'tx_level': 'Medium',
        'emission_constraint': 'economy_nz',
        'qt_reduction_grid': True,
    },
}

# QT reduction targets: 5% to 95% in 5% intervals
QT_REDUCTION_TARGETS = [i / 100.0 for i in range(5, 100, 5)]  # 0.05, 0.10, ..., 0.95


# ═══════════════════════════════════════════════════════════════════════════════
# EMISSION CONSTRAINT TRAJECTORIES (% of 2023 eGRID absolute emissions)
# ═══════════════════════════════════════════════════════════════════════════════

# Power sector net-zero: moderate pace
EMISSION_TRAJECTORY_POWER_NZ = {
    2025: 1.00,   # No constraint in base year
    2030: 0.43,   # 57% reduction from 2023
    2035: 0.18,   # 82% reduction
    2040: 0.12,   # 88% reduction
    2045: 0.06,   # 94% reduction
    2050: 0.00,   # Net-zero
}

# Economy-wide net-zero: power sector must overachieve (cheapest to abate)
EMISSION_TRAJECTORY_ECONOMY_NZ = {
    2025: 1.00,
    2030: 0.35,   # 65% reduction — faster than power-only
    2035: 0.15,   # 85% reduction
    2040: 0.08,   # 92% reduction
    2045: 0.03,   # 97% reduction
    2050: 0.00,   # Net-zero
}

EMISSION_TRAJECTORIES = {
    'power_nz': EMISSION_TRAJECTORY_POWER_NZ,
    'economy_nz': EMISSION_TRAJECTORY_ECONOMY_NZ,
}

# eGRID 2023 baseline emissions (loaded from JSON, fallback hardcoded)
EGRID_2023_BASELINE_PATH = os.path.join(ROOT_DIR, 'data', 'egrid_2023_baseline_emissions.json')


def load_egrid_baselines():
    """Load 2023 absolute emissions per ISO from eGRID data.

    Returns {iso: co2_metric_tons}.
    """
    if os.path.exists(EGRID_2023_BASELINE_PATH):
        with open(EGRID_2023_BASELINE_PATH) as f:
            data = json.load(f)
        return {iso: d['co2_metric_tons'] for iso, d in data.items() if iso in ISOS}

    # Fallback hardcoded from eGRID 2023 BA23 sheet
    return {
        'CAISO':  31_376_504,
        'ERCOT': 157_460_286,
        'PJM':   267_318_273,
        'NYISO':  28_193_964,
        'NEISO':  25_081_328,
        'MISO':  290_402_405,
        'SPP':   110_506_609,
    }


def get_emission_cap_mt(iso, year, constraint_type, baselines, reduction_target=1.0):
    """Get the emission cap in million metric tons for an ISO at a given year.

    Args:
        iso: ISO region
        year: Simulation year
        constraint_type: 'power_nz' or 'economy_nz'
        baselines: {iso: co2_metric_tons} from eGRID
        reduction_target: Final reduction fraction by 2050 (default 1.0 = 100% = net-zero).
            For QT sweep: 0.05 = 5% reduction, 0.50 = 50%, 0.95 = 95%.
            The AT trajectory shape is scaled proportionally to this target.

    Returns: cap in million metric tons, or None if no constraint.
    """
    if constraint_type is None:
        return None
    trajectory = EMISSION_TRAJECTORIES.get(constraint_type)
    if trajectory is None:
        return None

    # AT trajectory gives fraction REMAINING at each year (1.0 = no reduction, 0.0 = net-zero)
    # Scale the reduction proportionally: if AT reduces by X% at year Y,
    # QT at target T reduces by X*T% at year Y.
    # fraction_remaining = 1.0 - (1.0 - at_fraction) * reduction_target
    at_fraction = trajectory.get(year, 1.0)
    at_reduction = 1.0 - at_fraction  # How much AT reduces at this year
    scaled_reduction = at_reduction * reduction_target
    fraction_remaining = 1.0 - scaled_reduction

    baseline_mt = baselines.get(iso, 0) / 1e6  # metric tons → million metric tons
    return baseline_mt * fraction_remaining


# ═══════════════════════════════════════════════════════════════════════════════
# WRIGHT'S LAW LEARNING CURVES
# ═══════════════════════════════════════════════════════════════════════════════

def wright_cost(foak_cost, noak_floor, cumulative_gw, reference_gw, learning_rate):
    """Deployment-based Wright's Law cost at current cumulative GW.

    cost = FOAK × (cumulative / reference) ^ (-learning_exponent)
    Capped at NOAK floor.
    """
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
        frac = (year - 2025) / 10.0
        return gw_2035 * frac
    # 2035-2050
    frac = (year - 2035) / 15.0
    return gw_2035 + (gw_2050 - gw_2035) * frac


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
                              battery_pct=0, battery8_pct=0, ldes_pct=0, h2_pct=0):
    """Compute 8760-hour LMP at a given clean percentage.

    Args:
        demand_norm: Normalized demand profile (sums to ~1.0) — for dispatch reconstruction
        demand_mw_profile: MW demand profile (demand_norm × total_mwh) — for LMP pricing

    Returns (hourly_lmp_array, avg_lmp, lmp_p90).
    """
    # Build fossil merit-order stack
    stack, total_fossil_mw = build_merit_order_stack(
        iso, clean_pct, fuel_level=fuel_level,
        resource_mix=resource_pcts,
        battery_pct=battery_pct, battery8_pct=battery8_pct,
        ldes_pct=ldes_pct, h2_pct=h2_pct,
    )

    # Reconstruct dispatch — uses NORMALIZED demand profile
    dispatch = reconstruct_hourly_dispatch(
        demand_norm, supply_profiles, resource_pcts,
        procurement_pct=clean_pct,
        battery_dispatch_pct=battery_pct,
        battery8_dispatch_pct=battery8_pct,
        ldes_dispatch_pct=ldes_pct,
        h2_dispatch_pct=h2_pct,
    )

    # Compute LMP — uses MW demand profile
    price_model = PriceModel(iso, fuel_level)
    hourly_lmp, _ = compute_hourly_lmp_vectorized(
        dispatch, demand_mw_profile, stack, price_model, iso=iso,
    )

    avg_lmp = float(np.mean(hourly_lmp))
    p90_lmp = float(np.percentile(hourly_lmp, 90))
    return hourly_lmp, avg_lmp, p90_lmp


def compute_energy_revenue_by_resource(hourly_lmp, supply_profiles, resource_pcts,
                                        demand_total_mwh):
    """Compute energy revenue ($/MWh) per resource from hourly LMP × generation profile.

    Returns dict: {resource: revenue_per_mwh}
    """
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
            # Revenue = sum(gen[h] × lmp[h]) / sum(gen[h])
            revenue_weighted = float(np.sum(gen_profile * hourly_lmp))
            gen_total = float(np.sum(gen_profile))
            revenues[res] = revenue_weighted / gen_total if gen_total > 0 else 0
    return revenues


def compute_capacity_revenue(iso, clean_pct, resource_pcts):
    """Compute capacity market revenue ($/MWh) per resource, degraded by clean share.

    Returns dict: {resource: cap_rev_per_mwh}
    """
    base_price = CAPACITY_MARKET_PRICES.get(iso, 0)
    alpha = CAPACITY_DEGRADATION_ALPHA.get(iso, 0)
    degraded_price = base_price * max(0, 1.0 - alpha * clean_pct / 100.0)

    cap_revs = {}
    for res, pct in resource_pcts.items():
        if pct <= 0:
            continue
        elcc = PEAK_CAPACITY_CREDITS.get(res, 0)
        cf = RESOURCE_CAPACITY_FACTORS.get(res, {}).get(iso, 0.30)
        if cf > 0:
            # $/kW-yr → $/MWh: (price × ELCC) / (CF × 8760)
            cap_revs[res] = degraded_price * elcc / (cf * 8.760)  # kW-yr to MWh
        else:
            cap_revs[res] = 0
    return cap_revs


def get_compliance_eligible_pct(resource_pcts, iso):
    """Clean % counting only compliance-eligible resources.

    CES ISOs (NYISO, NEISO, CAISO): nuclear and CCS count.
    Traditional RPS ISOs: only renewables + hydro + geothermal.
    Critical: PJM has 32.1% nuclear but only 8.5% RPS-eligible → real scarcity.
    """
    eligible = CES_ELIGIBLE if iso in CES_ISOS else REC_ELIGIBLE
    return sum(pct for res, pct in resource_pcts.items()
               if res in eligible and pct > 0)


def compute_rec_price(iso, eligible_pct, year):
    """Scarcity-driven compliance REC price ($/MWh).

    When clean% < RPS target: scarcity drives prices toward ACP cap.
    When clean% > RPS target: surplus collapses prices toward voluntary floor.
    Voluntary corporate buyers compete for same supply, amplifying scarcity.
    """
    acp = ACP_RATES.get(iso, 0)
    if acp <= 0:
        return VOLUNTARY_REC_FLOOR.get(iso, 0)  # No RPS (ERCOT)

    # Effective demand = RPS mandate + voluntary corporate procurement.
    # Corporate buyers compete for same clean supply → amplifies scarcity.
    rps_target_frac = get_rps_target_at_year(iso, year)
    vol_adder = VOLUNTARY_DEMAND_ADDER.get(iso, 0)
    eff_target_pct = (rps_target_frac + vol_adder) * 100.0
    gap = eff_target_pct - eligible_pct  # positive = scarcity

    floor = VOLUNTARY_REC_FLOOR.get(iso, 1.0)
    k_scarcity = REC_SCARCITY_K.get(iso, 0.15)
    compliance_2025 = REC_COMPLIANCE_PRICE_2025.get(iso, 5.0)

    if gap > 0:
        # Scarcity: price ramps toward ACP. Calibrated per ISO to match
        # 2025 observed compliance REC prices.
        price = acp * (1.0 - np.exp(-k_scarcity * gap))
    else:
        # Surplus: price decays toward voluntary floor.
        price = floor + (compliance_2025 - floor) * np.exp(REC_SURPLUS_DECAY_K * gap)

    return max(floor, min(acp, price))


def compute_rec_revenue(iso, resource_pcts, clean_pct, year):
    """REC/CES revenue for eligible resources ($/MWh additive to energy).

    Scarcity-driven: price depends on gap between compliance-eligible clean%
    and RPS/CES target. ACP-capped. CES ISOs give nuclear/CCS partial credit.
    """
    eligible_pct = get_compliance_eligible_pct(resource_pcts, iso)
    rec_price = compute_rec_price(iso, eligible_pct, year)

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
                          supply_profiles, demand_total_mwh, year):
    """Total blended revenue ($/MWh) for resources at this zone.

    Combines energy (LMP) + capacity + REC. Storage revenue handled separately.
    """
    energy_revs = compute_energy_revenue_by_resource(
        hourly_lmp, supply_profiles, resource_pcts, demand_total_mwh)
    cap_revs = compute_capacity_revenue(iso, clean_pct, resource_pcts)
    rec_revs = compute_rec_revenue(iso, resource_pcts, clean_pct, year)

    # Blended revenue weighted by resource share
    total_pct = sum(pct for pct in resource_pcts.values() if pct > 0)
    if total_pct <= 0:
        return 0, {}

    per_resource_rev = {}
    blended = 0
    for res, pct in resource_pcts.items():
        if pct <= 0:
            continue
        rev = (energy_revs.get(res, 0)
               + cap_revs.get(res, 0)
               + rec_revs.get(res, 0))
        per_resource_rev[res] = round(rev, 2)
        blended += rev * (pct / total_pct)

    return round(blended, 2), per_resource_rev


# ═══════════════════════════════════════════════════════════════════════════════
# COST MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def get_resource_lcoe(res, iso, lcoe_level, cumulative_gw, learning_speed, year):
    """Get effective LCOE for a resource after Wright's Law learning.

    Solar/wind/battery LCOE tables already include all-in costs (no separate 45Y).
    New nuclear gets 45Y PTC reduction. CCS has 45Q baked into tables.
    """
    tech = RESOURCE_TO_TECH.get(res, res)

    # Map resource to LCOE table
    if res == 'clean_firm':
        # Nuclear new-build
        base_lcoe = NUCLEAR_NEWBUILD_LCOE.get(lcoe_level, {}).get(iso, 100)
        foak = FOAK_NUCLEAR_NEWBUILD.get(iso, 175)
        noak = NUCLEAR_NEWBUILD_LCOE.get('Low', {}).get(iso, 68)
        lr = WRIGHT_LEARNING_RATE.get('nuclear', {}).get(learning_speed, 0.12)
        ref_gw = WRIGHT_CUMULATIVE_GW_2025.get('nuclear', 2.0)
        eff_gw = get_effective_cumulative_gw('nuclear', cumulative_gw.get('nuclear', 0),
                                              learning_speed, year)
        cost = wright_cost(foak, noak, eff_gw, ref_gw, lr)
        # 45Y PTC for new nuclear reduces effective LCOE
        cost = max(noak, cost - PTC_45Y_NEW_NUCLEAR)
        return cost

    elif res == 'ccs_ccgt':
        base_lcoe = CCS_LCOE_45Q_ON.get(lcoe_level, {}).get(iso, 120)
        foak = FOAK_CCS_45Q_ON.get(iso, 130)
        noak = CCS_LCOE_45Q_ON.get('Low', {}).get(iso, 80)
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
        # VRE: LCOE tables already all-in. Apply mild Wright's Law for cost decline.
        table = LCOE_TABLES.get(res, {})
        base_lcoe = table.get(lcoe_level, {}).get(iso, 50) if isinstance(table.get(lcoe_level), dict) else 50
        noak = table.get('Low', {}).get(iso, 30) if isinstance(table.get('Low'), dict) else 30
        lr = WRIGHT_LEARNING_RATE.get(tech, {}).get(learning_speed, 0.03)
        ref_gw = WRIGHT_CUMULATIVE_GW_2025.get(tech, 100)
        eff_gw = get_effective_cumulative_gw(tech, cumulative_gw.get(tech, 0),
                                              learning_speed, year)
        return wright_cost(base_lcoe, noak, eff_gw, ref_gw, lr)

    elif res == 'hydro':
        # Existing only, near-zero cost (already built)
        return 0

    elif res in ('battery', 'battery8', 'ldes', 'h2'):
        # Storage: LCOE in special units ($/MWh-capacity-coeff).
        # For profitability, use the storage revenue credit approach.
        # Return a representative LCOE after revenue credit offset.
        table = LCOE_TABLES.get(res, {})
        stor_type = 'battery' if res in ('battery', 'battery8') else res
        base_val = table.get(lcoe_level, {}).get(iso, 5000) if isinstance(table.get(lcoe_level), dict) else 5000
        noak_val = table.get('Low', {}).get(iso, 2000) if isinstance(table.get('Low'), dict) else 2000
        lr = WRIGHT_LEARNING_RATE.get(tech, {}).get(learning_speed, 0.15)
        ref_gw = WRIGHT_CUMULATIVE_GW_2025.get(tech, 1.0)
        eff_gw = get_effective_cumulative_gw(tech, cumulative_gw.get(tech, 0),
                                              learning_speed, year)
        return wright_cost(base_val, noak_val, eff_gw, ref_gw, lr)

    else:
        return 50  # Fallback


def compute_zone_cost(iso, delta_resources, lcoe_level, cumulative_gw,
                       learning_speed, year, tx_level='Medium'):
    """Compute blended LCOE ($/MWh) for incremental resources in a zone.

    Returns (blended_cost, per_resource_cost_dict).
    """
    per_resource_cost = {}
    total_twh = 0
    weighted_cost = 0

    for res, delta_twh in delta_resources.items():
        if delta_twh <= 0:
            continue
        lcoe = get_resource_lcoe(res, iso, lcoe_level, cumulative_gw,
                                  learning_speed, year)

        # Add transmission for applicable resources
        if res in ('solar', 'wind', 'clean_firm', 'offshore_wind'):
            tx = get_tx(res if res != 'clean_firm' else 'clean_firm', tx_level, iso)
            lcoe += tx

        per_resource_cost[res] = round(lcoe, 2)
        weighted_cost += lcoe * delta_twh
        total_twh += delta_twh

    blended = weighted_cost / total_twh if total_twh > 0 else 0
    return round(blended, 2), per_resource_cost


# ═══════════════════════════════════════════════════════════════════════════════
# STEPPING LOOP — CORE MARKET SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def load_step3_data():
    """Load step3 cost optimization results for all ISOs.

    Returns {iso: {threshold_float: result_dict}}.
    """
    step3_dir = os.path.join(ROOT_DIR, 'data', 'step3-cost-opt-parquets')
    all_data = {}

    for iso in ISOS:
        path = os.path.join(step3_dir, f'step3_co_{iso}.parquet')
        if not os.path.exists(path):
            print(f"  WARNING: No step3 parquet for {iso}, skipping")
            continue

        df = pd.read_parquet(path)

        # Filter to Medium cost scenario (baseline resource mix)
        # Medium = ren_M, firm_M, batt_M, ldes_M, fuel_M, tx_M, ccs_M, q45_1
        # We'll use the best mix per threshold at Medium costs
        iso_data = {}
        for t_val, grp in df.groupby('threshold'):
            t = float(t_val)
            # Get the lowest-cost mix at this threshold (Medium scenario)
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

    return all_data


def get_demand_at_year(iso, year, growth_level):
    """Project demand TWh at a future year."""
    base = REGIONAL_DEMAND_TWH[iso]
    rate = DEMAND_GROWTH_RATES[iso][growth_level]
    return base * (1 + rate) ** (year - 2025)


def twh_from_resource_pcts(resource_pcts, demand_twh):
    """Convert resource percentages to TWh."""
    return {res: pct / 100.0 * demand_twh for res, pct in resource_pcts.items()}


def estimate_new_gw_from_delta(delta_resources_twh, iso):
    """Estimate GW of new capacity from TWh delta using capacity factors."""
    gw = {}
    for res, twh in delta_resources_twh.items():
        if twh <= 0:
            continue
        tech = RESOURCE_TO_TECH.get(res, res)
        cf = RESOURCE_CAPACITY_FACTORS.get(res, {}).get(iso, 0.25)
        if cf > 0:
            gw[tech] = twh / (cf * 8.760)  # TWh / (CF × 8760h) = GW
        else:
            gw[tech] = 0
    return gw


def run_market_simulation(scenario_id, conditions, isos=None, reduction_target=1.0):
    """Run market simulation with optional emission constraints.

    For R1/R2 (Reference): purely profit-driven — deploy where profitable, stop otherwise.
    For AT (Ambitious Transition): profit-driven first, then mandated deployment to meet
        emission constraint (% of 2023 eGRID absolute emissions). Builds sequentially
        on prior year's resources — least-cost path to each year's emission cap.
    For QT (Quick Transition): parametric sweep of reduction targets (5-95% in 5% steps).
        Each target scales the AT trajectory proportionally.

    Args:
        reduction_target: Final emission reduction fraction by 2050 (default 1.0 = 100%).
            Used by QT to scale the AT trajectory shape.

    Returns {iso: [year_result_dict, ...]} with per-year deployment trajectory.
    """
    if isos is None:
        isos = list(ISOS)

    emission_constraint = conditions.get('emission_constraint')

    print(f"\n{'='*70}")
    label = f"SMARTargets V1 — Scenario {scenario_id}: {conditions['name']}"
    if reduction_target < 1.0:
        label += f" [{reduction_target*100:.0f}% reduction]"
    print(label)
    if emission_constraint:
        print(f"  Emission constraint: {emission_constraint}, target={reduction_target*100:.0f}%")
    print(f"{'='*70}")

    # Load shared data
    t0 = time.time()
    print("Loading common data...")
    demand_data, gen_profiles, emission_rates, fossil_mix = load_common_data()
    print(f"  Common data loaded in {time.time()-t0:.1f}s")

    # Load step3 results
    print("Loading step3 cost optimization parquets...")
    step3_data = load_step3_data()

    # Load eGRID baselines (for AT/QT emission constraints)
    egrid_baselines = load_egrid_baselines() if emission_constraint else {}

    # Global cumulative GW tracker (cross-ISO learning pool)
    cumulative_gw = dict(WRIGHT_CUMULATIVE_GW_2025)

    results = {iso: [] for iso in isos}

    # Per-ISO state tracking
    iso_state = {}
    for iso in isos:
        existing_clean = sum(GRID_MIX_SHARES.get(iso, {}).values())
        iso_state[iso] = {
            'clean_pct': existing_clean,
            'market_stopped': False,
            'gas_built_gw': 0,
            'mandated_subsidy_total': 0,  # Cumulative subsidy for mandated build
        }

    for year in SIM_YEARS:
        print(f"\n--- Year {year} ---")

        for iso in isos:
            if iso not in step3_data:
                continue

            state = iso_state[iso]
            # Reset market_stopped each year — learning curves may unlock new zones
            if state['market_stopped'] and year > 2025:
                state['market_stopped'] = False

            demand_twh = get_demand_at_year(iso, year, conditions['demand_growth'])
            demand_total_mwh = demand_twh * 1e6

            # Get demand and supply profiles
            demand_norm, total_mwh_base = get_demand_profile(iso, demand_data)
            supply_profiles_iso = get_supply_profiles(iso, gen_profiles)

            # demand_mw_profile = demand_norm × total_annual_mwh (MW units)
            # per step5b pattern (line 1159): demand_mw_profile = demand_norm * total_mwh
            growth_factor = demand_twh / REGIONAL_DEMAND_TWH[iso]
            demand_mw_profile = np.array(demand_norm, dtype=np.float64) * total_mwh_base * growth_factor

            iso_data = step3_data[iso]
            current_pct = state['clean_pct']

            # Build candidate zones from current_pct upward
            candidate_thresholds = sorted(t for t in THRESHOLDS if t > current_pct)
            if not candidate_thresholds:
                state['market_stopped'] = True
                continue

            # Annual queue budget
            queue_budget_gw = QUEUE_CAP_GW[conditions['queue_type']][iso]
            # Scale to zone step years (5 years per milestone period)
            years_in_period = 5
            if year == 2025:
                years_in_period = 5  # 2025-2030
            queue_remaining_gw = queue_budget_gw * years_in_period

            zone_deployed = False
            zone_results = []
            avg_lmp = WHOLESALE_PRICES[iso]
            p90_lmp = avg_lmp * 1.5
            blended_revenue = 0
            blended_cost = 0

            for t_end in candidate_thresholds:
                if queue_remaining_gw <= 0:
                    break

                # Find the step3 data closest to this threshold
                if t_end not in iso_data:
                    # Find nearest available threshold
                    available = sorted(iso_data.keys())
                    nearest = min(available, key=lambda x: abs(x - t_end))
                    mix_data = iso_data[nearest]
                else:
                    mix_data = iso_data[t_end]

                resource_pcts = mix_data['resource_pcts']

                # --- FIND PRIOR THRESHOLD DATA ---
                lower_thresholds = sorted(t for t in iso_data if t <= current_pct)
                t_start = lower_thresholds[-1] if lower_thresholds else min(iso_data.keys())
                start_data = iso_data.get(t_start, {})
                start_pcts = start_data.get('resource_pcts', {r: 0 for r in resource_pcts})

                # --- DELTA RESOURCES ---
                delta_pcts = {}
                delta_twh = {}
                for res in resource_pcts:
                    dpct = resource_pcts[res] - start_pcts.get(res, 0)
                    if dpct > 0.1:
                        delta_pcts[res] = dpct
                        delta_twh[res] = dpct / 100.0 * demand_twh

                # If optimizer has same mix at both thresholds, use the zone's
                # clean% increase as a proxy for incremental build.
                # The zone still adds clean energy even if the mix % doesn't change
                # (because demand-normalized procurement increases).
                zone_clean_increase = t_end - current_pct
                if not delta_twh and zone_clean_increase > 0:
                    # The dominant resource is the largest in the mix
                    dominant = max(resource_pcts, key=resource_pcts.get)
                    delta_pcts = {dominant: zone_clean_increase}
                    delta_twh = {dominant: zone_clean_increase / 100.0 * demand_twh}

                if not delta_twh:
                    continue

                # --- REVENUE ---
                # LMP at the END threshold (reflects fossil stack at this clean %)
                hourly_lmp, avg_lmp, p90_lmp = compute_lmp_at_threshold(
                    iso, t_end, conditions['fuel_level'],
                    demand_norm, demand_mw_profile,
                    supply_profiles_iso, resource_pcts,
                    battery_pct=mix_data['battery_pct'],
                    battery8_pct=mix_data['battery8_pct'],
                    ldes_pct=mix_data['ldes_pct'],
                    h2_pct=mix_data['h2_pct'],
                )
                # Revenue for the DELTA resources at this LMP
                blended_revenue, per_res_rev = compute_zone_revenue(
                    iso, t_end, delta_pcts, hourly_lmp,
                    supply_profiles_iso, demand_total_mwh, year,
                )

                # --- COST ---
                # Use step3 incremental cost if available (delta of total_cost)
                step3_cost_end = mix_data.get('total_cost', 0)
                step3_cost_start = start_data.get('total_cost', 0) if start_data else 0
                step3_delta_cost = step3_cost_end - step3_cost_start

                # The step3 delta cost is the incremental cost at Medium settings.
                # Apply Wright's Law adjustment to the delta resources' LCOE.
                blended_cost, per_res_cost = compute_zone_cost(
                    iso, delta_twh, conditions['lcoe_level'], cumulative_gw,
                    conditions['learning_speed'], year, conditions['tx_level'],
                )

                # --- PROFIT ---
                delta_profit = blended_revenue - blended_cost

                # --- DEPLOY OR STOP ---
                if delta_profit > 0:
                    # Check queue capacity
                    new_gw = estimate_new_gw_from_delta(delta_twh, iso)
                    total_new_gw = sum(new_gw.values())

                    if total_new_gw > queue_remaining_gw:
                        # Partial deployment up to queue cap
                        scale = queue_remaining_gw / total_new_gw
                        new_gw = {k: v * scale for k, v in new_gw.items()}
                        total_new_gw = queue_remaining_gw

                    # Update cumulative GW (global learning pool)
                    for tech, gw in new_gw.items():
                        cumulative_gw[tech] = cumulative_gw.get(tech, 0) + gw

                    queue_remaining_gw -= total_new_gw
                    current_pct = t_end
                    state['clean_pct'] = current_pct
                    zone_deployed = True

                    zone_results.append({
                        'threshold': t_end,
                        'revenue': blended_revenue,
                        'cost': blended_cost,
                        'profit': round(delta_profit, 2),
                        'new_gw': round(total_new_gw, 2),
                        'avg_lmp': round(avg_lmp, 1),
                        'mandated': False,
                    })

                    print(f"  {iso} → {t_end:.0f}%: profit={delta_profit:+.1f} $/MWh "
                          f"(rev={blended_revenue:.1f}, cost={blended_cost:.1f}) "
                          f"+{total_new_gw:.1f} GW, LMP avg={avg_lmp:.1f}")
                else:
                    # Market stop — profitable deployment ends here
                    state['market_stopped'] = True
                    market_stop_pct = current_pct
                    market_stop_profit = delta_profit
                    market_stop_rev = blended_revenue
                    market_stop_cost = blended_cost
                    print(f"  {iso} STOP at {current_pct:.0f}%: profit={delta_profit:+.1f} $/MWh "
                          f"(rev={blended_revenue:.1f}, cost={blended_cost:.1f})")
                    break

            # --- EMISSION CONSTRAINT: MANDATED DEPLOYMENT ---
            # After profit-driven deployment, check if emissions exceed the cap.
            # If so, force additional clean zones at a loss (policy subsidy).
            # Builds sequentially on what was already deployed — least-cost path.
            mandated_subsidy = 0
            emission_cap_mt = get_emission_cap_mt(
                iso, year, emission_constraint, egrid_baselines,
                reduction_target=reduction_target)

            if emission_cap_mt is not None and year > 2025:
                # Compute current emissions at this clean_pct
                gf_check = demand_twh / REGIONAL_DEMAND_TWH[iso]
                er_check, _ = compute_fossil_retirement(
                    iso, current_pct, emission_rates, fossil_mix, gf_check)
                fossil_twh_check = (1 - current_pct / 100.0) * demand_twh
                current_co2_mt = fossil_twh_check * 1e6 * er_check / 1e6

                if current_co2_mt > emission_cap_mt:
                    # Need more clean — continue deploying at a loss
                    remaining_thresholds = sorted(
                        t for t in THRESHOLDS if t > current_pct)

                    for t_end in remaining_thresholds:
                        if queue_remaining_gw <= 0:
                            break

                        # Recompute emissions at this threshold
                        er_t, _ = compute_fossil_retirement(
                            iso, t_end, emission_rates, fossil_mix, gf_check)
                        fossil_twh_t = (1 - t_end / 100.0) * demand_twh
                        co2_at_t = fossil_twh_t * 1e6 * er_t / 1e6

                        # Get mix data for this threshold
                        if t_end not in iso_data:
                            available = sorted(iso_data.keys())
                            nearest = min(available, key=lambda x: abs(x - t_end))
                            mix_data = iso_data[nearest]
                        else:
                            mix_data = iso_data[t_end]

                        resource_pcts = mix_data['resource_pcts']

                        # Delta from current position
                        lower_ts = sorted(t for t in iso_data if t <= current_pct)
                        t_start = lower_ts[-1] if lower_ts else min(iso_data.keys())
                        start_data = iso_data.get(t_start, {})
                        start_pcts = start_data.get('resource_pcts',
                                                     {r: 0 for r in resource_pcts})

                        delta_pcts = {}
                        delta_twh = {}
                        for res in resource_pcts:
                            dpct = resource_pcts[res] - start_pcts.get(res, 0)
                            if dpct > 0.1:
                                delta_pcts[res] = dpct
                                delta_twh[res] = dpct / 100.0 * demand_twh

                        zone_clean_increase = t_end - current_pct
                        if not delta_twh and zone_clean_increase > 0:
                            dominant = max(resource_pcts, key=resource_pcts.get)
                            delta_pcts = {dominant: zone_clean_increase}
                            delta_twh = {dominant: zone_clean_increase / 100.0 * demand_twh}

                        if not delta_twh:
                            continue

                        # Cost of mandated deployment
                        m_cost, _ = compute_zone_cost(
                            iso, delta_twh, conditions['lcoe_level'], cumulative_gw,
                            conditions['learning_speed'], year, conditions['tx_level'],
                        )

                        # Revenue (still earned, just not enough to cover cost)
                        hourly_lmp_m, avg_lmp_m, p90_lmp_m = compute_lmp_at_threshold(
                            iso, t_end, conditions['fuel_level'],
                            demand_norm, demand_mw_profile,
                            supply_profiles_iso, resource_pcts,
                            battery_pct=mix_data['battery_pct'],
                            battery8_pct=mix_data['battery8_pct'],
                            ldes_pct=mix_data['ldes_pct'],
                            h2_pct=mix_data['h2_pct'],
                        )
                        m_rev, _ = compute_zone_revenue(
                            iso, t_end, delta_pcts, hourly_lmp_m,
                            supply_profiles_iso, demand_total_mwh, year,
                        )

                        m_profit = m_rev - m_cost
                        subsidy_per_mwh = max(0, -m_profit)  # Policy covers the loss

                        # Deploy (mandated)
                        new_gw = estimate_new_gw_from_delta(delta_twh, iso)
                        total_new_gw = sum(new_gw.values())
                        if total_new_gw > queue_remaining_gw:
                            scale = queue_remaining_gw / total_new_gw
                            new_gw = {k: v * scale for k, v in new_gw.items()}
                            total_new_gw = queue_remaining_gw

                        for tech, gw in new_gw.items():
                            cumulative_gw[tech] = cumulative_gw.get(tech, 0) + gw
                        queue_remaining_gw -= total_new_gw

                        current_pct = t_end
                        state['clean_pct'] = current_pct
                        zone_deployed = True
                        mandated_subsidy += subsidy_per_mwh
                        avg_lmp = avg_lmp_m
                        p90_lmp = p90_lmp_m
                        blended_cost = m_cost
                        blended_revenue = m_rev

                        zone_results.append({
                            'threshold': t_end,
                            'revenue': m_rev,
                            'cost': m_cost,
                            'profit': round(m_profit, 2),
                            'subsidy': round(subsidy_per_mwh, 2),
                            'new_gw': round(total_new_gw, 2),
                            'avg_lmp': round(avg_lmp_m, 1),
                            'mandated': True,
                        })

                        print(f"  {iso} → {t_end:.0f}% [MANDATED]: subsidy={subsidy_per_mwh:+.1f} $/MWh "
                              f"(rev={m_rev:.1f}, cost={m_cost:.1f}) "
                              f"+{total_new_gw:.1f} GW, LMP avg={avg_lmp_m:.1f}")

                        # Check if we've met the cap
                        if co2_at_t <= emission_cap_mt:
                            print(f"  {iso} emission cap met: {co2_at_t:.1f} Mt "
                                  f"≤ {emission_cap_mt:.1f} Mt ({year})")
                            break

                    state['mandated_subsidy_total'] += mandated_subsidy

            # --- NEW GAS BUILD (if demand growth exceeds clean supply) ---
            clean_twh = current_pct / 100.0 * demand_twh
            existing_gas_twh = (1 - current_pct / 100.0) * REGIONAL_DEMAND_TWH[iso]
            unserved = demand_twh - clean_twh - existing_gas_twh
            gas_built_gw = 0

            if unserved > 0:
                gas_lcoe = NEW_GAS_CCGT_LCOE.get(conditions['lcoe_level'], 55)
                gas_revenue = avg_lmp if zone_deployed else WHOLESALE_PRICES[iso]
                if gas_revenue > gas_lcoe:
                    gas_built_gw = unserved / (8.760 * 0.85)  # GW at 85% CF
                    gas_built_gw *= conditions['gas_friction']
                    state['gas_built_gw'] += gas_built_gw
                    print(f"  {iso} new gas: {gas_built_gw:.1f} GW CCGT "
                          f"(rev={gas_revenue:.0f} > cost={gas_lcoe})")

            # --- CO2 EMISSIONS ---
            gf = demand_twh / REGIONAL_DEMAND_TWH[iso]
            emission_rate, _ = compute_fossil_retirement(
                iso, current_pct, emission_rates, fossil_mix, gf)
            fossil_twh = (1 - current_pct / 100.0) * demand_twh
            co2_mt = fossil_twh * 1e6 * emission_rate / 1e6  # million metric tons

            # Record year result
            mix_at_pct = iso_data.get(current_pct, iso_data.get(
                min(iso_data.keys(), key=lambda x: abs(x - current_pct)), {}))
            resource_mix_twh = twh_from_resource_pcts(
                mix_at_pct.get('resource_pcts', {}), demand_twh) if mix_at_pct else {}

            year_result = {
                'iso': iso,
                'scenario': scenario_id,
                'reduction_target': reduction_target,
                'year': year,
                'clean_pct': round(current_pct, 1),
                'demand_twh': round(demand_twh, 1),
                'emissions_mt': round(co2_mt, 2),
                'emission_rate_tco2_mwh': round(emission_rate, 4),
                'emission_cap_mt': round(emission_cap_mt, 2) if emission_cap_mt is not None else None,
                'cost_per_mwh': round(blended_cost if zone_deployed else 0, 2),
                'revenue_per_mwh': round(blended_revenue if zone_deployed else 0, 2),
                'mandated_subsidy_mwh': round(mandated_subsidy, 2),
                'cumulative_subsidy_mwh': round(state['mandated_subsidy_total'], 2),
                'avg_lmp': round(avg_lmp if zone_deployed else WHOLESALE_PRICES[iso], 1),
                'lmp_p90': round(p90_lmp if zone_deployed else 0, 1),
                'gas_built_gw': round(gas_built_gw, 2),
                'total_gas_gw': round(state['gas_built_gw'], 2),
                'market_stop': state['market_stopped'],
                'resource_mix_twh': {k: round(v, 1) for k, v in resource_mix_twh.items()},
                'cumulative_gw': {k: round(v, 2) for k, v in cumulative_gw.items()},
                'queue_used_gw': round(
                    QUEUE_CAP_GW[conditions['queue_type']][iso] * years_in_period - queue_remaining_gw, 1),
                'zones_deployed': zone_results,
            }
            results[iso].append(year_result)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT & SERIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def save_results(results, scenario_id):
    """Save simulation results as parquet + JSON summary."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Flatten to DataFrame
    rows = []
    for iso, year_results in results.items():
        for yr in year_results:
            row = {
                'iso': yr['iso'],
                'scenario': yr['scenario'],
                'reduction_target': yr.get('reduction_target', 1.0),
                'year': yr['year'],
                'clean_pct': yr['clean_pct'],
                'demand_twh': yr['demand_twh'],
                'emissions_mt': yr['emissions_mt'],
                'emission_rate': yr['emission_rate_tco2_mwh'],
                'emission_cap_mt': yr.get('emission_cap_mt'),
                'cost_per_mwh': yr['cost_per_mwh'],
                'revenue_per_mwh': yr['revenue_per_mwh'],
                'mandated_subsidy_mwh': yr.get('mandated_subsidy_mwh', 0),
                'cumulative_subsidy_mwh': yr.get('cumulative_subsidy_mwh', 0),
                'avg_lmp': yr['avg_lmp'],
                'lmp_p90': yr['lmp_p90'],
                'gas_built_gw': yr['gas_built_gw'],
                'total_gas_gw': yr['total_gas_gw'],
                'market_stop': yr['market_stop'],
                'queue_used_gw': yr['queue_used_gw'],
            }
            # Add resource mix columns
            for res in ['clean_firm', 'solar', 'wind', 'offshore_wind',
                        'ccs_ccgt', 'hydro', 'battery', 'ldes']:
                row[f'mix_{res}_twh'] = yr['resource_mix_twh'].get(res, 0)
            rows.append(row)

    df = pd.DataFrame(rows)
    parquet_path = os.path.join(OUTPUT_DIR, f'smartargets_{scenario_id}.parquet')
    df.to_parquet(parquet_path, index=False)
    print(f"\nSaved: {parquet_path} ({len(df)} rows)")

    # JSON summary — market stop points per ISO
    summary = {
        'scenario': scenario_id,
        'name': SCENARIOS[scenario_id]['name'],
        'conditions': SCENARIOS[scenario_id],
        'results': {},
    }

    for iso, year_results in results.items():
        if not year_results:
            continue
        final = year_results[-1]
        stop_year = None
        stop_pct = None
        for yr in year_results:
            if yr['market_stop'] and stop_year is None:
                stop_year = yr['year']
                stop_pct = yr['clean_pct']
                break

        summary['results'][iso] = {
            'market_stop_year': stop_year,
            'market_stop_pct': stop_pct,
            'final_clean_pct': final['clean_pct'],
            'final_emissions_mt': final['emissions_mt'],
            'total_gas_built_gw': final['total_gas_gw'],
            'cumulative_subsidy_mwh': final.get('cumulative_subsidy_mwh', 0),
            'trajectory': [
                {
                    'year': yr['year'],
                    'clean_pct': yr['clean_pct'],
                    'emissions_mt': yr['emissions_mt'],
                    'emission_cap_mt': yr.get('emission_cap_mt'),
                    'cost_per_mwh': yr['cost_per_mwh'],
                    'mandated_subsidy_mwh': yr.get('mandated_subsidy_mwh', 0),
                    'avg_lmp': yr['avg_lmp'],
                }
                for yr in year_results
            ],
        }

    json_path = os.path.join(OUTPUT_DIR, f'smartargets_{scenario_id}_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved: {json_path}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='SMARTargets V1 — Market Simulation of Clean Energy Deployment')
    parser.add_argument('--scenarios', nargs='+', default=['R1', 'R2'],
                        choices=list(SCENARIOS.keys()),
                        help='Scenarios to run (default: R1 R2)')
    parser.add_argument('--isos', nargs='+', default=None,
                        help='ISOs to simulate (default: all 7)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Load data and validate, but skip simulation')
    args = parser.parse_args()

    isos = args.isos or list(ISOS)
    invalid_isos = [i for i in isos if i not in ISOS]
    if invalid_isos:
        print(f"ERROR: Invalid ISOs: {invalid_isos}")
        sys.exit(1)

    print(f"SMARTargets V1 — Market Simulation Engine")
    print(f"Scenarios: {args.scenarios}")
    print(f"ISOs: {isos}")
    print()

    if args.dry_run:
        print("Dry run — loading data to validate...")
        demand_data, gen_profiles, emission_rates, fossil_mix = load_common_data()
        step3_data = load_step3_data()
        for iso in isos:
            if iso in step3_data:
                thresholds = sorted(step3_data[iso].keys())
                print(f"  {iso}: {len(thresholds)} thresholds "
                      f"({thresholds[0]:.0f}% – {thresholds[-1]:.0f}%)")
            else:
                print(f"  {iso}: NO DATA")
        print("\nDry run complete — data validated.")
        return

    t_start = time.time()

    for scenario_id in args.scenarios:
        conditions = SCENARIOS[scenario_id]
        is_qt = conditions.get('qt_reduction_grid', False)

        if is_qt:
            # QT: parametric sweep over 19 reduction targets (5% to 95%)
            all_qt_results = {iso_name: [] for iso_name in isos}

            for target in QT_REDUCTION_TARGETS:
                pct_label = int(target * 100)
                results = run_market_simulation(
                    scenario_id, conditions, isos, reduction_target=target)

                # Accumulate results across all targets
                for iso_name in isos:
                    if iso_name in results:
                        all_qt_results[iso_name].extend(results[iso_name])

            # Save combined results for this QT scenario
            save_results(all_qt_results, scenario_id)

            # Print summary: one line per ISO showing range of outcomes
            print(f"\n{'='*50}")
            print(f"SUMMARY — {scenario_id}: {conditions['name']} (19 targets)")
            print(f"{'='*50}")
            for iso_name in isos:
                iso_results = all_qt_results.get(iso_name, [])
                if not iso_results:
                    continue
                # Group by reduction target — show 2050 outcome per target
                targets_2050 = {}
                for yr in iso_results:
                    t = yr.get('reduction_target', 1.0)
                    if yr['year'] == 2050:
                        targets_2050[t] = yr
                if not targets_2050:
                    continue
                # Find where market stops needing subsidy
                market_viable = [t for t, yr in sorted(targets_2050.items())
                                 if yr.get('mandated_subsidy_mwh', 0) == 0
                                 and yr.get('cumulative_subsidy_mwh', 0) == 0]
                max_viable = max(market_viable) * 100 if market_viable else 0
                min_target = min(targets_2050.keys()) * 100
                max_target = max(targets_2050.keys()) * 100
                print(f"  {iso_name}: targets {min_target:.0f}%-{max_target:.0f}%, "
                      f"market-viable up to {max_viable:.0f}% reduction")

        else:
            # R1/R2/AT: single simulation
            results = run_market_simulation(scenario_id, conditions, isos)
            save_results(results, scenario_id)

            # Print summary
            print(f"\n{'='*50}")
            print(f"SUMMARY — {scenario_id}: {conditions['name']}")
            print(f"{'='*50}")
            for iso_name in isos:
                if iso_name not in results or not results[iso_name]:
                    continue
                final = results[iso_name][-1]
                stop = [yr for yr in results[iso_name] if yr['market_stop']]
                stop_info = f"stops at {stop[0]['clean_pct']:.0f}% (year {stop[0]['year']})" if stop else "no stop"
                subsidy = final.get('cumulative_subsidy_mwh', 0)
                subsidy_info = f", subsidy=${subsidy:.1f}/MWh" if subsidy > 0 else ""
                cap_info = ""
                if final.get('emission_cap_mt') is not None:
                    cap_info = f", cap={final['emission_cap_mt']:.1f}Mt"
                print(f"  {iso_name}: {final['clean_pct']:.0f}% clean by 2050, "
                      f"{final['emissions_mt']:.0f} MtCO2{cap_info}, {stop_info}, "
                      f"+{final['total_gas_gw']:.1f} GW new gas{subsidy_info}")

    elapsed = time.time() - t_start
    print(f"\nTotal runtime: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
