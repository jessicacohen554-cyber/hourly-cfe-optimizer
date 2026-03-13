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
    CAPACITY_MARKET_PRICES, CAPACITY_DEGRADATION_ALPHA,
    PEAK_CAPACITY_CREDITS, RESOURCE_CAPACITY_FACTORS,
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

# Simulation years
SIM_YEARS = [2023, 2030, 2035, 2040, 2045, 2050]

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

# Conditions bundle
CONDITIONS_BUNDLE = {
    'Facilitating': {'learning_speed': 'Fast', 'queue_type': 'Facilitating'},
    'Challenging': {'learning_speed': 'Slow', 'queue_type': 'Challenging'},
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

# PPA market depth by ISO (for PPA discount scaling)
PPA_MARKET_DEPTH = {
    'CAISO': 0.90, 'ERCOT': 1.00, 'PJM': 0.85, 'NYISO': 0.70,
    'NEISO': 0.65, 'MISO': 0.60, 'SPP': 0.50,
}


# ═══════════════════════════════════════════════════════════════════════════════
# EGRID BASELINES
# ═══════════════════════════════════════════════════════════════════════════════

def load_egrid_baselines():
    """Load 2023 eGRID absolute emission baselines."""
    egrid_path = os.path.join(MODULE_ROOT, '..', 'data', 'egrid_2023_baseline_emissions.json')
    if not os.path.exists(egrid_path):
        # Also check in the market-simulator data dir
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
                              carbon_price=0):
    """Compute 8760-hour LMP at a given clean percentage.

    Returns (hourly_lmp_array, avg_lmp, lmp_p90, generator_economics).
    generator_economics is a dict of per-unit-type dispatch metrics.
    """
    stack, total_fossil_mw = build_merit_order_stack(
        iso, clean_pct, fuel_level=fuel_level,
        resource_mix=resource_pcts,
        battery_pct=battery_pct, battery8_pct=battery8_pct,
        ldes_pct=ldes_pct, h2_pct=h2_pct,
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
    )

    avg_lmp = float(np.mean(hourly_lmp))
    p90_lmp = float(np.percentile(hourly_lmp, 90))

    # --- GENERATOR-LEVEL ECONOMICS ---
    gen_econ = compute_generator_economics(
        stack, hourly_lmp, unit_idx, dispatch, demand_mw_profile,
        total_fossil_mw, iso, clean_pct, fuel_level, carbon_price,
    )

    return hourly_lmp, avg_lmp, p90_lmp, gen_econ


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
    # residual_demand is normalized [0-1]; scale to MW
    total_demand_mwh = float(np.sum(demand_mw_profile))
    demand_scale = float(np.mean(demand_mw_profile)) if total_demand_mwh > 0 else 1.0
    fossil_demand_mw = residual * demand_scale

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

        dispatch_hours = int(np.sum(dispatched))
        total_mwh = float(np.sum(mw_dispatched))
        cf = total_mwh / (cap_mw * H) if cap_mw > 0 else 0

        # Revenue: LMP in dispatched hours, weighted by MW
        if total_mwh > 0:
            avg_rev = float(np.sum(hourly_lmp * mw_dispatched)) / total_mwh
        else:
            avg_rev = 0.0

        # Variable cost includes carbon
        var_cost = mc + carbon_adder.get(utype, 0)

        margin = avg_rev - var_cost

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


def compute_nuclear_revenue(iso, clean_pct, hourly_lmp, year):
    """Compute nuclear plant revenue stack by ISO.

    Returns dict with energy_rev, capacity_rev, ptc, total (all $/MWh).
    """
    nuclear_cf = 0.93
    nuclear_gen = np.ones(H) * nuclear_cf  # Flat baseload

    # Energy revenue: LMP × generation
    energy_rev = float(np.mean(hourly_lmp * nuclear_gen)) / nuclear_cf

    # Capacity revenue
    base_price = CAPACITY_MARKET_PRICES.get(iso, 0)
    alpha = CAPACITY_DEGRADATION_ALPHA.get(iso, 0)
    degraded_price = base_price * max(0, 1.0 - alpha * clean_pct / 100.0)
    cap_rev = degraded_price * 1.0 / (nuclear_cf * 8.760)  # ELCC=1.0 for nuclear

    # PTC (45U for existing nuclear)
    ptc = PTC_45U_VALUE if year <= PTC_45U_SUNSET_YEAR else 0.0

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


def compute_capacity_revenue(iso, clean_pct, resource_pcts):
    """Compute capacity market revenue ($/MWh) per resource."""
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
            cap_revs[res] = degraded_price * elcc / (cf * 8.760)
        else:
            cap_revs[res] = 0
    return cap_revs


def get_compliance_eligible_pct(resource_pcts, iso):
    """Clean % counting only compliance-eligible resources."""
    eligible = CES_ELIGIBLE if iso in CES_ISOS else REC_ELIGIBLE
    return sum(pct for res, pct in resource_pcts.items()
               if res in eligible and pct > 0)


def compute_rec_price(iso, eligible_pct, year):
    """Scarcity-driven compliance REC price ($/MWh)."""
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


def compute_rec_revenue(iso, resource_pcts, clean_pct, year):
    """REC/CES revenue for eligible resources ($/MWh)."""
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
    """Total blended revenue ($/MWh) for resources at this zone."""
    energy_revs = compute_energy_revenue_by_resource(
        hourly_lmp, supply_profiles, resource_pcts, demand_total_mwh)
    cap_revs = compute_capacity_revenue(iso, clean_pct, resource_pcts)
    rec_revs = compute_rec_revenue(iso, resource_pcts, clean_pct, year)

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

def get_resource_lcoe(res, iso, lcoe_level, cumulative_gw, learning_speed, year):
    """Get effective LCOE after Wright's Law learning."""
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
        tables = LCOE_TABLES.get(res, {})
        return tables.get(lcoe_level, {}).get(iso, 50)

    elif res == 'hydro':
        return WHOLESALE_PRICES.get(iso, 30)

    else:
        tables = LCOE_TABLES.get(res, {})
        return tables.get(lcoe_level, {}).get(iso, 50)


def _get_ppa_discount(res, ppa_level, iso=None):
    """PPA risk premium discount, scaled by regional market depth."""
    category = 'VRE' if res in ('solar', 'wind', 'offshore_wind') else 'Firm'
    base_discount = PPA_PREMIUMS.get(category, {}).get(ppa_level, 0)
    depth = PPA_MARKET_DEPTH.get(iso, 0.75) if iso else 1.0
    return base_discount * depth


def compute_zone_cost(iso, delta_resources, lcoe_level, cumulative_gw,
                       learning_speed, year, tx_level='Medium', ppa_level=None):
    """Compute blended LCOE ($/MWh) for incremental resources."""
    per_resource_cost = {}
    total_twh = 0
    weighted_cost = 0

    for res, delta_twh in delta_resources.items():
        if delta_twh <= 0:
            continue
        lcoe = get_resource_lcoe(res, iso, lcoe_level, cumulative_gw,
                                  learning_speed, year)

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

def compute_ccs_retrofit_breakeven(iso, fuel_level='Medium'):
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
    ccs_lcoe = CCS_LCOE_45Q_ON.get('Medium', {}).get(iso, 100)
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
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_step3_data():
    """Load step2.2 cost optimization results for all ISOs.

    Returns {iso: {threshold_float: result_dict}}.
    """
    # Check market-simulator data dir first, then main repo
    step3_dir = os.path.join(MODULE_ROOT, 'data', 'step2.2-cost')
    if not os.path.exists(step3_dir) or not os.listdir(step3_dir):
        step3_dir = os.path.join(MODULE_ROOT, '..', 'data', 'step2.2-cost')

    all_data = {}
    for iso in ISOS:
        path = os.path.join(step3_dir, f'step3_co_{iso}.parquet')
        if not os.path.exists(path):
            print(f"  WARNING: No step3 parquet for {iso}, skipping")
            continue

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

    2 conditions × 3 demand × 5 price × 3 PPA × 3 gas friction = 270 scenarios.
    No emission constraints — purely market-driven.

    Returns list of (scenario_id_str, conditions_dict).
    """
    combos = []
    cond_keys = ['Facilitating', 'Challenging']
    demand_keys = DEMAND_GROWTH_LEVELS
    price_keys = list(PRICE_SENSITIVITIES.keys())
    ppa_keys = PPA_LEVELS
    gas_keys = list(GAS_FRICTION_LEVELS.keys())

    for cond, demand, price_name, ppa, gas_name in cartesian(
            cond_keys, demand_keys, price_keys, ppa_keys, gas_keys):

        bundle = CONDITIONS_BUNDLE[cond]
        price_mapping = _map_price_sens_to_lcoe_fuel_tx(PRICE_SENSITIVITIES[price_name])

        cond_code = 'F' if cond == 'Facilitating' else 'C'
        demand_code = demand[0]
        ppa_code = ppa[0]
        gas_code = gas_name[0]
        scenario_id = f"MKT_{cond_code}_{demand_code}_{price_name}_{ppa_code}_{gas_code}"

        conditions = {
            'name': f"Market: {cond} | {demand} demand | {price_name} | PPA={ppa} | Gas={gas_name}",
            'demand_growth': demand,
            'lcoe_level': price_mapping['lcoe_level'],
            'learning_speed': bundle['learning_speed'],
            'queue_type': bundle['queue_type'],
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
        'learning_speed': 'Fast',
        'queue_type': 'Facilitating',
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
                           _preloaded=None, _lmp_cache=None, _quiet=False):
    """Run purely profit-driven market simulation.

    No emission constraints, no mandated deployment, no DAC. Deploy where
    profitable, stop when not. That's the market outcome.

    Args:
        scenario_id: Identifier string for this scenario.
        conditions: Dict with demand_growth, lcoe_level, fuel_level, etc.
        isos: List of ISOs to simulate (default: all 7).
        nuclear_retirement_threshold: $/MWh — if nuclear total revenue falls
            below this, nuclear retires and model re-dispatches. None = no retirement.
        snapshot_mode: If True, run single-year snapshot (no year progression).
        _preloaded: Pre-loaded data dict to avoid re-reading.
        _lmp_cache: Shared LMP cache across scenarios.
        _quiet: Suppress per-zone print output.

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
        step3_data = _preloaded['step3_data']
        egrid_baselines = _preloaded['egrid_baselines']
    else:
        t0 = time.time()
        _log("Loading common data...")
        demand_data, gen_profiles, emission_rates, fossil_mix = load_common_data()
        _log(f"  Common data loaded in {time.time()-t0:.1f}s")
        _log("Loading step3 cost optimization parquets...")
        step3_data = load_step3_data()
        egrid_baselines = load_egrid_baselines()

    cumulative_gw = dict(WRIGHT_CUMULATIVE_GW_2025)
    results = {iso: [] for iso in isos}

    # Per-ISO state
    iso_state = {}
    for iso in isos:
        existing_clean = sum(GRID_MIX_SHARES.get(iso, {}).values())
        iso_state[iso] = {
            'clean_pct': existing_clean,
            'market_stopped': False,
            'gas_built_gw': 0,
            'nuclear_retired': False,
        }

    sim_years = [2025] if snapshot_mode else SIM_YEARS

    for year in sim_years:
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
                }
                results[iso].append(year_result)
                _log(f"  {iso}: 2023 baseline — {baseline_co2_mt:.1f} Mt, "
                     f"{baseline_clean:.1f}% clean, LMP=${baseline_lmp:.0f}")
            continue

        for iso in isos:
            if iso not in step3_data:
                continue

            state = iso_state[iso]
            if state['market_stopped'] and year > 2023:
                state['market_stopped'] = False

            demand_twh = get_demand_at_year(iso, year, conditions['demand_growth'])
            demand_total_mwh = demand_twh * 1e6

            demand_norm, total_mwh_base = get_demand_profile(iso, demand_data)
            supply_profiles_iso = get_supply_profiles(iso, gen_profiles)

            growth_factor = demand_twh / REGIONAL_DEMAND_TWH[iso]
            demand_mw_profile = np.array(demand_norm, dtype=np.float64) * total_mwh_base * growth_factor

            iso_data = step3_data[iso]
            current_pct = state['clean_pct']

            candidate_thresholds = sorted(t for t in THRESHOLDS if t > current_pct)
            if not candidate_thresholds:
                state['market_stopped'] = True
                continue

            # Annual queue budget
            queue_budget_gw = QUEUE_CAP_GW[conditions['queue_type']][iso]
            years_in_period = 7 if year == 2030 else 5
            queue_remaining_gw = queue_budget_gw * years_in_period

            zone_deployed = False
            zone_results = []
            avg_lmp = WHOLESALE_PRICES[iso]
            p90_lmp = avg_lmp * 1.5
            blended_revenue = 0
            blended_cost = 0
            rev_breakdown = {'energy_rev_mwh': 0, 'capacity_rev_mwh': 0, 'rec_rev_mwh': 0}
            gen_econ = {}
            nuclear_rev = {}

            for t_end in candidate_thresholds:
                if queue_remaining_gw <= 0:
                    break

                if t_end not in iso_data:
                    available = sorted(iso_data.keys())
                    nearest = min(available, key=lambda x: abs(x - t_end))
                    mix_data = iso_data[nearest]
                else:
                    mix_data = iso_data[t_end]

                resource_pcts = mix_data['resource_pcts']

                # Find prior threshold data
                lower_thresholds = sorted(t for t in iso_data if t <= current_pct)
                t_start = lower_thresholds[-1] if lower_thresholds else min(iso_data.keys())
                start_data = iso_data.get(t_start, {})
                start_pcts = start_data.get('resource_pcts', {r: 0 for r in resource_pcts})

                # Delta resources
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

                # --- LMP + GENERATOR ECONOMICS ---
                carbon_price = conditions.get('carbon_price', 0)
                _lmp_key = (iso, t_end, conditions['fuel_level'],
                            conditions['demand_growth'], year, carbon_price)
                if _lmp_cache is not None and _lmp_key in _lmp_cache:
                    hourly_lmp, avg_lmp, p90_lmp, gen_econ = _lmp_cache[_lmp_key]
                else:
                    hourly_lmp, avg_lmp, p90_lmp, gen_econ = compute_lmp_at_threshold(
                        iso, t_end, conditions['fuel_level'],
                        demand_norm, demand_mw_profile,
                        supply_profiles_iso, resource_pcts,
                        battery_pct=mix_data['battery_pct'],
                        battery8_pct=mix_data['battery8_pct'],
                        ldes_pct=mix_data['ldes_pct'],
                        h2_pct=mix_data['h2_pct'],
                        carbon_price=carbon_price,
                    )
                    if _lmp_cache is not None:
                        _lmp_cache[_lmp_key] = (hourly_lmp, avg_lmp, p90_lmp, gen_econ)

                # Nuclear revenue at this threshold
                nuclear_rev = compute_nuclear_revenue(iso, t_end, hourly_lmp, year)

                # --- NUCLEAR RETIREMENT CHECK ---
                if (nuclear_retirement_threshold is not None and
                        not state['nuclear_retired'] and
                        nuclear_rev['total_mwh'] < nuclear_retirement_threshold):
                    state['nuclear_retired'] = True
                    _log(f"  {iso} NUCLEAR RETIRES at {t_end:.0f}% — "
                         f"revenue ${nuclear_rev['total_mwh']:.1f}/MWh "
                         f"< threshold ${nuclear_retirement_threshold:.1f}/MWh")

                # Revenue for delta resources
                blended_revenue, per_res_rev, rev_breakdown = compute_zone_revenue(
                    iso, t_end, delta_pcts, hourly_lmp,
                    supply_profiles_iso, demand_total_mwh, year,
                )

                # Cost
                blended_cost, per_res_cost = compute_zone_cost(
                    iso, delta_twh, conditions['lcoe_level'], cumulative_gw,
                    conditions['learning_speed'], year, conditions['tx_level'],
                    ppa_level=conditions.get('ppa_level'),
                )

                # Profit
                delta_profit = blended_revenue - blended_cost

                # Deploy or stop
                if delta_profit > 0:
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

                    zone_results.append({
                        'threshold': t_end,
                        'revenue': blended_revenue,
                        'cost': blended_cost,
                        'profit': round(delta_profit, 2),
                        'new_gw': round(total_new_gw, 2),
                        'avg_lmp': round(avg_lmp, 1),
                        **rev_breakdown,
                    })

                    _log(f"  {iso} → {t_end:.0f}%: profit={delta_profit:+.1f} $/MWh "
                         f"(rev={blended_revenue:.1f}, cost={blended_cost:.1f}) "
                         f"+{total_new_gw:.1f} GW, LMP avg={avg_lmp:.1f}")
                else:
                    state['market_stopped'] = True
                    _log(f"  {iso} MARKET STOP at {current_pct:.0f}%: "
                         f"profit={delta_profit:+.1f} $/MWh "
                         f"(rev={blended_revenue:.1f}, cost={blended_cost:.1f})")
                    break

            # --- EMISSION ACCOUNTING ---
            gf = demand_twh / REGIONAL_DEMAND_TWH[iso]
            er, _ = compute_fossil_retirement(
                iso, current_pct, emission_rates, fossil_mix, gf)
            fossil_twh = (1 - current_pct / 100.0) * demand_twh
            emissions_mt = fossil_twh * 1e6 * er / 1e6

            # CCS breakeven at this ISO/fuel level
            ccs_breakeven = compute_ccs_retrofit_breakeven(iso, conditions['fuel_level'])

            # Resource mix in TWh
            if current_pct in iso_data:
                final_mix = iso_data[current_pct]['resource_pcts']
            else:
                avail = sorted(iso_data.keys())
                nearest = min(avail, key=lambda x: abs(x - current_pct))
                final_mix = iso_data[nearest]['resource_pcts']
            resource_mix_twh = {r: p / 100.0 * demand_twh for r, p in final_mix.items()}

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
                'zones_deployed': [z['threshold'] for z in zone_results],
                'zone_details': zone_results,
                'generator_economics': gen_econ,
                'nuclear_revenue': nuclear_rev,
                'nuclear_retired': state['nuclear_retired'],
                'ccs_breakeven': ccs_breakeven,
            }
            results[iso].append(year_result)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SWEEP RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_sweep(isos=None, nuclear_retirement_threshold=None,
                    snapshot_mode=False):
    """Run full 270-scenario market sweep.

    Pre-loads data once, shares LMP cache across scenarios.
    """
    t0 = time.time()
    print("Loading common data...")
    demand_data, gen_profiles, emission_rates, fossil_mix = load_common_data()
    print(f"  Common data loaded in {time.time()-t0:.1f}s")

    print("Loading step3 cost optimization parquets...")
    step3_data = load_step3_data()
    egrid_baselines = load_egrid_baselines()

    preloaded = {
        'demand_data': demand_data,
        'gen_profiles': gen_profiles,
        'emission_rates': emission_rates,
        'fossil_mix': fossil_mix,
        'step3_data': step3_data,
        'egrid_baselines': egrid_baselines,
    }

    scenarios = build_market_scenarios()
    print(f"\nRunning {len(scenarios)} market scenarios...")

    lmp_cache = {}
    all_results = {}

    for i, (scenario_id, conditions) in enumerate(scenarios):
        print(f"\n[{i+1}/{len(scenarios)}] {scenario_id}")
        results = run_market_simulation(
            scenario_id, conditions, isos=isos,
            nuclear_retirement_threshold=nuclear_retirement_threshold,
            snapshot_mode=snapshot_mode,
            _preloaded=preloaded,
            _lmp_cache=lmp_cache,
            _quiet=True,
        )
        all_results[scenario_id] = results

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Sweep complete: {len(scenarios)} scenarios in {elapsed:.1f}s")
    print(f"LMP cache hits: {len(lmp_cache)} unique threshold/fuel/demand combos")

    return all_results


def save_results(all_results, output_dir=None):
    """Save results as JSON."""
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
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for results')

    args = parser.parse_args()

    if args.single:
        overrides = {
            'carbon_price': args.carbon_price,
            'fuel_level': args.fuel_level,
            'lcoe_level': args.lcoe_level,
        }
        scenario_id, conditions = build_single_scenario(overrides)
        results = run_market_simulation(
            scenario_id, conditions, isos=args.isos,
            nuclear_retirement_threshold=args.nuclear_retirement,
            snapshot_mode=args.snapshot,
        )
        all_results = {scenario_id: results}
    else:
        all_results = run_full_sweep(
            isos=args.isos,
            nuclear_retirement_threshold=args.nuclear_retirement,
            snapshot_mode=args.snapshot,
        )

    save_results(all_results, args.output_dir)


if __name__ == '__main__':
    main()
