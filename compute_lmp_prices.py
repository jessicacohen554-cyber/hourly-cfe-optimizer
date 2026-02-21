#!/usr/bin/env python3
"""
LMP Price Calculation Module — Synthetic Hourly LMP from Dispatch Reconstruction
=================================================================================
Downstream of Step 4. Reads base case (ECF — existing clean floor) results from
overprocure_scenarios.parquet, reconstructs 8760-hour dispatch for each winning
mix, builds fossil merit-order stack, and computes synthetic hourly LMP.

Three analysis tracks:
  Track 1 (ECF): Existing clean floor — base case with existing generation credited
  Track 2 (NB):  New-build — hydro=0, all existing zeroed, uprates on (from track_results.json)
  Track 3 (CTR): Cost to replace — hydro at existing, everything else zeroed (from track_results.json)

This module runs on ECF only. Tracks NB/CTR can be added later by pointing at
track_results.json with the same pricing engine.

Pipeline position:
  Step 1 (PFS) → Step 2 (EF) → Step 3 (Cost) → Step 4 (Postprocess)
                                                      ↓
                                          compute_lmp_prices.py  ← THIS
                                                      ↓
                              data/lmp/{ISO}_lmp.parquet   (per-ISO output)
                              data/lmp/{ISO}_archetypes.parquet
                              data/lmp/lmp_summary.json

Usage:
  python compute_lmp_prices.py --iso PJM                    # PJM only, all thresholds
  python compute_lmp_prices.py --iso PJM --test             # PJM test: 2025/50%/95%
  python compute_lmp_prices.py                              # All ISOs
  python compute_lmp_prices.py --iso PJM --fuel-level M     # Medium fuel only
"""

import json
import os
import sys
import time
import argparse
import hashlib
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from dispatch_utils import (
    H, ISOS, RESOURCE_TYPES,
    GRID_MIX_SHARES, BASE_DEMAND_TWH,
    WHOLESALE_PRICES, FUEL_ADJUSTMENTS,
    COAL_OIL_RETIREMENT_THRESHOLD, COAL_CAP_TWH, OIL_CAP_TWH,
    load_common_data, get_demand_profile, get_supply_profiles,
    reconstruct_hourly_dispatch, compute_fossil_retirement,
    compute_fossil_capacity_at_threshold,
    load_dispatch_cache, save_dispatch_cache, get_or_compute_dispatch,
)

LMP_DIR = os.path.join(SCRIPT_DIR, 'data', 'lmp')
SCENARIOS_PATH = os.path.join(SCRIPT_DIR, 'dashboard', 'overprocure_scenarios.parquet')

# ══════════════════════════════════════════════════════════════════════════════
# FOSSIL MERIT-ORDER STACK — heat rates, VOM, marginal cost
# ══════════════════════════════════════════════════════════════════════════════

# Heat rates (MMBtu/MWh) — EIA Electric Power Annual Table 8.1
HEAT_RATES = {
    'coal_steam': 10.0,
    'gas_ccgt': 6.4,
    'gas_ct': 10.0,
    'oil_ct': 10.5,
}

# Variable O&M ($/MWh) — EIA AEO / NREL ATB
VOM = {
    'coal_steam': 4.50,
    'gas_ccgt': 2.00,
    'gas_ct': 4.00,
    'oil_ct': 5.00,
}

# Fuel prices ($/MMBtu) by sensitivity level
FUEL_PRICES = {
    'Low':    {'coal': 2.00, 'gas': 2.00, 'oil': 8.00},
    'Medium': {'coal': 2.25, 'gas': 3.50, 'oil': 10.50},
    'High':   {'coal': 2.50, 'gas': 6.00, 'oil': 13.00},
}

# Capacity shares within fossil fleet (fraction of total fossil capacity)
# Derived from EIA 860 — installed capacity shares by fuel type within each ISO
FOSSIL_CAPACITY_SHARES = {
    'CAISO': {'coal_steam': 0.00, 'gas_ccgt': 0.55, 'gas_ct': 0.40, 'oil_ct': 0.05},
    'ERCOT': {'coal_steam': 0.22, 'gas_ccgt': 0.50, 'gas_ct': 0.28, 'oil_ct': 0.00},
    'PJM':   {'coal_steam': 0.23, 'gas_ccgt': 0.48, 'gas_ct': 0.25, 'oil_ct': 0.04},
    'NYISO': {'coal_steam': 0.00, 'gas_ccgt': 0.45, 'gas_ct': 0.50, 'oil_ct': 0.05},
    'NEISO': {'coal_steam': 0.00, 'gas_ccgt': 0.52, 'gas_ct': 0.42, 'oil_ct': 0.06},
}

# Actual installed fossil capacity (MW) — EIA 860 2024
# Used instead of demand-based estimation for more accurate stack sizing
INSTALLED_FOSSIL_MW = {
    'CAISO': 47_000,   # ~47 GW gas fleet
    'ERCOT': 80_000,   # ~80 GW total fossil (gas + coal)
    'PJM':   140_000,  # ~140 GW fossil (gas ~90, coal ~40, oil ~10)
    'NYISO': 28_000,   # ~28 GW fossil (mostly gas)
    'NEISO': 16_000,   # ~16 GW fossil (mostly gas)
}

# Peak demand (MW) — matches step3_cost_optimization.py
PEAK_DEMAND_MW = {
    'CAISO': 43_860, 'ERCOT': 83_597, 'PJM': 160_560, 'NYISO': 31_857, 'NEISO': 25_898,
}

# Resource adequacy reserve margin — 15%, consistent with step3/step4
RESOURCE_ADEQUACY_MARGIN = 0.15

# Peak capacity credits — exact copy from step3_cost_optimization.py
PEAK_CAPACITY_CREDITS = {
    'clean_firm': 1.0,   # nuclear — fully accredited
    'solar': 0.30,       # ELCC — only afternoon contribution
    'wind': 0.10,        # ELCC — low correlation with peak
    'ccs_ccgt': 0.90,    # dispatchable
    'hydro': 0.50,       # seasonal/capacity-limited
    'battery': 0.95,     # 4hr Li-ion
    'battery8': 0.95,    # 8hr Li-ion
    'ldes': 0.90,        # 100hr iron-air
}

# Gas Availability Factor (GAF) — forced outages + correlated weather risk
# From step3: gas_needed_mw / GAF = nameplate needed to deliver firm capacity
GAS_AVAILABILITY_FACTOR = {
    'CAISO': 0.88,  # 12% deration — summer ambient + mechanical
    'ERCOT': 0.83,  # 17% deration — extreme weather both seasons
    'PJM':   0.82,  # 18% deration — PJM ELCC data, Winter Storm Elliott
    'NYISO': 0.82,  # 18% deration — pipeline constraints, winter gas
    'NEISO': 0.85,  # 15% deration — mechanical + weather (pipeline separate)
}


def compute_marginal_costs(fuel_level='Medium'):
    """Compute marginal cost ($/MWh) for each fossil unit type at given fuel prices."""
    fp = FUEL_PRICES[fuel_level]
    costs = {}
    costs['coal_steam'] = HEAT_RATES['coal_steam'] * fp['coal'] + VOM['coal_steam']
    costs['gas_ccgt'] = HEAT_RATES['gas_ccgt'] * fp['gas'] + VOM['gas_ccgt']
    costs['gas_ct'] = HEAT_RATES['gas_ct'] * fp['gas'] + VOM['gas_ct']
    costs['oil_ct'] = HEAT_RATES['oil_ct'] * fp['oil'] + VOM['oil_ct']
    return costs


def _compute_clean_peak_mw(iso, resource_mix, procurement_pct, battery_pct=0,
                           battery8_pct=0, ldes_pct=0):
    """Compute clean peak capacity contribution (MW) from resource mix.

    Mirrors step3_cost_optimization.py clean_peak_mw calculation exactly.
    Uses per-resource ELCC capacity credits at system peak.
    """
    peak_mw = PEAK_DEMAND_MW.get(iso, 80_000)
    demand_twh = BASE_DEMAND_TWH.get(iso, 0)
    avg_demand_mw = (demand_twh * 1e6) / H  # TWh → MWh / 8760 → avg MW

    proc = procurement_pct / 100.0
    clean_peak = (
        proc * resource_mix.get('clean_firm', 0) / 100.0 * avg_demand_mw * PEAK_CAPACITY_CREDITS['clean_firm'] +
        proc * resource_mix.get('solar', 0) / 100.0 * avg_demand_mw * PEAK_CAPACITY_CREDITS['solar'] +
        proc * resource_mix.get('wind', 0) / 100.0 * avg_demand_mw * PEAK_CAPACITY_CREDITS['wind'] +
        proc * resource_mix.get('ccs_ccgt', 0) / 100.0 * avg_demand_mw * PEAK_CAPACITY_CREDITS['ccs_ccgt'] +
        proc * resource_mix.get('hydro', 0) / 100.0 * avg_demand_mw * PEAK_CAPACITY_CREDITS['hydro'] +
        battery_pct / 100.0 * avg_demand_mw * PEAK_CAPACITY_CREDITS['battery'] +
        battery8_pct / 100.0 * avg_demand_mw * PEAK_CAPACITY_CREDITS['battery8'] +
        ldes_pct / 100.0 * avg_demand_mw * PEAK_CAPACITY_CREDITS['ldes']
    )
    return clean_peak


def build_merit_order_stack(iso, clean_pct, fuel_level='Medium', total_fossil_mw=None,
                             resource_mix=None, procurement_pct=100,
                             battery_pct=0, battery8_pct=0, ldes_pct=0):
    """Build merit-order stack: list of (unit_type, capacity_mw, marginal_cost).

    Ordered by marginal cost (cheapest first). Stack composition reflects
    retirement model: coal retires first, then oil, then gas.

    Fossil fleet is sized with a 15% RA reserve margin above peak residual demand,
    GAF-derated for gas availability — consistent with step3/step4. ISOs don't
    decommission below what's needed for reliability.

    Args:
        iso: ISO region
        clean_pct: clean energy threshold (determines retirements)
        fuel_level: 'Low', 'Medium', 'High'
        total_fossil_mw: total fossil capacity in MW (if None, RA+GAF estimate)
        resource_mix: dict with clean resource percentages (for ELCC calculation)
        procurement_pct: procurement level (%)
        battery_pct: battery dispatch percentage
        battery8_pct: battery8 dispatch percentage
        ldes_pct: LDES dispatch percentage

    Returns:
        stack: list of (unit_type, capacity_mw, marginal_cost_per_mwh)
        total_capacity_mw: total fossil MW
    """
    mc = compute_marginal_costs(fuel_level)

    if total_fossil_mw is None:
        installed = INSTALLED_FOSSIL_MW.get(iso, 80_000)
        baseline_clean = sum(GRID_MIX_SHARES.get(iso, {}).values())

        if clean_pct <= baseline_clean:
            total_fossil_mw = installed
        else:
            # RA-aware fleet sizing with GAF deration (matches step3 formula)
            peak_mw = PEAK_DEMAND_MW.get(iso, 80_000)
            ra_peak_mw = peak_mw * (1 + RESOURCE_ADEQUACY_MARGIN)

            # Compute clean peak MW using actual resource mix when available
            if resource_mix is not None:
                clean_peak_mw = _compute_clean_peak_mw(
                    iso, resource_mix, procurement_pct,
                    battery_pct, battery8_pct, ldes_pct)
            else:
                # Fallback: estimate from clean_pct with conservative blended credit
                # At low clean%, mix is mostly solar/wind (low ELCC ~0.25)
                # At high clean%, mix is dominated by clean_firm (ELCC ~1.0)
                # Sigmoid-like blend: transitions from 0.25 to 0.80 across 50-100%
                t = max(0, min(1, (clean_pct - 50) / 50))
                blended_credit = 0.25 + 0.55 * t
                avg_demand_mw = (BASE_DEMAND_TWH.get(iso, 0) * 1e6) / H
                clean_peak_mw = (clean_pct / 100.0) * avg_demand_mw * blended_credit

            # Residual peak demand that fossil must serve
            residual_peak_mw = max(0, ra_peak_mw - clean_peak_mw)

            # GAF deration: not all gas is available at peak (forced outages,
            # weather correlation). Need more nameplate MW to deliver firm capacity.
            gaf = GAS_AVAILABILITY_FACTOR.get(iso, 0.85)
            ra_floor_mw = residual_peak_mw / gaf

            # Linear retirement gives a supply-side estimate
            fossil_fraction = max(0.05, (100.0 - clean_pct) / (100.0 - baseline_clean))
            linear_mw = installed * fossil_fraction

            # Take the higher of RA floor and linear — fleet doesn't retire
            # below reliability requirement, but also doesn't magically grow
            total_fossil_mw = min(installed, max(ra_floor_mw, linear_mw))

    shares = FOSSIL_CAPACITY_SHARES.get(iso, FOSSIL_CAPACITY_SHARES['PJM'])

    # Apply retirement model
    if clean_pct >= COAL_OIL_RETIREMENT_THRESHOLD:
        # All coal and oil retired
        active_shares = {
            'gas_ccgt': shares.get('gas_ccgt', 0.5),
            'gas_ct': shares.get('gas_ct', 0.5),
        }
        # Renormalize
        total = sum(active_shares.values())
        if total > 0:
            active_shares = {k: v / total for k, v in active_shares.items()}
    else:
        active_shares = dict(shares)

    # Build stack: list of (type, capacity_mw, mc)
    stack = []
    for unit_type, share in active_shares.items():
        if share <= 0:
            continue
        cap_mw = total_fossil_mw * share
        stack.append((unit_type, cap_mw, mc[unit_type]))

    # Sort by marginal cost (cheapest first)
    stack.sort(key=lambda x: x[2])

    return stack, total_fossil_mw


# ══════════════════════════════════════════════════════════════════════════════
# ISO-SPECIFIC PRICE FORMATION MODELS
# ══════════════════════════════════════════════════════════════════════════════

class PriceModel:
    """Base price formation model. ISO-specific subclasses override parameters."""

    def __init__(self, iso, fuel_level='Medium'):
        self.iso = iso
        self.fuel_level = fuel_level
        self.scarcity_cap = 2000.0     # $/MWh cap during scarcity
        self.floor_price = -30.0       # $/MWh floor during surplus
        self.surplus_slope = 0.5       # steepness of negative price curve
        self.surplus_decay = 0.02      # decay rate for surplus pricing
        self.rt_sensitivity_factor = 1.0  # scale for RT volatility
        self.scarcity_threshold = 0.05    # reserves/demand ratio triggering scarcity

    def price_hour(self, residual_demand_mw, demand_mw, stack, surplus_mw=0.0):
        """Compute LMP for a single hour given residual demand and merit-order stack.

        Args:
            residual_demand_mw: MW of demand that must be met by fossil (>0)
                                or surplus clean energy (<0)
            demand_mw: total demand MW this hour (for reserve ratio)
            stack: merit-order stack from build_merit_order_stack()
            surplus_mw: MW of clean surplus available (for negative pricing)

        Returns:
            lmp: $/MWh price
            marginal_unit: index into stack of the marginal unit (or -1 for surplus)
        """
        if residual_demand_mw <= 0:
            # Clean surplus — negative/zero prices
            return self._price_surplus(-residual_demand_mw, demand_mw), -1

        # Walk the merit-order stack with np.searchsorted-style step function
        cumulative_mw = 0.0
        marginal_unit = -1
        marginal_cost = 0.0

        for i, (unit_type, cap_mw, mc) in enumerate(stack):
            cumulative_mw += cap_mw
            if cumulative_mw >= residual_demand_mw:
                marginal_unit = i
                marginal_cost = mc
                break

        if marginal_unit == -1:
            # Demand exceeds all capacity — scarcity pricing
            return self._price_scarcity(residual_demand_mw, cumulative_mw, demand_mw), len(stack)

        # Check reserve margin — partial scarcity adder
        remaining_capacity = cumulative_mw - residual_demand_mw
        reserve_ratio = remaining_capacity / demand_mw if demand_mw > 0 else 1.0

        if reserve_ratio < self.scarcity_threshold:
            scarcity_adder = self._scarcity_adder(reserve_ratio, demand_mw)
            return marginal_cost + scarcity_adder, marginal_unit

        return marginal_cost, marginal_unit

    def _price_surplus(self, surplus_mw, demand_mw):
        """Compute price during clean energy surplus. Returns $/MWh."""
        if demand_mw <= 0:
            return 0.0
        surplus_ratio = surplus_mw / demand_mw
        # Empirical curve: price decays from 0 toward floor as surplus grows
        price = self.floor_price * (1 - np.exp(-self.surplus_decay * surplus_ratio * 100))
        return max(self.floor_price, price)

    def _price_scarcity(self, demand_mw, available_mw, total_demand_mw):
        """Compute price when demand exceeds available capacity."""
        if available_mw <= 0:
            return self.scarcity_cap
        # Scarcity price scales with shortage severity
        shortage_ratio = (demand_mw - available_mw) / max(1.0, available_mw)
        # Starts at top marginal cost + small adder, ramps toward cap
        base = 100.0  # $/MWh — top of normal merit order
        price = base + (self.scarcity_cap - base) * min(1.0, shortage_ratio * 2.0)
        return min(self.scarcity_cap, price)

    def _scarcity_adder(self, reserve_ratio, demand_mw):
        """Scarcity adder as reserves decline. Base implementation: penalty factor."""
        if reserve_ratio >= self.scarcity_threshold:
            return 0.0
        # Exponential ramp — mild adder until very low reserves
        fraction = 1.0 - (reserve_ratio / self.scarcity_threshold)
        # Only 5-15% of cap at moderate shortage, ramps to 30% at zero
        return self.scarcity_cap * (fraction ** 2) * 0.15


class PJMPriceModel(PriceModel):
    """PJM: RPM capacity market, penalty factor scarcity, moderate negative prices."""

    def __init__(self, fuel_level='Medium'):
        super().__init__('PJM', fuel_level)
        self.scarcity_cap = 2000.0
        self.floor_price = -30.0
        self.surplus_slope = 0.4
        self.surplus_decay = 0.015
        self.scarcity_threshold = 0.03  # PJM has large reserves; scarcity is rare
        self.coal_min_gen_fraction = 0.4


class ERCOTPriceModel(PriceModel):
    """ERCOT: Energy-only market with ORDC (Operating Reserve Demand Curve).

    ORDC: adder = VOLL × LOLP(reserves). LOLP increases exponentially as
    reserves drop below ~3,000 MW. VOLL = $5,000/MWh (post-2023 reform).
    """

    def __init__(self, fuel_level='Medium'):
        super().__init__('ERCOT', fuel_level)
        self.scarcity_cap = 5000.0
        self.floor_price = -50.0
        self.surplus_decay = 0.025
        self.voll = 5000.0
        self.ordc_knee_mw = 3000.0  # reserves below this trigger exponential ORDC

    def _scarcity_adder(self, reserve_ratio, demand_mw):
        """ERCOT ORDC: smooth exponential adder, not a hard cap."""
        reserve_mw = reserve_ratio * demand_mw
        if reserve_mw >= self.ordc_knee_mw:
            return 0.0
        # Exponential LOLP curve: LOLP ≈ exp(-λ × reserve_mw)
        lam = 0.002  # calibratable
        lolp = np.exp(-lam * max(0, reserve_mw))
        return self.voll * lolp


class CAISOPriceModel(PriceModel):
    """CAISO: Resource Adequacy, aggressive negative prices (solar duck curve)."""

    def __init__(self, fuel_level='Medium'):
        super().__init__('CAISO', fuel_level)
        self.scarcity_cap = 2000.0
        self.floor_price = -60.0
        self.surplus_decay = 0.030  # steeper negative prices (more solar)
        self.scarcity_threshold = 0.05


class NYISOPriceModel(PriceModel):
    """NYISO: ICAP capacity market. Similar to PJM but tighter geography."""

    def __init__(self, fuel_level='Medium'):
        super().__init__('NYISO', fuel_level)
        self.scarcity_cap = 2000.0
        self.floor_price = -20.0
        self.surplus_decay = 0.012
        self.scarcity_threshold = 0.06


class NEISOPriceModel(PriceModel):
    """NEISO: FCM capacity market. Winter gas pipeline constraint creates scarcity."""

    WINTER_MONTHS = {12, 1, 2}  # Dec, Jan, Feb
    NEISO_WINTER_GAS_ADDER = 13.13  # $/MWh — from Step 4

    def __init__(self, fuel_level='Medium'):
        super().__init__('NEISO', fuel_level)
        self.scarcity_cap = 2000.0
        self.floor_price = -25.0
        self.surplus_decay = 0.012
        self.scarcity_threshold = 0.07  # tighter than PJM — winter gas

    def price_hour(self, residual_demand_mw, demand_mw, stack, surplus_mw=0.0,
                   hour_of_year=0):
        """Override to add winter gas pipeline constraint."""
        lmp, marginal_unit = super().price_hour(
            residual_demand_mw, demand_mw, stack, surplus_mw)

        # Winter gas adder: Dec-Feb hours get pipeline constraint premium
        month = _hour_to_month(hour_of_year)
        if month in self.WINTER_MONTHS and residual_demand_mw > 0:
            lmp += self.NEISO_WINTER_GAS_ADDER

        return lmp, marginal_unit


def _hour_to_month(hour):
    """Convert hour-of-year (0-8759) to month (1-12)."""
    month_hours = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
    cumulative = 0
    for m, mh in enumerate(month_hours, 1):
        cumulative += mh
        if hour < cumulative:
            return m
    return 12


def get_price_model(iso, fuel_level='Medium'):
    """Factory: return ISO-specific price model."""
    models = {
        'PJM': PJMPriceModel,
        'ERCOT': ERCOTPriceModel,
        'CAISO': CAISOPriceModel,
        'NYISO': NYISOPriceModel,
        'NEISO': NEISOPriceModel,
    }
    cls = models.get(iso, PriceModel)
    return cls(fuel_level)


# ══════════════════════════════════════════════════════════════════════════════
# ARCHETYPE DEDUP — unique (mix, fuel_level, threshold) combos
# ══════════════════════════════════════════════════════════════════════════════

def archetype_key(mix, fuel_level, threshold):
    """Deterministic key for deduplicating dispatch computations.

    Key: (mix_tuple, fuel_level, threshold) — threshold affects fossil stack
    (retirement changes available capacity).
    """
    mix_tuple = (
        mix.get('clean_firm', 0), mix.get('solar', 0), mix.get('wind', 0),
        mix.get('ccs_ccgt', 0), mix.get('hydro', 0),
    )
    return f"{mix_tuple}_{fuel_level}_{threshold}"


# ══════════════════════════════════════════════════════════════════════════════
# HOURLY LMP COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_hourly_lmp(dispatch_result, demand_mw_profile, stack, price_model,
                        iso=None):
    """Compute 8760 hourly LMP from dispatch result and merit-order stack.

    Args:
        dispatch_result: output of reconstruct_hourly_dispatch()
        demand_mw_profile: (H,) array of hourly demand in MW
        stack: merit-order stack from build_merit_order_stack()
        price_model: ISO-specific PriceModel instance
        iso: ISO name (for NEISO winter handling)

    Returns:
        hourly_lmp: (H,) array of $/MWh
        hourly_marginal_unit: (H,) array of int (index into stack, -1 for surplus)
    """
    residual_demand = dispatch_result['residual_demand']  # normalized
    demand_sum = demand_mw_profile.sum()

    hourly_lmp = np.zeros(H, dtype=np.float64)
    hourly_marginal_unit = np.full(H, -1, dtype=np.int8)

    for h in range(H):
        demand_mw = demand_mw_profile[h]
        # Convert normalized residual to MW
        if demand_sum > 0:
            residual_mw = residual_demand[h] * demand_mw / max(1e-10, dispatch_result['residual_demand'].sum() / H * H / demand_sum)
        else:
            residual_mw = 0.0

        # Simpler: residual_demand is fraction of normalized demand
        # Scale to MW directly
        residual_frac = residual_demand[h]
        total_demand_norm = np.array(demand_mw_profile).sum() / H  # avg MW
        residual_mw = residual_frac * (demand_mw_profile.sum() / max(1e-10, sum(dispatch_result['residual_demand']) + sum(dispatch_result['fossil_displaced']))) * demand_mw

        surplus_mw = dispatch_result['curtailed'][h] * demand_mw_profile.sum() / H if dispatch_result['curtailed'][h] > 0 else 0.0

        if isinstance(price_model, NEISOPriceModel):
            lmp, mu = price_model.price_hour(residual_mw, demand_mw, stack,
                                              surplus_mw, hour_of_year=h)
        else:
            lmp, mu = price_model.price_hour(residual_mw, demand_mw, stack, surplus_mw)

        hourly_lmp[h] = lmp
        hourly_marginal_unit[h] = mu

    return hourly_lmp, hourly_marginal_unit


def compute_hourly_lmp_vectorized(dispatch_result, demand_mw_profile, stack, price_model,
                                   iso=None):
    """Vectorized LMP computation — faster than per-hour loop.

    Uses the merit-order stack as a step function and np.searchsorted for
    marginal unit identification.
    """
    # Build cumulative capacity and marginal cost arrays from stack
    n_units = len(stack)
    if n_units == 0:
        return np.zeros(H, dtype=np.float64), np.full(H, -1, dtype=np.int8)

    cum_capacity = np.zeros(n_units, dtype=np.float64)
    marginal_costs = np.zeros(n_units, dtype=np.float64)
    running = 0.0
    for i, (unit_type, cap_mw, mc) in enumerate(stack):
        running += cap_mw
        cum_capacity[i] = running
        marginal_costs[i] = mc

    total_fossil_cap = cum_capacity[-1] if n_units > 0 else 0.0

    # Convert residual demand from normalized to MW
    # residual_demand is in same normalized units as demand_norm
    # Scale: residual_mw[h] = residual_demand[h] / demand_norm[h] * demand_mw[h]
    # But demand_norm is a fraction summing to 1, demand_mw is in MW
    # So: residual_mw = residual_demand * total_annual_mwh / H... but that overcounts
    # Better: residual_demand[h] represents the fraction of hourly demand not met by clean
    # residual_mw[h] = residual_demand[h] * (demand_mw_profile.sum() / demand_norm_sum)
    # where demand_norm is the normalized profile used in dispatch

    # The dispatch uses normalized demand where sum(demand_norm) ≈ 1.0
    # So: demand_mw[h] = demand_norm[h] * total_annual_mwh
    # And: residual_mw[h] = residual_demand_norm[h] * total_annual_mwh
    total_annual_mwh = demand_mw_profile.sum()  # sum of hourly MW = total MWh

    residual_norm = dispatch_result['residual_demand']
    curtailed_norm = dispatch_result['curtailed']

    # residual_demand_norm and demand_norm are on the same scale
    # Scale factor: total_annual_mwh converts normalized to MWh
    residual_mw = residual_norm * total_annual_mwh
    surplus_mw = curtailed_norm * total_annual_mwh

    hourly_lmp = np.zeros(H, dtype=np.float64)
    hourly_marginal_unit = np.full(H, -1, dtype=np.int8)

    # Positive residual: use searchsorted on cumulative capacity
    pos_mask = residual_mw > 0
    if pos_mask.any():
        pos_residual = residual_mw[pos_mask]
        # Find marginal unit: first unit where cumulative capacity >= residual demand
        unit_idx = np.searchsorted(cum_capacity, pos_residual, side='left')
        # Clamp to valid range
        unit_idx = np.clip(unit_idx, 0, n_units - 1)

        # Check for scarcity (demand exceeds all capacity)
        scarcity_mask = pos_residual > total_fossil_cap
        normal_mask = ~scarcity_mask

        # Normal pricing: marginal cost with load-dependent heat rate ramp
        if normal_mask.any():
            normal_idx = unit_idx[normal_mask]
            base_prices = marginal_costs[normal_idx].copy()

            # Load-dependent marginal cost ramp: as utilization within a unit's
            # capacity band increases, heat rate curves push marginal cost up.
            # This creates price variation instead of flat step-function prices.
            # Ramp factor: 0-15% above base cost depending on position within band.
            normal_residual = pos_residual[normal_mask]
            band_start = np.where(normal_idx > 0, cum_capacity[normal_idx - 1], 0.0)
            band_capacity = cum_capacity[normal_idx] - band_start
            position_in_band = np.where(
                band_capacity > 0,
                (normal_residual - band_start) / band_capacity,
                0.5)
            position_in_band = np.clip(position_in_band, 0.0, 1.0)
            # Quadratic ramp: steeper at high utilization (realistic heat rate curve)
            # Gas CCGT heat rate varies ~6.0-7.5 MMBtu/MWh across load range (~25%)
            # Gas CT varies ~9.0-12.0 (~33%). Coal ~9.5-11.0 (~16%).
            heat_rate_ramp = 1.0 + 0.30 * position_in_band ** 1.5
            normal_prices = base_prices * heat_rate_ramp

            # Reserve margin check for scarcity adder
            pos_demand = demand_mw_profile[pos_mask][normal_mask]
            remaining_cap = cum_capacity[normal_idx] - normal_residual
            reserve_ratio = np.where(pos_demand > 0, remaining_cap / pos_demand, 1.0)

            low_reserve = reserve_ratio < price_model.scarcity_threshold
            if low_reserve.any():
                for j in np.where(low_reserve)[0]:
                    normal_prices[j] += price_model._scarcity_adder(
                        float(reserve_ratio[j]), float(pos_demand[j]))

            pos_indices = np.where(pos_mask)[0]
            normal_global = pos_indices[normal_mask]
            hourly_lmp[normal_global] = normal_prices
            hourly_marginal_unit[normal_global] = normal_idx.astype(np.int8)

        # Scarcity pricing
        if scarcity_mask.any():
            pos_indices = np.where(pos_mask)[0]
            scarcity_global = pos_indices[scarcity_mask]
            for j, gi in enumerate(scarcity_global):
                hourly_lmp[gi] = price_model._price_scarcity(
                    float(residual_mw[gi]),
                    float(total_fossil_cap),
                    float(demand_mw_profile[gi]))
                hourly_marginal_unit[gi] = n_units

    # Negative/zero residual (surplus): negative pricing
    neg_mask = residual_mw <= 0
    if neg_mask.any():
        neg_indices = np.where(neg_mask)[0]
        for gi in neg_indices:
            hourly_lmp[gi] = price_model._price_surplus(
                float(surplus_mw[gi]), float(demand_mw_profile[gi]))
            hourly_marginal_unit[gi] = -1

    # Must-run baseload effect: nuclear and some coal can't economically ramp
    # down. During low-demand hours, must-run output pushes prices toward
    # or below zero. In PJM, ~30-40 GW of nuclear runs 24/7. When overnight
    # demand drops to ~60 GW, nuclear alone covers a large share.
    if n_units > 0 and len(stack) > 0:
        cheapest_mc = marginal_costs[0]
        # Absolute must-run level (MW) — nuclear + inflexible coal
        baseline_shares = GRID_MIX_SHARES.get(iso or 'PJM', {})
        nuclear_pct = baseline_shares.get('clean_firm', 0) / 100.0
        avg_demand = demand_mw_profile.mean()
        must_run_mw_level = avg_demand * nuclear_pct * 0.95  # nuclear at ~95% CF

        # Hours where total fossil demand is less than must-run surplus
        # (i.e., total demand - clean supply < must_run_level means clean + must_run > demand)
        fossil_demand = residual_mw.copy()
        total_demand = demand_mw_profile
        effective_surplus = must_run_mw_level - fossil_demand  # positive = must-run excess

        # Only affect hours where must-run creates genuine surplus pressure
        surplus_hours = (effective_surplus > 0) & (fossil_demand > 0)
        if surplus_hours.any():
            surplus_ratio = effective_surplus[surplus_hours] / must_run_mw_level
            surplus_ratio = np.clip(surplus_ratio, 0, 1)
            # Prices compress: from cheapest MC down toward floor
            depressed = cheapest_mc * (1 - surplus_ratio) + price_model.floor_price * surplus_ratio * 0.3
            # Only depress prices, never increase them
            hourly_lmp[surplus_hours] = np.minimum(hourly_lmp[surplus_hours], depressed)

    # NEISO winter gas adder
    if isinstance(price_model, NEISOPriceModel):
        month_hours = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
        h = 0
        for m_idx, mh in enumerate(month_hours):
            month = m_idx + 1
            if month in NEISOPriceModel.WINTER_MONTHS:
                winter_slice = slice(h, h + mh)
                winter_pos = pos_mask[winter_slice]
                hourly_lmp[winter_slice] = np.where(
                    winter_pos,
                    hourly_lmp[winter_slice] + NEISOPriceModel.NEISO_WINTER_GAS_ADDER,
                    hourly_lmp[winter_slice])
            h += mh

    return hourly_lmp, hourly_marginal_unit


# ══════════════════════════════════════════════════════════════════════════════
# LMP STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_lmp_stats(hourly_lmp, hourly_marginal_unit, demand_mw_profile,
                       dispatch_result):
    """Compute summary statistics from 8760 hourly LMP array.

    Returns dict matching the output schema in SPEC.md.
    """
    # Peak/off-peak classification (7am-11pm weekdays)
    peak_mask = np.zeros(H, dtype=bool)
    month_hours = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
    h = 0
    for day in range(365):
        dow = day % 7  # 0=Mon (Jan 1 2025 is Wednesday → adjust)
        # 2025: Jan 1 = Wednesday (dow=2)
        actual_dow = (day + 2) % 7  # 0=Mon, ..., 6=Sun
        is_weekday = actual_dow < 5
        for hour_of_day in range(24):
            if h < H and is_weekday and 7 <= hour_of_day <= 22:
                peak_mask[h] = True
            h += 1

    offpeak_mask = ~peak_mask

    # Time-weighted average
    avg_lmp = float(np.mean(hourly_lmp))
    peak_avg = float(np.mean(hourly_lmp[peak_mask])) if peak_mask.any() else avg_lmp
    offpeak_avg = float(np.mean(hourly_lmp[offpeak_mask])) if offpeak_mask.any() else avg_lmp

    # Price hours
    zero_price_hours = int(np.sum(hourly_lmp <= 0))
    negative_price_hours = int(np.sum(hourly_lmp < 0))

    # Scarcity hours (> $200/MWh as proxy)
    scarcity_hours = int(np.sum(hourly_lmp > 200))

    # Percentiles
    p10, p25, p50, p75, p90 = np.percentile(hourly_lmp, [10, 25, 50, 75, 90])

    # Volatility
    volatility = float(np.std(hourly_lmp))

    # Duck curve depth: max surplus MW
    surplus = dispatch_result['curtailed']
    total_mwh = demand_mw_profile.sum()
    duck_curve_depth = float(np.max(surplus) * total_mwh) if surplus.max() > 0 else 0.0

    # Net peak price: price at hour of max net demand (residual)
    residual = dispatch_result['residual_demand']
    max_residual_hour = int(np.argmax(residual))
    net_peak_price = float(hourly_lmp[max_residual_hour])

    # Fossil revenue: average $/MWh earned by fossil generators
    fossil_hours = hourly_lmp[residual > 0]
    fossil_revenue = float(np.mean(fossil_hours)) if len(fossil_hours) > 0 else 0.0

    return {
        'avg_lmp': round(avg_lmp, 2),
        'peak_avg_lmp': round(peak_avg, 2),
        'offpeak_avg_lmp': round(offpeak_avg, 2),
        'zero_price_hours': zero_price_hours,
        'negative_price_hours': negative_price_hours,
        'scarcity_hours': scarcity_hours,
        'lmp_p10': round(float(p10), 2),
        'lmp_p25': round(float(p25), 2),
        'lmp_p50': round(float(p50), 2),
        'lmp_p75': round(float(p75), 2),
        'lmp_p90': round(float(p90), 2),
        'price_volatility': round(volatility, 2),
        'duck_curve_depth_mw': round(duck_curve_depth, 0),
        'net_peak_price': round(net_peak_price, 2),
        'fossil_revenue_mwh': round(fossil_revenue, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def load_scenarios(iso=None, threshold=None):
    """Load base case (ECF) scenarios from overprocure_scenarios.parquet.

    Returns list of scenario dicts with resource mix, dispatch params, costs.
    """
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    table = pq.read_table(SCENARIOS_PATH)

    if iso:
        table = table.filter(pc.equal(table.column('iso'), iso))
    if threshold is not None:
        table = table.filter(pc.equal(table.column('threshold'), float(threshold)))

    rows = []
    for i in range(table.num_rows):
        row = {col: table.column(col)[i].as_py() for col in table.column_names}
        # Reconstruct resource_mix dict
        row['resource_mix'] = {
            'clean_firm': row.get('mix_clean_firm', 0),
            'solar': row.get('mix_solar', 0),
            'wind': row.get('mix_wind', 0),
            'ccs_ccgt': row.get('mix_ccs_ccgt', 0),
            'hydro': row.get('mix_hydro', 0),
        }
        rows.append(row)

    return rows


def extract_fuel_level(scenario_key):
    """Extract fuel level from 9-dim scenario key (e.g., 'MMMM_M_M_M1_X' → 'M')."""
    # Key format: RFBL_FF_TX_CCSq45_GEO
    # FF position = index 1 after first '_' split
    parts = scenario_key.split('_')
    if len(parts) >= 2:
        return parts[1]  # fuel/fossil toggle
    return 'M'


def fuel_code_to_level(code):
    """Convert fuel sensitivity code to level name."""
    return {'L': 'Low', 'M': 'Medium', 'H': 'High'}.get(code, 'Medium')


def run_lmp_for_iso(iso, scenarios, demand_data, gen_profiles,
                     fuel_level='Medium', dispatch_cache=None):
    """Compute LMP for all scenarios for a single ISO.

    Args:
        iso: ISO region
        scenarios: list of scenario dicts from load_scenarios()
        demand_data: demand profile data
        gen_profiles: generation profile data
        fuel_level: 'Low', 'Medium', 'High'
        dispatch_cache: optional mutable cache dict for reuse

    Returns:
        results: list of dicts with LMP stats per (threshold, scenario)
        archetypes: dict of {archetype_key: {'hourly_lmp': array, ...}}
    """
    demand_norm, total_mwh = get_demand_profile(iso, demand_data)
    supply_profiles = get_supply_profiles(iso, gen_profiles)

    # Convert normalized demand to MW profile
    demand_mw_profile = demand_norm * total_mwh

    price_model = get_price_model(iso, fuel_level)

    results = []
    archetypes = {}
    seen_archetypes = set()

    cache_hits = 0
    cache_misses = 0

    for sc in scenarios:
        threshold = sc['threshold']
        scenario_key = sc.get('scenario', '')
        resource_mix = sc['resource_mix']
        procurement_pct = sc.get('procurement_pct', 100)
        batt4 = sc.get('battery_dispatch_pct', 0)
        batt8 = sc.get('battery8_dispatch_pct', 0)
        ldes = sc.get('ldes_dispatch_pct', 0)

        # Archetype dedup
        akey = archetype_key(resource_mix, fuel_level, threshold)

        if akey in seen_archetypes:
            # Reuse existing archetype stats
            existing = archetypes[akey]
            results.append({
                'iso': iso,
                'threshold': threshold,
                'scenario': scenario_key,
                'archetype_key': akey,
                'fuel_level': fuel_level,
                **existing['stats'],
            })
            continue
        seen_archetypes.add(akey)

        # Get or compute dispatch
        dispatch, cache_hit = get_or_compute_dispatch(
            iso, demand_norm, supply_profiles, resource_mix,
            procurement_pct, batt4, batt8, ldes,
            cache=dispatch_cache)

        if cache_hit:
            cache_hits += 1
        else:
            cache_misses += 1

        # Build merit-order stack for this threshold (RA+GAF aware)
        stack, total_fossil_mw = build_merit_order_stack(
            iso, threshold, fuel_level,
            resource_mix=resource_mix, procurement_pct=procurement_pct,
            battery_pct=batt4, battery8_pct=batt8, ldes_pct=ldes)

        # Compute hourly LMP
        hourly_lmp, hourly_mu = compute_hourly_lmp_vectorized(
            dispatch, demand_mw_profile, stack, price_model, iso)

        # Compute stats
        stats = compute_lmp_stats(hourly_lmp, hourly_mu, demand_mw_profile, dispatch)

        # Store archetype
        archetypes[akey] = {
            'hourly_lmp': hourly_lmp,
            'hourly_residual_mw': dispatch['residual_demand'] * total_mwh,
            'hourly_marginal_unit': hourly_mu,
            'stats': stats,
            'threshold': threshold,
            'fuel_level': fuel_level,
        }

        results.append({
            'iso': iso,
            'threshold': threshold,
            'scenario': scenario_key,
            'archetype_key': akey,
            'fuel_level': fuel_level,
            **stats,
        })

    print(f"    Dispatch cache: {cache_hits} hits, {cache_misses} misses")
    return results, archetypes


def save_iso_results(iso, results, archetypes):
    """Save LMP results and archetype profiles for an ISO."""
    import pandas as pd

    os.makedirs(LMP_DIR, exist_ok=True)

    # Stats parquet
    if results:
        df = pd.DataFrame(results)
        stats_path = os.path.join(LMP_DIR, f'{iso}_lmp.parquet')
        df.to_parquet(stats_path, index=False, compression='zstd')
        print(f"    {iso}_lmp.parquet: {len(df)} rows, "
              f"{os.path.getsize(stats_path) / 1024:.0f} KB")

    # Archetype profiles parquet (8760 arrays as list columns)
    if archetypes:
        arch_rows = []
        for akey, arch in archetypes.items():
            arch_rows.append({
                'archetype_key': akey,
                'threshold': arch['threshold'],
                'fuel_level': arch['fuel_level'],
                'hourly_lmp': arch['hourly_lmp'].tolist(),
                'hourly_residual_mw': arch['hourly_residual_mw'].tolist(),
                'hourly_marginal_unit': arch['hourly_marginal_unit'].tolist(),
            })
        df_arch = pd.DataFrame(arch_rows)
        arch_path = os.path.join(LMP_DIR, f'{iso}_archetypes.parquet')
        df_arch.to_parquet(arch_path, index=False, compression='zstd')
        print(f"    {iso}_archetypes.parquet: {len(df_arch)} archetypes, "
              f"{os.path.getsize(arch_path) / (1024*1024):.1f} MB")


def run_test_cases(iso='PJM'):
    """Run 3 test cases for validation: 2025 baseline, 2032 50%, 2045 95%."""
    print(f"\n{'='*70}")
    print(f"  LMP TEST CASES — {iso}")
    print(f"{'='*70}")

    demand_data, gen_profiles, emission_rates, fossil_mix = load_common_data()
    demand_norm, total_mwh = get_demand_profile(iso, demand_data)
    supply_profiles = get_supply_profiles(iso, gen_profiles)
    demand_mw_profile = demand_norm * total_mwh

    baseline_clean = sum(GRID_MIX_SHARES.get(iso, {}).values())
    fuel_level = 'Medium'

    test_cases = [
        {'label': f'2025 Baseline ({baseline_clean:.1f}% clean)',
         'threshold': baseline_clean,
         'year': 2025},
        {'label': '2032 Target (50% clean)',
         'threshold': 50.0,
         'year': 2032},
        {'label': '2045 Target (95% clean)',
         'threshold': 95.0,
         'year': 2045},
    ]

    # Load scenarios for these thresholds
    all_scenarios = load_scenarios(iso=iso)

    for tc in test_cases:
        print(f"\n  --- {tc['label']} ---")

        # Find Medium scenario at this threshold (or nearest)
        threshold = tc['threshold']
        matching = [s for s in all_scenarios
                    if abs(s['threshold'] - threshold) < 3.0
                    and 'M' in s.get('scenario', 'M')]

        if not matching:
            # Use closest threshold
            closest = min(all_scenarios, key=lambda s: abs(s['threshold'] - threshold))
            matching = [closest]
            print(f"    (Using nearest threshold: {closest['threshold']}%)")

        # Pick the Medium-sensitivity scenario
        med_scenarios = [s for s in matching if s.get('scenario', '').startswith('MMMM_M')]
        if not med_scenarios:
            med_scenarios = matching[:1]

        sc = med_scenarios[0]
        resource_mix = sc['resource_mix']
        print(f"    Mix: CF={resource_mix['clean_firm']}% Sol={resource_mix['solar']}% "
              f"Wind={resource_mix['wind']}% CCS={resource_mix['ccs_ccgt']}% "
              f"Hydro={resource_mix['hydro']}%")
        print(f"    Procurement: {sc.get('procurement_pct', 100)}%, "
              f"Batt4: {sc.get('battery_dispatch_pct', 0)}%, "
              f"LDES: {sc.get('ldes_dispatch_pct', 0)}%")

        # Dispatch
        dispatch = reconstruct_hourly_dispatch(
            demand_norm, supply_profiles, resource_mix,
            sc.get('procurement_pct', 100),
            sc.get('battery_dispatch_pct', 0),
            sc.get('battery8_dispatch_pct', 0),
            sc.get('ldes_dispatch_pct', 0))

        # Merit-order stack (RA + GAF aware)
        proc_pct = sc.get('procurement_pct', 100)
        batt4_pct = sc.get('battery_dispatch_pct', 0)
        batt8_pct = sc.get('battery8_dispatch_pct', 0)
        ldes_pct = sc.get('ldes_dispatch_pct', 0)
        stack, fossil_mw = build_merit_order_stack(
            iso, sc['threshold'], fuel_level,
            resource_mix=resource_mix, procurement_pct=proc_pct,
            battery_pct=batt4_pct, battery8_pct=batt8_pct, ldes_pct=ldes_pct)
        print(f"    Fossil stack ({fossil_mw:,.0f} MW):")
        for unit_type, cap, mc in stack:
            print(f"      {unit_type:>12}: {cap:>8,.0f} MW @ ${mc:.2f}/MWh")

        # Price model
        price_model = get_price_model(iso, fuel_level)

        # LMP computation
        hourly_lmp, hourly_mu = compute_hourly_lmp_vectorized(
            dispatch, demand_mw_profile, stack, price_model, iso)

        # Stats
        stats = compute_lmp_stats(hourly_lmp, hourly_mu, demand_mw_profile, dispatch)

        print(f"\n    LMP Results:")
        print(f"      Avg LMP:          ${stats['avg_lmp']:.2f}/MWh")
        print(f"      Peak avg:         ${stats['peak_avg_lmp']:.2f}/MWh")
        print(f"      Off-peak avg:     ${stats['offpeak_avg_lmp']:.2f}/MWh")
        print(f"      P10/P50/P90:      ${stats['lmp_p10']:.2f} / ${stats['lmp_p50']:.2f} / ${stats['lmp_p90']:.2f}")
        print(f"      Volatility:       ${stats['price_volatility']:.2f}")
        print(f"      Zero-price hours: {stats['zero_price_hours']}")
        print(f"      Negative hours:   {stats['negative_price_hours']}")
        print(f"      Scarcity hours:   {stats['scarcity_hours']}")
        print(f"      Fossil revenue:   ${stats['fossil_revenue_mwh']:.2f}/MWh")
        print(f"      Net peak price:   ${stats['net_peak_price']:.2f}/MWh")

        # Retirement info
        _, retirement = compute_fossil_retirement(iso, sc['threshold'], emission_rates, fossil_mix)
        gas_only = retirement.get('forced_gas_only', False)
        print(f"      Gas-only fleet:   {'Yes' if gas_only else 'No'}")

    print(f"\n{'='*70}")
    print(f"  TEST COMPLETE")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='Compute synthetic hourly LMP')
    parser.add_argument('--iso', type=str, default=None,
                        help='ISO to process (default: all)')
    parser.add_argument('--fuel-level', type=str, default=None,
                        help='Fuel sensitivity level: Low/Medium/High (default: all)')
    parser.add_argument('--test', action='store_true',
                        help='Run test cases only (PJM: 2025/50%%/95%%)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Single threshold to process')
    args = parser.parse_args()

    if args.test:
        run_test_cases(args.iso or 'PJM')
        return

    print("=" * 70)
    print("  LMP PRICE CALCULATION MODULE")
    print("=" * 70)

    isos_to_run = [args.iso] if args.iso else ISOS
    fuel_levels = [args.fuel_level] if args.fuel_level else ['Low', 'Medium', 'High']

    demand_data, gen_profiles, emission_rates, fossil_mix = load_common_data()

    total_start = time.time()
    all_results = []

    for iso in isos_to_run:
        print(f"\n  Processing {iso}...")
        iso_start = time.time()

        # Load dispatch cache for this ISO
        dispatch_cache = load_dispatch_cache(iso)
        print(f"    Dispatch cache loaded: {len(dispatch_cache)} entries")

        iso_results = []
        iso_archetypes = {}

        for fuel_level in fuel_levels:
            print(f"    Fuel level: {fuel_level}")

            scenarios = load_scenarios(iso=iso, threshold=args.threshold)
            if not scenarios:
                print(f"    No scenarios found for {iso}")
                continue

            # Filter to Medium scenario key only (for now — full sweep later)
            med_key_prefix = 'MMMM_M' if fuel_level == 'Medium' else None
            if med_key_prefix:
                filtered = [s for s in scenarios if s.get('scenario', '').startswith(med_key_prefix)]
                if filtered:
                    scenarios = filtered

            results, archetypes = run_lmp_for_iso(
                iso, scenarios, demand_data, gen_profiles,
                fuel_level=fuel_level, dispatch_cache=dispatch_cache)

            iso_results.extend(results)
            iso_archetypes.update(archetypes)

        # Save dispatch cache (with new entries appended)
        save_dispatch_cache(iso, dispatch_cache)
        print(f"    Dispatch cache saved: {len(dispatch_cache)} entries")

        # Save ISO results
        save_iso_results(iso, iso_results, iso_archetypes)
        all_results.extend(iso_results)

        elapsed = time.time() - iso_start
        print(f"    {iso} complete: {len(iso_results)} scenarios, "
              f"{len(iso_archetypes)} archetypes — {elapsed:.0f}s")

    # Save cross-ISO summary
    if all_results:
        summary = {
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'isos': list(set(r['iso'] for r in all_results)),
            'fuel_levels': fuel_levels,
            'total_scenarios': len(all_results),
            'by_iso': {},
        }
        for iso in isos_to_run:
            iso_r = [r for r in all_results if r['iso'] == iso]
            if iso_r:
                summary['by_iso'][iso] = {
                    'n_scenarios': len(iso_r),
                    'thresholds': sorted(set(r['threshold'] for r in iso_r)),
                    'avg_lmp_by_threshold': {
                        str(t): round(np.mean([r['avg_lmp'] for r in iso_r
                                               if r['threshold'] == t]), 2)
                        for t in sorted(set(r['threshold'] for r in iso_r))
                    },
                }

        summary_path = os.path.join(LMP_DIR, 'lmp_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Summary: {summary_path}")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  LMP COMPUTATION COMPLETE — {total_elapsed:.0f}s")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
