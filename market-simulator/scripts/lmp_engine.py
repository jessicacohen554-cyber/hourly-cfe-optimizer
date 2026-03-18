#!/usr/bin/env python3
"""
LMP Engine — Synthetic Hourly LMP from Merit-Order Fossil Dispatch
==================================================================
Utility module imported by step4_1a_fossil_dispatch.py, step6_1_smartargets.py,
and calibrate_lmp_model.py. Provides:
  - build_merit_order_stack() — fossil plant merit-order by marginal cost
  - compute_hourly_lmp_vectorized() — 8,760-hour synthetic LMP
  - PriceModel / get_price_model() — fuel-price-dependent cost models
  - compute_lmp_stats() — summary statistics
  - compute_marginal_costs() — per-unit marginal cost calculation
  - load_scenarios() — scenario configuration loading

Originally step5b_compute_lmp_prices.py, now a shared utility.
"""

import json
import os
import sys
import time
import argparse
import hashlib
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from pipeline_config import MUST_RUN_PCT, ORDC_PARAMS, SCARCITY_MODE

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

LMP_DIR = os.path.join(SCRIPT_DIR, 'data', 'step4-analysis', 'lmp')
STEP3_PARQUET_DIR = os.path.join(SCRIPT_DIR, 'data', 'step2.2-cost')

# ══════════════════════════════════════════════════════════════════════════════
# FOSSIL MERIT-ORDER STACK — heat rates, VOM, marginal cost
# ══════════════════════════════════════════════════════════════════════════════

# Heat rates (MMBtu/MWh) — PJM IMM SOM 2024 benchmarks + EIA Table 8.1
# PJM SOM uses 7,000 Btu/kWh for CCGT spark spreads; fleet-weighted avg ~7.0
# EIA: modern H/J-class 6.7, F-class 7.0-7.5, B/D/E-class 8.0+
# Fleet-weighted PJM CCGT: ~7.0 (mix of vintages)
HEAT_RATES = {
    'coal_steam': 10.0,   # EIA Table 8.1 avg; PJM SOM dark spread benchmark
    'gas_ccgt': 7.0,      # PJM SOM spark spread benchmark (fleet-weighted)
    'gas_ct': 10.5,       # EIA simple-cycle avg 10,000-11,000 Btu/kWh
    'oil_ct': 10.5,       # Similar to gas CT
}

# Variable O&M ($/MWh) — PJM SOM 2024 decomposition
# SOM 2024 breakdown: Variable Maintenance $3.18 + Variable Operations $1.43 = $4.61 fleet avg
# Split by technology using NREL ATB relativities
VOM = {
    'coal_steam': 5.50,   # Higher maintenance (sorbent, ash, tube repair)
    'gas_ccgt': 3.50,     # SOM fleet avg ~$4.61; CCGT lower than fleet
    'gas_ct': 5.00,       # CT: more starts, higher per-MWh maintenance
    'oil_ct': 6.00,       # Oil: rare dispatch, high per-start costs
}

# CO2 emission rates (tons CO2 / MWh) — EPA eGRID 2022 + EIA
CO2_RATES = {
    'coal_steam': 0.95,   # ~2,100 lb/MWh ≈ 0.95 t/MWh
    'gas_ccgt': 0.37,     # ~820 lb/MWh ≈ 0.37 t/MWh
    'gas_ct': 0.55,       # ~1,210 lb/MWh ≈ 0.55 t/MWh (lower efficiency)
    'oil_ct': 0.65,       # ~1,430 lb/MWh ≈ 0.65 t/MWh
}

# NOx emission rates (lb/MWh) — EPA CAMPD 2023, Cross-State Air Pollution Rule
# Coal: 1.5-3.0 lb/MWh uncontrolled, 0.5-1.0 with SCR; using fleet avg w/ SCR
# Gas CCGT: 0.06-0.15 lb/MWh (low-NOx DLN burners); Gas CT: 0.15-0.4 lb/MWh
# Oil: 0.8-2.0 lb/MWh; using fleet avg with typical controls
NOX_RATES = {
    'coal_steam': 0.80,   # lb/MWh — fleet avg with SCR/SNCR (EPA CAMPD 2023)
    'gas_ccgt':   0.10,   # lb/MWh — DLN burners (EPA CAMPD 2023)
    'gas_ct':     0.25,   # lb/MWh — simple cycle, higher per-MWh (EPA CAMPD 2023)
    'oil_ct':     1.20,   # lb/MWh — distillate oil CT (EPA CAMPD 2023)
}

# SOx emission rates (lb/MWh) — EPA CAMPD 2023, Acid Rain Program
# Coal: highly variable, 2-12 lb/MWh uncontrolled; 0.5-3.0 with FGD
# Gas: essentially zero (no sulfur in pipeline gas)
# Oil: 0.5-2.0 lb/MWh depending on sulfur content
SOX_RATES = {
    'coal_steam': 1.80,   # lb/MWh — fleet avg with FGD scrubbers (EPA CAMPD 2023)
    'gas_ccgt':   0.01,   # lb/MWh — trace (mercaptan odorant)
    'gas_ct':     0.01,   # lb/MWh — trace
    'oil_ct':     0.80,   # lb/MWh — low-sulfur distillate oil (EPA CAMPD 2023)
}

# NOx allowance prices ($/ton) — CSAPR, state trading programs
NOX_PRICES = {
    'Low': 500.0,        # $/ton — low market, surplus allowances
    'Medium': 2500.0,    # $/ton — 2024 CSAPR Group 3 avg
    'High': 5000.0,      # $/ton — scarcity pricing / tighter caps
}

# SOx allowance prices ($/ton) — Acid Rain Program
SOX_PRICES = {
    'Low': 25.0,         # $/ton — 2024 ARP (surplus era)
    'Medium': 100.0,     # $/ton — moderate enforcement
    'High': 500.0,       # $/ton — tight cap scenario
}

# CO2 allowance prices ($/ton) — RGGI, state programs
# PJM SOM 2024: CO2 cost component = $1.94/MWh (5.8% of LMP)
# RGGI 2024 avg clearing price ~$14/ton; not all PJM states in RGGI
# Effective fleet-weighted CO2 cost: ~$5.50/ton × fleet-avg emission rate ≈ $1.94/MWh
CO2_PRICES = {
    'Low': 3.00,      # $/ton — low RGGI / no state program
    'Medium': 5.50,   # $/ton — 2024 effective (RGGI weighted by PJM participation)
    'High': 14.00,    # $/ton — full RGGI clearing price
}

# Cost-based offer adder — ISO-specific markup above marginal cost
# PJM Manual 15 allows 10% markup for cost-based offers (SOM 2024: $2.00/MWh, 5.9% of RT LMP)
# Energy-only markets (ERCOT, SPP) use competitive offers — no regulatory markup
# MISO has cost-based offers but lower effective markup due to market structure
TEN_PERCENT_ADDER = 0.10  # Legacy default (backward compat)

COST_BASED_ADDERS = {
    'CAISO': 0.10,   # RA market — cost-based offer rules similar to PJM
    'ERCOT': 0.00,   # Energy-only — competitive offers, no regulatory markup
    'PJM':   0.10,   # RPM capacity market — PJM Manual 15 cost-based offer rule
    'NYISO': 0.10,   # ICAP capacity market — NYISO OATT cost-based rules
    'NEISO': 0.10,   # FCM capacity market — ISO-NE Manual for Market Operations
    'MISO':  0.07,   # PRA capacity market — lower effective markup (Module C energy offer rules)
    'SPP':   0.00,   # Energy-only — competitive offers, no regulatory markup
}

# Fuel prices ($/MMBtu) by sensitivity level
FUEL_PRICES = {
    'Low':    {'coal': 2.00, 'gas': 2.00, 'oil': 8.00},
    'Medium': {'coal': 2.25, 'gas': 3.50, 'oil': 10.50},
    'High':   {'coal': 2.50, 'gas': 6.00, 'oil': 13.00},
}

# Capacity shares within fossil fleet (fraction of total fossil capacity)
# PJM: Monitoring Analytics 2024 SOM — coal 37.8/130.6=0.29, gas 88.8/130.6=0.68, oil 4.0/130.6=0.03
# PJM gas split: ~55% CCGT (~48.8 GW), ~45% CT (~40.0 GW) per SOM CC vs peaker breakdown
# Others: EIA 860 cross-referenced with ISO-specific capacity reports
FOSSIL_CAPACITY_SHARES = {
    'CAISO': {'coal_steam': 0.00, 'gas_ccgt': 0.55, 'gas_ct': 0.40, 'oil_ct': 0.05},
    'ERCOT': {'coal_steam': 0.22, 'gas_ccgt': 0.50, 'gas_ct': 0.28, 'oil_ct': 0.00},
    'PJM':   {'coal_steam': 0.29, 'gas_ccgt': 0.37, 'gas_ct': 0.31, 'oil_ct': 0.03},
    'NYISO': {'coal_steam': 0.00, 'gas_ccgt': 0.45, 'gas_ct': 0.50, 'oil_ct': 0.05},
    'NEISO': {'coal_steam': 0.00, 'gas_ccgt': 0.52, 'gas_ct': 0.42, 'oil_ct': 0.06},
    'MISO':  {'coal_steam': 0.35, 'gas_ccgt': 0.40, 'gas_ct': 0.24, 'oil_ct': 0.01},
    'SPP':   {'coal_steam': 0.30, 'gas_ccgt': 0.42, 'gas_ct': 0.27, 'oil_ct': 0.01},
}

# Actual installed fossil capacity (MW) — 2025 estimates
# PJM: Monitoring Analytics 2024 SOM baseline (gas 88.8 + coal 37.8 + oil 4.0 = 130.6 GW)
#   2025 coal retirements: Brandon Shores 1.3 GW, Wagner 0.8 GW, Indian River 0.4 GW, other ~0.5 GW = ~3.0 GW
#   Sources: PJM Gen Deactivations list, Utility Dive, IMM SOM Sec 12
#   Net 2025: coal ~34.8 GW, gas ~89.5 GW (minor additions), oil ~3.5 GW = ~127.8 GW
# Others: EIA 860M (2024) cross-referenced with ISO capacity reports
INSTALLED_FOSSIL_MW = {
    'CAISO': 47_000,   # ~47 GW gas fleet (no coal)
    'ERCOT': 80_000,   # ~80 GW total fossil (gas ~55, coal ~18, oil ~7)
    'PJM':   127_800,  # 127.8 GW fossil — 2025 est. after ~3 GW coal retirements
    'NYISO': 28_000,   # ~28 GW fossil (mostly gas)
    'NEISO': 16_000,   # ~16 GW fossil (mostly gas)
    'MISO':  105_000,  # ~105 GW fossil (gas ~55, coal ~45, oil ~5) — EIA 860M 2024
    'SPP':   58_000,   # ~58 GW fossil (gas ~35, coal ~20, oil ~3) — EIA 860M 2024
}

# Peak demand (MW) — matches step3a_cost_optimization.py
PEAK_DEMAND_MW = {
    'CAISO': 43_860, 'ERCOT': 83_597, 'PJM': 160_560, 'NYISO': 31_857, 'NEISO': 25_898,
    'MISO': 118_661, 'SPP': 54_745,
}

# Resource adequacy reserve margin — 15%, consistent with step3/step4
RESOURCE_ADEQUACY_MARGIN = 0.15

# Peak capacity credits — exact copy from step3a_cost_optimization.py
PEAK_CAPACITY_CREDITS = {
    'clean_firm': 1.0,   # nuclear — fully accredited
    'solar': 0.30,       # ELCC — only afternoon contribution
    'wind': 0.10,        # ELCC — low correlation with peak
    'ccs_ccgt': 0.90,    # dispatchable
    'hydro': 0.50,       # seasonal/capacity-limited
    'battery': 0.95,     # 4hr Li-ion
    'battery8': 0.95,    # 8hr Li-ion
    'ldes': 0.90,        # 100hr iron-air
    'h2': 0.95,          # H2 storage — dispatchable
}

# Gas Availability Factor (GAF) — forced outages + correlated weather risk
# From step3: gas_needed_mw / GAF = nameplate needed to deliver firm capacity
GAS_AVAILABILITY_FACTOR = {
    'CAISO': 0.88,  # 12% deration — summer ambient + mechanical
    'ERCOT': 0.83,  # 17% deration — extreme weather both seasons
    'PJM':   0.82,  # 18% deration — PJM ELCC data, Winter Storm Elliott
    'NYISO': 0.82,  # 18% deration — pipeline constraints, winter gas
    'NEISO': 0.85,  # 15% deration — mechanical + weather (pipeline separate)
    'MISO':  0.83,  # 17% deration — extreme weather (Winter Storm Heather 2024), large footprint
    'SPP':   0.84,  # 16% deration — weather exposure similar to ERCOT/MISO, less pipeline-constrained
}


def compute_marginal_costs(fuel_level='Medium', co2_level='Medium',
                           nox_price=0.0, sox_price=0.0,
                           custom_fuel_prices=None, custom_co2_price=None,
                           custom_heat_rates=None, custom_vom=None,
                           iso=None):
    """Compute marginal cost ($/MWh) for each fossil unit type.

    Cost-based offer formula:
      MC = (Incremental Heat Rate × Fuel Price + VOM + CO2 Rate × CO2 Price
            + NOx Rate × NOx Price + SOx Rate × SOx Price) × (1 + Adder)

    The adder is ISO-specific:
      - PJM/NYISO/NEISO/CAISO: 10% (PJM Manual 15 cost-based offer rule)
      - MISO: 7% (Module C energy offer rules, lower effective markup)
      - ERCOT/SPP: 0% (energy-only markets, competitive offers)

    NOx/SOx rates are in lb/MWh; prices are $/ton. Conversion: rate_lb * price_$/ton / 2000.
    CO2 costs reflect RGGI and state compliance programs (SOM 2024: $1.94/MWh).
    """
    fp = custom_fuel_prices if custom_fuel_prices else FUEL_PRICES[fuel_level]
    co2_price = custom_co2_price if custom_co2_price is not None else CO2_PRICES.get(co2_level, CO2_PRICES['Medium'])
    hr = custom_heat_rates if custom_heat_rates else HEAT_RATES
    vm = custom_vom if custom_vom else VOM
    adder_rate = COST_BASED_ADDERS.get(iso, TEN_PERCENT_ADDER) if iso else TEN_PERCENT_ADDER
    adder = 1.0 + adder_rate

    FUEL_KEY_MAP = {'coal_steam': 'coal', 'gas_ccgt': 'gas', 'gas_ct': 'gas', 'oil_ct': 'oil'}
    costs = {}
    for unit_type in hr:
        if unit_type not in FUEL_KEY_MAP:
            continue  # Skip non-fleet keys (e.g. new_gas_ccgt, new_gas_ct, new_coal)
        fuel_key = FUEL_KEY_MAP[unit_type]
        # Base: fuel + VOM + CO2
        base_cost = (hr[unit_type] * fp[fuel_key] + vm[unit_type]
                     + CO2_RATES[unit_type] * co2_price)
        # NOx: rate (lb/MWh) × price ($/ton) / 2000 (lb/ton)
        if nox_price > 0:
            base_cost += NOX_RATES.get(unit_type, 0) * nox_price / 2000.0
        # SOx: same conversion
        if sox_price > 0:
            base_cost += SOX_RATES.get(unit_type, 0) * sox_price / 2000.0
        costs[unit_type] = base_cost * adder

    return costs


def _compute_clean_peak_mw(iso, resource_mix, battery_pct=0,
                           battery8_pct=0, ldes_pct=0, h2_pct=0):
    """Compute clean peak capacity contribution (MW) from resource mix.

    Mirrors step3a_cost_optimization.py clean_peak_mw calculation exactly.
    Uses per-resource ELCC capacity credits at system peak.
    """
    peak_mw = PEAK_DEMAND_MW.get(iso, 80_000)
    demand_twh = BASE_DEMAND_TWH.get(iso, 0)
    avg_demand_mw = (demand_twh * 1e6) / H  # TWh → MWh / 8760 → avg MW

    clean_peak = (
        resource_mix.get('clean_firm', 0) / 100.0 * avg_demand_mw * PEAK_CAPACITY_CREDITS['clean_firm'] +
        resource_mix.get('solar', 0) / 100.0 * avg_demand_mw * PEAK_CAPACITY_CREDITS['solar'] +
        resource_mix.get('wind', 0) / 100.0 * avg_demand_mw * PEAK_CAPACITY_CREDITS['wind'] +
        resource_mix.get('offshore_wind', 0) / 100.0 * avg_demand_mw * PEAK_CAPACITY_CREDITS.get('offshore_wind', 0.25) +
        resource_mix.get('ccs_ccgt', 0) / 100.0 * avg_demand_mw * PEAK_CAPACITY_CREDITS['ccs_ccgt'] +
        resource_mix.get('hydro', 0) / 100.0 * avg_demand_mw * PEAK_CAPACITY_CREDITS['hydro'] +
        battery_pct / 100.0 * avg_demand_mw * PEAK_CAPACITY_CREDITS['battery'] +
        battery8_pct / 100.0 * avg_demand_mw * PEAK_CAPACITY_CREDITS['battery8'] +
        ldes_pct / 100.0 * avg_demand_mw * PEAK_CAPACITY_CREDITS['ldes'] +
        h2_pct / 100.0 * avg_demand_mw * PEAK_CAPACITY_CREDITS['h2']
    )
    return clean_peak


def build_merit_order_stack(iso, clean_pct, fuel_level='Medium', total_fossil_mw=None,
                             resource_mix=None,
                             battery_pct=0, battery8_pct=0, ldes_pct=0,
                             h2_pct=0, co2_level='Medium',
                             nox_price=0.0, sox_price=0.0,
                             nox_limit=None, sox_limit=None,
                             custom_fuel_prices=None, custom_co2_price=None,
                             custom_heat_rates=None, custom_vom=None,
                             firm_import_mw=0):
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
        battery_pct: battery dispatch percentage
        battery8_pct: battery8 dispatch percentage
        ldes_pct: LDES dispatch percentage
        h2_pct: H2 storage dispatch percentage
        co2_level: 'Low', 'Medium', 'High' — CO2 allowance pricing
        nox_price: $/ton NOx allowance price (added to marginal cost)
        sox_price: $/ton SOx allowance price (added to marginal cost)
        nox_limit: lb/MWh NOx emission cap — generators above this are retired
        sox_limit: lb/MWh SOx emission cap — generators above this are retired
        custom_fuel_prices: dict with 'coal', 'gas', 'oil' prices (overrides fuel_level)
        custom_co2_price: float CO2 price (overrides co2_level)

    Returns:
        stack: list of (unit_type, capacity_mw, marginal_cost_per_mwh)
        total_capacity_mw: total fossil MW
    """
    mc = compute_marginal_costs(fuel_level, co2_level,
                                nox_price=nox_price, sox_price=sox_price,
                                custom_fuel_prices=custom_fuel_prices,
                                custom_co2_price=custom_co2_price,
                                custom_heat_rates=custom_heat_rates,
                                custom_vom=custom_vom,
                                iso=iso)

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
                    iso, resource_mix,
                    battery_pct, battery8_pct, ldes_pct, h2_pct)
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
            # Firm imports reduce the peak that local fossil must cover
            residual_peak_mw = max(0, ra_peak_mw - clean_peak_mw - firm_import_mw)

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

    # Enforce NOx/SOx emission limits — retire generators exceeding caps
    if nox_limit is not None:
        retired_nox = [ut for ut in active_shares if NOX_RATES.get(ut, 0) > nox_limit]
        for ut in retired_nox:
            del active_shares[ut]
        # Renormalize remaining shares
        total_sh = sum(active_shares.values())
        if total_sh > 0:
            active_shares = {k: v / total_sh for k, v in active_shares.items()}

    if sox_limit is not None:
        retired_sox = [ut for ut in active_shares if SOX_RATES.get(ut, 0) > sox_limit]
        for ut in retired_sox:
            del active_shares[ut]
        total_sh = sum(active_shares.values())
        if total_sh > 0:
            active_shares = {k: v / total_sh for k, v in active_shares.items()}

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


def build_plant_level_merit_order(iso, clean_pct, fuel_level='Medium',
                                   carbon_price=0, nox_price=0.0, sox_price=0.0,
                                   custom_fuel_prices=None, custom_heat_rates=None,
                                   custom_vom=None, fleet_df=None):
    """Build plant-level merit-order stack using real EIA 860 generator data.

    Unlike build_merit_order_stack() which uses 4 aggregated unit types with average
    heat rates, this function creates a stack entry for each individual generator
    using its actual reported heat rate from EIA Form 860.

    Args:
        iso: ISO region
        clean_pct: clean energy threshold
        fuel_level: 'Low', 'Medium', 'High'
        carbon_price: $/ton CO2
        nox_price: $/ton NOx allowance
        sox_price: $/ton SOx allowance
        custom_fuel_prices: dict override
        custom_heat_rates: not used (we use actual per-plant rates)
        custom_vom: dict override for VOM by unit type
        fleet_df: Pre-loaded DataFrame from load_iso_fleet(). If None, loads it.

    Returns:
        plant_stack: list of dicts with keys:
            plant_id, gen_id, entity_name, plant_name, unit_type, capacity_mw,
            heat_rate, marginal_cost, latitude, longitude, county, state,
            fuel_type, prime_mover, online_year, co2_rate, nox_rate, sox_rate
        total_capacity_mw: total fossil MW
    """
    # Import here to avoid circular imports
    from fleet_model import load_iso_fleet, _classify_unit

    if fleet_df is None:
        fleet_df = load_iso_fleet(iso)

    if fleet_df is None or len(fleet_df) == 0:
        # Fallback to aggregated stack
        return None, 0

    fp = custom_fuel_prices if custom_fuel_prices else FUEL_PRICES[fuel_level]
    vm = custom_vom if custom_vom else VOM
    adder_rate = COST_BASED_ADDERS.get(iso, TEN_PERCENT_ADDER)
    adder = 1.0 + adder_rate

    # Classify each generator and compute marginal cost
    plant_stack = []

    for _, row in fleet_df.iterrows():
        # Classify unit type using the module-level function
        unit_type = _classify_unit(
            str(row.get('prime_mover', '')), str(row.get('fuel_type', '')))
        if unit_type is None:
            continue  # Skip non-fossil generators

        cap_mw = float(row.get('capacity_mw', 0))
        if cap_mw <= 0:
            continue

        # Use actual heat rate if available, else type default
        hr = row.get('heat_rate')
        if pd.isna(hr) or hr is None or hr <= 0:
            hr = HEAT_RATES.get(unit_type, 10.0)
        else:
            hr = float(hr)

        fuel_key = {'coal_steam': 'coal', 'gas_ccgt': 'gas', 'gas_ct': 'gas', 'oil_ct': 'oil'}[unit_type]

        co2_rate = CO2_RATES.get(unit_type, 0.5)
        nox_rate = NOX_RATES.get(unit_type, 0.5)
        sox_rate = SOX_RATES.get(unit_type, 0.5)

        # Marginal cost
        mc = (hr * fp[fuel_key] + vm.get(unit_type, 4.0)
              + co2_rate * carbon_price)
        if nox_price > 0:
            mc += nox_rate * nox_price / 2000.0
        if sox_price > 0:
            mc += sox_rate * sox_price / 2000.0
        mc *= adder

        online_year = row.get('online_year')
        age = 2025 - int(online_year) if pd.notna(online_year) and online_year else None

        plant_stack.append({
            'plant_id': row.get('plant_id'),
            'gen_id': row.get('gen_id'),
            'entity_name': row.get('entity_name', ''),
            'plant_name': row.get('plant_name', ''),
            'unit_type': unit_type,
            'capacity_mw': cap_mw,
            'heat_rate': round(hr, 2),
            'heat_rate_source': row.get('heat_rate_source', 'default'),
            'marginal_cost': round(mc, 2),
            'latitude': row.get('latitude'),
            'longitude': row.get('longitude'),
            'county': row.get('county', ''),
            'state': row.get('state', ''),
            'fuel_type': row.get('fuel_type', ''),
            'prime_mover': row.get('prime_mover', ''),
            'online_year': online_year,
            'age_years': age,
            'co2_rate': co2_rate,
            'nox_rate': nox_rate,
            'sox_rate': sox_rate,
        })

    # Sort by marginal cost (cheapest first)
    plant_stack.sort(key=lambda x: x['marginal_cost'])

    total_mw = sum(p['capacity_mw'] for p in plant_stack)

    return plant_stack, total_mw


# ══════════════════════════════════════════════════════════════════════════════
# ISO-SPECIFIC PRICE FORMATION MODELS
# ══════════════════════════════════════════════════════════════════════════════

class PriceModel:
    """Base price formation model. ISO-specific subclasses override parameters.

    Three pricing layers:
      1. Merit-order dispatch — marginal cost from heat rate × fuel + VOM
      2. Demand-quantile pricing — congestion/tightness adder on high-demand hours,
         negative pricing on low-demand hours with must-run surplus
      3. Scarcity pricing — exponential adder when reserves drop below threshold

    The demand-quantile layer captures real-world price formation that a single-bus
    merit-order model misses: transmission congestion, gas-supply tightness during
    winter peaks, bid markup above marginal cost, and must-run nuclear/wind curtailment
    economics. Parameters calibrated against PJM SOM 2024 price distribution.
    """

    def __init__(self, iso, fuel_level='Medium'):
        self.iso = iso
        self.fuel_level = fuel_level
        self.scarcity_cap = 2000.0     # $/MWh cap during scarcity
        self.floor_price = -30.0       # $/MWh floor during surplus
        self.surplus_slope = 0.5       # steepness of negative price curve
        self.surplus_decay = 0.02      # decay rate for surplus pricing
        self.rt_sensitivity_factor = 1.0  # scale for RT volatility
        self.scarcity_threshold = 0.05    # reserves/demand ratio triggering scarcity

        # Unit commitment: must-run depression strength
        # When residual demand < total must-run MW, prices are depressed
        # because must-run units (coal min-gen) bid below cost to avoid cycling
        self.must_run_depression = 0.30  # default; ISO subclasses override

        # Demand-quantile pricing parameters
        # High-demand adder: congestion + gas tightness on highest-demand hours
        self.dq_high_percentile = 80    # demand percentile above which adder applies
        self.dq_high_max_adder = 60.0   # $/MWh max adder at peak demand hour
        self.dq_high_exponent = 2.0     # curvature — higher = sharper peak premium
        # Scarcity tail: extreme high-demand hours get exponential scarcity adder
        self.dq_scarcity_percentile = 97  # demand percentile for scarcity tail
        self.dq_scarcity_max = 500.0     # $/MWh max scarcity-like adder
        # Low-demand negative pricing: must-run surplus drives prices negative
        self.dq_low_percentile = 15     # demand percentile below which prices depress
        self.dq_low_floor = -25.0       # $/MWh floor for low-demand hours
        self.dq_low_exponent = 1.5      # curvature for negative pricing

        # Demand elasticity for extreme prices (Rec #8)
        # At prices >$200/MWh, price-responsive industrial loads curtail (5-15%).
        # Interruptible contracts, demand response programs, and voluntary
        # curtailment moderate scarcity pricing. Elasticity applied as
        # post-pricing dampening: reduces extreme prices toward threshold.
        # Sources: FERC DR assessment (2024), PJM IMM DR participation rates,
        # ERCOT 4CP/load shed programs, CAISO RDRR/PDR programs.
        self.demand_elasticity_threshold = 200.0  # $/MWh — onset of curtailment
        self.demand_elasticity_max_curtailment = 0.12  # 12% max demand reduction
        self.demand_elasticity_damping = 0.5  # price damping factor per unit curtailment

        # ORDC parameters — loaded from pipeline_config.ORDC_PARAMS
        ordc = ORDC_PARAMS.get(iso, {'voll': 2000, 'reserve_target_mw': 2500, 'lolp_k': 0.003})
        self.ordc_voll = ordc['voll']
        self.ordc_reserve_target_mw = ordc['reserve_target_mw']
        self.ordc_lolp_k = ordc['lolp_k']

    def compute_ordc_adder(self, reserves_mw):
        """ORDC price adder: VOLL × LOLP(reserves). Fully vectorized.

        LOLP modeled as logistic sigmoid: LOLP = 1 / (1 + exp(k * (R - R_target))).
        When reserves >> target: LOLP ≈ 0, adder ≈ $0.
        When reserves << target: LOLP ≈ 1, adder ≈ VOLL.
        Sigmoid transitions smoothly around the reserve target.

        Args:
            reserves_mw: numpy array (8760,) of hourly operating reserves in MW
        Returns:
            numpy array (8760,) of ORDC adders in $/MWh
        """
        lolp = 1.0 / (1.0 + np.exp(self.ordc_lolp_k * (reserves_mw - self.ordc_reserve_target_mw)))
        return self.ordc_voll * lolp

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

        # Check reserve margin — use TOTAL stack remaining (not just within-band)
        total_stack_cap = sum(cap for _, cap, _ in stack)
        remaining_capacity = total_stack_cap - residual_demand_mw
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
    """PJM: RPM capacity market, penalty factor scarcity, moderate negative prices.

    Calibrated against PJM SOM 2024: avg $34.70, peak $42, offpeak $28,
    P10 $18, P90 $55, volatility $25, ~200 negative hours, ~100 scarcity hours.
    """

    def __init__(self, fuel_level='Medium'):
        super().__init__('PJM', fuel_level)
        self.scarcity_cap = 2000.0
        self.floor_price = -30.0
        self.surplus_slope = 0.4
        self.surplus_decay = 0.015
        self.scarcity_threshold = 0.03  # PJM has large reserves; scarcity is rare
        self.coal_min_gen_fraction = 0.4
        self.must_run_depression = 0.35  # PJM: 29% coal + large nuclear → strong must-run effect

        # PJM demand elasticity: RPM DR programs (Economic, Emergency, Pre-Emergency)
        # PJM had ~9.5 GW DR resources in 2024 (IMM SOM) — largest pool of any ISO
        self.demand_elasticity_threshold = 200.0
        self.demand_elasticity_max_curtailment = 0.15  # 15% — most DR of any ISO
        self.demand_elasticity_damping = 0.50

        # PJM demand-quantile calibration (v11 — iterative SOM calibration)
        # Base merit-order includes 10% adder + CO2 + realistic VOM/heat rates.
        # Demand-quantile layer adds: congestion ($3.01), markup beyond 10% (~$1.56),
        # ancillary redispatch ($1.33), and compresses off-peak to baseload pricing.
        #
        # v11: Minor volatility reduction (scarcity_max 165→140) to close $30→$25 gap
        #
        # High-demand: congestion + gas tightness on top 25% of hours
        self.dq_high_percentile = 75
        self.dq_high_max_adder = 23.0
        self.dq_high_exponent = 2.0
        # Scarcity tail: PJM penalty factor regime
        self.dq_scarcity_percentile = 95.5
        self.dq_scarcity_max = 155.0       # v11: 165→155, balance vol + scarcity hrs
        # Low-demand: PJM ~200 negative price hours
        self.dq_low_percentile = 9
        self.dq_low_floor = -25.0           # shallower than -30 for less vol
        self.dq_low_exponent = 1.8
        # Mid-low: P9-P70 — compress off-peak toward baseload pricing
        self.dq_midlow_percentile = 70
        self.dq_midlow_discount = 0.55


class ERCOTPriceModel(PriceModel):
    """ERCOT: Energy-only market with ORDC (Operating Reserve Demand Curve).

    ORDC: adder = VOLL × LOLP(reserves). LOLP increases exponentially as
    reserves drop below ~3,000 MW. VOLL = $5,000/MWh (post-2023 reform).
    """

    def __init__(self, fuel_level='Medium'):
        super().__init__('ERCOT', fuel_level)
        self.scarcity_cap = 5000.0
        self.floor_price = -50.0
        self.surplus_decay = 0.018        # v11.1: 0.025→0.018, reduce neg hrs 580→~400
        self.voll = 5000.0
        self.ordc_knee_mw = 3000.0  # reserves below this trigger exponential ORDC
        self.scarcity_threshold = 0.02   # v11: 0.05→0.02, ERCOT 2024 solar+storage reduced scarcity
        self.must_run_depression = 0.25  # ERCOT: 22% coal, lower must-run impact

        # ERCOT demand elasticity: 4CP program + load shed contracts
        # Higher threshold ($300) because ORDC already captures scarcity efficiently
        # 4CP industrial curtailment is ~3-5 GW responsive load
        self.demand_elasticity_threshold = 300.0
        self.demand_elasticity_max_curtailment = 0.10  # 10% — 4CP + voluntary
        self.demand_elasticity_damping = 0.45  # ORDC already moderates, so less damping

        # ERCOT demand-quantile calibration (v11.1 — SOM iterative calibration)
        # 2024 was mild: solar+storage entry reduced shortage pricing dramatically
        # Modo Energy 2024: avg RT ~$26/MWh, down from $63 in 2023
        # Key: ERCOT is energy-only → higher volatility than capacity markets
        self.dq_high_percentile = 80
        self.dq_high_max_adder = 28.0
        self.dq_high_exponent = 2.0
        self.dq_scarcity_percentile = 98.5
        self.dq_scarcity_max = 220.0
        self.dq_low_percentile = 12      # v11.3: 14→12, reduce neg hrs 517→~400
        self.dq_low_floor = -50.0        # v11.3: -45→-50, push P10 toward $5
        self.dq_low_exponent = 1.5
        # Mid-low compression: cheap overnight wind+baseload
        self.dq_midlow_percentile = 72
        self.dq_midlow_discount = 0.70

    def _scarcity_adder(self, reserve_ratio, demand_mw):
        """ERCOT ORDC: smooth exponential adder, not a hard cap."""
        reserve_mw = reserve_ratio * demand_mw
        if reserve_mw >= self.ordc_knee_mw:
            return 0.0
        # Exponential LOLP curve: LOLP ≈ exp(-λ × reserve_mw)
        # v11: lam 0.002→0.004 for steeper decay, cap ORDC adder at $500
        lam = 0.004
        lolp = np.exp(-lam * max(0, reserve_mw))
        return min(500.0, self.voll * lolp)


class CAISOPriceModel(PriceModel):
    """CAISO: Resource Adequacy, aggressive negative prices (solar duck curve)."""

    def __init__(self, fuel_level='Medium'):
        super().__init__('CAISO', fuel_level)
        self.scarcity_cap = 2000.0
        self.floor_price = -60.0
        self.surplus_decay = 0.022        # v11: 0.030→0.022, reduce neg hrs 889→~600
        self.scarcity_threshold = 0.03    # v11: 0.05→0.03
        self.must_run_depression = 0.15  # CAISO: no coal, gas-only; minimal must-run

        # CAISO demand elasticity: RDRR + PDR programs, ~2 GW responsive
        # RA market provides baseline, DR adds during evening ramp stress
        self.demand_elasticity_threshold = 200.0
        self.demand_elasticity_max_curtailment = 0.10  # 10% — RA + PDR
        self.demand_elasticity_damping = 0.50

        # CAISO demand-quantile calibration (v11.1 — SOM iterative calibration)
        # CAISO DMM 2024: avg ~$38, huge solar surplus midday, evening ramp premium
        # CAISO has largest peak/offpeak spread of any ISO (duck curve)
        self.dq_high_percentile = 72     # v11.2: 78→72, need P75→$50 (more hours get adder)
        self.dq_high_max_adder = 80.0    # v11.2: 75→80, target peak $55
        self.dq_high_exponent = 2.0
        self.dq_scarcity_percentile = 99 # v11.2: 97.5→99, reduce scarcity 146→~60
        self.dq_scarcity_max = 160.0     # v11.2: 280→160
        self.dq_low_percentile = 15
        self.dq_low_floor = -50.0
        self.dq_low_exponent = 1.5
        # Mid-low: solar midday compression — P25 target $12
        self.dq_midlow_percentile = 65
        self.dq_midlow_discount = 0.52


class NYISOPriceModel(PriceModel):
    """NYISO: ICAP capacity market. Similar to PJM but tighter geography."""

    def __init__(self, fuel_level='Medium'):
        super().__init__('NYISO', fuel_level)
        self.scarcity_cap = 2000.0
        self.floor_price = -20.0
        self.surplus_decay = 0.008         # v11: 0.012→0.008, reduce neg hrs 273→~150
        self.scarcity_threshold = 0.03     # v11: 0.06→0.03
        self.must_run_depression = 0.15  # NYISO: no coal, gas-only

        # NYISO demand elasticity: ICAP SCR/EDRP programs, ~1.3 GW responsive
        # NYC load pocket has limited DR flexibility
        self.demand_elasticity_threshold = 200.0
        self.demand_elasticity_max_curtailment = 0.12  # 12% — ICAP DR programs
        self.demand_elasticity_damping = 0.50

        # NYISO demand-quantile calibration (v11.1 — SOM iterative calibration)
        # Potomac Economics 2024: avg $42, tight geography → congestion, ICAP dampens
        self.dq_high_percentile = 76
        self.dq_high_max_adder = 50.0    # v11.2: 55→50
        self.dq_high_exponent = 2.0
        self.dq_scarcity_percentile = 98.5  # v11.2: 97.5→98.5, reduce scarcity 132→~70
        self.dq_scarcity_max = 160.0     # v11.2: 250→160
        self.dq_low_percentile = 7
        self.dq_low_floor = -20.0
        self.dq_low_exponent = 1.5
        # Mid-low compression: overnight baseload
        self.dq_midlow_percentile = 68   # v11.2: 65→68
        self.dq_midlow_discount = 0.45   # v11.2: 0.30→0.45, target P10 $18


class NEISOPriceModel(PriceModel):
    """NEISO: FCM capacity market. Winter gas pipeline constraint creates scarcity."""

    WINTER_MONTHS = {12, 1, 2}  # Dec, Jan, Feb
    NEISO_WINTER_GAS_ADDER = 13.13  # $/MWh — from Step 4

    def __init__(self, fuel_level='Medium'):
        super().__init__('NEISO', fuel_level)
        self.scarcity_cap = 400.0          # v11.2: 800→400, FCM + 4GW imports cap scarcity
        self.floor_price = -25.0
        self.surplus_decay = 0.008         # v11: 0.012→0.008, reduce neg hrs 314→~180
        self.scarcity_threshold = 0.02     # v11.1: 0.03→0.02, imports + FCM provide reserves
        self.must_run_depression = 0.15  # NEISO: no coal, gas-only

        # NEISO demand elasticity: FCM passive DR + active DR, ~1.5 GW responsive
        # Winter gas constraint limits DR effectiveness (can't substitute fuel)
        self.demand_elasticity_threshold = 200.0
        self.demand_elasticity_max_curtailment = 0.12  # 12% — FCM DR
        self.demand_elasticity_damping = 0.45  # less effective in winter gas events

        # NEISO demand-quantile calibration (v11.1 — SOM iterative calibration)
        # ISO-NE 2024 EMM: avg $39.50, winter gas pipeline premium, FCM capacity market
        # NEISO has only 16 GW fossil but 4+ GW import capability (NYISO, HQ, NB Power)
        # — scarcity pricing must reflect import availability
        self.dq_high_percentile = 80       # v11.1: 82→80
        self.dq_high_max_adder = 25.0      # v11.1: 28→25
        self.dq_high_exponent = 2.0
        self.dq_scarcity_percentile = 99.5
        self.dq_scarcity_max = 50.0        # v11.1: 80→50
        self.dq_low_percentile = 8
        self.dq_low_floor = -25.0
        self.dq_low_exponent = 1.5
        # Mid-low compression: overnight baseload pricing
        self.dq_midlow_percentile = 68     # v11.1: 65→68
        self.dq_midlow_discount = 0.55     # v11.1: 0.50→0.55

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


class MISOPriceModel(PriceModel):
    """MISO: PRA capacity market, coal-heavy fleet, significant wind congestion.

    MISO 2024 SOM (Potomac Economics): avg RT LMP ~$31/MWh, 14% decrease from 2023.
    VOLL: $3,500/MWh (2024), approved increase to $10,000 effective Sept 2025.
    ORDC upper bound proposed at $6,000/MWh.
    PRA capacity: $30/MW-day (summer 2024), Zone 5 outlier $719.81.
    Coal: 35% of fossil fleet (highest among ISOs with PJM at 29%).
    Wind: ~14.5% of generation, drives ~40% of real-time congestion.
    """

    def __init__(self, fuel_level='Medium'):
        super().__init__('MISO', fuel_level)
        self.scarcity_cap = 3500.0     # 2024 VOLL (pre-$10K increase)
        self.floor_price = -30.0       # moderate negative pricing
        self.surplus_decay = 0.015     # v11.1: 0.018→0.015, reduce neg hrs 400→~300
        self.scarcity_threshold = 0.02  # v11: 0.05→0.02, PRA provides ample cushion
        self.must_run_depression = 0.35  # MISO: 35% coal, heavy must-run floor

        # MISO demand elasticity: LMR (Load-Modifying Resources), ~8 GW responsive
        # Large industrial load base (aluminum, steel, refining) with interruptible contracts
        self.demand_elasticity_threshold = 200.0
        self.demand_elasticity_max_curtailment = 0.13  # 13% — LMR + industrial
        self.demand_elasticity_damping = 0.50

        # MISO demand-quantile calibration (v11 — SOM iterative calibration)
        # Potomac Economics 2024: avg RT $31/MWh, 14% decrease from 2023
        # Coal-heavy fleet (35%), wind congestion drives ~40% of RT congestion
        self.dq_high_percentile = 76
        self.dq_high_max_adder = 38.0
        self.dq_high_exponent = 2.0
        self.dq_scarcity_percentile = 97.5
        self.dq_scarcity_max = 140.0     # v11.3: 160→140, scarcity 64→~50
        self.dq_low_percentile = 12
        self.dq_low_floor = -30.0
        self.dq_low_exponent = 1.5
        # Mid-low compression: coal baseload + wind off-peak
        self.dq_midlow_percentile = 72   # v11.3: 68→72, wider band
        self.dq_midlow_discount = 0.70   # v11.3: 0.65→0.70, deeper offpeak ($32→$25)


class SPPPriceModel(PriceModel):
    """SPP: Limited capacity market, cheapest US wholesale market, extreme wind.

    SPP 2024 SOM (SPP MMU): avg RT LMP $26.18/MWh, down 4% from 2023.
    Wind: 37.1% of generation (~34.6 GW nameplate), drives massive negative pricing.
    Wind markups: avg -$37.99/MWh in 2024.
    Very limited capacity market — closer to energy-only than capacity-market ISOs.
    Gas prices very low in footprint ($1.65/MMBtu summer 2024).
    Cheapest wholesale electricity market in the US.
    """

    def __init__(self, fuel_level='Medium'):
        super().__init__('SPP', fuel_level)
        self.scarcity_cap = 3500.0     # SPP uses similar VOLL to MISO
        self.floor_price = -40.0       # deeper negative prices than MISO (more wind surplus)
        self.surplus_decay = 0.025     # steep negative pricing from wind over-generation
        self.scarcity_threshold = 0.02  # v11: 0.06→0.02
        self.must_run_depression = 0.30  # SPP: 30% coal, moderate must-run

        # SPP demand elasticity: limited formal DR, some interruptible industrial
        # Lowest price market → fewer hours hit threshold, but less DR infrastructure
        self.demand_elasticity_threshold = 200.0
        self.demand_elasticity_max_curtailment = 0.08  # 8% — limited DR programs
        self.demand_elasticity_damping = 0.45

        # SPP demand-quantile calibration (v11 — SOM iterative calibration)
        # SPP MMU 2024: avg RT $26.18, cheapest US market
        # Wind 37.1% of generation, markups avg -$38/MWh
        # Very low gas prices ($1.65/MMBtu summer 2024), flat geography = low congestion
        self.dq_high_percentile = 76
        self.dq_high_max_adder = 38.0
        self.dq_high_exponent = 2.0
        self.dq_scarcity_percentile = 98    # v11.3: 97.5→98, scarcity 49→~30
        self.dq_scarcity_max = 130.0     # v11.3: 160→130
        self.dq_low_percentile = 22
        self.dq_low_floor = -30.0
        self.dq_low_exponent = 1.8
        # Mid-low compression: massive wind makes off-peak very cheap
        self.dq_midlow_percentile = 75   # v11.3: 72→75, wider band
        self.dq_midlow_discount = 0.75   # v11.3: 0.72→0.75, deeper offpeak ($30→$20)


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
        'MISO': MISOPriceModel,
        'SPP': SPPPriceModel,
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
        mix.get('offshore_wind', 0), mix.get('ccs_ccgt', 0), mix.get('hydro', 0),
    )
    return f"{mix_tuple}_{fuel_level}_{threshold}"


# ══════════════════════════════════════════════════════════════════════════════
# HOURLY LMP COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════


def compute_hourly_lmp_vectorized(dispatch_result, demand_mw_profile, stack, price_model,
                                   iso=None, vre_penetration=None):
    """Vectorized LMP computation — faster than per-hour loop.

    Uses the merit-order stack as a step function and np.searchsorted for
    marginal unit identification.

    Args:
        vre_penetration: VRE share as fraction (0-1). When provided, scales
            negative pricing parameters to reflect increasing VRE surplus at
            higher penetration levels. Calibrated against observed ISO data:
            at 25% VRE (2024 baseline), no adjustment; at 50%+ VRE (trajectory
            2030+), negative price frequency and depth increase per CAISO/ERCOT
            empirical trends. Source: CAISO DMM Q-Reports 2019-2024 showing
            negative price hours scaling ~linearly with solar penetration.
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
            # Gas CCGT heat rate varies ~6.7-7.5 MMBtu/MWh across load range (~12%)
            # Gas CT varies ~9.5-11.5 (~20%). Coal ~9.5-11.0 (~16%).
            # Ramp applies to total MC so use ~15% (not full heat rate variation)
            heat_rate_ramp = 1.0 + 0.15 * position_in_band ** 1.5
            normal_prices = base_prices * heat_rate_ramp

            # Reserve margin check for scarcity adder
            # Use TOTAL stack remaining (not within-band), reflecting actual system reserves
            pos_demand = demand_mw_profile[pos_mask][normal_mask]
            remaining_cap = total_fossil_cap - normal_residual
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

    # ══════════════════════════════════════════════════════════════════════
    # MUST-RUN / MIN-GEN PRICING LAYER
    # ══════════════════════════════════════════════════════════════════════
    # Nuclear is fully must-run, coal steam has ~40% min stable generation.
    # When residual demand drops below the total must-run floor, these units
    # can't economically cycle off — they bid at or below marginal cost to
    # stay dispatched, depressing off-peak prices. This physically grounds
    # off-peak price depression instead of relying solely on statistical
    # demand-quantile adjustments.
    must_run_caps = np.zeros(n_units, dtype=np.float64)
    for i, (unit_type, cap_mw, mc) in enumerate(stack):
        must_run_caps[i] = cap_mw * MUST_RUN_PCT.get(unit_type, 0.0)
    total_must_run_mw = must_run_caps.sum()

    if total_must_run_mw > 0:
        must_run_surplus_mask = pos_mask & (residual_mw < total_must_run_mw)
        if must_run_surplus_mask.any():
            surplus_ratio = (total_must_run_mw - residual_mw[must_run_surplus_mask]) / total_must_run_mw
            surplus_ratio = np.clip(surplus_ratio, 0.0, 1.0)
            depression = surplus_ratio * price_model.must_run_depression
            hourly_lmp[must_run_surplus_mask] *= (1.0 - depression)

    # ══════════════════════════════════════════════════════════════════════
    # DEMAND-QUANTILE PRICING LAYER
    # ══════════════════════════════════════════════════════════════════════
    # Real-world LMP deviates from pure merit-order due to:
    #   - Transmission congestion (high-demand hours)
    #   - Gas supply tightness (winter peaks, pipeline limits)
    #   - Bid markup above marginal cost (generator bidding behavior)
    #   - Must-run nuclear/wind surplus (overnight negative prices)
    # This layer adds demand-quantile-dependent adders calibrated against
    # actual PJM price distribution statistics.

    # Compute demand percentile rank for each hour (0-1)
    demand_sorted = np.sort(demand_mw_profile)
    demand_rank = np.searchsorted(demand_sorted, demand_mw_profile, side='right') / H

    # --- SCARCITY PRICING: ORDC or DEMAND-QUANTILE ---
    # Controlled by pipeline_config.SCARCITY_MODE:
    #   'ordc' — Operating Reserve Demand Curve: price adder = VOLL × LOLP(reserves)
    #            Physically responsive to generation mix (more solar → more midday
    #            reserves → lower midday scarcity, but evening scarcity unchanged).
    #   'demand_quantile' — Legacy: demand-percentile congestion/scarcity overlays.
    #            Reproduces historical LMP shapes but cannot predict structural response.
    if SCARCITY_MODE == 'ordc':
        # Compute hourly operating reserves (MW) from fossil stack headroom
        # reserves = total fossil capacity available - fossil demand dispatched
        reserves_mw = np.maximum(total_fossil_cap - residual_mw, 0.0)
        # ORDC adder: sigmoid LOLP × VOLL — near $0 when reserves ample,
        # rises steeply as reserves fall below ISO-specific target.
        # Single numpy vectorized sigmoid on 8760-element array (<0.1ms).
        ordc_adder = price_model.compute_ordc_adder(reserves_mw)
        # Only apply to hours with positive residual demand (fossil dispatching)
        hourly_lmp[pos_mask] += ordc_adder[pos_mask]
    else:
        # --- HIGH-DEMAND CONGESTION/TIGHTNESS ADDER (demand-quantile mode) ---
        # Hours above dq_high_percentile get increasing adder
        high_threshold = price_model.dq_high_percentile / 100.0
        high_mask = demand_rank > high_threshold
        if high_mask.any():
            # Normalized position: 0 at threshold, 1 at rank=1.0
            high_position = (demand_rank[high_mask] - high_threshold) / (1.0 - high_threshold)
            high_position = np.clip(high_position, 0.0, 1.0)
            # Exponential ramp: most adder concentrated on hottest hours
            high_adder = price_model.dq_high_max_adder * (high_position ** price_model.dq_high_exponent)
            hourly_lmp[high_mask] += high_adder

        # --- SCARCITY TAIL (demand-quantile mode) ---
        # Extreme high-demand hours get additional scarcity-like pricing
        # (represents penalty factor / emergency pricing / ORDC tail)
        scarcity_threshold = price_model.dq_scarcity_percentile / 100.0
        scarcity_mask = demand_rank > scarcity_threshold
        if scarcity_mask.any():
            scar_position = (demand_rank[scarcity_mask] - scarcity_threshold) / (1.0 - scarcity_threshold)
            scar_position = np.clip(scar_position, 0.0, 1.0)
            # Linear scarcity ramp — distributes adder evenly across tail hours
            # This avoids extreme concentration that causes excess volatility
            scarcity_adder = price_model.dq_scarcity_max * scar_position
            hourly_lmp[scarcity_mask] += scarcity_adder

    # --- VRE-SCALED NEGATIVE PRICING PARAMETERS ---
    # At higher VRE penetration, negative pricing becomes more frequent and deeper.
    # Scaling calibrated against CAISO DMM data (2019-2024): negative price hours
    # increased ~40% as solar went from 20% to 35% of generation.
    # Baseline: 25% VRE (≈2024 US average). Above baseline, floor deepens and
    # the low-demand percentile band widens (more hours see negative prices).
    vre_floor_scale = 1.0
    vre_pct_scale = 1.0
    if vre_penetration is not None and vre_penetration > 0.25:
        # Linear scaling: each 10pp of VRE above 25% deepens floor by 15%
        # and widens negative-price band by 10%
        vre_excess = vre_penetration - 0.25
        vre_floor_scale = 1.0 + 1.5 * vre_excess   # e.g., 50% VRE → 1.375×
        vre_pct_scale = 1.0 + 1.0 * vre_excess      # e.g., 50% VRE → 1.25×
        # Cap scaling at 2× (prevents unrealistic extremes at very high VRE)
        vre_floor_scale = min(vre_floor_scale, 2.0)
        vre_pct_scale = min(vre_pct_scale, 2.0)

    # --- LOW-DEMAND NEGATIVE PRICING ---
    # Hours below dq_low_percentile get depressed/negative prices
    # Must-run nuclear + wind surplus → negative LMP
    effective_low_pct = price_model.dq_low_percentile * vre_pct_scale
    effective_low_floor = price_model.dq_low_floor * vre_floor_scale
    low_threshold = effective_low_pct / 100.0
    low_mask = demand_rank < low_threshold
    if low_mask.any():
        # Normalized position: 1 at lowest demand, 0 at threshold
        low_position = 1.0 - (demand_rank[low_mask] / low_threshold)
        low_position = np.clip(low_position, 0.0, 1.0)
        # Price depression: from merit-order price toward floor
        depression = low_position ** price_model.dq_low_exponent
        # Depress toward floor: lerp from current price toward (VRE-scaled) floor
        current = hourly_lmp[low_mask]
        target_price = effective_low_floor
        hourly_lmp[low_mask] = current * (1.0 - depression) + target_price * depression

    # --- MID-LOW PRICE COMPRESSION ---
    # Hours between low_percentile and midlow_percentile get mild depression
    # Represents cheap overnight baseload pricing, low congestion
    midlow_pct = getattr(price_model, 'dq_midlow_percentile', 0)
    midlow_discount = getattr(price_model, 'dq_midlow_discount', 0)
    if midlow_pct > 0 and midlow_discount > 0:
        midlow_threshold = midlow_pct / 100.0
        midlow_mask = (demand_rank >= low_threshold) & (demand_rank < midlow_threshold)
        if midlow_mask.any():
            # Normalized position: 1 at low_threshold, 0 at midlow_threshold
            midlow_position = 1.0 - (demand_rank[midlow_mask] - low_threshold) / (midlow_threshold - low_threshold)
            midlow_position = np.clip(midlow_position, 0.0, 1.0)
            # Gentle linear discount
            discount = midlow_discount * midlow_position
            hourly_lmp[midlow_mask] *= (1.0 - discount)

    # --- CLEAN SURPLUS MERIT-ORDER EFFECT (v11.3) ---
    # For high-renewable ISOs, hours with significant clean surplus get
    # additional price depression even if total demand is moderate.
    # This captures the merit-order effect of zero-marginal-cost renewables
    # displacing fossil from the supply stack. Without this, the demand-quantile
    # layer misses midday solar surplus in CAISO (P10 target -$5, P25 target $12)
    # and overnight wind surplus in SPP/ERCOT (P10 targets $5-8).
    if surplus_mw.max() > 0:
        surplus_ratio = surplus_mw / (demand_mw_profile + 1)
        surplus_thresh = 0.03  # >3% surplus triggers depression
        surplus_active = surplus_ratio > surplus_thresh
        if surplus_active.any():
            # Scale: 3% surplus → mild effect, 20%+ → strong effect
            surplus_factor = np.clip((surplus_ratio[surplus_active] - surplus_thresh) * 8, 0, 1)
            current = hourly_lmp[surplus_active]
            floor = effective_low_floor
            # Depress toward floor: stronger with more surplus
            depressed = current * (1 - surplus_factor * 0.6) + floor * surplus_factor * 0.6
            # Only depress, never increase
            hourly_lmp[surplus_active] = np.minimum(current, depressed)

    # ══════════════════════════════════════════════════════════════════════
    # DEMAND ELASTICITY FOR EXTREME PRICES (Rec #8)
    # ══════════════════════════════════════════════════════════════════════
    # At extreme prices (>$200/MWh), price-responsive industrial loads curtail.
    # Interruptible contracts, demand response programs (PJM Economic DR,
    # ERCOT 4CP, CAISO PDR/RDRR, MISO LMR), and voluntary curtailment
    # reduce effective demand by 5-15%, moderating scarcity pricing.
    #
    # Implementation: post-pricing dampening. For each hour where computed
    # LMP exceeds the elasticity threshold, compute curtailment fraction
    # proportional to price excess, then reduce price toward the threshold.
    # This approximates the equilibrium price that would result from
    # re-dispatching at reduced demand without requiring iterative solve.
    #
    # Sources: FERC Order 745 DR assessment (2024), PJM IMM DR participation,
    # ERCOT PUCT 4CP program data, CAISO DMM DR effectiveness reports.
    threshold = price_model.demand_elasticity_threshold
    max_curtail = price_model.demand_elasticity_max_curtailment
    damping = price_model.demand_elasticity_damping
    extreme_mask = hourly_lmp > threshold
    if extreme_mask.any():
        excess_ratio = (hourly_lmp[extreme_mask] - threshold) / threshold
        # Curtailment ramps logarithmically: quick onset, diminishing returns
        # At 2× threshold ($400): ~60% of max curtailment
        # At 5× threshold ($1000): ~90% of max curtailment
        curtailment_frac = max_curtail * (1.0 - np.exp(-1.5 * excess_ratio))
        # Damped price: reduce extreme price toward threshold
        # damping=0.5 means each 1% curtailment → 0.5% price reduction
        price_reduction = curtailment_frac * damping
        hourly_lmp[extreme_mask] *= (1.0 - price_reduction)
        # Floor at threshold — demand response shouldn't push price below onset
        hourly_lmp[extreme_mask] = np.maximum(hourly_lmp[extreme_mask], threshold)

    # NEISO winter gas adder — demand-dependent (v11.1)
    # Pipeline constraint only bites during peak winter hours (cold snaps),
    # not all 2160 winter hours. Apply adder only to hours above P60 demand
    # within winter months, scaled by demand intensity.
    if isinstance(price_model, NEISOPriceModel):
        month_hours = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
        winter_demand_threshold = np.percentile(demand_mw_profile, 60)
        p90_demand = np.percentile(demand_mw_profile, 90)
        h = 0
        for m_idx, mh in enumerate(month_hours):
            month = m_idx + 1
            if month in NEISOPriceModel.WINTER_MONTHS:
                winter_slice = slice(h, h + mh)
                winter_pos = pos_mask[winter_slice]
                winter_demand = demand_mw_profile[winter_slice]
                # Only apply adder when demand > P60 AND fossil is dispatching
                high_demand_winter = winter_pos & (winter_demand > winter_demand_threshold)
                # Scale adder by demand intensity: full at P90+, partial below
                demand_intensity = np.clip(
                    (winter_demand - winter_demand_threshold) /
                    (p90_demand - winter_demand_threshold + 1),
                    0.0, 1.0)
                scaled_adder = NEISOPriceModel.NEISO_WINTER_GAS_ADDER * demand_intensity
                hourly_lmp[winter_slice] = np.where(
                    high_demand_winter,
                    hourly_lmp[winter_slice] + scaled_adder,
                    hourly_lmp[winter_slice])
            h += mh

    # DR is applied as post-processing in market_simulation.py so it works
    # with both zonal and copper-plate LMP paths.
    dr_curtailed_mw = np.zeros(H, dtype=np.float64)
    return hourly_lmp, hourly_marginal_unit, dr_curtailed_mw


# ══════════════════════════════════════════════════════════════════════════════
# ZONAL LMP COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_hourly_lmp_zonal(dispatch_result, demand_mw_profile, zone_stacks,
                              zone_config, price_model, iso=None,
                              vre_penetration=None):
    """Zonal LMP computation — replaces copper-plate when zone data available.

    Solves a per-hour LP across 2-5 zones with inter-zonal transfer limits,
    then applies the same demand-quantile pricing layers used by the
    copper-plate model.

    Args:
        dispatch_result: dict with 'residual_demand' and 'curtailed' arrays
        demand_mw_profile: np.array (8760,) hourly demand in MW
        zone_stacks: dict {zone_name: [(unit_type, cap_mw, mc), ...]}
        zone_config: dict from ZONE_CONFIG[iso] with zones, demand_share,
            transfer_limits_mw
        price_model: PriceModel instance
        iso: ISO name (for special rules)
        vre_penetration: VRE share as fraction (0-1)

    Returns:
        zonal_lmp_matrix: np.array (n_zones, H) — $/MWh per zone per hour
        system_lmp: np.array (H,) — demand-weighted system average (backward compat)
        hourly_marginal_unit: np.array (H,) — set to -1 (not applicable for zonal)
        zonal_stats: dict with per-zone summary statistics
    """
    from zonal_lmp import compute_zonal_lmp_hourly

    H = len(demand_mw_profile)

    zonal_lmp_matrix, system_lmp, flows_matrix, zonal_stats = \
        compute_zonal_lmp_hourly(
            iso=iso,
            zone_config=zone_config,
            zone_stacks=zone_stacks,
            demand_mw_profile=demand_mw_profile,
            clean_supply_by_zone=None,  # TODO: split clean supply by zone
            price_model=price_model,
            vre_penetration=vre_penetration,
        )

    # Backward-compatible marginal unit array (not meaningful for zonal)
    hourly_marginal_unit = np.full(H, -1, dtype=np.int8)

    return zonal_lmp_matrix, system_lmp, hourly_marginal_unit, zonal_stats


# ══════════════════════════════════════════════════════════════════════════════
# LMP STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_lmp_stats(hourly_lmp, hourly_marginal_unit, demand_mw_profile,
                       dispatch_result):
    """Compute summary statistics from 8760 hourly LMP array.

    Returns dict matching the output schema in SPEC.md.
    """
    # Peak/off-peak classification (7am-11pm weekdays) — vectorized
    hours = np.arange(H)
    day_of_week = (hours // 24 + 2) % 7  # 2025: Jan 1 = Wednesday (offset 2), 0=Mon
    hour_of_day = hours % 24
    peak_mask = (day_of_week < 5) & (hour_of_day >= 7) & (hour_of_day <= 22)

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
    """Load scenarios from per-ISO parquets (step4 preferred, step3 fallback).

    Returns list of scenario dicts with resource mix, dispatch params, costs.
    """
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    # Determine input directory
    input_dir = STEP3_PARQUET_DIR

    def _find_iso_parquet(d, iso_name):
        for prefix in ['step3_co_', 'step3_']:
            path = os.path.join(d, f'{prefix}{iso_name}.parquet')
            if os.path.exists(path):
                return path
        return None

    if iso:
        # Load single ISO
        path = _find_iso_parquet(input_dir, iso)
        if path is None:
            print(f"    WARNING: No parquet found for {iso}")
            return []
        table = pq.read_table(path)
        print(f"    Loaded {os.path.basename(path)}: {table.num_rows:,} rows")
    else:
        # Load all ISOs
        tables = []
        for iso_name in ISOS:
            path = _find_iso_parquet(input_dir, iso_name)
            if path:
                tables.append(pq.read_table(path))
        if not tables:
            print(f"    WARNING: No parquets found in {input_dir}")
            return []
        import pyarrow
        table = pyarrow.concat_tables(tables, promote_options='permissive')

    if threshold is not None:
        table = table.filter(pc.equal(table.column('threshold'), float(threshold)))

    # Batch convert Arrow table → list of dicts (avoids per-cell .as_py() overhead)
    rows = table.to_pandas().to_dict('records')
    mix_cols = ('mix_clean_firm', 'mix_solar', 'mix_wind',
                'mix_offshore_wind', 'mix_ccs_ccgt', 'mix_hydro')
    mix_keys = ('clean_firm', 'solar', 'wind',
                'offshore_wind', 'ccs_ccgt', 'hydro')
    for row in rows:
        row['resource_mix'] = {k: row.get(mc, 0) for k, mc in zip(mix_keys, mix_cols)}

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
        batt4 = sc.get('battery_dispatch_pct', 0)
        batt8 = sc.get('battery8_dispatch_pct', 0)
        ldes = sc.get('ldes_dispatch_pct', 0)
        h2 = sc.get('h2_dispatch_pct', 0)

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

        dispatch, cache_hit = get_or_compute_dispatch(
            iso, demand_norm, supply_profiles, resource_mix,
            battery_dispatch_pct=batt4, battery8_dispatch_pct=batt8,
            ldes_dispatch_pct=ldes, cache=dispatch_cache)

        if cache_hit:
            cache_hits += 1
        else:
            cache_misses += 1

        # Build merit-order stack for this threshold (RA+GAF aware)
        stack, total_fossil_mw = build_merit_order_stack(
            iso, threshold, fuel_level,
            resource_mix=resource_mix,
            battery_pct=batt4, battery8_pct=batt8, ldes_pct=ldes,
            h2_pct=h2)

        # Compute hourly LMP
        hourly_lmp, hourly_mu, _dr = compute_hourly_lmp_vectorized(
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
    if cache_misses > 0:
        print(f"    WARNING: {cache_misses} cache misses — consider running step4_build_dispatch_cache.py first "
              f"to pre-populate the cache.")
    return results


def save_iso_results(iso, results):
    """Save LMP results for an ISO."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(LMP_DIR, exist_ok=True)

    if results:
        all_keys = list(dict.fromkeys(k for r in results for k in r))
        arrays = [pa.array([r.get(k) for r in results]) for k in all_keys]
        table = pa.table(dict(zip(all_keys, arrays)))
        stats_path = os.path.join(LMP_DIR, f'{iso}_lmp.parquet')
        pq.write_table(table, stats_path, compression='zstd')
        print(f"    {iso}_lmp.parquet: {table.num_rows} rows, "
              f"{os.path.getsize(stats_path) / 1024:.0f} KB")


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
              f"Wind={resource_mix['wind']}% OSW={resource_mix.get('offshore_wind', 0)}% "
              f"CCS={resource_mix['ccs_ccgt']}% Hydro={resource_mix['hydro']}%")
        print(f"    Batt4: {sc.get('battery_dispatch_pct', 0)}%, "
              f"LDES: {sc.get('ldes_dispatch_pct', 0)}%, "
              f"H2: {sc.get('h2_dispatch_pct', 0)}%")

        dispatch = reconstruct_hourly_dispatch(
            demand_norm, supply_profiles, resource_mix,
            battery_dispatch_pct=sc.get('battery_dispatch_pct', 0),
            battery8_dispatch_pct=sc.get('battery8_dispatch_pct', 0),
            ldes_dispatch_pct=sc.get('ldes_dispatch_pct', 0),
            h2_dispatch_pct=sc.get('h2_dispatch_pct', 0))

        # Merit-order stack (RA + GAF aware)
        batt4_pct = sc.get('battery_dispatch_pct', 0)
        batt8_pct = sc.get('battery8_dispatch_pct', 0)
        ldes_pct = sc.get('ldes_dispatch_pct', 0)
        h2_pct = sc.get('h2_dispatch_pct', 0)
        stack, fossil_mw = build_merit_order_stack(
            iso, sc['threshold'], fuel_level,
            resource_mix=resource_mix,
            battery_pct=batt4_pct, battery8_pct=batt8_pct, ldes_pct=ldes_pct,
            h2_pct=h2_pct)
        print(f"    Fossil stack ({fossil_mw:,.0f} MW):")
        for unit_type, cap, mc in stack:
            print(f"      {unit_type:>12}: {cap:>8,.0f} MW @ ${mc:.2f}/MWh")

        # Price model
        price_model = get_price_model(iso, fuel_level)

        # LMP computation
        hourly_lmp, hourly_mu, _dr = compute_hourly_lmp_vectorized(
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

    # All 7 ISOs now have calibrated price models (PJM, ERCOT, CAISO, NYISO, NEISO, MISO, SPP)
    if args.iso == 'ALL':
        isos_to_run = ISOS
    elif args.iso:
        isos_to_run = [args.iso]
    else:
        isos_to_run = ['PJM']  # default to PJM for backward compatibility
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

            results = run_lmp_for_iso(
                iso, scenarios, demand_data, gen_profiles,
                fuel_level=fuel_level, dispatch_cache=dispatch_cache)

            iso_results.extend(results)

        # Save dispatch cache (with new entries appended)
        save_dispatch_cache(iso, dispatch_cache)
        print(f"    Dispatch cache saved: {len(dispatch_cache)} entries")

        # Save ISO results
        save_iso_results(iso, iso_results)
        all_results.extend(iso_results)

        elapsed = time.time() - iso_start
        print(f"    {iso} complete: {len(iso_results)} scenarios — {elapsed:.0f}s")

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
