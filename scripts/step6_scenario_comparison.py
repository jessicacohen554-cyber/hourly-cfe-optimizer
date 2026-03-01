#!/usr/bin/env python3
"""
Dual-scenario comparison: Pure Consequential vs Hourly Matching.

Compares two procurement strategies:
  Scenario A (Pure Consequential): chase cheapest $/tCO₂ sequentially,
    locking in prior resource commitments at each threshold step.
  Scenario B (Hourly Matching): target a high CFE threshold directly,
    co-optimizing the full resource mix with clean firm deployed along
    the FOAK→NOAK learning curve.

Uses feasible mixes from shared-data.js pipeline output + cost tables
from step3, then traces through dispatch_utils for emission rates.

Output: JSON + JS data files for the comparison dashboard page.
"""

import json
import os
import re
import sys
import functools
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from scipy.interpolate import PchipInterpolator
from scipy.optimize import isotonic_regression

# Import dispatch utilities for emission rates and dispatch cache
sys.path.insert(0, str(Path(__file__).parent))
from dispatch_utils import (
    compute_fossil_retirement, compute_co2_from_dispatch,
    BASE_DEMAND_TWH, GRID_MIX_SHARES,
    COAL_CAP_TWH, OIL_CAP_TWH, COAL_OIL_RETIREMENT_THRESHOLD,
    H, load_common_data, get_supply_profiles, get_demand_profile,
    build_supply_matrix, reconstruct_hourly_dispatch,
    _archetype_key, load_dispatch_cache, get_or_compute_dispatch,
)

# ============================================================================
# CONSTANTS (from step3_cost_optimization.py)
# ============================================================================

from step3_cost_optimization import OUTPUT_THRESHOLDS as THRESHOLDS

ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']
RESOURCES = ['clean_firm', 'solar', 'wind', 'ccs_ccgt', 'hydro']

# Hydro is ALWAYS existing-only: capped at 2025 absolute TWh on the
# 2021-2026 weather-normalized curve.  Never grows with demand.
HYDRO_CAP_TWH = {iso: GRID_MIX_SHARES[iso].get('hydro', 0) / 100.0 * BASE_DEMAND_TWH[iso]
                 for iso in ISOS}

WHOLESALE_PRICES = {'CAISO': 30, 'ERCOT': 27, 'PJM': 34, 'NYISO': 42, 'NEISO': 41, 'MISO': 30, 'SPP': 25}
FUEL_ADJUSTMENTS = {
    'CAISO': {'Low': -5, 'Medium': 0, 'High': 10},
    'ERCOT': {'Low': -7, 'Medium': 0, 'High': 12},
    'PJM':   {'Low': -6, 'Medium': 0, 'High': 11},
    'NYISO': {'Low': -4, 'Medium': 0, 'High': 8},
    'NEISO': {'Low': -4, 'Medium': 0, 'High': 8},
    'MISO':  {'Low': -6, 'Medium': 0, 'High': 11},
    'SPP':   {'Low': -7, 'Medium': 0, 'High': 12},
}

LCOE_TABLES = {
    'solar': {
        'Low':    {'CAISO': 45, 'ERCOT': 40, 'PJM': 50, 'NYISO': 70, 'NEISO': 62, 'MISO': 48, 'SPP': 43},
        'Medium': {'CAISO': 60, 'ERCOT': 54, 'PJM': 65, 'NYISO': 92, 'NEISO': 82, 'MISO': 62, 'SPP': 57},
        'High':   {'CAISO': 78, 'ERCOT': 70, 'PJM': 85, 'NYISO': 120, 'NEISO': 107, 'MISO': 82, 'SPP': 74},
    },
    'wind': {
        'Low':    {'CAISO': 55, 'ERCOT': 30, 'PJM': 47, 'NYISO': 61, 'NEISO': 55, 'MISO': 33, 'SPP': 28},
        'Medium': {'CAISO': 73, 'ERCOT': 40, 'PJM': 62, 'NYISO': 81, 'NEISO': 73, 'MISO': 43, 'SPP': 37},
        'High':   {'CAISO': 95, 'ERCOT': 52, 'PJM': 81, 'NYISO': 105, 'NEISO': 95, 'MISO': 56, 'SPP': 48},
    },
    # Storage: annualized capacity cost ($/MWh-cap), NOT LCOS.
    # Regional variation baked in; TX=0 for all storage.
    'battery': {
        'Low':    {'CAISO': 2.27, 'ERCOT': 2.05, 'PJM': 2.18, 'NYISO': 2.41, 'NEISO': 2.34, 'MISO': 2.14, 'SPP': 2.07},
        'Medium': {'CAISO': 2.60, 'ERCOT': 2.34, 'PJM': 2.49, 'NYISO': 2.75, 'NEISO': 2.67, 'MISO': 2.44, 'SPP': 2.37},
        'High':   {'CAISO': 2.92, 'ERCOT': 2.63, 'PJM': 2.80, 'NYISO': 3.09, 'NEISO': 3.00, 'MISO': 2.75, 'SPP': 2.66},
    },
    'battery8': {
        'Low':    {'CAISO': 1.62, 'ERCOT': 1.46, 'PJM': 1.55, 'NYISO': 1.71, 'NEISO': 1.67, 'MISO': 1.51, 'SPP': 1.49},
        'Medium': {'CAISO': 1.94, 'ERCOT': 1.75, 'PJM': 1.86, 'NYISO': 2.05, 'NEISO': 2.00, 'MISO': 1.81, 'SPP': 1.78},
        'High':   {'CAISO': 2.26, 'ERCOT': 2.04, 'PJM': 2.17, 'NYISO': 2.39, 'NEISO': 2.33, 'MISO': 2.11, 'SPP': 2.08},
    },
    'ldes': {
        'Low':    {'CAISO': 0.38, 'ERCOT': 0.33, 'PJM': 0.36, 'NYISO': 0.42, 'NEISO': 0.40, 'MISO': 0.34, 'SPP': 0.33},
        'Medium': {'CAISO': 0.63, 'ERCOT': 0.54, 'PJM': 0.59, 'NYISO': 0.70, 'NEISO': 0.66, 'MISO': 0.56, 'SPP': 0.55},
        'High':   {'CAISO': 1.00, 'ERCOT': 0.86, 'PJM': 0.94, 'NYISO': 1.11, 'NEISO': 1.06, 'MISO': 0.90, 'SPP': 0.88},
    },
}

TX_TABLES = {
    'wind':       {'None': 0, 'Low': {'CAISO': 4, 'ERCOT': 3, 'PJM': 5, 'NYISO': 7, 'NEISO': 6, 'MISO': 5, 'SPP': 4},
                   'Medium': {'CAISO': 8, 'ERCOT': 6, 'PJM': 10, 'NYISO': 14, 'NEISO': 12, 'MISO': 9, 'SPP': 7},
                   'High': {'CAISO': 14, 'ERCOT': 10, 'PJM': 18, 'NYISO': 22, 'NEISO': 20, 'MISO': 16, 'SPP': 12}},
    'solar':      {'None': 0, 'Low': {'CAISO': 1, 'ERCOT': 1, 'PJM': 2, 'NYISO': 3, 'NEISO': 3, 'MISO': 2, 'SPP': 1},
                   'Medium': {'CAISO': 3, 'ERCOT': 3, 'PJM': 5, 'NYISO': 7, 'NEISO': 6, 'MISO': 4, 'SPP': 3},
                   'High': {'CAISO': 6, 'ERCOT': 5, 'PJM': 9, 'NYISO': 12, 'NEISO': 10, 'MISO': 8, 'SPP': 6}},
    'clean_firm': {'None': 0, 'Low': {'CAISO': 1, 'ERCOT': 1, 'PJM': 1, 'NYISO': 2, 'NEISO': 2, 'MISO': 1, 'SPP': 1},
                   'Medium': {'CAISO': 3, 'ERCOT': 2, 'PJM': 3, 'NYISO': 5, 'NEISO': 4, 'MISO': 3, 'SPP': 2},
                   'High': {'CAISO': 6, 'ERCOT': 4, 'PJM': 6, 'NYISO': 9, 'NEISO': 7, 'MISO': 5, 'SPP': 4}},
    'ccs_ccgt':   {'None': 0, 'Low': {'CAISO': 1, 'ERCOT': 1, 'PJM': 1, 'NYISO': 2, 'NEISO': 2, 'MISO': 1, 'SPP': 1},
                   'Medium': {'CAISO': 2, 'ERCOT': 2, 'PJM': 3, 'NYISO': 4, 'NEISO': 3, 'MISO': 2, 'SPP': 2},
                   'High': {'CAISO': 4, 'ERCOT': 3, 'PJM': 5, 'NYISO': 7, 'NEISO': 6, 'MISO': 4, 'SPP': 3}},
    # Storage TX = 0: regional variation baked into annualized capacity costs
    'battery':    {'None': 0, 'Low': 0, 'Medium': 0, 'High': 0},
    'battery8':   {'None': 0, 'Low': 0, 'Medium': 0, 'High': 0},
    'ldes':       {'None': 0, 'Low': 0, 'Medium': 0, 'High': 0},
    'hydro':      {'None': 0, 'Low': 0, 'Medium': 0, 'High': 0},
}

UPRATE_LCOE = {'L': 15, 'M': 25, 'H': 40}
NUCLEAR_NEWBUILD_LCOE = {
    'L': {'CAISO': 70, 'ERCOT': 68, 'PJM': 72, 'NYISO': 75, 'NEISO': 73, 'MISO': 70, 'SPP': 68},
    'M': {'CAISO': 95, 'ERCOT': 90, 'PJM': 105, 'NYISO': 110, 'NEISO': 108, 'MISO': 100, 'SPP': 92},
    'H': {'CAISO': 140, 'ERCOT': 135, 'PJM': 160, 'NYISO': 170, 'NEISO': 165, 'MISO': 155, 'SPP': 140},
}
GEOTHERMAL_LCOE = {'L': 63, 'M': 88, 'H': 110}
GEO_CAP_TWH = 39.0
CCS_LCOE_45Q_ON = {
    'L': {'CAISO': 58, 'ERCOT': 52, 'PJM': 62, 'NYISO': 78, 'NEISO': 75, 'MISO': 55, 'SPP': 50},
    'M': {'CAISO': 86, 'ERCOT': 71, 'PJM': 79, 'NYISO': 99, 'NEISO': 96, 'MISO': 74, 'SPP': 68},
    'H': {'CAISO': 115, 'ERCOT': 92, 'PJM': 102, 'NYISO': 128, 'NEISO': 122, 'MISO': 96, 'SPP': 88},
}

EXISTING_NUCLEAR_GW = {'CAISO': 2.3, 'ERCOT': 2.7, 'PJM': 32.0, 'NYISO': 3.4, 'NEISO': 3.5, 'MISO': 12.0, 'SPP': 1.2}
UPRATE_CAP_TWH = {iso: round(gw * 0.08 * 0.90 * 8760 / 1e3, 3)
                  for iso, gw in EXISTING_NUCLEAR_GW.items()}

PEAK_DEMAND_MW = {'CAISO': 43860, 'ERCOT': 83597, 'PJM': 160560, 'NYISO': 31857, 'NEISO': 25898, 'MISO': 127125, 'SPP': 54368}
EXISTING_GAS_CAPACITY_MW = {'CAISO': 37000, 'ERCOT': 55000, 'PJM': 75000, 'NYISO': 18000, 'NEISO': 14000, 'MISO': 68000, 'SPP': 32000}
NEW_CCGT_COST_KW_YR = {'CAISO': 112, 'ERCOT': 89, 'PJM': 99, 'NYISO': 114, 'NEISO': 105, 'MISO': 95, 'SPP': 88}
EXISTING_GAS_FOM_KW_YR = {'CAISO': 16, 'ERCOT': 13, 'PJM': 14, 'NYISO': 17, 'NEISO': 15, 'MISO': 14, 'SPP': 13}
PEAK_CAPACITY_CREDITS = {
    'clean_firm': 1.0, 'solar': 0.30, 'wind': 0.10, 'ccs_ccgt': 0.90,
    'hydro': 0.50, 'battery': 0.95, 'battery8': 0.95, 'ldes': 0.90,
}
GAS_AVAILABILITY_FACTOR = {
    'CAISO': 0.88, 'ERCOT': 0.83, 'PJM': 0.82, 'NYISO': 0.82, 'NEISO': 0.85,
    'MISO': 0.84, 'SPP': 0.84,
}
RESOURCE_ADEQUACY_MARGIN = 0.15

LEVEL_NAME = {'L': 'Low', 'M': 'Medium', 'H': 'High'}

# SBTi timeline mapping
# Canonical threshold → target year mapping (SBTi-interpolated)
# Imported from step3 when available; hardcoded fallback for standalone use
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from step3_cost_optimization import THRESHOLD_TARGET_YEARS
    SBTI_YEAR_MAP = THRESHOLD_TARGET_YEARS
except ImportError:
    SBTI_YEAR_MAP = {
        50: 2030, 55: 2031, 60: 2033, 65: 2034, 70: 2035, 75: 2036, 80: 2037,
        85: 2038, 87.5: 2039, 90: 2040, 92.5: 2043,
        95: 2045, 97.5: 2048, 99: 2049, 99.5: 2049, 99.9: 2050, 100: 2050,
    }

# Demand growth rates (decimal) — medium growth scenario for both A and B
DEMAND_GROWTH_RATES = {
    'CAISO': 0.019, 'ERCOT': 0.035, 'PJM': 0.024,
    'NYISO': 0.020, 'NEISO': 0.018,
    'MISO': 0.022, 'SPP': 0.018,
}
BASE_YEAR = 2025


def get_demand_twh(iso, threshold):
    """Get demand TWh for a given ISO and threshold, accounting for medium demand growth.

    Each threshold maps to a year via SBTI_YEAR_MAP. Demand grows at
    the ISO's medium growth rate compounded annually from 2025.
    """
    year = SBTI_YEAR_MAP.get(threshold, 2050)
    years = max(0, year - BASE_YEAR)
    rate = DEMAND_GROWTH_RATES[iso]
    return BASE_DEMAND_TWH[iso] * (1 + rate) ** years


def get_demand_growth_factor(iso, threshold):
    """Get the demand growth factor relative to BASE_YEAR."""
    year = SBTI_YEAR_MAP.get(threshold, 2050)
    years = max(0, year - BASE_YEAR)
    rate = DEMAND_GROWTH_RATES[iso]
    return (1 + rate) ** years

# Zone definitions for consequential queue
ZONES = [
    {'label': '50→55%', 'start': 50, 'end': 55},
    {'label': '55→60%', 'start': 55, 'end': 60},
    {'label': '60→65%', 'start': 60, 'end': 65},
    {'label': '65→70%', 'start': 65, 'end': 70},
    {'label': '70→75%', 'start': 70, 'end': 75},
    {'label': '75→90%', 'start': 75, 'end': 90},
    {'label': '90→95%', 'start': 90, 'end': 95},
    {'label': '95→97.5%', 'start': 95, 'end': 97.5},
    {'label': '97.5→99%', 'start': 97.5, 'end': 99},
    {'label': '99→99.5%', 'start': 99, 'end': 99.5},
    {'label': '99.5→99.9%', 'start': 99.5, 'end': 99.9},
    {'label': '99.9→100%', 'start': 99.9, 'end': 100},
]


# ============================================================================
# SCENARIO DEFINITIONS
# ============================================================================

# Scenario A: "Pure Consequential"
# Sequential path-dependent procurement — chase cheapest $/tCO₂ at each step.
# Uses Low renewables (cheap) but HIGH clean firm/CCS/LDES (FOAK).
# The consequential buyer chases cheap $/tCO₂ with renewables and never invests
# in clean firm development. When firm generation is finally needed at high
# thresholds, there's been no learning curve — costs remain at FOAK.
# Uprates are Medium: existing nuclear uprates don't depend on technology learning.
SCENARIO_A = {
    'name': 'Pure Consequential',
    'short': 'pure_consequential',
    'description': 'Sequential procurement chasing cheapest $/tCO₂ — no clean firm investment means FOAK prices when firm is finally needed',
    'toggles': {
        'ren': 'L',        # Low renewable (cheap solar/wind — what they chase)
        'firm': 'H',       # High firm gen (FOAK — never invested, no learning)
        'batt': 'L',       # Low battery (mature tech, already down learning curve)
        'ldes_lvl': 'H',   # High LDES (FOAK — no learning curve)
        'fuel': 'M',       # Medium fossil fuel
        'tx': 'M',         # Medium transmission
        'ccs': 'H',        # High CCS (FOAK — no learning curve)
        'q45': '1',        # 45Q on
        'geo': 'H',        # High geothermal (CAISO only, FOAK)
    },
    'uprate_level': 'M',   # Uprates at Medium — existing plants, no FOAK/NOAK dependency
}

# Scenario B: "Hourly Matching"
# Forward-looking investment strategy: determine the optimal 2045 endpoint
# (95% clean with all costs at NOAK/Low) and work backward to deploy resources
# toward that target. Pays FOAK for nuclear/LDES upfront but drives Wright's Law
# learning curve — by 2040 firm costs reach NOAK. CCS uses Medium cost with 45Q
# (mature pathway, no FOAK premium). Produces fundamentally different mixes from
# Scenario A because resource allocation is guided by the endpoint target, not
# greedy sequential cost minimization.
SCENARIO_B = {
    'name': 'Hourly Matching',
    'short': 'hourly_matching',
    'description': 'Forward-looking: optimal 2045 endpoint defines deployment trajectory. Nuclear/LDES FOAK(H)→NOAK(L). CCS FOAK(H)→NOAK(M) with 45Q. Early firm investment yields lower long-term system cost.',
    'toggles': {
        'ren': 'L',        # Low renewable (mature tech, already cheap)
        'firm': 'H',       # High firm gen (FOAK at start, learning curve drives to NOAK)
        'batt': 'L',       # Low battery (mature tech, already down learning curve)
        'ldes_lvl': 'H',   # High LDES (FOAK → NOAK via learning)
        'fuel': 'M',       # Medium fossil fuel
        'tx': 'M',         # Medium transmission
        'ccs': 'H',        # High CCS (FOAK start, learning curve H→M with 45Q)
        'q45': '1',        # 45Q on
        'geo': 'H',        # High geothermal (CAISO only, FOAK → NOAK)
    },
    'uprate_level': 'M',   # Uprates at Medium — existing plants, no FOAK/NOAK dependency
}

SCENARIOS = [SCENARIO_A, SCENARIO_B]


# ============================================================================
# LEARNING CURVE FRACTION
# ============================================================================

def learning_fraction(threshold, scenario='B'):
    """Map CFE threshold to FOAK→NOAK learning curve fraction [0, 1].

    0 = pure FOAK (High cost), 1 = full NOAK (Low cost).
    Year-based curves differentiated by scenario:

    Scenario B (Hourly Matching — aggressive deployment):
      - Stays at FOAK until 2030. Learning period: 2030-2040 (10 years).
      - NOAK by 2040, stable through 2050.
      - Sources: INL SOAK data (15% unit 2), NEA (-18 to -55% by unit 8),
        DOE Liftoff (NOAK early 2030s). Compressed sequential learning.

    Scenario A (Pure Consequential — delayed deployment):
      - Stays at FOAK until 2035. Learning period: 2035-2047 (12 years).
      - Same concave shape, stretched across longer timeline.
      - Less investment → fewer units/year → slower learning.

    Both use concave exponent 0.6 — steep front-end (first 4-5 years see
    most of the cost reduction) matching Wright's Law unit-doubling data
    (INL/NEA), then asymptotically approaching NOAK in the latter half.
    """
    year = SBTI_YEAR_MAP.get(threshold, 2050)

    if scenario == 'B':
        foak_start = 2030  # Learning begins with first operational SMRs
        noak_year = 2040   # 10-year learning period (2030-2040)
    else:  # Scenario A
        foak_start = 2035  # 5-year delay: less investment, slower regulatory
        noak_year = 2047   # 12-year learning period (2035-2047)

    if year < foak_start:
        return 0.0  # Pure FOAK, no deployment yet
    if year >= noak_year:
        return 1.0  # Full NOAK achieved

    # Concave ramp: steep front-end (Wright's Law early doublings), asymptotic tail.
    # Exponent 0.6 reflects INL data: 15% at SOAK, 20% by unit 4, then gradual.
    # First 4-5 years see most cost reduction; latter half approaches NOAK asymptotically.
    active = (year - foak_start) / (noak_year - foak_start)  # 0→1
    return active ** 0.6


# ============================================================================
# PARSE FEASIBLE MIXES FROM SHARED-DATA.JS
# ============================================================================

def _load_feasible_from_parquet(iso, step3_dir='data/step3-cost-opt-parquets'):
    """Load feasible mixes for a single ISO from the step3 parquet file.

    Returns dict: {threshold_str: [[cf%, sol%, wnd%, ccs%, hyd%, match%, bat4%, bat8%, ldes%, h2%], ...]}
    """
    feasible_path = os.path.join(step3_dir, f'step3_feasible_{iso}.parquet')
    if not os.path.exists(feasible_path):
        return {}

    df = pd.read_parquet(feasible_path)
    # v5.0: procurement baked into resource percentages
    mix_fields = ['clean_firm', 'solar', 'wind', 'ccs_ccgt', 'hydro',
                  'hourly_match_score',
                  'battery_dispatch_pct', 'battery8_dispatch_pct',
                  'ldes_dispatch_pct', 'h2_dispatch_pct']
    for col in mix_fields:
        if col not in df.columns:
            df[col] = 0

    result = {}
    for t_val, grp in df.groupby('threshold'):
        t_str = str(int(t_val)) if t_val == int(t_val) else str(t_val)
        rows = []
        for _, row in grp.iterrows():
            rows.append([
                row['clean_firm'], row['solar'], row['wind'],
                row['ccs_ccgt'], row['hydro'],
                round(row['hourly_match_score'], 1),
                row['battery_dispatch_pct'], row['battery8_dispatch_pct'],
                row['ldes_dispatch_pct'], row['h2_dispatch_pct'],
            ])
        result[t_str] = rows
    return result


def parse_feasible_mixes(js_path='dashboard/js/shared-data.js'):
    """Parse FEASIBLE_MIXES from the shared-data.js file.

    Falls back to loading directly from step3 parquets for any ISO
    missing from shared-data.js (e.g. MISO, SPP if step7 hasn't been
    re-run since their parquets were generated).
    """
    with open(js_path) as f:
        content = f.read()

    # Find the FEASIBLE_MIXES block
    start = content.find('const FEASIBLE_MIXES = {')
    if start < 0:
        raise ValueError("FEASIBLE_MIXES not found in shared-data.js")

    # Find matching closing brace
    brace_count = 0
    i = content.index('{', start)
    for j in range(i, len(content)):
        if content[j] == '{':
            brace_count += 1
        elif content[j] == '}':
            brace_count -= 1
            if brace_count == 0:
                end = j + 1
                break

    block = content[i:end]

    # Convert JS object to valid JSON
    # Replace unquoted keys (CAISO: → "CAISO":)
    block = re.sub(r'(\b[A-Z]+\b)\s*:', r'"\1":', block)
    # Remove trailing commas before } or ]
    block = re.sub(r',\s*([}\]])', r'\1', block)

    mixes = json.loads(block)

    # Fall back to step3 parquets for any ISO missing or empty in shared-data.js,
    # AND backfill any individual thresholds present in parquets but absent from JS
    expected_thresholds = {str(int(t)) if t == int(t) else str(t) for t in THRESHOLDS}
    for iso in ISOS:
        iso_data = mixes.get(iso, {})
        has_mixes = any(len(v) > 0 for v in iso_data.values()) if iso_data else False
        if not has_mixes:
            # ISO entirely missing from shared-data.js → full parquet load
            parquet_mixes = _load_feasible_from_parquet(iso)
            if parquet_mixes:
                mixes[iso] = parquet_mixes
                total = sum(len(v) for v in parquet_mixes.values())
                print(f"  Loaded {iso} feasible mixes from parquet fallback: {total} mixes")
            else:
                print(f"  WARNING: No feasible mixes for {iso} in shared-data.js or parquets")
        else:
            # ISO present but may be missing thresholds (e.g. 99.5, 99.9)
            present_thresholds = set(iso_data.keys())
            missing = expected_thresholds - present_thresholds
            if missing:
                parquet_mixes = _load_feasible_from_parquet(iso)
                if parquet_mixes:
                    filled = []
                    for t_str in sorted(missing):
                        if t_str in parquet_mixes and parquet_mixes[t_str]:
                            mixes[iso][t_str] = parquet_mixes[t_str]
                            filled.append(t_str)
                    if filled:
                        print(f"  Backfilled {iso} thresholds from parquet: {', '.join(filled)}")

    # v5.0: mixes[iso][threshold_str] = list of [cf%, sol%, wnd%, ccs%, hyd%, match%, bat4%, bat8%, ldes%, h2%]
    return mixes




# ============================================================================
# COST FUNCTION (simplified from step3)
# ============================================================================

def get_tx(rtype, tx_level, iso):
    val = TX_TABLES[rtype][tx_level]
    if isinstance(val, dict):
        return val[iso]
    return val


def _resource_new_build_lcoe(res, sens, iso):
    """Get new-build LCOE+transmission for a resource under given sensitivity toggles.

    Used to price locked-in excess resources from prior path-dependent steps.
    Returns $/MWh delivered.
    """
    ren_name = LEVEL_NAME[sens['ren']]
    batt_name = LEVEL_NAME[sens['batt']]
    ldes_name = LEVEL_NAME[sens['ldes_lvl']]
    tx_name = LEVEL_NAME[sens['tx']]
    firm_lev = sens['firm']
    ccs_lev = sens['ccs']
    q45 = sens['q45']

    if res == 'solar':
        return LCOE_TABLES['solar'][ren_name][iso] + get_tx('solar', tx_name, iso)
    elif res == 'wind':
        return LCOE_TABLES['wind'][ren_name][iso] + get_tx('wind', tx_name, iso)
    elif res == 'clean_firm':
        return NUCLEAR_NEWBUILD_LCOE[firm_lev][iso] + get_tx('clean_firm', tx_name, iso)
    elif res == 'ccs_ccgt':
        ccs_table = CCS_LCOE_45Q_ON if q45 == '1' else None
        return ccs_table[ccs_lev][iso] + get_tx('ccs_ccgt', tx_name, iso)
    elif res == 'hydro':
        return 0  # Existing-only, wholesale-priced
    elif res == 'battery':
        return LCOE_TABLES['battery'][batt_name][iso] + get_tx('battery', tx_name, iso)
    elif res == 'ldes':
        return LCOE_TABLES['ldes'][ldes_name][iso] + get_tx('ldes', tx_name, iso)
    return 0


def compute_mix_cost(mix, sens, iso, demand_twh, overrides=None, growth_factor=1.0):
    """
    Compute total system cost per MWh for a single mix under a sensitivity scenario.

    mix: [cf%, sol%, wnd%, ccs%, hyd%, match%, bat4%, bat8%, ldes%, h2%]  (v5.0 format)
    overrides: optional dict with explicit LCOE values (bypasses toggle lookups):
        nuclear_lcoe, ccs_lcoe, geo_lcoe, ldes_lcoe, uprate_lcoe
    growth_factor: demand growth multiplier (>1 means demand has grown from base year).
        Existing resource percentages are scaled DOWN by this factor so that existing
        absolute TWh stays fixed as demand grows.
    Returns: dict with cost details + gas backup
    """
    # v5.0: procurement baked into resource percentages
    cf_pct, sol_pct, wnd_pct, ccs_pct, hyd_pct = mix[0], mix[1], mix[2], mix[3], mix[4]
    match_score = mix[5]
    bat4_pct, bat8_pct, ldes_pct = mix[6], mix[7], mix[8]
    h2_pct = mix[9] if len(mix) > 9 else 0

    match_frac = match_score / 100.0

    ren_name = LEVEL_NAME[sens['ren']]
    batt_name = LEVEL_NAME[sens['batt']]
    ldes_name = LEVEL_NAME[sens['ldes_lvl']]
    fuel_name = LEVEL_NAME[sens['fuel']]
    tx_name = LEVEL_NAME[sens['tx']]
    firm_lev = sens['firm']
    ccs_lev = sens['ccs']
    q45 = sens['q45']
    geo_lev = sens.get('geo')

    # Existing resource percentages scaled for demand growth.
    # Existing infrastructure produces fixed absolute TWh (2025 levels).
    # As demand grows, existing covers a smaller percentage.
    existing = {k: v / growth_factor for k, v in GRID_MIX_SHARES[iso].items()}
    wholesale = max(5, WHOLESALE_PRICES[iso] + FUEL_ADJUSTMENTS[iso][fuel_name])

    # CCS price
    if overrides and 'ccs_lcoe' in overrides:
        ccs_lcoe = overrides['ccs_lcoe']
    else:
        ccs_table = CCS_LCOE_45Q_ON if q45 == '1' else None
        ccs_lcoe = ccs_table[ccs_lev][iso]
    ccs_tx = get_tx('ccs_ccgt', tx_name, iso)
    ccs_price = ccs_lcoe + ccs_tx

    # Nuclear price
    if overrides and 'nuclear_lcoe' in overrides:
        nuclear_price = overrides['nuclear_lcoe'] + get_tx('clean_firm', tx_name, iso)
    else:
        nuclear_price = NUCLEAR_NEWBUILD_LCOE[firm_lev][iso] + get_tx('clean_firm', tx_name, iso)
    remaining_price = min(nuclear_price, ccs_price)

    # Geothermal price
    geo_price = 0.0
    if iso == 'CAISO' and geo_lev:
        if overrides and 'geo_lcoe' in overrides:
            geo_price = overrides['geo_lcoe'] + get_tx('clean_firm', tx_name, iso)
        else:
            geo_price = GEOTHERMAL_LCOE[geo_lev] + get_tx('clean_firm', tx_name, iso)

    # Demand-weighted percentages  # v5.0: procurement baked into resource percentages
    sol_demand = sol_pct
    wnd_demand = wnd_pct
    ccs_demand = ccs_pct
    cf_demand = cf_pct
    # Hydro is existing-only: cap at 2025 absolute TWh.
    # Convert mix percentage to TWh, cap, then back to demand-weighted pct points.
    hydro_twh_raw = hyd_pct / 100.0 * demand_twh
    hydro_twh_capped = min(hydro_twh_raw, HYDRO_CAP_TWH[iso])
    hyd_demand = hydro_twh_capped / demand_twh * 100.0  # capped pct-points of demand

    # Existing/new splits
    sol_existing = min(sol_demand, existing['solar'])
    sol_new = max(0, sol_demand - existing['solar'])
    wnd_existing = min(wnd_demand, existing['wind'])
    wnd_new = max(0, wnd_demand - existing['wind'])
    ccs_existing = min(ccs_demand, existing.get('ccs_ccgt', 0))
    ccs_new = max(0, ccs_demand - existing.get('ccs_ccgt', 0))
    cf_existing = min(cf_demand, existing['clean_firm'])
    cf_new = max(0, cf_demand - existing['clean_firm'])

    # Clean firm tranche allocation
    new_cf_twh = cf_new / 100.0 * demand_twh
    uprate_twh = min(new_cf_twh, UPRATE_CAP_TWH[iso])
    remaining_after_uprate = max(0, new_cf_twh - uprate_twh)

    geo_twh = 0.0
    remaining_after_geo = remaining_after_uprate
    if iso == 'CAISO':
        geo_twh = min(remaining_after_uprate, GEO_CAP_TWH)
        remaining_after_geo = max(0, remaining_after_uprate - geo_twh)

    # Gas backup (scenario-invariant given the mix)
    demand_mwh = demand_twh * 1e6
    avg_demand_mw = demand_mwh / 8760
    # Peak demand grows proportionally with annual demand
    peak_mw = PEAK_DEMAND_MW[iso] * growth_factor
    ra_peak_mw = peak_mw * (1 + RESOURCE_ADEQUACY_MARGIN)

    # Hydro peak capacity from capped TWh (not inflated demand)
    hydro_avg_mw = hydro_twh_capped * 1e6 / 8760
    clean_peak_mw = (
        cf_pct / 100 * avg_demand_mw * PEAK_CAPACITY_CREDITS['clean_firm'] +
        sol_pct / 100 * avg_demand_mw * PEAK_CAPACITY_CREDITS['solar'] +
        wnd_pct / 100 * avg_demand_mw * PEAK_CAPACITY_CREDITS['wind'] +
        ccs_pct / 100 * avg_demand_mw * PEAK_CAPACITY_CREDITS['ccs_ccgt'] +
        hydro_avg_mw * PEAK_CAPACITY_CREDITS['hydro'] +
        bat4_pct / 100 * avg_demand_mw * PEAK_CAPACITY_CREDITS['battery'] +
        bat8_pct / 100 * avg_demand_mw * PEAK_CAPACITY_CREDITS['battery8'] +
        ldes_pct / 100 * avg_demand_mw * PEAK_CAPACITY_CREDITS['ldes']
    )

    gaf = GAS_AVAILABILITY_FACTOR[iso]
    gas_needed_mw = max(0, ra_peak_mw - clean_peak_mw) / gaf
    existing_gas_mw = EXISTING_GAS_CAPACITY_MW[iso]
    existing_gas_used_mw = min(gas_needed_mw, existing_gas_mw)
    new_gas_mw = max(0, gas_needed_mw - existing_gas_used_mw)

    gas_cost = (
        existing_gas_used_mw * EXISTING_GAS_FOM_KW_YR[iso] * 1000 +
        new_gas_mw * NEW_CCGT_COST_KW_YR[iso] * 1000
    ) / demand_mwh

    # Uprate and LDES prices (with optional override support)
    uprate_price = overrides.get('uprate_lcoe', UPRATE_LCOE[firm_lev]) if overrides else UPRATE_LCOE[firm_lev]
    if overrides and 'ldes_lcoe' in overrides:
        ldes_price = overrides['ldes_lcoe'] + get_tx('ldes', tx_name, iso)
    else:
        ldes_price = LCOE_TABLES['ldes'][ldes_name][iso] + get_tx('ldes', tx_name, iso)

    # Total cost = Σ(coefficient × price) + gas_cost
    # Existing clean resources priced at $0; only new-build resources have LCOE costs.
    total_cost = (
        sol_new / 100.0 * (LCOE_TABLES['solar'][ren_name][iso] + get_tx('solar', tx_name, iso)) +
        wnd_new / 100.0 * (LCOE_TABLES['wind'][ren_name][iso] + get_tx('wind', tx_name, iso)) +
        ccs_new / 100.0 * ccs_price +
        uprate_twh / demand_twh * uprate_price +
        geo_twh / demand_twh * geo_price +
        remaining_after_geo / demand_twh * remaining_price +
        bat4_pct / 100.0 * (LCOE_TABLES['battery'][batt_name][iso] + get_tx('battery', tx_name, iso)) +
        bat8_pct / 100.0 * (LCOE_TABLES['battery8'][batt_name][iso] + get_tx('battery8', tx_name, iso)) +
        ldes_pct / 100.0 * ldes_price +
        gas_cost
    )

    # New-build cost tracking (for MAC = delta_cost / delta_co2)
    new_build_per_mwh = total_cost - gas_cost
    new_gen_twh = (sol_new + wnd_new + ccs_new + cf_new) / 100.0 * demand_twh
    new_build_cost_total = new_build_per_mwh * demand_mwh
    if new_gen_twh > 0.01:
        blended_new_lcoe = new_build_per_mwh * demand_twh / new_gen_twh
    else:
        blended_new_lcoe = 0.0

    eff_cost = total_cost / match_frac if match_frac > 0 else 0
    incremental = eff_cost - wholesale

    # Resource TWh (hydro capped at 2025 absolute)
    resource_twh = {}
    for res, pct in zip(RESOURCES, [cf_pct, sol_pct, wnd_pct, ccs_pct, hyd_pct]):
        if res == 'hydro':
            resource_twh[res] = hydro_twh_capped
        else:
            resource_twh[res] = pct / 100.0 * demand_twh

    return {
        'total_cost': round(total_cost, 2),
        'effective_cost': round(eff_cost, 2),
        'incremental': round(incremental, 2),
        'wholesale': wholesale,
        'resource_twh': resource_twh,
        'resource_pct': {'clean_firm': cf_pct, 'solar': sol_pct, 'wind': wnd_pct,
                         'ccs_ccgt': ccs_pct, 'hydro': hyd_pct},
        'match_score': match_score,
        'battery_twh': bat4_pct / 100.0 * demand_twh,
        'battery8_twh': bat8_pct / 100.0 * demand_twh,
        'ldes_twh': ldes_pct / 100.0 * demand_twh,
        'h2_twh': h2_pct / 100.0 * demand_twh,
        'gas_backup_mw': round(gas_needed_mw),
        'new_gas_mw': round(new_gas_mw),
        'existing_gas_used_mw': round(existing_gas_used_mw),
        'clean_peak_mw': round(clean_peak_mw),
        'new_build_cost_total': new_build_cost_total,
        'new_gen_twh': round(new_gen_twh, 3),
        'blended_new_lcoe': round(blended_new_lcoe, 2),
    }


# ============================================================================
# VECTORIZED BATCH COST — eliminates per-mix Python loops
# ============================================================================

def _precompute_cost_params(sens, iso, demand_twh, overrides=None, growth_factor=1.0):
    """Pre-resolve all dict lookups into a flat numeric tuple for batch cost.

    Returns a tuple of scalars that the vectorized kernel consumes.
    Called once per (ISO, threshold, scenario) — O(1) not O(N_mixes).
    """
    ren_name = LEVEL_NAME[sens['ren']]
    batt_name = LEVEL_NAME[sens['batt']]
    ldes_name = LEVEL_NAME[sens['ldes_lvl']]
    fuel_name = LEVEL_NAME[sens['fuel']]
    tx_name = LEVEL_NAME[sens['tx']]
    firm_lev = sens['firm']
    ccs_lev = sens['ccs']
    q45 = sens['q45']
    geo_lev = sens.get('geo')

    existing = {k: v / growth_factor for k, v in GRID_MIX_SHARES[iso].items()}
    wholesale = max(5, WHOLESALE_PRICES[iso] + FUEL_ADJUSTMENTS[iso][fuel_name])

    # CCS price
    if overrides and 'ccs_lcoe' in overrides:
        ccs_lcoe = overrides['ccs_lcoe']
    else:
        ccs_table = CCS_LCOE_45Q_ON if q45 == '1' else None
        ccs_lcoe = ccs_table[ccs_lev][iso]
    ccs_tx = get_tx('ccs_ccgt', tx_name, iso)
    ccs_price = ccs_lcoe + ccs_tx

    # Nuclear price
    if overrides and 'nuclear_lcoe' in overrides:
        nuclear_price = overrides['nuclear_lcoe'] + get_tx('clean_firm', tx_name, iso)
    else:
        nuclear_price = NUCLEAR_NEWBUILD_LCOE[firm_lev][iso] + get_tx('clean_firm', tx_name, iso)
    remaining_price = min(nuclear_price, ccs_price)

    # Geothermal price
    geo_price = 0.0
    if iso == 'CAISO' and geo_lev:
        if overrides and 'geo_lcoe' in overrides:
            geo_price = overrides['geo_lcoe'] + get_tx('clean_firm', tx_name, iso)
        else:
            geo_price = GEOTHERMAL_LCOE[geo_lev] + get_tx('clean_firm', tx_name, iso)

    # Uprate / LDES prices
    uprate_price = overrides.get('uprate_lcoe', UPRATE_LCOE[firm_lev]) if overrides else UPRATE_LCOE[firm_lev]
    if overrides and 'ldes_lcoe' in overrides:
        ldes_price = overrides['ldes_lcoe'] + get_tx('ldes', tx_name, iso)
    else:
        ldes_price = LCOE_TABLES['ldes'][ldes_name][iso] + get_tx('ldes', tx_name, iso)

    sol_lcoe = LCOE_TABLES['solar'][ren_name][iso] + get_tx('solar', tx_name, iso)
    wnd_lcoe = LCOE_TABLES['wind'][ren_name][iso] + get_tx('wind', tx_name, iso)
    bat4_lcoe = LCOE_TABLES['battery'][batt_name][iso] + get_tx('battery', tx_name, iso)
    bat8_lcoe = LCOE_TABLES['battery8'][batt_name][iso] + get_tx('battery8', tx_name, iso)

    demand_mwh = demand_twh * 1e6
    avg_demand_mw = demand_mwh / 8760
    peak_mw = PEAK_DEMAND_MW[iso] * growth_factor
    ra_peak_mw = peak_mw * (1 + RESOURCE_ADEQUACY_MARGIN)
    gaf = GAS_AVAILABILITY_FACTOR[iso]
    existing_gas_mw = float(EXISTING_GAS_CAPACITY_MW[iso])

    ex_sol = existing.get('solar', 0.0)
    ex_wnd = existing.get('wind', 0.0)
    ex_ccs = existing.get('ccs_ccgt', 0.0)
    ex_cf = existing.get('clean_firm', 0.0)

    hydro_cap_twh = HYDRO_CAP_TWH[iso]
    geo_cap_twh = GEO_CAP_TWH if iso == 'CAISO' else 0.0
    uprate_cap_twh = UPRATE_CAP_TWH[iso]

    # Peak capacity credits as array [cf, sol, wnd, ccs, hyd, bat4, bat8, ldes]
    pcc_cf = PEAK_CAPACITY_CREDITS['clean_firm']
    pcc_sol = PEAK_CAPACITY_CREDITS['solar']
    pcc_wnd = PEAK_CAPACITY_CREDITS['wind']
    pcc_ccs = PEAK_CAPACITY_CREDITS['ccs_ccgt']
    pcc_hyd = PEAK_CAPACITY_CREDITS['hydro']
    pcc_bat = PEAK_CAPACITY_CREDITS['battery']
    pcc_bat8 = PEAK_CAPACITY_CREDITS.get('battery8', 0.95)
    pcc_ldes = PEAK_CAPACITY_CREDITS['ldes']

    ex_gas_fom = EXISTING_GAS_FOM_KW_YR[iso]
    new_ccgt_cost = NEW_CCGT_COST_KW_YR[iso]

    # Return flat tuple — no dicts, no strings, pure numerics
    return (
        demand_twh, demand_mwh, avg_demand_mw, wholesale,
        ex_sol, ex_wnd, ex_ccs, ex_cf,
        sol_lcoe, wnd_lcoe, ccs_price, nuclear_price, remaining_price,
        uprate_price, geo_price, ldes_price, bat4_lcoe, bat8_lcoe,
        hydro_cap_twh, geo_cap_twh, uprate_cap_twh,
        ra_peak_mw, gaf, existing_gas_mw,
        pcc_cf, pcc_sol, pcc_wnd, pcc_ccs, pcc_hyd, pcc_bat, pcc_bat8, pcc_ldes,
        ex_gas_fom, new_ccgt_cost,
        float(iso == 'CAISO'),  # is_caiso flag
    )

# Param index constants for readability
_P_DEMAND_TWH, _P_DEMAND_MWH, _P_AVG_MW, _P_WHOLESALE = 0, 1, 2, 3
_P_EX_SOL, _P_EX_WND, _P_EX_CCS, _P_EX_CF = 4, 5, 6, 7
_P_SOL_LCOE, _P_WND_LCOE, _P_CCS_PRICE, _P_NUC_PRICE, _P_REM_PRICE = 8, 9, 10, 11, 12
_P_UPR_PRICE, _P_GEO_PRICE, _P_LDES_PRICE, _P_BAT4_LCOE, _P_BAT8_LCOE = 13, 14, 15, 16, 17
_P_HYDRO_CAP, _P_GEO_CAP, _P_UPR_CAP = 18, 19, 20
_P_RA_PEAK, _P_GAF, _P_EX_GAS_MW = 21, 22, 23
_P_PCC_CF, _P_PCC_SOL, _P_PCC_WND, _P_PCC_CCS, _P_PCC_HYD = 24, 25, 26, 27, 28
_P_PCC_BAT, _P_PCC_BAT8, _P_PCC_LDES = 29, 30, 31
_P_EX_GAS_FOM, _P_NEW_CCGT = 32, 33
_P_IS_CAISO = 34


try:
    from numba import njit as _njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def _njit(*args, **kwargs):
        def decorator(f):
            return f
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


@_njit(cache=True)
def _batch_cost_kernel(mixes, params):
    """Numba-JIT vectorized cost kernel over N mixes.

    mixes: (N, 10) float64 array — [cf%, sol%, wnd%, ccs%, hyd%, match%, bat4%, bat8%, ldes%, h2%]
    params: (35,) float64 array — pre-resolved scalars from _precompute_cost_params

    Returns: (N, 8) array —
        [total_cost, effective_cost, incremental, gas_needed_mw, new_gas_mw,
         existing_gas_used_mw, clean_peak_mw, new_build_cost_total]
    """
    N = mixes.shape[0]
    out = np.empty((N, 8), dtype=np.float64)

    demand_twh = params[_P_DEMAND_TWH]
    demand_mwh = params[_P_DEMAND_MWH]
    avg_mw = params[_P_AVG_MW]
    wholesale = params[_P_WHOLESALE]
    ex_sol = params[_P_EX_SOL]
    ex_wnd = params[_P_EX_WND]
    ex_ccs = params[_P_EX_CCS]
    ex_cf = params[_P_EX_CF]
    sol_lcoe = params[_P_SOL_LCOE]
    wnd_lcoe = params[_P_WND_LCOE]
    ccs_price = params[_P_CCS_PRICE]
    nuc_price = params[_P_NUC_PRICE]
    rem_price = params[_P_REM_PRICE]
    upr_price = params[_P_UPR_PRICE]
    geo_price = params[_P_GEO_PRICE]
    ldes_price = params[_P_LDES_PRICE]
    bat4_lcoe = params[_P_BAT4_LCOE]
    bat8_lcoe = params[_P_BAT8_LCOE]
    hydro_cap = params[_P_HYDRO_CAP]
    geo_cap = params[_P_GEO_CAP]
    upr_cap = params[_P_UPR_CAP]
    ra_peak = params[_P_RA_PEAK]
    gaf = params[_P_GAF]
    ex_gas_mw = params[_P_EX_GAS_MW]
    pcc_cf = params[_P_PCC_CF]
    pcc_sol = params[_P_PCC_SOL]
    pcc_wnd = params[_P_PCC_WND]
    pcc_ccs = params[_P_PCC_CCS]
    pcc_hyd = params[_P_PCC_HYD]
    pcc_bat = params[_P_PCC_BAT]
    pcc_bat8 = params[_P_PCC_BAT8]
    pcc_ldes = params[_P_PCC_LDES]
    ex_gas_fom = params[_P_EX_GAS_FOM]
    new_ccgt = params[_P_NEW_CCGT]
    is_caiso = params[_P_IS_CAISO]

    for i in range(N):
        cf_pct = mixes[i, 0]
        sol_pct = mixes[i, 1]
        wnd_pct = mixes[i, 2]
        ccs_pct = mixes[i, 3]
        hyd_pct = mixes[i, 4]
        match_score = mixes[i, 5]
        bat4_pct = mixes[i, 6]
        bat8_pct = mixes[i, 7]
        ldes_pct = mixes[i, 8]
        h2_pct = mixes[i, 9]

        match_frac = match_score / 100.0

        # Hydro cap
        hydro_twh_raw = hyd_pct / 100.0 * demand_twh
        hydro_twh = min(hydro_twh_raw, hydro_cap)
        hyd_demand = hydro_twh / demand_twh * 100.0

        # Existing/new splits
        sol_new = max(0.0, sol_pct - ex_sol)
        wnd_new = max(0.0, wnd_pct - ex_wnd)
        ccs_new = max(0.0, ccs_pct - ex_ccs)
        cf_new = max(0.0, cf_pct - ex_cf)

        # Clean firm tranche allocation
        new_cf_twh = cf_new / 100.0 * demand_twh
        uprate_twh = min(new_cf_twh, upr_cap)
        remaining_after_uprate = max(0.0, new_cf_twh - uprate_twh)

        geo_twh = 0.0
        remaining_after_geo = remaining_after_uprate
        if is_caiso > 0.5:
            geo_twh = min(remaining_after_uprate, geo_cap)
            remaining_after_geo = max(0.0, remaining_after_uprate - geo_twh)

        # Gas backup
        hydro_avg_mw = hydro_twh * 1e6 / 8760.0
        clean_peak_mw = (
            cf_pct / 100.0 * avg_mw * pcc_cf +
            sol_pct / 100.0 * avg_mw * pcc_sol +
            wnd_pct / 100.0 * avg_mw * pcc_wnd +
            ccs_pct / 100.0 * avg_mw * pcc_ccs +
            hydro_avg_mw * pcc_hyd +
            bat4_pct / 100.0 * avg_mw * pcc_bat +
            bat8_pct / 100.0 * avg_mw * pcc_bat8 +
            ldes_pct / 100.0 * avg_mw * pcc_ldes
        )

        gas_needed_mw = max(0.0, ra_peak - clean_peak_mw) / gaf
        existing_gas_used_mw = min(gas_needed_mw, ex_gas_mw)
        new_gas_mw = max(0.0, gas_needed_mw - existing_gas_used_mw)

        gas_cost = (
            existing_gas_used_mw * ex_gas_fom * 1000.0 +
            new_gas_mw * new_ccgt * 1000.0
        ) / demand_mwh

        # Total cost
        total_cost = (
            sol_new / 100.0 * sol_lcoe +
            wnd_new / 100.0 * wnd_lcoe +
            ccs_new / 100.0 * ccs_price +
            uprate_twh / demand_twh * upr_price +
            geo_twh / demand_twh * geo_price +
            remaining_after_geo / demand_twh * rem_price +
            bat4_pct / 100.0 * bat4_lcoe +
            bat8_pct / 100.0 * bat8_lcoe +
            ldes_pct / 100.0 * ldes_price +
            gas_cost
        )

        # Effective cost and incremental
        eff_cost = total_cost / match_frac if match_frac > 0.0 else 0.0
        incremental = eff_cost - wholesale

        # New-build cost tracking
        new_build_per_mwh = total_cost - gas_cost
        new_build_cost_total = new_build_per_mwh * demand_mwh

        out[i, 0] = total_cost
        out[i, 1] = eff_cost
        out[i, 2] = incremental
        out[i, 3] = gas_needed_mw
        out[i, 4] = new_gas_mw
        out[i, 5] = existing_gas_used_mw
        out[i, 6] = clean_peak_mw
        out[i, 7] = new_build_cost_total

    return out


def _mixes_to_array(mixes):
    """Convert list-of-lists to (N, 10) float64 numpy array, padding to 10 cols."""
    N = len(mixes)
    arr = np.zeros((N, 10), dtype=np.float64)
    for i, m in enumerate(mixes):
        ncols = min(len(m), 10)
        for j in range(ncols):
            arr[i, j] = m[j]
    return arr


def compute_mix_cost_batch(mixes_arr, sens, iso, demand_twh, overrides=None, growth_factor=1.0):
    """Vectorized cost computation for N mixes at once.

    mixes_arr: (N, 10) numpy array
    Returns: dict with arrays of length N for each output field,
             plus 'best_idx' (index of cheapest effective_cost).
    """
    params_tuple = _precompute_cost_params(sens, iso, demand_twh, overrides, growth_factor)
    params_arr = np.array(params_tuple, dtype=np.float64)

    results = _batch_cost_kernel(mixes_arr, params_arr)

    demand_mwh = demand_twh * 1e6
    match_scores = mixes_arr[:, 5]
    match_fracs = match_scores / 100.0

    # New-gen TWh (vectorized)
    existing = {k: v / growth_factor for k, v in GRID_MIX_SHARES[iso].items()}
    cf_new = np.maximum(0, mixes_arr[:, 0] - existing.get('clean_firm', 0))
    sol_new = np.maximum(0, mixes_arr[:, 1] - existing.get('solar', 0))
    wnd_new = np.maximum(0, mixes_arr[:, 2] - existing.get('wind', 0))
    ccs_new = np.maximum(0, mixes_arr[:, 3] - existing.get('ccs_ccgt', 0))
    new_gen_twh = (cf_new + sol_new + wnd_new + ccs_new) / 100.0 * demand_twh

    new_build_per_mwh = results[:, 0] - (
        results[:, 5] * params_arr[_P_EX_GAS_FOM] * 1000.0 +
        results[:, 4] * params_arr[_P_NEW_CCGT] * 1000.0
    ) / demand_mwh
    blended_new_lcoe = np.where(
        new_gen_twh > 0.01,
        new_build_per_mwh * demand_twh / new_gen_twh,
        0.0
    )

    # Resource TWh (vectorized)
    hydro_cap_twh = HYDRO_CAP_TWH[iso]
    hydro_twh = np.minimum(mixes_arr[:, 4] / 100.0 * demand_twh, hydro_cap_twh)

    best_idx = int(np.argmin(results[:, 1]))

    return {
        'total_cost': results[:, 0],
        'effective_cost': results[:, 1],
        'incremental': results[:, 2],
        'gas_needed_mw': results[:, 3],
        'new_gas_mw': results[:, 4],
        'existing_gas_used_mw': results[:, 5],
        'clean_peak_mw': results[:, 6],
        'new_build_cost_total': results[:, 7],
        'wholesale': params_arr[_P_WHOLESALE],
        'match_scores': match_scores,
        'new_gen_twh': new_gen_twh,
        'blended_new_lcoe': blended_new_lcoe,
        'hydro_twh': hydro_twh,
        'best_idx': best_idx,
    }


def _batch_resource_twh(mixes_arr, demand_twh, iso):
    """Vectorized resource TWh for N mixes. Returns (N, 7) array.

    Columns: [cf, sol, wnd, ccs, hyd, battery, ldes]
    """
    N = mixes_arr.shape[0]
    out = np.empty((N, 7), dtype=np.float64)
    out[:, 0] = mixes_arr[:, 0] / 100.0 * demand_twh  # cf
    out[:, 1] = mixes_arr[:, 1] / 100.0 * demand_twh  # sol
    out[:, 2] = mixes_arr[:, 2] / 100.0 * demand_twh  # wnd
    out[:, 3] = mixes_arr[:, 3] / 100.0 * demand_twh  # ccs
    out[:, 4] = np.minimum(mixes_arr[:, 4] / 100.0 * demand_twh, HYDRO_CAP_TWH[iso])  # hyd
    out[:, 5] = (mixes_arr[:, 6] + mixes_arr[:, 7]) / 100.0 * demand_twh  # battery
    out[:, 6] = mixes_arr[:, 8] / 100.0 * demand_twh  # ldes
    return out


# Resource indices in the (N,7) resource_twh array
_R_CF, _R_SOL, _R_WND, _R_CCS, _R_HYD, _R_BAT, _R_LDES = 0, 1, 2, 3, 4, 5, 6
_FLOOR_KEYS = ['clean_firm', 'solar', 'wind', 'ccs_ccgt', 'hydro', 'battery', 'ldes']


def _floor_to_array(floor_twh):
    """Convert floor dict to (7,) array matching _batch_resource_twh columns."""
    return np.array([
        floor_twh.get('clean_firm', 0),
        floor_twh.get('solar', 0),
        floor_twh.get('wind', 0),
        floor_twh.get('ccs_ccgt', 0),
        floor_twh.get('hydro', 0),
        floor_twh.get('battery', 0),
        floor_twh.get('ldes', 0),
    ], dtype=np.float64)


def _batch_filter_by_floor(mixes_arr, floor_twh, demand_twh, iso):
    """Vectorized floor filter. Returns boolean mask of shape (N,)."""
    deployed = _batch_resource_twh(mixes_arr, demand_twh, iso)
    floor_arr = _floor_to_array(floor_twh)
    # Each resource must be >= floor - 0.01 tolerance
    mask = np.all(deployed >= floor_arr[np.newaxis, :] - 0.01, axis=1)
    return mask


def _batch_floor_excess_cost(deployed_twh, floor_arr, existing_twh_arr,
                              demand_twh, excess_lcoe_arr):
    """Vectorized floor excess cost. Returns (N,) excess $/MWh array.

    deployed_twh: (N, 7) — from _batch_resource_twh
    floor_arr: (7,) — current floor TWh
    existing_twh_arr: (7,) — 2025 existing TWh
    excess_lcoe_arr: (7,) — LCOE per resource for pricing excess
    """
    # Shortfall = max(0, floor - deployed)
    shortfall = np.maximum(0, floor_arr[np.newaxis, :] - deployed_twh)
    # Existing gap portion (priced at $0)
    existing_gap = np.maximum(0, np.minimum(shortfall,
                                             existing_twh_arr[np.newaxis, :] - deployed_twh))
    newbuild_gap = shortfall - np.maximum(0, existing_gap)
    # Zero out tiny gaps
    newbuild_gap = np.where(newbuild_gap > 0.01, newbuild_gap, 0.0)
    # Excess cost = sum over resources of (gap_twh / demand_twh * lcoe)
    excess_per_mwh = np.sum(newbuild_gap / demand_twh * excess_lcoe_arr[np.newaxis, :], axis=1)
    return excess_per_mwh


def _build_excess_lcoe_array(sens, iso, overrides=None):
    """Build (7,) LCOE array for pricing floor excess, matching _FLOOR_KEYS order."""
    return np.array([
        _get_excess_lcoe('clean_firm', sens, iso, overrides),
        _get_excess_lcoe('solar', sens, iso, overrides),
        _get_excess_lcoe('wind', sens, iso, overrides),
        _get_excess_lcoe('ccs_ccgt', sens, iso, overrides),
        _get_excess_lcoe('hydro', sens, iso, overrides),
        _get_excess_lcoe('battery', sens, iso, overrides),
        _get_excess_lcoe('ldes', sens, iso, overrides),
    ], dtype=np.float64)


def _batch_pick_best(mixes_arr, batch_results, deployed_twh, floor_arr,
                      existing_twh_arr, excess_lcoe_arr, demand_twh,
                      use_augmented_cost=True):
    """Find best mix by total cost (optionally augmented with floor excess).

    Returns (best_idx, best_excess_per_mwh) or (None, 0) if no mixes.
    """
    N = mixes_arr.shape[0]
    if N == 0:
        return None, 0.0

    if use_augmented_cost:
        excess_per_mwh = _batch_floor_excess_cost(
            deployed_twh, floor_arr, existing_twh_arr, demand_twh, excess_lcoe_arr)
        total_with_excess = batch_results['total_cost'] + excess_per_mwh
    else:
        total_with_excess = batch_results['total_cost']
        excess_per_mwh = np.zeros(N)

    best_idx = int(np.argmin(total_with_excess))
    return best_idx, float(excess_per_mwh[best_idx])


def _extract_single_result(mixes_arr, batch_results, idx, iso, demand_twh):
    """Extract a single result dict from batch arrays at given index."""
    mix = mixes_arr[idx]
    hydro_cap_twh = HYDRO_CAP_TWH[iso]

    cf_pct, sol_pct, wnd_pct, ccs_pct, hyd_pct = mix[0], mix[1], mix[2], mix[3], mix[4]
    match_score = mix[5]
    bat4_pct, bat8_pct, ldes_pct = mix[6], mix[7], mix[8]
    h2_pct = mix[9] if len(mix) > 9 else 0

    hydro_twh = min(hyd_pct / 100.0 * demand_twh, hydro_cap_twh)
    resource_twh = {
        'clean_firm': cf_pct / 100.0 * demand_twh,
        'solar': sol_pct / 100.0 * demand_twh,
        'wind': wnd_pct / 100.0 * demand_twh,
        'ccs_ccgt': ccs_pct / 100.0 * demand_twh,
        'hydro': hydro_twh,
    }

    return {
        'total_cost': round(float(batch_results['total_cost'][idx]), 2),
        'effective_cost': round(float(batch_results['effective_cost'][idx]), 2),
        'incremental': round(float(batch_results['incremental'][idx]), 2),
        'wholesale': float(batch_results['wholesale']),
        'resource_twh': resource_twh,
        'resource_pct': {'clean_firm': cf_pct, 'solar': sol_pct, 'wind': wnd_pct,
                         'ccs_ccgt': ccs_pct, 'hydro': hyd_pct},
        'match_score': match_score,
        'battery_twh': bat4_pct / 100.0 * demand_twh,
        'battery8_twh': bat8_pct / 100.0 * demand_twh,
        'ldes_twh': ldes_pct / 100.0 * demand_twh,
        'h2_twh': h2_pct / 100.0 * demand_twh,
        'gas_backup_mw': round(float(batch_results['gas_needed_mw'][idx])),
        'new_gas_mw': round(float(batch_results['new_gas_mw'][idx])),
        'existing_gas_used_mw': round(float(batch_results['existing_gas_used_mw'][idx])),
        'clean_peak_mw': round(float(batch_results['clean_peak_mw'][idx])),
        'new_build_cost_total': float(batch_results['new_build_cost_total'][idx]),
        'new_gen_twh': round(float(batch_results['new_gen_twh'][idx]), 3),
        'blended_new_lcoe': round(float(batch_results['blended_new_lcoe'][idx]), 2),
    }


# ============================================================================
# FIND OPTIMAL MIX PER (ISO, THRESHOLD, SCENARIO)
# ============================================================================

def find_optimal_mixes(feasible_mixes, scenario, demand_twh_map):
    """For each ISO × threshold, find the cheapest mix under this scenario's costs."""
    results = {}
    sens = scenario['toggles']

    for iso in ISOS:
        iso_results = {}
        iso_sens = dict(sens)
        if iso != 'CAISO':
            iso_sens['geo'] = None

        for t in THRESHOLDS:
            t_str = str(int(t)) if t == int(t) else str(t)
            mixes = feasible_mixes.get(iso, {}).get(t_str, [])
            if not mixes:
                continue

            mixes_arr = _mixes_to_array(mixes)
            batch = compute_mix_cost_batch(mixes_arr, iso_sens, iso, demand_twh_map[iso])
            best_idx = batch['best_idx']

            best_result = _extract_single_result(mixes_arr, batch, best_idx, iso, demand_twh_map[iso])
            best_mix = mixes[best_idx]

            if best_result and best_mix:
                # Preserve dispatch parameters for dispatch cache lookup
                best_result['battery_dispatch_pct'] = float(best_mix[6])
                best_result['battery8_dispatch_pct'] = float(best_mix[7])
                best_result['ldes_dispatch_pct'] = float(best_mix[8])
                best_result['h2_dispatch_pct'] = float(best_mix[9]) if len(best_mix) > 9 else 0.0
                best_result['demand_mwh'] = demand_twh_map[iso] * 1e6
                iso_results[t] = best_result

        results[iso] = iso_results
    return results


def _mix_resource_twh(mix, demand_twh, iso=None):
    """Extract deployed resource TWh from a raw mix vector for floor comparison.

    v5.0 mix = [cf%, sol%, wnd%, ccs%, hyd%, match%, bat4%, bat8%, ldes%, h2%]
    iso: required — used to cap hydro at 2025 absolute TWh.
    Returns dict of resource → TWh deployed.
    """
    cf, sol, wnd, ccs, hyd = mix[0], mix[1], mix[2], mix[3], mix[4]
    bat4, bat8, ldes = mix[6], mix[7], mix[8]
    h2 = mix[9] if len(mix) > 9 else 0
    # Hydro is existing-only: cap at 2025 absolute TWh regardless of demand growth
    hydro_twh = min(hyd / 100.0 * demand_twh,
                    HYDRO_CAP_TWH[iso]) if iso else hyd / 100.0 * demand_twh
    return {
        'clean_firm': cf / 100.0 * demand_twh,
        'solar':      sol / 100.0 * demand_twh,
        'wind':       wnd / 100.0 * demand_twh,
        'ccs_ccgt':   ccs / 100.0 * demand_twh,
        'hydro':      hydro_twh,
        'battery':    (bat4 + bat8) / 100.0 * demand_twh,
        'ldes':       ldes / 100.0 * demand_twh,
    }


def find_optimal_mixes_sequential(feasible_mixes, scenario, demand_twh_map):
    """Path-dependent sequential optimization for the consequential scenario.

    Both scenarios target 95% clean as the SBTi destination. Scenario A reaches
    it via sequential path-dependent procurement — chasing cheapest $/tCO₂ at each
    step, locking in prior resource commitments (floor ratchet).

    Algorithm:
      1. Find cost-optimal EF mix at 95% as the deployment target ("north star").
      2. At each threshold 50→100%, find the cheapest EF mix that doesn't overshoot
         any resource beyond the 95% target + 20% slack (prevents wasteful overbuilding).
      3. Overlay locked-in excess from prior steps: augmented = max(floor, EF_target).
      4. Price excess at LCOE — the cost penalty of path-dependent overbuilding.
      5. Floor ratchets up to augmented level.

    Demand grows year-over-year per SBTI timeline (medium growth rate).
    Existing resource floors are fixed at 2025 absolute TWh levels.
    """
    results = {}
    sens = scenario['toggles']

    for iso in ISOS:
        iso_results = {}
        iso_sens = dict(sens)
        if iso != 'CAISO':
            iso_sens['geo'] = None

        base_demand_twh = demand_twh_map[iso]

        # ------------------------------------------------------------------
        # Step 1: Find cost-optimal mix at 95% as the deployment target
        # ------------------------------------------------------------------
        t95_str = '95'
        t95_mixes = feasible_mixes.get(iso, {}).get(t95_str, [])
        gf_95 = get_demand_growth_factor(iso, 95)
        demand_95 = base_demand_twh * gf_95

        best_95_mix = None
        if t95_mixes:
            t95_arr = _mixes_to_array(t95_mixes)
            batch_95 = compute_mix_cost_batch(t95_arr, iso_sens, iso, demand_95, growth_factor=gf_95)
            best_95_mix = t95_mixes[batch_95['best_idx']]

        # Resource caps from 95% target (with 20% slack for thresholds > 95%)
        target_95_deployed = _mix_resource_twh(best_95_mix, demand_95, iso) if best_95_mix else {}
        if target_95_deployed:
            print(f"  {iso} 95% target: CF={target_95_deployed['clean_firm']:.0f} TWh, "
                  f"Sol={target_95_deployed['solar']:.0f}, Wnd={target_95_deployed['wind']:.0f}, "
                  f"CCS={target_95_deployed['ccs_ccgt']:.0f} (demand={demand_95:.0f} TWh)")

        # ------------------------------------------------------------------
        # Step 2: Sequential optimization with floor ratchet
        # ------------------------------------------------------------------
        # Floor starts at existing resource levels in ABSOLUTE TWh (2025 base year).
        existing = GRID_MIX_SHARES[iso]
        floor = {
            'clean_firm': existing.get('clean_firm', 0) / 100.0 * base_demand_twh,
            'solar': existing.get('solar', 0) / 100.0 * base_demand_twh,
            'wind': existing.get('wind', 0) / 100.0 * base_demand_twh,
            'ccs_ccgt': existing.get('ccs_ccgt', 0) / 100.0 * base_demand_twh,
            'hydro': existing.get('hydro', 0) / 100.0 * base_demand_twh,
            'battery': 0, 'ldes': 0,
        }

        for t in THRESHOLDS:
            t_str = str(int(t)) if t == int(t) else str(t)
            mixes = feasible_mixes.get(iso, {}).get(t_str, [])
            if not mixes:
                continue

            # Year-specific demand with growth
            gf = get_demand_growth_factor(iso, t)
            demand_twh = base_demand_twh * gf
            demand_mwh = demand_twh * 1e6

            # 2a. Filter EF mixes: prefer those that don't massively overshoot the
            #     95% target. Scale the 95% target to this threshold's demand level.
            #     Allow 20% slack for thresholds above 95% that may need more resources.
            demand_ratio = demand_twh / demand_95 if demand_95 > 0 else 1.0
            slack = 1.2 if t <= 95 else 1.5  # more slack above 95%

            # 2a. Vectorized target filter using numpy
            mixes_arr_seq = _mixes_to_array(mixes)
            if target_95_deployed:
                deployed_seq = _batch_resource_twh(mixes_arr_seq, demand_twh, iso)
                caps = np.array([
                    target_95_deployed.get('clean_firm', 0) * demand_ratio * slack,
                    target_95_deployed.get('solar', 0) * demand_ratio * slack,
                    target_95_deployed.get('wind', 0) * demand_ratio * slack,
                    target_95_deployed.get('ccs_ccgt', 0) * demand_ratio * slack,
                ], dtype=np.float64)
                # Check cf, sol, wnd, ccs (columns 0-3) against caps
                over_cap = (deployed_seq[:, :4] > caps[np.newaxis, :]) & (caps[np.newaxis, :] > 1.0)
                target_mask = ~np.any(over_cap, axis=1)
                if not np.any(target_mask):
                    target_mask = np.ones(len(mixes), dtype=bool)
            else:
                target_mask = np.ones(len(mixes), dtype=bool)

            # 2b. Batch-score ALL target-passing mixes
            tp_indices = np.where(target_mask)[0]
            tp_arr = mixes_arr_seq[tp_indices]
            batch_seq = compute_mix_cost_batch(tp_arr, iso_sens, iso, demand_twh, growth_factor=gf)
            deployed_tp = _batch_resource_twh(tp_arr, demand_twh, iso)
            floor_arr_seq = _floor_to_array(floor)
            # Use _resource_new_build_lcoe for the sequential case (no overrides)
            excess_lcoe_seq = np.array([
                _resource_new_build_lcoe(res, iso_sens, iso) for res in _FLOOR_KEYS
            ], dtype=np.float64)
            existing_twh_arr_seq = _floor_to_array({
                'clean_firm': existing.get('clean_firm', 0) / 100.0 * base_demand_twh,
                'solar': existing.get('solar', 0) / 100.0 * base_demand_twh,
                'wind': existing.get('wind', 0) / 100.0 * base_demand_twh,
                'ccs_ccgt': existing.get('ccs_ccgt', 0) / 100.0 * base_demand_twh,
                'hydro': existing.get('hydro', 0) / 100.0 * base_demand_twh,
                'battery': 0, 'ldes': 0,
            })

            # Floor excess (simpler version — no existing gap split, uses newbuild LCOE)
            shortfall = np.maximum(0, floor_arr_seq[np.newaxis, :] - deployed_tp)
            shortfall = np.where(shortfall > 0.01, shortfall, 0.0)
            mix_excess_arr = np.sum(shortfall / demand_twh * excess_lcoe_seq[np.newaxis, :], axis=1)

            match_fracs = batch_seq['match_scores'] / 100.0
            augmented_eff_arr = np.where(
                match_fracs > 0,
                (batch_seq['total_cost'] + mix_excess_arr) / match_fracs,
                np.inf
            )

            best_tp_idx = int(np.argmin(augmented_eff_arr))
            best_orig_idx = int(tp_indices[best_tp_idx])
            best_result = _extract_single_result(tp_arr, batch_seq, best_tp_idx, iso, demand_twh)
            best_mix = mixes[best_orig_idx]
            best_excess_per_mwh = float(mix_excess_arr[best_tp_idx])

            if not best_result:
                continue

            # 3. Compute augmented resources = max(floor, EF_target) per resource
            ef_deployed = _mix_resource_twh(best_mix, demand_twh, iso)
            augmented = {}
            excess_twh = {}
            for res in floor:
                ef_val = ef_deployed.get(res, 0)
                floor_val = floor.get(res, 0)
                augmented[res] = max(ef_val, floor_val)
                excess_twh[res] = max(0, floor_val - ef_val)

            total_excess = sum(excess_twh.values())
            excess_cost_per_mwh = best_excess_per_mwh

            # 4. Build augmented result — EF cost + excess cost penalty
            augmented_result = dict(best_result)
            augmented_result['total_cost'] = round(best_result['total_cost'] + excess_cost_per_mwh, 2)
            match_frac = best_result['match_score'] / 100
            augmented_result['effective_cost'] = round(
                augmented_result['total_cost'] / match_frac if match_frac > 0 else 0, 2)
            augmented_result['incremental'] = round(
                augmented_result['effective_cost'] - best_result['wholesale'], 2)

            # Override resource TWh with augmented values (floor + EF target)
            augmented_result['resource_twh'] = {
                res: augmented.get(res, 0) for res in RESOURCES}
            augmented_result['battery_twh'] = augmented.get('battery', 0)
            augmented_result['ldes_twh'] = augmented.get('ldes', 0)

            # 5. Recompute gas backup from augmented clean capacity
            clean_peak_mw = 0
            for r, twh in augmented_result['resource_twh'].items():
                pcc = PEAK_CAPACITY_CREDITS.get(r, 0)
                if pcc > 0 and twh > 0:
                    clean_peak_mw += (twh * 1e6 / 8760) * pcc
            clean_peak_mw += (augmented.get('battery', 0) * 1e6 / 8760) * PEAK_CAPACITY_CREDITS.get('battery', 0.95)
            clean_peak_mw += (augmented.get('ldes', 0) * 1e6 / 8760) * PEAK_CAPACITY_CREDITS.get('ldes', 0.90)

            ra_peak_mw = PEAK_DEMAND_MW[iso] * gf * (1 + RESOURCE_ADEQUACY_MARGIN)
            gaf = GAS_AVAILABILITY_FACTOR[iso]
            gas_needed_mw = max(0, ra_peak_mw - clean_peak_mw) / gaf
            augmented_result['gas_backup_mw'] = round(gas_needed_mw)
            augmented_result['new_gas_mw'] = round(max(0, gas_needed_mw - EXISTING_GAS_CAPACITY_MW[iso]))
            augmented_result['clean_peak_mw'] = round(clean_peak_mw)

            # Track new-build cost (for MAC calculation)
            augmented_result['new_build_cost_total'] = (
                best_result['new_build_cost_total'] + excess_cost_per_mwh * demand_mwh)
            augmented_result['new_gen_twh'] = round(
                best_result['new_gen_twh'] + sum(
                    v for k, v in excess_twh.items() if k != 'hydro'), 3)

            if total_excess > 1.0:
                print(f"  ↗ {iso} {t}%: {total_excess:.0f} TWh excess locked in "
                      f"(+${excess_cost_per_mwh:.1f}/MWh penalty)")

            # Preserve dispatch parameters for dispatch cache lookup
            augmented_result['battery_dispatch_pct'] = float(best_mix[6])
            augmented_result['battery8_dispatch_pct'] = float(best_mix[7])
            augmented_result['ldes_dispatch_pct'] = float(best_mix[8])
            augmented_result['h2_dispatch_pct'] = float(best_mix[9]) if len(best_mix) > 9 else 0.0
            augmented_result['demand_mwh'] = demand_mwh

            iso_results[t] = augmented_result

            # 6. Update floor: augmented resources become the new locked-in floor
            floor = dict(augmented)

        results[iso] = iso_results
    return results


def _build_learning_overrides(iso, frac):
    """Build LCOE overrides interpolated between High (FOAK) and Low (NOAK).

    frac: 0 = pure FOAK, 1 = full NOAK.
    Uprate always at Medium — existing nuclear uprates don't depend on learning.
    """
    overrides = {}
    nuc_h = NUCLEAR_NEWBUILD_LCOE['H'][iso]
    nuc_l = NUCLEAR_NEWBUILD_LCOE['L'][iso]
    overrides['nuclear_lcoe'] = nuc_h + frac * (nuc_l - nuc_h)
    overrides['uprate_lcoe'] = UPRATE_LCOE['M']  # Always Medium
    ccs_h = CCS_LCOE_45Q_ON['H'][iso]
    ccs_l = CCS_LCOE_45Q_ON['L'][iso]
    overrides['ccs_lcoe'] = ccs_h + frac * (ccs_l - ccs_h)
    if iso == 'CAISO':
        geo_h = GEOTHERMAL_LCOE['H']
        geo_l = GEOTHERMAL_LCOE['L']
        overrides['geo_lcoe'] = geo_h + frac * (geo_l - geo_h)
    ldes_h = LCOE_TABLES['ldes']['High'][iso]
    ldes_l = LCOE_TABLES['ldes']['Low'][iso]
    overrides['ldes_lcoe'] = ldes_h + frac * (ldes_l - ldes_h)
    return overrides


def _build_learning_overrides_b(iso, frac):
    """Build LCOE overrides for Scenario B with differentiated learning curves.

    frac: 0 = pure FOAK (High cost), 1 = full NOAK (Low cost).

    Nuclear: H→L interpolation (SMR learning curve, starts from FOAK).
    LDES: H→L interpolation (iron-air learning curve, starts from FOAK).
    CCS: M→L interpolation with 45Q (starts from Medium, not FOAK — CCS is
         a more mature pathway so the learning curve starts lower).
    Geothermal: H→L interpolation (CAISO only).
    Uprate: Always Medium (existing plants, no learning curve).
    """
    overrides = {}
    # Nuclear: FOAK→NOAK interpolation (H→L)
    nuc_h = NUCLEAR_NEWBUILD_LCOE['H'][iso]
    nuc_l = NUCLEAR_NEWBUILD_LCOE['L'][iso]
    overrides['nuclear_lcoe'] = nuc_h + frac * (nuc_l - nuc_h)
    overrides['uprate_lcoe'] = UPRATE_LCOE['M']  # Always Medium
    # CCS: High→Medium interpolation with 45Q (H→M, not H→L)
    # CCS is not mature — starts at pessimistic FOAK (High) but learning
    # curve only reaches Medium as NOAK endpoint (structural cost floor
    # limits how cheap CCS can get, unlike nuclear which can reach Low).
    ccs_h = CCS_LCOE_45Q_ON['H'][iso]
    ccs_m = CCS_LCOE_45Q_ON['M'][iso]
    overrides['ccs_lcoe'] = ccs_h + frac * (ccs_m - ccs_h)
    # Geothermal: FOAK→NOAK interpolation (CAISO only)
    if iso == 'CAISO':
        geo_h = GEOTHERMAL_LCOE['H']
        geo_l = GEOTHERMAL_LCOE['L']
        overrides['geo_lcoe'] = geo_h + frac * (geo_l - geo_h)
    # LDES: FOAK→NOAK interpolation (H→L)
    ldes_h = LCOE_TABLES['ldes']['High'][iso]
    ldes_l = LCOE_TABLES['ldes']['Low'][iso]
    overrides['ldes_lcoe'] = ldes_h + frac * (ldes_l - ldes_h)
    return overrides


def _get_excess_lcoe(res, sens, iso, overrides):
    """Get LCOE for pricing floor excess resources, using scenario-specific overrides.

    When the floor has more of a resource than the selected mix deploys,
    the excess must still be paid for. This function returns the $/MWh LCOE
    for that excess, respecting any learning-curve overrides.
    """
    tx_name = LEVEL_NAME[sens['tx']]

    if res == 'clean_firm':
        if overrides and 'nuclear_lcoe' in overrides:
            return overrides['nuclear_lcoe'] + get_tx('clean_firm', tx_name, iso)
        return NUCLEAR_NEWBUILD_LCOE[sens['firm']][iso] + get_tx('clean_firm', tx_name, iso)
    elif res == 'ccs_ccgt':
        if overrides and 'ccs_lcoe' in overrides:
            return overrides['ccs_lcoe'] + get_tx('ccs_ccgt', tx_name, iso)
        ccs_table = CCS_LCOE_45Q_ON if sens.get('q45') == '1' else None
        if ccs_table:
            return ccs_table[sens['ccs']][iso] + get_tx('ccs_ccgt', tx_name, iso)
        return 0
    elif res == 'ldes':
        if overrides and 'ldes_lcoe' in overrides:
            return overrides['ldes_lcoe'] + get_tx('ldes', tx_name, iso)
        ldes_name = LEVEL_NAME[sens['ldes_lvl']]
        return LCOE_TABLES['ldes'][ldes_name][iso] + get_tx('ldes', tx_name, iso)
    elif res == 'solar':
        ren_name = LEVEL_NAME[sens['ren']]
        return LCOE_TABLES['solar'][ren_name][iso] + get_tx('solar', tx_name, iso)
    elif res == 'wind':
        ren_name = LEVEL_NAME[sens['ren']]
        return LCOE_TABLES['wind'][ren_name][iso] + get_tx('wind', tx_name, iso)
    elif res == 'battery':
        batt_name = LEVEL_NAME[sens['batt']]
        return LCOE_TABLES['battery'][batt_name][iso] + get_tx('battery', tx_name, iso)
    elif res == 'hydro':
        return 0
    return 0


# ============================================================================
# PFS LOADER (for fallback when EF is empty after floor filter)
# ============================================================================

def _load_pfs_mixes(iso, threshold):
    """Load PFS mixes for a single ISO/threshold from step1 parquets.

    PFS has 4D grid: (clean_firm, solar, wind, hydro) + storage dispatch.
    CCS is NOT a separate resource in PFS — it's part of clean_firm tranche
    allocation computed during cost evaluation.

    Returns list of v5.0-format mix vectors:
        [cf%, sol%, wnd%, ccs%=0, hyd%, match%, bat4%, bat8%, ldes%, h2%]
    """
    t_str = str(int(threshold)) if threshold == int(threshold) else str(threshold)
    pfs_path = Path(f'data/step1-pfs-parquets/{iso}_t{t_str}_raw_pfs.parquet')
    if not pfs_path.exists():
        return []

    df = pd.read_parquet(pfs_path)
    mixes = []
    for _, row in df.iterrows():
        mixes.append([
            row.get('clean_firm', 0),
            row.get('solar', 0),
            row.get('wind', 0),
            0,  # ccs_ccgt — not in PFS, allocated during cost eval
            row.get('hydro', 0),
            round(row.get('hourly_match_score', 0), 1),
            row.get('battery_dispatch_pct', 0),
            row.get('battery8_dispatch_pct', 0),
            row.get('ldes_dispatch_pct', 0),
            row.get('h2_dispatch_pct', 0),
        ])
    return mixes


def _filter_mixes_by_floor(mixes, floor_twh, demand_twh, iso):
    """Filter mixes to those meeting per-resource TWh floors.

    Converts floor_twh (absolute) to floor_pct using the given demand_twh,
    then filters mixes where each resource >= floor_pct.

    Hydro floor is always capped at HYDRO_CAP_TWH[iso].

    Returns list of mixes that pass the floor filter.
    """
    # Convert floor TWh to floor percentages at this demand level
    floor_pct = {}
    for res in ['clean_firm', 'solar', 'wind', 'ccs_ccgt']:
        floor_pct[res] = floor_twh.get(res, 0) / demand_twh * 100.0 if demand_twh > 0 else 0
    # Hydro: floor in TWh, but cap at physical limit
    hydro_floor_twh = min(floor_twh.get('hydro', 0), HYDRO_CAP_TWH[iso])
    floor_pct['hydro'] = hydro_floor_twh / demand_twh * 100.0 if demand_twh > 0 else 0
    # Storage floors
    floor_pct['battery'] = floor_twh.get('battery', 0) / demand_twh * 100.0 if demand_twh > 0 else 0
    floor_pct['ldes'] = floor_twh.get('ldes', 0) / demand_twh * 100.0 if demand_twh > 0 else 0

    passed = []
    for mix in mixes:
        # mix = [cf%, sol%, wnd%, ccs%, hyd%, match%, bat4%, bat8%, ldes%, h2%]
        cf, sol, wnd, ccs, hyd = mix[0], mix[1], mix[2], mix[3], mix[4]
        bat4, bat8, ldes = mix[6], mix[7], mix[8]

        if cf < floor_pct['clean_firm'] - 0.01:
            continue
        if sol < floor_pct['solar'] - 0.01:
            continue
        if wnd < floor_pct['wind'] - 0.01:
            continue
        if ccs < floor_pct['ccs_ccgt'] - 0.01:
            continue
        # Hydro: cap the mix's hydro at physical limit before comparing
        hyd_twh = min(hyd / 100.0 * demand_twh, HYDRO_CAP_TWH[iso])
        if hyd_twh < hydro_floor_twh - 0.01:
            continue
        if (bat4 + bat8) < floor_pct['battery'] - 0.01:
            continue
        if ldes < floor_pct['ldes'] - 0.01:
            continue
        passed.append(mix)

    return passed


def _filter_pfs_by_floor_window(mixes, floor_twh, demand_twh, iso, max_pct_above=10.0):
    """Filter PFS mixes within a search window above the per-resource floor.

    For each resource: floor_pct <= mix_pct <= floor_pct + max_pct_above.
    No upper bound on hydro (existing-only, physically capped).

    Returns list of mixes within the window.
    """
    floor_pct = {}
    for res in ['clean_firm', 'solar', 'wind']:
        floor_pct[res] = floor_twh.get(res, 0) / demand_twh * 100.0 if demand_twh > 0 else 0
    # CCS is 0 in PFS
    floor_pct['ccs_ccgt'] = 0
    # Hydro
    hydro_floor_twh = min(floor_twh.get('hydro', 0), HYDRO_CAP_TWH[iso])
    floor_pct['hydro'] = hydro_floor_twh / demand_twh * 100.0 if demand_twh > 0 else 0
    # Storage
    floor_pct['battery'] = floor_twh.get('battery', 0) / demand_twh * 100.0 if demand_twh > 0 else 0
    floor_pct['ldes'] = floor_twh.get('ldes', 0) / demand_twh * 100.0 if demand_twh > 0 else 0

    passed = []
    for mix in mixes:
        cf, sol, wnd, ccs, hyd = mix[0], mix[1], mix[2], mix[3], mix[4]
        bat4, bat8, ldes = mix[6], mix[7], mix[8]

        # Per-resource: must be >= floor and <= floor + max_pct_above
        if cf < floor_pct['clean_firm'] - 0.01 or cf > floor_pct['clean_firm'] + max_pct_above + 0.01:
            continue
        if sol < floor_pct['solar'] - 0.01 or sol > floor_pct['solar'] + max_pct_above + 0.01:
            continue
        if wnd < floor_pct['wind'] - 0.01 or wnd > floor_pct['wind'] + max_pct_above + 0.01:
            continue
        # Hydro: no upper bound (existing-only, capped by physical limit)
        hyd_twh = min(hyd / 100.0 * demand_twh, HYDRO_CAP_TWH[iso])
        if hyd_twh < hydro_floor_twh - 0.01:
            continue
        # Storage: >= floor, no tight upper bound in PFS
        if (bat4 + bat8) < floor_pct['battery'] - 0.01:
            continue
        if ldes < floor_pct['ldes'] - 0.01:
            continue
        passed.append(mix)

    return passed


def _forward_step_optimization(feasible_mixes, sens, get_overrides_fn, label):
    """Core forward-stepping optimizer with per-resource TWh floor filter.

    Algorithm:
      1. At each threshold, convert prior-step deployed TWh into a per-resource
         floor percentage (floor_twh / next_demand_twh).
      2. Filter EF mixes to only those meeting ALL per-resource floors.
      3. Price filtered mixes under scenario cost assumptions.
      4. Pick the cheapest that achieves the threshold.
      5. If the floor eliminates all EF mixes, fall back to PFS:
         a. First pass: floor to floor+10% per resource
         b. Second pass: floor to floor+250% per resource (5% increments implicit in PFS grid)
         c. If a PFS mix goes under on a resource, carry that resource's cost forward.
      6. Update floor = max(prior_floor, deployed) for each resource.

    Args:
        feasible_mixes: from parse_feasible_mixes()
        sens: sensitivity toggle dict (defines base cost lookups)
        get_overrides_fn: fn(iso, threshold) → dict of LCOE overrides
        label: 'A' or 'B' for logging

    Returns: dict[iso][threshold] → result dict
    """
    results = {}

    for iso in ISOS:
        iso_sens = dict(sens)
        if iso != 'CAISO':
            iso_sens['geo'] = None

        base_demand = BASE_DEMAND_TWH[iso]
        existing = GRID_MIX_SHARES[iso]

        # Existing resource levels in absolute TWh (2025 baseline) — priced at $0
        existing_twh = {
            'clean_firm': existing.get('clean_firm', 0) / 100.0 * base_demand,
            'solar': existing.get('solar', 0) / 100.0 * base_demand,
            'wind': existing.get('wind', 0) / 100.0 * base_demand,
            'ccs_ccgt': existing.get('ccs_ccgt', 0) / 100.0 * base_demand,
            'hydro': existing.get('hydro', 0) / 100.0 * base_demand,
            'battery': 0, 'ldes': 0,
        }

        # Floor = 2025 existing clean resource levels in absolute TWh
        floor_twh = dict(existing_twh)

        iso_results = {}

        # Inject existing-only anchor entries for thresholds the 2025 fleet already meets.
        # This populates the pre-50% portion of Scenario A's trajectory with the correct
        # baseline (2025 existing clean mix, zero new build) and ensures the chart's
        # "Existing Clean" reference line aligns with the starting bars.
        exist_match = _existing_match_pct(iso)
        for t_pre in [t2 for t2 in THRESHOLDS if t2 < 50]:
            if t_pre <= exist_match:
                gf_pre = get_demand_growth_factor(iso, t_pre)
                demand_pre = base_demand * gf_pre
                iso_results[t_pre] = _build_existing_only_entry(
                    iso, t_pre, demand_pre, gf_pre, existing_twh, iso_sens)

        # Build threshold list starting at 50%
        scenario_thresholds = [t for t in THRESHOLDS if t >= 50]

        for t_idx, t in enumerate(scenario_thresholds):
            t_str = str(int(t)) if t == int(t) else str(t)

            gf = get_demand_growth_factor(iso, t)
            demand_twh = base_demand * gf
            demand_mwh = demand_twh * 1e6

            overrides = get_overrides_fn(iso, t)

            # --- Step 1: Filter EF mixes by per-resource floor ---
            ef_mixes = feasible_mixes.get(iso, {}).get(t_str, [])
            filtered = _filter_mixes_by_floor(ef_mixes, floor_twh, demand_twh, iso)
            source = 'EF'

            # --- Step 2: If no EF mixes survive, fall back to PFS ---
            if not filtered:
                pfs_mixes = _load_pfs_mixes(iso, t)
                if pfs_mixes:
                    # First pass: floor to floor+10%
                    filtered = _filter_pfs_by_floor_window(
                        pfs_mixes, floor_twh, demand_twh, iso, max_pct_above=10.0)
                    source = 'PFS+10'

                    if not filtered:
                        # Second pass: floor to floor+250%
                        filtered = _filter_pfs_by_floor_window(
                            pfs_mixes, floor_twh, demand_twh, iso, max_pct_above=250.0)
                        source = 'PFS+250'

                    if not filtered:
                        # Final fallback: take ALL PFS mixes that meet the floor
                        # as a minimum (no upper bound)
                        filtered = _filter_mixes_by_floor(
                            pfs_mixes, floor_twh, demand_twh, iso)
                        source = 'PFS-all'

                    if not filtered:
                        # Absolute last resort: relax floor, take any PFS mix,
                        # but carry under-floor resource costs forward
                        filtered = pfs_mixes
                        source = 'PFS-relaxed'

                if not filtered:
                    print(f"  ⚠ {iso} {t}% [{label}]: No mixes found (EF or PFS)")
                    continue

            if source != 'EF':
                print(f"  ↪ {iso} {t}% [{label}]: EF empty after floor filter, "
                      f"using {source} ({len(filtered)} mixes)")

            # --- Step 3: Batch-price filtered mixes, pick cheapest ---
            filt_arr = _mixes_to_array(filtered)
            batch = compute_mix_cost_batch(filt_arr, iso_sens, iso, demand_twh,
                                           overrides=overrides, growth_factor=gf)
            deployed_twh = _batch_resource_twh(filt_arr, demand_twh, iso)
            floor_arr = _floor_to_array(floor_twh)
            existing_twh_arr = _floor_to_array(existing_twh)
            excess_lcoe_arr = _build_excess_lcoe_array(iso_sens, iso, overrides)

            best_idx, best_excess_per_mwh = _batch_pick_best(
                filt_arr, batch, deployed_twh, floor_arr,
                existing_twh_arr, excess_lcoe_arr, demand_twh)

            if best_idx is None:
                continue

            best_total_cost = float(batch['total_cost'][best_idx]) + best_excess_per_mwh
            best_result = _extract_single_result(filt_arr, batch, best_idx, iso, demand_twh)
            best_mix = filtered[best_idx]

            if not best_result or not best_mix:
                continue

            # --- Step 4: Build result with augmented resources ---
            deployed = _mix_resource_twh(best_mix, demand_twh, iso)
            augmented = {}
            excess_twh_dict = {}
            for res in floor_twh:
                ef_val = deployed.get(res, 0)
                floor_val = floor_twh.get(res, 0)
                augmented[res] = max(ef_val, floor_val)
                excess_twh_dict[res] = max(0, floor_val - ef_val)

            total_excess = sum(excess_twh_dict.values())

            # Build result dict
            aug = dict(best_result)
            aug['total_cost'] = round(best_total_cost, 2)
            match_frac = best_result['match_score'] / 100.0
            aug['effective_cost'] = round(
                aug['total_cost'] / match_frac if match_frac > 0 else 0, 2)
            aug['incremental'] = round(
                aug['effective_cost'] - best_result['wholesale'], 2)

            # Augmented resource TWh (locked-in floor + new EF deployment)
            aug['resource_twh'] = {res: augmented.get(res, 0) for res in RESOURCES}
            aug['battery_twh'] = augmented.get('battery', 0)
            aug['battery8_twh'] = 0  # Not tracked separately post-augmentation
            aug['ldes_twh'] = augmented.get('ldes', 0)
            aug['demand_twh'] = demand_twh
            aug['mix_raw'] = list(best_mix)
            aug['mix_source'] = source

            # Recompute gas backup from augmented clean capacity
            clean_peak_mw = 0
            for r, twh in aug['resource_twh'].items():
                pcc = PEAK_CAPACITY_CREDITS.get(r, 0)
                if pcc > 0 and twh > 0:
                    clean_peak_mw += (twh * 1e6 / 8760) * pcc
            clean_peak_mw += (augmented.get('battery', 0) * 1e6 / 8760) * \
                PEAK_CAPACITY_CREDITS.get('battery', 0.95)
            clean_peak_mw += (augmented.get('ldes', 0) * 1e6 / 8760) * \
                PEAK_CAPACITY_CREDITS.get('ldes', 0.90)

            ra_peak_mw = PEAK_DEMAND_MW[iso] * gf * (1 + RESOURCE_ADEQUACY_MARGIN)
            gaf = GAS_AVAILABILITY_FACTOR[iso]
            gas_needed_mw = max(0, ra_peak_mw - clean_peak_mw) / gaf
            aug['gas_backup_mw'] = round(gas_needed_mw)
            aug['new_gas_mw'] = round(max(0, gas_needed_mw - EXISTING_GAS_CAPACITY_MW[iso]))
            aug['clean_peak_mw'] = round(clean_peak_mw)
            aug['existing_gas_used_mw'] = round(min(gas_needed_mw, EXISTING_GAS_CAPACITY_MW[iso]))

            # New-build cost tracking (for MAC calculation)
            aug['new_build_cost_total'] = (
                best_result['new_build_cost_total'] +
                best_excess_per_mwh * demand_mwh)
            aug['new_gen_twh'] = round(
                best_result['new_gen_twh'] +
                sum(v for k, v in excess_twh_dict.items() if k != 'hydro'), 3)

            if total_excess > 1.0:
                print(f"  ↗ {iso} {t}% [{label}]: {total_excess:.0f} TWh floor excess "
                      f"(+${best_excess_per_mwh:.1f}/MWh) [{source}]")

            iso_results[t] = aug

            # --- Step 5: Update floor = max(prior, deployed) for each resource ---
            for res in floor_twh:
                floor_twh[res] = max(floor_twh[res], deployed.get(res, 0))

        # Print trajectory summary
        for t in sorted(iso_results.keys()):
            r = iso_results[t]
            rt = r['resource_twh']
            src = r.get('mix_source', '?')
            print(f"    {t:5.1f}%: CF={rt['clean_firm']:7.0f} Sol={rt['solar']:6.0f} "
                  f"Wnd={rt['wind']:6.0f} CCS={rt.get('ccs_ccgt', 0):6.0f} "
                  f"${r['effective_cost']:.0f}/MWh [{label}] ({src})")

        results[iso] = iso_results

    return results


def _adjust_costs_no_learning(results, scenario):
    """Scenario A: Delayed learning curve — FOAK until 2036, NOAK by 2048.

    Same interpolation machinery as Scenario B, but with delayed start (2036)
    and stretched timeline (12 years vs 10). Less deployment → slower learning.

    Sources: INL SOAK data, NEA learning ranges, DOE Liftoff projections.
    In a lower-investment scenario, first SMRs don't deploy until ~2036,
    and fewer units per year means the learning curve takes 12 years to NOAK.
    """
    for iso in results:
        for t in results[iso]:
            r = results[iso][t]
            demand_twh = r['demand_twh']
            if demand_twh <= 0:
                continue

            frac = learning_fraction(t, scenario='A')
            overrides = _build_learning_overrides(iso, frac)

            total_delta = 0.0

            # 1. Uprate: step3 used H ($40), target is M ($25)
            uprate_twh = r.get('tranche_uprate_twh', 0)
            if uprate_twh > 0:
                uprate_delta = UPRATE_LCOE['M'] - UPRATE_LCOE['H']
                total_delta += uprate_delta * uprate_twh / demand_twh

            # 2. Nuclear newbuild: step3 used H, learning curve gives interpolated
            nuc_twh = r.get('tranche_nuclear_newbuild_twh', 0)
            if nuc_twh > 0:
                nuc_h = NUCLEAR_NEWBUILD_LCOE['H'][iso]
                nuc_delta = overrides['nuclear_lcoe'] - nuc_h
                total_delta += nuc_delta * nuc_twh / demand_twh

            # 3. CCS new-build
            existing_ccs_frac = GRID_MIX_SHARES[iso].get('ccs_ccgt', 0) / 100.0
            existing_ccs_twh = existing_ccs_frac * BASE_DEMAND_TWH[iso]
            total_ccs_twh = r['resource_twh'].get('ccs_ccgt', 0)
            ccs_new_twh = max(0, total_ccs_twh - existing_ccs_twh)
            if ccs_new_twh > 0:
                ccs_h = CCS_LCOE_45Q_ON['H'][iso]
                ccs_delta = overrides['ccs_lcoe'] - ccs_h
                total_delta += ccs_delta * ccs_new_twh / demand_twh

            # 4. Geothermal (CAISO only)
            if iso == 'CAISO' and 'geo_lcoe' in overrides:
                geo_twh = r.get('tranche_geo_twh', 0)
                if geo_twh > 0:
                    geo_h = GEOTHERMAL_LCOE['H']
                    geo_delta = overrides['geo_lcoe'] - geo_h
                    total_delta += geo_delta * geo_twh / demand_twh

            # 5. LDES (all new-build)
            ldes_twh = r.get('ldes_twh', 0)
            if ldes_twh > 0:
                ldes_h = LCOE_TABLES['ldes']['High'][iso]
                ldes_delta = overrides['ldes_lcoe'] - ldes_h
                total_delta += ldes_delta * ldes_twh / demand_twh

            # Apply total delta
            if abs(total_delta) > 0.001:
                r['total_cost'] = round(r['total_cost'] + total_delta, 2)
                match_frac = r['match_score'] / 100
                r['effective_cost'] = round(
                    r['total_cost'] / match_frac if match_frac > 0 else 0, 2)
                r['incremental'] = round(
                    r['effective_cost'] - r.get('wholesale', WHOLESALE_PRICES[iso]), 2)

            # Store learning curve metadata
            r['learning_fraction'] = round(frac, 3)
            r['learning_nuclear_lcoe'] = round(overrides['nuclear_lcoe'], 1)
            r['learning_ccs_lcoe'] = round(overrides['ccs_lcoe'], 1)
            r['learning_ldes_lcoe'] = round(overrides['ldes_lcoe'], 1)
    return results


def _adjust_costs_with_learning(results, scenario):
    """Scenario B: Apply learning curve — FOAK starts 2029, NOAK by 2038.

    At each threshold, compute learning_fraction(scenario='B') → interpolate:
      Nuclear/LDES: H→L (FOAK→NOAK), CCS: H→M (FOAK→Medium NOAK) with 45Q.
    Step 3 costs were computed at full FOAK (firm='H', ldes='H', ccs='H').
    We compute the cost delta from FOAK to the learning-curve-adjusted price
    for each resource tranche and apply it.

    10-year learning period (2030-2040), concave ramp (exponent 0.6).
    Sources: INL SOAK data, NEA learning ranges, DOE Liftoff.
    """
    for iso in results:
        for t in results[iso]:
            r = results[iso][t]
            demand_twh = r['demand_twh']
            if demand_twh <= 0:
                continue

            frac = learning_fraction(t, scenario='B')
            overrides = _build_learning_overrides_b(iso, frac)

            total_delta = 0.0

            # 1. Uprate: step3 used H ($40), target is M ($25)
            uprate_twh = r.get('tranche_uprate_twh', 0)
            if uprate_twh > 0:
                uprate_delta = UPRATE_LCOE['M'] - UPRATE_LCOE['H']
                total_delta += uprate_delta * uprate_twh / demand_twh

            # 2. Nuclear newbuild: step3 used H, learning curve gives interpolated
            nuc_twh = r.get('tranche_nuclear_newbuild_twh', 0)
            if nuc_twh > 0:
                nuc_h = NUCLEAR_NEWBUILD_LCOE['H'][iso]
                nuc_delta = overrides['nuclear_lcoe'] - nuc_h
                total_delta += nuc_delta * nuc_twh / demand_twh

            # 3. CCS new-build
            # CCS TWh from mix minus existing
            existing_ccs_frac = GRID_MIX_SHARES[iso].get('ccs_ccgt', 0) / 100.0
            gf = demand_twh / BASE_DEMAND_TWH[iso]
            existing_ccs_twh = existing_ccs_frac * BASE_DEMAND_TWH[iso]
            # v5.0: procurement baked into resource percentages
            total_ccs_twh = r['resource_twh'].get('ccs_ccgt', 0)
            ccs_new_twh = max(0, total_ccs_twh - existing_ccs_twh)
            if ccs_new_twh > 0:
                ccs_h = CCS_LCOE_45Q_ON['H'][iso]
                ccs_delta = overrides['ccs_lcoe'] - ccs_h
                total_delta += ccs_delta * ccs_new_twh / demand_twh

            # 4. Geothermal (CAISO only)
            if iso == 'CAISO' and 'geo_lcoe' in overrides:
                geo_twh = r.get('tranche_geo_twh', 0)
                if geo_twh > 0:
                    geo_h = GEOTHERMAL_LCOE['H']
                    geo_delta = overrides['geo_lcoe'] - geo_h
                    total_delta += geo_delta * geo_twh / demand_twh

            # 5. LDES (all new-build)
            ldes_twh = r.get('ldes_twh', 0)
            if ldes_twh > 0:
                ldes_h = LCOE_TABLES['ldes']['High'][iso]
                ldes_delta = overrides['ldes_lcoe'] - ldes_h
                total_delta += ldes_delta * ldes_twh / demand_twh

            # Apply total delta
            if abs(total_delta) > 0.001:
                r['total_cost'] = round(r['total_cost'] + total_delta, 2)
                match_frac = r['match_score'] / 100
                r['effective_cost'] = round(
                    r['total_cost'] / match_frac if match_frac > 0 else 0, 2)
                r['incremental'] = round(
                    r['effective_cost'] - r.get('wholesale', WHOLESALE_PRICES[iso]), 2)
                # Update new-build cost tracking
                r['new_build_cost_total'] = r.get('new_build_cost_total', 0) + \
                    total_delta * demand_twh * 1e6

            # Store learning curve metadata
            r['learning_fraction'] = round(frac, 3)
            r['learning_nuclear_lcoe'] = round(overrides['nuclear_lcoe'], 1)
            r['learning_ccs_lcoe'] = round(overrides['ccs_lcoe'], 1)
            r['learning_ldes_lcoe'] = round(overrides['ldes_lcoe'], 1)

    return results


def _build_scenario_key(scenario, iso):
    """Build the step 3 parquet scenario key from scenario toggles."""
    s = scenario['toggles']
    ren = s['ren']
    firm = s['firm']
    batt = s['batt']
    ldes = s['ldes_lvl']
    fuel = s['fuel']
    tx = s['tx']
    ccs = s['ccs']
    q45 = s['q45']
    geo = s.get('geo', 'X')
    if iso != 'CAISO':
        geo = 'X'
    tx_map = {'L': 'L', 'M': 'M', 'H': 'H', 'N': 'N'}
    return f"{ren}{firm}{batt}{ldes}_{fuel}_{tx_map.get(tx, tx)}_{ccs}{q45}_{geo}"


def _load_step3_results(scenario, iso):
    """Load step 3 cost optimization results for a given scenario and ISO.

    Returns dict: threshold → row dict with all step 3 columns.
    """
    parquet_path = Path(__file__).parent.parent / f'data/step3-cost-opt-parquets/step3_co_{iso}.parquet'
    if not parquet_path.exists():
        print(f"  ⚠ Step 3 parquet not found: {parquet_path}")
        return {}

    df = pd.read_parquet(parquet_path)
    key = _build_scenario_key(scenario, iso)
    subset = df[df['scenario'] == key].sort_values('threshold')
    if len(subset) == 0:
        print(f"  ⚠ No step 3 results for {iso} scenario {key}")
        return {}

    # Vectorized column operations instead of iterrows()
    s = subset.copy()
    s['demand_twh'] = s['annual_demand_mwh'] / 1e6

    # Resource TWh columns  # v5.0: procurement baked into resource percentages
    s['res_cf_twh'] = s['mix_clean_firm'] / 100.0 * s['demand_twh']
    s['res_sol_twh'] = s['mix_solar'] / 100.0 * s['demand_twh']
    s['res_wnd_twh'] = s['mix_wind'] / 100.0 * s['demand_twh']
    s['res_ccs_twh'] = s['mix_ccs_ccgt'] / 100.0 * s['demand_twh']
    s['res_hyd_twh'] = s['mix_hydro'] / 100.0 * s['demand_twh']
    s['bat_twh'] = s['battery_dispatch_pct'] / 100.0 * s['demand_twh']
    s['bat8_twh'] = s['battery8_dispatch_pct'] / 100.0 * s['demand_twh']
    s['ldes_twh_calc'] = s['ldes_dispatch_pct'] / 100.0 * s['demand_twh']
    s['h2_twh'] = (s['h2_dispatch_pct'] / 100.0 * s['demand_twh'] if 'h2_dispatch_pct' in s.columns else 0)

    # New-build cost tracking (vectorized)
    existing_pct = GRID_MIX_SHARES[iso]
    base_twh = BASE_DEMAND_TWH[iso]
    s['gf'] = s['demand_twh'] / base_twh
    # Existing percentages scaled by demand growth
    ex_cf = existing_pct.get('clean_firm', 0)
    ex_sol = existing_pct.get('solar', 0)
    ex_wnd = existing_pct.get('wind', 0)
    ex_ccs = existing_pct.get('ccs_ccgt', 0)

    # v5.0: proc=1.0, so resource percentages are already demand-scaled
    s['cf_new_pct'] = (s['mix_clean_firm'] - ex_cf / s['gf']).clip(lower=0)
    s['sol_new_pct'] = (s['mix_solar'] - ex_sol / s['gf']).clip(lower=0)
    s['wnd_new_pct'] = (s['mix_wind'] - ex_wnd / s['gf']).clip(lower=0)
    s['ccs_new_pct'] = (s['mix_ccs_ccgt'] * 100 - ex_ccs / s['gf']).clip(lower=0)
    s['new_gen_twh'] = (s['cf_new_pct'] + s['sol_new_pct'] + s['wnd_new_pct'] + s['ccs_new_pct']) / 100.0 * s['demand_twh']

    # ex_frac: sum of min(proc * resource_frac, existing_scaled_frac) for each resource
    s['ex_frac'] = 0.0
    for res_col, ex_val in [('res_cf_twh', ex_cf), ('res_sol_twh', ex_sol),
                            ('res_wnd_twh', ex_wnd), ('res_ccs_twh', ex_ccs),
                            ('res_hyd_twh', existing_pct.get('hydro', 0))]:
        resource_frac = s[res_col] / s['demand_twh']
        existing_frac = (ex_val / s['gf']) / 100.0
        s['ex_frac'] += np.minimum(resource_frac, existing_frac)

    s['new_build_per_mwh'] = (s['cost_total_cost'] - s['ex_frac'] * s['cost_wholesale']
                              - s['gas_gas_cost_per_mwh']).clip(lower=0)
    s['new_build_cost_total'] = s['new_build_per_mwh'] * s['demand_twh'] * 1e6
    s['blended_new_lcoe'] = np.where(
        s['new_gen_twh'] > 0.01,
        (s['new_build_per_mwh'] * s['demand_twh'] / s['new_gen_twh']).round(2),
        0.0)

    # Handle optional columns
    has_ccs_tranche = 'tranche_ccs_tranche_twh' in s.columns
    has_geo_tranche = 'tranche_geo_twh' in s.columns

    results = {}
    for t, row in s.set_index('threshold').iterrows():
        results[t] = {
            'threshold': t,
            'mix_raw': [row['mix_clean_firm'], row['mix_solar'], row['mix_wind'],
                        row['mix_ccs_ccgt'], row['mix_hydro'],
                        row['hourly_match_score'], row['battery_dispatch_pct'],
                        row['battery8_dispatch_pct'], row['ldes_dispatch_pct'],
                        row.get('h2_dispatch_pct', 0)],
            'demand_twh': row['demand_twh'],
            'effective_cost': row['cost_effective_cost'],
            'total_cost': row['cost_total_cost'],
            'incremental': row['cost_incremental'],
            'wholesale': row['cost_wholesale'],
            'match_score': row['hourly_match_score'],
            'resource_twh': {
                'clean_firm': row['res_cf_twh'], 'solar': row['res_sol_twh'],
                'wind': row['res_wnd_twh'], 'ccs_ccgt': row['res_ccs_twh'],
                'hydro': row['res_hyd_twh'],
            },
            'battery_twh': row['bat_twh'],
            'battery8_twh': row['bat8_twh'],
            'ldes_twh': row['ldes_twh_calc'],
            'gas_backup_mw': round(row['gas_gas_backup_needed_mw']),
            'new_gas_mw': round(row['gas_new_gas_build_mw']),
            'existing_gas_used_mw': round(row['gas_existing_gas_used_mw']),
            'clean_peak_mw': round(row['gas_clean_peak_capacity_mw']),
            'tranche_uprate_twh': row['tranche_uprate_twh'],
            'tranche_nuclear_newbuild_twh': row['tranche_nuclear_newbuild_twh'],
            'tranche_ccs_tranche_twh': row['tranche_ccs_tranche_twh'] if has_ccs_tranche else 0,
            'tranche_geo_twh': row['tranche_geo_twh'] if has_geo_tranche else 0,
            'tranche_new_cf_twh': row['tranche_new_cf_twh'],
            'new_gen_twh': round(row['new_gen_twh'], 3),
            'new_build_cost_total': row['new_build_cost_total'],
            'blended_new_lcoe': round(row['blended_new_lcoe'], 2),
        }

    return results


def _apply_floor_ratchet(step3_results, iso, sens):
    """Apply path-dependent floor ratchet to step 3 single-threshold results.

    At each threshold, augmented resources = max(floor, step3_target).
    Floor ratchets upward — you can't un-build what was deployed.
    Excess (floor > step3_target) is priced at new-build LCOE.

    Returns dict: threshold → augmented result dict.
    """
    base_demand_twh = BASE_DEMAND_TWH[iso]
    existing_twh = {k: v / 100.0 * base_demand_twh
                    for k, v in GRID_MIX_SHARES[iso].items()}
    existing_twh.setdefault('battery', 0)
    existing_twh.setdefault('ldes', 0)
    floor = dict(existing_twh)

    augmented_results = {}

    for t in THRESHOLDS:
        if t not in step3_results:
            continue

        s3 = step3_results[t]
        demand_twh = s3['demand_twh']
        demand_mwh = demand_twh * 1e6

        # Step 3 deployed resources
        deployed = dict(s3['resource_twh'])
        deployed['battery'] = s3['battery_twh'] + s3.get('battery8_twh', 0)
        deployed['ldes'] = s3['ldes_twh']

        # Augment: max(floor, step3_deployed)
        augmented = {}
        excess_twh = {}
        excess_per_mwh = 0.0
        for res in floor:
            ef_val = deployed.get(res, 0)
            floor_val = floor.get(res, 0)
            augmented[res] = max(ef_val, floor_val)
            excess = max(0, floor_val - ef_val)
            excess_twh[res] = excess
            if excess > 0.01:
                lcoe = _resource_new_build_lcoe(res, sens, iso)
                excess_per_mwh += excess / demand_twh * lcoe

        total_excess = sum(excess_twh.values())

        # Build augmented result
        result = dict(s3)
        result['total_cost'] = round(s3['total_cost'] + excess_per_mwh, 2)
        match_frac = s3['match_score'] / 100
        result['effective_cost'] = round(
            result['total_cost'] / match_frac if match_frac > 0 else 0, 2)
        result['incremental'] = round(
            result['effective_cost'] - s3['wholesale'], 2)

        result['resource_twh'] = {res: augmented.get(res, 0) for res in RESOURCES}
        result['battery_twh'] = augmented.get('battery', 0)
        result['ldes_twh'] = augmented.get('ldes', 0)

        # Recompute gas backup from augmented clean capacity
        clean_peak_mw = 0
        for r, twh in result['resource_twh'].items():
            pcc = PEAK_CAPACITY_CREDITS.get(r, 0)
            if pcc > 0 and twh > 0:
                clean_peak_mw += (twh * 1e6 / 8760) * pcc
        clean_peak_mw += (augmented.get('battery', 0) * 1e6 / 8760) * \
            PEAK_CAPACITY_CREDITS.get('battery', 0.95)
        clean_peak_mw += (augmented.get('ldes', 0) * 1e6 / 8760) * \
            PEAK_CAPACITY_CREDITS.get('ldes', 0.90)

        gf = demand_twh / base_demand_twh
        ra_peak_mw = PEAK_DEMAND_MW[iso] * gf * (1 + RESOURCE_ADEQUACY_MARGIN)
        gaf = GAS_AVAILABILITY_FACTOR[iso]
        gas_needed_mw = max(0, ra_peak_mw - clean_peak_mw) / gaf
        result['gas_backup_mw'] = round(gas_needed_mw)
        result['new_gas_mw'] = round(max(0, gas_needed_mw - EXISTING_GAS_CAPACITY_MW[iso]))
        result['clean_peak_mw'] = round(clean_peak_mw)

        # Track new-build cost (for MAC)
        result['new_build_cost_total'] = (
            s3['new_build_cost_total'] + excess_per_mwh * demand_mwh)
        result['new_gen_twh'] = round(
            s3['new_gen_twh'] +
            sum(v for k, v in excess_twh.items() if k != 'hydro'), 3)

        if total_excess > 1.0:
            print(f"    ↗ {iso} {t}%: {total_excess:.0f} TWh floor excess "
                  f"(+${excess_per_mwh:.1f}/MWh)")

        augmented_results[t] = result
        floor = dict(augmented)

    return augmented_results


def _process_iso_step3(scenario, iso):
    """Load step 3 results and apply floor ratchet for a single ISO."""
    sens = scenario['toggles']
    iso_sens = dict(sens)
    if iso != 'CAISO':
        iso_sens['geo'] = None

    step3 = _load_step3_results(scenario, iso)
    if not step3:
        return iso, {}

    return iso, _apply_floor_ratchet(step3, iso, iso_sens)


def _existing_match_pct(iso):
    """Estimate the hourly CFE match % that 2025 existing resources achieve.

    Firm resources (nuclear, hydro) are dispatchable 24/7 — ~90% effective hourly coverage.
    Variable resources (solar, wind) are intermittent — ~60% effective hourly coverage.
    Result used to determine which thresholds don't require any new build.
    """
    shares = GRID_MIX_SHARES[iso]
    cf_frac  = shares.get('clean_firm', 0) / 100.0
    hyd_frac = shares.get('hydro',      0) / 100.0
    sol_frac = shares.get('solar',      0) / 100.0
    wnd_frac = shares.get('wind',       0) / 100.0
    return min((cf_frac + hyd_frac) * 90.0 + (sol_frac + wnd_frac) * 60.0, 95.0)


def _build_existing_only_entry(iso, threshold, demand_twh, gf, existing_twh, sens):
    """Build a result entry representing 2025 existing resources with zero new build.

    Used for thresholds the existing clean fleet already achieves. The floor ratchet
    stays at 2025 levels; subsequent thresholds then compute incremental new build
    above that correct baseline rather than above an inflated EF mix.
    """
    demand_mwh = demand_twh * 1e6
    base_demand = BASE_DEMAND_TWH[iso]

    # Resource TWh = fixed 2025 absolute TWh (hydro capped)
    resource_twh = {}
    for res, pct in GRID_MIX_SHARES[iso].items():
        twh = pct / 100.0 * base_demand
        if res == 'hydro':
            twh = min(twh, HYDRO_CAP_TWH[iso])
        resource_twh[res] = twh

    # Gas backup from existing clean peak capacity
    clean_peak_mw = 0.0
    for res, twh in resource_twh.items():
        pcc = PEAK_CAPACITY_CREDITS.get(res, 0)
        if pcc > 0:
            clean_peak_mw += (twh * 1e6 / 8760) * pcc

    ra_peak_mw = PEAK_DEMAND_MW[iso] * gf * (1 + RESOURCE_ADEQUACY_MARGIN)
    gaf = GAS_AVAILABILITY_FACTOR[iso]
    gas_needed_mw = max(0, ra_peak_mw - clean_peak_mw) / gaf
    existing_gas_mw = EXISTING_GAS_CAPACITY_MW[iso]
    new_gas_mw = max(0, gas_needed_mw - existing_gas_mw)

    fuel_name = LEVEL_NAME[sens['fuel']]
    wholesale = max(5, WHOLESALE_PRICES[iso] + FUEL_ADJUSTMENTS[iso][fuel_name])
    est_match = _existing_match_pct(iso)

    return {
        'threshold': threshold,
        'demand_twh': demand_twh,
        'effective_cost': round(wholesale, 2),
        'total_cost': round(wholesale, 2),
        'incremental': 0.0,
        'wholesale': wholesale,
        'match_score': est_match,
        'resource_twh': resource_twh,
        'resource_pct': dict(GRID_MIX_SHARES[iso]),
        'battery_twh': 0.0,
        'battery8_twh': 0.0,
        'ldes_twh': 0.0,
        'battery_dispatch_pct': 0.0,
        'battery8_dispatch_pct': 0.0,
        'ldes_dispatch_pct': 0.0,
        'h2_dispatch_pct': 0.0,
        'gas_backup_mw': round(gas_needed_mw),
        'new_gas_mw': round(new_gas_mw),
        'existing_gas_used_mw': round(min(gas_needed_mw, existing_gas_mw)),
        'clean_peak_mw': round(clean_peak_mw),
        'new_build_cost_total': 0.0,
        'new_gen_twh': 0.0,
        'blended_new_lcoe': 0.0,
        'demand_mwh': demand_mwh,
    }


def find_scenario_b_mixes(feasible_mixes):
    """Scenario B: Forward-looking endpoint-optimal deployment.

    Fundamentally different from Scenario A's greedy sequential approach.

    Strategy:
      1. Find the NOAK-optimal mix at 95% with all costs Low (CCS at Medium+45Q
         as NOAK endpoint — CCS learning curve is H→M, not H→L).
         This is the "north star" — what the grid looks like in 2045.
      2. At each threshold, deploy resources toward this target using an S-curve
         ramp that frontloads firm/storage investment.
      3. Nuclear/LDES: accelerated FOAK(H)→NOAK(L) learning curve.
         CCS: FOAK(H)→NOAK(M) learning curve with 45Q.
      4. Select from feasible mixes those best matching the deployment target
         at learning-curve-adjusted prices. Strict pacing enforcement.
      5. Floor ratchet prevents un-deploying committed resources.

    This produces drastically different trajectories from Scenario A because:
      - A chases cheap carbon greedily → renewable-heavy early, FOAK cliff at 90%+
      - B works backward from optimal 2045 endpoint → balanced deployment from
        the start, with firm/CCS/storage allocated from early thresholds
      - CCS starts on a learning curve from H→M (not stuck at FOAK forever)
      - By 80-90%, B has driven nuclear/LDES toward NOAK via cumulative deployment
      - At 95%+, B benefits from NOAK firm and learned CCS; A faces cost cliff
    """
    import math
    results = {}

    # NOAK target sensitivities: all at Low except CCS at Medium+45Q
    # (CCS NOAK endpoint is Medium, not Low — structural cost floor)
    NOAK_SENS = {
        'ren': 'L', 'firm': 'L', 'batt': 'L', 'ldes_lvl': 'L',
        'fuel': 'M', 'tx': 'M', 'ccs': 'M', 'q45': '1', 'geo': 'L',
    }

    for iso in ISOS:
        iso_sens = dict(SCENARIO_B['toggles'])
        if iso != 'CAISO':
            iso_sens['geo'] = None

        noak_sens = dict(NOAK_SENS)
        if iso != 'CAISO':
            noak_sens['geo'] = None

        base_demand = BASE_DEMAND_TWH[iso]
        existing = GRID_MIX_SHARES[iso]

        # ==================================================================
        # Step 1: Find NOAK-optimal mix at 95% — the "north star"
        # All costs Low, CCS at Medium+45Q (its NOAK endpoint).
        # This is what the grid looks like in 2045 when learning investments
        # have paid off — the endpoint we're building toward.
        # ==================================================================
        gf_95 = get_demand_growth_factor(iso, 95)
        demand_95 = base_demand * gf_95
        mixes_95 = feasible_mixes.get(iso, {}).get('95', [])

        # Use full NOAK overrides for target finding
        noak_overrides = _build_learning_overrides_b(iso, 1.0)  # frac=1 = full NOAK

        if mixes_95:
            mixes_95_arr = _mixes_to_array(mixes_95)
            batch_95 = compute_mix_cost_batch(mixes_95_arr, noak_sens, iso, demand_95,
                                              overrides=noak_overrides, growth_factor=gf_95)
            best_idx_95 = batch_95['best_idx']
            best_mix_95 = mixes_95[best_idx_95]
        else:
            best_mix_95 = None

        if not best_mix_95:
            print(f"  ⚠ {iso}: No feasible mixes at 95%, skipping")
            results[iso] = {}
            continue

        target_deployed = _mix_resource_twh(best_mix_95, demand_95, iso)

        # Target resource levels at 95% NOAK
        target_cf_twh = target_deployed.get('clean_firm', 0)
        target_ccs_twh = target_deployed.get('ccs_ccgt', 0)
        target_firm_twh = target_cf_twh + target_ccs_twh
        target_ldes_twh = target_deployed.get('ldes', 0)
        target_batt_twh = (best_mix_95[6] + best_mix_95[7]) / 100.0 * demand_95
        target_sol_twh = target_deployed.get('solar', 0)
        target_wnd_twh = target_deployed.get('wind', 0)

        # Existing levels (2025 baseline)
        existing_cf_twh = existing.get('clean_firm', 0) / 100.0 * base_demand
        existing_ccs_twh = existing.get('ccs_ccgt', 0) / 100.0 * base_demand
        existing_firm_twh = existing_cf_twh + existing_ccs_twh
        existing_sol_twh = existing.get('solar', 0) / 100.0 * base_demand
        existing_wnd_twh = existing.get('wind', 0) / 100.0 * base_demand

        print(f"\n  {iso} 95% NOAK target: firm={target_firm_twh:.0f} TWh "
              f"(CF={target_cf_twh:.0f}, CCS={target_ccs_twh:.0f}), "
              f"LDES={target_ldes_twh:.0f} TWh, Batt={target_batt_twh:.0f} TWh, "
              f"Sol={target_sol_twh:.0f}, Wnd={target_wnd_twh:.0f}, "
              f"existing firm={existing_firm_twh:.0f} TWh")

        # ==================================================================
        # Step 2: Backward-from-endpoint deployment with S-curve pacing
        # ==================================================================
        existing_twh_b = {
            'clean_firm': existing_cf_twh,
            'solar': existing_sol_twh,
            'wind': existing_wnd_twh,
            'ccs_ccgt': existing_ccs_twh,
            'hydro': existing.get('hydro', 0) / 100.0 * base_demand,
            'battery': 0, 'ldes': 0,
        }
        floor = dict(existing_twh_b)

        exist_match = _existing_match_pct(iso)
        iso_results = {}

        for t in THRESHOLDS:
            gf = get_demand_growth_factor(iso, t)
            demand_twh = base_demand * gf
            demand_mwh = demand_twh * 1e6

            if t <= exist_match:
                iso_results[t] = _build_existing_only_entry(
                    iso, t, demand_twh, gf, existing_twh_b, iso_sens)
                floor = dict(existing_twh_b)
                continue

            t_str = str(int(t)) if t == int(t) else str(t)
            mixes = feasible_mixes.get(iso, {}).get(t_str, [])
            if not mixes:
                continue

            # Learning-curve overrides: nuclear/LDES H→L, CCS H→M
            frac = learning_fraction(t, scenario='B')
            overrides = _build_learning_overrides_b(iso, frac)

            # S-curve deployment fraction toward 95% target.
            # Sigmoid frontloads firm investment vs linear:
            #   T=50%: ~5%, T=70%: ~30%, T=80%: ~55%, T=90%: ~85%, T=95%: 100%
            if t <= 50:
                deploy_frac = 0.05
            elif t >= 95:
                deploy_frac = 1.0
            else:
                x = (t - 50) / (95 - 50)  # 0→1
                raw = 1.0 / (1.0 + math.exp(-6 * (x - 0.4)))
                f0 = 1.0 / (1.0 + math.exp(-6 * (0 - 0.4)))
                f1 = 1.0 / (1.0 + math.exp(-6 * (1 - 0.4)))
                deploy_frac = 0.05 + 0.95 * (raw - f0) / (f1 - f0)

            # Paced targets for firm, CCS, LDES at this threshold
            demand_ratio = demand_twh / demand_95 if demand_95 > 0 else 1.0
            paced_cf_twh = existing_cf_twh + deploy_frac * (
                target_cf_twh * demand_ratio - existing_cf_twh)
            paced_ccs_twh = existing_ccs_twh + deploy_frac * (
                target_ccs_twh * demand_ratio - existing_ccs_twh)
            paced_firm_twh = paced_cf_twh + paced_ccs_twh
            paced_ldes_twh = deploy_frac * target_ldes_twh * demand_ratio
            paced_batt_twh = deploy_frac * target_batt_twh * demand_ratio

            # ==================================================================
            # Step 3: Batch-score feasible mixes by target alignment + cost
            # Three-tier selection via vectorized numpy masks.
            # ==================================================================
            mixes_arr = _mixes_to_array(mixes)
            batch = compute_mix_cost_batch(mixes_arr, iso_sens, iso, demand_twh,
                                           overrides=overrides, growth_factor=gf)
            deployed_twh = _batch_resource_twh(mixes_arr, demand_twh, iso)
            floor_arr = _floor_to_array(floor)
            existing_twh_arr = _floor_to_array(existing_twh_b)
            excess_lcoe_arr = _build_excess_lcoe_array(iso_sens, iso, overrides)

            # Vectorized floor excess cost
            excess_per_mwh_arr = _batch_floor_excess_cost(
                deployed_twh, floor_arr, existing_twh_arr, demand_twh, excess_lcoe_arr)
            total_cost_aug_arr = batch['total_cost'] + excess_per_mwh_arr

            # Vectorized pacing tier classification
            # deployed_twh columns: [cf, sol, wnd, ccs, hyd, battery, ldes]
            mix_cf_twh = deployed_twh[:, _R_CF]
            mix_ccs_twh = deployed_twh[:, _R_CCS]
            mix_firm_twh = mix_cf_twh + mix_ccs_twh
            mix_ldes_twh = deployed_twh[:, _R_LDES]

            firm_ok = (mix_firm_twh >= paced_firm_twh * 0.85) | (paced_firm_twh < 1)
            ccs_ok = (mix_ccs_twh >= paced_ccs_twh * 0.85) | (paced_ccs_twh < 1)
            ldes_ok = (mix_ldes_twh >= paced_ldes_twh * 0.50) | (paced_ldes_twh < 1)

            tier1_mask = firm_ok & ccs_ok & ldes_ok
            tier2_mask = (mix_firm_twh >= paced_firm_twh * 0.50) | (paced_firm_twh < 1)

            # Select best from highest available tier
            if np.any(tier1_mask):
                pool_mask = tier1_mask
                source_flag = ''
            elif np.any(tier2_mask):
                pool_mask = tier2_mask
                source_flag = ' (relaxed pace)'
            else:
                pool_mask = np.ones(len(mixes), dtype=bool)
                source_flag = ' (no pace-compliant mix)'

            # Among pool, find cheapest augmented cost
            pool_costs = np.where(pool_mask, total_cost_aug_arr, np.inf)
            sel_idx = int(np.argmin(pool_costs))
            if pool_costs[sel_idx] == np.inf:
                continue

            sel_cost_aug = float(total_cost_aug_arr[sel_idx])
            sel_excess = float(excess_per_mwh_arr[sel_idx])
            sel_result = _extract_single_result(mixes_arr, batch, sel_idx, iso, demand_twh)
            sel_mix = mixes[sel_idx]
            sel_deployed_arr = deployed_twh[sel_idx]

            # Build augmented result with floor ratchet
            augmented = {}
            excess_twh = {}
            for ki, key in enumerate(_FLOOR_KEYS):
                ef_val = float(sel_deployed_arr[ki])
                floor_val = floor.get(key, 0)
                augmented[key] = max(ef_val, floor_val)
                excess_twh[key] = max(0, floor_val - ef_val)

            total_excess = sum(excess_twh.values())

            aug = dict(sel_result)
            aug['total_cost'] = round(sel_cost_aug, 2)
            match_frac = sel_result['match_score'] / 100.0
            aug['effective_cost'] = round(
                aug['total_cost'] / match_frac if match_frac > 0 else 0, 2)
            aug['incremental'] = round(
                aug['effective_cost'] - sel_result['wholesale'], 2)

            aug['resource_twh'] = {res: augmented.get(res, 0) for res in RESOURCES}
            aug['battery_twh'] = augmented.get('battery', 0)
            aug['battery8_twh'] = 0
            aug['ldes_twh'] = augmented.get('ldes', 0)
            aug['demand_twh'] = demand_twh
            aug['mix_raw'] = list(sel_mix)

            # Recompute gas backup from augmented clean capacity
            clean_peak_mw = 0
            for r, twh in aug['resource_twh'].items():
                pcc = PEAK_CAPACITY_CREDITS.get(r, 0)
                if pcc > 0 and twh > 0:
                    clean_peak_mw += (twh * 1e6 / 8760) * pcc
            clean_peak_mw += (augmented.get('battery', 0) * 1e6 / 8760) * \
                PEAK_CAPACITY_CREDITS.get('battery', 0.95)
            clean_peak_mw += (augmented.get('ldes', 0) * 1e6 / 8760) * \
                PEAK_CAPACITY_CREDITS.get('ldes', 0.90)

            ra_peak_mw = PEAK_DEMAND_MW[iso] * gf * (1 + RESOURCE_ADEQUACY_MARGIN)
            gaf = GAS_AVAILABILITY_FACTOR[iso]
            gas_needed_mw = max(0, ra_peak_mw - clean_peak_mw) / gaf
            aug['gas_backup_mw'] = round(gas_needed_mw)
            aug['new_gas_mw'] = round(max(0, gas_needed_mw - EXISTING_GAS_CAPACITY_MW[iso]))
            aug['clean_peak_mw'] = round(clean_peak_mw)
            aug['existing_gas_used_mw'] = round(min(gas_needed_mw, EXISTING_GAS_CAPACITY_MW[iso]))

            # Dispatch parameters
            aug['battery_dispatch_pct'] = float(sel_mix[6])
            aug['battery8_dispatch_pct'] = float(sel_mix[7])
            aug['ldes_dispatch_pct'] = float(sel_mix[8])
            aug['h2_dispatch_pct'] = float(sel_mix[9]) if len(sel_mix) > 9 else 0.0
            aug['demand_mwh'] = demand_mwh

            # New-build cost tracking (for MAC)
            aug['new_build_cost_total'] = (
                sel_result['new_build_cost_total'] + sel_excess * demand_mwh)
            aug['new_gen_twh'] = round(
                sel_result['new_gen_twh'] +
                sum(v for k, v in excess_twh.items() if k != 'hydro'), 3)

            # Learning curve metadata
            aug['learning_fraction'] = round(frac, 3)
            aug['learning_nuclear_lcoe'] = round(overrides['nuclear_lcoe'], 1)
            aug['learning_ccs_lcoe'] = round(overrides['ccs_lcoe'], 1)
            aug['learning_ldes_lcoe'] = round(overrides['ldes_lcoe'], 1)
            aug['paced_firm_target_twh'] = round(paced_firm_twh, 1)
            aug['paced_ccs_target_twh'] = round(paced_ccs_twh, 1)
            aug['deploy_fraction'] = round(deploy_frac, 3)

            if total_excess > 1.0:
                print(f"  ↗ {iso} {t}% [B]: {total_excess:.0f} TWh excess "
                      f"(+${sel_excess:.1f}/MWh){source_flag}")

            iso_results[t] = aug

            # Floor ratchet: lock in deployed + augmented resources
            for res in floor:
                floor[res] = max(floor[res], augmented.get(res, 0))

        # Print trajectory summary
        for t in sorted(iso_results.keys()):
            r = iso_results[t]
            rt = r['resource_twh']
            lf = r.get('learning_fraction', 0)
            paced = r.get('paced_firm_target_twh', 0)
            actual_firm = rt.get('clean_firm', 0) + rt.get('ccs_ccgt', 0)
            df = r.get('deploy_fraction', 0)
            print(f"    {t:5.1f}%: CF={rt['clean_firm']:7.0f} Sol={rt['solar']:6.0f} "
                  f"Wnd={rt['wind']:6.0f} CCS={rt.get('ccs_ccgt', 0):6.0f} "
                  f"Firm={actual_firm:6.0f}/{paced:5.0f} pace "
                  f"DF={df:.2f} LF={lf:.2f} ${r['effective_cost']:.0f}/MWh [B]")

        results[iso] = iso_results

    return results


def find_scenario_a_mixes(feasible_mixes):
    """Scenario A: Pure Consequential — forward-stepping with static FOAK firm costs.

    At each threshold, evaluates ALL feasible EF mixes under static cost assumptions
    and picks the cheapest augmented effective cost. Floor ratchets.

    With expensive firm (FOAK forever), the optimizer gravitates toward renewable-heavy
    mixes at early thresholds. At late thresholds (90%+), firm becomes unavoidable
    but stays at FOAK prices — creating the cost cliff, renewable overbuilding, and
    asset stranding that characterizes the "chase cheap carbon" strategy.

    Cost: Low renewables, Medium batteries/uprates/tx, High (static FOAK) firm/CCS/LDES/geo.
    """
    def get_overrides_a(iso, threshold):
        # Only override: uprate at Medium ($25/MWh) — existing plants
        # Firm/CCS/LDES/geo stay at FOAK (High) at ALL thresholds — no learning
        return {'uprate_lcoe': UPRATE_LCOE['M']}

    print(f"  Cost: ren=Low, batt=Med, tx=Med, uprate=Med, firm/CCS/LDES/geo=High (static FOAK)")
    return _forward_step_optimization(
        feasible_mixes, SCENARIO_A['toggles'], get_overrides_a, 'A')


# ============================================================================
# CONSEQUENTIAL QUEUE BUILDER
# ============================================================================

def _get_dispatch_co2_for_mix(iso, mix_result, egrid, demand_data, gen_profiles,
                              dispatch_caches):
    """Get CO₂ displacement for a scenario mix result using dispatch cache.

    mix_result: dict from compute_mix_cost() with resource_pct, etc.
    Returns compute_co2_from_dispatch() result or None.
    """
    resource_pcts = mix_result.get('resource_pct', {})
    if not resource_pcts:
        return None

    # v5.0: procurement baked into resource percentages (dispatch_utils defaults to 100)
    bat_pct = mix_result.get('battery_dispatch_pct', 0)
    bat8_pct = mix_result.get('battery8_dispatch_pct', 0)
    ldes_pct = mix_result.get('ldes_dispatch_pct', 0)
    h2_pct = mix_result.get('h2_dispatch_pct', 0)

    supply_profiles = get_supply_profiles(iso, gen_profiles)
    demand_norm, _ = get_demand_profile(iso, demand_data)
    cache = dispatch_caches.get(iso, {})

    dispatch_result, _ = get_or_compute_dispatch(
        iso, demand_norm, supply_profiles, resource_pcts,
        battery_dispatch_pct=bat_pct, battery8_dispatch_pct=bat8_pct,
        ldes_dispatch_pct=ldes_pct, cache=cache)

    dispatch_caches[iso] = cache

    # Use demand from the mix result for proper scaling
    demand_mwh = mix_result.get('demand_mwh', 0)
    if demand_mwh <= 0:
        # Fallback: estimate from resource_twh
        demand_twh = sum(mix_result.get('resource_twh', {}).values())
        demand_mwh = demand_twh * 1e6 if demand_twh > 0 else BASE_DEMAND_TWH[iso] * 1e6

    return compute_co2_from_dispatch(iso, dispatch_result, egrid, demand_mwh)


def build_consequential_queue(scenario_results, egrid, fossil_mix, cfr_fn=None,
                               demand_data=None, gen_profiles=None, dispatch_caches=None):
    """Build the consequential deployment queue for a scenario.

    Uses dispatch-cache-based CO₂ accounting when demand_data/gen_profiles are provided.
    Falls back to analytical compute_fossil_retirement() otherwise.
    """
    use_dispatch = demand_data is not None and gen_profiles is not None
    if cfr_fn is None:
        cfr_fn = compute_fossil_retirement
    if dispatch_caches is None:
        dispatch_caches = {}

    # Pre-compute dispatch CO₂ for all threshold mixes if using dispatch model
    _co2_cache = {}  # (iso, threshold) → co2_result
    if use_dispatch:
        for iso in ISOS:
            iso_data = scenario_results.get(iso, {})
            for t in THRESHOLDS:
                if t not in iso_data:
                    continue
                mix_result = iso_data[t]
                co2 = _get_dispatch_co2_for_mix(iso, mix_result, egrid,
                                                 demand_data, gen_profiles, dispatch_caches)
                if co2:
                    _co2_cache[(iso, t)] = co2

    zone_metrics = []

    for iso in ISOS:
        iso_data = scenario_results[iso]
        baseline_clean = sum(GRID_MIX_SHARES[iso].values())

        for zone in ZONES:
            t_start, t_end = zone['start'], zone['end']
            if t_start not in iso_data or t_end not in iso_data:
                continue

            start = iso_data[t_start]
            end = iso_data[t_end]

            demand_twh_start = get_demand_twh(iso, t_start)
            demand_twh_end = get_demand_twh(iso, t_end)

            delta_cost_per_mwh = end['effective_cost'] - start['effective_cost']

            # CO₂ displacement: dispatch-based or analytical fallback
            if use_dispatch and (iso, t_start) in _co2_cache and (iso, t_end) in _co2_cache:
                co2_s = _co2_cache[(iso, t_start)]
                co2_e = _co2_cache[(iso, t_end)]
                delta_co2_tons = co2_e['total_co2_abated_tons'] - co2_s['total_co2_abated_tons']
                co2_displaced_mt = delta_co2_tons / 1e6
                delta_displaced_twh = co2_e['displaced_twh'] - co2_s['displaced_twh']
                avg_rate = co2_displaced_mt / delta_displaced_twh if delta_displaced_twh > 0.01 else co2_e['weighted_emission_rate']
            else:
                clean_twh_start = max(0, (t_start - baseline_clean) / 100.0 * demand_twh_start)
                clean_twh_end = max(0, (t_end - baseline_clean) / 100.0 * demand_twh_end)
                delta_clean_twh = clean_twh_end - clean_twh_start
                rate_start, _ = cfr_fn(iso, t_start, egrid, fossil_mix)
                rate_end, _ = cfr_fn(iso, t_end, egrid, fossil_mix)
                avg_rate = (rate_start + rate_end) / 2
                co2_displaced_mt = delta_clean_twh * avg_rate

            # MAC = newbuild LCOE / emission rate at zone endpoint
            # Use blended new-build LCOE ($/MWh of new clean energy, excludes gas)
            newbuild_lcoe = end.get('blended_new_lcoe', 0)

            # Emission rate at endpoint from dispatch or analytical model
            if use_dispatch and (iso, t_end) in _co2_cache:
                endpoint_emission_rate = _co2_cache[(iso, t_end)]['weighted_emission_rate']
            else:
                endpoint_emission_rate = avg_rate

            if endpoint_emission_rate > 0.0001 and newbuild_lcoe > 0:
                marginal_mac = newbuild_lcoe / endpoint_emission_rate
            else:
                marginal_mac = float('inf')

            # Resource deltas
            delta_resources = {}
            for res in RESOURCES:
                delta_resources[res] = end['resource_twh'][res] - start['resource_twh'][res]
            delta_resources['battery'] = (end['battery_twh'] + end.get('battery8_twh', 0)) - \
                                         (start['battery_twh'] + start.get('battery8_twh', 0))
            delta_resources['ldes'] = end['ldes_twh'] - start['ldes_twh']

            year_start = SBTI_YEAR_MAP.get(t_start, 2025)
            year_end = SBTI_YEAR_MAP.get(t_end, 2050)

            zone_metrics.append({
                'iso': iso,
                'zone_label': zone['label'],
                'threshold_start': t_start,
                'threshold_end': t_end,
                'year_start': year_start,
                'year_end': year_end,
                'marginal_mac': round(marginal_mac, 1) if marginal_mac < 9999 else 9999,
                'newbuild_lcoe': round(newbuild_lcoe, 1),
                'endpoint_emission_rate': round(endpoint_emission_rate, 4),
                'co2_displaced_mt': round(co2_displaced_mt, 2),
                'delta_cost_per_mwh': round(delta_cost_per_mwh, 2),
                'delta_resources': {k: round(v, 1) for k, v in delta_resources.items()},
                'gas_backup_mw_start': start['gas_backup_mw'],
                'gas_backup_mw_end': end['gas_backup_mw'],
                'delta_gas_mw': end['gas_backup_mw'] - start['gas_backup_mw'],
                'new_gas_mw_end': end['new_gas_mw'],
                'eff_cost_start': start['effective_cost'],
                'eff_cost_end': end['effective_cost'],
            })

    # Constrained greedy sort: pick lowest-MAC step whose prerequisite
    # (prior threshold zone for same ISO) has already been placed.
    # This allows interleaving ISOs freely while ensuring no ISO jumps
    # ahead in threshold order (e.g., can't deploy 75→90% before 50→75%).
    zone_metrics.sort(key=lambda x: x['marginal_mac'])

    # Build prerequisite map: for each (iso, zone), what zone must come first?
    zone_order = {(z['start'], z['end']): i for i, z in enumerate(ZONES)}
    iso_next_zone = {iso: 0 for iso in ISOS}  # index into ZONES each ISO is at

    ordered = []
    remaining = list(zone_metrics)
    while remaining:
        placed = False
        for step in remaining:
            iso = step['iso']
            zone_idx = zone_order.get((step['threshold_start'], step['threshold_end']), -1)
            if zone_idx == iso_next_zone[iso]:
                ordered.append(step)
                iso_next_zone[iso] = zone_idx + 1
                remaining.remove(step)
                placed = True
                break
        if not placed:
            # No eligible step found — add remaining in MAC order
            ordered.extend(remaining)
            break

    for i, step in enumerate(ordered):
        step['queue_position'] = i + 1

    return ordered


# ============================================================================
# STRANDING ANALYSIS
# ============================================================================

def compute_stranding(scenario_results):
    """Compute stranding analysis for each ISO: peak vs final resource levels."""
    stranding = {}
    for iso in ISOS:
        iso_data = scenario_results[iso]
        iso_stranding = {}

        for res in RESOURCES + ['battery', 'ldes']:
            values = []
            for t in THRESHOLDS:
                if t not in iso_data:
                    continue
                if res in RESOURCES:
                    val = iso_data[t]['resource_twh'].get(res, 0)
                elif res == 'battery':
                    val = iso_data[t].get('battery_twh', 0) + iso_data[t].get('battery8_twh', 0)
                else:
                    val = iso_data[t].get(f'{res}_twh', 0)
                values.append((t, val))

            if not values:
                continue

            peak_t, peak_val = max(values, key=lambda x: x[1])
            # Final = value at 99% (or highest available)
            final_candidates = [(t, v) for t, v in values if t >= 99]
            if final_candidates:
                final_t, final_val = final_candidates[0]
            else:
                final_t, final_val = values[-1]

            ratio = peak_val / final_val if final_val > 0.1 else 1.0

            iso_stranding[res] = {
                'peak_twh': round(peak_val, 1),
                'peak_threshold': peak_t,
                'final_twh': round(final_val, 1),
                'final_threshold': final_t,
                'stranding_ratio': round(ratio, 2),
                'values_by_threshold': {t: round(v, 1) for t, v in values},
            }

        stranding[iso] = iso_stranding
    return stranding


# ============================================================================
# DOMINO SEQUENCE — SHOW WHICH ISOs FALL FIRST AT EACH MAC LEVEL
# ============================================================================

def compute_domino_sequence(queue_a, queue_b):
    """
    Build a domino sequence showing step-by-step what happens as you follow
    the cheapest $/tCO₂ path (A) vs clean firm path (B).

    Shows cumulative: CO₂ displaced, cost, gas capacity, stranded risk.
    """
    for scenario_label, queue in [('consequential', queue_a), ('hourly_matching', queue_b)]:
        running_co2 = 0
        running_cost = 0
        iso_thresholds = {iso: 50 for iso in ISOS}
        total_gas = sum(queue[0]['gas_backup_mw_start'] for q in [queue]
                        for s in [q[0]] if s['threshold_start'] == 50 and s['iso'] == s['iso'])

        for step in queue:
            running_co2 += step['co2_displaced_mt']
            iso_thresholds[step['iso']] = step['threshold_end']


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("DUAL-SCENARIO COMPARISON: PURE CONSEQUENTIAL vs HOURLY MATCHING")
    print("  A: Pure Consequential — FOAK firm costs (no learning), cheap $/tCO₂ first")
    print("  B: Hourly Matching — endpoint-optimal, backward deployment, H→L/H→M learning")
    print("=" * 80)

    # Load feasible mixes from shared-data.js (physics-validated EF mixes)
    print("\nLoading feasible mixes from shared-data.js...")
    feasible_mixes = parse_feasible_mixes()
    total_mixes = sum(len(v) for iso_data in feasible_mixes.values()
                      for v in iso_data.values())
    print(f"  Loaded {total_mixes} feasible mixes across {len(feasible_mixes)} ISOs")

    # Load egrid and fossil mix data
    with open('data/egrid_emission_rates.json') as f:
        egrid = json.load(f)
    from eia_data_io import load_fossil_mix
    fossil_mix = load_fossil_mix()

    # Load demand/gen profiles and dispatch caches for hourly emission accounting
    print("\n  Loading demand and generation profiles for dispatch-based CO₂...")
    demand_data, gen_profiles, _, _ = load_common_data()

    dispatch_caches = {}
    print("  Loading dispatch caches...")
    for iso in ISOS:
        cache = load_dispatch_cache(iso)
        if cache:
            dispatch_caches[iso] = cache
            print(f"    {iso}: {len(cache)} cached mixes")
        else:
            print(f"    {iso}: no cache (will compute live)")

    # ==========================================================================
    # Scenario A: Pure Consequential — greedy forward-stepping, static FOAK firm
    # Evaluates ALL feasible EF mixes under static cost assumptions.
    # With expensive firm (High/FOAK forever), optimizer picks renewable-heavy
    # mixes at each step. Hits cost cliff when firm is finally needed at 90%+.
    # ==========================================================================
    print("\nScenario A (Pure Consequential): Forward-stepping, FOAK firm...")
    results_a = find_scenario_a_mixes(feasible_mixes)

    # ==========================================================================
    # Scenario B: Hourly Matching — forward-stepping with FOAK→NOAK learning curve
    # Evaluates ALL feasible EF mixes under learning-curve-adjusted prices.
    # With declining firm costs, optimizer includes more firm at each step,
    # yielding smoother cost trajectory and less stranding than Scenario A.
    # ==========================================================================
    print("\nScenario B (Hourly Matching): Endpoint-optimal, backward from 95% NOAK...")
    results_b = find_scenario_b_mixes(feasible_mixes)

    # Memoize compute_fossil_retirement as fallback
    _cfr_cache = {}

    def cfr_cached(iso, threshold, er, fm, demand_growth_factor=1.0):
        key = (iso, threshold, demand_growth_factor)
        if key not in _cfr_cache:
            _cfr_cache[key] = compute_fossil_retirement(iso, threshold, er, fm, demand_growth_factor)
        return _cfr_cache[key]

    # Build consequential queues with dispatch-cache-based CO₂ accounting
    print("\nBuilding consequential queues (dispatch-cache emission accounting)...")
    queue_a = build_consequential_queue(results_a, egrid, fossil_mix, cfr_fn=cfr_cached,
                                         demand_data=demand_data, gen_profiles=gen_profiles,
                                         dispatch_caches=dispatch_caches)
    queue_b = build_consequential_queue(results_b, egrid, fossil_mix, cfr_fn=cfr_cached,
                                         demand_data=demand_data, gen_profiles=gen_profiles,
                                         dispatch_caches=dispatch_caches)

    # Stranding analysis
    print("\nComputing stranding analysis...")
    stranding_a = compute_stranding(results_a)
    stranding_b = compute_stranding(results_b)

    # ========================================================================
    # PRINT COMPARISON
    # ========================================================================

    for scenario, results, queue, stranding, label in [
        (SCENARIO_A, results_a, queue_a, stranding_a, 'A'),
        (SCENARIO_B, results_b, queue_b, stranding_b, 'B'),
    ]:
        print(f"\n{'=' * 120}")
        print(f"SCENARIO {label}: {scenario['name'].upper()}")
        print(f"  {scenario['description']}")
        print(f"  Toggles: ren={scenario['toggles']['ren']}, firm={scenario['toggles']['firm']}, "
              f"batt={scenario['toggles']['batt']}, ldes={scenario['toggles']['ldes_lvl']}, "
              f"ccs={scenario['toggles']['ccs']}, fuel={scenario['toggles']['fuel']}, "
              f"tx={scenario['toggles']['tx']}")
        print(f"{'=' * 120}")

        # Queue
        print(f"\n{'#':>3} {'ISO':<7} {'Zone':<12} {'MAC $/t':>10} {'CO₂ MT':>8} {'ΔCost':>8} "
              f"{'Gas End':>10} {'ΔGas':>10} {'Primary Resource':>40}")
        print("-" * 120)

        for step in queue:
            deltas = step['delta_resources']
            sorted_d = sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True)
            top = sorted_d[0]
            top_str = f"{top[0]}: {'+' if top[1]>0 else ''}{top[1]:.0f} TWh"
            if len(sorted_d) > 1 and abs(sorted_d[1][1]) > 1:
                r2 = sorted_d[1]
                top_str += f", {r2[0]}: {'+' if r2[1]>0 else ''}{r2[1]:.0f}"

            mac_str = f"${step['marginal_mac']:,.0f}" if step['marginal_mac'] < 9999 else "$∞"
            print(f"{step['queue_position']:>3} {step['iso']:<7} {step['zone_label']:<12} "
                  f"{mac_str:>10} {step['co2_displaced_mt']:>7.1f} "
                  f"${step['delta_cost_per_mwh']:>6.1f} "
                  f"{step['gas_backup_mw_end']:>9,} {step['delta_gas_mw']:>+9,} "
                  f"{top_str:>40}")

        total_co2 = sum(s['co2_displaced_mt'] for s in queue)
        total_cost_delta = sum(s['delta_cost_per_mwh'] * BASE_DEMAND_TWH[s['iso']] * 1e6
                               for s in queue) / 1e9
        print(f"\nTotal CO₂ displaced: {total_co2:.1f} MT")
        print(f"Total cost delta: ${total_cost_delta:.1f}B")

        # Stranding summary
        print(f"\nStranding flags (peak/final ratio > 1.5):")
        for iso in ISOS:
            for res in ['wind', 'solar', 'clean_firm']:
                s = stranding[iso].get(res, {})
                if s.get('stranding_ratio', 0) > 1.5 and s.get('peak_twh', 0) > 1:
                    print(f"  ⚠ {iso} {res}: peak {s['peak_twh']:.0f} TWh @ {s['peak_threshold']}% → "
                          f"final {s['final_twh']:.0f} TWh @ {s['final_threshold']}% "
                          f"(ratio: {s['stranding_ratio']:.1f}x)")

    # ========================================================================
    # SIDE-BY-SIDE COMPARISON
    # ========================================================================

    print(f"\n{'=' * 150}")
    print("SIDE-BY-SIDE: RESOURCE MIX + GAS + COST AT EACH THRESHOLD (demand grows per SBTI year)")
    print(f"{'=' * 150}")

    for iso in ISOS:
        print(f"\n{'─' * 130}")
        print(f"  {iso} (base demand: {BASE_DEMAND_TWH[iso]:.1f} TWh, growth: {DEMAND_GROWTH_RATES[iso]*100:.1f}%/yr)")
        print(f"{'─' * 130}")
        print(f"{'Thr':>5} {'Year':>5} {'Dem':>6} │ {'CF_A':>6} {'Sol_A':>6} {'Wnd_A':>6} {'Gas_A':>8} {'$/MWh_A':>8} │ "
              f"{'CF_B':>6} {'Sol_B':>6} {'Wnd_B':>6} {'Gas_B':>8} {'$/MWh_B':>8} │ "
              f"{'ΔCost':>7} {'ΔGas':>8}")
        print("-" * 130)

        for t in THRESHOLDS:
            year = SBTI_YEAR_MAP.get(t, '?')
            demand_twh = get_demand_twh(iso, t)
            a = results_a.get(iso, {}).get(t, {})
            b = results_b.get(iso, {}).get(t, {})
            if not a or not b:
                continue

            a_rt = a['resource_twh']
            b_rt = b['resource_twh']

            delta_cost = b['effective_cost'] - a['effective_cost']
            delta_gas = b['gas_backup_mw'] - a['gas_backup_mw']

            print(f"{t:>5} {year:>5} {demand_twh:>5.0f} │ "
                  f"{a_rt['clean_firm']:>6.0f} {a_rt['solar']:>6.0f} {a_rt['wind']:>6.0f} "
                  f"{a['gas_backup_mw']:>8,} {a['effective_cost']:>7.1f} │ "
                  f"{b_rt['clean_firm']:>6.0f} {b_rt['solar']:>6.0f} {b_rt['wind']:>6.0f} "
                  f"{b['gas_backup_mw']:>8,} {b['effective_cost']:>7.1f} │ "
                  f"{delta_cost:>+7.1f} {delta_gas:>+8,}")

    # ========================================================================
    # DOMINO SEQUENCE — MAC ESCALATION
    # ========================================================================

    print(f"\n{'=' * 140}")
    print("DOMINO SEQUENCE: CUMULATIVE CO₂ + GAS + COST AS YOU CLIMB THE MAC LADDER")
    print(f"{'=' * 140}")

    for scenario, queue, label in [(SCENARIO_A, queue_a, 'A'), (SCENARIO_B, queue_b, 'B')]:
        print(f"\nScenario {label}: {scenario['name']}")
        print(f"{'#':>3} {'ISO':<7} {'Zone':<12} {'MAC':>10} {'Cum CO₂':>9} {'Cum $B':>8} "
              f"{'Gas(end)':>10} {'New Gas':>10} {'ISO Progress':>50}")
        print("-" * 130)

        cum_co2 = 0
        cum_cost = 0
        iso_progress = {iso: 50 for iso in ISOS}

        for step in queue:
            cum_co2 += step['co2_displaced_mt']
            cum_cost += step['delta_cost_per_mwh'] * BASE_DEMAND_TWH[step['iso']] * 1e6 / 1e9
            iso_progress[step['iso']] = step['threshold_end']

            prog_str = " | ".join(f"{iso}:{iso_progress[iso]:.0f}%" for iso in ISOS)
            mac_str = f"${step['marginal_mac']:,.0f}" if step['marginal_mac'] < 9999 else "$∞"

            print(f"{step['queue_position']:>3} {step['iso']:<7} {step['zone_label']:<12} "
                  f"{mac_str:>10} {cum_co2:>8.1f} {cum_cost:>+7.1f} "
                  f"{step['gas_backup_mw_end']:>10,} {step['new_gas_mw_end']:>10,} "
                  f"{prog_str:>50}")

    # ========================================================================
    # WRITE OUTPUT FILES
    # ========================================================================

    # Pre-compute dispatch CO₂ for trajectory stepwise MAC
    _traj_co2_cache = {}
    for scenario_results in [results_a, results_b]:
        for iso in ISOS:
            for t in THRESHOLDS:
                d = scenario_results.get(iso, {}).get(t, {})
                if d and (iso, t, id(scenario_results)) not in _traj_co2_cache:
                    co2 = _get_dispatch_co2_for_mix(iso, d, egrid,
                                                     demand_data, gen_profiles, dispatch_caches)
                    if co2:
                        _traj_co2_cache[(iso, t, id(scenario_results))] = co2

    # Build resource trajectories with PCHIP-smoothed marginal MAC
    def _build_trajectory(results, enforce_monotonic=False):
        """Build trajectory dicts from optimization results.

        MAC smoothing uses a two-pass approach:
          Pass 1: Collect cumulative (CO2_abated, new_build_cost) at each threshold.
          Pass 2: Fit a PCHIP monotone spline to the cumulative supply curve,
                  take its derivative to get marginal MAC, then apply isotonic
                  regression to enforce non-decreasing marginal cost.

        This eliminates noisy spikes from independent per-threshold optimization
        while preserving the expected hockey-stick shape at high decarbonization.
        """
        results_id = id(results)
        traj = {}
        for iso in ISOS:
            iso_traj = []
            base_demand_twh = BASE_DEMAND_TWH[iso]

            # --- Pass 1: collect raw cumulative data points ---
            raw_entries = []  # list of (threshold, d, demand_twh)
            for t in THRESHOLDS:
                d = results.get(iso, {}).get(t, {})
                if not d:
                    continue
                demand_twh = get_demand_twh(iso, t)
                raw_entries.append((t, d, demand_twh))

            # Collect cumulative CO2 abated and new-build cost at each threshold
            cum_co2_points = []  # cumulative CO2 abated (tons)
            cum_cost_points = []  # cumulative new-build cost ($)
            valid_thresholds = []  # thresholds with valid CO2 data

            for t, d, _ in raw_entries:
                co2_data = _traj_co2_cache.get((iso, t, results_id))
                if co2_data:
                    cum_co2 = co2_data['total_co2_abated_tons']
                else:
                    # Fallback: estimate from emission rate × new generation
                    new_gen_twh = d.get('new_gen_twh', 0)
                    rate, _ = cfr_cached(iso, t, egrid, fossil_mix)
                    cum_co2 = new_gen_twh * 1e6 * rate if rate > 0 else 0

                cum_cost = d.get('new_build_cost_total', 0)
                cum_co2_points.append(cum_co2)
                cum_cost_points.append(cum_cost)
                valid_thresholds.append(t)

            # --- Pass 2: PCHIP spline + isotonic regression for smooth MAC ---
            smoothed_mac = {}  # threshold -> smoothed MAC value

            if len(cum_co2_points) >= 3:
                co2_arr = np.array(cum_co2_points, dtype=np.float64)
                cost_arr = np.array(cum_cost_points, dtype=np.float64)

                # Ensure cumulative CO2 is strictly increasing for PCHIP
                # (monotone interpolant requires strictly increasing x)
                mask = np.ones(len(co2_arr), dtype=bool)
                for i in range(1, len(co2_arr)):
                    if co2_arr[i] <= co2_arr[i - 1]:
                        mask[i] = False
                co2_mono = co2_arr[mask]
                cost_mono = cost_arr[mask]
                thresholds_mono = [valid_thresholds[i] for i in range(len(valid_thresholds)) if mask[i]]

                if len(co2_mono) >= 3:
                    # Fit PCHIP spline: cost = f(co2)
                    # PCHIP preserves monotonicity and avoids overshoot
                    pchip = PchipInterpolator(co2_mono, cost_mono)

                    # Derivative at each data point = marginal MAC ($/ton)
                    raw_mac = pchip.derivative()(co2_mono)

                    # Clamp any negative derivatives to a small positive value
                    raw_mac = np.maximum(raw_mac, 0.01)

                    # Apply isotonic regression to enforce non-decreasing MAC
                    # This is the key: as decarbonization deepens, marginal
                    # cost should never decrease (hockey stick, not zigzag)
                    iso_result = isotonic_regression(raw_mac)
                    smoothed_vals = iso_result.x if hasattr(iso_result, 'x') else iso_result

                    for i, t in enumerate(thresholds_mono):
                        val = float(smoothed_vals[i])
                        # Cap extreme values at 9999
                        smoothed_mac[t] = round(min(val, 9999), 1)

            # --- Build trajectory entries with smoothed MAC ---
            for t, d, demand_twh in raw_entries:
                stepwise_mac = smoothed_mac.get(t, None)
                # First threshold has no MAC (no prior to diff against)
                if t == raw_entries[0][0]:
                    stepwise_mac = None

                cf_twh = d['resource_twh'].get('clean_firm', 0)
                ccs_twh = d['resource_twh'].get('ccs_ccgt', 0)
                existing_cf_twh = GRID_MIX_SHARES[iso].get('clean_firm', 0) / 100.0 * base_demand_twh
                firm_total_twh = max(0, cf_twh - existing_cf_twh) + ccs_twh
                # Annual cost ($B) = effective_cost ($/MWh) × demand (TWh) × 1e6 / 1e9
                annual_cost_billion = d['effective_cost'] * demand_twh / 1000.0
                # 25-year NPV ($B) at 5% real WACC
                # Annuity factor = (1 - (1+r)^-n) / r
                _r, _n = 0.05, 25
                annuity_factor = (1 - (1 + _r) ** -_n) / _r  # ~14.09
                npv_25yr_billion = annual_cost_billion * annuity_factor
                iso_traj.append({
                    'threshold': t,
                    'year': SBTI_YEAR_MAP.get(t, 2050),
                    'demand_twh': round(demand_twh, 1),
                    'effective_cost': d['effective_cost'],
                    'total_cost': d['total_cost'],
                    'incremental': d['incremental'],
                    'annual_cost_billion': round(annual_cost_billion, 2),
                    'npv_25yr_billion': round(npv_25yr_billion, 1),
                    'resource_twh': d['resource_twh'],
                    'battery_twh': d.get('battery_twh', 0) + d.get('battery8_twh', 0),
                    'ldes_twh': d.get('ldes_twh', 0),
                    'gas_backup_mw': d['gas_backup_mw'],
                    'new_gas_mw': d['new_gas_mw'],
                    'existing_gas_used_mw': d.get('existing_gas_used_mw', 0),
                    'clean_peak_mw': d.get('clean_peak_mw', 0),
                    'firm_total_twh': round(firm_total_twh, 1),
                    'stepwise_mac': stepwise_mac,
                    'blended_new_lcoe': d.get('blended_new_lcoe', 0),
                    'new_gen_twh': d.get('new_gen_twh', 0),
                    'new_build_cost_total': d.get('new_build_cost_total', 0),
                })
            traj[iso] = iso_traj

        if enforce_monotonic:
            for iso in ISOS:
                if iso not in traj:
                    continue
                running_max = {}
                for entry in traj[iso]:
                    for res in list(entry['resource_twh'].keys()):
                        val = entry['resource_twh'][res]
                        if res in running_max:
                            entry['resource_twh'][res] = max(running_max[res], val)
                        running_max[res] = entry['resource_twh'][res]
                    for field in ['battery_twh', 'ldes_twh']:
                        val = entry.get(field, 0)
                        old_max = running_max.get(field, 0)
                        entry[field] = max(old_max, val)
                        running_max[field] = entry[field]
                    clean_peak_mw = 0
                    for r, twh in entry['resource_twh'].items():
                        pcc = PEAK_CAPACITY_CREDITS.get(r, 0)
                        if pcc > 0 and twh > 0:
                            clean_peak_mw += (twh * 1e6 / 8760) * pcc
                    clean_peak_mw += (entry.get('battery_twh', 0) * 1e6 / 8760) * PEAK_CAPACITY_CREDITS.get('battery', 0.95)
                    clean_peak_mw += (entry.get('ldes_twh', 0) * 1e6 / 8760) * PEAK_CAPACITY_CREDITS.get('ldes', 0.90)
                    gf = get_demand_growth_factor(iso, entry['threshold'])
                    ra_peak_mw = PEAK_DEMAND_MW[iso] * gf * (1 + RESOURCE_ADEQUACY_MARGIN)
                    gaf = GAS_AVAILABILITY_FACTOR[iso]
                    gas_needed_mw = max(0, ra_peak_mw - clean_peak_mw) / gaf
                    entry['gas_backup_mw'] = round(gas_needed_mw)
                    entry['new_gas_mw'] = round(max(0, gas_needed_mw - EXISTING_GAS_CAPACITY_MW[iso]))
                    entry['clean_peak_mw'] = round(clean_peak_mw)
                    cf_twh = entry['resource_twh'].get('clean_firm', 0)
                    ccs_twh = entry['resource_twh'].get('ccs_ccgt', 0)
                    existing_cf_twh = GRID_MIX_SHARES[iso].get('clean_firm', 0) / 100.0 * BASE_DEMAND_TWH[iso]
                    entry['firm_total_twh'] = round(max(0, cf_twh - existing_cf_twh) + ccs_twh, 1)
        return traj

    trajectories = {}
    trajectories['pure_consequential'] = _build_trajectory(results_a, enforce_monotonic=True)
    trajectories['hourly_matching'] = _build_trajectory(results_b, enforce_monotonic=False)

    # Build per-target Scenario B trajectories for dashboard slider compatibility
    # With step3-mining approach, all targets produce the same trajectory
    hourly_traj = trajectories['hourly_matching']
    trajectories_b_by_target = {}
    for target_t in THRESHOLDS:
        t_key = str(int(target_t)) if target_t == int(target_t) else str(target_t)
        trajectories_b_by_target[t_key] = hourly_traj

    output = {
        'metadata': {
            'description': 'Dual-scenario comparison: Pure Consequential vs Hourly Matching',
            'scenario_a': {
                'name': SCENARIO_A['name'],
                'description': SCENARIO_A['description'],
                'toggles': SCENARIO_A['toggles'],
                'method': 'ef_forward_step_foak',
                'method_description': 'Pure consequential: greedy forward-stepping through feasible EF mixes. At each threshold, evaluates all EF mixes under static cost assumptions (Low renewables, Med batteries/uprates/tx, High firm/CCS/LDES/geo). With expensive firm, optimizer picks renewable-heavy mixes. At 90%+, firm is unavoidable at FOAK prices — cost cliff. Floor ratchet locks in prior deployments. Each step is deterministic of the next.',
            },
            'scenario_b': {
                'name': SCENARIO_B['name'],
                'description': SCENARIO_B['description'],
                'toggles': SCENARIO_B['toggles'],
                'method': 'endpoint_backward_learning',
                'method_description': 'Forward-looking endpoint-optimal deployment: finds NOAK-optimal mix at 95% (all costs Low, CCS at Medium+45Q), then works backward with S-curve deployment pacing. Nuclear/LDES: H→L learning curve. CCS: H→M learning curve with 45Q. At each threshold, selects from feasible mixes those meeting firm/CCS/LDES pacing targets. Produces fundamentally different mixes from Scenario A — balanced firm/storage from early thresholds rather than renewable-heavy.',
            },
            'sbti_year_map': {str(k): v for k, v in SBTI_YEAR_MAP.items()},
            'thresholds': THRESHOLDS,
            'isos': ISOS,
            'grid_mix_shares': {iso: dict(GRID_MIX_SHARES[iso]) for iso in ISOS},
            'base_demand_twh': dict(BASE_DEMAND_TWH),
            'demand_growth_rates': DEMAND_GROWTH_RATES,
            'demand_growth_level': 'Medium',
        },
        'queue_a': queue_a,
        'queue_b': queue_b,
        'trajectories': trajectories,
        'trajectories_b_by_target': trajectories_b_by_target,
        'stranding_a': stranding_a,
        'stranding_b': stranding_b,
    }

    out_json = 'data/step5-post-processing/scenario_comparison.json'
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nJSON: {out_json} ({os.path.getsize(out_json) / 1024:.0f} KB)")

    out_js = 'dashboard/js/scenario-comparison-data.js'
    os.makedirs(os.path.dirname(out_js), exist_ok=True)
    with open(out_js, 'w') as f:
        f.write("// Auto-generated by step6_scenario_comparison.py\n")
        f.write("// Dual-scenario comparison: Pure Consequential vs Hourly Matching\n")
        f.write("// A: Pure Consequential  B: Hourly Matching\n\n")
        f.write(f"const SCENARIO_COMPARISON = {json.dumps(output, indent=2, default=str)};\n")
    print(f"JS:   {out_js} ({os.path.getsize(out_js) / 1024:.0f} KB)")

    print("\nDone.")


if __name__ == '__main__':
    main()
