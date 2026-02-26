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

ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']
THRESHOLDS = [50, 55, 60, 65, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 99.5, 99.9, 100]
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
    'battery': {
        'Low':    {'CAISO': 77, 'ERCOT': 69, 'PJM': 74, 'NYISO': 81, 'NEISO': 79, 'MISO': 72, 'SPP': 70},
        'Medium': {'CAISO': 102, 'ERCOT': 92, 'PJM': 98, 'NYISO': 108, 'NEISO': 105, 'MISO': 96, 'SPP': 93},
        'High':   {'CAISO': 133, 'ERCOT': 120, 'PJM': 127, 'NYISO': 140, 'NEISO': 137, 'MISO': 125, 'SPP': 121},
    },
    'battery8': {
        'Low':    {'CAISO': 85, 'ERCOT': 77, 'PJM': 82, 'NYISO': 90, 'NEISO': 88, 'MISO': 80, 'SPP': 78},
        'Medium': {'CAISO': 125, 'ERCOT': 113, 'PJM': 120, 'NYISO': 132, 'NEISO': 129, 'MISO': 117, 'SPP': 115},
        'High':   {'CAISO': 165, 'ERCOT': 149, 'PJM': 159, 'NYISO': 175, 'NEISO': 170, 'MISO': 155, 'SPP': 152},
    },
    'ldes': {
        'Low':    {'CAISO': 135, 'ERCOT': 116, 'PJM': 128, 'NYISO': 150, 'NEISO': 143, 'MISO': 122, 'SPP': 118},
        'Medium': {'CAISO': 180, 'ERCOT': 155, 'PJM': 170, 'NYISO': 200, 'NEISO': 190, 'MISO': 162, 'SPP': 158},
        'High':   {'CAISO': 234, 'ERCOT': 202, 'PJM': 221, 'NYISO': 260, 'NEISO': 247, 'MISO': 211, 'SPP': 205},
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
    'battery':    {'None': 0, 'Low': {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 1, 'NEISO': 1, 'MISO': 0, 'SPP': 0},
                   'Medium': {'CAISO': 1, 'ERCOT': 1, 'PJM': 1, 'NYISO': 2, 'NEISO': 2, 'MISO': 1, 'SPP': 1},
                   'High': {'CAISO': 2, 'ERCOT': 2, 'PJM': 3, 'NYISO': 4, 'NEISO': 3, 'MISO': 2, 'SPP': 2}},
    'battery8':   {'None': 0, 'Low': {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 1, 'NEISO': 1, 'MISO': 0, 'SPP': 0},
                   'Medium': {'CAISO': 1, 'ERCOT': 1, 'PJM': 1, 'NYISO': 2, 'NEISO': 2, 'MISO': 1, 'SPP': 1},
                   'High': {'CAISO': 2, 'ERCOT': 2, 'PJM': 3, 'NYISO': 4, 'NEISO': 3, 'MISO': 2, 'SPP': 2}},
    'ldes':       {'None': 0, 'Low': {'CAISO': 1, 'ERCOT': 1, 'PJM': 1, 'NYISO': 2, 'NEISO': 2, 'MISO': 1, 'SPP': 1},
                   'Medium': {'CAISO': 2, 'ERCOT': 2, 'PJM': 3, 'NYISO': 4, 'NEISO': 3, 'MISO': 2, 'SPP': 2},
                   'High': {'CAISO': 4, 'ERCOT': 3, 'PJM': 5, 'NYISO': 7, 'NEISO': 6, 'MISO': 4, 'SPP': 3}},
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
# Early commitment to clean firm deployment at FOAK prices in the late 2020s /
# early-to-mid 2030s. Pays premium FOAK costs upfront but drives the learning
# curve (Wright's Law) — by the 2040s firm costs decline to NOAK as cumulative
# deployment drives cost reduction. Overall system cost is LOWER because early
# investment enables cheaper deployment at scale when it matters most.
# Uprates are Medium: existing nuclear uprates don't depend on technology learning.
SCENARIO_B = {
    'name': 'Hourly Matching',
    'short': 'hourly_matching',
    'description': 'Early clean firm investment at FOAK prices drives learning curve — higher upfront cost but NOAK pricing by 2040s makes overall system cost lower',
    'toggles': {
        'ren': 'L',        # Low renewable (mature tech, already cheap)
        'firm': 'H',       # High firm gen (FOAK at start, learning curve drives to NOAK)
        'batt': 'L',       # Low battery (mature tech, already down learning curve)
        'ldes_lvl': 'H',   # High LDES (FOAK → NOAK via learning)
        'fuel': 'M',       # Medium fossil fuel
        'tx': 'M',         # Medium transmission
        'ccs': 'H',        # High CCS (FOAK → NOAK via learning)
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

    Returns dict: {threshold_str: [[cf%, sol%, wnd%, ccs%, hyd%, proc%, match%, bat4%, bat8%, ldes%], ...]}
    """
    feasible_path = os.path.join(step3_dir, f'step3_feasible_{iso}.parquet')
    if not os.path.exists(feasible_path):
        return {}

    df = pd.read_parquet(feasible_path)
    mix_fields = ['clean_firm', 'solar', 'wind', 'ccs_ccgt', 'hydro',
                  'procurement_pct', 'hourly_match_score',
                  'battery_dispatch_pct', 'battery8_dispatch_pct', 'ldes_dispatch_pct']
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
                row['ccs_ccgt'], row['hydro'], row['procurement_pct'],
                round(row['hourly_match_score'], 1),
                row['battery_dispatch_pct'], row['battery8_dispatch_pct'],
                row['ldes_dispatch_pct'],
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

    # mixes[iso][threshold_str] = list of [cf%, sol%, wnd%, ccs%, hyd%, proc%, match%, bat4%, bat8%, ldes%]
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

    mix: [cf%, sol%, wnd%, ccs%, hyd%, proc%, match%, bat4%, bat8%, ldes%]
    overrides: optional dict with explicit LCOE values (bypasses toggle lookups):
        nuclear_lcoe, ccs_lcoe, geo_lcoe, ldes_lcoe, uprate_lcoe
    growth_factor: demand growth multiplier (>1 means demand has grown from base year).
        Existing resource percentages are scaled DOWN by this factor so that existing
        absolute TWh stays fixed as demand grows.
    Returns: dict with cost details + gas backup
    """
    cf_pct, sol_pct, wnd_pct, ccs_pct, hyd_pct = mix[0], mix[1], mix[2], mix[3], mix[4]
    proc_pct, match_score = mix[5], mix[6]
    bat4_pct, bat8_pct, ldes_pct = mix[7], mix[8], mix[9]

    proc = proc_pct / 100.0
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

    # Demand-weighted percentages
    sol_demand = proc * sol_pct
    wnd_demand = proc * wnd_pct
    ccs_demand = proc * ccs_pct
    cf_demand = proc * cf_pct
    # Hydro is existing-only: cap at 2025 absolute TWh.
    # Convert mix percentage to TWh, cap, then back to demand-weighted pct points.
    hydro_twh_raw = proc * hyd_pct / 100.0 * demand_twh
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
        proc * cf_pct / 100 * avg_demand_mw * PEAK_CAPACITY_CREDITS['clean_firm'] +
        proc * sol_pct / 100 * avg_demand_mw * PEAK_CAPACITY_CREDITS['solar'] +
        proc * wnd_pct / 100 * avg_demand_mw * PEAK_CAPACITY_CREDITS['wind'] +
        proc * ccs_pct / 100 * avg_demand_mw * PEAK_CAPACITY_CREDITS['ccs_ccgt'] +
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
            resource_twh[res] = pct / 100.0 * proc * demand_twh

    return {
        'total_cost': round(total_cost, 2),
        'effective_cost': round(eff_cost, 2),
        'incremental': round(incremental, 2),
        'wholesale': wholesale,
        'resource_twh': resource_twh,
        'resource_pct': {'clean_firm': cf_pct, 'solar': sol_pct, 'wind': wnd_pct,
                         'ccs_ccgt': ccs_pct, 'hydro': hyd_pct},
        'procurement_pct': proc_pct,
        'match_score': match_score,
        'battery_twh': bat4_pct / 100.0 * demand_twh,
        'battery8_twh': bat8_pct / 100.0 * demand_twh,
        'ldes_twh': ldes_pct / 100.0 * demand_twh,
        'gas_backup_mw': round(gas_needed_mw),
        'new_gas_mw': round(new_gas_mw),
        'existing_gas_used_mw': round(existing_gas_used_mw),
        'clean_peak_mw': round(clean_peak_mw),
        'new_build_cost_total': new_build_cost_total,
        'new_gen_twh': round(new_gen_twh, 3),
        'blended_new_lcoe': round(blended_new_lcoe, 2),
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

            best_cost = float('inf')
            best_result = None
            best_mix = None

            for mix in mixes:
                result = compute_mix_cost(mix, iso_sens, iso, demand_twh_map[iso])
                if result['effective_cost'] < best_cost:
                    best_cost = result['effective_cost']
                    best_result = result
                    best_mix = mix

            if best_result and best_mix:
                # Preserve dispatch parameters for dispatch cache lookup
                best_result['battery_dispatch_pct'] = float(best_mix[7])
                best_result['battery8_dispatch_pct'] = float(best_mix[8])
                best_result['ldes_dispatch_pct'] = float(best_mix[9])
                best_result['demand_mwh'] = demand_twh_map[iso] * 1e6
                iso_results[t] = best_result

        results[iso] = iso_results
    return results


def _mix_resource_twh(mix, demand_twh, iso=None):
    """Extract deployed resource TWh from a raw mix vector for floor comparison.

    mix = [cf%, sol%, wnd%, ccs%, hyd%, proc%, match%, bat4%, bat8%, ldes%]
    iso: required — used to cap hydro at 2025 absolute TWh.
    Returns dict of resource → TWh deployed.
    """
    cf, sol, wnd, ccs, hyd, proc_pct = mix[0], mix[1], mix[2], mix[3], mix[4], mix[5]
    bat4, bat8, ldes = mix[7], mix[8], mix[9]
    proc = proc_pct / 100.0
    # Hydro is existing-only: cap at 2025 absolute TWh regardless of demand growth
    hydro_twh = min(proc * hyd / 100.0 * demand_twh,
                    HYDRO_CAP_TWH[iso]) if iso else proc * hyd / 100.0 * demand_twh
    return {
        'clean_firm': proc * cf / 100.0 * demand_twh,
        'solar':      proc * sol / 100.0 * demand_twh,
        'wind':       proc * wnd / 100.0 * demand_twh,
        'ccs_ccgt':   proc * ccs / 100.0 * demand_twh,
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

        best_95_cost = float('inf')
        best_95_mix = None
        for mix in t95_mixes:
            result = compute_mix_cost(mix, iso_sens, iso, demand_95, growth_factor=gf_95)
            if result['effective_cost'] < best_95_cost:
                best_95_cost = result['effective_cost']
                best_95_mix = mix

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

            if target_95_deployed:
                def _within_target(mix):
                    deployed = _mix_resource_twh(mix, demand_twh, iso)
                    for res in ['clean_firm', 'solar', 'wind', 'ccs_ccgt']:
                        cap = target_95_deployed.get(res, 0) * demand_ratio * slack
                        if cap > 1.0 and deployed.get(res, 0) > cap:
                            return False
                    return True

                target_passing = [m for m in mixes if _within_target(m)]
            else:
                target_passing = list(mixes)

            # Fall back to all mixes if target filter is too restrictive
            if not target_passing:
                target_passing = list(mixes)

            # 2b. Score ALL mixes by augmented cost (EF cost + floor excess penalty).
            #     This ensures we pick the mix that maximizes use of locked-in resources
            #     from prior steps, not just the cheapest mix in isolation.
            best_augmented_cost = float('inf')
            best_result = None
            best_mix = None
            best_excess_twh = None
            best_excess_per_mwh = 0

            for mix in target_passing:
                result = compute_mix_cost(mix, iso_sens, iso, demand_twh, growth_factor=gf)

                # Compute augmented cost: how expensive is this mix INCLUDING floor excess?
                ef_deployed = _mix_resource_twh(mix, demand_twh, iso)
                mix_excess_per_mwh = 0.0
                for res in floor:
                    excess = max(0, floor.get(res, 0) - ef_deployed.get(res, 0))
                    if excess > 0.01:
                        lcoe = _resource_new_build_lcoe(res, iso_sens, iso)
                        mix_excess_per_mwh += excess / demand_twh * lcoe

                augmented_eff = (result['total_cost'] + mix_excess_per_mwh) / (result['match_score'] / 100) if result['match_score'] > 0 else float('inf')

                if augmented_eff < best_augmented_cost:
                    best_augmented_cost = augmented_eff
                    best_result = result
                    best_mix = mix
                    best_excess_per_mwh = mix_excess_per_mwh

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
            augmented_result['battery_dispatch_pct'] = float(best_mix[7])
            augmented_result['battery8_dispatch_pct'] = float(best_mix[8])
            augmented_result['ldes_dispatch_pct'] = float(best_mix[9])
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


def _forward_step_optimization(feasible_mixes, sens, get_overrides_fn, label):
    """Core forward-stepping optimizer with floor ratchet.

    Evaluates ALL feasible EF mixes at each threshold under scenario-specific
    cost assumptions. Selects the cheapest augmented effective cost. Floor
    ratchets prevent un-building prior resource commitments.

    This is the key function that produces DIFFERENT resource mixes for
    Scenario A vs B — because different cost assumptions (via get_overrides_fn)
    make different mixes optimal at each step.

    Args:
        feasible_mixes: from parse_feasible_mixes()
        sens: sensitivity toggle dict (defines base cost lookups)
        get_overrides_fn: fn(iso, threshold) → dict of LCOE overrides
            Scenario A: static FOAK (returns only uprate override)
            Scenario B: learning-curve interpolated FOAK→NOAK prices
        label: 'A' or 'B' for logging

    Returns: dict[iso][threshold] → result dict (same format as old step3-based approach)
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
        floor = dict(existing_twh)

        iso_results = {}

        for t in THRESHOLDS:
            t_str = str(int(t)) if t == int(t) else str(t)
            mixes = feasible_mixes.get(iso, {}).get(t_str, [])
            if not mixes:
                continue

            gf = get_demand_growth_factor(iso, t)
            demand_twh = base_demand * gf
            demand_mwh = demand_twh * 1e6

            overrides = get_overrides_fn(iso, t)

            # Evaluate ALL feasible mixes under this scenario's cost assumptions
            best_augmented_eff = float('inf')
            best_result = None
            best_mix = None
            best_excess_per_mwh = 0.0

            for mix in mixes:
                result = compute_mix_cost(mix, iso_sens, iso, demand_twh,
                                         overrides=overrides, growth_factor=gf)

                deployed = _mix_resource_twh(mix, demand_twh, iso)

                # Floor excess cost: resources locked in from prior steps that
                # this mix doesn't use. Existing resources are $0; only
                # committed-new resources above existing get newbuild pricing.
                excess_per_mwh = 0.0
                for res in floor:
                    excess = max(0, floor.get(res, 0) - deployed.get(res, 0))
                    if excess > 0.01:
                        # Split excess into existing ($0) and committed-new (newbuild)
                        existing_gap = max(0, min(excess,
                                                  existing_twh.get(res, 0) - deployed.get(res, 0)))
                        newbuild_gap = excess - max(0, existing_gap)
                        if newbuild_gap > 0.01:
                            lcoe = _get_excess_lcoe(res, iso_sens, iso, overrides)
                            excess_per_mwh += newbuild_gap / demand_twh * lcoe

                augmented_eff = (result['total_cost'] + excess_per_mwh) / \
                    (result['match_score'] / 100.0) if result['match_score'] > 0 else float('inf')

                if augmented_eff < best_augmented_eff:
                    best_augmented_eff = augmented_eff
                    best_result = result
                    best_mix = mix
                    best_excess_per_mwh = excess_per_mwh

            if not best_result or not best_mix:
                continue

            # Build augmented result: resources = max(floor, EF deployed)
            deployed = _mix_resource_twh(best_mix, demand_twh, iso)
            augmented = {}
            excess_twh = {}
            for res in floor:
                ef_val = deployed.get(res, 0)
                floor_val = floor.get(res, 0)
                augmented[res] = max(ef_val, floor_val)
                excess_twh[res] = max(0, floor_val - ef_val)

            total_excess = sum(excess_twh.values())

            # Build result dict
            aug = dict(best_result)
            aug['total_cost'] = round(best_result['total_cost'] + best_excess_per_mwh, 2)
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
                sum(v for k, v in excess_twh.items() if k != 'hydro'), 3)

            if total_excess > 1.0:
                print(f"  ↗ {iso} {t}% [{label}]: {total_excess:.0f} TWh excess "
                      f"(+${best_excess_per_mwh:.1f}/MWh)")

            iso_results[t] = aug
            floor = dict(augmented)

        # Print trajectory summary
        for t in sorted(iso_results.keys()):
            r = iso_results[t]
            rt = r['resource_twh']
            print(f"    {t:5.1f}%: CF={rt['clean_firm']:7.0f} Sol={rt['solar']:6.0f} "
                  f"Wnd={rt['wind']:6.0f} CCS={rt.get('ccs_ccgt', 0):6.0f} "
                  f"Proc={r['procurement_pct']:3.0f}% "
                  f"${r['effective_cost']:.0f}/MWh [{label}]")

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

    At each threshold, compute learning_fraction(scenario='B') → interpolate
    all firm/CCS/LDES costs between High (FOAK) and Low (NOAK). Uprate always Medium.
    Step 3 costs were computed at full FOAK (firm='H', ldes='H', ccs='H').
    We compute the cost delta from FOAK to the learning-curve-adjusted price
    for each resource tranche and apply it.

    10-year learning period (2029-2038), concave ramp (exponent 0.6).
    Sources: INL SOAK data, NEA learning ranges, DOE Liftoff.
    """
    for iso in results:
        for t in results[iso]:
            r = results[iso][t]
            demand_twh = r['demand_twh']
            if demand_twh <= 0:
                continue

            frac = learning_fraction(t, scenario='B')
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
            # CCS TWh from mix minus existing
            existing_ccs_frac = GRID_MIX_SHARES[iso].get('ccs_ccgt', 0) / 100.0
            gf = demand_twh / BASE_DEMAND_TWH[iso]
            existing_ccs_twh = existing_ccs_frac * BASE_DEMAND_TWH[iso]
            proc = r['procurement_pct'] / 100.0
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
    s['proc'] = s['procurement_pct'] / 100.0

    # Resource TWh columns
    s['res_cf_twh'] = s['proc'] * s['mix_clean_firm'] / 100.0 * s['demand_twh']
    s['res_sol_twh'] = s['proc'] * s['mix_solar'] / 100.0 * s['demand_twh']
    s['res_wnd_twh'] = s['proc'] * s['mix_wind'] / 100.0 * s['demand_twh']
    s['res_ccs_twh'] = s['proc'] * s['mix_ccs_ccgt'] / 100.0 * s['demand_twh']
    s['res_hyd_twh'] = s['proc'] * s['mix_hydro'] / 100.0 * s['demand_twh']
    s['bat_twh'] = s['battery_dispatch_pct'] / 100.0 * s['demand_twh']
    s['bat8_twh'] = s['battery8_dispatch_pct'] / 100.0 * s['demand_twh']
    s['ldes_twh_calc'] = s['ldes_dispatch_pct'] / 100.0 * s['demand_twh']

    # New-build cost tracking (vectorized)
    existing_pct = GRID_MIX_SHARES[iso]
    base_twh = BASE_DEMAND_TWH[iso]
    s['gf'] = s['demand_twh'] / base_twh
    # Existing percentages scaled by demand growth
    ex_cf = existing_pct.get('clean_firm', 0)
    ex_sol = existing_pct.get('solar', 0)
    ex_wnd = existing_pct.get('wind', 0)
    ex_ccs = existing_pct.get('ccs_ccgt', 0)

    s['cf_new_pct'] = (s['proc'] * s['mix_clean_firm'] - ex_cf / s['gf']).clip(lower=0)
    s['sol_new_pct'] = (s['proc'] * s['mix_solar'] - ex_sol / s['gf']).clip(lower=0)
    s['wnd_new_pct'] = (s['proc'] * s['mix_wind'] - ex_wnd / s['gf']).clip(lower=0)
    s['ccs_new_pct'] = (s['proc'] * s['mix_ccs_ccgt'] * 100 - ex_ccs / s['gf']).clip(lower=0)
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
                        row['mix_ccs_ccgt'], row['mix_hydro'], row['procurement_pct'],
                        row['hourly_match_score'], row['battery_dispatch_pct'],
                        row['battery8_dispatch_pct'], row['ldes_dispatch_pct']],
            'demand_twh': row['demand_twh'],
            'effective_cost': row['cost_effective_cost'],
            'total_cost': row['cost_total_cost'],
            'incremental': row['cost_incremental'],
            'wholesale': row['cost_wholesale'],
            'match_score': row['hourly_match_score'],
            'procurement_pct': row['procurement_pct'],
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


def find_scenario_b_mixes(feasible_mixes):
    """Scenario B: Endpoint-deterministic with paced clean firm investment.

    Strategy (fundamentally different from Scenario A):
      1. Find the NOAK-optimal mix at the 95% target — this is the "north star"
         resource composition that the grid is building toward.
      2. Compute a paced firm investment ramp from existing levels to the 95% target.
         At each threshold T, firm deployment should be at least:
           existing_firm + (T - 50)/(95 - 50) * (target_firm - existing_firm)
      3. At each threshold, evaluate all feasible EF mixes. Prefer those that meet
         the pacing requirement (invest enough in firm clean). Among pacing-compliant
         mixes, pick the cheapest with learning-curve-adjusted prices.
      4. Floor ratchet locks in prior commitments.
      5. For thresholds > 95%, continue building on the 95% foundation.

    This produces a fundamentally different trajectory from Scenario A because:
      - At 50-75%, Scenario A picks renewable-heavy mixes (cheapest). Scenario B
        intentionally picks mixes with more firm clean (pacing requirement).
      - At 80-90%, the divergence compounds: B has been building firm, driving
        learning curve down. A has been deferring firm.
      - At 95%+, B benefits from NOAK firm prices. A faces FOAK cost cliff.

    Cost: Low renewables, Medium batteries/uprates/tx, FOAK→NOAK firm/CCS/LDES/geo.
    """
    results = {}

    # For finding the 95% target, use NOAK (Low) prices — this is what the grid
    # looks like when learning curve investments have paid off
    NOAK_SENS = {
        'ren': 'L', 'firm': 'L', 'batt': 'L', 'ldes_lvl': 'L',
        'fuel': 'M', 'tx': 'M', 'ccs': 'L', 'q45': '1', 'geo': 'L',
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
        # This tells us what resources the grid needs at deep decarbonization
        # when firm costs have reached NOAK via learning curve investment.
        # ==================================================================
        gf_95 = get_demand_growth_factor(iso, 95)
        demand_95 = base_demand * gf_95
        mixes_95 = feasible_mixes.get(iso, {}).get('95', [])

        best_cost_95 = float('inf')
        best_mix_95 = None
        for mix in mixes_95:
            result = compute_mix_cost(mix, noak_sens, iso, demand_95,
                                     growth_factor=gf_95)
            if result['effective_cost'] < best_cost_95:
                best_cost_95 = result['effective_cost']
                best_mix_95 = mix

        if not best_mix_95:
            print(f"  ⚠ {iso}: No feasible mixes at 95%, skipping")
            results[iso] = {}
            continue

        target_deployed = _mix_resource_twh(best_mix_95, demand_95, iso)

        # Firm resources at target (nuclear + CCS — the resources needing learning curve)
        target_cf_twh = target_deployed.get('clean_firm', 0)
        target_ccs_twh = target_deployed.get('ccs_ccgt', 0)
        target_firm_twh = target_cf_twh + target_ccs_twh
        target_ldes_twh = target_deployed.get('ldes', 0)

        # Existing firm baseline (2025 levels)
        existing_cf_twh = existing.get('clean_firm', 0) / 100.0 * base_demand
        existing_ccs_twh = existing.get('ccs_ccgt', 0) / 100.0 * base_demand
        existing_firm_twh = existing_cf_twh + existing_ccs_twh

        print(f"\n  {iso} 95% NOAK target: firm={target_firm_twh:.0f} TWh "
              f"(CF={target_cf_twh:.0f}, CCS={target_ccs_twh:.0f}), "
              f"LDES={target_ldes_twh:.0f} TWh, "
              f"existing firm={existing_firm_twh:.0f} TWh")

        # ==================================================================
        # Step 2: Forward-step with paced firm investment toward 95% target
        # ==================================================================
        # Existing resource levels (priced at $0)
        existing_twh_b = {
            'clean_firm': existing.get('clean_firm', 0) / 100.0 * base_demand,
            'solar': existing.get('solar', 0) / 100.0 * base_demand,
            'wind': existing.get('wind', 0) / 100.0 * base_demand,
            'ccs_ccgt': existing.get('ccs_ccgt', 0) / 100.0 * base_demand,
            'hydro': existing.get('hydro', 0) / 100.0 * base_demand,
            'battery': 0, 'ldes': 0,
        }
        floor = dict(existing_twh_b)

        iso_results = {}

        for t in THRESHOLDS:
            t_str = str(int(t)) if t == int(t) else str(t)
            mixes = feasible_mixes.get(iso, {}).get(t_str, [])
            if not mixes:
                continue

            gf = get_demand_growth_factor(iso, t)
            demand_twh = base_demand * gf
            demand_mwh = demand_twh * 1e6

            # Learning-curve overrides at this threshold (Scenario B)
            frac = learning_fraction(t, scenario='B')
            overrides = _build_learning_overrides(iso, frac)

            # Paced firm investment floor: linear ramp from existing to 95% target.
            # Scale target by demand growth ratio (firm TWh grows with demand).
            t_frac = min(1.0, max(0, (t - 50) / (95 - 50)))
            demand_ratio = demand_twh / demand_95 if demand_95 > 0 else 1.0
            paced_firm_twh = existing_firm_twh + t_frac * (
                target_firm_twh * demand_ratio - existing_firm_twh)

            # Evaluate ALL mixes. Prefer those meeting firm pacing requirement.
            # Among pacing-compliant mixes, pick cheapest with learning-curve prices.
            best_paced_eff = float('inf')
            best_paced_result = None
            best_paced_mix = None
            best_paced_excess = 0.0

            best_any_eff = float('inf')
            best_any_result = None
            best_any_mix = None
            best_any_excess = 0.0

            for mix in mixes:
                result = compute_mix_cost(mix, iso_sens, iso, demand_twh,
                                         overrides=overrides, growth_factor=gf)

                deployed = _mix_resource_twh(mix, demand_twh, iso)

                # Floor excess cost: existing resources at $0, only committed-new
                # resources above existing get newbuild pricing.
                excess_per_mwh = 0.0
                for res in floor:
                    excess = max(0, floor.get(res, 0) - deployed.get(res, 0))
                    if excess > 0.01:
                        existing_gap = max(0, min(excess,
                                                  existing_twh_b.get(res, 0) - deployed.get(res, 0)))
                        newbuild_gap = excess - max(0, existing_gap)
                        if newbuild_gap > 0.01:
                            lcoe = _get_excess_lcoe(res, iso_sens, iso, overrides)
                            excess_per_mwh += newbuild_gap / demand_twh * lcoe

                augmented_eff = (result['total_cost'] + excess_per_mwh) / \
                    (result['match_score'] / 100.0) if result['match_score'] > 0 else float('inf')

                # Check pacing: does this mix invest enough in firm?
                mix_firm_twh = deployed.get('clean_firm', 0) + deployed.get('ccs_ccgt', 0)
                meets_pace = mix_firm_twh >= paced_firm_twh * 0.7  # 70% tolerance

                if meets_pace and augmented_eff < best_paced_eff:
                    best_paced_eff = augmented_eff
                    best_paced_result = result
                    best_paced_mix = mix
                    best_paced_excess = excess_per_mwh

                if augmented_eff < best_any_eff:
                    best_any_eff = augmented_eff
                    best_any_result = result
                    best_any_mix = mix
                    best_any_excess = excess_per_mwh

            # Prefer pacing-compliant mix; fall back to any if none qualify
            if best_paced_result:
                sel_result = best_paced_result
                sel_mix = best_paced_mix
                sel_excess = best_paced_excess
                paced_flag = ''
            elif best_any_result:
                sel_result = best_any_result
                sel_mix = best_any_mix
                sel_excess = best_any_excess
                paced_flag = ' (no pace-compliant mix)'
            else:
                continue

            # Build augmented result
            deployed = _mix_resource_twh(sel_mix, demand_twh, iso)
            augmented = {}
            excess_twh = {}
            for res in floor:
                ef_val = deployed.get(res, 0)
                floor_val = floor.get(res, 0)
                augmented[res] = max(ef_val, floor_val)
                excess_twh[res] = max(0, floor_val - ef_val)

            total_excess = sum(excess_twh.values())

            aug = dict(sel_result)
            aug['total_cost'] = round(sel_result['total_cost'] + sel_excess, 2)
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

            if total_excess > 1.0:
                print(f"  ↗ {iso} {t}% [B]: {total_excess:.0f} TWh excess "
                      f"(+${sel_excess:.1f}/MWh)")

            iso_results[t] = aug
            floor = dict(augmented)

        # Print trajectory summary
        for t in sorted(iso_results.keys()):
            r = iso_results[t]
            rt = r['resource_twh']
            lf = r.get('learning_fraction', 0)
            paced = r.get('paced_firm_target_twh', 0)
            actual_firm = rt.get('clean_firm', 0) + rt.get('ccs_ccgt', 0)
            print(f"    {t:5.1f}%: CF={rt['clean_firm']:7.0f} Sol={rt['solar']:6.0f} "
                  f"Wnd={rt['wind']:6.0f} CCS={rt.get('ccs_ccgt', 0):6.0f} "
                  f"Firm={actual_firm:6.0f}/{paced:5.0f} pace "
                  f"LF={lf:.2f} ${r['effective_cost']:.0f}/MWh [B]")

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

    mix_result: dict from compute_mix_cost() with resource_pct, procurement_pct, etc.
    Returns compute_co2_from_dispatch() result or None.
    """
    resource_pcts = mix_result.get('resource_pct', {})
    if not resource_pcts:
        return None

    proc_pct = mix_result.get('procurement_pct', 100)
    bat_pct = mix_result.get('battery_dispatch_pct', 0)
    bat8_pct = mix_result.get('battery8_dispatch_pct', 0)
    ldes_pct = mix_result.get('ldes_dispatch_pct', 0)

    supply_profiles = get_supply_profiles(iso, gen_profiles)
    demand_norm, _ = get_demand_profile(iso, demand_data)
    cache = dispatch_caches.get(iso, {})

    dispatch_result, _ = get_or_compute_dispatch(
        iso, demand_norm, supply_profiles, resource_pcts,
        proc_pct, bat_pct, bat8_pct, ldes_pct,
        cache=cache)

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

            # MAC = delta_cost / delta_co2 (direct, no intermediate LCOE/rate split)
            delta_new_cost = end['new_build_cost_total'] - start['new_build_cost_total']

            # co2_displaced_mt already computed above from dispatch or analytical model
            if co2_displaced_mt > 0.001 and delta_new_cost > 0:
                marginal_mac = delta_new_cost / (co2_displaced_mt * 1e6)
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
    print("  B: Hourly Matching — FOAK→NOAK learning curve, early firm investment")
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
    print("\nScenario B (Hourly Matching): Forward-stepping, FOAK→NOAK learning...")
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

    # Build resource trajectories with per-threshold stepwise MAC
    def _build_trajectory(results, enforce_monotonic=False):
        """Build trajectory dicts from optimization results."""
        results_id = id(results)
        traj = {}
        for iso in ISOS:
            iso_traj = []
            base_demand_twh = BASE_DEMAND_TWH[iso]
            prev_t = None
            for t in THRESHOLDS:
                d = results.get(iso, {}).get(t, {})
                if not d:
                    continue
                demand_twh = get_demand_twh(iso, t)
                stepwise_mac = None
                if prev_t is not None and prev_t in results.get(iso, {}):
                    prev_d = results[iso][prev_t]
                    delta_new_cost = d['new_build_cost_total'] - prev_d['new_build_cost_total']

                    # MAC = delta_cost / delta_co2 (direct, no LCOE/rate split)
                    co2_cur = _traj_co2_cache.get((iso, t, results_id))
                    co2_prev = _traj_co2_cache.get((iso, prev_t, results_id))

                    if co2_cur and co2_prev:
                        delta_co2_tons = (co2_cur['total_co2_abated_tons']
                                          - co2_prev['total_co2_abated_tons'])
                    else:
                        # Fallback: estimate from emission rate × new generation delta
                        delta_new_gen = d['new_gen_twh'] - prev_d['new_gen_twh']
                        if co2_cur:
                            rate_cur = co2_cur['weighted_emission_rate']
                        else:
                            rate_cur, _ = cfr_cached(iso, t, egrid, fossil_mix)
                        delta_co2_tons = (delta_new_gen * 1e6 * rate_cur
                                          if rate_cur > 0 else 0)

                    if delta_co2_tons > 0 and delta_new_cost > 0:
                        stepwise_mac = round(delta_new_cost / delta_co2_tons, 1)
                    else:
                        stepwise_mac = 9999
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
                    'procurement_pct': d.get('procurement_pct', 100),
                    'stepwise_mac': stepwise_mac,
                    'blended_new_lcoe': d.get('blended_new_lcoe', 0),
                    'new_gen_twh': d.get('new_gen_twh', 0),
                    'new_build_cost_total': d.get('new_build_cost_total', 0),
                })
                prev_t = t
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
                'method': 'ef_forward_step_learning',
                'method_description': 'Hourly matching with FOAK→NOAK learning curve: greedy forward-stepping through feasible EF mixes with declining firm costs. At each threshold, firm/CCS/LDES costs interpolate from FOAK→NOAK via Wright\'s Law. Early investment in firm (even at FOAK) drives the learning curve down. By 2040s, firm at competitive NOAK levels. Produces more balanced mixes with less stranding than Scenario A.',
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
