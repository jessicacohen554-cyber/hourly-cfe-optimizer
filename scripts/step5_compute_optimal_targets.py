#!/usr/bin/env python3
"""
step5_compute_optimal_targets.py — Optimal CFE targets + no-regrets resource investments

PURPOSE:
  1. For each ISO, find the threshold range where grid decarbonization becomes
     more expensive than DAC. Range = 3 grid costs × 3 DAC scenarios.
  2. Within that range, identify "no regrets" resource investments that appear
     across ALL thresholds — the minimum investment floor regardless of where
     the optimal target lands.
  3. Scale resource investments by L/M/H demand growth to show absolute
     quantities under different growth scenarios.

METHODOLOGY:
  Option B — Smooth Marginal MAC from Cost Frontier
    1. At each threshold, take the independently-optimized cheapest system.
    2. Total cost premium and CO₂ abated form monotonic curves vs. threshold.
    3. Monotone cubic splines (PCHIP) interpolate between discrete thresholds.
    4. Marginal MAC = d(TotalCost)/d(CO₂) — the slope of the cost frontier.
    5. Cross 3 grid cost scenarios × 3 DAC scenarios = 9 crossover points.
    6. Range = [min crossover, max crossover].

  Note on Demand Growth & MAC:
    Marginal MAC ($/tCO₂) is scale-invariant: both d(cost) and d(CO₂) scale
    linearly with demand, so the ratio is unchanged. Demand growth DOES change:
    - Total investment $M needed
    - Total CO₂ abated (Mt)
    - Absolute resource quantities (TWh, GW)
    This means the crossover % doesn't shift with demand growth, but the
    SCALE of "no regrets" investments does — which is what the user cares about.

  Option A — Target-Specific Analysis Within the Range
    For each discrete threshold inside the crossover range:
    - Resource mix composition at that target
    - Marginal cost of the last step
    - Comparison to DAC at the corresponding SBTi year

  No-Regrets Investment Analysis
    For each resource type, across ALL thresholds within the crossover range:
    - Floor: minimum % share (absolute minimum you'd need regardless)
    - Consensus: resources that are non-zero at every threshold in the range
    - Average: expected investment level across the range
    Scaled by L/M/H demand growth for absolute TWh quantities.

INPUTS:
  - CLEAN_COST (L/M/H) — effective_cost (clean procurement only, NO gas backup)
    Medium: exact from EFFECTIVE_COST_DATA in shared-data.js
    Low/High: SYSTEM_COST(P10/P90) minus gas_backup_cost (approximation)
  - RESOURCE_MIX_DATA from shared-data.js (medium-cost physics optimization)
  - Emission rates, coal/oil caps, demand from dispatch_utils
  - DAC trajectory from SPEC.md
  - Demand growth rates per ISO × L/M/H

OUTPUTS:
  - data/step5-post-processing/optimal_targets.json
  - dashboard/js/optimal-target-data.js
  (Both consumed by step6_generate_shared_data.py for abatement dashboard)
"""

import json
import os
import sys
import numpy as np
from pathlib import Path

# Add scripts/ to path for dispatch_utils import
SCRIPTS_DIR = Path(__file__).parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from scipy.interpolate import PchipInterpolator
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not available — using linear interpolation fallback.")
    print("Install scipy for smooth spline derivatives: pip install scipy")

# Import canonical CO₂ model from dispatch_utils
try:
    from dispatch_utils import (
        compute_fossil_retirement,
        BASE_DEMAND_TWH as DU_DEMAND_TWH,
        GRID_MIX_SHARES as DU_GRID_MIX_SHARES,
        COAL_CAP_TWH as DU_COAL_CAP,
        OIL_CAP_TWH as DU_OIL_CAP,
    )
    HAS_DISPATCH_UTILS = True
    print("Using canonical CO₂ model from dispatch_utils.py")
except ImportError:
    HAS_DISPATCH_UTILS = False
    print("WARNING: dispatch_utils not available — using inline CO₂ model.")

# ============================================================================
# CONSTANTS
# ============================================================================

THRESHOLDS = [50, 55, 60, 65, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 99.5, 99.9, 99.99]
ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']
RESOURCES = ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro', 'battery', 'battery8', 'ldes', 'h2']

WHOLESALE_PRICES = {
    'CAISO': 30, 'ERCOT': 27, 'PJM': 34, 'NYISO': 42,
    'NEISO': 41, 'MISO': 30, 'SPP': 25,
}

DEMAND_TWH = {
    'CAISO': 224.039, 'ERCOT': 488.02, 'PJM': 843.331, 'NYISO': 151.599,
    'NEISO': 115.336, 'MISO': 660.0, 'SPP': 296.0,
}

# Annual demand growth rates per ISO × L/M/H
# Sources: EIA AEO 2024, regional IRP filings, data center projections
# MISO/SPP: uniform 2.0% across tiers (no differentiated forecasts available)
DEMAND_GROWTH_RATES = {
    'CAISO':  {'low': 0.014, 'medium': 0.019, 'high': 0.025},
    'ERCOT':  {'low': 0.020, 'medium': 0.035, 'high': 0.055},
    'PJM':    {'low': 0.015, 'medium': 0.024, 'high': 0.036},
    'NYISO':  {'low': 0.013, 'medium': 0.020, 'high': 0.044},
    'NEISO':  {'low': 0.009, 'medium': 0.018, 'high': 0.029},
    'MISO':   {'low': 0.020, 'medium': 0.020, 'high': 0.020},
    'SPP':    {'low': 0.020, 'medium': 0.020, 'high': 0.020},
}

# Baseline clean energy shares (% of demand, 2025)
GRID_MIX_SHARES = {
    'CAISO': {'clean_firm': 7.9, 'solar': 22.3, 'wind': 8.8, 'offshore_wind': 0, 'ccs_ccgt': 0, 'hydro': 9.5},
    'ERCOT': {'clean_firm': 8.6, 'solar': 13.8, 'wind': 23.6, 'offshore_wind': 0, 'ccs_ccgt': 0, 'hydro': 0.1},
    'PJM':   {'clean_firm': 32.1, 'solar': 2.9, 'wind': 3.8, 'offshore_wind': 0, 'ccs_ccgt': 0, 'hydro': 1.8},
    'NYISO': {'clean_firm': 18.4, 'solar': 0.0, 'wind': 4.7, 'offshore_wind': 0, 'ccs_ccgt': 0, 'hydro': 15.9},
    'NEISO': {'clean_firm': 23.8, 'solar': 1.4, 'wind': 3.9, 'offshore_wind': 0, 'ccs_ccgt': 0, 'hydro': 4.4},
    'MISO':  {'clean_firm': 13.1, 'solar': 2.1, 'wind': 14.5, 'offshore_wind': 0, 'ccs_ccgt': 0, 'hydro': 1.6},
    'SPP':   {'clean_firm': 5.2, 'solar': 0.4, 'wind': 37.1, 'offshore_wind': 0, 'ccs_ccgt': 0, 'hydro': 4.3},
}

# Coal/oil caps (TWh) — maximum fossil generation that can exist in each grid
COAL_CAP_TWH = {
    'CAISO': 0.00, 'ERCOT': 67.58, 'PJM': 139.09, 'NYISO': 0.00,
    'NEISO': 0.31, 'MISO': 125.0, 'SPP': 42.0,
}
OIL_CAP_TWH = {
    'CAISO': 0.60, 'ERCOT': 0.00, 'PJM': 4.59, 'NYISO': 0.15,
    'NEISO': 1.29, 'MISO': 0.50, 'SPP': 0.20,
}
COAL_OIL_RETIREMENT_THRESHOLD = 70.0

# Emission rates (tCO₂/MWh) — fallback if dispatch_utils not available
# Canonical source: data/egrid_emission_rates.json, loaded at runtime via dispatch_utils
EMISSION_RATES_FALLBACK = {
    'CAISO': {'coal': 1133.331 / 2204.62, 'oil': 1802.528 / 2204.62, 'gas': 862.196 / 2204.62},
    'ERCOT': {'coal': 2324.807 / 2204.62, 'oil': 2894.407 / 2204.62, 'gas': 867.401 / 2204.62},
    'PJM':   {'coal': 2216.439 / 2204.62, 'oil': 1918.987 / 2204.62, 'gas': 867.031 / 2204.62},
    'NYISO': {'coal': 0.0,                'oil': 954.721 / 2204.62,  'gas': 914.404 / 2204.62},
    'NEISO': {'coal': 2299.791 / 2204.62, 'oil': 2201.751 / 2204.62, 'gas': 843.608 / 2204.62},
    'MISO':  {'coal': 2280.0 / 2204.62,   'oil': 1900.0 / 2204.62,   'gas': 860.0 / 2204.62},
    'SPP':   {'coal': 2250.0 / 2204.62,   'oil': 1900.0 / 2204.62,   'gas': 865.0 / 2204.62},
}

# Runtime data holders — loaded from canonical data files if available
_EMISSION_RATES_JSON = None  # Raw eGRID JSON (for dispatch_utils interface)
_FOSSIL_MIX_JSON = None      # Raw EIA fossil mix JSON

# SBTi year mapping
THRESHOLD_YEAR_MAP = {
    50: 2030, 55: 2031, 60: 2033, 65: 2034, 70: 2035, 75: 2036,
    80: 2037, 85: 2038, 87.5: 2039, 90: 2040, 92.5: 2043, 95: 2045,
    97.5: 2048, 99: 2049, 99.5: 2049, 99.9: 2050, 100: 2050,
}

# DAC cost trajectories ($/tCO₂, 2024 USD)
DAC_TRAJECTORY = {
    'optimistic':   {2025: 600, 2030: 350, 2035: 230, 2040: 175, 2045: 130, 2050: 100},
    'central':      {2025: 800, 2030: 500, 2035: 375, 2040: 300, 2045: 250, 2050: 200},
    'conservative': {2025: 1100, 2030: 750, 2035: 550, 2040: 450, 2045: 375, 2050: 300},
}

# LEGACY FALLBACK: Clean procurement cost per threshold ($/MWh)
# WARNING: These are hardcoded fallback values only used when parquets are
# unavailable. When parquets are loaded, load_parquet_costs() uses
# cost_total_cost directly (no wholesale offset — Step 3 already prices
# existing resources at $0).
CLEAN_COST = {
    'medium': {
        'CAISO': [33.7, 38.5, 41.9, 44.4, 47.4, 52.1, 55.7, 61.4, 62.9, 66.8, 68.2, 72.2, 77.9, 84.3, 89.3, 89.3, 89.3],
        'ERCOT': [10.7, 13.7, 16.4, 18.6, 20.7, 23.6, 26.3, 30.0, 30.9, 33.9, 36.8, 41.5, 46.6, 47.6, 50.3, None, 50.3],
        'PJM':   [21.2, 25.8, 29.5, 32.8, 35.6, 38.2, 41.9, 46.7, 47.8, 49.7, 52.0, 54.6, 60.3, 67.5, 95.0, 95.0, 95.0],
        'NYISO': [44.1, 47.2, 50.1, 52.8, 55.4, 57.7, 63.9, 65.3, 66.5, 68.2, 70.8, 73.7, 76.9, 84.2, None, None, None],
        'NEISO': [69.8, 70.8, 71.9, 72.6, 73.4, 73.8, 79.9, 81.8, 83.5, 85.8, 88.4, 91.9, 98.5, 106.8, 117.7, 117.7, 117.7],
        'MISO':  [25.7, 27.9, 29.8, 31.5, 33.1, 35.0, 37.1, 41.0, 43.8, 46.0, 50.2, 55.4, 57.7, 57.7, 66.4, 66.4, 66.4],
        'SPP':   [10.7, 13.3, 15.7, 17.7, 19.5, 21.3, 23.4, 26.6, 28.1, 30.5, 33.1, 36.5, 41.6, 48.8, 48.8, 48.8, 48.8],
    },
    'low': {
        'CAISO': [25.87, 28.14, 29.96, 31.71, 33.76, 37.13, 39.84, 44.75, 48.4, 50.95, 53.3, 56.49, 60.89, 66.01, 72.16, 72.16, 72.16],
        'ERCOT': [8.43, 10.51, 12.43, 13.95, 15.35, 17.42, 19.33, 21.93, 22.6, 24.67, 26.79, 31.37, 33.75, 34.53, 38.08, None, 38.08],
        'PJM':   [16.31, 19.24, 21.83, 24.3, 26.09, 27.91, 31.41, 33.52, 36.6, 39.4, 40.47, 43.46, 47.63, 53.34, 76.39, 76.39, 76.39],
        'NYISO': [32.44, 34.61, 36.58, 38.44, 39.79, 40.66, 41.85, 43.75, 46.58, 49.01, 51.13, 53.02, 57.2, 62.08, None, None, None],
        'NEISO': [55.32, 55.84, 56.09, 56.83, 57.51, 58.47, 59.13, 62.28, 64.83, 67.21, 69.57, 71.83, 76.2, 82.14, 92.66, 92.66, 92.66],
        'MISO':  [19.39, 20.98, 22.25, 23.45, 24.6, 25.96, 27.58, 30.3, 32.38, 33.98, 35.96, 41.3, 41.52, 43.39, 48.24, 48.24, 48.24],
        'SPP':   [8.45, 10.31, 11.96, 13.39, 14.69, 15.97, 17.41, 19.76, 20.85, 22.54, 24.44, 26.91, 30.93, 36.34, 36.26, 36.26, 36.26],
    },
    'high': {
        'CAISO': [38.64, 44.54, 44.74, 47.54, 51.02, 56.52, 58.57, 65.78, 70.39, 73.94, 77.96, 82.47, 90.02, 95.43, 106.55, 106.55, 106.55],
        'ERCOT': [12.74, 16.58, 20.1, 22.92, 25.55, 29.31, 32.8, 36.96, 38.64, 41.59, 45.1, 49.9, 57.44, 60.29, 63.47, None, 63.47],
        'PJM':   [25.71, 31.16, 35.54, 39.42, 42.6, 45.56, 50.03, 55.85, 58.98, 63.93, 66.29, 71.54, 78.78, 87.86, 124.17, 124.17, 124.17],
        'NYISO': [49.27, 52.67, 56.33, 59.05, 61.61, 64.32, 67.88, 70.31, 72.97, 77.28, 79.43, 84.18, 93.28, 96.96, None, None, None],
        'NEISO': [77.09, 79.24, 81.38, 82.53, 84.57, 86.72, 93.72, 99.86, 107.37, 115.05, 120.37, 128.71, 137.41, 150.17, 144.06, 144.06, 144.06],
        'MISO':  [31.43, 34.32, 36.66, 38.84, 40.92, 43.29, 46.25, 48.42, 51.56, 54.94, 59.97, 66.4, 68.29, 70.88, 81.64, 81.64, 81.64],
        'SPP':   [12.71, 16.1, 19.09, 21.62, 23.99, 26.24, 28.88, 32.89, 35.0, 37.95, 41.31, 43.74, 50.04, 57.23, 60.54, 60.54, 60.54],
    },
}

# P25/P75 cost tiers — interpolated between L/M (P10/P50) and M/H (P50/P90)
# P25 = low + 0.375 * (medium - low)   [at 25th pctile between P10 and P50]
# P75 = medium + 0.625 * (high - medium) [at 75th pctile between P50 and P90]
def _interpolate_cost_tier(low_vals, med_vals, high_vals, target_pctile):
    """Interpolate a cost tier between L/M/H at a given percentile (10-90 scale)."""
    result = []
    for l, m, h in zip(low_vals, med_vals, high_vals):
        if l is None or m is None or h is None:
            result.append(None)
        elif target_pctile <= 50:
            frac = (target_pctile - 10) / (50 - 10)  # 0..1 between P10 and P50
            result.append(round(l + frac * (m - l), 2))
        else:
            frac = (target_pctile - 50) / (90 - 50)  # 0..1 between P50 and P90
            result.append(round(m + frac * (h - m), 2))
    return result

CLEAN_COST['p25'] = {}
CLEAN_COST['p75'] = {}
for _iso in ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']:
    CLEAN_COST['p25'][_iso] = _interpolate_cost_tier(
        CLEAN_COST['low'][_iso], CLEAN_COST['medium'][_iso], CLEAN_COST['high'][_iso], 25
    )
    CLEAN_COST['p75'][_iso] = _interpolate_cost_tier(
        CLEAN_COST['low'][_iso], CLEAN_COST['medium'][_iso], CLEAN_COST['high'][_iso], 75
    )

# Gas backup capacity cost per threshold ($/MWh) — medium scenario from Step 4
# This is NOT part of the MAC calculation — it's a separate system cost tracked
# for the dashboard warning: "chasing cheap carbon without considering what the
# grid needs to operate means retaining/building gas capacity."
GAS_BACKUP_COST = {
    'CAISO': [9.94, 9.52, 9.48, 9.35, 8.9, 8.68, 8.48, 6.76, 3.43, 4.01, 2.32, 2.27, 2.08, 1.82, 2.86, 2.86, 2.86],
    'ERCOT': [10.58, 10.49, 10.41, 10.36, 10.3, 10.32, 10.11, 10.0, 9.92, 9.89, 9.69, 8.89, 8.04, 8.16, 7.35, 0, 7.35],
    'PJM':   [13.35, 13.25, 13.26, 13.16, 13.19, 13.22, 11.66, 12.16, 9.6, 7.54, 7.53, 5.43, 5.31, 3.94, 1.0, 1.0, 1.0],
    'NYISO': [17.85, 17.73, 17.55, 17.4, 17.28, 17.09, 16.97, 16.05, 13.49, 11.61, 11.16, 9.86, 7.57, 6.84, 0, 0, 0],
    'NEISO': [16.31, 16.28, 16.21, 16.19, 16.12, 16.3, 15.78, 14.17, 11.89, 10.02, 9.44, 7.77, 7.11, 5.5, 12.96, 12.96, 12.96],
    'MISO':  [14.25, 14.17, 14.1, 14.08, 13.98, 13.9, 13.71, 13.58, 13.49, 13.03, 12.09, 9.59, 11.46, 11.78, 11.08, 11.08, 11.08],
    'SPP':   [12.65, 12.59, 12.54, 12.43, 12.4, 12.29, 12.23, 12.07, 11.84, 11.85, 11.68, 11.42, 10.88, 10.67, 10.99, 10.99, 10.99],
}

# Resource mix data (% of demand) — medium-cost physics optimization
# Note: mixes are optimized at medium costs. Sensitivity toggles recalculate cost
# on cached physics, so mixes don't vary by cost tier. This is a model limitation
# for the no-regrets analysis (we can't see if high-cost scenarios shift the mix).
RESOURCE_MIX_DATA = {
    'CAISO': {
        'clean_firm':  [14, 12, 11, 11, 14, 15, 13, 22, 44, 36, 58, 55, 57, 61, 26, 26, 26],
        'solar':       [33, 39, 35, 31, 27, 22, 22, 22, 22, 22, 22, 22, 22, 22, 26, 26, 26],
        'wind':        [44, 40, 45, 49, 50, 54, 56, 47, 25, 33, 11, 14, 12, 8, 39, 39, 39],
        'ccs_ccgt':    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'hydro':       [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
        'battery':     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'battery8':    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'ldes':        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        'h2':          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    'ERCOT': {
        'clean_firm':  [16, 15, 14, 13, 12, 11, 10, 9, 9, 8, 8, 12, 8, 8, 13, 0, 13],
        'solar':       [25, 23, 22, 20, 19, 15, 16, 14, 15, 13, 14, 16, 56, 47, 49, 0, 49],
        'wind':        [59, 62, 64, 67, 69, 74, 74, 77, 76, 79, 78, 72, 36, 45, 38, 0, 38],
        'ccs_ccgt':    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'hydro':       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'battery':     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'battery8':    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'ldes':        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'h2':          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    'PJM': {
        'clean_firm':  [63, 58, 53, 50, 46, 43, 56, 43, 66, 82, 77, 91, 82, 83, 85, 85, 85],
        'solar':       [29, 27, 23, 19, 16, 11, 4, 11, 5, 3, 5, 3, 7, 8, 5, 5, 5],
        'wind':        [7, 14, 23, 30, 37, 45, 39, 45, 28, 14, 17, 5, 10, 8, 9, 9, 9],
        'ccs_ccgt':    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'hydro':       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'battery':     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'battery8':    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'ldes':        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'h2':          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    'NYISO': {
        'clean_firm':  [36, 33, 31, 29, 27, 25, 22, 24, 42, 54, 53, 58, 69, 66, 0, 0, 0],
        'solar':       [0, 0, 0, 0, 0, 4, 0, 12, 10, 7, 9, 9, 7, 7, 0, 0, 0],
        'wind':        [49, 52, 54, 56, 58, 56, 63, 49, 33, 24, 23, 18, 9, 12, 0, 0, 0],
        'ccs_ccgt':    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'hydro':       [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 0, 0, 0],
        'battery':     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'battery8':    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'ldes':        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'h2':          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    'NEISO': {
        'clean_firm':  [47, 43, 40, 37, 35, 32, 29, 42, 60, 73, 73, 81, 77, 80, 23, 23, 23],
        'solar':       [31, 26, 22, 18, 14, 5, 16, 7, 4, 2, 3, 3, 5, 4, 25, 25, 25],
        'wind':        [18, 27, 34, 41, 47, 59, 51, 47, 32, 21, 20, 12, 14, 12, 48, 48, 48],
        'ccs_ccgt':    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'hydro':       [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        'battery':     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'battery8':    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'ldes':        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        'h2':          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    'MISO': {
        'clean_firm':  [26, 24, 22, 20, 19, 18, 17, 15, 14, 15, 21, 39, 13, 13, 13, 13, 13],
        'solar':       [4, 4, 4, 3, 3, 3, 8, 8, 7, 16, 13, 12, 55, 42, 45, 45, 45],
        'wind':        [69, 71, 73, 76, 77, 78, 74, 76, 78, 68, 65, 48, 31, 44, 41, 41, 41],
        'ccs_ccgt':    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'hydro':       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'battery':     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'battery8':    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'ldes':        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        'h2':          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    'SPP': {
        'clean_firm':  [10, 9, 8, 8, 7, 7, 6, 6, 6, 5, 5, 5, 6, 5, 5, 5, 5],
        'solar':       [1, 1, 1, 1, 1, 1, 3, 1, 9, 6, 7, 10, 16, 13, 1, 1, 1],
        'wind':        [85, 86, 87, 87, 88, 88, 87, 89, 81, 85, 84, 81, 74, 78, 90, 90, 90],
        'ccs_ccgt':    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'hydro':       [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        'battery':     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'battery8':    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'ldes':        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        'h2':          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
}


# ============================================================================
# DEMAND GROWTH
# ============================================================================

def demand_growth_factor(iso, threshold, growth_tier='medium'):
    """Compute demand growth factor: (1 + annual_rate)^(year - 2025)."""
    year = threshold_to_year(threshold)
    rate = DEMAND_GROWTH_RATES[iso][growth_tier]
    return (1 + rate) ** (year - 2025)


def demand_at_threshold(iso, threshold, growth_tier='medium'):
    """Demand in TWh at a given threshold/year under a growth scenario."""
    return DEMAND_TWH[iso] * demand_growth_factor(iso, threshold, growth_tier)


# ============================================================================
# DATA LOADING (canonical data files)
# ============================================================================

def load_canonical_data():
    """Load emission rates and fossil mix from canonical data files (same as step6)."""
    global _EMISSION_RATES_JSON, _FOSSIL_MIX_JSON

    if _EMISSION_RATES_JSON is not None:
        return  # Already loaded

    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'

    # Emission rates
    egrid_path = data_dir / 'egrid_emission_rates.json'
    if egrid_path.exists():
        with open(egrid_path) as f:
            _EMISSION_RATES_JSON = json.load(f)
        print(f"  Loaded emission rates: {egrid_path}")
    else:
        print(f"  WARNING: {egrid_path} not found — using fallback constants")

    # Fossil mix
    fossil_path = data_dir / 'eia_fossil_mix.json'
    if fossil_path.exists():
        with open(fossil_path) as f:
            _FOSSIL_MIX_JSON = json.load(f)
        print(f"  Loaded fossil mix: {fossil_path}")
    else:
        print(f"  WARNING: {fossil_path} not found — using fallback constants")


# Runtime parquet-sourced cost data: NEW resources only (per MWh of demand)
# Populated by load_parquet_costs() in main(), used by compute_marginal_mac_curve()
_PARQUET_COSTS = None  # {iso: {threshold: {'low': v, 'p25': v, 'medium': v, 'p75': v, 'high': v}}}

# Resource types for existing/new split
_RESOURCE_TYPES = ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro']


def load_parquet_costs():
    """Load NEW-resource-only cost per demand MWh from parquets.

    Computes P10/P25/P50/P75/P90 across all cost scenarios at each threshold.
    Cost = cost_total_cost - existing_wholesale_cost - gas_backup_cost

    Step 3 prices existing resources at wholesale. We subtract that to get
    only the LCOE cost of new clean energy investment. Gas backup is excluded
    (system reliability cost, not abatement cost).

    Falls back to CLEAN_COST if parquets unavailable.
    """
    global _PARQUET_COSTS
    import pandas as pd

    base_dir = Path(__file__).parent.parent
    data_dirs = [
        base_dir / 'data' / 'step5-post-processing' / 'co2_results',
        base_dir / 'data' / 'step4-gas-ccs-parquets',
        base_dir / 'data' / 'step3-cost-opt-parquets',
    ]
    prefixes = ['co2_', 'step4_', 'step3_co_']

    result = {}
    for iso in ISOS:
        df = None
        for data_dir in data_dirs:
            for prefix in prefixes:
                path = data_dir / f'{prefix}{iso}.parquet'
                if path.exists():
                    df = pd.read_parquet(path)
                    break
            if df is not None:
                break

        if df is None:
            print(f"  WARNING: No parquet found for {iso} — using CLEAN_COST fallback")
            continue

        # New resource cost: Step 3 already prices existing at $0 (sunk fleet).
        # cost_total_cost = LCOE of new-build clean only. No wholesale offset.
        new_cost = df['cost_total_cost'].copy()
        if 'ra_gas_backup_cost_per_mwh' in df.columns:
            new_cost = new_cost - df['ra_gas_backup_cost_per_mwh'].fillna(0)
        elif 'gas_gas_cost_per_mwh' in df.columns:
            new_cost = new_cost - df['gas_gas_cost_per_mwh'].fillna(0)
        new_cost = new_cost.clip(lower=0)

        result[iso] = {}
        for t in THRESHOLDS:
            t_mask = df['threshold'] == t
            t_costs = new_cost[t_mask]
            if len(t_costs) == 0:
                continue

            pcts = t_costs.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
            result[iso][t] = {
                'low': round(float(pcts[0.10]), 4),
                'p25': round(float(pcts[0.25]), 4),
                'medium': round(float(pcts[0.50]), 4),
                'p75': round(float(pcts[0.75]), 4),
                'high': round(float(pcts[0.90]), 4),
            }

        if result[iso]:
            med_50 = result[iso].get(50, {}).get('medium', '?')
            med_95 = result[iso].get(95, {}).get('medium', '?')
            print(f"  {iso}: new-resource cost loaded ({len(result[iso])} thresholds, "
                  f"med@50%=${med_50}, med@95%=${med_95})")

    _PARQUET_COSTS = result if result else None
    return result


# ============================================================================
# CO₂ MODEL — Compute total fossil emissions and CO₂ reduced by new resources
# ============================================================================

# Per-ISO fuel emission rates (tCO₂/MWh) from eGRID data
_LB_PER_TON = 2204.623
_FUEL_RATES = {}
_EGRID_PATH_OT = Path(__file__).parent.parent / 'data' / 'egrid_emission_rates.json'
if _EGRID_PATH_OT.exists():
    with open(_EGRID_PATH_OT) as _f_ot:
        _EGRID_OT = json.load(_f_ot)
    for _iso_ot in ISOS:
        _rates_ot = _EGRID_OT.get(_iso_ot, {})
        _FUEL_RATES[_iso_ot] = {
            'coal': _rates_ot.get('coal_co2_lb_per_mwh', 0.0) / _LB_PER_TON,
            'oil': _rates_ot.get('oil_co2_lb_per_mwh', 0.0) / _LB_PER_TON,
            'gas': _rates_ot.get('gas_co2_lb_per_mwh', 0.0) / _LB_PER_TON,
        }
else:
    _FUEL_RATES = EMISSION_RATES_FALLBACK

# Existing clean percentage per ISO (2025 baseline)
EXISTING_CLEAN_PCT = {iso: sum(GRID_MIX_SHARES[iso].values()) for iso in ISOS}


def compute_total_fossil_emissions_mt(iso, clean_pct, demand_twh=None):
    """Compute total fossil CO₂ emissions (Mt) at a given clean energy level.

    Uses merit-order dispatch: coal remains first, then oil, then gas.
    As clean_pct increases, fossil shrinks. Coal/oil fully retire at 70%.

    Returns emissions in Mt (= TWh × tCO₂/MWh, since 1 TWh = 1e6 MWh).
    """
    if demand_twh is None:
        demand_twh = DEMAND_TWH.get(iso, 0)

    fossil_pct = max(0, (100.0 - clean_pct)) / 100.0
    fossil_twh = demand_twh * fossil_pct
    if fossil_twh <= 0.01:
        return 0.0

    coal_cap = COAL_CAP_TWH.get(iso, 0)
    oil_cap = OIL_CAP_TWH.get(iso, 0)
    rates = _FUEL_RATES.get(iso, EMISSION_RATES_FALLBACK.get(iso, {'coal': 0, 'oil': 0, 'gas': 0}))

    if clean_pct >= COAL_OIL_RETIREMENT_THRESHOLD:
        return fossil_twh * rates['gas']
    else:
        coal_twh = min(coal_cap, fossil_twh)
        remaining = fossil_twh - coal_twh
        oil_twh = min(oil_cap, remaining)
        gas_twh = max(0, remaining - oil_twh)
        return (coal_twh * rates['coal'] + oil_twh * rates['oil']
                + gas_twh * rates['gas'])


def compute_co2_reduced_by_new(iso, threshold_pct, growth_tier='medium'):
    """Compute CO₂ reduced by NEW clean energy (Mt) at a given threshold.

    Methodology:
    1. Baseline: existing clean stays at 2025 TWh, demand grows per SBTi year.
       Existing clean as % of grown demand = existing_pct / growth_factor.
       Fossil fills the gap. Baseline emissions = fossil × fleet emission rate.
    2. Scenario: at threshold_pct clean, fossil = (100-threshold)/100 × demand.
       Scenario emissions = fossil × fleet rate (coal→oil→gas merit order).
    3. CO₂ reduced = baseline_emissions - scenario_emissions.
    """
    gf = demand_growth_factor(iso, threshold_pct, growth_tier)
    demand_twh = DEMAND_TWH[iso] * gf

    # Baseline: existing clean TWh stays fixed, but its pct shrinks with growth
    existing_pct = EXISTING_CLEAN_PCT[iso]
    existing_pct_diluted = existing_pct / gf  # diluted by demand growth

    # Compute emissions at both levels
    baseline_emissions = compute_total_fossil_emissions_mt(iso, existing_pct_diluted, demand_twh)
    scenario_emissions = compute_total_fossil_emissions_mt(iso, threshold_pct, demand_twh)

    co2_reduced = max(0, baseline_emissions - scenario_emissions)

    # Marginal rate: gas rate (dominant at high thresholds above 70%)
    rates = _FUEL_RATES.get(iso, EMISSION_RATES_FALLBACK.get(iso, {}))
    if threshold_pct >= COAL_OIL_RETIREMENT_THRESHOLD:
        marginal_rate = rates.get('gas', 0.39)
    else:
        marginal_rate = rates.get('coal', 1.0) if threshold_pct < 50 else rates.get('gas', 0.39)

    return {
        'total_co2_mt': co2_reduced,
        'marginal_rate': marginal_rate,
        'baseline_emissions_mt': baseline_emissions,
        'scenario_emissions_mt': scenario_emissions,
        'existing_pct_diluted': existing_pct_diluted,
        'demand_twh': demand_twh,
        'source': 'dispatch_model',
    }


# Keep old name as alias for backward compat (some code may still call it)
def compute_co2_abated(iso, threshold_pct, growth_tier='medium'):
    """Backward-compatible wrapper for compute_co2_reduced_by_new."""
    return compute_co2_reduced_by_new(iso, threshold_pct, growth_tier)


# ============================================================================
# DAC COST INTERPOLATION
# ============================================================================

def threshold_to_year(t):
    """Map threshold to SBTi year via linear interpolation."""
    keys = sorted(THRESHOLD_YEAR_MAP.keys())
    if t <= keys[0]:
        return THRESHOLD_YEAR_MAP[keys[0]]
    if t >= keys[-1]:
        return THRESHOLD_YEAR_MAP[keys[-1]]
    for i in range(len(keys) - 1):
        if keys[i] <= t <= keys[i + 1]:
            frac = (t - keys[i]) / (keys[i + 1] - keys[i])
            return THRESHOLD_YEAR_MAP[keys[i]] + frac * (THRESHOLD_YEAR_MAP[keys[i + 1]] - THRESHOLD_YEAR_MAP[keys[i]])
    return THRESHOLD_YEAR_MAP[keys[-1]]


def dac_cost_at_threshold(threshold, trajectory='central'):
    """Interpolate DAC cost at a threshold via SBTi year mapping."""
    year = threshold_to_year(threshold)
    traj = DAC_TRAJECTORY[trajectory]
    years = sorted(traj.keys())
    if year <= years[0]:
        return traj[years[0]]
    if year >= years[-1]:
        return traj[years[-1]]
    for i in range(len(years) - 1):
        if years[i] <= year <= years[i + 1]:
            frac = (year - years[i]) / (years[i + 1] - years[i])
            return traj[years[i]] + frac * (traj[years[i + 1]] - traj[years[i]])
    return traj[years[-1]]


# ============================================================================
# OPTION B: SMOOTH MARGINAL MAC × CROSSOVER RANGE
# ============================================================================

def enforce_monotonic(arr):
    """Forward-max: enforce monotonically non-decreasing (simple fallback)."""
    result = arr.copy()
    for i in range(1, len(result)):
        if result[i] < result[i - 1]:
            result[i] = result[i - 1]
    return result


def compute_marginal_mac_curve(iso, cost_tier='medium', growth_tier='medium'):
    """
    Compute smooth marginal MAC curve for one ISO × one cost × one growth scenario.

    Methodology (stepwise → isotonic → PCHIP re-interpolation):
      1. Compute raw stepwise MAC at each discrete threshold:
         delta_cost / delta_co2 between adjacent thresholds.
      2. Apply isotonic regression to enforce non-decreasing marginal MAC.
      3. Re-fit PCHIP through (threshold, isotonic_mac) for a smooth 300-point
         curve. Final forward-max pass catches any minor PCHIP wiggles.

    This avoids PCHIP derivative artifacts entirely — PCHIP is only used for
    interpolation (stable), never differentiation (unstable at boundaries).
    Eliminates: tail explosions (PJM $6880), boundary spikes (CAISO 50%),
    inflection spikes (NYISO 78%), and staircase artifacts from isotonic on
    dense points.

    Cost basis: NEW-resource-only LCOE per MWh of demand from parquets
    (P10/P25/P50/P75/P90 across sensitivity scenarios). Existing clean
    resources' wholesale cost is subtracted, gas backup excluded. Falls
    back to CLEAN_COST if parquet data unavailable.

    CO₂ basis: baseline emissions (existing clean at 2025 TWh, diluted by
    demand growth) minus scenario emissions (at threshold, grown demand).
    This captures only the CO₂ reduction attributable to new investment.
    """
    try:
        from scipy.optimize import isotonic_regression as _iso_reg
    except ImportError:
        _iso_reg = None

    # Build arrays, skipping nulls
    # Cost = new-resource LCOE per demand MWh × demand TWh
    valid_t, valid_clean_cost, valid_co2 = [], [], []
    use_parquet = (_PARQUET_COSTS is not None and iso in _PARQUET_COSTS)
    for i, t in enumerate(THRESHOLDS):
        # Get new-resource cost per MWh from parquets (preferred) or CLEAN_COST (fallback)
        if use_parquet and t in _PARQUET_COSTS[iso]:
            sc = _PARQUET_COSTS[iso][t].get(cost_tier)
        else:
            costs = CLEAN_COST.get(cost_tier, {}).get(iso, [])
            sc = costs[i] if i < len(costs) else None

        if sc is None:
            continue
        gf = demand_growth_factor(iso, t, growth_tier)
        demand_twh = DEMAND_TWH[iso] * gf
        cost_total = sc * demand_twh
        co2 = compute_co2_reduced_by_new(iso, t, growth_tier)['total_co2_mt']

        valid_t.append(t)
        valid_clean_cost.append(cost_total)
        valid_co2.append(co2)

    if len(valid_t) < 4:
        return None

    t_arr = np.array(valid_t)
    cost_arr = np.array(valid_clean_cost)
    co2_arr = np.array(valid_co2)

    # Enforce monotonicity on cumulative curves
    cost_mono = enforce_monotonic(cost_arr)
    co2_mono = enforce_monotonic(co2_arr)

    if HAS_SCIPY:
        # --- Step 1: Raw stepwise MAC at discrete thresholds ---
        step_mac = np.zeros(len(valid_t))
        # First point: average MAC (total_cost / total_co2)
        if co2_mono[0] > 1e-6:
            step_mac[0] = cost_mono[0] / co2_mono[0]
        for i in range(1, len(valid_t)):
            dc = cost_mono[i] - cost_mono[i - 1]
            dq = co2_mono[i] - co2_mono[i - 1]
            if dq > 1e-6:
                step_mac[i] = dc / dq
            else:
                # No additional CO2 abated — carry forward previous MAC
                step_mac[i] = step_mac[i - 1]

        # Floor at 0.01 (no negative MAC)
        step_mac = np.maximum(step_mac, 0.01)

        # --- Step 2: Isotonic regression at discrete points ---
        if _iso_reg is not None:
            iso_result = _iso_reg(step_mac)
            isotonic_mac = iso_result.x if hasattr(iso_result, 'x') else iso_result
        else:
            isotonic_mac = enforce_monotonic(step_mac)

        mac_at_t = [round(float(v), 2) for v in isotonic_mac]

        # --- Step 3: PCHIP re-interpolation for smooth 300-point curve ---
        t_dense = np.linspace(float(t_arr[0]), float(t_arr[-1]), 300)
        pchip_smooth = PchipInterpolator(t_arr, isotonic_mac)
        marginal_mac = pchip_smooth(t_dense)

        # Floor + forward-max to catch any minor PCHIP wiggles
        marginal_mac = np.maximum(marginal_mac, 0.01)
        marginal_mac = enforce_monotonic(marginal_mac)

    else:
        # Fallback: raw stepwise without scipy
        t_dense = t_arr
        marginal_mac = np.full(len(t_arr), np.nan)
        for i in range(1, len(t_arr)):
            dc = cost_mono[i] - cost_mono[i - 1]
            dq = co2_mono[i] - co2_mono[i - 1]
            marginal_mac[i] = dc / dq if dq > 1e-6 else np.nan
        marginal_mac[0] = marginal_mac[1] if len(marginal_mac) > 1 else np.nan
        mac_at_t = [float(m) if not np.isnan(m) else None for m in marginal_mac]

    return {
        'thresholds': [float(x) for x in valid_t],
        'clean_cost_M': [float(x) for x in cost_mono],
        'co2_Mt': [float(x) for x in co2_mono],
        'smooth_t': [float(x) for x in t_dense],
        'smooth_mac': [float(x) if not np.isnan(x) else None for x in marginal_mac],
        'mac_at_thresholds': mac_at_t,
        'cost_corrected': not np.allclose(cost_arr, cost_mono, atol=0.1),
        'co2_corrected': not np.allclose(co2_arr, co2_mono, atol=0.01),
        'growth_tier': growth_tier,
    }


def find_crossover(t_dense, mac_curve, dac_curve):
    """Find threshold where marginal MAC first exceeds DAC cost."""
    for i in range(len(t_dense)):
        mac = mac_curve[i]
        if mac is None or np.isnan(mac):
            continue
        dac = dac_curve[i]
        if mac > dac:
            if i > 0 and mac_curve[i - 1] is not None and not np.isnan(mac_curve[i - 1]):
                gap_prev = dac_curve[i - 1] - mac_curve[i - 1]
                gap_curr = mac - dac
                frac = gap_prev / (gap_prev + gap_curr) if (gap_prev + gap_curr) > 0 else 0.5
                cross_t = t_dense[i - 1] + frac * (t_dense[i] - t_dense[i - 1])
                cross_mac = mac_curve[i - 1] + frac * (mac - mac_curve[i - 1])
                cross_dac = dac_curve[i - 1] + frac * (dac - dac_curve[i - 1])
            else:
                cross_t, cross_mac, cross_dac = t_dense[i], mac, dac
            return {
                'threshold': round(float(cross_t), 1),
                'mac': round(float(cross_mac), 1),
                'dac': round(float(cross_dac), 1),
                'year': round(threshold_to_year(float(cross_t))),
            }
    return {'threshold': None, 'note': 'Grid cheaper than DAC at all thresholds'}


def compute_crossover_range(iso):
    """
    Cross 5 grid cost tiers (P10/P25/P50/P75/P90) × 3 DAC scenarios = 15 crossover points.

    Returns two ranges:
    - range: P25/P75 "analysis" range for optimal targets & no-regrets (tighter, more actionable)
    - range_full: P10/P90 full range for display on interactive dashboard (wider uncertainty band)

    Note: Demand growth doesn't change crossover thresholds (MAC is scale-invariant),
    so we compute crossovers at medium growth only. Growth affects absolute quantities,
    which are computed separately in the no-regrets analysis.
    """
    crossovers = {}
    mac_curves = {}

    for cost_tier in ['low', 'p25', 'medium', 'p75', 'high']:
        curve = compute_marginal_mac_curve(iso, cost_tier, 'medium')
        if curve is None:
            continue
        mac_curves[cost_tier] = curve
        t_dense = curve['smooth_t']
        mac_dense = curve['smooth_mac']

        for dac_scenario in ['optimistic', 'central', 'conservative']:
            dac_dense = [dac_cost_at_threshold(t, dac_scenario) for t in t_dense]
            xo = find_crossover(t_dense, mac_dense, dac_dense)
            key = f'{cost_tier}_grid__{dac_scenario}_dac'
            crossovers[key] = xo

    # Analysis range: P25/P75 tiers only (tighter, more actionable)
    analysis_tiers = {'p25', 'medium', 'p75'}
    analysis_thresholds = [
        xo['threshold'] for k, xo in crossovers.items()
        if xo.get('threshold') is not None and k.split('_grid__')[0] in analysis_tiers
    ]

    if not analysis_thresholds:
        range_result = {
            'lower_bound': None,
            'upper_bound': None,
            'note': 'Grid cheaper than DAC in all P25/P75 scenarios',
        }
    else:
        range_result = {
            'lower_bound': min(analysis_thresholds),
            'upper_bound': max(analysis_thresholds),
            'lower_scenario': next(
                k for k, v in crossovers.items()
                if v.get('threshold') == min(analysis_thresholds) and k.split('_grid__')[0] in analysis_tiers
            ),
            'upper_scenario': next(
                k for k, v in crossovers.items()
                if v.get('threshold') == max(analysis_thresholds) and k.split('_grid__')[0] in analysis_tiers
            ),
        }

    # Full display range: P10/P90 tiers (wider, shown on interactive dashboard)
    all_thresholds = [
        xo['threshold'] for xo in crossovers.values()
        if xo.get('threshold') is not None
    ]

    if not all_thresholds:
        range_full = {
            'lower_bound': None,
            'upper_bound': None,
            'note': 'Grid cheaper than DAC in all scenarios',
        }
    else:
        range_full = {
            'lower_bound': min(all_thresholds),
            'upper_bound': max(all_thresholds),
            'lower_scenario': next(
                k for k, v in crossovers.items() if v.get('threshold') == min(all_thresholds)
            ),
            'upper_scenario': next(
                k for k, v in crossovers.items() if v.get('threshold') == max(all_thresholds)
            ),
        }

    return {
        'crossovers': crossovers,
        'range': range_result,           # P25/P75 for analysis
        'range_full': range_full,         # P10/P90 for display
        'mac_curves': mac_curves,
    }


# ============================================================================
# OPTION A: TARGET-SPECIFIC ANALYSIS WITHIN THE RANGE
# ============================================================================

def analyze_targets_in_range(iso, lower_bound, upper_bound, mac_curves):
    """
    For each discrete threshold within the crossover range,
    compute marginal cost and compare to DAC, across L/M/H cost tiers
    and L/M/H demand growth scenarios.
    """
    if lower_bound is None or upper_bound is None:
        return []

    # Expand range by 1 step on each side for context
    all_t = THRESHOLDS
    lower_idx = max(0, next((i for i, t in enumerate(all_t) if t >= lower_bound), 0) - 1)
    upper_idx = min(len(all_t) - 1, next((i for i, t in enumerate(all_t) if t >= upper_bound), len(all_t) - 1) + 1)

    targets = all_t[lower_idx:upper_idx + 1]
    results = []

    for cost_tier in ['low', 'medium', 'high']:
        curve = mac_curves.get(cost_tier)
        if not curve:
            continue

        for growth_tier in ['low', 'medium', 'high']:
            for j, t in enumerate(targets):
                if t not in curve['thresholds']:
                    continue
                idx = curve['thresholds'].index(t)

                # Discrete marginal cost/CO₂
                if idx > 0:
                    dcost = curve['clean_cost_M'][idx] - curve['clean_cost_M'][idx - 1]
                    dco2 = curve['co2_Mt'][idx] - curve['co2_Mt'][idx - 1]
                    discrete_mac = dcost / dco2 if dco2 > 0.001 else None
                else:
                    dcost, dco2, discrete_mac = None, None, None

                spline_mac = curve['mac_at_thresholds'][idx] if idx < len(curve['mac_at_thresholds']) else None

                dac_opt = dac_cost_at_threshold(t, 'optimistic')
                dac_cen = dac_cost_at_threshold(t, 'central')
                dac_con = dac_cost_at_threshold(t, 'conservative')

                # Demand-growth-adjusted totals
                gf = demand_growth_factor(iso, t, growth_tier)
                demand_twh = DEMAND_TWH[iso] * gf
                # Use parquet-sourced cost (per demand MWh, existing at $0) if available
                if _PARQUET_COSTS and iso in _PARQUET_COSTS and t in _PARQUET_COSTS[iso]:
                    clean_cost = _PARQUET_COSTS[iso][t].get(cost_tier)
                else:
                    clean_cost = CLEAN_COST[cost_tier][iso][THRESHOLDS.index(t)]
                total_cost_M = (clean_cost * demand_twh) if clean_cost else None
                total_co2 = compute_co2_abated(iso, t, growth_tier)['total_co2_mt']

                results.append({
                    'threshold': t,
                    'year': round(threshold_to_year(t)),
                    'cost_tier': cost_tier,
                    'growth_tier': growth_tier,
                    'growth_factor': round(gf, 4),
                    'demand_twh': round(demand_twh, 1),
                    'clean_cost_mwh': clean_cost,
                    'total_clean_cost_M': round(total_cost_M, 1) if total_cost_M else None,
                    'total_co2_Mt': round(total_co2, 2),
                    'discrete_mac': round(discrete_mac, 1) if discrete_mac else None,
                    'spline_mac': round(spline_mac, 1) if spline_mac else None,
                    'dac_optimistic': round(dac_opt, 1),
                    'dac_central': round(dac_cen, 1),
                    'dac_conservative': round(dac_con, 1),
                    'grid_cheaper_than_dac_central': (
                        (spline_mac or discrete_mac or 0) < dac_cen
                        if (spline_mac or discrete_mac) else None
                    ),
                })

    return results


# ============================================================================
# NO-REGRETS RESOURCE INVESTMENT ANALYSIS
# ============================================================================

def compute_resource_twh_at_threshold(iso, threshold_idx, growth_tier='medium'):
    """
    Compute absolute resource TWh at a given threshold index.
    Resource TWh = (resource_pct / 100) × demand × gf
    (v5.0: procurement is baked into resource percentages directly)
    """
    mix = RESOURCE_MIX_DATA[iso]
    t = THRESHOLDS[threshold_idx]
    gf = demand_growth_factor(iso, t, growth_tier)
    demand = DEMAND_TWH[iso] * gf

    result = {}
    for res in RESOURCES:
        res_pct = mix[res][threshold_idx]
        twh = (res_pct / 100.0) * demand
        result[res] = round(twh, 2)
    result['total_clean_twh'] = round(sum(result[r] for r in RESOURCES), 2)
    return result


def compute_no_regrets_investments(iso, lower_bound, upper_bound):
    """
    Identify 'no regrets' resource investments within the crossover range.

    For each resource:
    - Floor: minimum % share AND minimum TWh across all thresholds in range
    - Consensus: whether the resource is non-zero at EVERY threshold in range
    - Average: mean investment level across the range
    - Scaled by L/M/H demand growth for absolute TWh

    Returns structured results for each resource type.
    """
    if lower_bound is None or upper_bound is None:
        return {'note': 'No crossover range — grid always cheaper than DAC'}

    # Find discrete thresholds within the range (inclusive ±1 step)
    all_t = THRESHOLDS
    lower_idx = max(0, next((i for i, t in enumerate(all_t) if t >= lower_bound), 0) - 1)
    upper_idx = min(len(all_t) - 1, next((i for i, t in enumerate(all_t) if t >= upper_bound), len(all_t) - 1) + 1)

    range_indices = list(range(lower_idx, upper_idx + 1))
    range_thresholds = [THRESHOLDS[i] for i in range_indices]

    mix = RESOURCE_MIX_DATA[iso]

    # Compute per-resource statistics across the range
    resource_stats = {}
    for res in RESOURCES:
        pct_values = [mix[res][i] for i in range_indices]

        # v5.0: procurement is baked into resource percentages directly
        # Effective share of demand = (res_pct/100)
        demand_shares = [pct / 100.0 for pct in pct_values]

        floor_pct = min(pct_values)
        floor_share = min(demand_shares)
        avg_pct = sum(pct_values) / len(pct_values)
        avg_share = sum(demand_shares) / len(demand_shares)
        max_pct = max(pct_values)
        max_share = max(demand_shares)
        is_consensus = all(p > 0 for p in pct_values)

        # Compute TWh at each demand growth tier
        twh_by_growth = {}
        for growth_tier in ['low', 'medium', 'high']:
            # Use the threshold with the minimum share for floor TWh
            # and average share for average TWh
            floor_twh_values = []
            avg_twh_values = []
            max_twh_values = []

            for i in range_indices:
                t = THRESHOLDS[i]
                gf = demand_growth_factor(iso, t, growth_tier)
                demand = DEMAND_TWH[iso] * gf
                res_twh = (mix[res][i] / 100.0) * demand
                floor_twh_values.append(res_twh)
                avg_twh_values.append(res_twh)
                max_twh_values.append(res_twh)

            twh_by_growth[growth_tier] = {
                'floor_twh': round(min(floor_twh_values), 2),
                'avg_twh': round(sum(avg_twh_values) / len(avg_twh_values), 2),
                'max_twh': round(max(max_twh_values), 2),
            }

        resource_stats[res] = {
            'floor_pct': floor_pct,
            'avg_pct': round(avg_pct, 1),
            'max_pct': max_pct,
            'floor_demand_share': round(floor_share * 100, 2),  # as % of demand
            'avg_demand_share': round(avg_share * 100, 2),
            'max_demand_share': round(max_share * 100, 2),
            'is_consensus': is_consensus,
            'twh_by_growth': twh_by_growth,
            'per_threshold': [
                {
                    'threshold': THRESHOLDS[i],
                    'pct': mix[res][i],
                    'pct_of_demand': round(mix[res][i] / 100.0, 4),
                }
                for i in range_indices
            ],
        }

    # Total clean energy stats across range (v5.0: procurement baked into resource %)
    total_clean_by_growth = {}
    for growth_tier in ['low', 'medium', 'high']:
        total_twh_values = []
        for i in range_indices:
            t = THRESHOLDS[i]
            gf = demand_growth_factor(iso, t, growth_tier)
            demand = DEMAND_TWH[iso] * gf
            total_pct = sum(mix[res][i] for res in RESOURCES)
            total_twh = (total_pct / 100.0) * demand
            total_twh_values.append(total_twh)
        total_clean_by_growth[growth_tier] = {
            'floor_twh': round(min(total_twh_values), 1),
            'avg_twh': round(sum(total_twh_values) / len(total_twh_values), 1),
            'max_twh': round(max(total_twh_values), 1),
        }

    return {
        'range_thresholds': range_thresholds,
        'resources': resource_stats,
        'total_clean_by_growth': total_clean_by_growth,
        'demand_growth_factors': {
            growth_tier: {
                'lower': round(demand_growth_factor(iso, lower_bound, growth_tier), 4),
                'upper': round(demand_growth_factor(iso, upper_bound, growth_tier), 4),
            }
            for growth_tier in ['low', 'medium', 'high']
        },
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / 'data' / 'step5-post-processing'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load canonical data files (emission rates + fossil mix from dispatch_utils pipeline)
    load_canonical_data()

    # Print baseline emissions diagnostic
    print("\nBaseline fossil emissions (existing clean only, 2025 demand):")
    for iso in ISOS:
        pct = EXISTING_CLEAN_PCT[iso]
        base_emit = compute_total_fossil_emissions_mt(iso, pct)
        # Show how dilution changes at 2035 (70% threshold, medium growth)
        gf_2035 = demand_growth_factor(iso, 70, 'medium')
        pct_diluted = pct / gf_2035
        grown_emit = compute_total_fossil_emissions_mt(iso, pct_diluted, DEMAND_TWH[iso] * gf_2035)
        print(f"  {iso}: existing={pct:.1f}%  base_emit={base_emit:.1f}Mt"
              f"  @2035(diluted={pct_diluted:.1f}%)={grown_emit:.1f}Mt")

    # Load NEW-resource-only cost from parquets (existing wholesale subtracted)
    print("\nLoading new-resource cost tiers from parquets (P10/P25/P50/P75/P90)...")
    parquet_costs = load_parquet_costs()
    if _PARQUET_COSTS:
        print(f"  Parquet costs loaded for {len(_PARQUET_COSTS)} ISOs — using for MAC computation")
    else:
        print("  WARNING: No parquet costs loaded — falling back to hardcoded CLEAN_COST")

    all_results = {}

    for iso in ISOS:
        print(f"\n{'=' * 70}")
        print(f"  {iso}")
        print(f"{'=' * 70}")

        # Option B: crossover range
        result = compute_crossover_range(iso)
        rng = result['range']           # P25/P75 analysis range
        rng_full = result['range_full']  # P10/P90 display range

        print(f"  Analysis range (P25/P75): {rng.get('lower_bound', '?')}% — {rng.get('upper_bound', '?')}%")
        if rng.get('lower_bound'):
            print(f"    Lower bound ({rng['lower_scenario']})")
            print(f"    Upper bound ({rng['upper_scenario']})")
        print(f"  Display range  (P10/P90): {rng_full.get('lower_bound', '?')}% — {rng_full.get('upper_bound', '?')}%")

        # Print 9 crossover combos
        print(f"\n  {'Grid Cost':<12} {'DAC Scenario':<16} {'Crossover':<12} {'MAC':<10} {'DAC':<10}")
        print(f"  {'-' * 12} {'-' * 16} {'-' * 12} {'-' * 10} {'-' * 10}")
        for key, xo in sorted(result['crossovers'].items()):
            parts = key.split('__')
            grid_tier = parts[0].replace('_grid', '')
            dac_tier = parts[1].replace('_dac', '')
            t_str = f"{xo['threshold']}%" if xo.get('threshold') else '>100%'
            mac_str = f"${xo.get('mac', '?')}" if xo.get('mac') else '—'
            dac_str = f"${xo.get('dac', '?')}" if xo.get('dac') else '—'
            print(f"  {grid_tier:<12} {dac_tier:<16} {t_str:<12} {mac_str:<10} {dac_str:<10}")

        # Demand growth impact on totals
        print(f"\n  Demand growth impact at crossover range:")
        print(f"  {'Growth':<10} {'GF (low T)':<12} {'GF (hi T)':<12} {'Demand lo':<12} {'Demand hi':<12}")
        print(f"  {'-' * 10} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 12}")
        for gt in ['low', 'medium', 'high']:
            if rng.get('lower_bound') and rng.get('upper_bound'):
                gf_lo = demand_growth_factor(iso, rng['lower_bound'], gt)
                gf_hi = demand_growth_factor(iso, rng['upper_bound'], gt)
                d_lo = DEMAND_TWH[iso] * gf_lo
                d_hi = DEMAND_TWH[iso] * gf_hi
                print(f"  {gt:<10} {gf_lo:<12.4f} {gf_hi:<12.4f} {d_lo:<12.1f} {d_hi:<12.1f}")

        # Option A: target-specific analysis (summarize medium only)
        targets = analyze_targets_in_range(
            iso, rng.get('lower_bound'), rng.get('upper_bound'),
            result['mac_curves']
        )
        if targets:
            print(f"\n  Target-specific analysis (medium cost × medium growth):")
            print(f"  {'Thresh':<8} {'Yr':<6} {'MAC':<10} {'DAC':<10} {'Total $M':<10} {'CO₂ Mt':<10} {'Winner'}")
            print(f"  {'-' * 8} {'-' * 6} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 8}")
            for ta in targets:
                if ta['cost_tier'] != 'medium' or ta['growth_tier'] != 'medium':
                    continue
                sm = f"${ta['spline_mac']}" if ta['spline_mac'] else '—'
                winner = 'Grid' if ta.get('grid_cheaper_than_dac_central') else 'DAC'
                cost_str = f"${ta['total_clean_cost_M']:,.0f}" if ta['total_clean_cost_M'] else '—'
                print(f"  {ta['threshold']:<8} {ta['year']:<6} {sm:<10} ${ta['dac_central']:<9} {cost_str:<10} {ta['total_co2_Mt']:<10} {winner}")

        # No-regrets investment analysis
        no_regrets = compute_no_regrets_investments(
            iso, rng.get('lower_bound'), rng.get('upper_bound')
        )

        if isinstance(no_regrets, dict) and 'resources' in no_regrets:
            print(f"\n  NO-REGRETS RESOURCE INVESTMENTS (within crossover range):")
            print(f"  Range thresholds: {no_regrets['range_thresholds']}")
            print(f"\n  {'Resource':<14} {'Consensus':<11} {'Floor%':<9} {'Avg%':<8} {'Floor TWh(M)':<14} {'Avg TWh(M)':<12}")
            print(f"  {'-' * 14} {'-' * 11} {'-' * 9} {'-' * 8} {'-' * 14} {'-' * 12}")
            for res in RESOURCES:
                stats = no_regrets['resources'][res]
                if stats['max_pct'] == 0:
                    continue  # Skip resources not present
                con = '✓' if stats['is_consensus'] else '✗'
                floor_twh = stats['twh_by_growth']['medium']['floor_twh']
                avg_twh = stats['twh_by_growth']['medium']['avg_twh']
                print(f"  {res:<14} {con:<11} {stats['floor_pct']:<9} {stats['avg_pct']:<8} {floor_twh:<14} {avg_twh:<12}")

            # Show demand growth scaling
            print(f"\n  Total clean energy (TWh) by demand growth:")
            for gt in ['low', 'medium', 'high']:
                tc = no_regrets['total_clean_by_growth'][gt]
                print(f"    {gt}: {tc['floor_twh']} – {tc['max_twh']} TWh")

        all_results[iso] = {
            'crossover_range': rng,             # P25/P75 analysis range
            'crossover_range_full': rng_full,   # P10/P90 display range
            'crossovers': result['crossovers'],
            'target_analysis': targets,
            'no_regrets': no_regrets,
            'demand_growth_rates': DEMAND_GROWTH_RATES[iso],
            # Gas backup cost — NOT in MAC, tracked separately for dashboard warning
            'gas_backup_cost_per_mwh': GAS_BACKUP_COST.get(iso, []),
            # Smooth curve for dashboard (medium cost)
            'smooth_curve': {
                'thresholds': result['mac_curves'].get('medium', {}).get('smooth_t', []),
                'marginal_mac': result['mac_curves'].get('medium', {}).get('smooth_mac', []),
            },
            # L/M/H cost sensitivity bands
            'smooth_curves_lmh': {
                tier: {
                    'thresholds': c.get('smooth_t', []),
                    'marginal_mac': c.get('smooth_mac', []),
                }
                for tier, c in result['mac_curves'].items()
            },
            # Discrete data points (medium)
            'discrete_points': {
                'thresholds': result['mac_curves'].get('medium', {}).get('thresholds', []),
                'clean_cost_M': result['mac_curves'].get('medium', {}).get('clean_cost_M', []),
                'co2_Mt': result['mac_curves'].get('medium', {}).get('co2_Mt', []),
                'mac_at_thresholds': result['mac_curves'].get('medium', {}).get('mac_at_thresholds', []),
            },
        }

    # ===== Summary Table =====
    print(f"\n{'=' * 100}")
    print("  OPTIMAL CFE TARGET RANGES PER ISO")
    print(f"  Analysis range = P25/P75 grid cost × 3 DAC scenarios (tighter, actionable)")
    print(f"  Display range  = P10/P90 grid cost × 3 DAC scenarios (wider, uncertainty band)")
    print(f"{'=' * 100}")
    print(f"  {'ISO':<8} {'P25/P75 Lo':<12} {'Central':<10} {'P25/P75 Hi':<12} {'P10/P90':<16} {'Key No-Regrets'}")
    print(f"  {'-' * 8} {'-' * 12} {'-' * 10} {'-' * 12} {'-' * 16} {'-' * 30}")
    for iso in ISOS:
        r = all_results[iso]
        rng = r['crossover_range']
        rng_full = r.get('crossover_range_full', rng)
        lo = f"{rng['lower_bound']}%" if rng.get('lower_bound') else '>100%'
        hi = f"{rng['upper_bound']}%" if rng.get('upper_bound') else '>100%'
        cen_xo = r['crossovers'].get('medium_grid__central_dac', {})
        ce = f"{cen_xo['threshold']}%" if cen_xo.get('threshold') else '>100%'
        full_lo = f"{rng_full['lower_bound']}%" if rng_full.get('lower_bound') else '>100%'
        full_hi = f"{rng_full['upper_bound']}%" if rng_full.get('upper_bound') else '>100%'

        # Summarize no-regrets
        nr = r['no_regrets']
        if isinstance(nr, dict) and 'resources' in nr:
            consensus = [res for res in RESOURCES
                        if nr['resources'][res]['is_consensus'] and nr['resources'][res]['max_pct'] > 0]
            nr_str = ', '.join(consensus) if consensus else 'none'
        else:
            nr_str = 'N/A'
        print(f"  {iso:<8} {lo:<12} {ce:<10} {hi:<12} {full_lo}–{full_hi:<8} {nr_str}")

    # ===== Save =====
    clean = json_clean(all_results)

    json_path = output_dir / 'optimal_targets.json'
    with open(json_path, 'w') as f:
        json.dump(clean, f, indent=2)
    print(f"\nSaved: {json_path}")

    js_path = base_dir / 'dashboard' / 'js' / 'optimal-target-data.js'
    write_dashboard_js(clean, js_path)
    print(f"Saved: {js_path}")

    return all_results


def json_clean(obj):
    """Recursively convert numpy types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: json_clean(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [json_clean(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj) if not np.isnan(obj) else None
    elif isinstance(obj, np.ndarray):
        return [json_clean(v) for v in obj.tolist()]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def write_dashboard_js(results, path):
    """Write dashboard JS with optimal target data + no-regrets investments."""
    lines = [
        '// Auto-generated by step5_compute_optimal_targets.py',
        '// Optimal CFE decarbonization targets: where marginal grid MAC > DAC',
        '// Crossover range = 3 grid cost tiers × 3 DAC scenarios = 9 combos',
        '// No-regrets investments: minimum resource floor across crossover range',
        '',
    ]

    # DAC trajectories for dashboard overlay
    lines.append('const DAC_COST_TRAJECTORIES = ' + json.dumps(DAC_TRAJECTORY, indent=2) + ';')
    lines.append('')

    # Demand growth rates
    lines.append('const DEMAND_GROWTH_RATES_STEP8 = ' + json.dumps(DEMAND_GROWTH_RATES, indent=2) + ';')
    lines.append('')

    lines.append('const OPTIMAL_TARGETS = {')
    for iso in ISOS:
        r = results.get(iso, {})
        rng = r.get('crossover_range', {})
        lines.append(f'  "{iso}": {{')
        lines.append(f'    crossover_range: {json.dumps(rng)},')

        # P10/P90 display range (wider band for interactive dashboard)
        rng_full = r.get('crossover_range_full', rng)
        lines.append(f'    crossover_range_full: {json.dumps(rng_full)},')

        # Medium×central crossover (reference point)
        cen = r.get('crossovers', {}).get('medium_grid__central_dac', {})
        lines.append(f'    crossover_central: {json.dumps(cen)},')

        # All crossovers (5 cost tiers × 3 DAC scenarios)
        lines.append(f'    crossovers: {json.dumps(r.get("crossovers", {}))},')

        # Smooth curves for charting (medium)
        sc = r.get('smooth_curve', {})
        lines.append(f'    smooth_thresholds: {json.dumps(sc.get("thresholds", []))},')
        lines.append(f'    smooth_marginal_mac: {json.dumps(sc.get("marginal_mac", []))},')

        # P10/P25/P50/P75/P90 sensitivity bands
        lmh = r.get('smooth_curves_lmh', {})
        for tier in ['low', 'p25', 'medium', 'p75', 'high']:
            tc = lmh.get(tier, {})
            lines.append(f'    smooth_mac_{tier}: {json.dumps(tc.get("marginal_mac", []))},')

        # DAC overlay at dense thresholds
        if sc.get('thresholds'):
            dac_cen = [round(dac_cost_at_threshold(t, 'central'), 1) for t in sc['thresholds']]
            dac_opt = [round(dac_cost_at_threshold(t, 'optimistic'), 1) for t in sc['thresholds']]
            dac_con = [round(dac_cost_at_threshold(t, 'conservative'), 1) for t in sc['thresholds']]
            lines.append(f'    dac_central: {json.dumps(dac_cen)},')
            lines.append(f'    dac_optimistic: {json.dumps(dac_opt)},')
            lines.append(f'    dac_conservative: {json.dumps(dac_con)},')

        # Discrete data points
        dp = r.get('discrete_points', {})
        lines.append(f'    discrete_thresholds: {json.dumps(dp.get("thresholds", []))},')
        lines.append(f'    discrete_clean_cost_M: {json.dumps(dp.get("clean_cost_M", []))},')
        lines.append(f'    discrete_co2_Mt: {json.dumps(dp.get("co2_Mt", []))},')
        lines.append(f'    discrete_mac: {json.dumps(dp.get("mac_at_thresholds", []))},')

        # Target analysis within range (filtered to medium cost + all growth)
        ta = r.get('target_analysis', [])
        ta_medium = [t for t in ta if t.get('cost_tier') == 'medium']
        lines.append(f'    target_analysis: {json.dumps(ta_medium)},')

        # No-regrets investments
        nr = r.get('no_regrets', {})
        if isinstance(nr, dict) and 'resources' in nr:
            # Simplified no-regrets for dashboard: floor and avg by resource + growth tier
            nr_summary = {}
            for res in RESOURCES:
                stats = nr['resources'][res]
                if stats['max_pct'] == 0:
                    continue
                nr_summary[res] = {
                    'floor_pct': stats['floor_pct'],
                    'avg_pct': stats['avg_pct'],
                    'max_pct': stats['max_pct'],
                    'is_consensus': stats['is_consensus'],
                    'twh_by_growth': stats['twh_by_growth'],
                }
            lines.append(f'    no_regrets: {json.dumps(nr_summary)},')
            lines.append(f'    no_regrets_thresholds: {json.dumps(nr.get("range_thresholds", []))},')
            lines.append(f'    total_clean_by_growth: {json.dumps(nr.get("total_clean_by_growth", {}))},')
        else:
            lines.append(f'    no_regrets: null,')

        # Demand growth rates for this ISO
        lines.append(f'    demand_growth_rates: {json.dumps(r.get("demand_growth_rates", {}))},')

        # Gas backup cost — NOT in MAC, but tracked for dashboard warning overlay
        # "Chasing cheap carbon without considering system needs = retaining/building gas"
        lines.append(f'    gas_backup_cost_per_mwh: {json.dumps(r.get("gas_backup_cost_per_mwh", []))},')

        lines.append(f'  }},')
    lines.append('};')

    with open(path, 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
