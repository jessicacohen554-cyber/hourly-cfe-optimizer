#!/usr/bin/env python3
"""
step6_compute_optimal_targets.py — Optimal CFE targets + no-regrets resource investments

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
  - SYSTEM_COST_DATA (L/M/H) from shared-data.js
  - RESOURCE_MIX_DATA from shared-data.js (medium-cost physics optimization)
  - Emission rates, coal/oil caps, demand from dispatch_utils
  - DAC trajectory from SPEC.md
  - Demand growth rates per ISO × L/M/H

OUTPUTS:
  - data/step5-post-processing/optimal_targets.json
  - dashboard/js/optimal-target-data.js
  (Both consumed by step7_generate_shared_data.py for abatement dashboard)
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

THRESHOLDS = [50, 55, 60, 65, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 99.5, 99.9, 100]
ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']
RESOURCES = ['clean_firm', 'solar', 'wind', 'ccs_ccgt', 'hydro', 'battery', 'battery8', 'ldes']

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
    'CAISO': {'clean_firm': 7.9, 'solar': 22.3, 'wind': 8.8, 'ccs_ccgt': 0, 'hydro': 9.5},
    'ERCOT': {'clean_firm': 8.6, 'solar': 13.8, 'wind': 23.6, 'ccs_ccgt': 0, 'hydro': 0.1},
    'PJM':   {'clean_firm': 32.1, 'solar': 2.9, 'wind': 3.8, 'ccs_ccgt': 0, 'hydro': 1.8},
    'NYISO': {'clean_firm': 18.4, 'solar': 0.0, 'wind': 4.7, 'ccs_ccgt': 0, 'hydro': 15.9},
    'NEISO': {'clean_firm': 23.8, 'solar': 1.4, 'wind': 3.9, 'ccs_ccgt': 0, 'hydro': 4.4},
    'MISO':  {'clean_firm': 13.1, 'solar': 2.1, 'wind': 14.5, 'ccs_ccgt': 0, 'hydro': 1.6},
    'SPP':   {'clean_firm': 5.2, 'solar': 0.4, 'wind': 37.1, 'ccs_ccgt': 0, 'hydro': 4.3},
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
    'optimistic':   {2025: 400, 2030: 200, 2035: 150, 2040: 115, 2045: 90,  2050: 75},
    'central':      {2025: 600, 2030: 350, 2035: 275, 2040: 225, 2045: 200, 2050: 180},
    'conservative': {2025: 800, 2030: 550, 2035: 450, 2040: 375, 2045: 325, 2050: 300},
}

# System cost per threshold ($/MWh) — L/M/H from Step 3 cost optimization
SYSTEM_COST = {
    'medium': {
        'CAISO': [39.38, 42.76, 45.66, 47.79, 50.22, 54.1, 55.02, 59.16, 61.17, 64.77, 66.14, 69.53, 74.19, 79.56, 91.06, 91.06, 91.06],
        'ERCOT': [20.95, 23.81, 26.34, 28.42, 30.39, 33.18, 34.89, 37.83, 38.98, 41.16, 43.59, 46.89, 52.61, 53.62, 57.25, None, 57.25],
        'PJM':   [33.16, 36.85, 39.87, 42.45, 44.83, 47.26, 49.98, 54.06, 53.93, 54.26, 56.89, 59.49, 64.11, 69.94, 94.78, 94.78, 94.78],
        'NYISO': [54.94, 57.19, 60.65, 62.56, 63.89, 64.35, 66.71, 67.44, 67.1, 70.36, 73.27, 77.91, 82.39, 84.88, None, None, None],
        'NEISO': [79.74, 80.82, 81.0, 81.71, 82.77, 83.54, 86.41, 90.09, 91.93, 94.05, 97.52, 100.23, 104.98, 109.84, 122.85, 122.85, 122.85],
        'MISO':  [39.0, 40.98, 42.53, 43.91, 45.31, 46.98, 48.48, 51.72, 54.39, 55.92, 58.58, 61.55, 64.35, 64.4, 72.23, 72.23, 72.23],
        'SPP':   [23.09, 25.57, 27.69, 29.55, 31.07, 32.55, 34.15, 37.02, 38.28, 39.94, 41.91, 44.17, 48.45, 53.64, 58.69, 58.69, 58.69],
    },
    'low': {
        'CAISO': [35.81, 37.66, 39.44, 41.06, 42.66, 45.81, 48.32, 51.51, 51.83, 54.96, 55.62, 58.76, 62.97, 67.83, 75.02, 75.02, 75.02],
        'ERCOT': [19.01, 21.0, 22.84, 24.31, 25.65, 27.74, 29.44, 31.93, 32.52, 34.56, 36.48, 40.26, 41.79, 42.69, 45.43, None, 45.43],
        'PJM':   [29.66, 32.49, 35.09, 37.46, 39.28, 41.13, 43.07, 45.68, 46.2, 46.94, 48.0, 48.89, 52.94, 57.28, 77.39, 77.39, 77.39],
        'NYISO': [50.29, 52.34, 54.13, 55.84, 57.07, 57.75, 58.82, 59.8, 60.07, 60.62, 62.29, 62.88, 64.77, 68.92, None, None, None],
        'NEISO': [71.63, 72.12, 72.3, 73.02, 73.63, 74.77, 74.91, 76.45, 76.72, 77.23, 79.01, 79.6, 83.31, 87.64, 105.62, 105.62, 105.62],
        'MISO':  [33.64, 35.15, 36.35, 37.53, 38.58, 39.86, 41.29, 43.88, 45.87, 47.01, 48.05, 50.89, 52.98, 55.17, 59.32, 59.32, 59.32],
        'SPP':   [21.1, 22.9, 24.5, 25.82, 27.09, 28.26, 29.64, 31.83, 32.69, 34.39, 36.12, 38.33, 41.81, 47.01, 47.25, 47.25, 47.25],
    },
    'high': {
        'CAISO': [48.58, 54.06, 54.22, 56.89, 59.92, 65.2, 67.05, 72.54, 73.82, 77.95, 80.28, 84.74, 92.1, 97.25, 109.41, 109.41, 109.41],
        'ERCOT': [23.32, 27.07, 30.51, 33.28, 35.85, 39.63, 42.91, 46.96, 48.56, 51.48, 54.79, 58.79, 65.48, 68.45, 70.82, None, 70.82],
        'PJM':   [39.06, 44.41, 48.8, 52.58, 55.79, 58.78, 61.69, 68.01, 68.58, 71.47, 73.82, 76.97, 84.09, 91.8, 125.17, 125.17, 125.17],
        'NYISO': [67.12, 70.4, 73.88, 76.45, 78.89, 81.41, 84.85, 86.36, 86.46, 88.89, 90.59, 94.04, 100.85, 103.8, None, None, None],
        'NEISO': [93.4, 95.52, 97.59, 98.72, 100.69, 103.02, 109.5, 114.03, 119.26, 125.07, 129.81, 136.48, 144.52, 155.67, 157.02, 157.02, 157.02],
        'MISO':  [45.68, 48.49, 50.76, 52.92, 54.9, 57.19, 59.96, 62.0, 65.05, 67.97, 72.06, 75.99, 79.75, 82.66, 92.72, 92.72, 92.72],
        'SPP':   [25.36, 28.69, 31.63, 34.05, 36.39, 38.53, 41.11, 44.96, 46.84, 49.8, 52.99, 55.16, 60.92, 67.9, 71.53, 71.53, 71.53],
    },
}

# Resource mix data (% of procurement portfolio) — medium-cost physics optimization
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
        'procurement': [60, 70, 75, 80, 85, 90, 100, 110, 112, 120, 122, 130, 140, 150, 160, 160, 160],
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
        'procurement': [55, 60, 65, 70, 75, 80, 90, 100, 102, 110, 117, 125, 130, 135, 135, 100, 135],
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
        'procurement': [55, 60, 65, 70, 75, 80, 85, 95, 95, 97, 102, 105, 115, 125, 160, 160, 160],
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
        'procurement': [55, 60, 65, 70, 75, 80, 90, 95, 97, 100, 105, 110, 115, 125, 100, 100, 100],
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
        'procurement': [55, 60, 65, 70, 75, 80, 90, 95, 97, 100, 105, 110, 120, 130, 150, 150, 150],
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
        'procurement': [55, 60, 65, 70, 75, 80, 85, 95, 102, 105, 112, 115, 120, 125, 140, 140, 140],
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
        'procurement': [55, 60, 65, 70, 75, 80, 85, 95, 97, 105, 112, 120, 130, 150, 150, 150, 150],
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


# ============================================================================
# CO₂ ABATEMENT MODEL — dispatch_utils canonical path with inline fallback
# ============================================================================

def compute_co2_abated(iso, threshold_pct, growth_tier='medium'):
    """
    Compute total CO₂ abated (million metric tons) at a given CFE threshold.

    Uses dispatch_utils.compute_fossil_retirement() if available (canonical
    dispatch-stack retirement model, same as step6_recompute_co2, step6_compute_mac_stats,
    and step6_compute_lmp_prices). Falls back to inline model otherwise.
    """
    gf = demand_growth_factor(iso, threshold_pct, growth_tier)

    if HAS_DISPATCH_UTILS and _EMISSION_RATES_JSON is not None:
        # ── Canonical path: use dispatch_utils ──
        displaced_rate, info = compute_fossil_retirement(
            iso, threshold_pct, _EMISSION_RATES_JSON, _FOSSIL_MIX_JSON,
            demand_growth_factor=gf
        )
        coal_d = info.get('coal_displaced_twh', 0)
        oil_d = info.get('oil_displaced_twh', 0)
        gas_d = info.get('gas_displaced_twh', 0)
        total_displaced = coal_d + oil_d + gas_d

        # CO₂ in Mt: displaced_rate (tCO₂/MWh) × displaced_TWh × 1e6 MWh/TWh ÷ 1e6 t/Mt = TWh × rate
        co2_mt = total_displaced * displaced_rate

        # Marginal rate: what the next displaced MWh would emit
        if info.get('forced_gas_only', False) or threshold_pct >= COAL_OIL_RETIREMENT_THRESHOLD:
            marginal_rate = info.get('remaining_rate_tco2_mwh', displaced_rate)
        else:
            marginal_rate = displaced_rate

        return {
            'total_co2_mt': co2_mt,
            'marginal_rate': marginal_rate,
            'displaced_twh': total_displaced,
            'retirement_info': info,
            'source': 'dispatch_utils',
        }

    # ── Fallback: inline model ──
    demand_twh = DEMAND_TWH[iso] * gf
    rates = EMISSION_RATES_FALLBACK[iso]
    baseline_clean_pct = sum(GRID_MIX_SHARES[iso].values())

    additional_clean_twh = max(0, (threshold_pct - baseline_clean_pct) / 100.0 * demand_twh)
    if additional_clean_twh < 0.01:
        return {'total_co2_mt': 0.0, 'marginal_rate': rates['gas'], 'displaced_twh': 0.0, 'source': 'inline'}

    coal_cap = COAL_CAP_TWH[iso] * gf
    oil_cap = OIL_CAP_TWH[iso] * gf

    if threshold_pct >= COAL_OIL_RETIREMENT_THRESHOLD:
        coal_displaced = coal_cap
        oil_displaced = oil_cap
        gas_displaced = max(0, additional_clean_twh - coal_cap - oil_cap)
    else:
        fossil_pct = max(0, 100.0 - threshold_pct)
        fossil_twh = demand_twh * fossil_pct / 100.0
        coal_current = min(coal_cap, fossil_twh)
        oil_current = min(oil_cap, max(0, fossil_twh - coal_current))
        coal_displaced = min(additional_clean_twh, coal_current)
        remaining = additional_clean_twh - coal_displaced
        oil_displaced = min(remaining, oil_current)
        remaining -= oil_displaced
        gas_displaced = max(0, remaining)

    co2 = (coal_displaced * rates['coal'] + oil_displaced * rates['oil'] + gas_displaced * rates['gas'])
    total_displaced = coal_displaced + oil_displaced + gas_displaced

    if threshold_pct >= COAL_OIL_RETIREMENT_THRESHOLD:
        marginal_rate = rates['gas']
    elif coal_displaced < coal_cap:
        marginal_rate = rates['coal']
    elif oil_displaced < oil_cap:
        marginal_rate = rates['oil']
    else:
        marginal_rate = rates['gas']

    return {
        'total_co2_mt': co2,
        'marginal_rate': marginal_rate,
        'displaced_twh': total_displaced,
        'source': 'inline_fallback',
    }


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
    """Isotonic regression: enforce monotonically non-decreasing."""
    result = arr.copy()
    for i in range(1, len(result)):
        if result[i] < result[i - 1]:
            result[i] = result[i - 1]
    return result


def compute_marginal_mac_curve(iso, cost_tier='medium', growth_tier='medium'):
    """
    Compute smooth marginal MAC curve for one ISO × one cost × one growth scenario.

    Note: Marginal MAC ($/tCO₂) is scale-invariant w.r.t. demand growth because
    both d(cost) and d(CO₂) scale by the same growth factor. The growth_tier
    parameter is included for completeness in the total cost/CO₂ outputs,
    but the MAC curve shape and crossover points are identical across growth tiers.
    """
    costs = SYSTEM_COST[cost_tier][iso]
    wholesale = WHOLESALE_PRICES[iso]

    # Build arrays, skipping nulls
    valid_t, valid_cost_premium, valid_co2 = [], [], []
    for i, t in enumerate(THRESHOLDS):
        sc = costs[i]
        if sc is None:
            continue
        # Growth factor at this threshold
        gf = demand_growth_factor(iso, t, growth_tier)
        demand_twh = DEMAND_TWH[iso] * gf
        # Cost premium: $/MWh above wholesale × growth-adjusted demand = $M/year
        premium = max(0, sc - wholesale) * demand_twh
        co2 = compute_co2_abated(iso, t, growth_tier)['total_co2_mt']

        valid_t.append(t)
        valid_cost_premium.append(premium)
        valid_co2.append(co2)

    if len(valid_t) < 4:
        return None

    t_arr = np.array(valid_t)
    cost_arr = np.array(valid_cost_premium)
    co2_arr = np.array(valid_co2)

    # Enforce monotonicity
    cost_mono = enforce_monotonic(cost_arr)
    co2_mono = enforce_monotonic(co2_arr)

    if HAS_SCIPY:
        cost_spline = PchipInterpolator(t_arr, cost_mono)
        co2_spline = PchipInterpolator(t_arr, co2_mono)

        t_dense = np.linspace(float(t_arr[0]), float(t_arr[-1]), 300)
        dcost = cost_spline.derivative()(t_dense)
        dco2 = co2_spline.derivative()(t_dense)
        marginal_mac = np.where(dco2 > 1e-6, dcost / dco2, np.nan)

        mac_at_t = []
        for t in valid_t:
            dc = float(cost_spline.derivative()(t))
            dq = float(co2_spline.derivative()(t))
            mac_at_t.append(dc / dq if dq > 1e-6 else None)
    else:
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
        'cost_premium_M': [float(x) for x in cost_mono],
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
    Cross 3 grid cost tiers × 3 DAC scenarios = 9 crossover points.
    Returns the full range + the medium×central as the reference.

    Note: Demand growth doesn't change crossover thresholds (MAC is scale-invariant),
    so we compute crossovers at medium growth only. Growth affects absolute quantities,
    which are computed separately in the no-regrets analysis.
    """
    crossovers = {}
    mac_curves = {}

    for cost_tier in ['low', 'medium', 'high']:
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

    # Extract range
    valid_thresholds = [
        xo['threshold'] for xo in crossovers.values()
        if xo.get('threshold') is not None
    ]

    if not valid_thresholds:
        range_result = {
            'lower_bound': None,
            'upper_bound': None,
            'note': 'Grid cheaper than DAC in all scenarios',
        }
    else:
        range_result = {
            'lower_bound': min(valid_thresholds),
            'upper_bound': max(valid_thresholds),
            'lower_scenario': next(
                k for k, v in crossovers.items() if v.get('threshold') == min(valid_thresholds)
            ),
            'upper_scenario': next(
                k for k, v in crossovers.items() if v.get('threshold') == max(valid_thresholds)
            ),
        }

    return {
        'crossovers': crossovers,
        'range': range_result,
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
                    dcost = curve['cost_premium_M'][idx] - curve['cost_premium_M'][idx - 1]
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
                sys_cost = SYSTEM_COST[cost_tier][iso][THRESHOLDS.index(t)]
                total_cost_M = (max(0, sys_cost - WHOLESALE_PRICES[iso]) * demand_twh) if sys_cost else None
                total_co2 = compute_co2_abated(iso, t, growth_tier)['total_co2_mt']

                results.append({
                    'threshold': t,
                    'year': round(threshold_to_year(t)),
                    'cost_tier': cost_tier,
                    'growth_tier': growth_tier,
                    'growth_factor': round(gf, 4),
                    'demand_twh': round(demand_twh, 1),
                    'system_cost_mwh': sys_cost,
                    'total_cost_premium_M': round(total_cost_M, 1) if total_cost_M else None,
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
    Resource TWh = (resource_pct / 100) × (procurement_pct / 100) × demand × gf
    """
    mix = RESOURCE_MIX_DATA[iso]
    procurement_pct = mix['procurement'][threshold_idx]
    t = THRESHOLDS[threshold_idx]
    gf = demand_growth_factor(iso, t, growth_tier)
    demand = DEMAND_TWH[iso] * gf

    result = {}
    for res in RESOURCES:
        res_pct = mix[res][threshold_idx]
        twh = (res_pct / 100.0) * (procurement_pct / 100.0) * demand
        result[res] = round(twh, 2)
    result['procurement_pct'] = procurement_pct
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
        procurement_values = [mix['procurement'][i] for i in range_indices]

        # Effective share of demand = (res_pct/100) × (procurement_pct/100)
        demand_shares = [(pct / 100.0) * (proc / 100.0)
                         for pct, proc in zip(pct_values, procurement_values)]

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
                res_twh = (mix[res][i] / 100.0) * (mix['procurement'][i] / 100.0) * demand
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
                    'procurement': mix['procurement'][i],
                }
                for i in range_indices
            ],
        }

    # Total procurement stats across range
    procurement_values = [mix['procurement'][i] for i in range_indices]
    total_clean_by_growth = {}
    for growth_tier in ['low', 'medium', 'high']:
        total_twh_values = []
        for i in range_indices:
            t = THRESHOLDS[i]
            gf = demand_growth_factor(iso, t, growth_tier)
            demand = DEMAND_TWH[iso] * gf
            total_twh = (mix['procurement'][i] / 100.0) * demand
            total_twh_values.append(total_twh)
        total_clean_by_growth[growth_tier] = {
            'floor_twh': round(min(total_twh_values), 1),
            'avg_twh': round(sum(total_twh_values) / len(total_twh_values), 1),
            'max_twh': round(max(total_twh_values), 1),
        }

    return {
        'range_thresholds': range_thresholds,
        'resources': resource_stats,
        'procurement_pct_range': [min(procurement_values), max(procurement_values)],
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

    all_results = {}

    for iso in ISOS:
        print(f"\n{'=' * 70}")
        print(f"  {iso}")
        print(f"{'=' * 70}")

        # Option B: crossover range
        result = compute_crossover_range(iso)
        rng = result['range']

        print(f"  Crossover range: {rng.get('lower_bound', '?')}% — {rng.get('upper_bound', '?')}%")
        if rng.get('lower_bound'):
            print(f"    Lower bound ({rng['lower_scenario']})")
            print(f"    Upper bound ({rng['upper_scenario']})")

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
                cost_str = f"${ta['total_cost_premium_M']:,.0f}" if ta['total_cost_premium_M'] else '—'
                print(f"  {ta['threshold']:<8} {ta['year']:<6} {sm:<10} ${ta['dac_central']:<9} {cost_str:<10} {ta['total_co2_Mt']:<10} {winner}")

        # No-regrets investment analysis
        no_regrets = compute_no_regrets_investments(
            iso, rng.get('lower_bound'), rng.get('upper_bound')
        )

        if isinstance(no_regrets, dict) and 'resources' in no_regrets:
            print(f"\n  NO-REGRETS RESOURCE INVESTMENTS (within crossover range):")
            print(f"  Range thresholds: {no_regrets['range_thresholds']}")
            print(f"  Procurement: {no_regrets['procurement_pct_range'][0]}% – {no_regrets['procurement_pct_range'][1]}% of demand")
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
            'crossover_range': rng,
            'crossovers': result['crossovers'],
            'target_analysis': targets,
            'no_regrets': no_regrets,
            'demand_growth_rates': DEMAND_GROWTH_RATES[iso],
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
                'cost_premium_M': result['mac_curves'].get('medium', {}).get('cost_premium_M', []),
                'co2_Mt': result['mac_curves'].get('medium', {}).get('co2_Mt', []),
                'mac_at_thresholds': result['mac_curves'].get('medium', {}).get('mac_at_thresholds', []),
            },
        }

    # ===== Summary Table =====
    print(f"\n{'=' * 90}")
    print("  OPTIMAL CFE TARGET RANGES PER ISO")
    print(f"  (where marginal grid MAC crosses DAC across all scenario combos)")
    print(f"{'=' * 90}")
    print(f"  {'ISO':<8} {'Lower':<10} {'Central':<10} {'Upper':<10} {'Key No-Regrets Investments'}")
    print(f"  {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 40}")
    for iso in ISOS:
        r = all_results[iso]
        rng = r['crossover_range']
        lo = f"{rng['lower_bound']}%" if rng.get('lower_bound') else '>100%'
        hi = f"{rng['upper_bound']}%" if rng.get('upper_bound') else '>100%'
        cen_xo = r['crossovers'].get('medium_grid__central_dac', {})
        ce = f"{cen_xo['threshold']}%" if cen_xo.get('threshold') else '>100%'

        # Summarize no-regrets
        nr = r['no_regrets']
        if isinstance(nr, dict) and 'resources' in nr:
            consensus = [res for res in RESOURCES
                        if nr['resources'][res]['is_consensus'] and nr['resources'][res]['max_pct'] > 0]
            nr_str = ', '.join(consensus) if consensus else 'none'
        else:
            nr_str = 'N/A'
        print(f"  {iso:<8} {lo:<10} {ce:<10} {hi:<10} {nr_str}")

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
        '// Auto-generated by step6_compute_optimal_targets.py',
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

        # Medium×central crossover (reference point)
        cen = r.get('crossovers', {}).get('medium_grid__central_dac', {})
        lines.append(f'    crossover_central: {json.dumps(cen)},')

        # All 9 crossovers
        lines.append(f'    crossovers: {json.dumps(r.get("crossovers", {}))},')

        # Smooth curves for charting (medium)
        sc = r.get('smooth_curve', {})
        lines.append(f'    smooth_thresholds: {json.dumps(sc.get("thresholds", []))},')
        lines.append(f'    smooth_marginal_mac: {json.dumps(sc.get("marginal_mac", []))},')

        # L/M/H sensitivity band
        lmh = r.get('smooth_curves_lmh', {})
        for tier in ['low', 'medium', 'high']:
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
        lines.append(f'    discrete_cost_premium_M: {json.dumps(dp.get("cost_premium_M", []))},')
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
            lines.append(f'    no_regrets_procurement_range: {json.dumps(nr.get("procurement_pct_range", []))},')
            lines.append(f'    total_clean_by_growth: {json.dumps(nr.get("total_clean_by_growth", {}))},')
        else:
            lines.append(f'    no_regrets: null,')

        # Demand growth rates for this ISO
        lines.append(f'    demand_growth_rates: {json.dumps(r.get("demand_growth_rates", {}))},')

        lines.append(f'  }},')
    lines.append('};')

    with open(path, 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
