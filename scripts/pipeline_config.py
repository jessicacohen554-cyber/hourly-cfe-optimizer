#!/usr/bin/env python3
"""
Pipeline Configuration — Single Source of Truth
=================================================
All shared constants, schemas, and data contracts for Steps 1–7.

Every pipeline script MUST import constants from here instead of defining
its own copies. This prevents the class of bugs where step3 and dispatch_utils
disagree on CCS_CAP_TWH (which happened — see git history).

Usage:
    from pipeline_config import (
        ISOS, REGIONAL_DEMAND_TWH, THRESHOLDS, ACTIVE_THRESHOLDS,
        CCS_CAP_TWH, OFFSHORE_ISOS, WHOLESALE_PRICES, ...
    )

Version: 1.0.0
Last updated: 2026-03-03
"""

from collections import OrderedDict
import numpy as np

# ============================================================================
# VERSION & METADATA
# ============================================================================

PIPELINE_VERSION = "1.0.0"
BASE_YEAR = 2025
MODEL_TYPE = "snapshot"  # 2025 snapshot model (no forward projections in Track 1)

# ============================================================================
# REGIONS
# ============================================================================

ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']

# ISOs with offshore wind resource
OFFSHORE_ISOS = ['NYISO', 'NEISO', 'PJM', 'CAISO']

# ISOs with geothermal resource
GEOTHERMAL_ISOS = ['CAISO']

# Resource dimensionality per ISO
# Interior ISOs (ERCOT, MISO, SPP): 4D
# Offshore ISOs (NYISO, NEISO, PJM): 5D (+ offshore_wind)
# CAISO: 6D (+ offshore_wind + geothermal)
ISO_DIMENSIONS = {
    'CAISO': 6, 'ERCOT': 4, 'PJM': 5,
    'NYISO': 5, 'NEISO': 5, 'MISO': 4, 'SPP': 4,
}

RESOURCE_COLS_BASE = ['clean_firm', 'solar', 'wind', 'hydro']
RESOURCE_COLS_OFFSHORE = ['clean_firm', 'solar', 'wind', 'hydro', 'offshore_wind']
RESOURCE_COLS_CAISO = ['clean_firm', 'solar', 'wind', 'hydro', 'offshore_wind', 'geothermal']

STORAGE_COLS = ['battery_dispatch_pct', 'battery8_dispatch_pct', 'ldes_dispatch_pct', 'h2_dispatch_pct']

def get_resource_cols(iso):
    """Return resource column names for the given ISO."""
    if iso == 'CAISO':
        return RESOURCE_COLS_CAISO
    elif iso in OFFSHORE_ISOS:
        return RESOURCE_COLS_OFFSHORE
    else:
        return RESOURCE_COLS_BASE

# ============================================================================
# THRESHOLDS
# ============================================================================

# All 21 thresholds used across the pipeline
THRESHOLDS = [
    10.0, 20.0, 30.0, 40.0,          # Coarse range (no step1d, no fine zone)
    50.0, 55.0, 60.0, 65.0,          # Mid range (5% steps)
    70.0, 75.0, 80.0,                # Upper-mid range
    85.0, 87.5, 90.0, 92.5,          # Inflection zone (2.5% steps)
    95.0, 97.5,                       # High range
    99.0, 99.5, 99.9, 99.99,         # Last mile
]

# Active thresholds (full pipeline coverage with step1d storage refinement)
ACTIVE_THRESHOLDS = [t for t in THRESHOLDS if t >= 50.0]  # 17 thresholds

# Coarse-only thresholds (no step1d, no fine zone search)
COARSE_THRESHOLDS = [t for t in THRESHOLDS if t < 50.0]   # 4 thresholds

THRESHOLD_SET = set(THRESHOLDS)

# ============================================================================
# REGIONAL DEMAND (TWh, 2025 base year)
# ============================================================================
# Source: EIA-930 hourly generation data, 2024 annualized and adjusted
# for 2025 growth trends. See SPEC.md §2.1 for methodology.

REGIONAL_DEMAND_TWH = {
    'CAISO': 224.039,
    'ERCOT': 488.020,
    'PJM':   843.331,
    'NYISO': 151.599,
    'NEISO': 115.336,
    'MISO':  660.000,
    'SPP':   296.000,
}

# ============================================================================
# EXISTING GRID MIX (% of demand, 2025 baseline)
# ============================================================================
# Source: eGRID 2022 subregion-to-ISO mapping, cross-validated with
# EIA-930 2024 hourly generation by fuel type.

GRID_MIX_SHARES = {
    'CAISO': {'clean_firm': 7.9, 'solar': 22.3, 'wind': 8.8, 'offshore_wind': 0, 'ccs_ccgt': 0, 'hydro': 9.5},
    'ERCOT': {'clean_firm': 8.6, 'solar': 13.8, 'wind': 23.6, 'offshore_wind': 0, 'ccs_ccgt': 0, 'hydro': 0.1},
    'PJM':   {'clean_firm': 32.1, 'solar': 2.9, 'wind': 3.8, 'offshore_wind': 0, 'ccs_ccgt': 0, 'hydro': 1.8},
    'NYISO': {'clean_firm': 18.4, 'solar': 0.0, 'wind': 4.7, 'offshore_wind': 0, 'ccs_ccgt': 0, 'hydro': 15.9},
    'NEISO': {'clean_firm': 23.8, 'solar': 1.4, 'wind': 3.9, 'offshore_wind': 0, 'ccs_ccgt': 0, 'hydro': 4.4},
    'MISO':  {'clean_firm': 13.1, 'solar': 2.1, 'wind': 14.5, 'offshore_wind': 0, 'ccs_ccgt': 0, 'hydro': 1.6},
    'SPP':   {'clean_firm': 5.2, 'solar': 0.4, 'wind': 37.1, 'offshore_wind': 0, 'ccs_ccgt': 0, 'hydro': 4.3},
}

# ============================================================================
# WHOLESALE ELECTRICITY PRICES ($/MWh)
# ============================================================================
# Source: EIA-930 + ISO annual market reports, 2024 weighted average DA LMP.

WHOLESALE_PRICES = {
    'CAISO': 30, 'ERCOT': 27, 'PJM': 34,
    'NYISO': 42, 'NEISO': 41, 'MISO': 30, 'SPP': 25,
}

# Fossil fuel price adjustments ($/MWh delta from base wholesale)
FUEL_ADJUSTMENTS = {
    'CAISO': {'Low': -5, 'Medium': 0, 'High': 10},
    'ERCOT': {'Low': -7, 'Medium': 0, 'High': 12},
    'PJM':   {'Low': -6, 'Medium': 0, 'High': 11},
    'NYISO': {'Low': -4, 'Medium': 0, 'High': 8},
    'NEISO': {'Low': -4, 'Medium': 0, 'High': 8},
    'MISO':  {'Low': -6, 'Medium': 0, 'High': 11},
    'SPP':   {'Low': -7, 'Medium': 0, 'High': 12},
}

# ============================================================================
# RESOURCE CAPACITY CAPS (TWh/yr)
# ============================================================================

# CCS-CCGT regional capacity caps — geologic CO2 storage availability
# Mixes with CCS deployment exceeding these caps are filtered out.
# Source: NETL Carbon Storage Atlas V, EPA Class VI well applications,
# DOE Regional Carbon Sequestration Partnerships.
CCS_CAP_TWH = {
    'CAISO': 25.0,    # 11% of demand — limited geology, regulatory barriers
    'ERCOT': 200.0,   # 41% of demand — Gulf Coast saline formations
    'PJM':   125.0,   # 15% of demand — Appalachian basin, east-west split
    'NYISO': 0.0,     # Hard zero — crystalline bedrock, no geologic storage
    'NEISO': 0.0,     # Hard zero — crystalline bedrock, no geologic storage
    'MISO':  200.0,   # 30% of demand — Illinois/Michigan basins
    'SPP':   50.0,    # 17% of demand — Anadarko basin, seismicity risk
}

# Offshore wind capacity caps (TWh/yr) — pipeline-derived
# Source: BOEM lease areas, state-level mandates, interconnection queue.
OFFSHORE_WIND_CAP_TWH = {
    'NYISO': 37.0,    # Empire Wind I+II, Sunrise Wind, Beacon Wind
    'NEISO': 37.0,    # Vineyard Wind, Revolution Wind, SouthCoast
    'PJM':   30.0,    # NJ 7.5 GW mandate, DE/MD/VA pipeline
    'CAISO': 20.0,    # Morro Bay, Humboldt (floating)
}

# Geothermal capacity cap (TWh/yr) — CAISO only
# Source: USGS assessment + Fervo EGS potential
GEOTHERMAL_CAP_TWH = 39.0

# Hydro caps: existing only, region-dependent
HYDRO_CAP_PCT = {
    'CAISO': 10.5, 'ERCOT': 0.5, 'PJM': 2.0,
    'NYISO': 17.5, 'NEISO': 5.0, 'MISO': 2.0, 'SPP': 5.0,
}

# Solar and total procurement caps
SOLAR_CAP_PCT = 100
TOTAL_PROCUREMENT_CAP_PCT = 350

# ============================================================================
# STORAGE PARAMETERS
# ============================================================================

# Battery (4-hour Li-ion)
BATTERY_EFFICIENCY = 0.85
BATTERY_DURATION_HOURS = 4

# Battery (8-hour Li-ion)
BATTERY8_EFFICIENCY = 0.85
BATTERY8_DURATION_HOURS = 8

# LDES (100-hour iron-air)
LDES_EFFICIENCY = 0.50
LDES_DURATION_HOURS = 100
LDES_WINDOW_DAYS = 7

# Green H2 (1000-hour electrolysis + salt cavern + H2 turbine)
H2_EFFICIENCY = 0.35
H2_DURATION_HOURS = 1000
H2_WINDOW_DAYS = 30
H2_MIN_THRESHOLD = 95.0  # Only available at ≥95% thresholds

# Storage dispatch grid resolution (% of annual demand)
STORAGE_FINE_RESOLUTION = 0.001  # 0.001% of annual demand resolution

# Storage dispatch maxima (% of annual demand — energy capacity as fraction of annual demand)
# Physical reference: CAISO 224 TWh → 0.01% = 22,400 MWh / 5,600 MW (4hr)
STORAGE_MAX = {
    'battery': 0.06,    # 0.06% of annual demand — CAISO: ~134 GWh / 33.6 GW
    'battery8': 0.08,   # 0.08% of annual demand
    'ldes': 0.5,        # 0.5% of annual demand
    'h2': 2.0,          # 2.0% of annual demand
}

# Step 1D.2: Research-informed 2050 storage caps (% of annual demand)
# Sources: NREL Storage Futures (200 GW / 1,200 GWh ref case),
#          DOE LDES Liftoff (225-460 GW LDES for net-zero),
#          Princeton Net Zero America (1,300 GWh by 2050)
STORAGE_MAX_V2 = {
    'battery': 0.10,    # 0.10% of annual demand — CAISO: ~224 GWh / 56 GW
    'battery8': 0.15,   # 0.15% of annual demand
    'ldes': 1.0,        # 1.0% of annual demand
    'h2': 3.0,          # 3.0% of annual demand
}

# ============================================================================
# STORAGE ECONOMICS (Step 1D.2 Economic Assessment)
# ============================================================================

# Capacity market prices ($/kW-yr) — from 2024-2025 auction results
# ERCOT and SPP have energy-only markets (no capacity payment)
CAPACITY_MARKET_PRICES = {
    'CAISO': 75,    # RA program, system-wide avg
    'ERCOT': 0,     # No capacity market (energy-only)
    'PJM': 120,     # RPM 2025/2026-2027/2028 BRA clearing ($98-269/MW-day, avg ~$120/kW-yr)
    'NYISO': 85,    # ICAP monthly spot, annualized
    'NEISO': 55,    # FCM FCA-19 clearing price
    'MISO': 25,     # PRA Zone 1-7 average
    'SPP': 0,       # No capacity market (energy-only)
}

# Ancillary service rates ($/MW-hr) by product × ISO
# Battery eligible for regulation (fast response); LDES for spinning only
ANCILLARY_SERVICE_RATES = {
    'regulation': {
        'CAISO': 12, 'ERCOT': 15, 'PJM': 18, 'NYISO': 14,
        'NEISO': 10, 'MISO': 8, 'SPP': 6,
    },
    'spinning': {
        'CAISO': 5, 'ERCOT': 8, 'PJM': 6, 'NYISO': 5,
        'NEISO': 4, 'MISO': 3, 'SPP': 3,
    },
}

# Ancillary service availability (hours/year available for service)
ANCILLARY_HOURS = {
    'regulation': 2000,  # ~23% of year — battery available when not cycling
    'spinning': 4000,    # ~46% of year — can provide while partially charged
}

# Storage arbitrage revenue ($/kW-yr) — from ISO LMP price spreads
# Battery: daily cycling captures peak-to-trough spread (duck curve, evening ramp)
# LDES: weekly cycling captures workday/weekend and multi-day weather patterns
# H2: seasonal cycling, minimal arbitrage value
# Sources: LBNL Utility-Scale Solar/Storage 2024, Lazard LCOS v9, ISO market reports
STORAGE_ARBITRAGE_REVENUE = {
    'battery': {
        'CAISO': 50,   # Duck curve compressed 2024-25 w/ 10GW+ storage ($30-55/MWh spread)
        'ERCOT': 50,   # Volatility moderated, 2024-25 spreads lower than 2022-23 peaks
        'PJM':   45,   # Moderate day/night spread
        'NYISO': 55,   # Moderate, higher in NYC/LI zones
        'NEISO': 35,   # Modest spreads
        'MISO':  30,   # Low-moderate spreads
        'SPP':   25,   # Lowest price differentials
    },
    'battery8': {  # ~85% of bat4 per kW (more energy shifted, lower peak capture)
        'CAISO': 43,
        'ERCOT': 43,
        'PJM':   38,
        'NYISO': 47,
        'NEISO': 30,
        'MISO':  26,
        'SPP':   21,
    },
    'ldes': {  # Weekly cycling — captures multi-day weather patterns
        'CAISO': 15,
        'ERCOT': 20,
        'PJM':   12,
        'NYISO': 14,
        'NEISO': 10,
        'MISO':   8,
        'SPP':    7,
    },
    'h2': {  # Seasonal cycling — minimal arbitrage
        'CAISO': 5,
        'ERCOT': 6,
        'PJM':   4,
        'NYISO': 5,
        'NEISO': 3,
        'MISO':  3,
        'SPP':   2,
    },
}

# Revenue stacking factor — can't simultaneously do arbitrage + ancillary in same hour
# Capacity payments are always earned (availability-based), so no stacking limit
# Arbitrage and ancillary compete for the same hours → 70% co-optimization efficiency
REVENUE_STACKING_FACTOR = 0.70

# Ancillary service product eligibility by storage type
# Battery: regulation (fast response); LDES: spinning reserve
STORAGE_ANCILLARY_PRODUCT = {
    'battery': 'regulation',
    'battery8': 'regulation',
    'ldes': 'spinning',
    'h2': 'spinning',  # H2 too slow for regulation, marginal spinning value
}


def compute_storage_revenue_credit(storage_type, iso):
    """Compute net revenue credit in LCOE units (same as LCOE_TABLES) for a storage type.

    Revenue credit = capacity_payment + (arbitrage + ancillary) × stacking_factor
    Converted to LCOE units: credit_lcoe = 1000 × total_$/kW-yr ÷ duration_hr

    Returns the credit value to SUBTRACT from gross LCOE.
    """
    durations = {'battery': 4, 'battery8': 8, 'ldes': 100, 'h2': 1000}

    capacity = CAPACITY_MARKET_PRICES.get(iso, 0)

    # Ancillary revenue
    product = STORAGE_ANCILLARY_PRODUCT[storage_type]
    anc_rate = ANCILLARY_SERVICE_RATES[product].get(iso, 0)
    anc_hours = ANCILLARY_HOURS[product]
    ancillary_kw_yr = anc_rate * anc_hours / 1000  # $/MW-hr → $/kW-yr

    # Arbitrage revenue
    arbitrage = STORAGE_ARBITRAGE_REVENUE[storage_type].get(iso, 0)

    # Stack: capacity is always earned; arbitrage + ancillary compete
    total_kw_yr = capacity + (arbitrage + ancillary_kw_yr) * REVENUE_STACKING_FACTOR

    # Convert to LCOE units: $/kW-yr → same units as LCOE_TABLES
    duration = durations[storage_type]
    credit_lcoe = 1000.0 * total_kw_yr / duration

    return credit_lcoe


# Pre-compute revenue credits for all types × ISOs
STORAGE_REVENUE_CREDITS = {}
for _stype in ['battery', 'battery8', 'ldes', 'h2']:
    STORAGE_REVENUE_CREDITS[_stype] = {}
    for _iso in ISOS:
        STORAGE_REVENUE_CREDITS[_stype][_iso] = compute_storage_revenue_credit(_stype, _iso)

# Battery degradation parameters
BATTERY_DEGRADATION = {
    'battery': {
        'cycles_per_year': 365,      # Daily cycling
        'cycle_life_80pct': 5000,    # Cycles to 80% capacity (Li-ion NMC/LFP)
        'replacement_fraction': 0.40, # Augmentation cost as fraction of original CAPEX
    },
    'battery8': {
        'cycles_per_year': 365,
        'cycle_life_80pct': 4000,    # Deeper discharge → faster degradation
        'replacement_fraction': 0.45,
    },
    'ldes': {
        'cycles_per_year': 52,       # Weekly cycling
        'cycle_life_80pct': 20000,   # Iron-air: minimal degradation
        'replacement_fraction': 0.15,
    },
    'h2': {
        'cycles_per_year': 12,       # Monthly/seasonal cycling
        'cycle_life_80pct': 50000,   # Electrolysis stack replacement is main cost
        'replacement_fraction': 0.25,
    },
}

# ============================================================================
# RESOURCE ADEQUACY
# ============================================================================

RESOURCE_ADEQUACY_MARGIN = 0.15  # 15% reserve margin

PEAK_DEMAND_MW = {
    'CAISO': 43860, 'ERCOT': 83597, 'PJM': 160560,
    'NYISO': 31857, 'NEISO': 25898, 'MISO': 127125, 'SPP': 54368,
}

EXISTING_GAS_CAPACITY_MW = {
    'CAISO': 37000, 'ERCOT': 55000, 'PJM': 75000,
    'NYISO': 18000, 'NEISO': 14000, 'MISO': 68000, 'SPP': 32000,
}

GAS_AVAILABILITY_FACTOR = {
    'CAISO': 0.88, 'ERCOT': 0.83, 'PJM': 0.82,
    'NYISO': 0.82, 'NEISO': 0.85, 'MISO': 0.84, 'SPP': 0.84,
}

PEAK_CAPACITY_CREDITS = {
    'clean_firm': 1.0, 'solar': 0.30, 'wind': 0.10,
    'ccs_ccgt': 0.90, 'hydro': 0.50, 'battery': 0.95,
    'battery8': 0.95, 'ldes': 0.90, 'h2': 0.85,
    'offshore_wind': 0.25,
}

RESOURCE_CAPACITY_FACTORS = {
    'clean_firm': {'CAISO': 0.90, 'ERCOT': 0.93, 'PJM': 0.93, 'NYISO': 0.90, 'NEISO': 0.90, 'MISO': 0.92, 'SPP': 0.92},
    'solar':      {'CAISO': 0.28, 'ERCOT': 0.24, 'PJM': 0.17, 'NYISO': 0.15, 'NEISO': 0.15, 'MISO': 0.19, 'SPP': 0.22},
    'wind':       {'CAISO': 0.25, 'ERCOT': 0.38, 'PJM': 0.30, 'NYISO': 0.28, 'NEISO': 0.30, 'MISO': 0.36, 'SPP': 0.42},
    'ccs_ccgt':   {'CAISO': 0.85, 'ERCOT': 0.85, 'PJM': 0.85, 'NYISO': 0.85, 'NEISO': 0.85, 'MISO': 0.85, 'SPP': 0.85},
    'hydro':      {'CAISO': 0.40, 'ERCOT': 0.30, 'PJM': 0.35, 'NYISO': 0.40, 'NEISO': 0.40, 'MISO': 0.35, 'SPP': 0.30},
    'offshore_wind': {'CAISO': 0.43, 'ERCOT': 0.35, 'PJM': 0.48, 'NYISO': 0.49, 'NEISO': 0.51, 'MISO': 0.35, 'SPP': 0.35},
}

# ============================================================================
# EMISSION RATES & CO2 MODEL
# ============================================================================

# CCS residual emission rate (tCO2/MWh) — 90% capture rate
CCS_RESIDUAL_EMISSION_RATE = 0.037

# Coal/oil retirement threshold (% clean energy)
# Above this threshold, coal and oil are fully retired from the fossil fleet
COAL_OIL_RETIREMENT_THRESHOLD = 70.0

# Dispatch cache version
DISPATCH_CACHE_VERSION = 3

# Dispatch order (merit order for hourly matching)
DISPATCH_ORDER = ['clean_firm', 'ccs_ccgt', 'hydro', 'offshore_wind', 'wind', 'solar']

# ============================================================================
# GAS/CCS ADJUSTMENTS (Step 4)
# ============================================================================

# NEISO Winter Gas Pipeline Constraint (Algonquin Citygates)
# During winter (~25% of year), gas pipeline capacity is constrained,
# driving gas prices $7.50/MMBtu above Henry Hub.
NEISO_CCS_GAS_ADDER = 13.13    # $/MWh annualized CCS adder (7 HR × $7.50 × 0.25)
NEISO_WHOLESALE_ADDER = 4.0    # $/MWh annualized wholesale adder

# 45Q Tax Credit for CCS
# $85/ton × 0.323 tCO2/MWh captured (90% capture × 0.359 tCO2/MWh unabated)
CCS_45Q_CREDIT_PER_MWH = 27.5  # $/MWh offset between 45Q ON and OFF tables

# ============================================================================
# NUCLEAR PARAMETERS
# ============================================================================

EXISTING_NUCLEAR_GW = {
    'CAISO': 2.3, 'ERCOT': 2.7, 'PJM': 32.0,
    'NYISO': 3.4, 'NEISO': 3.5, 'MISO': 12.0, 'SPP': 1.2,
}

# Uprate cap: 8% of existing nuclear × 90% CF → TWh/yr
UPRATE_FRACTION = 0.08
UPRATE_CF = 0.90
UPRATE_CAP_TWH = {
    iso: round(gw * UPRATE_FRACTION * UPRATE_CF * 8760 / 1e3, 3)
    for iso, gw in EXISTING_NUCLEAR_GW.items()
}

# Nuclear monthly capacity factors (seasonal derate)
NUCLEAR_MONTHLY_CF = {
    'CAISO': {1: 0.94, 2: 0.94, 3: 0.85, 4: 0.75, 5: 0.80, 6: 0.99,
              7: 1.0, 8: 1.0, 9: 0.90, 10: 0.78, 11: 0.82, 12: 0.94},
    'ERCOT': {m: 0.93 for m in range(1, 13)},
    'PJM':   {m: 0.93 for m in range(1, 13)},
    'NYISO': {m: 0.90 for m in range(1, 13)},
    'NEISO': {m: 0.90 for m in range(1, 13)},
    'MISO':  {m: 0.92 for m in range(1, 13)},
    'SPP':   {m: 0.92 for m in range(1, 13)},
}

# ============================================================================
# SENSITIVITY TOGGLE DEFINITIONS
# ============================================================================

LMH = ['L', 'M', 'H']
LEVEL_NAME = {'L': 'Low', 'M': 'Medium', 'H': 'High', 'N': 'None'}

# Toggle group definitions (for scenario key generation)
TOGGLE_GROUPS = OrderedDict([
    ('ren',      {'name': 'Renewable Gen',  'levels': LMH}),
    ('firm',     {'name': 'Firm Gen',       'levels': LMH}),
    ('batt',     {'name': 'Storage',        'levels': LMH}),
    ('fuel',     {'name': 'Fossil Fuel',    'levels': LMH}),
    ('tx',       {'name': 'Transmission',   'levels': ['N', 'L', 'M', 'H']}),
    ('ccs',      {'name': 'CCS',            'levels': LMH}),
    ('q45',      {'name': '45Q',            'levels': ['0', '1']}),
])

# CAISO adds geothermal toggle
CAISO_EXTRA_TOGGLES = OrderedDict([
    ('geo',      {'name': 'Geothermal',     'levels': LMH}),
])

# Number of scenarios per region/threshold
# Ren(3) × Firm(3) × Batt(3) × LDES(3) × Fuel(3) × Tx(4) × CCS(3) × 45Q(2) × Geo(1|3)
N_SCENARIOS_BASE = 3**6 * 4 * 2                       # = 5,832
N_SCENARIOS_CAISO = N_SCENARIOS_BASE * 3              # = 17,496 (Geo L/M/H)

def n_scenarios(iso):
    """Return number of cost scenarios for an ISO."""
    return N_SCENARIOS_CAISO if iso == 'CAISO' else N_SCENARIOS_BASE

# ============================================================================
# DATA PATHS
# ============================================================================

import os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATHS = {
    'project_root':     _PROJECT_ROOT,
    'data':             os.path.join(_PROJECT_ROOT, 'data'),
    'step1_pfs':        os.path.join(_PROJECT_ROOT, 'data', 'step1-pfs-parquets'),
    'step1d_storage':   os.path.join(_PROJECT_ROOT, 'data', 'step1d-storage-parquets'),
    'step2_ef':         os.path.join(_PROJECT_ROOT, 'data', 'step2-ef-parquets'),
    'step3_cost':       os.path.join(_PROJECT_ROOT, 'data', 'step3-cost-opt-parquets'),
    'step4_post':       os.path.join(_PROJECT_ROOT, 'data', 'step5-post-processing'),  # legacy dir name
    'dispatch_cache':   os.path.join(_PROJECT_ROOT, 'data', 'step5-post-processing', 'dispatch_cache'),
    'co2_results':      os.path.join(_PROJECT_ROOT, 'data', 'step5-post-processing', 'co2_results'),
    'lmp':              os.path.join(_PROJECT_ROOT, 'data', 'step5-post-processing', 'lmp'),
    'dashboard':        os.path.join(_PROJECT_ROOT, 'dashboard'),
    'dashboard_js':     os.path.join(_PROJECT_ROOT, 'dashboard', 'js'),
    'scripts':          os.path.join(_PROJECT_ROOT, 'scripts'),
    'eia_profiles':     os.path.join(_PROJECT_ROOT, 'data', 'eia-930'),
}

# ============================================================================
# FILE NAMING CONVENTIONS
# ============================================================================
# Standardized file naming patterns for all pipeline outputs.
# Format: step{N}_{type}_{ISO}[_t{threshold}].parquet
#
# Examples:
#   step1_pfs_CAISO_t90.parquet      (Step 1 PFS)
#   step1d_storage_PJM_t65.parquet   (Step 1d storage refinement)
#   step2_ef_CAISO_t90.parquet       (Step 2 efficient frontier)
#   step3_co_CAISO.parquet           (Step 3 cost optimization, all thresholds)
#   step4_CAISO.parquet              (Step 4 gas/CCS, all thresholds)
#
# Legacy naming is supported for backward compatibility but new outputs
# should follow the standard pattern.

def step1_pfs_filename(iso, threshold):
    """Step 1 PFS filename: {ISO}_t{T}_raw_pfs.parquet"""
    t_str = f'{threshold:g}'
    return f'{iso}_t{t_str}_raw_pfs.parquet'

def step1d_storage_filename(iso, threshold, batch=None):
    """Step 1d storage filename: {ISO}_t{T}_storage.parquet or _b{N}.parquet"""
    t_str = f'{threshold:g}'
    if batch is not None:
        return f'{iso}_t{t_str}_storage_b{batch}.parquet'
    return f'{iso}_t{t_str}_storage.parquet'

def step2_ef_filename(iso, threshold):
    """Step 2 EF filename: step2_ef_{ISO}_t{T}.parquet"""
    t_str = f'{threshold:g}'
    return f'step2_ef_{iso}_t{t_str}.parquet'

def step3_co_filename(iso):
    """Step 3 cost optimization filename: step3_co_{ISO}.parquet"""
    return f'step3_co_{iso}.parquet'

def step4_filename(iso):
    """Step 4 gas/CCS filename: step4_{ISO}.parquet"""
    return f'step4_{iso}.parquet'

def dispatch_cache_filename(iso):
    """Dispatch cache filename: {ISO}_dispatch_cache.parquet"""
    return f'{iso}_dispatch_cache.parquet'

def co2_results_filename(iso):
    """CO2 results filename: co2_{ISO}.parquet"""
    return f'co2_{iso}.parquet'

# ============================================================================
# PARQUET SCHEMA DEFINITIONS
# ============================================================================
# Column naming conventions:
#   - Step 1 PFS: raw resource names (clean_firm, solar, wind, ...)
#   - Step 3 output: prefixed (mix_clean_firm, cost_total_cost, ...)
#   - Step 4 output: adds ra_*, neiso_gas_adj_*, no45q_* prefixes
#
# The prefix scheme exists because Step 3+ combines mix composition
# with cost results in a single row. Prefixes disambiguate "solar" (the
# resource allocation %) from "cost_solar" (a potential cost column).

STEP1_PFS_SCHEMA = {
    'required': ['iso', 'threshold', 'clean_firm', 'solar', 'wind', 'hydro',
                 'battery_dispatch_pct', 'battery8_dispatch_pct',
                 'ldes_dispatch_pct', 'h2_dispatch_pct',
                 'hourly_match_score'],
    'optional': ['offshore_wind', 'geothermal'],
}

STEP2_EF_SCHEMA = {
    'required': ['iso', 'clean_firm', 'solar', 'wind', 'hydro',
                 'battery_dispatch_pct', 'battery8_dispatch_pct',
                 'ldes_dispatch_pct', 'h2_dispatch_pct',
                 'hourly_match_score', 'pareto_type'],
    'optional': ['offshore_wind', 'geothermal'],
}

STEP3_CO_SCHEMA = {
    'required': ['iso', 'threshold', 'scenario', 'annual_demand_mwh',
                 'mix_clean_firm', 'mix_solar', 'mix_wind', 'mix_ccs_ccgt', 'mix_hydro',
                 'hourly_match_score',
                 'battery_dispatch_pct', 'battery8_dispatch_pct',
                 'ldes_dispatch_pct', 'h2_dispatch_pct',
                 'cost_total_cost', 'cost_effective_cost',
                 'cost_incremental', 'cost_wholesale',
                 'tranche_cf_existing_twh', 'tranche_uprate_twh',
                 'tranche_nuclear_newbuild_twh', 'tranche_ccs_tranche_twh',
                 'tranche_new_cf_twh',
                 'gas_gas_backup_needed_mw', 'gas_existing_gas_used_mw',
                 'gas_new_gas_build_mw', 'gas_gas_cost_per_mwh',
                 'gas_clean_peak_capacity_mw', 'gas_ra_peak_mw'],
    'optional': ['tranche_geo_twh', 'mix_offshore_wind', 'mix_geothermal'],
}

STEP4_ADDITIONAL_COLS = {
    'ra': ['ra_peak_demand_mw', 'ra_ra_peak_mw', 'ra_clean_peak_capacity_mw',
           'ra_gas_backup_needed_mw', 'ra_existing_gas_used_mw',
           'ra_new_gas_build_mw', 'ra_existing_gas_cost_per_mwh',
           'ra_new_gas_cost_per_mwh', 'ra_gas_backup_cost_per_mwh',
           'ra_total_system_cost_per_mwh', 'ra_incremental_with_new_gas',
           'ra_clean_coverage_pct', 'ra_resource_adequacy_margin',
           'ra_gas_availability_factor'],
    'no45q': ['no45q_total_cost', 'no45q_effective_cost',
              'no45q_incremental', 'no45q_wholesale',
              'no45q_crossover_cf', 'no45q_ccs_no45q_baseload', 'no45q_ldes_cost'],
    'neiso_gas': ['neiso_gas_adj_total_cost', 'neiso_gas_adj_effective_cost',
                  'neiso_gas_adj_incremental', 'neiso_gas_adj_wholesale',
                  'neiso_gas_no45q_total_cost', 'neiso_gas_no45q_effective_cost',
                  'neiso_gas_no45q_incremental', 'neiso_gas_no45q_wholesale'],
    'pipeline': ['ra_pipeline_deliverable_mw', 'ra_pipeline_shortfall_mw',
                 'ra_pipeline_expansion_cost_per_mwh', 'ra_pipeline_constrained'],
}

# ============================================================================
# VALIDATION RANGES (for automated testing)
# ============================================================================
# Expected ranges for key metrics. Used by the test harness to flag anomalies.

VALID_RANGES = {
    'hourly_match_score': (0.0, 100.0),
    'cost_total_cost': (0.0, 500.0),       # $/MWh — extremely generous upper bound
    'cost_effective_cost': (0.0, 1000.0),   # $/MWh — can be high at low thresholds
    'battery_dispatch_pct': (0.0, 50.0),    # % of demand
    'battery8_dispatch_pct': (0.0, 50.0),
    'ldes_dispatch_pct': (0.0, 50.0),
    'h2_dispatch_pct': (0.0, 50.0),
    'clean_firm': (0, 100),                 # % of demand
    'solar': (0, 100),
    'wind': (0, 100),
    'hydro': (0, 50),
    'offshore_wind': (0, 50),
    'geothermal': (0, 30),
    'co2_emission_rate_tco2_mwh': (0.0, 1.5),  # tCO2/MWh
}

# Expected cost ranges by threshold ($/MWh total cost, all-Medium scenario)
# Used for directional reasonableness checks
EXPECTED_COST_RANGES_MEDIUM = {
    50: (20, 80),
    75: (20, 80),
    90: (25, 75),
    95: (28, 100),
    99: (30, 120),
    99.99: (30, 150),
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_threshold(thr):
    """Format threshold for filenames (e.g., 87.5 → '87.5', 10.0 → '10')."""
    return f'{thr:g}'

def is_offshore_iso(iso):
    """Check if an ISO has offshore wind resource."""
    return iso in OFFSHORE_ISOS

def is_geothermal_iso(iso):
    """Check if an ISO has geothermal resource (CAISO only)."""
    return iso in GEOTHERMAL_ISOS

def hours_per_year():
    """Return hours per year (8760, non-leap)."""
    return 8760

H = 8760  # Convenience constant
