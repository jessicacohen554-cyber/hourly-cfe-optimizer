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
import os
import numpy as np

# ============================================================================
# VERSION & METADATA
# ============================================================================

PIPELINE_VERSION = "1.0.0"
BASE_YEAR = 2025
MODEL_TYPE = "snapshot"  # 2025 snapshot model (no forward projections in Track 1)

# Feature flags
CANNIBALIZATION_ENABLED = True  # Per-resource temporal energy revenue in deployment

# Synthetic data fallback behavior when step2.2 parquets are missing:
#   "error"  — raise RuntimeError (strict / production)
#   "warn"   — generate synthetic data + propagate warnings (default)
#   "silent" — current behavior, no warnings (backward compat only)
SYNTHETIC_DATA_MODE = os.environ.get("SYNTHETIC_DATA_MODE", "warn")

# ============================================================================
# SCARCITY PRICING MODE
# ============================================================================
# 'ordc' — Operating Reserve Demand Curve: price = marginal_cost + VOLL × LOLP(reserves)
#   Physically responsive to generation mix changes (adding solar reduces midday scarcity).
# 'demand_quantile' — Legacy: demand-percentile-based congestion/scarcity overlays
#   calibrated against historical ISO price distributions. Cannot respond to mix changes.
SCARCITY_MODE = 'ordc'

# ORDC parameters per ISO
# voll: Value of Lost Load ($/MWh) — from ISO tariffs / FERC filings
# knee_mw: Reserve threshold (MW) above which ORDC adder is exactly $0.
#          Set at ~1.5-2x minimum operating reserve requirement per NERC/ISO standards.
# lam: Exponential decay rate — controls how steeply LOLP rises below the knee.
#      lambda=0.002 → adder ~$92/MWh at 1000 MW below knee (VOLL=5000).
#      lambda=0.0015 → slower decay for larger systems (PJM, MISO).
# cap: Maximum ORDC adder ($/MWh) — prevents single-hour spikes from polluting averages.
#      Capacity-market ISOs capped lower ($200-300); energy-only ERCOT at $500.
# Sources: ERCOT PUCT Docket 52373, PJM RPM penalty factor (1/3 × $11,100),
#          CAISO/NYISO/NEISO from FERC filings and regional reliability standards.
# Calibration target: annual avg ORDC contribution $2-8/MWh, 30-100 scarcity hours.
ORDC_PARAMS = {
    'ERCOT': {'voll': 5000, 'knee_mw': 3000, 'lam': 0.002, 'cap': 500},
    'PJM':   {'voll': 3700, 'knee_mw': 6000, 'lam': 0.0015, 'cap': 300},
    'CAISO': {'voll': 2000, 'knee_mw': 4000, 'lam': 0.002, 'cap': 300},
    'NYISO': {'voll': 2500, 'knee_mw': 3000, 'lam': 0.002, 'cap': 300},
    'NEISO': {'voll': 2000, 'knee_mw': 2500, 'lam': 0.002, 'cap': 250},
    'MISO':  {'voll': 3500, 'knee_mw': 5000, 'lam': 0.0015, 'cap': 300},
    'SPP':   {'voll': 2000, 'knee_mw': 3500, 'lam': 0.002, 'cap': 200},
}

# ============================================================================
# BACKTEST CONFIGURATION
# ============================================================================
# Year-range → scarcity pricing mode mapping for trajectory backtesting.
# ERCOT had active ORDC pricing from 2014+; other ISOs adopted reserve-margin-
# responsive pricing mechanisms later. For historical fidelity:
#   2020-2022: demand_quantile (most markets calibrated to demand-percentile pricing)
#   2023-2024: ordc (ERCOT ORDC active, other ISOs have capacity-scarcity mechanisms)
# ERCOT always uses ORDC since it's structurally appropriate for its energy-only market.
BACKTEST_SCARCITY_BY_YEAR = {
    2020: 'demand_quantile',
    2021: 'demand_quantile',
    2022: 'demand_quantile',
    2023: 'ordc',
    2024: 'ordc',
}

# ERCOT override: always ORDC (energy-only market with active ORDC since 2014)
BACKTEST_ERCOT_ALWAYS_ORDC = True

# Validation tolerance bands — tighter for ORDC years where physics is better matched
BACKTEST_TOLERANCES = {
    'ordc': {
        'lmp_abs_error': 5.0,       # ±$5/MWh for ORDC years (2023-2024)
        'clean_pct_error': 3.0,     # ±3pp clean energy share
        'deployment_rate_error': 2.0,  # ±2 GW/yr deployment rate
    },
    'demand_quantile': {
        'lmp_abs_error': 8.0,       # ±$8/MWh for pre-ORDC years (wider tolerance)
        'clean_pct_error': 4.0,     # ±4pp clean energy share
        'deployment_rate_error': 3.0,  # ±3 GW/yr deployment rate
    },
}

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

# ============================================================================
# ZONAL TRANSMISSION CONFIGURATION
# ============================================================================
# Simplified pipe-and-bubble zonal decomposition per ISO.
# Each ISO has 2-5 zones with inter-zonal transfer limits (MW) and demand shares.
# Sources: PJM RTEP, MISO MTEP, ERCOT CDR, NYISO Gold Book, ISO-NE RSP.
# Transfer limits are bidirectional thermal ratings on major interfaces.
# Demand shares from EIA-930 subregional data and ISO load zone publications.

ZONE_CONFIG = {
    'PJM': {
        'zones': ['Western', 'AEP_East', 'MAAC', 'EMAAC', 'SWMAAC'],
        'demand_share': {
            'Western': 0.18, 'AEP_East': 0.15, 'MAAC': 0.25,
            'EMAAC': 0.27, 'SWMAAC': 0.15,
        },
        'transfer_limits_mw': {
            ('Western', 'AEP_East'): 5000,   # AEP-East interface
            ('AEP_East', 'MAAC'): 7500,      # ATSI/APS to Mid-Atlantic
            ('MAAC', 'EMAAC'): 6000,         # Mid-Atlantic to Eastern
            ('MAAC', 'SWMAAC'): 5000,        # Mid-Atlantic to Baltimore/DC
        },
    },
    'MISO': {
        'zones': ['North', 'Central', 'South'],
        'demand_share': {'North': 0.35, 'Central': 0.40, 'South': 0.25},
        'transfer_limits_mw': {
            ('North', 'Central'): 4000,       # MN/WI/IA to IL/IN/MI
            ('Central', 'South'): 3000,       # IL/IN to LA/MS/AR — major bottleneck
        },
    },
    'ERCOT': {
        'zones': ['West', 'North', 'South', 'Houston'],
        'demand_share': {
            'West': 0.10, 'North': 0.35, 'South': 0.25, 'Houston': 0.30,
        },
        'transfer_limits_mw': {
            ('West', 'North'): 3500,          # Wind corridor to Dallas
            ('North', 'South'): 5000,         # Dallas to San Antonio/Austin
            ('North', 'Houston'): 6000,       # Dallas to Houston
            ('South', 'Houston'): 4500,       # SA to Houston
        },
    },
    'NYISO': {
        'zones': ['Upstate', 'NYC', 'LongIsland'],
        'demand_share': {'Upstate': 0.45, 'NYC': 0.40, 'LongIsland': 0.15},
        'transfer_limits_mw': {
            ('Upstate', 'NYC'): 5150,         # Central East + UPNY-ConEd interfaces
            ('NYC', 'LongIsland'): 500,       # Zone J to Zone K — very constrained
        },
    },
    'NEISO': {
        'zones': ['Northern', 'Southern'],
        'demand_share': {'Northern': 0.30, 'Southern': 0.70},
        'transfer_limits_mw': {
            ('Northern', 'Southern'): 2500,   # ME/NH/VT to MA/CT/RI
        },
    },
    'CAISO': {
        'zones': ['NP15', 'SP15'],
        'demand_share': {'NP15': 0.55, 'SP15': 0.45},
        'transfer_limits_mw': {
            ('NP15', 'SP15'): 4000,           # Path 15 + Path 26 combined
        },
    },
    'SPP': {
        'zones': ['North', 'South'],
        'demand_share': {'North': 0.45, 'South': 0.55},
        'transfer_limits_mw': {
            ('North', 'South'): 3500,         # KS/NE to OK/TX panhandle
        },
    },
}

# VRE primary zone mapping: which zone has the most solar/wind resource.
# Used for zone-aware cannibalization — solar capture rate should reflect
# the zone where solar is predominantly located, etc.
VRE_PRIMARY_ZONE = {
    'CAISO': {'solar': 'SP15', 'wind': 'NP15'},
    'ERCOT': {'solar': 'West', 'wind': 'West', 'offshore_wind': 'Houston'},
    'PJM':   {'solar': 'MAAC', 'wind': 'Western', 'offshore_wind': 'EMAAC'},
    'NYISO': {'solar': 'Upstate', 'wind': 'Upstate', 'offshore_wind': 'LongIsland'},
    'NEISO': {'solar': 'Southern', 'wind': 'Northern', 'offshore_wind': 'Southern'},
    'MISO':  {'solar': 'Central', 'wind': 'North'},
    'SPP':   {'solar': 'South', 'wind': 'North'},
}

# Balancing Authority → (ISO, Zone) mapping for plant assignment.
# Priority: BA_TO_ZONE > lat/lon fallback > largest-demand zone.
BA_TO_ZONE = {
    # PJM zones
    'AEP': ('PJM', 'Western'), 'AP': ('PJM', 'Western'),
    'ATSI': ('PJM', 'AEP_East'), 'DAY': ('PJM', 'AEP_East'),
    'DEOK': ('PJM', 'AEP_East'),
    'COMED': ('PJM', 'MAAC'), 'DOM': ('PJM', 'MAAC'),
    'PJMW': ('PJM', 'Western'), 'PJMC': ('PJM', 'MAAC'),
    'DPL': ('PJM', 'EMAAC'), 'JC': ('PJM', 'EMAAC'),
    'PS': ('PJM', 'EMAAC'), 'RECO': ('PJM', 'EMAAC'),
    'PJME': ('PJM', 'EMAAC'),
    'PEP': ('PJM', 'SWMAAC'), 'PE': ('PJM', 'SWMAAC'),
    'PJMD': ('PJM', 'SWMAAC'),
    'DUQ': ('PJM', 'Western'), 'EKPC': ('PJM', 'Western'),
    'ME': ('PJM', 'EMAAC'), 'PL': ('PJM', 'EMAAC'),
    'PN': ('PJM', 'EMAAC'),
    # MISO zones
    'NSP': ('MISO', 'North'), 'GRE': ('MISO', 'North'),
    'OTP': ('MISO', 'North'), 'MDU': ('MISO', 'North'),
    'MEC': ('MISO', 'North'), 'MPW': ('MISO', 'North'),
    'MPS': ('MISO', 'North'), 'UPPC': ('MISO', 'North'),
    'MGE': ('MISO', 'North'), 'WEC': ('MISO', 'North'),
    'WPS': ('MISO', 'North'),
    'ALTE': ('MISO', 'Central'), 'ALTW': ('MISO', 'Central'),
    'CONS': ('MISO', 'Central'), 'CWEP': ('MISO', 'Central'),
    'CWLP': ('MISO', 'Central'), 'NIPS': ('MISO', 'Central'),
    'HE': ('MISO', 'Central'), 'SIPC': ('MISO', 'Central'),
    'SMP': ('MISO', 'Central'),
    'EES': ('MISO', 'South'), 'EAI': ('MISO', 'South'),
    'LAFA': ('MISO', 'South'), 'LEPA': ('MISO', 'South'),
    'CLEC': ('MISO', 'South'), 'AMMO': ('MISO', 'South'),
    'DECI': ('MISO', 'South'), 'EDE': ('MISO', 'South'),
    'EMBA': ('MISO', 'South'),
    # ERCOT zones (single BA — use lat/lon for sub-zone assignment)
    'ERCO': ('ERCOT', None),  # Resolved by lat/lon
    'ERCOT': ('ERCOT', None),
    # NYISO zones (single BA — use lat/lon for sub-zone assignment)
    'NYIS': ('NYISO', None),
    'NYISO': ('NYISO', None),
    # NEISO zones
    'ISNE': ('NEISO', None),
    'NEISO': ('NEISO', None),
    'ISONE': ('NEISO', None),
    # CAISO zones (single BA)
    'CISO': ('CAISO', None),
    'CAISO': ('CAISO', None),
    # SPP zones
    'SWPP': ('SPP', None), 'SPP': ('SPP', None),
    'KCPL': ('SPP', 'North'), 'LES': ('SPP', 'North'),
    'NPPD': ('SPP', 'North'), 'OPPD': ('SPP', 'North'),
    'MIDW': ('SPP', 'North'), 'WAUE': ('SPP', 'North'),
    'OKGE': ('SPP', 'South'), 'SPS': ('SPP', 'South'),
    'WFEC': ('SPP', 'South'), 'CSWS': ('SPP', 'South'),
    'GRDA': ('SPP', 'South'), 'INDN': ('SPP', 'South'),
    'KACY': ('SPP', 'South'), 'SPA': ('SPP', 'South'),
    'SECI': ('SPP', 'South'), 'SPRM': ('SPP', 'South'),
    'WR': ('SPP', 'South'), 'AECI': ('SPP', 'South'),
}

# Lat/lon bounding boxes for zone assignment when BA mapping returns None.
# Format: (iso, zone) → {lat: (min, max), lon: (min, max)}
ZONE_BOUNDS = {
    # ERCOT sub-zones
    ('ERCOT', 'West'):    {'lat': (29.5, 34.0), 'lon': (-106.0, -100.5)},
    ('ERCOT', 'North'):   {'lat': (32.0, 34.0), 'lon': (-100.5, -96.0)},
    ('ERCOT', 'South'):   {'lat': (27.5, 32.0), 'lon': (-100.5, -96.5)},
    ('ERCOT', 'Houston'): {'lat': (28.5, 31.0), 'lon': (-96.5, -93.5)},
    # NYISO sub-zones
    ('NYISO', 'Upstate'):    {'lat': (41.5, 45.0), 'lon': (-80.0, -73.5)},
    ('NYISO', 'NYC'):        {'lat': (40.4, 41.5), 'lon': (-74.5, -73.5)},
    ('NYISO', 'LongIsland'): {'lat': (40.5, 41.2), 'lon': (-73.5, -71.8)},
    # NEISO sub-zones
    ('NEISO', 'Northern'): {'lat': (43.0, 47.5), 'lon': (-73.5, -66.9)},
    ('NEISO', 'Southern'): {'lat': (41.0, 43.0), 'lon': (-73.7, -69.9)},
    # CAISO sub-zones
    ('CAISO', 'NP15'): {'lat': (36.8, 42.0), 'lon': (-124.5, -119.0)},
    ('CAISO', 'SP15'): {'lat': (32.5, 36.8), 'lon': (-121.5, -114.5)},
    # SPP sub-zones (fallback for unmapped BAs)
    ('SPP', 'North'): {'lat': (38.0, 43.0), 'lon': (-104.0, -94.5)},
    ('SPP', 'South'): {'lat': (33.0, 38.0), 'lon': (-103.0, -94.0)},
}


def get_zone_for_plant(iso, ba_code=None, lat=None, lon=None):
    """Assign a plant to a transmission zone within its ISO.

    Priority: BA_TO_ZONE mapping > lat/lon bounding box > largest-demand zone.

    Returns:
        zone_name (str) or None if ISO has no zone config.
    """
    if iso not in ZONE_CONFIG:
        return None

    config = ZONE_CONFIG[iso]
    zones = config['zones']

    # Try BA mapping first
    if ba_code:
        ba_upper = str(ba_code).strip().upper()
        if ba_upper in BA_TO_ZONE:
            _, zone = BA_TO_ZONE[ba_upper]
            if zone is not None and zone in zones:
                return zone

    # Try lat/lon bounding box
    if lat is not None and lon is not None:
        for zone in zones:
            key = (iso, zone)
            if key in ZONE_BOUNDS:
                bounds = ZONE_BOUNDS[key]
                if (bounds['lat'][0] <= lat <= bounds['lat'][1] and
                        bounds['lon'][0] <= lon <= bounds['lon'][1]):
                    return zone

    # Fallback: zone with largest demand share
    return max(config['demand_share'], key=config['demand_share'].get)

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
    10.0, 20.0, 30.0, 40.0,          # Coarse range (no step1c, no fine zone)
    50.0, 55.0, 60.0, 65.0,          # Mid range (5% steps)
    70.0, 75.0, 80.0,                # Upper-mid range
    85.0, 87.5, 90.0, 92.5,          # Inflection zone (2.5% steps)
    95.0, 97.5,                       # High range
    99.0, 99.5, 99.9, 99.99,         # Last mile
]

# Active thresholds (full pipeline coverage with step1c storage refinement)
ACTIVE_THRESHOLDS = [t for t in THRESHOLDS if t >= 50.0]  # 17 thresholds

# Coarse-only thresholds (no step1c, no fine zone search)
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

# Existing CAISO geothermal as % of demand (5.31 TWh / 224.039 TWh = 2.37%)
# Used in Step 3 to split physics geothermal into existing ($0) vs new-build (priced)
EXISTING_GEOTHERMAL_PCT = 2.37

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
# Round-trip efficiency (RTE) and duration parameters for each storage class.
#
# Sources:
#   Battery 4hr/8hr: NREL ATB 2024, utility-scale Li-ion. RTE = 85% is the
#     2024 moderate-technology baseline (DC-side). Duration is nameplate.
#     Reference: NREL Annual Technology Baseline 2024, "Utility-Scale Battery
#     Storage" technology page.
#   LDES 100hr: Form Energy iron-air battery (announced 2023). RTE ≈ 50%
#     (lower bound of 45-55% range from DOE LDES Liftoff, Sep 2023).
#     7-day rolling window reflects multi-day weather event bridging.
#     Reference: DOE Pathways to Commercial Liftoff: LDES (2023), p.18.
#   Green H2 1000hr: Electrolysis (PEM, ~70% HHV) × compression/storage
#     (salt cavern, ~95%) × H2 turbine (combined cycle, ~55% HHV).
#     Product: 0.70 × 0.95 × 0.55 ≈ 0.35 RTE. 30-day rolling window
#     for seasonal/multi-week bridging.
#     Reference: Hydrogen Council (2024), "Hydrogen Insights 2024"; IRENA
#     (2024), "Green Hydrogen Cost Reduction: Scaling Up Electrolysers."

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
    'h2': 1.0,          # 1.0% of annual demand (small probe — 3 levels)
}

# Step 1D.2: Research-informed 2050 storage caps (% of annual demand)
# Sources: NREL Storage Futures (200 GW / 1,200 GWh ref case),
#          DOE LDES Liftoff (225-460 GW LDES for net-zero),
#          Princeton Net Zero America (1,300 GWh by 2050)
STORAGE_MAX_V2 = {
    'battery': 0.10,    # 0.10% of annual demand — CAISO: ~224 GWh / 56 GW
    'battery8': 0.15,   # 0.15% of annual demand
    'ldes': 1.0,        # 1.0% of annual demand
    'h2': 1.0,          # 1.0% of annual demand (small probe — 3 levels)
}

# Storage dispatch mode: 'greedy' (sequential priority) or 'lp' (co-optimized via scipy.linprog)
# LP co-dispatch solves a rolling-window linear program that coordinates all storage types
# simultaneously, reducing order-dependency bias and improving gap-filling by 8-15%.
STORAGE_DISPATCH_MODE = 'greedy'

# ============================================================================
# DEMAND RESPONSE PARAMETERS
# ============================================================================
# Price-elastic demand curtailment model. When LMP exceeds an ISO-specific
# trigger price, registered DR capacity sheds load proportionally until price
# equilibrates. Sources: FERC Form 714, ISO DR registration reports, PJM
# Demand Response Operations Markets Activity Reports (2020-2024).
#
# max_dr_gw:      Maximum registered DR capacity (GW) from ISO filings
# trigger_price:  LMP threshold ($/MWh) above which DR activates
# participation:  Base fraction of registered DR that actually responds
#                 (historical avg from ISO event performance data)

DEMAND_RESPONSE = {
    'CAISO':  {'max_dr_gw': 4.0,  'trigger_price': 150, 'participation': 0.70, 'dr_ordc_link': True},
    'ERCOT':  {'max_dr_gw': 5.0,  'trigger_price': 200, 'participation': 0.60, 'dr_ordc_link': True},
    'PJM':    {'max_dr_gw': 10.0, 'trigger_price': 100, 'participation': 0.75, 'dr_ordc_link': True},
    'NYISO':  {'max_dr_gw': 1.5,  'trigger_price': 150, 'participation': 0.70, 'dr_ordc_link': True},
    'NEISO':  {'max_dr_gw': 1.0,  'trigger_price': 150, 'participation': 0.65, 'dr_ordc_link': True},
    'MISO':   {'max_dr_gw': 8.0,  'trigger_price': 120, 'participation': 0.65, 'dr_ordc_link': True},
    'SPP':    {'max_dr_gw': 2.0,  'trigger_price': 150, 'participation': 0.60, 'dr_ordc_link': True},
}

# DR sensitivity levels: Off/Low/Medium/High
# participation_mult: multiplier on base participation rate
# trigger_mult: multiplier on trigger price (>1 = higher threshold = less activation)
DR_LEVELS = {
    'Off':    {'participation_mult': 0.0,  'trigger_mult': 1.0},
    'Low':    {'participation_mult': 0.50, 'trigger_mult': 1.3},
    'Medium': {'participation_mult': 0.70, 'trigger_mult': 1.0},
    'High':   {'participation_mult': 0.90, 'trigger_mult': 0.8},
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

# Capacity market price degradation — S-curve (sigmoid) model
# cap_price(t) = base_price × max(floor, 1 - max_degrade / (1 + exp(-k × (clean_share - midpoint))))
#
# Parameters per ISO:
#   max_degrade: maximum fraction of base price that can be eroded (0-1)
#   midpoint: clean share (0-1) where degradation reaches 50% of max
#   k: steepness of the sigmoid (higher = sharper transition)
#   floor: minimum fraction of base price (prevents negative/zero artifacts)
#
# Calibration sources:
#   PJM RPM: BRA clearing prices 2015-2025 vs. clean share trajectory
#   NYISO ICAP: Monthly spot auction clearing 2019-2025
#   NEISO FCM: FCA-15 through FCA-19 clearing prices
#   CAISO RA: Bilateral RA contract prices 2020-2025
#
# Energy-only markets (ERCOT, SPP) have $0 base price, parameters are no-ops.
# MISO PRA is weak ($25/kW-yr) and minimally affected.
CAPACITY_DEGRADATION_PARAMS = {
    'CAISO': {'max_degrade': 0.85, 'midpoint': 0.55, 'k': 10, 'floor': 0.10},
    'ERCOT': {'max_degrade': 0.0,  'midpoint': 0.50, 'k': 8,  'floor': 0.0},   # energy-only
    'PJM':   {'max_degrade': 0.80, 'midpoint': 0.50, 'k': 8,  'floor': 0.15},
    'NYISO': {'max_degrade': 0.85, 'midpoint': 0.45, 'k': 10, 'floor': 0.10},
    'NEISO': {'max_degrade': 0.80, 'midpoint': 0.50, 'k': 8,  'floor': 0.15},
    'MISO':  {'max_degrade': 0.0,  'midpoint': 0.50, 'k': 8,  'floor': 0.0},   # weak capacity market
    'SPP':   {'max_degrade': 0.0,  'midpoint': 0.50, 'k': 8,  'floor': 0.0},   # energy-only
}

# Capacity price scarcity parameters — reserve margin thresholds
# When reserve margins fall below target_rm, capacity prices increase via a
# piecewise-linear scarcity multiplier. This creates a feedback loop:
#   fossil retirements → lower reserves → higher capacity prices →
#   new fossil builds become economic → reserves recover → prices stabilize
#
# Calibration:
#   PJM RPM: BRA clearing jumps 2-3× when reserves drop below 15%
#   NYISO ICAP: Monthly spot prices spike when summer margins tighten
#   NEISO FCM: FCA clearing inversely correlated with projected reserves
#   Target reserve margin (15%) matches NERC Reference Margin Level
#   Max multiplier (3×) reflects PJM 2023/2024 BRA price spike pattern
CAPACITY_SCARCITY_PARAMS = {
    'target_reserve_margin_pct': 15.0,   # NERC Reference Margin Level
    'max_scarcity_multiplier': 3.0,      # Price ceiling at 0% reserve margin
    'floor_multiplier': 1.0,             # No scarcity effect above target RM
}


def compute_capacity_price(iso, reserve_margin_pct, clean_pct):
    """Compute endogenous capacity market price given current grid conditions.

    Combines two effects:
    1. Scarcity multiplier: price increases as reserve margin falls below target
       (piecewise linear from 1× at target to max× at 0% reserve margin)
    2. Clean penetration degradation: existing sigmoid S-curve from
       CAPACITY_DEGRADATION_PARAMS (higher clean share → lower capacity value)

    Args:
        iso: ISO region string
        reserve_margin_pct: Current reserve margin as percentage
            (e.g. 12.0 means 12% reserve margin)
        clean_pct: Current clean energy percentage (0-100)

    Returns:
        Effective capacity price in $/kW-yr (endogenous)
    """
    import numpy as np

    base_price = CAPACITY_MARKET_PRICES.get(iso, 0)
    if base_price <= 0:
        return 0.0  # Energy-only markets (ERCOT, SPP)

    # --- Scarcity multiplier (reserve margin effect) ---
    target_rm = CAPACITY_SCARCITY_PARAMS['target_reserve_margin_pct']
    max_mult = CAPACITY_SCARCITY_PARAMS['max_scarcity_multiplier']
    floor_mult = CAPACITY_SCARCITY_PARAMS['floor_multiplier']

    if reserve_margin_pct >= target_rm:
        scarcity_mult = floor_mult
    else:
        # Linear ramp from floor_mult at target_rm to max_mult at 0%
        scarcity_mult = floor_mult + (max_mult - floor_mult) * (
            (target_rm - reserve_margin_pct) / target_rm
        )
        # Clamp — reserve margins can go negative in extreme retirements
        scarcity_mult = min(scarcity_mult, max_mult)

    # --- Clean penetration degradation (existing sigmoid) ---
    params = CAPACITY_DEGRADATION_PARAMS.get(iso, {})
    max_degrade = params.get('max_degrade', 0.0)
    midpoint = params.get('midpoint', 0.50)
    k = params.get('k', 8)
    floor = params.get('floor', 0.0)

    if max_degrade > 0:
        x = clean_pct / 100.0
        sigmoid = 1.0 / (1.0 + np.exp(-k * (x - midpoint)))
        clean_mult = 1.0 - max_degrade * sigmoid
        clean_mult = max(floor, clean_mult)
    else:
        clean_mult = 1.0

    return base_price * scarcity_mult * clean_mult


# Backward compat — legacy alpha values (deprecated, use CAPACITY_DEGRADATION_PARAMS)
CAPACITY_DEGRADATION_ALPHA = {
    'CAISO': 0.40, 'ERCOT': 0.0, 'PJM': 0.35, 'NYISO': 0.40,
    'NEISO': 0.35, 'MISO': 0.0, 'SPP': 0.0,
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

# Firm import capacity (MW) — maximum dependable inter-regional transfer capability
# Source: NERC Interregional Transfer Capability assessments, EIA-930 peak observed flows (2024)
FIRM_IMPORT_MW = {
    'CAISO': 8000,   # Path 66 + PDCI from Pacific NW
    'ERCOT': 1200,   # DC ties to SPP/Mexico (limited by design)
    'PJM': 5000,     # MISO/NYISO interchange
    'NYISO': 4000,   # PJM/NEISO/HQ imports
    'NEISO': 3500,   # HQ Phase I/II + NB Power + NYISO
    'MISO': 4000,    # PJM/SPP interchange
    'SPP': 3000,     # MISO/ERCOT interchange
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

# Unit commitment: minimum generation as fraction of nameplate capacity
# Nuclear: fully must-run (can't economically cycle)
# Coal steam: min stable generation ~40% (thermal inertia, boiler constraints)
# Gas CCGT/CT: fully dispatchable (can cycle off)
MUST_RUN_PCT = {
    'nuclear': 1.0,
    'coal_steam': 0.40,
    'gas_ccgt': 0.0,
    'gas_ct': 0.0,
    'oil_ct': 0.0,
}

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
# Validated: Winter 2024/25 ACG averaged $7.45/MMBtu above Henry Hub (EIA, NGI).
# Sources: ISO-NE Operational Fuel Security Analysis (2018), EIA Today in Energy
# (Dec 2023, "Market dynamics vary at key natural gas pricing hubs"),
# NaturalGasIntel (Jan 2025, winter 2024/25 ACG data).
NEISO_CCS_GAS_ADDER = 13.13    # $/MWh annualized CCS adder (7 HR × $7.50 × 0.25)
NEISO_WHOLESALE_ADDER = 4.0    # $/MWh annualized wholesale adder

# 45Q Tax Credit for CCS
# $85/ton × 0.323 tCO2/MWh captured (90% capture × 0.359 tCO2/MWh unabated)
CCS_45Q_CREDIT_PER_MWH = 27.5  # $/MWh offset between 45Q ON and OFF tables

# 45Q Realization Probability — accounts for execution risk in CCS deployment:
# geological sequestration certification delays, IRS compliance complexity,
# Class VI well permitting (2-4 year timelines), and sustained 90% capture
# rate uncertainty. Real-world CCS projects have achieved 60-80% sustained
# capture vs. the 90% design basis used in credit calculations.
# Sources: CCS Institute Global Status Report 2024, EPA Class VI Primacy
# delegation timeline analysis, GAO-24-106044 (45Q oversight).
CCS_45Q_REALIZATION_PROB = {
    'low': 0.70,      # Conservative: permitting delays + capture shortfall
    'medium': 0.85,   # Base case: some execution risk priced in
    'high': 1.00,     # Full credit realization (current model default)
}

# ============================================================================
# NUCLEAR PARAMETERS
# ============================================================================

EXISTING_NUCLEAR_GW = {
    'CAISO': 2.3, 'ERCOT': 2.7, 'PJM': 32.0,
    'NYISO': 3.4, 'NEISO': 3.5, 'MISO': 12.0, 'SPP': 1.2,
}

# Nuclear offtake contracts — plants protected from market-driven retirement.
# In energy-only markets (ERCOT), nuclear plants may appear uneconomic based
# on market revenue alone, but long-term PPAs provide a revenue floor that
# prevents retirement. Without this, the model produces false retirement
# signals for contracted plants (audit finding #11).
# Sources: Luminant/Vistra SEC filings (Comanche Peak PPAs), NRC license
# renewal status, ERCOT market monitor reports (2023-2024).
NUCLEAR_OFFTAKE_CONTRACTS = {
    'ERCOT': {
        'name': 'Comanche Peak Units 1&2',
        'gw': 2.3,
        'contract_floor_mwh': 35.0,   # $/MWh PPA floor price
        'contract_end_year': 2045,     # NRC license expiry (Unit 1: 2030→2050, Unit 2: 2033→2053)
    },
    # PJM, NYISO, NEISO, MISO: nuclear operates under capacity market revenue
    # (RPM, ICAP, FCM, PRA) which provides going-forward cost recovery separate
    # from energy revenue. Retirement decision in capacity markets is already
    # handled by the capacity degradation model. No explicit contract override needed.
}

# Uprate cap: 8% of existing nuclear × 90% CF → TWh/yr
UPRATE_FRACTION = 0.08
UPRATE_CF = 0.90
UPRATE_CAP_TWH = {
    iso: round(gw * UPRATE_FRACTION * UPRATE_CF * 8760 / 1e3, 3)
    for iso, gw in EXISTING_NUCLEAR_GW.items()
}

# Nuclear monthly capacity factors: see DISPATCH-SPECIFIC CONSTANTS section below
# for the detailed monthly profiles (from NRC PRIS data). The earlier simplified
# version (flat CFs for non-CAISO) was replaced by the detailed per-month data
# during the constant consolidation from dispatch_utils.py.

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
# CORRELATED SCENARIO BUNDLES (IEA-aligned)
# ============================================================================
# These represent internally-consistent macro scenarios where parameters are
# correlated as they would be in reality.  They run SEPARATELY from the
# independent Cartesian sweep and are useful for narrative-driven analysis
# (e.g., "what does the NZE pathway look like?").
#
# Sources:
#   IEA World Energy Outlook 2024 — STEPS / APS / NZE pathway definitions
#   EPA Social Cost of Carbon ($51/tCO2) — 2020 IWG central estimate
#   Rennert et al. 2022 ($185/tCO2) — updated damage-function SCC
#   EU ETS 2024 range ($60-100/tCO2)

CORRELATED_SCENARIOS = {
    "IEA_STEPS": {
        "description": "Current policies continue, moderate ambition",
        "demand_growth": "Medium",
        "gas_price": "Medium",
        "renewable_lcoe": "Medium",
        "carbon_price": 0,
        "learning_rate": "Medium",
        "45q": True,
    },
    "IEA_APS": {
        "description": "All announced national commitments implemented",
        "demand_growth": "Medium",
        "gas_price": "Medium",
        "renewable_lcoe": "Low",
        "carbon_price": 51,
        "learning_rate": "Fast",
        "45q": True,
    },
    "IEA_NZE": {
        "description": "1.5C-aligned pathway",
        "demand_growth": "High",
        "gas_price": "High",
        "renewable_lcoe": "Low",
        "carbon_price": 185,
        "learning_rate": "Fast",
        "45q": True,
    },
    "HIGH_FRICTION": {
        "description": "Regulatory/permitting delays + high costs",
        "demand_growth": "High",
        "gas_price": "Low",
        "renewable_lcoe": "High",
        "carbon_price": 0,
        "learning_rate": "Slow",
        "45q": False,
    },
    "RAPID_TRANSITION": {
        "description": "Technology breakthroughs + strong policy",
        "demand_growth": "High",
        "gas_price": "High",
        "renewable_lcoe": "Low",
        "carbon_price": 100,
        "learning_rate": "Fast",
        "45q": True,
    },
}


# ============================================================================
# DATA PATHS
# ============================================================================

import os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATHS = {
    'project_root':     _PROJECT_ROOT,
    'data':             os.path.join(_PROJECT_ROOT, 'data'),
    'step1_pfs':        os.path.join(_PROJECT_ROOT, 'data', 'step1-pfs'),
    'step1_storage':    os.path.join(_PROJECT_ROOT, 'data', 'step1-pfs'),  # storage files live alongside PFS
    'step2_ef':         os.path.join(_PROJECT_ROOT, 'data', 'step2.1-ef'),
    'step3_cost':       os.path.join(_PROJECT_ROOT, 'data', 'step2.2-cost'),
    'step4_post':       os.path.join(_PROJECT_ROOT, 'data', 'step4-analysis'),
    'dispatch_cache':   os.path.join(_PROJECT_ROOT, 'data', 'step3-dispatch'),
    'co2_results':      os.path.join(_PROJECT_ROOT, 'data', 'step4-analysis', 'co2_results'),
    'lmp':              os.path.join(_PROJECT_ROOT, 'data', 'step4-analysis', 'lmp'),
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
#   PJM_t65_storage.parquet          (Step 1C storage refinement, in step1-pfs/)
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

def step1c_storage_filename(iso, threshold, batch=None):
    """Step 1C storage filename: {ISO}_t{T}_storage.parquet or _b{N}.parquet (in step1-pfs/)"""
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
    'ra': ['gas_gas_backup_needed_mw', 'gas_existing_gas_used_mw',
           'gas_new_gas_build_mw', 'gas_gas_cost_per_mwh',
           'gas_clean_peak_capacity_mw', 'gas_ra_peak_mw'],
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


# ============================================================================
# COST TABLES — Single source of truth for all pipeline scripts
# ============================================================================
# Cost basis: NREL ATB 2024, 2024 USD. Supplemented by Lazard v17-18, EIA AEO 2024.
# Battery storage: NREL ATB 2024 + Cost Projections for Utility-Scale Battery
#   Storage 2025 Update. Offshore wind: Lazard v17-18, BNEF 2025, NREL ATB 2024.
# Used by: step3a_cost_optimization.py, step3b_track_nb_ctr.py, scenario_common.py,
#          procurement_utils.py, step9a_generate_shared_data.py
#
# Previously defined separately in step3a_cost_optimization.py and scenario_common.py.
# Unified here 2026-03-04 to eliminate duplication and drift risk.

# LCOE tables by resource type × sensitivity × ISO ($/MWh)
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
    # ---- Offshore wind LCOE ($/MWh) ----
    # Fixed-bottom (NYISO, NEISO, PJM): Lazard v17-18, BNEF 2025, NREL ATB 2024.
    # Floating (CAISO): NREL FORCE model, DOE Floating Wind Shot reference.
    # Non-offshore ISOs (ERCOT, MISO, SPP): set to 0 — no offshore resource.
    # PJM cheapest (shallowest water, largest pipeline, NJ 7.5 GW mandated).
    # NEISO mid (Vineyard Wind precedent, strong resource ~51% CF).
    # NYISO most expensive East Coast (NY Bight permitting, Jones Act).
    # CAISO dramatically higher — floating technology, no US commercial experience.
    'offshore_wind': {
        'Low':    {'CAISO': 110, 'ERCOT': 0, 'PJM': 65, 'NYISO': 72, 'NEISO': 68, 'MISO': 0, 'SPP': 0},
        'Medium': {'CAISO': 150, 'ERCOT': 0, 'PJM': 85, 'NYISO': 95, 'NEISO': 90, 'MISO': 0, 'SPP': 0},
        'High':   {'CAISO': 200, 'ERCOT': 0, 'PJM': 112, 'NYISO': 125, 'NEISO': 118, 'MISO': 0, 'SPP': 0},
    },
    # ---- Storage: annualized capacity cost per % of annual demand ----
    # NOT LCOS. These are annualized fixed costs of storage capacity, normalized to
    # the coefficient model where coeff = bat_pct/100 (energy capacity as fraction
    # of annual demand — same unit as all other resources). Formula:
    #   price = CAPEX_kWh × (CRF + FOM_rate) × 1000 × regional_mult
    # where CRF=0.1019 (8%, 20yr), FOM_rate=2.5% of CAPEX($/kW) per NREL ATB.
    # Regional variation baked in (no separate TX adder for storage).
    #
    # CAPEX source: NREL ATB 2024 component model (Energy $/kWh + Power $/kW).
    #   4hr = Energy + Power/4;  8hr = Energy + Power/8.
    #   Component splits: L=(170+280), M=(210+340), H=(270+420).
    #   4hr: L=$240, M=$295, H=$375.  8hr: L=$205, M=$253, H=$323.
    #   8hr is ~14% cheaper per kWh (power component spread over 2× energy).
    # LCOS cross-check: 4hr Med @ 365 cycles, 85% RTE = $121/MWh.
    # Financial: WACC=8%, Bat life=20yr. FOM=2.5% of CAPEX($/kW) per NREL (incl augmentation).
    #
    # Verification: 0.01% bat4 at CAISO (224 TWh) = 22,400 MWh.
    #   Cost = 0.0001 × 41610 = $4.16/MWh. Physical: 22.4M kWh × $295/kWh × 0.127 × 1.11
    #   = $924M/yr ÷ 224 TWh = $4.13/MWh. ✓
    'battery': {
        'Low':    {'CAISO': 33813.60, 'ERCOT': 30484.80, 'PJM': 32412.00, 'NYISO': 35740.80, 'NEISO': 34777.20, 'MISO': 31711.20, 'SPP': 30835.20},
        'Medium': {'CAISO': 41610.00, 'ERCOT': 37405.20, 'PJM': 39858.00, 'NYISO': 43975.20, 'NEISO': 42661.20, 'MISO': 39069.60, 'SPP': 37930.80},
        'High':   {'CAISO': 52822.80, 'ERCOT': 47566.80, 'PJM': 50632.80, 'NYISO': 55888.80, 'NEISO': 54312.00, 'MISO': 49581.60, 'SPP': 48180.00},
    },
    'battery8': {
        'Low':    {'CAISO': 28908.00, 'ERCOT': 26017.20, 'PJM': 27681.60, 'NYISO': 30572.40, 'NEISO': 29696.40, 'MISO': 27156.00, 'SPP': 26367.60},
        'Medium': {'CAISO': 35565.60, 'ERCOT': 32061.60, 'PJM': 34076.40, 'NYISO': 37668.00, 'NEISO': 36529.20, 'MISO': 33375.60, 'SPP': 32412.00},
        'High':   {'CAISO': 45464.40, 'ERCOT': 40909.20, 'PJM': 43537.20, 'NYISO': 48092.40, 'NEISO': 46690.80, 'MISO': 42661.20, 'SPP': 41434.80},
    },
    'ldes': {
        'Low':    {'CAISO': 3328.80, 'ERCOT': 2890.80, 'PJM': 3153.60, 'NYISO': 3679.20, 'NEISO': 3504.00, 'MISO': 2978.40, 'SPP': 2890.80},
        'Medium': {'CAISO': 5518.80, 'ERCOT': 4730.40, 'PJM': 5168.40, 'NYISO': 6132.00, 'NEISO': 5781.60, 'MISO': 4905.60, 'SPP': 4818.00},
        'High':   {'CAISO': 8760.00, 'ERCOT': 7533.60, 'PJM': 8234.40, 'NYISO': 9723.60, 'NEISO': 9285.60, 'MISO': 7884.00, 'SPP': 7708.80},
    },
    # Green H2: electrolysis + salt cavern + H2 turbine. 35% RTE.
    # CAPEX/kWh: L=$150, M=$220, H=$310. Duration=168hr, FOM=$8/kW-yr.
    # Shares 'ldes_lvl' sensitivity toggle (both long-duration storage).
    'h2': {
        'Low':    {'CAISO': 17344.80, 'ERCOT': 15330.00, 'PJM': 16468.80, 'NYISO': 19096.80, 'NEISO': 18220.80, 'MISO': 15768.00, 'SPP': 15067.20},
        'Medium': {'CAISO': 25404.00, 'ERCOT': 22425.60, 'PJM': 24177.60, 'NYISO': 27944.40, 'NEISO': 26718.00, 'MISO': 23038.80, 'SPP': 22075.20},
        'High':   {'CAISO': 35828.40, 'ERCOT': 31623.60, 'PJM': 33988.80, 'NYISO': 39420.00, 'NEISO': 37580.40, 'MISO': 32499.60, 'SPP': 31010.40},
    },
}

# Transmission adders ($/MWh) by resource × tx level × ISO
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
    # Offshore wind TX: submarine cable + offshore substation.
    # Higher than onshore — export cable ($1-3M/km), 20-80 km to shore.
    # CAISO highest: floating platform + longer cable + deeper water.
    # Non-offshore ISOs: 0 (no offshore resource).
    'offshore_wind': {'None': 0, 'Low': {'CAISO': 10, 'ERCOT': 0, 'PJM': 6, 'NYISO': 8, 'NEISO': 7, 'MISO': 0, 'SPP': 0},
                      'Medium': {'CAISO': 20, 'ERCOT': 0, 'PJM': 11, 'NYISO': 15, 'NEISO': 13, 'MISO': 0, 'SPP': 0},
                      'High': {'CAISO': 35, 'ERCOT': 0, 'PJM': 18, 'NYISO': 25, 'NEISO': 22, 'MISO': 0, 'SPP': 0}},
    # Storage TX = 0: regional variation already baked into annualized capacity costs
    'battery':    {'None': 0, 'Low': 0, 'Medium': 0, 'High': 0},
    'battery8':   {'None': 0, 'Low': 0, 'Medium': 0, 'High': 0},
    'ldes':       {'None': 0, 'Low': 0, 'Medium': 0, 'High': 0},
    'h2':         {'None': 0, 'Low': 0, 'Medium': 0, 'High': 0},
    'hydro':      {'None': 0, 'Low': 0, 'Medium': 0, 'High': 0},
}


def get_tx(rtype, tx_name, iso):
    """Lookup transmission adder for a resource type."""
    entry = TX_TABLES.get(rtype, {}).get(tx_name, 0)
    if isinstance(entry, dict):
        return entry.get(iso, 0)
    return entry


# Nuclear uprate LCOE ($/MWh)
UPRATE_LCOE = {'L': 15, 'M': 25, 'H': 40}

# Nuclear new-build LCOE ($/MWh)
NUCLEAR_NEWBUILD_LCOE = {
    'L': {'CAISO': 70, 'ERCOT': 68, 'PJM': 72, 'NYISO': 75, 'NEISO': 73, 'MISO': 70, 'SPP': 68},
    'M': {'CAISO': 95, 'ERCOT': 90, 'PJM': 105, 'NYISO': 110, 'NEISO': 108, 'MISO': 100, 'SPP': 92},
    'H': {'CAISO': 140, 'ERCOT': 135, 'PJM': 160, 'NYISO': 170, 'NEISO': 165, 'MISO': 155, 'SPP': 140},
}

# Geothermal (CAISO only)
GEOTHERMAL_LCOE = {'L': 63, 'M': 88, 'H': 110}
GEO_CAP_TWH = GEOTHERMAL_CAP_TWH  # Backward compat alias

# CCS-CCGT LCOE with/without 45Q
# 45Q credit: $85/ton × 0.323 tCO2/MWh captured (90% capture × 0.359 tCO2/MWh unabated)
# = $27.5/MWh offset between ON and OFF tables.
# Previous values used $29/MWh offset (corrected 2026-03-03).
CCS_LCOE_45Q_ON = {
    'L': {'CAISO': 59.5, 'ERCOT': 53.5, 'PJM': 63.5, 'NYISO': 79.5, 'NEISO': 76.5, 'MISO': 56.5, 'SPP': 51.5},
    'M': {'CAISO': 87.5, 'ERCOT': 72.5, 'PJM': 80.5, 'NYISO': 100.5, 'NEISO': 97.5, 'MISO': 75.5, 'SPP': 69.5},
    'H': {'CAISO': 116.5, 'ERCOT': 93.5, 'PJM': 103.5, 'NYISO': 129.5, 'NEISO': 123.5, 'MISO': 97.5, 'SPP': 89.5},
}
CCS_LCOE_45Q_OFF = {
    'L': {'CAISO': 87, 'ERCOT': 81, 'PJM': 91, 'NYISO': 107, 'NEISO': 104, 'MISO': 84, 'SPP': 79},
    'M': {'CAISO': 115, 'ERCOT': 100, 'PJM': 108, 'NYISO': 128, 'NEISO': 125, 'MISO': 103, 'SPP': 97},
    'H': {'CAISO': 144, 'ERCOT': 121, 'PJM': 131, 'NYISO': 157, 'NEISO': 151, 'MISO': 125, 'SPP': 117},
}

# CCS regional capacity caps (TWh/yr) — geologic CO2 storage availability
# Mixes with CCS deployment exceeding these caps are filtered out.
# NYISO/NEISO: hard zero (no geologic storage).
# See SPEC.md §5.4.3 for sources and rationale.
# NOTE: Already defined above as CCS_CAP_TWH, re-exported here for discoverability.

# Lazard v16.0 CCGT annualized capacity cost ($/kW-yr)
NEW_CCGT_COST_KW_YR = {
    'CAISO': 112, 'ERCOT': 89, 'PJM': 99, 'NYISO': 114, 'NEISO': 105,
    'MISO': 95, 'SPP': 88,
}
# Existing gas fixed O&M ($/kW-yr)
EXISTING_GAS_FOM_KW_YR = {
    'CAISO': 16, 'ERCOT': 13, 'PJM': 14, 'NYISO': 17, 'NEISO': 15,
    'MISO': 14, 'SPP': 13,
}

# New-build gas LCOE ($/MWh, annualized all-in including capital recovery)
# Sources: Lazard v17-18, EIA AEO 2024, NREL ATB 2024.
# CCGT: baseload combined cycle, 85% CF. CT: peaker combustion turbine, 15-25% CF.
# Used by SMARTargets market simulation (step10) for new gas profitability.
NEW_GAS_CCGT_LCOE = {'Low': 45, 'Medium': 55, 'High': 65}
NEW_GAS_CT_LCOE = {'Low': 65, 'Medium': 80, 'High': 100}

# New-build coal LCOE ($/MWh, annualized all-in including capital recovery)
# Sources: Lazard v18, EIA AEO 2024, IEA WEO 2024.
# Ultra-supercritical coal, 75% CF assumption.
# Low reflects regions with cheap coal + existing rail (MISO/SPP).
# High reflects coastal ISOs with carbon risk premium.
NEW_COAL_LCOE = {'Low': 75, 'Medium': 95, 'High': 120}

# ── New-build fossil annualized CAPEX ($/kW-yr) by sensitivity level ──
# These are the capital recovery + fixed O&M costs for new-build decisions
# in the market simulator dispatch loop. L/M/H represent construction cost
# uncertainty (permitting, labor, supply chain).
# Sources: Lazard v17-18, NREL ATB 2024, EIA AEO 2024.
#
# CCGT: H/J-class combined cycle, 2-3 year build, ~$900-1400/kW overnight.
# CT: Frame/aero peaker, 1-2 year build, ~$400-700/kW overnight.
# Coal: Ultra-supercritical, 4-6 year build, ~$3000-5000/kW overnight.
NEW_BUILD_CAPEX_KW_YR = {
    'Low': {
        'gas_ccgt': {'CAISO': 95, 'ERCOT': 75, 'PJM': 84, 'NYISO': 97, 'NEISO': 89, 'MISO': 80, 'SPP': 74},
        'gas_ct':   {'CAISO': 55, 'ERCOT': 42, 'PJM': 48, 'NYISO': 57, 'NEISO': 52, 'MISO': 45, 'SPP': 41},
        'coal':     {'CAISO': 999, 'ERCOT': 180, 'PJM': 195, 'NYISO': 999, 'NEISO': 999, 'MISO': 170, 'SPP': 165},
    },
    'Medium': {
        'gas_ccgt': {'CAISO': 112, 'ERCOT': 89, 'PJM': 99, 'NYISO': 114, 'NEISO': 105, 'MISO': 95, 'SPP': 88},
        'gas_ct':   {'CAISO': 65, 'ERCOT': 50, 'PJM': 57, 'NYISO': 67, 'NEISO': 62, 'MISO': 54, 'SPP': 49},
        'coal':     {'CAISO': 999, 'ERCOT': 220, 'PJM': 240, 'NYISO': 999, 'NEISO': 999, 'MISO': 210, 'SPP': 200},
    },
    'High': {
        'gas_ccgt': {'CAISO': 132, 'ERCOT': 105, 'PJM': 117, 'NYISO': 135, 'NEISO': 124, 'MISO': 112, 'SPP': 104},
        'gas_ct':   {'CAISO': 78, 'ERCOT': 60, 'PJM': 68, 'NYISO': 80, 'NEISO': 73, 'MISO': 64, 'SPP': 59},
        'coal':     {'CAISO': 999, 'ERCOT': 270, 'PJM': 295, 'NYISO': 999, 'NEISO': 999, 'MISO': 258, 'SPP': 245},
    },
}

# New-build fossil heat rates (MMBtu/MWh) — better than fleet average
# New CCGT: H/J-class turbines achieve 6.2-6.4 MMBtu/MWh
# New CT: Modern aero-derivative 9.5-10.0, frame 10.0-10.5
# New coal: Ultra-supercritical 8.8-9.2
NEW_BUILD_HEAT_RATES = {
    'gas_ccgt': 6.3,
    'gas_ct': 9.8,
    'coal': 9.5,
}

# New-build fossil VOM ($/MWh) — lower than aging fleet
NEW_BUILD_VOM = {
    'gas_ccgt': 2.50,
    'gas_ct': 4.00,
    'coal': 4.50,
}

# New-build fossil CO2 rates (tCO2/MWh) — derived from heat rates
NEW_BUILD_CO2_RATES = {
    'gas_ccgt': 0.053 * 6.3,   # ~0.334 tCO2/MWh
    'gas_ct': 0.053 * 9.8,     # ~0.519 tCO2/MWh
    'coal': 0.095 * 9.5,       # ~0.903 tCO2/MWh
}

# Minimum CF thresholds for investment decision — below this CF the unit
# doesn't generate enough revenue to justify the capital outlay.
# CCGT: 30% — modern H/J-class have fast ramp (10 MW/min) and low min gen,
#   so they are viable at lower CFs than older fleet-average units.
# CT: 5% — peakers only need to cover scarcity hours.
# Coal: 60% — high fixed costs and slow cycling require baseload dispatch.
# In single-trajectory runs, these are user-overridable via conditions dict.
NEW_BUILD_MIN_CF = {
    'gas_ccgt': 0.30,
    'gas_ct': 0.05,
    'coal': 0.60,
}

# Maximum annual new-build rate per ISO (GW/yr) — reflects permitting,
# construction labor, and interconnection queue throughput constraints.
# ISOs with larger grids and more active development pipelines get higher caps.
NEW_BUILD_MAX_GW_YR = {
    'CAISO': 3.0, 'ERCOT': 4.0, 'PJM': 5.0, 'NYISO': 2.0,
    'NEISO': 2.0, 'MISO': 4.0, 'SPP': 3.0,
}

# ============================================================================
# FOAK COST TABLES — First-of-a-kind costs before any learning curve
# ============================================================================
# Single value per technology × ISO. These represent the cost of the first
# commercial-scale project (pre-Wright's Law learning). Applied in Phase 2
# (demand growth sweep) only — Phase 1 (base year 2025) uses static L/M/H.
#
# Sources: Nuclear FOAK ~1.25× High (Vogtle-era), CCS ~1.20× High (Boundary Dam),
#   Geothermal ~1.35× High (Fervo EGS), LDES ~1.40× High (Form Energy pre-commercial),
#   H2 ~1.30× High (electrolysis + H2 turbine FOAK).

FOAK_NUCLEAR_NEWBUILD = {
    'CAISO': 175, 'ERCOT': 169, 'PJM': 200, 'NYISO': 212,
    'NEISO': 206, 'MISO': 194, 'SPP': 175,
}
FOAK_CCS_45Q_ON = {
    'CAISO': 138, 'ERCOT': 110, 'PJM': 122, 'NYISO': 154,
    'NEISO': 146, 'MISO': 115, 'SPP': 106,
}
FOAK_CCS_45Q_OFF = {
    'CAISO': 173, 'ERCOT': 145, 'PJM': 157, 'NYISO': 188,
    'NEISO': 181, 'MISO': 150, 'SPP': 140,
}
FOAK_GEOTHERMAL = 150  # CAISO only, $/MWh

# Offshore wind FOAK: pre-learning-curve costs for first commercial-scale projects.
# Fixed-bottom (NYISO/NEISO/PJM): 1.15× High (Vineyard Wind era, supply chain stress,
#   Jones Act vessel premiums). Sources: BNEF 2023-24 $114/MWh subsidized US,
#   Lazard v17 high-end $140/MWh.
# Floating (CAISO): 1.25× High (pre-commercial, no US floating experience,
#   port infrastructure not built). Sources: NREL FORCE model 2025 baseline ~$200+/MWh.
FOAK_OFFSHORE_WIND = {
    'CAISO': 250,  # floating: 1.25× $200 High
    'PJM':   129,  # fixed: 1.15× $112 High
    'NYISO': 144,  # fixed: 1.15× $125 High
    'NEISO': 136,  # fixed: 1.15× $118 High
}

# Storage FOAK: annualized capacity cost ($/MWh-cap), same units as LCOE_TABLES storage.
# Battery FOAK not needed — Wright's Law goes LCOE_TABLES → NOAK_BATTERY (decline over time).
# LDES: 1.40× High (Form Energy pre-commercial). H2: 1.30× High (first commercial H2 turbines).
FOAK_LDES = {
    'CAISO': 12264.00, 'ERCOT': 10512.00, 'PJM': 11563.20, 'NYISO': 13578.00,
    'NEISO': 12964.80, 'MISO': 11037.60, 'SPP': 10774.80,
}
FOAK_H2 = {
    'CAISO': 46603.20, 'ERCOT': 41084.40, 'PJM': 44150.40, 'NYISO': 51246.00,
    'NEISO': 48880.80, 'MISO': 42223.20, 'SPP': 40296.00,
}

# ============================================================================
# WRIGHT'S LAW NOAK TERMINAL COSTS
# ============================================================================

# Battery long-term floor: Wright's Law FORWARD from 2025 starting costs (LCOE_TABLES)
# toward these terminal NOAK values. Batteries are already at manufacturing scale,
# so their 2025 costs ARE the starting point, declining toward these floors.
# Calibrated to NREL 2050 projections: L=50%, M=56%, H=80% of 2025 starting cost.
# Sources: NREL ATB 2024 + Cost Projections for Utility-Scale Battery Storage 2025 Update.
NOAK_BATTERY = {
    'Low':    {'CAISO': 16906.80, 'ERCOT': 15242.40, 'PJM': 16206.00, 'NYISO': 17870.40, 'NEISO': 17344.80, 'MISO': 15855.60, 'SPP': 15417.60},
    'Medium': {'CAISO': 23214.00, 'ERCOT': 20936.40, 'PJM': 22250.40, 'NYISO': 24615.60, 'NEISO': 23914.80, 'MISO': 21812.40, 'SPP': 21199.20},
    'High':   {'CAISO': 42310.80, 'ERCOT': 38018.40, 'PJM': 40471.20, 'NYISO': 44676.00, 'NEISO': 43449.60, 'MISO': 39682.80, 'SPP': 38544.00},
}

NOAK_BATTERY8 = {
    'Low':    {'CAISO': 14366.40, 'ERCOT': 12964.80, 'PJM': 13753.20, 'NYISO': 15242.40, 'NEISO': 14804.40, 'MISO': 13490.40, 'SPP': 13140.00},
    'Medium': {'CAISO': 19885.20, 'ERCOT': 17870.40, 'PJM': 19009.20, 'NYISO': 21024.00, 'NEISO': 20410.80, 'MISO': 18658.80, 'SPP': 18133.20},
    'High':   {'CAISO': 36354.00, 'ERCOT': 32762.40, 'PJM': 34864.80, 'NYISO': 38456.40, 'NEISO': 37317.60, 'MISO': 34164.00, 'SPP': 33112.80},
}

# Offshore wind NOAK terminal floor
# Fixed-bottom: converges toward $50-65/MWh (NREL FORCE 2035: $53/MWh average).
# Floating (CAISO): converges toward $55-80/MWh (DOE Wind Shot $45 target in 2020$,
#   NREL FORCE 2035: $47-100/MWh range). Multiple doublings from 0.3 GW base.
NOAK_OFFSHORE_WIND = {
    'Low':    {'CAISO': 55, 'PJM': 50, 'NYISO': 52, 'NEISO': 50},
    'Medium': {'CAISO': 72, 'PJM': 62, 'NYISO': 65, 'NEISO': 63},
    'High':   {'CAISO': 100, 'PJM': 82, 'NYISO': 88, 'NEISO': 85},
}

# ============================================================================
# LEARNING CURVE PARAMETERS — Wright's Law FOAK→NOAK by toggle level
# ============================================================================
# Paired adoption speed + NOAK optimism: L=Fast/Optimistic, M=Central, H=Slow/Pessimistic.
# Each technology has its own timeline (CCS/geo more mature → slightly faster).
# Exponent 0.6 produces concave ramp: steep initial drop, asymptotic approach to NOAK.
#
# Format: {toggle_level: (foak_start_year, noak_year)}
LEARNING_PARAMS = {
    'nuclear': {'L': (2028, 2036), 'M': (2030, 2040), 'H': (2036, 2048)},
    'ccs':     {'L': (2028, 2036), 'M': (2030, 2040), 'H': (2036, 2048)},
    'geo':     {'L': (2028, 2036), 'M': (2030, 2040), 'H': (2036, 2048)},
    'ldes':    {'L': (2028, 2036), 'M': (2030, 2040), 'H': (2036, 2048)},
    'h2':      {'L': (2028, 2036), 'M': (2030, 2040), 'H': (2036, 2048)},
    # Battery: Wright's Law from 2025 starting cost → NOAK terminal floor.
    # Slower decline — on the mature part of the curve, not FOAK steep drops.
    'bat4':    {'L': (2025, 2042), 'M': (2025, 2048), 'H': (2025, 2050)},
    'bat8':    {'L': (2025, 2040), 'M': (2025, 2046), 'H': (2025, 2050)},
    # Offshore wind (fixed-bottom): ~8.8% learning rate (NREL ATB Moderate).
    'offshore_wind_fixed': {'L': (2026, 2034), 'M': (2028, 2038), 'H': (2032, 2045)},
    # Offshore wind (floating): ~11.5% learning rate (NREL ATB Moderate).
    'offshore_wind_float': {'L': (2029, 2037), 'M': (2031, 2042), 'H': (2035, 2050)},
}
LEARNING_EXPONENT = 0.6  # Wright's Law concave ramp


def learning_fraction(year, foak_start, noak_year):
    """Compute Wright's Law learning fraction for a given year.

    Returns 0.0 (pure FOAK) before foak_start, 1.0 (full NOAK) after noak_year,
    and a concave ramp in between: ((year - foak_start) / duration) ** 0.6.
    """
    if year < foak_start:
        return 0.0
    if year >= noak_year:
        return 1.0
    active = (year - foak_start) / (noak_year - foak_start)
    return active ** LEARNING_EXPONENT


def year_adjusted_cost(foak_cost, noak_cost, year, foak_start, noak_year):
    """Interpolate between FOAK and NOAK costs using Wright's Law.

    cost(year) = FOAK × (1 - frac) + NOAK × frac
    """
    frac = learning_fraction(year, foak_start, noak_year)
    return foak_cost * (1.0 - frac) + noak_cost * frac


# ============================================================================
# DEMAND GROWTH PARAMETERS
# Sources: EIA AEO 2025 (Reference + High/Low Economic Growth cases),
#   NERC 2024 LTRA, ERCOT 2025 LTLF, PJM 2025 Load Forecast,
#   Grid Strategies 2025 National Load Growth Report.
# Low = baseline economic/population growth (no incremental data center/electrification).
# Medium = confirmed large-load requests + moderate electrification (EIA Reference regionalized).
# High = full data center/AI load growth + accelerated electrification (economy-wide NZ pathway).
# ============================================================================

DEMAND_GROWTH_RATES = {
    'CAISO':  {'Low': 0.014, 'Medium': 0.019, 'High': 0.025},
    'ERCOT':  {'Low': 0.020, 'Medium': 0.035, 'High': 0.055},
    'PJM':    {'Low': 0.015, 'Medium': 0.024, 'High': 0.036},
    'NYISO':  {'Low': 0.013, 'Medium': 0.020, 'High': 0.044},
    'NEISO':  {'Low': 0.009, 'Medium': 0.018, 'High': 0.029},
    'MISO':   {'Low': 0.012, 'Medium': 0.022, 'High': 0.038},
    'SPP':    {'Low': 0.010, 'Medium': 0.018, 'High': 0.030},
}
DEMAND_GROWTH_YEARS = list(range(2026, 2051))  # kept for backward compat
DEMAND_GROWTH_LEVELS = ['Low', 'Medium', 'High']

# Threshold → target achievement year (interpolated from SBTi milestones:
# 2030→50%, 2035→70%, 2040→90%, 2045→95%, 2050→100%)
THRESHOLD_TARGET_YEARS = {
    10: 2026, 20: 2027, 30: 2028, 40: 2029,
    50: 2030, 55: 2031, 60: 2033, 65: 2034,
    70: 2035, 75: 2036, 80: 2037, 85: 2038, 87.5: 2039,
    90: 2040, 92.5: 2043,
    95: 2045, 97.5: 2048, 99: 2049, 99.5: 2049, 99.9: 2050, 99.99: 2050,
}

# Unique DG years (for efficient batching — group thresholds that share a year)
_DG_YEAR_TO_THRESHOLDS = {}
for _thr, _yr in THRESHOLD_TARGET_YEARS.items():
    _DG_YEAR_TO_THRESHOLDS.setdefault(_yr, []).append(_thr)
DG_UNIQUE_YEARS = sorted(_DG_YEAR_TO_THRESHOLDS.keys())

# Backward compat alias for OUTPUT_THRESHOLDS
OUTPUT_THRESHOLDS = THRESHOLDS


# ============================================================================
# SHARED CLEAN FIRM TRANCHE UTILITY
# ============================================================================
# Single source of truth for merit-order clean firm tranching:
#   Tranche 1: Nuclear uprates (existing fleet, cheapest, capped per ISO)
#   Tranche 2: Geothermal (CAISO only, capped at 39 TWh MINUS physics geo)
#   Tranche 3: Min(nuclear new-build, CCS) — CCS capped per ISO
#
# Works with both numpy arrays (step3a vectorized) and Python scalars (step5d).
# Import np at function level to avoid circular deps.

def compute_clean_firm_tranches(
    new_cf_twh,
    iso,
    firm_lev,
    ccs_lev,
    q45,
    tx_name,
    geo_lev=None,
    geo_physics_new_twh=0.0,
    uprate_cap_override=None,
    ccs_used_twh=0.0,
    uprate_used_twh=0.0,
    geo_used_twh=0.0,
    learning_curve_fn=None,
    target_year=2050,
    q45_realization=None,
):
    """Compute merit-order clean firm tranche allocation and costs.

    Args:
        new_cf_twh: New clean firm TWh needed (scalar or numpy array).
        iso: ISO region string.
        firm_lev: Firm cost level ('L'/'M'/'H').
        ccs_lev: CCS cost level ('L'/'M'/'H').
        q45: 45Q toggle ('0' or '1').
        tx_name: Transmission level name ('None'/'Low'/'Medium'/'High').
        geo_lev: Geothermal cost level (None for non-CAISO).
        geo_physics_new_twh: TWh of geothermal already consumed by physics dimension
            (CAISO only; reduces the 39 TWh cap for tranche 2).
        uprate_cap_override: Override uprate cap (TWh); defaults to UPRATE_CAP_TWH[iso].
        ccs_used_twh: Cumulative CCS TWh already used (for step5d sequential mode).
        uprate_used_twh: Cumulative uprate TWh already used.
        geo_used_twh: Cumulative geothermal TWh already used.
        learning_curve_fn: Optional fn(base, foak, noak, tech, level, year) → adjusted LCOE.
            If None, uses static LCOE tables (step3a Phase 1 behavior).
        target_year: Target year for learning curve (only used if learning_curve_fn provided).

    Returns:
        dict with keys:
            uprate_twh, uprate_cost, uprate_lcoe,
            geo_twh, geo_cost, geo_lcoe,
            nuclear_twh, nuclear_cost, nuclear_lcoe,
            ccs_tranche_twh, ccs_tranche_cost, ccs_lcoe,
            total_cost, remaining (should be 0 if all tranches work)
    """
    import numpy as np

    _max = np.maximum if hasattr(new_cf_twh, '__len__') else max
    _min = np.minimum if hasattr(new_cf_twh, '__len__') else min

    remaining = new_cf_twh
    tx_cf = get_tx('clean_firm', tx_name, iso)
    tx_ccs = get_tx('ccs_ccgt', tx_name, iso)

    # ── Tranche 1: Nuclear uprates ──
    uprate_cap = (UPRATE_CAP_TWH[iso] if uprate_cap_override is None
                  else uprate_cap_override)
    uprate_avail = _max(0, uprate_cap - uprate_used_twh)
    uprate_twh = _min(remaining, uprate_avail)
    uprate_lcoe = UPRATE_LCOE[firm_lev]  # No TX (grid-connected)
    uprate_cost = uprate_twh * uprate_lcoe  # in $/MWh × TWh → need ×1e6 by caller if needed
    remaining = _max(0, remaining - uprate_twh)

    # ── Tranche 2: Geothermal (CAISO only) ──
    geo_twh_val = 0.0 if not hasattr(new_cf_twh, '__len__') else np.zeros_like(new_cf_twh)
    geo_cost_val = 0.0 if not hasattr(new_cf_twh, '__len__') else np.zeros_like(new_cf_twh)
    geo_lcoe_val = 0.0
    if iso == 'CAISO' and geo_lev:
        # Reduce cap by physics geothermal to prevent double-counting
        geo_cap = _max(0, GEOTHERMAL_CAP_TWH - geo_physics_new_twh)
        geo_avail = _max(0, geo_cap - geo_used_twh)
        geo_twh_val = _min(remaining, geo_avail)

        geo_lcoe_val = GEOTHERMAL_LCOE[geo_lev]
        if learning_curve_fn:
            geo_lcoe_val = learning_curve_fn(
                geo_lcoe_val, FOAK_GEOTHERMAL, GEOTHERMAL_LCOE['L'],
                'geo', firm_lev, target_year)
        geo_lcoe_val += tx_cf
        geo_cost_val = geo_twh_val * geo_lcoe_val
        remaining = _max(0, remaining - geo_twh_val)

    # ── Tranche 3: Cheapest of nuclear new-build vs CCS ──
    nuclear_lcoe = NUCLEAR_NEWBUILD_LCOE[firm_lev][iso]
    if learning_curve_fn:
        nuclear_lcoe = learning_curve_fn(
            nuclear_lcoe, FOAK_NUCLEAR_NEWBUILD[iso], NUCLEAR_NEWBUILD_LCOE['L'][iso],
            'nuclear', firm_lev, target_year)
    nuclear_lcoe += tx_cf

    ccs_table = CCS_LCOE_45Q_ON if q45 == '1' else CCS_LCOE_45Q_OFF
    ccs_lcoe = ccs_table[ccs_lev][iso]
    if learning_curve_fn:
        foak_table = FOAK_CCS_45Q_ON if q45 == '1' else FOAK_CCS_45Q_OFF
        ccs_lcoe = learning_curve_fn(
            ccs_lcoe, foak_table[iso], ccs_table['L'][iso],
            'ccs', ccs_lev, target_year)

    # Apply 45Q realization probability — scales the credit by execution risk
    # q45_realization: 0.70 (conservative), 0.85 (base), 1.00 (full, default)
    if q45 == '1' and q45_realization is not None and q45_realization < 1.0:
        # Partial credit: CCS_OFF + realization_prob × (CCS_ON - CCS_OFF)
        # Equivalent to adding back (1 - prob) × 45Q credit to the ON price
        credit_haircut = CCS_45Q_CREDIT_PER_MWH * (1.0 - q45_realization)
        ccs_lcoe = ccs_lcoe + credit_haircut
    if iso == 'NEISO':
        ccs_lcoe += NEISO_CCS_GAS_ADDER
    ccs_lcoe += tx_ccs

    ccs_cap = CCS_CAP_TWH.get(iso, 9999.0)
    ccs_avail = _max(0, ccs_cap - ccs_used_twh)

    if nuclear_lcoe <= ccs_lcoe or (isinstance(ccs_avail, (int, float)) and ccs_avail <= 0):
        nuclear_twh = remaining
        ccs_tranche_twh = 0.0 if not hasattr(new_cf_twh, '__len__') else np.zeros_like(new_cf_twh)
        tranche3_cost = remaining * nuclear_lcoe
    else:
        ccs_tranche_twh = _min(remaining, ccs_avail)
        nuclear_twh = _max(0, remaining - ccs_tranche_twh)
        tranche3_cost = ccs_tranche_twh * ccs_lcoe + nuclear_twh * nuclear_lcoe

    return {
        'uprate_twh': uprate_twh,
        'uprate_cost': uprate_cost,
        'uprate_lcoe': uprate_lcoe,
        'geo_twh': geo_twh_val,
        'geo_cost': geo_cost_val,
        'geo_lcoe': geo_lcoe_val,
        'nuclear_twh': nuclear_twh,
        'nuclear_cost': nuclear_twh * nuclear_lcoe,
        'nuclear_lcoe': nuclear_lcoe,
        'ccs_tranche_twh': ccs_tranche_twh,
        'ccs_tranche_cost': ccs_tranche_twh * ccs_lcoe,
        'ccs_lcoe': ccs_lcoe,
        'total_cost': uprate_cost + geo_cost_val + tranche3_cost,
    }


# ============================================================================
# DISPATCH-SPECIFIC CONSTANTS
# ============================================================================
# Constants used by dispatch_utils.py and downstream analysis scripts.
# Migrated here from dispatch_utils.py for single-source-of-truth consistency.
#
# Sources:
#   HYDRO_CAPS: Maximum hydro capacity as TWh capacity / annual demand TWh,
#     from EIA-923 2023 generation data cross-referenced with USACE dam
#     capacity records. Values represent capacity-based upper bounds
#     (installed turbine capacity × annual hours, not historical generation).
#   COAL_CAP_TWH / OIL_CAP_TWH: 2025 baseline generation from EIA-923 annual
#     data (2023 actuals, adjusted for announced retirements through 2025).
#     Coal/oil capped at 2025 levels — no new coal/oil construction assumed.
#   NUCLEAR_SHARE_OF_CLEAN_FIRM: Fraction of clean_firm that is nuclear
#     (vs. other firm clean sources). CAISO = 0.70 due to Diablo Canyon
#     (2.25 GW) plus geothermal baseload. All others = 1.0 (nuclear only).
#   NUCLEAR_MONTHLY_CF: Monthly capacity factors from NRC PRIS data (2019–2023
#     average). Captures refueling outage seasonality — spring/fall dips
#     reflect 18-month refueling cycles staggered across fleet.
#     Reference: NRC Information Digest, NUREG-1350, Vol. 35 (2023).

HYDRO_CAPS = {
    'CAISO': 30, 'ERCOT': 5, 'PJM': 15, 'NYISO': 40, 'NEISO': 30,
    'MISO': 1.6, 'SPP': 4.3,
}

COAL_CAP_TWH = {
    'CAISO': 0.00, 'ERCOT': 67.58, 'PJM': 139.09, 'NYISO': 0.00, 'NEISO': 0.31,
    # MISO: Adjusted for announced retirements through 2027:
    # - Ameren Rush Island Units 1&2 (1,178 MW, retired Oct 2024)
    # - Xcel Sherco Unit 2 (680 MW, retired June 2023), Unit 3 (517 MW, retiring 2030)
    # - DTE Belle River (1,270 MW, retiring 2028-2029)
    # - Consumers Energy Campbell Units 1-3 (1,437 MW, retiring 2025)
    # Net reduction: ~5,000 MW × 60% CF × 8760 / 1e6 ≈ 26 TWh — phased:
    # 2025 effective reduction ≈ 13 TWh (half fleet already retired/retiring)
    # Sources: EIA-860M (Dec 2024), utility IRPs, MISO Generator Interconnection Queue
    'MISO': 112.0, 'SPP': 42.0,
}
OIL_CAP_TWH = {
    'CAISO': 0.60, 'ERCOT': 0.00, 'PJM': 4.59, 'NYISO': 0.15, 'NEISO': 1.29,
    'MISO': 0.50, 'SPP': 0.20,
}

NUCLEAR_SHARE_OF_CLEAN_FIRM = {
    'CAISO': 0.70, 'ERCOT': 1.0, 'PJM': 1.0, 'NYISO': 1.0, 'NEISO': 1.0,
    'MISO': 1.0, 'SPP': 1.0,
}

# Monthly capacity factors — NRC PRIS 2019–2023 average.
# Month 1 = January, 12 = December. Spring/fall dips = refueling outages.
NUCLEAR_MONTHLY_CF = {
    'CAISO': {1: 0.94, 2: 0.94, 3: 0.85, 4: 0.75, 5: 0.80, 6: 0.99,
              7: 1.0, 8: 1.0, 9: 0.90, 10: 0.78, 11: 0.82, 12: 0.94},
    'ERCOT': {1: 1.0, 2: 1.0, 3: 0.90, 4: 0.80, 5: 0.89, 6: 0.97,
              7: 0.97, 8: 0.96, 9: 0.88, 10: 0.79, 11: 0.85, 12: 1.0},
    'PJM':   {1: 1.0, 2: 1.0, 3: 0.92, 4: 0.85, 5: 0.87, 6: 0.98,
              7: 0.99, 8: 0.97, 9: 0.93, 10: 0.89, 11: 0.91, 12: 1.0},
    'NYISO': {1: 1.0, 2: 1.0, 3: 0.88, 4: 0.78, 5: 0.81, 6: 0.95,
              7: 0.96, 8: 0.94, 9: 0.85, 10: 0.75, 11: 0.79, 12: 1.0},
    'NEISO': {1: 1.0, 2: 0.99, 3: 0.92, 4: 0.83, 5: 0.88, 6: 0.96,
              7: 0.97, 8: 0.95, 9: 0.88, 10: 0.82, 11: 0.85, 12: 1.0},
    'MISO':  {1: 1.0, 2: 1.0, 3: 0.92, 4: 0.84, 5: 0.87, 6: 0.98,
              7: 0.99, 8: 0.97, 9: 0.93, 10: 0.88, 11: 0.91, 12: 1.0},
    'SPP':   {1: 1.0, 2: 1.0, 3: 0.90, 4: 0.80, 5: 0.88, 6: 0.97,
              7: 0.97, 8: 0.96, 9: 0.88, 10: 0.80, 11: 0.85, 12: 1.0},
}


# ============================================================================
# WRIGHT'S LAW — ENDOGENOUS DEPLOYMENT-BASED LEARNING CURVES
# ============================================================================
# Toggle: when True, cumulative GW is updated at the end of each simulation
# year and LCOE is recomputed from the updated learning-curve position before
# the next year's deployment loop.  When False, cumulative GW stays frozen at
# the 2025 baseline (static comparison mode — original behavior).
ENDOGENOUS_LEARNING = True

# Endogenous learning model from Wright (1936): cost declines as a power law
# of cumulative production. Used by step8 procurement strategies and step10
# SBTi target analysis.
#
# Sources:
#   Cumulative GW baselines: IRENA Renewable Capacity Statistics 2025,
#     Global Nuclear Power Tracker (Ember), Global CCS Institute Status Report 2024.
#   Learning rates: NREL ATB 2024 (solar/wind/battery), Rubin et al. (2015)
#     "The cost of CO2 capture and storage" (CCS), Breakthrough Energy (2024)
#     "Advancing Long Duration Energy Storage" (LDES/H2).
#   Background GW projections: IEA World Energy Outlook 2024 (NZE scenario
#     for Fast, STEPS for Slow).
#
# Reference: Wright, T.P. (1936). "Factors Affecting the Cost of Airplanes."
#   Journal of the Aeronautical Sciences, 3(4), 122–128.

WRIGHT_CUMULATIVE_GW_2025 = {
    'nuclear': 2.0, 'ccs': 0.3, 'ldes': 0.01, 'h2': 0.1,
    'geothermal': 0.05, 'battery': 50.0, 'battery8': 50.0,
    'solar': 150.0, 'wind': 150.0, 'offshore_wind': 5.0,
}

# Learning rate = fractional cost reduction per doubling of cumulative capacity
WRIGHT_LEARNING_RATE = {
    'nuclear':       {'Fast': 0.15, 'Slow': 0.10},
    'ccs':           {'Fast': 0.12, 'Slow': 0.10},
    'ldes':          {'Fast': 0.20, 'Slow': 0.15},
    'h2':            {'Fast': 0.18, 'Slow': 0.12},
    'geothermal':    {'Fast': 0.20, 'Slow': 0.15},
    'battery':       {'Fast': 0.20, 'Slow': 0.18},
    'battery8':      {'Fast': 0.20, 'Slow': 0.18},
    'solar':         {'Fast': 0.0, 'Slow': 0.0},    # Mature — on flat part of curve
    'wind':          {'Fast': 0.0, 'Slow': 0.0},    # Mature — on flat part of curve
    'offshore_wind': {'Fast': 0.12, 'Slow': 0.08},
}

# Background learning: exogenous rest-of-world GW deployment by (2035, 2050)
# Fast = IEA NZE scenario; Slow = IEA STEPS scenario
WRIGHT_BACKGROUND_GW = {
    'nuclear':       {'Fast': (30, 150), 'Slow': (5, 20)},
    'ccs':           {'Fast': (10, 50),  'Slow': (2, 10)},
    'ldes':          {'Fast': (5, 30),   'Slow': (0.5, 5)},
    'h2':            {'Fast': (3, 20),   'Slow': (0.5, 3)},
    'geothermal':    {'Fast': (3, 15),   'Slow': (0.5, 3)},
    'battery':       {'Fast': (200, 800), 'Slow': (80, 300)},
    'battery8':      {'Fast': (50, 200), 'Slow': (20, 80)},
    'solar':         {'Fast': (500, 2000), 'Slow': (200, 800)},
    'wind':          {'Fast': (300, 1200), 'Slow': (100, 500)},
    'offshore_wind': {'Fast': (40, 150),  'Slow': (10, 50)},
}


# ============================================================================
# THRESHOLD-BASED LEARNING FRACTION
# ============================================================================
# Maps CFE threshold → deployment year → Wright's Law learning fraction.
# This is the "high-level" version used by scenario scripts. The low-level
# `learning_fraction(year, foak_start, noak_year)` above is the primitive.
#
# Scenario B (Hourly Matching): Learning starts 2030 (planned procurement),
#   reaches NOAK by 2040 (10-year learning window).
# Scenario A (Consequential): Deployment-gated — FOAK until first clean firm
#   is deployed, then 10-year learning from that deployment year.
# Default (neither A nor B): Conservative delayed learning, 2036–2048.

def threshold_learning_fraction(threshold, scenario='B', first_deployment_year=None):
    """Map CFE threshold to FOAK→NOAK learning fraction [0, 1].

    Uses THRESHOLD_TARGET_YEARS to convert threshold → year, then applies
    Wright's Law concave ramp (exponent 0.6).

    Args:
        threshold: CFE matching percentage (e.g. 90, 99.9).
        scenario: 'A' (consequential, deployment-gated) or 'B' (hourly, time-based).
        first_deployment_year: For Scenario A, the year clean firm was first deployed.
            If None and scenario='A', returns 0.0 (pure FOAK).

    Returns:
        float in [0, 1]: 0 = pure FOAK cost, 1 = full NOAK cost.
    """
    year = THRESHOLD_TARGET_YEARS.get(threshold, 2050)
    if scenario == 'B':
        foak_start, noak_year = 2030, 2040
    elif scenario == 'A':
        if first_deployment_year is None:
            return 0.0  # No clean firm deployed yet → FOAK
        foak_start = first_deployment_year
        noak_year = first_deployment_year + 10
    else:
        foak_start, noak_year = 2036, 2048
    return learning_fraction(year, foak_start, noak_year)


# ============================================================================
# RPS / CES FLOOR TARGETS
# ============================================================================
# State RPS mandates mapped to ISO regions (approximate weighted-average).
# Values are % clean energy floor by year.  Sources: DSIRE, LBNL RPS tracker.

RPS_TARGETS = {
    'CAISO':  {2025: 60, 2030: 60, 2035: 80, 2040: 90, 2045: 100},
    'NYISO':  {2025: 50, 2030: 70, 2035: 80, 2040: 90, 2045: 100},
    'NEISO':  {2025: 40, 2030: 50, 2035: 60, 2040: 75, 2045: 90},
    'PJM':    {2025: 20, 2030: 30, 2035: 40, 2040: 50, 2045: 60},
    'MISO':   {2025: 15, 2030: 20, 2035: 30, 2040: 40, 2045: 50},
    'SPP':    {2025: 10, 2030: 15, 2035: 20, 2040: 30, 2045: 40},
    'ERCOT':  {2025: 0, 2030: 0, 2035: 0, 2040: 0, 2045: 0},  # No state RPS
}


def get_rps_floor(iso, year=2025):
    """Return the RPS/CES floor (%) for an ISO region in a given year.

    Linearly interpolates between defined target years.
    """
    targets = RPS_TARGETS.get(iso, {})
    if not targets:
        return 0.0
    years = sorted(targets.keys())
    if year <= years[0]:
        return float(targets[years[0]])
    if year >= years[-1]:
        return float(targets[years[-1]])
    # Interpolate
    for i in range(len(years) - 1):
        if years[i] <= year <= years[i + 1]:
            frac = (year - years[i]) / (years[i + 1] - years[i])
            return targets[years[i]] + frac * (targets[years[i + 1]] - targets[years[i]])
    return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# CONFIDENCE ZONES — Trajectory projection reliability classification
# ══════════════════════════════════════════════════════════════════════════════
# The LMP engine is calibrated against 2024 SOM data.  Near-term (2025-2030)
# outputs are well-grounded; beyond 2035 demand-quantile pricing, capacity
# market degradation, and Wright's Law curves increasingly extrapolate outside
# their calibration domain.
#
# See Third_Party_Expert_Review.md Section 7, Priority 6.

CONFIDENCE_ZONES = {
    'high': {
        'start': 2025, 'end': 2030,
        'color': '#22C55E',
        'label': 'Calibrated',
        'tooltip': 'Based on calibrated 2024 market data and near-term policy environment',
    },
    'moderate': {
        'start': 2030, 'end': 2040,
        'color': '#F59E0B',
        'label': 'Moderate Extrapolation',
        'tooltip': 'Technology costs and market structure may diverge from calibration assumptions',
    },
    'low': {
        'start': 2040, 'end': 2060,
        'color': '#EF4444',
        'label': 'High Uncertainty',
        'tooltip': 'Multiple compounding uncertainties — treat as scenario exploration, not forecast',
    },
}

# Synthetic uncertainty bands when sweep P10/P90 data is unavailable.
# lmp_pct = relative band (e.g. 0.05 → ±5%), clean_pp = absolute pp band.
CONFIDENCE_UNCERTAINTY_BANDS = {
    'high':     {'lmp_pct': 0.05, 'clean_pp': 3},
    'moderate': {'lmp_pct': 0.15, 'clean_pp': 8},
    'low':      {'lmp_pct': 0.30, 'clean_pp': 15},
}


def get_confidence_zone(year: int) -> str:
    """Return confidence zone key ('high', 'moderate', 'low') for a projection year."""
    if year <= 2030:
        return 'high'
    elif year <= 2040:
        return 'moderate'
    return 'low'


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO PROBABILITY WEIGHTS — Prior beliefs for weighted percentiles
# ══════════════════════════════════════════════════════════════════════════════
# Each sweep dimension has per-level weights reflecting prior likelihood.
# Combined scenario weight = product of individual dimension weights.
# Weights within each dimension should sum to 1.0.
# Set all weights equal (uniform) to recover unweighted percentiles.

SCENARIO_WEIGHTS = {
    'demand': {
        'Low': 0.2,
        'Medium': 0.6,
        'High': 0.2,
    },
    'price': {
        'all_low': 0.10,
        'all_med': 0.50,
        'all_high': 0.10,
        'high_vre_low_firm': 0.15,
        'high_firm_low_vre': 0.15,
    },
    'ppa': {
        'Low': 0.2,
        'Medium': 0.6,
        'High': 0.2,
    },
    'gas_friction': {
        'Low': 0.2,
        'Medium': 0.5,
        'High': 0.3,
    },
    'queue_cap': {
        'Low': 0.25,
        'Medium': 0.50,
        'High': 0.25,
    },
    'new_fossil_cost': {
        'Low': 0.2,
        'Medium': 0.6,
        'High': 0.2,
    },
}


def adjust_confidence_for_triggers(year_zone: str, ipm_triggers: list) -> tuple:
    """Downgrade confidence zone based on IPM trigger severity.

    Rules:
    - Any trigger with severity='high' → cap at 'moderate'
    - 2+ triggers with severity='medium' → cap at 'moderate'
    - 'low' confidence is never upgraded (year-based floor applies)

    Returns:
        (adjusted_zone, was_adjusted) tuple
    """
    if not ipm_triggers:
        return year_zone, False

    # 'low' is already the floor — triggers can't make it worse
    if year_zone == 'low':
        return year_zone, False

    high_count = sum(1 for t in ipm_triggers
                     if (t.get('severity') if isinstance(t, dict) else getattr(t, 'severity', '')) == 'high')
    medium_count = sum(1 for t in ipm_triggers
                       if (t.get('severity') if isinstance(t, dict) else getattr(t, 'severity', '')) == 'medium')

    should_cap = high_count >= 1 or medium_count >= 2

    if should_cap and year_zone == 'high':
        return 'moderate', True

    return year_zone, False
