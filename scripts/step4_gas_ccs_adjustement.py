#!/usr/bin/env python3
"""
Step 4: Post-Processing — Gas & CCS Corrections & Overlays
============================================================
Applies corrections to Step 3 cost optimization results.

Pipeline position: Step 4 of 4
  Step 1 — PFS Generator (step1_pfs_generator.py)
  Step 2 — Efficient Frontier extraction (step2_efficient_frontier.py)
  Step 3 — Cost optimization (step3_cost_optimization.py)
  Step 4 — Post-processing (this file)

Corrections:
  1. NEISO winter gas pipeline constraint (+$13/MWh CCS, +$4/MWh wholesale)
  2. CCS vs LDES crossover analysis
  3. Gas capacity backup & resource adequacy

Input:  data/step3-cost-opt-parquets/step3_co_<ISO>.parquet (per-ISO, from Step 3)
Output: data/step4-gas-ccs-parquets/step4_<ISO>.parquet (per-ISO, corrected)
        data/step4-gas-ccs-parquets/step4_analysis.json (crossover/RA analysis)

Usage:
  python scripts/step4_gas_ccs_adjustement.py                 # Process all ISOs
  python scripts/step4_gas_ccs_adjustement.py --iso ERCOT     # Process single ISO
  python scripts/step4_gas_ccs_adjustement.py --iso ERCOT --iso PJM  # Multiple ISOs
  python scripts/step4_gas_ccs_adjustement.py --all            # Explicit all (default)

Dependencies: pandas, pyarrow (same as Step 3)

See SPEC.md for methodology documentation.
"""

import argparse
import collections
import functools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'scripts'))
DEFAULT_INPUT_DIR = os.path.join(SCRIPT_DIR, 'data', 'step3-cost-opt-parquets')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'data', 'step4-gas-ccs-parquets')
ANALYSIS_PATH = os.path.join(SCRIPT_DIR, 'data', 'step4-gas-ccs-parquets', 'step4_analysis.json')

ALL_ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']

from step3_cost_optimization import OUTPUT_THRESHOLDS as THRESHOLDS

# ══════════════════════════════════════════════════════════════════════════════
# COST TABLES (duplicated from optimizer for standalone operation)
# ══════════════════════════════════════════════════════════════════════════════

WHOLESALE_PRICES = {
    'CAISO': 30, 'ERCOT': 27, 'PJM': 34, 'NYISO': 42, 'NEISO': 41,
    'MISO': 30, 'SPP': 25,
}

WHOLESALE_FUEL_ADJUSTMENTS = {
    'CAISO': {'Low': -5, 'Medium': 0, 'High': 10},
    'ERCOT': {'Low': -7, 'Medium': 0, 'High': 12},
    'PJM':   {'Low': -6, 'Medium': 0, 'High': 11},
    'NYISO': {'Low': -4, 'Medium': 0, 'High': 8},
    'NEISO': {'Low': -4, 'Medium': 0, 'High': 8},
    'MISO':  {'Low': -6, 'Medium': 0, 'High': 11},
    'SPP':   {'Low': -5, 'Medium': 0, 'High': 10},
}

FULL_LCOE_TABLES = {
    'solar': {
        'Low':    {'CAISO': 45, 'ERCOT': 40, 'PJM': 50, 'NYISO': 70, 'NEISO': 62, 'MISO': 42, 'SPP': 38},
        'Medium': {'CAISO': 60, 'ERCOT': 54, 'PJM': 65, 'NYISO': 92, 'NEISO': 82, 'MISO': 56, 'SPP': 50},
        'High':   {'CAISO': 78, 'ERCOT': 70, 'PJM': 85, 'NYISO': 120, 'NEISO': 107, 'MISO': 73, 'SPP': 65},
    },
    'wind': {
        'Low':    {'CAISO': 55, 'ERCOT': 30, 'PJM': 47, 'NYISO': 61, 'NEISO': 55, 'MISO': 28, 'SPP': 25},
        'Medium': {'CAISO': 73, 'ERCOT': 40, 'PJM': 62, 'NYISO': 81, 'NEISO': 73, 'MISO': 37, 'SPP': 33},
        'High':   {'CAISO': 95, 'ERCOT': 52, 'PJM': 81, 'NYISO': 105, 'NEISO': 95, 'MISO': 48, 'SPP': 43},
    },
    'clean_firm': {
        'Low':    {'CAISO': 58, 'ERCOT': 56, 'PJM': 48, 'NYISO': 64, 'NEISO': 69, 'MISO': 50, 'SPP': 54},
        'Medium': {'CAISO': 79, 'ERCOT': 79, 'PJM': 68, 'NYISO': 86, 'NEISO': 92, 'MISO': 72, 'SPP': 76},
        'High':   {'CAISO': 115, 'ERCOT': 115, 'PJM': 108, 'NYISO': 136, 'NEISO': 143, 'MISO': 112, 'SPP': 118},
    },
    'ccs_ccgt': {
        'Low':    {'CAISO': 58, 'ERCOT': 52, 'PJM': 62, 'NYISO': 78, 'NEISO': 75, 'MISO': 55, 'SPP': 50},
        'Medium': {'CAISO': 86, 'ERCOT': 71, 'PJM': 79, 'NYISO': 99, 'NEISO': 96, 'MISO': 74, 'SPP': 68},
        'High':   {'CAISO': 115, 'ERCOT': 92, 'PJM': 102, 'NYISO': 128, 'NEISO': 122, 'MISO': 96, 'SPP': 88},
    },
    'offshore_wind': {
        'Low':    {'CAISO': 110, 'ERCOT': 0, 'PJM': 65, 'NYISO': 72, 'NEISO': 68, 'MISO': 0, 'SPP': 0},
        'Medium': {'CAISO': 150, 'ERCOT': 0, 'PJM': 85, 'NYISO': 95, 'NEISO': 90, 'MISO': 0, 'SPP': 0},
        'High':   {'CAISO': 200, 'ERCOT': 0, 'PJM': 112, 'NYISO': 125, 'NEISO': 118, 'MISO': 0, 'SPP': 0},
    },
    # ---- Storage: annualized capacity cost ($/MWh-cap) ----
    # Same format as step3 LCOE_TABLES. NOT LCOS. Annualized fixed costs of storage
    # capacity, normalized to demand. TX=0 for storage (baked into regional CAPEX).
    # Formula: CAPEX_kWh × (CRF + FOM_rate) / 8760 × 1000 × regional_mult
    # Source: NREL ATB 2024 component model. See step3 for full derivation.
    'battery': {
        'Low':    {'CAISO': 3.86, 'ERCOT': 3.48, 'PJM': 3.70, 'NYISO': 4.08, 'NEISO': 3.97, 'MISO': 3.62, 'SPP': 3.52},
        'Medium': {'CAISO': 4.75, 'ERCOT': 4.27, 'PJM': 4.55, 'NYISO': 5.02, 'NEISO': 4.87, 'MISO': 4.46, 'SPP': 4.33},
        'High':   {'CAISO': 6.03, 'ERCOT': 5.43, 'PJM': 5.78, 'NYISO': 6.38, 'NEISO': 6.20, 'MISO': 5.66, 'SPP': 5.50},
    },
    'battery8': {
        'Low':    {'CAISO': 3.30, 'ERCOT': 2.97, 'PJM': 3.16, 'NYISO': 3.49, 'NEISO': 3.39, 'MISO': 3.10, 'SPP': 3.01},
        'Medium': {'CAISO': 4.06, 'ERCOT': 3.66, 'PJM': 3.89, 'NYISO': 4.30, 'NEISO': 4.17, 'MISO': 3.81, 'SPP': 3.70},
        'High':   {'CAISO': 5.19, 'ERCOT': 4.67, 'PJM': 4.97, 'NYISO': 5.49, 'NEISO': 5.33, 'MISO': 4.87, 'SPP': 4.73},
    },
    'ldes': {
        'Low':    {'CAISO': 0.38, 'ERCOT': 0.33, 'PJM': 0.36, 'NYISO': 0.42, 'NEISO': 0.40, 'MISO': 0.34, 'SPP': 0.33},
        'Medium': {'CAISO': 0.63, 'ERCOT': 0.54, 'PJM': 0.59, 'NYISO': 0.70, 'NEISO': 0.66, 'MISO': 0.56, 'SPP': 0.55},
        'High':   {'CAISO': 1.00, 'ERCOT': 0.86, 'PJM': 0.94, 'NYISO': 1.11, 'NEISO': 1.06, 'MISO': 0.90, 'SPP': 0.88},
    },
    # Green H2: electrolysis + salt cavern + H2 turbine. 35% RTE.
    # Shares 'ldes_lvl' sensitivity toggle (both long-duration storage).
    'h2': {
        'Low':    {'CAISO': 1.98, 'ERCOT': 1.75, 'PJM': 1.88, 'NYISO': 2.18, 'NEISO': 2.08, 'MISO': 1.80, 'SPP': 1.72},
        'Medium': {'CAISO': 2.90, 'ERCOT': 2.56, 'PJM': 2.76, 'NYISO': 3.19, 'NEISO': 3.05, 'MISO': 2.63, 'SPP': 2.52},
        'High':   {'CAISO': 4.09, 'ERCOT': 3.61, 'PJM': 3.88, 'NYISO': 4.50, 'NEISO': 4.29, 'MISO': 3.71, 'SPP': 3.54},
    },
}

FULL_TRANSMISSION_TABLES = {
    'wind': {
        'None': {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 0, 'NEISO': 0, 'MISO': 0, 'SPP': 0},
        'Low': {'CAISO': 4, 'ERCOT': 3, 'PJM': 5, 'NYISO': 7, 'NEISO': 6, 'MISO': 4, 'SPP': 3},
        'Medium': {'CAISO': 8, 'ERCOT': 6, 'PJM': 10, 'NYISO': 14, 'NEISO': 12, 'MISO': 7, 'SPP': 5},
        'High': {'CAISO': 14, 'ERCOT': 10, 'PJM': 18, 'NYISO': 22, 'NEISO': 20, 'MISO': 12, 'SPP': 9},
    },
    'solar': {
        'None': {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 0, 'NEISO': 0, 'MISO': 0, 'SPP': 0},
        'Low': {'CAISO': 1, 'ERCOT': 1, 'PJM': 2, 'NYISO': 3, 'NEISO': 3, 'MISO': 1, 'SPP': 1},
        'Medium': {'CAISO': 3, 'ERCOT': 3, 'PJM': 5, 'NYISO': 7, 'NEISO': 6, 'MISO': 3, 'SPP': 2},
        'High': {'CAISO': 6, 'ERCOT': 5, 'PJM': 9, 'NYISO': 12, 'NEISO': 10, 'MISO': 5, 'SPP': 4},
    },
    'clean_firm': {
        'None': {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 0, 'NEISO': 0, 'MISO': 0, 'SPP': 0},
        'Low': {'CAISO': 1, 'ERCOT': 1, 'PJM': 1, 'NYISO': 2, 'NEISO': 2, 'MISO': 1, 'SPP': 1},
        'Medium': {'CAISO': 3, 'ERCOT': 2, 'PJM': 3, 'NYISO': 5, 'NEISO': 4, 'MISO': 2, 'SPP': 2},
        'High': {'CAISO': 6, 'ERCOT': 4, 'PJM': 6, 'NYISO': 9, 'NEISO': 7, 'MISO': 4, 'SPP': 4},
    },
    'ccs_ccgt': {
        'None': {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 0, 'NEISO': 0, 'MISO': 0, 'SPP': 0},
        'Low': {'CAISO': 1, 'ERCOT': 1, 'PJM': 1, 'NYISO': 2, 'NEISO': 2, 'MISO': 1, 'SPP': 1},
        'Medium': {'CAISO': 2, 'ERCOT': 2, 'PJM': 3, 'NYISO': 4, 'NEISO': 3, 'MISO': 2, 'SPP': 2},
        'High': {'CAISO': 4, 'ERCOT': 3, 'PJM': 5, 'NYISO': 7, 'NEISO': 6, 'MISO': 4, 'SPP': 3},
    },
    'offshore_wind': {
        'None': {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 0, 'NEISO': 0, 'MISO': 0, 'SPP': 0},
        'Low':  {'CAISO': 8, 'ERCOT': 0, 'PJM': 5, 'NYISO': 6, 'NEISO': 5, 'MISO': 0, 'SPP': 0},
        'Medium': {'CAISO': 15, 'ERCOT': 0, 'PJM': 10, 'NYISO': 12, 'NEISO': 10, 'MISO': 0, 'SPP': 0},
        'High': {'CAISO': 25, 'ERCOT': 0, 'PJM': 18, 'NYISO': 20, 'NEISO': 18, 'MISO': 0, 'SPP': 0},
    },
    # Storage TX = 0: regional variation already baked into annualized capacity costs
    'battery':  {'None': 0, 'Low': 0, 'Medium': 0, 'High': 0},
    'battery8': {'None': 0, 'Low': 0, 'Medium': 0, 'High': 0},
    'ldes':     {'None': 0, 'Low': 0, 'Medium': 0, 'High': 0},
    'h2':       {'None': 0, 'Low': 0, 'Medium': 0, 'High': 0},
    'hydro': {
        'None': {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 0, 'NEISO': 0, 'MISO': 0, 'SPP': 0},
        'Low': {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 0, 'NEISO': 0, 'MISO': 0, 'SPP': 0},
        'Medium': {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 0, 'NEISO': 0, 'MISO': 0, 'SPP': 0},
        'High': {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 0, 'NEISO': 0, 'MISO': 0, 'SPP': 0},
    },
}

GRID_MIX_SHARES = {
    'CAISO': {'clean_firm': 7.9, 'solar': 22.3, 'wind': 8.8, 'offshore_wind': 0, 'ccs_ccgt': 0, 'hydro': 9.5},
    'ERCOT': {'clean_firm': 8.6, 'solar': 13.8, 'wind': 23.6, 'offshore_wind': 0, 'ccs_ccgt': 0, 'hydro': 0.1},
    'PJM':   {'clean_firm': 32.1, 'solar': 2.9, 'wind': 3.8, 'offshore_wind': 0, 'ccs_ccgt': 0, 'hydro': 1.8},
    'NYISO': {'clean_firm': 18.4, 'solar': 0.0, 'wind': 4.7, 'offshore_wind': 0, 'ccs_ccgt': 0, 'hydro': 15.9},
    'NEISO': {'clean_firm': 23.8, 'solar': 1.4, 'wind': 3.9, 'offshore_wind': 0, 'ccs_ccgt': 0, 'hydro': 4.4},
    'MISO':  {'clean_firm': 13.1, 'solar': 2.1, 'wind': 14.5, 'offshore_wind': 0, 'ccs_ccgt': 0, 'hydro': 1.6},
    'SPP':   {'clean_firm': 5.2, 'solar': 0.4, 'wind': 37.1, 'offshore_wind': 0, 'ccs_ccgt': 0, 'hydro': 4.3},
}

RESOURCE_TYPES = ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro']

# ── Pre-computed flat lookup dicts for O(1) access in hot loops ──
# Avoids repeated 3-level nested dict traversals like
# FULL_LCOE_TABLES[resource][level][iso] on every call.
_LCOE_FLAT = {}    # key: (resource, level, iso) → float
_TX_FLAT = {}      # key: (resource, level, iso) → float

_ALL_ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']

def _build_flat_lookups():
    """Build flat (resource, level, iso) → cost lookup dicts at module load."""
    for resource, level_dict in FULL_LCOE_TABLES.items():
        for level, iso_dict in level_dict.items():
            for iso, val in iso_dict.items():
                _LCOE_FLAT[(resource, level, iso)] = val
    for resource, level_dict in FULL_TRANSMISSION_TABLES.items():
        for level, val_or_dict in level_dict.items():
            if isinstance(val_or_dict, dict):
                for iso, val in val_or_dict.items():
                    _TX_FLAT[(resource, level, iso)] = val
            else:
                # Scalar value (e.g., 0 for storage with TX baked into CAPEX)
                for iso in _ALL_ISOS:
                    _TX_FLAT[(resource, level, iso)] = val_or_dict

_build_flat_lookups()

# 45Q credit: corrected value
FOURTY_FIVE_Q_OFFSET_CORRECTED = 27.5  # $85/ton × 0.323 tCO2/MWh captured
FOURTY_FIVE_Q_OFFSET_ORIGINAL = 29.0   # What the optimizer used

# CCS LCOE decomposition (NETL Baseline Rev 4a)
CCS_CAPITAL_SHARE = 0.55    # Capital recovery portion of LCOE
CCS_FIXED_OM_SHARE = 0.08   # Fixed O&M (scales with CF)
CCS_FUEL_SHARE = 0.30       # Fuel (constant per MWh)
CCS_VOM_TS_SHARE = 0.07     # Variable O&M + T&S (constant per MWh)
CCS_REFERENCE_CF = 0.85     # NETL reference capacity factor

# NEISO winter gas pipeline constraint (Algonquin Citygates)
NEISO_CCS_GAS_ADDER = 13.13    # $/MWh annualized: 7 HR × $7.50 premium × 0.25 winter fraction
NEISO_WHOLESALE_ADDER = 4.0    # $/MWh annualized: winter gas on marginal pricing

# ══════════════════════════════════════════════════════════════════════════════
# GAS CAPACITY BACKUP & RESOURCE ADEQUACY
# ══════════════════════════════════════════════════════════════════════════════

RESOURCE_ADEQUACY_MARGIN = 0.15  # 15% reserve margin

PEAK_DEMAND_MW = {
    'CAISO': 43860, 'ERCOT': 83597, 'PJM': 160560, 'NYISO': 31857, 'NEISO': 25898,
    'MISO': 127125, 'SPP': 54368,
}

EXISTING_GAS_CAPACITY_MW = {
    'CAISO': 37000, 'ERCOT': 55000, 'PJM': 75000, 'NYISO': 18000,
    'NEISO': 14000, 'MISO': 68000, 'SPP': 32000,
}

NEW_CCGT_COST_KW_YR = {
    'CAISO': 112, 'ERCOT': 89, 'PJM': 99, 'NYISO': 114,
    'NEISO': 105, 'MISO': 95, 'SPP': 88,
}

EXISTING_GAS_FOM_KW_YR = {
    'CAISO': 16, 'ERCOT': 13, 'PJM': 14, 'NYISO': 17, 'NEISO': 15,
    'MISO': 14, 'SPP': 13,
}

PEAK_CAPACITY_CREDITS = {
    'clean_firm': 1.0, 'solar': 0.30, 'wind': 0.10, 'offshore_wind': 0.25, 'ccs_ccgt': 0.90,
    'hydro': 0.50, 'battery': 0.95, 'battery8': 0.95, 'ldes': 0.90,
    'h2': 0.85,  # H2 turbine: dispatchable but slower ramp than gas/battery
}

GAS_AVAILABILITY_FACTOR = {
    'CAISO': 0.88, 'ERCOT': 0.83, 'PJM': 0.82, 'NYISO': 0.82,
    'NEISO': 0.85, 'MISO': 0.84, 'SPP': 0.84,
}

NEISO_PIPELINE_GAS_CAPACITY_MW = 8300
NEISO_PIPELINE_EXPANSION_COST_MW_YR = 2400

# Regional demand (TWh → MWh) for ISOs not yet in parquet
REGIONAL_DEMAND_MWH = {
    'CAISO': 224.039e6, 'ERCOT': 488.020e6, 'PJM': 843.331e6,
    'NYISO': 151.599e6, 'NEISO': 115.336e6, 'MISO': 660.000e6, 'SPP': 296.000e6,
}


# ══════════════════════════════════════════════════════════════════════════════
# PARQUET I/O — Load per-ISO parquets → nested dict
# ══════════════════════════════════════════════════════════════════════════════

def load_from_parquets(input_dir, isos):
    """
    Read per-ISO Step 3 parquets and reconstruct the nested dict format
    used by all processing functions.

    Parquet schema (flat):
        iso, threshold, scenario, annual_demand_mwh,
        mix_clean_firm, mix_solar, mix_wind, mix_offshore_wind, mix_ccs_ccgt, mix_hydro,
        hourly_match_score,
        battery_dispatch_pct, battery8_dispatch_pct, ldes_dispatch_pct, h2_dispatch_pct,
        cost_total_cost, cost_effective_cost, cost_incremental, cost_wholesale,
        tranche_cf_existing_twh, tranche_uprate_twh, tranche_geo_twh,
        tranche_nuclear_newbuild_twh, tranche_ccs_tranche_twh, tranche_new_cf_twh,
        gas_gas_backup_needed_mw, gas_existing_gas_used_mw, gas_new_gas_build_mw,
        gas_gas_cost_per_mwh, gas_clean_peak_capacity_mw, gas_ra_peak_mw

    Nested dict output:
        data['results'][iso]['annual_demand_mwh'] = float
        data['results'][iso]['thresholds'][t_str]['scenarios'][scenario_key] = {
            'resource_mix': {clean_firm, solar, wind, offshore_wind, ccs_ccgt, hydro},
            'costs': {total_cost, effective_cost, incremental, wholesale},
            'tranche_costs': {cf_existing_twh, uprate_twh, ...},
            'hourly_match_score',
            'battery_dispatch_pct', 'battery8_dispatch_pct', 'ldes_dispatch_pct', 'h2_dispatch_pct',
            'gas_backup_step3': {gas_backup_needed_mw, ...},
        }
    """
    data = {'results': {}}
    loaded_count = 0

    for iso in isos:
        parquet_path = os.path.join(input_dir, f'step3_co_{iso}.parquet')
        if not os.path.exists(parquet_path):
            print(f"  WARNING: {parquet_path} not found — skipping {iso}")
            continue

        table = pq.read_table(parquet_path)
        col_names = set(table.column_names)
        print(f"  Loaded {iso}: {table.num_rows:,} rows from {parquet_path} "
              f"({os.path.getsize(parquet_path) / 1024:.0f} KB)")

        # Convert to pandas for fast columnar + row access (much faster than
        # per-column .to_pylist() for wide tables with many rows)
        df = table.to_pandas()
        n_rows = len(df)

        # Extract annual_demand_mwh (same for all rows of this ISO)
        annual_demand = float(df['annual_demand_mwh'].iloc[0]) if 'annual_demand_mwh' in col_names else REGIONAL_DEMAND_MWH.get(iso, 0)

        iso_data = {
            'annual_demand_mwh': annual_demand,
            'thresholds': {},
        }

        # Identify prefix-based columns once
        tranche_cols = [c for c in col_names if c.startswith('tranche_')]
        gas_cols = [c for c in col_names if c.startswith('gas_')]

        # Pre-extract numpy arrays for hot columns (avoids repeated pandas indexing)
        threshold_arr = df['threshold'].values
        scenario_arr = df['scenario'].values

        mix_arrays = {}
        for rtype in ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro']:
            col = f'mix_{rtype}'
            mix_arrays[rtype] = df[col].values if col in col_names else None

        cost_arrays = {}
        for ckey in ['total_cost', 'effective_cost', 'incremental', 'wholesale']:
            col = f'cost_{ckey}'
            cost_arrays[ckey] = df[col].values if col in col_names else None

        tranche_arrays = {col: df[col].values for col in tranche_cols}
        gas_arrays = {col: df[col].values for col in gas_cols}

        scalar_arrays = {}
        for scol in ['hourly_match_score',
                     'battery_dispatch_pct', 'battery8_dispatch_pct',
                     'ldes_dispatch_pct', 'h2_dispatch_pct']:
            scalar_arrays[scol] = df[scol].values if scol in col_names else None

        # Group by threshold using numpy for speed
        unique_thresholds = np.unique(threshold_arr)

        for threshold in unique_thresholds:
            threshold_val = float(threshold)
            # Handle float thresholds like 87.5, 92.5, 97.5
            if threshold_val != int(threshold_val):
                t_str = str(threshold_val)
            else:
                t_str = str(int(threshold_val))

            row_mask = threshold_arr == threshold
            row_indices = np.where(row_mask)[0]

            scenarios = {}

            # Pre-extract scalar arrays once outside threshold loop
            match_arr = scalar_arrays['hourly_match_score']
            batt_arr = scalar_arrays['battery_dispatch_pct']
            batt8_arr = scalar_arrays['battery8_dispatch_pct']
            ldes_arr = scalar_arrays['ldes_dispatch_pct']
            h2_arr = scalar_arrays['h2_dispatch_pct']

            # Vectorized: extract sub-arrays for this threshold's rows
            sub_sc_keys = scenario_arr[row_indices]

            # Build resource mix arrays for all rows at once
            sub_mix = {rtype: (mix_arrays[rtype][row_indices].astype(int)
                               if mix_arrays[rtype] is not None
                               else np.zeros(len(row_indices), dtype=int))
                       for rtype in ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro']}

            # Build cost arrays for all rows at once
            sub_cost = {ckey: (cost_arrays[ckey][row_indices].astype(float)
                               if cost_arrays[ckey] is not None
                               else np.zeros(len(row_indices)))
                        for ckey in ['total_cost', 'effective_cost', 'incremental', 'wholesale']}

            # Extract scalar values vectorized
            sub_match = match_arr[row_indices].astype(float) if match_arr is not None else np.zeros(len(row_indices))
            sub_batt = np.round(batt_arr[row_indices].astype(float), 4) if batt_arr is not None else np.zeros(len(row_indices))
            sub_batt8 = np.round(batt8_arr[row_indices].astype(float), 4) if batt8_arr is not None else np.zeros(len(row_indices))
            sub_ldes = np.round(ldes_arr[row_indices].astype(float), 4) if ldes_arr is not None else np.zeros(len(row_indices))
            sub_h2 = np.round(h2_arr[row_indices].astype(float), 4) if h2_arr is not None else np.zeros(len(row_indices))

            # Pre-extract tranche sub-arrays and detect NaNs vectorized
            sub_tranche = {}
            tranche_nan_masks = {}
            for col in tranche_cols:
                arr = tranche_arrays[col][row_indices].astype(float)
                sub_tranche[col] = arr
                tranche_nan_masks[col] = np.isnan(arr)

            # Pre-extract gas sub-arrays and detect NaNs vectorized
            sub_gas = {}
            gas_nan_masks = {}
            gas_is_float = {}
            for col in gas_cols:
                raw = gas_arrays[col][row_indices]
                try:
                    arr = raw.astype(float)
                    sub_gas[col] = arr
                    gas_nan_masks[col] = np.isnan(arr)
                    gas_is_float[col] = True
                except (ValueError, TypeError):
                    sub_gas[col] = raw
                    gas_nan_masks[col] = np.zeros(len(row_indices), dtype=bool)
                    gas_is_float[col] = False

            # Build scenario dicts using pre-extracted sub-arrays (index into flat arrays)
            for j in range(len(row_indices)):
                resource_mix = {rtype: int(sub_mix[rtype][j]) for rtype in sub_mix}
                costs = {ckey: float(sub_cost[ckey][j]) for ckey in sub_cost}

                tranche_costs = {}
                for col in tranche_cols:
                    if not tranche_nan_masks[col][j]:
                        tranche_costs[col[len('tranche_'):]] = float(sub_tranche[col][j])

                gas_step3 = {}
                for col in gas_cols:
                    if not gas_nan_masks[col][j]:
                        val = sub_gas[col][j]
                        gas_step3[col[len('gas_'):]] = float(val) if gas_is_float[col] else int(val)

                scenarios[sub_sc_keys[j]] = {
                    'resource_mix': resource_mix,
                    'costs': costs,
                    'tranche_costs': tranche_costs,
                    'hourly_match_score': float(sub_match[j]),
                    'battery_dispatch_pct': float(sub_batt[j]),
                    'battery8_dispatch_pct': float(sub_batt8[j]),
                    'ldes_dispatch_pct': float(sub_ldes[j]),
                    'h2_dispatch_pct': float(sub_h2[j]),
                    'gas_backup_step3': gas_step3,
                }

            iso_data['thresholds'][t_str] = {'scenarios': scenarios}

        data['results'][iso] = iso_data
        loaded_count += 1

    print(f"  Loaded {loaded_count} ISOs from parquet files")
    return data


def save_to_parquets(data, output_dir, isos):
    """
    Flatten the nested dict back to per-ISO parquets with step 4 additions.

    New columns added by step 4 (beyond step 3's original columns):
        neiso_gas_adj_total_cost, neiso_gas_adj_effective_cost,
        neiso_gas_adj_incremental, neiso_gas_adj_wholesale,
        neiso_gas_no45q_total_cost, neiso_gas_no45q_effective_cost,
        neiso_gas_no45q_incremental, neiso_gas_no45q_wholesale,
        no45q_total_cost, no45q_effective_cost, no45q_incremental, no45q_wholesale,
        no45q_crossover_cf, no45q_ccs_no45q_baseload, no45q_ldes_cost,
        ra_peak_demand_mw, ra_ra_peak_mw, ra_clean_peak_capacity_mw,
        ra_gas_backup_needed_mw, ra_existing_gas_used_mw, ra_new_gas_build_mw,
        ra_existing_gas_cost_per_mwh, ra_new_gas_cost_per_mwh,
        ra_gas_backup_cost_per_mwh, ra_total_system_cost_per_mwh,
        ra_incremental_with_new_gas, ra_clean_coverage_pct, ra_gas_availability_factor,
        ra_pipeline_deliverable_mw, ra_pipeline_shortfall_mw,
        ra_pipeline_expansion_cost_per_mwh, ra_pipeline_constrained,
    """
    os.makedirs(output_dir, exist_ok=True)

    for iso in isos:
        if iso not in data.get('results', {}):
            continue

        iso_data = data['results'][iso]
        annual_demand = iso_data.get('annual_demand_mwh', 0)
        rows = []

        for t_str, t_data in iso_data.get('thresholds', {}).items():
            for sc_key, sc in t_data.get('scenarios', {}).items():
                row = {
                    'iso': iso,
                    'threshold': float(t_str),
                    'scenario': sc_key,
                    'annual_demand_mwh': annual_demand,
                }

                # Resource mix
                mix = sc.get('resource_mix', {})
                for rtype in ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro']:
                    row[f'mix_{rtype}'] = mix.get(rtype, 0)

                # Physics
                row['hourly_match_score'] = sc.get('hourly_match_score', 0.0)
                row['battery_dispatch_pct'] = sc.get('battery_dispatch_pct', 0)
                row['battery8_dispatch_pct'] = sc.get('battery8_dispatch_pct', 0)
                row['ldes_dispatch_pct'] = sc.get('ldes_dispatch_pct', 0)
                row['h2_dispatch_pct'] = sc.get('h2_dispatch_pct', 0)

                # Costs (may have been updated by NEISO gas constraint)
                costs = sc.get('costs', {})
                for ckey in ['total_cost', 'effective_cost', 'incremental', 'wholesale']:
                    row[f'cost_{ckey}'] = costs.get(ckey, 0.0)

                # Tranche costs (passthrough from step 3)
                for k, v in sc.get('tranche_costs', {}).items():
                    row[f'tranche_{k}'] = v

                # Step 3 gas backup (passthrough)
                for k, v in sc.get('gas_backup_step3', {}).items():
                    row[f'gas_{k}'] = v

                # --- Step 4 additions ---

                # NEISO gas adjusted costs
                neiso_gas = sc.get('neiso_gas_adjusted', {})
                for ckey in ['total_cost', 'effective_cost', 'incremental', 'wholesale']:
                    row[f'neiso_gas_adj_{ckey}'] = neiso_gas.get(ckey, None)

                # NEISO gas + no-45Q costs
                neiso_no45q = sc.get('neiso_gas_no45q', {})
                for ckey in ['total_cost', 'effective_cost', 'incremental', 'wholesale']:
                    row[f'neiso_gas_no45q_{ckey}'] = neiso_no45q.get(ckey, None)

                # No-45Q overlay costs
                no45q = sc.get('no_45q_costs', {})
                for ckey in ['total_cost', 'effective_cost', 'incremental', 'wholesale']:
                    row[f'no45q_{ckey}'] = no45q.get(ckey, None)

                # CCS vs LDES crossover
                crossover = no45q.get('ccs_vs_ldes', {})
                row['no45q_crossover_cf'] = crossover.get('crossover_cf', None)
                row['no45q_ccs_no45q_baseload'] = crossover.get('ccs_no45q_baseload', None)
                row['no45q_ldes_cost'] = crossover.get('ldes_cost', None)

                # Resource adequacy (step 4's own gas backup calculation)
                gb = sc.get('gas_backup', {})
                ra_fields = [
                    'peak_demand_mw', 'ra_peak_mw', 'clean_peak_capacity_mw',
                    'gas_backup_needed_mw', 'existing_gas_used_mw', 'new_gas_build_mw',
                    'existing_gas_cost_per_mwh', 'new_gas_cost_per_mwh',
                    'gas_backup_cost_per_mwh', 'total_system_cost_per_mwh',
                    'incremental_with_new_gas', 'clean_coverage_pct',
                    'resource_adequacy_margin', 'gas_availability_factor',
                ]
                for field in ra_fields:
                    row[f'ra_{field}'] = gb.get(field, None)

                # NEISO pipeline constraint
                pipeline = gb.get('pipeline_constraint', {})
                row['ra_pipeline_deliverable_mw'] = pipeline.get('pipeline_deliverable_mw', None)
                row['ra_pipeline_shortfall_mw'] = pipeline.get('pipeline_shortfall_mw', None)
                row['ra_pipeline_expansion_cost_per_mwh'] = pipeline.get('pipeline_expansion_cost_per_mwh', None)
                row['ra_pipeline_constrained'] = pipeline.get('pipeline_constrained', None)

                rows.append(row)

        if not rows:
            continue

        # Build pyarrow table from list-of-dicts (batch conversion)
        table = pa.Table.from_pylist(rows)
        out_path = os.path.join(output_dir, f'step4_{iso}.parquet')
        pq.write_table(table, out_path, compression='zstd')
        print(f"  {out_path}: {len(rows):,} rows, "
              f"{os.path.getsize(out_path) / 1e6:.1f} MB")


# ══════════════════════════════════════════════════════════════════════════════
# PROCESSING FUNCTIONS (unchanged logic from original step 4)
# ══════════════════════════════════════════════════════════════════════════════

def medium_key(iso):
    """Return the all-Medium scenario key for an ISO."""
    geo = 'M' if iso == 'CAISO' else 'X'
    return f'MMMM_M_M_M1_{geo}'


@functools.lru_cache(maxsize=None)
def decode_scenario_key(key):
    """Decode scenario key into toggle levels.
    Returns: (renewable, firm, battery, ldes, fuel, tx, ccs, q45, geo)
    """
    level_map = {'L': 'Low', 'M': 'Medium', 'H': 'High', 'N': 'None', 'X': None}
    parts = key.split('_')
    gen_part = parts[0]
    renewable = level_map.get(gen_part[0], 'Medium')
    firm = level_map.get(gen_part[1], 'Medium')

    if len(gen_part) >= 4:
        battery = level_map.get(gen_part[2], 'Medium')
        ldes = level_map.get(gen_part[3], 'Medium')
    else:
        battery = level_map.get(gen_part[2], 'Medium')
        ldes = battery

    fuel = level_map.get(parts[1], 'Medium')
    tx = level_map.get(parts[2], 'Medium')

    if len(parts) >= 5:
        ccs_q45 = parts[3]
        ccs = level_map.get(ccs_q45[0], 'Medium')
        q45 = ccs_q45[1] if len(ccs_q45) > 1 else '1'
        geo_code = parts[4]
        geo = level_map.get(geo_code, None)
    else:
        ccs = firm
        q45 = '1'
        geo = None

    return renewable, firm, battery, ldes, fuel, tx, ccs, q45, geo


def ccs_lcoe_corrected_45q(iso, firm_level):
    """Get CCS LCOE with corrected 45Q offset ($27.5 instead of $29)."""
    table_lcoe = FULL_LCOE_TABLES['ccs_ccgt'][firm_level][iso]
    return table_lcoe + (FOURTY_FIVE_Q_OFFSET_ORIGINAL - FOURTY_FIVE_Q_OFFSET_CORRECTED)


def ccs_lcoe_no45q(iso, firm_level):
    """Get CCS LCOE without any 45Q offset (gross LCOE)."""
    table_lcoe = FULL_LCOE_TABLES['ccs_ccgt'][firm_level][iso]
    return table_lcoe + FOURTY_FIVE_Q_OFFSET_ORIGINAL


def ccs_lcoe_dispatchable(lcoe_no45q, capacity_factor):
    """CCS LCOE at a given capacity factor (dispatchable operation)."""
    fixed_share = CCS_CAPITAL_SHARE + CCS_FIXED_OM_SHARE   # 0.63
    variable_share = CCS_FUEL_SHARE + CCS_VOM_TS_SHARE      # 0.37

    if capacity_factor <= 0.01:
        return lcoe_no45q * 10

    cf_ratio = CCS_REFERENCE_CF / capacity_factor
    return lcoe_no45q * (fixed_share * cf_ratio + variable_share)


@functools.lru_cache(maxsize=None)
def _cached_base_maps(scenario_key, iso):
    """Pre-compute base LCOE and TX maps for a (scenario_key, iso) pair.

    Returns:
        (lcoe_base, tx_map, firm_level, fuel_level, wholesale_base)
    where lcoe_base is a dict of resource → base LCOE (without CCS 45Q logic),
    tx_map is resource → transmission cost, and wholesale_base is the fuel-
    adjusted wholesale price (before any NEISO adder).

    Cached because the same scenario_key × iso pair is evaluated multiple
    times (with/without 45Q, with/without gas adder).
    """
    renewable, firm, battery, ldes, fuel, tx, ccs, q45, geo = decode_scenario_key(scenario_key)

    lcoe_base = {
        'solar': _LCOE_FLAT[('solar', renewable, iso)],
        'wind': _LCOE_FLAT[('wind', renewable, iso)],
        'clean_firm': _LCOE_FLAT[('clean_firm', firm, iso)],
        'battery': _LCOE_FLAT[('battery', battery, iso)],
        'battery8': _LCOE_FLAT[('battery8', battery, iso)],
        'ldes': _LCOE_FLAT[('ldes', ldes, iso)],
        'h2': _LCOE_FLAT[('h2', ldes, iso)],  # H2 shares LDES toggle
        'hydro': 0,
    }

    tx_map = {rtype: _TX_FLAT[(rtype, tx, iso)]
              for rtype in FULL_TRANSMISSION_TABLES}

    wholesale_base = WHOLESALE_PRICES[iso] + WHOLESALE_FUEL_ADJUSTMENTS[iso][fuel]

    return lcoe_base, tx_map, firm, fuel, tx, wholesale_base


def compute_costs_for_scenario(iso, resource_mix, battery_pct,
                                ldes_pct, match_score, scenario_key,
                                apply_45q=True, neiso_gas_adder=False,
                                tranche_cf_lcoe=None,
                                existing_shares_override=None,
                                h2_pct=0, battery8_pct=0):
    """Recalculate costs for a scenario with optional corrections.

    Args:
        existing_shares_override: If provided, overrides GRID_MIX_SHARES[iso]
            for demand-growth scenarios where existing generation is a smaller
            fraction of total demand.
    """
    # Use cached base maps — avoids redundant decode + nested dict lookups
    lcoe_base, tx_map, firm, fuel, tx, wholesale_base = _cached_base_maps(scenario_key, iso)

    # Build per-call lcoe_map from cached base (shallow copy + overrides)
    lcoe_map = dict(lcoe_base)
    if tranche_cf_lcoe is not None:
        lcoe_map['clean_firm'] = tranche_cf_lcoe

    if apply_45q:
        lcoe_map['ccs_ccgt'] = ccs_lcoe_corrected_45q(iso, firm)
    else:
        base_no45q = ccs_lcoe_no45q(iso, firm)
        ccs_share = resource_mix.get('ccs_ccgt', 0) / 100.0
        ccs_fraction_of_demand = ccs_share
        if ccs_fraction_of_demand > 0.01:
            estimated_cf = min(0.85, max(0.20, ccs_fraction_of_demand * 0.8))
            lcoe_map['ccs_ccgt'] = ccs_lcoe_dispatchable(base_no45q, estimated_cf)
        else:
            lcoe_map['ccs_ccgt'] = base_no45q

    wholesale = wholesale_base
    wholesale = max(5, wholesale)

    if neiso_gas_adder and iso == 'NEISO':
        wholesale += NEISO_WHOLESALE_ADDER

    grid_shares = existing_shares_override if existing_shares_override else GRID_MIX_SHARES[iso]
    total_cost_per_demand = 0.0

    for rtype in RESOURCE_TYPES:
        pct = resource_mix.get(rtype, 0)
        if pct <= 0:
            continue

        resource_pct_of_demand = float(pct)
        existing_share = grid_shares.get(rtype, 0)
        existing_pct = min(resource_pct_of_demand, existing_share)
        new_pct = max(0, resource_pct_of_demand - existing_share)

        if rtype == 'hydro':
            # Hydro is always existing, sunk fleet — $0 cost
            cost_per_demand = 0.0
        else:
            new_build_cost = lcoe_map.get(rtype, 0) + tx_map.get(rtype, 0)
            if neiso_gas_adder and iso == 'NEISO' and rtype == 'ccs_ccgt':
                new_build_cost += NEISO_CCS_GAS_ADDER
            # Existing resources = $0 (sunk fleet). Only new-build costs.
            cost_per_demand = new_pct / 100.0 * new_build_cost

        total_cost_per_demand += cost_per_demand

    battery_cost_rate = lcoe_map['battery'] + tx_map.get('battery', 0)
    battery_cost = (battery_pct / 100.0) * battery_cost_rate
    total_cost_per_demand += battery_cost

    battery8_cost_rate = lcoe_map.get('battery8', lcoe_map['battery']) + tx_map.get('battery8', 0)
    battery8_cost = (battery8_pct / 100.0) * battery8_cost_rate
    total_cost_per_demand += battery8_cost

    ldes_cost_rate = lcoe_map['ldes'] + tx_map.get('ldes', 0)
    ldes_cost = (ldes_pct / 100.0) * ldes_cost_rate
    total_cost_per_demand += ldes_cost

    h2_cost_rate = lcoe_map.get('h2', 0) + tx_map.get('h2', 0)
    h2_cost = (h2_pct / 100.0) * h2_cost_rate
    total_cost_per_demand += h2_cost

    matched_fraction = match_score / 100.0 if match_score > 0 else 1.0
    effective_cost = total_cost_per_demand / matched_fraction

    return {
        'total_cost': round(total_cost_per_demand, 2),
        'effective_cost': round(effective_cost, 2),
        'incremental': round(effective_cost - wholesale, 2),
        'wholesale': wholesale,
    }


def get_tranche_cf_lcoe(scenario):
    """Extract tranche-effective clean firm LCOE from tranche data, if available."""
    tc = scenario.get('tranche_costs', {})
    lcoe = tc.get('effective_new_cf_lcoe')
    if lcoe and lcoe > 0:
        return lcoe
    return None


def add_neiso_gas_constraint(data):
    """Apply NEISO winter gas pipeline constraint."""
    print("\n  [1] NEISO Winter Gas Pipeline Constraint")

    iso = 'NEISO'
    if iso not in data['results']:
        print("      NEISO not in results — skipping")
        return 0

    thresholds_data = data['results'][iso].get('thresholds', {})
    adjustments = 0

    for t_str in thresholds_data:
        scenarios = thresholds_data[t_str].get('scenarios', {})
        for sk, scenario in scenarios.items():
            effective_key = scenario.get('tranche_optimal_source', sk)
            mix = scenario.get('resource_mix', {})

            tcl = get_tranche_cf_lcoe(scenario)
            neiso_costs = compute_costs_for_scenario(
                iso, mix,
                scenario.get('battery_dispatch_pct', 0),
                scenario.get('ldes_dispatch_pct', 0),
                scenario.get('hourly_match_score', 0),
                effective_key,
                apply_45q=True,
                neiso_gas_adder=True,
                tranche_cf_lcoe=tcl,
                h2_pct=scenario.get('h2_dispatch_pct', 0),
                battery8_pct=scenario.get('battery8_dispatch_pct', 0),
            )

            neiso_no45q_costs = compute_costs_for_scenario(
                iso, mix,
                scenario.get('battery_dispatch_pct', 0),
                scenario.get('ldes_dispatch_pct', 0),
                scenario.get('hourly_match_score', 0),
                effective_key,
                apply_45q=False,
                neiso_gas_adder=True,
                tranche_cf_lcoe=tcl,
                h2_pct=scenario.get('h2_dispatch_pct', 0),
                battery8_pct=scenario.get('battery8_dispatch_pct', 0),
            )

            scenario['neiso_gas_adjusted'] = neiso_costs
            scenario['neiso_gas_no45q'] = neiso_no45q_costs
            scenario['costs'] = neiso_costs

            if 'costs_detail' in scenario:
                detail = scenario['costs_detail']
                detail['effective_cost_per_useful_mwh'] = neiso_costs['effective_cost']
                detail['total_cost_per_demand_mwh'] = neiso_costs['total_cost']
                detail['incremental_above_baseline'] = neiso_costs['incremental']
                detail['baseline_wholesale_cost'] = neiso_costs['wholesale']

            scenario['no_45q_costs'] = neiso_no45q_costs
            adjustments += 1

    print(f"      {adjustments} NEISO scenarios adjusted for gas pipeline constraint")
    return adjustments


def add_no45q_overlay(data, run_isos):
    """Add without-45Q cost overlay to every scenario with CCS in the mix."""
    print("\n  [2] Without-45Q Toggle Layer")
    overlays = 0

    for iso in run_isos:
        if iso not in data['results']:
            continue
        # NEISO already handled by add_neiso_gas_constraint (its no_45q includes gas adder)
        if iso == 'NEISO':
            continue

        thresholds_data = data['results'][iso].get('thresholds', {})

        for t_str in thresholds_data:
            scenarios = thresholds_data[t_str].get('scenarios', {})
            for sk, scenario in scenarios.items():
                effective_key = sk
                mix = scenario.get('resource_mix', {})
                ccs_pct = mix.get('ccs_ccgt', 0)

                no45q_costs = compute_costs_for_scenario(
                    iso, mix,
                    scenario.get('battery_dispatch_pct', 0),
                    scenario.get('ldes_dispatch_pct', 0),
                    scenario.get('hourly_match_score', 0),
                    effective_key,
                    apply_45q=False,
                    neiso_gas_adder=False,
                    tranche_cf_lcoe=get_tranche_cf_lcoe(scenario),
                    h2_pct=scenario.get('h2_dispatch_pct', 0),
                    battery8_pct=scenario.get('battery8_dispatch_pct', 0),
                )

                scenario['no_45q_costs'] = no45q_costs

                if ccs_pct > 0:
                    # Use cached base maps to avoid redundant decode + nested lookups
                    lcoe_base_cr, tx_map_cr, firm_cr, _, _, _ = _cached_base_maps(effective_key, iso)
                    ldes_cost = lcoe_base_cr['ldes'] + tx_map_cr['ldes']
                    ccs_no45q_base = ccs_lcoe_no45q(iso, firm_cr)
                    ccs_tx = tx_map_cr['ccs_ccgt']

                    rhs = ldes_cost - ccs_tx - ccs_no45q_base * 0.37
                    if rhs > 0:
                        crossover_cf = ccs_no45q_base * 0.63 * 0.85 / rhs
                        crossover_cf = round(min(1.0, max(0.0, crossover_cf)), 3)
                    else:
                        crossover_cf = 0.0

                    scenario['no_45q_costs']['ccs_vs_ldes'] = {
                        'ccs_no45q_baseload': round(ccs_no45q_base + ccs_tx, 2),
                        'ldes_cost': round(ldes_cost, 2),
                        'crossover_cf': crossover_cf,
                        'ccs_cheaper_above_cf': crossover_cf,
                        'ldes_cheaper_below_cf': crossover_cf,
                    }

                overlays += 1

    print(f"      {overlays} scenarios with no-45Q overlay added")
    return overlays


def analyze_crossover(data, run_isos):
    """Analyze CCS vs LDES crossover and without-45Q cost curve impact."""
    print("\n  [3] CCS vs LDES Crossover Analysis")

    analysis = {'crossover_by_iso': {}, 'cost_curve_impact': {}}

    for iso in run_isos:
        if iso not in data['results']:
            continue

        thresholds_data = data['results'][iso].get('thresholds', {})
        med_key = medium_key(iso)

        crossovers = []
        curve_impact = []

        for threshold in THRESHOLDS:
            t_str = str(threshold) if threshold != int(threshold) else str(int(threshold))
            if t_str not in thresholds_data:
                continue
            scenarios = thresholds_data[t_str].get('scenarios', {})
            scenario = (scenarios.get(med_key) or
                        scenarios.get('MMM_M_M_M1_M') or scenarios.get('MMM_M_M_M1_X') or
                        scenarios.get('MMM_M_M'))
            if not scenario:
                continue

            costs_with = scenario.get('costs', {})
            costs_without = scenario.get('no_45q_costs', {})
            mix = scenario.get('resource_mix', {})
            ccs_pct = mix.get('ccs_ccgt', 0)

            eff_with = costs_with.get('effective_cost', 0)
            eff_without = costs_without.get('effective_cost', 0)

            entry = {
                'threshold': threshold,
                'ccs_pct': ccs_pct,
                'cost_with_45q': eff_with,
                'cost_without_45q': eff_without,
                'cost_delta': round(eff_without - eff_with, 2),
                'cost_increase_pct': round((eff_without - eff_with) / eff_with * 100, 1) if eff_with > 0 else 0,
            }

            crossover_data = costs_without.get('ccs_vs_ldes', {})
            if crossover_data:
                entry['crossover_cf'] = crossover_data.get('crossover_cf', 0)
                entry['ccs_baseload_no45q'] = crossover_data.get('ccs_no45q_baseload', 0)
                entry['ldes_cost'] = crossover_data.get('ldes_cost', 0)
                crossovers.append(crossover_data.get('crossover_cf', 0))

            curve_impact.append(entry)

        analysis['crossover_by_iso'][iso] = {
            'avg_crossover_cf': round(sum(crossovers) / len(crossovers), 3) if crossovers else 0,
            'min_crossover_cf': round(min(crossovers), 3) if crossovers else 0,
            'max_crossover_cf': round(max(crossovers), 3) if crossovers else 0,
        }
        analysis['cost_curve_impact'][iso] = curve_impact

    # Print summary
    print("\n      CCS vs LDES Crossover (Medium scenario):")
    print(f"      {'ISO':>6}  {'Avg CF':>7}  {'Range':>15}  {'Interpretation':>30}")
    for iso in run_isos:
        cr = analysis['crossover_by_iso'].get(iso, {})
        avg_cf = cr.get('avg_crossover_cf', 0)
        min_cf = cr.get('min_crossover_cf', 0)
        max_cf = cr.get('max_crossover_cf', 0)
        if avg_cf > 0:
            interp = f"CCS viable above {avg_cf:.0%} CF"
        else:
            interp = "No CCS in mix"
        print(f"      {iso:>6}  {avg_cf:>6.1%}  [{min_cf:.0%}-{max_cf:.0%}]  {interp}")

    print("\n      Without-45Q Cost Impact (Medium, effective $/MWh):")
    print(f"      {'ISO':>6}  {'Thr':>5}  {'With 45Q':>9}  {'No 45Q':>9}  {'Delta':>7}  {'%':>6}  {'CCS%':>5}")
    for iso in run_isos:
        for entry in analysis['cost_curve_impact'].get(iso, []):
            if entry['threshold'] in [75, 90, 95, 99]:
                print(f"      {iso:>6}  {entry['threshold']:>4}%  "
                      f"${entry['cost_with_45q']:>7.2f}  ${entry['cost_without_45q']:>7.2f}  "
                      f"${entry['cost_delta']:>5.2f}  {entry['cost_increase_pct']:>5.1f}%  "
                      f"{entry['ccs_pct']:>4}%")

    # NEISO gas constraint impact
    if 'NEISO' in data['results'] and 'NEISO' in run_isos:
        print("\n      NEISO Gas Constraint Impact (Medium, effective $/MWh):")
        thresholds_data = data['results']['NEISO'].get('thresholds', {})
        neiso_impact = []
        for threshold in THRESHOLDS:
            t_str = str(threshold) if threshold != int(threshold) else str(int(threshold))
            neiso_mk = medium_key('NEISO')
            scenarios = thresholds_data.get(t_str, {}).get('scenarios', {})
            scenario = (scenarios.get(neiso_mk) or
                        scenarios.get('MMM_M_M_M1_X') or scenarios.get('MMM_M_M'))
            if not scenario:
                continue
            base = scenario.get('costs', {}).get('effective_cost', 0)
            gas_adj = scenario.get('neiso_gas_adjusted', {}).get('effective_cost', 0)
            gas_no45q = scenario.get('neiso_gas_no45q', {}).get('effective_cost', 0)
            ccs = scenario.get('resource_mix', {}).get('ccs_ccgt', 0)
            neiso_impact.append({
                'threshold': threshold, 'base': base, 'gas_adjusted': gas_adj,
                'gas_no45q': gas_no45q, 'ccs_pct': ccs,
            })

        print(f"      {'Thr':>5}  {'Base':>8}  {'+ Gas':>8}  {'+ Gas-No45Q':>12}  {'CCS%':>5}")
        for e in neiso_impact:
            if e['threshold'] in [75, 87.5, 90, 92.5, 95, 97.5, 99]:
                print(f"      {e['threshold']:>4}%  ${e['base']:>6.2f}  "
                      f"${e['gas_adjusted']:>6.2f}  ${e['gas_no45q']:>10.2f}  {e['ccs_pct']:>4}%")

    return analysis


def compute_gas_capacity_and_ra(data, run_isos):
    """
    For each scenario, compute gas capacity backup & resource adequacy.
    """
    print("\n  [4] Gas Capacity Backup & Resource Adequacy")

    total_computed = 0
    for iso in run_isos:
        if iso not in data.get('results', {}):
            continue

        iso_data = data['results'][iso]
        peak_mw = PEAK_DEMAND_MW.get(iso, 0)
        demand_mwh = iso_data.get('annual_demand_mwh', REGIONAL_DEMAND_MWH.get(iso, 0))
        existing_gas_mw = EXISTING_GAS_CAPACITY_MW[iso]
        wholesale = WHOLESALE_PRICES[iso]

        ra_peak_mw = peak_mw * (1 + RESOURCE_ADEQUACY_MARGIN)
        avg_demand_mw = demand_mwh / 8760

        thresholds_data = iso_data.get('thresholds', {})
        for t_str, t_data in thresholds_data.items():
            scenarios = t_data.get('scenarios', {})
            for sk, scenario in scenarios.items():
                mix = scenario.get('resource_mix', {})
                batt = scenario.get('battery_dispatch_pct', 0)
                batt8 = scenario.get('battery8_dispatch_pct', 0)
                ldes = scenario.get('ldes_dispatch_pct', 0)
                h2 = scenario.get('h2_dispatch_pct', 0)

                # Vectorized peak capacity: dot product of resource pcts × credits
                resource_pcts = np.array([mix.get(r, 0) for r in RESOURCE_TYPES])
                resource_credits = np.array([PEAK_CAPACITY_CREDITS.get(r, 0) for r in RESOURCE_TYPES])
                clean_peak_mw = float(np.dot(resource_pcts, resource_credits)) / 100.0 * avg_demand_mw

                storage_pcts = np.array([batt, batt8, ldes, h2])
                storage_credits = np.array([
                    PEAK_CAPACITY_CREDITS['battery'], PEAK_CAPACITY_CREDITS['battery8'],
                    PEAK_CAPACITY_CREDITS['ldes'], PEAK_CAPACITY_CREDITS['h2']])
                clean_peak_mw += float(np.dot(storage_pcts, storage_credits)) / 100.0 * avg_demand_mw

                gaf = GAS_AVAILABILITY_FACTOR[iso]
                gas_needed_mw = max(0, ra_peak_mw - clean_peak_mw) / gaf

                existing_gas_used_mw = min(gas_needed_mw, existing_gas_mw)
                new_gas_mw = max(0, gas_needed_mw - existing_gas_used_mw)

                existing_gas_cost_annual = existing_gas_used_mw * EXISTING_GAS_FOM_KW_YR[iso] * 1000
                new_gas_cost_annual = new_gas_mw * NEW_CCGT_COST_KW_YR[iso] * 1000
                gas_backup_cost_annual = existing_gas_cost_annual + new_gas_cost_annual
                gas_backup_cost_per_mwh = gas_backup_cost_annual / demand_mwh if demand_mwh > 0 else 0

                new_gas_cost_per_mwh = new_gas_cost_annual / demand_mwh if demand_mwh > 0 else 0
                existing_gas_cost_per_mwh = existing_gas_cost_annual / demand_mwh if demand_mwh > 0 else 0

                clean_cost = scenario.get('costs', {}).get('effective_cost', 0)
                total_system_cost = clean_cost + gas_backup_cost_per_mwh
                incremental_with_new_gas = clean_cost + new_gas_cost_per_mwh

                scenario['gas_backup'] = {
                    'peak_demand_mw': round(peak_mw),
                    'ra_peak_mw': round(ra_peak_mw),
                    'clean_peak_capacity_mw': round(clean_peak_mw),
                    'gas_backup_needed_mw': round(gas_needed_mw),
                    'existing_gas_used_mw': round(existing_gas_used_mw),
                    'new_gas_build_mw': round(new_gas_mw),
                    'existing_gas_cost_per_mwh': round(existing_gas_cost_per_mwh, 2),
                    'new_gas_cost_per_mwh': round(new_gas_cost_per_mwh, 2),
                    'gas_backup_cost_per_mwh': round(gas_backup_cost_per_mwh, 2),
                    'total_system_cost_per_mwh': round(total_system_cost, 2),
                    'incremental_with_new_gas': round(incremental_with_new_gas, 2),
                    'clean_coverage_pct': round(clean_peak_mw / ra_peak_mw * 100, 1) if ra_peak_mw > 0 else 0,
                    'resource_adequacy_margin': RESOURCE_ADEQUACY_MARGIN,
                    'gas_availability_factor': gaf,
                }

                if iso == 'NEISO':
                    pipeline_cap = NEISO_PIPELINE_GAS_CAPACITY_MW
                    pipeline_shortfall_mw = max(0, gas_needed_mw - pipeline_cap)
                    pipeline_expansion_cost_yr = pipeline_shortfall_mw * NEISO_PIPELINE_EXPANSION_COST_MW_YR
                    pipeline_expansion_per_mwh = pipeline_expansion_cost_yr / demand_mwh if demand_mwh > 0 else 0
                    scenario['gas_backup']['pipeline_constraint'] = {
                        'pipeline_deliverable_mw': pipeline_cap,
                        'pipeline_shortfall_mw': round(pipeline_shortfall_mw),
                        'pipeline_expansion_cost_per_mwh': round(pipeline_expansion_per_mwh, 2),
                        'pipeline_constrained': pipeline_shortfall_mw > 0,
                    }

                total_computed += 1

        # Log summary for Medium scenario at key thresholds
        mk = medium_key(iso)
        for t in ['75', '90', '95', '99']:
            scenarios_t = thresholds_data.get(t, {}).get('scenarios', {})
            sc = (scenarios_t.get(mk) or
                  scenarios_t.get('MMM_M_M_M1_M') or scenarios_t.get('MMM_M_M_M1_X') or
                  scenarios_t.get('MMM_M_M'))
            if sc and 'gas_backup' in sc:
                gb = sc['gas_backup']
                gaf_str = f"GAF={gb.get('gas_availability_factor', 1.0):.0%}"
                pipeline_str = ''
                if 'pipeline_constraint' in gb:
                    pc = gb['pipeline_constraint']
                    if pc['pipeline_constrained']:
                        pipeline_str = f" | PIPELINE SHORTFALL: {pc['pipeline_shortfall_mw']:,} MW (+${pc['pipeline_expansion_cost_per_mwh']:.2f}/MWh)"
                    else:
                        pipeline_str = f" | pipeline OK ({gb['gas_backup_needed_mw']:,}/{pc['pipeline_deliverable_mw']:,} MW)"
                print(f"      {iso} {t:>3}%: peak={gb['ra_peak_mw']:,} MW(+RA), "
                      f"clean={gb['clean_peak_capacity_mw']:,} MW, "
                      f"gas={gb['gas_backup_needed_mw']:,} MW "
                      f"(existing={gb['existing_gas_used_mw']:,}, new={gb['new_gas_build_mw']:,}), "
                      f"coverage={gb['clean_coverage_pct']}%, {gaf_str}{pipeline_str}")

    print(f"      Computed gas backup for {total_computed:,} scenarios")
    return total_computed


# ══════════════════════════════════════════════════════════════════════════════
# DEMAND GROWTH PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def load_dg_parquets(input_dir, iso):
    """Load all step3_dg_<ISO>_t*.parquet files for an ISO.

    Returns list of dicts (one per DG row) with columns:
        iso, threshold, scenario, year, growth_level, growth_factor,
        annual_demand_mwh, mix_*, hourly_match_score,
        battery_dispatch_pct, battery8_dispatch_pct, ldes_dispatch_pct, h2_dispatch_pct,
        tranche_*, ra_*, cost_*
    """
    import glob as glob_mod
    pattern = os.path.join(input_dir, f'step3_dg_{iso}_t*.parquet')
    files = sorted(glob_mod.glob(pattern))
    if not files:
        return []

    rows = []
    for fpath in files:
        table = pq.read_table(fpath)
        # Use pandas .to_dict('records') for fast batch conversion
        # instead of per-column .to_pylist() + row-by-row dict construction
        df = table.to_pandas()
        rows.extend(df.to_dict('records'))
    return rows


def process_dg_corrections(dg_rows, run_isos):
    """Apply Step 4 corrections to demand growth rows.

    For each DG row:
      - NEISO gas constraint (if NEISO)
      - No-45Q overlay (all ISOs)
      - Gas backup & resource adequacy (with scaled peak demand)

    Modifies rows in-place, adding correction columns.
    """
    if not dg_rows:
        return 0

    print(f"\n  [DG] Processing {len(dg_rows):,} demand growth rows")
    processed = 0

    for row in dg_rows:
        iso = row['iso']
        if iso not in run_isos:
            continue

        scenario_key = row['scenario']
        growth_factor = row.get('growth_factor', 1.0) or 1.0
        dg_demand_mwh = row.get('annual_demand_mwh') or REGIONAL_DEMAND_MWH.get(iso, 0)

        # Build resource_mix dict from flat columns
        resource_mix = {}
        for rtype in ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro']:
            resource_mix[rtype] = row.get(f'mix_{rtype}', 0) or 0

        battery_pct = row.get('battery_dispatch_pct', 0) or 0
        ldes_pct = row.get('ldes_dispatch_pct', 0) or 0
        h2_pct = row.get('h2_dispatch_pct', 0) or 0
        match_score = row.get('hourly_match_score', 0) or 0

        # Scale existing shares for demand growth
        existing_shares_scaled = {
            k: min(v / growth_factor, 100.0)
            for k, v in GRID_MIX_SHARES[iso].items()
        }

        # [1] NEISO gas constraint
        battery8_pct_dg = row.get('battery8_dispatch_pct', 0) or 0
        if iso == 'NEISO':
            neiso_costs = compute_costs_for_scenario(
                iso, resource_mix, battery_pct, ldes_pct,
                match_score, scenario_key, apply_45q=True, neiso_gas_adder=True,
                existing_shares_override=existing_shares_scaled, h2_pct=h2_pct,
                battery8_pct=battery8_pct_dg)
            neiso_no45q = compute_costs_for_scenario(
                iso, resource_mix, battery_pct, ldes_pct,
                match_score, scenario_key, apply_45q=False, neiso_gas_adder=True,
                existing_shares_override=existing_shares_scaled, h2_pct=h2_pct,
                battery8_pct=battery8_pct_dg)
            for ckey in ['total_cost', 'effective_cost', 'incremental', 'wholesale']:
                row[f'neiso_gas_adj_{ckey}'] = neiso_costs.get(ckey)
                row[f'neiso_gas_no45q_{ckey}'] = neiso_no45q.get(ckey)
            # Update base costs to gas-adjusted
            row['cost_total_cost'] = neiso_costs['total_cost']
            row['cost_effective_cost'] = neiso_costs['effective_cost']
            row['cost_incremental'] = neiso_costs['incremental']

        # [2] No-45Q overlay (non-NEISO already handled; NEISO above)
        if iso != 'NEISO':
            no45q_costs = compute_costs_for_scenario(
                iso, resource_mix, battery_pct, ldes_pct,
                match_score, scenario_key, apply_45q=False, neiso_gas_adder=False,
                existing_shares_override=existing_shares_scaled, h2_pct=h2_pct,
                battery8_pct=battery8_pct_dg)
        else:
            no45q_costs = neiso_no45q  # Already computed above
        for ckey in ['total_cost', 'effective_cost', 'incremental', 'wholesale']:
            row[f'no45q_{ckey}'] = no45q_costs.get(ckey)

        # [3] Gas backup & resource adequacy (scaled peak demand)
        peak_mw = PEAK_DEMAND_MW.get(iso, 0) * growth_factor
        ra_peak_mw = peak_mw * (1 + RESOURCE_ADEQUACY_MARGIN)
        avg_demand_mw = dg_demand_mwh / 8760

        # Vectorized peak capacity: dot product of resource pcts × credits
        resource_pcts = np.array([resource_mix.get(r, 0) for r in RESOURCE_TYPES])
        resource_credits = np.array([PEAK_CAPACITY_CREDITS.get(r, 0) for r in RESOURCE_TYPES])
        clean_peak_mw = float(np.dot(resource_pcts, resource_credits)) / 100.0 * avg_demand_mw

        battery8_pct = row.get('battery8_dispatch_pct', 0) or 0
        storage_pcts = np.array([battery_pct, battery8_pct, ldes_pct, h2_pct])
        storage_credits = np.array([
            PEAK_CAPACITY_CREDITS['battery'],
            PEAK_CAPACITY_CREDITS.get('battery8', PEAK_CAPACITY_CREDITS['battery']),
            PEAK_CAPACITY_CREDITS['ldes'], PEAK_CAPACITY_CREDITS['h2']])
        clean_peak_mw += float(np.dot(storage_pcts, storage_credits)) / 100.0 * avg_demand_mw

        existing_gas_mw = EXISTING_GAS_CAPACITY_MW[iso]
        gaf = GAS_AVAILABILITY_FACTOR[iso]
        gas_needed_mw = max(0, ra_peak_mw - clean_peak_mw) / gaf
        existing_gas_used_mw = min(gas_needed_mw, existing_gas_mw)
        new_gas_mw = max(0, gas_needed_mw - existing_gas_used_mw)

        existing_gas_cost_annual = existing_gas_used_mw * EXISTING_GAS_FOM_KW_YR[iso] * 1000
        new_gas_cost_annual = new_gas_mw * NEW_CCGT_COST_KW_YR[iso] * 1000
        gas_backup_cost_annual = existing_gas_cost_annual + new_gas_cost_annual
        gas_backup_per_mwh = gas_backup_cost_annual / dg_demand_mwh if dg_demand_mwh > 0 else 0

        clean_cost = row.get('cost_effective_cost', 0)
        row['ra_peak_demand_mw'] = round(peak_mw)
        row['ra_ra_peak_mw'] = round(ra_peak_mw)
        row['ra_clean_peak_capacity_mw'] = round(clean_peak_mw)
        row['ra_gas_backup_needed_mw'] = round(gas_needed_mw)
        row['ra_existing_gas_used_mw'] = round(existing_gas_used_mw)
        row['ra_new_gas_build_mw'] = round(new_gas_mw)
        row['ra_existing_gas_cost_per_mwh'] = round(
            existing_gas_cost_annual / dg_demand_mwh if dg_demand_mwh > 0 else 0, 2)
        row['ra_new_gas_cost_per_mwh'] = round(
            new_gas_cost_annual / dg_demand_mwh if dg_demand_mwh > 0 else 0, 2)
        row['ra_gas_backup_cost_per_mwh'] = round(gas_backup_per_mwh, 2)
        row['ra_total_system_cost_per_mwh'] = round(clean_cost + gas_backup_per_mwh, 2)
        row['ra_clean_coverage_pct'] = round(
            clean_peak_mw / ra_peak_mw * 100, 1) if ra_peak_mw > 0 else 0

        processed += 1

    print(f"      Processed {processed:,} DG rows with corrections")
    return processed


def save_dg_parquets(dg_rows, output_dir, isos):
    """Save corrected DG rows to per-ISO per-threshold parquets."""
    if not dg_rows:
        return

    os.makedirs(output_dir, exist_ok=True)

    # Group rows by (iso, threshold) using defaultdict
    groups = collections.defaultdict(list)
    for row in dg_rows:
        iso = row.get('iso')
        if iso not in isos:
            continue
        thr = row.get('threshold', 0)
        groups[(iso, thr)].append(row)

    for (iso, thr), rows in sorted(groups.items()):
        t_label = f"{thr:g}"
        out_path = os.path.join(output_dir, f'step4_dg_{iso}_t{t_label}.parquet')
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, out_path, compression='zstd')

    # Summary
    iso_counts = {}
    for row in dg_rows:
        iso = row.get('iso')
        if iso in isos:
            iso_counts[iso] = iso_counts.get(iso, 0) + 1
    for iso, count in sorted(iso_counts.items()):
        thr_count = len(set(r['threshold'] for r in dg_rows if r['iso'] == iso))
        print(f"  step4_dg_{iso}: {count:,} corrected DG rows across {thr_count} thresholds")


# ══════════════════════════════════════════════════════════════════════════════
# CLI & MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description='Step 4: Gas & CCS post-processing with per-ISO parquet I/O.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Process all available ISOs (default)
  %(prog)s --all                    # Explicit all
  %(prog)s --iso ERCOT              # Process single ISO
  %(prog)s --iso ERCOT --iso PJM    # Process multiple ISOs
  %(prog)s --iso NEISO              # Process NEISO (includes gas constraint)
        """,
    )
    parser.add_argument(
        '--iso',
        dest='isos',
        action='append',
        choices=ALL_ISOS,
        metavar='ISO',
        help=f'ISO to process (repeatable). Choices: {", ".join(ALL_ISOS)}. '
             f'Default: all available ISOs.',
    )
    parser.add_argument(
        '--all',
        dest='run_all',
        action='store_true',
        default=False,
        help='Process all ISOs (default if no --iso given).',
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default=DEFAULT_INPUT_DIR,
        help=f'Directory containing step3_co_<ISO>.parquet files (default: {DEFAULT_INPUT_DIR}).',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Directory for step4_<ISO>.parquet output (default: {DEFAULT_OUTPUT_DIR}).',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine which ISOs to process
    if args.isos and not args.run_all:
        run_isos = list(dict.fromkeys(args.isos))  # Dedupe, preserve order
    else:
        # Default: all ISOs that have parquet files in the input directory
        run_isos = []
        for iso in ALL_ISOS:
            pq_path = os.path.join(args.input_dir, f'step3_co_{iso}.parquet')
            if os.path.exists(pq_path):
                run_isos.append(iso)
        if not run_isos:
            print(f"ERROR: No step3_co_*.parquet files found in {args.input_dir}")
            print(f"  Available ISOs: {ALL_ISOS}")
            print(f"  Run Step 3 first to generate per-ISO parquets.")
            sys.exit(1)

    print("=" * 70)
    print("  STEP 4: GAS & CCS POST-PROCESSING")
    print("=" * 70)
    print(f"  ISOs:       {', '.join(run_isos)}")
    print(f"  Input dir:  {args.input_dir}")
    print(f"  Output dir: {args.output_dir}")

    start = time.time()

    # Load per-ISO parquets → nested dict
    data = load_from_parquets(args.input_dir, run_isos)

    isos_present = [iso for iso in run_isos if iso in data.get('results', {})]
    if not isos_present:
        print("  ERROR: No ISO data loaded. Check input files.")
        sys.exit(1)
    print(f"  ISOs loaded: {isos_present}")

    # Apply corrections in order
    # [1] NEISO gas constraint (only if NEISO is in run set)
    neiso_count = 0
    if 'NEISO' in isos_present:
        neiso_count = add_neiso_gas_constraint(data)

    # [2] Without-45Q overlay (all non-NEISO ISOs; NEISO handled in step 1)
    no45q_count = add_no45q_overlay(data, isos_present)

    # [3] Crossover analysis
    analysis = analyze_crossover(data, isos_present)

    # [4] Gas capacity backup & resource adequacy
    gas_count = compute_gas_capacity_and_ra(data, isos_present)

    # Add metadata
    data['postprocessing'] = {
        'applied': True,
        'isos_processed': isos_present,
        'corrections': {
            'neiso_gas_adjustments': neiso_count,
            'no45q_overlays': no45q_count,
            'gas_ra_computed': gas_count,
        },
        'parameters': {
            'neiso_ccs_gas_adder': NEISO_CCS_GAS_ADDER,
            'neiso_wholesale_adder': NEISO_WHOLESALE_ADDER,
            'resource_adequacy_margin': RESOURCE_ADEQUACY_MARGIN,
        },
    }

    # Save corrected per-ISO parquets (baseline)
    print(f"\n  Saving corrected baseline results to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    save_to_parquets(data, args.output_dir, isos_present)

    # ── Demand Growth processing ──
    # Load DG parquets from Step 3, apply same corrections, save to Step 4 output
    print(f"\n  Processing demand growth scenarios...")
    all_dg_rows = []
    for iso in isos_present:
        iso_dg = load_dg_parquets(args.input_dir, iso)
        if iso_dg:
            print(f"    Loaded {len(iso_dg):,} DG rows for {iso}")
            all_dg_rows.extend(iso_dg)

    if all_dg_rows:
        dg_count = process_dg_corrections(all_dg_rows, isos_present)
        save_dg_parquets(all_dg_rows, args.output_dir, isos_present)
        data['postprocessing']['corrections']['dg_rows_processed'] = dg_count
    else:
        print("    No DG parquets found — skipping demand growth processing")

    # Save analysis JSON
    analysis_path = os.path.join(args.output_dir, 'step4_analysis.json')
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"  Analysis → {analysis_path}")

    # Save metadata JSON
    meta_path = os.path.join(args.output_dir, 'step4_meta.json')
    meta = {k: v for k, v in data.items() if k != 'results'}
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata → {meta_path}")

    elapsed = time.time() - start
    print(f"\n{'=' * 70}")
    print(f"  STEP 4 COMPLETE in {elapsed:.1f}s")
    print(f"  Processed: {', '.join(isos_present)}")
    if all_dg_rows:
        print(f"  DG rows:   {len(all_dg_rows):,}")
    print(f"  Output:    {args.output_dir}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
