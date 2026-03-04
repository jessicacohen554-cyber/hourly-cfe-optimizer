#!/usr/bin/env python3
"""
Step 3: Cost Optimization — Vectorized Cross-Evaluation
========================================================
For each (region, threshold):
  1. Load physically feasible mixes from PFS post-EF (Step 2 output)
  2. For each sensitivity combo: vectorized evaluation of ALL mixes
  3. Select cheapest mix → that becomes the scenario result
  4. Extract archetypes (unique winning mixes) for demand growth sweep
  5. For each (year, growth): evaluate archetypes, select cheapest

Pipeline position: Step 3 of 4
  Step 1 — PFS Generator (step1_pfs_generator.py)
  Step 2 — Efficient Frontier extraction (step2_efficient_frontier.py)
  Step 3 — Cost optimization (this file)
  Step 4 — Post-processing (step4_postprocess.py)

Input:  data/step2-ef-parquets/step2_ef_{ISO}_t{T}.parquet  (from Step 2, per-ISO/threshold)
Output: data/step3-cost-opt-parquets/step3_co_<ISO>.parquet  (per-ISO cost optimization results)

Key format: RFS_FF_TX_CCSq45_GEO (e.g., MMM_M_M_M1_M for CAISO all-Medium)
  CAISO: 17,496 combos per threshold. Non-CAISO: 5,832 combos per threshold.

Dependencies
------------
Install all required packages before running:

    pip install numpy numba pyarrow pandas

  - numpy   : Vectorized array math for cost evaluation across millions of mixes.
  - numba   : JIT compilation via @njit decorators for 10-50× speedup over pure numpy.
              Falls back to numpy if missing, but runs MUCH slower — always install.
  - pyarrow : Reading/writing Parquet files (Step 2 input, Step 3 output).
  - pandas  : DataFrame construction for flattening results to Parquet output.

Or install from the project root:

    pip install -r requirements.txt

Verify Numba is working before launching a run:

    python3 -c "from numba import njit; print('Numba OK')"
"""

import gc
import json
import os
import time
import argparse
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pipeline_config import (
    ISOS, REGIONAL_DEMAND_TWH, GRID_MIX_SHARES,
    WHOLESALE_PRICES, FUEL_ADJUSTMENTS, LMH, LEVEL_NAME,
    LCOE_TABLES, TX_TABLES, get_tx, UPRATE_LCOE,
    NUCLEAR_NEWBUILD_LCOE, GEOTHERMAL_LCOE, GEO_CAP_TWH,
    CCS_LCOE_45Q_ON, CCS_LCOE_45Q_OFF,
    CCS_CAP_TWH, OFFSHORE_ISOS,
    NEISO_CCS_GAS_ADDER, NEISO_WHOLESALE_ADDER,
    FOAK_NUCLEAR_NEWBUILD, FOAK_CCS_45Q_ON, FOAK_CCS_45Q_OFF,
    FOAK_GEOTHERMAL, FOAK_OFFSHORE_WIND, FOAK_LDES, FOAK_H2,
    NOAK_BATTERY, NOAK_BATTERY8, NOAK_OFFSHORE_WIND,
    LEARNING_PARAMS, LEARNING_EXPONENT, learning_fraction, year_adjusted_cost,
    EXISTING_NUCLEAR_GW, UPRATE_CAP_TWH,
    DEMAND_GROWTH_RATES, DEMAND_GROWTH_YEARS, DEMAND_GROWTH_LEVELS,
    THRESHOLD_TARGET_YEARS, DG_UNIQUE_YEARS, _DG_YEAR_TO_THRESHOLDS,
    RESOURCE_ADEQUACY_MARGIN, PEAK_DEMAND_MW,
    EXISTING_GAS_CAPACITY_MW, GAS_AVAILABILITY_FACTOR,
    PEAK_CAPACITY_CREDITS, RESOURCE_CAPACITY_FACTORS,
    NEW_CCGT_COST_KW_YR, EXISTING_GAS_FOM_KW_YR,
    STORAGE_REVENUE_CREDITS, OUTPUT_THRESHOLDS,
)
from pathlib import Path
from itertools import chain, product
from functools import lru_cache

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    import warnings
    warnings.warn(
        "\n"
        "╔══════════════════════════════════════════════════════════════════╗\n"
        "║  WARNING: Numba is not installed — falling back to pure numpy. ║\n"
        "║  This will be 10-50× SLOWER. Install it:                       ║\n"
        "║                                                                 ║\n"
        "║    pip install numba                                            ║\n"
        "║                                                                 ║\n"
        "║  Or install all dependencies:                                   ║\n"
        "║    pip install -r requirements.txt                              ║\n"
        "╚══════════════════════════════════════════════════════════════════╝",
        RuntimeWarning,
        stacklevel=2,
    )

# ============================================================================
# STEP3-SPECIFIC CONSTANTS (cost tables imported from pipeline_config above)
# ============================================================================

RESOURCE_TYPES = ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro']



# Existing clean generation in absolute TWh (constant, does NOT change with demand growth)
EXISTING_CLEAN_TWH = {}
for _iso, _shares in GRID_MIX_SHARES.items():
    _dem = REGIONAL_DEMAND_TWH[_iso]
    EXISTING_CLEAN_TWH[_iso] = {k: round(v / 100.0 * _dem, 3) for k, v in _shares.items()}


def _compute_existing_clean_peak_mw(iso):
    """Compute 2025 existing fleet peak capacity contribution (MW).

    Converts energy-based shares to installed capacity using capacity factors,
    then applies capacity credits to get peak contribution.
    """
    avg_demand_mw = REGIONAL_DEMAND_TWH[iso] * 1e6 / 8760
    total = 0.0
    for resource in ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro']:
        share_pct = GRID_MIX_SHARES[iso].get(resource, 0)
        if share_pct == 0:
            continue
        avg_mw = share_pct / 100.0 * avg_demand_mw
        cf = RESOURCE_CAPACITY_FACTORS[resource][iso]
        cc = PEAK_CAPACITY_CREDITS[resource]
        installed = avg_mw / cf
        total += installed * cc
    return total


EXISTING_CLEAN_PEAK_MW = {iso: _compute_existing_clean_peak_mw(iso)
                          for iso in REGIONAL_DEMAND_TWH}

# 2025 baseline gas_raw for delta RA calibration
# At base year, total_gas must equal EXISTING_GAS_CAPACITY_MW (calibrated to reality)
GAS_RAW_2025 = {
    iso: max(0, (PEAK_DEMAND_MW[iso] * (1 + RESOURCE_ADEQUACY_MARGIN)
                 - EXISTING_CLEAN_PEAK_MW[iso]) / GAS_AVAILABILITY_FACTOR[iso])
    for iso in PEAK_DEMAND_MW
}




# ============================================================================
# VECTORIZED COST FUNCTION
# ============================================================================

def price_mix_batch(iso, arrays, sens, demand_twh, target_year=None, growth_rate=None,
                     uprate_cap_override=None, existing_override=None):
    """
    Vectorized pricing of N mixes under a single sensitivity combo.

    Args:
        iso: region string
        arrays: dict with numpy arrays (shape N) for each mix dimension:
            'clean_firm', 'solar', 'wind', 'hydro' (% of demand),
            'battery_dispatch_pct', 'battery8_dispatch_pct',
            'ldes_dispatch_pct', 'hourly_match_score'
        sens: dict with scalar sensitivity parameters:
            'ren', 'firm', 'batt', 'ldes_lvl', 'ccs', 'q45', 'fuel', 'tx', 'geo'
        demand_twh: base year demand (scalar)
        target_year, growth_rate: demand growth params (scalars, optional)
        uprate_cap_override: if not None, override UPRATE_CAP_TWH[iso] with this
            value. Set to 0 to disable uprate tranche.
        existing_override: if not None, dict overriding GRID_MIX_SHARES[iso].
            Set resources to 0 to force new-build pricing (greenfield).

    Returns:
        numpy array of total_cost (shape N), and effective_cost (shape N)
    """
    N = len(arrays['clean_firm'])
    if N == 0:
        return np.array([]), np.array([])

    # Demand growth
    years = max(0, (target_year or 2025) - 2025)
    gf = (1 + (growth_rate or 0)) ** years if years > 0 else 1.0
    demand = demand_twh * gf
    existing_scale = 1.0 / gf

    # Sensitivity lookups (scalar)
    ren_name = LEVEL_NAME[sens['ren']]
    batt_name = LEVEL_NAME[sens['batt']]
    ldes_name = LEVEL_NAME[sens['ldes_lvl']]
    fuel_name = LEVEL_NAME[sens['fuel']]
    tx_name = LEVEL_NAME[sens['tx']]
    firm_lev = sens['firm']
    ccs_lev = sens['ccs']
    q45 = sens['q45']
    geo_lev = sens.get('geo')

    wholesale = max(5, WHOLESALE_PRICES[iso] + FUEL_ADJUSTMENTS[iso][fuel_name])
    # NEISO winter gas pipeline constraint — wholesale adder
    if iso == 'NEISO':
        wholesale += NEISO_WHOLESALE_ADDER
    existing = existing_override if existing_override is not None else GRID_MIX_SHARES[iso]

    # Arrays are pre-converted to float64 in _table_to_arrays — no .astype() needed
    # Resource fractions are already % of demand (no procurement multiplier)
    match_frac = arrays['hourly_match_score'] / 100.0

    total_cost = np.zeros(N, dtype=np.float64)

    # CCS pct = 100 - (cf + sol + wnd + hyd + osw) -- implicit residual resource
    cf_pct = arrays['clean_firm']
    sol_pct = arrays['solar']
    wnd_pct = arrays['wind']
    hyd_pct = arrays['hydro']
    osw_pct = arrays.get('offshore_wind', np.zeros(N, dtype=np.float64))
    ccs_pct = 100.0 - (cf_pct + sol_pct + wnd_pct + hyd_pct + osw_pct)
    ccs_pct = np.maximum(ccs_pct, 0.0)

    # CCS regional cap: limit CCS deployment to geologically feasible TWh.
    # Excess CCS above cap → priced as nuclear new-build instead.
    # For NYISO/NEISO (cap=0), ALL implicit CCS → nuclear new-build.
    ccs_cap = CCS_CAP_TWH.get(iso, 9999.0)
    ccs_total_twh = ccs_pct / 100.0 * demand
    ccs_capped_twh = np.minimum(ccs_total_twh, ccs_cap)
    ccs_overflow_twh = np.maximum(0, ccs_total_twh - ccs_capped_twh)
    ccs_capped_pct = np.where(demand > 0, ccs_capped_twh / demand * 100.0, 0.0)

    bat_pct = arrays['battery_dispatch_pct']
    bat8_pct = arrays.get('battery8_dispatch_pct', np.zeros(N, dtype=np.float64))
    ldes_pct = arrays['ldes_dispatch_pct']
    h2_pct = arrays.get('h2_dispatch_pct', np.zeros(N, dtype=np.float64))

    # --- Solar (existing = $0, only new-build costs money) ---
    sol_existing = min(existing['solar'] * existing_scale, 100.0)
    sol_existing_pct = np.minimum(sol_pct, sol_existing)
    sol_new_pct = np.maximum(0, sol_pct - sol_existing)
    sol_lcoe = LCOE_TABLES['solar'][ren_name][iso]
    sol_tx = get_tx('solar', tx_name, iso)
    total_cost += sol_new_pct / 100.0 * (sol_lcoe + sol_tx)

    # --- Wind (existing = $0, only new-build costs money) ---
    wnd_existing = min(existing['wind'] * existing_scale, 100.0)
    wnd_existing_pct = np.minimum(wnd_pct, wnd_existing)
    wnd_new_pct = np.maximum(0, wnd_pct - wnd_existing)
    wnd_lcoe = LCOE_TABLES['wind'][ren_name][iso]
    wnd_tx = get_tx('wind', tx_name, iso)
    total_cost += wnd_new_pct / 100.0 * (wnd_lcoe + wnd_tx)

    # --- Offshore Wind (all new-build — no existing offshore wind fleet) ---
    if iso in OFFSHORE_ISOS:
        osw_lcoe = LCOE_TABLES['offshore_wind'][ren_name][iso]
        osw_tx = get_tx('offshore_wind', tx_name, iso)
        total_cost += osw_pct / 100.0 * (osw_lcoe + osw_tx)

    # --- Hydro (always existing, $0 cost — sunk fleet) ---
    # No cost added for hydro (existing fleet, $0 price)

    # --- CCS-CCGT (existing = $0, only new-build costs money; capped at regional TWh limit) ---
    ccs_existing = min(existing.get('ccs_ccgt', 0) * existing_scale, 100.0)
    ccs_existing_pct = np.minimum(ccs_capped_pct, ccs_existing)
    ccs_new_pct = np.maximum(0, ccs_capped_pct - ccs_existing)
    ccs_table = CCS_LCOE_45Q_ON if q45 == '1' else CCS_LCOE_45Q_OFF
    ccs_lcoe = ccs_table[ccs_lev][iso]
    # NEISO winter gas pipeline constraint — CCS fuel cost adder
    if iso == 'NEISO':
        ccs_lcoe += NEISO_CCS_GAS_ADDER
    ccs_tx = get_tx('ccs_ccgt', tx_name, iso)
    total_cost += ccs_new_pct / 100.0 * (ccs_lcoe + ccs_tx)

    # CCS overflow → nuclear new-build (CCS exceeding regional geologic cap)
    # This capacity is physically needed but can't be CCS, so it's nuclear.
    nuclear_overflow_price = NUCLEAR_NEWBUILD_LCOE[firm_lev][iso] + get_tx('clean_firm', tx_name, iso)
    total_cost += np.where(demand > 0, ccs_overflow_twh / demand * nuclear_overflow_price, 0.0)

    # --- Clean Firm (existing = $0, new = merit-order tranche pricing) ---
    cf_existing = min(existing['clean_firm'] * existing_scale, 100.0)
    cf_existing_pct = np.minimum(cf_pct, cf_existing)
    cf_new_pct = np.maximum(0, cf_pct - cf_existing)

    # New CF in TWh for tranche pricing
    new_cf_twh = cf_new_pct / 100.0 * demand

    # Transmission adders
    tx_cf = get_tx('clean_firm', tx_name, iso)
    tx_ccs_cf = get_tx('ccs_ccgt', tx_name, iso)

    # Tranche 1: Nuclear uprates (capped, no tx beyond grid connection)
    uprate_cap = UPRATE_CAP_TWH[iso] if uprate_cap_override is None else uprate_cap_override
    uprate_twh = np.minimum(new_cf_twh, uprate_cap)
    uprate_cost = uprate_twh * UPRATE_LCOE[firm_lev]
    remaining = np.maximum(0, new_cf_twh - uprate_twh)

    # Tranche 2: Geothermal (CAISO only, capped)
    geo_cost = np.zeros(N)
    if iso == 'CAISO' and geo_lev:
        geo_price = GEOTHERMAL_LCOE[geo_lev] + tx_cf
        geo_twh = np.minimum(remaining, GEO_CAP_TWH)
        geo_cost = geo_twh * geo_price
        remaining = np.maximum(0, remaining - geo_twh)

    # Tranche 3: Cheapest of nuclear new-build vs CCS (CCS capped at regional headroom)
    nuclear_price = NUCLEAR_NEWBUILD_LCOE[firm_lev][iso] + tx_cf
    ccs_tranche_price = ccs_table[ccs_lev][iso] + tx_ccs_cf
    if iso == 'NEISO':
        ccs_tranche_price += NEISO_CCS_GAS_ADDER

    # CCS headroom: cap minus what's already used by implicit CCS residual
    ccs_headroom_twh = np.maximum(0, ccs_cap - ccs_capped_twh)

    if nuclear_price <= ccs_tranche_price:
        # Nuclear cheaper → all remaining goes to nuclear
        nuclear_newbuild_twh = remaining
        ccs_tranche_twh = np.zeros(N)
        tranche3_cost = remaining * nuclear_price
    else:
        # CCS cheaper → CCS gets min(remaining, headroom), nuclear gets overflow
        ccs_tranche_twh = np.minimum(remaining, ccs_headroom_twh)
        nuclear_newbuild_twh = np.maximum(0, remaining - ccs_tranche_twh)
        tranche3_cost = (ccs_tranche_twh * ccs_tranche_price +
                         nuclear_newbuild_twh * nuclear_price)

    cf_total_new_cost = uprate_cost + geo_cost + tranche3_cost
    cf_cost_per_demand = cf_total_new_cost / demand
    total_cost += cf_cost_per_demand  # existing CF = $0

    # --- Storage (annualized capacity costs net of revenue credits) ---
    # coeff = bat_pct/100 is energy capacity as fraction of annual demand.
    # price = annualized fixed cost per % of annual demand, minus revenue credit
    # (capacity market + arbitrage + ancillary, capped so net ≥ 0).
    bat4_price = max(0, LCOE_TABLES['battery'][batt_name][iso] - STORAGE_REVENUE_CREDITS['battery'][iso])
    bat8_price = max(0, LCOE_TABLES['battery8'][batt_name][iso] - STORAGE_REVENUE_CREDITS['battery8'][iso])
    ldes_price = max(0, LCOE_TABLES['ldes'][ldes_name][iso] - STORAGE_REVENUE_CREDITS['ldes'][iso])
    h2_price = max(0, LCOE_TABLES['h2'][ldes_name][iso] - STORAGE_REVENUE_CREDITS['h2'][iso])
    total_cost += (bat_pct / 100.0 * bat4_price +
                   bat8_pct / 100.0 * bat8_price +
                   ldes_pct / 100.0 * ldes_price +
                   h2_pct / 100.0 * h2_price)

    # --- Gas Capacity Backup (delta RA approach) ---
    # Calibrated to 2025 reality: gas = EXISTING_GAS at base year.
    # Compute incremental gas from demand growth net of new clean peak capacity.
    demand_mwh = demand * 1e6  # TWh → MWh
    avg_demand_mw = demand_mwh / 8760
    gaf = GAS_AVAILABILITY_FACTOR[iso]

    # Peak scales with demand growth
    peak_grown = PEAK_DEMAND_MW[iso] * gf
    ra_peak_grown = peak_grown * (1 + RESOURCE_ADEQUACY_MARGIN)

    # New-build clean peak capacity: energy → installed MW → peak MW
    # Only new-build above existing floor contributes incremental peak
    new_clean_peak_mw = np.zeros(N, dtype=np.float64)
    # Offshore wind is all new-build (no existing fleet)
    _osw_new_pct = osw_pct if iso in OFFSHORE_ISOS else np.zeros(N, dtype=np.float64)
    for _res, _new_pct in [('clean_firm', cf_new_pct), ('solar', sol_new_pct),
                           ('wind', wnd_new_pct), ('offshore_wind', _osw_new_pct),
                           ('ccs_ccgt', ccs_new_pct)]:
        _cf_r = RESOURCE_CAPACITY_FACTORS[_res][iso]
        _cc_r = PEAK_CAPACITY_CREDITS[_res]
        _new_avg_mw = _new_pct / 100.0 * avg_demand_mw
        _new_installed = _new_avg_mw / _cf_r
        new_clean_peak_mw += _new_installed * _cc_r

    # Storage peak capacity: convert energy fraction (% of annual demand) → MW via duration.
    # energy_MWh = pct/100 * demand_mwh; power_MW = energy_MWh / duration_hr; peak = power * cc
    new_clean_peak_mw += (
        bat_pct / 100.0 * demand_mwh / 4.0 * PEAK_CAPACITY_CREDITS['battery'] +
        bat8_pct / 100.0 * demand_mwh / 8.0 * PEAK_CAPACITY_CREDITS['battery8'] +
        ldes_pct / 100.0 * demand_mwh / 100.0 * PEAK_CAPACITY_CREDITS['ldes'] +
        h2_pct / 100.0 * demand_mwh / 1000.0 * PEAK_CAPACITY_CREDITS['h2']
    )

    # Total system clean peak = existing (constant) + new-build
    total_clean_peak = EXISTING_CLEAN_PEAK_MW[iso] + new_clean_peak_mw

    # Gas raw for this scenario vs 2025 baseline
    gas_raw = np.maximum(0, ra_peak_grown - total_clean_peak) / gaf
    gas_delta = gas_raw - GAS_RAW_2025[iso]
    gas_needed_mw = np.maximum(0, EXISTING_GAS_CAPACITY_MW[iso] + gas_delta)

    existing_gas_mw = EXISTING_GAS_CAPACITY_MW[iso]
    existing_gas_used_mw = np.minimum(gas_needed_mw, existing_gas_mw)
    new_gas_mw = np.maximum(0, gas_needed_mw - existing_gas_mw)

    # Annualized capacity cost: existing gas FOM + new CCGT full cost ($/yr)
    gas_cost_per_mwh = (
        existing_gas_used_mw * EXISTING_GAS_FOM_KW_YR[iso] * 1000 +
        new_gas_mw * NEW_CCGT_COST_KW_YR[iso] * 1000
    ) / demand_mwh

    total_cost += gas_cost_per_mwh

    # Effective cost (total cost ÷ match fraction)
    effective_cost = np.where(match_frac > 0, total_cost / match_frac, 0)

    # Build tranche breakdown (per-mix arrays, shape N)
    ra_peak_mw = ra_peak_grown  # for output
    tranche_data = {
        'cf_existing_twh': cf_existing_pct / 100.0 * demand,
        'uprate_twh': uprate_twh,
        'geo_twh': geo_twh if (iso == 'CAISO' and geo_lev) else np.zeros(N),
        'nuclear_newbuild_twh': nuclear_newbuild_twh,
        'ccs_tranche_twh': ccs_tranche_twh,
        'new_cf_twh': new_cf_twh,
        # Gas backup fields (delta RA, post-GAF)
        'gas_backup_mw': gas_needed_mw,
        'existing_gas_used_mw': existing_gas_used_mw,
        'new_gas_build_mw': new_gas_mw,
        'gas_cost_per_mwh': gas_cost_per_mwh,
        'clean_peak_mw': total_clean_peak,
        'ra_peak_mw': np.full(N, ra_peak_mw),
    }

    return total_cost, effective_cost, tranche_data


# ============================================================================
# EXISTING CLEAN FLOOR FILTER
# ============================================================================
# For the baseline (Phase 1), existing clean assets must be part of the mix
# no matter what.  A PFS mix with solar < existing solar is infeasible under
# the mandate that you can't un-build existing clean generation.  This filter
# removes mixes where any resource is below the existing share.
#
# Tracks (newbuild, CTR) bypass this filter via their own existing_override.

def _apply_mask(arrays, mask, N):
    """Apply a boolean mask to all length-N numpy arrays in a dict.

    Non-array values and arrays with length != N are passed through unchanged.
    """
    filtered = {}
    for key, arr in arrays.items():
        if isinstance(arr, np.ndarray) and len(arr) == N:
            filtered[key] = arr[mask]
        else:
            filtered[key] = arr
    return filtered


def apply_existing_clean_floor(arrays, iso):
    """Filter PFS arrays to keep only mixes where each resource >= existing share.

    Args:
        arrays: dict of numpy arrays keyed by resource name + dispatch fields
        iso: region string

    Returns:
        filtered_arrays: same structure, with sub-existing mixes removed
        n_removed: count of mixes removed
    """
    existing = GRID_MIX_SHARES[iso]
    N = len(arrays['clean_firm'])

    mask = np.ones(N, dtype=bool)
    for resource in ['clean_firm', 'solar', 'wind', 'hydro']:
        floor_raw = existing.get(resource, 0)
        if floor_raw > 0:
            # PFS arrays use integer % (0-100); snap floor down to match grid
            floor = int(floor_raw)
            mask &= arrays[resource] >= floor

    n_removed = N - int(mask.sum())
    if n_removed == 0:
        return arrays, 0
    return _apply_mask(arrays, mask, N), n_removed


def apply_hydro_cap(arrays, iso):
    """Filter PFS arrays to keep only mixes where hydro <= existing hydro cap.

    The PFS (Step 1) explores hydro up to HYDRO_CAP + 10% adder for physics
    experimentation. Those extended-hydro mixes belong in the EF cache for
    Track 2 (newbuild analysis) but must NOT enter Track 1 baseline costing,
    where hydro is existing-only at $0 wholesale.

    Cap is ceil(GRID_MIX_SHARES[iso]['hydro']) since PFS uses integer %.

    Args:
        arrays: dict of numpy arrays keyed by resource name + dispatch fields
        iso: region string

    Returns:
        filtered_arrays: same structure, with above-cap hydro mixes removed
        n_removed: count of mixes removed
    """
    import math
    hydro_cap = math.ceil(GRID_MIX_SHARES[iso].get('hydro', 0))
    N = len(arrays['clean_firm'])
    mask = arrays['hydro'] <= hydro_cap
    n_removed = N - int(mask.sum())
    if n_removed == 0:
        return arrays, 0
    return _apply_mask(arrays, mask, N), n_removed


# ============================================================================
# PRE-COMPUTED COEFFICIENT MODEL (Phase 1 acceleration)
# ============================================================================
# For base year (no growth), total_cost decomposes into:
#   tc[i] = Σ(coeff_matrix[i,k] × prices[k]) + constant[i]
# where coefficients are scenario-invariant and prices are 10 scalars.
# This avoids recomputing existing/new splits, tranche allocations, and
# gas backup for every scenario — just 10 scalar-vector multiplies.

# Coefficient column indices
_COL_WHOLESALE = 0  # existing generation — $0 (sunk fleet, coefficient zeroed out)
_COL_SOL_NEW   = 1  # new solar (LCOE + tx)
_COL_WND_NEW   = 2  # new wind (LCOE + tx)
_COL_OSW_NEW   = 3  # new offshore wind (LCOE + tx) — offshore ISOs only, 0 elsewhere
_COL_CCS_NEW   = 4  # new CCS-CCGT standalone (LCOE + tx)
_COL_UPRATE    = 5  # nuclear uprate tranche
_COL_GEO       = 6  # geothermal tranche (CAISO only, 0 elsewhere)
_COL_REMAINING = 7  # tranche 3: min(nuclear, CCS) new-build
_COL_BAT4      = 8  # 4hr battery dispatch
_COL_BAT8      = 9  # 8hr battery dispatch
_COL_LDES      = 10 # LDES dispatch
_COL_H2        = 11 # Green H2 seasonal storage dispatch
_N_COEFFS = 12


def precompute_base_year_coefficients(iso, arrays, demand_twh, uprate_cap_override=None,
                                       existing_override=None):
    """Pre-compute scenario-invariant coefficient arrays for base year.

    Args:
        iso: region string
        arrays: dict with numpy arrays for each mix dimension
        demand_twh: base year demand (scalar)
        uprate_cap_override: if not None, override UPRATE_CAP_TWH[iso] with this
            value. Set to 0 to disable uprate tranche (Track 2: replace).
        existing_override: if not None, dict overriding GRID_MIX_SHARES[iso].
            Set individual resources to 0 to force new-build pricing.
            e.g., {'clean_firm': 0, 'solar': 0, 'wind': 0, 'ccs_ccgt': 0, 'hydro': 9.5}
            keeps hydro at existing share, zeros everything else.

    Returns:
        coeff_matrix: (N, 12) float64 — multiply by scenario prices
        constant: (N,) float64 — gas backup cost (scenario-invariant)
        extras: dict with per-element data needed for winner detail extraction
    """
    N = len(arrays['clean_firm'])

    # Arrays are pre-converted to float64 in _table_to_arrays — no .astype() needed
    # Resource fractions are already % of demand (no procurement multiplier)
    match_frac = arrays['hourly_match_score'] / 100.0

    cf_pct = arrays['clean_firm']
    sol_pct = arrays['solar']
    wnd_pct = arrays['wind']
    hyd_pct = arrays['hydro']
    osw_pct = arrays.get('offshore_wind', np.zeros(N, dtype=np.float64))
    ccs_pct = np.maximum(0.0, 100.0 - (cf_pct + sol_pct + wnd_pct + hyd_pct + osw_pct))

    bat_pct = arrays['battery_dispatch_pct']
    bat8_pct = arrays.get('battery8_dispatch_pct', np.zeros(N, dtype=np.float64))
    ldes_pct = arrays['ldes_dispatch_pct']
    h2_pct = arrays.get('h2_dispatch_pct', np.zeros(N, dtype=np.float64))

    existing = existing_override if existing_override is not None else GRID_MIX_SHARES[iso]

    # Existing/new splits (base year, existing_scale=1.0)
    sol_existing_pct = np.minimum(sol_pct, existing['solar'])
    sol_new_pct = np.maximum(0, sol_pct - existing['solar'])

    wnd_existing_pct = np.minimum(wnd_pct, existing['wind'])
    wnd_new_pct = np.maximum(0, wnd_pct - existing['wind'])

    # Offshore wind: all new-build (no existing fleet)
    osw_new_pct = osw_pct

    ccs_ex = existing.get('ccs_ccgt', 0)
    ccs_existing_pct = np.minimum(ccs_pct, ccs_ex)
    ccs_new_pct = np.maximum(0, ccs_pct - ccs_ex)

    cf_existing_pct = np.minimum(cf_pct, existing['clean_firm'])
    cf_new_pct = np.maximum(0, cf_pct - existing['clean_firm'])

    # Clean firm tranche allocation (scenario-invariant quantities)
    new_cf_twh = cf_new_pct / 100.0 * demand_twh
    uprate_cap = UPRATE_CAP_TWH[iso] if uprate_cap_override is None else uprate_cap_override
    uprate_twh = np.minimum(new_cf_twh, uprate_cap)
    remaining_after_uprate = np.maximum(0, new_cf_twh - uprate_twh)

    geo_twh = np.zeros(N)
    remaining_after_geo = remaining_after_uprate
    if iso == 'CAISO':
        geo_twh = np.minimum(remaining_after_uprate, GEO_CAP_TWH)
        remaining_after_geo = np.maximum(0, remaining_after_uprate - geo_twh)

    # --- Gas backup (delta RA approach, scenario-invariant) ---
    # Infer growth factor from demand: gf = demand_twh / base_demand
    _base_demand = REGIONAL_DEMAND_TWH[iso]
    _gf = demand_twh / _base_demand if _base_demand > 0 else 1.0
    demand_mwh = demand_twh * 1e6
    avg_demand_mw = demand_mwh / 8760
    gaf = GAS_AVAILABILITY_FACTOR[iso]

    # Peak scales with demand growth
    peak_grown = PEAK_DEMAND_MW[iso] * _gf
    ra_peak_grown = peak_grown * (1 + RESOURCE_ADEQUACY_MARGIN)

    # New-build clean peak capacity: energy → installed MW → peak MW
    new_clean_peak_mw = np.zeros(N, dtype=np.float64)
    # Offshore wind is all new-build (no existing fleet)
    _osw_new_pct = osw_pct if iso in OFFSHORE_ISOS else np.zeros(N, dtype=np.float64)
    for _res, _new_pct in [('clean_firm', cf_new_pct), ('solar', sol_new_pct),
                           ('wind', wnd_new_pct), ('offshore_wind', _osw_new_pct),
                           ('ccs_ccgt', ccs_new_pct)]:
        _cf_r = RESOURCE_CAPACITY_FACTORS[_res][iso]
        _cc_r = PEAK_CAPACITY_CREDITS[_res]
        _new_avg_mw = _new_pct / 100.0 * avg_demand_mw
        _new_installed = _new_avg_mw / _cf_r
        new_clean_peak_mw += _new_installed * _cc_r

    # Storage peak capacity: energy fraction (% annual demand) → MW via duration
    new_clean_peak_mw += (
        bat_pct / 100.0 * demand_mwh / 4.0 * PEAK_CAPACITY_CREDITS['battery'] +
        bat8_pct / 100.0 * demand_mwh / 8.0 * PEAK_CAPACITY_CREDITS['battery8'] +
        ldes_pct / 100.0 * demand_mwh / 100.0 * PEAK_CAPACITY_CREDITS['ldes'] +
        h2_pct / 100.0 * demand_mwh / 1000.0 * PEAK_CAPACITY_CREDITS['h2']
    )

    # Total system clean peak = existing (constant) + new-build
    total_clean_peak = EXISTING_CLEAN_PEAK_MW[iso] + new_clean_peak_mw

    # Gas raw for this scenario vs 2025 baseline (delta approach)
    gas_raw = np.maximum(0, ra_peak_grown - total_clean_peak) / gaf
    gas_delta = gas_raw - GAS_RAW_2025[iso]
    gas_needed_mw = np.maximum(0, EXISTING_GAS_CAPACITY_MW[iso] + gas_delta)

    existing_gas_mw = EXISTING_GAS_CAPACITY_MW[iso]
    existing_gas_used_mw = np.minimum(gas_needed_mw, existing_gas_mw)
    new_gas_mw = np.maximum(0, gas_needed_mw - existing_gas_mw)

    constant = (
        existing_gas_used_mw * EXISTING_GAS_FOM_KW_YR[iso] * 1000 +
        new_gas_mw * NEW_CCGT_COST_KW_YR[iso] * 1000
    ) / demand_mwh

    # Build coefficient matrix (N, 12)
    coeff_matrix = np.empty((N, _N_COEFFS), dtype=np.float64)
    # Existing clean resources = $0 (sunk fleet, no cost to buyer)
    coeff_matrix[:, _COL_WHOLESALE] = 0.0
    coeff_matrix[:, _COL_SOL_NEW] = sol_new_pct / 100.0
    coeff_matrix[:, _COL_WND_NEW] = wnd_new_pct / 100.0
    coeff_matrix[:, _COL_OSW_NEW] = osw_new_pct / 100.0  # all new-build, no existing
    coeff_matrix[:, _COL_CCS_NEW] = ccs_new_pct / 100.0
    coeff_matrix[:, _COL_UPRATE] = uprate_twh / demand_twh
    coeff_matrix[:, _COL_GEO] = geo_twh / demand_twh
    coeff_matrix[:, _COL_REMAINING] = remaining_after_geo / demand_twh
    coeff_matrix[:, _COL_BAT4] = bat_pct / 100.0
    coeff_matrix[:, _COL_BAT8] = bat8_pct / 100.0
    coeff_matrix[:, _COL_LDES] = ldes_pct / 100.0
    coeff_matrix[:, _COL_H2] = h2_pct / 100.0

    ra_peak_mw = ra_peak_grown  # for output

    extras = {
        'match_frac': match_frac,
        'cf_existing_twh': cf_existing_pct / 100.0 * demand_twh,
        'new_cf_twh': new_cf_twh,
        'uprate_twh': uprate_twh,
        'geo_twh': geo_twh,
        'remaining_twh': remaining_after_geo,
        'gas_needed_mw': gas_needed_mw,
        'existing_gas_used_mw': existing_gas_used_mw,
        'new_gas_mw': new_gas_mw,
        'clean_peak_mw': total_clean_peak,
        'ra_peak_mw': ra_peak_mw,
        # Constant existing TWh available (does NOT depend on mix or demand growth)
        'existing_cf_twh_available': EXISTING_CLEAN_TWH[iso]['clean_firm'],
        'existing_solar_twh_available': EXISTING_CLEAN_TWH[iso]['solar'],
        'existing_wind_twh_available': EXISTING_CLEAN_TWH[iso]['wind'],
        'existing_hydro_twh_available': EXISTING_CLEAN_TWH[iso]['hydro'],
    }

    return coeff_matrix, constant, extras


def get_scenario_prices(iso, sens):
    """Look up 11 scenario-dependent price scalars from sensitivity toggles.

    Returns:
        prices: (11,) float64 array matching coefficient column order
        wholesale: scalar for incremental cost calculation
        nuclear_price: scalar for tranche detail extraction
        ccs_tranche_price: scalar for tranche detail extraction
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

    wholesale = max(5, WHOLESALE_PRICES[iso] + FUEL_ADJUSTMENTS[iso][fuel_name])
    ccs_table = CCS_LCOE_45Q_ON if q45 == '1' else CCS_LCOE_45Q_OFF
    ccs_lcoe = ccs_table[ccs_lev][iso]
    ccs_tx = get_tx('ccs_ccgt', tx_name, iso)
    ccs_price = ccs_lcoe + ccs_tx

    nuclear_price = NUCLEAR_NEWBUILD_LCOE[firm_lev][iso] + get_tx('clean_firm', tx_name, iso)
    remaining_price = min(nuclear_price, ccs_price)

    geo_price = 0.0
    if iso == 'CAISO' and geo_lev:
        geo_price = GEOTHERMAL_LCOE[geo_lev] + get_tx('clean_firm', tx_name, iso)

    osw_price = 0.0
    if iso in OFFSHORE_ISOS:
        osw_price = LCOE_TABLES['offshore_wind'][ren_name][iso] + get_tx('offshore_wind', tx_name, iso)

    prices = np.array([
        wholesale,
        LCOE_TABLES['solar'][ren_name][iso] + get_tx('solar', tx_name, iso),
        LCOE_TABLES['wind'][ren_name][iso] + get_tx('wind', tx_name, iso),
        osw_price,
        ccs_price,
        UPRATE_LCOE[firm_lev],
        geo_price,
        remaining_price,
        max(0, LCOE_TABLES['battery'][batt_name][iso] + get_tx('battery', tx_name, iso) - STORAGE_REVENUE_CREDITS['battery'][iso]),
        max(0, LCOE_TABLES['battery8'][batt_name][iso] + get_tx('battery8', tx_name, iso) - STORAGE_REVENUE_CREDITS['battery8'][iso]),
        max(0, LCOE_TABLES['ldes'][ldes_name][iso] + get_tx('ldes', tx_name, iso) - STORAGE_REVENUE_CREDITS['ldes'][iso]),
        max(0, LCOE_TABLES['h2'][ldes_name][iso] + get_tx('h2', tx_name, iso) - STORAGE_REVENUE_CREDITS['h2'][iso]),
    ], dtype=np.float64)

    return prices, wholesale, nuclear_price, ccs_price


# Numba-accelerated cost evaluation
if HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _eval_cost_numba(coeff_matrix, constant, prices):
        """Fused multiply-add: tc[i] = Σ(coeff[i,k] × prices[k]) + constant[i]"""
        N = coeff_matrix.shape[0]
        K = coeff_matrix.shape[1]
        tc = np.empty(N, dtype=np.float64)
        for i in prange(N):
            s = constant[i]
            for k in range(K):
                s += coeff_matrix[i, k] * prices[k]
            tc[i] = s
        return tc

    @njit(cache=True)
    def _argmin_indexed(tc, idx):
        """Argmin over a subset of tc without allocating a temporary array."""
        best_val = tc[idx[0]]
        best_pos = 0
        for i in range(1, len(idx)):
            v = tc[idx[i]]
            if v < best_val:
                best_val = v
                best_pos = i
        return idx[best_pos], best_val

    @njit(cache=True)
    def _argmin_bucketed(tc, scores, thresholds_desc):
        """O(N) multi-threshold argmin via bucket accumulation.

        Assigns each element to its highest qualifying threshold bucket
        (single pass), then takes cumulative min across buckets.
        Total work: N × ~7 comparisons + N × 1 comparison = O(N).
        vs naive 13 × argmin = O(13N).

        Thresholds must be sorted descending (100, 99, 97.5, ..., 50).
        """
        N = len(tc)
        n_thr = len(thresholds_desc)
        bucket_mins = np.full(n_thr, np.inf)
        bucket_min_idxs = np.zeros(n_thr, dtype=np.int64)

        # Pass 1: assign each element to its highest qualifying bucket
        for i in range(N):
            s = scores[i]
            t = tc[i]
            for k in range(n_thr):
                if s >= thresholds_desc[k]:
                    if t < bucket_mins[k]:
                        bucket_mins[k] = t
                        bucket_min_idxs[k] = i
                    break

        # Pass 2: cumulative min from highest to lowest threshold
        best_vals = np.full(n_thr, np.inf)
        best_idxs = np.zeros(n_thr, dtype=np.int64)
        running_min = np.inf
        running_idx = np.int64(0)
        for k in range(n_thr):
            if bucket_mins[k] < running_min:
                running_min = bucket_mins[k]
                running_idx = bucket_min_idxs[k]
            best_vals[k] = running_min
            best_idxs[k] = running_idx

        return best_idxs, best_vals


def eval_cost_fast(coeff_matrix, constant, prices):
    """Evaluate total cost using Numba if available, else numpy."""
    if HAS_NUMBA:
        return _eval_cost_numba(coeff_matrix, constant, prices)
    return coeff_matrix @ prices + constant


def argmin_indexed(tc, idx):
    """Find argmin of tc[idx] without allocating a copy."""
    if HAS_NUMBA:
        return _argmin_indexed(tc, idx)
    local_best = int(np.argmin(tc[idx]))
    return int(idx[local_best]), float(tc[idx[local_best]])


def eval_and_argmin_all(coeff_matrix, constant, prices, scores, thresholds_desc):
    """Parallel cost eval + bucketed multi-threshold argmin."""
    tc = eval_cost_fast(coeff_matrix, constant, prices)
    # Use effective gates (100% → 99.5%) for score comparison
    gates = np.where(thresholds_desc == 100, 99.5, thresholds_desc)
    if HAS_NUMBA:
        return _argmin_bucketed(tc, scores, gates)
    # Numpy fallback — pre-compute qualifying indices via sort+searchsorted
    n_thr = len(gates)
    sorted_order = np.argsort(scores)
    sorted_scores = scores[sorted_order]
    best_idxs = np.zeros(n_thr, dtype=np.int64)
    best_vals = np.full(n_thr, np.inf)
    for k in range(n_thr):
        insert_pos = np.searchsorted(sorted_scores, gates[k], side='left')
        if insert_pos < len(scores):
            qualifying = sorted_order[insert_pos:]
            local_best = int(np.argmin(tc[qualifying]))
            best_idxs[k] = qualifying[local_best]
            best_vals[k] = tc[qualifying[local_best]]
    return best_idxs, best_vals


def precompute_all_prices(iso, all_combos, target_year=None):
    """Pre-compute price vectors + metadata for all sensitivity combos at once.

    Inlines price lookups and fills the pre-allocated matrix directly, avoiding
    ~5,832-17,496 temporary np.array allocations from get_scenario_prices().

    Args:
        iso: region string
        all_combos: list of (scenario_key, sens_dict) tuples
        target_year: optional int — if provided, applies Wright's Law FOAK→NOAK
            learning curves to new-build clean firm technologies (nuclear, CCS,
            geothermal, LDES, H2). Used in Phase 2 (demand growth sweep) where
            each threshold maps to a future SBTi year. If None (Phase 1),
            uses static L/M/H costs from the LCOE tables.

    Returns:
        price_matrix: (B, 12) float64 price vectors
        wholesale_arr: (B,) float64 wholesale prices
        nuclear_arr: (B,) float64 nuclear new-build prices
        ccs_arr: (B,) float64 CCS tranche prices
    """
    B = len(all_combos)
    price_matrix = np.empty((B, _N_COEFFS), dtype=np.float64)
    wholesale_arr = np.empty(B, dtype=np.float64)
    nuclear_arr = np.empty(B, dtype=np.float64)
    ccs_arr = np.empty(B, dtype=np.float64)

    # Pre-resolve ISO-specific lookup tables to avoid repeated dict lookups
    _ws_base = WHOLESALE_PRICES[iso]
    _fuel_adj = FUEL_ADJUSTMENTS[iso]
    _ccs_on_iso = {lev: CCS_LCOE_45Q_ON[lev][iso] for lev in LMH}
    _ccs_off_iso = {lev: CCS_LCOE_45Q_OFF[lev][iso] for lev in LMH}
    _nuc_nb_iso = {lev: NUCLEAR_NEWBUILD_LCOE[lev][iso] for lev in LMH}
    _sol_lcoe_iso = {name: LCOE_TABLES['solar'][name][iso] for name in ['Low', 'Medium', 'High']}
    _wnd_lcoe_iso = {name: LCOE_TABLES['wind'][name][iso] for name in ['Low', 'Medium', 'High']}
    _bat_lcoe_iso = {name: LCOE_TABLES['battery'][name][iso] for name in ['Low', 'Medium', 'High']}
    _bat8_lcoe_iso = {name: LCOE_TABLES['battery8'][name][iso] for name in ['Low', 'Medium', 'High']}
    _ldes_lcoe_iso = {name: LCOE_TABLES['ldes'][name][iso] for name in ['Low', 'Medium', 'High']}
    _h2_lcoe_iso = {name: LCOE_TABLES['h2'][name][iso] for name in ['Low', 'Medium', 'High']}
    _osw_lcoe_iso = {name: LCOE_TABLES['offshore_wind'][name][iso] for name in ['Low', 'Medium', 'High']}
    _is_caiso = (iso == 'CAISO')
    _is_offshore = (iso in OFFSHORE_ISOS)

    # Pre-resolve transmission adders for all (resource, tx_level) combos for this ISO
    _tx_cache = {}
    for rtype in ['solar', 'wind', 'offshore_wind', 'clean_firm', 'ccs_ccgt', 'battery', 'battery8', 'ldes', 'h2']:
        for tx_name in ['None', 'Low', 'Medium', 'High']:
            _tx_cache[(rtype, tx_name)] = get_tx(rtype, tx_name, iso)

    # Pre-compute year-adjusted FOAK→NOAK costs if target_year provided
    # Each (technology, toggle_level) gets a pre-resolved cost for this year
    _use_learning = target_year is not None
    if _use_learning:
        _foak_nuc = FOAK_NUCLEAR_NEWBUILD[iso]
        _foak_ccs_on = FOAK_CCS_45Q_ON[iso]
        _foak_ccs_off = FOAK_CCS_45Q_OFF[iso]
        _foak_geo = FOAK_GEOTHERMAL  # scalar, CAISO only
        _foak_ldes = FOAK_LDES[iso]
        _foak_h2 = FOAK_H2[iso]
        # Battery FOAK tables kept for reference but not used for learning curves.
        # Battery learning goes LCOE_TABLES (2025) → NOAK_BATTERY (terminal floor).

        # Pre-compute year-adjusted nuclear new-build per firm level
        _nuc_yr = {}
        for lev in LMH:
            foak_s, noak_y = LEARNING_PARAMS['nuclear'][lev]
            _nuc_yr[lev] = year_adjusted_cost(_foak_nuc, _nuc_nb_iso[lev], target_year, foak_s, noak_y)

        # Pre-compute year-adjusted CCS per CCS level × 45Q
        _ccs_on_yr = {}
        _ccs_off_yr = {}
        for lev in LMH:
            foak_s, noak_y = LEARNING_PARAMS['ccs'][lev]
            _ccs_on_yr[lev] = year_adjusted_cost(_foak_ccs_on, _ccs_on_iso[lev], target_year, foak_s, noak_y)
            _ccs_off_yr[lev] = year_adjusted_cost(_foak_ccs_off, _ccs_off_iso[lev], target_year, foak_s, noak_y)

        # Pre-compute year-adjusted geothermal per geo level
        _geo_yr = {}
        if _is_caiso:
            for lev in LMH:
                foak_s, noak_y = LEARNING_PARAMS['geo'][lev]
                _geo_yr[lev] = year_adjusted_cost(_foak_geo, GEOTHERMAL_LCOE[lev], target_year, foak_s, noak_y)

        # Pre-compute year-adjusted LDES and H2 per LDES level
        _ldes_yr = {}
        _h2_yr = {}
        for name, lev in [('Low', 'L'), ('Medium', 'M'), ('High', 'H')]:
            foak_s, noak_y = LEARNING_PARAMS['ldes'][lev]
            _ldes_yr[name] = year_adjusted_cost(_foak_ldes, _ldes_lcoe_iso[name], target_year, foak_s, noak_y)
            foak_s_h2, noak_y_h2 = LEARNING_PARAMS['h2'][lev]
            _h2_yr[name] = year_adjusted_cost(_foak_h2, _h2_lcoe_iso[name], target_year, foak_s_h2, noak_y_h2)

        # Pre-compute year-adjusted offshore wind per ren level
        _osw_yr = {}
        if _is_offshore:
            _foak_osw = FOAK_OFFSHORE_WIND.get(iso, 0)
            _noak_osw_iso = {name: NOAK_OFFSHORE_WIND[name].get(iso, 0) for name in ['Low', 'Medium', 'High']}
            _lp_key = 'offshore_wind_float' if _is_caiso else 'offshore_wind_fixed'
            for name, lev in [('Low', 'L'), ('Medium', 'M'), ('High', 'H')]:
                foak_s, noak_y = LEARNING_PARAMS[_lp_key][lev]
                _osw_yr[name] = year_adjusted_cost(_foak_osw, _noak_osw_iso[name], target_year, foak_s, noak_y)

        # Pre-compute year-adjusted battery (4hr, 8hr) per batt level.
        # Battery Wright's Law: DECLINE from 2025 starting costs (LCOE_TABLES)
        # toward NOAK terminal floor. Direction is REVERSED vs other technologies
        # because batteries are already at scale — no FOAK premium, only future decline.
        _bat4_yr = {}
        _bat8_yr = {}
        _noak_bat4_iso = {name: NOAK_BATTERY[name][iso] for name in ['Low', 'Medium', 'High']}
        _noak_bat8_iso = {name: NOAK_BATTERY8[name][iso] for name in ['Low', 'Medium', 'High']}
        for name, lev in [('Low', 'L'), ('Medium', 'M'), ('High', 'H')]:
            foak_s_4, noak_y_4 = LEARNING_PARAMS['bat4'][lev]
            # start_cost=LCOE_TABLES (2025), terminal=NOAK (Wright's Law floor)
            _bat4_yr[name] = year_adjusted_cost(_bat_lcoe_iso[name], _noak_bat4_iso[name], target_year, foak_s_4, noak_y_4)
            foak_s_8, noak_y_8 = LEARNING_PARAMS['bat8'][lev]
            _bat8_yr[name] = year_adjusted_cost(_bat8_lcoe_iso[name], _noak_bat8_iso[name], target_year, foak_s_8, noak_y_8)

    for j, (scenario_key, sens) in enumerate(all_combos):
        ren_name = LEVEL_NAME[sens['ren']]
        batt_name = LEVEL_NAME[sens['batt']]
        ldes_name = LEVEL_NAME[sens['ldes_lvl']]
        fuel_name = LEVEL_NAME[sens['fuel']]
        tx_name = LEVEL_NAME[sens['tx']]
        firm_lev = sens['firm']
        ccs_lev = sens['ccs']
        q45 = sens['q45']
        geo_lev = sens.get('geo')

        wholesale = max(5, _ws_base + _fuel_adj[fuel_name])

        # CCS price: year-adjusted if learning curves active, static otherwise
        if _use_learning:
            ccs_lcoe = _ccs_on_yr[ccs_lev] if q45 == '1' else _ccs_off_yr[ccs_lev]
        else:
            ccs_lcoe = _ccs_on_iso[ccs_lev] if q45 == '1' else _ccs_off_iso[ccs_lev]
        ccs_tx = _tx_cache[('ccs_ccgt', tx_name)]
        ccs_price = ccs_lcoe + ccs_tx

        # Nuclear new-build price: year-adjusted if learning curves active
        tx_cf = _tx_cache[('clean_firm', tx_name)]
        if _use_learning:
            nuclear_price = _nuc_yr[firm_lev] + tx_cf
        else:
            nuclear_price = _nuc_nb_iso[firm_lev] + tx_cf
        remaining_price = min(nuclear_price, ccs_price)

        # Geothermal price: year-adjusted if learning curves active
        geo_price = 0.0
        if _is_caiso and geo_lev:
            if _use_learning:
                geo_price = _geo_yr[geo_lev] + tx_cf
            else:
                geo_price = GEOTHERMAL_LCOE[geo_lev] + tx_cf

        # LDES and H2 prices: year-adjusted if learning curves active
        if _use_learning:
            ldes_price = _ldes_yr[ldes_name] + _tx_cache[('ldes', tx_name)]
            h2_price = _h2_yr[ldes_name] + _tx_cache[('h2', tx_name)]
        else:
            ldes_price = _ldes_lcoe_iso[ldes_name] + _tx_cache[('ldes', tx_name)]
            h2_price = _h2_lcoe_iso[ldes_name] + _tx_cache[('h2', tx_name)]

        # Offshore wind price: year-adjusted if learning curves active
        if _is_offshore:
            if _use_learning:
                osw_price = _osw_yr[ren_name] + _tx_cache[('offshore_wind', tx_name)]
            else:
                osw_price = _osw_lcoe_iso[ren_name] + _tx_cache[('offshore_wind', tx_name)]
        else:
            osw_price = 0.0

        # Fill directly into pre-allocated matrix (avoids np.array() allocation per call)
        price_matrix[j, _COL_WHOLESALE] = wholesale
        price_matrix[j, _COL_SOL_NEW] = _sol_lcoe_iso[ren_name] + _tx_cache[('solar', tx_name)]
        price_matrix[j, _COL_WND_NEW] = _wnd_lcoe_iso[ren_name] + _tx_cache[('wind', tx_name)]
        price_matrix[j, _COL_OSW_NEW] = osw_price
        price_matrix[j, _COL_CCS_NEW] = ccs_price
        price_matrix[j, _COL_UPRATE] = UPRATE_LCOE[firm_lev]
        price_matrix[j, _COL_GEO] = geo_price
        price_matrix[j, _COL_REMAINING] = remaining_price
        # Battery prices: year-adjusted if learning curves active, else static.
        # TX is always 0 for storage (baked into regional capacity costs).
        # Revenue credits subtracted (capacity market + arbitrage + ancillary), floored at 0.
        _rc_bat4 = STORAGE_REVENUE_CREDITS['battery'][iso]
        _rc_bat8 = STORAGE_REVENUE_CREDITS['battery8'][iso]
        _rc_ldes = STORAGE_REVENUE_CREDITS['ldes'][iso]
        _rc_h2 = STORAGE_REVENUE_CREDITS['h2'][iso]
        if _use_learning:
            price_matrix[j, _COL_BAT4] = max(0, _bat4_yr[batt_name] - _rc_bat4)
            price_matrix[j, _COL_BAT8] = max(0, _bat8_yr[batt_name] - _rc_bat8)
        else:
            price_matrix[j, _COL_BAT4] = max(0, _bat_lcoe_iso[batt_name] - _rc_bat4)
            price_matrix[j, _COL_BAT8] = max(0, _bat8_lcoe_iso[batt_name] - _rc_bat8)
        price_matrix[j, _COL_LDES] = max(0, ldes_price - _rc_ldes)
        price_matrix[j, _COL_H2] = max(0, h2_price - _rc_h2)
        wholesale_arr[j] = wholesale
        nuclear_arr[j] = nuclear_price
        ccs_arr[j] = ccs_price

    return price_matrix, wholesale_arr, nuclear_arr, ccs_arr


# Numba-accelerated batched evaluation (tiled for cache efficiency)
if HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _batch_eval_and_argmin(coeff_matrix, constant, price_matrix, scores, thresholds_desc):
        """Fused batched cost eval + bucketed argmin, tiled for cache efficiency.

        Tiles over N (mixes) so each tile fits in L3 cache, then
        parallelizes over B (combos) with prange.  Each tile is loaded
        from main memory once and reused across all B combos — up to
        ~5000× reduction in memory bandwidth vs per-combo evaluation.

        Args:
            coeff_matrix: (N, K) coefficient matrix
            constant: (N,) constant terms
            price_matrix: (B, K) price vectors for all combos
            scores: (N,) hourly match scores
            thresholds_desc: (T,) thresholds sorted descending

        Returns:
            all_best_idxs: (B, T) int64
            all_best_vals: (B, T) float64
        """
        N = coeff_matrix.shape[0]
        K = coeff_matrix.shape[1]
        B = price_matrix.shape[0]
        n_thr = len(thresholds_desc)
        TILE = 200000  # ~16 MB per tile at K=10, fits in L3

        # Per-combo bucket accumulators (persist across tiles)
        bucket_mins = np.full((B, n_thr), np.inf)
        bucket_min_idxs = np.zeros((B, n_thr), dtype=np.int64)

        # Tile over N for cache efficiency
        for t_start in range(0, N, TILE):
            t_end = min(t_start + TILE, N)

            # All B combos process this tile in parallel
            for j in prange(B):
                for i in range(t_start, t_end):
                    # Fused multiply-add: tc = dot(coeff[i], prices[j]) + constant[i]
                    s = constant[i]
                    for k in range(K):
                        s += coeff_matrix[i, k] * price_matrix[j, k]

                    # Bucketed argmin: assign to highest qualifying threshold
                    score_i = scores[i]
                    for kt in range(n_thr):
                        if score_i >= thresholds_desc[kt]:
                            if s < bucket_mins[j, kt]:
                                bucket_mins[j, kt] = s
                                bucket_min_idxs[j, kt] = i
                            break

        # Cumulative min pass (parallel over combos)
        all_best_idxs = np.zeros((B, n_thr), dtype=np.int64)
        all_best_vals = np.full((B, n_thr), np.inf)
        for j in prange(B):
            running_min = np.inf
            running_idx = np.int64(0)
            for kt in range(n_thr):
                if bucket_mins[j, kt] < running_min:
                    running_min = bucket_mins[j, kt]
                    running_idx = bucket_min_idxs[j, kt]
                all_best_vals[j, kt] = running_min
                all_best_idxs[j, kt] = running_idx

        return all_best_idxs, all_best_vals


def batch_eval_and_argmin_all(coeff_matrix, constant, price_matrix, scores, thresholds_desc):
    """Batched cost eval + multi-threshold argmin for all combos at once.

    Args:
        coeff_matrix: (N, K) coefficient matrix
        constant: (N,) constant terms
        price_matrix: (B, K) price vectors for B sensitivity combos
        scores: (N,) hourly match scores
        thresholds_desc: (T,) thresholds sorted descending

    Returns:
        all_best_idxs: (B, T) int64 — best mix index per combo × threshold
        all_best_vals: (B, T) float64 — best cost per combo × threshold
    """
    if HAS_NUMBA:
        gates = np.where(thresholds_desc == 100, 99.5, thresholds_desc)
        return _batch_eval_and_argmin(coeff_matrix, constant, price_matrix, scores, gates)
    # Numpy fallback: sequential per-combo
    # Pre-compute qualifying indices per threshold ONCE (O(T×N) → avoids O(B×T×N))
    B = price_matrix.shape[0]
    gates = np.where(thresholds_desc == 100, 99.5, thresholds_desc)
    n_thr = len(gates)

    # Sort+searchsorted: O(N log N + T log N) instead of O(T×N)
    sorted_order = np.argsort(scores)
    sorted_scores = scores[sorted_order]
    qualifying_per_thr = []
    for k in range(n_thr):
        insert_pos = np.searchsorted(sorted_scores, gates[k], side='left')
        qualifying_per_thr.append(sorted_order[insert_pos:] if insert_pos < len(scores) else np.array([], dtype=np.int64))

    all_best_idxs = np.zeros((B, n_thr), dtype=np.int64)
    all_best_vals = np.full((B, n_thr), np.inf)
    for j in range(B):
        tc = coeff_matrix @ price_matrix[j] + constant
        for k in range(n_thr):
            qualifying = qualifying_per_thr[k]
            if len(qualifying) > 0:
                local_best = int(np.argmin(tc[qualifying]))
                all_best_idxs[j, k] = qualifying[local_best]
                all_best_vals[j, k] = tc[qualifying[local_best]]
    return all_best_idxs, all_best_vals


def preextract_winner_data(arrays, extras, unique_indices, iso, demand_twh):
    """Batch-extract mix + tranche + gas data for all unique winning indices at once.

    Instead of per-element array indexing in build_winner_scenario (called ~87K times
    per ISO), this extracts data for all unique winners (~100-2000) in one vectorized
    pass using numpy fancy indexing. Returns a dict keyed by index for O(1) lookup.

    Speedup: eliminates ~87K x 17 individual array element accesses, replacing them
    with ~2000 x 17 bulk extractions + dict lookups. ~1.5-2x faster for winner extraction.
    """
    if not unique_indices:
        return {}

    idx_arr = np.array(sorted(unique_indices), dtype=np.int64)
    n = len(idx_arr)

    # Batch extract from arrays (one fancy-index op per field, not per-element)
    cf_vals = arrays['clean_firm'][idx_arr]
    sol_vals = arrays['solar'][idx_arr]
    wnd_vals = arrays['wind'][idx_arr]
    hyd_vals = arrays['hydro'][idx_arr]
    match_vals = arrays['hourly_match_score'][idx_arr]
    bat_vals = arrays['battery_dispatch_pct'][idx_arr]
    bat8_arr = arrays.get('battery8_dispatch_pct')
    bat8_vals = bat8_arr[idx_arr] if bat8_arr is not None else np.zeros(n)
    ldes_vals = arrays['ldes_dispatch_pct'][idx_arr]

    # CCS = 100 - sum of other resources (int arithmetic for consistency)
    ccs_vals = np.maximum(0, 100 - (cf_vals.astype(np.int64) + sol_vals.astype(np.int64) +
                                     wnd_vals.astype(np.int64) + hyd_vals.astype(np.int64)))

    # Batch extract from extras
    match_frac_vals = extras['match_frac'][idx_arr]
    cf_existing_twh_vals = extras['cf_existing_twh'][idx_arr]
    new_cf_twh_vals = extras['new_cf_twh'][idx_arr]
    uprate_twh_vals = extras['uprate_twh'][idx_arr]
    geo_twh_vals = extras['geo_twh'][idx_arr]
    remaining_twh_vals = extras['remaining_twh'][idx_arr]
    gas_needed_mw_vals = extras['gas_needed_mw'][idx_arr]
    existing_gas_used_mw_vals = extras['existing_gas_used_mw'][idx_arr]
    new_gas_mw_vals = extras['new_gas_mw'][idx_arr]
    clean_peak_mw_vals = extras['clean_peak_mw'][idx_arr]
    ra_peak_mw = float(extras['ra_peak_mw'])

    # Pre-compute gas cost per MWh for all unique indices at once
    demand_mwh = demand_twh * 1e6
    gas_cost_per_mwh_vals = (
        existing_gas_used_mw_vals * EXISTING_GAS_FOM_KW_YR[iso] * 1000 +
        new_gas_mw_vals * NEW_CCGT_COST_KW_YR[iso] * 1000
    ) / demand_mwh

    # Build lookup dict: original_index -> pre-extracted data tuple
    # Tuple layout: [0]resource_mix, [1]match_score, [2]bat4, [3]bat8,
    #   [4]ldes, [5]match_frac, [6]cf_ex_twh, [7]uprate_twh, [8]geo_twh,
    #   [9]remaining_twh, [10]new_cf_twh, [11]gas_needed_mw, [12]ex_gas_used_mw,
    #   [13]new_gas_mw, [14]gas_cost_mwh, [15]clean_peak_mw, [16]ra_peak_mw
    winner_data = {}
    for pos in range(n):
        winner_data[int(idx_arr[pos])] = (
            {'clean_firm': int(cf_vals[pos]), 'solar': int(sol_vals[pos]),
             'wind': int(wnd_vals[pos]), 'ccs_ccgt': int(ccs_vals[pos]),
             'hydro': int(hyd_vals[pos])},
            round(float(match_vals[pos]), 4),
            round(float(bat_vals[pos]), 4),
            round(float(bat8_vals[pos]), 4),
            round(float(ldes_vals[pos]), 4),
            float(match_frac_vals[pos]),
            round(float(cf_existing_twh_vals[pos]), 3),
            round(float(uprate_twh_vals[pos]), 3),
            round(float(geo_twh_vals[pos]), 3),
            round(float(remaining_twh_vals[pos]), 3),
            round(float(new_cf_twh_vals[pos]), 3),
            round(float(gas_needed_mw_vals[pos])),
            round(float(existing_gas_used_mw_vals[pos])),
            round(float(new_gas_mw_vals[pos])),
            round(float(gas_cost_per_mwh_vals[pos]), 2),
            round(float(clean_peak_mw_vals[pos])),
            round(ra_peak_mw),
        )

    return winner_data


def build_winner_scenario_from_cache(winner_cache, best_idx, tc_val, wholesale,
                                      nuclear_price, ccs_price):
    """Build scenario result dict from pre-extracted winner data cache.

    Uses O(1) dict lookup instead of per-element array indexing. The winner_cache
    is populated by preextract_winner_data() which does batch numpy fancy indexing.
    """
    w = winner_cache[best_idx]
    match_frac = w[5]
    ec_val = tc_val / match_frac if match_frac > 0 else 0.0
    tranche3_is_nuclear = nuclear_price <= ccs_price
    remaining_twh = w[9]

    return {
        'resource_mix': w[0],
        'hourly_match_score': w[1],
        'battery_dispatch_pct': w[2],
        'battery8_dispatch_pct': w[3],
        'ldes_dispatch_pct': w[4],
        'costs': {
            'total_cost': round(tc_val, 2),
            'effective_cost': round(ec_val, 2),
            'incremental': round(ec_val - wholesale, 2),
            'wholesale': wholesale,
        },
        'tranche_costs': {
            'cf_existing_twh': w[6],
            'uprate_twh': w[7],
            'geo_twh': w[8],
            'nuclear_newbuild_twh': remaining_twh if tranche3_is_nuclear else 0.0,
            'ccs_tranche_twh': 0.0 if tranche3_is_nuclear else remaining_twh,
            'new_cf_twh': w[10],
        },
        'gas_backup': {
            'gas_backup_needed_mw': w[11],
            'existing_gas_used_mw': w[12],
            'new_gas_build_mw': w[13],
            'gas_cost_per_mwh': w[14],
            'clean_peak_capacity_mw': w[15],
            'ra_peak_mw': w[16],
        },
    }


def build_winner_scenario(arrays, extras, best_idx, sens, iso, demand_twh,
                          tc_val, wholesale, nuclear_price, ccs_price,
                          gas_fom=None, gas_ccgt=None, demand_mwh_inv=None):
    """Build scenario result dict for a single winning mix.

    Performance: inlines array indexing instead of calling arrays_to_mix_dict,
    and accepts pre-computed gas cost constants to avoid redundant lookups.

    Args:
        gas_fom: EXISTING_GAS_FOM_KW_YR[iso] * 1000 (pre-computed by caller)
        gas_ccgt: NEW_CCGT_COST_KW_YR[iso] * 1000 (pre-computed by caller)
        demand_mwh_inv: 1.0 / (demand_twh * 1e6) (pre-computed by caller)
    """
    match_frac = extras['match_frac'][best_idx]
    ec_val = tc_val / match_frac if match_frac > 0 else 0.0

    # Inline array indexing (avoids arrays_to_mix_dict function call + intermediate dict)
    cf = int(arrays['clean_firm'][best_idx])
    sol = int(arrays['solar'][best_idx])
    wnd = int(arrays['wind'][best_idx])
    hyd = int(arrays['hydro'][best_idx])
    osw_arr = arrays.get('offshore_wind')
    osw = int(osw_arr[best_idx]) if osw_arr is not None else 0
    ccs_alloc = max(0, 100 - (cf + sol + wnd + hyd + osw))
    bat8_arr = arrays.get('battery8_dispatch_pct')
    h2_arr = arrays.get('h2_dispatch_pct')

    # Tranche detail (scenario-dependent: which is cheaper, nuclear or CCS?)
    tranche3_is_nuclear = nuclear_price <= ccs_price
    remaining_twh = float(extras['remaining_twh'][best_idx])

    # Gas backup cost (use pre-computed constants if provided)
    ex_gas = float(extras['existing_gas_used_mw'][best_idx])
    new_gas = float(extras['new_gas_mw'][best_idx])
    if gas_fom is not None:
        gas_cost = (ex_gas * gas_fom + new_gas * gas_ccgt) * demand_mwh_inv
    else:
        gas_cost = (ex_gas * EXISTING_GAS_FOM_KW_YR[iso] * 1000 +
                    new_gas * NEW_CCGT_COST_KW_YR[iso] * 1000) / (demand_twh * 1e6)

    return {
        'resource_mix': {
            'clean_firm': cf,
            'solar': sol,
            'wind': wnd,
            'offshore_wind': osw,
            'ccs_ccgt': ccs_alloc,
            'hydro': hyd,
        },
        'hourly_match_score': round(float(arrays['hourly_match_score'][best_idx]), 4),
        'battery_dispatch_pct': round(float(arrays['battery_dispatch_pct'][best_idx]), 4),
        'battery8_dispatch_pct': round(float(bat8_arr[best_idx]), 4) if bat8_arr is not None else 0.0,
        'ldes_dispatch_pct': round(float(arrays['ldes_dispatch_pct'][best_idx]), 4),
        'h2_dispatch_pct': round(float(h2_arr[best_idx]), 4) if h2_arr is not None else 0.0,
        'costs': {
            'total_cost': round(tc_val, 2),
            'effective_cost': round(ec_val, 2),
            'incremental': round(ec_val - wholesale, 2),
            'wholesale': wholesale,
        },
        'tranche_costs': {
            'cf_existing_twh': round(float(extras['cf_existing_twh'][best_idx]), 3),
            'uprate_twh': round(float(extras['uprate_twh'][best_idx]), 3),
            'geo_twh': round(float(extras['geo_twh'][best_idx]), 3),
            'nuclear_newbuild_twh': round(remaining_twh if tranche3_is_nuclear else 0.0, 3),
            'ccs_tranche_twh': round(0.0 if tranche3_is_nuclear else remaining_twh, 3),
            'new_cf_twh': round(float(extras['new_cf_twh'][best_idx]), 3),
        },
        'gas_backup': {
            'gas_backup_needed_mw': round(float(extras['gas_needed_mw'][best_idx])),
            'existing_gas_used_mw': round(ex_gas),
            'new_gas_build_mw': round(new_gas),
            'gas_cost_per_mwh': round(gas_cost, 2),
            'clean_peak_capacity_mw': round(float(extras['clean_peak_mw'][best_idx])),
            'ra_peak_mw': round(float(extras['ra_peak_mw'])),
        },
    }


# ============================================================================
# SENSITIVITY KEY HELPERS
# ============================================================================

def make_scenario_key(r, f, batt, ldes, ff, tx, ccs, q45, geo):
    """Build 9-dim scenario key.
    Format: {Ren}{Firm}{Batt}{LDES}_{Fuel}_{Tx}_{CCS}{45Q}_{Geo}
    Example: MMMM_M_M_M1_M (all-Medium, 45Q on, CAISO)
    Geo='X' for non-CAISO ISOs (no geothermal resource).
    """
    geo_code = geo if geo else 'X'
    return f"{r}{f}{batt}{ldes}_{ff}_{tx}_{ccs}{q45}_{geo_code}"


def compute_tranche_for_mix(iso, cf_pct, demand_twh,
                            scenario_key, existing_scale=1.0, uprate_cap_override=None):
    """Compute clean firm tranche breakdown for a single mix.

    Returns dict with tranche TWh values: uprate, geo, nuclear_newbuild, ccs_tranche.
    Used by DG parquet writer to enrich each winning mix with tranche data.
    cf_pct is already % of demand (no procurement multiplier).
    """
    # Decode scenario key to get toggle levels
    parts = scenario_key.split('_')
    gen = parts[0]  # RFBL
    tx_name = parts[2]  # Tx level
    ccs_q45 = parts[3]  # e.g., M1
    geo_code = parts[4] if len(parts) > 4 else 'X'

    firm_lev = gen[1]  # F from RFBL
    ccs_lev = ccs_q45[0]
    q45 = ccs_q45[1] if len(ccs_q45) > 1 else '1'
    geo_lev = geo_code if geo_code != 'X' else None

    # Existing clean firm share, scaled for DG
    existing_cf = min(GRID_MIX_SHARES[iso].get('clean_firm', 0) * existing_scale, 100.0)

    # Clean firm demand % and new CF TWh (cf_pct is already % of demand)
    cf_existing_pct = min(cf_pct, existing_cf)
    cf_new_pct = max(0, cf_pct - existing_cf)
    new_cf_twh = cf_new_pct / 100.0 * demand_twh

    # Existing CF TWh
    cf_existing_twh = cf_existing_pct / 100.0 * demand_twh

    # Tranche 1: Nuclear uprates (capped)
    uprate_cap = UPRATE_CAP_TWH[iso] if uprate_cap_override is None else uprate_cap_override
    uprate_twh = min(new_cf_twh, uprate_cap)
    remaining = max(0, new_cf_twh - uprate_twh)

    # Tranche 2: Geothermal (CAISO only, capped)
    geo_twh = 0.0
    if iso == 'CAISO' and geo_lev:
        geo_twh = min(remaining, GEO_CAP_TWH)
        remaining = max(0, remaining - geo_twh)

    # Tranche 3: Cheapest of nuclear new-build vs CCS
    tx_cf = get_tx('clean_firm', tx_name, iso)
    tx_ccs_cf = get_tx('ccs_ccgt', tx_name, iso)
    ccs_table = CCS_LCOE_45Q_ON if q45 == '1' else CCS_LCOE_45Q_OFF
    nuclear_price = NUCLEAR_NEWBUILD_LCOE[firm_lev][iso] + tx_cf
    ccs_tranche_price = ccs_table[ccs_lev][iso] + tx_ccs_cf
    if iso == 'NEISO':
        ccs_tranche_price += NEISO_CCS_GAS_ADDER
    tranche3_is_nuclear = nuclear_price <= ccs_tranche_price

    nuclear_newbuild_twh = remaining if tranche3_is_nuclear else 0.0
    ccs_tranche_twh = 0.0 if tranche3_is_nuclear else remaining

    return {
        'cf_existing_twh': round(cf_existing_twh, 3),
        'uprate_twh': round(uprate_twh, 3),
        'geo_twh': round(geo_twh, 3),
        'nuclear_newbuild_twh': round(nuclear_newbuild_twh, 3),
        'ccs_tranche_twh': round(ccs_tranche_twh, 3),
        'new_cf_twh': round(new_cf_twh, 3),
    }


def medium_key(iso):
    """Return the all-Medium scenario key for an ISO.
    CAISO: MMMM_M_M_M1_M  (geothermal=M)
    Others: MMMM_M_M_M1_X  (no geothermal)
    """
    geo = 'M' if iso == 'CAISO' else 'X'
    return f'MMMM_M_M_M1_{geo}'


def build_sensitivity_combos(iso):
    """Build all sensitivity combos for an ISO.
    Returns list of (scenario_key, sens_dict) tuples.

    9 toggles: Ren(3) × Firm(3) × Battery(3) × LDES(3) × CCS(3) × 45Q(2) × Fuel(3) × Tx(4) × Geo(3|1)
    CAISO:     3^6 × 2 × 4 × 3 = 17,496 combos per threshold.
    Non-CAISO: 3^6 × 2 × 4 × 1 = 5,832 combos per threshold.
    """
    combos = []
    tx_levels = ['N', 'L', 'M', 'H']
    q45_states = ['1', '0']
    geo_levels = LMH if iso == 'CAISO' else [None]

    for r, f, batt, ldes, ff, tx, ccs, q45, geo in product(
            LMH, LMH, LMH, LMH, LMH, tx_levels, LMH, q45_states, geo_levels):
        key = make_scenario_key(r, f, batt, ldes, ff, tx, ccs, q45, geo)
        sens = {
            'ren': r, 'firm': f,
            'batt': batt, 'ldes_lvl': ldes,
            'ccs': ccs, 'q45': q45,
            'fuel': ff, 'tx': tx, 'geo': geo,
        }
        combos.append((key, sens))
    return combos


@lru_cache(maxsize=None)
def get_sensitivity_combos(iso):
    """Cached sensitivity combos for an ISO.

    Combo generation is deterministic and reused across multiple phases
    (base-year, track variants, and demand-growth sweeps). Caching avoids
    repeatedly re-allocating thousands of dictionaries per ISO.
    """
    return tuple(build_sensitivity_combos(iso))


def prepare_threshold_metadata(scores, thresholds):
    """Build reusable threshold metadata from match scores.

    Returns:
        active_thresholds: sorted ascending thresholds with at least one feasible mix
        thresholds_desc: numpy array sorted descending (for bucketed argmin kernel)
        threshold_pos: mapping threshold -> position in thresholds_desc
    """
    # Use max(scores) to skip full-array np.any per threshold — O(N) once instead of O(N×T)
    max_score = scores.max() if len(scores) > 0 else -np.inf
    active_thresholds = [thr for thr in thresholds if max_score >= effective_gate(thr)]
    thresholds_desc = np.array(sorted(active_thresholds, reverse=True), dtype=np.float64)
    threshold_pos = {float(thresholds_desc[k]): k for k in range(len(thresholds_desc))}
    return active_thresholds, thresholds_desc, threshold_pos


# ============================================================================
# LOAD PFS POST-EF
# ============================================================================

INPUT_DIR = Path('data/step2-ef-parquets')
OUTPUT_DIR = Path('data/step3-cost-opt-parquets')

# Thresholds to evaluate (imported from pipeline_config, re-assigned here for backward compat)


def effective_gate(thr):
    """Map nominal threshold to effective gate for score comparison.

    The ≥99.99% threshold is practically unreachable due to float precision —
    Step 1 caps hourly match at ~99.9%, so the best mixes score ~99.5-99.95%.
    We gate 99.99% at 99.5% to include those near-perfect mixes.
    """
    return 99.5 if thr == 99.99 else float(thr)


def precompute_threshold_indices(scores, thresholds):
    """Pre-compute qualifying indices for all thresholds using sort+searchsorted.

    Instead of O(N) np.where per threshold (O(N×T) total), this sorts once
    O(N log N) then uses binary search O(log N) per threshold. For 1.8M mixes
    × 21 thresholds, this eliminates ~37M comparisons.

    Returns dict: threshold → array of qualifying indices (into original scores).
    """
    sorted_order = np.argsort(scores)
    sorted_scores = scores[sorted_order]
    n = len(scores)

    thr_indices = {}
    for thr in thresholds:
        gate = effective_gate(thr)
        # Binary search: first position where sorted_scores >= gate
        insert_pos = np.searchsorted(sorted_scores, gate, side='left')
        if insert_pos < n:
            thr_indices[thr] = sorted_order[insert_pos:]

    return thr_indices


def _table_to_arrays(table):
    """Convert an Arrow table into in-memory numpy arrays.

    All arrays are pre-converted to float64 to avoid repeated .astype(np.float64)
    calls in downstream functions (precompute_base_year_coefficients, price_mix_batch,
    etc.). Integer fields like clean_firm are stored as float64 but remain integer-valued.
    """
    result = {
        'clean_firm': table.column('clean_firm').to_numpy().astype(np.float64),
        'solar': table.column('solar').to_numpy().astype(np.float64),
        'wind': table.column('wind').to_numpy().astype(np.float64),
        'hydro': table.column('hydro').to_numpy().astype(np.float64),
        'battery_dispatch_pct': table.column('battery_dispatch_pct').to_numpy().astype(np.float64),
        'battery8_dispatch_pct': (table.column('battery8_dispatch_pct').to_numpy().astype(np.float64)
                                  if 'battery8_dispatch_pct' in table.column_names
                                  else np.zeros(table.num_rows, dtype=np.float64)),
        'ldes_dispatch_pct': table.column('ldes_dispatch_pct').to_numpy().astype(np.float64),
        'h2_dispatch_pct': (table.column('h2_dispatch_pct').to_numpy().astype(np.float64)
                            if 'h2_dispatch_pct' in table.column_names
                            else np.zeros(table.num_rows, dtype=np.float64)),
        'hourly_match_score': table.column('hourly_match_score').to_numpy().astype(np.float64),
    }
    # Offshore wind (CAISO, PJM, NYISO, NEISO)
    if 'offshore_wind' in table.column_names:
        result['offshore_wind'] = table.column('offshore_wind').to_numpy().astype(np.float64)
    # CAISO geothermal (5th resource dimension)
    if 'geothermal' in table.column_names:
        result['geothermal'] = table.column('geothermal').to_numpy().astype(np.float64)
    return result


def _discover_iso_threshold_files(input_dir, iso):
    """Discover per-threshold EF band files for an ISO.

    Looks for step2_ef_{ISO}_t{T}.parquet files. Falls back to the legacy
    monolithic step2_ef_{ISO}.parquet if no per-threshold files exist.

    Returns list of Path objects, or empty list if nothing found.
    """
    import glob as globmod

    # Try per-threshold files first (new format)
    pattern = str(input_dir / f'step2_ef_{iso}_t*.parquet')
    thr_files = sorted(globmod.glob(pattern))
    # Exclude batch files
    thr_files = [f for f in thr_files if '_batch_' not in os.path.basename(f)]

    if thr_files:
        return [Path(f) for f in thr_files]

    # Fall back to legacy monolithic file
    legacy = input_dir / f'step2_ef_{iso}.parquet'
    if legacy.exists():
        return [legacy]

    return []


def load_pfs_post_ef(input_dir, selected_isos=None):
    """Load PFS post-EF from per-ISO/threshold parquet files in input_dir.

    Reads from data/step2-ef-parquets/step2_ef_{ISO}_t{T}.parquet (per-threshold
    band files). Falls back to legacy step2_ef_{ISO}.parquet if per-threshold
    files don't exist.

    Per-threshold band files contain non-overlapping score ranges. All band
    files for an ISO are concatenated to reconstruct the full EF.

    Returns:
        pfs: dict keyed by ISO → numpy arrays of all mixes
        thr_indices: dict keyed by (ISO, threshold) → index array of qualifying mixes
    """
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}\n"
            f"Run Step 2 first to generate per-ISO EF parquets."
        )

    pfs = {}
    thr_indices = {}
    isos_to_load = selected_isos or ISOS

    for iso in isos_to_load:
        iso_files = _discover_iso_threshold_files(input_dir, iso)
        if not iso_files:
            print(f"  WARNING: No EF parquets found for {iso} in {input_dir} — skipping")
            continue

        # Load and concatenate all files for this ISO
        tables = []
        for fpath in iso_files:
            t = pq.read_table(fpath)
            if t.num_rows > 0:
                tables.append(t)

        if not tables:
            print(f"  WARNING: All EF parquets empty for {iso} — skipping")
            continue

        if len(tables) == 1:
            combined = tables[0]
        else:
            combined = pa.concat_tables(tables, promote_options='permissive')
        del tables

        arrays = _table_to_arrays(combined)
        del combined
        pfs[iso] = arrays

        # Sort+searchsorted: O(N log N + T log N) instead of O(N×T)
        scores = arrays['hourly_match_score']
        iso_thr_idx = precompute_threshold_indices(scores, OUTPUT_THRESHOLDS)
        for thr, idx in iso_thr_idx.items():
            thr_indices[(iso, thr)] = idx

        n_files = len(iso_files)
        fmt = "file" if n_files == 1 else "files"
        print(f"  {iso}: loaded {len(scores):,} mixes from {n_files} {fmt}")

    if not pfs:
        raise FileNotFoundError(
            f"No valid EF parquet files found in {input_dir} for ISOs: {isos_to_load}\n"
            f"Run Step 2 first to generate per-ISO EF parquets."
        )

    return pfs, thr_indices


def arrays_to_mix_dict(arrays, idx):
    """Extract a single mix from arrays as a dict."""
    osw_arr = arrays.get('offshore_wind')
    osw = int(osw_arr[idx]) if osw_arr is not None else 0
    ccs_pct = max(0, 100 - (int(arrays['clean_firm'][idx]) + int(arrays['solar'][idx]) +
                             int(arrays['wind'][idx]) + int(arrays['hydro'][idx]) + osw))
    bat8 = arrays.get('battery8_dispatch_pct')
    h2 = arrays.get('h2_dispatch_pct')
    return {
        'resource_mix': {
            'clean_firm': int(arrays['clean_firm'][idx]),
            'solar': int(arrays['solar'][idx]),
            'wind': int(arrays['wind'][idx]),
            'offshore_wind': osw,
            'ccs_ccgt': ccs_pct,
            'hydro': int(arrays['hydro'][idx]),
        },
        'hourly_match_score': round(float(arrays['hourly_match_score'][idx]), 4),
        'battery_dispatch_pct': round(float(arrays['battery_dispatch_pct'][idx]), 4),
        'battery8_dispatch_pct': round(float(bat8[idx]), 4) if bat8 is not None else 0.0,
        'ldes_dispatch_pct': round(float(arrays['ldes_dispatch_pct'][idx]), 4),
        'h2_dispatch_pct': round(float(h2[idx]), 4) if h2 is not None else 0.0,
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Step 3 cost optimization runner.')
    parser.add_argument(
        '--iso',
        dest='isos',
        action='append',
        choices=ISOS,
        help='ISO to run (repeatable). Defaults to all ISOs.'
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=INPUT_DIR,
        help=f'Directory containing step2_ef_<ISO>.parquet files (default: {INPUT_DIR}).'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=OUTPUT_DIR,
        help=f'Directory for per-ISO output parquets (default: {OUTPUT_DIR}).'
    )
    parser.add_argument(
        '--track',
        type=str,
        default='baseline',
        choices=['baseline', 'newbuild', 'replace'],
        help='Which track to run. Only one track per invocation to bound memory. '
             'baseline=Phase 1+2 (clean floor + hydro cap filtered mixes), '
             'newbuild=hydro=0 mixes with uprates ON, '
             'replace=all mixes with uprates OFF. Default: baseline.'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of parallel ISO workers. Numba already parallelizes within each '
             'ISO via prange, so multi-worker mode is most useful when Numba thread '
             'count is limited per worker. 0 = auto (CPU count). Default: 1.'
    )
    return parser.parse_args()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    args = parse_args()
    run_isos = args.isos if args.isos else ISOS
    output_dir = args.output_dir
    track = args.track

    print("=" * 70)
    print("  STEP 3: COST OPTIMIZATION (v4 — vectorized, threshold-free PFS)")
    print("=" * 70)
    print(f"  Track:      {track}")
    print(f"  Requested ISOs: {', '.join(run_isos)}")
    print(f"  Input dir:  {args.input_dir}")
    print(f"  Output dir: {output_dir}")

    total_start = time.time()

    # Load PFS post-EF (threshold-free: one set of mixes per ISO)
    pfs, thr_indices = load_pfs_post_ef(input_dir=args.input_dir, selected_isos=run_isos)
    print(f"  ISOs loaded: {len(pfs)}")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build output structure
    output = {
        'config': {
            'data_year': 2025,
            'thresholds': OUTPUT_THRESHOLDS,
            'resource_types': RESOURCE_TYPES,
            'wholesale_prices': WHOLESALE_PRICES,
            'grid_mix_shares': GRID_MIX_SHARES,
            'lcoe_tables': LCOE_TABLES,
            'transmission_tables': TX_TABLES,
            'fuel_prices': FUEL_ADJUSTMENTS,
            'tranche_model': {
                'uprate_caps_twh': UPRATE_CAP_TWH,
                'uprate_lcoe': UPRATE_LCOE,
                'nuclear_newbuild_lcoe': NUCLEAR_NEWBUILD_LCOE,
                'geothermal_lcoe': {'L': {'CAISO': 63}, 'M': {'CAISO': 88}, 'H': {'CAISO': 110}},
                'geothermal_cap_twh': GEO_CAP_TWH,
                'ccs_lcoe_45q_on': CCS_LCOE_45Q_ON,
                'ccs_lcoe_45q_off': CCS_LCOE_45Q_OFF,
                'existing_nuclear_gw': EXISTING_NUCLEAR_GW,
            },
            'demand_growth_rates': DEMAND_GROWTH_RATES,
        },
        'results': {},
    }
    # ================================================================
    # PHASE 1: Base year cross-evaluation (pre-computed + Numba)
    # ================================================================
    # Pre-compute scenario-invariant coefficient arrays per ISO, then
    # evaluate total_cost as a fused multiply-add (10 coefficients × 10 prices).
    # Numba prange parallelizes across CPU cores; no temporary arrays.
    print("\n--- PHASE 1: Base year cross-evaluation (pre-computed + Numba) ---")
    print(f"  Numba JIT: {'enabled' if HAS_NUMBA else 'DISABLED (numpy fallback)'}")
    phase1_start = time.time()

    # Warm up Numba JIT on first call (compile cost)
    if HAS_NUMBA:
        _dummy_cm = np.zeros((2, _N_COEFFS))
        _dummy_c = np.zeros(2)
        _dummy_p = np.zeros(_N_COEFFS)
        _dummy_s = np.array([50.0, 100.0])
        _dummy_t = np.array([100.0, 50.0])
        _dummy_tc = _eval_cost_numba(_dummy_cm, _dummy_c, _dummy_p)
        _argmin_bucketed(_dummy_tc, _dummy_s, _dummy_t)
        _dummy_pm = np.zeros((2, _N_COEFFS))
        _batch_eval_and_argmin(_dummy_cm, _dummy_c, _dummy_pm, _dummy_s, _dummy_t)
        print("  Numba JIT warmup complete (incl. batched)")

    # Archetypes per ISO (union of winners across all thresholds) for Phase 2
    archetypes = {}

    # Cache price matrices per ISO — precompute_all_prices depends only on
    # (iso, sensitivity_combos), not on arrays, uprate settings, or demand growth.
    # Caching avoids recomputing the same price lookups in Phase 1.5 tracks,
    # Phase 2 demand growth, and Phase 2.5 track DG sweeps.
    iso_price_cache = {}  # iso → (price_matrix, wholesale_arr, nuclear_arr, ccs_arr)

    # Cache Phase 1 filtered arrays per ISO (after clean floor + hydro cap).
    # Phase 2 DG sweep must use the same filtered arrays as Phase 1 so archetype
    # indices are consistent (they index into filtered arrays, not raw pfs[iso]).
    phase1_arrays = {}  # iso → filtered arrays dict

    for iso in run_isos:
        if iso not in pfs:
            continue

        # Apply existing clean floor filter: mandate existing clean in every mix
        raw_arrays = pfs[iso]
        arrays, n_filtered = apply_existing_clean_floor(raw_arrays, iso)
        N_raw = len(raw_arrays['clean_firm'])
        N = len(arrays['clean_firm'])
        if n_filtered > 0:
            print(f"    {iso}: existing clean floor filter removed {n_filtered:,} / {N_raw:,} mixes "
                  f"({n_filtered/N_raw*100:.1f}%) — {N:,} remaining")

        # Apply hydro cap: exclude mixes with hydro above existing ceiling.
        # PFS explores hydro +10% adder for physics experimentation; those mixes
        # stay in the EF for Track 2 but must not enter Track 1 baseline costing.
        arrays, n_hydro_removed = apply_hydro_cap(arrays, iso)
        if n_hydro_removed > 0:
            print(f"    {iso}: hydro cap filter removed {n_hydro_removed:,} / {N:,} mixes "
                  f"(hydro > {GRID_MIX_SHARES[iso].get('hydro', 0)}%) — "
                  f"{len(arrays['clean_firm']):,} remaining")
            N = len(arrays['clean_firm'])

        if n_filtered > 0 or n_hydro_removed > 0:
            # Recompute threshold indices for filtered arrays (old indices pointed into raw)
            # Sort+searchsorted: O(N log N + T log N) instead of O(N×T)
            scores_filt = arrays['hourly_match_score']
            filt_thr_idx = precompute_threshold_indices(scores_filt, OUTPUT_THRESHOLDS)
            # Replace old indices, remove thresholds that no longer have qualifying mixes
            for thr in OUTPUT_THRESHOLDS:
                if thr in filt_thr_idx:
                    thr_indices[(iso, thr)] = filt_thr_idx[thr]
                elif (iso, thr) in thr_indices:
                    del thr_indices[(iso, thr)]

        # Cache filtered arrays for Phase 2 DG sweep (archetype indices are into these)
        phase1_arrays[iso] = arrays
        demand_twh = REGIONAL_DEMAND_TWH[iso]

        output['results'][iso] = {
            'annual_demand_mwh': demand_twh * 1e6,
            'thresholds': {},
        }

        # Pre-compute coefficient matrix + constant (one-time per ISO)
        coeff_matrix, constant, extras = precompute_base_year_coefficients(
            iso, arrays, demand_twh)
        # Already float64 from _table_to_arrays
        scores = arrays['hourly_match_score']

        all_combos = get_sensitivity_combos(iso)
        n_combos = len(all_combos)

        active_thresholds, thresholds_desc, thr_pos = prepare_threshold_metadata(
            scores, OUTPUT_THRESHOLDS)

        thr_data = {}
        thr_arch_sets = {}
        for thr in active_thresholds:
            thr_data[thr] = {'scenarios': {}, 'feasible_mixes': {}}
            thr_arch_sets[thr] = set()

        iso_arch_set = set()
        iso_start = time.time()

        # Batched evaluation: all combos at once via Numba kernel
        # Cache price matrix for reuse in Phase 1.5, 2, and 2.5
        price_matrix, wholesale_arr, nuclear_arr, ccs_arr = precompute_all_prices(
            iso, all_combos)
        iso_price_cache[iso] = (price_matrix, wholesale_arr, nuclear_arr, ccs_arr)
        eval_start = time.time()
        all_best_idxs, all_best_vals = batch_eval_and_argmin_all(
            coeff_matrix, constant, price_matrix, scores, thresholds_desc)
        eval_elapsed = time.time() - eval_start
        print(f"    {iso}: batched eval {n_combos} combos × {N:,} mixes — {eval_elapsed:.1f}s")

        # Two-pass winner extraction:
        # Pass 1: Collect unique winning indices + archetype sets (fast, no array access)
        # Pass 2: Batch-extract all unique winner data, then build scenarios from cache
        # This avoids ~87K redundant per-element array accesses (only ~100-2000 unique winners).
        n_thr = len(active_thresholds)
        thr_k_map = [thr_pos[float(thr)] for thr in active_thresholds]
        for combo_i in range(n_combos):
            for ti in range(n_thr):
                k = thr_k_map[ti]
                if all_best_vals[combo_i, k] == np.inf:
                    continue
                best_idx = int(all_best_idxs[combo_i, k])
                thr_arch_sets[active_thresholds[ti]].add(best_idx)
                iso_arch_set.add(best_idx)

        # Batch pre-extract all unique winner data using numpy fancy indexing
        winner_cache = preextract_winner_data(arrays, extras, iso_arch_set, iso, demand_twh)
        print(f"    {iso}: {len(iso_arch_set)} unique winners pre-extracted")

        # Build scenario dicts from cache (O(1) dict lookup per winner, no array indexing)
        for combo_i, (scenario_key, sens) in enumerate(all_combos):
            ws = float(wholesale_arr[combo_i])
            nuc = float(nuclear_arr[combo_i])
            ccs_p = float(ccs_arr[combo_i])
            for ti in range(n_thr):
                k = thr_k_map[ti]
                if all_best_vals[combo_i, k] == np.inf:
                    continue
                best_idx = int(all_best_idxs[combo_i, k])
                tc_val = float(all_best_vals[combo_i, k])

                scenario = build_winner_scenario_from_cache(
                    winner_cache, best_idx, tc_val, ws, nuc, ccs_p)
                thr_data[active_thresholds[ti]]['scenarios'][scenario_key] = scenario

            if (combo_i + 1) % 5000 == 0:
                print(f"    {iso}: extracted {combo_i+1}/{n_combos} winners")

        # Store feasible mixes per threshold
        for thr in active_thresholds:
            idx = thr_indices[(iso, thr)]
            N_thr = len(idx)

            max_feasible = min(N_thr, 500)
            step = max(1, N_thr // max_feasible)
            sample_full = list(idx[::step])
            for aidx in thr_arch_sets[thr]:
                if aidx not in sample_full:
                    sample_full.append(aidx)
            sample_full.sort()

            idx_arr = np.array(sample_full)
            # Index + convert to int once; reuse for CCS computation and .tolist()
            _cf_i = arrays['clean_firm'][idx_arr].astype(np.int64)
            _sol_i = arrays['solar'][idx_arr].astype(np.int64)
            _wnd_i = arrays['wind'][idx_arr].astype(np.int64)
            _hyd_i = arrays['hydro'][idx_arr].astype(np.int64)
            ccs_pct = np.maximum(0, 100 - (_cf_i + _sol_i + _wnd_i + _hyd_i))
            bat8 = arrays.get('battery8_dispatch_pct', np.zeros(N, dtype=np.float64))
            h2 = arrays.get('h2_dispatch_pct', np.zeros(N, dtype=np.float64))
            thr_data[thr]['feasible_mixes'] = {
                'clean_firm': _cf_i.tolist(),
                'solar': _sol_i.tolist(),
                'wind': _wnd_i.tolist(),
                'ccs_ccgt': ccs_pct.tolist(),
                'hydro': _hyd_i.tolist(),
                'hourly_match_score': np.round(arrays['hourly_match_score'][idx_arr], 4).tolist(),
                'battery_dispatch_pct': np.round(arrays['battery_dispatch_pct'][idx_arr], 4).tolist(),
                'battery8_dispatch_pct': np.round(bat8[idx_arr], 4).tolist(),
                'ldes_dispatch_pct': np.round(arrays['ldes_dispatch_pct'][idx_arr], 4).tolist(),
                'h2_dispatch_pct': np.round(h2[idx_arr], 4).tolist(),
            }

            t_str = str(thr)
            output['results'][iso]['thresholds'][t_str] = thr_data[thr]

        archetypes[iso] = iso_arch_set
        iso_elapsed = time.time() - iso_start

        print(f"  {iso:>6}: {N:>7,} mixes, {n_combos} scenarios, "
              f"{len(active_thresholds)} thresholds, "
              f"{len(iso_arch_set)} archetypes — {iso_elapsed:.0f}s")

    phase1_elapsed = time.time() - phase1_start
    print(f"\nPhase 1 complete: {phase1_elapsed:.0f}s")

    # Free raw pfs arrays for baseline track — Phase 1.5 won't run, Phase 2
    # uses phase1_arrays (filtered). For non-baseline, pfs is still needed.
    if track == 'baseline':
        for _iso_key in list(pfs.keys()):
            del pfs[_iso_key]
        gc.collect()
        print("  Freed raw PFS arrays (baseline: Phase 1.5 skipped)")

    # ================================================================
    # PHASE 1.5: TRACK ANALYSIS (newbuild + replace)
    # ================================================================
    # Skipped for baseline track — tracks 2/3 run via separate --track invocations.
    # This prevents OOM on large ISOs (e.g., MISO 46M mixes → 4+ GB coeff_matrix).
    _skip_phase15 = (track == 'baseline')
    if _skip_phase15:
        print("\n--- Skipping Phase 1.5 (track=baseline) ---")
    else:
        print(f"\n--- PHASE 1.5: Track analysis ({track}) ---")
    phase15_start = time.time()

    track_output = {
        'meta': {
            'tracks': {
                'newbuild': {
                    'description': 'New-build requirement for hourly matching',
                    'hydro': 'excluded (hydro=0 mixes only)',
                    'uprates': 'on (uprate tranche active)',
                    'purpose': 'What resources does hourly matching incentivize vs what the grid needs?',
                },
                'replace': {
                    'description': 'Cost to replace existing clean generation',
                    'hydro': 'included (up to existing floor, $0 — sunk fleet)',
                    'uprates': 'off (uprate_cap=0, all new CF at new-build prices)',
                    'purpose': 'True cost of replacing existing nuclear/firm with new-build',
                },
            },
            'thresholds': OUTPUT_THRESHOLDS,
            'resource_types': RESOURCE_TYPES,
        },
        'results': {},
    }

    # Track archetypes for Phase 2 demand growth
    track_archetypes = {'newbuild': {}, 'replace': {}}

    for iso in run_isos:
        if _skip_phase15:
            break  # Skip entire Phase 1.5 loop for baseline track
        if iso not in pfs:
            continue

        arrays = pfs[iso]
        N = len(arrays['clean_firm'])
        demand_twh = REGIONAL_DEMAND_TWH[iso]
        all_combos = get_sensitivity_combos(iso)
        n_combos = len(all_combos)
        # Already float64 from _table_to_arrays
        scores_all = arrays['hourly_match_score']

        track_output['results'][iso] = {}

        # ------ Track 1 (newbuild): Filter to hydro=0 ------
        hydro_arr = arrays['hydro']
        h0_mask = hydro_arr == 0
        n_h0 = h0_mask.sum()

        if n_h0 > 0:
            h0_idx = np.where(h0_mask)[0]
            nb_arrays = {k: arrays[k][h0_idx] for k in arrays}
            # Already float64 from _table_to_arrays (slicing preserves dtype)
            nb_scores = nb_arrays['hourly_match_score']

            # Pre-compute threshold indices via sort+searchsorted
            nb_thr_indices = precompute_threshold_indices(nb_scores, OUTPUT_THRESHOLDS)

            if nb_thr_indices:
                nb_cm, nb_const, nb_extras = precompute_base_year_coefficients(
                    iso, nb_arrays, demand_twh)  # uprates ON (default)

                active_thr_nb, thresholds_desc_nb, thr_pos_nb = prepare_threshold_metadata(
                    nb_scores, OUTPUT_THRESHOLDS)

                nb_thr_data = {thr: {'scenarios': {}} for thr in active_thr_nb}
                nb_arch_set = set()

                iso_start_nb = time.time()
                # Reuse cached price matrix from Phase 1 (same iso + combos)
                nb_price_matrix, nb_ws_arr, nb_nuc_arr, nb_ccs_arr = iso_price_cache[iso]
                nb_all_idxs, nb_all_vals = batch_eval_and_argmin_all(
                    nb_cm, nb_const, nb_price_matrix, nb_scores, thresholds_desc_nb)
                print(f"    {iso} newbuild: batched eval {n_combos} combos × {n_h0:,} mixes "
                      f"— {time.time()-iso_start_nb:.1f}s")

                # Two-pass extraction: collect unique indices, then batch extract
                nb_n_thr = len(active_thr_nb)
                nb_thr_k_map = [thr_pos_nb[float(thr)] for thr in active_thr_nb]
                for combo_i in range(n_combos):
                    for ti in range(nb_n_thr):
                        k = nb_thr_k_map[ti]
                        if nb_all_vals[combo_i, k] == np.inf:
                            continue
                        nb_arch_set.add(int(nb_all_idxs[combo_i, k]))

                nb_winner_cache = preextract_winner_data(
                    nb_arrays, nb_extras, nb_arch_set, iso, demand_twh)

                for combo_i, (scenario_key, sens) in enumerate(all_combos):
                    ws = float(nb_ws_arr[combo_i])
                    nuc = float(nb_nuc_arr[combo_i])
                    ccs_p = float(nb_ccs_arr[combo_i])
                    for ti in range(nb_n_thr):
                        k = nb_thr_k_map[ti]
                        if nb_all_vals[combo_i, k] == np.inf:
                            continue
                        best_idx = int(nb_all_idxs[combo_i, k])
                        tc_val = float(nb_all_vals[combo_i, k])

                        scenario = build_winner_scenario_from_cache(
                            nb_winner_cache, best_idx, tc_val, ws, nuc, ccs_p)
                        nb_thr_data[active_thr_nb[ti]]['scenarios'][scenario_key] = scenario

                track_output['results'][iso]['newbuild'] = {
                    str(thr): nb_thr_data[thr] for thr in active_thr_nb
                }
                track_archetypes['newbuild'][iso] = {
                    'arch_set': nb_arch_set,
                    'arrays': nb_arrays,
                    'h0_idx': h0_idx,
                    'uprate_cap_override': None,  # uprates ON
                }
                print(f"  {iso:>6} newbuild: {n_h0:,} hydro=0 mixes, "
                      f"{len(active_thr_nb)} thresholds, {len(nb_arch_set)} archetypes "
                      f"— {time.time()-iso_start_nb:.0f}s")
            else:
                print(f"  {iso:>6} newbuild: {n_h0:,} hydro=0 mixes but none meet any threshold")
        else:
            print(f"  {iso:>6} newbuild: no hydro=0 mixes available")

        # ------ Track 2 (replace): All mixes, uprate_cap=0 ------
        rp_cm, rp_const, rp_extras = precompute_base_year_coefficients(
            iso, arrays, demand_twh, uprate_cap_override=0)  # uprates OFF

        active_thr_rp, thresholds_desc_rp, thr_pos_rp = prepare_threshold_metadata(
            scores_all, OUTPUT_THRESHOLDS)

        rp_thr_data = {thr: {'scenarios': {}} for thr in active_thr_rp}
        rp_arch_set = set()

        iso_start_rp = time.time()
        # Reuse cached price matrix from Phase 1 (same iso + combos)
        rp_price_matrix, rp_ws_arr, rp_nuc_arr, rp_ccs_arr = iso_price_cache[iso]
        rp_all_idxs, rp_all_vals = batch_eval_and_argmin_all(
            rp_cm, rp_const, rp_price_matrix, scores_all, thresholds_desc_rp)
        print(f"    {iso} replace: batched eval {n_combos} combos × {N:,} mixes "
              f"— {time.time()-iso_start_rp:.1f}s")

        # Two-pass extraction: collect unique indices, then batch extract
        rp_n_thr = len(active_thr_rp)
        rp_thr_k_map = [thr_pos_rp[float(thr)] for thr in active_thr_rp]
        for combo_i in range(n_combos):
            for ti in range(rp_n_thr):
                k = rp_thr_k_map[ti]
                if rp_all_vals[combo_i, k] == np.inf:
                    continue
                rp_arch_set.add(int(rp_all_idxs[combo_i, k]))

        rp_winner_cache = preextract_winner_data(
            arrays, rp_extras, rp_arch_set, iso, demand_twh)

        for combo_i, (scenario_key, sens) in enumerate(all_combos):
            ws = float(rp_ws_arr[combo_i])
            nuc = float(rp_nuc_arr[combo_i])
            ccs_p = float(rp_ccs_arr[combo_i])
            for ti in range(rp_n_thr):
                k = rp_thr_k_map[ti]
                if rp_all_vals[combo_i, k] == np.inf:
                    continue
                best_idx = int(rp_all_idxs[combo_i, k])
                tc_val = float(rp_all_vals[combo_i, k])

                scenario = build_winner_scenario_from_cache(
                    rp_winner_cache, best_idx, tc_val, ws, nuc, ccs_p)
                rp_thr_data[active_thr_rp[ti]]['scenarios'][scenario_key] = scenario

        track_output['results'][iso]['replace'] = {
            str(thr): rp_thr_data[thr] for thr in active_thr_rp
        }
        track_archetypes['replace'][iso] = {
            'arch_set': rp_arch_set,
            'arrays': arrays,
            'uprate_cap_override': 0,  # uprates OFF
        }
        print(f"  {iso:>6} replace:  {N:,} mixes, "
              f"{len(active_thr_rp)} thresholds, {len(rp_arch_set)} archetypes "
              f"— {time.time()-iso_start_rp:.0f}s")

    phase15_elapsed = time.time() - phase15_start
    print(f"\nPhase 1.5 complete: {phase15_elapsed:.0f}s")

    # ================================================================
    # PHASE 2: Demand growth sweep on archetypes (threshold-year paired)
    # ================================================================
    # Each threshold is paired with its SBTi-interpolated target year.
    # Instead of 25 years × 3 growth × all thresholds (75 evals), we do
    # 14 unique years × 3 growth levels = 42 evals, extracting only the
    # threshold(s) that map to each year. Produces 15 × 3 = 45 DG results
    # per ISO per scenario combo.
    n_dg_evals = len(DG_UNIQUE_YEARS) * len(DEMAND_GROWTH_LEVELS)
    print(f"\n--- PHASE 2: Demand growth sweep (threshold-year paired, {n_dg_evals} evals) ---")
    phase2_start = time.time()

    dg_output = {
        'meta': {
            'threshold_target_years': THRESHOLD_TARGET_YEARS,
            'growth_levels': DEMAND_GROWTH_LEVELS,
            'growth_rates': DEMAND_GROWTH_RATES,
            'base_year': 2025,
            'fields': ['mix_idx', 'total_cost', 'effective_cost', 'incremental',
                        'growth_factor', 'annual_demand_mwh'],
            'note': 'Threshold-year paired DG with Wright\'s Law FOAK→NOAK learning curves. '
                    'Each DG year uses year-adjusted clean firm costs (nuclear, CCS, geo, LDES, H2). '
                    'L/M/H toggles control both NOAK endpoint and adoption speed.',
            'learning_curves': {
                'enabled': True,
                'exponent': LEARNING_EXPONENT,
                'params': {tech: {lev: {'foak_start': s, 'noak_year': n}
                                  for lev, (s, n) in params.items()}
                           for tech, params in LEARNING_PARAMS.items()},
            },
        },
        'results': {},
    }

    for iso in run_isos:
        if iso not in phase1_arrays or iso not in archetypes:
            continue

        dg_output['results'][iso] = {}
        # Use Phase 1 filtered arrays (after clean floor + hydro cap) so archetype
        # indices are consistent — they were computed against filtered arrays.
        arrays = phase1_arrays[iso]
        demand_twh = REGIONAL_DEMAND_TWH[iso]
        iso_rates = DEMAND_GROWTH_RATES[iso]
        all_combos = get_sensitivity_combos(iso)

        arch_indices = sorted(archetypes[iso])
        n_arch = len(arch_indices)
        if n_arch == 0:
            continue

        # Extract archetype sub-arrays (evaluate only these in Phase 2)
        arch_arrays = {k: arrays[k][arch_indices] for k in arrays}

        # Pre-compute which archetypes qualify for each threshold (sort+searchsorted)
        arch_scores = arch_arrays['hourly_match_score']
        arch_thr_mask = precompute_threshold_indices(arch_scores, OUTPUT_THRESHOLDS)

        # Initialize per-threshold result dicts
        thr_dg = {thr: {} for thr in arch_thr_mask}

        # Phase 1 wholesale prices (scenario-invariant, no learning curve)
        _, dg_ws_arr, _, _ = iso_price_cache[iso]

        # Pre-compute year-specific price matrices with learning curves.
        # Each DG year gets its own price matrix where nuclear/CCS/geo/LDES/H2
        # costs are adjusted via Wright's Law FOAK→NOAK interpolation.
        dg_price_cache = {}
        for year in DG_UNIQUE_YEARS:
            dg_price_cache[year] = precompute_all_prices(iso, all_combos, target_year=year)

        # Archetype scores for batched eval + effective cost
        # Already float64 from _table_to_arrays — no .astype() needed
        arch_scores_f64 = arch_scores
        arch_match_frac = arch_scores_f64 / 100.0

        # Thresholds for batched eval (sorted descending)
        active_thr_dg, thresholds_desc_dg, thr_pos_dg = prepare_threshold_metadata(
            arch_scores_f64, OUTPUT_THRESHOLDS)

        # Pre-initialize result structure: thr → scenario_key → {year_str → {growth_level → vals}}
        for scenario_key, _ in all_combos:
            for thr in arch_thr_mask:
                thr_dg[thr][scenario_key] = {}

        # Batched demand growth: iterate unique DG years × 3 growth levels
        # Each year maps to specific threshold(s); only store results for matched thresholds
        dg_iso_start = time.time()
        existing_base = GRID_MIX_SHARES[iso]
        n_growth_evals = 0

        for year in DG_UNIQUE_YEARS:
            years_out = year - 2025
            # Which thresholds map to this year?
            matched_thresholds = [t for t in _DG_YEAR_TO_THRESHOLDS[year]
                                  if t in arch_thr_mask]
            if not matched_thresholds:
                continue

            # Year-specific price matrix with learning curves applied
            dg_price_matrix_yr, _, _, _ = dg_price_cache[year]

            for g_level in DEMAND_GROWTH_LEVELS:
                g_rate = iso_rates[g_level]
                gf = (1 + g_rate) ** years_out
                demand_grown = demand_twh * gf
                existing_scale = 1.0 / gf
                demand_grown_mwh = demand_grown * 1e6

                # Scale existing shares for this growth factor
                scaled_existing = {k: min(v * existing_scale, 100.0)
                                   for k, v in existing_base.items()}

                # Pre-compute coefficients for this growth year (keep extras for tranche data)
                cm_g, const_g, dg_extras = precompute_base_year_coefficients(
                    iso, arch_arrays, demand_grown, existing_override=scaled_existing)

                # Batch eval all combos at once using year-adjusted prices
                dg_best_idxs, dg_best_vals = batch_eval_and_argmin_all(
                    cm_g, const_g, dg_price_matrix_yr, arch_scores_f64, thresholds_desc_dg)

                # Extract winners — ONLY for thresholds that map to this year
                yr_str = str(year)
                for combo_i in range(len(all_combos)):
                    ws = float(dg_ws_arr[combo_i])
                    scenario_key = all_combos[combo_i][0]
                    for thr in matched_thresholds:
                        k = thr_pos_dg[float(thr)]
                        if dg_best_vals[combo_i, k] == np.inf:
                            continue
                        best_local = int(dg_best_idxs[combo_i, k])
                        full_idx = arch_indices[best_local]
                        tc = float(dg_best_vals[combo_i, k])
                        mf = float(arch_match_frac[best_local])
                        ec = tc / mf if mf > 0 else 0.0

                        # Extract tranche data for this winner from extras
                        tranche = {
                            'cf_existing_twh': float(dg_extras['cf_existing_twh'][best_local]),
                            'uprate_twh': float(dg_extras['uprate_twh'][best_local]),
                            'geo_twh': float(dg_extras['geo_twh'][best_local]),
                            'remaining_twh': float(dg_extras['remaining_twh'][best_local]),
                            'new_cf_twh': float(dg_extras['new_cf_twh'][best_local]),
                            'gas_needed_mw': float(dg_extras['gas_needed_mw'][best_local]),
                            'existing_gas_mw': float(dg_extras['existing_gas_used_mw'][best_local]),
                            'new_gas_mw': float(dg_extras['new_gas_mw'][best_local]),
                        }

                        if yr_str not in thr_dg[thr][scenario_key]:
                            thr_dg[thr][scenario_key][yr_str] = {}
                        thr_dg[thr][scenario_key][yr_str][g_level] = [
                            full_idx, round(tc, 2), round(ec, 2), round(ec - ws, 2),
                            round(gf, 6), round(demand_grown_mwh, 0), tranche]

                n_growth_evals += 1
                if n_growth_evals % 10 == 0:
                    elapsed = time.time() - dg_iso_start
                    rate = n_growth_evals / elapsed if elapsed > 0 else 1
                    remaining = n_dg_evals - n_growth_evals
                    print(f"    {iso}: {n_growth_evals}/{n_dg_evals} growth evals "
                          f"({elapsed:.0f}s, ~{remaining/rate:.0f}s left)")

        for thr in arch_thr_mask:
            dg_output['results'][iso][str(thr)] = thr_dg[thr]

        dg_iso_elapsed = time.time() - dg_iso_start
        print(f"  {iso:>6}: {n_arch} archetypes, "
              f"{len(arch_thr_mask)} thresholds, "
              f"{len(all_combos)} scenarios × {n_dg_evals} growth evals — {dg_iso_elapsed:.0f}s")

    phase2_elapsed = time.time() - phase2_start
    print(f"\nPhase 2 complete: {phase2_elapsed:.0f}s")

    # ================================================================
    # PHASE 2.5: Track demand growth sweeps — SKIP FOR BASELINE
    # ================================================================
    _skip_phase25 = (track == 'baseline')
    if _skip_phase25:
        print("\n--- Skipping Phase 2.5 (track=baseline) ---")
    else:
        print(f"\n--- PHASE 2.5: Track demand growth sweeps ({track}) ---")
    phase25_start = time.time()

    track_dg = {'newbuild': {}, 'replace': {}}

    for track_name in ['newbuild', 'replace']:
        if _skip_phase25:
            break  # Skip entire Phase 2.5 for baseline track
        for iso in run_isos:
            if iso not in track_archetypes[track_name]:
                continue

            ta = track_archetypes[track_name][iso]
            arch_set = ta['arch_set']
            t_arrays = ta['arrays']
            uprate_override = ta['uprate_cap_override']

            demand_twh = REGIONAL_DEMAND_TWH[iso]
            iso_rates = DEMAND_GROWTH_RATES[iso]
            all_combos = get_sensitivity_combos(iso)

            arch_indices = sorted(arch_set)
            n_arch = len(arch_indices)
            if n_arch == 0:
                continue

            arch_arrays = {k: t_arrays[k][arch_indices] for k in t_arrays}
            arch_scores = arch_arrays['hourly_match_score']
            arch_thr_mask = precompute_threshold_indices(arch_scores, OUTPUT_THRESHOLDS)

            thr_dg = {thr: {} for thr in arch_thr_mask}

            # Phase 1 wholesale prices (scenario-invariant, no learning curve)
            _, tk_ws_arr, _, _ = iso_price_cache[iso]

            # Pre-compute year-specific price matrices with learning curves
            tk_price_cache = {}
            for year in DG_UNIQUE_YEARS:
                tk_price_cache[year] = precompute_all_prices(iso, all_combos, target_year=year)

            # Archetype scores for batched eval
            # Already float64 from _table_to_arrays — no .astype() needed
            tk_scores_f64 = arch_scores
            tk_match_frac = tk_scores_f64 / 100.0

            # Thresholds for batched eval
            _, tk_thr_desc, tk_thr_pos = prepare_threshold_metadata(
                tk_scores_f64, OUTPUT_THRESHOLDS)

            # Pre-initialize result structure
            for scenario_key, _ in all_combos:
                for thr in arch_thr_mask:
                    thr_dg[thr][scenario_key] = {}

            # Batched demand growth: threshold-year paired (same as Phase 2)
            tk_existing_base = GRID_MIX_SHARES[iso]
            tk_dg_start = time.time()
            tk_n_evals = 0

            for year in DG_UNIQUE_YEARS:
                years_out = year - 2025
                matched_thresholds = [t for t in _DG_YEAR_TO_THRESHOLDS[year]
                                      if t in arch_thr_mask]
                if not matched_thresholds:
                    continue

                # Year-specific price matrix with learning curves applied
                tk_price_matrix_yr, _, _, _ = tk_price_cache[year]

                for g_level in DEMAND_GROWTH_LEVELS:
                    g_rate = iso_rates[g_level]
                    gf = (1 + g_rate) ** years_out
                    demand_grown = demand_twh * gf
                    existing_scale = 1.0 / gf
                    demand_grown_mwh = demand_grown * 1e6

                    scaled_existing = {k: min(v * existing_scale, 100.0)
                                       for k, v in tk_existing_base.items()}

                    cm_g, const_g, tk_extras = precompute_base_year_coefficients(
                        iso, arch_arrays, demand_grown,
                        existing_override=scaled_existing,
                        uprate_cap_override=uprate_override)

                    tk_best_idxs, tk_best_vals = batch_eval_and_argmin_all(
                        cm_g, const_g, tk_price_matrix_yr, tk_scores_f64, tk_thr_desc)

                    yr_str = str(year)
                    for combo_i in range(len(all_combos)):
                        ws = float(tk_ws_arr[combo_i])
                        scenario_key = all_combos[combo_i][0]
                        for thr in matched_thresholds:
                            k = tk_thr_pos[float(thr)]
                            if tk_best_vals[combo_i, k] == np.inf:
                                continue
                            best_local = int(tk_best_idxs[combo_i, k])
                            full_idx = arch_indices[best_local]
                            tc = float(tk_best_vals[combo_i, k])
                            mf = float(tk_match_frac[best_local])
                            ec = tc / mf if mf > 0 else 0.0

                            tranche = {
                                'cf_existing_twh': float(tk_extras['cf_existing_twh'][best_local]),
                                'uprate_twh': float(tk_extras['uprate_twh'][best_local]),
                                'geo_twh': float(tk_extras['geo_twh'][best_local]),
                                'remaining_twh': float(tk_extras['remaining_twh'][best_local]),
                                'new_cf_twh': float(tk_extras['new_cf_twh'][best_local]),
                                'gas_needed_mw': float(tk_extras['gas_needed_mw'][best_local]),
                                'existing_gas_mw': float(tk_extras['existing_gas_used_mw'][best_local]),
                                'new_gas_mw': float(tk_extras['new_gas_mw'][best_local]),
                            }

                            if yr_str not in thr_dg[thr][scenario_key]:
                                thr_dg[thr][scenario_key][yr_str] = {}
                            thr_dg[thr][scenario_key][yr_str][g_level] = [
                                full_idx, round(tc, 2), round(ec, 2), round(ec - ws, 2),
                                round(gf, 6), round(demand_grown_mwh, 0), tranche]

                    tk_n_evals += 1

            if iso not in track_dg[track_name]:
                track_dg[track_name][iso] = {}
            for thr in arch_thr_mask:
                track_dg[track_name][iso][str(thr)] = thr_dg[thr]

            tk_dg_elapsed = time.time() - tk_dg_start
            print(f"  {iso:>6} {track_name}: {n_arch} archetypes, "
                  f"{len(arch_thr_mask)} thresholds, {tk_n_evals} growth evals — {tk_dg_elapsed:.0f}s")

    track_output['demand_growth'] = track_dg

    phase25_elapsed = time.time() - phase25_start
    print(f"\nPhase 2.5 complete: {phase25_elapsed:.0f}s")

    # ================================================================
    # SAVE OUTPUTS — Per-ISO parquets only
    # ================================================================
    print("\n--- Saving per-ISO parquet outputs ---")

    def _flatten_scenarios(iso, iso_data, track_name=None):
        """Yield scenario result rows for parquet output (generator)."""
        annual = iso_data.get('annual_demand_mwh', 0)
        thresholds_dict = iso_data.get('thresholds', {})
        for t_str, thr_data in thresholds_dict.items():
            for sc_key, sc in thr_data.get('scenarios', {}).items():
                row = {'iso': iso, 'threshold': float(t_str),
                       'scenario': sc_key, 'annual_demand_mwh': annual}
                if track_name:
                    row['track'] = track_name
                for k, v in sc.get('resource_mix', {}).items():
                    row[f'mix_{k}'] = v
                row['hourly_match_score'] = sc.get('hourly_match_score')
                row['battery_dispatch_pct'] = sc.get('battery_dispatch_pct')
                row['battery8_dispatch_pct'] = sc.get('battery8_dispatch_pct')
                row['ldes_dispatch_pct'] = sc.get('ldes_dispatch_pct')
                row['h2_dispatch_pct'] = sc.get('h2_dispatch_pct', 0)
                for k, v in sc.get('costs', {}).items():
                    row[f'cost_{k}'] = v
                for k, v in sc.get('tranche_costs', {}).items():
                    row[f'tranche_{k}'] = v
                for k, v in sc.get('gas_backup', {}).items():
                    row[f'gas_{k}'] = v
                yield row


    def _flatten_feasible_mixes(iso, iso_data):
        """Yield feasible mix rows for parquet output (generator)."""
        for t_str, thr_data in iso_data.get('thresholds', {}).items():
            fm = thr_data.get('feasible_mixes', {})
            if not fm:
                continue
            n = len(list(fm.values())[0])
            for i in range(n):
                row = {'iso': iso, 'threshold': float(t_str)}
                for k, vals in fm.items():
                    row[k] = vals[i]
                yield row

    def _rows_to_parquet(row_iter, path, chunk_size=250_000):
        """Write rows from an iterable to parquet in chunks (low memory).

        Uses pq.ParquetWriter to stream chunks so we never hold the full
        dataset in memory as both list-of-dicts AND columnar arrays.
        Schema is discovered from the first chunk and reused for all subsequent
        chunks, avoiding O(chunk_size × n_keys) key discovery per flush.
        """
        writer = None
        schema_keys = None
        total = 0

        def _flush(chunk):
            nonlocal writer, schema_keys
            if schema_keys is None:
                schema_keys = list(dict.fromkeys(k for r in chunk for k in r))
            arrays = {key: pa.array([r.get(key) for r in chunk])
                      for key in schema_keys}
            tbl = pa.table(arrays)
            if writer is None:
                writer = pq.ParquetWriter(str(path), tbl.schema,
                                          compression='zstd')
            writer.write_table(tbl)
            return len(chunk)

        buf = []
        for row in row_iter:
            buf.append(row)
            if len(buf) >= chunk_size:
                total += _flush(buf)
                buf = []
        if buf:
            total += _flush(buf)
        if writer is not None:
            writer.close()
        return total

    # Collect lightweight summary data BEFORE the save loop deletes results.
    # Only keeps the all-Medium scenario per ISO × threshold for the final printout.
    summary_baseline = {}
    for iso in run_isos:
        iso_res = output.get('results', {}).get(iso, {})
        mk = medium_key(iso)
        iso_summary = {}
        for thr in OUTPUT_THRESHOLDS:
            sc = iso_res.get('thresholds', {}).get(str(thr), {}).get('scenarios', {}).get(mk)
            if sc:
                iso_summary[thr] = sc
        if iso_summary:
            summary_baseline[iso] = iso_summary

    for iso in run_isos:
        if iso not in output['results']:
            continue

        iso_data = output['results'][iso]
        annual = iso_data.get('annual_demand_mwh', 0)

        # --- 1. Baseline scenarios (generator → chunked write) ---
        iso_out = output_dir / f'step3_co_{iso}.parquet'
        n = _rows_to_parquet(_flatten_scenarios(iso, iso_data), iso_out)
        if n > 0:
            print(f"  {iso_out}: {n:,} scenario rows, "
                  f"{os.path.getsize(iso_out) / 1e6:.1f} MB")
        else:
            print(f"  {iso_out}: 0 scenario rows (no baseline mixes passed filter)")
        gc.collect()

        # --- 2. Track scenarios (chain newbuild + replace generators) ---
        iso_track_data = track_output.get('results', {}).get(iso, {})
        track_iters = []
        for track_name in ['newbuild', 'replace']:
            track_thresholds = iso_track_data.get(track_name, {})
            if track_thresholds:
                track_iso_data = {
                    'annual_demand_mwh': annual,
                    'thresholds': track_thresholds,
                }
                track_iters.append(
                    _flatten_scenarios(iso, track_iso_data, track_name=track_name))
        if track_iters:
            tracks_out = output_dir / f'step3_tracks_{iso}.parquet'
            n = _rows_to_parquet(chain(*track_iters), tracks_out)
            print(f"  {tracks_out}: {n:,} track rows, "
                  f"{os.path.getsize(tracks_out) / 1e6:.1f} MB")
            gc.collect()

        # --- 3. Demand growth (per-threshold files, full resource mix) ---
        # Write one parquet per threshold. Schema matches step3_co plus
        # year, growth_level, growth_factor columns for downstream pipeline.
        iso_dg = dg_output.get('results', {}).get(iso, {})
        # For baseline track, pfs is freed after Phase 1 — use phase1_arrays instead.
        # DG archetypes index into Phase 1 filtered arrays, so this is correct.
        iso_arrays = phase1_arrays.get(iso, pfs.get(iso, {}))  # PFS arrays for resolving mix_idx

        # Pre-compute nuclear-vs-CCS decision cache ONCE per ISO (invariant across
        # thresholds, years, growth levels — only depends on iso + sensitivity toggles).
        # Reuse nuclear_arr/ccs_arr from iso_price_cache instead of re-calling get_scenario_prices.
        iso_combos = get_sensitivity_combos(iso)
        _, _, _cached_nuc_arr, _cached_ccs_arr = iso_price_cache[iso]
        iso_t3_cache = {}
        for _ci, (skey, _) in enumerate(iso_combos):
            iso_t3_cache[skey] = _cached_nuc_arr[_ci] <= _cached_ccs_arr[_ci]

        def _dg_row_gen(iso_, t_str_, thr_sc_, arrs_, t3_cache_, track_name_=None):
            """Yield DG rows with full resource mix + tranche breakdown.

            Unified generator for both baseline and track demand growth parquets.
            If track_name_ is provided, a 'track' column is added to each row.
            """
            _ex = EXISTING_CLEAN_TWH[iso_]
            for sc_key, year_data in thr_sc_.items():
                for year_str, growth_data in year_data.items():
                    for g_level, vals in growth_data.items():
                        mix_idx = vals[0]
                        tranche = vals[6] if len(vals) > 6 else {}
                        row = {
                            'iso': iso_,
                            'threshold': float(t_str_),
                            'scenario': sc_key,
                            'year': int(year_str),
                            'growth_level': g_level,
                            'growth_factor': vals[4],
                            'annual_demand_mwh': vals[5],
                        }
                        if track_name_ is not None:
                            row['track'] = track_name_

                        # Resource mix from PFS arrays
                        cf = int(arrs_['clean_firm'][mix_idx])
                        sol = int(arrs_['solar'][mix_idx])
                        wnd = int(arrs_['wind'][mix_idx])
                        hyd = int(arrs_['hydro'][mix_idx])
                        row['mix_clean_firm'] = cf
                        row['mix_solar'] = sol
                        row['mix_wind'] = wnd
                        row['mix_ccs_ccgt'] = max(0, 100 - (cf + sol + wnd + hyd))
                        row['mix_hydro'] = hyd
                        row['hourly_match_score'] = float(
                            arrs_['hourly_match_score'][mix_idx])

                        # Storage (all 4 types)
                        row['battery_dispatch_pct'] = round(float(
                            arrs_.get('battery_dispatch_pct', np.zeros(1))[mix_idx]), 4)
                        row['battery8_dispatch_pct'] = round(float(
                            arrs_.get('battery8_dispatch_pct', np.zeros(1))[mix_idx]), 4)
                        row['ldes_dispatch_pct'] = round(float(
                            arrs_.get('ldes_dispatch_pct', np.zeros(1))[mix_idx]), 4)
                        row['h2_dispatch_pct'] = round(float(
                            arrs_.get('h2_dispatch_pct', np.zeros(1))[mix_idx]), 4)

                        # Existing clean TWh available (constant)
                        row['existing_cf_twh_available'] = _ex['clean_firm']
                        row['existing_solar_twh_available'] = _ex['solar']
                        row['existing_wind_twh_available'] = _ex['wind']
                        row['existing_hydro_twh_available'] = _ex['hydro']

                        # Tranche breakdown (from precompute extras)
                        if tranche:
                            row['tranche_cf_existing_twh'] = round(tranche['cf_existing_twh'], 3)
                            row['tranche_uprate_twh'] = round(tranche['uprate_twh'], 3)
                            row['tranche_geo_twh'] = round(tranche['geo_twh'], 3)
                            row['tranche_new_cf_twh'] = round(tranche['new_cf_twh'], 3)
                            remaining_twh = tranche['remaining_twh']
                            t3_is_nuclear = t3_cache_.get(sc_key, False)
                            row['tranche_nuclear_newbuild_twh'] = round(
                                remaining_twh if t3_is_nuclear else 0.0, 3)
                            row['tranche_ccs_tranche_twh'] = round(
                                0.0 if t3_is_nuclear else remaining_twh, 3)
                            row['ra_gas_needed_mw'] = round(tranche['gas_needed_mw'])
                            row['ra_existing_gas_mw'] = round(tranche['existing_gas_mw'])
                            row['ra_new_gas_mw'] = round(tranche['new_gas_mw'])

                        # Costs
                        row['cost_total_cost'] = vals[1]
                        row['cost_effective_cost'] = vals[2]
                        row['cost_incremental'] = vals[3]
                        yield row

        if iso_dg and iso_arrays:
            dg_total = 0

            for t_str, thr_scenarios in iso_dg.items():
                t_label = f"{float(t_str):g}"
                dg_t_out = output_dir / f'step3_dg_{iso}_t{t_label}.parquet'
                n = _rows_to_parquet(
                    _dg_row_gen(iso, t_str, thr_scenarios, iso_arrays, iso_t3_cache),
                    dg_t_out)
                dg_total += n
            print(f"  step3_dg_{iso}: {dg_total:,} demand growth rows "
                  f"across {len(iso_dg)} threshold files")
            del iso_dg
            gc.collect()

        # --- 4. Track demand growth (per-threshold files, full resource mix) ---
        # Track archetypes use different PFS arrays (newbuild: hydro=0, replace: uprates=OFF)
        tdg_total = 0
        tdg_thresholds = set()

        for track_name in ['newbuild', 'replace']:
            iso_tdg = track_dg.get(track_name, {}).get(iso, {})
            ta = track_archetypes.get(track_name, {}).get(iso)
            track_pfs_arrays = ta['arrays'] if ta else iso_arrays
            for t_str, thr_scenarios in iso_tdg.items():
                tdg_thresholds.add(t_str)
                t_label = f"{float(t_str):g}"
                tdg_t_out = output_dir / f'step3_track_dg_{iso}_t{t_label}_{track_name}.parquet'
                n = _rows_to_parquet(
                    _dg_row_gen(iso, t_str, thr_scenarios, track_pfs_arrays,
                                iso_t3_cache, track_name_=track_name),
                    tdg_t_out)
                tdg_total += n
        if tdg_total > 0:
            print(f"  step3_track_dg_{iso}: {tdg_total:,} track DG rows "
                  f"across {len(tdg_thresholds)} thresholds")
        gc.collect()

        # --- 5. Feasible mixes (generator → chunked write) ---
        mix_out = output_dir / f'step3_feasible_{iso}.parquet'
        n = _rows_to_parquet(_flatten_feasible_mixes(iso, iso_data), mix_out)
        if n > 0:
            print(f"  {mix_out}: {n:,} feasible mix rows, "
                  f"{os.path.getsize(mix_out) / 1e6:.1f} MB")

        # Free this ISO's source data before processing the next one
        del output['results'][iso]
        gc.collect()

    # Save config metadata as JSON (small, one file for all ISOs)
    meta = {k: output[k] for k in output if k != 'results'}
    meta_path = output_dir / 'step3_meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  {meta_path}: {os.path.getsize(meta_path) / 1e3:.1f} KB")

    # ================================================================
    # SUMMARY
    # ================================================================
    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  STEP 3 COMPLETE in {total_elapsed:.0f}s")
    print(f"  Output: {output_dir}/")
    print(f"{'='*70}")

    # Print Medium scenario summary from pre-collected data
    print("\nAll-Medium (45Q=ON) summary — Baseline:")
    for iso in run_isos:
        iso_summary = summary_baseline.get(iso)
        if not iso_summary:
            continue
        mk = medium_key(iso)
        print(f"\n  {iso} (key={mk}):")
        for thr in OUTPUT_THRESHOLDS:
            sc = iso_summary.get(thr)
            if not sc:
                continue
            rm = sc['resource_mix']
            eff = sc['costs']['effective_cost']
            match = sc['hourly_match_score']
            print(f"    {thr:>5}%: CF={rm.get('clean_firm',0):>2} Sol={rm.get('solar',0):>2} "
                  f"Wnd={rm.get('wind',0):>2} CCS={rm.get('ccs_ccgt',0):>2} Hyd={rm.get('hydro',0):>2} | "
                  f"eff=${eff:.1f}/MWh match={match}%")

    # Track summaries (only when tracks were computed)
    if track != 'baseline':
        for track_name in ['newbuild', 'replace']:
            label = 'New-Build (no hydro, +uprates)' if track_name == 'newbuild' else 'Replace (with hydro, no uprates)'
            print(f"\nAll-Medium (45Q=ON) summary — {label}:")
            for iso in run_isos:
                iso_track = track_output.get('results', {}).get(iso, {}).get(track_name, {})
                if not iso_track:
                    print(f"\n  {iso}: no data")
                    continue
                mk = medium_key(iso)
                print(f"\n  {iso} (key={mk}):")
                for thr in OUTPUT_THRESHOLDS:
                    t_str = str(thr)
                    sc = iso_track.get(t_str, {}).get('scenarios', {}).get(mk)
                    if not sc:
                        continue
                    rm = sc['resource_mix']
                    eff = sc['costs']['effective_cost']
                    match = sc['hourly_match_score']
                    print(f"    {thr:>5}%: CF={rm.get('clean_firm',0):>2} Sol={rm.get('solar',0):>2} "
                          f"Wnd={rm.get('wind',0):>2} CCS={rm.get('ccs_ccgt',0):>2} Hyd={rm.get('hydro',0):>2} | "
                          f"eff=${eff:.1f}/MWh match={match}%")


if __name__ == '__main__':
    main()
