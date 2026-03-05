#!/usr/bin/env python3
"""
Step 5D: Sequential Consequential Optimizer
============================================
Mines the Step 1 PFS parquets to build optimal clean energy portfolios
under $/tCO2 optimization — completely independent of hourly CFE matching.

Same grid physics (generation profiles, storage dispatch, merit-order fossil
retirement) as the hourly matching pipeline. Different optimization objective:
minimize $/tCO2 abated at each sequential threshold step.

Uses Wright's Law learning curves from pipeline_config (same as step 3b)
mapped to SBTi target years. Floor ratcheting ensures committed resources
persist across thresholds.

5 cost scenarios:
  1. all_low      — best-case costs across the board
  2. all_medium   — central estimate
  3. all_high     — worst-case costs
  4. cheap_firm   — high VRE + low firm (tests: what if firm achieves NOAK fast?)
  5. cheap_vre    — low VRE + high firm (DEFAULT: realistic consequential case)

Reads:  data/step1-pfs-parquets/, data/step1d-storage-parquets/, pipeline_config
Writes: data/step5-post-processing/consequential_queue.json,
        dashboard/js/consequential-queue-data.js
"""

import json
import os
import sys
import time
import argparse
import numpy as np

# Add project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'scripts'))

try:
    import pyarrow.parquet as pq
except ImportError:
    print("ERROR: pyarrow required. pip install pyarrow")
    sys.exit(1)

from pipeline_config import (
    ISOS, REGIONAL_DEMAND_TWH, THRESHOLDS, ACTIVE_THRESHOLDS,
    GRID_MIX_SHARES, OFFSHORE_ISOS,
    LCOE_TABLES, TX_TABLES, get_tx, UPRATE_LCOE, WHOLESALE_PRICES,
    NUCLEAR_NEWBUILD_LCOE, GEOTHERMAL_LCOE, GEO_CAP_TWH,
    CCS_LCOE_45Q_ON, CCS_LCOE_45Q_OFF, CCS_CAP_TWH,
    NEISO_CCS_GAS_ADDER,
    FOAK_NUCLEAR_NEWBUILD, FOAK_CCS_45Q_ON, FOAK_CCS_45Q_OFF,
    FOAK_GEOTHERMAL, FOAK_OFFSHORE_WIND, FOAK_LDES, FOAK_H2,
    NOAK_BATTERY, NOAK_BATTERY8, NOAK_OFFSHORE_WIND,
    LEARNING_PARAMS, year_adjusted_cost,
    EXISTING_NUCLEAR_GW, UPRATE_CAP_TWH, UPRATE_FRACTION, UPRATE_CF,
    STORAGE_REVENUE_CREDITS,
    THRESHOLD_TARGET_YEARS,
    GEOTHERMAL_CAP_TWH,
    DEMAND_GROWTH_RATES,
)
from dispatch_utils import (
    COAL_CAP_TWH, OIL_CAP_TWH,
)

# ============================================================================
# CONSTANTS
# ============================================================================

PFS_DIR = os.path.join(BASE_DIR, 'data', 'step1-pfs-parquets')
STORAGE_DIR = os.path.join(BASE_DIR, 'data', 'step1d-storage-parquets')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'step5-post-processing')
JS_DIR = os.path.join(BASE_DIR, 'dashboard', 'js')

# Columns in PFS parquets
GEN_COLS = ['clean_firm', 'solar', 'wind', 'hydro', 'offshore_wind']
GEO_COL = 'geothermal'
STORAGE_COLS = ['battery_dispatch_pct', 'battery8_dispatch_pct',
                'ldes_dispatch_pct', 'h2_dispatch_pct']
SCORE_COL = 'hourly_match_score'

# Consequential queue operates on 50%+ thresholds
CONSQ_THRESHOLDS = sorted([t for t in ACTIVE_THRESHOLDS if t >= 50])

# SBTi year map
SBTI_YEAR_MAP = {float(k): v for k, v in THRESHOLD_TARGET_YEARS.items()}

# eGRID emission rates (tCO2/MWh) — loaded from file
EGRID_PATH = os.path.join(BASE_DIR, 'data', 'egrid_emission_rates.json')

# ============================================================================
# SCENARIO DEFINITIONS
# ============================================================================

SCENARIOS = {
    'all_low': {
        'vre_level': 'Low', 'firm_level': 'L', 'storage_level': 'Low',
        'tx_level': 'Low', '45q': 'on',
        'description': 'Best-case costs across the board',
    },
    'all_medium': {
        'vre_level': 'Medium', 'firm_level': 'M', 'storage_level': 'Medium',
        'tx_level': 'Medium', '45q': 'on',
        'description': 'Central estimate',
    },
    'all_high': {
        'vre_level': 'High', 'firm_level': 'H', 'storage_level': 'High',
        'tx_level': 'High', '45q': 'on',
        'description': 'Worst-case costs',
    },
    'cheap_firm': {
        'vre_level': 'High', 'firm_level': 'L', 'storage_level': 'High',
        'tx_level': 'Medium', '45q': 'on',
        'description': 'High VRE + low firm (what if firm achieves NOAK fast?)',
    },
    'cheap_vre': {
        'vre_level': 'Low', 'firm_level': 'H', 'storage_level': 'Low',
        'tx_level': 'Medium', '45q': 'on',
        'description': 'Low VRE + high firm (DEFAULT: realistic consequential case)',
    },
}
DEFAULT_SCENARIO = 'cheap_vre'
SCENARIO_ORDER = ['all_low', 'all_medium', 'all_high', 'cheap_firm', 'cheap_vre']


# ============================================================================
# DATA LOADING
# ============================================================================

def load_egrid():
    """Load eGRID emission rates and convert to tCO2/MWh."""
    with open(EGRID_PATH) as f:
        raw = json.load(f)
    rates = {}
    for iso in ISOS:
        r = raw.get(iso, {})
        rates[iso] = {
            'coal': r.get('coal_co2_lb_per_mwh', 0) / 2204.62,
            'oil': r.get('oil_co2_lb_per_mwh', 0) / 2204.62,
            'gas': r.get('gas_co2_lb_per_mwh', 0) / 2204.62,
        }
    return rates


def load_pfs_pool(iso):
    """Load ALL PFS mixes for an ISO (raw + storage-refined) into numpy arrays.

    Returns dict with numpy arrays for each column + metadata.
    """
    all_rows = []

    # Column order: clean_firm, solar, wind, hydro, offshore_wind, geothermal,
    #               bat4_pct, bat8_pct, ldes_pct, h2_pct, match_score
    n_cols = 11

    # Load raw PFS parquets (step1)
    for fname in sorted(os.listdir(PFS_DIR)):
        if not fname.startswith(f'{iso}_t') or not fname.endswith('_raw_pfs.parquet'):
            continue
        path = os.path.join(PFS_DIR, fname)
        table = pq.read_table(path)
        cols = table.column_names
        n = table.num_rows
        if n == 0:
            continue

        row = np.zeros((n, n_cols), dtype=np.float64)
        for i, col in enumerate(GEN_COLS):
            if col in cols:
                row[:, i] = table.column(col).to_numpy().astype(np.float64)
        if GEO_COL in cols:
            row[:, 5] = table.column(GEO_COL).to_numpy().astype(np.float64)
        for j, col in enumerate(STORAGE_COLS):
            if col in cols:
                row[:, 6 + j] = table.column(col).to_numpy().astype(np.float64)
        if SCORE_COL in cols:
            row[:, 10] = table.column(SCORE_COL).to_numpy().astype(np.float64)
        elif 'score' in cols:
            row[:, 10] = table.column('score').to_numpy().astype(np.float64)
        all_rows.append(row)

    # Load storage-refined parquets (step1d)
    if os.path.isdir(STORAGE_DIR):
        for fname in sorted(os.listdir(STORAGE_DIR)):
            if not fname.startswith(f'{iso}_t') or '_storage' not in fname:
                continue
            if not fname.endswith('.parquet'):
                continue
            path = os.path.join(STORAGE_DIR, fname)
            table = pq.read_table(path)
            cols = table.column_names
            n = table.num_rows
            if n == 0:
                continue

            row = np.zeros((n, n_cols), dtype=np.float64)
            for i, col in enumerate(GEN_COLS):
                if col in cols:
                    row[:, i] = table.column(col).to_numpy().astype(np.float64)
            if GEO_COL in cols:
                row[:, 5] = table.column(GEO_COL).to_numpy().astype(np.float64)
            for j, col in enumerate(STORAGE_COLS):
                if col in cols:
                    row[:, 6 + j] = table.column(col).to_numpy().astype(np.float64)
            if SCORE_COL in cols:
                row[:, 10] = table.column(SCORE_COL).to_numpy().astype(np.float64)
            all_rows.append(row)

    if not all_rows:
        return None

    pool = np.concatenate(all_rows, axis=0)

    # Apply hydro cap at 2025 baseline: hydro is NOT viable as a new resource.
    # Only allow mixes where hydro <= existing fleet share. PFS explores
    # higher hydro for physics experimentation, but consequential can't procure it.
    # Note: hydro cap shrinks further with demand growth (applied per-threshold
    # in optimize_iso), but this initial filter removes obviously invalid mixes.
    import math
    existing_hydro = GRID_MIX_SHARES.get(iso, {}).get('hydro', 0)
    hydro_cap = math.ceil(existing_hydro)  # PFS uses integer %, so ceil of existing
    pre_cap = pool.shape[0]
    pool = pool[pool[:, I_HYD] <= hydro_cap + 0.01]  # tight tolerance
    post_cap = pool.shape[0]
    if pre_cap != post_cap:
        print(f"  {iso}: hydro cap {hydro_cap}% removed {pre_cap - post_cap:,} mixes")

    # Deduplicate: round to 1 decimal, unique rows only
    pool_rounded = np.round(pool, 1)
    _, unique_idx = np.unique(pool_rounded, axis=0, return_index=True)
    pool = pool[np.sort(unique_idx)]

    print(f"  {iso}: loaded {pool.shape[0]:,} unique mixes "
          f"(score range: {pool[:, 10].min():.1f}-{pool[:, 10].max():.1f}%)")
    return pool


# Column index shortcuts
I_CF, I_SOL, I_WND, I_HYD, I_OSW, I_GEO = 0, 1, 2, 3, 4, 5
I_BAT4, I_BAT8, I_LDES, I_H2 = 6, 7, 8, 9
I_SCORE = 10


# ============================================================================
# CO2 COMPUTATION (VECTORIZED MERIT-ORDER)
# ============================================================================

def merit_order_co2(match_scores, demand_twh, coal_cap, oil_cap,
                    coal_rate, oil_rate, gas_rate):
    """Compute remaining CO2 (MT) from effective clean percentage.

    Merit order: coal dispatched first, then oil, then gas.
    Higher clean % → less fossil → less CO2.
    """
    fossil_twh = np.maximum(0, (1 - match_scores / 100.0) * demand_twh)
    coal_twh = np.minimum(coal_cap, fossil_twh)
    remaining = np.maximum(0, fossil_twh - coal_twh)
    oil_twh = np.minimum(oil_cap, remaining)
    gas_twh = np.maximum(0, remaining - oil_twh)
    # TWh × tCO2/MWh = MT (since 1 TWh × 1 tCO2/MWh = 1e6 MWh × tCO2/MWh = 1 MT)
    return coal_twh * coal_rate + oil_twh * oil_rate + gas_twh * gas_rate


def compute_fossil_breakdown(match_score, demand_twh, coal_cap, oil_cap):
    """Compute fossil TWh breakdown for a single mix."""
    fossil_twh = max(0, (1 - match_score / 100.0) * demand_twh)
    coal = min(coal_cap, fossil_twh)
    oil = min(oil_cap, max(0, fossil_twh - coal))
    gas = max(0, fossil_twh - coal - oil)
    return {'coal_twh': round(coal, 2), 'oil_twh': round(oil, 2),
            'gas_twh': round(gas, 2)}


# ============================================================================
# COST COMPUTATION (VECTORIZED, YEAR-ADJUSTED)
# ============================================================================

def build_lcoe_at_year(iso, scenario, year):
    """Build LCOE dict for all resources at a given year using Wright's Law.

    Returns dict: resource → $/MWh (or annualized coeff cost for storage).
    Uses the SAME learning curves as step 3b.
    """
    sc = SCENARIOS[scenario]
    vre_lev = sc['vre_level']
    firm_lev = sc['firm_level']
    stor_lev = sc['storage_level']
    tx_lev = sc['tx_level']
    use_45q = sc['45q'] == 'on'

    lcoe = {}

    # --- Static LCOE (mature tech) ---
    lcoe['solar'] = LCOE_TABLES['solar'][vre_lev][iso] + get_tx('solar', tx_lev, iso)
    lcoe['wind'] = LCOE_TABLES['wind'][vre_lev][iso] + get_tx('wind', tx_lev, iso)
    lcoe['hydro'] = 0  # existing only

    # --- Offshore wind (learning curve) ---
    if iso in OFFSHORE_ISOS:
        foak = FOAK_OFFSHORE_WIND.get(iso, 0)
        noak = NOAK_OFFSHORE_WIND.get(vre_lev, {}).get(iso, 0)
        lp_key = 'offshore_wind_float' if iso == 'CAISO' else 'offshore_wind_fixed'
        lp = LEARNING_PARAMS[lp_key][firm_lev]  # use firm_lev for learning speed
        if foak > 0 and noak > 0:
            lcoe['offshore_wind'] = year_adjusted_cost(foak, noak, year, *lp) + get_tx('offshore_wind', tx_lev, iso)
        else:
            lcoe['offshore_wind'] = 0
    else:
        lcoe['offshore_wind'] = 0

    # --- Nuclear uprate (no learning curve — existing fleet) ---
    lcoe['uprate'] = UPRATE_LCOE[firm_lev]

    # --- Geothermal (CAISO only, learning curve) ---
    if iso == 'CAISO':
        lp = LEARNING_PARAMS['geo'][firm_lev]
        lcoe['geothermal'] = year_adjusted_cost(
            FOAK_GEOTHERMAL, GEOTHERMAL_LCOE[firm_lev],
            year, *lp) + get_tx('clean_firm', tx_lev, iso)
    else:
        lcoe['geothermal'] = 0

    # --- CCS-CCGT (learning curve) ---
    ccs_table = CCS_LCOE_45Q_ON if use_45q else CCS_LCOE_45Q_OFF
    foak_table = FOAK_CCS_45Q_ON if use_45q else FOAK_CCS_45Q_OFF
    lp = LEARNING_PARAMS['ccs'][firm_lev]
    ccs_noak = ccs_table[firm_lev][iso]
    ccs_foak = foak_table[iso]
    lcoe['ccs'] = year_adjusted_cost(ccs_foak, ccs_noak, year, *lp) + get_tx('ccs_ccgt', tx_lev, iso)
    if iso == 'NEISO':
        lcoe['ccs'] += NEISO_CCS_GAS_ADDER

    # --- Nuclear newbuild (learning curve) ---
    lp = LEARNING_PARAMS['nuclear'][firm_lev]
    lcoe['nuclear'] = year_adjusted_cost(
        FOAK_NUCLEAR_NEWBUILD[iso], NUCLEAR_NEWBUILD_LCOE[firm_lev][iso],
        year, *lp) + get_tx('clean_firm', tx_lev, iso)

    # --- Storage (learning curves, coefficient-model pricing) ---
    # Battery 4hr
    lp = LEARNING_PARAMS['bat4'][firm_lev]
    bat4_raw = year_adjusted_cost(
        LCOE_TABLES['battery'][stor_lev][iso],
        NOAK_BATTERY[stor_lev][iso], year, *lp)
    lcoe['battery4'] = max(0, bat4_raw - STORAGE_REVENUE_CREDITS['battery'][iso])

    # Battery 8hr
    lp = LEARNING_PARAMS['bat8'][firm_lev]
    bat8_raw = year_adjusted_cost(
        LCOE_TABLES['battery8'][stor_lev][iso],
        NOAK_BATTERY8[stor_lev][iso], year, *lp)
    lcoe['battery8'] = max(0, bat8_raw - STORAGE_REVENUE_CREDITS['battery8'][iso])

    # LDES
    lp = LEARNING_PARAMS['ldes'][firm_lev]
    ldes_raw = year_adjusted_cost(
        FOAK_LDES[iso], LCOE_TABLES['ldes'][stor_lev][iso], year, *lp)
    lcoe['ldes'] = max(0, ldes_raw - STORAGE_REVENUE_CREDITS['ldes'][iso])

    # Green H2
    lp = LEARNING_PARAMS['h2'][firm_lev]
    h2_raw = year_adjusted_cost(
        FOAK_H2[iso], LCOE_TABLES['h2'][stor_lev][iso], year, *lp)
    lcoe['h2'] = max(0, h2_raw - STORAGE_REVENUE_CREDITS['h2'][iso])

    return lcoe


def vectorized_cost(pool, lcoe, iso, demand_twh):
    """Compute total annualized cost ($B) for each mix in the pool.

    Replicates step 3a's cost function:
    - Generation: (new_pct / 100) × LCOE × demand_TWh / 1e3 → $B
    - Existing resources: $0 (sunk fleet)
    - Clean firm: tranche pricing (uprate → geothermal → CCS → nuclear newbuild)
    - Storage: coeff × price / 100
    """
    N = pool.shape[0]
    existing = GRID_MIX_SHARES.get(iso, {})

    # Solar: only new-build costs
    sol_existing = existing.get('solar', 0)
    sol_new = np.maximum(0, pool[:, I_SOL] - sol_existing)
    cost = sol_new / 100.0 * lcoe['solar'] * demand_twh / 1e3

    # Wind: only new-build
    wnd_existing = existing.get('wind', 0)
    wnd_new = np.maximum(0, pool[:, I_WND] - wnd_existing)
    cost += wnd_new / 100.0 * lcoe['wind'] * demand_twh / 1e3

    # Offshore wind: all new-build
    if iso in OFFSHORE_ISOS and lcoe['offshore_wind'] > 0:
        cost += pool[:, I_OSW] / 100.0 * lcoe['offshore_wind'] * demand_twh / 1e3

    # Hydro: $0

    # Clean firm: tranche pricing (matches step 3a exactly)
    cf_existing = existing.get('clean_firm', 0)
    cf_new_pct = np.maximum(0, pool[:, I_CF] - cf_existing)
    cf_new_twh = cf_new_pct / 100.0 * demand_twh

    # Tranche 1: Nuclear uprate
    uprate_cap = UPRATE_CAP_TWH[iso]
    uprate_twh = np.minimum(cf_new_twh, uprate_cap)
    cf_cost = uprate_twh * lcoe['uprate']
    remaining = np.maximum(0, cf_new_twh - uprate_twh)

    # Tranche 2: Geothermal (CAISO only)
    if iso == 'CAISO' and lcoe['geothermal'] > 0:
        geo_twh = np.minimum(remaining, GEO_CAP_TWH)
        cf_cost += geo_twh * lcoe['geothermal']
        remaining = np.maximum(0, remaining - geo_twh)

    # Tranche 3: Cheapest of CCS vs nuclear newbuild
    ccs_cap = CCS_CAP_TWH.get(iso, 0)
    if ccs_cap > 0 and lcoe['ccs'] < lcoe['nuclear']:
        # CCS cheaper: fill CCS first, nuclear overflow
        ccs_twh = np.minimum(remaining, ccs_cap)
        nuc_twh = np.maximum(0, remaining - ccs_twh)
        cf_cost += ccs_twh * lcoe['ccs'] + nuc_twh * lcoe['nuclear']
    else:
        # Nuclear cheaper (or no CCS cap)
        cf_cost += remaining * lcoe['nuclear']

    cost += cf_cost / 1e3  # TWh × $/MWh / 1e3 = $B

    # Geothermal (separate column for CAISO — in addition to clean_firm tranche)
    # Note: geothermal in the PFS is part of the clean_firm tranche, not a separate column
    # unless the PFS explicitly has it. The `pool[:, I_GEO]` is the geo-specific allocation.
    # For non-CAISO, this is 0.

    # Storage: coefficient × price / 100 × demand_TWh / 1e3
    cost += pool[:, I_BAT4] * lcoe['battery4'] / 100.0 * demand_twh / 1e3
    cost += pool[:, I_BAT8] * lcoe['battery8'] / 100.0 * demand_twh / 1e3
    cost += pool[:, I_LDES] * lcoe['ldes'] / 100.0 * demand_twh / 1e3
    cost += pool[:, I_H2] * lcoe['h2'] / 100.0 * demand_twh / 1e3

    return cost


# ============================================================================
# FLOOR RATCHET & FILTERING
# ============================================================================

def ratchet_mask(pool, floor):
    """Boolean mask: True for mixes that meet floor ratchet on all resources."""
    mask = np.ones(pool.shape[0], dtype=bool)
    mask &= pool[:, I_CF] >= floor[0] - 0.5
    mask &= pool[:, I_SOL] >= floor[1] - 0.5
    mask &= pool[:, I_WND] >= floor[2] - 0.5
    mask &= pool[:, I_HYD] >= floor[3] - 0.5
    mask &= pool[:, I_OSW] >= floor[4] - 0.5
    mask &= pool[:, I_GEO] >= floor[5] - 0.5
    mask &= pool[:, I_BAT4] >= floor[6] - 0.001
    mask &= pool[:, I_BAT8] >= floor[7] - 0.001
    mask &= pool[:, I_LDES] >= floor[8] - 0.001
    mask &= pool[:, I_H2] >= floor[9] - 0.001
    return mask


def compute_stranding_cost(pool_resources, floor, stranded_assets, demand_twh):
    """Compute stranding cost for candidates that fall below the floor.

    When a candidate has resources below the floor (stranding), it incurs
    a cost for the stranded assets priced at the LCOE when they were built.

    Args:
        pool_resources: (N, 10) array of candidate resource levels
        floor: (10,) array of current floor resource levels
        stranded_assets: dict {resource_idx: (stranded_pct, lcoe_per_mwh, build_year)}
        demand_twh: demand at current threshold year

    Returns: (N,) array of stranding costs in $B
    """
    N = pool_resources.shape[0]
    strand_cost = np.zeros(N, dtype=np.float64)

    for r_idx, (s_pct, s_lcoe, _) in stranded_assets.items():
        # How much of this resource does the candidate strand?
        # Stranded = floor[r] - candidate[r], clipped to [0, s_pct]
        new_strand = np.maximum(0, floor[r_idx] - pool_resources[:, r_idx])
        # Cost: stranded % × LCOE at build year × demand / 1e3
        strand_cost += np.minimum(new_strand, s_pct) / 100.0 * s_lcoe * demand_twh / 1e3

    return strand_cost


def resource_lcoe_vector(lcoe, iso):
    """Map LCOE dict to per-resource-dimension vector for stranding cost.

    Returns: (10,) array of $/MWh for each resource dimension.
    For clean_firm, uses the highest tranche price (conservative for stranding).
    """
    vec = np.zeros(10, dtype=np.float64)
    vec[I_CF] = lcoe.get('nuclear', lcoe.get('ccs', 100))  # highest tranche
    vec[I_SOL] = lcoe.get('solar', 0)
    vec[I_WND] = lcoe.get('wind', 0)
    vec[I_HYD] = 0  # hydro is existing, no stranding cost
    vec[I_OSW] = lcoe.get('offshore_wind', 0)
    vec[I_GEO] = lcoe.get('geothermal', 0)
    vec[I_BAT4] = lcoe.get('battery4', 0)
    vec[I_BAT8] = lcoe.get('battery8', 0)
    vec[I_LDES] = lcoe.get('ldes', 0)
    vec[I_H2] = lcoe.get('h2', 0)
    return vec


# ============================================================================
# MAIN OPTIMIZER
# ============================================================================

def optimize_iso(iso, pool, egrid_rates, scenarios=None):
    """Run the sequential consequential optimizer for one ISO.

    Each threshold is a FRESH, independent MAC calculation:
    MAC = (new_resource_cost + stranding_cost) × 1e3 / CO2_abated_vs_existing

    Floor ratcheting constrains resource deployment (can't un-build).
    When no PFS mix passes the ratchet, we relax it and add stranding costs.
    """
    if scenarios is None:
        scenarios = SCENARIO_ORDER

    demand_twh_2025 = REGIONAL_DEMAND_TWH[iso]
    existing = GRID_MIX_SHARES.get(iso, {})
    existing_clean_pct = sum(existing.values())

    existing_hydro_twh = existing.get('hydro', 0) / 100.0 * demand_twh_2025
    growth_rate = DEMAND_GROWTH_RATES.get(iso, {}).get('Medium', 0.02)

    coal_rate = egrid_rates[iso]['coal']
    oil_rate = egrid_rates[iso]['oil']
    gas_rate = egrid_rates[iso]['gas']
    coal_cap = COAL_CAP_TWH.get(iso, 0)
    oil_cap = OIL_CAP_TWH.get(iso, 0)

    pool_co2 = merit_order_co2(
        pool[:, I_SCORE], demand_twh_2025, coal_cap, oil_cap,
        coal_rate, oil_rate, gas_rate)
    baseline_co2 = merit_order_co2(
        np.array([existing_clean_pct]), demand_twh_2025, coal_cap, oil_cap,
        coal_rate, oil_rate, gas_rate)[0]

    results = {}

    for scenario in scenarios:
        print(f"    {scenario}...", end='', flush=True)
        t0 = time.time()

        floor = np.array([
            existing.get('clean_firm', 0),
            existing.get('solar', 0),
            existing.get('wind', 0),
            existing.get('hydro', 0),
            existing.get('offshore_wind', 0),
            0, 0, 0, 0, 0,
        ], dtype=np.float64)

        trajectory = []
        prev_resources = floor.copy()
        stranded_assets = {}  # {resource_idx: (stranded_pct, lcoe_$/MWh, build_year)}

        for t_idx, threshold in enumerate(CONSQ_THRESHOLDS):
            year = SBTI_YEAR_MAP.get(threshold, 2050)
            years_out = max(0, year - 2025)
            demand_twh = demand_twh_2025 * (1 + growth_rate) ** years_out
            hydro_cap_pct = (existing_hydro_twh / demand_twh) * 100.0

            lcoe = build_lcoe_at_year(iso, scenario, year)
            pool_cost = vectorized_cost(pool, lcoe, iso, demand_twh)
            lcoe_vec = resource_lcoe_vector(lcoe, iso)

            hydro_mask = pool[:, I_HYD] <= hydro_cap_pct + 0.5
            score_mask = pool[:, I_SCORE] >= threshold
            next_t = CONSQ_THRESHOLDS[t_idx + 1] if t_idx + 1 < len(CONSQ_THRESHOLDS) else 101.0

            # Stage 1: Full ratchet + progressive score ceiling
            ratchet_ok = ratchet_mask(pool, floor)
            candidate_idx = np.array([], dtype=int)
            used_relaxed = False

            for ceil_margin in [1.0, 2.0, 3.0, 5.0, next_t - threshold + 1.0]:
                score_ceil = min(threshold + ceil_margin, next_t + 1.0)
                if t_idx + 1 >= len(CONSQ_THRESHOLDS):
                    score_ceil = 101.0
                mask = ratchet_ok & hydro_mask & score_mask & (pool[:, I_SCORE] < score_ceil)
                candidate_idx = np.where(mask)[0]
                if len(candidate_idx) > 0:
                    break

            # Stage 2: Relaxed ratchet (allow stranding) + progressive ceiling
            if len(candidate_idx) == 0:
                used_relaxed = True
                relaxed_floor = np.maximum(0, floor - 5.0)
                relaxed_floor[6:] = np.maximum(0, floor[6:] - 0.05)
                ratchet_relaxed = ratchet_mask(pool, relaxed_floor)
                for ceil_margin in [1.0, 2.0, 3.0, 5.0, next_t - threshold + 1.0]:
                    score_ceil = min(threshold + ceil_margin, next_t + 1.0)
                    if t_idx + 1 >= len(CONSQ_THRESHOLDS):
                        score_ceil = 101.0
                    mask = ratchet_relaxed & hydro_mask & score_mask & (pool[:, I_SCORE] < score_ceil)
                    candidate_idx = np.where(mask)[0]
                    if len(candidate_idx) > 0:
                        break

            # Stage 3: Very relaxed ratchet + no ceiling
            if len(candidate_idx) == 0:
                used_relaxed = True
                very_relaxed = np.maximum(0, floor - 15.0)
                very_relaxed[6:] = np.maximum(0, floor[6:] - 0.1)
                mask = ratchet_mask(pool, very_relaxed) & hydro_mask & score_mask
                candidate_idx = np.where(mask)[0]

            if len(candidate_idx) == 0:
                if trajectory:
                    entry = dict(trajectory[-1])
                    entry['threshold'] = threshold
                    entry['year'] = year
                    entry['note'] = 'no_candidates'
                    trajectory.append(entry)
                continue

            # ============================================================
            # FRESH MAC with stranding costs
            # MAC = (new_resource_cost + stranding_cost) × 1e3 / CO2_abated
            # ============================================================
            cand_cost = pool_cost[candidate_idx]
            cand_co2_abated = baseline_co2 - pool_co2[candidate_idx]

            # Compute stranding cost for each candidate
            if used_relaxed and stranded_assets:
                strand_cost = compute_stranding_cost(
                    pool[candidate_idx, :10], floor, stranded_assets, demand_twh)
            else:
                strand_cost = np.zeros(len(candidate_idx))

            # Also compute NEW stranding from this step
            new_strand_cost = np.zeros(len(candidate_idx))
            for r in range(10):
                shortfall = np.maximum(0, floor[r] - pool[candidate_idx, r])
                new_strand_cost += shortfall / 100.0 * lcoe_vec[r] * demand_twh / 1e3

            effective_cost = cand_cost + strand_cost + new_strand_cost

            valid = cand_co2_abated > 0.01
            if not np.any(valid):
                best_local = np.argmin(effective_cost)
                best_idx = candidate_idx[best_local]
            else:
                mac_vals = np.full(len(candidate_idx), np.inf)
                mac_vals[valid] = effective_cost[valid] * 1e3 / cand_co2_abated[valid]
                best_local = np.argmin(mac_vals)
                best_idx = candidate_idx[best_local]

            winner = pool[best_idx]
            winner_cost = pool_cost[best_idx]
            winner_co2 = pool_co2[best_idx]
            winner_co2_abated = baseline_co2 - winner_co2
            winner_strand = strand_cost[best_local] + new_strand_cost[best_local]
            winner_effective = effective_cost[best_local]

            mac_val = (
                winner_effective * 1e3 / winner_co2_abated
                if winner_co2_abated > 0.001 else float('inf')
            )

            # Update stranding tracker
            floor_before = floor.copy()
            for r in range(10):
                if winner[r] < floor_before[r] - 0.5:
                    s_pct = floor_before[r] - winner[r]
                    stranded_assets[r] = (s_pct, lcoe_vec[r], year)
                elif r in stranded_assets and winner[r] >= floor_before[r] - 0.5:
                    del stranded_assets[r]

            # Update floor ratchet
            floor = np.maximum(floor, winner[:10])

            delta_resources = winner[:10] - prev_resources

            fossil = compute_fossil_breakdown(
                winner[I_SCORE], demand_twh, coal_cap, oil_cap)
            baseline_fossil = compute_fossil_breakdown(
                existing_clean_pct, demand_twh, coal_cap, oil_cap)
            coal_delta = baseline_fossil['coal_twh'] - fossil['coal_twh']
            oil_delta = baseline_fossil['oil_twh'] - fossil['oil_twh']
            gas_delta = baseline_fossil['gas_twh'] - fossil['gas_twh']
            primary_fuel = 'coal' if coal_delta >= max(oil_delta, gas_delta) else (
                'oil' if oil_delta >= gas_delta else 'gas')

            entry = {
                'threshold': threshold,
                'year': year,
                'resources_pct': {
                    'clean_firm': round(float(winner[I_CF]), 1),
                    'solar': round(float(winner[I_SOL]), 1),
                    'wind': round(float(winner[I_WND]), 1),
                    'hydro': round(float(winner[I_HYD]), 1),
                    'offshore_wind': round(float(winner[I_OSW]), 1),
                    'geothermal': round(float(winner[I_GEO]), 1),
                },
                'storage_pct': {
                    'battery4': round(float(winner[I_BAT4]), 4),
                    'battery8': round(float(winner[I_BAT8]), 4),
                    'ldes': round(float(winner[I_LDES]), 4),
                    'h2': round(float(winner[I_H2]), 4),
                },
                'year_adjusted_lcoe': {k: round(v, 1) for k, v in lcoe.items()},
                'total_cost_bn': round(float(winner_cost), 3),
                'stranded_cost_bn': round(float(winner_strand), 3),
                'effective_cost_bn': round(float(winner_effective), 3),
                'co2_remaining_mt': round(float(winner_co2), 2),
                'co2_abated_mt': round(float(winner_co2_abated), 2),
                'mac': round(float(min(mac_val, 9999)), 1),
                'fossil_dispatch': fossil,
                'primary_fuel_displaced': primary_fuel,
                'effective_clean_pct': round(float(winner[I_SCORE]), 2),
                'used_relaxed_ratchet': used_relaxed,
                'new_resources_this_zone': {
                    'clean_firm': round(float(delta_resources[0]), 1),
                    'solar': round(float(delta_resources[1]), 1),
                    'wind': round(float(delta_resources[2]), 1),
                    'hydro': round(float(delta_resources[3]), 1),
                    'offshore_wind': round(float(delta_resources[4]), 1),
                    'geothermal': round(float(delta_resources[5]), 1),
                    'battery4': round(float(delta_resources[6]), 4),
                    'battery8': round(float(delta_resources[7]), 4),
                    'ldes': round(float(delta_resources[8]), 4),
                    'h2': round(float(delta_resources[9]), 4),
                },
            }
            trajectory.append(entry)
            prev_resources = winner[:10].copy()

        elapsed = time.time() - t0
        print(f" {len(trajectory)} thresholds, {elapsed:.1f}s")
        results[scenario] = trajectory

    return results


def build_deployment_queue(all_results, egrid_rates, scenario='all_medium'):
    """Build cross-ISO deployment queue sorted by stepwise MAC.

    Takes the per-threshold entries from each ISO and interleaves them
    using a greedy algorithm that always picks the lowest MAC next step.
    """
    queue = []
    for iso in ISOS:
        traj = all_results.get(iso, {}).get(scenario, [])
        for i, entry in enumerate(traj):
            if entry.get('note') == 'no_candidates':
                continue
            prev_threshold = traj[i - 1]['threshold'] if i > 0 else GRID_MIX_SHARES.get(iso, {})
            zone_label = f"{entry['threshold'] - 5 if entry['threshold'] > 50 else int(sum(GRID_MIX_SHARES.get(iso, {}).values()))}→{entry['threshold']}%"
            if i > 0:
                prev_t = traj[i - 1]['threshold']
                zone_label = f"{prev_t}→{entry['threshold']}%"
            else:
                existing_pct = int(sum(GRID_MIX_SHARES.get(iso, {}).values()))
                zone_label = f"{existing_pct}→{entry['threshold']}%"

            queue.append({
                'iso': iso,
                'zone_label': zone_label,
                'threshold': entry['threshold'],
                'year': entry['year'],
                'marginal_mac': entry['mac'],
                'co2_displaced_mt': round(
                    (traj[i - 1]['co2_remaining_mt'] if i > 0 else
                     merit_order_co2(
                         np.array([sum(GRID_MIX_SHARES.get(iso, {}).values())]),
                         REGIONAL_DEMAND_TWH[iso],
                         COAL_CAP_TWH.get(iso, 0), OIL_CAP_TWH.get(iso, 0),
                         egrid_rates[iso]['coal'], egrid_rates[iso]['oil'],
                         egrid_rates[iso]['gas'])[0])
                    - entry['co2_remaining_mt'], 2),
                'total_cost_bn': entry['total_cost_bn'],
                'resources_pct': entry['resources_pct'],
                'storage_pct': entry['storage_pct'],
                'new_resources': entry['new_resources_this_zone'],
                'primary_fuel': entry['primary_fuel_displaced'],
            })

    # Sort by MAC (greedy queue)
    queue.sort(key=lambda x: (x['marginal_mac'], -x['co2_displaced_mt']))

    # Add rank
    for i, step in enumerate(queue):
        step['rank'] = i + 1

    return queue


def build_scenario_bands(all_results):
    """Per-ISO, per-threshold: min/max cost & MAC across 5 scenarios."""
    bands = {}
    for iso in ISOS:
        bands[iso] = {}
        for threshold in CONSQ_THRESHOLDS:
            costs = {}
            macs = {}
            for scenario in SCENARIO_ORDER:
                traj = all_results.get(iso, {}).get(scenario, [])
                entry = next((e for e in traj if e['threshold'] == threshold), None)
                if entry:
                    costs[scenario] = entry['total_cost_bn']
                    macs[scenario] = entry['mac']
            if costs:
                bands[iso][str(threshold)] = {
                    'cost_bn': costs,
                    'mac': macs,
                }
    return bands


# ============================================================================
# OUTPUT
# ============================================================================

def write_outputs(all_results, deployment_queue, scenario_bands, egrid_rates):
    """Write JSON + JS output files."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(JS_DIR, exist_ok=True)

    output = {
        'metadata': {
            'description': 'Sequential consequential optimizer — mines PFS with $/tCO2 objective',
            'methodology': (
                'Greedy path-dependent optimization: at each SBTi threshold, selects the '
                'PFS mix that minimizes incremental $/tCO2 subject to floor ratcheting '
                '(committed resources cannot decrease). Uses Wright\'s Law learning curves '
                'from pipeline_config mapped to SBTi target years.'
            ),
            'scenarios': {k: v['description'] for k, v in SCENARIOS.items()},
            'default_scenario': DEFAULT_SCENARIO,
            'scenario_order': SCENARIO_ORDER,
            'thresholds': CONSQ_THRESHOLDS,
            'threshold_years': {str(t): SBTI_YEAR_MAP.get(t, 2050) for t in CONSQ_THRESHOLDS},
            'isos': ISOS,
            'grid_mix_shares': {iso: dict(GRID_MIX_SHARES.get(iso, {})) for iso in ISOS},
            'existing_clean_pct': {
                iso: round(sum(GRID_MIX_SHARES.get(iso, {}).values()), 1) for iso in ISOS
            },
            'baseline_co2_mt': {
                iso: round(float(merit_order_co2(
                    np.array([sum(GRID_MIX_SHARES.get(iso, {}).values())]),
                    REGIONAL_DEMAND_TWH[iso],
                    COAL_CAP_TWH.get(iso, 0), OIL_CAP_TWH.get(iso, 0),
                    egrid_rates[iso]['coal'], egrid_rates[iso]['oil'],
                    egrid_rates[iso]['gas'])[0]), 2) for iso in ISOS
            },
            'sbti_year_map': {str(k): v for k, v in THRESHOLD_TARGET_YEARS.items()},
        },
        'trajectories': all_results,
        'deployment_queue': deployment_queue,
        'scenario_bands': scenario_bands,
    }

    # Write JSON
    json_path = os.path.join(OUTPUT_DIR, 'consequential_queue.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nWrote {json_path} ({os.path.getsize(json_path) / 1e6:.1f} MB)")

    # Write JS
    js_path = os.path.join(JS_DIR, 'consequential-queue-data.js')
    with open(js_path, 'w') as f:
        f.write('// Auto-generated by step5d_consequential_optimizer.py\n')
        f.write('// Sequential consequential optimizer — mines PFS with $/tCO2 objective\n')
        f.write('// Wright\'s Law learning curves mapped to SBTi target years\n\n')
        f.write('const CQ_DATA = ')
        json.dump(output, f, indent=2, default=str)
        f.write(';\n')
    print(f"Wrote {js_path} ({os.path.getsize(js_path) / 1e6:.1f} MB)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Step 5D: Sequential Consequential Optimizer')
    parser.add_argument('--iso', type=str, help='Run single ISO (e.g., SPP)')
    parser.add_argument('--scenario', type=str, help='Run single scenario (e.g., all_medium)')
    args = parser.parse_args()

    print("=" * 70)
    print("Step 5D: Sequential Consequential Optimizer")
    print("=" * 70)

    # Load emission rates
    egrid_rates = load_egrid()

    # Determine ISOs to process
    isos = [args.iso] if args.iso else ISOS
    scenarios = [args.scenario] if args.scenario else SCENARIO_ORDER

    all_results = {}
    for iso in isos:
        print(f"\n{'─' * 50}")
        print(f"Processing {iso} (demand: {REGIONAL_DEMAND_TWH[iso]:.0f} TWh, "
              f"existing clean: {sum(GRID_MIX_SHARES.get(iso, {}).values()):.1f}%)")
        print(f"{'─' * 50}")

        pool = load_pfs_pool(iso)
        if pool is None or len(pool) == 0:
            print(f"  WARNING: No PFS data for {iso}, skipping")
            continue

        results = optimize_iso(iso, pool, egrid_rates, scenarios)
        all_results[iso] = results

        # Print summary for medium scenario
        summary_sc = 'all_medium' if 'all_medium' in results else scenarios[0]
        traj = results.get(summary_sc, [])
        if traj:
            print(f"\n  Summary ({summary_sc}):")
            print(f"  {'Thresh':>7} {'Year':>5} {'Clean%':>7} {'Cost$B':>7} {'MAC$/t':>7} {'CO2 MT':>7} {'Primary':>8}")
            for e in traj:
                if e.get('note') == 'no_candidates':
                    continue
                mac_str = '∞' if e['mac'] >= 9999 else f"${e['mac']:.0f}"
                fuel = e['primary_fuel_displaced']
                print(f"  {e['threshold']:>6.1f}% {e['year']:>5} "
                      f"{e['effective_clean_pct']:>6.1f}% "
                      f"{e['total_cost_bn']:>6.1f} "
                      f"{mac_str:>7} "
                      f"{e['co2_abated_mt']:>6.1f} "
                      f"{fuel:>8}")

    # Build cross-ISO deployment queue
    print("\nBuilding cross-ISO deployment queue...")
    queue = build_deployment_queue(all_results, egrid_rates)
    bands = build_scenario_bands(all_results)

    # Write outputs
    write_outputs(all_results, queue, bands, egrid_rates)

    print("\nDone!")


if __name__ == '__main__':
    main()
