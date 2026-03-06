#!/usr/bin/env python3
"""
Step 5D: MAC-Optimized Consequential Queue
============================================
Path-dependent deployment queue optimizing for cheapest $/mtCO2 avoided,
with resource lock-in ("ratcheting floor") at each threshold.

Algorithm:
  For each ISO × price_sensitivity × demand_growth:
    1. Start with existing clean resources as floor
    2. At each threshold T (SBTi year mapping):
       a. Compute demand at year Y with growth
       b. Dispatch floor resources → compute CO2 baseline (path-dependent)
       c. Sample archetypes from PFS (raw, fine, floor, storage) that respect floor
       d. Score: new_build_cost / CO2_avoided  ($/mtCO2)
       e. Winner = argmin(MAC) with overshoot ≤ 1%
       f. Phase 2: refine around best archetypes
       g. Ratchet: lock winner resources as new floor

15 pathways per ISO: 3 demand growth × 5 price sensitivities = 105 total.

Cost model: NEW BUILD ONLY (no gas backup, no wholesale, no system cost).
  - Existing clean resources = $0
  - New build: LCOE + transmission, learning curves at target year
  - Clean firm tranching: uprate → geothermal (CAISO) → min(nuclear, CCS)
  - Cumulative caps: uprate, geothermal, CCS track usage across thresholds
  - Hydro: existing only, $0

CO2 model: Merit-order fossil retirement (coal → oil → gas).
  - Path-dependent baseline: CO2 after prior threshold procurement
  - Hourly dispatch reconstruction for shape-accurate fossil displacement

Input:  data/step1-pfs-parquets/ (raw_pfs, fine_pfs, floor_pfs, storage)
Output: data/step5-post-processing/mac_queue/mac_queue_{ISO}.parquet + mac_queue_summary.json
"""

import argparse
import gc
import json
import os
import sys
import time
import glob as globmod
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Add scripts dir to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from pipeline_config import (
    ISOS, OFFSHORE_ISOS, GEOTHERMAL_ISOS,
    REGIONAL_DEMAND_TWH, GRID_MIX_SHARES,
    LCOE_TABLES, TX_TABLES, get_tx,
    UPRATE_LCOE, UPRATE_CAP_TWH,
    NUCLEAR_NEWBUILD_LCOE, GEOTHERMAL_LCOE, GEOTHERMAL_CAP_TWH,
    CCS_LCOE_45Q_ON, CCS_LCOE_45Q_OFF, CCS_CAP_TWH,
    NEISO_CCS_GAS_ADDER, EXISTING_GEOTHERMAL_PCT,
    FOAK_NUCLEAR_NEWBUILD, FOAK_CCS_45Q_ON, FOAK_CCS_45Q_OFF,
    FOAK_GEOTHERMAL, FOAK_OFFSHORE_WIND, FOAK_LDES, FOAK_H2,
    NOAK_BATTERY, NOAK_BATTERY8, NOAK_OFFSHORE_WIND,
    LEARNING_PARAMS, learning_fraction, year_adjusted_cost,
    STORAGE_REVENUE_CREDITS,
    DEMAND_GROWTH_RATES, THRESHOLD_TARGET_YEARS,
    HYDRO_CAP_PCT,
    LEVEL_NAME, LMH,
    PATHS, H,
    compute_clean_firm_tranches,
)
from dispatch_utils import (
    load_common_data,
    get_demand_profile,
    get_supply_profiles,
    build_supply_matrix,
    reconstruct_hourly_dispatch,
    compute_co2_from_dispatch,
    compute_fossil_retirement,
    COAL_CAP_TWH, OIL_CAP_TWH,
    RESOURCE_TYPES,
)
import step1_pfs_generator as s1

from scenario_common import (
    SCENARIO_A, THRESHOLDS as SCENARIO_THRESHOLDS,
    RESOURCES, WHOLESALE_PRICES, FUEL_ADJUSTMENTS,
    BASE_DEMAND_TWH, HYDRO_CAP_TWH as SC_HYDRO_CAP_TWH,
    PEAK_DEMAND_MW, EXISTING_GAS_CAPACITY_MW, EXISTING_GAS_FOM_KW_YR,
    NEW_CCGT_COST_KW_YR, PEAK_CAPACITY_CREDITS, GAS_AVAILABILITY_FACTOR,
    RESOURCE_ADEQUACY_MARGIN, SBTI_YEAR_MAP,
    UPRATE_CAP_TWH as SC_UPRATE_CAP_TWH,
    GEO_CAP_TWH, LEVEL_NAME as SC_LEVEL_NAME,
    compute_mix_cost, save_scenario_results,
    get_demand_growth_factor,
    _build_existing_only_entry, _existing_match_pct,
    learning_fraction as scenario_learning_fraction,
    _build_learning_overrides,
)

# ============================================================================
# CONSTANTS
# ============================================================================

PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'step5-post-processing', 'mac_queue')

PFS_DIRS = [
    os.path.join(PROJECT_ROOT, 'data', 'step1-pfs-parquets'),
]

RESOURCE_COLS = ['clean_firm', 'solar', 'wind', 'hydro', 'offshore_wind', 'geothermal']
STORAGE_COLS = ['battery_dispatch_pct', 'battery8_dispatch_pct', 'ldes_dispatch_pct', 'h2_dispatch_pct']
MIX_COLS = RESOURCE_COLS + STORAGE_COLS + ['hourly_match_score']

# Active thresholds for the MAC queue (50%+ only — coarse thresholds excluded)
MAC_THRESHOLDS = [50, 55, 60, 65, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 99.5, 99.9, 99.99]

# Hydro cap in TWh (existing only, derived from HYDRO_CAP_PCT × base demand)
HYDRO_CAP_TWH = {iso: HYDRO_CAP_PCT[iso] / 100.0 * REGIONAL_DEMAND_TWH[iso]
                 for iso in ISOS}


def cap_hydro_in_mix(mix_pct, iso):
    """Cap hydro at physical maximum and adjust match score accordingly.

    Hydro is existing-only and priced at $0. If a PFS mix has hydro above the
    physical cap, the excess is phantom generation that inflates both the match
    score (making CO2 displacement look too high) and the cost (by substituting
    $0 hydro for priced resources). This function caps hydro and reduces the
    match score by the excess percentage.

    Modifies mix_pct in place and returns the excess hydro percentage.
    """
    hydro_cap_pct = HYDRO_CAP_PCT.get(iso, 50.0)
    hydro_pct = mix_pct.get('hydro', 0)
    if hydro_pct > hydro_cap_pct:
        excess = hydro_pct - hydro_cap_pct
        mix_pct['hydro'] = hydro_cap_pct
        # Reduce match score by excess hydro — this generation doesn't exist
        if 'hourly_match_score' in mix_pct:
            mix_pct['hourly_match_score'] = max(0, mix_pct['hourly_match_score'] - excess)
        return excess
    return 0.0

# 5 price sensitivities
PRICE_SENSITIVITIES = {
    'all_low': {
        'ren': 'Low', 'firm': 'L', 'batt': 'Low', 'ldes_lvl': 'Low',
        'ccs': 'L', 'q45': '1', 'fuel': 'Low', 'tx': 'Low', 'geo': 'L',
    },
    'all_med': {
        'ren': 'Medium', 'firm': 'M', 'batt': 'Medium', 'ldes_lvl': 'Medium',
        'ccs': 'M', 'q45': '1', 'fuel': 'Medium', 'tx': 'Medium', 'geo': 'M',
    },
    'all_high': {
        'ren': 'High', 'firm': 'H', 'batt': 'High', 'ldes_lvl': 'High',
        'ccs': 'H', 'q45': '1', 'fuel': 'High', 'tx': 'High', 'geo': 'H',
    },
    'high_vre_low_firm': {
        'ren': 'High', 'firm': 'L', 'batt': 'High', 'ldes_lvl': 'High',
        'ccs': 'L', 'q45': '1', 'fuel': 'Medium', 'tx': 'Medium', 'geo': 'L',
    },
    'high_firm_low_vre': {
        'ren': 'Low', 'firm': 'H', 'batt': 'Low', 'ldes_lvl': 'Low',
        'ccs': 'H', 'q45': '1', 'fuel': 'Medium', 'tx': 'Medium', 'geo': 'H',
    },
}

DEMAND_GROWTH_LEVELS = ['Low', 'Medium', 'High']

MAX_ARCHETYPES = 5000
PHASE2_PERTURBATIONS = 2000
MAX_OVERSHOOT = 0.5  # percentage points — tight target adherence


# ============================================================================
# PFS LOADING
# ============================================================================

def _load_pfs_files(iso, threshold):
    """Load PFS parquet files for a single threshold from all 3 directories."""
    t_str = f'{threshold:g}'
    dfs = []

    for pfs_dir in PFS_DIRS:
        if not os.path.isdir(pfs_dir):
            continue
        patterns = [
            os.path.join(pfs_dir, f'{iso}_t{t_str}_raw_pfs.parquet'),
            os.path.join(pfs_dir, f'{iso}_t{t_str}_floor_pfs.parquet'),
            os.path.join(pfs_dir, f'{iso}_t{t_str}_fine_pfs.parquet'),
            os.path.join(pfs_dir, f'{iso}_t{t_str}_storage.parquet'),
        ]
        batch_files = globmod.glob(os.path.join(pfs_dir, f'{iso}_t{t_str}_storage_b*.parquet'))

        for pat in patterns:
            if os.path.isfile(pat):
                try:
                    df = pd.read_parquet(pat)
                    dfs.append(df)
                except Exception:
                    pass

        for bf in batch_files:
            try:
                df = pd.read_parquet(bf)
                dfs.append(df)
            except Exception:
                pass

    return dfs


def load_pfs_for_threshold(iso, threshold):
    """Load PFS mixes from target threshold AND adjacent thresholds.

    For early thresholds near existing clean, PFS mixes are full portfolios
    not incremental additions. Loading from adjacent thresholds (and the coarse
    cache/near-miss files) provides mixes closer to the floor that need less new build.

    Handles batch files (_b1, _b2, etc.) and deduplicates.
    """
    dfs = []

    # Load from target threshold
    dfs.extend(_load_pfs_files(iso, threshold))

    # Also load from the threshold below (may have mixes that overshoot into our range)
    idx = MAC_THRESHOLDS.index(threshold) if threshold in MAC_THRESHOLDS else -1
    if idx > 0:
        lower_t = MAC_THRESHOLDS[idx - 1]
        dfs.extend(_load_pfs_files(iso, lower_t))

    # Also load coarse cache (has mixes across ALL thresholds, useful for interpolation)
    # Coarse cache has different schema: 'score' instead of 'hourly_match_score',
    # no storage columns, no 'iso'/'threshold' columns
    coarse_file = os.path.join(PFS_DIRS[0], f'{iso}_coarse_cache.parquet')
    if os.path.isfile(coarse_file):
        try:
            coarse_df = pd.read_parquet(coarse_file)
            # Rename score → hourly_match_score
            if 'score' in coarse_df.columns and 'hourly_match_score' not in coarse_df.columns:
                coarse_df = coarse_df.rename(columns={'score': 'hourly_match_score'})
            # Add missing storage columns as 0
            for col in STORAGE_COLS:
                if col not in coarse_df.columns:
                    coarse_df[col] = 0.0
            # Filter to relevant match score range before adding (avoid huge concat)
            score_min = threshold - 2.0
            score_max = threshold + MAX_OVERSHOOT + 1.0
            coarse_df = coarse_df[
                (coarse_df['hourly_match_score'] >= score_min) &
                (coarse_df['hourly_match_score'] <= score_max)
            ]
            if len(coarse_df) > 0:
                dfs.append(coarse_df[MIX_COLS] if all(c in coarse_df.columns for c in MIX_COLS) else coarse_df)
        except Exception:
            pass

    # Also load near-miss file (mixes that barely missed various thresholds)
    near_miss_file = os.path.join(PFS_DIRS[0], f'{iso}_near_miss.parquet')
    if os.path.isfile(near_miss_file):
        try:
            nm_df = pd.read_parquet(near_miss_file)
            if 'score' in nm_df.columns and 'hourly_match_score' not in nm_df.columns:
                nm_df = nm_df.rename(columns={'score': 'hourly_match_score'})
            for col in STORAGE_COLS:
                if col not in nm_df.columns:
                    nm_df[col] = 0.0
            # Filter to relevant range
            nm_df = nm_df[
                (nm_df['hourly_match_score'] >= threshold - 2.0) &
                (nm_df['hourly_match_score'] <= threshold + MAX_OVERSHOOT + 1.0)
            ]
            if len(nm_df) > 0:
                dfs.append(nm_df[[c for c in MIX_COLS if c in nm_df.columns]])
        except Exception:
            pass

    if not dfs:
        return pd.DataFrame(columns=MIX_COLS)

    combined = pd.concat(dfs, ignore_index=True)

    # Fill missing columns with 0
    for col in MIX_COLS:
        if col not in combined.columns:
            combined[col] = 0.0

    # Deduplicate by resource composition
    dedup_cols = RESOURCE_COLS + STORAGE_COLS
    combined_rounded = combined.copy()
    for c in dedup_cols:
        combined_rounded[c] = combined_rounded[c].round(3)
    combined = combined.loc[combined_rounded.drop_duplicates(subset=dedup_cols).index]

    return combined.reset_index(drop=True)


# ============================================================================
# NEW-BUILD COST FUNCTION
# ============================================================================

def _apply_learning_curve(base_lcoe, foak_cost, noak_cost, tech_key, toggle_level, target_year):
    """Apply Wright's Law learning curve to get year-adjusted cost.

    For technologies with FOAK→NOAK curves (nuclear, CCS, geo, LDES, H2, offshore),
    interpolates between FOAK and NOAK at the target year.

    For VRE (solar, wind onshore), uses static L/M/H LCOE (no FOAK/NOAK — already at scale).
    """
    if foak_cost is None or noak_cost is None:
        return base_lcoe  # No learning curve — use static LCOE

    params = LEARNING_PARAMS.get(tech_key, {}).get(toggle_level)
    if params is None:
        return base_lcoe

    foak_start, noak_year = params
    return year_adjusted_cost(foak_cost, noak_cost, target_year, foak_start, noak_year)


def compute_new_build_cost(iso, mix_pct, floor_twh, demand_twh, sens, target_year,
                            cumulative_caps):
    """Compute total annualized new-build cost in $ for resources above the floor.

    Args:
        iso: ISO region
        mix_pct: dict of resource allocations (% of demand at target year)
        floor_twh: dict of locked-in floor (absolute TWh)
        demand_twh: demand at target year (TWh)
        sens: price sensitivity dict
        target_year: SBTi year for learning curve
        cumulative_caps: dict tracking cumulative usage of capped resources

    Returns:
        total_cost ($), cost_breakdown (dict), updated cumulative_caps
    """
    ren_name = sens['ren']
    firm_lev = sens['firm']
    batt_name = sens['batt']
    ldes_name = sens['ldes_lvl']
    ccs_lev = sens['ccs']
    q45 = sens['q45']
    tx_name = sens['tx']
    geo_lev = sens.get('geo')

    total_cost = 0.0
    breakdown = {}

    # Convert mix from % of demand to TWh
    mix_twh = {}
    for res in RESOURCE_COLS:
        mix_twh[res] = mix_pct.get(res, 0.0) / 100.0 * demand_twh

    # Cap hydro at physical limit (existing only, no new-build possible)
    hydro_cap_twh = HYDRO_CAP_TWH.get(iso, mix_twh.get('hydro', 0))
    mix_twh['hydro'] = min(mix_twh.get('hydro', 0), hydro_cap_twh)

    # --- Solar (existing = $0, new build = LCOE + TX) ---
    existing_solar_twh = floor_twh.get('solar', 0.0)
    sol_new_twh = max(0, mix_twh['solar'] - existing_solar_twh)
    if sol_new_twh > 0:
        sol_lcoe = LCOE_TABLES['solar'][ren_name][iso] + get_tx('solar', tx_name, iso)
        sol_cost = sol_new_twh * 1e6 * sol_lcoe
        total_cost += sol_cost
        breakdown['solar'] = {'new_twh': sol_new_twh, 'lcoe': sol_lcoe, 'cost': sol_cost}

    # --- Wind (existing = $0, new build = LCOE + TX) ---
    existing_wind_twh = floor_twh.get('wind', 0.0)
    wnd_new_twh = max(0, mix_twh['wind'] - existing_wind_twh)
    if wnd_new_twh > 0:
        wnd_lcoe = LCOE_TABLES['wind'][ren_name][iso] + get_tx('wind', tx_name, iso)
        wnd_cost = wnd_new_twh * 1e6 * wnd_lcoe
        total_cost += wnd_cost
        breakdown['wind'] = {'new_twh': wnd_new_twh, 'lcoe': wnd_lcoe, 'cost': wnd_cost}

    # --- Offshore wind (all new build, learning curve applied) ---
    existing_osw_twh = floor_twh.get('offshore_wind', 0.0)
    osw_new_twh = max(0, mix_twh.get('offshore_wind', 0) - existing_osw_twh)
    if iso in OFFSHORE_ISOS and osw_new_twh > 0:
        osw_base = LCOE_TABLES['offshore_wind'][ren_name][iso]
        osw_tx = get_tx('offshore_wind', tx_name, iso)
        # Learning curve: FOAK → NOAK
        foak = FOAK_OFFSHORE_WIND.get(iso)
        noak = NOAK_OFFSHORE_WIND.get(ren_name, {}).get(iso)
        tech_key = 'offshore_wind_float' if iso == 'CAISO' else 'offshore_wind_fixed'
        if foak and noak:
            osw_lcoe = _apply_learning_curve(osw_base, foak, noak, tech_key, firm_lev, target_year)
        else:
            osw_lcoe = osw_base
        osw_lcoe += osw_tx
        osw_cost = osw_new_twh * 1e6 * osw_lcoe
        total_cost += osw_cost
        breakdown['offshore_wind'] = {'new_twh': osw_new_twh, 'lcoe': osw_lcoe, 'cost': osw_cost}

    # --- Physics Geothermal (CAISO only, priced separately from clean firm) ---
    # Geothermal from the Step 1 physics dimension: existing (≤2.37%) at $0,
    # new-build above existing at geothermal LCOE. Tracks against the shared
    # 39 TWh cap to prevent double-counting with the clean firm tranche.
    geo_physics_new_twh = 0.0
    if iso == 'CAISO' and geo_lev:
        geo_pct = mix_pct.get('geothermal', 0.0)
        existing_geo_pct = EXISTING_GEOTHERMAL_PCT
        geo_new_pct = max(0, geo_pct - existing_geo_pct)
        geo_physics_new_twh = geo_new_pct / 100.0 * demand_twh
        if geo_physics_new_twh > 0:
            geo_base = GEOTHERMAL_LCOE[geo_lev]
            foak_geo = FOAK_GEOTHERMAL
            noak_geo = GEOTHERMAL_LCOE['L']
            geo_lcoe = _apply_learning_curve(geo_base, foak_geo, noak_geo, 'geo', firm_lev, target_year)
            geo_lcoe += get_tx('clean_firm', tx_name, iso)
            geo_physics_cost = geo_physics_new_twh * 1e6 * geo_lcoe
            total_cost += geo_physics_cost
            breakdown['geothermal_physics'] = {
                'new_twh': geo_physics_new_twh, 'lcoe': geo_lcoe, 'cost': geo_physics_cost}

    # --- Clean Firm (tranched via shared utility: uprate → geo → min(nuclear, CCS)) ---
    existing_cf_twh = floor_twh.get('clean_firm', 0.0)
    cf_new_twh = max(0, mix_twh['clean_firm'] - existing_cf_twh)
    if cf_new_twh > 0:
        tranches = compute_clean_firm_tranches(
            new_cf_twh=cf_new_twh,
            iso=iso,
            firm_lev=firm_lev,
            ccs_lev=ccs_lev,
            q45=q45,
            tx_name=tx_name,
            geo_lev=geo_lev,
            geo_physics_new_twh=geo_physics_new_twh,
            ccs_used_twh=cumulative_caps.get('ccs_twh', 0.0),
            uprate_used_twh=cumulative_caps.get('uprate_twh', 0.0),
            geo_used_twh=cumulative_caps.get('geo_twh', 0.0),
            learning_curve_fn=_apply_learning_curve,
            target_year=target_year,
        )
        # Scale costs from $/MWh×TWh to $ (shared utility returns in TWh×$/MWh units)
        total_cost += tranches['total_cost'] * 1e6

        # Update cumulative caps
        cumulative_caps['uprate_twh'] = cumulative_caps.get('uprate_twh', 0.0) + tranches['uprate_twh']
        cumulative_caps['geo_twh'] = cumulative_caps.get('geo_twh', 0.0) + tranches['geo_twh']
        cumulative_caps['ccs_twh'] = cumulative_caps.get('ccs_twh', 0.0) + tranches['ccs_tranche_twh']

        # Breakdown
        breakdown['uprate'] = {'new_twh': tranches['uprate_twh'],
                               'lcoe': tranches['uprate_lcoe'],
                               'cost': tranches['uprate_cost'] * 1e6}
        if tranches['geo_twh'] > 0:
            breakdown['geothermal_tranche'] = {'new_twh': tranches['geo_twh'],
                                                'lcoe': tranches['geo_lcoe'],
                                                'cost': tranches['geo_cost'] * 1e6}
        breakdown['nuclear_newbuild'] = {'new_twh': tranches['nuclear_twh'],
                                         'lcoe': tranches['nuclear_lcoe']}
        breakdown['ccs_tranche'] = {'new_twh': tranches['ccs_tranche_twh'],
                                     'lcoe': tranches['ccs_lcoe']}
        breakdown['tranche3_cost'] = (tranches['nuclear_cost'] + tranches['ccs_tranche_cost']) * 1e6

    # --- CCS-CCGT implicit residual ---
    # The CCS residual (100% - explicit resources) represents remaining fossil
    # generation, NOT CCS infrastructure being built. CCS new-build is already
    # handled through the clean_firm tranche (uprate → geo → min(nuclear, CCS)).
    # Do NOT charge for the implicit residual — it's just the grid continuing to
    # run fossil while clean resources displace what they can hourly.
    ccs_pct = 100.0 - (mix_pct.get('clean_firm', 0) + mix_pct.get('solar', 0) +
                        mix_pct.get('wind', 0) + mix_pct.get('hydro', 0) +
                        mix_pct.get('offshore_wind', 0) + mix_pct.get('geothermal', 0))
    ccs_pct = max(0, ccs_pct)
    breakdown['ccs_residual_pct'] = ccs_pct  # Track for diagnostics only

    # --- Storage (annualized capacity cost - revenue credits) ---
    storage_map = {
        'battery_dispatch_pct': ('battery', batt_name),
        'battery8_dispatch_pct': ('battery8', batt_name),
        'ldes_dispatch_pct': ('ldes', ldes_name),
        'h2_dispatch_pct': ('h2', ldes_name),
    }
    for col, (stype, slevel) in storage_map.items():
        mix_val = mix_pct.get(col, 0.0)
        floor_val = floor_twh.get(col, 0.0)  # Storage floor is in % (not TWh)
        new_pct = max(0, mix_val - floor_val)
        if new_pct > 0:
            base_price = LCOE_TABLES[stype][slevel][iso]
            rev_credit = STORAGE_REVENUE_CREDITS[stype][iso]
            price = max(0, base_price - rev_credit)

            # Apply learning curves for storage technologies
            if stype == 'battery':
                foak_s = base_price  # Batteries start at current cost
                noak_s = NOAK_BATTERY.get(slevel, {}).get(iso, base_price)
                tech_key_s = 'bat4'
            elif stype == 'battery8':
                foak_s = base_price
                noak_s = NOAK_BATTERY8.get(slevel, {}).get(iso, base_price)
                tech_key_s = 'bat8'
            elif stype == 'ldes':
                foak_s = FOAK_LDES.get(iso, base_price)
                noak_s = LCOE_TABLES['ldes']['Low'][iso]
                tech_key_s = 'ldes'
            elif stype == 'h2':
                foak_s = FOAK_H2.get(iso, base_price)
                noak_s = LCOE_TABLES['h2']['Low'][iso]
                tech_key_s = 'h2'
            else:
                foak_s, noak_s, tech_key_s = None, None, None

            if foak_s is not None and noak_s is not None:
                adj_price = _apply_learning_curve(base_price, foak_s, noak_s, tech_key_s,
                                                   firm_lev, target_year)
                adj_rev = STORAGE_REVENUE_CREDITS[stype][iso]
                price = max(0, adj_price - adj_rev)

            # Storage pricing: LCOE_TABLES[stype] values are annualized $/MWh-of-total-demand
            # per percentage point of dispatch. So for pct% dispatch:
            #   cost_per_mwh_demand = pct/100 × price
            #   annual_cost = cost_per_mwh_demand × demand_mwh
            # This matches step3a: total_cost += bat_pct / 100.0 * bat4_price
            # (which adds $/MWh of demand, then gets multiplied by demand later)
            s_cost_total = new_pct / 100.0 * price * demand_twh * 1e6
            total_cost += s_cost_total
            breakdown[stype] = {'new_pct': new_pct, 'price_per_pct': price, 'cost': s_cost_total}

    # Hydro: $0 (existing only, already in floor)

    return total_cost, breakdown, cumulative_caps


# ============================================================================
# CO2 CALCULATION
# ============================================================================

def compute_co2_at_mix(iso, mix_pct, demand_twh, demand_norm, supply_profiles,
                       supply_matrix, emission_rates):
    """Compute total fossil CO2 emissions for a given mix dispatched against demand.

    Returns:
        co2_total_tons: total annual fossil CO2 (metric tons)
        co2_info: detailed breakdown dict
    """
    demand_mwh = demand_twh * 1e6

    resource_pcts = {
        'clean_firm': mix_pct.get('clean_firm', 0),
        'solar': mix_pct.get('solar', 0),
        'wind': mix_pct.get('wind', 0),
        'offshore_wind': mix_pct.get('offshore_wind', 0),
        'ccs_ccgt': 0,  # CCS is implicit residual, handled separately
        'hydro': mix_pct.get('hydro', 0),
    }

    # CCS implicit residual
    explicit_sum = sum(resource_pcts.values()) + mix_pct.get('geothermal', 0)
    ccs_pct = max(0, 100.0 - explicit_sum)
    resource_pcts['ccs_ccgt'] = ccs_pct

    bat_pct = mix_pct.get('battery_dispatch_pct', 0)
    bat8_pct = mix_pct.get('battery8_dispatch_pct', 0)
    ldes_pct = mix_pct.get('ldes_dispatch_pct', 0)
    h2_pct = mix_pct.get('h2_dispatch_pct', 0)

    dispatch = reconstruct_hourly_dispatch(
        demand_norm, supply_profiles, resource_pcts,
        procurement_pct=100,
        battery_dispatch_pct=bat_pct,
        battery8_dispatch_pct=bat8_pct,
        ldes_dispatch_pct=ldes_pct,
        h2_dispatch_pct=h2_pct,
        supply_matrix=supply_matrix,
    )

    co2_info = compute_co2_from_dispatch(iso, dispatch, emission_rates, demand_mwh)
    return co2_info


# ============================================================================
# FLOOR MANAGEMENT
# ============================================================================

def init_floor(iso):
    """Initialize floor from existing clean resources (absolute TWh)."""
    shares = GRID_MIX_SHARES[iso]
    demand = REGIONAL_DEMAND_TWH[iso]
    floor = {}
    for res in RESOURCE_COLS:
        floor[res] = shares.get(res, 0.0) / 100.0 * demand
    # Storage floor starts at 0 (no existing storage)
    for col in STORAGE_COLS:
        floor[col] = 0.0
    return floor


def floor_to_pct(floor_twh, demand_twh):
    """Convert absolute TWh floor to % of demand at a given demand level."""
    pct = {}
    for res in RESOURCE_COLS:
        pct[res] = floor_twh.get(res, 0.0) / demand_twh * 100.0
    for col in STORAGE_COLS:
        pct[col] = floor_twh.get(col, 0.0)  # Storage is already in %
    return pct


def mix_row_to_pct(row):
    """Convert a DataFrame row to a mix_pct dict."""
    pct = {}
    for col in RESOURCE_COLS + STORAGE_COLS:
        pct[col] = float(row.get(col, 0.0))
    pct['hourly_match_score'] = float(row.get('hourly_match_score', 0.0))
    return pct


def ratchet_floor(floor_twh, winner_pct, demand_twh):
    """Update FILTER floor — only ratchets when winner deploys a higher SHARE.

    The floor is stored in absolute TWh but only increases when the winner's
    percentage of a resource exceeds the floor's implicit percentage at the
    current demand level. This ensures the floor diminishes as a share of
    demand when demand grows (e.g., 50 TWh at 100 TWh demand = 50%, but at
    103 TWh demand = 48.5%), widening the search space for cheaper mixes.

    Demand-growth "maintenance" builds (same % × higher demand) do NOT raise
    the floor — they are costed but don't constrain the next threshold's search.
    """
    new_floor = dict(floor_twh)
    for res in RESOURCE_COLS:
        # Current floor as % of current demand
        floor_pct_res = (floor_twh.get(res, 0.0) / demand_twh * 100.0) if demand_twh > 0 else 0.0
        winner_pct_res = winner_pct.get(res, 0.0)
        if winner_pct_res > floor_pct_res + 0.01:  # Tolerance for float comparison
            # Winner deployed a HIGHER SHARE → ratchet floor to winner's absolute TWh
            new_floor[res] = winner_pct_res / 100.0 * demand_twh
        # else: keep floor at old absolute TWh (diminishes as % of growing demand)
    for col in STORAGE_COLS:
        new_floor[col] = max(floor_twh.get(col, 0.0), winner_pct.get(col, 0.0))
    return new_floor


def ratchet_deployed(deployed_twh, winner_pct, demand_twh):
    """Update COST floor — always takes max of old deployed and winner's TWh.

    Tracks actual physical capacity deployed so far. Used for cost calculations
    to avoid double-charging for capacity that already exists.

    NOTE: deployed_twh is NOT scaled with demand growth — it represents physical
    capacity already built. When demand grows, the winner's TWh (= pct × new demand)
    may exceed previously deployed TWh, requiring new build. This is correct:
    demand growth requires additional physical capacity to maintain the same %.
    """
    new_deployed = dict(deployed_twh)
    for res in RESOURCE_COLS:
        winner_res_twh = winner_pct.get(res, 0.0) / 100.0 * demand_twh
        new_deployed[res] = max(deployed_twh.get(res, 0.0), winner_res_twh)
    for col in STORAGE_COLS:
        new_deployed[col] = max(deployed_twh.get(col, 0.0), winner_pct.get(col, 0.0))
    return new_deployed


# ============================================================================
# ARCHETYPE FILTERING & SAMPLING
# ============================================================================

def filter_and_sample(df, floor_pct, threshold, max_samples=MAX_ARCHETYPES):
    """Filter PFS mixes that respect floor and overshoot, then sample.

    Args:
        df: DataFrame of PFS mixes
        floor_pct: dict of minimum resource allocations (% of demand)
        threshold: target threshold
        max_samples: max number of archetypes to return

    Returns:
        filtered DataFrame (up to max_samples rows)
    """
    if df.empty:
        return df

    # Floor filter: each resource >= floor (with small tolerance)
    mask = np.ones(len(df), dtype=bool)
    for res in RESOURCE_COLS:
        floor_val = floor_pct.get(res, 0.0)
        if floor_val > 0.01:
            mask &= (df[res].values >= floor_val - 0.5)  # 0.5% tolerance

    for col in STORAGE_COLS:
        floor_val = floor_pct.get(col, 0.0)
        if floor_val > 0.01:
            mask &= (df[col].values >= floor_val - 0.1)

    # Overshoot filter: match score within [threshold, threshold + MAX_OVERSHOOT]
    # Strict lower bound: mixes must MEET the target, not just come close.
    # This prevents picking "free" mixes that barely miss and cost $0.
    scores = df['hourly_match_score'].values
    mask &= (scores >= threshold)
    mask &= (scores <= threshold + MAX_OVERSHOOT)

    filtered = df.loc[mask].copy()

    if len(filtered) == 0:
        # Fallback: relax overshoot to 2%
        mask2 = np.ones(len(df), dtype=bool)
        for res in RESOURCE_COLS:
            floor_val = floor_pct.get(res, 0.0)
            if floor_val > 0.01:
                mask2 &= (df[res].values >= floor_val - 1.0)
        for col in STORAGE_COLS:
            floor_val = floor_pct.get(col, 0.0)
            if floor_val > 0.01:
                mask2 &= (df[col].values >= floor_val - 0.5)
        mask2 &= (scores >= threshold - 0.25)
        mask2 &= (scores <= threshold + 1.0)
        filtered = df.loc[mask2].copy()

    if len(filtered) <= max_samples:
        return filtered.reset_index(drop=True)

    # Stratified sampling for diversity
    return filtered.sample(n=max_samples, random_state=42).reset_index(drop=True)


# ============================================================================
# PHASE 2: REFINED GRID SEARCH
# ============================================================================

def phase2_refine(top_mixes, floor_pct, threshold, num_perturbations=PHASE2_PERTURBATIONS):
    """Generate perturbations around top archetypes for refined search.

    Args:
        top_mixes: list of (mix_pct, mac_score) tuples (top 5)
        floor_pct: current floor in %
        threshold: target threshold
        num_perturbations: number of variants to generate

    Returns:
        DataFrame of perturbed mixes
    """
    if not top_mixes:
        return pd.DataFrame(columns=MIX_COLS)

    per_archetype = max(1, num_perturbations // len(top_mixes))
    perturbations = []

    rng = np.random.RandomState(42)

    for mix_pct, _ in top_mixes:
        for _ in range(per_archetype):
            perturbed = dict(mix_pct)
            # Perturb resource allocations by ±2%
            for res in ['clean_firm', 'solar', 'wind']:
                delta = rng.uniform(-2.0, 2.0)
                new_val = max(floor_pct.get(res, 0.0), perturbed[res] + delta)
                perturbed[res] = min(100.0, new_val)

            # Perturb hydro within cap
            hyd_cap = HYDRO_CAP_PCT.get('', 50.0)  # Will be overridden per-ISO
            perturbed['hydro'] = max(floor_pct.get('hydro', 0.0),
                                     min(hyd_cap, perturbed['hydro'] + rng.uniform(-1.0, 1.0)))

            # Perturb storage by ±0.5%
            for col in STORAGE_COLS:
                delta = rng.uniform(-0.5, 0.5)
                new_val = max(floor_pct.get(col, 0.0), perturbed[col] + delta)
                perturbed[col] = max(0, new_val)

            # Estimate match score (approximate — will be validated)
            perturbed['hourly_match_score'] = mix_pct.get('hourly_match_score', threshold)

            perturbations.append(perturbed)

    return pd.DataFrame(perturbations)


# ============================================================================
# ON-THE-FLY FLOOR-AWARE MIX GENERATION
# ============================================================================

MIN_FILTERED_MIXES = 1000  # Generate on-the-fly if fewer than this diverse mixes pass floor filter

def generate_floor_mixes(iso, floor_pct, threshold, demand_norm, supply_matrix,
                         step_resource=2, step_storage=1, max_add_resource=40, max_add_storage=15):
    """Generate mixes on-the-fly starting from the current ratcheted floor.

    When pre-computed PFS has too few mixes passing the floor filter at high
    thresholds, this generates a dense grid of incremental additions above
    the floor, scores them with batch_hourly_scores, and returns mixes in
    the target score range.

    Args:
        iso: ISO region
        floor_pct: dict of minimum resource allocations (% of demand)
        threshold: target threshold %
        demand_norm: normalized demand array (8760,)
        supply_matrix: (n_resources, 8760) supply profiles
        step_resource: grid step for solar/wind/CF additions (%)
        step_storage: grid step for storage additions (%)
        max_add_resource: max addition above floor for solar/wind (%)
        max_add_storage: max addition above floor for storage (%)

    Returns:
        DataFrame of scored mixes matching MIX_COLS schema
    """
    floor_cf = floor_pct.get('clean_firm', 0)
    floor_sol = floor_pct.get('solar', 0)
    floor_wnd = floor_pct.get('wind', 0)
    floor_hyd = floor_pct.get('hydro', 0)
    floor_osw = floor_pct.get('offshore_wind', 0)

    # Additions above floor
    cf_adds = np.arange(0, max_add_resource // 2 + step_resource, step_resource, dtype=np.float64)
    sol_adds = np.arange(0, max_add_resource + step_resource, step_resource, dtype=np.float64)
    wnd_adds = np.arange(0, max_add_resource + step_resource, step_resource, dtype=np.float64)

    # Offshore wind additions (only for offshore ISOs)
    if iso in OFFSHORE_ISOS and floor_osw < 30:
        osw_adds = np.arange(0, min(20, max_add_resource) + step_resource, step_resource, dtype=np.float64)
    else:
        osw_adds = np.array([0.0])

    # Storage additions (battery 4hr, battery 8hr, LDES)
    batt_adds = np.arange(0, max_add_storage + step_storage, step_storage, dtype=np.float64)
    batt8_adds = np.arange(0, max_add_storage + step_storage, step_storage * 2, dtype=np.float64)
    ldes_adds = np.arange(0, min(10, max_add_storage) + step_storage, step_storage * 2, dtype=np.float64)

    # For very high thresholds (>95%), use finer grid but smaller range
    if threshold >= 95:
        step_resource = max(1, step_resource)
        cf_adds = np.arange(0, 15 + step_resource, step_resource, dtype=np.float64)
        sol_adds = np.arange(0, 25 + step_resource, step_resource, dtype=np.float64)
        wnd_adds = np.arange(0, 25 + step_resource, step_resource, dtype=np.float64)
        batt_adds = np.arange(0, 20 + 1, 1, dtype=np.float64)
        batt8_adds = np.arange(0, 15 + 1, 1, dtype=np.float64)
        ldes_adds = np.arange(0, 10 + 1, 1, dtype=np.float64)

    n_resource_combos = len(cf_adds) * len(sol_adds) * len(wnd_adds) * len(osw_adds)
    n_storage_combos = len(batt_adds) * len(batt8_adds) * len(ldes_adds)

    # Cap total combos to avoid OOM — reduce grid if needed
    MAX_COMBOS = 5_000_000
    total_combos = n_resource_combos * n_storage_combos
    if total_combos > MAX_COMBOS:
        # Coarsen resource grid
        cf_adds = cf_adds[::2]
        sol_adds = sol_adds[::2]
        wnd_adds = wnd_adds[::2]
        n_resource_combos = len(cf_adds) * len(sol_adds) * len(wnd_adds) * len(osw_adds)
        total_combos = n_resource_combos * n_storage_combos
    if total_combos > MAX_COMBOS:
        # Coarsen storage grid too
        batt_adds = batt_adds[::2]
        batt8_adds = batt8_adds[::2]
        ldes_adds = ldes_adds[::2]
        n_storage_combos = len(batt_adds) * len(batt8_adds) * len(ldes_adds)
        total_combos = n_resource_combos * n_storage_combos

    print(f"      On-the-fly generation: {total_combos:,} combos "
          f"({len(cf_adds)}cf × {len(sol_adds)}sol × {len(wnd_adds)}wnd × "
          f"{len(osw_adds)}osw × {len(batt_adds)}b4 × {len(batt8_adds)}b8 × {len(ldes_adds)}ldes)")

    # Step 1: Generate resource grid (without storage first) and score
    grids_r = np.meshgrid(cf_adds, sol_adds, wnd_adds, osw_adds, indexing='ij')
    flat_r = [g.ravel() for g in grids_r]
    N_r = len(flat_r[0])

    cf_vals = floor_cf + flat_r[0]
    sol_vals = floor_sol + flat_r[1]
    wnd_vals = floor_wnd + flat_r[2]
    osw_vals = floor_osw + flat_r[3]
    hyd_vals = np.full(N_r, floor_hyd)

    # CCS residual
    explicit_sum = cf_vals + sol_vals + wnd_vals + hyd_vals + osw_vals
    ccs_vals = np.maximum(0, 100.0 - explicit_sum)

    # Build mix_batch for resource-only scoring (RESOURCE_TYPES order)
    # ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro']
    mix_batch = np.zeros((N_r, supply_matrix.shape[0]), dtype=np.float64)
    mix_batch[:, 0] = cf_vals
    mix_batch[:, 1] = sol_vals
    mix_batch[:, 2] = wnd_vals
    mix_batch[:, 3] = osw_vals
    mix_batch[:, 4] = ccs_vals
    mix_batch[:, 5] = hyd_vals

    # Score resource-only mixes
    scores_r = s1.batch_hourly_scores(demand_norm, supply_matrix, mix_batch, chunk_size=20000)
    total_demand = demand_norm.sum()
    if total_demand > 0:
        scores_r = scores_r / total_demand * 100.0

    # Filter to mixes near target (within reasonable range that storage can push up)
    # Storage can add ~5-15% to hourly match, so include mixes below threshold
    storage_headroom = max(5, min(20, max_add_storage))
    resource_mask = (scores_r >= threshold - storage_headroom) & (scores_r <= threshold + 5.0)

    # For mixes already at/above threshold, they don't need storage
    at_target = (scores_r >= threshold - 0.5) & (scores_r <= threshold + 1.0)

    # Collect resource-only mixes that hit threshold
    at_target_indices = np.where(at_target)[0]
    near_target_indices = np.where(resource_mask & ~at_target)[0]

    results = []

    # Add resource-only mixes that already hit threshold (no storage needed)
    if len(at_target_indices) > 0:
        for idx in at_target_indices[:2000]:  # Cap at 2000
            results.append({
                'clean_firm': cf_vals[idx],
                'solar': sol_vals[idx],
                'wind': wnd_vals[idx],
                'hydro': hyd_vals[idx],
                'offshore_wind': osw_vals[idx],
                'geothermal': 0.0,
                'battery_dispatch_pct': 0.0,
                'battery8_dispatch_pct': 0.0,
                'ldes_dispatch_pct': 0.0,
                'h2_dispatch_pct': 0.0,
                'hourly_match_score': scores_r[idx],
            })

    # Step 2: For near-target mixes, add storage combos
    # Sample up to 500 near-target resource mixes to pair with storage
    if len(near_target_indices) > 500:
        rng = np.random.RandomState(42)
        near_target_indices = rng.choice(near_target_indices, 500, replace=False)

    if len(near_target_indices) > 0 and n_storage_combos > 1:
        # Generate storage grid
        grids_s = np.meshgrid(batt_adds, batt8_adds, ldes_adds, indexing='ij')
        flat_s = [g.ravel() for g in grids_s]
        N_s = len(flat_s[0])

        # For each sampled resource mix, test storage combos
        # Use batch scoring: build combined resource+storage mixes
        for r_idx in near_target_indices:
            base_cf = cf_vals[r_idx]
            base_sol = sol_vals[r_idx]
            base_wnd = wnd_vals[r_idx]
            base_osw = osw_vals[r_idx]
            base_hyd = hyd_vals[r_idx]
            base_score = scores_r[r_idx]

            # Estimate storage contribution (heuristic: each % of battery adds ~0.5-1% match)
            # Only build full dispatch for promising combos
            for s_idx in range(N_s):
                b4 = floor_pct.get('battery_dispatch_pct', 0) + flat_s[0][s_idx]
                b8 = floor_pct.get('battery8_dispatch_pct', 0) + flat_s[1][s_idx]
                ld = floor_pct.get('ldes_dispatch_pct', 0) + flat_s[2][s_idx]

                # Rough estimate: storage adds ~0.5% per 1% dispatch
                est_boost = (b4 + b8 + ld) * 0.5
                est_score = base_score + est_boost

                if est_score >= threshold - 2.0 and est_score <= threshold + 5.0:
                    results.append({
                        'clean_firm': base_cf,
                        'solar': base_sol,
                        'wind': base_wnd,
                        'hydro': base_hyd,
                        'offshore_wind': base_osw,
                        'geothermal': 0.0,
                        'battery_dispatch_pct': b4,
                        'battery8_dispatch_pct': b8,
                        'ldes_dispatch_pct': ld,
                        'h2_dispatch_pct': 0.0,
                        'hourly_match_score': est_score,  # Approximate — will be refined
                    })

    if not results:
        print(f"      On-the-fly: 0 mixes generated in target range")
        return pd.DataFrame(columns=MIX_COLS)

    df = pd.DataFrame(results)

    # Ensure all MIX_COLS present
    for col in MIX_COLS:
        if col not in df.columns:
            df[col] = 0.0

    print(f"      On-the-fly: {len(df):,} mixes generated ({len(at_target_indices)} resource-only + "
          f"{len(results) - min(len(at_target_indices), 2000)} with storage)")

    return df


# ============================================================================
# VECTORIZED BATCH SCORING
# ============================================================================

def _vectorized_remaining_co2_rate(clean_pcts, iso, emission_rates, growth_factor):
    """Vectorized merit-order fossil retirement: returns remaining CO2 rate per MWh.

    Same logic as dispatch_utils.compute_fossil_retirement but operates on arrays.
    coal → oil → gas merit order. Returns remaining_rate_tco2_mwh for each clean_pct.
    """
    regional_data = emission_rates.get(iso, {})
    coal_rate = regional_data.get('coal_co2_lb_per_mwh', 0.0) / 2204.62
    gas_rate = regional_data.get('gas_co2_lb_per_mwh', 0.0) / 2204.62
    oil_rate = regional_data.get('oil_co2_lb_per_mwh', 0.0) / 2204.62

    base_demand_twh = REGIONAL_DEMAND_TWH.get(iso, 0)
    grown_demand_twh = base_demand_twh * growth_factor
    baseline_clean = sum(GRID_MIX_SHARES.get(iso, {}).values())

    coal_cap = COAL_CAP_TWH.get(iso, 0)
    oil_cap = OIL_CAP_TWH.get(iso, 0)

    N = len(clean_pcts)
    remaining_rates = np.full(N, gas_rate, dtype=np.float64)

    fossil_pct = (100.0 - clean_pcts) / 100.0
    fossil_twh = grown_demand_twh * fossil_pct

    # Merit-order: coal fills first, then oil, then gas
    coal_twh = np.minimum(coal_cap, fossil_twh)
    oil_twh = np.minimum(oil_cap, np.maximum(0, fossil_twh - coal_twh))
    gas_twh = np.maximum(0, fossil_twh - coal_twh - oil_twh)

    # Additional clean TWh displaces coal → oil → gas
    additional_clean_twh = np.maximum(0, (clean_pcts - baseline_clean) / 100.0 * grown_demand_twh)

    # For clean_pct >= 70 (COAL_OIL_RETIREMENT_THRESHOLD), all coal+oil is displaced
    from dispatch_utils import COAL_OIL_RETIREMENT_THRESHOLD
    high_mask = clean_pcts >= COAL_OIL_RETIREMENT_THRESHOLD

    # Low clean (< 70%): merit-order displacement
    coal_displaced = np.minimum(additional_clean_twh, coal_twh)
    remaining = additional_clean_twh - coal_displaced
    oil_displaced = np.minimum(remaining, oil_twh)
    remaining = remaining - oil_displaced
    gas_displaced = np.minimum(remaining, gas_twh)

    coal_remaining = coal_twh - coal_displaced
    oil_remaining = oil_twh - oil_displaced
    gas_remaining = gas_twh - gas_displaced
    fossil_remaining = coal_remaining + oil_remaining + gas_remaining

    # Remaining rate = weighted avg of remaining fossil
    valid = fossil_remaining > 0.01
    remaining_rates[valid] = (
        coal_remaining[valid] * coal_rate +
        oil_remaining[valid] * oil_rate +
        gas_remaining[valid] * gas_rate
    ) / fossil_remaining[valid]

    # High clean (>= 70%): all coal+oil retired, only gas remains
    remaining_rates[high_mask] = gas_rate

    # Fully clean: 0
    remaining_rates[fossil_twh <= 0.01] = 0.0

    return remaining_rates


def _precompute_nb_cost_params(iso, sens, target_year):
    """Precompute LCOE + TX parameters for vectorized new-build cost.

    Returns a dict of flat scalars that can be used in numpy operations.
    """
    ren_name = sens['ren']
    firm_lev = sens['firm']
    batt_name = sens['batt']
    ldes_name = sens['ldes_lvl']
    ccs_lev = sens['ccs']
    q45 = sens['q45']
    tx_name = sens['tx']
    geo_lev = sens.get('geo')

    params = {
        'sol_lcoe': LCOE_TABLES['solar'][ren_name][iso] + get_tx('solar', tx_name, iso),
        'wnd_lcoe': LCOE_TABLES['wind'][ren_name][iso] + get_tx('wind', tx_name, iso),
    }

    # Offshore wind
    if iso in OFFSHORE_ISOS:
        osw_base = LCOE_TABLES['offshore_wind'][ren_name][iso]
        osw_tx = get_tx('offshore_wind', tx_name, iso)
        foak = FOAK_OFFSHORE_WIND.get(iso)
        noak = NOAK_OFFSHORE_WIND.get(ren_name, {}).get(iso)
        tech_key = 'offshore_wind_float' if iso == 'CAISO' else 'offshore_wind_fixed'
        if foak and noak:
            osw_lcoe = _apply_learning_curve(osw_base, foak, noak, tech_key, firm_lev, target_year)
        else:
            osw_lcoe = osw_base
        params['osw_lcoe'] = osw_lcoe + osw_tx
    else:
        params['osw_lcoe'] = 0.0

    # Storage (net of revenue credits, with learning curves)
    for col, stype, slevel in [
        ('bat4', 'battery', batt_name), ('bat8', 'battery8', batt_name),
        ('ldes', 'ldes', ldes_name), ('h2', 'h2', ldes_name),
    ]:
        base_price = LCOE_TABLES[stype][slevel][iso]
        rev_credit = STORAGE_REVENUE_CREDITS[stype][iso]

        # Learning curve
        if stype == 'battery':
            foak_s, noak_s, tech_key_s = base_price, NOAK_BATTERY.get(slevel, {}).get(iso, base_price), 'bat4'
        elif stype == 'battery8':
            foak_s, noak_s, tech_key_s = base_price, NOAK_BATTERY8.get(slevel, {}).get(iso, base_price), 'bat8'
        elif stype == 'ldes':
            foak_s, noak_s, tech_key_s = FOAK_LDES.get(iso, base_price), LCOE_TABLES['ldes']['Low'][iso], 'ldes'
        elif stype == 'h2':
            foak_s, noak_s, tech_key_s = FOAK_H2.get(iso, base_price), LCOE_TABLES['h2']['Low'][iso], 'h2'
        else:
            foak_s, noak_s, tech_key_s = None, None, None

        if foak_s is not None and noak_s is not None:
            adj_price = _apply_learning_curve(base_price, foak_s, noak_s, tech_key_s,
                                               firm_lev, target_year)
            price = max(0, adj_price - rev_credit)
        else:
            price = max(0, base_price - rev_credit)
        params[f'{col}_price'] = price

    # Clean firm: use a blended estimate (cheapest of nuclear vs CCS at this LCOE level)
    # This is approximate — the full tranche model is applied on top-N candidates later.
    nuc_lcoe = NUCLEAR_NEWBUILD_LCOE[firm_lev][iso] + get_tx('clean_firm', tx_name, iso)
    foak_nuc = FOAK_NUCLEAR_NEWBUILD.get(iso)
    noak_nuc = NUCLEAR_NEWBUILD_LCOE['L'][iso]
    if foak_nuc and noak_nuc:
        nuc_lcoe_adj = _apply_learning_curve(NUCLEAR_NEWBUILD_LCOE[firm_lev][iso],
                                              foak_nuc, noak_nuc, 'nuclear', firm_lev, target_year)
        nuc_lcoe = nuc_lcoe_adj + get_tx('clean_firm', tx_name, iso)

    ccs_table = CCS_LCOE_45Q_ON if q45 == '1' else CCS_LCOE_45Q_OFF
    ccs_lcoe = ccs_table[ccs_lev][iso]
    if iso == 'NEISO':
        ccs_lcoe += NEISO_CCS_GAS_ADDER
    ccs_lcoe += get_tx('ccs_ccgt', tx_name, iso)
    foak_ccs = (FOAK_CCS_45Q_ON if q45 == '1' else FOAK_CCS_45Q_OFF).get(iso)
    noak_ccs = (CCS_LCOE_45Q_ON if q45 == '1' else CCS_LCOE_45Q_OFF)['L'][iso]
    if foak_ccs and noak_ccs:
        ccs_lcoe_adj = _apply_learning_curve(ccs_table[ccs_lev][iso],
                                              foak_ccs, noak_ccs, 'ccs', firm_lev, target_year)
        ccs_lcoe = ccs_lcoe_adj + get_tx('ccs_ccgt', tx_name, iso)
        if iso == 'NEISO':
            ccs_lcoe += NEISO_CCS_GAS_ADDER

    # Blended firm price: min(nuclear, CCS) as estimate for vectorized pass
    # The full tranche (uprate → geo → min) is applied in scalar refinement
    params['cf_lcoe_approx'] = min(nuc_lcoe, ccs_lcoe)

    # Geothermal (CAISO only)
    if iso == 'CAISO' and geo_lev:
        geo_base = GEOTHERMAL_LCOE[geo_lev]
        foak_geo = FOAK_GEOTHERMAL
        noak_geo = GEOTHERMAL_LCOE['L']
        geo_lcoe = _apply_learning_curve(geo_base, foak_geo, noak_geo, 'geo', firm_lev, target_year)
        geo_lcoe += get_tx('clean_firm', tx_name, iso)
        params['geo_lcoe'] = geo_lcoe
    else:
        params['geo_lcoe'] = 0.0

    return params


def batch_score_mixes(filtered_df, iso, sens, demand_twh, target_year,
                      deployed_twh, emission_rates, growth_factor,
                      baseline_co2_mt, baseline_effective_clean_pct):
    """Vectorized Phase 1 scoring: returns (mac_array, co2_avoided_array, nb_cost_array).

    Computes approximate new-build cost and CO2 avoided for all mixes simultaneously.
    Clean firm uses a blended LCOE estimate (no tranching) — top candidates get
    full scalar refinement in Phase 2.

    Returns arrays of shape (N,) or None if no valid mixes.
    """
    N = len(filtered_df)
    if N == 0:
        return None, None, None

    # Extract resource percentages as arrays
    cf_pct = filtered_df['clean_firm'].values.astype(np.float64)
    sol_pct = filtered_df['solar'].values.astype(np.float64)
    wnd_pct = filtered_df['wind'].values.astype(np.float64)
    hyd_pct = filtered_df['hydro'].values.astype(np.float64)
    osw_pct = filtered_df['offshore_wind'].values.astype(np.float64)
    geo_pct = filtered_df['geothermal'].values.astype(np.float64) if 'geothermal' in filtered_df.columns else np.zeros(N)
    bat4_pct = filtered_df['battery_dispatch_pct'].values.astype(np.float64)
    bat8_pct = filtered_df['battery8_dispatch_pct'].values.astype(np.float64)
    ldes_pct = filtered_df['ldes_dispatch_pct'].values.astype(np.float64)
    h2_pct = filtered_df['h2_dispatch_pct'].values.astype(np.float64) if 'h2_dispatch_pct' in filtered_df.columns else np.zeros(N)
    scores = filtered_df['hourly_match_score'].values.astype(np.float64)

    # Cap hydro at physical limit
    hydro_cap_pct = HYDRO_CAP_PCT.get(iso, 50.0)
    hydro_excess = np.maximum(0, hyd_pct - hydro_cap_pct)
    hyd_pct = np.minimum(hyd_pct, hydro_cap_pct)
    scores = np.maximum(0, scores - hydro_excess)

    # --- CO2 avoided (vectorized) ---
    cand_clean_pct = np.minimum(scores, 99.99)
    remaining_rates = _vectorized_remaining_co2_rate(
        cand_clean_pct, iso, emission_rates, growth_factor)
    fossil_twh_after = np.maximum(0, demand_twh * (100.0 - cand_clean_pct) / 100.0)
    co2_after_mt = fossil_twh_after * 1e6 * remaining_rates / 1e6
    co2_avoided_mt = baseline_co2_mt - co2_after_mt

    # --- New-build cost (vectorized, approximate for clean firm) ---
    params = _precompute_nb_cost_params(iso, sens, target_year)

    # Existing resources (floor = deployed_twh for cost purposes)
    existing_sol_twh = deployed_twh.get('solar', 0.0)
    existing_wnd_twh = deployed_twh.get('wind', 0.0)
    existing_osw_twh = deployed_twh.get('offshore_wind', 0.0)
    existing_cf_twh = deployed_twh.get('clean_firm', 0.0)
    existing_geo_twh = EXISTING_GEOTHERMAL_PCT / 100.0 * demand_twh if iso == 'CAISO' else 0.0
    floor_bat4 = deployed_twh.get('battery_dispatch_pct', 0.0)
    floor_bat8 = deployed_twh.get('battery8_dispatch_pct', 0.0)
    floor_ldes = deployed_twh.get('ldes_dispatch_pct', 0.0)
    floor_h2 = deployed_twh.get('h2_dispatch_pct', 0.0)

    # New-build TWh = max(0, mix_twh - existing_twh)
    sol_new_twh = np.maximum(0, sol_pct / 100.0 * demand_twh - existing_sol_twh)
    wnd_new_twh = np.maximum(0, wnd_pct / 100.0 * demand_twh - existing_wnd_twh)
    osw_new_twh = np.maximum(0, osw_pct / 100.0 * demand_twh - existing_osw_twh)
    cf_new_twh = np.maximum(0, cf_pct / 100.0 * demand_twh - existing_cf_twh)

    # New storage = max(0, pct - floor_pct)
    bat4_new = np.maximum(0, bat4_pct - floor_bat4)
    bat8_new = np.maximum(0, bat8_pct - floor_bat8)
    ldes_new = np.maximum(0, ldes_pct - floor_ldes)
    h2_new = np.maximum(0, h2_pct - floor_h2)

    # Geothermal new-build (CAISO only, physics dimension)
    geo_new_twh = np.zeros(N)
    if iso == 'CAISO' and params['geo_lcoe'] > 0:
        geo_new_twh = np.maximum(0, geo_pct / 100.0 * demand_twh - existing_geo_twh)

    # Total new-build cost ($)
    nb_cost = (
        sol_new_twh * 1e6 * params['sol_lcoe'] +
        wnd_new_twh * 1e6 * params['wnd_lcoe'] +
        osw_new_twh * 1e6 * params['osw_lcoe'] +
        cf_new_twh * 1e6 * params['cf_lcoe_approx'] +
        geo_new_twh * 1e6 * params['geo_lcoe'] +
        bat4_new / 100.0 * params['bat4_price'] * demand_twh * 1e6 +
        bat8_new / 100.0 * params['bat8_price'] * demand_twh * 1e6 +
        ldes_new / 100.0 * params['ldes_price'] * demand_twh * 1e6 +
        h2_new / 100.0 * params['h2_price'] * demand_twh * 1e6
    )

    # MAC = nb_cost / (co2_avoided * 1e6)
    valid = co2_avoided_mt > 0.001
    mac = np.full(N, np.inf)
    mac[valid] = nb_cost[valid] / (co2_avoided_mt[valid] * 1e6)

    return mac, co2_avoided_mt, nb_cost


# ============================================================================
# SINGLE THRESHOLD OPTIMIZER
# ============================================================================

def optimize_threshold(iso, threshold, floor_twh, deployed_twh, cumulative_caps,
                       sens, demand_twh, target_year,
                       demand_norm, supply_profiles, supply_matrix, emission_rates,
                       existing_clean_twh, prev_threshold=None,
                       existing_clean_hourly_pct=None):
    """Optimize a single threshold step.

    Two-pass architecture:
      Phase 1 (vectorized): Score ALL mixes with numpy — approximate clean firm
              cost (blended LCOE, no tranching). Selects top 50 candidates.
      Phase 2 (scalar): Re-score top candidates with full tranche model
              (uprate → geo → min(nuclear, CCS)) for accurate cost ranking.
      Phase 3 (dispatch): 8760-hour CO2 rescore on top 10 for final selection.

    Args:
        floor_twh: Filter floor (absolute TWh, only ratchets on % increase).
        deployed_twh: Cost floor (absolute TWh, always max).
    """
    # Convert floor to %
    floor_pct = floor_to_pct(floor_twh, demand_twh)

    # 1. Compute CO2 baseline (path-dependent)
    growth_factor = demand_twh / REGIONAL_DEMAND_TWH[iso]

    if prev_threshold is not None:
        baseline_effective_clean_pct = prev_threshold
    else:
        baseline_effective_clean_pct = existing_clean_hourly_pct if existing_clean_hourly_pct else sum(GRID_MIX_SHARES[iso].values())
    baseline_effective_clean_pct = min(baseline_effective_clean_pct, 99.99)

    _, retirement_info = compute_fossil_retirement(
        iso, baseline_effective_clean_pct, emission_rates, {},
        demand_growth_factor=growth_factor)
    remaining_rate = retirement_info.get('remaining_rate_tco2_mwh', 0.3911)
    fossil_twh_baseline = max(0, demand_twh * (100.0 - baseline_effective_clean_pct) / 100.0)
    baseline_co2_mt = fossil_twh_baseline * 1e6 * remaining_rate / 1e6

    # 2. Load PFS archetypes
    pfs_df = load_pfs_for_threshold(iso, threshold)

    if not pfs_df.empty:
        filtered = filter_and_sample(pfs_df, floor_pct, threshold)
    else:
        filtered = pd.DataFrame(columns=MIX_COLS)

    # 2b. On-the-fly enhancement if too few mixes pass floor filter
    if len(filtered) < MIN_FILTERED_MIXES:
        otf_df = generate_floor_mixes(iso, floor_pct, threshold,
                                       demand_norm, supply_matrix)
        if not otf_df.empty:
            scores = otf_df['hourly_match_score'].values
            score_mask = (scores >= threshold) & (scores <= threshold + MAX_OVERSHOOT + 0.5)
            otf_filtered = otf_df.loc[score_mask].copy()

            if not otf_filtered.empty:
                if len(otf_filtered) > MAX_ARCHETYPES:
                    otf_filtered = otf_filtered.sample(n=MAX_ARCHETYPES, random_state=42)
                filtered = pd.concat([filtered, otf_filtered], ignore_index=True)
                dedup_cols = RESOURCE_COLS + STORAGE_COLS
                filtered_rounded = filtered.copy()
                for c in dedup_cols:
                    if c in filtered_rounded.columns:
                        filtered_rounded[c] = filtered_rounded[c].round(2)
                filtered = filtered.loc[filtered_rounded.drop_duplicates(subset=dedup_cols).index].reset_index(drop=True)

    if filtered.empty:
        print(f"    WARNING: No mixes pass floor filter for {iso} t{threshold:g}")
        return None, floor_twh, deployed_twh, cumulative_caps

    print(f"    {iso} t{threshold:g}: {len(pfs_df)} PFS mixes → {len(filtered)} after filtering")

    # ── Phase 1: Vectorized batch scoring (approximate clean firm cost) ──
    mac_arr, co2_avoided_arr, nb_cost_arr = batch_score_mixes(
        filtered, iso, sens, demand_twh, target_year,
        deployed_twh, emission_rates, growth_factor,
        baseline_co2_mt, baseline_effective_clean_pct)

    if mac_arr is None or not np.any(np.isfinite(mac_arr)):
        print(f"    WARNING: No valid MAC found for {iso} t{threshold:g} (batch)")
        return None, floor_twh, deployed_twh, cumulative_caps

    # Select top 50 candidates for scalar refinement
    TOP_N_REFINE = 50
    TOP_N_DISPATCH = 10
    finite_mask = np.isfinite(mac_arr)
    n_finite = int(np.sum(finite_mask))
    if n_finite <= TOP_N_REFINE:
        top_indices = np.where(finite_mask)[0]
    else:
        # Partial sort: get indices of the TOP_N_REFINE smallest MACs
        top_indices = np.argpartition(mac_arr, TOP_N_REFINE)[:TOP_N_REFINE]

    # ── Phase 2: Scalar refinement with full tranche model on top candidates ──
    best_mac = float('inf')
    best_mix = None
    best_cost = 0
    best_co2_avoided = 0
    best_co2_after = 0
    best_breakdown = {}
    top_candidates = []

    for idx in top_indices:
        row = filtered.iloc[idx]
        mix_pct = mix_row_to_pct(row)
        cap_hydro_in_mix(mix_pct, iso)

        # Full scalar cost with tranching
        candidate_clean_pct = min(mix_pct.get('hourly_match_score', 0), 99.99)
        _, cand_info = compute_fossil_retirement(
            iso, candidate_clean_pct, emission_rates, {},
            demand_growth_factor=growth_factor)
        cand_remaining_rate = cand_info.get('remaining_rate_tco2_mwh', 0.3911)
        fossil_twh_after = max(0, demand_twh * (100.0 - candidate_clean_pct) / 100.0)
        co2_after_mt = fossil_twh_after * 1e6 * cand_remaining_rate / 1e6
        co2_avoided_mt = baseline_co2_mt - co2_after_mt
        if co2_avoided_mt <= 0.001:
            continue

        caps_copy = dict(cumulative_caps)
        nb_cost, nb_breakdown, _ = compute_new_build_cost(
            iso, mix_pct, deployed_twh, demand_twh, sens, target_year, caps_copy)
        mac = nb_cost / (co2_avoided_mt * 1e6)

        if len(top_candidates) < TOP_N_DISPATCH or mac < top_candidates[-1][1]:
            top_candidates.append((mix_pct, mac, nb_cost, nb_breakdown))
            top_candidates.sort(key=lambda x: x[1])
            top_candidates = top_candidates[:TOP_N_DISPATCH]

        if mac < best_mac:
            best_mac = mac
            best_mix = mix_pct
            best_cost = nb_cost
            best_co2_avoided = co2_avoided_mt
            best_co2_after = co2_after_mt
            best_breakdown = nb_breakdown

    # Also refine Phase 2 perturbations around top archetypes
    top_archetypes = [(m, s) for m, s, _, _ in top_candidates[:5]]
    if top_archetypes and len(filtered) > 10:
        phase2_df = phase2_refine(top_archetypes, floor_pct, threshold)
        if not phase2_df.empty:
            # Batch-score perturbations first, then refine top
            p2_mac, p2_co2, p2_cost = batch_score_mixes(
                phase2_df, iso, sens, demand_twh, target_year,
                deployed_twh, emission_rates, growth_factor,
                baseline_co2_mt, baseline_effective_clean_pct)
            if p2_mac is not None:
                p2_finite = np.isfinite(p2_mac)
                n_p2 = int(np.sum(p2_finite))
                if n_p2 > 0:
                    p2_top_n = min(20, n_p2)
                    if n_p2 <= p2_top_n:
                        p2_top_idx = np.where(p2_finite)[0]
                    else:
                        p2_top_idx = np.argpartition(p2_mac, p2_top_n)[:p2_top_n]
                    for pidx in p2_top_idx:
                        row = phase2_df.iloc[pidx]
                        mix_pct = mix_row_to_pct(row)
                        cap_hydro_in_mix(mix_pct, iso)
                        ccp = min(mix_pct.get('hourly_match_score', 0), 99.99)
                        _, ci = compute_fossil_retirement(
                            iso, ccp, emission_rates, {}, demand_growth_factor=growth_factor)
                        crr = ci.get('remaining_rate_tco2_mwh', 0.3911)
                        fta = max(0, demand_twh * (100.0 - ccp) / 100.0)
                        ca_mt = fta * 1e6 * crr / 1e6
                        ca_avoided = baseline_co2_mt - ca_mt
                        if ca_avoided <= 0.001:
                            continue
                        caps_copy = dict(cumulative_caps)
                        nbc, nbb, _ = compute_new_build_cost(
                            iso, mix_pct, deployed_twh, demand_twh, sens, target_year, caps_copy)
                        m = nbc / (ca_avoided * 1e6)
                        if len(top_candidates) < TOP_N_DISPATCH or m < top_candidates[-1][1]:
                            top_candidates.append((mix_pct, m, nbc, nbb))
                            top_candidates.sort(key=lambda x: x[1])
                            top_candidates = top_candidates[:TOP_N_DISPATCH]
                        if m < best_mac:
                            best_mac, best_mix, best_cost = m, mix_pct, nbc
                            best_co2_avoided, best_co2_after = ca_avoided, ca_mt
                            best_breakdown = nbb

    # ── Phase 3: 8760-hour dispatch rescore on top candidates ──
    if top_candidates:
        dispatch_best_mac = float('inf')
        dispatch_best_mix = None
        dispatch_best_cost = 0
        dispatch_best_co2_avoided = 0
        dispatch_best_co2_after = 0
        dispatch_best_breakdown = {}
        dispatch_best_co2_detail = {}

        for mix_pct, scalar_mac, _, _ in top_candidates:
            resource_pcts = {res: mix_pct.get(res, 0) for res in RESOURCE_COLS}
            dispatch_result = reconstruct_hourly_dispatch(
                demand_norm, supply_profiles, resource_pcts,
                procurement_pct=100,
                battery_dispatch_pct=mix_pct.get('battery_dispatch_pct', 0),
                battery8_dispatch_pct=mix_pct.get('battery8_dispatch_pct', 0),
                ldes_dispatch_pct=mix_pct.get('ldes_dispatch_pct', 0),
                supply_matrix=supply_matrix,
                h2_dispatch_pct=mix_pct.get('h2_dispatch_pct', 0),
            )
            co2_result = compute_co2_from_dispatch(
                iso, dispatch_result, emission_rates, demand_twh * 1e6)

            # Marginal CO2 via scalar retirement (consistent with Phase 2)
            ccp_d = min(mix_pct.get('hourly_match_score', 0), 99.99)
            _, ci_d = compute_fossil_retirement(
                iso, ccp_d, emission_rates, {}, demand_growth_factor=growth_factor)
            crr_d = ci_d.get('remaining_rate_tco2_mwh', 0.3911)
            fta_d = max(0, demand_twh * (100.0 - ccp_d) / 100.0)
            co2_after_mt = fta_d * 1e6 * crr_d / 1e6
            co2_avoided_mt = baseline_co2_mt - co2_after_mt

            if co2_avoided_mt <= 0.001:
                continue

            caps_copy = dict(cumulative_caps)
            nb_cost, nb_breakdown, _ = compute_new_build_cost(
                iso, mix_pct, deployed_twh, demand_twh, sens, target_year, caps_copy)
            d_mac = nb_cost / (co2_avoided_mt * 1e6)

            if d_mac < dispatch_best_mac:
                dispatch_best_mac = d_mac
                dispatch_best_mix = mix_pct
                dispatch_best_cost = nb_cost
                dispatch_best_co2_avoided = co2_avoided_mt
                dispatch_best_co2_after = co2_after_mt
                dispatch_best_breakdown = nb_breakdown
                dispatch_best_co2_detail = co2_result

        if dispatch_best_mix is not None:
            best_mac = dispatch_best_mac
            best_mix = dispatch_best_mix
            best_cost = dispatch_best_cost
            best_co2_avoided = dispatch_best_co2_avoided
            best_co2_after = dispatch_best_co2_after
            best_breakdown = dispatch_best_breakdown
            print(f"      Dispatch rescore: MAC ${dispatch_best_mac:.1f}/tCO2 "
                  f"(coal {dispatch_best_co2_detail.get('coal_displaced_twh', 0):.1f} TWh, "
                  f"gas {dispatch_best_co2_detail.get('gas_displaced_twh', 0):.1f} TWh)")

    if best_mix is None:
        print(f"    WARNING: No valid MAC found for {iso} t{threshold:g}")
        return None, floor_twh, deployed_twh, cumulative_caps

    # 7. Update cumulative caps with winner (uses deployed_twh for accurate costing)
    _, _, updated_caps = compute_new_build_cost(
        iso, best_mix, deployed_twh, demand_twh, sens, target_year, dict(cumulative_caps))

    # 8. Ratchet floors
    # Filter floor: only ratchets when winner % > floor % (widens search at next threshold)
    new_floor = ratchet_floor(floor_twh, best_mix, demand_twh)
    # Deployed floor: always takes max (tracks actual physical capacity for costing)
    new_deployed = ratchet_deployed(deployed_twh, best_mix, demand_twh)

    # Compute new-build MWh for $/MWh reference (uses deployed_twh for accuracy)
    total_new_build_twh = 0.0
    for res in RESOURCE_COLS:
        nb_twh = max(0, best_mix.get(res, 0) / 100.0 * demand_twh - deployed_twh.get(res, 0))
        total_new_build_twh += nb_twh

    # 9. Build result
    result = {
        'iso': iso,
        'threshold': threshold,
        'target_year': target_year,
        'demand_twh': round(demand_twh, 3),
        'baseline_clean_pct': round(baseline_effective_clean_pct, 2),
        'baseline_co2_mt': round(baseline_co2_mt, 4),
        'co2_after_mt': round(best_co2_after, 4),
        'co2_avoided_mt': round(best_co2_avoided, 4),
        'new_build_cost_total': round(best_cost, 0),
        'new_build_twh_total': round(total_new_build_twh, 3),
        'new_build_cost_per_mwh_nb': round(best_cost / (total_new_build_twh * 1e6) if total_new_build_twh > 0.001 else 0, 2),
        'mac_this_step': round(best_mac, 2),
        'winner_match_score': round(best_mix.get('hourly_match_score', 0), 2),
    }

    # Winner mix
    for res in RESOURCE_COLS:
        result[f'winner_{res}_pct'] = round(best_mix.get(res, 0), 3)
        result[f'winner_{res}_twh'] = round(best_mix.get(res, 0) / 100.0 * demand_twh, 3)
    for col in STORAGE_COLS:
        result[f'winner_{col}'] = round(best_mix.get(col, 0), 3)

    # New build incremental (uses deployed_twh for accurate costing)
    for res in RESOURCE_COLS:
        nb_twh = max(0, best_mix.get(res, 0) / 100.0 * demand_twh - deployed_twh.get(res, 0))
        result[f'new_build_{res}_twh'] = round(nb_twh, 3)
    for col in STORAGE_COLS:
        nb_pct = max(0, best_mix.get(col, 0) - deployed_twh.get(col, 0))
        result[f'new_build_{col}'] = round(nb_pct, 3)

    # Floor state (record both filter floor and deployed floor)
    for res in RESOURCE_COLS:
        result[f'floor_{res}_twh'] = round(new_floor.get(res, 0), 3)
        result[f'deployed_{res}_twh'] = round(new_deployed.get(res, 0), 3)
    for col in STORAGE_COLS:
        result[f'floor_{col}'] = round(new_floor.get(col, 0), 3)
        result[f'deployed_{col}'] = round(new_deployed.get(col, 0), 3)

    print(f"    → MAC: ${best_mac:.1f}/tCO2 | CO2 avoided: {best_co2_avoided:.3f} Mt "
          f"| NB cost: ${best_cost/1e6:.1f}M")

    return result, new_floor, new_deployed, updated_caps


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pathway(iso, price_sens_name, growth_level, demand_norm, supply_profiles,
                supply_matrix, emission_rates):
    """Run full MAC queue for one ISO × price_sensitivity × demand_growth pathway."""
    sens = PRICE_SENSITIVITIES[price_sens_name]
    growth_rate = DEMAND_GROWTH_RATES[iso][growth_level]
    base_demand = REGIONAL_DEMAND_TWH[iso]

    # Initialize floors from existing clean
    # filter floor: only ratchets on % increase → diminishes as demand grows
    # deployed floor: always max → tracks actual physical capacity for costing
    floor_twh = init_floor(iso)
    deployed_twh = dict(floor_twh)  # Starts identical to filter floor
    existing_clean_twh = dict(floor_twh)
    cumulative_caps = {'uprate_twh': 0.0, 'geo_twh': 0.0, 'ccs_twh': 0.0}
    cumulative_cost = 0.0
    cumulative_co2_avoided = 0.0

    # Compute existing clean HOURLY match score (not annual energy share!)
    # This accounts for solar curtailment — CAISO annual ~48% but hourly ~39.6%
    existing_mix = np.zeros((1, supply_matrix.shape[0]))
    shares = GRID_MIX_SHARES[iso]
    from dispatch_utils import RESOURCE_TYPES
    for i, rtype in enumerate(RESOURCE_TYPES):
        existing_mix[0, i] = shares.get(rtype, 0.0)
    # CCS residual: the existing fossil fleet (not clean, don't include)
    # Zero out CCS column — only explicit clean resources count
    ccs_idx = RESOURCE_TYPES.index('ccs_ccgt') if 'ccs_ccgt' in RESOURCE_TYPES else -1
    if ccs_idx >= 0:
        existing_mix[0, ccs_idx] = 0.0
    supply_existing = (existing_mix / 100.0) @ supply_matrix
    matched = np.minimum(demand_norm, supply_existing)
    existing_clean_hourly_pct = (matched.sum() / demand_norm.sum()) * 100.0
    # Annual energy share for reference
    existing_clean_annual_pct = sum(shares.values())
    print(f"    {iso} existing clean: {existing_clean_hourly_pct:.1f}% hourly match "
          f"(vs {existing_clean_annual_pct:.1f}% annual share)")

    results = []
    prev_threshold = None  # Track previous threshold for path-dependent CO2 baseline

    for threshold in MAC_THRESHOLDS:
        if threshold <= existing_clean_hourly_pct:
            continue

        target_year = THRESHOLD_TARGET_YEARS.get(threshold, 2050)
        years = max(0, target_year - 2025)
        demand_twh = base_demand * (1 + growth_rate) ** years

        result, floor_twh, deployed_twh, cumulative_caps = optimize_threshold(
            iso, threshold, floor_twh, deployed_twh, cumulative_caps,
            sens, demand_twh, target_year,
            demand_norm, supply_profiles, supply_matrix, emission_rates,
            existing_clean_twh, prev_threshold=prev_threshold,
            existing_clean_hourly_pct=existing_clean_hourly_pct)

        if result is None:
            continue

        # Update prev_threshold for path-dependent CO2 baseline
        prev_threshold = threshold

        # Cumulative tracking
        cumulative_cost += result['new_build_cost_total']
        cumulative_co2_avoided += result['co2_avoided_mt']
        result['price_sensitivity'] = price_sens_name
        result['demand_growth'] = growth_level
        result['cumulative_new_build_cost'] = round(cumulative_cost, 0)
        result['cumulative_co2_avoided_mt'] = round(cumulative_co2_avoided, 4)
        result['cumulative_mac'] = round(
            cumulative_cost / (cumulative_co2_avoided * 1e6) if cumulative_co2_avoided > 0 else 0, 2)

        results.append(result)

    return results


def run_iso(iso, demand_data, gen_profiles, emission_rates):
    """Run all 15 pathways for a single ISO."""
    print(f"\n{'='*60}")
    print(f"  Processing {iso}")
    print(f"{'='*60}")

    # Load profiles
    demand_norm, total_mwh = get_demand_profile(iso, demand_data)
    supply_profiles = get_supply_profiles(iso, gen_profiles)
    supply_matrix = build_supply_matrix(supply_profiles)

    all_results = []

    for growth_level in DEMAND_GROWTH_LEVELS:
        for price_sens_name in PRICE_SENSITIVITIES:
            print(f"\n  {iso} | {price_sens_name} | {growth_level} growth")
            print(f"  {'-'*50}")

            pathway_results = run_pathway(
                iso, price_sens_name, growth_level,
                demand_norm, supply_profiles, supply_matrix, emission_rates)

            all_results.extend(pathway_results)

    return all_results


def _build_consequential_queue(all_results):
    """Convert MAC queue results to consequential_queue.json format for step8a.

    Uses the 'all_med' sensitivity + 'Medium' growth canonical pathway.
    Builds zone-step entries sorted by marginal_mac with delta_resources.
    Falls back to any available sensitivity if all_med is missing.
    """
    # Filter to canonical pathway
    canonical = [r for r in all_results
                 if r.get('price_sensitivity') == 'all_med'
                 and r.get('demand_growth') == 'Medium']
    if not canonical:
        # Fallback: use whatever sensitivity/growth is available
        canonical = all_results

    # Group by ISO, sort by threshold
    from collections import defaultdict
    iso_results = defaultdict(list)
    for r in canonical:
        iso_results[r['iso']].append(r)
    for iso in iso_results:
        iso_results[iso].sort(key=lambda x: x['threshold'])

    queue = []
    for iso, results in iso_results.items():
        prev = None
        for r in results:
            # Compute delta resources vs previous threshold
            delta_resources = {}
            res_keys = ['solar', 'wind', 'clean_firm', 'hydro', 'offshore_wind', 'geothermal']
            for res in res_keys:
                twh_key = f'winner_{res}_twh'
                curr_twh = r.get(twh_key, 0)
                prev_twh = prev.get(twh_key, 0) if prev else 0
                delta = curr_twh - prev_twh
                if abs(delta) > 0.5:
                    delta_resources[res] = round(delta, 2)

            # Storage deltas
            for stor_key, out_key in [('winner_battery_4h', 'battery'),
                                       ('winner_battery_8h', 'battery8'),
                                       ('winner_ldes_100h', 'ldes')]:
                curr = r.get(stor_key, 0)
                prev_val = prev.get(stor_key, 0) if prev else 0
                delta = curr - prev_val
                if abs(delta) > 0.01:
                    delta_resources[out_key] = round(delta, 3)

            queue.append({
                'iso': iso,
                'zone_label': f"{prev['threshold'] if prev else 'existing'}→{r['threshold']}%",
                'threshold_start': prev['threshold'] if prev else 0,
                'threshold_end': r['threshold'],
                'marginal_mac': r.get('mac_this_step', 9999),
                'co2_displaced_mt': r.get('co2_avoided_mt', 0),
                'delta_cost_per_mwh': round(r.get('new_build_cost_per_mwh_nb', 0), 2),
                'delta_resources': delta_resources,
                'demand_twh': r.get('demand_twh', 0),
                'target_year': r.get('target_year', 2050),
            })
            prev = r

    # Sort by marginal MAC (cheapest first) for cross-ISO merit order
    queue.sort(key=lambda x: (x['marginal_mac'], -x['co2_displaced_mt']))
    for i, step in enumerate(queue):
        step['queue_position'] = i + 1

    return queue


_BASELINE_NET_PEAK_CACHE = {}


def _compute_baseline_net_peak(iso, demand_norm, supply_profiles, supply_matrix):
    """Compute 2025 baseline net peak (MW) from existing mix dispatch.

    Used for delta calibration: at 2025 baseline, gas = EXISTING_GAS_CAPACITY_MW.
    """
    if iso in _BASELINE_NET_PEAK_CACHE:
        return _BASELINE_NET_PEAK_CACHE[iso]

    baseline_pcts = {k: v for k, v in GRID_MIX_SHARES[iso].items()}
    result = reconstruct_hourly_dispatch(
        demand_norm, supply_profiles, baseline_pcts,
        procurement_pct=100,
        battery_dispatch_pct=0, battery8_dispatch_pct=0,
        ldes_dispatch_pct=0, h2_dispatch_pct=0,
        supply_matrix=supply_matrix,
    )
    # residual_demand is in fraction-of-annual-energy units; multiply by
    # total annual MWh to get MW (since each value = 1 hour)
    demand_mwh = BASE_DEMAND_TWH[iso] * 1e6
    baseline_net_peak = float(np.max(result['residual_demand'])) * demand_mwh
    _BASELINE_NET_PEAK_CACHE[iso] = baseline_net_peak
    return baseline_net_peak


def _compute_gas_backup(iso, mix_pcts, demand_twh, growth_factor,
                        demand_norm, supply_profiles, supply_matrix):
    """Compute gas backup capacity and cost using dispatch-based net peak hour.

    Uses reconstruct_hourly_dispatch() to find the actual worst-coverage hour,
    then applies RA margin and GAF. Delta-calibrated to 2025 baseline so that
    at baseline, total gas = EXISTING_GAS_CAPACITY_MW.

    Returns (gas_backup_mw, new_gas_mw, existing_gas_used_mw, net_peak_mw, gas_cost_per_mwh).
    """
    demand_mwh = demand_twh * 1e6

    # Build resource_pcts dict for dispatch (generation resources only)
    resource_pcts = {
        'clean_firm': mix_pcts.get('clean_firm', 0),
        'solar': mix_pcts.get('solar', 0),
        'wind': mix_pcts.get('wind', 0),
        'offshore_wind': mix_pcts.get('offshore_wind', 0),
        'ccs_ccgt': mix_pcts.get('ccs_ccgt', 0),
        'hydro': mix_pcts.get('hydro', 0),
    }

    bat4_pct = mix_pcts.get('battery_dispatch_pct', 0)
    bat8_pct = mix_pcts.get('battery8_dispatch_pct', 0)
    ldes_pct = mix_pcts.get('ldes_dispatch_pct', 0)
    h2_pct = mix_pcts.get('h2_dispatch_pct', 0)

    # Run 8760-hour dispatch to find actual net peak residual
    result = reconstruct_hourly_dispatch(
        demand_norm, supply_profiles, resource_pcts,
        procurement_pct=100,
        battery_dispatch_pct=bat4_pct,
        battery8_dispatch_pct=bat8_pct,
        ldes_dispatch_pct=ldes_pct,
        h2_dispatch_pct=h2_pct,
        supply_matrix=supply_matrix,
    )

    # residual_demand is in fraction-of-annual-energy units; multiply by
    # total annual MWh to get MW (since each value = 1 hour)
    net_peak_norm = float(np.max(result['residual_demand']))
    net_peak_mw = net_peak_norm * demand_mwh

    # Apply RA margin and GAF
    gaf = GAS_AVAILABILITY_FACTOR[iso]
    gas_raw = net_peak_mw * (1 + RESOURCE_ADEQUACY_MARGIN) / gaf

    # Delta calibration: anchor to 2025 baseline
    baseline_net_peak = _compute_baseline_net_peak(
        iso, demand_norm, supply_profiles, supply_matrix)
    baseline_gas_raw = baseline_net_peak * (1 + RESOURCE_ADEQUACY_MARGIN) / gaf
    gas_delta = gas_raw - baseline_gas_raw

    existing_gas_mw = EXISTING_GAS_CAPACITY_MW[iso]
    gas_needed_mw = max(0, existing_gas_mw + gas_delta)
    existing_gas_used_mw = min(gas_needed_mw, existing_gas_mw)
    new_gas_mw = max(0, gas_needed_mw - existing_gas_used_mw)

    gas_cost = (
        existing_gas_used_mw * EXISTING_GAS_FOM_KW_YR[iso] * 1000 +
        new_gas_mw * NEW_CCGT_COST_KW_YR[iso] * 1000
    ) / demand_mwh

    return round(gas_needed_mw), round(new_gas_mw), round(existing_gas_used_mw), round(net_peak_mw), gas_cost


def _export_scenario_a(all_results, isos_processed, demand_data, gen_profiles,
                       sensitivity='high_firm_low_vre', growth='Medium'):
    """Export step5d MAC queue results as scenario_a_{ISO}.json for step7c compatibility.

    Step5d picks the winning mix at each threshold (MAC queue optimization).
    System cost is computed via compute_mix_cost() — the SAME cost function
    Scenario B uses — ensuring apples-to-apples comparison.

    Deployment-gated learning: FOAK pricing until first clean firm deployment
    in the MAC queue, then Wright's Law ramp from deployment year.

    MAC = step5d's own mac_this_step (authoritative, pass-through).
    """
    # Filter to target sensitivity + growth
    canonical = [r for r in all_results
                 if r.get('price_sensitivity') == sensitivity
                 and r.get('demand_growth') == growth]

    if not canonical:
        print(f"  WARNING: No results for {sensitivity}/{growth} — skipping scenario A export")
        return

    # Group by ISO
    from collections import defaultdict
    iso_groups = defaultdict(list)
    for r in canonical:
        iso_groups[r['iso']].append(r)

    results_by_iso = {}
    for iso in isos_processed:
        iso_results = sorted(iso_groups.get(iso, []), key=lambda x: x['threshold'])
        if not iso_results:
            print(f"  {iso}: No canonical results — skipping")
            continue

        iso_data = {}

        # Load dispatch data for this ISO
        demand_norm, _ = get_demand_profile(iso, demand_data)
        supply_profiles = get_supply_profiles(iso, gen_profiles)
        supply_matrix = build_supply_matrix(supply_profiles)

        # Track first clean firm deployment year for deployment-gated learning
        first_cf_deployment_year = None
        existing_cf_pct = GRID_MIX_SHARES[iso].get('clean_firm', 0)

        # Inject pre-50% existing-only entries
        sens_toggles = SCENARIO_A['toggles']
        exist_match = _existing_match_pct(iso)
        for t_pre in [t for t in SCENARIO_THRESHOLDS if t < 50]:
            if t_pre <= exist_match:
                gf_pre = get_demand_growth_factor(iso, t_pre)
                demand_pre = BASE_DEMAND_TWH[iso] * gf_pre
                existing_twh = {
                    res: GRID_MIX_SHARES[iso].get(res, 0) / 100.0 * BASE_DEMAND_TWH[iso]
                    for res in ['clean_firm', 'solar', 'wind', 'ccs_ccgt', 'hydro']
                }
                existing_twh.update({'battery': 0, 'ldes': 0})
                iso_data[t_pre] = _build_existing_only_entry(
                    iso, t_pre, demand_pre, gf_pre, existing_twh, sens_toggles)

        # Convert each threshold winner using step5d's own costs
        for r in iso_results:
            t = r['threshold']
            demand_twh = r['demand_twh']
            demand_mwh = demand_twh * 1e6
            gf = demand_twh / BASE_DEMAND_TWH[iso] if BASE_DEMAND_TWH[iso] > 0 else 1.0

            # --- Resource TWh (use step5d's winner values directly) ---
            cf_pct = r.get('winner_clean_firm_pct', 0)
            sol_pct = r.get('winner_solar_pct', 0)
            wnd_pct = r.get('winner_wind_pct', 0)
            osw_pct = r.get('winner_offshore_wind_pct', 0)
            hyd_pct = r.get('winner_hydro_pct', 0)
            geo_pct = r.get('winner_geothermal_pct', 0)
            match_score = r.get('winner_match_score', 0)
            match_frac = match_score / 100.0

            bat4_pct = r.get('winner_battery_dispatch_pct', 0)
            bat8_pct = r.get('winner_battery8_dispatch_pct', 0)
            ldes_pct = r.get('winner_ldes_dispatch_pct', 0)
            h2_pct = r.get('winner_h2_dispatch_pct', 0)

            hydro_cap = HYDRO_CAP_TWH.get(iso, hyd_pct / 100.0 * demand_twh)
            hydro_twh = min(hyd_pct / 100.0 * demand_twh, hydro_cap)

            # Split clean_firm into nuclear vs CCS using compute_clean_firm_tranches
            existing_cf_twh = GRID_MIX_SHARES[iso].get('clean_firm', 0) / 100.0 * BASE_DEMAND_TWH[iso]
            total_cf_twh = cf_pct / 100.0 * demand_twh
            new_cf_twh = max(0, total_cf_twh - existing_cf_twh)
            geo_physics_twh = geo_pct / 100.0 * demand_twh if iso == 'CAISO' else 0

            # Use pipeline_config's tranche splitter for CCS/nuclear breakdown
            if new_cf_twh > 0:
                tranches = compute_clean_firm_tranches(
                    new_cf_twh=new_cf_twh, iso=iso,
                    firm_lev='H', ccs_lev='H', q45='1', tx_name='Medium',
                    geo_lev='H' if iso == 'CAISO' else None,
                    geo_physics_new_twh=geo_physics_twh,
                )
                ccs_twh = tranches.get('ccs_tranche_twh', 0)
            else:
                ccs_twh = 0

            resource_twh = {
                'clean_firm': total_cf_twh,
                'solar': sol_pct / 100.0 * demand_twh,
                'wind': wnd_pct / 100.0 * demand_twh,
                'offshore_wind': osw_pct / 100.0 * demand_twh,
                'ccs_ccgt': ccs_twh,
                'hydro': hydro_twh,
            }

            # --- Deployment-gated learning: detect first clean firm deployment ---
            if first_cf_deployment_year is None and cf_pct > existing_cf_pct + 0.1:
                first_cf_deployment_year = SBTI_YEAR_MAP.get(t, 2050)

            learning_frac = scenario_learning_fraction(
                t, scenario='A', first_deployment_year=first_cf_deployment_year)

            # Build learning-curve overrides for compute_mix_cost
            overrides = _build_learning_overrides(iso, learning_frac) if learning_frac > 0 else {}

            # --- System cost via compute_mix_cost (same model as Scenario B) ---
            mix_array = [cf_pct, sol_pct, wnd_pct, osw_pct, 0.0, hyd_pct,
                         match_score, bat4_pct, bat8_pct, ldes_pct, h2_pct]
            sens = SCENARIO_A['toggles']

            cost_result = compute_mix_cost(
                mix_array, sens, iso, demand_twh,
                overrides=overrides, growth_factor=gf,
                demand_norm=demand_norm,
                supply_profiles=supply_profiles,
                supply_matrix=supply_matrix,
            )

            result = {
                # System cost fields from compute_mix_cost (unified with Scenario B)
                'total_cost': cost_result['total_cost'],
                'effective_cost': cost_result['effective_cost'],
                'incremental': cost_result['incremental'],
                'wholesale': cost_result['wholesale'],
                'gas_cost': cost_result['gas_cost'],
                'resource_twh': resource_twh,
                'resource_pct': cost_result['resource_pct'],
                'match_score': match_score,
                'battery_twh': cost_result['battery_twh'],
                'battery8_twh': cost_result['battery8_twh'],
                'ldes_twh': cost_result['ldes_twh'],
                'h2_twh': cost_result['h2_twh'],
                'gas_backup_mw': cost_result['gas_backup_mw'],
                'new_gas_mw': cost_result['new_gas_mw'],
                'existing_gas_used_mw': cost_result['existing_gas_used_mw'],
                'clean_peak_mw': cost_result['clean_peak_mw'],
                'net_peak_residual_mw': cost_result['clean_peak_mw'],
                'learning_fraction': round(learning_frac, 3),
                'first_cf_deployment_year': first_cf_deployment_year,
                'new_build_cost_total': cost_result['new_build_cost_total'],
                'new_gen_twh': cost_result['new_gen_twh'],
                'blended_new_lcoe': cost_result['blended_new_lcoe'],
                'demand_twh': demand_twh,
                'demand_mwh': demand_mwh,
                'mix_raw': mix_array,
                'mix_source': 'MAC-queue',
                # Step5d MAC and CO2 (authoritative, pass-through)
                'mac_this_step': r.get('mac_this_step', 0),
                'co2_avoided_mt': r.get('co2_avoided_mt', 0),
                'co2_after_mt': r.get('co2_after_mt', 0),
                'baseline_co2_mt': r.get('baseline_co2_mt', 0),
                'cumulative_new_build_cost': r.get('cumulative_new_build_cost', 0),
                'cumulative_co2_avoided_mt': r.get('cumulative_co2_avoided_mt', 0),
                'cumulative_mac': r.get('cumulative_mac', 0),
                'target_year': r.get('target_year', 2050),
                # Dispatch percentages for dispatch cache lookups
                'battery_dispatch_pct': bat4_pct,
                'battery8_dispatch_pct': bat8_pct,
                'ldes_dispatch_pct': ldes_pct,
                'h2_dispatch_pct': h2_pct,
            }

            iso_data[t] = result

        results_by_iso[iso] = iso_data
        t_min = min(iso_data.keys())
        t_max = max(iso_data.keys())
        print(f"  {iso}: {len(iso_data)} thresholds, "
              f"${iso_data[t_min]['effective_cost']:.0f}/MWh @ {t_min}% -> "
              f"${iso_data[t_max]['effective_cost']:.0f}/MWh @ {t_max}%")

    if results_by_iso:
        save_scenario_results(results_by_iso, 'A', list(results_by_iso.keys()),
                              out_dir=OUTPUT_DIR)
        print(f"\n  Scenario A export complete: {len(results_by_iso)} ISOs → {OUTPUT_DIR}")


def main():
    parser = argparse.ArgumentParser(description='MAC-Optimized Consequential Queue')
    parser.add_argument('--iso', type=str, default=None,
                        help='Single ISO to process (default: all)')
    parser.add_argument('--sensitivity', type=str, default=None,
                        help='Single price sensitivity (default: all)')
    parser.add_argument('--growth', type=str, default=None,
                        help='Single demand growth level (default: all)')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading common data...")
    demand_data, gen_profiles, emission_rates, fossil_mix = load_common_data()

    isos = [args.iso] if args.iso else ISOS
    t0 = time.time()

    all_results = []
    for iso in isos:
        iso_results = run_iso(iso, demand_data, gen_profiles, emission_rates)
        all_results.extend(iso_results)

        # Save per-ISO parquet
        if iso_results:
            iso_df = pd.DataFrame(iso_results)
            out_path = os.path.join(args.output_dir, f'mac_queue_{iso}.parquet')
            iso_df.to_parquet(out_path, index=False)
            print(f"  Saved {out_path} ({len(iso_df)} rows)")

        gc.collect()

    # Save combined summary JSON
    if all_results:
        summary = {
            'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
            'isos': list(set(r['iso'] for r in all_results)),
            'sensitivities': list(PRICE_SENSITIVITIES.keys()),
            'growth_levels': DEMAND_GROWTH_LEVELS,
            'thresholds': MAC_THRESHOLDS,
            'total_rows': len(all_results),
            'results': all_results,
        }
        summary_path = os.path.join(args.output_dir, 'mac_queue_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSaved summary: {summary_path}")

    # ── Export consequential_queue.json for step8a compatibility ──
    # Step 8A reads this file and walks the queue sorted by marginal_mac.
    # Format: list of zone-step dicts with iso, delta_resources, marginal_mac.
    # Uses 'all_med' sensitivity + 'Medium' growth as the canonical pathway.
    if all_results:
        queue = _build_consequential_queue(all_results)
        queue_path = os.path.join(args.output_dir, 'consequential_queue.json')
        with open(queue_path, 'w') as f:
            json.dump({'queue': queue, 'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
                       'source': 'step5d_mac_queue.py'}, f, indent=2, default=str)
        print(f"Saved consequential queue: {queue_path} ({len(queue)} steps)")

    # ── Export scenario_a_{ISO}.json for step7c compatibility ──
    # Re-prices MAC queue winners under SCENARIO_A toggles (Low VRE, High firm)
    # so step7c can read them as Scenario A for comparison with Scenario B.
    if all_results:
        print("\n" + "=" * 60)
        print("EXPORTING SCENARIO A (for step7c scenario comparison)")
        print("  Sensitivity: high_firm_low_vre | Growth: Medium")
        print("=" * 60)
        _export_scenario_a(all_results, isos, demand_data, gen_profiles,
                           sensitivity='high_firm_low_vre', growth='Medium')

    elapsed = time.time() - t0
    print(f"\nDone. {len(all_results)} results in {elapsed:.1f}s")


if __name__ == '__main__':
    main()
