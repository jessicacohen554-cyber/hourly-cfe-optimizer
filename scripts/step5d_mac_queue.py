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
       c. Sample archetypes from PFS + step1d + step1d2 that respect floor
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

Input:  data/step1-pfs-parquets/, data/step1d-storage-parquets/, data/step1d2-storage-parquets/
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
    NEISO_CCS_GAS_ADDER,
    FOAK_NUCLEAR_NEWBUILD, FOAK_CCS_45Q_ON, FOAK_CCS_45Q_OFF,
    FOAK_GEOTHERMAL, FOAK_OFFSHORE_WIND, FOAK_LDES, FOAK_H2,
    NOAK_BATTERY, NOAK_BATTERY8, NOAK_OFFSHORE_WIND,
    LEARNING_PARAMS, learning_fraction, year_adjusted_cost,
    STORAGE_REVENUE_CREDITS,
    DEMAND_GROWTH_RATES, THRESHOLD_TARGET_YEARS,
    HYDRO_CAP_PCT,
    LEVEL_NAME, LMH,
    PATHS, H,
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
)

# ============================================================================
# CONSTANTS
# ============================================================================

PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'step5-post-processing', 'mac_queue')

PFS_DIRS = [
    os.path.join(PROJECT_ROOT, 'data', 'step1-pfs-parquets'),
    os.path.join(PROJECT_ROOT, 'data', 'step1d-storage-parquets'),
    os.path.join(PROJECT_ROOT, 'data', 'step1d2-storage-parquets'),
]

RESOURCE_COLS = ['clean_firm', 'solar', 'wind', 'hydro', 'offshore_wind', 'geothermal']
STORAGE_COLS = ['battery_dispatch_pct', 'battery8_dispatch_pct', 'ldes_dispatch_pct', 'h2_dispatch_pct']
MIX_COLS = RESOURCE_COLS + STORAGE_COLS + ['hourly_match_score']

# Active thresholds for the MAC queue (50%+ only — coarse thresholds excluded)
MAC_THRESHOLDS = [50, 55, 60, 65, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 99.5, 99.9, 99.99]

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
MAX_OVERSHOOT = 1.0  # percentage points


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

    # --- Clean Firm (tranched: uprate → geothermal → min(nuclear, CCS)) ---
    existing_cf_twh = floor_twh.get('clean_firm', 0.0)
    cf_new_twh = max(0, mix_twh['clean_firm'] - existing_cf_twh)
    if cf_new_twh > 0:
        remaining = cf_new_twh
        cf_tx = get_tx('clean_firm', tx_name, iso)
        ccs_tx = get_tx('ccs_ccgt', tx_name, iso)

        # Tranche 1: Nuclear uprates (cumulative cap)
        uprate_cap = UPRATE_CAP_TWH[iso]
        uprate_used = cumulative_caps.get('uprate_twh', 0.0)
        uprate_avail = max(0, uprate_cap - uprate_used)
        uprate_twh = min(remaining, uprate_avail)
        uprate_price = UPRATE_LCOE[firm_lev]  # No TX for uprates (grid-connected)
        uprate_cost = uprate_twh * 1e6 * uprate_price
        total_cost += uprate_cost
        cumulative_caps['uprate_twh'] = uprate_used + uprate_twh
        remaining -= uprate_twh
        breakdown['uprate'] = {'new_twh': uprate_twh, 'lcoe': uprate_price, 'cost': uprate_cost}

        # Tranche 2: Geothermal (CAISO only, cumulative cap)
        geo_cost = 0.0
        geo_twh = 0.0
        if iso == 'CAISO' and geo_lev and remaining > 0:
            geo_cap = GEOTHERMAL_CAP_TWH
            geo_used = cumulative_caps.get('geo_twh', 0.0)
            geo_avail = max(0, geo_cap - geo_used)
            geo_twh = min(remaining, geo_avail)
            if geo_twh > 0:
                geo_base = GEOTHERMAL_LCOE[geo_lev]
                foak_geo = FOAK_GEOTHERMAL
                noak_geo = GEOTHERMAL_LCOE['L']  # NOAK = Low cost floor
                geo_lcoe = _apply_learning_curve(geo_base, foak_geo, noak_geo, 'geo', firm_lev, target_year)
                geo_lcoe += cf_tx
                geo_cost = geo_twh * 1e6 * geo_lcoe
                total_cost += geo_cost
                cumulative_caps['geo_twh'] = geo_used + geo_twh
                remaining -= geo_twh
                breakdown['geothermal'] = {'new_twh': geo_twh, 'lcoe': geo_lcoe, 'cost': geo_cost}

        # Tranche 3: Cheapest of nuclear new-build vs CCS (cumulative CCS cap)
        if remaining > 0:
            # Nuclear new-build with learning curve
            nuc_base = NUCLEAR_NEWBUILD_LCOE[firm_lev][iso]
            nuc_foak = FOAK_NUCLEAR_NEWBUILD[iso]
            nuc_noak = NUCLEAR_NEWBUILD_LCOE['L'][iso]  # NOAK = Low cost
            nuc_price = _apply_learning_curve(nuc_base, nuc_foak, nuc_noak, 'nuclear', firm_lev, target_year)
            nuc_price += cf_tx

            # CCS with learning curve
            ccs_table = CCS_LCOE_45Q_ON if q45 == '1' else CCS_LCOE_45Q_OFF
            ccs_base = ccs_table[ccs_lev][iso]
            ccs_foak_table = FOAK_CCS_45Q_ON if q45 == '1' else FOAK_CCS_45Q_OFF
            ccs_foak = ccs_foak_table[iso]
            ccs_noak = ccs_table['L'][iso]
            ccs_price = _apply_learning_curve(ccs_base, ccs_foak, ccs_noak, 'ccs', ccs_lev, target_year)
            if iso == 'NEISO':
                ccs_price += NEISO_CCS_GAS_ADDER
            ccs_price += ccs_tx

            # CCS cumulative cap
            ccs_cap = CCS_CAP_TWH.get(iso, 9999.0)
            ccs_used = cumulative_caps.get('ccs_twh', 0.0)
            ccs_avail = max(0, ccs_cap - ccs_used)

            if nuc_price <= ccs_price or ccs_avail <= 0:
                # Nuclear for everything
                nuc_twh = remaining
                ccs_tranche_twh = 0.0
                tranche3_cost = nuc_twh * 1e6 * nuc_price
            else:
                # CCS gets min(remaining, headroom), nuclear gets overflow
                ccs_tranche_twh = min(remaining, ccs_avail)
                nuc_twh = remaining - ccs_tranche_twh
                tranche3_cost = (ccs_tranche_twh * 1e6 * ccs_price +
                                 nuc_twh * 1e6 * nuc_price)
                cumulative_caps['ccs_twh'] = ccs_used + ccs_tranche_twh

            total_cost += tranche3_cost
            breakdown['nuclear_newbuild'] = {'new_twh': nuc_twh, 'lcoe': nuc_price}
            breakdown['ccs_tranche'] = {'new_twh': ccs_tranche_twh, 'lcoe': ccs_price}
            breakdown['tranche3_cost'] = tranche3_cost

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
    """Update floor with winner's resources (max of current floor and winner)."""
    new_floor = dict(floor_twh)
    for res in RESOURCE_COLS:
        winner_res_twh = winner_pct.get(res, 0.0) / 100.0 * demand_twh
        new_floor[res] = max(floor_twh.get(res, 0.0), winner_res_twh)
    for col in STORAGE_COLS:
        new_floor[col] = max(floor_twh.get(col, 0.0), winner_pct.get(col, 0.0))
    return new_floor


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

    # Overshoot filter: match score within [threshold - 0.5, threshold + MAX_OVERSHOOT]
    scores = df['hourly_match_score'].values
    mask &= (scores >= threshold - 0.5)
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
        mask2 &= (scores >= threshold - 1.0)
        mask2 &= (scores <= threshold + 2.0)
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
# SINGLE THRESHOLD OPTIMIZER
# ============================================================================

def optimize_threshold(iso, threshold, floor_twh, cumulative_caps,
                       sens, demand_twh, target_year,
                       demand_norm, supply_profiles, supply_matrix, emission_rates,
                       existing_clean_twh, prev_threshold=None,
                       existing_clean_hourly_pct=None):
    """Optimize a single threshold step.

    Returns:
        result dict with winner, costs, CO2, MAC
        updated floor_twh
        updated cumulative_caps
    """
    # Convert floor to %
    floor_pct = floor_to_pct(floor_twh, demand_twh)

    # 1. Compute CO2 baseline (path-dependent)
    # The effective clean % at baseline is the PREVIOUS threshold achieved
    # (not the raw sum of resource allocations, which can exceed 100% due to curtailment)
    growth_factor = demand_twh / REGIONAL_DEMAND_TWH[iso]

    if prev_threshold is not None:
        baseline_effective_clean_pct = prev_threshold
    else:
        # First threshold: use existing clean HOURLY match (not annual share!)
        # This accounts for solar curtailment gap
        baseline_effective_clean_pct = existing_clean_hourly_pct if existing_clean_hourly_pct else sum(GRID_MIX_SHARES[iso].values())

    # Cap at 100% for safety
    baseline_effective_clean_pct = min(baseline_effective_clean_pct, 99.99)

    _, retirement_info = compute_fossil_retirement(
        iso, baseline_effective_clean_pct, emission_rates, {},
        demand_growth_factor=growth_factor)

    # Total fossil CO2 at baseline = remaining fossil TWh × remaining emission rate
    # (NOT displaced_rate, which is the rate of fuels pushed out by clean energy)
    remaining_rate = retirement_info.get('remaining_rate_tco2_mwh', 0.3911)
    fossil_twh_baseline = max(0, demand_twh * (100.0 - baseline_effective_clean_pct) / 100.0)
    baseline_co2_mt = fossil_twh_baseline * 1e6 * remaining_rate / 1e6  # million tons

    # 2. Load PFS archetypes
    pfs_df = load_pfs_for_threshold(iso, threshold)

    if pfs_df.empty:
        print(f"    WARNING: No PFS data for {iso} t{threshold:g}")
        return None, floor_twh, cumulative_caps

    # 3. Filter and sample
    filtered = filter_and_sample(pfs_df, floor_pct, threshold)

    if filtered.empty:
        print(f"    WARNING: No mixes pass floor filter for {iso} t{threshold:g}")
        return None, floor_twh, cumulative_caps

    print(f"    {iso} t{threshold:g}: {len(pfs_df)} PFS mixes → {len(filtered)} after filtering")

    # 4. Score each archetype
    best_mac = float('inf')
    best_mix = None
    best_cost = 0
    best_co2_avoided = 0
    best_co2_after = 0
    best_breakdown = {}
    top_archetypes = []

    def _score_mix(mix_pct):
        """Score a single mix: returns (mac, co2_avoided_mt, co2_after_mt, nb_cost, breakdown) or None."""
        # Use hourly_match_score as the effective clean % (accounts for curtailment)
        candidate_clean_pct = min(mix_pct.get('hourly_match_score', 0), 99.99)
        _, cand_info = compute_fossil_retirement(
            iso, candidate_clean_pct, emission_rates, {},
            demand_growth_factor=growth_factor)
        cand_remaining_rate = cand_info.get('remaining_rate_tco2_mwh', 0.3911)
        fossil_twh_after = max(0, demand_twh * (100.0 - candidate_clean_pct) / 100.0)
        co2_after_mt = fossil_twh_after * 1e6 * cand_remaining_rate / 1e6

        co2_avoided_mt = baseline_co2_mt - co2_after_mt
        if co2_avoided_mt <= 0.001:  # At least 1000 tons avoided
            return None

        caps_copy = dict(cumulative_caps)
        nb_cost, nb_breakdown, _ = compute_new_build_cost(
            iso, mix_pct, floor_twh, demand_twh, sens, target_year, caps_copy)

        # $0 new-build cost is valid (floor already covers this threshold)
        mac = nb_cost / (co2_avoided_mt * 1e6) if co2_avoided_mt > 0.001 else 0.0  # $/tCO2
        return mac, co2_avoided_mt, co2_after_mt, nb_cost, nb_breakdown

    for idx in range(len(filtered)):
        row = filtered.iloc[idx]
        mix_pct = mix_row_to_pct(row)

        scored = _score_mix(mix_pct)
        if scored is None:
            continue
        mac, co2_avoided_mt, co2_after_mt, nb_cost, nb_breakdown = scored

        # Track top 5 for phase 2
        if len(top_archetypes) < 5 or mac < top_archetypes[-1][1]:
            top_archetypes.append((mix_pct, mac))
            top_archetypes.sort(key=lambda x: x[1])
            top_archetypes = top_archetypes[:5]

        if mac < best_mac:
            best_mac = mac
            best_mix = mix_pct
            best_cost = nb_cost
            best_co2_avoided = co2_avoided_mt
            best_co2_after = co2_after_mt
            best_breakdown = nb_breakdown

    # 5. Phase 2: Refined search around top archetypes
    if top_archetypes and len(filtered) > 10:
        phase2_df = phase2_refine(top_archetypes, floor_pct, threshold)
        if not phase2_df.empty:
            for idx in range(len(phase2_df)):
                row = phase2_df.iloc[idx]
                mix_pct = mix_row_to_pct(row)

                scored = _score_mix(mix_pct)
                if scored is None:
                    continue
                mac, co2_avoided_mt, co2_after_mt, nb_cost, nb_breakdown = scored

                if mac < best_mac:
                    best_mac = mac
                    best_mix = mix_pct
                    best_cost = nb_cost
                    best_co2_avoided = co2_avoided_mt
                    best_co2_after = co2_after_mt
                    best_breakdown = nb_breakdown

    if best_mix is None:
        print(f"    WARNING: No valid MAC found for {iso} t{threshold:g}")
        return None, floor_twh, cumulative_caps

    # 6. Update cumulative caps with winner
    _, _, updated_caps = compute_new_build_cost(
        iso, best_mix, floor_twh, demand_twh, sens, target_year, dict(cumulative_caps))

    # 7. Ratchet floor
    new_floor = ratchet_floor(floor_twh, best_mix, demand_twh)

    # Compute new-build MWh for $/MWh reference
    total_new_build_twh = 0.0
    for res in RESOURCE_COLS:
        nb_twh = max(0, best_mix.get(res, 0) / 100.0 * demand_twh - floor_twh.get(res, 0))
        total_new_build_twh += nb_twh

    # 8. Build result
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

    # New build incremental
    for res in RESOURCE_COLS:
        nb_twh = max(0, best_mix.get(res, 0) / 100.0 * demand_twh - floor_twh.get(res, 0))
        result[f'new_build_{res}_twh'] = round(nb_twh, 3)
    for col in STORAGE_COLS:
        nb_pct = max(0, best_mix.get(col, 0) - floor_twh.get(col, 0))
        result[f'new_build_{col}'] = round(nb_pct, 3)

    # Floor state
    for res in RESOURCE_COLS:
        result[f'floor_{res}_twh'] = round(new_floor.get(res, 0), 3)
    for col in STORAGE_COLS:
        result[f'floor_{col}'] = round(new_floor.get(col, 0), 3)

    print(f"    → MAC: ${best_mac:.1f}/tCO2 | CO2 avoided: {best_co2_avoided:.3f} Mt "
          f"| NB cost: ${best_cost/1e6:.1f}M")

    return result, new_floor, updated_caps


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pathway(iso, price_sens_name, growth_level, demand_norm, supply_profiles,
                supply_matrix, emission_rates):
    """Run full MAC queue for one ISO × price_sensitivity × demand_growth pathway."""
    sens = PRICE_SENSITIVITIES[price_sens_name]
    growth_rate = DEMAND_GROWTH_RATES[iso][growth_level]
    base_demand = REGIONAL_DEMAND_TWH[iso]

    # Initialize floor from existing clean
    floor_twh = init_floor(iso)
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

        result, floor_twh, cumulative_caps = optimize_threshold(
            iso, threshold, floor_twh, cumulative_caps,
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

    elapsed = time.time() - t0
    print(f"\nDone. {len(all_results)} results in {elapsed:.1f}s")


if __name__ == '__main__':
    main()
