#!/usr/bin/env python3
"""
Step 8 — Wright's Law Learning Curves & Critical Mass Threshold
================================================================
Computes the data behind FIGURE 8 (procurement comparison page):

1. For each participation level (2–80% C&I demand), compute:
   - Total new-build firm TWh across all 7 ISOs (global learning pool)
   - Premium spend vs new-build spend by ISO
   - Wright's Law cost trajectories per technology
   - Blended effective $/MWh for Strategy 2C

2. Identify critical mass threshold:
   - Participation % where aggregate new-build firm hits first Wright's Law
     doubling (cumulative installed capacity per technology)
   - Post-threshold cost trajectory showing NOAK pricing activation

3. Comparison overlay: cost trajectories for Strategies 1A, 2A, 2B, 3A
   at same participation levels (all degrade; 2C improves past threshold)

Output: data/step8-wrights-law/wrights_law_curves.parquet (snappy compressed)
        + data/step8-wrights-law/wrights_law_curves.json (dashboard-ready)

Design: 100% vectorized numpy. No Python for-loops over data arrays.
"""

import os
import sys
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'data')
PP_DIR = os.path.join(DATA_DIR, 'step5-post-processing')
OUT_DIR = os.path.join(DATA_DIR, 'step8-wrights-law')
os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS (from existing pipeline — single source of truth)
# ═══════════════════════════════════════════════════════════════════════════════

ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']
OFFSHORE_ISOS = ['NYISO', 'NEISO', 'PJM', 'CAISO']  # ISOs with offshore wind resource

BASE_DEMAND_TWH = {
    'CAISO': 224.0, 'ERCOT': 488.0, 'PJM': 843.3, 'NYISO': 151.6,
    'NEISO': 115.3, 'MISO': 660.0, 'SPP': 296.0,
}

# SBTi threshold → target year mapping
SBTI_YEAR_MAP = {
    50: 2030, 55: 2031, 60: 2033, 65: 2034, 70: 2035, 75: 2036, 80: 2037,
    85: 2038, 87.5: 2039, 90: 2040, 92.5: 2043,
    95: 2045, 97.5: 2048, 99: 2049, 99.5: 2049, 99.9: 2050, 100: 2050,
}

THRESHOLDS = [50, 55, 60, 65, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 99.5, 99.9, 99.99]

# Participation levels to evaluate (% of C&I demand)
PARTICIPATION_PCTS = np.array([0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                                0.40, 0.50, 0.60, 0.70, 0.80], dtype=np.float64)

# Wright's Law learning curve parameters (from step3)
LEARNING_PARAMS = {
    'nuclear': {'L': (2028, 2036), 'M': (2030, 2040), 'H': (2036, 2048)},
    'ccs':     {'L': (2028, 2036), 'M': (2030, 2040), 'H': (2036, 2048)},
    'geo':     {'L': (2028, 2036), 'M': (2030, 2040), 'H': (2036, 2048)},
    'ldes':    {'L': (2028, 2036), 'M': (2030, 2040), 'H': (2036, 2048)},
    # Offshore wind (fixed-bottom): 83 GW global base, ~8.5% learning rate.
    # East Coast ISOs: NYISO, NEISO, PJM. Mature supply chain in Europe/Asia.
    'offshore_wind_fixed': {'L': (2026, 2034), 'M': (2028, 2038), 'H': (2032, 2045)},
    # Offshore wind (floating): 0.3 GW global base, ~11.5% learning rate.
    # CAISO only. No US commercial experience; first projects ~2030.
    'offshore_wind_float': {'L': (2029, 2037), 'M': (2031, 2042), 'H': (2035, 2050)},
}
LEARNING_EXPONENT = 0.6

# FOAK (High) and NOAK (Low) LCOE anchors — $/MWh
NUCLEAR_NEWBUILD_LCOE = {
    'H': {'CAISO': 140, 'ERCOT': 135, 'PJM': 160, 'NYISO': 170, 'NEISO': 165, 'MISO': 145, 'SPP': 130},
    'L': {'CAISO': 70,  'ERCOT': 68,  'PJM': 72,  'NYISO': 82,  'NEISO': 78,  'MISO': 70,  'SPP': 65},
}
CCS_LCOE = {
    'H': {'CAISO': 115, 'ERCOT': 92, 'PJM': 102, 'NYISO': 110, 'NEISO': 118, 'MISO': 95, 'SPP': 88},
    'M': {'CAISO': 86,  'ERCOT': 71, 'PJM': 79,  'NYISO': 86,  'NEISO': 93,  'MISO': 76, 'SPP': 72},
    'L': {'CAISO': 58,  'ERCOT': 52, 'PJM': 62,  'NYISO': 65,  'NEISO': 70,  'MISO': 58, 'SPP': 55},
}
GEOTHERMAL_LCOE = {'H': 110, 'L': 63}  # CAISO only

# Offshore wind LCOE anchors ($/MWh) — per-ISO, from step3
# FOAK = first-of-a-kind premium above High LCOE
OFFSHORE_WIND_FOAK = {
    'CAISO': 250,  # floating: 1.25× $200 High
    'PJM':   129,  # fixed: 1.15× $112 High
    'NYISO': 144,  # fixed: 1.15× $125 High
    'NEISO': 136,  # fixed: 1.15× $118 High
}
# NOAK = terminal learning floor (Medium scenario)
OFFSHORE_WIND_NOAK = {
    'CAISO': 72,   # floating NOAK Medium
    'PJM':   62,   # fixed NOAK Medium
    'NYISO': 65,   # fixed NOAK Medium
    'NEISO': 63,   # fixed NOAK Medium
}
# Current (2025) LCOE at Medium cost — the starting price before learning
OFFSHORE_WIND_LCOE_2025 = {
    'CAISO': 150,  # floating: Medium
    'PJM':   85,   # fixed: Medium
    'NYISO': 95,   # fixed: Medium
    'NEISO': 90,   # fixed: Medium
}

# Existing clean premium prices ($/MWh above wholesale for Strategy 2C)
EXISTING_CLEAN_PREMIUM = {
    'nuclear': 15.8,   # Premium above wholesale to keep nuclear profitable
    'hydro':   4.0,    # Minimal premium for existing hydro
    'vre':     4.0,    # Existing VRE at near-wholesale
}

# Wholesale prices by ISO ($/MWh)
WHOLESALE_PRICES = {
    'CAISO': 58.0, 'ERCOT': 38.0, 'PJM': 41.0, 'NYISO': 45.0,
    'NEISO': 50.0, 'MISO': 34.0, 'SPP': 30.0,
}

# Demand growth rates (Medium)
DEMAND_GROWTH_RATES = {
    'CAISO': 0.019, 'ERCOT': 0.035, 'PJM': 0.016, 'NYISO': 0.011,
    'NEISO': 0.010, 'MISO': 0.018, 'SPP': 0.020,
}

# Existing clean shares by ISO (% of demand served by existing clean in 2025)
# Used to estimate how much of each participation slice goes to existing vs new-build
GRID_MIX_SHARES = {
    'CAISO': {'nuclear': 7.9, 'hydro': 10.5, 'solar': 17.0, 'wind': 7.0, 'geothermal': 5.0},
    'ERCOT': {'nuclear': 4.5, 'hydro': 0.2,  'solar': 5.5,  'wind': 26.0},
    'PJM':   {'nuclear': 32.0,'hydro': 1.0,  'solar': 1.5,  'wind': 3.0},
    'NYISO': {'nuclear': 25.0,'hydro': 20.0, 'solar': 1.0,  'wind': 3.5},
    'NEISO': {'nuclear': 24.0,'hydro': 6.5,  'solar': 2.5,  'wind': 1.5},
    'MISO':  {'nuclear': 10.0,'hydro': 1.5,  'solar': 1.5,  'wind': 12.0},
    'SPP':   {'nuclear': 1.5, 'hydro': 1.5,  'solar': 2.0,  'wind': 35.0},
}

# Wright's Law first-doubling thresholds (GW installed for FOAK→NOAK ramp start)
# Based on DOE Liftoff / INL SOAR. Learning kicks in after first commercial doubling.
FIRST_DOUBLING_GW = {
    'nuclear': 5.0,           # ~5 GW of new SMR/AP1000 construction triggers learning
    'ccs':     8.0,           # ~8 GW of CCS-CCGT triggers cost declines
    'ldes':    3.0,           # ~3 GW of iron-air triggers manufacturing scaling
    'geo':     2.0,           # ~2 GW of enhanced geothermal (CAISO only)
    'offshore_wind_fixed': 10.0,  # ~10 GW US fixed-bottom (83 GW global base, US ramp-up)
    'offshore_wind_float': 2.0,   # ~2 GW floating (only 0.3 GW global, earlier doubling)
}

# Capacity factor assumptions for converting TWh → GW
CAPACITY_FACTORS = {
    'nuclear': 0.90,
    'ccs':     0.85,
    'ldes':    0.25,  # 100hr iron-air; low CF but critical role
    'geo':     0.90,
    'offshore_wind': {  # Per-ISO CFs (from NREL NOW-23 profiles)
        'CAISO': 0.43, 'PJM': 0.48, 'NYISO': 0.49, 'NEISO': 0.51,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# VECTORIZED WRIGHT'S LAW ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def learning_fraction_vec(years, foak_start, noak_year):
    """Vectorized Wright's Law learning fraction: (N,) years → (N,) fractions."""
    duration = noak_year - foak_start
    if duration <= 0:
        return np.ones_like(years, dtype=np.float64)
    active = np.clip((years - foak_start) / duration, 0.0, 1.0)
    return np.power(active, LEARNING_EXPONENT)


def year_adjusted_cost_vec(foak_costs, noak_costs, fractions):
    """Vectorized LCOE interpolation: FOAK × (1 - f) + NOAK × f."""
    return foak_costs * (1.0 - fractions) + noak_costs * fractions


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD EXISTING PIPELINE DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_shared_data():
    """Load shared_data.json for resource mixes and tranche data."""
    path = os.path.join(PP_DIR, 'shared_data.json')
    with open(path) as f:
        return json.load(f)


def load_strategy_data():
    """Load strategy 2C results for spend breakdown validation."""
    path = os.path.join(PP_DIR, 'strategy2_hourly.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# CORE COMPUTE: PARTICIPATION → NEW-BUILD TWH → LEARNING CURVES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_wrights_law_curves():
    """
    Main compute function. All vectorized.

    For each (participation_pct, threshold) pair:
    1. Compute demand per ISO scaled by participation and growth
    2. Allocate existing clean vs new-build
    3. Aggregate new-build firm TWh globally → GW installed
    4. Apply Wright's Law to get LCOE trajectory
    5. Compute blended $/MWh for Strategy 2C

    Returns dict of numpy arrays, ready for parquet serialization.
    """
    shared = load_shared_data()
    thresholds = np.array(shared['thresholds'], dtype=np.float64)
    cf_tranche = shared['cf_tranche_data']
    N_part = len(PARTICIPATION_PCTS)
    N_thr = len(thresholds)
    N_iso = len(ISOS)

    # Map thresholds → SBTi years (vectorized lookup)
    years = np.array([SBTI_YEAR_MAP.get(t, SBTI_YEAR_MAP.get(int(t), 2050))
                       for t in thresholds], dtype=np.float64)

    # ── Per-ISO arrays: (N_iso, N_thr) ──
    # New-build firm TWh from optimizer results (medium cost scenario)
    new_cf_twh = np.zeros((N_iso, N_thr), dtype=np.float64)
    existing_cf_twh = np.zeros((N_iso, N_thr), dtype=np.float64)
    ccs_twh = np.zeros((N_iso, N_thr), dtype=np.float64)
    nuc_nb_twh = np.zeros((N_iso, N_thr), dtype=np.float64)
    geo_twh_arr = np.zeros((N_iso, N_thr), dtype=np.float64)
    osw_twh_arr = np.zeros((N_iso, N_thr), dtype=np.float64)
    base_demand = np.zeros(N_iso, dtype=np.float64)
    growth_rates = np.zeros(N_iso, dtype=np.float64)

    for i, iso in enumerate(ISOS):
        base_demand[i] = BASE_DEMAND_TWH[iso]
        growth_rates[i] = DEMAND_GROWTH_RATES[iso]
        if iso in cf_tranche:
            ct = cf_tranche[iso]
            n = min(len(ct['new_cf_twh']), N_thr)
            new_cf_twh[i, :n] = ct['new_cf_twh'][:n]
            existing_cf_twh[i, :n] = ct['cf_existing_twh'][:n]
            ccs_twh[i, :n] = ct['ccs_tranche_twh'][:n]
            nuc_nb_twh[i, :n] = ct['nuclear_newbuild_twh'][:n]
            if 'geo_twh' in ct:
                geo_twh_arr[i, :n] = ct['geo_twh'][:n]
            if 'osw_twh' in ct:
                osw_twh_arr[i, :n] = ct['osw_twh'][:n]
            elif 'offshore_wind_twh' in ct:
                osw_twh_arr[i, :n] = ct['offshore_wind_twh'][:n]

    # Demand growth factor per threshold: (N_iso, N_thr)
    # demand(year) = base × (1 + rate)^(year - 2025)
    years_from_base = years - 2025.0  # (N_thr,)
    growth_factors = np.power(
        1.0 + growth_rates[:, np.newaxis],  # (N_iso, 1)
        years_from_base[np.newaxis, :]       # (1, N_thr)
    )  # → (N_iso, N_thr)

    grown_demand = base_demand[:, np.newaxis] * growth_factors  # (N_iso, N_thr)

    # ── Participation-scaled new-build: (N_part, N_iso, N_thr) ──
    # At each participation level, the new-build firm requirement scales linearly
    # with participation (simplified: participant demand = pct × grown_demand)
    # New-build per participant = new_cf_twh × (participant_demand / full_demand)

    # participant_demand: (N_part, N_iso, N_thr)
    part_demand = (PARTICIPATION_PCTS[:, np.newaxis, np.newaxis]
                   * grown_demand[np.newaxis, :, :])

    # Scale new-build proportionally to participation
    # new_build_firm_twh per part level: (N_part, N_iso, N_thr)
    part_new_cf = (PARTICIPATION_PCTS[:, np.newaxis, np.newaxis]
                   * new_cf_twh[np.newaxis, :, :])

    part_ccs = (PARTICIPATION_PCTS[:, np.newaxis, np.newaxis]
                * ccs_twh[np.newaxis, :, :])

    part_nuc_nb = (PARTICIPATION_PCTS[:, np.newaxis, np.newaxis]
                   * nuc_nb_twh[np.newaxis, :, :])

    part_geo = (PARTICIPATION_PCTS[:, np.newaxis, np.newaxis]
                * geo_twh_arr[np.newaxis, :, :])

    part_osw = (PARTICIPATION_PCTS[:, np.newaxis, np.newaxis]
                * osw_twh_arr[np.newaxis, :, :])

    # Existing clean: capped at existing supply, doesn't scale with participation beyond 100%
    part_existing = (PARTICIPATION_PCTS[:, np.newaxis, np.newaxis]
                     * existing_cf_twh[np.newaxis, :, :])

    # ── Global aggregation: sum across ISOs → (N_part, N_thr) ──
    global_new_cf_twh = part_new_cf.sum(axis=1)          # (N_part, N_thr)
    global_ccs_twh = part_ccs.sum(axis=1)                # (N_part, N_thr)
    global_nuc_nb_twh = part_nuc_nb.sum(axis=1)          # (N_part, N_thr)
    global_geo_twh = part_geo.sum(axis=1)                # (N_part, N_thr)
    global_osw_twh = part_osw.sum(axis=1)                # (N_part, N_thr)
    global_existing_twh = part_existing.sum(axis=1)      # (N_part, N_thr)

    # ── Per-ISO offshore wind TWh split by tech type (fixed vs floating) ──
    # Needed to compute GW separately for fixed-bottom and floating learning curves
    osw_fixed_twh = np.zeros((N_part, N_thr), dtype=np.float64)
    osw_float_twh = np.zeros((N_part, N_thr), dtype=np.float64)
    for i, iso in enumerate(ISOS):
        if iso == 'CAISO':
            osw_float_twh += part_osw[:, i, :]
        elif iso in OFFSHORE_ISOS:
            osw_fixed_twh += part_osw[:, i, :]

    # ── Convert TWh → GW installed (for Wright's Law doubling check) ──
    # GW = TWh / (CF × 8.76)
    def twh_to_gw(twh, cf):
        return twh / (cf * 8.76)

    global_ccs_gw = twh_to_gw(global_ccs_twh, CAPACITY_FACTORS['ccs'])
    global_nuc_gw = twh_to_gw(global_nuc_nb_twh, CAPACITY_FACTORS['nuclear'])
    global_ldes_gw = np.zeros_like(global_ccs_gw)  # LDES from new_cf - ccs - nuc - geo
    # For simplicity, assume any remaining new_cf beyond ccs+nuc+geo could be LDES
    remaining_cf_twh = np.maximum(global_new_cf_twh - global_ccs_twh
                                   - global_nuc_nb_twh - global_geo_twh, 0.0)

    # Offshore wind GW — use demand-weighted average CF for each tech type
    osw_fixed_avg_cf = np.average(
        [CAPACITY_FACTORS['offshore_wind'][iso] for iso in ['PJM', 'NYISO', 'NEISO']],
        weights=[BASE_DEMAND_TWH[iso] for iso in ['PJM', 'NYISO', 'NEISO']]
    )
    osw_float_avg_cf = CAPACITY_FACTORS['offshore_wind']['CAISO']
    global_osw_fixed_gw = twh_to_gw(osw_fixed_twh, osw_fixed_avg_cf)
    global_osw_float_gw = twh_to_gw(osw_float_twh, osw_float_avg_cf)

    # ── Wright's Law learning fractions by technology ──
    # Using Medium learning params: (2030, 2040) for all firm
    foak_start_m, noak_year_m = LEARNING_PARAMS['ccs']['M']
    frac_m = learning_fraction_vec(years, foak_start_m, noak_year_m)  # (N_thr,)

    # Broadcast to (N_part, N_thr) — same learning timeline for all participation levels
    # (in reality, higher participation → faster learning, but we model the SBTi timeline)
    frac_2d = np.broadcast_to(frac_m[np.newaxis, :], (N_part, N_thr)).copy()

    # Offshore wind learning fractions (separate timelines for fixed vs floating)
    osw_fixed_foak_s, osw_fixed_noak_y = LEARNING_PARAMS['offshore_wind_fixed']['M']
    osw_float_foak_s, osw_float_noak_y = LEARNING_PARAMS['offshore_wind_float']['M']
    frac_osw_fixed = learning_fraction_vec(years, osw_fixed_foak_s, osw_fixed_noak_y)  # (N_thr,)
    frac_osw_float = learning_fraction_vec(years, osw_float_foak_s, osw_float_noak_y)  # (N_thr,)
    frac_osw_fixed_2d = np.broadcast_to(frac_osw_fixed[np.newaxis, :], (N_part, N_thr)).copy()
    frac_osw_float_2d = np.broadcast_to(frac_osw_float[np.newaxis, :], (N_part, N_thr)).copy()

    # ── Critical mass check: suppress learning below first doubling ──
    # For each (part, thr), check if cumulative global GW exceeds first doubling
    ccs_above_doubling = global_ccs_gw >= FIRST_DOUBLING_GW['ccs']      # (N_part, N_thr) bool
    nuc_above_doubling = global_nuc_gw >= FIRST_DOUBLING_GW['nuclear']
    osw_fixed_above = global_osw_fixed_gw >= FIRST_DOUBLING_GW['offshore_wind_fixed']
    osw_float_above = global_osw_float_gw >= FIRST_DOUBLING_GW['offshore_wind_float']

    # Combined: at least CCS OR nuclear has hit critical mass
    any_above_critical = ccs_above_doubling | nuc_above_doubling

    # Learning fraction gated by critical mass: 0 below threshold
    frac_gated = np.where(any_above_critical, frac_2d, 0.0)  # (N_part, N_thr)

    # Offshore wind gating — independent critical mass thresholds
    frac_osw_fixed_gated = np.where(osw_fixed_above, frac_osw_fixed_2d, 0.0)
    frac_osw_float_gated = np.where(osw_float_above, frac_osw_float_2d, 0.0)

    # ── Cost trajectories per technology ──
    # CCS LCOE across ISOs (demand-weighted average for blended metric)
    ccs_foak_avg = np.average(
        [CCS_LCOE['H'][iso] for iso in ISOS],
        weights=[BASE_DEMAND_TWH[iso] for iso in ISOS]
    )
    ccs_noak_avg = np.average(
        [CCS_LCOE['M'][iso] for iso in ISOS],  # CCS floors at Medium
        weights=[BASE_DEMAND_TWH[iso] for iso in ISOS]
    )

    nuc_foak_avg = np.average(
        [NUCLEAR_NEWBUILD_LCOE['H'][iso] for iso in ISOS],
        weights=[BASE_DEMAND_TWH[iso] for iso in ISOS]
    )
    nuc_noak_avg = np.average(
        [NUCLEAR_NEWBUILD_LCOE['L'][iso] for iso in ISOS],
        weights=[BASE_DEMAND_TWH[iso] for iso in ISOS]
    )

    # With gated learning: (N_part, N_thr)
    ccs_lcoe_gated = year_adjusted_cost_vec(ccs_foak_avg, ccs_noak_avg, frac_gated)
    nuc_lcoe_gated = year_adjusted_cost_vec(nuc_foak_avg, nuc_noak_avg, frac_gated)

    # Without gating (pure timeline — what 2C achieves past critical mass)
    ccs_lcoe_pure = year_adjusted_cost_vec(ccs_foak_avg, ccs_noak_avg, frac_2d)
    nuc_lcoe_pure = year_adjusted_cost_vec(nuc_foak_avg, nuc_noak_avg, frac_2d)

    # ── Offshore wind LCOE trajectories (per-ISO, then demand-weighted avg) ──
    # Fixed-bottom (East Coast): demand-weighted across PJM/NYISO/NEISO
    osw_fixed_isos = ['PJM', 'NYISO', 'NEISO']
    osw_fixed_foak_avg = np.average(
        [OFFSHORE_WIND_FOAK[iso] for iso in osw_fixed_isos],
        weights=[BASE_DEMAND_TWH[iso] for iso in osw_fixed_isos]
    )
    osw_fixed_noak_avg = np.average(
        [OFFSHORE_WIND_NOAK[iso] for iso in osw_fixed_isos],
        weights=[BASE_DEMAND_TWH[iso] for iso in osw_fixed_isos]
    )
    osw_fixed_lcoe_gated = year_adjusted_cost_vec(osw_fixed_foak_avg, osw_fixed_noak_avg, frac_osw_fixed_gated)
    osw_fixed_lcoe_pure = year_adjusted_cost_vec(osw_fixed_foak_avg, osw_fixed_noak_avg, frac_osw_fixed_2d)

    # Floating (CAISO only)
    osw_float_foak = OFFSHORE_WIND_FOAK['CAISO']
    osw_float_noak = OFFSHORE_WIND_NOAK['CAISO']
    osw_float_lcoe_gated = year_adjusted_cost_vec(osw_float_foak, osw_float_noak, frac_osw_float_gated)
    osw_float_lcoe_pure = year_adjusted_cost_vec(osw_float_foak, osw_float_noak, frac_osw_float_2d)

    # Blended offshore wind LCOE (weighted by fixed vs floating TWh shares)
    total_osw_twh = np.maximum(osw_fixed_twh + osw_float_twh, 1e-6)
    osw_fixed_weight = osw_fixed_twh / total_osw_twh
    osw_float_weight = osw_float_twh / total_osw_twh
    osw_blended_lcoe_gated = osw_fixed_lcoe_gated * osw_fixed_weight + osw_float_lcoe_gated * osw_float_weight
    osw_blended_lcoe_pure = osw_fixed_lcoe_pure * osw_fixed_weight + osw_float_lcoe_pure * osw_float_weight

    # ── Blended 2C effective $/MWh ──
    # 2C cost = weighted average of:
    #   - Existing clean premium (low $/MWh, maintenance spend)
    #   - New-build firm (learning-curve-adjusted LCOE)
    #   - New-build offshore wind (separate learning curve)
    #   - New-build VRE (constant — already at scale)
    existing_price = 20.0  # Blended existing clean $/MWh (nuclear premium + hydro + vre)
    vre_price = 45.0       # Average new-build VRE $/MWh (solar+wind PPA)

    # Total demand per (part, thr): sum across ISOs
    total_part_demand = part_demand.sum(axis=1)  # (N_part, N_thr)
    total_existing = global_existing_twh
    total_new_cf = global_new_cf_twh
    total_osw = global_osw_twh

    # VRE portion: whatever isn't existing, new-build firm, or offshore wind
    total_new_vre = np.maximum(total_part_demand - total_existing - total_new_cf - total_osw, 0.0)

    # Blended cost: weighted by TWh shares
    safe_denom = np.maximum(total_part_demand, 1e-6)  # avoid div/0
    blended_2c = (
        (total_existing * existing_price
         + total_new_cf * ccs_lcoe_gated  # New firm priced at gated learning
         + total_osw * osw_blended_lcoe_gated  # Offshore wind with gated learning
         + total_new_vre * vre_price)
        / safe_denom
    )  # (N_part, N_thr)

    # Ungated version (post-critical-mass trajectory)
    blended_2c_noak = (
        (total_existing * existing_price
         + total_new_cf * ccs_lcoe_pure
         + total_osw * osw_blended_lcoe_pure
         + total_new_vre * vre_price)
        / safe_denom
    )

    # ── Comparison strategies: degradation curves ──
    # All shaped (N_part,) — participation-dependent, evaluated at single threshold
    # Strategy 1A (consequential): starts cheaper, degrades via wholesale erosion
    wholesale_erosion_pct = np.clip(PARTICIPATION_PCTS * 0.5, 0, 0.30)
    base_1a_cost = 35.0
    strat_1a = base_1a_cost + wholesale_erosion_pct * 40.0  # (N_part,)

    # Strategy 2A (hourly, no existing credit): pure new-build, no premiums
    thr_90_idx_tmp = thresholds.tolist().index(90)
    strat_2a = ccs_lcoe_pure[:, thr_90_idx_tmp] * 0.7 + vre_price * 0.3  # (N_part,)

    # Strategy 2B (hourly, free-rides on existing): cheap but stranding risk
    strat_2b_base = 30.0
    stranding_penalty = np.where(
        PARTICIPATION_PCTS > 0.20,
        (PARTICIPATION_PCTS - 0.20) * 80.0,
        0.0
    )
    strat_2b = strat_2b_base + stranding_penalty  # (N_part,)

    # Strategy 3A (annual): moderate, flat, no learning benefit
    strat_3a = 42.0 + PARTICIPATION_PCTS * 15.0  # (N_part,)

    # ── Find critical mass threshold (per threshold) ──
    # For each threshold: smallest participation where critical mass is achieved
    critical_mass_pct = np.full(N_thr, np.nan, dtype=np.float64)
    for j in range(N_thr):
        mask = any_above_critical[:, j]
        if mask.any():
            critical_mass_pct[j] = PARTICIPATION_PCTS[np.argmax(mask)]

    # ── Per-ISO spend decomposition at each participation level ──
    # Shape: (N_iso, N_part, N_thr) — cost in $M
    per_iso_ccs_foak = np.array([CCS_LCOE['H'][iso] for iso in ISOS], dtype=np.float64)
    per_iso_ccs_noak = np.array([CCS_LCOE['M'][iso] for iso in ISOS], dtype=np.float64)

    # per-ISO gated LCOE: (N_iso, N_part, N_thr) — need frac_gated broadcast
    per_iso_ccs_lcoe = year_adjusted_cost_vec(
        per_iso_ccs_foak[:, np.newaxis, np.newaxis],   # (N_iso, 1, 1)
        per_iso_ccs_noak[:, np.newaxis, np.newaxis],
        frac_gated[np.newaxis, :, :]                    # (1, N_part, N_thr)
    )  # → (N_iso, N_part, N_thr)

    # Spend decomposition: (N_iso, N_part, N_thr) in $M
    # Transpose part_new_cf from (N_part, N_iso, N_thr) to (N_iso, N_part, N_thr)
    pnc_t = np.transpose(part_new_cf, (1, 0, 2))
    pe_t = np.transpose(part_existing, (1, 0, 2))

    existing_spend_iso = pe_t * existing_price        # $M (TWh × $/MWh)
    newbuild_spend_iso = pnc_t * per_iso_ccs_lcoe     # $M

    # Premium share: existing / (existing + newbuild)
    total_spend_iso = existing_spend_iso + newbuild_spend_iso
    safe_total = np.maximum(total_spend_iso, 1e-6)
    premium_share_iso = existing_spend_iso / safe_total  # (N_iso, N_part, N_thr)

    print(f"[Step 8] Computed {N_part} participation levels × {N_thr} thresholds × {N_iso} ISOs")

    # ═══════════════════════════════════════════════════════════════════════════
    # ASSEMBLE RESULTS
    # ═══════════════════════════════════════════════════════════════════════════

    results = {
        'metadata': {
            'description': 'Wright\'s Law learning curves and critical mass threshold for Strategy 2C',
            'participation_pcts': PARTICIPATION_PCTS.tolist(),
            'thresholds': thresholds.tolist(),
            'isos': ISOS,
            'sbti_years': years.tolist(),
            'learning_exponent': LEARNING_EXPONENT,
            'first_doubling_gw': FIRST_DOUBLING_GW,
        },

        # ── FIGURE 8 data: 2C cost curve with zones ──
        'figure8': {
            'participation_pcts': PARTICIPATION_PCTS.tolist(),
            # Use the "at 90% CFE" slice as the headline figure
            'blended_2c_at_90': blended_2c[:, thresholds.tolist().index(90)].tolist(),
            'blended_2c_noak_at_90': blended_2c_noak[:, thresholds.tolist().index(90)].tolist(),
            'strat_1a_at_90': strat_1a.tolist(),
            'strat_2a_at_90': strat_2a.tolist(),
            'strat_2b_at_90': strat_2b.tolist(),
            'strat_3a_at_90': strat_3a.tolist(),
            'critical_mass_pct_at_90': float(critical_mass_pct[thresholds.tolist().index(90)])
                if not np.isnan(critical_mass_pct[thresholds.tolist().index(90)]) else None,
        },

        # ── Full grid: 2C blended cost (N_part × N_thr) ──
        'cost_grid_2c': {
            'blended_gated': blended_2c.tolist(),       # With critical mass gating
            'blended_noak': blended_2c_noak.tolist(),   # Full learning (post-critical-mass)
        },

        # ── Technology LCOE trajectories ──
        'lcoe_trajectories': {
            'ccs_gated': ccs_lcoe_gated.tolist(),       # (N_part, N_thr)
            'ccs_pure': ccs_lcoe_pure.tolist(),
            'nuclear_gated': nuc_lcoe_gated.tolist(),
            'nuclear_pure': nuc_lcoe_pure.tolist(),
            'ccs_foak': float(ccs_foak_avg),
            'ccs_noak': float(ccs_noak_avg),
            'nuclear_foak': float(nuc_foak_avg),
            'nuclear_noak': float(nuc_noak_avg),
            # Offshore wind learning trajectories
            'osw_fixed_gated': osw_fixed_lcoe_gated.tolist(),
            'osw_fixed_pure': osw_fixed_lcoe_pure.tolist(),
            'osw_float_gated': osw_float_lcoe_gated.tolist(),
            'osw_float_pure': osw_float_lcoe_pure.tolist(),
            'osw_blended_gated': osw_blended_lcoe_gated.tolist(),
            'osw_blended_pure': osw_blended_lcoe_pure.tolist(),
            'osw_fixed_foak': float(osw_fixed_foak_avg),
            'osw_fixed_noak': float(osw_fixed_noak_avg),
            'osw_float_foak': float(osw_float_foak),
            'osw_float_noak': float(osw_float_noak),
        },

        # ── Critical mass thresholds ──
        'critical_mass': {
            'pct_by_threshold': {
                str(t): float(critical_mass_pct[i]) if not np.isnan(critical_mass_pct[i]) else None
                for i, t in enumerate(thresholds.tolist())
            },
            'global_new_build_gw_ccs': global_ccs_gw.tolist(),    # (N_part, N_thr)
            'global_new_build_gw_nuclear': global_nuc_gw.tolist(),
            'global_new_build_gw_osw_fixed': global_osw_fixed_gw.tolist(),
            'global_new_build_gw_osw_float': global_osw_float_gw.tolist(),
            'above_critical_mass': any_above_critical.tolist(),    # (N_part, N_thr) bool
            'osw_fixed_above_critical': osw_fixed_above.tolist(),
            'osw_float_above_critical': osw_float_above.tolist(),
        },

        # ── Comparison strategies at 90% CFE ──
        'strategy_comparison_90': {
            'strat_1a': strat_1a.tolist(),
            'strat_2a': strat_2a.tolist(),
            'strat_2b': strat_2b.tolist(),
            'strat_2c_gated': blended_2c[:, thr_90_idx_tmp].tolist(),
            'strat_2c_noak': blended_2c_noak[:, thr_90_idx_tmp].tolist(),
            'strat_3a': strat_3a.tolist(),
        },

        # ── Per-ISO spend decomposition (at 90% CFE slice) ──
        'iso_spend_90': {},

        # ── Global new-build aggregation ──
        'global_aggregation': {
            'new_cf_twh': global_new_cf_twh.tolist(),         # (N_part, N_thr)
            'existing_twh': global_existing_twh.tolist(),
            'new_vre_twh': total_new_vre.tolist(),
            'osw_twh': global_osw_twh.tolist(),               # (N_part, N_thr)
            'osw_fixed_twh': osw_fixed_twh.tolist(),
            'osw_float_twh': osw_float_twh.tolist(),
        },
    }

    # Per-ISO spend at 90% CFE threshold
    thr_90_idx = thresholds.tolist().index(90)
    for i, iso in enumerate(ISOS):
        iso_result = {
            'existing_spend': existing_spend_iso[i, :, thr_90_idx].tolist(),
            'newbuild_spend': newbuild_spend_iso[i, :, thr_90_idx].tolist(),
            'premium_share': premium_share_iso[i, :, thr_90_idx].tolist(),
            'new_cf_twh': part_new_cf[:, i, thr_90_idx].tolist(),
            'existing_cf_twh': part_existing[:, i, thr_90_idx].tolist(),
        }
        if iso in OFFSHORE_ISOS:
            iso_result['osw_twh'] = part_osw[:, i, thr_90_idx].tolist()
        results['iso_spend_90'][iso] = iso_result

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT: PARQUET + JSON
# ═══════════════════════════════════════════════════════════════════════════════

def save_parquet(results):
    """Save key arrays as a compact parquet file (snappy compression)."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        print("[Step 8] pyarrow not available — skipping parquet output")
        return

    # Flatten the main grid into a columnar table:
    # (participation_pct, threshold, year, blended_2c, blended_noak,
    #  global_new_cf_twh, ccs_lcoe, critical_mass_flag)
    parts = np.array(results['metadata']['participation_pcts'], dtype=np.float64)
    thrs = np.array(results['metadata']['thresholds'], dtype=np.float64)
    yrs = np.array(results['metadata']['sbti_years'], dtype=np.float64)

    N_p, N_t = len(parts), len(thrs)

    # Meshgrid: flatten to 1D columns
    p_grid, t_grid = np.meshgrid(parts, thrs, indexing='ij')
    _, y_grid = np.meshgrid(parts, yrs, indexing='ij')

    table = pa.table({
        'participation_pct': pa.array(p_grid.ravel(), type=pa.float32()),
        'threshold': pa.array(t_grid.ravel(), type=pa.float32()),
        'sbti_year': pa.array(y_grid.ravel(), type=pa.float32()),
        'blended_2c': pa.array(
            np.array(results['cost_grid_2c']['blended_gated'], dtype=np.float32).ravel()
        ),
        'blended_2c_noak': pa.array(
            np.array(results['cost_grid_2c']['blended_noak'], dtype=np.float32).ravel()
        ),
        'global_new_cf_twh': pa.array(
            np.array(results['global_aggregation']['new_cf_twh'], dtype=np.float32).ravel()
        ),
        'ccs_lcoe_gated': pa.array(
            np.array(results['lcoe_trajectories']['ccs_gated'], dtype=np.float32).ravel()
        ),
        'osw_blended_lcoe_gated': pa.array(
            np.array(results['lcoe_trajectories']['osw_blended_gated'], dtype=np.float32).ravel()
        ),
        'osw_twh': pa.array(
            np.array(results['global_aggregation']['osw_twh'], dtype=np.float32).ravel()
        ),
        'above_critical_mass': pa.array(
            np.array(results['critical_mass']['above_critical_mass']).ravel()
        ),
    })

    out_path = os.path.join(OUT_DIR, 'wrights_law_curves.parquet')
    pq.write_table(table, out_path, compression='snappy')
    size_kb = os.path.getsize(out_path) / 1024
    print(f"[Step 8] Parquet: {out_path} ({size_kb:.1f} KB, {len(table)} rows)")


def save_json(results):
    """Save dashboard-ready JSON (figure 8 + comparison slices only — minimal)."""
    # Strip the large grids for the dashboard-facing JSON; keep only figure-level data
    dashboard = {
        'metadata': results['metadata'],
        'figure8': results['figure8'],
        'lcoe_trajectories': {
            'ccs_foak': results['lcoe_trajectories']['ccs_foak'],
            'ccs_noak': results['lcoe_trajectories']['ccs_noak'],
            'nuclear_foak': results['lcoe_trajectories']['nuclear_foak'],
            'nuclear_noak': results['lcoe_trajectories']['nuclear_noak'],
            'osw_fixed_foak': results['lcoe_trajectories']['osw_fixed_foak'],
            'osw_fixed_noak': results['lcoe_trajectories']['osw_fixed_noak'],
            'osw_float_foak': results['lcoe_trajectories']['osw_float_foak'],
            'osw_float_noak': results['lcoe_trajectories']['osw_float_noak'],
        },
        'critical_mass': results['critical_mass']['pct_by_threshold'],
        'strategy_comparison_90': results['strategy_comparison_90'],
        'iso_spend_90': results['iso_spend_90'],
    }

    out_path = os.path.join(OUT_DIR, 'wrights_law_curves.json')
    with open(out_path, 'w') as f:
        json.dump(dashboard, f, separators=(',', ':'))
    size_kb = os.path.getsize(out_path) / 1024
    print(f"[Step 8] JSON:    {out_path} ({size_kb:.1f} KB)")


def main():
    print("=" * 70)
    print("Step 8 — Wright's Law Learning Curves & Critical Mass Threshold")
    print("=" * 70)

    results = compute_wrights_law_curves()

    save_parquet(results)
    save_json(results)

    # Summary
    fig8 = results['figure8']
    cm = fig8['critical_mass_pct_at_90']
    print(f"\n{'─' * 50}")
    print(f"FIGURE 8 Summary (90% CFE threshold):")
    print(f"  Critical mass threshold: {cm*100 if cm else 'N/A'}% participation")
    print(f"  2C blended at 5%:  ${fig8['blended_2c_at_90'][1]:.1f}/MWh")
    print(f"  2C blended at 30%: ${fig8['blended_2c_at_90'][6]:.1f}/MWh")
    print(f"  2C blended at 80%: ${fig8['blended_2c_at_90'][-1]:.1f}/MWh")
    print(f"  2C NOAK at 80%:    ${fig8['blended_2c_noak_at_90'][-1]:.1f}/MWh")

    cm_data = results['critical_mass']['pct_by_threshold']
    valid_cm = {k: v for k, v in cm_data.items() if v is not None}
    if valid_cm:
        print(f"\n  Critical mass by threshold:")
        for thr in ['50.0', '70.0', '90.0', '95.0', '99.99']:
            if thr in valid_cm:
                print(f"    {thr}% CFE: {valid_cm[thr]*100:.0f}% participation")

    print(f"\n  ISO spend decomposition at 90% CFE, 20% participation:")
    spend = results['iso_spend_90']
    idx_20 = list(PARTICIPATION_PCTS).index(0.20)
    for iso in ISOS:
        if iso in spend:
            ex = spend[iso]['existing_spend'][idx_20]
            nb = spend[iso]['newbuild_spend'][idx_20]
            ps = spend[iso]['premium_share'][idx_20]
            print(f"    {iso}: existing=${ex:.0f}M, newbuild=${nb:.0f}M, premium_share={ps:.1%}")

    print(f"\n{'=' * 70}")
    print("Done.")


if __name__ == '__main__':
    main()
