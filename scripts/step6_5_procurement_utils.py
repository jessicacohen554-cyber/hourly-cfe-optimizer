#!/usr/bin/env python3
"""
Step 6.5 — Shared Procurement Strategy Utilities
=================================================
Cross-cutting logic used by all three strategy family scripts:
  - SSS allocation (8760 shape per ISO)
  - EAC pricing tranches (§15.14.4)
  - PPA premium calculations
  - Participation-to-demand scaling
  - Learning curve integration
  - 25-year SBTi timeline progression
  - Wholesale LMP price degradation feedback

Constants and data structures are sourced from:
  - dispatch_utils.py (grid mix, demand, ISOS)
  - step3_cost_optimization.py (LCOE tables, uprate costs, CCS costs)
  - step6_scenario_comparison.py (learning_fraction, SBTI_YEAR_MAP)
  - step7_generate_shared_data.py (SBTI_MILESTONES, DAC_TRAJECTORY)
"""

import os
import sys
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from dispatch_utils import (
    ISOS, H, BASE_DEMAND_TWH, GRID_MIX_SHARES,
    WHOLESALE_PRICES, FUEL_ADJUSTMENTS,
    HYDRO_CAPS, NUCLEAR_SHARE_OF_CLEAN_FIRM,
    NUCLEAR_MONTHLY_CF, get_demand_profile, get_supply_profiles,
    load_common_data,
)
from step3_cost_optimization import (
    LCOE_TABLES, TX_TABLES, UPRATE_LCOE,
    NUCLEAR_NEWBUILD_LCOE, CCS_LCOE_45Q_ON, CCS_LCOE_45Q_OFF,
    GEOTHERMAL_LCOE, EXISTING_NUCLEAR_GW, UPRATE_CAP_TWH,
    DEMAND_GROWTH_RATES, THRESHOLD_TARGET_YEARS,
)
# Inline learning curve logic from step6_scenario_comparison.py
# to avoid dragging in pandas + heavy dispatch imports.
# Original source: step6_scenario_comparison.py lines 155-310

SBTI_YEAR_MAP = {
    50: 2030, 55: 2031, 60: 2033, 65: 2034, 70: 2035, 75: 2036, 80: 2037,
    85: 2038, 87.5: 2039, 90: 2040, 92.5: 2043,
    95: 2045, 97.5: 2048, 99: 2049, 99.5: 2049, 99.9: 2050, 100: 2050,
}


def learning_fraction(threshold, scenario='B'):
    """Map CFE threshold to FOAK→NOAK learning curve fraction [0, 1].

    0 = pure FOAK (High cost), 1 = full NOAK (Low cost).
    Scenario B (Hourly): FOAK until 2030, NOAK by 2040 (10yr learning).
    Scenario A (Consequential): FOAK until 2035, NOAK by 2047 (12yr learning).
    Concave exponent 0.6 (Wright's Law early doublings).
    """
    year = SBTI_YEAR_MAP.get(threshold, 2050)

    if scenario == 'B':
        foak_start = 2030
        noak_year = 2040
    else:  # Scenario A
        foak_start = 2035
        noak_year = 2047

    if year < foak_start:
        return 0.0
    if year >= noak_year:
        return 1.0

    active = (year - foak_start) / (noak_year - foak_start)
    return active ** 0.6

DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'data')
PP_DIR = os.path.join(DATA_DIR, 'step5-post-processing')

# ═══════════════════════════════════════════════════════════════════════════════
# SBTi MILESTONES & TIMELINE
# ═══════════════════════════════════════════════════════════════════════════════

SBTI_MILESTONES = [
    {'year': 2025, 'threshold': 0,     'label': 'Today'},
    {'year': 2030, 'threshold': 50,    'label': 'SBTi 50%'},
    {'year': 2035, 'threshold': 70,    'label': 'SBTi ~70%'},
    {'year': 2040, 'threshold': 90,    'label': 'SBTi 90%'},
    {'year': 2045, 'threshold': 95,    'label': 'SBTi ~95%'},
    {'year': 2050, 'threshold': 99.99, 'label': 'Net-zero'},
]

# 17 thresholds used by the optimizer (must match step3)
THRESHOLDS = [50, 55, 60, 65, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 99.5, 99.9, 99.99]

BASE_YEAR = 2025
TIMELINE_YEARS = list(range(2025, 2051))  # 26 years

# ═══════════════════════════════════════════════════════════════════════════════
# C&I DEMAND & PARTICIPATION
# ═══════════════════════════════════════════════════════════════════════════════

CI_SHARE = 0.62  # Commercial + Industrial share of total demand (EIA 2024)

# Default participation slider values
DEFAULT_HYPERSCALER_PCT = 0.055   # 5.5% of C&I load
DEFAULT_OTHER_CORP_PCT = 0.075    # 7.5% of C&I load
HYPERSCALER_CI_FRACTION = 0.084   # Hyperscalers = 8.4% of C&I demand (BNEF 2025)

# Participation range for sweep
PARTICIPATION_LEVELS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80]

# ═══════════════════════════════════════════════════════════════════════════════
# SSS (STATE-SPONSORED/SUBSCRIBED) CLEAN ENERGY
# ═══════════════════════════════════════════════════════════════════════════════
# Two-component: fixed fleet (nuclear, hydro — constant TWh) + RPS (scales with demand)
# Source: EIA-860/861, state RPS databases, ISO SOMs

SSS_FIXED_FLEET_TWH = {
    # Nuclear + large hydro committed via state contracts/mandates — constant TWh
    'CAISO': 35.0,   # Diablo Canyon (16 TWh) + state-contracted large hydro (~19 TWh)
    'ERCOT': 0.0,    # No SSS — deregulated, no nuclear subsidies
    'PJM':   95.0,   # IL ZEC/CMC (50 TWh Exelon fleet) + NJ ZEC (15 TWh) + PA nuclear (~30 TWh)
    'NYISO': 42.0,   # NY ZEC Tier 3 (FitzPatrick + Nine Mile 1&2 + Ginna) + NYPA hydro
    'NEISO': 20.0,   # Millstone (CT CCEF, ~17 TWh) + state-contracted hydro
    'MISO':  30.0,   # IL CMC/ZEC (some MISO-zone Exelon plants) + MN nuclear (Prairie Island, Monticello)
    'SPP':   5.0,    # Wolf Creek (~8 TWh shared with MISO/SPP) + small hydro
}

# RPS clean targets (fraction of total demand) — linear interpolation between years
RPS_TARGETS = {
    'CAISO': {2025: 0.60, 2030: 0.60, 2035: 0.70, 2040: 0.80, 2045: 0.90, 2050: 1.00},  # CA SB 100
    'ERCOT': {2025: 0.10, 2030: 0.12, 2035: 0.15, 2040: 0.18, 2045: 0.20, 2050: 0.22},  # TX: no RPS, market-driven
    'PJM':   {2025: 0.15, 2030: 0.22, 2035: 0.35, 2040: 0.50, 2045: 0.65, 2050: 0.80},  # Weighted avg (PA/NJ/MD/VA/IL)
    'NYISO': {2025: 0.35, 2030: 0.70, 2035: 0.80, 2040: 0.90, 2045: 0.95, 2050: 1.00},  # NY CLCPA
    'NEISO': {2025: 0.30, 2030: 0.45, 2035: 0.60, 2040: 0.75, 2045: 0.85, 2050: 1.00},  # MA/CT/ME weighted
    'MISO':  {2025: 0.12, 2030: 0.18, 2035: 0.25, 2040: 0.35, 2045: 0.45, 2050: 0.55},  # MN/IL/MI weighted
    'SPP':   {2025: 0.10, 2030: 0.12, 2035: 0.15, 2040: 0.20, 2045: 0.25, 2050: 0.30},  # KS/OK/NM modest
}

# Fraction of new RPS build that is SSS (state-contracted vs merchant)
SSS_NEW_BUILD_FRACTION = 0.40  # 40% of new RPS build goes to SSS, 60% to merchant

# ═══════════════════════════════════════════════════════════════════════════════
# PPA PREMIUM MODEL (§15.14.4)
# ═══════════════════════════════════════════════════════════════════════════════
# PPA_price = LCOE × (1 + premium_pct)
# Percentage model: developer returns scale with capital deployed (LBNL PPA tracking)

PPA_PREMIUMS = {
    'VRE':    {'Low': 0.05, 'Medium': 0.12, 'High': 0.22},
    'Firm':   {'Low': 0.12, 'Medium': 0.22, 'High': 0.38},
    'Uprate': {'Low': 0.10, 'Medium': 0.20, 'High': 0.35},
}

# 45U Production Tax Credit for existing nuclear ($/MWh, inflation-adjusted)
PTC_45U_VALUE = 15.0  # Base $15/MWh
PTC_45U_MARGIN = {'Low': 0.03, 'Medium': 0.05, 'High': 0.08}  # Margin above 45U

# EAC market proxy for existing non-nuclear clean ($/MWh)
EXISTING_EAC_PRICE = {'Low': 3.0, 'Medium': 4.0, 'High': 5.0}

# ═══════════════════════════════════════════════════════════════════════════════
# DEMAND GROWTH & TIMELINE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


def get_demand_twh_at_year(iso, year, growth_level='Medium'):
    """Get projected demand TWh for an ISO at a given year with demand growth."""
    base = BASE_DEMAND_TWH[iso]
    rate = DEMAND_GROWTH_RATES[iso][growth_level]
    years = max(0, year - BASE_YEAR)
    return base * (1 + rate) ** years


def get_threshold_for_year(year):
    """Interpolate CFE threshold target for a given year along SBTi trajectory.

    Uses THRESHOLD_TARGET_YEARS (from step3) inverted: year → threshold.
    Linearly interpolates between defined milestone thresholds.
    """
    # Build year→threshold mapping from THRESHOLD_TARGET_YEARS
    # Only use the 15 thresholds we care about
    year_to_threshold = {}
    for thr in THRESHOLDS:
        yr = THRESHOLD_TARGET_YEARS.get(thr)
        if yr is not None:
            year_to_threshold[yr] = max(year_to_threshold.get(yr, 0), thr)

    # Add endpoints
    year_to_threshold[2025] = 0.0
    sorted_years = sorted(year_to_threshold.keys())
    sorted_thresholds = [year_to_threshold[y] for y in sorted_years]

    if year <= sorted_years[0]:
        return sorted_thresholds[0]
    if year >= sorted_years[-1]:
        return sorted_thresholds[-1]

    # Linear interpolation
    for i in range(len(sorted_years) - 1):
        if sorted_years[i] <= year <= sorted_years[i + 1]:
            frac = (year - sorted_years[i]) / (sorted_years[i + 1] - sorted_years[i])
            return sorted_thresholds[i] + frac * (sorted_thresholds[i + 1] - sorted_thresholds[i])

    return sorted_thresholds[-1]


def get_ci_demand_twh(iso, year, growth_level='Medium'):
    """Get C&I demand TWh (commercial + industrial, excluding residential)."""
    return get_demand_twh_at_year(iso, year, growth_level) * CI_SHARE


def get_buyer_demand_twh(iso, year, participation_pct, growth_level='Medium'):
    """Get buyer's demand TWh at a given participation level.

    participation_pct: fraction of C&I load (0.0 to 1.0)
    """
    return get_ci_demand_twh(iso, year, growth_level) * participation_pct


# ═══════════════════════════════════════════════════════════════════════════════
# SSS ALLOCATION LOGIC
# ═══════════════════════════════════════════════════════════════════════════════


def get_rps_target_at_year(iso, year):
    """Interpolate RPS target fraction for an ISO at a given year."""
    targets = RPS_TARGETS[iso]
    years = sorted(targets.keys())
    values = [targets[y] for y in years]

    if year <= years[0]:
        return values[0]
    if year >= years[-1]:
        return values[-1]

    for i in range(len(years) - 1):
        if years[i] <= year <= years[i + 1]:
            frac = (year - years[i]) / (years[i + 1] - years[i])
            return values[i] + frac * (values[i + 1] - values[i])
    return values[-1]


def get_sss_twh(iso, year, growth_level='Medium'):
    """Get total SSS (state-sponsored/subscribed) clean TWh at a given year.

    SSS = fixed fleet (nuclear/hydro, constant) + fraction of RPS new build.
    """
    fixed = SSS_FIXED_FLEET_TWH[iso]

    # RPS-mandated clean TWh
    total_demand = get_demand_twh_at_year(iso, year, growth_level)
    rps_target = get_rps_target_at_year(iso, year)
    rps_clean_twh = rps_target * total_demand

    # Existing clean from grid mix (constant at 2025 levels)
    gm = GRID_MIX_SHARES[iso]
    existing_clean_pct = sum(gm[r] for r in ['clean_firm', 'solar', 'wind', 'offshore_wind', 'hydro']) / 100.0
    existing_clean_twh = existing_clean_pct * BASE_DEMAND_TWH[iso]

    # New RPS build needed above existing clean
    rps_new_build = max(0, rps_clean_twh - existing_clean_twh)

    # SSS portion of new RPS build
    sss_from_rps_new = rps_new_build * SSS_NEW_BUILD_FRACTION

    return fixed + sss_from_rps_new


def get_existing_clean_twh(iso):
    """Get total existing clean generation TWh (2025 baseline)."""
    gm = GRID_MIX_SHARES[iso]
    existing_pct = sum(gm[r] for r in ['clean_firm', 'solar', 'wind', 'offshore_wind', 'hydro']) / 100.0
    return existing_pct * BASE_DEMAND_TWH[iso]


def get_sss_hourly_shape(iso, gen_profiles):
    """Get 8760 hourly SSS generation shape (normalized to sum=1).

    SSS generation follows the mix of its constituent resources:
    nuclear (flat baseload with seasonal derate), solar (daytime),
    hydro (seasonal), wind (stochastic).
    """
    profiles = get_supply_profiles(iso, gen_profiles)
    sss_fixed = SSS_FIXED_FLEET_TWH[iso]

    # SSS is predominantly nuclear + hydro (the fixed fleet)
    # Weight by approximate resource composition within SSS
    nuclear_twh = NUCLEAR_SHARE_OF_CLEAN_FIRM.get(iso, 1.0) * sss_fixed * 0.80
    hydro_twh = sss_fixed * 0.20  # Rough split

    total = nuclear_twh + hydro_twh
    if total <= 0:
        return np.ones(H) / H

    # Weighted blend of nuclear + hydro profiles
    nuc_profile = np.array(profiles.get('clean_firm', [1.0/H]*H), dtype=np.float64)
    hydro_profile_raw = gen_profiles.get(iso, {}).get('2025', gen_profiles.get(iso, {}))
    if isinstance(hydro_profile_raw, dict):
        hydro_vals = hydro_profile_raw.get('hydro', [1.0/H]*H)
    else:
        hydro_vals = [1.0/H]*H
    hydro_profile = np.array(hydro_vals[:H], dtype=np.float64)

    # Normalize each profile
    nuc_sum = nuc_profile.sum()
    if nuc_sum > 0:
        nuc_profile = nuc_profile / nuc_sum
    hydro_sum = hydro_profile.sum()
    if hydro_sum > 0:
        hydro_profile = hydro_profile / hydro_sum

    # Weighted blend
    w_nuc = nuclear_twh / total
    w_hyd = hydro_twh / total
    blended = w_nuc * nuc_profile + w_hyd * hydro_profile
    blended_sum = blended.sum()
    if blended_sum > 0:
        blended = blended / blended_sum

    return blended


def get_existing_clean_hourly_shape(iso, gen_profiles):
    """Get 8760 hourly shape for ALL existing clean generation (normalized to sum=1).

    Follows the actual 8760 shape of existing resources:
    nuclear (seasonal derate), solar (daytime), wind (stochastic), hydro.
    """
    profiles = get_supply_profiles(iso, gen_profiles)
    gm = GRID_MIX_SHARES[iso]

    # Weights by resource share in existing clean
    weights = {}
    total_pct = 0
    for r in ['clean_firm', 'solar', 'wind', 'offshore_wind', 'hydro']:
        pct = gm.get(r, 0)
        weights[r] = pct
        total_pct += pct

    if total_pct <= 0:
        return np.ones(H) / H

    # Blend profiles weighted by generation share
    blended = np.zeros(H, dtype=np.float64)
    for r, pct in weights.items():
        if pct <= 0:
            continue
        p = np.array(profiles.get(r, [1.0/H]*H)[:H], dtype=np.float64)
        p_sum = p.sum()
        if p_sum > 0:
            p = p / p_sum
        blended += (pct / total_pct) * p

    blended_sum = blended.sum()
    if blended_sum > 0:
        blended = blended / blended_sum

    return blended


# ═══════════════════════════════════════════════════════════════════════════════
# LCOE & PPA PRICING
# ═══════════════════════════════════════════════════════════════════════════════


def get_lcoe(resource, iso, level='Medium'):
    """Get LCOE for a resource/ISO/sensitivity combination.

    Handles all resource types including special cases:
    - 'uprate': nuclear uprate LCOE (not ISO-specific in step3)
    - 'nuclear_newbuild': from NUCLEAR_NEWBUILD_LCOE
    - 'ccs_45q_on'/'ccs_45q_off': CCS with/without 45Q
    - 'geothermal': CAISO only
    """
    level_key = level[0] if len(level) > 1 else level  # 'Low'→'L', 'Medium'→'M', 'High'→'H'
    level_full = level  # Keep for LCOE_TABLES which use 'Low'/'Medium'/'High'

    if resource == 'uprate':
        return UPRATE_LCOE.get(level_key, 25)

    if resource == 'nuclear_newbuild':
        return NUCLEAR_NEWBUILD_LCOE.get(level_key, {}).get(iso, 100)

    if resource == 'ccs_45q_on':
        return CCS_LCOE_45Q_ON.get(level_key, {}).get(iso, 80)

    if resource == 'ccs_45q_off':
        return CCS_LCOE_45Q_OFF.get(level_key, {}).get(iso, 110)

    if resource == 'geothermal':
        return GEOTHERMAL_LCOE.get(level_key, 88)

    # Standard resources from LCOE_TABLES
    if resource in LCOE_TABLES:
        return LCOE_TABLES[resource].get(level_full, {}).get(iso, 50)

    return 50  # Fallback


def get_ppa_price(resource, iso, level='Medium', ppa_level='Medium'):
    """Get PPA price = LCOE × (1 + premium_pct).

    resource: 'solar', 'wind', 'nuclear_newbuild', 'ccs_45q_on', 'uprate', etc.
    level: LCOE sensitivity level ('Low'/'Medium'/'High')
    ppa_level: PPA premium sensitivity level ('Low'/'Medium'/'High')
    """
    lcoe = get_lcoe(resource, iso, level)

    # Determine premium category
    if resource in ('solar', 'wind'):
        premium = PPA_PREMIUMS['VRE'][ppa_level]
    elif resource == 'uprate':
        premium = PPA_PREMIUMS['Uprate'][ppa_level]
    else:  # nuclear_newbuild, ccs, geothermal, ldes, etc.
        premium = PPA_PREMIUMS['Firm'][ppa_level]

    return lcoe * (1 + premium)


def get_existing_nuclear_eac_price(iso, level='Medium', use_45u=True, use_ctr=False, ctr_value=None):
    """Get EAC price for existing nuclear (non-SSS).

    Two independent premium mechanisms (§15.14.2):
    1. 45U-based: PTC_45U + margin
    2. CTR-based: cost-to-replace delta

    Both can be on simultaneously (additive).
    """
    price = 0.0

    if use_45u:
        margin = PTC_45U_MARGIN[level]
        price += PTC_45U_VALUE * (1 + margin)

    if use_ctr and ctr_value is not None:
        price += ctr_value

    return price


# ═══════════════════════════════════════════════════════════════════════════════
# LEARNING CURVE COST ADJUSTMENT
# ═══════════════════════════════════════════════════════════════════════════════


def get_learning_adjusted_lcoe(resource, iso, threshold, scenario='B', base_level='Medium'):
    """Get LCOE adjusted by learning curve (FOAK→NOAK interpolation).

    For firm resources (nuclear, CCS, LDES, geothermal), the learning curve
    maps threshold→year→fraction, then interpolates between High (FOAK) and Low (NOAK).
    VRE and battery are already mature — no learning adjustment.
    Uprates are always Medium (existing plants).
    """
    # Resources subject to learning curve
    learning_resources = {'nuclear_newbuild', 'ccs_45q_on', 'ccs_45q_off', 'ldes', 'geothermal', 'h2'}

    if resource not in learning_resources:
        return get_lcoe(resource, iso, base_level)

    if resource == 'uprate':
        return get_lcoe('uprate', iso, 'Medium')

    frac = learning_fraction(threshold, scenario)

    # FOAK = High cost, NOAK = Low cost
    lcoe_foak = get_lcoe(resource, iso, 'High')
    lcoe_noak = get_lcoe(resource, iso, 'Low')

    return lcoe_foak * (1 - frac) + lcoe_noak * frac


def get_learning_adjusted_ppa(resource, iso, threshold, scenario='B',
                               base_level='Medium', ppa_level='Medium'):
    """Get PPA price with learning-adjusted LCOE.

    PPA = learning_adjusted_LCOE × (1 + premium_pct)
    """
    lcoe = get_learning_adjusted_lcoe(resource, iso, threshold, scenario, base_level)

    if resource in ('solar', 'wind'):
        premium = PPA_PREMIUMS['VRE'][ppa_level]
    elif resource == 'uprate':
        premium = PPA_PREMIUMS['Uprate'][ppa_level]
    else:
        premium = PPA_PREMIUMS['Firm'][ppa_level]

    return lcoe * (1 + premium)


# ═══════════════════════════════════════════════════════════════════════════════
# PROCUREMENT COST TRANCHES (Strategy 2C merit order)
# ═══════════════════════════════════════════════════════════════════════════════


def build_procurement_tranches(iso, threshold, year, scenario='B',
                                level='Medium', ppa_level='Medium',
                                use_45u=True, use_ctr=False, ctr_value=None,
                                growth_level='Medium'):
    """Build merit-order procurement tranches for Strategy 2C.

    Returns list of dicts sorted by price (cheapest first):
    [{'source': str, 'price': $/MWh, 'available_twh': float, 'category': str}, ...]

    Tranche order (§15.14.4):
    1. Existing nuclear (non-SSS) at 45U + margin or CTR
    2. Nuclear uprates at PPA price
    3. Existing hydro/solar/wind (non-SSS) at EAC market proxy
    4. New-build VRE at PPA price
    5. New-build clean firm at learning-adjusted PPA price
    """
    tranches = []
    sss_twh = get_sss_twh(iso, year, growth_level)
    existing_clean = get_existing_clean_twh(iso)

    # --- Tranche 1: Existing nuclear (non-SSS) ---
    gm = GRID_MIX_SHARES[iso]
    existing_nuclear_twh = (gm.get('clean_firm', 0) / 100.0) * BASE_DEMAND_TWH[iso]
    nuclear_non_sss = max(0, existing_nuclear_twh - sss_twh * 0.80)  # ~80% of SSS fixed is nuclear
    if nuclear_non_sss > 0:
        price = get_existing_nuclear_eac_price(iso, level, use_45u, use_ctr, ctr_value)
        tranches.append({
            'source': 'existing_nuclear',
            'price': price,
            'available_twh': nuclear_non_sss,
            'category': 'existing',
        })

    # --- Tranche 2: Nuclear uprates ---
    uprate_twh = UPRATE_CAP_TWH.get(iso, 0)
    if uprate_twh > 0:
        price = get_learning_adjusted_ppa('uprate', iso, threshold, scenario, level, ppa_level)
        tranches.append({
            'source': 'nuclear_uprate',
            'price': price,
            'available_twh': uprate_twh,
            'category': 'uprate',
        })

    # --- Tranche 3: Existing non-nuclear clean (hydro/solar/wind non-SSS) ---
    existing_non_nuclear = existing_clean - existing_nuclear_twh
    non_nuclear_non_sss = max(0, existing_non_nuclear - sss_twh * 0.20)  # ~20% of SSS is hydro
    if non_nuclear_non_sss > 0:
        price = EXISTING_EAC_PRICE[level]
        tranches.append({
            'source': 'existing_vre_hydro',
            'price': price,
            'available_twh': non_nuclear_non_sss,
            'category': 'existing',
        })

    # --- Tranche 4: New-build VRE ---
    # Available capacity is large — use 10× current capacity as practical cap
    total_demand = get_demand_twh_at_year(iso, year, growth_level)
    new_vre_cap = total_demand * 0.50  # Up to 50% of demand in new VRE
    solar_ppa = get_learning_adjusted_ppa('solar', iso, threshold, scenario, level, ppa_level)
    wind_ppa = get_learning_adjusted_ppa('wind', iso, threshold, scenario, level, ppa_level)
    avg_vre_ppa = min(solar_ppa, wind_ppa)  # Cheapest VRE
    tranches.append({
        'source': 'new_build_vre',
        'price': avg_vre_ppa,
        'available_twh': new_vre_cap,
        'category': 'new_build',
    })

    # --- Tranche 5: New-build clean firm ---
    # Nuclear new-build, CCS, geothermal — learning-adjusted
    nuc_ppa = get_learning_adjusted_ppa('nuclear_newbuild', iso, threshold, scenario, level, ppa_level)
    ccs_ppa = get_learning_adjusted_ppa('ccs_45q_on', iso, threshold, scenario, level, ppa_level)
    cheapest_firm = min(nuc_ppa, ccs_ppa)
    firm_cap = total_demand * 0.30  # Up to 30% of demand in new firm
    tranches.append({
        'source': 'new_build_firm',
        'price': cheapest_firm,
        'available_twh': firm_cap,
        'category': 'new_build',
    })

    # Sort by price (cheapest first)
    tranches.sort(key=lambda t: t['price'])

    return tranches


# ═══════════════════════════════════════════════════════════════════════════════
# WHOLESALE PRICE FEEDBACK
# ═══════════════════════════════════════════════════════════════════════════════


def get_wholesale_price(iso, fuel_level='Medium'):
    """Get wholesale electricity price with fuel adjustment."""
    base = WHOLESALE_PRICES[iso]
    adj = FUEL_ADJUSTMENTS[iso][fuel_level]
    return base + adj


def estimate_lmp_at_clean_pct(iso, clean_pct, fuel_level='Medium'):
    """Estimate average LMP at a given clean energy penetration level.

    Simple merit-order price depression model:
    - At 0% clean, LMP = wholesale base (fossil-only fleet)
    - As clean % increases, cheap clean displaces expensive fossil → LMP drops
    - At very high clean %, scarcity pricing partially offsets depression
    - Cannibalization S-curve: steep depression 30-70%, flattening at extremes

    For full 8760 LMP, use step6_compute_lmp_prices.py.
    This is a fast approximation for cost modeling.
    """
    base_lmp = get_wholesale_price(iso, fuel_level)

    # Cannibalization factor: S-curve merit-order depression
    # At 50% clean → ~15% depression, 80% → ~35%, 95% → ~50%, 99%+ → partial recovery from scarcity
    if clean_pct <= 0:
        return base_lmp

    x = clean_pct / 100.0 if clean_pct > 1.0 else clean_pct

    # Sigmoid cannibalization: depression peaks around 90-95%, with scarcity bounce at extremes
    depression = 0.55 * (1 / (1 + np.exp(-8 * (x - 0.6))))  # Max ~55% depression

    # Scarcity premium at very high penetration (gas peakers get extreme margins)
    scarcity_bounce = 0.15 * max(0, (x - 0.92)) / 0.08 if x > 0.92 else 0.0

    effective_factor = 1.0 - depression + scarcity_bounce
    return base_lmp * max(0.35, effective_factor)  # Floor at 35% of base


# ═══════════════════════════════════════════════════════════════════════════════
# EMISSION RATES
# ═══════════════════════════════════════════════════════════════════════════════

# Emission rates by accounting type (tCO₂/MWh) — from eGRID 2022 + VERACI-T
EMISSION_RATES = {
    'grid_average': {
        'CAISO': 0.210, 'ERCOT': 0.370, 'PJM': 0.370, 'NYISO': 0.210,
        'NEISO': 0.230, 'MISO': 0.430, 'SPP': 0.400,
    },
    'fossil_average': {
        'CAISO': 0.430, 'ERCOT': 0.440, 'PJM': 0.530, 'NYISO': 0.380,
        'NEISO': 0.380, 'MISO': 0.580, 'SPP': 0.530,
    },
    'marginal': {
        'CAISO': 0.440, 'ERCOT': 0.450, 'PJM': 0.540, 'NYISO': 0.390,
        'NEISO': 0.390, 'MISO': 0.680, 'SPP': 0.650,
    },
}


def get_emission_rate(iso, baseline_type='grid_average'):
    """Get emission rate (tCO₂/MWh) for an ISO and baseline type."""
    return EMISSION_RATES.get(baseline_type, {}).get(iso, 0.4)


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY RESULT CONTAINERS
# ═══════════════════════════════════════════════════════════════════════════════


def make_strategy_result(strategy_id, iso, year, threshold, participation_pct,
                         cost_per_mwh, total_cost_m, co2_abated_mmt,
                         mac_per_ton, resource_mix=None, metadata=None):
    """Create a standardized strategy result dict."""
    return {
        'strategy': strategy_id,
        'iso': iso,
        'year': year,
        'threshold': threshold,
        'participation_pct': participation_pct,
        'cost_per_mwh': round(cost_per_mwh, 2),
        'total_cost_m': round(total_cost_m, 2),
        'co2_abated_mmt': round(co2_abated_mmt, 4),
        'mac_per_ton': round(mac_per_ton, 2) if mac_per_ton is not None else None,
        'resource_mix': resource_mix or {},
        'metadata': metadata or {},
    }


def compute_co2_abated(iso, demand_twh, clean_pct, baseline_type='grid_average'):
    """Compute CO₂ abated (MMT) from clean energy procurement.

    clean_pct: fraction of demand met by clean (0.0 to 1.0)
    """
    emission_rate = get_emission_rate(iso, baseline_type)
    clean_twh = demand_twh * clean_pct
    return clean_twh * emission_rate / 1e6  # TWh × tCO₂/MWh = MtCO₂


# ═══════════════════════════════════════════════════════════════════════════════
# 25-YEAR TIMELINE BUILDER
# ═══════════════════════════════════════════════════════════════════════════════


def build_25yr_trajectory(iso, strategy_fn, participation_pct=0.10,
                          growth_level='Medium', scenario='B', level='Medium',
                          ppa_level='Medium', **kwargs):
    """Build a 25-year cost trajectory for a given strategy + ISO.

    strategy_fn: callable(iso, year, threshold, participation_pct, **kwargs)
                 → dict with at least 'cost_per_mwh', 'co2_abated_mmt'

    Returns list of 26 annual results (2025-2050), each with:
    - year, threshold (from SBTi), demand_twh, cost, co2, etc.
    """
    trajectory = []

    for year in TIMELINE_YEARS:
        threshold_pct = get_threshold_for_year(year) / 100.0
        demand_twh = get_demand_twh_at_year(iso, year, growth_level)
        buyer_demand = get_buyer_demand_twh(iso, year, participation_pct, growth_level)

        # Find nearest optimizer threshold for this year
        threshold_num = get_threshold_for_year(year)

        result = strategy_fn(
            iso=iso,
            year=year,
            threshold=threshold_num,
            participation_pct=participation_pct,
            growth_level=growth_level,
            scenario=scenario,
            level=level,
            ppa_level=ppa_level,
            **kwargs,
        )

        result['year'] = year
        result['demand_twh'] = round(demand_twh, 2)
        result['buyer_demand_twh'] = round(buyer_demand, 2)
        result['threshold'] = round(threshold_num, 2)

        trajectory.append(result)

    return trajectory


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


def save_results_json(results, filename):
    """Save results to JSON in the post-processing directory."""
    os.makedirs(PP_DIR, exist_ok=True)
    path = os.path.join(PP_DIR, filename)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved {path} ({len(results) if isinstance(results, list) else 'dict'} entries)")
    return path


def save_js_data(results, filename='procurement-strategy-data.js', var_name='PROCUREMENT_DATA'):
    """Save results as a JS file for the dashboard."""
    js_dir = os.path.join(os.path.dirname(SCRIPT_DIR), 'dashboard', 'js')
    os.makedirs(js_dir, exist_ok=True)
    path = os.path.join(js_dir, filename)

    with open(path, 'w') as f:
        f.write(f'// Auto-generated by Step 6.5 procurement strategy pipeline\n')
        f.write(f'// Generated: {__import__("datetime").datetime.now().isoformat()}\n\n')
        f.write(f'const {var_name} = ')
        json.dump(results, f, indent=2, default=str)
        f.write(';\n')

    print(f"  Saved {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=== Step 6.5 Procurement Utils — Self-Test ===\n")

    for iso in ISOS:
        print(f"\n--- {iso} ---")
        print(f"  Base demand: {BASE_DEMAND_TWH[iso]:.1f} TWh")
        print(f"  C&I demand (2025): {get_ci_demand_twh(iso, 2025):.1f} TWh")
        print(f"  C&I demand (2035, Med growth): {get_ci_demand_twh(iso, 2035):.1f} TWh")
        print(f"  C&I demand (2050, Med growth): {get_ci_demand_twh(iso, 2050):.1f} TWh")
        print(f"  Existing clean: {get_existing_clean_twh(iso):.1f} TWh")
        print(f"  SSS (2025): {get_sss_twh(iso, 2025):.1f} TWh")
        print(f"  SSS (2035): {get_sss_twh(iso, 2035):.1f} TWh")
        print(f"  SSS (2050): {get_sss_twh(iso, 2050):.1f} TWh")
        print(f"  RPS target (2030): {get_rps_target_at_year(iso, 2030)*100:.0f}%")
        print(f"  RPS target (2050): {get_rps_target_at_year(iso, 2050)*100:.0f}%")
        print(f"  Wholesale LMP: ${get_wholesale_price(iso)}/MWh")
        print(f"  LMP at 50% clean: ${estimate_lmp_at_clean_pct(iso, 50):.1f}/MWh")
        print(f"  LMP at 90% clean: ${estimate_lmp_at_clean_pct(iso, 90):.1f}/MWh")

        # PPA pricing examples
        print(f"  Solar PPA (Med): ${get_ppa_price('solar', iso, 'Medium', 'Medium'):.1f}/MWh")
        print(f"  Wind PPA (Med): ${get_ppa_price('wind', iso, 'Medium', 'Medium'):.1f}/MWh")
        print(f"  Nuc newbuild PPA (Med): ${get_ppa_price('nuclear_newbuild', iso, 'Medium', 'Medium'):.1f}/MWh")

        # Learning-adjusted at 90% threshold
        print(f"  Nuc newbuild PPA @90% (ScenB): ${get_learning_adjusted_ppa('nuclear_newbuild', iso, 90, 'B'):.1f}/MWh")
        print(f"  Nuc newbuild PPA @90% (ScenA): ${get_learning_adjusted_ppa('nuclear_newbuild', iso, 90, 'A'):.1f}/MWh")

    print("\n\n--- SBTi Timeline Interpolation ---")
    for year in [2025, 2028, 2030, 2033, 2035, 2038, 2040, 2045, 2050]:
        print(f"  {year}: {get_threshold_for_year(year):.1f}% CFE target")

    print("\n--- Learning Fraction ---")
    for thr in THRESHOLDS:
        frac_a = learning_fraction(thr, 'A')
        frac_b = learning_fraction(thr, 'B')
        yr = SBTI_YEAR_MAP.get(thr, '?')
        print(f"  {thr:6.1f}% (yr {yr}): Scenario A={frac_a:.3f}, B={frac_b:.3f}")

    print("\n=== Self-test complete ===")
