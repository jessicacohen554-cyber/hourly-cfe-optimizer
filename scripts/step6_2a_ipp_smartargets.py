#!/usr/bin/env python3
"""
Step 6.2A: IPP SMARTargets Application (V2 — Parametric Sweep)
=============================================================
Maps Step 6.1 parametric sweep results onto individual IPP fleets.

V2 changes vs V1:
- Reads new sweep parquets (smartargets_sweep_*.parquet / sweep_*_*.parquet)
  with 270 scenarios per sweep type (Conditions×Demand×PriceSens×PPA×GasFriction)
- Produces P10/P50/P90 fan bands across all 270 scenarios
- Identifies breakeven scenario conditions per parametric dimension
- Integrates PPA level and gas friction into per-plant economics
- Vectorized: no Python for-loops over the scenario dimension

Inputs:  data/step6-smartargets/sweep_{reference|power_nz|economy_nz}_*.parquet
Outputs: dashboard/js/ipp-smartargets-data.js
         data/step6-smartargets/ipp_sweep_results.parquet
"""

import json
import math
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, ROOT)

# Import shared constants from pipeline_config where available
from pipeline_config import (
    ISOS,
    WHOLESALE_PRICES, CAPACITY_MARKET_PRICES,
    EXISTING_GAS_FOM_KW_YR,
    CCS_CAP_TWH, CCS_RESIDUAL_EMISSION_RATE, CCS_45Q_CREDIT_PER_MWH,
    CCS_RETROFIT_CAPTURE_COST_KW_YR, CCS_RETROFIT_HR_PENALTY_PCT,
    CCS_RETROFIT_FOM_ADDER_KW_YR, CCS_RETROFIT_MIN_CAP_MW,
    CCS_RETROFIT_ELIGIBLE_FUELS, CCS_RETROFIT_CF, CCS_RETROFIT_EARLIEST_YEAR,
    DAC_COST_PER_TON, get_dac_cost_per_ton,
)

# ─── PATHS ───────────────────────────────────────────────────────
STEP10_DIR = os.path.join(ROOT, 'data', 'step6-smartargets')
OUT_JS = os.path.join(ROOT, 'dashboard', 'js', 'ipp-smartargets-data.js')
OUT_PARQUET = os.path.join(STEP10_DIR, 'ipp_sweep_results.parquet')
FLEET_CONFIG = os.path.join(ROOT, 'data', 'ipp_fleet_config.json')
SIM_YEARS = [2023, 2030, 2035, 2040, 2045, 2050]
SWEEP_TYPES = ['reference', 'power_nz', 'economy_nz']

# ─── FIXED O&M ($/kW-yr) — NREL ATB 2024 ────────────────────────
FIXED_OM = {
    'coal': 42,
    'gas_ccgt': 13,
    'gas_peaker': 7,
    'nuclear': 120,
    'solar': 18,
    'wind': 28,
    'battery': 20,
    'hydro': 15,
    'geothermal': 30,
    'oil': 35,
}

# ─── ELCC (fraction of nameplate for capacity revenue) ───────────
ELCC = {
    'coal': 0.85,
    'gas_ccgt': 0.90,
    'gas_peaker': 0.95,
    'nuclear': 0.95,
    'solar': 0.20,
    'wind': 0.15,
    'battery': 0.70,
    'hydro': 0.50,
    'geothermal': 0.90,
    'oil': 0.85,
}

# ─── CAPACITY MARKET PRICES ($/kW-yr) — from pipeline_config ────
CAP_PRICES = {iso: CAPACITY_MARKET_PRICES.get(iso, 0) for iso in ISOS}

# Capacity price degrades as clean % rises (per-ISO from step10)
CAP_DEGRADE_ALPHA = {
    'CAISO': 0.40, 'ERCOT': 0.0, 'PJM': 0.35, 'NYISO': 0.40,
    'NEISO': 0.35, 'MISO': 0.0, 'SPP': 0.0,
}

# ─── 45U NUCLEAR PTC (through 2032) ─────────────────────────────
NUCLEAR_45U_MWH = 15  # $/MWh for existing nuclear, expires 2032

# ─── DEFAULT HEAT RATES (BTU/kWh) for plants without explicit data
DEFAULT_HEAT_RATES = {
    'coal': 10500,
    'gas_ccgt': 6800,
    'gas_peaker': 9800,
    'oil': 10200,
}

# ─── TYPICAL CAPACITY FACTORS (clean resources, stable) ──────────
CLEAN_CF = {
    'nuclear': 0.90,
    'solar': {'CAISO': 0.27, 'ERCOT': 0.25, 'PJM': 0.20, 'NYISO': 0.18,
              'NEISO': 0.17, 'MISO': 0.21, 'SPP': 0.23},
    'wind': {'CAISO': 0.28, 'ERCOT': 0.38, 'PJM': 0.30, 'NYISO': 0.28,
             'NEISO': 0.32, 'MISO': 0.40, 'SPP': 0.42},
    'battery': 0.10,
    'hydro': 0.38,
    'geothermal': 0.85,
}

# ─── FOSSIL EMISSION RATES (tCO2/MWh) ───────────────────────────
FOSSIL_EMISSION_RATE = {
    'coal': 0.95,
    'gas_ccgt': 0.40,
    'gas_peaker': 0.65,
    'oil': 0.75,
}

# ─── PPA DISCOUNT MODEL (replicated from step10) ────────────────
PPA_PREMIUMS = {
    'VRE':    {'Low': 0.05, 'Medium': 0.12, 'High': 0.22},
    'Firm':   {'Low': 0.12, 'Medium': 0.22, 'High': 0.38},
}
PPA_MARKET_DEPTH = {
    'CAISO': 0.95, 'ERCOT': 1.00, 'PJM': 0.90, 'NYISO': 0.75,
    'NEISO': 0.65, 'MISO': 0.60, 'SPP': 0.50,
}

# ─── ACTIVE QT: NEW CLEAN DEPLOYMENT ─────────────────────────────
NEW_LCOE_2025 = {'solar': 60, 'wind': 50, 'battery': 10}
# LCOE-based learning rates (not CapEx-only). Sources:
#   Solar 24%: Bolinger et al. (2022) found ~19% LCOE LR; we use 24% to account for
#     BOS/soft cost reductions beyond module price (Swanson's Law = 20% module only).
#   Wind 15%: Bolinger et al. (2022) LCOE LR ~15% for utility-scale US wind.
#   Battery 18%: BloombergNEF 2024 Li-ion pack price LR.
LEARNING_RATE = {'solar': 0.24, 'wind': 0.15, 'battery': 0.18}
CUMULATIVE_GW_2025 = {'solar': 180, 'wind': 160, 'battery': 35}
NATIONAL_DEPLOY_GW_YR = {'solar': 40, 'wind': 15, 'battery': 12}
DEPLOY_RATE_PER_5YR = {'solar': 0.08, 'wind': 0.04, 'battery': 0.03}


FOSSIL_FUELS = {'coal', 'gas_ccgt', 'gas_peaker', 'oil'}
CLEAN_FUELS = {'nuclear', 'solar', 'wind', 'battery', 'hydro', 'geothermal'}

# ─── FLEET DATA — loaded from external JSON config ──────────────
# To add a new company, edit data/ipp_fleet_config.json — no code changes needed.

def load_fleet_config(path=None):
    """Load IPP fleet configuration from JSON file."""
    config_path = path or FLEET_CONFIG
    if not os.path.exists(config_path):
        print(f'ERROR: Fleet config not found at {config_path}')
        sys.exit(1)
    with open(config_path) as f:
        config = json.load(f)
    companies = config['companies']
    print(f'  Loaded {len(companies)} companies from {os.path.basename(config_path)}')
    return companies


# ═══════════════════════════════════════════════════════════════════════════════
# VECTORIZED PLANT ECONOMICS
# ═══════════════════════════════════════════════════════════════════════════════

def _build_plant_arrays(company):
    """Pre-compute flat numpy arrays of plant parameters for vectorized eval.

    Returns dict of arrays, each of length N_plants.
    """
    plants = company['plants']
    n = len(plants)
    cap_mw = np.array([p['cap_mw'] for p in plants], dtype=np.float64)
    gen_twh = np.array([p['gen_twh'] for p in plants], dtype=np.float64)
    co2_mt = np.array([p['co2_mt'] for p in plants], dtype=np.float64)
    hr = np.array([p.get('hr', DEFAULT_HEAT_RATES.get(p['fuel'], 7000))
                   for p in plants], dtype=np.float64)
    retire_by = np.array([p.get('retire_by', 9999) for p in plants], dtype=np.float64)
    is_fossil = np.array([p['fuel'] in FOSSIL_FUELS for p in plants])
    is_nuclear = np.array([p['fuel'] == 'nuclear' for p in plants])
    fom_kw_yr = np.array([FIXED_OM.get(p['fuel'], 20) for p in plants], dtype=np.float64)
    elcc = np.array([ELCC.get(p['fuel'], 0.5) for p in plants], dtype=np.float64)

    # Base CF for each plant
    base_cf = np.zeros(n, dtype=np.float64)
    for i, p in enumerate(plants):
        if cap_mw[i] > 0 and gen_twh[i] > 0:
            base_cf[i] = gen_twh[i] / (cap_mw[i] * 8.760 / 1000)
        elif p['fuel'] in FOSSIL_FUELS:
            base_cf[i] = 0.3
        else:
            val = CLEAN_CF.get(p['fuel'])
            if isinstance(val, dict):
                base_cf[i] = val.get(p['iso'], 0.25)
            elif val is not None:
                base_cf[i] = val
            else:
                base_cf[i] = 0.1

    # Clean CF for clean plants
    clean_cf_arr = np.zeros(n, dtype=np.float64)
    for i, p in enumerate(plants):
        if not is_fossil[i]:
            val = CLEAN_CF.get(p['fuel'])
            if isinstance(val, dict):
                clean_cf_arr[i] = val.get(p['iso'], 0.25)
            elif val is not None:
                clean_cf_arr[i] = val
            else:
                clean_cf_arr[i] = 0.1

    # Per-plant ISO index for looking up conditions
    iso_list = [p['iso'] for p in plants]
    cap_price_base = np.array([CAP_PRICES.get(p['iso'], 0) for p in plants], dtype=np.float64)
    cap_degrade = np.array([CAP_DEGRADE_ALPHA.get(p['iso'], 0) for p in plants], dtype=np.float64)

    # CCS retrofit eligibility arrays
    is_retrofit_eligible = np.array([
        p['fuel'] in CCS_RETROFIT_ELIGIBLE_FUELS
        and p['cap_mw'] >= CCS_RETROFIT_MIN_CAP_MW
        and CCS_CAP_TWH.get(p['iso'], 0) > 0
        for p in plants
    ])
    # ISO index for CCS cap tracking (np.bincount)
    iso_names_unique = sorted(set(iso_list))
    iso_to_idx = {iso: idx for idx, iso in enumerate(iso_names_unique)}
    iso_idx = np.array([iso_to_idx[iso] for iso in iso_list], dtype=np.int64)
    iso_ccs_cap_twh = np.array([CCS_CAP_TWH.get(iso, 0) for iso in iso_names_unique],
                                dtype=np.float64)

    return {
        'n': n,
        'cap_mw': cap_mw,
        'gen_twh': gen_twh,
        'co2_mt': co2_mt,
        'hr': hr,
        'retire_by': retire_by,
        'is_fossil': is_fossil,
        'is_nuclear': is_nuclear,
        'fom_kw_yr': fom_kw_yr,
        'elcc': elcc,
        'base_cf': base_cf,
        'clean_cf': clean_cf_arr,
        'iso_list': iso_list,
        'cap_price_base': cap_price_base,
        'cap_degrade': cap_degrade,
        'is_retrofit_eligible': is_retrofit_eligible,
        'iso_idx': iso_idx,
        'iso_names_unique': iso_names_unique,
        'iso_ccs_cap_twh': iso_ccs_cap_twh,
    }


def _vectorized_plant_economics(pa, clean_pct_per_plant, avg_lmp_per_plant,
                                 gas_friction_per_plant, year):
    """Compute economics for ALL plants in one vectorized pass.

    Args:
        pa: plant arrays from _build_plant_arrays
        clean_pct_per_plant: (N,) array of clean_pct for each plant's ISO
        avg_lmp_per_plant: (N,) array of avg_lmp for each plant's ISO
        gas_friction_per_plant: (N,) scalar gas_friction for the scenario
        year: int

    Returns:
        cf (N,), gen_twh (N,), co2_mt (N,), revenue_m (N,), cost_m (N,),
        profit_m (N,), status (N,) string array
    """
    n = pa['n']
    cap_mw = pa['cap_mw']
    hr = pa['hr']
    is_fossil = pa['is_fossil']
    is_nuclear = pa['is_nuclear']
    retire_by = pa['retire_by']
    base_cf = pa['base_cf']
    clean_cf = pa['clean_cf']

    # ── Forced retirement mask ──
    retired_mask = year >= retire_by

    # ── Fossil CF via merit-order ──
    fossil_frac = np.maximum(0.0, (100.0 - clean_pct_per_plant) / 100.0)
    hr_min, hr_max = 6200.0, 11500.0
    marginal_hr = hr_min + (hr_max - hr_min) * np.power(fossil_frac, 0.6)

    # Three zones: well above marginal, near marginal, below marginal
    above = hr > (marginal_hr + 500)
    near = (hr > marginal_hr) & (~above)
    below = ~above & ~near

    fossil_cf = np.zeros(n, dtype=np.float64)
    # Below marginal — dispatches at scaled CF
    scale = np.minimum(1.0, fossil_frac / 0.3)
    fossil_cf[below] = np.maximum(0.05, base_cf[below] * scale[below])
    # Near marginal — linear ramp
    frac_near = 1.0 - (hr - marginal_hr) / 500.0
    near_cf = np.maximum(0.02, base_cf * frac_near * fossil_frac / 0.5)
    fossil_cf[near] = near_cf[near]
    # Above marginal — zero
    fossil_cf[above] = 0.0
    # Zero fossil frac → zero dispatch
    fossil_cf[fossil_frac <= 0.01] = 0.0

    # ── Combined CF ──
    cf = np.where(is_fossil, fossil_cf, clean_cf)
    cf[retired_mask] = 0.0

    # ── Generation ──
    gen_twh_out = cap_mw * cf * 8.760 / 1000.0
    gen_mwh = gen_twh_out * 1e6

    # ── CO2 ──
    co2_ratio = np.where(base_cf > 0, cf / base_cf, 0.0)
    co2_mt_out = pa['co2_mt'] * co2_ratio
    co2_mt_out[retired_mask] = 0.0

    # ── Revenue ──
    energy_rev_m = gen_mwh * avg_lmp_per_plant / 1e6

    # Capacity revenue (degrades with clean %)
    cap_price_eff = pa['cap_price_base'] * np.maximum(0.0, 1.0 - pa['cap_degrade'] * clean_pct_per_plant / 100.0)
    capacity_rev_m = cap_mw * pa['elcc'] * cap_price_eff / 1e3

    # 45U nuclear PTC
    ptc_rev_m = np.zeros(n, dtype=np.float64)
    if year <= 2032:
        ptc_rev_m[is_nuclear] = gen_mwh[is_nuclear] * NUCLEAR_45U_MWH / 1e6

    total_rev_m = energy_rev_m + capacity_rev_m + ptc_rev_m

    # ── Cost ──
    fixed_om_m = cap_mw * pa['fom_kw_yr'] / 1e3

    # ── Profit ──
    profit_m = total_rev_m - fixed_om_m

    # ── Status ──
    status = np.full(n, 'operating', dtype='U16')
    status[retired_mask | (cf <= 0.01)] = 'retired'
    status[(~retired_mask) & (cf > 0.01) & (profit_m < -5)] = 'stranded'
    status[(~retired_mask) & (cf > 0.01) & (profit_m >= -5) & (profit_m < 0)] = 'marginal'

    return cf, gen_twh_out, co2_mt_out, total_rev_m, fixed_om_m, profit_m, status


def _vectorized_retrofit_economics(pa, status, profit_m, co2_mt_out,
                                    avg_lmp_per_plant, year,
                                    cost_level='Medium', is_45q_on=True):
    """Evaluate CCS retrofit for stranded/marginal gas CCGT plants.

    Pure economic trigger: retrofit when retrofit profit beats stranding cost
    AND is cheaper than DAC offsets for the same CO2 reduction.

    Returns:
        retrofit_mask (N,): boolean, True if plant retrofits
        retro_profit_m (N,): profit after retrofit ($M/yr)
        retro_co2_mt (N,): emissions after retrofit (Mt/yr)
        retro_gen_twh (N,): generation after retrofit (TWh/yr)
        retro_cost_m (N,): total cost after retrofit ($M/yr)
        retro_rev_m (N,): total revenue after retrofit ($M/yr)
        retro_45q_m (N,): 45Q credit revenue ($M/yr)
        retro_co2_avoided_mt (N,): CO2 avoided vs unabated baseline (Mt/yr)
    """
    n = pa['n']
    cap_mw = pa['cap_mw']
    eligible = pa['is_retrofit_eligible'].copy()

    # Only stranded/marginal plants can retrofit, and only after earliest year
    can_retrofit = ((status == 'stranded') | (status == 'marginal')) & (year >= CCS_RETROFIT_EARLIEST_YEAR)
    eligible = eligible & can_retrofit

    # Retrofit generation: plant runs at CCS_RETROFIT_CF (flat baseload)
    retro_gen_twh = cap_mw * CCS_RETROFIT_CF * 8.760 / 1000.0

    # Retrofit revenue: energy + 45Q credit
    retro_energy_rev_m = retro_gen_twh * 1e6 * avg_lmp_per_plant / 1e6
    retro_45q_m = np.zeros(n, dtype=np.float64)
    if is_45q_on:
        retro_45q_m = retro_gen_twh * 1e6 * CCS_45Q_CREDIT_PER_MWH / 1e6
    retro_rev_m = retro_energy_rev_m + retro_45q_m

    # Retrofit cost: existing FOM + capture FOM adder + capture capex annualized
    fom_adder = CCS_RETROFIT_FOM_ADDER_KW_YR.get(cost_level, 6.5)
    capture_cost_arr = np.array([
        CCS_RETROFIT_CAPTURE_COST_KW_YR.get(cost_level, {}).get(iso, 45)
        for iso in pa['iso_list']
    ], dtype=np.float64)
    retro_cost_m = cap_mw * (pa['fom_kw_yr'] + fom_adder + capture_cost_arr) / 1e3

    # Retrofit profit
    retro_profit_m = retro_rev_m - retro_cost_m

    # Retrofit CO2: residual emissions at 90% capture
    retro_co2_mt = retro_gen_twh * 1e6 * CCS_RESIDUAL_EMISSION_RATE / 1e6

    # CO2 avoided vs unabated baseline
    # Use the plant's original co2_mt (from fleet config) as baseline
    baseline_co2 = pa['co2_mt']
    retro_co2_avoided_mt = np.maximum(0, baseline_co2 - retro_co2_mt)

    # DAC alternative cost: what it would cost to offset the baseline CO2 with DAC
    dac_cost_per_ton = get_dac_cost_per_ton(year, cost_level)
    dac_alt_cost_m = baseline_co2 * 1e6 * dac_cost_per_ton / 1e6  # $M

    # Decision: retrofit if (a) eligible, (b) retrofit loss is tolerable (<$2M/yr),
    # and (c) retrofit annual cost is less than DAC offset cost
    retrofit_mask = eligible & (retro_profit_m > -2.0) & (retro_cost_m < dac_alt_cost_m)

    # CCS cap enforcement: check cumulative retrofit TWh per ISO doesn't exceed cap
    if retrofit_mask.any():
        retro_twh_by_iso = np.bincount(
            pa['iso_idx'][retrofit_mask],
            weights=retro_gen_twh[retrofit_mask],
            minlength=len(pa['iso_names_unique'])
        )
        # If any ISO exceeds cap, un-retrofit most expensive plants in that ISO
        for iso_i, iso_name in enumerate(pa['iso_names_unique']):
            cap_twh = pa['iso_ccs_cap_twh'][iso_i]
            if retro_twh_by_iso[iso_i] > cap_twh and cap_twh > 0:
                # Find plants in this ISO that are retrofitting, sorted by cost (desc)
                iso_retro = np.where(retrofit_mask & (pa['iso_idx'] == iso_i))[0]
                costs = retro_cost_m[iso_retro]
                order = np.argsort(-costs)  # Most expensive first
                cum_twh = 0.0
                for idx in order:
                    plant_idx = iso_retro[idx]
                    cum_twh += retro_gen_twh[plant_idx]
                    if cum_twh > cap_twh:
                        retrofit_mask[plant_idx] = False

    # Zero out values for non-retrofitting plants
    retro_profit_m = np.where(retrofit_mask, retro_profit_m, 0.0)
    retro_co2_mt = np.where(retrofit_mask, retro_co2_mt, 0.0)
    retro_gen_twh = np.where(retrofit_mask, retro_gen_twh, 0.0)
    retro_cost_m = np.where(retrofit_mask, retro_cost_m, 0.0)
    retro_rev_m = np.where(retrofit_mask, retro_rev_m, 0.0)
    retro_45q_m = np.where(retrofit_mask, retro_45q_m, 0.0)
    retro_co2_avoided_mt = np.where(retrofit_mask, retro_co2_avoided_mt, 0.0)

    return (retrofit_mask, retro_profit_m, retro_co2_mt, retro_gen_twh,
            retro_cost_m, retro_rev_m, retro_45q_m, retro_co2_avoided_mt)


# ═══════════════════════════════════════════════════════════════════════════════
# ACTIVE DEPLOYMENT (Wright's Law + PPA discount)
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_lcoe(tech, year):
    """Wright's Law LCOE for new-build clean resources."""
    base_lcoe = NEW_LCOE_2025[tech]
    base_cum = CUMULATIVE_GW_2025[tech]
    annual_deploy = NATIONAL_DEPLOY_GW_YR[tech]
    lr = LEARNING_RATE[tech]
    years_elapsed = year - 2025
    cum_gw = base_cum + annual_deploy * years_elapsed
    exponent = math.log2(1 - lr)
    return base_lcoe * (cum_gw / base_cum) ** exponent


def _get_ppa_discount_for_new_clean(tech, ppa_level, iso):
    """PPA discount for new-build clean, scaled by regional market depth."""
    if ppa_level is None or ppa_level == 'None':
        return 0.0
    category = 'VRE' if tech in ('solar', 'wind', 'battery') else 'Firm'
    base_discount = PPA_PREMIUMS.get(category, {}).get(ppa_level, 0)
    depth = PPA_MARKET_DEPTH.get(iso, 0.75)
    return base_discount * depth


def compute_active_deployment(company, year, ppa_level=None):
    """Compute cumulative new clean capacity (MW) deployed by the IPP by year.

    Returns dict of {tech: {cum_mw, gen_twh, lcoe, cost_m, cf}} and total annual cost $M.
    """
    cap_gw = company['cap_gw']
    periods_elapsed = (year - 2025) / 5.0
    isos = list(set(p['iso'] for p in company['plants']))

    deployment = {}
    annual_cost_m = 0

    for tech in ['solar', 'wind', 'battery']:
        rate = DEPLOY_RATE_PER_5YR[tech]
        cum_mw = cap_gw * rate * periods_elapsed * 1000

        if tech == 'battery':
            cf = 0.10
        elif tech == 'solar':
            cfs = [CLEAN_CF['solar'].get(iso, 0.22) for iso in isos]
            cf = sum(cfs) / len(cfs) if cfs else 0.22
        else:
            cfs = [CLEAN_CF['wind'].get(iso, 0.32) for iso in isos]
            cf = sum(cfs) / len(cfs) if cfs else 0.32

        gen_twh = cum_mw * cf * 8.760 / 1000
        lcoe = _compute_lcoe(tech, year)

        # Apply PPA discount per ISO (average across company ISOs)
        if ppa_level and ppa_level != 'None':
            avg_disc = np.mean([_get_ppa_discount_for_new_clean(tech, ppa_level, iso)
                                for iso in isos])
            lcoe *= (1 - avg_disc)

        cost_m = gen_twh * lcoe
        deployment[tech] = {
            'cum_mw': round(cum_mw),
            'gen_twh': round(gen_twh, 2),
            'lcoe': round(lcoe, 1),
            'cost_m': round(cost_m, 1),
            'cf': round(cf, 3),
        }
        annual_cost_m += cost_m

    return deployment, round(annual_cost_m, 1)


def compute_active_emission_reduction(company, deployment):
    """New clean generation displaces highest-emitting fossil plants first.
    Returns additional emission reduction in Mt.
    """
    new_clean_twh = sum(d['gen_twh'] for d in deployment.values())
    fossil_plants = [p for p in company['plants'] if p['fuel'] in FOSSIL_FUELS]
    fossil_plants.sort(key=lambda p: FOSSIL_EMISSION_RATE.get(p['fuel'], 0.5), reverse=True)

    displaced_co2 = 0.0
    remaining_twh = new_clean_twh
    for plant in fossil_plants:
        if remaining_twh <= 0:
            break
        displace = min(remaining_twh, plant['gen_twh'])
        displaced_co2 += displace * FOSSIL_EMISSION_RATE.get(plant['fuel'], 0.5)
        remaining_twh -= displace
    return round(displaced_co2, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# SWEEP DATA LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def load_sweep_data():
    """Load all sweep parquets into a single DataFrame.

    Handles both naming conventions:
    - smartargets_sweep_{type}.parquet (single file per type)
    - sweep_{type}_{ISO}.parquet (per-ISO files)
    """
    frames = []
    for sweep_type in SWEEP_TYPES:
        # Try single file first
        single_path = os.path.join(STEP10_DIR, f'smartargets_sweep_{sweep_type}.parquet')
        if os.path.exists(single_path):
            df = pd.read_parquet(single_path)
            df['sweep_type'] = sweep_type
            frames.append(df)
            continue

        # Try per-ISO files
        for iso in ISOS:
            path = os.path.join(STEP10_DIR, f'sweep_{sweep_type}_{iso}.parquet')
            if os.path.exists(path):
                df = pd.read_parquet(path)
                df['sweep_type'] = sweep_type
                frames.append(df)

    if not frames:
        print('ERROR: No sweep parquets found in', STEP10_DIR)
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    print(f'  Loaded {len(combined)} rows, {combined["scenario"].nunique()} unique scenarios, '
          f'{len(combined["iso"].unique())} ISOs, sweep types: {combined["sweep_type"].unique().tolist()}')
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SIMULATION — VECTORIZED OVER SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_company_all_scenarios(company, sweep_df):
    """Simulate one company across ALL sweep scenarios × years.

    Vectorized: builds scenario conditions into arrays, broadcasts plant parameters.

    Returns:
        results_df: DataFrame with one row per (scenario, year, company) with
                    fleet_emissions_mt, fleet_profit_m, operating_mw, stranded_mw,
                    retired_mw, fleet_revenue_m, fleet_cost_m
        plant_detail: dict for dashboard (only P50 scenario)
    """
    pa = _build_plant_arrays(company)
    plants = company['plants']
    n_plants = pa['n']

    # Get unique (scenario, sweep_type) pairs to avoid collisions across sweep types
    # Same scenario name exists in reference/power_nz/economy_nz with different clean_pct
    scenario_sweep_pairs = sweep_df[['scenario', 'sweep_type']].drop_duplicates()
    scenarios_list = list(scenario_sweep_pairs.itertuples(index=False, name=None))
    n_scenarios = len(scenarios_list)

    # Build lookup: for each plant, find its ISO in the sweep data
    plant_isos = pa['iso_list']
    unique_isos = list(set(plant_isos))

    all_results = []

    for year in SIM_YEARS:
        if year == 2023:
            # Baseline year: all plants at reported values
            baseline_em = company['co2_2023_mt']
            fleet_rev = sum(p['gen_twh'] * 1e6 * 35 / 1e6 for p in plants)
            fleet_cost = sum(p['cap_mw'] * FIXED_OM.get(p['fuel'], 20) / 1e3 for p in plants)
            total_cap = sum(p['cap_mw'] for p in plants)

            for scn, st in scenarios_list:
                # Get sweep metadata for this scenario+sweep_type
                scn_st_df = sweep_df[(sweep_df['scenario'] == scn) & (sweep_df['sweep_type'] == st)]
                scn_row = scn_st_df.iloc[0] if not scn_st_df.empty else sweep_df[sweep_df['scenario'] == scn].iloc[0]
                all_results.append({
                    'scenario': f'{scn}__{st}',
                    'sweep_type': st,
                    'conditions': str(scn_row.get('conditions', '')),
                    'demand_growth': str(scn_row.get('demand_growth', '')),
                    'price_sens': str(scn_row.get('price_sens', '')),
                    'ppa_level': str(scn_row.get('ppa_level', 'None')),
                    'gas_friction': float(scn_row.get('gas_friction', 0.7)),
                    'year': year,
                    'fleet_emissions_mt': baseline_em,
                    'fleet_revenue_m': round(fleet_rev, 1),
                    'fleet_cost_m': round(fleet_cost, 1),
                    'fleet_profit_m': round(fleet_rev - fleet_cost, 1),
                    'operating_mw': total_cap,
                    'stranded_mw': 0,
                    'retired_mw': 0,
                    'retrofitted_mw': 0,
                    'retrofit_co2_avoided_mt': 0.0,
                    'retrofit_cost_m': 0.0,
                    'retrofit_revenue_m': 0.0,
                    'retrofit_45q_revenue_m': 0.0,
                })
            continue

        # For non-baseline years: vectorize across scenarios
        # Build condition arrays: for each (scenario, sweep_type) × plant, look up clean_pct & avg_lmp
        year_df = sweep_df[sweep_df['year'] == year]
        if year_df.empty:
            continue

        # Create a lookup dict: (scenario, sweep_type) → {iso → row}
        scn_iso_lookup = {}
        for _, row in year_df.iterrows():
            key = (row['scenario'], row['sweep_type'])
            iso = row['iso']
            if key not in scn_iso_lookup:
                scn_iso_lookup[key] = {}
            scn_iso_lookup[key][iso] = row

        # Process scenarios in bulk — vectorize over plants for each scenario
        # (We iterate scenarios but vectorize the N_plants dimension)
        for scn, st in scenarios_list:
            key = (scn, st)
            iso_data = scn_iso_lookup.get(key, {})
            if not iso_data:
                continue

            # Build per-plant condition arrays from scenario's ISO data
            clean_pct_arr = np.full(n_plants, 50.0)
            avg_lmp_arr = np.full(n_plants, 30.0)
            gas_friction_val = 0.7

            for i, iso in enumerate(plant_isos):
                row = iso_data.get(iso)
                if row is not None:
                    clean_pct_arr[i] = float(row.get('clean_pct', 50))
                    avg_lmp_arr[i] = float(row.get('avg_lmp', 30))
                    gas_friction_val = float(row.get('gas_friction', 0.7))

            gas_friction_arr = np.full(n_plants, gas_friction_val)

            cf, gen_twh_out, co2_mt_out, rev_m, cost_m, profit_m, status = \
                _vectorized_plant_economics(pa, clean_pct_arr, avg_lmp_arr,
                                            gas_friction_arr, year)

            # ── CCS Retrofit evaluation ──
            # Determine cost level from scenario metadata
            any_row = next(iter(iso_data.values()))
            cost_level = str(any_row.get('price_sens', 'Medium'))
            if cost_level not in ('Low', 'Medium', 'High'):
                cost_level = 'Medium'

            (retrofit_mask, retro_profit, retro_co2, retro_gen,
             retro_cost, retro_rev, retro_45q, retro_co2_avoided) = \
                _vectorized_retrofit_economics(
                    pa, status, profit_m, co2_mt_out,
                    avg_lmp_arr, year, cost_level=cost_level)

            # Override economics for retrofitted plants
            if retrofit_mask.any():
                profit_m = np.where(retrofit_mask, retro_profit, profit_m)
                co2_mt_out = np.where(retrofit_mask, retro_co2, co2_mt_out)
                gen_twh_out = np.where(retrofit_mask, retro_gen, gen_twh_out)
                cost_m = np.where(retrofit_mask, retro_cost, cost_m)
                rev_m = np.where(retrofit_mask, retro_rev, rev_m)
                status = np.where(retrofit_mask, 'retrofitted', status)

            # Aggregate fleet metrics
            fleet_em = float(co2_mt_out.sum())
            fleet_rev = float(rev_m.sum())
            fleet_cost = float(cost_m.sum())
            fleet_profit = float(profit_m.sum())

            # MW by status
            cap = pa['cap_mw']
            op_mask = (status == 'operating')
            ret_mask = (status == 'retired')
            retro_mask_status = (status == 'retrofitted')
            strand_mask = ~op_mask & ~ret_mask & ~retro_mask_status

            all_results.append({
                'scenario': f'{scn}__{st}',
                'sweep_type': st,
                'conditions': str(any_row.get('conditions', '')),
                'demand_growth': str(any_row.get('demand_growth', '')),
                'price_sens': str(any_row.get('price_sens', '')),
                'ppa_level': str(any_row.get('ppa_level', 'None')),
                'gas_friction': float(any_row.get('gas_friction', 0.7)),
                'year': year,
                'fleet_emissions_mt': round(fleet_em, 2),
                'fleet_revenue_m': round(fleet_rev, 1),
                'fleet_cost_m': round(fleet_cost, 1),
                'fleet_profit_m': round(fleet_profit, 1),
                'operating_mw': int(cap[op_mask].sum()),
                'stranded_mw': int(cap[strand_mask].sum()),
                'retired_mw': int(cap[ret_mask].sum()),
                'retrofitted_mw': int(cap[retro_mask_status].sum()),
                'retrofit_co2_avoided_mt': round(float(retro_co2_avoided.sum()), 3),
                'retrofit_cost_m': round(float(retro_cost.sum()), 1),
                'retrofit_revenue_m': round(float(retro_rev.sum()), 1),
                'retrofit_45q_revenue_m': round(float(retro_45q.sum()), 1),
            })

    return pd.DataFrame(all_results)


def compute_fan_bands(results_df, metric, percentiles=(10, 50, 90)):
    """Compute P10/P50/P90 fan bands for a metric across scenarios per year.

    Returns dict with keys 'p10', 'p50', 'p90', 'min', 'max', each a list per year.
    """
    bands = {f'p{p}': [] for p in percentiles}
    bands['min'] = []
    bands['max'] = []

    for year in SIM_YEARS:
        year_vals = results_df[results_df['year'] == year][metric].values
        if len(year_vals) == 0:
            for key in bands:
                bands[key].append(0)
            continue
        for p in percentiles:
            bands[f'p{p}'].append(round(float(np.percentile(year_vals, p)), 2))
        bands['min'].append(round(float(year_vals.min()), 2))
        bands['max'].append(round(float(year_vals.max()), 2))

    return bands


def compute_breakeven_analysis(results_df):
    """Identify which parametric dimension combos produce profitable decarbonization.

    Returns dict with per-dimension breakdowns:
    {dimension: {value: {profitable_pct, avg_profit_2050, avg_emissions_2050}}}
    """
    # Focus on 2050 results
    df_2050 = results_df[results_df['year'] == 2050].copy()
    if df_2050.empty:
        return {}

    df_2050['profitable'] = df_2050['fleet_profit_m'] >= 0

    dimensions = ['conditions', 'demand_growth', 'price_sens', 'ppa_level', 'gas_friction']
    analysis = {}

    for dim in dimensions:
        if dim not in df_2050.columns:
            continue
        groups = df_2050.groupby(dim)
        dim_data = {}
        for val, group in groups:
            dim_data[str(val)] = {
                'profitable_pct': round(float(group['profitable'].mean() * 100), 1),
                'avg_profit_2050': round(float(group['fleet_profit_m'].mean()), 1),
                'avg_emissions_2050': round(float(group['fleet_emissions_mt'].mean()), 2),
                'n_scenarios': int(len(group)),
            }
        analysis[dim] = dim_data

    return analysis


def compute_active_qt_fan(company, results_df, sweep_df):
    """Compute active QT (new clean deployment) fan bands.

    For each scenario, layers new clean deployment on top of passive fleet economics.
    PPA level from each scenario adjusts Wright's Law LCOE.

    Returns results_df with additional active_* columns.
    """
    active_results = []

    for _, row in results_df.iterrows():
        year = row['year']
        if year == 2023:
            active_results.append({
                'active_emissions_mt': row['fleet_emissions_mt'],
                'active_profit_m': row['fleet_profit_m'],
            })
            continue

        ppa_level = row.get('ppa_level', 'None')
        deployment, deploy_cost_m = compute_active_deployment(company, year, ppa_level)
        em_reduction = compute_active_emission_reduction(company, deployment)
        active_em = max(0, row['fleet_emissions_mt'] - em_reduction)

        # Revenue from new clean
        new_gen_twh = sum(d['gen_twh'] for d in deployment.values())
        # Use scenario's average LMP (approximate from fleet revenue)
        avg_lmp = 30.0  # fallback
        # Try to get from sweep data
        scn = row['scenario']
        scn_rows = sweep_df[(sweep_df['scenario'] == scn) & (sweep_df['year'] == year)]
        if not scn_rows.empty:
            avg_lmp = float(scn_rows['avg_lmp'].mean())

        new_energy_rev_m = new_gen_twh * avg_lmp
        isos = list(set(p['iso'] for p in company['plants']))
        new_cap_rev_m = 0
        for tech, d in deployment.items():
            el = ELCC.get(tech, 0.2)
            avg_cap_price = sum(CAP_PRICES.get(iso, 0) for iso in isos) / len(isos)
            new_cap_rev_m += d['cum_mw'] * el * avg_cap_price / 1e3

        net_clean_profit = new_energy_rev_m + new_cap_rev_m - deploy_cost_m
        active_profit = row['fleet_profit_m'] + net_clean_profit

        active_results.append({
            'active_emissions_mt': round(active_em, 2),
            'active_profit_m': round(active_profit, 1),
        })

    active_df = pd.DataFrame(active_results)
    return active_df


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print('Step 6.2A: IPP SMARTargets Application (V2 — Parametric Sweep)')
    print('=' * 60)

    companies = load_fleet_config()
    sweep_df = load_sweep_data()

    output = {'companies': {}, 'sweep_types': SWEEP_TYPES, 'years': SIM_YEARS}
    all_company_results = []

    for company in companies:
        cid = company['id']
        cname = company['name']
        print(f'\n  Processing {cname}...')

        # Simulate across all scenarios
        results_df = simulate_company_all_scenarios(company, sweep_df)
        results_df['company'] = cid
        all_company_results.append(results_df)

        n_scn = results_df['scenario'].nunique()
        print(f'    {n_scn} scenarios × {len(SIM_YEARS)} years = {len(results_df)} rows')

        # Compute active QT overlay
        print(f'    Computing active deployment overlay...')
        active_df = compute_active_qt_fan(company, results_df, sweep_df)
        results_df = pd.concat([results_df.reset_index(drop=True),
                                active_df.reset_index(drop=True)], axis=1)

        # Build dashboard output
        co_output = {
            'name': cname,
            'shortName': company['shortName'],
            'co2_2023_mt': company['co2_2023_mt'],
            'co2_2024_mt': company['co2_2024_mt'],
            'intensity_kg': company['intensity_kg'],
            'gen_twh': company['gen_twh'],
            'cap_gw': company['cap_gw'],
            'target': company['target'],
        }

        # Fleet summary by ISO and fuel
        iso_summary = {}
        for plant in company['plants']:
            iso = plant['iso']
            fuel = plant['fuel']
            if iso not in iso_summary:
                iso_summary[iso] = {}
            if fuel not in iso_summary[iso]:
                iso_summary[iso][fuel] = {'cap_mw': 0, 'gen_twh': 0, 'co2_mt': 0}
            iso_summary[iso][fuel]['cap_mw'] += plant['cap_mw']
            iso_summary[iso][fuel]['gen_twh'] += plant['gen_twh']
            iso_summary[iso][fuel]['co2_mt'] += plant['co2_mt']
        co_output['fleet_summary'] = iso_summary

        # Fan bands — overall and per sweep type
        print(f'    Computing fan bands...')
        co_output['fan_bands'] = {}

        # Overall fan bands (all sweep types combined)
        co_output['fan_bands']['all'] = {
            'emissions': compute_fan_bands(results_df, 'fleet_emissions_mt'),
            'profit': compute_fan_bands(results_df, 'fleet_profit_m'),
            'operating_mw': compute_fan_bands(results_df, 'operating_mw'),
            'stranded_mw': compute_fan_bands(results_df, 'stranded_mw'),
        }

        # Active fan bands
        if 'active_emissions_mt' in results_df.columns:
            co_output['fan_bands']['all']['active_emissions'] = \
                compute_fan_bands(results_df, 'active_emissions_mt')
            co_output['fan_bands']['all']['active_profit'] = \
                compute_fan_bands(results_df, 'active_profit_m')

        # Per sweep type fan bands
        for st in SWEEP_TYPES:
            st_df = results_df[results_df['sweep_type'] == st]
            if st_df.empty:
                continue
            co_output['fan_bands'][st] = {
                'emissions': compute_fan_bands(st_df, 'fleet_emissions_mt'),
                'profit': compute_fan_bands(st_df, 'fleet_profit_m'),
                'operating_mw': compute_fan_bands(st_df, 'operating_mw'),
                'stranded_mw': compute_fan_bands(st_df, 'stranded_mw'),
            }
            if 'active_emissions_mt' in st_df.columns:
                co_output['fan_bands'][st]['active_emissions'] = \
                    compute_fan_bands(st_df, 'active_emissions_mt')
                co_output['fan_bands'][st]['active_profit'] = \
                    compute_fan_bands(st_df, 'active_profit_m')

        # Per-dimension filter fan bands (for dashboard controls)
        co_output['dimension_fans'] = {}
        for dim in ['conditions', 'demand_growth', 'price_sens', 'ppa_level', 'gas_friction']:
            if dim not in results_df.columns:
                continue
            dim_fans = {}
            for val in results_df[dim].unique():
                val_df = results_df[results_df[dim] == val]
                dim_fans[str(val)] = {
                    'emissions': compute_fan_bands(val_df, 'fleet_emissions_mt'),
                    'profit': compute_fan_bands(val_df, 'fleet_profit_m'),
                }
            co_output['dimension_fans'][dim] = dim_fans

        # Breakeven analysis
        print(f'    Computing breakeven analysis...')
        co_output['breakeven'] = compute_breakeven_analysis(results_df)

        # Print summary
        em_2050 = results_df[results_df['year'] == 2050]['fleet_emissions_mt']
        prof_2050 = results_df[results_df['year'] == 2050]['fleet_profit_m']
        if not em_2050.empty:
            print(f'    2050 emissions: P10={em_2050.quantile(0.1):.1f}, '
                  f'P50={em_2050.median():.1f}, P90={em_2050.quantile(0.9):.1f} Mt')
            print(f'    2050 profit:   P10=${prof_2050.quantile(0.1):.0f}M, '
                  f'P50=${prof_2050.median():.0f}M, P90=${prof_2050.quantile(0.9):.0f}M')

        output['companies'][cid] = co_output

    # ── Combine all results for parquet output ──
    full_results = pd.concat(all_company_results, ignore_index=True)

    # ── Write parquet ──
    full_results.to_parquet(OUT_PARQUET, index=False)
    parquet_kb = os.path.getsize(OUT_PARQUET) / 1024
    print(f'\n  Written {OUT_PARQUET} ({parquet_kb:.0f} KB, {len(full_results)} rows)')

    # ── Write JS file ──
    json_str = json.dumps(output, separators=(',', ':'))
    js_content = f'// Auto-generated by step11_ipp_smartargets.py (V2) — do not edit manually\nconst IPP_SMARTARGETS_DATA = {json_str};\n'

    os.makedirs(os.path.dirname(OUT_JS), exist_ok=True)
    with open(OUT_JS, 'w') as f:
        f.write(js_content)

    size_kb = os.path.getsize(OUT_JS) / 1024
    print(f'  Written {OUT_JS} ({size_kb:.0f} KB)')
    print('Done.')


if __name__ == '__main__':
    main()
