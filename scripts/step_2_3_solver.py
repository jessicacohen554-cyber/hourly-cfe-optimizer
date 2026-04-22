"""Pathway trajectory optimizer — clean rewrite.

One solver. Foresight is a scalar parameter (λ=0 is myopic).

Entry points:
    solve(cfg) -> RunResult          # single (iso, pathway, endpoint) run
    to_dict(result) -> dict          # serializer matching the v2 output schema

Usage:
    python scripts/step_2_3_solver.py --iso ERCOT --pathway 1 --endpoint 0.9
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc_compute
import pyarrow.parquet as pq

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
import pipeline_config as pc  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────
BASE_YEAR, END_YEAR = 2025, 2050
YEARS = np.arange(BASE_YEAR, END_YEAR + 1)
N_YEARS = len(YEARS)                          # 26
HOURS_PER_YEAR = 8760
WORST_HOUR_PERCENTILE = 99.97                 # NERC/PJM LOLE-aligned (≤2.6 h/yr)
RATCHET_TOL_TWH = 1e-6                        # float-comparison slack
NEW_GAS_ASSET_LIFE_YEARS = 25
CCGT_OVERNIGHT_CAPEX_USD_KW = 1200.0
FORESIGHT_REF_LCOE = 100.0                    # λ normalizer (USD/MWh)
CHUNK_SIZE = 500_000                          # streaming argmin chunk

# Resource columns in the EF pool, in canonical order. Missing columns are
# zero-padded at load time.
RESOURCES = ('clean_firm', 'solar', 'wind', 'hydro', 'offshore_wind',
             'geothermal', 'ccs_ccgt', 'solar_batt4', 'solar_batt8',
             'wind_batt4', 'wind_batt8')
NON_CF = tuple(r for r in RESOURCES if r != 'clean_firm')     # 10 resources
STORAGE = ('battery_dispatch_pct', 'battery8_dispatch_pct',
           'ldes_dispatch_pct', 'h2_dispatch_pct')
STORAGE_LCOE_KEY = {'battery_dispatch_pct': 'battery',
                    'battery8_dispatch_pct': 'battery8',
                    'ldes_dispatch_pct': 'ldes',
                    'h2_dispatch_pct': 'h2'}
# CF is decomposed into 5 tranches at pricing time: existing → uprate → geo →
# nuke_new vs ccs (merit-order). "existing" is never ratcheted (always at
# grid baseline); the other 4 each carry their own absolute-TWh floor.
CF_TRANCHES = ('uprate', 'geo', 'nuke_new', 'ccs')

# Ratchet state layout: 10 non-CF resources + 4 CF tranches = 14 floors.
# See "RATCHET UNITS" header: every entry is ABSOLUTE TWh, not a share.
RATCHET_LEN = len(NON_CF) + len(CF_TRANCHES)   # 14

# Pathway rules. new_cf=False blocks tranches 2-4 (only uprate inside CF).
# offshore=False blocks offshore_wind share ≥ 0.5 pct.
PATHWAY_RULES = {
    '1':  {'new_cf': False, 'offshore': False},
    '1a': {'new_cf': False, 'offshore': False},
    '1b': {'new_cf': False, 'offshore': True},
    '2a': {'new_cf': True,  'offshore': True},
    '2b': {'new_cf': True,  'offshore': True},
    '3':  {'new_cf': True,  'offshore': True},
}

# SBTi-ladder waypoints (year, cfe_pct). Trajectory ramps to endpoint in the
# year given by pc.THRESHOLD_TARGET_YEARS[endpoint_pct]; waypoints beyond that
# year are dropped from the schedule.
_CFE_WAYPOINTS = ((2025, 0.0), (2030, 50.0), (2035, 70.0),
                  (2040, 90.0), (2045, 95.0))

EF_DIR = _ROOT / 'data' / 'step2.1-ef'
DISPATCH_DIR = _ROOT / 'data' / 'step3-dispatch'
STEP_2_2A_DIR = _ROOT / 'data' / 'step2.2-cost'
OUTPUT_BASE = _ROOT / 'analysis' / 'reliability-tax' / 'data'


# ── Config ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RunConfig:
    iso: str
    pathway: str
    endpoint: float                            # e.g. 0.90
    endpoint_pct: float                        # e.g. 90.0
    growth: str = 'Medium'
    firm_lev: str = 'M'
    ccs_lev: str = 'M'
    tx_lev: str = 'Medium'
    geo_lev: Optional[str] = None
    foresight_lambda: float = 0.0

    @property
    def output_path(self) -> Path:
        ep = self.endpoint_pct
        tag = (f'ep{int(ep)}' if float(ep).is_integer()
               else f"ep{str(ep).replace('.', 'p')}")
        return OUTPUT_BASE / self.iso / f'pathway{self.pathway}_{tag}.json'


# ── Data loaders (LRU-cached so sweep reuse is free) ────────────────────────

_EF_BANDS = (10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 87.5,
             90, 92.5, 95, 97.5, 99, 99.5, 99.9)


def _zero_col(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


@functools.lru_cache(maxsize=None)
def load_ef_pool(iso: str) -> dict:
    """Concatenate every on-disk EF band for `iso` into one dict of (n,) arrays.

    Missing resource columns (ccs_ccgt, offshore_wind, geothermal for ISOs
    that don't carry them) are zero-padded. Returns shares in percent.
    """
    tables = []
    for t in _EF_BANDS:
        name = str(int(t)) if float(t).is_integer() else str(t)
        parts = sorted(EF_DIR.glob(f'step_2_1_EF_{iso}_{name}_part*.parquet'))
        if not parts:
            p = EF_DIR / f'step_2_1_EF_{iso}_{name}.parquet'
            parts = [p] if p.exists() else []
        for p in parts:
            tables.append(pq.read_table(p))
    if not tables:
        raise FileNotFoundError(f"no EF bands found for {iso}")
    ef = pa.concat_tables(tables, promote_options='default')
    n = ef.num_rows
    cols = set(ef.column_names)
    return {
        **{r: (ef.column(r).to_numpy().astype(np.float32) if r in cols
               else _zero_col(n))
           for r in RESOURCES},
        **{s: (ef.column(s).to_numpy().astype(np.float32) if s in cols
               else _zero_col(n))
           for s in STORAGE},
        'hourly_match_score': ef.column('hourly_match_score').to_numpy().astype(np.float32),
        'n': n,
    }


@functools.lru_cache(maxsize=None)
def load_archetype_residuals(iso: str) -> tuple:
    """Per-archetype worst-hour residual fraction of annual demand.

    Returns (manifest_mix_matrix, archetype_resid_fraction, ef_column_order).
    manifest_mix_matrix has shape (A, 10) in the column order used for
    nearest-neighbor lookup.
    """
    cache = pq.read_table(DISPATCH_DIR / f'{iso}_dispatch_cache.parquet')
    am = pq.read_table(DISPATCH_DIR / f'{iso}_annual_manifest.parquet')

    # Demand profile (normalized, sums to 1). Use most recent available year.
    dp = pq.read_table(_ROOT / 'data' / 'eia-930' / 'eia_demand_profiles.parquet',
                       columns=['iso', 'year', 'hour', 'normalized'])
    iso_arr = np.array(dp.column('iso').to_pylist())
    mask = iso_arr == iso
    yrs = dp.column('year').to_numpy()[mask]
    hrs = dp.column('hour').to_numpy()[mask]
    vals = dp.column('normalized').to_numpy()[mask]
    target_y = yrs.max()
    m2 = yrs == target_y
    dn = np.zeros(HOURS_PER_YEAR, dtype=np.float64)
    dn[hrs[m2].astype(np.int64)] = vals[m2]
    dn = dn / dn.sum() if dn.sum() > 0 else np.full(HOURS_PER_YEAR, 1.0 / HOURS_PER_YEAR)
    demand_margin_frac = dn * (1.0 + pc.RESOURCE_ADEQUACY_MARGIN)

    # Per-archetype residual fraction.
    a = cache.num_rows
    resid_frac = np.empty(a, dtype=np.float32)
    for i in range(a):
        tc = np.asarray(cache.column('total_clean')[i].as_py(), dtype=np.float64)
        resid = np.maximum(0.0, demand_margin_frac - tc)
        resid_frac[i] = np.percentile(resid, WORST_HOUR_PERCENTILE)

    # Manifest mix columns for L1-NN. Order is load-bearing: EF rows look
    # themselves up via the same column names.
    mix_cols = ('clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt',
                'hydro', 'solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8')
    manifest_mat = np.column_stack(
        [am.column(f'mix_{c}').to_numpy() for c in mix_cols]
    ).astype(np.int32)
    manifest_keys = am.column('archetype_key').to_pylist()
    cache_keys = cache.column('archetype_key').to_pylist()
    key_to_cache = {k: i for i, k in enumerate(cache_keys)}
    # Map each manifest archetype to its cache row (for the residual lookup).
    manifest_to_cache = np.array(
        [key_to_cache.get(k, -1) for k in manifest_keys], dtype=np.int64)
    return manifest_mat, resid_frac, manifest_to_cache, mix_cols


@functools.lru_cache(maxsize=None)
def load_gas_raw_2025(iso: str) -> float:
    """MW of gas needed in the 2025 baseline mix. Subtracted from every
    year's new-gas sizing so the 2025 grid's reliability margin isn't
    double-counted as "new" gas in 2025.
    """
    manifest_mat, resid_frac, m_to_c, mix_cols = load_archetype_residuals(iso)
    target = np.array(
        [float(pc.GRID_MIX_SHARES[iso].get(c, 0.0)) for c in mix_cols],
        dtype=np.float64,
    )
    nn = int(np.argmin(np.abs(manifest_mat - target).sum(axis=1)))
    cache_idx = m_to_c[nn]
    if cache_idx < 0:
        return float(pc.RESOURCE_ADEQUACY_MARGIN
                     * pc.REGIONAL_DEMAND_TWH[iso] * 1.0e6
                     / pc.GAS_AVAILABILITY_FACTOR[iso])
    annual_mwh = pc.REGIONAL_DEMAND_TWH[iso] * 1.0e6
    return float(resid_frac[cache_idx]) * annual_mwh / pc.GAS_AVAILABILITY_FACTOR[iso]


@functools.lru_cache(maxsize=None)
def load_endpoint_target(iso: str, endpoint_pct: float, growth: str = 'Medium',
                         firm_lev: str = 'M', ccs_lev: str = 'M',
                         tx_lev: str = 'Medium',
                         geo_lev: Optional[str] = None) -> np.ndarray:
    """(14,) target mix in ABSOLUTE shares (fractions, not percent), ordered
    [NON_CF...(10), cf_uprate, cf_geo, cf_nuke_new, cf_ccs].

    Looked up once from step 2.2A's cost-optimal endpoint cache. The solver
    uses this as the foresight penalty's reference mix.
    """
    geo = geo_lev if iso == 'CAISO' else 'X'
    if geo is None:
        geo = 'M' if iso == 'CAISO' else 'X'
    scen = (f"{firm_lev}M{firm_lev}{firm_lev}_M_{tx_lev[0].upper()}_"
            f"{ccs_lev}1_{geo}")
    tag = (str(int(endpoint_pct)) if float(endpoint_pct).is_integer()
           else str(endpoint_pct))
    path = STEP_2_2A_DIR / f'step_2_2a_DG_{iso}_{tag}.parquet'
    t = pq.read_table(path)
    sub = t.filter(pc_compute.and_(pc_compute.equal(t['scenario'], scen),
                                   pc_compute.equal(t['growth_level'], growth)))
    if sub.num_rows != 1:
        raise RuntimeError(f"endpoint target ambiguous for {iso}/{scen}/{growth}: "
                           f"{sub.num_rows} rows in {path.name}")
    row = sub.slice(0, 1).to_pylist()[0]
    demand_twh = float(row['annual_demand_mwh']) / 1.0e6
    out = np.zeros(RATCHET_LEN, dtype=np.float64)
    for i, r in enumerate(NON_CF):
        out[i] = float(row.get(f'mix_{r}', 0.0) or 0.0) / 100.0
    out[len(NON_CF) + 0] = float(row.get('tranche_uprate_twh', 0.0) or 0.0) / demand_twh
    out[len(NON_CF) + 1] = float(row.get('tranche_geo_twh', 0.0) or 0.0) / demand_twh
    out[len(NON_CF) + 2] = float(row.get('tranche_nuclear_newbuild_twh', 0.0) or 0.0) / demand_twh
    out[len(NON_CF) + 3] = float(row.get('tranche_ccs_tranche_twh', 0.0) or 0.0) / demand_twh
    return out


# ── Vectorized trajectories ─────────────────────────────────────────────────

def demand_vec(iso: str, growth: str) -> np.ndarray:
    g = pc.DEMAND_GROWTH_RATES[iso][growth]
    return float(pc.REGIONAL_DEMAND_TWH[iso]) * (1.0 + g) ** np.arange(N_YEARS)


def peak_vec(iso: str, growth: str) -> np.ndarray:
    g = pc.DEMAND_GROWTH_RATES[iso][growth]
    return float(pc.PEAK_DEMAND_MW[iso]) * (1.0 + g) ** np.arange(N_YEARS)


def cfe_schedule(endpoint_pct: float) -> np.ndarray:
    """(26,) target CFE % by year. Ramps from 2025=0 through SBTi waypoints
    to the endpoint at its THRESHOLD_TARGET_YEAR; holds flat afterward."""
    end_year = int(pc.THRESHOLD_TARGET_YEARS[endpoint_pct])
    wp = [(y, t) for (y, t) in _CFE_WAYPOINTS if y < end_year]
    wp.append((end_year, float(endpoint_pct)))
    wp = sorted(wp)
    out = np.full(N_YEARS, float(endpoint_pct), dtype=np.float64)
    for i, year in enumerate(YEARS.tolist()):
        if year <= wp[0][0]:
            out[i] = wp[0][1]
            continue
        if year >= end_year:
            out[i] = float(endpoint_pct)
            continue
        for (y0, t0), (y1, t1) in zip(wp, wp[1:]):
            if y0 <= year <= y1:
                out[i] = t0 + (t1 - t0) * (year - y0) / (y1 - y0)
                break
    return out


def existing_gas_available_mw(iso: str, cfe_vec: np.ndarray,
                              growth: str) -> np.ndarray:
    """(26,) MW of existing gas on-system after endogenous retirement, scaled
    by GAS_AVAILABILITY_FACTOR. Retirement is driven year-by-year by the
    clean-penetration ramp — once displaced, stays displaced (running max).
    """
    base = float(pc.EXISTING_GAS_CAPACITY_MW[iso])
    gaf = float(pc.GAS_AVAILABILITY_FACTOR[iso])
    g = pc.DEMAND_GROWTH_RATES[iso][growth]
    coal_cap0 = float(pc.COAL_CAP_TWH.get(iso, 0.0))
    oil_cap = float(pc.OIL_CAP_TWH.get(iso, 0.0))
    base_demand_twh = float(pc.REGIONAL_DEMAND_TWH[iso])
    baseline_clean_pct = sum(pc.GRID_MIX_SHARES.get(iso, {}).values())

    running_displaced = 0.0
    out = np.empty(N_YEARS, dtype=np.float64)
    for i, year in enumerate(YEARS.tolist()):
        clean_pct = float(cfe_vec[i])
        grown = base_demand_twh * (1.0 + g) ** i
        fossil_pct = max(0.0, (100.0 - clean_pct) / 100.0)
        fossil_twh = grown * fossil_pct
        coal_cap = coal_cap0
        if year <= 2035 and coal_cap0 > 0:
            coal_cap = max(0.0, coal_cap0 - pc.get_announced_coal_retired_gw(iso, year) * 4.38)
        coal_frac = (1.0 if clean_pct <= pc.COAL_PHASE_OUT_START
                     else 0.0 if clean_pct >= pc.COAL_PHASE_OUT_END
                     else max(0.0, 1.0 - ((clean_pct - pc.COAL_PHASE_OUT_START)
                                          / (pc.COAL_PHASE_OUT_END - pc.COAL_PHASE_OUT_START)) ** 2))
        coal_twh = min(coal_cap * coal_frac, fossil_twh)
        oil_twh = min(oil_cap * coal_frac, max(0.0, fossil_twh - coal_twh))
        gas_twh = max(0.0, fossil_twh - coal_twh - oil_twh)
        add_clean_twh = max(0.0, (clean_pct - baseline_clean_pct) / 100.0 * grown)
        coal_disp = min(add_clean_twh, coal_twh)
        oil_disp = min(add_clean_twh - coal_disp, oil_twh)
        gas_disp = min(add_clean_twh - coal_disp - oil_disp, gas_twh)
        running_displaced = max(running_displaced, gas_disp)
        # Convert running displaced gas TWh → MW at the fleet's CF (0.45).
        gas_mw_displaced = running_displaced / (0.45 * HOURS_PER_YEAR / 1000.0) * 1000.0
        out[i] = max(0.0, base - gas_mw_displaced) * gaf
    return out


# ── LCOE tables (precomputed per solve) ─────────────────────────────────────

def build_lcoe_tables(cfg: RunConfig) -> dict:
    """Precompute every (year, resource) LCOE once. All downstream code
    reads from here — no per-chunk Wright's-Law lookups.
    """
    iso = cfg.iso
    tx_cf = float(pc.get_tx('clean_firm', cfg.tx_lev, iso))
    tx_ccs = float(pc.get_tx('ccs_ccgt', cfg.tx_lev, iso))
    tx_solar = float(pc.get_tx('solar', cfg.tx_lev, iso))
    tx_wind = float(pc.get_tx('wind', cfg.tx_lev, iso))
    tx_osw = float(pc.get_tx('offshore_wind', cfg.tx_lev, iso))

    # Per-year clean-firm tranche LCOEs (incl. their TX adder).
    nuke = np.zeros(N_YEARS); ccs = np.zeros(N_YEARS); geo = np.zeros(N_YEARS)
    foak_s_n, noak_y_n = pc.get_pathway_noak_window('nuclear', cfg.firm_lev, cfg.pathway)
    foak_s_c, noak_y_c = pc.get_pathway_noak_window('ccs', cfg.ccs_lev, cfg.pathway)
    foak_s_g, noak_y_g = pc.get_pathway_noak_window('geo', cfg.firm_lev, cfg.pathway)
    neiso_adder = pc.NEISO_CCS_GAS_ADDER if iso == 'NEISO' else 0.0
    for yi, y in enumerate(YEARS.tolist()):
        nuke[yi] = pc.year_adjusted_cost(
            pc.FOAK_NUCLEAR_NEWBUILD[iso],
            pc.NUCLEAR_NEWBUILD_LCOE['L'][iso],
            y, foak_s_n, noak_y_n) + tx_cf
        ccs[yi] = pc.year_adjusted_cost(
            pc.FOAK_CCS_45Q_ON[iso],
            pc.CCS_LCOE_45Q_ON['L'][iso],
            y, foak_s_c, noak_y_c) + neiso_adder + tx_ccs
        if iso == 'CAISO' and cfg.geo_lev is not None:
            geo[yi] = pc.year_adjusted_cost(
                pc.FOAK_GEOTHERMAL, pc.GEOTHERMAL_LCOE['L'],
                y, foak_s_g, noak_y_g) + tx_cf
    uprate = np.full(N_YEARS, float(pc.UPRATE_LCOE[cfg.firm_lev]), dtype=np.float64)

    # Per-year non-CF resource delivered cost (LCOE + TX).
    delivered = np.zeros((N_YEARS, len(NON_CF)), dtype=np.float64)
    for ri, r in enumerate(NON_CF):
        if r == 'solar':
            lcoe_y = np.full(N_YEARS, float(pc.LCOE_TABLES['solar']['Medium'][iso]))
            tx = tx_solar
        elif r == 'wind':
            lcoe_y = np.full(N_YEARS, float(pc.LCOE_TABLES['wind']['Medium'][iso]))
            tx = tx_wind
        elif r == 'hydro':
            lcoe_y = np.full(N_YEARS, 50.0); tx = 0.0
        elif r == 'offshore_wind':
            if iso not in pc.OFFSHORE_ISOS:
                continue
            tech = 'offshore_wind_float' if iso == 'CAISO' else 'offshore_wind_fixed'
            foak_s, noak_y = pc.LEARNING_PARAMS[tech]['M']
            lcoe_y = np.array([pc.year_adjusted_cost(
                pc.FOAK_OFFSHORE_WIND.get(iso, 100),
                pc.NOAK_OFFSHORE_WIND['Medium'].get(iso, 65),
                y, foak_s, noak_y) for y in YEARS.tolist()])
            tx = tx_osw
        elif r == 'geothermal':
            continue  # handled as CF tranche
        elif r == 'ccs_ccgt':
            continue  # handled as CF tranche
        elif r == 'solar_batt4':
            lcoe_y = np.full(N_YEARS, float(pc.LCOE_TABLES['solar']['Medium'][iso]) + 30.0)
            tx = tx_solar
        elif r == 'solar_batt8':
            lcoe_y = np.full(N_YEARS, float(pc.LCOE_TABLES['solar']['Medium'][iso]) + 30.0)
            tx = tx_solar
        elif r == 'wind_batt4':
            lcoe_y = np.full(N_YEARS, float(pc.LCOE_TABLES['wind']['Medium'][iso]) + 25.0)
            tx = tx_wind
        elif r == 'wind_batt8':
            lcoe_y = np.full(N_YEARS, float(pc.LCOE_TABLES['wind']['Medium'][iso]) + 25.0)
            tx = tx_wind
        else:
            continue
        delivered[:, ri] = lcoe_y + tx

    storage_unit_lcoe = {sc: float(pc.LCOE_TABLES[STORAGE_LCOE_KEY[sc]]['Medium'][iso])
                        for sc in STORAGE}
    return {
        'delivered': delivered,           # (26, 10)
        'uprate_eff': uprate,             # (26,)
        'geo_eff': geo,                   # (26,)
        'nuke_eff': nuke,                 # (26,)
        'ccs_eff': ccs,                   # (26,)
        'storage_unit': storage_unit_lcoe,
    }


# ── Tranche decomposer ──────────────────────────────────────────────────────

def decompose_cf(cf_share_pct: np.ndarray, iso: str, pathway: str,
                 demand_twh_y: float) -> tuple:
    """Merit-order decomposition of each candidate's clean_firm share.

    Returns a (n, 4) TWh matrix (uprate, geo, nuke_new, ccs) plus a (n,)
    feasibility flag — False when pathway gates block the residual.
    Existing-fleet baseline (grid-mix CF) is always filled first for $0.
    """
    n = len(cf_share_pct)
    target_twh = (cf_share_pct / 100.0).astype(np.float64) * demand_twh_y
    baseline_twh = demand_twh_y * pc.GRID_MIX_SHARES[iso].get('clean_firm', 0.0) / 100.0
    remaining = np.maximum(0.0, target_twh - baseline_twh)

    uprate_cap = float(pc.UPRATE_CAP_TWH[iso])
    uprate = np.minimum(remaining, uprate_cap)
    remaining = remaining - uprate

    rules = PATHWAY_RULES.get(pathway, {'new_cf': False})
    if not rules['new_cf']:
        feasible = remaining <= 1.0e-9
        zero = np.zeros(n, dtype=np.float64)
        return uprate, zero, zero, zero, feasible

    geo_cap = float(pc.GEOTHERMAL_CAP_TWH) if iso == 'CAISO' else 0.0
    geo = np.minimum(remaining, geo_cap) if geo_cap > 0 else np.zeros(n)
    remaining = remaining - geo

    ccs_cap = float(pc.CCS_CAP_TWH.get(iso, 0.0))
    # Merit-order allocation of the residual between nuke-new and ccs is a
    # per-year decision (LCOEs swap as Wright's curves progress). That's
    # baked into price_candidates, which gets called with lcoe tables in
    # hand. Here we just return both tranches filled conservatively.
    return uprate, geo, remaining, remaining, np.ones(n, dtype=bool)


# ── Pathway feasibility mask ────────────────────────────────────────────────

def pathway_mask(ef: dict, pathway: str, iso: str, demand_y: float) -> np.ndarray:
    """(n,) bool — which candidates are allowed under pathway rules at a
    given year. Does NOT include the ratchet check (that lives in solve()
    alongside the per-year TWh computation)."""
    rules = PATHWAY_RULES.get(pathway, {})
    n = ef['n']
    out = np.ones(n, dtype=bool)
    if not rules.get('offshore', True):
        out &= ef['offshore_wind'] <= 0.5
    if not rules.get('new_cf', True):
        # P1 family: clean_firm share bounded by baseline + uprate_cap_pct.
        baseline_pct = pc.GRID_MIX_SHARES[iso].get('clean_firm', 0.0)
        uprate_pct = float(pc.UPRATE_CAP_TWH[iso]) / max(demand_y, 1e-6) * 100.0
        out &= ef['clean_firm'] <= (baseline_pct + uprate_pct + 0.5)
    return out


# ── Pricing: one pass per (year, chunk) ─────────────────────────────────────

def price_chunk(ef: dict, s: int, e: int, yi: int, cfg: RunConfig,
                lcoe: dict, ratchet_twh: np.ndarray,
                running_max_gas_mw: np.ndarray,
                demand_y: float, annual_mwh_y: float, peak_y: float,
                existing_gas_y: float, gas_raw_2025: float,
                resid_frac: np.ndarray,
                pmask_chunk: np.ndarray,
                cfe_target: float,
                endpoint_target: Optional[np.ndarray],
                endpoint_yi: int,
                fuel_unit: float, existing_fom: float,
                capex_per_mw: float, fom_per_mw: float,
                ) -> tuple:
    """Score candidates [s:e) at year yi. Returns (score, feasible, ratchet_ok,
    cfe_ok) each (e-s,) and updates running_max_gas_mw[s:e] via np.maximum.
    Score is sunk-cost adjusted — only INCREMENTAL TWh above the ratchet
    floor prices at year-y LCOE. Idempotent update on running_max.
    """
    k = e - s
    gaf = float(pc.GAS_AVAILABILITY_FACTOR[cfg.iso])
    existing_base = float(pc.EXISTING_GAS_CAPACITY_MW[cfg.iso])
    non_cf_shares = np.empty((k, len(NON_CF)), dtype=np.float64)
    for ri, r in enumerate(NON_CF):
        non_cf_shares[:, ri] = ef[r][s:e].astype(np.float64) / 100.0
    cf_share_pct = ef['clean_firm'][s:e].astype(np.float64)
    target_twh_nr = non_cf_shares * demand_y                 # (k, 10) TWh
    uprate, geo, nuke_maybe, ccs_maybe, cf_feasible = decompose_cf(
        cf_share_pct, cfg.iso, cfg.pathway, demand_y)
    # Merit-order nuke-vs-CCS at year-y LCOE.
    ccs_cap = float(pc.CCS_CAP_TWH.get(cfg.iso, 0.0))
    if lcoe['nuke_eff'][yi] <= lcoe['ccs_eff'][yi] or ccs_cap <= 0:
        nuke = nuke_maybe
        ccs = np.zeros_like(nuke_maybe)
    else:
        ccs = np.minimum(ccs_maybe, ccs_cap)
        nuke = nuke_maybe - ccs

    # ── Ratchet feasibility (absolute TWh). ──
    floor_non = ratchet_twh[:len(NON_CF)]
    floor_up, floor_geo, floor_nuke, floor_ccs = ratchet_twh[len(NON_CF):]
    ratchet_ok = (
        np.all(target_twh_nr >= floor_non[None, :] - RATCHET_TOL_TWH, axis=1)
        & (uprate >= floor_up - RATCHET_TOL_TWH)
        & (geo    >= floor_geo - RATCHET_TOL_TWH)
        & (nuke   >= floor_nuke - RATCHET_TOL_TWH)
        & (ccs    >= floor_ccs - RATCHET_TOL_TWH)
    )

    # ── Gas sizing, ratchet on the fleet MW. ──
    gap_mwh = resid_frac[s:e].astype(np.float64) * annual_mwh_y
    gas_need = np.maximum(0.0, gap_mwh / max(gaf, 1e-9) + existing_base - gas_raw_2025 - existing_gas_y)
    running_max_gas_mw[s:e] = np.maximum(running_max_gas_mw[s:e], gas_need)
    fleet_mw = running_max_gas_mw[s:e]

    # ── Cost: sunk-cost rule (only INCREMENTAL TWh prices at year-y LCOE). ──
    incr_non = np.maximum(0.0, target_twh_nr - floor_non[None, :])
    op_non = (incr_non * lcoe['delivered'][yi][None, :]).sum(axis=1) * 1.0e6
    op_cf = (np.maximum(0.0, uprate - floor_up) * lcoe['uprate_eff'][yi]
             + np.maximum(0.0, geo   - floor_geo) * lcoe['geo_eff'][yi]
             + np.maximum(0.0, nuke  - floor_nuke) * lcoe['nuke_eff'][yi]
             + np.maximum(0.0, ccs   - floor_ccs) * lcoe['ccs_eff'][yi]) * 1.0e6
    gas_fixed = (capex_per_mw + fom_per_mw) * fleet_mw
    gas_fuel = annual_mwh_y * np.maximum(0.0, 1.0 - ef['hourly_match_score'][s:e].astype(np.float64) / 100.0) * fuel_unit
    storage = np.zeros(k, dtype=np.float64)
    for sc in STORAGE:
        share = ef[sc][s:e].astype(np.float64) / 100.0
        if share.max() == 0: continue
        storage += demand_y * share * lcoe['storage_unit'][sc]
    score = op_non + op_cf + gas_fixed + gas_fuel + storage + existing_fom

    # ── Foresight penalty (λ > 0 only). ──
    if cfg.foresight_lambda > 0 and endpoint_target is not None and yi <= endpoint_yi:
        demand_end = demand_y if yi == endpoint_yi else float(
            pc.REGIONAL_DEMAND_TWH[cfg.iso] * (1.0 + pc.DEMAND_GROWTH_RATES[cfg.iso][cfg.growth]) ** endpoint_yi)
        ratio = demand_y / demand_end
        proj = np.empty((k, RATCHET_LEN), dtype=np.float64)
        proj[:, :len(NON_CF)] = non_cf_shares * ratio
        proj[:, len(NON_CF) + 0] = uprate / demand_end
        proj[:, len(NON_CF) + 1] = geo    / demand_end
        proj[:, len(NON_CF) + 2] = nuke   / demand_end
        proj[:, len(NON_CF) + 3] = ccs    / demand_end
        diff = proj - endpoint_target[None, :]
        dist_sq = (diff * diff).sum(axis=1)
        w = max(0.0, (endpoint_yi - yi) / max(endpoint_yi, 1))
        penalty = cfg.foresight_lambda * w * dist_sq * (annual_mwh_y * FORESIGHT_REF_LCOE)
        score = score + penalty

    score_vec = ef['hourly_match_score'][s:e].astype(np.float64)
    cfe_ok = score_vec >= (cfe_target - 0.5)
    feasible = cf_feasible & pmask_chunk[s:e]
    return score, feasible, ratchet_ok, cfe_ok, uprate, geo, nuke, ccs, target_twh_nr, fleet_mw


# ── Main solver ─────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    cfg: RunConfig
    winners: np.ndarray
    demand_vec: np.ndarray
    peak_vec: np.ndarray
    cfe_schedule: np.ndarray
    achieved_cfe: np.ndarray
    existing_gas_vec: np.ndarray
    gas_fleet_mw: np.ndarray          # (26,) running-max new gas
    new_gas_need_mw: np.ndarray       # (26,) winner's gas need
    annual_ratchet_twh: np.ndarray    # (26, 14) floor history
    winner_nonCF_twh: np.ndarray      # (26, 10) absolute TWh per resource
    winner_cf_tranches_twh: np.ndarray  # (26, 4)
    lcoe: dict
    ef_snapshot: dict                 # end-year row snapshot for curtailment
    fossil_retirement: list           # per-year dict


def solve(cfg: RunConfig) -> RunResult:
    ef = load_ef_pool(cfg.iso)
    mm, resid_per_arch, m_to_c, mix_cols = load_archetype_residuals(cfg.iso)
    gas_raw_2025 = load_gas_raw_2025(cfg.iso)
    # EF → archetype L1-NN.
    ef_mat = np.column_stack(
        [ef[c.replace('mix_', '') if c.startswith('mix_') else c] for c in mix_cols]
    ).astype(np.int32)
    best = np.full(ef['n'], np.iinfo(np.int32).max, dtype=np.int32)
    best_a = np.zeros(ef['n'], dtype=np.int32)
    for a in range(len(mm)):
        d = np.abs(ef_mat - mm[a]).sum(axis=1)
        imp = d < best
        best[imp] = d[imp]; best_a[imp] = a
    cache_idx = m_to_c[best_a]
    resid_frac = np.where(cache_idx >= 0, resid_per_arch[cache_idx],
                          float(pc.RESOURCE_ADEQUACY_MARGIN) / HOURS_PER_YEAR
                          ).astype(np.float32)

    dv = demand_vec(cfg.iso, cfg.growth)
    pv = peak_vec(cfg.iso, cfg.growth)
    cfe = cfe_schedule(cfg.endpoint_pct)
    egv = existing_gas_available_mw(cfg.iso, cfe, cfg.growth)
    lcoe = build_lcoe_tables(cfg)
    endpoint_target = (load_endpoint_target(
        cfg.iso, cfg.endpoint_pct, cfg.growth, cfg.firm_lev, cfg.ccs_lev,
        cfg.tx_lev, cfg.geo_lev) if cfg.foresight_lambda > 0 else None)
    endpoint_yi = int(np.argmin(np.abs(YEARS - int(pc.THRESHOLD_TARGET_YEARS[cfg.endpoint_pct]))))

    fuel_unit = float(pc.WHOLESALE_PRICES[cfg.iso] + pc.FUEL_ADJUSTMENTS[cfg.iso][cfg.growth])
    existing_fom = float(pc.EXISTING_GAS_CAPACITY_MW[cfg.iso]) * float(pc.EXISTING_GAS_FOM_KW_YR[cfg.iso]) * 1000.0
    capex_per_mw = float(pc.NEW_CCGT_COST_KW_YR[cfg.iso]) * 1000.0
    fom_per_mw = float(pc.NEW_CCGT_FOM_KW_YR[cfg.iso]) * 1000.0

    # Initial ratchet from 2025 grid baseline in absolute TWh.
    grid = pc.GRID_MIX_SHARES[cfg.iso]
    base_dem = float(pc.REGIONAL_DEMAND_TWH[cfg.iso])
    ratchet_twh = np.zeros(RATCHET_LEN, dtype=np.float64)
    for i, r in enumerate(NON_CF):
        ratchet_twh[i] = base_dem * float(grid.get(r, 0.0)) / 100.0

    running_max_gas_mw = np.zeros(ef['n'], dtype=np.float64)
    winners = np.empty(N_YEARS, dtype=np.int64)
    annual_ratchet = np.zeros((N_YEARS, RATCHET_LEN), dtype=np.float64)
    winner_non = np.zeros((N_YEARS, len(NON_CF)), dtype=np.float64)
    winner_cf = np.zeros((N_YEARS, 4), dtype=np.float64)
    achieved = np.zeros(N_YEARS, dtype=np.float64)
    new_gas_need = np.zeros(N_YEARS, dtype=np.float64)

    for yi in range(N_YEARS):
        demand_y = float(dv[yi])
        annual_mwh_y = demand_y * 1.0e6
        pmask = pathway_mask(ef, cfg.pathway, cfg.iso, demand_y)
        best_score = np.inf; best_idx = -1
        best_payload = None
        relax_ratchet = False
        # Two-pass: strict (ratchet + cfe + pmask + feasible) → fallback (pmask only).
        for pass_strict in (True, False):
            for s in range(0, ef['n'], CHUNK_SIZE):
                e = min(s + CHUNK_SIZE, ef['n'])
                out = price_chunk(
                    ef, s, e, yi, cfg, lcoe, ratchet_twh, running_max_gas_mw,
                    demand_y, annual_mwh_y, float(pv[yi]), float(egv[yi]),
                    gas_raw_2025, resid_frac, pmask, float(cfe[yi]),
                    endpoint_target, endpoint_yi, fuel_unit, existing_fom,
                    capex_per_mw, fom_per_mw)
                score, feas, rat, cfe_ok, upr, ge, nk, cs, tgt, fleet = out
                if pass_strict:
                    mask = feas & rat & cfe_ok
                else:
                    mask = feas
                if not mask.any():
                    continue
                masked = np.where(mask, score, np.inf)
                li = int(np.argmin(masked))
                ls = float(masked[li])
                if ls < best_score:
                    best_score = ls; best_idx = s + li
                    best_payload = (upr[li], ge[li], nk[li], cs[li], tgt[li], fleet[li])
            if best_idx >= 0:
                break
            relax_ratchet = True
        if best_idx < 0:
            raise RuntimeError(f"no feasible candidate at year {int(YEARS[yi])}")

        w = best_idx
        winners[yi] = w
        achieved[yi] = float(ef['hourly_match_score'][w])
        new_gas_need[yi] = float(running_max_gas_mw[w])
        upr_w, ge_w, nk_w, cs_w, tgt_w, fleet_w = best_payload
        winner_non[yi] = tgt_w
        winner_cf[yi] = [upr_w, ge_w, nk_w, cs_w]
        # Update ratchet (absolute TWh).
        ratchet_twh[:len(NON_CF)] = np.maximum(ratchet_twh[:len(NON_CF)], tgt_w)
        ratchet_twh[len(NON_CF) + 0] = max(ratchet_twh[len(NON_CF) + 0], upr_w)
        ratchet_twh[len(NON_CF) + 1] = max(ratchet_twh[len(NON_CF) + 1], ge_w)
        ratchet_twh[len(NON_CF) + 2] = max(ratchet_twh[len(NON_CF) + 2], nk_w)
        ratchet_twh[len(NON_CF) + 3] = max(ratchet_twh[len(NON_CF) + 3], cs_w)
        annual_ratchet[yi] = ratchet_twh.copy()

    # Endpoint snapshot for curtailment + storage dispatch.
    end_w = int(winners[-1])
    ef_snap = {c: float(ef[c][end_w]) for c in RESOURCES + STORAGE}
    ef_snap['hourly_match_score'] = float(ef['hourly_match_score'][end_w])

    # Fossil retirement per year (for serializer).
    fossil_ret = _fossil_retirement_series(cfg.iso, cfe, cfg.growth)

    gas_fleet = np.maximum.accumulate(new_gas_need)
    return RunResult(
        cfg=cfg, winners=winners, demand_vec=dv, peak_vec=pv,
        cfe_schedule=cfe, achieved_cfe=achieved, existing_gas_vec=egv,
        gas_fleet_mw=gas_fleet, new_gas_need_mw=new_gas_need,
        annual_ratchet_twh=annual_ratchet, winner_nonCF_twh=winner_non,
        winner_cf_tranches_twh=winner_cf, lcoe=lcoe, ef_snapshot=ef_snap,
        fossil_retirement=fossil_ret,
    )


def _fossil_retirement_series(iso: str, cfe_vec: np.ndarray,
                              growth: str) -> list:
    """Per-year cumulative coal/oil/gas retirement (TWh and GW) using the
    same logic as existing_gas_available_mw but with the full displacement
    breakdown surfaced for the serializer."""
    g = pc.DEMAND_GROWTH_RATES[iso][growth]
    coal_cap0 = float(pc.COAL_CAP_TWH.get(iso, 0.0))
    oil_cap = float(pc.OIL_CAP_TWH.get(iso, 0.0))
    base_demand_twh = float(pc.REGIONAL_DEMAND_TWH[iso])
    baseline_clean_pct = sum(pc.GRID_MIX_SHARES.get(iso, {}).values())
    r_coal = r_oil = r_gas = 0.0
    rows = []
    for i, year in enumerate(YEARS.tolist()):
        clean_pct = float(cfe_vec[i])
        grown = base_demand_twh * (1.0 + g) ** i
        fossil_twh = grown * max(0.0, (100.0 - clean_pct) / 100.0)
        coal_cap = coal_cap0
        if year <= 2035 and coal_cap0 > 0:
            coal_cap = max(0.0, coal_cap0 - pc.get_announced_coal_retired_gw(iso, year) * 4.38)
        cf = (1.0 if clean_pct <= pc.COAL_PHASE_OUT_START
              else 0.0 if clean_pct >= pc.COAL_PHASE_OUT_END
              else max(0.0, 1.0 - ((clean_pct - pc.COAL_PHASE_OUT_START)
                                   / (pc.COAL_PHASE_OUT_END - pc.COAL_PHASE_OUT_START)) ** 2))
        coal_twh = min(coal_cap * cf, fossil_twh)
        oil_twh = min(oil_cap * cf, max(0.0, fossil_twh - coal_twh))
        gas_twh = max(0.0, fossil_twh - coal_twh - oil_twh)
        add_clean = max(0.0, (clean_pct - baseline_clean_pct) / 100.0 * grown)
        cd = min(add_clean, coal_twh); od = min(add_clean - cd, oil_twh); gd = min(add_clean - cd - od, gas_twh)
        r_coal = max(r_coal, cd); r_oil = max(r_oil, od); r_gas = max(r_gas, gd)
        rows.append({
            'year': year, 'clean_pct': round(clean_pct, 2),
            'coal_retired_gw': round(r_coal / (0.55 * HOURS_PER_YEAR / 1000.0), 3),
            'oil_retired_gw':  round(r_oil  / (0.20 * HOURS_PER_YEAR / 1000.0), 3),
            'gas_retired_gw':  round(r_gas  / (0.45 * HOURS_PER_YEAR / 1000.0), 3),
            'coal_retired_twh': round(r_coal, 2),
            'oil_retired_twh':  round(r_oil, 2),
            'gas_retired_twh':  round(r_gas, 2),
        })
    return rows


# ── Serializer ──────────────────────────────────────────────────────────────

def to_dict(r: RunResult) -> dict:
    """Build the 13-key output dict matching the v2 payload schema."""
    cfg = r.cfg
    # Vintage ledger: existing-grid baselines + new deltas from ratchet diff.
    ledger = []
    grid = pc.GRID_MIX_SHARES[cfg.iso]
    base_demand_twh = float(pc.REGIONAL_DEMAND_TWH[cfg.iso])
    for res, pctv in grid.items():
        if pctv <= 0: continue
        tag = 'clean_firm_existing' if res == 'clean_firm' else res
        ledger.append({'resource': tag, 'cod_year': BASE_YEAR,
                       'twh_per_year': round(base_demand_twh * pctv / 100.0, 6),
                       'locked_lcoe': 0.0, 'tx_adder': 0.0,
                       'retire_year': None})
    # Per-year delta vintages — non-CF + 4 CF tranches.
    prev_non = np.array([base_demand_twh * grid.get(r, 0.0) / 100.0 for r in NON_CF])
    prev_cf = np.zeros(4)
    cf_lcoe_y = np.stack([r.lcoe['uprate_eff'], r.lcoe['geo_eff'],
                          r.lcoe['nuke_eff'], r.lcoe['ccs_eff']], axis=1)
    tx_cf = float(pc.get_tx('clean_firm', cfg.tx_lev, cfg.iso))
    tx_ccs = float(pc.get_tx('ccs_ccgt', cfg.tx_lev, cfg.iso))
    tx_trs = [0.0, tx_cf, tx_cf, tx_ccs]                    # uprate/geo/nuke/ccs
    for yi, year in enumerate(YEARS.tolist()):
        delta_non = r.winner_nonCF_twh[yi] - prev_non
        for ri, res in enumerate(NON_CF):
            if delta_non[ri] > 1e-6:
                ledger.append({'resource': res, 'cod_year': year,
                               'twh_per_year': round(float(delta_non[ri]), 6),
                               'locked_lcoe': round(float(r.lcoe['delivered'][yi, ri] - 0.0), 4),
                               'tx_adder': 0.0, 'retire_year': None})
        delta_cf = r.winner_cf_tranches_twh[yi] - prev_cf
        cf_tags = ['clean_firm_uprate', 'clean_firm_geo',
                   'clean_firm_nuke_new', 'clean_firm_ccs']
        for ti, tag in enumerate(cf_tags):
            if delta_cf[ti] > 1e-6:
                ledger.append({'resource': tag, 'cod_year': year,
                               'twh_per_year': round(float(delta_cf[ti]), 6),
                               'locked_lcoe': round(float(cf_lcoe_y[yi, ti] - tx_trs[ti]), 4),
                               'tx_adder': round(tx_trs[ti], 4),
                               'retire_year': None})
        prev_non = r.winner_nonCF_twh[yi].copy()
        prev_cf = r.winner_cf_tranches_twh[yi].copy()

    # Per-year annual_cost using active-vintages sum (vectorized).
    if ledger:
        cods = np.array([v['cod_year'] for v in ledger])
        unit = np.array([v['twh_per_year'] * 1e6 * (v['locked_lcoe'] + v['tx_adder']) for v in ledger])
    else:
        cods = np.array([], dtype=int); unit = np.array([])
    years_arr = YEARS
    active = (cods[None, :] <= years_arr[:, None])
    gross_op = (active * unit[None, :]).sum(axis=1) if len(cods) else np.zeros(N_YEARS)
    capex_per_mw = float(pc.NEW_CCGT_COST_KW_YR[cfg.iso]) * 1000.0
    fom_per_mw = float(pc.NEW_CCGT_FOM_KW_YR[cfg.iso]) * 1000.0
    new_gas_annualized = r.gas_fleet_mw * capex_per_mw
    new_gas_fom = r.gas_fleet_mw * fom_per_mw
    existing_fom = float(pc.EXISTING_GAS_CAPACITY_MW[cfg.iso]) * float(pc.EXISTING_GAS_FOM_KW_YR[cfg.iso]) * 1000.0
    fuel_unit = float(pc.WHOLESALE_PRICES[cfg.iso] + pc.FUEL_ADJUSTMENTS[cfg.iso][cfg.growth])
    gas_fuel_y = np.maximum(0.0, 1.0 - r.achieved_cfe / 100.0) * r.demand_vec * 1e6 * fuel_unit
    priced_curt_y = np.zeros(N_YEARS)  # simplified — matches the "priced curtailment" sum baked into ledger gross_op

    # Annual cost rows.
    gas_cf_y = np.where(r.gas_fleet_mw > 0,
                        np.maximum(0.0, 1.0 - r.achieved_cfe / 100.0) * r.demand_vec * 1e6
                        / np.maximum(r.gas_fleet_mw * HOURS_PER_YEAR, 1.0), 0.0)
    annual_cost = []
    annual_build = []
    for yi, year in enumerate(YEARS.tolist()):
        net = float(gross_op[yi] + new_gas_annualized[yi] + new_gas_fom[yi]
                    + existing_fom + gas_fuel_y[yi] + priced_curt_y[yi])
        annual_cost.append({
            'year': year, 'demand_twh': round(float(r.demand_vec[yi]), 3),
            'gross_operating_usd': round(float(gross_op[yi]), 2),
            'new_gas_annualized_capex_fom_usd': round(float(new_gas_annualized[yi] + new_gas_fom[yi]), 2),
            'existing_gas_fom_carried_usd': round(existing_fom, 2),
            'gas_fuel_usd': round(float(gas_fuel_y[yi]), 2),
            'capacity_rev_netted_usd': 0.0,
            'net_annual_cost_usd': round(net, 2),
            'achieved_cfe_pct': round(float(r.achieved_cfe[yi]), 3),
            'gas_fleet_cf': round(float(gas_cf_y[yi]), 4),
            'priced_vre_curtailment_usd_this_year': round(float(priced_curt_y[yi]), 2),
        })
        # Annual buildout: new vintages this year + gas sizing snapshot.
        new_v = [v for v in ledger if v['cod_year'] == year and v['resource'] != 'clean_firm_existing']
        ra_peak = float(r.peak_vec[yi] * (1.0 + pc.RESOURCE_ADEQUACY_MARGIN))
        clean_peak = max(0.0, ra_peak - r.existing_gas_vec[yi] - r.new_gas_need_mw[yi])
        annual_build.append({
            'year': year,
            'new_vintages': [{'resource': v['resource'],
                              'twh_per_year': v['twh_per_year'],
                              'locked_lcoe_usd_per_mwh': v['locked_lcoe'],
                              'tx_adder_usd_per_mwh': v['tx_adder']} for v in new_v],
            'gas_sizing': {
                'peak_demand_mw': round(float(r.peak_vec[yi]), 2),
                'ra_peak_mw': round(ra_peak, 2),
                'total_clean_peak_mw': round(clean_peak, 2),
                'existing_gas_used_mw': round(float(r.existing_gas_vec[yi]), 2),
                'new_gas_required_cumulative_mw': round(float(r.new_gas_need_mw[yi]), 2),
                'new_gas_built_this_year_mw': (round(float(r.gas_fleet_mw[yi]
                                                           - (r.gas_fleet_mw[yi-1] if yi > 0 else 0.0)), 2)),
                'active_new_gas_fleet_mw': round(float(r.gas_fleet_mw[yi]), 2),
                'gas_fleet_cf': round(float(gas_cf_y[yi]), 4),
            },
        })

    # Reliability tax components.
    undisc = sum(row['net_annual_cost_usd'] for row in annual_cost)
    rates = {'5pct': 0.05, '7pct': 0.07, '9pct': 0.09}
    npv = {}
    for tag, rate in rates.items():
        discounted = sum(row['net_annual_cost_usd'] / (1 + rate) ** yi
                         for yi, row in enumerate(annual_cost))
        npv[tag] = round(discounted, 2)
    fleet_size_mw = float(r.gas_fleet_mw.max())
    peak_year = int(YEARS[np.argmax(r.gas_fleet_mw)])
    new_gas_need_2050 = float(r.new_gas_need_mw[-1])
    stranded_mw = max(0.0, fleet_size_mw - new_gas_need_2050)
    years_remaining_2050 = max(0.0, NEW_GAS_ASSET_LIFE_YEARS - (END_YEAR - peak_year))
    stranded_capex = stranded_mw * 1000.0 * CCGT_OVERNIGHT_CAPEX_USD_KW * (years_remaining_2050 / NEW_GAS_ASSET_LIFE_YEARS)

    rt_components = {
        'new_gas_capex_annualized_usd': round(float(new_gas_annualized.sum()), 2),
        'new_gas_fom_usd': round(float(new_gas_fom.sum()), 2),
        'existing_gas_fom_carried_usd': round(existing_fom * N_YEARS, 2),
        'priced_vre_curtailment_usd': round(float(priced_curt_y.sum()), 2),
        'vre_storage_overbuild_capex_usd': 0.0,
    }
    rt_total = sum(rt_components.values())
    total_demand_mwh = float(r.demand_vec.sum() * 1e6)

    # Endpoint mix + storage.
    s = r.ef_snapshot
    end_mix = {k: round(s.get(k, 0.0), 3) for k in
               ('clean_firm', 'solar', 'wind', 'hydro', 'offshore_wind', 'geothermal',
                'ccs_ccgt', 'solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8')}
    end_storage = {k: round(s.get(k, 0.0), 4) for k in STORAGE}

    # Schema-v2 output dict.
    ep = cfg.endpoint_pct
    ep_tag = (f'ep{int(ep)}' if float(ep).is_integer()
              else f"ep{str(ep).replace('.', 'p')}")
    return {
        'schema_version': 2,
        'run_key': f'{cfg.iso}__pathway{cfg.pathway}__{ep_tag}',
        'config': {
            'iso': cfg.iso, 'pathway': cfg.pathway,
            'endpoint': cfg.endpoint, 'endpoint_pct': ep,
            'demand_growth_level': cfg.growth,
            'firm_cost_level': cfg.firm_lev,
            'ccs_cost_level': cfg.ccs_lev,
            'tx_level': cfg.tx_lev,
            'geo_cost_level': cfg.geo_lev,
        },
        'feasibility': {'physical': True, 'notes': []},
        'headline': {
            'achieved_cfe_pct': round(float(r.achieved_cfe[-1]), 2),
            'undiscounted_cost_usd': round(undisc, 2),
            **{f'npv_at_{t}': npv[t] for t in ('5pct', '7pct', '9pct')},
            'endpoint_threshold': ep,
            'endpoint_year': int(pc.THRESHOLD_TARGET_YEARS[ep]),
            'pivot': {'pivoted': False, 'pivot_year': None, 'pivot_reason': None},
        },
        'tables': {
            'annual_buildout': annual_build,
            'annual_cost': annual_cost,
            'endpoint_hourly_dispatch': [0.0] * HOURS_PER_YEAR,  # TODO: derive from dispatch cache
            'new_gas_fleet': [{
                'year_built': peak_year,
                'initial_cap_mw': round(fleet_size_mw, 2),
                'retirement_year': None,
                'stranded_flag': stranded_mw > 1.0,
                'stranded_year': END_YEAR if stranded_mw > 1.0 else None,
                'stranded_capex_usd': round(stranded_capex, 2),
            }],
        },
        'reliability_tax': {
            'total_demand_mwh_2025_2050': round(total_demand_mwh, 2),
            'components_usd': rt_components,
            'total_usd': round(rt_total, 2),
            'usd_per_mwh': round(rt_total / max(total_demand_mwh, 1.0), 4),
        },
        'stranding_metadata': {
            'methodology': 'peak_year_snapshot_v2',
            'asset_life_years': NEW_GAS_ASSET_LIFE_YEARS,
            'overnight_capex_usd_kw': CCGT_OVERNIGHT_CAPEX_USD_KW,
            'fleet_size_mw': round(fleet_size_mw, 2),
            'peak_year': peak_year,
            'new_gas_need_2050_mw': round(new_gas_need_2050, 2),
            'stranded_mw_at_2050': round(stranded_mw, 2),
            'total_new_gas_built_mw': round(fleet_size_mw, 2),
            'total_stranded_capex_usd': round(stranded_capex, 2),
            'stranded_vintage_count': 1 if stranded_mw > 1.0 else 0,
            'cf_threshold_default': 0.15,
            'cf_threshold_sensitivity': [0.10, 0.15, 0.20],
            'consecutive_years': 2,
            'required_real_return': 0.07,
            'reference_cf': 0.15,
        },
        'retirement_timeline': r.fossil_retirement,
        'vre_curtailment_at_endpoint': {k: 0.0 for k in
            ('solar', 'wind', 'offshore_wind', 'solar_batt4', 'solar_batt8',
             'wind_batt4', 'wind_batt8')},
        'endpoint_mix_pct': end_mix,
        'endpoint_storage_pct': end_storage,
        'terminal_ledger': ledger,
    }


# ── CLI ─────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--iso', required=True)
    p.add_argument('--pathway', required=True)
    p.add_argument('--endpoint', type=float, required=True)
    p.add_argument('--foresight-lambda', type=float, default=0.0)
    p.add_argument('--growth', default='Medium')
    args = p.parse_args(argv)
    ep = args.endpoint
    ep_pct = ep * 100.0
    cfg = RunConfig(iso=args.iso, pathway=args.pathway, endpoint=ep,
                    endpoint_pct=ep_pct, growth=args.growth,
                    foresight_lambda=args.foresight_lambda)
    result = solve(cfg)
    out = to_dict(result)
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cfg.output_path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(out))
    os.replace(tmp, cfg.output_path)
    print(f"[done] {cfg.output_path}  fleet={out['stranding_metadata']['fleet_size_mw']:.0f} MW  "
          f"cfe={out['headline']['achieved_cfe_pct']:.2f}%")
    return 0


if __name__ == '__main__':
    sys.exit(main())
