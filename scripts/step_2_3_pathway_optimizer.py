#!/usr/bin/env python3
"""Pathway optimizer — v2 (frozen schema per reliability_tax/PATHWAY_OUTPUT_SCHEMA.md).

Two-stage architecture: a one-time ISO-level Stage-1 sidecar precomputes the
worst-hour clean MW per unique mix; the per-year Stage-2 hot loop is one
broadcast op producing the (26, n_mixes) gas-need matrix. Eliminates the
per-mix-per-year dispatch-cache call that dominated v1 wall-clock.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.compute as pacompute
import pyarrow.parquet as pq

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
import pipeline_config as pc

ISOS = list(pc.ISOS)
PATHWAYS = ('1', '1a', '2a', '2b', '3')
BASE_YEAR = 2025
END_YEAR = 2050
YEARS = list(range(BASE_YEAR, END_YEAR + 1))
N_YEARS = len(YEARS)
HOURS_PER_YEAR = 8760
DISCOUNT_RATES = {'5pct': 0.05, '7pct': 0.07, '9pct': 0.09}

# Inline (not in pipeline_config) — match v1 conventions.
CCGT_OVERNIGHT_CAPEX_USD_KW = 1200.0
NEW_GAS_ASSET_LIFE_YEARS = 25
_FOSSIL_CAPACITY_FACTORS = {'coal': 0.55, 'oil': 0.20, 'gas': 0.45}

ENDPOINT_TO_THRESHOLD = {
    0.60: 60, 0.70: 70, 0.75: 75, 0.80: 80, 0.85: 85,
    0.90: 90, 0.95: 95, 0.975: 97.5, 0.99: 99, 0.999: 99.9,
}

EF_DIR = _ROOT / 'data' / 'step2.1-ef'
DISPATCH_DIR = _ROOT / 'data' / 'step3-dispatch'
OUTPUT_BASE = _ROOT / 'analysis' / 'reliability-tax' / 'data'

_RESOURCE_COLS = (
    'clean_firm', 'solar', 'wind', 'hydro', 'offshore_wind', 'geothermal',
    'ccs_ccgt', 'solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8',
)
_STORAGE_COLS = ('battery_dispatch_pct', 'battery8_dispatch_pct',
                 'ldes_dispatch_pct', 'h2_dispatch_pct')
_FIRM_RESOURCES = ('clean_firm', 'ccs_ccgt')

# ─── Module-scope loads (once, at import) ───────────────────────────────────

with open(_ROOT / 'data' / 'egrid_emission_rates.json') as _f:
    _EGRID_EMISSION_RATES = json.load(_f)

with open(_ROOT / 'data' / 'step5-wrights' / 'wrights_law_curves.json') as _f:
    _WRIGHTS = json.load(_f)


def _load_fossil_mix_table():
    t = pq.read_table(_ROOT / 'data' / 'eia-930' / 'eia_fossil_mix.parquet')
    out = {}
    for iso in ISOS:
        out[iso] = t.filter(pacompute.equal(t['iso'], iso))
    return out


_FOSSIL_MIX_BY_ISO = _load_fossil_mix_table()


def _load_dispatch_caches():
    out, manifests = {}, {}
    for iso in ISOS:
        cache_path = DISPATCH_DIR / f'{iso}_dispatch_cache.parquet'
        manifest_path = DISPATCH_DIR / f'{iso}_annual_manifest.parquet'
        if cache_path.exists():
            out[iso] = pq.read_table(cache_path)
            manifests[iso] = pq.read_table(manifest_path)
    return out, manifests


_DISPATCH_CACHE_BY_ISO, _ANNUAL_MANIFEST_BY_ISO = _load_dispatch_caches()


def _dispatch_cache_version(iso: str) -> str:
    pf = pq.ParquetFile(DISPATCH_DIR / f'{iso}_dispatch_cache.parquet')
    md = pf.metadata.metadata or {}
    return (md.get(b'cache_version') or b'unknown').decode()


def _load_gen_profiles():
    """Per-ISO normalized 8760-h profile per resource (sums to 1)."""
    t = pq.read_table(_ROOT / 'data' / 'eia-930' / 'eia_generation_profiles.parquet')
    years = sorted(set(t.column('year').to_pylist()))
    target_year = years[-1]
    t = t.filter(pacompute.equal(t['year'], target_year))
    out: dict[str, dict[str, np.ndarray]] = {iso: {} for iso in ISOS}
    iso_arr = np.asarray(t.column('iso').to_pylist())
    fuel_arr = np.asarray(t.column('fuel').to_pylist())
    hour_arr = t.column('hour').to_numpy()
    val_arr = t.column('value').to_numpy()
    for iso in ISOS:
        for fuel in ('solar', 'wind', 'hydro', 'nuclear', 'offshore_wind', 'geothermal'):
            mask = (iso_arr == iso) & (fuel_arr == fuel)
            if not mask.any():
                continue
            prof = np.zeros(HOURS_PER_YEAR, dtype=np.float64)
            prof[hour_arr[mask]] = val_arr[mask]
            out[iso][fuel] = prof
    return out


_GEN_PROFILES_BY_ISO = _load_gen_profiles()


def _load_hybrid_profiles():
    out = {}
    for iso in ISOS:
        path = _ROOT / 'data' / 'hybrid_profiles' / f'{iso}_hybrid_profiles.npz'
        if path.exists():
            z = np.load(path)
            out[iso] = {k: z[k].astype(np.float64) for k in z.keys()}
    return out


_HYBRID_PROFILES_BY_ISO = _load_hybrid_profiles()


# ─── Ported from scripts/dispatch_utils.py lines 925–1051 (verbatim logic) ──

def _twh_to_gw(twh_per_year: float, capacity_factor: float) -> float:
    if capacity_factor <= 0:
        return 0.0
    return twh_per_year / (capacity_factor * HOURS_PER_YEAR / 1000.0)


def coal_fraction_at_clean_pct(clean_pct: float) -> float:
    if clean_pct <= pc.COAL_PHASE_OUT_START:
        return 1.0
    if clean_pct >= pc.COAL_PHASE_OUT_END:
        return 0.0
    t = ((clean_pct - pc.COAL_PHASE_OUT_START)
         / (pc.COAL_PHASE_OUT_END - pc.COAL_PHASE_OUT_START))
    return max(0.0, 1.0 - t * t)


def compute_fossil_retirement(iso, clean_pct, emission_rates, fossil_mix,
                              demand_growth_factor=1.0, year=None):
    regional_data = emission_rates.get(iso, {})
    coal_rate = regional_data.get('coal_co2_lb_per_mwh', 0.0) / 2204.62
    gas_rate = regional_data.get('gas_co2_lb_per_mwh', 0.0) / 2204.62
    oil_rate = regional_data.get('oil_co2_lb_per_mwh', 0.0) / 2204.62
    base_demand_twh = pc.REGIONAL_DEMAND_TWH.get(iso, 0)
    grown_demand_twh = base_demand_twh * demand_growth_factor
    fossil_pct = (100.0 - clean_pct) / 100.0
    fossil_twh = grown_demand_twh * fossil_pct
    coal_cap = pc.COAL_CAP_TWH.get(iso, 0)
    oil_cap = pc.OIL_CAP_TWH.get(iso, 0)
    baseline_clean = sum(pc.GRID_MIX_SHARES.get(iso, {}).values())
    if year is not None and year <= 2035 and coal_cap > 0:
        retired_gw = pc.get_announced_coal_retired_gw(iso, year)
        retired_twh = retired_gw * 4.38
        coal_cap = max(0, coal_cap - retired_twh)
    if fossil_twh <= 0.01 or clean_pct >= 100:
        total_fossil_twh = grown_demand_twh * (100.0 - baseline_clean) / 100.0
        coal_twh = min(coal_cap, total_fossil_twh)
        oil_twh = min(oil_cap, max(0, total_fossil_twh - coal_twh))
        gas_twh = max(0, total_fossil_twh - coal_twh - oil_twh)
        rate = ((coal_twh * coal_rate + oil_twh * oil_rate + gas_twh * gas_rate)
                / max(0.01, total_fossil_twh)) if total_fossil_twh > 0.01 else gas_rate
        return rate, {
            'coal_displaced_twh': round(coal_twh, 2),
            'oil_displaced_twh': round(oil_twh, 2),
            'gas_displaced_twh': round(gas_twh, 2),
            'displaced_rate_tco2_mwh': round(rate, 4),
            'remaining_rate_tco2_mwh': 0,
            'demand_growth_factor': demand_growth_factor,
        }
    cf = coal_fraction_at_clean_pct(clean_pct)
    effective_coal_cap = coal_cap * cf
    effective_oil_cap = oil_cap * cf
    coal_twh = min(effective_coal_cap, fossil_twh)
    oil_twh = min(effective_oil_cap, max(0, fossil_twh - coal_twh))
    gas_twh = max(0, fossil_twh - coal_twh - oil_twh)
    additional_clean_twh = max(
        0, (clean_pct - baseline_clean) / 100.0 * grown_demand_twh)
    coal_displaced = min(additional_clean_twh, coal_twh)
    oil_displaced = min(additional_clean_twh - coal_displaced, oil_twh)
    gas_displaced = min(
        additional_clean_twh - coal_displaced - oil_displaced, gas_twh)
    total_displaced = coal_displaced + oil_displaced + gas_displaced
    if total_displaced > 0.01:
        displaced_rate = (
            coal_displaced * coal_rate + oil_displaced * oil_rate
            + gas_displaced * gas_rate) / total_displaced
    else:
        displaced_rate = (coal_twh * coal_rate + oil_twh * oil_rate
                          + gas_twh * gas_rate) / fossil_twh
    return displaced_rate, {
        'coal_displaced_twh': round(coal_displaced, 2),
        'oil_displaced_twh': round(oil_displaced, 2),
        'gas_displaced_twh': round(gas_displaced, 2),
        'gas_remaining_twh': round(max(0, gas_twh - gas_displaced), 2),
        'displaced_rate_tco2_mwh': round(displaced_rate, 4),
        'demand_growth_factor': demand_growth_factor,
    }


# ─── Dataclasses ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Vintage:
    resource: str
    cod_year: int
    twh_per_year: float
    locked_lcoe: float
    tx_adder: float
    retire_year: Optional[int] = None


@dataclass
class VintageLedger:
    entries: list = field(default_factory=list)

    def add(self, v: Vintage) -> None:
        self.entries.append(v)

    def serialize(self) -> list[dict]:
        return [
            {
                'resource': v.resource,
                'cod_year': v.cod_year,
                'twh_per_year': round(v.twh_per_year, 6),
                'locked_lcoe': round(v.locked_lcoe, 4),
                'tx_adder': round(v.tx_adder, 4),
                'retire_year': v.retire_year,
            }
            for v in self.entries
        ]


@dataclass(frozen=True)
class RunConfig:
    iso: str
    pathway: str
    endpoint: float
    endpoint_pct: float
    demand_growth_level: str = 'Medium'
    firm_cost_level: str = 'M'
    ccs_cost_level: str = 'M'
    tx_level: str = 'Medium'
    geo_cost_level: Optional[str] = None

    @property
    def output_path(self) -> Path:
        ep_int = int(round(self.endpoint_pct))
        return OUTPUT_BASE / self.iso / f'pathway{self.pathway}_ep{ep_int}.json'


# ─── Stage-1 precompute: per-mix worst-hour clean MW (sidecar) ──────────────

def _build_clean_profile_table(iso: str) -> dict[str, np.ndarray]:
    """Return per-resource normalized 8760-h profile; resource keys match EF cols."""
    gen = _GEN_PROFILES_BY_ISO.get(iso, {})
    hyb = _HYBRID_PROFILES_BY_ISO.get(iso, {})
    flat = np.full(HOURS_PER_YEAR, 1.0 / HOURS_PER_YEAR, dtype=np.float64)
    return {
        'clean_firm':    gen.get('nuclear', flat),
        'solar':         gen.get('solar', np.zeros(HOURS_PER_YEAR)),
        'wind':          gen.get('wind', np.zeros(HOURS_PER_YEAR)),
        'hydro':         gen.get('hydro', flat),
        'offshore_wind': gen.get('offshore_wind', np.zeros(HOURS_PER_YEAR)),
        'geothermal':    gen.get('geothermal', flat),
        'ccs_ccgt':      flat,
        'solar_batt4':   hyb.get('solar_batt4', np.zeros(HOURS_PER_YEAR)),
        'solar_batt8':   hyb.get('solar_batt8', np.zeros(HOURS_PER_YEAR)),
        'wind_batt4':    hyb.get('wind_batt4', np.zeros(HOURS_PER_YEAR)),
        'wind_batt8':    hyb.get('wind_batt8', np.zeros(HOURS_PER_YEAR)),
    }


def _read_ef_table(iso: str, threshold) -> pa.Table:
    name = str(int(threshold)) if float(threshold).is_integer() else str(threshold)
    parts = sorted(EF_DIR.glob(f'step_2_1_EF_{iso}_{name}_part*.parquet'))
    if parts:
        return pa.concat_tables([pq.read_table(p) for p in parts])
    return pq.read_table(EF_DIR / f'step_2_1_EF_{iso}_{name}.parquet')


def _peakclean_path(iso: str, threshold) -> Path:
    name = str(int(threshold)) if float(threshold).is_integer() else str(threshold)
    return EF_DIR / f'step_2_1_EF_{iso}_{name}_peakclean.parquet'


def precompute_clean_peak_hour_mw(iso: str, threshold) -> pa.Table:
    """Stage-1: per-EF-row worst-hour clean MW (vectorized in 5K-row batches)."""
    ef = _read_ef_table(iso, threshold)
    profiles = _build_clean_profile_table(iso)
    P = np.stack([profiles[r] for r in _RESOURCE_COLS], axis=0)   # (R, 8760)
    n = ef.num_rows
    M = np.column_stack([
        ef.column(c).to_numpy() if c in ef.column_names else np.zeros(n)
        for c in _RESOURCE_COLS
    ]).astype(np.float64)
    base_demand_mwh = pc.REGIONAL_DEMAND_TWH[iso] * 1.0e6
    out = np.empty(n, dtype=np.float32)
    batch = 5000
    for i in range(0, n, batch):
        chunk = (M[i:i + batch] / 100.0) * base_demand_mwh        # (b, R)
        out[i:i + batch] = (chunk @ P).min(axis=1).astype(np.float32)
    norm = out / max(1e-9, float(pc.REGIONAL_DEMAND_TWH[iso]))
    tbl = pa.table({
        'clean_peak_hour_mw': pa.array(out),
        'clean_pct_mw_per_twh_demand': pa.array(norm.astype(np.float32)),
    })
    return tbl.replace_schema_metadata({
        'source_cache_version': _dispatch_cache_version(iso),
        'iso': iso, 'threshold': str(threshold),
    })


def _load_or_build_peakclean(iso: str, threshold) -> pa.Table:
    sp = _peakclean_path(iso, threshold)
    target_ver = _dispatch_cache_version(iso)
    if sp.exists():
        meta = pq.read_schema(sp).metadata or {}
        if meta.get(b'source_cache_version', b'').decode() == target_ver:
            return pq.read_table(sp)
    print(f"[stage1] building peakclean sidecar for {iso} threshold {threshold}", flush=True)
    tbl = precompute_clean_peak_hour_mw(iso, threshold)
    tmp = sp.with_suffix('.parquet.tmp')
    pq.write_table(tbl, tmp)
    os.replace(tmp, sp)
    return tbl


def load_ef_mixes(iso: str, threshold) -> dict[str, np.ndarray]:
    """Load EF + peakclean for (iso, threshold). Returns dict of (n_mixes,) arrays."""
    ef = _read_ef_table(iso, threshold)
    pc_tbl = _load_or_build_peakclean(iso, threshold)
    n = ef.num_rows
    arrs = {c: (ef.column(c).to_numpy() if c in ef.column_names else np.zeros(n))
            for c in _RESOURCE_COLS + _STORAGE_COLS + ('hourly_match_score',)}
    arrs['clean_peak_hour_mw'] = pc_tbl.column('clean_peak_hour_mw').to_numpy()
    return arrs


# ─── Vectorized year helpers ────────────────────────────────────────────────

def peak_demand_vec(iso: str, growth_level: str) -> np.ndarray:
    g = pc.DEMAND_GROWTH_RATES[iso][growth_level]
    yrs = np.arange(N_YEARS)
    return float(pc.PEAK_DEMAND_MW[iso]) * (1.0 + g) ** yrs


def demand_twh_vec(iso: str, growth_level: str) -> np.ndarray:
    g = pc.DEMAND_GROWTH_RATES[iso][growth_level]
    yrs = np.arange(N_YEARS)
    return float(pc.REGIONAL_DEMAND_TWH[iso]) * (1.0 + g) ** yrs


def existing_gas_available_vec(iso: str, clean_pct_vec: np.ndarray,
                               growth_level: str) -> np.ndarray:
    """Per-year MW of existing gas still on-system after endogenous retirement."""
    fmix = _FOSSIL_MIX_BY_ISO[iso]
    base_existing = float(pc.EXISTING_GAS_CAPACITY_MW[iso])
    g = pc.DEMAND_GROWTH_RATES[iso][growth_level]
    out = np.empty(N_YEARS, dtype=np.float64)
    cumulative_gas_displaced_twh = 0.0
    for i, year in enumerate(YEARS):
        gf = (1.0 + g) ** i
        _, info = compute_fossil_retirement(
            iso, float(clean_pct_vec[i]), _EGRID_EMISSION_RATES, fmix, gf, year)
        cumulative_gas_displaced_twh = max(
            cumulative_gas_displaced_twh, float(info.get('gas_displaced_twh', 0.0)))
        gas_displaced_mw = _twh_to_gw(cumulative_gas_displaced_twh,
                                      _FOSSIL_CAPACITY_FACTORS['gas']) * 1000.0
        out[i] = max(0.0, base_existing - gas_displaced_mw) * pc.GAS_AVAILABILITY_FACTOR[iso]
    return out


def gas_sizing_matrix(peak_vec: np.ndarray, existing_gas_vec: np.ndarray,
                      clean_arr: np.ndarray, ra_margin: float) -> np.ndarray:
    """Per-year-per-mix new gas requirement (MW). One numpy op."""
    return np.maximum(0.0,
                      peak_vec[:, None] * (1.0 + ra_margin)
                      - clean_arr[None, :]
                      - existing_gas_vec[:, None])


def _resource_lcoe_year(iso: str, resource: str, year: int,
                        cfg: 'RunConfig') -> float:
    """Year-adjusted $/MWh for a resource. Wright's curve for clean_firm/ccs/osw."""
    if resource == 'clean_firm':
        foak_s, noak_y = pc.get_pathway_noak_window('nuclear', 'M', cfg.pathway)
        foak = pc.FOAK_NUCLEAR_NEWBUILD[iso]
        noak = pc.LCOE_TABLES.get('clean_firm', {}).get('Medium', {}).get(iso, foak * 0.55)
        return pc.year_adjusted_cost(foak, noak, year, foak_s, noak_y)
    if resource == 'ccs_ccgt':
        foak_s, noak_y = pc.get_pathway_noak_window('ccs', 'M', cfg.pathway)
        foak = pc.FOAK_CCS_45Q_ON[iso]
        noak = pc.LCOE_TABLES.get('ccs_ccgt', {}).get('Medium', {}).get(iso, foak * 0.55)
        return pc.year_adjusted_cost(foak, noak, year, foak_s, noak_y)
    if resource == 'offshore_wind':
        if iso not in pc.OFFSHORE_ISOS:
            return 0.0
        tech = 'offshore_wind_float' if iso == 'CAISO' else 'offshore_wind_fixed'
        foak_s, noak_y = pc.LEARNING_PARAMS[tech]['M']
        foak = pc.FOAK_OFFSHORE_WIND.get(iso, 100)
        noak = pc.NOAK_OFFSHORE_WIND['Medium'].get(iso, 65)
        return pc.year_adjusted_cost(foak, noak, year, foak_s, noak_y)
    if resource == 'geothermal':
        return pc.FOAK_GEOTHERMAL if iso == 'CAISO' else 0.0
    if resource in ('solar', 'wind'):
        return pc.LCOE_TABLES[resource]['Medium'][iso]
    if resource in ('solar_batt4', 'solar_batt8'):
        return pc.LCOE_TABLES['solar']['Medium'][iso] + 30.0
    if resource in ('wind_batt4', 'wind_batt8'):
        return pc.LCOE_TABLES['wind']['Medium'][iso] + 25.0
    if resource == 'hydro':
        return 50.0
    return 0.0


def _resource_tx(iso: str, resource: str, tx_level: str) -> float:
    if resource in ('solar', 'wind', 'clean_firm', 'ccs_ccgt', 'offshore_wind'):
        return float(pc.get_tx(resource, tx_level, iso))
    if resource in ('solar_batt4', 'solar_batt8'):
        return float(pc.get_tx('solar', tx_level, iso))
    if resource in ('wind_batt4', 'wind_batt8'):
        return float(pc.get_tx('wind', tx_level, iso))
    return 0.0


def operating_cost_matrix(ef: dict, demand_vec: np.ndarray, cfg: 'RunConfig') -> np.ndarray:
    """Per-year-per-mix operating cost (USD). Resource share × LCOE_at_y × demand."""
    n = len(ef['hourly_match_score'])
    out = np.zeros((N_YEARS, n), dtype=np.float64)
    for r in _RESOURCE_COLS:
        share = ef[r] / 100.0          # (n,)
        if share.sum() == 0:
            continue
        for yi, year in enumerate(YEARS):
            unit = _resource_lcoe_year(cfg.iso, r, year, cfg) + _resource_tx(cfg.iso, r, cfg.tx_level)
            if unit == 0:
                continue
            out[yi] += demand_vec[yi] * 1.0e6 * share * unit
    # Add storage costs (annualized capacity × demand)
    storage_lcoe_keys = {'battery_dispatch_pct': 'battery', 'battery8_dispatch_pct': 'battery8',
                         'ldes_dispatch_pct': 'ldes', 'h2_dispatch_pct': 'h2'}
    for sc, lk in storage_lcoe_keys.items():
        share = ef[sc] / 100.0
        if share.sum() == 0:
            continue
        unit = pc.LCOE_TABLES[lk]['Medium'][cfg.iso]
        out += demand_vec[:, None] * 1.0e6 * share[None, :] * unit / 1.0e6
    return out


def gas_fuel_cost_matrix(ef: dict, demand_vec: np.ndarray, fuel_price: float) -> np.ndarray:
    """Per-year-per-mix gas-fuel cost: (1 - cfe/100) × demand × $/MWh."""
    score = ef['hourly_match_score']                         # (n,)
    gas_frac = np.maximum(0.0, 1.0 - score / 100.0)          # (n,)
    return demand_vec[:, None] * 1.0e6 * gas_frac[None, :] * fuel_price


# ─── Pathway constraint mask ─────────────────────────────────────────────────

def _pathway_firm_floor_pct(iso: str) -> float:
    return float(pc.GRID_MIX_SHARES[iso].get('clean_firm', 0.0))


def _pathway_mask(ef: dict, cfg: 'RunConfig') -> np.ndarray:
    """(N_YEARS, n_mixes) bool mask: True = mix m is eligible at year y."""
    n = len(ef['hourly_match_score'])
    cf = ef['clean_firm']
    ccs = ef['ccs_ccgt']
    floor = _pathway_firm_floor_pct(cfg.iso) + 0.01
    no_clean_firm = cf <= floor
    no_ccs = ccs <= 0.5
    mask = np.zeros((N_YEARS, n), dtype=bool)
    if cfg.pathway in ('1', '1a'):
        eligible = no_clean_firm & no_ccs
        mask[:] = eligible
    else:
        if cfg.pathway == '3':
            unlock_year = pc.NOAK_YEAR_BY_PATHWAY['3'] - 5
        elif cfg.pathway == '2b':
            unlock_year = pc.NOAK_YEAR_BY_PATHWAY['2b'] - 5
        else:
            unlock_year = pc.NOAK_YEAR_BY_PATHWAY['2a'] - 5
        unlock_idx = max(0, unlock_year - BASE_YEAR)
        for yi in range(N_YEARS):
            if yi < unlock_idx:
                mask[yi] = no_clean_firm & no_ccs
            else:
                mask[yi] = np.ones(n, dtype=bool)
    return mask


def _cfe_target_for_year(year: int, endpoint_pct: float) -> float:
    targets = ((2025, 0.0), (2030, 50.0), (2035, 70.0), (2040, 90.0),
               (2045, 95.0), (2050, endpoint_pct))
    if year <= targets[0][0]:
        return 0.0
    if year >= targets[-1][0]:
        return endpoint_pct
    for (y0, t0), (y1, t1) in zip(targets, targets[1:]):
        if y0 <= year <= y1:
            frac = (year - y0) / (y1 - y0)
            return min(endpoint_pct, t0 + (t1 - t0) * frac)
    return endpoint_pct


# ─── Phase A: per-year argmin solver ─────────────────────────────────────────

@dataclass
class PathwayRunResult:
    config: 'RunConfig'
    ledger: VintageLedger
    winners: np.ndarray
    new_gas_required_cumulative_mw: np.ndarray
    active_new_gas_fleet_mw: np.ndarray
    gas_fleet_cf: np.ndarray
    achieved_cfe_pct: np.ndarray
    demand_vec: np.ndarray
    peak_vec: np.ndarray
    existing_gas_used_mw: np.ndarray
    annual_cost_rows: list
    endpoint_hourly_dispatch: np.ndarray
    feasibility: dict
    stranding_metadata: dict
    retirement_timeline: list
    vre_curtailment_at_endpoint: dict
    endpoint_mix_pct: dict
    endpoint_storage_pct: dict
    reliability_tax: dict
    headline: dict
    ef: dict
    clean_pct_vec: np.ndarray
    priced_vre_curtailment_vec: np.ndarray


def solve_pathway(cfg: 'RunConfig') -> PathwayRunResult:
    threshold = ENDPOINT_TO_THRESHOLD[round(cfg.endpoint, 4)]
    ef = load_ef_mixes(cfg.iso, threshold)
    n_mixes = len(ef['hourly_match_score'])

    peak_vec = peak_demand_vec(cfg.iso, cfg.demand_growth_level)
    demand_vec = demand_twh_vec(cfg.iso, cfg.demand_growth_level)
    score_vec = ef['hourly_match_score']
    clean_arr = ef['clean_peak_hour_mw']

    # Two-pass: first pass picks winners assuming clean_pct = endpoint each year
    # to seed existing-gas retirement; second pass uses winners' actual scores.
    cfe_targets = np.array([_cfe_target_for_year(y, cfg.endpoint_pct) for y in YEARS])
    pmask = _pathway_mask(ef, cfg)
    cfe_mask = score_vec[None, :] >= cfe_targets[:, None] - 0.5

    # Seed pass: clean_pct ≈ cfe_target, build existing_gas_vec
    existing_gas_vec = existing_gas_available_vec(cfg.iso, cfe_targets, cfg.demand_growth_level)
    gas_required = gas_sizing_matrix(peak_vec, existing_gas_vec, clean_arr,
                                     pc.RESOURCE_ADEQUACY_MARGIN)
    op_cost = operating_cost_matrix(ef, demand_vec, cfg)
    fuel_unit = pc.WHOLESALE_PRICES[cfg.iso] + pc.FUEL_ADJUSTMENTS[cfg.iso]['Medium']
    gas_fuel = gas_fuel_cost_matrix(ef, demand_vec, fuel_unit)

    # Argmin needs the ratchet: active_new_gas_fleet[y] = max over y' <= y.
    # Approximate by using gas_required directly in cost; ratchet applied in Phase B.
    new_gas_capex_y = (pc.NEW_CCGT_COST_KW_YR[cfg.iso] * 1000.0) * gas_required
    existing_fom = (float(pc.EXISTING_GAS_CAPACITY_MW[cfg.iso])
                    * pc.EXISTING_GAS_FOM_KW_YR[cfg.iso] * 1000.0)
    cost_matrix = op_cost + new_gas_capex_y + gas_fuel + existing_fom
    valid = pmask & cfe_mask
    cost_matrix = np.where(valid, cost_matrix, np.inf)

    winners = np.empty(N_YEARS, dtype=np.int64)
    for yi in range(N_YEARS):
        if not np.isfinite(cost_matrix[yi]).any():
            # No feasible mix this year; relax cfe constraint
            relaxed = pmask[yi] & (score_vec >= cfe_targets[yi] - 5.0)
            row = np.where(relaxed, op_cost[yi] + new_gas_capex_y[yi] + gas_fuel[yi]
                           + existing_fom, np.inf)
            winners[yi] = int(np.argmin(row))
        else:
            winners[yi] = int(np.argmin(cost_matrix[yi]))

    # Phase B: realized clean_pct from winners → re-run existing_gas_vec, apply ratchet
    achieved_cfe = score_vec[winners]
    existing_gas_vec_real = existing_gas_available_vec(cfg.iso, achieved_cfe, cfg.demand_growth_level)
    gas_required_real = gas_sizing_matrix(peak_vec, existing_gas_vec_real, clean_arr,
                                          pc.RESOURCE_ADEQUACY_MARGIN)
    cum_new_gas = gas_required_real[np.arange(N_YEARS), winners]
    active_fleet = np.maximum.accumulate(cum_new_gas)
    fleet_size_mw = float(active_fleet.max())
    peak_year = int(YEARS[int(np.argmax(active_fleet))])

    # gas fleet CF (energy-balance)
    total_gas_mw = float(pc.EXISTING_GAS_CAPACITY_MW[cfg.iso]) + active_fleet
    gas_gen_mwh = np.maximum(0.0, 1.0 - achieved_cfe / 100.0) * demand_vec * 1.0e6
    gas_cf = np.where(total_gas_mw > 1e-6,
                      np.minimum(1.0, gas_gen_mwh / (total_gas_mw * HOURS_PER_YEAR)), 0.0)

    # Existing gas used = min(peak_demand × (1+RA) - clean, existing_gas)
    existing_gas_used = np.minimum(
        existing_gas_vec_real,
        np.maximum(0.0, peak_vec * (1 + pc.RESOURCE_ADEQUACY_MARGIN) - clean_arr[winners]))
    return _finalize_run(cfg, ef, winners, peak_vec, demand_vec,
                         achieved_cfe, existing_gas_vec_real, existing_gas_used,
                         cum_new_gas, active_fleet, gas_cf, fleet_size_mw, peak_year,
                         clean_arr)


# ─── Output-side helpers ─────────────────────────────────────────────────────

def _retirement_timeline(iso: str, clean_pct_vec: np.ndarray,
                         growth_level: str) -> list[dict]:
    g = pc.DEMAND_GROWTH_RATES[iso][growth_level]
    fmix = _FOSSIL_MIX_BY_ISO[iso]
    rows = []
    cum = {'coal': 0.0, 'oil': 0.0, 'gas': 0.0}
    for i, year in enumerate(YEARS):
        gf = (1.0 + g) ** i
        _, info = compute_fossil_retirement(
            iso, float(clean_pct_vec[i]), _EGRID_EMISSION_RATES, fmix, gf, year)
        for fuel in ('coal', 'oil', 'gas'):
            cum[fuel] = max(cum[fuel], float(info.get(f'{fuel}_displaced_twh', 0.0)))
        rows.append({
            'year': year,
            'clean_pct': round(float(clean_pct_vec[i]), 3),
            'coal_retired_gw': round(_twh_to_gw(cum['coal'], _FOSSIL_CAPACITY_FACTORS['coal']), 3),
            'oil_retired_gw':  round(_twh_to_gw(cum['oil'],  _FOSSIL_CAPACITY_FACTORS['oil']),  3),
            'gas_retired_gw':  round(_twh_to_gw(cum['gas'],  _FOSSIL_CAPACITY_FACTORS['gas']),  3),
            'coal_retired_twh': round(cum['coal'], 2),
            'oil_retired_twh':  round(cum['oil'],  2),
            'gas_retired_twh':  round(cum['gas'],  2),
        })
    return rows


def _archetype_for_endpoint_mix(iso: str, mix_pct: dict) -> Optional[str]:
    """Find closest cached archetype to mix_pct (L1 over mix_* cols)."""
    am = _ANNUAL_MANIFEST_BY_ISO.get(iso)
    if am is None:
        return None
    cols = ['mix_clean_firm', 'mix_solar', 'mix_wind', 'mix_offshore_wind', 'mix_ccs_ccgt',
            'mix_hydro', 'mix_solar_batt4', 'mix_solar_batt8', 'mix_wind_batt4', 'mix_wind_batt8']
    target = np.array([mix_pct.get(c.replace('mix_', ''), 0.0) for c in cols])
    M = np.column_stack([am.column(c).to_numpy() for c in cols])
    dists = np.abs(M - target).sum(axis=1)
    best = int(np.argmin(dists))
    return am.column('archetype_key')[best].as_py()


def _vre_curtailment_at_endpoint(iso: str, mix_pct: dict) -> dict:
    keys = ('solar', 'wind', 'offshore_wind', 'solar_batt4', 'solar_batt8',
            'wind_batt4', 'wind_batt8')
    out = {k: 0.0 for k in keys}
    arch = _archetype_for_endpoint_mix(iso, mix_pct)
    cache = _DISPATCH_CACHE_BY_ISO.get(iso)
    if cache is None or arch is None:
        return out
    arch_arr = cache.column('archetype_key').to_pylist()
    if arch not in arch_arr:
        return out
    ridx = arch_arr.index(arch)
    for r in keys:
        if mix_pct.get(r, 0.0) <= 0:
            continue
        m_col = f'matched_{r}'
        s_col = f'surplus_{r}'
        if m_col not in cache.column_names:
            continue
        matched = float(np.asarray(cache.column(m_col)[ridx].as_py()).sum())
        surplus = float(np.asarray(cache.column(s_col)[ridx].as_py()).sum())
        denom = matched + surplus
        out[r] = round(surplus / denom, 4) if denom > 0 else 0.0
    return out


def _endpoint_hourly_dispatch(iso: str, mix_pct: dict) -> np.ndarray:
    arch = _archetype_for_endpoint_mix(iso, mix_pct)
    cache = _DISPATCH_CACHE_BY_ISO.get(iso)
    if cache is None or arch is None:
        return np.zeros(HOURS_PER_YEAR)
    arch_arr = cache.column('archetype_key').to_pylist()
    if arch not in arch_arr:
        return np.zeros(HOURS_PER_YEAR)
    ridx = arch_arr.index(arch)
    return np.asarray(cache.column('total_clean')[ridx].as_py(), dtype=np.float64)


def _priced_vre_curtailment_vec(ef: dict, winners: np.ndarray, demand_vec: np.ndarray,
                                cfg: 'RunConfig') -> np.ndarray:
    """Per SPEC §24.7: priced curtailment per year using endpoint surplus fractions."""
    end_mix = {r: float(ef[r][winners[-1]]) for r in _RESOURCE_COLS}
    curtail_frac = _vre_curtailment_at_endpoint(cfg.iso, end_mix)
    out = np.zeros(N_YEARS, dtype=np.float64)
    for r in ('solar', 'wind', 'offshore_wind', 'solar_batt4', 'solar_batt8',
              'wind_batt4', 'wind_batt8'):
        frac = curtail_frac.get(r, 0.0)
        if frac <= 0:
            continue
        unit = _resource_lcoe_year(cfg.iso, r, END_YEAR, cfg)
        share_y = ef[r][winners] / 100.0
        out += demand_vec * 1.0e6 * share_y * frac * unit
    return out


def _build_ledger(ef: dict, winners: np.ndarray, cfg: 'RunConfig') -> VintageLedger:
    """Existing-fleet entries + per-year delta vintages with locked LCOE."""
    led = VintageLedger()
    base_demand = float(pc.REGIONAL_DEMAND_TWH[cfg.iso])
    grid = pc.GRID_MIX_SHARES[cfg.iso]
    for r, share in grid.items():
        if share <= 0:
            continue
        key = 'clean_firm_existing' if r == 'clean_firm' else r
        led.add(Vintage(resource=key, cod_year=BASE_YEAR,
                        twh_per_year=base_demand * share / 100.0,
                        locked_lcoe=0.0, tx_adder=0.0, retire_year=None))
    prev_share = {r: 0.0 for r in _RESOURCE_COLS}
    for yi, year in enumerate(YEARS):
        w = int(winners[yi])
        gf = (1.0 + pc.DEMAND_GROWTH_RATES[cfg.iso][cfg.demand_growth_level]) ** yi
        demand_y_twh = base_demand * gf
        for r in _RESOURCE_COLS:
            cur = float(ef[r][w])
            baseline = grid.get(r, 0.0)
            delta = cur - max(prev_share[r], baseline)
            if delta > 0.5:
                lcoe = _resource_lcoe_year(cfg.iso, r, year, cfg)
                tx = _resource_tx(cfg.iso, r, cfg.tx_level)
                led.add(Vintage(resource=r, cod_year=year,
                                twh_per_year=demand_y_twh * delta / 100.0,
                                locked_lcoe=lcoe, tx_adder=tx, retire_year=None))
            prev_share[r] = max(prev_share[r], cur)
    return led


def _finalize_run(cfg: 'RunConfig', ef: dict, winners: np.ndarray,
                  peak_vec, demand_vec, achieved_cfe, existing_gas_vec_real,
                  existing_gas_used, cum_new_gas, active_fleet, gas_cf,
                  fleet_size_mw, peak_year, clean_arr) -> PathwayRunResult:
    ledger = _build_ledger(ef, winners, cfg)
    priced_curt_vec = _priced_vre_curtailment_vec(ef, winners, demand_vec, cfg)
    new_gas_annualized = active_fleet * pc.NEW_CCGT_COST_KW_YR[cfg.iso] * 1000.0
    existing_fom = float(pc.EXISTING_GAS_CAPACITY_MW[cfg.iso]) * pc.EXISTING_GAS_FOM_KW_YR[cfg.iso] * 1000.0
    fuel_unit = pc.WHOLESALE_PRICES[cfg.iso] + pc.FUEL_ADJUSTMENTS[cfg.iso]['Medium']
    gas_fuel_y = np.maximum(0.0, 1.0 - achieved_cfe / 100.0) * demand_vec * 1.0e6 * fuel_unit
    # gross operating cost: sum of vintage LCOE × twh, ledger sub-set active at year y
    annual_rows = []
    for yi, year in enumerate(YEARS):
        active_v = [v for v in ledger.entries if v.cod_year <= year and (v.retire_year is None or v.retire_year > year)]
        gross_op = sum(v.twh_per_year * 1.0e6 * (v.locked_lcoe + v.tx_adder) for v in active_v) \
                   * (demand_vec[yi] / max(1e-6, demand_vec[0]))  # scale by demand growth
        net = gross_op + new_gas_annualized[yi] + existing_fom + gas_fuel_y[yi] + priced_curt_vec[yi]
        annual_rows.append({
            'year': year, 'demand_twh': round(float(demand_vec[yi]), 3),
            'gross_operating_usd': round(gross_op, 2),
            'new_gas_annualized_capex_fom_usd': round(float(new_gas_annualized[yi]), 2),
            'existing_gas_fom_carried_usd': round(existing_fom, 2),
            'gas_fuel_usd': round(float(gas_fuel_y[yi]), 2),
            'capacity_rev_netted_usd': 0.0,
            'net_annual_cost_usd': round(net, 2),
            'achieved_cfe_pct': round(float(achieved_cfe[yi]), 3),
            'gas_fleet_cf': round(float(gas_cf[yi]), 4),
            'priced_vre_curtailment_usd_this_year': round(float(priced_curt_vec[yi]), 2),
        })

    new_gas_2050 = float(cum_new_gas[-1])
    stranded_mw = max(0.0, fleet_size_mw - new_gas_2050)
    years_remaining_2050 = max(0.0, NEW_GAS_ASSET_LIFE_YEARS - (END_YEAR - peak_year))
    stranded_capex = stranded_mw * 1000.0 * CCGT_OVERNIGHT_CAPEX_USD_KW * (years_remaining_2050 / NEW_GAS_ASSET_LIFE_YEARS)

    end_w = int(winners[-1])
    end_mix_pct = {r: round(float(ef[r][end_w]), 3) for r in
                   ('clean_firm', 'solar', 'wind', 'hydro', 'offshore_wind', 'geothermal',
                    'ccs_ccgt', 'solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8')}
    end_storage_pct = {sc: round(float(ef[sc][end_w]), 4) for sc in _STORAGE_COLS}
    vre_curt = _vre_curtailment_at_endpoint(cfg.iso, end_mix_pct)
    end_dispatch = _endpoint_hourly_dispatch(cfg.iso, end_mix_pct)

    rt_components = {
        'new_gas_capex_annualized_usd': round(float(new_gas_annualized.sum()), 2),
        'new_gas_fom_usd': 0.0,
        'existing_gas_fom_carried_usd': round(existing_fom * N_YEARS, 2),
        'priced_vre_curtailment_usd': round(float(priced_curt_vec.sum()), 2),
        'vre_storage_overbuild_capex_usd': 0.0,
    }
    total_demand_mwh = float((demand_vec * 1.0e6).sum())
    rt_total = sum(rt_components.values())
    reliability_tax = {
        'total_demand_mwh_2025_2050': round(total_demand_mwh, 2),
        'components_usd': rt_components,
        'total_usd': round(rt_total, 2),
        'usd_per_mwh': round(rt_total / max(1.0, total_demand_mwh), 4),
    }

    undisc = float(sum(r['net_annual_cost_usd'] for r in annual_rows))
    npvs = {f'npv_at_{k}': round(sum(r['net_annual_cost_usd'] / (1 + rate) ** (r['year'] - BASE_YEAR)
                                     for r in annual_rows), 2)
            for k, rate in DISCOUNT_RATES.items()}
    headline = {
        'achieved_cfe_pct': round(float(achieved_cfe[-1]), 3),
        'undiscounted_cost_usd': round(undisc, 2),
        **npvs,
        'endpoint_threshold': cfg.endpoint_pct,
        'pivot': {'pivoted': False, 'pivot_year': None, 'pivot_reason': None},
    }

    stranding_metadata = {
        'methodology': 'peak_year_snapshot_v2',
        'asset_life_years': NEW_GAS_ASSET_LIFE_YEARS,
        'overnight_capex_usd_kw': CCGT_OVERNIGHT_CAPEX_USD_KW,
        'fleet_size_mw': round(fleet_size_mw, 2),
        'peak_year': peak_year,
        'new_gas_need_2050_mw': round(new_gas_2050, 2),
        'stranded_mw_at_2050': round(stranded_mw, 2),
        'total_new_gas_built_mw': round(fleet_size_mw, 2),
        'total_stranded_capex_usd': round(stranded_capex, 2),
        'stranded_vintage_count': 1 if stranded_mw > 0 else 0,
        'cf_threshold_default': 0.15,
        'cf_threshold_sensitivity': [0.10, 0.15, 0.20],
        'consecutive_years': 2,
        'required_real_return': 0.07,
        'reference_cf': 0.15,
    }

    rt_timeline = _retirement_timeline(cfg.iso, achieved_cfe, cfg.demand_growth_level)
    feasibility = {'physical': True, 'notes': []}

    return PathwayRunResult(
        config=cfg, ledger=ledger, winners=winners,
        new_gas_required_cumulative_mw=cum_new_gas,
        active_new_gas_fleet_mw=active_fleet, gas_fleet_cf=gas_cf,
        achieved_cfe_pct=achieved_cfe, demand_vec=demand_vec, peak_vec=peak_vec,
        existing_gas_used_mw=existing_gas_used,
        annual_cost_rows=annual_rows, endpoint_hourly_dispatch=end_dispatch,
        feasibility=feasibility, stranding_metadata=stranding_metadata,
        retirement_timeline=rt_timeline, vre_curtailment_at_endpoint=vre_curt,
        endpoint_mix_pct=end_mix_pct, endpoint_storage_pct=end_storage_pct,
        reliability_tax=reliability_tax, headline=headline, ef=ef,
        clean_pct_vec=achieved_cfe, priced_vre_curtailment_vec=priced_curt_vec,
    )


# ─── Serialization to 13-key JSON ────────────────────────────────────────────

def serialize_run_result(r: PathwayRunResult) -> dict:
    cfg = r.config
    ra = pc.RESOURCE_ADEQUACY_MARGIN
    annual_buildout = []
    prev_share = {res: 0.0 for res in _RESOURCE_COLS}
    grid = pc.GRID_MIX_SHARES[cfg.iso]
    base_demand = float(pc.REGIONAL_DEMAND_TWH[cfg.iso])
    for yi, year in enumerate(YEARS):
        w = int(r.winners[yi])
        new_v = []
        for res in _RESOURCE_COLS:
            cur = float(r.ef[res][w])
            baseline = grid.get(res, 0.0)
            delta = cur - max(prev_share[res], baseline)
            if delta > 0.5:
                new_v.append({
                    'resource': res,
                    'twh_per_year': round(r.demand_vec[yi] * delta / 100.0, 4),
                    'locked_lcoe_usd_per_mwh': round(_resource_lcoe_year(cfg.iso, res, year, cfg), 2),
                    'tx_adder_usd_per_mwh': round(_resource_tx(cfg.iso, res, cfg.tx_level), 2),
                })
            prev_share[res] = max(prev_share[res], cur)
        is_peak = (year == r.stranding_metadata['peak_year'])
        annual_buildout.append({
            'year': year, 'new_vintages': new_v,
            'gas_sizing': {
                'peak_demand_mw': round(float(r.peak_vec[yi]), 2),
                'ra_peak_mw': round(float(r.peak_vec[yi] * (1 + ra)), 2),
                'total_clean_peak_mw': round(float(r.ef['clean_peak_hour_mw'][w]), 2),
                'existing_gas_used_mw': round(float(r.existing_gas_used_mw[yi]), 2),
                'new_gas_required_cumulative_mw': round(float(r.new_gas_required_cumulative_mw[yi]), 2),
                'new_gas_built_this_year_mw': round(r.stranding_metadata['fleet_size_mw'], 2) if is_peak else 0.0,
                'active_new_gas_fleet_mw': round(float(r.active_new_gas_fleet_mw[yi]), 2),
                'gas_fleet_cf': round(float(r.gas_fleet_cf[yi]), 4),
            },
        })

    fleet = [{
        'year_built': r.stranding_metadata['peak_year'],
        'initial_cap_mw': r.stranding_metadata['fleet_size_mw'],
        'retirement_year': None,
        'stranded_flag': r.stranding_metadata['stranded_mw_at_2050'] > 0,
        'stranded_year': END_YEAR if r.stranding_metadata['stranded_mw_at_2050'] > 0 else None,
        'stranded_capex_usd': r.stranding_metadata['total_stranded_capex_usd'],
    }]

    return {
        'schema_version': 2,
        'run_key': f'{cfg.iso}__pathway{cfg.pathway}__ep{int(round(cfg.endpoint_pct))}',
        'config': {
            'iso': cfg.iso, 'pathway': cfg.pathway,
            'endpoint': cfg.endpoint, 'endpoint_pct': cfg.endpoint_pct,
            'demand_growth_level': cfg.demand_growth_level,
            'firm_cost_level': cfg.firm_cost_level,
            'ccs_cost_level': cfg.ccs_cost_level,
            'tx_level': cfg.tx_level,
            'geo_cost_level': cfg.geo_cost_level,
        },
        'feasibility': r.feasibility,
        'headline': r.headline,
        'tables': {
            'annual_buildout': annual_buildout,
            'annual_cost': r.annual_cost_rows,
            'endpoint_hourly_dispatch': [round(float(x), 8) for x in r.endpoint_hourly_dispatch],
            'new_gas_fleet': fleet,
        },
        'reliability_tax': r.reliability_tax,
        'stranding_metadata': r.stranding_metadata,
        'retirement_timeline': r.retirement_timeline,
        'vre_curtailment_at_endpoint': r.vre_curtailment_at_endpoint,
        'endpoint_mix_pct': r.endpoint_mix_pct,
        'endpoint_storage_pct': r.endpoint_storage_pct,
        'terminal_ledger': r.ledger.serialize(),
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description='Pathway optimizer v2')
    ap.add_argument('--iso', choices=ISOS)
    ap.add_argument('--pathway', choices=PATHWAYS)
    ap.add_argument('--endpoint', type=float, choices=list(ENDPOINT_TO_THRESHOLD.keys()))
    ap.add_argument('--all', action='store_true', help='Run full 7×5×10 sweep')
    return ap


def _run_one(iso: str, pathway: str, endpoint: float) -> Path:
    ep_pct = endpoint * 100
    cfg = RunConfig(iso=iso, pathway=pathway, endpoint=endpoint, endpoint_pct=ep_pct)
    result = solve_pathway(cfg)
    payload = serialize_run_result(result)
    out = cfg.output_path
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, out)
    print(f"[done] {out}  fleet={result.stranding_metadata['fleet_size_mw']:.0f} MW  "
          f"cfe={result.headline['achieved_cfe_pct']:.2f}%", flush=True)
    return out


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    if args.all:
        for iso in ISOS:
            for p in PATHWAYS:
                for ep in ENDPOINT_TO_THRESHOLD.keys():
                    _run_one(iso, p, ep)
    else:
        if not (args.iso and args.pathway and args.endpoint):
            print('Provide --iso, --pathway, --endpoint OR --all', file=sys.stderr)
            return 2
        _run_one(args.iso, args.pathway, args.endpoint)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
