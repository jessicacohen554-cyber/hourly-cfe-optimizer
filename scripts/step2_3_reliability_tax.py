#!/usr/bin/env python3
"""Pathway optimizer v3 — refactored for performance and clarity.

Two-stage architecture: Stage-1 precomputes per-mix worst-hour clean MW
via numba-accelerated L1-NN; Stage-2 streams chunked year-by-year argmins
over multi-million-row candidate pools without materializing (Y, N) tensors.
"""
from __future__ import annotations

import argparse
import functools
import gc
import json
import multiprocessing
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numba
import numpy as np
import pyarrow as pa
import pyarrow.compute as pacompute
import pyarrow.parquet as pq

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
import pipeline_config as pc

ISOS = list(pc.ISOS)
_ISO_FILTER = os.environ.get('ISO_FILTER', '').strip().upper()
if _ISO_FILTER:
    if _ISO_FILTER not in ISOS:
        raise SystemExit(f"ISO_FILTER={_ISO_FILTER!r} not in known ISOs {ISOS}")
    ISOS = [_ISO_FILTER]

PATHWAYS = ('1', '1a', '2a', '2b', '3')
BASE_YEAR, END_YEAR = 2025, 2050
YEARS = list(range(BASE_YEAR, END_YEAR + 1))
N_YEARS = len(YEARS)
HOURS_PER_YEAR = 8760
DISCOUNT_RATES = {'5pct': 0.05, '7pct': 0.07, '9pct': 0.09}
WORST_HOUR_PERCENTILE = 99.97
CCGT_OVERNIGHT_CAPEX_USD_KW = 1200.0
NEW_GAS_ASSET_LIFE_YEARS = 25
_FOSSIL_CFS = {'coal': 0.55, 'oil': 0.20, 'gas': 0.45}
FORESIGHT_REF_LCOE = 100.0  # $/MWh normalizer for foresight penalty
ENDPOINT_TO_THRESHOLD = {0.90: 90, 0.95: 95, 0.99: 99, 0.999: 99.9}
_EP_TAG_MAP = {90: 'ep90', 95: 'ep95', 99: 'ep99', 99.9: 'ep99p9'}
RATCHET_TOL = 1.0e-6  # TWh tolerance for ratchet checks
SCORE_FLOOR_PCT = 10.0  # defensive hourly_match_score filter band
_POOL_CHUNK = 500_000  # streaming chunk size for candidate argmin

EF_DIR = _ROOT / 'data' / 'step2.1-ef'
DISPATCH_DIR = _ROOT / 'data' / 'step3-dispatch'
OUTPUT_BASE = _ROOT / 'analysis' / 'reliability-tax' / 'data'
_STEP22A_DIR = _ROOT / 'data' / 'step2.2-cost'

_RESOURCE_COLS = (
    'clean_firm', 'solar', 'wind', 'hydro', 'offshore_wind', 'geothermal',
    'ccs_ccgt', 'solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8',
)
_STORAGE_COLS = (
    'battery_dispatch_pct', 'battery8_dispatch_pct',
    'ldes_dispatch_pct', 'h2_dispatch_pct',
)
_NON_CF = tuple(r for r in _RESOURCE_COLS if r != 'clean_firm')
_N_NON_CF = len(_NON_CF)
_FORESIGHT_SHARE_KEYS = _RESOURCE_COLS + ('cf_uprate', 'cf_geo', 'cf_nuke_new', 'cf_ccs')
_CF_TRANCHE_VKEY = {
    'existing': 'clean_firm_existing', 'uprate': 'clean_firm_uprate',
    'geo': 'clean_firm_geo', 'nuke_new': 'clean_firm_nuke_new',
    'ccs': 'clean_firm_ccs',
}
_TRANCHES_AVAILABLE = {
    '1': False, '1a': False, '1b': False, '2a': True, '2b': True, '3': True,
}
_EF_BAND_THRESHOLDS = (
    10, 20, 30, 40, 50, 55, 60, 65, 70, 75,
    80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 99.5, 99.9,
)
_STAGE1_VERSION = '3'

# ─── Module-scope loads (once at import) ─────────────────────────────────────

with open(_ROOT / 'data' / 'egrid_emission_rates.json') as _f:
    _EGRID = json.load(_f)
with open(_ROOT / 'data' / 'step5-wrights' / 'wrights_law_curves.json') as _f:
    _WRIGHTS = json.load(_f)


def _load_fossil_mix():
    t = pq.read_table(_ROOT / 'data' / 'eia-930' / 'eia_fossil_mix.parquet')
    return {iso: t.filter(pacompute.equal(t['iso'], iso)) for iso in ISOS}


_FOSSIL_MIX = _load_fossil_mix()


def _load_dispatch_caches():
    caches, manifests = {}, {}
    for iso in ISOS:
        cp = DISPATCH_DIR / f'{iso}_dispatch_cache.parquet'
        mp = DISPATCH_DIR / f'{iso}_annual_manifest.parquet'
        if cp.exists():
            caches[iso] = pq.read_table(cp)
            manifests[iso] = pq.read_table(mp)
    return caches, manifests


_DCACHE, _MANIFEST = _load_dispatch_caches()


def _cache_version(iso: str) -> str:
    pf = pq.ParquetFile(DISPATCH_DIR / f'{iso}_dispatch_cache.parquet')
    md = pf.metadata.metadata or {}
    return (md.get(b'cache_version') or b'unknown').decode()


def _load_gen_profiles():
    t = pq.read_table(_ROOT / 'data' / 'eia-930' / 'eia_generation_profiles.parquet')
    target_year = pacompute.max(t.column('year')).as_py()
    t = t.filter(pacompute.equal(t['year'], target_year))
    iso_arr = np.asarray(t.column('iso').to_pylist())
    fuel_arr = np.asarray(t.column('fuel').to_pylist())
    hour_arr, val_arr = t.column('hour').to_numpy(), t.column('value').to_numpy()
    out: dict[str, dict[str, np.ndarray]] = {}
    for iso in ISOS:
        out[iso] = {}
        im = iso_arr == iso
        for fuel in ('solar', 'wind', 'hydro', 'nuclear', 'offshore_wind', 'geothermal'):
            mask = im & (fuel_arr == fuel)
            if not mask.any():
                continue
            prof = np.zeros(HOURS_PER_YEAR, dtype=np.float64)
            prof[hour_arr[mask]] = val_arr[mask]
            out[iso][fuel] = prof
    return out


_GEN_PROF = _load_gen_profiles()


def _load_hybrid_profiles():
    out = {}
    for iso in ISOS:
        p = _ROOT / 'data' / 'hybrid_profiles' / f'{iso}_hybrid_profiles.npz'
        if p.exists():
            z = np.load(p)
            out[iso] = {k: z[k].astype(np.float64) for k in z.keys()}
    return out


_HYB_PROF = _load_hybrid_profiles()


def _load_demand_profiles():
    path = _ROOT / 'data' / 'eia-930' / 'eia_demand_profiles.parquet'
    flat = np.full(HOURS_PER_YEAR, 1.0 / HOURS_PER_YEAR)
    out: dict[str, np.ndarray] = {}
    if not path.exists():
        for iso in ISOS:
            out[iso] = flat * float(pc.REGIONAL_DEMAND_TWH[iso]) * 1e6
        return out
    t = pq.read_table(path, columns=['iso', 'year', 'hour', 'normalized'])
    iso_arr = np.asarray(t.column('iso').to_pylist())
    year_arr, hour_arr = t.column('year').to_numpy(), t.column('hour').to_numpy()
    norm_arr = t.column('normalized').to_numpy()
    for iso in ISOS:
        mwh = float(pc.REGIONAL_DEMAND_TWH[iso]) * 1e6
        mask = iso_arr == iso
        if not mask.any():
            out[iso] = flat * mwh
            continue
        m2 = mask & (year_arr == int(year_arr[mask].max()))
        norm = np.zeros(HOURS_PER_YEAR, dtype=np.float64)
        norm[hour_arr[m2].astype(np.int64)] = norm_arr[m2]
        norm = norm / norm.sum() if norm.sum() > 0 else flat.copy()
        out[iso] = norm * mwh
    return out


_DEMAND = _load_demand_profiles()


def _compute_gas_raw_2025():
    out: dict[str, float] = {}
    mcols = ['mix_clean_firm', 'mix_solar', 'mix_wind', 'mix_offshore_wind',
             'mix_ccs_ccgt', 'mix_hydro']
    for iso in ISOS:
        cache, am = _DCACHE.get(iso), _MANIFEST.get(iso)
        mwh = float(pc.REGIONAL_DEMAND_TWH[iso]) * 1e6
        gaf = float(pc.GAS_AVAILABILITY_FACTOR[iso])
        if cache is None or am is None:
            out[iso] = float(pc.RESOURCE_ADEQUACY_MARGIN) * mwh / gaf
            continue
        shares = pc.GRID_MIX_SHARES.get(iso, {})
        target = np.array([float(shares.get(c[4:], 0.0)) for c in mcols], dtype=np.float64)
        am_mat = np.column_stack([am.column(c).to_numpy() for c in mcols]).astype(np.float64)
        nn = int(np.argmin(np.abs(am_mat - target).sum(axis=1)))
        arch = am.column('archetype_key')[nn].as_py()
        k2i = {k: i for i, k in enumerate(cache.column('archetype_key').to_pylist())}
        ri = k2i.get(arch, -1)
        if ri < 0:
            out[iso] = float(pc.RESOURCE_ADEQUACY_MARGIN) * mwh / gaf
            continue
        tc = np.asarray(cache.column('total_clean')[ri].as_py(), dtype=np.float64)
        dfrac = _DEMAND[iso] / max(1.0, mwh)
        resid = np.maximum(0.0, dfrac * (1.0 + pc.RESOURCE_ADEQUACY_MARGIN) - tc)
        out[iso] = float(np.percentile(resid, 99.97)) * mwh / gaf
    return out


_GAS_RAW_2025 = _compute_gas_raw_2025()

# ─── Fossil retirement ──────────────────────────────────────────────────────

def _twh_to_gw(twh: float, cf: float) -> float:
    return twh / (cf * HOURS_PER_YEAR / 1000.0) if cf > 0 else 0.0


def coal_fraction_at_clean_pct(pct: float) -> float:
    if pct <= pc.COAL_PHASE_OUT_START:
        return 1.0
    if pct >= pc.COAL_PHASE_OUT_END:
        return 0.0
    t = (pct - pc.COAL_PHASE_OUT_START) / (pc.COAL_PHASE_OUT_END - pc.COAL_PHASE_OUT_START)
    return max(0.0, 1.0 - t * t)


def compute_fossil_retirement(iso, clean_pct, emission_rates, fossil_mix,
                              demand_growth_factor=1.0, year=None):
    rd = emission_rates.get(iso, {})
    rates = {f: rd.get(f'{f}_co2_lb_per_mwh', 0.0) / 2204.62 for f in ('coal', 'gas', 'oil')}
    grown_twh = pc.REGIONAL_DEMAND_TWH.get(iso, 0) * demand_growth_factor
    fossil_twh = grown_twh * (100.0 - clean_pct) / 100.0
    coal_cap, oil_cap = pc.COAL_CAP_TWH.get(iso, 0), pc.OIL_CAP_TWH.get(iso, 0)
    baseline_clean = sum(pc.GRID_MIX_SHARES.get(iso, {}).values())
    if year is not None and year <= 2035 and coal_cap > 0:
        coal_cap = max(0, coal_cap - pc.get_announced_coal_retired_gw(iso, year) * 4.38)
    if fossil_twh <= 0.01 or clean_pct >= 100:
        total_f = grown_twh * (100.0 - baseline_clean) / 100.0
        c = min(coal_cap, total_f)
        o = min(oil_cap, max(0, total_f - c))
        g = max(0, total_f - c - o)
        r = ((c * rates['coal'] + o * rates['oil'] + g * rates['gas'])
             / max(0.01, total_f)) if total_f > 0.01 else rates['gas']
        return r, {'coal_displaced_twh': round(c, 2), 'oil_displaced_twh': round(o, 2),
                   'gas_displaced_twh': round(g, 2), 'displaced_rate_tco2_mwh': round(r, 4),
                   'remaining_rate_tco2_mwh': 0, 'demand_growth_factor': demand_growth_factor}
    cf = coal_fraction_at_clean_pct(clean_pct)
    ct = min(coal_cap * cf, fossil_twh)
    ot = min(oil_cap * cf, max(0, fossil_twh - ct))
    gt = max(0, fossil_twh - ct - ot)
    add_clean = max(0, (clean_pct - baseline_clean) / 100.0 * grown_twh)
    cd = min(add_clean, ct)
    od = min(add_clean - cd, ot)
    gd = min(add_clean - cd - od, gt)
    td = cd + od + gd
    dr = ((cd * rates['coal'] + od * rates['oil'] + gd * rates['gas']) / td if td > 0.01
          else (ct * rates['coal'] + ot * rates['oil'] + gt * rates['gas']) / fossil_twh)
    return dr, {'coal_displaced_twh': round(cd, 2), 'oil_displaced_twh': round(od, 2),
                'gas_displaced_twh': round(gd, 2), 'gas_remaining_twh': round(max(0, gt - gd), 2),
                'displaced_rate_tco2_mwh': round(dr, 4), 'demand_growth_factor': demand_growth_factor}


# ─── Dataclasses ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Vintage:
    resource: str; cod_year: int; twh_per_year: float
    locked_lcoe: float; tx_adder: float; retire_year: Optional[int] = None


@dataclass
class VintageLedger:
    entries: list = field(default_factory=list)

    def add(self, v: Vintage):
        self.entries.append(v)

    def serialize(self) -> list[dict]:
        return [{'resource': v.resource, 'cod_year': v.cod_year,
                 'twh_per_year': round(v.twh_per_year, 6),
                 'locked_lcoe': round(v.locked_lcoe, 4),
                 'tx_adder': round(v.tx_adder, 4),
                 'retire_year': v.retire_year} for v in self.entries]


def _ep_tag(pct: float) -> str:
    return _EP_TAG_MAP.get(pct, f'ep{int(round(pct))}')


def _threshold_tag(pct: float) -> str:
    return str(int(pct)) if float(pct).is_integer() else str(pct)


def _endpoint_year(pct: float) -> int:
    return int(pc.THRESHOLD_TARGET_YEARS[pct])


@dataclass(frozen=True)
class RunConfig:
    iso: str; pathway: str; endpoint: float; endpoint_pct: float
    demand_growth_level: str = 'Medium'
    firm_cost_level: str = 'M'; ccs_cost_level: str = 'M'
    tx_level: str = 'Medium'; geo_cost_level: Optional[str] = None
    solver_mode: str = 'myopic'
    foresight_lambda: Optional[float] = None
    endpoint_year_val: Optional[int] = None
    endpoint_target_shares: Optional[tuple] = None

    @property
    def output_path(self) -> Path:
        return OUTPUT_BASE / self.iso / f'pathway{self.pathway}_{_ep_tag(self.endpoint_pct)}.json'


# ─── Numba-accelerated L1 nearest-neighbor ──────────────────────────────────

@numba.njit(parallel=True, cache=True)
def _l1_nn(ef_mat, manifest_mat):
    """For each EF row, find nearest manifest archetype by L1 distance.
    Parallelized over EF rows (millions); ~63 archetypes is the inner loop."""
    n, n_arch, D = ef_mat.shape[0], manifest_mat.shape[0], ef_mat.shape[1]
    best = np.zeros(n, dtype=numba.int32)
    for i in numba.prange(n):
        bd = numba.int32(2147483647)
        for a in range(n_arch):
            d = numba.int32(0)
            for j in range(D):
                d += abs(ef_mat[i, j] - manifest_mat[a, j])
            if d < bd:
                bd = d
                best[i] = numba.int32(a)
    return best


# ─── Stage-1: precompute per-mix worst-hour residual ────────────────────────

def _build_clean_profile_table(iso: str) -> dict[str, np.ndarray]:
    gen, hyb = _GEN_PROF.get(iso, {}), _HYB_PROF.get(iso, {})
    flat = np.full(HOURS_PER_YEAR, 1.0 / HOURS_PER_YEAR, dtype=np.float64)
    z = np.zeros(HOURS_PER_YEAR, dtype=np.float64)
    return {
        'clean_firm': gen.get('nuclear', flat), 'solar': gen.get('solar', z),
        'wind': gen.get('wind', z), 'hydro': gen.get('hydro', flat),
        'offshore_wind': gen.get('offshore_wind', z),
        'geothermal': gen.get('geothermal', flat), 'ccs_ccgt': flat,
        'solar_batt4': hyb.get('solar_batt4', z), 'solar_batt8': hyb.get('solar_batt8', z),
        'wind_batt4': hyb.get('wind_batt4', z), 'wind_batt8': hyb.get('wind_batt8', z),
    }


def _read_ef_table(iso: str, threshold) -> pa.Table:
    name = _threshold_tag(threshold)
    parts = sorted(EF_DIR.glob(f'step_2_1_EF_{iso}_{name}_part*.parquet'))
    if parts:
        return pa.concat_tables([pq.read_table(p) for p in parts])
    return pq.read_table(EF_DIR / f'step_2_1_EF_{iso}_{name}.parquet')


def _peakclean_path(iso: str, threshold) -> Path:
    return EF_DIR / f'step_2_1_EF_{iso}_{_threshold_tag(threshold)}_peakclean.parquet'


def _per_archetype_resid(iso: str, cache: pa.Table) -> np.ndarray:
    """Vectorized 99.97th-pctile margin-on-demand residual per archetype.
    Uses pyarrow list_flatten + reshape instead of per-row Python loop."""
    dmarg = (_DEMAND[iso] / max(1.0, float(pc.REGIONAL_DEMAND_TWH[iso]) * 1e6)
             ) * (1.0 + pc.RESOURCE_ADEQUACY_MARGIN)
    tc = pacompute.list_flatten(cache.column('total_clean')).to_numpy(
        ).astype(np.float64).reshape(cache.num_rows, HOURS_PER_YEAR)
    return np.percentile(np.maximum(0.0, dmarg[None, :] - tc),
                         WORST_HOUR_PERCENTILE, axis=1)


def precompute_clean_peak_hour_mw(iso: str, threshold) -> pa.Table:
    ef = _read_ef_table(iso, threshold)
    n = ef.num_rows
    resid = np.empty(n, dtype=np.float32)
    cache, am = _DCACHE.get(iso), _MANIFEST.get(iso)
    miss_val = np.float32(float(pc.RESOURCE_ADEQUACY_MARGIN) / HOURS_PER_YEAR)

    if cache is None or am is None:
        resid[:] = miss_val
        print(f"[stage1] {iso} t={threshold}: no cache, fallback {n} rows", flush=True)
    else:
        resid_arch = _per_archetype_resid(iso, cache).astype(np.float32)
        k2i = {k: i for i, k in enumerate(cache.column('archetype_key').to_pylist())}
        mcols = ['mix_clean_firm', 'mix_solar', 'mix_wind', 'mix_offshore_wind',
                 'mix_ccs_ccgt', 'mix_hydro', 'mix_solar_batt4', 'mix_solar_batt8',
                 'mix_wind_batt4', 'mix_wind_batt8']
        mmat = np.column_stack([am.column(c).to_numpy() for c in mcols]).astype(np.int32)
        m2c = np.array([k2i.get(k, -1) for k in am.column('archetype_key').to_pylist()],
                       dtype=np.int64)
        ef_cols = [c[4:] for c in mcols]  # strip 'mix_'
        emat = np.column_stack([
            ef.column(c).to_numpy() if c in ef.column_names else np.zeros(n)
            for c in ef_cols]).astype(np.int32)

        if n > 0 and len(mmat) > 0:
            cidx = m2c[_l1_nn(emat, mmat)]
        else:
            cidx = np.empty(n, dtype=np.int64)
        hit = cidx >= 0
        resid[:] = np.where(hit, resid_arch[np.where(hit, cidx, 0)], miss_val)
        hits = int(hit.sum())
        print(f"[stage1] {iso} t={threshold}: match {hits}/{n} "
              f"= {hits / max(1, n) * 100:.1f}%", flush=True)

    if 'hourly_match_score' in ef.column_names:
        resid[ef.column('hourly_match_score').to_numpy() < (float(threshold) - SCORE_FLOOR_PCT)] = np.float32(np.inf)

    peak = float(pc.PEAK_DEMAND_MW[iso])
    safe = np.where(np.isfinite(resid), resid, 0.0).astype(np.float64)
    return pa.table({
        'resid_norm_p9997': pa.array(resid),
        'clean_peak_hour_mw': pa.array(((1.0 - safe) * peak).astype(np.float32)),
    }).replace_schema_metadata({
        'source_cache_version': _cache_version(iso),
        'stage1_version': _STAGE1_VERSION, 'iso': iso, 'threshold': str(threshold),
    })


def _load_or_build_peakclean(iso: str, threshold) -> pa.Table:
    sp = _peakclean_path(iso, threshold)
    tgt = _cache_version(iso)
    if sp.exists():
        meta = pq.read_schema(sp).metadata or {}
        if (meta.get(b'source_cache_version', b'').decode() == tgt
                and meta.get(b'stage1_version', b'').decode() == _STAGE1_VERSION):
            return pq.read_table(sp)
    print(f"[stage1] building peakclean for {iso} t={threshold}", flush=True)
    tbl = precompute_clean_peak_hour_mw(iso, threshold)
    tmp = sp.with_suffix('.parquet.tmp')
    pq.write_table(tbl, tmp)
    os.replace(tmp, sp)
    return tbl

# ─── Pool loader ─────────────────────────────────────────────────────────────

def _iter_ef_bands(iso: str):
    for t in _EF_BAND_THRESHOLDS:
        name = _threshold_tag(t)
        parts = sorted(EF_DIR.glob(f'step_2_1_EF_{iso}_{name}_part*.parquet'))
        if parts:
            yield t, parts
        else:
            main = EF_DIR / f'step_2_1_EF_{iso}_{name}.parquet'
            if main.exists():
                yield t, [main]


def load_ef_pool(iso: str) -> dict[str, np.ndarray]:
    """Full-pool EF loader: concatenate all threshold bands + peakclean sidecars."""
    ef_tables, pc_tables = [], []
    for t, paths in _iter_ef_bands(iso):
        for p in paths:
            ef_tables.append(pq.read_table(p))
        pc_tables.append(_load_or_build_peakclean(iso, t))
    if not ef_tables:
        raise FileNotFoundError(f"No EF bands for {iso} under {EF_DIR}")
    ef = pa.concat_tables(ef_tables, promote_options='default')
    pctbl = pa.concat_tables(pc_tables, promote_options='default')
    n = ef.num_rows
    if pctbl.num_rows != n:
        raise RuntimeError(f"Peakclean rows ({pctbl.num_rows}) != EF rows ({n}) for {iso}")
    arrs = {c: (ef.column(c).to_numpy() if c in ef.column_names else np.zeros(n))
            for c in _RESOURCE_COLS + _STORAGE_COLS + ('hourly_match_score',)}
    arrs['clean_peak_hour_mw'] = pctbl.column('clean_peak_hour_mw').to_numpy()
    arrs['resid_norm_p9997'] = (pctbl.column('resid_norm_p9997').to_numpy()
                                if 'resid_norm_p9997' in pctbl.column_names
                                else np.zeros(n, dtype=np.float32))
    return arrs


# ─── Year helpers ────────────────────────────────────────────────────────────

def _gf(iso: str, level: str) -> np.ndarray:
    """Growth factor vector (N_YEARS,)."""
    return (1.0 + pc.DEMAND_GROWTH_RATES[iso][level]) ** np.arange(N_YEARS)


def peak_demand_vec(iso, level):
    return float(pc.PEAK_DEMAND_MW[iso]) * _gf(iso, level)


def demand_twh_vec(iso, level):
    return float(pc.REGIONAL_DEMAND_TWH[iso]) * _gf(iso, level)


def existing_gas_vec(iso, clean_pct_vec, level):
    base = float(pc.EXISTING_GAS_CAPACITY_MW[iso])
    g = pc.DEMAND_GROWTH_RATES[iso][level]
    out = np.empty(N_YEARS, dtype=np.float64)
    cum = 0.0
    for i, year in enumerate(YEARS):
        _, info = compute_fossil_retirement(
            iso, float(clean_pct_vec[i]), _EGRID, _FOSSIL_MIX[iso], (1 + g) ** i, year)
        cum = max(cum, float(info.get('gas_displaced_twh', 0.0)))
        out[i] = max(0.0, base - _twh_to_gw(cum, _FOSSIL_CFS['gas']) * 1000.0
                     ) * pc.GAS_AVAILABILITY_FACTOR[iso]
    return out


def _cfe_target(year: int, ep_pct: float) -> float:
    ey = _endpoint_year(ep_pct)
    pts = [(y, t) for y, t in [(2025, 0), (2030, 50), (2035, 70), (2040, 90), (2045, 95)]
           if y < ey] + [(ey, ep_pct)]
    if year <= pts[0][0]:
        return 0.0
    if year >= ey:
        return ep_pct
    for (y0, t0), (y1, t1) in zip(pts, pts[1:]):
        if y0 <= year <= y1:
            return min(ep_pct, t0 + (t1 - t0) * (year - y0) / (y1 - y0))
    return ep_pct


# ─── LCOE / TX ──────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=None)
def _lcoe(iso: str, resource: str, year: int, cfg: RunConfig) -> float:
    if resource == 'clean_firm':
        fs, ny = pc.get_pathway_noak_window('nuclear', 'M', cfg.pathway)
        noak = pc.LCOE_TABLES.get('clean_firm', {}).get('Medium', {}).get(
            iso, pc.FOAK_NUCLEAR_NEWBUILD[iso] * 0.55)
        return pc.year_adjusted_cost(pc.FOAK_NUCLEAR_NEWBUILD[iso], noak, year, fs, ny)
    if resource == 'ccs_ccgt':
        fs, ny = pc.get_pathway_noak_window('ccs', 'M', cfg.pathway)
        noak = pc.LCOE_TABLES.get('ccs_ccgt', {}).get('Medium', {}).get(
            iso, pc.FOAK_CCS_45Q_ON[iso] * 0.55)
        return pc.year_adjusted_cost(pc.FOAK_CCS_45Q_ON[iso], noak, year, fs, ny)
    if resource == 'offshore_wind':
        if iso not in pc.OFFSHORE_ISOS:
            return 0.0
        tech = 'offshore_wind_float' if iso == 'CAISO' else 'offshore_wind_fixed'
        fs, ny = pc.LEARNING_PARAMS[tech]['M']
        return pc.year_adjusted_cost(
            pc.FOAK_OFFSHORE_WIND.get(iso, 100),
            pc.NOAK_OFFSHORE_WIND['Medium'].get(iso, 65), year, fs, ny)
    if resource == 'geothermal':
        return pc.FOAK_GEOTHERMAL if iso == 'CAISO' else 0.0
    tbl = {'solar': 'solar', 'wind': 'wind'}
    if resource in tbl:
        return pc.LCOE_TABLES[resource]['Medium'][iso]
    if resource.startswith('solar_batt'):
        return pc.LCOE_TABLES['solar']['Medium'][iso] + 30.0
    if resource.startswith('wind_batt'):
        return pc.LCOE_TABLES['wind']['Medium'][iso] + 25.0
    return 50.0 if resource == 'hydro' else 0.0


@functools.lru_cache(maxsize=None)
def _tx(iso: str, resource: str, tx_level: str) -> float:
    remap = {'solar_batt4': 'solar', 'solar_batt8': 'solar',
             'wind_batt4': 'wind', 'wind_batt8': 'wind'}
    key = remap.get(resource, resource)
    return float(pc.get_tx(key, tx_level, iso)) if key in (
        'solar', 'wind', 'clean_firm', 'ccs_ccgt', 'offshore_wind') else 0.0


def _delivered_matrix(iso, cfg):
    """(N_YEARS, R) delivered cost matrix for non-CF resources."""
    tx_r = np.array([_tx(iso, r, cfg.tx_level) for r in _NON_CF], dtype=np.float64)
    lcoe_mat = np.array([[_lcoe(iso, r, y, cfg) for r in _NON_CF]
                         for y in YEARS], dtype=np.float64)
    return lcoe_mat + tx_r[None, :]


# ─── Tranche decomposition ──────────────────────────────────────────────────

def _tranche_lcoe_vecs(iso, cfg):
    """Per-year LCOE vectors for nuclear, CCS, geothermal tranches."""
    fl, cl = cfg.firm_cost_level, cfg.ccs_cost_level
    txcf = float(pc.get_tx('clean_firm', cfg.tx_level, iso))
    txccs = float(pc.get_tx('ccs_ccgt', cfg.tx_level, iso))
    gl = cfg.geo_cost_level or ('M' if iso == 'CAISO' else None)
    nuke, ccs, geo = [np.zeros(N_YEARS, dtype=np.float64) for _ in range(3)]
    for yi, y in enumerate(YEARS):
        fs, ny = pc.get_pathway_noak_window('nuclear', fl, cfg.pathway)
        nuke[yi] = pc.year_adjusted_cost(
            pc.FOAK_NUCLEAR_NEWBUILD[iso], pc.NUCLEAR_NEWBUILD_LCOE['L'][iso],
            y, fs, ny) + txcf
        fs, ny = pc.get_pathway_noak_window('ccs', cl, cfg.pathway)
        cv = pc.year_adjusted_cost(
            pc.FOAK_CCS_45Q_ON[iso], pc.CCS_LCOE_45Q_ON['L'][iso], y, fs, ny)
        if iso == 'NEISO':
            cv += pc.NEISO_CCS_GAS_ADDER
        ccs[yi] = cv + txccs
        if iso == 'CAISO' and gl:
            fs, ny = pc.get_pathway_noak_window('geo', fl, cfg.pathway)
            geo[yi] = pc.year_adjusted_cost(
                pc.FOAK_GEOTHERMAL, pc.GEOTHERMAL_LCOE['L'], y, fs, ny) + txcf
    return nuke, ccs, geo


def decompose_clean_firm_tranches(cf_pct, iso, pathway, annual_mwh_vec, cfg):
    """Disaggregate clean_firm share % into 5 merit-order tranches.
    Returns per-tranche (Y, n) TWh and per-year (Y,) effective $/MWh."""
    n = len(cf_pct)
    atwh = annual_mwh_vec / 1e6
    bp = float(pc.GRID_MIX_SHARES[iso].get('clean_firm', 0.0))
    target = (cf_pct[None, :] / 100.0) * atwh[:, None]
    bl = (bp / 100.0) * atwh

    existing = np.minimum(target, bl[:, None])
    rem = target - existing
    ta = _TRANCHES_AVAILABLE.get(pathway, False)

    uc = float(pc.UPRATE_CAP_TWH[iso])
    uprate = np.minimum(rem, uc)
    rem = rem - uprate
    up_eff = np.full(N_YEARS, float(pc.UPRATE_LCOE[cfg.firm_cost_level]))
    feasible = np.ones_like(target, dtype=bool) if ta else ~(rem > 1e-9)

    nuke_eff, ccs_eff, geo_eff = _tranche_lcoe_vecs(iso, cfg)
    gl = cfg.geo_cost_level or ('M' if iso == 'CAISO' else None)
    cc = float(pc.CCS_CAP_TWH.get(iso, 9999.0))
    gc = float(pc.GEOTHERMAL_CAP_TWH) if (iso == 'CAISO' and gl) else 0.0

    geo_t = np.zeros_like(target)
    nuke_t = np.zeros_like(target)
    ccs_t = np.zeros_like(target)
    if ta:
        if gc > 0:
            geo_t[:] = np.minimum(rem, gc)
        rem2 = rem - geo_t
        nw = (nuke_eff <= ccs_eff) | (cc <= 0)
        ccs_t[:] = np.where(nw[:, None], 0.0, np.minimum(rem2, cc))
        nuke_t[:] = rem2 - ccs_t

    return {
        'existing_twh': existing, 'uprate_twh': uprate,
        'geo_twh': geo_t, 'nuke_new_twh': nuke_t, 'ccs_twh': ccs_t,
        'uprate_eff_lcoe': up_eff, 'geo_eff_lcoe': geo_eff,
        'nuke_new_eff_lcoe': nuke_eff, 'ccs_eff_lcoe': ccs_eff,
        'feasible': feasible,
    }

# ─── Solver context (shared setup) ───────────────────────────────────────────

@dataclass
class _Ctx:
    """Precomputed scalars/vectors shared by both solver modes."""
    ef: dict; n: int; peak_vec: np.ndarray; demand_vec: np.ndarray
    amwh: np.ndarray; score: np.ndarray; resid: np.ndarray; cf_share: np.ndarray
    cfe_tgt: np.ndarray; eg_vec: np.ndarray
    gas_raw_2025: float; exist_base: float; gaf: float; fuel_unit: float
    capex_mw: float; fom_mw: float; exist_fom: float
    delivered: np.ndarray  # (Y, R)
    nuke_eff: np.ndarray; ccs_eff: np.ndarray; geo_eff: np.ndarray
    up_eff: np.ndarray; ta: bool; is_p1: bool
    bp: float; uc: float; uc_pct_y: np.ndarray; pf: float
    cc: float; gc: float
    storage_pairs: list; grid: dict; base_dem: float; cfg: RunConfig


def _build_ctx(cfg: RunConfig, ef: dict) -> _Ctx:
    iso = cfg.iso
    n = len(ef['hourly_match_score'])
    pv = peak_demand_vec(iso, cfg.demand_growth_level)
    dv = demand_twh_vec(iso, cfg.demand_growth_level)
    amwh = dv * 1e6
    cfe_tgt = np.array([_cfe_target(y, cfg.endpoint_pct) for y in YEARS])
    eg = existing_gas_vec(iso, cfe_tgt, cfg.demand_growth_level)
    gl = cfg.geo_cost_level or ('M' if iso == 'CAISO' else None)
    nuke_eff, ccs_eff, geo_eff = _tranche_lcoe_vecs(iso, cfg)
    slm = {'battery_dispatch_pct': 'battery', 'battery8_dispatch_pct': 'battery8',
           'ldes_dispatch_pct': 'ldes', 'h2_dispatch_pct': 'h2'}
    sp = [(sc, float(pc.LCOE_TABLES[lk]['Medium'][iso]))
          for sc, lk in slm.items() if float(ef[sc].sum()) > 0]
    uc = float(pc.UPRATE_CAP_TWH[iso])
    return _Ctx(
        ef=ef, n=n, peak_vec=pv, demand_vec=dv, amwh=amwh,
        score=ef['hourly_match_score'],
        resid=ef['resid_norm_p9997'].astype(np.float64),
        cf_share=ef['clean_firm'].astype(np.float64) / 100.0,
        cfe_tgt=cfe_tgt, eg_vec=eg,
        gas_raw_2025=_GAS_RAW_2025[iso],
        exist_base=float(pc.EXISTING_GAS_CAPACITY_MW[iso]),
        gaf=float(pc.GAS_AVAILABILITY_FACTOR[iso]),
        fuel_unit=pc.WHOLESALE_PRICES[iso] + pc.FUEL_ADJUSTMENTS[iso]['Medium'],
        capex_mw=pc.NEW_CCGT_COST_KW_YR[iso] * 1000.0,
        fom_mw=pc.NEW_CCGT_FOM_KW_YR[iso] * 1000.0,
        exist_fom=float(pc.EXISTING_GAS_CAPACITY_MW[iso]) * pc.EXISTING_GAS_FOM_KW_YR[iso] * 1000.0,
        delivered=_delivered_matrix(iso, cfg),
        nuke_eff=nuke_eff, ccs_eff=ccs_eff, geo_eff=geo_eff,
        up_eff=np.full(N_YEARS, float(pc.UPRATE_LCOE[cfg.firm_cost_level])),
        ta=_TRANCHES_AVAILABLE.get(cfg.pathway, False),
        is_p1=cfg.pathway in ('1', '1a'),
        bp=float(pc.GRID_MIX_SHARES[iso].get('clean_firm', 0.0)),
        uc=uc,
        uc_pct_y=uc / np.maximum(dv, 1e-6) * 100.0,
        pf=float(pc.GRID_MIX_SHARES[iso].get('clean_firm', 0.0)) + 0.01,
        cc=float(pc.CCS_CAP_TWH.get(iso, 9999.0)),
        gc=float(pc.GEOTHERMAL_CAP_TWH) if (iso == 'CAISO' and gl) else 0.0,
        storage_pairs=sp, grid=pc.GRID_MIX_SHARES[iso],
        base_dem=float(pc.REGIONAL_DEMAND_TWH[iso]), cfg=cfg,
    )


def _init_floors(ctx):
    fr = np.array([ctx.base_dem * ctx.grid.get(r, 0.0) / 100.0 for r in _NON_CF], dtype=np.float64)
    fc = ctx.base_dem * ctx.grid.get('clean_firm', 0.0) / 100.0
    return fr, fc, 0.0, 0.0, 0.0, 0.0


# ─── Chunk scoring kernel ───────────────────────────────────────────────────

def _score_chunk(s, e, yi, ctx, fr, fc, fu, fg, fn, fcc, rmg):
    """Score chunk [s,e) at year yi. Returns scores, masks, tranche values."""
    k, ef = e - s, ctx.ef
    dy = float(ctx.demand_vec[yi])
    amwh_y = float(ctx.amwh[yi])
    bl_y = (ctx.bp / 100.0) * dy

    # Resource shares
    snr = np.empty((k, _N_NON_CF), dtype=np.float64)
    for ri, r in enumerate(_NON_CF):
        snr[:, ri] = ef[r][s:e].astype(np.float64) / 100.0
    tnr = snr * dy
    tcf = ctx.cf_share[s:e] * dy

    # Inline tranche decomposition
    ex = np.minimum(tcf, bl_y)
    rem = tcf - ex
    up = np.minimum(rem, ctx.uc); rem = rem - up
    nk_yi, cc_yi = float(ctx.nuke_eff[yi]), float(ctx.ccs_eff[yi])
    if ctx.ta:
        feas = np.ones(k, dtype=bool)
        ge = np.minimum(rem, ctx.gc) if ctx.gc > 0 else np.zeros(k)
        rem = rem - ge
        if nk_yi <= cc_yi or ctx.cc <= 0:
            nk, cc = rem, np.zeros(k)
        else:
            cc = np.minimum(rem, ctx.cc); nk = rem - cc
    else:
        feas = rem <= 1e-9
        ge = nk = cc = np.zeros(k, dtype=np.float64)

    # Ratchet
    ratch = (np.all(tnr >= fr[None, :] - RATCHET_TOL, axis=1)
             & (tcf >= fc - RATCHET_TOL) & (up >= fu - RATCHET_TOL)
             & (ge >= fg - RATCHET_TOL) & (nk >= fn - RATCHET_TOL)
             & (cc >= fcc - RATCHET_TOL))

    # Pathway mask
    if ctx.is_p1:
        pm = ((ef['clean_firm'][s:e] <= ctx.pf + float(ctx.uc_pct_y[yi]) + 0.5)
              & (ef['ccs_ccgt'][s:e] <= 0.5))
    else:
        pm = np.ones(k, dtype=bool)

    cfe_s = ctx.score[s:e] >= float(ctx.cfe_tgt[yi]) - 0.5
    cfe_r = ctx.score[s:e] >= float(ctx.cfe_tgt[yi]) - 5.0

    # Gas sizing
    gap = ctx.resid[s:e] * amwh_y / max(1e-9, ctx.gaf)
    gn = np.maximum(0.0, ctx.exist_base + gap - ctx.gas_raw_2025)
    gr = np.maximum(0.0, gn - float(ctx.eg_vec[yi]))
    rmg[s:e] = np.maximum(rmg[s:e], gr)
    fl = rmg[s:e]

    # Costs
    gfx = (ctx.capex_mw * fl + ctx.fom_mw * fl
           + dy * 1e6 * np.maximum(0.0, 1.0 - ctx.score[s:e] / 100.0)
           * ctx.fuel_unit + ctx.exist_fom)
    sc = np.zeros(k, dtype=np.float64)
    for col, unit in ctx.storage_pairs:
        sc += dy * (ef[col][s:e] / 100.0) * unit

    # Sunk-cost score
    inc_nr = np.maximum(0.0, tnr - fr[None, :])
    s_nr = (inc_nr * ctx.delivered[yi][None, :]).sum(axis=1) * 1e6
    s_cf = (np.maximum(0.0, up - fu) * float(ctx.up_eff[yi])
            + np.maximum(0.0, ge - fg) * float(ctx.geo_eff[yi])
            + np.maximum(0.0, nk - fn) * float(ctx.nuke_eff[yi])
            + np.maximum(0.0, cc - fcc) * float(ctx.ccs_eff[yi])) * 1e6
    rs = s_nr + s_cf + sc + gfx

    return rs, ratch, pm, feas, cfe_s, cfe_r, tnr, tcf, up, ge, nk, cc, snr


def _update_tiers(rs, ratch, pm, feas, cfe_s, cfe_r, bs, bi, off):
    """Update 4-tier best-score/idx trackers for a chunk."""
    masks = [ratch & pm & cfe_s & feas, ratch & pm & cfe_r & feas,
             ratch & pm, pm]
    for ti, m in enumerate(masks):
        if not m.any():
            continue
        v = np.where(m, rs, np.inf)
        li = int(np.argmin(v))
        if float(v[li]) < bs[ti]:
            bs[ti] = float(v[li]); bi[ti] = off + li


def _pick_winner(bs, bi):
    for ti in range(4):
        if np.isfinite(bs[ti]):
            return bi[ti], ti
    return -1, -1


def _update_floors(w, yi, ctx, fr, fc, fu, fg, fn, fcc):
    dy = float(ctx.demand_vec[yi])
    wr = np.array([float(ctx.ef[r][w]) / 100.0 * dy for r in _NON_CF], dtype=np.float64)
    wc = float(ctx.cf_share[w]) * dy
    we = min(wc, (ctx.bp / 100.0) * dy)
    rem = wc - we
    wu = min(rem, ctx.uc); rem -= wu
    if ctx.ta:
        wg = min(rem, ctx.gc) if ctx.gc > 0 else 0.0; rem -= wg
        if float(ctx.nuke_eff[yi]) <= float(ctx.ccs_eff[yi]) or ctx.cc <= 0:
            wn, wcc = rem, 0.0
        else:
            wcc = min(rem, ctx.cc); wn = rem - wcc
    else:
        wg = wn = wcc = 0.0
    fr = np.maximum(fr, wr)
    if ctx.cfg.pathway != '1':
        fc = max(fc, wc)
    return fr, fc, max(fu, wu), max(fg, wg), max(fn, wn), max(fcc, wcc)


# ─── PathwayRunResult ────────────────────────────────────────────────────────

@dataclass
class PathwayRunResult:
    config: RunConfig; ledger: VintageLedger; winners: np.ndarray
    cum_gas: np.ndarray; active_fleet: np.ndarray; gas_cf: np.ndarray
    achieved_cfe: np.ndarray; demand_vec: np.ndarray; peak_vec: np.ndarray
    eg_used: np.ndarray; annual_rows: list; end_dispatch: np.ndarray
    feasibility: dict; stranding: dict; retirement: list
    vre_curt: dict; end_mix: dict; end_storage: dict
    rel_tax: dict; headline: dict; ef: dict
    clean_pct: np.ndarray; priced_curt: np.ndarray
    foresight_meta: Optional[dict] = None
    cascade_tier_y: Optional[np.ndarray] = None


# ─── Myopic solver ───────────────────────────────────────────────────────────

def solve_pathway(cfg: RunConfig, *, ef_override=None) -> PathwayRunResult:
    ef = ef_override or load_ef_pool(cfg.iso)
    ctx = _build_ctx(cfg, ef)
    n = ctx.n
    eyi = YEARS.index(_endpoint_year(float(cfg.endpoint_pct)))

    rmg = np.zeros(n, dtype=np.float64)
    fr, fc, fu, fg, fn, fcc = _init_floors(ctx)
    winners = np.empty(N_YEARS, dtype=np.int64)
    ctier = np.zeros(N_YEARS, dtype=np.int32)

    for yi in range(eyi + 1):
        bs, bi = [np.inf] * 4, [-1] * 4
        for s in range(0, n, _POOL_CHUNK):
            e = min(s + _POOL_CHUNK, n)
            rs, ra, pm, fe, cs, cr, *_ = _score_chunk(
                s, e, yi, ctx, fr, fc, fu, fg, fn, fcc, rmg)
            _update_tiers(rs, ra, pm, fe, cs, cr, bs, bi, s)

        w, tier = _pick_winner(bs, bi)
        if tier == -1:
            # Safety fallback: global argmin without masks
            gb, gi = np.inf, 0
            for s in range(0, n, _POOL_CHUNK):
                e = min(s + _POOL_CHUNK, n)
                rs = _score_chunk(s, e, yi, ctx, fr, fc, fu, fg, fn, fcc, rmg)[0]
                li = int(np.argmin(rs))
                if float(rs[li]) < gb:
                    gb, gi = float(rs[li]), s + li
            w, tier = gi, 3

        winners[yi] = w; ctier[yi] = tier
        fr, fc, fu, fg, fn, fcc = _update_floors(
            w, yi, ctx, fr, fc, fu, fg, fn, fcc)

    # Post-endpoint freeze
    winners[eyi + 1:] = int(winners[eyi])
    ctier[eyi + 1:] = -1
    result = _finalize(ctx, winners, ctier)
    result.cascade_tier_y = ctier
    return result


# ─── Foresight solver ────────────────────────────────────────────────────────

# Target shares cache
_TS_CACHE: dict = {}


def _get_target_shares(iso, ep_pct, dgl='Medium', gl=None, fl='M', cl='M', tl='Medium'):
    ck = (iso, round(ep_pct, 4), dgl, fl, cl, tl, gl or 'X')
    if ck in _TS_CACHE:
        return _TS_CACHE[ck]
    gc = (gl or 'M') if iso == 'CAISO' else 'X'
    tc = tl[0].upper(); fk = fl[0].upper(); ccl = cl[0].upper()
    sk = f'{fk}M{fk}{fk}_M_{tc}_{ccl}1_{gc}'
    tbl = pq.read_table(_STEP22A_DIR / f'step_2_2a_DG_{iso}_{_threshold_tag(ep_pct)}.parquet')
    sub = tbl.filter(pacompute.and_(
        pacompute.equal(tbl.column('scenario'), sk),
        pacompute.equal(tbl.column('growth_level'), dgl)))
    if sub.num_rows != 1:
        raise RuntimeError(f"Expected 1 row for {iso}/{sk}, got {sub.num_rows}")
    row = sub.slice(0, 1).to_pylist()[0]
    at = float(row['annual_demand_mwh']) / 1e6
    tgt = np.zeros(len(_FORESIGHT_SHARE_KEYS), dtype=np.float64)
    for i, r in enumerate(_RESOURCE_COLS):
        tgt[i] = float(row.get(f'mix_{r}', 0.0) or 0.0) / 100.0
    tgt[11] = float(row.get('tranche_uprate_twh', 0.0) or 0.0) / at
    tgt[12] = float(row.get('tranche_geo_twh', 0.0) or 0.0) / at
    tgt[13] = float(row.get('tranche_nuclear_newbuild_twh', 0.0) or 0.0) / at
    tgt[14] = float(row.get('tranche_ccs_tranche_twh', 0.0) or 0.0) / at
    _TS_CACHE[ck] = tgt
    return tgt


def solve_pathway_with_foresight(cfg: RunConfig, *, ef_override=None) -> PathwayRunResult:
    """Foresight streaming solver: row_score + λ·w·C·‖projection−target‖²."""
    if cfg.endpoint_year_val is None or cfg.endpoint_target_shares is None:
        raise ValueError("Foresight solver requires endpoint_year_val and endpoint_target_shares")

    ef = ef_override or load_ef_pool(cfg.iso)
    ctx = _build_ctx(cfg, ef)
    n = ctx.n
    eyi = YEARS.index(int(cfg.endpoint_year_val))
    tgt = np.asarray(cfg.endpoint_target_shares, dtype=np.float64)
    lam = float(cfg.foresight_lambda or 0.0)
    fC = float(ctx.demand_vec[eyi]) * 1e6 * FORESIGHT_REF_LCOE
    hdenom = float(cfg.endpoint_year_val - BASE_YEAR)
    dend = float(ctx.demand_vec[eyi])
    ncf_idx = {r: i for i, r in enumerate(_NON_CF)}

    rmg = np.zeros(n, dtype=np.float64)
    fr, fc, fu, fg, fn, fcc = _init_floors(ctx)
    winners = np.empty(N_YEARS, dtype=np.int64)
    ctier = np.zeros(N_YEARS, dtype=np.int32)
    diag = []

    for yi in range(eyi + 1):
        year = YEARS[yi]
        wt = max(0.0, (cfg.endpoint_year_val - year) / hdenom)
        dy = float(ctx.demand_vec[yi])
        ratio = dy / dend
        bs, bi = [np.inf] * 4, [-1] * 4
        feas_cnt = 0

        for s in range(0, n, _POOL_CHUNK):
            e = min(s + _POOL_CHUNK, n)
            rs, ra, pm, fe, cs, cr, tnr, tcf, up, ge, nk, cc, snr = \
                _score_chunk(s, e, yi, ctx, fr, fc, fu, fg, fn, fcc, rmg)

            # 15-dim endpoint projection
            k = e - s
            proj = np.empty((k, len(_FORESIGHT_SHARE_KEYS)), dtype=np.float64)
            for j, r in enumerate(_RESOURCE_COLS):
                proj[:, j] = (ctx.cf_share[s:e] if r == 'clean_firm'
                              else snr[:, ncf_idx[r]]) * ratio
            proj[:, 11] = up / dend
            proj[:, 12] = ge / dend
            proj[:, 13] = nk / dend
            proj[:, 14] = cc / dend

            diff = proj - tgt[None, :]
            penalty = (lam * wt * fC) * (diff * diff).sum(axis=1)
            fs = rs + penalty

            _update_tiers(fs, ra, pm, fe, cs, cr, bs, bi, s)
            feas_cnt += int((ra & pm & cs & fe).sum())

        w, tier = _pick_winner(bs, bi)
        if tier == -1:
            w, tier = 0, 3

        winners[yi] = w; ctier[yi] = tier
        fr, fc, fu, fg, fn, fcc = _update_floors(
            w, yi, ctx, fr, fc, fu, fg, fn, fcc)
        diag.append({'year': year, 'feasible': feas_cnt,
                     'tier': tier, 'weight': round(wt, 4)})

    winners[eyi + 1:] = int(winners[eyi])
    ctier[eyi + 1:] = -1

    result = _finalize(ctx, winners, ctier)
    result.cascade_tier_y = ctier
    result.foresight_meta = {
        'solver_mode': 'foresight', 'endpoint_year': int(cfg.endpoint_year_val),
        'endpoint_yi': eyi, 'foresight_lambda': lam, 'foresight_C_usd': fC,
        'target_shares': {k: float(v) for k, v in zip(_FORESIGHT_SHARE_KEYS, tgt)},
        'cascade_tier_y': ctier[:eyi + 1].tolist(),
        'cascade_activations': int((ctier[:eyi + 1] > 0).sum()),
        'per_year': diag,
    }
    return result

# ─── Output helpers ──────────────────────────────────────────────────────────

def _archetype_for_mix(iso, mix_pct):
    am = _MANIFEST.get(iso)
    if am is None:
        return None
    cols = ['mix_clean_firm', 'mix_solar', 'mix_wind', 'mix_offshore_wind', 'mix_ccs_ccgt',
            'mix_hydro', 'mix_solar_batt4', 'mix_solar_batt8', 'mix_wind_batt4', 'mix_wind_batt8']
    target = np.array([mix_pct.get(c[4:], 0.0) for c in cols])
    M = np.column_stack([am.column(c).to_numpy() for c in cols])
    return am.column('archetype_key')[int(np.argmin(np.abs(M - target).sum(axis=1)))].as_py()


def _vre_curtailment(iso, mix_pct):
    keys = ('solar', 'wind', 'offshore_wind', 'solar_batt4', 'solar_batt8',
            'wind_batt4', 'wind_batt8')
    out = {k: 0.0 for k in keys}
    arch = _archetype_for_mix(iso, mix_pct)
    cache = _DCACHE.get(iso)
    if cache is None or arch is None:
        return out
    al = cache.column('archetype_key').to_pylist()
    if arch not in al:
        return out
    ri = al.index(arch)
    for r in keys:
        if mix_pct.get(r, 0.0) <= 0:
            continue
        mc, sc = f'matched_{r}', f'surplus_{r}'
        if mc not in cache.column_names:
            continue
        m = float(np.asarray(cache.column(mc)[ri].as_py()).sum())
        s = float(np.asarray(cache.column(sc)[ri].as_py()).sum())
        out[r] = round(s / (m + s), 4) if (m + s) > 0 else 0.0
    return out


def _endpoint_dispatch(iso, mix_pct):
    arch = _archetype_for_mix(iso, mix_pct)
    cache = _DCACHE.get(iso)
    if cache is None or arch is None:
        return np.zeros(HOURS_PER_YEAR)
    al = cache.column('archetype_key').to_pylist()
    if arch not in al:
        return np.zeros(HOURS_PER_YEAR)
    return np.asarray(cache.column('total_clean')[al.index(arch)].as_py(), dtype=np.float64)


def _priced_curt_vec(ef, winners, dv, cfg):
    end_mix = {r: float(ef[r][winners[-1]]) for r in _RESOURCE_COLS}
    cf = _vre_curtailment(cfg.iso, end_mix)
    out = np.zeros(N_YEARS, dtype=np.float64)
    for r in ('solar', 'wind', 'offshore_wind', 'solar_batt4', 'solar_batt8',
              'wind_batt4', 'wind_batt8'):
        f = cf.get(r, 0.0)
        if f <= 0:
            continue
        u = _lcoe(cfg.iso, r, END_YEAR, cfg)
        out += dv * 1e6 * (ef[r][winners] / 100.0) * f * u
    return out


def _retirement_timeline(iso, clean_pct_vec, level):
    g = pc.DEMAND_GROWTH_RATES[iso][level]
    cum = {'coal': 0.0, 'oil': 0.0, 'gas': 0.0}
    rows = []
    for i, year in enumerate(YEARS):
        _, info = compute_fossil_retirement(
            iso, float(clean_pct_vec[i]), _EGRID, _FOSSIL_MIX[iso], (1 + g) ** i, year)
        for f in cum:
            cum[f] = max(cum[f], float(info.get(f'{f}_displaced_twh', 0.0)))
        rows.append({
            'year': year, 'clean_pct': round(float(clean_pct_vec[i]), 3),
            **{f'{f}_retired_gw': round(_twh_to_gw(cum[f], _FOSSIL_CFS[f]), 3) for f in cum},
            **{f'{f}_retired_twh': round(cum[f], 2) for f in cum},
        })
    return rows


# ─── Ledger builder ──────────────────────────────────────────────────────────

def _build_ledger(ef, winners, cfg):
    led = VintageLedger()
    bd = float(pc.REGIONAL_DEMAND_TWH[cfg.iso])
    grid = pc.GRID_MIX_SHARES[cfg.iso]
    g = pc.DEMAND_GROWTH_RATES[cfg.iso][cfg.demand_growth_level]

    # Base-year existing vintages
    for r, sh in grid.items():
        if sh <= 0:
            continue
        led.add(Vintage(resource='clean_firm_existing' if r == 'clean_firm' else r,
                        cod_year=BASE_YEAR, twh_per_year=bd * sh / 100.0,
                        locked_lcoe=0.0, tx_adder=0.0))

    # Per-year delta vintages (non-CF)
    floor = {r: bd * grid.get(r, 0.0) / 100.0 for r in _NON_CF}
    for yi, year in enumerate(YEARS):
        w = int(winners[yi])
        dy = bd * (1 + g) ** yi
        for r in _NON_CF:
            tgt = dy * float(ef[r][w]) / 100.0
            delta = tgt - floor[r]
            if delta > 0.005:
                led.add(Vintage(resource=r, cod_year=year, twh_per_year=delta,
                                locked_lcoe=_lcoe(cfg.iso, r, year, cfg),
                                tx_adder=_tx(cfg.iso, r, cfg.tx_level)))
            floor[r] = max(floor[r], tgt)

    # Per-year delta vintages (CF tranches)
    amwh = np.array([bd * (1 + g) ** i * 1e6 for i in range(N_YEARS)], dtype=np.float64)
    tg = decompose_clean_firm_tranches(ef['clean_firm'][winners], cfg.iso, cfg.pathway, amwh, cfg)
    idx = np.arange(N_YEARS)
    for tn, twh_y, eff_y in [
        ('uprate', tg['uprate_twh'][idx, idx], tg['uprate_eff_lcoe']),
        ('geo', tg['geo_twh'][idx, idx], tg['geo_eff_lcoe']),
        ('nuke_new', tg['nuke_new_twh'][idx, idx], tg['nuke_new_eff_lcoe']),
        ('ccs', tg['ccs_twh'][idx, idx], tg['ccs_eff_lcoe']),
    ]:
        fl = 0.0
        for yi, year in enumerate(YEARS):
            cur = float(twh_y[yi])
            if cur - fl > 0.005:
                led.add(Vintage(resource=_CF_TRANCHE_VKEY[tn], cod_year=year,
                                twh_per_year=cur - fl,
                                locked_lcoe=float(eff_y[yi]), tx_adder=0.0))
            fl = max(fl, cur)
    return led


def _gross_op_vec(ledger):
    """Vectorized gross operating cost from ledger entries."""
    if not ledger.entries:
        return np.zeros(N_YEARS, dtype=np.float64)
    cods = np.array([v.cod_year for v in ledger.entries], dtype=np.int32)
    rets = np.array([v.retire_year or (END_YEAR + 1) for v in ledger.entries], dtype=np.int32)
    costs = np.array([v.twh_per_year * 1e6 * (v.locked_lcoe + v.tx_adder)
                      for v in ledger.entries], dtype=np.float64)
    ya = np.array(YEARS, dtype=np.int32)
    active = (cods[None, :] <= ya[:, None]) & (rets[None, :] > ya[:, None])
    return (active * costs[None, :]).sum(axis=1)


# ─── Finalization ────────────────────────────────────────────────────────────

def _finalize(ctx, winners, ctier):
    cfg = ctx.cfg
    acfe = ctx.score[winners]
    eg_real = existing_gas_vec(cfg.iso, acfe, cfg.demand_growth_level)

    rw = ctx.resid[winners]
    gap = rw * ctx.amwh / max(1e-9, ctx.gaf)
    gn = np.maximum(0.0, ctx.exist_base + gap - ctx.gas_raw_2025)
    cng = np.maximum(0.0, gn - eg_real)
    af = np.maximum.accumulate(cng)
    fm = float(af.max())
    py = int(YEARS[int(np.argmax(af))])

    tgmw = ctx.exist_base + af
    gg = np.maximum(0.0, 1.0 - acfe / 100.0) * ctx.demand_vec * 1e6
    gcf = np.where(tgmw > 1e-6, np.minimum(1.0, gg / (tgmw * HOURS_PER_YEAR)), 0.0)
    egu = np.minimum(eg_real, gn)

    ledger = _build_ledger(ctx.ef, winners, cfg)
    pc_vec = _priced_curt_vec(ctx.ef, winners, ctx.demand_vec, cfg)
    gop = _gross_op_vec(ledger)

    ga = af * ctx.capex_mw
    gf = af * ctx.fom_mw
    gfu = np.maximum(0.0, 1.0 - acfe / 100.0) * ctx.demand_vec * 1e6 * ctx.fuel_unit

    rows = []
    for yi, year in enumerate(YEARS):
        g = float(gop[yi])
        gaf_val = float(ga[yi] + gf[yi])
        net = g + gaf_val + ctx.exist_fom + gfu[yi] + pc_vec[yi]
        rows.append({
            'year': year, 'demand_twh': round(float(ctx.demand_vec[yi]), 3),
            'gross_operating_usd': round(g, 2),
            'new_gas_annualized_capex_fom_usd': round(gaf_val, 2),
            'existing_gas_fom_carried_usd': round(ctx.exist_fom, 2),
            'gas_fuel_usd': round(float(gfu[yi]), 2),
            'capacity_rev_netted_usd': 0.0,
            'net_annual_cost_usd': round(net, 2),
            'achieved_cfe_pct': round(float(acfe[yi]), 3),
            'gas_fleet_cf': round(float(gcf[yi]), 4),
            'priced_vre_curtailment_usd_this_year': round(float(pc_vec[yi]), 2),
        })

    g2050 = float(cng[-1])
    smw = max(0.0, fm - g2050)
    yr = max(0.0, NEW_GAS_ASSET_LIFE_YEARS - (END_YEAR - py))
    scap = smw * 1000.0 * CCGT_OVERNIGHT_CAPEX_USD_KW * (yr / NEW_GAS_ASSET_LIFE_YEARS)

    ew = int(winners[-1])
    em = {r: round(float(ctx.ef[r][ew]), 3) for r in _RESOURCE_COLS}
    es = {c: round(float(ctx.ef[c][ew]), 4) for c in _STORAGE_COLS}
    vc = _vre_curtailment(cfg.iso, em)
    ed = _endpoint_dispatch(cfg.iso, em)

    ra = pc.RESOURCE_ADEQUACY_MARGIN
    rt_comp = {
        'new_gas_capex_annualized_usd': round(float(ga.sum()), 2),
        'new_gas_fom_usd': round(float(gf.sum()), 2),
        'existing_gas_fom_carried_usd': round(ctx.exist_fom * N_YEARS, 2),
        'priced_vre_curtailment_usd': round(float(pc_vec.sum()), 2),
        'vre_storage_overbuild_capex_usd': 0.0,
    }
    td = float((ctx.demand_vec * 1e6).sum())
    rt_total = sum(rt_comp.values())

    undisc = sum(r['net_annual_cost_usd'] for r in rows)
    npvs = {f'npv_at_{k}': round(sum(r['net_annual_cost_usd'] / (1 + rate) ** (r['year'] - BASE_YEAR)
                                     for r in rows), 2) for k, rate in DISCOUNT_RATES.items()}

    infeas_y = [YEARS[i] for i in range(N_YEARS) if ctier[i] >= 2]
    ratch_y = [YEARS[i] for i in range(N_YEARS) if ctier[i] >= 3]
    notes = []
    if infeas_y:
        notes.append(f"cf target exceeds tranche budget in: {infeas_y}")
    if ratch_y:
        notes.append(f"ratchet relaxed in: {ratch_y}")

    return PathwayRunResult(
        config=cfg, ledger=ledger, winners=winners, cum_gas=cng,
        active_fleet=af, gas_cf=gcf, achieved_cfe=acfe,
        demand_vec=ctx.demand_vec, peak_vec=ctx.peak_vec, eg_used=egu,
        annual_rows=rows, end_dispatch=ed,
        feasibility={'physical': not infeas_y and not ratch_y, 'notes': notes},
        stranding={
            'methodology': 'peak_year_snapshot_v2',
            'asset_life_years': NEW_GAS_ASSET_LIFE_YEARS,
            'overnight_capex_usd_kw': CCGT_OVERNIGHT_CAPEX_USD_KW,
            'fleet_size_mw': round(fm, 2), 'peak_year': py,
            'new_gas_need_2050_mw': round(g2050, 2),
            'stranded_mw_at_2050': round(smw, 2),
            'total_new_gas_built_mw': round(fm, 2),
            'total_stranded_capex_usd': round(scap, 2),
            'stranded_vintage_count': int(smw > 0),
            'cf_threshold_default': 0.15,
            'cf_threshold_sensitivity': [0.10, 0.15, 0.20],
            'consecutive_years': 2, 'required_real_return': 0.07, 'reference_cf': 0.15,
        },
        retirement=_retirement_timeline(cfg.iso, acfe, cfg.demand_growth_level),
        vre_curt=vc, end_mix=em, end_storage=es,
        rel_tax={
            'total_demand_mwh_2025_2050': round(td, 2),
            'components_usd': rt_comp, 'total_usd': round(rt_total, 2),
            'usd_per_mwh': round(rt_total / max(1.0, td), 4),
        },
        headline={
            'achieved_cfe_pct': round(float(acfe[-1]), 3),
            'undiscounted_cost_usd': round(undisc, 2), **npvs,
            'endpoint_threshold': cfg.endpoint_pct,
            'endpoint_year': int(_endpoint_year(cfg.endpoint_pct)),
            'pivot': {'pivoted': False, 'pivot_year': None, 'pivot_reason': None},
        },
        ef=ctx.ef, clean_pct=acfe, priced_curt=pc_vec,
    )

# ─── Serialization ───────────────────────────────────────────────────────────

def _serialize_buildout(r: PathwayRunResult):
    """Build annual_buildout table for JSON output."""
    cfg, ef = r.config, r.ef
    grid = pc.GRID_MIX_SHARES[cfg.iso]
    bd = float(pc.REGIONAL_DEMAND_TWH[cfg.iso])
    ra = pc.RESOURCE_ADEQUACY_MARGIN

    amwh = r.demand_vec * 1e6
    wcf = ef['clean_firm'][r.winners]
    tg = decompose_clean_firm_tranches(wcf, cfg.iso, cfg.pathway, amwh, cfg)
    idx = np.arange(N_YEARS)
    tr_twh = {n: tg[f'{n}_twh'][idx, idx]
              for n in ('uprate', 'geo', 'nuke_new', 'ccs')}
    tr_eff = {n: tg[f'{n}_eff_lcoe'] if n != 'nuke_new' else tg['nuke_new_eff_lcoe']
              for n in ('uprate', 'geo', 'nuke_new', 'ccs')}
    prev_tr = {n: 0.0 for n in tr_twh}
    floor = {res: bd * grid.get(res, 0.0) / 100.0 for res in _NON_CF}

    out = []
    for yi, year in enumerate(YEARS):
        w = int(r.winners[yi])
        dy = float(r.demand_vec[yi])
        nv = []
        for res in _NON_CF:
            tgt = dy * float(ef[res][w]) / 100.0
            delta = tgt - floor[res]
            if delta > 0.005:
                nv.append({'resource': res, 'twh_per_year': round(delta, 4),
                           'locked_lcoe_usd_per_mwh': round(_lcoe(cfg.iso, res, year, cfg), 2),
                           'tx_adder_usd_per_mwh': round(_tx(cfg.iso, res, cfg.tx_level), 2)})
            floor[res] = max(floor[res], tgt)
        for tn, twh_y in tr_twh.items():
            cur = float(twh_y[yi])
            d = cur - prev_tr[tn]
            if d > 0.005:
                nv.append({'resource': _CF_TRANCHE_VKEY[tn], 'twh_per_year': round(d, 4),
                           'locked_lcoe_usd_per_mwh': round(float(tr_eff[tn][yi]), 2),
                           'tx_adder_usd_per_mwh': 0.0})
            prev_tr[tn] = max(prev_tr[tn], cur)
        is_pk = (year == r.stranding['peak_year'])
        out.append({
            'year': year, 'new_vintages': nv,
            'gas_sizing': {
                'peak_demand_mw': round(float(r.peak_vec[yi]), 2),
                'ra_peak_mw': round(float(r.peak_vec[yi] * (1 + ra)), 2),
                'total_clean_peak_mw': round(float(ef['clean_peak_hour_mw'][w]), 2),
                'existing_gas_used_mw': round(float(r.eg_used[yi]), 2),
                'new_gas_required_cumulative_mw': round(float(r.cum_gas[yi]), 2),
                'new_gas_built_this_year_mw': round(r.stranding['fleet_size_mw'], 2) if is_pk else 0.0,
                'active_new_gas_fleet_mw': round(float(r.active_fleet[yi]), 2),
                'gas_fleet_cf': round(float(r.gas_cf[yi]), 4),
            },
        })
    return out


def serialize_v2(r: PathwayRunResult) -> dict:
    cfg = r.config
    return {
        'schema_version': 2,
        'run_key': f'{cfg.iso}__pathway{cfg.pathway}__{_ep_tag(cfg.endpoint_pct)}',
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
            'annual_buildout': _serialize_buildout(r),
            'annual_cost': r.annual_rows,
            'endpoint_hourly_dispatch': [round(float(x), 8) for x in r.end_dispatch],
            'new_gas_fleet': [{
                'year_built': r.stranding['peak_year'],
                'initial_cap_mw': r.stranding['fleet_size_mw'],
                'retirement_year': None,
                'stranded_flag': r.stranding['stranded_mw_at_2050'] > 0,
                'stranded_year': END_YEAR if r.stranding['stranded_mw_at_2050'] > 0 else None,
                'stranded_capex_usd': r.stranding['total_stranded_capex_usd'],
            }],
        },
        'reliability_tax': r.rel_tax,
        'stranding_metadata': r.stranding,
        'retirement_timeline': r.retirement,
        'vre_curtailment_at_endpoint': r.vre_curt,
        'endpoint_mix_pct': r.end_mix,
        'endpoint_storage_pct': r.end_storage,
        'terminal_ledger': r.ledger.serialize(),
    }


def serialize_v3(r: PathwayRunResult) -> dict:
    """Schema v3: v2 + foresight diagnostics."""
    base = serialize_v2(r)
    cfg, meta = r.config, r.foresight_meta or {}
    base['schema_version'] = '3.0.0'
    base['config']['solver_mode'] = cfg.solver_mode
    base['config']['foresight_lambda'] = (
        float(cfg.foresight_lambda) if cfg.foresight_lambda is not None else None)
    base['config']['endpoint_year'] = (
        int(cfg.endpoint_year_val) if cfg.endpoint_year_val is not None else None)
    base['config']['endpoint_target_shares'] = (
        {k: round(float(v), 6) for k, v in meta.get('target_shares', {}).items()}
        if meta else None)

    eyi = meta.get('endpoint_yi')
    t90 = None
    if eyi is not None:
        hits = np.where(r.achieved_cfe[:eyi + 1] >= 89.9999)[0]
        if hits.size:
            t90 = int(YEARS[int(hits[0])])

    base['headline'].update({
        'foresight_penalty_vs_myopic': None, 'p1_penalty_vs_foresight': None,
        'p1_reliability_tax_premium': None, 'time_to_90pct': t90,
        'cascade_activations': meta.get('cascade_activations'),
    })
    base['foresight_diagnostics'] = {
        'endpoint_year': meta.get('endpoint_year'),
        'foresight_C_usd': meta.get('foresight_C_usd'),
        'cascade_tier_y': meta.get('cascade_tier_y'),
        'per_year': meta.get('per_year', []),
    }
    return base


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _run_one(iso, pathway, endpoint, solver_mode='myopic',
             foresight_lambda=None, ey=None, ets=None):
    ep_pct = endpoint * 100
    if solver_mode == 'foresight':
        if ey is None:
            ey = _endpoint_year(ep_pct)
        if ets is None:
            ets = tuple(float(x) for x in _get_target_shares(iso, ep_pct))
    cfg = RunConfig(
        iso=iso, pathway=pathway, endpoint=endpoint, endpoint_pct=ep_pct,
        solver_mode=solver_mode, foresight_lambda=foresight_lambda,
        endpoint_year_val=ey, endpoint_target_shares=ets,
    )
    if cfg.solver_mode == 'foresight':
        result = solve_pathway_with_foresight(cfg)
        payload = serialize_v3(result)
    else:
        result = solve_pathway(cfg)
        payload = serialize_v2(result)
    out = cfg.output_path
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(payload, f)
    os.replace(tmp, out)
    print(f"[done] {out}  fleet={result.stranding['fleet_size_mw']:.0f} MW  "
          f"cfe={result.headline['achieved_cfe_pct']:.2f}%", flush=True)
    return out


def _prebuild_sidecar(pair):
    _load_or_build_peakclean(*pair)


def _run_iso_all(iso):
    for p in PATHWAYS:
        for ep in ENDPOINT_TO_THRESHOLD:
            _run_one(iso, p, ep)
            _lcoe.cache_clear(); _tx.cache_clear()
            gc.collect()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Pathway optimizer v3')
    ap.add_argument('--iso', choices=ISOS)
    ap.add_argument('--pathway', choices=PATHWAYS)
    ap.add_argument('--endpoint', type=float, choices=list(ENDPOINT_TO_THRESHOLD))
    ap.add_argument('--all', action='store_true')
    args = ap.parse_args(argv)
    if args.all:
        pairs = [(iso, t) for iso in ISOS for t in ENDPOINT_TO_THRESHOLD]
        nw = min(len(pairs), os.cpu_count() or 4)
        ctx = multiprocessing.get_context('spawn')
        print(f"[prebuild] {len(pairs)} sidecars with {nw} workers", flush=True)
        with ctx.Pool(nw) as pool:
            pool.map(_prebuild_sidecar, pairs)
        print("[prebuild] done", flush=True)
        niw = min(len(ISOS), os.cpu_count() or 4)
        print(f"[sweep] {len(ISOS)} ISOs with {niw} workers", flush=True)
        with ctx.Pool(niw) as pool:
            pool.map(_run_iso_all, ISOS)
    else:
        if not (args.iso and args.pathway and args.endpoint):
            print('Provide --iso, --pathway, --endpoint OR --all', file=sys.stderr)
            return 2
        _run_one(args.iso, args.pathway, args.endpoint)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
