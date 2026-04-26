#!/usr/bin/env python3
"""Pathway optimizer v3 — refactored for performance and clarity.

Two-stage architecture: Stage-1 loads per-mix worst-hour clean MW from
pre-computed peakclean sidecars (step2_3a_regenerate_peakclean.py);
Stage-2 streams chunked year-by-year argmins over multi-million-row
candidate pools without materializing (Y, N) tensors.
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
_FORESIGHT_PATHWAYS = ('2a', '2b', '3')
BASE_YEAR, END_YEAR = 2025, 2050
YEARS = list(range(BASE_YEAR, END_YEAR + 1))
N_YEARS = len(YEARS)
HOURS_PER_YEAR = 8760
DISCOUNT_RATES = {'5pct': 0.05, '7pct': 0.07, '9pct': 0.09}
WORST_HOUR_PERCENTILE = 99.97
CCGT_OVERNIGHT_CAPEX_USD_KW = 1200.0
NEW_GAS_ASSET_LIFE_YEARS = 25
_FOSSIL_CFS = {'gas': 0.45}
ENDPOINT_TO_THRESHOLD = {0.90: 90, 0.95: 95, 0.99: 99, 0.999: 99.9}
_EP_TAG_MAP = {90: 'ep90', 95: 'ep95', 99: 'ep99', 99.9: 'ep99p9'}
# Ratchet tolerance: 1% of current-year demand (TWh).  Allows the solver
# to swap e.g. 2 pp of solar for wind+battery without tripping the
# per-resource ratchet floor.  The actual enforcement is inside the numba
# kernel (_score_chunk_inner) as  ratch_slack = RATCHET_TOL_PCT * dy.
RATCHET_TOL_PCT = 0.01   # 1% of demand — replaces the old epsilon (1e-6)
SCORE_FLOOR_PCT = 10.0   # defensive hourly_match_score filter band
_POOL_CHUNK = 2_000_000  # streaming chunk size for candidate argmin

EF_DIR = _ROOT / 'data' / 'step2.1-ef'
OUTPUT_BASE = _ROOT / 'data' / 'step2.3-pathway'
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
_NON_CF_IDX = {r: i for i, r in enumerate(_NON_CF)}
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
_STAGE1_VERSION = '4'
_SORTED_BANDS = sorted(_EF_BAND_THRESHOLDS)


def _next_band_ceiling(target_pct: float) -> float:
    """Next EF band threshold strictly above target_pct.

    Used to cap the search window so the optimizer builds incrementally
    through each CFE band instead of jumping to the global cost optimum.
    """
    for b in _SORTED_BANDS:
        if b > target_pct + 0.01:
            return float(b)
    return 100.0


# ─── Module-scope loads (once at import) ─────────────────────────────────────


def _load_gen_profiles():
    t = pq.read_table(_ROOT / 'data' / 'eia-930' / 'eia_generation_profiles.parquet')
    target_year = pacompute.max(t.column('year')).as_py()
    t = t.filter(pacompute.equal(t['year'], target_year))
    out: dict[str, dict[str, np.ndarray]] = {}
    for iso in ISOS:
        t_iso = t.filter(pacompute.equal(t['iso'], iso))
        out[iso] = {}
        if t_iso.num_rows == 0:
            continue
        for fuel in ('solar', 'wind', 'hydro', 'nuclear', 'offshore_wind', 'geothermal'):
            t_fuel = t_iso.filter(pacompute.equal(t_iso['fuel'], fuel))
            if t_fuel.num_rows == 0:
                continue
            hour_arr = t_fuel.column('hour').to_numpy().astype(np.int64)
            val_arr = t_fuel.column('value').to_numpy()
            prof = np.zeros(HOURS_PER_YEAR, dtype=np.float64)
            prof[hour_arr] = val_arr
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
    for iso in ISOS:
        mwh = float(pc.REGIONAL_DEMAND_TWH[iso]) * 1e6
        t_iso = t.filter(pacompute.equal(t['iso'], iso))
        if t_iso.num_rows == 0:
            out[iso] = flat * mwh
            continue
        max_year = pacompute.max(t_iso.column('year')).as_py()
        t_y = t_iso.filter(pacompute.equal(t_iso['year'], max_year))
        hour_arr = t_y.column('hour').to_numpy().astype(np.int64)
        norm_arr = t_y.column('normalized').to_numpy()
        norm = np.zeros(HOURS_PER_YEAR, dtype=np.float64)
        norm[hour_arr] = norm_arr
        norm = norm / norm.sum() if norm.sum() > 0 else flat.copy()
        out[iso] = norm * mwh
    return out

_DEMAND = _load_demand_profiles()


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


def _compute_gas_raw_2025():
    """Compute baseline 2025 gas capacity need per ISO from direct profile computation.

    Builds the current grid mix's hourly supply from generation profiles,
    computes the 99.97th-percentile gap vs RA-adjusted demand, and converts
    to gas capacity via the gas availability factor.
    """
    out: dict[str, float] = {}
    for iso in ISOS:
        mwh = float(pc.REGIONAL_DEMAND_TWH[iso]) * 1e6
        gaf = float(pc.GAS_AVAILABILITY_FACTOR[iso])
        shares = pc.GRID_MIX_SHARES.get(iso, {})
        profiles = _build_clean_profile_table(iso)

        # Compose hourly supply for the current grid mix
        supply = np.zeros(HOURS_PER_YEAR, dtype=np.float64)
        for res, share in shares.items():
            if res in profiles and share > 0:
                supply += (share / 100.0) * profiles[res]

        dfrac = _DEMAND[iso] / max(1.0, mwh)
        resid = np.maximum(0.0, dfrac * (1.0 + pc.RESOURCE_ADEQUACY_MARGIN) - supply)
        out[iso] = float(np.percentile(resid, 99.97)) * mwh / gaf
    return out

_GAS_RAW_2025 = _compute_gas_raw_2025()


# ─── Fossil retirement ──────────────────────────────────────────────────────

def _twh_to_gw(twh: float, cf: float) -> float:
    return twh / (cf * HOURS_PER_YEAR / 1000.0) if cf > 0 else 0.0


def compute_fossil_retirement(iso, clean_pct, demand_growth_factor=1.0, year=None):
    """Simplified: returns only gas_displaced_twh for existing_gas_vec credit."""
    grown_twh = pc.REGIONAL_DEMAND_TWH.get(iso, 0) * demand_growth_factor
    fossil_twh = grown_twh * (100.0 - clean_pct) / 100.0
    baseline_clean = sum(pc.GRID_MIX_SHARES.get(iso, {}).values())
    coal_cap = pc.COAL_CAP_TWH.get(iso, 0)
    oil_cap = pc.OIL_CAP_TWH.get(iso, 0)
    if fossil_twh <= 0.01 or clean_pct >= 100:
        total_f = grown_twh * (100.0 - baseline_clean) / 100.0
        c = min(coal_cap, total_f)
        o = min(oil_cap, max(0, total_f - c))
        g = max(0, total_f - c - o)
        return {'gas_displaced_twh': round(g, 2)}
    add_clean = max(0, (clean_pct - baseline_clean) / 100.0 * grown_twh)
    ct = min(coal_cap, fossil_twh)
    ot = min(oil_cap, max(0, fossil_twh - ct))
    cd = min(add_clean, ct)
    od = min(add_clean - cd, ot)
    gd = max(0.0, add_clean - cd - od)
    return {'gas_displaced_twh': round(gd, 2)}


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


_DEFAULT_CFE_WAYPOINTS = ((2030, 50), (2035, 70), (2040, 90), (2045, 95))


@dataclass(frozen=True)
class RunConfig:
    iso: str; pathway: str; endpoint: float; endpoint_pct: float
    demand_growth_level: str = 'Medium'
    firm_cost_level: str = 'M'; ccs_cost_level: str = 'M'
    tx_level: str = 'Medium'; geo_cost_level: Optional[str] = None
    solver_mode: str = 'myopic'
    endpoint_year_val: Optional[int] = None
    endpoint_target_shares: Optional[tuple] = None
    cfe_waypoints: tuple = _DEFAULT_CFE_WAYPOINTS
    beam_width: int = 1

    @property
    def output_path(self) -> Path:
        mode_tag = '_foresight' if self.solver_mode == 'foresight' else ''
        beam_tag = f'_beam{self.beam_width}' if self.beam_width > 1 else ''
        return OUTPUT_BASE / self.iso / f'pathway{self.pathway}_{_ep_tag(self.endpoint_pct)}{mode_tag}{beam_tag}.json'


# ─── Stage-1: load pre-computed peakclean sidecars ───────────────────────────


def _peakclean_path(iso: str, threshold) -> Path:
    return EF_DIR / f'step_2_1_EF_{iso}_{_threshold_tag(threshold)}_peakclean.parquet'


def _load_peakclean_with_chunks(base_path: Path) -> pa.Table | None:
    """Load a peakclean sidecar that may be split into chunks.

    Checks for {stem}_chunk*.parquet first (chunked output from
    step2_3a_regenerate_peakclean when files exceed GitHub's 100 MB limit).
    Falls back to the monolithic file if no chunks exist.
    Returns None if neither exists.
    """
    chunks = sorted(base_path.parent.glob(base_path.stem + '_chunk*.parquet'))
    if chunks:
        return pa.concat_tables(
            [pq.read_table(c) for c in chunks],
            promote_options='permissive')
    if base_path.exists():
        return pq.read_table(base_path)
    return None


def _load_or_build_peakclean(iso: str, threshold) -> pa.Table:
    """Load a v4+ peakclean sidecar. Raises if missing or outdated.

    All peakclean sidecars must be pre-computed by step2_3a_regenerate_peakclean.py.
    """
    sp = _peakclean_path(iso, threshold)
    tbl = _load_peakclean_with_chunks(sp)
    if tbl is not None:
        meta = tbl.schema.metadata or {}
        sv = meta.get(b'stage1_version', b'').decode()
        if sv >= _STAGE1_VERSION:
            return tbl
        raise RuntimeError(
            f"[FATAL] {iso} t={threshold}: peakclean sidecar is v{sv} "
            f"(need v{_STAGE1_VERSION}+). Run:\n"
            f"  python scripts/step2_3a_regenerate_peakclean.py --iso {iso}\n"
            f"before running the optimizer."
        )
    raise FileNotFoundError(
        f"[FATAL] {iso} t={threshold}: no peakclean sidecar found at {sp}. Run:\n"
        f"  python scripts/step2_3a_regenerate_peakclean.py --iso {iso}"
    )


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


def _load_or_build_interp_peakclean(iso: str, interp_path: Path) -> pa.Table:
    """Load a v4+ peakclean sidecar for an interp file. Raises if missing.

    All interp peakclean sidecars must be pre-computed by:
      python scripts/step2_3a_regenerate_peakclean.py --interp-only
    """
    pc_path = interp_path.with_name(
        interp_path.stem + '_peakclean.parquet')
    tbl = _load_peakclean_with_chunks(pc_path)
    if tbl is not None:
        meta = tbl.schema.metadata or {}
        sv = meta.get(b'stage1_version', b'').decode()
        if sv >= _STAGE1_VERSION:
            return tbl
        raise RuntimeError(
            f"[FATAL] interp sidecar {pc_path.name} is v{sv} "
            f"(need v{_STAGE1_VERSION}+). Run:\n"
            f"  python scripts/step2_3a_regenerate_peakclean.py --interp-only"
        )
    raise FileNotFoundError(
        f"[FATAL] interp peakclean not found: {pc_path.name}. Run:\n"
        f"  python scripts/step2_3a_regenerate_peakclean.py --interp-only"
    )


def load_ef_pool(iso: str) -> dict[str, np.ndarray]:
    """Full-pool EF loader: concatenate all threshold bands + peakclean sidecars.

    Also picks up augmented interp files (step_2_1_EF_{iso}_*_interp_*.parquet)
    produced by diagnose_and_augment_ef_pool.py, building peakclean sidecars
    for them independently.
    """
    ef_tables, pc_tables = [], []

    # ── Pass 1: standard EF bands ──
    for t, paths in _iter_ef_bands(iso):
        for p in paths:
            ef_tables.append(pq.read_table(p))
        pc_tables.append(_load_or_build_peakclean(iso, t))

    # ── Pass 2: interpolated augmentation files ──
    interp_files = sorted(EF_DIR.glob(f'step_2_1_EF_{iso}_*_interp_*.parquet'))
    # Exclude peakclean sidecars from the glob
    interp_files = [p for p in interp_files if '_peakclean' not in p.name]
    for ip in interp_files:
        ef_tables.append(pq.read_table(ip))
        pc_tables.append(_load_or_build_interp_peakclean(iso, ip))
        print(f"[pool] loaded interp file {ip.name} "
              f"({ef_tables[-1].num_rows} rows)", flush=True)

    if not ef_tables:
        raise FileNotFoundError(f"No EF bands for {iso} under {EF_DIR}")
    # 'permissive' unifies string/large_string (and widens numeric types) across
    # bands — some EF parquets were written with string, others with large_string,
    # so the stricter 'default' rejects the concat with an ArrowTypeError.
    ef = pa.concat_tables(ef_tables, promote_options='permissive')
    pctbl = pa.concat_tables(pc_tables, promote_options='permissive')
    n = ef.num_rows
    if pctbl.num_rows != n:
        raise RuntimeError(f"Peakclean rows ({pctbl.num_rows}) != EF rows ({n}) for {iso}")
    arrs = {c: (ef.column(c).to_numpy() if c in ef.column_names else np.zeros(n))
            for c in _RESOURCE_COLS + _STORAGE_COLS + ('hourly_match_score',)}
    # ── FIX: NaN → 0 for resource/storage columns ──────────────────
    # Some interp files omit columns like ccs_ccgt, producing NaN after
    # concat.  Missing resource = zero resource.  Without this, NaN fails
    # the P1 pathway mask (ccs_ccgt <= 0.5 → False for NaN), killing
    # otherwise-valid mixes (observed: NYISO cf=18-19% all rejected).
    for c in _RESOURCE_COLS + _STORAGE_COLS:
        np.nan_to_num(arrs[c], copy=False, nan=0.0)
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
        info = compute_fossil_retirement(
            iso, float(clean_pct_vec[i]), (1 + g) ** i, year)
        cum = max(cum, float(info.get('gas_displaced_twh', 0.0)))
        out[i] = max(0.0, base - _twh_to_gw(cum, _FOSSIL_CFS['gas']) * 1000.0
                     ) * pc.GAS_AVAILABILITY_FACTOR[iso]
    return out


def _cfe_target(year: int, ep_pct: float,
                waypoints: tuple = _DEFAULT_CFE_WAYPOINTS,
                base_pct: float = 0.0) -> float:
    ey = _endpoint_year(ep_pct)
    pts = [(y, t) for y, t in [(BASE_YEAR, base_pct)] + list(waypoints)
           if y < ey] + [(ey, ep_pct)]
    if year <= pts[0][0]:
        return base_pct
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

def _tranche_merit_order(cf_twh, bp_twh, uc, gc, cc, ta, nuke_cost, ccs_cost):
    """Shared tranche merit-order: existing -> uprate -> geo -> {nuke,ccs}.

    Accepts scalars or 1-D arrays for cf_twh/bp_twh. Returns matching shapes.
    Single source of truth — called by _score_chunk, _update_floors,
    and decompose_clean_firm_tranches.
    """
    existing = np.minimum(cf_twh, bp_twh)
    rem = cf_twh - existing
    uprate = np.minimum(rem, uc)
    rem = rem - uprate
    if ta:
        feasible = np.ones_like(cf_twh, dtype=bool)
        geo = np.minimum(rem, gc) if gc > 0 else np.zeros_like(rem)
        rem2 = rem - geo
        if nuke_cost <= ccs_cost or cc <= 0:
            nuke_new, ccs = rem2, np.zeros_like(rem2)
        else:
            ccs = np.minimum(rem2, cc)
            nuke_new = rem2 - ccs
    else:
        feasible = rem <= 1e-9
        geo = nuke_new = ccs = np.zeros_like(rem, dtype=np.float64)
    return existing, uprate, geo, nuke_new, ccs, feasible


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
    Fully vectorized over years — no Python loop."""
    n = len(cf_pct)
    atwh = annual_mwh_vec / 1e6
    bp = float(pc.GRID_MIX_SHARES[iso].get('clean_firm', 0.0))
    target = (cf_pct[None, :] / 100.0) * atwh[:, None]  # (N_YEARS, n)
    # ── FIX 4b: existing nuclear = fixed TWh (base-year capacity) ──
    bl = np.full(N_YEARS, (bp / 100.0) * float(pc.REGIONAL_DEMAND_TWH[iso]))

    nuke_eff, ccs_eff, geo_eff = _tranche_lcoe_vecs(iso, cfg)
    gl = cfg.geo_cost_level or ('M' if iso == 'CAISO' else None)
    ta = _TRANCHES_AVAILABLE.get(pathway, False)
    cc = float(pc.CCS_CAP_TWH.get(iso, 9999.0))
    gc = float(pc.GEOTHERMAL_CAP_TWH) if (iso == 'CAISO' and gl) else 0.0
    uc = float(pc.UPRATE_CAP_TWH[iso])
    up_eff = np.full(N_YEARS, float(pc.UPRATE_LCOE[cfg.firm_cost_level]))

    # Vectorized tranche decomposition (broadcasting bl over n)
    existing = np.minimum(target, bl[:, None])
    rem = target - existing
    uprate_a = np.minimum(rem, uc)
    rem = rem - uprate_a

    if ta:
        feasible = np.ones_like(target, dtype=bool)
        geo_t = np.minimum(rem, gc) if gc > 0 else np.zeros_like(rem)
        rem2 = rem - geo_t
        # Per-year nuke-vs-ccs priority (broadcasts over n)
        nuke_first = (nuke_eff[:, None] <= ccs_eff[:, None]) | (cc <= 0)
        ccs_if_first = np.minimum(rem2, cc)
        nuke_t = np.where(nuke_first, rem2, rem2 - ccs_if_first)
        ccs_t = np.where(nuke_first, 0.0, ccs_if_first)
    else:
        feasible = rem <= 1e-9
        geo_t = nuke_t = ccs_t = np.zeros_like(rem, dtype=np.float64)

    return {
        'existing_twh': existing, 'uprate_twh': uprate_a,
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
    cfe_tgt: np.ndarray; cfe_ceil: np.ndarray; eg_vec: np.ndarray
    gas_raw_2025: float; exist_base: float; gaf: float; fuel_unit: float
    capex_mw: float; fom_mw: float; exist_fom: float
    delivered: np.ndarray  # (Y, R)
    nuke_eff: np.ndarray; ccs_eff: np.ndarray; geo_eff: np.ndarray
    up_eff: np.ndarray; ta: bool; is_p1: bool
    bp: float; uc: float; uc_pct_y: np.ndarray; pf_y: np.ndarray; bl_twh: float
    cc: float; gc: float
    storage_pairs: list; grid: dict; base_dem: float; cfg: RunConfig
    ef_noncf_mat: np.ndarray   # (n, R) pre-stacked non-CF shares / 100
    storage_mat: np.ndarray    # (n, S) pre-stacked storage dispatch shares / 100
    storage_costs: np.ndarray  # (S,) LCOE per storage column
    storage_col_names: tuple   # names of storage columns in storage_mat


def _build_ctx(cfg: RunConfig, ef: dict) -> _Ctx:
    iso = cfg.iso
    n = len(ef['hourly_match_score'])
    pv = peak_demand_vec(iso, cfg.demand_growth_level)
    dv = demand_twh_vec(iso, cfg.demand_growth_level)
    amwh = dv * 1e6
    # Base-year CFE floor: sum of existing clean shares. The CFE target ramp
    # starts here and slopes linearly to the first SBTi waypoint, preventing
    # the optimizer from searching the entire EF with a 0% floor in year 1.
    base_clean_pct = sum(pc.GRID_MIX_SHARES[iso].values())
    cfe_tgt = np.array([_cfe_target(y, cfg.endpoint_pct, cfg.cfe_waypoints,
                                    base_pct=base_clean_pct) for y in YEARS])
    # ── FIX 1: Constant endpoint ceiling ────────────────────────────
    # Ceiling is the next EF band above the ENDPOINT target, constant for
    # all years.  The old per-year ceiling (_next_band_ceiling per cfe_tgt[yi])
    # was too narrow in early years, so the ratchet floor from a prior year
    # would exceed the next year's ceiling → safety-net fired → ceiling dropped
    # entirely → solver jumped to unconstrained global optimum (Bug C-residual).
    _ep_ceil = _next_band_ceiling(cfg.endpoint_pct)
    cfe_ceil = np.full(N_YEARS, _ep_ceil, dtype=np.float64)
    eg = existing_gas_vec(iso, cfe_tgt, cfg.demand_growth_level)
    gl = cfg.geo_cost_level or ('M' if iso == 'CAISO' else None)
    nuke_eff, ccs_eff, geo_eff = _tranche_lcoe_vecs(iso, cfg)
    slm = {'battery_dispatch_pct': 'battery', 'battery8_dispatch_pct': 'battery8',
           'ldes_dispatch_pct': 'ldes', 'h2_dispatch_pct': 'h2'}
    _src = getattr(pc, 'STORAGE_REVENUE_CREDITS', {})
    sp = [(sc, max(0.0, float(pc.LCOE_TABLES[lk]['Medium'][iso])
                   - float(_src.get(lk, {}).get(iso, 0.0))))
          for sc, lk in slm.items() if float(ef[sc].sum()) > 0]
    storage_col_names = tuple(sc for sc, _ in sp)
    uc = float(pc.UPRATE_CAP_TWH[iso])
    # Pre-stack non-CF resource shares into contiguous (n, R) matrix
    ef_noncf_mat = np.column_stack(
        [ef[r].astype(np.float64) for r in _NON_CF]
    ) / 100.0
    # Pre-stack storage dispatch columns and costs
    if sp:
        storage_mat = np.column_stack(
            [ef[sc].astype(np.float64) for sc, _ in sp]
        ) / 100.0
        storage_costs = np.array([c for _, c in sp], dtype=np.float64)
    else:
        storage_mat = np.empty((n, 0), dtype=np.float64)
        storage_costs = np.empty(0, dtype=np.float64)
    ctx = _Ctx(
        ef=ef, n=n, peak_vec=pv, demand_vec=dv, amwh=amwh,
        score=ef['hourly_match_score'],
        resid=ef['resid_norm_p9997'].astype(np.float64),
        cf_share=ef['clean_firm'].astype(np.float64) / 100.0,
        cfe_tgt=cfe_tgt, cfe_ceil=cfe_ceil, eg_vec=eg,
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
        # ── FIX 4: Existing-nuclear baseline is fixed TWh ──────────
        # Existing nuclear capacity doesn't grow with demand.  bl_twh is
        # the constant TWh output (base-year share × base-year demand).
        # pf_y is the per-year clean_firm ceiling in %, declining as demand
        # grows: pf_y[yi] = bl_twh / demand_twh[yi] * 100 + 0.01.
        bl_twh=float(pc.GRID_MIX_SHARES[iso].get('clean_firm', 0.0)) / 100.0
               * float(pc.REGIONAL_DEMAND_TWH[iso]),
        pf_y=(float(pc.GRID_MIX_SHARES[iso].get('clean_firm', 0.0)) / 100.0
              * float(pc.REGIONAL_DEMAND_TWH[iso]))
             / np.maximum(dv, 1e-6) * 100.0 + 0.01,
        cc=float(pc.CCS_CAP_TWH.get(iso, 9999.0)),
        gc=float(pc.GEOTHERMAL_CAP_TWH) if (iso == 'CAISO' and gl) else 0.0,
        storage_pairs=sp, grid=pc.GRID_MIX_SHARES[iso],
        base_dem=float(pc.REGIONAL_DEMAND_TWH[iso]), cfg=cfg,
        ef_noncf_mat=ef_noncf_mat,
        storage_mat=storage_mat, storage_costs=storage_costs,
        storage_col_names=storage_col_names,
    )
    # ── OOM FIX: drop redundant per-column arrays from ef dict ──
    # The individual non-CF and storage columns are now in the pre-stacked
    # matrices (ef_noncf_mat, storage_mat).  Deleting them from the ef dict
    # frees ~1.5 GB on PJM's 26M-row pool, keeping us under 7 GB.
    # Retained: clean_firm, hourly_match_score, clean_peak_hour_mw, resid_norm_p9997.
    _keep = {'clean_firm', 'hourly_match_score', 'clean_peak_hour_mw', 'resid_norm_p9997'}
    for _del_key in list(ef.keys()):
        if _del_key not in _keep:
            del ef[_del_key]
    return ctx


def _init_floors(ctx):
    fr = np.array([ctx.base_dem * ctx.grid.get(r, 0.0) / 100.0 for r in _NON_CF], dtype=np.float64)
    fc = ctx.base_dem * ctx.grid.get('clean_firm', 0.0) / 100.0
    return fr, fc, 0.0, 0.0, 0.0, 0.0


# ─── Chunk scoring kernel ───────────────────────────────────────────────────

@numba.njit(cache=True, parallel=True)
def _score_chunk_inner(
        # Chunk slices (already sliced to [s:e])
        ef_noncf_chunk,     # (k, R) float64 — non-CF shares / 100
        cf_share_chunk,     # (k,) float64 — clean_firm share / 100
        score_chunk,        # (k,) float64 — hourly_match_score
        rmg_chunk,          # (k,) float64 — running max gas (read-only, pre-pass updates)
        storage_chunk,      # (k, S) float64 — storage dispatch / 100
        ef_cf_raw_chunk,    # (k,) float64 — raw clean_firm % (for pathway mask)
        ef_ccs_raw_chunk,   # (k,) float64 — raw ccs_ccgt % (for pathway mask)
        # Per-year scalars
        dy, bl_y,
        nuke_eff_yi, ccs_eff_yi, up_eff_yi, geo_eff_yi,
        cfe_tgt_yi, cfe_ceil_yi, uc_pct_yi,
        # Structural scalars
        uc, gc, cc_cap,
        capex_mw, fom_mw, fuel_unit, exist_fom,
        pf, ratchet_tol_pct,
        # Flags
        ta, is_p1,
        # Floor arrays / scalars
        fr,             # (R,) float64 — non-CF ratchet floors
        fc, fu, fg, fn, fcc,
        # Cost vectors
        delivered_yi,   # (R,) float64
        storage_costs,  # (S,) float64
):
    """Numba-jit scoring kernel.  All Python dispatch eliminated."""
    k = ef_noncf_chunk.shape[0]
    R = ef_noncf_chunk.shape[1]
    S = storage_chunk.shape[1]

    # Output arrays
    rs    = np.empty(k, dtype=numba.float64)
    ratch = np.empty(k, dtype=numba.boolean)
    pm    = np.empty(k, dtype=numba.boolean)
    feas  = np.empty(k, dtype=numba.boolean)
    cfe_s = np.empty(k, dtype=numba.boolean)
    tcf   = np.empty(k, dtype=numba.float64)
    up_out = np.empty(k, dtype=numba.float64)
    ge_out = np.empty(k, dtype=numba.float64)
    nk_out = np.empty(k, dtype=numba.float64)
    cc_out = np.empty(k, dtype=numba.float64)
    tnr   = np.empty((k, R), dtype=numba.float64)

    for i in numba.prange(k):
        # ── Resource shares ──
        for j in range(R):
            tnr[i, j] = ef_noncf_chunk[i, j] * dy

        cf_val = cf_share_chunk[i] * dy
        tcf[i] = cf_val

        # ── Tranche decomposition (inlined _tranche_merit_order) ──
        existing = min(cf_val, bl_y)
        rem = cf_val - existing
        uprate = min(rem, uc)
        rem = rem - uprate

        if ta:
            feas[i] = True
            geo = min(rem, gc) if gc > 0.0 else 0.0
            rem2 = rem - geo
            if nuke_eff_yi <= ccs_eff_yi or cc_cap <= 0.0:
                nuke_new = rem2
                ccs = 0.0
            else:
                ccs = min(rem2, cc_cap)
                nuke_new = rem2 - ccs
        else:
            feas[i] = rem <= 1e-9
            geo = 0.0
            nuke_new = 0.0
            ccs = 0.0

        up_out[i] = uprate
        ge_out[i] = geo
        nk_out[i] = nuke_new
        cc_out[i] = ccs

        # ── FIX 3: Ratchet with demand-proportional tolerance ──
        # Allow each resource to drop by up to 1% of current-year demand
        # (in TWh) to permit resource-substitution (e.g. swap solar for
        # wind+battery) without tripping the ratchet.  Prevents the
        # per-resource floors from starving the solver of viable candidates
        # when the total CFE is actually increasing.
        ratch_slack = ratchet_tol_pct * dy   # 1% of demand in TWh
        ratch_ok = True
        for j in range(R):
            if tnr[i, j] < fr[j] - ratch_slack:
                ratch_ok = False
                break
        if ratch_ok:
            ratch_ok = (cf_val >= fc - ratch_slack
                        and uprate >= fu - ratch_slack
                        and geo >= fg - ratch_slack
                        and nuke_new >= fn - ratch_slack
                        and ccs >= fcc - ratch_slack)
        ratch[i] = ratch_ok

        # ── Pathway mask ──
        if is_p1:
            pm[i] = (ef_cf_raw_chunk[i] <= pf + uc_pct_yi + 0.5
                     and ef_ccs_raw_chunk[i] <= 0.5)
        else:
            pm[i] = True

        # ── CFE threshold (band window) ──
        # Candidates must be within [target, next_band_ceiling) so the
        # optimizer builds incrementally through each CFE band instead of
        # jumping to the global cost optimum at a higher CFE level.
        cfe_s[i] = (score_chunk[i] >= cfe_tgt_yi - 0.5
                     and score_chunk[i] < cfe_ceil_yi)

        # ── Gas sizing (fleet from pre-pass running max) ──
        fl = rmg_chunk[i]

        # ── Costs ──
        gfx = (capex_mw * fl + fom_mw * fl
               + dy * 1e6 * max(0.0, 1.0 - score_chunk[i] / 100.0) * fuel_unit
               + exist_fom)

        # Storage cost (dot product)
        # storage_chunk is capacity fraction (0-1), storage_costs is $/MW-yr.
        # dy is TWh; ×1e6 converts to MWh so units match other cost terms ($).
        sc = 0.0
        for j in range(S):
            sc += storage_chunk[i, j] * storage_costs[j]
        sc *= dy * 1e6

        # Sunk-cost: non-CF incremental
        s_nr = 0.0
        for j in range(R):
            inc = tnr[i, j] - fr[j]
            if inc > 0.0:
                s_nr += inc * delivered_yi[j]
        s_nr *= 1e6

        # Sunk-cost: CF tranches
        s_cf = 0.0
        if uprate > fu:
            s_cf += (uprate - fu) * up_eff_yi
        if geo > fg:
            s_cf += (geo - fg) * geo_eff_yi
        if nuke_new > fn:
            s_cf += (nuke_new - fn) * nuke_eff_yi
        if ccs > fcc:
            s_cf += (ccs - fcc) * ccs_eff_yi
        s_cf *= 1e6

        rs[i] = s_nr + s_cf + sc + gfx

    return rs, ratch, pm, feas, cfe_s, tnr, tcf, up_out, ge_out, nk_out, cc_out


def _score_chunk(s, e, yi, ctx, fr, fc, fu, fg, fn, fcc, rmg,
                 dy=None, bl_y=None, nuke_eff_yi=None,
                 ccs_eff_yi=None, cfe_tgt_yi=None,
                 cfe_ceil_yi=None,
                 uc_pct_yi=None, up_eff_yi=None, geo_eff_yi=None,
                 delivered_yi=None, pf_yi=None):
    """Score chunk [s,e) at year yi. Thin wrapper around numba kernel.
    Pre-computed per-year scalars can be passed to avoid redundant extraction."""
    if dy is None:
        dy = float(ctx.demand_vec[yi])
    if bl_y is None:
        bl_y = ctx.bl_twh
    if nuke_eff_yi is None:
        nuke_eff_yi = float(ctx.nuke_eff[yi])
    if ccs_eff_yi is None:
        ccs_eff_yi = float(ctx.ccs_eff[yi])
    if cfe_tgt_yi is None:
        cfe_tgt_yi = float(ctx.cfe_tgt[yi])
    if cfe_ceil_yi is None:
        cfe_ceil_yi = float(ctx.cfe_ceil[yi])
    if uc_pct_yi is None:
        uc_pct_yi = float(ctx.uc_pct_y[yi])
    if up_eff_yi is None:
        up_eff_yi = float(ctx.up_eff[yi])
    if geo_eff_yi is None:
        geo_eff_yi = float(ctx.geo_eff[yi])
    if delivered_yi is None:
        delivered_yi = ctx.delivered[yi]
    if pf_yi is None:
        pf_yi = float(ctx.pf_y[yi])

    return _score_chunk_inner(
        ctx.ef_noncf_mat[s:e],
        ctx.cf_share[s:e],
        ctx.score[s:e].astype(np.float64),
        rmg[s:e],
        ctx.storage_mat[s:e],
        ctx.ef['clean_firm'][s:e].astype(np.float64),
        ctx.ef_noncf_mat[s:e, _NON_CF_IDX['ccs_ccgt']] * 100.0,
        dy, bl_y,
        nuke_eff_yi, ccs_eff_yi, up_eff_yi, geo_eff_yi,
        cfe_tgt_yi, cfe_ceil_yi, uc_pct_yi,
        ctx.uc, ctx.gc, ctx.cc,
        ctx.capex_mw, ctx.fom_mw, ctx.fuel_unit, ctx.exist_fom,
        pf_yi, RATCHET_TOL_PCT,
        ctx.ta, ctx.is_p1,
        fr, fc, fu, fg, fn, fcc,
        delivered_yi,
        ctx.storage_costs,
    )


def _update_floors(w, yi, ctx, fr, fc, fu, fg, fn, fcc):
    """Update ratchet floors using shared tranche helper."""
    dy = float(ctx.demand_vec[yi])
    wr = ctx.ef_noncf_mat[w].copy() * dy
    wc = float(ctx.cf_share[w]) * dy
    bl_y = ctx.bl_twh

    _, wu, wg, wn, wcc, _ = _tranche_merit_order(
        np.float64(wc), np.float64(bl_y), ctx.uc, ctx.gc, ctx.cc, ctx.ta,
        float(ctx.nuke_eff[yi]), float(ctx.ccs_eff[yi]))

    fr = np.maximum(fr, wr)
    if ctx.cfg.pathway != '1':
        fc = max(fc, wc)
    return fr, fc, max(fu, float(wu)), max(fg, float(wg)), max(fn, float(wn)), max(fcc, float(wcc))


# ─── Beam search helpers ────────────────────────────────────────────────────

from typing import NamedTuple


class BeamEntry(NamedTuple):
    """One trajectory in the beam search."""
    cum_cost: float           # cumulative system cost through current year
    fr: np.ndarray            # (R,) non-CF ratchet floors — always a COPY
    fc: float                 # CF ratchet floor
    fu: float                 # uprate floor
    fg: float                 # geo floor
    fn: float                 # nuke_new floor
    fcc: float                # ccs floor
    history: list             # list[int] — winner mix indices, years 0..yi
    rmg: np.ndarray           # (n,) running-max gas — trajectory-specific COPY


def _beam_adjust_scores(
    rs_ref,          # (k,) kernel scores (computed with ref beam's floors)
    tnr,             # (k, R) non-CF TWh per candidate
    up, ge, nk, cc,  # (k,) CF tranche TWh arrays
    ref_fr, ref_fc, ref_fu, ref_fg, ref_fn, ref_fcc,
    beam_fr, beam_fc, beam_fu, beam_fg, beam_fn, beam_fcc,
    delivered_yi,    # (R,) delivered LCOE vector for this year
    up_eff, geo_eff, nuke_eff, ccs_eff,
):
    """Adjust kernel scores from reference beam's floors to target beam's
    floors. The kernel bakes ref beam's sunk costs into rs_ref. This
    computes the delta to swap in target beam's floors instead.
    Pure numpy, O(k * R).
    """
    # Non-CF sunk cost delta
    delta_nr = np.sum(
        (np.maximum(0.0, tnr - beam_fr[None, :])
         - np.maximum(0.0, tnr - ref_fr[None, :]))
        * delivered_yi[None, :],
        axis=1) * 1e6

    # CF tranche sunk cost delta
    delta_cf = (
        (np.maximum(0.0, up - beam_fu) - np.maximum(0.0, up - ref_fu)) * up_eff
        + (np.maximum(0.0, ge - beam_fg) - np.maximum(0.0, ge - ref_fg)) * geo_eff
        + (np.maximum(0.0, nk - beam_fn) - np.maximum(0.0, nk - ref_fn)) * nuke_eff
        + (np.maximum(0.0, cc - beam_fcc) - np.maximum(0.0, cc - ref_fcc)) * ccs_eff
    ) * 1e6

    return rs_ref + delta_nr + delta_cf


def _beam_ratchet_mask(
    tnr, tcf, up, ge, nk, cc,
    beam_fr, beam_fc, beam_fu, beam_fg, beam_fn, beam_fcc,
    ratch_slack,
):
    """Vectorized ratchet feasibility check for one beam entry.
    Mirrors _score_chunk_inner lines 986-997. Returns bool array (k,).
    """
    ok = np.all(tnr >= beam_fr[None, :] - ratch_slack, axis=1)
    ok &= (tcf >= beam_fc - ratch_slack)
    ok &= (up >= beam_fu - ratch_slack)
    ok &= (ge >= beam_fg - ratch_slack)
    ok &= (nk >= beam_fn - ratch_slack)
    ok &= (cc >= beam_fcc - ratch_slack)
    return ok


# ─── PathwayRunResult ────────────────────────────────────────────────────────

@dataclass
class PathwayRunResult:
    config: RunConfig; ledger: VintageLedger; winners: np.ndarray
    gas_need: np.ndarray; active_fleet: np.ndarray; gas_cf: np.ndarray
    achieved_cfe: np.ndarray; demand_vec: np.ndarray; peak_vec: np.ndarray
    eg_used: np.ndarray; annual_rows: list; end_dispatch: np.ndarray
    feasibility: dict; stranding: dict
    vre_curt: dict; end_mix: dict; end_storage: dict
    rel_tax: dict; headline: dict; ef: dict
    clean_pct: np.ndarray
    ef_noncf_mat: np.ndarray   # (n, R) non-CF shares / 100
    storage_mat: np.ndarray    # (n, S) storage dispatch shares / 100
    storage_col_names: tuple   # column names for storage_mat columns
    foresight_meta: Optional[dict] = None


# ─── Myopic solver ───────────────────────────────────────────────────────────

def solve_pathway(cfg: RunConfig, *, ef_override=None) -> PathwayRunResult:
    """Unified solver: myopic or foresight (2.2A ceiling constraints).

    When cfg.beam_width > 1, maintains K parallel trajectories to avoid
    composition lock from greedy argmin.  beam_width=1 reproduces the
    existing greedy path bit-for-bit.
    """
    ef = ef_override or load_ef_pool(cfg.iso)

    # ── Pre-sort pool by hourly_match_score for binary-search filtering ──
    _sort_idx = np.argsort(ef['hourly_match_score'], kind='mergesort')
    for _k in ef:
        ef[_k] = ef[_k][_sort_idx]
    del _sort_idx

    ctx = _build_ctx(cfg, ef)
    n = ctx.n
    eyi = YEARS.index(_endpoint_year(float(cfg.endpoint_pct)))

    rmg = np.zeros(n, dtype=np.float64)
    fr, fc, fu, fg, fn, fcc = _init_floors(ctx)
    winners = np.empty(N_YEARS, dtype=np.int64)

    # ── Load endpoint ceilings from Step 2.2A (foresight mode) ───
    ceilings = None
    if cfg.solver_mode == 'foresight':
        tgt = np.asarray(cfg.endpoint_target_shares, dtype=np.float64)
        ncf_idx = {r: i for i, r in enumerate(_NON_CF)}
        ceilings = {
            'non_cf_shares': np.array([
                tgt[1 + ncf_idx[r]] if r in ncf_idx else 0.0
                for r in _NON_CF
            ], dtype=np.float64),
            'cf_share': float(tgt[0]),
        }

    # ── Pre-compute gas-sizing constants for vectorized rmg pre-pass ──
    _gaf_safe = max(ctx.gaf, 1e-9)

    # ── Beam search setup ──
    use_beam = cfg.beam_width > 1
    K = cfg.beam_width

    # Saved kernel intermediates for beam reuse (only allocated in beam mode)
    if use_beam:
        saved_tnr = np.zeros((n, _N_NON_CF), dtype=np.float64)
        saved_tcf = np.zeros(n, dtype=np.float64)
        saved_up  = np.zeros(n, dtype=np.float64)
        saved_ge  = np.zeros(n, dtype=np.float64)
        saved_nk  = np.zeros(n, dtype=np.float64)
        saved_cc  = np.zeros(n, dtype=np.float64)
        saved_pm  = np.zeros(n, dtype=bool)
        saved_fe  = np.zeros(n, dtype=bool)
        saved_cs  = np.zeros(n, dtype=bool)

    if use_beam:
        fr0, fc0, fu0, fg0, fn0, fcc0 = _init_floors(ctx)
        beams = [BeamEntry(
            cum_cost=0.0,
            fr=fr0.copy(), fc=fc0, fu=fu0, fg=fg0, fn=fn0, fcc=fcc0,
            history=[],
            rmg=rmg.copy(),
        )]

    for yi in range(eyi + 1):
        best_score, best_idx = np.inf, 0
        # Pre-compute per-year scalars once
        dy = float(ctx.demand_vec[yi])
        amwh_y = float(ctx.amwh[yi])
        bl_y = ctx.bl_twh
        nuke_eff_yi = float(ctx.nuke_eff[yi])
        ccs_eff_yi = float(ctx.ccs_eff[yi])
        eg_yi = float(ctx.eg_vec[yi])
        cfe_tgt_yi = float(ctx.cfe_tgt[yi])
        cfe_ceil_yi = float(ctx.cfe_ceil[yi])
        uc_pct_yi = float(ctx.uc_pct_y[yi])
        pf_yi = float(ctx.pf_y[yi])
        up_eff_yi = float(ctx.up_eff[yi])
        geo_eff_yi = float(ctx.geo_eff[yi])
        delivered_yi = ctx.delivered[yi]

        # ── Vectorized rmg pre-pass (all n candidates, O(n) numpy) ──
        _gr_all = np.maximum(0.0, np.maximum(0.0,
            ctx.exist_base + ctx.resid * amwh_y / _gaf_safe
            - ctx.gas_raw_2025) - eg_yi)
        np.maximum(rmg, _gr_all, out=rmg)

        # ── Binary-search for CFE band window on sorted score array ──
        _win_lo = cfe_tgt_yi - 0.5
        _win_hi = cfe_ceil_yi
        ws = int(np.searchsorted(ctx.score, _win_lo, side='left'))
        we = int(np.searchsorted(ctx.score, _win_hi, side='left'))
        _win_n = we - ws
        if yi == 0 or yi == eyi or yi % 5 == 0:
            print(f"[solve] year {YEARS[yi]}: window [{_win_lo:.1f}, {_win_hi:.1f}) "
                  f"→ {_win_n:,}/{n:,} candidates ({_win_n/max(1,n)*100:.0f}%)",
                  flush=True)

        # Buffer for raw scores
        saved_rs = np.full(n, np.inf, dtype=np.float64)

        # ── Determine reference floors for kernel scoring ──
        if use_beam:
            ref_beam = beams[0]
            ref_fr, ref_fc = ref_beam.fr, ref_beam.fc
            ref_fu, ref_fg = ref_beam.fu, ref_beam.fg
            ref_fn, ref_fcc = ref_beam.fn, ref_beam.fcc
        else:
            ref_fr, ref_fc = fr, fc
            ref_fu, ref_fg = fu, fg
            ref_fn, ref_fcc = fn, fcc

        # ── Primary path: score only the window slice [ws, we) ──
        for s in range(ws, max(ws, we), _POOL_CHUNK):
            e = min(s + _POOL_CHUNK, we)
            rs, ra, pm, fe, cs, tnr, tcf, up, ge, nk, cc = _score_chunk(
                s, e, yi, ctx, ref_fr, ref_fc, ref_fu, ref_fg, ref_fn, ref_fcc, rmg,
                dy=dy, bl_y=bl_y,
                nuke_eff_yi=nuke_eff_yi, ccs_eff_yi=ccs_eff_yi,
                cfe_tgt_yi=cfe_tgt_yi,
                cfe_ceil_yi=cfe_ceil_yi,
                uc_pct_yi=uc_pct_yi, up_eff_yi=up_eff_yi,
                geo_eff_yi=geo_eff_yi, delivered_yi=delivered_yi,
                pf_yi=pf_yi)
            saved_rs[s:e]  = rs
            if use_beam:
                saved_tnr[s:e] = tnr
                saved_tcf[s:e] = tcf
                saved_up[s:e]  = up
                saved_ge[s:e]  = ge
                saved_nk[s:e]  = nk
                saved_cc[s:e]  = cc
                saved_pm[s:e]  = pm
                saved_fe[s:e]  = fe
                saved_cs[s:e]  = cs

            if not use_beam:
                # ── GREEDY: inline best-finding (existing logic) ──
                mask = ra & pm & cs & fe

                # Ceiling constraint: 2.2A endpoint shares with transition headroom
                if ceilings is not None:
                    progress = yi / max(1, eyi)
                    headroom = 1.0 - progress
                    noncf_shares = ctx.ef_noncf_mat[s:e]
                    ceil_ok = np.all(
                        noncf_shares <= ceilings['non_cf_shares'][None, :] * (1.0 + headroom) + RATCHET_TOL_PCT,
                        axis=1)
                    ceil_ok &= (ctx.cf_share[s:e] <= ceilings['cf_share'] * (1.0 + headroom) + RATCHET_TOL_PCT)
                    mask = mask & ceil_ok

                v = np.where(mask, rs, np.inf)
                li = int(np.argmin(v))
                if float(v[li]) < best_score:
                    best_score, best_idx = float(v[li]), s + li

        # ── BEAM PATH: multi-trajectory selection ──
        if use_beam:
            indep_mask = saved_pm[ws:we] & saved_fe[ws:we] & saved_cs[ws:we]

            # Window slices for beam helpers
            w_rs  = saved_rs[ws:we]
            w_tnr = saved_tnr[ws:we]
            w_tcf = saved_tcf[ws:we]
            w_up  = saved_up[ws:we]
            w_ge  = saved_ge[ws:we]
            w_nk  = saved_nk[ws:we]
            w_cc  = saved_cc[ws:we]
            w_n   = we - ws
            ratch_slack = RATCHET_TOL_PCT * dy

            if w_n == 0:
                # Empty window — all beams carry forward unchanged
                if yi == 0:
                    raise RuntimeError(
                        f"[FATAL] year {YEARS[yi]}: empty window, no beams can start")
                # Log diagnostic
                print(f"[beam] year {YEARS[yi]}: empty window — "
                      f"{len(beams)} beams carry forward", flush=True)
            else:
                new_beams = []

                for beam in beams:
                    # Adjust scores from ref beam's floors to this beam's
                    if beam is beams[0]:
                        adj = w_rs.copy()
                    else:
                        adj = _beam_adjust_scores(
                            w_rs, w_tnr, w_up, w_ge, w_nk, w_cc,
                            ref_fr, ref_fc, ref_fu, ref_fg, ref_fn, ref_fcc,
                            beam.fr, beam.fc, beam.fu, beam.fg,
                            beam.fn, beam.fcc,
                            delivered_yi,
                            up_eff_yi, geo_eff_yi, nuke_eff_yi, ccs_eff_yi,
                        )

                    # Ratchet mask for this beam
                    ratch_k = _beam_ratchet_mask(
                        w_tnr, w_tcf, w_up, w_ge, w_nk, w_cc,
                        beam.fr, beam.fc, beam.fu, beam.fg,
                        beam.fn, beam.fcc,
                        ratch_slack,
                    )

                    full_mask = indep_mask & ratch_k

                    # Foresight ceilings (if present)
                    if ceilings is not None:
                        progress = yi / max(1, eyi)
                        headroom = 1.0 - progress
                        snr_w = w_tnr / max(dy, 1e-9)
                        ceil_ok = np.all(
                            snr_w <= ceilings['non_cf_shares'][None, :]
                            * (1.0 + headroom) + RATCHET_TOL_PCT, axis=1)
                        ceil_ok &= (ctx.cf_share[ws:we]
                                    <= ceilings['cf_share']
                                    * (1.0 + headroom) + RATCHET_TOL_PCT)
                        full_mask &= ceil_ok

                    masked = np.where(full_mask, adj, np.inf)

                    if np.all(np.isinf(masked)):
                        # Beam stalls — carry forward unchanged
                        if beam.history:
                            new_beams.append(beam)
                        continue

                    # Pick this beam's best candidate
                    best_local = int(np.argmin(masked))
                    best_pool_idx = best_local + ws
                    year_cost = float(adj[best_local])

                    # Update floors for this trajectory
                    new_fr, new_fc, new_fu, new_fg, new_fn, new_fcc = \
                        _update_floors(best_pool_idx, yi, ctx,
                                       beam.fr.copy(), beam.fc, beam.fu,
                                       beam.fg, beam.fn, beam.fcc)

                    new_rmg = beam.rmg.copy()
                    np.maximum(new_rmg, _gr_all, out=new_rmg)

                    new_beams.append(BeamEntry(
                        cum_cost=beam.cum_cost + year_cost,
                        fr=new_fr, fc=new_fc, fu=new_fu, fg=new_fg,
                        fn=new_fn, fcc=new_fcc,
                        history=beam.history + [best_pool_idx],
                        rmg=new_rmg,
                    ))

                # ── Seed new beams from top-K candidates (fills beam to K) ──
                if len(new_beams) < K:
                    ref_ratch = _beam_ratchet_mask(
                        w_tnr, w_tcf, w_up, w_ge, w_nk, w_cc,
                        ref_fr, ref_fc, ref_fu, ref_fg, ref_fn, ref_fcc,
                        ratch_slack)
                    ref_masked = np.where(indep_mask & ref_ratch, w_rs, np.inf)

                    n_need = min(K - len(new_beams), w_n)
                    if n_need > 0 and not np.all(np.isinf(ref_masked)):
                        finite_count = int(np.isfinite(ref_masked).sum())
                        n_part = min(n_need, finite_count)
                        if n_part > 0:
                            top_k_local = np.argpartition(
                                ref_masked, n_part)[:n_part]
                            top_k_local = top_k_local[
                                np.isfinite(ref_masked[top_k_local])]
                            # Sort by cost so cheapest seeds enter first
                            top_k_local = top_k_local[
                                np.argsort(ref_masked[top_k_local])]

                            existing_winners = {
                                b.history[-1] for b in new_beams if b.history}

                            for li in top_k_local:
                                pool_idx = li + ws
                                if pool_idx in existing_winners:
                                    continue
                                if len(new_beams) >= K:
                                    break

                                year_cost = float(w_rs[li])
                                s_fr, s_fc, s_fu, s_fg, s_fn, s_fcc = \
                                    _update_floors(pool_idx, yi, ctx,
                                                   ref_fr.copy(), ref_fc,
                                                   ref_fu, ref_fg, ref_fn,
                                                   ref_fcc)
                                new_beams.append(BeamEntry(
                                    cum_cost=year_cost,
                                    fr=s_fr, fc=s_fc, fu=s_fu, fg=s_fg,
                                    fn=s_fn, fcc=s_fcc,
                                    history=[pool_idx],
                                    rmg=rmg.copy(),
                                ))
                                existing_winners.add(pool_idx)

                # ── Prune to top K by cumulative cost ──
                if len(new_beams) > K:
                    new_beams.sort(key=lambda b: b.cum_cost)
                    new_beams = new_beams[:K]

                beams = new_beams

                if not beams:
                    raise RuntimeError(
                        f"[FATAL] year {YEARS[yi]}: all beam entries died")

            if yi == 0 or yi == eyi or yi % 5 == 0:
                costs = [b.cum_cost for b in beams]
                print(f"[beam] year {YEARS[yi]}: {len(beams)} beams, "
                      f"cost [{min(costs):.0f} .. {max(costs):.0f}]",
                      flush=True)

        else:
            # ── GREEDY FALLBACK: hold previous year's winner ──
            if np.isinf(best_score):
                # ── Diagnostic: identify which constraints killed all candidates ──
                _diag_n = we - ws
                if _diag_n > 0:
                    _d_rs, _d_ra, _d_pm, _d_fe, _d_cs, _, _, _, _, _, _ = _score_chunk(
                        ws, we, yi, ctx, fr, fc, fu, fg, fn, fcc, rmg,
                        dy=dy, bl_y=bl_y,
                        nuke_eff_yi=nuke_eff_yi, ccs_eff_yi=ccs_eff_yi,
                        cfe_tgt_yi=cfe_tgt_yi,
                        cfe_ceil_yi=cfe_ceil_yi,
                        uc_pct_yi=uc_pct_yi, up_eff_yi=up_eff_yi,
                        geo_eff_yi=geo_eff_yi, delivered_yi=delivered_yi,
                        pf_yi=pf_yi)
                    _n_ratch_fail = int((~_d_ra).sum())
                    _n_path_fail = int((~_d_pm).sum())
                    _n_feas_fail = int((~_d_fe).sum())
                    _n_cfe_fail = int((~_d_cs).sum())
                    _n_ceil_fail = 0
                    if ceilings is not None:
                        progress = yi / max(1, eyi)
                        headroom = 1.0 - progress
                        _d_noncf = ctx.ef_noncf_mat[ws:we]
                        _ceil_ok = np.all(
                            _d_noncf <= ceilings['non_cf_shares'][None, :] * (1.0 + headroom) + RATCHET_TOL_PCT,
                            axis=1)
                        _ceil_ok &= (ctx.cf_share[ws:we] <= ceilings['cf_share'] * (1.0 + headroom) + RATCHET_TOL_PCT)
                        _n_ceil_fail = int((~_ceil_ok).sum())
                    print(f"[DIAG] year {YEARS[yi]}: {_diag_n} candidates in window, "
                          f"all failed — ratchet:{_n_ratch_fail} pathway:{_n_path_fail} "
                          f"tranche:{_n_feas_fail} cfe_band:{_n_cfe_fail} "
                          f"ceiling:{_n_ceil_fail}", flush=True)
                else:
                    print(f"[DIAG] year {YEARS[yi]}: empty window "
                          f"[{cfe_tgt_yi - 0.5:.1f}, {cfe_ceil_yi:.1f}) — "
                          f"0 candidates in CFE band", flush=True)

                if yi > 0:
                    best_idx = int(winners[yi - 1])
                    best_score = float(saved_rs[best_idx]) if np.isfinite(saved_rs[best_idx]) else 0.0
                    _prev_cfe = float(ctx.score[best_idx])
                    print(f"[HOLD] year {YEARS[yi]}: holding winner from "
                          f"{YEARS[yi - 1]} (mix {best_idx}, CFE {_prev_cfe:.1f}%)",
                          flush=True)
                else:
                    best_idx = int(np.argmin(saved_rs))
                    best_score = float(saved_rs[best_idx])
                    print(f"[WARN] year {YEARS[yi]}: no feasible candidate "
                          f"at CFE>={cfe_tgt_yi:.1f}%, using lowest-cost mix "
                          f"(idx {best_idx}, score {best_score:.2f})",
                          flush=True)

            winners[yi] = best_idx
            fr, fc, fu, fg, fn, fcc = _update_floors(
                best_idx, yi, ctx, fr, fc, fu, fg, fn, fcc)

    # ── Post-solve: beam trajectory selection or greedy freeze ──
    if use_beam:
        best_beam = min(beams, key=lambda b: b.cum_cost)
        winners = np.array(best_beam.history, dtype=np.int64)

        # Pad with freeze-forward to N_YEARS
        if len(winners) < N_YEARS:
            pad = np.full(N_YEARS - len(winners),
                          winners[-1], dtype=np.int64)
            winners = np.concatenate([winners, pad])

        # Restore state from winning beam for _finalize
        fr, fc = best_beam.fr, best_beam.fc
        fu, fg = best_beam.fu, best_beam.fg
        fn, fcc = best_beam.fn, best_beam.fcc
        rmg = best_beam.rmg

        # ── Beam diagnostics ──
        print(f"[beam] selected beam: cum_cost={best_beam.cum_cost:.2f}, "
              f"{len(beams)} survivors", flush=True)
        if len(best_beam.history) > 0:
            # Report final CFE trajectory
            beam_cfe = ctx.score[np.array(best_beam.history[:eyi + 1], dtype=np.int64)]
            print(f"[beam] endpoint CFE: {beam_cfe[-1]:.1f}%, "
                  f"min CFE: {beam_cfe.min():.1f}%", flush=True)
    else:
        # Post-endpoint freeze (greedy path)
        winners[eyi + 1:] = int(winners[eyi])

    return _finalize(ctx, winners)


# ─── Target shares (Step 2.2A) ───────────────────────────────────────────────

_TS_CACHE: dict = {}


def _get_target_shares(iso, ep_pct, dgl='Medium', gl=None, fl='M', cl='M', tl='Medium'):
    """Load cost-optimal endpoint resource shares from Step 2.2A."""
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
    tgt = np.zeros(len(_RESOURCE_COLS), dtype=np.float64)
    for i, r in enumerate(_RESOURCE_COLS):
        tgt[i] = float(row.get(f'mix_{r}', 0.0) or 0.0) / 100.0
    _TS_CACHE[ck] = tgt
    return tgt


# ─── Output helpers ──────────────────────────────────────────────────────────

def _vre_curtailment(iso, mix_pct):
    """Estimate VRE curtailment from direct profile computation.

    Computes total supply, identifies surplus above demand, and attributes
    curtailment proportionally to each VRE resource based on its share of
    total generation.
    """
    keys = ('solar', 'wind', 'offshore_wind', 'solar_batt4', 'solar_batt8',
            'wind_batt4', 'wind_batt8')
    out = {k: 0.0 for k in keys}
    profiles = _build_clean_profile_table(iso)
    demand_norm = _DEMAND[iso] / max(1.0, float(pc.REGIONAL_DEMAND_TWH[iso]) * 1e6)

    total_supply = np.zeros(HOURS_PER_YEAR, dtype=np.float64)
    for res, share in mix_pct.items():
        if res in profiles and share > 0:
            total_supply += (share / 100.0) * profiles[res]

    surplus = np.maximum(0.0, total_supply - demand_norm)
    total_surplus = surplus.sum()
    total_supply_sum = total_supply.sum()

    for k in keys:
        share = mix_pct.get(k, 0.0)
        if share <= 0 or total_supply_sum <= 0:
            continue
        res_total = ((share / 100.0) * profiles.get(k, np.zeros(HOURS_PER_YEAR))).sum()
        if res_total > 0:
            res_surplus = total_surplus * (res_total / total_supply_sum)
            out[k] = round(res_surplus / res_total, 4)
    return out


def _endpoint_dispatch(iso, mix_pct):
    """Compute 8760 total clean supply profile for the endpoint mix.

    Direct profile computation — no archetype lookup needed.
    """
    profiles = _build_clean_profile_table(iso)
    supply = np.zeros(HOURS_PER_YEAR, dtype=np.float64)
    for res, share in mix_pct.items():
        if res in profiles and share > 0:
            supply += (share / 100.0) * profiles[res]
    return supply


# ─── Ledger builder ──────────────────────────────────────────────────────────

def _build_ledger(ef, winners, cfg, ef_noncf_mat):
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
        for ri, r in enumerate(_NON_CF):
            tgt = dy * float(ef_noncf_mat[w, ri])  # already fraction (0-1)
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

def _finalize(ctx, winners):
    cfg = ctx.cfg
    acfe = ctx.score[winners].copy()

    # ── BUG 2 FIX: P1/P1a CFE correction ──────────────────────────────
    # The pool's hourly_match_score assumes the full clean_firm share is
    # deployed.  Under P1/P1a, tranches_available=False caps actual CF
    # delivery at existing + uprate.  Subtract the excess CF percentage
    # that can't physically be delivered.  Nuclear is baseload, so each
    # percentage point of excess CF ≈ one percentage point of CFE.
    if ctx.is_p1:
        cf_pct_winners = ctx.ef['clean_firm'][winners].astype(np.float64)
        cap_pct = ctx.pf_y + ctx.uc_pct_y  # existing ceiling + uprate cap (%)
        excess = np.maximum(0.0, cf_pct_winners - cap_pct)
        acfe = np.maximum(0.0, acfe - excess)

    eg_real = existing_gas_vec(cfg.iso, acfe, cfg.demand_growth_level)

    rw = ctx.resid[winners]
    gap = rw * ctx.amwh / max(1e-9, ctx.gaf)
    gn = np.maximum(0.0, ctx.exist_base + gap - ctx.gas_raw_2025)
    cng = np.maximum(0.0, gn - eg_real)
    af = np.maximum.accumulate(cng)

    # ── Per-year gas vintage tracking (Card 12) ──────────────────────
    gas_vintages = []
    prev_af = 0.0
    for yi in range(N_YEARS):
        cur_af = float(af[yi])
        delta_mw = cur_af - prev_af
        if delta_mw > 0.01:
            gas_vintages.append({'year': YEARS[yi], 'mw': delta_mw})
        prev_af = cur_af

    fm = float(af.max())
    g2050 = float(cng[-1])
    smw_total = max(0.0, fm - g2050)

    per_vintage_stranding = []
    total_stranded_capex = 0.0
    for vint in gas_vintages:
        vy, vmw = vint['year'], vint['mw']
        age_at_2050 = END_YEAR - vy
        remaining_life = max(0.0, NEW_GAS_ASSET_LIFE_YEARS - age_at_2050)
        vint_stranded_mw = max(0.0, vmw * (smw_total / fm)) if fm > 0 else 0.0
        vint_capex = (vint_stranded_mw * 1000.0 * CCGT_OVERNIGHT_CAPEX_USD_KW
                      * (remaining_life / NEW_GAS_ASSET_LIFE_YEARS))
        total_stranded_capex += vint_capex
        per_vintage_stranding.append({
            'year_built': vy, 'initial_mw': round(vmw, 2),
            'stranded_mw': round(vint_stranded_mw, 2),
            'age_at_2050': age_at_2050,
            'remaining_asset_life_years': round(remaining_life, 1),
            'stranded_capex_usd': round(vint_capex, 2),
        })

    peak_year = int(YEARS[int(np.argmax(af))]) if fm > 0 else BASE_YEAR

    tgmw = ctx.exist_base + af
    gg = np.maximum(0.0, 1.0 - acfe / 100.0) * ctx.demand_vec * 1e6
    gcf = np.where(tgmw > 1e-6, np.minimum(1.0, gg / (tgmw * HOURS_PER_YEAR)), 0.0)
    egu = np.minimum(eg_real, gn)

    ledger = _build_ledger(ctx.ef, winners, cfg, ctx.ef_noncf_mat)
    gop = _gross_op_vec(ledger)

    ga = af * ctx.capex_mw
    gf = af * ctx.fom_mw
    gfu = np.maximum(0.0, 1.0 - acfe / 100.0) * ctx.demand_vec * 1e6 * ctx.fuel_unit

    rows = []
    for yi, year in enumerate(YEARS):
        g = float(gop[yi])
        gaf_val = float(ga[yi] + gf[yi])
        net = g + gaf_val + ctx.exist_fom + gfu[yi]
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
        })

    ew = int(winners[-1])
    # ── OOM FIX: build end_mix and end_storage from matrices ──
    em = {'clean_firm': round(float(ctx.ef['clean_firm'][ew]), 3)}
    for i, r in enumerate(_NON_CF):
        em[r] = round(float(ctx.ef_noncf_mat[ew, i] * 100.0), 3)
    es = {}
    for j, sc in enumerate(ctx.storage_col_names):
        es[sc] = round(float(ctx.storage_mat[ew, j] * 100.0), 4)
    for sc in _STORAGE_COLS:
        if sc not in es:
            es[sc] = 0.0
    vc = _vre_curtailment(cfg.iso, em)
    ed = _endpoint_dispatch(cfg.iso, em)

    rt_comp = {
        'new_gas_capex_annualized_usd': round(float(ga.sum()), 2),
        'new_gas_fom_usd': round(float(gf.sum()), 2),
        'existing_gas_fom_carried_usd': round(ctx.exist_fom * N_YEARS, 2),
        'vre_storage_overbuild_capex_usd': 0.0,
    }
    td = float((ctx.demand_vec * 1e6).sum())
    rt_total = sum(rt_comp.values())

    undisc = sum(r['net_annual_cost_usd'] for r in rows)
    npvs = {f'npv_at_{k}': round(sum(r['net_annual_cost_usd'] / (1 + rate) ** (r['year'] - BASE_YEAR)
                                     for r in rows), 2) for k, rate in DISCOUNT_RATES.items()}

    return PathwayRunResult(
        config=cfg, ledger=ledger, winners=winners, gas_need=cng,
        active_fleet=af, gas_cf=gcf, achieved_cfe=acfe,
        demand_vec=ctx.demand_vec, peak_vec=ctx.peak_vec, eg_used=egu,
        annual_rows=rows, end_dispatch=ed,
        feasibility={'physical': True, 'notes': []},
        stranding={
            'methodology': 'per_year_vintage_v3',
            'asset_life_years': NEW_GAS_ASSET_LIFE_YEARS,
            'overnight_capex_usd_kw': CCGT_OVERNIGHT_CAPEX_USD_KW,
            'fleet_size_mw': round(fm, 2), 'peak_year': peak_year,
            'new_gas_need_2050_mw': round(g2050, 2),
            'stranded_mw_at_2050': round(smw_total, 2),
            'total_new_gas_built_mw': round(fm, 2),
            'total_stranded_capex_usd': round(total_stranded_capex, 2),
            'stranded_vintage_count': len([v for v in per_vintage_stranding if v['stranded_mw'] > 0]),
            'per_vintage': per_vintage_stranding,
        },
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
        ef=ctx.ef, clean_pct=acfe,
        ef_noncf_mat=ctx.ef_noncf_mat,
        storage_mat=ctx.storage_mat,
        storage_col_names=ctx.storage_col_names,
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
        for ri, res in enumerate(_NON_CF):
            tgt = dy * float(r.ef_noncf_mat[w, ri])  # already fraction (0-1)
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
        growth = float(r.peak_vec[yi]) / float(pc.PEAK_DEMAND_MW[cfg.iso])
        scaled_clean_peak = float(ef['clean_peak_hour_mw'][w]) * growth
        prev_af = float(r.active_fleet[yi - 1]) if yi > 0 else 0.0
        delta_af = max(0.0, float(r.active_fleet[yi]) - prev_af)
        out.append({
            'year': year, 'new_vintages': nv,
            'gas_sizing': {
                'peak_demand_mw': round(float(r.peak_vec[yi]), 2),
                'ra_peak_mw': round(float(r.peak_vec[yi] * (1 + ra)), 2),
                'total_clean_peak_mw': round(scaled_clean_peak, 2),
                'existing_gas_used_mw': round(float(r.eg_used[yi]), 2),
                'new_gas_instantaneous_need_mw': round(float(r.gas_need[yi]), 2),
                'new_gas_built_this_year_mw': round(delta_af, 2),
                'active_new_gas_fleet_mw': round(float(r.active_fleet[yi]), 2),
                'gas_fleet_cf': round(float(r.gas_cf[yi]), 4),
            },
        })
    return out


def serialize_v2(r: PathwayRunResult) -> dict:
    cfg = r.config
    return {
        'schema_version': 2,
        'run_key': f'{cfg.iso}__pathway{cfg.pathway}__{_ep_tag(cfg.endpoint_pct)}{"__foresight" if cfg.solver_mode == "foresight" else ""}',
        'config': {
            'iso': cfg.iso, 'pathway': cfg.pathway,
            'endpoint': cfg.endpoint, 'endpoint_pct': cfg.endpoint_pct,
            'demand_growth_level': cfg.demand_growth_level,
            'firm_cost_level': cfg.firm_cost_level,
            'ccs_cost_level': cfg.ccs_cost_level,
            'tx_level': cfg.tx_level,
            'geo_cost_level': cfg.geo_cost_level,
            'solver_mode': cfg.solver_mode,
        },
        'feasibility': r.feasibility,
        'headline': r.headline,
        'tables': {
            'annual_buildout': _serialize_buildout(r),
            'annual_cost': r.annual_rows,
            'endpoint_hourly_dispatch': [round(float(x), 8) for x in r.end_dispatch],
            'new_gas_fleet': r.stranding.get('per_vintage', []),
        },
        'reliability_tax': r.rel_tax,
        'stranding_metadata': r.stranding,
        'vre_curtailment_at_endpoint': r.vre_curt,
        'endpoint_mix_pct': r.end_mix,
        'endpoint_storage_pct': r.end_storage,
        'terminal_ledger': r.ledger.serialize(),
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _run_one(iso, pathway, endpoint, solver_mode='myopic', ey=None, ets=None,
             beam_width=1):
    ep_pct = endpoint * 100
    if solver_mode == 'foresight':
        if ey is None:
            ey = _endpoint_year(ep_pct)
        if ets is None:
            ets = tuple(float(x) for x in _get_target_shares(iso, ep_pct))
    cfg = RunConfig(
        iso=iso, pathway=pathway, endpoint=endpoint, endpoint_pct=ep_pct,
        solver_mode=solver_mode,
        endpoint_year_val=ey, endpoint_target_shares=ets,
        beam_width=beam_width,
    )
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


def _run_iso_all(iso, beam_width=1):
    for p in PATHWAYS:
        for ep in ENDPOINT_TO_THRESHOLD:
            _run_one(iso, p, ep, beam_width=beam_width)
            if p in _FORESIGHT_PATHWAYS:
                _run_one(iso, p, ep, solver_mode='foresight',
                         beam_width=beam_width)
    _lcoe.cache_clear(); _tx.cache_clear()
    gc.collect()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Pathway optimizer v3')
    ap.add_argument('--iso', choices=ISOS)
    ap.add_argument('--pathway', choices=PATHWAYS)
    ap.add_argument('--endpoint', type=float, choices=list(ENDPOINT_TO_THRESHOLD))
    ap.add_argument('--beam-width', type=int, default=1,
                    help='Beam search width (1=greedy, default). Higher values '
                         'maintain K parallel trajectories to avoid composition lock.')
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
        _run_one(args.iso, args.pathway, args.endpoint,
                 beam_width=args.beam_width)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

