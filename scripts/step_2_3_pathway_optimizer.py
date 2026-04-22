#!/usr/bin/env python3
"""Pathway optimizer — v2 (frozen schema per reliability_tax/PATHWAY_OUTPUT_SCHEMA.md).

Two-stage architecture: a one-time ISO-level Stage-1 sidecar precomputes the
worst-hour clean MW per unique mix; the per-year Stage-2 hot loop is one
broadcast op producing the (26, n_mixes) gas-need matrix. Eliminates the
per-mix-per-year dispatch-cache call that dominated v1 wall-clock.
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

import numpy as np
import pyarrow as pa
import pyarrow.compute as pacompute
import pyarrow.parquet as pq

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
import pipeline_config as pc

ISOS = list(pc.ISOS)

# Memory-constrained runner escape hatch: if ISO_FILTER is set, restrict the
# module-scope loaders (dispatch cache, gen profiles, hybrid profiles, fossil
# mix) to that single ISO. Unset locally → full 7-ISO load as before.
_ISO_FILTER = os.environ.get('ISO_FILTER', '').strip().upper()
if _ISO_FILTER:
    if _ISO_FILTER not in ISOS:
        raise SystemExit(
            f"ISO_FILTER={_ISO_FILTER!r} not in known ISOs {ISOS}"
        )
    ISOS = [_ISO_FILTER]
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
    0.90: 90, 0.95: 95, 0.99: 99, 0.999: 99.9,
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


def _load_demand_profiles_by_iso() -> dict[str, np.ndarray]:
    """Hourly ISO demand (MW), scaled so annual sum = REGIONAL_DEMAND_TWH * 1e6.

    Reads `data/eia-930/eia_demand_profiles.parquet` directly with pyarrow
    (schema: iso, year, hour, raw_mw, normalized). Uses the most recent
    year per ISO and renormalizes to target annual MWh.
    """
    path = _ROOT / 'data' / 'eia-930' / 'eia_demand_profiles.parquet'
    flat = np.full(HOURS_PER_YEAR, 1.0 / HOURS_PER_YEAR)
    out: dict[str, np.ndarray] = {}
    if not path.exists():
        for iso in ISOS:
            out[iso] = flat * (float(pc.REGIONAL_DEMAND_TWH[iso]) * 1.0e6)
        return out
    t = pq.read_table(path, columns=['iso', 'year', 'hour', 'normalized'])
    iso_arr = np.asarray(t.column('iso').to_pylist())
    year_arr = t.column('year').to_numpy()
    hour_arr = t.column('hour').to_numpy()
    norm_arr = t.column('normalized').to_numpy()
    for iso in ISOS:
        mask = iso_arr == iso
        if not mask.any():
            out[iso] = flat * (float(pc.REGIONAL_DEMAND_TWH[iso]) * 1.0e6)
            continue
        target_year = int(year_arr[mask].max())
        m2 = mask & (year_arr == target_year)
        norm = np.zeros(HOURS_PER_YEAR, dtype=np.float64)
        norm[hour_arr[m2].astype(np.int64)] = norm_arr[m2]
        if norm.sum() <= 0:
            norm = flat.copy()
        else:
            norm = norm / norm.sum()
        out[iso] = norm * (float(pc.REGIONAL_DEMAND_TWH[iso]) * 1.0e6)
    return out


_DEMAND_PROFILE_BY_ISO = _load_demand_profiles_by_iso()

# Stage-1 archetype-miss tracker: (iso, threshold) -> list[row_idx].
# Stage-1 runs once per (iso, threshold), not per-year — the N_YEARS=26
# per-year invariant is untouched.
_STAGE1_MISSES: dict[tuple, list] = {}


def _compute_gas_raw_2025_by_iso() -> dict[str, float]:
    """2025-baseline gas_raw per ISO under the 99.97 margin-on-demand rule.

    Mirrors `_gas_raw_2025_worst_hour` in step2_2a_cost_optimization.py:435
    without importing from it. Uses the existing-grid mix from
    `GRID_MIX_SHARES[iso]` (no storage) matched to its closest cached
    archetype via manifest L1-NN. `gas_raw = resid_p9997 * annual_mwh /
    GAS_AVAILABILITY_FACTOR`. Constant per ISO — precomputed at import.
    """
    out: dict[str, float] = {}
    mcols = ['mix_clean_firm', 'mix_solar', 'mix_wind', 'mix_offshore_wind',
             'mix_ccs_ccgt', 'mix_hydro']
    for iso in ISOS:
        cache = _DISPATCH_CACHE_BY_ISO.get(iso)
        am = _ANNUAL_MANIFEST_BY_ISO.get(iso)
        annual_mwh = float(pc.REGIONAL_DEMAND_TWH[iso]) * 1.0e6
        gaf = float(pc.GAS_AVAILABILITY_FACTOR[iso])
        if cache is None or am is None:
            # Conservative: assume the full RA margin is a gap at 2025.
            out[iso] = float(pc.RESOURCE_ADEQUACY_MARGIN) * annual_mwh / gaf
            continue
        shares = pc.GRID_MIX_SHARES.get(iso, {})
        target = np.array([float(shares.get(c.replace('mix_', ''), 0.0))
                           for c in mcols], dtype=np.float64)
        am_mat = np.column_stack(
            [am.column(c).to_numpy() for c in mcols]).astype(np.float64)
        nn = int(np.argmin(np.abs(am_mat - target).sum(axis=1)))
        arch_key = am.column('archetype_key')[nn].as_py()
        cache_keys = cache.column('archetype_key').to_pylist()
        if arch_key not in cache_keys:
            out[iso] = float(pc.RESOURCE_ADEQUACY_MARGIN) * annual_mwh / gaf
            continue
        ridx = cache_keys.index(arch_key)
        tc = np.asarray(cache.column('total_clean')[ridx].as_py(),
                        dtype=np.float64)
        demand_hr = _DEMAND_PROFILE_BY_ISO[iso]
        demand_frac = demand_hr / max(1.0, annual_mwh)
        demand_margin_frac = demand_frac * (1.0 + pc.RESOURCE_ADEQUACY_MARGIN)
        margin_resid = np.maximum(0.0, demand_margin_frac - tc)
        resid_2025 = float(np.percentile(margin_resid, 99.97))
        out[iso] = resid_2025 * annual_mwh / gaf
    return out


_GAS_RAW_2025_BY_ISO = _compute_gas_raw_2025_by_iso()


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


_EP_TAG_MAP = {
    90: 'ep90', 95: 'ep95', 99: 'ep99', 99.9: 'ep99p9',
}


def _ep_tag(endpoint_pct: float) -> str:
    return _EP_TAG_MAP.get(endpoint_pct, f'ep{int(round(endpoint_pct))}')


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
    # --- Phase C foresight-solver fields (memo §3, §5.2, §6.4) ---
    # solver_mode='myopic' (default) preserves every existing call site
    # bit-identically. solver_mode='foresight' dispatches _run_one to
    # solve_pathway_with_foresight and emits schema-v3 payloads.
    solver_mode: str = 'myopic'
    # foresight_lambda: None ≡ 0.0 (degenerate-to-myopic on truncated prefix).
    foresight_lambda: Optional[float] = None
    # endpoint_year: SBTi-ladder target year. None → resolved via
    # _foresight_endpoint_year(endpoint_pct) at dispatch time.
    endpoint_year: Optional[int] = None
    # endpoint_target_shares: tuple of 15 floats in _FORESIGHT_SHARE_KEYS
    # order. frozen dataclass → tuple, not np.ndarray. None → looked up
    # from step 2.2A cache by solve_pathway_with_foresight.
    endpoint_target_shares: Optional[tuple] = None

    @property
    def output_path(self) -> Path:
        return OUTPUT_BASE / self.iso / f'pathway{self.pathway}_{_ep_tag(self.endpoint_pct)}.json'


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


def _synthetic_clean_peak_fallback(iso: str, ef: pa.Table,
                                   row_indices: np.ndarray) -> np.ndarray:
    """Legacy synthetic P@M min path — used only for archetype-miss rows.

    Preserves prior v1 behavior verbatim (ignores storage, takes .min over
    hours). Never called on the hot path when the dispatch cache is healthy.
    """
    profiles = _build_clean_profile_table(iso)
    P = np.stack([profiles[r] for r in _RESOURCE_COLS], axis=0)  # (R, 8760)
    n = ef.num_rows
    M = np.column_stack([
        ef.column(c).to_numpy() if c in ef.column_names else np.zeros(n)
        for c in _RESOURCE_COLS
    ]).astype(np.float64)
    base_demand_mwh = float(pc.REGIONAL_DEMAND_TWH[iso]) * 1.0e6
    sub = (M[row_indices] / 100.0) * base_demand_mwh             # (k, R)
    return (sub @ P).min(axis=1).astype(np.float64)


WORST_HOUR_PERCENTILE = 99.97  # SPEC §24.5 — NERC/PJM LOLE-aligned (≤2.6 h/yr).


def _per_archetype_resid_norm(iso: str, cache: pa.Table) -> np.ndarray:
    """Per-archetype 99.97th-percentile margin-on-demand residual (fraction).

    Residual[h] = max(0, demand_frac[h] × (1 + RA) − total_clean_frac[h]).
    Returns a (n_archetypes,) array of float64 fractions of annual demand.
    Mirrors `_worst_hour_residual_norm` in step2_2a_cost_optimization.py:485
    without importing from it.
    """
    demand_hr = _DEMAND_PROFILE_BY_ISO[iso]
    annual_mwh = float(pc.REGIONAL_DEMAND_TWH[iso]) * 1.0e6
    demand_frac = demand_hr / max(1.0, annual_mwh)  # sums to 1.0
    demand_margin_frac = demand_frac * (1.0 + pc.RESOURCE_ADEQUACY_MARGIN)
    TC = np.stack([
        np.asarray(cache.column('total_clean')[i].as_py(), dtype=np.float64)
        for i in range(cache.num_rows)
    ], axis=0)  # (A, 8760), fractions of annual demand
    margin_resid = np.maximum(0.0, demand_margin_frac[None, :] - TC)
    return np.percentile(margin_resid, WORST_HOUR_PERCENTILE, axis=1)


def precompute_clean_peak_hour_mw(iso: str, threshold) -> pa.Table:
    """Stage-1: per-EF-row 99.97th-percentile residual-gap fraction.

    v3 rewrite — preserves v2's nearest-archetype semantics exactly while
    restructuring the L1 search for ~2x throughput and adding a defensive
    ``hourly_match_score`` filter:

    1. L1 nearest-neighbor via iterate-over-archetypes in int32 space.
       The v2 implementation used ``np.abs(eb[:,None,:] - mm[None,:,:]).sum(2)``
       in float64 on 5k-row batches, allocating ~25 MB intermediates per
       batch. v3 iterates the ~63 manifest archetypes, computing ``np.abs(ef
       - manifest[a]).sum(axis=1)`` per archetype and tracking a running
       best — O(n) memory, int32 arithmetic, and no Python-level batching
       loop. Semantics unchanged (argmin of L1 distance to manifest).

    2. Secondary ``hourly_match_score`` defensive floor. Rows whose score
       is more than ``SCORE_FLOOR_PCT`` below the threshold are assigned
       ``+inf`` residual, which Stage-2's argmin auto-excludes via
       ``np.isfinite``. No-op on correctly-produced per-threshold EF
       files (score band is upstream-narrow already).

    Output columns (unchanged vs. v2):
      - ``resid_norm_p9997``  : float32, fraction of annual demand (or
                                ``+inf`` for score-outside-window rows).
      - ``clean_peak_hour_mw``: float32, diagnostic MW value. Infinities
                                are replaced with 0 before the
                                ``(1 - resid) * peak`` computation so the
                                column stays finite.
    """
    ef = _read_ef_table(iso, threshold)
    n = ef.num_rows
    resid_out = np.empty(n, dtype=np.float32)

    cache = _DISPATCH_CACHE_BY_ISO.get(iso)
    am = _ANNUAL_MANIFEST_BY_ISO.get(iso)

    miss_resid = np.float32(float(pc.RESOURCE_ADEQUACY_MARGIN) / HOURS_PER_YEAR)

    if cache is None or am is None:
        # No cache — conservative fallback: treat as full residual (all RA).
        resid_out[:] = miss_resid
        _STAGE1_MISSES[(iso, threshold)] = list(range(n))
        print(f"[stage1] {iso} t={threshold}: no dispatch cache; "
              f"conservative fallback for all {n} rows", flush=True)
    else:
        # Per-archetype 99.97-percentile residual fraction — computed once.
        resid_per_arch = _per_archetype_resid_norm(iso, cache).astype(np.float32)

        cache_keys = cache.column('archetype_key').to_pylist()
        key_to_cache_row = {k: i for i, k in enumerate(cache_keys)}

        mcols = ['mix_clean_firm', 'mix_solar', 'mix_wind', 'mix_offshore_wind',
                 'mix_ccs_ccgt', 'mix_hydro', 'mix_solar_batt4', 'mix_solar_batt8',
                 'mix_wind_batt4', 'mix_wind_batt8']
        manifest_mat = np.column_stack(
            [am.column(c).to_numpy() for c in mcols]).astype(np.int32)  # (A_man, 10)
        manifest_keys = am.column('archetype_key').to_pylist()
        manifest_to_cache = np.array(
            [key_to_cache_row.get(k, -1) for k in manifest_keys],
            dtype=np.int64)

        ef_cols = [c.replace('mix_', '') for c in mcols]
        ef_mat = np.column_stack([
            ef.column(c).to_numpy() if c in ef.column_names else np.zeros(n)
            for c in ef_cols
        ]).astype(np.int32)  # (n, 10)

        # L1-NN by iterating over A_man archetypes with a running argmin —
        # O(n * A_man * D) time, O(n) memory, int32 throughout.
        if n > 0 and len(manifest_mat) > 0:
            best_dist = np.full(n, np.iinfo(np.int32).max, dtype=np.int32)
            best_arch = np.zeros(n, dtype=np.int32)
            for a in range(len(manifest_mat)):
                d = np.abs(ef_mat - manifest_mat[a]).sum(axis=1)
                imp = d < best_dist
                best_dist[imp] = d[imp]
                best_arch[imp] = a
            cache_idx = manifest_to_cache[best_arch]
        else:
            cache_idx = np.empty(n, dtype=np.int64)

        hit_mask = cache_idx >= 0
        safe_idx = np.where(hit_mask, cache_idx, 0)
        resid_out[:] = np.where(hit_mask, resid_per_arch[safe_idx], miss_resid
                                ).astype(np.float32)
        miss_rows = np.nonzero(~hit_mask)[0].tolist()

        _STAGE1_MISSES[(iso, threshold)] = miss_rows
        hits = n - len(miss_rows)
        match_rate = hits / max(1, n)
        print(f"[stage1] {iso} t={threshold}: archetype match "
              f"{hits}/{n} = {match_rate * 100:.1f}%  "
              f"(misses={len(miss_rows)})", flush=True)

    # Secondary defensive filter: rows with hourly_match_score far below the
    # threshold get +inf resid, excluding them from the Stage-2 argmin.
    # No-op on correctly-produced per-threshold EF files.
    SCORE_FLOOR_PCT = 10.0
    if 'hourly_match_score' in ef.column_names:
        score_vec = ef.column('hourly_match_score').to_numpy()
        outside = score_vec < (float(threshold) - SCORE_FLOOR_PCT)
        if outside.any():
            resid_out[outside] = np.float32(np.inf)

    # Diagnostic MW value for display (line ~1029). Not consumed by Stage-2.
    peak_mw = float(pc.PEAK_DEMAND_MW[iso])
    resid_safe = np.where(np.isfinite(resid_out), resid_out, 0.0).astype(np.float64)
    clean_peak_diag = (1.0 - resid_safe) * peak_mw
    tbl = pa.table({
        'resid_norm_p9997': pa.array(resid_out),
        'clean_peak_hour_mw': pa.array(clean_peak_diag.astype(np.float32)),
    })
    return tbl.replace_schema_metadata({
        'source_cache_version': _dispatch_cache_version(iso),
        'stage1_version': '3',
        'iso': iso, 'threshold': str(threshold),
    })


_STAGE1_VERSION = '3'


def _load_or_build_peakclean(iso: str, threshold) -> pa.Table:
    sp = _peakclean_path(iso, threshold)
    target_cache = _dispatch_cache_version(iso)
    if sp.exists():
        meta = pq.read_schema(sp).metadata or {}
        if (meta.get(b'source_cache_version', b'').decode() == target_cache
                and meta.get(b'stage1_version', b'').decode() == _STAGE1_VERSION):
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
    arrs['resid_norm_p9997'] = (pc_tbl.column('resid_norm_p9997').to_numpy()
                                if 'resid_norm_p9997' in pc_tbl.column_names
                                else np.zeros(n, dtype=np.float32))
    return arrs


# ─── Pool loader: concatenated mixes across ALL threshold bands ─────────────
# Step 2.1's docstring (`scripts/step2_1_efficient_frontier.py:33-37`) commits
# to a non-overlapping band layout: each mix lives in exactly one band (the
# band whose threshold is the highest T with T ≤ score). Consumers that need
# the full qualifying set must load every band — per-band `load_ef_mixes`
# under-samples the pool for any yearly CFE target below the endpoint. The
# pool loader fixes that for the pathway optimizer (both myopic + foresight).
#
# Two downstream modes:
#   - solve_pathway (myopic P1/P1a/P2a/P2b/myopic-P3): full pool, no filter.
#     The SBTi ladder (`_cfe_target_for_year`) + `cfe_mask` do the narrowing
#     year-by-year.
#   - solve_pathway_with_foresight (foresight-P3): max-share-per-resource
#     cap across 4 canonical endpoints' step-2.2A targets + slack. Stays
#     semantically consistent with foresight's endpoint-awareness and keeps
#     the filtered pool small enough for dense (Y, N) tensor math.

# Union of all band thresholds across ISOs (ordered low → high).
_EF_BAND_THRESHOLDS = (
    10.0, 20.0, 30.0, 40.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0,
    80.0, 85.0, 87.5, 90.0, 92.5, 95.0, 97.5, 99.0, 99.5, 99.9,
)

# Slack added to the per-resource cap. Resources whose max across the 4
# canonical endpoints' targets is ≤ _FILTER_ZERO_MAX_THRESHOLD get the
# larger "zero-target slack"; positive-target resources get the tighter
# slack. Kept as module-level constants for easy retuning when post-filter
# pool counts land below ~500 mixes (see SPEC filter table).
_FILTER_ZERO_MAX_THRESHOLD = 0.01   # ≤1pp → "zero-target"
_FILTER_ZERO_SLACK = 0.05           # 5pp slack on zero-target resources
_FILTER_POS_SLACK = 0.005           # 0.5pp slack on positive-target resources

# Streaming chunk size for candidate-dimension argmin in solve_pathway. Sized
# so per-chunk transient allocations (shares_nr[:k], row_score[:k], ratchet
# masks) stay in the low-hundreds of MB on the full ERCOT (8.7M) / CAISO
# (21M) pools. Myopic-P3 full-pool wall-clock scales linearly in this.
_POOL_CHUNK_SIZE = 500_000


def _iter_ef_band_paths(iso: str):
    """Yield (threshold, [paths]) for every on-disk band for this ISO.

    Handles `_partN` split files (step 2.1 splits >45MB parquets for GitHub).
    Skips `_peakclean` sidecars — those are picked up by the peakclean-pool
    loader in parallel.
    """
    for t in _EF_BAND_THRESHOLDS:
        name = str(int(t)) if float(t).is_integer() else str(t)
        parts = sorted(EF_DIR.glob(f'step_2_1_EF_{iso}_{name}_part*.parquet'))
        if parts:
            yield t, parts
            continue
        main = EF_DIR / f'step_2_1_EF_{iso}_{name}.parquet'
        if main.exists():
            yield t, [main]


def _read_ef_pool_table(iso: str) -> pa.Table:
    """Concatenate every on-disk EF band for `iso` into one pa.Table.

    Schema drift handled:
      - `iso` column dtype varies (string vs large_string across 4 SPP/NYISO
        bands) — `pa.concat_tables(..., promote_options='default')` unifies.
      - Resource-column coverage varies by ISO (CAISO has +offshore_wind+
        geothermal; NEISO/NYISO/PJM have +offshore_wind; the other 3 have
        only the base set; no band has `ccs_ccgt`). Missing columns are
        zero-filled downstream in `load_ef_pool` via the same
        `if c in column_names else zeros(n)` idiom already used by
        `load_ef_mixes`.
    """
    tables: list[pa.Table] = []
    for _, paths in _iter_ef_band_paths(iso):
        for p in paths:
            tables.append(pq.read_table(p))
    if not tables:
        raise FileNotFoundError(
            f"_read_ef_pool_table: no EF bands found for {iso} under {EF_DIR}"
        )
    return pa.concat_tables(tables, promote_options='default')


def _load_or_build_peakclean_pool(iso: str) -> pa.Table:
    """Stitch per-band peakclean sidecars in the same order as the EF pool.

    Each band is handled by the existing `_load_or_build_peakclean` (which
    rebuilds the sidecar on dispatch-cache/version drift). The row order of
    the concatenated sidecar table matches `_read_ef_pool_table(iso)` 1:1
    because we iterate the same `_iter_ef_band_paths` sequence and
    `precompute_clean_peak_hour_mw` uses `_read_ef_table` (per-band,
    single-threshold) internally — so the sidecar for a split band is
    rebuilt as one table covering all `_partN` rows in file order.
    """
    tables: list[pa.Table] = []
    for t, _paths in _iter_ef_band_paths(iso):
        tables.append(_load_or_build_peakclean(iso, t))
    if not tables:
        raise FileNotFoundError(
            f"_load_or_build_peakclean_pool: no bands found for {iso}"
        )
    return pa.concat_tables(tables, promote_options='default')


def _compute_pool_endpoint_caps(cfg: 'RunConfig') -> np.ndarray:
    """Per-resource filter caps from max across 4 canonical endpoints' targets.

    Returns a (len(_RESOURCE_COLS),) vector of upper bounds on each resource's
    share (fraction). A mix passes the filter iff every resource's share ≤
    the cap. Cap = per-resource max target across endpoints {90, 95, 99, 99.9}
    plus slack — _FILTER_ZERO_SLACK if that max is ≤ _FILTER_ZERO_MAX_THRESHOLD
    (the resource is effectively off-target at every endpoint), else
    _FILTER_POS_SLACK.
    """
    n_res = len(_RESOURCE_COLS)
    max_targets = np.zeros(n_res, dtype=np.float64)
    for ep_pct in (90.0, 95.0, 99.0, 99.9):
        targets = _compute_endpoint_target_shares(
            iso=cfg.iso,
            endpoint_pct=ep_pct,
            demand_growth_level=cfg.demand_growth_level,
            geo_cost_level=cfg.geo_cost_level,
            firm_cost_level=cfg.firm_cost_level,
            ccs_cost_level=cfg.ccs_cost_level,
            tx_level=cfg.tx_level,
        )
        # First n_res dims of _FORESIGHT_SHARE_KEYS are _RESOURCE_COLS.
        max_targets = np.maximum(max_targets, targets[:n_res])
    slack = np.where(max_targets <= _FILTER_ZERO_MAX_THRESHOLD,
                     _FILTER_ZERO_SLACK, _FILTER_POS_SLACK)
    return max_targets + slack


def load_ef_pool(
    iso: str,
    *,
    filter_to_endpoints: bool = False,
    cfg_for_filter: Optional['RunConfig'] = None,
) -> dict[str, np.ndarray]:
    """Full-pool EF loader.

    Returns the same dict shape as `load_ef_mixes` (same keys: all
    _RESOURCE_COLS, all _STORAGE_COLS, 'hourly_match_score',
    'clean_peak_hour_mw', 'resid_norm_p9997') — concatenated across every
    on-disk threshold band for `iso`.

    When `filter_to_endpoints=True`, rejects any mix whose per-resource
    share exceeds the cap computed by `_compute_pool_endpoint_caps(cfg)`.
    """
    ef = _read_ef_pool_table(iso)
    pc_tbl = _load_or_build_peakclean_pool(iso)
    n = ef.num_rows
    if pc_tbl.num_rows != n:
        raise RuntimeError(
            f"load_ef_pool: peakclean pool rows ({pc_tbl.num_rows}) != "
            f"EF pool rows ({n}) for {iso} — band order drift"
        )
    arrs: dict[str, np.ndarray] = {
        c: (ef.column(c).to_numpy() if c in ef.column_names else np.zeros(n))
        for c in _RESOURCE_COLS + _STORAGE_COLS + ('hourly_match_score',)
    }
    arrs['clean_peak_hour_mw'] = pc_tbl.column('clean_peak_hour_mw').to_numpy()
    arrs['resid_norm_p9997'] = (
        pc_tbl.column('resid_norm_p9997').to_numpy()
        if 'resid_norm_p9997' in pc_tbl.column_names
        else np.zeros(n, dtype=np.float32)
    )
    # Drop the source tables so their arrow buffers can be reclaimed before
    # the filter pass allocates an (N, 11) share matrix (~700 MB on ERCOT).
    del ef, pc_tbl

    if not filter_to_endpoints:
        return arrs

    if cfg_for_filter is None:
        raise ValueError(
            "load_ef_pool(filter_to_endpoints=True) requires cfg_for_filter"
        )
    caps = _compute_pool_endpoint_caps(cfg_for_filter)
    # Chunked share-cap check to avoid an (N, 11) float64 allocation on 8M+
    # row pools. Keeps filter peak transient under ~100 MB per chunk.
    caps_pct = (caps * 100.0).astype(np.float64)               # compare in %
    keep = np.ones(n, dtype=bool)
    chunk = _POOL_CHUNK_SIZE
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        kc = np.ones(e - s, dtype=bool)
        for ri, r in enumerate(_RESOURCE_COLS):
            kc &= arrs[r][s:e].astype(np.float64) <= (caps_pct[ri] + 1.0e-9)
        keep[s:e] = kc
    kept_n = int(keep.sum())
    print(f"[pool] {iso} filter_to_endpoints: {n:,} → {kept_n:,} mixes "
          f"(caps={np.round(caps, 4).tolist()})", flush=True)
    return {k: v[keep] for k, v in arrs.items()}


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


def gas_sizing_matrix(annual_mwh_vec: np.ndarray,
                      existing_gas_vec: np.ndarray,
                      resid_arr: np.ndarray,
                      gas_raw_2025: float,
                      existing_base_mw: float,
                      gaf: float) -> np.ndarray:
    """Per-year-per-mix new gas (MW) under v1 margin-on-demand residual rule.

    gap_mwh[y, mix] = resid_arr[mix] * annual_mwh_vec[y]     # (N_YEARS, n_mixes)
    gas_raw[y, mix] = gap_mwh / gaf
    gas_needed[y, mix] = max(0, existing_base + gas_raw - gas_raw_2025)
    new_gas[y, mix] = max(0, gas_needed - existing_gas_vec[y])

    Matches the formula in step2_2a_cost_optimization.py:865–888 without
    importing it. Fully vectorized — preserves N_YEARS=26 invariant.
    """
    gap_mwh = resid_arr[None, :] * annual_mwh_vec[:, None]
    gas_raw = gap_mwh / max(1e-9, gaf)
    gas_needed = np.maximum(0.0, existing_base_mw + gas_raw - gas_raw_2025)
    return np.maximum(0.0, gas_needed - existing_gas_vec[:, None])


@functools.lru_cache(maxsize=None)
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


@functools.lru_cache(maxsize=None)
def _resource_tx(iso: str, resource: str, tx_level: str) -> float:
    if resource in ('solar', 'wind', 'clean_firm', 'ccs_ccgt', 'offshore_wind'):
        return float(pc.get_tx(resource, tx_level, iso))
    if resource in ('solar_batt4', 'solar_batt8'):
        return float(pc.get_tx('solar', tx_level, iso))
    if resource in ('wind_batt4', 'wind_batt8'):
        return float(pc.get_tx('wind', tx_level, iso))
    return 0.0


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
        # P1/P1a — no new nuclear, no CCS, but existing-nuclear UPRATES are
        # still allowed. Allow mixes up to baseline clean_firm + uprate-cap
        # fraction (UPRATE_CAP_TWH / annual demand). The tranche decomposer
        # (uprate-only locked branch) flags over-cap targets as
        # feasible=False; Phase 4 propagates that to ∞ cost so they lose
        # the argmin. For Phase 3 the cost signal alone does the work.
        uprate_cap_twh = float(pc.UPRATE_CAP_TWH[cfg.iso])
        demand_twh_y = demand_twh_vec(cfg.iso, cfg.demand_growth_level)
        uprate_cap_pct_y = uprate_cap_twh / np.maximum(demand_twh_y, 1e-6) * 100.0
        # (Y, n) bool — mix's clean_firm share ≤ baseline + uprate-cap (+0.5 pct
        # slack to admit the nearest integer EF bucket above the cap).
        cf_allowed = cf[None, :] <= (floor + uprate_cap_pct_y[:, None] + 0.5)
        mask[:] = cf_allowed & no_ccs[None, :]
    else:
        # P2a/P2b/P3 — admit all mixes in every year. The tranche decomposer
        # prices any pre-FOAK new-nuke/CCS/geo at FOAK LCOE (Wright's curve),
        # so the argmin self-penalizes early deployment on cost rather than
        # being hard-blocked. No pre-NOAK pmask ceiling.
        mask[:] = np.ones((N_YEARS, n), dtype=bool)
    return mask


def _cfe_target_for_year(year: int, endpoint_pct: float) -> float:
    # Ramp terminates at _foresight_endpoint_year(endpoint_pct) — the same
    # SBTi-ladder target year used by the foresight path (canonical source:
    # pipeline_config.THRESHOLD_TARGET_YEARS). Interior buildup waypoints
    # stay in the schedule only if they fall strictly before the deadline;
    # years after the deadline hold flat at endpoint_pct.
    end_year = _foresight_endpoint_year(endpoint_pct)
    waypoints = [(2025, 0.0), (2030, 50.0), (2035, 70.0),
                 (2040, 90.0), (2045, 95.0)]
    waypoints = [(y, t) for (y, t) in waypoints if y < end_year]
    waypoints.append((end_year, endpoint_pct))
    if year <= waypoints[0][0]:
        return 0.0
    if year >= end_year:
        return endpoint_pct
    for (y0, t0), (y1, t1) in zip(waypoints, waypoints[1:]):
        if y0 <= year <= y1:
            frac = (year - y0) / (y1 - y0)
            return min(endpoint_pct, t0 + (t1 - t0) * frac)
    return endpoint_pct


# ─── Clean-firm merit-order tranche decomposition ───────────────────────────

# Pathway → whether non-uprate tranches (geo, new nuclear, CCS retrofit) are
# ever allowed. `None` = never (P1 family — existing-nuclear uprates only).
# `True` = always buildable; the year-adjusted LCOE returns FOAK cost in
# pre-FOAK years, so the argmin naturally avoids them until they're cheap
# enough to win on cost. No hard pre-FOAK lock — pre-NOAK build is a cost
# question, not a feasibility question.
_PATHWAY_TRANCHES_AVAILABLE = {
    '1': False, '1a': False, '1b': False,
    '2a': True, '2b': True, '3': True,
}


def decompose_clean_firm_tranches(
    cf_pct: np.ndarray,
    iso: str,
    pathway: str,
    annual_mwh_vec: np.ndarray,
    cfg: 'RunConfig',
) -> dict:
    """Disaggregate per-mix clean_firm share into 5 merit-order tranches.

    No blending. Every caller gets per-tranche (N_YEARS, n_mixes) TWh arrays
    and per-year effective $/MWh (LCOE + TX baked in). Argmin, ledger, and
    output layers all consume the tranche grid directly and compute
    cost = Σ tranche_twh × tranche_eff_lcoe × 1e6 — the existing tranche
    contributes $0 by construction.

    Merit order (each tranche fills from what the previous tranche left):
      Tranche 0 — existing nuclear baseline. LCOE=0, TX=0. Cap per mix per
        year = GRID_MIX_SHARES[iso].clean_firm × demand. Always available
        (it's the currently-operating fleet).
      Tranche 1 — nuclear uprates. LCOE = UPRATE_LCOE[firm_lev], TX=0
        (grid-connected reuse of existing interconnect). Cap = UPRATE_CAP_TWH.
      Tranche 2 — geothermal. CAISO only, else 0. Year-adjusted LCOE + TX.
        Cap = GEOTHERMAL_CAP_TWH (39 TWh). Gated by pathway unlock.
      Tranche 3 — new-build nuclear. Year-adjusted LCOE + TX. Uncapped.
      Tranche 4 — CCS-CCGT retrofit. Year-adjusted LCOE + TX (separate
        ccs_ccgt TX table). Cap = CCS_CAP_TWH[iso]. NEISO carries an
        additional gas-adder inside the LCOE. Tranches 3 and 4 compete
        head-to-head each year: the cheaper tranche fills remaining first;
        the loser fills only what the winner's cap left behind.

    Pathway gating: P1/P1a/P1b never unlock tranches 2-4 — if a mix's target
    exceeds baseline + uprate_cap the locked-year portion is flagged
    feasible=False, which solve_pathway converts to ∞ cost in the argmin.
    P2a/P2b/P3 unlock at NOAK_YEAR_BY_PATHWAY[pw] − 5. Pre-unlock years of
    those pathways behave identically to P1 (uprate-only ceiling).

    Args:
        cf_pct:          (n_mixes,) clean_firm share in percent (0–100).
        iso:             ISO code.
        pathway:         '1', '1a', '2a', '2b', '3'.
        annual_mwh_vec:  (N_YEARS,) annual demand in MWh, one per YEARS.
        cfg:             RunConfig supplying firm/ccs/geo/tx levels.

    Returns dict with keys:
        existing_twh, uprate_twh, geo_twh, nuke_new_twh, ccs_twh : (Y, n) TWh
        uprate_eff_lcoe, geo_eff_lcoe, nuke_new_eff_lcoe, ccs_eff_lcoe : (Y,)
            effective $/MWh including the appropriate TX adder (0 for
            uprate). The existing tranche's LCOE is 0 by construction —
            callers treat it as a no-op in cost sums.
        feasible : (Y, n) bool — False in locked (y, n) where the mix's
            new-clean_firm demand exceeds uprate_cap.
    """
    n = len(cf_pct)
    annual_twh_vec = annual_mwh_vec / 1.0e6                         # (Y,)
    baseline_pct = float(pc.GRID_MIX_SHARES[iso].get('clean_firm', 0.0))
    target_yn     = (cf_pct[None, :] / 100.0) * annual_twh_vec[:, None]  # (Y, n)
    baseline_twh_y = (baseline_pct / 100.0) * annual_twh_vec         # (Y,)

    # ── Tranche 0: existing baseline nuclear, $0 ──
    existing_twh = np.minimum(target_yn, baseline_twh_y[:, None])    # (Y, n)
    remaining    = target_yn - existing_twh                          # (Y, n), ≥ 0

    # Pathway new-tranche eligibility. P1 family: never. Others: always —
    # pre-FOAK build is priced at FOAK LCOE, so the argmin naturally won't
    # pick it until it's cheap enough to win on cost.
    tranches_available = _PATHWAY_TRANCHES_AVAILABLE.get(pathway, False)

    firm_lev = cfg.firm_cost_level
    ccs_lev  = cfg.ccs_cost_level
    tx_name  = cfg.tx_level
    geo_lev  = cfg.geo_cost_level or ('M' if iso == 'CAISO' else None)
    tx_cf    = float(pc.get_tx('clean_firm', tx_name, iso))
    tx_ccs   = float(pc.get_tx('ccs_ccgt',   tx_name, iso))

    # ── Tranche 1: Uprate (cap = UPRATE_CAP_TWH, TX = 0, always available) ──
    uprate_cap      = float(pc.UPRATE_CAP_TWH[iso])
    uprate_lcoe_val = float(pc.UPRATE_LCOE[firm_lev])
    uprate_twh      = np.minimum(remaining, uprate_cap)              # (Y, n)
    remaining       = remaining - uprate_twh                          # (Y, n)
    uprate_eff_lcoe = np.full(N_YEARS, uprate_lcoe_val, dtype=np.float64)

    # Feasibility: P1 family with remaining > uprate_cap ⇒ infeasible (no
    # new-nuke/CCS/geo ever available). All other pathways: always feasible
    # since the 2-4 tranches are always buildable at their year-adjusted LCOE.
    if tranches_available:
        feasible_yn = np.ones_like(target_yn, dtype=bool)
    else:
        feasible_yn = ~(remaining > 1.0e-9)                          # (Y, n)

    # ── Per-year effective LCOEs (Wright's-Law-adjusted) for tranches 2-4 ──
    nuke_eff_lcoe     = np.zeros(N_YEARS, dtype=np.float64)
    ccs_eff_lcoe      = np.zeros(N_YEARS, dtype=np.float64)
    geo_eff_lcoe      = np.zeros(N_YEARS, dtype=np.float64)
    for yi, y in enumerate(YEARS):
        # New-nuclear: FOAK→NOAK on pathway window, + clean_firm TX.
        foak_s, noak_y = pc.get_pathway_noak_window('nuclear', firm_lev, pathway)
        nuke_eff_lcoe[yi] = pc.year_adjusted_cost(
            pc.FOAK_NUCLEAR_NEWBUILD[iso],
            pc.NUCLEAR_NEWBUILD_LCOE['L'][iso],
            y, foak_s, noak_y,
        ) + tx_cf
        # CCS 45Q-on: FOAK→NOAK on pathway window, + ccs_ccgt TX (+ NEISO adder).
        foak_s, noak_y = pc.get_pathway_noak_window('ccs', ccs_lev, pathway)
        ccs_val = pc.year_adjusted_cost(
            pc.FOAK_CCS_45Q_ON[iso],
            pc.CCS_LCOE_45Q_ON['L'][iso],
            y, foak_s, noak_y,
        )
        if iso == 'NEISO':
            ccs_val += pc.NEISO_CCS_GAS_ADDER
        ccs_eff_lcoe[yi] = ccs_val + tx_ccs
        # Geo (CAISO only, else stays at 0).
        if iso == 'CAISO' and geo_lev:
            foak_s, noak_y = pc.get_pathway_noak_window('geo', firm_lev, pathway)
            geo_eff_lcoe[yi] = pc.year_adjusted_cost(
                pc.FOAK_GEOTHERMAL,
                pc.GEOTHERMAL_LCOE['L'],
                y, foak_s, noak_y,
            ) + tx_cf

    # ── Tranches 2-4: filled in ALL years when tranches_available. In
    # pre-FOAK years the effective LCOE is the FOAK cost (Wright's curve
    # returns FOAK before foak_start), so these tranches self-price out of
    # the argmin until they're competitive. For P1 family (no new firm), the
    # whole block is skipped and remaining>ε shows up as feasible_yn=False.
    geo_twh      = np.zeros_like(target_yn)
    nuke_new_twh = np.zeros_like(target_yn)
    ccs_twh      = np.zeros_like(target_yn)
    ccs_cap      = float(pc.CCS_CAP_TWH.get(iso, 9999.0))
    geo_cap_twh  = float(pc.GEOTHERMAL_CAP_TWH) if (iso == 'CAISO' and geo_lev) else 0.0

    if tranches_available:
        for yi in range(N_YEARS):
            rem_y = remaining[yi].copy()                              # (n,)
            # Tranche 2 — Geothermal (CAISO only).
            if geo_cap_twh > 0:
                geo_fill = np.minimum(rem_y, geo_cap_twh)
                geo_twh[yi] = geo_fill
                rem_y = rem_y - geo_fill
            # Tranches 3 & 4 — merit-order new-nuke vs CCS at year-y LCOE.
            if nuke_eff_lcoe[yi] <= ccs_eff_lcoe[yi] or ccs_cap <= 0:
                nuke_new_twh[yi] = rem_y
            else:
                ccs_fill = np.minimum(rem_y, ccs_cap)
                ccs_twh[yi] = ccs_fill
                nuke_new_twh[yi] = rem_y - ccs_fill

    return {
        'existing_twh':      existing_twh,
        'uprate_twh':        uprate_twh,
        'geo_twh':           geo_twh,
        'nuke_new_twh':      nuke_new_twh,
        'ccs_twh':           ccs_twh,
        'uprate_eff_lcoe':   uprate_eff_lcoe,
        'geo_eff_lcoe':      geo_eff_lcoe,
        'nuke_new_eff_lcoe': nuke_eff_lcoe,
        'ccs_eff_lcoe':      ccs_eff_lcoe,
        'feasible':          feasible_yn,
    }


_CLEAN_FIRM_TRANCHES = ('existing', 'uprate', 'geo', 'nuke_new', 'ccs')
_CLEAN_FIRM_TRANCHE_VINTAGE_KEY = {
    'existing': 'clean_firm_existing',
    'uprate':   'clean_firm_uprate',
    'geo':      'clean_firm_geo',
    'nuke_new': 'clean_firm_nuke_new',
    'ccs':      'clean_firm_ccs',
}
_CLEAN_FIRM_TRANCHE_TX = {
    'existing': 0.0,
    'uprate':   0.0,
}  # geo/nuke_new/ccs — TX is already baked into their eff_lcoe vectors


# ─── P3 foresight primitives (memo §§4-5) ────────────────────────────────────
#
# Three pure helpers for the endpoint-aware solver mode. solve_pathway (myopic)
# consumes them for the Phase B read-only preview. solve_pathway_with_foresight
# consumes them for real winner selection.
#
# 15-dim share vector basis: 11 _RESOURCE_COLS + 4 non-existing CF tranches
# (uprate/geo/nuke_new/ccs). The "existing" CF tranche is omitted — it is
# always the GRID_MIX_SHARES baseline and offers no steering handle.
#
# ───────────────────────────────────────────────────────────────────────────
# DIMENSIONAL CONVENTION — Phase C decision 2026-04-21
# ───────────────────────────────────────────────────────────────────────────
# Phase B's 28-cell preview produced ZERO interior-year divergence at any
# λ ∈ {0.05, 0.15, 0.5} on any ISO. Root cause: row_score is in USD (O(1e10))
# while ‖shares − target‖² is O(1), so the literal-λ penalty is ~10 orders
# of magnitude too small to move the argmin.
#
# Fix (Phase C): the foresight scorer adds a USD-scale constant C to the
# penalty so the sum sits in the same units as row_score, i.e.
#     score_y[m] = row_score[m] + λ · w(y) · ‖shares[m] − target‖² · C
# with C = endpoint_year_demand_MWh × FORESIGHT_REF_LCOE_USD_PER_MWH. That
# keeps λ in the memo's {0, 0.05, 0.15, 0.5, 1.5} band and keeps row_score
# untouched everywhere else. Per-ISO λ calibration (memo Q2) absorbs the
# cross-ISO share_distance variance (ERCOT ~1.0 vs SPP ~0.22 ≈ 20×).
# ───────────────────────────────────────────────────────────────────────────

# Reference LCOE for the foresight-scorer USD normalizer. Fixed cross-ISO so
# λ values are comparable in Phase D's calibration table; per-ISO variance
# lives in λ itself.
FORESIGHT_REF_LCOE_USD_PER_MWH = 100.0

_FORESIGHT_SHARE_KEYS = _RESOURCE_COLS + (
    'cf_uprate', 'cf_geo', 'cf_nuke_new', 'cf_ccs',
)
_FORESIGHT_LAMBDAS = (0.05, 0.15, 0.5)
_FORESIGHT_ENDPOINT_PCTS = frozenset({90.0, 95.0, 99.0, 99.9})


def _foresight_endpoint_year(endpoint_pct: float) -> int:
    """Map an endpoint CFE threshold to its SBTi-ladder target year (shared
    canon with step 2.2A via pipeline_config.THRESHOLD_TARGET_YEARS). This
    is both the foresight trajectory's stopping year and the year at which
    step 2.2A's cost-optimal target mix is referenced.
    """
    return int(pc.THRESHOLD_TARGET_YEARS[endpoint_pct])


_STEP_2_2A_DIR = _ROOT / 'data' / 'step2.2-cost'
_STEP_2_2A_TARGET_SHARES_CACHE: dict = {}


def _threshold_tag(endpoint_pct: float) -> str:
    return (str(int(endpoint_pct))
            if float(endpoint_pct).is_integer() else str(endpoint_pct))


def _compute_endpoint_target_shares(
    iso: str,
    endpoint_pct: float,
    demand_growth_level: str = 'Medium',
    geo_cost_level: Optional[str] = None,
    firm_cost_level: str = 'M',
    ccs_cost_level: str = 'M',
    tx_level: str = 'Medium',
) -> np.ndarray:                       # (15,) in _FORESIGHT_SHARE_KEYS order
    """Endpoint target-mix lookup from the step 2.2A cost-optimizer cache.

    `reliability_tax/README.md:44` declares
    `data/step2.2-cost/step_2_2a_DG_{ISO}_{THRESHOLD}.parquet` as the
    per-year cost-optimal reference for this sub-project: step 2.2A solves
    a static (demand-growth-adjusted) single-year cost optimization across
    5,832 sensitivity scenarios and caches the winning mix per (iso,
    threshold, scenario, growth_level, year). The foresight penalty in
    `solve_pathway_with_foresight` steers candidates toward this mix, so
    we simply look up the row matching the trajectory-solver's
    sensitivity defaults and convert it into the 15-dim share vector the
    scorer compares against.

    Scenario-key encoding (step 2.2A ``make_scenario_key``) is 9-dim:
    `{Ren}{Firm}{Batt}{LDES}_{Fuel}_{Tx}_{CCS}{45Q}_{Geo}`. The
    trajectory solver exposes firm/ccs/tx as per-RunConfig fields; the
    other toggles (renewables cost, batt/LDES cost, fuel, 45Q) default to
    'M' / '1' for the base case. geo_cost_level is 'X' for non-CAISO ISOs.

    Cached per (iso, endpoint_pct, demand_growth_level, scenario) because
    the parquet load dominates (<1s warm, ~10s cold for a 17k-row DG
    file) and multiple foresight runs share the same (iso, endpoint).
    """
    cache_key = (iso, round(endpoint_pct, 4), demand_growth_level,
                 firm_cost_level, ccs_cost_level, tx_level,
                 geo_cost_level or 'X')
    if cache_key in _STEP_2_2A_TARGET_SHARES_CACHE:
        return _STEP_2_2A_TARGET_SHARES_CACHE[cache_key]

    # Geo toggle: step 2.2A uses 'X' for ISOs without a geothermal resource
    # (non-CAISO) and an actual cost-level char ('M' default) for CAISO.
    geo_code = (geo_cost_level or 'M') if iso == 'CAISO' else 'X'
    tx_code  = tx_level[0].upper()  # 'Medium' → 'M', 'Low' → 'L', etc.
    firm_code = firm_cost_level.upper()[0]
    ccs_code  = ccs_cost_level.upper()[0]
    # Non-tunable defaults for axes the trajectory solver doesn't expose:
    # renewables/batt/LDES cost, fuel price, 45Q.
    scen_key = (f'{firm_code}M{firm_code}{firm_code}_M_{tx_code}_'
                f'{ccs_code}1_{geo_code}')

    dg_path = (_STEP_2_2A_DIR
               / f'step_2_2a_DG_{iso}_{_threshold_tag(endpoint_pct)}.parquet')
    tbl = pq.read_table(dg_path)
    mask = pacompute.and_(
        pacompute.equal(tbl.column('scenario'), scen_key),
        pacompute.equal(tbl.column('growth_level'), demand_growth_level),
    )
    sub = tbl.filter(mask)
    if sub.num_rows != 1:
        raise RuntimeError(
            f"_compute_endpoint_target_shares: expected 1 row for "
            f"{iso}/{scen_key}/{demand_growth_level}/ep{endpoint_pct}, "
            f"got {sub.num_rows} in {dg_path.name}"
        )
    row = sub.slice(0, 1).to_pylist()[0]
    annual_demand_twh = float(row['annual_demand_mwh']) / 1.0e6

    target = np.zeros(len(_FORESIGHT_SHARE_KEYS), dtype=np.float64)
    # Dims 0-10: resource shares (mix_* columns are % of endpoint-year demand).
    for i, r in enumerate(_RESOURCE_COLS):
        target[i] = float(row.get(f'mix_{r}', 0.0) or 0.0) / 100.0
    # Dims 11-14: new-build clean-firm tranche shares (TWh / endpoint TWh).
    target[11] = float(row.get('tranche_uprate_twh', 0.0) or 0.0) / annual_demand_twh
    target[12] = float(row.get('tranche_geo_twh', 0.0) or 0.0) / annual_demand_twh
    target[13] = float(row.get('tranche_nuclear_newbuild_twh', 0.0) or 0.0) / annual_demand_twh
    target[14] = float(row.get('tranche_ccs_tranche_twh', 0.0) or 0.0) / annual_demand_twh

    _STEP_2_2A_TARGET_SHARES_CACHE[cache_key] = target
    return target


def _score_year_sunk_cost(
    target_twh_nr: np.ndarray,        # (N, R) shares_nr × demand_vec[yi]
    floor_twh_r: np.ndarray,          # (R,)   running-max non-CF floors
    delivered_yr_yi: np.ndarray,      # (R,)   delivered_yr[yi]
    uprate_twh_n: np.ndarray,         # (N,)   uprate_twh_yn[yi]
    geo_twh_n: np.ndarray,            # (N,)
    nuke_new_twh_n: np.ndarray,       # (N,)
    ccs_twh_n: np.ndarray,            # (N,)
    floor_uprate: float,
    floor_geo: float,
    floor_nuke_new: float,
    floor_ccs: float,
    uprate_eff_yi: float,             # uprate_eff[yi]
    geo_eff_yi: float,
    nuke_new_eff_yi: float,
    ccs_eff_yi: float,
    storage_cost_n: np.ndarray,       # (N,)   storage_cost_yn[yi]
    gas_fixed_n: np.ndarray,          # (N,)   gas_fixed_y[yi]
) -> np.ndarray:                      # (N,)   row_score in USD
    """Sunk-cost-B scorer (memo §5.2). Only the INCREMENTAL TWh above each
    running-max floor prices at year-y LCOE; prior-built capacity is sunk
    and enters at $0. Returns the year-y row_score vector in USD, for all
    N candidate mixes. Fully vectorized — no Python loops over candidates.
    Shared by solve_pathway (myopic) and solve_pathway_with_foresight.
    """
    incr_nonCF  = np.maximum(0.0, target_twh_nr - floor_twh_r[None, :])   # (N, R)
    score_nonCF = (incr_nonCF * delivered_yr_yi[None, :]).sum(axis=1) * 1.0e6
    score_cf    = (
          np.maximum(0.0, uprate_twh_n   - floor_uprate)   * uprate_eff_yi
        + np.maximum(0.0, geo_twh_n      - floor_geo)      * geo_eff_yi
        + np.maximum(0.0, nuke_new_twh_n - floor_nuke_new) * nuke_new_eff_yi
        + np.maximum(0.0, ccs_twh_n      - floor_ccs)      * ccs_eff_yi
    ) * 1.0e6
    return score_nonCF + score_cf + storage_cost_n + gas_fixed_n


def _project_shares_to_endpoint(
    shares_non_cf: np.ndarray,  # (N, 10) non-CF resource shares, fraction
    non_cf_keys: tuple,         # column order of shares_non_cf (matches non_cf)
    cf_share: np.ndarray,       # (N,)    clean_firm share, fraction
    tg: dict,                   # decompose_clean_firm_tranches output
    demand_vec: np.ndarray,     # (Y,)    TWh per year
    yi: int,                    # current year index
    endpoint_yi: int,           # endpoint year index
) -> np.ndarray:                # (N, 15) frozen-projection share vector per mix
    """Linear-freeze projection of candidate mixes at year yi to endpoint_yi
    (memo §5.6). If mix m is committed at year yi and no further commits are
    made, what resource-share vector do we land at, at endpoint_yi's demand?

    Assembles an 11-col resource matrix in _RESOURCE_COLS order from the
    caller's 10-col non-CF matrix + cf_share, then appends the 4 CF tranche
    shares. Fully vectorized — no Python loops over candidate mixes.
    """
    ratio = float(demand_vec[yi]) / float(demand_vec[endpoint_yi])
    n = shares_non_cf.shape[0]
    # Rebuild full 11-col resource matrix in _RESOURCE_COLS order so the
    # output column order matches _FORESIGHT_SHARE_KEYS[:11] regardless of
    # which non-CF column layout the caller used.
    resource_shares = np.empty((n, len(_RESOURCE_COLS)), dtype=np.float64)
    non_cf_idx = {r: i for i, r in enumerate(non_cf_keys)}
    for j, r in enumerate(_RESOURCE_COLS):
        if r == 'clean_firm':
            resource_shares[:, j] = cf_share * ratio
        else:
            resource_shares[:, j] = shares_non_cf[:, non_cf_idx[r]] * ratio

    denom = float(demand_vec[endpoint_yi])
    uprate_shr   = tg['uprate_twh'][yi]   / denom                 # (N,)
    geo_shr      = tg['geo_twh'][yi]      / denom
    nuke_new_shr = tg['nuke_new_twh'][yi] / denom
    ccs_shr      = tg['ccs_twh'][yi]      / denom

    return np.concatenate(
        [resource_shares,
         uprate_shr[:, None], geo_shr[:, None],
         nuke_new_shr[:, None], ccs_shr[:, None]],
        axis=1,
    )


def _score_year_with_endpoint(
    row_score: np.ndarray,           # (N,)   myopic year-yi scores
    shares_to_endpoint: np.ndarray,  # (N, 15) projected shares
    target_shares: np.ndarray,       # (15,)   endpoint target
    lam: float,                      # λ
    weight: float,                   # w(y, endpoint_year)
) -> np.ndarray:
    """myopic + λ · w · ‖shares − target‖²  (memo §5.2).

    Vectorized: one broadcast subtraction, square, and sum.
    """
    diff = shares_to_endpoint - target_shares[None, :]            # (N, 15)
    penalty = lam * weight * np.sum(diff * diff, axis=1)          # (N,)
    return row_score + penalty


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
    # Phase B diagnostic sidecar payload — populated only for P3 runs at the
    # 4 canonical endpoints ({90, 95, 99, 99.9}); None otherwise. Does not
    # affect winner selection, ledger, or any existing cached field.
    foresight_preview: Optional[dict] = None
    # Phase C foresight-solver metadata — populated only by
    # solve_pathway_with_foresight. Drives schema-v3 emission in
    # serialize_run_result_v3. None for myopic runs.
    foresight_metadata: Optional[dict] = None
    # Phase D cascade-tier vector (memo §5.4). Populated by both
    # solve_pathway and solve_pathway_with_foresight so Phase D's
    # calibration rule (cascade_activations ≤ myopic + 1) can read
    # the myopic baseline directly from the run result without
    # re-running. Values: 0 = Tier-1 full feasibility, 1 = Tier-2
    # relaxed CFE ±5, 2 = Tier-3 cf_feasible dropped, 3 = Tier-4
    # ratchet dropped, -1 = post-endpoint frozen (not simulated).
    # None only for legacy callers that bypass the two solvers.
    cascade_tier_y: Optional[np.ndarray] = None


def solve_pathway(cfg: 'RunConfig') -> PathwayRunResult:
    # Phase 1a Commit B: streaming chunked argmin. The myopic path now
    # consumes the full unfiltered mix pool (ERCOT ~8.7M, CAISO ~21M rows)
    # — dense (Y, N) tensors are infeasible at that size, so setup here
    # produces only per-year SCALARS / (Y,) and (R,) arrays plus (N,)
    # columns that stream through the per-chunk kernel in the year loop.
    # Any (Y, N) quantity (gas_required, ratchet_fleet_y, gas_fuel,
    # uprate/geo/nuke_new/ccs tranche grids, storage_cost, pmask, cfe_mask,
    # shares_nr) is computed inside the chunk loop from these scalars.
    ef = load_ef_pool(cfg.iso, filter_to_endpoints=False)
    n_mixes = len(ef['hourly_match_score'])

    peak_vec = peak_demand_vec(cfg.iso, cfg.demand_growth_level)
    demand_vec = demand_twh_vec(cfg.iso, cfg.demand_growth_level)
    annual_mwh_vec = demand_vec * 1.0e6  # per-year annual MWh (grown)
    score_vec = ef['hourly_match_score']
    resid_arr = ef['resid_norm_p9997'].astype(np.float64)
    cf_share = ef['clean_firm'].astype(np.float64) / 100.0  # (N,) fraction
    gas_raw_2025 = _GAS_RAW_2025_BY_ISO[cfg.iso]
    existing_base = float(pc.EXISTING_GAS_CAPACITY_MW[cfg.iso])
    gaf = float(pc.GAS_AVAILABILITY_FACTOR[cfg.iso])

    # Seed pass: clean_pct ≈ cfe_target to build existing_gas_vec. Second
    # pass (post-loop, prompt 3) recomputes with the realized winner scores.
    cfe_targets = np.array([_cfe_target_for_year(y, cfg.endpoint_pct) for y in YEARS])
    existing_gas_vec = existing_gas_available_vec(cfg.iso, cfe_targets, cfg.demand_growth_level)
    fuel_unit = pc.WHOLESALE_PRICES[cfg.iso] + pc.FUEL_ADJUSTMENTS[cfg.iso]['Medium']

    # Scalar gas-cost multipliers. Per-chunk code combines these with the
    # chunk's gas-sizing vector to produce fixed costs on the fly without
    # ever materializing (Y, N) new_gas_capex / new_gas_fom arrays.
    new_ccgt_capex_per_mw = pc.NEW_CCGT_COST_KW_YR[cfg.iso] * 1000.0
    new_ccgt_fom_per_mw   = pc.NEW_CCGT_FOM_KW_YR[cfg.iso]  * 1000.0
    existing_fom = (float(pc.EXISTING_GAS_CAPACITY_MW[cfg.iso])
                    * pc.EXISTING_GAS_FOM_KW_YR[cfg.iso] * 1000.0)

    # Non-CF resource tables (shares stay in the EF dict, indexed per chunk).
    non_cf = tuple(r for r in _RESOURCE_COLS if r != 'clean_firm')
    R = len(non_cf)
    tx_r = np.array(
        [_resource_tx(cfg.iso, r, cfg.tx_level) for r in non_cf],
        dtype=np.float64,
    )
    lcoe_yr = np.empty((N_YEARS, R), dtype=np.float64)
    for ri, r in enumerate(non_cf):
        for yi, y in enumerate(YEARS):
            lcoe_yr[yi, ri] = _resource_lcoe_year(cfg.iso, r, y, cfg)
    delivered_yr = lcoe_yr + tx_r[None, :]                            # (Y, R)

    # Per-year tranche SCALARS — same formulas as decompose_clean_firm_tranches
    # but without the (Y, N) outputs. Prompt 2 inlines the per-chunk tranche
    # split using these scalars + cf_share[s:e].
    firm_lev = cfg.firm_cost_level
    ccs_lev  = cfg.ccs_cost_level
    tx_name  = cfg.tx_level
    geo_lev  = cfg.geo_cost_level or ('M' if cfg.iso == 'CAISO' else None)
    tx_cf    = float(pc.get_tx('clean_firm', tx_name, cfg.iso))
    tx_ccs   = float(pc.get_tx('ccs_ccgt',   tx_name, cfg.iso))

    baseline_pct = float(pc.GRID_MIX_SHARES[cfg.iso].get('clean_firm', 0.0))
    tranches_available = _PATHWAY_TRANCHES_AVAILABLE.get(cfg.pathway, False)
    uprate_cap      = float(pc.UPRATE_CAP_TWH[cfg.iso])
    uprate_lcoe_val = float(pc.UPRATE_LCOE[firm_lev])
    uprate_eff      = np.full(N_YEARS, uprate_lcoe_val, dtype=np.float64)
    ccs_cap         = float(pc.CCS_CAP_TWH.get(cfg.iso, 9999.0))
    geo_cap_twh     = (float(pc.GEOTHERMAL_CAP_TWH)
                       if (cfg.iso == 'CAISO' and geo_lev) else 0.0)

    nuke_eff_lcoe = np.zeros(N_YEARS, dtype=np.float64)
    ccs_eff_lcoe  = np.zeros(N_YEARS, dtype=np.float64)
    geo_eff_lcoe  = np.zeros(N_YEARS, dtype=np.float64)
    for yi, y in enumerate(YEARS):
        foak_s, noak_y = pc.get_pathway_noak_window('nuclear', firm_lev, cfg.pathway)
        nuke_eff_lcoe[yi] = pc.year_adjusted_cost(
            pc.FOAK_NUCLEAR_NEWBUILD[cfg.iso],
            pc.NUCLEAR_NEWBUILD_LCOE['L'][cfg.iso],
            y, foak_s, noak_y,
        ) + tx_cf
        foak_s, noak_y = pc.get_pathway_noak_window('ccs', ccs_lev, cfg.pathway)
        ccs_val = pc.year_adjusted_cost(
            pc.FOAK_CCS_45Q_ON[cfg.iso],
            pc.CCS_LCOE_45Q_ON['L'][cfg.iso],
            y, foak_s, noak_y,
        )
        if cfg.iso == 'NEISO':
            ccs_val += pc.NEISO_CCS_GAS_ADDER
        ccs_eff_lcoe[yi] = ccs_val + tx_ccs
        if cfg.iso == 'CAISO' and geo_lev:
            foak_s, noak_y = pc.get_pathway_noak_window('geo', firm_lev, cfg.pathway)
            geo_eff_lcoe[yi] = pc.year_adjusted_cost(
                pc.FOAK_GEOTHERMAL,
                pc.GEOTHERMAL_LCOE['L'],
                y, foak_s, noak_y,
            ) + tx_cf

    # Storage per-year SCALAR pairs. Per-chunk code fetches ef[col][s:e]
    # and applies the unit-LCOE × annual demand multiplier without ever
    # building a (Y, N) storage_cost matrix.
    storage_lcoe_keys = {
        'battery_dispatch_pct': 'battery', 'battery8_dispatch_pct': 'battery8',
        'ldes_dispatch_pct': 'ldes', 'h2_dispatch_pct': 'h2',
    }
    storage_pairs: list[tuple[str, float]] = []
    for sc, lk in storage_lcoe_keys.items():
        if float(ef[sc].sum()) == 0.0:
            continue
        storage_pairs.append((sc, float(pc.LCOE_TABLES[lk]['Medium'][cfg.iso])))

    # P1/P1a pmask SCALARS. Matches _pathway_mask(): P1 family caps
    # clean_firm share at baseline + uprate-cap% (+0.5 pct slack) and
    # requires ccs_ccgt ≤ 0.5 pct. Per-chunk code reconstructs the mask
    # from ef['clean_firm'][s:e] + ef['ccs_ccgt'][s:e].
    is_p1_family = cfg.pathway in ('1', '1a')
    pathway_floor = _pathway_firm_floor_pct(cfg.iso) + 0.01
    uprate_cap_pct_y = uprate_cap / np.maximum(demand_vec, 1e-6) * 100.0  # (Y,)

    # Absolute-TWh floors (scalar + (R,)). Non-CF initial floor = baseline
    # grid-share TWh at base-year demand; CF-total floor = clean_firm
    # baseline TWh; CF-tranche floors start at 0.
    grid = pc.GRID_MIX_SHARES[cfg.iso]
    base_demand = float(pc.REGIONAL_DEMAND_TWH[cfg.iso])
    floor_twh_r    = np.array(
        [base_demand * grid.get(r, 0.0) / 100.0 for r in non_cf],
        dtype=np.float64,
    )
    floor_twh_cf   = base_demand * grid.get('clean_firm', 0.0) / 100.0
    floor_uprate   = 0.0
    floor_geo      = 0.0
    floor_nuke_new = 0.0
    floor_ccs      = 0.0

    winners           = np.empty(N_YEARS, dtype=np.int64)
    winner_feasible   = np.ones(N_YEARS, dtype=bool)
    ratchet_violated  = np.zeros(N_YEARS, dtype=bool)
    # Cascade tier per year: 0 = Tier-1, 1 = Tier-2 (relaxed CFE ±5),
    # 2 = Tier-3 (dropped cf_feasible), 3 = Tier-4 (dropped ratchet).
    cascade_tier_y    = np.zeros(N_YEARS, dtype=np.int32)

    # Endpoint-year schedule. Loop runs 2025 → endpoint_year; years past
    # the endpoint are frozen to the endpoint winner (mirrors the foresight
    # path's memo §6.5 post-endpoint freeze).
    _myopic_endpoint_year = _foresight_endpoint_year(float(cfg.endpoint_pct))
    _myopic_endpoint_yi = YEARS.index(_myopic_endpoint_year)

    # Phase B foresight preview is DROPPED on the full-pool streaming path
    # (task brief: the projection + 3 extra argmins per year would 4× the
    # chunk pass; Phase C solver's foresight_metadata is the canonical
    # diagnostic now). solve_pathway_with_foresight retains the preview.

    # Streaming ratchet state: the old code derived new-gas capex/FOM
    # ratcheting by `np.maximum.accumulate(gas_required, axis=0)` over the
    # full (Y, N) gas-sizing matrix. Here we carry a persistent (N,) vector
    # that tracks the per-mix running-max MW across prior years; the year
    # loop updates it chunk by chunk.
    running_max_gas_n = np.zeros(n_mixes, dtype=np.float64)

    RATCHET_TOL_TWH = 1.0e-6
    _CHUNK_SIZE = _POOL_CHUNK_SIZE
    for yi in range(_myopic_endpoint_yi + 1):
        # Target TWh per (mix, non-CF resource) at year-y grown demand.
        target_twh_nr = shares_nr * float(demand_vec[yi])             # (N, R)
        # Ratchet masks: a mix is admissible only if every already-committed
        # resource's year-y TWh claim ≥ its running-max floor.
        ratchet_nonCF = np.all(
            target_twh_nr >= (floor_twh_r[None, :] - RATCHET_TOL_TWH), axis=1)
        # Clean_firm TOTAL-share floor: mix's year-y absolute CF TWh must be
        # ≥ the running-max CF commitment. Prevents retiring existing nuclear
        # below its locked baseline and prevents later years from shrinking
        # below whatever CF was committed earlier. The tranche decomposer
        # handles the merit-order breakdown within clean_firm separately.
        target_twh_cf_n = cf_share * float(demand_vec[yi])             # (N,)
        ratchet_cf     = target_twh_cf_n >= (floor_twh_cf - RATCHET_TOL_TWH)
        ratchet_uprate = uprate_twh_yn[yi]   >= (floor_uprate   - RATCHET_TOL_TWH)
        ratchet_geo    = geo_twh_yn[yi]      >= (floor_geo      - RATCHET_TOL_TWH)
        ratchet_nuke   = nuke_new_twh_yn[yi] >= (floor_nuke_new - RATCHET_TOL_TWH)
        ratchet_ccs    = ccs_twh_yn[yi]      >= (floor_ccs      - RATCHET_TOL_TWH)
        ratchet_mask_y = (ratchet_nonCF & ratchet_cf & ratchet_uprate
                          & ratchet_geo & ratchet_nuke & ratchet_ccs)

        # Sunk-cost scorer (memo §5.2 — shared with solve_pathway_with_foresight).
        row_score = _score_year_sunk_cost(
            target_twh_nr=target_twh_nr,
            floor_twh_r=floor_twh_r,
            delivered_yr_yi=delivered_yr[yi],
            uprate_twh_n=uprate_twh_yn[yi],
            geo_twh_n=geo_twh_yn[yi],
            nuke_new_twh_n=nuke_new_twh_yn[yi],
            ccs_twh_n=ccs_twh_yn[yi],
            floor_uprate=floor_uprate, floor_geo=floor_geo,
            floor_nuke_new=floor_nuke_new, floor_ccs=floor_ccs,
            uprate_eff_yi=uprate_eff[yi], geo_eff_yi=geo_eff[yi],
            nuke_new_eff_yi=nuke_new_eff[yi], ccs_eff_yi=ccs_eff[yi],
            storage_cost_n=storage_cost_yn[yi],
            gas_fixed_n=gas_fixed_y[yi],
        )

        # Fallback cascade. Ratchet is non-negotiable except under the Tier-4
        # emergency case (all-infeasible even without cfe and cf_feasible).
        valid = ratchet_mask_y & pmask[yi] & cfe_mask[yi] & cf_feasible_yn[yi]
        tier = 0
        if not valid.any():
            # Tier 2: relax CFE ±5 pts (keep ratchet + pmask + cf_feasible).
            relaxed_cfe = score_vec >= cfe_targets[yi] - 5.0
            valid = ratchet_mask_y & pmask[yi] & relaxed_cfe & cf_feasible_yn[yi]
            tier = 1
            if not valid.any():
                # Tier 3: drop cf_feasible (ratchet still respected). Year flagged
                # as physically infeasible — mix reported cost is right, but it
                # can't actually be built by the available clean_firm tranches.
                valid = ratchet_mask_y & pmask[yi]
                winner_feasible[yi] = False
                tier = 2
                if not valid.any():
                    # Tier 4 emergency: drop ratchet too. Flag for review.
                    valid = pmask[yi].copy()
                    ratchet_violated[yi] = True
                    tier = 3
                    if not valid.any():
                        valid = np.ones(n_mixes, dtype=bool)

        masked_score = np.where(valid, row_score, np.inf)
        w = int(np.argmin(masked_score))
        winners[yi] = w
        cascade_tier_y[yi] = tier
        # If we landed via Tier-1/2, winner_feasible still reflects the mask's
        # cf_feasible membership; reassert it explicitly for safety.
        if winner_feasible[yi]:
            winner_feasible[yi] = bool(cf_feasible_yn[yi, w])

        # ── Phase B foresight-preview capture (read-only) ───────────────
        # Runs on the same `valid` mask and `row_score` that myopic argmin
        # used above. Floors update below is unaffected.
        if _foresight_active and yi <= _foresight_endpoint_yi:
            _fy = int(YEARS[yi])
            _fw_weight = max(
                0.0,
                (_foresight_endpoint_year_val - _fy)
                / (_foresight_endpoint_year_val - BASE_YEAR),
            )
            _proj = _project_shares_to_endpoint(
                shares_nr, non_cf, cf_share, tg, demand_vec,
                yi=yi, endpoint_yi=_foresight_endpoint_yi,
            )
            _m_diff = _proj[w] - _foresight_target_shares
            _m_dist = float(np.sqrt(np.sum(_m_diff * _m_diff)))
            _lambda_rows = {}
            for _lam in _FORESIGHT_LAMBDAS:
                _f_score = _score_year_with_endpoint(
                    row_score, _proj, _foresight_target_shares,
                    _lam, _fw_weight)
                _f_masked = np.where(valid, _f_score, np.inf)
                _f_w = int(np.argmin(_f_masked))
                _fd = _proj[_f_w] - _foresight_target_shares
                _fd_dist = float(np.sqrt(np.sum(_fd * _fd)))
                _lambda_rows[f"{_lam:g}"] = {
                    "mix_idx": _f_w,
                    "score_usd": float(_f_score[_f_w]),
                    "share_distance": round(_fd_dist, 6),
                }
            _foresight_preview_rows.append({
                "year": _fy,
                "feasible_candidates": int(valid.sum()),
                "myopic_winner_idx": int(w),
                "myopic_score_usd": float(row_score[w]),
                "share_distance_to_target_at_myopic_winner": round(_m_dist, 6),
                "foresight_winners_by_lambda": _lambda_rows,
            })

        # Update floors from the winning mix (monotone non-decreasing).
        floor_twh_r    = np.maximum(floor_twh_r, target_twh_nr[w])
        # P1: total-CF floor stays frozen at existing 2025 CF TWh (= share
        # floor that declines as demand grows). No new-CF tranche unlocks
        # beyond the fixed UPRATE_CAP_TWH, so ratcheting the floor would
        # force later years to carry prior winners' peak absolute CF with no
        # buildable tranche to cover the gap. P1a keeps the ratchet.
        if cfg.pathway != '1':
            floor_twh_cf = max(floor_twh_cf, float(target_twh_cf_n[w]))
        floor_uprate   = max(floor_uprate,   float(uprate_twh_yn[yi, w]))
        floor_geo      = max(floor_geo,      float(geo_twh_yn[yi, w]))
        floor_nuke_new = max(floor_nuke_new, float(nuke_new_twh_yn[yi, w]))
        floor_ccs      = max(floor_ccs,      float(ccs_twh_yn[yi, w]))

    # Post-endpoint freeze (mirrors solve_pathway_with_foresight memo §6.5):
    # hold the endpoint winner forward through 2050 so _finalize_run and
    # downstream helpers receive full N_YEARS arrays. The solver does not
    # simulate post-endpoint argmins — per-endpoint trajectories stop at
    # _myopic_endpoint_year (2040/2045/2049/2050 for 90/95/99/99.9%).
    _endpoint_w = int(winners[_myopic_endpoint_yi])
    for yi in range(_myopic_endpoint_yi + 1, N_YEARS):
        winners[yi] = _endpoint_w
        winner_feasible[yi] = winner_feasible[_myopic_endpoint_yi]
        ratchet_violated[yi] = False
        cascade_tier_y[yi] = -1  # sentinel: post-endpoint, not simulated

    # Phase B: realized clean_pct from winners → re-run existing_gas_vec, apply ratchet
    achieved_cfe = score_vec[winners]
    existing_gas_vec_real = existing_gas_available_vec(cfg.iso, achieved_cfe, cfg.demand_growth_level)
    gas_required_real = gas_sizing_matrix(annual_mwh_vec, existing_gas_vec_real,
                                          resid_arr, gas_raw_2025,
                                          existing_base, gaf)
    cum_new_gas = gas_required_real[np.arange(N_YEARS), winners]
    active_fleet = np.maximum.accumulate(cum_new_gas)
    fleet_size_mw = float(active_fleet.max())
    peak_year = int(YEARS[int(np.argmax(active_fleet))])

    # gas fleet CF (energy-balance)
    total_gas_mw = float(pc.EXISTING_GAS_CAPACITY_MW[cfg.iso]) + active_fleet
    gas_gen_mwh = np.maximum(0.0, 1.0 - achieved_cfe / 100.0) * demand_vec * 1.0e6
    gas_cf = np.where(total_gas_mw > 1e-6,
                      np.minimum(1.0, gas_gen_mwh / (total_gas_mw * HOURS_PER_YEAR)), 0.0)

    # Existing gas used = min(gas_needed_at_winner, existing_gas_available).
    gap_mwh_winner = resid_arr[winners] * annual_mwh_vec
    gas_raw_winner = gap_mwh_winner / max(1e-9, gaf)
    gas_needed_winner = np.maximum(0.0, existing_base + gas_raw_winner - gas_raw_2025)
    existing_gas_used = np.minimum(existing_gas_vec_real, gas_needed_winner)
    clean_arr_diag = ef['clean_peak_hour_mw']
    result = _finalize_run(cfg, ef, winners, peak_vec, demand_vec,
                           achieved_cfe, existing_gas_vec_real, existing_gas_used,
                           cum_new_gas, active_fleet, gas_cf, fleet_size_mw, peak_year,
                           clean_arr_diag, winner_feasible, ratchet_violated)
    result.cascade_tier_y = cascade_tier_y
    if _foresight_active:
        result.foresight_preview = {
            "iso": cfg.iso,
            "pathway": cfg.pathway,
            "endpoint_pct": float(cfg.endpoint_pct),
            "endpoint_year": int(_foresight_endpoint_year_val),
            "schema_note": ("Phase B diagnostic sidecar. Not a pipeline "
                            "artifact. target_shares sourced from step 2.2A "
                            "cost-optimizer cache per reliability_tax/"
                            "README.md:44 — see target_shares_method."),
            "target_shares_method":
                "step_2_2a_cost_optimal_at_threshold_year",
            "target_shares": {
                k: round(float(v), 6)
                for k, v in zip(
                    _FORESIGHT_SHARE_KEYS, _foresight_target_shares)
            },
            "lambda_values": [float(x) for x in _FORESIGHT_LAMBDAS],
            "per_year": _foresight_preview_rows,
        }
    return result


def solve_pathway_with_foresight(cfg: 'RunConfig') -> PathwayRunResult:
    """Endpoint-aware solver (memo §3 option iv, algorithm §4(b)).

    Year-by-year argmin steered by a lookahead penalty

        score_y[m] = row_score[m] + λ · w(y) · ‖shares[m] − target‖² · C

    where C = demand_MWh(endpoint_year) × FORESIGHT_REF_LCOE_USD_PER_MWH puts
    the penalty on the same USD scale as row_score (DIMENSIONAL CONVENTION
    block above), w(y) = max(0, (endpoint_year − y)/(endpoint_year − BASE_YEAR))
    decays linearly to zero at the endpoint, and `target` is the step 2.2A
    cost-optimal share vector cached by `_compute_endpoint_target_shares`.

    Year loop runs [0, endpoint_yi] inclusive — foresight does not simulate
    post-endpoint behavior (memo §6.5). Post-endpoint winners are frozen to
    the endpoint-year winner so `_finalize_run` receives full-length arrays
    unchanged; Phase E's post-processor truncates costs at `endpoint_year`.

    Degenerate at λ = 0 to solve_pathway(cfg) bit-identically on the
    [BASE_YEAR, endpoint_year] prefix — this is the Task 6 regression.
    """
    if cfg.endpoint_year is None:
        raise ValueError("solve_pathway_with_foresight: cfg.endpoint_year is required")
    if cfg.endpoint_target_shares is None:
        raise ValueError("solve_pathway_with_foresight: cfg.endpoint_target_shares is required")

    # Pool loader: concatenated across all step 2.1 threshold bands, then
    # filtered by max-share-per-resource caps taken across the 4 canonical
    # endpoints' step-2.2A targets (+ slack). Non-overlapping band layout
    # means per-band load_ef_mixes under-samples the qualifying pool — that
    # was the Phase D pool-pathology root cause (LESSONS.md #22). Foresight
    # keeps the dense (Y, N) path because the filtered pool stays in the
    # ~40k–370k range across ISOs, tractable for dense tensor math.
    ef = load_ef_pool(cfg.iso, filter_to_endpoints=True, cfg_for_filter=cfg)
    n_mixes = len(ef['hourly_match_score'])

    peak_vec = peak_demand_vec(cfg.iso, cfg.demand_growth_level)
    demand_vec = demand_twh_vec(cfg.iso, cfg.demand_growth_level)
    annual_mwh_vec = demand_vec * 1.0e6
    score_vec = ef['hourly_match_score']
    resid_arr = ef['resid_norm_p9997'].astype(np.float64)
    gas_raw_2025 = _GAS_RAW_2025_BY_ISO[cfg.iso]
    existing_base = float(pc.EXISTING_GAS_CAPACITY_MW[cfg.iso])
    gaf = float(pc.GAS_AVAILABILITY_FACTOR[cfg.iso])

    cfe_targets = np.array([_cfe_target_for_year(y, cfg.endpoint_pct) for y in YEARS])
    pmask = _pathway_mask(ef, cfg)
    cfe_mask = score_vec[None, :] >= cfe_targets[:, None] - 0.5

    existing_gas_vec = existing_gas_available_vec(cfg.iso, cfe_targets, cfg.demand_growth_level)
    gas_required = gas_sizing_matrix(annual_mwh_vec, existing_gas_vec, resid_arr,
                                     gas_raw_2025, existing_base, gaf)
    fuel_unit = pc.WHOLESALE_PRICES[cfg.iso] + pc.FUEL_ADJUSTMENTS[cfg.iso]['Medium']
    gas_fuel = gas_fuel_cost_matrix(ef, demand_vec, fuel_unit)

    ratchet_fleet_y = np.maximum.accumulate(gas_required, axis=0)
    new_gas_capex_y = (pc.NEW_CCGT_COST_KW_YR[cfg.iso] * 1000.0) * ratchet_fleet_y
    new_gas_fom_y   = (pc.NEW_CCGT_FOM_KW_YR[cfg.iso]  * 1000.0) * ratchet_fleet_y
    existing_fom = (float(pc.EXISTING_GAS_CAPACITY_MW[cfg.iso])
                    * pc.EXISTING_GAS_FOM_KW_YR[cfg.iso] * 1000.0)

    non_cf = tuple(r for r in _RESOURCE_COLS if r != 'clean_firm')
    R = len(non_cf)
    tx_r = np.array(
        [_resource_tx(cfg.iso, r, cfg.tx_level) for r in non_cf],
        dtype=np.float64,
    )
    lcoe_yr = np.empty((N_YEARS, R), dtype=np.float64)
    for ri, r in enumerate(non_cf):
        for yi, y in enumerate(YEARS):
            lcoe_yr[yi, ri] = _resource_lcoe_year(cfg.iso, r, y, cfg)
    delivered_yr = lcoe_yr + tx_r[None, :]
    shares_nr = np.stack([ef[r] for r in non_cf], axis=1).astype(np.float64) / 100.0

    tg = decompose_clean_firm_tranches(
        ef['clean_firm'], cfg.iso, cfg.pathway, annual_mwh_vec, cfg)
    uprate_twh_yn   = tg['uprate_twh']
    geo_twh_yn      = tg['geo_twh']
    nuke_new_twh_yn = tg['nuke_new_twh']
    ccs_twh_yn      = tg['ccs_twh']
    cf_feasible_yn  = tg['feasible']
    uprate_eff      = tg['uprate_eff_lcoe']
    geo_eff         = tg['geo_eff_lcoe']
    nuke_new_eff    = tg['nuke_new_eff_lcoe']
    ccs_eff         = tg['ccs_eff_lcoe']

    storage_lcoe_keys = {
        'battery_dispatch_pct': 'battery', 'battery8_dispatch_pct': 'battery8',
        'ldes_dispatch_pct': 'ldes', 'h2_dispatch_pct': 'h2',
    }
    storage_cost_yn = np.zeros((N_YEARS, n_mixes), dtype=np.float64)
    for sc, lk in storage_lcoe_keys.items():
        share = ef[sc] / 100.0
        if share.sum() == 0:
            continue
        unit = pc.LCOE_TABLES[lk]['Medium'][cfg.iso]
        storage_cost_yn += demand_vec[:, None] * 1.0e6 * share[None, :] * unit / 1.0e6

    gas_fixed_y = new_gas_capex_y + new_gas_fom_y + gas_fuel + existing_fom

    grid = pc.GRID_MIX_SHARES[cfg.iso]
    base_demand = float(pc.REGIONAL_DEMAND_TWH[cfg.iso])
    floor_twh_r    = np.array(
        [base_demand * grid.get(r, 0.0) / 100.0 for r in non_cf],
        dtype=np.float64,
    )
    floor_twh_cf   = base_demand * grid.get('clean_firm', 0.0) / 100.0
    cf_share       = ef['clean_firm'].astype(np.float64) / 100.0
    floor_uprate   = 0.0
    floor_geo      = 0.0
    floor_nuke_new = 0.0
    floor_ccs      = 0.0

    winners           = np.empty(N_YEARS, dtype=np.int64)
    winner_feasible   = np.ones(N_YEARS, dtype=bool)
    ratchet_violated  = np.zeros(N_YEARS, dtype=bool)
    # Cascade tier per year: 0 = Tier-1 (full feasibility set),
    # 1 = Tier-2 (relaxed CFE ±5), 2 = Tier-3 (dropped cf_feasible),
    # 3 = Tier-4 emergency (dropped ratchet). Used for schema-v3
    # cascade_activations + per-year diagnostic sidecar.
    cascade_tier_y    = np.zeros(N_YEARS, dtype=np.int32)

    # Foresight inputs.
    endpoint_year = int(cfg.endpoint_year)
    if endpoint_year not in YEARS:
        raise ValueError(f"endpoint_year={endpoint_year} not in YEARS range")
    endpoint_yi = YEARS.index(endpoint_year)
    target_shares = np.asarray(cfg.endpoint_target_shares, dtype=np.float64)
    if target_shares.shape != (len(_FORESIGHT_SHARE_KEYS),):
        raise ValueError(
            f"endpoint_target_shares has shape {target_shares.shape}, "
            f"expected ({len(_FORESIGHT_SHARE_KEYS)},)"
        )
    lam = float(cfg.foresight_lambda or 0.0)
    # DIMENSIONAL CONVENTION (see block above _compute_endpoint_target_shares):
    # C = endpoint_year_demand_MWh × FORESIGHT_REF_LCOE_USD_PER_MWH. This is
    # the USD-scale normalizer; λ stays in the {0, 0.05, 0.15, 0.5, 1.5} band.
    endpoint_demand_mwh = float(demand_vec[endpoint_yi]) * 1.0e6
    foresight_C = endpoint_demand_mwh * FORESIGHT_REF_LCOE_USD_PER_MWH
    horizon_denom = float(endpoint_year - BASE_YEAR)

    # Per-year diagnostic rows (Phase E consumes; schema §6.4).
    per_year_diag: list = []

    RATCHET_TOL_TWH = 1.0e-6
    # Year loop [0, endpoint_yi] inclusive — foresight-only truncation per §6.5.
    for yi in range(endpoint_yi + 1):
        target_twh_nr = shares_nr * float(demand_vec[yi])
        ratchet_nonCF = np.all(
            target_twh_nr >= (floor_twh_r[None, :] - RATCHET_TOL_TWH), axis=1)
        target_twh_cf_n = cf_share * float(demand_vec[yi])
        ratchet_cf     = target_twh_cf_n >= (floor_twh_cf - RATCHET_TOL_TWH)
        ratchet_uprate = uprate_twh_yn[yi]   >= (floor_uprate   - RATCHET_TOL_TWH)
        ratchet_geo    = geo_twh_yn[yi]      >= (floor_geo      - RATCHET_TOL_TWH)
        ratchet_nuke   = nuke_new_twh_yn[yi] >= (floor_nuke_new - RATCHET_TOL_TWH)
        ratchet_ccs    = ccs_twh_yn[yi]      >= (floor_ccs      - RATCHET_TOL_TWH)
        ratchet_mask_y = (ratchet_nonCF & ratchet_cf & ratchet_uprate
                          & ratchet_geo & ratchet_nuke & ratchet_ccs)

        row_score = _score_year_sunk_cost(
            target_twh_nr=target_twh_nr,
            floor_twh_r=floor_twh_r,
            delivered_yr_yi=delivered_yr[yi],
            uprate_twh_n=uprate_twh_yn[yi],
            geo_twh_n=geo_twh_yn[yi],
            nuke_new_twh_n=nuke_new_twh_yn[yi],
            ccs_twh_n=ccs_twh_yn[yi],
            floor_uprate=floor_uprate, floor_geo=floor_geo,
            floor_nuke_new=floor_nuke_new, floor_ccs=floor_ccs,
            uprate_eff_yi=uprate_eff[yi], geo_eff_yi=geo_eff[yi],
            nuke_new_eff_yi=nuke_new_eff[yi], ccs_eff_yi=ccs_eff[yi],
            storage_cost_n=storage_cost_yn[yi],
            gas_fixed_n=gas_fixed_y[yi],
        )

        # Endpoint-aware penalty (memo §5.2 with USD-scale normalizer).
        year = YEARS[yi]
        weight = max(0.0, (endpoint_year - year) / horizon_denom)
        shares_to_end = _project_shares_to_endpoint(
            shares_nr, non_cf, cf_share, tg, demand_vec,
            yi=yi, endpoint_yi=endpoint_yi,
        )
        diff = shares_to_end - target_shares[None, :]
        dist_sq = np.sum(diff * diff, axis=1)
        penalty = (lam * weight * foresight_C) * dist_sq
        foresight_score = row_score + penalty

        valid = ratchet_mask_y & pmask[yi] & cfe_mask[yi] & cf_feasible_yn[yi]
        tier = 0
        if not valid.any():
            relaxed_cfe = score_vec >= cfe_targets[yi] - 5.0
            valid = ratchet_mask_y & pmask[yi] & relaxed_cfe & cf_feasible_yn[yi]
            tier = 1
            if not valid.any():
                valid = ratchet_mask_y & pmask[yi]
                winner_feasible[yi] = False
                tier = 2
                if not valid.any():
                    valid = pmask[yi].copy()
                    ratchet_violated[yi] = True
                    tier = 3
                    if not valid.any():
                        valid = np.ones(n_mixes, dtype=bool)

        masked_score = np.where(valid, foresight_score, np.inf)
        w = int(np.argmin(masked_score))
        winners[yi] = w
        cascade_tier_y[yi] = tier
        if winner_feasible[yi]:
            winner_feasible[yi] = bool(cf_feasible_yn[yi, w])

        # Per-year diagnostic: myopic winner (no penalty) for head-to-head,
        # penalty/sunk components at chosen mix, share-distance, cascade tier.
        myopic_masked = np.where(valid, row_score, np.inf)
        myopic_w = int(np.argmin(myopic_masked))
        w_diff = shares_to_end[w] - target_shares
        per_year_diag.append({
            'year': int(year),
            'feasible_candidates': int(valid.sum()),
            'cascade_tier_activated': int(tier),
            'myopic_winner_idx': myopic_w,
            'foresight_winner_idx': w,
            'sunk_cost_component_usd': float(row_score[w]),
            'penalty_component_usd': float(penalty[w]),
            'share_distance_to_endpoint': float(np.sqrt(np.sum(w_diff * w_diff))),
            'lookahead_weight': float(weight),
            # Rank of foresight winner among myopic-sorted feasible candidates
            # (Phase E diagnostic for "how far from cheapest did we move?").
            'selected_mix_candidate_rank_among_feasible': int(
                (row_score[valid] < row_score[w]).sum()
            ),
        })

        floor_twh_r    = np.maximum(floor_twh_r, target_twh_nr[w])
        if cfg.pathway != '1':
            floor_twh_cf = max(floor_twh_cf, float(target_twh_cf_n[w]))
        floor_uprate   = max(floor_uprate,   float(uprate_twh_yn[yi, w]))
        floor_geo      = max(floor_geo,      float(geo_twh_yn[yi, w]))
        floor_nuke_new = max(floor_nuke_new, float(nuke_new_twh_yn[yi, w]))
        floor_ccs      = max(floor_ccs,      float(ccs_twh_yn[yi, w]))

    # Post-endpoint freeze: hold the endpoint winner forward so _finalize_run
    # receives full N_YEARS arrays unchanged. Phase E post-processor truncates
    # cost accumulation at endpoint_year (memo §6.5).
    endpoint_w = int(winners[endpoint_yi])
    for yi in range(endpoint_yi + 1, N_YEARS):
        winners[yi] = endpoint_w
        winner_feasible[yi] = winner_feasible[endpoint_yi]
        ratchet_violated[yi] = False
        cascade_tier_y[yi] = -1  # sentinel: post-endpoint, not simulated

    achieved_cfe = score_vec[winners]
    existing_gas_vec_real = existing_gas_available_vec(cfg.iso, achieved_cfe, cfg.demand_growth_level)
    gas_required_real = gas_sizing_matrix(annual_mwh_vec, existing_gas_vec_real,
                                          resid_arr, gas_raw_2025,
                                          existing_base, gaf)
    cum_new_gas = gas_required_real[np.arange(N_YEARS), winners]
    active_fleet = np.maximum.accumulate(cum_new_gas)
    fleet_size_mw = float(active_fleet.max())
    peak_year = int(YEARS[int(np.argmax(active_fleet))])

    total_gas_mw = float(pc.EXISTING_GAS_CAPACITY_MW[cfg.iso]) + active_fleet
    gas_gen_mwh = np.maximum(0.0, 1.0 - achieved_cfe / 100.0) * demand_vec * 1.0e6
    gas_cf = np.where(total_gas_mw > 1e-6,
                      np.minimum(1.0, gas_gen_mwh / (total_gas_mw * HOURS_PER_YEAR)), 0.0)

    gap_mwh_winner = resid_arr[winners] * annual_mwh_vec
    gas_raw_winner = gap_mwh_winner / max(1e-9, gaf)
    gas_needed_winner = np.maximum(0.0, existing_base + gas_raw_winner - gas_raw_2025)
    existing_gas_used = np.minimum(existing_gas_vec_real, gas_needed_winner)
    clean_arr_diag = ef['clean_peak_hour_mw']
    result = _finalize_run(cfg, ef, winners, peak_vec, demand_vec,
                           achieved_cfe, existing_gas_vec_real, existing_gas_used,
                           cum_new_gas, active_fleet, gas_cf, fleet_size_mw, peak_year,
                           clean_arr_diag, winner_feasible, ratchet_violated)
    result.cascade_tier_y = cascade_tier_y
    result.foresight_metadata = {
        'solver_mode': 'foresight',
        'endpoint_year': endpoint_year,
        'endpoint_yi': int(endpoint_yi),
        'foresight_lambda': lam,
        'foresight_C_usd': float(foresight_C),
        'target_shares': {
            k: float(v) for k, v in zip(_FORESIGHT_SHARE_KEYS, target_shares)
        },
        'cascade_tier_y': [int(t) for t in cascade_tier_y],
        'cascade_activations': int((cascade_tier_y[:endpoint_yi + 1] > 0).sum()),
        'per_year': per_year_diag,
    }
    return result


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
    """Existing-fleet entries + per-year delta vintages with locked LCOE.

    Every vintage carries an ABSOLUTE TWh claim at its locked LCOE — no demand
    scaling anywhere. Baseline existing vintages sit at BASE_YEAR with
    twh = base_demand × grid_share / 100 and locked_lcoe = 0. Each new year,
    the winner's year-y absolute-TWh target per resource is compared against
    a running-max floor; the positive delta (if any) becomes a new vintage at
    that year's LCOE. Floors are absolute TWh, so demand growth at constant
    share mechanically produces NEW vintages covering the gap (existing
    vintages don't scale with demand — they stay frozen at their locked TWh).
    Clean_firm is disaggregated: the baseline `clean_firm_existing` vintage
    at BASE_YEAR sits at locked_lcoe=0, then each year any positive tranche
    delta (uprate / geo / nuke_new / ccs) becomes its own tranche-tagged
    vintage with the year's effective LCOE.
    """
    led = VintageLedger()
    base_demand = float(pc.REGIONAL_DEMAND_TWH[cfg.iso])
    grid = pc.GRID_MIX_SHARES[cfg.iso]
    g = pc.DEMAND_GROWTH_RATES[cfg.iso][cfg.demand_growth_level]

    # ── Base-year existing vintages (absolute TWh, locked_lcoe = 0) ──
    for r, share in grid.items():
        if share <= 0:
            continue
        key = 'clean_firm_existing' if r == 'clean_firm' else r
        led.add(Vintage(resource=key, cod_year=BASE_YEAR,
                        twh_per_year=base_demand * share / 100.0,
                        locked_lcoe=0.0, tx_adder=0.0, retire_year=None))

    # ── Per-year delta vintages (non-clean_firm resources) — absolute-TWh floor ──
    non_cf_resources = tuple(r for r in _RESOURCE_COLS if r != 'clean_firm')
    floor_twh = {r: base_demand * grid.get(r, 0.0) / 100.0 for r in non_cf_resources}
    for yi, year in enumerate(YEARS):
        w = int(winners[yi])
        demand_y_twh = base_demand * (1.0 + g) ** yi
        for r in non_cf_resources:
            target_twh = demand_y_twh * float(ef[r][w]) / 100.0
            delta = target_twh - floor_twh[r]
            if delta > 0.005:  # 5 GWh floor
                lcoe = _resource_lcoe_year(cfg.iso, r, year, cfg)
                tx = _resource_tx(cfg.iso, r, cfg.tx_level)
                led.add(Vintage(resource=r, cod_year=year,
                                twh_per_year=delta,
                                locked_lcoe=lcoe, tx_adder=tx, retire_year=None))
            if target_twh > floor_twh[r]:
                floor_twh[r] = target_twh

    # ── Per-year delta vintages (clean_firm tranches) — absolute-TWh floor ──
    # Decompose only the per-year winner's cf_pct (Y values) instead of the
    # full (Y, N) pool grid — scales with years, not pool size. Safe on
    # multi-million-mix full pools where the dense (Y, N) allocation would
    # OOM on ERCOT (8.7M) / CAISO (21M).
    annual_mwh_vec = np.array(
        [base_demand * (1.0 + g) ** i * 1.0e6 for i in range(N_YEARS)],
        dtype=np.float64,
    )
    winner_cf_pct = ef['clean_firm'][winners]                          # (Y,)
    tg = decompose_clean_firm_tranches(
        winner_cf_pct, cfg.iso, cfg.pathway, annual_mwh_vec, cfg)
    idx = np.arange(N_YEARS)
    tranche_series = (
        ('uprate',   tg['uprate_twh'][idx, idx],   tg['uprate_eff_lcoe']),
        ('geo',      tg['geo_twh'][idx, idx],      tg['geo_eff_lcoe']),
        ('nuke_new', tg['nuke_new_twh'][idx, idx], tg['nuke_new_eff_lcoe']),
        ('ccs',      tg['ccs_twh'][idx, idx],      tg['ccs_eff_lcoe']),
    )
    for tranche_name, twh_y, eff_lcoe_y in tranche_series:
        floor = 0.0
        key = _CLEAN_FIRM_TRANCHE_VINTAGE_KEY[tranche_name]
        for yi, year in enumerate(YEARS):
            cur = float(twh_y[yi])
            delta = cur - floor
            if delta > 0.005:  # 5 GWh floor
                led.add(Vintage(
                    resource=key, cod_year=year,
                    twh_per_year=delta,
                    # TX is baked into eff_lcoe for geo/nuke_new/ccs; uprate
                    # has TX=0 by design — the vintage's delivered cost is
                    # (locked_lcoe + 0) × twh.
                    locked_lcoe=float(eff_lcoe_y[yi]), tx_adder=0.0,
                    retire_year=None,
                ))
            if cur > floor:
                floor = cur
    return led


def _finalize_run(cfg: 'RunConfig', ef: dict, winners: np.ndarray,
                  peak_vec, demand_vec, achieved_cfe, existing_gas_vec_real,
                  existing_gas_used, cum_new_gas, active_fleet, gas_cf,
                  fleet_size_mw, peak_year, clean_arr,
                  winner_feasible: np.ndarray,
                  ratchet_violated: np.ndarray) -> PathwayRunResult:
    ledger = _build_ledger(ef, winners, cfg)
    priced_curt_vec = _priced_vre_curtailment_vec(ef, winners, demand_vec, cfg)
    new_gas_annualized = active_fleet * pc.NEW_CCGT_COST_KW_YR[cfg.iso] * 1000.0
    new_gas_fom_y_out  = active_fleet * pc.NEW_CCGT_FOM_KW_YR[cfg.iso]  * 1000.0
    existing_fom = float(pc.EXISTING_GAS_CAPACITY_MW[cfg.iso]) * pc.EXISTING_GAS_FOM_KW_YR[cfg.iso] * 1000.0
    fuel_unit = pc.WHOLESALE_PRICES[cfg.iso] + pc.FUEL_ADJUSTMENTS[cfg.iso]['Medium']
    gas_fuel_y = np.maximum(0.0, 1.0 - achieved_cfe / 100.0) * demand_vec * 1.0e6 * fuel_unit
    # Gross operating cost: sum of active vintages' absolute twh × locked LCOE.
    # Vintages carry absolute TWh claims — no demand scaling at readout time.
    annual_rows = []
    for yi, year in enumerate(YEARS):
        active_v = [v for v in ledger.entries if v.cod_year <= year and (v.retire_year is None or v.retire_year > year)]
        gross_op = sum(v.twh_per_year * 1.0e6 * (v.locked_lcoe + v.tx_adder) for v in active_v)
        new_gas_annualized_plus_fom = float(new_gas_annualized[yi] + new_gas_fom_y_out[yi])
        net = gross_op + new_gas_annualized_plus_fom + existing_fom + gas_fuel_y[yi] + priced_curt_vec[yi]
        annual_rows.append({
            'year': year, 'demand_twh': round(float(demand_vec[yi]), 3),
            'gross_operating_usd': round(gross_op, 2),
            'new_gas_annualized_capex_fom_usd': round(new_gas_annualized_plus_fom, 2),
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
        'new_gas_fom_usd': round(float(new_gas_fom_y_out.sum()), 2),
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
        'endpoint_year': int(_foresight_endpoint_year(cfg.endpoint_pct)),
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
    # feasibility.physical: True iff every year's winning mix could be physically
    # filled by the available clean_firm tranches AND the sequential ratchet was
    # never violated. False when either:
    #   (a) the argmin fell through to Tier 3 (dropped cf_feasible) because no
    #       ratchet-respecting + cfe-respecting mix could actually be built —
    #       reported cost is right but clean_firm target exceeds tranche budget;
    #   (b) the argmin fell through to Tier 4 emergency (dropped ratchet) —
    #       the sequential-build model couldn't keep a prior commitment.
    infeasible_years     = [int(YEARS[i]) for i in range(N_YEARS)
                            if not bool(winner_feasible[i])]
    ratchet_violated_yrs = [int(YEARS[i]) for i in range(N_YEARS)
                            if bool(ratchet_violated[i])]
    notes = []
    if infeasible_years:
        notes.append("clean_firm target exceeds available tranche budget in: "
                     f"{infeasible_years}")
    if ratchet_violated_yrs:
        notes.append("ratchet relaxed (Tier-4 emergency) in: "
                     f"{ratchet_violated_yrs}")
    feasibility = {
        'physical': not infeasible_years and not ratchet_violated_yrs,
        'notes': notes,
    }

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
    grid = pc.GRID_MIX_SHARES[cfg.iso]
    base_demand = float(pc.REGIONAL_DEMAND_TWH[cfg.iso])
    # Prebuild the clean_firm tranche grid once so each year's new_vintages
    # can emit per-tranche lines (mirrors _build_ledger). Decompose only the
    # winners' cf_pct (Y values) so full-pool runs don't OOM on the (Y, N)
    # allocation.
    annual_mwh_vec = r.demand_vec * 1.0e6
    winner_cf_pct = r.ef['clean_firm'][r.winners]                      # (Y,)
    tg_series = decompose_clean_firm_tranches(
        winner_cf_pct, cfg.iso, cfg.pathway, annual_mwh_vec, cfg)
    idx = np.arange(N_YEARS)
    tranche_twh_by_name = {
        'uprate':   tg_series['uprate_twh'][idx, idx],
        'geo':      tg_series['geo_twh'][idx, idx],
        'nuke_new': tg_series['nuke_new_twh'][idx, idx],
        'ccs':      tg_series['ccs_twh'][idx, idx],
    }
    tranche_eff_by_name = {
        'uprate':   tg_series['uprate_eff_lcoe'],
        'geo':      tg_series['geo_eff_lcoe'],
        'nuke_new': tg_series['nuke_new_eff_lcoe'],
        'ccs':      tg_series['ccs_eff_lcoe'],
    }
    prev_tranche_twh = {name: 0.0 for name in tranche_twh_by_name}
    non_cf_resources = tuple(res for res in _RESOURCE_COLS if res != 'clean_firm')
    # Mirror _build_ledger's absolute-TWh floor: new vintages price the
    # incremental TWh above the running-max target for each resource at
    # year-y LCOE. Initial floor = baseline grid-share TWh at base demand.
    floor_twh = {res: base_demand * grid.get(res, 0.0) / 100.0
                 for res in non_cf_resources}
    for yi, year in enumerate(YEARS):
        w = int(r.winners[yi])
        demand_y_twh = float(r.demand_vec[yi])
        new_v = []
        for res in non_cf_resources:
            target_twh = demand_y_twh * float(r.ef[res][w]) / 100.0
            delta = target_twh - floor_twh[res]
            if delta > 0.005:  # 5 GWh floor
                new_v.append({
                    'resource': res,
                    'twh_per_year': round(delta, 4),
                    'locked_lcoe_usd_per_mwh': round(_resource_lcoe_year(cfg.iso, res, year, cfg), 2),
                    'tx_adder_usd_per_mwh': round(_resource_tx(cfg.iso, res, cfg.tx_level), 2),
                })
            if target_twh > floor_twh[res]:
                floor_twh[res] = target_twh
        for tranche_name, twh_y in tranche_twh_by_name.items():
            cur = float(twh_y[yi])
            delta = cur - prev_tranche_twh[tranche_name]
            if delta > 0.005:  # 5 GWh floor
                new_v.append({
                    'resource': _CLEAN_FIRM_TRANCHE_VINTAGE_KEY[tranche_name],
                    'twh_per_year': round(delta, 4),
                    'locked_lcoe_usd_per_mwh': round(
                        float(tranche_eff_by_name[tranche_name][yi]), 2),
                    # TX is folded into locked_lcoe for geo/nuke_new/ccs; uprate
                    # is TX-free by design — so the adder is always 0 here.
                    'tx_adder_usd_per_mwh': 0.0,
                })
            if cur > prev_tranche_twh[tranche_name]:
                prev_tranche_twh[tranche_name] = cur
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


def serialize_run_result_v3(r: PathwayRunResult) -> dict:
    """Schema v3 emitter — foresight runs only (memo §6.4). Myopic runs
    continue to emit the v2 shape via serialize_run_result. The v3 payload
    is the v2 payload plus:
      - top-level `schema_version: "3.0.0"` (breaking — v2 is int, v3 is str)
      - `config.solver_mode`, `config.foresight_lambda`,
        `config.endpoint_year`, `config.endpoint_target_shares`
      - `headline` gains:
        `foresight_penalty_vs_myopic`, `p1_penalty_vs_foresight`,
        `p1_reliability_tax_premium` (all null here — require paired runs,
        filled by Phase E post-processor), `time_to_90pct`,
        `cascade_activations`
      - `foresight_diagnostics` block with per-year rows and cascade tier
        vector
    """
    base = serialize_run_result(r)  # v2 shape
    cfg = r.config
    meta = r.foresight_metadata or {}

    base['schema_version'] = '3.0.0'
    base['config']['solver_mode'] = cfg.solver_mode
    base['config']['foresight_lambda'] = (
        float(cfg.foresight_lambda) if cfg.foresight_lambda is not None else None
    )
    base['config']['endpoint_year'] = (
        int(cfg.endpoint_year) if cfg.endpoint_year is not None else None
    )
    base['config']['endpoint_target_shares'] = (
        {k: round(float(v), 6) for k, v in meta.get('target_shares', {}).items()}
        if meta else None
    )

    # time_to_90pct: first year where achieved_cfe crosses 90.0 within the
    # simulated prefix [BASE_YEAR, endpoint_year]. None if never crossed.
    endpoint_yi = meta.get('endpoint_yi')
    time_to_90 = None
    if endpoint_yi is not None:
        pfx = r.achieved_cfe_pct[: endpoint_yi + 1]
        hits = np.where(pfx >= 90.0 - 1e-9)[0]
        if hits.size:
            time_to_90 = int(YEARS[int(hits[0])])

    cascade_activations = int(meta.get('cascade_activations', 0)) if meta else None

    base['headline'] = {
        **base['headline'],
        # Pairwise metrics require Phase-E's external reference runs; emit null.
        'foresight_penalty_vs_myopic': None,
        'p1_penalty_vs_foresight': None,
        'p1_reliability_tax_premium': None,
        'time_to_90pct': time_to_90,
        'cascade_activations': cascade_activations,
    }

    # Diagnostic sidecar block (memo §6.4 — "Diagnostic sidecar (new)").
    # Inlined here rather than emitted as a separate file so dashboard
    # consumers see one schema-checked payload per foresight run.
    base['foresight_diagnostics'] = {
        'endpoint_year': meta.get('endpoint_year'),
        'foresight_C_usd': meta.get('foresight_C_usd'),
        'cascade_tier_y': meta.get('cascade_tier_y'),
        'per_year': meta.get('per_year', []),
    }
    return base


# ─── CLI ─────────────────────────────────────────────────────────────────────

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description='Pathway optimizer v2')
    ap.add_argument('--iso', choices=ISOS)
    ap.add_argument('--pathway', choices=PATHWAYS)
    ap.add_argument('--endpoint', type=float, choices=list(ENDPOINT_TO_THRESHOLD.keys()))
    ap.add_argument('--all', action='store_true', help='Run full 7×5×4 sweep')
    return ap


def _run_one(
    iso: str,
    pathway: str,
    endpoint: float,
    solver_mode: str = 'myopic',
    foresight_lambda: Optional[float] = None,
    endpoint_year: Optional[int] = None,
    endpoint_target_shares: Optional[tuple] = None,
) -> Path:
    ep_pct = endpoint * 100
    # Resolve foresight prerequisites lazily: if caller selected foresight
    # but didn't pre-compute endpoint_year / endpoint_target_shares, pull
    # them from pipeline_config + step 2.2A cache. Keeps myopic callers
    # unchanged and lets a dumb sweep driver drop into foresight mode with
    # just `solver_mode='foresight', foresight_lambda=λ`.
    if solver_mode == 'foresight':
        if endpoint_year is None:
            endpoint_year = _foresight_endpoint_year(ep_pct)
        if endpoint_target_shares is None:
            cfg_tmp_geo = None  # match trajectory-solver default (CAISO-only handles geo internally)
            target_arr = _compute_endpoint_target_shares(
                iso=iso, endpoint_pct=ep_pct,
                demand_growth_level='Medium',
                geo_cost_level=cfg_tmp_geo,
                firm_cost_level='M', ccs_cost_level='M',
                tx_level='Medium',
            )
            endpoint_target_shares = tuple(float(x) for x in target_arr)
    cfg = RunConfig(
        iso=iso, pathway=pathway, endpoint=endpoint, endpoint_pct=ep_pct,
        solver_mode=solver_mode,
        foresight_lambda=foresight_lambda,
        endpoint_year=endpoint_year,
        endpoint_target_shares=endpoint_target_shares,
    )
    if cfg.solver_mode == 'foresight':
        result = solve_pathway_with_foresight(cfg)
        payload = serialize_run_result_v3(result)
    elif cfg.solver_mode == 'myopic':
        result = solve_pathway(cfg)
        payload = serialize_run_result(result)
    else:
        raise ValueError(f"Unknown solver_mode={cfg.solver_mode!r}; "
                         "expected 'myopic' or 'foresight'")
    out = cfg.output_path
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(payload, f)
    os.replace(tmp, out)
    # Phase B sidecar (memo §7 Phase B): colocated with the main payload.
    # Only written when solve_pathway populated result.foresight_preview
    # (P3 runs at the 4 canonical endpoints); skipped otherwise.
    if result.foresight_preview is not None:
        sidecar_out = out.with_name(
            f'pathway{cfg.pathway}_{_ep_tag(cfg.endpoint_pct)}'
            f'_foresight_preview.json'
        )
        sidecar_tmp = sidecar_out.with_suffix('.json.tmp')
        with open(sidecar_tmp, 'w') as f:
            json.dump(result.foresight_preview, f)
        os.replace(sidecar_tmp, sidecar_out)
    print(f"[done] {out}  fleet={result.stranding_metadata['fleet_size_mw']:.0f} MW  "
          f"cfe={result.headline['achieved_cfe_pct']:.2f}%", flush=True)
    return out


def _prebuild_sidecar(iso_threshold: tuple) -> None:
    iso, threshold = iso_threshold
    _load_or_build_peakclean(iso, threshold)


def _run_iso_all(iso: str) -> None:
    for p in PATHWAYS:
        for ep in ENDPOINT_TO_THRESHOLD.keys():
            _run_one(iso, p, ep)
            # Combos reuse module-level tables but the LCOE/tx lru_caches grow
            # unbounded — clear them plus any ephemeral numpy arrays so we
            # don't blow the runner's memory mid-sweep.
            _resource_lcoe_year.cache_clear()
            _resource_tx.cache_clear()
            gc.collect()


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    if args.all:
        # Step 1: pre-build all Stage-1 sidecars in parallel (one per iso×threshold).
        # Each worker is read-only on the dispatch caches; safe to fork.
        pairs = [(iso, t) for iso in ISOS for t in ENDPOINT_TO_THRESHOLD.keys()]
        n_workers = min(len(pairs), (os.cpu_count() or 4))
        print(f"[prebuild] building {len(pairs)} sidecars with {n_workers} workers", flush=True)
        # Use spawn to avoid fork/numpy-thread deadlocks on Linux.
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(n_workers) as pool:
            pool.map(_prebuild_sidecar, pairs)
        print("[prebuild] done", flush=True)
        # Step 2: run combos per ISO in parallel (one process per ISO).
        n_iso_workers = min(len(ISOS), (os.cpu_count() or 4))
        print(f"[sweep] running {len(ISOS)} ISOs with {n_iso_workers} workers", flush=True)
        with ctx.Pool(n_iso_workers) as pool:
            pool.map(_run_iso_all, ISOS)
    else:
        if not (args.iso and args.pathway and args.endpoint):
            print('Provide --iso, --pathway, --endpoint OR --all', file=sys.stderr)
            return 2
        _run_one(args.iso, args.pathway, args.endpoint)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
