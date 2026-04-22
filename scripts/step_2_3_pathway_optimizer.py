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

[TRUNCATED_FOR_BREVITY]