#!/usr/bin/env python3
"""
regenerate_peakclean.py — Replace cache-based peakclean with inline vectorized dispatch.

Iterates over every step_2_1_EF_{ISO}_{threshold}[_part*].parquet in data/step2.1-ef/,
computes resid_norm_p9997 and clean_peak_hour_mw per mix via inline dispatch, and
writes/overwrites the matching _peakclean.parquet sidecar.

Also processes interp/dense-augment files (step_2_1_EF_{ISO}_*_interp_*.parquet)
so the optimizer doesn't fall back to L1-NN approximations for those mixes.

Performance: uses a fused numba kernel that computes matmul + gap + 99.97th-pctile
in a single pass per mix with zero intermediate arrays.  ~3x faster and ~875x less
memory than the chunked numpy approach.

Optimizations over v1:
  - Single fused kernel handles both storage and no-storage mixes (eliminates
    dual kernel launch, partitioning, and boolean fancy-index copies).
  - Index-array dispatch: kernel receives int64 indices into the full W array
    instead of pre-copied subsets — avoids O(n*11) memcpy per partition.
  - Float32 profile/demand casts hoisted to ISO level (1 cast per ISO, not per
    threshold).
  - Thresholds within an ISO processed in parallel via ThreadPoolExecutor
    (numba releases the GIL).

v2.0 additions:
  - H2 seasonal storage dispatch (4th stage): duration=1000hr, eff=0.35,
    window=30d.  Activates when h2_dispatch_pct > 0 in the EF parquet
    (step 2.1d lowcf outputs include H2 for bands >= 95%).
  - --lowcf-only flag: process only *_interp_lowcf.parquet files (targeted
    run after step 2.1d, skips other interp/dense-augment files).

v2.1 additions:
  - GitHub file-size chunking: peakclean parquets exceeding 49 MB are
    auto-split into _chunk{NNN}.parquet files. The optimizer's
    _load_peakclean_with_chunks globs for these chunks transparently.
  - Stale chunks from prior runs cleaned before each write.
  - Summary table includes chunk count column.
  - Validation handles chunked files (sums row counts across chunks).

Usage:
    python scripts/step2_3a_regenerate_peakclean.py
    python scripts/step2_3a_regenerate_peakclean.py --iso ERCOT
    python scripts/step2_3a_regenerate_peakclean.py --iso ERCOT --threshold 95
    python scripts/step2_3a_regenerate_peakclean.py --interp-only
    python scripts/step2_3a_regenerate_peakclean.py --interp-only --iso CAISO
    python scripts/step2_3a_regenerate_peakclean.py --lowcf-only
    python scripts/step2_3a_regenerate_peakclean.py --lowcf-only --iso PJM
    python scripts/step2_3a_regenerate_peakclean.py --validate
    python scripts/step2_3a_regenerate_peakclean.py --workers 4
"""
from __future__ import annotations

import argparse
import gc
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pac
import pyarrow.parquet as pq

try:
    from numba import njit, prange, float32 as nb_f32
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from pipeline_config import (
    ISOS,
    REGIONAL_DEMAND_TWH,
    PEAK_DEMAND_MW,
    RESOURCE_ADEQUACY_MARGIN,
    BATTERY_DURATION_HOURS,
    BATTERY_EFFICIENCY,
    BATTERY8_DURATION_HOURS,
    BATTERY8_EFFICIENCY,
    H2_EFFICIENCY,
    H2_DURATION_HOURS,
    H2_WINDOW_DAYS,
)

try:
    from dispatch_utils import _dispatch_battery, _dispatch_ldes, _dispatch_h2
    HAS_DISPATCH_UTILS = True
except ImportError:
    HAS_DISPATCH_UTILS = False
    print("[peakclean] WARNING: dispatch_utils not importable — storage mixes "
          "will use no-storage fallback", flush=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
H = 8760
WORST_HOUR_PCTILE = 99.97
SCORE_FLOOR_PCT = 10.0
NUMPY_CHUNK_SIZE = 40_000
STAGE1_VERSION = "4"
GITHUB_MAX_BYTES = 49 * 1024 * 1024  # 49 MB — stay under GitHub's 50 MB limit

EF_DIR           = PROJECT_ROOT / "data" / "step2.1-ef"
GEN_PROF_PATH    = PROJECT_ROOT / "data" / "eia-930" / "eia_generation_profiles.parquet"
DEMAND_PROF_PATH = PROJECT_ROOT / "data" / "eia-930" / "eia_demand_profiles.parquet"
HYBRID_DIR       = PROJECT_ROOT / "data" / "hybrid_profiles"

RESOURCE_ORDER = [
    "clean_firm", "solar", "wind", "hydro", "offshore_wind",
    "ccs_ccgt", "geothermal",
    "solar_batt4", "solar_batt8", "wind_batt4", "wind_batt8",
]
N_RESOURCES = len(RESOURCE_ORDER)
H2_WINDOW_HOURS = H2_WINDOW_DAYS * 24
_TOP_K = int(np.ceil(H * (1.0 - WORST_HOUR_PCTILE / 100.0)))


# ---------------------------------------------------------------------------
# Profile loading
# ---------------------------------------------------------------------------

def _load_gen_profiles(iso: str) -> dict[str, np.ndarray]:
    if not GEN_PROF_PATH.exists():
        return {}
    t = pq.read_table(GEN_PROF_PATH)
    t = t.filter(pac.equal(t["iso"], iso))
    if t.num_rows == 0:
        return {}
    max_yr = pac.max(t["year"]).as_py()
    t = t.filter(pac.equal(t["year"], max_yr))
    out: dict[str, np.ndarray] = {}
    for fuel in set(t.column("fuel").to_pylist()):
        tf = t.filter(pac.equal(t["fuel"], fuel))
        hrs = tf.column("hour").to_numpy().astype(np.int64)
        vals = tf.column("value").to_numpy().astype(np.float64)
        prof = np.zeros(H, dtype=np.float64)
        prof[hrs] = vals
        out[str(fuel)] = prof
    return out


def _load_hybrid_profiles(iso: str) -> dict[str, np.ndarray]:
    keys = ("solar_batt4", "solar_batt8", "wind_batt4", "wind_batt8")
    npz_path = HYBRID_DIR / f"{iso}_hybrid_profiles.npz"
    if npz_path.exists():
        z = np.load(npz_path)
        return {k: z[k].astype(np.float64) if k in z
                else np.zeros(H, dtype=np.float64) for k in keys}
    return {k: np.zeros(H, dtype=np.float64) for k in keys}


def _load_demand_norm(iso: str) -> np.ndarray:
    flat = np.full(H, 1.0 / H, dtype=np.float64)
    if not DEMAND_PROF_PATH.exists():
        return flat
    t = pq.read_table(DEMAND_PROF_PATH, columns=["iso", "year", "hour", "normalized"])
    t = t.filter(pac.equal(t["iso"], iso))
    if t.num_rows == 0:
        return flat
    max_yr = pac.max(t["year"]).as_py()
    t = t.filter(pac.equal(t["year"], max_yr))
    hrs = t.column("hour").to_numpy().astype(np.int64)
    vals = t.column("normalized").to_numpy().astype(np.float64)
    norm = np.zeros(H, dtype=np.float64)
    norm[hrs] = vals
    s = norm.sum()
    if s > 0:
        norm /= s
    else:
        norm = flat.copy()
    return norm


def build_profile_matrix(iso: str) -> np.ndarray:
    gen = _load_gen_profiles(iso)
    hyb = _load_hybrid_profiles(iso)
    flat = np.full(H, 1.0 / H, dtype=np.float64)
    zero = np.zeros(H, dtype=np.float64)
    _DEFAULTS: dict[str, tuple[str | None, np.ndarray]] = {
        "clean_firm":    ("nuclear",       flat),
        "solar":         ("solar",         zero),
        "wind":          ("wind",          zero),
        "hydro":         ("hydro",         flat),
        "offshore_wind": ("offshore_wind", zero),
        "ccs_ccgt":      (None,            flat),
        "geothermal":    ("geothermal",    flat),
    }
    rows = []
    for res in RESOURCE_ORDER:
        if res in hyb:
            rows.append(hyb[res])
        elif res in _DEFAULTS:
            gen_key, default = _DEFAULTS[res]
            if gen_key is not None:
                rows.append(gen.get(gen_key, default.copy()))
            else:
                rows.append(default.copy())
        elif res in gen:
            rows.append(gen[res])
        else:
            rows.append(zero.copy())
    return np.stack(rows, axis=0)


# ---------------------------------------------------------------------------
# EF file discovery
# ---------------------------------------------------------------------------

_EF_PAT = re.compile(
    r"^step_2_1_EF_([A-Z]+)_([0-9]+(?:\.[0-9]+)?)(?:_part(\d+))?\.parquet$"
)
_INTERP_PAT = re.compile(
    r"^step_2_1_EF_([A-Z]+)_([0-9]+(?:\.[0-9]+)?)_interp_.+\.parquet$"
)
_LOWCF_PAT = re.compile(
    r"^step_2_1_EF_([A-Z]+)_([0-9]+(?:\.[0-9]+)?)_interp_lowcf\.parquet$"
)


def discover_ef_files() -> dict[str, dict[str, list[Path]]]:
    tree: dict[str, dict[str, list[tuple[int, Path]]]] = defaultdict(
        lambda: defaultdict(list))
    for f in sorted(EF_DIR.iterdir()):
        if "peakclean" in f.name or "interp" in f.name:
            continue
        m = _EF_PAT.match(f.name)
        if m:
            iso, ttag, part = m.group(1), m.group(2), m.group(3)
            tree[iso][ttag].append(
                (int(part) if part is not None else -1, f))
    result: dict[str, dict[str, list[Path]]] = {}
    for iso in sorted(tree):
        result[iso] = {}
        for ttag in sorted(tree[iso], key=lambda t: float(t)):
            parts = sorted(tree[iso][ttag], key=lambda x: x[0])
            result[iso][ttag] = [p for _, p in parts]
    return result


def discover_interp_files(iso_filter: str | None = None,
                          lowcf_only: bool = False,
                          ) -> dict[str, list[tuple[str, Path]]]:
    pat = _LOWCF_PAT if lowcf_only else _INTERP_PAT
    result: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    if not EF_DIR.exists():
        return dict(result)
    for f in sorted(EF_DIR.iterdir()):
        if "peakclean" in f.name:
            continue
        m = pat.match(f.name)
        if m:
            iso, ttag = m.group(1), m.group(2)
            if iso_filter and iso != iso_filter:
                continue
            result[iso].append((ttag, f))
    return dict(result)


def _peakclean_path(iso: str, threshold_tag: str) -> Path:
    return EF_DIR / f"step_2_1_EF_{iso}_{threshold_tag}_peakclean.parquet"


def _interp_peakclean_path(source_path: Path) -> Path:
    return source_path.with_name(source_path.stem + "_peakclean.parquet")


def _chunk_if_oversized(out_path: Path, metadata: dict) -> list[Path]:
    """Split a peakclean parquet into chunks if it exceeds GITHUB_MAX_BYTES.

    If the file is small enough, returns [out_path] unchanged.
    If chunked, writes {stem}_chunk{NNN}.parquet files, removes the
    monolithic file, and returns the list of chunk paths.
    """
    if not out_path.exists():
        return [out_path]
    fsize = out_path.stat().st_size
    if fsize <= GITHUB_MAX_BYTES:
        return [out_path]

    tbl = pq.read_table(out_path)
    n = tbl.num_rows
    bytes_per_row = fsize / n
    rows_per_chunk = int(GITHUB_MAX_BYTES / bytes_per_row * 0.95)
    rows_per_chunk = max(rows_per_chunk, 1)
    n_chunks = (n + rows_per_chunk - 1) // rows_per_chunk

    stem = out_path.stem
    parent = out_path.parent
    chunk_paths = []

    for ci in range(n_chunks):
        s = ci * rows_per_chunk
        e = min(s + rows_per_chunk, n)
        chunk_tbl = tbl.slice(s, e - s).replace_schema_metadata(metadata)
        cp = parent / f"{stem}_chunk{ci:03d}.parquet"
        tmp = cp.with_suffix(".parquet.tmp")
        pq.write_table(chunk_tbl, tmp)
        os.replace(tmp, cp)
        chunk_paths.append(cp)

    out_path.unlink()
    print(f"  [chunk] {out_path.name} was {fsize / 1048576:.1f} MB -> "
          f"split into {n_chunks} chunks of ~{rows_per_chunk:,} rows",
          flush=True)
    return chunk_paths


# ---------------------------------------------------------------------------
# Core dispatch — fused numba kernel + numpy fallback
# ---------------------------------------------------------------------------

if HAS_NUMBA:
    @njit(cache=True)
    def _inline_storage_stage(surplus, gap, total_dispatch,
                              capacity, power_rating, efficiency,
                              window_hours):
        if capacity <= 0.0:
            return
        soc = 0.0
        n_windows = (H + window_hours - 1) // window_hours
        for w in range(n_windows):
            ws = w * window_hours
            we = ws + window_hours
            if we > H:
                we = H
            for h in range(ws, we):
                s = surplus[h]
                if s > 0.0 and soc < capacity:
                    charge = s
                    if charge > power_rating:
                        charge = power_rating
                    remaining = capacity - soc
                    if charge > remaining:
                        charge = remaining
                    soc += charge
                    surplus[h] -= charge
            for h in range(ws, we):
                g = gap[h]
                if g > 0.0 and soc > 0.0:
                    available = soc * efficiency
                    discharge = g
                    if discharge > power_rating:
                        discharge = power_rating
                    if discharge > available:
                        discharge = available
                    total_dispatch[h] += discharge
                    soc -= discharge / efficiency
                    gap[h] -= discharge

    @njit(parallel=True, cache=True)
    def _resid_fused_all(resid, W, P32, dm32, dn32,
                         batt4, batt8, ldes, h2, compute_idx):
        NR = W.shape[1]
        n_compute = compute_idx.shape[0]
        for idx in prange(n_compute):
            i = compute_idx[idx]
            b4 = float(batt4[i])
            b8 = float(batt8[i])
            ld = float(ldes[i])
            h2_val = float(h2[i])
            has_stor = (b4 > 0.0) or (b8 > 0.0) or (ld > 0.0) or (h2_val > 0.0)

            if not has_stor:
                t0 = nb_f32(-1.0)
                t1 = nb_f32(-1.0)
                t2 = nb_f32(-1.0)
                for h in range(H):
                    s = nb_f32(0.0)
                    for r in range(NR):
                        s += W[i, r] * P32[r, h]
                    gap_val = dm32[h] - s
                    if gap_val < nb_f32(0.0):
                        gap_val = nb_f32(0.0)
                    if gap_val > t0:
                        t2 = t1; t1 = t0; t0 = gap_val
                    elif gap_val > t1:
                        t2 = t1; t1 = gap_val
                    elif gap_val > t2:
                        t2 = gap_val
                resid[i] = t2
            else:
                supply = np.empty(H, dtype=np.float64)
                surplus = np.empty(H, dtype=np.float64)
                gap_arr = np.empty(H, dtype=np.float64)
                total_dispatch = np.zeros(H, dtype=np.float64)
                for h in range(H):
                    s = 0.0
                    for r in range(NR):
                        s += float(W[i, r]) * float(P32[r, h])
                    supply[h] = s
                    dn_h = float(dn32[h])
                    diff = s - dn_h
                    if diff > 0.0:
                        surplus[h] = diff
                        gap_arr[h] = 0.0
                    else:
                        surplus[h] = 0.0
                        gap_arr[h] = -diff
                b4_cap = b4 / 100.0
                _inline_storage_stage(surplus, gap_arr, total_dispatch,
                                      b4_cap, b4_cap / 4.0, 0.85, 24)
                b8_cap = b8 / 100.0
                _inline_storage_stage(surplus, gap_arr, total_dispatch,
                                      b8_cap, b8_cap / 8.0, 0.85, 48)
                ldes_cap = ld / 100.0
                _inline_storage_stage(surplus, gap_arr, total_dispatch,
                                      ldes_cap, ldes_cap / 100.0, 0.50, 168)
                h2_cap = h2_val / 100.0
                _inline_storage_stage(surplus, gap_arr, total_dispatch,
                                      h2_cap, h2_cap / 1000.0, 0.35, 720)
                t0 = -1.0; t1 = -1.0; t2 = -1.0
                for h in range(H):
                    total_clean = supply[h] + total_dispatch[h]
                    fg = float(dm32[h]) - total_clean
                    if fg < 0.0:
                        fg = 0.0
                    if fg > t0:
                        t2 = t1; t1 = t0; t0 = fg
                    elif fg > t1:
                        t2 = t1; t1 = fg
                    elif fg > t2:
                        t2 = fg
                resid[i] = nb_f32(t2)


def _compute_resid_no_storage_numpy(W, P32, dm32):
    n = W.shape[0]
    resid = np.empty(n, dtype=np.float32)
    for s in range(0, n, NUMPY_CHUNK_SIZE):
        e = min(s + NUMPY_CHUNK_SIZE, n)
        supply = W[s:e] @ P32
        gap = np.maximum(np.float32(0.0), dm32[None, :] - supply)
        del supply
        top_k = np.partition(gap, -_TOP_K, axis=1)[:, -_TOP_K:]
        resid[s:e] = np.min(top_k, axis=1)
        del gap, top_k
    return resid


def _compute_resid_single_storage(w, P, demand_margin, demand_norm,
                                   batt4_pct, batt8_pct, ldes_pct,
                                   h2_pct=0.0):
    supply_total = w.astype(np.float64) @ P
    residual_surplus = np.maximum(0.0, supply_total - demand_norm)
    residual_gap     = np.maximum(0.0, demand_norm - supply_total)
    batt4_prof = _dispatch_battery(residual_surplus, residual_gap, batt4_pct,
                                   BATTERY_DURATION_HOURS, BATTERY_EFFICIENCY)
    batt8_prof = _dispatch_battery(residual_surplus, residual_gap, batt8_pct,
                                   BATTERY8_DURATION_HOURS, BATTERY8_EFFICIENCY,
                                   window_hours=48)
    ldes_prof = _dispatch_ldes(residual_surplus, residual_gap, ldes_pct,
                               demand_norm)
    h2_prof = _dispatch_h2(residual_surplus, residual_gap, h2_pct, demand_norm)
    total_clean = supply_total + batt4_prof + batt8_prof + ldes_prof + h2_prof
    gap = np.maximum(0.0, demand_margin - total_clean)
    top_k = np.partition(gap, -_TOP_K)[-_TOP_K:]
    return float(np.min(top_k))


# ---------------------------------------------------------------------------
# Process one ISO x threshold
# ---------------------------------------------------------------------------

def process_ef_file(iso, threshold_tag, paths, P, P32, dm32, dn32,
                    demand_norm, demand_margin, peak_mw, out_path=None):
    threshold_val = float(threshold_tag)
    needed_cols = list(RESOURCE_ORDER) + [
        "battery_dispatch_pct", "battery8_dispatch_pct",
        "ldes_dispatch_pct", "h2_dispatch_pct", "hourly_match_score",
    ]
    tbls = []
    for p in paths:
        schema = pq.read_schema(p)
        avail = [c for c in needed_cols if c in schema.names]
        tbls.append(pq.read_table(p, columns=avail))
    ef_tbl = pa.concat_tables(tbls, promote_options="permissive") if len(tbls) > 1 else tbls[0]
    del tbls
    n = ef_tbl.num_rows

    W = np.zeros((n, N_RESOURCES), dtype=np.float32)
    for i, res in enumerate(RESOURCE_ORDER):
        if res in ef_tbl.schema.names:
            W[:, i] = ef_tbl.column(res).to_numpy().astype(np.float32) / 100.0

    scores = (ef_tbl.column("hourly_match_score").to_numpy()
              if "hourly_match_score" in ef_tbl.schema.names else None)
    score_ok = np.ones(n, dtype=bool)
    n_infeasible = 0
    if scores is not None:
        score_ok = scores >= (threshold_val - SCORE_FLOOR_PCT)
        n_infeasible = int(np.sum(~score_ok))

    batt4 = (ef_tbl.column("battery_dispatch_pct").to_numpy().astype(np.float32)
             if "battery_dispatch_pct" in ef_tbl.schema.names
             else np.zeros(n, dtype=np.float32))
    batt8 = (ef_tbl.column("battery8_dispatch_pct").to_numpy().astype(np.float32)
             if "battery8_dispatch_pct" in ef_tbl.schema.names
             else np.zeros(n, dtype=np.float32))
    ldes_arr = (ef_tbl.column("ldes_dispatch_pct").to_numpy().astype(np.float32)
                if "ldes_dispatch_pct" in ef_tbl.schema.names
                else np.zeros(n, dtype=np.float32))
    h2_arr = (ef_tbl.column("h2_dispatch_pct").to_numpy().astype(np.float32)
              if "h2_dispatch_pct" in ef_tbl.schema.names
              else np.zeros(n, dtype=np.float32))

    stor_mask = (batt4 > 0) | (batt8 > 0) | (ldes_arr > 0) | (h2_arr > 0)
    n_storage = int(stor_mask.sum())
    n_h2 = int((h2_arr > 0).sum())
    del ef_tbl

    resid = np.full(n, np.inf, dtype=np.float32)
    compute_idx = np.where(score_ok)[0].astype(np.int64)
    n_computed = compute_idx.shape[0]

    if n_computed > 0 and HAS_NUMBA:
        _resid_fused_all(resid, W, P32, dm32, dn32,
                         batt4, batt8, ldes_arr, h2_arr, compute_idx)
    elif n_computed > 0:
        ns_idx = compute_idx[~stor_mask[compute_idx]]
        stor_idx = compute_idx[stor_mask[compute_idx]]
        if ns_idx.shape[0] > 0:
            resid[ns_idx] = _compute_resid_no_storage_numpy(W[ns_idx], P32, dm32)
        if stor_idx.shape[0] > 0 and HAS_DISPATCH_UTILS:
            for count, idx in enumerate(stor_idx):
                resid[idx] = _compute_resid_single_storage(
                    W[idx], P, demand_margin, demand_norm,
                    float(batt4[idx]), float(batt8[idx]),
                    float(ldes_arr[idx]), float(h2_arr[idx]))
                if (count + 1) % 5000 == 0:
                    print(f"    storage {count+1}/{stor_idx.shape[0]}", flush=True)
        elif stor_idx.shape[0] > 0:
            resid[stor_idx] = _compute_resid_no_storage_numpy(W[stor_idx], P32, dm32)
    del W

    safe = np.where(np.isfinite(resid), resid, np.float32(0.0)).astype(np.float64)
    cpk = ((1.0 - safe) * peak_mw).astype(np.float32)

    if out_path is None:
        out_path = _peakclean_path(iso, threshold_tag)
    meta = {
        "source_cache_version": "inline_dispatch_v1",
        "stage1_version": STAGE1_VERSION,
        "iso": iso,
        "threshold": threshold_tag,
    }
    tbl = pa.table({
        "resid_norm_p9997": pa.array(resid, type=pa.float32()),
        "clean_peak_hour_mw": pa.array(cpk, type=pa.float32()),
    }).replace_schema_metadata(meta)

    # Clean up stale chunks from prior runs before writing
    for sc in sorted(out_path.parent.glob(out_path.stem + "_chunk*.parquet")):
        sc.unlink()

    tmp = out_path.with_suffix(".parquet.tmp")
    pq.write_table(tbl, tmp)
    os.replace(tmp, out_path)

    # Split into chunks if over GitHub's file-size limit
    chunk_paths = _chunk_if_oversized(out_path, meta)

    finite_mask = np.isfinite(resid)
    finite_cpk = cpk[finite_mask]
    return {
        "iso": iso, "threshold": threshold_tag, "out_path": str(out_path),
        "n_mixes": n, "n_computed": n_computed,
        "cpk_min": float(finite_cpk.min()) if len(finite_cpk) > 0 else 0.0,
        "cpk_max": float(finite_cpk.max()) if len(finite_cpk) > 0 else 0.0,
        "cpk_std": float(finite_cpk.std()) if len(finite_cpk) > 0 else 0.0,
        "n_storage": n_storage, "n_h2": n_h2,
        "n_infeasible": n_infeasible,
        "n_chunks": len(chunk_paths),
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_all(summaries):
    ok = True
    print(f"\n{'='*90}")
    print("VALIDATION")
    print(f"{'='*90}")

    for s in summaries:
        pc_path = Path(s["out_path"])
        n_chunks = s.get("n_chunks", 1)
        if n_chunks > 1:
            chunk_files = sorted(pc_path.parent.glob(
                pc_path.stem + "_chunk*.parquet"))
            if not chunk_files:
                print(f"  FAIL: {pc_path.name} chunked but no chunks found")
                ok = False
                continue
            pc_rows = sum(pq.ParquetFile(c).metadata.num_rows for c in chunk_files)
            if pc_rows != s["n_mixes"]:
                print(f"  FAIL: {pc_path.stem}_chunk* total rows "
                      f"{pc_rows} != source {s['n_mixes']}")
                ok = False
            meta = pq.read_schema(chunk_files[0]).metadata or {}
        else:
            if not pc_path.exists():
                print(f"  FAIL: {pc_path.name} missing")
                ok = False
                continue
            pc_rows = pq.ParquetFile(pc_path).metadata.num_rows
            if pc_rows != s["n_mixes"]:
                print(f"  FAIL: {pc_path.name} row count "
                      f"{pc_rows} != source {s['n_mixes']}")
                ok = False
            meta = pq.read_schema(pc_path).metadata or {}

        sv = (meta.get(b"stage1_version") or b"").decode()
        if sv != STAGE1_VERSION:
            print(f"  FAIL: {pc_path.name} stage1_version={sv!r} "
                  f"(expected {STAGE1_VERSION!r})")
            ok = False

    for s in summaries:
        if float(s["threshold"]) >= 90 and s["n_mixes"] > 100:
            spread = s["cpk_max"] - s["cpk_min"]
            if spread < 100:
                print(f"  WARN: {s['iso']} t={s['threshold']} "
                      f"cpk spread = {spread:.0f} MW")
                ok = False

    ercot95 = [s for s in summaries
               if s["iso"] == "ERCOT" and s["threshold"] == "95"]
    if ercot95:
        spread = ercot95[0]["cpk_max"] - ercot95[0]["cpk_min"]
        if spread < 100:
            print(f"  FAIL: ERCOT 95 cpk spread = {spread:.0f} MW")
            ok = False
        else:
            print(f"  OK:   ERCOT 95 cpk spread = {spread:,.0f} MW")

    if ok:
        print("  All checks passed.")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _process_one_threshold(iso, ttag, paths, P, P32, dm32, dn32,
                           demand_norm, demand_margin, peak_mw,
                           out_path=None):
    t1 = time.time()
    src_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in paths)
    label = paths[0].name if out_path else f"t={ttag}"
    print(f"[peakclean] {iso} {label}: {src_rows:,} mixes "
          f"({len(paths)} part{'s' if len(paths) > 1 else ''})", flush=True)

    s = process_ef_file(iso, ttag, paths, P, P32, dm32, dn32,
                        demand_norm, demand_margin, peak_mw, out_path=out_path)

    dt = time.time() - t1
    rate = s["n_computed"] / max(0.01, dt)
    h2_tag = f"h2={s['n_h2']} " if s.get("n_h2", 0) > 0 else ""
    print(f"  -> {iso} {label}: {dt:.1f}s ({rate:,.0f} dispatched/s) | "
          f"cpk [{s['cpk_min']:,.0f}, {s['cpk_max']:,.0f}] "
          f"std={s['cpk_std']:,.0f} | storage={s['n_storage']} {h2_tag}"
          f"skipped={s['n_infeasible']}", flush=True)
    return s


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Regenerate peakclean sidecars with inline dispatch")
    ap.add_argument("--iso", type=str, default=None)
    ap.add_argument("--threshold", type=str, default=None)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--interp-only", action="store_true")
    ap.add_argument("--lowcf-only", action="store_true")
    ap.add_argument("--skip-interp", action="store_true")
    args = ap.parse_args()

    if args.lowcf_only:
        args.interp_only = True

    t_global = time.time()
    if HAS_NUMBA:
        print("[peakclean] numba available — using fused kernel", flush=True)
    else:
        print("[peakclean] numba not found — numpy fallback (~3x slower)", flush=True)

    summaries: list[dict] = []
    total_mixes = 0

    if not args.interp_only:
        ef_tree = discover_ef_files()
        if args.iso:
            iso_up = args.iso.upper()
            if iso_up not in ef_tree:
                print(f"[peakclean] No EF files found for {iso_up}")
                if args.skip_interp:
                    return 1
            else:
                ef_tree = {iso_up: ef_tree[iso_up]}

        for iso in sorted(ef_tree.keys()):
            print(f"\n{'='*60}\n[peakclean] Loading profiles for {iso}\n{'='*60}")
            P = build_profile_matrix(iso)
            demand_norm = _load_demand_norm(iso)
            demand_margin = demand_norm * (1.0 + RESOURCE_ADEQUACY_MARGIN)
            peak_mw = float(PEAK_DEMAND_MW[iso])
            P32 = P.astype(np.float32)
            dm32 = demand_margin.astype(np.float32)
            dn32 = demand_norm.astype(np.float32)
            print(f"  demand TWh={float(REGIONAL_DEMAND_TWH[iso]):.1f}  "
                  f"peak MW={peak_mw:,.0f}  RA={RESOURCE_ADEQUACY_MARGIN}", flush=True)

            thresholds = ef_tree[iso]
            if args.threshold:
                if args.threshold not in thresholds:
                    print(f"  Threshold {args.threshold} not found for {iso}, skipping")
                    continue
                thresholds = {args.threshold: thresholds[args.threshold]}

            sorted_ttags = sorted(thresholds, key=lambda t: float(t))
            n_thresh = len(sorted_ttags)
            max_workers = args.workers or min(n_thresh, os.cpu_count() or 1)
            max_workers = max(1, min(max_workers, n_thresh))

            if max_workers == 1 or n_thresh == 1:
                for ttag in sorted_ttags:
                    s = _process_one_threshold(iso, ttag, thresholds[ttag],
                                               P, P32, dm32, dn32,
                                               demand_norm, demand_margin, peak_mw)
                    summaries.append(s)
                    total_mixes += s["n_mixes"]
            else:
                print(f"  [parallel] {n_thresh} thresholds x {max_workers} workers",
                      flush=True)
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = {
                        pool.submit(_process_one_threshold, iso, ttag,
                                    thresholds[ttag], P, P32, dm32, dn32,
                                    demand_norm, demand_margin, peak_mw): ttag
                        for ttag in sorted_ttags}
                    for fut in as_completed(futures):
                        s = fut.result()
                        summaries.append(s)
                        total_mixes += s["n_mixes"]
            gc.collect()

    if not args.skip_interp:
        iso_filter = args.iso.upper() if args.iso else None
        interp_tree = discover_interp_files(iso_filter=iso_filter,
                                            lowcf_only=args.lowcf_only)
        if interp_tree:
            label = "lowcf" if args.lowcf_only else "interp/dense-augment"
            print(f"\n{'='*60}\n[peakclean] Processing {label} files\n{'='*60}")

        _profile_cache: dict[str, tuple] = {}
        for iso in sorted(interp_tree.keys()):
            entries = interp_tree[iso]
            if not entries:
                continue
            if iso not in _profile_cache:
                print(f"\n[peakclean] Loading profiles for {iso} ({label} pass)")
                P = build_profile_matrix(iso)
                dn = _load_demand_norm(iso)
                dm = dn * (1.0 + RESOURCE_ADEQUACY_MARGIN)
                pmw = float(PEAK_DEMAND_MW[iso])
                _profile_cache[iso] = (P, P.astype(np.float32),
                                       dm.astype(np.float32), dn.astype(np.float32),
                                       dn, dm, pmw)
            P, P32, dm32, dn32, demand_norm, demand_margin, peak_mw = _profile_cache[iso]
            print(f"  {iso}: {len(entries)} {label} file(s)", flush=True)
            for ttag, src_path in entries:
                out = _interp_peakclean_path(src_path)
                s = _process_one_threshold(iso, ttag, [src_path], P, P32, dm32, dn32,
                                           demand_norm, demand_margin, peak_mw,
                                           out_path=out)
                summaries.append(s)
                total_mixes += s["n_mixes"]
            gc.collect()

    elapsed = time.time() - t_global
    print(f"\n{'='*107}")
    print(f"{'ISO':<8}{'threshold':>10}{'n_mixes':>12}{'computed':>10}"
          f"{'cpk_min':>12}{'cpk_max':>12}{'cpk_std':>12}"
          f"{'n_storage':>10}{'n_h2':>6}{'skipped':>10}{'chunks':>7}")
    print("-" * 107)
    for s in summaries:
        print(f"{s['iso']:<8}{s['threshold']:>10}{s['n_mixes']:>12,}"
              f"{s['n_computed']:>10,}"
              f"{s['cpk_min']:>12,.0f}{s['cpk_max']:>12,.0f}"
              f"{s['cpk_std']:>12,.0f}{s['n_storage']:>10,}"
              f"{s.get('n_h2', 0):>6,}"
              f"{s['n_infeasible']:>10,}"
              f"{s.get('n_chunks', 1):>7}")
    print("-" * 107)
    print(f"Total: {total_mixes:,} mixes in {elapsed:.0f}s "
          f"({total_mixes / max(1, elapsed):,.0f}/s)")

    if args.validate:
        validate_all(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
