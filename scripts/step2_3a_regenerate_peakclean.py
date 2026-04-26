#!/usr/bin/env python3
"""
regenerate_peakclean.py — Replace cache-based peakclean with inline vectorized dispatch.

Iterates over every step_2_1_EF_{ISO}_{threshold}[_part*].parquet in data/step2.1-ef/,
computes resid_norm_p9997 and clean_peak_hour_mw per mix via inline dispatch, and
writes/overwrites the matching _peakclean.parquet sidecar.

Also processes interp/dense-augment/lowcf files so the optimizer doesn't fall back
to L1-NN approximations for those mixes.

Performance: uses a fused numba kernel that computes matmul + gap + 99.97th-pctile
in a single pass per mix with zero intermediate arrays.

Usage:
    python scripts/step2_3a_regenerate_peakclean.py
    python scripts/step2_3a_regenerate_peakclean.py --iso ERCOT
    python scripts/step2_3a_regenerate_peakclean.py --iso ERCOT --threshold 95
    python scripts/step2_3a_regenerate_peakclean.py --interp-only
    python scripts/step2_3a_regenerate_peakclean.py --lowcf-only
    python scripts/step2_3a_regenerate_peakclean.py --validate
"""
from __future__ import annotations

import argparse
import gc
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pac
import pyarrow.parquet as pq

try:
    from numba import njit, prange, float32 as nb_f32
except ImportError:
    print("[peakclean] FATAL: numba is required. Install with: pip install numba",
          file=sys.stderr)
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# Resolve project root  (script lives in <root>/scripts/)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from pipeline_config import (
    ISOS,
    REGIONAL_DEMAND_TWH,
    PEAK_DEMAND_MW,
    RESOURCE_ADEQUACY_MARGIN,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
H = 8760
WORST_HOUR_PCTILE = 99.97
SCORE_FLOOR_PCT = 10.0
STAGE1_VERSION = "4"

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

# Storage stage parameters: (duration_hrs, efficiency, window_hrs)
_B4 = (4, 0.85, 24)
_B8 = (8, 0.85, 48)
_LDES = (100, 0.50, 168)
_H2 = (1000, 0.35, 720)


# ---------------------------------------------------------------------------
# Profile loading  (pyarrow only -- no pandas)
# ---------------------------------------------------------------------------

def _load_gen_profiles(iso: str) -> dict[str, np.ndarray]:
    """Return {fuel: (8760,) float64} for the latest year of the given ISO.

    Uses Arrow unique() + boolean mask for vectorized groupby.
    Direct assignment (not np.add.at) since hour indices are unique per fuel.
    """
    if not GEN_PROF_PATH.exists():
        return {}
    t = pq.read_table(GEN_PROF_PATH)
    t = t.filter(pac.equal(t["iso"], iso))
    if t.num_rows == 0:
        return {}
    max_yr = pac.max(t["year"]).as_py()
    t = t.filter(pac.equal(t["year"], max_yr))

    hrs = t.column("hour").to_numpy().astype(np.int64)
    vals = t.column("value").to_numpy().astype(np.float64)
    fuels_col = t.column("fuel")
    unique_fuels = fuels_col.unique().to_pylist()

    out: dict[str, np.ndarray] = {}
    for fuel in unique_fuels:
        mask = pac.equal(fuels_col, fuel).to_numpy()
        prof = np.zeros(H, dtype=np.float64)
        # Direct assignment — hours are unique per fuel, so no accumulation
        # needed.  ~5-10x faster than np.add.at which disables SIMD.
        prof[hrs[mask]] = vals[mask]
        out[str(fuel)] = prof
    return out


def _load_hybrid_profiles(iso: str) -> dict[str, np.ndarray]:
    """Return {hybrid_key: (8760,)} from the .npz file, or zeros if missing."""
    keys = ("solar_batt4", "solar_batt8", "wind_batt4", "wind_batt8")
    npz_path = HYBRID_DIR / f"{iso}_hybrid_profiles.npz"
    if npz_path.exists():
        z = np.load(npz_path)
        return {k: z[k].astype(np.float64) if k in z
                else np.zeros(H, dtype=np.float64) for k in keys}
    return {k: np.zeros(H, dtype=np.float64) for k in keys}


def _load_demand_norm(iso: str) -> np.ndarray:
    """Normalized demand profile (sums to 1.0), (8760,) float64."""
    flat = np.full(H, 1.0 / H, dtype=np.float64)
    if not DEMAND_PROF_PATH.exists():
        return flat
    t = pq.read_table(DEMAND_PROF_PATH, columns=["iso", "year", "hour", "normalized"])
    t = t.filter(pac.equal(t["iso"], iso))
    if t.num_rows == 0:
        return flat
    max_yr = pac.max(t["year"]).as_py()
    t = t.filter(pac.equal(t["year"], max_yr))
    norm = np.zeros(H, dtype=np.float64)
    norm[t.column("hour").to_numpy().astype(np.int64)] = (
        t.column("normalized").to_numpy().astype(np.float64))
    s = norm.sum()
    return norm / s if s > 0 else flat


def build_profile_matrix(iso: str) -> np.ndarray:
    """(N_resources, 8760) supply-profile matrix in RESOURCE_ORDER.

    Read-only defaults (flat/zero) are shared references, not copied.
    np.stack creates a new array so the originals are never mutated.
    """
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
            rows.append(gen.get(gen_key, default) if gen_key else default)
        else:
            rows.append(gen.get(res, zero))
    return np.stack(rows, axis=0)


# ---------------------------------------------------------------------------
# EF file discovery -- unified for standard, interp, and lowcf files
# ---------------------------------------------------------------------------

_EF_PAT = re.compile(
    r"^step_2_1_EF_([A-Z]+)_([0-9]+(?:\.[0-9]+)?)(?:_part(\d+))?\.parquet$"
)
_INTERP_PAT = re.compile(
    r"^step_2_1_EF_([A-Z]+)_([0-9]+(?:\.[0-9]+)?)_(?:p1)?interp_.+\.parquet$"
)
_LOWCF_PAT = re.compile(
    r"^step_2_1_EF_([A-Z]+)_([0-9]+(?:\.[0-9]+)?)_interp_lowcf\.parquet$"
)


def discover_files(
    mode: str = "standard",
    iso_filter: str | None = None,
) -> dict[str, list[tuple[str, list[Path]]]]:
    """Unified file discovery.  Returns {iso: [(threshold_tag, [paths]), ...]}.

    mode: 'standard' -- multi-part EF bands (skip interp/peakclean)
          'interp'   -- interp/dense-augment/p1interp files
          'lowcf'    -- only *_interp_lowcf.parquet
    """
    if not EF_DIR.exists():
        return {}

    if mode == "standard":
        tree: dict[str, dict[str, list[tuple[int, Path]]]] = defaultdict(
            lambda: defaultdict(list))
        for f in sorted(EF_DIR.iterdir()):
            if "peakclean" in f.name or "interp" in f.name:
                continue
            m = _EF_PAT.match(f.name)
            if not m:
                continue
            iso, ttag, part = m.group(1), m.group(2), m.group(3)
            if iso_filter and iso != iso_filter:
                continue
            tree[iso][ttag].append((int(part) if part else -1, f))
        result: dict[str, list[tuple[str, list[Path]]]] = {}
        for iso in sorted(tree):
            result[iso] = [
                (ttag, [p for _, p in sorted(tree[iso][ttag])])
                for ttag in sorted(tree[iso], key=float)
            ]
        return result

    pat = _LOWCF_PAT if mode == "lowcf" else _INTERP_PAT
    result_flat: dict[str, list[tuple[str, list[Path]]]] = defaultdict(list)
    for f in sorted(EF_DIR.iterdir()):
        if "peakclean" in f.name:
            continue
        m = pat.match(f.name)
        if not m:
            continue
        iso, ttag = m.group(1), m.group(2)
        if iso_filter and iso != iso_filter:
            continue
        result_flat[iso].append((ttag, [f]))
    return dict(result_flat)


def _peakclean_path_for(paths: list[Path], iso: str, ttag: str,
                        is_interp: bool) -> Path:
    """Determine the output peakclean sidecar path."""
    if is_interp:
        return paths[0].with_name(paths[0].stem + "_peakclean.parquet")
    return EF_DIR / f"step_2_1_EF_{iso}_{ttag}_peakclean.parquet"


# ---------------------------------------------------------------------------
# Core dispatch -- fused numba kernel
# ---------------------------------------------------------------------------

@njit(cache=True)
def _inline_storage_stage(surplus, gap, total_dispatch,
                          capacity, power_rating, efficiency,
                          window_hours):
    """Run one windowed charge/discharge stage.  Modifies surplus/gap in-place.

    Two-pass (charge-then-discharge) per window.  Do NOT fuse into a single
    pass without re-validating — all charging must complete before any
    discharging to match the existing algorithm.
    """
    if capacity <= 0.0:
        return
    soc = 0.0
    n_windows = (H + window_hours - 1) // window_hours
    for w in range(n_windows):
        ws = w * window_hours
        we = min(ws + window_hours, H)
        for h in range(ws, we):
            s = surplus[h]
            if s > 0.0 and soc < capacity:
                charge = min(s, power_rating, capacity - soc)
                soc += charge
                surplus[h] -= charge
        for h in range(ws, we):
            g = gap[h]
            if g > 0.0 and soc > 0.0:
                discharge = min(g, power_rating, soc * efficiency)
                total_dispatch[h] += discharge
                soc -= discharge / efficiency
                gap[h] -= discharge


@njit(parallel=True, cache=True)
def _resid_fused_all(resid, curtail, W, P32, dm32, dn32,
                     batt4, batt8, ldes, h2, compute_idx):
    """Unified dispatch kernel — handles storage and no-storage mixes.

    Parallelism is via numba prange over mixes.  No external thread pool
    needed — prange maps directly to OS threads outside the GIL.
    """
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
            # Fast path: float32, no intermediates
            t0 = nb_f32(-1.0); t1 = nb_f32(-1.0); t2 = nb_f32(-1.0)
            curt_sum = nb_f32(0.0)
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
                surplus_val = s - dn32[h]
                if surplus_val > nb_f32(0.0):
                    curt_sum += surplus_val
            resid[i] = t2
            curtail[i] = curt_sum
        else:
            # Storage path: float64 + 4-stage dispatch
            supply = np.empty(H, dtype=np.float64)
            surplus = np.empty(H, dtype=np.float64)
            gap_arr = np.empty(H, dtype=np.float64)
            total_dispatch = np.zeros(H, dtype=np.float64)

            for h in range(H):
                s = 0.0
                for r in range(NR):
                    s += float(W[i, r]) * float(P32[r, h])
                supply[h] = s
                diff = s - float(dn32[h])
                if diff > 0.0:
                    surplus[h] = diff; gap_arr[h] = 0.0
                else:
                    surplus[h] = 0.0; gap_arr[h] = -diff

            # 4-stage sequential dispatch: B4 -> B8 -> LDES -> H2
            b4_c = b4 / 100.0
            _inline_storage_stage(surplus, gap_arr, total_dispatch,
                                  b4_c, b4_c / 4.0, 0.85, 24)
            b8_c = b8 / 100.0
            _inline_storage_stage(surplus, gap_arr, total_dispatch,
                                  b8_c, b8_c / 8.0, 0.85, 48)
            ld_c = ld / 100.0
            _inline_storage_stage(surplus, gap_arr, total_dispatch,
                                  ld_c, ld_c / 100.0, 0.50, 168)
            h2_c = h2_val / 100.0
            _inline_storage_stage(surplus, gap_arr, total_dispatch,
                                  h2_c, h2_c / 1000.0, 0.35, 720)

            curt_sum = 0.0
            for h in range(H):
                curt_sum += surplus[h]

            t0 = -1.0; t1 = -1.0; t2 = -1.0
            for h in range(H):
                fg = float(dm32[h]) - (supply[h] + total_dispatch[h])
                if fg < 0.0:
                    fg = 0.0
                if fg > t0:
                    t2 = t1; t1 = t0; t0 = fg
                elif fg > t1:
                    t2 = t1; t1 = fg
                elif fg > t2:
                    t2 = fg
            resid[i] = nb_f32(t2)
            curtail[i] = nb_f32(curt_sum)


# ---------------------------------------------------------------------------
# JIT warm-up -- front-load compilation so first real call is fast
# ---------------------------------------------------------------------------

def _warmup_jit():
    """Compile both kernel code paths with tiny dummy data.

    Numba caches by dtype signature, not array shape, so using H=8760 here
    doesn't matter — the compiled code is reused for any array length with
    the same dtypes.  The purpose is to pay LLVM compilation cost (~10-20s)
    before the first real ISO, so per-file timings aren't inflated.
    """
    t0 = time.time()
    n, nr = 2, N_RESOURCES
    W = np.zeros((n, nr), dtype=np.float32)
    P = np.zeros((nr, H), dtype=np.float32)
    dm = np.ones(H, dtype=np.float32) / H
    dn = dm.copy()
    resid = np.full(n, np.inf, dtype=np.float32)
    curtail = np.zeros(n, dtype=np.float32)
    zero_s = np.zeros(n, dtype=np.float32)
    # Mix 0: no storage (fast path). Mix 1: has storage (slow path).
    batt_warmup = np.array([0.0, 10.0], dtype=np.float32)
    idx = np.arange(n, dtype=np.int64)
    _resid_fused_all(resid, curtail, W, P, dm, dn,
                     batt_warmup, zero_s, zero_s, zero_s, idx)
    dt = time.time() - t0
    print(f"[peakclean] JIT warm-up: {dt:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Process one ISO x threshold -> write peakclean sidecar
# ---------------------------------------------------------------------------

def process_ef_file(
    iso: str,
    threshold_tag: str,
    paths: list[Path],
    P32: np.ndarray,           # (11, 8760) float32
    dm32: np.ndarray,          # (8760,) float32
    dn32: np.ndarray,          # (8760,) float32
    peak_mw: float,
    out_path: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Process one EF file group.  Returns summary dict with timing."""
    t1 = time.time()
    threshold_val = float(threshold_tag)

    needed_cols = list(RESOURCE_ORDER) + [
        "battery_dispatch_pct", "battery8_dispatch_pct",
        "ldes_dispatch_pct", "h2_dispatch_pct", "hourly_match_score",
    ]

    # Read schema once per file, cache for column intersection.
    tbls = []
    for p in paths:
        schema = pq.read_schema(p)
        avail = [c for c in needed_cols if c in schema.names]
        tbls.append(pq.read_table(p, columns=avail))

    # promote_options="default" avoids schema unification overhead when all
    # parts share the same schema (the common case for multi-part EF files).
    ef_tbl = (pa.concat_tables(tbls, promote_options="default")
              if len(tbls) > 1 else tbls[0])
    del tbls
    n = ef_tbl.num_rows
    tbl_names = set(ef_tbl.schema.names)  # cached for all lookups below

    label = paths[0].name if out_path else f"t={threshold_tag}"
    if verbose:
        print(f"[peakclean] {iso} {label}: {n:,} mixes "
              f"({len(paths)} part{'s' if len(paths) > 1 else ''})", flush=True)

    # Build weight matrix in one pass over present resource columns.
    W = np.zeros((n, N_RESOURCES), dtype=np.float32)
    for i, res in enumerate(RESOURCE_ORDER):
        if res in tbl_names:
            W[:, i] = ef_tbl.column(res).to_numpy().astype(np.float32) / 100.0

    # Score floor filter (uses cached tbl_names)
    if "hourly_match_score" in tbl_names:
        scores = ef_tbl.column("hourly_match_score").to_numpy()
        score_ok = scores >= (threshold_val - SCORE_FLOOR_PCT)
        n_infeasible = int(np.sum(~score_ok))
    else:
        score_ok = np.ones(n, dtype=bool)
        n_infeasible = 0

    def _col_or_zeros(name: str) -> np.ndarray:
        if name in tbl_names:
            return ef_tbl.column(name).to_numpy().astype(np.float32)
        return np.zeros(n, dtype=np.float32)

    batt4 = _col_or_zeros("battery_dispatch_pct")
    batt8 = _col_or_zeros("battery8_dispatch_pct")
    ldes_arr = _col_or_zeros("ldes_dispatch_pct")
    h2_arr = _col_or_zeros("h2_dispatch_pct")

    stor_mask = (batt4 > 0) | (batt8 > 0) | (ldes_arr > 0) | (h2_arr > 0)
    n_storage = int(stor_mask.sum())
    n_h2 = int((h2_arr > 0).sum())
    del ef_tbl

    # Dispatch
    resid = np.full(n, np.inf, dtype=np.float32)
    curtail = np.zeros(n, dtype=np.float32)
    compute_idx = np.where(score_ok)[0].astype(np.int64)
    n_computed = compute_idx.shape[0]

    if n_computed > 0:
        _resid_fused_all(resid, curtail, W, P32, dm32, dn32,
                         batt4, batt8, ldes_arr, h2_arr, compute_idx)

    # Derive clean_peak_hour_mw.
    # NOTE: this uses the 2025 PEAK_DEMAND_MW constant.  Step 2.3 rescales
    # by peak_vec[yi]/PEAK_DEMAND_MW when it consumes this column for
    # reporting — so storing the base-year value here is correct.
    cpk = ((1.0 - np.where(np.isfinite(resid), resid,
                           np.float32(0.0)).astype(np.float64))
           * peak_mw).astype(np.float32)
    del W

    # Write parquet
    if out_path is None:
        out_path = EF_DIR / f"step_2_1_EF_{iso}_{threshold_tag}_peakclean.parquet"

    tbl = pa.table({
        "resid_norm_p9997": pa.array(resid, type=pa.float32()),
        "clean_peak_hour_mw": pa.array(cpk, type=pa.float32()),
        "curtailment_total": pa.array(curtail, type=pa.float32()),
    }).replace_schema_metadata({
        "source_cache_version": "inline_dispatch_v4",
        "stage1_version": STAGE1_VERSION,
        "iso": iso, "threshold": threshold_tag,
    })
    tmp = out_path.with_suffix(".parquet.tmp")
    pq.write_table(tbl, tmp)
    os.replace(tmp, out_path)

    # Summary
    finite_mask = np.isfinite(resid)
    finite_cpk = cpk[finite_mask]
    finite_curt = curtail[finite_mask]
    summary = {
        "iso": iso, "threshold": threshold_tag, "out_path": str(out_path),
        "n_mixes": n, "n_computed": n_computed,
        "cpk_min": float(finite_cpk.min()) if len(finite_cpk) else 0.0,
        "cpk_max": float(finite_cpk.max()) if len(finite_cpk) else 0.0,
        "cpk_std": float(finite_cpk.std()) if len(finite_cpk) else 0.0,
        "curt_mean": float(finite_curt.mean()) if len(finite_curt) else 0.0,
        "curt_max": float(finite_curt.max()) if len(finite_curt) else 0.0,
        "n_storage": n_storage, "n_h2": n_h2, "n_infeasible": n_infeasible,
    }

    if verbose:
        dt = time.time() - t1
        rate = n_computed / max(0.01, dt)
        h2_tag = f"h2={n_h2} " if n_h2 > 0 else ""
        print(f"  -> {iso} {label}: {dt:.1f}s ({rate:,.0f}/s) | "
              f"cpk [{summary['cpk_min']:,.0f}, {summary['cpk_max']:,.0f}] "
              f"std={summary['cpk_std']:,.0f} | "
              f"curt_mean={summary['curt_mean']:.4f} max={summary['curt_max']:.4f} | "
              f"storage={n_storage} {h2_tag}skipped={n_infeasible}", flush=True)
    return summary


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_all(summaries: list[dict]) -> bool:
    ok = True
    print(f"\n{'='*90}\nVALIDATION\n{'='*90}")

    for s in summaries:
        pc_path = Path(s["out_path"])
        if not pc_path.exists():
            print(f"  FAIL: {pc_path.name} missing"); ok = False; continue

        pc_rows = pq.ParquetFile(pc_path).metadata.num_rows
        if pc_rows != s["n_mixes"]:
            print(f"  FAIL: {pc_path.name} rows {pc_rows} != {s['n_mixes']}"); ok = False

        meta = pq.read_schema(pc_path).metadata or {}
        sv = (meta.get(b"stage1_version") or b"").decode()
        if sv != STAGE1_VERSION:
            print(f"  FAIL: {pc_path.name} stage1_version={sv!r} != {STAGE1_VERSION!r}"); ok = False

        schema = pq.read_schema(pc_path)
        if "curtailment_total" not in schema.names:
            print(f"  FAIL: {pc_path.name} missing curtailment_total"); ok = False

    # Spread checks
    for s in summaries:
        if float(s["threshold"]) >= 90 and s["n_mixes"] > 100:
            spread = s["cpk_max"] - s["cpk_min"]
            if spread < 100:
                print(f"  WARN: {s['iso']} t={s['threshold']} cpk spread={spread:.0f} MW"); ok = False

    # ERCOT 95 canary
    ercot95 = [s for s in summaries if s["iso"] == "ERCOT" and s["threshold"] == "95"]
    if ercot95:
        spread = ercot95[0]["cpk_max"] - ercot95[0]["cpk_min"]
        tag = "FAIL" if spread < 100 else "OK"
        print(f"  {tag}:   ERCOT 95 cpk spread = {spread:,.0f} MW")
        if spread < 100:
            ok = False

    if ok:
        print("  All checks passed.")
    return ok


# ---------------------------------------------------------------------------
# Main — serial execution (numba prange handles intra-mix parallelism)
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Regenerate peakclean sidecars with inline dispatch")
    ap.add_argument("--iso", type=str, default=None)
    ap.add_argument("--threshold", type=str, default=None)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--interp-only", action="store_true")
    ap.add_argument("--lowcf-only", action="store_true")
    ap.add_argument("--skip-interp", action="store_true")
    args = ap.parse_args()
    if args.lowcf_only:
        args.interp_only = True

    t_global = time.time()
    print("[peakclean] numba available — using fused kernel", flush=True)

    # Front-load LLVM compilation so per-file timings are accurate.
    _warmup_jit()

    summaries: list[dict] = []
    total_mixes = 0
    iso_filter = args.iso.upper() if args.iso else None

    def _load_iso_context(iso: str):
        """Load profiles + demand for an ISO.  Returns (P32, dm32, dn32, peak_mw)."""
        P = build_profile_matrix(iso)
        dn = _load_demand_norm(iso)
        dm = dn * (1.0 + RESOURCE_ADEQUACY_MARGIN)
        pmw = float(PEAK_DEMAND_MW[iso])
        print(f"  demand TWh={float(REGIONAL_DEMAND_TWH[iso]):.1f}  "
              f"peak MW={pmw:,.0f}  RA={RESOURCE_ADEQUACY_MARGIN}", flush=True)
        return P.astype(np.float32), dm.astype(np.float32), dn.astype(np.float32), pmw

    def _run_iso_entries(iso, entries, P32, dm32, dn32, peak_mw, is_interp):
        """Process a list of (ttag, [paths]) entries serially.

        Returns (list[dict], int) — summaries and total mix count.
        Serial execution is correct here: numba prange inside the kernel
        already saturates CPU cores across mixes.  Wrapping this in a
        ThreadPoolExecutor would GIL-block on the pyarrow I/O and numpy
        allocations surrounding each kernel call.
        """
        local_summaries = []
        local_mixes = 0

        if args.threshold and not is_interp:
            entries = [(t, p) for t, p in entries if t == args.threshold]
            if not entries:
                print(f"  Threshold {args.threshold} not found for {iso}, skipping")
                return local_summaries, local_mixes

        for ttag, paths in entries:
            out = _peakclean_path_for(paths, iso, ttag, is_interp) if is_interp else None
            s = process_ef_file(iso, ttag, paths, P32, dm32, dn32,
                                peak_mw, out_path=out)
            local_summaries.append(s)
            local_mixes += s["n_mixes"]

        return local_summaries, local_mixes

    # -- Pass 1: Standard EF bands -----------------------------------------
    if not args.interp_only:
        ef_tree = discover_files("standard", iso_filter)
        for iso in sorted(ef_tree):
            print(f"\n{'='*60}\n[peakclean] Loading profiles for {iso}\n{'='*60}")
            ctx = _load_iso_context(iso)
            s_list, n_mix = _run_iso_entries(iso, ef_tree[iso], *ctx, is_interp=False)
            summaries.extend(s_list)
            total_mixes += n_mix
            gc.collect()

    # -- Pass 2: Interp / lowcf files --------------------------------------
    if not args.skip_interp:
        mode = "lowcf" if args.lowcf_only else "interp"
        interp_tree = discover_files(mode, iso_filter)
        if interp_tree:
            print(f"\n{'='*60}\n[peakclean] Processing {mode} files\n{'='*60}")
        _ctx_cache: dict[str, tuple] = {}
        for iso in sorted(interp_tree):
            if iso not in _ctx_cache:
                print(f"\n[peakclean] Loading profiles for {iso} ({mode} pass)")
                _ctx_cache[iso] = _load_iso_context(iso)
            print(f"  {iso}: {len(interp_tree[iso])} {mode} file(s)", flush=True)
            s_list, n_mix = _run_iso_entries(iso, interp_tree[iso], *_ctx_cache[iso], is_interp=True)
            summaries.extend(s_list)
            total_mixes += n_mix
            gc.collect()

    # -- Summary table ------------------------------------------------------
    elapsed = time.time() - t_global
    hdr = (f"{'ISO':<8}{'thr':>6}{'n_mixes':>12}{'computed':>10}"
           f"{'cpk_min':>10}{'cpk_max':>10}{'cpk_std':>10}"
           f"{'c_mean':>8}{'c_max':>8}{'stor':>7}{'h2':>5}{'skip':>7}")
    print(f"\n{'='*100}\n{hdr}\n{'-'*100}")
    for s in summaries:
        print(f"{s['iso']:<8}{s['threshold']:>6}{s['n_mixes']:>12,}{s['n_computed']:>10,}"
              f"{s['cpk_min']:>10,.0f}{s['cpk_max']:>10,.0f}{s['cpk_std']:>10,.0f}"
              f"{s['curt_mean']:>8.4f}{s['curt_max']:>8.4f}"
              f"{s['n_storage']:>7,}{s.get('n_h2',0):>5,}{s['n_infeasible']:>7,}")
    print(f"{'-'*100}\nTotal: {total_mixes:,} mixes in {elapsed:.0f}s "
          f"({total_mixes / max(1, elapsed):,.0f}/s)")

    if args.validate:
        validate_all(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
