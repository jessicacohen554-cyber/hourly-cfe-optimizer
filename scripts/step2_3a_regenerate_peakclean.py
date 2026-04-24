#!/usr/bin/env python3
"""
regenerate_peakclean.py — Replace cache-based peakclean with inline vectorized dispatch.

Iterates over every step_2_1_EF_{ISO}_{threshold}[_part*].parquet in data/step2.1-ef/,
computes resid_norm_p9997 and clean_peak_hour_mw per mix via inline dispatch, and
writes/overwrites the matching _peakclean.parquet sidecar.

Performance: uses a fused numba kernel that computes matmul + gap + 99.97th-pctile
in a single pass per mix with zero intermediate arrays.  ~3x faster and ~875x less
memory than the chunked numpy approach.

Usage:
    python scripts/regenerate_peakclean.py
    python scripts/regenerate_peakclean.py --iso ERCOT
    python scripts/regenerate_peakclean.py --iso ERCOT --threshold 95
    python scripts/regenerate_peakclean.py --validate
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

# Numba — 3x speedup via fused kernel; falls back to numpy if unavailable
try:
    from numba import njit, prange, float32 as nb_f32
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

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
    BATTERY_DURATION_HOURS,
    BATTERY_EFFICIENCY,
    BATTERY8_DURATION_HOURS,
    BATTERY8_EFFICIENCY,
)

try:
    from dispatch_utils import _dispatch_battery, _dispatch_ldes
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
NUMPY_CHUNK_SIZE = 40_000    # only used in numpy fallback path
STAGE1_VERSION = "4"

EF_DIR           = PROJECT_ROOT / "data" / "step2.1-ef"
GEN_PROF_PATH    = PROJECT_ROOT / "data" / "eia-930" / "eia_generation_profiles.parquet"
DEMAND_PROF_PATH = PROJECT_ROOT / "data" / "eia-930" / "eia_demand_profiles.parquet"
HYBRID_DIR       = PROJECT_ROOT / "data" / "hybrid_profiles"

# Canonical resource order — columns of the profile matrix.
# Must match step2_3's _build_clean_profile_table key order.
RESOURCE_ORDER = [
    "clean_firm", "solar", "wind", "hydro", "offshore_wind",
    "ccs_ccgt", "geothermal",
    "solar_batt4", "solar_batt8", "wind_batt4", "wind_batt8",
]
N_RESOURCES = len(RESOURCE_ORDER)

# 99.97th pctile of 8760 values → 3rd-largest value
_TOP_K = int(np.ceil(H * (1.0 - WORST_HOUR_PCTILE / 100.0)))  # 3


# ---------------------------------------------------------------------------
# Profile loading  (pyarrow only — no pandas)
# ---------------------------------------------------------------------------

def _load_gen_profiles(iso: str) -> dict[str, np.ndarray]:
    """Return {fuel: (8760,) float64} for the latest year of the given ISO."""
    if not GEN_PROF_PATH.exists():
        return {}
    t = pq.read_table(GEN_PROF_PATH)
    t = t.filter(pac.equal(t["iso"], iso))
    if t.num_rows == 0:
        return {}
    max_yr = pac.max(t["year"]).as_py()
    t = t.filter(pac.equal(t["year"], max_yr))
    out: dict[str, np.ndarray] = {}
    fuels = set(t.column("fuel").to_pylist())
    for fuel in fuels:
        tf = t.filter(pac.equal(t["fuel"], fuel))
        hrs = tf.column("hour").to_numpy().astype(np.int64)
        vals = tf.column("value").to_numpy().astype(np.float64)
        prof = np.zeros(H, dtype=np.float64)
        prof[hrs] = vals
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
    """Normalized demand profile (sums to 1.0), (8760,) float64.

    Falls back to flat (1/H) if the parquet doesn't exist or has no data
    for the ISO — matches step2_3 fallback.
    """
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
        norm /= s       # re-normalize to exactly 1.0
    else:
        norm = flat.copy()
    return norm


def build_profile_matrix(iso: str) -> np.ndarray:
    """(N_resources, 8760) supply-profile matrix in RESOURCE_ORDER.

    Default profiles match step2_3's _build_clean_profile_table exactly:
      clean_firm → nuclear gen profile or flat
      solar, wind, offshore_wind → gen profile or ZERO
      hydro → gen profile or FLAT        ← NOT zero
      ccs_ccgt → flat
      geothermal → gen profile or flat
      hybrids → hybrid profile or zero
    """
    gen = _load_gen_profiles(iso)
    hyb = _load_hybrid_profiles(iso)
    flat = np.full(H, 1.0 / H, dtype=np.float64)
    zero = np.zeros(H, dtype=np.float64)

    # Map from resource name → (gen_key_or_None, default_profile)
    # Mirrors step2_3 _build_clean_profile_table exactly
    _DEFAULTS: dict[str, tuple[str | None, np.ndarray]] = {
        "clean_firm":    ("nuclear",       flat),
        "solar":         ("solar",         zero),
        "wind":          ("wind",          zero),
        "hydro":         ("hydro",         flat),   # flat, NOT zero
        "offshore_wind": ("offshore_wind", zero),
        "ccs_ccgt":      (None,            flat),   # always flat
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
    return np.stack(rows, axis=0)  # (11, 8760)


# ---------------------------------------------------------------------------
# EF file discovery
# ---------------------------------------------------------------------------

_EF_PAT = re.compile(
    r"^step_2_1_EF_([A-Z]+)_([0-9]+(?:\.[0-9]+)?)(?:_part(\d+))?\.parquet$"
)


def discover_ef_files() -> dict[str, dict[str, list[Path]]]:
    """Return {iso: {threshold_tag: [sorted part paths]}}.

    Skips files containing 'peakclean' or 'interp' in the name.
    """
    tree: dict[str, dict[str, list[tuple[int, Path]]]] = defaultdict(
        lambda: defaultdict(list)
    )
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


def _peakclean_path(iso: str, threshold_tag: str) -> Path:
    return EF_DIR / f"step_2_1_EF_{iso}_{threshold_tag}_peakclean.parquet"


# ---------------------------------------------------------------------------
# Core dispatch — no storage: fused numba kernel (primary) + numpy fallback
# ---------------------------------------------------------------------------

if HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _resid_fused(W, P, dm):
        """Fused matmul + gap + 99.97th-pctile in one pass per mix.

        Computes supply = W[i] · P[:, h], gap = max(0, dm[h] - supply),
        and tracks the 3rd-largest gap (= 99.97th pctile of 8760) — all
        without materializing any (n, 8760) intermediate.

        Memory: O(n) output only.  ~3x faster than chunked numpy.
        """
        n = W.shape[0]
        NR = W.shape[1]
        resid = np.empty(n, dtype=nb_f32)
        for i in prange(n):
            # Top-3 gap tracker (descending: t0 >= t1 >= t2)
            t0 = nb_f32(-1.0)
            t1 = nb_f32(-1.0)
            t2 = nb_f32(-1.0)
            for h in range(H):
                # Inline dot product — 11 multiply-adds
                s = nb_f32(0.0)
                for r in range(NR):
                    s += W[i, r] * P[r, h]
                # Gap (clamp to 0)
                gap = dm[h] - s
                if gap < nb_f32(0.0):
                    gap = nb_f32(0.0)
                # Insert into top-3
                if gap > t0:
                    t2 = t1
                    t1 = t0
                    t0 = gap
                elif gap > t1:
                    t2 = t1
                    t1 = gap
                elif gap > t2:
                    t2 = gap
            resid[i] = t2   # 3rd-largest = 99.97th pctile
        return resid

    # ── Storage dispatch helper: single storage stage (njit, NOT parallel) ──
    @njit(cache=True)
    def _inline_storage_stage(surplus, gap, total_dispatch,
                              capacity, power_rating, efficiency,
                              window_hours):
        """Run one windowed charge/discharge stage.

        Modifies surplus and gap in-place (so downstream stages see
        post-dispatch residuals).  Accumulates discharge into
        total_dispatch (shared across all three stages).

        Algorithm mirrors dispatch_utils._battery_loop / _ldes_loop exactly:
        SOC carries across window boundaries, RTE applied per discharge event.
        """
        if capacity <= 0.0:
            return
        soc = 0.0
        n_windows = (H + window_hours - 1) // window_hours
        for w in range(n_windows):
            ws = w * window_hours
            we = ws + window_hours
            if we > H:
                we = H
            # Charge pass — sequential, hour by hour
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
            # Discharge pass — sequential, hour by hour
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

    # ── Fused storage kernel: parallel over mixes ────────────────────
    @njit(parallel=True, cache=True)
    def _resid_fused_storage(W, P, dm, dn,
                             batt4_pcts, batt8_pcts, ldes_pcts):
        """Fused matmul + 3-stage storage dispatch + top-3 gap per mix.

        Parallelized over mixes via prange.  Within each mix, the three
        storage types run sequentially (battery4 → battery8 → LDES) because
        each stage modifies surplus/gap in-place for the next.

        Per-thread memory: 4 × 8760 × 8 bytes ≈ 280 KB (trivial).
        Internal precision: float64 (matches _compute_resid_single_storage).
        Output: (n,) float32.

        Storage constants (from pipeline_config.py):
          Battery 4hr:  duration=4,   eff=0.85, window=24
          Battery 8hr:  duration=8,   eff=0.85, window=48
          LDES:         duration=100, eff=0.50, window=168
        """
        n = W.shape[0]
        NR = W.shape[1]
        resid = np.empty(n, dtype=nb_f32)

        for i in prange(n):
            # ── Per-thread arrays (stack-allocated by numba) ──
            supply = np.empty(H, dtype=np.float64)
            surplus = np.empty(H, dtype=np.float64)
            gap = np.empty(H, dtype=np.float64)
            total_dispatch = np.zeros(H, dtype=np.float64)

            # ── Step 1: matmul + surplus/gap vs demand_norm ──
            for h in range(H):
                s = 0.0
                for r in range(NR):
                    s += float(W[i, r]) * float(P[r, h])
                supply[h] = s
                dn_h = float(dn[h])
                diff = s - dn_h
                if diff > 0.0:
                    surplus[h] = diff
                    gap[h] = 0.0
                else:
                    surplus[h] = 0.0
                    gap[h] = -diff

            # ── Steps 2-4: sequential storage dispatch ──
            # Battery 4hr (duration=4, eff=0.85, window=24)
            b4_cap = float(batt4_pcts[i]) / 100.0
            _inline_storage_stage(
                surplus, gap, total_dispatch,
                b4_cap, b4_cap / 4.0, 0.85, 24)

            # Battery 8hr (duration=8, eff=0.85, window=48)
            b8_cap = float(batt8_pcts[i]) / 100.0
            _inline_storage_stage(
                surplus, gap, total_dispatch,
                b8_cap, b8_cap / 8.0, 0.85, 48)

            # LDES (duration=100, eff=0.50, window=168)
            ldes_cap = float(ldes_pcts[i]) / 100.0
            _inline_storage_stage(
                surplus, gap, total_dispatch,
                ldes_cap, ldes_cap / 100.0, 0.50, 168)

            # ── Step 5: final gap vs demand_margin + top-3 tracking ──
            t0 = -1.0
            t1 = -1.0
            t2 = -1.0
            for h in range(H):
                total_clean = supply[h] + total_dispatch[h]
                fg = float(dm[h]) - total_clean
                if fg < 0.0:
                    fg = 0.0
                if fg > t0:
                    t2 = t1
                    t1 = t0
                    t0 = fg
                elif fg > t1:
                    t2 = t1
                    t1 = fg
                elif fg > t2:
                    t2 = fg
            resid[i] = nb_f32(t2)

        return resid


def _compute_resid_no_storage_numpy(
    W: np.ndarray,         # (n, 11) float32
    P32: np.ndarray,       # (11, 8760) float32
    dm32: np.ndarray,      # (8760,) float32
) -> np.ndarray:
    """Fallback: chunked numpy path when numba is unavailable."""
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


def _compute_resid_no_storage(
    W: np.ndarray,         # (n, 11) float32  — mix fractions
    P: np.ndarray,         # (11, 8760) float64 or float32
    dm: np.ndarray,        # (8760,) float64 or float32  — demand_margin
) -> np.ndarray:
    """Dispatch to numba fused kernel or numpy fallback.

    Returns (n,) float32 — 99.97th-pctile gap per mix.
    """
    P32 = P.astype(np.float32) if P.dtype != np.float32 else P
    dm32 = dm.astype(np.float32) if dm.dtype != np.float32 else dm
    if HAS_NUMBA:
        return _resid_fused(W, P32, dm32)
    return _compute_resid_no_storage_numpy(W, P32, dm32)


# ---------------------------------------------------------------------------
# Core dispatch — with storage (per-mix loop via dispatch_utils)
# ---------------------------------------------------------------------------

def _compute_resid_single_storage(
    w: np.ndarray,               # (11,) float32  — one mix's weights
    P: np.ndarray,               # (11, 8760) float64
    demand_margin: np.ndarray,   # (8760,) float64  — demand_norm * (1 + RA)
    demand_norm: np.ndarray,     # (8760,) float64  — normalized demand
    batt4_pct: float,
    batt8_pct: float,
    ldes_pct: float,
) -> float:
    """Compute residual for a single mix with standalone storage dispatch.

    Storage dispatch runs against demand_norm (matching reconstruct_hourly_dispatch).
    Final gap is computed against demand_margin (matching _per_archetype_resid).
    """
    supply_total = w.astype(np.float64) @ P   # (8760,)

    # Storage dispatch operates against the raw demand (not margin)
    residual_surplus = np.maximum(0.0, supply_total - demand_norm)
    residual_gap     = np.maximum(0.0, demand_norm - supply_total)

    # Sequential dispatch: battery4 → battery8 → LDES
    # Each call modifies residual_surplus/residual_gap in-place and returns
    # a dispatch profile (additional clean energy delivered).
    batt4_prof = _dispatch_battery(
        residual_surplus, residual_gap, batt4_pct,
        BATTERY_DURATION_HOURS, BATTERY_EFFICIENCY)
    batt8_prof = _dispatch_battery(
        residual_surplus, residual_gap, batt8_pct,
        BATTERY8_DURATION_HOURS, BATTERY8_EFFICIENCY, window_hours=48)
    ldes_prof = _dispatch_ldes(
        residual_surplus, residual_gap, ldes_pct, demand_norm)

    total_clean = supply_total + batt4_prof + batt8_prof + ldes_prof

    # Final gap against demand_margin (includes RA headroom)
    gap = np.maximum(0.0, demand_margin - total_clean)
    top_k = np.partition(gap, -_TOP_K)[-_TOP_K:]
    return float(np.min(top_k))


# ---------------------------------------------------------------------------
# Process one ISO × threshold → write peakclean sidecar
# ---------------------------------------------------------------------------

def process_ef_file(
    iso: str,
    threshold_tag: str,
    paths: list[Path],
    P: np.ndarray,             # (11, 8760) float64 — profile matrix
    demand_norm: np.ndarray,   # (8760,) float64
    demand_margin: np.ndarray, # (8760,) float64
    peak_mw: float,
) -> dict:
    """Process one EF file (or multi-part group). Returns summary dict."""
    threshold_val = float(threshold_tag)

    # Columns we actually need — read only these to save memory
    needed_cols = list(RESOURCE_ORDER) + [
        "battery_dispatch_pct", "battery8_dispatch_pct",
        "ldes_dispatch_pct", "hourly_match_score",
    ]

    # Load & concatenate parts — read only needed columns
    tbls = []
    for p in paths:
        schema = pq.read_schema(p)
        avail = [c for c in needed_cols if c in schema.names]
        tbls.append(pq.read_table(p, columns=avail))
    if len(tbls) > 1:
        ef_tbl = pa.concat_tables(tbls, promote_options="permissive")
    else:
        ef_tbl = tbls[0]
    del tbls
    n = ef_tbl.num_rows

    # ── Build weight matrix directly from arrow columns ───────────
    W = np.zeros((n, N_RESOURCES), dtype=np.float32)
    for i, res in enumerate(RESOURCE_ORDER):
        if res in ef_tbl.schema.names:
            W[:, i] = ef_tbl.column(res).to_numpy().astype(np.float32) / 100.0

    # ── Score floor: identify infeasible mixes early ──────────────
    # Pre-filter before expensive dispatch so we skip them entirely.
    scores = (ef_tbl.column("hourly_match_score").to_numpy()
              if "hourly_match_score" in ef_tbl.schema.names else None)
    score_ok = np.ones(n, dtype=bool)
    n_infeasible = 0
    if scores is not None:
        floor = threshold_val - SCORE_FLOOR_PCT
        score_ok = scores >= floor
        n_infeasible = int(np.sum(~score_ok))

    # ── Partition into storage / no-storage ───────────────────────
    stor_mask = np.zeros(n, dtype=bool)
    for col in ("battery_dispatch_pct", "battery8_dispatch_pct",
                "ldes_dispatch_pct"):
        if col in ef_tbl.schema.names:
            stor_mask |= (ef_tbl.column(col).to_numpy() > 0)
    n_storage = int(stor_mask.sum())

    # Extract storage cols if needed, then free the arrow table
    if n_storage > 0:
        batt4 = (ef_tbl.column("battery_dispatch_pct").to_numpy()
                 if "battery_dispatch_pct" in ef_tbl.schema.names
                 else np.zeros(n))
        batt8 = (ef_tbl.column("battery8_dispatch_pct").to_numpy()
                 if "battery8_dispatch_pct" in ef_tbl.schema.names
                 else np.zeros(n))
        ldes_arr = (ef_tbl.column("ldes_dispatch_pct").to_numpy()
                    if "ldes_dispatch_pct" in ef_tbl.schema.names
                    else np.zeros(n))
    del ef_tbl

    # ── Allocate output; pre-fill infeasible with inf ─────────────
    resid = np.full(n, np.inf, dtype=np.float32)

    # ── No-storage mixes (fused numba kernel — no intermediates) ──
    ns_compute = score_ok & ~stor_mask
    n_ns_compute = int(ns_compute.sum())
    if n_ns_compute > 0:
        resid[ns_compute] = _compute_resid_no_storage(
            W[ns_compute], P, demand_margin)

    # ── Storage mixes (fused numba kernel or per-mix fallback) ─────
    stor_compute = score_ok & stor_mask
    n_stor_compute = int(stor_compute.sum())
    if n_stor_compute > 0 and HAS_NUMBA:
        # Primary path: fused parallel numba kernel
        P32 = P.astype(np.float32)
        dm32 = demand_margin.astype(np.float32)
        dn32 = demand_norm.astype(np.float32)
        resid[stor_compute] = _resid_fused_storage(
            W[stor_compute],
            P32, dm32, dn32,
            batt4[stor_compute].astype(np.float32),
            batt8[stor_compute].astype(np.float32),
            ldes_arr[stor_compute].astype(np.float32))
        del P32, dm32, dn32
    elif n_stor_compute > 0 and HAS_DISPATCH_UTILS:
        # Fallback: per-mix Python loop (no numba)
        stor_idx = np.where(stor_compute)[0]
        for count, idx in enumerate(stor_idx):
            resid[idx] = _compute_resid_single_storage(
                W[idx], P, demand_margin, demand_norm,
                float(batt4[idx]), float(batt8[idx]), float(ldes_arr[idx]))
            if (count + 1) % 5000 == 0:
                print(f"    storage {count+1}/{n_stor_compute}", flush=True)
    elif n_stor_compute > 0:
        # Last resort: treat storage mixes as no-storage
        resid[stor_compute] = _compute_resid_no_storage(
            W[stor_compute], P, demand_margin)

    del W

    # ── Compute output — replicate step2_3 logic exactly ──────────
    #   safe = where(isfinite(resid), resid, 0.0)
    #   clean_peak_hour_mw = (1 - safe) * PEAK_DEMAND_MW
    safe = np.where(np.isfinite(resid), resid, np.float32(0.0)).astype(np.float64)
    cpk = ((1.0 - safe) * peak_mw).astype(np.float32)

    # ── Write parquet ─────────────────────────────────────────────
    out_path = _peakclean_path(iso, threshold_tag)
    tbl = pa.table({
        "resid_norm_p9997": pa.array(resid, type=pa.float32()),
        "clean_peak_hour_mw": pa.array(cpk, type=pa.float32()),
    }).replace_schema_metadata({
        "source_cache_version": "inline_dispatch_v1",
        "stage1_version": STAGE1_VERSION,
        "iso": iso,
        "threshold": threshold_tag,
    })
    tmp = out_path.with_suffix(".parquet.tmp")
    pq.write_table(tbl, tmp)
    os.replace(tmp, out_path)

    # ── Summary stats ─────────────────────────────────────────────
    finite_mask = np.isfinite(resid)
    finite_cpk = cpk[finite_mask]
    return {
        "iso": iso,
        "threshold": threshold_tag,
        "n_mixes": n,
        "n_computed": n_ns_compute + n_stor_compute,
        "cpk_min": float(finite_cpk.min()) if len(finite_cpk) > 0 else 0.0,
        "cpk_max": float(finite_cpk.max()) if len(finite_cpk) > 0 else 0.0,
        "cpk_std": float(finite_cpk.std()) if len(finite_cpk) > 0 else 0.0,
        "n_storage": n_storage,
        "n_infeasible": n_infeasible,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_all(summaries: list[dict]) -> bool:
    """Run basic sanity checks and print results."""
    ok = True

    print(f"\n{'='*90}")
    print("VALIDATION")
    print(f"{'='*90}")

    for s in summaries:
        iso, ttag = s["iso"], s["threshold"]
        pc_path = _peakclean_path(iso, ttag)
        if not pc_path.exists():
            print(f"  FAIL: {pc_path.name} missing")
            ok = False
            continue
        pc_rows = pq.ParquetFile(pc_path).metadata.num_rows
        if pc_rows != s["n_mixes"]:
            print(f"  FAIL: {pc_path.name} row count "
                  f"{pc_rows} != source {s['n_mixes']}")
            ok = False

        # Check metadata
        meta = pq.read_schema(pc_path).metadata or {}
        sv = (meta.get(b"stage1_version") or b"").decode()
        if sv != STAGE1_VERSION:
            print(f"  FAIL: {pc_path.name} stage1_version={sv!r} "
                  f"(expected {STAGE1_VERSION!r})")
            ok = False

    # Spread checks on high-CFE bands
    for s in summaries:
        if float(s["threshold"]) >= 90 and s["n_mixes"] > 100:
            spread = s["cpk_max"] - s["cpk_min"]
            if spread < 100:
                print(f"  WARN: {s['iso']} t={s['threshold']} "
                      f"cpk spread = {spread:.0f} MW — expected thousands")
                ok = False

    # ERCOT 95 specific check (known regression canary)
    ercot95 = [s for s in summaries
               if s["iso"] == "ERCOT" and s["threshold"] == "95"]
    if ercot95:
        spread = ercot95[0]["cpk_max"] - ercot95[0]["cpk_min"]
        if spread < 100:
            print(f"  FAIL: ERCOT 95 cpk spread = {spread:.0f} MW "
                  f"— expected thousands (was 6 MW in old cache)")
            ok = False
        else:
            print(f"  OK:   ERCOT 95 cpk spread = {spread:,.0f} MW")

    if ok:
        print("  All checks passed.")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Regenerate peakclean sidecars with inline dispatch")
    ap.add_argument("--iso", type=str, default=None,
                    help="Process single ISO (e.g. ERCOT)")
    ap.add_argument("--threshold", type=str, default=None,
                    help="Process single threshold (e.g. 95 or 87.5)")
    ap.add_argument("--validate", action="store_true",
                    help="Run validation after generation")
    args = ap.parse_args()

    t_global = time.time()

    if HAS_NUMBA:
        print("[peakclean] numba available — using fused kernel", flush=True)
    else:
        print("[peakclean] numba not found — using numpy fallback "
              "(~3x slower)", flush=True)

    ef_tree = discover_ef_files()

    if args.iso:
        iso_up = args.iso.upper()
        if iso_up not in ef_tree:
            print(f"[peakclean] No EF files found for {iso_up}")
            return 1
        isos_to_process = [iso_up]
    else:
        isos_to_process = sorted(ef_tree.keys())

    summaries: list[dict] = []
    total_mixes = 0

    for iso in isos_to_process:
        print(f"\n{'='*60}")
        print(f"[peakclean] Loading profiles for {iso}")
        print(f"{'='*60}")

        P = build_profile_matrix(iso)
        demand_norm = _load_demand_norm(iso)
        demand_margin = demand_norm * (1.0 + RESOURCE_ADEQUACY_MARGIN)
        peak_mw = float(PEAK_DEMAND_MW[iso])

        print(f"  demand TWh={float(REGIONAL_DEMAND_TWH[iso]):.1f}  "
              f"peak MW={peak_mw:,.0f}  RA={RESOURCE_ADEQUACY_MARGIN}",
              flush=True)

        thresholds = ef_tree[iso]
        if args.threshold:
            if args.threshold not in thresholds:
                print(f"  Threshold {args.threshold} not found for {iso}, "
                      f"skipping")
                continue
            thresholds = {args.threshold: thresholds[args.threshold]}

        for ttag in sorted(thresholds, key=lambda t: float(t)):
            paths = thresholds[ttag]
            t1 = time.time()
            src_rows = sum(
                pq.ParquetFile(p).metadata.num_rows for p in paths)
            print(f"[peakclean] {iso} t={ttag}: {src_rows:,} mixes "
                  f"({len(paths)} part{'s' if len(paths) > 1 else ''})",
                  flush=True)

            s = process_ef_file(iso, ttag, paths, P,
                                demand_norm, demand_margin, peak_mw)
            summaries.append(s)
            total_mixes += s["n_mixes"]

            dt = time.time() - t1
            rate = s["n_computed"] / max(0.01, dt)
            skipped = s["n_infeasible"]
            print(f"  → {dt:.1f}s ({rate:,.0f} dispatched/s) | "
                  f"cpk [{s['cpk_min']:,.0f}, {s['cpk_max']:,.0f}] "
                  f"std={s['cpk_std']:,.0f} | "
                  f"storage={s['n_storage']} "
                  f"skipped={skipped}",
                  flush=True)

        gc.collect()  # once per ISO, not per threshold

    # ── Summary table ─────────────────────────────────────────────
    elapsed = time.time() - t_global
    print(f"\n{'='*100}")
    print(f"{'ISO':<8}{'threshold':>10}{'n_mixes':>12}{'computed':>10}"
          f"{'cpk_min':>12}{'cpk_max':>12}{'cpk_std':>12}"
          f"{'n_storage':>10}{'skipped':>10}")
    print("-" * 100)
    for s in summaries:
        print(f"{s['iso']:<8}{s['threshold']:>10}{s['n_mixes']:>12,}"
              f"{s['n_computed']:>10,}"
              f"{s['cpk_min']:>12,.0f}{s['cpk_max']:>12,.0f}"
              f"{s['cpk_std']:>12,.0f}{s['n_storage']:>10,}"
              f"{s['n_infeasible']:>10,}")
    print("-" * 100)
    print(f"Total: {total_mixes:,} mixes in {elapsed:.0f}s "
          f"({total_mixes / max(1, elapsed):,.0f}/s)")

    # ── Validation ────────────────────────────────────────────────
    if args.validate:
        validate_all(summaries)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
