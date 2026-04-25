#!/usr/bin/env python3
"""Step 2.1d — Low-CF augmentation for nuclear-heavy ISOs.

Runs AFTER Step 2.1 Efficient Frontier, BEFORE Step 2.2 Cost Optimization.

Problem: Step 1's CF_WINDOW floors excluded low-nuclear mixes:
  NEISO: cf floor = 20  →  P1 stalls at 89.7% CFE
  NYISO: cf floor = 10  →  P1 stalls at 96.6% CFE
  PJM:   cf floor = 30  →  P1 stalls at 74.9% CFE

For each target ISO × each EF band ≥ 50%, generates new candidate mixes
with cf reduced below the current floor by:
  1. Seed selection: existing mixes at cf floor and cf floor + 1pp
  2. CF reduction: reduce cf by 1–5pp, redistribute via 4 strategies
  3. Interpolation: pairwise 25/50/75% blends between new variants
  4. Scoring: fused Numba prange kernel (storage) + vectorized numpy (no-storage)
  5. Deduplication: numpy lexsort on int16 resources + 20x-scaled int32 storage

v2.0 additions:
  - --extend-to-40: adds band 40 for MISO, NEISO, NYISO (P1 year-0 coverage)
  - --dense-band40: generates _interp_dense at band 40 for MISO/NEISO/NYISO
  - P1 ceiling fallback in seed selection: when cf_floor seeds absent,
    widens to cf ≤ P1 ceiling (14.7/26.2/20.3% for MISO/NEISO/NYISO)
  - NYISO bands 50-65 lowcf gap filled via ceiling-widened seeds
  - --max-per-band budget cap (default 8000) keeps total under 50k

Performance (vs v1.0):
  - Fused Numba kernel: ~10x faster on storage mixes (one prange launch,
    not N Python→Numba calls). Inlines 4-stage dispatch per mix.
  - Parallel bands: ThreadPoolExecutor across bands (Numba releases GIL).
  - Numpy dedup: lexsort + diff-based unique detection replaces pandas groupby.

Output: step_2_1_EF_{ISO}_{band}_interp_lowcf.parquet
        step_2_1_EF_{ISO}_40_interp_dense.parquet  (--dense-band40 only)
  — pool loader in step2_3 auto-picks up *_interp_*.parquet files.

Usage:
    python scripts/step2_1d_lowcf_augment.py [--iso ISO] [--band BAND] [--dry-run]
    python scripts/step2_1d_lowcf_augment.py --extend-to-40          # band 40 lowcf
    python scripts/step2_1d_lowcf_augment.py --dense-band40          # band 40 dense
    python scripts/step2_1d_lowcf_augment.py --extend-to-40 --dense-band40  # both
    python scripts/step2_1d_lowcf_augment.py --profile-one  # time one band
    python scripts/step2_1d_lowcf_augment.py --workers 4    # parallel bands
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Add scripts dir to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import step1_pfs_generator as s1
from pipeline_config import (
    ACTIVE_THRESHOLDS, THRESHOLDS,
    BATTERY_EFFICIENCY, BATTERY_DURATION_HOURS,
    BATTERY8_EFFICIENCY, BATTERY8_DURATION_HOURS,
    LDES_EFFICIENCY, LDES_DURATION_HOURS, LDES_WINDOW_DAYS,
    H2_EFFICIENCY, H2_DURATION_HOURS, H2_WINDOW_DAYS,
    H2_MIN_THRESHOLD,
)

# Numba — fused kernel is the primary perf win; falls back to chunked loop
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("WARNING: numba not available — falling back to serial scoring (10x slower)")

H = 8760

EF_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'data', 'step2.1-ef')

STORAGE_COLS = ['battery_dispatch_pct', 'battery8_dispatch_pct',
                'ldes_dispatch_pct', 'h2_dispatch_pct']
STORAGE_SCALE = 20  # 0.05% resolution → integer keys

# Target ISOs and their current CF floors from step1's CF_WINDOW
# ISOs with cf_floor=0 (CAISO, ERCOT, SPP) have full coverage — selecting
# them is a safe no-op (no seeds found at cf=0, script exits with 0 mixes).
TARGET_ISOS = {
    'CAISO': {'cf_floor': 0, 'cf_floor_above': 1},
    'ERCOT': {'cf_floor': 0, 'cf_floor_above': 1},
    'MISO':  {'cf_floor': 10, 'cf_floor_above': 11},
    'NEISO': {'cf_floor': 20, 'cf_floor_above': 21},
    'NYISO': {'cf_floor': 10, 'cf_floor_above': 11},
    'PJM':   {'cf_floor': 30, 'cf_floor_above': 31},
    'SPP':   {'cf_floor': 0, 'cf_floor_above': 1},
}

# ============================================================================
# P1 COVERAGE EXTENSION (v2.0)
# ============================================================================
# P1 clean_firm ceilings: existing + uprate + 0.5pp slack.
# Used as fallback in seed selection when cf_floor seeds are absent.
P1_CF_CEILINGS = {
    'MISO':  14.7,
    'NEISO': 26.2,
    'NYISO': 20.3,
}

# ISOs that need band 40 coverage for P1 year-0.
BAND40_ISOS = {'MISO', 'NEISO', 'NYISO'}

# NYISO lowcf gap: bands 50-65 have base parquets but no _interp_lowcf
# because step1's CF_COARSE_STEP=10 leaves no seeds at cf=10/11 in those
# bands. Filled via P1-ceiling fallback.
NYISO_FILL_BANDS = {50.0, 55.0, 60.0, 65.0}

# Dense interpolation blend weights for band-40 augmentation
DENSE_BLEND_WEIGHTS = np.array([0.20, 0.40, 0.50, 0.60, 0.80],
                               dtype=np.float64)
DENSE_MAX_PAIRS = 5000

# Budget cap: max rows written per band (controls memory in solver pool)
DEFAULT_MAX_PER_BAND = 8000

# CF reduction steps below current floor
CF_REDUCTION_STEPS = np.array([1, 2, 3, 4, 5], dtype=np.float64)

# Interpolation
BLEND_WEIGHTS = np.array([0.25, 0.50, 0.75], dtype=np.float64)
MAX_INTERP_PAIRS = 3000

# Max seeds per band (controls candidate volume)
MAX_SEEDS = 2000


# ============================================================================
# FUSED NUMBA SCORING KERNEL — primary performance optimization
# ============================================================================
# One prange launch scores ALL storage mixes. Inlines 4-stage dispatch
# (bat4 → bat8 → LDES → H2) per mix.  ~10x vs per-mix Python→Numba calls.

if HAS_NUMBA:
    @njit(cache=True)
    def _inline_dispatch_stage(surplus, gap, soc_init,
                               capacity, power_rating, efficiency,
                               window_hours):
        """Single windowed charge/discharge stage.  Returns dispatched energy."""
        if capacity <= 0.0:
            return 0.0
        dispatched = 0.0
        soc = soc_init
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
                    dispatched += discharge
                    soc -= discharge / efficiency
                    gap[h] -= discharge
        return dispatched

    @njit(parallel=True, cache=True, fastmath=True)
    def _batch_score_storage(W, supply_matrix, demand_arr,
                             bat4_pcts, bat8_pcts, ldes_pcts, h2_pcts):
        """Score N mixes WITH storage in one parallel launch.

        Args:
            W: (N, n_resources) float64 resource percentages
            supply_matrix: (n_resources, 8760) float64
            demand_arr: (8760,) float64 normalized demand
            bat4_pcts, bat8_pcts, ldes_pcts, h2_pcts: (N,) float64

        Returns: (N,) float64 total matched energy (unnormalized)
        """
        N = W.shape[0]
        NR = W.shape[1]
        scores = np.empty(N, dtype=np.float64)

        for i in prange(N):
            # Step 1: matmul — build supply profile, compute surplus/gap
            surplus = np.empty(H, dtype=np.float64)
            gap = np.empty(H, dtype=np.float64)
            base_matched = 0.0
            for h in range(H):
                s = 0.0
                for r in range(NR):
                    s += (W[i, r] / 100.0) * supply_matrix[r, h]
                d = demand_arr[h]
                if s < d:
                    base_matched += s
                else:
                    base_matched += d
                diff = s - d
                if diff > 0.0:
                    surplus[h] = diff
                    gap[h] = 0.0
                else:
                    surplus[h] = 0.0
                    gap[h] = -diff

            total_dispatched = 0.0

            # Phase 1: Battery 4hr (window=24)
            b4_cap = bat4_pcts[i] / 100.0
            b4_pow = b4_cap / 4.0 if b4_cap > 0.0 else 0.0
            total_dispatched += _inline_dispatch_stage(
                surplus, gap, 0.0, b4_cap, b4_pow, 0.85, 24)

            # Phase 2: Battery 8hr (window=48)
            b8_cap = bat8_pcts[i] / 100.0
            b8_pow = b8_cap / 8.0 if b8_cap > 0.0 else 0.0
            total_dispatched += _inline_dispatch_stage(
                surplus, gap, 0.0, b8_cap, b8_pow, 0.85, 48)

            # Phase 3: LDES (window=168)
            ldes_cap = ldes_pcts[i] / 100.0
            ldes_pow = ldes_cap / 100.0 if ldes_cap > 0.0 else 0.0
            total_dispatched += _inline_dispatch_stage(
                surplus, gap, 0.0, ldes_cap, ldes_pow, 0.50, 168)

            # Phase 4: H2 (window=720)
            h2_cap = h2_pcts[i] / 100.0
            h2_pow = h2_cap / 1000.0 if h2_cap > 0.0 else 0.0
            total_dispatched += _inline_dispatch_stage(
                surplus, gap, 0.0, h2_cap, h2_pow, 0.35, 720)

            scores[i] = base_matched + total_dispatched

        return scores


# ============================================================================
# FALLBACK SCORING (no numba)
# ============================================================================

def _score_storage_fallback(W, supply_matrix, demand_arr,
                            storage_arr, total_demand):
    """Chunked serial scoring fallback when Numba unavailable."""
    N = len(W)
    scores = np.empty(N, dtype=np.float64)
    ldes_window = LDES_WINDOW_DAYS * 24
    h2_window = H2_WINDOW_DAYS * 24

    chunk_size = 10000
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk_r = W[start:end]
        chunk_s = storage_arr[start:end]
        supply_profiles = (chunk_r / 100.0) @ supply_matrix
        for i in range(end - start):
            b4 = chunk_s[i, 0] / 100.0
            b8 = chunk_s[i, 1] / 100.0
            ld = chunk_s[i, 2] / 100.0
            h2 = chunk_s[i, 3] / 100.0
            scores[start + i] = s1._score_with_all_storage(
                demand_arr, supply_profiles[i], 1.0,
                b4, b4 / BATTERY_DURATION_HOURS if b4 > 0 else 0.0,
                BATTERY_EFFICIENCY,
                b8, b8 / BATTERY8_DURATION_HOURS if b8 > 0 else 0.0,
                BATTERY8_EFFICIENCY,
                ld, ld / LDES_DURATION_HOURS if ld > 0 else 0.0,
                LDES_EFFICIENCY, ldes_window,
                h2, h2 / H2_DURATION_HOURS if h2 > 0 else 0.0,
                H2_EFFICIENCY, h2_window,
            )
        del supply_profiles
    return scores / total_demand * 100.0


# ============================================================================
# DATA LOADING
# ============================================================================

def load_ef_parquet(iso, threshold):
    """Load an EF parquet file, handling part files."""
    t_str = f'{threshold:g}'
    path = os.path.join(EF_DIR, f'step_2_1_EF_{iso}_{t_str}.parquet')
    if os.path.exists(path):
        return pq.read_table(path).to_pandas()

    # Try part files (exclude peakclean and interp)
    parts = sorted([
        f for f in os.listdir(EF_DIR)
        if f.startswith(f'step_2_1_EF_{iso}_{t_str}')
        and f.endswith('.parquet')
        and '_peakclean' not in f
        and '_interp_' not in f
    ])
    if parts:
        tables = [pq.read_table(os.path.join(EF_DIR, p)) for p in parts]
        return pa.concat_tables(tables, promote_options='default').to_pandas()

    return None


def load_ef_parquet_with_interp(iso, threshold):
    """Load base + interp_dense parquets for dense seed pool.

    Used by dense-band40 mode to seed from a richer mix pool.
    """
    t_str = f'{threshold:g}'
    all_parts = sorted([
        f for f in os.listdir(EF_DIR)
        if f.startswith(f'step_2_1_EF_{iso}_{t_str}')
        and f.endswith('.parquet')
        and '_peakclean' not in f
        and '_interp_lowcf' not in f  # exclude lowcf to avoid circularity
    ])
    if not all_parts:
        return None
    tables = [pq.read_table(os.path.join(EF_DIR, p)) for p in all_parts]
    return pa.concat_tables(tables, promote_options='default').to_pandas()


def get_resource_cols(df):
    """Extract resource column names from a dataframe."""
    exclude = {'iso', 'hourly_match_score', 'pareto_type'} | set(STORAGE_COLS)
    return [c for c in df.columns if c not in exclude]


def _find_col_idx(resource_cols, name):
    """Find index of a column name, or None if absent."""
    try:
        return resource_cols.index(name)
    except ValueError:
        return None


# ============================================================================
# NUMPY-BASED DEDUP — replaces pandas groupby (~3x faster, less memory)
# ============================================================================

def deduplicate(resource_arr, storage_arr, scores, n_resources):
    """Deduplicate using packed integer keys + lexsort.

    Returns bool mask of rows to keep (max score per unique key).
    """
    N = len(resource_arr)
    if N == 0:
        return np.zeros(0, dtype=bool)
    if N == 1:
        return np.ones(1, dtype=bool)

    # Build integer keys: int16 resources + 20x-scaled int32 storage
    r_int = resource_arr.astype(np.int32)
    s_int = np.round(storage_arr * STORAGE_SCALE).astype(np.int32)
    n_storage = s_int.shape[1]

    # lexsort sorts by last key first — reverse column order
    sort_keys = []
    for j in range(n_storage - 1, -1, -1):
        sort_keys.append(s_int[:, j])
    for j in range(n_resources - 1, -1, -1):
        sort_keys.append(r_int[:, j])

    order = np.lexsort(sort_keys)

    # Detect group boundaries in sorted order
    r_sorted = r_int[order]
    s_sorted = s_int[order]
    scores_sorted = scores[order]

    r_diff = np.any(np.diff(r_sorted, axis=0) != 0, axis=1)
    s_diff = np.any(np.diff(s_sorted, axis=0) != 0, axis=1)
    boundaries = r_diff | s_diff  # (N-1,) bool

    # Group IDs
    group_ids = np.empty(N, dtype=np.int64)
    group_ids[0] = 0
    group_ids[1:] = np.cumsum(boundaries)
    n_groups = int(group_ids[-1]) + 1

    # Max-score index per group
    best_score = np.full(n_groups, -np.inf, dtype=np.float64)
    best_sorted_idx = np.zeros(n_groups, dtype=np.int64)

    for i in range(N):
        g = group_ids[i]
        if scores_sorted[i] > best_score[g]:
            best_score[g] = scores_sorted[i]
            best_sorted_idx[g] = i

    # Map back to original indices
    keep_original = order[best_sorted_idx]
    mask = np.zeros(N, dtype=bool)
    mask[keep_original] = True
    return mask


# ============================================================================
# SCORING — unified interface
# ============================================================================

def score_all(candidate_r, candidate_s, demand_arr, supply_matrix,
              total_demand):
    """Score candidates: vectorized numpy for no-storage, fused Numba for storage.

    Returns (N,) scores in percent.
    """
    N = len(candidate_r)
    scores = np.empty(N, dtype=np.float64)

    has_storage = candidate_s.sum(axis=1) > 0
    n_ro = int((~has_storage).sum())
    n_ws = int(has_storage.sum())

    t0 = time.time()

    # Track 1: Resource-only (vectorized numpy, chunked)
    if n_ro > 0:
        raw = s1.batch_hourly_scores(demand_arr, supply_matrix,
                                     candidate_r[~has_storage])
        scores[~has_storage] = raw / total_demand * 100.0

    # Track 2: Storage mixes — fused Numba kernel or fallback
    if n_ws > 0:
        ws_r = np.ascontiguousarray(candidate_r[has_storage])
        ws_s = np.ascontiguousarray(candidate_s[has_storage])
        if HAS_NUMBA:
            raw = _batch_score_storage(
                ws_r, supply_matrix, demand_arr,
                ws_s[:, 0].copy(), ws_s[:, 1].copy(),
                ws_s[:, 2].copy(), ws_s[:, 3].copy())
            scores[has_storage] = raw / total_demand * 100.0
        else:
            scores[has_storage] = _score_storage_fallback(
                ws_r, supply_matrix, demand_arr, ws_s, total_demand)

    elapsed = time.time() - t0
    print(f"    Scoring: {n_ro:,} no-storage + {n_ws:,} storage "
          f"in {elapsed:.1f}s")

    return scores


# ============================================================================
# SEED SELECTION (v2.0 — tiered widening with P1 ceiling fallback)
# ============================================================================

def select_seeds(iso, threshold):
    """Select seed mixes for low-CF augmentation at this ISO/band.

    Tiered search:
      1. cf == cf_floor  OR  cf == cf_floor + 1
      2. cf in [cf_floor - 1, cf_floor + 2]   (widen ±1)
      3. cf <= P1 ceiling for this ISO         (fallback for NYISO gaps)

    Returns (resource_arr, storage_arr, resource_cols, cf_col_idx) or Nones.
    """
    config = TARGET_ISOS[iso]
    cf_floor = config['cf_floor']
    cf_above = config['cf_floor_above']
    p1_ceiling = P1_CF_CEILINGS.get(iso)

    df = load_ef_parquet(iso, threshold)
    if df is None or len(df) == 0:
        return None, None, None, None

    resource_cols = get_resource_cols(df)
    cf_col_idx = _find_col_idx(resource_cols, 'clean_firm')
    if cf_col_idx is None:
        print(f"  WARNING: No clean_firm column for {iso} @ {threshold}")
        return None, None, None, None

    # Extract arrays once — no DataFrame downstream
    r_arr = df[resource_cols].values.astype(np.float64)
    s_arr = df[STORAGE_COLS].values.astype(np.float64)
    cf_vals = r_arr[:, cf_col_idx]

    # --- Tier 1: exact cf floor / floor + 1 ---
    mask = (cf_vals == cf_floor) | (cf_vals == cf_above)
    seed_tier = 'exact'

    if mask.sum() == 0:
        # --- Tier 2: widen to [cf_floor - 1, cf_above + 1] ---
        mask = (cf_vals >= cf_floor - 1) & (cf_vals <= cf_above + 1)
        seed_tier = 'widen_1'

    if mask.sum() == 0 and p1_ceiling is not None:
        # --- Tier 3: P1 ceiling fallback ---
        # Select all mixes with cf <= P1 ceiling.  These are already
        # P1-eligible; the augmentation pipeline generates interpolated
        # blends that fill in the low-cf region of this band.
        mask = cf_vals <= p1_ceiling
        seed_tier = f'p1_ceiling(<={p1_ceiling})'

    if mask.sum() == 0:
        return None, None, None, None

    if seed_tier == 'exact':
        n_floor = int((cf_vals == cf_floor).sum())
        n_above = int((cf_vals == cf_above).sum())
        print(f"    Seeds [{seed_tier}]: {n_floor} at cf={cf_floor}, "
              f"{n_above} at cf={cf_above}")
    else:
        unique_cfs_in_mask = sorted(np.unique(cf_vals[mask].astype(int)))
        print(f"    Seeds [{seed_tier}]: {mask.sum()} mixes, "
              f"cf levels: {unique_cfs_in_mask[:10]}")

    seed_r = r_arr[mask]
    seed_s = s_arr[mask]

    if len(seed_r) > MAX_SEEDS:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(seed_r), MAX_SEEDS, replace=False)
        seed_r = seed_r[idx]
        seed_s = seed_s[idx]
        print(f"    Sampled to {MAX_SEEDS} seeds")

    return seed_r, seed_s, resource_cols, cf_col_idx


# ============================================================================
# CF REDUCTION PERTURBATION — vectorized
# ============================================================================

def generate_cf_reductions(seed_r, seed_s, resource_cols, cf_col_idx):
    """Generate low-CF variants by reducing cf and redistributing delta.

    4 strategies × 5 deltas × n_seeds variants.
    Returns (resource_arr, storage_arr).
    """
    n_seeds = len(seed_r)
    n_resources = seed_r.shape[1]
    n_deltas = len(CF_REDUCTION_STEPS)
    n_strategies = 4
    n_total = n_seeds * n_deltas * n_strategies

    out_r = np.empty((n_total, n_resources), dtype=np.float64)
    out_s = np.empty((n_total, seed_s.shape[1]), dtype=np.float64)

    # Column indices
    solar_idx = _find_col_idx(resource_cols, 'solar')
    wind_idx = _find_col_idx(resource_cols, 'wind')
    sb4_idx = _find_col_idx(resource_cols, 'solar_batt4')
    sb8_idx = _find_col_idx(resource_cols, 'solar_batt8')
    wb4_idx = _find_col_idx(resource_cols, 'wind_batt4')
    wb8_idx = _find_col_idx(resource_cols, 'wind_batt8')
    hydro_idx = _find_col_idx(resource_cols, 'hydro')

    # Non-CF, non-hydro mask for proportional redistribution
    non_cf_mask = np.ones(n_resources, dtype=bool)
    non_cf_mask[cf_col_idx] = False
    if hydro_idx is not None:
        non_cf_mask[hydro_idx] = False

    cursor = 0
    for delta in CF_REDUCTION_STEPS:
        for s_i in range(n_strategies):
            sl = slice(cursor, cursor + n_seeds)
            new_r = seed_r.copy()

            orig_cf = new_r[:, cf_col_idx].copy()
            new_r[:, cf_col_idx] = np.maximum(orig_cf - delta, 0.0)
            actual_delta = orig_cf - new_r[:, cf_col_idx]

            if s_i == 0:  # proportional
                non_cf_sum = seed_r[:, non_cf_mask].sum(axis=1)
                safe_sum = np.where(non_cf_sum > 0, non_cf_sum, 1.0)
                scale = actual_delta / safe_sum
                new_r[:, non_cf_mask] += (
                    seed_r[:, non_cf_mask] * scale[:, None])

            elif s_i == 1:  # solar-heavy
                targets = [i for i in [solar_idx, sb4_idx, sb8_idx]
                           if i is not None]
                if targets:
                    per = actual_delta / len(targets)
                    for t in targets:
                        new_r[:, t] += per
                elif wind_idx is not None:
                    new_r[:, wind_idx] += actual_delta

            elif s_i == 2:  # wind-heavy
                targets = [i for i in [wind_idx, wb4_idx, wb8_idx]
                           if i is not None]
                if targets:
                    per = actual_delta / len(targets)
                    for t in targets:
                        new_r[:, t] += per
                elif solar_idx is not None:
                    new_r[:, solar_idx] += actual_delta

            else:  # storage-heavy: solar_batt4 + wind_batt4
                targets = [i for i in [sb4_idx, wb4_idx]
                           if i is not None]
                if len(targets) == 2:
                    new_r[:, targets[0]] += actual_delta * 0.5
                    new_r[:, targets[1]] += actual_delta * 0.5
                elif len(targets) == 1:
                    new_r[:, targets[0]] += actual_delta
                else:
                    fb = [i for i in [solar_idx, wind_idx]
                          if i is not None]
                    if fb:
                        per = actual_delta / len(fb)
                        for t in fb:
                            new_r[:, t] += per

            np.clip(new_r, 0, 350, out=new_r)
            out_r[sl] = new_r
            out_s[sl] = seed_s
            cursor += n_seeds

    return out_r, out_s


# ============================================================================
# INTERPOLATION — vectorized
# ============================================================================

def generate_interpolations(resource_arr, storage_arr):
    """Pairwise 25/50/75% blends between low-cf variants."""
    n = len(resource_arr)
    if n < 2:
        return (np.empty((0, resource_arr.shape[1]), dtype=np.float64),
                np.empty((0, storage_arr.shape[1]), dtype=np.float64))

    rng = np.random.default_rng(42)
    n_pairs = min(MAX_INTERP_PAIRS, n * (n - 1) // 2)

    idx_a = rng.integers(0, n, size=n_pairs)
    idx_b = rng.integers(0, n, size=n_pairs)
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]
    n_valid = len(idx_a)

    n_blends = len(BLEND_WEIGHTS)
    n_out = n_valid * n_blends
    out_r = np.empty((n_out, resource_arr.shape[1]), dtype=np.float64)
    out_s = np.empty((n_out, storage_arr.shape[1]), dtype=np.float64)

    for k, w in enumerate(BLEND_WEIGHTS):
        sl = slice(k * n_valid, (k + 1) * n_valid)
        np.add(resource_arr[idx_a] * w,
               resource_arr[idx_b] * (1 - w),
               out=out_r[sl])
        np.add(storage_arr[idx_a] * w,
               storage_arr[idx_b] * (1 - w),
               out=out_s[sl])

    # Round storage to dedup resolution
    np.round(out_s * STORAGE_SCALE, out=out_s)
    out_s /= STORAGE_SCALE
    np.clip(out_s, 0, None, out=out_s)

    return out_r, out_s


# ============================================================================
# SINGLE-BAND AUGMENTATION
# ============================================================================

def augment_band(iso, threshold, demand_arr, supply_matrix, total_demand,
                 resource_cols_hint=None, dry_run=False, max_per_band=None):
    """Augment a single ISO/band. Returns (out_path, n_new) or (None, 0)."""
    print(f"\n  --- {iso} @ {threshold}% ---")

    seed_r, seed_s, resource_cols, cf_col_idx = select_seeds(iso, threshold)
    if seed_r is None or len(seed_r) == 0:
        print(f"    No seeds — skipping")
        return None, 0

    # Use hint resource_cols if provided (consistency across bands)
    if resource_cols_hint is not None:
        resource_cols = resource_cols_hint
        cf_col_idx = _find_col_idx(resource_cols, 'clean_firm')

    n_resources = len(resource_cols)
    n_seeds = len(seed_r)

    # --- Step 1: CF reduction ---
    reduced_r, reduced_s = generate_cf_reductions(
        seed_r, seed_s, resource_cols, cf_col_idx)
    print(f"    CF reductions: {len(reduced_r):,} variants "
          f"from {n_seeds} seeds")

    # --- Step 2: Interpolation ---
    interp_r, interp_s = generate_interpolations(reduced_r, reduced_s)
    print(f"    Interpolations: {len(interp_r):,} variants")

    # Combine
    all_r = np.vstack([reduced_r, interp_r])
    all_s = np.vstack([reduced_s, interp_s])
    n_total = len(all_r)
    print(f"    Total candidates: {n_total:,}")

    if dry_run:
        print(f"    [DRY RUN] Would score {n_total:,} — skipping")
        return None, n_total

    # --- Step 3: Score ---
    scores = score_all(all_r, all_s, demand_arr, supply_matrix,
                       total_demand)

    # --- Step 4: Filter to band ---
    sorted_t = sorted(THRESHOLDS)
    idx = sorted_t.index(threshold)
    lower = threshold
    upper = sorted_t[idx + 1] if idx + 1 < len(sorted_t) else 100.0
    in_band = (scores >= lower) & (scores < upper)
    n_in_band = int(in_band.sum())
    print(f"    In-band: {n_in_band:,} / {n_total:,} "
          f"(band [{lower}, {upper}))")

    if n_in_band == 0:
        return None, 0

    band_r = all_r[in_band]
    band_s = all_s[in_band]
    band_scores = scores[in_band]

    # --- Step 5: Deduplicate ---
    keep_mask = deduplicate(band_r, band_s, band_scores, n_resources)
    n_kept = int(keep_mask.sum())
    print(f"    After dedup: {n_kept:,} "
          f"(removed {n_in_band - n_kept:,})")

    final_r = band_r[keep_mask]
    final_s = band_s[keep_mask]
    final_scores = band_scores[keep_mask]

    # --- Step 5b: Budget cap (v2.0) ---
    if max_per_band is not None and n_kept > max_per_band:
        # Prioritize mixes with clean_firm ≤ P1 ceiling, then by score desc
        p1_ceil = P1_CF_CEILINGS.get(iso)
        if p1_ceil is not None:
            cf_vals = final_r[:, cf_col_idx]
            is_p1 = cf_vals <= p1_ceil
            # Sort: P1-eligible first (desc), then by score desc
            sort_key = is_p1.astype(np.float64) * 1e6 + final_scores
            top_idx = np.argsort(-sort_key)[:max_per_band]
        else:
            top_idx = np.argsort(-final_scores)[:max_per_band]
        final_r = final_r[top_idx]
        final_s = final_s[top_idx]
        final_scores = final_scores[top_idx]
        print(f"    Budget cap: kept {max_per_band:,} / {n_kept:,}")
        n_kept = max_per_band

    # --- Step 6: Write parquet (PyArrow native, skip pandas) ---
    arrays = [pa.array(np.full(n_kept, iso, dtype=object))]
    names = ['iso']
    for i, col in enumerate(resource_cols):
        arrays.append(pa.array(final_r[:, i].astype(np.int16)))
        names.append(col)
    for i, sc in enumerate(STORAGE_COLS):
        arrays.append(pa.array(final_s[:, i]))
        names.append(sc)
    arrays.append(pa.array(final_scores))
    names.append('hourly_match_score')
    arrays.append(
        pa.array(np.full(n_kept, 'augmented_lowcf', dtype=object)))
    names.append('pareto_type')

    table = pa.table(dict(zip(names, arrays)))
    t_str = f'{threshold:g}'
    out_name = f'step_2_1_EF_{iso}_{t_str}_interp_lowcf.parquet'
    out_path = os.path.join(EF_DIR, out_name)
    pq.write_table(table, out_path, compression='snappy')
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"    Wrote {out_name} ({n_kept:,} rows, {size_mb:.1f} MB)")

    # Verify CF distribution
    cf_vals = final_r[:, cf_col_idx].astype(int)
    unique_cfs = np.unique(cf_vals)
    print(f"    CF levels: {sorted(unique_cfs)}")

    return out_path, n_kept


# ============================================================================
# DENSE BAND-40 INTERPOLATION (v2.0)
# ============================================================================
# Generates _interp_dense files at band 40 for MISO/NEISO/NYISO.
# Simpler than the full rel_tax_augment dense grid: pairwise blends of
# existing base parquet mixes, filtered to cf ≤ P1 ceiling.  Enough to
# give the solver pool adequate resolution at band 40 without needing the
# exhaustive VRE×hybrid×storage sweep.

def generate_dense_band40(iso, demand_arr, supply_matrix, total_demand,
                          dry_run=False, max_per_band=None):
    """Generate _interp_dense at band 40 for one ISO.

    Loads the base parquet at band 40, takes all mixes with cf ≤ P1 ceiling,
    generates pairwise blends at 5 weights, scores, deduplicates, writes.
    Returns (out_path, n_new) or (None, 0).
    """
    threshold = 40.0
    t_str = '40'
    p1_ceil = P1_CF_CEILINGS.get(iso)

    # Check if already exists
    out_name = f'step_2_1_EF_{iso}_{t_str}_interp_dense.parquet'
    out_path = os.path.join(EF_DIR, out_name)
    if os.path.exists(out_path):
        existing = pq.read_metadata(out_path).num_rows
        print(f"  [dense-band40] {iso} @ 40: already exists "
              f"({existing:,} rows) — skipping")
        return None, 0

    print(f"\n  --- {iso} @ 40% [dense interpolation] ---")

    # Load base parquet (NOT interp files — avoid circularity)
    df = load_ef_parquet(iso, threshold)
    if df is None or len(df) == 0:
        print(f"    No base parquet at band 40 — skipping")
        return None, 0

    resource_cols = get_resource_cols(df)
    cf_col_idx = _find_col_idx(resource_cols, 'clean_firm')

    r_arr = df[resource_cols].values.astype(np.float64)
    s_arr = df[STORAGE_COLS].values.astype(np.float64)
    n_base = len(r_arr)
    print(f"    Base pool: {n_base:,} mixes")

    # Filter to cf ≤ P1 ceiling (prioritize P1-eligible mixes) + some above
    # for interpolation diversity
    if cf_col_idx is not None and p1_ceil is not None:
        cf_vals = r_arr[:, cf_col_idx]
        # Keep all cf ≤ ceiling, plus a sample of higher-cf for blending
        p1_mask = cf_vals <= p1_ceil
        above_mask = cf_vals > p1_ceil
        n_p1 = int(p1_mask.sum())
        n_above = int(above_mask.sum())

        # Sample up to 500 above-ceiling mixes for blend diversity
        max_above = 500
        if n_above > max_above:
            rng = np.random.default_rng(42)
            above_idx = np.where(above_mask)[0]
            keep_above = rng.choice(above_idx, max_above, replace=False)
            combined_mask = p1_mask.copy()
            combined_mask[keep_above] = True
        else:
            combined_mask = np.ones(n_base, dtype=bool)

        r_arr = r_arr[combined_mask]
        s_arr = s_arr[combined_mask]
        print(f"    After cf filter: {len(r_arr):,} mixes "
              f"({n_p1} P1-eligible, up to {min(n_above, max_above)} above)")

    n_pool = len(r_arr)
    if n_pool < 2:
        print(f"    Too few mixes for interpolation — skipping")
        return None, 0

    # Generate pairwise blends
    rng = np.random.default_rng(42)
    n_pairs = min(DENSE_MAX_PAIRS, n_pool * (n_pool - 1) // 2)
    idx_a = rng.integers(0, n_pool, size=n_pairs)
    idx_b = rng.integers(0, n_pool, size=n_pairs)
    diff_mask = idx_a != idx_b
    idx_a, idx_b = idx_a[diff_mask], idx_b[diff_mask]
    n_valid = len(idx_a)

    n_blends = len(DENSE_BLEND_WEIGHTS)
    n_interp = n_valid * n_blends
    interp_r = np.empty((n_interp, r_arr.shape[1]), dtype=np.float64)
    interp_s = np.empty((n_interp, s_arr.shape[1]), dtype=np.float64)

    for k, w in enumerate(DENSE_BLEND_WEIGHTS):
        sl = slice(k * n_valid, (k + 1) * n_valid)
        np.add(r_arr[idx_a] * w, r_arr[idx_b] * (1 - w), out=interp_r[sl])
        np.add(s_arr[idx_a] * w, s_arr[idx_b] * (1 - w), out=interp_s[sl])

    # Round storage to dedup resolution
    np.round(interp_s * STORAGE_SCALE, out=interp_s)
    interp_s /= STORAGE_SCALE
    np.clip(interp_s, 0, None, out=interp_s)

    print(f"    Dense interpolations: {n_interp:,} candidates")

    if dry_run:
        print(f"    [DRY RUN] Would score {n_interp:,} — skipping")
        return None, n_interp

    # Score
    scores = score_all(interp_r, interp_s, demand_arr, supply_matrix,
                       total_demand)

    # Filter to band [40, 50)
    sorted_t = sorted(THRESHOLDS)
    t_idx = sorted_t.index(40.0)
    upper = sorted_t[t_idx + 1] if t_idx + 1 < len(sorted_t) else 100.0
    in_band = (scores >= 40.0) & (scores < upper)
    n_in_band = int(in_band.sum())
    print(f"    In-band: {n_in_band:,} / {n_interp:,} "
          f"(band [40, {upper}))")

    if n_in_band == 0:
        return None, 0

    band_r = interp_r[in_band]
    band_s = interp_s[in_band]
    band_scores = scores[in_band]

    # Deduplicate
    n_resources = len(resource_cols)
    keep_mask = deduplicate(band_r, band_s, band_scores, n_resources)
    n_kept = int(keep_mask.sum())
    print(f"    After dedup: {n_kept:,}")

    final_r = band_r[keep_mask]
    final_s = band_s[keep_mask]
    final_scores = band_scores[keep_mask]

    # Budget cap
    cap = max_per_band or DEFAULT_MAX_PER_BAND
    if n_kept > cap:
        # Prioritize P1-eligible mixes
        if cf_col_idx is not None and p1_ceil is not None:
            cf_vals = final_r[:, cf_col_idx]
            is_p1 = cf_vals <= p1_ceil
            sort_key = is_p1.astype(np.float64) * 1e6 + final_scores
            top_idx = np.argsort(-sort_key)[:cap]
        else:
            top_idx = np.argsort(-final_scores)[:cap]
        final_r = final_r[top_idx]
        final_s = final_s[top_idx]
        final_scores = final_scores[top_idx]
        print(f"    Budget cap: kept {cap:,} / {n_kept:,}")
        n_kept = cap

    # Write parquet
    arrays = [pa.array(np.full(n_kept, iso, dtype=object))]
    names = ['iso']
    for i, col in enumerate(resource_cols):
        arrays.append(pa.array(final_r[:, i].astype(np.int16)))
        names.append(col)
    for i, sc in enumerate(STORAGE_COLS):
        arrays.append(pa.array(final_s[:, i]))
        names.append(sc)
    arrays.append(pa.array(final_scores))
    names.append('hourly_match_score')
    arrays.append(
        pa.array(np.full(n_kept, 'interp_dense', dtype=object)))
    names.append('pareto_type')

    table = pa.table(dict(zip(names, arrays)))
    pq.write_table(table, out_path, compression='snappy')
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"    Wrote {out_name} ({n_kept:,} rows, {size_mb:.1f} MB)")

    if cf_col_idx is not None:
        cf_vals = final_r[:, cf_col_idx].astype(int)
        unique_cfs = np.unique(cf_vals)
        print(f"    CF levels: {sorted(unique_cfs)}")

    return out_path, n_kept


# ============================================================================
# BAND LIST BUILDER (v2.0)
# ============================================================================

def get_bands_for_iso(iso, user_band=None, extend_to_40=False):
    """Return the list of bands to process for this ISO.

    Default: ACTIVE_THRESHOLDS (≥ 50).
    --extend-to-40: adds band 40 for BAND40_ISOS.
    NYISO always includes 50-65 (fill gap) when extend_to_40 is set.
    """
    if user_band is not None:
        return [user_band]

    bands = [t for t in ACTIVE_THRESHOLDS if t >= 50.0]

    if extend_to_40 and iso in BAND40_ISOS:
        # Add band 40 (from THRESHOLDS, not ACTIVE_THRESHOLDS)
        if 40.0 not in bands:
            bands = [40.0] + bands

    # For NYISO: bands 50-65 have base parquets but no lowcf.
    # Always include them when extending (they'll use P1-ceiling seeds).
    # When NOT extending, still include them — the tiered seed selection
    # will find seeds via P1 ceiling fallback.
    if iso == 'NYISO':
        for b in sorted(NYISO_FILL_BANDS):
            if b not in bands:
                bands.append(b)
        bands = sorted(bands)

    return bands


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Step 2.1d: Low-CF augmentation for nuclear-heavy ISOs')
    parser.add_argument('--iso', type=str, default='ALL',
                        choices=['ALL'] + list(TARGET_ISOS.keys()),
                        help='ISO to process (default: ALL target ISOs)')
    parser.add_argument('--band', type=float, default=None,
                        help='Single band to process (default: all ≥ 50)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Count candidates without scoring')
    parser.add_argument('--profile-one', action='store_true',
                        help='Run one band only (for profiling)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Parallel band workers (default: 1, max: 4)')
    parser.add_argument('--extend-to-40', action='store_true',
                        help='Include band 40 for MISO/NEISO/NYISO '
                             '(P1 year-0 coverage)')
    parser.add_argument('--dense-band40', action='store_true',
                        help='Generate _interp_dense at band 40 for '
                             'MISO/NEISO/NYISO')
    parser.add_argument('--max-per-band', type=int,
                        default=DEFAULT_MAX_PER_BAND,
                        help=f'Max rows per band (default: '
                             f'{DEFAULT_MAX_PER_BAND})')
    args = parser.parse_args()

    isos = (list(TARGET_ISOS.keys())
            if args.iso == 'ALL' else [args.iso])

    workers = max(1, min(args.workers, 4))

    print("Step 2.1d: Low-CF Augmentation")
    print(f"  ISOs: {isos}")
    print(f"  Workers: {workers}")
    print(f"  Numba: {'available' if HAS_NUMBA else 'UNAVAILABLE'}")
    print(f"  EF dir: {EF_DIR}")
    print(f"  Max per band: {args.max_per_band:,}")
    if args.extend_to_40:
        print(f"  Band 40 extension: ENABLED for {sorted(BAND40_ISOS)}")
    if args.dense_band40:
        print(f"  Dense band-40: ENABLED for {sorted(BAND40_ISOS)}")
    if args.dry_run:
        print("  MODE: DRY RUN")
    print()

    # Load profiles
    print("Loading generation/demand profiles...")
    t_load = time.time()
    demand_data, gen_profiles, _, _ = s1.load_data()
    print(f"  Loaded in {time.time() - t_load:.1f}s")

    # JIT warmup (avoids counting compilation in first-band timing)
    if HAS_NUMBA and not args.dry_run:
        print("Warming up Numba kernels...")
        _w_r = np.zeros((2, 4), dtype=np.float64)
        _w_z = np.zeros(2, dtype=np.float64)
        _w_d = np.ones(H, dtype=np.float64) / H
        _w_sm = np.ones((4, H), dtype=np.float64) / H
        _batch_score_storage(_w_r, _w_sm, _w_d, _w_z, _w_z, _w_z, _w_z)
        print("  JIT compilation done")

    total_written = 0
    total_mixes = 0
    t_start = time.time()

    for iso in isos:
        bands = get_bands_for_iso(iso, args.band, args.extend_to_40)

        print(f"\n{'='*60}")
        print(f"Processing {iso} "
              f"(CF floor: {TARGET_ISOS[iso]['cf_floor']}, "
              f"P1 ceiling: {P1_CF_CEILINGS.get(iso, 'n/a')})")
        print(f"  Bands: {len(bands)} ({bands[0]}–{bands[-1]})")
        print(f"{'='*60}")

        supply_profiles = s1.get_supply_profiles(iso, gen_profiles)
        hybrid_profiles = s1.load_hybrid_profiles(iso)
        demand_norm = demand_data[iso]['normalized']
        demand_arr, supply_matrix = s1.prepare_numpy_profiles(
            iso, demand_norm, supply_profiles,
            include_hybrids=True, hybrid_profiles=hybrid_profiles)
        total_demand = demand_arr.sum()

        # Get resource_cols from first available band (consistency)
        resource_cols = None
        for probe_band in bands:
            probe_df = load_ef_parquet(iso, probe_band)
            if probe_df is not None:
                resource_cols = get_resource_cols(probe_df)
                del probe_df
                break
        if resource_cols is None:
            print(f"  WARNING: No EF parquets found for {iso}")
            continue

        # --- Dense band-40 interpolation (v2.0) ---
        if args.dense_band40 and iso in BAND40_ISOS:
            out_path, n_new = generate_dense_band40(
                iso, demand_arr, supply_matrix, total_demand,
                dry_run=args.dry_run, max_per_band=args.max_per_band)
            if out_path is not None:
                total_written += 1
                total_mixes += n_new

        # --- Low-CF augmentation ---
        if workers <= 1 or args.profile_one:
            for band in bands:
                out_path, n_new = augment_band(
                    iso, band, demand_arr, supply_matrix,
                    total_demand, resource_cols, args.dry_run,
                    max_per_band=args.max_per_band)
                if out_path is not None:
                    total_written += 1
                    total_mixes += n_new
                if args.profile_one:
                    print(f"\n  --profile-one: stopping after first band")
                    break
        else:
            # Parallel bands (Numba releases GIL)
            def _run(band):
                return augment_band(
                    iso, band, demand_arr, supply_matrix,
                    total_demand, resource_cols, args.dry_run,
                    max_per_band=args.max_per_band)

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {pool.submit(_run, b): b for b in bands}
                for fut in as_completed(futs):
                    out_path, n_new = fut.result()
                    if out_path is not None:
                        total_written += 1
                        total_mixes += n_new

        if args.profile_one:
            break

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Done. Wrote {total_written} files "
          f"({total_mixes:,} new mixes) in {elapsed:.0f}s.")

    if not args.dry_run and total_written > 0:
        print(f"\nNext steps:")
        for iso in isos:
            print(f"  python scripts/step2_3a_regenerate_peakclean.py "
                  f"--iso {iso} --interp-only")


if __name__ == '__main__':
    main()
