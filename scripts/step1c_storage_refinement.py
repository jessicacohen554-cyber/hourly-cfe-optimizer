#!/usr/bin/env python3
"""Step 1C: Three-pass adaptive storage sweep with floor/fine augmentation.

Reads the union near-miss mixes from step1b and runs storage dispatch using
an adaptive funnel that minimizes compute while preserving accuracy.

Architecture:
  Pass 0 — Max screen: score each mix with maximum storage (1 combo per mix).
           Eliminates mixes that can't reach any active threshold even with
           max storage. ~40s per ISO.
  Pass 1 — Adaptive coarse: group surviving mixes by gap-to-threshold into
           buckets. Each bucket gets a right-sized storage grid (fewer levels
           for small gaps). Captures cross-dimensional interactions without
           testing irrelevant combos.
  Pass 2 — Fine targeted: for mixes near each threshold's storage-enhanced
           boundary, refine storage at 0.05% resolution.

Floor/fine mix augmentation (always on):
  Loads floor_pfs (step1b2) and fine_pfs (step1b3) parquets — generation-only
  mixes built from existing clean resource floors with incremental additions.
  Filters to mixes within the near-miss window for active thresholds,
  deduplicates against the existing pool, and runs storage sweep on them.
  Exposes lean floor+storage combinations that the standard near-miss
  pool (which is biased toward high-procurement mixes) would miss.

Storage grids are the union of V1 (near-term) and V2 (2050-oriented) caps,
giving full coverage across both ranges in a single run.

Output: data/step1-pfs-parquets/{ISO}_t{XX}_storage.parquet
        data/step1-pfs-parquets/{ISO}_t{XX}_storage_b{N}.parquet  (auto-batched)

Usage:
  python scripts/step1c_storage_refinement.py --iso CAISO
  python scripts/step1c_storage_refinement.py --iso PJM --auto-commit
  python scripts/step1c_storage_refinement.py --iso ALL
  python scripts/step1c_storage_refinement.py --iso ERCOT --batch-info
  python scripts/step1c_storage_refinement.py --iso ERCOT --thresholds "70,75,95,99" --auto-commit
"""

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

# Collision-free int64 hash per mix row (base-301, same as step1c _row_keys).
# Each resource value ∈ [0, 300], so 301^i is collision-free up to 7 dims.
_MIX_HASH_BASES = np.array([301**i for i in range(7)], dtype=np.int64)


def _mix_keys(combos):
    """Vectorized collision-free int64 hash per row. No Python loops."""
    n_res = combos.shape[1]
    return np.round(combos).astype(np.int64) @ _MIX_HASH_BASES[:n_res]


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import step1_pfs_generator as s1

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("ERROR: pyarrow required. pip install pyarrow")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'data')
STEP1D_OUTPUT_DIR = os.path.join(DATA_DIR, 'step1-pfs-parquets')

# Thresholds for storage refinement
STORAGE_THRESHOLDS = [50, 55, 60, 65, 70, 75, 80, 85, 87.5,
                      90, 92.5, 95, 97.5, 99, 99.5, 99.9, 99.99]

# Pass 0: Maximum storage levels (ceiling screen) — % of annual demand
# Physical reference: CAISO 224 TWh → 0.01% = 22,400 MWh / 5,600 MW (4hr)
# Union of V1 (near-term) and V2 (2050-oriented) caps — covers full range.
MAX_BAT4 = np.array([0.10], dtype=np.float64)
MAX_BAT8 = np.array([0.15], dtype=np.float64)
MAX_LDES = np.array([1.0], dtype=np.float64)
MAX_H2 = np.array([3.0], dtype=np.float64)
NO_H2 = np.array([0.0], dtype=np.float64)

# Pass 1: Coarse grids in % of annual demand (energy capacity as fraction of annual demand).
# CAISO: 0.01% = 22.4 GWh / 5.6 GW (comparable to real 10 GW / 40 GWh fleet).
# Union of V1 and V2 grid points — sorted, deduplicated.
FULL_BAT4 = np.array([0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.10], dtype=np.float64)
FULL_BAT8 = np.array([0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.15], dtype=np.float64)
FULL_LDES = np.array([0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0], dtype=np.float64)
FULL_H2 = np.array([0, 0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0], dtype=np.float64)

# Gap bucket boundaries (in percentage points).
# Mixes are grouped by how far their base score is from the threshold.
# Each bucket gets a grid capped at ~3× its max gap (storage efficiency headroom).
GAP_BUCKET_PP = [5, 10, 25, 50]  # bucket edges: 0-5pp, 5-10pp, 10-25pp, 25-50pp

# Pass 2: Fine storage resolution (% of annual demand)
FINE_STEP = 0.001        # 0.001 percentage points of annual demand
FINE_HALF_WIDTH = 0.005  # ±0.005pp around coarse winner
BOUNDARY_MARGIN_LOW = 2  # 2pp below threshold for boundary identification
BOUNDARY_MARGIN_HIGH = 1 # 1pp above threshold

# Max storage combos per mix in fine sweep (importance-weighted sampling)
MAX_FINE_COMBOS = 1000

# Batch sizes
NM_CHUNK = 10000         # storage cap computation chunk
MAX_MIX_BATCH = 500      # storage evaluation batch (Numba handles large batches)
CAISO_MIX_BATCH = 500    # CAISO (5D, same batch size)

# Minimum surplus days for battery deployment
MIN_SURPLUS_DAYS_FOR_BATTERY = 150

# Dominance filter: max non-dominated combos per mix per threshold
N_KEEP = 10

# Storage sweep floor
STORAGE_SWEEP_FLOOR = 0.50

# Progress save interval
PROGRESS_INTERVAL = 25   # save every N batches

# Auto-batch: flush results to disk when accumulator exceeds this per threshold.
# Targets ~10 min of results per batch file on CI (2 vCPU, Numba JIT).
# Only triggers for thresholds with large eligible mix counts.
BATCH_FLUSH_SIZE = 500_000

# High thresholds (>=97.5%) get more aggressive auto-save/commit to protect
# against CI timeouts — these are the hardest thresholds and most likely to
# time out. Smaller batch flush means more frequent saves; auto-commit is
# forced on regardless of --auto-commit flag.
HIGH_THRESHOLD_FLOOR = 97.5
HIGH_BATCH_FLUSH_SIZE = 100_000



def _load_floor_fine_mixes(iso):
    """Load floor_pfs and fine_pfs parquets for an ISO.

    Returns (combos, scores) numpy arrays, or (None, None) if no files found.
    Extracts resource columns matching near-miss schema and uses
    hourly_match_score as the base score.
    """
    rtypes = s1.get_resource_types(iso)
    all_combos = []
    all_scores = []

    for suffix in ('floor_pfs', 'fine_pfs'):
        pattern = os.path.join(s1.STEP1_RAW_PFS_PARQUET_DIR,
                               f'{iso}_t*_{suffix}.parquet')
        import glob as _glob
        files = sorted(_glob.glob(pattern))
        for fpath in files:
            table = pq.read_table(fpath)
            if table.num_rows == 0:
                continue
            combos = np.column_stack(
                [table.column(rt).to_numpy() for rt in rtypes])
            scores = table.column('hourly_match_score').to_numpy()
            all_combos.append(combos)
            all_scores.append(scores)

    if not all_combos:
        return None, None

    combos = np.vstack(all_combos)
    scores = np.concatenate(all_scores)

    # Deduplicate within floor/fine pool (same mix can appear at multiple thresholds)
    keys = _mix_keys(combos)
    _, unique_idx = np.unique(keys, return_index=True)
    combos = combos[unique_idx]
    scores = scores[unique_idx]

    return combos, scores


def _augment_with_floor_fine(original_load_fn):
    """Wrap loader to inject floor/fine PFS mixes into the near-miss pool.

    Loads floor_pfs and fine_pfs parquets, filters to mixes within the
    near-miss window for active thresholds, deduplicates against existing
    pool, and returns the combined pool. Forces Pass 0 re-run since
    floor/fine mixes don't have max_storage_scores.
    """
    def augmented_loader(iso, active_thresholds):
        nm_combos, nm_scores, nm_max_scores = original_load_fn(iso, active_thresholds)
        if nm_combos is None:
            return None, None, None

        n_original = len(nm_combos)

        ff_combos, ff_scores = _load_floor_fine_mixes(iso)
        if ff_combos is None or len(ff_combos) == 0:
            print(f"  Floor/fine augmentation: no files found for {iso}")
            return nm_combos, nm_scores, nm_max_scores

        n_loaded = len(ff_combos)

        # Filter to mixes in the near-miss window for at least one active threshold
        eligible = np.zeros(len(ff_combos), dtype=bool)
        for t in active_thresholds:
            target = t / 100.0
            nm_width = get_near_miss_width(t)
            nm_floor = max(target - nm_width, STORAGE_SWEEP_FLOOR)
            t_mask = (ff_scores < target) & (ff_scores >= nm_floor)
            eligible |= t_mask

        ff_combos = ff_combos[eligible]
        ff_scores = ff_scores[eligible]

        if len(ff_combos) == 0:
            print(f"  Floor/fine augmentation: {n_loaded:,} loaded, "
                  f"0 in near-miss window")
            return nm_combos, nm_scores, nm_max_scores

        # Deduplicate against existing near-miss pool (vectorized np.isin)
        n_cols = min(nm_combos.shape[1], len(_MIX_HASH_BASES))
        nm_keys = nm_combos.astype(np.int64)[:, :n_cols] @ _MIX_HASH_BASES[:n_cols]
        ff_keys = ff_combos.astype(np.int64)[:, :n_cols] @ _MIX_HASH_BASES[:n_cols]
        new_mask = ~np.isin(ff_keys, nm_keys)
        ff_combos = ff_combos[new_mask]
        ff_scores = ff_scores[new_mask]
        n_new = len(ff_combos)

        if n_new == 0:
            print(f"  Floor/fine augmentation: {n_loaded:,} loaded, "
                  f"all duplicates of existing pool")
            return nm_combos, nm_scores, nm_max_scores

        combined_combos = np.vstack([nm_combos, ff_combos])
        combined_scores = np.concatenate([nm_scores, ff_scores])
        print(f"  Floor/fine augmentation: {n_loaded:,} loaded → "
              f"{int(eligible.sum()):,} in near-miss window → "
              f"{n_new:,} new unique mixes")
        print(f"  Combined pool: {n_original:,} original + {n_new:,} floor/fine = "
              f"{len(combined_combos):,} total")

        # Force Pass 0 re-run (floor/fine mixes don't have max_storage_scores)
        return combined_combos, combined_scores, None

    return augmented_loader


def get_near_miss_width(threshold):
    """Near-miss half-width by threshold range.

    Maximum pp below the threshold a mix's base score can be and still
    be a candidate for storage enhancement.  High-solar mixes can have
    low base scores but massive curtailment surplus that storage captures,
    so sub-85% thresholds need a wider window.  The surplus >= gap filter
    in the binning loop is the primary gate that prevents candidate bloat.
    """
    if threshold >= 99:
        return 0.20   # 20pp — last-mile, few mixes
    elif threshold >= 85:
        return 0.20   # 20pp
    return 0.25        # 25pp — high-solar + storage mixes need wider window


# ══════════════════════════════════════════════════════════════════════════════
# LOAD NEAR-MISS DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_near_miss(iso):
    """Load union near-miss mixes from step1c output.

    Returns (combos, scores, max_storage_scores).
    max_storage_scores is None if the column is absent (old near_miss file).
    """
    path = os.path.join(s1.STEP1_RAW_PFS_PARQUET_DIR,
                        f'{iso}_near_miss.parquet')
    if not os.path.exists(path):
        return None, None, None

    table = pq.read_table(path)
    rtypes = s1.get_resource_types(iso)
    combos = np.column_stack([table.column(rt).to_numpy() for rt in rtypes])
    scores = table.column('base_score').to_numpy()

    # Read pre-computed max-storage scores if available (step1c full_filter)
    max_storage_scores = None
    if 'max_storage_score' in table.column_names:
        max_storage_scores = table.column('max_storage_score').to_numpy()
        print(f"  Near-miss has max_storage_score column — Pass 0 can be skipped")

    return combos, scores, max_storage_scores


def load_coarse_cache(iso):
    """Load coarse cache mixes from step1b output.

    Used as fallback when the near-miss file has no eligible mixes at low
    thresholds (e.g., ERCOT 50-75% where all near-miss mixes score >80%).
    The coarse cache contains mixes at all score levels.
    """
    path = os.path.join(s1.STEP1_RAW_PFS_PARQUET_DIR,
                        f'{iso}_coarse_cache.parquet')
    if not os.path.exists(path):
        return None, None

    table = pq.read_table(path)
    rtypes = s1.get_resource_types(iso)
    combos = np.column_stack([table.column(rt).to_numpy() for rt in rtypes])
    scores = table.column('score').to_numpy()
    return combos, scores


def load_mixes_with_coarse_fallback(iso, active_thresholds):
    """Load near-miss mixes, augmenting with coarse cache for low thresholds.

    For ISOs like ERCOT where the near-miss file has no mixes below 80%,
    the coarse cache provides low-score mixes that are storage candidates
    at lower thresholds.

    Returns (combos, scores, max_storage_scores) with coarse fallback mixes
    appended if needed. max_storage_scores is None if not available in
    near_miss or if coarse fallback mixes were added (forces Pass 0).
    """
    nm_combos, nm_scores, nm_max_scores = load_near_miss(iso)
    if nm_combos is None:
        return None, None, None

    # Check if near-miss covers the lowest requested threshold
    min_threshold = min(active_thresholds)
    min_target = min_threshold / 100.0
    n_eligible_low = int(np.sum(nm_scores < min_target))

    if n_eligible_low > 0:
        # Near-miss has eligible mixes for all thresholds — no fallback needed
        return nm_combos, nm_scores, nm_max_scores

    # Need coarse fallback: near-miss has no mixes below lowest threshold
    cc_combos, cc_scores = load_coarse_cache(iso)
    if cc_combos is None:
        print(f"  WARNING: No coarse cache for {iso}, cannot augment near-miss")
        return nm_combos, nm_scores, nm_max_scores

    # Only add coarse mixes that are in the near-miss window for any threshold
    # and aren't already in the near-miss file (vectorized np.isin)
    nm_keys = _mix_keys(nm_combos)
    cc_keys = _mix_keys(cc_combos)
    new_mask = ~np.isin(cc_keys, nm_keys)

    # Filter to mixes in the near-miss window for at least one active threshold
    nm_eligible = np.zeros(len(cc_combos), dtype=bool)
    for t in active_thresholds:
        target = t / 100.0
        nm_width = get_near_miss_width(t)
        nm_floor = max(target - nm_width, STORAGE_SWEEP_FLOOR)
        t_mask = (cc_scores < target) & (cc_scores >= nm_floor)
        nm_eligible |= t_mask

    add_mask = new_mask & nm_eligible
    n_add = int(add_mask.sum())

    if n_add == 0:
        return nm_combos, nm_scores, nm_max_scores

    print(f"  Coarse cache fallback: adding {n_add:,} mixes "
          f"(near-miss had 0 eligible below {min_threshold}%)")

    combined_combos = np.vstack([nm_combos, cc_combos[add_mask]])
    combined_scores = np.concatenate([nm_scores, cc_scores[add_mask]])
    # Coarse fallback mixes don't have max_storage_scores — force Pass 0
    return combined_combos, combined_scores, None


# ══════════════════════════════════════════════════════════════════════════════
# PASS 1: COARSE GLOBAL STORAGE SWEEP
# ══════════════════════════════════════════════════════════════════════════════

def _get_storage_constants():
    """Return all storage dispatch constants as a dict."""
    return dict(
        batt_eff=s1.BATTERY_EFFICIENCY,
        batt8_eff=s1.BATTERY8_EFFICIENCY,
        ldes_eff=s1.LDES_EFFICIENCY,
        batt4_dur=s1.BATTERY_DURATION_HOURS,
        batt8_dur=s1.BATTERY8_DURATION_HOURS,
        ldes_dur=s1.LDES_DURATION_HOURS,
        ldes_window=s1.LDES_WINDOW_DAYS * 24,
        batt8_window=48,
        h2_eff=s1.H2_EFFICIENCY,
        h2_dur=float(s1.H2_DURATION_HOURS),
        h2_window=s1.H2_WINDOW_DAYS * 24,
    )


def _build_bucket_grid(max_gap_pp, include_h2):
    """Build storage grid sized to the gap bucket.

    Returns (bat4_arr, bat8_arr, ldes_arr, h2_arr) numpy arrays.
    Grid values are in % of annual demand (bat4 up to 0.06, bat8 up to 0.08,
    LDES up to 0.5). The full grid is compact — always used in full.
    The gap-bucket cap (in pp) is kept for future-proofing but always
    exceeds grid maxes at current scale, so all points are selected.
    """
    cap = max_gap_pp * 3.0

    # Floor values match grid maxes to ensure full grid is always used.
    # cap (in pp) >> grid max (% of annual demand), so floor dominates.
    bat4 = FULL_BAT4[FULL_BAT4 <= max(cap, 0.06)]
    if len(bat4) < 2:
        bat4 = FULL_BAT4[:2]

    bat8 = FULL_BAT8[FULL_BAT8 <= max(cap, 0.08)]
    if len(bat8) < 2:
        bat8 = FULL_BAT8[:2]

    ldes = FULL_LDES[FULL_LDES <= max(cap, 0.5)]
    if len(ldes) < 2:
        ldes = FULL_LDES[:2]

    h2 = FULL_H2 if include_h2 else NO_H2

    return bat4, bat8, ldes, h2


def run_max_screen(nm_combos, nm_base_scores, demand_arr, supply_matrix,
                   active_thresholds, sc):
    """Pass 0: Score each mix with maximum storage to find the ceiling.

    Returns max_scores array (N,) — the best score each mix can achieve
    with maximum storage. Mixes where max_score < min(active_thresholds)
    are guaranteed to be unreachable and can be eliminated.
    """
    n_mixes = len(nm_combos)
    include_h2 = any(t >= 95 for t in active_thresholds)
    h2_arr = MAX_H2 if include_h2 else NO_H2

    print(f"\n  Pass 0 — Max-storage screen: {n_mixes:,} mixes "
          f"(H2={'yes' if include_h2 else 'no'})")

    max_scores = np.empty(n_mixes, dtype=np.float64)
    batch_size = MAX_MIX_BATCH
    t0 = time.time()

    for cs in range(0, n_mixes, batch_size):
        ce = min(cs + batch_size, n_mixes)
        fracs = nm_combos[cs:ce].astype(np.float64) / 100.0
        supply = fracs @ supply_matrix
        n_batch = ce - cs

        result = s1._batch_mixes_storage_screen(
            demand_arr, supply, 1.0, n_batch,
            MAX_BAT4, MAX_BAT8, MAX_LDES,
            1, 1, 1,
            sc['batt_eff'], sc['batt8_eff'], sc['ldes_eff'],
            sc['batt4_dur'], sc['batt8_dur'], sc['ldes_dur'],
            sc['ldes_window'], sc['batt8_window'],
            h2_arr, len(h2_arr), sc['h2_eff'], sc['h2_dur'], sc['h2_window'])

        max_scores[cs:ce] = result[:, 0]

    elapsed = time.time() - t0
    print(f"    Completed in {elapsed:.1f}s")
    return max_scores


def run_adaptive_coarse(iso, nm_combos, nm_base_scores, max_scores,
                        demand_arr, supply_matrix, active_thresholds, sc,
                        global_idx_map=None, on_bucket_done=None):
    """Pass 1: Adaptive coarse storage sweep with gap-bucketed grids.

    Groups mixes by their gap to the closest active threshold, builds
    a right-sized storage grid per bucket, and runs the Numba kernel
    on each bucket. This avoids testing LDES=25% on mixes that only
    need 3pp of storage boost.

    Args:
        global_idx_map: Optional array mapping local indices → full nm_combos
            indices. When provided, result mix_idx values use full indices
            (eliminating the need for post-hoc remapping in process_iso).
        on_bucket_done: Optional callback(results, active_thresholds) called
            after each bucket completes. Used for auto-flush to batch files.

    Returns per-threshold feasible lists:
        results[threshold] = list of (mix_idx, bat4, bat8, ldes, h2, score)
    """
    n_mixes = len(nm_combos)
    batch_size = CAISO_MIX_BATCH if iso == 'CAISO' else MAX_MIX_BATCH
    include_h2 = any(t >= 95 for t in active_thresholds)

    # Compute curtailment flags and surplus days (needed for bucketing/gating)
    print(f"\n  Pass 1 — Adaptive coarse sweep")
    print(f"    Computing curtailment stats...")
    surplus_days = np.empty(n_mixes, dtype=np.int64)
    hc_arr = np.empty(n_mixes, dtype=np.int64)

    for cs in range(0, n_mixes, NM_CHUNK):
        ce = min(cs + NM_CHUNK, n_mixes)
        chunk_fracs = nm_combos[cs:ce].astype(np.float64) / 100.0
        chunk_supply = chunk_fracs @ supply_matrix
        chunk_n = ce - cs

        _, _, _, chc, csd = s1._batch_compute_storage_caps(
            demand_arr, chunk_supply, 1.0, chunk_n,
            sc['batt4_dur'], sc['batt8_dur'], sc['ldes_dur'])
        hc_arr[cs:ce] = chc
        surplus_days[cs:ce] = csd

    has_curtailment = hc_arr.astype(bool)

    # For each mix, compute min gap to any active threshold it could serve.
    # This determines which bucket (grid size) to use.
    # gap = max(0, target - base_score) for the closest reachable threshold.
    min_gap_pp = np.full(n_mixes, 999.0, dtype=np.float64)
    for t in active_thresholds:
        target = t / 100.0
        gap = np.maximum(target - nm_base_scores, 0.0) * 100.0
        reachable = max_scores >= target
        mix_gap = np.where(reachable, gap, 999.0)
        min_gap_pp = np.minimum(min_gap_pp, mix_gap)

    # ── Bucket mixes by gap range ──
    bucket_edges = [0] + GAP_BUCKET_PP
    results = {t: [] for t in active_thresholds}
    total_feasible = 0
    total_kernel_evals = 0

    for bi in range(len(bucket_edges) - 1):
        lo_pp, hi_pp = bucket_edges[bi], bucket_edges[bi + 1]
        mask = (min_gap_pp >= lo_pp) & (min_gap_pp < hi_pp) & has_curtailment
        bucket_idx = np.where(mask)[0]
        if len(bucket_idx) == 0:
            continue

        # Build grid for this bucket
        b4_grid, b8_grid, ldes_grid, h2_grid = _build_bucket_grid(
            hi_pp, include_h2)
        n_combos = len(b4_grid) * len(b8_grid) * len(ldes_grid) * len(h2_grid)
        n_b4, n_b8, n_l, n_h2 = (len(b4_grid), len(b8_grid),
                                   len(ldes_grid), len(h2_grid))

        print(f"    Bucket {lo_pp}-{hi_pp}pp: {len(bucket_idx):,} mixes × "
              f"{n_combos:,} combos "
              f"(b4={n_b4} b8={n_b8} ldes={n_l} h2={n_h2})")

        # Pre-sort combo indices by total storage (ascending) — vectorized meshgrid.
        mg_b4i, mg_b8i, mg_li, mg_h2i = np.meshgrid(
            np.arange(n_b4), np.arange(n_b8), np.arange(n_l), np.arange(n_h2),
            indexing='ij')
        all_bp = b4_grid[mg_b4i.ravel()]
        all_b8p = b8_grid[mg_b8i.ravel()]
        all_lp = ldes_grid[mg_li.ravel()]
        all_h2p = h2_grid[mg_h2i.ravel()]
        all_flat = (mg_b4i.ravel() * n_b8 * n_l * n_h2 +
                    mg_b8i.ravel() * n_l * n_h2 +
                    mg_li.ravel() * n_h2 + mg_h2i.ravel())
        all_total = all_bp + all_b8p + all_lp + all_h2p

        # Filter out all-zero combo, sort by total storage
        nonzero = all_total > 0
        sort_idx = np.argsort(all_total[nonzero])
        n_sorted = int(nonzero.sum())
        sc_flat_idx = all_flat[nonzero][sort_idx].astype(np.intp)
        sc_bp = all_bp[nonzero][sort_idx]
        sc_b8p = all_b8p[nonzero][sort_idx]
        sc_lp = all_lp[nonzero][sort_idx]
        sc_h2p = all_h2p[nonzero][sort_idx]

        sc_has_battery = (sc_bp > 0) | (sc_b8p > 0)

        # Precompute static dominance matrix: dom[i,j] = True means combo i
        # dominates combo j (i has ≤ values in ALL 4 storage dims).
        # This is a property of the grid, not the mixes — compute once per bucket.
        dom = ((sc_bp[:, np.newaxis] <= sc_bp[np.newaxis, :]) &
               (sc_b8p[:, np.newaxis] <= sc_b8p[np.newaxis, :]) &
               (sc_lp[:, np.newaxis] <= sc_lp[np.newaxis, :]) &
               (sc_h2p[:, np.newaxis] <= sc_h2p[np.newaxis, :]))
        np.fill_diagonal(dom, False)  # combo doesn't dominate itself

        # Process in batches — fully vectorized across mixes
        for batch_start in range(0, len(bucket_idx), batch_size):
            batch_end = min(batch_start + batch_size, len(bucket_idx))
            b_idx = bucket_idx[batch_start:batch_end]
            n_batch = len(b_idx)

            batch_fracs = nm_combos[b_idx].astype(np.float64) / 100.0
            batch_supply = batch_fracs @ supply_matrix

            batch_scores = s1._batch_mixes_storage_screen(
                demand_arr, batch_supply, 1.0, n_batch,
                b4_grid, b8_grid, ldes_grid,
                n_b4, n_b8, n_l,
                sc['batt_eff'], sc['batt8_eff'], sc['ldes_eff'],
                sc['batt4_dur'], sc['batt8_dur'], sc['ldes_dur'],
                sc['ldes_window'], sc['batt8_window'],
                h2_grid, n_h2, sc['h2_eff'], sc['h2_dur'], sc['h2_window'])

            total_kernel_evals += n_batch * n_combos

            # Pre-compute per-mix constraints (vectorized)
            battery_ok = surplus_days[b_idx] >= MIN_SURPLUS_DAYS_FOR_BATTERY

            # Score matrix reindexed to sorted combo order (computed once per batch)
            sorted_scores = batch_scores[:, sc_flat_idx]  # (n_batch, n_sorted)

            # Pre-compute per-threshold base eligibility
            base_below = {}
            for t in active_thresholds:
                base_below[t] = nm_base_scores[b_idx] < (t / 100.0)

            # Battery ban mask (same for all thresholds)
            battery_ban = np.outer(~battery_ok, sc_has_battery) if sc_has_battery.any() else None

            # Vectorized dominance-filtered extraction.
            # Iterates combos in sorted order (ascending total storage).
            # For each combo, vectorized feasibility check across all mixes.
            # When a combo is recorded for a mix, all combos it dominates
            # are blocked for that mix via the precomputed dominance matrix.
            for t in active_thresholds:
                target = t / 100.0

                # Per-mix blocked mask and record count
                blocked = np.zeros((n_batch, n_sorted), dtype=bool)
                n_recorded = np.zeros(n_batch, dtype=np.int32)

                for ci in range(n_sorted):
                    # Early exit: all mixes have N_KEEP combos
                    if n_recorded.min() >= N_KEEP:
                        break

                    # H2 gate
                    if sc_h2p[ci] > 0 and t < 95:
                        continue

                    fidx = int(sc_flat_idx[ci])
                    scores_col = batch_scores[:, fidx]

                    # Vectorized feasibility
                    hits = ((scores_col >= target) &
                            base_below[t] &
                            ~blocked[:, ci] &
                            (n_recorded < N_KEEP))

                    if sc_has_battery[ci] and battery_ban is not None:
                        hits &= battery_ok

                    if not hits.any():
                        continue

                    # Record results for all hit mixes (bulk vectorized)
                    hit_idx = np.where(hits)[0]
                    bp = float(sc_bp[ci])
                    b8p = float(sc_b8p[ci])
                    lp = float(sc_lp[ci])
                    h2p = float(sc_h2p[ci])
                    score_pcts = np.round(scores_col[hit_idx] * 100, 2)

                    # Vectorized index mapping — no per-element Python loop
                    local_mis = b_idx[hit_idx]
                    if global_idx_map is not None:
                        mis = global_idx_map[local_mis]
                    else:
                        mis = local_mis
                    results[t].extend(
                        [(int(mis[k]), bp, b8p, lp, h2p,
                          float(score_pcts[k]))
                         for k in range(len(hit_idx))])
                    total_feasible += len(hit_idx)

                    # Block dominated combos for hit mixes (vectorized)
                    blocked[hits] |= dom[ci]
                    n_recorded[hits] += 1

        # Auto-flush callback after each bucket completes
        if on_bucket_done is not None:
            on_bucket_done(results, active_thresholds)

    print(f"    Pass 1 complete: {total_feasible:,} feasible "
          f"(dominance-filtered, N≤{N_KEEP} per mix), "
          f"{total_kernel_evals:,.0f} kernel evals")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# PASS 2: FINE TARGETED STORAGE
# ══════════════════════════════════════════════════════════════════════════════

def _fine_range(center, step, half_width, cap_lo, cap_hi):
    """Generate fine sweep range around center value."""
    lo = max(cap_lo, center - half_width)
    hi = min(cap_hi, center + half_width)
    vals = []
    v = lo
    while v <= hi + step * 0.01:
        vals.append(round(v, 4))
        v += step
    return np.array(vals, dtype=np.float64)


def _vectorized_dominance_filter(b4_vals, b8_vals, l_vals, h2_vals, scores,
                                  target, n_keep):
    """Vectorized dominance filter for feasible storage combos.

    Returns list of (b4, b8, ldes, h2, score) tuples — non-dominated,
    sorted by ascending total storage, capped at n_keep.
    """
    # Filter to feasible
    feasible_mask = (scores >= 0) & (scores >= target)
    if not feasible_mask.any():
        return []

    fb4 = b4_vals[feasible_mask]
    fb8 = b8_vals[feasible_mask]
    fl = l_vals[feasible_mask]
    fh2 = h2_vals[feasible_mask]
    fsc = scores[feasible_mask]
    total_s = fb4 + fb8 + fl + fh2

    # Sort ascending by total storage
    sort_idx = np.argsort(total_s)
    fb4, fb8, fl, fh2, fsc = (fb4[sort_idx], fb8[sort_idx],
                               fl[sort_idx], fh2[sort_idx], fsc[sort_idx])

    # Greedy dominance filter (small N after feasibility filter, fast enough)
    recorded = []
    results = []
    for i in range(len(fb4)):
        b4v, b8v, lv, h2v = fb4[i], fb8[i], fl[i], fh2[i]
        dominated = False
        for rb4, rb8, rld, rh2 in recorded:
            if rb4 <= b4v and rb8 <= b8v and rld <= lv and rh2 <= h2v:
                dominated = True
                break
        if dominated:
            continue
        recorded.append((b4v, b8v, lv, h2v))
        results.append((float(b4v), float(b8v), float(lv), float(h2v),
                        float(fsc[i])))
        if len(recorded) >= n_keep:
            break
    return results


def run_fine_storage(iso, threshold, nm_combos, coarse_results,
                     demand_arr, supply_matrix):
    """Fine 0.05% storage refinement for one threshold's boundary mixes.

    Groups boundary mixes by coarse winner to batch Numba kernel calls,
    enabling prange parallelism across mixes within each group.
    Vectorized combo extraction replaces 4-deep nested loops.

    Returns list of (mix_idx, bat4, bat8, ldes, h2, score).
    """
    target = threshold / 100.0
    batch_size = CAISO_MIX_BATCH if iso == 'CAISO' else MAX_MIX_BATCH

    # Storage constants
    sc = _get_storage_constants()

    if not coarse_results:
        return []

    # Find boundary mixes: score within [target-2pp, target+1pp] in coarse results
    boundary_mixes = {}  # mix_idx → best (bat4, bat8, ldes, h2, score)
    low_bound = (target - BOUNDARY_MARGIN_LOW / 100.0) * 100
    high_bound = (target + BOUNDARY_MARGIN_HIGH / 100.0) * 100

    for (mi, bp, b8p, lp, h2p, score_pct) in coarse_results:
        if low_bound <= score_pct <= high_bound:
            if mi not in boundary_mixes or score_pct > boundary_mixes[mi][-1]:
                boundary_mixes[mi] = (bp, b8p, lp, h2p, score_pct)

    if not boundary_mixes:
        return []

    # ── Group boundary mixes by coarse winner → batch kernel calls ──
    # Mixes with the same coarse winner get the same fine grid, so they
    # can share a single kernel call with N>1 (enabling Numba prange).
    winner_groups = {}  # (best_b4, best_b8, best_l, best_h2) → [mix_idx, ...]
    for mi, (best_b4, best_b8, best_l, best_h2, _) in boundary_mixes.items():
        key = (best_b4, best_b8, best_l, best_h2)
        if key not in winner_groups:
            winner_groups[key] = []
        winner_groups[key].append(mi)

    fine_results = []

    for (best_b4, best_b8, best_l, best_h2), mix_indices in winner_groups.items():
        # Generate fine ranges around this coarse winner
        fine_b4 = _fine_range(best_b4, FINE_STEP, FINE_HALF_WIDTH,
                              0, float(FULL_BAT4[-1]))
        fine_b8 = _fine_range(best_b8, FINE_STEP, FINE_HALF_WIDTH,
                              0, float(FULL_BAT8[-1]))
        fine_l = _fine_range(best_l, FINE_STEP, FINE_HALF_WIDTH,
                             0, float(FULL_LDES[-1]))

        if threshold >= 95 and best_h2 > 0:
            fine_h2 = _fine_range(best_h2, FINE_STEP * 10,
                                  FINE_HALF_WIDTH * 5, 0, 25)
        else:
            fine_h2 = np.array([0.0], dtype=np.float64)

        # Check if cross-product exceeds limit
        n_combos = len(fine_b4) * len(fine_b8) * len(fine_l) * len(fine_h2)
        if n_combos > MAX_FINE_COMBOS:
            fine_b8 = np.array([best_b8], dtype=np.float64)
            fine_h2 = np.array([best_h2 if best_h2 > 0 else 0.0],
                               dtype=np.float64)
            n_combos = len(fine_b4) * len(fine_b8) * len(fine_l) * len(fine_h2)

        n_b4 = len(fine_b4)
        n_b8 = len(fine_b8)
        n_l = len(fine_l)
        n_h2 = len(fine_h2)

        # Pre-compute combo value arrays (vectorized meshgrid)
        mg_b4, mg_b8, mg_l, mg_h2 = np.meshgrid(
            fine_b4, fine_b8, fine_l, fine_h2, indexing='ij')
        combo_b4 = mg_b4.ravel()
        combo_b8 = mg_b8.ravel()
        combo_l = mg_l.ravel()
        combo_h2 = mg_h2.ravel()

        # Process in batches to limit memory
        for bs in range(0, len(mix_indices), batch_size):
            be = min(bs + batch_size, len(mix_indices))
            batch_mis = mix_indices[bs:be]
            n_batch = len(batch_mis)

            # Build batched supply profiles → (n_batch, 8760)
            batch_fracs = nm_combos[batch_mis].astype(np.float64) / 100.0
            batch_supply = batch_fracs @ supply_matrix

            # Single batched kernel call — Numba prange across n_batch mixes
            batch_scores = s1._batch_mixes_storage_screen(
                demand_arr, batch_supply, 1.0, n_batch,
                fine_b4, fine_b8, fine_l,
                n_b4, n_b8, n_l,
                sc['batt_eff'], sc['batt8_eff'], sc['ldes_eff'],
                sc['batt4_dur'], sc['batt8_dur'], sc['ldes_dur'],
                sc['ldes_window'], sc['batt8_window'],
                fine_h2, n_h2, sc['h2_eff'], sc['h2_dur'], sc['h2_window'])

            # Extract results per mix (vectorized combo extraction + dominance)
            for local_i in range(n_batch):
                mi = batch_mis[local_i]
                scores_flat = batch_scores[local_i]  # (n_combos,)

                non_dom = _vectorized_dominance_filter(
                    combo_b4, combo_b8, combo_l, combo_h2,
                    scores_flat, target, N_KEEP)

                for b4v, b8v, lv, h2v, sc_val in non_dom:
                    fine_results.append(
                        (int(mi), b4v, b8v, lv, h2v,
                         round(sc_val * 100, 2)))

    return fine_results


# ══════════════════════════════════════════════════════════════════════════════
# SAVE / COMMIT
# ══════════════════════════════════════════════════════════════════════════════

def save_storage_results(iso, threshold, nm_combos, results, rtypes,
                         batch_num=None):
    """Save storage-enhanced feasible mixes for one threshold.

    Args:
        batch_num: If set, saves as {ISO}_t{XX}_storage_b{N}.parquet
    """
    if not results:
        return

    os.makedirs(STEP1D_OUTPUT_DIR, exist_ok=True)

    n = len(results)
    # Convert results to structured numpy array for vectorized column extraction
    result_arr = np.array(results, dtype=np.float64)  # (N, 6)
    result_indices = result_arr[:, 0].astype(np.intp)

    data = {
        'iso': [iso] * n,
        'threshold': [float(threshold)] * n,
    }
    for j, rt in enumerate(rtypes):
        data[rt] = nm_combos[result_indices, j].astype(np.float64)
    data['battery_dispatch_pct'] = result_arr[:, 1]
    data['battery8_dispatch_pct'] = result_arr[:, 2]
    data['ldes_dispatch_pct'] = result_arr[:, 3]
    data['h2_dispatch_pct'] = result_arr[:, 4]
    data['hourly_match_score'] = result_arr[:, 5]

    table = pa.table(data)
    t_str = s1._normalize_threshold_str(threshold)
    if batch_num is not None:
        fname = f'{iso}_t{t_str}_storage_b{batch_num}.parquet'
    else:
        fname = f'{iso}_t{t_str}_storage.parquet'
    out_path = os.path.join(STEP1D_OUTPUT_DIR, fname)
    pq.write_table(table, out_path, compression='snappy')
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    return out_path


def git_commit_threshold(iso, threshold, phase_label, auto_commit,
                         batch_num=None):
    """Commit and push threshold results."""
    if not auto_commit:
        return

    try:
        t_str = s1._normalize_threshold_str(threshold)
        if batch_num is not None:
            fname = f'{iso}_t{t_str}_storage_b{batch_num}.parquet'
        else:
            fname = f'{iso}_t{t_str}_storage.parquet'
        out_path = os.path.join(STEP1D_OUTPUT_DIR, fname)
        if not os.path.exists(out_path):
            return

        subprocess.run(['git', 'add', '-f', out_path],
                       check=True, capture_output=True, text=True)

        result = subprocess.run(['git', 'diff', '--cached', '--quiet'],
                                capture_output=True)
        if result.returncode == 0:
            return

        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        batch_label = f" batch {batch_num}" if batch_num is not None else ""
        msg = (f"Storage {phase_label}: {iso} {threshold}%{batch_label} "
               f"({size_mb:.1f} MB) — auto-commit")
        subprocess.run(['git', 'commit', '-m', msg],
                       check=True, capture_output=True, text=True)

        for attempt in range(1, 5):
            result = subprocess.run(
                ['git', 'push', '-u', 'origin', 'HEAD'],
                capture_output=True, text=True)
            if result.returncode == 0:
                batch_str = f" b{batch_num}" if batch_num is not None else ""
                print(f"    [auto-commit] {iso} {threshold}%{batch_str} committed & pushed")
                return
            if attempt < 4:
                time.sleep(2 ** attempt)

        print(f"    [auto-commit] Push failed — committed locally")

    except subprocess.CalledProcessError as e:
        print(f"    [auto-commit] Git error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# MANIFEST (resume support)
# ══════════════════════════════════════════════════════════════════════════════

def _manifest_path(iso):
    return os.path.join(STEP1D_OUTPUT_DIR, f'{iso}_storage_manifest.json')


def _compute_code_hash(iso):
    h = hashlib.sha256()
    for fname in ['step1c_storage_refinement.py', 'step1_pfs_generator.py']:
        fpath = os.path.join(SCRIPT_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath, 'rb') as f:
                h.update(f.read())
    nm_path = os.path.join(s1.STEP1_RAW_PFS_PARQUET_DIR,
                           f'{iso}_near_miss.parquet')
    if os.path.exists(nm_path):
        stat = os.stat(nm_path)
        h.update(f"{nm_path}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def _load_manifest(iso, current_hash):
    mpath = _manifest_path(iso)
    if not os.path.exists(mpath):
        return None
    try:
        with open(mpath, 'r') as f:
            manifest = json.load(f)
        if manifest.get('code_hash') != current_hash:
            os.remove(mpath)
            return None
        return manifest
    except (json.JSONDecodeError, OSError):
        return None


def _save_manifest(iso, code_hash, pass1_thresholds, pass2_done):
    os.makedirs(STEP1D_OUTPUT_DIR, exist_ok=True)
    manifest = {
        'code_hash': code_hash,
        'pass1_thresholds': sorted(pass1_thresholds),
        'pass2_done': sorted(pass2_done),
    }
    with open(_manifest_path(iso), 'w') as f:
        json.dump(manifest, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def process_iso(iso, auto_commit=False, thresholds_filter=None):
    """Run three-pass adaptive storage sweep for one ISO.

    Pass 0: Max-storage screen (1 combo/mix) → eliminates unreachable mixes
    Pass 1: Adaptive coarse (gap-bucketed grids) → finds feasible combos
    Pass 2: Fine 0.05% refinement → sharpens boundary mixes

    Args:
        thresholds_filter: list of thresholds to process (None = all)
    """
    iso_start = time.time()
    rtypes = s1.get_resource_types(iso)
    n_res = len(rtypes)

    print(f"\n{'=' * 70}")
    print(f"  Step 1d Fine Storage — {iso}")
    print(f"  Resources: {n_res}D ({', '.join(rtypes)})")
    print(f"  Numba: {'enabled' if s1.HAS_NUMBA else 'disabled'}")
    print(f"{'=' * 70}")

    # ── Determine which thresholds to process ──
    active_thresholds = thresholds_filter if thresholds_filter else STORAGE_THRESHOLDS
    if thresholds_filter:
        print(f"  Threshold filter: {active_thresholds}")

    # ── Resume logic ──
    code_hash = _compute_code_hash(iso)
    manifest = _load_manifest(iso, code_hash)
    # Per-threshold tracking (backward-compat: old bool pass1_done → empty set)
    if manifest:
        old_p1 = manifest.get('pass1_done', False)
        if isinstance(old_p1, bool) and old_p1:
            # Old manifest had pass1_done=True for all thresholds at once.
            # Migrate: treat pass2_done as the set of completed thresholds.
            pass1_thresholds = set(manifest.get('pass2_done', []))
        else:
            pass1_thresholds = set(manifest.get('pass1_thresholds', []))
    else:
        pass1_thresholds = set()
    pass2_done = set(manifest.get('pass2_done', [])) if manifest else set()

    # ── Load near-miss mixes (with coarse cache fallback for low thresholds) ──
    nm_combos, nm_base_scores, precomputed_max_scores = \
        load_mixes_with_coarse_fallback(iso, active_thresholds)
    if nm_combos is None:
        print(f"  ERROR: No near-miss data for {iso}. Run step1c first.")
        return

    n_raw = len(nm_combos)
    print(f"  Near-miss mixes (raw): {n_raw:,}")

    # ── Load EIA data (needed for curtailment bridge filter) ──
    print(f"  Loading EIA data...")
    demand_data, gen_profiles, _, _ = s1.load_data()
    demand_norm = demand_data[iso]['normalized']
    supply_profiles = s1.get_supply_profiles(iso, gen_profiles)
    demand_arr, supply_matrix = s1.prepare_numpy_profiles(
        iso, demand_norm, supply_profiles)

    # ── Storage dispatch constants (shared across all passes) ──
    sc = _get_storage_constants()

    # ── JIT warmup ──
    if s1.HAS_NUMBA:
        print(f"  Warming up Numba JIT...")
        dummy_supply = np.ones((1, s1.H), dtype=np.float64)
        dummy_b = np.array([0.0, 1.0], dtype=np.float64)
        dummy_l = np.array([0.0], dtype=np.float64)
        dummy_h = np.array([0.0], dtype=np.float64)
        s1._batch_mixes_storage_screen(
            demand_arr, dummy_supply, 1.0, 1,
            dummy_b, dummy_b, dummy_l,
            2, 2, 1,
            0.85, 0.85, 0.50,
            4, 8, 100, 168, 48,
            dummy_h, 1, 0.35, 1000.0, 720)
        s1._batch_compute_storage_caps(demand_arr, dummy_supply, 1.0, 1, 4, 8, 100)
        print(f"  JIT ready")

    os.makedirs(STEP1D_OUTPUT_DIR, exist_ok=True)

    # ══════════════════════════════════════════════════════
    # PASS 0: Max-storage ceiling screen
    # ══════════════════════════════════════════════════════

    if precomputed_max_scores is not None:
        # Step1c already computed max-storage scores — skip Pass 0 entirely
        print(f"\n  Pass 0 — SKIPPED (using pre-computed max_storage_score from near-miss)")
        max_scores = precomputed_max_scores
    else:
        # Old near-miss file or coarse fallback — run Pass 0 as before
        print(f"\n  Pass 0 — Running max-storage screen (no pre-computed scores)")
        max_scores = run_max_screen(
            nm_combos, nm_base_scores, demand_arr, supply_matrix,
            active_thresholds, sc)

    # Eliminate mixes that can't reach ANY active threshold with max storage
    min_target = min(active_thresholds) / 100.0
    reachable = max_scores >= min_target
    # Also require base < max target (needs storage for at least one threshold)
    max_target = max(STORAGE_THRESHOLDS) / 100.0
    needs_storage = nm_base_scores < max_target
    keep = reachable & needs_storage
    n_before = len(nm_combos)
    nm_combos = nm_combos[keep]
    nm_base_scores = nm_base_scores[keep]
    max_scores = max_scores[keep]
    print(f"    Max-screen filter: {n_before:,} → {len(nm_combos):,} "
          f"({len(nm_combos)/max(n_before,1)*100:.1f}%)")

    # ══════════════════════════════════════════════════════
    # PASS 1: Adaptive coarse storage sweep (multi-threshold)
    # ══════════════════════════════════════════════════════
    #
    # PERF FIX: run_adaptive_coarse handles multiple thresholds in one pass —
    # the Numba kernel is called ONCE per batch and scores are reused for all
    # thresholds.  The old per-threshold loop re-evaluated overlapping mixes
    # for each threshold, causing ~4× redundant kernel evaluations on ERCOT
    # (6.4M near-miss mixes, heavily overlapping eligible sets across t85–t99.99).

    # Determine which thresholds need Pass 1 (coarse sweep)
    pass1_needed = [t for t in active_thresholds if t not in pass1_thresholds]
    pass1_cached = [t for t in active_thresholds if t in pass1_thresholds]

    coarse_results = {t: [] for t in active_thresholds}

    # ── Auto-batch state: tracks per-threshold batch file numbers ──
    # When a threshold accumulates > BATCH_FLUSH_SIZE results during Pass 1,
    # results are flushed to {ISO}_t{XX}_storage_b{N}.parquet and committed
    # immediately. This protects against CI timeout — partial progress is
    # banked even if the job dies. Step 2 reads all batch files natively.
    batch_nums = {}  # threshold → last batch number written
    # Parallel accumulator for Pass 2 boundary detection. Keeps ALL coarse
    # results in memory (flushed + unflushed) so run_fine_storage can find
    # boundary mixes. Memory is bounded (~48 bytes/result, 5M results ≈ 240MB).
    pass2_coarse = {t: [] for t in active_thresholds}

    def _auto_flush(results, thresholds):
        """Flush accumulated results to batch files when they exceed threshold.

        High thresholds (>=97.5%) use a smaller flush size and always auto-commit
        to protect against CI timeouts on expensive last-mile computations.
        """
        for t in thresholds:
            flush_size = HIGH_BATCH_FLUSH_SIZE if t >= HIGH_THRESHOLD_FLOOR else BATCH_FLUSH_SIZE
            if len(results[t]) >= flush_size:
                batch_nums[t] = batch_nums.get(t, 0) + 1
                bn = batch_nums[t]
                n_flush = len(results[t])
                # Keep for Pass 2 boundary detection
                pass2_coarse[t].extend(results[t])
                # Save batch file and commit.
                # High thresholds always auto-commit regardless of flag.
                save_storage_results(iso, t, nm_combos, results[t], rtypes,
                                     batch_num=bn)
                should_commit = auto_commit or t >= HIGH_THRESHOLD_FLOOR
                git_commit_threshold(iso, t, f"Pass1-b{bn}", should_commit,
                                     batch_num=bn)
                print(f"    Auto-flush {iso} t{t}%: {n_flush:,} results → "
                      f"b{bn}")
                # Clear accumulator for next batch
                results[t] = []

    if pass1_needed:
        # ── Build union eligible mask: mixes eligible for ANY pending threshold ──
        # A mix is eligible for threshold t if base < t/100 AND max >= t/100.
        # The union avoids re-running the Numba kernel on overlapping eligible sets.
        union_mask = np.zeros(len(nm_combos), dtype=bool)
        per_t_eligible = {}
        for t in pass1_needed:
            target = t / 100.0
            t_mask = (max_scores >= target) & (nm_base_scores < target)
            per_t_eligible[t] = int(t_mask.sum())
            union_mask |= t_mask

        n_union = int(union_mask.sum())
        # Skip thresholds with 0 eligible mixes (log them, keep in pass1_needed
        # so they get marked done in the manifest)
        real_needed = [t for t in pass1_needed if per_t_eligible[t] > 0]

        print(f"\n  Pass 1 — {len(pass1_needed)} thresholds, "
              f"{len(real_needed)} with eligible mixes")
        for t in pass1_needed:
            print(f"    t{t:>6.1f}%: {per_t_eligible[t]:>10,} eligible")
        print(f"    Union: {n_union:,} unique mixes "
              f"(vs {sum(per_t_eligible.values()):,} sum-of-per-threshold)")

        if n_union == 0:
            print(f"    No eligible mixes for any threshold — skipping all")
            for t in pass1_needed:
                coarse_results[t] = []
                git_commit_threshold(iso, t, "Pass1", auto_commit)
                pass1_thresholds.add(t)
            _save_manifest(iso, code_hash, pass1_thresholds, pass2_done)
        else:
            # Split thresholds by H2 requirement to avoid grid inflation.
            # Thresholds < 95% use a 150-combo grid (no H2);
            # thresholds >= 95% use 900 combos (with H2). Running them
            # together forces the 900-combo grid on sub-95% mixes — 6× waste.
            non_h2_thresholds = [t for t in real_needed if t < 95]
            h2_thresholds = [t for t in real_needed if t >= 95]
            groups = []
            if non_h2_thresholds:
                groups.append(('non-H2', non_h2_thresholds))
            if h2_thresholds:
                groups.append(('H2', h2_thresholds))

            for group_label, group_thresholds in groups:
                # Build union mask for this group's thresholds only
                group_mask = np.zeros(len(nm_combos), dtype=bool)
                for t in group_thresholds:
                    target = t / 100.0
                    t_mask = (max_scores >= target) & (nm_base_scores < target)
                    group_mask |= t_mask

                n_group = int(group_mask.sum())
                combos_est = 900 if group_label == 'H2' else 150
                print(f"\n    Group {group_label}: {len(group_thresholds)} thresholds, "
                      f"{n_group:,} union mixes, ~{combos_est} combos/mix")

                # Pass global_idx_map so results use full nm_combos indices
                # (no post-hoc remapping needed). Pass _auto_flush callback
                # so results get committed to batch files during processing.
                group_full_idx = np.where(group_mask)[0]
                group_coarse = run_adaptive_coarse(
                    iso, nm_combos[group_mask], nm_base_scores[group_mask],
                    max_scores[group_mask],
                    demand_arr, supply_matrix, group_thresholds, sc,
                    global_idx_map=group_full_idx,
                    on_bucket_done=_auto_flush)

                # Results already use full indices — no remapping needed.
                # Flush remaining unflushed results for each threshold.
                for t in group_thresholds:
                    remaining = group_coarse.get(t, [])
                    if remaining:
                        pass2_coarse[t].extend(remaining)
                    # High thresholds always auto-commit
                    t_commit = auto_commit or t >= HIGH_THRESHOLD_FLOOR
                    if t in batch_nums:
                        # Threshold was auto-batched — save remainder as
                        # next batch file (even if small)
                        if remaining:
                            batch_nums[t] += 1
                            bn = batch_nums[t]
                            save_storage_results(iso, t, nm_combos, remaining,
                                                 rtypes, batch_num=bn)
                            git_commit_threshold(iso, t, f"Pass1-b{bn}",
                                                 t_commit, batch_num=bn)
                            print(f"    Final flush {iso} t{t}%: "
                                  f"{len(remaining):,} results → b{bn}")
                        n_total = len(pass2_coarse[t])
                        print(f"    {iso} t{t}%: {n_total:,} storage-feasible "
                              f"(coarse, {batch_nums[t]} batches)")
                    else:
                        # Small threshold — save as monolithic file
                        coarse_results[t] = remaining
                        if remaining:
                            save_storage_results(iso, t, nm_combos, remaining,
                                                 rtypes)
                            print(f"    {iso} t{t}%: {len(remaining):,} "
                                  f"storage-feasible (coarse)")
                        git_commit_threshold(iso, t, "Pass1", t_commit)

            # Mark all thresholds as Pass 1 done (including 0-eligible ones)
            for t in pass1_needed:
                pass1_thresholds.add(t)
            _save_manifest(iso, code_hash, pass1_thresholds, pass2_done)
    else:
        print(f"\n  Pass 1: all active thresholds already done")

    # Reload coarse results from saved parquets for cached thresholds.
    # Supports both monolithic (_storage.parquet) and batch (_storage_b{N})
    # files from previous runs.
    if pass1_cached:
        import glob as globmod
        # Vectorized key→index lookup via sorted search
        nm_keys_arr = _mix_keys(nm_combos)
        sort_order = np.argsort(nm_keys_arr)
        sorted_keys = nm_keys_arr[sort_order]

        for t in pass1_cached:
            t_str = s1._normalize_threshold_str(t)
            # Discover all files for this threshold (monolithic + batches)
            mono_path = os.path.join(STEP1D_OUTPUT_DIR,
                                     f'{iso}_t{t_str}_storage.parquet')
            batch_pattern = os.path.join(STEP1D_OUTPUT_DIR,
                                         f'{iso}_t{t_str}_storage_b*.parquet')
            batch_files = sorted(globmod.glob(batch_pattern))

            t_files = []
            if batch_files:
                t_files = batch_files  # prefer batches if they exist
            elif os.path.exists(mono_path):
                t_files = [mono_path]

            for t_path in t_files:
                table = pq.read_table(t_path)
                mix_mat = np.column_stack(
                    [table.column(rt).to_numpy() for rt in rtypes])
                row_keys = _mix_keys(mix_mat)
                bat4 = table.column('battery_dispatch_pct').to_numpy()
                bat8 = table.column('battery8_dispatch_pct').to_numpy()
                ldes = table.column('ldes_dispatch_pct').to_numpy()
                h2_col = table.column('h2_dispatch_pct').to_numpy()
                score = table.column('hourly_match_score').to_numpy()

                # Vectorized key lookup via searchsorted
                insert_pos = np.searchsorted(sorted_keys, row_keys)
                insert_pos = np.clip(insert_pos, 0, len(sorted_keys) - 1)
                matched = sorted_keys[insert_pos] == row_keys
                mis = sort_order[insert_pos]

                # Bulk collect matched rows
                valid_idx = np.where(matched)[0]
                if len(valid_idx) > 0:
                    coarse_results[t].extend(
                        [(int(mis[i]), float(bat4[i]), float(bat8[i]),
                          float(ldes[i]), float(h2_col[i]), float(score[i]))
                         for i in valid_idx])
            # Also populate pass2_coarse for cached thresholds
            pass2_coarse[t] = coarse_results[t]

    # ══════════════════════════════════════════════════════
    # PASS 2: Fine targeted storage (per-threshold boundary)
    # ══════════════════════════════════════════════════════

    print(f"\n  Pass 2 — Fine storage refinement (0.05% resolution)")

    for t in active_thresholds:
        if t in pass2_done:
            print(f"    {iso} t{t}%: skipped (already done)")
            continue

        t_start = time.time()
        # Use pass2_coarse which has ALL coarse results (flushed + unflushed)
        coarse_for_t = pass2_coarse.get(t, coarse_results.get(t, []))
        fine_results = run_fine_storage(
            iso, t, nm_combos, coarse_for_t,
            demand_arr, supply_matrix)

        t_elapsed = time.time() - t_start
        n_coarse = len(coarse_for_t)
        n_fine = len(fine_results)

        # High thresholds always auto-commit
        t_commit = auto_commit or t >= HIGH_THRESHOLD_FLOOR

        if t in batch_nums:
            # Batched threshold — save fine results as next batch file.
            # Step 2 loads all batch files and deduplicates natively.
            if fine_results:
                batch_nums[t] += 1
                bn = batch_nums[t]
                save_storage_results(iso, t, nm_combos, fine_results, rtypes,
                                     batch_num=bn)
                git_commit_threshold(iso, t, f"Pass2-b{bn}", t_commit,
                                     batch_num=bn)
            print(f"    {iso} t{t}%: {n_fine:,} fine + {n_coarse:,} coarse "
                  f"({batch_nums[t]} batch files, {t_elapsed:.1f}s)")
        else:
            # Non-batched — merge coarse + fine, save as monolithic file
            all_results = coarse_for_t + fine_results

            # Deduplicate: keep best score per unique (mix, storage) tuple
            seen = {}
            for r in all_results:
                key = (r[0], round(r[1], 4), round(r[2], 4),
                       round(r[3], 4), round(r[4], 4))
                if key not in seen or r[5] > seen[key][5]:
                    seen[key] = r
            deduped = list(seen.values())

            if deduped:
                save_storage_results(iso, t, nm_combos, deduped, rtypes)

            n_total = len(deduped)
            print(f"    {iso} t{t}%: {n_fine:,} fine + {n_coarse:,} coarse "
                  f"= {n_total:,} total ({t_elapsed:.1f}s)")
            git_commit_threshold(iso, t, "Pass2", t_commit)

        pass2_done.add(t)
        _save_manifest(iso, code_hash, pass1_thresholds, pass2_done)

    iso_elapsed = time.time() - iso_start
    print(f"\n{'=' * 70}")
    print(f"  {iso} COMPLETE — {iso_elapsed:.1f}s total")
    print(f"{'=' * 70}")


def _parse_thresholds(raw):
    """Parse comma-separated threshold list, return sorted list of floats or None."""
    if not raw or raw.strip().upper() in ('', 'ALL'):
        return None
    parts = [p.strip() for p in raw.split(',') if p.strip()]
    result = []
    for p in parts:
        try:
            result.append(float(p))
        except ValueError:
            print(f"WARNING: Ignoring invalid threshold '{p}'")
    valid = set(STORAGE_THRESHOLDS)
    filtered = [t for t in sorted(set(result)) if t in valid]
    bad = [t for t in result if t not in valid]
    if bad:
        print(f"WARNING: Thresholds not in STORAGE_THRESHOLDS: {bad}")
    return filtered if filtered else None


# ══════════════════════════════════════════════════════════════════════════════
# BATCH INFO — Estimates batch counts for each ISO/threshold pair
# ══════════════════════════════════════════════════════════════════════════════

# Throughput estimates (mixes per minute) for CI runners (2 vCPU, Numba JIT).
# H2 thresholds (>=95%) use 900-combo grids; non-H2 use 150-combo grids.
# These are conservative estimates calibrated from actual CI runs.
THROUGHPUT_H2 = 10_000        # mixes/min with 900-combo H2 grid
THROUGHPUT_NON_H2 = 30_000    # mixes/min with 150-combo non-H2 grid


def print_batch_info(iso, target_minutes=10):
    """Print batch count estimates for an ISO.

    Loads near-miss data, runs Pass 0 max-screen, and for each threshold
    reports eligible mix count and recommended batch count.
    """
    print(f"\n{'=' * 70}")
    print(f"  Step 1d Batch Info — {iso}")
    print(f"  Target: ~{target_minutes} min per batch")
    print(f"{'=' * 70}")

    rtypes = s1.get_resource_types(iso)

    # Load mixes (with coarse fallback)
    nm_combos, nm_scores, _ = load_mixes_with_coarse_fallback(
        iso, STORAGE_THRESHOLDS)
    if nm_combos is None:
        print(f"  ERROR: No near-miss data for {iso}. Run step1c first.")
        return

    print(f"  Near-miss mixes: {len(nm_combos):,}")
    print(f"  Score range: {nm_scores.min()*100:.2f}% — {nm_scores.max()*100:.2f}%")

    # Load EIA data for Pass 0
    print(f"  Loading EIA data for Pass 0...")
    demand_data, gen_profiles, _, _ = s1.load_data()
    demand_norm = demand_data[iso]['normalized']
    supply_profiles = s1.get_supply_profiles(iso, gen_profiles)
    demand_arr, supply_matrix = s1.prepare_numpy_profiles(
        iso, demand_norm, supply_profiles)

    sc = _get_storage_constants()

    # JIT warmup
    if s1.HAS_NUMBA:
        print(f"  Warming up Numba JIT...")
        dummy_supply = np.ones((1, s1.H), dtype=np.float64)
        dummy_b = np.array([0.0, 1.0], dtype=np.float64)
        dummy_l = np.array([0.0], dtype=np.float64)
        dummy_h = np.array([0.0], dtype=np.float64)
        s1._batch_mixes_storage_screen(
            demand_arr, dummy_supply, 1.0, 1,
            dummy_b, dummy_b, dummy_l, 2, 2, 1,
            0.85, 0.85, 0.50, 4, 8, 100, 168, 48,
            dummy_h, 1, 0.35, 1000.0, 720)

    # Run Pass 0 max-screen
    max_scores = run_max_screen(
        nm_combos, nm_scores, demand_arr, supply_matrix,
        STORAGE_THRESHOLDS, sc)

    # Print table header
    print(f"\n  {'Threshold':>10} | {'Eligible':>10} | {'Group':>6} | "
          f"{'Est. min':>8} | {'Batches':>7} | {'Per batch':>10}")
    print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*6}-+-{'-'*8}-+-{'-'*7}-+-{'-'*10}")

    total_batches_needed = 0
    batch_table = []  # for JSON output

    for t in STORAGE_THRESHOLDS:
        target = t / 100.0
        eligible = (max_scores >= target) & (nm_scores < target)
        n_eligible = int(eligible.sum())

        if n_eligible == 0:
            print(f"  {t:>9.1f}% | {n_eligible:>10,} | {'—':>6} | "
                  f"{'—':>8} | {'0':>7} | {'—':>10}")
            batch_table.append({
                'threshold': t, 'eligible': 0, 'group': 'none',
                'est_minutes': 0, 'batches': 0, 'per_batch': 0})
            continue

        group = 'H2' if t >= 95 else 'non-H2'
        throughput = THROUGHPUT_H2 if t >= 95 else THROUGHPUT_NON_H2
        est_minutes = n_eligible / throughput
        n_batches = max(1, int(np.ceil(est_minutes / target_minutes)))
        per_batch = int(np.ceil(n_eligible / n_batches))

        total_batches_needed += n_batches
        print(f"  {t:>9.1f}% | {n_eligible:>10,} | {group:>6} | "
              f"{est_minutes:>7.1f}m | {n_batches:>7} | {per_batch:>10,}")

        batch_table.append({
            'threshold': t, 'eligible': n_eligible, 'group': group,
            'est_minutes': round(est_minutes, 1),
            'batches': n_batches, 'per_batch': per_batch})

    print(f"\n  Total batches needed: {total_batches_needed}")
    print(f"  Total estimated compute: "
          f"{sum(b['est_minutes'] for b in batch_table):.0f} min")

    # Save batch info as JSON for workflow reference
    info_path = os.path.join(STEP1D_OUTPUT_DIR, f'{iso}_batch_info.json')
    os.makedirs(STEP1D_OUTPUT_DIR, exist_ok=True)
    with open(info_path, 'w') as f:
        json.dump({
            'iso': iso,
            'target_minutes': target_minutes,
            'total_mixes': len(nm_combos),
            'thresholds': batch_table
        }, f, indent=2)
    print(f"\n  Batch info saved: {info_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Step 1C: Three-pass storage sweep with auto-batching.")
    parser.add_argument("--iso", required=True,
                        help="ISO name or 'ALL'")
    parser.add_argument("--auto-commit", action="store_true",
                        help="Commit & push after each threshold (and each "
                             "auto-batch for large thresholds)")
    parser.add_argument("--thresholds", default="",
                        help="Comma-separated thresholds to process "
                             "(e.g. '90,95,99'). Default: all 17.")
    parser.add_argument("--batch-info", action="store_true",
                        help="Print eligible mix counts and runtime estimates "
                             "per threshold (no processing).")
    parser.add_argument("--target-minutes", type=float, default=10,
                        help="Target minutes per auto-batch for --batch-info "
                             "(default: 10).")
    args = parser.parse_args()

    isos = list(s1.ISOS) if args.iso.upper() == 'ALL' else [args.iso.upper()]

    for iso in isos:
        if iso not in s1.ISOS:
            print(f"ERROR: Unknown ISO '{iso}'")
            sys.exit(1)

    # Floor/fine augmentation is always on — augment near-miss pool with
    # floor_pfs (step1b2) and fine_pfs (step1b3) mixes automatically.
    global load_mixes_with_coarse_fallback
    load_mixes_with_coarse_fallback = _augment_with_floor_fine(
        load_mixes_with_coarse_fallback)
    print(f"  Floor/fine mix augmentation enabled (always on)")

    # Batch info mode
    if args.batch_info:
        for iso in isos:
            print_batch_info(iso, target_minutes=args.target_minutes)
        return

    # Standard processing (auto-batching happens internally for large thresholds)
    thresholds_filter = _parse_thresholds(args.thresholds)
    if thresholds_filter:
        print(f"Threshold filter: {thresholds_filter}")

    for iso in isos:
        process_iso(iso, auto_commit=args.auto_commit,
                    thresholds_filter=thresholds_filter)


if __name__ == "__main__":
    main()
