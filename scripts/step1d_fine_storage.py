#!/usr/bin/env python3
"""Step 1d: Three-pass adaptive storage sweep.

Reads the union near-miss mixes from step1c and runs storage dispatch using
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

Output: data/step1d-storage-parquets/{ISO}_t{XX}_storage.parquet

Usage:
  python scripts/step1d_fine_storage.py --iso CAISO
  python scripts/step1d_fine_storage.py --iso PJM --auto-commit
  python scripts/step1d_fine_storage.py --iso ALL
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
STEP1D_OUTPUT_DIR = os.path.join(DATA_DIR, 'step1d-storage-parquets')

# Thresholds for storage refinement
STORAGE_THRESHOLDS = [50, 55, 60, 65, 70, 75, 80, 85, 87.5,
                      90, 92.5, 95, 97.5, 99, 99.5, 99.9, 99.99]

# Pass 0: Maximum storage levels (ceiling screen)
MAX_BAT4 = np.array([6.0], dtype=np.float64)
MAX_BAT8 = np.array([8.0], dtype=np.float64)
MAX_LDES = np.array([25.0], dtype=np.float64)
MAX_H2 = np.array([25.0], dtype=np.float64)
NO_H2 = np.array([0.0], dtype=np.float64)

# Pass 1: Full coarse grids (used to build per-bucket adaptive grids)
FULL_BAT4 = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.float64)
FULL_BAT8 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
FULL_LDES = np.array([0, 2, 5, 8, 10, 15, 20, 25], dtype=np.float64)
FULL_H2 = np.array([0, 5, 10, 15, 20, 25], dtype=np.float64)

# Gap bucket boundaries (in percentage points).
# Mixes are grouped by how far their base score is from the threshold.
# Each bucket gets a grid capped at ~3× its max gap (storage efficiency headroom).
GAP_BUCKET_PP = [5, 10, 20, 50]  # bucket edges: 0-5pp, 5-10pp, 10-20pp, 20-50pp

# Pass 2: Fine storage resolution
FINE_STEP = 0.05         # 0.05 percentage points
FINE_HALF_WIDTH = 0.5    # ±0.5pp around coarse winner
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

# Storage sweep floor
STORAGE_SWEEP_FLOOR = 0.50

# Progress save interval
PROGRESS_INTERVAL = 25   # save every N batches


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
    """Load union near-miss mixes from step1c output."""
    path = os.path.join(s1.STEP1_RAW_PFS_PARQUET_DIR,
                        f'{iso}_near_miss.parquet')
    if not os.path.exists(path):
        return None, None

    table = pq.read_table(path)
    rtypes = s1.get_resource_types(iso)
    combos = np.column_stack([table.column(rt).to_numpy() for rt in rtypes])
    scores = table.column('base_score').to_numpy()
    return combos, scores


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
    Storage levels are capped at ~3× the bucket's max gap (efficiency headroom).
    """
    cap = max_gap_pp * 3.0  # storage capacity headroom

    bat4 = FULL_BAT4[FULL_BAT4 <= max(cap, 1)]
    if len(bat4) < 2:
        bat4 = FULL_BAT4[:2]  # at least [0, 1]

    bat8 = FULL_BAT8[FULL_BAT8 <= max(cap, 1)]
    if len(bat8) < 2:
        bat8 = FULL_BAT8[:2]

    ldes = FULL_LDES[FULL_LDES <= max(cap, 2)]
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
                        demand_arr, supply_matrix, active_thresholds, sc):
    """Pass 1: Adaptive coarse storage sweep with gap-bucketed grids.

    Groups mixes by their gap to the closest active threshold, builds
    a right-sized storage grid per bucket, and runs the Numba kernel
    on each bucket. This avoids testing LDES=25% on mixes that only
    need 3pp of storage boost.

    Returns per-threshold feasible lists:
        results[threshold] = list of (mix_idx, bat4, bat8, ldes, h2, score)
    """
    n_mixes = len(nm_combos)
    batch_size = CAISO_MIX_BATCH if iso == 'CAISO' else MAX_MIX_BATCH
    include_h2 = any(t >= 95 for t in active_thresholds)

    # Compute storage caps for all mixes (needed for per-mix cap filtering)
    print(f"\n  Pass 1 — Adaptive coarse sweep")
    print(f"    Computing storage caps...")
    b4_caps = np.empty(n_mixes, dtype=np.float64)
    b8_caps = np.empty(n_mixes, dtype=np.float64)
    l_caps = np.empty(n_mixes, dtype=np.float64)
    surplus_days = np.empty(n_mixes, dtype=np.int64)
    hc_arr = np.empty(n_mixes, dtype=np.int64)
    surplus_pct = np.empty(n_mixes, dtype=np.float64)
    demand_total = demand_arr.sum()

    for cs in range(0, n_mixes, NM_CHUNK):
        ce = min(cs + NM_CHUNK, n_mixes)
        chunk_fracs = nm_combos[cs:ce].astype(np.float64) / 100.0
        chunk_supply = chunk_fracs @ supply_matrix
        chunk_n = ce - cs

        cb4, cb8, cl, chc, csd = s1._batch_compute_storage_caps(
            demand_arr, chunk_supply, 1.0, chunk_n,
            sc['batt4_dur'], sc['batt8_dur'], sc['ldes_dur'])
        b4_caps[cs:ce] = cb4
        b8_caps[cs:ce] = cb8
        l_caps[cs:ce] = cl
        hc_arr[cs:ce] = chc
        surplus_days[cs:ce] = csd

        chunk_surplus = np.maximum(chunk_supply - demand_arr[np.newaxis, :], 0.0)
        surplus_pct[cs:ce] = chunk_surplus.sum(axis=1) / demand_total

    has_curtailment = hc_arr.astype(bool)

    # Pre-compute per-threshold floors
    threshold_floors = {
        t: max(STORAGE_SWEEP_FLOOR, t / 100.0 - get_near_miss_width(t))
        for t in active_thresholds
    }

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

        # Process in batches
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

            # Extract feasible results and bin to thresholds
            for local_i in range(n_batch):
                mi = int(b_idx[local_i])
                b4_max = b4_caps[mi] * 100.0 * 1.1
                b8_max = b8_caps[mi] * 100.0 * 1.1
                l_max = l_caps[mi] * 100.0 * 1.1
                n_sd = int(surplus_days[mi])
                mix_base = nm_base_scores[mi]
                mix_surplus = surplus_pct[mi]

                scores = batch_scores[local_i]

                for b4i in range(n_b4):
                    bp = b4_grid[b4i]
                    if bp > 0 and (n_sd < MIN_SURPLUS_DAYS_FOR_BATTERY
                                   or bp > b4_max):
                        continue
                    for b8i in range(n_b8):
                        b8p = b8_grid[b8i]
                        if b8p > 0 and (n_sd < MIN_SURPLUS_DAYS_FOR_BATTERY
                                        or b8p > b8_max):
                            continue
                        for li in range(n_l):
                            lp = ldes_grid[li]
                            if lp > 0 and lp > l_max:
                                continue
                            for h2i in range(n_h2):
                                h2p = h2_grid[h2i]
                                if bp == 0 and b8p == 0 and lp == 0 and h2p == 0:
                                    continue

                                idx = (b4i * n_b8 * n_l * n_h2 +
                                       b8i * n_l * n_h2 +
                                       li * n_h2 + h2i)
                                score = scores[idx]
                                if score < 0:
                                    continue

                                for t in active_thresholds:
                                    target = t / 100.0
                                    if score < target:
                                        continue
                                    if mix_base >= target:
                                        continue
                                    gap = target - mix_base
                                    if mix_surplus < gap:
                                        continue
                                    if (mix_base < threshold_floors[t]
                                            and mix_surplus < 1.5 * gap):
                                        continue
                                    if h2p > 0 and t < 95:
                                        continue
                                    results[t].append(
                                        (mi, float(bp), float(b8p),
                                         float(lp), float(h2p),
                                         round(score * 100, 2)))
                                    total_feasible += 1

    print(f"    Pass 1 complete: {total_feasible:,} feasible, "
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


def run_fine_storage(iso, threshold, nm_combos, coarse_results,
                     demand_arr, supply_matrix):
    """Fine 0.05% storage refinement for one threshold's boundary mixes.

    Identifies mixes near the threshold boundary from coarse results,
    refines storage around the coarse winner at 0.05% resolution.

    Returns list of (mix_idx, bat4, bat8, ldes, h2, score).
    """
    target = threshold / 100.0
    batch_size = CAISO_MIX_BATCH if iso == 'CAISO' else MAX_MIX_BATCH

    # Storage constants
    batt_eff = s1.BATTERY_EFFICIENCY
    batt8_eff = s1.BATTERY8_EFFICIENCY
    ldes_eff = s1.LDES_EFFICIENCY
    batt4_dur = s1.BATTERY_DURATION_HOURS
    batt8_dur = s1.BATTERY8_DURATION_HOURS
    ldes_dur = s1.LDES_DURATION_HOURS
    ldes_window = s1.LDES_WINDOW_DAYS * 24
    batt8_window = 48
    h2_eff = s1.H2_EFFICIENCY
    h2_dur = float(s1.H2_DURATION_HOURS)
    h2_window = s1.H2_WINDOW_DAYS * 24

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

    fine_results = []
    n_boundary = len(boundary_mixes)
    processed = 0

    # Process boundary mixes in batches
    boundary_items = list(boundary_mixes.items())

    for batch_start in range(0, len(boundary_items), batch_size):
        batch_end = min(batch_start + batch_size, len(boundary_items))
        batch = boundary_items[batch_start:batch_end]

        for mi, (best_b4, best_b8, best_l, best_h2, best_score) in batch:
            # Generate fine ranges around the coarse winner
            fine_b4 = _fine_range(best_b4, FINE_STEP, FINE_HALF_WIDTH, 0, 6)
            fine_b8 = _fine_range(best_b8, FINE_STEP, FINE_HALF_WIDTH, 0, 8)
            fine_l = _fine_range(best_l, FINE_STEP, FINE_HALF_WIDTH, 0, 25)

            if threshold >= 95 and best_h2 > 0:
                fine_h2 = _fine_range(best_h2, FINE_STEP * 10, FINE_HALF_WIDTH * 5, 0, 25)
            else:
                fine_h2 = np.array([0.0], dtype=np.float64)

            # Check if cross-product exceeds limit
            n_combos = len(fine_b4) * len(fine_b8) * len(fine_l) * len(fine_h2)
            if n_combos > MAX_FINE_COMBOS:
                # Importance sampling: fix least-important dims at winner,
                # sweep the 2 most-important (most variation in coarse results)
                # Default: sweep b4 and ldes (usually highest impact)
                fine_b8 = np.array([best_b8], dtype=np.float64)
                fine_h2 = np.array([best_h2 if best_h2 > 0 else 0.0],
                                   dtype=np.float64)
                n_combos = len(fine_b4) * len(fine_b8) * len(fine_l) * len(fine_h2)

            # Build supply for this mix
            mix_fracs = nm_combos[mi].astype(np.float64) / 100.0
            mix_supply = (mix_fracs @ supply_matrix).reshape(1, -1)

            # Run Numba storage screen on single mix
            mix_scores = s1._batch_mixes_storage_screen(
                demand_arr, mix_supply, 1.0, 1,
                fine_b4, fine_b8, fine_l,
                len(fine_b4), len(fine_b8), len(fine_l),
                batt_eff, batt8_eff, ldes_eff,
                batt4_dur, batt8_dur, ldes_dur,
                ldes_window, batt8_window,
                fine_h2, len(fine_h2), h2_eff, h2_dur, h2_window)

            scores_flat = mix_scores[0]  # (n_combos,)

            # Extract feasible results
            n_b4, n_b8, n_l, n_h2 = (len(fine_b4), len(fine_b8),
                                      len(fine_l), len(fine_h2))
            for b4i in range(n_b4):
                for b8i in range(n_b8):
                    for li in range(n_l):
                        for h2i in range(n_h2):
                            idx = (b4i * n_b8 * n_l * n_h2 +
                                   b8i * n_l * n_h2 +
                                   li * n_h2 + h2i)
                            score = scores_flat[idx]
                            if score >= 0 and score >= target:
                                fine_results.append(
                                    (int(mi), float(fine_b4[b4i]),
                                     float(fine_b8[b8i]),
                                     float(fine_l[li]),
                                     float(fine_h2[h2i]),
                                     round(score * 100, 2)))

            processed += 1

    return fine_results


# ══════════════════════════════════════════════════════════════════════════════
# SAVE / COMMIT
# ══════════════════════════════════════════════════════════════════════════════

def save_storage_results(iso, threshold, nm_combos, results, rtypes):
    """Save storage-enhanced feasible mixes for one threshold."""
    if not results:
        return

    os.makedirs(STEP1D_OUTPUT_DIR, exist_ok=True)

    n = len(results)
    data = {
        'iso': [iso] * n,
        'threshold': [float(threshold)] * n,
    }
    for j, rt in enumerate(rtypes):
        data[rt] = np.array([nm_combos[r[0]][j] for r in results],
                            dtype=np.float64)
    data['battery_dispatch_pct'] = np.array([r[1] for r in results],
                                            dtype=np.float64)
    data['battery8_dispatch_pct'] = np.array([r[2] for r in results],
                                             dtype=np.float64)
    data['ldes_dispatch_pct'] = np.array([r[3] for r in results],
                                         dtype=np.float64)
    data['h2_dispatch_pct'] = np.array([r[4] for r in results],
                                       dtype=np.float64)
    data['hourly_match_score'] = np.array([r[5] for r in results],
                                          dtype=np.float64)

    table = pa.table(data)
    t_str = s1._normalize_threshold_str(threshold)
    out_path = os.path.join(STEP1D_OUTPUT_DIR,
                            f'{iso}_t{t_str}_storage.parquet')
    pq.write_table(table, out_path, compression='snappy')
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    return out_path


def git_commit_threshold(iso, threshold, phase_label, auto_commit):
    """Commit and push threshold results."""
    if not auto_commit:
        return

    try:
        t_str = s1._normalize_threshold_str(threshold)
        out_path = os.path.join(STEP1D_OUTPUT_DIR,
                                f'{iso}_t{t_str}_storage.parquet')
        if not os.path.exists(out_path):
            return

        subprocess.run(['git', 'add', '-f', out_path],
                       check=True, capture_output=True, text=True)

        result = subprocess.run(['git', 'diff', '--cached', '--quiet'],
                                capture_output=True)
        if result.returncode == 0:
            return

        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        msg = (f"Storage {phase_label}: {iso} {threshold}% "
               f"({size_mb:.1f} MB) — auto-commit")
        subprocess.run(['git', 'commit', '-m', msg],
                       check=True, capture_output=True, text=True)

        for attempt in range(1, 5):
            result = subprocess.run(
                ['git', 'push', '-u', 'origin', 'HEAD'],
                capture_output=True, text=True)
            if result.returncode == 0:
                print(f"    [auto-commit] {iso} {threshold}% committed & pushed")
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
    for fname in ['step1d_fine_storage.py', 'step1_pfs_generator.py']:
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


def _save_manifest(iso, code_hash, pass1_done, pass2_done):
    os.makedirs(STEP1D_OUTPUT_DIR, exist_ok=True)
    manifest = {
        'code_hash': code_hash,
        'pass1_done': pass1_done,
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
    pass1_done = manifest.get('pass1_done', False) if manifest else False
    pass2_done = set(manifest.get('pass2_done', [])) if manifest else set()

    # ── Load near-miss mixes ──
    nm_combos, nm_base_scores = load_near_miss(iso)
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

    max_scores = run_max_screen(
        nm_combos, nm_base_scores, demand_arr, supply_matrix,
        active_thresholds, sc)

    # Eliminate mixes that can't reach ANY active threshold with max storage
    min_target = min(active_thresholds) / 100.0
    reachable = max_scores >= min_target
    # Also require base < max target (needs storage for at least one threshold)
    max_target = max(active_thresholds) / 100.0
    needs_storage = nm_base_scores < max_target
    keep = reachable & needs_storage
    n_before = len(nm_combos)
    nm_combos = nm_combos[keep]
    nm_base_scores = nm_base_scores[keep]
    max_scores = max_scores[keep]
    print(f"    Max-screen filter: {n_before:,} → {len(nm_combos):,} "
          f"({len(nm_combos)/max(n_before,1)*100:.1f}%)")

    # ══════════════════════════════════════════════════════
    # PASS 1: Adaptive coarse storage sweep
    # ══════════════════════════════════════════════════════

    coarse_results = None
    if not pass1_done:
        coarse_results = run_adaptive_coarse(
            iso, nm_combos, nm_base_scores, max_scores,
            demand_arr, supply_matrix, active_thresholds, sc)

        # Save coarse results per threshold (filtered to active set)
        for t in active_thresholds:
            t_results = coarse_results[t]
            if t_results:
                save_storage_results(iso, t, nm_combos, t_results, rtypes)
                print(f"    {iso} t{t}%: {len(t_results):,} storage-feasible "
                      f"(coarse)")
            git_commit_threshold(iso, t, "Pass1", auto_commit)

        pass1_done = True
        _save_manifest(iso, code_hash, pass1_done, pass2_done)
    else:
        print(f"\n  Pass 1: skipped (already done)")
        # Reload coarse results from saved parquets for Pass 2.
        # Use hash-map lookup (O(1) per row) instead of O(N) linear search.
        nm_key_to_idx = {k: i for i, k in
                         enumerate(_mix_keys(nm_combos).tolist())}

        coarse_results = {t: [] for t in active_thresholds}
        for t in active_thresholds:
            t_str = s1._normalize_threshold_str(t)
            t_path = os.path.join(STEP1D_OUTPUT_DIR,
                                  f'{iso}_t{t_str}_storage.parquet')
            if not os.path.exists(t_path):
                continue
            table = pq.read_table(t_path)
            # Bulk-extract resource columns as numpy arrays for vectorized hashing
            mix_mat = np.column_stack(
                [table.column(rt).to_numpy() for rt in rtypes])
            row_keys = _mix_keys(mix_mat).tolist()
            bat4 = table.column('battery_dispatch_pct').to_numpy()
            bat8 = table.column('battery8_dispatch_pct').to_numpy()
            ldes = table.column('ldes_dispatch_pct').to_numpy()
            h2 = table.column('h2_dispatch_pct').to_numpy()
            score = table.column('hourly_match_score').to_numpy()
            for i, key in enumerate(row_keys):
                mi = nm_key_to_idx.get(key, -1)
                if mi == -1:
                    continue  # mix not in near-miss set (shouldn't happen)
                coarse_results[t].append(
                    (mi, float(bat4[i]), float(bat8[i]),
                     float(ldes[i]), float(h2[i]), float(score[i])))

    # ══════════════════════════════════════════════════════
    # PASS 2: Fine targeted storage (per-threshold boundary)
    # ══════════════════════════════════════════════════════

    print(f"\n  Pass 2 — Fine storage refinement (0.05% resolution)")

    for t in active_thresholds:
        if t in pass2_done:
            print(f"    {iso} t{t}%: skipped (already done)")
            continue

        t_start = time.time()
        fine_results = run_fine_storage(
            iso, t, nm_combos, coarse_results[t],
            demand_arr, supply_matrix)

        # Merge coarse + fine results (fine supersedes coarse for boundary mixes)
        all_results = coarse_results[t] + fine_results

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

        t_elapsed = time.time() - t_start
        n_fine = len(fine_results)
        n_total = len(deduped)
        print(f"    {iso} t{t}%: {n_fine:,} fine + {len(coarse_results[t]):,} "
              f"coarse = {n_total:,} total ({t_elapsed:.1f}s)")

        git_commit_threshold(iso, t, "Pass2", auto_commit)
        pass2_done.add(t)
        _save_manifest(iso, code_hash, pass1_done, pass2_done)

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


def main():
    parser = argparse.ArgumentParser(
        description="Step 1d: Two-pass storage sweep.")
    parser.add_argument("--iso", required=True,
                        help="ISO name or 'ALL'")
    parser.add_argument("--auto-commit", action="store_true",
                        help="Commit & push after each threshold")
    parser.add_argument("--thresholds", default="",
                        help="Comma-separated thresholds to process "
                             "(e.g. '90,95,99'). Default: all 17.")
    args = parser.parse_args()

    isos = list(s1.ISOS) if args.iso.upper() == 'ALL' else [args.iso.upper()]

    for iso in isos:
        if iso not in s1.ISOS:
            print(f"ERROR: Unknown ISO '{iso}'")
            sys.exit(1)

    thresholds_filter = _parse_thresholds(args.thresholds)
    if thresholds_filter:
        print(f"Threshold filter: {thresholds_filter}")

    for iso in isos:
        process_iso(iso, auto_commit=args.auto_commit,
                    thresholds_filter=thresholds_filter)


if __name__ == "__main__":
    main()
