#!/usr/bin/env python3
"""
Step 1D: Storage Refinement Module
===================================
Fills storage exploration gaps in Step 1C PFS results by testing intermediate
storage levels that the coarse grid missed.

Problem: Step 1C's coarse storage levels at <95% thresholds ([0,1,3] for bat4,
[0,2,4] for bat8, [0,5,10] for LDES) are too wide — the first non-zero level
exceeds the physical storage cap for most mixes (typical caps: bat4=0.2-0.5%,
bat8=0.5-1.0%, LDES=1.0-3.0%), so storage is never actually explored.

Solution: Test fine-grained intermediate levels that fill the 0-to-cap range:
  65-92.5%:  Full intermediate sweep (bat4/bat8/LDES fine levels)
  >=95%:     LDES intermediates only (bat4/bat8/H2 same as 1C)

Uses Step 1C coarse cache to identify candidate mixes (no rerun of 1A-1C).
Outputs to separate parquets in data/step1d-storage-parquets/.

Pipeline position: Step 1D of 7 (runs after Step 1C, before Step 2)
"""

import os
import sys
import time
import numpy as np

# Numba JIT
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def wrapper(fn):
            return fn
        if args and callable(args[0]):
            return args[0]
        return wrapper
    prange = range

# PyArrow
import pyarrow as pa
import pyarrow.parquet as pq

# Add scripts dir to path for step1 imports
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

from step1_pfs_generator import (
    load_data, get_supply_profiles, prepare_numpy_profiles,
    get_resource_types, ISOS,
    BATTERY_EFFICIENCY, BATTERY_DURATION_HOURS,
    BATTERY8_EFFICIENCY, BATTERY8_DURATION_HOURS,
    LDES_EFFICIENCY, LDES_DURATION_HOURS, LDES_WINDOW_DAYS,
    H2_EFFICIENCY, H2_DURATION_HOURS, H2_WINDOW_DAYS,
    _batch_compute_storage_caps, _batch_mixes_storage_screen,
    _normalize_threshold_str, H, STEP1_RAW_PFS_PARQUET_DIR,
)

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

DATA_DIR = os.path.join(os.path.dirname(SCRIPTS_DIR), 'data')
STEP1D_OUTPUT_DIR = os.path.join(DATA_DIR, 'step1d-storage-parquets')

# Thresholds covered by Step 1D (65% and above)
STEP1D_THRESHOLDS = [65, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 99.5, 99.9, 99.99]

# Near-miss floor: don't test storage on mixes below 50% — they lack
# meaningful surplus to shift.
STORAGE_SWEEP_FLOOR = 0.50

# Near-miss half-width in 0-1 score space (same as Step 1C: 40pp)
NEAR_MISS_WIDTH = 0.40

# Minimum surplus days for battery deployment (same as Step 1C)
MIN_SURPLUS_DAYS_FOR_BATTERY = 150

# Saturation dominance filter: once increasing a storage dimension doesn't
# improve the score (within epsilon), skip all higher levels for that
# dimension in the current outer context. Scores are monotonically
# non-decreasing in each storage dimension (more capacity = same or better).
SATURATION_EPS = 1e-6  # in 0-1 score space (~0.0001 percentage points)

# Batch sizes
NM_CHUNK = 10000        # cap computation chunk
MAX_MIX_BATCH = 100     # storage evaluation batch
CHUNK_CANDIDATE_LIMIT = 500_000  # flush to disk above this


# ══════════════════════════════════════════════════════════════════════════════
# NUMBA KERNEL: Cap-check + saturation pruning + feasible extraction
# ══════════════════════════════════════════════════════════════════════════════

@njit(cache=True, fastmath=True)
def _extract_feasible(scores, n_mixes, n_b4, n_b8, n_l, n_h2,
                      bat4_arr, bat8_arr, ldes_arr, h2_arr,
                      bat4_caps, bat8_caps, ldes_caps, surplus_days,
                      target, min_surplus_days, saturation_eps):
    """Extract feasible (score >= target, storage > 0) combos with cap-check
    and saturation dominance pruning, entirely in Numba.

    Returns:
      out_mix_idx: which mix (within batch) each result belongs to
      out_b4, out_b8, out_ldes, out_h2: storage levels for each result
      out_score: hourly match score for each result
      n_out: number of valid results written

    All output arrays are pre-allocated to worst-case size; only [:n_out] is valid.
    """
    max_out = n_mixes * n_b4 * n_b8 * n_l * n_h2
    out_mix_idx = np.empty(max_out, dtype=np.int64)
    out_b4 = np.empty(max_out, dtype=np.float64)
    out_b8 = np.empty(max_out, dtype=np.float64)
    out_ldes = np.empty(max_out, dtype=np.float64)
    out_h2 = np.empty(max_out, dtype=np.float64)
    out_score = np.empty(max_out, dtype=np.float64)
    n_out = 0

    for mi in range(n_mixes):
        b4_max = bat4_caps[mi]
        b8_max = bat8_caps[mi]
        l_max = ldes_caps[mi]
        n_sd = surplus_days[mi]
        s_base = mi * n_b4 * n_b8 * n_l * n_h2  # offset into flat scores

        prev_b4_best = -1.0
        for b4_idx in range(n_b4):
            bp = bat4_arr[b4_idx]
            if bp > 0 and (n_sd < min_surplus_days or bp > b4_max):
                break  # sorted ascending

            cur_b4_best = -1.0
            prev_b8_best = -1.0
            for b8_idx in range(n_b8):
                b8p = bat8_arr[b8_idx]
                if b8p > 0 and (n_sd < min_surplus_days or b8p > b8_max):
                    break

                cur_b8_best = -1.0
                prev_l_best = -1.0
                for l_idx in range(n_l):
                    lp = ldes_arr[l_idx]
                    if lp > 0 and lp > l_max:
                        break

                    cur_l_best = -1.0
                    for h2_idx in range(n_h2):
                        h2p = h2_arr[h2_idx]
                        idx = (s_base + b4_idx * n_b8 * n_l * n_h2 +
                               b8_idx * n_l * n_h2 + l_idx * n_h2 + h2_idx)
                        score = scores[idx]

                        if score >= 0:
                            if score > cur_l_best:
                                cur_l_best = score

                        # Skip no-storage and below-target
                        if bp == 0 and b8p == 0 and lp == 0 and h2p == 0:
                            continue
                        if score < 0 or score < target:
                            continue

                        out_mix_idx[n_out] = mi
                        out_b4[n_out] = bp
                        out_b8[n_out] = b8p
                        out_ldes[n_out] = lp
                        out_h2[n_out] = h2p
                        out_score[n_out] = score
                        n_out += 1

                    # LDES saturation
                    if (l_idx > 0 and cur_l_best >= 0 and prev_l_best >= 0
                            and cur_l_best <= prev_l_best + saturation_eps):
                        break
                    if cur_l_best >= 0 and cur_l_best > prev_l_best:
                        prev_l_best = cur_l_best
                    if cur_l_best > cur_b8_best:
                        cur_b8_best = cur_l_best

                # bat8 saturation
                if (b8_idx > 0 and cur_b8_best >= 0 and prev_b8_best >= 0
                        and cur_b8_best <= prev_b8_best + saturation_eps):
                    break
                if cur_b8_best >= 0 and cur_b8_best > prev_b8_best:
                    prev_b8_best = cur_b8_best
                if cur_b8_best > cur_b4_best:
                    cur_b4_best = cur_b8_best

            # bat4 saturation
            if (b4_idx > 0 and cur_b4_best >= 0 and prev_b4_best >= 0
                    and cur_b4_best <= prev_b4_best + saturation_eps):
                break
            if cur_b4_best >= 0 and cur_b4_best > prev_b4_best:
                prev_b4_best = cur_b4_best

    return out_mix_idx[:n_out], out_b4[:n_out], out_b8[:n_out], out_ldes[:n_out], out_h2[:n_out], out_score[:n_out], n_out


# ══════════════════════════════════════════════════════════════════════════════
# STORAGE LEVEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_storage_levels(threshold):
    """Return storage level arrays for a given threshold regime.

    All values are percentages of annual demand (same units as Step 1C).

    For 65-92.5%: Fine-grained levels across all storage types. Fills the
    critical 0-to-1% range for bat4, 0-to-2% for bat8, 0-to-5% for LDES
    where Step 1C's coarse grid had zero coverage.

    For >=95%: LDES intermediates only. Step 1C's bat4/bat8/H2 levels
    already work at >=95% (higher procurement → larger caps → coarse levels
    fit within caps). Only LDES 0→5% gap needs filling.
    """
    if threshold >= 95:
        return {
            'bat4': [0, 1, 3, 5],          # same as 1C (caps are larger at high procurement)
            'bat8': [0, 2, 4, 6],           # same as 1C
            'ldes': [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0,
                     5.0, 7.0, 10.0, 15.0, 20.0],   # intermediates filling 0→5 gap
            'h2':   [0, 5, 10, 20],         # same as 1C
        }
    else:
        return {
            'bat4': [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
                     0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
            'bat8': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75,
                     1.0, 1.5, 2.0, 3.0, 4.0],
            'ldes': [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5,
                     2.0, 2.5, 3.0, 5.0, 7.0, 10.0],
            'h2':   [0],
        }


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_coarse_cache(iso):
    """Load coarse cache from Step 1B.

    Returns (combos, scores) where:
      combos: (N, n_resources) int array of resource percentages
      scores: (N,) float64 array of hourly match scores in [0, 1]
    """
    path = os.path.join(STEP1_RAW_PFS_PARQUET_DIR, f'{iso}_coarse_cache.parquet')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Coarse cache not found: {path}")

    table = pq.read_table(path)
    rtypes = get_resource_types(iso)
    combos = np.column_stack([
        table.column(rt).to_numpy().astype(np.int64) for rt in rtypes
    ])
    scores = table.column('score').to_numpy().astype(np.float64)
    return combos, scores


# ══════════════════════════════════════════════════════════════════════════════
# CORE: Process a single ISO/threshold
# ══════════════════════════════════════════════════════════════════════════════

def process_threshold(iso, threshold, demand_arr, supply_matrix,
                      coarse_combos, coarse_scores):
    """Evaluate intermediate storage levels for near-miss mixes at one threshold.

    Algorithm:
      1. Identify near-miss mixes from coarse cache (score close to but below target)
      2. Compute physical storage caps per mix (max useful capacity)
      3. Evaluate all intermediate storage combos via batched Numba kernel
      4. Filter: keep combos where score >= target AND at least one storage > 0
      5. Return count of new feasible solutions

    Saves results to {STEP1D_OUTPUT_DIR}/{ISO}_t{XX}_storage_refined.parquet.
    """
    target = threshold / 100.0
    rtypes = get_resource_types(iso)
    n_res = len(rtypes)
    t_start = time.time()

    # ── Identify near-miss mixes ──
    near_miss_lower = max(target - NEAR_MISS_WIDTH, STORAGE_SWEEP_FLOOR)
    near_miss_mask = (coarse_scores >= near_miss_lower) & (coarse_scores < target)
    near_miss_idx = np.where(near_miss_mask)[0]

    if len(near_miss_idx) == 0:
        print(f"    {iso} {threshold}%: 0 near-miss mixes — skipping")
        return 0

    n_nm = len(near_miss_idx)

    # ── Storage levels for this threshold regime ──
    levels = get_storage_levels(threshold)
    bat4_arr = np.array(levels['bat4'], dtype=np.float64)
    bat8_arr = np.array(levels['bat8'], dtype=np.float64)
    ldes_arr = np.array(levels['ldes'], dtype=np.float64)
    h2_arr = np.array(levels['h2'], dtype=np.float64)
    n_b4 = len(bat4_arr)
    n_b8 = len(bat8_arr)
    n_l = len(ldes_arr)
    n_h2 = len(h2_arr)
    n_combos = n_b4 * n_b8 * n_l * n_h2

    print(f"    {iso} {threshold}%: {n_nm:,} near-miss mixes, "
          f"{n_b4}×{n_b8}×{n_l}×{n_h2}={n_combos:,} storage combos")

    # ── Constants ──
    batt_eff = BATTERY_EFFICIENCY
    batt8_eff = BATTERY8_EFFICIENCY
    ldes_eff = LDES_EFFICIENCY
    ldes_window_hours = LDES_WINDOW_DAYS * 24
    h2_eff = H2_EFFICIENCY
    h2_dur = float(H2_DURATION_HOURS)
    h2_window_hours = H2_WINDOW_DAYS * 24
    batt8_window = 48

    # ── Compute storage caps (chunked, Numba parallel) ──
    b4_caps = np.empty(n_nm, dtype=np.float64)
    b8_caps = np.empty(n_nm, dtype=np.float64)
    l_caps = np.empty(n_nm, dtype=np.float64)
    hc_arr = np.empty(n_nm, dtype=np.int64)
    sd_arr = np.empty(n_nm, dtype=np.int64)

    n_cap_chunks = (n_nm + NM_CHUNK - 1) // NM_CHUNK
    for cs in range(0, n_nm, NM_CHUNK):
        ce = min(cs + NM_CHUNK, n_nm)
        if n_cap_chunks > 1:
            print(f"\r      Cap computation: {cs:,}/{n_nm:,}", end="", flush=True)
        chunk_fracs = coarse_combos[near_miss_idx[cs:ce]].astype(np.float64) / 100.0
        chunk_supply = chunk_fracs @ supply_matrix
        chunk_n = ce - cs
        cb4, cb8, cl, chc, csd = _batch_compute_storage_caps(
            demand_arr, chunk_supply, 1.0, chunk_n,
            BATTERY_DURATION_HOURS, BATTERY8_DURATION_HOURS, LDES_DURATION_HOURS)
        b4_caps[cs:ce] = cb4
        b8_caps[cs:ce] = cb8
        l_caps[cs:ce] = cl
        hc_arr[cs:ce] = chc
        sd_arr[cs:ce] = csd

    if n_cap_chunks > 1:
        print()

    # ── Filter: only mixes with any curtailment ──
    has_curtailment = hc_arr.astype(bool)
    valid_ci = np.where(has_curtailment)[0]

    # Build valid mix list with cap info (caps converted to percentage with headroom)
    nm_valid = [
        (int(ci), int(near_miss_idx[ci]),
         b4_caps[ci] * 100.0 * 1.1,
         b8_caps[ci] * 100.0 * 1.1,
         l_caps[ci] * 100.0 * 1.1,
         int(sd_arr[ci]))
        for ci in valid_ci
    ]

    if not nm_valid:
        print(f"      0 mixes with curtailment — skipping")
        return 0

    print(f"      {len(nm_valid):,} mixes with curtailment")

    # ── Cap distribution summary (diagnostic) ──
    valid_b4 = b4_caps[valid_ci] * 100.0
    valid_l = l_caps[valid_ci] * 100.0
    print(f"      Cap ranges: bat4=[{valid_b4.min():.3f}%, {np.median(valid_b4):.3f}%, {valid_b4.max():.3f}%]  "
          f"LDES=[{valid_l.min():.3f}%, {np.median(valid_l):.3f}%, {valid_l.max():.3f}%]")

    # ── Batch evaluate storage combos ──
    # Uses Numba kernel for cap-check + saturation pruning + feasible extraction.
    # Columnar arrays replace per-candidate dict construction.
    n_valid = len(nm_valid)
    n_total_batches = (n_valid + MAX_MIX_BATCH - 1) // MAX_MIX_BATCH
    total_feasible = 0
    chunk_num = 0

    # Columnar accumulators for all feasible results
    all_orig_idx = []       # original coarse_combos index per feasible result
    all_b4 = []
    all_b8 = []
    all_ldes = []
    all_h2 = []
    all_score = []

    # Pre-build numpy arrays of caps/surplus_days for valid mixes (for Numba kernel)
    valid_b4_caps_pct = np.array([v[2] for v in nm_valid], dtype=np.float64)
    valid_b8_caps_pct = np.array([v[3] for v in nm_valid], dtype=np.float64)
    valid_l_caps_pct = np.array([v[4] for v in nm_valid], dtype=np.float64)
    valid_sd = np.array([v[5] for v in nm_valid], dtype=np.int64)
    valid_orig_idx = np.array([v[1] for v in nm_valid], dtype=np.int64)

    # Dedup set (stays in Python — Numba can't do tuple-set dedup across batches)
    seen = set()

    for batch_start in range(0, n_valid, MAX_MIX_BATCH):
        batch_end = min(batch_start + MAX_MIX_BATCH, n_valid)
        n_batch = batch_end - batch_start
        batch_num = batch_start // MAX_MIX_BATCH + 1

        print(f"\r      Batch {batch_num}/{n_total_batches} "
              f"({total_feasible:,} feasible so far)", end="", flush=True)

        # Vectorized supply: single matmul
        batch_orig_idx = valid_orig_idx[batch_start:batch_end]
        batch_supply = (coarse_combos[batch_orig_idx].astype(np.float64) / 100.0) @ supply_matrix

        # Numba parallel batch evaluation — scores shape (n_batch, n_combos)
        batch_scores = _batch_mixes_storage_screen(
            demand_arr, batch_supply, 1.0, n_batch,
            bat4_arr, bat8_arr, ldes_arr, n_b4, n_b8, n_l,
            batt_eff, batt8_eff, ldes_eff,
            BATTERY_DURATION_HOURS, BATTERY8_DURATION_HOURS, LDES_DURATION_HOURS,
            ldes_window_hours, batt8_window,
            h2_arr, n_h2, h2_eff, h2_dur, h2_window_hours)

        # Flatten scores for Numba kernel (expects 1D with mix-major layout)
        flat_scores = batch_scores.ravel()

        # Numba kernel: cap-check + saturation pruning + feasible extraction
        batch_b4_caps = valid_b4_caps_pct[batch_start:batch_end]
        batch_b8_caps = valid_b8_caps_pct[batch_start:batch_end]
        batch_l_caps = valid_l_caps_pct[batch_start:batch_end]
        batch_sd = valid_sd[batch_start:batch_end]

        f_mix_idx, f_b4, f_b8, f_ldes, f_h2, f_score, n_f = _extract_feasible(
            flat_scores, n_batch, n_b4, n_b8, n_l, n_h2,
            bat4_arr, bat8_arr, ldes_arr, h2_arr,
            batch_b4_caps, batch_b8_caps, batch_l_caps, batch_sd,
            target, MIN_SURPLUS_DAYS_FOR_BATTERY, SATURATION_EPS)

        # Dedup in Python (cross-batch dedup requires set)
        for fi in range(n_f):
            mi = int(f_mix_idx[fi])
            orig_idx = int(batch_orig_idx[mi])
            mix = coarse_combos[orig_idx]
            mix_key = tuple(int(mix[k]) for k in range(n_res))
            bp = f_b4[fi]
            b8p = f_b8[fi]
            lp = f_ldes[fi]
            h2p = f_h2[fi]
            key = (mix_key, bp, b8p, lp, h2p)
            if key in seen:
                continue
            seen.add(key)
            all_orig_idx.append(orig_idx)
            all_b4.append(bp)
            all_b8.append(b8p)
            all_ldes.append(lp)
            all_h2.append(h2p)
            all_score.append(round(f_score[fi] * 100.0, 2))
            total_feasible += 1

        # Flush to disk if too many candidates in memory
        if total_feasible >= CHUNK_CANDIDATE_LIMIT:
            candidates = _columnar_to_candidates(
                iso, rtypes, coarse_combos, all_orig_idx,
                all_b4, all_b8, all_ldes, all_h2, all_score)
            chunk_num = _save_chunk(iso, threshold, candidates, chunk_num)
            all_orig_idx.clear()
            all_b4.clear()
            all_b8.clear()
            all_ldes.clear()
            all_h2.clear()
            all_score.clear()

    print(f"\r      {n_total_batches}/{n_total_batches} batches done — "
          f"{total_feasible:,} feasible solutions"
          f"                              ")

    # ── Save results ──
    if all_orig_idx or chunk_num > 0:
        candidates = _columnar_to_candidates(
            iso, rtypes, coarse_combos, all_orig_idx,
            all_b4, all_b8, all_ldes, all_h2, all_score)
        if chunk_num > 0:
            _merge_chunks_and_finalize(iso, threshold, candidates, chunk_num)
        else:
            _save_final(iso, threshold, candidates)

    elapsed = time.time() - t_start
    print(f"    {iso} {threshold}%: {total_feasible:,} new solutions in {elapsed:.1f}s")
    return total_feasible


# ══════════════════════════════════════════════════════════════════════════════
# PARQUET OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def _columnar_to_candidates(iso, rtypes, coarse_combos, orig_idx_list,
                            b4_list, b8_list, ldes_list, h2_list, score_list):
    """Convert columnar accumulator lists into candidate dicts for parquet output."""
    n_res = len(rtypes)
    candidates = []
    for i in range(len(orig_idx_list)):
        mix = coarse_combos[orig_idx_list[i]]
        candidates.append({
            'resource_mix': {rt: int(mix[j]) for j, rt in enumerate(rtypes)},
            'battery_dispatch_pct': b4_list[i],
            'battery8_dispatch_pct': b8_list[i],
            'ldes_dispatch_pct': ldes_list[i],
            'h2_dispatch_pct': h2_list[i],
            'hourly_match_score': score_list[i],
        })
    return candidates


def _candidate_to_row(iso, threshold, c):
    """Convert a candidate dict to a flat row dict."""
    rtypes = get_resource_types(iso)
    row = {
        'iso': iso,
        'threshold': float(threshold),
    }
    for rt in rtypes:
        row[rt] = c['resource_mix'].get(rt, 0)
    row['battery_dispatch_pct'] = float(c['battery_dispatch_pct'])
    row['battery8_dispatch_pct'] = float(c['battery8_dispatch_pct'])
    row['ldes_dispatch_pct'] = float(c['ldes_dispatch_pct'])
    row['h2_dispatch_pct'] = float(c.get('h2_dispatch_pct', 0))
    row['hourly_match_score'] = c['hourly_match_score']
    row['pareto_type'] = 'storage_refined'
    return row


def _rows_to_table(rows):
    """Convert row dicts to a PyArrow table."""
    if not rows:
        return None
    return pa.table({col: [r[col] for r in rows] for col in rows[0].keys()})


def _output_path(iso, threshold):
    """Output parquet path for Step 1D results."""
    t_str = _normalize_threshold_str(threshold)
    return os.path.join(STEP1D_OUTPUT_DIR, f'{iso}_t{t_str}_storage_refined.parquet')


def _chunk_path(iso, threshold, chunk_num):
    """Temporary chunk path."""
    t_str = _normalize_threshold_str(threshold)
    return os.path.join(STEP1D_OUTPUT_DIR, f'{iso}_t{t_str}_chunk{chunk_num}.parquet')


def _save_chunk(iso, threshold, candidates, chunk_num):
    """Save candidates to a numbered chunk parquet. Returns chunk_num + 1."""
    if not candidates:
        return chunk_num
    os.makedirs(STEP1D_OUTPUT_DIR, exist_ok=True)
    rows = [_candidate_to_row(iso, threshold, c) for c in candidates]
    table = _rows_to_table(rows)
    if table is None:
        return chunk_num
    path = _chunk_path(iso, threshold, chunk_num)
    pq.write_table(table, path, compression='snappy')
    print(f"\n      Chunk {chunk_num}: {len(candidates):,} candidates → {os.path.basename(path)}")
    return chunk_num + 1


def _save_final(iso, threshold, candidates):
    """Save all candidates to the final output parquet."""
    if not candidates:
        return
    os.makedirs(STEP1D_OUTPUT_DIR, exist_ok=True)
    rows = [_candidate_to_row(iso, threshold, c) for c in candidates]
    table = _rows_to_table(rows)
    if table is None:
        return
    path = _output_path(iso, threshold)
    pq.write_table(table, path, compression='snappy')
    print(f"      Saved {len(candidates):,} solutions → {os.path.basename(path)}")


def _merge_chunks_and_finalize(iso, threshold, remaining_candidates, total_chunks):
    """Merge chunk files + remaining into final parquet, clean up chunks."""
    os.makedirs(STEP1D_OUTPUT_DIR, exist_ok=True)
    chunk_tables = []
    chunk_files = []

    for cn in range(total_chunks):
        cp = _chunk_path(iso, threshold, cn)
        if os.path.exists(cp):
            try:
                chunk_tables.append(pq.read_table(cp))
                chunk_files.append(cp)
            except Exception as e:
                print(f"      Warning: Failed to read chunk {cp}: {e}")

    if remaining_candidates:
        rows = [_candidate_to_row(iso, threshold, c) for c in remaining_candidates]
        remaining_table = _rows_to_table(rows)
        if remaining_table is not None:
            chunk_tables.append(remaining_table)

    if not chunk_tables:
        return

    merged = pa.concat_tables(chunk_tables, promote_options='permissive')
    path = _output_path(iso, threshold)
    pq.write_table(merged, path, compression='snappy')
    print(f"      Merged {len(chunk_files)} chunks → "
          f"{merged.num_rows:,} total → {os.path.basename(path)}")

    for cp in chunk_files:
        try:
            os.remove(cp)
        except OSError:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# ISO PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def process_iso(iso, thresholds, demand_data, gen_profiles):
    """Process all thresholds for a single ISO."""
    print(f"\n{'='*60}")
    print(f"  Processing {iso}")
    print(f"{'='*60}")

    iso_start = time.time()

    # Load profiles and build numpy arrays
    supply_profiles = get_supply_profiles(iso, gen_profiles)
    demand_arr, supply_matrix = prepare_numpy_profiles(iso, demand_data[iso]['normalized'], supply_profiles)

    # Load coarse cache
    coarse_combos, coarse_scores = load_coarse_cache(iso)
    print(f"  Coarse cache: {len(coarse_combos):,} mixes, "
          f"score range [{coarse_scores.min():.4f}, {coarse_scores.max():.4f}]")

    total_new = 0
    for threshold in thresholds:
        # Skip if output already exists
        out_path = _output_path(iso, threshold)
        if os.path.exists(out_path):
            existing = pq.read_metadata(out_path).num_rows
            print(f"    {iso} {threshold}%: Already done ({existing:,} solutions) — skipping")
            continue

        n_new = process_threshold(
            iso, threshold, demand_arr, supply_matrix,
            coarse_combos, coarse_scores)
        total_new += n_new

    elapsed = time.time() - iso_start
    print(f"\n  {iso}: {total_new:,} total new solutions in {elapsed:.1f}s")
    return total_new


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def parse_args(argv):
    """Parse CLI args.

    Usage:
      step1d_storage_refinement.py [ISO ...] [--threshold T1,T2,...] [--force]

    Examples:
      step1d_storage_refinement.py ERCOT --threshold 75
      step1d_storage_refinement.py CAISO ERCOT PJM
      step1d_storage_refinement.py --force   # rerun even if output exists
    """
    target_isos = []
    target_thresholds = None
    force = False

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == '--force':
            force = True
        elif arg.startswith('--threshold'):
            if '=' in arg:
                val = arg.split('=', 1)[1]
            else:
                i += 1
                val = argv[i]
            target_thresholds = [float(t.strip()) for t in val.split(',')]
        elif arg.upper() in ISOS:
            target_isos.append(arg.upper())
        i += 1

    if not target_isos:
        target_isos = ISOS

    if target_thresholds is None:
        target_thresholds = STEP1D_THRESHOLDS
    else:
        # Validate thresholds
        valid = set(STEP1D_THRESHOLDS)
        for t in target_thresholds:
            if t not in valid:
                print(f"  Warning: threshold {t} not in Step 1D range (65+) — skipping")
        target_thresholds = [t for t in target_thresholds if t in valid]

    return target_isos, target_thresholds, force


def main():
    print("=" * 70)
    print("  Step 1D: Storage Refinement Module")
    print("  Fills intermediate storage levels missing from Step 1C")
    print("=" * 70)

    target_isos, thresholds, force = parse_args(sys.argv[1:])
    print(f"  ISOs: {target_isos}")
    print(f"  Thresholds: {thresholds}")
    print(f"  Force rerun: {force}")

    if force:
        # Remove existing outputs for target ISOs/thresholds
        for iso in target_isos:
            for t in thresholds:
                out_path = _output_path(iso, t)
                if os.path.exists(out_path):
                    os.remove(out_path)
                    print(f"  Removed {os.path.basename(out_path)}")

    # Load shared data (demand, generation profiles)
    demand_data, gen_profiles, _, _ = load_data()

    # Numba JIT warmup
    if HAS_NUMBA:
        print("\n  Warming up Numba JIT...")
        warmup_start = time.time()
        _warmup_jit(demand_data, gen_profiles)
        print(f"  JIT warmup done in {time.time() - warmup_start:.1f}s")

    # Process each ISO
    grand_total = 0
    for iso in target_isos:
        try:
            n = process_iso(iso, thresholds, demand_data, gen_profiles)
            grand_total += n
        except FileNotFoundError as e:
            print(f"  Skipping {iso}: {e}")
        except Exception as e:
            print(f"  ERROR processing {iso}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"  Step 1D complete: {grand_total:,} total new solutions")
    print(f"  Output: {STEP1D_OUTPUT_DIR}/")
    print(f"{'='*70}")


def _warmup_jit(demand_data, gen_profiles):
    """Compile Numba functions with a tiny dummy run."""
    iso = 'ERCOT'  # smallest, fastest
    supply_profiles = get_supply_profiles(iso, gen_profiles)
    demand_arr, supply_matrix = prepare_numpy_profiles(
        iso, demand_data[iso]['normalized'], supply_profiles)

    # Tiny 2-mix warmup
    dummy_mix = np.array([[50, 25, 25, 0]], dtype=np.float64)
    dummy_supply = (dummy_mix / 100.0) @ supply_matrix

    # Warmup _batch_compute_storage_caps
    _batch_compute_storage_caps(
        demand_arr, dummy_supply, 1.0, 1,
        BATTERY_DURATION_HOURS, BATTERY8_DURATION_HOURS, LDES_DURATION_HOURS)

    # Warmup _batch_mixes_storage_screen
    b4 = np.array([0.0, 0.1], dtype=np.float64)
    b8 = np.array([0.0, 0.1], dtype=np.float64)
    ld = np.array([0.0, 0.5], dtype=np.float64)
    h2 = np.array([0.0], dtype=np.float64)
    scores_2d = _batch_mixes_storage_screen(
        demand_arr, dummy_supply, 1.0, 1,
        b4, b8, ld, 2, 2, 2,
        BATTERY_EFFICIENCY, BATTERY8_EFFICIENCY, LDES_EFFICIENCY,
        BATTERY_DURATION_HOURS, BATTERY8_DURATION_HOURS, LDES_DURATION_HOURS,
        LDES_WINDOW_DAYS * 24, 48,
        h2, 1, H2_EFFICIENCY, float(H2_DURATION_HOURS), H2_WINDOW_DAYS * 24)

    # Warmup _extract_feasible
    dummy_caps = np.array([1.0], dtype=np.float64)
    dummy_sd = np.array([200], dtype=np.int64)
    _extract_feasible(
        scores_2d.ravel(), 1, 2, 2, 2, 1,
        b4, b8, ld, h2,
        dummy_caps, dummy_caps, dummy_caps, dummy_sd,
        0.5, 150, 1e-6)


if __name__ == '__main__':
    main()
