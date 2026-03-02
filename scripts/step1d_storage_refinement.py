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
import subprocess
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

# Tiered near-miss half-widths by threshold range (in 0-1 score space).
# Higher thresholds use tighter windows because storage contributes fewer
# percentage points when procurement is already high.
def get_near_miss_width(threshold):
    """Return near-miss half-width for a given threshold."""
    if threshold >= 95:
        return 0.15   # 15pp — storage adds at most 5-10pp at high procurement
    elif threshold >= 85:
        return 0.30   # 30pp
    else:
        return 0.40   # 40pp — wide window needed for low-procurement mixes

# Minimum surplus days for battery deployment (same as Step 1C)
MIN_SURPLUS_DAYS_FOR_BATTERY = 150

# Saturation dominance filter: once increasing a storage dimension doesn't
# improve the score (within epsilon), skip all higher levels for that
# dimension in the current outer context. Scores are monotonically
# non-decreasing in each storage dimension (more capacity = same or better).
SATURATION_EPS = 1e-6  # in 0-1 score space (~0.0001 percentage points)

# Batch sizes
NM_CHUNK = 10000        # cap computation chunk
MAX_MIX_BATCH = 100     # storage evaluation batch (non-CAISO)
CAISO_MIX_BATCH = 500   # larger batches for CAISO (5x more mixes, amortize overhead)
CHUNK_CANDIDATE_LIMIT = 500_000  # flush to disk above this (non-CAISO)
CAISO_CHUNK_LIMIT = 100_000      # flush more often for CAISO (avoid OOM on 1.6M cache)
PROGRESS_SAVE_INTERVAL = 25      # save progress file every N batches (for resume)


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
    """Load base mix cache: coarse 5% grid + fine-grid mixes from raw PFS.

    The coarse cache (5% grid) only has hydro at 0,5,10,... which misses
    exact hydro cap values (e.g., hydro=2 for PJM). The raw PFS files from
    Step 1's adaptive refinement contain fine-grid mixes (1% step) at all
    hydro levels. Merging both ensures Step 1d can add storage to mixes at
    valid hydro cap values — without this, ALL storage mixes get killed by
    the hydro cap filter in Step 3.

    Returns (combos, scores) where:
      combos: (N, n_resources) int array of resource percentages
      scores: (N,) float64 array of hourly match scores in [0, 1]
    """
    import glob as globmod

    rtypes = get_resource_types(iso)

    # 1. Load coarse cache (5% grid) if it exists
    path = os.path.join(STEP1_RAW_PFS_PARQUET_DIR, f'{iso}_coarse_cache.parquet')
    has_coarse = os.path.exists(path)
    if has_coarse:
        table = pq.read_table(path)
        coarse_combos = np.column_stack([
            table.column(rt).to_numpy().astype(np.int64) for rt in rtypes
        ])
        coarse_scores = table.column('score').to_numpy().astype(np.float64)
        coarse_hydro_set = set(np.unique(coarse_combos[:, rtypes.index('hydro')]).tolist())
        coarse_hydro_arr = np.unique(coarse_combos[:, rtypes.index('hydro')])
        coarse_hydro_arr = np.unique(coarse_combos[:, rtypes.index('hydro')])
    else:
        coarse_combos = None
        coarse_scores = None
        coarse_hydro_set = set()
        coarse_hydro_arr = np.array([], dtype=np.int64)

    # 2. Load fine-grid mixes from raw PFS (1% adaptive refinement).
    #    These contain hydro at exact cap values (1,2,3,4,etc.) that the
    #    coarse 5% grid misses. If no coarse cache exists (some ISOs), the
    #    raw PFS becomes the sole source of base mixes.
    pfs_pattern = os.path.join(STEP1_RAW_PFS_PARQUET_DIR, f'{iso}_t*_raw_pfs.parquet')
    pfs_files = sorted(globmod.glob(pfs_pattern))

    if not pfs_files and not has_coarse:
        raise FileNotFoundError(
            f"No coarse cache or raw PFS files found for {iso}")

    hydro_col = rtypes.index('hydro')
    fine_combos_list = []
    fine_scores_list = []

    for fpath in pfs_files:
        try:
            t = pq.read_table(fpath, columns=rtypes + ['hourly_match_score'])
            c = np.column_stack([
                t.column(rt).to_numpy().astype(np.int64) for rt in rtypes
            ])
            s = t.column('hourly_match_score').to_numpy().astype(np.float64) / 100.0

            if has_coarse:
                # Only keep mixes at hydro levels NOT in the coarse grid
                hydro_vals = c[:, hydro_col]
                new_hydro_mask = ~np.isin(hydro_vals, coarse_hydro_arr)
                if new_hydro_mask.any():
                    fine_combos_list.append(c[new_hydro_mask])
                    fine_scores_list.append(s[new_hydro_mask])
            else:
                # No coarse cache — take all mixes from raw PFS
                fine_combos_list.append(c)
                fine_scores_list.append(s)
        except Exception:
            continue

    if fine_combos_list:
        fine_combos = np.vstack(fine_combos_list)
        fine_scores = np.concatenate(fine_scores_list)

        # Deduplicate fine-grid mixes: keep max score per unique combo
        n_res = fine_combos.shape[1]
        fine_contig = np.ascontiguousarray(fine_combos)
        void_view = fine_contig.view(
            np.dtype((np.void, fine_contig.dtype.itemsize * n_res))
        ).ravel()
        _, unique_idx, inverse = np.unique(
            void_view, return_index=True, return_inverse=True)

        unique_combos = fine_combos[unique_idx]
        unique_scores = np.full(len(unique_idx), -1.0)
        np.maximum.at(unique_scores, inverse, fine_scores)

        new_hydro_levels = sorted(
            set(unique_combos[:, hydro_col].tolist()) - coarse_hydro_set)

        if has_coarse:
            combos = np.vstack([coarse_combos, unique_combos])
            scores = np.concatenate([coarse_scores, unique_scores])
            print(f"  Coarse cache: {len(coarse_combos):,} + "
                  f"fine-grid: {len(unique_combos):,} = "
                  f"{len(combos):,} total mixes")
        else:
            combos = unique_combos
            scores = unique_scores
            print(f"  Raw PFS only (no coarse cache): "
                  f"{len(combos):,} unique mixes from "
                  f"{len(pfs_files)} files")

        if new_hydro_levels:
            print(f"  New hydro levels from fine grid: {new_hydro_levels}")

        return combos, scores

    if has_coarse:
        return coarse_combos, coarse_scores

    raise FileNotFoundError(
        f"No usable mix data found for {iso}")


# ══════════════════════════════════════════════════════════════════════════════
# CORE: Process a single ISO/threshold
# ══════════════════════════════════════════════════════════════════════════════

def _progress_path(iso, threshold):
    """Path for batch progress file (enables resume after crash)."""
    t_str = _normalize_threshold_str(threshold)
    return os.path.join(STEP1D_OUTPUT_DIR, f'{iso}_t{t_str}_progress.json')


def _partial_path(iso, threshold):
    """Path for partial results file (streaming parquet, in-progress)."""
    t_str = _normalize_threshold_str(threshold)
    return os.path.join(STEP1D_OUTPUT_DIR, f'{iso}_t{t_str}_partial.parquet')


def _save_progress(iso, threshold, batch_num, total_feasible, n_total_batches):
    """Save batch progress for resume capability."""
    import json
    os.makedirs(STEP1D_OUTPUT_DIR, exist_ok=True)
    progress = {
        'iso': iso,
        'threshold': threshold,
        'batch_num': batch_num,
        'total_feasible': total_feasible,
        'n_total_batches': n_total_batches,
        'timestamp': time.time(),
    }
    path = _progress_path(iso, threshold)
    with open(path, 'w') as f:
        json.dump(progress, f)


def _load_progress(iso, threshold):
    """Load batch progress for resume. Returns (batch_num, total_feasible) or None."""
    import json
    path = _progress_path(iso, threshold)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            progress = json.load(f)
        return progress.get('batch_num', 0), progress.get('total_feasible', 0)
    except Exception:
        return None


def _cleanup_progress(iso, threshold):
    """Remove progress file after successful completion."""
    path = _progress_path(iso, threshold)
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def _build_batch_table(iso, threshold, rtypes, coarse_combos,
                       batch_orig_idx, f_mix_idx, f_b4, f_b8, f_ldes,
                       f_h2, f_score, n_f):
    """Build a PyArrow table directly from Numba kernel output arrays.

    Avoids the slow Python dict intermediate — goes numpy → PyArrow columnar
    directly. ~10x faster than _columnar_to_candidates + _candidate_to_row.
    """
    if n_f == 0:
        return None

    # Map batch-local mix indices to original coarse combo indices
    orig_indices = batch_orig_idx[f_mix_idx[:n_f].astype(np.int64)]

    # Build resource columns directly from coarse_combos array
    arrays = {
        'iso': pa.array([iso] * n_f, type=pa.string()),
        'threshold': pa.array(np.full(n_f, float(threshold)), type=pa.float64()),
    }
    for j, rt in enumerate(rtypes):
        arrays[rt] = pa.array(coarse_combos[orig_indices, j])

    arrays['battery_dispatch_pct'] = pa.array(f_b4[:n_f])
    arrays['battery8_dispatch_pct'] = pa.array(f_b8[:n_f])
    arrays['ldes_dispatch_pct'] = pa.array(f_ldes[:n_f])
    arrays['h2_dispatch_pct'] = pa.array(f_h2[:n_f])
    arrays['hourly_match_score'] = pa.array(np.round(f_score[:n_f] * 100.0, 2))
    arrays['pareto_type'] = pa.array(['storage_refined'] * n_f, type=pa.string())

    return pa.table(arrays)


def process_threshold(iso, threshold, demand_arr, supply_matrix,
                      coarse_combos, coarse_scores):
    """Evaluate intermediate storage levels for near-miss mixes at one threshold.

    Algorithm:
      1. Identify near-miss mixes from coarse cache (score close to but below target)
      2. Compute physical storage caps per mix (max useful capacity)
      3. Evaluate all intermediate storage combos via batched Numba kernel
      4. Filter: keep combos where score >= target AND at least one storage > 0
      5. Return count of new feasible solutions

    CAISO optimizations (1.6M coarse mixes vs 195K-390K for other ISOs):
      - Larger batch size (500 vs 100) to amortize Python loop overhead
      - Streaming ParquetWriter — writes each batch directly to disk (no accumulation)
      - More frequent flush (100K vs 500K candidate limit)
      - Batch-level resume from progress file after crash
      - Direct numpy→PyArrow columnar conversion (no Python dict intermediate)
      - No cross-batch dedup set (coarse cache has zero duplicates)

    Saves results to {STEP1D_OUTPUT_DIR}/{ISO}_t{XX}_storage_refined.parquet.
    """
    target = threshold / 100.0
    rtypes = get_resource_types(iso)
    n_res = len(rtypes)
    t_start = time.time()

    # ── ISO-adaptive batch parameters ──
    mix_batch = CAISO_MIX_BATCH if iso == 'CAISO' else MAX_MIX_BATCH
    chunk_limit = CAISO_CHUNK_LIMIT if iso == 'CAISO' else CHUNK_CANDIDATE_LIMIT

    # ── Identify near-miss mixes ──
    near_miss_lower = max(target - get_near_miss_width(threshold), STORAGE_SWEEP_FLOOR)
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
          f"{n_b4}×{n_b8}×{n_l}×{n_h2}={n_combos:,} storage combos "
          f"(batch={mix_batch})")

    # ── Constants ──
    batt_eff = BATTERY_EFFICIENCY
    batt8_eff = BATTERY8_EFFICIENCY
    ldes_eff = LDES_EFFICIENCY
    ldes_window_hours = LDES_WINDOW_DAYS * 24
    h2_eff = H2_EFFICIENCY
    h2_dur = float(H2_DURATION_HOURS)
    h2_window_hours = H2_WINDOW_DAYS * 24
    batt8_window = 48

    # ── Compute storage caps + total surplus (chunked, Numba parallel) ──
    b4_caps = np.empty(n_nm, dtype=np.float64)
    b8_caps = np.empty(n_nm, dtype=np.float64)
    l_caps = np.empty(n_nm, dtype=np.float64)
    hc_arr = np.empty(n_nm, dtype=np.int64)
    sd_arr = np.empty(n_nm, dtype=np.int64)
    total_surplus = np.empty(n_nm, dtype=np.float64)

    demand_total = demand_arr.sum()
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
        # Total annual surplus energy per mix (for curtailment-magnitude gate)
        chunk_surplus = np.maximum(chunk_supply - demand_arr[np.newaxis, :], 0.0)
        total_surplus[cs:ce] = chunk_surplus.sum(axis=1)

    if n_cap_chunks > 1:
        print()

    # ── Filter 1: only mixes with any curtailment ──
    has_curtailment = hc_arr.astype(bool)
    n_with_curtailment = has_curtailment.sum()

    # ── Filter 2: curtailment-magnitude gate ──
    # Physical upper bound: max score lift = total_surplus × best_eff / demand_total
    # If this is less than the score gap, storage CANNOT bridge it — skip.
    best_eff = max(BATTERY_EFFICIENCY, BATTERY8_EFFICIENCY, LDES_EFFICIENCY)
    score_gap = target - coarse_scores[near_miss_idx]
    max_score_lift = total_surplus * best_eff / demand_total
    can_bridge = max_score_lift >= score_gap

    # Combine both filters
    valid_mask = has_curtailment & can_bridge
    valid_ci = np.where(valid_mask)[0]
    n_gated = int(n_with_curtailment) - len(valid_ci)

    if len(valid_ci) == 0:
        print(f"      0 viable mixes (curtailment: {n_with_curtailment:,}, "
              f"gated out: {n_gated:,} insufficient surplus) — skipping")
        return 0

    # Build numpy arrays directly (no Python list of tuples)
    valid_b4_caps_pct = b4_caps[valid_ci] * 100.0 * 1.1
    valid_b8_caps_pct = b8_caps[valid_ci] * 100.0 * 1.1
    valid_l_caps_pct = l_caps[valid_ci] * 100.0 * 1.1
    valid_sd = sd_arr[valid_ci]
    valid_orig_idx = near_miss_idx[valid_ci]

    n_valid = len(valid_ci)
    print(f"      {n_valid:,} viable mixes (curtailment: {n_with_curtailment:,}, "
          f"gated: {n_gated:,} insufficient surplus)")

    # ── Cap distribution summary (diagnostic) ──
    valid_b4_diag = b4_caps[valid_ci] * 100.0
    valid_l_diag = l_caps[valid_ci] * 100.0
    print(f"      Cap ranges: bat4=[{valid_b4_diag.min():.3f}%, {np.median(valid_b4_diag):.3f}%, {valid_b4_diag.max():.3f}%]  "
          f"LDES=[{valid_l_diag.min():.3f}%, {np.median(valid_l_diag):.3f}%, {valid_l_diag.max():.3f}%]")

    # ── Check for resume state ──
    n_total_batches = (n_valid + mix_batch - 1) // mix_batch
    resume_batch = 0
    total_feasible = 0
    partial_path = _partial_path(iso, threshold)

    progress = _load_progress(iso, threshold)
    if progress is not None:
        resume_batch, total_feasible = progress
        if os.path.exists(partial_path):
            try:
                existing_rows = pq.read_metadata(partial_path).num_rows
                print(f"      Resuming from batch {resume_batch}/{n_total_batches} "
                      f"({existing_rows:,} rows in partial file)")
            except Exception:
                # Partial file corrupted — restart
                resume_batch = 0
                total_feasible = 0
                if os.path.exists(partial_path):
                    os.remove(partial_path)
        else:
            # Progress file but no partial — restart
            resume_batch = 0
            total_feasible = 0

    # ── Batch evaluate with streaming ParquetWriter ──
    os.makedirs(STEP1D_OUTPUT_DIR, exist_ok=True)
    writer = None
    schema = None
    chunk_tables = []        # buffer small tables before writing
    chunk_rows_buffered = 0
    prev_partial_path = None  # tracks renamed partial from previous run (for resume merge)

    # If resuming, rename existing partial so we don't overwrite it
    # (ParquetWriter creates new files, can't append)
    if resume_batch > 0 and os.path.exists(partial_path):
        prev_partial_path = partial_path + '.prev'
        try:
            os.rename(partial_path, prev_partial_path)
            schema = pq.read_schema(prev_partial_path)
        except Exception:
            prev_partial_path = None

    start_mix = resume_batch * mix_batch

    for batch_start in range(start_mix, n_valid, mix_batch):
        batch_end = min(batch_start + mix_batch, n_valid)
        n_batch = batch_end - batch_start
        batch_num = batch_start // mix_batch + 1

        if batch_num % max(1, n_total_batches // 20) == 0 or batch_num == 1:
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

        # Build PyArrow table directly from numpy arrays (no Python dict overhead)
        if n_f > 0:
            batch_table = _build_batch_table(
                iso, threshold, rtypes, coarse_combos,
                batch_orig_idx, f_mix_idx, f_b4, f_b8, f_ldes,
                f_h2, f_score, n_f)
            if batch_table is not None:
                chunk_tables.append(batch_table)
                chunk_rows_buffered += batch_table.num_rows
                total_feasible += batch_table.num_rows

        # Flush buffered tables to streaming writer when above limit
        if chunk_rows_buffered >= chunk_limit:
            merged_chunk = pa.concat_tables(chunk_tables, promote_options='permissive')
            if writer is None:
                schema = merged_chunk.schema
                writer = pq.ParquetWriter(partial_path, schema, compression='snappy')
            writer.write_table(merged_chunk)
            chunk_tables.clear()
            chunk_rows_buffered = 0

        # Save progress periodically (for resume after crash)
        if batch_num % PROGRESS_SAVE_INTERVAL == 0:
            # Flush any buffered tables before saving progress
            if chunk_tables:
                merged_chunk = pa.concat_tables(chunk_tables, promote_options='permissive')
                if writer is None:
                    schema = merged_chunk.schema
                    writer = pq.ParquetWriter(partial_path, schema, compression='snappy')
                writer.write_table(merged_chunk)
                chunk_tables.clear()
                chunk_rows_buffered = 0
            _save_progress(iso, threshold, batch_num, total_feasible, n_total_batches)

    print(f"\r      {n_total_batches}/{n_total_batches} batches done — "
          f"{total_feasible:,} feasible solutions"
          f"                              ")

    # ── Finalize: flush remaining buffer + close writer ──
    if chunk_tables:
        merged_chunk = pa.concat_tables(chunk_tables, promote_options='permissive')
        if writer is None:
            schema = merged_chunk.schema
            writer = pq.ParquetWriter(partial_path, schema, compression='snappy')
        writer.write_table(merged_chunk)
        chunk_tables.clear()

    if writer is not None:
        writer.close()

    # ── Merge previous partial (if resuming) + current partial → final output ──
    final_path = _output_path(iso, threshold)
    tables_to_merge = []
    if prev_partial_path and os.path.exists(prev_partial_path):
        try:
            tables_to_merge.append(pq.read_table(prev_partial_path))
        except Exception:
            pass
    if os.path.exists(partial_path):
        try:
            tables_to_merge.append(pq.read_table(partial_path))
        except Exception:
            pass

    if tables_to_merge:
        merged = pa.concat_tables(tables_to_merge, promote_options='permissive')
        pq.write_table(merged, final_path, compression='snappy')
        total_feasible = merged.num_rows  # accurate count including resumed rows
        # Clean up temp files
        for tmp in [partial_path, prev_partial_path]:
            if tmp and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        print(f"      Saved {total_feasible:,} solutions → {os.path.basename(final_path)}")
    elif total_feasible == 0:
        # Write an empty parquet for "done" signal so we don't re-attempt
        _save_empty(iso, threshold)

    # Cleanup progress file
    _cleanup_progress(iso, threshold)

    elapsed = time.time() - t_start
    print(f"    {iso} {threshold}%: {total_feasible:,} new solutions in {elapsed:.1f}s")
    return total_feasible


# ══════════════════════════════════════════════════════════════════════════════
# PARQUET OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def _output_path(iso, threshold):
    """Output parquet path for Step 1D results."""
    t_str = _normalize_threshold_str(threshold)
    return os.path.join(STEP1D_OUTPUT_DIR, f'{iso}_t{t_str}_storage_refined.parquet')


def _save_empty(iso, threshold):
    """Write an empty parquet so we don't re-attempt this threshold."""
    os.makedirs(STEP1D_OUTPUT_DIR, exist_ok=True)
    rtypes = get_resource_types(iso)
    arrays = {
        'iso': pa.array([], type=pa.string()),
        'threshold': pa.array([], type=pa.float64()),
    }
    for rt in rtypes:
        arrays[rt] = pa.array([], type=pa.float64())
    arrays['battery_dispatch_pct'] = pa.array([], type=pa.float64())
    arrays['battery8_dispatch_pct'] = pa.array([], type=pa.float64())
    arrays['ldes_dispatch_pct'] = pa.array([], type=pa.float64())
    arrays['h2_dispatch_pct'] = pa.array([], type=pa.float64())
    arrays['hourly_match_score'] = pa.array([], type=pa.float64())
    arrays['pareto_type'] = pa.array([], type=pa.string())
    table = pa.table(arrays)
    path = _output_path(iso, threshold)
    pq.write_table(table, path, compression='snappy')
    print(f"      Saved empty parquet → {os.path.basename(path)}")


# ══════════════════════════════════════════════════════════════════════════════
# AUTO-COMMIT
# ══════════════════════════════════════════════════════════════════════════════

def git_commit_threshold(iso, threshold, n_solutions):
    """Commit and push a single threshold's parquet file to preserve progress.

    Uses subprocess so failures don't crash the optimizer — just warns.
    """
    out_path = _output_path(iso, threshold)
    if not os.path.exists(out_path):
        print(f"      [auto-commit] No file to commit for {iso} {threshold}%")
        return False

    try:
        subprocess.run(['git', 'add', '-f', out_path],
                       check=True, capture_output=True, text=True)

        result = subprocess.run(['git', 'diff', '--cached', '--quiet'],
                                capture_output=True)
        if result.returncode == 0:
            return False  # no changes

        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        msg = (f"Step 1D: Storage refinement for {iso} ({threshold}%) — "
               f"{n_solutions:,} solutions, {size_mb:.1f} MB")
        subprocess.run(['git', 'commit', '-m', msg],
                       check=True, capture_output=True, text=True)

        for attempt in range(1, 5):
            result = subprocess.run(
                ['git', 'push', '-u', 'origin', 'HEAD'],
                capture_output=True, text=True)
            if result.returncode == 0:
                print(f"      [auto-commit] {iso} {threshold}% committed & pushed")
                return True
            if attempt < 4:
                wait = 2 ** attempt
                print(f"      [auto-commit] Push attempt {attempt} failed, retrying in {wait}s...")
                time.sleep(wait)

        print(f"      [auto-commit] Push failed for {iso} {threshold}% "
              f"(committed locally)")
        return True

    except subprocess.CalledProcessError as e:
        print(f"      [auto-commit] Git error for {iso} {threshold}%: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# ISO PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def process_iso(iso, thresholds, demand_data, gen_profiles, auto_commit=False):
    """Process all thresholds for a single ISO."""
    print(f"\n{'='*60}")
    print(f"  Processing {iso}")
    if iso == 'CAISO':
        print(f"  (CAISO mode: batch={CAISO_MIX_BATCH}, "
              f"flush={CAISO_CHUNK_LIMIT:,}, streaming writes)")
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
        # Skip if final output already exists (and no partial/resume state)
        out_path = _output_path(iso, threshold)
        progress_path = _progress_path(iso, threshold)
        has_progress = os.path.exists(progress_path)

        if os.path.exists(out_path) and not has_progress:
            existing = pq.read_metadata(out_path).num_rows
            print(f"    {iso} {threshold}%: Already done ({existing:,} solutions) — skipping")
            continue

        if has_progress:
            print(f"    {iso} {threshold}%: Resuming from crash...")

        n_new = process_threshold(
            iso, threshold, demand_arr, supply_matrix,
            coarse_combos, coarse_scores)
        total_new += n_new

        if auto_commit and n_new > 0:
            git_commit_threshold(iso, threshold, n_new)

    elapsed = time.time() - iso_start
    print(f"\n  {iso}: {total_new:,} total new solutions in {elapsed:.1f}s")
    return total_new


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def parse_args(argv):
    """Parse CLI args.

    Usage:
      step1d_storage_refinement.py [ISO ...] [--threshold T1,T2,...] [--force] [--auto-commit]

    Examples:
      step1d_storage_refinement.py ERCOT --threshold 75
      step1d_storage_refinement.py CAISO ERCOT PJM
      step1d_storage_refinement.py --force   # rerun even if output exists
      step1d_storage_refinement.py MISO --auto-commit
    """
    target_isos = []
    target_thresholds = None
    force = False
    auto_commit = False

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == '--force':
            force = True
        elif arg == '--auto-commit':
            auto_commit = True
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

    return target_isos, target_thresholds, force, auto_commit


def main():
    print("=" * 70)
    print("  Step 1D: Storage Refinement Module")
    print("  Fills intermediate storage levels missing from Step 1C")
    print("=" * 70)

    target_isos, thresholds, force, auto_commit = parse_args(sys.argv[1:])
    print(f"  ISOs: {target_isos}")
    print(f"  Thresholds: {thresholds}")
    print(f"  Force rerun: {force}")
    print(f"  Auto-commit: {'ON' if auto_commit else 'OFF'}")

    if force:
        # Remove existing outputs + progress/partial files for target ISOs/thresholds
        for iso in target_isos:
            for t in thresholds:
                for path_fn in [_output_path, _partial_path, _progress_path]:
                    p = path_fn(iso, t)
                    if os.path.exists(p):
                        os.remove(p)
                        print(f"  Removed {os.path.basename(p)}")

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
            n = process_iso(iso, thresholds, demand_data, gen_profiles, auto_commit)
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
