#!/usr/bin/env python3
"""Step 1d: Two-pass storage sweep — coarse global + fine targeted.

Reads the union near-miss mixes from step1c and runs storage dispatch on
each unique (mix, storage_config) exactly ONCE, then bins results to thresholds.

Architecture:
  Pass 1 — Coarse global: sweep ALL near-miss mixes at 1% storage steps.
           Score each (mix, storage) combo once, bin by threshold.
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

# Pass 1: Coarse storage levels (global sweep)
COARSE_BAT4 = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.float64)
COARSE_BAT8 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
COARSE_LDES = np.array([0, 2, 5, 8, 10, 15, 20, 25], dtype=np.float64)
COARSE_H2 = np.array([0, 5, 10, 15, 20, 25], dtype=np.float64)
COARSE_H2_NONE = np.array([0], dtype=np.float64)  # H2 only for >= 95%

# Pass 2: Fine storage resolution
FINE_STEP = 0.05         # 0.05 percentage points
FINE_HALF_WIDTH = 0.5    # ±0.5pp around coarse winner
BOUNDARY_MARGIN_LOW = 2  # 2pp below threshold for boundary identification
BOUNDARY_MARGIN_HIGH = 1 # 1pp above threshold

# Max storage combos per mix in fine sweep (importance-weighted sampling)
MAX_FINE_COMBOS = 1000

# Batch sizes
NM_CHUNK = 10000         # storage cap computation chunk
MAX_MIX_BATCH = 100      # storage evaluation batch
CAISO_MIX_BATCH = 500    # larger for CAISO (amortize overhead)

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

def run_coarse_storage(iso, nm_combos, nm_base_scores, demand_arr, supply_matrix,
                       active_thresholds=None):
    """Sweep coarse storage levels on near-miss mixes.

    Args:
        active_thresholds: list of thresholds to bin into. Defaults to all.

    Returns per-threshold feasible lists:
        results[threshold] = list of (mix_idx, bat4, bat8, ldes, h2, score)
    """
    if active_thresholds is None:
        active_thresholds = STORAGE_THRESHOLDS
    n_mixes = len(nm_combos)
    rtypes = s1.get_resource_types(iso)
    n_res = len(rtypes)
    batch_size = CAISO_MIX_BATCH if iso == 'CAISO' else MAX_MIX_BATCH

    print(f"\n  Pass 1 — Coarse storage sweep: {n_mixes:,} near-miss mixes")

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

    # Compute storage caps for all near-miss mixes (vectorized in chunks)
    print(f"    Computing storage caps...")
    b4_caps = np.empty(n_mixes, dtype=np.float64)
    b8_caps = np.empty(n_mixes, dtype=np.float64)
    l_caps = np.empty(n_mixes, dtype=np.float64)
    surplus_days = np.empty(n_mixes, dtype=np.int64)
    hc_arr = np.empty(n_mixes, dtype=np.int64)
    total_surplus = np.empty(n_mixes, dtype=np.float64)
    demand_total = demand_arr.sum()

    for cs in range(0, n_mixes, NM_CHUNK):
        ce = min(cs + NM_CHUNK, n_mixes)
        chunk_fracs = nm_combos[cs:ce].astype(np.float64) / 100.0
        chunk_supply = chunk_fracs @ supply_matrix
        chunk_n = ce - cs

        cb4, cb8, cl, chc, csd = s1._batch_compute_storage_caps(
            demand_arr, chunk_supply, 1.0, chunk_n,
            batt4_dur, batt8_dur, ldes_dur)
        b4_caps[cs:ce] = cb4
        b8_caps[cs:ce] = cb8
        l_caps[cs:ce] = cl
        hc_arr[cs:ce] = chc
        surplus_days[cs:ce] = csd

        chunk_surplus = np.maximum(chunk_supply - demand_arr[np.newaxis, :], 0.0)
        total_surplus[cs:ce] = chunk_surplus.sum(axis=1)

    # Pre-filter: must have curtailment (fast skip in inner loop)
    has_curtailment = hc_arr.astype(bool)

    # Surplus fraction per mix (curtailment as % of demand).
    # Used to gate per-threshold binning: only bin a mix to threshold t
    # if its curtailment surplus can bridge the gap from base score to t.
    surplus_pct = total_surplus / demand_total

    # Pre-compute per-threshold near-miss floors (avoid repeated calls)
    threshold_floors = {
        t: max(STORAGE_SWEEP_FLOOR, t / 100.0 - get_near_miss_width(t))
        for t in active_thresholds
    }

    # Per-threshold results
    results = {t: [] for t in active_thresholds}

    # Process in batches (vectorized Numba kernel per batch)
    n_batches = (n_mixes + batch_size - 1) // batch_size
    total_feasible = 0

    for batch_start in range(0, n_mixes, batch_size):
        batch_end = min(batch_start + batch_size, n_mixes)
        batch_idx = np.arange(batch_start, batch_end)
        n_batch = len(batch_idx)
        batch_num = batch_start // batch_size + 1

        if batch_num % 10 == 0 or batch_num == 1:
            print(f"\r    Batch {batch_num}/{n_batches} "
                  f"({batch_start:,}/{n_mixes:,}), "
                  f"{total_feasible:,} feasible so far", end="", flush=True)

        # Build supply for batch
        batch_fracs = nm_combos[batch_idx].astype(np.float64) / 100.0
        batch_supply = batch_fracs @ supply_matrix

        # Determine H2 levels (only for thresholds >= 95)
        # We score with H2 for all mixes but only assign H2 results to >= 95%
        h2_arr = COARSE_H2
        n_h2 = len(h2_arr)

        # Run Numba storage screen (vectorized across mixes in batch)
        batch_scores = s1._batch_mixes_storage_screen(
            demand_arr, batch_supply, 1.0, n_batch,
            COARSE_BAT4, COARSE_BAT8, COARSE_LDES,
            len(COARSE_BAT4), len(COARSE_BAT8), len(COARSE_LDES),
            batt_eff, batt8_eff, ldes_eff,
            batt4_dur, batt8_dur, ldes_dur,
            ldes_window, batt8_window,
            h2_arr, n_h2, h2_eff, h2_dur, h2_window)

        # Extract feasible results and bin to thresholds
        n_b4, n_b8, n_l = len(COARSE_BAT4), len(COARSE_BAT8), len(COARSE_LDES)

        for bi in range(n_batch):
            mi = batch_idx[bi]
            b4_max = b4_caps[mi] * 100.0 * 1.1
            b8_max = b8_caps[mi] * 100.0 * 1.1
            l_max = l_caps[mi] * 100.0 * 1.1
            n_sd = int(surplus_days[mi])

            if not has_curtailment[mi]:
                continue

            # Per-mix base score and surplus for threshold gating
            mix_base = nm_base_scores[mi]
            mix_surplus = surplus_pct[mi]

            scores = batch_scores[bi]  # (n_combos,) flat array

            for b4i in range(n_b4):
                bp = COARSE_BAT4[b4i]
                if bp > 0 and (n_sd < MIN_SURPLUS_DAYS_FOR_BATTERY or bp > b4_max):
                    continue
                for b8i in range(n_b8):
                    b8p = COARSE_BAT8[b8i]
                    if b8p > 0 and (n_sd < MIN_SURPLUS_DAYS_FOR_BATTERY or b8p > b8_max):
                        continue
                    for li in range(n_l):
                        lp = COARSE_LDES[li]
                        if lp > 0 and lp > l_max:
                            continue
                        for h2i in range(n_h2):
                            h2p = h2_arr[h2i]
                            if bp == 0 and b8p == 0 and lp == 0 and h2p == 0:
                                continue

                            idx = (b4i * n_b8 * n_l * n_h2 +
                                   b8i * n_l * n_h2 +
                                   li * n_h2 + h2i)
                            score = scores[idx]
                            if score < 0:
                                continue

                            # Bin to thresholds where this mix is a
                            # genuine near-miss candidate:
                            #  1. score >= target  (storage reaches it)
                            #  2. base < target    (actually needs storage)
                            #  3. surplus >= gap   (enough curtailment to bridge)
                            #  4. EITHER within near-miss window (base >= floor)
                            #     OR outlier with surplus >= 1.5x gap
                            #     (high-solar mixes with massive curtailment)
                            for t in active_thresholds:
                                target = t / 100.0
                                if score < target:
                                    continue
                                if mix_base >= target:
                                    continue   # already meets threshold w/o storage
                                gap = target - mix_base
                                if mix_surplus < gap:
                                    continue   # not enough surplus to bridge
                                # Near-miss window OR outlier surplus gate
                                if (mix_base < threshold_floors[t]
                                        and mix_surplus < 1.5 * gap):
                                    continue   # outside window & not an outlier
                                # H2 only for >= 95%
                                if h2p > 0 and t < 95:
                                    continue
                                results[t].append(
                                    (int(mi), float(bp), float(b8p),
                                     float(lp), float(h2p),
                                     round(score * 100, 2)))
                                total_feasible += 1

    print(f"\n    Pass 1 complete: {total_feasible:,} total feasible "
          f"(mix, storage) combos across all thresholds")
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
    """Run two-pass storage sweep for one ISO.

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

    # ── Curtailment bridge filter (replaces dominance filter) ──
    # Keep mixes where raw curtailment surplus can bridge the gap to at least
    # one active threshold.  Check: score + surplus_pct >= min_target.
    t_bridge = time.time()
    demand_total = demand_arr.sum()
    total_surplus = np.empty(n_raw, dtype=np.float64)
    for cs in range(0, n_raw, NM_CHUNK):
        ce = min(cs + NM_CHUNK, n_raw)
        chunk_fracs = nm_combos[cs:ce].astype(np.float64) / 100.0
        chunk_supply = chunk_fracs @ supply_matrix
        chunk_surplus = np.maximum(chunk_supply - demand_arr[np.newaxis, :], 0.0)
        total_surplus[cs:ce] = chunk_surplus.sum(axis=1)
    surplus_pct = total_surplus / demand_total
    min_target = min(active_thresholds) / 100.0
    bridge_mask = (nm_base_scores + surplus_pct) >= min_target
    nm_combos = nm_combos[bridge_mask]
    nm_base_scores = nm_base_scores[bridge_mask]
    print(f"  Bridge filter: {n_raw:,} → {len(nm_combos):,} "
          f"({len(nm_combos)/n_raw*100:.1f}%) in {time.time()-t_bridge:.1f}s")

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
    # PASS 1: Coarse global storage sweep
    # ══════════════════════════════════════════════════════

    coarse_results = None
    if not pass1_done:
        coarse_results = run_coarse_storage(
            iso, nm_combos, nm_base_scores, demand_arr, supply_matrix,
            active_thresholds=active_thresholds)

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
