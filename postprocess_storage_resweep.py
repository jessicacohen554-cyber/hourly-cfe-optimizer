#!/usr/bin/env python3
"""
Post-Process Storage Re-Sweep
==============================
Expands the Physics Feasible Space (PFS) by evaluating additional storage
configurations on near-miss generation mixes using vectorized Numba parallel
kernels.

Pipeline position: Between Step 1 (PFS generator) and Step 2 (EF extraction).
Run after Step 1 to find solutions Step 1's sequential storage sweep may miss.

Key optimizations over Step 1's inline storage sweep:
  - @njit(parallel=True) with prange: evaluates N mixes across all CPU cores
  - Batch evaluation: all near-miss mixes at a single (procurement, storage) config
  - Early stopping per mix: once feasible at min procurement, skip higher levels
  - Wider near-miss window (25% vs Step 1's 15%)

Input:  data/physics_cache_v4.parquet (existing PFS from Step 1)
Output: data/resweep_checkpoints/<ISO>_<threshold>.parquet (one file per pair)
        Step 2 reads PFS + all checkpoint files together.

Usage:
  python3 postprocess_storage_resweep.py                   # Full sweep
  python3 postprocess_storage_resweep.py --iso PJM         # Single ISO
  python3 postprocess_storage_resweep.py --threshold 90    # Single threshold
  python3 postprocess_storage_resweep.py --dry-run         # Report only, no write
"""

import os
import sys
import time
import argparse
import numpy as np

try:
    from numba import njit, prange
    HAS_NUMBA = True
    print("  Numba available — parallel scoring enabled")
except ImportError:
    raise RuntimeError("Numba required for parallel scoring. Install: pip install numba")

import json
import shutil
import subprocess

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
PFS_PATH = os.path.join(DATA_DIR, 'physics_cache_v4.parquet')
CHECKPOINT_DIR = os.path.join(DATA_DIR, 'resweep_checkpoints')
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'resweep_progress.parquet')

H = 8760
ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO']
RESOURCE_TYPES = ['clean_firm', 'solar', 'wind', 'hydro']
HYDRO_CAPS = {'CAISO': 9.5, 'ERCOT': 0.1, 'PJM': 1.8, 'NYISO': 15.9, 'NEISO': 4.4}
THRESHOLDS = [50, 60, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 100]

PROCUREMENT_BOUNDS = {
    50: (50, 150), 60: (60, 150), 70: (70, 175), 75: (75, 200),
    80: (80, 200), 85: (85, 225), 87.5: (87, 250), 90: (90, 250),
    92.5: (92, 250), 95: (95, 250), 97.5: (100, 250), 99: (100, 250),
    100: (100, 350),
}

# Storage parameters (must match Step 1)
BATTERY_EFFICIENCY = 0.85
BATTERY_DURATION_HOURS = 4
BATTERY8_EFFICIENCY = 0.85
BATTERY8_DURATION_HOURS = 8
LDES_EFFICIENCY = 0.50
LDES_DURATION_HOURS = 100
LDES_WINDOW_DAYS = 7

# Storage levels (same grid as Step 1)
BATT_LEVELS = [0, 2, 5, 8, 10, 15, 20]
BATT8_LEVELS = [0, 2, 5, 8, 10, 15, 20]
LDES_LEVELS = [0, 2, 5, 8, 10, 15, 20]

# Near-miss window (wider than Step 1's 0.15 to catch more mixes)
NEAR_MISS_WINDOW = 0.25

# Partial checkpoint: save every N storage combos within an (ISO, threshold) pair
PARTIAL_SAVE_COMBO_INTERVAL = 50


# ══════════════════════════════════════════════════════════════════════════════
# NUMBA KERNELS — identical dispatch logic to Step 1, with parallel wrapper
# ══════════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def _score_with_all_storage(demand, supply_row, procurement,
                            batt_cap, batt_pow, batt_eff,
                            batt8_cap, batt8_pow, batt8_eff,
                            ldes_cap, ldes_pow, ldes_eff,
                            ldes_window_hours):
    """Score a single mix with battery4 + battery8 + LDES dispatch.

    Dispatch order: battery4 → battery8 → LDES (sequential, each reduces residuals).
    Returns total matched energy (normalized, ≈ fraction of annual demand met).
    Identical to Step 1's _score_with_all_storage.
    """
    supply = np.empty(8760)
    surplus = np.empty(8760)
    gap = np.empty(8760)
    for h in range(8760):
        s = procurement * supply_row[h]
        supply[h] = s
        d = demand[h]
        if s > d:
            surplus[h] = s - d
            gap[h] = 0.0
        else:
            surplus[h] = 0.0
            gap[h] = d - s

    base_matched = 0.0
    for h in range(8760):
        base_matched += min(demand[h], supply[h])

    # Phase 1: Battery 4hr daily cycle
    batt_dispatched = 0.0
    residual_surplus = np.copy(surplus)
    residual_gap = np.copy(gap)

    if batt_cap > 0:
        for day in range(365):
            ds = day * 24
            stored = 0.0
            for h in range(24):
                s = residual_surplus[ds + h]
                if s > 0 and stored < batt_cap:
                    charge = s
                    if charge > batt_pow:
                        charge = batt_pow
                    remaining = batt_cap - stored
                    if charge > remaining:
                        charge = remaining
                    stored += charge
                    residual_surplus[ds + h] -= charge
            available = stored * batt_eff
            for h in range(24):
                g = residual_gap[ds + h]
                if g > 0 and available > 0:
                    discharge = g
                    if discharge > batt_pow:
                        discharge = batt_pow
                    if discharge > available:
                        discharge = available
                    batt_dispatched += discharge
                    available -= discharge
                    residual_gap[ds + h] -= discharge

    # Phase 2: Battery 8hr daily cycle on post-4hr residual
    batt8_dispatched = 0.0
    if batt8_cap > 0:
        for day in range(365):
            ds = day * 24
            stored = 0.0
            for h in range(24):
                s = residual_surplus[ds + h]
                if s > 0 and stored < batt8_cap:
                    charge = s
                    if charge > batt8_pow:
                        charge = batt8_pow
                    remaining = batt8_cap - stored
                    if charge > remaining:
                        charge = remaining
                    stored += charge
                    residual_surplus[ds + h] -= charge
            available = stored * batt8_eff
            for h in range(24):
                g = residual_gap[ds + h]
                if g > 0 and available > 0:
                    discharge = g
                    if discharge > batt8_pow:
                        discharge = batt8_pow
                    if discharge > available:
                        discharge = available
                    batt8_dispatched += discharge
                    available -= discharge
                    residual_gap[ds + h] -= discharge

    # Phase 3: LDES multi-day rolling window on post-battery residual
    ldes_dispatched = 0.0
    if ldes_cap > 0:
        soc = 0.0
        n_windows = (8760 + ldes_window_hours - 1) // ldes_window_hours
        for w in range(n_windows):
            ws = w * ldes_window_hours
            we = ws + ldes_window_hours
            if we > 8760:
                we = 8760
            for h in range(ws, we):
                s = residual_surplus[h]
                if s > 0 and soc < ldes_cap:
                    charge = s
                    if charge > ldes_pow:
                        charge = ldes_pow
                    remaining = ldes_cap - soc
                    if charge > remaining:
                        charge = remaining
                    soc += charge
            for h in range(ws, we):
                g = residual_gap[h]
                if g > 0 and soc > 0:
                    available_e = soc * ldes_eff
                    discharge = g
                    if discharge > ldes_pow:
                        discharge = ldes_pow
                    if discharge > available_e:
                        discharge = available_e
                    ldes_dispatched += discharge
                    soc -= discharge / ldes_eff

    return base_matched + batt_dispatched + batt8_dispatched + ldes_dispatched


@njit(parallel=True, cache=True)
def batch_score_storage(demand, supply_rows, procurement, N,
                        batt_cap, batt_pow, batt_eff,
                        batt8_cap, batt8_pow, batt8_eff,
                        ldes_cap, ldes_pow, ldes_eff,
                        ldes_window_hours):
    """Evaluate N mixes in parallel at a single (procurement, storage) config.

    Uses Numba prange to distribute mix evaluation across CPU cores.
    Each mix is independent — no data dependencies between iterations.

    Args:
        demand: (8760,) normalized demand profile
        supply_rows: (N, 8760) pre-computed supply profiles per mix
        procurement: scalar procurement fraction (same for all mixes)
        N: number of mixes
        batt_cap..ldes_window_hours: storage parameters (scalars)

    Returns:
        (N,) scores — total matched energy fraction per mix
    """
    scores = np.empty(N, dtype=np.float64)
    for i in prange(N):
        scores[i] = _score_with_all_storage(
            demand, supply_rows[i], procurement,
            batt_cap, batt_pow, batt_eff,
            batt8_cap, batt8_pow, batt8_eff,
            ldes_cap, ldes_pow, ldes_eff,
            ldes_window_hours)
    return scores


@njit(parallel=True, cache=True)
def batch_score_no_storage(demand, supply_rows, procurement, N):
    """Vectorized no-storage scoring using same metric as storage kernels.

    Uses sum(min(demand, supply)) — total matched energy, consistent with
    _score_with_all_storage's base_matched computation.
    """
    scores = np.empty(N, dtype=np.float64)
    for i in prange(N):
        total = 0.0
        for h in range(8760):
            s = procurement * supply_rows[i, h]
            d = demand[h]
            if s < d:
                total += s
            else:
                total += d
        scores[i] = total
    return scores


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING — imports from Step 1 for consistency
# ══════════════════════════════════════════════════════════════════════════════

def _load_step1_functions():
    """Import data loading functions from Step 1."""
    sys.path.insert(0, SCRIPT_DIR)
    from step1_pfs_generator import (
        load_data, get_supply_profiles, prepare_numpy_profiles,
        generate_4d_combos, get_seed_combos,
    )
    return load_data, get_supply_profiles, prepare_numpy_profiles, \
           generate_4d_combos, get_seed_combos


# ══════════════════════════════════════════════════════════════════════════════
# STORAGE COMBO GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_storage_combos():
    """Generate all non-zero storage configurations to sweep.

    Returns list of (bp, b8p, lp) tuples.
    Excludes (0, 0, 0) — no-storage case handled separately.
    """
    combos = []
    for bp in BATT_LEVELS:
        for b8p in BATT8_LEVELS:
            for lp in LDES_LEVELS:
                if bp == 0 and b8p == 0 and lp == 0:
                    continue
                combos.append((bp, b8p, lp))
    return combos


# ══════════════════════════════════════════════════════════════════════════════
# CORE RE-SWEEP LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def process_iso_threshold(iso, threshold, demand_arr, supply_matrix,
                          all_mixes, supply_rows, existing_keys_set,
                          combo_start_idx=0, initial_rows=None,
                          save_partial_fn=None):
    """Re-sweep storage for near-miss mixes at a single (ISO, threshold).

    Args:
        demand_arr: (8760,) normalized demand
        supply_matrix: (4, 8760) resource profiles
        all_mixes: (N, 4) gen mix allocations (percentage points)
        supply_rows: (N, 8760) pre-computed supply per mix = (mix/100) @ supply_matrix
        existing_keys_set: set of (cf, sol, wnd, hyd, proc, bp, b8p, lp) already in PFS
        combo_start_idx: resume from this storage combo index (for partial checkpoint)
        initial_rows: pre-loaded rows from partial checkpoint
        save_partial_fn: callback(combo_idx, new_rows) for periodic partial saves

    Returns:
        list of new solution dicts
    """
    target = threshold / 100.0
    proc_min, proc_max = PROCUREMENT_BOUNDS.get(threshold, (70, 200))
    N = len(all_mixes)

    # Adaptive procurement step
    proc_range = proc_max - proc_min
    if proc_range > 200:
        proc_step = 10
    elif proc_range > 100:
        proc_step = 5
    else:
        proc_step = 2
    proc_levels = list(range(proc_min, proc_max + 1, proc_step))
    if proc_max not in proc_levels:
        proc_levels.append(proc_max)

    # Step 1: Find near-miss mixes (best no-storage score within NEAR_MISS_WINDOW)
    best_no_storage = np.zeros(N, dtype=np.float64)
    for proc in proc_levels:
        pf = proc / 100.0
        scores = batch_score_no_storage(demand_arr, supply_rows, pf, N)
        best_no_storage = np.maximum(best_no_storage, scores)

    # Near-miss: within window but below target
    near_mask = (best_no_storage >= target - NEAR_MISS_WINDOW) & (best_no_storage < target)
    near_idx = np.where(near_mask)[0]

    if len(near_idx) == 0:
        return []

    near_supply = supply_rows[near_idx].copy()  # (M, 8760)
    near_mixes = all_mixes[near_idx]
    M = len(near_idx)

    # Step 2: Sweep storage combos with Numba parallel batch evaluation
    storage_combos = generate_storage_combos()
    new_rows = list(initial_rows) if initial_rows else []
    ldes_window_hours = LDES_WINDOW_DAYS * 24

    for combo_idx, (bp, b8p, lp) in enumerate(storage_combos):
        # Skip combos already processed in a previous partial checkpoint
        if combo_idx < combo_start_idx:
            continue
        batt_cap = bp / 100.0
        batt_pow = batt_cap / BATTERY_DURATION_HOURS if bp > 0 else 0.0
        batt8_cap = b8p / 100.0
        batt8_pow = batt8_cap / BATTERY8_DURATION_HOURS if b8p > 0 else 0.0
        ldes_cap = lp / 100.0
        ldes_pow = ldes_cap / LDES_DURATION_HOURS if lp > 0 else 0.0

        # Track which mixes have already been found feasible (for early stop)
        found_mask = np.zeros(M, dtype=np.bool_)

        for proc in proc_levels:
            if found_mask.all():
                break

            pf = proc / 100.0
            scores = batch_score_storage(
                demand_arr, near_supply, pf, M,
                batt_cap, batt_pow, BATTERY_EFFICIENCY,
                batt8_cap, batt8_pow, BATTERY8_EFFICIENCY,
                ldes_cap, ldes_pow, LDES_EFFICIENCY,
                ldes_window_hours)

            # Newly feasible at this procurement level
            newly_feasible = (scores >= target) & ~found_mask
            found_mask |= (scores >= target)

            for j in np.where(newly_feasible)[0]:
                mix = near_mixes[j]
                key = (int(mix[0]), int(mix[1]), int(mix[2]), int(mix[3]),
                       proc, bp, b8p, lp)
                if key not in existing_keys_set:
                    new_rows.append({
                        'iso': iso,
                        'threshold': float(threshold),
                        'clean_firm': int(mix[0]),
                        'solar': int(mix[1]),
                        'wind': int(mix[2]),
                        'hydro': int(mix[3]),
                        'procurement_pct': proc,
                        'battery_dispatch_pct': bp,
                        'battery8_dispatch_pct': b8p,
                        'ldes_dispatch_pct': lp,
                        'hourly_match_score': round(float(scores[j]) * 100, 2),
                        'pareto_type': '',
                    })
                    existing_keys_set.add(key)

        # Periodic partial checkpoint: save every N combos within this pair
        if save_partial_fn and (combo_idx + 1) % PARTIAL_SAVE_COMBO_INTERVAL == 0:
            save_partial_fn(combo_idx, new_rows)

    return new_rows


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINTING — parquet-based, per ISO×threshold
# ══════════════════════════════════════════════════════════════════════════════

def _checkpoint_path(iso, threshold):
    """Per-ISO/threshold checkpoint file path.

    Uses float representation for consistency (e.g., 50 -> '50_0', 87.5 -> '87_5').
    """
    thr_str = str(float(threshold)).replace('.', '_')
    return os.path.join(CHECKPOINT_DIR, f'{iso}_{thr_str}.parquet')


def load_completed_checkpoints():
    """Scan checkpoint directory for completed (iso, threshold) pairs.

    Returns:
        completed: set of (iso, threshold) tuples already processed
    """
    completed = set()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    for fname in os.listdir(CHECKPOINT_DIR):
        if not fname.endswith('.parquet'):
            continue
        stem = fname.replace('.parquet', '')
        # Match against known ISOs for reliable parsing
        for known_iso in ISOS:
            prefix = known_iso + '_'
            if stem.startswith(prefix):
                thr_str = stem[len(prefix):].replace('_', '.')
                try:
                    threshold = float(thr_str)
                    completed.add((known_iso, threshold))
                except ValueError:
                    pass
                break
    if completed:
        print(f"  Checkpoint: {len(completed)} (iso, threshold) pairs already completed")
    return completed


def save_threshold_checkpoint(iso, threshold, new_rows_list):
    """Save results for a single ISO×threshold to its own parquet file.

    Atomic write via temp file — safe against interruption.
    """
    if not new_rows_list:
        # Write empty marker file so we know this pair was processed (0 new solutions)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        path = _checkpoint_path(iso, threshold)
        # Create empty parquet with correct schema
        empty_df = pd.DataFrame(columns=[
            'iso', 'threshold', 'clean_firm', 'solar', 'wind', 'hydro',
            'procurement_pct', 'battery_dispatch_pct', 'battery8_dispatch_pct',
            'ldes_dispatch_pct', 'hourly_match_score', 'pareto_type'])
        empty_table = pa.Table.from_pandas(empty_df, preserve_index=False)
        pq.write_table(empty_table, path, compression='snappy')
        return

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    new_df = pd.DataFrame(new_rows_list)
    new_table = pa.Table.from_pandas(new_df, preserve_index=False)
    path = _checkpoint_path(iso, threshold)
    tmp_path = path + '.tmp'
    pq.write_table(new_table, tmp_path, compression='snappy')
    os.replace(tmp_path, path)


def _partial_checkpoint_dir(iso, threshold):
    """Directory for partial (in-progress) checkpoint within a single (iso, threshold) pair."""
    thr_str = str(float(threshold)).replace('.', '_')
    return os.path.join(CHECKPOINT_DIR, f'partial_{iso}_{thr_str}')


def save_partial_checkpoint(iso, threshold, combo_idx, new_rows):
    """Save intermediate results during processing of a single (iso, threshold) pair.

    Stores the last-completed combo index and all new rows found so far.
    On resume, processing starts from combo_idx + 1 with these rows pre-loaded.
    Atomic write via temp file.
    """
    partial_dir = _partial_checkpoint_dir(iso, threshold)
    os.makedirs(partial_dir, exist_ok=True)

    # Save metadata (combo index, row count)
    meta_path = os.path.join(partial_dir, 'meta.json')
    meta_tmp = meta_path + '.tmp'
    with open(meta_tmp, 'w') as f:
        json.dump({'combo_idx': combo_idx, 'n_rows': len(new_rows)}, f)
    os.replace(meta_tmp, meta_path)

    # Save rows (if any)
    rows_path = os.path.join(partial_dir, 'rows.parquet')
    if new_rows:
        df = pd.DataFrame(new_rows)
        table = pa.Table.from_pandas(df, preserve_index=False)
        rows_tmp = rows_path + '.tmp'
        pq.write_table(table, rows_tmp, compression='snappy')
        os.replace(rows_tmp, rows_path)
    elif os.path.exists(rows_path):
        os.remove(rows_path)


def load_partial_checkpoint(iso, threshold):
    """Load partial checkpoint for an in-progress (iso, threshold) pair.

    Returns:
        (combo_start_idx, partial_rows): combo index to resume from and pre-loaded rows.
        Returns (0, []) if no partial checkpoint exists.
    """
    partial_dir = _partial_checkpoint_dir(iso, threshold)
    meta_path = os.path.join(partial_dir, 'meta.json')

    if not os.path.exists(meta_path):
        return 0, []

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    combo_idx = meta['combo_idx']
    rows_path = os.path.join(partial_dir, 'rows.parquet')
    if os.path.exists(rows_path):
        df = pd.read_parquet(rows_path)
        partial_rows = df.to_dict('records')
    else:
        partial_rows = []

    print(f"      Resuming from combo {combo_idx + 1}/342 with {len(partial_rows)} rows cached")
    return combo_idx + 1, partial_rows  # Start from the NEXT combo


def clear_partial_checkpoint(iso, threshold):
    """Remove partial checkpoint after pair is fully completed."""
    partial_dir = _partial_checkpoint_dir(iso, threshold)
    if os.path.exists(partial_dir):
        shutil.rmtree(partial_dir)


def git_commit_checkpoint(iso, threshold):
    """Git add and commit the checkpoint file for this ISO/threshold pair."""
    path = _checkpoint_path(iso, threshold)
    rel_path = os.path.relpath(path, SCRIPT_DIR)
    try:
        subprocess.run(
            ['git', 'add', rel_path], cwd=SCRIPT_DIR,
            check=True, capture_output=True, timeout=30)
        msg = f"Bank resweep checkpoint: {iso} {threshold}%"
        subprocess.run(
            ['git', 'commit', '-m', msg], cwd=SCRIPT_DIR,
            check=True, capture_output=True, timeout=30)
        print(f"      [git] Committed {iso} {threshold}%")
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode().strip() if e.stderr else str(e)
        print(f"      [git] Warning: commit failed — {stderr}")
    except Exception as e:
        print(f"      [git] Warning: {e}")


def load_checkpoint_solutions(iso, threshold):
    """Load existing resweep solutions from checkpoint file for deduplication.

    Returns a DataFrame of existing solutions for this (iso, threshold) pair,
    or an empty DataFrame if no checkpoint exists.
    """
    path = _checkpoint_path(iso, threshold)
    if os.path.exists(path):
        try:
            t = pq.read_table(path)
            if t.num_rows > 0:
                return t.to_pandas()
        except Exception as e:
            print(f"  WARNING: Could not read checkpoint {path}: {e}")
    return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Post-process storage re-sweep')
    parser.add_argument('--iso', type=str, help='Process single ISO (e.g., PJM)')
    parser.add_argument('--threshold', type=float, help='Process single threshold (e.g., 90)')
    parser.add_argument('--dry-run', action='store_true', help='Report only, do not write')
    parser.add_argument('--no-git', action='store_true', help='Skip git commits (for parallel runs)')
    args = parser.parse_args()

    print("=" * 70)
    print("  POST-PROCESS STORAGE RE-SWEEP")
    print("  Vectorized Numba parallel | Wider near-miss window")
    print("=" * 70)

    total_start = time.time()

    # Load data (reuse Step 1's multi-year averaged profiles)
    load_data, get_supply_profiles, prepare_numpy_profiles, \
        generate_4d_combos, get_seed_combos = _load_step1_functions()

    demand_data, gen_profiles, _, _ = load_data()

    # Determine ISOs and thresholds to process
    isos = [args.iso] if args.iso else ISOS
    thresholds = [args.threshold] if args.threshold else THRESHOLDS

    # JIT warmup: compile Numba kernels with a tiny batch
    print("\n  Warming up Numba kernels...")
    warmup_start = time.time()
    _dummy_d = np.ones(8760, dtype=np.float64) / 8760
    _dummy_s = np.ones((2, 8760), dtype=np.float64) / 8760
    _ = batch_score_storage(_dummy_d, _dummy_s, 1.0, 2,
                            0.01, 0.01, 0.85, 0.0, 0.0, 0.85,
                            0.0, 0.0, 0.5, 168)
    _ = batch_score_no_storage(_dummy_d, _dummy_s, 1.0, 2)
    print(f"    Compiled in {time.time() - warmup_start:.1f}s\n")

    # Load checkpoint (resume from previous interrupted run)
    completed_pairs = load_completed_checkpoints()
    summary = {}

    for iso in isos:
        iso_start = time.time()
        print(f"\n  {iso}: Loading profiles and building mix grid...")

        # Load demand/supply profiles
        demand_norm = demand_data[iso]['normalized']
        supply_profiles = get_supply_profiles(iso, gen_profiles)
        demand_arr, supply_matrix = prepare_numpy_profiles(demand_norm, supply_profiles)
        hydro_cap = HYDRO_CAPS[iso]

        # Build full gen mix grid (5% step + seeds)
        grid_mixes = generate_4d_combos(hydro_cap, step=5)
        seeds = get_seed_combos(hydro_cap)
        if len(seeds) > 0:
            grid_mixes = np.vstack([grid_mixes, seeds])
            grid_mixes = np.unique(grid_mixes, axis=0)

        # Also include unique mixes from PFS + checkpoints (captures 1% refinement mixes)
        iso_table = pq.read_table(PFS_PATH, filters=[('iso', '==', iso)],
                                  columns=['clean_firm', 'solar', 'wind', 'hydro',
                                           'threshold', 'procurement_pct',
                                           'battery_dispatch_pct', 'battery8_dispatch_pct',
                                           'ldes_dispatch_pct', 'hourly_match_score'])
        iso_df = iso_table.to_pandas()

        # Load checkpoint mixes too (they may have mixes not in original PFS)
        ckpt_mixes_list = []
        for fname in os.listdir(CHECKPOINT_DIR):
            if fname.endswith('.parquet') and fname.startswith(iso + '_'):
                ckpt_path = os.path.join(CHECKPOINT_DIR, fname)
                try:
                    ct = pq.read_table(ckpt_path,
                                       columns=['clean_firm', 'solar', 'wind', 'hydro'])
                    if ct.num_rows > 0:
                        ckpt_mixes_list.append(ct.to_pandas()[['clean_firm', 'solar', 'wind', 'hydro']].values)
                except Exception:
                    pass

        pfs_mixes = iso_df[['clean_firm', 'solar', 'wind', 'hydro']].drop_duplicates().values.astype(np.float64)
        all_mixes = np.vstack([grid_mixes, pfs_mixes])
        for cm in ckpt_mixes_list:
            all_mixes = np.vstack([all_mixes, cm.astype(np.float64)])
        all_mixes = np.unique(all_mixes, axis=0)
        N = len(all_mixes)

        # Pre-compute supply_rows: (N, 8760)
        mix_fracs = all_mixes / 100.0
        supply_rows = mix_fracs @ supply_matrix
        print(f"    {N:,} unique gen mixes, supply_rows shape {supply_rows.shape}")

        iso_new = 0
        for threshold in thresholds:
            if threshold not in THRESHOLDS:
                continue

            # Skip if already completed in previous run
            if (iso, float(threshold)) in completed_pairs:
                print(f"    {iso} {threshold:>5}%: skipped (checkpoint)")
                continue

            thr_start = time.time()

            # Check for partial checkpoint (interrupted mid-pair)
            combo_start_idx, partial_rows = load_partial_checkpoint(iso, threshold)

            # Build existing solution keys for deduplication
            # Include both PFS solutions and any prior checkpoint solutions
            thr_df = iso_df[iso_df['threshold'] == threshold]
            ckpt_df = load_checkpoint_solutions(iso, threshold)
            if not ckpt_df.empty:
                dedup_df = pd.concat([thr_df, ckpt_df], ignore_index=True)
            else:
                dedup_df = thr_df
            existing_keys = set(zip(
                dedup_df['clean_firm'].astype(int),
                dedup_df['solar'].astype(int),
                dedup_df['wind'].astype(int),
                dedup_df['hydro'].astype(int),
                dedup_df['procurement_pct'].astype(int),
                dedup_df['battery_dispatch_pct'].astype(int),
                dedup_df['battery8_dispatch_pct'].astype(int),
                dedup_df['ldes_dispatch_pct'].astype(int),
            ))

            # Add partial rows' keys to existing set (prevent duplicates on resume)
            for row in partial_rows:
                key = (row['clean_firm'], row['solar'], row['wind'], row['hydro'],
                       row['procurement_pct'], row['battery_dispatch_pct'],
                       row['battery8_dispatch_pct'], row['ldes_dispatch_pct'])
                existing_keys.add(key)

            # Partial save callback for periodic mid-pair checkpointing
            def _save_partial(combo_idx, rows, _iso=iso, _thr=threshold):
                save_partial_checkpoint(_iso, _thr, combo_idx, rows)

            new_rows = process_iso_threshold(
                iso, threshold, demand_arr, supply_matrix,
                all_mixes, supply_rows, existing_keys,
                combo_start_idx=combo_start_idx,
                initial_rows=partial_rows,
                save_partial_fn=None if args.dry_run else _save_partial)

            # Checkpoint immediately after each threshold completes
            if not args.dry_run:
                save_threshold_checkpoint(iso, threshold, new_rows)
                clear_partial_checkpoint(iso, threshold)
                if not args.no_git:
                    git_commit_checkpoint(iso, threshold)

            iso_new += len(new_rows)
            elapsed = time.time() - thr_start
            print(f"    {iso} {threshold:>5}%: {len(new_rows):>6,} new solutions ({elapsed:.1f}s)"
                  + (" [saved+committed]" if not args.dry_run else ""))

        iso_elapsed = time.time() - iso_start
        summary[iso] = iso_new
        print(f"  {iso} done: {iso_new:,} new solutions in {iso_elapsed:.1f}s")

        # Free memory
        del iso_df, iso_table, supply_rows, all_mixes, pfs_mixes

    # Summary
    total_new = sum(summary.values())
    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  SUMMARY: {total_new:,} new solutions found in {total_elapsed:.1f}s")
    for iso, count in summary.items():
        print(f"    {iso}: {count:,}")
    print(f"{'='*70}")

    # Checkpoints stay as separate files — Step 2 reads PFS + checkpoints together
    if not args.dry_run:
        total_ckpt = 0
        for f in os.listdir(CHECKPOINT_DIR):
            if f.endswith('.parquet'):
                total_ckpt += pq.read_metadata(os.path.join(CHECKPOINT_DIR, f)).num_rows
        print(f"\n  Resweep checkpoints: {total_ckpt:,} total rows in {CHECKPOINT_DIR}/")
        print(f"  Step 2 will read PFS + checkpoints together")
    else:
        print(f"\n  Dry run — {total_new:,} new rows would be checkpointed")


if __name__ == '__main__':
    main()
