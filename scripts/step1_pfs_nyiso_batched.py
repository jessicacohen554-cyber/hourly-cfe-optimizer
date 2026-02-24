#!/usr/bin/env python3
"""Step 1 PFS — NYISO Batched Runner

Runs NYISO thresholds in fixed-size batches of outer-loop mixes, producing
separately tagged parquet files per batch. This avoids the need for monolithic
runs and gives clear visibility into progress.

Output naming:  data/step1_raw_pfs_parquets/NYISO_t{XX}_raw_pfs_b{N}.parquet
Manifest:       data/step1_raw_pfs_parquets/NYISO_batch_manifest.json

The manifest tracks per-threshold state so each invocation knows:
  - Which batch number to produce next
  - The mix cursor (where the last batch left off)
  - Cumulative candidate count (for slicing new-only candidates)
  - Whether the threshold is fully complete

Usage:
  # Run the next batch for NYISO 100% (auto-detects batch number from manifest)
  python scripts/step1_pfs_nyiso_batched.py --threshold 100

  # Override batch size (default 10000)
  python scripts/step1_pfs_nyiso_batched.py --threshold 100 --batch-size 5000

  # Run with a runtime cap as a safety net
  python scripts/step1_pfs_nyiso_batched.py --threshold 100 --max-runtime-minutes 330
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# Ensure step1_pfs_generator is importable from scripts/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import step1_pfs_generator as s1

ISO = 'NYISO'
MANIFEST_FILENAME = 'NYISO_batch_manifest.json'


def _manifest_path():
    return os.path.join(s1.STEP1_RAW_PFS_PARQUET_DIR, MANIFEST_FILENAME)


def _load_manifest():
    """Load or initialize the NYISO batch manifest."""
    path = _manifest_path()
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {'iso': ISO, 'thresholds': {}}


def _save_manifest(manifest):
    """Save the manifest atomically (write tmp + rename)."""
    path = _manifest_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp_path, path)


def _threshold_key(threshold):
    """Normalize threshold to a consistent string key for the manifest."""
    t = float(threshold)
    return str(int(t)) if t == int(t) else str(t)


def _batch_parquet_path(threshold, batch_num):
    """Path for a batch-tagged parquet: NYISO_t{XX}_raw_pfs_b{N}.parquet."""
    t_str = s1._normalize_threshold_str(threshold)
    return os.path.join(
        s1.STEP1_RAW_PFS_PARQUET_DIR,
        f'{ISO}_t{t_str}_raw_pfs_b{batch_num}.parquet',
    )


def _save_batch_parquet(threshold, batch_num, candidates):
    """Write a batch of candidates to a tagged parquet file."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not candidates:
        print(f"  WARNING: Batch {batch_num} has 0 candidates — skipping parquet write")
        return None

    rows = []
    for c in candidates:
        rows.append({
            'iso': ISO,
            'threshold': float(threshold),
            'clean_firm': c['resource_mix']['clean_firm'],
            'solar': c['resource_mix']['solar'],
            'wind': c['resource_mix']['wind'],
            'hydro': c['resource_mix']['hydro'],
            'procurement_pct': c['procurement_pct'],
            'battery_dispatch_pct': c['battery_dispatch_pct'],
            'battery8_dispatch_pct': c.get('battery8_dispatch_pct', 0),
            'ldes_dispatch_pct': c['ldes_dispatch_pct'],
            'hourly_match_score': c['hourly_match_score'],
            'pareto_type': c.get('pareto_type', ''),
        })

    table = s1._rows_to_table(rows)
    if table is None:
        return None

    path = _batch_parquet_path(threshold, batch_num)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(table, path, compression='snappy')
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run NYISO Step 1 PFS in fixed-size batches with manifest tracking.",
    )
    parser.add_argument(
        "--threshold",
        required=True,
        type=float,
        help="Threshold percentage (e.g., 100, 95, 92.5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Number of outer-loop mixes per batch (default: 10000).",
    )
    parser.add_argument(
        "--max-runtime-minutes",
        type=float,
        default=0,
        help="Runtime safety cap in minutes. 0 = no cap (default).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    threshold = args.threshold
    batch_size = args.batch_size
    max_runtime_minutes = args.max_runtime_minutes

    # Validate threshold
    matched_threshold = None
    for t in s1.THRESHOLDS:
        if abs(float(t) - threshold) < 1e-9:
            matched_threshold = t
            break
    if matched_threshold is None:
        print(f"ERROR: Unsupported threshold {threshold}. "
              f"Valid: {', '.join(str(t) for t in s1.THRESHOLDS)}")
        sys.exit(1)
    threshold = matched_threshold

    # Load manifest
    manifest = _load_manifest()
    t_key = _threshold_key(threshold)
    t_state = manifest['thresholds'].get(t_key, {
        'batch_size': batch_size,
        'batches_completed': 0,
        'cumulative_candidates': 0,
        'completed': False,
    })

    if t_state.get('completed', False):
        print(f"NYISO {threshold}% is already fully completed "
              f"({t_state['batches_completed']} batches). Nothing to do.")
        sys.exit(0)

    batch_num = t_state['batches_completed'] + 1
    prev_candidate_count = t_state.get('cumulative_candidates', 0)

    max_runtime_seconds = None
    if max_runtime_minutes > 0:
        max_runtime_seconds = max_runtime_minutes * 60

    print("=" * 70)
    print(f"  Step 1 PFS — NYISO Batched Runner")
    print(f"  Threshold: {threshold}%")
    print(f"  Batch: #{batch_num}  (batch_size={batch_size:,})")
    print(f"  Previous batches: {t_state['batches_completed']}")
    print(f"  Cumulative candidates so far: {prev_candidate_count:,}")
    print(f"  Numba: {'enabled' if s1.HAS_NUMBA else 'disabled (NumPy fallback)'}")
    if max_runtime_minutes > 0:
        print(f"  Runtime cap: {max_runtime_minutes:.0f} minutes")
    print("=" * 70)

    # Load data
    demand_data, gen_profiles, _, _ = s1.load_data()

    # JIT warmup
    if s1.HAS_NUMBA:
        print("  Warming up Numba JIT...")
        H = s1.H
        dd = np.ones(H) / H
        ds = np.ones(H) / H
        ds2 = np.ones((2, H)) / H
        s1._score_with_all_storage(dd, ds, 1.0, 0.01, 0.0025, 0.85,
                                   0.01, 0.00125, 0.85, 0.01, 0.0001, 0.50, 168)
        s1._batch_score_no_storage(dd, ds2, 1.0, 2)
        s1._batch_score_storage(dd, ds2, 1.0, 2,
                                0.01, 0.0025, 0.85, 0.01, 0.00125, 0.85,
                                0.01, 0.0001, 0.50, 168)
        s1._compute_storage_caps(dd, ds, 1.0, 4, 8, 100)
        dl = np.array([0.0, 0.1], dtype=np.float64)
        s1._batch_storage_scores(dd, ds, 1.0, dl, dl, dl, 2, 2, 2,
                                 0.85, 0.85, 0.50, 4, 8, 100, 168, 48)
        s1._batch_mixes_storage_screen(dd, ds2, 1.0, 2, dl, dl, dl, 2, 2, 2,
                                       0.85, 0.85, 0.50, 4, 8, 100, 168, 48)
        print("  JIT compilation complete")

    # Prepare ISO data
    demand_norm = demand_data[ISO]["normalized"]
    supply_profiles = s1.get_supply_profiles(ISO, gen_profiles)
    demand_arr, supply_matrix = s1.prepare_numpy_profiles(demand_norm, supply_profiles)
    hydro_cap = s1.HYDRO_CAPS[ISO]

    # Seed cross_feasible_mixes from existing completed parquets + batch parquets
    cross_feasible = set()
    if os.path.exists(s1.STEP1_RAW_PFS_PARQUET_DIR):
        import pyarrow.parquet as pq
        import re
        for fname in os.listdir(s1.STEP1_RAW_PFS_PARQUET_DIR):
            # Match both NYISO_tXX_raw_pfs.parquet and NYISO_tXX_raw_pfs_bN.parquet
            if fname.startswith(f"{ISO}_t") and fname.endswith('.parquet'):
                if re.match(r'^NYISO_t[\d.]+_raw_pfs(_b\d+)?\.parquet$', fname):
                    try:
                        t = pq.read_table(os.path.join(s1.STEP1_RAW_PFS_PARQUET_DIR, fname))
                        for row in t.to_pandas()[["clean_firm", "solar", "wind", "hydro"]].drop_duplicates().itertuples(index=False):
                            cross_feasible.add((row.clean_firm, row.solar, row.wind, row.hydro))
                    except Exception as e:
                        print(f"  Warning: Could not read {fname} for cross-pollination: {e}")
    if cross_feasible:
        print(f"  Loaded {len(cross_feasible):,} feasible mixes from existing parquets")

    # Check checkpoint state
    existing_checkpoint = s1._load_mix_progress(ISO, threshold)
    if existing_checkpoint is not None:
        n_existing = len(existing_checkpoint.get("candidates", []))
        phase = existing_checkpoint.get("phase", "1a")
        cursor = existing_checkpoint.get("mix_cursor", 0)
        print(f"  Resuming from checkpoint: {n_existing:,} candidates, phase={phase}, cursor={cursor}")
    else:
        print(f"  Starting fresh (no checkpoint found)")

    # Set THRESHOLDS to just this one
    s1.THRESHOLDS = [threshold]

    # Run optimization with batch_size as the mix cap
    start = time.time()
    print(f"\n  Running optimize_threshold({ISO}, {threshold}%) — batch #{batch_num}, "
          f"max {batch_size:,} mixes...")

    candidates, pruning_info = s1.optimize_threshold(
        ISO,
        threshold,
        demand_arr,
        supply_matrix,
        hydro_cap,
        prev_pruning=None,
        cross_feasible_mixes=cross_feasible if cross_feasible else None,
        storage_levels=None,
        max_runtime_seconds=max_runtime_seconds,
        max_mixes_per_run=batch_size,
    )

    elapsed = time.time() - start
    completed = pruning_info.get("completed", True)
    total_candidates = len(candidates)
    new_candidates = candidates[prev_candidate_count:]
    n_new = len(new_candidates)

    print(f"\n  Batch #{batch_num} results:")
    print(f"    Total candidates (cumulative): {total_candidates:,}")
    print(f"    New candidates (this batch):   {n_new:,}")
    print(f"    Elapsed: {elapsed:.1f}s")
    print(f"    Threshold status: {'COMPLETED' if completed else 'IN PROGRESS'}")

    # Save batch parquet (new candidates only)
    if n_new > 0:
        path = _save_batch_parquet(threshold, batch_num, new_candidates)
        if path:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"    Output: {path} ({size_mb:.1f} MB)")
    else:
        print(f"    No new candidates in batch #{batch_num}")
        # If completed with 0 new candidates, this means all mixes were processed
        # but the remaining ones produced no feasible solutions

    # NOTE: We intentionally do NOT write the canonical combined file
    # (NYISO_t{XX}_raw_pfs.parquet). The batch files (b1, b2, ...) ARE the
    # output. Step 2 concatenates them directly. Writing both would cause
    # duplicate rows in Step 2.

    # Clean up the progress checkpoint when fully completed
    if completed:
        progress_path = s1._mix_progress_path(ISO, threshold)
        if os.path.exists(progress_path):
            os.remove(progress_path)
            print(f"    Cleaned up checkpoint: {progress_path}")

    # Update manifest
    t_state['batch_size'] = batch_size
    t_state['batches_completed'] = batch_num
    t_state['cumulative_candidates'] = total_candidates
    t_state['completed'] = completed

    # Track per-batch candidate counts
    batch_counts = t_state.get('batch_candidate_counts', {})
    batch_counts[str(batch_num)] = n_new
    t_state['batch_candidate_counts'] = batch_counts

    manifest['thresholds'][t_key] = t_state
    _save_manifest(manifest)
    print(f"    Manifest updated: {_manifest_path()}")

    # Summary
    print(f"\n{'=' * 70}")
    if completed:
        print(f"  DONE: NYISO {threshold}% fully completed in {batch_num} batches")
        print(f"  Total candidates: {total_candidates:,}")
        print(f"  Batch files: b1..b{batch_num} + combined canonical file")
    else:
        print(f"  BATCH #{batch_num} SAVED: NYISO {threshold}%")
        print(f"  Candidates this batch: {n_new:,}")
        print(f"  Cumulative candidates: {total_candidates:,}")
        print(f"  Next batch: #{batch_num + 1}")
        print(f"  Re-dispatch the workflow to continue.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
