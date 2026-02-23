#!/usr/bin/env python3
"""Step 1 PFS — Individual ISO/Threshold Runner

Runs step1_pfs_generator.optimize_threshold() for a single ISO × threshold pair.
Designed for GitHub Actions workflows where each ISO/threshold is dispatched as
a separate job, with support for runtime caps and checkpoint-based resume.

Output: data/step1_raw_pfs_parquets/{ISO}_t{threshold}_raw_pfs.parquet

Key features:
  - Standalone: no dependency on existing parquets. Loads raw EIA data and runs
    the full PFS grid search for the requested threshold.
  - Tranching: --max-runtime-minutes caps runtime per invocation. Progress is
    checkpointed in data/checkpoints_v4/ and auto-resumed on re-trigger.
  - Consistent output: writes directly to the step1_raw_pfs_parquets/ directory
    using the canonical {ISO}_t{threshold}_raw_pfs.parquet naming.
  - GitHub Actions friendly: exit code 0 on clean completion, exit code 0 on
    checkpoint save (partial progress), exit code 1 on error.

Usage:
  # Run NYISO 100% with no runtime cap (full run)
  python scripts/step1_pfs_individual_iso_threshold.py --iso NYISO --threshold 100

  # Run with 300-minute cap (saves checkpoint, resume on re-trigger)
  python scripts/step1_pfs_individual_iso_threshold.py --iso NYISO --threshold 100 --max-runtime-minutes 300

  # Run with mix cap (process N outer-loop mixes then checkpoint)
  python scripts/step1_pfs_individual_iso_threshold.py --iso NYISO --threshold 100 --max-mixes-per-run 5000
"""

import argparse
import os
import sys
import time

import numpy as np

# Ensure step1_pfs_generator is importable from scripts/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import step1_pfs_generator as s1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Step 1 PFS for a single ISO × threshold pair.",
    )
    parser.add_argument(
        "--iso",
        required=True,
        help="ISO name (e.g., CAISO, NYISO, PJM, ERCOT, NEISO)",
    )
    parser.add_argument(
        "--threshold",
        required=True,
        type=float,
        help="Threshold percentage (e.g., 100, 95, 92.5)",
    )
    parser.add_argument(
        "--max-runtime-minutes",
        type=float,
        default=0,
        help="Max runtime in minutes before checkpointing. 0 = no cap (default).",
    )
    parser.add_argument(
        "--max-mixes-per-run",
        type=int,
        default=0,
        help="Max outer-loop mixes to process before checkpointing. 0 = no cap (default).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    iso = args.iso.upper()
    threshold = args.threshold
    max_runtime_minutes = args.max_runtime_minutes
    max_mixes_per_run = args.max_mixes_per_run

    # Validate ISO
    if iso not in s1.ISOS:
        print(f"ERROR: Unknown ISO '{iso}'. Valid: {', '.join(s1.ISOS)}")
        sys.exit(1)

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

    max_runtime_seconds = None
    if max_runtime_minutes > 0:
        max_runtime_seconds = max_runtime_minutes * 60

    print("=" * 70)
    print(f"  Step 1 PFS — Individual ISO/Threshold Runner")
    print(f"  ISO: {iso}")
    print(f"  Threshold: {threshold}%")
    print(f"  Numba: {'enabled' if s1.HAS_NUMBA else 'disabled (NumPy fallback)'}")
    if max_runtime_minutes > 0:
        print(f"  Runtime cap: {max_runtime_minutes:.0f} minutes")
    if max_mixes_per_run > 0:
        print(f"  Mix cap: {max_mixes_per_run:,} mixes per run")
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
    demand_norm = demand_data[iso]["normalized"]
    supply_profiles = s1.get_supply_profiles(iso, gen_profiles)
    demand_arr, supply_matrix = s1.prepare_numpy_profiles(demand_norm, supply_profiles)
    hydro_cap = s1.HYDRO_CAPS[iso]

    # Seed cross_feasible_mixes from any existing completed parquets for this ISO
    # (helps with pruning if lower thresholds were run first, but NOT required)
    cross_feasible = set()
    if os.path.exists(s1.STEP1_RAW_PFS_PARQUET_DIR):
        import pyarrow.parquet as pq
        for fname in os.listdir(s1.STEP1_RAW_PFS_PARQUET_DIR):
            if fname.startswith(f"{iso}_t") and fname.endswith("_raw_pfs.parquet"):
                try:
                    t = pq.read_table(os.path.join(s1.STEP1_RAW_PFS_PARQUET_DIR, fname))
                    for row in t.to_pandas()[["clean_firm", "solar", "wind", "hydro"]].drop_duplicates().itertuples(index=False):
                        cross_feasible.add((row.clean_firm, row.solar, row.wind, row.hydro))
                except Exception as e:
                    print(f"  Warning: Could not read {fname} for cross-pollination: {e}")
    if cross_feasible:
        print(f"  Loaded {len(cross_feasible):,} feasible mixes from existing parquets for cross-pollination")

    # Check for existing checkpoint (resume support)
    existing_checkpoint = s1._load_mix_progress(iso, threshold)
    if existing_checkpoint is not None:
        n_existing = len(existing_checkpoint.get("candidates", []))
        phase = existing_checkpoint.get("phase", "1a")
        cursor = existing_checkpoint.get("mix_cursor", 0)
        print(f"  Resuming from checkpoint: {n_existing:,} solutions, phase={phase}, cursor={cursor}")
    else:
        print(f"  Starting fresh (no checkpoint found)")

    # Set THRESHOLDS to just this one so optimize_threshold uses correct bounds
    s1.THRESHOLDS = [threshold]

    # Run optimization
    start = time.time()
    print(f"\n  Running optimize_threshold({iso}, {threshold}%)...")

    candidates, pruning_info = s1.optimize_threshold(
        iso,
        threshold,
        demand_arr,
        supply_matrix,
        hydro_cap,
        prev_pruning=None,
        cross_feasible_mixes=cross_feasible if cross_feasible else None,
        storage_levels=None,  # Use default coarse storage sweep
        max_runtime_seconds=max_runtime_seconds,
        max_mixes_per_run=max_mixes_per_run if max_mixes_per_run > 0 else None,
    )

    elapsed = time.time() - start
    completed = pruning_info.get("completed", True)

    print(f"\n  Results: {len(candidates):,} solutions in {elapsed:.1f}s")
    print(f"  Status: {'COMPLETED' if completed else 'CHECKPOINTED (partial — resume on re-trigger)'}")

    # Save final parquet if completed
    if completed:
        s1._save_threshold_done(iso, threshold, candidates)
        done_path = s1._threshold_done_path(iso, threshold)
        if os.path.exists(done_path):
            size_mb = os.path.getsize(done_path) / (1024 * 1024)
            print(f"  Output: {done_path} ({size_mb:.1f} MB)")
        else:
            print(f"  WARNING: Expected output at {done_path} but file not found")
    else:
        # Progress already saved by optimize_threshold's internal checkpointing
        progress_path = s1._mix_progress_path(iso, threshold)
        if os.path.exists(progress_path):
            size_mb = os.path.getsize(progress_path) / (1024 * 1024)
            print(f"  Checkpoint: {progress_path} ({size_mb:.1f} MB)")

    # Summary
    print(f"\n{'=' * 70}")
    if completed:
        print(f"  DONE: {iso} {threshold}% — {len(candidates):,} solutions written")
    else:
        print(f"  PAUSED: {iso} {threshold}% — {len(candidates):,} solutions checkpointed")
        print(f"  Re-run this same command to resume from checkpoint.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
