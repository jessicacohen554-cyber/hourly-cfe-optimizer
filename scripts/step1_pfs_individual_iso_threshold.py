#!/usr/bin/env python3
"""Step 1 PFS — ISO/Threshold Runner

Runs step1_pfs_generator.optimize_threshold() for one or more ISO × threshold
pairs. Designed for GitHub Actions workflows where each ISO/threshold combo is
dispatched as a job, with support for runtime caps and checkpoint-based resume.

Output: data/step1_raw_pfs_parquets/{ISO}_t{threshold}_raw_pfs.parquet

Key features:
  - Multi-threshold: pass a comma-separated list or "all" to run multiple
    thresholds sequentially. Data loading and JIT warmup happen once.
  - Cross-pollination: feasible mixes found by earlier thresholds are fed into
    later ones for better pruning.
  - Standalone: no dependency on existing parquets. Loads raw EIA data and runs
    the full PFS grid search for the requested threshold(s).
  - Tranching: --max-runtime-minutes caps runtime per threshold. Progress is
    checkpointed in data/checkpoints_v4/ and auto-resumed on re-trigger.
  - Consistent output: writes directly to the step1_raw_pfs_parquets/ directory
    using the canonical {ISO}_t{threshold}_raw_pfs.parquet naming.
  - GitHub Actions friendly: exit code 0 on clean completion, exit code 0 on
    checkpoint save (partial progress), exit code 1 on error.

Usage:
  # Run NYISO 100% with no runtime cap (full run)
  python scripts/step1_pfs_individual_iso_threshold.py --iso NYISO --thresholds 100

  # Run multiple thresholds in one invocation
  python scripts/step1_pfs_individual_iso_threshold.py --iso NYISO --thresholds "100,95,92.5"

  # Run all 13 thresholds
  python scripts/step1_pfs_individual_iso_threshold.py --iso NYISO --thresholds all

  # Run with 300-minute cap per threshold (saves checkpoint, resume on re-trigger)
  python scripts/step1_pfs_individual_iso_threshold.py --iso NYISO --thresholds "100,95" --max-runtime-minutes 300

  # Run with mix cap (process N outer-loop mixes per threshold then checkpoint)
  python scripts/step1_pfs_individual_iso_threshold.py --iso NYISO --thresholds 100 --max-mixes-per-run 5000

  # Legacy single-threshold flag still works
  python scripts/step1_pfs_individual_iso_threshold.py --iso NYISO --threshold 100
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


def parse_thresholds(raw):
    """Parse threshold input: comma-separated values or 'all'."""
    if raw.strip().lower() == 'all':
        return list(s1.THRESHOLDS)

    parts = [p.strip() for p in raw.split(',') if p.strip()]
    parsed = []
    for p in parts:
        try:
            parsed.append(float(p))
        except ValueError:
            print(f"ERROR: Cannot parse threshold '{p}' as a number.")
            sys.exit(1)
    return parsed


def validate_thresholds(raw_thresholds):
    """Match each requested threshold to s1.THRESHOLDS and return matched list."""
    matched = []
    for req in raw_thresholds:
        found = None
        for t in s1.THRESHOLDS:
            if abs(float(t) - req) < 1e-9:
                found = t
                break
        if found is None:
            print(f"ERROR: Unsupported threshold {req}. "
                  f"Valid: {', '.join(str(t) for t in s1.THRESHOLDS)}")
            sys.exit(1)
        matched.append(found)
    return matched


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Step 1 PFS for one or more ISO × threshold pairs.",
    )
    parser.add_argument(
        "--iso",
        required=True,
        help="ISO name (e.g., CAISO, NYISO, PJM, ERCOT, NEISO)",
    )
    # Support both --thresholds (new) and --threshold (legacy)
    parser.add_argument(
        "--thresholds",
        default=None,
        help='Comma-separated thresholds or "all" (e.g., "100", "95,92.5,90", "all")',
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="(Legacy) Single threshold percentage (e.g., 100, 95, 92.5)",
    )
    parser.add_argument(
        "--max-runtime-minutes",
        type=float,
        default=0,
        help="Max runtime in minutes per threshold before checkpointing. 0 = no cap (default).",
    )
    parser.add_argument(
        "--max-mixes-per-run",
        type=int,
        default=0,
        help="Max outer-loop mixes per threshold before checkpointing. 0 = no cap (default).",
    )
    return parser.parse_args()


def run_single_threshold(iso, threshold, demand_arr, supply_matrix, hydro_cap,
                         cross_feasible, max_runtime_seconds, max_mixes_per_run):
    """Run optimization for a single threshold. Returns (candidates, completed, cross_feasible)."""

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

    # Add this threshold's feasible mixes to cross_feasible for subsequent thresholds
    for c in candidates:
        rm = c['resource_mix']
        cross_feasible.add((rm['clean_firm'], rm['solar'], rm['wind'], rm['hydro']))

    return candidates, completed, cross_feasible


def main():
    args = parse_args()

    iso = args.iso.upper()
    max_runtime_minutes = args.max_runtime_minutes
    max_mixes_per_run = args.max_mixes_per_run

    # Resolve thresholds: --thresholds takes priority, fallback to --threshold
    if args.thresholds is not None:
        thresholds = parse_thresholds(args.thresholds)
    elif args.threshold is not None:
        thresholds = [args.threshold]
    else:
        print("ERROR: Must provide --thresholds or --threshold.")
        sys.exit(1)

    # Validate ISO
    if iso not in s1.ISOS:
        print(f"ERROR: Unknown ISO '{iso}'. Valid: {', '.join(s1.ISOS)}")
        sys.exit(1)

    # Validate and match thresholds
    thresholds = validate_thresholds(thresholds)

    max_runtime_seconds = None
    if max_runtime_minutes > 0:
        max_runtime_seconds = max_runtime_minutes * 60

    threshold_strs = ', '.join(f"{t}%" for t in thresholds)
    print("=" * 70)
    print(f"  Step 1 PFS — ISO/Threshold Runner")
    print(f"  ISO: {iso}")
    print(f"  Thresholds: {threshold_strs}  ({len(thresholds)} total)")
    print(f"  Numba: {'enabled' if s1.HAS_NUMBA else 'disabled (NumPy fallback)'}")
    if max_runtime_minutes > 0:
        print(f"  Runtime cap: {max_runtime_minutes:.0f} minutes per threshold")
    if max_mixes_per_run > 0:
        print(f"  Mix cap: {max_mixes_per_run:,} mixes per threshold")
    print("=" * 70)

    # Load data (once for all thresholds)
    demand_data, gen_profiles, _, _ = s1.load_data()

    # JIT warmup (once)
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

    # Prepare ISO data (once for all thresholds)
    demand_norm = demand_data[iso]["normalized"]
    supply_profiles = s1.get_supply_profiles(iso, gen_profiles)
    demand_arr, supply_matrix = s1.prepare_numpy_profiles(demand_norm, supply_profiles)
    hydro_cap = s1.HYDRO_CAPS[iso]

    # Seed cross_feasible_mixes from any existing completed parquets for this ISO
    cross_feasible = set()
    CROSS_POLL_COLS = {"clean_firm", "solar", "wind", "hydro"}
    if os.path.exists(s1.STEP1_RAW_PFS_PARQUET_DIR):
        import pyarrow.parquet as pq
        for fname in os.listdir(s1.STEP1_RAW_PFS_PARQUET_DIR):
            if fname.startswith(f"{iso}_t") and fname.endswith("_raw_pfs.parquet"):
                fpath = os.path.join(s1.STEP1_RAW_PFS_PARQUET_DIR, fname)
                try:
                    schema = pq.read_schema(fpath)
                    if not CROSS_POLL_COLS.issubset(schema.names):
                        continue  # Old-schema parquet, skip silently
                    t = pq.read_table(fpath, columns=list(CROSS_POLL_COLS))
                    for row in t.to_pandas().drop_duplicates().itertuples(index=False):
                        cross_feasible.add((row.clean_firm, row.solar, row.wind, row.hydro))
                except Exception as e:
                    print(f"  Warning: Could not read {fname} for cross-pollination: {e}")
    if cross_feasible:
        print(f"  Loaded {len(cross_feasible):,} feasible mixes from existing parquets for cross-pollination")

    # Run each threshold sequentially
    results = {}  # threshold -> (n_candidates, completed)
    total_start = time.time()

    for i, threshold in enumerate(thresholds):
        print(f"\n{'=' * 70}")
        print(f"  [{i+1}/{len(thresholds)}] Threshold: {threshold}%")
        print(f"{'=' * 70}")

        candidates, completed, cross_feasible = run_single_threshold(
            iso, threshold, demand_arr, supply_matrix, hydro_cap,
            cross_feasible, max_runtime_seconds, max_mixes_per_run,
        )
        results[threshold] = (len(candidates), completed)

    total_elapsed = time.time() - total_start

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY — {iso}  ({len(thresholds)} threshold{'s' if len(thresholds) > 1 else ''})")
    print(f"  Total elapsed: {total_elapsed:.1f}s")
    print(f"  {'Threshold':<12} {'Solutions':>12} {'Status':<15}")
    print(f"  {'-'*12} {'-'*12} {'-'*15}")
    for threshold in thresholds:
        n_cands, completed = results[threshold]
        status = 'COMPLETED' if completed else 'CHECKPOINTED'
        print(f"  {str(threshold)+'%':<12} {n_cands:>12,} {status:<15}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
