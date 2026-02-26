#!/usr/bin/env python3
"""Step 1 PFS — ISO/Threshold Runner (v5.0)

Runs step1_pfs_generator.process_iso() for a single ISO, optionally filtered
to specific thresholds. Designed for GitHub Actions workflows where each ISO
is dispatched as a job.

Output: data/step1-pfs-parquets/{ISO}_t{threshold}_raw_pfs.parquet

Architecture (v5.0 — no procurement):
  1. Load data + JIT warmup (once)
  2. Build or load coarse cache (one-time per ISO)
  3. For each threshold: read cache → filter → storage sweep → fine refinement
  4. Two-phase adaptive storage: coarse sweep → analyze saturation → fine sweep

Key features:
  - Multi-threshold: pass a comma-separated list or "all" to run multiple
    thresholds sequentially. Data loading and JIT warmup happen once.
  - Standalone: no dependency on existing parquets. Loads raw EIA data and runs
    the full PFS grid search for the requested threshold(s).
  - Consistent output: writes directly to the step1-pfs-parquets/ directory
    using the canonical {ISO}_t{threshold}_raw_pfs.parquet naming.
  - GitHub Actions friendly: exit code 0 on clean completion, exit code 1 on error.

Usage:
  # Run NYISO for all thresholds
  python scripts/step1_pfs_individual_iso_threshold.py --iso NYISO --thresholds all

  # Run specific thresholds
  python scripts/step1_pfs_individual_iso_threshold.py --iso NYISO --thresholds "100,95,92.5"

  # Run a single threshold
  python scripts/step1_pfs_individual_iso_threshold.py --iso NYISO --thresholds 100
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
        description="Run Step 1 PFS for a single ISO (all or selected thresholds).",
    )
    parser.add_argument(
        "--iso",
        required=True,
        help="ISO name (e.g., CAISO, NYISO, PJM, ERCOT, NEISO, MISO, SPP)",
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
    # Kept for backward compatibility with existing workflow YAML but no longer
    # used — the v5 optimizer runs to completion (no checkpoint/resume).
    parser.add_argument(
        "--max-runtime-minutes",
        type=float,
        default=0,
        help="(Ignored in v5) Max runtime in minutes per threshold.",
    )
    parser.add_argument(
        "--max-mixes-per-run",
        type=int,
        default=0,
        help="(Ignored in v5) Max outer-loop mixes per threshold.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    iso = args.iso.upper()

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

    threshold_strs = ', '.join(f"{t}%" for t in thresholds)
    rtypes = s1.get_resource_types(iso)
    print("=" * 70)
    print(f"  Step 1 PFS v5.0 — ISO/Threshold Runner")
    print(f"  ISO: {iso}")
    print(f"  Thresholds: {threshold_strs}  ({len(thresholds)} total)")
    print(f"  Resources: {len(rtypes)}D ({', '.join(rtypes)})")
    print(f"  Numba: {'enabled' if s1.HAS_NUMBA else 'disabled (NumPy fallback)'}")
    print("=" * 70)

    # Load data (once for all thresholds)
    print("\nLoading data...")
    demand_data, gen_profiles, _, _ = s1.load_data()

    # JIT warmup (once) — matches step1_pfs_generator.main() warmup
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
        s1._batch_compute_storage_caps(dd, ds2, 1.0, 2, 4, 8, 100)
        dl = np.array([0.0, 0.1], dtype=np.float64)
        s1._batch_storage_scores(dd, ds, 1.0, dl, dl, dl, 2, 2, 2,
                                 0.85, 0.85, 0.50, 4, 8, 100, 168, 48)
        s1._batch_mixes_storage_screen(dd, ds2, 1.0, 2, dl, dl, dl, 2, 2, 2,
                                       0.85, 0.85, 0.50, 4, 8, 100, 168, 48)
        print("  JIT compilation complete")

    # Override THRESHOLDS to only run the requested subset
    original_thresholds = list(s1.THRESHOLDS)
    s1.THRESHOLDS = thresholds

    # Prepare ISO data
    demand_norm = demand_data[iso]["normalized"]
    supply_profiles = s1.get_supply_profiles(iso, gen_profiles)
    demand_arr, supply_matrix = s1.prepare_numpy_profiles(iso, demand_norm, supply_profiles)

    os.makedirs(s1.STEP1_RAW_PFS_PARQUET_DIR, exist_ok=True)

    # Use process_iso — the same function main() uses. This ensures consistent
    # behavior (coarse cache → two-phase adaptive storage → save per threshold).
    print(f"\nRunning process_iso for {iso} ({len(thresholds)} thresholds)...")
    total_start = time.time()

    iso_name, iso_results = s1.process_iso((iso, demand_data, gen_profiles))

    total_elapsed = time.time() - total_start

    # Count solutions from parquet outputs
    import pyarrow.parquet as pq
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY — {iso}  ({len(thresholds)} threshold{'s' if len(thresholds) > 1 else ''})")
    print(f"  Total elapsed: {total_elapsed:.1f}s")
    print(f"  {'Threshold':<12} {'Solutions':>12} {'Output':<50}")
    print(f"  {'-'*12} {'-'*12} {'-'*50}")

    total_solutions = 0
    for threshold in thresholds:
        done_path = s1._threshold_done_path(iso, threshold)
        if os.path.exists(done_path):
            try:
                n_rows = pq.read_table(done_path).num_rows
                size_mb = os.path.getsize(done_path) / (1024 * 1024)
                total_solutions += n_rows
                print(f"  {str(threshold)+'%':<12} {n_rows:>12,} {done_path} ({size_mb:.1f} MB)")
            except Exception as e:
                print(f"  {str(threshold)+'%':<12} {'ERROR':>12} {e}")
        else:
            print(f"  {str(threshold)+'%':<12} {'MISSING':>12} {done_path}")

    print(f"\n  Total: {total_solutions:,} solutions in {total_elapsed:.1f}s")
    print(f"{'=' * 70}")

    # Restore thresholds
    s1.THRESHOLDS = original_thresholds


if __name__ == "__main__":
    main()
