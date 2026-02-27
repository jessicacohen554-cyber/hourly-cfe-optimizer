#!/usr/bin/env python3
"""Step 1c: Build PFS by mining the scored mix database.

Reads the pre-scored coarse cache ({ISO}_coarse_cache.parquet) built by
step1b_score_mixes.py. For each requested threshold:

  1. Filter: score >= target → feasible without storage
  2. Near-miss: score >= target - 0.40 → storage sweep (in batches)
  3. Fine refinement: 1% combos around archetypes → score + storage sweep
  4. Save: {ISO}_t{XX}_raw_pfs.parquet

The scored database is ~40 MB (1.6M rows × 6 float64 cols). Loading it is
trivial. Supply rows for storage dispatch are computed on-the-fly per batch
of 100 mixes — never holding more than 100 × 8760 in memory.

Fine refinement also scores new combos in chunks (same as step1b), keeping
peak memory bounded at ~1.4 GiB.

Usage:
  python scripts/step1c_build_pfs.py --iso NYISO --thresholds all
  python scripts/step1c_build_pfs.py --iso CAISO --thresholds "100,95,92.5"
  python scripts/step1c_build_pfs.py --iso MISO --thresholds 100
"""

import argparse
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import step1_pfs_generator as s1

try:
    import pyarrow.parquet as pq
except ImportError:
    print("ERROR: pyarrow required. pip install pyarrow")
    sys.exit(1)


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


def main():
    parser = argparse.ArgumentParser(
        description="Step 1c: Build PFS from scored mix database.",
    )
    parser.add_argument(
        "--iso", required=True,
        help="ISO name (e.g., CAISO, NYISO, PJM, ERCOT, NEISO, MISO, SPP)",
    )
    parser.add_argument(
        "--thresholds", required=True,
        help='Comma-separated thresholds or "all" (e.g., "100", "95,92.5", "all")',
    )
    args = parser.parse_args()

    iso = args.iso.upper()
    if iso not in s1.ISOS:
        print(f"ERROR: Unknown ISO '{iso}'. Valid: {', '.join(s1.ISOS)}")
        sys.exit(1)

    thresholds = parse_thresholds(args.thresholds)
    thresholds = validate_thresholds(thresholds)

    # Verify scored database exists
    cache_path = s1._coarse_cache_path(iso)
    if not os.path.exists(cache_path):
        print(f"ERROR: Scored mix database not found: {cache_path}")
        print(f"Run step1b_score_mixes.py --iso {iso} first.")
        sys.exit(1)

    rtypes = s1.get_resource_types(iso)
    threshold_strs = ', '.join(f"{t}%" for t in thresholds)

    print("=" * 70)
    print(f"  Step 1c — Build PFS from Scored Database")
    print(f"  ISO: {iso}")
    print(f"  Thresholds: {threshold_strs}  ({len(thresholds)} total)")
    print(f"  Resources: {len(rtypes)}D ({', '.join(rtypes)})")
    print(f"  Database: {cache_path}")
    print(f"  Numba: {'enabled' if s1.HAS_NUMBA else 'disabled (NumPy fallback)'}")
    print("=" * 70)

    # Load EIA data (needed for storage dispatch + fine refinement scoring)
    print("\nLoading EIA data...")
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
        s1._batch_compute_storage_caps(dd, ds2, 1.0, 2, 4, 8, 100)
        dl = np.array([0.0, 0.1], dtype=np.float64)
        s1._batch_storage_scores(dd, ds, 1.0, dl, dl, dl, 2, 2, 2,
                                 0.85, 0.85, 0.50, 4, 8, 100, 168, 48)
        s1._batch_mixes_storage_screen(dd, ds2, 1.0, 2, dl, dl, dl, 2, 2, 2,
                                       0.85, 0.85, 0.50, 4, 8, 100, 168, 48)
        print("  JIT compilation complete")

    # Prepare supply data (for storage dispatch + fine refinement)
    demand_norm = demand_data[iso]["normalized"]
    supply_profiles = s1.get_supply_profiles(iso, gen_profiles)
    demand_arr, supply_matrix = s1.prepare_numpy_profiles(
        iso, demand_norm, supply_profiles)

    # Load the scored database
    print(f"\nLoading scored database...")
    cached = s1.load_coarse_cache(iso)
    if cached is None:
        print(f"ERROR: Failed to load scored database for {iso}")
        sys.exit(1)
    coarse_combos, coarse_scores = cached

    os.makedirs(s1.STEP1_RAW_PFS_PARQUET_DIR, exist_ok=True)

    # Override THRESHOLDS so optimize_threshold uses only requested set
    original_thresholds = list(s1.THRESHOLDS)
    s1.THRESHOLDS = thresholds

    # ── Phase 1: Coarse storage sweep ──
    print(f"\nPhase 1 — Coarse storage sweep ({len(thresholds)} thresholds)")
    total_start = time.time()

    # Track storage saturation incrementally — don't keep full result lists
    max_bat4 = max_bat8 = max_ldes = max_h2 = 0.0

    for threshold in thresholds:
        t_start = time.time()
        feasible = s1.optimize_threshold(
            iso, threshold, demand_arr, supply_matrix,
            coarse_combos, coarse_scores)

        archetypes = set()
        for c in feasible:
            archetypes.add(tuple(c['resource_mix'][rt] for rt in rtypes))
            # Update saturation stats inline (avoids keeping millions of dicts in memory)
            max_bat4 = max(max_bat4, c.get('battery_dispatch_pct', 0) or 0)
            max_bat8 = max(max_bat8, c.get('battery8_dispatch_pct', 0) or 0)
            max_ldes = max(max_ldes, c.get('ldes_dispatch_pct', 0) or 0)
            max_h2 = max(max_h2, c.get('h2_dispatch_pct', 0) or 0)

        t_elapsed = time.time() - t_start
        print(f"    {iso} {threshold}%: {len(feasible)} solutions "
              f"({len(archetypes)} archetypes), {t_elapsed:.1f}s")

        s1._save_threshold_done(iso, threshold, feasible)
        del feasible, archetypes  # free memory before next threshold

    print(f"\n  Coarse saturation — bat4={max_bat4:.2f}%, bat8={max_bat8:.2f}%, "
          f"ldes={max_ldes:.1f}%, h2={max_h2:.1f}%")

    # ── Phase 2: Fine storage sweep within saturation range ──
    fine_levels = s1.build_fine_storage_levels(
        max_bat4, max_bat8, max_ldes, max_h2)

    if fine_levels is not None:
        print(f"\nPhase 2 — Fine storage sweep")
        for threshold in thresholds:
            t_start = time.time()
            feasible = s1.optimize_threshold(
                iso, threshold, demand_arr, supply_matrix,
                coarse_combos, coarse_scores,
                storage_levels=fine_levels)

            archetypes = set()
            for c in feasible:
                archetypes.add(tuple(c['resource_mix'][rt] for rt in rtypes))

            t_elapsed = time.time() - t_start
            print(f"    {iso} {threshold}%: {len(feasible)} solutions "
                  f"({len(archetypes)} archetypes), {t_elapsed:.1f}s")

            s1._save_threshold_done(iso, threshold, feasible)
    else:
        print("\n  No storage saturation detected — skipping Phase 2")

    total_elapsed = time.time() - total_start

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY — {iso}  ({len(thresholds)} threshold"
          f"{'s' if len(thresholds) > 1 else ''})")
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
                print(f"  {str(threshold)+'%':<12} {n_rows:>12,} "
                      f"{done_path} ({size_mb:.1f} MB)")
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
