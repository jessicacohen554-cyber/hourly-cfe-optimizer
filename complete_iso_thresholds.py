#!/usr/bin/env python3
"""Complete remaining thresholds for one or more ISOs from interim cache files.

Usage:
  python complete_iso_thresholds.py <ISO> <threshold1> <threshold2> ...

If no arguments are supplied, this script defaults to the known residual runs:
  - NYISO 100%
  - CAISO 100%
"""
import os
import sys
import time
import argparse

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import step1_pfs_generator as s1

DEFAULT_REMAINING = [
    ("NYISO", [100.0]),
    ("CAISO", [100.0]),
]


def _parse_run_plan(argv):
    parser = argparse.ArgumentParser(
        description="Complete one or more thresholds for a single ISO.",
    )
    parser.add_argument(
        "iso_positional",
        nargs="?",
        help="ISO name (legacy positional style: <ISO> <threshold1> <threshold2> ...)",
    )
    parser.add_argument(
        "thresholds_positional",
        nargs="*",
        help="Threshold values in percent for positional style.",
    )
    parser.add_argument("--iso", dest="iso_flag", help="ISO name (e.g., CAISO, NYISO, PJM, ERCOT, NEISO)")
    parser.add_argument(
        "--threshold",
        dest="threshold_flags",
        action="append",
        help=(
            "Threshold value. Pass multiple times for multiple thresholds "
            "(e.g., --threshold 95 --threshold 100)"
        ),
    )
    parser.add_argument(
        "--max-runtime-minutes",
        type=float,
        default=15,
        help="Maximum runtime per threshold before saving a checkpoint and exiting (default: 15).",
    )

    args = parser.parse_args(argv[1:])

    if args.iso_flag and args.iso_positional:
        raise SystemExit("Provide ISO only once: use either positional ISO or --iso, not both.")

    iso = args.iso_flag or args.iso_positional
    threshold_tokens = []

    if args.threshold_flags:
        threshold_tokens.extend(args.threshold_flags)
    if args.thresholds_positional:
        threshold_tokens.extend(args.thresholds_positional)

    if not iso and not threshold_tokens:
        return DEFAULT_REMAINING, args.max_runtime_minutes

    if not iso:
        raise SystemExit(
            "No ISO provided. Usage: python complete_iso_thresholds.py --iso <ISO> --threshold <VALUE> "
            "[--threshold <VALUE> ...]"
        )

    thresholds = [float(t) for t in threshold_tokens]

    if not thresholds:
        raise SystemExit(
            f"No thresholds provided for {iso}. "
            "Usage: python complete_iso_thresholds.py --iso <ISO> --threshold <VALUE> [--threshold <VALUE> ...]"
        )
    return [(iso, thresholds)], args.max_runtime_minutes


def _run_iso_thresholds(iso, remaining, max_runtime_minutes):
    s1.THRESHOLDS = remaining

    interim_path = os.path.join(s1.CHECKPOINT_DIR, f"{iso}_v4_interim.parquet")
    df = pq.read_table(interim_path).to_pandas()

    max_bat4 = df["battery_dispatch_pct"].max()
    max_bat8 = df.get("battery8_dispatch_pct", 0)
    if hasattr(max_bat8, "max"):
        max_bat8 = max_bat8.max()

    fine_bat4_max = max_bat4 + 0.25
    fine_bat8_max = max_bat8 + 0.25
    fine_bat4 = [0] + [round(x * 0.05, 2) for x in range(1, int(fine_bat4_max / 0.05) + 1)]
    fine_bat8 = [0] + [round(x * 0.05, 2) for x in range(1, int(fine_bat8_max / 0.05) + 1)]
    fine_ldes = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 5, 8, 10]
    fine_levels = {"bat4": fine_bat4, "bat8": fine_bat8, "ldes": fine_ldes}

    print(f"{iso}: Completing thresholds {remaining}")
    print(f"  bat4: {len(fine_bat4)} levels (0-{fine_bat4[-1]:.2f}%)")
    print(f"  bat8: {len(fine_bat8)} levels (0-{fine_bat8[-1]:.2f}%)")

    demand_data, gen_profiles, _, _ = s1.load_data()

    if s1.HAS_NUMBA:
        print("  JIT warmup...")
        H = s1.H
        dd = np.ones(H) / H
        ds = np.ones(H) / H
        ds2 = np.ones((2, H)) / H
        s1._score_with_all_storage(dd, ds, 1.0, 0.01, 0.0025, 0.85, 0.01, 0.00125, 0.85, 0.01, 0.0001, 0.50, 168)
        s1._batch_score_no_storage(dd, ds2, 1.0, 2)
        s1._batch_score_storage(dd, ds2, 1.0, 2, 0.01, 0.0025, 0.85, 0.01, 0.00125, 0.85, 0.01, 0.0001, 0.50, 168)
        s1._compute_storage_caps(dd, ds, 1.0, 4, 8, 100)
        dl = np.array([0.0, 0.1], dtype=np.float64)
        s1._batch_storage_scores(dd, ds, 1.0, dl, dl, dl, 2, 2, 2, 0.85, 0.85, 0.50, 4, 8, 100, 168, 48)
        s1._batch_mixes_storage_screen(dd, ds2, 1.0, 2, dl, dl, dl, 2, 2, 2, 0.85, 0.85, 0.50, 4, 8, 100, 168, 48)
        print("  JIT ready")

    demand_norm = demand_data[iso]["normalized"]
    supply_profiles = s1.get_supply_profiles(iso, gen_profiles)
    demand_arr, supply_matrix = s1.prepare_numpy_profiles(demand_norm, supply_profiles)
    hydro_cap = s1.HYDRO_CAPS[iso]

    cross_feasible = set()
    for _, row in df[["clean_firm", "solar", "wind", "hydro"]].drop_duplicates().iterrows():
        cross_feasible.add((row["clean_firm"], row["solar"], row["wind"], row["hydro"]))
    print(f"  {len(cross_feasible):,} feasible mixes from cache")
    del df

    for t in remaining:
        done_path = s1._threshold_done_path(iso, t)
        if os.path.exists(done_path):
            os.remove(done_path)

    start = time.time()
    prev_pruning = None

    for threshold in remaining:
        progress = s1._load_mix_progress(iso, threshold)
        if progress is not None:
            n_existing = len(progress.get("candidates", []))
            phase = progress.get("phase", "1a")
            cursor = progress.get("mix_cursor", 0)
            print(
                f"  Resuming {iso} {threshold}% from checkpoint: "
                f"{n_existing:,} solutions, phase={phase}, cursor={cursor}"
            )
        else:
            print(f"  Starting {iso} {threshold}% fresh")

        t_start = time.time()
        candidates, prev_pruning = s1.optimize_threshold(
            iso,
            threshold,
            demand_arr,
            supply_matrix,
            hydro_cap,
            prev_pruning=prev_pruning,
            cross_feasible_mixes=cross_feasible,
            storage_levels=fine_levels,
            max_runtime_seconds=max_runtime_minutes * 60,
        )
        threshold_completed = prev_pruning.get("completed", True)
        elapsed = time.time() - t_start
        print(f"  {iso} {threshold}%: {len(candidates):,} solutions, {elapsed:.1f}s")
        if threshold_completed:
            s1._save_threshold_done(iso, threshold, candidates)
            s1.append_threshold_to_cache(iso, threshold, candidates)
        else:
            print(
                f"  {iso} {threshold}% paused at runtime cap ({max_runtime_minutes} min). "
                "Progress checkpointed for resume."
            )
            break

    print(f"\nSaving {iso} cache...")
    interim_table = pq.read_table(interim_path)

    tables = [interim_table]
    for t in remaining:
        done_path = s1._threshold_done_path(iso, t)
        if os.path.exists(done_path):
            tables.append(pq.read_table(done_path))

    merged = pa.concat_tables(tables)
    merged = s1._dedup_parquet_table(merged)

    cache_path = os.path.join(s1.DATA_DIR, f"physics_cache_v4_{iso}.parquet")
    pq.write_table(merged, cache_path, compression="snappy")
    size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    print(f"  Cache: {cache_path} ({merged.num_rows:,} solutions, {size_mb:.1f} MB)")

    dash_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard", f"physics_results_v4_{iso}.parquet")
    pq.write_table(merged, dash_path, compression="snappy")
    print(f"  Dashboard: {dash_path}")

    df_final = merged.to_pandas()
    for t in sorted(df_final.threshold.unique()):
        print(f"  {t}%: {len(df_final[df_final.threshold == t]):,}")
    print(f"  Total: {df_final.shape[0]:,}")
    print(f"\nDone in {time.time() - start:.1f}s")


if __name__ == "__main__":
    run_plan, max_runtime_minutes = _parse_run_plan(sys.argv)
    for iso_name, thresholds in run_plan:
        _run_iso_thresholds(iso_name, thresholds, max_runtime_minutes)
