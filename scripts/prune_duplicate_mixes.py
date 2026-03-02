#!/usr/bin/env python3
"""
Prune duplicate mixes from PFS, storage-refined, and EF parquet files.

Each threshold file should only contain mixes whose hourly_match_score falls
within its band: [threshold, next_threshold).  The top threshold (99.99) keeps
everything >= 99.99.

Targets:
  - Step 1c: data/step1-pfs-parquets/{ISO}_t{T}_raw_pfs.parquet
  - Step 1d: data/step1d-storage-parquets/{ISO}_t{T}_storage_refined.parquet
  - Step 2:  data/step2-ef-parquets/step2_ef_{ISO}_t{T}.parquet
  - Step 2.5: data/step2-ef-parquets/step2_ef_{ISO}_t{T}_expanded.parquet (if present)
"""

import argparse
import os
import sys
import pandas as pd

THRESHOLDS = [10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 87.5, 90,
              92.5, 95, 97.5, 99, 99.5, 99.9, 99.99]

ISOS = ["CAISO", "ERCOT", "PJM", "NYISO", "NEISO", "MISO", "SPP"]

SCORE_COL = "hourly_match_score"

# Directories relative to repo root
STEP1C_DIR = "data/step1-pfs-parquets"
STEP1D_DIR = "data/step1d-storage-parquets"
STEP2_DIR  = "data/step2-ef-parquets"


def threshold_str(t):
    """Format threshold for filenames: 65.0 -> '65', 87.5 -> '87.5'."""
    return str(int(t)) if t == int(t) else str(t)


def get_upper_bound(t_idx):
    """Return upper bound for threshold band (exclusive). Last threshold has no upper."""
    if t_idx + 1 < len(THRESHOLDS):
        return THRESHOLDS[t_idx + 1]
    return None  # no upper bound for 99.99


def prune_file(filepath, lower, upper, dry_run=False):
    """
    Filter a parquet file to keep only rows where lower <= score < upper.
    Returns (original_count, pruned_count, was_modified).
    """
    if not os.path.exists(filepath):
        return 0, 0, False

    df = pd.read_parquet(filepath)
    if SCORE_COL not in df.columns:
        print(f"  SKIP {filepath} — no '{SCORE_COL}' column")
        return len(df), len(df), False

    orig = len(df)
    mask = df[SCORE_COL] >= lower
    if upper is not None:
        mask &= df[SCORE_COL] < upper
    pruned = df[mask]
    new_count = len(pruned)

    if new_count == orig:
        return orig, new_count, False

    if not dry_run:
        pruned.to_parquet(filepath, index=False)

    return orig, new_count, True


def build_file_list(iso, t_str):
    """Return list of (label, filepath) tuples for a given ISO and threshold."""
    files = []
    # Step 1c
    files.append(("1c", os.path.join(STEP1C_DIR, f"{iso}_t{t_str}_raw_pfs.parquet")))
    # Step 1d
    files.append(("1d", os.path.join(STEP1D_DIR, f"{iso}_t{t_str}_storage_refined.parquet")))
    # Step 2
    files.append(("2", os.path.join(STEP2_DIR, f"step2_ef_{iso}_t{t_str}.parquet")))
    # Step 2.5 (expanded EF, if present)
    files.append(("2.5", os.path.join(STEP2_DIR, f"step2_ef_{iso}_t{t_str}_expanded.parquet")))
    return files


def main():
    parser = argparse.ArgumentParser(description="Prune duplicate mixes across threshold bands")
    parser.add_argument("--iso", default="ALL", choices=["ALL"] + ISOS,
                        help="ISO to process (default: ALL)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be pruned without modifying files")
    parser.add_argument("--step", default="all", choices=["all", "1c", "1d", "2", "2.5"],
                        help="Which step's files to prune (default: all)")
    args = parser.parse_args()

    isos = ISOS if args.iso == "ALL" else [args.iso]
    mode = "DRY RUN" if args.dry_run else "PRUNING"

    total_before = 0
    total_after = 0
    files_modified = 0

    for iso in isos:
        print(f"\n{'='*60}")
        print(f" {iso} — {mode}")
        print(f"{'='*60}")
        iso_before = 0
        iso_after = 0

        for i, t in enumerate(THRESHOLDS):
            t_str = threshold_str(t)
            upper = get_upper_bound(i)
            upper_str = str(upper) if upper is not None else "∞"

            all_files = build_file_list(iso, t_str)
            # Filter by step if specified
            if args.step != "all":
                all_files = [(label, fp) for label, fp in all_files if label == args.step]

            for label, filepath in all_files:
                orig, kept, modified = prune_file(filepath, t, upper, dry_run=args.dry_run)
                if orig == 0:
                    continue
                dropped = orig - kept
                iso_before += orig
                iso_after += kept
                if modified:
                    files_modified += 1
                    pct = (dropped / orig * 100) if orig > 0 else 0
                    print(f"  Step {label} t{t_str} [{t}–{upper_str}): "
                          f"{orig:>12,} → {kept:>12,}  (dropped {dropped:>12,}, {pct:.1f}%)")

        total_before += iso_before
        total_after += iso_after
        iso_dropped = iso_before - iso_after
        iso_pct = (iso_dropped / iso_before * 100) if iso_before > 0 else 0
        print(f"\n  {iso} total: {iso_before:,} → {iso_after:,} "
              f"(dropped {iso_dropped:,}, {iso_pct:.1f}%)")

    print(f"\n{'='*60}")
    total_dropped = total_before - total_after
    total_pct = (total_dropped / total_before * 100) if total_before > 0 else 0
    print(f" GRAND TOTAL: {total_before:,} → {total_after:,} "
          f"(dropped {total_dropped:,}, {total_pct:.1f}%)")
    print(f" Files modified: {files_modified}")
    print(f"{'='*60}")

    if args.dry_run:
        print("\n  ** DRY RUN — no files were modified **\n")


if __name__ == "__main__":
    main()
