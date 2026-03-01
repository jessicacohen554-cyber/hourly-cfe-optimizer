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
  python scripts/step1c_build_pfs.py --iso NYISO --thresholds all --auto-commit
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


def _compute_code_hash(iso):
    """Hash step1c + step1_pfs_generator source + scored database mtime+size.

    If any of these change, cached thresholds are stale and must be recomputed.
    The scored database (coarse_cache parquet) is included because different
    database contents produce different PFS results even with identical code.
    """
    h = hashlib.sha256()

    # Hash source files
    for fname in ['step1c_build_pfs.py', 'step1_pfs_generator.py']:
        fpath = os.path.join(SCRIPT_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath, 'rb') as f:
                h.update(f.read())

    # Hash scored database identity (mtime + size, not full content — it's 40+ MB)
    cache_path = s1._coarse_cache_path(iso)
    if os.path.exists(cache_path):
        stat = os.stat(cache_path)
        h.update(f"{cache_path}:{stat.st_size}:{stat.st_mtime_ns}".encode())

    return h.hexdigest()[:16]


def _manifest_path(iso):
    """Path to the resume manifest for an ISO."""
    return os.path.join(s1.STEP1_RAW_PFS_PARQUET_DIR,
                        f'{iso}_pfs_manifest.json')


def _load_manifest(iso, current_hash):
    """Load manifest if it exists and code hash matches. Otherwise return empty.

    Returns dict like {"code_hash": "abc123", "phase1": [50, 55, ...], "phase2": [50, ...]}
    """
    mpath = _manifest_path(iso)
    if not os.path.exists(mpath):
        return None
    try:
        with open(mpath, 'r') as f:
            manifest = json.load(f)
        if manifest.get('code_hash') != current_hash:
            print(f"  Code/data changed (hash {manifest.get('code_hash', '?')[:8]}→{current_hash[:8]})"
                  f" — recomputing all thresholds")
            os.remove(mpath)
            return None
        return manifest
    except (json.JSONDecodeError, OSError):
        return None


def _save_manifest(iso, code_hash, phase1_done, phase2_done):
    """Write the resume manifest after each threshold completes."""
    mpath = _manifest_path(iso)
    os.makedirs(os.path.dirname(mpath), exist_ok=True)
    manifest = {
        'code_hash': code_hash,
        'phase1': sorted(phase1_done),
        'phase2': sorted(phase2_done),
    }
    with open(mpath, 'w') as f:
        json.dump(manifest, f, indent=2)


def git_commit_threshold(iso, threshold, phase_label):
    """Commit and push a single threshold's parquet file to preserve progress.

    Uses subprocess so failures don't crash the optimizer — just warns.
    """
    done_path = s1._threshold_done_path(iso, threshold)
    if not os.path.exists(done_path):
        print(f"      [auto-commit] No file to commit for {iso} {threshold}%")
        return False

    try:
        # Stage the file
        subprocess.run(['git', 'add', '-f', done_path],
                       check=True, capture_output=True, text=True)

        # Check if there are actual changes
        result = subprocess.run(['git', 'diff', '--cached', '--quiet'],
                                capture_output=True)
        if result.returncode == 0:
            # No changes to commit (file unchanged from last commit)
            return False

        # Get file size for commit message
        size_mb = os.path.getsize(done_path) / (1024 * 1024)

        # Commit
        msg = (f"PFS {phase_label}: {iso} {threshold}% "
               f"({size_mb:.1f} MB) — auto-commit")
        subprocess.run(['git', 'commit', '-m', msg],
                       check=True, capture_output=True, text=True)

        # Push with retry (up to 3 attempts with exponential backoff)
        # Use 'git push -u origin HEAD' to auto-create remote branch and set tracking
        for attempt in range(1, 4):
            result = subprocess.run(
                ['git', 'push', '-u', 'origin', 'HEAD'],
                capture_output=True, text=True)
            if result.returncode == 0:
                print(f"      [auto-commit] {iso} {threshold}% committed & pushed ({phase_label})")
                return True
            if attempt < 3:
                wait = 2 ** attempt
                print(f"      [auto-commit] Push attempt {attempt} failed, retrying in {wait}s...")
                time.sleep(wait)

        print(f"      [auto-commit] Push failed for {iso} {threshold}% "
              f"(committed locally, will push at end)")
        return True  # committed locally even if push failed

    except subprocess.CalledProcessError as e:
        print(f"      [auto-commit] Git error for {iso} {threshold}%: {e}")
        return False


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
    parser.add_argument(
        "--auto-commit", action="store_true",
        help="Commit and push each threshold's parquet as it completes "
             "(prevents work loss on freeze/timeout)",
    )
    args = parser.parse_args()

    iso = args.iso.upper()
    if iso not in s1.ISOS:
        print(f"ERROR: Unknown ISO '{iso}'. Valid: {', '.join(s1.ISOS)}")
        sys.exit(1)

    auto_commit = args.auto_commit
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
    print(f"  Auto-commit: {'ON — each threshold committed as it completes' if auto_commit else 'OFF'}")
    print("=" * 70)

    # ── Resume logic: hash code+data, load manifest ──
    code_hash = _compute_code_hash(iso)
    manifest = _load_manifest(iso, code_hash)
    phase1_done = set(manifest.get('phase1', [])) if manifest else set()
    phase2_done = set(manifest.get('phase2', [])) if manifest else set()

    if phase1_done:
        print(f"\n  Resuming — {len(phase1_done)} Phase 1 thresholds already done "
              f"(hash {code_hash[:8]})")
    if phase2_done:
        print(f"  Resuming — {len(phase2_done)} Phase 2 thresholds already done")

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
    full_coarse_combos, full_coarse_scores = cached
    full_coarse_procurement = full_coarse_combos.sum(axis=1)
    print(f"  Total coarse cache: {len(full_coarse_combos):,} rows")

    os.makedirs(s1.STEP1_RAW_PFS_PARQUET_DIR, exist_ok=True)

    # Override THRESHOLDS so optimize_threshold uses only requested set
    original_thresholds = list(s1.THRESHOLDS)
    s1.THRESHOLDS = thresholds

    # Sort ascending for consistent processing order
    thresholds_sorted = sorted(thresholds)

    # ── Phase 1: Coarse storage sweep ──
    print(f"\nPhase 1 — Coarse storage sweep ({len(thresholds_sorted)} thresholds, ascending)")
    total_start = time.time()

    # Track storage saturation incrementally — don't keep full result lists
    max_bat4 = max_bat8 = max_ldes = max_h2 = 0.0

    for threshold in thresholds_sorted:
        if threshold in phase1_done:
            # Verify the parquet actually exists (manifest says done but file missing = recompute)
            done_path = s1._threshold_done_path(iso, threshold)
            if os.path.exists(done_path):
                print(f"    {iso} {threshold}%: skipped (already done)")
                continue
            else:
                print(f"    {iso} {threshold}%: manifest says done but parquet missing — recomputing")

        # Pre-filter coarse cache to procurement window for this threshold.
        # optimize_threshold() does this internally too, but pre-filtering here
        # avoids passing the full 1.6M-row cache into the function, reducing
        # peak memory on GitHub Actions runners (~7 GB RAM).
        # Match the procurement window logic in optimize_threshold():
        proc_lower = threshold
        if threshold < 50:
            proc_upper = max(threshold + 30, 50)
        elif threshold <= 75:
            proc_upper = 2.0 * threshold
        elif threshold <= 90:
            proc_upper = 200.0
        else:
            proc_upper = 250.0

        proc_mask = full_coarse_procurement >= proc_lower
        if proc_upper is not None:
            proc_mask &= full_coarse_procurement <= proc_upper
        t_combos = full_coarse_combos[proc_mask]
        t_scores = full_coarse_scores[proc_mask]
        print(f"    {iso} {threshold}%: pre-filtered to {len(t_combos):,} / "
              f"{len(full_coarse_combos):,} coarse combos "
              f"(procurement [{proc_lower}%, {proc_upper}%])")

        t_start = time.time()
        feasible = s1.optimize_threshold(
            iso, threshold, demand_arr, supply_matrix,
            t_combos, t_scores)
        del t_combos, t_scores  # free filtered copy immediately

        # Within-threshold dominance post-filter: remove mixes where a
        # leaner mix (all resources <=) also hits this same target.
        n_before_dom = len(feasible)
        feasible = s1.dominance_filter_candidates(feasible, rtypes)
        n_removed = n_before_dom - len(feasible)

        archetypes = set()
        for c in feasible:
            archetypes.add(tuple(c['resource_mix'][rt] for rt in rtypes))
            max_bat4 = max(max_bat4, c.get('battery_dispatch_pct', 0) or 0)
            max_bat8 = max(max_bat8, c.get('battery8_dispatch_pct', 0) or 0)
            max_ldes = max(max_ldes, c.get('ldes_dispatch_pct', 0) or 0)
            max_h2 = max(max_h2, c.get('h2_dispatch_pct', 0) or 0)

        t_elapsed = time.time() - t_start
        dom_str = f", {n_removed} dominated removed" if n_removed > 0 else ""
        print(f"    {iso} {threshold}%: {len(feasible)} solutions "
              f"({len(archetypes)} archetypes{dom_str}), {t_elapsed:.1f}s")

        # Merge chunk files + any remaining in-memory candidates into final parquet.
        # optimize_threshold() flushes candidates to chunk files during processing,
        # so `feasible` is typically empty — _merge_chunks_and_finalize reads the
        # chunks back and produces the final {ISO}_t{XX}_raw_pfs.parquet.
        s1._merge_chunks_and_finalize(iso, threshold, feasible)

        # Record this threshold as done and persist manifest
        phase1_done.add(threshold)
        _save_manifest(iso, code_hash, phase1_done, phase2_done)

        # Auto-commit this threshold's results immediately
        if auto_commit:
            git_commit_threshold(iso, threshold, "Phase1")

        del feasible, archetypes  # free memory before next threshold
        gc.collect()

    print(f"\n  Coarse saturation — bat4={max_bat4:.2f}%, bat8={max_bat8:.2f}%, "
          f"ldes={max_ldes:.1f}%, h2={max_h2:.1f}%")

    # ── Phase 2: Fine storage sweep within saturation range ──
    fine_levels = s1.build_fine_storage_levels(
        max_bat4, max_bat8, max_ldes, max_h2)

    if fine_levels is not None:
        print(f"\nPhase 2 — Fine storage sweep")

        for threshold in thresholds_sorted:
            if threshold in phase2_done:
                done_path = s1._threshold_done_path(iso, threshold)
                if os.path.exists(done_path):
                    print(f"    {iso} {threshold}%: skipped (already done)")
                    continue
                else:
                    print(f"    {iso} {threshold}%: manifest says done but parquet missing — recomputing")

            # Pre-filter coarse cache for this threshold (same window as Phase 1)
            proc_lower = threshold
            if threshold < 50:
                proc_upper = max(threshold + 30, 50)
            elif threshold <= 75:
                proc_upper = 2.0 * threshold
            elif threshold <= 90:
                proc_upper = 200.0
            else:
                proc_upper = 250.0
            proc_mask = full_coarse_procurement >= proc_lower
            if proc_upper is not None:
                proc_mask &= full_coarse_procurement <= proc_upper
            t_combos = full_coarse_combos[proc_mask]
            t_scores = full_coarse_scores[proc_mask]

            t_start = time.time()
            feasible = s1.optimize_threshold(
                iso, threshold, demand_arr, supply_matrix,
                t_combos, t_scores,
                storage_levels=fine_levels)
            del t_combos, t_scores

            # Within-threshold dominance post-filter
            feasible = s1.dominance_filter_candidates(feasible, rtypes)

            archetypes = set()
            for c in feasible:
                archetypes.add(tuple(c['resource_mix'][rt] for rt in rtypes))

            t_elapsed = time.time() - t_start
            print(f"    {iso} {threshold}%: {len(feasible)} solutions "
                  f"({len(archetypes)} archetypes), {t_elapsed:.1f}s")

            # Merge chunk files + remaining candidates into final parquet
            s1._merge_chunks_and_finalize(iso, threshold, feasible)

            # Record this threshold as done and persist manifest
            phase2_done.add(threshold)
            _save_manifest(iso, code_hash, phase1_done, phase2_done)

            # Auto-commit this threshold's updated results immediately
            if auto_commit:
                git_commit_threshold(iso, threshold, "Phase2")
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
