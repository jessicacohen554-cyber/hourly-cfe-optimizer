#!/usr/bin/env python3
"""In-process pathway sweep driver — keeps the python interpreter hot
across all (pathway, endpoint) combos for a single ISO.

Why this exists
---------------
`run_pathway_sweep.py` spawns a fresh Python subprocess per run, paying
~5 seconds of numba JIT + parquet-load + module-import overhead every
time. ``run_pathway``'s internal caches (``_WH_STATE``, ``_EF_CACHE``,
``_LEARNING_CURVE_CACHE``, dispatch cache) are discarded on subprocess
exit. Keeping the process alive for all 50 runs per ISO collapses the
cold-start tax to a single payment per ISO — profiled ~5.7 s → ~0.5–1 s
per run.

Design (post SPEC §24.6 — each run independent)
-----------------------------------------------
- One process per ISO; every (pathway, endpoint) combo runs in-process
  via ``run_pathway(cfg)`` with NO cross-endpoint seeding. Per §24.6,
  each (pathway, endpoint) trajectory is a standalone 2025–2050 scenario.
  ``initial_ledger`` and ``initial_fleet`` are both unused here.
- Pathway 3 runs first (SPEC §24.4 Card K' — comparative-to-P3 stranding
  requires the P3 run to exist when the 1/1a/2a/2b ``run_pathway`` call
  reads it back for the stranding-ledger computation). P3 runs are
  submitted to the ThreadPool first; other pathways are submitted only
  after all P3 futures resolve, ensuring P3 files are on disk.
- Per-run JSON files are written atomically by ``write_run_json`` so a
  sandbox-kill mid-sweep is recoverable: re-running picks up at the first
  missing output file. Any existing output JSON is treated as done and
  skipped.
- MANIFEST.json is written ONCE per ISO worker at the end of the sweep
  (``_batch_append_to_manifest``), collapsing all per-run fcntl RMW
  operations into one locked write. The fcntl lock still serialises
  concurrent ISO workers so the cross-ISO MANIFEST is never corrupted.
- Within an ISO worker, 4 threads run concurrently via ThreadPoolExecutor.
  Caches (_WH_STATE, _EF_CACHE) are shared in-memory; prewarm_caches()
  fully populates them single-threaded before any thread is spawned so
  threaded runs are read-only against the hot path.
- 7 ISOs parallelise by launching 7 copies of this driver concurrently
  from the outer shell; each worker is independent (no shared memory,
  no contention beyond the git-backed dispatch-cache parquets which
  grow monotonically).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import sys
import threading
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from step_2_3_pathway_optimizer import (
    ENDPOINTS as ALL_ENDPOINTS,
    PATHWAYS as _PATHWAYS_ALL,
    RunConfig,
    _batch_append_to_manifest,
    run_pathway,
)
from step2_2a_cost_optimization import (
    flush_expanded_cache as _flush_wh_cache,
    _get_worst_hour_state as _init_wh_state,
    prewarm_caches,
)

# Run Pathway 3 first per SPEC §24.4 Card K' so later pathways have the
# comparative-stranding reference. Other pathways follow alphabetically.
_PATHWAY_ORDER = ('3', '1', '1a', '1b', '2a', '2b')


def _endpoint_tag(ep: float) -> str:
    return f"{ep * 100:g}".replace('.', 'p')


def sweep_one_iso(
    iso: str,
    output_root: Path,
    endpoints=None,
    pathways=None,
    verbose: bool = True,
) -> dict:
    """Run every (pathway, endpoint) for one ISO in a single Python process.

    Returns a summary dict with {completed, skipped, failed, total, seconds}.
    """
    endpoints = tuple(endpoints or ALL_ENDPOINTS)
    pathways = tuple(pathways or _PATHWAYS_ALL)
    pathways_sorted = tuple(p for p in _PATHWAY_ORDER if p in pathways)

    # Pre-assign sequential display indices so threaded prints are informative.
    combos = [(pw, ep) for pw in pathways_sorted for ep in sorted(endpoints)]
    n_by_combo = {(pw, ep): i + 1 for i, (pw, ep) in enumerate(combos)}
    total_runs = len(combos)
    t_start = time.time()

    # Prewarm: load dispatch cache and EF parquets for the thresholds this
    # sweep will actually touch.  Archetype expansion is deferred to first-miss
    # inside each thread (thread-safe via _WH_STATE_LOCK).
    print(f"[sweep] {iso}: prewarming caches ({len(combos)} combos)", flush=True)
    prewarm_caches(iso, combos)

    # Mark the ISO state as deferred so per-run flush_expanded_cache calls
    # inside run_pathway are no-ops; we flush once in the finally block.
    _iso_state = _init_wh_state(iso)
    _iso_state['deferred'] = True

    # Accumulate manifest entries in-memory; one locked disk write per ISO.
    manifest_acc: dict[str, dict] = {}
    manifest_lock = threading.Lock()

    # Counters — written only from the main thread after futures resolve.
    completed = 0
    skipped = 0
    failed = 0

    def _run_one(pathway: str, ep: float) -> tuple[str, str, float, dict | None, Exception | None]:
        """Execute one (pathway, endpoint) run; return (status, pathway, ep, result, exc)."""
        n = n_by_combo[(pathway, ep)]
        cfg = RunConfig(
            iso=iso,
            pathway=pathway,
            endpoint=ep,
            output_root=output_root,
        )
        if cfg.output_path.exists():
            if verbose:
                print(
                    f"[{iso}] [{n}/{total_runs}] {pathway}@{ep:g} SKIP (exists)",
                    flush=True,
                )
            return ('skip', pathway, ep, None, None)

        t0 = time.time()
        try:
            # skip_manifest=True: suppress per-run fcntl RMW; we batch-write
            # all ISO entries once at end of this function.  §24.6 — no cross-
            # endpoint seeding.
            result = run_pathway(cfg, skip_manifest=True)
        except Exception as exc:
            elapsed = time.time() - t0
            print(
                f"[{iso}] [{n}/{total_runs}] {pathway}@{ep:g} "
                f"FAILED in {elapsed:.2f}s: {exc!r}",
                flush=True,
            )
            return ('fail', pathway, ep, None, exc)

        elapsed = time.time() - t0
        achieved = result.get('achieved_cfe_pct')
        if verbose:
            print(
                f"[{iso}] [{n}/{total_runs}] {pathway}@{ep:g} "
                f"OK  cfe={achieved:.2f}%  in {elapsed:.2f}s",
                flush=True,
            )
        entry = result.get('_manifest_entry')
        if entry is not None:
            with manifest_lock:
                manifest_acc[result['run_key']] = entry
        p3_entry = result.get('_p3_manifest_entry')
        if p3_entry is not None:
            p3_run_key = result['pathway3_reference_run_key']
            with manifest_lock:
                manifest_acc[p3_run_key] = p3_entry
        return ('ok', pathway, ep, result, None)

    try:
        # Phase 1 — Pathway 3 runs (must land on disk before other pathways
        # can read them back for comparative-stranding via _load_ledger_from_json).
        p3_combos = [(pw, ep) for pw, ep in combos if pw == '3']
        other_combos = [(pw, ep) for pw, ep in combos if pw != '3']

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            # Submit P3 runs first and wait for all to complete.
            p3_futures = [pool.submit(_run_one, pw, ep) for pw, ep in p3_combos]
            for fut in concurrent.futures.as_completed(p3_futures):
                status, pw, ep, _, _ = fut.result()
                if status == 'ok':
                    completed += 1
                elif status == 'skip':
                    skipped += 1
                else:
                    failed += 1

            # Submit remaining pathways now that P3 results are on disk.
            other_futures = [pool.submit(_run_one, pw, ep) for pw, ep in other_combos]
            for fut in concurrent.futures.as_completed(other_futures):
                status, pw, ep, _, _ = fut.result()
                if status == 'ok':
                    completed += 1
                elif status == 'skip':
                    skipped += 1
                else:
                    failed += 1

    finally:
        # Single end-of-ISO flush: clear deferred and persist any new
        # archetype expansions accumulated during the threaded runs.
        _iso_state['deferred'] = False
        _flush_wh_cache(iso)
        # Batch-write all accumulated manifest entries in one locked RMW.
        if manifest_acc:
            _batch_append_to_manifest(output_root, manifest_acc)

    total_elapsed = time.time() - t_start
    per_run = total_elapsed / max(1, completed)
    print(
        f"[{iso}] DONE — completed={completed} skipped={skipped} "
        f"failed={failed} of {total_runs} in {total_elapsed:.1f}s "
        f"({per_run:.2f}s per new run)",
        flush=True,
    )
    return {
        'iso': iso,
        'completed': completed,
        'skipped': skipped,
        'failed': failed,
        'total': total_runs,
        'seconds': total_elapsed,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description='In-process pathway sweep driver (single ISO).',
    )
    ap.add_argument('--iso', required=True)
    ap.add_argument(
        '--output-root',
        default='analysis/reliability-tax/data',
        help='Root directory for per-run JSON outputs and MANIFEST.json.',
    )
    ap.add_argument(
        '--endpoints',
        default=','.join(str(e) for e in ALL_ENDPOINTS),
        help='Comma-separated endpoints (default: all ten).',
    )
    ap.add_argument(
        '--pathways',
        default=','.join(_PATHWAYS_ALL),
        help='Comma-separated pathways (default: all six).',
    )
    args = ap.parse_args()

    endpoints = tuple(float(e.strip()) for e in args.endpoints.split(',') if e.strip())
    pathways = tuple(p.strip() for p in args.pathways.split(',') if p.strip())

    summary = sweep_one_iso(
        iso=args.iso,
        output_root=Path(args.output_root),
        endpoints=endpoints,
        pathways=pathways,
    )
    return 0 if summary['failed'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
