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
  reads it back for the stranding-ledger computation).
- MANIFEST.json is written atomically by ``run_pathway`` after each run,
  so a sandbox-kill mid-sweep is recoverable: re-running picks up at the
  first missing run_key. Any existing output JSON is treated as done
  and skipped (no --force here; pass ``--force`` to the parent sweep
  orchestrator if you need to overwrite).
- 7 ISOs parallelise by launching 7 copies of this driver concurrently
  from the outer shell; each worker is independent (no shared memory,
  no contention beyond the git-backed dispatch-cache parquets which
  grow monotonically).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from step_2_3_pathway_optimizer import (
    ENDPOINTS as ALL_ENDPOINTS,
    PATHWAYS as _PATHWAYS_ALL,
    RunConfig,
    run_pathway,
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

    total_runs = len(pathways_sorted) * len(endpoints)
    t_start = time.time()
    completed = 0
    skipped = 0
    failed = 0
    n = 0

    for pathway in pathways_sorted:
        for ep in sorted(endpoints):
            n += 1
            cfg = RunConfig(
                iso=iso,
                pathway=pathway,
                endpoint=ep,
                output_root=output_root,
            )
            out = cfg.output_path
            if out.exists():
                if verbose:
                    print(
                        f"[{iso}] [{n}/{total_runs}] {pathway}@{ep:g} SKIP (exists)",
                        flush=True,
                    )
                skipped += 1
                continue

            t0 = time.time()
            try:
                result = run_pathway(cfg)  # §24.6 — no cross-endpoint seeding.
            except Exception as exc:
                elapsed = time.time() - t0
                print(
                    f"[{iso}] [{n}/{total_runs}] {pathway}@{ep:g} "
                    f"FAILED in {elapsed:.2f}s: {exc!r}",
                    flush=True,
                )
                failed += 1
                continue

            elapsed = time.time() - t0
            achieved = result.get('achieved_cfe_pct')
            if verbose:
                print(
                    f"[{iso}] [{n}/{total_runs}] {pathway}@{ep:g} "
                    f"OK  cfe={achieved:.2f}%  in {elapsed:.2f}s",
                    flush=True,
                )
            completed += 1

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
