#!/usr/bin/env python3
"""In-process pathway sweep driver — keeps the python interpreter hot
across all (pathway, endpoint) combos for a single ISO.

Why this exists
---------------
`run_pathway_sweep.py` spawns a fresh Python subprocess per run. Profiling
with cProfile (Apr 17 2026) showed:

    Per-run wall-clock: 5.7 s
      1.6 s  — load_dispatch_cache  (step3-dispatch/{ISO}_dispatch_cache.parquet)
      0.9 s  — load_generation_profiles (EIA hourly profiles)
      1.7 s  — load_ef_mixes via read_parquet  (Step 2.1 EF parquets)
      1.0 s  — numpy.ndarray.copy inside _worst_hour_residual_norm
      0.5 s  — everything else (actual per-year pathway compute)

The first three costs are one-time loads that step_2_3 already memoizes
WITHIN a python process (`_WH_STATE`, `_EF_CACHE`). But the subprocess
architecture discards those caches after every run. Keeping the process
alive for all 60 runs per ISO cuts per-run time from 5.7 s → ~1 s.

Design
------
- One process per ISO; all 60 (pathway, endpoint) combos run in-process
  via ``run_pathway(RunConfig, ...)``.
- Pathway 3 runs first (SPEC §24.4 Card K' — comparative-to-P3 stranding).
- Within a pathway, endpoints chain in ascending order (ep60 seeds ep70,
  etc.) via ``initial_ledger`` / ``initial_fleet`` loaded from the prior
  endpoint's JSON.
- MANIFEST.json is written atomically by ``run_pathway`` after each run,
  so a sandbox-kill mid-sweep is recoverable: re-running picks up at the
  first missing run_key.
- 7 ISOs parallelised by launching 7 copies of this driver concurrently
  from the outer shell; each worker is independent (no shared memory).
"""
from __future__ import annotations

import argparse
import os
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
    _load_ledger_from_json,
    _load_fleet_from_json,
)

# Run Pathway 3 first per SPEC §24.4 Card K' so later pathways have the
# comparative-stranding reference. Other pathways follow alphabetically.
_PATHWAY_ORDER = ('3', '1', '1a', '1b', '2a', '2b')


def _endpoint_tag(ep: float) -> str:
    return f"{ep * 100:g}".replace('.', 'p')


def sweep_one_iso(iso: str, output_root: Path,
                  endpoints=None, pathways=None, verbose: bool = True) -> None:
    """Run every (pathway, endpoint) for one ISO in a single Python process.

    Skip-logic: if the output JSON already exists for a (pathway, endpoint),
    we treat it as completed and load its terminal ledger/fleet as the seed
    for the NEXT endpoint in the same pathway (so the chain stays intact
    even on resumption).
    """
    endpoints = tuple(endpoints or ALL_ENDPOINTS)
    pathways = tuple(pathways or _PATHWAYS_ALL)
    pathways_sorted = tuple(
        p for p in _PATHWAY_ORDER if p in pathways
    )

    total_runs = len(pathways_sorted) * len(endpoints)
    t_start = time.time()
    completed = 0
    skipped = 0
    n = 0

    for pathway in pathways_sorted:
        # Track the previous endpoint's output path for ledger/fleet seeding.
        prior_path: Path | None = None
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
                    print(f"[{iso}] [{n}/{total_runs}] {pathway}@{ep:g} "
                          f"SKIP (exists) — seed for next", flush=True)
                prior_path = out
                skipped += 1
                continue

            initial_ledger = None
            initial_fleet = None
            if prior_path is not None and prior_path.exists():
                initial_ledger = _load_ledger_from_json(prior_path)
                initial_fleet = _load_fleet_from_json(prior_path, iso)

            t0 = time.time()
            try:
                result = run_pathway(
                    cfg,
                    initial_ledger=initial_ledger,
                    initial_fleet=initial_fleet,
                )
            except Exception as exc:
                elapsed = time.time() - t0
                print(f"[{iso}] [{n}/{total_runs}] {pathway}@{ep:g} "
                      f"FAILED in {elapsed:.2f}s: {exc!r}", flush=True)
                # Do not set prior_path — next endpoint seeds from the last
                # successful run.
                continue

            elapsed = time.time() - t0
            achieved = result.get('achieved_cfe_pct')
            if verbose:
                print(f"[{iso}] [{n}/{total_runs}] {pathway}@{ep:g} "
                      f"OK  cfe={achieved:.2f}%  in {elapsed:.2f}s", flush=True)
            prior_path = out
            completed += 1

    total_elapsed = time.time() - t_start
    print(f"[{iso}] DONE — completed={completed} skipped={skipped} "
          f"of {total_runs} in {total_elapsed:.1f}s "
          f"({total_elapsed/max(1, completed):.2f}s per new run)", flush=True)


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

    sweep_one_iso(
        iso=args.iso,
        output_root=Path(args.output_root),
        endpoints=endpoints,
        pathways=pathways,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
