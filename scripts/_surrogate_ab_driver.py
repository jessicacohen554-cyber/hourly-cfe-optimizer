#!/usr/bin/env python3
"""SPEC §24.5 surrogate A/B driver — ERCOT four-combo validation sweep.

Runs (pathway, endpoint) ∈ {(1,0.80), (1,0.95), (3,0.80), (3,0.95)} twice
each: once with USE_SURROGATE_RESIDUAL_IN_ARGMIN=False (truth, via
true 99.97-percentile dispatch) and once with the flag True (surrogate,
via HistGBM). Writes truth JSONs to the repo data dir and surrogate
JSONs to a side directory, then moves the surrogate files into
`pathway{P}_ep{EE}.surrogate.json` so the truth outputs are not
clobbered. Records per-run wallclock for speedup comparison.

Usage:
    python3 scripts/_surrogate_ab_driver.py

Emits one JSON summary to stdout when done. Downstream metric computation
lives in `scripts/_surrogate_ab_metrics.py`.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import pipeline_config as pc
from step_2_3_pathway_optimizer import RunConfig, run_pathway
from step2_2a_cost_optimization import (
    flush_expanded_cache as _flush_wh_cache,
    _get_worst_hour_state as _init_wh_state,
    prewarm_caches,
)

REPO_ROOT = _THIS_DIR.parent
TRUTH_ROOT = REPO_ROOT / 'analysis' / 'reliability-tax' / 'data'
SURR_ROOT = REPO_ROOT / 'analysis' / 'reliability-tax' / '_tmp_surrogate_ab'
FINAL_SURR_DIR = TRUTH_ROOT / 'ERCOT'

# Four A/B combos (pathway, endpoint).
COMBOS = (
    ('3', 0.80),
    ('1', 0.80),
    ('3', 0.95),
    ('1', 0.95),
)
ISO = 'ERCOT'


def _endpoint_tag(ep: float) -> str:
    return f"{ep * 100:g}".replace('.', 'p')


def _run_one(pathway: str, endpoint: float, output_root: Path,
             use_surrogate: bool) -> dict:
    pc.USE_SURROGATE_RESIDUAL_IN_ARGMIN = bool(use_surrogate)

    cfg = RunConfig(iso=ISO, pathway=pathway, endpoint=endpoint,
                    output_root=output_root)

    # If the target output already exists, remove it so the run is fresh
    # (we want honest wallclock, not a skip).
    if cfg.output_path.exists():
        cfg.output_path.unlink()

    label = 'SURR' if use_surrogate else 'TRU '
    print(f"[ab] {label} pathway={pathway} ep={endpoint:g} ...", flush=True)

    t0 = time.perf_counter()
    result = run_pathway(cfg, skip_manifest=True)
    wallclock_s = time.perf_counter() - t0

    print(f"[ab] {label} pathway={pathway} ep={endpoint:g} "
          f"cfe={result['achieved_cfe_pct']:.2f}% in {wallclock_s:.2f}s",
          flush=True)
    return {
        'pathway': pathway,
        'endpoint': endpoint,
        'wallclock_s': wallclock_s,
        'output_path': result['output_path'],
    }


def main() -> int:
    SURR_ROOT.mkdir(parents=True, exist_ok=True)
    (SURR_ROOT / ISO).mkdir(parents=True, exist_ok=True)
    FINAL_SURR_DIR.mkdir(parents=True, exist_ok=True)

    # Prewarm dispatch + EF caches for the two thresholds this sweep touches.
    # Threshold-based prewarm keyed by (pathway, endpoint) — we only need
    # the EF data for the two endpoints.
    warm_combos = [(p, ep) for p, ep in COMBOS]
    print(f"[ab] prewarming caches for {ISO} ({len(warm_combos)} combos)", flush=True)
    prewarm_caches(ISO, warm_combos)

    # Defer archetype-expansion writes to disk until the very end.
    iso_state = _init_wh_state(ISO)
    iso_state['deferred'] = True

    timings = {}
    try:
        for pathway, endpoint in COMBOS:
            truth_res = _run_one(pathway, endpoint, TRUTH_ROOT,
                                 use_surrogate=False)
            surr_res = _run_one(pathway, endpoint, SURR_ROOT,
                                use_surrogate=True)
            timings[(pathway, endpoint)] = {
                'wallclock_truth_s': truth_res['wallclock_s'],
                'wallclock_surrogate_s': surr_res['wallclock_s'],
                'truth_path': truth_res['output_path'],
                'surrogate_tmp_path': surr_res['output_path'],
            }
    finally:
        # End-of-sweep flush — one archetype-expansion write for the full sweep.
        iso_state['deferred'] = False
        _flush_wh_cache(ISO)
        # Reset flag to the committed default so the driver leaves the module
        # state clean.
        pc.USE_SURROGATE_RESIDUAL_IN_ARGMIN = False

    # Move surrogate JSONs into the canonical .surrogate.json slots.
    moved = {}
    for (pathway, endpoint), t in timings.items():
        tag = _endpoint_tag(endpoint)
        src = Path(t['surrogate_tmp_path'])
        dst = FINAL_SURR_DIR / f"pathway{pathway}_ep{tag}.surrogate.json"
        shutil.move(str(src), str(dst))
        moved[(pathway, endpoint)] = {
            **t,
            'surrogate_path': str(dst),
        }

    # Emit machine-readable summary.
    summary = {
        'iso': ISO,
        'combos': [
            {
                'pathway': p,
                'endpoint': ep,
                'wallclock_truth_s': round(moved[(p, ep)]['wallclock_truth_s'], 3),
                'wallclock_surrogate_s': round(moved[(p, ep)]['wallclock_surrogate_s'], 3),
                'truth_path': moved[(p, ep)]['truth_path'],
                'surrogate_path': moved[(p, ep)]['surrogate_path'],
            }
            for p, ep in COMBOS
        ],
    }
    summary_path = SURR_ROOT / 'ab_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[ab] summary written → {summary_path}", flush=True)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
