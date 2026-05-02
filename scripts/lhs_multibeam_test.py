#!/usr/bin/env python3
"""Usage: python3 -u scripts/lhs_multibeam_test.py ERCOT A 2500 4"""
from __future__ import annotations
import json, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import step2_3_reliability_tax_adaptive as solver
import pipeline_config as pc

KEY_THRESHOLDS = [90.0, 95.0, 99.0, 99.9]
KEY_MAP = {t: f"t{int(t) if t == int(t) else t}" for t in KEY_THRESHOLDS}


def _solve_one_beam(seed, cfg, P32, dm32, dn32, bi):
    """Worker: solve a single beam and return its result row."""
    solver.warmup_jit()                       # each forked process needs its own JIT cache
    t0 = time.time()
    result = solver.solve_pathway(seed, cfg, P32, dm32, dn32, beam_idx=bi)
    dt = time.time() - t0
    snaps = {s["threshold_pct"]: s for s in result.get("threshold_snapshots", [])}
    row = {"beam_idx": bi, "archetype": seed["archetype"], "wall_s": round(dt, 1),
           "peak_gas_mw": result["peak_gas_mw"], "peak_h2_mw": result["peak_h2_mw"]}
    for t, k in KEY_MAP.items():
        if t in snaps:
            row[f"{k}_cost_B"] = round(snaps[t]["cumulative_cost_usd"] / 1e9, 4)
            row[f"{k}_year"] = snaps[t]["year_achieved"]
        else:
            row[f"{k}_cost_B"] = None
    return row


def run(iso, pathway, lhs_size, n_beams):
    solver.warmup_jit()
    P = solver.build_profile_matrix(iso)
    dn = solver._load_demand_norm(iso)
    dm = dn * (1.0 + solver.RA_MARGIN)
    P32, dm32, dn32 = P.astype(np.float32), dm.astype(np.float32), dn.astype(np.float32)
    cfg = solver.RunConfig(iso=iso, pathway=pathway, scenario_name="base",
        demand_growth="Medium", n_beams=n_beams, stage2_samples=lhs_size,
        cost_mode=2, **solver.COST_SCENARIOS["base"])
    seeds = solver.stage1_select(iso, cfg, P32, dm32, dn32)
    if not seeds:
        print(json.dumps({"error": "no seeds"})); return

    for bi, seed in enumerate(seeds):
        print(f"  beam {bi}: {seed['archetype']} CFE={seed['cfe']:.1f}%", flush=True)

    t_total = time.time()
    with ProcessPoolExecutor(max_workers=len(seeds)) as pool:
        futures = [pool.submit(_solve_one_beam, seed, cfg, P32, dm32, dn32, bi)
                   for bi, seed in enumerate(seeds)]
        beam_results = [f.result() for f in futures]

    # Print per-beam timings (after all complete, preserving order)
    for row in beam_results:
        print(f"  beam {row['beam_idx']}: {row['wall_s']:.1f}s | "
              f"99.9%={row.get('t99.9_cost_B', 'N/A')}", flush=True)

    best = {}
    for t, k in KEY_MAP.items():
        costs = [(r[f"{k}_cost_B"], r["beam_idx"], r["archetype"])
                 for r in beam_results if r[f"{k}_cost_B"] is not None]
        if costs:
            costs.sort()
            best[k] = {"best_B": costs[0][0], "worst_B": costs[-1][0],
                       "best_arch": costs[0][2], "spread_pct": round(
                       (costs[-1][0] - costs[0][0]) / costs[0][0] * 100, 1) if costs[0][0] else 0}
    print("RESULT_JSON:" + json.dumps({"iso": iso, "pathway": pathway,
        "lhs": lhs_size, "n_beams": len(seeds), "wall_s": round(time.time() - t_total, 1),
        "archetypes": [s["archetype"] for s in seeds],
        "best": best, "beams": beam_results}), flush=True)


if __name__ == "__main__":
    run(sys.argv[1].upper(), sys.argv[2].upper(), int(sys.argv[3]), int(sys.argv[4]))
