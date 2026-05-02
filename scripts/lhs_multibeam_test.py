#!/usr/bin/env python3
"""
LHS × beam cross-reference test for Step 2.3 pathway optimizer.

Usage:
  python3 -u scripts/lhs_multibeam_test.py ERCOT A 5000 4
  python3 -u scripts/lhs_multibeam_test.py ERCOT B 10000 8 --archetype nuclear-heavy
  python3 -u scripts/lhs_multibeam_test.py ERCOT A 2500 2 --archetype balanced

Without --archetype: seeds are selected across all archetypes (one per arch,
cheapest first — original behavior).

With --archetype: seeds are selected within that single archetype via greedy
maximin on normalized resource+storage feature vector.  A pool of max(N, 8)
seeds is selected; --beams N takes the first N, so pools are nested across
beam counts sharing the same archetype and LHS.
"""
from __future__ import annotations
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import json, sys, time, argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy.spatial.distance import cdist
import step2_3_reliability_tax_adaptive as solver
import pipeline_config as pc

KEY_THRESHOLDS = [90.0, 95.0, 99.0, 99.9]
KEY_MAP = {t: f"t{int(t) if t == int(t) else t}" for t in KEY_THRESHOLDS}


# ── Shared pool loading (vectorized costing) ───────────────────────────────

def _load_valid_pool(iso, cfg, P32, dm32, dn32):
    """Load EF band, score, cost.  Returns shared arrays for seed selectors."""
    seed_band = solver.SEED_BAND_BY_ISO[iso]
    ef = solver.load_ef_band(iso, seed_band)
    if ef is None:
        return None

    n = ef.num_rows
    tbl_names = set(ef.schema.names)
    W = np.zeros((n, solver.N_RESOURCES), dtype=np.float32)
    for i, res in enumerate(solver.RESOURCE_ORDER):
        if res in tbl_names:
            W[:, i] = ef.column(res).to_numpy().astype(np.float32) * 0.01

    def _col(name):
        return (ef.column(name).to_numpy().astype(np.float32)
                if name in tbl_names else np.zeros(n, np.float32))

    batt4 = _col("battery_dispatch_pct")
    batt8 = _col("battery8_dispatch_pct")
    ldes  = _col("ldes_dispatch_pct")
    scores, resid, _ = solver.score_candidates(W, P32, dm32, dn32,
                                                batt4, batt8, ldes)

    next_thresh = 100.0
    for t in solver.EF_BAND_THRESHOLDS:
        if t > seed_band:
            next_thresh = t
            break
    mask = (scores >= seed_band - 0.5) & (scores < next_thresh)
    valid_idx = np.where(mask)[0]
    if len(valid_idx) == 0:
        base_pct = sum(pc.GRID_MIX_SHARES[iso].values())
        valid_idx = np.where(scores >= base_pct)[0]
    if len(valid_idx) == 0:
        return None

    # Vectorized costing — single matmul instead of 21k × 11 Python loop
    base_dem = float(pc.REGIONAL_DEMAND_TWH[iso])
    lcoe_vec = np.array([solver.get_resource_lcoe(iso, res, solver.BASE_YEAR, cfg)
                         for res in solver.RESOURCE_ORDER], dtype=np.float64)
    stor_vec = np.array([solver.storage_net_cost(iso, sc, cfg, year=solver.BASE_YEAR)
                         for sc in solver.STORAGE_COLS], dtype=np.float64)

    W_valid = W[valid_idx].astype(np.float64)           # already 0–1 scale
    S_valid = np.column_stack([batt4[valid_idx],
                               batt8[valid_idx],
                               ldes[valid_idx]]).astype(np.float64) / 100.0
    costs = (W_valid @ lcoe_vec + S_valid @ stor_vec) * base_dem * 1e6

    print(f"[seeds] {len(valid_idx):,} mixes in valid band")
    return dict(W=W, batt4=batt4, batt8=batt8, ldes=ldes,
                scores=scores, resid=resid, valid_idx=valid_idx, costs=costs)


# ── Seed selection: single archetype (greedy maximin + cdist) ──────────────

def select_seeds_single_archetype(pool, cfg, target_arch, n_beams):
    """Greedy maximin within one archetype.  Selects max(n_beams, 8) seeds,
    returns first n_beams so pools nest across beam counts."""
    W         = pool["W"]
    batt4     = pool["batt4"]
    batt8     = pool["batt8"]
    ldes      = pool["ldes"]
    scores    = pool["scores"]
    resid     = pool["resid"]
    valid_idx = pool["valid_idx"]
    costs     = pool["costs"]

    arch_ki = []
    for ki, vi in enumerate(valid_idx):
        mix = {res: float(W[vi, ri]) * 100.0
               for ri, res in enumerate(solver.RESOURCE_ORDER)}
        if solver.classify_archetype(mix, cfg.pathway, iso=cfg.iso) == target_arch:
            arch_ki.append(ki)

    if not arch_ki:
        print(f"[seeds] 0 mixes in archetype '{target_arch}'")
        return []

    print(f"[seeds] {len(arch_ki):,} mixes in archetype '{target_arch}'")

    feats = np.zeros((len(arch_ki), solver.N_RESOURCES + solver.N_STORAGE),
                     dtype=np.float64)
    for fi, ki in enumerate(arch_ki):
        vi = valid_idx[ki]
        for ri in range(solver.N_RESOURCES):
            feats[fi, ri] = float(W[vi, ri]) * 100.0
        feats[fi, solver.N_RESOURCES]     = float(batt4[vi])
        feats[fi, solver.N_RESOURCES + 1] = float(batt8[vi])
        feats[fi, solver.N_RESOURCES + 2] = float(ldes[vi])

    stds = feats.std(axis=0)
    stds[stds < 1e-6] = 1.0
    feats_norm = feats / stds

    pool_size = max(n_beams, 8)
    k = min(pool_size, len(arch_ki))

    # Greedy maximin with cdist
    arch_costs = costs[arch_ki]
    selected = [int(np.argmin(arch_costs))]
    for _ in range(k - 1):
        dists = cdist(feats_norm, feats_norm[selected]).min(axis=1)
        dists[selected] = -1.0
        selected.append(int(np.argmax(dists)))

    seeds = _build_seed_dicts(pool, cfg, arch_ki, selected[:n_beams], target_arch)
    print(f"[seeds] selected {len(seeds)} '{target_arch}' seeds "
          f"(CFE {min(s['cfe'] for s in seeds):.1f}–"
          f"{max(s['cfe'] for s in seeds):.1f}%)")
    return seeds


# ── Seed selection: all archetypes (original behavior) ─────────────────────

def select_seeds_all_archetypes(pool, cfg, n_beams):
    """One cheapest seed per archetype, sorted by cost — original behavior."""
    W         = pool["W"]
    valid_idx = pool["valid_idx"]
    costs     = pool["costs"]

    archetypes = {}
    for ki, vi in enumerate(valid_idx):
        mix = {res: float(W[vi, ri]) * 100.0
               for ri, res in enumerate(solver.RESOURCE_ORDER)}
        arch = solver.classify_archetype(mix, cfg.pathway, iso=cfg.iso)
        if arch not in archetypes or costs[ki] < archetypes[arch][1]:
            archetypes[arch] = (ki, costs[ki])

    ranked = sorted(archetypes, key=lambda a: archetypes[a][1])
    ki_list = [archetypes[a][0] for a in ranked[:n_beams]]
    arch_list = ranked[:n_beams]

    seeds = _build_seed_dicts(pool, cfg,
                               list(range(len(valid_idx))),  # identity map
                               ki_list, None)
    # Patch archetype names back (identity map means ki == index into arch_ki)
    for s, a in zip(seeds, arch_list):
        s["archetype"] = a

    print(f"[seeds] selected {len(seeds)} seeds across "
          f"{len(set(s['archetype'] for s in seeds))} archetypes")
    return seeds


def _build_seed_dicts(pool, cfg, arch_ki, selected_indices, archetype):
    """Build seed dicts from selected indices into arch_ki."""
    W         = pool["W"]
    batt4     = pool["batt4"]
    batt8     = pool["batt8"]
    ldes      = pool["ldes"]
    scores    = pool["scores"]
    resid     = pool["resid"]
    valid_idx = pool["valid_idx"]
    costs     = pool["costs"]

    seeds = []
    for si in selected_indices:
        ki = arch_ki[si]
        vi = valid_idx[ki]
        seeds.append({
            "archetype": archetype or "unknown",
            "resource_pcts": {res: float(W[vi, ri]) * 100.0
                              for ri, res in enumerate(solver.RESOURCE_ORDER)},
            "storage_pcts": {
                solver.STORAGE_COLS[0]: float(batt4[vi]),
                solver.STORAGE_COLS[1]: float(batt8[vi]),
                solver.STORAGE_COLS[2]: float(ldes[vi]),
            },
            "cfe": float(scores[vi]),
            "resid": float(resid[vi]),
            "cost": float(costs[ki]),
        })
    return seeds


# ── Parallel beam solver ──────────────────────────────────────────────────

def _solve_one_beam(seed, cfg, P32, dm32, dn32, bi):
    """Worker: solve a single beam.  Each spawned process gets clean JIT."""
    solver.warmup_jit()
    t0 = time.time()
    result = solver.solve_pathway(seed, cfg, P32, dm32, dn32, beam_idx=bi)
    dt = time.time() - t0

    snaps = {s["threshold_pct"]: s
             for s in result.get("threshold_snapshots", [])}
    row = {"beam_idx": bi, "archetype": seed["archetype"],
           "wall_s": round(dt, 1),
           "peak_gas_mw": result["peak_gas_mw"],
           "peak_h2_mw": result["peak_h2_mw"]}
    for t, k in KEY_MAP.items():
        if t in snaps:
            row[f"{k}_cost_B"] = round(
                snaps[t]["cumulative_cost_usd"] / 1e9, 4)
            row[f"{k}_year"] = snaps[t]["year_achieved"]
        else:
            row[f"{k}_cost_B"] = None
    return row


def run(iso, pathway, lhs_size, n_beams, archetype=None):
    solver.warmup_jit()
    P  = solver.build_profile_matrix(iso)
    dn = solver._load_demand_norm(iso)
    dm = dn * (1.0 + solver.RA_MARGIN)
    P32, dm32, dn32 = (P.astype(np.float32), dm.astype(np.float32),
                        dn.astype(np.float32))

    cfg = solver.RunConfig(
        iso=iso, pathway=pathway, scenario_name="base",
        demand_growth="Medium", n_beams=n_beams,
        stage2_samples=lhs_size, cost_mode=2,
        **solver.COST_SCENARIOS["base"],
    )

    pool = _load_valid_pool(iso, cfg, P32, dm32, dn32)
    if pool is None:
        print(json.dumps({"error": "no valid mixes"}))
        return

    if archetype:
        seeds = select_seeds_single_archetype(pool, cfg, archetype, n_beams)
    else:
        seeds = select_seeds_all_archetypes(pool, cfg, n_beams)

    if not seeds:
        print(json.dumps({"error": "no seeds"}))
        return

    for bi, seed in enumerate(seeds):
        print(f"  beam {bi}: {seed['archetype']} CFE={seed['cfe']:.1f}%",
              flush=True)

    t_total = time.time()
    workers = min(len(seeds), max(1, multiprocessing.cpu_count() - 1))
    with ProcessPoolExecutor(max_workers=workers) as exe:
        futures = [exe.submit(_solve_one_beam, seed, cfg, P32, dm32, dn32, bi)
                   for bi, seed in enumerate(seeds)]
        beam_results = [f.result() for f in futures]

    wall = round(time.time() - t_total, 1)

    for row in beam_results:
        print(f"  beam {row['beam_idx']}: {row['wall_s']:.1f}s | "
              f"99.9%={row.get('t99.9_cost_B', 'N/A')}", flush=True)

    best = {}
    for t, k in KEY_MAP.items():
        vals = [(r[f"{k}_cost_B"], r["beam_idx"])
                for r in beam_results if r[f"{k}_cost_B"] is not None]
        if vals:
            vals.sort()
            best[k] = {"best_B": vals[0][0], "worst_B": vals[-1][0],
                        "spread_pct": round(
                            (vals[-1][0] - vals[0][0]) / vals[0][0] * 100, 1)
                        if vals[0][0] else 0}

    print("RESULT_JSON:" + json.dumps({
        "iso": iso, "pathway": pathway,
        "archetype": archetype or "all",
        "lhs": lhs_size, "n_beams": len(seeds), "wall_s": wall,
        "archetypes": [s["archetype"] for s in seeds],
        "best": best, "beams": beam_results,
    }))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="LHS × beam cross-reference test for Step 2.3")
    ap.add_argument("iso", help="ISO region (e.g. ERCOT)")
    ap.add_argument("pathway", help="Pathway letter (A, B, ...)")
    ap.add_argument("lhs", type=int, help="LHS sample count per solver step")
    ap.add_argument("beams", type=int, help="Number of beams")
    ap.add_argument("--archetype", default=None,
                    help="Single archetype for within-arch maximin seeding. "
                         "Omit for cross-archetype (one seed per arch).")
    args = ap.parse_args()
    run(args.iso.upper(), args.pathway.upper(), args.lhs, args.beams,
        args.archetype)
