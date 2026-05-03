#!/usr/bin/env python3
"""
Sobol × beam cross-reference test for Step 2.3 pathway optimizer.

Mirrors lhs_multibeam_test.py but uses the Sobol-based solver
(step2_3_adaptive_sobol) to test whether Sobol's nested sequence property
produces better or more predictable cost improvements at higher sample
and beam counts.

Usage:
  python3 -u scripts/sobol_multibeam_test.py ERCOT A 5000 4 --archetype balanced
  python3 -u scripts/sobol_multibeam_test.py ERCOT A 10000 8 --archetype balanced
  python3 -u scripts/sobol_multibeam_test.py ERCOT B 20000 16 --archetype nuclear-heavy
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
import step2_3_adaptive_sobol as solver
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

    W_valid = W[valid_idx].astype(np.float64)
    S_valid = np.column_stack([batt4[valid_idx],
                               batt8[valid_idx],
                               ldes[valid_idx]]).astype(np.float64) / 100.0
    costs = (W_valid @ lcoe_vec + S_valid @ stor_vec) * base_dem * 1e6

    print(f"[seeds] {len(valid_idx):,} mixes in valid band")
    return dict(W=W, batt4=batt4, batt8=batt8, ldes=ldes,
                scores=scores, resid=resid, valid_idx=valid_idx, costs=costs)


# ── Feature matrix construction ────────────────────────────────────────────

def _build_feature_matrix(pool, ki_list):
    """Build normalized feature matrix for a subset of valid-pool indices.
    Returns (feats_norm, stds) with shape (len(ki_list), N_RESOURCES + N_STORAGE).
    """
    W         = pool["W"]
    batt4     = pool["batt4"]
    batt8     = pool["batt8"]
    ldes      = pool["ldes"]
    valid_idx = pool["valid_idx"]

    n_feat = solver.N_RESOURCES + solver.N_STORAGE
    feats = np.zeros((len(ki_list), n_feat), dtype=np.float64)
    for fi, ki in enumerate(ki_list):
        vi = valid_idx[ki]
        for ri in range(solver.N_RESOURCES):
            feats[fi, ri] = float(W[vi, ri]) * 100.0
        feats[fi, solver.N_RESOURCES]     = float(batt4[vi])
        feats[fi, solver.N_RESOURCES + 1] = float(batt8[vi])
        feats[fi, solver.N_RESOURCES + 2] = float(ldes[vi])

    stds = feats.std(axis=0)
    stds[stds < 1e-6] = 1.0
    feats_norm = feats / stds
    return feats_norm


# ── Greedy maximin seed selection ──────────────────────────────────────────

def _maximin_select(feats_norm, costs, k):
    """Greedy maximin: start with cheapest, then iteratively pick the point
    farthest from all already-selected points.  Returns list of indices into
    feats_norm/costs arrays (length = min(k, len(feats_norm))).
    """
    n = len(feats_norm)
    k = min(k, n)
    if k == 0:
        return []

    selected = [int(np.argmin(costs))]
    if k == 1:
        return selected

    # Precompute: track min distance to selected set for each point
    min_dists = cdist(feats_norm, feats_norm[selected]).ravel()  # (n,)

    for _ in range(k - 1):
        min_dists[selected[-1]] = -1.0  # exclude last selected
        best = int(np.argmax(min_dists))
        selected.append(best)
        # Update min_dists: for each point, take min of current and dist to new selection
        new_dists = np.linalg.norm(feats_norm - feats_norm[best], axis=1)
        min_dists = np.minimum(min_dists, new_dists)
        min_dists[selected[-1]] = -1.0

    return selected


# ── Classify all valid mixes by archetype ──────────────────────────────────

def _classify_pool(pool, cfg):
    """Returns dict: archetype_name -> list of ki (indices into valid_idx)."""
    W         = pool["W"]
    valid_idx = pool["valid_idx"]

    arch_groups = {}
    for ki, vi in enumerate(valid_idx):
        mix = {res: float(W[vi, ri]) * 100.0
               for ri, res in enumerate(solver.RESOURCE_ORDER)}
        arch = solver.classify_archetype(mix, cfg.pathway, iso=cfg.iso)
        if arch not in arch_groups:
            arch_groups[arch] = []
        arch_groups[arch].append(ki)

    return arch_groups


# ── Seed selection: single archetype ───────────────────────────────────────

def select_seeds_single_archetype(pool, cfg, target_arch, n_beams):
    """Maximin diversification within one archetype.  Selects from a pool of
    max(n_beams, 8) so that pools are nested across beam counts."""
    arch_groups = _classify_pool(pool, cfg)

    arch_ki = arch_groups.get(target_arch, [])
    if not arch_ki:
        print(f"[seeds] 0 mixes in archetype '{target_arch}'")
        return []

    print(f"[seeds] {len(arch_ki):,} mixes in archetype '{target_arch}'")

    feats_norm = _build_feature_matrix(pool, arch_ki)
    arch_costs = pool["costs"][arch_ki]

    pool_size = max(n_beams, 8)
    selected = _maximin_select(feats_norm, arch_costs, pool_size)

    seeds = _build_seed_dicts(pool, cfg, arch_ki, selected[:n_beams], target_arch)
    print(f"[seeds] selected {len(seeds)} '{target_arch}' seeds "
          f"(CFE {min(s['cfe'] for s in seeds):.1f}–"
          f"{max(s['cfe'] for s in seeds):.1f}%)")
    return seeds


# ── Seed selection: all archetypes with diversification ────────────────────

def select_seeds_all_archetypes(pool, cfg, n_beams):
    """n_beams diverse seeds PER archetype, via maximin within each.

    Every archetype gets exactly n_beams seeds (capped at archetype pool
    size).  Total beams = n_beams × num_archetypes.
    """
    arch_groups = _classify_pool(pool, cfg)
    costs = pool["costs"]

    # Sort archetypes by pool size descending (largest pool = most to explore)
    ranked_archs = sorted(arch_groups.keys(),
                          key=lambda a: len(arch_groups[a]), reverse=True)

    n_archs = len(ranked_archs)
    print(f"[seeds] {n_archs} archetypes: "
          + ", ".join(f"{a} ({len(arch_groups[a]):,})" for a in ranked_archs))

    # Every archetype gets n_beams (capped at pool size)
    allocation = {a: min(n_beams, len(arch_groups[a])) for a in ranked_archs}

    alloc_str = ", ".join(f"{a}: {allocation[a]}" for a in ranked_archs)
    print(f"[seeds] beam allocation: {alloc_str} "
          f"({n_beams}/arch × {n_archs} archs = {sum(allocation.values())} total)")

    # Select seeds within each archetype using maximin
    all_seeds = []
    for arch in ranked_archs:
        if arch not in allocation:
            continue
        n_arch_beams = allocation[arch]
        arch_ki = arch_groups[arch]

        feats_norm = _build_feature_matrix(pool, arch_ki)
        arch_costs = costs[arch_ki]

        selected = _maximin_select(feats_norm, arch_costs, n_arch_beams)

        seeds = _build_seed_dicts(pool, cfg, arch_ki, selected, arch)
        print(f"[seeds]   {arch}: {len(seeds)} seeds "
              f"(CFE {min(s['cfe'] for s in seeds):.1f}–"
              f"{max(s['cfe'] for s in seeds):.1f}%)")
        all_seeds.extend(seeds)

    return all_seeds


# ── Build seed dicts ───────────────────────────────────────────────────────

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


def run(iso, pathway, sample_size, n_beams, archetype=None):
    solver.warmup_jit()
    P  = solver.build_profile_matrix(iso)
    dn = solver._load_demand_norm(iso)
    dm = dn * (1.0 + solver.RA_MARGIN)
    P32, dm32, dn32 = (P.astype(np.float32), dm.astype(np.float32),
                        dn.astype(np.float32))

    cfg = solver.RunConfig(
        iso=iso, pathway=pathway, scenario_name="base",
        demand_growth="Medium", n_beams=n_beams,
        stage2_samples=sample_size, cost_mode=2,
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

    # Per-archetype summary
    arch_best = {}
    for row in beam_results:
        a = row["archetype"]
        if a not in arch_best:
            arch_best[a] = {"count": 0}
        arch_best[a]["count"] += 1
        for t, k in KEY_MAP.items():
            cost = row.get(f"{k}_cost_B")
            if cost is not None:
                prev = arch_best[a].get(f"{k}_best_B")
                if prev is None or cost < prev:
                    arch_best[a][f"{k}_best_B"] = cost

    # Write output JSON with 'sobol' tag
    out_dir = Path(__file__).resolve().parent.parent / "data" / "sampling-tests"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    arch_tag = archetype or "all"
    fname = (f"{iso}__pw{pathway}__sobol{sample_size}__beams{n_beams}"
             f"__{arch_tag}__{ts}.json")

    result_obj = {
        "iso": iso, "pathway": pathway,
        "archetype": arch_tag,
        "strategy": "sobol",
        "sampler": "sobol",
        "sample_size": sample_size, "beams_per_arch": n_beams,
        "n_beams_total": len(seeds), "wall_s": wall,
        "archetypes": [s["archetype"] for s in seeds],
        "archetype_beam_counts": {a: v["count"] for a, v in arch_best.items()},
        "archetype_best": arch_best,
        "best": best, "beams": beam_results,
    }

    with open(out_dir / fname, "w") as f:
        json.dump(result_obj, f, indent=2)
    print(f"\n[output] {fname}")

    print("RESULT_JSON:" + json.dumps(result_obj))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Sobol × beam cross-reference test for Step 2.3")
    ap.add_argument("iso", help="ISO region (e.g. ERCOT)")
    ap.add_argument("pathway", help="Pathway letter (A, B, ...)")
    ap.add_argument("sample_size", type=int,
                    help="Sobol sample count per solver step")
    ap.add_argument("beams", type=int,
                    help="Number of beams PER archetype")
    ap.add_argument("--archetype", default=None,
                    help="Single archetype for within-arch maximin seeding. "
                         "Omit for cross-archetype diversified seeding.")
    args = ap.parse_args()
    run(args.iso.upper(), args.pathway.upper(), args.sample_size, args.beams,
        args.archetype)
