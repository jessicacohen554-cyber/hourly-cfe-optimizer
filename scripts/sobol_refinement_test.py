#!/usr/bin/env python3
"""
Sobol + local refinement A/B test for Step 2.3 pathway optimizer.

Tests whether Nelder-Mead local refinement after each Sobol winner selection
improves pathway cost.  Three arms:

  baseline:   Sobol N samples, no refinement
  refined:    Sobol N samples + Nelder-Mead polish per step
  low_refined: Sobol N/4 samples + Nelder-Mead polish per step

The critical question: does 10k+refinement match or beat 40k raw?

Per-step diagnostics logged:
  - cost_before / cost_after  ($ delta from refinement)
  - mix shift per dimension   (escape radius)
  - whether tech mix changed  (same resources or different)
  - CFE before / after

Usage:
  python3 -u scripts/sobol_refinement_test.py ERCOT A 40000 8 --archetype balanced
  python3 -u scripts/sobol_refinement_test.py ERCOT A 40000 8 --archetype balanced --radius 3.0
"""
from __future__ import annotations
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy.optimize import minimize

import step2_3_adaptive_sobol as solver
import pipeline_config as pc


# ── Refinement wrapper ────────────────────────────────────────────────────

REFINEMENT_LOG = []  # populated per-beam, reset between beams


def _make_refined_find_winner(radius_pct: float, max_iter: int = 80):
    """Build a patched _find_winner that applies Nelder-Mead after Sobol."""

    original_fw = solver._find_winner.__wrapped__ if hasattr(
        solver._find_winner, '__wrapped__') else solver._find_winner

    def _refined_find_winner(
        floor_pcts, floor_storage, target, ceiling,
        floor_cfe, res_yields, stor_yields,
        free_idx, cfg, P32, dm32, dn32,
        year, dem_twh,
        beam_idx=0, existing_gas_mw=0.0, gaf=1.0,
        prev_gas_cost=0.0, current_peak_gas=0.0,
    ):
        # Step 1: get Sobol winner via original
        result = original_fw(
            floor_pcts, floor_storage, target, ceiling,
            floor_cfe, res_yields, stor_yields,
            free_idx, cfg, P32, dm32, dn32,
            year, dem_twh,
            beam_idx=beam_idx, existing_gas_mw=existing_gas_mw, gaf=gaf,
            prev_gas_cost=prev_gas_cost, current_peak_gas=current_peak_gas,
        )
        winner_pcts, winner_storage, winner_cfe, winner_resid, h2_info = result
        if winner_pcts is None:
            return result

        # Step 2: build objective function for refinement
        capex_yr = solver.h2_peaker_capex_kw_yr(year, cfg)
        fuel_mwh_cost = solver.h2_peaker_fuel_mwh(year, cfg)

        n_free = len(free_idx)
        n_dims = n_free + solver.N_STORAGE

        # Pack winner into flat vector: [free_res_pcts..., storage_pcts...]
        x0 = np.zeros(n_dims)
        for k, ri in enumerate(free_idx):
            x0[k] = winner_pcts[ri]
        for j in range(solver.N_STORAGE):
            x0[n_free + j] = winner_storage[j]

        # Bounds: ±radius around winner, clamped to [floor, physical_max]
        bounds = []
        for k, ri in enumerate(free_idx):
            lo = max(floor_pcts[ri], x0[k] - radius_pct)
            hi = min(200.0, x0[k] + radius_pct)
            bounds.append((lo, hi))
        for j in range(solver.N_STORAGE):
            lo = max(floor_storage[j], x0[n_free + j] - radius_pct)
            hi = min(solver._STORAGE_TYPE_CAPS[j],
                     x0[n_free + j] + radius_pct)
            bounds.append((lo, hi))

        eval_count = [0]

        def objective(x):
            eval_count[0] += 1
            pcts = floor_pcts.copy()
            stor = np.zeros(solver.N_STORAGE, dtype=np.float64)
            for k, ri in enumerate(free_idx):
                pcts[ri] = x[k]
            for j in range(solver.N_STORAGE):
                stor[j] = x[n_free + j]

            cfe, resid_val, _ = solver._score_single(pcts, stor, P32, dm32, dn32)

            # Must be in valid CFE range
            min_pre_h2 = max(target - solver.H2_MAX_FILL_PP, floor_cfe)
            if cfe < min_pre_h2 or cfe >= ceiling:
                return 1e18

            # Resource cost
            c = 0.0
            for k, ri in enumerate(free_idx):
                delta = pcts[ri] - floor_pcts[ri]
                if delta > 0:
                    c += (delta / 100.0 * dem_twh
                          * solver.get_resource_lcoe(
                              cfg.iso, solver.RESOURCE_ORDER[ri], year, cfg)
                          * 1e6)
            for j, sc in enumerate(solver.STORAGE_COLS):
                delta = stor[j] - floor_storage[j]
                if delta > 0:
                    c += (delta / 100.0
                          * solver.storage_net_cost(cfg.iso, sc, cfg, year=year)
                          * dem_twh * 1e6)

            # H2 sizing
            if cfe >= target:
                h2_cost = 0.0
                post_resid = resid_val
            else:
                W_row = np.zeros(solver.N_RESOURCES, dtype=np.float32)
                W_row[:] = (pcts * 0.01).astype(np.float32)
                cfe_gaps, ra_gaps, _ = solver._compute_post_storage_gaps(
                    W_row, P32, dn32, dm32,
                    float(stor[0]), float(stor[1]), float(stor[2]))
                h2r = solver.h2_size_for_target(cfe_gaps, ra_gaps, cfe,
                                                target, dem_twh)
                if h2r["h2_capacity_mw"] >= np.inf:
                    return 1e18
                h2_cost = (h2r["h2_capacity_mw"] * capex_yr * 1000
                           + h2r["h2_dispatched_mwh"] * fuel_mwh_cost)
                post_resid = h2r["post_h2_resid"]
            c += h2_cost

            # Gas stranding shadow
            if solver.STRANDING_SHADOW_WEIGHT > 0.0:
                g_mw = solver.gas_need_mw(post_resid, dem_twh,
                                          existing_gas_mw, gaf)
                shadow = solver.gas_stranding_shadow(
                    g_mw, current_peak_gas, year, cfg.iso)
                c += shadow

            # Gas savings (mode 2)
            if cfg.cost_mode == 2 and prev_gas_cost > 0:
                g_mw = solver.gas_need_mw(post_resid, dem_twh,
                                          existing_gas_mw, gaf)
                g_cost = solver.gas_annual_cost(g_mw, cfg.iso, cfg)
                c -= (prev_gas_cost - g_cost)

            return c

        # Step 3: run Nelder-Mead (gradient-free, handles non-smooth H2 sizing)
        cost_before = objective(x0)
        t0 = time.time()
        opt = minimize(objective, x0, method="Nelder-Mead",
                       bounds=bounds,
                       options={"maxiter": max_iter, "xatol": 0.01,
                                "fatol": 1e4, "adaptive": True})
        refine_wall = time.time() - t0

        # Step 4: check if refined solution is actually better
        x_refined = opt.x
        cost_after = opt.fun

        if cost_after < cost_before - 1e3:
            # Accept refinement — unpack back to pcts/storage
            refined_pcts = floor_pcts.copy()
            refined_storage = np.zeros(solver.N_STORAGE, dtype=np.float64)
            for k, ri in enumerate(free_idx):
                refined_pcts[ri] = x_refined[k]
            for j in range(solver.N_STORAGE):
                refined_storage[j] = x_refined[n_free + j]

            # Re-score to get final CFE and H2 info
            ref_cfe, ref_resid, _ = solver._score_single(
                refined_pcts, refined_storage, P32, dm32, dn32)

            if ref_cfe >= target:
                ref_h2 = {"h2_capacity_mw": 0.0, "h2_dispatched_mwh": 0.0,
                           "h2_cf_pct": 0.0, "h2_dispatch_hours": 0,
                           "post_h2_resid": ref_resid}
                post_cfe = ref_cfe
            else:
                W_row = np.zeros(solver.N_RESOURCES, dtype=np.float32)
                W_row[:] = (refined_pcts * 0.01).astype(np.float32)
                cfe_gaps, ra_gaps, _ = solver._compute_post_storage_gaps(
                    W_row, P32, dn32, dm32,
                    float(refined_storage[0]), float(refined_storage[1]),
                    float(refined_storage[2]))
                ref_h2 = solver.h2_size_for_target(
                    cfe_gaps, ra_gaps, ref_cfe, target, dem_twh)
                post_cfe = ref_cfe
                if ref_h2["h2_dispatched_mwh"] > 0 and dem_twh > 0:
                    post_cfe += (ref_h2["h2_dispatched_mwh"]
                                 / (dem_twh * 1e6) * 100.0)

            # Compute escape radius per dimension
            escape = np.abs(x_refined - x0)
            mix_changed = any(escape[k] > 0.5 for k in range(n_free))

            log_entry = {
                "year": year, "target": target,
                "cost_before": round(cost_before / 1e9, 6),
                "cost_after": round(cost_after / 1e9, 6),
                "cost_delta_pct": round(
                    (cost_before - cost_after) / max(abs(cost_before), 1) * 100, 3),
                "cfe_before": round(float(winner_cfe), 2),
                "cfe_after": round(float(post_cfe), 2),
                "escape_radius": {
                    solver.RESOURCE_ORDER[free_idx[k]]: round(float(escape[k]), 3)
                    for k in range(n_free)
                },
                "escape_storage": {
                    solver.STORAGE_COLS[j]: round(float(escape[n_free + j]), 4)
                    for j in range(solver.N_STORAGE)
                },
                "mix_changed": mix_changed,
                "nelder_mead_evals": eval_count[0],
                "refine_wall_s": round(refine_wall, 2),
                "accepted": True,
            }
            REFINEMENT_LOG.append(log_entry)

            return (refined_pcts, refined_storage, post_cfe,
                    ref_h2["post_h2_resid"], ref_h2)
        else:
            # Refinement didn't improve — keep Sobol winner
            log_entry = {
                "year": year, "target": target,
                "cost_before": round(cost_before / 1e9, 6),
                "cost_after": round(cost_after / 1e9, 6),
                "cost_delta_pct": 0.0,
                "cfe_before": round(float(winner_cfe), 2),
                "cfe_after": round(float(winner_cfe), 2),
                "escape_radius": {},
                "escape_storage": {},
                "mix_changed": False,
                "nelder_mead_evals": eval_count[0],
                "refine_wall_s": round(refine_wall, 2),
                "accepted": False,
            }
            REFINEMENT_LOG.append(log_entry)
            return result

    _refined_find_winner.__wrapped__ = original_fw
    return _refined_find_winner


# ── Seed loading (reuse from sobol_multibeam_test) ────────────────────────

def load_seeds(iso, cfg, P32, dm32, dn32, n_beams, archetype=None):
    """Load and select seeds — mirrors sobol_multibeam_test logic."""
    # Import seed selection from the multibeam test
    import sobol_multibeam_test as smt
    pool = smt._load_valid_pool(iso, cfg, P32, dm32, dn32)
    if pool is None:
        return []
    if archetype and archetype != "all":
        return smt.select_seeds_single_archetype(pool, cfg, archetype, n_beams)
    else:
        return smt.select_seeds_all_archetypes(pool, cfg, n_beams)


# ── Single beam solver (one arm) ─────────────────────────────────────────

KEY_THRESHOLDS = [90.0, 95.0, 99.0, 99.9]
KEY_MAP = {t: f"t{int(t) if t == int(t) else t}" for t in KEY_THRESHOLDS}


def _solve_beam(seed, cfg, P32, dm32, dn32, bi, refine, radius):
    """Solve one beam with or without refinement."""
    global REFINEMENT_LOG
    REFINEMENT_LOG = []

    solver.warmup_jit()

    if refine:
        solver._find_winner = _make_refined_find_winner(radius)
    elif hasattr(solver._find_winner, '__wrapped__'):
        solver._find_winner = solver._find_winner.__wrapped__

    t0 = time.time()
    result = solver.solve_pathway(seed, cfg, P32, dm32, dn32, beam_idx=bi)
    wall = time.time() - t0

    snaps = {s["threshold_pct"]: s for s in result.get("threshold_snapshots", [])}
    row = {
        "beam_idx": bi,
        "archetype": seed["archetype"],
        "wall_s": round(wall, 1),
        "peak_gas_mw": result["peak_gas_mw"],
        "peak_h2_mw": result["peak_h2_mw"],
        "refinement_log": list(REFINEMENT_LOG),
    }
    for t, k in KEY_MAP.items():
        if t in snaps:
            row[f"{k}_cost_B"] = round(snaps[t]["cumulative_cost_usd"] / 1e9, 4)
            row[f"{k}_year"] = snaps[t]["year_achieved"]
            row[f"{k}_mix"] = snaps[t]["resource_mix_pct"]
            row[f"{k}_storage"] = snaps[t]["storage_dispatch_pct"]
        else:
            row[f"{k}_cost_B"] = None
    return row


# ── Main ──────────────────────────────────────────────────────────────────

def _load_baseline_from_file(path: str) -> dict:
    """Load an existing sobol multibeam test JSON as the baseline arm."""
    with open(path) as f:
        d = json.load(f)

    # Map existing beam format to our arm beam format
    beams = []
    for b in d.get("beams", []):
        row = {
            "beam_idx": b["beam_idx"],
            "archetype": b.get("archetype", "unknown"),
            "wall_s": b.get("wall_s", 0),
            "peak_gas_mw": b.get("peak_gas_mw", 0),
            "peak_h2_mw": b.get("peak_h2_mw", 0),
            "refinement_log": [],
        }
        for t, k in KEY_MAP.items():
            row[f"{k}_cost_B"] = b.get(f"{k}_cost_B")
            row[f"{k}_year"] = b.get(f"{k}_year")
        # t99.9 key in existing files uses dot notation
        if row.get("t99.9_cost_B") is None and "t99.9_cost_B" in b:
            row["t99.9_cost_B"] = b["t99.9_cost_B"]
            row["t99.9_year"] = b.get("t99.9_year")
        beams.append(row)

    # Build best summary
    best = {}
    for t, k in KEY_MAP.items():
        vals = [(r[f"{k}_cost_B"], r["beam_idx"])
                for r in beams if r.get(f"{k}_cost_B") is not None]
        if vals:
            vals.sort()
            best[k] = {
                "best_B": vals[0][0],
                "worst_B": vals[-1][0],
                "best_beam": vals[0][1],
            }

    return {
        "samples": d.get("sample_size", 0),
        "refine": False,
        "radius": 0.0,
        "n_beams": len(beams),
        "wall_s": d.get("wall_s", 0),
        "best": best,
        "beams": beams,
        "source_file": path,
        "refinement_summary": {
            "total_steps": 0, "accepted": 0, "rejected": 0,
            "avg_cost_delta_pct": 0.0, "avg_nelder_mead_evals": 0,
            "avg_refine_wall_s": 0.0,
        },
    }


def run(iso, pathway, sample_size, n_beams, archetype, radius,
        baseline_file=None):
    solver.warmup_jit()
    P = solver.build_profile_matrix(iso)
    dn = solver._load_demand_norm(iso)
    dm = dn * (1.0 + solver.RA_MARGIN)
    P32, dm32, dn32 = (P.astype(np.float32), dm.astype(np.float32),
                        dn.astype(np.float32))

    all_results = {}

    # ── Baseline: load from file or run fresh ─────────────────────────
    if baseline_file:
        print(f"\n{'='*60}")
        print(f"  ARM: baseline  (loaded from {baseline_file})")
        print(f"{'='*60}")
        all_results["baseline"] = _load_baseline_from_file(baseline_file)
        c99 = all_results["baseline"]["best"].get("t99.9", {}).get("best_B", "N/A")
        print(f"  {all_results['baseline']['n_beams']} beams, "
              f"99.9%=${c99}")
    else:
        arms_to_run = {"baseline": {"samples": sample_size, "refine": False}}
        _run_arms(arms_to_run, all_results, iso, pathway, n_beams,
                  archetype, radius, P32, dm32, dn32)

    # ── Refinement arms: always run fresh ─────────────────────────────
    refine_arms = {
        "refined": {"samples": sample_size, "refine": True},
        "low_refined": {"samples": sample_size // 4, "refine": True},
    }
    _run_arms(refine_arms, all_results, iso, pathway, n_beams,
              archetype, radius, P32, dm32, dn32)

    # ── Summary comparison ────────────────────────────────────────────
    _print_summary(all_results)

    # ── Write output ──────────────────────────────────────────────────
    _write_output(all_results, iso, pathway, sample_size, n_beams,
                  archetype, radius, baseline_file)


def _run_arms(arms, all_results, iso, pathway, n_beams, archetype,
              radius, P32, dm32, dn32):
    """Run one or more arms and append to all_results."""
    for arm_name, arm_cfg in arms.items():
        print(f"\n{'='*60}")
        print(f"  ARM: {arm_name}  (samples={arm_cfg['samples']}, "
              f"refine={arm_cfg['refine']}, radius={radius})")
        print(f"{'='*60}")

        cfg = solver.RunConfig(
            iso=iso, pathway=pathway, scenario_name="base",
            demand_growth="Medium", n_beams=n_beams,
            stage2_samples=arm_cfg["samples"], cost_mode=2,
            **solver.COST_SCENARIOS["base"],
        )

        seeds = load_seeds(iso, cfg, P32, dm32, dn32, n_beams, archetype)
        if not seeds:
            print(f"  [SKIP] no seeds for {arm_name}")
            continue

        t_arm = time.time()
        beam_results = []
        for bi, seed in enumerate(seeds):
            print(f"\n  beam {bi}: {seed['archetype']} CFE={seed['cfe']:.1f}%")
            row = _solve_beam(seed, cfg, P32, dm32, dn32, bi,
                              arm_cfg["refine"], radius)
            beam_results.append(row)
            c99 = row.get("t99.9_cost_B", "N/A")
            n_accepted = sum(1 for e in row["refinement_log"] if e["accepted"])
            n_total = len(row["refinement_log"])
            print(f"    99.9%=${c99} | wall={row['wall_s']:.1f}s"
                  f" | refine={n_accepted}/{n_total} accepted")
        arm_wall = round(time.time() - t_arm, 1)

        # Summarize arm
        best = {}
        for t, k in KEY_MAP.items():
            vals = [(r[f"{k}_cost_B"], r["beam_idx"])
                    for r in beam_results if r[f"{k}_cost_B"] is not None]
            if vals:
                vals.sort()
                best[k] = {
                    "best_B": vals[0][0],
                    "worst_B": vals[-1][0],
                    "best_beam": vals[0][1],
                }

        # Aggregate refinement stats
        all_logs = [e for r in beam_results for e in r["refinement_log"]]
        n_steps = len(all_logs)
        n_accepted = sum(1 for e in all_logs if e["accepted"])
        avg_delta = (np.mean([e["cost_delta_pct"] for e in all_logs if e["accepted"]])
                     if n_accepted > 0 else 0.0)
        avg_evals = (np.mean([e["nelder_mead_evals"] for e in all_logs])
                     if n_steps > 0 else 0)
        avg_refine_s = (np.mean([e["refine_wall_s"] for e in all_logs])
                        if n_steps > 0 else 0)

        all_results[arm_name] = {
            "samples": arm_cfg["samples"],
            "refine": arm_cfg["refine"],
            "radius": radius,
            "n_beams": len(seeds),
            "wall_s": arm_wall,
            "best": best,
            "beams": beam_results,
            "refinement_summary": {
                "total_steps": n_steps,
                "accepted": n_accepted,
                "rejected": n_steps - n_accepted,
                "avg_cost_delta_pct": round(avg_delta, 4),
                "avg_nelder_mead_evals": round(avg_evals, 1),
                "avg_refine_wall_s": round(avg_refine_s, 2),
            },
        }


def _print_summary(all_results):
    print(f"\n{'='*60}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Arm':<16} {'Samples':>8} {'99.9%$B':>10} {'95%$B':>9} "
          f"{'Wall':>7} {'Accept':>7}")
    print("-" * 62)
    for arm_name, arm in all_results.items():
        c99 = arm["best"].get("t99.9", {}).get("best_B", "N/A")
        c95 = arm["best"].get("t95", {}).get("best_B", "N/A")
        c99_s = f"{c99:.1f}" if isinstance(c99, float) else c99
        c95_s = f"{c95:.1f}" if isinstance(c95, float) else c95
        rs = arm["refinement_summary"]
        acc = f"{rs['accepted']}/{rs['total_steps']}" if rs["total_steps"] else "-"
        print(f"{arm_name:<16} {arm['samples']:>8} {c99_s:>10} {c95_s:>9} "
              f"{arm['wall_s']:>7.0f} {acc:>7}")


def _write_output(all_results, iso, pathway, sample_size, n_beams,
                  archetype, radius, baseline_file):
    out_dir = Path(__file__).resolve().parent.parent / "data" / "sampling-tests"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    arch_tag = archetype or "all"
    fname = (f"{iso}__pw{pathway}__refinement_test__s{sample_size}"
             f"__b{n_beams}__r{radius}__{arch_tag}__{ts}.json")

    output = {
        "iso": iso,
        "pathway": pathway,
        "archetype": arch_tag,
        "base_sample_size": sample_size,
        "beams_per_arch": n_beams,
        "refinement_radius": radius,
        "baseline_source": baseline_file or "fresh_run",
        "timestamp": ts,
        "arms": all_results,
    }

    # Strip refinement_log from beams for compact output
    compact = copy.deepcopy(output)
    for arm in compact["arms"].values():
        for beam in arm.get("beams", []):
            beam.pop("refinement_log", None)

    with open(out_dir / fname, "w") as f:
        json.dump(compact, f, indent=2)
    print(f"\n[output] {fname}")

    # Full logs (with per-step refinement data)
    log_fname = fname.replace(".json", "__logs.json")
    log_data = {}
    for arm_name, arm in all_results.items():
        log_data[arm_name] = [
            {"beam_idx": b["beam_idx"], "refinement_log": b.get("refinement_log", [])}
            for b in arm.get("beams", [])
        ]
    with open(out_dir / log_fname, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"[output] {log_fname}")

    print("RESULT_JSON:" + json.dumps(compact))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Sobol + local refinement A/B test for Step 2.3")
    ap.add_argument("iso", help="ISO region (e.g. ERCOT)")
    ap.add_argument("pathway", help="Pathway letter (A, B, ...)")
    ap.add_argument("sample_size", type=int,
                    help="Sobol sample count for baseline arm")
    ap.add_argument("beams", type=int,
                    help="Number of beams PER archetype")
    ap.add_argument("--archetype", default="balanced",
                    help="Archetype filter (default: balanced)")
    ap.add_argument("--radius", type=float, default=2.0,
                    help="Refinement radius in pct per dimension (default: 2.0)")
    ap.add_argument("--baseline-file", default=None,
                    help="Path to existing sobol multibeam test JSON to use as "
                         "baseline instead of rerunning (saves ~500s)")
    args = ap.parse_args()
    run(args.iso.upper(), args.pathway.upper(), args.sample_size, args.beams,
        args.archetype, args.radius, args.baseline_file)
