#!/usr/bin/env python3
"""
step2_3_de_pathway.py
Year-by-year differential evolution pathway optimizer.

Uses dispatch_utils for canonical 8760 hourly dispatch scoring.
Replaces Sobol/LHS sampling with global optimization at each year-step.
No archetypes or beams — one DE run per pathway per ISO.

CFE trajectory: smooth linear ramp from actual baseline (dispatch_utils scored)
to --target-2040 (default 90%) to --target-2050 (default 99.9%).
No arbitrary waypoints.

Usage:
    # Single pathway
    python scripts/step2_3_de_pathway.py --iso CAISO --pathway A

    # Pathway B seeded with A's results (guarantees B <= A)
    python scripts/step2_3_de_pathway.py --iso CAISO --pathway A
    python scripts/step2_3_de_pathway.py --iso CAISO --pathway B \
        --ref-winners data/step2.3-de/CAISO_A_base_Medium_winners.json

    # Custom targets and production settings
    python scripts/step2_3_de_pathway.py --iso PJM --pathway B \
        --popsize 20 --maxiter 200 --workers 2 \
        --target-2040 85 --target-2050 99
"""
from __future__ import annotations
import argparse, json, sys, time, types
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

_stub = types.ModuleType("step_2_3_pathway_optimizer")
def _call_with_timeout(fn, timeout, label=""):
    return fn()
_stub._call_with_timeout = _call_with_timeout
sys.modules["step_2_3_pathway_optimizer"] = _stub

import step2_3_adaptive_sobol as solver
import dispatch_utils as du
import pipeline_config as pc
from eia_data_io import load_demand_profiles, load_generation_profiles
from scipy.optimize import differential_evolution

PENALTY_WEIGHT = 1e12
MAX_RESOURCE_PCT = 200.0
STORAGE_MAX_PCT = 50.0

def load_iso_data(iso):
    """Load demand + supply profiles and precompute supply matrix."""
    demand_data = load_demand_profiles()
    gen_profiles = load_generation_profiles()
    demand_norm, total_mwh = du.get_demand_profile(iso, demand_data)
    hybrid_profiles = du._load_hybrid_profiles(iso)
    supply_profiles = du.get_supply_profiles(
        iso, gen_profiles, include_hybrids=True, hybrid_profiles=hybrid_profiles)
    for k in supply_profiles:
        if isinstance(supply_profiles[k], list):
            supply_profiles[k] = np.array(supply_profiles[k], dtype=np.float64)
    supply_matrix = du.build_supply_matrix(
        supply_profiles, resource_types=du.RESOURCE_TYPES_HYBRID)
    return demand_norm, supply_profiles, supply_matrix

# [OPT-3] Pre-allocated dict buffer for score_cfe
_score_pcts_buf = {}

def score_cfe(res_pcts, b4_pct, b8_pct, ldes_pct,
              demand_norm, supply_profiles, supply_matrix):
    """Score portfolio CFE using canonical dispatch. Returns (cfe%, residual array)."""
    _score_pcts_buf.clear()
    _score_pcts_buf.update(res_pcts)
    result = du.reconstruct_hourly_dispatch(
        demand_norm, supply_profiles, _score_pcts_buf,
        procurement_pct=100, battery_dispatch_pct=b4_pct,
        battery8_dispatch_pct=b8_pct, ldes_dispatch_pct=ldes_pct,
        h2_dispatch_pct=0, supply_matrix=supply_matrix,
        resource_types=du.RESOURCE_TYPES_HYBRID)
    total_clean = result["total_clean"]
    matched = np.minimum(total_clean, demand_norm)
    return float(np.sum(matched) / np.sum(demand_norm) * 100), result["residual_demand"]

# [OPT-1] O(n) 3rd-largest via np.partition
def _p9997(arr):
    """3rd-largest element of arr (p99.97 proxy for gas sizing)."""
    if len(arr) < 3:
        return float(arr.max()) if len(arr) > 0 else 0.0
    return float(np.partition(arr, -3)[-3])

def size_h2(residual_demand, demand_norm, cfe_pre, target_cfe, dem_twh):
    """Find minimum H2 capacity to close CFE gap via binary search."""
    if cfe_pre >= target_cfe:
        return {"h2_mw": 0.0, "h2_mwh": 0.0, "resid_p9997": _p9997(residual_demand)}
    total_demand = np.sum(demand_norm)
    needed_norm = (target_cfe - cfe_pre) / 100.0 * total_demand
    if float(residual_demand.sum()) < needed_norm * 0.999:
        return {"h2_mw": np.inf, "h2_mwh": 0.0, "resid_p9997": np.inf}
    lo_c, hi_c = 0.0, float(residual_demand.max()) * 1.001
    for _ in range(60):
        mid = (lo_c + hi_c) * 0.5
        if float(np.minimum(residual_demand, mid).sum()) >= needed_norm:
            hi_c = mid
        else:
            lo_c = mid
    h2_disp = np.minimum(residual_demand, hi_c)
    post_resid = np.maximum(residual_demand - h2_disp, 0.0)
    return {"h2_mw": hi_c * dem_twh * 1e6,
            "h2_mwh": float(h2_disp.sum()) * dem_twh * 1e6,
            "resid_p9997": _p9997(post_resid)}

def pcts_to_dict(pcts_arr):
    return {res: float(pcts_arr[ri])
            for ri, res in enumerate(solver.RESOURCE_ORDER) if pcts_arr[ri] > 0.001}

def run_de_pathway(iso, pathway, scenario="base", demand_growth="Medium",
                   popsize=10, maxiter=100, workers=1, seed=42,
                   target_2040=90.0, target_2050=99.9, reference_winners=None):
    """Run year-by-year DE pathway optimization.
    WARNING [OPT-5]: workers > 1 pickles the objective closure including large
    numpy arrays on every batch. Profile before using — may be slower than serial.
    """
    sc = solver.COST_SCENARIOS[scenario]
    cfg = solver.RunConfig(iso=iso, pathway=pathway, scenario_name=scenario,
                           demand_growth=demand_growth, cost_mode=1, **sc)
    demand_vec = solver.demand_twh_vec(iso, demand_growth)
    cfe_targets = solver.cfe_target_vec(iso).copy()
    existing_gas = float(pc.EXISTING_GAS_CAPACITY_MW[iso])
    gaf = float(pc.GAS_AVAILABILITY_FACTOR[iso])
    free_idx = cfg.free_indices
    free_idx_arr = np.array(free_idx)  # [OPT-4]
    n_free = len(free_idx)
    baseline = pc.GRID_MIX_SHARES[iso]

    du.warm_dispatch_kernels()
    demand_norm, supply_profiles, supply_matrix = load_iso_data(iso)

    floor_pcts = np.zeros(solver.N_RESOURCES, dtype=np.float64)
    for ri, res in enumerate(solver.RESOURCE_ORDER):
        floor_pcts[ri] = baseline.get(res, 0.0)
    floor_stor = np.zeros(solver.N_STORAGE, dtype=np.float64)

    rng = np.random.default_rng(seed)
    vintage_ledger = []
    vintage_cost_running = 0.0  # [OPT-7]
    cumulative_cost = 0.0
    peak_gas_mw = peak_h2_mw = 0.0
    year_results = []
    winners_by_year = {}

    base_demand_twh = float(pc.REGIONAL_DEMAND_TWH[iso])
    baseline_cf_twh = baseline.get("clean_firm", 0) / 100.0 * base_demand_twh
    frozen_twh = {"hydro": baseline.get("hydro", 0) / 100.0 * base_demand_twh}
    if pathway == "A":
        frozen_twh["clean_firm"] = baseline.get("clean_firm", 0) / 100.0 * base_demand_twh

    baseline_cfe, _ = score_cfe(
        pcts_to_dict(floor_pcts), 0, 0, 0, demand_norm, supply_profiles, supply_matrix)

    for yi_t, year_t in enumerate(solver.YEARS):
        if year_t <= 2040:
            frac = (year_t - solver.BASE_YEAR) / (2040 - solver.BASE_YEAR)
            cfe_targets[yi_t] = baseline_cfe + frac * (target_2040 - baseline_cfe)
        else:
            frac = (year_t - 2040) / (2050 - 2040)
            cfe_targets[yi_t] = target_2040 + frac * (target_2050 - target_2040)

    _storage_set = set(solver.STORAGE_COLS)  # [OPT-7]
    _obj_pcts = np.zeros(solver.N_RESOURCES, dtype=np.float64)  # [OPT-4]
    START_YI = 1

    print(f"\n{'='*70}")
    print(f"DE Pathway Optimizer (dispatch_utils scoring)")
    print(f"ISO={iso}  Pathway={pathway}  Scenario={scenario}  Demand={demand_growth}")
    print(f"popsize={popsize}  maxiter={maxiter}  workers={workers}  seed={seed}")
    print(f"Baseline CFE: {baseline_cfe:.1f}%  -> {target_2040:.0f}% by 2040 -> {target_2050:.0f}% by 2050")
    print(f"Annual ramp: {(target_2040 - baseline_cfe)/15:.1f}pp/yr (2025-2040)"
          f"  {(target_2050 - target_2040)/10:.1f}pp/yr (2040-2050)")
    print(f"{'='*70}\n")

    t0 = time.time()
    for yi in range(START_YI, solver.N_YEARS):
        _yt = time.time()
        year = solver.YEARS[yi]
        dem_twh = demand_vec[yi]
        target = cfe_targets[yi]

        for res, twh in frozen_twh.items():
            floor_pcts[solver.RES_IDX[res]] = twh / dem_twh * 100.0
        if pathway == "A":
            for res in ["offshore_wind", "geothermal", "ccs_ccgt"]:
                floor_pcts[solver.RES_IDX[res]] = 0.0
        elif pathway in solver.FIRM_PATHWAYS:
            if iso not in pc.OFFSHORE_ISOS:
                floor_pcts[solver.RES_IDX["offshore_wind"]] = 0.0
            if iso != "CAISO":
                floor_pcts[solver.RES_IDX["geothermal"]] = 0.0
            if float(pc.CCS_CAP_TWH.get(iso, 0)) <= 0:
                floor_pcts[solver.RES_IDX["ccs_ccgt"]] = 0.0
            floor_pcts[solver.RES_IDX["clean_firm"]] = max(
                floor_pcts[solver.RES_IDX["clean_firm"]],
                baseline_cf_twh / dem_twh * 100.0)

        floor_cfe, floor_resid = score_cfe(
            pcts_to_dict(floor_pcts), floor_stor[0], floor_stor[1], floor_stor[2],
            demand_norm, supply_profiles, supply_matrix)

        if floor_cfe >= target:
            winner_pcts = floor_pcts.copy()
            winner_stor = floor_stor.copy()
            winner_cfe = floor_cfe
            h2_mw = h2_mwh = 0.0
            resid_for_gas = _p9997(floor_resid)
        else:
            cfe_ceiling = cfe_targets[yi + 1] if yi < solver.N_YEARS - 1 else 100.0
            lo = np.zeros(n_free + solver.N_STORAGE)
            hi = np.zeros(n_free + solver.N_STORAGE)
            for k, ri in enumerate(free_idx):
                lo[k] = floor_pcts[ri]
                hi[k] = MAX_RESOURCE_PCT
                res = solver.RESOURCE_ORDER[ri]
                if pathway in solver.FIRM_PATHWAYS:
                    if res == "ccs_ccgt":
                        hi[k] = min(hi[k], float(pc.CCS_CAP_TWH.get(iso, 9999)) / dem_twh * 100)
                    elif res == "geothermal":
                        hi[k] = min(hi[k], float(pc.GEOTHERMAL_CAP_TWH) / dem_twh * 100)
                    elif res == "offshore_wind" and iso not in pc.OFFSHORE_ISOS:
                        hi[k] = 0.0
            for j in range(solver.N_STORAGE):
                lo[n_free + j] = floor_stor[j]
                hi[n_free + j] = STORAGE_MAX_PCT

            capex_yr = solver.h2_peaker_capex_kw_yr(year, cfg)
            fuel_mwh_rate = solver.h2_peaker_fuel_mwh(year, cfg)

            # [OPT-6] Pre-compute cost coefficient vectors
            lcoe_coeffs = np.array([
                solver.get_resource_lcoe(iso, solver.RESOURCE_ORDER[ri], year, cfg)
                * dem_twh * 1e4 for ri in free_idx], dtype=np.float64)
            stor_coeffs = np.array([
                solver.storage_net_cost(iso, sc, cfg, year=year)
                * dem_twh * 1e4 for sc in solver.STORAGE_COLS], dtype=np.float64)

            floor_pcts_snap = floor_pcts.copy()
            floor_stor_snap = floor_stor.copy()

            def objective(x):
                xc = np.clip(x, lo, hi)
                np.copyto(_obj_pcts, floor_pcts_snap)  # [OPT-4]
                _obj_pcts[free_idx_arr] = xc[:n_free]

                cfe, resid = score_cfe(
                    pcts_to_dict(_obj_pcts), xc[n_free], xc[n_free+1], xc[n_free+2],
                    demand_norm, supply_profiles, supply_matrix)

                h2 = size_h2(resid, demand_norm, cfe, target, dem_twh)
                if h2["h2_mw"] > 1e8:
                    return 1e18
                post_cfe = cfe + (h2["h2_mwh"] / (dem_twh * 1e6) * 100 if h2["h2_mwh"] > 0 else 0)

                # [OPT-6] Vectorized cost
                res_deltas = _obj_pcts[free_idx_arr] - floor_pcts_snap[free_idx_arr]
                np.maximum(res_deltas, 0, out=res_deltas)
                cost = float(np.dot(res_deltas, lcoe_coeffs))
                stor_deltas = xc[n_free:n_free + solver.N_STORAGE] - floor_stor_snap
                np.maximum(stor_deltas, 0, out=stor_deltas)
                cost += float(np.dot(stor_deltas, stor_coeffs))

                cost += h2["h2_mw"] * capex_yr * 1000 + h2["h2_mwh"] * fuel_mwh_rate
                g = solver.gas_need_mw(h2["resid_p9997"], dem_twh, existing_gas, gaf)
                cost += solver.gas_annual_cost(g, iso, cfg)
                cost += solver.gas_stranding_shadow(g, peak_gas_mw, year, iso)

                if post_cfe < target:
                    cost += PENALTY_WEIGHT * (target - post_cfe) ** 2
                if cfe > cfe_ceiling:
                    cost += PENALTY_WEIGHT * (cfe - cfe_ceiling) ** 2
                return cost

            active = [i for i in range(len(lo)) if hi[i] - lo[i] > 1e-6]
            if not active:
                winner_pcts = floor_pcts.copy()
                winner_stor = floor_stor.copy()
                winner_cfe = floor_cfe
                h2_mw = h2_mwh = 0.0
                resid_for_gas = _p9997(floor_resid)
            else:
                bounds = [(lo[i], hi[i]) for i in active]
                full_x0 = np.concatenate([
                    np.array([floor_pcts[ri] for ri in free_idx]), floor_stor.copy()])

                def obj_active(x_active):
                    full = full_x0.copy()
                    for ai, idx in enumerate(active):
                        full[idx] = x_active[ai]
                    return objective(full)

                de_seed = int(rng.integers(0, 2**31))
                init_pop = "sobol"
                if reference_winners and year in reference_winners:
                    ref_pcts, ref_stor = reference_winners[year]
                    ref_full = np.concatenate([
                        np.array([ref_pcts[ri] for ri in free_idx]), ref_stor.copy()])
                    ref_active = np.array([
                        np.clip(ref_full[idx], bounds[ai][0], bounds[ai][1])
                        for ai, idx in enumerate(active)])
                    n_act = len(active)
                    pop_size = popsize * n_act
                    pop = np.zeros((pop_size, n_act))
                    pop[0] = ref_active
                    pop_rng = np.random.default_rng(de_seed)
                    for i in range(1, pop_size):
                        for j in range(n_act):
                            pop[i, j] = pop_rng.uniform(bounds[j][0], bounds[j][1])
                    init_pop = pop

                de_result = differential_evolution(
                    obj_active, bounds, seed=de_seed, maxiter=maxiter,
                    popsize=popsize, tol=0, mutation=(0.5, 1.0),
                    recombination=0.7, polish=True, init=init_pop,
                    workers=workers)

                full_best = full_x0.copy()
                for ai, idx in enumerate(active):
                    full_best[idx] = de_result.x[ai]
                full_best = np.clip(full_best, lo, hi)
                winner_pcts = floor_pcts.copy()
                for k, ri in enumerate(free_idx):
                    winner_pcts[ri] = full_best[k]
                winner_stor = full_best[n_free:].copy()

                winner_cfe, winner_resid = score_cfe(
                    pcts_to_dict(winner_pcts), winner_stor[0], winner_stor[1],
                    winner_stor[2], demand_norm, supply_profiles, supply_matrix)
                h2 = size_h2(winner_resid, demand_norm, winner_cfe, target, dem_twh)
                h2_mw = h2["h2_mw"]
                h2_mwh = h2["h2_mwh"]
                if h2_mwh > 0:
                    winner_cfe += h2_mwh / (dem_twh * 1e6) * 100
                resid_for_gas = h2["resid_p9997"]

        g_mw = solver.gas_need_mw(resid_for_gas, dem_twh, existing_gas, gaf)
        g_cost = solver.gas_annual_cost(g_mw, iso, cfg)
        capex_yr = solver.h2_peaker_capex_kw_yr(year, cfg)
        fuel_rate = solver.h2_peaker_fuel_mwh(year, cfg)

        year_inc = 0.0
        for k, ri in enumerate(free_idx):
            delta = winner_pcts[ri] - floor_pcts[ri]
            if delta > solver.RATCHET_TOL_PCT:
                lcoe = solver.get_resource_lcoe(iso, solver.RESOURCE_ORDER[ri], year, cfg)
                year_inc += delta / 100 * dem_twh * lcoe * 1e6
                vintage_ledger.append((year, solver.RESOURCE_ORDER[ri],
                                       delta / 100 * dem_twh, lcoe))
                vintage_cost_running += delta / 100 * dem_twh * lcoe * 1e6  # [OPT-7]

        for j, sc in enumerate(solver.STORAGE_COLS):
            ds = (winner_stor[j] - floor_stor[j]) / 100
            if ds > solver.RATCHET_TOL_PCT:
                net = solver.storage_net_cost(iso, sc, cfg, year=year)
                stor_cost_val = ds * net * dem_twh * 1e6
                year_inc += stor_cost_val
                vintage_ledger.append((year, sc, stor_cost_val, ds))
                vintage_cost_running += stor_cost_val  # [OPT-7]

        year_inc += h2_mw * capex_yr * 1000 + h2_mwh * fuel_rate
        year_inc += g_cost
        year_inc += solver.gas_stranding_shadow(g_mw, peak_gas_mw, year, iso)

        total_annual = vintage_cost_running + year_inc  # [OPT-7]
        cumulative_cost += total_annual
        if g_mw > peak_gas_mw: peak_gas_mw = g_mw
        if h2_mw > peak_h2_mw: peak_h2_mw = h2_mw

        yt = time.time() - _yt
        print(f"  {year}: CFE={winner_cfe:.1f}% tgt={target:.1f}% "
              f"gas={g_mw:,.0f}MW h2={h2_mw:,.0f}MW "
              f"ann=${total_annual/1e9:.2f}B cum=${cumulative_cost/1e9:.1f}B ({yt:.1f}s)")

        year_results.append({
            "year": year, "target": round(target, 1),
            "achieved_cfe": round(float(winner_cfe), 2),
            "h2_mw": round(h2_mw, 1), "gas_mw": round(g_mw, 1),
            "total_annual_B": round(total_annual / 1e9, 3),
            "cumulative_B": round(cumulative_cost / 1e9, 3),
        })
        winners_by_year[year] = (winner_pcts.copy(), winner_stor.copy())

        for ri in range(solver.N_RESOURCES):
            built_twh = max(floor_pcts[ri], winner_pcts[ri]) / 100 * dem_twh
            if yi < solver.N_YEARS - 1:
                floor_pcts[ri] = built_twh / demand_vec[yi + 1] * 100
            else:
                floor_pcts[ri] = winner_pcts[ri]
        for j in range(solver.N_STORAGE):
            floor_stor[j] = max(floor_stor[j], winner_stor[j])

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.1f}s")
    print(f"Peak gas: {peak_gas_mw:,.0f} MW | Peak H2: {peak_h2_mw:,.0f} MW")
    print(f"Cumulative cost: ${cumulative_cost/1e9:.2f}B")
    return year_results, winners_by_year

def main():
    ap = argparse.ArgumentParser(description="DE pathway optimizer with dispatch_utils scoring")
    ap.add_argument("--iso", required=True)
    ap.add_argument("--pathway", required=True, choices=["A", "B", "C", "D"])
    ap.add_argument("--popsize", type=int, default=15)
    ap.add_argument("--maxiter", type=int, default=150)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scenario", default="base", choices=list(solver.COST_SCENARIOS.keys()))
    ap.add_argument("--demand-growth", default="Medium", choices=["Low", "Medium", "High"])
    ap.add_argument("--target-2040", type=float, default=90.0)
    ap.add_argument("--target-2050", type=float, default=99.9)
    ap.add_argument("--ref-winners", default=None)
    args = ap.parse_args()

    iso = args.iso.upper()
    if iso not in pc.ISOS:
        raise SystemExit(f"Unknown ISO: {iso}. Known: {list(pc.ISOS)}")

    ref = None
    if args.ref_winners:
        with open(args.ref_winners) as f:
            data = json.load(f)
        ref = {int(y): (np.array(p), np.array(s)) for y, (p, s) in data["winners"].items()}
        print(f"Loaded reference winners from {args.ref_winners}")

    results, winners = run_de_pathway(
        iso, args.pathway, scenario=args.scenario, demand_growth=args.demand_growth,
        popsize=args.popsize, maxiter=args.maxiter, workers=args.workers, seed=args.seed,
        target_2040=args.target_2040, target_2050=args.target_2050, reference_winners=ref)

    import pyarrow as pa, pyarrow.parquet as pq
    out_dir = PROJECT_ROOT / "data" / "step2.3-de"
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = (f"{iso}_pathway{args.pathway}_{args.scenario}_{args.demand_growth}"
           f"_p{args.popsize}_i{args.maxiter}_s{args.seed}")
    results_path = out_dir / f"{tag}.parquet"
    winners_path = out_dir / f"{tag}_winners.json"

    rows = []
    for yr in results:
        row = {**yr, "iso": iso, "pathway": args.pathway, "scenario": args.scenario,
               "demand_growth": args.demand_growth, "popsize": args.popsize,
               "maxiter": args.maxiter, "seed": args.seed,
               "target_2040": args.target_2040, "target_2050": args.target_2050}
        rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows), results_path)

    with open(winners_path, "w") as f:
        json.dump({"winners": {str(y): (p.tolist(), s.tolist())
                               for y, (p, s) in winners.items()}}, f)

    print(f"\nResults: {results_path} ({results_path.stat().st_size / 1024:.0f} KB)")
    print(f"Winners: {winners_path}")

if __name__ == "__main__":
    main()

