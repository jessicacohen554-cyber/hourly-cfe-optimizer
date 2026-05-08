#!/usr/bin/env python3
"""
step2_3_de_myopic_tf.py
Myopic (standalone-point) DE optimizer for individual CFE thresholds.

Each threshold is solved independently — no year-by-year ratchet, no build
history carried between thresholds.  Answers "what is the cheapest generation
mix to reach X% CFE, starting from today's grid baseline, using the cost
curves and demand at the year where X% falls on the trajectory?"

Pathway matters:
    A = VRE + hybrids + storage only.  Clean_firm frozen at baseline.
        No offshore wind, geothermal, or CCS.
    B = All resources.  Clean_firm expandable (uprate tranche priced at
        UPRATE_LCOE, then Wright's Law newbuild).  Proactive NOAK (2035).
    C = Same resource set as B.  Delayed NOAK (2040).
    D = Same resource set as B.  Further-delayed NOAK (2045).

Year mapping: linear ramp 90% @ 2040 → 99.9% @ 2050.  Each threshold snaps
to the nearest model year for Wright's Law cost lookups and demand growth.

Gas modes:
    2 = optimizer sees gas cost only (default)
    3 = optimizer blind to gas economics
    Mode 1 (cost + stranding shadow) is excluded — no prior-year peak gas
    exists in a myopic solve, so the shadow has no reference point.

Thermal fleet: coal/oil retirement computed using the target CFE as prev_cfe.
At the year where X% CFE is reached on the trajectory, the fleet has already
responded to that level of clean penetration via the economic retirement
sigmoid.

Known limitations:
    - Whole-ISO framing (same as pathway solver).
    - No demand flexibility or load-shape evolution.
    - Myopic solves overestimate cost at high thresholds relative to the
      pathway solver because they can't exploit cheap builds at lower
      thresholds (no ratchet benefit).  They reveal the standalone marginal
      cost of each CFE tier, free of path-dependency artifacts.

Usage:
    # Pathway A vs B comparison (default thresholds)
    python scripts/step2_3_de_myopic_tf.py --iso ERCOT --pathway A
    python scripts/step2_3_de_myopic_tf.py --iso ERCOT --pathway B

    # Single threshold, high resolution
    python scripts/step2_3_de_myopic_tf.py --iso PJM --pathway B \\
        --thresholds 95 --popsize 20 --maxiter 200

    # Gas-blind, custom thresholds
    python scripts/step2_3_de_myopic_tf.py --iso PJM --pathway A \\
        --thresholds 90 95 99 99.9 --gas-mode 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

# Stub the timeout wrapper that dispatch_utils expects on import
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Default thresholds — the standard high-CFE test points
DEFAULT_THRESHOLDS = [90.0, 92.5, 95.0, 97.5, 99.0, 99.9]

# Ramp endpoints for year mapping
RAMP_CFE_START, RAMP_YEAR_START = 90.0, 2040
RAMP_CFE_END, RAMP_YEAR_END = 99.9, 2050

# DE objective tuning (same as pathway solver)
PENALTY_WEIGHT = 1e12        # $/pp^2 for CFE shortfall and overshoot
MAX_RESOURCE_PCT = 200.0     # hard ceiling per resource dimension
STORAGE_MAX_PCT = 50.0       # generous storage cap — cost function regulates
EARLY_EXIT_GAP_PP = 10.0     # skip cost loop if CFE misses target by >10pp


# ---------------------------------------------------------------------------
# Year mapping: threshold → model year on the linear ramp
# ---------------------------------------------------------------------------
def threshold_to_year(target: float) -> int:
    """Map a CFE threshold to the year it falls on the linear ramp.

    90% → 2040, 99.9% → 2050, linearly interpolated in between.
    Snapped to the nearest integer year within the [2025, 2050] model range.
    """
    if target <= RAMP_CFE_START:
        return RAMP_YEAR_START
    if target >= RAMP_CFE_END:
        return RAMP_YEAR_END
    frac = (target - RAMP_CFE_START) / (RAMP_CFE_END - RAMP_CFE_START)
    year_f = RAMP_YEAR_START + frac * (RAMP_YEAR_END - RAMP_YEAR_START)
    return max(solver.BASE_YEAR, min(2050, round(year_f)))


# ---------------------------------------------------------------------------
# Pathway-aware free resource indices
# ---------------------------------------------------------------------------
def pathway_free_indices(iso: str, pathway: str) -> list[int]:
    """Resources the optimizer can adjust, gated by pathway.

    Pathway A: solar, wind, 4 hybrids.  No firm expansion.
    Pathway B/C/D: A's set + clean_firm + offshore (where applicable)
                   + CCS (where applicable) + geothermal (CAISO only).

    Hydro is always frozen at baseline (not in the free set).
    """
    # VRE core — always available in every pathway
    free = list(solver.PATHWAY_A_FREE)  # solar, wind, 4 hybrids

    # Firm resources — only in pathways B/C/D
    if pathway in solver.FIRM_PATHWAYS:
        free.append("clean_firm")
        if iso in pc.OFFSHORE_ISOS:
            free.append("offshore_wind")
        if float(pc.CCS_CAP_TWH.get(iso, 0)) > 0:
            free.append("ccs_ccgt")
        if iso == "CAISO":
            free.append("geothermal")

    return [solver.RES_IDX[r] for r in free]


# ---------------------------------------------------------------------------
# Profile loading (matches pathway solver)
# ---------------------------------------------------------------------------
def load_iso_data(iso: str):
    """Load demand + supply profiles and precompute supply matrix."""
    demand_data = load_demand_profiles()
    gen_profiles = load_generation_profiles()
    demand_norm, total_mwh = du.get_demand_profile(iso, demand_data)
    hybrid_profiles = du._load_hybrid_profiles(iso)
    supply_profiles = du.get_supply_profiles(
        iso, gen_profiles, include_hybrids=True,
        hybrid_profiles=hybrid_profiles)
    for k in supply_profiles:
        if isinstance(supply_profiles[k], list):
            supply_profiles[k] = np.array(supply_profiles[k], dtype=np.float64)
    supply_matrix = du.build_supply_matrix(
        supply_profiles, resource_types=du.RESOURCE_TYPES_HYBRID)
    return demand_norm, supply_profiles, supply_matrix


# ---------------------------------------------------------------------------
# CFE scoring and H2 sizing (identical to pathway solver)
# ---------------------------------------------------------------------------
def score_cfe(res_pcts, b4_pct, b8_pct, ldes_pct,
              demand_norm, supply_profiles, supply_matrix):
    """Score portfolio CFE using canonical dispatch. Returns (cfe%, residual)."""
    result = du.reconstruct_hourly_dispatch(
        demand_norm, supply_profiles, res_pcts,
        procurement_pct=100,
        battery_dispatch_pct=b4_pct,
        battery8_dispatch_pct=b8_pct,
        ldes_dispatch_pct=ldes_pct,
        h2_dispatch_pct=0,
        supply_matrix=supply_matrix,
        resource_types=du.RESOURCE_TYPES_HYBRID)
    total_clean = result["total_clean"]
    matched = np.minimum(total_clean, demand_norm)
    cfe = float(np.sum(matched) / np.sum(demand_norm) * 100)
    return cfe, result["residual_demand"]


def _p9997(arr):
    """3rd-largest value (p99.97 proxy for 8760 hours). O(n) via partition."""
    n = len(arr)
    if n < 3:
        return 0.0
    return float(np.partition(arr, -3)[-3])


def pcts_to_dict(pcts_arr):
    """Convert resource pcts array to dict for dispatch_utils."""
    return {res: float(pcts_arr[ri])
            for ri, res in enumerate(solver.RESOURCE_ORDER)
            if pcts_arr[ri] > 0.001}


def size_h2(residual_demand, demand_norm, cfe_pre, target_cfe, dem_twh):
    """Find minimum H2 capacity to close CFE gap (analytical O(n log n)).

    Sorts residual demand descending, walks prefix sums to find the capacity
    threshold.  One sort + one linear scan replaces the 60-iteration binary
    search used in older solvers.
    """
    if cfe_pre >= target_cfe:
        return {"h2_mw": 0.0, "h2_mwh": 0.0,
                "resid_p9997": _p9997(residual_demand)}

    total_demand = np.sum(demand_norm)
    needed_norm = (target_cfe - cfe_pre) / 100.0 * total_demand
    total_gap = float(residual_demand.sum())

    if total_gap < needed_norm * 0.999:
        return {"h2_mw": np.inf, "h2_mwh": 0.0, "resid_p9997": np.inf}

    sorted_desc = np.sort(residual_demand)[::-1]
    n = len(sorted_desc)
    h2_cap_norm = sorted_desc[0]

    running_prefix = 0.0
    for k in range(n):
        val = sorted_desc[k]
        remaining_tail = total_gap - running_prefix - val
        total_disp = (k + 1) * val + remaining_tail
        if total_disp < needed_norm:
            if k == 0:
                h2_cap_norm = needed_norm / n
            else:
                hours_below = total_gap - running_prefix
                h2_cap_norm = (needed_norm - hours_below) / k
            break
        running_prefix += val
    else:
        h2_cap_norm = sorted_desc[-1]

    h2_disp = np.minimum(residual_demand, h2_cap_norm)
    h2_mw = h2_cap_norm * dem_twh * 1e6
    h2_mwh = float(h2_disp.sum()) * dem_twh * 1e6
    post_resid = np.maximum(residual_demand - h2_disp, 0.0)
    return {"h2_mw": h2_mw, "h2_mwh": h2_mwh, "resid_p9997": _p9997(post_resid)}


# ---------------------------------------------------------------------------
# DE objective — myopic variant with uprate tranche
# ---------------------------------------------------------------------------
class _MyopicDEObjective:
    """Picklable DE objective for a single-threshold myopic solve.

    Same core logic as the pathway solver's _DEObjective but simplified:
    no gas stranding shadow (mode 1 excluded), no year-to-year state.
    Includes clean_firm uprate tranche pricing for pathway B/C/D.
    For pathway A, clean_firm is never in free_idx so the uprate branch
    is never reached (_cf_free_k stays at -1).
    """

    __slots__ = (
        'lo', 'hi', 'floor_pcts', 'free_idx', 'n_free',
        'demand_norm', 'supply_profiles', 'supply_matrix',
        'dem_twh', 'cfg', 'iso', 'year', 'target', 'cfe_ceiling',
        'capex_yr', 'fuel_mwh_rate', 'thermal_peak_credit', 'gaf',
        'floor_stor', 'full_x0', 'active', 'gas_mode',
        # Precomputed cost arrays
        '_lcoe_by_free_k', '_stor_net',
        # Pre-allocated buffers (workers=1 only)
        '_buf_full', '_buf_pcts', '_use_buffers',
        # Clean firm uprate tranche (pathway B/C/D only)
        '_cf_free_k', '_remaining_uprate_pct', '_uprate_lcoe', '_newbuild_lcoe',
    )

    def __init__(self, *, lo, hi, floor_pcts, free_idx, n_free,
                 demand_norm, supply_profiles, supply_matrix,
                 dem_twh, cfg, iso, year, target, cfe_ceiling,
                 capex_yr, fuel_mwh_rate, thermal_peak_credit, gaf,
                 floor_stor, full_x0, active, gas_mode=2, workers=1):
        self.lo = lo
        self.hi = hi
        self.floor_pcts = floor_pcts
        self.free_idx = free_idx
        self.n_free = n_free
        self.demand_norm = demand_norm
        self.supply_profiles = supply_profiles
        self.supply_matrix = supply_matrix
        self.dem_twh = dem_twh
        self.cfg = cfg
        self.iso = iso
        self.year = year
        self.target = target
        self.cfe_ceiling = cfe_ceiling
        self.capex_yr = capex_yr
        self.fuel_mwh_rate = fuel_mwh_rate
        self.thermal_peak_credit = thermal_peak_credit
        self.gaf = gaf
        self.floor_stor = floor_stor
        self.full_x0 = full_x0
        self.active = active
        self.gas_mode = gas_mode

        # --- Precompute year-constant LCOEs ---
        # Eliminates per-evaluation function call overhead.
        self._lcoe_by_free_k = np.array([
            solver.get_resource_lcoe(iso, solver.RESOURCE_ORDER[ri], year, cfg)
            for ri in free_idx], dtype=np.float64)

        self._stor_net = np.array([
            solver.storage_net_cost(iso, sc, cfg, year=year)
            for sc in solver.STORAGE_COLS], dtype=np.float64)

        # --- Clean firm uprate tranche ---
        # In pathway A, clean_firm is not in free_idx so _cf_free_k = -1
        # and the piecewise branch is never entered.  In B/C/D, the first
        # UPRATE_CAP_TWH above the existing fleet baseline is priced at
        # UPRATE_LCOE (no TX); remainder at Wright's Law newbuild (with TX).
        cf_ri = solver.RES_IDX["clean_firm"]
        self._cf_free_k = -1
        for k, ri in enumerate(free_idx):
            if ri == cf_ri:
                self._cf_free_k = k
                break
        if self._cf_free_k >= 0:
            baseline_cf_twh = (pc.GRID_MIX_SHARES[iso].get("clean_firm", 0.0)
                               / 100.0 * float(pc.REGIONAL_DEMAND_TWH[iso]))
            uprate_cap_twh = float(pc.UPRATE_CAP_TWH[iso])
            floor_cf_twh = floor_pcts[cf_ri] / 100.0 * dem_twh
            remaining_twh = max(0.0, baseline_cf_twh + uprate_cap_twh
                                - floor_cf_twh)
            self._remaining_uprate_pct = remaining_twh / dem_twh * 100.0
            self._uprate_lcoe = float(pc.UPRATE_LCOE[cfg.firm_cost])
            # Newbuild = what get_resource_lcoe returns (Wright's Law + TX)
            self._newbuild_lcoe = self._lcoe_by_free_k[self._cf_free_k]
        else:
            self._remaining_uprate_pct = 0.0
            self._uprate_lcoe = 0.0
            self._newbuild_lcoe = 0.0

        # --- Pre-allocate work buffers (single-threaded) ---
        self._use_buffers = (workers <= 1)
        if self._use_buffers:
            self._buf_full = full_x0.copy()
            self._buf_pcts = floor_pcts.copy()
        else:
            self._buf_full = None
            self._buf_pcts = None

    def __call__(self, x_active):
        """Map active-dim vector -> full vector -> cost."""
        if self._use_buffers:
            full = self._buf_full
            np.copyto(full, self.full_x0)
        else:
            full = self.full_x0.copy()
        for ai, idx in enumerate(self.active):
            full[idx] = x_active[ai]
        return self._objective(full)

    def _objective(self, x):
        lo, hi = self.lo, self.hi
        floor_pcts = self.floor_pcts
        free_idx, n_free = self.free_idx, self.n_free

        xc = np.clip(x, lo, hi)

        if self._use_buffers:
            pcts = self._buf_pcts
            np.copyto(pcts, floor_pcts)
        else:
            pcts = floor_pcts.copy()

        for k, ri in enumerate(free_idx):
            pcts[ri] = xc[k]

        # Score CFE via dispatch_utils canonical dispatch
        cfe, resid = score_cfe(
            pcts_to_dict(pcts), xc[n_free], xc[n_free + 1], xc[n_free + 2],
            self.demand_norm, self.supply_profiles, self.supply_matrix)

        # Size H2 peaker to close remaining gap
        h2 = size_h2(resid, self.demand_norm, cfe, self.target, self.dem_twh)
        if h2["h2_mw"] > 1e8:
            return 1e18

        post_cfe = cfe + (h2["h2_mwh"] / (self.dem_twh * 1e6) * 100
                          if h2["h2_mwh"] > 0 else 0)

        # Early-exit penalty — skip cost loop if hopelessly far from target
        if post_cfe < self.target - EARLY_EXIT_GAP_PP:
            return PENALTY_WEIGHT * (self.target - post_cfe) ** 2

        # --- Incremental cost above floor ---
        cost = 0.0
        dem_twh = self.dem_twh
        cf_k = self._cf_free_k
        for k, ri in enumerate(free_idx):
            delta = pcts[ri] - floor_pcts[ri]
            if delta > 0:
                if k == cf_k:
                    # Piecewise: uprate tranche (cheap) then newbuild
                    uprate_d = min(delta, self._remaining_uprate_pct)
                    newbuild_d = delta - uprate_d
                    cost += uprate_d / 100 * dem_twh * self._uprate_lcoe * 1e6
                    cost += newbuild_d / 100 * dem_twh * self._newbuild_lcoe * 1e6
                else:
                    cost += delta / 100 * dem_twh * self._lcoe_by_free_k[k] * 1e6

        # Storage cost
        for j in range(solver.N_STORAGE):
            ds = (xc[n_free + j] - self.floor_stor[j]) / 100
            if ds > 0:
                cost += ds * self._stor_net[j] * dem_twh * 1e6

        # H2 peaker cost
        cost += h2["h2_mw"] * self.capex_yr * 1000
        cost += h2["h2_mwh"] * self.fuel_mwh_rate

        # Gas — mode 2: cost visible to optimizer, mode 3: blind
        g = solver.gas_need_mw(h2["resid_p9997"], dem_twh,
                               self.thermal_peak_credit, self.gaf)
        if self.gas_mode == 2:
            cost += solver.gas_annual_cost(g, self.iso, self.cfg)

        # Penalties for missing target or overshooting ceiling
        if post_cfe < self.target:
            cost += PENALTY_WEIGHT * (self.target - post_cfe) ** 2
        if cfe > self.cfe_ceiling:
            cost += PENALTY_WEIGHT * (cfe - self.cfe_ceiling) ** 2

        return cost


# ---------------------------------------------------------------------------
# Single-threshold solver
# ---------------------------------------------------------------------------
def solve_one_threshold(
    iso, target, year, pathway, cfg, free_idx, n_free,
    demand_norm, supply_profiles, supply_matrix,
    dem_twh, floor_pcts, floor_stor,
    thermal_peak_credit, gaf,
    popsize, maxiter, workers, seed, gas_mode,
):
    """Solve a single CFE threshold from the grid baseline.

    Returns dict with target, year, achieved_cfe, cost breakdown, resource
    mix, gas/H2 sizing, and per-resource incremental TWh.
    """
    capex_yr = solver.h2_peaker_capex_kw_yr(year, cfg)
    fuel_mwh_rate = solver.h2_peaker_fuel_mwh(year, cfg)

    # Score baseline floor
    floor_cfe, floor_resid = score_cfe(
        pcts_to_dict(floor_pcts), floor_stor[0], floor_stor[1], floor_stor[2],
        demand_norm, supply_profiles, supply_matrix)

    # Uprate parameters for cost tally (used outside the DE objective)
    cf_ri = solver.RES_IDX["clean_firm"]
    baseline_cf_twh = (pc.GRID_MIX_SHARES[iso].get("clean_firm", 0.0)
                       / 100.0 * float(pc.REGIONAL_DEMAND_TWH[iso]))
    uprate_cap_twh = float(pc.UPRATE_CAP_TWH[iso])
    uprate_lcoe = float(pc.UPRATE_LCOE[cfg.firm_cost])

    if floor_cfe >= target:
        # Floor already meets target — no builds needed
        h2_mw = h2_mwh = 0.0
        winner_pcts = floor_pcts.copy()
        winner_stor = floor_stor.copy()
        winner_cfe = floor_cfe
        resid_for_gas = _p9997(floor_resid)
    else:
        # CFE ceiling: don't overbuild beyond a reasonable margin
        cfe_ceiling = min(target + 3.0, 100.0)

        # Build bounds — pathway gates enforce resource availability
        lo = np.zeros(n_free + solver.N_STORAGE)
        hi = np.zeros(n_free + solver.N_STORAGE)
        for k, ri in enumerate(free_idx):
            lo[k] = floor_pcts[ri]
            hi[k] = MAX_RESOURCE_PCT
            res = solver.RESOURCE_ORDER[ri]
            # Regional caps on firm resources (pathway B/C/D only —
            # pathway A never includes these in free_idx)
            if res == "ccs_ccgt":
                hi[k] = min(hi[k],
                            float(pc.CCS_CAP_TWH.get(iso, 9999)) / dem_twh * 100)
            elif res == "geothermal":
                hi[k] = min(hi[k],
                            float(pc.GEOTHERMAL_CAP_TWH) / dem_twh * 100)
            elif res == "offshore_wind" and iso not in pc.OFFSHORE_ISOS:
                hi[k] = 0.0
        for j in range(solver.N_STORAGE):
            lo[n_free + j] = floor_stor[j]
            hi[n_free + j] = STORAGE_MAX_PCT

        # Filter zero-width dims (e.g. offshore in non-coastal ISO)
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
                np.array([floor_pcts[ri] for ri in free_idx]),
                floor_stor.copy()])

            obj_fn = _MyopicDEObjective(
                lo=lo, hi=hi, floor_pcts=floor_pcts,
                free_idx=free_idx, n_free=n_free,
                demand_norm=demand_norm,
                supply_profiles=supply_profiles,
                supply_matrix=supply_matrix,
                dem_twh=dem_twh, cfg=cfg, iso=iso, year=year,
                target=target, cfe_ceiling=cfe_ceiling,
                capex_yr=capex_yr, fuel_mwh_rate=fuel_mwh_rate,
                thermal_peak_credit=thermal_peak_credit, gaf=gaf,
                floor_stor=floor_stor, full_x0=full_x0,
                active=active, gas_mode=gas_mode, workers=workers)

            rng = np.random.default_rng(seed)
            de_seed = int(rng.integers(0, 2**31))

            de_result = differential_evolution(
                obj_fn, bounds,
                seed=de_seed, maxiter=maxiter, popsize=popsize,
                tol=0, mutation=(0.5, 1.0), recombination=0.7,
                polish=True, init="latinhypercube",
                workers=workers)

            # Reconstruct winner from DE result
            full_best = full_x0.copy()
            for ai, idx in enumerate(active):
                full_best[idx] = de_result.x[ai]
            full_best = np.clip(full_best, lo, hi)

            winner_pcts = floor_pcts.copy()
            for k, ri in enumerate(free_idx):
                winner_pcts[ri] = full_best[k]
            winner_stor = full_best[n_free:].copy()

            # Final scoring with canonical dispatch
            winner_cfe, winner_resid = score_cfe(
                pcts_to_dict(winner_pcts),
                winner_stor[0], winner_stor[1], winner_stor[2],
                demand_norm, supply_profiles, supply_matrix)

            h2 = size_h2(winner_resid, demand_norm, winner_cfe, target, dem_twh)
            h2_mw = h2["h2_mw"]
            h2_mwh = h2["h2_mwh"]
            if h2_mwh > 0:
                winner_cfe += h2_mwh / (dem_twh * 1e6) * 100
            resid_for_gas = h2["resid_p9997"]

    # --- Gas (always computed for reporting, regardless of gas_mode) ---
    g_mw = solver.gas_need_mw(resid_for_gas, dem_twh, thermal_peak_credit, gaf)
    g_cost = solver.gas_annual_cost(g_mw, iso, cfg)

    # --- Incremental cost tally with piecewise clean_firm ---
    inc_cost = 0.0
    resource_twh = {}
    for k, ri in enumerate(free_idx):
        delta = winner_pcts[ri] - floor_pcts[ri]
        res = solver.RESOURCE_ORDER[ri]
        if delta > solver.RATCHET_TOL_PCT:
            delta_twh = delta / 100 * dem_twh
            resource_twh[res] = delta_twh
            if ri == cf_ri:
                # Piecewise: uprate tranche (cheap) then newbuild (expensive)
                floor_cf_twh = floor_pcts[ri] / 100 * dem_twh
                remaining_twh = max(0.0, baseline_cf_twh + uprate_cap_twh
                                    - floor_cf_twh)
                remaining_pct = remaining_twh / dem_twh * 100.0
                uprate_d = min(delta, remaining_pct)
                newbuild_d = delta - uprate_d
                newbuild_lcoe = solver.get_resource_lcoe(
                    iso, "clean_firm", year, cfg)
                inc_cost += uprate_d / 100 * dem_twh * uprate_lcoe * 1e6
                inc_cost += newbuild_d / 100 * dem_twh * newbuild_lcoe * 1e6
                # Record tranche decomposition for inspection
                resource_twh["clean_firm_uprate"] = uprate_d / 100 * dem_twh
                resource_twh["clean_firm_newbuild"] = newbuild_d / 100 * dem_twh
            else:
                lcoe = solver.get_resource_lcoe(iso, res, year, cfg)
                inc_cost += delta_twh * lcoe * 1e6

    for j, sc in enumerate(solver.STORAGE_COLS):
        ds = (winner_stor[j] - floor_stor[j]) / 100
        if ds > solver.RATCHET_TOL_PCT:
            inc_cost += ds * solver.storage_net_cost(
                iso, sc, cfg, year=year) * dem_twh * 1e6

    inc_cost += h2_mw * capex_yr * 1000 + h2_mwh * fuel_mwh_rate
    total_cost = inc_cost + g_cost

    return {
        "target": target,
        "year": year,
        "pathway": pathway,
        "achieved_cfe": round(float(winner_cfe), 2),
        "total_cost_B": round(total_cost / 1e9, 4),
        "clean_inc_cost_B": round(inc_cost / 1e9, 4),
        "gas_cost_B": round(g_cost / 1e9, 4),
        "gas_mw": round(g_mw, 1),
        "h2_mw": round(h2_mw, 1),
        "h2_mwh": round(h2_mwh, 1),
        "dem_twh": round(dem_twh, 2),
        "thermal_peak_credit_mw": round(thermal_peak_credit, 1),
        # Per-resource build percentages (absolute, not delta)
        "winner_pcts": {solver.RESOURCE_ORDER[ri]: round(float(winner_pcts[ri]), 3)
                        for ri in range(solver.N_RESOURCES)},
        "winner_stor": {solver.STORAGE_COLS[j]: round(float(winner_stor[j]), 3)
                        for j in range(solver.N_STORAGE)},
        # Per-resource incremental TWh above baseline
        "resource_twh": {k: round(v, 3) for k, v in resource_twh.items()},
    }


# ---------------------------------------------------------------------------
# Main driver — loop over thresholds for a single ISO + pathway
# ---------------------------------------------------------------------------
def run_myopic(iso, pathway, thresholds, scenario="base",
               demand_growth="Medium", popsize=15, maxiter=150,
               workers=1, seed=42, gas_mode=2):
    """Run myopic optimization for each threshold independently.

    Each threshold starts from the grid baseline — no ratchet between
    thresholds.  Returns list of per-threshold result dicts.
    """
    sc = solver.COST_SCENARIOS[scenario]
    # Pathway is used for: (a) noak_pathway_key (Wright's Law timeline for
    # firm resources), (b) offshore_noak_cap, (c) free_resources property
    # (which we bypass via pathway_free_indices).  We pass the real pathway
    # to RunConfig so cost lookups are pathway-consistent.
    cfg = solver.RunConfig(
        iso=iso, pathway=pathway, scenario_name=scenario,
        demand_growth=demand_growth, cost_mode=1, **sc)

    demand_vec = solver.demand_twh_vec(iso, demand_growth)
    gaf = float(pc.GAS_AVAILABILITY_FACTOR[iso])
    free_idx = pathway_free_indices(iso, pathway)
    n_free = len(free_idx)
    baseline = pc.GRID_MIX_SHARES[iso]

    # Base-year demand (for converting baseline grid mix pct -> TWh)
    base_demand_twh = float(pc.REGIONAL_DEMAND_TWH[iso])

    # Load profiles once — shared across all thresholds for this ISO
    du.warm_dispatch_kernels()
    demand_norm, supply_profiles, supply_matrix = load_iso_data(iso)

    # Grid baseline floor — reused fresh for every threshold (no ratchet)
    base_floor_pcts = np.zeros(solver.N_RESOURCES, dtype=np.float64)
    for ri, res in enumerate(solver.RESOURCE_ORDER):
        base_floor_pcts[ri] = baseline.get(res, 0.0)
    base_floor_stor = np.zeros(solver.N_STORAGE, dtype=np.float64)

    # Pathway A: zero out firm resources that A excludes entirely.
    # These are not just absent from free_idx — they must be zeroed in
    # the floor so they don't contribute to CFE scoring.
    if pathway == "A":
        for res in ["offshore_wind", "geothermal", "ccs_ccgt"]:
            base_floor_pcts[solver.RES_IDX[res]] = 0.0

    # Score actual baseline CFE (with pathway-appropriate floor)
    baseline_cfe, _ = score_cfe(
        pcts_to_dict(base_floor_pcts), 0, 0, 0,
        demand_norm, supply_profiles, supply_matrix)

    GAS_MODE_LABELS = {
        2: "gas cost ON in optimizer",
        3: "gas cost OFF in optimizer (blind)",
    }

    print(f"\n{'='*70}")
    print(f"MYOPIC DE Optimizer — THERMAL FLEET variant")
    print(f"ISO={iso}  Pathway={pathway}  Scenario={scenario}  "
          f"Demand={demand_growth}")
    print(f"popsize={popsize}  maxiter={maxiter}  workers={workers}  "
          f"seed={seed}")
    print(f"Gas mode: {gas_mode} — {GAS_MODE_LABELS[gas_mode]}")
    print(f"Thresholds: {thresholds}")
    print(f"Baseline CFE: {baseline_cfe:.1f}%")
    free_names = [solver.RESOURCE_ORDER[ri] for ri in free_idx]
    print(f"Free resources ({len(free_names)}): {free_names}")
    if pathway == "A":
        print(f"  clean_firm FROZEN at baseline "
              f"({baseline.get('clean_firm', 0):.1f}% of base demand)")
    else:
        print(f"  clean_firm EXPANDABLE — uprate cap "
              f"{pc.UPRATE_CAP_TWH[iso]:.1f} TWh @ "
              f"${pc.UPRATE_LCOE[cfg.firm_cost]}/MWh, "
              f"then newbuild at Wright's Law")
    print(f"{'='*70}\n")

    results = []
    t0 = time.time()

    for target in thresholds:
        _tt = time.time()
        year = threshold_to_year(target)
        yi = year - solver.BASE_YEAR

        dem_twh = demand_vec[yi]

        # Thermal fleet retirement keyed to target CFE — by the time the
        # grid reaches X% CFE, the coal/oil fleet has already responded
        # via the economic retirement sigmoid.
        thermal_peak = pc.thermal_peak_credit_mw(iso, year, target)
        yr_fleet = pc.thermal_fleet_mw(iso, year, target)

        # Fresh floor from baseline for each threshold (no ratchet).
        # base_floor_pcts already has pathway A exclusions applied.
        floor_pcts = base_floor_pcts.copy()
        floor_stor = base_floor_stor.copy()

        # Frozen resources: adjust baseline TWh for demand growth at this
        # year.  Hydro stays at baseline nameplate TWh.
        hydro_twh = baseline.get("hydro", 0) / 100.0 * base_demand_twh
        floor_pcts[solver.RES_IDX["hydro"]] = hydro_twh / dem_twh * 100.0

        # Clean_firm floor: baseline TWh adjusted for demand growth.
        # Pathway A: frozen (not expandable — not in free_idx).
        # Pathway B/C/D: minimum commitment (optimizer can build above).
        cf_twh_base = baseline.get("clean_firm", 0) / 100.0 * base_demand_twh
        if pathway == "A":
            floor_pcts[solver.RES_IDX["clean_firm"]] = (
                cf_twh_base / dem_twh * 100.0)
        else:
            floor_pcts[solver.RES_IDX["clean_firm"]] = max(
                floor_pcts[solver.RES_IDX["clean_firm"]],
                cf_twh_base / dem_twh * 100.0)

        print(f"  Target {target:.1f}% @ year {year} "
              f"(demand={dem_twh:.1f} TWh, "
              f"thermal_peak={thermal_peak:,.0f} MW, "
              f"coal={yr_fleet['coal_mw']:,.0f} MW)")

        result = solve_one_threshold(
            iso=iso, target=target, year=year, pathway=pathway, cfg=cfg,
            free_idx=free_idx, n_free=n_free,
            demand_norm=demand_norm,
            supply_profiles=supply_profiles,
            supply_matrix=supply_matrix,
            dem_twh=dem_twh, floor_pcts=floor_pcts, floor_stor=floor_stor,
            thermal_peak_credit=thermal_peak, gaf=gaf,
            popsize=popsize, maxiter=maxiter, workers=workers,
            seed=seed, gas_mode=gas_mode)

        elapsed = time.time() - _tt
        print(f"    -> CFE={result['achieved_cfe']:.1f}%  "
              f"cost=${result['total_cost_B']:.3f}B  "
              f"gas={result['gas_mw']:,.0f}MW  "
              f"h2={result['h2_mw']:,.0f}MW  "
              f"({elapsed:.1f}s)")
        results.append(result)

    total_elapsed = time.time() - t0
    print(f"\nTotal: {total_elapsed:.1f}s")

    # Summary table
    print(f"\n{'='*70}")
    print(f"Pathway {pathway}  |  {iso}  |  {scenario}  |  gas_mode={gas_mode}")
    print(f"{'Thr%':>6}  {'Year':>4}  {'CFE%':>6}  {'Cost $B':>8}  "
          f"{'Clean $B':>9}  {'Gas $B':>7}  {'Gas MW':>8}  {'H2 MW':>8}")
    print(f"{'-'*6}  {'-'*4}  {'-'*6}  {'-'*8}  "
          f"{'-'*9}  {'-'*7}  {'-'*8}  {'-'*8}")
    for r in results:
        print(f"{r['target']:6.1f}  {r['year']:4d}  {r['achieved_cfe']:6.1f}  "
              f"{r['total_cost_B']:8.3f}  {r['clean_inc_cost_B']:9.3f}  "
              f"{r['gas_cost_B']:7.3f}  {r['gas_mw']:8,.0f}  "
              f"{r['h2_mw']:8,.0f}")
    print(f"{'='*70}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Myopic DE optimizer — independent CFE threshold solves "
                    "with pathway A/B/C/D resource constraints")
    ap.add_argument("--iso", required=True)
    ap.add_argument("--pathway", required=True, choices=["A", "B", "C", "D"],
                    help="A = VRE only (no firm expansion). "
                         "B = all resources (proactive NOAK 2035). "
                         "C = all resources (delayed NOAK 2040). "
                         "D = all resources (further-delayed NOAK 2045).")
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=DEFAULT_THRESHOLDS,
                    help="CFE thresholds to solve "
                         "(default: 90 92.5 95 97.5 99 99.9)")
    ap.add_argument("--popsize", type=int, default=15,
                    help="DE population multiplier (pop = popsize x n_dims). "
                         "Default 15")
    ap.add_argument("--maxiter", type=int, default=150,
                    help="DE generations per threshold. Default 150")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel objective evaluations. Default 1 (serial)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scenario", default="base",
                    choices=list(solver.COST_SCENARIOS.keys()))
    ap.add_argument("--demand-growth", default="Medium",
                    choices=["Low", "Medium", "High"])
    ap.add_argument("--gas-mode", type=int, default=2, choices=[2, 3],
                    help="Gas in optimizer: 2=cost only (default), 3=blind. "
                         "Mode 1 (shadow) not applicable — no prior peak "
                         "exists in a myopic solve.")
    ap.add_argument("--ramp-start-year", type=int, default=2040,
                    help="Year for 90%% target (default 2040)")
    ap.add_argument("--ramp-end-year", type=int, default=2050,
                    help="Year for 99.9%% target (default 2050)")
    args = ap.parse_args()

    iso = args.iso.upper()
    if iso not in pc.ISOS:
        raise SystemExit(f"Unknown ISO: {iso}. Known: {list(pc.ISOS)}")

    # Allow overriding ramp endpoints
    global RAMP_YEAR_START, RAMP_YEAR_END
    RAMP_YEAR_START = args.ramp_start_year
    RAMP_YEAR_END = args.ramp_end_year

    results = run_myopic(
        iso, args.pathway, sorted(args.thresholds),
        scenario=args.scenario, demand_growth=args.demand_growth,
        popsize=args.popsize, maxiter=args.maxiter,
        workers=args.workers, seed=args.seed, gas_mode=args.gas_mode)

    # --- Save outputs ---
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir = PROJECT_ROOT / "data" / "step2.3-de" / "myopic"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filename encodes pathway + all config for collision-free output
    tag = (f"{iso}_pw{args.pathway}_myopic_{args.scenario}"
           f"_{args.demand_growth}"
           f"_p{args.popsize}_i{args.maxiter}_s{args.seed}"
           f"_g{args.gas_mode}_tf")

    # Flat parquet — one row per threshold, resource pcts flattened
    rows = []
    for r in results:
        row = {
            "iso": iso, "pathway": args.pathway,
            "scenario": args.scenario,
            "demand_growth": args.demand_growth,
            "gas_mode": args.gas_mode,
            "popsize": args.popsize, "maxiter": args.maxiter,
            "seed": args.seed,
        }
        for k in ("target", "year", "achieved_cfe", "total_cost_B",
                  "clean_inc_cost_B", "gas_cost_B", "gas_mw", "h2_mw",
                  "h2_mwh", "dem_twh", "thermal_peak_credit_mw"):
            row[k] = r[k]
        for res, pct in r["winner_pcts"].items():
            row[f"pct_{res}"] = pct
        for sc, pct in r["winner_stor"].items():
            row[f"stor_{sc}"] = pct
        for res, twh in r["resource_twh"].items():
            row[f"twh_{res}"] = twh
        rows.append(row)

    table = pa.Table.from_pylist(rows)
    pq_path = out_dir / f"{tag}.parquet"
    pq.write_table(table, pq_path)

    # JSON with full detail (for inspection and downstream analysis)
    json_path = out_dir / f"{tag}.json"
    with open(json_path, "w") as f:
        json.dump({
            "iso": iso, "pathway": args.pathway,
            "scenario": args.scenario,
            "demand_growth": args.demand_growth,
            "gas_mode": args.gas_mode,
            "thresholds": results,
        }, f, indent=2)

    print(f"\nResults: {pq_path} ({pq_path.stat().st_size / 1024:.0f} KB)")
    print(f"Detail:  {json_path}")


if __name__ == "__main__":
    main()
