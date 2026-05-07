#!/usr/bin/env python3
"""
step2_3_de_pathway_tf.py
Year-by-year DE pathway optimizer — THERMAL FLEET DISAGGREGATION variant.

Fork of step2_3_de_pathway.py that replaces the gas-only peak credit with
full thermal fleet (gas + coal + oil) derated for availability. Coal retires
via max(announced schedule, economic sigmoid driven by CFE%). Oil declines
~3%/yr on age. Feedback loop: more clean → lower LMP proxy → faster coal
retirement → more peak gap → optimizer must fill with clean or new gas.

Output files tagged `_tf` and written to data/step2.3-de/thermal-fleet/.

CFE trajectory: smooth linear ramp from actual baseline (dispatch_utils scored)
to --target-2040 (default 90%) to --target-2050 (default 99.9%).
No arbitrary waypoints.

Gas modes (--gas-mode):
    1 = optimizer sees gas cost + stranding shadow (default)
    2 = optimizer sees gas cost only (shadow OFF)
    3 = optimizer blind to gas economics
    All modes: real gas cost + shadow always included in reported results.
    The distinction is what the optimizer considers vs. what physically exists.

Known limitations:
    - Whole-ISO framing: assumes the entire grid follows the modeled
      procurement strategy. prev_cfe drives coal retirement as a proxy
      for grid-wide clean penetration, not a single buyer's portfolio.
    - Load shape is frozen at the base year. Year-to-year demand growth
      scales magnitude (dem_twh) but not the hourly profile. Electrification
      or data-center load shifts are not captured.
    - No resource degradation. The ratchet preserves nameplate TWh across
      years; solar (~0.5%/yr) and battery (~0.5-1%/yr) degradation are not
      modeled. This introduces a small optimistic bias on absolute cost
      (~3-5% by 2050) that largely cancels in pathway comparisons.

Usage:
    # Single pathway
    python scripts/step2_3_de_pathway_tf.py --iso ERCOT --pathway A

    # Pathway B seeded with A's results (guarantees B ≤ A)
    python scripts/step2_3_de_pathway_tf.py --iso ERCOT --pathway B \\
        --ref-winners data/step2.3-de/thermal-fleet/ERCOT_pathwayA_base_Medium_p15_i150_s42_g1_tf_winners.json

    # Production settings
    python scripts/step2_3_de_pathway_tf.py --iso PJM --pathway B \\
        --popsize 20 --maxiter 200 --workers 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from collections import namedtuple
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

# Stub missing import for dispatch_utils
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
PENALTY_WEIGHT = 1e12        # $/pp² for CFE shortfall and overshoot
MAX_RESOURCE_PCT = 200.0     # hard ceiling per resource dimension
STORAGE_MAX_PCT = 50.0       # generous storage cap — cost function regulates
EARLY_EXIT_GAP_PP = 10.0     # skip cost loop if CFE misses target by >10pp

# Vintage ledger entry. Two schemas share one type:
#   Generation: quantity = delta_TWh,       unit_cost = LCOE ($/MWh)
#   Storage:    quantity = annual_cost ($),  unit_cost = delta_fraction (dimensionless)
VintageEntry = namedtuple("VintageEntry", ["year", "resource", "quantity", "unit_cost"])


# ---------------------------------------------------------------------------
# Profile loading (once per ISO)
# ---------------------------------------------------------------------------
def load_iso_data(iso):
    """Load demand + supply profiles and precompute supply matrix.

    Note: the hourly load *shape* is frozen at the base year. Year-to-year
    demand growth scales the magnitude (dem_twh) but not the profile shape.
    """
    demand_data = load_demand_profiles()
    gen_profiles = load_generation_profiles()
    demand_norm, total_mwh = du.get_demand_profile(iso, demand_data)
    hybrid_profiles = du._load_hybrid_profiles(iso)
    supply_profiles = du.get_supply_profiles(
        iso, gen_profiles, include_hybrids=True,
        hybrid_profiles=hybrid_profiles)
    # Ensure arrays
    for k in supply_profiles:
        if isinstance(supply_profiles[k], list):
            supply_profiles[k] = np.array(supply_profiles[k], dtype=np.float64)
    # Precompute matrix for ~3x dispatch speedup
    supply_matrix = du.build_supply_matrix(
        supply_profiles, resource_types=du.RESOURCE_TYPES_HYBRID)
    return demand_norm, supply_profiles, supply_matrix


# ---------------------------------------------------------------------------
# CFE scoring via dispatch_utils
# ---------------------------------------------------------------------------
def score_cfe(res_pcts, b4_pct, b8_pct, ldes_pct,
              demand_norm, supply_profiles, supply_matrix):
    """Score portfolio CFE using canonical dispatch. Returns (cfe%, residual array)."""
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


# ---------------------------------------------------------------------------
# Fast p9997: O(n) via np.partition instead of O(n log n) full sort
# ---------------------------------------------------------------------------
def _p9997(arr):
    """3rd-largest value (p99.97 proxy for 8760 hours). O(n) via partition."""
    n = len(arr)
    if n < 3:
        return 0.0
    return float(np.partition(arr, -3)[-3])


# ---------------------------------------------------------------------------
# H2 peaker sizing — analytical O(n log n) replaces 60-iteration binary search
# ---------------------------------------------------------------------------
def size_h2(residual_demand, demand_norm, cfe_pre, target_cfe, dem_twh):
    """Find minimum H2 capacity to close CFE gap.

    Analytical solution: sort residual descending, walk prefix sums to find
    the capacity threshold. One sort + one linear scan replaces 60 numpy
    passes from the binary search.
    """
    if cfe_pre >= target_cfe:
        return {"h2_mw": 0.0, "h2_mwh": 0.0, "resid_p9997": _p9997(residual_demand)}

    total_demand = np.sum(demand_norm)
    needed_norm = (target_cfe - cfe_pre) / 100.0 * total_demand
    total_gap = float(residual_demand.sum())

    if total_gap < needed_norm * 0.999:
        return {"h2_mw": np.inf, "h2_mwh": 0.0, "resid_p9997": np.inf}

    # Sort descending — one O(n log n) step replaces 60 × O(n) numpy passes
    sorted_desc = np.sort(residual_demand)[::-1]
    n = len(sorted_desc)

    # Walk sorted array to find minimum capacity C such that
    # sum(min(resid_h, C) for all h) >= needed_norm.
    #
    # If C >= sorted_desc[0], all hours contribute their full value.
    # If C = sorted_desc[k], hours 0..k-1 are capped at C, rest contribute full.
    # Total dispatched = k * C + sum(sorted_desc[k:])
    #
    # We find the breakpoint k where reducing C below sorted_desc[k-1]
    # would drop total below needed.
    h2_cap_norm = sorted_desc[0]  # start with max (trivially sufficient)

    running_prefix = 0.0
    for k in range(n):
        val = sorted_desc[k]
        # If C = val: hours 0..k contribute val each, rest contribute actual
        remaining_tail = total_gap - running_prefix - val
        total_disp = (k + 1) * val + remaining_tail

        if total_disp < needed_norm:
            # C must be between sorted_desc[k-1] and sorted_desc[k]
            if k == 0:
                h2_cap_norm = needed_norm / n
            else:
                # k hours capped at C, rest contribute full value
                hours_below = total_gap - running_prefix  # sum(sorted_desc[k:])
                h2_cap_norm = (needed_norm - hours_below) / k
            break

        running_prefix += val
    else:
        # Walked entire array without breaking — all hours contribute full value
        # at minimum capacity = sorted_desc[-1]. Edge case.
        h2_cap_norm = sorted_desc[-1]

    h2_disp = np.minimum(residual_demand, h2_cap_norm)
    h2_mw = h2_cap_norm * dem_twh * 1e6
    h2_mwh = float(h2_disp.sum()) * dem_twh * 1e6

    post_resid = np.maximum(residual_demand - h2_disp, 0.0)
    return {"h2_mw": h2_mw, "h2_mwh": h2_mwh, "resid_p9997": _p9997(post_resid)}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def pcts_to_dict(pcts_arr):
    """Convert resource pcts array to dict for dispatch_utils."""
    return {res: float(pcts_arr[ri])
            for ri, res in enumerate(solver.RESOURCE_ORDER)
            if pcts_arr[ri] > 0.001}


# ---------------------------------------------------------------------------
# Picklable DE objective (enables workers > 1)
# ---------------------------------------------------------------------------
class _DEObjective:
    """Callable wrapper that scipy.optimize.differential_evolution can pickle
    when ``workers > 1`` triggers multiprocessing.

    v2.0 optimizations vs. v1.x:
    - Precomputed LCOE and storage cost arrays (eliminates per-eval function calls)
    - Pre-allocated work buffers when workers=1 (skipped for multiprocessing safety)
    - Early-exit penalty when CFE misses target by >EARLY_EXIT_GAP_PP
    """

    __slots__ = (
        'lo', 'hi', 'floor_pcts', 'free_idx', 'n_free',
        'demand_norm', 'supply_profiles', 'supply_matrix',
        'dem_twh', 'cfg', 'iso', 'year', 'target', 'cfe_ceiling',
        'capex_yr', 'fuel_mwh_rate', 'thermal_peak_credit', 'gaf', 'peak_gas_mw',
        'floor_stor', 'full_x0', 'active', 'gas_mode',
        # Precomputed cost arrays (v2.0)
        '_lcoe_by_free_k', '_stor_net',
        # Pre-allocated buffers (workers=1 only)
        '_buf_full', '_buf_pcts', '_use_buffers',
        # Clean firm uprate tranche (v2.1)
        '_cf_free_k', '_remaining_uprate_pct', '_uprate_lcoe', '_newbuild_lcoe',
    )

    def __init__(self, *, lo, hi, floor_pcts, free_idx, n_free,
                 demand_norm, supply_profiles, supply_matrix,
                 dem_twh, cfg, iso, year, target, cfe_ceiling,
                 capex_yr, fuel_mwh_rate, thermal_peak_credit, gaf, peak_gas_mw,
                 floor_stor, full_x0, active, gas_mode=1, workers=1):
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
        self.peak_gas_mw = peak_gas_mw
        self.floor_stor = floor_stor
        self.full_x0 = full_x0
        self.active = active
        self.gas_mode = gas_mode

        # --- Precompute year-constant costs (v2.0) ---
        # These depend only on (iso, resource, year, cfg) — invariant within
        # a year's DE run. Eliminates function call overhead per evaluation.
        self._lcoe_by_free_k = np.array([
            solver.get_resource_lcoe(iso, solver.RESOURCE_ORDER[ri], year, cfg)
            for ri in free_idx], dtype=np.float64)

        self._stor_net = np.array([
            solver.storage_net_cost(iso, sc, cfg, year=year)
            for sc in solver.STORAGE_COLS], dtype=np.float64)

        # --- Clean firm uprate tranche (v2.1) ---
        # First UPRATE_CAP_TWH of increment above existing fleet baseline
        # is priced at UPRATE_LCOE (no TX adder); remainder at Wright's Law
        # newbuild cost (with TX).  _cf_free_k = -1 when clean_firm is frozen
        # (Pathway A), so the special pricing is never triggered.
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
            remaining_twh = max(0.0, baseline_cf_twh + uprate_cap_twh - floor_cf_twh)
            self._remaining_uprate_pct = remaining_twh / dem_twh * 100.0
            self._uprate_lcoe = float(pc.UPRATE_LCOE[cfg.firm_cost])
            # Newbuild = what get_resource_lcoe already returns (Wright's Law + TX)
            self._newbuild_lcoe = self._lcoe_by_free_k[self._cf_free_k]
        else:
            self._remaining_uprate_pct = 0.0
            self._uprate_lcoe = 0.0
            self._newbuild_lcoe = 0.0

        # --- Pre-allocate work buffers if single-threaded (v2.0) ---
        # With workers > 1, each process gets its own copy via pickle,
        # so shared buffers would be unsafe.
        self._use_buffers = (workers <= 1)
        if self._use_buffers:
            self._buf_full = full_x0.copy()
            self._buf_pcts = floor_pcts.copy()
        else:
            self._buf_full = None
            self._buf_pcts = None

    def __call__(self, x_active):
        """Map active-dim vector → full vector → cost."""
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
        floor_pcts, free_idx, n_free = self.floor_pcts, self.free_idx, self.n_free

        xc = np.clip(x, lo, hi)

        if self._use_buffers:
            pcts = self._buf_pcts
            np.copyto(pcts, floor_pcts)
        else:
            pcts = floor_pcts.copy()

        for k, ri in enumerate(free_idx):
            pcts[ri] = xc[k]

        cfe, resid = score_cfe(
            pcts_to_dict(pcts), xc[n_free], xc[n_free+1], xc[n_free+2],
            self.demand_norm, self.supply_profiles, self.supply_matrix)

        h2 = size_h2(resid, self.demand_norm, cfe, self.target, self.dem_twh)
        if h2["h2_mw"] > 1e8:
            return 1e18

        post_cfe = cfe + (h2["h2_mwh"] / (self.dem_twh * 1e6) * 100
                          if h2["h2_mwh"] > 0 else 0)

        # --- Early-exit penalty (v2.0) ---
        # If CFE misses target by a wide margin, the penalty dominates
        # regardless of cost details. Skip the cost loop entirely.
        if post_cfe < self.target - EARLY_EXIT_GAP_PP:
            return PENALTY_WEIGHT * (self.target - post_cfe) ** 2

        # --- Incremental cost above floor (precomputed LCOEs, v2.0) ---
        # v2.1: clean_firm uses piecewise pricing — uprate tranche at
        # UPRATE_LCOE, remainder at Wright's Law newbuild.
        cost = 0.0
        dem_twh = self.dem_twh
        cf_k = self._cf_free_k
        for k, ri in enumerate(free_idx):
            delta = pcts[ri] - floor_pcts[ri]
            if delta > 0:
                if k == cf_k:
                    # Piecewise: uprate tranche (cheap) then newbuild (expensive)
                    uprate_d = min(delta, self._remaining_uprate_pct)
                    newbuild_d = delta - uprate_d
                    cost += uprate_d / 100 * dem_twh * self._uprate_lcoe * 1e6
                    cost += newbuild_d / 100 * dem_twh * self._newbuild_lcoe * 1e6
                else:
                    cost += delta / 100 * dem_twh * self._lcoe_by_free_k[k] * 1e6

        for j in range(solver.N_STORAGE):
            ds = (xc[n_free + j] - self.floor_stor[j]) / 100
            if ds > 0:
                cost += ds * self._stor_net[j] * dem_twh * 1e6

        cost += h2["h2_mw"] * self.capex_yr * 1000 + h2["h2_mwh"] * self.fuel_mwh_rate

        g = solver.gas_need_mw(h2["resid_p9997"], dem_twh,
                               self.thermal_peak_credit, self.gaf)
        # Gas mode: 1=both on, 2=gas cost only, 3=both off
        if self.gas_mode <= 2:
            cost += solver.gas_annual_cost(g, self.iso, self.cfg)
        if self.gas_mode == 1:
            cost += solver.gas_stranding_shadow(g, self.peak_gas_mw,
                                                self.year, self.iso)

        # Penalties
        if post_cfe < self.target:
            cost += PENALTY_WEIGHT * (self.target - post_cfe) ** 2
        if cfe > self.cfe_ceiling:
            cost += PENALTY_WEIGHT * (cfe - self.cfe_ceiling) ** 2

        return cost


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------
def run_de_pathway(iso, pathway, scenario="base", demand_growth="Medium",
                   popsize=10, maxiter=100, workers=1, seed=42,
                   target_2040=90.0, target_2050=99.9,
                   reference_winners=None, gas_mode=1):
    """Run year-by-year DE pathway optimization.

    Args:
        iso: ISO region name
        pathway: A, B, C, or D
        popsize: DE population size multiplier (pop = popsize × n_dims)
        maxiter: DE generations per year-step
        workers: parallel objective evaluations (1=serial, 2+=parallel)
        seed: RNG seed for reproducibility
        target_2040: CFE% target at 2040. Linear ramp from baseline.
        target_2050: CFE% target at 2050. Linear ramp from target_2040.
        reference_winners: dict {year: (pcts_array, stor_array)} from prior
            pathway run. Seeds B's DE population with A's solution to
            guarantee B ≤ A.
        gas_mode: 1 = gas cost + shadow on (default),
                  2 = gas cost on / shadow off,
                  3 = both off

    Returns:
        (year_results_list, winners_by_year_dict)
    """
    sc = solver.COST_SCENARIOS[scenario]
    cfg = solver.RunConfig(
        iso=iso, pathway=pathway, scenario_name=scenario,
        demand_growth=demand_growth, cost_mode=1, **sc)

    demand_vec = solver.demand_twh_vec(iso, demand_growth)
    cfe_targets = solver.cfe_target_vec(iso).copy()  # copy — we'll override early years
    # Thermal fleet: computed per-year with CFE feedback (replaces static existing_gas)
    gaf = float(pc.GAS_AVAILABILITY_FACTOR[iso])
    free_idx = cfg.free_indices
    n_free = len(free_idx)
    baseline = pc.GRID_MIX_SHARES[iso]

    # Load profiles + precomputed supply matrix
    du.warm_dispatch_kernels()
    demand_norm, supply_profiles, supply_matrix = load_iso_data(iso)

    # Initialize floor from grid baseline
    floor_pcts = np.zeros(solver.N_RESOURCES, dtype=np.float64)
    for ri, res in enumerate(solver.RESOURCE_ORDER):
        floor_pcts[ri] = baseline.get(res, 0.0)
    floor_stor = np.zeros(solver.N_STORAGE, dtype=np.float64)

    rng = np.random.default_rng(seed)
    vintage_ledger = []
    cumulative_cost = 0.0
    peak_gas_mw = 0.0
    peak_h2_mw = 0.0
    year_results = []
    winners_by_year = {}

    base_demand_twh = float(pc.REGIONAL_DEMAND_TWH[iso])
    baseline_cf_twh = baseline.get("clean_firm", 0) / 100.0 * base_demand_twh

    frozen_twh = {"hydro": baseline.get("hydro", 0) / 100.0 * base_demand_twh}
    if pathway == "A":
        frozen_twh["clean_firm"] = baseline.get("clean_firm", 0) / 100.0 * base_demand_twh

    # Score actual baseline CFE using dispatch_utils
    baseline_cfe, _ = score_cfe(
        pcts_to_dict(floor_pcts), 0, 0, 0,
        demand_norm, supply_profiles, supply_matrix)

    # Build smooth linear ramp from actual baseline → target_2040 → target_2050.
    # Replaces arbitrary waypoints that were calibrated to the simplified kernel.
    for yi_t, year_t in enumerate(solver.YEARS):
        if year_t <= 2040:
            frac = (year_t - solver.BASE_YEAR) / (2040 - solver.BASE_YEAR)
            cfe_targets[yi_t] = baseline_cfe + frac * (target_2040 - baseline_cfe)
        else:
            frac = (year_t - 2040) / (2050 - 2040)
            cfe_targets[yi_t] = target_2040 + frac * (target_2050 - target_2040)

    # Start at 2026 (yi=1). 2025 is baseline year with no builds.
    START_YI = 1

    GAS_MODE_LABELS = {
        1: "gas cost + shadow ON in optimizer",
        2: "gas cost ON / shadow OFF in optimizer",
        3: "both OFF in optimizer (cost-blind)",
    }

    # Whole-ISO framing: prev_cfe tracks grid-wide clean share (not a single
    # buyer's portfolio), so it correctly drives coal retirement via the
    # thermal_fleet_mw sigmoid. See module docstring "Known limitations".
    prev_cfe = baseline_cfe

    print(f"\n{'='*70}")
    print(f"DE Pathway Optimizer — THERMAL FLEET variant (dispatch_utils scoring)")
    print(f"ISO={iso}  Pathway={pathway}  Scenario={scenario}  Demand={demand_growth}")
    print(f"popsize={popsize}  maxiter={maxiter}  workers={workers}  seed={seed}")
    print(f"Gas mode: {gas_mode} — {GAS_MODE_LABELS[gas_mode]}")
    print(f"  (real gas cost + shadow always in reported results)")
    print(f"  THERMAL FLEET: gas + coal + oil peak credit (coal retires with CFE feedback)")
    init_fleet = pc.thermal_fleet_mw(iso, solver.BASE_YEAR, baseline_cfe)
    init_peak = pc.thermal_peak_credit_mw(iso, solver.BASE_YEAR, baseline_cfe)
    gas_only_peak = float(pc.EXISTING_GAS_CAPACITY_MW[iso]) * gaf
    print(f"  Baseline thermal fleet: gas={init_fleet['gas_mw']:,.0f} "
          f"coal={init_fleet['coal_mw']:,.0f} oil={init_fleet['oil_mw']:,.0f} MW")
    print(f"  Baseline peak credit: {init_peak:,.0f} MW "
          f"(old gas-only: {gas_only_peak:,.0f} MW, delta: +{init_peak - gas_only_peak:,.0f} MW)")
    print(f"Baseline CFE: {baseline_cfe:.1f}%  → {target_2040:.0f}% by 2040 → {target_2050:.0f}% by 2050")
    print(f"Annual ramp: {(target_2040 - baseline_cfe) / 15:.1f}pp/yr (2025-2040)"
          f"  {(target_2050 - target_2040) / 10:.1f}pp/yr (2040-2050)")
    print(f"{'='*70}\n")

    t0 = time.time()

    for yi in range(START_YI, solver.N_YEARS):
        _yt = time.time()
        year = solver.YEARS[yi]
        dem_twh = demand_vec[yi]
        target = cfe_targets[yi]

        # Thermal fleet: compute remaining fleet and peak credit using prior year's CFE
        yr_fleet = pc.thermal_fleet_mw(iso, year, prev_cfe)
        yr_thermal_peak = pc.thermal_peak_credit_mw(iso, year, prev_cfe)

        # Frozen resource pct adjustments (fixed TWh / changing demand)
        for res, twh in frozen_twh.items():
            floor_pcts[solver.RES_IDX[res]] = twh / dem_twh * 100.0

        # Pathway constraints
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

        # H2 cost parameters — year-constant, used by both DE objective and
        # post-DE tally. Computed once here to avoid duplication.
        capex_yr = solver.h2_peaker_capex_kw_yr(year, cfg)
        fuel_mwh_rate = solver.h2_peaker_fuel_mwh(year, cfg)

        # Score floor
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
            # --- Build bounds ---
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

            # --- Filter zero-width dims ---
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

                obj_fn = _DEObjective(
                    lo=lo, hi=hi, floor_pcts=floor_pcts,
                    free_idx=free_idx, n_free=n_free,
                    demand_norm=demand_norm,
                    supply_profiles=supply_profiles,
                    supply_matrix=supply_matrix,
                    dem_twh=dem_twh, cfg=cfg, iso=iso, year=year,
                    target=target, cfe_ceiling=cfe_ceiling,
                    capex_yr=capex_yr, fuel_mwh_rate=fuel_mwh_rate,
                    thermal_peak_credit=yr_thermal_peak, gaf=gaf,
                    peak_gas_mw=peak_gas_mw, floor_stor=floor_stor,
                    full_x0=full_x0, active=active, gas_mode=gas_mode,
                    workers=workers)

                de_seed = int(rng.integers(0, 2**31))

                # Seed population with reference pathway winner if available
                init_pop = "latinhypercube"
                if reference_winners and year in reference_winners:
                    ref_pcts, ref_stor = reference_winners[year]
                    ref_full = np.concatenate([
                        np.array([ref_pcts[ri] for ri in free_idx]),
                        ref_stor.copy()])
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
                    obj_fn, bounds,
                    seed=de_seed, maxiter=maxiter, popsize=popsize,
                    tol=0, mutation=(0.5, 1.0), recombination=0.7,
                    polish=True, init=init_pop,
                    workers=workers)

                # Reconstruct winner
                full_best = full_x0.copy()
                for ai, idx in enumerate(active):
                    full_best[idx] = de_result.x[ai]
                full_best = np.clip(full_best, lo, hi)

                winner_pcts = floor_pcts.copy()
                for k, ri in enumerate(free_idx):
                    winner_pcts[ri] = full_best[k]
                winner_stor = full_best[n_free:].copy()

                # Final scoring
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
        g_mw = solver.gas_need_mw(resid_for_gas, dem_twh, yr_thermal_peak, gaf)
        g_cost = solver.gas_annual_cost(g_mw, iso, cfg)
        g_shadow = solver.gas_stranding_shadow(g_mw, peak_gas_mw, year, iso)

        # --- Annual cost (all vintages + new) ---
        # v2.1: clean_firm uses piecewise pricing in tally to match optimizer.
        cf_ri = solver.RES_IDX["clean_firm"]
        baseline_cf_twh = (pc.GRID_MIX_SHARES[iso].get("clean_firm", 0.0)
                           / 100.0 * float(pc.REGIONAL_DEMAND_TWH[iso]))
        uprate_cap_twh = float(pc.UPRATE_CAP_TWH[iso])
        uprate_lcoe = float(pc.UPRATE_LCOE[cfg.firm_cost])

        year_inc = 0.0
        for k, ri in enumerate(free_idx):
            delta = winner_pcts[ri] - floor_pcts[ri]
            if delta > solver.RATCHET_TOL_PCT:
                if ri == cf_ri:
                    # Piecewise: uprate tranche then newbuild
                    floor_cf_twh = floor_pcts[ri] / 100 * dem_twh
                    remaining_twh = max(0.0, baseline_cf_twh + uprate_cap_twh - floor_cf_twh)
                    remaining_pct = remaining_twh / dem_twh * 100.0
                    uprate_d = min(delta, remaining_pct)
                    newbuild_d = delta - uprate_d
                    newbuild_lcoe = solver.get_resource_lcoe(iso, "clean_firm", year, cfg)
                    year_inc += uprate_d / 100 * dem_twh * uprate_lcoe * 1e6
                    year_inc += newbuild_d / 100 * dem_twh * newbuild_lcoe * 1e6
                else:
                    year_inc += delta / 100 * dem_twh * solver.get_resource_lcoe(
                        iso, solver.RESOURCE_ORDER[ri], year, cfg) * 1e6
        for j, sc in enumerate(solver.STORAGE_COLS):
            ds = (winner_stor[j] - floor_stor[j]) / 100
            if ds > solver.RATCHET_TOL_PCT:
                year_inc += ds * solver.storage_net_cost(iso, sc, cfg, year=year) * dem_twh * 1e6
        year_inc += h2_mw * capex_yr * 1000 + h2_mwh * fuel_mwh_rate
        # Real gas operating cost always counted; shadow is solver-only (never in tally)
        year_inc += g_cost

        _sn = set(solver.STORAGE_COLS)
        vintage_cost = sum(
            v.quantity if v.resource in _sn else v.quantity * v.unit_cost * 1e6
            for v in vintage_ledger)
        total_annual = vintage_cost + year_inc
        cumulative_cost += total_annual

        if g_mw > peak_gas_mw:
            peak_gas_mw = g_mw
        if h2_mw > peak_h2_mw:
            peak_h2_mw = h2_mw

        # Record vintages — clean_firm decomposed into uprate + newbuild tranches
        for k, ri in enumerate(free_idx):
            delta = winner_pcts[ri] - floor_pcts[ri]
            if delta > solver.RATCHET_TOL_PCT:
                res = solver.RESOURCE_ORDER[ri]
                if ri == cf_ri:
                    # Decompose into uprate + newbuild tranches for vintage tracking
                    floor_cf_twh_v = floor_pcts[ri] / 100 * dem_twh
                    remaining_twh_v = max(0.0, baseline_cf_twh + uprate_cap_twh - floor_cf_twh_v)
                    remaining_pct_v = remaining_twh_v / dem_twh * 100.0
                    uprate_d_v = min(delta, remaining_pct_v)
                    newbuild_d_v = delta - uprate_d_v
                    if uprate_d_v > solver.RATCHET_TOL_PCT:
                        vintage_ledger.append(VintageEntry(
                            year, "clean_firm_uprate",
                            uprate_d_v / 100 * dem_twh, uprate_lcoe))
                    if newbuild_d_v > solver.RATCHET_TOL_PCT:
                        vintage_ledger.append(VintageEntry(
                            year, "clean_firm_newbuild",
                            newbuild_d_v / 100 * dem_twh,
                            solver.get_resource_lcoe(iso, "clean_firm", year, cfg)))
                else:
                    vintage_ledger.append(VintageEntry(
                        year, res, delta / 100 * dem_twh,
                        solver.get_resource_lcoe(iso, res, year, cfg)))
        for j, sc in enumerate(solver.STORAGE_COLS):
            ds = (winner_stor[j] - floor_stor[j]) / 100
            if ds > solver.RATCHET_TOL_PCT:
                net = solver.storage_net_cost(iso, sc, cfg, year=year)
                vintage_ledger.append(VintageEntry(
                    year, sc, ds * net * dem_twh * 1e6, ds))

        yt = time.time() - _yt
        print(f"  {year}: CFE={winner_cfe:.1f}% tgt={target:.1f}% "
              f"gas={g_mw:,.0f}MW h2={h2_mw:,.0f}MW "
              f"coal={yr_fleet['coal_mw']:,.0f}MW oil={yr_fleet['oil_mw']:,.0f}MW "
              f"pk_cred={yr_thermal_peak:,.0f}MW "
              f"ann=${total_annual/1e9:.2f}B cum=${cumulative_cost/1e9:.1f}B ({yt:.1f}s)")

        year_results.append({
            "year": year, "target": round(target, 1),
            "achieved_cfe": round(float(winner_cfe), 2),
            "h2_mw": round(h2_mw, 1), "gas_mw": round(g_mw, 1),
            "gas_cost_B": round(g_cost / 1e9, 4),
            "gas_shadow_B": round(g_shadow / 1e9, 4),
            "total_annual_B": round(total_annual / 1e9, 3),
            "cumulative_B": round(cumulative_cost / 1e9, 3),
            "coal_mw": round(yr_fleet['coal_mw'], 1),
            "oil_mw": round(yr_fleet['oil_mw'], 1),
            "thermal_peak_credit_mw": round(yr_thermal_peak, 1),
        })
        winners_by_year[year] = (winner_pcts.copy(), winner_stor.copy())

        # Update CFE feedback for next year's thermal fleet computation
        prev_cfe = float(winner_cfe)

        # Ratchet: lock in builds, adjust for demand growth.
        # Note: preserves nameplate TWh — does not model degradation.
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="DE pathway optimizer — thermal fleet variant")
    ap.add_argument("--iso", required=True)
    ap.add_argument("--pathway", required=True, choices=["A", "B", "C", "D"])
    ap.add_argument("--popsize", type=int, default=15,
                    help="DE population multiplier (pop = popsize × n_dims). Default 15")
    ap.add_argument("--maxiter", type=int, default=150,
                    help="DE generations per year-step. Default 150")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel objective evaluations. Default 1 (serial)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scenario", default="base",
                    choices=list(solver.COST_SCENARIOS.keys()))
    ap.add_argument("--demand-growth", default="Medium",
                    choices=["Low", "Medium", "High"])
    ap.add_argument("--target-2040", type=float, default=90.0,
                    help="CFE%%%% target at 2040. Linear ramp from baseline. Default 90")
    ap.add_argument("--target-2050", type=float, default=99.9,
                    help="CFE%%%% target at 2050. Linear ramp from 2040 target. Default 99.9")
    ap.add_argument("--ref-winners", default=None,
                    help="Path to JSON with reference pathway winners for seeding")
    ap.add_argument("--gas-mode", type=int, default=1, choices=[1, 2, 3],
                    help="Gas visibility in optimizer: "
                         "1=cost+shadow (default), 2=cost only, 3=blind. "
                         "Real gas cost always in reported results.")
    args = ap.parse_args()

    iso = args.iso.upper()
    if iso not in pc.ISOS:
        raise SystemExit(f"Unknown ISO: {iso}. Known: {list(pc.ISOS)}")

    # Load reference winners if provided
    ref = None
    if args.ref_winners:
        with open(args.ref_winners) as f:
            data = json.load(f)
        ref = {int(y): (np.array(p), np.array(s))
               for y, (p, s) in data["winners"].items()}
        print(f"Loaded reference winners from {args.ref_winners}")

    results, winners = run_de_pathway(
        iso, args.pathway,
        scenario=args.scenario, demand_growth=args.demand_growth,
        popsize=args.popsize, maxiter=args.maxiter,
        workers=args.workers, seed=args.seed,
        target_2040=args.target_2040, target_2050=args.target_2050,
        reference_winners=ref, gas_mode=args.gas_mode)

    # Save results as parquet + winners as JSON (for seeding)
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir = PROJECT_ROOT / "data" / "step2.3-de" / "thermal-fleet"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sweep-safe naming: includes popsize/maxiter/seed/gas-mode + _tf tag
    # Subdirectory thermal-fleet/ separates from baseline DE results.
    tag = (f"{iso}_pathway{args.pathway}_{args.scenario}_{args.demand_growth}"
           f"_p{args.popsize}_i{args.maxiter}_s{args.seed}_g{args.gas_mode}_tf")
    results_path = out_dir / f"{tag}.parquet"
    winners_path = out_dir / f"{tag}_winners.json"

    # Build parquet with meta columns + year data
    rows = []
    for yr in results:
        row = {**yr}
        row["iso"] = iso
        row["pathway"] = args.pathway
        row["scenario"] = args.scenario
        row["demand_growth"] = args.demand_growth
        row["popsize"] = args.popsize
        row["maxiter"] = args.maxiter
        row["seed"] = args.seed
        row["gas_mode"] = args.gas_mode
        row["target_2040"] = args.target_2040
        row["target_2050"] = args.target_2050
        rows.append(row)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, results_path)

    serializable = {str(y): (p.tolist(), s.tolist())
                    for y, (p, s) in winners.items()}
    with open(winners_path, "w") as f:
        json.dump({"winners": serializable}, f)

    print(f"\nResults: {results_path} ({results_path.stat().st_size / 1024:.0f} KB)")
    print(f"Winners: {winners_path}")


if __name__ == "__main__":
    main()

