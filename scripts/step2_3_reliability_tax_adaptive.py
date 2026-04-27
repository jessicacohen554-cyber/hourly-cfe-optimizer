#!/usr/bin/env python3
"""
step2_3_reliability_tax_adaptive.py  (v4.0)
On-the-fly mix generation with two-stage adaptive LHS sampling.

Architecture:
  Pre-period: Single-cost from baseline to 2030 waypoint at 2030 LCOEs.
  Stage 1:    Load Step 2.1 EF parquets at the 2030 waypoint band (50%/60%).
              Cluster into archetypes, pick N diverse low-cost seeds.
  Stage 2:    For each seed, build year-by-year pathway from 2030 to 99.9% by
              2050 via two-stage adaptive LHS:
                Round 1 — Coarse screen at marginal-yield bounds (with storage
                          headroom). If <min_hits, widen bounds and resample.
                Round 2 — Dense refinement within tight bounding box derived
                          from Round 1 feasible hits.

Two pathways:
  A — VRE + hybrids + storage only. No offshore, geo, CCS, or clean_firm expansion.
  B — Full P3: all resources incl nuclear, CCS, offshore, geothermal (CAISO only).

ISO-specific CFE waypoints:
  High-baseline (CAISO, SPP, ERCOT):       60% (2030) → 90% (2040) → 95% (2045) → 99.9% (2050)
  Low-baseline  (PJM, NYISO, NEISO, MISO): 50% (2030) → 90% (2040) → 95% (2045) → 99.9% (2050)

Gas shadow cost:
  Always on for Pathway B. Toggle via --gas-shadow for Pathway A.
  When on, stranded gas capex is added to the cost function, steering the
  optimizer toward lower-residual (less gas-dependent) solutions.

Usage:
    python scripts/step2_3_reliability_tax_adaptive.py --iso NYISO --pathway A
    python scripts/step2_3_reliability_tax_adaptive.py --iso ERCOT --pathway B --demand-growth High
    python scripts/step2_3_reliability_tax_adaptive.py --iso CAISO --pathway A --gas-shadow --batch
    python scripts/step2_3_reliability_tax_adaptive.py --iso PJM --pathway B --screen-samples 8000 --refine-samples 15000
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product as iterproduct
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.compute as pac
import pyarrow.parquet as pq

try:
    from numba import njit, prange, float32 as nb_f32
except ImportError:
    print("[adaptive] FATAL: numba required. pip install numba", file=sys.stderr)
    raise SystemExit(1)

try:
    from scipy.stats.qmc import LatinHypercube
except ImportError:
    print("[adaptive] FATAL: scipy required. pip install scipy", file=sys.stderr)
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import pipeline_config as pc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
H = 8760
BASE_YEAR, END_YEAR = 2025, 2050
SOLVE_START_YEAR = 2030

YEARS = list(range(BASE_YEAR, END_YEAR + 1))
N_YEARS = len(YEARS)

SOLVE_YEARS = list(range(SOLVE_START_YEAR, END_YEAR + 1))
N_SOLVE_YEARS = len(SOLVE_YEARS)
SOLVE_YEAR_OFFSET = SOLVE_START_YEAR - BASE_YEAR

DISCOUNT_RATES = {"5pct": 0.05, "7pct": 0.07, "9pct": 0.09}
WORST_HOUR_PCTILE = 99.97
CCGT_OVERNIGHT_CAPEX_USD_KW = 1200.0
GAS_CF = 0.45
RA_MARGIN = pc.RESOURCE_ADEQUACY_MARGIN
RATCHET_TOL_PCT = 0.01
SAFETY_FACTOR = 1.5
WRIGHTS_LAW_K = 5.0
NUCLEAR_FOAK_DISCOUNT = 0.80
CLEAN_FIRM_DEPLOYMENT_SHARE = 0.30

EF_DIR = PROJECT_ROOT / "data" / "step2.1-ef"
OUTPUT_DIR = PROJECT_ROOT / "data" / "step2.3-adaptive"
GEN_PROF_PATH = PROJECT_ROOT / "data" / "eia-930" / "eia_generation_profiles.parquet"
DEMAND_PROF_PATH = PROJECT_ROOT / "data" / "eia-930" / "eia_demand_profiles.parquet"
HYBRID_DIR = PROJECT_ROOT / "data" / "hybrid_profiles"

RESOURCE_ORDER = [
    "clean_firm", "solar", "wind", "hydro", "offshore_wind",
    "ccs_ccgt", "geothermal",
    "solar_batt4", "solar_batt8", "wind_batt4", "wind_batt8",
]
N_RESOURCES = len(RESOURCE_ORDER)
RES_IDX = {r: i for i, r in enumerate(RESOURCE_ORDER)}

STORAGE_COLS = [
    "battery_dispatch_pct", "battery8_dispatch_pct",
    "ldes_dispatch_pct", "h2_dispatch_pct",
]
STORAGE_PARAMS = {
    "battery_dispatch_pct":  (4,    0.85, 24,  "battery",  4),
    "battery8_dispatch_pct": (8,    0.85, 48,  "battery8", 8),
    "ldes_dispatch_pct":     (100,  0.50, 168, "ldes",     100),
    "h2_dispatch_pct":       (1000, 0.35, 720, "h2",       1000),
}

# Storage minimum headroom — ensures LHS explores meaningful storage depth
# regardless of marginal yield. Physically grounded: approximate range where
# each technology is competitive before diminishing returns.
STORAGE_MIN_HEADROOM = {
    "battery_dispatch_pct":  15.0,  # 4hr — most liquid, widest range
    "battery8_dispatch_pct": 10.0,  # 8hr
    "ldes_dispatch_pct":      8.0,  # 100hr
    "h2_dispatch_pct":        5.0,  # 1000hr — expensive, tighter range
}
STORAGE_MIN_HEADROOM_ARR = np.array([
    STORAGE_MIN_HEADROOM[sc] for sc in STORAGE_COLS
], dtype=np.float64)

EF_BAND_THRESHOLDS = (
    10, 20, 30, 40, 50, 55, 60, 65, 70, 75,
    80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 99.5, 99.9,
)

# ---------------------------------------------------------------------------
# Per-ISO CFE waypoints
# ---------------------------------------------------------------------------
HIGH_BASELINE_ISOS = {"CAISO", "SPP", "ERCOT"}


def get_cfe_waypoints(iso: str) -> tuple[tuple[int, float], ...]:
    wp_2030 = 60.0 if iso in HIGH_BASELINE_ISOS else 50.0
    return ((2030, wp_2030), (2040, 90.0), (2045, 95.0), (2050, 99.9))


def get_waypoint_2030(iso: str) -> float:
    return 60.0 if iso in HIGH_BASELINE_ISOS else 50.0


PATHWAY_A_FREE = ["solar", "wind", "solar_batt4", "solar_batt8",
                  "wind_batt4", "wind_batt8"]
PATHWAY_B_FREE = PATHWAY_A_FREE + ["offshore_wind", "clean_firm"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class RunConfig:
    iso: str
    pathway: str
    scenario_name: str = "base"
    demand_growth: str = "Medium"
    ren_cost: str = "M"
    firm_cost: str = "M"
    batt_cost: str = "M"
    ldes_cost: str = "M"
    fuel_cost: str = "M"
    tx_level: str = "M"
    ccs_cost: str = "M"
    geo_cost: str = "M"
    q45: str = "1"
    n_beams: int = 5
    # Two-stage LHS sample sizes
    screen_samples: int = 5000     # Round 1 coarse screen
    widen_samples: int = 10000     # Round 1b if <min_hits
    refine_samples: int = 10000    # Round 2 tight bounding box
    min_hits: int = 50             # Threshold for bounding box vs widen
    bbox_margin: float = 0.15     # Bounding box pad fraction
    # Gas shadow cost
    gas_shadow: bool = False       # Pathway A toggle; always True for B

    @property
    def dim_key(self) -> str:
        return (f"ren_{self.ren_cost}__firm_{self.firm_cost}__"
                f"batt_{self.batt_cost}__ldes_{self.ldes_cost}__"
                f"fuel_{self.fuel_cost}__tx_{self.tx_level}__"
                f"ccs_{self.ccs_cost}__geo_{self.geo_cost}__"
                f"q45_{self.q45}__dg_{self.demand_growth}")

    @property
    def free_resources(self) -> list[str]:
        free = list(PATHWAY_A_FREE)
        if self.pathway == "B":
            if self.iso in pc.OFFSHORE_ISOS:
                free.append("offshore_wind")
            free.append("clean_firm")
        return free

    @property
    def free_indices(self) -> list[int]:
        return [RES_IDX[r] for r in self.free_resources]

    @property
    def n_dims(self) -> int:
        return len(self.free_resources) + 4

    @property
    def use_gas_shadow(self) -> bool:
        """Gas shadow always on for B, toggled for A."""
        return True if self.pathway == "B" else self.gas_shadow

    @property
    def scaled_beams(self) -> int:
        extra = 2 if self.pathway == "B" else 0
        return self.n_beams + extra


COST_SCENARIOS = {
    "base": {
        "ren_cost": "M", "firm_cost": "M", "batt_cost": "M", "ldes_cost": "M",
        "fuel_cost": "M", "tx_level": "M", "ccs_cost": "M", "geo_cost": "M",
        "q45": "1",
    },
    "firm_high_vre_low": {
        "ren_cost": "L", "firm_cost": "H", "batt_cost": "L", "ldes_cost": "L",
        "fuel_cost": "M", "tx_level": "L", "ccs_cost": "H", "geo_cost": "H",
        "q45": "1",
    },
    "firm_low_vre_high": {
        "ren_cost": "H", "firm_cost": "L", "batt_cost": "H", "ldes_cost": "H",
        "fuel_cost": "M", "tx_level": "H", "ccs_cost": "L", "geo_cost": "L",
        "q45": "1",
    },
}

LEVEL_NAME = {"L": "Low", "M": "Medium", "H": "High"}


# ---------------------------------------------------------------------------
# Profile loading
# ---------------------------------------------------------------------------
def _load_gen_profiles(iso: str) -> dict[str, np.ndarray]:
    if not GEN_PROF_PATH.exists():
        return {}
    t = pq.read_table(GEN_PROF_PATH)
    t = t.filter(pac.equal(t["iso"], iso))
    if t.num_rows == 0:
        return {}
    max_yr = pac.max(t["year"]).as_py()
    t = t.filter(pac.equal(t["year"], max_yr))
    hrs = t.column("hour").to_numpy().astype(np.int64)
    vals = t.column("value").to_numpy().astype(np.float64)
    fuels_col = t.column("fuel")
    out = {}
    for fuel in fuels_col.unique().to_pylist():
        mask = pac.equal(fuels_col, fuel).to_numpy()
        prof = np.zeros(H, dtype=np.float64)
        prof[hrs[mask]] = vals[mask]
        out[str(fuel)] = prof
    return out


def _load_hybrid_profiles(iso: str) -> dict[str, np.ndarray]:
    keys = ("solar_batt4", "solar_batt8", "wind_batt4", "wind_batt8")
    npz_path = HYBRID_DIR / f"{iso}_hybrid_profiles.npz"
    if npz_path.exists():
        z = np.load(npz_path)
        return {k: z[k].astype(np.float64) if k in z
                else np.zeros(H, dtype=np.float64) for k in keys}
    return {k: np.zeros(H, dtype=np.float64) for k in keys}


def _load_demand_norm(iso: str) -> np.ndarray:
    flat = np.full(H, 1.0 / H, dtype=np.float64)
    if not DEMAND_PROF_PATH.exists():
        return flat
    t = pq.read_table(DEMAND_PROF_PATH, columns=["iso", "year", "hour", "normalized"])
    t = t.filter(pac.equal(t["iso"], iso))
    if t.num_rows == 0:
        return flat
    max_yr = pac.max(t["year"]).as_py()
    t = t.filter(pac.equal(t["year"], max_yr))
    norm = np.zeros(H, dtype=np.float64)
    norm[t.column("hour").to_numpy().astype(np.int64)] = (
        t.column("normalized").to_numpy().astype(np.float64))
    s = norm.sum()
    return norm / s if s > 0 else flat


def build_profile_matrix(iso: str) -> np.ndarray:
    gen = _load_gen_profiles(iso)
    hyb = _load_hybrid_profiles(iso)
    flat = np.full(H, 1.0 / H, dtype=np.float64)
    zero = np.zeros(H, dtype=np.float64)
    defaults = {
        "clean_firm": ("nuclear", flat), "solar": ("solar", zero),
        "wind": ("wind", zero), "hydro": ("hydro", flat),
        "offshore_wind": ("offshore_wind", zero),
        "ccs_ccgt": (None, flat), "geothermal": ("geothermal", flat),
    }
    rows = []
    for res in RESOURCE_ORDER:
        if res in hyb:
            rows.append(hyb[res])
        elif res in defaults:
            gen_key, default = defaults[res]
            rows.append(gen.get(gen_key, default) if gen_key else default)
        else:
            rows.append(gen.get(res, zero))
    return np.stack(rows, axis=0)


# ---------------------------------------------------------------------------
# Numba kernels (unchanged from v3.7)
# ---------------------------------------------------------------------------
@njit
def _inline_storage_stage(surplus, gap, total_dispatch,
                          capacity, power_rating, efficiency, window_hours):
    if capacity <= 0.0:
        return
    n_windows = (H + window_hours - 1) // window_hours
    for w in range(n_windows):
        ws = w * window_hours
        we = min(ws + window_hours, H)
        soc = 0.0
        for h in range(ws, we):
            s = surplus[h]
            if s > 0.0 and soc < capacity:
                charge = min(s, power_rating, capacity - soc)
                soc += charge
                surplus[h] -= charge
        for h in range(ws, we):
            g = gap[h]
            if g > 0.0 and soc > 0.0:
                discharge = min(g, power_rating, soc * efficiency)
                total_dispatch[h] += discharge
                soc -= discharge / efficiency
                gap[h] -= discharge


@njit
def _inline_storage_peaker(surplus, gap, total_dispatch,
                           capacity, power_rating, efficiency):
    if capacity <= 0.0:
        return
    soc = 0.0
    for h in range(H):
        s = surplus[h]
        if s > 0.0 and soc < capacity:
            charge = min(s, power_rating, capacity - soc)
            soc += charge
            surplus[h] -= charge
        g = gap[h]
        if g > 0.0 and soc > 0.0:
            discharge = min(g, power_rating, soc * efficiency)
            total_dispatch[h] += discharge
            soc -= discharge / efficiency
            gap[h] -= discharge


@njit
def _score_one(i, W, P32, dm32, dn32, b4_val, b8_val, ld_val, h2_val,
               scores, resid, curtail):
    NR = W.shape[1]
    has_stor = (b4_val > 0.0) or (b8_val > 0.0) or (ld_val > 0.0) or (h2_val > 0.0)

    if not has_stor:
        t0 = nb_f32(-1.0); t1 = nb_f32(-1.0); t2 = nb_f32(-1.0)
        curt_sum = nb_f32(0.0)
        matched_sum = nb_f32(0.0)
        for h in range(H):
            s = nb_f32(0.0)
            for r in range(NR):
                s += W[i, r] * P32[r, h]
            m = min(s, dn32[h])
            matched_sum += m
            gap_val = dm32[h] - s
            if gap_val < nb_f32(0.0):
                gap_val = nb_f32(0.0)
            if gap_val > t0:
                t2 = t1; t1 = t0; t0 = gap_val
            elif gap_val > t1:
                t2 = t1; t1 = gap_val
            elif gap_val > t2:
                t2 = gap_val
            surplus_val = s - dn32[h]
            if surplus_val > nb_f32(0.0):
                curt_sum += surplus_val
        scores[i] = matched_sum * nb_f32(100.0)
        resid[i] = t2
        curtail[i] = curt_sum
    else:
        supply = np.empty(H, dtype=np.float64)
        surplus = np.empty(H, dtype=np.float64)
        gap_arr = np.empty(H, dtype=np.float64)
        total_dispatch = np.zeros(H, dtype=np.float64)
        for h in range(H):
            s = 0.0
            for r in range(NR):
                s += float(W[i, r]) * float(P32[r, h])
            supply[h] = s
            diff = s - float(dn32[h])
            if diff > 0.0:
                surplus[h] = diff; gap_arr[h] = 0.0
            else:
                surplus[h] = 0.0; gap_arr[h] = -diff
        b4_c = b4_val * 0.01
        _inline_storage_stage(surplus, gap_arr, total_dispatch,
                              b4_c, b4_c / 4.0, 0.85, 24)
        b8_c = b8_val * 0.01
        _inline_storage_stage(surplus, gap_arr, total_dispatch,
                              b8_c, b8_c / 8.0, 0.85, 48)
        ld_c = ld_val * 0.01
        _inline_storage_stage(surplus, gap_arr, total_dispatch,
                              ld_c, ld_c / 100.0, 0.50, 168)
        h2_c = h2_val * 0.01
        _inline_storage_peaker(surplus, gap_arr, total_dispatch,
                               h2_c, h2_c / 1000.0, 0.35)
        matched_sum = 0.0
        for h in range(H):
            eff = supply[h] + total_dispatch[h]
            matched_sum += min(eff, float(dn32[h]))
        curt_sum = 0.0
        for h in range(H):
            curt_sum += surplus[h]
        t0 = -1.0; t1 = -1.0; t2 = -1.0
        for h in range(H):
            fg = float(dm32[h]) - (supply[h] + total_dispatch[h])
            if fg < 0.0:
                fg = 0.0
            if fg > t0:
                t2 = t1; t1 = t0; t0 = fg
            elif fg > t1:
                t2 = t1; t1 = fg
            elif fg > t2:
                t2 = fg
        scores[i] = nb_f32(matched_sum * 100.0)
        resid[i] = nb_f32(t2)
        curtail[i] = nb_f32(curt_sum)


@njit(parallel=True)
def _score_mixes_parallel(scores, resid, curtail, W, P32, dm32, dn32,
                          batt4, batt8, ldes, h2, compute_idx):
    n_compute = compute_idx.shape[0]
    for idx in prange(n_compute):
        i = compute_idx[idx]
        _score_one(i, W, P32, dm32, dn32,
                   float(batt4[i]), float(batt8[i]),
                   float(ldes[i]), float(h2[i]),
                   scores, resid, curtail)


@njit
def _score_mixes_serial(scores, resid, curtail, W, P32, dm32, dn32,
                        batt4, batt8, ldes, h2):
    n = W.shape[0]
    for i in range(n):
        _score_one(i, W, P32, dm32, dn32,
                   float(batt4[i]), float(batt8[i]),
                   float(ldes[i]), float(h2[i]),
                   scores, resid, curtail)


SERIAL_THRESHOLD = 50


def score_candidates(W, P32, dm32, dn32, batt4, batt8, ldes, h2):
    n = W.shape[0]
    scores = np.full(n, -1.0, dtype=np.float32)
    resid = np.full(n, np.inf, dtype=np.float32)
    curtail = np.zeros(n, dtype=np.float32)
    if n <= SERIAL_THRESHOLD:
        _score_mixes_serial(scores, resid, curtail, W, P32, dm32, dn32,
                            batt4, batt8, ldes, h2)
    else:
        idx = np.arange(n, dtype=np.int64)
        _score_mixes_parallel(scores, resid, curtail, W, P32, dm32, dn32,
                              batt4, batt8, ldes, h2, idx)
    return scores, resid, curtail


# ---------------------------------------------------------------------------
# Wright's Law cost model
# ---------------------------------------------------------------------------
def wrights_law_cost(foak, noak, year, t_start, t_noak):
    if year <= t_start:
        return foak
    if year >= t_noak:
        return noak
    progress = (year - t_start) / (t_noak - t_start)
    return noak + (foak - noak) * np.exp(-WRIGHTS_LAW_K * progress)


def get_resource_lcoe(iso, resource, year, cfg):
    return _base_lcoe(iso, resource, year, cfg) + _tx_adder(iso, resource, cfg.tx_level)


def _base_lcoe(iso, resource, year, cfg):
    ren_name = LEVEL_NAME[cfg.ren_cost]
    if resource == "solar":
        return pc.LCOE_TABLES["solar"][ren_name][iso]
    if resource == "wind":
        return pc.LCOE_TABLES["wind"][ren_name][iso]
    if resource.startswith("solar_batt"):
        return pc.LCOE_TABLES["solar"][ren_name][iso] + {"L":20,"M":30,"H":45}[cfg.batt_cost]
    if resource.startswith("wind_batt"):
        return pc.LCOE_TABLES["wind"][ren_name][iso] + {"L":17,"M":25,"H":38}[cfg.batt_cost]
    if resource == "hydro":
        return 50.0
    fl, cl, gl = cfg.firm_cost, cfg.ccs_cost, cfg.geo_cost
    if resource == "clean_firm":
        foak = pc.FOAK_NUCLEAR_NEWBUILD[iso] * NUCLEAR_FOAK_DISCOUNT
        noak = pc.NUCLEAR_NEWBUILD_LCOE[fl][iso]
        fs, ny = pc.get_pathway_noak_window("nuclear", fl, "3")
        return wrights_law_cost(foak, noak, year, fs, ny)
    if resource == "ccs_ccgt":
        q45_on = cfg.q45 == "1"
        noak = pc.CCS_LCOE_45Q_ON[cl][iso] if q45_on else pc.CCS_LCOE_45Q_OFF[cl][iso]
        foak = pc.FOAK_CCS_45Q_ON[iso] if q45_on else pc.FOAK_CCS_45Q_OFF[iso]
        fs, ny = pc.get_pathway_noak_window("ccs", cl, "3")
        cost = wrights_law_cost(foak, noak, year, fs, ny)
        if iso == "NEISO":
            cost += pc.NEISO_CCS_GAS_ADDER
        return cost
    if resource == "offshore_wind":
        if iso not in pc.OFFSHORE_ISOS:
            return 0.0
        tech = "offshore_wind_float" if iso == "CAISO" else "offshore_wind_fixed"
        fs, ny = pc.LEARNING_PARAMS[tech]["M"]
        return wrights_law_cost(
            pc.FOAK_OFFSHORE_WIND.get(iso, 100),
            pc.NOAK_OFFSHORE_WIND[ren_name].get(iso, 65), year, fs, ny)
    if resource == "geothermal":
        if iso != "CAISO":
            return 0.0
        fs, ny = pc.get_pathway_noak_window("geo", gl, "3")
        return wrights_law_cost(pc.FOAK_GEOTHERMAL, pc.GEOTHERMAL_LCOE[gl],
                                year, fs, ny)
    return 0.0


def _tx_adder(iso, resource, tx_level):
    remap = {"solar_batt4": "solar", "solar_batt8": "solar",
             "wind_batt4": "wind", "wind_batt8": "wind"}
    key = remap.get(resource, resource)
    if key in ("solar", "wind", "clean_firm", "ccs_ccgt", "offshore_wind"):
        return float(pc.get_tx(key, LEVEL_NAME.get(tx_level, tx_level), iso))
    return 0.0


def storage_net_cost(iso, storage_key, cfg):
    dur, _eff, _window, lcoe_key, cost_dur = STORAGE_PARAMS[storage_key]
    if lcoe_key in ("battery", "battery8"):
        cost_name = LEVEL_NAME[cfg.batt_cost]
    else:
        cost_name = LEVEL_NAME[cfg.ldes_cost]
    coeff = pc.LCOE_TABLES[lcoe_key][cost_name][iso]
    credits = getattr(pc, "STORAGE_REVENUE_CREDITS", {})
    credit = credits.get(lcoe_key, {}).get(iso, 0.0)
    return max(0.0, (float(coeff) - float(credit)) * cost_dur)


# ---------------------------------------------------------------------------
# LCOE pre-computation — SOLVE_YEARS (2030-2050)
# ---------------------------------------------------------------------------
def precompute_lcoe_matrix(iso, cfg):
    mat = np.zeros((N_SOLVE_YEARS, N_RESOURCES), dtype=np.float64)
    for yi, year in enumerate(SOLVE_YEARS):
        for ri, res in enumerate(RESOURCE_ORDER):
            mat[yi, ri] = get_resource_lcoe(iso, res, year, cfg)
    return mat


def precompute_storage_costs(iso, cfg):
    return np.array([storage_net_cost(iso, sc, cfg)
                     for sc in STORAGE_COLS], dtype=np.float64)


# ---------------------------------------------------------------------------
# Demand model — full 2025-2050 for indexing
# ---------------------------------------------------------------------------
def demand_twh_vec(iso, level="Medium"):
    base = float(pc.REGIONAL_DEMAND_TWH[iso])
    rate = pc.DEMAND_GROWTH_RATES[iso][level]
    return np.array([base * (1 + rate) ** (y - BASE_YEAR) for y in YEARS])


def peak_demand_vec(iso, level="Medium"):
    base = float(pc.PEAK_DEMAND_MW[iso])
    rate = pc.DEMAND_GROWTH_RATES[iso][level]
    return np.array([base * (1 + rate) ** (y - BASE_YEAR) for y in YEARS])


# ---------------------------------------------------------------------------
# CFE target trajectory — per-ISO waypoints, 2030-2050
# ---------------------------------------------------------------------------
def cfe_target(year, iso):
    waypoints = get_cfe_waypoints(iso)
    pts = list(waypoints)
    if year <= pts[0][0]:
        return pts[0][1]
    if year >= pts[-1][0]:
        return pts[-1][1]
    for (y0, t0), (y1, t1) in zip(pts, pts[1:]):
        if y0 <= year <= y1:
            return t0 + (t1 - t0) * (year - y0) / (y1 - y0)
    return pts[-1][1]


def cfe_target_vec(iso):
    return np.array([cfe_target(y, iso) for y in SOLVE_YEARS])


# ---------------------------------------------------------------------------
# Clean firm tranche decomposition (Pathway B)
# ---------------------------------------------------------------------------
def decompose_clean_firm(cf_twh, iso, year, cfg):
    base_twh = (pc.GRID_MIX_SHARES[iso].get("clean_firm", 0.0) / 100.0
                * float(pc.REGIONAL_DEMAND_TWH[iso]))
    existing = min(cf_twh, base_twh)
    rem = cf_twh - existing
    uc = float(pc.UPRATE_CAP_TWH[iso])
    uprate = min(rem, uc)
    uprate_lcoe = float(pc.UPRATE_LCOE[cfg.firm_cost])
    rem -= uprate
    gc = float(pc.GEOTHERMAL_CAP_TWH) if iso == "CAISO" else 0.0
    geo = min(rem, gc) if gc > 0 else 0.0
    geo_lcoe = _base_lcoe(iso, "geothermal", year, cfg) + _tx_adder(
        iso, "clean_firm", cfg.tx_level)
    rem -= geo
    cc = float(pc.CCS_CAP_TWH.get(iso, 0.0))
    nuke_lcoe = _base_lcoe(iso, "clean_firm", year, cfg) + _tx_adder(
        iso, "clean_firm", cfg.tx_level)
    ccs_lcoe = _base_lcoe(iso, "ccs_ccgt", year, cfg) + _tx_adder(
        iso, "ccs_ccgt", cfg.tx_level)
    if nuke_lcoe <= ccs_lcoe or cc <= 0:
        nuke_new = rem; ccs = 0.0
    else:
        ccs = min(rem, cc); nuke_new = rem - ccs
    return {"existing_twh": existing, "existing_lcoe": 0.0,
            "uprate_twh": uprate, "uprate_lcoe": uprate_lcoe,
            "geo_twh": geo, "geo_lcoe": geo_lcoe,
            "nuke_new_twh": nuke_new, "nuke_new_lcoe": nuke_lcoe,
            "ccs_twh": ccs, "ccs_lcoe": ccs_lcoe}


# ---------------------------------------------------------------------------
# Gas sizing
# ---------------------------------------------------------------------------
def gas_need_mw(resid_p9997, demand_twh, existing_gas_mw, gaf):
    return max(0.0, resid_p9997 * demand_twh * 1e6 / gaf - existing_gas_mw)


def gas_annual_cost(new_gas_mw, iso, cfg):
    if new_gas_mw <= 0:
        return 0.0
    capex_ann = float(pc.NEW_CCGT_COST_KW_YR[iso]) * new_gas_mw * 1000
    fom = float(pc.NEW_CCGT_FOM_KW_YR[iso]) * new_gas_mw * 1000
    fuel_level = LEVEL_NAME[cfg.fuel_cost]
    fuel_mwh = new_gas_mw * GAS_CF * H / 1000
    fuel_cost = fuel_mwh * 1000 * (float(pc.WHOLESALE_PRICES[iso])
                                    + float(pc.FUEL_ADJUSTMENTS[iso][fuel_level]))
    return capex_ann + fom + fuel_cost


# ---------------------------------------------------------------------------
# Marginal yield computation
# ---------------------------------------------------------------------------
def compute_marginal_yields(floor_pcts, floor_storage, free_res_idx,
                            P32, dm32, dn32):
    n_free = len(free_res_idx)
    n_total = 1 + n_free + 4
    base_w = (floor_pcts * 0.01).astype(np.float32)
    W_batch = np.tile(base_w, (n_total, 1))
    for k, ri in enumerate(free_res_idx):
        W_batch[1 + k, ri] += np.float32(0.01)
    sf = floor_storage.astype(np.float32)
    b4 = np.full(n_total, sf[0], dtype=np.float32)
    b8 = np.full(n_total, sf[1], dtype=np.float32)
    ld = np.full(n_total, sf[2], dtype=np.float32)
    h2 = np.full(n_total, sf[3], dtype=np.float32)
    so = 1 + n_free
    b4[so] += np.float32(1.0)
    b8[so + 1] += np.float32(1.0)
    ld[so + 2] += np.float32(1.0)
    h2[so + 3] += np.float32(1.0)
    scores = np.full(n_total, -1.0, dtype=np.float32)
    resid = np.full(n_total, np.inf, dtype=np.float32)
    curtail = np.zeros(n_total, dtype=np.float32)
    _score_mixes_serial(scores, resid, curtail, W_batch, P32, dm32, dn32,
                        b4, b8, ld, h2)
    floor_cfe = float(scores[0])
    floor_resid = float(resid[0])
    floor_curtail = float(curtail[0])
    res_yields = scores[1:1 + n_free].astype(np.float64) - floor_cfe
    stor_yields = scores[1 + n_free:].astype(np.float64) - floor_cfe
    return res_yields, stor_yields, floor_cfe, floor_resid, floor_curtail


# ---------------------------------------------------------------------------
# LHS sample bank
# ---------------------------------------------------------------------------
class LHSBank:
    def __init__(self, n_dims, total_samples, seed=42):
        sampler = LatinHypercube(d=n_dims, seed=seed)
        self.samples = sampler.random(n=total_samples).astype(np.float64)
        self.n_dims = n_dims
        self._offset = 0

    def reset(self):
        self._offset = 0

    def draw(self, n):
        end = self._offset + n
        if end <= len(self.samples):
            batch = self.samples[self._offset:end]
            self._offset = end
            return batch
        sampler = LatinHypercube(d=self.n_dims)
        return sampler.random(n=n).astype(np.float64)


# ---------------------------------------------------------------------------
# LHS candidate generation — with storage minimum headroom
# ---------------------------------------------------------------------------
def generate_candidates(floor_pcts, floor_storage, target_cfe, floor_cfe,
                        res_yields, stor_yields, free_res_idx, cfg,
                        n_samples, unit_samples,
                        bounds_override=None):
    """Generate LHS candidates. If bounds_override is provided (from bounding
    box), use those instead of marginal-yield-derived bounds."""
    n_free = len(free_res_idx)
    n_dims = n_free + 4

    if bounds_override is not None:
        floors_all, ceilings_all = bounds_override
    else:
        increment = max(0.1, target_cfe - floor_cfe)
        n_active = max(1, sum(1 for y in res_yields if y > 0.01)
                       + sum(1 for y in stor_yields if y > 0.01))
        if target_cfe >= 97.0:
            dim_sf = SAFETY_FACTOR * 3.0
        else:
            dim_sf = SAFETY_FACTOR * 2.0 / n_active

        res_ceilings = np.zeros(n_free, dtype=np.float64)
        for k in range(n_free):
            yld = res_yields[k]
            if yld > 0.001:
                res_ceilings[k] = floor_pcts[free_res_idx[k]] + increment / yld * dim_sf
            else:
                res_ceilings[k] = floor_pcts[free_res_idx[k]] + 1.0

        # Storage ceilings: max of yield-based and minimum headroom
        stor_ceilings = np.zeros(4, dtype=np.float64)
        for j in range(4):
            yld = stor_yields[j]
            if yld > 0.001:
                yield_based = floor_storage[j] + increment / yld * dim_sf
            else:
                yield_based = floor_storage[j] + 1.0
            headroom_based = floor_storage[j] + STORAGE_MIN_HEADROOM_ARR[j]
            stor_ceilings[j] = max(yield_based, headroom_based)

        if cfg.pathway == "B":
            demand_twh = float(pc.REGIONAL_DEMAND_TWH[cfg.iso])
            for k, ri in enumerate(free_res_idx):
                res = RESOURCE_ORDER[ri]
                if res == "ccs_ccgt":
                    cap_pct = float(pc.CCS_CAP_TWH.get(cfg.iso, 9999)) / demand_twh * 100
                    res_ceilings[k] = min(res_ceilings[k], cap_pct)
                elif res == "geothermal":
                    cap_pct = float(pc.GEOTHERMAL_CAP_TWH) / demand_twh * 100
                    res_ceilings[k] = min(res_ceilings[k], cap_pct)

        floors_all = np.zeros(n_dims)
        ceilings_all = np.zeros(n_dims)
        for k in range(n_free):
            floors_all[k] = floor_pcts[free_res_idx[k]]
            ceilings_all[k] = res_ceilings[k]
        for j in range(4):
            floors_all[n_free + j] = floor_storage[j]
            ceilings_all[n_free + j] = stor_ceilings[j]

    ceilings_all = np.maximum(ceilings_all, floors_all + 0.01)
    raw = floors_all[None, :] + unit_samples[:n_samples] * (ceilings_all - floors_all)[None, :]

    W = np.zeros((n_samples, N_RESOURCES), dtype=np.float32)
    for ri in range(N_RESOURCES):
        W[:, ri] = np.float32(floor_pcts[ri] * 0.01)
    for k, ri in enumerate(free_res_idx):
        W[:, ri] = (raw[:, k] * 0.01).astype(np.float32)

    batt4 = raw[:, n_free + 0].astype(np.float32)
    batt8 = raw[:, n_free + 1].astype(np.float32)
    ldes_arr = raw[:, n_free + 2].astype(np.float32)
    h2_arr = raw[:, n_free + 3].astype(np.float32)

    return W, batt4, batt8, ldes_arr, h2_arr, raw


# ---------------------------------------------------------------------------
# Stage 1 — Archetype selection from EF parquets at 2030 waypoint
# ---------------------------------------------------------------------------
def load_ef_band(iso, threshold):
    ttag = str(int(threshold)) if threshold == int(threshold) else str(threshold)
    pattern = f"step_2_1_EF_{iso}_{ttag}"
    paths = sorted(p for p in EF_DIR.iterdir()
                   if p.name.startswith(pattern)
                   and "peakclean" not in p.name
                   and "interp" not in p.name
                   and p.suffix == ".parquet")
    if not paths:
        return None
    tbls = [pq.read_table(p) for p in paths]
    if len(tbls) > 1:
        return pa.concat_tables(tbls, promote_options="default")
    return tbls[0]


def classify_archetype(mix_pcts, pathway):
    solar_family = sum(mix_pcts.get(r, 0) for r in
                       ["solar", "solar_batt4", "solar_batt8"])
    wind_family = sum(mix_pcts.get(r, 0) for r in
                      ["wind", "wind_batt4", "wind_batt8"])
    total_clean = sum(mix_pcts.get(r, 0) for r in RESOURCE_ORDER)
    if total_clean < 1e-6:
        return "balanced"
    sf = solar_family / total_clean
    wf = wind_family / total_clean
    if pathway == "B":
        nuke = mix_pcts.get("clean_firm", 0) / total_clean
        off = mix_pcts.get("offshore_wind", 0) / total_clean
        if nuke > 0.4:
            return "nuclear-heavy"
        if off > 0.3:
            return "offshore-heavy"
    if sf > 0.5:
        return "solar-led"
    if wf > 0.5:
        return "wind-led"
    return "balanced"


def stage1_select(iso, cfg, P32, dm32, dn32):
    waypoint = get_waypoint_2030(iso)
    print(f"[stage1] {iso}: baseline={sum(pc.GRID_MIX_SHARES[iso].values()):.1f}%, "
          f"2030 waypoint={waypoint:.0f}%")

    ef = load_ef_band(iso, waypoint)
    if ef is None:
        print(f"[stage1] {iso}: no EF parquet for waypoint {waypoint}")
        return []

    n = ef.num_rows
    print(f"[stage1] {iso}: loaded {n:,} mixes (band={waypoint}%)")

    W = np.zeros((n, N_RESOURCES), dtype=np.float32)
    tbl_names = set(ef.schema.names)
    for i, res in enumerate(RESOURCE_ORDER):
        if res in tbl_names:
            W[:, i] = ef.column(res).to_numpy().astype(np.float32) * 0.01

    def _col(name):
        if name in tbl_names:
            return ef.column(name).to_numpy().astype(np.float32)
        return np.zeros(n, dtype=np.float32)

    batt4 = _col("battery_dispatch_pct")
    batt8 = _col("battery8_dispatch_pct")
    ldes_arr = _col("ldes_dispatch_pct")
    h2_arr = _col("h2_dispatch_pct")

    scores, resid, curtail = score_candidates(W, P32, dm32, dn32,
                                               batt4, batt8, ldes_arr, h2_arr)
    next_threshold = 100.0
    for t in EF_BAND_THRESHOLDS:
        if t > waypoint:
            next_threshold = t
            break
    mask = (scores >= waypoint - 0.5) & (scores < next_threshold)
    valid_idx = np.where(mask)[0]
    if len(valid_idx) == 0:
        base_pct = sum(pc.GRID_MIX_SHARES[iso].values())
        valid_idx = np.where(scores >= base_pct)[0]
    if len(valid_idx) == 0:
        print(f"[stage1] {iso}: no valid mixes found")
        return []

    print(f"[stage1] {iso}: {len(valid_idx):,} mixes in valid band")

    base_dem = float(pc.REGIONAL_DEMAND_TWH[iso])
    res_lcoes = np.array([get_resource_lcoe(iso, res, SOLVE_START_YEAR, cfg)
                          for res in RESOURCE_ORDER], dtype=np.float64)
    stor_costs = precompute_storage_costs(iso, cfg)
    W_valid_pct = W[valid_idx].astype(np.float64) * 100.0
    costs = np.sum(W_valid_pct / 100.0 * base_dem * res_lcoes * 1e6, axis=1)
    stor_vals = np.column_stack([batt4[valid_idx], batt8[valid_idx],
                                  ldes_arr[valid_idx], h2_arr[valid_idx]]
                                 ).astype(np.float64)
    stor_mw = stor_vals / 100.0 * base_dem * 1e6 / 8760
    costs += np.sum(stor_mw * stor_costs, axis=1)

    archetypes = {}
    for ki, vi in enumerate(valid_idx):
        mix = {res: float(W[vi, ri]) * 100.0 for ri, res in enumerate(RESOURCE_ORDER)}
        arch = classify_archetype(mix, cfg.pathway)
        if arch not in archetypes or costs[ki] < archetypes[arch][1]:
            archetypes[arch] = (vi, costs[ki], ki)

    seeds = []
    for arch in sorted(archetypes, key=lambda a: archetypes[a][1]):
        vi, cost, ki = archetypes[arch]
        seeds.append({
            "archetype": arch,
            "resource_pcts": {res: float(W[vi, ri]) * 100.0
                              for ri, res in enumerate(RESOURCE_ORDER)},
            "storage_pcts": {STORAGE_COLS[j]: float([batt4, batt8, ldes_arr, h2_arr][j][vi])
                             for j in range(4)},
            "cfe": float(scores[vi]), "resid": float(resid[vi]), "cost": cost,
        })
        if len(seeds) >= cfg.scaled_beams:
            break

    if len(seeds) < cfg.scaled_beams:
        rank = np.argsort(costs)
        for ki in rank:
            if len(seeds) >= cfg.scaled_beams:
                break
            vi = valid_idx[ki]
            mix = {res: float(W[vi, ri]) * 100.0 for ri, res in enumerate(RESOURCE_ORDER)}
            arch = classify_archetype(mix, cfg.pathway)
            seeds.append({
                "archetype": arch, "resource_pcts": mix,
                "storage_pcts": {STORAGE_COLS[j]: float([batt4, batt8, ldes_arr, h2_arr][j][vi])
                                 for j in range(4)},
                "cfe": float(scores[vi]), "resid": float(resid[vi]), "cost": costs[ki],
            })

    print(f"[stage1] {iso}: selected {len(seeds)} seeds: "
          f"{[s['archetype'] for s in seeds]}")
    return seeds


# ---------------------------------------------------------------------------
# Pre-period cost (2025-2029)
# ---------------------------------------------------------------------------
def compute_preperiod(seed, cfg, P32, dm32, dn32):
    iso = cfg.iso
    demand_vec_full = demand_twh_vec(iso, cfg.demand_growth)
    dem_2030 = demand_vec_full[SOLVE_YEAR_OFFSET]
    existing_gas = float(pc.EXISTING_GAS_CAPACITY_MW[iso])
    gaf = float(pc.GAS_AVAILABILITY_FACTOR[iso])
    lcoes_2030 = np.array([get_resource_lcoe(iso, res, SOLVE_START_YEAR, cfg)
                           for res in RESOURCE_ORDER], dtype=np.float64)
    stor_costs = precompute_storage_costs(iso, cfg)

    baseline_pcts = np.zeros(N_RESOURCES, dtype=np.float64)
    seed_pcts = np.zeros(N_RESOURCES, dtype=np.float64)
    for ri, res in enumerate(RESOURCE_ORDER):
        baseline_pcts[ri] = pc.GRID_MIX_SHARES[iso].get(res, 0.0)
        seed_pcts[ri] = seed["resource_pcts"].get(res, 0.0)

    seed_cfe, seed_resid, seed_curtail = _score_single(
        seed_pcts,
        np.array([seed["storage_pcts"].get(sc, 0.0) for sc in STORAGE_COLS],
                 dtype=np.float64),
        P32, dm32, dn32)

    vintage_ledger = []
    total_cost = 0.0
    for ri, res in enumerate(RESOURCE_ORDER):
        delta_pct = seed_pcts[ri] - baseline_pcts[ri]
        if delta_pct > RATCHET_TOL_PCT:
            delta_twh = delta_pct / 100.0 * dem_2030
            if res == "clean_firm" and cfg.pathway == "B":
                prev_t = decompose_clean_firm(baseline_pcts[ri]/100*dem_2030, iso, SOLVE_START_YEAR, cfg)
                new_t = decompose_clean_firm(seed_pcts[ri]/100*dem_2030, iso, SOLVE_START_YEAR, cfg)
                for tr in ["uprate","geo","nuke_new","ccs"]:
                    td = new_t[f"{tr}_twh"] - prev_t[f"{tr}_twh"]
                    if td > 0.01:
                        tl = new_t[f"{tr}_lcoe"]
                        vintage_ledger.append((SOLVE_START_YEAR, f"clean_firm_{tr}", td, tl))
                        total_cost += td * tl * 1e6
            else:
                vintage_ledger.append((SOLVE_START_YEAR, res, delta_twh, lcoes_2030[ri]))
                total_cost += delta_twh * lcoes_2030[ri] * 1e6

    for j, sc in enumerate(STORAGE_COLS):
        sp = seed["storage_pcts"].get(sc, 0.0)
        if sp > RATCHET_TOL_PCT:
            cap_mw = sp / 100.0 * dem_2030 * 1e6 / 8760
            vintage_ledger.append((SOLVE_START_YEAR, sc, cap_mw, stor_costs[j]))
            total_cost += cap_mw * stor_costs[j]

    g_mw = gas_need_mw(seed_resid, dem_2030, existing_gas, gaf)
    g_cost = gas_annual_cost(g_mw, iso, cfg)

    return {
        "preperiod_year_range": [BASE_YEAR, SOLVE_START_YEAR - 1],
        "costed_at_year": SOLVE_START_YEAR,
        "demand_twh_2030": round(dem_2030, 1),
        "seed_cfe_pct": round(seed_cfe, 2),
        "seed_resid": round(seed_resid, 6),
        "gas_need_mw": round(g_mw, 1),
        "preperiod_clean_cost_usd": round(total_cost, 0),
        "preperiod_gas_cost_usd": round(g_cost, 0),
        "preperiod_total_cost_usd": round(total_cost + g_cost, 0),
        "vintage_ledger": vintage_ledger,
    }


# ---------------------------------------------------------------------------
# Stage 2 — Two-stage adaptive LHS _find_winner
# ---------------------------------------------------------------------------
def _find_winner(floor_pcts, floor_storage, target, ceiling,
                 floor_cfe, res_yields, stor_yields,
                 free_idx, cfg, P32, dm32, dn32,
                 year, dem_twh, lhs_bank, res_lcoes_free, stor_costs):
    """Two-stage adaptive search for cheapest feasible candidate.

    Round 1:  Coarse screen at marginal-yield bounds (screen_samples).
              If <min_hits feasible, widen to 2× bounds (widen_samples).
    Round 2:  Tight bounding box from Round 1 hits + margin.
              Dense LHS within that box (refine_samples).
    Fallback: Storage-focused 4D sampling at ≥97% CFE.
    """
    free_idx_arr = np.array(free_idx, dtype=np.intp)
    floor_free = floor_pcts[free_idx_arr]
    n_free = len(free_idx)

    # Gas shadow cost setup (now applies to both pathways if toggled on)
    use_gas_shadow = cfg.use_gas_shadow
    if use_gas_shadow:
        existing_gas = float(pc.EXISTING_GAS_CAPACITY_MW[cfg.iso])
        gaf = float(pc.GAS_AVAILABILITY_FACTOR[cfg.iso])
        gas_useful_life = 30.0
        years_remaining = max(1.0, float(END_YEAR - year))
        stranding_frac = max(0.0, 1.0 - years_remaining / gas_useful_life)
        gas_capex_per_mw = CCGT_OVERNIGHT_CAPEX_USD_KW * 1000.0

    def _cost_valid(valid, all_W, all_b4, all_b8, all_ld, all_h2, all_resid):
        """Compute incremental cost for valid candidates."""
        W_valid_pct = all_W[valid][:, free_idx_arr].astype(np.float64) * 100.0
        res_deltas = np.maximum(W_valid_pct - floor_free, 0.0)
        costs = np.sum(res_deltas * (dem_twh / 100.0 * res_lcoes_free * 1e6), axis=1)
        sv = np.column_stack([all_b4[valid], all_b8[valid],
                               all_ld[valid], all_h2[valid]]).astype(np.float64)
        sd = np.maximum(sv - floor_storage, 0.0)
        sm = sd * (dem_twh * 1e6 / 8760 / 100.0)
        costs += np.sum(sm * stor_costs, axis=1)
        if use_gas_shadow and stranding_frac > 0:
            rv = all_resid[valid].astype(np.float64)
            gm = np.maximum(rv * dem_twh * 1e6 / gaf - existing_gas, 0.0)
            costs += gm * gas_capex_per_mw * stranding_frac
        return costs

    def _extract_winner(valid, costs, all_W, all_b4, all_b8, all_ld, all_h2,
                        all_scores, all_resid, all_curtail):
        best_local = np.argmin(costs)
        best = valid[best_local]
        wp = all_W[best].astype(np.float64) * 100.0
        fm = np.ones(N_RESOURCES, dtype=bool)
        fm[free_idx_arr] = False
        wp[fm] = floor_pcts[fm]
        ws = np.array([all_b4[best], all_b8[best],
                        all_ld[best], all_h2[best]], dtype=np.float64)
        return (wp, ws, float(all_scores[best]),
                float(all_resid[best]), float(all_curtail[best]))

    # ── Round 1: Coarse screen ──
    lhs_bank.reset()
    unit1 = lhs_bank.draw(cfg.screen_samples)
    W1, b4_1, b8_1, ld_1, h2_1, raw1 = generate_candidates(
        floor_pcts, floor_storage, target, floor_cfe,
        res_yields, stor_yields, free_idx, cfg, cfg.screen_samples, unit1)
    sc1, re1, cu1 = score_candidates(W1, P32, dm32, dn32, b4_1, b8_1, ld_1, h2_1)

    mask1 = (sc1 >= target) & (sc1 < ceiling)
    valid1 = np.where(mask1)[0]
    n_hits_r1 = len(valid1)
    print(f"    R1 screen: {cfg.screen_samples} samples → {n_hits_r1} hits "
          f"in [{target:.1f}, {ceiling:.1f})")

    # If too few hits, widen bounds and resample
    if n_hits_r1 < cfg.min_hits:
        # Widen: 2× the marginal-yield bounds
        n_free = len(free_idx)
        n_dims = n_free + 4
        increment = max(0.1, target - floor_cfe)
        n_active = max(1, sum(1 for y in res_yields if y > 0.01)
                       + sum(1 for y in stor_yields if y > 0.01))
        if target >= 97.0:
            dim_sf = SAFETY_FACTOR * 3.0 * 2.0  # 2× widen
        else:
            dim_sf = SAFETY_FACTOR * 2.0 / n_active * 2.0  # 2× widen

        res_ceil_w = np.zeros(n_free, dtype=np.float64)
        for k in range(n_free):
            yld = res_yields[k]
            if yld > 0.001:
                res_ceil_w[k] = floor_pcts[free_idx[k]] + increment / yld * dim_sf
            else:
                res_ceil_w[k] = floor_pcts[free_idx[k]] + 2.0
        stor_ceil_w = np.zeros(4, dtype=np.float64)
        for j in range(4):
            yld = stor_yields[j]
            if yld > 0.001:
                yield_based = floor_storage[j] + increment / yld * dim_sf
            else:
                yield_based = floor_storage[j] + 2.0
            stor_ceil_w[j] = max(yield_based, floor_storage[j] + STORAGE_MIN_HEADROOM_ARR[j] * 1.5)

        floors_w = np.zeros(n_dims)
        ceilings_w = np.zeros(n_dims)
        for k in range(n_free):
            floors_w[k] = floor_pcts[free_idx[k]]
            ceilings_w[k] = res_ceil_w[k]
        for j in range(4):
            floors_w[n_free + j] = floor_storage[j]
            ceilings_w[n_free + j] = stor_ceil_w[j]

        unit_w = lhs_bank.draw(cfg.widen_samples)
        Ww, b4w, b8w, ldw, h2w, raww = generate_candidates(
            floor_pcts, floor_storage, target, floor_cfe,
            res_yields, stor_yields, free_idx, cfg,
            cfg.widen_samples, unit_w,
            bounds_override=(floors_w, ceilings_w))
        scw, rew, cuw = score_candidates(Ww, P32, dm32, dn32, b4w, b8w, ldw, h2w)

        # Combine with Round 1
        sc1 = np.concatenate([sc1, scw])
        re1 = np.concatenate([re1, rew])
        cu1 = np.concatenate([cu1, cuw])
        W1 = np.concatenate([W1, Ww])
        b4_1 = np.concatenate([b4_1, b4w])
        b8_1 = np.concatenate([b8_1, b8w])
        ld_1 = np.concatenate([ld_1, ldw])
        h2_1 = np.concatenate([h2_1, h2w])
        raw1 = np.concatenate([raw1, raww])

        mask1 = (sc1 >= target) & (sc1 < ceiling)
        valid1 = np.where(mask1)[0]
        print(f"    R1 widen:  {cfg.widen_samples} samples → "
              f"{len(valid1)} total hits (was {n_hits_r1})")
        n_hits_r1 = len(valid1)

    # If still no hits after screen + widen
    if n_hits_r1 == 0:
        # Try storage-focused fallback at ≥97%
        if target >= 97.0:
            return _storage_fallback(
                floor_pcts, floor_storage, target, ceiling,
                free_idx, cfg, P32, dm32, dn32,
                dem_twh, floor_free, res_lcoes_free, stor_costs,
                use_gas_shadow, locals())
        return None, None, None, None, None

    # ── Round 2: Tight bounding box refinement ──
    # Compute per-dimension bounding box from feasible hits
    raw_valid = raw1[valid1]  # (n_hits, n_dims)
    bb_min = raw_valid.min(axis=0)
    bb_max = raw_valid.max(axis=0)
    bb_range = bb_max - bb_min
    margin = cfg.bbox_margin
    bb_floors = bb_min - bb_range * margin
    bb_ceilings = bb_max + bb_range * margin

    # Clamp: don't go below original floor
    n_free = len(free_idx)
    for k in range(n_free):
        bb_floors[k] = max(bb_floors[k], floor_pcts[free_idx[k]])
    for j in range(4):
        bb_floors[n_free + j] = max(bb_floors[n_free + j], floor_storage[j])

    print(f"    R2 refine: {cfg.refine_samples} samples in bbox "
          f"(margin={margin:.0%})")

    unit2 = lhs_bank.draw(cfg.refine_samples)
    W2, b4_2, b8_2, ld_2, h2_2, raw2 = generate_candidates(
        floor_pcts, floor_storage, target, floor_cfe,
        res_yields, stor_yields, free_idx, cfg,
        cfg.refine_samples, unit2,
        bounds_override=(bb_floors, bb_ceilings))
    sc2, re2, cu2 = score_candidates(W2, P32, dm32, dn32, b4_2, b8_2, ld_2, h2_2)

    # Combine R1 + R2 hits
    all_sc = np.concatenate([sc1, sc2])
    all_re = np.concatenate([re1, re2])
    all_cu = np.concatenate([cu1, cu2])
    all_W = np.concatenate([W1, W2])
    all_b4 = np.concatenate([b4_1, b4_2])
    all_b8 = np.concatenate([b8_1, b8_2])
    all_ld = np.concatenate([ld_1, ld_2])
    all_h2 = np.concatenate([h2_1, h2_2])

    mask_all = (all_sc >= target) & (all_sc < ceiling)
    valid_all = np.where(mask_all)[0]

    r2_mask = (sc2 >= target) & (sc2 < ceiling)
    print(f"    R2 hits: {int(r2_mask.sum())}, combined: {len(valid_all)}")

    if len(valid_all) == 0:
        if target >= 97.0:
            return _storage_fallback(
                floor_pcts, floor_storage, target, ceiling,
                free_idx, cfg, P32, dm32, dn32,
                dem_twh, floor_free, res_lcoes_free, stor_costs,
                use_gas_shadow, locals())
        return None, None, None, None, None

    costs = _cost_valid(valid_all, all_W, all_b4, all_b8, all_ld, all_h2, all_re)
    return _extract_winner(valid_all, costs, all_W, all_b4, all_b8, all_ld, all_h2,
                           all_sc, all_re, all_cu)


def _storage_fallback(floor_pcts, floor_storage, target, ceiling,
                      free_idx, cfg, P32, dm32, dn32,
                      dem_twh, floor_free, res_lcoes_free, stor_costs,
                      use_gas_shadow, ctx):
    """Storage-focused 4D fallback for last mile (≥97% CFE)."""
    free_idx_arr = np.array(free_idx, dtype=np.intp)
    n_stor = cfg.refine_samples
    print(f"    storage fallback: {n_stor} samples, 4D")
    sampler = LatinHypercube(d=4, seed=int(time.time() * 1000) % (2**31))
    stor_unit = sampler.random(n=n_stor)

    stor_floors = floor_storage.copy()
    stor_ceil = np.maximum(floor_storage * 5.0, floor_storage + 30.0)
    stor_ceil = np.minimum(stor_ceil, 50.0)
    stor_ceil = np.maximum(stor_ceil, stor_floors + 1.0)
    stor_raw = stor_floors + stor_unit * (stor_ceil - stor_floors)

    rng = np.random.default_rng(seed=42)
    W_sf = np.zeros((n_stor, N_RESOURCES), dtype=np.float32)
    for ri in range(N_RESOURCES):
        W_sf[:, ri] = np.float32(floor_pcts[ri] * 0.01)
    for ri in free_idx:
        noise = rng.uniform(0, 5.0, size=n_stor)
        W_sf[:, ri] = ((floor_pcts[ri] + noise) * 0.01).astype(np.float32)

    sf_b4 = stor_raw[:, 0].astype(np.float32)
    sf_b8 = stor_raw[:, 1].astype(np.float32)
    sf_ld = stor_raw[:, 2].astype(np.float32)
    sf_h2 = stor_raw[:, 3].astype(np.float32)

    sf_sc, sf_re, sf_cu = score_candidates(
        W_sf, P32, dm32, dn32, sf_b4, sf_b8, sf_ld, sf_h2)

    mask = (sf_sc >= target) & (sf_sc < ceiling)
    valid = np.where(mask)[0]

    if len(valid) > 0:
        print(f"    storage fallback: {len(valid)} hits")
        Wv = W_sf[valid][:, free_idx_arr].astype(np.float64) * 100.0
        rd = np.maximum(Wv - floor_free, 0.0)
        costs = np.sum(rd * (dem_twh / 100.0 * res_lcoes_free * 1e6), axis=1)
        sv = np.column_stack([sf_b4[valid], sf_b8[valid],
                               sf_ld[valid], sf_h2[valid]]).astype(np.float64)
        sd = np.maximum(sv - floor_storage, 0.0)
        sm = sd * (dem_twh * 1e6 / 8760 / 100.0)
        costs += np.sum(sm * stor_costs, axis=1)

        if use_gas_shadow:
            existing_gas = float(pc.EXISTING_GAS_CAPACITY_MW[cfg.iso])
            gaf = float(pc.GAS_AVAILABILITY_FACTOR[cfg.iso])
            yrs_rem = max(1.0, float(END_YEAR - ctx.get('year', 2045)))
            sf_frac = max(0.0, 1.0 - yrs_rem / 30.0)
            if sf_frac > 0:
                rv = sf_re[valid].astype(np.float64)
                gm = np.maximum(rv * dem_twh * 1e6 / gaf - existing_gas, 0.0)
                costs += gm * CCGT_OVERNIGHT_CAPEX_USD_KW * 1000.0 * sf_frac

        bl = np.argmin(costs)
        best = valid[bl]
        wp = W_sf[best].astype(np.float64) * 100.0
        fm = np.ones(N_RESOURCES, dtype=bool)
        fm[free_idx_arr] = False
        wp[fm] = floor_pcts[fm]
        ws = np.array([sf_b4[best], sf_b8[best],
                        sf_ld[best], sf_h2[best]], dtype=np.float64)
        return (wp, ws, float(sf_sc[best]), float(sf_re[best]), float(sf_cu[best]))
    else:
        print(f"    storage fallback: 0 hits — infeasible")
    return None, None, None, None, None


def _score_single(pcts, storage, P32, dm32, dn32):
    W = np.zeros((1, N_RESOURCES), dtype=np.float32)
    W[0] = (pcts * 0.01).astype(np.float32)
    s = storage.astype(np.float32)
    scores, resid, curtail = score_candidates(
        W, P32, dm32, dn32, s[0:1], s[1:2], s[2:3], s[3:4])
    return float(scores[0]), float(resid[0]), float(curtail[0])


# ---------------------------------------------------------------------------
# Stage 2 — Year-by-year solve (2030-2050)
# ---------------------------------------------------------------------------
def solve_pathway(seed, cfg, P32, dm32, dn32):
    iso = cfg.iso
    demand_vec = demand_twh_vec(iso, cfg.demand_growth)
    peak_vec = peak_demand_vec(iso, cfg.demand_growth)
    cfe_targets = cfe_target_vec(iso)
    existing_gas = float(pc.EXISTING_GAS_CAPACITY_MW[iso])
    gaf = float(pc.GAS_AVAILABILITY_FACTOR[iso])
    free_idx = cfg.free_indices

    lcoe_matrix = precompute_lcoe_matrix(iso, cfg)
    stor_costs = precompute_storage_costs(iso, cfg)

    # LHS bank sized for two-stage: (screen + widen + refine) × years × margin
    bank_total = (cfg.screen_samples + cfg.widen_samples + cfg.refine_samples) * 2
    lhs_bank = LHSBank(cfg.n_dims, bank_total, seed=42)

    base_demand = float(pc.REGIONAL_DEMAND_TWH[iso])
    frozen_twh = {}
    frozen_twh["hydro"] = pc.GRID_MIX_SHARES[iso].get("hydro", 0) / 100.0 * base_demand
    if cfg.pathway == "A":
        frozen_twh["clean_firm"] = pc.GRID_MIX_SHARES[iso].get("clean_firm", 0) / 100.0 * base_demand

    baseline_cf_twh = pc.GRID_MIX_SHARES[iso].get("clean_firm", 0) / 100.0 * base_demand

    floor_pcts = np.zeros(N_RESOURCES, dtype=np.float64)
    for ri, res in enumerate(RESOURCE_ORDER):
        floor_pcts[ri] = seed["resource_pcts"].get(res, 0.0)

    cf_baseline_pct = pc.GRID_MIX_SHARES[iso].get("clean_firm", 0)
    hydro_baseline_pct = pc.GRID_MIX_SHARES[iso].get("hydro", 0)
    floor_pcts[RES_IDX["clean_firm"]] = max(floor_pcts[RES_IDX["clean_firm"]], cf_baseline_pct)
    floor_pcts[RES_IDX["hydro"]] = max(floor_pcts[RES_IDX["hydro"]], hydro_baseline_pct)

    floor_storage = np.array([seed["storage_pcts"].get(sc, 0.0) for sc in STORAGE_COLS],
                             dtype=np.float64)

    preperiod = compute_preperiod(seed, cfg, P32, dm32, dn32)

    vintage_ledger = list(preperiod["vintage_ledger"])
    cumulative_cost = preperiod["preperiod_total_cost_usd"]
    threshold_snapshots = []
    thresholds_crossed = set()
    peak_gas_mw = preperiod["gas_need_mw"]
    peak_gas_year = SOLVE_START_YEAR

    # Pre-period threshold snapshots
    seed_cfe = preperiod["seed_cfe_pct"]
    dem_2030 = demand_vec[SOLVE_YEAR_OFFSET]
    for t in EF_BAND_THRESHOLDS:
        if t <= seed_cfe and t not in thresholds_crossed:
            thresholds_crossed.add(t)
            threshold_snapshots.append({
                "threshold_pct": t, "year_achieved": SOLVE_START_YEAR,
                "achieved_cfe_pct": round(seed_cfe, 2),
                "resource_mix_pct": {res: round(float(floor_pcts[ri]), 2)
                                     for ri, res in enumerate(RESOURCE_ORDER)},
                "storage_dispatch_pct": {STORAGE_COLS[j]: round(float(floor_storage[j]), 2)
                                         for j in range(4)},
                "gas_need_mw": round(preperiod["gas_need_mw"], 1),
                "peak_gas_mw": round(peak_gas_mw, 1),
                "stranded_vs_peak_mw": 0.0,
                "total_annual_cost_usd": round(preperiod["preperiod_total_cost_usd"], 0),
                "incremental_cost_usd": round(preperiod["preperiod_clean_cost_usd"], 0),
                "cumulative_cost_usd": round(cumulative_cost, 0),
                "curtailment_total": 0.0,
                "demand_twh": round(dem_2030, 1),
            })

    for yi, year in enumerate(SOLVE_YEARS):
        full_yi = yi + SOLVE_YEAR_OFFSET
        dem_twh = demand_vec[full_yi]
        pk_mw = peak_vec[full_yi]
        target = cfe_targets[yi]

        for res, twh in frozen_twh.items():
            floor_pcts[RES_IDX[res]] = twh / dem_twh * 100.0

        if cfg.pathway == "A":
            for res in ["offshore_wind", "geothermal", "ccs_ccgt"]:
                floor_pcts[RES_IDX[res]] = 0.0
        elif cfg.pathway == "B":
            if iso not in pc.OFFSHORE_ISOS:
                floor_pcts[RES_IDX["offshore_wind"]] = 0.0
            if iso != "CAISO":
                floor_pcts[RES_IDX["geothermal"]] = 0.0
            if float(pc.CCS_CAP_TWH.get(iso, 0)) <= 0:
                floor_pcts[RES_IDX["ccs_ccgt"]] = 0.0
            min_cf_pct = baseline_cf_twh / dem_twh * 100.0
            floor_pcts[RES_IDX["clean_firm"]] = max(
                floor_pcts[RES_IDX["clean_firm"]], min_cf_pct)
            cf_fs, cf_ny = pc.get_pathway_noak_window("nuclear", cfg.firm_cost, "3")
            if year >= cf_fs:
                lf = pc.learning_fraction(year, cf_fs, cf_ny)
                baseline_pct = baseline_cf_twh / dem_twh * 100.0
                expansion_gap = max(0.0, target - baseline_pct)
                cf_commitment = baseline_pct + lf * expansion_gap * CLEAN_FIRM_DEPLOYMENT_SHARE
                floor_pcts[RES_IDX["clean_firm"]] = max(
                    floor_pcts[RES_IDX["clean_firm"]], cf_commitment)

        cfe_ceiling = cfe_targets[yi + 1] if yi < N_SOLVE_YEARS - 1 else 100.01

        res_yields, stor_yields, floor_cfe, floor_resid, floor_curtail = (
            compute_marginal_yields(floor_pcts, floor_storage, free_idx, P32, dm32, dn32))

        if floor_cfe >= target:
            winner_pcts = floor_pcts.copy()
            winner_storage = floor_storage.copy()
            winner_cfe = floor_cfe
            winner_resid = floor_resid
            winner_curtail = floor_curtail
        else:
            lhs_bank.reset()
            res_lcoes_free = lcoe_matrix[yi, cfg.free_indices]
            result = _find_winner(
                floor_pcts, floor_storage, target, cfe_ceiling,
                floor_cfe, res_yields, stor_yields,
                free_idx, cfg, P32, dm32, dn32,
                year, dem_twh, lhs_bank, res_lcoes_free, stor_costs)
            if result[0] is None:
                print(f"[solve] {iso} beam={seed['archetype']}: "
                      f"INFEASIBLE at year {year}, target={target:.1f}%")
                break
            winner_pcts, winner_storage, winner_cfe, winner_resid, winner_curtail = result

        # Vintage cost accounting
        year_incremental_cost = 0.0
        for ri, res in enumerate(RESOURCE_ORDER):
            delta_pct = winner_pcts[ri] - floor_pcts[ri]
            if delta_pct > RATCHET_TOL_PCT:
                delta_twh = delta_pct / 100.0 * dem_twh
                if res == "clean_firm" and cfg.pathway == "B":
                    prev_t = decompose_clean_firm(floor_pcts[ri]/100*dem_twh, iso, year, cfg)
                    new_t = decompose_clean_firm(winner_pcts[ri]/100*dem_twh, iso, year, cfg)
                    for tr in ["uprate","geo","nuke_new","ccs"]:
                        td = new_t[f"{tr}_twh"] - prev_t[f"{tr}_twh"]
                        if td > 0.01:
                            tl = new_t[f"{tr}_lcoe"]
                            vintage_ledger.append((year, f"clean_firm_{tr}", td, tl))
                            year_incremental_cost += td * tl * 1e6
                else:
                    lcoe = lcoe_matrix[yi, ri]
                    vintage_ledger.append((year, res, delta_twh, lcoe))
                    year_incremental_cost += delta_twh * lcoe * 1e6

        for j, sc in enumerate(STORAGE_COLS):
            ds = winner_storage[j] - floor_storage[j]
            if ds > RATCHET_TOL_PCT:
                cap_mw = ds / 100.0 * dem_twh * 1e6 / 8760
                vintage_ledger.append((year, sc, cap_mw, stor_costs[j]))
                year_incremental_cost += cap_mw * stor_costs[j]

        g_mw = gas_need_mw(winner_resid, dem_twh, existing_gas, gaf)
        g_cost = gas_annual_cost(g_mw, iso, cfg)
        if g_mw > peak_gas_mw:
            peak_gas_mw = g_mw
            peak_gas_year = year

        _sn = set(STORAGE_COLS)
        total_vintage_cost = sum(q * uc if r in _sn else q * uc * 1e6
                                  for _, r, q, uc in vintage_ledger)
        total_annual = total_vintage_cost + g_cost
        cumulative_cost += total_annual

        for ri in range(N_RESOURCES):
            ft = floor_pcts[ri] / 100.0 * dem_twh
            wt = winner_pcts[ri] / 100.0 * dem_twh
            nf = max(ft, wt)
            if yi < N_SOLVE_YEARS - 1:
                floor_pcts[ri] = nf / demand_vec[full_yi + 1] * 100.0
            else:
                floor_pcts[ri] = winner_pcts[ri]
        for j in range(4):
            floor_storage[j] = max(floor_storage[j], winner_storage[j])

        for t in EF_BAND_THRESHOLDS:
            if t not in thresholds_crossed and winner_cfe >= t:
                thresholds_crossed.add(t)
                threshold_snapshots.append({
                    "threshold_pct": t, "year_achieved": year,
                    "achieved_cfe_pct": round(float(winner_cfe), 2),
                    "resource_mix_pct": {res: round(float(winner_pcts[ri]), 2)
                                         for ri, res in enumerate(RESOURCE_ORDER)},
                    "storage_dispatch_pct": {STORAGE_COLS[j]: round(float(winner_storage[j]), 2)
                                             for j in range(4)},
                    "gas_need_mw": round(g_mw, 1),
                    "peak_gas_mw": round(peak_gas_mw, 1),
                    "stranded_vs_peak_mw": round(peak_gas_mw - g_mw, 1),
                    "total_annual_cost_usd": round(total_annual, 0),
                    "incremental_cost_usd": round(year_incremental_cost, 0),
                    "cumulative_cost_usd": round(cumulative_cost, 0),
                    "curtailment_total": round(float(winner_curtail), 4),
                    "demand_twh": round(dem_twh, 1),
                })

        if year % 5 == 0 or winner_cfe >= 99.0:
            print(f"  {year}: CFE={winner_cfe:.1f}% target={target:.1f}% "
                  f"gas={g_mw:,.0f}MW cost=${total_annual/1e9:.2f}B")

    return {
        "schema_version": "4.0",
        "iso": iso,
        "pathway": cfg.pathway,
        "dim_key": cfg.dim_key,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "config": {
            "demand_growth": cfg.demand_growth,
            "firm_cost": cfg.firm_cost,
            "ccs_cost": cfg.ccs_cost,
            "tx_level": cfg.tx_level,
            "n_beams": cfg.n_beams,
            "n_beams_scaled": cfg.scaled_beams,
            "screen_samples": cfg.screen_samples,
            "widen_samples": cfg.widen_samples,
            "refine_samples": cfg.refine_samples,
            "min_hits": cfg.min_hits,
            "bbox_margin": cfg.bbox_margin,
            "safety_factor": SAFETY_FACTOR,
            "wrights_law_k": WRIGHTS_LAW_K,
            "gas_shadow": cfg.use_gas_shadow,
            "waypoint_2030_pct": get_waypoint_2030(iso),
            "solve_start_year": SOLVE_START_YEAR,
        },
        "beam_archetype": seed["archetype"],
        "peak_gas_mw": round(peak_gas_mw, 1),
        "peak_gas_year": peak_gas_year,
        "preperiod": {
            "year_range": preperiod["preperiod_year_range"],
            "costed_at_year": preperiod["costed_at_year"],
            "total_cost_usd": preperiod["preperiod_total_cost_usd"],
            "clean_cost_usd": preperiod["preperiod_clean_cost_usd"],
            "gas_cost_usd": preperiod["preperiod_gas_cost_usd"],
        },
        "n_thresholds_achieved": len(threshold_snapshots),
        "threshold_snapshots": threshold_snapshots,
        "vintage_ledger_summary": {
            "n_vintages": len(vintage_ledger),
            "total_gen_twh_built": round(sum(
                t for _, r, t, _ in vintage_ledger if r not in STORAGE_COLS), 2),
            "total_storage_mw_built": round(sum(
                t for _, r, t, _ in vintage_ledger if r in STORAGE_COLS), 1),
        },
    }


# ---------------------------------------------------------------------------
# JIT warm-up
# ---------------------------------------------------------------------------
def warmup_jit():
    t0 = time.time()
    n, nr = 2, N_RESOURCES
    W = np.zeros((n, nr), dtype=np.float32)
    P = np.zeros((nr, H), dtype=np.float32)
    dm = np.ones(H, dtype=np.float32) / H
    dn = dm.copy()
    batt = np.array([0.0, 10.0], dtype=np.float32)
    zero = np.zeros(n, dtype=np.float32)
    s = np.full(n, -1.0, dtype=np.float32)
    r = np.full(n, np.inf, dtype=np.float32)
    c = np.zeros(n, dtype=np.float32)
    _score_mixes_serial(s, r, c, W, P, dm, dn, batt, zero, zero, zero)
    s2 = np.full(n, -1.0, dtype=np.float32)
    r2 = np.full(n, np.inf, dtype=np.float32)
    c2 = np.zeros(n, dtype=np.float32)
    idx = np.arange(n, dtype=np.int64)
    _score_mixes_parallel(s2, r2, c2, W, P, dm, dn, batt, zero, zero, zero, idx)
    print(f"[adaptive] JIT warm-up: {time.time()-t0:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_result(result):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fname = (f"{result['iso']}__pathway{result['pathway']}__"
             f"{result.get('scenario_name', 'custom')}__"
             f"{result['beam_archetype']}__"
             f"dg_{result['config']['demand_growth']}__"
             f"{result['timestamp']}.json")
    path = OUTPUT_DIR / fname
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[output] wrote {path.name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_configs(iso, pathway, scenarios, demand_levels,
                  n_beams, screen_samples, widen_samples,
                  refine_samples, min_hits, bbox_margin,
                  gas_shadow):
    configs = []
    for scen_name in scenarios:
        scen = COST_SCENARIOS[scen_name]
        for dg in demand_levels:
            configs.append(RunConfig(
                iso=iso, pathway=pathway,
                scenario_name=scen_name, demand_growth=dg,
                ren_cost=scen["ren_cost"], firm_cost=scen["firm_cost"],
                batt_cost=scen["batt_cost"], ldes_cost=scen["ldes_cost"],
                fuel_cost=scen["fuel_cost"], tx_level=scen["tx_level"],
                ccs_cost=scen["ccs_cost"], geo_cost=scen["geo_cost"],
                q45=scen["q45"], n_beams=n_beams,
                screen_samples=screen_samples,
                widen_samples=widen_samples,
                refine_samples=refine_samples,
                min_hits=min_hits, bbox_margin=bbox_margin,
                gas_shadow=gas_shadow,
            ))
    return configs


def main():
    ap = argparse.ArgumentParser(
        description="Step 2.3 Adaptive — two-stage LHS reliability tax optimizer (v4.0)")
    ap.add_argument("--iso", type=str, required=True)
    ap.add_argument("--pathway", type=str, required=True, choices=["A", "B"])
    ap.add_argument("--demand-growth", type=str, default="Medium",
                    choices=["Low", "Medium", "High"])
    ap.add_argument("--scenario", type=str, default="base",
                    choices=list(COST_SCENARIOS.keys()))
    ap.add_argument("--batch", action="store_true",
                    help="Run all 3 cost scenarios")
    ap.add_argument("--batch-demand", action="store_true",
                    help="Cross all 3 demand growth levels")
    ap.add_argument("--n-beams", type=int, default=5)
    # Two-stage LHS controls
    ap.add_argument("--screen-samples", type=int, default=5000,
                    help="Round 1 coarse screen sample count")
    ap.add_argument("--widen-samples", type=int, default=10000,
                    help="Round 1b widened resample if <min-hits")
    ap.add_argument("--refine-samples", type=int, default=10000,
                    help="Round 2 bounding box refinement sample count")
    ap.add_argument("--min-hits", type=int, default=50,
                    help="Minimum R1 hits before bounding box (else widen)")
    ap.add_argument("--bbox-margin", type=float, default=0.15,
                    help="Bounding box margin fraction (0.15 = 15%%)")
    # Gas shadow cost
    ap.add_argument("--gas-shadow", action="store_true",
                    help="Enable gas stranding shadow cost for Pathway A "
                         "(always on for B)")
    args = ap.parse_args()

    iso = args.iso.upper()
    if iso not in pc.ISOS:
        raise SystemExit(f"Unknown ISO: {iso}. Known: {list(pc.ISOS)}")

    scenarios = list(COST_SCENARIOS.keys()) if args.batch else [args.scenario]
    demand_levels = ["Low", "Medium", "High"] if args.batch_demand else [args.demand_growth]

    wp = get_waypoint_2030(iso)
    gas_mode = "ON" if (args.pathway == "B" or args.gas_shadow) else "OFF"
    print(f"\n{'='*70}")
    print(f"[adaptive] Step 2.3 Reliability Tax Adaptive (v4.0)")
    print(f"[adaptive] ISO={iso}  Pathway={args.pathway}  "
          f"2030 waypoint={wp:.0f}%  Gas shadow={gas_mode}")
    print(f"[adaptive] Waypoints: {get_cfe_waypoints(iso)}")
    print(f"[adaptive] LHS: screen={args.screen_samples} widen={args.widen_samples} "
          f"refine={args.refine_samples} min_hits={args.min_hits} "
          f"bbox_margin={args.bbox_margin:.0%}")
    print(f"[adaptive] Scenarios: {scenarios}  Demand: {demand_levels}")
    print(f"[adaptive] Solve range: {SOLVE_START_YEAR}-{END_YEAR} "
          f"({N_SOLVE_YEARS} years) + pre-period at 2030 LCOEs")
    print(f"{'='*70}\n")

    warmup_jit()

    print(f"[adaptive] Loading profiles for {iso}...")
    P = build_profile_matrix(iso)
    dn = _load_demand_norm(iso)
    dm = dn * (1.0 + RA_MARGIN)
    P32 = P.astype(np.float32)
    dm32 = dm.astype(np.float32)
    dn32 = dn.astype(np.float32)
    print(f"[adaptive] Profiles loaded. Demand TWh={float(pc.REGIONAL_DEMAND_TWH[iso]):.1f}")

    configs = build_configs(
        iso, args.pathway, scenarios, demand_levels,
        args.n_beams, args.screen_samples, args.widen_samples,
        args.refine_samples, args.min_hits, args.bbox_margin,
        args.gas_shadow)
    print(f"[adaptive] {len(configs)} runs queued")
    sc = configs[0]
    print(f"[adaptive] Dims: {sc.n_dims} | Beams: {sc.n_beams}→{sc.scaled_beams}")

    seeds = stage1_select(iso, configs[0], P32, dm32, dn32)
    if not seeds:
        print(f"[adaptive] No seeds found. Exiting.")
        return 1

    t_total = time.time()
    for ci, cfg in enumerate(configs):
        print(f"\n{'─'*60}")
        print(f"[adaptive] Config {ci+1}/{len(configs)}: "
              f"{cfg.scenario_name} | {cfg.demand_growth}")
        print(f"{'─'*60}")

        for bi, seed in enumerate(seeds):
            print(f"\n[solve] Beam {bi+1}/{len(seeds)}: {seed['archetype']} "
                  f"(seed CFE={seed['cfe']:.1f}%)")
            t1 = time.time()
            result = solve_pathway(seed, cfg, P32, dm32, dn32)
            result["beam_index"] = bi
            result["scenario_name"] = cfg.scenario_name
            dt = time.time() - t1
            print(f"[solve] Done in {dt:.1f}s — {result['n_thresholds_achieved']} "
                  f"thresholds, peak gas={result['peak_gas_mw']:,.0f} MW")
            write_result(result)

    print(f"\n{'='*70}")
    print(f"[adaptive] All done in {time.time()-t_total:.0f}s")
    print(f"{'='*70}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
