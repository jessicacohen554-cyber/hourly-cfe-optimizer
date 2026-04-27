#!/usr/bin/env python3
"""
step2_3_reliability_tax_adaptive.py
On-the-fly mix generation with marginal-yield-bounded Latin Hypercube sampling.

Architecture:
  Stage 1: Load Step 2.1 EF parquets for first threshold band above baseline.
           Cluster into archetypes, pick N diverse low-cost seeds.
  Stage 2: For each seed, build year-by-year pathway to 99.9% CFE via on-the-fly
           LHS generation. Write results at every threshold crossed.

Two pathways:
  A — VRE + hybrids + storage only. No offshore, geo, CCS, or clean_firm expansion.
  B — Full P3: all resources incl nuclear, CCS, offshore, geothermal (CAISO only).

Two cost modes:
  Mode 1 — Incremental clean cost only.
  Mode 2 — Incremental clean cost minus gas savings.

Performance notes (v3.7):
  - H2 uses interleaved peaker dispatch (no windowing) — cycles freely like
    a zero-carbon peaker. LCOE_TABLE recomputed for 1000hr CAPEX basis.
  - Serial numba kernel for marginal yields (~15 candidates) avoids prange overhead.
  - LHS sample bank pre-generated at pathway start; affine-scaled per year.
  - Additive cascade: rounds accumulate instead of discarding prior results.
  - Score caching: _find_winner returns curtailment; no redundant _score_single.
  - LCOE/storage cost matrices pre-computed for all 26 years before year loop.

See step2_3_rebuild_decisions.md for full methodology specification (25 decisions).

Usage:
    python scripts/step2_3_reliability_tax_adaptive.py --iso NYISO --pathway A
    python scripts/step2_3_reliability_tax_adaptive.py --iso ERCOT --pathway B --demand-growth High
    python scripts/step2_3_reliability_tax_adaptive.py --iso CAISO --pathway B --batch
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
YEARS = list(range(BASE_YEAR, END_YEAR + 1))
N_YEARS = len(YEARS)
DISCOUNT_RATES = {"5pct": 0.05, "7pct": 0.07, "9pct": 0.09}
WORST_HOUR_PCTILE = 99.97
CCGT_OVERNIGHT_CAPEX_USD_KW = 1200.0
GAS_CF = 0.45
RA_MARGIN = pc.RESOURCE_ADEQUACY_MARGIN  # 0.15
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
    # name: (duration_hrs, efficiency, window_hrs, lcoe_key, cost_duration_hrs)
    # cost_duration_hrs: used in storage_net_cost (coeff × cost_dur = $/MW-yr).
    # Distinct from dispatch duration to allow future decoupling.
    "battery_dispatch_pct":  (4,    0.85, 24,  "battery",  4),
    "battery8_dispatch_pct": (8,    0.85, 48,  "battery8", 8),
    "ldes_dispatch_pct":     (100,  0.50, 168, "ldes",     100),
    # H2: 1000hr seasonal salt cavern storage. LCOE_TABLE recomputed at 1000hr
    # CAPEX basis ($54/kWh Medium). Dispatch uses _inline_storage_peaker (no
    # windowing) — H2 cycles hour-by-hour like a zero-carbon peaker.
    # Window field unused by peaker kernel but kept for schema consistency.
    "h2_dispatch_pct":       (1000, 0.35, 720, "h2",       1000),
}

EF_BAND_THRESHOLDS = (
    10, 20, 30, 40, 50, 55, 60, 65, 70, 75,
    80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 99.5, 99.9,
)

CFE_WAYPOINTS = ((2030, 50), (2035, 70), (2040, 90), (2045, 95), (2050, 99.9))

PATHWAY_A_FREE = ["solar", "wind", "solar_batt4", "solar_batt8",
                  "wind_batt4", "wind_batt8"]
PATHWAY_B_FREE = PATHWAY_A_FREE + ["offshore_wind", "clean_firm"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class RunConfig:
    iso: str
    pathway: str  # "A" or "B"
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
    stage1_samples: int = 0
    stage2_samples: int = 5000
    cost_mode: int = 1

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
    def scaled_samples(self) -> int:
        return int(self.stage2_samples * (self.n_dims / 10.0) ** 1.5)

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
    """(N_RESOURCES, 8760) supply-profile matrix in RESOURCE_ORDER."""
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
# Numba kernels
# ---------------------------------------------------------------------------
@njit
def _inline_storage_stage(surplus, gap, total_dispatch,
                          capacity, power_rating, efficiency, window_hours):
    """Windowed charge/discharge for one storage tier. Modifies arrays in-place."""
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
    """Hour-by-hour interleaved charge/discharge. No windowing.

    SOC carries across all 8760 hours — models a salt cavern + turbine
    that charges from surplus VRE whenever available and dispatches into
    deficit hours on demand, like a zero-carbon peaker.

    Used for H2 (1000hr salt cavern): the enormous energy capacity means
    SOC rarely constrains dispatch. The turbine fires hundreds of hours/yr
    covering overnight lulls and multi-day weather events.
    """
    if capacity <= 0.0:
        return
    soc = 0.0
    for h in range(H):
        # Charge: absorb surplus into cavern
        s = surplus[h]
        if s > 0.0 and soc < capacity:
            charge = min(s, power_rating, capacity - soc)
            soc += charge
            surplus[h] -= charge
        # Discharge: fire turbine into deficit
        g = gap[h]
        if g > 0.0 and soc > 0.0:
            discharge = min(g, power_rating, soc * efficiency)
            total_dispatch[h] += discharge
            soc -= discharge / efficiency
            gap[h] -= discharge


@njit
def _score_one(i, W, P32, dm32, dn32, b4_val, b8_val, ld_val, h2_val,
               scores, resid, curtail):
    """Score a single candidate. Shared logic for parallel and serial kernels."""
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
    """Parallel scoring kernel for large batches (>50 candidates)."""
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
    """Serial scoring kernel for small batches (<50 candidates).

    Avoids prange thread-pool wake overhead that dominates when N is small.
    """
    n = W.shape[0]
    for i in range(n):
        _score_one(i, W, P32, dm32, dn32,
                   float(batt4[i]), float(batt8[i]),
                   float(ldes[i]), float(h2[i]),
                   scores, resid, curtail)


# ---------------------------------------------------------------------------
# Public scoring wrappers
# ---------------------------------------------------------------------------
SERIAL_THRESHOLD = 50


def score_candidates(W: np.ndarray, P32: np.ndarray, dm32: np.ndarray,
                     dn32: np.ndarray, batt4: np.ndarray, batt8: np.ndarray,
                     ldes: np.ndarray, h2: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score N candidate mixes. Auto-selects serial vs parallel kernel."""
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
def wrights_law_cost(foak: float, noak: float, year: int,
                     t_start: int, t_noak: int) -> float:
    if year <= t_start:
        return foak
    if year >= t_noak:
        return noak
    progress = (year - t_start) / (t_noak - t_start)
    return noak + (foak - noak) * np.exp(-WRIGHTS_LAW_K * progress)


def get_resource_lcoe(iso: str, resource: str, year: int, cfg: RunConfig) -> float:
    """Delivered LCOE (LCOE + TX adder) for a resource in a given year."""
    lcoe = _base_lcoe(iso, resource, year, cfg)
    tx = _tx_adder(iso, resource, cfg.tx_level)
    return lcoe + tx


def _base_lcoe(iso: str, resource: str, year: int, cfg: RunConfig) -> float:
    ren_name = LEVEL_NAME[cfg.ren_cost]
    batt_name = LEVEL_NAME[cfg.batt_cost]

    if resource == "solar":
        return pc.LCOE_TABLES["solar"][ren_name][iso]
    if resource == "wind":
        return pc.LCOE_TABLES["wind"][ren_name][iso]
    if resource.startswith("solar_batt"):
        ren_lcoe = pc.LCOE_TABLES["solar"][ren_name][iso]
        batt_adder = {"L": 20.0, "M": 30.0, "H": 45.0}[cfg.batt_cost]
        return ren_lcoe + batt_adder
    if resource.startswith("wind_batt"):
        ren_lcoe = pc.LCOE_TABLES["wind"][ren_name][iso]
        batt_adder = {"L": 17.0, "M": 25.0, "H": 38.0}[cfg.batt_cost]
        return ren_lcoe + batt_adder
    if resource == "hydro":
        return 50.0

    fl = cfg.firm_cost
    cl = cfg.ccs_cost
    gl = cfg.geo_cost

    if resource == "clean_firm":
        foak = pc.FOAK_NUCLEAR_NEWBUILD[iso] * NUCLEAR_FOAK_DISCOUNT
        noak = pc.NUCLEAR_NEWBUILD_LCOE[fl][iso]
        fs, ny = pc.get_pathway_noak_window("nuclear", fl, "3")
        return wrights_law_cost(foak, noak, year, fs, ny)

    if resource == "ccs_ccgt":
        q45_on = cfg.q45 == "1"
        if q45_on:
            noak = pc.CCS_LCOE_45Q_ON[cl][iso]
            foak = pc.FOAK_CCS_45Q_ON[iso]
        else:
            noak = pc.CCS_LCOE_45Q_OFF[cl][iso]
            foak = pc.FOAK_CCS_45Q_OFF[iso]
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


def _tx_adder(iso: str, resource: str, tx_level: str) -> float:
    remap = {"solar_batt4": "solar", "solar_batt8": "solar",
             "wind_batt4": "wind", "wind_batt8": "wind"}
    key = remap.get(resource, resource)
    if key in ("solar", "wind", "clean_firm", "ccs_ccgt", "offshore_wind"):
        level_name = LEVEL_NAME.get(tx_level, tx_level)
        return float(pc.get_tx(key, level_name, iso))
    return 0.0


def storage_net_cost(iso: str, storage_key: str, cfg: RunConfig) -> float:
    """Annualized cost of 1 MW of storage capacity ($/MW-yr).

    Coefficient × cost_duration_hrs converts system-cost coefficient → $/MW-yr.
    cost_duration_hrs matches the LCOE_TABLE CAPEX basis (e.g. 1000hr for H2
    after recompute, vs dispatch duration which is also 1000hr but conceptually
    distinct).
    """
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
# LCOE pre-computation — avoids per-year dict lookups in hot loop
# ---------------------------------------------------------------------------
def precompute_lcoe_matrix(iso: str, cfg: RunConfig) -> np.ndarray:
    """(N_YEARS, N_RESOURCES) delivered LCOE matrix."""
    mat = np.zeros((N_YEARS, N_RESOURCES), dtype=np.float64)
    for yi, year in enumerate(YEARS):
        for ri, res in enumerate(RESOURCE_ORDER):
            mat[yi, ri] = get_resource_lcoe(iso, res, year, cfg)
    return mat


def precompute_storage_costs(iso: str, cfg: RunConfig) -> np.ndarray:
    """(4,) storage net costs — constant across years."""
    return np.array([storage_net_cost(iso, sc, cfg)
                     for sc in STORAGE_COLS], dtype=np.float64)


# ---------------------------------------------------------------------------
# Demand model
# ---------------------------------------------------------------------------
def demand_twh_vec(iso: str, level: str = "Medium") -> np.ndarray:
    base = float(pc.REGIONAL_DEMAND_TWH[iso])
    rate = pc.DEMAND_GROWTH_RATES[iso][level]
    return np.array([base * (1 + rate) ** (y - BASE_YEAR) for y in YEARS])


def peak_demand_vec(iso: str, level: str = "Medium") -> np.ndarray:
    base = float(pc.PEAK_DEMAND_MW[iso])
    rate = pc.DEMAND_GROWTH_RATES[iso][level]
    return np.array([base * (1 + rate) ** (y - BASE_YEAR) for y in YEARS])


def cfe_target(year: int, base_pct: float) -> float:
    pts = [(BASE_YEAR, base_pct)] + list(CFE_WAYPOINTS)
    if year <= pts[0][0]:
        return pts[0][1]
    if year >= pts[-1][0]:
        return pts[-1][1]
    for (y0, t0), (y1, t1) in zip(pts, pts[1:]):
        if y0 <= year <= y1:
            return t0 + (t1 - t0) * (year - y0) / (y1 - y0)
    return pts[-1][1]


def cfe_target_vec(iso: str) -> np.ndarray:
    base = sum(pc.GRID_MIX_SHARES[iso].values())
    return np.array([cfe_target(y, base) for y in YEARS])


# ---------------------------------------------------------------------------
# Clean firm tranche decomposition (Pathway B only)
# ---------------------------------------------------------------------------
def decompose_clean_firm(cf_twh: float, iso: str, year: int,
                         cfg: RunConfig) -> dict:
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
        nuke_new = rem
        ccs = 0.0
    else:
        ccs = min(rem, cc)
        nuke_new = rem - ccs

    return {
        "existing_twh": existing, "existing_lcoe": 0.0,
        "uprate_twh": uprate, "uprate_lcoe": uprate_lcoe,
        "geo_twh": geo, "geo_lcoe": geo_lcoe,
        "nuke_new_twh": nuke_new, "nuke_new_lcoe": nuke_lcoe,
        "ccs_twh": ccs, "ccs_lcoe": ccs_lcoe,
    }


# ---------------------------------------------------------------------------
# Gas sizing
# ---------------------------------------------------------------------------
def gas_need_mw(resid_p9997: float, demand_twh: float, existing_gas_mw: float,
                gaf: float) -> float:
    worst_hour_gap_mw = resid_p9997 * demand_twh * 1e6
    return max(0.0, worst_hour_gap_mw / gaf - existing_gas_mw)


def gas_annual_cost(new_gas_mw: float, iso: str, cfg: RunConfig) -> float:
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
# Marginal yield computation — uses serial kernel to avoid prange overhead
# ---------------------------------------------------------------------------
def compute_marginal_yields(
    floor_pcts: np.ndarray,
    floor_storage: np.ndarray,
    free_res_idx: list[int],
    P32: np.ndarray, dm32: np.ndarray, dn32: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Marginal CFE yield per +1pp of each free dimension.

    Returns (res_yields, stor_yields, floor_cfe, floor_resid, floor_curtail).
    Uses serial kernel — only ~15 candidates, prange overhead would dominate.
    """
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
    b4[so + 0] += np.float32(1.0)
    b8[so + 1] += np.float32(1.0)
    ld[so + 2] += np.float32(1.0)
    h2[so + 3] += np.float32(1.0)

    # Serial kernel — ~15 candidates, avoids thread-pool wake
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
# LHS sample bank — pre-generated at pathway start, reused across years
# ---------------------------------------------------------------------------
class LHSBank:
    """Pre-generated unit hypercube samples. Affine-scaled per year.

    LHS stratification is preserved under affine transforms, so pre-generating
    once and scaling to [floor, ceiling] each year gives identical statistical
    properties to fresh generation — without the O(n·d·log n) construction cost.
    """

    def __init__(self, n_dims: int, total_samples: int, seed: int = 42):
        sampler = LatinHypercube(d=n_dims, seed=seed)
        self.samples = sampler.random(n=total_samples).astype(np.float64)
        self.n_dims = n_dims
        self._offset = 0

    def reset(self):
        """Reset offset for a new year's cascade."""
        self._offset = 0

    def draw(self, n: int) -> np.ndarray:
        """Draw n unit samples in [0,1]^d, advancing the offset."""
        end = self._offset + n
        if end <= len(self.samples):
            batch = self.samples[self._offset:end]
            self._offset = end
            return batch
        # Bank exhausted — fall back to fresh generation
        sampler = LatinHypercube(d=self.n_dims)
        return sampler.random(n=n).astype(np.float64)


# ---------------------------------------------------------------------------
# LHS candidate generation
# ---------------------------------------------------------------------------
def generate_candidates(
    floor_pcts: np.ndarray,
    floor_storage: np.ndarray,
    target_cfe: float,
    floor_cfe: float,
    res_yields: np.ndarray,
    stor_yields: np.ndarray,
    free_res_idx: list[int],
    cfg: RunConfig,
    n_samples: int,
    unit_samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray]:
    """Generate LHS candidates from pre-drawn unit samples.

    Returns (W, batt4, batt8, ldes, h2, raw_pcts).
    """
    n_free = len(free_res_idx)
    increment = max(0.1, target_cfe - floor_cfe)

    n_active = (sum(1 for y in res_yields if y > 0.01)
                + sum(1 for y in stor_yields if y > 0.01))
    n_active = max(n_active, 1)
    # Last mile (≥97% CFE): each dimension needs full headroom — overshooting
    # is cheap (just curtailment) but undershooting is a cascade miss.
    # Below 97%: divide by n_active so total expected CFE ≈ increment.
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

    stor_ceilings = np.zeros(4, dtype=np.float64)
    for j in range(4):
        yld = stor_yields[j]
        if yld > 0.001:
            stor_ceilings[j] = floor_storage[j] + increment / yld * dim_sf
        else:
            stor_ceilings[j] = floor_storage[j] + 1.0

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

    n_dims = n_free + 4
    floors = np.zeros(n_dims)
    ceilings = np.zeros(n_dims)
    for k in range(n_free):
        floors[k] = floor_pcts[free_res_idx[k]]
        ceilings[k] = res_ceilings[k]
    for j in range(4):
        floors[n_free + j] = floor_storage[j]
        ceilings[n_free + j] = stor_ceilings[j]

    ceilings = np.maximum(ceilings, floors + 0.01)

    # Scale pre-drawn unit samples to [floor, ceiling]
    raw = floors[None, :] + unit_samples[:n_samples] * (ceilings - floors)[None, :]

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
# Stage 1 — Archetype selection from EF parquets
# ---------------------------------------------------------------------------
def load_ef_band(iso: str, threshold: float) -> Optional[pa.Table]:
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


def classify_archetype(mix_pcts: dict[str, float], pathway: str) -> str:
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


def stage1_select(iso: str, cfg: RunConfig, P32: np.ndarray,
                  dm32: np.ndarray, dn32: np.ndarray
                  ) -> list[dict]:
    base_pct = sum(pc.GRID_MIX_SHARES[iso].values())
    first_threshold = None
    for t in EF_BAND_THRESHOLDS:
        if t > base_pct:
            first_threshold = t
            break
    if first_threshold is None:
        print(f"[stage1] {iso}: baseline {base_pct:.1f}% exceeds all thresholds")
        return []

    print(f"[stage1] {iso}: baseline={base_pct:.1f}%, first threshold={first_threshold}%")
    ef = load_ef_band(iso, first_threshold)
    if ef is None:
        print(f"[stage1] {iso}: no EF parquet for threshold {first_threshold}")
        return []

    n = ef.num_rows
    print(f"[stage1] {iso}: loaded {n:,} mixes from EF parquets")

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
        if t > first_threshold:
            next_threshold = t
            break
    mask = (scores >= first_threshold - 0.5) & (scores < next_threshold)
    valid_idx = np.where(mask)[0]
    if len(valid_idx) == 0:
        mask = scores >= base_pct
        valid_idx = np.where(mask)[0]
    if len(valid_idx) == 0:
        print(f"[stage1] {iso}: no valid mixes found")
        return []

    print(f"[stage1] {iso}: {len(valid_idx):,} mixes in valid band")

    # Vectorized cost ranking
    base_dem = float(pc.REGIONAL_DEMAND_TWH[iso])
    res_lcoes = np.array([get_resource_lcoe(iso, res, BASE_YEAR, cfg)
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
        seed = {
            "archetype": arch,
            "resource_pcts": {res: float(W[vi, ri]) * 100.0
                              for ri, res in enumerate(RESOURCE_ORDER)},
            "storage_pcts": {
                STORAGE_COLS[0]: float(batt4[vi]),
                STORAGE_COLS[1]: float(batt8[vi]),
                STORAGE_COLS[2]: float(ldes_arr[vi]),
                STORAGE_COLS[3]: float(h2_arr[vi]),
            },
            "cfe": float(scores[vi]),
            "resid": float(resid[vi]),
            "cost": cost,
        }
        seeds.append(seed)
        if len(seeds) >= cfg.scaled_beams:
            break

    if len(seeds) < cfg.scaled_beams:
        rank = np.argsort(costs)
        selected_vis = {s["resource_pcts"]["solar"] for s in seeds}
        for ki in rank:
            if len(seeds) >= cfg.scaled_beams:
                break
            vi = valid_idx[ki]
            mix = {res: float(W[vi, ri]) * 100.0 for ri, res in enumerate(RESOURCE_ORDER)}
            arch = classify_archetype(mix, cfg.pathway)
            seed = {
                "archetype": arch,
                "resource_pcts": mix,
                "storage_pcts": {
                    STORAGE_COLS[0]: float(batt4[vi]),
                    STORAGE_COLS[1]: float(batt8[vi]),
                    STORAGE_COLS[2]: float(ldes_arr[vi]),
                    STORAGE_COLS[3]: float(h2_arr[vi]),
                },
                "cfe": float(scores[vi]),
                "resid": float(resid[vi]),
                "cost": costs[ki],
            }
            seeds.append(seed)

    print(f"[stage1] {iso}: selected {len(seeds)} seeds: "
          f"{[s['archetype'] for s in seeds]}")
    return seeds


# ---------------------------------------------------------------------------
# Stage 2 — Year-by-year on-the-fly solve
# ---------------------------------------------------------------------------
def solve_pathway(seed: dict, cfg: RunConfig,
                  P32: np.ndarray, dm32: np.ndarray, dn32: np.ndarray
                  ) -> dict:
    """Build a full 2025-2050 pathway from a seed mix."""
    iso = cfg.iso
    demand_vec = demand_twh_vec(iso, cfg.demand_growth)
    peak_vec = peak_demand_vec(iso, cfg.demand_growth)
    cfe_targets = cfe_target_vec(iso)
    existing_gas = float(pc.EXISTING_GAS_CAPACITY_MW[iso])
    gaf = float(pc.GAS_AVAILABILITY_FACTOR[iso])
    free_idx = cfg.free_indices

    # Pre-compute LCOE and storage cost matrices (Patch 6)
    lcoe_matrix = precompute_lcoe_matrix(iso, cfg)  # (N_YEARS, N_RESOURCES)
    stor_costs = precompute_storage_costs(iso, cfg)  # (4,)

    # Pre-generate LHS sample bank (Patch 3)
    # Bank holds enough for 5× scaled_samples per year (1× + 1× + 2× cascade + margin)
    bank_total = cfg.scaled_samples * 5
    lhs_bank = LHSBank(cfg.n_dims, bank_total, seed=42)

    base_demand = float(pc.REGIONAL_DEMAND_TWH[iso])
    frozen_twh = {}
    frozen_twh["hydro"] = (pc.GRID_MIX_SHARES[iso].get("hydro", 0) / 100.0
                           * base_demand)
    if cfg.pathway == "A":
        frozen_twh["clean_firm"] = (pc.GRID_MIX_SHARES[iso].get("clean_firm", 0)
                                    / 100.0 * base_demand)

    baseline_cf_twh = (pc.GRID_MIX_SHARES[iso].get("clean_firm", 0) / 100.0
                       * base_demand)

    floor_pcts = np.zeros(N_RESOURCES, dtype=np.float64)
    for ri, res in enumerate(RESOURCE_ORDER):
        floor_pcts[ri] = seed["resource_pcts"].get(res, 0.0)

    cf_baseline_pct = pc.GRID_MIX_SHARES[iso].get("clean_firm", 0)
    hydro_baseline_pct = pc.GRID_MIX_SHARES[iso].get("hydro", 0)
    floor_pcts[RES_IDX["clean_firm"]] = max(floor_pcts[RES_IDX["clean_firm"]],
                                             cf_baseline_pct)
    floor_pcts[RES_IDX["hydro"]] = max(floor_pcts[RES_IDX["hydro"]],
                                        hydro_baseline_pct)

    floor_storage = np.array([
        seed["storage_pcts"].get(sc, 0.0) for sc in STORAGE_COLS
    ], dtype=np.float64)

    vintage_ledger = []
    prev_gas_cost = 0.0
    cumulative_cost = 0.0
    threshold_snapshots = []
    thresholds_crossed = set()
    peak_gas_mw = 0.0
    peak_gas_year = BASE_YEAR
    annual_records = []

    for yi, year in enumerate(YEARS):
        dem_twh = demand_vec[yi]
        pk_mw = peak_vec[yi]
        target = cfe_targets[yi]

        for res, twh in frozen_twh.items():
            ri = RES_IDX[res]
            floor_pcts[ri] = twh / dem_twh * 100.0

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

            cf_fs, cf_ny = pc.get_pathway_noak_window(
                "nuclear", cfg.firm_cost, "3")
            if year >= cf_fs:
                lf = pc.learning_fraction(year, cf_fs, cf_ny)
                baseline_pct = baseline_cf_twh / dem_twh * 100.0
                expansion_gap = max(0.0, target - baseline_pct)
                cf_commitment = baseline_pct + (
                    lf * expansion_gap * CLEAN_FIRM_DEPLOYMENT_SHARE)
                floor_pcts[RES_IDX["clean_firm"]] = max(
                    floor_pcts[RES_IDX["clean_firm"]], cf_commitment)

        if yi < N_YEARS - 1:
            cfe_ceiling = cfe_targets[yi + 1]
        else:
            cfe_ceiling = 100.01  # > max possible score (100.0), accepts all ≥ target

        # Marginal yields — serial kernel, returns floor scores (Patch 2 + 5)
        res_yields, stor_yields, floor_cfe, floor_resid, floor_curtail = (
            compute_marginal_yields(
                floor_pcts, floor_storage, free_idx, P32, dm32, dn32))

        if floor_cfe >= target:
            winner_pcts = floor_pcts.copy()
            winner_storage = floor_storage.copy()
            winner_cfe = floor_cfe
            winner_resid = floor_resid
            winner_curtail = floor_curtail
        else:
            # Reset LHS bank offset for this year's cascade (Patch 3)
            lhs_bank.reset()

            # Pre-computed LCOE row for this year (Patch 6)
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
                    prev_cf_twh = floor_pcts[ri] / 100.0 * dem_twh
                    new_cf_twh = winner_pcts[ri] / 100.0 * dem_twh
                    prev_tranches = decompose_clean_firm(prev_cf_twh, iso, year, cfg)
                    new_tranches = decompose_clean_firm(new_cf_twh, iso, year, cfg)
                    for tranche in ["uprate", "geo", "nuke_new", "ccs"]:
                        t_delta = (new_tranches[f"{tranche}_twh"]
                                   - prev_tranches[f"{tranche}_twh"])
                        if t_delta > 0.01:
                            t_lcoe = new_tranches[f"{tranche}_lcoe"]
                            vintage_ledger.append(
                                (year, f"clean_firm_{tranche}", t_delta, t_lcoe))
                            year_incremental_cost += t_delta * t_lcoe * 1e6
                else:
                    lcoe = lcoe_matrix[yi, ri]  # pre-computed (Patch 6)
                    vintage_ledger.append((year, res, delta_twh, lcoe))
                    year_incremental_cost += delta_twh * lcoe * 1e6

        for j, sc in enumerate(STORAGE_COLS):
            delta_stor = winner_storage[j] - floor_storage[j]
            if delta_stor > RATCHET_TOL_PCT:
                capacity_mw = delta_stor / 100.0 * dem_twh * 1e6 / 8760
                net_cost_mw_yr = stor_costs[j]  # pre-computed (Patch 6)
                annual_cost = capacity_mw * net_cost_mw_yr
                vintage_ledger.append((year, sc, capacity_mw, net_cost_mw_yr))
                year_incremental_cost += annual_cost

        g_mw = gas_need_mw(winner_resid, dem_twh, existing_gas, gaf)
        g_cost = gas_annual_cost(g_mw, iso, cfg)
        if g_mw > peak_gas_mw:
            peak_gas_mw = g_mw
            peak_gas_year = year

        _storage_names = set(STORAGE_COLS)
        total_vintage_cost = 0.0
        for _, res, qty, unit_cost in vintage_ledger:
            if res in _storage_names:
                total_vintage_cost += qty * unit_cost
            else:
                total_vintage_cost += qty * unit_cost * 1e6
        total_annual = total_vintage_cost + g_cost

        if cfg.cost_mode == 2:
            gas_savings = prev_gas_cost - g_cost
            year_net_cost = year_incremental_cost - gas_savings
        else:
            year_net_cost = year_incremental_cost

        cumulative_cost += total_annual
        prev_gas_cost = g_cost

        # Update ratchet floors
        for ri in range(N_RESOURCES):
            floor_twh = floor_pcts[ri] / 100.0 * dem_twh
            winner_twh = winner_pcts[ri] / 100.0 * dem_twh
            new_floor_twh = max(floor_twh, winner_twh)
            if yi < N_YEARS - 1:
                next_dem = demand_vec[yi + 1]
                floor_pcts[ri] = new_floor_twh / next_dem * 100.0
            else:
                floor_pcts[ri] = winner_pcts[ri]

        for j in range(4):
            floor_storage[j] = max(floor_storage[j], winner_storage[j])

        # Threshold snapshots — use cached curtailment (Patch 5)
        for t in EF_BAND_THRESHOLDS:
            if t not in thresholds_crossed and winner_cfe >= t:
                thresholds_crossed.add(t)
                snapshot = {
                    "threshold_pct": t,
                    "year_achieved": year,
                    "achieved_cfe_pct": round(float(winner_cfe), 2),
                    "resource_mix_pct": {
                        res: round(float(winner_pcts[ri]), 2)
                        for ri, res in enumerate(RESOURCE_ORDER)
                    },
                    "storage_dispatch_pct": {
                        STORAGE_COLS[j]: round(float(winner_storage[j]), 2)
                        for j in range(4)
                    },
                    "gas_need_mw": round(g_mw, 1),
                    "peak_gas_mw": round(peak_gas_mw, 1),
                    "stranded_vs_peak_mw": round(peak_gas_mw - g_mw, 1),
                    "total_annual_cost_usd": round(total_annual, 0),
                    "incremental_cost_usd": round(year_incremental_cost, 0),
                    "cumulative_cost_usd": round(cumulative_cost, 0),
                    "curtailment_total": round(float(winner_curtail), 4),
                    "demand_twh": round(dem_twh, 1),
                }
                threshold_snapshots.append(snapshot)

        if year % 5 == 0 or winner_cfe >= 99.0:
            print(f"  {year}: CFE={winner_cfe:.1f}% target={target:.1f}% "
                  f"gas={g_mw:,.0f}MW cost=${total_annual/1e9:.2f}B")

    return {
        "schema_version": "3.7",
        "iso": iso,
        "pathway": cfg.pathway,
        "dim_key": cfg.dim_key,
        "cost_mode": cfg.cost_mode,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "config": {
            "demand_growth": cfg.demand_growth,
            "firm_cost": cfg.firm_cost,
            "ccs_cost": cfg.ccs_cost,
            "tx_level": cfg.tx_level,
            "n_beams": cfg.n_beams,
            "n_beams_scaled": cfg.scaled_beams,
            "stage2_samples": cfg.stage2_samples,
            "stage2_samples_scaled": cfg.scaled_samples,
            "safety_factor": SAFETY_FACTOR,
            "wrights_law_k": WRIGHTS_LAW_K,
        },
        "beam_archetype": seed["archetype"],
        "peak_gas_mw": round(peak_gas_mw, 1),
        "peak_gas_year": peak_gas_year,
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


def _find_winner(floor_pcts, floor_storage, target, ceiling,
                 floor_cfe, res_yields, stor_yields,
                 free_idx, cfg, P32, dm32, dn32,
                 year, dem_twh, lhs_bank, res_lcoes_free, stor_costs):
    """Find cheapest candidate in [target, ceiling) CFE band.

    Additive cascade (Patch 4): rounds accumulate — prior results are kept,
    only new samples are scored each round.

    Returns (winner_pcts, winner_storage, cfe, resid, curtail) or 5× None.
    """
    free_idx_arr = np.array(free_idx, dtype=np.intp)
    floor_free = floor_pcts[free_idx_arr]

    use_gas_shadow = (getattr(pc, "INCLUDE_GAS_COST_IN_ARGMIN", False)
                      and cfg.pathway == "B")
    if use_gas_shadow:
        existing_gas = float(pc.EXISTING_GAS_CAPACITY_MW[cfg.iso])
        gaf = float(pc.GAS_AVAILABILITY_FACTOR[cfg.iso])
        gas_useful_life = 30.0
        years_remaining = max(1.0, float(END_YEAR - year))
        stranding_frac = max(0.0, 1.0 - years_remaining / gas_useful_life)
        gas_capex_per_mw = CCGT_OVERNIGHT_CAPEX_USD_KW * 1000.0

    # Additive cascade: accumulate results across rounds
    # Round sizes: [1×, 1×, 2×] = 4× total in worst case
    round_sizes = [cfg.scaled_samples,
                   cfg.scaled_samples,
                   cfg.scaled_samples * 2]
    acc_scores = []
    acc_resid = []
    acc_curtail = []
    acc_W = []
    acc_b4, acc_b8, acc_ld, acc_h2 = [], [], [], []
    total_scored = 0

    for round_num, n_new in enumerate(round_sizes):
        unit = lhs_bank.draw(n_new)
        W, b4, b8, ld, h2, raw = generate_candidates(
            floor_pcts, floor_storage, target, floor_cfe,
            res_yields, stor_yields, free_idx, cfg, n_new, unit)

        scores, resid_arr, curtail_arr = score_candidates(
            W, P32, dm32, dn32, b4, b8, ld, h2)

        acc_scores.append(scores)
        acc_resid.append(resid_arr)
        acc_curtail.append(curtail_arr)
        acc_W.append(W)
        acc_b4.append(b4)
        acc_b8.append(b8)
        acc_ld.append(ld)
        acc_h2.append(h2)
        total_scored += n_new

        # Check all accumulated candidates
        all_scores = np.concatenate(acc_scores)
        mask = (all_scores >= target) & (all_scores < ceiling)
        valid = np.where(mask)[0]

        if len(valid) == 0:
            if round_num < 2:
                print(f"    cascade round {round_num}: "
                      f"0/{total_scored} in [{target:.1f}, {ceiling:.1f})")
            continue

        # Found valid candidates — compute costs
        all_W = np.concatenate(acc_W)
        all_b4 = np.concatenate(acc_b4)
        all_b8 = np.concatenate(acc_b8)
        all_ld = np.concatenate(acc_ld)
        all_h2 = np.concatenate(acc_h2)
        all_resid = np.concatenate(acc_resid)
        all_curtail = np.concatenate(acc_curtail)

        # Vectorized incremental cost (Patch 6: uses pre-computed LCOEs)
        W_valid_pct = all_W[valid][:, free_idx_arr].astype(np.float64) * 100.0
        res_deltas = np.maximum(W_valid_pct - floor_free, 0.0)
        costs = np.sum(res_deltas * (dem_twh / 100.0 * res_lcoes_free * 1e6),
                       axis=1)

        stor_vals = np.column_stack([all_b4[valid], all_b8[valid],
                                     all_ld[valid], all_h2[valid]]
                                     ).astype(np.float64)
        stor_deltas = np.maximum(stor_vals - floor_storage, 0.0)
        stor_mw = stor_deltas * (dem_twh * 1e6 / 8760 / 100.0)
        costs += np.sum(stor_mw * stor_costs, axis=1)

        if use_gas_shadow and stranding_frac > 0:
            resid_valid = all_resid[valid].astype(np.float64)
            gas_mw = np.maximum(resid_valid * dem_twh * 1e6 / gaf - existing_gas,
                                0.0)
            costs += gas_mw * gas_capex_per_mw * stranding_frac

        best_local = np.argmin(costs)
        best = valid[best_local]
        winner_pcts = all_W[best].astype(np.float64) * 100.0
        frozen_mask = np.ones(N_RESOURCES, dtype=bool)
        frozen_mask[free_idx_arr] = False
        winner_pcts[frozen_mask] = floor_pcts[frozen_mask]
        winner_storage = np.array([all_b4[best], all_b8[best],
                                   all_ld[best], all_h2[best]],
                                   dtype=np.float64)
        return (winner_pcts, winner_storage,
                float(all_scores[best]), float(all_resid[best]),
                float(all_curtail[best]))

    # ── Storage-focused fallback for last mile (≥97% CFE) ──
    # Standard cascade failed. At high CFE the binding constraint is storage
    # depth, not VRE volume. Fix VRE near floor with small perturbation,
    # sample storage aggressively in 4D only → much denser coverage of the
    # feasible region.
    if target >= 97.0:
        n_stor_focused = cfg.scaled_samples * 2
        print(f"    storage-focused fallback: {n_stor_focused} samples, 4D")
        sampler = LatinHypercube(d=4, seed=int(time.time() * 1000) % (2**31))
        stor_unit = sampler.random(n=n_stor_focused)

        # Storage ceilings: 5× floor or floor + 30pp, whichever is larger
        stor_floors = floor_storage.copy()
        stor_ceil = np.maximum(floor_storage * 5.0, floor_storage + 30.0)
        stor_ceil = np.minimum(stor_ceil, 50.0)  # hard cap
        stor_ceil = np.maximum(stor_ceil, stor_floors + 1.0)
        stor_raw = stor_floors + stor_unit * (stor_ceil - stor_floors)

        # VRE: small random perturbation above floor (0-5pp per resource)
        rng = np.random.default_rng(seed=42)
        W_sf = np.zeros((n_stor_focused, N_RESOURCES), dtype=np.float32)
        for ri in range(N_RESOURCES):
            W_sf[:, ri] = np.float32(floor_pcts[ri] * 0.01)
        for ri in free_idx:
            noise = rng.uniform(0, 5.0, size=n_stor_focused)
            W_sf[:, ri] = ((floor_pcts[ri] + noise) * 0.01).astype(np.float32)

        sf_b4 = stor_raw[:, 0].astype(np.float32)
        sf_b8 = stor_raw[:, 1].astype(np.float32)
        sf_ld = stor_raw[:, 2].astype(np.float32)
        sf_h2 = stor_raw[:, 3].astype(np.float32)

        sf_scores, sf_resid, sf_curtail = score_candidates(
            W_sf, P32, dm32, dn32, sf_b4, sf_b8, sf_ld, sf_h2)

        mask = (sf_scores >= target) & (sf_scores < ceiling)
        valid = np.where(mask)[0]

        if len(valid) > 0:
            print(f"    storage-focused: {len(valid)} hits")
            W_valid_pct = W_sf[valid][:, free_idx_arr].astype(np.float64) * 100.0
            res_deltas = np.maximum(W_valid_pct - floor_free, 0.0)
            costs = np.sum(res_deltas * (dem_twh / 100.0 * res_lcoes_free * 1e6),
                           axis=1)
            sf_stor_vals = np.column_stack([sf_b4[valid], sf_b8[valid],
                                            sf_ld[valid], sf_h2[valid]]
                                            ).astype(np.float64)
            sf_stor_deltas = np.maximum(sf_stor_vals - floor_storage, 0.0)
            sf_stor_mw = sf_stor_deltas * (dem_twh * 1e6 / 8760 / 100.0)
            costs += np.sum(sf_stor_mw * stor_costs, axis=1)

            best_local = np.argmin(costs)
            best = valid[best_local]
            winner_pcts = W_sf[best].astype(np.float64) * 100.0
            frozen_mask = np.ones(N_RESOURCES, dtype=bool)
            frozen_mask[free_idx_arr] = False
            winner_pcts[frozen_mask] = floor_pcts[frozen_mask]
            winner_storage = np.array([sf_b4[best], sf_b8[best],
                                       sf_ld[best], sf_h2[best]],
                                       dtype=np.float64)
            return (winner_pcts, winner_storage,
                    float(sf_scores[best]), float(sf_resid[best]),
                    float(sf_curtail[best]))
        else:
            print(f"    storage-focused: 0 hits — infeasible")

    return None, None, None, None, None


def _score_single(pcts, storage, P32, dm32, dn32):
    """Score a single mix. Returns (cfe, resid, curtail)."""
    W = np.zeros((1, N_RESOURCES), dtype=np.float32)
    W[0] = (pcts * 0.01).astype(np.float32)
    s = storage.astype(np.float32)
    scores, resid, curtail = score_candidates(
        W, P32, dm32, dn32,
        s[0:1], s[1:2], s[2:3], s[3:4])
    return float(scores[0]), float(resid[0]), float(curtail[0])


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

    # Warm up serial kernel
    scores_s = np.full(n, -1.0, dtype=np.float32)
    resid_s = np.full(n, np.inf, dtype=np.float32)
    curtail_s = np.zeros(n, dtype=np.float32)
    _score_mixes_serial(scores_s, resid_s, curtail_s, W, P, dm, dn,
                        batt, zero, zero, zero)

    # Warm up parallel kernel
    scores_p = np.full(n, -1.0, dtype=np.float32)
    resid_p = np.full(n, np.inf, dtype=np.float32)
    curtail_p = np.zeros(n, dtype=np.float32)
    idx = np.arange(n, dtype=np.int64)
    _score_mixes_parallel(scores_p, resid_p, curtail_p, W, P, dm, dn,
                          batt, zero, zero, zero, idx)

    print(f"[adaptive] JIT warm-up: {time.time()-t0:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------
def write_result(result: dict):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fname = (f"{result['iso']}__pathway{result['pathway']}__"
             f"{result.get('scenario_name', 'custom')}__"
             f"mode{result['cost_mode']}__{result['beam_archetype']}__"
             f"dg_{result['config']['demand_growth']}__"
             f"{result['timestamp']}.json")
    path = OUTPUT_DIR / fname
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[output] wrote {path.name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_configs(iso: str, pathway: str, scenarios: list[str],
                  demand_levels: list[str], n_beams: int,
                  stage1_samples: int, stage2_samples: int) -> list[RunConfig]:
    configs = []
    for scen_name in scenarios:
        scen = COST_SCENARIOS[scen_name]
        for dg in demand_levels:
            for mode in [1, 2]:
                configs.append(RunConfig(
                    iso=iso, pathway=pathway,
                    scenario_name=scen_name,
                    demand_growth=dg,
                    ren_cost=scen["ren_cost"], firm_cost=scen["firm_cost"],
                    batt_cost=scen["batt_cost"], ldes_cost=scen["ldes_cost"],
                    fuel_cost=scen["fuel_cost"], tx_level=scen["tx_level"],
                    ccs_cost=scen["ccs_cost"], geo_cost=scen["geo_cost"],
                    q45=scen["q45"],
                    n_beams=n_beams,
                    stage1_samples=stage1_samples,
                    stage2_samples=stage2_samples,
                    cost_mode=mode,
                ))
    return configs


def main():
    ap = argparse.ArgumentParser(
        description="Step 2.3 Adaptive — on-the-fly reliability tax optimizer")
    ap.add_argument("--iso", type=str, required=True,
                    help="ISO/RTO region (e.g. NYISO, ERCOT)")
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
    ap.add_argument("--stage1-samples", type=int, default=0)
    ap.add_argument("--stage2-samples", type=int, default=5000)
    args = ap.parse_args()

    iso = args.iso.upper()
    if iso not in pc.ISOS:
        raise SystemExit(f"Unknown ISO: {iso}. Known: {list(pc.ISOS)}")

    scenarios = list(COST_SCENARIOS.keys()) if args.batch else [args.scenario]
    demand_levels = (["Low", "Medium", "High"] if args.batch_demand
                     else [args.demand_growth])

    print(f"\n{'='*70}")
    print(f"[adaptive] Step 2.3 Reliability Tax Adaptive (v3.7)")
    print(f"[adaptive] ISO={iso}  Pathway={args.pathway}")
    print(f"[adaptive] Scenarios: {scenarios}")
    print(f"[adaptive] Demand levels: {demand_levels}")
    print(f"{'='*70}\n")

    warmup_jit()

    print(f"[adaptive] Loading profiles for {iso}...")
    P = build_profile_matrix(iso)
    dn = _load_demand_norm(iso)
    dm = dn * (1.0 + RA_MARGIN)
    P32 = P.astype(np.float32)
    dm32 = dm.astype(np.float32)
    dn32 = dn.astype(np.float32)
    print(f"[adaptive] Profiles loaded. "
          f"Demand TWh={float(pc.REGIONAL_DEMAND_TWH[iso]):.1f}")

    configs = build_configs(iso, args.pathway, scenarios, demand_levels,
                            args.n_beams, args.stage1_samples, args.stage2_samples)
    print(f"[adaptive] {len(configs)} runs queued "
          f"({len(scenarios)} scenarios × {len(demand_levels)} demand levels "
          f"× 2 cost modes)")
    sample_cfg = configs[0]
    print(f"[adaptive] Dimensions: {sample_cfg.n_dims} | "
          f"Samples: {sample_cfg.stage2_samples}→{sample_cfg.scaled_samples} "
          f"(×{sample_cfg.scaled_samples/sample_cfg.stage2_samples:.2f}) | "
          f"Beams: {sample_cfg.n_beams}→{sample_cfg.scaled_beams}")

    base_cfg = configs[0]
    seeds = stage1_select(iso, base_cfg, P32, dm32, dn32)
    if not seeds:
        print(f"[adaptive] No seeds found. Exiting.")
        return 1

    t_total = time.time()
    for ci, cfg in enumerate(configs):
        print(f"\n{'─'*60}")
        print(f"[adaptive] Config {ci+1}/{len(configs)}: "
              f"{cfg.scenario_name} | {cfg.demand_growth} | mode={cfg.cost_mode}")
        print(f"[adaptive] dim_key: {cfg.dim_key}")
        print(f"{'─'*60}")

        for bi, seed in enumerate(seeds):
            print(f"\n[solve] Beam {bi+1}/{len(seeds)}: {seed['archetype']} "
                  f"(seed CFE={seed['cfe']:.1f}%)")
            t1 = time.time()
            result = solve_pathway(seed, cfg, P32, dm32, dn32)
            result["beam_index"] = bi
            result["scenario_name"] = cfg.scenario_name
            dt = time.time() - t1

            n_achieved = result["n_thresholds_achieved"]
            print(f"[solve] Done in {dt:.1f}s — {n_achieved} thresholds achieved, "
                  f"peak gas={result['peak_gas_mw']:,.0f} MW")
            write_result(result)

    print(f"\n{'='*70}")
    print(f"[adaptive] All done in {time.time()-t_total:.0f}s")
    print(f"{'='*70}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
