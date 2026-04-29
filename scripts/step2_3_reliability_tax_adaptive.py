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

See step2_3_rebuild_decisions.md for full methodology specification (25 decisions).

v3.8 changes:
  - FIX: _inline_storage_stage now writes total_dispatch (was broken — storage invisible)
  - ADD: Storage learning curves (Wright's Law k=5) for battery/LDES/H2
  - ADD: Gas stranding shadow price in _find_winner
  - FIX: Offshore wind learning curve timing tracks cfg.ren_cost
  - FIX: gas_annual_cost fuel calc simplified

Usage:
    python scripts/step2_3_reliability_tax_adaptive.py --iso NYISO --pathway A
    python scripts/step2_3_reliability_tax_adaptive.py --iso ERCOT --pathway B --demand-growth High
    python scripts/step2_3_reliability_tax_adaptive.py --iso CAISO --pathway B --batch
"""
from __future__ import annotations

import argparse
import json
import sys
import hashlib
import time
from dataclasses import dataclass
from datetime import datetime
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
GAS_CF = 0.45
RA_MARGIN = pc.RESOURCE_ADEQUACY_MARGIN  # 0.15
RATCHET_TOL_PCT = 0.01
SAFETY_FACTOR = 1.5
WRIGHTS_LAW_K = 5.0  # compressed curve — reaches ~99% of NOAK by t_noak
NUCLEAR_FOAK_DISCOUNT = 0.80  # next builds start at 80% of Vogtle-era FOAK

# Gas stranding shadow price: penalizes candidates that increase peak gas MW.
# Weight × annualized capex × years_remaining_fraction applied per MW of new peak.
# 0.5 = charge 50% of expected stranding cost. Set to 0.0 to disable.
STRANDING_SHADOW_WEIGHT = 0.5

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
    # name: (duration_hrs, efficiency, window_hrs, lcoe_key)
    "battery_dispatch_pct":  (4,   0.85, 24,  "battery"),
    "battery8_dispatch_pct": (8,   0.85, 48,  "battery8"),
    "ldes_dispatch_pct":     (100, 0.50, 168, "ldes"),
    "h2_dispatch_pct":       (168, 0.35, 720, "h2"),
}

# Combined storage power cap: 115% of peak demand.
# Set per-ISO at profile load; used in generate_candidates.
_STORAGE_CAP_PCT = 999.0  # default (no cap); overwritten at runtime

# Fraction of LHS candidates per dimension pinned to floor (zero new build).
# With ~8 free dims and 5% per dim, ~40% of tail candidates have one dim
# pinned — enough to explore "skip resource X" without starving the search.
ZERO_INC_FRAC = 0.05

EF_BAND_THRESHOLDS = (
    10, 20, 30, 40, 50, 55, 60, 65, 70, 75,
    80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 99.5, 99.9,
)

# CFE waypoints for trajectory
CFE_WAYPOINTS = ((2030, 50), (2035, 70), (2040, 90), (2045, 95), (2050, 99.9))

# Pathway A free dimensions (indices into RESOURCE_ORDER)
PATHWAY_A_FREE = ["solar", "wind", "solar_batt4", "solar_batt8",
                  "wind_batt4", "wind_batt8"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class RunConfig:
    iso: str
    pathway: str  # "A" or "B"
    scenario_name: str = "base"
    demand_growth: str = "Medium"  # Low / Medium / High
    # Cost toggles (L / M / H unless noted)
    ren_cost: str = "M"       # Renewable gen (solar, wind)
    firm_cost: str = "M"      # Firm gen (nuclear new-build)
    batt_cost: str = "M"      # Battery storage
    ldes_cost: str = "M"      # LDES storage
    fuel_cost: str = "M"      # Fossil fuel (gas)
    tx_level: str = "M"       # Transmission adders
    ccs_cost: str = "M"       # CCS
    geo_cost: str = "M"       # Geothermal (grouped with firm/CCS)
    q45: str = "1"            # 45Q tax credit: "1" = on, "0" = off
    # Solver params
    n_beams: int = 5
    stage1_samples: int = 0   # 0 = use all from parquets
    stage2_samples: int = 5000
    cost_mode: int = 1        # 1 = incremental clean only, 2 = clean minus gas savings

    @property
    def dim_key(self) -> str:
        return (f"ren_{self.ren_cost}__firm_{self.firm_cost}__"
                f"batt_{self.batt_cost}__ldes_{self.ldes_cost}__"
                f"fuel_{self.fuel_cost}__tx_{self.tx_level}__"
                f"ccs_{self.ccs_cost}__geo_{self.geo_cost}__"
                f"q45_{self.q45}__dg_{self.demand_growth}")

    @property
    def free_resources(self) -> list[str]:
        """Resources the optimizer can adjust (not frozen)."""
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
        """Total sampling dimensions = free resources + 4 storage."""
        return len(self.free_resources) + 4

    @property
    def scaled_samples(self) -> int:
        """Stage 2 samples scaled by dimensionality.

        Base sample count targets 10 dimensions (Pathway A).
        Scales by (n_dims/10)^1.5 for higher-dimensional pathways.
        """
        return int(self.stage2_samples * (self.n_dims / 10.0) ** 1.5)

    @property
    def scaled_beams(self) -> int:
        """N beams scaled for pathway — Pathway B gets +2 for nuclear/offshore archetypes."""
        extra = 2 if self.pathway == "B" else 0
        return self.n_beams + extra


# Three named scenarios covering the spread of cost assumptions.
# Gas (fuel) always Medium. 45Q always on. Geothermal grouped with firm/CCS.
COST_SCENARIOS = {
    "base": {
        "ren_cost": "M", "firm_cost": "M", "batt_cost": "M", "ldes_cost": "M",
        "fuel_cost": "M", "tx_level": "M", "ccs_cost": "M", "geo_cost": "M",
        "q45": "1",
    },
    "firm_high_vre_low": {
        # Firm power expensive, VRE cheap — stress-tests pathway A advantage
        "ren_cost": "L", "firm_cost": "H", "batt_cost": "L", "ldes_cost": "L",
        "fuel_cost": "M", "tx_level": "L", "ccs_cost": "H", "geo_cost": "H",
        "q45": "1",
    },
    "firm_low_vre_high": {
        # Firm power cheap, VRE expensive — stress-tests pathway B advantage
        "ren_cost": "H", "firm_cost": "L", "batt_cost": "H", "ldes_cost": "H",
        "fuel_cost": "M", "tx_level": "H", "ccs_cost": "L", "geo_cost": "L",
        "q45": "1",
    },
}

LEVEL_NAME = {"L": "Low", "M": "Medium", "H": "High"}


# ---------------------------------------------------------------------------
# Profile loading (reused from step2_3a_regenerate_peakclean.py)
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
# Enhanced Numba kernel — computes hourly_match_score + resid + curtailment
# ---------------------------------------------------------------------------
@njit(cache=True)
def _inline_storage_stage(surplus, gap, total_dispatch,
                          capacity, power_rating, efficiency, window_hours):
    """Windowed charge/discharge for one storage tier. Returns leaked SOC."""
    if capacity <= 0.0:
        return nb_f32(0.0)
    leaked = nb_f32(0.0)
    n_windows = (H + window_hours - 1) // window_hours
    for w in range(n_windows):
        ws = w * window_hours
        we = min(ws + window_hours, H)
        soc = 0.0
        for h in range(ws, we):  # charge pass
            s = surplus[h]
            if s > 0.0 and soc < capacity:
                charge = min(s, power_rating, capacity - soc)
                soc += charge
                surplus[h] -= charge
        for h in range(ws, we):  # discharge pass
            g = gap[h]
            if g > 0.0 and soc > 0.0:
                discharge = min(g, power_rating, soc * efficiency)
                total_dispatch[h] += discharge  # v3.8 FIX: was missing
                soc -= discharge / efficiency
                gap[h] -= discharge
        leaked += nb_f32(soc)
    return leaked


@njit(parallel=True, cache=True)
def _score_mixes(scores, resid, curtail, W, P32, dm32, dn32,
                 batt4, batt8, ldes, h2, compute_idx):
    """Fused kernel: hourly_match_score + resid_norm_p9997 + curtailment.

    scores[i] = sum(min(supply_h, demand_h)) / sum(demand_h) * 100
              = sum(min(supply_h, dn32[h])) * 100  (since dn32 sums to 1.0)
    resid[i]  = 3rd-largest residual gap against RA-adjusted demand (dm32)
    curtail[i] = total surplus above raw demand (dn32)
    """
    NR = W.shape[1]
    n_compute = compute_idx.shape[0]

    for idx in prange(n_compute):
        i = compute_idx[idx]
        b4 = float(batt4[i])
        b8 = float(batt8[i])
        ld = float(ldes[i])
        h2_val = float(h2[i])
        has_stor = (b4 > 0.0) or (b8 > 0.0) or (ld > 0.0) or (h2_val > 0.0)

        if not has_stor:
            # Fast path: no storage
            t0 = nb_f32(-1.0); t1 = nb_f32(-1.0); t2 = nb_f32(-1.0)
            curt_sum = nb_f32(0.0)
            matched_sum = nb_f32(0.0)
            for h in range(H):
                s = nb_f32(0.0)
                for r in range(NR):
                    s += W[i, r] * P32[r, h]
                # CFE: matched MWh
                m = min(s, dn32[h])
                matched_sum += m
                # Residual gap against RA demand
                gap_val = dm32[h] - s
                if gap_val < nb_f32(0.0):
                    gap_val = nb_f32(0.0)
                if gap_val > t0:
                    t2 = t1; t1 = t0; t0 = gap_val
                elif gap_val > t1:
                    t2 = t1; t1 = gap_val
                elif gap_val > t2:
                    t2 = gap_val
                # Curtailment
                surplus_val = s - dn32[h]
                if surplus_val > nb_f32(0.0):
                    curt_sum += surplus_val
            scores[i] = matched_sum * nb_f32(100.0)
            resid[i] = t2
            curtail[i] = curt_sum
        else:
            # Storage path: float64 + 4-stage dispatch
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

            # 4-stage sequential dispatch
            b4_c = b4 * 0.01
            leaked_b4 = _inline_storage_stage(surplus, gap_arr, total_dispatch,
                                  b4_c, b4_c / 4.0, 0.85, 24)
            b8_c = b8 * 0.01
            leaked_b8 = _inline_storage_stage(surplus, gap_arr, total_dispatch,
                                  b8_c, b8_c / 8.0, 0.85, 48)
            ld_c = ld * 0.01
            leaked_ld = _inline_storage_stage(surplus, gap_arr, total_dispatch,
                                  ld_c, ld_c / 100.0, 0.50, 168)
            h2_c = h2_val * 0.01
            leaked_h2 = _inline_storage_stage(surplus, gap_arr, total_dispatch,
                                  h2_c, h2_c / 168.0, 0.35, 720)

            # CFE score
            matched_sum = 0.0
            for h in range(H):
                eff = supply[h] + total_dispatch[h]
                matched_sum += min(eff, float(dn32[h]))

            # Curtailment (post-storage surplus + leaked SOC)
            curt_sum = leaked_b4 + leaked_b8 + leaked_ld + leaked_h2
            for h in range(H):
                curt_sum += surplus[h]

            # Residual gap against RA demand (post-storage)
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


def score_candidates(W: np.ndarray, P32: np.ndarray, dm32: np.ndarray,
                     dn32: np.ndarray, batt4: np.ndarray, batt8: np.ndarray,
                     ldes: np.ndarray, h2: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score N candidate mixes. Returns (scores, resid, curtail) arrays."""
    n = W.shape[0]
    scores = np.full(n, -1.0, dtype=np.float32)
    resid = np.full(n, np.inf, dtype=np.float32)
    curtail = np.zeros(n, dtype=np.float32)
    idx = np.arange(n, dtype=np.int64)
    _score_mixes(scores, resid, curtail, W, P32, dm32, dn32,
                 batt4, batt8, ldes, h2, idx)
    return scores, resid, curtail


# ---------------------------------------------------------------------------
# Wright's Law cost model
# ---------------------------------------------------------------------------
def wrights_law_cost(foak: float, noak: float, year: int,
                     t_start: int, t_noak: int) -> float:
    """Concave learning curve: exponential decay from FOAK toward NOAK."""
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
    """Base LCOE before TX adder. Uses per-dimension cost levels."""
    ren_name = LEVEL_NAME[cfg.ren_cost]

    if resource == "solar":
        return pc.LCOE_TABLES["solar"][ren_name][iso]
    if resource == "wind":
        return pc.LCOE_TABLES["wind"][ren_name][iso]
    if resource.startswith("solar_batt"):
        # Renewable component at ren level + battery adder at batt level
        ren_lcoe = pc.LCOE_TABLES["solar"][ren_name][iso]
        batt_adder = {"L": 20.0, "M": 30.0, "H": 45.0}[cfg.batt_cost]
        return ren_lcoe + batt_adder
    if resource.startswith("wind_batt"):
        ren_lcoe = pc.LCOE_TABLES["wind"][ren_name][iso]
        batt_adder = {"L": 17.0, "M": 25.0, "H": 38.0}[cfg.batt_cost]
        return ren_lcoe + batt_adder
    if resource == "hydro":
        return 50.0

    # Pathway B learning-curve resources
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
        fs, ny = pc.LEARNING_PARAMS[tech][cfg.ren_cost]  # v3.8 FIX: was "M"
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


# ---------------------------------------------------------------------------
# Storage cost model — Wright's Law learning curves (v3.8)
# ---------------------------------------------------------------------------
def _storage_year_lcoe(lcoe_key: str, cost_name: str, iso: str,
                       year: int, cfg: RunConfig) -> float:
    """Year-adjusted storage annualized cost with Wright's Law decline.

    Battery 4hr/8hr: LCOE_TABLES (2025 start) -> NOAK_BATTERY (terminal floor).
    LDES: FOAK_LDES (pre-commercial) -> LCOE_TABLES (NOAK target).
    H2: FOAK_H2 (pre-commercial) -> LCOE_TABLES (NOAK target).
    """
    current = float(pc.LCOE_TABLES[lcoe_key][cost_name][iso])

    if lcoe_key == "battery":
        foak = current
        noak = float(pc.NOAK_BATTERY[cost_name][iso])
        lp_key = "bat4"
        level = cfg.batt_cost
    elif lcoe_key == "battery8":
        foak = current
        noak = float(pc.NOAK_BATTERY8[cost_name][iso])
        lp_key = "bat8"
        level = cfg.batt_cost
    elif lcoe_key == "ldes":
        foak = float(pc.FOAK_LDES[iso])
        noak = current  # LCOE_TABLES IS the NOAK target for LDES
        lp_key = "ldes"
        level = cfg.ldes_cost
    elif lcoe_key == "h2":
        foak = float(pc.FOAK_H2[iso])
        noak = current  # LCOE_TABLES IS the NOAK target for H2
        lp_key = "h2"
        level = cfg.ldes_cost
    else:
        return current

    fs, ny = pc.LEARNING_PARAMS[lp_key][level]
    return wrights_law_cost(foak, noak, year, fs, ny)


def storage_net_cost(iso: str, storage_key: str, cfg: RunConfig,
                     year: int = BASE_YEAR) -> float:
    """Net cost per unit for standalone storage: year-adjusted LCOE - revenue credit.

    Returns value in LCOE_TABLES units (annualized cost per % of annual demand).
    Callers must use: (pct/100) x net_cost x demand_MWh for $/yr.
    """
    lcoe_key = STORAGE_PARAMS[storage_key][3]
    if lcoe_key in ("battery", "battery8"):
        cost_name = LEVEL_NAME[cfg.batt_cost]
    else:  # ldes, h2
        cost_name = LEVEL_NAME[cfg.ldes_cost]
    lcoe = _storage_year_lcoe(lcoe_key, cost_name, iso, year, cfg)
    credits = getattr(pc, "STORAGE_REVENUE_CREDITS", {})
    credit = credits.get(lcoe_key, {}).get(iso, 0.0)
    return max(0.0, float(lcoe) - float(credit))


# ---------------------------------------------------------------------------
# Demand model
# ---------------------------------------------------------------------------
def demand_twh_vec(iso: str, level: str = "Medium") -> np.ndarray:
    """(N_YEARS,) demand in TWh per year."""
    base = float(pc.REGIONAL_DEMAND_TWH[iso])
    rate = pc.DEMAND_GROWTH_RATES[iso][level]
    return np.array([base * (1 + rate) ** (y - BASE_YEAR) for y in YEARS])


def peak_demand_vec(iso: str, level: str = "Medium") -> np.ndarray:
    """(N_YEARS,) peak demand in MW per year."""
    base = float(pc.PEAK_DEMAND_MW[iso])
    rate = pc.DEMAND_GROWTH_RATES[iso][level]
    return np.array([base * (1 + rate) ** (y - BASE_YEAR) for y in YEARS])


def cfe_target(year: int, base_pct: float) -> float:
    """CFE target for a given year via linear interpolation of waypoints."""
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
    """(N_YEARS,) CFE target percentages."""
    base = sum(pc.GRID_MIX_SHARES[iso].values())
    return np.array([cfe_target(y, base) for y in YEARS])


# ---------------------------------------------------------------------------
# Clean firm tranche decomposition (Pathway B only)
# ---------------------------------------------------------------------------
def decompose_clean_firm(cf_twh: float, iso: str, year: int,
                         cfg: RunConfig) -> dict:
    """Break clean_firm TWh into tranches with per-tranche LCOE."""
    base_twh = (pc.GRID_MIX_SHARES[iso].get("clean_firm", 0.0) / 100.0
                * float(pc.REGIONAL_DEMAND_TWH[iso]))
    existing = min(cf_twh, base_twh)
    rem = cf_twh - existing

    uc = float(pc.UPRATE_CAP_TWH[iso])
    uprate = min(rem, uc)
    uprate_lcoe = float(pc.UPRATE_LCOE[cfg.firm_cost])
    rem -= uprate

    # Geothermal (CAISO only)
    gc = float(pc.GEOTHERMAL_CAP_TWH) if iso == "CAISO" else 0.0
    geo = min(rem, gc) if gc > 0 else 0.0
    geo_lcoe = _base_lcoe(iso, "geothermal", year, cfg) + _tx_adder(
        iso, "clean_firm", cfg.tx_level)
    rem -= geo

    # Nuclear vs CCS — cheaper goes first
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
        "existing_twh": existing, "existing_lcoe": 0.0,  # sunk cost
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
    """MW of new gas needed to cover worst-hour residual gap.

    resid_p9997 is in normalized demand space (dn sums to 1.0).
    Convert to MW: gap_fraction x annual_demand_MWh = gap_MW per hour.
    dm32 already includes RA margin, so no further adjustment needed.
    """
    worst_hour_gap_mw = resid_p9997 * demand_twh * 1e6
    return max(0.0, worst_hour_gap_mw / gaf - existing_gas_mw)


def gas_annual_cost(new_gas_mw: float, iso: str, cfg: RunConfig) -> float:
    """Annual cost of new gas fleet (capex annualized + FOM + fuel)."""
    if new_gas_mw <= 0:
        return 0.0
    capex_ann = float(pc.NEW_CCGT_COST_KW_YR[iso]) * new_gas_mw * 1000
    fom = float(pc.NEW_CCGT_FOM_KW_YR[iso]) * new_gas_mw * 1000
    fuel_level = LEVEL_NAME[cfg.fuel_cost]
    # v3.8 FIX: simplified — MW * CF * hours * $/MWh = $/yr
    fuel_cost = (new_gas_mw * GAS_CF * H
                 * (float(pc.WHOLESALE_PRICES[iso])
                    + float(pc.FUEL_ADJUSTMENTS[iso][fuel_level])))
    return capex_ann + fom + fuel_cost


def gas_stranding_shadow(new_gas_mw: float, current_peak_gas: float,
                         year: int, iso: str) -> float:
    """Shadow cost penalizing increases in peak gas capacity (v3.8).

    Charges STRANDING_SHADOW_WEIGHT x annualized capex for each MW of new peak,
    scaled by years remaining (early builds penalized more than late builds).
    Returns 0 if new_gas_mw does not exceed current peak.
    """
    if STRANDING_SHADOW_WEIGHT <= 0.0:
        return 0.0
    marginal_peak = max(0.0, new_gas_mw - current_peak_gas)
    if marginal_peak <= 0.0:
        return 0.0
    capex_kw_yr = float(pc.NEW_CCGT_COST_KW_YR[iso])
    years_remaining_frac = max(0.0, (END_YEAR - year)) / (END_YEAR - BASE_YEAR)
    return marginal_peak * capex_kw_yr * 1000 * years_remaining_frac * STRANDING_SHADOW_WEIGHT


# ---------------------------------------------------------------------------
# Marginal yield computation
# ---------------------------------------------------------------------------
def compute_marginal_yields(
    floor_pcts: np.ndarray,  # (N_RESOURCES,) current floor in %
    floor_storage: np.ndarray,  # (4,) current storage dispatch %
    free_res_idx: list[int],
    P32: np.ndarray, dm32: np.ndarray, dn32: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute marginal CFE yield per +1pp of each free dimension.

    Returns (res_yields, stor_yields, floor_cfe).
    res_yields[i] = dCFE/d(1pp) for free_res_idx[i].
    stor_yields[j] = dCFE/d(1pp) for STORAGE_COLS[j].
    """
    n_free = len(free_res_idx)

    # Build floor weight vector
    W_floor = np.zeros((1, N_RESOURCES), dtype=np.float32)
    W_floor[0] = (floor_pcts * 0.01).astype(np.float32)

    s_floor = floor_storage.astype(np.float32)
    floor_scores, _, _ = score_candidates(
        W_floor, P32, dm32, dn32,
        s_floor[0:1], s_floor[1:2], s_floor[2:3], s_floor[3:4])
    floor_cfe = float(floor_scores[0])

    # Perturb each free resource by +1pp
    res_yields = np.zeros(n_free, dtype=np.float64)
    for k, ri in enumerate(free_res_idx):
        W_pert = W_floor.copy()
        W_pert[0, ri] += np.float32(0.01)  # +1pp
        sc, _, _ = score_candidates(
            W_pert, P32, dm32, dn32,
            s_floor[0:1], s_floor[1:2], s_floor[2:3], s_floor[3:4])
        res_yields[k] = float(sc[0]) - floor_cfe

    # Perturb each storage type by +1pp
    stor_yields = np.zeros(4, dtype=np.float64)
    for j in range(4):
        s_pert = s_floor.copy()
        s_pert[j] += np.float32(1.0)  # +1pp of storage dispatch capacity
        sc, _, _ = score_candidates(
            W_floor, P32, dm32, dn32,
            s_pert[0:1], s_pert[1:2], s_pert[2:3], s_pert[3:4])
        stor_yields[j] = float(sc[0]) - floor_cfe

    return res_yields, stor_yields, floor_cfe


# ---------------------------------------------------------------------------
# LHS candidate generation
# ---------------------------------------------------------------------------
def generate_candidates(
    floor_pcts: np.ndarray,   # (N_RESOURCES,) floor in %
    floor_storage: np.ndarray,  # (4,) floor storage dispatch %
    target_cfe: float,
    floor_cfe: float,
    res_yields: np.ndarray,    # (n_free,) marginal yields for free resources
    stor_yields: np.ndarray,   # (4,) marginal yields for storage
    free_res_idx: list[int],
    cfg: RunConfig,
    n_samples: int,
    beam_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray]:
    """Generate LHS candidates. Returns (W, batt4, batt8, ldes, h2, raw_pcts).

    raw_pcts is (n_samples, n_free_res + 4) for cost accounting.
    """
    n_free = len(free_res_idx)
    increment = max(0.1, target_cfe - floor_cfe)

    # Compute per-dimension ceilings
    res_ceilings = np.zeros(n_free, dtype=np.float64)
    for k in range(n_free):
        yld = res_yields[k]
        if yld > 0.001:
            res_ceilings[k] = floor_pcts[free_res_idx[k]] + increment / yld * SAFETY_FACTOR
        else:
            res_ceilings[k] = floor_pcts[free_res_idx[k]] + 1.0  # minimal headroom

    stor_ceilings = np.zeros(4, dtype=np.float64)
    for j in range(4):
        yld = stor_yields[j]
        if yld > 0.001:
            stor_ceilings[j] = floor_storage[j] + increment / yld * SAFETY_FACTOR
        else:
            stor_ceilings[j] = floor_storage[j] + 1.0

    # Hard caps
    MAX_RESOURCE_PCT = 200.0
    for k in range(n_free):
        res_ceilings[k] = min(res_ceilings[k], MAX_RESOURCE_PCT)
    for j in range(4):
        stor_ceilings[j] = min(stor_ceilings[j], _STORAGE_CAP_PCT)

    # Apply hard caps (CCS, geo, uprate for Pathway B)
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

    # Build bounds
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

    # Latin Hypercube sampling
    seed_str = (f"{cfg.iso}_{cfg.pathway}_{target_cfe:.1f}"
                f"_{cfg.demand_growth}_{beam_idx}")
    seed_val = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)
    sampler = LatinHypercube(d=n_dims, seed=seed_val)
    unit_samples = sampler.random(n=n_samples)

    # Scale to [floor, ceiling]
    raw = floors[None, :] + unit_samples * (ceilings - floors)[None, :]

    # Zero-incremental exploration
    per_dim = max(1, int(n_samples * ZERO_INC_FRAC))
    pin_start = max(0, n_samples - per_dim * n_dims)
    for d in range(n_dims):
        row_lo = pin_start + d * per_dim
        row_hi = min(row_lo + per_dim, n_samples)
        if row_lo < n_samples:
            raw[row_lo:row_hi, d] = floors[d]

    # Zero-storage pool (10% of candidates)
    zs_count = max(1, int(n_samples * 0.10))
    zs_start = max(0, n_samples - zs_count)
    for j in range(4):
        raw[zs_start:, n_free + j] = floors[n_free + j]

    # Build output arrays
    W = np.zeros((n_samples, N_RESOURCES), dtype=np.float32)
    for ri in range(N_RESOURCES):
        W[:, ri] = np.float32(floor_pcts[ri] * 0.01)
    for k, ri in enumerate(free_res_idx):
        W[:, ri] = (raw[:, k] * 0.01).astype(np.float32)

    batt4 = raw[:, n_free + 0].astype(np.float32)
    batt8 = raw[:, n_free + 1].astype(np.float32)
    ldes_arr = raw[:, n_free + 2].astype(np.float32)
    h2_arr = raw[:, n_free + 3].astype(np.float32)

    # Storage cap: 115% of peak demand
    cap = np.float32(_STORAGE_CAP_PCT)
    stor_total = batt4 + batt8 + ldes_arr + h2_arr
    over = stor_total > cap
    if over.any():
        scale = np.where(over, cap / np.maximum(stor_total, 1e-6),
                         1.0).astype(np.float32)
        batt4 *= scale; batt8 *= scale
        ldes_arr *= scale; h2_arr *= scale
        raw[:, n_free + 0] = batt4; raw[:, n_free + 1] = batt8
        raw[:, n_free + 2] = ldes_arr; raw[:, n_free + 3] = h2_arr

    return W, batt4, batt8, ldes_arr, h2_arr, raw


# ---------------------------------------------------------------------------
# Stage 1 — Archetype selection from EF parquets
# ---------------------------------------------------------------------------
def load_ef_band(iso: str, threshold: float) -> Optional[pa.Table]:
    """Load EF parquet(s) for a given ISO and threshold band."""
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
    """Classify a mix into an archetype by dominant resource family."""
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
    """Load EF parquets for first threshold above baseline, select N seed mixes."""
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

    costs = np.zeros(len(valid_idx), dtype=np.float64)
    base_dem = float(pc.REGIONAL_DEMAND_TWH[iso])
    for ki, vi in enumerate(valid_idx):
        c = 0.0
        for ri, res in enumerate(RESOURCE_ORDER):
            pct = float(W[vi, ri]) * 100.0
            if pct > 0.001:
                c += pct / 100.0 * base_dem * get_resource_lcoe(
                    iso, res, BASE_YEAR, cfg) * 1e6
        for j, sc in enumerate(STORAGE_COLS):
            sv = [batt4, batt8, ldes_arr, h2_arr][j][vi]
            if sv > 0:
                c += float(sv) / 100.0 * storage_net_cost(
                    iso, sc, cfg, year=BASE_YEAR) * base_dem * 1e6
        costs[ki] = c

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
                  P32: np.ndarray, dm32: np.ndarray, dn32: np.ndarray,
                  beam_idx: int = 0,
                  ) -> dict:
    """Build a full 2025-2050 pathway from a seed mix.

    Returns a result dict with threshold snapshots.
    """
    iso = cfg.iso
    demand_vec = demand_twh_vec(iso, cfg.demand_growth)
    cfe_targets = cfe_target_vec(iso)
    existing_gas = float(pc.EXISTING_GAS_CAPACITY_MW[iso])
    gaf = float(pc.GAS_AVAILABILITY_FACTOR[iso])
    free_idx = cfg.free_indices

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

    for yi, year in enumerate(YEARS):
        dem_twh = demand_vec[yi]
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

        if yi < N_YEARS - 1:
            cfe_ceiling = min(target + 15.0, 100.0)
        else:
            cfe_ceiling = 100.0

        res_yields, stor_yields, floor_cfe = compute_marginal_yields(
            floor_pcts, floor_storage, free_idx, P32, dm32, dn32)

        if floor_cfe >= target:
            winner_pcts = floor_pcts.copy()
            winner_storage = floor_storage.copy()
            winner_cfe = floor_cfe
            winner_resid = _score_single(floor_pcts, floor_storage,
                                          P32, dm32, dn32)[1]
        else:
            winner_pcts, winner_storage, winner_cfe, winner_resid = (
                _find_winner(floor_pcts, floor_storage, target, cfe_ceiling,
                             floor_cfe, res_yields, stor_yields,
                             free_idx, cfg, P32, dm32, dn32,
                             year, dem_twh,
                             beam_idx=beam_idx,
                             existing_gas_mw=existing_gas,
                             gaf=gaf,
                             prev_gas_cost=prev_gas_cost,
                             current_peak_gas=peak_gas_mw))

            if winner_pcts is None:
                print(f"[solve] {iso} beam={seed['archetype']}: "
                      f"INFEASIBLE at year {year}, target={target:.1f}%")
                break

        # Record vintage costs
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
                    lcoe = get_resource_lcoe(iso, res, year, cfg)
                    vintage_ledger.append((year, res, delta_twh, lcoe))
                    year_incremental_cost += delta_twh * lcoe * 1e6

        for j, sc in enumerate(STORAGE_COLS):
            delta_stor = winner_storage[j] - floor_storage[j]
            if delta_stor > RATCHET_TOL_PCT:
                net_lcoe = storage_net_cost(iso, sc, cfg, year=year)
                annual_cost = delta_stor / 100.0 * net_lcoe * dem_twh * 1e6
                vintage_ledger.append((year, sc, annual_cost, delta_stor))
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
                total_vintage_cost += qty
            else:
                total_vintage_cost += qty * unit_cost * 1e6
        total_annual = total_vintage_cost + g_cost

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

        # Check threshold crossings
        for t in EF_BAND_THRESHOLDS:
            if t not in thresholds_crossed and winner_cfe >= t:
                thresholds_crossed.add(t)
                _, _, curt = _score_single(winner_pcts, winner_storage,
                                           P32, dm32, dn32)
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
                    "curtailment_total": round(float(curt), 4),
                    "demand_twh": round(dem_twh, 1),
                }
                threshold_snapshots.append(snapshot)

        if year % 5 == 0 or winner_cfe >= 99.0:
            print(f"  {year}: CFE={winner_cfe:.1f}% target={target:.1f}% "
                  f"gas={g_mw:,.0f}MW cost=${total_annual/1e9:.2f}B")

    return {
        "schema_version": "3.8",
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
            "stranding_shadow_weight": STRANDING_SHADOW_WEIGHT,
        },
        "beam_archetype": seed["archetype"],
        "beam_index": beam_idx,
        "peak_gas_mw": round(peak_gas_mw, 1),
        "peak_gas_year": peak_gas_year,
        "n_thresholds_achieved": len(threshold_snapshots),
        "threshold_snapshots": threshold_snapshots,
        "vintage_ledger_summary": {
            "n_vintages": len(vintage_ledger),
            "total_gen_twh_built": round(sum(
                t for _, r, t, _ in vintage_ledger if r not in STORAGE_COLS), 2),
            "total_storage_pct_built": round(sum(
                dp for _, r, _, dp in vintage_ledger if r in STORAGE_COLS), 1),
        },
    }


def _find_winner(floor_pcts, floor_storage, target, ceiling,
                 floor_cfe, res_yields, stor_yields,
                 free_idx, cfg, P32, dm32, dn32,
                 year, dem_twh,
                 beam_idx=0, existing_gas_mw=0.0, gaf=1.0,
                 prev_gas_cost=0.0, current_peak_gas=0.0):
    """Try to find cheapest candidate in [target, ceiling) CFE band.

    Cascade: 1x -> 2x -> 4x samples, then infeasible.
    Mode 2: ranks by (incremental clean cost - gas savings).
    Gas stranding shadow price applied to both modes (v3.8).
    """
    for multiplier in [1, 2, 4]:
        n = cfg.scaled_samples * multiplier
        W, b4, b8, ld, h2, raw = generate_candidates(
            floor_pcts, floor_storage, target, floor_cfe,
            res_yields, stor_yields, free_idx, cfg, n,
            beam_idx=beam_idx)

        scores, resid, curtail = score_candidates(
            W, P32, dm32, dn32, b4, b8, ld, h2)

        mask = (scores >= target) & (scores < ceiling)
        valid = np.where(mask)[0]

        if len(valid) == 0:
            if multiplier < 4:
                print(f"    cascade {multiplier}x->{multiplier*2}x: "
                      f"0 hits in [{target:.1f}, {ceiling:.1f})")
            continue

        # Compute incremental cost for each valid candidate
        costs = np.zeros(len(valid), dtype=np.float64)
        for ki, vi in enumerate(valid):
            c = 0.0
            for k, ri in enumerate(free_idx):
                delta = float(W[vi, ri]) * 100.0 - floor_pcts[ri]
                if delta > 0:
                    c += delta / 100.0 * dem_twh * get_resource_lcoe(
                        cfg.iso, RESOURCE_ORDER[ri], year, cfg) * 1e6
            for j, sc in enumerate(STORAGE_COLS):
                sv = [b4, b8, ld, h2][j][vi]
                delta = float(sv) - floor_storage[j]
                if delta > 0:
                    c += (delta / 100.0 * storage_net_cost(
                        cfg.iso, sc, cfg, year=year) * dem_twh * 1e6)
            costs[ki] = c

        # Gas stranding shadow price (v3.8) — applied to all modes
        if STRANDING_SHADOW_WEIGHT > 0.0:
            for ki, vi in enumerate(valid):
                g_mw = gas_need_mw(float(resid[vi]), dem_twh,
                                   existing_gas_mw, gaf)
                shadow = gas_stranding_shadow(g_mw, current_peak_gas,
                                             year, cfg.iso)
                costs[ki] += shadow

        # Mode 2: adjust costs by gas savings vs. current gas fleet
        if cfg.cost_mode == 2 and prev_gas_cost > 0:
            for ki, vi in enumerate(valid):
                g_mw = gas_need_mw(float(resid[vi]), dem_twh,
                                   existing_gas_mw, gaf)
                g_cost = gas_annual_cost(g_mw, cfg.iso, cfg)
                gas_savings = prev_gas_cost - g_cost
                costs[ki] -= gas_savings

        best = valid[np.argmin(costs)]
        winner_pcts = np.array([float(W[best, ri]) * 100.0
                                for ri in range(N_RESOURCES)], dtype=np.float64)
        for ri in range(N_RESOURCES):
            if ri not in free_idx:
                winner_pcts[ri] = floor_pcts[ri]
        winner_storage = np.array([
            float([b4, b8, ld, h2][j][best]) for j in range(4)
        ], dtype=np.float64)

        return winner_pcts, winner_storage, float(scores[best]), float(resid[best])

    return None, None, None, None


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
    scores = np.full(n, -1.0, dtype=np.float32)
    resid = np.full(n, np.inf, dtype=np.float32)
    curtail = np.zeros(n, dtype=np.float32)
    batt = np.array([0.0, 10.0], dtype=np.float32)
    zero = np.zeros(n, dtype=np.float32)
    idx = np.arange(n, dtype=np.int64)
    _score_mixes(scores, resid, curtail, W, P, dm, dn,
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
    """Build RunConfig list from scenario names x demand levels x cost modes."""
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
                    choices=["Low", "Medium", "High"],
                    help="Single demand level (ignored if --batch-demand)")
    ap.add_argument("--scenario", type=str, default="base",
                    choices=list(COST_SCENARIOS.keys()),
                    help="Single cost scenario (ignored if --batch)")
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
    print(f"[adaptive] Step 2.3 Reliability Tax Adaptive v3.8")
    print(f"[adaptive] ISO={iso}  Pathway={args.pathway}")
    print(f"[adaptive] Scenarios: {scenarios}")
    print(f"[adaptive] Demand levels: {demand_levels}")
    print(f"[adaptive] Stranding shadow weight: {STRANDING_SHADOW_WEIGHT}")
    print(f"{'='*70}\n")

    warmup_jit()

    print(f"[adaptive] Loading profiles for {iso}...")
    P = build_profile_matrix(iso)
    dn = _load_demand_norm(iso)
    dm = dn * (1.0 + RA_MARGIN)
    P32 = P.astype(np.float32)
    dm32 = dm.astype(np.float32)
    dn32 = dn.astype(np.float32)
    global _STORAGE_CAP_PCT
    _STORAGE_CAP_PCT = float(dn.max() * H * 100.0 * 1.15)
    print(f"[adaptive] Profiles loaded. Demand TWh={float(pc.REGIONAL_DEMAND_TWH[iso]):.1f}"
          f"  storage_cap={_STORAGE_CAP_PCT:.1f}% of avg demand"
          f" (peak/avg={dn.max()*H:.2f})")

    configs = build_configs(iso, args.pathway, scenarios, demand_levels,
                            args.n_beams, args.stage1_samples, args.stage2_samples)
    print(f"[adaptive] {len(configs)} runs queued "
          f"({len(scenarios)} scenarios x {len(demand_levels)} demand levels "
          f"x 2 cost modes)")
    sample_cfg = configs[0]
    print(f"[adaptive] Dimensions: {sample_cfg.n_dims} | "
          f"Samples: {sample_cfg.stage2_samples}->{sample_cfg.scaled_samples} "
          f"(x{sample_cfg.scaled_samples/sample_cfg.stage2_samples:.2f}) | "
          f"Beams: {sample_cfg.n_beams}->{sample_cfg.scaled_beams}")

    base_cfg = configs[0]
    seeds = stage1_select(iso, base_cfg, P32, dm32, dn32)
    if not seeds:
        print(f"[adaptive] No seeds found. Exiting.")
        return 1

    t_total = time.time()
    for ci, cfg in enumerate(configs):
        print(f"\n{'_'*60}")
        print(f"[adaptive] Config {ci+1}/{len(configs)}: "
              f"{cfg.scenario_name} | {cfg.demand_growth} | mode={cfg.cost_mode}")
        print(f"[adaptive] dim_key: {cfg.dim_key}")
        print(f"{'_'*60}")

        for bi, seed in enumerate(seeds):
            print(f"\n[solve] Beam {bi+1}/{len(seeds)}: {seed['archetype']} "
                  f"(seed CFE={seed['cfe']:.1f}%)")
            t1 = time.time()
            result = solve_pathway(seed, cfg, P32, dm32, dn32, beam_idx=bi)
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
