#!/usr/bin/env python3
"""
step2_3_reliability_tax_adaptive.py
On-the-fly mix generation with marginal-yield-bounded Latin Hypercube sampling.

Architecture:
  Stage 1: Load Step 2.1 EF parquets for first threshold band above baseline.
           Cluster into archetypes, pick N diverse low-cost seeds.
  Stage 2: For each seed, build year-by-year pathway to 99.9% CFE via on-the-fly
           LHS generation. Write results at every threshold crossed.

Four pathways:
  A — VRE + hybrids + storage only. No offshore, geo, CCS, or clean_firm expansion.
  B — Full P3: all resources incl nuclear, CCS, offshore, geothermal. NOAK 2035.
  C — Same resources as B. Delayed deployment: NOAK 2040 for advanced clean tech.
  D — Same resources as B. Further-delayed deployment: NOAK 2045 for advanced clean tech.

Two cost modes:
  Mode 1 — Incremental clean cost only.
  Mode 2 — Incremental clean cost minus gas savings.

See step2_3_rebuild_decisions.md for full methodology specification (25 decisions).

v4.4 changes:
  - FIX: Storage cap bug — _STORAGE_CAP_PCT (≈230% of demand) was conflating
    power-based cap (115% of peak demand) with energy-based storage dispatch_pct
    units (% of annual demand).  This allowed storage dispatch_pct values up to
    ~230% when physical maximums are 0.10%/0.15%/1.0% (STORAGE_MAX_V2).
    Beams that leaned on storage saw trillion-dollar costs.
    Replaced with per-type physical caps from pipeline_config.STORAGE_MAX_V2.
    Combined cap = sum of per-type caps (1.25% of annual demand).
  - REMOVE: Runtime _STORAGE_CAP_PCT computation in main().  Per-type caps are
    constants derived from STORAGE_MAX_V2 — no runtime calculation needed.
  - ADD: _STORAGE_TYPE_CAPS tuple and _STORAGE_COMBINED_CAP constant.
  - ADD: Storage dispatch columns (stor_battery, stor_battery8, stor_ldes) in
    consolidate_to_parquet().  Previously only resource_mix_pct was flattened
    into the parquet; storage_dispatch_pct was silently dropped.

v4.3 changes:
  - FIX: Seeds no longer inflate 2025 CFE above grid baseline.
    Solver loop starts at MODEL_START_YEAR (2030); years 2025-2029 are
    not modeled.  Floor initialized directly from seed.
  - CHANGE: Per-ISO seed bands — ISOs with baseline >= 45% (CAISO, ERCOT, SPP)
    seed from EF band 60; others (MISO, NEISO, NYISO, PJM) from band 50.
  - CHANGE: Per-group CFE waypoints — seed=60 group uses (2030,60)→(2035,75)→(2040,90);
    seed=50 group keeps (2030,50)→(2035,70)→(2040,90).  Both converge at 90% by 2040.
  - ADD: SEED_BAND_BY_ISO, CFE_WAYPOINTS_50/60, MODEL_START_YEAR constants.
  - ADD: consolidate_to_parquet() — writes one parquet per ISO+pathway after
    all beams finish.  JSONs remain as intermediate debug output.

v4.2 changes:
  - ADD: Pathways C and D — identical resource availability to B, but with
    delayed NOAK years (2040, 2045) for nuclear, CCS, geothermal, offshore wind.
    Wright's Law k unchanged; per-year learning rate slows via longer window
    normalization — delayed deployment = higher costs at any given calendar year.
  - ADD: noak_pathway_key and offshore_noak_cap properties on RunConfig
  - CHANGE: All pathway-B-only checks now accept B/C/D ("firm pathways")
  - CHANGE: _base_lcoe uses cfg.noak_pathway_key instead of hardcoded "3"

v4.1 changes:
  - CHANGE: Wright's Law k from 5 to 3 (APR-1400 calibrated; Finding 8)
  - CHANGE: H2_MAX_FILL_PP from 10 to 5 (tighter peaker cap)
  - ADD: Second-order yield probing at +5pp for concavity detection (Finding 4)
  - ADD: Top-2 joint interaction probe for synergy ceiling boost (Finding 5)
  - CHANGE: compute_marginal_yields returns 5 values (raw + adjusted yields)

v4.0 changes (carried forward):
  - REFACTOR: H2 removed from storage kernel; now deterministic gap-filling peaker
  - ADD: Two-part H2 cost (capacity $/kW-yr + fuel $/MWh dispatched)
  - ADD: _compute_post_storage_gaps Numba kernel for per-candidate H2 sizing
  - ADD: h2_size_for_target binary search for minimum H2 capacity
  - ADD: Gas sized from post-H2 residual (H2 contributes to resource adequacy)
  - CHANGE: LHS drops 1 dim (H2 no longer sampled); 3 storage tiers only
  - CHANGE: _find_winner returns 5 values including h2_info dict

v3.8 changes (carried forward):
  - FIX: _inline_storage_stage now writes total_dispatch (was broken — storage invisible)
  - ADD: Storage learning curves (Wright's Law) for battery/LDES
  - ADD: Gas stranding shadow price in _find_winner
  - FIX: Offshore wind learning curve timing tracks cfg.ren_cost, NOAK clamped to 2035
  - FIX: gas_annual_cost fuel calc simplified

Usage:
    python scripts/step2_3_reliability_tax_adaptive.py --iso NYISO --pathway A
    python scripts/step2_3_reliability_tax_adaptive.py --iso ERCOT --pathway B --demand-growth High
    python scripts/step2_3_reliability_tax_adaptive.py --iso CAISO --pathway C --batch
    python scripts/step2_3_reliability_tax_adaptive.py --iso PJM --pathway D --scenario base
"""
from __future__ import annotations

import argparse
import io
import json
import multiprocessing
import os
import sys
import hashlib
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# Must precede numba import so spawned children inherit the method.
multiprocessing.set_start_method("spawn", force=True)

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
    from scipy.stats.qmc import Sobol
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

# Inject Pathway C/D NOAK overrides into pipeline_config's lookup table.
# C = delayed deployment (NOAK 2040), D = further-delayed (NOAK 2045).
# Uses the same Wright's Law k; per-year cost decline is slower because the
# normalized progress advances more slowly over the wider window.
pc.NOAK_YEAR_BY_PATHWAY["C"] = 2040
pc.NOAK_YEAR_BY_PATHWAY["D"] = 2045

# Convenience: set of pathways that include firm clean resources (nuclear, CCS, etc.)
FIRM_PATHWAYS = frozenset({"B", "C", "D"})

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
WRIGHTS_LAW_K = 3.0  # APR-1400 calibrated — ~95% of NOAK by end of window
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
    "ldes_dispatch_pct",
]
N_STORAGE = 3  # battery4, battery8, LDES (H2 is now a deterministic peaker)
STORAGE_PARAMS = {
    # name: (duration_hrs, efficiency, window_hrs, lcoe_key)
    "battery_dispatch_pct":  (4,   0.85, 24,  "battery"),
    "battery8_dispatch_pct": (8,   0.85, 48,  "battery8"),
    "ldes_dispatch_pct":     (100, 0.50, 168, "ldes"),
}

# ---------------------------------------------------------------------------
# Storage dispatch caps — per-type physical maximums (% of annual demand).
# v4.4 FIX: Replaces the broken _STORAGE_CAP_PCT (≈230%) which conflated
# power-based cap (115% of peak demand MW) with energy-based dispatch_pct
# units (% of annual demand MWh).  Physical caps from STORAGE_MAX_V2 are
# the research-informed 2050 upper bounds on installed storage energy capacity.
#
# battery:  0.10% of annual demand — CAISO: 224 GWh / 56 GW (4hr)
# battery8: 0.15% of annual demand — CAISO: 336 GWh / 42 GW (8hr)
# ldes:     1.00% of annual demand — CAISO: 2,240 GWh / 22.4 GW (100hr)
# ---------------------------------------------------------------------------
_STORAGE_TYPE_CAPS = (
    pc.STORAGE_MAX_V2['battery'],     # 0.10 (% of annual demand)
    pc.STORAGE_MAX_V2['battery8'],    # 0.15
    pc.STORAGE_MAX_V2['ldes'],        # 1.0
)
_STORAGE_COMBINED_CAP = sum(_STORAGE_TYPE_CAPS)  # 1.25

# H2 peaker: max CFE gap (pp) that H2 is allowed to close per year.
# Prevents accepting candidates far below target that lean on massive H2.
H2_MAX_FILL_PP = 5.0

# Fraction of LHS candidates per dimension pinned to floor (zero new build).
EF_BAND_THRESHOLDS = (
    10, 20, 30, 40, 50, 55, 60, 65, 70, 75,
    80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 99.5, 99.9,
)

# Per-ISO seed band: ISOs with baseline >= 45% seed from band 60,
# others from band 50.  Represents the 2030 starting point.
SEED_BAND_HIGH = frozenset({"CAISO", "ERCOT", "SPP"})  # baseline >= 45%
SEED_BAND_BY_ISO = {iso: 60 if iso in SEED_BAND_HIGH else 50
                    for iso in pc.ISOS}

# Per-group CFE waypoints — both converge to 90% by 2040.
# 2030s slope differs: seed=50 needs 4pp/yr, seed=60 needs 3pp/yr.
CFE_WAYPOINTS_50 = ((2030, 50), (2035, 70), (2040, 90), (2045, 95), (2050, 99.9))
CFE_WAYPOINTS_60 = ((2030, 60), (2035, 75), (2040, 90), (2045, 95), (2050, 99.9))

def _waypoints_for_iso(iso: str) -> tuple:
    return CFE_WAYPOINTS_60 if iso in SEED_BAND_HIGH else CFE_WAYPOINTS_50

# Model starts at 2030; 2025-2029 use actual grid baseline (no optimization).
MODEL_START_YEAR = 2030
MODEL_START_YI = MODEL_START_YEAR - BASE_YEAR  # = 5

# Pathway A free dimensions (indices into RESOURCE_ORDER)
PATHWAY_A_FREE = ["solar", "wind", "solar_batt4", "solar_batt8",
                  "wind_batt4", "wind_batt8"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class RunConfig:
    iso: str
    pathway: str  # "A", "B", "C", or "D"
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
        if self.pathway in FIRM_PATHWAYS:
            if self.iso in pc.OFFSHORE_ISOS:
                free.append("offshore_wind")
            free.append("clean_firm")
        return free

    @property
    def free_indices(self) -> list[int]:
        return [RES_IDX[r] for r in self.free_resources]

    @property
    def n_dims(self) -> int:
        """Total sampling dimensions = free resources + 3 storage."""
        return len(self.free_resources) + N_STORAGE

    @property
    def scaled_samples(self) -> int:
        """Stage 2 samples scaled by dimensionality.

        Base sample count targets 9 dimensions (Pathway A).
        Scales by (n_dims/9)^1.5 for higher-dimensional pathways.
        """
        return int(self.stage2_samples * (self.n_dims / 9.0) ** 1.5)

    @property
    def scaled_beams(self) -> int:
        """N beams scaled for pathway — firm pathways get +2 for nuclear/offshore archetypes."""
        extra = 2 if self.pathway in FIRM_PATHWAYS else 0
        return self.n_beams + extra

    @property
    def noak_pathway_key(self) -> str | None:
        """Lookup key into NOAK_YEAR_BY_PATHWAY for learning-curve resources.

        B → "3" (proactive, NOAK 2035)
        C → "C" (delayed, NOAK 2040)
        D → "D" (further-delayed, NOAK 2045)
        A → None (no firm resources, learning curve irrelevant)
        """
        return {"B": "3", "C": "C", "D": "D"}.get(self.pathway)

    @property
    def offshore_noak_cap(self) -> int:
        """Offshore wind NOAK year cap — aligned with firm deployment timing."""
        return {"B": 2035, "C": 2040, "D": 2045}.get(self.pathway, 2035)


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
    "all_low": {
        # Everything cheap — floor case, fastest possible decarbonization.
        # Neither pathway has structural advantage; tests whether A-B cost
        # spread collapses when all technologies hit aggressive learning targets.
        "ren_cost": "L", "firm_cost": "L", "batt_cost": "L", "ldes_cost": "L",
        "fuel_cost": "L", "tx_level": "L", "ccs_cost": "L", "geo_cost": "L",
        "q45": "1",
    },
    "all_high": {
        # Everything expensive — ceiling case, costliest decarbonization.
        # Tests which pathway degrades less gracefully under universal cost
        # pressure, and whether gas backstop dependency increases symmetrically.
        "ren_cost": "H", "firm_cost": "H", "batt_cost": "H", "ldes_cost": "H",
        "fuel_cost": "H", "tx_level": "H", "ccs_cost": "H", "geo_cost": "H",
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
                 batt4, batt8, ldes, compute_idx):
    """Fused kernel: hourly_match_score + resid_norm_p9997 + curtailment.

    v4.0: 3 storage tiers only (battery4, battery8, LDES). H2 is a
    deterministic peaker computed post-scoring.

    scores[i] = sum(min(supply_h, dn32[h])) * 100  (dn32 sums to 1.0)
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
        has_stor = (b4 > 0.0) or (b8 > 0.0) or (ld > 0.0)

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
            # Storage path: float64 + 3-stage dispatch
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

            # 3-stage sequential dispatch (battery4 -> battery8 -> LDES)
            b4_c = b4 * 0.01
            leaked_b4 = _inline_storage_stage(surplus, gap_arr, total_dispatch,
                                  b4_c, b4_c / 4.0, 0.85, 24)
            b8_c = b8 * 0.01
            leaked_b8 = _inline_storage_stage(surplus, gap_arr, total_dispatch,
                                  b8_c, b8_c / 8.0, 0.85, 48)
            ld_c = ld * 0.01
            leaked_ld = _inline_storage_stage(surplus, gap_arr, total_dispatch,
                                  ld_c, ld_c / 100.0, 0.50, 168)

            # CFE score
            matched_sum = 0.0
            for h in range(H):
                eff = supply[h] + total_dispatch[h]
                matched_sum += min(eff, float(dn32[h]))

            # Curtailment (post-storage surplus + leaked SOC)
            curt_sum = leaked_b4 + leaked_b8 + leaked_ld
            for h in range(H):
                curt_sum += surplus[h]

            # Residual gap against RA demand (post-storage, pre-H2)
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
                     ldes: np.ndarray,
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score N candidate mixes (3-tier storage, no H2). Returns (scores, resid, curtail)."""
    n = W.shape[0]
    scores = np.full(n, -1.0, dtype=np.float32)
    resid = np.full(n, np.inf, dtype=np.float32)
    curtail = np.zeros(n, dtype=np.float32)
    idx = np.arange(n, dtype=np.int64)
    _score_mixes(scores, resid, curtail, W, P32, dm32, dn32,
                 batt4, batt8, ldes, idx)
    return scores, resid, curtail


# ---------------------------------------------------------------------------
# H2 Peaker — deterministic gap-fill model (v4.0)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _compute_post_storage_gaps(W_row, P32, dn32, dm32,
                                batt4_val, batt8_val, ldes_val):
    """Compute hourly post-3-tier-storage gaps for one candidate.

    Returns (cfe_gaps, ra_gaps, base_cfe) where:
      cfe_gaps[h] = max(0, dn32[h] - effective_supply[h])
      ra_gaps[h]  = max(0, dm32[h] - effective_supply[h])
      base_cfe    = pre-H2 CFE score (percentage)
    """
    NR = W_row.shape[0]
    cfe_gaps = np.empty(H, dtype=np.float64)
    ra_gaps = np.empty(H, dtype=np.float64)
    supply = np.empty(H, dtype=np.float64)
    surplus = np.empty(H, dtype=np.float64)
    gap_arr = np.empty(H, dtype=np.float64)
    total_dispatch = np.zeros(H, dtype=np.float64)

    for h in range(H):
        s = 0.0
        for r in range(NR):
            s += float(W_row[r]) * float(P32[r, h])
        supply[h] = s
        diff = s - float(dn32[h])
        if diff > 0.0:
            surplus[h] = diff; gap_arr[h] = 0.0
        else:
            surplus[h] = 0.0; gap_arr[h] = -diff

    if batt4_val > 0.0:
        b4_c = batt4_val * 0.01
        _inline_storage_stage(surplus, gap_arr, total_dispatch,
                              b4_c, b4_c / 4.0, 0.85, 24)
    if batt8_val > 0.0:
        b8_c = batt8_val * 0.01
        _inline_storage_stage(surplus, gap_arr, total_dispatch,
                              b8_c, b8_c / 8.0, 0.85, 48)
    if ldes_val > 0.0:
        ld_c = ldes_val * 0.01
        _inline_storage_stage(surplus, gap_arr, total_dispatch,
                              ld_c, ld_c / 100.0, 0.50, 168)

    matched_sum = 0.0
    for h in range(H):
        eff = supply[h] + total_dispatch[h]
        g = float(dn32[h]) - eff
        cfe_gaps[h] = g if g > 0.0 else 0.0
        rg = float(dm32[h]) - eff
        ra_gaps[h] = rg if rg > 0.0 else 0.0
        matched_sum += min(eff, float(dn32[h]))

    return cfe_gaps, ra_gaps, matched_sum * 100.0


def h2_size_for_target(cfe_gaps, ra_gaps, base_cfe, target_cfe, dem_twh):
    """Find minimum H2 peaker capacity to close CFE gap to target.

    H2 turbine with rated capacity X dispatches min(cfe_gap[h], X) in
    every gap hour. Binary search finds minimum X.

    Returns dict: h2_capacity_mw, h2_dispatched_mwh, h2_cf_pct,
                  h2_dispatch_hours, post_h2_resid.
    """
    needed = target_cfe - base_cfe
    if needed <= 0:
        sorted_ra = np.sort(ra_gaps)[::-1]
        resid_val = float(sorted_ra[2]) if len(sorted_ra) > 2 else 0.0
        return {"h2_capacity_mw": 0.0, "h2_dispatched_mwh": 0.0,
                "h2_cf_pct": 0.0, "h2_dispatch_hours": 0,
                "post_h2_resid": resid_val}

    needed_norm = needed / 100.0
    total_gap = float(cfe_gaps.sum())
    if total_gap < needed_norm * 0.999:
        return {"h2_capacity_mw": np.inf, "h2_dispatched_mwh": 0.0,
                "h2_cf_pct": 0.0, "h2_dispatch_hours": 0,
                "post_h2_resid": np.inf}

    lo, hi = 0.0, float(cfe_gaps.max()) * 1.001
    for _ in range(60):
        mid = (lo + hi) * 0.5
        if float(np.minimum(cfe_gaps, mid).sum()) >= needed_norm:
            hi = mid
        else:
            lo = mid

    h2_cap = hi
    h2_disp = np.minimum(cfe_gaps, h2_cap)
    h2_norm = float(h2_disp.sum())
    h2_mw = h2_cap * dem_twh * 1e6
    h2_mwh = h2_norm * dem_twh * 1e6
    h2_hrs = int((h2_disp > 1e-12).sum())
    h2_cf = (h2_mwh / (h2_mw * H) * 100) if h2_mw > 0 else 0.0

    post_h2_ra = np.maximum(ra_gaps - h2_disp, 0.0)
    sorted_ra = np.sort(post_h2_ra)[::-1]
    resid_val = float(sorted_ra[2]) if len(sorted_ra) > 2 else 0.0

    return {"h2_capacity_mw": h2_mw, "h2_dispatched_mwh": h2_mwh,
            "h2_cf_pct": round(h2_cf, 2), "h2_dispatch_hours": h2_hrs,
            "post_h2_resid": resid_val}


def h2_peaker_capex_kw_yr(year: int, cfg: RunConfig) -> float:
    """Annualized H2 peaker capacity cost ($/kW-yr) with Wright's Law."""
    level = cfg.ldes_cost
    level_name = LEVEL_NAME[level]
    foak = float(pc.H2_PEAKER_CAPEX_KW_YR[level_name]['FOAK'])
    noak = float(pc.H2_PEAKER_CAPEX_KW_YR[level_name]['NOAK'])
    fs, ny = pc.LEARNING_PARAMS['h2_peaker'][level]
    return wrights_law_cost(foak, noak, year, fs, ny)


def h2_peaker_fuel_mwh(year: int, cfg: RunConfig) -> float:
    """H2 fuel cost ($/MWh dispatched) with Wright's Law."""
    level = cfg.ldes_cost
    level_name = LEVEL_NAME[level]
    foak = float(pc.H2_FUEL_COST_MWH[level_name]['FOAK'])
    noak = float(pc.H2_FUEL_COST_MWH[level_name]['NOAK'])
    fs, ny = pc.LEARNING_PARAMS['h2_peaker'][level]
    return wrights_law_cost(foak, noak, year, fs, ny)


def h2_annual_cost(h2_capacity_mw, h2_dispatched_mwh, year, cfg):
    """Total annual H2 peaker cost = capacity + fuel."""
    if h2_capacity_mw <= 0:
        return 0.0
    return (h2_capacity_mw * h2_peaker_capex_kw_yr(year, cfg) * 1000
            + h2_dispatched_mwh * h2_peaker_fuel_mwh(year, cfg))


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
        fs, ny = pc.get_pathway_noak_window("nuclear", fl, cfg.noak_pathway_key)
        return wrights_law_cost(foak, noak, year, fs, ny)

    if resource == "ccs_ccgt":
        q45_on = cfg.q45 == "1"
        if q45_on:
            noak = pc.CCS_LCOE_45Q_ON[cl][iso]
            foak = pc.FOAK_CCS_45Q_ON[iso]
        else:
            noak = pc.CCS_LCOE_45Q_OFF[cl][iso]
            foak = pc.FOAK_CCS_45Q_OFF[iso]
        fs, ny = pc.get_pathway_noak_window("ccs", cl, cfg.noak_pathway_key)
        cost = wrights_law_cost(foak, noak, year, fs, ny)
        if iso == "NEISO":
            cost += pc.NEISO_CCS_GAS_ADDER
        return cost

    if resource == "offshore_wind":
        if iso not in pc.OFFSHORE_ISOS:
            return 0.0
        tech = "offshore_wind_float" if iso == "CAISO" else "offshore_wind_fixed"
        fs, ny = pc.LEARNING_PARAMS[tech][cfg.ren_cost]  # v3.8 FIX: was "M"
        ny = min(ny, cfg.offshore_noak_cap)  # Offshore NOAK aligned with pathway deployment timing
        return wrights_law_cost(
            pc.FOAK_OFFSHORE_WIND.get(iso, 100),
            pc.NOAK_OFFSHORE_WIND[ren_name].get(iso, 65), year, fs, ny)

    if resource == "geothermal":
        if iso != "CAISO":
            return 0.0
        fs, ny = pc.get_pathway_noak_window("geo", gl, cfg.noak_pathway_key)
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


def cfe_target(year: int, base_pct: float, waypoints: tuple) -> float:
    """CFE target for a given year via linear interpolation of waypoints."""
    pts = [(BASE_YEAR, base_pct)] + list(waypoints)
    if year <= pts[0][0]:
        return pts[0][1]
    if year >= pts[-1][0]:
        return pts[-1][1]
    for (y0, t0), (y1, t1) in zip(pts, pts[1:]):
        if y0 <= year <= y1:
            return t0 + (t1 - t0) * (year - y0) / (y1 - y0)
    return pts[-1][1]


def cfe_target_vec(iso: str) -> np.ndarray:
    """(N_YEARS,) CFE target percentages, using per-ISO waypoints."""
    base = sum(pc.GRID_MIX_SHARES[iso].values())
    waypoints = _waypoints_for_iso(iso)
    return np.array([cfe_target(y, base, waypoints) for y in YEARS])


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
# Second-order probe size for concavity detection (Finding 4).
# Probe at +5pp in addition to +1pp. If yield decays, widen ceiling.
SECOND_ORDER_PROBE_PP = 5.0
CONCAVITY_THRESHOLD = 0.9  # decay below 90% of 1pp yield triggers adjustment

# Joint interaction threshold (Finding 5).
# If top-2 dimensions yield more jointly than independently, boost ceilings.
INTERACTION_THRESHOLD = 0.01  # 0.01pp synergy to trigger

def compute_marginal_yields(
    floor_pcts: np.ndarray,  # (N_RESOURCES,) current floor in %
    floor_storage: np.ndarray,  # (N_STORAGE,) current storage dispatch %
    free_res_idx: list[int],
    P32: np.ndarray, dm32: np.ndarray, dn32: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """Compute marginal CFE yield per +1pp of each free dimension.

    v4.1: Second-order probing for concavity detection (Finding 4) and
    top-2 joint interaction testing (Finding 5). Returns adjusted yields
    that account for nonlinear response at high CFE.

    Returns (res_yields, stor_yields, floor_cfe, res_yields_adj, stor_yields_adj).
    The _adj arrays have yields lowered where concavity is detected,
    and ceilings should be sized from these for better search bounds.
    """
    n_free = len(free_res_idx)

    # Build floor weight vector
    W_floor = np.zeros((1, N_RESOURCES), dtype=np.float32)
    W_floor[0] = (floor_pcts * 0.01).astype(np.float32)

    s_floor = floor_storage.astype(np.float32)
    floor_scores, _, _ = score_candidates(
        W_floor, P32, dm32, dn32,
        s_floor[0:1], s_floor[1:2], s_floor[2:3])
    floor_cfe = float(floor_scores[0])

    # --- First-order probes: +1pp each dimension ---
    res_yields = np.zeros(n_free, dtype=np.float64)
    for k, ri in enumerate(free_res_idx):
        W_pert = W_floor.copy()
        W_pert[0, ri] += np.float32(0.01)  # +1pp
        sc, _, _ = score_candidates(
            W_pert, P32, dm32, dn32,
            s_floor[0:1], s_floor[1:2], s_floor[2:3])
        res_yields[k] = float(sc[0]) - floor_cfe

    stor_yields = np.zeros(N_STORAGE, dtype=np.float64)
    for j in range(N_STORAGE):
        s_pert = s_floor.copy()
        s_pert[j] += np.float32(1.0)
        sc, _, _ = score_candidates(
            W_floor, P32, dm32, dn32,
            s_pert[0:1], s_pert[1:2], s_pert[2:3])
        stor_yields[j] = float(sc[0]) - floor_cfe

    # --- Second-order probes: +5pp for concavity detection (Finding 4) ---
    # If yield at +5pp is significantly lower per-pp than at +1pp,
    # the response surface is concave. Lower the effective yield so
    # ceilings expand (more exploration in the nonlinear region).
    res_yields_adj = res_yields.copy()
    probe_pp = SECOND_ORDER_PROBE_PP
    for k, ri in enumerate(free_res_idx):
        if res_yields[k] < 0.001:
            continue
        W_pert5 = W_floor.copy()
        W_pert5[0, ri] += np.float32(probe_pp * 0.01)
        sc5, _, _ = score_candidates(
            W_pert5, P32, dm32, dn32,
            s_floor[0:1], s_floor[1:2], s_floor[2:3])
        yield_5pp = (float(sc5[0]) - floor_cfe) / probe_pp
        if yield_5pp < res_yields[k] * CONCAVITY_THRESHOLD:
            # Concave: use geometric mean of the two yields as effective rate.
            # This widens the ceiling ~1.5-2× in the concave regime.
            decay = yield_5pp / max(res_yields[k], 1e-6)
            res_yields_adj[k] = res_yields[k] * max(decay, 0.1)

    stor_yields_adj = stor_yields.copy()
    for j in range(N_STORAGE):
        if stor_yields[j] < 0.001:
            continue
        s_pert5 = s_floor.copy()
        s_pert5[j] += np.float32(probe_pp)
        sc5, _, _ = score_candidates(
            W_floor, P32, dm32, dn32,
            s_pert5[0:1], s_pert5[1:2], s_pert5[2:3])
        yield_5pp = (float(sc5[0]) - floor_cfe) / probe_pp
        if yield_5pp < stor_yields[j] * CONCAVITY_THRESHOLD:
            decay = yield_5pp / max(stor_yields[j], 1e-6)
            stor_yields_adj[j] = stor_yields[j] * max(decay, 0.1)

    # --- Joint interaction probe: top-2 dimensions (Finding 5) ---
    # If two dimensions yield more together than independently,
    # boost both adjusted yields by the synergy ratio.
    all_yields = np.concatenate([res_yields, stor_yields])
    if len(all_yields) >= 2:
        top2 = np.argsort(all_yields)[-2:]
        W_joint = W_floor.copy()
        s_joint = s_floor.copy()
        for idx in top2:
            if idx < n_free:
                W_joint[0, free_res_idx[idx]] += np.float32(0.01)
            else:
                s_joint[idx - n_free] += np.float32(1.0)
        sc_joint, _, _ = score_candidates(
            W_joint, P32, dm32, dn32,
            s_joint[0:1], s_joint[1:2], s_joint[2:3])
        joint_yield = float(sc_joint[0]) - floor_cfe
        independent_sum = all_yields[top2[0]] + all_yields[top2[1]]
        interaction = joint_yield - independent_sum

        if interaction > INTERACTION_THRESHOLD:
            boost = 1.0 + interaction / max(independent_sum, 1e-6)
            for idx in top2:
                if idx < n_free:
                    res_yields_adj[idx] /= boost  # lower eff yield → wider ceiling
                else:
                    stor_yields_adj[idx - n_free] /= boost

    return res_yields, stor_yields, floor_cfe, res_yields_adj, stor_yields_adj


# ---------------------------------------------------------------------------
# Sobol candidate generation (nested: first N of 2N draw = standalone N)
# ---------------------------------------------------------------------------
def generate_candidates(
    floor_pcts: np.ndarray,   # (N_RESOURCES,) floor in %
    floor_storage: np.ndarray,  # (N_STORAGE,) floor storage dispatch %
    target_cfe: float,
    floor_cfe: float,
    res_yields: np.ndarray,    # (n_free,) marginal yields for free resources
    stor_yields: np.ndarray,   # (N_STORAGE,) marginal yields for storage
    free_res_idx: list[int],
    cfg: RunConfig,
    n_samples: int,
    beam_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate Sobol candidates (3 storage, no H2).

    Returns (W, batt4, batt8, ldes, raw_pcts) where raw_pcts has shape
    (n_total, n_free_res + N_STORAGE).  n_total = n_samples + n_zero_storage.

    Uses a nested Sobol sequence so the first N points of a 2N draw are
    identical to a standalone N draw — increasing sample count is guaranteed
    to weakly improve results.

    A small zero-storage pool is appended (not overwritten) to ensure the
    optimizer evaluates no-storage solutions.  Pool is larger at CFE targets
    below 80% where storage-free paths are more likely to be cost-effective.
    """
    n_free = len(free_res_idx)
    increment = max(0.1, target_cfe - floor_cfe)

    # ── Per-dimension ceilings ─────────────────────────────────────────
    res_ceilings = np.zeros(n_free, dtype=np.float64)
    for k in range(n_free):
        yld = res_yields[k]
        if yld > 0.001:
            res_ceilings[k] = floor_pcts[free_res_idx[k]] + increment / yld * SAFETY_FACTOR
        else:
            res_ceilings[k] = floor_pcts[free_res_idx[k]] + 1.0  # minimal headroom

    stor_ceilings = np.zeros(N_STORAGE, dtype=np.float64)
    for j in range(N_STORAGE):
        yld = stor_yields[j]
        if yld > 0.001:
            stor_ceilings[j] = floor_storage[j] + increment / yld * SAFETY_FACTOR
        else:
            stor_ceilings[j] = floor_storage[j] + 1.0

    # Hard caps — resources
    MAX_RESOURCE_PCT = 200.0
    for k in range(n_free):
        res_ceilings[k] = min(res_ceilings[k], MAX_RESOURCE_PCT)

    # Hard caps — storage: per-type physical maximums (v4.4 FIX)
    for j in range(N_STORAGE):
        stor_ceilings[j] = min(stor_ceilings[j], _STORAGE_TYPE_CAPS[j])

    # Apply hard caps (CCS, geo, uprate for firm pathways B/C/D)
    if cfg.pathway in FIRM_PATHWAYS:
        demand_twh = float(pc.REGIONAL_DEMAND_TWH[cfg.iso])
        for k, ri in enumerate(free_res_idx):
            res = RESOURCE_ORDER[ri]
            if res == "ccs_ccgt":
                cap_pct = float(pc.CCS_CAP_TWH.get(cfg.iso, 9999)) / demand_twh * 100
                res_ceilings[k] = min(res_ceilings[k], cap_pct)
            elif res == "geothermal":
                cap_pct = float(pc.GEOTHERMAL_CAP_TWH) / demand_twh * 100
                res_ceilings[k] = min(res_ceilings[k], cap_pct)

    # ── Build bounds ───────────────────────────────────────────────────
    n_dims = n_free + N_STORAGE
    floors = np.zeros(n_dims)
    ceilings = np.zeros(n_dims)
    for k in range(n_free):
        floors[k] = floor_pcts[free_res_idx[k]]
        ceilings[k] = res_ceilings[k]
    for j in range(N_STORAGE):
        floors[n_free + j] = floor_storage[j]
        ceilings[n_free + j] = stor_ceilings[j]

    ceilings = np.maximum(ceilings, floors + 0.01)

    # ── Sobol sequence (nested: first N of 2N draw ≡ standalone N) ────
    seed_str = (f"{cfg.iso}_{cfg.pathway}_{target_cfe:.1f}"
                f"_{cfg.demand_growth}_{beam_idx}")
    seed_val = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)
    sampler = Sobol(d=n_dims, seed=seed_val)
    unit_samples = sampler.random(n=n_samples)

    # Scale to [floor, ceiling]
    raw_sobol = floors[None, :] + unit_samples * (ceilings - floors)[None, :]

    # ── Zero-storage pool (appended, not overwritten) ─────────────────
    # Larger pool at low CFE thresholds where storage-free paths are viable;
    # small pool at high thresholds as a safety check.
    zs_frac = 0.10 if target_cfe < 80.0 else 0.02
    zs_count = max(n_dims, int(n_samples * zs_frac))

    # Copy resource dims from evenly-spaced Sobol rows, zero all storage
    zs_stride = max(1, n_samples // zs_count)
    zs_source_idx = np.arange(0, n_samples, zs_stride)[:zs_count]
    raw_zs = raw_sobol[zs_source_idx].copy()
    for j in range(N_STORAGE):
        raw_zs[:, n_free + j] = floors[n_free + j]

    # ── Combine pools ─────────────────────────────────────────────────
    raw = np.vstack([raw_sobol, raw_zs])
    n_total = raw.shape[0]

    # ── Build output arrays ───────────────────────────────────────────
    W = np.zeros((n_total, N_RESOURCES), dtype=np.float32)
    for ri in range(N_RESOURCES):
        W[:, ri] = np.float32(floor_pcts[ri] * 0.01)
    for k, ri in enumerate(free_res_idx):
        W[:, ri] = (raw[:, k] * 0.01).astype(np.float32)

    batt4 = raw[:, n_free + 0].astype(np.float32)
    batt8 = raw[:, n_free + 1].astype(np.float32)
    ldes_arr = raw[:, n_free + 2].astype(np.float32)

    # Combined storage cap: sum of per-type physical caps (v4.4 FIX)
    cap = np.float32(_STORAGE_COMBINED_CAP)
    stor_total = batt4 + batt8 + ldes_arr
    over = stor_total > cap
    if over.any():
        scale = np.where(over, cap / np.maximum(stor_total, 1e-6),
                         1.0).astype(np.float32)
        batt4 *= scale; batt8 *= scale; ldes_arr *= scale
        raw[:, n_free + 0] = batt4; raw[:, n_free + 1] = batt8
        raw[:, n_free + 2] = ldes_arr

    return W, batt4, batt8, ldes_arr, raw


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


def classify_archetype(mix_pcts: dict[str, float], pathway: str,
                       iso: str = "") -> str:
    """Classify a mix into an archetype by dominant *incremental* resource.

    Subtracts the ISO's grid baseline so the archetype reflects what was ADDED,
    not the total mix.  PJM with 32% nuclear + 2% solar + 16% wind added
    → wind-led, not balanced.
    """
    baseline = pc.GRID_MIX_SHARES.get(iso, {}) if iso else {}

    solar_family = sum(max(mix_pcts.get(r, 0) - baseline.get(r, 0), 0)
                       for r in ["solar", "solar_batt4", "solar_batt8"])
    wind_family = sum(max(mix_pcts.get(r, 0) - baseline.get(r, 0), 0)
                      for r in ["wind", "wind_batt4", "wind_batt8"])
    total_delta = sum(max(mix_pcts.get(r, 0) - baseline.get(r, 0), 0)
                      for r in RESOURCE_ORDER)
    if total_delta < 1e-6:
        return "balanced"

    sf = solar_family / total_delta
    wf = wind_family / total_delta

    if pathway in FIRM_PATHWAYS:
        nuke_delta = max(mix_pcts.get("clean_firm", 0)
                         - baseline.get("clean_firm", 0), 0)
        off_delta = max(mix_pcts.get("offshore_wind", 0)
                        - baseline.get("offshore_wind", 0), 0)
        if nuke_delta / total_delta > 0.4:
            return "nuclear-heavy"
        if off_delta / total_delta > 0.3:
            return "offshore-heavy"

    if sf > 0.5:
        return "solar-led"
    if wf > 0.5:
        return "wind-led"
    return "balanced"


def stage1_select(iso: str, cfg: RunConfig, P32: np.ndarray,
                  dm32: np.ndarray, dn32: np.ndarray
                  ) -> list[dict]:
    """Load EF parquets at the ISO's seed band (50 or 60), select diverse seeds.

    The seed band represents the 2030 starting point — the first year modeled.
    ISOs with baseline >= 45% seed from band 60; others from band 50.
    """
    seed_band = SEED_BAND_BY_ISO[iso]
    base_pct = sum(pc.GRID_MIX_SHARES[iso].values())

    print(f"[stage1] {iso}: baseline={base_pct:.1f}%, seed band={seed_band}%")
    ef = load_ef_band(iso, seed_band)
    if ef is None:
        print(f"[stage1] {iso}: no EF parquet for seed band {seed_band}")
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
    scores, resid, curtail = score_candidates(W, P32, dm32, dn32,
                                               batt4, batt8, ldes_arr)

    next_threshold = 100.0
    for t in EF_BAND_THRESHOLDS:
        if t > seed_band:
            next_threshold = t
            break
    mask = (scores >= seed_band - 0.5) & (scores < next_threshold)
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
            sv = [batt4, batt8, ldes_arr][j][vi]
            if sv > 0:
                c += float(sv) / 100.0 * storage_net_cost(
                    iso, sc, cfg, year=BASE_YEAR) * base_dem * 1e6
        costs[ki] = c

    archetypes = {}
    for ki, vi in enumerate(valid_idx):
        mix = {res: float(W[vi, ri]) * 100.0 for ri, res in enumerate(RESOURCE_ORDER)}
        arch = classify_archetype(mix, cfg.pathway, iso=iso)
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
            arch = classify_archetype(mix, cfg.pathway, iso=iso)
            seed = {
                "archetype": arch,
                "resource_pcts": mix,
                "storage_pcts": {
                    STORAGE_COLS[0]: float(batt4[vi]),
                    STORAGE_COLS[1]: float(batt8[vi]),
                    STORAGE_COLS[2]: float(ldes_arr[vi]),
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
    """Build a 2030-2050 pathway from a seed mix.

    Solver starts at MODEL_START_YEAR (2030). Years 2025-2029 are not modeled.
    Floor is initialized directly from the seed's resource mix.

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

    # Initialize floor from seed — the 2030 starting mix.
    # Years 2025-2029 are not modeled.
    floor_pcts = np.zeros(N_RESOURCES, dtype=np.float64)
    for ri, res in enumerate(RESOURCE_ORDER):
        floor_pcts[ri] = seed["resource_pcts"].get(res, 0.0)

    # Ensure baseline clean_firm and hydro are not below grid actuals
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
    peak_h2_mw = 0.0

    threshold_snapshots = []
    thresholds_crossed = set()
    peak_gas_mw = 0.0
    peak_gas_year = MODEL_START_YEAR

    for yi in range(MODEL_START_YI, N_YEARS):
        year = YEARS[yi]
        dem_twh = demand_vec[yi]
        target = cfe_targets[yi]

        for res, twh in frozen_twh.items():
            ri = RES_IDX[res]
            floor_pcts[ri] = twh / dem_twh * 100.0

        if cfg.pathway == "A":
            for res in ["offshore_wind", "geothermal", "ccs_ccgt"]:
                floor_pcts[RES_IDX[res]] = 0.0
        elif cfg.pathway in FIRM_PATHWAYS:
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
            cfe_ceiling = cfe_targets[yi + 1]
        else:
            cfe_ceiling = 100.0

        res_yields, stor_yields, floor_cfe, res_yields_adj, stor_yields_adj = compute_marginal_yields(
            floor_pcts, floor_storage, free_idx, P32, dm32, dn32)

        if floor_cfe >= target:
            winner_pcts = floor_pcts.copy()
            winner_storage = floor_storage.copy()
            winner_cfe = floor_cfe
            _, winner_resid, _ = _score_single(floor_pcts, floor_storage,
                                               P32, dm32, dn32)
            h2_info = {"h2_capacity_mw": 0.0, "h2_dispatched_mwh": 0.0,
                       "h2_cf_pct": 0.0, "h2_dispatch_hours": 0,
                       "post_h2_resid": winner_resid}
        else:
            # Pass adjusted yields (concavity + interaction corrected) for
            # ceiling sizing; raw yields still available for diagnostics.
            result_tuple = _find_winner(
                floor_pcts, floor_storage, target, cfe_ceiling,
                floor_cfe, res_yields_adj, stor_yields_adj,
                free_idx, cfg, P32, dm32, dn32,
                year, dem_twh,
                beam_idx=beam_idx,
                existing_gas_mw=existing_gas,
                gaf=gaf,
                prev_gas_cost=prev_gas_cost,
                current_peak_gas=peak_gas_mw)
            winner_pcts, winner_storage, winner_cfe, winner_resid, h2_info = result_tuple

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

                if res == "clean_firm" and cfg.pathway in FIRM_PATHWAYS:
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

        # H2 cost this year
        h2_yr_cost = h2_annual_cost(h2_info["h2_capacity_mw"],
                                     h2_info["h2_dispatched_mwh"], year, cfg)
        year_incremental_cost += h2_yr_cost
        if h2_info["h2_capacity_mw"] > peak_h2_mw:
            peak_h2_mw = h2_info["h2_capacity_mw"]

        # Gas from post-H2 residual
        g_mw = gas_need_mw(h2_info["post_h2_resid"], dem_twh, existing_gas, gaf)
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

        for j in range(N_STORAGE):
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
                        for j in range(N_STORAGE)
                    },
                    "h2_capacity_mw": round(h2_info["h2_capacity_mw"], 1),
                    "h2_dispatched_mwh": round(h2_info["h2_dispatched_mwh"], 0),
                    "h2_cf_pct": h2_info["h2_cf_pct"],
                    "h2_dispatch_hours": h2_info["h2_dispatch_hours"],
                    "h2_annual_cost_usd": round(h2_yr_cost, 0),
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
                  f"gas={g_mw:,.0f}MW h2={h2_info['h2_capacity_mw']:,.0f}MW "
                  f"cost=${total_annual/1e9:.2f}B")

    return {
        "schema_version": "4.4",
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
            "noak_pathway_key": cfg.noak_pathway_key,
            "offshore_noak_cap": cfg.offshore_noak_cap,
            "stranding_shadow_weight": STRANDING_SHADOW_WEIGHT,
            "seed_band": SEED_BAND_BY_ISO[iso],
            "model_start_year": MODEL_START_YEAR,
            "storage_type_caps": list(_STORAGE_TYPE_CAPS),
        },
        "beam_archetype": seed["archetype"],
        "beam_index": beam_idx,
        "peak_gas_mw": round(peak_gas_mw, 1),
        "peak_gas_year": peak_gas_year,
        "peak_h2_mw": round(peak_h2_mw, 1),
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
    """Find cheapest candidate; H2 fills remaining CFE gap post-scoring (v4.0).

    Candidates scored with 3-tier storage. H2 sized per-candidate to close
    gap to target. Cost ranking includes clean + storage + H2 + gas shadow.
    Returns (winner_pcts, winner_storage, post_h2_cfe, post_h2_resid, h2_info).
    """
    capex_yr = h2_peaker_capex_kw_yr(year, cfg)
    fuel_mwh_cost = h2_peaker_fuel_mwh(year, cfg)

    for multiplier in [1, 2, 4]:
        n = cfg.scaled_samples * multiplier
        W, b4, b8, ld, raw = generate_candidates(
            floor_pcts, floor_storage, target, floor_cfe,
            res_yields, stor_yields, free_idx, cfg, n,
            beam_idx=beam_idx)

        scores, resid, curtail = score_candidates(
            W, P32, dm32, dn32, b4, b8, ld)

        # Accept candidates below target (H2 fills gap) but not overbuilt
        min_pre_h2 = max(target - H2_MAX_FILL_PP, floor_cfe)
        mask = (scores >= min_pre_h2) & (scores < ceiling)
        valid = np.where(mask)[0]

        if len(valid) == 0:
            if multiplier < 4:
                print(f"    cascade {multiplier}x->{multiplier*2}x: "
                      f"0 hits in [{min_pre_h2:.1f}, {ceiling:.1f})")
            continue

        costs = np.zeros(len(valid), dtype=np.float64)
        h2_results = []

        for ki, vi in enumerate(valid):
            c = 0.0
            for k, ri in enumerate(free_idx):
                delta = float(W[vi, ri]) * 100.0 - floor_pcts[ri]
                if delta > 0:
                    c += delta / 100.0 * dem_twh * get_resource_lcoe(
                        cfg.iso, RESOURCE_ORDER[ri], year, cfg) * 1e6
            for j, sc in enumerate(STORAGE_COLS):
                sv = [b4, b8, ld][j][vi]
                delta = float(sv) - floor_storage[j]
                if delta > 0:
                    c += (delta / 100.0 * storage_net_cost(
                        cfg.iso, sc, cfg, year=year) * dem_twh * 1e6)

            # H2 sizing
            pre_cfe = float(scores[vi])
            if pre_cfe >= target:
                h2r = {"h2_capacity_mw": 0.0, "h2_dispatched_mwh": 0.0,
                       "h2_cf_pct": 0.0, "h2_dispatch_hours": 0,
                       "post_h2_resid": float(resid[vi])}
            else:
                cfe_gaps, ra_gaps, _ = _compute_post_storage_gaps(
                    W[vi], P32, dn32, dm32,
                    float(b4[vi]), float(b8[vi]), float(ld[vi]))
                h2r = h2_size_for_target(cfe_gaps, ra_gaps, pre_cfe,
                                         target, dem_twh)
            h2_results.append(h2r)
            c += (h2r["h2_capacity_mw"] * capex_yr * 1000
                  + h2r["h2_dispatched_mwh"] * fuel_mwh_cost)
            costs[ki] = c

        # Gas stranding shadow (post-H2)
        if STRANDING_SHADOW_WEIGHT > 0.0:
            for ki, vi in enumerate(valid):
                g_mw = gas_need_mw(h2_results[ki]["post_h2_resid"],
                                   dem_twh, existing_gas_mw, gaf)
                shadow = gas_stranding_shadow(g_mw, current_peak_gas,
                                             year, cfg.iso)
                costs[ki] += shadow

        # Mode 2: gas savings
        if cfg.cost_mode == 2 and prev_gas_cost > 0:
            for ki, vi in enumerate(valid):
                g_mw = gas_need_mw(h2_results[ki]["post_h2_resid"],
                                   dem_twh, existing_gas_mw, gaf)
                g_cost = gas_annual_cost(g_mw, cfg.iso, cfg)
                costs[ki] -= (prev_gas_cost - g_cost)

        # Filter infeasible H2
        feas = np.array([h2_results[ki]["h2_capacity_mw"] < np.inf
                         for ki in range(len(valid))])
        if not feas.any():
            if multiplier < 4:
                print(f"    cascade {multiplier}x->{multiplier*2}x: all H2-infeasible")
            continue

        fc = np.where(feas, costs, np.inf)
        best_ki = int(np.argmin(fc))
        best_vi = valid[best_ki]
        best_h2 = h2_results[best_ki]

        winner_pcts = np.array([float(W[best_vi, ri]) * 100.0
                                for ri in range(N_RESOURCES)], dtype=np.float64)
        for ri in range(N_RESOURCES):
            if ri not in free_idx:
                winner_pcts[ri] = floor_pcts[ri]
        winner_storage = np.array([
            float([b4, b8, ld][j][best_vi]) for j in range(N_STORAGE)
        ], dtype=np.float64)

        post_cfe = float(scores[best_vi])
        if best_h2["h2_dispatched_mwh"] > 0 and dem_twh > 0:
            post_cfe += best_h2["h2_dispatched_mwh"] / (dem_twh * 1e6) * 100.0

        return (winner_pcts, winner_storage, post_cfe,
                best_h2["post_h2_resid"], best_h2)

    return None, None, None, None, None


def _score_single(pcts, storage, P32, dm32, dn32):
    """Score a single mix (3-tier storage). Returns (cfe, resid, curtail)."""
    W = np.zeros((1, N_RESOURCES), dtype=np.float32)
    W[0] = (pcts * 0.01).astype(np.float32)
    s = storage.astype(np.float32)
    scores, resid, curtail = score_candidates(
        W, P32, dm32, dn32,
        s[0:1], s[1:2], s[2:3])
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
                 batt, zero, zero, idx)
    # Warm up H2 gap kernel
    W_row = np.zeros(nr, dtype=np.float32)
    P_w = np.zeros((nr, H), dtype=np.float32)
    _compute_post_storage_gaps(W_row, P_w, dn, dm, 0.0, 0.0, 0.0)
    print(f"[adaptive] JIT warm-up: {time.time()-t0:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------
def consolidate_to_parquet(iso: str, pathway: str):
    """Read all JSONs for this ISO+pathway, write a single parquet.

    One row per threshold snapshot.  Beam-level metadata (archetype, scenario,
    cost_mode, demand_growth, peak values) are repeated on every row so the
    parquet is self-describing without joins.

    Output: data/step2.3-adaptive/{ISO}__pathway{X}.parquet
    """
    pattern = OUTPUT_DIR / f"{iso}__pathway{pathway}__*.json"
    files = sorted(OUTPUT_DIR.glob(f"{iso}__pathway{pathway}__*.json"))
    if not files:
        print(f"[consolidate] No JSONs found for {iso} pathway {pathway}")
        return

    rows = []
    for fp in files:
        with open(fp) as f:
            try:
                d = json.load(f)
            except json.JSONDecodeError:
                print(f"[consolidate] WARNING: skipping malformed {fp.name}")
                continue

        # Beam-level fields
        beam_meta = {
            "iso": d["iso"],
            "pathway": d["pathway"],
            "scenario": d.get("scenario_name", "custom"),
            "cost_mode": d["cost_mode"],
            "archetype": d["beam_archetype"],
            "demand_growth": d["config"]["demand_growth"],
            "dim_key": d.get("dim_key", ""),
            "schema_version": d.get("schema_version", ""),
            "seed_band": d["config"].get("seed_band", 0),
            "peak_gas_mw": d["peak_gas_mw"],
            "peak_gas_year": d["peak_gas_year"],
            "peak_h2_mw": d.get("peak_h2_mw", 0),
            "n_thresholds_achieved": d["n_thresholds_achieved"],
        }

        for snap in d["threshold_snapshots"]:
            row = dict(beam_meta)
            # Snapshot scalars
            for k in ("threshold_pct", "year_achieved", "achieved_cfe_pct",
                       "h2_capacity_mw", "h2_dispatched_mwh", "h2_cf_pct",
                       "h2_dispatch_hours", "h2_annual_cost_usd",
                       "gas_need_mw", "stranded_vs_peak_mw",
                       "total_annual_cost_usd", "incremental_cost_usd",
                       "cumulative_cost_usd", "curtailment_total",
                       "demand_twh"):
                row[k] = snap.get(k, 0.0)
            # Flatten resource mix
            rmix = snap.get("resource_mix_pct", {})
            for res in RESOURCE_ORDER:
                row[f"res_{res}"] = rmix.get(res, 0.0)
            # Flatten storage dispatch (v4.4)
            smix = snap.get("storage_dispatch_pct", {})
            for sc in STORAGE_COLS:
                row[f"stor_{sc.replace('_dispatch_pct', '')}"] = smix.get(sc, 0.0)
            rows.append(row)

    if not rows:
        print(f"[consolidate] No snapshots extracted for {iso} pathway {pathway}")
        return

    table = pa.Table.from_pylist(rows)
    out_path = OUTPUT_DIR / f"{iso}__pathway{pathway}.parquet"
    pq.write_table(table, out_path, compression="zstd")
    print(f"[consolidate] Wrote {out_path.name}: {len(rows)} rows from {len(files)} JSONs "
          f"({out_path.stat().st_size / 1024:.1f} KB)")


def write_result(result: dict):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fname = (f"{result['iso']}__pathway{result['pathway']}__"
             f"{result.get('scenario_name', 'custom')}__"
             f"mode{result['cost_mode']}__{result['beam_archetype']}__"
             f"dg_{result['config']['demand_growth']}.json")
    path = OUTPUT_DIR / fname
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[output] wrote {path.name}")


# ---------------------------------------------------------------------------
# Parallel beam worker
# ---------------------------------------------------------------------------
def _solve_beam_worker(seed, cfg, P32, dm32, dn32, bi, n_workers):
    """Worker for ProcessPoolExecutor.  Each spawned process gets:
      - numba thread budget = total_cores / n_workers (avoids prange contention)
      - stdout redirected to StringIO (replayed in order by parent)
      - its own JIT warmup (spawn = fresh process)
    Returns (result_dict, captured_log_str, wall_seconds).
    """
    import numba
    threads = max(1, os.cpu_count() // n_workers)
    numba.set_num_threads(threads)

    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        warmup_jit()
        t0 = time.time()
        result = solve_pathway(seed, cfg, P32, dm32, dn32, beam_idx=bi)
        result["scenario_name"] = cfg.scenario_name
        dt = time.time() - t0
        n_achieved = result["n_thresholds_achieved"]
        print(f"[solve] Done in {dt:.1f}s — {n_achieved} thresholds achieved, "
              f"peak gas={result['peak_gas_mw']:,.0f} MW")
        write_result(result)
    finally:
        sys.stdout = old_stdout
    return result, buf.getvalue(), dt


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
    ap.add_argument("--pathway", type=str, required=True, choices=["A", "B", "C", "D"])
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
    ap.add_argument("--serial", action="store_true",
                    help="Disable beam parallelism (debug/profiling)")
    args = ap.parse_args()

    iso = args.iso.upper()
    if iso not in pc.ISOS:
        raise SystemExit(f"Unknown ISO: {iso}. Known: {list(pc.ISOS)}")

    scenarios = list(COST_SCENARIOS.keys()) if args.batch else [args.scenario]
    demand_levels = (["Low", "Medium", "High"] if args.batch_demand
                     else [args.demand_growth])

    print(f"\n{'='*70}")
    print(f"[adaptive] Step 2.3 Reliability Tax Adaptive v4.4 (per-type storage caps)")
    print(f"[adaptive] ISO={iso}  Pathway={args.pathway}  Seed band={SEED_BAND_BY_ISO[iso]}%")
    print(f"[adaptive] Scenarios: {scenarios}")
    print(f"[adaptive] Demand levels: {demand_levels}")
    print(f"[adaptive] Storage caps: bat4={_STORAGE_TYPE_CAPS[0]}% "
          f"bat8={_STORAGE_TYPE_CAPS[1]}% ldes={_STORAGE_TYPE_CAPS[2]}% "
          f"combined={_STORAGE_COMBINED_CAP}%")
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
    print(f"[adaptive] Profiles loaded. Demand TWh={float(pc.REGIONAL_DEMAND_TWH[iso]):.1f}"
          f"  peak/avg={dn.max()*H:.2f}")

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
    n_workers = min(len(seeds), max(1, os.cpu_count() - 1))
    use_parallel = (not args.serial) and len(seeds) > 1

    if use_parallel:
        print(f"[adaptive] Parallel beams: {len(seeds)} beams across "
              f"{n_workers} workers "
              f"({os.cpu_count()} cores, ~{os.cpu_count()//n_workers} "
              f"numba threads/worker)")
    else:
        print(f"[adaptive] Serial beams: {len(seeds)} beams")

    for ci, cfg in enumerate(configs):
        print(f"\n{'_'*60}")
        print(f"[adaptive] Config {ci+1}/{len(configs)}: "
              f"{cfg.scenario_name} | {cfg.demand_growth} | mode={cfg.cost_mode}")
        print(f"[adaptive] dim_key: {cfg.dim_key}")
        print(f"{'_'*60}")

        if use_parallel:
            # ── Parallel beam execution ───────────────────────────────
            with ProcessPoolExecutor(max_workers=n_workers) as exe:
                futures = {
                    exe.submit(_solve_beam_worker,
                               seed, cfg, P32, dm32, dn32, bi, n_workers): bi
                    for bi, seed in enumerate(seeds)
                }
                # Collect in submission order
                beam_outputs = [None] * len(seeds)
                for fut in futures:
                    bi = futures[fut]
                    beam_outputs[bi] = fut.result()

            # Replay logs in beam order
            for bi, (result, log, dt) in enumerate(beam_outputs):
                print(f"\n[solve] Beam {bi+1}/{len(seeds)}: "
                      f"{seeds[bi]['archetype']} "
                      f"(seed CFE={seeds[bi]['cfe']:.1f}%)")
                sys.stdout.write(log)
        else:
            # ── Serial fallback (--serial or single beam) ─────────────
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

    # Consolidate all JSONs for this ISO+pathway into a single parquet.
    consolidate_to_parquet(iso, args.pathway)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
