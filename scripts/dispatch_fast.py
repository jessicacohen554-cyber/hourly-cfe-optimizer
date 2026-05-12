"""
dispatch_fast.py — Array-in / scalar-out dispatch scoring for DE optimizer hot path.

Drop-in replacement for the score_cfe → reconstruct_hourly_dispatch → pcts_to_dict
chain used inside _DEObjective._objective.  Same math, fewer allocations:

  - No dict construction/lookup (arrays in, arrays out)
  - No demand_arr copy per call (pre-allocated once)
  - No fossil_displaced / curtailed arrays (TF script never reads them)
  - No H2 dispatch call when pct=0 (always zero in TF)
  - No result dict (returns scalar + array reference)
  - Pre-allocated buffers for all intermediate 8760 arrays

NOT a general-purpose replacement for dispatch_utils.reconstruct_hourly_dispatch.
Only correct when procurement_pct=100 and h2_dispatch_pct=0 (both true in TF).
All other callers (floor scoring, winner scoring, cache builders) should continue
using the canonical dispatch_utils entry point.

Deploy to: scripts/dispatch_fast.py
"""

import numpy as np

from dispatch_utils import (
    _battery_loop, _ldes_loop,
    H,
    BATTERY_DURATION_HOURS, BATTERY_EFFICIENCY,
    BATTERY8_DURATION_HOURS, BATTERY8_EFFICIENCY,
    LDES_DURATION_HOURS, LDES_EFFICIENCY, LDES_WINDOW_DAYS,
    RESOURCE_TYPES_HYBRID,
)
# Only needed for RESOURCE_ORDER index mapping
from step2_3_adaptive_sobol import RESOURCE_ORDER


# ── Index map (computed once at import time) ──────────────────────────
# Maps each RESOURCE_TYPES_HYBRID position to its RESOURCE_ORDER index.
# Geothermal (in RESOURCE_ORDER, not in HYBRID) is correctly excluded:
# its pct never enters the matmul.
_HYBRID_TO_RES = np.array(
    [RESOURCE_ORDER.index(rt) for rt in RESOURCE_TYPES_HYBRID],
    dtype=np.int64,
)
_N_HYBRID = len(RESOURCE_TYPES_HYBRID)

# ── Storage constants (avoid module-level lookups per call) ───────────
_B4_DUR = float(BATTERY_DURATION_HOURS)
_B4_EFF = float(BATTERY_EFFICIENCY)
_B8_DUR = float(BATTERY8_DURATION_HOURS)
_B8_EFF = float(BATTERY8_EFFICIENCY)
_LDES_DUR = float(LDES_DURATION_HOURS)
_LDES_EFF = float(LDES_EFFICIENCY)
_LDES_WIN = int(LDES_WINDOW_DAYS * 24)


class DispatchBuffers:
    """Pre-allocated work arrays for hot-path dispatch scoring.

    Create once per year-step in _DEObjective.__init__.  Pickle-safe
    for scipy DE workers > 1: each subprocess gets its own copy so
    buffer reuse within sequential evaluations is safe.
    """

    __slots__ = (
        'demand_arr', 'demand_sum', 'supply_matrix',
        'buf_weights', 'buf_supply', 'buf_surplus', 'buf_gap',
        'buf_b4', 'buf_b8', 'buf_ldes', 'buf_total', 'buf_resid',
    )

    def __init__(self, demand_norm, supply_matrix):
        self.demand_arr = np.array(demand_norm[:H], dtype=np.float64)
        self.demand_sum = float(np.sum(self.demand_arr))
        self.supply_matrix = supply_matrix

        self.buf_weights = np.zeros(_N_HYBRID, dtype=np.float64)
        self.buf_supply = np.zeros(H, dtype=np.float64)
        self.buf_surplus = np.zeros(H, dtype=np.float64)
        self.buf_gap = np.zeros(H, dtype=np.float64)
        self.buf_b4 = np.zeros(H, dtype=np.float64)
        self.buf_b8 = np.zeros(H, dtype=np.float64)
        self.buf_ldes = np.zeros(H, dtype=np.float64)
        self.buf_total = np.zeros(H, dtype=np.float64)
        self.buf_resid = np.zeros(H, dtype=np.float64)


def score_cfe_fast(pcts_arr, b4_pct, b8_pct, ldes_pct, bufs):
    """Score portfolio CFE — array-in, scalar-out.

    Computes exactly the same result as:
        score_cfe(pcts_to_dict(pcts_arr), b4, b8, ldes,
                  demand_norm, supply_profiles, supply_matrix)

    but skips dict construction, demand copies, unused array ops,
    and result dict overhead.

    Args:
        pcts_arr: (N_RESOURCES,) array in RESOURCE_ORDER
        b4_pct, b8_pct, ldes_pct: storage dispatch percentages
        bufs: DispatchBuffers instance

    Returns:
        (cfe_percent, residual_array).
        residual_array is bufs.buf_resid — valid until the next call
        with the same bufs instance.
    """
    # ── Generation: mix_weights @ supply_matrix ──
    # Fancy-index maps RESOURCE_ORDER → HYBRID order in one shot.
    bufs.buf_weights[:] = pcts_arr[_HYBRID_TO_RES] / 100.0
    np.matmul(bufs.buf_weights, bufs.supply_matrix, out=bufs.buf_supply)

    # ── Residuals (in-place, no temporaries) ──
    np.subtract(bufs.buf_supply, bufs.demand_arr, out=bufs.buf_surplus)
    np.maximum(bufs.buf_surplus, 0.0, out=bufs.buf_surplus)
    np.subtract(bufs.demand_arr, bufs.buf_supply, out=bufs.buf_gap)
    np.maximum(bufs.buf_gap, 0.0, out=bufs.buf_gap)

    # ── Storage dispatch chain: battery4 → battery8 → LDES ──
    # Each numba kernel modifies buf_surplus / buf_gap in-place,
    # so downstream kernels see post-upstream residuals.
    bufs.buf_b4[:] = 0.0
    if b4_pct > 0:
        cap4 = b4_pct / 100.0
        _battery_loop(bufs.buf_surplus, bufs.buf_gap, bufs.buf_b4,
                      cap4, cap4 / _B4_DUR, _B4_EFF, 24, H)

    bufs.buf_b8[:] = 0.0
    if b8_pct > 0:
        cap8 = b8_pct / 100.0
        _battery_loop(bufs.buf_surplus, bufs.buf_gap, bufs.buf_b8,
                      cap8, cap8 / _B8_DUR, _B8_EFF, 48, H)

    bufs.buf_ldes[:] = 0.0
    if ldes_pct > 0:
        ecap = ldes_pct / 100.0
        _ldes_loop(bufs.buf_surplus, bufs.buf_gap, bufs.buf_ldes,
                   ecap, ecap / _LDES_DUR, _LDES_EFF, _LDES_WIN, H)

    # ── Total clean + CFE ──
    np.add(bufs.buf_supply, bufs.buf_b4, out=bufs.buf_total)
    np.add(bufs.buf_total, bufs.buf_b8, out=bufs.buf_total)
    np.add(bufs.buf_total, bufs.buf_ldes, out=bufs.buf_total)

    # One temporary for the min — acceptable at ~4 µs
    matched_sum = float(np.minimum(bufs.buf_total, bufs.demand_arr).sum())
    cfe = matched_sum / bufs.demand_sum * 100.0

    # ── Residual demand (in-place) ──
    np.subtract(bufs.demand_arr, bufs.buf_total, out=bufs.buf_resid)
    np.maximum(bufs.buf_resid, 0.0, out=bufs.buf_resid)

    return cfe, bufs.buf_resid
