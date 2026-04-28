#!/usr/bin/env python3
"""
step2_3_scoring_fixes.py  v3.0

Apply these str_replace patches in order against
scripts/step2_3_reliability_tax_adaptive.py.

Patch 1 (a–d): Curtailment tracking — reporting only, no optimization impact.
Patch 2 (a–b): Storage cap — physical limit from peak demand.
Patch 3:       Zero-incremental LHS + storage cap clipping (replaces return block).
"""


# ═══════════════════════════════════════════════════════════════════════
# PATCH 1a: _inline_storage_stage returns leaked SOC
# ═══════════════════════════════════════════════════════════════════════

PATCH1A_OLD = '''\
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
                gap[h] -= discharge'''

PATCH1A_NEW = '''\
@njit
def _inline_storage_stage(surplus, gap, total_dispatch,
                          capacity, power_rating, efficiency, window_hours):
    """Returns total leaked SOC (charged but undischarged at window resets)."""
    if capacity <= 0.0:
        return nb_f32(0.0)
    leaked = nb_f32(0.0)
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
        leaked += nb_f32(soc)
    return leaked'''


# ═══════════════════════════════════════════════════════════════════════
# PATCH 1b: _inline_storage_peaker returns residual SOC
# ═══════════════════════════════════════════════════════════════════════

PATCH1B_OLD = '''\
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
            gap[h] -= discharge'''

PATCH1B_NEW = '''\
@njit
def _inline_storage_peaker(surplus, gap, total_dispatch,
                           capacity, power_rating, efficiency):
    """Returns residual SOC at end of year."""
    if capacity <= 0.0:
        return nb_f32(0.0)
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
    return nb_f32(soc)'''


# ═══════════════════════════════════════════════════════════════════════
# PATCH 1c: Storage calls capture leaked SOC return values
# ═══════════════════════════════════════════════════════════════════════

PATCH1C_OLD = '''\
        _inline_storage_stage(surplus, gap_arr, total_dispatch,
                              b4_p * 4.0, b4_p, 0.85, 24)
        b8_p = b8_val * 0.01 / 8760.0
        _inline_storage_stage(surplus, gap_arr, total_dispatch,
                              b8_p * 8.0, b8_p, 0.85, 48)
        ld_p = ld_val * 0.01 / 8760.0
        _inline_storage_stage(surplus, gap_arr, total_dispatch,
                              ld_p * 100.0, ld_p, 0.50, 168)
        h2_p = h2_val * 0.01 / 8760.0
        _inline_storage_peaker(surplus, gap_arr, total_dispatch,
                               h2_p * 1000.0, h2_p, 0.35)'''

PATCH1C_NEW = '''\
        leaked_b4 = _inline_storage_stage(surplus, gap_arr, total_dispatch,
                              b4_p * 4.0, b4_p, 0.85, 24)
        b8_p = b8_val * 0.01 / 8760.0
        leaked_b8 = _inline_storage_stage(surplus, gap_arr, total_dispatch,
                              b8_p * 8.0, b8_p, 0.85, 48)
        ld_p = ld_val * 0.01 / 8760.0
        leaked_ld = _inline_storage_stage(surplus, gap_arr, total_dispatch,
                              ld_p * 100.0, ld_p, 0.50, 168)
        h2_p = h2_val * 0.01 / 8760.0
        leaked_h2 = _inline_storage_peaker(surplus, gap_arr, total_dispatch,
                               h2_p * 1000.0, h2_p, 0.35)'''


# ═══════════════════════════════════════════════════════════════════════
# PATCH 1d: Curtailment sum includes leaked SOC
# ═══════════════════════════════════════════════════════════════════════

PATCH1D_OLD = '''\
        curt_sum = 0.0
        for h in range(H):
            curt_sum += surplus[h]'''

PATCH1D_NEW = '''\
        curt_sum = leaked_b4 + leaked_b8 + leaked_ld + leaked_h2
        for h in range(H):
            curt_sum += surplus[h]'''


# ═══════════════════════════════════════════════════════════════════════
# PATCH 2a: Module-level storage cap variable
# ═══════════════════════════════════════════════════════════════════════

PATCH2A_OLD = '''\
STORAGE_MIN_HEADROOM_ARR = np.array([
    STORAGE_MIN_HEADROOM[sc] for sc in STORAGE_COLS
], dtype=np.float64)'''

PATCH2A_NEW = '''\
STORAGE_MIN_HEADROOM_ARR = np.array([
    STORAGE_MIN_HEADROOM[sc] for sc in STORAGE_COLS
], dtype=np.float64)

# Combined storage power cap: 115% of peak demand.
# Set per-ISO at profile load; used in generate_candidates.
_STORAGE_CAP_PCT = 999.0  # default (no cap); overwritten at runtime

# Fraction of LHS candidates per dimension pinned to floor (zero new build).
# With ~8 free dims and 5% per dim, ~40% of tail candidates have one dim
# pinned — enough to explore "skip resource X" without starving the search.
ZERO_INC_FRAC = 0.05'''


# ═══════════════════════════════════════════════════════════════════════
# PATCH 2b: Compute cap at profile load
# ═══════════════════════════════════════════════════════════════════════

PATCH2B_OLD = '''\
    print(f"[adaptive] Profiles loaded. Demand TWh={float(pc.REGIONAL_DEMAND_TWH[iso]):.1f}")'''

PATCH2B_NEW = '''\
    global _STORAGE_CAP_PCT
    _STORAGE_CAP_PCT = float(dn.max() * H * 100.0 * 1.15)
    print(f"[adaptive] Profiles loaded. Demand TWh={float(pc.REGIONAL_DEMAND_TWH[iso]):.1f}"
          f"  storage_cap={_STORAGE_CAP_PCT:.1f}% of avg demand"
          f" (peak/avg={dn.max()*H:.2f})")'''


# ═══════════════════════════════════════════════════════════════════════
# PATCH 3: Zero-incremental LHS + storage cap (replaces return block)
# ═══════════════════════════════════════════════════════════════════════
#
# After LHS sampling produces `raw`, two things happen before return:
#
# (A) Zero-incremental pinning: for each free dimension d, pin a fraction
#     of candidates to floors_all[d]. This guarantees "zero new build of
#     resource d" is explored. Uses the TAIL of the sample array so the
#     bulk of the LHS stratification is preserved.
#
# (B) Storage cap: clip combined storage to 115% of peak demand,
#     rescaling proportionally when over cap.
#
# Both applied to generation AND storage dims, both pathways.

PATCH3_OLD = '''\
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

    return W, batt4, batt8, ldes_arr, h2_arr, raw'''

PATCH3_NEW = '''\
    ceilings_all = np.maximum(ceilings_all, floors_all + 0.01)
    raw = floors_all[None, :] + unit_samples[:n_samples] * (ceilings_all - floors_all)[None, :]

    # ── Zero-incremental exploration ──────────────────────────────────
    # For each free dimension, pin a fraction of candidates to floor
    # (zero new build). Lets the optimizer discover when NOT building a
    # resource is cheaper. Uses the tail of the sample array so the bulk
    # of the LHS stratification is preserved in the leading candidates.
    per_dim = max(1, int(n_samples * ZERO_INC_FRAC))
    pin_start = max(0, n_samples - per_dim * n_dims)
    for d in range(n_dims):
        row_lo = pin_start + d * per_dim
        row_hi = min(row_lo + per_dim, n_samples)
        if row_lo < n_samples:
            raw[row_lo:row_hi, d] = floors_all[d]

    # ── Build output arrays from (possibly pinned) raw ────────────────
    W = np.zeros((n_samples, N_RESOURCES), dtype=np.float32)
    for ri in range(N_RESOURCES):
        W[:, ri] = np.float32(floor_pcts[ri] * 0.01)
    for k, ri in enumerate(free_res_idx):
        W[:, ri] = (raw[:, k] * 0.01).astype(np.float32)

    batt4 = raw[:, n_free + 0].astype(np.float32)
    batt8 = raw[:, n_free + 1].astype(np.float32)
    ldes_arr = raw[:, n_free + 2].astype(np.float32)
    h2_arr = raw[:, n_free + 3].astype(np.float32)

    # ── Storage cap: 115% of peak demand ──────────────────────────────
    # Total storage power can never exceed what the grid can absorb.
    # Rescale proportionally when over cap (preserves relative allocation).
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

    return W, batt4, batt8, ldes_arr, h2_arr, raw'''


# ═══════════════════════════════════════════════════════════════════════
# APPLICATION ORDER
# ═══════════════════════════════════════════════════════════════════════
#
#   1a  _inline_storage_stage → returns leaked SOC
#   1b  _inline_storage_peaker → returns residual SOC
#   1c  Storage calls → capture return values
#   1d  Curtailment sum → add leaked SOC
#   2a  Module-level _STORAGE_CAP_PCT + ZERO_INC_FRAC constants
#   2b  Compute cap at profile load
#   3   Zero-incremental pinning + storage cap clipping (replaces
#       the sampling → return block in generate_candidates)
#
# Expected storage_cap_pct values by ISO:
#   CAISO  197%  (peak/avg 1.71)
#   ERCOT  173%  (peak/avg 1.50)
#   MISO   181%  (peak/avg 1.57)
#   NEISO  226%  (peak/avg 1.97)
#   NYISO  212%  (peak/avg 1.84)
#   PJM    192%  (peak/avg 1.67)
#   SPP    184%  (peak/avg 1.60)
#
# Zero-incremental arithmetic (screen_samples=5000, ~8 free dims):
#   per_dim = int(5000 * 0.05) = 250
#   Total pinned = 250 × 8 = 2000 (40% of samples)
#   pin_start = 5000 - 2000 = 3000
#   Candidates 0–2999: pure LHS (undisturbed)
#   Candidates 3000–3249: dim 0 pinned to floor
#   Candidates 3250–3499: dim 1 pinned to floor
#   ... etc
#
# Smoke test:
#   python scripts/step2_3_reliability_tax_adaptive.py \
#       --iso SPP --pathway A --demand-growth Medium
#
# Verify:
#   1. Storage pcts sum < cap in every snapshot
#   2. curtailment_total at 99.9% >> 0 (hundreds of TWh, not 0.0)
#   3. At early thresholds (<80%), winning mix may have 0 storage
#   4. Storage entry point should align with step 2.4 VRE-only elbow
