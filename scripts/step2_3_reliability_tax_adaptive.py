#!/usr/bin/env python3
"""
step2_3_scoring_fixes.py  v2.0

Apply these str_replace patches in order against
scripts/step2_3_reliability_tax_adaptive.py.

Patch 1 (a–d): Curtailment tracking — reporting only, no optimization impact.
Patch 2 (a–c): Storage cap — affects candidate generation bounds.
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
# PATCH 2a: Module-level storage cap variable + compute at profile load
# ═══════════════════════════════════════════════════════════════════════
#
# Peak demand is a static property of the ISO demand shape.
# dn is normalised (sum=1 over 8760h), so:
#   peak / average = dn.max() * 8760
# Storage pcts are power as % of average demand, so:
#   peak in storage-pct units = dn.max() * 8760 * 100
# Cap at 115% of that.

# Add module-level default near the top, after STORAGE_MIN_HEADROOM block:

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
_STORAGE_CAP_PCT = 999.0  # default (no cap); overwritten at runtime'''


# Compute and set at profile load time:

PATCH2B_OLD = '''\
    print(f"[adaptive] Profiles loaded. Demand TWh={float(pc.REGIONAL_DEMAND_TWH[iso]):.1f}")'''

PATCH2B_NEW = '''\
    global _STORAGE_CAP_PCT
    _STORAGE_CAP_PCT = float(dn.max() * H * 100.0 * 1.15)
    print(f"[adaptive] Profiles loaded. Demand TWh={float(pc.REGIONAL_DEMAND_TWH[iso]):.1f}"
          f"  storage_cap={_STORAGE_CAP_PCT:.1f}% of avg demand"
          f" (peak/avg={dn.max()*H:.2f})")'''


# ═══════════════════════════════════════════════════════════════════════
# PATCH 2c: Clip total storage after LHS sampling in generate_candidates
# ═══════════════════════════════════════════════════════════════════════

PATCH2C_OLD = '''\
    batt4 = raw[:, n_free + 0].astype(np.float32)
    batt8 = raw[:, n_free + 1].astype(np.float32)
    ldes_arr = raw[:, n_free + 2].astype(np.float32)
    h2_arr = raw[:, n_free + 3].astype(np.float32)

    return W, batt4, batt8, ldes_arr, h2_arr, raw'''

PATCH2C_NEW = '''\
    batt4 = raw[:, n_free + 0].astype(np.float32)
    batt8 = raw[:, n_free + 1].astype(np.float32)
    ldes_arr = raw[:, n_free + 2].astype(np.float32)
    h2_arr = raw[:, n_free + 3].astype(np.float32)

    # Clip combined storage to 115% of peak demand power.
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
#   2a  Module-level _STORAGE_CAP_PCT default
#   2b  Compute cap at profile load
#   2c  Clip storage in generate_candidates
#
# Expected storage_cap_pct values by ISO:
#   CAISO  197%  (peak/avg 1.71)
#   MISO   181%  (peak/avg 1.57)
#   NEISO  226%  (peak/avg 1.97)
#   NYISO  212%  (peak/avg 1.84)
#   PJM    192%  (peak/avg 1.67)
#   SPP    184%  (peak/avg 1.60)
#   ERCOT  173%  (peak/avg 1.50)
#
# Smoke test:
#   python scripts/step2_3_reliability_tax_adaptive.py \
#       --iso SPP --pathway A --demand-growth Medium
#
# Verify:
#   1. All storage_dispatch_pct values sum < cap in every snapshot
#   2. curtailment_total at 99.9% >> 0 (hundreds of TWh, not 0.0)
#   3. CFE trajectory unchanged; cumulative costs drop 10-20%
