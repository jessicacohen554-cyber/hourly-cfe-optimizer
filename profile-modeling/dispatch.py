"""
Numba-accelerated storage dispatch and hourly matching score computation.

Matches the main pipeline's dispatch model:
  - Two-pass per window (charge then discharge)
  - SOC carries across window boundaries
  - RTE applied per discharge event
  - Dispatch order: battery4 → battery8 → LDES → H2
"""

import numpy as np
from numba import njit

H = 8760


@njit(cache=True)
def _battery_dispatch(surplus, gap, capacity, power_rating, rte, window_hours):
    """Single storage dispatch — modifies surplus/gap in-place, returns discharge profile."""
    discharge = np.zeros(H, dtype=np.float64)
    if capacity <= 0.0:
        return discharge

    soc = 0.0
    n_windows = (H + window_hours - 1) // window_hours

    for w in range(n_windows):
        ws = w * window_hours
        we = ws + window_hours
        if we > H:
            we = H

        # Charge pass
        for h in range(ws, we):
            s = surplus[h]
            if s > 0.0 and soc < capacity:
                charge = s
                if charge > power_rating:
                    charge = power_rating
                space = capacity - soc
                if charge > space:
                    charge = space
                soc += charge
                surplus[h] -= charge

        # Discharge pass
        for h in range(ws, we):
            g = gap[h]
            if g > 0.0 and soc > 0.0:
                available = soc * rte
                dis = g
                if dis > power_rating:
                    dis = power_rating
                if dis > available:
                    dis = available
                discharge[h] = dis
                soc -= dis / rte
                gap[h] -= dis

    return discharge


@njit(cache=True)
def compute_score_with_storage(supply_hourly, demand_hourly,
                               bat4_cap, bat4_dur, bat4_win, bat4_rte,
                               bat8_cap, bat8_dur, bat8_win, bat8_rte,
                               ldes_cap, ldes_dur, ldes_win, ldes_rte,
                               h2_cap, h2_dur, h2_win, h2_rte):
    """Compute hourly match score with cascaded storage dispatch.

    All capacities in same units as supply/demand (MWh).
    Returns score as fraction (0.0 to 1.0).
    """
    # Compute initial surplus and gap
    surplus = np.empty(H, dtype=np.float64)
    gap = np.empty(H, dtype=np.float64)
    for h in range(H):
        diff = supply_hourly[h] - demand_hourly[h]
        if diff >= 0:
            surplus[h] = diff
            gap[h] = 0.0
        else:
            surplus[h] = 0.0
            gap[h] = -diff

    # Dispatch in order: battery4 → battery8 → LDES → H2
    power4 = bat4_cap / bat4_dur if bat4_cap > 0 else 0.0
    d4 = _battery_dispatch(surplus, gap, bat4_cap, power4, bat4_rte, bat4_win)

    power8 = bat8_cap / bat8_dur if bat8_cap > 0 else 0.0
    d8 = _battery_dispatch(surplus, gap, bat8_cap, power8, bat8_rte, bat8_win)

    power_l = ldes_cap / ldes_dur if ldes_cap > 0 else 0.0
    dl = _battery_dispatch(surplus, gap, ldes_cap, power_l, ldes_rte, ldes_win)

    power_h = h2_cap / h2_dur if h2_cap > 0 else 0.0
    dh = _battery_dispatch(surplus, gap, h2_cap, power_h, h2_rte, h2_win)

    # Compute matched energy
    total_matched = 0.0
    total_demand = 0.0
    for h in range(H):
        total_clean = supply_hourly[h] + d4[h] + d8[h] + dl[h] + dh[h]
        d = demand_hourly[h]
        total_demand += d
        if total_clean < d:
            total_matched += total_clean
        else:
            total_matched += d

    if total_demand == 0.0:
        return 0.0
    return total_matched / total_demand


@njit(cache=True)
def compute_score_no_storage(supply_hourly, demand_hourly):
    """Compute hourly match score without any storage."""
    matched = 0.0
    demand_total = 0.0
    for h in range(H):
        d = demand_hourly[h]
        demand_total += d
        s = supply_hourly[h]
        if s < d:
            matched += s
        else:
            matched += d
    if demand_total == 0.0:
        return 0.0
    return matched / demand_total


def warmup():
    """Pre-compile all Numba functions."""
    d = np.ones(H, dtype=np.float64)
    s = np.ones(H, dtype=np.float64)
    compute_score_with_storage(s, d, 1.0, 4, 24, 0.85,
                               0.0, 8, 48, 0.85,
                               0.0, 100, 168, 0.50,
                               0.0, 1000, 720, 0.35)
    compute_score_no_storage(s, d)
