#!/usr/bin/env python3
"""Compare analytical vectorized solver vs LP solver for validation."""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline_config import ZONE_CONFIG
from lmp_engine import build_merit_order_stack
from market_simulation import _split_stack_by_zone
from zonal_lmp import (compute_zonal_lmp_hourly, _solve_2zone_vectorized,
                        _solve_3zone_chain_vectorized, _solve_nzone_vectorized,
                        _solve_lp_all_hours, _build_lp_structure)


def compare_solvers(iso, demand_scale=1.0, _preloaded=None):
    """Compare LP vs analytical solver for one ISO."""
    from market_simulation import load_common_data
    from dispatch_utils import get_demand_profile, get_supply_profiles, REGIONAL_DEMAND_TWH

    if _preloaded is None:
        _preloaded = load_common_data()
    demand_data, gen_profiles = _preloaded[0], _preloaded[1]

    demand_norm, total_mwh_base = get_demand_profile(iso, demand_data)
    demand_mw = np.asarray(demand_norm, dtype=np.float64) * total_mwh_base

    # Scale demand
    demand_mw *= demand_scale

    # Build merit stack (40% clean as test baseline)
    stack_result = build_merit_order_stack(iso, 40.0, fuel_level='Medium',
                                           demand_growth_factor=demand_scale)
    stack = stack_result[0] if isinstance(stack_result, tuple) else stack_result
    zone_config = ZONE_CONFIG[iso]
    zone_stacks = _split_stack_by_zone(stack, zone_config)

    zone_names = zone_config['zones']
    transfer_limits = zone_config['transfer_limits_mw']
    interfaces = list(transfer_limits.keys())
    n_zones = len(zone_names)
    n_interfaces = len(interfaces)
    H = len(demand_mw)

    # Compute zone demand
    demand_shares = zone_config['demand_share']
    zone_demand_shares = np.array([demand_shares[z] for z in zone_names])
    zone_demand_all = np.outer(zone_demand_shares, demand_mw)

    # Clean dispatch: simulate 40% clean energy
    clean_pct = 0.4
    clean_supply = demand_mw * clean_pct * (0.5 + 0.5 * np.sin(np.linspace(0, 4*np.pi, H)))
    residual_demand = np.maximum(demand_mw - clean_supply, 0.0)
    zone_residual = np.outer(zone_demand_shares, residual_demand)

    # --- LP solver ---
    lp_struct = _build_lp_structure(zone_stacks, zone_names, interfaces, transfer_limits)
    t0 = time.time()
    lp_lmp, lp_gen, lp_flows, lp_infeas = _solve_lp_all_hours(
        zone_residual, lp_struct, n_zones, n_interfaces, H)
    t_lp = time.time() - t0

    # --- Analytical solver ---
    t0 = time.time()
    if n_zones == 2:
        limit = transfer_limits[interfaces[0]]
        an_lmp, an_gen, an_flows, an_infeas = _solve_2zone_vectorized(
            zone_residual, zone_stacks, zone_names, limit)
    elif n_zones == 3:
        an_lmp, an_gen, an_flows, an_infeas = _solve_3zone_chain_vectorized(
            zone_residual, zone_stacks, zone_names, transfer_limits)
    else:
        an_lmp, an_gen, an_flows, an_infeas = _solve_nzone_vectorized(
            zone_residual, zone_stacks, zone_names, transfer_limits)
    t_an = time.time() - t0

    # Compare
    speedup = t_lp / max(t_an, 0.001)

    # System LMP (demand-weighted average)
    lp_sys = np.sum(lp_lmp * zone_demand_shares[:, np.newaxis], axis=0)
    an_sys = np.sum(an_lmp * zone_demand_shares[:, np.newaxis], axis=0)

    lmp_diff = np.abs(lp_sys - an_sys)
    mean_diff = np.mean(lmp_diff)
    max_diff = np.max(lmp_diff)
    pct_close = np.mean(lmp_diff < 5.0) * 100  # within $5/MWh

    print(f"\n{'='*60}")
    print(f"  {iso} ({n_zones} zones, {n_interfaces} interfaces)")
    print(f"{'='*60}")
    print(f"  LP time:         {t_lp:.3f}s")
    print(f"  Analytical time: {t_an:.3f}s")
    print(f"  Speedup:         {speedup:.0f}×")
    print(f"  LP infeasible:   {lp_infeas} hours")
    print(f"  Analytical infeas: {an_infeas} hours")
    print(f"  LP avg sys LMP:  ${np.mean(lp_sys):.1f}")
    print(f"  AN avg sys LMP:  ${np.mean(an_sys):.1f}")
    print(f"  Mean |diff|:     ${mean_diff:.2f}/MWh")
    print(f"  Max |diff|:      ${max_diff:.2f}/MWh")
    print(f"  Hours within $5: {pct_close:.1f}%")

    # Per-zone comparison
    for z_idx, zname in enumerate(zone_names):
        lp_avg = np.mean(lp_lmp[z_idx])
        an_avg = np.mean(an_lmp[z_idx])
        print(f"  Zone {zname}: LP=${lp_avg:.1f}  AN=${an_avg:.1f}  diff=${an_avg-lp_avg:+.1f}")

    return {
        'iso': iso, 'speedup': speedup,
        'mean_diff': mean_diff, 'max_diff': max_diff,
        'pct_close': pct_close, 't_lp': t_lp, 't_an': t_an,
    }


if __name__ == '__main__':
    from market_simulation import load_common_data
    print("Loading common data...")
    preloaded = load_common_data()
    results = []
    for iso in ['CAISO', 'SPP', 'NEISO', 'NYISO', 'MISO', 'ERCOT', 'PJM']:
        if iso not in ZONE_CONFIG:
            continue
        r = compare_solvers(iso, demand_scale=1.0, _preloaded=preloaded)
        results.append(r)

    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total_lp = sum(r['t_lp'] for r in results)
    total_an = sum(r['t_an'] for r in results)
    print(f"Total LP time:    {total_lp:.1f}s")
    print(f"Total AN time:    {total_an:.3f}s")
    print(f"Overall speedup:  {total_lp/max(total_an, 0.001):.0f}×")
    print(f"Max mean diff:    ${max(r['mean_diff'] for r in results):.2f}/MWh")
    print(f"Min hours close:  {min(r['pct_close'] for r in results):.1f}%")
