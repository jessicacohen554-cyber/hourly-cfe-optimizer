#!/usr/bin/env python3
"""
Zonal LMP Solver — Pipe-and-Bubble Transmission Model
======================================================
Solves hourly zonal market clearing via linear programming with inter-zonal
transfer limits. Each ISO is decomposed into 2-5 zones connected by
transmission interfaces with thermal ratings.

LP formulation:
  minimize  sum(mc[z][u] * g[z][u])  over all zones z, units u
  s.t.
    sum(g[z]) + net_import[z] = demand[z]     for each zone z  (balance)
    0 <= g[z][u] <= cap[z][u]                 for each unit    (capacity)
    -limit[a,b] <= flow[a,b] <= limit[a,b]    for each link    (transfer)

Dual prices on balance constraints = zonal LMPs.

Uses scipy.optimize.linprog with HiGHS solver (~0.5ms per 5-zone problem).
"""

import os
import numpy as np
from scipy.optimize import linprog
from concurrent.futures import ProcessPoolExecutor, as_completed

from pipeline_config import SCARCITY_MODE


def _build_lp_structure(zone_stacks, zone_names, interfaces, transfer_limits):
    """Pre-build LP constraint matrices that are constant across hours.

    Returns a dict with reusable LP components. Only the demand RHS changes
    per hour.

    Variables layout:
        g[0..N-1]  = generation for each unit (across all zones)
        f[0..2I-1] = flow variables: f_pos[i] and f_neg[i] for each interface
                      net flow from a→b = f_pos[i] - f_neg[i]
    """
    # Flatten all units across zones, tracking zone membership
    units = []  # (zone_idx, unit_type, cap_mw, mc)
    zone_unit_ranges = {}  # zone_idx → (start, end) in units list
    for z_idx, zname in enumerate(zone_names):
        start = len(units)
        for unit_type, cap_mw, mc in zone_stacks.get(zname, []):
            units.append((z_idx, unit_type, cap_mw, mc))
        zone_unit_ranges[z_idx] = (start, len(units))

    n_units = len(units)
    n_interfaces = len(interfaces)
    n_flow_vars = 2 * n_interfaces  # f_pos and f_neg per interface
    n_vars = n_units + n_flow_vars

    # Objective: minimize generation cost (flows have zero cost)
    c = np.zeros(n_vars)
    for i, (_, _, _, mc) in enumerate(units):
        c[i] = mc

    # Upper bounds for generator capacity
    bounds = []
    for i, (_, _, cap_mw, _) in enumerate(units):
        bounds.append((0.0, cap_mw))
    # Flow variable bounds: 0 to transfer_limit for each direction
    for iface_idx, (a, b) in enumerate(interfaces):
        limit = transfer_limits.get((a, b), transfer_limits.get((b, a), 0.0))
        bounds.append((0.0, limit))  # f_pos
        bounds.append((0.0, limit))  # f_neg

    # Build interface → zone incidence for balance constraints
    # interface_zones[i] = (from_zone_idx, to_zone_idx)
    zone_name_to_idx = {name: i for i, name in enumerate(zone_names)}
    interface_zones = []
    for a, b in interfaces:
        interface_zones.append((zone_name_to_idx[a], zone_name_to_idx[b]))

    n_zones = len(zone_names)

    # Equality constraint matrix: A_eq @ x = b_eq (demand)
    # One row per zone: sum(gen in zone) + net_import = demand
    A_eq = np.zeros((n_zones, n_vars))

    # Generation columns
    for i, (z_idx, _, _, _) in enumerate(units):
        A_eq[z_idx, i] = 1.0

    # Flow columns: net flow into zone z
    # For interface i (a→b): f_pos imports to b, exports from a
    #   flow a→b = f_pos - f_neg
    #   zone a: -(f_pos - f_neg) = -f_pos + f_neg
    #   zone b: +(f_pos - f_neg) = +f_pos - f_neg
    for iface_idx, (from_z, to_z) in enumerate(interface_zones):
        fpos_col = n_units + 2 * iface_idx
        fneg_col = n_units + 2 * iface_idx + 1
        # From zone loses f_pos, gains f_neg
        A_eq[from_z, fpos_col] = -1.0
        A_eq[from_z, fneg_col] = 1.0
        # To zone gains f_pos, loses f_neg
        A_eq[to_z, fpos_col] = 1.0
        A_eq[to_z, fneg_col] = -1.0

    return {
        'c': c,
        'A_eq': A_eq,
        'bounds': bounds,
        'n_units': n_units,
        'n_interfaces': n_interfaces,
        'n_zones': n_zones,
        'n_vars': n_vars,
        'units': units,
        'interfaces': interfaces,
        'interface_zones': interface_zones,
        'zone_unit_ranges': zone_unit_ranges,
    }


def solve_zonal_dispatch_hour(lp_struct, zone_demand_mw):
    """Solve zonal market clearing for one hour.

    Args:
        lp_struct: Pre-built LP structure from _build_lp_structure().
        zone_demand_mw: np.array of shape (n_zones,) with demand per zone in MW.

    Returns:
        zonal_lmp: np.array (n_zones,) — shadow prices ($/MWh) per zone
        zonal_gen: np.array (n_zones,) — total generation per zone (MW)
        flows: np.array (n_interfaces,) — net flow per interface (MW, positive = a→b)
        success: bool
    """
    b_eq = zone_demand_mw.copy()
    n_zones = lp_struct['n_zones']
    n_units = lp_struct['n_units']
    n_interfaces = lp_struct['n_interfaces']

    result = linprog(
        c=lp_struct['c'],
        A_eq=lp_struct['A_eq'],
        b_eq=b_eq,
        bounds=lp_struct['bounds'],
        method='highs',
        options={'presolve': True, 'time_limit': 0.1},
    )

    if not result.success:
        # Infeasible — return NaN LMPs (demand exceeds total capacity + imports)
        return (np.full(n_zones, np.nan),
                np.zeros(n_zones),
                np.zeros(n_interfaces),
                False)

    # Extract generation per zone
    x = result.x
    zonal_gen = np.zeros(n_zones)
    for i, (z_idx, _, _, _) in enumerate(lp_struct['units']):
        zonal_gen[z_idx] += x[i]

    # Extract net flows
    flows = np.zeros(n_interfaces)
    for iface_idx in range(n_interfaces):
        fpos = x[n_units + 2 * iface_idx]
        fneg = x[n_units + 2 * iface_idx + 1]
        flows[iface_idx] = fpos - fneg

    # Zonal LMPs from dual prices on equality constraints
    # scipy linprog (HiGHS): eqlin.marginals are shadow prices on balance constraints.
    # For a cost-minimization LP, the shadow price on demand = marginal cost of
    # serving one more MW in that zone = the zonal LMP (positive values).
    if hasattr(result, 'eqlin') and result.eqlin is not None:
        zonal_lmp = result.eqlin.marginals.copy()
    else:
        # Fallback: approximate LMP from marginal unit cost per zone
        zonal_lmp = _approximate_zonal_lmp(lp_struct, x, zone_demand_mw)

    return zonal_lmp, zonal_gen, flows, True


def _approximate_zonal_lmp(lp_struct, x, zone_demand_mw):
    """Approximate zonal LMP when dual prices unavailable.

    Uses the marginal cost of the most expensive dispatched unit in each zone.
    Vectorized per-zone: boolean mask + np.max instead of inner Python loop.
    """
    n_zones = lp_struct['n_zones']
    zonal_lmp = np.zeros(n_zones)
    units = lp_struct['units']
    # Pre-extract marginal costs as array for vectorized masking
    mc_arr = np.array([u[3] for u in units])

    for z_idx in range(n_zones):
        start, end = lp_struct['zone_unit_ranges'].get(z_idx, (0, 0))
        if start >= end:
            continue
        zone_x = x[start:end]
        zone_mc = mc_arr[start:end]
        dispatched_mask = zone_x > 0.1
        if np.any(dispatched_mask):
            zonal_lmp[z_idx] = np.max(zone_mc[dispatched_mask])

    return zonal_lmp


def _solve_hour_batch(lp_c, lp_A_eq, lp_bounds, lp_units_arr, n_zones,
                      n_units, n_interfaces, zone_demand_batch, hour_indices):
    """Solve zonal LP for a batch of hours. Top-level function for ProcessPoolExecutor.

    Args:
        lp_c, lp_A_eq, lp_bounds: LP structure components (pickle-friendly)
        lp_units_arr: numpy array of (zone_idx, mc) per unit for gen extraction
        n_zones, n_units, n_interfaces: LP dimensions
        zone_demand_batch: np.array (n_zones, n_hours_batch) demand per zone
        hour_indices: np.array of hour indices in the batch

    Returns:
        (lmp_batch, flows_batch, infeasible_count) — arrays for this batch
    """
    n_batch = len(hour_indices)
    lmp_batch = np.zeros((n_zones, n_batch))
    gen_batch = np.zeros((n_zones, n_batch))
    flows_batch = np.zeros((n_interfaces, n_batch))
    infeasible = 0

    for local_h in range(n_batch):
        b_eq = zone_demand_batch[:, local_h].copy()

        result = linprog(
            c=lp_c, A_eq=lp_A_eq, b_eq=b_eq, bounds=lp_bounds,
            method='highs', options={'presolve': True, 'time_limit': 0.1},
        )

        if not result.success:
            infeasible += 1
            lmp_batch[:, local_h] = 500.0  # Scarcity cap
            continue

        x = result.x

        # Extract per-zone generation
        for z_idx in range(n_zones):
            z_start, z_end = int(lp_units_arr[z_idx, 0]), int(lp_units_arr[z_idx, 1])
            if z_start < z_end:
                gen_batch[z_idx, local_h] = np.sum(x[z_start:z_end])

        # Extract net flows
        for iface_idx in range(n_interfaces):
            fpos = x[n_units + 2 * iface_idx]
            fneg = x[n_units + 2 * iface_idx + 1]
            flows_batch[iface_idx, local_h] = fpos - fneg

        # Zonal LMPs from dual prices
        if hasattr(result, 'eqlin') and result.eqlin is not None:
            lmp_batch[:, local_h] = result.eqlin.marginals
        else:
            # Fallback: marginal cost of most expensive dispatched unit per zone
            for z_idx in range(n_zones):
                z_start, z_end = int(lp_units_arr[z_idx, 0]), int(lp_units_arr[z_idx, 1])
                if z_start >= z_end:
                    continue
                zone_x = x[z_start:z_end]
                zone_mc = lp_units_arr[z_idx, 2:2 + (z_end - z_start)]
                mask = zone_x > 0.1
                if np.any(mask):
                    lmp_batch[z_idx, local_h] = np.max(zone_mc[mask])

    return lmp_batch, gen_batch, flows_batch, infeasible


def compute_zonal_lmp_hourly(iso, zone_config, zone_stacks, demand_mw_profile,
                              clean_supply_by_zone=None, price_model=None,
                              vre_penetration=None,
                              full_demand_mw_profile=None):
    """Compute zonal LMPs for all 8760 hours.

    Args:
        iso: ISO name
        zone_config: Dict from ZONE_CONFIG[iso]
        zone_stacks: Dict {zone_name: [(unit_type, cap_mw, mc), ...]}
        demand_mw_profile: np.array (8760,) — demand for the LP solver (MW).
            When zone stacks are fossil-only, this should be residual demand
            after clean dispatch. The LP matches this demand with fossil gen.
        clean_supply_by_zone: Optional dict {zone_name: np.array (8760,)} of
            clean generation in MW per zone. If None, clean supply is split
            proportionally to demand shares.
        price_model: PriceModel instance for post-LP pricing adjustments
        vre_penetration: VRE share for pricing layer adjustments
        full_demand_mw_profile: np.array (8760,) — total system demand (MW)
            used for demand-percentile pricing layers (congestion adders,
            low-demand depression). If None, defaults to demand_mw_profile.
            Must be full demand (not residual) for correct calibration.

    Returns:
        zonal_lmp_matrix: np.array (n_zones, H) — $/MWh per zone per hour
        system_lmp: np.array (H,) — demand-weighted system average
        flows_matrix: np.array (n_interfaces, H) — net flows per interface
        zonal_stats: dict with per-zone summary statistics
    """
    # Pricing layers need full demand for percentile ranking;
    # LP needs residual demand for dispatch. Separate the two signals.
    if full_demand_mw_profile is None:
        full_demand_mw_profile = demand_mw_profile
    zone_names = zone_config['zones']
    demand_shares = zone_config['demand_share']
    transfer_limits = zone_config['transfer_limits_mw']
    n_zones = len(zone_names)
    H = len(demand_mw_profile)

    # Build interface list from transfer_limits keys
    interfaces = list(transfer_limits.keys())
    n_interfaces = len(interfaces)

    # Pre-build LP structure (constant across hours)
    lp_struct = _build_lp_structure(zone_stacks, zone_names, interfaces,
                                     transfer_limits)

    # Compute zonal demand profiles
    zone_demand_shares = np.array([demand_shares[z] for z in zone_names])
    zone_demand_all = np.outer(zone_demand_shares, demand_mw_profile)  # (n_zones, H)

    # Subtract clean supply per zone to get fossil residual demand
    if clean_supply_by_zone is not None:
        for z_idx, zname in enumerate(zone_names):
            if zname in clean_supply_by_zone:
                zone_demand_all[z_idx] -= clean_supply_by_zone[zname]
                # Floor at zero — excess clean supply = curtailment
                zone_demand_all[z_idx] = np.maximum(zone_demand_all[z_idx], 0.0)

    # Compute per-zone total fossil capacity (for ORDC reserve calculation)
    zone_capacity = np.zeros(n_zones)
    for z_idx, zname in enumerate(zone_names):
        for unit_type, cap_mw, mc in zone_stacks.get(zname, []):
            zone_capacity[z_idx] += cap_mw

    # Solve LP for each hour — parallel batches when beneficial
    zonal_lmp_matrix = np.zeros((n_zones, H))
    zonal_gen_matrix = np.zeros((n_zones, H))
    flows_matrix = np.zeros((n_interfaces, H))
    infeasible_hours = 0

    # Pre-filter: identify hours that actually need LP solving
    demand_sums = zone_demand_all.sum(axis=0)
    active_hours = np.where(demand_sums >= 1.0)[0]
    # Hours with ~zero demand get LMP=0 (already initialized to zero)

    use_parallel = (
        len(active_hours) >= 1000
        and os.environ.get('DISABLE_PARALLEL_LP', '0') != '1'
    )

    if use_parallel:
        # Parallel batch LP solving via ProcessPoolExecutor
        # Prepare pickle-friendly LP data (no dicts-of-tuples)
        lp_c = lp_struct['c']
        lp_A_eq = lp_struct['A_eq']
        lp_bounds = lp_struct['bounds']

        # Build zone_unit_ranges as a simple array for the batch solver
        # We pass the approximate_zonal_lmp data inline since it's a rare fallback
        max_units_per_zone = max(
            (end - start) for start, end in lp_struct['zone_unit_ranges'].values()
        ) if lp_struct['zone_unit_ranges'] else 0
        # Pack zone ranges + marginal costs into a 2D array
        units_info = np.zeros((n_zones, 2 + max_units_per_zone))
        mc_arr = np.array([u[3] for u in lp_struct['units']])
        for z_idx in range(n_zones):
            start, end = lp_struct['zone_unit_ranges'].get(z_idx, (0, 0))
            units_info[z_idx, 0] = start
            units_info[z_idx, 1] = end
            if end > start:
                units_info[z_idx, 2:2 + (end - start)] = mc_arr[start:end]

        n_units_lp = lp_struct['n_units']
        chunk_size = 500
        n_active = len(active_hours)
        chunks = [
            active_hours[i:i + chunk_size]
            for i in range(0, n_active, chunk_size)
        ]

        max_workers = min(os.cpu_count() or 4, 8, len(chunks))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for chunk_hrs in chunks:
                demand_batch = np.ascontiguousarray(zone_demand_all[:, chunk_hrs])
                fut = executor.submit(
                    _solve_hour_batch,
                    lp_c, lp_A_eq, lp_bounds, units_info,
                    n_zones, n_units_lp, n_interfaces,
                    demand_batch, chunk_hrs,
                )
                futures[fut] = chunk_hrs

            for fut in as_completed(futures):
                chunk_hrs = futures[fut]
                lmp_batch, gen_batch, flows_batch, infeas = fut.result()
                zonal_lmp_matrix[:, chunk_hrs] = lmp_batch
                zonal_gen_matrix[:, chunk_hrs] = gen_batch
                flows_matrix[:, chunk_hrs] = flows_batch
                infeasible_hours += infeas
    else:
        # Sequential fallback (small problem or parallel disabled)
        for h in active_hours:
            zone_demand_h = zone_demand_all[:, h]

            zonal_lmp, zonal_gen, flows, success = solve_zonal_dispatch_hour(
                lp_struct, zone_demand_h)

            if success:
                zonal_lmp_matrix[:, h] = zonal_lmp
                zonal_gen_matrix[:, h] = zonal_gen
                flows_matrix[:, h] = flows
            else:
                infeasible_hours += 1
                zonal_lmp_matrix[:, h] = 500.0  # Scarcity cap

    # Apply post-LP pricing adjustments if price_model provided
    # Use full_demand_mw_profile (not residual) for demand-percentile
    # calibration — congestion adders and low-demand depression are
    # calibrated against total system load, not fossil residual.
    if price_model is not None:
        zonal_lmp_matrix = _apply_pricing_layers(
            zonal_lmp_matrix, full_demand_mw_profile, zone_demand_shares,
            price_model, vre_penetration,
            zone_capacity=zone_capacity,
            zonal_gen_matrix=zonal_gen_matrix)

    # Compute demand-weighted system average LMP
    system_lmp = np.sum(zonal_lmp_matrix * zone_demand_shares[:, np.newaxis],
                        axis=0)

    # Compute per-zone statistics (including inter-zonal congestion metrics)
    zonal_stats = _compute_zonal_stats(zonal_lmp_matrix, system_lmp,
                                        zone_names, demand_mw_profile,
                                        flows_matrix=flows_matrix,
                                        interfaces=interfaces,
                                        transfer_limits=transfer_limits)

    return zonal_lmp_matrix, system_lmp, flows_matrix, zonal_stats


def _apply_pricing_layers(zonal_lmp_matrix, demand_mw_profile,
                           zone_demand_shares, price_model, vre_penetration,
                           zone_capacity=None, zonal_gen_matrix=None):
    """Apply pricing adjustments per zone.

    Supports two scarcity pricing modes (controlled by pipeline_config.SCARCITY_MODE):
      - 'ordc': ORDC reserve-margin-responsive scarcity pricing per zone.
                Computes reserves = zone_capacity - zone_dispatched, then applies
                VOLL × LOLP(reserves) adder. Skips demand-quantile scarcity overlay.
      - 'demand_quantile': Legacy demand-percentile scarcity overlay (backward compatible).

    In both modes, the high-demand congestion adder and low-demand surplus depression
    are always applied — these represent congestion and surplus, not scarcity.
    """
    n_zones, H = zonal_lmp_matrix.shape
    use_ordc = (SCARCITY_MODE == 'ordc'
                and zone_capacity is not None
                and zonal_gen_matrix is not None)

    for z_idx in range(n_zones):
        zone_demand = demand_mw_profile * zone_demand_shares[z_idx]
        zone_lmp = zonal_lmp_matrix[z_idx]

        # Demand percentile ranking for this zone
        sorted_demand = np.sort(zone_demand)
        percentile_rank = np.searchsorted(sorted_demand, zone_demand) / H

        # --- HIGH-DEMAND CONGESTION ADDER (both modes) ---
        high_mask = percentile_rank > price_model.dq_high_percentile / 100.0
        if np.any(high_mask):
            frac = ((percentile_rank[high_mask] -
                     price_model.dq_high_percentile / 100.0) /
                    (1.0 - price_model.dq_high_percentile / 100.0))
            adder = price_model.dq_high_max_adder * frac ** 2
            zone_lmp[high_mask] += adder

        # --- SCARCITY PRICING: ORDC or DEMAND-QUANTILE ---
        if use_ordc:
            # Per-zone reserves: total zone capacity minus dispatched generation
            zone_reserves = np.maximum(
                zone_capacity[z_idx] - zonal_gen_matrix[z_idx], 0.0)
            # ORDC adder: VOLL × LOLP(reserves) — vectorized sigmoid
            ordc_adder = price_model.compute_ordc_adder(zone_reserves)
            # Only apply where generation is positive (fossil dispatching)
            pos_gen_mask = zonal_gen_matrix[z_idx] > 0.1
            zone_lmp[pos_gen_mask] += ordc_adder[pos_gen_mask]
        else:
            # Legacy demand-quantile scarcity tail
            scarcity_mask = percentile_rank > price_model.dq_scarcity_percentile / 100.0
            if np.any(scarcity_mask):
                frac = ((percentile_rank[scarcity_mask] -
                         price_model.dq_scarcity_percentile / 100.0) /
                        (1.0 - price_model.dq_scarcity_percentile / 100.0))
                zone_lmp[scarcity_mask] += price_model.dq_scarcity_max * frac

        # --- LOW-DEMAND SURPLUS DEPRESSION (both modes) ---
        low_mask = percentile_rank < price_model.dq_low_percentile / 100.0
        if np.any(low_mask):
            frac = ((price_model.dq_low_percentile / 100.0 -
                     percentile_rank[low_mask]) /
                    (price_model.dq_low_percentile / 100.0))
            floor = price_model.dq_low_floor
            if vre_penetration is not None and vre_penetration > 0.25:
                excess = min(vre_penetration - 0.25, 0.25)
                floor *= (1.0 + 0.6 * excess / 0.25)
            depression = frac * abs(floor)
            zone_lmp[low_mask] -= depression

        zonal_lmp_matrix[z_idx] = zone_lmp

    return zonal_lmp_matrix


def _compute_zonal_stats(zonal_lmp_matrix, system_lmp, zone_names,
                          demand_mw_profile, flows_matrix=None,
                          interfaces=None, transfer_limits=None):
    """Compute per-zone LMP summary statistics and inter-zonal congestion."""
    n_zones = zonal_lmp_matrix.shape[0]
    H = len(demand_mw_profile)

    # Peak hours: weekday 7am-11pm (simplified: hours 7-22 on weekdays)
    # Approximate: hours 0-23 repeat 365 times
    hour_of_day = np.tile(np.arange(24), 365)[:H]
    # Simplification: assume first 5 of every 7 days are weekdays
    day_of_week = np.repeat(np.arange(365), 24)[:H] % 7
    peak_mask = (hour_of_day >= 7) & (hour_of_day <= 22) & (day_of_week < 5)
    offpeak_mask = ~peak_mask

    system_avg = np.mean(system_lmp)
    stats = {}
    for z_idx, zname in enumerate(zone_names):
        zlmp = zonal_lmp_matrix[z_idx]
        stats[zname] = {
            'zone_name': zname,
            'avg_lmp': float(np.mean(zlmp)),
            'peak_lmp': float(np.mean(zlmp[peak_mask])) if peak_mask.any() else 0.0,
            'offpeak_lmp': float(np.mean(zlmp[offpeak_mask])) if offpeak_mask.any() else 0.0,
            'p10_lmp': float(np.percentile(zlmp, 10)),
            'p90_lmp': float(np.percentile(zlmp, 90)),
            'price_spread_vs_system': float(np.mean(zlmp) - system_avg),
        }

    # ── Inter-zonal congestion metrics ──────────────────────────────────
    # Compute pairwise LMP spreads and flow utilization for each interface
    zone_name_to_idx = {z: i for i, z in enumerate(zone_names)}
    interface_stats = []
    max_spread_p50 = 0.0
    max_spread_pair = None
    total_congested_hours = 0

    if interfaces and len(interfaces) > 0:
        for iface_idx, (zone_a, zone_b) in enumerate(interfaces):
            za_idx = zone_name_to_idx.get(zone_a)
            zb_idx = zone_name_to_idx.get(zone_b)
            if za_idx is None or zb_idx is None:
                continue

            # Hourly LMP spread between zone pair (absolute)
            hourly_spread = np.abs(
                zonal_lmp_matrix[za_idx] - zonal_lmp_matrix[zb_idx])
            spread_p50 = float(np.median(hourly_spread))
            spread_avg = float(np.mean(hourly_spread))
            spread_p90 = float(np.percentile(hourly_spread, 90))

            # Flow utilization (fraction of transfer limit used)
            iface_stat = {
                'zone_a': zone_a,
                'zone_b': zone_b,
                'spread_avg': round(spread_avg, 2),
                'spread_p50': round(spread_p50, 2),
                'spread_p90': round(spread_p90, 2),
            }

            if (flows_matrix is not None and transfer_limits
                    and iface_idx < flows_matrix.shape[0]):
                limit_mw = transfer_limits.get((zone_a, zone_b), 1.0)
                if limit_mw > 0:
                    abs_flows = np.abs(flows_matrix[iface_idx])
                    utilization = abs_flows / limit_mw
                    iface_stat['avg_utilization_pct'] = round(
                        float(np.mean(utilization)) * 100, 1)
                    iface_stat['hours_above_70pct'] = int(
                        np.sum(utilization > 0.70))
                    iface_stat['hours_above_95pct'] = int(
                        np.sum(utilization > 0.95))
                    total_congested_hours = max(
                        total_congested_hours,
                        iface_stat['hours_above_70pct'])

            interface_stats.append(iface_stat)

            if spread_p50 > max_spread_p50:
                max_spread_p50 = spread_p50
                max_spread_pair = f"{zone_a}-{zone_b}"

    stats['_congestion'] = {
        'interfaces': interface_stats,
        'max_spread_p50': round(max_spread_p50, 2),
        'max_spread_pair': max_spread_pair,
        'max_congested_hours': total_congested_hours,
    }

    return stats
