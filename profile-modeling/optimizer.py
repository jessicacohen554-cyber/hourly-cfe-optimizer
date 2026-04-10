"""
Cost optimizer — finds minimum-cost resource mix for a given CFE target.

Supports user-selected resource portfolios (max ~6 resources).
Uses scipy.optimize.differential_evolution with Numba-accelerated dispatch.

Resource selection examples:
  ['wind', 'solar']                          — RE only, no storage
  ['wind', 'solar', 'battery4']              — RE + short storage
  ['wind', 'solar', 'battery4', 'clean_firm'] — RE + storage + nuclear
  ['wind', 'solar', 'battery4', 'ldes']      — RE + short + long storage
"""

import numpy as np
from scipy.optimize import differential_evolution

import sys, os
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from config import (
    H, STORAGE_PARAMS, STORAGE_RESOURCES, GENERATION_RESOURCES,
    get_generation_cost, get_storage_cost,
    get_hydro_cap_pct,
)
from dispatch import compute_score_with_storage, compute_score_no_storage

PENALTY = 10_000.0

# Preset resource portfolios for convenience
PRESETS = {
    'wind_solar':           ['wind', 'solar'],
    'wind_solar_bat4':      ['wind', 'solar', 'battery4'],
    'wind_solar_bat4_bat8': ['wind', 'solar', 'battery4', 'battery8'],
    'wind_solar_storage':   ['wind', 'solar', 'battery4', 'ldes'],
    'wind_solar_firm':      ['wind', 'solar', 'battery4', 'clean_firm'],
    'wind_solar_ccs':       ['wind', 'solar', 'battery4', 'ccs_ccgt'],
    'full_re_storage':      ['wind', 'solar', 'battery4', 'battery8', 'ldes'],
    'full_portfolio':       ['wind', 'solar', 'clean_firm', 'battery4', 'ldes'],
}


def resolve_resources(resources_arg):
    """Resolve a resource list — could be a preset name or explicit list."""
    if isinstance(resources_arg, str):
        if resources_arg in PRESETS:
            return PRESETS[resources_arg]
        return [r.strip() for r in resources_arg.split(',')]
    return list(resources_arg)


def _classify_resources(resources):
    """Split resource list into generation and storage."""
    gen = [r for r in resources if r in GENERATION_RESOURCES]
    stor = [r for r in resources if r in STORAGE_RESOURCES]
    unknown = [r for r in resources if r not in gen and r not in stor]
    if unknown:
        raise ValueError(f"Unknown resources: {unknown}. "
                         f"Gen: {GENERATION_RESOURCES}, Storage: {STORAGE_RESOURCES}")
    return gen, stor


def _build_problem(iso, cost_level, resources, supply_shapes):
    """Build optimization problem for selected resources.

    Returns:
        gen_resources, stor_resources, gen_costs, stor_costs,
        gen_shapes, stor_params_list, bounds
    """
    gen_resources, stor_resources = _classify_resources(resources)

    # Filter gen to those with available shapes
    gen_resources = [r for r in gen_resources if r in supply_shapes]

    gen_costs = [get_generation_cost(r, cost_level, iso) for r in gen_resources]
    stor_costs = [get_storage_cost(s, cost_level, iso) for s in stor_resources]

    gen_shapes = np.zeros((len(gen_resources), H), dtype=np.float64)
    for i, r in enumerate(gen_resources):
        gen_shapes[i, :] = supply_shapes[r]

    # Storage params in the canonical dispatch order: bat4, bat8, ldes, h2
    # We always pass all 4 slots to the Numba kernel; unused ones get cap=0.
    stor_order = ['battery4', 'battery8', 'ldes', 'h2']
    stor_params_list = []
    for s in stor_resources:
        p = STORAGE_PARAMS[s]
        stor_params_list.append((s, p['duration'], p['window'], p['rte']))

    # Bounds
    bounds = []
    hydro_cap = get_hydro_cap_pct(iso) / 100.0

    for r in gen_resources:
        if r == 'hydro':
            bounds.append((0.0, max(hydro_cap, 0.001)))
        elif r in ('clean_firm', 'ccs_ccgt', 'geothermal'):
            bounds.append((0.0, 1.0))
        else:
            bounds.append((0.0, 4.0))

    for s in stor_resources:
        if s == 'h2':
            bounds.append((0.0, 0.10))
        elif s == 'ldes':
            bounds.append((0.0, 0.05))
        else:
            bounds.append((0.0, 0.02))

    return (gen_resources, stor_resources, gen_costs, stor_costs,
            gen_shapes, stor_params_list, bounds)


def _make_storage_args(stor_resources, stor_caps_frac, stor_params_list, annual_demand):
    """Map selected storage resources to the 4-slot dispatch kernel args.

    Returns (bat4_cap, b4d, b4w, b4r, bat8_cap, ..., ldes_cap, ..., h2_cap, ...).
    """
    # Defaults: zero capacity for unused slots
    slots = {
        'battery4': (0.0, 4, 24, 0.85),
        'battery8': (0.0, 8, 48, 0.85),
        'ldes':     (0.0, 100, 168, 0.50),
        'h2':       (0.0, 1000, 720, 0.35),
    }

    for i, (name, dur, win, rte) in enumerate(stor_params_list):
        cap = stor_caps_frac[i] * annual_demand if i < len(stor_caps_frac) else 0.0
        slots[name] = (cap, dur, win, rte)

    # Unpack in order
    b4 = slots['battery4']
    b8 = slots['battery8']
    ld = slots['ldes']
    h2 = slots['h2']
    return (b4[0], b4[1], b4[2], b4[3],
            b8[0], b8[1], b8[2], b8[3],
            ld[0], ld[1], ld[2], ld[3],
            h2[0], h2[1], h2[2], h2[3])


def _objective(x, gen_shapes, demand, annual_demand, gen_costs, stor_costs,
               stor_resources, stor_params_list, n_gen, target):
    """Objective: total $/MWh + penalty for missing target."""
    gen_allocs = x[:n_gen]
    stor_caps = x[n_gen:]

    # Build hourly supply
    supply_hourly = np.zeros(H, dtype=np.float64)
    for i in range(n_gen):
        if gen_allocs[i] > 0:
            supply_hourly += gen_allocs[i] * annual_demand * gen_shapes[i]

    # Map to dispatch kernel slots
    storage_args = _make_storage_args(stor_resources, stor_caps,
                                      stor_params_list, annual_demand)

    score = compute_score_with_storage(supply_hourly, demand, *storage_args)

    # Cost
    cost = 0.0
    for i in range(n_gen):
        cost += gen_allocs[i] * gen_costs[i]
    for i in range(len(stor_caps)):
        cost += stor_caps[i] * stor_costs[i]

    if score < target:
        cost += PENALTY * (target - score)

    return cost


def optimize_single(iso, cost_level, target, resources, supply_shapes, demand,
                    maxiter=300, popsize=30, seed=42, tol=0.005):
    """Find cost-optimal mix for (ISO, cost_level, target, resource set).

    Args:
        resources: list of resource names to include (e.g., ['wind', 'solar', 'battery4'])
        supply_shapes: dict of resource → (8760,) shape arrays
        demand: (8760,) demand array

    Returns:
        dict with optimal allocations, costs, and score
    """
    annual_demand = demand.sum()

    (gen_resources, stor_resources, gen_costs, stor_costs,
     gen_shapes, stor_params_list, bounds) = _build_problem(
        iso, cost_level, resources, supply_shapes)

    n_gen = len(gen_resources)
    n_stor = len(stor_resources)
    n_vars = n_gen + n_stor

    if n_vars == 0:
        return None

    args = (gen_shapes, demand, annual_demand, gen_costs, stor_costs,
            stor_resources, stor_params_list, n_gen, target)

    result = differential_evolution(
        _objective, bounds, args=args,
        maxiter=maxiter, popsize=popsize, seed=seed, tol=tol,
        mutation=(0.5, 1.5), recombination=0.9,
        polish=True,
    )

    x = result.x
    gen_allocs = x[:n_gen]
    stor_caps = x[n_gen:]

    # Recompute final score
    supply_hourly = np.zeros(H, dtype=np.float64)
    for i in range(n_gen):
        if gen_allocs[i] > 0:
            supply_hourly += gen_allocs[i] * annual_demand * gen_shapes[i]

    storage_args = _make_storage_args(stor_resources, stor_caps,
                                      stor_params_list, annual_demand)
    final_score = compute_score_with_storage(supply_hourly, demand, *storage_args)

    # Build result
    cost_breakdown = {}
    total_cost = 0.0
    total_gen = 0.0

    for i, r in enumerate(gen_resources):
        alloc = gen_allocs[i]
        c = alloc * gen_costs[i]
        cost_breakdown[r] = {'alloc_pct': alloc * 100, 'cost_per_mwh': c,
                             'unit_cost': gen_costs[i]}
        total_cost += c
        total_gen += alloc

    for i, s in enumerate(stor_resources):
        cap = stor_caps[i]
        c = cap * stor_costs[i]
        cap_mwh = cap * annual_demand
        cost_breakdown[s] = {'cap_pct': cap * 100, 'cap_mwh': cap_mwh,
                             'cost_per_mwh': c, 'unit_cost': stor_costs[i]}
        total_cost += c

    return {
        'iso': iso,
        'cost_level': cost_level,
        'target': target,
        'score': final_score,
        'total_cost': total_cost,
        'total_gen_pct': total_gen * 100,
        'resources': cost_breakdown,
        'resource_list': resources,
        'gen_resources': gen_resources,
        'stor_resources': stor_resources,
        'converged': result.success,
    }


def optimize_sweep(iso, cost_level, targets, resources, supply_shapes, demand,
                   maxiter=300, popsize=30, seed=42, verbose=True):
    """Run optimization across multiple targets for a fixed resource set.

    Returns list of result dicts.
    """
    results = []
    sorted_targets = sorted(targets)

    for i, target in enumerate(sorted_targets):
        if verbose:
            print(f"    {target*100:5.1f}% ... ", end='', flush=True)

        res = optimize_single(
            iso, cost_level, target, resources, supply_shapes, demand,
            maxiter=maxiter, popsize=popsize, seed=seed + i,
        )

        if res:
            if verbose:
                score_str = f"{res['score']*100:.2f}%"
                cost_str = f"${res['total_cost']:.2f}/MWh"
                hit = "OK" if res['score'] >= target - 0.001 else "MISS"
                print(f"{score_str} → {cost_str}  [{hit}]")
            results.append(res)
        else:
            if verbose:
                print("SKIP (no resources)")

    return results
