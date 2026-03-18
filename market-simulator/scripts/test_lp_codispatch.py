#!/usr/bin/env python3
"""Validation tests for LP co-optimized storage dispatch.

Tests:
1. Single-type dispatch: LP and greedy produce identical results
2. Multi-type LP >= greedy on gap reduction
3. Energy conservation per storage type
4. SOC bounds never violated
5. Residual arrays consistent after dispatch
"""

import sys
import time
import numpy as np

# Ensure scripts dir is on path
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dispatch_utils import (
    reconstruct_hourly_dispatch,
    co_dispatch_storage_lp,
    _build_active_storage_params,
    _count_active_storage,
    _solve_storage_window_lp,
    H,
    BATTERY_EFFICIENCY, BATTERY_DURATION_HOURS,
    BATTERY8_EFFICIENCY, BATTERY8_DURATION_HOURS,
    LDES_EFFICIENCY, LDES_DURATION_HOURS, LDES_WINDOW_DAYS,
    H2_EFFICIENCY, H2_DURATION_HOURS, H2_WINDOW_DAYS,
)

TOLERANCE = 1e-4  # Numerical tolerance for comparisons


def make_synthetic_profiles(seed=42):
    """Generate synthetic demand and supply profiles for testing."""
    rng = np.random.RandomState(seed)

    # Demand: sinusoidal daily pattern + noise
    hours = np.arange(H)
    demand = 0.8 + 0.3 * np.sin(2 * np.pi * hours / 24) + 0.05 * rng.randn(H)
    demand = np.clip(demand, 0.3, 1.5)

    # Solar: peaks midday, zero at night
    hour_of_day = hours % 24
    solar = np.where((hour_of_day >= 6) & (hour_of_day <= 18),
                     np.sin(np.pi * (hour_of_day - 6) / 12) * 1.2 + 0.05 * rng.randn(H),
                     0.0)
    solar = np.clip(solar, 0, 2)

    # Wind: variable
    wind = 0.3 + 0.4 * np.sin(2 * np.pi * hours / (24 * 3)) + 0.15 * rng.randn(H)
    wind = np.clip(wind, 0, 1.5)

    # Build supply profiles dict matching RESOURCE_TYPES
    supply_profiles = {
        'clean_firm': [0.3] * H,  # flat baseload
        'solar': solar.tolist(),
        'wind': wind.tolist(),
        'offshore_wind': [0.0] * H,
        'ccs_ccgt': [0.0] * H,
        'hydro': [0.1] * H,
    }

    # Resource mix: 40% solar, 30% wind, 10% clean_firm, 5% hydro
    resource_pcts = {
        'clean_firm': 10,
        'solar': 40,
        'wind': 30,
        'offshore_wind': 0,
        'ccs_ccgt': 0,
        'hydro': 5,
    }

    return demand.tolist(), supply_profiles, resource_pcts


def test_single_type_equivalence():
    """When only one storage type is active, LP falls back to greedy — results should match."""
    print("\n=== Test 1: Single-type equivalence (LP falls back to greedy) ===")
    demand, profiles, pcts = make_synthetic_profiles()

    # Battery4 only
    r_greedy = reconstruct_hourly_dispatch(
        demand, profiles, pcts,
        battery_dispatch_pct=5, battery8_dispatch_pct=0,
        ldes_dispatch_pct=0, h2_dispatch_pct=0,
        dispatch_mode='greedy')

    r_lp = reconstruct_hourly_dispatch(
        demand, profiles, pcts,
        battery_dispatch_pct=5, battery8_dispatch_pct=0,
        ldes_dispatch_pct=0, h2_dispatch_pct=0,
        dispatch_mode='lp')

    # With only 1 active type, LP should fall back to greedy (identical results)
    diff = np.max(np.abs(r_greedy['battery4_profile'] - r_lp['battery4_profile']))
    print(f"  Battery4 max diff: {diff:.2e}")
    assert diff < TOLERANCE, f"Single-type battery4 should be identical, got diff={diff}"

    # LDES only
    r_greedy = reconstruct_hourly_dispatch(
        demand, profiles, pcts,
        battery_dispatch_pct=0, battery8_dispatch_pct=0,
        ldes_dispatch_pct=5, h2_dispatch_pct=0,
        dispatch_mode='greedy')

    r_lp = reconstruct_hourly_dispatch(
        demand, profiles, pcts,
        battery_dispatch_pct=0, battery8_dispatch_pct=0,
        ldes_dispatch_pct=5, h2_dispatch_pct=0,
        dispatch_mode='lp')

    diff = np.max(np.abs(r_greedy['ldes_profile'] - r_lp['ldes_profile']))
    print(f"  LDES max diff: {diff:.2e}")
    assert diff < TOLERANCE, f"Single-type LDES should be identical, got diff={diff}"

    print("  PASSED")


def test_lp_improves_gap_reduction():
    """LP co-dispatch should reduce at least as much gap as greedy for multi-type."""
    print("\n=== Test 2: LP >= greedy on gap reduction (multi-type) ===")
    demand, profiles, pcts = make_synthetic_profiles()

    # Multi-type: battery4 + battery8 + LDES
    r_greedy = reconstruct_hourly_dispatch(
        demand, profiles, pcts,
        battery_dispatch_pct=3, battery8_dispatch_pct=3,
        ldes_dispatch_pct=5, h2_dispatch_pct=0,
        dispatch_mode='greedy')

    r_lp = reconstruct_hourly_dispatch(
        demand, profiles, pcts,
        battery_dispatch_pct=3, battery8_dispatch_pct=3,
        ldes_dispatch_pct=5, h2_dispatch_pct=0,
        dispatch_mode='lp')

    greedy_gap = np.sum(r_greedy['residual_demand'])
    lp_gap = np.sum(r_lp['residual_demand'])
    greedy_total_storage = (np.sum(r_greedy['battery4_profile']) +
                            np.sum(r_greedy['battery8_profile']) +
                            np.sum(r_greedy['ldes_profile']))
    lp_total_storage = (np.sum(r_lp['battery4_profile']) +
                        np.sum(r_lp['battery8_profile']) +
                        np.sum(r_lp['ldes_profile']))

    improvement_pct = (greedy_gap - lp_gap) / greedy_gap * 100 if greedy_gap > 0 else 0

    print(f"  Greedy residual gap:  {greedy_gap:.4f}")
    print(f"  LP residual gap:      {lp_gap:.4f}")
    print(f"  Improvement:          {improvement_pct:.2f}%")
    print(f"  Greedy total storage: {greedy_total_storage:.4f}")
    print(f"  LP total storage:     {lp_total_storage:.4f}")

    # LP should be at least as good (within tolerance)
    assert lp_gap <= greedy_gap + TOLERANCE, \
        f"LP gap ({lp_gap}) should be <= greedy gap ({greedy_gap})"
    print("  PASSED")


def test_energy_conservation():
    """For each storage type, discharge energy <= charge energy * RTE."""
    print("\n=== Test 3: Energy conservation ===")
    demand, profiles, pcts = make_synthetic_profiles()

    r = reconstruct_hourly_dispatch(
        demand, profiles, pcts,
        battery_dispatch_pct=3, battery8_dispatch_pct=3,
        ldes_dispatch_pct=5, h2_dispatch_pct=0,
        detailed=True, dispatch_mode='lp')

    checks = [
        ('battery4', BATTERY_EFFICIENCY),
        ('battery8', BATTERY8_EFFICIENCY),
        ('ldes', LDES_EFFICIENCY),
    ]

    for name, rte in checks:
        total_charge = np.sum(r[f'{name}_charge'])
        total_discharge = np.sum(r[f'{name}_profile'])
        max_discharge = total_charge * rte
        print(f"  {name}: charge={total_charge:.4f}, discharge={total_discharge:.4f}, "
              f"max_allowed={max_discharge:.4f}")
        assert total_discharge <= max_discharge + TOLERANCE, \
            f"{name} violates energy conservation: discharge={total_discharge} > charge*rte={max_discharge}"

    print("  PASSED")


def test_residual_consistency():
    """Residual arrays should be non-negative and consistent with dispatch."""
    print("\n=== Test 4: Residual consistency ===")
    demand, profiles, pcts = make_synthetic_profiles()

    r = reconstruct_hourly_dispatch(
        demand, profiles, pcts,
        battery_dispatch_pct=3, battery8_dispatch_pct=3,
        ldes_dispatch_pct=5, h2_dispatch_pct=0,
        dispatch_mode='lp')

    demand_arr = np.array(demand[:H])
    total_clean = r['total_clean']
    residual = r['residual_demand']

    # Residual should be non-negative
    assert np.all(residual >= -TOLERANCE), "Residual demand has negative values"

    # total_clean + residual should approximate demand (within tolerance for curtailment)
    fossil_displaced = r['fossil_displaced']
    recon = fossil_displaced + residual
    max_err = np.max(np.abs(recon - demand_arr))
    print(f"  Max reconstruction error: {max_err:.2e}")
    assert max_err < TOLERANCE, f"Reconstruction error too large: {max_err}"

    print("  PASSED")


def test_detailed_mode_consistency():
    """Detailed mode should return charge profiles and match non-detailed dispatch."""
    print("\n=== Test 5: Detailed mode consistency ===")
    demand, profiles, pcts = make_synthetic_profiles()

    r_simple = reconstruct_hourly_dispatch(
        demand, profiles, pcts,
        battery_dispatch_pct=3, battery8_dispatch_pct=3,
        ldes_dispatch_pct=5, h2_dispatch_pct=0,
        detailed=False, dispatch_mode='lp')

    r_detailed = reconstruct_hourly_dispatch(
        demand, profiles, pcts,
        battery_dispatch_pct=3, battery8_dispatch_pct=3,
        ldes_dispatch_pct=5, h2_dispatch_pct=0,
        detailed=True, dispatch_mode='lp')

    # Dispatch profiles should match
    for key in ['battery4_profile', 'battery8_profile', 'ldes_profile', 'h2_profile']:
        diff = np.max(np.abs(r_simple[key] - r_detailed[key]))
        print(f"  {key} diff: {diff:.2e}")
        assert diff < TOLERANCE, f"Detailed vs simple mismatch on {key}: {diff}"

    # Charge profiles should exist in detailed
    for name in ['battery4', 'battery8', 'ldes', 'h2']:
        key = f'{name}_charge'
        assert key in r_detailed, f"Missing {key} in detailed result"

    print("  PASSED")


def test_lp_window_solver():
    """Direct test of _solve_storage_window_lp with a simple scenario."""
    print("\n=== Test 6: LP window solver direct test ===")

    W = 24
    # Surplus in first 12 hours, gap in last 12
    surplus = np.zeros(W)
    surplus[:12] = 0.05
    gap = np.zeros(W)
    gap[12:] = 0.04

    # Two storage types: fast (battery-like) and slow (LDES-like)
    capacities = np.array([0.3, 0.5])
    power_ratings = np.array([0.075, 0.005])
    rtes = np.array([0.85, 0.50])
    soc_init = np.array([0.0, 0.0])

    result = _solve_storage_window_lp(surplus, gap, capacities, power_ratings, rtes, soc_init)
    assert result is not None, "LP solver returned None"

    charge, discharge, soc_final = result

    # Total discharge should not exceed total gap
    total_discharge = np.sum(discharge)
    total_gap = np.sum(gap)
    print(f"  Total gap: {total_gap:.4f}, Total discharge: {total_discharge:.4f}")
    assert total_discharge <= total_gap + TOLERANCE

    # Total charge should not exceed total surplus
    total_charge = np.sum(charge)
    total_surplus = np.sum(surplus)
    print(f"  Total surplus: {total_surplus:.4f}, Total charge: {total_charge:.4f}")
    assert total_charge <= total_surplus + TOLERANCE

    # SOC final should be non-negative and <= capacity
    for k in range(2):
        assert soc_final[k] >= -TOLERANCE, f"SOC[{k}] negative: {soc_final[k]}"
        assert soc_final[k] <= capacities[k] + TOLERANCE, f"SOC[{k}] exceeds cap: {soc_final[k]}"

    print("  PASSED")


def test_performance():
    """Benchmark LP vs greedy runtime on full 8760-hour dispatch."""
    print("\n=== Test 7: Performance benchmark ===")
    demand, profiles, pcts = make_synthetic_profiles()

    # Greedy
    t0 = time.time()
    for _ in range(3):
        reconstruct_hourly_dispatch(
            demand, profiles, pcts,
            battery_dispatch_pct=3, battery8_dispatch_pct=3,
            ldes_dispatch_pct=5, h2_dispatch_pct=0,
            dispatch_mode='greedy')
    greedy_time = (time.time() - t0) / 3

    # LP
    t0 = time.time()
    for _ in range(3):
        reconstruct_hourly_dispatch(
            demand, profiles, pcts,
            battery_dispatch_pct=3, battery8_dispatch_pct=3,
            ldes_dispatch_pct=5, h2_dispatch_pct=0,
            dispatch_mode='lp')
    lp_time = (time.time() - t0) / 3

    ratio = lp_time / greedy_time if greedy_time > 0 else float('inf')
    print(f"  Greedy: {greedy_time*1000:.1f}ms")
    print(f"  LP:     {lp_time*1000:.1f}ms")
    print(f"  Ratio:  {ratio:.1f}x")
    print("  (informational — no pass/fail threshold)")


if __name__ == '__main__':
    print("=" * 60)
    print("LP Co-Dispatch Storage Validation Tests")
    print("=" * 60)

    test_single_type_equivalence()
    test_lp_improves_gap_reduction()
    test_energy_conservation()
    test_residual_consistency()
    test_detailed_mode_consistency()
    test_lp_window_solver()
    test_performance()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
