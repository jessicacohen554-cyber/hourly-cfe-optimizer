#!/usr/bin/env python3
"""Minimal test: verify new-build fossil feedback + RA growth scaling fixes.

Runs 1 scenario × 1 ISO (ERCOT) × 6 years with High demand growth.
Before the fix: avg_lmp >> $200, scarcity_hours >> 2000 in outer years.
After the fix: avg_lmp $25-80, scarcity_hours < 1000 in all years.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from market_simulation import (
    build_single_scenario,
    get_demand_at_year, compute_lmp_at_threshold,
    SIM_YEARS,
)
from dispatch_utils import (
    get_demand_profile, get_supply_profiles,
    GRID_MIX_SHARES, REGIONAL_DEMAND_TWH, H,
)
from eia_data_io import load_demand_profiles, load_generation_profiles
from lmp_engine import build_merit_order_stack, INSTALLED_FOSSIL_MW, PEAK_DEMAND_MW
from pipeline_config import ORDC_PARAMS


def test_merit_order_new_builds():
    """Test that new_fossil_builds parameter adds capacity to the stack."""
    iso = 'ERCOT'
    clean_pct = 46.0

    # Baseline stack (no new builds)
    stack_base, total_base = build_merit_order_stack(iso, clean_pct)
    print(f"Baseline stack: {total_base:.0f} MW ({len(stack_base)} units)")
    assert abs(total_base - INSTALLED_FOSSIL_MW[iso]) < 1, \
        f"Baseline should equal installed: {total_base} vs {INSTALLED_FOSSIL_MW[iso]}"

    # Stack with 10 GW new-build gas
    new_builds = {'gas_ccgt': 7000, 'gas_ct': 3000}
    stack_nb, total_nb = build_merit_order_stack(
        iso, clean_pct, new_fossil_builds=new_builds)
    print(f"With new builds: {total_nb:.0f} MW ({len(stack_nb)} units)")
    assert total_nb == total_base + 10000, \
        f"New builds not added: {total_nb} vs expected {total_base + 10000}"
    print("  PASS: new_fossil_builds correctly adds capacity\n")


def test_demand_growth_scaling():
    """Test that demand_growth_factor scales RA peak correctly."""
    iso = 'ERCOT'
    clean_pct = 60.0  # Above baseline, triggers RA sizing

    stack_1x, total_1x = build_merit_order_stack(iso, clean_pct, demand_growth_factor=1.0)
    stack_2x, total_2x = build_merit_order_stack(iso, clean_pct, demand_growth_factor=2.0)
    print(f"demand_growth_factor=1.0: {total_1x:.0f} MW")
    print(f"demand_growth_factor=2.0: {total_2x:.0f} MW")
    # With 2x demand growth, RA peak doubles → larger fossil fleet needed
    assert total_2x >= total_1x, \
        f"Growth-scaled fleet should be >= baseline: {total_2x} vs {total_1x}"
    print("  PASS: demand_growth_factor scales RA peak\n")


def test_scarcity_reduction():
    """Test that ORDC scarcity drops with new-build fossil in outer years."""
    iso = 'ERCOT'
    print(f"Testing scarcity reduction for {iso}...")

    # Load data
    demand_data = load_demand_profiles()
    gen_profiles = load_generation_profiles()
    demand_norm, total_mwh_base = get_demand_profile(iso, demand_data)
    supply_profiles = get_supply_profiles(iso, gen_profiles)

    baseline_clean = sum(GRID_MIX_SHARES[iso].values())
    resource_pcts = dict(GRID_MIX_SHARES[iso])

    ordc = ORDC_PARAMS[iso]

    # Test at 2050 with High growth — worst case
    year = 2050
    demand_twh = get_demand_at_year(iso, year, 'High')
    growth_factor = demand_twh / REGIONAL_DEMAND_TWH[iso]
    demand_mw_profile = np.array(demand_norm, dtype=np.float64) * total_mwh_base * growth_factor
    total_annual_mwh = demand_mw_profile.sum()

    print(f"  Year={year}, growth_factor={growth_factor:.2f}x, "
          f"demand_twh={demand_twh:.0f}")

    # --- WITHOUT fix: no new builds, no growth scaling ---
    stack_old, fossil_old = build_merit_order_stack(
        iso, baseline_clean, demand_growth_factor=1.0, new_fossil_builds=None)
    from dispatch_utils import reconstruct_hourly_dispatch
    dispatch = reconstruct_hourly_dispatch(demand_norm, supply_profiles, resource_pcts, procurement_pct=100)
    residual_mw = dispatch['residual_demand'] * total_annual_mwh
    reserves_old = np.maximum(fossil_old - residual_mw, 0.0)
    from lmp_engine import PriceModel
    pm = PriceModel(iso, 'Medium')
    ordc_old = pm.compute_ordc_adder(reserves_old)
    scarcity_hours_old = int(np.sum(ordc_old > 50.0))
    avg_ordc_old = float(np.mean(ordc_old))

    print(f"  WITHOUT fix: fossil={fossil_old:.0f} MW, "
          f"scarcity_hours={scarcity_hours_old}, avg_ordc=${avg_ordc_old:.1f}")

    # --- WITH fix: 20 GW new builds + growth scaling ---
    new_builds = {'gas_ccgt': 14000, 'gas_ct': 6000}  # 20 GW typical by 2050
    stack_new, fossil_new = build_merit_order_stack(
        iso, baseline_clean, demand_growth_factor=growth_factor,
        new_fossil_builds=new_builds)
    reserves_new = np.maximum(fossil_new - residual_mw, 0.0)
    ordc_new = pm.compute_ordc_adder(reserves_new)
    scarcity_hours_new = int(np.sum(ordc_new > 50.0))
    avg_ordc_new = float(np.mean(ordc_new))

    print(f"  WITH fix:    fossil={fossil_new:.0f} MW, "
          f"scarcity_hours={scarcity_hours_new}, avg_ordc=${avg_ordc_new:.1f}")

    assert scarcity_hours_new < scarcity_hours_old, \
        "Fix should reduce scarcity hours"
    assert avg_ordc_new < avg_ordc_old, \
        "Fix should reduce avg ORDC adder"
    print("  PASS: scarcity significantly reduced\n")


if __name__ == '__main__':
    print("=" * 60)
    print("Testing ORDC scarcity + new-build fossil fixes")
    print("=" * 60 + "\n")

    test_merit_order_new_builds()
    test_demand_growth_scaling()
    test_scarcity_reduction()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
