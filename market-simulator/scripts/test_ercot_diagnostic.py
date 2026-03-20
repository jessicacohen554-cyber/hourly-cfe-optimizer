#!/usr/bin/env python3
"""Diagnostic: why ERCOT High growth still shows $400+ avg_lmp.

Traces the hourly LMP distribution for ERCOT 2030 (worst case: no prior new builds)
to identify how many hours hit capacity shortage vs ORDC vs normal pricing.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from market_simulation import (
    compute_lmp_at_threshold, load_common_data, load_egrid_baselines,
    get_demand_at_year,
)
from dispatch_utils import (
    get_demand_profile, get_supply_profiles, reconstruct_hourly_dispatch,
    GRID_MIX_SHARES, REGIONAL_DEMAND_TWH, H,
)
from lmp_engine import build_merit_order_stack, PriceModel, INSTALLED_FOSSIL_MW
from pipeline_config import ORDC_PARAMS

iso = 'ERCOT'
year = 2030
growth_level = 'High'

print("Loading data...")
demand_data, gen_profiles, _, _ = load_common_data()
demand_norm, total_mwh_base = get_demand_profile(iso, demand_data)
supply_profiles = get_supply_profiles(iso, gen_profiles)

demand_twh = get_demand_at_year(iso, year, growth_level)
growth_factor = demand_twh / REGIONAL_DEMAND_TWH[iso]
demand_mw_profile = np.array(demand_norm, dtype=np.float64) * total_mwh_base * growth_factor
total_annual_mwh = demand_mw_profile.sum()

resource_pcts = dict(GRID_MIX_SHARES[iso])
baseline_clean = sum(resource_pcts.values())

print(f"\nERCOT {year} — High growth ({growth_level})")
print(f"  Growth factor: {growth_factor:.3f}x")
print(f"  Demand TWh: {demand_twh:.0f}")
print(f"  Avg demand MW: {total_annual_mwh/H:.0f}")
print(f"  Peak demand MW: {demand_mw_profile.max():.0f}")
print(f"  Baseline clean: {baseline_clean:.1f}%")

# Build dispatch (no new builds, first year)
dispatch = reconstruct_hourly_dispatch(demand_norm, supply_profiles, resource_pcts, procurement_pct=100)
residual_mw = dispatch['residual_demand'] * total_annual_mwh

# Build merit stack (no new builds, baseline growth)
stack, total_fossil_mw = build_merit_order_stack(
    iso, baseline_clean, demand_growth_factor=growth_factor, new_fossil_builds=None)
print(f"  Fossil fleet: {total_fossil_mw:.0f} MW")
print(f"  Installed: {INSTALLED_FOSSIL_MW[iso]} MW")

# Analyze residual vs fossil capacity
print(f"\nResidual demand distribution:")
print(f"  Min: {residual_mw.min():.0f} MW")
print(f"  Mean: {residual_mw.mean():.0f} MW")
print(f"  P90: {np.percentile(residual_mw, 90):.0f} MW")
print(f"  P95: {np.percentile(residual_mw, 95):.0f} MW")
print(f"  P99: {np.percentile(residual_mw, 99):.0f} MW")
print(f"  Max: {residual_mw.max():.0f} MW")

# Hours where residual exceeds fossil capacity (capacity shortage)
shortage_hours = int(np.sum(residual_mw > total_fossil_mw))
shortage_pct = shortage_hours / H * 100
print(f"\nCapacity shortage (residual > {total_fossil_mw:.0f} MW fossil): {shortage_hours} hours ({shortage_pct:.1f}%)")

# Hours where reserves < knee (ORDC fires)
reserves = np.maximum(total_fossil_mw - residual_mw, 0.0)
knee = ORDC_PARAMS[iso]['knee_mw']
ordc_hours = int(np.sum(reserves < knee))
print(f"ORDC active (reserves < {knee} MW): {ordc_hours} hours ({ordc_hours/H*100:.1f}%)")

# Compute actual LMP
hourly_lmp, avg_lmp, p90_lmp, _, _, _, scarcity_frac = compute_lmp_at_threshold(
    iso, baseline_clean, 'Medium', demand_norm, demand_mw_profile,
    supply_profiles, resource_pcts, demand_growth_factor=growth_factor)

print(f"\nLMP breakdown:")
print(f"  Avg: ${avg_lmp:.1f}")
print(f"  P90: ${p90_lmp:.1f}")
print(f"  Min: ${hourly_lmp.min():.1f}")
print(f"  Max: ${hourly_lmp.max():.1f}")
print(f"  Hours > $500: {int(np.sum(hourly_lmp > 500))}")
print(f"  Hours > $1000: {int(np.sum(hourly_lmp > 1000))}")
print(f"  Hours > $2000: {int(np.sum(hourly_lmp > 2000))}")

# Contribution breakdown
normal_hours = (residual_mw > 0) & (residual_mw <= total_fossil_mw)
shortage_mask = residual_mw > total_fossil_mw
surplus_mask = residual_mw <= 0

if normal_hours.any():
    print(f"\n  Normal hours ({int(normal_hours.sum())}): avg LMP = ${hourly_lmp[normal_hours].mean():.1f}")
if shortage_mask.any():
    print(f"  Shortage hours ({int(shortage_mask.sum())}): avg LMP = ${hourly_lmp[shortage_mask].mean():.1f}")
if surplus_mask.any():
    print(f"  Surplus hours ({int(surplus_mask.sum())}): avg LMP = ${hourly_lmp[surplus_mask].mean():.1f}")

# Show impact: what would avg_lmp be WITHOUT shortage hours?
if normal_hours.any() and shortage_mask.any():
    avg_without_shortage = float(np.mean(hourly_lmp[~shortage_mask]))
    print(f"\n  Avg LMP EXCLUDING shortage hours: ${avg_without_shortage:.1f}")
    shortage_contribution = float(hourly_lmp[shortage_mask].sum()) / H
    normal_contribution = float(hourly_lmp[normal_hours].sum()) / H
    print(f"  Shortage hours contribute ${shortage_contribution:.1f}/MWh to annual avg")
    print(f"  Normal hours contribute ${normal_contribution:.1f}/MWh to annual avg")

# Now test with 42 GW new builds (simulating 2035 state)
print(f"\n{'='*60}")
print(f"WITH 42 GW new-build fossil (simulating 2035 fleet state):")
stack2, fossil2 = build_merit_order_stack(
    iso, baseline_clean, demand_growth_factor=growth_factor,
    new_fossil_builds={'gas_ccgt': 29400, 'gas_ct': 12600})
print(f"  Fossil fleet: {fossil2:.0f} MW")
shortage2 = int(np.sum(residual_mw > fossil2))
print(f"  Shortage hours: {shortage2} ({shortage2/H*100:.1f}%)")
reserves2 = np.maximum(fossil2 - residual_mw, 0.0)
ordc2 = int(np.sum(reserves2 < knee))
print(f"  ORDC hours: {ordc2} ({ordc2/H*100:.1f}%)")
