#!/usr/bin/env python3
"""Quick A/B comparison: unified deployment vs existing sweep baseline.

Runs ERCOT and PJM under all-Medium conditions through the current (unified)
model, then compares key metrics against the existing sweep_1215 parquet.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np

from market_simulation import run_market_simulation, build_sim_years

# ── Load baseline from existing sweep ────────────────────────────────────
PARQUET = os.path.join(os.path.dirname(__file__),
                       'results', 'sweep_1215', 'sweep_1215_flat.parquet')
print(f"Loading baseline: {PARQUET}")
df = pd.read_parquet(PARQUET)

# Find the all-Medium scenario (demand=Medium, price=Medium, etc.)
# The scenario_id encoding uses M for Medium across all dimensions
# Look for scenarios with all-Medium conditions
medium_mask = df['scenario_id'].str.contains('_M_', na=False)
if medium_mask.sum() == 0:
    # Try to find by examining scenario_id patterns
    print(f"Sample scenario_ids: {df['scenario_id'].unique()[:10]}")
    # Fallback: use the scenario closest to medium by filtering on key metrics
    medium_mask = pd.Series([True] * len(df))

ISOS = ['ERCOT', 'PJM']
YEARS = [2030, 2040, 2050]

# ── Run unified model ────────────────────────────────────────────────────
conditions = {
    'name': 'All-Medium (Unified A/B Test)',
    'demand_growth': 'Medium',
    'lcoe_level': 'Medium',
    'fuel_level': 'Medium',
    'tx_level': 'Medium',
    'ccs_level': 'Medium',
    'q45': True,
    'ppa_level': 'Medium',
    'gas_friction': 'Medium',
    'queue_level': 'Medium',
    'new_fossil_cost_level': 'Medium',
    'new_fossil_enabled': True,
    'carbon_price': 0,
    'learning_speed': 'Medium',
    'learning_curves_enabled': True,
}

sim_years = build_sim_years(2023, 2050, step=5)
print(f"\nRunning unified model for {ISOS}, years: {sim_years}")
results = run_market_simulation(
    'UNIFIED_AB_TEST', conditions,
    isos=ISOS, sim_years=sim_years,
)

# ── Compare ──────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("COMPARISON: Unified Deployment vs Existing Sweep Baseline")
print(f"{'='*80}")

METRICS = ['clean_pct', 'avg_lmp', 'emissions_mt', 'cost_per_mwh',
           'total_new_fossil_mw', 'total_economic_retirement_mw',
           'gas_built_gw', 'fossil_built_gw']

for iso in ISOS:
    print(f"\n{'─'*60}")
    print(f"  {iso}")
    print(f"{'─'*60}")

    # Unified results
    unified_years = {r['year']: r for r in results.get(iso, [])}

    # Baseline: filter to this ISO, all-Medium scenarios, target years
    iso_df = df[(df['iso'] == iso) & medium_mask]

    for year in YEARS:
        u = unified_years.get(year)
        if u is None:
            continue

        # Baseline: median across medium scenarios for this year
        b = iso_df[iso_df['year'] == year]
        if b.empty:
            print(f"\n  {year}: No baseline data")
            continue

        print(f"\n  {year}:")
        print(f"  {'Metric':<30} {'Baseline (P50)':<18} {'Unified':<18} {'Delta':<12}")
        print(f"  {'─'*78}")

        for m in METRICS:
            b_val = b[m].median() if m in b.columns else float('nan')
            u_val = u.get(m, float('nan'))
            if isinstance(u_val, dict):
                u_val = sum(u_val.values()) if u_val else 0
            delta = u_val - b_val if not (np.isnan(b_val) or np.isnan(u_val)) else float('nan')
            sign = '+' if delta > 0 else ''
            print(f"  {m:<30} {b_val:>14.1f}    {u_val:>14.1f}    {sign}{delta:>8.1f}")

        # Show unified merit order for this year
        merit = u.get('unified_merit_order', [])
        if merit:
            print(f"\n  Unified merit order (by net cost):")
            for i, c in enumerate(merit):
                tag = '★' if c['profit'] > 0 else ' '
                print(f"    {tag} {i+1}. {c['resource']:<16} ({c['type']:<6}) "
                      f"${c['net_cost']:>7.1f}/MWh  margin ${c['profit']:>6.1f}/MWh")

        # Show new fossil details
        nfd = u.get('new_fossil_details', {})
        if nfd:
            print(f"\n  New fossil candidate economics:")
            for ft, d in nfd.items():
                print(f"    {ft}: all-in ${d.get('all_in_at_min_cf', 0):.1f}/MWh, "
                      f"CF {d.get('expected_cf', 0):.1%}, "
                      f"margin ${d.get('net_margin', 0):.1f}/MWh, "
                      f"{'VIABLE' if d.get('viable') else 'not viable'}")

print(f"\n{'='*80}")
print("Done.")
