#!/usr/bin/env python3
"""G1 Validation: Compare fleet-fraction vs plant-level retirement on PJM.

Two validation modes:
1. SYNTHETIC: Uses controlled mock plant data to demonstrate behavioral
   differences between fleet-fraction and plant-level retirement.
2. LIVE: Uses real EIA 860 fleet data (if available) with full simulation.

Usage:
    python validate_plant_retirement.py
"""
import sys
import os
import json
import time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(SCRIPT_DIR), 'backend'))

from market_simulation import (
    run_market_simulation, apply_economic_retirement,
    _apply_fleet_fraction_retirement, _apply_plant_level_retirement,
    compute_plant_level_economics, compute_reserve_margin,
    _PEAK_TO_AVG_RATIO,
)
from lmp_engine import build_plant_level_merit_order, FUEL_PRICES
from pipeline_config import (
    REGIONAL_DEMAND_TWH, CAPACITY_MARKET_PRICES,
    PEAK_DEMAND_MW, EXISTING_NUCLEAR_GW,
    NUCLEAR_OFFTAKE_CONTRACTS,
)

ISO = 'PJM'
SIM_YEARS = [2025, 2030, 2035, 2040, 2045, 2050]


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC VALIDATION — Controlled mock data, deterministic comparison
# ═══════════════════════════════════════════════════════════════════════════════

def _make_plant(pid, unit_type, cap_mw, margin, zone='PJM-East'):
    """Create a mock plant dict for plant-level retirement."""
    return {
        'plant_id': pid,
        'generator_id': f'{pid}_G1',
        'plant_name': f'Plant {pid}',
        'capacity_mw': cap_mw,
        'unit_type': unit_type,
        'fuel_type': {'coal_steam': 'BIT', 'gas_ccgt': 'NG', 'gas_ct': 'NG',
                      'oil_ct': 'DFO', 'nuclear': 'NUC'}.get(unit_type, 'NG'),
        'prime_mover': {'coal_steam': 'ST', 'gas_ccgt': 'CA', 'gas_ct': 'CT',
                        'oil_ct': 'CT', 'nuclear': 'ST'}.get(unit_type, 'CT'),
        'profit_per_mwh': margin,
        'revenue_per_mwh': max(0, margin + 35),
        'zone': zone,
        'status': 'stranded' if margin < -5 else ('at_risk' if margin < 2 else 'operating'),
    }


def run_synthetic_validation():
    """Compare retirement modes with synthetic plant fleets across 3 scenarios."""

    print("\n" + "=" * 78)
    print("PART 1: SYNTHETIC VALIDATION — Controlled Comparison")
    print("=" * 78)

    # Synthetic fleet: mix of efficient and inefficient plants
    # PJM has ~70 GW gas, ~20 GW coal, ~32 GW nuclear
    plant_fleet = [
        # Coal — mixed efficiency
        _make_plant('COAL_1', 'coal_steam', 2000, -25.0, 'PJM-West'),  # Ancient, terrible heat rate
        _make_plant('COAL_2', 'coal_steam', 1500, -18.0, 'PJM-West'),  # Old, bad
        _make_plant('COAL_3', 'coal_steam', 1200, -12.0, 'PJM-East'),  # Marginal
        _make_plant('COAL_4', 'coal_steam', 800,  -6.0,  'PJM-East'),  # Barely stranded
        _make_plant('COAL_5', 'coal_steam', 600,  -3.0,  'PJM-East'),  # At risk but above threshold
        # Gas CCGT — mostly profitable
        _make_plant('GAS_1', 'gas_ccgt', 3000, 12.0, 'PJM-East'),
        _make_plant('GAS_2', 'gas_ccgt', 2500, 8.0,  'PJM-West'),
        _make_plant('GAS_3', 'gas_ccgt', 2000, 3.0,  'PJM-East'),
        _make_plant('GAS_4', 'gas_ccgt', 1500, -2.0, 'PJM-West'),   # At risk
        _make_plant('GAS_5', 'gas_ccgt', 1000, -7.0, 'PJM-East'),   # Stranded
        # Gas CT — peakers, marginal economics
        _make_plant('CT_1', 'gas_ct', 500, 5.0, 'PJM-East'),
        _make_plant('CT_2', 'gas_ct', 400, -1.0, 'PJM-West'),
        _make_plant('CT_3', 'gas_ct', 300, -8.0, 'PJM-West'),      # Stranded
        # Nuclear — contract protected in some scenarios
        _make_plant('NUC_1', 'nuclear', 2000, -4.0, 'PJM-East'),   # Marginal but capacity market helps
        _make_plant('NUC_2', 'nuclear', 1000, -12.0, 'PJM-West'),  # Severely uneconomic
    ]

    # Fleet-level gen_econ (used by fleet-fraction path)
    gen_econ_scenarios = {
        'low_lmp': {
            'coal_steam': {'capacity_mw': 6100, 'margin_mwh': -15, 'cf': 0.30, 'dispatch_hours': 2600},
            'gas_ccgt':   {'capacity_mw': 10000, 'margin_mwh': -1, 'cf': 0.40, 'dispatch_hours': 3500},
            'gas_ct':     {'capacity_mw': 1200, 'margin_mwh': -3, 'cf': 0.10, 'dispatch_hours': 876},
        },
        'mid_lmp': {
            'coal_steam': {'capacity_mw': 6100, 'margin_mwh': -8, 'cf': 0.35, 'dispatch_hours': 3000},
            'gas_ccgt':   {'capacity_mw': 10000, 'margin_mwh': 5, 'cf': 0.50, 'dispatch_hours': 4380},
            'gas_ct':     {'capacity_mw': 1200, 'margin_mwh': 2, 'cf': 0.12, 'dispatch_hours': 1050},
        },
        'high_lmp': {
            'coal_steam': {'capacity_mw': 6100, 'margin_mwh': -3, 'cf': 0.40, 'dispatch_hours': 3500},
            'gas_ccgt':   {'capacity_mw': 10000, 'margin_mwh': 12, 'cf': 0.60, 'dispatch_hours': 5256},
            'gas_ct':     {'capacity_mw': 1200, 'margin_mwh': 8, 'cf': 0.15, 'dispatch_hours': 1314},
        },
    }

    # Adjust plant margins per scenario
    margin_offsets = {'low_lmp': -8, 'mid_lmp': 0, 'high_lmp': +8}

    results = {}

    for scenario_key, offset in margin_offsets.items():
        scenario_name = {'low_lmp': 'Low LMP ($25 avg)',
                         'mid_lmp': 'Medium LMP ($38 avg)',
                         'high_lmp': 'High LMP ($54 avg)'}[scenario_key]

        print(f"\n{'─' * 78}")
        print(f"Scenario: {scenario_name}")
        print(f"{'─' * 78}")

        # Adjust plant margins
        adjusted_plants = []
        for p in plant_fleet:
            ap = dict(p)
            ap['profit_per_mwh'] = p['profit_per_mwh'] + offset
            ap['revenue_per_mwh'] = max(0, ap['profit_per_mwh'] + 35)
            ap['status'] = ('stranded' if ap['profit_per_mwh'] < -5
                            else 'at_risk' if ap['profit_per_mwh'] < 2
                            else 'operating')
            adjusted_plants.append(ap)

        gen_econ = gen_econ_scenarios[scenario_key]

        # Simulate 3 years of retirement for each mode
        results[scenario_key] = {'scenario': scenario_name, 'plant_level': [], 'fleet_fraction': []}

        # --- Plant-level mode ---
        state_plant = {
            'economic_retirements': {},
            'retired_plants': [],
            'cumulative_gw': {'solar': 15.0, 'wind': 20.0, 'nuclear': 32.0},
            'nuclear_retired': False,
        }
        print(f"\n  Plant-Level Retirement:")
        print(f"  {'Year':>6} │ {'Retired':>10} │ {'#Plants':>7} │ {'Types retired':<40} │ {'Reserve%':>8}")
        print(f"  {'─'*6}─┼─{'─'*10}─┼─{'─'*7}─┼─{'─'*40}─┼─{'─'*8}")

        for yr_idx, year in enumerate([2025, 2030, 2035]):
            adj, by_type, total_mw, plant_list = apply_economic_retirement(
                gen_econ, ISO, year, state_plant, _log=lambda *a: None,
                plant_economics=adjusted_plants, demand_twh=300)

            rm = compute_reserve_margin(adj, state_plant.get('cumulative_gw', {}), 300)
            type_str = ', '.join(f"{t}:{mw:.0f}MW" for t, mw in sorted(by_type.items())) or '(none)'
            print(f"  {year:>6} │ {total_mw:>9.0f}MW │ {len(plant_list):>7} │ {type_str:<40} │ {rm:>7.1f}%")

            results[scenario_key]['plant_level'].append({
                'year': year,
                'retired_mw': total_mw,
                'n_plants': len(plant_list),
                'by_type': by_type,
                'plant_ids': [p['plant_id'] for p in plant_list],
                'reserve_margin': round(rm, 1),
            })

        # --- Fleet-fraction mode ---
        state_fleet = {
            'economic_retirements': {},
            'retired_plants': [],
            'cumulative_gw': {'solar': 15.0, 'wind': 20.0, 'nuclear': 32.0},
            'nuclear_retired': False,
        }
        print(f"\n  Fleet-Fraction Retirement:")
        print(f"  {'Year':>6} │ {'Retired':>10} │ {'Types retired':<50} │ {'Reserve%':>8}")
        print(f"  {'─'*6}─┼─{'─'*10}─┼─{'─'*50}─┼─{'─'*8}")

        for yr_idx, year in enumerate([2025, 2030, 2035]):
            adj_f, by_type_f, total_mw_f = _apply_fleet_fraction_retirement(
                gen_econ, ISO, year, state_fleet, _log=lambda *a: None)

            rm_f = compute_reserve_margin(adj_f, state_fleet.get('cumulative_gw', {}), 300)
            type_str_f = ', '.join(f"{t}:{mw:.0f}MW" for t, mw in sorted(by_type_f.items())) or '(none)'
            print(f"  {year:>6} │ {total_mw_f:>9.0f}MW │ {type_str_f:<50} │ {rm_f:>7.1f}%")

            results[scenario_key]['fleet_fraction'].append({
                'year': year,
                'retired_mw': total_mw_f,
                'by_type': by_type_f,
                'reserve_margin': round(rm_f, 1),
            })

        # --- Comparison ---
        print(f"\n  Comparison:")
        for yr_idx in range(3):
            pl = results[scenario_key]['plant_level'][yr_idx]
            ff = results[scenario_key]['fleet_fraction'][yr_idx]
            d_mw = pl['retired_mw'] - ff['retired_mw']
            d_rm = pl['reserve_margin'] - ff['reserve_margin']
            print(f"    {pl['year']}: ΔRetirement={d_mw:+.0f}MW  ΔReserve={d_rm:+.1f}pp  "
                  f"Plant IDs: {pl.get('plant_ids', [])}")

    # --- Behavioral Observations ---
    print(f"\n{'=' * 78}")
    print("BEHAVIORAL OBSERVATIONS")
    print(f"{'=' * 78}")

    for scenario_key, data in results.items():
        pl = data['plant_level']
        ff = data['fleet_fraction']
        print(f"\n  {data['scenario']}:")

        total_pl = sum(r['retired_mw'] for r in pl)
        total_ff = sum(r['retired_mw'] for r in ff)
        print(f"    Total retirement: Plant={total_pl:,.0f} MW  Fleet={total_ff:,.0f} MW  Δ={total_pl-total_ff:+,.0f} MW")

        # Plant-level specifics
        all_retired_ids = set()
        for r in pl:
            all_retired_ids.update(r.get('plant_ids', []))
        print(f"    Plants retired (plant-level): {sorted(all_retired_ids)}")

        # Type breakdown
        combined_by_type_pl = {}
        combined_by_type_ff = {}
        for r in pl:
            for t, mw in r.get('by_type', {}).items():
                combined_by_type_pl[t] = combined_by_type_pl.get(t, 0) + mw
        for r in ff:
            for t, mw in r.get('by_type', {}).items():
                combined_by_type_ff[t] = combined_by_type_ff.get(t, 0) + mw
        print(f"    By type (plant): {dict((k, f'{v:.0f}MW') for k,v in sorted(combined_by_type_pl.items()))}")
        print(f"    By type (fleet): {dict((k, f'{v:.0f}MW') for k,v in sorted(combined_by_type_ff.items()))}")

    print(f"\n{'=' * 78}")
    print("KEY DIFFERENCES:")
    print("  1. SELECTIVITY: Plant-level retires worst-performing units first,")
    print("     preserving efficient units of the same type. Fleet-fraction retires")
    print("     a percentage of ALL units of a type regardless of individual merit.")
    print("  2. RELIABILITY: Plant-level enforces 15% zonal reserve margin floor.")
    print("     Fleet-fraction uses per-type 5% minimum (allows lower reserves).")
    print("  3. NUCLEAR: Plant-level evaluates each nuclear plant individually")
    print("     using contract economics (PPA floors, capacity market revenue).")
    print("     Fleet-fraction retires the entire nuclear fleet at a trigger year.")
    print("  4. PERSISTENCE: Plant-level tracks retired plant IDs across years,")
    print("     ensuring a retired plant never re-enters the fleet.")
    print(f"{'=' * 78}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE VALIDATION — Full simulation with EIA fleet data (if available)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_metrics(year_results):
    """Extract key comparison metrics from year results."""
    metrics = []
    for yr in year_results:
        econ_ret_mw = yr.get('total_economic_retirement_mw', 0)
        plant_rets = yr.get('plant_retirements', [])
        gen_econ = yr.get('adjusted_generator_economics', yr.get('generator_economics', {}))
        cumulative_gw = yr.get('cumulative_gw', {})
        demand_twh = yr.get('demand_twh', 0)
        rm = compute_reserve_margin(gen_econ, cumulative_gw, demand_twh)
        cap_price = yr.get('capacity_price_kw_yr', CAPACITY_MARKET_PRICES.get(ISO, 0))
        metrics.append({
            'year': yr['year'],
            'clean_pct': yr.get('clean_pct', 0),
            'emissions_mt': yr.get('emissions_mt', 0),
            'avg_lmp': yr.get('avg_lmp', 0),
            'econ_retired_mw': econ_ret_mw,
            'n_plants_retired': len(plant_rets),
            'retired_by_type': yr.get('economic_retirements_mw', {}),
            'reserve_margin_pct': round(rm, 1),
            'capacity_price': cap_price,
            'fossil_cap_mw': sum(e.get('capacity_mw', 0) for e in gen_econ.values()),
            'nuclear_retired': yr.get('nuclear_retired', False),
        })
    return metrics


LIVE_SCENARIOS = [
    {'id': 'low_fuel', 'name': 'Low Fuel Prices', 'conditions': {
        'name': 'Low Fuel', 'demand_growth': 'Medium', 'fuel_level': 'Low',
        'lcoe_level': 'Medium', 'tx_level': 'Medium', 'ppa_premium_pct': 0,
        'gas_friction': 1.0, 'queue_cap_level': 'Medium',
        'new_fossil_cost_level': 'Medium', 'carbon_price': 0}},
    {'id': 'medium_fuel', 'name': 'Medium Fuel Prices', 'conditions': {
        'name': 'Medium Fuel', 'demand_growth': 'Medium', 'fuel_level': 'Medium',
        'lcoe_level': 'Medium', 'tx_level': 'Medium', 'ppa_premium_pct': 0,
        'gas_friction': 1.0, 'queue_cap_level': 'Medium',
        'new_fossil_cost_level': 'Medium', 'carbon_price': 0}},
    {'id': 'high_fuel', 'name': 'High Fuel Prices', 'conditions': {
        'name': 'High Fuel', 'demand_growth': 'Medium', 'fuel_level': 'High',
        'lcoe_level': 'Medium', 'tx_level': 'Medium', 'ppa_premium_pct': 0,
        'gas_friction': 1.0, 'queue_cap_level': 'Medium',
        'new_fossil_cost_level': 'Medium', 'carbon_price': 0}},
]


def run_live_validation():
    """Run full simulation comparison if EIA fleet data is available."""

    print("\n" + "=" * 78)
    print("PART 2: LIVE VALIDATION — Full PJM Simulation")
    print("=" * 78)

    # Check if fleet data is available
    try:
        stack, total = build_plant_level_merit_order('PJM', 40.0, fuel_level='Medium')
        if not stack:
            print("\n  SKIPPED: EIA 860 fleet data not available in this environment.")
            print("  Plant-level retirement falls back to fleet-fraction gracefully.")
            print("  To run live validation, populate market-simulator/data/eia-860/")
            print("  with EIA Form 860 generator data.")
            return {}
        print(f"\n  Fleet data available: {len(stack)} plants, {total:,.0f} MW")
    except Exception as e:
        print(f"\n  SKIPPED: Could not load fleet data ({e})")
        return {}

    # Preload common data
    print("  Loading common data...")
    try:
        from dispatch_utils import load_common_data as _load
        demand_data, gen_profiles, emission_rates, fossil_mix = _load()
        from market_simulation import load_egrid_baselines
        egrid_baselines = load_egrid_baselines()
        preloaded = {
            'demand_data': demand_data, 'gen_profiles': gen_profiles,
            'emission_rates': emission_rates, 'fossil_mix': fossil_mix,
            'egrid_baselines': egrid_baselines,
        }
    except Exception as e:
        print(f"  SKIPPED: Could not load data ({e})")
        return {}

    all_results = {}
    import market_simulation as ms

    for scenario in LIVE_SCENARIOS:
        sid = scenario['id']
        conditions = scenario['conditions']
        print(f"\n  Scenario: {scenario['name']}")

        # Plant-level
        t1 = time.time()
        plant_results = run_market_simulation(
            f'{sid}_plant', conditions, isos=[ISO], sim_years=SIM_YEARS,
            _preloaded=preloaded, _quiet=True)
        plant_metrics = extract_metrics(plant_results.get(ISO, []))
        print(f"    Plant-level: {time.time()-t1:.1f}s")

        # Fleet-fraction (patch to force fallback)
        _orig = ms.build_plant_level_merit_order
        ms.build_plant_level_merit_order = lambda *a, **k: (None, 0)
        t2 = time.time()
        fleet_results = run_market_simulation(
            f'{sid}_fleet', conditions, isos=[ISO], sim_years=SIM_YEARS,
            _preloaded=preloaded, _quiet=True)
        fleet_metrics = extract_metrics(fleet_results.get(ISO, []))
        ms.build_plant_level_merit_order = _orig
        print(f"    Fleet-fraction: {time.time()-t2:.1f}s")

        all_results[sid] = {
            'scenario': scenario['name'],
            'plant_level': plant_metrics,
            'fleet_fraction': fleet_metrics,
        }

        # Print comparison
        if plant_metrics and fleet_metrics:
            print(f"\n    {'Year':>6} │ {'Mode':<14} │ {'Retired MW':>10} │ {'#Plants':>7} │ "
                  f"{'Reserve%':>8} │ {'Fossil MW':>10}")
            print(f"    {'─'*6}─┼─{'─'*14}─┼─{'─'*10}─┼─{'─'*7}─┼─{'─'*8}─┼─{'─'*10}")
            for pm, fm in zip(plant_metrics, fleet_metrics):
                print(f"    {pm['year']:>6} │ {'Plant-level':<14} │ {pm['econ_retired_mw']:>10.0f} │ "
                      f"{pm['n_plants_retired']:>7} │ {pm['reserve_margin_pct']:>7.1f}% │ {pm['fossil_cap_mw']:>10.0f}")
                print(f"    {pm['year']:>6} │ {'Fleet-fraction':<14} │ {fm['econ_retired_mw']:>10.0f} │ "
                      f"{fm['n_plants_retired']:>7} │ {fm['reserve_margin_pct']:>7.1f}% │ {fm['fossil_cap_mw']:>10.0f}")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 78)
    print("G1 VALIDATION: Fleet-Fraction vs Plant-Level Retirement — PJM")
    print("=" * 78)

    # Part 1: Always runs — synthetic controlled comparison
    synthetic_results = run_synthetic_validation()

    # Part 2: Only if EIA data available
    live_results = run_live_validation()

    # Save combined results
    output_path = os.path.join(
        os.path.dirname(SCRIPT_DIR), 'data', 'results',
        'g1_validation_plant_vs_fleet.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined = {'synthetic': synthetic_results, 'live': live_results}
    with open(output_path, 'w') as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
