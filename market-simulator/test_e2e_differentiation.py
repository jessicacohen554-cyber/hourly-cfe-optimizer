#!/usr/bin/env python3
"""End-to-end simulation test — verify results are responsive and differentiated
based on different inputs, and that multi-year output is correct."""

import sys
import os

# Add scripts/ to path for simulation engine
test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, 'scripts'))

from market_simulation import run_market_simulation

from pathlib import Path
import json
import csv
import tempfile

ISO = "ERCOT"

# ── Test Scenarios ──
# Each scenario varies one dimension to verify results change
SCENARIOS = {
    "baseline": {
        "fuel_level": "Medium",
        "lcoe_level": "Medium",
        "tx_level": "Medium",
        "carbon_price": 5.50,
        "demand_growth": "Medium",
        "q45": True,
        "learning_speed": "Medium",
        "ppa_level": "Medium",
        "gas_friction": "Medium",
        "interchange_enabled": True,
        "dr_level": "Off",
        "learning_curves_enabled": True,
    },
    "high_gas_price": {
        "fuel_level": "High",
        "lcoe_level": "Medium",
        "tx_level": "Medium",
        "carbon_price": 5.50,
        "demand_growth": "Medium",
        "q45": True,
        "learning_speed": "Medium",
        "ppa_level": "Medium",
        "gas_friction": "Medium",
        "interchange_enabled": True,
        "dr_level": "Off",
        "learning_curves_enabled": True,
    },
    "high_carbon_price": {
        "fuel_level": "Medium",
        "lcoe_level": "Medium",
        "tx_level": "Medium",
        "carbon_price": 100.0,
        "demand_growth": "Medium",
        "q45": True,
        "learning_speed": "Medium",
        "ppa_level": "Medium",
        "gas_friction": "Medium",
        "interchange_enabled": True,
        "dr_level": "Off",
        "learning_curves_enabled": True,
    },
    "low_clean_lcoe": {
        "fuel_level": "Medium",
        "lcoe_level": "Low",
        "tx_level": "Medium",
        "carbon_price": 5.50,
        "demand_growth": "Medium",
        "q45": True,
        "learning_speed": "Medium",
        "ppa_level": "Medium",
        "gas_friction": "Medium",
        "interchange_enabled": True,
        "dr_level": "Off",
        "learning_curves_enabled": True,
    },
    "no_learning_curves": {
        "fuel_level": "Medium",
        "lcoe_level": "Medium",
        "tx_level": "Medium",
        "carbon_price": 5.50,
        "demand_growth": "Medium",
        "q45": True,
        "learning_speed": "Medium",
        "ppa_level": "Medium",
        "gas_friction": "Medium",
        "interchange_enabled": True,
        "dr_level": "Off",
        "learning_curves_enabled": False,
    },
    "custom_new_build_hr": {
        "fuel_level": "Medium",
        "lcoe_level": "Medium",
        "tx_level": "Medium",
        "carbon_price": 5.50,
        "demand_growth": "Medium",
        "q45": True,
        "learning_speed": "Medium",
        "ppa_level": "Medium",
        "gas_friction": "Medium",
        "interchange_enabled": True,
        "dr_level": "Off",
        "learning_curves_enabled": True,
        "custom_heat_rates": {"new_gas_ccgt": 7.5, "gas_ccgt": 7.0, "gas_ct": 10.5, "oil_ct": 10.5, "coal_steam": 10.0},
    },
    "tx_overrides_high": {
        "fuel_level": "Medium",
        "lcoe_level": "Medium",
        "tx_level": "Medium",
        "carbon_price": 5.50,
        "demand_growth": "Medium",
        "q45": True,
        "learning_speed": "Medium",
        "ppa_level": "Medium",
        "gas_friction": "Medium",
        "interchange_enabled": True,
        "dr_level": "Off",
        "learning_curves_enabled": True,
        "tx_overrides": {"solar": 25.0, "wind": 25.0},  # Very high TX overrides
    },
    "high_demand_growth": {
        "fuel_level": "Medium",
        "lcoe_level": "Medium",
        "tx_level": "Medium",
        "carbon_price": 5.50,
        "demand_growth": "High",
        "q45": True,
        "learning_speed": "Medium",
        "ppa_level": "Medium",
        "gas_friction": "Medium",
        "interchange_enabled": True,
        "dr_level": "Off",
        "learning_curves_enabled": True,
    },
}

SIM_YEARS = [2025, 2030, 2035, 2040]  # Multi-year trajectory


def run_scenario(name, conditions):
    """Run a single simulation scenario and return key metrics."""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")

    # Engine expects 'name' key in conditions
    conditions_with_name = {**conditions, "name": name}
    results = run_market_simulation(
        scenario_id=f"TEST_{name}",
        conditions=conditions_with_name,
        isos=[ISO],
        nuclear_retirement_threshold=30,
        snapshot_mode=False,
        sim_years=SIM_YEARS,
        _quiet=True,
    )

    iso_results = results.get(ISO, [])
    num_years = len(iso_results)
    print(f"  Years returned: {num_years}")

    if not iso_results:
        print(f"  ERROR: No results returned!")
        return None

    # Use raw iso_results directly (skip backend response model for this test)

    # Extract per-year metrics
    metrics = {
        "num_years": num_years,
        "years": [],
        "final_clean_pct": None,
        "final_avg_lmp": None,
        "final_emissions_mt": None,
        "final_demand_twh": None,
        "ccs_breakeven": None,
    }

    for yr in iso_results:
        yr_metrics = {
            "year": yr.get("year"),
            "clean_pct": yr.get("clean_pct"),
            "demand_twh": yr.get("demand_twh"),
            "emissions_mt": yr.get("emissions_mt"),
            "avg_lmp": yr.get("avg_lmp"),
            "cost_per_mwh": yr.get("cost_per_mwh"),
            "zones_deployed": yr.get("zones_deployed", []),
            "resource_mix_twh": yr.get("resource_mix_twh", {}),
        }
        metrics["years"].append(yr_metrics)
        print(f"  Year {yr.get('year')}: clean={yr.get('clean_pct', 0):.1f}%, "
              f"LMP=${yr.get('avg_lmp', 0):.1f}, "
              f"demand={yr.get('demand_twh', 0):.0f} TWh, "
              f"emissions={yr.get('emissions_mt', 0):.1f} MT, "
              f"cost=${yr.get('cost_per_mwh', 0):.1f}/MWh")

    final = iso_results[-1]
    metrics["final_clean_pct"] = final.get("clean_pct", 0)
    metrics["final_avg_lmp"] = final.get("avg_lmp", 0)
    metrics["final_emissions_mt"] = final.get("emissions_mt", 0)
    metrics["final_demand_twh"] = final.get("demand_twh", 0)
    ccs_be = final.get("ccs_breakeven", {})
    metrics["ccs_breakeven"] = ccs_be.get("ccs_vs_existing_carbon_price", 0) if isinstance(ccs_be, dict) else 0
    metrics["new_gas_breakeven"] = ccs_be.get("new_gas_vs_old_carbon_price", 0) if isinstance(ccs_be, dict) else 0

    return metrics, iso_results


def main():
    results = {}
    all_pass = True

    for name, conditions in SCENARIOS.items():
        try:
            result = run_scenario(name, conditions)
            if result is None:
                print(f"\n  ✗ FAILED: {name} returned no results")
                all_pass = False
                continue
            metrics, iso_results = result
            results[name] = metrics

            # Test multi-year output
            if metrics["num_years"] != len(SIM_YEARS):
                print(f"\n  ✗ FAILED: Expected {len(SIM_YEARS)} years, got {metrics['num_years']}")
                all_pass = False
            else:
                print(f"  ✓ Multi-year output: {metrics['num_years']} years")

            # Verify each year has generator_economics and resource_mix_twh
            for yr in iso_results:
                yr_year = yr.get("year")
                gen_econ = yr.get("generator_economics", {})
                rmix = yr.get("resource_mix_twh", {})
                if not gen_econ:
                    print(f"  ✗ Year {yr_year} missing generator_economics")
                    all_pass = False
                if not rmix:
                    print(f"  ✗ Year {yr_year} missing resource_mix_twh")
                    all_pass = False

        except Exception as e:
            print(f"\n  ✗ EXCEPTION in {name}: {e}")
            import traceback
            traceback.print_exc()
            all_pass = False

    # ── Differentiation Tests ──
    print(f"\n{'='*60}")
    print("DIFFERENTIATION TESTS")
    print(f"{'='*60}")

    def get_year(metrics, year_idx):
        """Get metrics for a specific year index (0=first, 1=second, etc.)"""
        if metrics and metrics["years"] and len(metrics["years"]) > year_idx:
            return metrics["years"][year_idx]
        return {}

    # Compare at 2025 (idx=0) and 2030 (idx=1) where scenarios diverge.
    # By 2040, all scenarios converge to 100% clean so differentiation disappears.
    for yr_idx, yr_label in [(0, "2025"), (1, "2030")]:
        print(f"\n  --- Comparing at {yr_label} ---")

        if "baseline" in results and "high_gas_price" in results:
            b_yr = get_year(results["baseline"], yr_idx)
            h_yr = get_year(results["high_gas_price"], yr_idx)
            b_lmp = b_yr.get("avg_lmp", 0)
            h_lmp = h_yr.get("avg_lmp", 0)
            diff = h_lmp - b_lmp
            print(f"\n  [{yr_label}] High gas LMP: ${b_lmp:.1f} → ${h_lmp:.1f} (Δ ${diff:+.1f})")
            if abs(diff) > 0.01:
                print(f"  ✓ LMP responds to fuel prices at {yr_label}")
            else:
                print(f"  ✗ FAIL: LMP should differ with fuel price at {yr_label}")
                all_pass = False

        if "baseline" in results and "high_carbon_price" in results:
            b_yr = get_year(results["baseline"], yr_idx)
            h_yr = get_year(results["high_carbon_price"], yr_idx)
            b_em = b_yr.get("emissions_mt", 0)
            h_em = h_yr.get("emissions_mt", 0)
            em_diff = h_em - b_em
            print(f"\n  [{yr_label}] High carbon emissions: {b_em:.1f} → {h_em:.1f} MT (Δ {em_diff:+.1f})")
            # At 2025, high carbon should reduce emissions via coal→gas switching
            if yr_idx == 0 and abs(em_diff) > 0.01:
                print(f"  ✓ Carbon price affects emissions at {yr_label}")
            elif yr_idx == 0:
                print(f"  ✗ FAIL: Carbon price should affect emissions at {yr_label}")
                all_pass = False

        if "baseline" in results and "low_clean_lcoe" in results:
            b_yr = get_year(results["baseline"], yr_idx)
            l_yr = get_year(results["low_clean_lcoe"], yr_idx)
            b_cost = b_yr.get("cost_per_mwh", 0)
            l_cost = l_yr.get("cost_per_mwh", 0)
            cost_diff = l_cost - b_cost
            print(f"\n  [{yr_label}] Low LCOE cost: ${b_cost:.1f} → ${l_cost:.1f} (Δ ${cost_diff:+.1f})")
            if abs(cost_diff) > 0.01:
                print(f"  ✓ LCOE level affects deployment cost at {yr_label}")
            else:
                # Low LCOE may not differentiate at year 0 if no deployment has happened
                if yr_idx > 0:
                    print(f"  ✗ FAIL: LCOE level should affect cost at {yr_label}")
                    all_pass = False
                else:
                    print(f"  ~ No deployment cost difference at baseline year (expected)")

        if "baseline" in results and "tx_overrides_high" in results:
            b_yr = get_year(results["baseline"], yr_idx)
            t_yr = get_year(results["tx_overrides_high"], yr_idx)
            b_clean = b_yr.get("clean_pct", 0)
            t_clean = t_yr.get("clean_pct", 0)
            diff = t_clean - b_clean
            print(f"\n  [{yr_label}] TX overrides clean%: {b_clean:.1f}% → {t_clean:.1f}% (Δ {diff:+.1f}pp)")
            if abs(diff) > 0.01:
                print(f"  ✓ TX overrides affect deployment at {yr_label}")

    if "baseline" in results and "custom_new_build_hr" in results:
        b = results["baseline"]
        c = results["custom_new_build_hr"]
        b_ccs = b.get("ccs_breakeven", 0)
        c_ccs = c.get("ccs_breakeven", 0)
        b_gas = b.get("new_gas_breakeven", 0)
        c_gas = c.get("new_gas_breakeven", 0)
        ccs_diff = c_ccs - b_ccs
        gas_diff = c_gas - b_gas
        print(f"\n  Custom new-build HR (7.5 vs 6.2):")
        print(f"    CCS breakeven: ${b_ccs:.1f} → ${c_ccs:.1f} (Δ ${ccs_diff:+.1f})")
        print(f"    New gas breakeven: ${b_gas:.1f} → ${c_gas:.1f} (Δ ${gas_diff:+.1f})")
        if abs(gas_diff) < 0.01 and abs(ccs_diff) < 0.01:
            print(f"  ✗ FAIL: CCS/new-gas breakeven should change with new-build heat rate")
            all_pass = False
        else:
            print(f"  ✓ New-build heat rate affects breakeven economics")

    if "baseline" in results and "high_demand_growth" in results:
        b = results["baseline"]
        d = results["high_demand_growth"]
        demand_diff = d["final_demand_twh"] - b["final_demand_twh"]
        print(f"\n  High demand growth vs baseline demand: {b['final_demand_twh']:.0f} → {d['final_demand_twh']:.0f} TWh (Δ {demand_diff:+.0f})")
        if abs(demand_diff) < 0.1:
            print(f"  ✗ FAIL: Demand should increase with higher growth rate")
            all_pass = False
        else:
            print(f"  ✓ Demand growth is responsive")

    # ── Trajectory Monotonicity Test ──
    print(f"\n{'='*60}")
    print("TRAJECTORY TESTS")
    print(f"{'='*60}")
    for name, metrics in results.items():
        years_data = metrics["years"]
        if len(years_data) < 2:
            continue
        # Demand should be non-decreasing (growth rate is positive)
        demands = [y["demand_twh"] for y in years_data if y["demand_twh"]]
        if all(d2 >= d1 - 0.1 for d1, d2 in zip(demands, demands[1:])):
            print(f"  ✓ {name}: Demand monotonically non-decreasing")
        else:
            print(f"  ✗ {name}: Demand DECREASED during trajectory: {demands}")
            all_pass = False

    # ── Summary ──
    print(f"\n{'='*60}")
    if all_pass:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print(f"{'='*60}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
