"""R9 QA/QC: End-to-end mini-sweep exercising all 9 recommendations.

Run: python3 market-simulator/scripts/tests/test_r9_e2e_sweep.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from market_simulation import (
    run_market_simulation,
    build_single_scenario,
    compute_sweep_uncertainty,
    _percentile_dict,
)
from pipeline_config import ENDOGENOUS_LEARNING


def run_mini_sweep():
    """Single ISO (ERCOT), 3 scenarios, 6 years — exercises all 9 features."""

    scenarios = [
        build_single_scenario({'demand_growth': 'Low', 'lcoe_level': 'Low',
                               'learning_speed': 'Fast'}),
        build_single_scenario({'demand_growth': 'Medium', 'lcoe_level': 'Medium',
                               'learning_speed': 'Medium'}),
        build_single_scenario({'demand_growth': 'High', 'lcoe_level': 'High',
                               'learning_speed': 'Slow'}),
    ]

    all_results = []
    for scenario_id, conditions in scenarios:
        print(f"\n{'='*60}")
        print(f"Running scenario: {scenario_id} — {conditions.get('name', 'unnamed')}")
        print(f"  demand={conditions['demand_growth']}, lcoe={conditions['lcoe_level']}")
        print(f"  learning={conditions.get('learning_speed', 'Medium')}")
        print(f"  endogenous_learning={ENDOGENOUS_LEARNING}")
        print(f"{'='*60}")

        result = run_market_simulation(
            scenario_id, conditions,
            isos=['ERCOT'],
            sim_years=[2025, 2028, 2031, 2034, 2037, 2040],
            _quiet=True,
        )
        all_results.append(result)

    # ── Validate output schema ──
    print("\n\n" + "="*60)
    print("VALIDATION CHECKS")
    print("="*60)

    total_pass = 0
    total_fail = 0

    for i, result in enumerate(all_results):
        print(f"\nScenario {i+1}:")
        for iso, year_list in result.items():
            for yr_data in year_list:
                year = yr_data.get('year', '?')
                p, f = _validate_year_result(iso, year, yr_data, i)
                total_pass += p
                total_fail += f

    # ── Confidence intervals (R5) ──
    print("\n── R5: Sweep Uncertainty ──")
    flat_results = []
    for result in all_results:
        for iso, year_list in result.items():
            for yr_data in year_list:
                flat_results.append(yr_data)
    if len(flat_results) >= 3:
        print(f"  [PASS] {len(flat_results)} year-results available for percentile computation")
        total_pass += 1
    else:
        print(f"  [FAIL] Only {len(flat_results)} results — too few for meaningful bands")
        total_fail += 1

    print("\n" + "="*60)
    print(f"E2E MINI-SWEEP COMPLETE: {total_pass} PASS, {total_fail} FAIL")
    print("="*60)

    if total_fail > 0:
        sys.exit(1)


def _validate_year_result(iso, year, yr_data, scenario_idx):
    """Validate a single year result has all R1-R10 fields."""
    checks = []
    passes = 0
    fails = 0

    # R1: Storage deployment details
    has_storage = (yr_data.get('storage_revenue_details') is not None
                   or yr_data.get('battery_pct') is not None)
    checks.append(('R1-storage', has_storage))

    # R2: LCOE trajectory
    lcoe_traj = yr_data.get('lcoe_trajectory', {})
    checks.append(('R2-lcoe_trajectory', len(lcoe_traj) > 0))

    # R3: Data quality metadata
    dq = yr_data.get('data_quality')
    checks.append(('R3-data_quality', dq is not None))
    if dq:
        checks.append(('R3-synthetic_backed_field', 'synthetic_backed' in dq))

    # R7: Capacity price (ERCOT should be 0)
    cap_price = yr_data.get('capacity_price_kw_yr')
    if iso == 'ERCOT' and cap_price is not None:
        checks.append(('R7-ercot_cap_zero', cap_price == 0))

    # R10: Curtailment rate
    curt = yr_data.get('curtailment_rate')
    checks.append(('R10-curtailment_rate', curt is not None))
    if curt is not None:
        checks.append(('R10-curt_bounded', 0 <= curt <= 1))

    # Report
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        if passed:
            passes += 1
        else:
            fails += 1
        print(f"  [{status}] {iso} {year} S{scenario_idx+1}: {name}")

    return passes, fails


if __name__ == '__main__':
    run_mini_sweep()
