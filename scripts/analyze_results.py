#!/usr/bin/env python3
"""
Post-Optimizer Analysis Script
================================
Runs after the pipeline completes. Performs:
1. QA/QC: monotonicity validation, literature alignment checks
2. Resource mix analysis: VRE waste between lower and >90% targets
3. Curtailment quantification: inputs for DAC-VRE co-optimization
4. Summary statistics for dashboard/narrative updates

Reads from: data/step5-post-processing/co2_results/{ISO}_{threshold}_2025.json
Outputs: analysis summary to stdout + data/analysis_results.json
"""

import json
import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
CO2_DIR = os.path.join(ROOT_DIR, 'data', 'step5-post-processing', 'co2_results')
OUTPUT_PATH = os.path.join(ROOT_DIR, 'data', 'analysis_results.json')

# Import constants from step3
sys.path.insert(0, SCRIPT_DIR)
from pipeline_config import REGIONAL_DEMAND_TWH

from pipeline_config import OUTPUT_THRESHOLDS as THRESHOLDS
ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO']


def medium_key(iso):
    """Return the all-Medium scenario key for a given ISO (9-dim format)."""
    return 'MMM_M_M_M_M1_M' if iso == 'CAISO' else 'MMM_M_M_M_M1_X'

OLD_MEDIUM_KEYS = ['MMM_M_M_M_M1_M', 'MMM_M_M_M_M1_X', 'MMM_M_M']


_PARQUET_CACHE = {}


def _load_iso_parquet(iso):
    """Load an ISO's CO2 parquet into the cache (lazy, one-time)."""
    if iso in _PARQUET_CACHE:
        return
    _PARQUET_CACHE[iso] = {}
    for prefix in ['co2_', 'step4_', 'step3_co_']:
        ppath = os.path.join(CO2_DIR, f'{prefix}{iso}.parquet')
        if os.path.exists(ppath):
            try:
                sys.path.insert(0, ROOT_DIR)
                from parquet_io import load_from_parquets
                data = load_from_parquets(CO2_DIR, [iso])
                iso_data = data.get('results', {}).get(iso, {})
                for t_str, t_data in iso_data.get('thresholds', {}).items():
                    _PARQUET_CACHE[iso][t_str] = t_data.get('scenarios', {})
                return
            except Exception:
                pass


def load_scenarios(iso, threshold):
    """Load scenarios from CO2 parquets or batch JSON files."""
    _load_iso_parquet(iso)
    t_str = str(threshold)
    if iso in _PARQUET_CACHE and t_str in _PARQUET_CACHE[iso]:
        return _PARQUET_CACHE[iso][t_str]

    # Fall back to legacy JSON
    fname = f"{iso}_{t_str}_2025.json"
    fpath = os.path.join(CO2_DIR, fname)
    if not os.path.exists(fpath):
        return {}
    with open(fpath) as f:
        data = json.load(f)
    return data.get('scenarios', {})


def get_medium_scenario(scenarios, iso):
    """Get medium scenario data with fallback to old key formats."""
    key = medium_key(iso)
    if key in scenarios:
        return scenarios[key]
    for fallback in OLD_MEDIUM_KEYS:
        if fallback in scenarios:
            return scenarios[fallback]
    return None


# Literature reference ranges for validation ($/MWh effective cost at Medium scenario)
# Sources: NREL ATB 2024, Lazard LCOE 16.0, LBNL Utility-Scale Solar/Wind 2024
LITERATURE_RANGES = {
    '90': {
        'CAISO': (55, 100),
        'ERCOT': (45, 85),
        'PJM':   (55, 100),
        'NYISO': (65, 120),
        'NEISO': (60, 115),
    },
    '95': {
        'CAISO': (70, 140),
        'ERCOT': (55, 120),
        'PJM':   (70, 140),
        'NYISO': (85, 170),
        'NEISO': (80, 160),
    }
}


def check_monotonicity():
    """Check that cost is non-decreasing across thresholds for each scenario."""
    print("\n" + "=" * 70)
    print("  1. MONOTONICITY VALIDATION")
    print("=" * 70)

    all_violations = {}
    total_checks = 0
    total_violations = 0

    for iso in ISOS:
        # Collect all scenario keys and costs across thresholds
        threshold_scenarios = {}
        all_scenario_keys = set()
        for threshold in THRESHOLDS:
            scenarios = load_scenarios(iso, threshold)
            threshold_scenarios[threshold] = scenarios
            all_scenario_keys.update(scenarios.keys())

        violations = []
        for sk in sorted(all_scenario_keys):
            prev_cost = None
            prev_t = None
            for threshold in THRESHOLDS:
                scenarios = threshold_scenarios[threshold]
                if sk not in scenarios:
                    continue
                result = scenarios[sk]
                if 'costs' not in result:
                    continue

                cost = result['costs'].get('effective_cost', 0)
                total_checks += 1

                if prev_cost is not None and cost < prev_cost - 0.01:
                    violations.append({
                        'scenario': sk,
                        'lower_threshold': prev_t,
                        'higher_threshold': threshold,
                        'lower_cost': prev_cost,
                        'higher_cost': cost,
                        'delta': round(prev_cost - cost, 2),
                    })
                    total_violations += 1

                prev_cost = cost
                prev_t = threshold

        all_violations[iso] = violations
        status = "PASS" if len(violations) == 0 else f"FAIL ({len(violations)} violations)"
        print(f"  {iso}: {status}")
        if violations:
            for v in violations[:5]:
                print(f"    {v['scenario']}: {v['lower_threshold']}% (${v['lower_cost']:.2f}) > "
                      f"{v['higher_threshold']}% (${v['higher_cost']:.2f}) — Δ${v['delta']:.2f}")
            if len(violations) > 5:
                print(f"    ... and {len(violations) - 5} more")

    print(f"\n  Total checks: {total_checks}, Violations: {total_violations}")
    return all_violations


def check_literature_alignment():
    """Check Medium scenario costs against published literature ranges."""
    print("\n" + "=" * 70)
    print("  2. LITERATURE ALIGNMENT CHECK")
    print("=" * 70)

    warnings = []

    for iso in ISOS:
        print(f"\n  {iso}:")

        for t_check in ['90', '95']:
            scenarios = load_scenarios(iso, t_check)
            if not scenarios:
                continue
            result = get_medium_scenario(scenarios, iso)
            if result is None or 'costs' not in result:
                continue

            cost = result['costs'].get('effective_cost', 0)
            expected = LITERATURE_RANGES.get(t_check, {}).get(iso, (0, 999))

            status = "OK" if expected[0] <= cost <= expected[1] else "WARNING"
            print(f"    {t_check}% Medium: ${cost:.2f}/MWh "
                  f"(expected ${expected[0]}-${expected[1]}) — {status}")
            if status == "WARNING":
                warnings.append(f"{iso} {t_check}%: ${cost:.2f} outside [{expected[0]}, {expected[1]}]")

    if warnings:
        print(f"\n  {len(warnings)} warning(s) — review these costs against sources")
    else:
        print(f"\n  All Medium scenario costs within expected literature ranges")

    return warnings


def analyze_resource_mixes():
    """Analyze resource mix evolution across thresholds — identify VRE waste."""
    print("\n" + "=" * 70)
    print("  3. RESOURCE MIX ANALYSIS — VRE WASTE BETWEEN TARGETS")
    print("=" * 70)

    mix_analysis = {}

    for iso in ISOS:
        print(f"\n  {iso}:")
        print(f"    {'Threshold':>10}  {'CF':>5} {'Sol':>5} {'Wnd':>5} {'CCS':>5} {'Hyd':>5} "
              f"{'Batt':>5} {'LDES':>5} {'H2':>5} {'Cost':>8}")
        print(f"    {'-'*70}")

        iso_mixes = {}
        for threshold in THRESHOLDS:
            scenarios = load_scenarios(iso, threshold)
            if not scenarios:
                continue

            result = get_medium_scenario(scenarios, iso)
            if result is None:
                continue

            mix = result.get('resource_mix', {})
            costs = result.get('costs', {})
            cost_val = costs.get('effective_cost', 0)
            batt = result.get('battery_dispatch_pct', 0)
            ldes = result.get('ldes_dispatch_pct', 0)
            h2 = result.get('h2_dispatch_pct', 0)

            iso_mixes[threshold] = {
                'mix': mix,
                'cost': cost_val,
                'battery_pct': batt,
                'ldes_pct': ldes,
                'h2_pct': h2,
            }

            print(f"    {threshold:>8}%  "
                  f"{mix.get('clean_firm', 0):>5} {mix.get('solar', 0):>5} {mix.get('wind', 0):>5} "
                  f"{mix.get('ccs_ccgt', 0):>5} {mix.get('hydro', 0):>5} "
                  f"{batt:>5.1f} {ldes:>5.1f} {h2:>5.1f} "
                  f"${cost_val:>7.2f}")

        # Compute VRE waste: 90→95 transition
        if 90 in iso_mixes and 95 in iso_mixes:
            mix_90 = iso_mixes[90]['mix']
            mix_95 = iso_mixes[95]['mix']
            vre_90 = mix_90.get('solar', 0) + mix_90.get('wind', 0)
            vre_95 = mix_95.get('solar', 0) + mix_95.get('wind', 0)
            cost_jump = iso_mixes[95]['cost'] - iso_mixes[90]['cost']

            print(f"\n    90%→95% transition:")
            print(f"      VRE share: {vre_90}% → {vre_95}% (Δ {vre_95 - vre_90:+.1f}%)")
            print(f"      Cost jump: ${cost_jump:+.2f}/MWh")

        mix_analysis[iso] = iso_mixes

    return mix_analysis


def analyze_curtailment_for_dac():
    """Quantify curtailed energy at each threshold — inputs for DAC-VRE analysis."""
    print("\n" + "=" * 70)
    print("  4. CURTAILMENT ANALYSIS — DAC-VRE CO-OPTIMIZATION INPUTS")
    print("=" * 70)

    dac_inputs = {}

    for iso in ISOS:
        annual_demand_mwh = REGIONAL_DEMAND_TWH.get(iso, 0) * 1e6

        print(f"\n  {iso} (annual demand: {annual_demand_mwh/1e6:.1f} TWh):")
        print(f"    {'Threshold':>10} {'Match%':>7} "
              f"{'MAC':>8}")
        print(f"    {'-'*35}")

        iso_dac = {}
        for threshold in THRESHOLDS:
            scenarios = load_scenarios(iso, threshold)
            if not scenarios:
                continue

            result = get_medium_scenario(scenarios, iso)
            if result is None:
                continue

            costs = result.get('costs', {})
            match_score = result.get('hourly_match_score', 0)

            # MAC from co2_abated
            co2 = result.get('co2_abated', {})
            co2_tons = co2.get('tons_co2_abated', 0) if isinstance(co2, dict) else 0
            mac = 0
            if co2_tons > 0 and annual_demand_mwh > 0:
                incremental = costs.get('incremental', 0)
                mac = (incremental * annual_demand_mwh) / co2_tons if co2_tons > 0 else 0

            iso_dac[threshold] = {
                'mac_per_ton': round(mac, 2),
            }

            print(f"    {threshold:>8}%  {match_score:>6.1f} "
                  f"${mac:>7.0f}")

        dac_inputs[iso] = iso_dac

    return dac_inputs


def print_summary():
    """Print high-level summary statistics."""
    print("\n" + "=" * 70)
    print("  5. SUMMARY STATISTICS")
    print("=" * 70)

    for iso in ISOS:
        scenario_count = 0
        threshold_count = 0
        costs = []

        for threshold in THRESHOLDS:
            scenarios = load_scenarios(iso, threshold)
            if not scenarios:
                continue
            threshold_count += 1
            scenario_count += len(scenarios)

            result = get_medium_scenario(scenarios, iso)
            if result is not None:
                c = result.get('costs', {})
                cost_val = c.get('effective_cost', 0)
                if cost_val > 0:
                    costs.append((threshold, cost_val))

        demand_twh = REGIONAL_DEMAND_TWH.get(iso, 0)
        print(f"\n  {iso}:")
        print(f"    Thresholds computed: {threshold_count}")
        print(f"    Total scenario-results: {scenario_count}")
        print(f"    Annual demand: {demand_twh:.1f} TWh")

        if costs:
            min_c = min(costs, key=lambda x: x[1])
            max_c = max(costs, key=lambda x: x[1])
            print(f"    Cost range (Medium): ${min_c[1]:.2f}/MWh ({min_c[0]}%) → ${max_c[1]:.2f}/MWh ({max_c[0]}%)")
            if min_c[1] > 0:
                print(f"    Cost multiplier: {max_c[1]/min_c[1]:.1f}x")


def main():
    print("=" * 70)
    print("  POST-OPTIMIZER ANALYSIS")
    print("=" * 70)

    if not os.path.isdir(CO2_DIR):
        print(f"ERROR: co2_results directory not found: {CO2_DIR}")
        sys.exit(1)

    # Check which ISOs/thresholds are available
    for iso in ISOS:
        thresholds_present = []
        for t in THRESHOLDS:
            fpath = os.path.join(CO2_DIR, f"{iso}_{t}_2025.json")
            if os.path.exists(fpath):
                thresholds_present.append(t)
        print(f"  {iso} thresholds: {thresholds_present}")

    # Run all analyses
    violations = check_monotonicity()
    warnings = check_literature_alignment()
    mix_analysis = analyze_resource_mixes()
    dac_inputs = analyze_curtailment_for_dac()
    print_summary()

    # Save analysis results
    output = {
        'monotonicity': {iso: len(v) for iso, v in violations.items()},
        'monotonicity_details': violations,
        'literature_warnings': warnings,
        'mix_analysis': {},
        'dac_inputs': dac_inputs,
    }

    for iso, mixes in mix_analysis.items():
        output['mix_analysis'][iso] = {
            str(k): v for k, v in mixes.items()
        }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Analysis saved to: {OUTPUT_PATH}")

    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
