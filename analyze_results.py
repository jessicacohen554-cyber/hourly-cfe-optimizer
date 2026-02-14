#!/usr/bin/env python3
"""
Post-Optimizer Analysis Script
================================
Runs after optimize_overprocure.py completes. Performs:
1. QA/QC: monotonicity validation, literature alignment checks
2. Resource mix analysis: VRE waste between lower and >90% targets
3. Curtailment quantification: inputs for DAC-VRE co-optimization
4. Summary statistics for dashboard/narrative updates

Reads: dashboard/overprocure_results.json (or data/optimizer_cache.json)
Outputs: analysis summary to stdout + data/analysis_results.json
"""

import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(SCRIPT_DIR, 'dashboard', 'overprocure_results.json')
CACHE_PATH = os.path.join(SCRIPT_DIR, 'data', 'optimizer_cache.json')
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'data', 'analysis_results.json')

THRESHOLDS = [75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 100]
ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO']
MEDIUM_KEY = 'MMM_M_M'

# Literature reference ranges for validation ($/MWh effective cost at Medium scenario)
# Sources: NREL ATB 2024, Lazard LCOE 16.0, LBNL Utility-Scale Solar/Wind 2024
LITERATURE_RANGES = {
    # (min_cost, max_cost) at 90% hourly matching, Medium scenario
    '90': {
        'CAISO': (55, 100),
        'ERCOT': (45, 85),
        'PJM':   (55, 100),
        'NYISO': (65, 120),
        'NEISO': (60, 115),
    },
    # Very rough ranges for 95% matching
    '95': {
        'CAISO': (70, 140),
        'ERCOT': (55, 120),
        'PJM':   (70, 140),
        'NYISO': (85, 170),
        'NEISO': (80, 160),
    }
}


def load_results():
    """Load optimizer results from either the dashboard file or cache."""
    for path in [RESULTS_PATH, CACHE_PATH]:
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            print(f"Loaded results from: {path}")
            print(f"  File size: {os.path.getsize(path) / 1024:.0f} KB")
            return data
    print("ERROR: No results file found!")
    sys.exit(1)


def check_monotonicity(data):
    """
    Check that cost is non-decreasing across thresholds for each scenario.
    Returns dict of violations by ISO.
    """
    print("\n" + "=" * 70)
    print("  1. MONOTONICITY VALIDATION")
    print("=" * 70)

    all_violations = {}
    total_checks = 0
    total_violations = 0

    for iso in ISOS:
        if iso not in data['results']:
            continue

        iso_data = data['results'][iso]
        thresholds_data = iso_data.get('thresholds', {})
        violations = []

        # Collect all scenario keys across thresholds
        all_scenario_keys = set()
        for t_str in thresholds_data:
            scenarios = thresholds_data[t_str].get('scenarios', {})
            all_scenario_keys.update(scenarios.keys())

        for sk in sorted(all_scenario_keys):
            prev_cost = None
            prev_t = None
            for threshold in THRESHOLDS:
                t_str = str(threshold)
                if t_str not in thresholds_data:
                    continue
                scenarios = thresholds_data[t_str].get('scenarios', {})
                if sk not in scenarios:
                    continue
                result = scenarios[sk]
                if 'costs' not in result:
                    continue

                cost = result['costs'].get('effective_cost',
                       result['costs'].get('effective_cost_per_useful_mwh', 0))
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


def check_literature_alignment(data):
    """Check Medium scenario costs against published literature ranges."""
    print("\n" + "=" * 70)
    print("  2. LITERATURE ALIGNMENT CHECK")
    print("=" * 70)

    warnings = []

    for iso in ISOS:
        if iso not in data['results']:
            continue

        iso_data = data['results'][iso]
        thresholds_data = iso_data.get('thresholds', {})
        print(f"\n  {iso}:")

        for t_check in ['90', '95']:
            if t_check not in thresholds_data:
                continue
            scenarios = thresholds_data[t_check].get('scenarios', {})
            if MEDIUM_KEY not in scenarios:
                continue

            result = scenarios[MEDIUM_KEY]
            if 'costs' not in result:
                continue

            cost = result['costs'].get('effective_cost',
                   result['costs'].get('effective_cost_per_useful_mwh', 0))
            expected = LITERATURE_RANGES.get(t_check, {}).get(iso, (0, 999))

            status = "OK" if expected[0] <= cost <= expected[1] else "WARNING"
            print(f"    {t_check}% Medium: ${cost:.2f}/MWh "
                  f"(expected ${expected[0]}-${expected[1]}) — {status}")
            if status == "WARNING":
                warnings.append(f"{iso} {t_check}%: ${cost:.2f} outside [{expected[0]}, {expected[1]}]")

    if warnings:
        print(f"\n  ⚠ {len(warnings)} warning(s) — review these costs against sources")
    else:
        print(f"\n  All Medium scenario costs within expected literature ranges")

    return warnings


def analyze_resource_mixes(data):
    """Analyze resource mix evolution across thresholds — identify VRE waste."""
    print("\n" + "=" * 70)
    print("  3. RESOURCE MIX ANALYSIS — VRE WASTE BETWEEN TARGETS")
    print("=" * 70)

    mix_analysis = {}

    for iso in ISOS:
        if iso not in data['results']:
            continue

        iso_data = data['results'][iso]
        thresholds_data = iso_data.get('thresholds', {})

        print(f"\n  {iso}:")
        print(f"    {'Threshold':>10}  {'CF':>5} {'Sol':>5} {'Wnd':>5} {'CCS':>5} {'Hyd':>5} "
              f"{'Batt':>5} {'LDES':>5} {'Proc%':>6} {'Cost':>8} {'Curt%':>6}")
        print(f"    {'-'*80}")

        iso_mixes = {}
        for threshold in THRESHOLDS:
            t_str = str(threshold)
            if t_str not in thresholds_data:
                continue

            scenarios = thresholds_data[t_str].get('scenarios', {})
            if MEDIUM_KEY not in scenarios:
                continue

            result = scenarios[MEDIUM_KEY]
            mix = result.get('resource_mix', {})
            costs = result.get('costs', {})
            cost_val = costs.get('effective_cost', costs.get('effective_cost_per_useful_mwh', 0))
            curt = costs.get('curtailment_pct', 0)
            proc = result.get('procurement_pct', 0)
            batt = result.get('battery_dispatch_pct', 0)
            ldes = result.get('ldes_dispatch_pct', 0)

            iso_mixes[threshold] = {
                'mix': mix,
                'cost': cost_val,
                'curtailment_pct': curt,
                'procurement_pct': proc,
                'battery_pct': batt,
                'ldes_pct': ldes,
            }

            vre_pct = mix.get('solar', 0) + mix.get('wind', 0)
            print(f"    {threshold:>8}%  "
                  f"{mix.get('clean_firm', 0):>5} {mix.get('solar', 0):>5} {mix.get('wind', 0):>5} "
                  f"{mix.get('ccs_ccgt', 0):>5} {mix.get('hydro', 0):>5} "
                  f"{batt:>5.1f} {ldes:>5.1f} "
                  f"{proc:>6.1f} ${cost_val:>7.2f} {curt:>5.1f}%")

        # Compute VRE waste: how much of the VRE capacity at lower thresholds
        # becomes underutilized or stranded at higher thresholds
        if 90 in iso_mixes and 95 in iso_mixes:
            mix_90 = iso_mixes[90]['mix']
            mix_95 = iso_mixes[95]['mix']
            vre_90 = mix_90.get('solar', 0) + mix_90.get('wind', 0)
            vre_95 = mix_95.get('solar', 0) + mix_95.get('wind', 0)
            curt_90 = iso_mixes[90]['curtailment_pct']
            curt_95 = iso_mixes[95]['curtailment_pct']
            cost_jump = iso_mixes[95]['cost'] - iso_mixes[90]['cost']

            print(f"\n    90%→95% transition:")
            print(f"      VRE share: {vre_90}% → {vre_95}% (Δ {vre_95 - vre_90:+.1f}%)")
            print(f"      Curtailment: {curt_90:.1f}% → {curt_95:.1f}% (Δ {curt_95 - curt_90:+.1f}%)")
            print(f"      Cost jump: ${cost_jump:+.2f}/MWh")

        mix_analysis[iso] = iso_mixes

    return mix_analysis


def analyze_curtailment_for_dac(data):
    """Quantify curtailed energy at each threshold — inputs for DAC-VRE analysis."""
    print("\n" + "=" * 70)
    print("  4. CURTAILMENT ANALYSIS — DAC-VRE CO-OPTIMIZATION INPUTS")
    print("=" * 70)

    dac_inputs = {}

    for iso in ISOS:
        if iso not in data['results']:
            continue

        iso_data = data['results'][iso]
        thresholds_data = iso_data.get('thresholds', {})
        annual_demand = iso_data.get('annual_demand_mwh', 0)

        print(f"\n  {iso} (annual demand: {annual_demand/1e6:.1f} TWh):")
        print(f"    {'Threshold':>10} {'Proc%':>7} {'Match%':>7} {'Curt%':>7} "
              f"{'Curt TWh':>9} {'DAC Mt':>7} {'MAC':>8}")
        print(f"    {'-'*65}")

        iso_dac = {}
        for threshold in THRESHOLDS:
            t_str = str(threshold)
            if t_str not in thresholds_data:
                continue

            scenarios = thresholds_data[t_str].get('scenarios', {})
            if MEDIUM_KEY not in scenarios:
                continue

            result = scenarios[MEDIUM_KEY]
            costs = result.get('costs', {})
            proc = result.get('procurement_pct', 0)
            match_score = result.get('hourly_match_score', 0)
            curt = costs.get('curtailment_pct', 0)

            # Curtailed energy in MWh
            proc_factor = proc / 100.0
            match_factor = match_score / 100.0
            total_procured = annual_demand * proc_factor
            curtailed_mwh = total_procured * (curt / 100.0) if curt > 0 else 0

            # DAC capacity at 2 MWh/ton CO2
            dac_tons = curtailed_mwh / 2.0

            # MAC (marginal abatement cost): cost increase per % of threshold increase
            cost_val = costs.get('effective_cost', costs.get('effective_cost_per_useful_mwh', 0))
            co2 = result.get('co2_abated', {})
            co2_tons = co2.get('tons_co2_abated', 0) if isinstance(co2, dict) else 0

            # MAC = incremental cost / incremental CO2 (approximate)
            mac = 0
            if co2_tons > 0 and annual_demand > 0:
                incremental = costs.get('incremental_above_baseline', 0)
                mac = (incremental * annual_demand) / co2_tons if co2_tons > 0 else 0

            iso_dac[threshold] = {
                'procurement_pct': proc,
                'curtailment_pct': curt,
                'curtailed_mwh': round(curtailed_mwh),
                'dac_potential_tons': round(dac_tons),
                'mac_per_ton': round(mac, 2),
            }

            print(f"    {threshold:>8}%  {proc:>6.1f} {match_score:>6.1f} {curt:>6.1f} "
                  f"{curtailed_mwh/1e6:>8.3f}  {dac_tons/1e6:>6.3f} ${mac:>7.0f}")

        dac_inputs[iso] = iso_dac

    return dac_inputs


def print_summary(data):
    """Print high-level summary statistics."""
    print("\n" + "=" * 70)
    print("  5. SUMMARY STATISTICS")
    print("=" * 70)

    for iso in ISOS:
        if iso not in data['results']:
            continue

        iso_data = data['results'][iso]
        thresholds_data = iso_data.get('thresholds', {})
        total_scenarios = sum(
            thresholds_data[t].get('scenario_count', len(thresholds_data[t].get('scenarios', {})))
            for t in thresholds_data
        )

        print(f"\n  {iso}:")
        print(f"    Thresholds computed: {len(thresholds_data)}")
        print(f"    Total scenario-results: {total_scenarios}")
        print(f"    Annual demand: {iso_data.get('annual_demand_mwh', 0)/1e6:.1f} TWh")

        # Min/max cost at Medium scenario across thresholds
        costs = []
        for threshold in THRESHOLDS:
            t_str = str(threshold)
            if t_str not in thresholds_data:
                continue
            scenarios = thresholds_data[t_str].get('scenarios', {})
            if MEDIUM_KEY in scenarios:
                result = scenarios[MEDIUM_KEY]
                c = result.get('costs', {})
                cost_val = c.get('effective_cost', c.get('effective_cost_per_useful_mwh', 0))
                if cost_val > 0:
                    costs.append((threshold, cost_val))

        if costs:
            min_c = min(costs, key=lambda x: x[1])
            max_c = max(costs, key=lambda x: x[1])
            print(f"    Cost range (Medium): ${min_c[1]:.2f}/MWh ({min_c[0]}%) → ${max_c[1]:.2f}/MWh ({max_c[0]}%)")
            print(f"    Cost multiplier: {max_c[1]/min_c[1]:.1f}x")


def main():
    print("=" * 70)
    print("  POST-OPTIMIZER ANALYSIS")
    print("=" * 70)

    data = load_results()

    # Check what we got
    results = data.get('results', {})
    isos_present = [iso for iso in ISOS if iso in results]
    print(f"  ISOs in results: {isos_present}")

    for iso in isos_present:
        thresholds_present = sorted([float(t) for t in results[iso].get('thresholds', {}).keys()])
        print(f"  {iso} thresholds: {thresholds_present}")

    # Run all analyses
    violations = check_monotonicity(data)
    warnings = check_literature_alignment(data)
    mix_analysis = analyze_resource_mixes(data)
    dac_inputs = analyze_curtailment_for_dac(data)
    print_summary(data)

    # Save analysis results
    output = {
        'monotonicity': {iso: len(v) for iso, v in violations.items()},
        'monotonicity_details': violations,
        'literature_warnings': warnings,
        'mix_analysis': {},
        'dac_inputs': dac_inputs,
    }

    # Convert mix analysis (has non-serializable keys)
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
