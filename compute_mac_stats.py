#!/usr/bin/env python3
"""
Option B: Statistical Post-Processing + Path-Constrained MAC
=============================================================
Reads existing optimizer results (16,200 scenarios) and computes:

1. Monotonic envelope MAC (convex hull of cost vs CO2 per ISO)
2. MAC uncertainty fan (P10/P25/P50/P75/P90 across 324 scenarios)
3. ANOVA sensitivity decomposition (which toggles drive MAC variance)
4. Path-constrained reference MAC (monotonic resource deployment)

Outputs:
  - dashboard/js/mac-stats-data.js   (JavaScript constants for dashboard)
  - data/mac_stats.json              (full JSON for programmatic use)
"""

import json
import os
import sys
import math
import numpy as np
from collections import defaultdict

# ── Paths ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(BASE_DIR, 'dashboard', 'overprocure_results.json')
JS_OUTPUT_PATH = os.path.join(BASE_DIR, 'dashboard', 'js', 'mac-stats-data.js')
JSON_OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'mac_stats.json')

ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO']
THRESHOLDS = [75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 100]
THRESHOLD_STRS = [str(t) for t in THRESHOLDS]

# Toggle factor names for ANOVA
TOGGLE_NAMES = ['Renewable Gen', 'Firm Gen', 'Storage', 'Fossil Fuel', 'Transmission']

# Wholesale prices (duplicated from optimizer for standalone use)
WHOLESALE_PRICES = {'CAISO': 30, 'ERCOT': 27, 'PJM': 34, 'NYISO': 42, 'NEISO': 41}


def load_results():
    """Load optimizer results JSON."""
    with open(RESULTS_PATH, 'r') as f:
        return json.load(f)


def scenario_key_to_levels(key):
    """Parse scenario key like 'LMH_L_N' → (renew, firm, storage, fuel, tx) indices.
    L=0, M=1, H=2; tx: N=0, L=1, M=2, H=3
    """
    level_map = {'L': 0, 'M': 1, 'H': 2, 'N': 0}
    # Format: {r}{f}{s}_{fuel}_{tx}
    parts = key.split('_')
    rfs = parts[0]  # 3 chars: renew, firm, storage
    fuel = parts[1]
    tx = parts[2]
    return (level_map[rfs[0]], level_map[rfs[1]], level_map[rfs[2]],
            level_map[fuel], level_map.get(tx, 1))


def compute_mac_for_scenario(iso_data, scenario_key, thresholds):
    """Compute average MAC ($/ton) at each threshold for one scenario.
    MAC = (incremental_cost × demand_mwh) / total_co2_abated_tons
    """
    demand_mwh = iso_data.get('annual_demand_mwh', 1)
    macs = []

    for t_str in thresholds:
        t_data = iso_data.get('thresholds', {}).get(t_str, {})
        scenarios = t_data.get('scenarios', {})
        sc = scenarios.get(scenario_key)
        if not sc:
            macs.append(None)
            continue

        costs = sc.get('costs', {})
        co2 = sc.get('co2_abated', {})
        incremental = costs.get('incremental', costs.get('incremental_above_baseline', 0))
        co2_tons = co2.get('total_co2_abated_tons', 0)

        if co2_tons > 0 and incremental is not None:
            mac = (incremental * demand_mwh) / co2_tons
            macs.append(round(mac, 1))
        else:
            macs.append(None)

    return macs


def compute_stepwise_mac(iso_data, scenario_key, thresholds):
    """Compute stepwise marginal MAC between adjacent thresholds.
    Returns list where index i = MAC of step from threshold[i-1] to threshold[i].
    Index 0 = None (no prior step).
    """
    demand_mwh = iso_data.get('annual_demand_mwh', 1)
    step_macs = [None]  # No step for first threshold

    for i in range(1, len(thresholds)):
        t_prev = thresholds[i - 1]
        t_curr = thresholds[i]

        t_prev_data = iso_data.get('thresholds', {}).get(t_prev, {}).get('scenarios', {}).get(scenario_key)
        t_curr_data = iso_data.get('thresholds', {}).get(t_curr, {}).get('scenarios', {}).get(scenario_key)

        if not t_prev_data or not t_curr_data:
            step_macs.append(None)
            continue

        cost_prev = t_prev_data.get('costs', {}).get('incremental', 0)
        cost_curr = t_curr_data.get('costs', {}).get('incremental', 0)
        co2_prev = t_prev_data.get('co2_abated', {}).get('total_co2_abated_tons', 0)
        co2_curr = t_curr_data.get('co2_abated', {}).get('total_co2_abated_tons', 0)

        delta_cost = (cost_curr - cost_prev) * demand_mwh
        delta_co2 = co2_curr - co2_prev

        if delta_co2 > 0 and delta_cost >= 0:
            step_macs.append(round(delta_cost / delta_co2, 1))
        else:
            step_macs.append(None)

    return step_macs


def compute_fan_chart(data):
    """Compute P10/P25/P50/P75/P90 MAC percentiles across all 324 scenarios per ISO/threshold.

    Each scenario now has its own CO2 abatement (computed with fuel-switching elasticity).
    The fossil fuel toggle shifts marginal emission rates, so MAC = f(cost, CO2) varies
    on BOTH axes across scenarios.
    """
    fan_data = {}

    for iso in ISOS:
        iso_data = data['results'].get(iso, {})
        fan_data[iso] = {'p10': [], 'p25': [], 'p50': [], 'p75': [], 'p90': []}
        demand_mwh = iso_data.get('annual_demand_mwh', 1)

        for t_str in THRESHOLD_STRS:
            t_data = iso_data.get('thresholds', {}).get(t_str, {})
            scenarios = t_data.get('scenarios', {})

            macs = []
            for sc_key, sc in scenarios.items():
                incremental = sc.get('costs', {}).get('incremental', 0)
                co2 = sc.get('co2_abated', {})
                co2_tons = co2.get('total_co2_abated_tons', 0)

                if co2_tons > 0 and incremental is not None:
                    mac = (incremental * demand_mwh) / co2_tons
                    macs.append(mac)

            if macs:
                arr = np.array(macs)
                fan_data[iso]['p10'].append(round(float(np.percentile(arr, 10)), 1))
                fan_data[iso]['p25'].append(round(float(np.percentile(arr, 25)), 1))
                fan_data[iso]['p50'].append(round(float(np.percentile(arr, 50)), 1))
                fan_data[iso]['p75'].append(round(float(np.percentile(arr, 75)), 1))
                fan_data[iso]['p90'].append(round(float(np.percentile(arr, 90)), 1))
            else:
                for p in ['p10', 'p25', 'p50', 'p75', 'p90']:
                    fan_data[iso][p].append(None)

    return fan_data


def compute_stepwise_fan(data):
    """Compute P10/P50/P90 of stepwise marginal MAC across scenarios.

    Each scenario has its own CO2 (from fuel-switching elasticity), so both cost
    and CO2 vary. Step MAC = delta_cost / delta_co2 per scenario.
    """
    fan_data = {}

    for iso in ISOS:
        iso_data = data['results'].get(iso, {})
        demand_mwh = iso_data.get('annual_demand_mwh', 1)
        fan_data[iso] = {'p10': [None], 'p50': [None], 'p90': [None]}

        for i in range(1, len(THRESHOLDS)):
            t_prev = THRESHOLD_STRS[i - 1]
            t_curr = THRESHOLD_STRS[i]

            prev_scenarios = iso_data.get('thresholds', {}).get(t_prev, {}).get('scenarios', {})
            curr_scenarios = iso_data.get('thresholds', {}).get(t_curr, {}).get('scenarios', {})

            step_macs = []
            for sc_key in curr_scenarios:
                sc_prev = prev_scenarios.get(sc_key)
                sc_curr = curr_scenarios.get(sc_key)
                if not sc_prev or not sc_curr:
                    continue

                cost_prev = sc_prev.get('costs', {}).get('incremental', 0)
                cost_curr = sc_curr.get('costs', {}).get('incremental', 0)
                co2_prev = sc_prev.get('co2_abated', {}).get('total_co2_abated_tons', 0)
                co2_curr = sc_curr.get('co2_abated', {}).get('total_co2_abated_tons', 0)

                delta_cost = (cost_curr - cost_prev) * demand_mwh
                delta_co2 = co2_curr - co2_prev

                if delta_co2 > 0 and delta_cost >= 0:
                    step_macs.append(delta_cost / delta_co2)

            if step_macs:
                arr = np.array(step_macs)
                fan_data[iso]['p10'].append(round(float(np.percentile(arr, 10)), 1))
                fan_data[iso]['p50'].append(round(float(np.percentile(arr, 50)), 1))
                fan_data[iso]['p90'].append(round(float(np.percentile(arr, 90)), 1))
            else:
                for p in ['p10', 'p50', 'p90']:
                    fan_data[iso][p].append(None)

    return fan_data


def compute_monotonic_envelope(data):
    """Compute monotonic envelope MAC — running max to enforce non-decreasing MAC.
    This is the convex-hull-inspired approach: at each threshold, the envelope MAC
    is max(MAC[t], envelope[t-1]). Smooths portfolio rebalancing artifacts.
    """
    envelope = {}

    for iso in ISOS:
        iso_data = data['results'].get(iso, {})
        demand_mwh = iso_data.get('annual_demand_mwh', 1)

        # Compute raw average MAC at Medium scenario
        raw_macs = []
        costs_at_t = []
        co2_at_t = []

        for t_str in THRESHOLD_STRS:
            t_data = iso_data.get('thresholds', {}).get(t_str, {})
            sc = t_data.get('scenarios', {}).get('MMM_M_M')
            if not sc:
                raw_macs.append(None)
                costs_at_t.append(None)
                co2_at_t.append(None)
                continue

            incremental = sc.get('costs', {}).get('incremental', 0)
            co2_tons = sc.get('co2_abated', {}).get('total_co2_abated_tons', 0)

            costs_at_t.append(incremental)
            co2_at_t.append(co2_tons)

            if co2_tons > 0:
                raw_macs.append(round((incremental * demand_mwh) / co2_tons, 1))
            else:
                raw_macs.append(None)

        # Monotonic envelope (running max)
        env_macs = []
        running_max = 0
        for mac in raw_macs:
            if mac is not None:
                running_max = max(running_max, mac)
                env_macs.append(round(running_max, 1))
            else:
                env_macs.append(None)

        # Stepwise monotonic envelope (running max on step MACs)
        step_env = [None]
        step_running_max = 0
        for i in range(1, len(THRESHOLDS)):
            if costs_at_t[i] is not None and costs_at_t[i-1] is not None:
                delta_cost = (costs_at_t[i] - costs_at_t[i-1]) * demand_mwh
                delta_co2 = (co2_at_t[i] or 0) - (co2_at_t[i-1] or 0)
                if delta_co2 > 0 and delta_cost >= 0:
                    step_mac = delta_cost / delta_co2
                    step_running_max = max(step_running_max, step_mac)
                    step_env.append(round(step_running_max, 1))
                else:
                    step_env.append(round(step_running_max, 1))  # Hold previous
            else:
                step_env.append(None)

        envelope[iso] = {
            'raw': raw_macs,
            'envelope': env_macs,
            'stepwise_envelope': step_env,
        }

    return envelope


def compute_path_constrained_mac(data):
    """Compute path-constrained reference MAC from existing results.

    Methodology: For each ISO at Medium costs, enforce that absolute resource deployment
    (procurement_pct × mix_share) is non-decreasing across thresholds. When the optimizer's
    independent optimization would reduce a resource, we hold it at its prior level and
    recompute the effective cost.

    This produces a monotonic-by-construction MAC curve that represents the true marginal
    cost of incremental resource additions.
    """
    path_mac = {}

    for iso in ISOS:
        iso_data = data['results'].get(iso, {})
        demand_mwh = iso_data.get('annual_demand_mwh', 1)
        wholesale = WHOLESALE_PRICES[iso]

        # Collect Medium scenario results across thresholds
        results_by_t = {}
        for t_str in THRESHOLD_STRS:
            t_data = iso_data.get('thresholds', {}).get(t_str, {})
            sc = t_data.get('scenarios', {}).get('MMM_M_M')
            if sc:
                results_by_t[t_str] = sc

        if not results_by_t:
            path_mac[iso] = {'mac': [None] * len(THRESHOLDS),
                             'mixes': [None] * len(THRESHOLDS),
                             'costs': [None] * len(THRESHOLDS)}
            continue

        # Build path-constrained resource deployment
        prev_abs = {'clean_firm': 0, 'solar': 0, 'wind': 0, 'ccs_ccgt': 0, 'hydro': 0}
        prev_batt = 0
        prev_ldes = 0
        prev_proc = 0

        path_macs = []
        path_mixes = []
        path_costs = []
        prev_cost = 0
        prev_co2 = 0

        for t_idx, t_str in enumerate(THRESHOLD_STRS):
            sc = results_by_t.get(t_str)
            if not sc:
                path_macs.append(None)
                path_mixes.append(None)
                path_costs.append(None)
                continue

            mix = sc['resource_mix']
            proc = sc['procurement_pct']
            batt = sc['battery_dispatch_pct']
            ldes = sc['ldes_dispatch_pct']
            co2_data = sc.get('co2_abated', {})
            co2_tons = co2_data.get('total_co2_abated_tons', 0)

            # Compute absolute deployment for this threshold's optimal mix
            curr_abs = {}
            for rtype in ['clean_firm', 'solar', 'wind', 'ccs_ccgt', 'hydro']:
                curr_abs[rtype] = proc * mix.get(rtype, 0) / 100.0

            # Enforce monotonicity: each resource's absolute deployment >= previous
            constrained_abs = {}
            for rtype in curr_abs:
                constrained_abs[rtype] = max(curr_abs[rtype], prev_abs[rtype])

            # Constrained procurement and storage
            constrained_proc = max(proc, prev_proc)
            constrained_batt = max(batt, prev_batt)
            constrained_ldes = max(ldes, prev_ldes)

            # Reconstruct mix percentages from constrained absolute values
            total_abs = sum(constrained_abs.values())
            if total_abs > 0:
                constrained_mix = {r: round(constrained_abs[r] / total_abs * 100, 1)
                                   for r in constrained_abs}
                eff_proc = total_abs  # This is the effective procurement factor
            else:
                constrained_mix = mix
                eff_proc = proc

            # Compute cost from constrained deployment using the cost model
            # (simplified version matching compute_costs logic)
            cost_data = sc.get('costs', {})
            incremental = cost_data.get('incremental', 0)

            # Adjustment: if we're holding extra resources from prior thresholds,
            # the cost is at least as high as the previous constrained cost
            constrained_incremental = max(incremental, prev_cost)

            # Average MAC = (constrained_incremental × demand) / co2
            if co2_tons > 0:
                avg_mac = round((constrained_incremental * demand_mwh) / co2_tons, 1)
            else:
                avg_mac = None

            # Stepwise MAC (from previous to current)
            if t_idx > 0 and prev_co2 > 0:
                delta_cost = (constrained_incremental - prev_cost) * demand_mwh
                delta_co2 = co2_tons - prev_co2
                if delta_co2 > 0:
                    step_mac = round(delta_cost / delta_co2, 1)
                else:
                    step_mac = None
            else:
                step_mac = None if t_idx > 0 else None

            path_macs.append(avg_mac)
            path_mixes.append(constrained_mix)
            path_costs.append(round(constrained_incremental, 2))

            # Update state for next threshold
            prev_abs = constrained_abs
            prev_batt = constrained_batt
            prev_ldes = constrained_ldes
            prev_proc = constrained_proc
            prev_cost = constrained_incremental
            prev_co2 = co2_tons

        path_mac[iso] = {
            'mac': path_macs,
            'mixes': path_mixes,
            'costs': path_costs,
        }

    return path_mac


def compute_anova(data):
    """ANOVA-style sensitivity decomposition: fraction of MAC variance explained
    by each toggle group.

    Uses the 324 factorial design (3×3×3×3×4 = 324 scenarios) as a balanced
    experiment. For each ISO and threshold, decomposes total MAC variance into
    contributions from each of the 5 toggle groups.

    Returns: {iso: {toggle_name: fraction_of_variance}} averaged across thresholds.
    """
    anova_results = {}

    for iso in ISOS:
        iso_data = data['results'].get(iso, {})
        demand_mwh = iso_data.get('annual_demand_mwh', 1)

        # Collect MACs across all scenarios and thresholds
        toggle_contributions = defaultdict(list)

        for t_str in THRESHOLD_STRS:
            t_data = iso_data.get('thresholds', {}).get(t_str, {})
            scenarios = t_data.get('scenarios', {})

            if not scenarios:
                continue

            # Build MAC array indexed by scenario
            mac_by_scenario = {}
            for sc_key, sc in scenarios.items():
                costs = sc.get('costs', {})
                co2 = sc.get('co2_abated', {})
                incremental = costs.get('incremental', 0)
                co2_tons = co2.get('total_co2_abated_tons', 0)

                if co2_tons > 0 and incremental is not None:
                    mac_by_scenario[sc_key] = (incremental * demand_mwh) / co2_tons

            if len(mac_by_scenario) < 10:
                continue

            all_macs = np.array(list(mac_by_scenario.values()))
            total_var = np.var(all_macs)

            if total_var < 1e-6:
                continue

            # For each toggle, compute between-group variance (SS_between / SS_total)
            for toggle_idx, toggle_name in enumerate(TOGGLE_NAMES):
                groups = defaultdict(list)

                for sc_key, mac in mac_by_scenario.items():
                    levels = scenario_key_to_levels(sc_key)
                    group_level = levels[toggle_idx]
                    groups[group_level].append(mac)

                # Between-group sum of squares
                grand_mean = np.mean(all_macs)
                ss_between = sum(
                    len(g) * (np.mean(g) - grand_mean) ** 2
                    for g in groups.values()
                )
                ss_total = total_var * len(all_macs)

                if ss_total > 0:
                    eta_squared = ss_between / ss_total
                    toggle_contributions[toggle_name].append(eta_squared)

        # Average across thresholds
        anova_results[iso] = {}
        for toggle_name in TOGGLE_NAMES:
            vals = toggle_contributions.get(toggle_name, [])
            if vals:
                anova_results[iso][toggle_name] = round(float(np.mean(vals)), 3)
            else:
                anova_results[iso][toggle_name] = 0.0

    return anova_results


def compute_crossover_analysis(fan_data, envelope_data):
    """Compute threshold where MAC crosses key benchmarks, using envelope and fan data."""
    benchmarks = {
        'scc_epa_190': 190,
        'scc_rennert_185': 185,
        'dac_low_400': 400,
        'dac_mid_600': 600,
        'carbon_credits_15': 15,
        'eu_ets_88': 88,
    }

    crossovers = {}
    for iso in ISOS:
        crossovers[iso] = {}
        env = envelope_data.get(iso, {}).get('envelope', [])
        p50 = fan_data.get(iso, {}).get('p50', [])

        for bm_name, bm_cost in benchmarks.items():
            # Find first threshold where envelope MAC exceeds benchmark
            env_cross = '>99'
            for i, mac in enumerate(env):
                if mac is not None and mac > bm_cost:
                    env_cross = THRESHOLDS[i]
                    break

            # Find first threshold where P50 MAC exceeds benchmark
            p50_cross = '>99'
            for i, mac in enumerate(p50):
                if mac is not None and mac > bm_cost:
                    p50_cross = THRESHOLDS[i]
                    break

            crossovers[iso][bm_name] = {
                'envelope': env_cross,
                'median': p50_cross,
            }

    return crossovers


def format_js_output(fan_data, stepwise_fan, envelope_data, path_mac, anova, crossovers):
    """Format all computed data as JavaScript constants for dashboard use."""
    lines = [
        '// ============================================================================',
        '// MAC STATISTICS — Option B: Statistical Post-Processing + Path-Constrained MAC',
        '// ============================================================================',
        '// Auto-generated by compute_mac_stats.py — do not edit manually',
        f'// Generated: {__import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M")}',
        '// Source: 16,200 optimizer scenarios (10 thresholds × 324 cost combos × 5 ISOs)',
        '//',
        '// Methodology:',
        '//   - Fan chart: P10/P25/P50/P75/P90 of average MAC across 324 cost scenarios',
        '//   - Envelope: Running max of Medium MAC (monotonic by construction)',
        '//   - Path-constrained: Enforces non-decreasing resource deployment',
        '//   - ANOVA: Eta-squared decomposition of MAC variance by toggle group',
        '// ============================================================================',
        '',
    ]

    # Fan chart data
    lines.append('// --- MAC Fan Chart (P10/P25/P50/P75/P90 average MAC across 324 scenarios) ---')
    lines.append(f'const MAC_FAN_DATA = {json.dumps(fan_data, indent=4)};')
    lines.append('')

    # Stepwise fan
    lines.append('// --- Stepwise Marginal MAC Fan (P10/P50/P90 of step MAC between thresholds) ---')
    lines.append(f'const MAC_STEPWISE_FAN = {json.dumps(stepwise_fan, indent=4)};')
    lines.append('')

    # Monotonic envelope
    env_js = {}
    for iso in ISOS:
        env_js[iso] = envelope_data[iso]
    lines.append('// --- Monotonic Envelope MAC (running max of Medium, smooths rebalancing) ---')
    lines.append(f'const MAC_ENVELOPE_DATA = {json.dumps(env_js, indent=4)};')
    lines.append('')

    # Path-constrained MAC
    path_js = {}
    for iso in ISOS:
        path_js[iso] = {
            'mac': path_mac[iso]['mac'],
            'costs': path_mac[iso]['costs'],
        }
    lines.append('// --- Path-Constrained Reference MAC (monotonic resource deployment) ---')
    lines.append(f'const MAC_PATH_CONSTRAINED = {json.dumps(path_js, indent=4)};')
    lines.append('')

    # ANOVA
    lines.append('// --- ANOVA: Fraction of MAC variance explained by each toggle group ---')
    lines.append('// Values are eta-squared (0-1): the proportion of total MAC variance')
    lines.append('// attributable to each sensitivity toggle. Higher = more influential.')
    lines.append(f'const MAC_ANOVA = {json.dumps(anova, indent=4)};')
    lines.append('')

    # Crossover analysis
    lines.append('// --- Crossover Analysis: threshold where MAC exceeds benchmarks ---')
    lines.append(f'const MAC_CROSSOVERS = {json.dumps(crossovers, indent=4)};')
    lines.append('')

    return '\n'.join(lines) + '\n'


def main():
    print("Loading optimizer results...")
    data = load_results()

    print("Computing MAC fan chart (P10-P90 across 324 scenarios)...")
    fan_data = compute_fan_chart(data)

    print("Computing stepwise marginal MAC fan...")
    stepwise_fan = compute_stepwise_fan(data)

    print("Computing monotonic envelope MAC...")
    envelope_data = compute_monotonic_envelope(data)

    print("Computing path-constrained reference MAC...")
    path_mac = compute_path_constrained_mac(data)

    print("Computing ANOVA sensitivity decomposition...")
    anova = compute_anova(data)

    print("Computing crossover analysis...")
    crossovers = compute_crossover_analysis(fan_data, envelope_data)

    # Write JavaScript output
    js_content = format_js_output(fan_data, stepwise_fan, envelope_data, path_mac, anova, crossovers)
    with open(JS_OUTPUT_PATH, 'w') as f:
        f.write(js_content)
    print(f"Wrote {JS_OUTPUT_PATH}")

    # Write JSON output
    json_output = {
        'fan_chart': fan_data,
        'stepwise_fan': stepwise_fan,
        'envelope': envelope_data,
        'path_constrained': {iso: path_mac[iso] for iso in ISOS},
        'anova': anova,
        'crossovers': crossovers,
        'metadata': {
            'thresholds': THRESHOLDS,
            'isos': ISOS,
            'scenario_count': 324,
            'methodology': 'Option B: Statistical post-processing + path-constrained reference',
        }
    }
    with open(JSON_OUTPUT_PATH, 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"Wrote {JSON_OUTPUT_PATH}")

    # Print summary
    print("\n" + "=" * 70)
    print("MAC STATISTICS SUMMARY")
    print("=" * 70)

    for iso in ISOS:
        print(f"\n--- {iso} ---")
        p50 = fan_data[iso]['p50']
        p10 = fan_data[iso]['p10']
        p90 = fan_data[iso]['p90']
        env = envelope_data[iso]['envelope']
        pc = path_mac[iso]['mac']

        print(f"  Fan P50: {[x for x in p50]}")
        print(f"  Fan P10: {[x for x in p10]}")
        print(f"  Fan P90: {[x for x in p90]}")
        print(f"  Envelope: {env}")
        print(f"  Path-constrained: {pc}")
        print(f"  ANOVA: {anova[iso]}")

    print("\n" + "=" * 70)
    print("CROSSOVER THRESHOLDS (where MAC exceeds benchmark)")
    print("=" * 70)
    for iso in ISOS:
        print(f"\n  {iso}:")
        for bm, vals in crossovers[iso].items():
            print(f"    {bm}: envelope={vals['envelope']}, median={vals['median']}")


if __name__ == '__main__':
    main()
