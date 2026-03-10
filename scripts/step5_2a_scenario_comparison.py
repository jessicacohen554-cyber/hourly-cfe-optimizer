#!/usr/bin/env python3
"""Scenario comparison: reads pre-computed A and B results, builds queues, trajectories, and comparison output.

Reads intermediate JSON files produced by the Scenario A and Scenario B scripts,
then performs:
  1. Consequential queue construction (dispatch-cache CO2 accounting)
  2. Stranding analysis
  3. MAC trajectories (Scenario A: step5d pass-through, Scenario B: incremental dispatch-based)
  4. Side-by-side comparison tables + domino sequence
  5. Final JSON + JS output for the dashboard

This is the third stage of the split scenario pipeline:
  scenario_common.py  ->  scenario A script  ->  scenario B script  ->  THIS FILE

Usage:
  python step7c_scenario_comparison.py                # All ISOs
  python step7c_scenario_comparison.py --iso CAISO     # Single ISO
  python step7c_scenario_comparison.py --iso CAISO ERCOT  # Multiple ISOs
"""

import json
import os
import sys
import functools
import argparse
import numpy as np
from pathlib import Path

# Ensure scripts/ is on the path
sys.path.insert(0, str(Path(__file__).parent))

from scenario_common import (
    ISOS, RESOURCES, THRESHOLDS, SCENARIO_A, SCENARIO_B,
    HYDRO_CAP_TWH, WHOLESALE_PRICES, FUEL_ADJUSTMENTS,
    GRID_MIX_SHARES, BASE_DEMAND_TWH, SBTI_YEAR_MAP,
    PEAK_DEMAND_MW, EXISTING_GAS_CAPACITY_MW, EXISTING_GAS_FOM_KW_YR,
    NEW_CCGT_COST_KW_YR, PEAK_CAPACITY_CREDITS, GAS_AVAILABILITY_FACTOR,
    RESOURCE_ADEQUACY_MARGIN, ZONES, DEMAND_GROWTH_RATES,
    get_demand_twh, get_demand_growth_factor,
    recompute_gas_backup,
    compute_fossil_retirement, compute_co2_from_dispatch,
    load_common_data, get_supply_profiles, get_demand_profile,
    _archetype_key, load_dispatch_cache, get_or_compute_dispatch,
    build_consequential_queue, compute_stranding,
    _get_dispatch_co2_for_mix,
    load_scenario_results, parse_iso_args,
)


# ============================================================================
# STEP5D QUEUE LOADER
# ============================================================================

def _load_step5d_queue(filename='consequential_queue_scenario_a.json',
                       mac_queue_dir=None):
    """Load pre-computed consequential queue from step5d output.

    Returns list of queue step dicts, or None if file not found.
    """
    if mac_queue_dir is None:
        project_root = Path(__file__).parent.parent
        mac_queue_dir = project_root / 'data' / 'step3-dispatch'
    queue_path = os.path.join(mac_queue_dir, filename)
    if not os.path.exists(queue_path):
        return None
    with open(queue_path) as f:
        data = json.load(f)
    return data.get('queue', [])


# ============================================================================
# TRAJECTORY BUILDER (step5d pass-through or incremental dispatch-based MAC)
# ============================================================================

def _build_trajectory(results, egrid, fossil_mix, demand_data, gen_profiles,
                      dispatch_caches, cfr_cached, _traj_co2_cache,
                      isos, enforce_monotonic=False, mac_source='step5d'):
    """Build trajectory dicts from optimization results.

    MAC sourcing:
      mac_source='step5d': Pass through mac_this_step from step5d results
        (authoritative for Scenario A — step5d computes MAC as incremental
        new-build cost / incremental CO2 displaced).
      mac_source='incremental': Compute incremental MAC from dispatch-based
        CO2 and new-build costs (for Scenario B — same formula as step5d:
        delta_new_build_cost / delta_CO2_displaced between thresholds).
    """
    results_id = id(results)
    traj = {}
    for iso in isos:
        iso_traj = []
        base_demand_twh = BASE_DEMAND_TWH[iso]

        # --- Collect raw data points ---
        raw_entries = []  # list of (threshold, d, demand_twh)
        for t in THRESHOLDS:
            d = results.get(iso, {}).get(t, {})
            if not d:
                continue
            demand_twh = get_demand_twh(iso, t)
            raw_entries.append((t, d, demand_twh))

        # --- Compute MAC based on source ---
        computed_mac = {}  # threshold -> MAC value

        if mac_source == 'step5d':
            # Direct pass-through from step5d's mac_this_step
            for t, d, _ in raw_entries:
                mac_val = d.get('mac_this_step')
                if mac_val is not None and mac_val > 0:
                    computed_mac[t] = round(mac_val, 1)

        elif mac_source == 'incremental':
            # Incremental MAC: delta_new_build_cost / delta_CO2_displaced
            # Same formula as step5d uses for Scenario A
            prev_co2 = 0
            prev_cost = 0
            for t, d, _ in raw_entries:
                # Get CO2 displaced at this threshold
                co2_data = _traj_co2_cache.get((iso, t, results_id))
                if co2_data:
                    co2_displaced = co2_data['total_co2_abated_tons']
                else:
                    # Fallback: estimate from emission rate x new generation
                    new_gen_twh = d.get('new_gen_twh', 0)
                    rate, _ = cfr_cached(iso, t, egrid, fossil_mix)
                    co2_displaced = new_gen_twh * 1e6 * rate if rate > 0 else 0

                # Cumulative new-build cost
                cum_cost = d.get('new_build_cost_total', 0)

                # Incremental MAC = delta_cost / delta_CO2
                delta_co2 = co2_displaced - prev_co2
                delta_cost = cum_cost - prev_cost
                if delta_co2 > 1000:  # minimum 1000 tons CO2 to avoid noise
                    mac_val = delta_cost / delta_co2
                    computed_mac[t] = round(min(max(mac_val, 0), 9999), 1)

                prev_co2 = co2_displaced
                prev_cost = cum_cost

        # --- Build trajectory entries ---
        for idx, (t, d, demand_twh) in enumerate(raw_entries):
            stepwise_mac = computed_mac.get(t, None)
            # First threshold has no MAC (no prior to diff against)
            if idx == 0:
                stepwise_mac = None

            cf_twh = d['resource_twh'].get('clean_firm', 0)
            ccs_twh = d['resource_twh'].get('ccs_ccgt', 0)
            existing_cf_twh = GRID_MIX_SHARES[iso].get('clean_firm', 0) / 100.0 * base_demand_twh
            firm_total_twh = max(0, cf_twh - existing_cf_twh) + ccs_twh
            # Annual cost ($B) = effective_cost ($/MWh) x demand (TWh) x 1e6 / 1e9
            annual_cost_billion = d['effective_cost'] * demand_twh / 1000.0
            # 25-year NPV ($B) at 5% real WACC
            # Annuity factor = (1 - (1+r)^-n) / r
            _r, _n = 0.05, 25
            annuity_factor = (1 - (1 + _r) ** -_n) / _r  # ~14.09
            npv_25yr_billion = annual_cost_billion * annuity_factor
            # Zone classification
            if t <= 65:
                zone = 'early'
            elif t <= 90:
                zone = 'inflection'
            else:
                zone = 'last_mile'

            entry = {
                'threshold': t,
                'year': SBTI_YEAR_MAP.get(t, 2050),
                'zone': zone,
                'demand_twh': round(demand_twh, 1),
                'effective_cost': d['effective_cost'],
                'total_cost': d['total_cost'],
                'incremental': d['incremental'],
                'annual_cost_billion': round(annual_cost_billion, 2),
                'npv_25yr_billion': round(npv_25yr_billion, 1),
                'resource_twh': d['resource_twh'],
                'battery_twh': d.get('battery_twh', 0) + d.get('battery8_twh', 0),
                'ldes_twh': d.get('ldes_twh', 0),
                'gas_backup_mw': d['gas_backup_mw'],
                'new_gas_mw': d['new_gas_mw'],
                'existing_gas_used_mw': d.get('existing_gas_used_mw', 0),
                'clean_peak_mw': d.get('clean_peak_mw', 0),
                'firm_total_twh': round(firm_total_twh, 1),
                'stepwise_mac': stepwise_mac,
                'blended_new_lcoe': d.get('blended_new_lcoe', 0),
                'new_gen_twh': d.get('new_gen_twh', 0),
                'new_build_cost_total': d.get('new_build_cost_total', 0),
            }

            # Four-pool metadata (from Scenario B redesign)
            pool_fields = [
                'pool1_sss_twh', 'pool2_contracted_twh', 'pool3_merchant_twh',
                'pool4_newbuild_twh', 'sss_hourly_min_coverage_pct',
                'sss_hourly_max_coverage_pct', 'available_hourly_avg_pct',
                'residual_gap_twh', 'firm_demand_hours',
                'storage_opportunity_hours', 'surplus_twh', 'gap_hours',
                'pool_adjusted_cost', 'sss_savings_per_mwh',
                'merchant_savings_per_mwh', 'eac_price',
                'learning_fraction',
            ]
            for field in pool_fields:
                if field in d:
                    entry[field] = d[field]

            iso_traj.append(entry)
        traj[iso] = iso_traj

    if enforce_monotonic:
        for iso in isos:
            if iso not in traj:
                continue
            running_max = {}
            for entry in traj[iso]:
                for res in list(entry['resource_twh'].keys()):
                    val = entry['resource_twh'][res]
                    if res in running_max:
                        entry['resource_twh'][res] = max(running_max[res], val)
                    running_max[res] = entry['resource_twh'][res]
                for field in ['battery_twh', 'ldes_twh']:
                    val = entry.get(field, 0)
                    old_max = running_max.get(field, 0)
                    entry[field] = max(old_max, val)
                    running_max[field] = entry[field]
                gf = get_demand_growth_factor(iso, entry['threshold'])
                gas_mw, new_gas_mw, _, clean_peak_mw = recompute_gas_backup(
                    iso, entry['resource_twh'],
                    entry.get('battery_twh', 0), entry.get('ldes_twh', 0), gf)
                entry['gas_backup_mw'] = gas_mw
                entry['new_gas_mw'] = new_gas_mw
                entry['clean_peak_mw'] = clean_peak_mw
                cf_twh = entry['resource_twh'].get('clean_firm', 0)
                ccs_twh = entry['resource_twh'].get('ccs_ccgt', 0)
                existing_cf_twh = GRID_MIX_SHARES[iso].get('clean_firm', 0) / 100.0 * BASE_DEMAND_TWH[iso]
                entry['firm_total_twh'] = round(max(0, cf_twh - existing_cf_twh) + ccs_twh, 1)
    return traj


# ============================================================================
# COMPARISON PRINTING
# ============================================================================

def print_comparison(results_a, results_b, queue_a, queue_b, stranding_a, stranding_b, isos):
    """Print scenario-by-scenario tables, side-by-side comparison, and domino sequence."""

    for scenario, results, queue, stranding, label in [
        (SCENARIO_A, results_a, queue_a, stranding_a, 'A'),
        (SCENARIO_B, results_b, queue_b, stranding_b, 'B'),
    ]:
        print(f"\n{'=' * 120}")
        print(f"SCENARIO {label}: {scenario['name'].upper()}")
        print(f"  {scenario['description']}")
        print(f"  Toggles: ren={scenario['toggles']['ren']}, firm={scenario['toggles']['firm']}, "
              f"batt={scenario['toggles']['batt']}, ldes={scenario['toggles']['ldes_lvl']}, "
              f"ccs={scenario['toggles']['ccs']}, fuel={scenario['toggles']['fuel']}, "
              f"tx={scenario['toggles']['tx']}")
        print(f"{'=' * 120}")

        # Queue
        print(f"\n{'#':>3} {'ISO':<7} {'Zone':<12} {'MAC $/t':>10} {'CO2 MT':>8} {'dCost':>8} "
              f"{'Gas End':>10} {'dGas':>10} {'Primary Resource':>40}")
        print("-" * 120)

        for step in queue:
            deltas = step['delta_resources']
            sorted_d = sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True)
            if sorted_d:
                top = sorted_d[0]
                top_str = f"{top[0]}: {'+' if top[1]>0 else ''}{top[1]:.0f} TWh"
                if len(sorted_d) > 1 and abs(sorted_d[1][1]) > 1:
                    r2 = sorted_d[1]
                    top_str += f", {r2[0]}: {'+' if r2[1]>0 else ''}{r2[1]:.0f}"
            else:
                top_str = "(no resource delta)"

            mac_str = f"${step['marginal_mac']:,.0f}" if step['marginal_mac'] < 9999 else "$inf"
            print(f"{step['queue_position']:>3} {step['iso']:<7} {step['zone_label']:<12} "
                  f"{mac_str:>10} {step['co2_displaced_mt']:>7.1f} "
                  f"${step['delta_cost_per_mwh']:>6.1f} "
                  f"{step['gas_backup_mw_end']:>9,} {step['delta_gas_mw']:>+9,} "
                  f"{top_str:>40}")

        total_co2 = sum(s['co2_displaced_mt'] for s in queue)
        total_cost_delta = sum(s['delta_cost_per_mwh'] * BASE_DEMAND_TWH[s['iso']] * 1e6
                               for s in queue) / 1e9
        print(f"\nTotal CO2 displaced: {total_co2:.1f} MT")
        print(f"Total cost delta: ${total_cost_delta:.1f}B")

        # Stranding summary
        print(f"\nStranding flags (peak/final ratio > 1.5):")
        for iso in isos:
            for res in ['wind', 'solar', 'clean_firm']:
                s = stranding.get(iso, {}).get(res, {})
                if s.get('stranding_ratio', 0) > 1.5 and s.get('peak_twh', 0) > 1:
                    print(f"  ! {iso} {res}: peak {s['peak_twh']:.0f} TWh @ {s['peak_threshold']}% -> "
                          f"final {s['final_twh']:.0f} TWh @ {s['final_threshold']}% "
                          f"(ratio: {s['stranding_ratio']:.1f}x)")

    # ========================================================================
    # SIDE-BY-SIDE COMPARISON
    # ========================================================================

    print(f"\n{'=' * 150}")
    print("SIDE-BY-SIDE: RESOURCE MIX + GAS + COST AT EACH THRESHOLD (demand grows per SBTI year)")
    print(f"{'=' * 150}")

    for iso in isos:
        print(f"\n{'_' * 130}")
        print(f"  {iso} (base demand: {BASE_DEMAND_TWH[iso]:.1f} TWh, growth: {DEMAND_GROWTH_RATES[iso]*100:.1f}%/yr)")
        print(f"{'_' * 130}")
        print(f"{'Thr':>5} {'Year':>5} {'Dem':>6} | {'CF_A':>6} {'Sol_A':>6} {'Wnd_A':>6} {'Gas_A':>8} {'$/MWh_A':>8} | "
              f"{'CF_B':>6} {'Sol_B':>6} {'Wnd_B':>6} {'Gas_B':>8} {'$/MWh_B':>8} | "
              f"{'dCost':>7} {'dGas':>8}")
        print("-" * 130)

        for t in THRESHOLDS:
            year = SBTI_YEAR_MAP.get(t, '?')
            demand_twh = get_demand_twh(iso, t)
            a = results_a.get(iso, {}).get(t, {})
            b = results_b.get(iso, {}).get(t, {})
            if not a or not b:
                continue

            a_rt = a['resource_twh']
            b_rt = b['resource_twh']

            delta_cost = b['effective_cost'] - a['effective_cost']
            delta_gas = b['gas_backup_mw'] - a['gas_backup_mw']

            print(f"{t:>5} {year:>5} {demand_twh:>5.0f} | "
                  f"{a_rt['clean_firm']:>6.0f} {a_rt['solar']:>6.0f} {a_rt['wind']:>6.0f} "
                  f"{a_rt.get('offshore_wind', 0):>6.0f} "
                  f"{a['gas_backup_mw']:>8,} {a['effective_cost']:>7.1f} | "
                  f"{b_rt['clean_firm']:>6.0f} {b_rt['solar']:>6.0f} {b_rt['wind']:>6.0f} "
                  f"{b_rt.get('offshore_wind', 0):>6.0f} "
                  f"{b['gas_backup_mw']:>8,} {b['effective_cost']:>7.1f} | "
                  f"{delta_cost:>+7.1f} {delta_gas:>+8,}")

    # ========================================================================
    # DOMINO SEQUENCE -- MAC ESCALATION
    # ========================================================================

    print(f"\n{'=' * 140}")
    print("DOMINO SEQUENCE: CUMULATIVE CO2 + GAS + COST AS YOU CLIMB THE MAC LADDER")
    print(f"{'=' * 140}")

    for scenario, queue, label in [(SCENARIO_A, queue_a, 'A'), (SCENARIO_B, queue_b, 'B')]:
        print(f"\nScenario {label}: {scenario['name']}")
        print(f"{'#':>3} {'ISO':<7} {'Zone':<12} {'MAC':>10} {'Cum CO2':>9} {'Cum $B':>8} "
              f"{'Gas(end)':>10} {'New Gas':>10} {'ISO Progress':>50}")
        print("-" * 130)

        cum_co2 = 0
        cum_cost = 0
        iso_progress = {iso: 50 for iso in isos}

        for step in queue:
            cum_co2 += step['co2_displaced_mt']
            cum_cost += step['delta_cost_per_mwh'] * BASE_DEMAND_TWH[step['iso']] * 1e6 / 1e9
            iso_progress[step['iso']] = step['threshold_end']

            prog_str = " | ".join(f"{iso}:{iso_progress[iso]:.0f}%" for iso in isos)
            mac_str = f"${step['marginal_mac']:,.0f}" if step['marginal_mac'] < 9999 else "$inf"

            print(f"{step['queue_position']:>3} {step['iso']:<7} {step['zone_label']:<12} "
                  f"{mac_str:>10} {cum_co2:>8.1f} {cum_cost:>+7.1f} "
                  f"{step['gas_backup_mw_end']:>10,} {step['new_gas_mw_end']:>10,} "
                  f"{prog_str:>50}")


# ============================================================================
# ZONE-BASED COMPARISON (EARLY / INFLECTION / LAST MILE)
# ============================================================================

def _build_zone_comparison(trajectories, isos):
    """Build per-zone comparison deltas between Scenario A and B.

    Groups thresholds into three zones:
      early (50-65%): Strategies should be similar
      inflection (70-90%): Hourly matching diverges — firm/storage signal
      last_mile (92.5-≥99.99%): Maximum divergence — learning curve payoff
    """
    traj_a = trajectories.get('pure_consequential', {})
    traj_b = trajectories.get('hourly_matching', {})

    comparison = {}
    for iso in isos:
        entries_a = {e['threshold']: e for e in traj_a.get(iso, [])}
        entries_b = {e['threshold']: e for e in traj_b.get(iso, [])}

        zone_data = {}
        for zone, t_range in [('early', (50, 65)), ('inflection', (70, 90)), ('last_mile', (92.5, 100))]:
            zone_thresholds = [t for t in sorted(set(entries_a.keys()) & set(entries_b.keys()))
                               if t_range[0] <= t <= t_range[1]]
            if not zone_thresholds:
                continue

            # Representative threshold for each zone (midpoint or highest)
            rep_t = zone_thresholds[-1]  # Highest in zone
            a = entries_a[rep_t]
            b = entries_b[rep_t]

            a_rt = a.get('resource_twh', {})
            b_rt = b.get('resource_twh', {})
            a_firm = a_rt.get('clean_firm', 0) + a_rt.get('ccs_ccgt', 0)
            b_firm = b_rt.get('clean_firm', 0) + b_rt.get('ccs_ccgt', 0)

            delta = {
                'representative_threshold': rep_t,
                'thresholds_in_zone': zone_thresholds,
                'cost_a': a.get('effective_cost', 0),
                'cost_b': b.get('effective_cost', 0),
                'cost_gap_pct': round(
                    (b.get('effective_cost', 0) - a.get('effective_cost', 0))
                    / max(a.get('effective_cost', 1), 1) * 100, 1),
                'firm_a_twh': round(a_firm, 1),
                'firm_b_twh': round(b_firm, 1),
                'firm_deployment_gap_twh': round(b_firm - a_firm, 1),
                'storage_a_twh': round(a.get('battery_twh', 0) + a.get('ldes_twh', 0), 1),
                'storage_b_twh': round(b.get('battery_twh', 0) + b.get('ldes_twh', 0), 1),
                'gas_a_mw': a.get('gas_backup_mw', 0),
                'gas_b_mw': b.get('gas_backup_mw', 0),
            }

            # Pool metadata from Scenario B (if available)
            for field in ['pool1_sss_twh', 'pool3_merchant_twh', 'pool4_newbuild_twh',
                          'residual_gap_twh', 'firm_demand_hours', 'available_hourly_avg_pct',
                          'learning_fraction']:
                if field in b:
                    delta[f'b_{field}'] = b[field]

            # Learning position labels
            lf_a = a.get('learning_fraction', 0)
            lf_b = b.get('learning_fraction', 0)
            if lf_a < 0.1:
                delta['learning_position_a'] = 'FOAK'
            elif lf_a >= 0.9:
                delta['learning_position_a'] = 'NOAK'
            else:
                delta['learning_position_a'] = f'NOAK_{lf_a:.1f}'
            if lf_b < 0.1:
                delta['learning_position_b'] = 'FOAK'
            elif lf_b >= 0.9:
                delta['learning_position_b'] = 'NOAK'
            else:
                delta['learning_position_b'] = f'NOAK_{lf_b:.1f}'

            zone_data[zone] = delta

        comparison[iso] = zone_data

    return comparison


# ============================================================================
# OUTPUT WRITING
# ============================================================================

def write_output(trajectories, queue_a, queue_b,
                 stranding_a, stranding_b, isos):
    """Write JSON + JS output files for the dashboard."""

    # Build zone-based comparison deltas
    zone_comparison = _build_zone_comparison(trajectories, isos)

    output = {
        'zone_comparison': zone_comparison,
        'metadata': {
            'description': 'Dual-scenario comparison: Pure Consequential vs Hourly Matching',
            'scenario_a': {
                'name': SCENARIO_A['name'],
                'description': SCENARIO_A['description'],
                'toggles': SCENARIO_A['toggles'],
                'method': 'ef_forward_step_foak',
                'method_description': (
                    'Pure consequential: greedy forward-stepping through feasible EF mixes. '
                    'At each threshold, evaluates all EF mixes under static cost assumptions '
                    '(Low renewables, Med batteries/uprates/tx, High firm/CCS/LDES/geo). '
                    'With expensive firm, optimizer picks renewable-heavy mixes. At 90%+, '
                    'firm is unavoidable at FOAK prices -- cost cliff. Floor ratchet locks in '
                    'prior deployments. Each step is deterministic of the next.'
                ),
            },
            'scenario_b': {
                'name': SCENARIO_B['name'],
                'description': SCENARIO_B['description'],
                'toggles': SCENARIO_B['toggles'],
                'method': SCENARIO_B.get('method', 'four_pool_hourly_matching'),
                'method_description': (
                    'GHG Protocol hourly Scope 2 incentive-driven grid buildout. '
                    'Four supply pools: (1) SSS at $0 — policy-supported nuclear ZECs, '
                    'public hydro, RPS mandates; (2) Contracted — locked via hyperscaler '
                    'PPAs (excluded); (3) Existing merchant — solar/wind/merchant nuclear '
                    'at EAC premium ($3-5/MWh); (4) New-build — investment signal from '
                    'hourly matching gap. Hourly constraint forces firm clean + storage '
                    'for night/winter coverage. Learning curve 2030-2040, FOAK(H)->NOAK(L).'
                ),
                'pool_model': {
                    'pool1_description': 'SSS: policy-supported clean (nuclear ZECs, public hydro, RPS)',
                    'pool2_description': 'Contracted: locked via hyperscaler PPAs (unavailable)',
                    'pool3_description': 'Existing merchant: solar/wind/merchant nuclear at EAC premium',
                    'pool4_description': 'New-build: hourly matching investment signal',
                },
            },
            'sbti_year_map': {str(k): v for k, v in SBTI_YEAR_MAP.items()},
            'thresholds': THRESHOLDS,
            'isos': isos,
            'grid_mix_shares': {iso: dict(GRID_MIX_SHARES[iso]) for iso in isos},
            'base_demand_twh': {iso: BASE_DEMAND_TWH[iso] for iso in isos},
            'demand_growth_rates': {iso: DEMAND_GROWTH_RATES[iso] for iso in isos},
            'demand_growth_level': 'Medium',
        },
        'queue_a': queue_a,
        'queue_b': queue_b,
        'trajectories': trajectories,
        'stranding_a': stranding_a,
        'stranding_b': stranding_b,
    }

    out_json = 'data/step5-scenarios/scenario_comparison.json'
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nJSON: {out_json} ({os.path.getsize(out_json) / 1024:.0f} KB)")

    out_js = 'dashboard/js/scenario-comparison-data.js'
    os.makedirs(os.path.dirname(out_js), exist_ok=True)
    with open(out_js, 'w') as f:
        f.write("// Auto-generated by step7c_scenario_comparison.py\n")
        f.write("// Dual-scenario comparison: Pure Consequential vs Hourly Matching\n")
        f.write("// A: Pure Consequential  B: Hourly Matching\n\n")
        f.write(f"const SCENARIO_COMPARISON = {json.dumps(output, indent=2, default=str)};\n")
    print(f"JS:   {out_js} ({os.path.getsize(out_js) / 1024:.0f} KB)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    # ------------------------------------------------------------------
    # Parse --iso flag
    # ------------------------------------------------------------------
    requested_isos, _ = parse_iso_args()

    print("=" * 80)
    print("SCENARIO COMPARISON: PURE CONSEQUENTIAL vs HOURLY MATCHING")
    print(f"  ISOs: {', '.join(requested_isos)}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Load pre-computed scenario results from intermediate JSON files
    # ------------------------------------------------------------------
    print("\nLoading Scenario A results...")
    # Try mac_queue subdirectory first (step5d output), fall back to step5 root
    mac_queue_dir = 'data/step3-dispatch'
    results_a, _meta_a = load_scenario_results('A', isos=requested_isos, out_dir=mac_queue_dir)
    if results_a is None:
        # Fallback: legacy location in step5 root
        results_a, _meta_a = load_scenario_results('A', isos=requested_isos)
    if results_a is None:
        print("ERROR: Scenario A results not found.")
        print("  Run step5d_deployment_queue.py first to produce intermediate results.")
        print("  Expected: data/step3-dispatch/scenario_a_{ISO}.json per ISO")
        sys.exit(1)

    print(f"  Loaded A: {sorted(results_a.keys())}")

    print("Loading Scenario B results...")
    results_b, _meta_b = load_scenario_results('B', isos=requested_isos)
    if results_b is None:
        print("ERROR: Scenario B results not found.")
        print("  Run the Scenario B script first to produce intermediate results.")
        print("  Expected: data/step5-scenarios/scenario_b_{ISO}.json per ISO")
        sys.exit(1)

    print(f"  Loaded B: {sorted(results_b.keys())}")

    # ------------------------------------------------------------------
    # Determine which ISOs to compare (intersection of A, B, and requested)
    # ------------------------------------------------------------------
    isos_in_a = set(results_a.keys())
    isos_in_b = set(results_b.keys())
    isos_both = isos_in_a & isos_in_b
    isos = [iso for iso in requested_isos if iso in isos_both]

    if not isos:
        print(f"ERROR: No overlapping ISOs between A ({sorted(isos_in_a)}) "
              f"and B ({sorted(isos_in_b)}) for requested ISOs ({requested_isos}).")
        sys.exit(1)

    missing_a = set(requested_isos) - isos_in_a
    missing_b = set(requested_isos) - isos_in_b
    if missing_a:
        print(f"  NOTE: ISOs {sorted(missing_a)} not in Scenario A results -- skipping.")
    if missing_b:
        print(f"  NOTE: ISOs {sorted(missing_b)} not in Scenario B results -- skipping.")
    print(f"  Comparing ISOs: {', '.join(isos)}")

    # Filter results to only the ISOs we are comparing
    results_a = {iso: results_a[iso] for iso in isos if iso in results_a}
    results_b = {iso: results_b[iso] for iso in isos if iso in results_b}

    # ------------------------------------------------------------------
    # Load egrid, fossil_mix, demand/gen profiles, dispatch caches
    # ------------------------------------------------------------------
    print("\nLoading egrid emission rates...")
    egrid_path = 'data/egrid_emission_rates.json'
    with open(egrid_path) as f:
        egrid = json.load(f)

    print("Loading fossil mix data...")
    from eia_data_io import load_fossil_mix
    fossil_mix = load_fossil_mix()

    print("Loading demand and generation profiles for dispatch-based CO2...")
    demand_data, gen_profiles, _, _ = load_common_data()

    dispatch_caches = {}
    print("Loading dispatch caches...")
    for iso in isos:
        cache = load_dispatch_cache(iso)
        if cache:
            dispatch_caches[iso] = cache
            print(f"  {iso}: {len(cache)} cached mixes")
        else:
            print(f"  {iso}: no cache (will compute live)")

    # ------------------------------------------------------------------
    # Memoized compute_fossil_retirement wrapper
    # ------------------------------------------------------------------
    _cfr_cache = {}

    def cfr_cached(iso, threshold, er, fm, demand_growth_factor=1.0):
        key = (iso, threshold, demand_growth_factor)
        if key not in _cfr_cache:
            _cfr_cache[key] = compute_fossil_retirement(iso, threshold, er, fm, demand_growth_factor)
        return _cfr_cache[key]

    # ------------------------------------------------------------------
    # Build consequential queues (dispatch-cache emission accounting)
    # ------------------------------------------------------------------
    print("\nLoading consequential queue A from step5d...")
    queue_a = _load_step5d_queue('consequential_queue_scenario_a.json')
    if queue_a is not None:
        # Filter to requested ISOs and re-number
        queue_a = [step for step in queue_a if step['iso'] in isos]
        for i, step in enumerate(queue_a):
            step['queue_position'] = i + 1
        print(f"  Loaded {len(queue_a)} queue steps from step5d (high_firm_low_vre)")
    else:
        print("  WARNING: consequential_queue_scenario_a.json not found, computing from scratch...")
        queue_a = build_consequential_queue(results_a, egrid, fossil_mix, cfr_fn=cfr_cached,
                                             demand_data=demand_data, gen_profiles=gen_profiles,
                                             dispatch_caches=dispatch_caches)
    queue_b = build_consequential_queue(results_b, egrid, fossil_mix, cfr_fn=cfr_cached,
                                         demand_data=demand_data, gen_profiles=gen_profiles,
                                         dispatch_caches=dispatch_caches)

    # ------------------------------------------------------------------
    # Stranding analysis
    # ------------------------------------------------------------------
    print("Computing stranding analysis...")
    stranding_a = compute_stranding(results_a)
    stranding_b = compute_stranding(results_b)

    # ------------------------------------------------------------------
    # Pre-compute dispatch CO2 for trajectory MAC smoothing
    # ------------------------------------------------------------------
    print("Pre-computing dispatch CO2 for trajectory MAC...")
    _traj_co2_cache = {}
    for scenario_results in [results_a, results_b]:
        for iso in isos:
            for t in THRESHOLDS:
                d = scenario_results.get(iso, {}).get(t, {})
                if d and (iso, t, id(scenario_results)) not in _traj_co2_cache:
                    co2 = _get_dispatch_co2_for_mix(iso, d, egrid,
                                                     demand_data, gen_profiles, dispatch_caches)
                    if co2:
                        _traj_co2_cache[(iso, t, id(scenario_results))] = co2

    # ------------------------------------------------------------------
    # Build resource trajectories with PCHIP-smoothed marginal MAC
    # ------------------------------------------------------------------
    print("Building trajectories...")
    trajectories = {}
    trajectories['pure_consequential'] = _build_trajectory(
        results_a, egrid, fossil_mix, demand_data, gen_profiles,
        dispatch_caches, cfr_cached, _traj_co2_cache, isos,
        enforce_monotonic=False, mac_source='step5d')
    trajectories['hourly_matching'] = _build_trajectory(
        results_b, egrid, fossil_mix, demand_data, gen_profiles,
        dispatch_caches, cfr_cached, _traj_co2_cache, isos,
        enforce_monotonic=False, mac_source='incremental')

    # ------------------------------------------------------------------
    # Print comparison tables
    # ------------------------------------------------------------------
    print_comparison(results_a, results_b, queue_a, queue_b,
                     stranding_a, stranding_b, isos)

    # ------------------------------------------------------------------
    # Write output files
    # ------------------------------------------------------------------
    write_output(trajectories, queue_a, queue_b,
                 stranding_a, stranding_b, isos)

    print("\nDone.")


if __name__ == '__main__':
    main()
