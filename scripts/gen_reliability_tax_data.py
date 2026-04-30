#!/usr/bin/env python3
"""
gen_reliability_tax_data.py
Generates dashboard/js/reliability-tax-data.js from step 2.3 adaptive results.

Reads all data/step2.3-adaptive/*.json, deduplicates beams, computes
per-ISO/pathway/scenario metrics, and emits a single JS variable (RT_DATA)
for direct <script> inclusion in reliability_tax.html.

Usage:
    python scripts/gen_reliability_tax_data.py
"""
import json
import glob
import os
import sys
from collections import defaultdict
from statistics import median

# ——————————————————————————————————————————————
# Gas cost back-derivation (replicates optimizer's gas_annual_cost)
# ——————————————————————————————————————————————

NEW_CCGT_COST_KW_YR = {
    'CAISO': 112, 'ERCOT': 89, 'PJM': 99, 'NYISO': 114,
    'NEISO': 105, 'MISO': 95, 'SPP': 88
}
NEW_CCGT_FOM_KW_YR = {
    'CAISO': 25, 'ERCOT': 22, 'PJM': 22, 'NYISO': 25,
    'NEISO': 24, 'MISO': 22, 'SPP': 22
}
WHOLESALE_PRICES = {
    'CAISO': 30, 'ERCOT': 27, 'PJM': 34, 'NYISO': 42,
    'NEISO': 41, 'MISO': 30, 'SPP': 25
}
GAS_CF = 0.45
H = 8760

ISOS = ['CAISO', 'ERCOT', 'MISO', 'NEISO', 'NYISO', 'PJM', 'SPP']
PATHWAYS = ['A', 'B', 'C', 'D']
KEY_THRESHOLDS = [50, 70, 90, 95, 99, 99.9]

# Resource field → JS key mapping for stacked area
RESOURCE_MAP = {
    'clean_firm': 'cleanFirm', 'solar': 'solar', 'wind': 'wind',
    'hydro': 'hydro', 'offshore_wind': 'offshoreWind', 'ccs_ccgt': 'ccs',
    'geothermal': 'geothermal', 'solar_batt4': 'solarBatt4',
    'solar_batt8': 'solarBatt8', 'wind_batt4': 'windBatt4',
    'wind_batt8': 'windBatt8'
}

# Canonical stack order (bottom → top) from chart-colors.js
STACK_ORDER = [
    'cleanFirm', 'geothermal', 'ccs', 'hydro', 'offshoreWind',
    'wind', 'solar', 'windBatt4', 'windBatt8', 'solarBatt4', 'solarBatt8'
]


def gas_annual_cost(gas_mw, iso):
    """Annual cost of gas fleet: capex + FOM + fuel."""
    if gas_mw <= 0:
        return 0.0
    capex = NEW_CCGT_COST_KW_YR[iso] * gas_mw * 1000
    fom = NEW_CCGT_FOM_KW_YR[iso] * gas_mw * 1000
    fuel = gas_mw * GAS_CF * H * WHOLESALE_PRICES[iso]
    return capex + fom + fuel


def _cost_at(run, threshold):
    """Get cumulative cost at a threshold, or None."""
    for s in run['threshold_snapshots']:
        if abs(s['threshold_pct'] - threshold) < 0.01:
            return s['cumulative_cost_usd']
    return None


def _snap_at(run, threshold):
    """Get snapshot dict at a threshold, or None."""
    for s in run['threshold_snapshots']:
        if abs(s['threshold_pct'] - threshold) < 0.01:
            return s
    return None


def _find_run(best, iso, pathway, cost_mode=1, archetype=None,
              scenario='base', dg='Medium'):
    """Find a run, trying archetype fallbacks."""
    archetypes = ([archetype] if archetype
                  else ['balanced', 'nuclear-heavy', 'solar-led', 'wind-led'])
    for arch in archetypes:
        key = (iso, pathway, cost_mode, arch, scenario, dg)
        if key in best:
            return best[key]
    return None


def load_all_runs(data_dir):
    """Load all JSONs, deduplicate beams, return dict keyed by config tuple."""
    files = glob.glob(os.path.join(data_dir, '*.json'))
    print(f"[gen] Found {len(files)} JSON files")

    best = {}
    for f in files:
        with open(f) as fh:
            try:
                d = json.load(fh)
            except json.JSONDecodeError:
                print(f"[gen] WARNING: skipping malformed {f}",
                      file=sys.stderr)
                continue

        key = (
            d['iso'], d['pathway'],
            d.get('cost_mode', 1),
            d['beam_archetype'],
            d['scenario_name'],
            d['config']['demand_growth']
        )

        # Keep best beam: highest thresholds achieved, then lowest cost at 99.9
        if key in best:
            old = best[key]
            if d['n_thresholds_achieved'] > old['n_thresholds_achieved']:
                best[key] = d
            elif d['n_thresholds_achieved'] == old['n_thresholds_achieved']:
                new_c = _cost_at(d, 99.9)
                old_c = _cost_at(old, 99.9)
                if new_c is not None and (old_c is None or new_c < old_c):
                    best[key] = d
        else:
            best[key] = d

    print(f"[gen] Deduplicated to {len(best)} unique configs")
    return best


def build_hero_stats(best):
    """Per-ISO summary for hero tiles."""
    hero = {}
    for iso in ISOS:
        # Pathway A & B median cost at 99.9 across scenarios/demand
        # (mode1, balanced)
        costs_a, costs_b = [], []
        peak_gas_a, peak_gas_b = None, None

        for scen in ['base', 'firm_high_vre_low', 'firm_low_vre_high']:
            for dg in ['Low', 'Medium', 'High']:
                ra = _find_run(best, iso, 'A', 1, None, scen, dg)
                rb = _find_run(best, iso, 'B', 1, None, scen, dg)
                if ra:
                    c = _cost_at(ra, 99.9)
                    if c is not None:
                        costs_a.append(c)
                if rb:
                    c = _cost_at(rb, 99.9)
                    if c is not None:
                        costs_b.append(c)

        # Base/Medium for peak gas
        ra_base = _find_run(best, iso, 'A', 1, None, 'base', 'Medium')
        rb_base = _find_run(best, iso, 'B', 1, None, 'base', 'Medium')
        if ra_base:
            peak_gas_a = ra_base['peak_gas_mw']
        if rb_base:
            peak_gas_b = rb_base['peak_gas_mw']

        hero[iso] = {
            'pathway_a_median': (round(median(costs_a) / 1e9, 1)
                                 if costs_a else None),
            'pathway_b_median': (round(median(costs_b) / 1e9, 1)
                                 if costs_b else None),
            'cost_ratio': (round(median(costs_b) / median(costs_a), 2)
                           if costs_a and costs_b else None),
            'peak_gas_a_gw': (round(peak_gas_a / 1000, 1)
                              if peak_gas_a else None),
            'peak_gas_b_gw': (round(peak_gas_b / 1000, 1)
                              if peak_gas_b else None),
            'peak_gas_delta_gw': (round((peak_gas_a - peak_gas_b) / 1000, 1)
                                  if peak_gas_a and peak_gas_b else None),
            'min_cost_a': (round(min(costs_a) / 1e9, 1)
                           if costs_a else None),
            'max_cost_a': (round(max(costs_a) / 1e9, 1)
                           if costs_a else None),
            'min_cost_b': (round(min(costs_b) / 1e9, 1)
                           if costs_b else None),
            'max_cost_b': (round(max(costs_b) / 1e9, 1)
                           if costs_b else None),
        }

    return hero


def build_gas_hump(best):
    """Per ISO × pathway × scenario × demand: gas trajectory over
    thresholds."""
    hump = {}
    for iso in ISOS:
        for pw in PATHWAYS:
            for scen in ['base', 'firm_high_vre_low', 'firm_low_vre_high']:
                for dg in ['Low', 'Medium', 'High']:
                    run = _find_run(best, iso, pw, 1, None, scen, dg)
                    key = f"{iso}_{pw}_{scen}_{dg}"
                    if not run:
                        continue
                    points = []
                    for s in run['threshold_snapshots']:
                        if s['threshold_pct'] in KEY_THRESHOLDS:
                            points.append({
                                't': s['threshold_pct'],
                                'year': s['year_achieved'],
                                'gas_gw': round(
                                    s['gas_need_mw'] / 1000, 2),
                                'peak_gw': round(
                                    s['peak_gas_mw'] / 1000, 2),
                                'stranded_gw': round(
                                    s['stranded_vs_peak_mw'] / 1000, 2),
                            })
                    if points:
                        hump[key] = {
                            'points': points,
                            'peak_gas_gw': round(
                                run['peak_gas_mw'] / 1000, 1),
                            'peak_gas_year': run['peak_gas_year'],
                        }
    return hump


def build_cost_ribbons(best):
    """Per ISO × pathway: min/median/max cumulative cost at each
    threshold."""
    ribbons = {}
    for iso in ISOS:
        for pw in PATHWAYS:
            # Collect costs across 9 scenario×demand combos
            # (mode1, balanced)
            costs_by_threshold = defaultdict(list)
            for scen in ['base', 'firm_high_vre_low', 'firm_low_vre_high']:
                for dg in ['Low', 'Medium', 'High']:
                    run = _find_run(best, iso, pw, 1, None, scen, dg)
                    if not run:
                        continue
                    for s in run['threshold_snapshots']:
                        if s['threshold_pct'] in KEY_THRESHOLDS:
                            costs_by_threshold[s['threshold_pct']].append(
                                s['cumulative_cost_usd'])

            if not costs_by_threshold:
                continue

            points = []
            for t in KEY_THRESHOLDS:
                vals = costs_by_threshold.get(t, [])
                if not vals:
                    continue
                sv = sorted(vals)
                points.append({
                    't': t,
                    'min': round(sv[0] / 1e9, 1),
                    'median': round(sv[len(sv) // 2] / 1e9, 1),
                    'max': round(sv[-1] / 1e9, 1),
                })

            # Gas cost share for base/Medium case
            run_base = _find_run(best, iso, pw, 1, None, 'base', 'Medium')
            gas_share = []
            if run_base:
                for s in run_base['threshold_snapshots']:
                    if s['threshold_pct'] in KEY_THRESHOLDS:
                        gc = gas_annual_cost(s['gas_need_mw'], iso)
                        total = s['total_annual_cost_usd']
                        share = (round(gc / total * 100, 1)
                                 if total > 0 else 0)
                        gas_share.append({
                            't': s['threshold_pct'],
                            'gas_cost_bn': round(gc / 1e9, 1),
                            'gas_pct': share,
                        })

            ribbons[f"{iso}_{pw}"] = {
                'points': points,
                'gas_share': gas_share,
            }

    return ribbons


def build_resource_mix(best):
    """Per ISO × pathway: normalized resource shares at key thresholds."""
    mixes = {}
    for iso in ISOS:
        for pw in PATHWAYS:
            run = _find_run(best, iso, pw, 1, None, 'base', 'Medium')
            if not run:
                continue

            columns = []
            for s in run['threshold_snapshots']:
                if s['threshold_pct'] not in KEY_THRESHOLDS:
                    continue
                mix = s['resource_mix_pct']
                # Map to JS keys and filter tiny values
                mapped = {}
                total = 0
                for data_key, js_key in RESOURCE_MAP.items():
                    v = max(0, mix.get(data_key, 0))
                    if v > 0.3:
                        mapped[js_key] = v
                        total += v
                # Normalize to 100%
                if total > 0:
                    normalized = {k: round(v / total * 100, 2)
                                  for k, v in mapped.items()}
                else:
                    normalized = {}
                columns.append({
                    't': s['threshold_pct'],
                    'mix': normalized,
                    'raw_total_pct': round(total, 1),
                })
            if columns:
                mixes[f"{iso}_{pw}"] = columns

    return mixes


def build_mode_comparison(best):
    """Per ISO × pathway: mode1 vs mode2 at 99.9%."""
    comp = {}
    for iso in ISOS:
        for pw in ['A', 'B', 'C', 'D']:
            r1 = _find_run(best, iso, pw, 1, None, 'base', 'Medium')
            r2 = _find_run(best, iso, pw, 2, None, 'base', 'Medium')
            if not r1 or not r2:
                continue
            c1 = _cost_at(r1, 99.9)
            c2 = _cost_at(r2, 99.9)
            if c1 is None or c2 is None:
                continue
            comp[f"{iso}_{pw}"] = {
                'mode1_cost_bn': round(c1 / 1e9, 1),
                'mode2_cost_bn': round(c2 / 1e9, 1),
                'delta_pct': round((c2 - c1) / c1 * 100, 1),
                'mode1_peak_gas_gw': round(r1['peak_gas_mw'] / 1000, 1),
                'mode2_peak_gas_gw': round(r2['peak_gas_mw'] / 1000, 1),
            }
    return comp


def build_reliability_tax(best):
    """Per ISO × pathway: cost ratio (95→99.9) / (at 90)."""
    tax = {}
    for iso in ISOS:
        for pw in PATHWAYS:
            run = _find_run(best, iso, pw, 1, None, 'base', 'Medium')
            if not run:
                continue
            c90 = _cost_at(run, 90)
            c95 = _cost_at(run, 95)
            c999 = _cost_at(run, 99.9)
            if c90 and c95 and c999 and c90 > 0:
                tax[f"{iso}_{pw}"] = {
                    'cost_at_90_bn': round(c90 / 1e9, 1),
                    'cost_at_95_bn': round(c95 / 1e9, 1),
                    'cost_at_999_bn': round(c999 / 1e9, 1),
                    'tax_ratio': round((c999 - c95) / c90, 1),
                }
    return tax


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(repo_root, 'data', 'step2.3-adaptive')
    out_path = os.path.join(repo_root, 'dashboard', 'js',
                            'reliability-tax-data.js')

    if not os.path.isdir(data_dir):
        print(f"[gen] ERROR: data dir not found: {data_dir}",
              file=sys.stderr)
        sys.exit(1)

    best = load_all_runs(data_dir)

    # Count available pathways
    pw_counts = defaultdict(int)
    for key in best:
        pw_counts[key[1]] += 1
    for pw in PATHWAYS:
        print(f"[gen] Pathway {pw}: {pw_counts.get(pw, 0)} configs")

    payload = {
        'meta': {
            'total_files': len(
                glob.glob(os.path.join(data_dir, '*.json'))),
            'unique_configs': len(best),
            'pathways_present': [pw for pw in PATHWAYS
                                 if pw_counts.get(pw, 0) > 0],
            'isos': ISOS,
            'stack_order': STACK_ORDER,
        },
        'hero_stats': build_hero_stats(best),
        'gas_hump': build_gas_hump(best),
        'cost_ribbons': build_cost_ribbons(best),
        'resource_mix': build_resource_mix(best),
        'mode_comparison': build_mode_comparison(best),
        'reliability_tax': build_reliability_tax(best),
    }

    # Print summary
    print(f"\n[gen] === Hero Stats ===")
    for iso in ISOS:
        h = payload['hero_stats'].get(iso, {})
        print(f"  {iso}: "
              f"A=${h.get('pathway_a_median', '-')}B  "
              f"B=${h.get('pathway_b_median', '-')}B  "
              f"ratio={h.get('cost_ratio', '-')}  "
              f"gas_delta={h.get('peak_gas_delta_gw', '-')}GW")

    print(f"\n[gen] === Reliability Tax (A) ===")
    for iso in ISOS:
        t = payload['reliability_tax'].get(f"{iso}_A", {})
        print(f"  {iso}: "
              f"90%=${t.get('cost_at_90_bn', '-')}B  "
              f"95%=${t.get('cost_at_95_bn', '-')}B  "
              f"99.9%=${t.get('cost_at_999_bn', '-')}B  "
              f"tax={t.get('tax_ratio', '-')}x")

    print(f"\n[gen] === Gas Hump keys: "
          f"{len(payload['gas_hump'])} ===")
    print(f"[gen] === Cost Ribbon keys: "
          f"{len(payload['cost_ribbons'])} ===")
    print(f"[gen] === Resource Mix keys: "
          f"{len(payload['resource_mix'])} ===")
    print(f"[gen] === Mode Comparison keys: "
          f"{len(payload['mode_comparison'])} ===")

    # Write JS
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    js_content = (
        "// Auto-generated by gen_reliability_tax_data.py"
        " — do not edit\n"
        f"var RT_DATA = {json.dumps(payload, separators=(',', ':'))};\n"
    )
    with open(out_path, 'w') as f:
        f.write(js_content)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"\n[gen] Wrote {out_path} ({size_kb:.0f} KB)")


if __name__ == '__main__':
    main()
