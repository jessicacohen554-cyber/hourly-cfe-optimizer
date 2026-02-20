#!/usr/bin/env python3
"""
Analyze track results: cost envelopes, mix differentials, replacement premium.

Reads:
  - dashboard/track_results.json (newbuild + replace tracks)
  - dashboard/overprocure_results.json (baseline)

Outputs:
  - Console report with P10/P50/P90 cost envelopes per track
  - Resource mix differentials (newbuild vs baseline)
  - Replacement premium (replace - baseline cost delta)
"""

import json
import os
import numpy as np
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRACK_PATH = os.path.join(SCRIPT_DIR, 'dashboard', 'track_results.json')
BASELINE_PATH = os.path.join(SCRIPT_DIR, 'dashboard', 'overprocure_results.json')

ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO']
THRESHOLDS = [50, 60, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 100]
RESOURCES = ['clean_firm', 'solar', 'wind', 'ccs_ccgt', 'hydro']


def extract_costs_and_mixes(scenarios):
    """Extract cost and mix arrays from a scenarios dict."""
    costs = []
    mixes = []
    for key, sc in scenarios.items():
        costs.append(sc['costs']['effective_cost'])
        mixes.append(sc['resource_mix'])
    return np.array(costs), mixes


def percentiles(arr, ps=[10, 25, 50, 75, 90]):
    """Compute percentiles of an array."""
    if len(arr) == 0:
        return {p: 0 for p in ps}
    return {p: float(np.percentile(arr, p)) for p in ps}


def mix_stats(mixes, resource):
    """Compute P10/P50/P90 for a resource across all scenario mixes."""
    vals = np.array([m.get(resource, 0) for m in mixes])
    return percentiles(vals)


def main():
    print("Loading results...")
    with open(TRACK_PATH) as f:
        tracks = json.load(f)
    with open(BASELINE_PATH) as f:
        baseline = json.load(f)

    mode = tracks.get('meta', {}).get('mode', 'unknown')
    print(f"Track mode: {mode}")

    # ================================================================
    # 1. COST ENVELOPES: P10/P50/P90 per track per threshold
    # ================================================================
    print("\n" + "=" * 90)
    print("  COST ENVELOPES (Effective $/MWh)")
    print("=" * 90)

    for iso in ISOS:
        print(f"\n  {iso}")
        print(f"  {'Thr':>6} | {'Baseline':^25} | {'Newbuild':^25} | {'Replace':^25} | {'Δ Replace':^12}")
        print(f"  {'':>6} | {'P10':>7} {'P50':>7} {'P90':>7}   | {'P10':>7} {'P50':>7} {'P90':>7}   | "
              f"{'P10':>7} {'P50':>7} {'P90':>7}   | {'P50':>12}")
        print(f"  {'-'*6}-+-{'-'*25}-+-{'-'*25}-+-{'-'*25}-+-{'-'*12}")

        for thr in THRESHOLDS:
            t_str = str(thr)

            # Baseline
            bl_sc = (baseline.get('results', {}).get(iso, {})
                     .get('thresholds', {}).get(t_str, {}).get('scenarios', {}))
            if bl_sc:
                bl_costs, bl_mixes = extract_costs_and_mixes(bl_sc)
                bl_p = percentiles(bl_costs)
            else:
                bl_p = {10: 0, 50: 0, 90: 0}

            # Newbuild
            nb_sc = (tracks.get('results', {}).get(iso, {})
                     .get('newbuild', {}).get(t_str, {}).get('scenarios', {}))
            if nb_sc:
                nb_costs, nb_mixes = extract_costs_and_mixes(nb_sc)
                nb_p = percentiles(nb_costs)
            else:
                nb_p = {10: 0, 50: 0, 90: 0}

            # Replace
            rp_sc = (tracks.get('results', {}).get(iso, {})
                     .get('replace', {}).get(t_str, {}).get('scenarios', {}))
            if rp_sc:
                rp_costs, rp_mixes = extract_costs_and_mixes(rp_sc)
                rp_p = percentiles(rp_costs)
            else:
                rp_p = {10: 0, 50: 0, 90: 0}

            # Replacement premium (P50)
            delta = rp_p[50] - bl_p[50] if bl_p[50] > 0 and rp_p[50] > 0 else 0

            print(f"  {thr:>5}% | ${bl_p[10]:>6.1f} ${bl_p[50]:>6.1f} ${bl_p[90]:>6.1f}   | "
                  f"${nb_p[10]:>6.1f} ${nb_p[50]:>6.1f} ${nb_p[90]:>6.1f}   | "
                  f"${rp_p[10]:>6.1f} ${rp_p[50]:>6.1f} ${rp_p[90]:>6.1f}   | "
                  f"+${delta:>5.1f}/MWh")

    # ================================================================
    # 2. RESOURCE MIX DIFFERENTIALS: Newbuild vs Baseline
    # ================================================================
    print("\n" + "=" * 90)
    print("  RESOURCE MIX DIFFERENTIALS — Newbuild vs Baseline (P50 allocation %)")
    print("  Shows what hourly matching incentivizes vs what grids actually use")
    print("=" * 90)

    for iso in ISOS:
        print(f"\n  {iso}")
        print(f"  {'Thr':>6} | {'':>8} | {'CF':>5} {'Sol':>5} {'Wnd':>5} {'CCS':>5} {'Hyd':>5} | {'Proc':>5}")
        print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*5}-{'-'*5}-{'-'*5}-{'-'*5}-{'-'*5}-+-{'-'*5}")

        for thr in THRESHOLDS:
            t_str = str(thr)

            # Baseline mixes
            bl_sc = (baseline.get('results', {}).get(iso, {})
                     .get('thresholds', {}).get(t_str, {}).get('scenarios', {}))
            # Newbuild mixes
            nb_sc = (tracks.get('results', {}).get(iso, {})
                     .get('newbuild', {}).get(t_str, {}).get('scenarios', {}))

            if not bl_sc or not nb_sc:
                continue

            bl_costs, bl_mixes = extract_costs_and_mixes(bl_sc)
            nb_costs, nb_mixes = extract_costs_and_mixes(nb_sc)

            # P50 mixes (median cost scenario's mix)
            bl_med_idx = np.argsort(bl_costs)[len(bl_costs)//2]
            nb_med_idx = np.argsort(nb_costs)[len(nb_costs)//2]
            bl_mix = bl_mixes[bl_med_idx]
            nb_mix = nb_mixes[nb_med_idx]

            # Also get P50 procurement
            bl_procs = np.array([sc['procurement_pct'] for sc in bl_sc.values()])
            nb_procs = np.array([sc['procurement_pct'] for sc in nb_sc.values()])
            bl_proc_med = int(np.median(bl_procs))
            nb_proc_med = int(np.median(nb_procs))

            print(f"  {thr:>5}% | {'baseline':>8} | "
                  f"{bl_mix.get('clean_firm',0):>5} {bl_mix.get('solar',0):>5} "
                  f"{bl_mix.get('wind',0):>5} {bl_mix.get('ccs_ccgt',0):>5} "
                  f"{bl_mix.get('hydro',0):>5} | {bl_proc_med:>4}%")
            print(f"  {'':>6} | {'newbuild':>8} | "
                  f"{nb_mix.get('clean_firm',0):>5} {nb_mix.get('solar',0):>5} "
                  f"{nb_mix.get('wind',0):>5} {nb_mix.get('ccs_ccgt',0):>5} "
                  f"{nb_mix.get('hydro',0):>5} | {nb_proc_med:>4}%")
            # Delta
            d = {r: nb_mix.get(r,0) - bl_mix.get(r,0) for r in RESOURCES}
            print(f"  {'':>6} | {'Δ':>8} | "
                  f"{d['clean_firm']:>+5} {d['solar']:>+5} "
                  f"{d['wind']:>+5} {d['ccs_ccgt']:>+5} "
                  f"{d['hydro']:>+5} | {nb_proc_med - bl_proc_med:>+4}%")

    # ================================================================
    # 3. REPLACEMENT PREMIUM — Cost delta distributions
    # ================================================================
    print("\n" + "=" * 90)
    print("  REPLACEMENT PREMIUM — Cost to go greenfield (Replace - Baseline)")
    print("  P10/P50/P90 of per-scenario cost differences across all sensitivity combos")
    print("=" * 90)

    for iso in ISOS:
        print(f"\n  {iso}")
        print(f"  {'Thr':>6} | {'P10':>8} {'P25':>8} {'P50':>8} {'P75':>8} {'P90':>8} | "
              f"{'% of BL':>8}")
        print(f"  {'-'*6}-+-{'-'*8}-{'-'*8}-{'-'*8}-{'-'*8}-{'-'*8}-+-{'-'*8}")

        for thr in THRESHOLDS:
            t_str = str(thr)

            bl_sc = (baseline.get('results', {}).get(iso, {})
                     .get('thresholds', {}).get(t_str, {}).get('scenarios', {}))
            rp_sc = (tracks.get('results', {}).get(iso, {})
                     .get('replace', {}).get(t_str, {}).get('scenarios', {}))

            if not bl_sc or not rp_sc:
                continue

            # Match scenarios by key to compute per-scenario deltas
            deltas = []
            bl_costs_matched = []
            for key in rp_sc:
                if key in bl_sc:
                    rp_cost = rp_sc[key]['costs']['effective_cost']
                    bl_cost = bl_sc[key]['costs']['effective_cost']
                    deltas.append(rp_cost - bl_cost)
                    bl_costs_matched.append(bl_cost)

            if not deltas:
                continue

            d_arr = np.array(deltas)
            bl_arr = np.array(bl_costs_matched)
            dp = percentiles(d_arr)
            pct_premium = dp[50] / np.median(bl_arr) * 100 if np.median(bl_arr) > 0 else 0

            print(f"  {thr:>5}% | +${dp[10]:>6.1f} +${dp[25]:>6.1f} +${dp[50]:>6.1f} "
                  f"+${dp[75]:>6.1f} +${dp[90]:>6.1f} | {pct_premium:>6.1f}%")

    # ================================================================
    # 4. COST ENVELOPE WIDTHS — Uncertainty from sensitivity toggles
    # ================================================================
    print("\n" + "=" * 90)
    print("  COST ENVELOPE WIDTH (P90-P10) — Sensitivity-driven uncertainty")
    print("=" * 90)

    for iso in ISOS:
        print(f"\n  {iso}")
        print(f"  {'Thr':>6} | {'Baseline':>10} {'Newbuild':>10} {'Replace':>10} | {'NB wider?':>10}")
        print(f"  {'-'*6}-+-{'-'*10}-{'-'*10}-{'-'*10}-+-{'-'*10}")

        for thr in THRESHOLDS:
            t_str = str(thr)
            widths = {}
            for label, source in [('baseline', baseline), ('newbuild', tracks), ('replace', tracks)]:
                if label == 'baseline':
                    sc = (source.get('results', {}).get(iso, {})
                          .get('thresholds', {}).get(t_str, {}).get('scenarios', {}))
                else:
                    sc = (source.get('results', {}).get(iso, {})
                          .get(label, {}).get(t_str, {}).get('scenarios', {}))
                if sc:
                    costs, _ = extract_costs_and_mixes(sc)
                    p = percentiles(costs)
                    widths[label] = p[90] - p[10]
                else:
                    widths[label] = 0

            nb_wider = widths.get('newbuild', 0) > widths.get('baseline', 0)
            print(f"  {thr:>5}% | ${widths.get('baseline',0):>8.1f} "
                  f"${widths.get('newbuild',0):>8.1f} "
                  f"${widths.get('replace',0):>8.1f} | "
                  f"{'YES' if nb_wider else 'no':>10}")

    print("\n" + "=" * 90)
    print("  ANALYSIS COMPLETE")
    print("=" * 90)


if __name__ == '__main__':
    main()
