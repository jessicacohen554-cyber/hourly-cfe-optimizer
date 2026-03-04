#!/usr/bin/env python3
"""
Analyze track results: cost envelopes, mix differentials, replacement premium.

Reads:
  - dashboard/track_results.json (newbuild + cost_to_replace tracks)
  - dashboard/overprocure_results.json (baseline)

Outputs:
  - Console report with P10/P50/P90 cost envelopes per track
  - Resource mix differentials (newbuild vs baseline)
  - Replacement premium (cost_to_replace - baseline cost delta)
"""

import json
import os
import sys
import glob as glob_mod
import numpy as np
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'scripts'))

TRACK_PATH = os.path.join(SCRIPT_DIR, 'dashboard', 'track_results.json')
BASELINE_PATH = os.path.join(SCRIPT_DIR, 'dashboard', 'overprocure_results.json')
CO2_BATCH_DIR = os.path.join(SCRIPT_DIR, 'data', 'step5-post-processing', 'co2_results')

from pipeline_config import OUTPUT_THRESHOLDS as THRESHOLDS

ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']
THRESHOLDS_STR = [str(t) for t in THRESHOLDS]
RESOURCES = ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro']


def load_from_co2_parquets(batch_dir):
    """Load CO2 results from parquet files in batch_dir (vectorized, fast).

    Only reads columns needed for analysis (scenario, threshold, effective_cost,
    resource mix). Uses vectorized column access instead of iterrows() for
    50-100x speedup on large DataFrames (e.g., CAISO has 367K rows).
    """
    import pandas as pd
    co2_parquets = [f for f in os.listdir(batch_dir) if f.endswith('.parquet')]
    if not co2_parquets:
        return None

    # Columns we actually need for analysis
    needed_cols = ['scenario', 'threshold', 'cost_effective_cost', 'cost_total_cost',
                   'cost_incremental', 'cost_wholesale', 'hourly_match_score'] + \
                  [f'mix_{r}' for r in RESOURCES]

    data = {'results': {}}
    for iso in ISOS:
        parquet_path = None
        for prefix in ['step4_', 'step3_co_', 'co2_']:
            candidate = os.path.join(batch_dir, f'{prefix}{iso}.parquet')
            if os.path.exists(candidate):
                parquet_path = candidate
                break
        if parquet_path is None:
            continue

        # Only read columns we need — use pyarrow to inspect schema without loading data
        import pyarrow.parquet as pq_inspect
        schema_cols = [f.name for f in pq_inspect.read_schema(parquet_path)]
        read_cols = [c for c in needed_cols if c in schema_cols]
        df = pd.read_parquet(parquet_path, columns=read_cols)
        print(f"  Loaded {iso}: {len(df):,} rows from {os.path.basename(parquet_path)}")

        iso_data = {'thresholds': {}}
        for threshold, thr_group in df.groupby('threshold'):
            t_str = str(threshold) if threshold != int(threshold) else str(int(threshold))
            # Build scenario dicts using vectorized column access
            sc_keys = thr_group['scenario'].values
            scenarios = {}
            eff_costs = thr_group['cost_effective_cost'].values if 'cost_effective_cost' in thr_group.columns else np.zeros(len(thr_group))
            tot_costs = thr_group['cost_total_cost'].values if 'cost_total_cost' in thr_group.columns else np.zeros(len(thr_group))
            inc_costs = thr_group['cost_incremental'].values if 'cost_incremental' in thr_group.columns else np.zeros(len(thr_group))
            ws_costs = thr_group['cost_wholesale'].values if 'cost_wholesale' in thr_group.columns else np.zeros(len(thr_group))
            hms = thr_group['hourly_match_score'].values if 'hourly_match_score' in thr_group.columns else np.zeros(len(thr_group))
            mix_arrs = {}
            for r in RESOURCES:
                col = f'mix_{r}'
                mix_arrs[r] = thr_group[col].values if col in thr_group.columns else np.zeros(len(thr_group), dtype=int)

            for i, sc_key in enumerate(sc_keys):
                scenarios[sc_key] = {
                    'resource_mix': {r: int(mix_arrs[r][i]) for r in RESOURCES},
                    'costs': {
                        'total_cost': float(tot_costs[i]),
                        'effective_cost': float(eff_costs[i]),
                        'incremental': float(inc_costs[i]),
                        'wholesale': float(ws_costs[i]),
                    },
                    'hourly_match_score': float(hms[i]),
                }
            iso_data['thresholds'][t_str] = {'scenarios': scenarios}

        data['results'][iso] = iso_data
        print(f"    → {len(iso_data['thresholds'])} thresholds")

    return data if data['results'] else None


def load_from_co2_batches(batch_dir):
    """Reassemble results dict from batched co2 result JSON files (legacy)."""
    config_path = os.path.join(batch_dir, '_config.json')
    if not os.path.exists(config_path):
        return None
    with open(config_path) as f:
        data = json.load(f)
    data.setdefault('results', {})
    for iso in ISOS:
        iso_entry = {}
        for mf in sorted(glob_mod.glob(os.path.join(batch_dir, f'{iso}_meta_*.json'))):
            with open(mf) as f:
                iso_entry.update(json.load(f))
        iso_entry.setdefault('thresholds', {})
        for t in THRESHOLDS_STR:
            matches = sorted(glob_mod.glob(os.path.join(batch_dir, f'{iso}_{t}_*.json')))
            if matches:
                with open(matches[0]) as f:
                    iso_entry['thresholds'][t] = json.load(f)
        if iso_entry.get('thresholds') or 'annual_demand_mwh' in iso_entry:
            data['results'][iso] = iso_entry
    return data


def load_baseline():
    """Load baseline results: CO2 parquets → monolithic JSON → batched co2 JSONs → step4 parquets."""
    # Prefer CO2-enriched parquets (output of step5_compute_co2.py)
    if os.path.isdir(CO2_BATCH_DIR):
        data = load_from_co2_parquets(CO2_BATCH_DIR)
        if data and data.get('results'):
            print(f"  Baseline loaded from CO2 parquets ({len(data['results'])} ISOs)")
            return data
    # Monolithic JSON
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH) as f:
            return json.load(f)
    # Legacy batched CO2 JSONs
    if os.path.isdir(CO2_BATCH_DIR):
        data = load_from_co2_batches(CO2_BATCH_DIR)
        if data and data.get('results'):
            print(f"  Baseline loaded from batched co2 JSONs ({len(data['results'])} ISOs)")
            return data
    # Step 4 parquet fallback
    try:
        from parquet_io import load_from_parquets, find_input_dir
        input_dir = find_input_dir(ISOS)
        if input_dir:
            data = load_from_parquets(input_dir, ISOS)
            print(f"  Baseline loaded from parquets: {input_dir}")
            return data
    except ImportError:
        pass
    raise FileNotFoundError(
        f"No baseline found. Checked:\n  {CO2_BATCH_DIR}/ (parquets)\n  {BASELINE_PATH}\n  {CO2_BATCH_DIR}/ (JSONs)\n  step4 parquet dirs")


def extract_costs_and_mixes(scenarios):
    """Extract cost and mix arrays from a scenarios dict."""
    vals = list(scenarios.values())
    costs = np.array([sc['costs']['effective_cost'] for sc in vals])
    mixes = [sc['resource_mix'] for sc in vals]
    return costs, mixes


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
    baseline = load_baseline()

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
        print(f"  {'Thr':>6} | {'Baseline':^25} | {'Newbuild':^25} | {'Cost-to-Replace':^25} | {'Δ CTR':^12}")
        print(f"  {'':>6} | {'P10':>7} {'P50':>7} {'P90':>7}   | {'P10':>7} {'P50':>7} {'P90':>7}   | "
              f"{'P10':>7} {'P50':>7} {'P90':>7}   | {'P50':>12}")
        print(f"  {'-'*6}-+-{'-'*25}-+-{'-'*25}-+-{'-'*25}-+-{'-'*12}")

        # Extract ISO-level data once to avoid repeated nested dict lookups
        iso_baseline = baseline.get('results', {}).get(iso, {})
        iso_tracks = tracks.get('results', {}).get(iso, {})
        iso_bl_thresholds = iso_baseline.get('thresholds', {})
        iso_nb_data = iso_tracks.get('newbuild', {})
        iso_rp_data = iso_tracks.get('cost_to_replace', {})

        for thr in THRESHOLDS:
            t_str = str(thr)

            bl_sc = iso_bl_thresholds.get(t_str, {}).get('scenarios', {})
            if bl_sc:
                bl_costs, bl_mixes = extract_costs_and_mixes(bl_sc)
                bl_p = percentiles(bl_costs)
            else:
                bl_p = {10: 0, 50: 0, 90: 0}

            nb_sc = iso_nb_data.get(t_str, {}).get('scenarios', {})
            if nb_sc:
                nb_costs, nb_mixes = extract_costs_and_mixes(nb_sc)
                nb_p = percentiles(nb_costs)
            else:
                nb_p = {10: 0, 50: 0, 90: 0}

            rp_sc = iso_rp_data.get(t_str, {}).get('scenarios', {})
            if rp_sc:
                rp_costs, rp_mixes = extract_costs_and_mixes(rp_sc)
                rp_p = percentiles(rp_costs)
            else:
                rp_p = {10: 0, 50: 0, 90: 0}

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
        print(f"  {'Thr':>6} | {'':>8} | {'CF':>5} {'Sol':>5} {'Wnd':>5} {'OSW':>5} {'CCS':>5} {'Hyd':>5}")
        print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*5}-{'-'*5}-{'-'*5}-{'-'*5}-{'-'*5}-{'-'*5}")

        # Extract ISO-level data once
        iso_bl_thresholds = baseline.get('results', {}).get(iso, {}).get('thresholds', {})
        iso_nb_data = tracks.get('results', {}).get(iso, {}).get('newbuild', {})

        for thr in THRESHOLDS:
            t_str = str(thr)

            bl_sc = iso_bl_thresholds.get(t_str, {}).get('scenarios', {})
            nb_sc = iso_nb_data.get(t_str, {}).get('scenarios', {})

            if not bl_sc or not nb_sc:
                continue

            bl_costs, bl_mixes = extract_costs_and_mixes(bl_sc)
            nb_costs, nb_mixes = extract_costs_and_mixes(nb_sc)

            # P50 mixes — use argpartition (O(n)) instead of argsort (O(n log n))
            bl_med_idx = np.argpartition(bl_costs, len(bl_costs)//2)[len(bl_costs)//2]
            nb_med_idx = np.argpartition(nb_costs, len(nb_costs)//2)[len(nb_costs)//2]
            bl_mix = bl_mixes[bl_med_idx]
            nb_mix = nb_mixes[nb_med_idx]

            print(f"  {thr:>5}% | {'baseline':>8} | "
                  f"{bl_mix.get('clean_firm',0):>5} {bl_mix.get('solar',0):>5} "
                  f"{bl_mix.get('wind',0):>5} {bl_mix.get('offshore_wind',0):>5} "
                  f"{bl_mix.get('ccs_ccgt',0):>5} {bl_mix.get('hydro',0):>5}")
            print(f"  {'':>6} | {'newbuild':>8} | "
                  f"{nb_mix.get('clean_firm',0):>5} {nb_mix.get('solar',0):>5} "
                  f"{nb_mix.get('wind',0):>5} {nb_mix.get('offshore_wind',0):>5} "
                  f"{nb_mix.get('ccs_ccgt',0):>5} {nb_mix.get('hydro',0):>5}")
            d = {r: nb_mix.get(r,0) - bl_mix.get(r,0) for r in RESOURCES}
            print(f"  {'':>6} | {'Δ':>8} | "
                  f"{d['clean_firm']:>+5} {d['solar']:>+5} "
                  f"{d['wind']:>+5} {d['offshore_wind']:>+5} "
                  f"{d['ccs_ccgt']:>+5} {d['hydro']:>+5}")

    # ================================================================
    # 3. REPLACEMENT PREMIUM — Cost delta distributions
    # ================================================================
    print("\n" + "=" * 90)
    print("  REPLACEMENT PREMIUM — Cost to go greenfield (Cost-to-Replace - Baseline)")
    print("  P10/P50/P90 of per-scenario cost differences across all sensitivity combos")
    print("=" * 90)

    for iso in ISOS:
        print(f"\n  {iso}")
        print(f"  {'Thr':>6} | {'P10':>8} {'P25':>8} {'P50':>8} {'P75':>8} {'P90':>8} | "
              f"{'% of BL':>8}")
        print(f"  {'-'*6}-+-{'-'*8}-{'-'*8}-{'-'*8}-{'-'*8}-{'-'*8}-+-{'-'*8}")

        # Extract ISO-level data once
        iso_bl_thresholds = baseline.get('results', {}).get(iso, {}).get('thresholds', {})
        iso_rp_data = tracks.get('results', {}).get(iso, {}).get('cost_to_replace', {})

        for thr in THRESHOLDS:
            t_str = str(thr)

            bl_sc = iso_bl_thresholds.get(t_str, {}).get('scenarios', {})
            rp_sc = iso_rp_data.get(t_str, {}).get('scenarios', {})

            if not bl_sc or not rp_sc:
                continue

            # Vectorized: match scenarios by key using set intersection
            common_keys = set(rp_sc) & set(bl_sc)
            if not common_keys:
                continue

            rp_costs = np.array([rp_sc[k]['costs']['effective_cost'] for k in common_keys])
            bl_costs_matched = np.array([bl_sc[k]['costs']['effective_cost'] for k in common_keys])
            d_arr = rp_costs - bl_costs_matched

            dp = percentiles(d_arr)
            bl_median = np.median(bl_costs_matched)
            pct_premium = dp[50] / bl_median * 100 if bl_median > 0 else 0

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
        print(f"  {'Thr':>6} | {'Baseline':>10} {'Newbuild':>10} {'CTR':>10} | {'NB wider?':>10}")
        print(f"  {'-'*6}-+-{'-'*10}-{'-'*10}-{'-'*10}-+-{'-'*10}")

        # Extract ISO-level data once
        iso_bl_thresholds = baseline.get('results', {}).get(iso, {}).get('thresholds', {})
        iso_tracks = tracks.get('results', {}).get(iso, {})

        for thr in THRESHOLDS:
            t_str = str(thr)
            widths = {}

            bl_sc = iso_bl_thresholds.get(t_str, {}).get('scenarios', {})
            nb_sc = iso_tracks.get('newbuild', {}).get(t_str, {}).get('scenarios', {})
            rp_sc = iso_tracks.get('cost_to_replace', {}).get(t_str, {}).get('scenarios', {})

            for label, sc in [('baseline', bl_sc), ('newbuild', nb_sc), ('cost_to_replace', rp_sc)]:
                if sc:
                    costs, _ = extract_costs_and_mixes(sc)
                    p = percentiles(costs)
                    widths[label] = p[90] - p[10]
                else:
                    widths[label] = 0

            nb_wider = widths.get('newbuild', 0) > widths.get('baseline', 0)
            print(f"  {thr:>5}% | ${widths.get('baseline',0):>8.1f} "
                  f"${widths.get('newbuild',0):>8.1f} "
                  f"${widths.get('cost_to_replace',0):>8.1f} | "
                  f"{'YES' if nb_wider else 'no':>10}")

    print("\n" + "=" * 90)
    print("  ANALYSIS COMPLETE")
    print("=" * 90)


if __name__ == '__main__':
    main()
