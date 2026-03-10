#!/usr/bin/env python3
"""Extract convergence analysis data comparing Step 5B and Step 10 LMP models.

Generates dashboard/js/convergence-data.js with:
1. Step 5B avg LMP by threshold per ISO (equilibrium wholesale price)
2. Step 10 reference case avg LMP at 2050 by clean% bin (deployment-time price)
3. Revenue component breakdown comparison
4. Key divergence metrics

IMPORTANT NOTE on interpretation:
- Step 5B avg_lmp: System-average wholesale price AFTER the mix is established
  (equilibrium price at X% clean)
- Step 10 avg_lmp: LMP at the time of DEPLOYMENT (reflects fossil stack that
  existed BEFORE the zone was deployed, not after)
- These measure fundamentally different things and are NOT directly comparable
- Convergence analysis should compare Step 5B at X% with Step 10 at X% while
  acknowledging the temporal framing difference
"""
import os
import sys
import json
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LMP_DIR = os.path.join(ROOT_DIR, 'data', 'step4-analysis', 'lmp')
STEP10_DIR = os.path.join(ROOT_DIR, 'data', 'step6-smartargets')
OUTPUT_JS = os.path.join(ROOT_DIR, 'dashboard', 'js', 'convergence-data.js')

ALL_ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']
COMPARE_THRESHOLDS = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]


def main():
    print("=" * 70)
    print("  Convergence Analysis: Step 5B vs Step 10")
    print("=" * 70)

    convergence = {}

    for iso in ALL_ISOS:
        s5_path = os.path.join(LMP_DIR, f'{iso}_lmp.parquet')
        s10_path = os.path.join(STEP10_DIR, f'sweep_reference_{iso}.parquet')

        if not os.path.exists(s5_path) or not os.path.exists(s10_path):
            print(f"  {iso}: missing data, skipping")
            continue

        s5 = pd.read_parquet(s5_path)
        s10 = pd.read_parquet(s10_path)

        s5_med = s5[s5['fuel_level'] == 'Medium'] if 'fuel_level' in s5.columns else s5
        s10_2050 = s10[s10['year'] == 2050].copy()

        # Step 5B: avg LMP stats by threshold
        s5_by_t = {}
        for t in COMPARE_THRESHOLDS:
            subset = s5_med[s5_med['threshold'] == t]['avg_lmp']
            if len(subset) > 0:
                s5_by_t[str(t)] = {
                    'p10': round(float(subset.quantile(0.10)), 1),
                    'p50': round(float(subset.quantile(0.50)), 1),
                    'p90': round(float(subset.quantile(0.90)), 1),
                }

        # Step 5B: total market revenue stats by threshold
        s5_total_by_t = {}
        if 'total_market_rev_mwh' in s5_med.columns:
            for t in COMPARE_THRESHOLDS:
                subset = s5_med[s5_med['threshold'] == t]['total_market_rev_mwh']
                if len(subset) > 0:
                    s5_total_by_t[str(t)] = {
                        'p10': round(float(subset.quantile(0.10)), 1),
                        'p50': round(float(subset.quantile(0.50)), 1),
                        'p90': round(float(subset.quantile(0.90)), 1),
                    }

        # Step 10: avg LMP by clean% bin at 2050
        s10_2050['clean_bin'] = (s10_2050['clean_pct'] / 5).round() * 5
        s10_by_bin = {}
        for b in sorted(s10_2050['clean_bin'].unique()):
            subset = s10_2050[s10_2050['clean_bin'] == b]
            if len(subset) >= 3:  # need enough data points
                s10_by_bin[str(int(b))] = {
                    'n': len(subset),
                    'avg_lmp_p50': round(float(subset['avg_lmp'].median()), 1),
                    'clean_pct_range': [
                        round(float(subset['clean_pct'].min()), 0),
                        round(float(subset['clean_pct'].max()), 0)
                    ],
                }
                # Revenue breakdown if available
                for col in ['energy_rev_mwh', 'capacity_rev_mwh', 'rec_rev_mwh', 'revenue_per_mwh']:
                    if col in subset.columns:
                        vals = subset[col].dropna()
                        if len(vals) > 0:
                            s10_by_bin[str(int(b))][col + '_p50'] = round(float(vals.median()), 1)

        # Reference case P10/P50/P90 trajectory
        ref_trajectory = {}
        years = [2023, 2030, 2035, 2040, 2045, 2050]
        for metric in ['clean_pct', 'avg_lmp', 'emissions_mt', 'revenue_per_mwh',
                        'energy_rev_mwh', 'capacity_rev_mwh', 'rec_rev_mwh']:
            if metric not in s10.columns:
                continue
            p10, p50, p90 = [], [], []
            for yr in years:
                subset = s10[s10['year'] == yr][metric].dropna()
                if len(subset) > 0:
                    p10.append(round(float(subset.quantile(0.10)), 1))
                    p50.append(round(float(subset.quantile(0.50)), 1))
                    p90.append(round(float(subset.quantile(0.90)), 1))
                else:
                    p10.append(0)
                    p50.append(0)
                    p90.append(0)
            ref_trajectory[metric] = {'p10': p10, 'p50': p50, 'p90': p90}

        # Compute delta at overlapping thresholds
        deltas = []
        for t in COMPARE_THRESHOLDS:
            s5_val = s5_by_t.get(str(t), {}).get('p50')
            s10_val = s10_by_bin.get(str(t), {}).get('avg_lmp_p50')
            if s5_val is not None and s10_val is not None:
                deltas.append({
                    'threshold': t,
                    'step5b_lmp': s5_val,
                    'step10_lmp': s10_val,
                    'delta': round(s10_val - s5_val, 1),
                })

        convergence[iso] = {
            'step5b_lmp': s5_by_t,
            'step5b_total_rev': s5_total_by_t,
            'step10_by_clean_pct': s10_by_bin,
            'step10_trajectory': ref_trajectory,
            'deltas': deltas,
            'step10_clean_range_2050': [
                round(float(s10_2050['clean_pct'].min()), 0),
                round(float(s10_2050['clean_pct'].max()), 0),
            ],
            'years': years,
        }

        # Summary
        avg_delta = np.mean([d['delta'] for d in deltas]) if deltas else 0
        print(f"  {iso}: {len(deltas)} comparison points, avg delta = ${avg_delta:+.1f}/MWh")

    # Write JS
    js = "// Auto-generated by extract_convergence_data.py\n"
    js += "// Convergence analysis: Step 5B (SBTi pathway) vs Step 10 (market reference)\n"
    js += "//\n"
    js += "// IMPORTANT: Step 5B avg_lmp = equilibrium price AFTER mix is established\n"
    js += "// Step 10 avg_lmp = price during DEPLOYMENT (before zone deployed)\n"
    js += "// These measure different things - see convergence page for full explanation\n\n"
    js += f"var CONVERGENCE = {json.dumps(convergence, indent=2)};\n"

    with open(OUTPUT_JS, 'w') as f:
        f.write(js)

    print(f"\n→ Wrote {OUTPUT_JS}")
    print(f"  {len(convergence)} ISOs")


if __name__ == '__main__':
    main()
