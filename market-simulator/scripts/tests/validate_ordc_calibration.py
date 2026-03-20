#!/usr/bin/env python3
"""Validate ORDC scarcity pricing calibration after 1,215-scenario sweep.

Checks that scarcity hours are realistic and that new fossil builds produce
the expected moderate reduction in scarcity events.

Prerequisites:
    - results/sweep_1215/sweep_1215_flat.parquet must exist (run run_sweep_405.py first)

Usage:
    cd market-simulator/scripts
    python validate_ordc_calibration.py [--parquet PATH]

Output:
    - Console summary tables
    - results/sweep_1215/ordc_calibration_report.json
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pipeline_config import ORDC_PARAMS

# ── Calibration targets (approximate, based on real-world data) ──────────
# Expected annual scarcity hours (ORDC adder > $50/MWh)
CALIBRATION_TARGETS = {
    'ERCOT': {'expected_avg': (100, 400), 'acceptable': (50, 800)},
    'CAISO': {'expected_avg': (50, 200),  'acceptable': (20, 500)},
    'PJM':   {'expected_avg': (30, 150),  'acceptable': (10, 400)},
    'NYISO': {'expected_avg': (40, 200),  'acceptable': (15, 500)},
    'NEISO': {'expected_avg': (30, 150),  'acceptable': (10, 400)},
    'MISO':  {'expected_avg': (50, 250),  'acceptable': (20, 600)},
    'SPP':   {'expected_avg': (40, 200),  'acceptable': (15, 500)},
}


def load_sweep(parquet_path):
    """Load sweep parquet and validate required columns exist."""
    if not os.path.exists(parquet_path):
        print(f"ERROR: Sweep parquet not found: {parquet_path}")
        print("Run run_sweep_405.py first to generate the 1,215-scenario sweep.")
        sys.exit(1)

    df = pd.read_parquet(parquet_path)
    required = ['iso', 'year', 'ordc_scarcity_hours']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns in parquet: {missing}")
        sys.exit(1)

    total_rows = len(df)
    # Drop 2023 rows — historical year, ordc_scarcity_hours is null
    n_2023 = (df['year'] == 2023).sum()
    df = df[df['year'] != 2023].copy()

    print(f"Loaded {total_rows:,} rows from {parquet_path}")
    if n_2023 > 0:
        print(f"  Excluded {n_2023:,} year-2023 rows (ORDC not applied to historical year)")
    print(f"  Analyzing {len(df):,} rows")
    print(f"  ISOs: {sorted(df['iso'].dropna().unique())}")
    print(f"  Years: {sorted(df['year'].dropna().unique())}")
    return df


def classify_new_fossil(df):
    """Add has_new_fossil flag based on total_new_fossil_mw column."""
    if 'total_new_fossil_mw' in df.columns:
        df['has_new_fossil'] = df['total_new_fossil_mw'].fillna(0) > 0
    else:
        # Fallback: check for nb_* columns
        nb_cols = [c for c in df.columns if c.startswith('nb_') and c.endswith('_mw')]
        if nb_cols:
            df['has_new_fossil'] = df[nb_cols].fillna(0).sum(axis=1) > 0
        else:
            print("WARNING: No new fossil build columns found. Cannot compare with/without.")
            df['has_new_fossil'] = False
    return df


def parse_new_fossil_cost_level(df):
    """Extract new_fossil_cost_level from scenario_id if present."""
    if 'scenario_id' in df.columns:
        # Format: MKT_{D}_{P}_{PPA}_{G}_{Q}_{NFC} — last segment is cost level
        df['new_fossil_cost_level'] = df['scenario_id'].str.split('_').str[-1]
    return df


def check_scarcity_distribution(df):
    """Check 1: Scarcity hour distribution by ISO and year."""
    print("\n" + "=" * 80)
    print("CHECK 1: Scarcity Hours by ISO × Year")
    print("=" * 80)

    stats = df.groupby(['iso', 'year'])['ordc_scarcity_hours'].describe()
    stats = stats[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    print(stats.round(1).to_string())
    return stats


def check_new_fossil_impact(df):
    """Check 2: Compare scarcity hours with/without new fossil builds."""
    print("\n" + "=" * 80)
    print("CHECK 2: Scarcity Hours — With vs. Without New Fossil Builds")
    print("=" * 80)

    compare = df.groupby(['iso', 'has_new_fossil'])['ordc_scarcity_hours'].agg(
        ['count', 'mean', 'std', 'median']
    ).round(1)
    print(compare.to_string())

    # Compute reduction percentages using NF cost level (more meaningful than has/hasn't)
    print("\n  Scarcity hour impact of new fossil cost level (L=cheap/easy → H=expensive):")
    reductions = {}
    if 'new_fossil_cost_level' in df.columns:
        for iso in sorted(df['iso'].dropna().unique()):
            iso_df = df[df['iso'] == iso]
            low_nf = iso_df[iso_df['new_fossil_cost_level'] == 'L']['ordc_scarcity_hours'].mean()
            high_nf = iso_df[iso_df['new_fossil_cost_level'] == 'H']['ordc_scarcity_hours'].mean()
            avg_all = iso_df['ordc_scarcity_hours'].mean()
            if avg_all > 0 and low_nf > 0:
                pct = (high_nf - low_nf) / low_nf * 100
                reductions[iso] = {
                    'low_nf_cost_avg': round(low_nf, 1),
                    'high_nf_cost_avg': round(high_nf, 1),
                    'increase_pct': round(pct, 1),
                }
                direction = "✓" if pct > 0 else "⚠ UNEXPECTED"
                print(f"    {iso}: NF_L={low_nf:.0f} → NF_H={high_nf:.0f} hrs "
                      f"({pct:+.1f}% {direction})")
            else:
                reductions[iso] = {
                    'low_nf_cost_avg': round(low_nf, 1),
                    'high_nf_cost_avg': round(high_nf, 1),
                    'increase_pct': 0,
                }
                print(f"    {iso}: avg={avg_all:.0f} hrs (near-zero scarcity)")
    else:
        print("    Cannot parse NF cost level from scenario IDs")

    return compare, reductions


def check_calibration_targets(df):
    """Check 3: Compare against real-world calibration targets."""
    print("\n" + "=" * 80)
    print("CHECK 3: Calibration vs. Real-World Targets")
    print("=" * 80)

    results = {}
    all_pass = True

    for iso in sorted(df['iso'].dropna().unique()):
        if iso not in CALIBRATION_TARGETS:
            continue

        avg_hours = df[df['iso'] == iso]['ordc_scarcity_hours'].mean()
        target = CALIBRATION_TARGETS[iso]
        exp_lo, exp_hi = target['expected_avg']
        acc_lo, acc_hi = target['acceptable']

        in_expected = exp_lo <= avg_hours <= exp_hi
        in_acceptable = acc_lo <= avg_hours <= acc_hi

        if in_expected:
            status = "✓ PASS (in expected range)"
        elif in_acceptable:
            status = "~ WARN (acceptable but outside expected)"
        else:
            status = "✗ FAIL (outside acceptable range)"
            all_pass = False

        results[iso] = {
            'avg_scarcity_hours': round(avg_hours, 1),
            'expected_range': list(target['expected_avg']),
            'acceptable_range': list(target['acceptable']),
            'status': 'pass' if in_expected else ('warn' if in_acceptable else 'fail'),
        }

        print(f"  {iso:6s}: avg={avg_hours:6.1f} hrs  "
              f"expected=[{exp_lo}, {exp_hi}]  acceptable=[{acc_lo}, {acc_hi}]  {status}")

    return results, all_pass


def check_red_flags(df):
    """Check 4: Red flags — extreme values and pathological patterns."""
    print("\n" + "=" * 80)
    print("CHECK 4: Red Flag Detection")
    print("=" * 80)

    flags = []

    # Flag 1: Zero scarcity in >50% of scenarios
    for iso in sorted(df['iso'].dropna().unique()):
        iso_df = df[df['iso'] == iso]
        zero_pct = (iso_df['ordc_scarcity_hours'] == 0).mean() * 100
        if zero_pct > 50:
            msg = (f"⚠ {iso}: {zero_pct:.0f}% of scenarios have ZERO scarcity hours — "
                   "ORDC caps may be too low or knee_mw too aggressive")
            flags.append(msg)
            print(f"  {msg}")

    # Flag 2: Scarcity hours > 2000 in any scenario
    extreme = df[df['ordc_scarcity_hours'] > 2000]
    if len(extreme) > 0:
        for _, row in extreme.head(10).iterrows():
            msg = (f"⚠ {row['iso']} {row.get('year', '?')} "
                   f"scenario={row.get('scenario_id', '?')}: "
                   f"{row['ordc_scarcity_hours']:.0f} scarcity hours (>2000 is extreme)")
            flags.append(msg)
            print(f"  {msg}")

    # Flag 3: No difference with/without new fossil
    for iso in sorted(df['iso'].dropna().unique()):
        iso_df = df[df['iso'] == iso]
        no_f = iso_df[~iso_df['has_new_fossil']]['ordc_scarcity_hours'].mean()
        with_f = iso_df[iso_df['has_new_fossil']]['ordc_scarcity_hours'].mean()
        if no_f > 0 and abs(no_f - with_f) / no_f < 0.01:
            msg = (f"⚠ {iso}: <1% difference in scarcity hours with/without new fossil — "
                   "builds may be too small to matter or ORDC too loose")
            flags.append(msg)
            print(f"  {msg}")

    if not flags:
        print("  ✓ No red flags detected")

    return flags


def check_ordc_params_vs_results(df):
    """Check 5: Cross-reference ORDC params with observed behavior."""
    print("\n" + "=" * 80)
    print("CHECK 5: ORDC Parameter Summary")
    print("=" * 80)

    print(f"  {'ISO':6s}  {'VOLL':>6s}  {'knee_mw':>8s}  {'lambda':>8s}  {'cap':>5s}  "
          f"{'avg_hrs':>8s}  {'med_hrs':>8s}  {'p90_hrs':>8s}")
    print("  " + "-" * 70)

    for iso in sorted(ORDC_PARAMS.keys()):
        p = ORDC_PARAMS[iso]
        iso_df = df[df['iso'] == iso] if iso in df['iso'].values else pd.DataFrame()
        if len(iso_df) > 0:
            avg = iso_df['ordc_scarcity_hours'].mean()
            med = iso_df['ordc_scarcity_hours'].median()
            p90 = iso_df['ordc_scarcity_hours'].quantile(0.9)
        else:
            avg = med = p90 = float('nan')
        print(f"  {iso:6s}  {p['voll']:6.0f}  {p['knee_mw']:8.0f}  {p['lam']:8.4f}  "
              f"{p['cap']:5.0f}  {avg:8.1f}  {med:8.1f}  {p90:8.1f}")


def check_nf_cost_level_impact(df):
    """Check 6: Scarcity hours broken down by NF cost level (L/M/H)."""
    if 'new_fossil_cost_level' not in df.columns:
        return

    print("\n" + "=" * 80)
    print("CHECK 6: Scarcity by New Fossil Cost Level (L=cheap → H=expensive)")
    print("=" * 80)
    print("Expected: H (expensive fossil) → less fossil built → MORE scarcity hours")

    # Only show ISOs with non-zero scarcity
    active_isos = []
    for iso in sorted(df['iso'].unique()):
        if df[df['iso'] == iso]['ordc_scarcity_hours'].max() > 0:
            active_isos.append(iso)

    if not active_isos:
        print("  No ISOs with non-zero scarcity hours — cannot assess NF cost impact.")
        return

    for iso in active_isos:
        iso_df = df[df['iso'] == iso]
        pivot = iso_df.pivot_table(
            values=['ordc_scarcity_hours', 'total_new_fossil_mw', 'avg_lmp'],
            index='new_fossil_cost_level',
            aggfunc='mean',
        ).round(1)
        if len(pivot) > 0:
            print(f"\n  {iso}:")
            for level in ['L', 'M', 'H']:
                if level in pivot.index:
                    row = pivot.loc[level]
                    print(f"    NF={level}: scarcity={row.get('ordc_scarcity_hours', 0):6.1f} hrs, "
                          f"new_fossil={row.get('total_new_fossil_mw', 0):7.0f} MW, "
                          f"LMP=${row.get('avg_lmp', 0):.1f}")

            # Check directionality: H should have more scarcity than L
            if 'L' in pivot.index and 'H' in pivot.index:
                l_scarcity = pivot.loc['L', 'ordc_scarcity_hours']
                h_scarcity = pivot.loc['H', 'ordc_scarcity_hours']
                if h_scarcity > l_scarcity:
                    print(f"    → Feedback loop CORRECT: H cost → +{h_scarcity - l_scarcity:.0f} "
                          f"more scarcity hrs than L")
                elif h_scarcity == l_scarcity:
                    print(f"    → No difference between L and H — feedback loop NOT visible")
                else:
                    print(f"    → UNEXPECTED: L cost has MORE scarcity than H "
                          f"({l_scarcity:.0f} vs {h_scarcity:.0f})")


def generate_recommendations(calibration_results, flags, reductions):
    """Generate actionable recommendations."""
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    recs = []

    # Check for ISOs that need recalibration
    for iso, result in calibration_results.items():
        if result['status'] == 'fail':
            avg = result['avg_scarcity_hours']
            exp_lo, exp_hi = result['expected_range']
            if avg < exp_lo:
                recs.append(
                    f"  {iso}: avg={avg:.0f} hrs is BELOW target [{exp_lo}, {exp_hi}]. "
                    f"Consider INCREASING knee_mw (currently {ORDC_PARAMS[iso]['knee_mw']} MW) "
                    f"or DECREASING cap (currently ${ORDC_PARAMS[iso]['cap']}/MWh)."
                )
            else:
                recs.append(
                    f"  {iso}: avg={avg:.0f} hrs is ABOVE target [{exp_lo}, {exp_hi}]. "
                    f"Consider DECREASING knee_mw (currently {ORDC_PARAMS[iso]['knee_mw']} MW) "
                    f"or INCREASING cap (currently ${ORDC_PARAMS[iso]['cap']}/MWh)."
                )

    # Check ERCOT specifically (should show largest ORDC sensitivity to NF builds)
    if 'ERCOT' in reductions:
        ercot_pct = reductions['ERCOT'].get('increase_pct', 0)
        other_pcts = [v['increase_pct'] for k, v in reductions.items()
                      if k != 'ERCOT' and v.get('increase_pct', 0) > 0]
        if other_pcts and ercot_pct < np.mean(other_pcts):
            recs.append(
                f"  ERCOT: Expected LARGEST scarcity sensitivity to NF cost "
                f"(energy-only market) but increase ({ercot_pct:.0f}%) is below average "
                f"of other ISOs ({np.mean(other_pcts):.0f}%). May need ERCOT-specific tuning."
            )

    if not recs:
        recs.append("  ✓ All ISOs within acceptable calibration range. No parameter changes needed.")

    for r in recs:
        print(r)

    return recs


def main():
    parser = argparse.ArgumentParser(description='Validate ORDC calibration after 1,215 sweep')
    parser.add_argument('--parquet', default=None,
                        help='Path to sweep_1215_flat.parquet')
    args = parser.parse_args()

    default_path = os.path.join(
        os.path.dirname(__file__), '..', 'results', 'sweep_1215', 'sweep_1215_flat.parquet')
    parquet_path = args.parquet or default_path

    # Load and prepare data
    df = load_sweep(parquet_path)
    df = classify_new_fossil(df)
    df = parse_new_fossil_cost_level(df)

    # Run all checks
    iso_year_stats = check_scarcity_distribution(df)
    fossil_compare, reductions = check_new_fossil_impact(df)
    calibration_results, all_pass = check_calibration_targets(df)
    flags = check_red_flags(df)
    check_ordc_params_vs_results(df)
    check_nf_cost_level_impact(df)
    recs = generate_recommendations(calibration_results, flags, reductions)

    # Build report
    report = {
        'sweep_rows': len(df),
        'isos': sorted(df['iso'].dropna().unique().tolist()),
        'years': sorted(df['year'].dropna().unique().tolist()),
        'calibration': calibration_results,
        'new_fossil_impact': reductions,
        'red_flags': flags,
        'recommendations': recs,
        'all_pass': all_pass and len(flags) == 0,
        'ordc_params': {k: dict(v) for k, v in ORDC_PARAMS.items()},
    }

    # Save report
    report_dir = os.path.dirname(parquet_path)
    report_path = os.path.join(report_dir, 'ordc_calibration_report.json')
    os.makedirs(report_dir, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nReport saved: {report_path}")

    # Final verdict
    print("\n" + "=" * 80)
    if report['all_pass']:
        print("VERDICT: ✓ ORDC calibration PASSED — no parameter changes needed")
    else:
        print("VERDICT: ⚠ ORDC calibration needs attention — see recommendations above")
    print("=" * 80)

    return 0 if report['all_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
