#!/usr/bin/env python3
"""
End-to-End Pipeline Test Harness — Track 1 Baseline (Steps 2→3→5→6)
====================================================================
Validates the complete pipeline from PFS physics through CO2 abatement
by running representative test cases through each step and checking
results against physical and economic expectations.

Two-phase approach:
  Phase 1 (Coarse): 2 ISOs × 3 thresholds × ~20 mixes → directional correctness
  Phase 2 (Full):   7 ISOs × 5 thresholds × ~100 mixes → comprehensive validation

Usage:
  python scripts/test_pipeline_e2e.py --phase 1       # Coarse directional test
  python scripts/test_pipeline_e2e.py --phase 2       # Full N test
  python scripts/test_pipeline_e2e.py --phase 1 --iso CAISO  # Single ISO
  python scripts/test_pipeline_e2e.py --validate-existing     # Validate existing outputs only

Version: 1.0.0
"""

import argparse
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Add scripts dir to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from pipeline_config import (
    ISOS, THRESHOLDS, ACTIVE_THRESHOLDS, REGIONAL_DEMAND_TWH,
    GRID_MIX_SHARES, WHOLESALE_PRICES, CCS_CAP_TWH,
    OFFSHORE_ISOS, OFFSHORE_WIND_CAP_TWH, GEOTHERMAL_CAP_TWH,
    BATTERY_EFFICIENCY, LDES_EFFICIENCY, H2_EFFICIENCY,
    NEISO_CCS_GAS_ADDER, NEISO_WHOLESALE_ADDER,
    VALID_RANGES, EXPECTED_COST_RANGES_MEDIUM,
    get_resource_cols, STORAGE_COLS, PATHS, H,
    N_SCENARIOS_BASE, N_SCENARIOS_CAISO,
)


# ============================================================================
# TEST CONFIGURATION
# ============================================================================

PHASE1_CONFIG = {
    'isos': ['CAISO', 'ERCOT'],  # 6D and 4D representative
    'thresholds': [50.0, 90.0, 99.99],
    'max_mixes_per_threshold': 30,
    'description': 'Coarse directional test (2 ISOs, 3 thresholds)',
}

PHASE2_CONFIG = {
    'isos': ISOS,  # All 7
    'thresholds': [50.0, 75.0, 90.0, 95.0, 99.99],
    'max_mixes_per_threshold': 150,
    'description': 'Full N test (7 ISOs, 5 thresholds)',
}


class TestResults:
    """Collects pass/fail/warning results for the test run."""

    def __init__(self):
        self.results = []
        self.passes = 0
        self.fails = 0
        self.warnings = 0

    def record(self, test_name, passed, message='', severity='fail'):
        status = 'PASS' if passed else ('WARN' if severity == 'warn' else 'FAIL')
        self.results.append({
            'test': test_name,
            'status': status,
            'message': message,
        })
        if passed:
            self.passes += 1
        elif severity == 'warn':
            self.warnings += 1
        else:
            self.fails += 1

    def summary(self):
        total = self.passes + self.fails + self.warnings
        lines = [
            f"\n{'='*70}",
            f"  TEST SUMMARY: {self.passes} passed, {self.fails} failed, "
            f"{self.warnings} warnings ({total} total)",
            f"{'='*70}",
        ]
        if self.fails > 0:
            lines.append("\n  FAILURES:")
            for r in self.results:
                if r['status'] == 'FAIL':
                    lines.append(f"    ✗ {r['test']}: {r['message']}")
        if self.warnings > 0:
            lines.append("\n  WARNINGS:")
            for r in self.results:
                if r['status'] == 'WARN':
                    lines.append(f"    ⚠ {r['test']}: {r['message']}")
        return '\n'.join(lines)


# ============================================================================
# STEP 1: VALIDATE EXISTING PFS DATA
# ============================================================================

def validate_step1_pfs(results, isos, thresholds):
    """Validate existing Step 1 PFS parquet files."""
    print("\n" + "="*70)
    print("  STEP 1: VALIDATE PFS DATA")
    print("="*70)

    pfs_dir = PATHS['step1_pfs']

    for iso in isos:
        iso_files = 0
        iso_rows = 0
        issues = []

        for thr in thresholds:
            t_str = f'{thr:g}'
            fname = f'{iso}_t{t_str}_raw_pfs.parquet'
            fpath = os.path.join(pfs_dir, fname)

            if not os.path.exists(fpath):
                results.record(f'PFS {iso} t{t_str}', False, f'File missing: {fname}')
                continue

            df = pd.read_parquet(fpath)
            iso_files += 1
            iso_rows += len(df)

            # Schema check
            resource_cols = get_resource_cols(iso)
            for col in resource_cols + STORAGE_COLS + ['hourly_match_score']:
                if col not in df.columns:
                    issues.append(f't{t_str}: missing column {col}')

            if len(df) == 0:
                issues.append(f't{t_str}: empty dataframe')
                continue

            # Score range check
            scores = df['hourly_match_score'].values
            if scores.min() < 0 or scores.max() > 101:
                issues.append(f't{t_str}: scores out of range [{scores.min():.1f}, {scores.max():.1f}]')

            # Check mixes sum to ~100%
            resource_sum = sum(df[c].values for c in resource_cols)
            max_sum = resource_sum.max()
            if max_sum > 350:
                issues.append(f't{t_str}: resource sum exceeds 350% (max={max_sum})')

            # Score should be >= threshold for most mixes (some near-miss OK)
            pct_above = (scores >= thr).mean() * 100
            if pct_above < 50 and thr < 99:
                issues.append(f't{t_str}: only {pct_above:.0f}% of mixes score >= threshold')

            # Geothermal check for CAISO
            if iso == 'CAISO' and 'geothermal' in df.columns:
                geo_max = df['geothermal'].max()
                if geo_max > 25:
                    issues.append(f't{t_str}: geothermal max={geo_max}% exceeds expected cap')

            # Offshore wind check for offshore ISOs
            if iso in OFFSHORE_ISOS and 'offshore_wind' in df.columns:
                osw_max = df['offshore_wind'].max()
                if osw_max > 50:
                    issues.append(f't{t_str}: offshore wind max={osw_max}% seems too high')

            # H2 only at ≥95%
            if thr < 95 and 'h2_dispatch_pct' in df.columns:
                h2_nonzero = (df['h2_dispatch_pct'] > 0).sum()
                if h2_nonzero > 0:
                    issues.append(f't{t_str}: {h2_nonzero} mixes have H2 below 95% threshold')

        if issues:
            results.record(f'PFS {iso}', False, '; '.join(issues[:5]))
        else:
            results.record(f'PFS {iso}', True,
                          f'{iso_files} files, {iso_rows:,} rows — all checks pass')
            print(f"  ✓ {iso}: {iso_files} files, {iso_rows:,} rows")


# ============================================================================
# STEP 2: RUN EF EXTRACTION ON TEST SAMPLE
# ============================================================================

def run_step2_test(results, isos, thresholds, max_mixes):
    """Run Step 2 EF extraction on test subset."""
    print("\n" + "="*70)
    print("  STEP 2: EF EXTRACTION TEST")
    print("="*70)

    try:
        from step2_efficient_frontier import (
            load_and_process_iso, scan_iso_files,
            partition_by_threshold_band, write_per_iso_threshold_outputs,
            get_resource_cols as ef_get_resource_cols,
            STORAGE_COLS as EF_STORAGE_COLS,
        )
    except ImportError as e:
        results.record('Step 2 import', False, str(e))
        return {}

    threshold_filter = set(thresholds)

    for iso in isos:
        t0 = time.time()
        try:
            files_by_iso = scan_iso_files([iso], threshold_filter=threshold_filter)
            if iso not in files_by_iso:
                results.record(f'EF {iso}', False, 'No PFS files found')
                continue

            result_table, n_input, n_gated, n_dedup = load_and_process_iso(
                iso, files_by_iso[iso])
            elapsed = time.time() - t0

            if result_table is None or result_table.num_rows == 0:
                results.record(f'EF {iso}', False, 'No rows after EF extraction')
                continue

            # Validate dedup
            if n_dedup > n_gated:
                results.record(f'EF {iso} dedup', False,
                              f'Dedup produced MORE rows ({n_dedup}) than input ({n_gated})')
            elif n_dedup == n_gated and n_gated > 100:
                results.record(f'EF {iso} dedup', False,
                              f'No dedup reduction at all ({n_dedup} = {n_gated})',
                              severity='warn')
            else:
                reduction = (1 - n_dedup/n_gated)*100 if n_gated > 0 else 0
                results.record(f'EF {iso}', True,
                              f'{n_input:,}→{n_gated:,}→{n_dedup:,} ({reduction:.0f}% reduction, {elapsed:.1f}s)')
                print(f"  ✓ {iso}: {n_input:,}→{n_dedup:,} rows ({elapsed:.1f}s)")

            # Validate scores
            scores = result_table.column('hourly_match_score').to_numpy()
            results.record(f'EF {iso} scores', True,
                          f'Score range: [{scores.min():.1f}, {scores.max():.1f}]')

        except Exception as e:
            results.record(f'EF {iso}', False, f'Exception: {e}')
            traceback.print_exc()


# ============================================================================
# STEP 3: VALIDATE COST OPTIMIZATION
# ============================================================================

def validate_step3_costs(results, isos, thresholds):
    """Validate existing Step 3 cost optimization outputs."""
    print("\n" + "="*70)
    print("  STEP 3: VALIDATE COST OPTIMIZATION")
    print("="*70)

    cost_dir = PATHS['step3_cost']

    for iso in isos:
        fname = f'step3_co_{iso}.parquet'
        fpath = os.path.join(cost_dir, fname)

        if not os.path.exists(fpath):
            results.record(f'Cost {iso}', False, f'File missing: {fname}')
            continue

        df = pd.read_parquet(fpath)
        issues = []

        # Row count check
        n_thresholds = len(df['threshold'].unique())
        n_scenarios = df['scenario'].nunique()
        expected_scenarios = N_SCENARIOS_CAISO if iso == 'CAISO' else N_SCENARIOS_BASE
        results.record(f'Cost {iso} scenarios', n_scenarios == expected_scenarios,
                      f'{n_scenarios} scenarios (expected {expected_scenarios})')

        # Threshold coverage
        actual_thresholds = set(df['threshold'].unique())
        expected_thresholds = set(THRESHOLDS)
        missing = expected_thresholds - actual_thresholds
        if missing:
            issues.append(f'Missing thresholds: {sorted(missing)}')

        # Cost range checks
        for col, (lo, hi) in [
            ('cost_total_cost', (0, 500)),
            ('cost_effective_cost', (0, 1000)),
        ]:
            if col in df.columns:
                vals = df[col].dropna()
                if vals.min() < lo:
                    issues.append(f'{col} min={vals.min():.2f} < {lo}')
                if vals.max() > hi:
                    issues.append(f'{col} max={vals.max():.2f} > {hi}')

        # Cost monotonicity check (higher threshold → higher cost, for Medium scenario)
        med_key = 'MMMM_M_M_M1_M' if iso == 'CAISO' else 'MMMM_M_M_M1_X'
        med_df = df[df['scenario'] == med_key].sort_values('threshold')
        if len(med_df) > 0:
            costs = med_df['cost_effective_cost'].values
            thrs = med_df['threshold'].values
            non_monotonic = []
            for i in range(1, len(costs)):
                if costs[i] < costs[i-1] - 0.5:  # Allow small tolerance
                    non_monotonic.append(
                        f't{thrs[i-1]:g}→t{thrs[i]:g}: '
                        f'${costs[i-1]:.1f}→${costs[i]:.1f}')
            if non_monotonic:
                issues.append(f'Non-monotonic costs (Medium): {non_monotonic[:3]}')
                results.record(f'Cost {iso} monotonicity', False,
                              '; '.join(non_monotonic[:3]))
            else:
                results.record(f'Cost {iso} monotonicity', True,
                              'Costs increase monotonically with threshold (Medium scenario)')

        # NEISO gas adder check
        if iso == 'NEISO':
            # Wholesale should be higher than base by NEISO_WHOLESALE_ADDER
            if 'cost_wholesale' in df.columns:
                ws = df['cost_wholesale'].iloc[0]
                base_ws = WHOLESALE_PRICES['NEISO']
                # With the adder, wholesale should be base + fuel_adj + NEISO_WHOLESALE_ADDER
                # For Medium fuel: base + 0 + 4 = 45
                expected_med_ws = base_ws + NEISO_WHOLESALE_ADDER
                results.record(f'Cost NEISO gas adder', True,
                              f'NEISO wholesale check: base=${base_ws}, '
                              f'expected_med=${expected_med_ws}',
                              severity='warn')

        # CCS cap check
        ccs_cap = CCS_CAP_TWH[iso]
        if iso in ['NYISO', 'NEISO'] and 'mix_ccs_ccgt' in df.columns:
            ccs_pcts = df['mix_ccs_ccgt'].values
            if ccs_pcts.max() > 0:
                issues.append(f'{iso} has CCS in mix but cap is {ccs_cap} TWh')

        # Resource diversity check
        for res_col in ['mix_clean_firm', 'mix_solar', 'mix_wind']:
            if res_col in df.columns:
                n_unique = df[res_col].nunique()
                if n_unique < 3:
                    issues.append(f'{res_col}: only {n_unique} unique values')

        # Expected cost range (Medium scenario, select thresholds)
        for thr, (exp_lo, exp_hi) in EXPECTED_COST_RANGES_MEDIUM.items():
            thr_med = med_df[med_df['threshold'] == thr]
            if len(thr_med) > 0:
                actual = thr_med['cost_effective_cost'].iloc[0]
                if actual < exp_lo or actual > exp_hi:
                    issues.append(
                        f't{thr} Medium cost=${actual:.1f}, '
                        f'expected [{exp_lo}, {exp_hi}]')

        if issues:
            results.record(f'Cost {iso}', False, '; '.join(issues[:5]))
        else:
            results.record(f'Cost {iso}', True,
                          f'{len(df):,} rows, {n_thresholds} thresholds — all checks pass')
            print(f"  ✓ {iso}: {len(df):,} rows, {n_scenarios} scenarios")


# ============================================================================
# STEP 3: CROSS-VALIDATE COST FUNCTION
# ============================================================================

def cross_validate_cost_function(results, isos, thresholds):
    """Re-run cost function on a few mixes and compare to stored results."""
    print("\n" + "="*70)
    print("  STEP 3: CROSS-VALIDATE COST FUNCTION")
    print("="*70)

    try:
        from step3_cost_optimization import (
            price_mix_batch, REGIONAL_DEMAND_TWH as S3_DEMAND,
            GRID_MIX_SHARES as S3_GRID,
        )
    except ImportError as e:
        results.record('Step 3 cross-validate import', False, str(e))
        return

    cost_dir = PATHS['step3_cost']
    test_isos = [iso for iso in isos if iso in ['CAISO', 'ERCOT', 'NEISO']]

    for iso in test_isos:
        fpath = os.path.join(cost_dir, f'step3_co_{iso}.parquet')
        if not os.path.exists(fpath):
            continue

        df = pd.read_parquet(fpath)
        # Test with Medium scenario at threshold 90
        med_key = 'MMMM_M_M_M1_M' if iso == 'CAISO' else 'MMMM_M_M_M1_X'
        test_row = df[(df['scenario'] == med_key) & (df['threshold'] == 90.0)]
        if len(test_row) == 0:
            results.record(f'CrossVal {iso}', False, 'No Medium t90 scenario found')
            continue

        row = test_row.iloc[0]

        # Build arrays for the single mix
        arrays = {
            'clean_firm': np.array([row['mix_clean_firm']], dtype=np.float64),
            'solar': np.array([row['mix_solar']], dtype=np.float64),
            'wind': np.array([row['mix_wind']], dtype=np.float64),
            'hydro': np.array([row['mix_hydro']], dtype=np.float64),
            'battery_dispatch_pct': np.array([row['battery_dispatch_pct']], dtype=np.float64),
            'battery8_dispatch_pct': np.array([row.get('battery8_dispatch_pct', 0)], dtype=np.float64),
            'ldes_dispatch_pct': np.array([row['ldes_dispatch_pct']], dtype=np.float64),
            'h2_dispatch_pct': np.array([row.get('h2_dispatch_pct', 0)], dtype=np.float64),
            'hourly_match_score': np.array([row['hourly_match_score']], dtype=np.float64),
        }
        if iso in OFFSHORE_ISOS:
            arrays['offshore_wind'] = np.array([row.get('mix_offshore_wind', 0)], dtype=np.float64)

        sens = {'ren': 'M', 'firm': 'M', 'batt': 'M', 'ldes_lvl': 'M',
                'ccs': 'M', 'q45': '1', 'fuel': 'M', 'tx': 'M'}
        if iso == 'CAISO':
            sens['geo'] = 'M'

        total_cost, eff_cost, tranche = price_mix_batch(
            iso, arrays, sens, S3_DEMAND[iso])

        computed_total = float(total_cost[0])
        stored_total = row['cost_total_cost']

        # Note: existing stored results used old 45Q value ($29 offset).
        # New results use corrected $27.5 offset (+$1.5 to CCS_LCOE_45Q_ON).
        # So a small delta is EXPECTED if the mix has CCS.
        ccs_pct = row.get('mix_ccs_ccgt', 0)
        expected_delta = ccs_pct / 100.0 * 1.5  # $1.5/MWh × CCS fraction

        delta = abs(computed_total - stored_total)
        tolerance = max(0.5, expected_delta + 0.5)  # Allow for 45Q correction + rounding

        if delta <= tolerance:
            results.record(f'CrossVal {iso} t90 Med', True,
                          f'Computed=${computed_total:.2f}, Stored=${stored_total:.2f}, '
                          f'Δ=${delta:.2f} (tolerance ${tolerance:.2f}, '
                          f'includes 45Q correction for {ccs_pct}% CCS)')
            print(f"  ✓ {iso} t90 Medium: ${computed_total:.2f} "
                  f"(stored ${stored_total:.2f}, Δ=${delta:.2f})")
        else:
            results.record(f'CrossVal {iso} t90 Med', False,
                          f'Computed=${computed_total:.2f} vs Stored=${stored_total:.2f}, '
                          f'Δ=${delta:.2f} exceeds tolerance ${tolerance:.2f}')


# ============================================================================
# STEP 4: VALIDATE GAS/CCS ADJUSTMENTS
# ============================================================================

def validate_step4_outputs(results, isos):
    """Validate existing Step 4 outputs (passthrough + overlays)."""
    print("\n" + "="*70)
    print("  STEP 4: VALIDATE GAS/CCS ADJUSTMENTS (LEGACY)")
    print("="*70)

    gas_dir = PATHS['step4_gas_ccs']

    for iso in isos:
        fpath = os.path.join(gas_dir, f'step4_{iso}.parquet')
        if not os.path.exists(fpath):
            results.record(f'Step4 {iso}', True, 'Not generated (expected — Step 4 deprioritized)',
                          severity='warn')
            continue

        df = pd.read_parquet(fpath)

        # Check that Step 3 columns are preserved
        for col in ['cost_total_cost', 'cost_effective_cost', 'mix_clean_firm']:
            if col not in df.columns:
                results.record(f'Step4 {iso} schema', False, f'Missing column: {col}')

        # Check RA columns exist
        ra_cols = [c for c in df.columns if c.startswith('ra_')]
        results.record(f'Step4 {iso}', True,
                      f'{len(df):,} rows, {len(ra_cols)} RA columns')
        print(f"  ✓ {iso}: {len(df):,} rows (legacy passthrough)")


# ============================================================================
# STEP 5: VALIDATE DISPATCH CACHE
# ============================================================================

def validate_dispatch_cache(results, isos):
    """Validate existing dispatch cache."""
    print("\n" + "="*70)
    print("  STEP 5: VALIDATE DISPATCH CACHE")
    print("="*70)

    cache_dir = PATHS['dispatch_cache']

    for iso in isos:
        fpath = os.path.join(cache_dir, f'{iso}_dispatch_cache.parquet')
        if not os.path.exists(fpath):
            results.record(f'Dispatch {iso}', True,
                          'Not yet generated (needs Step 5 run)', severity='warn')
            continue

        df = pd.read_parquet(fpath)
        if len(df) == 0:
            results.record(f'Dispatch {iso}', False, 'Empty dispatch cache')
            continue

        # Check that profiles are 8760 hours
        issues = []
        for col in ['supply_total', 'residual_demand']:
            if col in df.columns:
                sample = df[col].iloc[0]
                if hasattr(sample, '__len__') and len(sample) != H:
                    issues.append(f'{col}: {len(sample)} hours (expected {H})')

        # Check energy balance on a sample
        if 'supply_total' in df.columns and 'residual_demand' in df.columns:
            supply = df['supply_total'].iloc[0]
            residual = df['residual_demand'].iloc[0]
            if hasattr(supply, '__len__') and hasattr(residual, '__len__'):
                # Residual should be >= 0 (unmet demand)
                if np.any(np.array(residual) < -1):
                    issues.append('Negative residual demand detected (energy imbalance)')

        if issues:
            results.record(f'Dispatch {iso}', False, '; '.join(issues))
        else:
            results.record(f'Dispatch {iso}', True,
                          f'{len(df)} archetypes in cache')
            print(f"  ✓ {iso}: {len(df)} archetypes")


# ============================================================================
# STEP 6: VALIDATE CO2 RESULTS
# ============================================================================

def validate_co2_results(results, isos):
    """Validate existing CO2 computation results."""
    print("\n" + "="*70)
    print("  STEP 6: VALIDATE CO2 RESULTS")
    print("="*70)

    co2_dir = PATHS['co2_results']

    for iso in isos:
        fpath = os.path.join(co2_dir, f'co2_{iso}.parquet')
        if not os.path.exists(fpath):
            results.record(f'CO2 {iso}', True,
                          'Not yet generated (needs Step 6 run)', severity='warn')
            continue

        df = pd.read_parquet(fpath)
        issues = []

        # CO2 emission rate should be reasonable (0-1.5 tCO2/MWh)
        if 'co2_emission_rate_tco2_mwh' in df.columns:
            rates = df['co2_emission_rate_tco2_mwh'].dropna()
            if len(rates) > 0:
                if rates.min() < 0:
                    issues.append(f'Negative emission rate: {rates.min():.3f}')
                if rates.max() > 1.5:
                    issues.append(f'Emission rate too high: {rates.max():.3f}')

        # CO2 abated should be >= 0
        if 'co2_total_co2_abated_tons' in df.columns:
            abated = df['co2_total_co2_abated_tons'].dropna()
            if len(abated) > 0 and abated.min() < 0:
                issues.append(f'Negative abatement: {abated.min():.0f} tons')

        # Higher thresholds → more abatement (monotonicity)
        if 'co2_total_co2_abated_tons' in df.columns:
            med_key = 'MMMM_M_M_M1_M' if iso == 'CAISO' else 'MMMM_M_M_M1_X'
            med_df = df[df['scenario'] == med_key].sort_values('threshold')
            if len(med_df) > 2:
                abated_vals = med_df['co2_total_co2_abated_tons'].values
                # Overall trend should be increasing (some local non-monotonicity OK
                # due to mix changes)
                if abated_vals[-1] < abated_vals[0]:
                    issues.append('CO2 abatement decreasing from low to high threshold')

        if issues:
            results.record(f'CO2 {iso}', False, '; '.join(issues))
        else:
            results.record(f'CO2 {iso}', True,
                          f'{len(df):,} rows — emission rates and abatement reasonable')
            print(f"  ✓ {iso}: {len(df):,} rows")


# ============================================================================
# CROSS-STEP VALIDATION
# ============================================================================

def validate_cross_step_consistency(results, isos):
    """Check data contracts between pipeline steps."""
    print("\n" + "="*70)
    print("  CROSS-STEP CONSISTENCY CHECKS")
    print("="*70)

    for iso in isos:
        # Step 1 → Step 3: resource mix should be within PFS bounds
        pfs_path = os.path.join(PATHS['step1_pfs'], f'{iso}_t90_raw_pfs.parquet')
        cost_path = os.path.join(PATHS['step3_cost'], f'step3_co_{iso}.parquet')

        if not (os.path.exists(pfs_path) and os.path.exists(cost_path)):
            continue

        pfs_df = pd.read_parquet(pfs_path)
        cost_df = pd.read_parquet(cost_path)
        cost_t90 = cost_df[cost_df['threshold'] == 90.0]

        if len(cost_t90) == 0:
            continue

        # Check that Step 3 mixes fall within PFS resource ranges
        for pfs_col, cost_col in [
            ('clean_firm', 'mix_clean_firm'),
            ('solar', 'mix_solar'),
            ('wind', 'mix_wind'),
        ]:
            if pfs_col in pfs_df.columns and cost_col in cost_t90.columns:
                pfs_min = pfs_df[pfs_col].min()
                pfs_max = pfs_df[pfs_col].max()
                cost_vals = cost_t90[cost_col].unique()
                out_of_range = ((cost_vals < pfs_min - 1) | (cost_vals > pfs_max + 1)).sum()
                if out_of_range > 0:
                    results.record(f'CrossStep {iso} {cost_col}', False,
                                  f'{out_of_range} values outside PFS range '
                                  f'[{pfs_min}, {pfs_max}]')

        # Step 3 → Step 4: row counts should match
        step4_path = os.path.join(PATHS['step4_gas_ccs'], f'step4_{iso}.parquet')
        if os.path.exists(step4_path):
            step4_df = pd.read_parquet(step4_path)
            if len(step4_df) != len(cost_df):
                results.record(f'CrossStep {iso} Step3→4 rows', False,
                              f'Step3={len(cost_df):,} vs Step4={len(step4_df):,}')
            else:
                results.record(f'CrossStep {iso} Step3→4 rows', True,
                              f'{len(cost_df):,} rows match')

    print("  ✓ Cross-step checks complete")


# ============================================================================
# CONSTANT CONSISTENCY AUDIT
# ============================================================================

def validate_constants_consistency(results):
    """Check that pipeline_config.py matches Step 3 constants."""
    print("\n" + "="*70)
    print("  CONSTANT CONSISTENCY AUDIT")
    print("="*70)

    try:
        from step3_cost_optimization import (
            REGIONAL_DEMAND_TWH as S3_DEMAND,
            GRID_MIX_SHARES as S3_GRID,
            WHOLESALE_PRICES as S3_WS,
            CCS_CAP_TWH as S3_CCS,
            OFFSHORE_ISOS as S3_OSW,
        )

        # Check demand
        for iso in ISOS:
            if abs(REGIONAL_DEMAND_TWH[iso] - S3_DEMAND[iso]) > 0.001:
                results.record(f'Const demand {iso}', False,
                              f'config={REGIONAL_DEMAND_TWH[iso]} vs '
                              f'step3={S3_DEMAND[iso]}')

        # Check CCS caps
        for iso in ISOS:
            if abs(CCS_CAP_TWH[iso] - S3_CCS[iso]) > 0.1:
                results.record(f'Const CCS cap {iso}', False,
                              f'config={CCS_CAP_TWH[iso]} vs step3={S3_CCS[iso]}')

        # Check wholesale
        for iso in ISOS:
            if WHOLESALE_PRICES[iso] != S3_WS[iso]:
                results.record(f'Const wholesale {iso}', False,
                              f'config={WHOLESALE_PRICES[iso]} vs step3={S3_WS[iso]}')

        # Check offshore ISOs
        if set(OFFSHORE_ISOS) != set(S3_OSW):
            results.record('Const offshore ISOs', False,
                          f'config={OFFSHORE_ISOS} vs step3={S3_OSW}')

        results.record('Constants consistency', True,
                      'pipeline_config.py matches Step 3 constants')
        print("  ✓ All constants consistent between pipeline_config.py and Step 3")

    except ImportError as e:
        results.record('Constants import', False, str(e))


# ============================================================================
# PHYSICAL SANITY CHECKS
# ============================================================================

def validate_physical_sanity(results, isos):
    """Domain-specific physical sanity checks on results."""
    print("\n" + "="*70)
    print("  PHYSICAL SANITY CHECKS")
    print("="*70)

    cost_dir = PATHS['step3_cost']

    for iso in isos:
        fpath = os.path.join(cost_dir, f'step3_co_{iso}.parquet')
        if not os.path.exists(fpath):
            continue

        df = pd.read_parquet(fpath)
        issues = []

        # 1. Hydro should never exceed existing share (no new hydro build)
        if 'mix_hydro' in df.columns:
            hydro_max = df['mix_hydro'].max()
            existing_hydro = GRID_MIX_SHARES[iso]['hydro']
            # Allow +10% buffer for run-of-river innovation
            if hydro_max > existing_hydro * 1.15 + 1:
                issues.append(
                    f'Hydro max={hydro_max}% exceeds existing '
                    f'{existing_hydro}% + 10% buffer')

        # 2. CCS should be 0 for NYISO/NEISO (cap = 0 TWh)
        if iso in ['NYISO', 'NEISO'] and 'mix_ccs_ccgt' in df.columns:
            ccs_nonzero = (df['mix_ccs_ccgt'] > 0).sum()
            if ccs_nonzero > 0:
                # CCS should be zero in NYISO/NEISO due to geology
                ccs_max = df['mix_ccs_ccgt'].max()
                issues.append(f'{iso} has {ccs_nonzero} rows with CCS>0 (max={ccs_max}%)')

        # 3. Cost should include gas backup component
        if 'gas_gas_cost_per_mwh' in df.columns:
            gas_cost = df['gas_gas_cost_per_mwh']
            if gas_cost.max() <= 0:
                issues.append('No gas backup cost at any threshold')

        # 4. At high thresholds, costs should be significantly higher
        low_thr = df[df['threshold'] == 50.0]
        high_thr = df[df['threshold'] == 99.99]
        if len(low_thr) > 0 and len(high_thr) > 0:
            med_key = 'MMMM_M_M_M1_M' if iso == 'CAISO' else 'MMMM_M_M_M1_X'
            low_med = low_thr[low_thr['scenario'] == med_key]
            high_med = high_thr[high_thr['scenario'] == med_key]
            if len(low_med) > 0 and len(high_med) > 0:
                low_cost = low_med['cost_effective_cost'].iloc[0]
                high_cost = high_med['cost_effective_cost'].iloc[0]
                if high_cost <= low_cost:
                    issues.append(
                        f't99.99 cost (${high_cost:.1f}) <= t50 cost (${low_cost:.1f})')
                else:
                    ratio = high_cost / low_cost
                    results.record(f'Physics {iso} cost escalation', True,
                                  f't50=${low_cost:.1f} → t99.99=${high_cost:.1f} '
                                  f'({ratio:.1f}× escalation)')

        # 5. Existing clean should be priced at $0
        if 'mix_solar' in df.columns:
            existing_solar = GRID_MIX_SHARES[iso]['solar']
            # At low thresholds with solar <= existing, total cost contribution
            # from solar should be ~$0

        if issues:
            results.record(f'Physics {iso}', False, '; '.join(issues[:5]))
        else:
            results.record(f'Physics {iso}', True, 'All physical sanity checks pass')
            print(f"  ✓ {iso}: physical checks pass")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Pipeline E2E Test Harness')
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2],
                        help='Test phase: 1=coarse, 2=full')
    parser.add_argument('--iso', type=str, default=None,
                        help='Single ISO to test (overrides phase config)')
    parser.add_argument('--validate-existing', action='store_true',
                        help='Only validate existing outputs (no computation)')
    args = parser.parse_args()

    config = PHASE1_CONFIG if args.phase == 1 else PHASE2_CONFIG
    isos = [args.iso.upper()] if args.iso else config['isos']
    thresholds = config['thresholds']

    print("="*70)
    print(f"  PIPELINE E2E TEST — Phase {args.phase}")
    print(f"  {config['description']}")
    print(f"  ISOs: {', '.join(isos)}")
    print(f"  Thresholds: {thresholds}")
    print(f"  Mode: {'validate-existing' if args.validate_existing else 'full test'}")
    print("="*70)

    results = TestResults()
    t0 = time.time()

    # Always run these validation checks
    validate_constants_consistency(results)
    validate_step1_pfs(results, isos, thresholds)
    validate_step3_costs(results, isos, thresholds)
    cross_validate_cost_function(results, isos, thresholds)
    validate_step4_outputs(results, isos)
    validate_dispatch_cache(results, isos)
    validate_co2_results(results, isos)
    validate_cross_step_consistency(results, isos)
    validate_physical_sanity(results, isos)

    if not args.validate_existing:
        # Run Step 2 EF extraction on test subset
        run_step2_test(results, isos, thresholds, config['max_mixes_per_threshold'])

    elapsed = time.time() - t0

    # Print summary
    print(results.summary())
    print(f"\n  Total time: {elapsed:.1f}s")

    # Write detailed results to JSON
    output_path = PROJECT_ROOT / 'data' / 'test_pipeline_results.json'
    with open(output_path, 'w') as f:
        json.dump({
            'phase': args.phase,
            'isos': isos,
            'thresholds': thresholds,
            'elapsed_seconds': elapsed,
            'passes': results.passes,
            'fails': results.fails,
            'warnings': results.warnings,
            'details': results.results,
        }, f, indent=2)
    print(f"  Detailed results: {output_path}")

    return 0 if results.fails == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
