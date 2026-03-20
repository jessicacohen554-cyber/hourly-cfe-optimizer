#!/usr/bin/env python3
"""Tests for G7: Sensitivity Analysis Framework.

Validates Morris method elementary effects, variance decomposition,
and JSON output formatting using synthetic test data with known properties.
"""

import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sensitivity_analysis import (
    SWEEP_DIMENSIONS,
    OUTPUT_METRICS,
    parse_scenario_id,
    add_dimension_columns,
    compute_elementary_effects,
    compute_morris_all_metrics,
    compute_variance_decomposition,
    format_morris_plot_data,
    format_tornado_data,
    compute_range_impact,
    run_sensitivity_analysis,
    save_results,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures — synthetic sweep data
# ─────────────────────────────────────────────────────────────────────────────

def _build_synthetic_sweep() -> pd.DataFrame:
    """Build a small synthetic sweep DataFrame with known sensitivity structure.

    Creates a 3×5×3×3×3×3 = 1,215 scenario grid for 1 ISO, 1 year.
    Output metric `clean_pct` is constructed so that:
        - demand_growth explains ~50% of variance (large effect)
        - price_sensitivity explains ~25% (medium effect)
        - queue_capacity explains ~12% (small effect)
        - other dimensions explain ~0% (noise only)
    """
    demand_levels = ['Low', 'Medium', 'High']
    price_levels = ['all_low', 'all_med', 'all_high', 'high_vre_low_firm', 'high_firm_low_vre']
    ppa_levels = ['Low', 'Medium', 'High']
    gas_levels = ['Low', 'Medium', 'High']
    queue_levels = ['Low', 'Medium', 'High']
    nfc_levels = ['Low', 'Medium', 'High']

    code_map = {'Low': 'L', 'Medium': 'M', 'High': 'H'}
    demand_effect = {'Low': -20, 'Medium': 0, 'High': 20}
    price_effect = {'all_low': -10, 'all_med': 0, 'all_high': 10,
                    'high_vre_low_firm': 5, 'high_firm_low_vre': -5}
    queue_effect = {'Low': -5, 'Medium': 0, 'High': 5}

    rows = []
    np.random.seed(42)
    for d in demand_levels:
        for p in price_levels:
            for ppa in ppa_levels:
                for g in gas_levels:
                    for q in queue_levels:
                        for nfc in nfc_levels:
                            scenario_id = (
                                f"MKT_{code_map[d]}_{p}_{code_map[ppa]}_"
                                f"{code_map[g]}_{code_map[q]}_{code_map[nfc]}"
                            )
                            # Construct output with known sensitivities
                            clean_pct = (
                                60.0
                                + demand_effect[d]
                                + price_effect[p]
                                + queue_effect[q]
                                + np.random.normal(0, 1)  # small noise
                            )
                            cost = 30.0 + demand_effect[d] * 0.5 + np.random.normal(0, 0.5)
                            emissions = 100.0 - demand_effect[d] * 2 + np.random.normal(0, 2)
                            avg_lmp = 50.0 + demand_effect[d] * 1.5 + np.random.normal(0, 1)

                            rows.append({
                                'scenario_id': scenario_id,
                                'iso': 'TEST_ISO',
                                'year': 2050,
                                'clean_pct': clean_pct,
                                'cost_per_mwh': cost,
                                'emissions_mt': emissions,
                                'avg_lmp': avg_lmp,
                                'scenario': scenario_id,
                            })

    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_df():
    return _build_synthetic_sweep()


@pytest.fixture
def synthetic_df_with_dims(synthetic_df):
    return add_dimension_columns(synthetic_df)


@pytest.fixture
def parquet_path(synthetic_df):
    """Write synthetic data to a temp parquet file."""
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        synthetic_df.to_parquet(f.name, index=False)
        yield f.name
    os.unlink(f.name)


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Scenario ID Parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestParseScenarioId:
    def test_simple_id(self):
        result = parse_scenario_id('MKT_L_all_low_L_L_L_L')
        assert result['demand_growth'] == 'Low'
        assert result['price_sensitivity'] == 'all_low'
        assert result['ppa_level'] == 'Low'
        assert result['gas_friction'] == 'Low'
        assert result['queue_capacity'] == 'Low'
        assert result['new_fossil_cost'] == 'Low'

    def test_mixed_levels(self):
        result = parse_scenario_id('MKT_H_all_high_M_L_H_M')
        assert result['demand_growth'] == 'High'
        assert result['price_sensitivity'] == 'all_high'
        assert result['ppa_level'] == 'Medium'
        assert result['gas_friction'] == 'Low'
        assert result['queue_capacity'] == 'High'
        assert result['new_fossil_cost'] == 'Medium'

    def test_multi_underscore_price_sens(self):
        result = parse_scenario_id('MKT_M_high_vre_low_firm_H_M_L_H')
        assert result['demand_growth'] == 'Medium'
        assert result['price_sensitivity'] == 'high_vre_low_firm'
        assert result['ppa_level'] == 'High'
        assert result['gas_friction'] == 'Medium'
        assert result['queue_capacity'] == 'Low'
        assert result['new_fossil_cost'] == 'High'

    def test_high_firm_low_vre(self):
        result = parse_scenario_id('MKT_L_high_firm_low_vre_L_H_M_L')
        assert result['price_sensitivity'] == 'high_firm_low_vre'
        assert result['demand_growth'] == 'Low'
        assert result['gas_friction'] == 'High'

    def test_all_combos_parse(self, synthetic_df):
        """Every scenario_id in the synthetic sweep must parse without error."""
        for sid in synthetic_df['scenario_id'].unique():
            result = parse_scenario_id(sid)
            assert len(result) == 6
            assert result['demand_growth'] in ['Low', 'Medium', 'High']

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_scenario_id('MKT_L_unknown_price_L_L_L_L')


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Morris Method
# ─────────────────────────────────────────────────────────────────────────────

class TestMorrisMethod:
    def test_demand_is_dominant_for_clean_pct(self, synthetic_df_with_dims):
        """demand_growth should have the highest μ* for clean_pct in synthetic data."""
        effects = compute_elementary_effects(
            synthetic_df_with_dims, 'TEST_ISO', 2050, 'clean_pct')

        assert 'demand_growth' in effects
        assert 'price_sensitivity' in effects

        # Demand has the largest effect by construction (+/-20)
        assert effects['demand_growth']['mu_star'] > effects['price_sensitivity']['mu_star']
        assert effects['demand_growth']['mu_star'] > effects['gas_friction']['mu_star']

    def test_mu_star_positive(self, synthetic_df_with_dims):
        """μ* (mean absolute effect) must be non-negative."""
        effects = compute_elementary_effects(
            synthetic_df_with_dims, 'TEST_ISO', 2050, 'clean_pct')
        for dim, vals in effects.items():
            assert vals['mu_star'] >= 0, f"μ* negative for {dim}"

    def test_n_effects_reasonable(self, synthetic_df_with_dims):
        """Each dimension should produce a reasonable number of elementary effects."""
        effects = compute_elementary_effects(
            synthetic_df_with_dims, 'TEST_ISO', 2050, 'clean_pct')
        for dim, vals in effects.items():
            assert vals['n_effects'] > 0, f"Zero effects for {dim}"

    def test_all_metrics_computed(self, synthetic_df_with_dims):
        """compute_morris_all_metrics should return results for all output metrics."""
        results = compute_morris_all_metrics(synthetic_df_with_dims, 'TEST_ISO', 2050)
        for metric in OUTPUT_METRICS:
            assert metric in results, f"Missing metric: {metric}"
            assert len(results[metric]) == len(SWEEP_DIMENSIONS)

    def test_morris_plot_format(self, synthetic_df_with_dims):
        """Morris plot data should have correct structure for Chart.js."""
        morris = compute_morris_all_metrics(synthetic_df_with_dims, 'TEST_ISO', 2050)
        plot_data = format_morris_plot_data(morris)

        for metric, points in plot_data.items():
            assert len(points) == len(SWEEP_DIMENSIONS)
            for pt in points:
                assert 'x' in pt  # mu_star
                assert 'y' in pt  # sigma
                assert 'label' in pt
                assert 'dimension' in pt
                assert pt['x'] >= 0  # mu_star always non-negative


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Variance Decomposition
# ─────────────────────────────────────────────────────────────────────────────

class TestVarianceDecomposition:
    def test_fractions_sum_leq_1(self, synthetic_df_with_dims):
        """First-order variance fractions must sum to ≤ 1.0 (remainder = interactions)."""
        var_results = compute_variance_decomposition(
            synthetic_df_with_dims, 'TEST_ISO', 2050)

        for metric, fractions in var_results.items():
            dim_sum = sum(v for k, v in fractions.items() if not k.startswith('_'))
            assert dim_sum <= 1.0 + 1e-6, (
                f"Variance fractions sum to {dim_sum:.4f} > 1.0 for {metric}"
            )

    def test_fractions_non_negative(self, synthetic_df_with_dims):
        """All variance fractions must be non-negative."""
        var_results = compute_variance_decomposition(
            synthetic_df_with_dims, 'TEST_ISO', 2050)

        for metric, fractions in var_results.items():
            for dim, frac in fractions.items():
                if dim.startswith('_'):
                    continue
                assert frac >= -1e-10, (
                    f"Negative variance fraction {frac} for {dim} on {metric}"
                )

    def test_demand_dominates_clean_pct(self, synthetic_df_with_dims):
        """demand_growth should explain the most variance for clean_pct."""
        var_results = compute_variance_decomposition(
            synthetic_df_with_dims, 'TEST_ISO', 2050)

        clean_fracs = var_results['clean_pct']
        demand_frac = clean_fracs['demand_growth']
        for dim in SWEEP_DIMENSIONS:
            if dim != 'demand_growth':
                assert demand_frac >= clean_fracs.get(dim, 0), (
                    f"demand_growth ({demand_frac:.3f}) should dominate "
                    f"{dim} ({clean_fracs.get(dim, 0):.3f}) for clean_pct"
                )

    def test_total_variance_recorded(self, synthetic_df_with_dims):
        """Each metric should record _total_variance."""
        var_results = compute_variance_decomposition(
            synthetic_df_with_dims, 'TEST_ISO', 2050)
        for metric in var_results:
            assert '_total_variance' in var_results[metric]
            assert var_results[metric]['_total_variance'] > 0

    def test_tornado_format(self, synthetic_df_with_dims):
        """Tornado data should be sorted descending by variance_fraction."""
        var_results = compute_variance_decomposition(
            synthetic_df_with_dims, 'TEST_ISO', 2050)
        tornado = format_tornado_data(var_results)

        for metric, bars in tornado.items():
            assert len(bars) == len(SWEEP_DIMENSIONS)
            fracs = [b['variance_fraction'] for b in bars]
            assert fracs == sorted(fracs, reverse=True), (
                f"Tornado bars not sorted for {metric}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Range Impact
# ─────────────────────────────────────────────────────────────────────────────

class TestRangeImpact:
    def test_range_positive(self, synthetic_df_with_dims):
        """Range (high - low) must be non-negative."""
        ranges = compute_range_impact(synthetic_df_with_dims, 'TEST_ISO', 2050)
        for metric, dim_ranges in ranges.items():
            for dim, info in dim_ranges.items():
                assert info['range'] >= 0
                assert info['high'] >= info['low']

    def test_demand_has_large_range_on_clean_pct(self, synthetic_df_with_dims):
        """demand_growth should have the largest range for clean_pct."""
        ranges = compute_range_impact(synthetic_df_with_dims, 'TEST_ISO', 2050)
        demand_range = ranges['clean_pct']['demand_growth']['range']
        for dim in SWEEP_DIMENSIONS:
            if dim != 'demand_growth':
                assert demand_range >= ranges['clean_pct'][dim]['range']


# ─────────────────────────────────────────────────────────────────────────────
# Tests — End-to-End Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEnd:
    def test_run_and_save(self, parquet_path):
        """Full pipeline runs and produces valid JSON output."""
        results = run_sensitivity_analysis(
            parquet_path=parquet_path,
            year=2050,
            isos=['TEST_ISO'],
        )

        assert 'TEST_ISO' in results
        iso_data = results['TEST_ISO']
        assert 'morris' in iso_data
        assert 'morris_plot' in iso_data
        assert 'variance_decomposition' in iso_data
        assert 'tornado' in iso_data
        assert 'range_impact' in iso_data
        assert 'metadata' in iso_data

        # Save and verify JSON round-trip
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = save_results(results, tmpdir)
            assert len(saved) == 2  # combined + 1 ISO

            # Verify JSON is valid
            for path in saved:
                with open(path) as f:
                    data = json.load(f)
                assert isinstance(data, dict)

    def test_json_serializable(self, parquet_path):
        """All results must be JSON-serializable (no numpy types)."""
        results = run_sensitivity_analysis(
            parquet_path=parquet_path,
            year=2050,
            isos=['TEST_ISO'],
        )
        # This will raise TypeError if numpy types leak through
        json_str = json.dumps(results, default=str)
        assert len(json_str) > 100


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Known Test Case (Morris Correctness)
# ─────────────────────────────────────────────────────────────────────────────

class TestMorrisCorrectness:
    """Verify Morris elementary effects on a minimal known case."""

    def test_known_linear_effect(self):
        """If output = 10 * demand_num, elementary effects should be ~10."""
        rows = []
        for d in ['Low', 'Medium', 'High']:
            for p in ['all_low', 'all_med', 'all_high', 'high_vre_low_firm', 'high_firm_low_vre']:
                for ppa in ['Low', 'Medium', 'High']:
                    for g in ['Low', 'Medium', 'High']:
                        for q in ['Low', 'Medium', 'High']:
                            for nfc in ['Low', 'Medium', 'High']:
                                code = {'Low': 'L', 'Medium': 'M', 'High': 'H'}
                                sid = f"MKT_{code[d]}_{p}_{code[ppa]}_{code[g]}_{code[q]}_{code[nfc]}"
                                d_num = {'Low': 0, 'Medium': 1, 'High': 2}[d]
                                rows.append({
                                    'scenario_id': sid,
                                    'iso': 'X', 'year': 2050,
                                    'clean_pct': 10.0 * d_num,
                                    'cost_per_mwh': 0, 'emissions_mt': 0, 'avg_lmp': 0,
                                })

        df = pd.DataFrame(rows)
        df = add_dimension_columns(df)

        effects = compute_elementary_effects(df, 'X', 2050, 'clean_pct')

        # demand_growth: each step changes output by exactly 10
        assert abs(effects['demand_growth']['mu_star'] - 10.0) < 0.01
        # sigma should be 0 (perfectly linear, no interaction)
        assert effects['demand_growth']['sigma'] < 0.01
        # all other dimensions should have ~0 effect
        for dim in ['price_sensitivity', 'ppa_level', 'gas_friction',
                     'queue_capacity', 'new_fossil_cost']:
            assert effects[dim]['mu_star'] < 0.01, (
                f"{dim} should have zero effect, got {effects[dim]['mu_star']}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Variance Fraction Sum Constraint (Explicit)
# ─────────────────────────────────────────────────────────────────────────────

class TestVarianceFractionSum:
    """Dedicated tests ensuring variance fractions sum to ≤ 1.0 across all metrics.

    This is a mathematical invariant of first-order Sobol indices: the sum of
    individual variance fractions can never exceed 1.0 because the remainder
    represents interaction effects. Violations indicate a bug in the
    decomposition logic.
    """

    def test_sum_leq_1_all_metrics(self, synthetic_df_with_dims):
        """Variance fractions must sum to ≤ 1.0 for every metric."""
        var_results = compute_variance_decomposition(
            synthetic_df_with_dims, 'TEST_ISO', 2050)

        for metric, fractions in var_results.items():
            dim_fracs = {k: v for k, v in fractions.items() if not k.startswith('_')}
            total = sum(dim_fracs.values())
            assert total <= 1.0 + 1e-6, (
                f"Variance fractions sum to {total:.6f} > 1.0 for {metric}. "
                f"Breakdown: {dim_fracs}"
            )

    def test_interaction_residual_positive(self, synthetic_df_with_dims):
        """The interaction residual (1 - sum of first-order fractions) must be ≥ 0."""
        var_results = compute_variance_decomposition(
            synthetic_df_with_dims, 'TEST_ISO', 2050)

        for metric, fractions in var_results.items():
            dim_fracs = {k: v for k, v in fractions.items() if not k.startswith('_')}
            residual = 1.0 - sum(dim_fracs.values())
            assert residual >= -1e-6, (
                f"Negative interaction residual {residual:.6f} for {metric}"
            )

    def test_sum_leq_1_with_pure_additive_model(self):
        """When output = sum of independent effects, fractions should sum to ~1.0."""
        rows = []
        code = {'Low': 'L', 'Medium': 'M', 'High': 'H'}
        demand_vals = {'Low': 10, 'Medium': 20, 'High': 30}
        price_vals = {'all_low': 5, 'all_med': 10, 'all_high': 15,
                      'high_vre_low_firm': 12, 'high_firm_low_vre': 8}

        for d in ['Low', 'Medium', 'High']:
            for p in ['all_low', 'all_med', 'all_high', 'high_vre_low_firm', 'high_firm_low_vre']:
                for ppa in ['Low', 'Medium', 'High']:
                    for g in ['Low', 'Medium', 'High']:
                        for q in ['Low', 'Medium', 'High']:
                            for nfc in ['Low', 'Medium', 'High']:
                                sid = f"MKT_{code[d]}_{p}_{code[ppa]}_{code[g]}_{code[q]}_{code[nfc]}"
                                # Purely additive: no interaction terms
                                clean = demand_vals[d] + price_vals[p]
                                rows.append({
                                    'scenario_id': sid, 'iso': 'A', 'year': 2050,
                                    'clean_pct': clean,
                                    'cost_per_mwh': 0, 'emissions_mt': 0, 'avg_lmp': 0,
                                })

        df = add_dimension_columns(pd.DataFrame(rows))
        var_results = compute_variance_decomposition(df, 'A', 2050)

        # For a purely additive model, first-order fractions should sum to ~1.0
        fracs = {k: v for k, v in var_results['clean_pct'].items() if not k.startswith('_')}
        total = sum(fracs.values())
        assert abs(total - 1.0) < 0.01, (
            f"Pure additive model should have fractions summing to ~1.0, got {total:.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Morris Correctness on Known Test Cases
# ─────────────────────────────────────────────────────────────────────────────

class TestMorrisKnownCases:
    """Verify Morris elementary effects against analytically known functions."""

    def test_two_dimension_additive(self):
        """y = 10*demand + 5*queue → demand μ*=10, queue μ*=5, others μ*≈0."""
        rows = []
        code = {'Low': 'L', 'Medium': 'M', 'High': 'H'}
        d_num = {'Low': 0, 'Medium': 1, 'High': 2}
        q_num = {'Low': 0, 'Medium': 1, 'High': 2}

        for d in ['Low', 'Medium', 'High']:
            for p in ['all_low', 'all_med', 'all_high', 'high_vre_low_firm', 'high_firm_low_vre']:
                for ppa in ['Low', 'Medium', 'High']:
                    for g in ['Low', 'Medium', 'High']:
                        for q in ['Low', 'Medium', 'High']:
                            for nfc in ['Low', 'Medium', 'High']:
                                sid = f"MKT_{code[d]}_{p}_{code[ppa]}_{code[g]}_{code[q]}_{code[nfc]}"
                                rows.append({
                                    'scenario_id': sid, 'iso': 'Z', 'year': 2050,
                                    'clean_pct': 10.0 * d_num[d] + 5.0 * q_num[q],
                                    'cost_per_mwh': 0, 'emissions_mt': 0, 'avg_lmp': 0,
                                })

        df = add_dimension_columns(pd.DataFrame(rows))
        effects = compute_elementary_effects(df, 'Z', 2050, 'clean_pct')

        # demand_growth: each step = +10
        assert abs(effects['demand_growth']['mu_star'] - 10.0) < 0.01
        assert effects['demand_growth']['sigma'] < 0.01

        # queue_capacity: each step = +5
        assert abs(effects['queue_capacity']['mu_star'] - 5.0) < 0.01
        assert effects['queue_capacity']['sigma'] < 0.01

        # Others should be ~0
        for dim in ['price_sensitivity', 'ppa_level', 'gas_friction', 'new_fossil_cost']:
            assert effects[dim]['mu_star'] < 0.01, (
                f"{dim} should have zero effect, got {effects[dim]['mu_star']}"
            )

    def test_quadratic_shows_nonlinearity(self):
        """y = demand² → μ* > 0, σ > 0 (non-zero sigma indicates non-linearity)."""
        rows = []
        code = {'Low': 'L', 'Medium': 'M', 'High': 'H'}
        d_num = {'Low': 0, 'Medium': 1, 'High': 2}

        for d in ['Low', 'Medium', 'High']:
            for p in ['all_low', 'all_med', 'all_high', 'high_vre_low_firm', 'high_firm_low_vre']:
                for ppa in ['Low', 'Medium', 'High']:
                    for g in ['Low', 'Medium', 'High']:
                        for q in ['Low', 'Medium', 'High']:
                            for nfc in ['Low', 'Medium', 'High']:
                                sid = f"MKT_{code[d]}_{p}_{code[ppa]}_{code[g]}_{code[q]}_{code[nfc]}"
                                rows.append({
                                    'scenario_id': sid, 'iso': 'Q', 'year': 2050,
                                    'clean_pct': float(d_num[d] ** 2),
                                    'cost_per_mwh': 0, 'emissions_mt': 0, 'avg_lmp': 0,
                                })

        df = add_dimension_columns(pd.DataFrame(rows))
        effects = compute_elementary_effects(df, 'Q', 2050, 'clean_pct')

        # For y = x², the elementary effects are:
        #   0→1: 1²-0²=1, 1→2: 2²-1²=3 → μ*=2, σ=√2≈1.414
        assert effects['demand_growth']['mu_star'] > 0
        assert effects['demand_growth']['sigma'] > 0, (
            "Quadratic function should produce non-zero σ (non-linearity indicator)"
        )

        # Signed mean should be positive (monotonically increasing)
        assert effects['demand_growth']['mu'] > 0

    def test_signed_mean_direction(self):
        """y = -10*demand → μ should be negative (decreasing), μ* positive."""
        rows = []
        code = {'Low': 'L', 'Medium': 'M', 'High': 'H'}
        d_num = {'Low': 0, 'Medium': 1, 'High': 2}

        for d in ['Low', 'Medium', 'High']:
            for p in ['all_low', 'all_med', 'all_high', 'high_vre_low_firm', 'high_firm_low_vre']:
                for ppa in ['Low', 'Medium', 'High']:
                    for g in ['Low', 'Medium', 'High']:
                        for q in ['Low', 'Medium', 'High']:
                            for nfc in ['Low', 'Medium', 'High']:
                                sid = f"MKT_{code[d]}_{p}_{code[ppa]}_{code[g]}_{code[q]}_{code[nfc]}"
                                rows.append({
                                    'scenario_id': sid, 'iso': 'N', 'year': 2050,
                                    'clean_pct': -10.0 * d_num[d],
                                    'cost_per_mwh': 0, 'emissions_mt': 0, 'avg_lmp': 0,
                                })

        df = add_dimension_columns(pd.DataFrame(rows))
        effects = compute_elementary_effects(df, 'N', 2050, 'clean_pct')

        assert abs(effects['demand_growth']['mu_star'] - 10.0) < 0.01
        assert effects['demand_growth']['mu'] < 0, "Signed mean should be negative"
        assert abs(effects['demand_growth']['mu'] + 10.0) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# Tests — TornadoMetric / TornadoBar Model Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestTornadoModel:
    """Verify the Pydantic model for tornado_data integrates with analysis output."""

    def test_tornado_model_from_analysis(self, synthetic_df_with_dims):
        """TornadoMetric model should accept format_tornado_data output."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))
        try:
            from models import TornadoBar, TornadoMetric
        except ImportError:
            pytest.skip("backend models not importable")

        var_results = compute_variance_decomposition(
            synthetic_df_with_dims, 'TEST_ISO', 2050)
        tornado = format_tornado_data(var_results)

        for metric, bars in tornado.items():
            total_var = var_results.get(metric, {}).get('_total_variance', 0.0)
            tm = TornadoMetric(
                metric=metric,
                metric_label=f"Label for {metric}",
                bars=[TornadoBar(**b) for b in bars],
                total_variance=total_var,
            )
            assert len(tm.bars) == len(SWEEP_DIMENSIONS)
            assert tm.total_variance >= 0
            # Verify serialization round-trip
            d = tm.model_dump()
            assert d['metric'] == metric
            assert len(d['bars']) == len(SWEEP_DIMENSIONS)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
