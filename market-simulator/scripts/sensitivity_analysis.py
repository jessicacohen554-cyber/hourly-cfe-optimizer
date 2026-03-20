#!/usr/bin/env python3
"""G7: Sensitivity Analysis Framework — Morris Method Screening & Variance Decomposition.

Post-processing module that decomposes output variance across the 1,215-scenario
parametric sweep. No new simulations needed — reads cached sweep results from parquet.

Usage:
    cd market-simulator/scripts
    python sensitivity_analysis.py [--parquet ../results/sweep_1215/sweep_1215_flat.parquet]
                                   [--output-dir ../results/sensitivity]
                                   [--year 2050]

Produces per-ISO JSON files with:
    - Morris method elementary effects (mean & std per input dimension)
    - Morris plot data (for Chart.js scatter visualization)
    - First-order variance decomposition (tornado diagram data)
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Constants — sweep dimension definitions
# ─────────────────────────────────────────────────────────────────────────────

# The 6 independent input dimensions of the 1,215-scenario Cartesian sweep.
# Each dimension is encoded in the scenario_id as single-letter codes.
# Format: MKT_{demand}_{price_sens}_{ppa}_{gas}_{queue}_{nfc}

SWEEP_DIMENSIONS = {
    'demand_growth': {
        'levels': ['Low', 'Medium', 'High'],
        'codes': ['L', 'M', 'H'],       # position 1 in scenario_id
        'label': 'Demand Growth',
        'numeric_map': {'Low': 0, 'Medium': 1, 'High': 2},
    },
    'price_sensitivity': {
        'levels': ['all_low', 'all_med', 'all_high', 'high_vre_low_firm', 'high_firm_low_vre'],
        'codes': ['all_low', 'all_med', 'all_high', 'high_vre_low_firm', 'high_firm_low_vre'],
        'label': 'Price Sensitivity (Renewable/Firm LCOE)',
        'numeric_map': {'all_low': 0, 'all_med': 1, 'all_high': 2,
                        'high_vre_low_firm': 3, 'high_firm_low_vre': 4},
    },
    'ppa_level': {
        'levels': ['Low', 'Medium', 'High'],
        'codes': ['L', 'M', 'H'],       # position 3
        'label': 'PPA Premium Level',
        'numeric_map': {'Low': 0, 'Medium': 1, 'High': 2},
    },
    'gas_friction': {
        'levels': ['Low', 'Medium', 'High'],
        'codes': ['L', 'M', 'H'],       # position 4
        'label': 'Gas Friction (Permitting/Pipeline)',
        'numeric_map': {'Low': 0, 'Medium': 1, 'High': 2},
    },
    'queue_capacity': {
        'levels': ['Low', 'Medium', 'High'],
        'codes': ['L', 'M', 'H'],       # position 5
        'label': 'Interconnection Queue Capacity',
        'numeric_map': {'Low': 0, 'Medium': 1, 'High': 2},
    },
    'new_fossil_cost': {
        'levels': ['Low', 'Medium', 'High'],
        'codes': ['L', 'M', 'H'],       # position 6
        'label': 'New Fossil Build Cost',
        'numeric_map': {'Low': 0, 'Medium': 1, 'High': 2},
    },
}

# Output metrics to analyze
OUTPUT_METRICS = ['clean_pct', 'cost_per_mwh', 'emissions_mt', 'avg_lmp']

METRIC_LABELS = {
    'clean_pct': 'Clean Energy %',
    'cost_per_mwh': 'System Cost ($/MWh)',
    'emissions_mt': 'CO₂ Emissions (Mt)',
    'avg_lmp': 'Average LMP ($/MWh)',
}

ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']


# ─────────────────────────────────────────────────────────────────────────────
# Scenario ID parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_scenario_id(scenario_id: str) -> Dict[str, str]:
    """Parse a scenario_id string into its dimension values.

    Format: MKT_{demand}_{price_sens}_{ppa}_{gas}_{queue}_{nfc}
    Examples:
        MKT_L_all_low_L_L_L_L  → demand=L, price=all_low, ppa=L, gas=L, queue=L, nfc=L
        MKT_M_high_vre_low_firm_H_M_L_H → demand=M, price=high_vre_low_firm, ppa=H, ...
    """
    # Remove MKT_ prefix
    rest = scenario_id[4:]  # strip "MKT_"

    # demand is the first char
    demand_code = rest[0]
    rest = rest[2:]  # skip demand + underscore

    # price_sensitivity is the tricky part — it can contain underscores
    # Known price sensitivity names (sorted longest-first for greedy match)
    price_names = sorted(SWEEP_DIMENSIONS['price_sensitivity']['levels'],
                         key=len, reverse=True)
    price_sens = None
    for pn in price_names:
        if rest.startswith(pn):
            price_sens = pn
            rest = rest[len(pn) + 1:]  # skip name + underscore
            break

    if price_sens is None:
        raise ValueError(f"Cannot parse price_sensitivity from scenario_id: {scenario_id}")

    # Remaining: {ppa}_{gas}_{queue}_{nfc} — all single chars separated by underscores
    parts = rest.split('_')
    if len(parts) != 4:
        raise ValueError(f"Expected 4 trailing parts, got {len(parts)} in: {scenario_id}")

    ppa_code, gas_code, queue_code, nfc_code = parts

    # Map codes back to level names
    code_to_level = {'L': 'Low', 'M': 'Medium', 'H': 'High'}

    return {
        'demand_growth': code_to_level.get(demand_code, demand_code),
        'price_sensitivity': price_sens,
        'ppa_level': code_to_level.get(ppa_code, ppa_code),
        'gas_friction': code_to_level.get(gas_code, gas_code),
        'queue_capacity': code_to_level.get(queue_code, queue_code),
        'new_fossil_cost': code_to_level.get(nfc_code, nfc_code),
    }


def add_dimension_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parse scenario_id into separate dimension columns + numeric encodings."""
    parsed = df['scenario_id'].apply(parse_scenario_id)
    parsed_df = pd.DataFrame(parsed.tolist(), index=df.index)

    for dim_name, dim_info in SWEEP_DIMENSIONS.items():
        df[dim_name] = parsed_df[dim_name]
        df[f'{dim_name}_num'] = parsed_df[dim_name].map(dim_info['numeric_map'])

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Morris Method — Elementary Effects
# ─────────────────────────────────────────────────────────────────────────────

def compute_elementary_effects(
    df: pd.DataFrame,
    iso: str,
    year: int,
    metric: str,
) -> Dict[str, Dict[str, float]]:
    """Compute Morris elementary effects for each input dimension.

    For each dimension d, we find pairs of scenarios that differ ONLY in d
    (all other dimensions held constant), compute the change in the output
    metric, and report the mean (μ*) and standard deviation (σ) of these
    elementary effects.

    Returns:
        {dimension_name: {'mu_star': float, 'sigma': float, 'mu': float, 'n_effects': int}}
    """
    subset = df[(df['iso'] == iso) & (df['year'] == year)].copy()
    if subset.empty:
        return {}

    dim_names = list(SWEEP_DIMENSIONS.keys())
    results = {}

    for target_dim in dim_names:
        # Group by all dimensions EXCEPT the target
        other_dims = [d for d in dim_names if d != target_dim]
        target_levels = SWEEP_DIMENSIONS[target_dim]['levels']
        numeric_map = SWEEP_DIMENSIONS[target_dim]['numeric_map']

        effects = []

        # Group by the "held constant" dimensions
        grouped = subset.groupby(other_dims)

        for _group_key, group in grouped:
            # Within this group, only the target dimension varies
            # Sort by the target dimension's numeric value
            group = group.sort_values(f'{target_dim}_num')

            # Get unique levels present in this group
            levels_present = group[target_dim].unique()
            if len(levels_present) < 2:
                continue

            # Compute elementary effects between adjacent levels
            level_values = group.groupby(target_dim)[metric].mean()

            for i in range(len(target_levels) - 1):
                l_lo = target_levels[i]
                l_hi = target_levels[i + 1]
                if l_lo in level_values.index and l_hi in level_values.index:
                    delta = level_values[l_hi] - level_values[l_lo]
                    effects.append(delta)

        effects = np.array(effects, dtype=np.float64)
        if len(effects) == 0:
            results[target_dim] = {
                'mu_star': 0.0, 'sigma': 0.0, 'mu': 0.0, 'n_effects': 0,
            }
            continue

        # μ* = mean of absolute elementary effects (Morris's improved measure)
        mu_star = float(np.mean(np.abs(effects)))
        # σ = std of elementary effects (indicates non-linearity / interactions)
        sigma = float(np.std(effects, ddof=1)) if len(effects) > 1 else 0.0
        # μ = mean of signed effects (net direction)
        mu = float(np.mean(effects))

        results[target_dim] = {
            'mu_star': round(mu_star, 6),
            'sigma': round(sigma, 6),
            'mu': round(mu, 6),
            'n_effects': len(effects),
        }

    return results


def compute_morris_all_metrics(
    df: pd.DataFrame,
    iso: str,
    year: int,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute Morris elementary effects for all output metrics.

    Returns:
        {metric: {dimension: {mu_star, sigma, mu, n_effects}}}
    """
    results = {}
    for metric in OUTPUT_METRICS:
        if metric not in df.columns:
            continue
        results[metric] = compute_elementary_effects(df, iso, year, metric)
    return results


def format_morris_plot_data(
    morris_results: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, List[Dict]]:
    """Format Morris results as Chart.js scatter plot data.

    Each metric gets a list of {x: mu_star, y: sigma, label: dimension_name} points.
    Quadrant interpretation:
        - High mu_star + high sigma → non-linear, important
        - High mu_star + low sigma → linear, important
        - Low mu_star + low sigma → unimportant
        - Low mu_star + high sigma → interaction effects only
    """
    plot_data = {}
    for metric, dim_effects in morris_results.items():
        points = []
        for dim_name, effects in dim_effects.items():
            points.append({
                'x': effects['mu_star'],
                'y': effects['sigma'],
                'label': SWEEP_DIMENSIONS[dim_name]['label'],
                'dimension': dim_name,
                'mu': effects['mu'],
                'n_effects': effects['n_effects'],
            })
        plot_data[metric] = points
    return plot_data


# ─────────────────────────────────────────────────────────────────────────────
# First-Order Variance Decomposition (One-at-a-Time)
# ─────────────────────────────────────────────────────────────────────────────

def compute_variance_decomposition(
    df: pd.DataFrame,
    iso: str,
    year: int,
) -> Dict[str, Dict[str, float]]:
    """Compute first-order variance decomposition for each output metric.

    For each input dimension d, computes:
        V_d = Var(E[Y | X_d]) / Var(Y)

    This is the fraction of total output variance explained by dimension d alone
    (i.e., the first-order Sobol index approximation via ANOVA decomposition).

    Returns:
        {metric: {dimension: variance_fraction, ..., '_total_variance': float}}
    """
    subset = df[(df['iso'] == iso) & (df['year'] == year)].copy()
    if subset.empty:
        return {}

    dim_names = list(SWEEP_DIMENSIONS.keys())
    results = {}

    for metric in OUTPUT_METRICS:
        if metric not in subset.columns:
            continue

        y = subset[metric].dropna()
        if len(y) < 2:
            continue

        total_var = float(y.var(ddof=1))
        if total_var < 1e-12:
            results[metric] = {d: 0.0 for d in dim_names}
            results[metric]['_total_variance'] = 0.0
            continue

        fractions = {}
        for dim_name in dim_names:
            # E[Y | X_d] — conditional mean for each level of dimension d
            conditional_means = subset.groupby(dim_name)[metric].mean()
            # Var(E[Y | X_d]) — variance of the conditional means, weighted by
            # the number of observations at each level
            level_counts = subset.groupby(dim_name)[metric].count()
            weights = level_counts / level_counts.sum()
            grand_mean = float(y.mean())

            # Weighted variance of conditional means
            var_conditional = float(
                np.sum(weights * (conditional_means - grand_mean) ** 2)
            )

            fractions[dim_name] = round(var_conditional / total_var, 6)

        fractions['_total_variance'] = round(total_var, 4)
        results[metric] = fractions

    return results


def format_tornado_data(
    variance_results: Dict[str, Dict[str, float]],
) -> Dict[str, List[Dict]]:
    """Format variance decomposition as tornado diagram data.

    Returns sorted bars per metric: [{dimension, label, variance_fraction}]
    sorted by variance_fraction descending.
    """
    tornado = {}
    for metric, fractions in variance_results.items():
        bars = []
        for dim_name in SWEEP_DIMENSIONS:
            frac = fractions.get(dim_name, 0.0)
            bars.append({
                'dimension': dim_name,
                'label': SWEEP_DIMENSIONS[dim_name]['label'],
                'variance_fraction': frac,
            })
        # Sort descending by importance
        bars.sort(key=lambda b: b['variance_fraction'], reverse=True)
        tornado[metric] = bars
    return tornado


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate analysis — range impact per dimension
# ─────────────────────────────────────────────────────────────────────────────

def compute_range_impact(
    df: pd.DataFrame,
    iso: str,
    year: int,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute the range of output metric values across each dimension's levels.

    Useful for tornado diagrams showing absolute impact (min→max swing)
    rather than variance fraction.

    Returns:
        {metric: {dimension: {low: float, high: float, range: float, baseline: float}}}
    """
    subset = df[(df['iso'] == iso) & (df['year'] == year)].copy()
    if subset.empty:
        return {}

    results = {}
    for metric in OUTPUT_METRICS:
        if metric not in subset.columns:
            continue

        baseline = float(subset[metric].mean())
        dim_ranges = {}

        for dim_name in SWEEP_DIMENSIONS:
            cond_means = subset.groupby(dim_name)[metric].mean()
            low = float(cond_means.min())
            high = float(cond_means.max())
            dim_ranges[dim_name] = {
                'low': round(low, 4),
                'high': round(high, 4),
                'range': round(high - low, 4),
                'baseline': round(baseline, 4),
                'label': SWEEP_DIMENSIONS[dim_name]['label'],
            }

        results[metric] = dim_ranges
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_sensitivity_analysis(
    parquet_path: str,
    year: int = 2050,
    isos: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """Run the full sensitivity analysis pipeline.

    Args:
        parquet_path: Path to sweep_1215_flat.parquet
        year: Analysis year (default: 2050, the terminal projection year)
        isos: List of ISOs to analyze (default: all 7)

    Returns:
        {iso: {
            'morris': {metric: {dim: {mu_star, sigma, mu}}},
            'morris_plot': {metric: [{x, y, label, dimension}]},
            'variance_decomposition': {metric: {dim: fraction}},
            'tornado': {metric: [{dimension, label, variance_fraction}]},
            'range_impact': {metric: {dim: {low, high, range, baseline}}},
            'metadata': {year, n_scenarios, iso, ...}
        }}
    """
    df = pd.read_parquet(parquet_path)
    df = add_dimension_columns(df)

    target_isos = isos or ISOS
    all_results = {}

    for iso in target_isos:
        iso_subset = df[df['iso'] == iso]
        if iso_subset.empty:
            continue

        n_scenarios = iso_subset[iso_subset['year'] == year]['scenario_id'].nunique()

        # Morris method
        morris = compute_morris_all_metrics(df, iso, year)
        morris_plot = format_morris_plot_data(morris)

        # Variance decomposition
        var_decomp = compute_variance_decomposition(df, iso, year)
        tornado = format_tornado_data(var_decomp)

        # Range impact
        range_impact = compute_range_impact(df, iso, year)

        all_results[iso] = {
            'morris': morris,
            'morris_plot': morris_plot,
            'variance_decomposition': var_decomp,
            'tornado': tornado,
            'range_impact': range_impact,
            'metadata': {
                'iso': iso,
                'year': year,
                'n_scenarios': n_scenarios,
                'n_dimensions': len(SWEEP_DIMENSIONS),
                'output_metrics': OUTPUT_METRICS,
                'metric_labels': METRIC_LABELS,
                'dimension_labels': {k: v['label'] for k, v in SWEEP_DIMENSIONS.items()},
            },
        }

    return all_results


def save_results(
    results: Dict[str, Dict],
    output_dir: str,
) -> List[str]:
    """Save sensitivity analysis results as JSON files.

    Produces:
        - sensitivity_all.json — combined results for all ISOs
        - sensitivity_{ISO}.json — per-ISO results
    """
    os.makedirs(output_dir, exist_ok=True)
    saved = []

    # Combined file
    combined_path = os.path.join(output_dir, 'sensitivity_all.json')
    with open(combined_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    saved.append(combined_path)

    # Per-ISO files
    for iso, iso_data in results.items():
        iso_path = os.path.join(output_dir, f'sensitivity_{iso}.json')
        with open(iso_path, 'w') as f:
            json.dump(iso_data, f, indent=2, default=str)
        saved.append(iso_path)

    return saved


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='G7: Sensitivity Analysis — Morris Method & Variance Decomposition')
    parser.add_argument('--parquet', default=None,
                        help='Path to sweep_1215_flat.parquet')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for JSON results')
    parser.add_argument('--year', type=int, default=2050,
                        help='Analysis year (default: 2050)')
    parser.add_argument('--isos', nargs='+', default=None,
                        help='ISOs to analyze (default: all 7)')
    args = parser.parse_args()

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parquet_path = args.parquet or os.path.join(
        script_dir, '..', 'results', 'sweep_1215', 'sweep_1215_flat.parquet')
    output_dir = args.output_dir or os.path.join(
        script_dir, '..', 'results', 'sensitivity')

    if not os.path.exists(parquet_path):
        print(f"ERROR: Parquet not found at {parquet_path}")
        print("Run the 1,215-scenario sweep first: python run_sweep_405.py")
        sys.exit(1)

    print(f"Loading sweep results from: {parquet_path}")
    print(f"Analysis year: {args.year}")
    print(f"ISOs: {args.isos or 'all 7'}")
    print()

    results = run_sensitivity_analysis(
        parquet_path=parquet_path,
        year=args.year,
        isos=args.isos,
    )

    saved = save_results(results, output_dir)

    # Print summary
    for iso, data in results.items():
        print(f"\n{'='*60}")
        print(f"  {iso} — Sensitivity Analysis (Year {args.year})")
        print(f"{'='*60}")

        # Tornado summary for each metric
        for metric in OUTPUT_METRICS:
            if metric not in data['tornado']:
                continue
            print(f"\n  {METRIC_LABELS.get(metric, metric)}:")
            print(f"  {'Dimension':<45} {'Var. Fraction':>14}")
            print(f"  {'─'*45} {'─'*14}")
            for bar in data['tornado'][metric]:
                pct = bar['variance_fraction'] * 100
                bar_str = '█' * int(pct / 2)
                print(f"  {bar['label']:<45} {pct:>10.1f}%  {bar_str}")

    print(f"\n\nResults saved to: {output_dir}")
    for f in saved:
        print(f"  {f}")


if __name__ == '__main__':
    main()
