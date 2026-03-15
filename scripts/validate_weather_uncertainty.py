#!/usr/bin/env python3
"""
validate_weather_uncertainty.py — Re-score efficient frontier mixes against
individual weather years (2021–2025) to quantify weather-driven uncertainty.

The main pipeline uses 5-year element-wise averaged profiles. This script
loads each year's profiles independently and re-scores the ~5.87M EF mixes
to measure how hourly match scores shift across weather years.

Output: data/validation/weather_year_uncertainty.json
  Per-ISO, per-threshold: P10/P50/P90 score deltas, worst-year analysis,
  and summary statistics for the methodology page.

Usage:
    python validate_weather_uncertainty.py
    python validate_weather_uncertainty.py --isos CAISO ERCOT
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq

# Add scripts dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from step1_pfs_generator import (
    ISOS, H, PROFILE_YEARS, DATA_YEAR,
    get_resource_types, get_supply_profiles, prepare_numpy_profiles,
    batch_hourly_scores, load_data, _average_profiles, _remove_leap_day,
    NUCLEAR_MONTHLY_CF, OFFSHORE_ISOS,
)
from eia_data_io import load_generation_profiles, load_demand_profiles

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
EF_DIR = os.path.join(DATA_DIR, 'step2.1-ef')
OUTPUT_DIR = os.path.join(DATA_DIR, 'validation')

# Thresholds to bucket results into (match pipeline thresholds)
THRESHOLDS = [50, 55, 60, 65, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 99.5, 99.9, 99.99]


def load_ef_mixes(iso):
    """Load all EF mixes for an ISO from step2.1-ef parquets.

    Returns (mix_matrix, scores, n_mixes) where:
      mix_matrix: (N, n_resources) int16 array of resource fractions
      scores: (N,) float64 array of baseline hourly match scores
    """
    rtypes = get_resource_types(iso)
    pattern = os.path.join(EF_DIR, f'step_2_1_EF_{iso}_*.parquet')
    files = sorted(glob.glob(pattern))
    if not files:
        return None, None

    tables = []
    for f in files:
        tables.append(pq.read_table(f))

    import pyarrow as pa
    combined = pa.concat_tables(tables, promote_options='permissive')
    n = combined.num_rows

    # Extract resource columns as numpy arrays
    cols = []
    for rt in rtypes:
        arr = combined.column(rt).to_numpy()
        cols.append(arr.astype(np.float64))
    mix_matrix = np.column_stack(cols)

    # Baseline scores
    scores = combined.column('hourly_match_score').to_numpy().astype(np.float64)

    return mix_matrix, scores


def build_year_profiles(iso, year_str, gen_raw, demand_raw):
    """Build supply profiles and demand array for a single weather year.

    Mirrors load_data() + get_supply_profiles() + prepare_numpy_profiles()
    but for one specific year instead of the 5-year average.
    """
    # Build gen_profiles dict for this year (same structure as load_data output)
    iso_gen = gen_raw.get(iso, {})
    year_data = iso_gen.get(year_str, {})
    if not year_data:
        return None, None

    # Create single-year gen_profiles in the format get_supply_profiles expects
    gen_profiles_single = {iso: {}}
    for fuel, raw in year_data.items():
        if len(raw) > H:
            raw = _remove_leap_day(raw)
        else:
            raw = list(raw[:H])
        # Normalize to sum=1.0
        arr = np.array(raw, dtype=np.float64)
        total = arr.sum()
        if total > 0:
            arr /= total
        gen_profiles_single[iso][fuel] = arr.tolist()

    # Build supply profiles using the standard function
    supply_profiles = get_supply_profiles(iso, gen_profiles_single)

    # Build demand profile for this year
    iso_demand = demand_raw.get(iso, {})
    year_demand = iso_demand.get(year_str, {})
    if not year_demand:
        return None, None

    demand_norm = year_demand.get('normalized', [])
    if len(demand_norm) > H:
        demand_norm = _remove_leap_day(demand_norm)
    else:
        demand_norm = list(demand_norm[:H])

    demand_arr, supply_matrix = prepare_numpy_profiles(iso, demand_norm, supply_profiles)
    return demand_arr, supply_matrix


def rescore_ef_for_year(iso, mix_matrix, demand_arr, supply_matrix):
    """Re-score all EF mixes against a single year's profiles.

    Returns scores as percentages (0-100) to match EF parquet convention.
    batch_hourly_scores returns fractions (0-1); multiply by 100.
    """
    raw = batch_hourly_scores(demand_arr, supply_matrix, mix_matrix, chunk_size=20000)
    return raw * 100.0


def compute_threshold_stats(baseline_scores, year_scores_dict, thresholds):
    """Compute per-threshold uncertainty statistics.

    For each threshold, takes all mixes scoring >= threshold under the baseline,
    then computes how their scores shift across individual years.
    """
    results = {}
    for thr in thresholds:
        # Both baseline_scores and year_scores are in percentage units (0-100)
        mask = baseline_scores >= thr
        n_mixes = int(mask.sum())
        if n_mixes == 0:
            results[str(thr)] = {'n_mixes': 0, 'skip': True}
            continue

        base_scores = baseline_scores[mask]
        deltas_by_year = {}
        all_deltas = []

        for year, year_scores in year_scores_dict.items():
            yr_scores = year_scores[mask]
            delta = yr_scores - base_scores  # Already in percentage points
            deltas_by_year[year] = {
                'mean_delta_pp': round(float(delta.mean()), 3),
                'p10_delta_pp': round(float(np.percentile(delta, 10)), 3),
                'p50_delta_pp': round(float(np.percentile(delta, 50)), 3),
                'p90_delta_pp': round(float(np.percentile(delta, 90)), 3),
                'min_delta_pp': round(float(delta.min()), 3),
                'max_delta_pp': round(float(delta.max()), 3),
                'pct_cross_threshold': round(float((yr_scores < thr).sum() / n_mixes * 100), 2),
            }
            all_deltas.append(delta)

        all_deltas_arr = np.concatenate(all_deltas)
        worst_year = min(deltas_by_year, key=lambda y: deltas_by_year[y]['mean_delta_pp'])
        best_year = max(deltas_by_year, key=lambda y: deltas_by_year[y]['mean_delta_pp'])

        results[str(thr)] = {
            'n_mixes': n_mixes,
            'overall_p10_pp': round(float(np.percentile(all_deltas_arr, 10)), 3),
            'overall_p50_pp': round(float(np.percentile(all_deltas_arr, 50)), 3),
            'overall_p90_pp': round(float(np.percentile(all_deltas_arr, 90)), 3),
            'overall_spread_pp': round(float(np.percentile(all_deltas_arr, 90) - np.percentile(all_deltas_arr, 10)), 3),
            'worst_year': worst_year,
            'best_year': best_year,
            'by_year': deltas_by_year,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description='Weather-year uncertainty validation')
    parser.add_argument('--isos', nargs='+', default=ISOS,
                        help='ISOs to process (default: all 7)')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("WEATHER-YEAR UNCERTAINTY VALIDATION")
    print("Re-scoring EF mixes against individual year profiles (2021-2025)")
    print("=" * 70)

    # Load raw per-year data
    print("\nLoading per-year EIA profiles...")
    t0 = time.time()
    gen_raw = load_generation_profiles()
    demand_raw = load_demand_profiles()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Also load the baseline (5-year averaged) profiles for comparison
    print("\nLoading baseline (5-year averaged) profiles...")
    t0 = time.time()
    baseline_demand, baseline_gen, _, _ = load_data()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    all_results = {}
    grand_start = time.time()

    for iso in args.isos:
        iso_start = time.time()
        print(f"\n{'='*50}")
        print(f"Processing {iso}")
        print(f"{'='*50}")

        # Load EF mixes
        mix_matrix, baseline_scores = load_ef_mixes(iso)
        if mix_matrix is None:
            print(f"  No EF data for {iso}, skipping")
            continue

        n_mixes = len(baseline_scores)
        rtypes = get_resource_types(iso)
        print(f"  Loaded {n_mixes:,} EF mixes ({len(rtypes)}D)")

        # Verify baseline scores by re-scoring against averaged profiles
        # batch_hourly_scores returns 0-1 fractions; EF parquet stores 0-100 percentages
        baseline_supply = get_supply_profiles(iso, baseline_gen)
        baseline_demand_arr, baseline_supply_matrix = prepare_numpy_profiles(
            iso, baseline_demand[iso]['normalized'], baseline_supply)
        rescore_baseline = batch_hourly_scores(
            baseline_demand_arr, baseline_supply_matrix, mix_matrix, chunk_size=20000) * 100.0

        # Check baseline consistency
        score_diff = np.abs(rescore_baseline - baseline_scores)
        max_diff = score_diff.max()
        mean_diff = score_diff.mean()
        print(f"  Baseline re-score check: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        if max_diff > 0.01:
            print(f"  WARNING: Large baseline discrepancy — scores may have been computed with different profiles")

        # Use the re-scored baseline (without storage) as reference, so storage
        # offset cancels out and we isolate pure weather-driven variation
        rescore_ref = rescore_baseline

        # Score against each individual year
        year_scores = {}
        for year_str in PROFILE_YEARS:
            t1 = time.time()
            demand_arr, supply_matrix = build_year_profiles(
                iso, year_str, gen_raw, demand_raw)
            if demand_arr is None:
                print(f"  {year_str}: No data, skipping")
                continue

            scores = rescore_ef_for_year(iso, mix_matrix, demand_arr, supply_matrix)
            year_scores[year_str] = scores

            # Compare against re-scored baseline to isolate weather variation
            delta = scores - rescore_ref
            print(f"  {year_str}: mean delta = {delta.mean():+.2f}pp, "
                  f"P10/P90 = [{np.percentile(delta,10):+.2f}, {np.percentile(delta,90):+.2f}]pp "
                  f"({time.time()-t1:.1f}s)")

        # Use re-scored baseline for threshold stats (apples-to-apples without storage)
        threshold_stats = compute_threshold_stats(rescore_ref, year_scores, THRESHOLDS)

        # ISO-level summary
        all_deltas = []
        for yr, sc in year_scores.items():
            all_deltas.append(sc - baseline_scores)  # Already percentage points
        if all_deltas:
            combined = np.concatenate(all_deltas)
            iso_summary = {
                'n_ef_mixes': n_mixes,
                'n_resources': len(rtypes),
                'resource_types': rtypes,
                'overall_p10_pp': round(float(np.percentile(combined, 10)), 3),
                'overall_p50_pp': round(float(np.percentile(combined, 50)), 3),
                'overall_p90_pp': round(float(np.percentile(combined, 90)), 3),
                'overall_range_pp': round(float(combined.max() - combined.min()), 3),
                'years_evaluated': list(year_scores.keys()),
                'baseline_rescore_max_diff': round(float(max_diff), 6),
            }
        else:
            iso_summary = {'n_ef_mixes': n_mixes, 'error': 'No year data'}

        all_results[iso] = {
            'summary': iso_summary,
            'by_threshold': threshold_stats,
        }

        iso_elapsed = time.time() - iso_start
        print(f"  {iso} complete in {iso_elapsed:.0f}s")

    # Grand summary
    print(f"\n{'='*70}")
    print("GRAND SUMMARY")
    print(f"{'='*70}")

    grand_summary = {
        'description': 'Weather-year uncertainty validation: EF mixes re-scored against individual 2021-2025 profiles',
        'baseline': '5-year element-wise average (2021-2025)',
        'years_tested': PROFILE_YEARS,
        'methodology': 'Re-score existing efficient frontier mixes (no storage re-dispatch) against per-year generation and demand profiles',
        'note': 'Scores reflect generation-only matching (no storage dispatch). Actual weather sensitivity may be lower with storage smoothing.',
    }

    for iso in args.isos:
        if iso in all_results:
            s = all_results[iso]['summary']
            if 'error' not in s:
                print(f"  {iso:>6}: {s['n_ef_mixes']:>10,} mixes, "
                      f"P10/P90 = [{s['overall_p10_pp']:+.2f}, {s['overall_p90_pp']:+.2f}]pp")

    total_mixes = sum(all_results[iso]['summary']['n_ef_mixes']
                      for iso in all_results if 'error' not in all_results[iso]['summary'])
    grand_summary['total_ef_mixes_evaluated'] = total_mixes
    grand_summary['total_evaluations'] = total_mixes * len(PROFILE_YEARS)
    grand_summary['runtime_seconds'] = round(time.time() - grand_start, 1)

    output = {
        'meta': grand_summary,
        'results': all_results,
    }

    out_path = os.path.join(OUTPUT_DIR, 'weather_year_uncertainty.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {out_path}")
    print(f"Total runtime: {time.time() - grand_start:.0f}s")


if __name__ == '__main__':
    main()
