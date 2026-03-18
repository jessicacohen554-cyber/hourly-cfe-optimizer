#!/usr/bin/env python3
"""Batch simulation test for LP co-dispatch integration.

Loads real ISO data and Step 2 EF mixes, runs both greedy and LP dispatch
on a sample of mixes with multiple storage types, and compares results.

This validates that:
1. LP dispatch integrates correctly with the full reconstruct_hourly_dispatch pipeline
2. Results are valid (energy conservation, non-negative residuals, etc.)
3. LP produces >= greedy gap reduction on real data
4. Performance is acceptable for batch use
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from dispatch_utils import (
    reconstruct_hourly_dispatch,
    get_supply_profiles,
    build_supply_matrix,
    load_common_data,
    H,
    BATTERY_EFFICIENCY, BATTERY8_EFFICIENCY, LDES_EFFICIENCY, H2_EFFICIENCY,
)
from pipeline_config import ISOS, REGIONAL_DEMAND_TWH


def load_ef_mixes(iso, max_mixes=20):
    """Load a sample of EF mixes from Step 2.1 parquets."""
    data_dir = os.path.join(os.path.dirname(SCRIPT_DIR), 'data', 'step2.1-ef')
    # Try multiple thresholds to find some with storage
    mixes = []
    for thresh in [80, 85, 90, 95, 75, 70]:
        pq_path = os.path.join(data_dir, f'step_2_1_EF_{iso}_{thresh}.parquet')
        if not os.path.exists(pq_path):
            continue
        df = pd.read_parquet(pq_path)
        # Filter to mixes with at least 2 storage types active
        storage_cols = []
        for col in ['battery_pct', 'battery8_pct', 'ldes_pct', 'h2_pct']:
            if col in df.columns:
                storage_cols.append(col)
        if len(storage_cols) >= 2:
            mask = sum(df[c] > 0 for c in storage_cols) >= 2
            multi_storage = df[mask]
            if len(multi_storage) > 0:
                sample = multi_storage.head(max_mixes - len(mixes))
                for _, row in sample.iterrows():
                    mix = {
                        'threshold': thresh,
                        'resource_pcts': {},
                        'battery_pct': row.get('battery_pct', 0) or 0,
                        'battery8_pct': row.get('battery8_pct', 0) or 0,
                        'ldes_pct': row.get('ldes_pct', 0) or 0,
                        'h2_pct': row.get('h2_pct', 0) or 0,
                    }
                    # Extract resource percentages
                    for col in ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro']:
                        if col in row.index:
                            mix['resource_pcts'][col] = row[col]
                    mixes.append(mix)
                    if len(mixes) >= max_mixes:
                        break
        if len(mixes) >= max_mixes:
            break

    return mixes


def run_batch_test(iso='PJM', max_mixes=20):
    """Run batch LP vs greedy comparison on real data."""
    print(f"\n{'='*70}")
    print(f"Batch Simulation Test — {iso}")
    print(f"{'='*70}")

    # Load real data
    print(f"\nLoading {iso} data...")
    demand_data, gen_profiles, emission_rates, fossil_mix = load_common_data()

    # Extract demand profile for ISO
    from dispatch_utils import get_demand_profile
    demand_norm, _ = get_demand_profile(iso, demand_data)
    supply_profiles = get_supply_profiles(iso, gen_profiles)
    supply_matrix = build_supply_matrix(supply_profiles)

    # Load EF mixes
    mixes = load_ef_mixes(iso, max_mixes)
    if not mixes:
        print(f"  No multi-storage EF mixes found for {iso}. Creating synthetic mixes.")
        # Fallback: create synthetic multi-storage mixes
        for i in range(min(10, max_mixes)):
            mixes.append({
                'threshold': 80,
                'resource_pcts': {
                    'clean_firm': 15 + i,
                    'solar': 30 - i,
                    'wind': 25,
                    'offshore_wind': 0,
                    'ccs_ccgt': 5,
                    'hydro': 5,
                },
                'battery_pct': 2 + i * 0.5,
                'battery8_pct': 1 + i * 0.3,
                'ldes_pct': 3 + i * 0.5,
                'h2_pct': 0,
            })

    n_mixes = len(mixes)
    print(f"  Testing {n_mixes} mixes with multi-storage...")

    # Run both modes
    greedy_results = []
    lp_results = []
    greedy_times = []
    lp_times = []

    for i, mix in enumerate(mixes):
        rp = mix['resource_pcts']
        b4 = mix['battery_pct']
        b8 = mix['battery8_pct']
        ldes = mix['ldes_pct']
        h2 = mix['h2_pct']

        # Greedy
        t0 = time.time()
        r_greedy = reconstruct_hourly_dispatch(
            demand_norm, supply_profiles, rp,
            procurement_pct=100,
            battery_dispatch_pct=b4,
            battery8_dispatch_pct=b8,
            ldes_dispatch_pct=ldes,
            supply_matrix=supply_matrix,
            detailed=True,
            h2_dispatch_pct=h2,
            dispatch_mode='greedy',
        )
        greedy_times.append(time.time() - t0)
        greedy_results.append(r_greedy)

        # LP
        t0 = time.time()
        r_lp = reconstruct_hourly_dispatch(
            demand_norm, supply_profiles, rp,
            procurement_pct=100,
            battery_dispatch_pct=b4,
            battery8_dispatch_pct=b8,
            ldes_dispatch_pct=ldes,
            supply_matrix=supply_matrix,
            detailed=True,
            h2_dispatch_pct=h2,
            dispatch_mode='lp',
        )
        lp_times.append(time.time() - t0)
        lp_results.append(r_lp)

        if (i + 1) % 5 == 0:
            print(f"  Processed {i+1}/{n_mixes} mixes...")

    # --- Validate results ---
    print(f"\n--- Validation Results ---")

    n_pass = 0
    n_fail = 0
    improvements = []
    demand_arr = np.array(demand_norm[:H])

    for i in range(n_mixes):
        mix = mixes[i]
        rg = greedy_results[i]
        rl = lp_results[i]
        errors = []

        # 1. Non-negative residual demand
        if np.any(rl['residual_demand'] < -1e-6):
            errors.append(f"Negative residual demand: min={np.min(rl['residual_demand']):.6f}")

        # 2. Energy conservation per storage type
        for name, rte in [('battery4', BATTERY_EFFICIENCY), ('battery8', BATTERY8_EFFICIENCY),
                          ('ldes', LDES_EFFICIENCY), ('h2', H2_EFFICIENCY)]:
            charge = np.sum(rl.get(f'{name}_charge', np.zeros(1)))
            discharge = np.sum(rl[f'{name}_profile'])
            if discharge > charge * rte + 1e-4:
                errors.append(f"{name} violates conservation: discharge={discharge:.4f} > charge*rte={charge*rte:.4f}")

        # 3. LP >= greedy on gap reduction
        greedy_gap = np.sum(rg['residual_demand'])
        lp_gap = np.sum(rl['residual_demand'])
        if lp_gap > greedy_gap + 1e-3:
            errors.append(f"LP worse than greedy: LP gap={lp_gap:.4f} > greedy={greedy_gap:.4f}")

        # 4. Supply total should be same (storage doesn't affect clean generation)
        supply_diff = np.max(np.abs(rg['supply_total'] - rl['supply_total']))
        if supply_diff > 1e-10:
            errors.append(f"Supply total mismatch: max_diff={supply_diff:.2e}")

        # 5. Reconstruction consistency
        total_clean = rl['total_clean']
        fossil_displaced = rl['fossil_displaced']
        residual = rl['residual_demand']
        recon_err = np.max(np.abs(fossil_displaced + residual - demand_arr))
        if recon_err > 1e-4:
            errors.append(f"Reconstruction error: {recon_err:.2e}")

        if errors:
            n_fail += 1
            print(f"  Mix {i} (thresh={mix['threshold']}) FAILED:")
            for e in errors:
                print(f"    - {e}")
        else:
            n_pass += 1

        improvement = (greedy_gap - lp_gap) / greedy_gap * 100 if greedy_gap > 0 else 0
        improvements.append(improvement)

    # --- Summary ---
    print(f"\n--- Summary ---")
    print(f"  Passed: {n_pass}/{n_mixes}")
    print(f"  Failed: {n_fail}/{n_mixes}")

    avg_improvement = np.mean(improvements)
    max_improvement = np.max(improvements)
    min_improvement = np.min(improvements)
    print(f"\n  Gap reduction (LP vs greedy):")
    print(f"    Mean:  {avg_improvement:.2f}%")
    print(f"    Max:   {max_improvement:.2f}%")
    print(f"    Min:   {min_improvement:.2f}%")

    avg_greedy = np.mean(greedy_times) * 1000
    avg_lp = np.mean(lp_times) * 1000
    total_greedy = sum(greedy_times)
    total_lp = sum(lp_times)
    print(f"\n  Performance:")
    print(f"    Greedy: {avg_greedy:.1f}ms avg, {total_greedy:.2f}s total")
    print(f"    LP:     {avg_lp:.1f}ms avg, {total_lp:.2f}s total")
    print(f"    Ratio:  {avg_lp/avg_greedy:.1f}x" if avg_greedy > 0 else "    Ratio: N/A")

    # Detailed per-mix table
    print(f"\n  Per-mix details:")
    print(f"  {'Mix':>4} {'Thresh':>6} {'B4%':>5} {'B8%':>5} {'LDES%':>6} {'H2%':>5} "
          f"{'Greedy Gap':>11} {'LP Gap':>11} {'Imprv%':>7} {'LP ms':>7}")
    print(f"  {'-'*4} {'-'*6} {'-'*5} {'-'*5} {'-'*6} {'-'*5} "
          f"{'-'*11} {'-'*11} {'-'*7} {'-'*7}")
    for i in range(n_mixes):
        m = mixes[i]
        g_gap = np.sum(greedy_results[i]['residual_demand'])
        l_gap = np.sum(lp_results[i]['residual_demand'])
        print(f"  {i:4d} {m['threshold']:6d} {m['battery_pct']:5.1f} {m['battery8_pct']:5.1f} "
              f"{m['ldes_pct']:6.2f} {m['h2_pct']:5.2f} "
              f"{g_gap:11.2f} {l_gap:11.2f} {improvements[i]:7.2f} {lp_times[i]*1000:7.0f}")

    if n_fail == 0:
        print(f"\n{'='*70}")
        print(f"BATCH TEST PASSED — All {n_mixes} mixes validated successfully")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")
        print(f"BATCH TEST FAILED — {n_fail} mixes had errors")
        print(f"{'='*70}")
        sys.exit(1)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--iso', default='PJM', help='ISO to test')
    parser.add_argument('--max-mixes', type=int, default=20, help='Max mixes to test')
    args = parser.parse_args()

    run_batch_test(iso=args.iso, max_mixes=args.max_mixes)
