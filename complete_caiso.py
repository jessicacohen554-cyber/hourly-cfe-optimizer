#!/usr/bin/env python3
"""Complete CAISO Phase 2 for remaining thresholds (95, 97.5, 99, 100).

Uses the fine storage levels from the interrupted run:
  bat4: 26 levels (0-1.25% in 0.05% steps)
  bat8: 46 levels (0-2.25% in 0.05% steps)
  LDES: standard coarse [0, 0.5, 1.0, 1.5, 2.0, 2.5, 5, 8, 10]
"""
import sys
import os
import time
import pyarrow as pa
import pyarrow.parquet as pq

# Patch THRESHOLDS before importing step1
REMAINING = [95, 97.5, 99, 100]

# Import the optimizer module
import step1_pfs_generator as s1

# Override thresholds to only run the missing ones
s1.THRESHOLDS = REMAINING

# Fine storage levels matching the interrupted CAISO Phase 2
fine_bat4 = [0] + [round(x * 0.05, 2) for x in range(1, int(1.25 / 0.05) + 1)]
fine_bat8 = [0] + [round(x * 0.05, 2) for x in range(1, int(2.25 / 0.05) + 1)]
fine_ldes = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 5, 8, 10]
fine_levels = {'bat4': fine_bat4, 'bat8': fine_bat8, 'ldes': fine_ldes}

print(f"Completing CAISO Phase 2: thresholds {REMAINING}")
print(f"  bat4: {len(fine_bat4)} levels (0-{fine_bat4[-1]:.2f}%)")
print(f"  bat8: {len(fine_bat8)} levels (0-{fine_bat8[-1]:.2f}%)")
print(f"  ldes: {len(fine_ldes)} levels")

# Load data
demand_data, gen_profiles, emission_rates, fossil_mix = s1.load_data()

# JIT warmup
if s1.HAS_NUMBA:
    import numpy as np
    print("  Warming up Numba JIT...")
    H = s1.H
    dd = np.ones(H) / H
    ds = np.ones(H) / H
    ds2 = np.ones((2, H)) / H
    s1._score_with_all_storage(dd, ds, 1.0, 0.01, 0.0025, 0.85,
                                0.01, 0.00125, 0.85, 0.01, 0.0001, 0.50, 168)
    s1._batch_score_no_storage(dd, ds2, 1.0, 2)
    s1._batch_score_storage(dd, ds2, 1.0, 2,
                             0.01, 0.0025, 0.85, 0.01, 0.00125, 0.85,
                             0.01, 0.0001, 0.50, 168)
    s1._compute_storage_caps(dd, ds, 1.0, 4, 8, 100)
    dl = np.array([0.0, 0.1], dtype=np.float64)
    s1._batch_storage_scores(dd, ds, 1.0, dl, dl, dl,
                              2, 2, 2, 0.85, 0.85, 0.50, 4, 8, 100, 168, 48)
    s1._batch_mixes_storage_screen(dd, ds2, 1.0, 2,
                                     dl, dl, dl,
                                     2, 2, 2, 0.85, 0.85, 0.50, 4, 8, 100, 168, 48)
    print("  JIT ready")

# Clear only the progress/done checkpoints for remaining thresholds
for t in REMAINING:
    for path in [s1._threshold_done_path('CAISO', t),
                 s1._mix_progress_path('CAISO', t)]:
        if os.path.exists(path):
            os.remove(path)

# Monkeypatch process_iso to skip Phase 1 and use our fine levels directly
# We'll call the internal optimize_threshold directly
iso = 'CAISO'
demand_norm = demand_data[iso]['normalized']
supply_profiles = s1.get_supply_profiles(iso, gen_profiles)
demand_arr, supply_matrix = s1.prepare_numpy_profiles(demand_norm, supply_profiles)
hydro_cap = s1.HYDRO_CAPS[iso]

# Load existing solutions from interim for cross-threshold seeding
interim_path = os.path.join(s1.CHECKPOINT_DIR, f'{iso}_v4_interim.parquet')
existing_solutions = []
if os.path.exists(interim_path):
    t = pq.read_table(interim_path)
    df = t.to_pandas()
    # Build solution dicts for seeding
    for _, row in df.iterrows():
        existing_solutions.append({
            'resource_mix': {
                'clean_firm': row['clean_firm'],
                'solar': row['solar'],
                'wind': row['wind'],
                'hydro': row['hydro'],
            },
            'procurement_pct': row['procurement_pct'],
            'battery_dispatch_pct': row['battery_dispatch_pct'],
            'battery8_dispatch_pct': row.get('battery8_dispatch_pct', 0),
            'ldes_dispatch_pct': row['ldes_dispatch_pct'],
            'hourly_match_score': row['hourly_match_score'],
        })
    print(f"  Loaded {len(existing_solutions):,} existing solutions for seeding")

start = time.time()
prev_pruning = None

for threshold in REMAINING:
    t_start = time.time()

    # Build cross_feasible from existing solutions at this or lower thresholds
    cross_feasible = set()
    for sol in existing_solutions:
        mk = (sol['resource_mix']['clean_firm'], sol['resource_mix']['solar'],
              sol['resource_mix']['wind'], sol['resource_mix']['hydro'])
        cross_feasible.add(mk)

    candidates, prev_pruning = s1.optimize_threshold(
        iso, threshold, demand_arr, supply_matrix, hydro_cap,
        prev_pruning=prev_pruning,
        cross_feasible_mixes=cross_feasible,
        storage_levels=fine_levels,
    )

    n = len(candidates)
    elapsed = time.time() - t_start
    print(f"  CAISO {threshold}%: {n:,} solutions, {elapsed:.1f}s")

    # Save done checkpoint and append to interim
    s1._save_threshold_done(iso, threshold, candidates)
    s1.append_threshold_to_cache(iso, threshold, candidates)

# Merge with existing interim
print("\nMerging completion results with existing interim...")
tables = []
if os.path.exists(interim_path):
    tables.append(pq.read_table(interim_path))

for t in REMAINING:
    done_path = s1._threshold_done_path(iso, t)
    if os.path.exists(done_path):
        tables.append(pq.read_table(done_path))

if tables:
    merged = pa.concat_tables(tables)
    merged = s1._dedup_parquet_table(merged)

    # Save as per-ISO cache
    cache_path = os.path.join(s1.DATA_DIR, f'physics_cache_v4_{iso}.parquet')
    pq.write_table(merged, cache_path, compression='snappy')
    size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    print(f"  Cache: {cache_path} ({merged.num_rows:,} solutions, {size_mb:.1f} MB)")

    # Dashboard copy
    dash_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'dashboard', f'physics_results_v4_{iso}.parquet')
    pq.write_table(merged, dash_path, compression='snappy')
    print(f"  Dashboard: {dash_path}")

total_time = time.time() - start
print(f"\nDone in {total_time:.1f}s")
