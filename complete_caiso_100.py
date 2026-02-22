#!/usr/bin/env python3
"""Complete CAISO: run 100% threshold and merge all results."""
import os, time
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

import step1_pfs_generator as s1

s1.THRESHOLDS = [100]

# Fine storage levels matching Phase 2
fine_bat4 = [0] + [round(x * 0.05, 2) for x in range(1, int(1.25 / 0.05) + 1)]
fine_bat8 = [0] + [round(x * 0.05, 2) for x in range(1, int(2.25 / 0.05) + 1)]
fine_ldes = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 5, 8, 10]
fine_levels = {'bat4': fine_bat4, 'bat8': fine_bat8, 'ldes': fine_ldes}

print("CAISO: Running 100% threshold + merge")

demand_data, gen_profiles, _, _ = s1.load_data()

# JIT warmup
if s1.HAS_NUMBA:
    print("  Warming up JIT...")
    H = s1.H
    dd = np.ones(H) / H; ds = np.ones(H) / H; ds2 = np.ones((2, H)) / H
    s1._score_with_all_storage(dd, ds, 1.0, 0.01, 0.0025, 0.85, 0.01, 0.00125, 0.85, 0.01, 0.0001, 0.50, 168)
    s1._batch_score_no_storage(dd, ds2, 1.0, 2)
    s1._batch_score_storage(dd, ds2, 1.0, 2, 0.01, 0.0025, 0.85, 0.01, 0.00125, 0.85, 0.01, 0.0001, 0.50, 168)
    s1._compute_storage_caps(dd, ds, 1.0, 4, 8, 100)
    dl = np.array([0.0, 0.1], dtype=np.float64)
    s1._batch_storage_scores(dd, ds, 1.0, dl, dl, dl, 2, 2, 2, 0.85, 0.85, 0.50, 4, 8, 100, 168, 48)
    s1._batch_mixes_storage_screen(dd, ds2, 1.0, 2, dl, dl, dl, 2, 2, 2, 0.85, 0.85, 0.50, 4, 8, 100, 168, 48)
    print("  JIT ready")

iso = 'CAISO'
demand_norm = demand_data[iso]['normalized']
supply_profiles = s1.get_supply_profiles(iso, gen_profiles)
demand_arr, supply_matrix = s1.prepare_numpy_profiles(demand_norm, supply_profiles)
hydro_cap = s1.HYDRO_CAPS[iso]

# Clear checkpoints for 100%
for path in [s1._threshold_done_path(iso, 100), s1._mix_progress_path(iso, 100)]:
    if os.path.exists(path):
        os.remove(path)

# Build cross_feasible from interim
interim_path = os.path.join(s1.CHECKPOINT_DIR, f'{iso}_v4_interim.parquet')
cross_feasible = set()
if os.path.exists(interim_path):
    df = pq.read_table(interim_path).to_pandas()
    for _, row in df[['clean_firm', 'solar', 'wind', 'hydro']].drop_duplicates().iterrows():
        cross_feasible.add((row['clean_firm'], row['solar'], row['wind'], row['hydro']))
    print(f"  {len(cross_feasible):,} feasible mixes from cache")

start = time.time()
candidates, _ = s1.optimize_threshold(
    iso, 100, demand_arr, supply_matrix, hydro_cap,
    prev_pruning=None,
    cross_feasible_mixes=cross_feasible,
    storage_levels=fine_levels,
)
print(f"  100%: {len(candidates):,} solutions, {time.time()-start:.1f}s")

s1._save_threshold_done(iso, 100, candidates)
s1.append_threshold_to_cache(iso, 100, candidates)

# Merge ALL thresholds from interim + done parquets
print("\nMerging all CAISO results...")
tables = []
if os.path.exists(interim_path):
    tables.append(pq.read_table(interim_path))

# Also pick up any done parquets not already in interim
ALL_T = [50, 60, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 100]
for t in ALL_T:
    done_path = s1._threshold_done_path(iso, t)
    if os.path.exists(done_path):
        tables.append(pq.read_table(done_path))

merged = pa.concat_tables(tables)
merged = s1._dedup_parquet_table(merged)

cache_path = os.path.join(s1.DATA_DIR, f'physics_cache_v4_{iso}.parquet')
pq.write_table(merged, cache_path, compression='snappy')
size_mb = os.path.getsize(cache_path) / (1024 * 1024)
print(f"  Cache: {cache_path} ({merged.num_rows:,} solutions, {size_mb:.1f} MB)")

dash_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard', f'physics_results_v4_{iso}.parquet')
pq.write_table(merged, dash_path, compression='snappy')
print(f"  Dashboard: {dash_path}")

# Verify all thresholds present
df_final = merged.to_pandas()
print(f"\nFinal verification:")
for t in ALL_T:
    n = len(df_final[df_final.threshold == t])
    print(f"  {t}%: {n:,}")
print(f"  Total: {df_final.shape[0]:,}")
print(f"\nDone in {time.time()-start:.1f}s")
