#!/usr/bin/env python3
"""Complete remaining thresholds for an ISO from its interim cache.

Usage: python complete_iso_thresholds.py <ISO> <threshold1> <threshold2> ...
"""
import sys, os, time
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

iso = sys.argv[1]
remaining = [float(t) for t in sys.argv[2:]]

import step1_pfs_generator as s1
s1.THRESHOLDS = remaining

# Determine fine levels from the ISO's coarse saturation
# Read interim to find max storage usage
interim_path = os.path.join(s1.CHECKPOINT_DIR, f'{iso}_v4_interim.parquet')
df = pq.read_table(interim_path).to_pandas()
max_bat4 = df['battery_dispatch_pct'].max()
max_bat8 = df.get('battery8_dispatch_pct', 0)
if hasattr(max_bat8, 'max'):
    max_bat8 = max_bat8.max()
max_ldes = df['ldes_dispatch_pct'].max()

fine_bat4_max = max_bat4 + 0.25
fine_bat8_max = max_bat8 + 0.25
fine_bat4 = [0] + [round(x * 0.05, 2) for x in range(1, int(fine_bat4_max / 0.05) + 1)]
fine_bat8 = [0] + [round(x * 0.05, 2) for x in range(1, int(fine_bat8_max / 0.05) + 1)]
fine_ldes = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 5, 8, 10]
fine_levels = {'bat4': fine_bat4, 'bat8': fine_bat8, 'ldes': fine_ldes}

print(f"{iso}: Completing thresholds {remaining}")
print(f"  bat4: {len(fine_bat4)} levels (0-{fine_bat4[-1]:.2f}%)")
print(f"  bat8: {len(fine_bat8)} levels (0-{fine_bat8[-1]:.2f}%)")

demand_data, gen_profiles, _, _ = s1.load_data()

# JIT warmup
if s1.HAS_NUMBA:
    print("  JIT warmup...")
    H = s1.H
    dd = np.ones(H)/H; ds = np.ones(H)/H; ds2 = np.ones((2,H))/H
    s1._score_with_all_storage(dd,ds,1.0,0.01,0.0025,0.85,0.01,0.00125,0.85,0.01,0.0001,0.50,168)
    s1._batch_score_no_storage(dd,ds2,1.0,2)
    s1._batch_score_storage(dd,ds2,1.0,2,0.01,0.0025,0.85,0.01,0.00125,0.85,0.01,0.0001,0.50,168)
    s1._compute_storage_caps(dd,ds,1.0,4,8,100)
    dl = np.array([0.0,0.1],dtype=np.float64)
    s1._batch_storage_scores(dd,ds,1.0,dl,dl,dl,2,2,2,0.85,0.85,0.50,4,8,100,168,48)
    s1._batch_mixes_storage_screen(dd,ds2,1.0,2,dl,dl,dl,2,2,2,0.85,0.85,0.50,4,8,100,168,48)
    print("  JIT ready")

demand_norm = demand_data[iso]['normalized']
supply_profiles = s1.get_supply_profiles(iso, gen_profiles)
demand_arr, supply_matrix = s1.prepare_numpy_profiles(demand_norm, supply_profiles)
hydro_cap = s1.HYDRO_CAPS[iso]

# Build cross_feasible from interim (unique mixes only, not full rows)
cross_feasible = set()
for _, row in df[['clean_firm','solar','wind','hydro']].drop_duplicates().iterrows():
    cross_feasible.add((row['clean_firm'], row['solar'], row['wind'], row['hydro']))
print(f"  {len(cross_feasible):,} feasible mixes from cache")
del df  # free memory

# Clear checkpoints for remaining thresholds only
for t in remaining:
    for path in [s1._threshold_done_path(iso, t), s1._mix_progress_path(iso, t)]:
        if os.path.exists(path):
            os.remove(path)

start = time.time()
prev_pruning = None

for threshold in remaining:
    t_start = time.time()
    candidates, prev_pruning = s1.optimize_threshold(
        iso, threshold, demand_arr, supply_matrix, hydro_cap,
        prev_pruning=prev_pruning,
        cross_feasible_mixes=cross_feasible,
        storage_levels=fine_levels,
    )
    elapsed = time.time() - t_start
    print(f"  {iso} {threshold}%: {len(candidates):,} solutions, {elapsed:.1f}s")
    s1._save_threshold_done(iso, threshold, candidates)
    s1.append_threshold_to_cache(iso, threshold, candidates)

# Now merge interim with any new done parquets
print(f"\nSaving {iso} cache...")
tables = [pq.read_table(interim_path)]
for t in remaining:
    done_path = s1._threshold_done_path(iso, t)
    if os.path.exists(done_path):
        tables.append(pq.read_table(done_path))

merged = pa.concat_tables(tables)
merged = s1._dedup_parquet_table(merged)

cache_path = os.path.join(s1.DATA_DIR, f'physics_cache_v4_{iso}.parquet')
pq.write_table(merged, cache_path, compression='snappy')
size_mb = os.path.getsize(cache_path) / (1024*1024)
print(f"  Cache: {cache_path} ({merged.num_rows:,} solutions, {size_mb:.1f} MB)")

dash_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard', f'physics_results_v4_{iso}.parquet')
pq.write_table(merged, dash_path, compression='snappy')
print(f"  Dashboard: {dash_path}")

# Verify
df_final = merged.to_pandas()
for t in sorted(df_final.threshold.unique()):
    print(f"  {t}%: {len(df_final[df_final.threshold==t]):,}")
print(f"  Total: {df_final.shape[0]:,}")
print(f"\nDone in {time.time()-start:.1f}s")
