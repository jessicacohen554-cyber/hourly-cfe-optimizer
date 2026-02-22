#!/usr/bin/env python3
"""Run two-phase adaptive storage sweep for a single ISO.

Usage: python run_single_iso.py <ISO_NAME>
"""
import sys
import os
import time
import numpy as np

iso_name = sys.argv[1] if len(sys.argv) > 1 else None
if not iso_name:
    print("Usage: python run_single_iso.py <ISO_NAME>")
    sys.exit(1)

import step1_pfs_generator as s1

print(f"\n{'='*60}")
print(f"  Running {iso_name} — Two-Phase Adaptive Storage Sweep")
print(f"{'='*60}")

# Load data
demand_data, gen_profiles, emission_rates, fossil_mix = s1.load_data()

if iso_name not in demand_data:
    print(f"ERROR: {iso_name} not found. Available: {list(demand_data.keys())}")
    sys.exit(1)

# JIT warmup
if s1.HAS_NUMBA:
    print("  Warming up Numba JIT...")
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

start = time.time()

# Run the full two-phase process
iso, iso_results = s1.process_iso((iso_name, demand_data, gen_profiles))

# Save final per-ISO cache from interim
import pyarrow.parquet as pq
interim_path = os.path.join(s1.CHECKPOINT_DIR, f'{iso_name}_v4_interim.parquet')
if os.path.exists(interim_path):
    t = pq.read_table(interim_path)

    cache_path = os.path.join(s1.DATA_DIR, f'physics_cache_v4_{iso_name}.parquet')
    pq.write_table(t, cache_path, compression='snappy')
    size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    print(f"\n  Cache: {cache_path} ({t.num_rows:,} solutions, {size_mb:.1f} MB)")

    dash_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard', f'physics_results_v4_{iso_name}.parquet')
    pq.write_table(t, dash_path, compression='snappy')
    print(f"  Dashboard: {dash_path}")

total = time.time() - start
print(f"\n  {iso_name} DONE in {total:.1f}s ({total/60:.1f}min)")
