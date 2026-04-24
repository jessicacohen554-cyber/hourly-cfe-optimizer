#!/usr/bin/env python3
"""Step 2.1b — Augment thin EF cells with perturbation + interpolation.

Runs AFTER Step 2.1 Efficient Frontier, BEFORE Step 2.2 Cost Optimization.

For each ISO/threshold band with fewer than --min-threshold unique mixes,
generates new candidate mixes via:
  1. Perturbation: ±1-2% on each resource dimension from existing seeds
  2. Interpolation: pairwise blends (25/50/75%) of existing mixes
  3. Storage dispatch perturbation: ±0.05% on storage columns

Candidates are scored using the same physics as Step 1:
  - Resource-only mixes: batch_hourly_scores (vectorized numpy)
  - Resource+storage mixes: _score_with_all_storage (Numba JIT)

Deduplication matches Step 2.1 logic: int16 resources, 20x-scaled int32 storage.

Usage:
    python scripts/step2_1b_augment_thin_ef.py [--iso ISO] [--min-threshold N]
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Add scripts dir to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import step1_pfs_generator as s1
from pipeline_config import (
    ISOS, THRESHOLDS, ACTIVE_THRESHOLDS,
    BATTERY_EFFICIENCY, BATTERY_DURATION_HOURS,
    BATTERY8_EFFICIENCY, BATTERY8_DURATION_HOURS,
    LDES_EFFICIENCY, LDES_DURATION_HOURS, LDES_WINDOW_DAYS,
    H2_EFFICIENCY, H2_DURATION_HOURS, H2_WINDOW_DAYS,
)

EF_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'data', 'step2.1-ef')

STORAGE_COLS = ['battery_dispatch_pct', 'battery8_dispatch_pct',
                'ldes_dispatch_pct', 'h2_dispatch_pct']
STORAGE_SCALE = 20  # 0.05% resolution → integer keys

# Perturbation deltas (% of demand)
RESOURCE_DELTAS = np.array([-2, -1, 1, 2], dtype=np.float64)

# Interpolation blend weights
BLEND_WEIGHTS = [0.25, 0.50, 0.75]
MAX_INTERP_PAIRS = 5000

# Storage perturbation step (one dedup step = 0.05%)
STORAGE_DELTA = 0.05


def get_threshold_bounds(threshold):
    """Get [lower, upper) score bounds for a threshold band."""
    sorted_t = sorted(ACTIVE_THRESHOLDS)
    idx = sorted_t.index(threshold)
    lower = threshold
    upper = sorted_t[idx + 1] if idx + 1 < len(sorted_t) else 100.0
    return lower, upper


def get_adjacent_thresholds(threshold):
    """Get adjacent threshold values (one below, one above)."""
    sorted_t = sorted(ACTIVE_THRESHOLDS)
    idx = sorted_t.index(threshold)
    below = sorted_t[idx - 1] if idx > 0 else None
    above = sorted_t[idx + 1] if idx + 1 < len(sorted_t) else None
    return below, above


def load_ef_parquet(iso, threshold):
    """Load an EF parquet file, handling part files."""
    t_str = f'{threshold:g}'
    # Try single file first
    path = os.path.join(EF_DIR, f'step_2_1_EF_{iso}_{t_str}.parquet')
    if os.path.exists(path):
        return pd.read_parquet(path)

    # Try part files
    parts = sorted([
        f for f in os.listdir(EF_DIR)
        if f.startswith(f'step_2_1_EF_{iso}_{t_str}') and f.endswith('.parquet')
    ])
    if parts:
        dfs = [pd.read_parquet(os.path.join(EF_DIR, p)) for p in parts]
        return pd.concat(dfs, ignore_index=True)

    return None


def get_resource_cols(df):
    """Extract resource column names from a dataframe."""
    exclude = {'iso', 'hourly_match_score', 'pareto_type'} | set(STORAGE_COLS)
    return [c for c in df.columns if c not in exclude]


def deduplicate(df, resource_cols):
    """Deduplicate using Step 2.1 logic: int16 resources, 20x storage, keep max score."""
    if len(df) == 0:
        return df

    data = {}
    for col in resource_cols:
        data[col] = df[col].values.astype(np.int32)
    for sc in STORAGE_COLS:
        data[f'_{sc}'] = np.round(df[sc].values * STORAGE_SCALE).astype(np.int32)
    data['_score'] = df['hourly_match_score'].values

    key_df = pd.DataFrame(data)
    group_cols = list(resource_cols) + [f'_{sc}' for sc in STORAGE_COLS]
    keep_idx = key_df.groupby(group_cols, sort=False)['_score'].idxmax().values

    return df.iloc[keep_idx].reset_index(drop=True)


def generate_perturbations(seeds_arr, resource_cols, n_resources):
    """Generate perturbed mixes from seed arrays. Returns (N_new, n_resources) array."""
    n_seeds = len(seeds_arr)
    n_deltas = len(RESOURCE_DELTAS)

    # For each seed, perturb each resource dimension independently
    # Total new candidates: n_seeds × n_resources × n_deltas
    n_total = n_seeds * n_resources * n_deltas
    result = np.tile(seeds_arr, (n_resources * n_deltas, 1))  # (n_total, n_resources)

    for d_idx, delta in enumerate(RESOURCE_DELTAS):
        for r_idx in range(n_resources):
            start = (d_idx * n_resources + r_idx) * n_seeds
            end = start + n_seeds
            result[start:end, r_idx] += delta

    # Clamp to [0, 350]
    np.clip(result, 0, 350, out=result)

    return result


def generate_interpolations(seeds_arr, n_pairs=MAX_INTERP_PAIRS):
    """Generate interpolated mixes from random pairs. Returns (N_new, n_cols) array."""
    n_seeds = len(seeds_arr)
    if n_seeds < 2:
        return np.empty((0, seeds_arr.shape[1]))

    rng = np.random.default_rng(42)
    n_pairs = min(n_pairs, n_seeds * (n_seeds - 1) // 2)

    # Random pair indices
    idx_a = rng.integers(0, n_seeds, size=n_pairs)
    idx_b = rng.integers(0, n_seeds, size=n_pairs)
    # Avoid self-pairs
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]

    results = []
    for w in BLEND_WEIGHTS:
        blended = seeds_arr[idx_a] * w + seeds_arr[idx_b] * (1 - w)
        results.append(blended)

    return np.vstack(results) if results else np.empty((0, seeds_arr.shape[1]))


def generate_storage_perturbations(df, resource_cols):
    """For mixes with non-zero storage, perturb storage dispatch by ±0.05%.
    Returns DataFrame with perturbed storage values."""
    has_storage = (df[STORAGE_COLS].sum(axis=1) > 0)
    storage_seeds = df[has_storage].copy()
    if len(storage_seeds) == 0:
        return pd.DataFrame(columns=df.columns)

    results = []
    for sc in STORAGE_COLS:
        for delta in [-STORAGE_DELTA, STORAGE_DELTA]:
            perturbed = storage_seeds.copy()
            perturbed[sc] = np.clip(perturbed[sc] + delta, 0, None)
            perturbed['pareto_type'] = 'augmented'
            results.append(perturbed)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=df.columns)


def score_resource_only(candidates_arr, demand_arr, supply_matrix, total_demand):
    """Score resource-only candidates using batch_hourly_scores. Returns (N,) scores in %."""
    raw_scores = s1.batch_hourly_scores(demand_arr, supply_matrix, candidates_arr)
    return raw_scores / total_demand * 100.0


def score_with_storage(candidates_resource_arr, storage_values, demand_arr, supply_matrix):
    """Score candidates that have non-zero storage dispatch.

    Chunks the computation to avoid OOM on large candidate sets.
    Each chunk builds supply profiles and calls _score_with_all_storage per mix.

    Args:
        candidates_resource_arr: (N, n_resources) resource allocations in %
        storage_values: (N, 4) array of [bat4_pct, bat8_pct, ldes_pct, h2_pct]
        demand_arr: (8760,) normalized demand
        supply_matrix: (n_resources, 8760) supply profiles

    Returns: (N,) scores in % of total demand
    """
    N = len(candidates_resource_arr)
    total_demand = demand_arr.sum()
    scores = np.empty(N, dtype=np.float64)

    ldes_window_hours = LDES_WINDOW_DAYS * 24
    h2_window_hours = H2_WINDOW_DAYS * 24

    # Process in chunks to keep supply_profiles under ~800MB
    # 10K mixes × 8760 × 8 bytes ≈ 700 MB
    chunk_size = 10000
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk_resources = candidates_resource_arr[start:end]
        chunk_storage = storage_values[start:end]

        # Build supply profiles for this chunk: (chunk_size, 8760)
        supply_profiles = (chunk_resources / 100.0) @ supply_matrix

        for i in range(end - start):
            bat4_pct = chunk_storage[i, 0]
            bat8_pct = chunk_storage[i, 1]
            ldes_pct = chunk_storage[i, 2]
            h2_pct = chunk_storage[i, 3]

            bat4_cap = bat4_pct / 100.0
            bat4_pow = bat4_cap / BATTERY_DURATION_HOURS if bat4_cap > 0 else 0.0
            bat8_cap = bat8_pct / 100.0
            bat8_pow = bat8_cap / BATTERY8_DURATION_HOURS if bat8_cap > 0 else 0.0
            ldes_cap = ldes_pct / 100.0
            ldes_pow = ldes_cap / LDES_DURATION_HOURS if ldes_cap > 0 else 0.0
            h2_cap = h2_pct / 100.0
            h2_pow = h2_cap / H2_DURATION_HOURS if h2_cap > 0 else 0.0

            scores[start + i] = s1._score_with_all_storage(
                demand_arr, supply_profiles[i], 1.0,
                bat4_cap, bat4_pow, BATTERY_EFFICIENCY,
                bat8_cap, bat8_pow, BATTERY8_EFFICIENCY,
                ldes_cap, ldes_pow, LDES_EFFICIENCY,
                ldes_window_hours,
                h2_cap, h2_pow, H2_EFFICIENCY, h2_window_hours,
            )

        del supply_profiles

    return scores / total_demand * 100.0


def augment_thin_cell(iso, threshold, min_threshold, demand_arr, supply_matrix,
                      total_demand, resource_cols):
    """Augment a single thin EF cell. Returns augmented DataFrame or None."""
    df = load_ef_parquet(iso, threshold)
    if df is None:
        print(f"  WARNING: No EF file for {iso} @ {threshold}%")
        return None

    current_count = len(df)
    if current_count >= min_threshold:
        return None  # Already sufficient

    deficit = min_threshold - current_count
    lower, upper = get_threshold_bounds(threshold)
    print(f"\n  {iso} @ {threshold}%: {current_count} mixes (need {min_threshold}, deficit {deficit})")
    print(f"    Band: [{lower}, {upper})")

    # Load adjacent bands as additional seeds (sample to keep candidate count manageable)
    # Cap adjacent seeds so we don't generate far more than needed
    MAX_ADJACENT_SEEDS = min(2000, max(500, deficit))
    below, above = get_adjacent_thresholds(threshold)
    seed_dfs = [df]
    for adj_t in [below, above]:
        if adj_t is not None:
            adj_df = load_ef_parquet(iso, adj_t)
            if adj_df is not None:
                if len(adj_df) > MAX_ADJACENT_SEEDS:
                    adj_df = adj_df.sample(n=MAX_ADJACENT_SEEDS, random_state=42)
                seed_dfs.append(adj_df)
                print(f"    Adjacent seeds from {adj_t}%: {len(adj_df)} mixes")

    all_seeds = pd.concat(seed_dfs, ignore_index=True)
    n_resources = len(resource_cols)

    # Extract resource arrays for perturbation/interpolation
    seed_resource_arr = all_seeds[resource_cols].values.astype(np.float64)
    seed_storage_arr = all_seeds[STORAGE_COLS].values.astype(np.float64)
    seed_full_arr = np.hstack([seed_resource_arr, seed_storage_arr])

    # --- Strategy 1: Resource perturbation ---
    print(f"    Generating perturbations from {len(all_seeds)} seeds...")
    perturbed_resources = generate_perturbations(seed_resource_arr, resource_cols, n_resources)

    # Inherit storage from seeds (tile to match perturbation shape)
    n_deltas = len(RESOURCE_DELTAS)
    perturbed_storage = np.tile(seed_storage_arr, (n_resources * n_deltas, 1))

    # --- Strategy 2: Interpolation ---
    print(f"    Generating interpolations...")
    interp_full = generate_interpolations(seed_full_arr, n_pairs=MAX_INTERP_PAIRS)
    if len(interp_full) > 0:
        interp_resources = interp_full[:, :n_resources]
        interp_storage = interp_full[:, n_resources:]
        # Round storage to 0.05% granularity (dedup resolution)
        interp_storage = np.round(interp_storage * STORAGE_SCALE) / STORAGE_SCALE
        np.clip(interp_storage, 0, None, out=interp_storage)
    else:
        interp_resources = np.empty((0, n_resources))
        interp_storage = np.empty((0, 4))

    # Combine perturbation + interpolation candidates
    all_candidate_resources = np.vstack([perturbed_resources, interp_resources])
    all_candidate_storage = np.vstack([perturbed_storage, interp_storage])

    # --- Strategy 3: Storage dispatch perturbation ---
    storage_perturbed_df = generate_storage_perturbations(all_seeds, resource_cols)

    print(f"    Total candidates: {len(all_candidate_resources)} (perturbation+interpolation) "
          f"+ {len(storage_perturbed_df)} (storage perturbation)")

    # --- Score candidates ---
    # Split into resource-only (zero storage) and resource+storage tracks
    has_storage_mask = all_candidate_storage.sum(axis=1) > 0
    n_resource_only = (~has_storage_mask).sum()
    n_with_storage = has_storage_mask.sum()

    print(f"    Scoring: {n_resource_only} resource-only, {n_with_storage} with storage...")
    t0 = time.time()

    all_scores = np.empty(len(all_candidate_resources), dtype=np.float64)

    # Track 1: Resource-only scoring (fast vectorized)
    if n_resource_only > 0:
        ro_scores = score_resource_only(
            all_candidate_resources[~has_storage_mask],
            demand_arr, supply_matrix, total_demand)
        all_scores[~has_storage_mask] = ro_scores

    # Track 2: Resource+storage scoring (Numba per-mix)
    if n_with_storage > 0:
        ws_scores = score_with_storage(
            all_candidate_resources[has_storage_mask],
            all_candidate_storage[has_storage_mask],
            demand_arr, supply_matrix)
        all_scores[has_storage_mask] = ws_scores

    print(f"    Scoring took {time.time() - t0:.1f}s")

    # --- Filter to target band ---
    in_band = (all_scores >= lower) & (all_scores < upper)
    n_in_band = in_band.sum()
    print(f"    In-band candidates: {n_in_band} / {len(all_scores)}")

    # Build candidate DataFrame
    candidate_data = {'iso': iso}
    for i, col in enumerate(resource_cols):
        candidate_data[col] = all_candidate_resources[in_band, i]
    for i, sc in enumerate(STORAGE_COLS):
        candidate_data[sc] = all_candidate_storage[in_band, i]
    candidate_data['hourly_match_score'] = all_scores[in_band]
    candidate_data['pareto_type'] = 'augmented'

    candidate_df = pd.DataFrame(candidate_data)

    # Add storage-perturbed candidates (already have scores from seeds, need re-scoring)
    if len(storage_perturbed_df) > 0:
        sp_resource_arr = storage_perturbed_df[resource_cols].values.astype(np.float64)
        sp_storage_arr = storage_perturbed_df[STORAGE_COLS].values.astype(np.float64)
        sp_has_storage = sp_storage_arr.sum(axis=1) > 0

        sp_scores = np.empty(len(sp_resource_arr), dtype=np.float64)
        if (~sp_has_storage).sum() > 0:
            sp_scores[~sp_has_storage] = score_resource_only(
                sp_resource_arr[~sp_has_storage], demand_arr, supply_matrix, total_demand)
        if sp_has_storage.sum() > 0:
            sp_scores[sp_has_storage] = score_with_storage(
                sp_resource_arr[sp_has_storage], sp_storage_arr[sp_has_storage],
                demand_arr, supply_matrix)

        sp_in_band = (sp_scores >= lower) & (sp_scores < upper)
        if sp_in_band.sum() > 0:
            storage_perturbed_df = storage_perturbed_df[sp_in_band].copy()
            storage_perturbed_df['hourly_match_score'] = sp_scores[sp_in_band]
            candidate_df = pd.concat([candidate_df, storage_perturbed_df], ignore_index=True)

    # --- Combine with existing and deduplicate ---
    combined = pd.concat([df, candidate_df], ignore_index=True)
    combined = deduplicate(combined, resource_cols)

    new_count = len(combined)
    added = new_count - current_count
    print(f"    Result: {current_count} → {new_count} mixes (+{added})")

    if new_count < min_threshold:
        print(f"    WARNING: Still below target ({new_count} < {min_threshold})")

    return combined


def write_ef_parquet(df, iso, threshold):
    """Write augmented EF parquet, removing any existing part files."""
    t_str = f'{threshold:g}'

    # Remove existing files (single + parts)
    for f in os.listdir(EF_DIR):
        if f.startswith(f'step_2_1_EF_{iso}_{t_str}') and f.endswith('.parquet'):
            os.remove(os.path.join(EF_DIR, f))

    path = os.path.join(EF_DIR, f'step_2_1_EF_{iso}_{t_str}.parquet')
    df.to_parquet(path, engine='pyarrow', compression='snappy', index=False)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"    Wrote {path} ({len(df):,} rows, {size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description='Step 2.1b: Augment thin EF cells')
    parser.add_argument('--iso', type=str, default='ALL',
                        help='ISO to process (default: ALL)')
    parser.add_argument('--min-threshold', type=int, default=2500,
                        help='Minimum mixes per threshold band (default: 2500)')
    args = parser.parse_args()

    isos = ISOS if args.iso == 'ALL' else [args.iso]
    min_threshold = args.min_threshold

    print(f"Step 2.1b: Augment Thin EF Cells")
    print(f"  ISOs: {isos}")
    print(f"  Min threshold: {min_threshold}")
    print(f"  EF dir: {EF_DIR}")
    print()

    # Scan for thin cells first
    thin_cells = []
    for iso in isos:
        for t in ACTIVE_THRESHOLDS:
            df = load_ef_parquet(iso, t)
            if df is not None and len(df) < min_threshold:
                thin_cells.append((iso, t, len(df)))

    if not thin_cells:
        print("No thin cells found — all bands have sufficient mixes.")
        return

    print(f"Found {len(thin_cells)} thin cells:")
    for iso, t, count in thin_cells:
        print(f"  {iso} @ {t}%: {count} mixes")
    print()

    # Load data (shared across all ISOs)
    print("Loading generation/demand profiles...")
    demand_data, gen_profiles, _, _ = s1.load_data()

    # Group thin cells by ISO to avoid reloading profiles
    from collections import defaultdict
    by_iso = defaultdict(list)
    for iso, t, count in thin_cells:
        by_iso[iso].append(t)

    total_augmented = 0
    for iso in by_iso:
        print(f"\n{'='*60}")
        print(f"Processing {iso}")
        print(f"{'='*60}")

        # Load profiles for this ISO
        supply_profiles = s1.get_supply_profiles(iso, gen_profiles)
        hybrid_profiles = s1.load_hybrid_profiles(iso)
        demand_norm = demand_data[iso]['normalized']
        demand_arr, supply_matrix = s1.prepare_numpy_profiles(
            iso, demand_norm, supply_profiles,
            include_hybrids=True, hybrid_profiles=hybrid_profiles)
        total_demand = demand_arr.sum()

        # Get resource columns from first thin cell
        sample_df = load_ef_parquet(iso, by_iso[iso][0])
        resource_cols = get_resource_cols(sample_df)

        for threshold in by_iso[iso]:
            result = augment_thin_cell(
                iso, threshold, min_threshold,
                demand_arr, supply_matrix, total_demand, resource_cols)
            if result is not None:
                write_ef_parquet(result, iso, threshold)
                total_augmented += 1

    print(f"\n{'='*60}")
    print(f"Done. Augmented {total_augmented} thin cells.")

    # Final verification
    print("\nVerification — thin cells after augmentation:")
    still_thin = False
    for iso, t, old_count in thin_cells:
        df = load_ef_parquet(iso, t)
        new_count = len(df) if df is not None else 0
        status = "OK" if new_count >= min_threshold else "STILL THIN"
        if new_count < min_threshold:
            still_thin = True
        print(f"  {iso} @ {t}%: {old_count} → {new_count} [{status}]")

    if still_thin:
        print("\nWARNING: Some cells remain below threshold. Consider lowering "
              "--min-threshold or running additional PFS generation.")
        sys.exit(1)


if __name__ == '__main__':
    main()
