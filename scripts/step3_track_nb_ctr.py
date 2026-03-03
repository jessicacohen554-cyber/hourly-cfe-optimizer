#!/usr/bin/env python3
"""
Track 2-3: New-Build (NB) + Cost-to-Replace (CTR) analysis.
Does NOT rerun baseline — preserves existing overprocure_results.json.

Track 2 (newbuild / NB): hydro=0 mixes, uprates ON
  Source: per-ISO EF parquets (data/step2-ef-parquets/step2_ef_{ISO}_t{T}.parquet)
  Filtered to hydro=0 mixes.
  Purpose: What does hourly matching incentivize from scratch?

Track 3 (cost-to-replace / CTR): all mixes (hydro≤existing), uprates OFF
  Source: per-ISO EF parquets (data/step2-ef-parquets/step2_ef_{ISO}_t{T}.parquet)
  Purpose: Cost to replace existing clean generation

Checkpoint: Parquet-based. After each (iso, track) completes, results are
  appended to track_scenarios.parquet. On resume, completed (iso, track) pairs
  are read from the parquet header — no full data load needed.

Usage:
  python step3_track_nb_ctr.py              # Medium-only (fast, ~30s)
  python step3_track_nb_ctr.py --full       # All 5,832+ combos (hours)
  python step3_track_nb_ctr.py --iso PJM    # Single ISO
"""

import gc
import os
import sys
import time
import argparse
from functools import lru_cache
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPT_DIR)

from step3_cost_optimization import (
    ISOS, OUTPUT_THRESHOLDS, REGIONAL_DEMAND_TWH, GRID_MIX_SHARES,
    WHOLESALE_PRICES, FUEL_ADJUSTMENTS, LEVEL_NAME, RESOURCE_TYPES,
    DEMAND_GROWTH_RATES, DEMAND_GROWTH_YEARS, DEMAND_GROWTH_LEVELS,
    THRESHOLD_TARGET_YEARS, DG_UNIQUE_YEARS, _DG_YEAR_TO_THRESHOLDS,
    precompute_base_year_coefficients, get_scenario_prices,
    eval_cost_fast, eval_and_argmin_all, build_winner_scenario,
    build_sensitivity_combos, medium_key, price_mix_batch,
    precompute_all_prices, batch_eval_and_argmin_all,
    HAS_NUMBA, effective_gate,
)

from step3_cost_optimization import (
    _N_COEFFS, _COL_WHOLESALE, _COL_SOL_NEW, _COL_WND_NEW, _COL_OSW_NEW,
    _COL_CCS_NEW, _COL_UPRATE, _COL_GEO, _COL_REMAINING, _COL_BAT4,
    _COL_BAT8, _COL_LDES, _COL_H2, OFFSHORE_ISOS,
)

# Additional constants needed for DG coefficient computation
from step3_cost_optimization import (
    UPRATE_CAP_TWH, GEO_CAP_TWH,
    PEAK_DEMAND_MW, RESOURCE_ADEQUACY_MARGIN, PEAK_CAPACITY_CREDITS,
    GAS_AVAILABILITY_FACTOR, EXISTING_GAS_CAPACITY_MW,
    EXISTING_GAS_FOM_KW_YR, NEW_CCGT_COST_KW_YR,
)

# ── Pre-computed threshold lookups (avoid repeated function calls) ──
# effective_gate values for each threshold, computed once at import
_EFFECTIVE_GATES = {thr: effective_gate(thr) for thr in OUTPUT_THRESHOLDS}
# Pre-computed str(thr) for each threshold (avoids repeated float→str conversion)
_THR_STR = {thr: str(thr) for thr in OUTPUT_THRESHOLDS}
# Pre-computed str(year) for DG years
_YEAR_STR = {yr: str(yr) for yr in DG_UNIQUE_YEARS}

# Pareto pruning price bounds (module-level constant, avoid per-call allocation)
# Cols: wholesale, sol_new, wnd_new, ccs_new, uprate, geo, remaining, bat4, bat8, ldes, h2
_PARETO_MIN_PRICES = np.array([20, 40, 30, 52, 15, 0, 52, 69, 77, 116, 182], dtype=np.float64)
_PARETO_MAX_PRICES = np.array([50, 82, 83, 164, 15, 116, 84, 144, 179, 267, 470], dtype=np.float64)

# Per-ISO EF parquet directory (Step 2 output)
EF_ISO_DIR = os.path.join(SCRIPT_DIR, 'data', 'step2-ef-parquets')

# Parquet paths — these ARE the checkpoints
PQ_SCENARIOS_PATH = os.path.join(SCRIPT_DIR, 'dashboard', 'track_scenarios.parquet')
PQ_DG_PATH = os.path.join(SCRIPT_DIR, 'dashboard', 'track_demand_growth.parquet')


def load_completed_tracks(pq_path):
    """Read parquet metadata to find completed (iso, track) pairs.

    Only reads the iso and track columns — O(rows) but tiny memory since
    it's just two string columns from a ~1MB file.

    Also cleans up stale track names (e.g., 'replace' → 'cost_to_replace').
    """
    if not os.path.exists(pq_path):
        return set()
    table = pq.read_table(pq_path, columns=['iso', 'track'])
    iso_col = table.column('iso').to_pylist()
    track_col = table.column('track').to_pylist()

    # Clean up stale 'replace' track name from older runs
    VALID_TRACKS = {'newbuild', 'cost_to_replace'}
    stale_indices = [i for i, t in enumerate(track_col) if t not in VALID_TRACKS]
    if stale_indices:
        stale_names = list(set(track_col[i] for i in stale_indices))
        print(f"  Cleaning {len(stale_indices):,} stale rows with track names: {stale_names}")
        # Re-read full file, filter valid rows, rewrite
        full_table = pq.read_table(pq_path)
        full_track = full_table.column('track').to_pylist()
        keep_mask = pa.array([t in VALID_TRACKS for t in full_track])
        full_table = full_table.filter(keep_mask)
        tmp = pq_path + '.tmp'
        pq.write_table(full_table, tmp, compression='zstd')
        os.replace(tmp, pq_path)
        print(f"  Rewrote checkpoint: {full_table.num_rows:,} rows (removed {len(stale_indices):,} stale)")
        iso_col = full_table.column('iso').to_pylist()
        track_col = full_table.column('track').to_pylist()

    completed = set(zip(iso_col, track_col))
    print(f"  Parquet checkpoint: {len(completed)} (iso, track) pairs completed")
    # Count per pair
    pair_counts = {}
    for iso_val, track_val in zip(iso_col, track_col):
        pair_counts[(iso_val, track_val)] = pair_counts.get((iso_val, track_val), 0) + 1
    for (iso_val, track_val), n in sorted(pair_counts.items()):
        print(f"    {iso_val}/{track_val}: {n:,} scenarios")
    return completed


def flatten_track_rows(iso, track_name, result_dict):
    """Flatten nested track result dict into flat rows for parquet.

    Optimized: pre-computes threshold float conversion per outer loop iteration,
    uses direct dict access with .get() defaults to avoid repeated lookups.
    """
    rows = []
    _get = dict.get  # local reference for faster attribute resolution
    for thr_str, thr_val in result_dict.items():
        thr_float = float(thr_str)
        scenarios = _get(thr_val, 'scenarios', None)
        if not scenarios:
            continue
        for sc_key, sc in scenarios.items():
            row = {
                'iso': iso,
                'track': track_name,
                'threshold': thr_float,
                'scenario': sc_key,
            }
            resource_mix = _get(sc, 'resource_mix', None)
            if resource_mix:
                for k, v in resource_mix.items():
                    row[f'mix_{k}'] = v
            row['hourly_match_score'] = _get(sc, 'hourly_match_score', None)
            row['battery_dispatch_pct'] = _get(sc, 'battery_dispatch_pct', None)
            row['battery8_dispatch_pct'] = _get(sc, 'battery8_dispatch_pct', None)
            row['ldes_dispatch_pct'] = _get(sc, 'ldes_dispatch_pct', None)
            row['h2_dispatch_pct'] = _get(sc, 'h2_dispatch_pct', None)
            costs = _get(sc, 'costs', None)
            if costs:
                for k, v in costs.items():
                    row[f'cost_{k}'] = v
            tranche_costs = _get(sc, 'tranche_costs', None)
            if tranche_costs:
                for k, v in tranche_costs.items():
                    row[f'tranche_{k}'] = v
            gas_backup = _get(sc, 'gas_backup', None)
            if gas_backup:
                for k, v in gas_backup.items():
                    row[f'gas_{k}'] = v
            rows.append(row)
    return rows


def flatten_dg_rows(iso, track_name, dg_dict, arrays=None):
    """Flatten demand growth result dict into flat rows for parquet.

    If arrays (PFS arrays) are provided, resolves mix_idx to full resource mix
    for downstream pipeline compatibility. Uses vectorized numpy fancy indexing
    to batch-resolve mix indices per growth-level group.
    """
    rows = []
    # Pre-resolve array references once (avoids repeated dict lookups in inner loop)
    if arrays is not None:
        arr_cf = arrays['clean_firm']
        arr_sol = arrays['solar']
        arr_wnd = arrays['wind']
        arr_hyd = arrays['hydro']
        arr_osw = arrays.get('offshore_wind', np.zeros(len(arr_cf), dtype=np.int64))
        arr_score = arrays['hourly_match_score']
        arr_bat = arrays.get('battery_dispatch_pct', np.zeros(1))
        arr_bat8 = arrays.get('battery8_dispatch_pct', np.zeros(1))
        arr_ldes = arrays.get('ldes_dispatch_pct', np.zeros(1))
        arr_h2 = arrays.get('h2_dispatch_pct', np.zeros(1))

    for thr_str, sc_dict in dg_dict.items():
        thr_float = float(thr_str)
        for sc_key, year_data in sc_dict.items():
            for year_str, growth_data in year_data.items():
                year_int = int(year_str)

                # Vectorized: batch-resolve all mix indices for this group
                g_levels = list(growth_data.keys())
                g_vals = [growth_data[g] for g in g_levels]

                if arrays is not None and len(g_levels) > 1:
                    idxs = np.array([v[0] for v in g_vals], dtype=np.intp)
                    cf_batch = arr_cf[idxs].astype(int)
                    sol_batch = arr_sol[idxs].astype(int)
                    wnd_batch = arr_wnd[idxs].astype(int)
                    hyd_batch = arr_hyd[idxs].astype(int)
                    osw_batch = arr_osw[idxs].astype(int)
                    score_batch = arr_score[idxs]
                    bat_batch = np.round(arr_bat[idxs].astype(np.float64), 4)
                    bat8_batch = np.round(arr_bat8[idxs].astype(np.float64), 4)
                    ldes_batch = np.round(arr_ldes[idxs].astype(np.float64), 4)
                    h2_batch = np.round(arr_h2[idxs].astype(np.float64), 4)

                    for i, (g_level, vals) in enumerate(zip(g_levels, g_vals)):
                        cf, sol, wnd, hyd, osw = (
                            int(cf_batch[i]), int(sol_batch[i]),
                            int(wnd_batch[i]), int(hyd_batch[i]), int(osw_batch[i]))
                        rows.append({
                            'iso': iso, 'track': track_name,
                            'threshold': thr_float, 'scenario': sc_key,
                            'year': year_int, 'growth_level': g_level,
                            'growth_factor': vals[4] if len(vals) > 4 else None,
                            'annual_demand_mwh': vals[5] if len(vals) > 5 else None,
                            'cost_total_cost': vals[1],
                            'cost_effective_cost': vals[2],
                            'cost_incremental': vals[3],
                            'mix_clean_firm': cf, 'mix_solar': sol,
                            'mix_wind': wnd, 'mix_offshore_wind': osw,
                            'mix_ccs_ccgt': max(0, 100 - (cf + sol + wnd + hyd + osw)),
                            'mix_hydro': hyd,
                            'hourly_match_score': float(score_batch[i]),
                            'battery_dispatch_pct': float(bat_batch[i]),
                            'battery8_dispatch_pct': float(bat8_batch[i]),
                            'ldes_dispatch_pct': float(ldes_batch[i]),
                            'h2_dispatch_pct': float(h2_batch[i]),
                        })
                else:
                    for g_level, vals in zip(g_levels, g_vals):
                        mix_idx = vals[0]
                        row = {
                            'iso': iso, 'track': track_name,
                            'threshold': thr_float, 'scenario': sc_key,
                            'year': year_int, 'growth_level': g_level,
                            'growth_factor': vals[4] if len(vals) > 4 else None,
                            'annual_demand_mwh': vals[5] if len(vals) > 5 else None,
                            'cost_total_cost': vals[1],
                            'cost_effective_cost': vals[2],
                            'cost_incremental': vals[3],
                        }
                        if arrays is not None:
                            cf = int(arr_cf[mix_idx])
                            sol = int(arr_sol[mix_idx])
                            wnd = int(arr_wnd[mix_idx])
                            hyd = int(arr_hyd[mix_idx])
                            osw = int(arr_osw[mix_idx])
                            row['mix_clean_firm'] = cf
                            row['mix_solar'] = sol
                            row['mix_wind'] = wnd
                            row['mix_offshore_wind'] = osw
                            row['mix_ccs_ccgt'] = max(0, 100 - (cf + sol + wnd + hyd + osw))
                            row['mix_hydro'] = hyd
                            row['hourly_match_score'] = float(arr_score[mix_idx])
                            row['battery_dispatch_pct'] = round(float(arr_bat[mix_idx]), 4)
                            row['battery8_dispatch_pct'] = round(float(arr_bat8[mix_idx]), 4)
                            row['ldes_dispatch_pct'] = round(float(arr_ldes[mix_idx]), 4)
                            row['h2_dispatch_pct'] = round(float(arr_h2[mix_idx]), 4)
                        else:
                            row['best_mix_idx'] = mix_idx
                        rows.append(row)
    return rows


# Module-level type constants (avoid re-creating per call)
_LARGE_STRING = pa.large_string()
_STRING = pa.string()

# Cache normalized schemas by original schema fingerprint to avoid redundant iteration
_SCHEMA_CACHE = {}


def _normalize_schema(table):
    """Normalize large_string → string in a PyArrow table to avoid schema merge failures.

    Uses a schema cache keyed on the original schema to skip re-computation when
    the same column layout is seen again (common during iterative appends).
    """
    orig_schema = table.schema
    cache_key = str(orig_schema)
    if cache_key in _SCHEMA_CACHE:
        cached = _SCHEMA_CACHE[cache_key]
        if cached is None:
            return table  # No cast needed
        return table.cast(cached)

    new_fields = []
    needs_cast = False
    for field in orig_schema:
        if field.type == _LARGE_STRING:
            new_fields.append(pa.field(field.name, _STRING, nullable=field.nullable))
            needs_cast = True
        else:
            new_fields.append(field)
    if needs_cast:
        new_schema = pa.schema(new_fields)
        _SCHEMA_CACHE[cache_key] = new_schema
        table = table.cast(new_schema)
    else:
        _SCHEMA_CACHE[cache_key] = None  # Sentinel: no cast needed
    return table


def append_to_parquet(new_rows, pq_path):
    """Append rows to parquet, replacing any existing data for the same (iso, track) pairs."""
    if not new_rows:
        return 0

    # Build pyarrow table from new rows
    all_keys = list(dict.fromkeys(k for r in new_rows for k in r))
    new_table = pa.table({k: pa.array([r.get(k) for r in new_rows]) for k in all_keys})

    if os.path.exists(pq_path):
        existing_table = pq.read_table(pq_path)
        existing_table = _normalize_schema(existing_table)
        new_table = _normalize_schema(new_table)
        # Find (iso, track) pairs in new data to replace.
        # Build composite key strings for fast set-membership via PyArrow compute.
        new_iso = new_table.column('iso')
        new_track = new_table.column('track')
        new_keys = set(zip(new_iso.to_pylist(), new_track.to_pylist()))

        # Fast path: if new data has only one (iso, track) pair (common case),
        # use PyArrow compute filters directly instead of Python iteration
        if len(new_keys) == 1:
            iso_val, track_val = next(iter(new_keys))
            iso_match = pc.equal(existing_table.column('iso'), iso_val)
            track_match = pc.equal(existing_table.column('track'), track_val)
            drop_mask = pc.and_(iso_match, track_match)
            keep_mask = pc.invert(drop_mask)
            keep_table = existing_table.filter(keep_mask)
        else:
            # Multiple pairs: build composite key for existing rows
            ex_iso = existing_table.column('iso').to_pylist()
            ex_track = existing_table.column('track').to_pylist()
            keep_np = np.array([(i, t) not in new_keys for i, t in zip(ex_iso, ex_track)], dtype=bool)
            keep_mask = pa.array(keep_np, type=pa.bool_())
            keep_table = existing_table.filter(keep_mask)

        # Unify schemas before concat
        combined_schema = pa.unify_schemas([keep_table.schema, new_table.schema])
        keep_table = keep_table.cast(combined_schema)
        new_table = new_table.cast(combined_schema)
        merged = pa.concat_tables([keep_table, new_table], promote_options='permissive')
        print(f"    Parquet merge: kept {keep_table.num_rows:,} + added {new_table.num_rows:,} = {merged.num_rows:,} total")
    else:
        merged = new_table

    # Atomic write
    tmp = pq_path + '.tmp'
    pq.write_table(merged, tmp, compression='zstd')
    os.replace(tmp, pq_path)
    size_mb = os.path.getsize(pq_path) / (1024 * 1024)
    print(f"    Saved: {pq_path} ({merged.num_rows:,} rows, {size_mb:.1f} MB)")
    return merged.num_rows


def load_iso_ef_parquet(iso):
    """Load numpy arrays for a single ISO from its per-threshold EF parquets.

    Reads from data/step2-ef-parquets/step2_ef_{ISO}_t{T}.parquet (Step 2 output).
    Falls back to legacy monolithic step2_ef_{ISO}.parquet if per-threshold
    files don't exist. Returns None if no files found or all empty.
    """
    import glob as globmod

    # Try per-threshold files first (new format)
    pattern = os.path.join(EF_ISO_DIR, f'step2_ef_{iso}_t*.parquet')
    thr_files = sorted(globmod.glob(pattern))
    thr_files = [f for f in thr_files if '_batch_' not in os.path.basename(f)]

    if thr_files:
        tables = []
        for fpath in thr_files:
            t = pq.read_table(fpath)
            if t.num_rows > 0:
                tables.append(t)
        if not tables:
            return None
        sub = pa.concat_tables(tables, promote_options='permissive') if len(tables) > 1 else tables[0]
        del tables
        n = sub.num_rows
        print(f"  {iso}: loaded {n:,} EF mixes from {len(thr_files)} threshold files")
    else:
        # Fall back to legacy monolithic file
        path = os.path.join(EF_ISO_DIR, f'step2_ef_{iso}.parquet')
        if not os.path.exists(path):
            print(f"  WARNING: EF parquet not found for {iso}: {path}")
            return None
        sub = pq.read_table(path)
        if sub.num_rows == 0:
            return None
        n = sub.num_rows
        print(f"  {iso}: loaded {n:,} EF mixes from {path}")
    result = {
        'clean_firm': sub.column('clean_firm').to_numpy(),
        'solar': sub.column('solar').to_numpy(),
        'wind': sub.column('wind').to_numpy(),
        'hydro': sub.column('hydro').to_numpy(),
        'offshore_wind': (sub.column('offshore_wind').to_numpy()
                          if 'offshore_wind' in sub.column_names
                          else np.zeros(n, dtype=np.int64)),
        'battery_dispatch_pct': sub.column('battery_dispatch_pct').to_numpy(),
        'battery8_dispatch_pct': (sub.column('battery8_dispatch_pct').to_numpy()
                                   if 'battery8_dispatch_pct' in sub.column_names
                                   else np.zeros(n, dtype=np.int64)),
        'ldes_dispatch_pct': sub.column('ldes_dispatch_pct').to_numpy(),
        'h2_dispatch_pct': (sub.column('h2_dispatch_pct').to_numpy()
                            if 'h2_dispatch_pct' in sub.column_names
                            else np.zeros(n, dtype=np.int64)),
        'hourly_match_score': sub.column('hourly_match_score').to_numpy(),
    }
    del sub  # Free Arrow table immediately (~3GB for large ISOs)
    gc.collect()
    return result


def pareto_prune_fast(coeff_matrix, constant, scores, thresholds):
    """Vectorized price-bound Pareto pruning per threshold.

    For each threshold t, among qualifying mixes (score >= t):
    1. Compute each mix's cost under lowest possible prices (min_cost)
       and under highest possible prices (max_cost)
    2. Find the best mix's max_cost (upper bound on best achievable)
    3. Prune any mix whose min_cost > that bound (can never win)

    Uses module-level precomputed price ranges. O(N) per threshold.

    Returns: boolean mask of NON-dominated mixes (True = keep)
    """
    N = len(scores)
    if N < 100:
        return np.ones(N, dtype=bool)

    keep = np.ones(N, dtype=bool)

    for thr in thresholds:
        gate = _EFFECTIVE_GATES.get(thr, effective_gate(thr))
        qual_mask = (scores >= gate) & keep
        qual_idx = np.where(qual_mask)[0]
        Q = len(qual_idx)
        if Q < 2:
            continue

        q_coeff = coeff_matrix[qual_idx]  # (Q, 11)
        q_const = constant[qual_idx]      # (Q,)

        # Min possible cost = q_coeff @ min_prices + q_const
        min_cost = q_coeff @ _PARETO_MIN_PRICES + q_const
        # Max possible cost = q_coeff @ max_prices + q_const
        max_cost = q_coeff @ _PARETO_MAX_PRICES + q_const

        # Best achievable upper bound: lowest max_cost among qualifying mixes
        best_max_cost = np.min(max_cost)

        # Prune: any mix whose min cost exceeds the best worst-case
        dominated = min_cost > best_max_cost
        n_pruned = dominated.sum()
        if n_pruned > 0:
            keep[qual_idx[dominated]] = False

    return keep


def run_track(track_name, iso, arrays, demand_twh, combos, uprate_cap_override=None,
              existing_override=None, apply_dominance_filter=True):
    """Run cost optimization for a track (newbuild or replace).

    Returns:
        track_data: dict of {threshold_str: {'scenarios': {key: scenario_dict}}}
        arch_set: set of winning mix indices (archetypes)
    """
    N = len(arrays['clean_firm'])
    if N == 0:
        return {}, set()

    scores = arrays['hourly_match_score'].astype(np.float64)

    # Pre-compute threshold indices using cached effective_gate values
    thr_indices = {}
    for thr in OUTPUT_THRESHOLDS:
        gate = _EFFECTIVE_GATES[thr]
        idx = np.where(scores >= gate)[0]
        if len(idx) > 0:
            thr_indices[thr] = idx

    if not thr_indices:
        return {}, set()

    # Pre-compute coefficients
    coeff_matrix, constant, extras = precompute_base_year_coefficients(
        iso, arrays, demand_twh, uprate_cap_override=uprate_cap_override,
        existing_override=existing_override)

    # Apply Pareto dominance filter — prune mixes that can never win
    if apply_dominance_filter and N > 500:
        active_thr_list = sorted(thr_indices.keys())
        keep_mask = pareto_prune_fast(coeff_matrix, constant, scores, active_thr_list)
        n_pruned = N - keep_mask.sum()
        if n_pruned > 0:
            keep_idx = np.where(keep_mask)[0]
            arrays = {k: arrays[k][keep_idx] for k in arrays}
            coeff_matrix = coeff_matrix[keep_idx]
            constant = constant[keep_idx]
            scores = scores[keep_idx]
            extras = {k: (v[keep_idx] if isinstance(v, np.ndarray) and v.shape[0] == N else v)
                      for k, v in extras.items()}
            N_new = len(keep_idx)
            print(f"    {iso} {track_name}: Pareto pruned {n_pruned:,} / {N:,} "
                  f"({100*n_pruned/N:.1f}%) → {N_new:,} mixes remain")
            N = N_new
            # Recompute threshold indices on pruned arrays
            thr_indices = {}
            for thr in OUTPUT_THRESHOLDS:
                gate = _EFFECTIVE_GATES[thr]
                idx = np.where(scores >= gate)[0]
                if len(idx) > 0:
                    thr_indices[thr] = idx

    active_thresholds = sorted(thr_indices.keys())
    thresholds_desc = np.array(sorted(active_thresholds, reverse=True), dtype=np.float64)
    thr_pos = {float(thresholds_desc[k]): k for k in range(len(thresholds_desc))}

    thr_data = {thr: {'scenarios': {}} for thr in active_thresholds}
    arch_set = set()
    n_combos = len(combos)
    iso_start = time.time()

    # Pre-compute all price vectors at once
    price_matrix, wholesale_arr, nuclear_arr, ccs_arr = precompute_all_prices(iso, combos)
    price_time = time.time() - iso_start
    print(f"    {iso} {track_name}: prices pre-computed ({price_time:.1f}s), "
          f"launching batched eval ({N:,} mixes × {n_combos:,} combos)...")

    # Single batched Numba call — tiled for cache efficiency
    batch_start = time.time()
    all_best_idxs, all_best_vals = batch_eval_and_argmin_all(
        coeff_matrix, constant, price_matrix, scores, thresholds_desc)
    batch_elapsed = time.time() - batch_start
    print(f"    {iso} {track_name}: batched eval+argmin done in {batch_elapsed:.1f}s")

    # Build winner scenarios from batched results
    build_start = time.time()
    for j, (scenario_key, sens) in enumerate(combos):
        for thr in active_thresholds:
            k = thr_pos[float(thr)]
            if all_best_vals[j, k] == np.inf:
                continue
            best_idx = int(all_best_idxs[j, k])
            tc_val = float(all_best_vals[j, k])
            arch_set.add(best_idx)

            scenario = build_winner_scenario(
                arrays, extras, best_idx, sens, iso, demand_twh,
                tc_val, float(wholesale_arr[j]),
                float(nuclear_arr[j]), float(ccs_arr[j]))
            thr_data[thr]['scenarios'][scenario_key] = scenario
    build_elapsed = time.time() - build_start
    print(f"    {iso} {track_name}: winner scenarios built in {build_elapsed:.1f}s")

    result = {_THR_STR.get(thr, str(thr)): thr_data[thr] for thr in active_thresholds}
    elapsed = time.time() - iso_start
    print(f"  {iso:>6} {track_name:>10}: {N:,} mixes, "
          f"{len(active_thresholds)} thresholds, {len(arch_set)} archetypes — {elapsed:.0f}s")
    return result, arch_set


# 8M mixes per chunk keeps peak memory under ~5GB on 7GB GitHub Actions runners.
# Full arrays (~3GB) stay resident; each chunk adds ~2GB for coeff_matrix + extras.
_CHUNK_THRESHOLD = 15_000_000


def _run_track_chunked(track_name, iso, arrays, demand_twh, combos,
                        uprate_cap_override=None, existing_override=None,
                        chunk_size=8_000_000):
    """Tournament-style chunked run_track for arrays too large for runner RAM.

    Splits N mixes into chunks of chunk_size, finds per-chunk winners via
    run_track (with Pareto pruning), then collects all chunk winners and
    does a single final evaluation on the candidate union to find global optima.

    Returns:
        (track_data, arch_set, final_arrays) — final_arrays is the candidate
        subset used in the final pass (needed for downstream DG evaluation).
    """
    N = len(arrays['clean_firm'])
    n_chunks = (N + chunk_size - 1) // chunk_size
    print(f"    {iso} {track_name}: {N:,} mixes → {n_chunks} chunks of ≤{chunk_size:,} (OOM guard)")

    candidate_indices = set()
    for c in range(n_chunks):
        start = c * chunk_size
        end = min(start + chunk_size, N)
        chunk_n = end - start
        print(f"    Chunk {c+1}/{n_chunks}: mixes [{start:,}..{end:,}) ({chunk_n:,})")

        # Basic slicing creates views — no extra memory for chunk_arrays
        chunk_arrays = {k: arrays[k][start:end] for k in arrays}

        _chunk_data, chunk_arch = run_track(
            track_name, iso, chunk_arrays, demand_twh, combos,
            uprate_cap_override=uprate_cap_override,
            existing_override=existing_override,
            apply_dominance_filter=True)

        # Map chunk-local winner indices back to full-array indices
        for idx in chunk_arch:
            candidate_indices.add(start + idx)

        print(f"    Chunk {c+1}/{n_chunks}: {len(chunk_arch):,} candidates kept")
        del chunk_arrays, _chunk_data, chunk_arch
        gc.collect()

    # Build candidate arrays from the union of all chunk winners
    candidates = sorted(candidate_indices)
    n_cand = len(candidates)
    if n_cand == 0:
        return {}, set(), None

    cand_idx = np.array(candidates, dtype=np.int64)
    cand_arrays = {k: arrays[k][cand_idx] for k in arrays}
    print(f"    {iso} {track_name}: {n_cand:,} candidates from {n_chunks} chunks → final eval")

    # Final evaluation on candidate set — no dominance filter needed,
    # these are already proven competitive within their chunks
    final_data, final_arch = run_track(
        track_name, iso, cand_arrays, demand_twh, combos,
        uprate_cap_override=uprate_cap_override,
        existing_override=existing_override,
        apply_dominance_filter=False)

    return final_data, final_arch, cand_arrays


def _precompute_dg_coefficients(iso, arch_arrays, demand_twh,
                                 uprate_cap_override=None, existing_override=None):
    """Pre-compute demand-growth-adjusted coefficient matrices for all unique (year, growth_level) pairs.

    Instead of calling price_mix_batch (which recomputes arrays from scratch) for each
    (year, growth_level) combo, this function pre-computes the coefficient matrices once
    for all 42 unique growth scenarios (14 years × 3 levels).

    Hoists growth-invariant array computations (proc, demand pcts, ccs_pct, storage pcts)
    outside the loop, recomputing only the growth-dependent existing/new splits and gas backup.

    Returns:
        dg_coeffs: dict of (year, g_level) → (coeff_matrix, constant, match_frac)
        dg_meta: dict of (year, g_level) → (gf, demand_grown_mwh)
    """
    iso_rates = DEMAND_GROWTH_RATES[iso]
    dg_coeffs = {}
    dg_meta = {}

    # ── Growth-INVARIANT computations (hoisted outside loop) ──
    N = len(arch_arrays['clean_firm'])
    existing = existing_override if existing_override is not None else GRID_MIX_SHARES[iso]

    # Resource fractions are already % of demand (no procurement multiplier)
    match_frac = arch_arrays['hourly_match_score'] / 100.0

    cf_pct = arch_arrays['clean_firm']
    sol_pct = arch_arrays['solar']
    wnd_pct = arch_arrays['wind']
    hyd_pct = arch_arrays['hydro']
    osw_pct = arch_arrays.get('offshore_wind', np.zeros(N, dtype=np.float64))
    ccs_pct = np.maximum(0.0, 100.0 - (cf_pct + sol_pct + wnd_pct + hyd_pct + osw_pct))
    bat_pct = arch_arrays['battery_dispatch_pct']
    bat8_pct = arch_arrays.get('battery8_dispatch_pct', np.zeros(N, dtype=np.float64))
    ldes_pct = arch_arrays['ldes_dispatch_pct']
    h2_pct = arch_arrays.get('h2_dispatch_pct', np.zeros(N, dtype=np.float64))

    # Offshore wind coefficient (all new-build, no existing fleet)
    osw_coeff = osw_pct / 100.0

    # Storage coefficients are fully growth-invariant (columns 8-11)
    bat4_coeff = bat_pct / 100.0
    bat8_coeff = bat8_pct / 100.0
    ldes_coeff = ldes_pct / 100.0
    h2_coeff = h2_pct / 100.0

    # Uprate cap and geo cap are per-ISO constants
    uprate_cap = UPRATE_CAP_TWH[iso] if uprate_cap_override is None else uprate_cap_override
    is_caiso = (iso == 'CAISO')

    # Gas backup constants (per-ISO, invariant)
    peak_mw = PEAK_DEMAND_MW[iso]
    ra_peak_mw = peak_mw * (1 + RESOURCE_ADEQUACY_MARGIN)
    gaf = GAS_AVAILABILITY_FACTOR[iso]
    existing_gas_mw = EXISTING_GAS_CAPACITY_MW[iso]
    gas_fom = EXISTING_GAS_FOM_KW_YR[iso] * 1000
    new_ccgt_cost = NEW_CCGT_COST_KW_YR[iso] * 1000

    # Pre-compute peak capacity coefficient (invariant factor for clean_peak_mw / avg_demand_mw)
    # clean_peak_mw = peak_coeff_per_avg_mw * avg_demand_mw
    # where peak_coeff_per_avg_mw is independent of demand growth
    peak_coeff = (
        cf_pct / 100.0 * PEAK_CAPACITY_CREDITS['clean_firm'] +
        sol_pct / 100.0 * PEAK_CAPACITY_CREDITS['solar'] +
        wnd_pct / 100.0 * PEAK_CAPACITY_CREDITS['wind'] +
        osw_pct / 100.0 * PEAK_CAPACITY_CREDITS['offshore_wind'] +
        ccs_pct / 100.0 * PEAK_CAPACITY_CREDITS['ccs_ccgt'] +
        hyd_pct / 100.0 * PEAK_CAPACITY_CREDITS['hydro'] +
        bat_pct / 100.0 * PEAK_CAPACITY_CREDITS['battery'] +
        bat8_pct / 100.0 * PEAK_CAPACITY_CREDITS['battery8'] +
        ldes_pct / 100.0 * PEAK_CAPACITY_CREDITS['ldes'] +
        h2_pct / 100.0 * PEAK_CAPACITY_CREDITS['h2']
    )

    # Base existing shares (before scaling)
    ex_sol_base = existing['solar']
    ex_wnd_base = existing['wind']
    ex_ccs_base = existing.get('ccs_ccgt', 0)
    ex_cf_base = existing['clean_firm']

    # ── Growth-DEPENDENT loop ──
    for year in DG_UNIQUE_YEARS:
        years_out = year - 2025
        for g_level in DEMAND_GROWTH_LEVELS:
            g_rate = iso_rates[g_level]
            gf = (1 + g_rate) ** years_out
            demand_grown = demand_twh * gf
            demand_grown_mwh = demand_grown * 1e6
            existing_scale = 1.0 / gf

            # Existing/new splits with growth-adjusted existing share
            sol_ex = min(ex_sol_base * existing_scale, 100.0)
            sol_existing_pct = np.minimum(sol_pct, sol_ex)
            sol_new_pct = np.maximum(0, sol_pct - sol_ex)

            wnd_ex = min(ex_wnd_base * existing_scale, 100.0)
            wnd_existing_pct = np.minimum(wnd_pct, wnd_ex)
            wnd_new_pct = np.maximum(0, wnd_pct - wnd_ex)

            ccs_ex = min(ex_ccs_base * existing_scale, 100.0)
            ccs_existing_pct = np.minimum(ccs_pct, ccs_ex)
            ccs_new_pct = np.maximum(0, ccs_pct - ccs_ex)

            cf_ex = min(ex_cf_base * existing_scale, 100.0)
            cf_existing_pct = np.minimum(cf_pct, cf_ex)
            cf_new_pct = np.maximum(0, cf_pct - cf_ex)

            # Clean firm tranche allocation
            new_cf_twh = cf_new_pct / 100.0 * demand_grown
            uprate_twh = np.minimum(new_cf_twh, uprate_cap)
            remaining_after_uprate = np.maximum(0, new_cf_twh - uprate_twh)

            if is_caiso:
                geo_twh = np.minimum(remaining_after_uprate, GEO_CAP_TWH)
                remaining_after_geo = np.maximum(0, remaining_after_uprate - geo_twh)
            else:
                geo_twh = np.zeros(N)
                remaining_after_geo = remaining_after_uprate

            # Gas backup: clean_peak_mw = peak_coeff * avg_demand_mw
            avg_demand_mw = demand_grown_mwh / 8760
            clean_peak_mw = peak_coeff * avg_demand_mw

            gas_needed_mw = np.maximum(0, ra_peak_mw - clean_peak_mw) / gaf
            existing_gas_used_mw = np.minimum(gas_needed_mw, existing_gas_mw)
            new_gas_mw = np.maximum(0, gas_needed_mw - existing_gas_used_mw)

            constant = (
                existing_gas_used_mw * gas_fom +
                new_gas_mw * new_ccgt_cost
            ) / demand_grown_mwh

            # Build coefficient matrix (N, 12)
            coeff_matrix = np.empty((N, _N_COEFFS), dtype=np.float64)
            coeff_matrix[:, _COL_WHOLESALE] = (sol_existing_pct + wnd_existing_pct +
                                                hyd_pct + ccs_existing_pct +
                                                cf_existing_pct) / 100.0
            coeff_matrix[:, _COL_SOL_NEW] = sol_new_pct / 100.0
            coeff_matrix[:, _COL_WND_NEW] = wnd_new_pct / 100.0
            coeff_matrix[:, _COL_OSW_NEW] = osw_coeff  # all new-build, growth-invariant
            coeff_matrix[:, _COL_CCS_NEW] = ccs_new_pct / 100.0
            coeff_matrix[:, _COL_UPRATE] = uprate_twh / demand_grown
            coeff_matrix[:, _COL_GEO] = geo_twh / demand_grown
            coeff_matrix[:, _COL_REMAINING] = remaining_after_geo / demand_grown
            coeff_matrix[:, _COL_BAT4] = bat4_coeff
            coeff_matrix[:, _COL_BAT8] = bat8_coeff
            coeff_matrix[:, _COL_LDES] = ldes_coeff
            coeff_matrix[:, _COL_H2] = h2_coeff

            dg_coeffs[(year, g_level)] = (coeff_matrix, constant, match_frac)
            dg_meta[(year, g_level)] = (gf, demand_grown_mwh)

    return dg_coeffs, dg_meta


def run_track_demand_growth(track_name, iso, arrays, arch_set, combos,
                             uprate_cap_override=None, existing_override=None):
    """Run demand growth sweep for track archetypes (threshold-year paired).

    Each threshold is evaluated only at its SBTi-interpolated target year x L/M/H,
    producing 15 x 3 = 45 DG results per ISO per scenario combo.

    Optimized with two levels of batching:
    1. Pre-computes coefficient matrices for all ~42 unique (year, growth_level) pairs
       (scenario-INVARIANT), with growth-invariant computations hoisted out.
    2. For each (year, g_level), uses batch_eval_and_argmin_all to evaluate ALL combos
       in a single Numba-accelerated call instead of sequential eval_cost_fast calls.
    3. Loop order inverted: (year, g_level) outer, all combos inner -- each coefficient
       matrix is loaded once and evaluated against all price vectors simultaneously.
    """
    demand_twh = REGIONAL_DEMAND_TWH[iso]

    arch_indices = sorted(arch_set)
    n_arch = len(arch_indices)
    if n_arch == 0:
        return {}

    arch_arrays = {k: arrays[k][arch_indices] for k in arrays}
    arch_scores = arch_arrays['hourly_match_score']
    match_frac = arch_scores / 100.0

    # Pre-compute threshold qualifying masks using cached effective gates
    arch_thr_mask = {}
    for thr in OUTPUT_THRESHOLDS:
        gate = _EFFECTIVE_GATES[thr]
        qualifying = np.where(arch_scores >= gate)[0]
        if len(qualifying) > 0:
            arch_thr_mask[thr] = qualifying

    if not arch_thr_mask:
        return {}

    thr_dg = {thr: {} for thr in arch_thr_mask}
    n_combos = len(combos)

    # Pre-compute which years have matching thresholds (avoid recomputing per combo)
    year_thr_map = {}
    for year in DG_UNIQUE_YEARS:
        matched = [t for t in _DG_YEAR_TO_THRESHOLDS[year] if t in arch_thr_mask]
        if matched:
            year_thr_map[year] = matched

    if not year_thr_map:
        return {}

    # Pre-compute all price vectors + wholesale for all combos (once)
    price_matrix, wholesale_arr, _, _ = precompute_all_prices(iso, combos)

    # Build active thresholds for DG (only those with matched years)
    dg_active_thresholds = set()
    for _year, matched in year_thr_map.items():
        dg_active_thresholds.update(matched)
    dg_active_thresholds = sorted(dg_active_thresholds)

    # Descending thresholds for batch_eval_and_argmin_all
    dg_thr_desc = np.array(sorted(dg_active_thresholds, reverse=True), dtype=np.float64)
    dg_thr_pos = {float(dg_thr_desc[k]): k for k in range(len(dg_thr_desc))}

    # Pre-compute DG coefficient matrices for all (year, g_level) pairs
    # These are scenario-INVARIANT: depend on physical arrays and demand growth,
    # not on price sensitivities. Computed once, reused for every combo.
    dg_coeffs, dg_meta = _precompute_dg_coefficients(
        iso, arch_arrays, demand_twh,
        uprate_cap_override=uprate_cap_override,
        existing_override=existing_override)

    # Initialize per-combo result storage
    combo_results = [{thr: {} for thr in arch_thr_mask} for _ in range(n_combos)]

    # Inverted loop: iterate (year, g_level) in outer loop, batch all combos.
    # Each batch_eval_and_argmin_all call evaluates n_combos price vectors
    # against one coefficient matrix in a single Numba kernel.
    for year, matched_thresholds in year_thr_map.items():
        yr_str = _YEAR_STR[year]

        for g_level in DEMAND_GROWTH_LEVELS:
            coeff_matrix, constant, _mf = dg_coeffs[(year, g_level)]
            gf, demand_grown_mwh = dg_meta[(year, g_level)]
            gf_rounded = round(gf, 6)
            demand_grown_rounded = round(demand_grown_mwh, 0)

            # Batch-evaluate all combos at once via Numba-accelerated kernel
            all_best_idxs, all_best_vals = batch_eval_and_argmin_all(
                coeff_matrix, constant, price_matrix, arch_scores, dg_thr_desc)

            # Extract results per combo per matched threshold
            for j in range(n_combos):
                wholesale_j = float(wholesale_arr[j])
                for thr in matched_thresholds:
                    k = dg_thr_pos[float(thr)]
                    if all_best_vals[j, k] == np.inf:
                        continue
                    best_local = int(all_best_idxs[j, k])
                    full_idx = arch_indices[best_local]
                    tc_val = float(all_best_vals[j, k])
                    mf = float(match_frac[best_local])
                    ec_val = tc_val / mf if mf > 0 else 0.0

                    if yr_str not in combo_results[j][thr]:
                        combo_results[j][thr][yr_str] = {}
                    combo_results[j][thr][yr_str][g_level] = [
                        full_idx,
                        round(tc_val, 2),
                        round(ec_val, 2),
                        round(ec_val - wholesale_j, 2),
                        gf_rounded,
                        demand_grown_rounded,
                    ]

    # Assemble into output structure
    for j, (scenario_key, _sens) in enumerate(combos):
        for thr in arch_thr_mask:
            thr_dg[thr][scenario_key] = combo_results[j][thr]

    n_dg_evals = len(DG_UNIQUE_YEARS) * len(DEMAND_GROWTH_LEVELS)
    result = {_THR_STR.get(thr, str(thr)): thr_dg[thr] for thr in arch_thr_mask}
    print(f"  {iso:>6} {track_name:>10} DG: {n_arch} archetypes, "
          f"{len(arch_thr_mask)} thresholds, {n_dg_evals} paired evals, "
          f"{n_combos} combos (batched)")
    return result


def main():
    parser = argparse.ArgumentParser(description='Incremental track analysis')
    parser.add_argument('--full', action='store_true',
                        help='Run all sensitivity combos (hours). Default: Medium-only.')
    parser.add_argument('--fresh', action='store_true',
                        help='Ignore checkpoint and start from scratch.')
    parser.add_argument('--iso', type=str, default=None,
                        help='Run only this ISO (e.g., PJM). Default: all ISOs.')
    parser.add_argument('--track', type=str, default='both',
                        choices=['nb', 'ctr', 'both'],
                        help='Which track to run: nb (newbuild), ctr (cost-to-replace), or both. Default: both.')
    args = parser.parse_args()

    track_labels = {'nb': 'NEW-BUILD (NB) only', 'ctr': 'COST-TO-REPLACE (CTR) only', 'both': 'NEW-BUILD (NB) + COST-TO-REPLACE (CTR)'}
    print("=" * 70)
    print(f"  TRACK 2-3: {track_labels[args.track]}")
    print(f"  Mode: {'Full sweep (all combos)' if args.full else 'Medium-only (fast)'}")
    print(f"  EF source: {EF_ISO_DIR}")
    print(f"  Checkpoint: Parquet-based ({PQ_SCENARIOS_PATH})")
    print("=" * 70)
    total_start = time.time()

    # Load completed tracks from parquet checkpoint
    if args.fresh:
        print("  --fresh flag: ignoring existing parquet checkpoint")
        completed_tracks = set()
        # Remove existing parquet files so append_to_parquet starts clean
        for _fp in (PQ_SCENARIOS_PATH, PQ_DG_PATH):
            if os.path.exists(_fp):
                os.remove(_fp)
                print(f"    Removed stale {_fp}")
    else:
        completed_tracks = load_completed_tracks(PQ_SCENARIOS_PATH)

    # Warm up Numba JIT (batched function)
    if HAS_NUMBA:
        from step3_cost_optimization import _batch_eval_and_argmin, _eval_cost_numba, _argmin_bucketed, _N_COEFFS
        _dcm = np.zeros((2, _N_COEFFS))
        _dc = np.zeros(2)
        _dp = np.zeros(_N_COEFFS)
        _ds = np.array([50.0, 100.0])
        _dt = np.array([100.0, 50.0])
        _eval_cost_numba(_dcm, _dc, _dp)
        _argmin_bucketed(np.zeros(2), _ds, _dt)
        _dpm = np.zeros((2, _N_COEFFS))
        _batch_eval_and_argmin(_dcm, _dc, _dpm, _ds, _dt)
        print(f"  Numba JIT warmup complete (batched mode)")

    run_isos = [args.iso] if args.iso else ISOS
    for iso in run_isos:
        demand_twh = REGIONAL_DEMAND_TWH[iso]

        # Load per-ISO EF parquet (Step 2 output)
        iso_arrays = load_iso_ef_parquet(iso)
        if iso_arrays is None:
            print(f"  {iso:>6}: no EF data, skipping")
            continue

        # Build combos
        if args.full:
            combos = build_sensitivity_combos(iso)
        else:
            mk = medium_key(iso)
            geo = 'M' if iso == 'CAISO' else None
            sens = {'ren': 'M', 'firm': 'M', 'batt': 'M', 'ldes_lvl': 'M',
                    'ccs': 'M', 'q45': '1', 'fuel': 'M', 'tx': 'M', 'geo': geo}
            combos = [(mk, sens)]

        # Greenfield existing override: all clean resources zeroed
        greenfield_all = {'clean_firm': 0, 'solar': 0, 'wind': 0, 'offshore_wind': 0, 'ccs_ccgt': 0, 'hydro': 0}
        # CTR override: hydro stays at existing floor, everything else zeroed
        existing_shares = GRID_MIX_SHARES[iso]
        greenfield_keep_hydro = {
            'clean_firm': 0, 'solar': 0, 'wind': 0, 'offshore_wind': 0, 'ccs_ccgt': 0,
            'hydro': existing_shares['hydro'],
        }

        # Track 2: newbuild / NB (hydro=0, all existing zeroed, uprates ON)
        if args.track in ('nb', 'both') and (iso, 'newbuild') not in completed_tracks:
            h0_mask = iso_arrays['hydro'] == 0
            n_h0 = h0_mask.sum()
            if n_h0 > 0:
                h0_idx = np.where(h0_mask)[0]
                nb_arrays = {k: iso_arrays[k][h0_idx] for k in iso_arrays}

                nb_data, nb_arch = run_track(
                    'newbuild', iso, nb_arrays, demand_twh, combos,
                    existing_override=greenfield_all)

                # Save to parquet immediately
                sc_rows = flatten_track_rows(iso, 'newbuild', nb_data)
                append_to_parquet(sc_rows, PQ_SCENARIOS_PATH)

                if nb_arch:
                    nb_dg = run_track_demand_growth(
                        'newbuild', iso, nb_arrays, nb_arch, combos,
                        existing_override=greenfield_all)
                    dg_rows = flatten_dg_rows(iso, 'newbuild', nb_dg)
                    if dg_rows:
                        append_to_parquet(dg_rows, PQ_DG_PATH)
            else:
                print(f"  {iso:>6}   newbuild: no hydro=0 mixes")
        elif args.track in ('nb', 'both'):
            print(f"  {iso:>6}   newbuild: skipped (in parquet)")

        # Track 3: cost-to-replace / CTR (hydro at existing floor, everything else zeroed, uprates OFF)
        if args.track in ('ctr', 'both') and (iso, 'cost_to_replace') not in completed_tracks:
            # Compute 2050 high-demand hydro floor
            hydro_existing_share = GRID_MIX_SHARES[iso]['hydro']
            high_rate = DEMAND_GROWTH_RATES[iso].get('High',
                        DEMAND_GROWTH_RATES[iso].get('high', 0))
            target_year = max(DEMAND_GROWTH_YEARS) if DEMAND_GROWTH_YEARS else 2050
            years_of_growth = target_year - 2025
            demand_scale_2050 = (1 + high_rate) ** years_of_growth
            hydro_floor_raw = hydro_existing_share / demand_scale_2050 if demand_scale_2050 > 0 else 0
            # Physics grid uses integer % steps — snap floor down to match
            hydro_floor = int(hydro_floor_raw)
            n_before = len(iso_arrays['hydro'])
            h_mask = iso_arrays['hydro'] >= hydro_floor
            n_pass = int(h_mask.sum())
            if hydro_floor > 0 and n_pass > 0 and n_pass < n_before:
                h_idx = np.where(h_mask)[0]
                ctr_arrays = {k: iso_arrays[k][h_idx] for k in iso_arrays}
                print(f"    {iso} cost_to_replace: hydro>={hydro_floor}% (2050 high-demand floor, "
                      f"raw={hydro_floor_raw:.2f}%) "
                      f"{n_before:,} → {n_pass:,} ({100*(n_before-n_pass)/n_before:.1f}% pruned)")
            elif hydro_floor > 0 and n_pass == 0:
                print(f"    {iso} cost_to_replace: no mixes with hydro>={hydro_floor}%, skipping")
                continue
            else:
                # hydro_floor == 0 or all mixes qualify — use all mixes
                if hydro_floor == 0:
                    print(f"    {iso} cost_to_replace: hydro floor rounds to 0% "
                          f"(raw={hydro_floor_raw:.2f}%) — using all {n_before:,} mixes")
                ctr_arrays = iso_arrays

            N_ctr = len(ctr_arrays['clean_firm'])
            if N_ctr > _CHUNK_THRESHOLD:
                ctr_data, ctr_arch, ctr_eval = _run_track_chunked(
                    'cost_to_replace', iso, ctr_arrays, demand_twh, combos,
                    uprate_cap_override=0, existing_override=greenfield_keep_hydro)
                del ctr_arrays
                gc.collect()
            else:
                ctr_data, ctr_arch = run_track(
                    'cost_to_replace', iso, ctr_arrays, demand_twh, combos,
                    uprate_cap_override=0, existing_override=greenfield_keep_hydro)
                ctr_eval = ctr_arrays

            # Save to parquet immediately
            sc_rows = flatten_track_rows(iso, 'cost_to_replace', ctr_data)
            append_to_parquet(sc_rows, PQ_SCENARIOS_PATH)

            if ctr_arch:
                ctr_dg = run_track_demand_growth(
                    'cost_to_replace', iso, ctr_eval, ctr_arch, combos,
                    uprate_cap_override=0, existing_override=greenfield_keep_hydro)
                dg_rows = flatten_dg_rows(iso, 'cost_to_replace', ctr_dg)
                if dg_rows:
                    append_to_parquet(dg_rows, PQ_DG_PATH)
            del ctr_eval
            gc.collect()
        elif args.track in ('ctr', 'both'):
            print(f"  {iso:>6}    cost_to_replace: skipped (in parquet)")

    total_elapsed = time.time() - total_start
    pq_size = os.path.getsize(PQ_SCENARIOS_PATH) / (1024 * 1024) if os.path.exists(PQ_SCENARIOS_PATH) else 0
    dg_size = os.path.getsize(PQ_DG_PATH) / (1024 * 1024) if os.path.exists(PQ_DG_PATH) else 0
    print(f"\n{'='*70}")
    print(f"  TRACK ANALYSIS COMPLETE in {total_elapsed:.0f}s")
    print(f"  Scenarios: {PQ_SCENARIOS_PATH} ({pq_size:.1f} MB)")
    print(f"  Demand Growth: {PQ_DG_PATH} ({dg_size:.1f} MB)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
