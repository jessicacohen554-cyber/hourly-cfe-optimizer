#!/usr/bin/env python3
"""Step 1c: Zone-based fine search with global deduplication.

Replaces per-threshold fine refinement with score-band zones that eliminate
redundant scoring across thresholds. Each mix is scored exactly ONCE and
assigned to all thresholds where it's feasible or near-miss.

Architecture:
  1. Load scored coarse cache from step1b
  2. For each score-band zone (A/B/C):
     a. Identify coarse boundary mixes in zone's score range
     b. Compute zone-specific resource windows (from prior EF or coarse data)
     c. Generate fine 1% grid within zone bounds
     d. Dedup against global hash set (no mix scored twice)
     e. Score all new mixes ONCE via vectorized batch_hourly_scores
     f. Assign each scored mix to relevant thresholds
  3. Per-threshold: dominance filter + save PFS parquet
  4. Save union near-miss list for step1d storage sweep

Output:
  data/step1-pfs-parquets/{ISO}_t{XX}_raw_pfs.parquet  (per-threshold feasible)
  data/step1-pfs-parquets/{ISO}_near_miss.parquet       (union near-miss for 1d)

Usage:
  python scripts/step1c_zone_search.py --iso CAISO
  python scripts/step1c_zone_search.py --iso PJM --auto-commit
  python scripts/step1c_zone_search.py --iso ALL
"""

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import step1_pfs_generator as s1
from step1_prior_windows import load_prior_windows, ZONES

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("ERROR: pyarrow required. pip install pyarrow")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Zone-internal buffer: how much to expand resource bounds from observed
# coarse boundary mixes (absolute percentage points)
ZONE_RESOURCE_BUFFER = 5

# Fine grid step size (percentage points)
FINE_STEP = 1

# Fine grid radius around boundary archetypes when prior windows unavailable
FINE_RADIUS_DEFAULT = 4
FINE_RADIUS_5D = 2  # tighter for CAISO 5D to control combinatorial blowup

# Max fine archetypes per zone (safety cap for 5D ISOs)
MAX_FINE_ARCHETYPES = 2000
MAX_FINE_ARCHETYPES_5D = 500

# Scoring chunk size (mixes per batch for batch_hourly_scores)
SCORE_CHUNK_SIZE = 20000

# Near-miss width for storage sweep (in 0-1 score space)
STORAGE_SWEEP_FLOOR = 0.50

# Flush candidates to disk above this count
CHUNK_CANDIDATE_LIMIT = 500_000

# All active thresholds
ACTIVE_THRESHOLDS = [50, 55, 60, 65, 70, 75, 80, 85, 87.5,
                     90, 92.5, 95, 97.5, 99, 99.5, 99.9, 99.99]

# Low thresholds (coarse only, no fine refinement)
LOW_THRESHOLDS = [10, 20, 30, 40]


def get_near_miss_width(threshold):
    """Near-miss half-width by threshold range (in 0-1 score space)."""
    if threshold >= 95:
        return 0.15
    elif threshold >= 85:
        return 0.30
    else:
        return 0.40


def _row_keys(combos):
    """Vectorized collision-free integer key per row for dedup.

    Resource values are integers 0-300, so base-301 polynomial
    hash is collision-free and fits in int64 for up to 7 dimensions
    (301^7 ≈ 2.2e17 < 2^63 ≈ 9.2e18).
    """
    int_arr = np.round(combos).astype(np.int64)
    n_res = int_arr.shape[1]
    multipliers = np.array([301**i for i in range(n_res)], dtype=np.int64)
    return int_arr @ multipliers


# ══════════════════════════════════════════════════════════════════════════════
# VECTORIZED HASH / DEDUP UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

# Collision-free hashing: base-351 positional encoding.
# Each resource dimension ∈ [0, 350], so 351^i gives unique int64 per mix.
# Max 7 dims × 9 bits = 63 bits → fits int64 with no collisions.
_HASH_BASES = np.array([351**i for i in range(7)], dtype=np.int64)


def _hash_mixes(combos):
    """Vectorized collision-free int64 hash per mix row. No Python loops."""
    n_res = combos.shape[1]
    return combos.astype(np.int64) @ _HASH_BASES[:n_res]


# ══════════════════════════════════════════════════════════════════════════════
# NUMBA JIT DOMINANCE FILTER
# ══════════════════════════════════════════════════════════════════════════════

if s1.HAS_NUMBA:
    from numba import njit as _njit

    @_njit(cache=True)
    def _dominance_filter_jit(combos, order):
        """Numba JIT Pareto front with early-exit inner loop."""
        n = len(combos)
        n_res = combos.shape[1]
        kept = np.ones(n, dtype=np.bool_)
        front = np.empty((n, n_res), dtype=np.float64)
        front_size = 0

        for oi in range(n):
            idx = order[oi]
            dominated = False
            for fi in range(front_size):
                all_leq = True
                strict = False
                for d in range(n_res):
                    if front[fi, d] > combos[idx, d]:
                        all_leq = False
                        break
                    if front[fi, d] < combos[idx, d]:
                        strict = True
                if all_leq and strict:
                    dominated = True
                    break
            if dominated:
                kept[idx] = False
            else:
                front[front_size] = combos[idx]
                front_size += 1
        return kept


# ══════════════════════════════════════════════════════════════════════════════
# ZONE FINE GRID GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_zone_resource_bounds(iso, coarse_combos, coarse_scores,
                                 zone_score_low, zone_score_high,
                                 prior_zone_bounds=None):
    """Compute resource bounds for fine search within a score zone.

    Uses prior EF bounds if available, otherwise derives from coarse boundary
    mixes. Always clamps to resource caps.
    """
    rtypes = s1.get_resource_types(iso)
    n_res = len(rtypes)

    caps = []
    for rt in rtypes:
        if rt == 'hydro':
            caps.append(int(s1.HYDRO_CAPS[iso] + s1.HYDRO_ADDER_PCT))
        elif rt == 'geothermal':
            caps.append(s1.GEO_CAP_PCT)
        elif rt == 'offshore_wind':
            caps.append(int(s1.OFFSHORE_WIND_CAP_PCT.get(iso, 0)))
        else:
            caps.append(s1.RESOURCE_CAPS[rt])

    if prior_zone_bounds:
        # Use prior-informed bounds
        bounds = []
        for i, rt in enumerate(rtypes):
            if rt in prior_zone_bounds:
                lo, hi = prior_zone_bounds[rt]
            else:
                lo, hi = 0, caps[i]
            bounds.append((max(0, lo), min(caps[i], hi)))
        return bounds

    # Derive from coarse boundary mixes
    zone_mask = (coarse_scores >= zone_score_low) & (coarse_scores <= zone_score_high)
    zone_mixes = coarse_combos[zone_mask]

    if len(zone_mixes) == 0:
        return [(0, c) for c in caps]

    bounds = []
    for i in range(n_res):
        col = zone_mixes[:, i]
        lo = max(0, int(np.min(col)) - ZONE_RESOURCE_BUFFER)
        hi = min(caps[i], int(np.max(col)) + ZONE_RESOURCE_BUFFER)
        bounds.append((lo, hi))

    return bounds


def generate_fine_grid(bounds, step=1):
    """Generate Cartesian product of 1%-step ranges within bounds.

    Uses numpy meshgrid instead of itertools.product for speed.
    Chunks by first dimension to limit peak memory for 5D+ grids.

    Args:
        bounds: list of (lo, hi) per resource dimension
        step: step size in percentage points

    Returns:
        numpy array (N, n_res) of mix combinations
    """
    ranges = [np.arange(lo, hi + 1, step, dtype=np.float64) for lo, hi in bounds]

    # Estimate total Cartesian product size
    total_size = 1
    for r in ranges:
        total_size *= len(r)

    if total_size == 0:
        return np.empty((0, len(bounds)), dtype=np.float64)

    if total_size <= 10_000_000:
        # Small enough for full meshgrid in memory
        grids = np.meshgrid(*ranges, indexing='ij')
        combos = np.column_stack([g.ravel() for g in grids])

        row_sums = combos.sum(axis=1)
        mask = (row_sums > 0) & (row_sums <= s1.TOTAL_PROCUREMENT_CAP)
        return combos[mask]

    # Large grid: chunk by first dimension to limit peak memory.
    # Build the (N-1)-dimensional sub-product once, then iterate the first axis.
    first = ranges[0]
    rest = ranges[1:]
    rest_grids = np.meshgrid(*rest, indexing='ij')
    rest_product = np.column_stack([g.ravel() for g in rest_grids])
    rest_sums = rest_product.sum(axis=1)

    parts = []
    for val in first:
        total_sum = val + rest_sums
        mask = (total_sum > 0) & (total_sum <= s1.TOTAL_PROCUREMENT_CAP)
        n_keep = int(np.sum(mask))
        if n_keep > 0:
            col0 = np.full((n_keep, 1), val, dtype=np.float64)
            parts.append(np.hstack([col0, rest_product[mask]]))

    if not parts:
        return np.empty((0, len(bounds)), dtype=np.float64)
    return np.vstack(parts)


def generate_archetype_fine_grid(iso, boundary_mixes, n_res):
    """Generate fine mixes around boundary archetypes (fallback when no prior windows)."""
    radius = FINE_RADIUS_5D if n_res >= 5 else FINE_RADIUS_DEFAULT
    max_arch = MAX_FINE_ARCHETYPES_5D if n_res >= 5 else MAX_FINE_ARCHETYPES

    # Vectorized dedup of boundary mixes to integer archetypes
    int_mixes = boundary_mixes.astype(np.int64)
    if len(int_mixes) > 0:
        hashes = _hash_mixes(int_mixes)
        _, uniq_idx = np.unique(hashes, return_index=True)
        unique_archetypes = int_mixes[uniq_idx]
    else:
        unique_archetypes = int_mixes

    if len(unique_archetypes) > max_arch:
        totals = unique_archetypes.sum(axis=1)
        order = np.argsort(totals)
        unique_archetypes = unique_archetypes[order]
        step_size = max(1, len(unique_archetypes) // max_arch)
        unique_archetypes = unique_archetypes[::step_size][:max_arch]

    # Generate fine grid around each archetype
    parts = []
    for i in range(len(unique_archetypes)):
        base = unique_archetypes[i].astype(np.float64)
        fine = s1.generate_resource_combos_around(base, iso, step=FINE_STEP, radius=radius)
        if len(fine) > 0:
            parts.append(fine)

    if not parts:
        return np.empty((0, n_res), dtype=np.float64)

    combos = np.unique(np.vstack(parts), axis=0)
    return combos


# ══════════════════════════════════════════════════════════════════════════════
# THRESHOLD ASSIGNMENT
# ══════════════════════════════════════════════════════════════════════════════

def assign_to_thresholds(combos, scores, thresholds):
    """Assign scored mixes to thresholds: feasible and near-miss lists.

    Returns:
        feasible: dict[threshold] → list of (mix_array, score) tuples
        near_miss: dict[threshold] → list of (mix_array, score) tuples
    """
    feasible = {t: [] for t in thresholds}
    near_miss = {t: [] for t in thresholds}

    for t in thresholds:
        target = t / 100.0
        nm_width = get_near_miss_width(t)
        nm_floor = max(target - nm_width, STORAGE_SWEEP_FLOOR)

        feas_mask = scores >= target
        nm_mask = (~feas_mask) & (scores >= nm_floor)

        feas_idx = np.where(feas_mask)[0]
        nm_idx = np.where(nm_mask)[0]

        for idx in feas_idx:
            feasible[t].append((combos[idx], scores[idx]))
        for idx in nm_idx:
            near_miss[t].append((combos[idx], scores[idx]))

    return feasible, near_miss


def assign_to_thresholds_vectorized(combos, scores, thresholds):
    """Vectorized threshold assignment — returns index arrays per threshold.

    Much faster than the tuple-based version for large combo sets.

    Returns:
        feasible_indices: dict[threshold] → numpy index array
        near_miss_indices: dict[threshold] → numpy index array
    """
    feasible_indices = {}
    near_miss_indices = {}

    for t in thresholds:
        target = t / 100.0
        nm_width = get_near_miss_width(t)
        nm_floor = max(target - nm_width, STORAGE_SWEEP_FLOOR)

        feas_mask = scores >= target
        nm_mask = (~feas_mask) & (scores >= nm_floor)

        feasible_indices[t] = np.where(feas_mask)[0]
        near_miss_indices[t] = np.where(nm_mask)[0]

    return feasible_indices, near_miss_indices


# ══════════════════════════════════════════════════════════════════════════════
# DOMINANCE FILTER (vectorized wrapper)
# ══════════════════════════════════════════════════════════════════════════════

def dominance_filter_arrays(combos, scores, storage_pcts=None):
    """Dominance filter: Numba JIT with early-exit, numpy fallback.

    A mix is dominated if another mix has all resources <= AND at least one <.
    Returns boolean mask of non-dominated entries.
    """
    n = len(combos)
    if n <= 1:
        return np.ones(n, dtype=bool)

    # Sort by total procurement ascending (leanest first)
    total = combos.sum(axis=1)
    order = np.argsort(total).astype(np.int64)

    if s1.HAS_NUMBA:
        return _dominance_filter_jit(combos.astype(np.float64), order)

    # Numpy fallback (Python outer loop, vectorized inner comparison)
    kept = np.ones(n, dtype=bool)
    n_res = combos.shape[1]
    front = np.empty((min(n, 50000), n_res), dtype=np.float64)
    front_size = 0

    for idx in order:
        mix_i = combos[idx]
        if front_size > 0:
            fslice = front[:front_size]
            leq = np.all(fslice <= mix_i, axis=1)
            if np.any(leq):
                eq = np.all(fslice == mix_i, axis=1)
                if np.any(leq & ~eq):
                    kept[idx] = False
                    continue
        if front_size >= len(front):
            front = np.vstack([front, np.empty((10000, n_res), dtype=np.float64)])
        front[front_size] = mix_i
        front_size += 1

    return kept


# ══════════════════════════════════════════════════════════════════════════════
# SAVE / COMMIT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def save_threshold_pfs(iso, threshold, combos, scores, rtypes):
    """Save feasible mixes for one threshold as parquet."""
    if len(combos) == 0:
        return

    os.makedirs(s1.STEP1_RAW_PFS_PARQUET_DIR, exist_ok=True)
    data = {
        'iso': [iso] * len(combos),
        'threshold': [float(threshold)] * len(combos),
    }
    for i, rt in enumerate(rtypes):
        data[rt] = combos[:, i].astype(np.float64)
    # No storage in zone search output — all zeros
    data['battery_dispatch_pct'] = np.zeros(len(combos), dtype=np.float64)
    data['battery8_dispatch_pct'] = np.zeros(len(combos), dtype=np.float64)
    data['ldes_dispatch_pct'] = np.zeros(len(combos), dtype=np.float64)
    data['h2_dispatch_pct'] = np.zeros(len(combos), dtype=np.float64)
    data['hourly_match_score'] = np.round(scores * 100, 2)

    table = pa.table(data)
    t_str = s1._normalize_threshold_str(threshold)
    out_path = os.path.join(s1.STEP1_RAW_PFS_PARQUET_DIR,
                            f'{iso}_t{t_str}_raw_pfs.parquet')
    pq.write_table(table, out_path, compression='snappy')
    return out_path


def save_near_miss(iso, combos, scores, rtypes):
    """Save union near-miss mixes for step1d storage sweep."""
    if len(combos) == 0:
        return

    os.makedirs(s1.STEP1_RAW_PFS_PARQUET_DIR, exist_ok=True)
    data = {}
    for i, rt in enumerate(rtypes):
        data[rt] = combos[:, i].astype(np.float64)
    data['base_score'] = scores

    table = pa.table(data)
    out_path = os.path.join(s1.STEP1_RAW_PFS_PARQUET_DIR,
                            f'{iso}_near_miss.parquet')
    pq.write_table(table, out_path, compression='snappy')
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  Near-miss union: {len(combos):,} unique mixes → "
          f"{out_path} ({size_mb:.1f} MB)")
    return out_path


def _save_incremental_near_miss(iso, combos_list, scores_list,
                                selected_thresholds, rtypes):
    """Save near_miss after each zone so step1d can start even if run is partial.

    Overwrites the previous file with the union of all near-miss mixes from
    every completed zone so far.
    """
    if not combos_list:
        return
    try:
        cur_combos = np.vstack(combos_list)
        cur_scores = np.concatenate(scores_list)
        _, nm_idx = assign_to_thresholds_vectorized(
            cur_combos, cur_scores, selected_thresholds)
        all_nm: set = set()
        for t in selected_thresholds:
            all_nm.update(nm_idx[t].tolist())
        nm_arr = np.array(sorted(all_nm), dtype=np.int64)
        if len(nm_arr) > 0:
            save_near_miss(iso, cur_combos[nm_arr], cur_scores[nm_arr], rtypes)
            print(f"    [near_miss] Incremental save: {len(nm_arr):,} mixes")
    except Exception as exc:
        print(f"    [near_miss] Incremental save failed: {exc}")


def git_commit_iso_progress(iso, zone_name, n_thresholds, auto_commit):
    """Commit current ISO progress after a zone completes."""
    if not auto_commit:
        return

    try:
        # Stage all PFS files for this ISO
        pfs_dir = s1.STEP1_RAW_PFS_PARQUET_DIR
        result = subprocess.run(
            ['git', 'add', '-A', pfs_dir],
            capture_output=True, text=True)
        if result.returncode != 0:
            return

        # Check for changes
        result = subprocess.run(['git', 'diff', '--cached', '--quiet'],
                                capture_output=True)
        if result.returncode == 0:
            return  # nothing to commit

        msg = (f"PFS zone {zone_name}: {iso} ({n_thresholds} thresholds) — "
               f"auto-commit")
        subprocess.run(['git', 'commit', '-m', msg],
                       check=True, capture_output=True, text=True)

        # Push with retry
        for attempt in range(1, 5):
            result = subprocess.run(
                ['git', 'push', '-u', 'origin', 'HEAD'],
                capture_output=True, text=True)
            if result.returncode == 0:
                print(f"    [auto-commit] Zone {zone_name} committed & pushed")
                return
            if attempt < 4:
                wait = 2 ** attempt
                time.sleep(wait)

        print(f"    [auto-commit] Push failed — committed locally")

    except subprocess.CalledProcessError as e:
        print(f"    [auto-commit] Git error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# MANIFEST (resume support)
# ══════════════════════════════════════════════════════════════════════════════

def _manifest_path(iso):
    return os.path.join(s1.STEP1_RAW_PFS_PARQUET_DIR,
                        f'{iso}_zone_manifest.json')


def _compute_code_hash(iso):
    """Hash source files + coarse cache identity."""
    h = hashlib.sha256()
    for fname in ['step1c_zone_search.py', 'step1_pfs_generator.py']:
        fpath = os.path.join(SCRIPT_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath, 'rb') as f:
                h.update(f.read())
    cache_path = s1._coarse_cache_path(iso)
    if os.path.exists(cache_path):
        stat = os.stat(cache_path)
        h.update(f"{cache_path}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def _load_manifest(iso, current_hash):
    mpath = _manifest_path(iso)
    if not os.path.exists(mpath):
        return None
    try:
        with open(mpath, 'r') as f:
            manifest = json.load(f)
        if manifest.get('code_hash') != current_hash:
            print(f"  Code/data changed — recomputing all zones")
            os.remove(mpath)
            return None
        return manifest
    except (json.JSONDecodeError, OSError):
        return None


def _save_manifest(iso, code_hash, zones_done, thresholds_done):
    os.makedirs(s1.STEP1_RAW_PFS_PARQUET_DIR, exist_ok=True)
    manifest = {
        'code_hash': code_hash,
        'zones_done': sorted(zones_done),
        'thresholds_done': sorted(thresholds_done),
    }
    with open(_manifest_path(iso), 'w') as f:
        json.dump(manifest, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def process_iso(iso, auto_commit=False, selected_thresholds=None,
               max_zone_combos=3_000_000):
    """Run zone-based fine search for one ISO."""
    if selected_thresholds is None:
        selected_thresholds = LOW_THRESHOLDS + ACTIVE_THRESHOLDS

    iso_start = time.time()
    rtypes = s1.get_resource_types(iso)
    n_res = len(rtypes)

    print(f"\n{'=' * 70}")
    print(f"  Step 1c Zone Search — {iso}")
    print(f"  Resources: {n_res}D ({', '.join(rtypes)})")
    print(f"  Numba: {'enabled' if s1.HAS_NUMBA else 'disabled'}")
    print(f"  Thresholds: {selected_thresholds}")
    print(f"  Max zone combos cap: {max_zone_combos:,}")
    print(f"{'=' * 70}")

    # ── Resume logic ──
    code_hash = _compute_code_hash(iso)
    manifest = _load_manifest(iso, code_hash)
    zones_done = set(manifest.get('zones_done', [])) if manifest else set()
    thresholds_done = set(manifest.get('thresholds_done', [])) if manifest else set()

    if zones_done:
        print(f"  Resuming — zones done: {sorted(zones_done)}")

    # ── Determine which zones are needed for selected thresholds ──
    required_zones: set = set()
    for zn, _zlo, _zhi, z_thresholds in ZONES:
        if any(t in selected_thresholds for t in z_thresholds):
            required_zones.add(zn)
    # Zone A also covers low thresholds (coarse-only)
    if any(t in selected_thresholds for t in LOW_THRESHOLDS):
        required_zones.add('A')
    print(f"  Required zones: {sorted(required_zones) or '(none)'}")

    # ── Load coarse cache ──
    print(f"\n  Loading coarse cache...")
    cached = s1.load_coarse_cache(iso)
    if cached is None:
        print(f"  ERROR: No coarse cache for {iso}. Run step1b first.")
        return
    coarse_combos, coarse_scores = cached
    print(f"  Coarse cache: {len(coarse_combos):,} mixes")

    # ── Load EIA data for scoring fine mixes ──
    print(f"  Loading EIA data...")
    demand_data, gen_profiles, _, _ = s1.load_data()
    demand_norm = demand_data[iso]['normalized']
    supply_profiles = s1.get_supply_profiles(iso, gen_profiles)
    demand_arr, supply_matrix = s1.prepare_numpy_profiles(
        iso, demand_norm, supply_profiles)

    # ── Load prior windows (optional) ──
    prior_windows = load_prior_windows(iso)
    if prior_windows:
        print(f"  Prior windows loaded — search space narrowed")
    else:
        print(f"  No prior windows — using coarse-derived bounds")

    # ── JIT warmup ──
    if s1.HAS_NUMBA:
        print(f"  Warming up Numba JIT...")
        _ = s1.batch_hourly_scores(demand_arr, supply_matrix,
                                   coarse_combos[:2])
        print(f"  JIT ready")

    # ── Global tracking ──
    # Vectorized dedup keys (collision-free int64 hash per row)
    global_scored_keys = _row_keys(coarse_combos)

    # Accumulate ALL scored mixes (coarse + fine) with their scores
    all_combos_list = [coarse_combos]
    all_scores_list = [coarse_scores]

    # ── Process zones ──
    os.makedirs(s1.STEP1_RAW_PFS_PARQUET_DIR, exist_ok=True)

    for zone_name, z_score_low, z_score_high, z_thresholds in ZONES:
        if zone_name not in required_zones:
            print(f"\n  Zone {zone_name}: skipped (not needed for selected thresholds)")
            continue
        if zone_name in zones_done:
            print(f"\n  Zone {zone_name}: skipped (already done)")
            continue

        zone_start = time.time()
        print(f"\n  Zone {zone_name}: score [{z_score_low:.2f}, {z_score_high:.2f}]"
              f" → thresholds {z_thresholds}")

        # 1. Get zone-specific resource bounds
        prior_zone_key = f'zone_{zone_name}'
        pzb = None
        if prior_windows and prior_zone_key in prior_windows:
            pzb = prior_windows[prior_zone_key].get('bounds')

        bounds = compute_zone_resource_bounds(
            iso, coarse_combos, coarse_scores,
            z_score_low, z_score_high, pzb)

        bounds_str = ', '.join(f"{rtypes[i]}=[{lo},{hi}]"
                               for i, (lo, hi) in enumerate(bounds))
        print(f"    Bounds: {bounds_str}")

        # 2. Generate fine 1% grid within zone bounds
        #    Pre-check estimated size to avoid OOM / silent hangs on large zones
        est_size = 1
        for lo, hi in bounds:
            est_size *= max(1, int((hi - lo) / FINE_STEP) + 1)
        print(f"    Estimated Cartesian size: {est_size:,}  cap: {max_zone_combos:,}")

        if prior_windows and est_size <= max_zone_combos:
            # Prior-informed Cartesian within zone bounds
            fine_combos = generate_fine_grid(bounds, step=FINE_STEP)
        else:
            # Archetype fallback: enumerate fine grid around coarse boundary mixes.
            # Used when no prior windows exist OR when the Cartesian would exceed cap.
            if prior_windows and est_size > max_zone_combos:
                print(f"    Cartesian exceeds cap — switching to archetype fallback")
            zone_mask = ((coarse_scores >= z_score_low) &
                         (coarse_scores <= z_score_high))
            boundary_mixes = coarse_combos[zone_mask]
            fine_combos = generate_archetype_fine_grid(iso, boundary_mixes, n_res)

        print(f"    Raw fine grid: {len(fine_combos):,} combos")

        # 3. Dedup against global keys (vectorized — no Python loop)
        if len(fine_combos) > 0:
            fine_keys = _row_keys(fine_combos)
            new_mask = ~np.isin(fine_keys, global_scored_keys)
            fine_combos = fine_combos[new_mask]
            if np.any(new_mask):
                global_scored_keys = np.concatenate([
                    global_scored_keys, fine_keys[new_mask]])

        print(f"    After dedup: {len(fine_combos):,} new mixes to score")

        # 4. Score new mixes in chunks (vectorized — NO Python loop over mixes)
        if len(fine_combos) > 0:
            score_start = time.time()
            fine_scores = s1.batch_hourly_scores(
                demand_arr, supply_matrix, fine_combos,
                chunk_size=SCORE_CHUNK_SIZE)
            score_elapsed = time.time() - score_start
            print(f"    Scored {len(fine_combos):,} mixes in {score_elapsed:.1f}s")

            all_combos_list.append(fine_combos)
            all_scores_list.append(fine_scores)

        zone_elapsed = time.time() - zone_start
        print(f"    Zone {zone_name} complete: {zone_elapsed:.1f}s")

        zones_done.add(zone_name)
        _save_manifest(iso, code_hash, zones_done, thresholds_done)

        # Save near_miss incrementally so step1d can start even if run is partial
        _save_incremental_near_miss(iso, all_combos_list, all_scores_list,
                                    selected_thresholds, rtypes)

        # Auto-commit after each zone (includes near_miss)
        git_commit_iso_progress(iso, zone_name, len(z_thresholds), auto_commit)
        gc.collect()

    # ── Combine all scored mixes ──
    print(f"\n  Combining all scored mixes...")
    all_combos = np.vstack(all_combos_list)
    all_scores = np.concatenate(all_scores_list)
    print(f"  Total unique scored mixes: {len(all_combos):,}")

    # ── Assign to thresholds + save ──
    all_thresholds = selected_thresholds
    print(f"\n  Assigning to {len(all_thresholds)} thresholds...")

    # Vectorized assignment
    feasible_idx, near_miss_idx = assign_to_thresholds_vectorized(
        all_combos, all_scores, all_thresholds)

    # Collect union of near-miss mixes (unique, for step1d)
    all_nm_indices = set()
    for t in all_thresholds:
        all_nm_indices.update(near_miss_idx[t].tolist())
    all_nm_indices = np.array(sorted(all_nm_indices), dtype=np.int64)

    # Save near-miss union
    if len(all_nm_indices) > 0:
        save_near_miss(iso, all_combos[all_nm_indices],
                       all_scores[all_nm_indices], rtypes)

    # Per-threshold: dominance filter + save
    for t in all_thresholds:
        if t in thresholds_done:
            print(f"    {iso} t{t}%: skipped (already done)")
            continue

        feas_i = feasible_idx[t]
        if len(feas_i) == 0:
            print(f"    {iso} t{t}%: 0 feasible (no storage)")
            save_threshold_pfs(iso, t, np.empty((0, n_res)), np.empty(0), rtypes)
            thresholds_done.add(t)
            _save_manifest(iso, code_hash, zones_done, thresholds_done)
            continue

        t_combos = all_combos[feas_i]
        t_scores = all_scores[feas_i]

        # Dominance filter
        n_before = len(t_combos)
        kept = dominance_filter_arrays(t_combos, t_scores)
        t_combos = t_combos[kept]
        t_scores = t_scores[kept]
        n_removed = n_before - len(t_combos)

        save_threshold_pfs(iso, t, t_combos, t_scores, rtypes)

        nm_count = len(near_miss_idx[t])
        dom_str = f", {n_removed} dominated" if n_removed > 0 else ""
        print(f"    {iso} t{t}%: {len(t_combos):,} feasible{dom_str}"
              f", {nm_count:,} near-miss")

        thresholds_done.add(t)
        _save_manifest(iso, code_hash, zones_done, thresholds_done)

    # Auto-commit final results
    git_commit_iso_progress(iso, "final", len(all_thresholds), auto_commit)

    iso_elapsed = time.time() - iso_start
    print(f"\n{'=' * 70}")
    print(f"  {iso} COMPLETE — {len(all_combos):,} total scored mixes, "
          f"{len(all_nm_indices):,} near-miss for step1d")
    print(f"  Elapsed: {iso_elapsed:.1f}s")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Step 1c: Zone-based fine search with global dedup.")
    parser.add_argument("--iso", required=True,
                        help="ISO name or 'ALL'")
    parser.add_argument("--auto-commit", action="store_true",
                        help="Commit & push after each zone completes")
    parser.add_argument("--thresholds", default="ALL",
                        help="Comma-separated thresholds (e.g. '90,95,99.99') or 'ALL'")
    parser.add_argument("--max-zone-combos", type=int, default=3_000_000,
                        help="Max Cartesian combos per zone before archetype fallback "
                             "(default: 3M)")
    args = parser.parse_args()

    isos = list(s1.ISOS) if args.iso.upper() == 'ALL' else [args.iso.upper()]

    for iso in isos:
        if iso not in s1.ISOS:
            print(f"ERROR: Unknown ISO '{iso}'")
            sys.exit(1)

    # Parse threshold selection
    all_defined = LOW_THRESHOLDS + ACTIVE_THRESHOLDS
    if args.thresholds.strip().upper() == 'ALL':
        selected = all_defined
    else:
        try:
            selected = [float(t.strip()) for t in args.thresholds.split(',')]
        except ValueError:
            print(f"ERROR: Invalid --thresholds value '{args.thresholds}'")
            sys.exit(1)

    for iso in isos:
        process_iso(iso, auto_commit=args.auto_commit,
                    selected_thresholds=selected,
                    max_zone_combos=args.max_zone_combos)


if __name__ == "__main__":
    main()
