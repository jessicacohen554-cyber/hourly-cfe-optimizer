#!/usr/bin/env python3
"""Step 1b: Zone-based fine search with global deduplication.

Replaces per-threshold fine refinement with score-band zones that eliminate
redundant scoring across thresholds. Each mix is scored exactly ONCE and
assigned to all thresholds where it's feasible or near-miss.

Architecture:
  1. Load scored coarse cache from step1a
  2. For each score-band zone (A/B/C):
     a. Identify coarse boundary mixes in zone's score range
     b. Compute zone-specific resource windows (from prior EF or coarse data)
     c. Generate fine 1% grid within zone bounds
     d. Dedup against global hash set (no mix scored twice)
     e. Score all new mixes ONCE via vectorized batch_hourly_scores
     f. Assign each scored mix to relevant thresholds
  3. Per-threshold: dominance filter + save PFS parquet
  4. Save union near-miss list for step1c storage sweep

Output:
  data/step1-pfs/{ISO}_t{XX}_raw_pfs.parquet  (per-threshold feasible)
  data/step1-pfs/{ISO}_near_miss.parquet       (union near-miss for 1d)

Usage:
  python scripts/step1b_zone_search.py --iso CAISO
  python scripts/step1b_zone_search.py --iso PJM --auto-commit
  python scripts/step1b_zone_search.py --iso ALL
"""

import argparse
import gc
import hashlib
import json
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import step1_pfs_generator as s1
from step1_prior_windows import load_prior_windows, ZONES
from parquet_utils import write_parquet_chunked

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

# Fine grid radius around boundary archetypes when prior windows unavailable.
# For D dimensions with radius R, each archetype produces (2R+1)^D combos.
# 4D: R=4 → 9^4 =  6,561/arch   (×2000 archetypes ≈ 13M → fine)
# 5D: R=2 → 5^5 =  3,125/arch   (× 500 archetypes ≈  1.6M → fine)
# 8D: R=2 → 5^8 = 390,625/arch  (× 500 archetypes ≈ 195M → OOM!)
# 9D: R=2 → 5^9 =  ~2M/arch     (× 500 archetypes ≈  1B  → OOM!)
# Fix: use R=1 for ≥8D (3^9 = 19,683/arch → × 200 archetypes ≈ 4M → safe)
FINE_RADIUS_DEFAULT = 4
FINE_RADIUS_5D = 2   # 5–7D (CAISO etc.)
FINE_RADIUS_8D = 1   # 8–9D (ISOs with offshore_wind + 4 hybrids)

# Max fine archetypes per zone (safety cap by dimensionality)
MAX_FINE_ARCHETYPES = 2000
MAX_FINE_ARCHETYPES_5D = 500
MAX_FINE_ARCHETYPES_8D = 200  # tighter for high-D to keep memory < 2GB

# Hard cap on total combos from archetype expansion (safety net for OOM)
MAX_ARCHETYPE_TOTAL_COMBOS = 5_000_000

# Scoring chunk size (mixes per batch for batch_hourly_scores)
# High-D ISOs (≥8D) use smaller chunks to reduce peak memory during scoring:
# 20K × 8760 × 8 bytes = 1.4 GB per chunk — too much when coarse cache is large.
SCORE_CHUNK_SIZE = 20000
SCORE_CHUNK_SIZE_HIGH_D = 5000

# Max fine grid combos before falling back to archetype-based search.
# 10M keeps Zone A feasible for 4D ISOs (~11-14M after procurement cap)
# while pushing Zones B/C into the fast archetype path.
# Previous value of 50M caused 4D ISOs (ERCOT/MISO/SPP) to score
# 30-46M mixes per zone while 5D ISOs fell back to archetypes.
MAX_GRID_COMBOS_BEFORE_FALLBACK = 10_000_000

# Maximum near-miss mixes per ISO (union across all thresholds)
MAX_NEAR_MISS = 100_000

# Memory cap for full coarse cache load (bytes). If estimated memory exceeds
# this, use streaming mode that loads one part file at a time.
MAX_COARSE_LOAD_BYTES = 5_000_000_000  # 5 GB

# Lower streaming threshold for high-D ISOs (≥8 resources).
# NEISO (28.5M×10 = 2.28 GB) and NYISO (25.5M×10 = 2.04 GB) fit the 5 GB
# threshold but peak memory during full-load processing (near-miss copies,
# archetype generation, scoring) exceeds the 7 GB GitHub runner limit.
MAX_COARSE_LOAD_BYTES_HIGH_D = 2_000_000_000  # 2 GB

# Max boundary mixes to retain per zone during streaming.
# Only used for archetype selection (200 archetypes max for ≥8D).
# 10K provides ample diversity for archetype extraction.
MAX_ZONE_BOUNDARY_SAMPLE = 10_000

# Flush candidates to disk above this count
CHUNK_CANDIDATE_LIMIT = 500_000

# All active thresholds
ACTIVE_THRESHOLDS = [50, 55, 60, 65, 70, 75, 80, 85, 87.5,
                     90, 92.5, 95, 97.5, 99, 99.5, 99.9]

# Low thresholds (coarse only, no fine refinement)
LOW_THRESHOLDS = [10, 20, 30, 40]


def get_near_miss_width(threshold):
    """Near-miss half-width by threshold range (in 0-1 score space).

    High-solar mixes can have low base scores but massive curtailment
    surplus that storage captures, so sub-85% thresholds need a wider
    window.  Step1d applies a surplus >= gap filter at binning time to
    prevent candidate bloat even with wider windows here.
    """
    if threshold >= 99:
        return 0.20   # 20pp — last-mile, few mixes
    elif threshold >= 85:
        return 0.20   # 20pp
    return 0.25        # 25pp — high-solar + storage mixes need wider window


# ══════════════════════════════════════════════════════════════════════════════
# VECTORIZED HASH / DEDUP UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

# Collision-free hashing: base-351 positional encoding.
# Each resource dimension ∈ [0, 350], so 351^i gives unique int64 per mix.
# 351^7 ≈ 8.0e17 < 2^63 ≈ 9.2e18 — fits int64 with no collisions.
# For >7 dims (hybrid), we use two independent 6-dim hash lanes.
_HASH_BASES = np.array([351**i for i in range(7)], dtype=np.int64)


def _hash_mixes(combos):
    """Vectorized collision-free int64 hash per mix row. No Python loops.

    For >7 dimensions, uses two independent hash lanes (split at dim 6)
    to stay within int64 range while remaining collision-free.
    Returns (N,) int64 array.

    Memory-optimised: avoids a full int64 copy of the combos array by
    computing lane hashes from column slices, keeping peak memory at
    ~N×8 bytes instead of ~N×dims×8 bytes (saves ~2 GB for 28M×9 caches).
    """
    n_res = combos.shape[1]
    if n_res <= 7:
        return combos.astype(np.int64) @ _HASH_BASES[:n_res]
    # Two-lane hash: compute directly from column slices to avoid
    # materialising a full (N, dims) int64 copy.
    # Lane A: dims 0-5 (351^6 ≈ 1.87e15, safe)
    lane_a = combos[:, :6].astype(np.int64) @ _HASH_BASES[:6]
    # Lane B: dims 6+ (≤5 dims, 351^5 ≈ 5.3e12, safe)
    lane_b = combos[:, 6:].astype(np.int64) @ _HASH_BASES[:n_res - 6]
    # Combine: lane_a fits in ~51 bits, lane_b in ~43 bits
    # Use XOR with shifted lane_b for collision-free combination in int64
    return lane_a ^ (lane_b << np.int64(20))


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

def _tighten_bounds_by_procurement_cap(bounds):
    """Tighten per-resource upper bounds using TOTAL_PROCUREMENT_CAP.

    For each resource i, hi[i] <= cap - sum(lo[j] for j != i).
    E.g., if cap=350 and other resources' minimums sum to 0, wind can be 250
    (its own cap). But if clean_firm min=10, solar min=5, hydro min=2, then
    wind <= 350-10-5-2 = 333 (may still be capped by resource cap).

    Iterates until convergence since tightening one bound can allow others
    to tighten further.
    """
    pcap = s1.TOTAL_PROCUREMENT_CAP
    n = len(bounds)
    bounds = list(bounds)  # make mutable copy

    for _iteration in range(5):  # converges in 2-3 iterations
        changed = False
        sum_lo = sum(lo for lo, hi in bounds)
        for i in range(n):
            lo_i, hi_i = bounds[i]
            other_min = sum_lo - lo_i
            new_hi = min(hi_i, pcap - other_min)
            if new_hi < hi_i:
                bounds[i] = (lo_i, max(lo_i, new_hi))
                changed = True
        if not changed:
            break

    return bounds


def compute_zone_resource_bounds(iso, coarse_combos, coarse_scores,
                                 zone_score_low, zone_score_high,
                                 prior_zone_bounds=None,
                                 include_hybrids=False):
    """Compute resource bounds for fine search within a score zone.

    Uses prior EF bounds if available, otherwise derives from coarse boundary
    mixes. Always clamps to resource caps and TOTAL_PROCUREMENT_CAP.
    """
    rtypes = s1.get_resource_types(iso, include_hybrids=include_hybrids)
    n_res = len(rtypes)

    caps = []
    for rt in rtypes:
        if rt == 'hydro':
            caps.append(int(s1.HYDRO_CAPS[iso] + s1.HYDRO_ADDER_PCT))
        elif rt == 'geothermal':
            caps.append(s1.GEO_CAP_PCT)
        elif rt == 'offshore_wind':
            caps.append(int(s1.OFFSHORE_WIND_CAP_PCT.get(iso, 0)))
        elif rt in s1.HYBRID_TYPES:
            caps.append(s1.HYBRID_MAX_PER_TYPE)
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
        return _tighten_bounds_by_procurement_cap(bounds)

    # Derive from coarse boundary mixes
    zone_mask = (coarse_scores >= zone_score_low) & (coarse_scores <= zone_score_high)
    zone_mixes = coarse_combos[zone_mask]

    if len(zone_mixes) == 0:
        return _tighten_bounds_by_procurement_cap([(0, c) for c in caps])

    bounds = []
    for i in range(n_res):
        col = zone_mixes[:, i]
        lo = max(0, int(np.min(col)) - ZONE_RESOURCE_BUFFER)
        hi = min(caps[i], int(np.max(col)) + ZONE_RESOURCE_BUFFER)
        bounds.append((lo, hi))

    return _tighten_bounds_by_procurement_cap(bounds)


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

    if total_size > MAX_GRID_COMBOS_BEFORE_FALLBACK:
        print(f"    WARNING: Fine grid too large ({total_size:,.0f} combos > "
              f"{MAX_GRID_COMBOS_BEFORE_FALLBACK:,}) — falling back to "
              f"archetype search")
        return None  # Caller handles fallback

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


def generate_archetype_fine_grid(iso, boundary_mixes, n_res, include_hybrids=False):
    """Generate fine mixes around boundary archetypes (fallback when no prior windows).

    Dimension-aware radius and archetype caps prevent OOM on high-D ISOs:
      4D: radius=4, 2000 archetypes  →  ~13M combos max
      5-7D: radius=2, 500 archetypes →  ~1.6M combos max
      8-9D: radius=1, 200 archetypes →  ~4M combos max
    """
    if n_res >= 8:
        radius = FINE_RADIUS_8D
        max_arch = MAX_FINE_ARCHETYPES_8D
    elif n_res >= 5:
        radius = FINE_RADIUS_5D
        max_arch = MAX_FINE_ARCHETYPES_5D
    else:
        radius = FINE_RADIUS_DEFAULT
        max_arch = MAX_FINE_ARCHETYPES

    per_arch_estimate = (2 * radius + 1) ** n_res
    print(f"    Archetype grid: {n_res}D, radius={radius}, "
          f"~{per_arch_estimate:,}/arch, max_arch={max_arch}")

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

    print(f"    Using {len(unique_archetypes)} archetypes from "
          f"{len(int_mixes):,} boundary mixes")

    # Generate fine grid around each archetype, with streaming dedup.
    # Previous approach: vstack all parts then np.unique — OOM on large arrays
    # (4M × 9 float64 = 288 MB data + ~600 MB sort temp).
    # New approach: hash-based dedup per archetype, no global sort needed.
    parts = []
    seen_hashes = set()
    total_raw = 0
    total_unique = 0
    for i in range(len(unique_archetypes)):
        base = unique_archetypes[i].astype(np.float64)
        fine = s1.generate_resource_combos_around(
            base, iso, step=FINE_STEP, radius=radius,
            include_hybrids=include_hybrids)
        if len(fine) > 0:
            total_raw += len(fine)
            # Streaming dedup: hash each batch and keep only unseen
            hashes = _hash_mixes(fine)
            new_mask = np.array([h not in seen_hashes for h in hashes],
                                dtype=bool)
            if np.any(new_mask):
                seen_hashes.update(hashes[new_mask].tolist())
                parts.append(fine[new_mask])
                total_unique += int(new_mask.sum())
        if total_unique >= MAX_ARCHETYPE_TOTAL_COMBOS:
            print(f"    Hit {MAX_ARCHETYPE_TOTAL_COMBOS:,} combo cap at "
                  f"archetype {i+1}/{len(unique_archetypes)}")
            break

    if not parts:
        return np.empty((0, n_res), dtype=np.float64)

    combos = np.vstack(parts)
    del parts, seen_hashes
    print(f"    Archetype grid: {len(combos):,} unique combos "
          f"(from {total_raw:,} raw)")
    return combos


# ══════════════════════════════════════════════════════════════════════════════
# THRESHOLD ASSIGNMENT
# ══════════════════════════════════════════════════════════════════════════════

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
        nm_floor = max(target - nm_width, 0.50)

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

def _build_pfs_table(iso, threshold, combos, scores, rtypes):
    """Build a pyarrow Table for PFS mixes (shared by save and batch-save)."""
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
    return pa.table(data)


def save_threshold_pfs(iso, threshold, combos, scores, rtypes, max_file_mb=45):
    """Save feasible mixes for one threshold as parquet (chunked if >max_file_mb)."""
    if len(combos) == 0:
        return

    os.makedirs(s1.STEP1_RAW_PFS_PARQUET_DIR, exist_ok=True)
    table = _build_pfs_table(iso, threshold, combos, scores, rtypes)
    t_str = s1._normalize_threshold_str(threshold)
    out_path = os.path.join(s1.STEP1_RAW_PFS_PARQUET_DIR,
                            f'{iso}_t{t_str}_raw_pfs.parquet')
    written = write_parquet_chunked(table, out_path, max_mb=max_file_mb,
                                    compression='snappy')
    return written[0] if len(written) == 1 else written


def save_threshold_pfs_batch(iso, threshold, combos, scores, rtypes,
                              batch_idx, max_file_mb=45):
    """Save feasible mixes as a batch file: {ISO}_t{XX}_raw_pfs_b{N}.parquet.

    Step 2.1 already handles this naming pattern and merges batches per
    ISO/threshold. Each zone's results get their own batch number so
    partial results survive timeouts.
    """
    if len(combos) == 0:
        return None

    os.makedirs(s1.STEP1_RAW_PFS_PARQUET_DIR, exist_ok=True)
    table = _build_pfs_table(iso, threshold, combos, scores, rtypes)
    t_str = s1._normalize_threshold_str(threshold)
    out_path = os.path.join(s1.STEP1_RAW_PFS_PARQUET_DIR,
                            f'{iso}_t{t_str}_raw_pfs_b{batch_idx}.parquet')
    written = write_parquet_chunked(table, out_path, max_mb=max_file_mb,
                                    compression='snappy')
    return written[0] if len(written) == 1 else written


def _has_curtailment_mask(combos, demand_arr, supply_matrix, chunk_size=5000):
    """Vectorized curtailment check: True if any hour has supply > demand.

    Processes in chunks to limit peak memory (combos × 8760 can be large).
    """
    n = len(combos)
    mask = np.empty(n, dtype=np.bool_)
    for cs in range(0, n, chunk_size):
        ce = min(cs + chunk_size, n)
        fracs = combos[cs:ce].astype(np.float64) / 100.0
        supply = fracs @ supply_matrix              # (chunk, 8760)
        surplus = supply - demand_arr[np.newaxis, :]  # broadcast
        mask[cs:ce] = np.any(surplus > 0, axis=1)
    return mask


def _get_storage_constants():
    """Extract storage dispatch constants from step1_pfs_generator.

    Returns a dict of scalar parameters used by _batch_mixes_storage_screen.
    Centralized here to avoid duplicating extraction across step1b and step1c.
    """
    return {
        'batt_eff': s1.BATTERY_EFFICIENCY,
        'batt8_eff': s1.BATTERY8_EFFICIENCY,
        'ldes_eff': s1.LDES_EFFICIENCY,
        'batt4_dur': s1.BATTERY_DURATION_HOURS,
        'batt8_dur': s1.BATTERY8_DURATION_HOURS,
        'ldes_dur': s1.LDES_DURATION_HOURS,
        'ldes_window': s1.LDES_WINDOW_DAYS * 24,
        'batt8_window': 48,
        'h2_eff': s1.H2_EFFICIENCY,
        'h2_dur': float(s1.H2_DURATION_HOURS),
        'h2_window': s1.H2_WINDOW_DAYS * 24,
    }


def _compute_max_storage_scores(combos, demand_arr, supply_matrix, chunk_size=500):
    """Max-storage ceiling score for each mix (same kernel as step1c Pass 0).

    Returns float64 array (N,) — best score achievable with maximum storage.
    Mixes with max_score >= 0.50 necessarily have curtailment
    (subsumes the curtailment check). Uses same constants and Numba kernel
    as step1c, so the math is identical.

    Always includes H2 (MAX_H2=[25.0]) for a conservative ceiling — step1c
    still gates H2 by threshold in its Pass 1 sweep.
    """
    n = len(combos)
    max_scores = np.empty(n, dtype=np.float64)

    sc = _get_storage_constants()

    # Max storage arrays (same as step1c Pass 0)
    MAX_BAT4 = np.array([0.5], dtype=np.float64)
    MAX_BAT8 = np.array([1.0], dtype=np.float64)
    MAX_LDES = np.array([5.0], dtype=np.float64)
    MAX_H2 = np.array([25.0], dtype=np.float64)

    t0 = time.time()
    for cs in range(0, n, chunk_size):
        ce = min(cs + chunk_size, n)
        fracs = combos[cs:ce].astype(np.float64) / 100.0
        supply = fracs @ supply_matrix
        n_batch = ce - cs

        result = s1._batch_mixes_storage_screen(
            demand_arr, supply, 1.0, n_batch,
            MAX_BAT4, MAX_BAT8, MAX_LDES,
            1, 1, 1,
            sc['batt_eff'], sc['batt8_eff'], sc['ldes_eff'],
            sc['batt4_dur'], sc['batt8_dur'], sc['ldes_dur'],
            sc['ldes_window'], sc['batt8_window'],
            MAX_H2, 1, sc['h2_eff'], sc['h2_dur'], sc['h2_window'])

        max_scores[cs:ce] = result[:, 0]

    elapsed = time.time() - t0
    print(f"    Max-storage screen: {n:,} mixes in {elapsed:.1f}s")
    return max_scores


def _stratified_sample(combos, scores, max_storage_scores, rtypes, max_n):
    """Proportional sample by archetype ensuring diversity.

    Archetypes: solar-heavy (solar>40%), wind-heavy (RE>70% & solar<=40%),
                balanced (30-70% RE), high-firm (<30% RE).
    Within each archetype, select by max_storage_score descending.
    """
    total = combos.sum(axis=1).astype(np.float64)
    solar_idx = rtypes.index('solar')
    wind_idx = rtypes.index('wind')
    solar_frac = combos[:, solar_idx] / np.maximum(total, 1.0)
    re_frac = (combos[:, solar_idx] + combos[:, wind_idx]) / np.maximum(total, 1.0)

    # Assign archetypes: 0=solar-heavy, 1=wind-heavy, 2=balanced, 3=high-firm
    arch = np.full(len(combos), 2, dtype=np.int8)
    arch[solar_frac > 0.40] = 0
    arch[(re_frac > 0.70) & (solar_frac <= 0.40)] = 1
    arch[re_frac < 0.30] = 3

    result = []
    n_total = len(combos)
    for a in range(4):
        mask = arch == a
        n_in_arch = int(mask.sum())
        if n_in_arch == 0:
            continue
        n_alloc = max(1, int(round(max_n * n_in_arch / n_total)))
        n_alloc = min(n_alloc, n_in_arch)
        arch_idx = np.where(mask)[0]
        top_k = np.argsort(max_storage_scores[arch_idx])[-n_alloc:]
        result.append(arch_idx[top_k])

    combined = np.concatenate(result)
    if len(combined) > max_n:
        keep = np.argsort(max_storage_scores[combined])[-max_n:]
        combined = combined[keep]
    return combined


def save_near_miss(iso, combos, scores, rtypes,
                   demand_arr=None, supply_matrix=None,
                   full_filter=False):
    """Save union near-miss mixes for step1c storage sweep.

    Args:
        full_filter: If True (final save), compute max-storage scores,
            apply threshold-crossing filter (keep only mixes where
            max_storage_score crosses at least one threshold above
            base_score), then cap at MAX_NEAR_MISS with stratified
            archetype sampling. No max_storage_score column persisted.
            If False (interim saves), use the cheaper curtailment-only check.
    """
    if len(combos) == 0:
        return

    n_before = len(combos)

    if full_filter and demand_arr is not None and supply_matrix is not None:
        # 1. Compute max-storage ceiling scores
        print(f"  Running max-storage pre-filter on {n_before:,} near-miss mixes...")
        max_storage_scores = _compute_max_storage_scores(
            combos, demand_arr, supply_matrix)

        # 2. Threshold-crossing filter: keep only mixes where
        #    max_storage_score >= some threshold T > base_score
        targets = np.array([t / 100.0 for t in ACTIVE_THRESHOLDS])
        crosses_any = np.zeros(len(combos), dtype=bool)
        for target in targets:
            crosses_any |= (scores < target) & (max_storage_scores >= target)
        combos = combos[crosses_any]
        scores = scores[crosses_any]
        max_storage_scores = max_storage_scores[crosses_any]
        n_after_cross = len(combos)
        print(f"  Threshold-crossing filter: {n_before:,} → {n_after_cross:,} "
              f"({n_before - n_after_cross:,} pruned)")

        # 3. Cap at MAX_NEAR_MISS with stratified archetype sampling
        if len(combos) > MAX_NEAR_MISS:
            idx = _stratified_sample(combos, scores, max_storage_scores,
                                     rtypes, MAX_NEAR_MISS)
            combos = combos[idx]
            scores = scores[idx]
            print(f"  Stratified cap: {n_after_cross:,} → {len(combos):,}")

    elif demand_arr is not None and supply_matrix is not None:
        # Cheap curtailment-only filter (interim saves)
        mask = _has_curtailment_mask(combos, demand_arr, supply_matrix)
        combos = combos[mask]
        scores = scores[mask]
        n_pruned = n_before - len(combos)
        if n_pruned > 0:
            print(f"  Near-miss curtailment filter: {n_before:,} → "
                  f"{len(combos):,} ({n_pruned:,} pruned, no curtailment)")

    if len(combos) == 0:
        print(f"  Near-miss union: 0 mixes after filter (skipped)")
        return

    # Save WITHOUT max_storage_score column (step1c computes its own Pass 0)
    os.makedirs(s1.STEP1_RAW_PFS_PARQUET_DIR, exist_ok=True)
    data = {}
    for i, rt in enumerate(rtypes):
        data[rt] = combos[:, i].astype(np.float64)
    data['base_score'] = scores

    table = pa.table(data)
    out_path = os.path.join(s1.STEP1_RAW_PFS_PARQUET_DIR,
                            f'{iso}_near_miss.parquet')
    written = write_parquet_chunked(table, out_path, max_mb=45,
                                    compression='snappy')
    total_mb = sum(os.path.getsize(p) / (1024*1024) for p in written)
    print(f"  Near-miss union: {len(combos):,} mixes → "
          f"{len(written)} file(s) ({total_mb:.1f} MB)")
    return written[0] if len(written) == 1 else written


def _collect_near_miss_indices(combos, scores, thresholds):
    """Return sorted unique near-miss index array across all thresholds.

    Memory-efficient: uses a single boolean mask union instead of
    materializing per-threshold index arrays (which OOM on 28M+ row
    caches in 7 GB GitHub runners).
    """
    n = len(scores)
    nm_union = np.zeros(n, dtype=bool)
    for t in thresholds:
        target = t / 100.0
        nm_width = get_near_miss_width(t)
        nm_floor = max(target - nm_width, 0.50)
        # near-miss = below target but above floor
        nm_mask = (scores < target) & (scores >= nm_floor)
        nm_union |= nm_mask
    return np.where(nm_union)[0]


def _cleanup_batch_files(iso):
    """Remove batch files after canonical files have been written.

    When all zones complete, canonical {ISO}_t{XX}_raw_pfs.parquet files
    contain the full dominance-filtered results. The intermediate batch
    files (_raw_pfs_b{N}.parquet) are no longer needed.
    """
    import glob as globmod
    pattern = os.path.join(s1.STEP1_RAW_PFS_PARQUET_DIR,
                           f'{iso}_t*_raw_pfs_b*.parquet')
    batch_files = globmod.glob(pattern)
    if batch_files:
        for f in batch_files:
            os.remove(f)
        print(f"  Cleaned up {len(batch_files)} batch files")


def save_near_miss_interim(iso, all_combos_lists, all_scores_lists, rtypes,
                           thresholds_to_use,
                           coarse_nm_combos=None, coarse_nm_scores=None):
    """Save near-miss parquet from accumulated scored mixes so far.

    Called after each zone so step1c has a valid near-miss file even if the
    job times out before all zones complete. Skips the curtailment filter —
    it's wasted work since the final save applies a stricter max-storage
    filter that subsumes it.

    coarse_nm_combos/scores: pre-computed coarse near-miss (passed through
    to avoid re-vstacking the full coarse cache which causes OOM).
    """
    parts_c, parts_s = [], []

    # Include pre-computed coarse near-miss
    if coarse_nm_combos is not None and len(coarse_nm_combos) > 0:
        parts_c.append(coarse_nm_combos)
        parts_s.append(coarse_nm_scores)

    # Compute fine near-miss
    if all_combos_lists:
        fine_combos = np.vstack(all_combos_lists)
        fine_scores = np.concatenate(all_scores_lists)
        nm_arr = _collect_near_miss_indices(
            fine_combos, fine_scores, thresholds_to_use)
        if len(nm_arr) > 0:
            parts_c.append(fine_combos[nm_arr])
            parts_s.append(fine_scores[nm_arr])

    if not parts_c:
        return

    all_nm_combos = np.vstack(parts_c)
    all_nm_scores = np.concatenate(parts_s)
    save_near_miss(iso, all_nm_combos, all_nm_scores, rtypes)


# NOTE: Auto-commit to git removed — all outputs are written locally.


# ══════════════════════════════════════════════════════════════════════════════
# STREAMING COARSE CACHE LOADER (for caches that exceed memory)
# ══════════════════════════════════════════════════════════════════════════════

def _estimate_coarse_memory(iso):
    """Estimate memory (bytes) needed to load full coarse cache as float64."""
    paths = s1.coarse_cache_paths(iso)
    if not paths:
        return 0
    total_rows = 0
    n_cols = 0
    for p in paths:
        pf = pq.ParquetFile(p)
        total_rows += pf.metadata.num_rows
        n_cols = len(pf.schema_arrow.names)
    return total_rows * n_cols * 8


def _stream_coarse_cache(iso, rtypes, zones, active_thresholds):
    """Stream coarse cache part-by-part, extracting only what's needed.

    Memory-efficient: instead of loading 87M×11 array (7+ GB for CAISO),
    processes each parquet part independently and accumulates only:
      1. Per-zone resource bounds (incremental min/max — negligible memory)
      2. Per-zone archetype samples (capped at MAX_ZONE_BOUNDARY_SAMPLE)
      3. Coarse near-miss mixes (rows near any threshold)
      4. A few rows for JIT warmup

    Does NOT accumulate global coarse hash keys (was 700 MB for CAISO).
    Fine-vs-coarse dedup is skipped — step 2.1 deduplicates downstream.
    Inter-zone fine dedup is handled in process_iso() with a much smaller set.

    Returns:
        dict with keys:
          'zone_bounds': dict[zone_name] → list of (lo, hi) per resource
          'zone_archetypes': dict[zone_name] → (combos, scores) capped sample
          'coarse_nm_combos': near-miss combos array
          'coarse_nm_scores': near-miss scores array
          'warmup_rows': 2-row combo array for JIT warmup
          'total_rows': total row count
    """
    paths = s1.coarse_cache_paths(iso)
    if not paths:
        return None

    n_res = len(rtypes)

    # Incremental per-zone bounds (min/max per resource)
    zone_mins = {z[0]: np.full(n_res, np.inf) for z in zones}
    zone_maxs = {z[0]: np.full(n_res, -np.inf) for z in zones}
    zone_counts = {z[0]: 0 for z in zones}

    # Reservoir samples for archetype selection (capped per zone)
    zone_sample_c = {z[0]: [] for z in zones}  # combos parts
    zone_sample_s = {z[0]: [] for z in zones}  # scores parts
    zone_sample_n = {z[0]: 0 for z in zones}   # total seen per zone

    nm_parts_c, nm_parts_s = [], []
    warmup_rows = None
    total_rows = 0

    print(f"    Streaming {len(paths)} part files...")
    for pi, path in enumerate(paths):
        table = pq.read_table(path)
        combos = np.column_stack([table.column(rt).to_numpy() for rt in rtypes])
        scores = table.column('score').to_numpy()
        n_part = len(combos)
        total_rows += n_part

        # Warmup rows (just need 2)
        if warmup_rows is None and n_part >= 2:
            warmup_rows = combos[:2].copy()

        # Per-zone: update bounds incrementally + reservoir sample
        for zone_name, z_lo, z_hi, _ in zones:
            mask = (scores >= z_lo) & (scores <= z_hi)
            n_in = int(mask.sum())
            if n_in == 0:
                continue

            z_combos = combos[mask]
            z_scores = scores[mask]

            # Update incremental min/max
            np.minimum(zone_mins[zone_name],
                       z_combos.min(axis=0), out=zone_mins[zone_name])
            np.maximum(zone_maxs[zone_name],
                       z_combos.max(axis=0), out=zone_maxs[zone_name])
            zone_counts[zone_name] += n_in

            # Reservoir sample for archetype selection
            prev_n = zone_sample_n[zone_name]
            cap = MAX_ZONE_BOUNDARY_SAMPLE
            if prev_n < cap:
                # Still filling — take up to remaining capacity
                take = min(n_in, cap - prev_n)
                zone_sample_c[zone_name].append(z_combos[:take])
                zone_sample_s[zone_name].append(z_scores[:take])
            else:
                # Reservoir replacement: each new row has cap/(prev_n+i)
                # probability of inclusion. For efficiency, compute how
                # many replacements to make and pick random positions.
                n_replace = 0
                for k in range(n_in):
                    j = prev_n + k
                    if np.random.randint(0, j + 1) < cap:
                        n_replace += 1
                if n_replace > 0:
                    # Pick n_replace random rows from this batch and
                    # random positions in the sample to replace
                    src_idx = np.random.choice(n_in, size=min(n_replace, n_in),
                                               replace=False)
                    # Materialize current sample for replacement
                    if len(zone_sample_c[zone_name]) > 1:
                        zone_sample_c[zone_name] = [
                            np.vstack(zone_sample_c[zone_name])]
                        zone_sample_s[zone_name] = [
                            np.concatenate(zone_sample_s[zone_name])]
                    sample_arr = zone_sample_c[zone_name][0]
                    sample_sc = zone_sample_s[zone_name][0]
                    dst_idx = np.random.choice(cap, size=len(src_idx),
                                               replace=False)
                    sample_arr[dst_idx] = z_combos[src_idx]
                    sample_sc[dst_idx] = z_scores[src_idx]
            zone_sample_n[zone_name] = prev_n + n_in

            del z_combos, z_scores

        # Near-miss mixes
        nm_mask = np.zeros(n_part, dtype=bool)
        for t in active_thresholds:
            target = t / 100.0
            nm_width = get_near_miss_width(t)
            nm_floor = max(target - nm_width, 0.50)
            nm_mask |= (scores < target) & (scores >= nm_floor)
        if nm_mask.any():
            nm_parts_c.append(combos[nm_mask])
            nm_parts_s.append(scores[nm_mask])

        # Free this part's full arrays
        del table, combos, scores
        gc.collect()
        print(f"      Part {pi+1}/{len(paths)}: {total_rows:,} rows processed")

    # Assemble per-zone bounds (with buffer + caps)
    caps = []
    for rt in rtypes:
        if rt == 'hydro':
            caps.append(int(s1.HYDRO_CAPS[iso] + s1.HYDRO_ADDER_PCT))
        elif rt == 'geothermal':
            caps.append(s1.GEO_CAP_PCT)
        elif rt == 'offshore_wind':
            caps.append(int(s1.OFFSHORE_WIND_CAP_PCT.get(iso, 0)))
        elif rt in s1.HYBRID_TYPES:
            caps.append(s1.HYBRID_MAX_PER_TYPE)
        else:
            caps.append(s1.RESOURCE_CAPS[rt])

    zone_bounds = {}
    for zone_name in zone_mins:
        if zone_counts[zone_name] == 0:
            zone_bounds[zone_name] = [(0, c) for c in caps]
        else:
            bounds = []
            for i in range(n_res):
                lo = max(0, int(zone_mins[zone_name][i]) - ZONE_RESOURCE_BUFFER)
                hi = min(caps[i], int(zone_maxs[zone_name][i]) + ZONE_RESOURCE_BUFFER)
                bounds.append((lo, hi))
            zone_bounds[zone_name] = _tighten_bounds_by_procurement_cap(bounds)
    del zone_mins, zone_maxs, zone_counts

    # Assemble zone archetype samples
    zone_archetypes = {}
    for zone_name in zone_sample_c:
        cl = zone_sample_c[zone_name]
        sl = zone_sample_s[zone_name]
        if cl:
            zone_archetypes[zone_name] = (np.vstack(cl), np.concatenate(sl))
        else:
            zone_archetypes[zone_name] = (np.empty((0, n_res), dtype=np.float64),
                                           np.empty(0, dtype=np.float64))
    del zone_sample_c, zone_sample_s, zone_sample_n

    if nm_parts_c:
        coarse_nm_combos = np.vstack(nm_parts_c)
        coarse_nm_scores = np.concatenate(nm_parts_s)
    else:
        coarse_nm_combos = np.empty((0, n_res), dtype=np.float64)
        coarse_nm_scores = np.empty(0, dtype=np.float64)
    del nm_parts_c, nm_parts_s

    gc.collect()
    nm_mb = coarse_nm_combos.nbytes / (1024**2) if len(coarse_nm_combos) > 0 else 0
    arch_total = sum(len(v[0]) for v in zone_archetypes.values())
    print(f"    Streaming complete: {total_rows:,} rows, "
          f"{arch_total:,} archetype samples, "
          f"{len(coarse_nm_combos):,} near-miss ({nm_mb:.0f} MB)")

    return {
        'zone_bounds': zone_bounds,
        'zone_archetypes': zone_archetypes,
        'coarse_nm_combos': coarse_nm_combos,
        'coarse_nm_scores': coarse_nm_scores,
        'warmup_rows': warmup_rows if warmup_rows is not None else np.zeros((2, n_res)),
        'total_rows': total_rows,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MANIFEST (resume support)
# ══════════════════════════════════════════════════════════════════════════════

def _manifest_path(iso):
    return os.path.join(s1.STEP1_RAW_PFS_PARQUET_DIR,
                        f'{iso}_zone_manifest.json')


def _compute_code_hash(iso):
    """Hash source files + coarse cache identity."""
    h = hashlib.sha256()
    for fname in ['step1b_zone_search.py', 'step1_pfs_generator.py']:
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

def process_iso(iso, thresholds_filter=None, zones_filter=None,
                include_hybrids=False):
    """Run zone-based fine search for one ISO.

    Args:
        thresholds_filter: list of thresholds to process (None = all)
        zones_filter: set of zone names to run, e.g. {'B', 'C'} (None = all)
        include_hybrids: if True, use hybrid resource types (8-10D)
    """
    iso_start = time.time()

    # ── Auto-detect hybrid mode from coarse cache (supports multi-part files) ──
    coarse_schema = s1.read_coarse_cache_schema(iso)
    if not include_hybrids and coarse_schema is not None:
        if 'solar_batt4' in coarse_schema.names:
            include_hybrids = True
            print(f"  Auto-detected hybrid columns in coarse cache")

    rtypes = s1.get_resource_types(iso, include_hybrids=include_hybrids)
    n_res = len(rtypes)

    hybrid_str = " [HYBRID]" if include_hybrids else ""
    print(f"\n{'=' * 70}")
    print(f"  Step 1b Zone Search — {iso}{hybrid_str}")
    print(f"  Resources: {n_res}D ({', '.join(rtypes)})")
    print(f"  Numba: {'enabled' if s1.HAS_NUMBA else 'disabled'}")
    print(f"{'=' * 70}")

    # ── Resume logic ──
    code_hash = _compute_code_hash(iso)
    manifest = _load_manifest(iso, code_hash)
    zones_done = set(manifest.get('zones_done', [])) if manifest else set()
    thresholds_done = set(manifest.get('thresholds_done', [])) if manifest else set()

    if zones_done:
        print(f"  Resuming — zones done: {sorted(zones_done)}")

    # ── Determine which thresholds / zones to run ──
    all_thresholds = LOW_THRESHOLDS + ACTIVE_THRESHOLDS
    active_thresholds = thresholds_filter if thresholds_filter else all_thresholds

    # ── Load EIA data for scoring fine mixes ──
    print(f"  Loading EIA data...")
    demand_data, gen_profiles, _, _ = s1.load_data()
    demand_norm = demand_data[iso]['normalized']
    supply_profiles = s1.get_supply_profiles(iso, gen_profiles)

    # Load hybrid profiles if needed
    hybrid_profiles = None
    if include_hybrids:
        hybrid_profiles = s1.load_hybrid_profiles(iso)
        print(f"  Loaded hybrid profiles: {list(hybrid_profiles.keys())}")

    demand_arr, supply_matrix = s1.prepare_numpy_profiles(
        iso, demand_norm, supply_profiles,
        include_hybrids=include_hybrids,
        hybrid_profiles=hybrid_profiles)

    # ── Load prior windows (optional) ──
    prior_windows = load_prior_windows(iso)
    if prior_windows:
        print(f"  Prior windows loaded — search space narrowed")
    else:
        print(f"  No prior windows — using coarse-derived bounds")

    # ── Load coarse cache — streaming or full depending on memory ──
    est_mem = _estimate_coarse_memory(iso)
    mem_cap = MAX_COARSE_LOAD_BYTES_HIGH_D if n_res >= 8 else MAX_COARSE_LOAD_BYTES
    use_streaming = est_mem > mem_cap
    print(f"\n  Coarse cache estimated memory: {est_mem / (1024**3):.2f} GB"
          f" (threshold: {mem_cap / (1024**3):.1f} GB for {n_res}D)"
          f" → {'STREAMING' if use_streaming else 'full load'} mode")

    if use_streaming:
        # ── STREAMING MODE: never load full cache into memory ──
        stream = _stream_coarse_cache(iso, rtypes, ZONES, active_thresholds)
        if stream is None:
            print(f"  ERROR: No coarse cache for {iso}. Run step1b first.")
            return

        # No coarse hash keys — fine mixes dedup only against each other
        # (inter-zone). Coarse-fine duplicates resolved by step 2.1.
        global_scored_keys = np.empty(0, dtype=np.int64)
        coarse_nm_combos = stream['coarse_nm_combos']
        coarse_nm_scores = stream['coarse_nm_scores']
        coarse_nm_idx = None  # not used in streaming mode (already materialized)
        warmup_rows = stream['warmup_rows']
        stream_zone_bounds = stream['zone_bounds']
        stream_zone_archetypes = stream['zone_archetypes']
        coarse_combos = None  # not loaded in streaming mode
        coarse_scores = None
        print(f"  Coarse cache: {stream['total_rows']:,} mixes (streamed)")
        print(f"  Coarse near-miss: {len(coarse_nm_combos):,} mixes")
        del stream  # free the dict shell
    else:
        # ── FULL LOAD MODE: fits in memory ──
        print(f"  Loading coarse cache...")
        if include_hybrids:
            table = s1.read_coarse_cache_table(iso)
            if table is None:
                print(f"  ERROR: No coarse cache for {iso}. Run step1b first.")
                return
            coarse_combos = np.column_stack([
                table.column(rt).to_numpy() for rt in rtypes
            ])
            coarse_scores = table.column('score').to_numpy()
            del table
        else:
            cached = s1.load_coarse_cache(iso)
            if cached is None:
                print(f"  ERROR: No coarse cache for {iso}. Run step1b first.")
                return
            coarse_combos, coarse_scores = cached
        print(f"  Coarse cache: {len(coarse_combos):,} mixes")

        global_scored_keys = _hash_mixes(coarse_combos)
        warmup_rows = coarse_combos[:2]
        stream_zone_bounds = None    # not used in full-load mode
        stream_zone_archetypes = None

        # Pre-compute coarse near-miss — store indices only to defer
        # materialization and save ~500 MB for large ISOs
        print(f"  Pre-computing coarse near-miss indices...")
        coarse_nm_idx = _collect_near_miss_indices(
            coarse_combos, coarse_scores, active_thresholds)
        print(f"  Coarse near-miss: {len(coarse_nm_idx):,} mixes (deferred)")
        # coarse_nm_combos/scores will be materialized on demand
        coarse_nm_combos = None
        coarse_nm_scores = None

    # ── JIT warmup ──
    if s1.HAS_NUMBA:
        print(f"  Warming up Numba JIT...")
        _ = s1.batch_hourly_scores(demand_arr, supply_matrix, warmup_rows)
        # Warmup storage kernel for final max-storage pre-filter
        dummy_supply = np.ones((1, s1.H), dtype=np.float64)
        dummy_b = np.array([0.0, 1.0], dtype=np.float64)
        dummy_l = np.array([0.0], dtype=np.float64)
        dummy_h = np.array([0.0], dtype=np.float64)
        s1._batch_mixes_storage_screen(
            demand_arr, dummy_supply, 1.0, 1,
            dummy_b, dummy_b, dummy_l,
            2, 2, 1,
            s1.BATTERY_EFFICIENCY, s1.BATTERY8_EFFICIENCY, s1.LDES_EFFICIENCY,
            s1.BATTERY_DURATION_HOURS, s1.BATTERY8_DURATION_HOURS,
            s1.LDES_DURATION_HOURS,
            s1.LDES_WINDOW_DAYS * 24, 48,
            dummy_h, 1, s1.H2_EFFICIENCY,
            float(s1.H2_DURATION_HOURS), s1.H2_WINDOW_DAYS * 24)
        print(f"  JIT ready (scoring + storage kernels)")
    del warmup_rows

    # Accumulate only FINE scored mixes (coarse stays separate to avoid OOM
    # on large caches like NEISO 28M+ rows — every np.vstack copies the full
    # array, and 3 copies of 2.3 GB exceeds the 7 GB GitHub runner limit).
    all_combos_list = []
    all_scores_list = []

    # ── Process zones ──
    os.makedirs(s1.STEP1_RAW_PFS_PARQUET_DIR, exist_ok=True)

    # Zone batch index for incremental parquet saves.
    # Maps zone_name → int so each zone's results get a unique batch file.
    zone_batch_map = {'A': 0, 'B': 1, 'C': 2}

    for zone_name, z_score_low, z_score_high, z_thresholds in ZONES:
        # Skip zone if caller requested specific zones and this isn't one of them
        if zones_filter and zone_name not in zones_filter:
            print(f"\n  Zone {zone_name}: skipped (not in zones_filter={zones_filter})")
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

        if use_streaming:
            # Streaming mode: bounds already computed incrementally
            if pzb:
                # Prior windows override streaming bounds
                bounds = compute_zone_resource_bounds(
                    iso, np.empty((0, n_res)), np.empty(0),
                    z_score_low, z_score_high, pzb,
                    include_hybrids=include_hybrids)
            else:
                bounds = stream_zone_bounds[zone_name]
        else:
            bounds = compute_zone_resource_bounds(
                iso, coarse_combos, coarse_scores,
                z_score_low, z_score_high, pzb,
                include_hybrids=include_hybrids)

        bounds_str = ', '.join(f"{rtypes[i]}=[{lo},{hi}]"
                               for i, (lo, hi) in enumerate(bounds))
        print(f"    Bounds: {bounds_str}")

        # 2. Generate fine 1% grid within zone bounds.
        #    generate_fine_grid returns None when the grid would exceed
        #    MAX_GRID_COMBOS_BEFORE_FALLBACK — fall back to archetype search.
        fine_combos = None
        if prior_windows:
            # Prior-informed: full Cartesian within zone bounds
            fine_combos = generate_fine_grid(bounds, step=FINE_STEP)

        if fine_combos is None:
            # Either no prior windows or grid too large — archetype-based fallback
            if use_streaming:
                boundary_mixes = stream_zone_archetypes[zone_name][0]
            else:
                zone_mask = ((coarse_scores >= z_score_low) &
                             (coarse_scores <= z_score_high))
                boundary_mixes = coarse_combos[zone_mask]
            fine_combos = generate_archetype_fine_grid(iso, boundary_mixes, n_res, include_hybrids=include_hybrids)

        print(f"    Raw fine grid: {len(fine_combos):,} combos")

        # 2b. Free coarse/archetype arrays to reclaim memory after last zone.
        remaining_zones = [z[0] for z in ZONES
                           if z[0] not in zones_done and z[0] != zone_name
                           and (not zones_filter or z[0] in zones_filter)]
        if use_streaming and not remaining_zones:
            del stream_zone_archetypes, stream_zone_bounds
            stream_zone_archetypes = None
            stream_zone_bounds = None
            gc.collect()
            print(f"    Freed streaming archetype/bounds data — last zone")
        elif n_res >= 8 and coarse_combos is not None and not remaining_zones:
            coarse_mem_mb = coarse_combos.nbytes / (1024 * 1024)
            del coarse_combos, coarse_scores
            coarse_combos = None
            coarse_scores = None
            gc.collect()
            print(f"    Freed coarse arrays ({coarse_mem_mb:.0f} MB) — last zone, reducing peak memory")

        # 3. Dedup against global keys (vectorized — no Python loop)
        if len(fine_combos) > 0:
            fine_keys = _hash_mixes(fine_combos)
            new_mask = ~np.isin(fine_keys, global_scored_keys)
            fine_combos = fine_combos[new_mask]
            if np.any(new_mask):
                global_scored_keys = np.concatenate([
                    global_scored_keys, fine_keys[new_mask]])

        print(f"    After dedup: {len(fine_combos):,} new mixes to score")

        # 4. Score new mixes in chunks (vectorized — NO Python loop over mixes)
        if len(fine_combos) > 0:
            score_start = time.time()
            chunk_sz = SCORE_CHUNK_SIZE_HIGH_D if n_res >= 8 else SCORE_CHUNK_SIZE
            fine_scores = s1.batch_hourly_scores(
                demand_arr, supply_matrix, fine_combos,
                chunk_size=chunk_sz)
            score_elapsed = time.time() - score_start
            print(f"    Scored {len(fine_combos):,} mixes in {score_elapsed:.1f}s")

            all_combos_list.append(fine_combos)
            all_scores_list.append(fine_scores)

        # 5. Save per-threshold batch files immediately (survive timeouts).
        #    Each zone writes _raw_pfs_b{N}.parquet per threshold.
        #    Step 2.1 merges all batch files for a given ISO/threshold.
        batch_idx = zone_batch_map.get(zone_name, 0)
        zone_combos = np.vstack(all_combos_list)
        zone_scores = np.concatenate(all_scores_list)
        zone_feas, zone_nm = assign_to_thresholds_vectorized(
            zone_combos, zone_scores, active_thresholds)
        n_saved = 0
        for t in active_thresholds:
            feas_i = zone_feas[t]
            if len(feas_i) == 0:
                continue
            t_combos = zone_combos[feas_i]
            t_scores = zone_scores[feas_i]
            save_threshold_pfs_batch(iso, t, t_combos, t_scores, rtypes,
                                     batch_idx)
            n_saved += 1
        print(f"    Saved batch b{batch_idx} for {n_saved} thresholds")

        zone_elapsed = time.time() - zone_start
        print(f"    Zone {zone_name} complete: {zone_elapsed:.1f}s")

        zones_done.add(zone_name)
        _save_manifest(iso, code_hash, zones_done, thresholds_done)

        # Save incremental near-miss so step1c has valid input even if
        # this job times out before all zones complete.
        print(f"    Saving interim near-miss parquet...")
        # Materialize coarse near-miss on demand (deferred in full-load mode)
        _c_nm_combos = coarse_nm_combos
        _c_nm_scores = coarse_nm_scores
        if _c_nm_combos is None and coarse_combos is not None and coarse_nm_idx is not None:
            _c_nm_combos = coarse_combos[coarse_nm_idx] if len(coarse_nm_idx) > 0 else np.empty((0, n_res), dtype=np.float64)
            _c_nm_scores = coarse_scores[coarse_nm_idx] if len(coarse_nm_idx) > 0 else np.empty(0, dtype=np.float64)
        save_near_miss_interim(iso, all_combos_list, all_scores_list,
                               rtypes, active_thresholds,
                               coarse_nm_combos=_c_nm_combos,
                               coarse_nm_scores=_c_nm_scores)
        del _c_nm_combos, _c_nm_scores

        gc.collect()

    # ── Materialize deferred near-miss before freeing coarse arrays ──
    if coarse_nm_combos is None and coarse_combos is not None and coarse_nm_idx is not None:
        print(f"  Materializing deferred coarse near-miss ({len(coarse_nm_idx):,} mixes)...")
        coarse_nm_combos = coarse_combos[coarse_nm_idx] if len(coarse_nm_idx) > 0 else np.empty((0, n_res), dtype=np.float64)
        coarse_nm_scores = coarse_scores[coarse_nm_idx] if len(coarse_nm_idx) > 0 else np.empty(0, dtype=np.float64)
        del coarse_nm_idx
        coarse_nm_idx = None

    # ── Free coarse arrays (no longer needed — saves ~2 GB for large ISOs) ──
    # May already be freed for high-D ISOs (freed after archetype generation)
    if coarse_combos is not None:
        del coarse_combos, coarse_scores
    del global_scored_keys
    gc.collect()

    # ── Combine fine scored mixes ──
    print(f"\n  Combining fine scored mixes...")
    if all_combos_list:
        all_combos = np.vstack(all_combos_list)
        all_scores = np.concatenate(all_scores_list)
    else:
        all_combos = np.empty((0, n_res), dtype=np.float64)
        all_scores = np.empty(0, dtype=np.float64)
    del all_combos_list, all_scores_list
    print(f"  Total fine scored mixes: {len(all_combos):,}")

    # ── Assign to thresholds + save final canonical files ──
    all_thresholds = LOW_THRESHOLDS + ACTIVE_THRESHOLDS
    print(f"\n  Assigning to {len(active_thresholds)} thresholds "
          f"(of {len(all_thresholds)} total)...")

    # Vectorized assignment on fine mixes only (coarse already saved by step1.1)
    feasible_idx, near_miss_idx = assign_to_thresholds_vectorized(
        all_combos, all_scores, active_thresholds)

    # Collect fine near-miss, merge with pre-computed coarse near-miss
    fine_nm_indices = _collect_near_miss_indices(
        all_combos, all_scores, active_thresholds)

    # coarse_nm_combos already materialized before coarse arrays were freed
    nm_parts_c, nm_parts_s = [], []
    if coarse_nm_combos is not None and len(coarse_nm_combos) > 0:
        nm_parts_c.append(coarse_nm_combos)
        nm_parts_s.append(coarse_nm_scores)
    if len(fine_nm_indices) > 0:
        nm_parts_c.append(all_combos[fine_nm_indices])
        nm_parts_s.append(all_scores[fine_nm_indices])

    # Save near-miss union (final authoritative version overwrites interim)
    # full_filter=True: compute max-storage scores and persist as column,
    # so step1c can skip its Pass 0 recomputation.
    if nm_parts_c:
        all_nm_combos = np.vstack(nm_parts_c)
        all_nm_scores = np.concatenate(nm_parts_s)
        save_near_miss(iso, all_nm_combos, all_nm_scores, rtypes,
                       demand_arr=demand_arr, supply_matrix=supply_matrix,
                       full_filter=True)
        n_total_nm = len(all_nm_combos)
        del all_nm_combos, all_nm_scores, nm_parts_c, nm_parts_s
    else:
        n_total_nm = 0

    # Per-threshold: dominance filter + save canonical files.
    # These overwrite the batch files when all zones complete successfully.
    for t in active_thresholds:
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

    # Clean up batch files now that canonical files exist
    _cleanup_batch_files(iso)

    iso_elapsed = time.time() - iso_start
    print(f"\n{'=' * 70}")
    print(f"  {iso} COMPLETE — {len(all_combos):,} fine scored mixes, "
          f"{n_total_nm:,} near-miss for step1c")
    print(f"  Elapsed: {iso_elapsed:.1f}s")
    print(f"{'=' * 70}")


def _parse_thresholds(raw):
    """Parse comma-separated threshold list, return sorted list of floats or None."""
    if not raw or raw.strip().upper() in ('', 'ALL'):
        return None
    parts = [p.strip() for p in raw.split(',') if p.strip()]
    result = []
    for p in parts:
        try:
            result.append(float(p))
        except ValueError:
            print(f"WARNING: Ignoring invalid threshold '{p}'")
    return sorted(set(result)) if result else None


def _parse_zones(raw):
    """Parse comma-separated zone list (A/B/C), return set or None (= all)."""
    if not raw or raw.strip().upper() in ('', 'ALL'):
        return None
    valid = {'A', 'B', 'C'}
    parts = {p.strip().upper() for p in raw.split(',') if p.strip()}
    bad = parts - valid
    if bad:
        print(f"WARNING: Unknown zones {bad} — ignoring. Valid: A, B, C")
    result = parts & valid
    return result if result else None


def main():
    parser = argparse.ArgumentParser(
        description="Step 1b: Zone-based fine search with global dedup.")
    parser.add_argument("--iso", required=True,
                        help="ISO name or 'ALL'")
    parser.add_argument("--thresholds", default="",
                        help="Comma-separated thresholds to process "
                             "(e.g. '90,95,99'). Default: all 21.")
    parser.add_argument("--zones", default="",
                        help="Comma-separated zones to run: A, B, C "
                             "(e.g. 'B,C'). Default: all zones.")
    parser.add_argument("--hybrid", action="store_true",
                        help="Enable hybrid resource types (solar+batt, wind+batt)")
    args = parser.parse_args()

    isos = list(s1.ISOS) if args.iso.upper() == 'ALL' else [args.iso.upper()]

    for iso in isos:
        if iso not in s1.ISOS:
            print(f"ERROR: Unknown ISO '{iso}'")
            sys.exit(1)

    thresholds_filter = _parse_thresholds(args.thresholds)
    zones_filter = _parse_zones(args.zones)

    if thresholds_filter:
        print(f"Threshold filter: {thresholds_filter}")
    if zones_filter:
        print(f"Zone filter: {sorted(zones_filter)}")
    if args.hybrid:
        print(f"Hybrid mode: enabled (CLI flag)")

    for iso in isos:
        process_iso(iso,
                    thresholds_filter=thresholds_filter,
                    zones_filter=zones_filter,
                    include_hybrids=args.hybrid)


if __name__ == "__main__":
    main()
