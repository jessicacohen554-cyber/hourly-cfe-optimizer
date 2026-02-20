#!/usr/bin/env python3
"""
Step 2: Efficient Frontier (EF) Extraction from Physics Feasible Space (PFS)
=============================================================================
Reads the PFS parquet (21.4M rows) and applies a 4-phase reduction:

  Phase 0: Existing generation filter — remove mixes that don't fully utilize
           existing clean generation (even at 2050 high demand growth). Any mix
           that allocates less than the minimum existing share for a resource
           (where min = existing / growth_factor_2050_high) wastes free/cheap
           generation and will never be cost-optimal.
  Phase 1: Threshold gate — keep only target thresholds (50-100%)
  Phase 2: Procurement minimization — for each unique resource allocation
           (CF/Sol/Wnd/CCS/Hyd/Bat/LDES), keep only the lowest procurement
           level that achieves the threshold.
  Phase 3: Strict dominance removal — remove mixes where another mix is ≤
           on ALL dimensions (procurement, battery, LDES) while achieving
           the same or better match score. Skip for groups > 50K.

Pipeline position: Step 2 of 4
  Step 1 — PFS Generator (step1_pfs_generator.py)
  Step 2 — Efficient Frontier extraction (this file)
  Step 3 — Cost optimization (step3_cost_optimization.py)
  Step 4 — Post-processing (step4_postprocess.py)

Input:  data/physics_cache_v4.parquet  (21.4M rows — the PFS)
Output: data/pfs_post_ef.parquet       (PFS post-EF)

The output preserves all mixes that could be optimal under ANY cost assumption,
ensuring no true optimum is lost during cost evaluation in Step 3.
"""

import os
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PFS_PATH = os.path.join(SCRIPT_DIR, 'data', 'physics_cache_v4.parquet')
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'data', 'pfs_post_ef.parquet')

# Target thresholds — all 13 from v4 PFS (50-100%)
TARGET_THRESHOLDS = [50.0, 60.0, 70.0, 75.0, 80.0, 85.0, 87.5, 90.0, 92.5, 95.0, 97.5, 99.0, 100.0]

# Max group size for Step 3 dominance check (O(n²) — skip larger groups)
DOMINANCE_MAX_GROUP = 50000


ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO']

# Existing clean generation as % of 2025 demand (from eGRID/EIA)
GRID_MIX_SHARES = {
    'CAISO': {'clean_firm': 7.9, 'solar': 22.3, 'wind': 8.8, 'ccs_ccgt': 0, 'hydro': 9.5},
    'ERCOT': {'clean_firm': 8.6, 'solar': 13.8, 'wind': 23.6, 'ccs_ccgt': 0, 'hydro': 0.1},
    'PJM':   {'clean_firm': 32.1, 'solar': 2.9, 'wind': 3.8, 'ccs_ccgt': 0, 'hydro': 1.8},
    'NYISO': {'clean_firm': 18.4, 'solar': 0.0, 'wind': 4.7, 'ccs_ccgt': 0, 'hydro': 15.9},
    'NEISO': {'clean_firm': 23.8, 'solar': 1.4, 'wind': 3.9, 'ccs_ccgt': 0, 'hydro': 4.4},
}

# High demand growth rates (annual) — most aggressive scenario
# At 2050 high growth, existing share is smallest → tightest filter
DEMAND_GROWTH_HIGH = {
    'CAISO': 0.025, 'ERCOT': 0.055, 'PJM': 0.036, 'NYISO': 0.044, 'NEISO': 0.029,
}


def compute_existing_min_shares():
    """Compute minimum existing shares at 2050 high growth for each ISO.
    Returns dict of {iso: {rtype: min_share_as_pct_of_demand}}."""
    result = {}
    for iso in ISOS:
        growth_factor = (1 + DEMAND_GROWTH_HIGH[iso]) ** 25  # 2050 - 2025 = 25 years
        result[iso] = {}
        for rtype, share in GRID_MIX_SHARES[iso].items():
            result[iso][rtype] = share / growth_factor  # as % of grown demand
    return result


def load_pfs():
    """Load the v4 Physics Feasible Space."""
    if not os.path.exists(PFS_PATH):
        raise FileNotFoundError(f"PFS not found: {PFS_PATH}")

    print(f"Loading PFS: {PFS_PATH}")
    table = pq.read_table(PFS_PATH)
    print(f"  Total rows: {table.num_rows:,}")
    return table


def step0_existing_generation_filter(table):
    """Step 0: Remove mixes that don't fully utilize existing clean generation.

    A mix is filtered if, for any resource type, the mix's allocation
    (procurement × resource_pct) is less than the existing generation's share
    of demand at 2050 high growth. Such mixes waste free/cheap existing
    generation and will never be cost-optimal under any sensitivity.

    Only filters on resources with >0 existing share.
    """
    print("\nStep 0: Existing generation utilization filter")
    min_shares = compute_existing_min_shares()

    cf = table.column('clean_firm').to_numpy()
    sol = table.column('solar').to_numpy()
    wnd = table.column('wind').to_numpy()
    hyd = table.column('hydro').to_numpy()
    proc = table.column('procurement_pct').to_numpy()
    iso_col = table.column('iso').to_pylist()

    # Convert ISO strings to indices for vectorized lookup
    iso_to_idx = {iso: i for i, iso in enumerate(ISOS)}
    iso_indices = np.array([iso_to_idx.get(s, 0) for s in iso_col])

    # Build min-share arrays indexed by ISO
    min_cf = np.array([min_shares[iso]['clean_firm'] for iso in ISOS])
    min_sol = np.array([min_shares[iso]['solar'] for iso in ISOS])
    min_wnd = np.array([min_shares[iso]['wind'] for iso in ISOS])
    min_hyd = np.array([min_shares[iso]['hydro'] for iso in ISOS])

    # For each mix: demand_fraction = procurement/100 * pct/100 * 100 = procurement * pct / 100
    # Min share is in % of demand. So check: procurement * pct / 100 >= min_share
    # Or: procurement * pct >= min_share * 100
    keep = np.ones(len(cf), dtype=np.bool_)

    # Materiality threshold: only filter on resources with >1% min existing share
    # This avoids filtering on negligible existing generation (e.g., ERCOT hydro 0.03%)
    MATERIALITY_THRESHOLD = 1.0  # % of demand at 2050 high growth

    for r_pct, r_min in [(cf, min_cf), (sol, min_sol), (wnd, min_wnd), (hyd, min_hyd)]:
        # Look up min share for each row's ISO
        row_min = r_min[iso_indices]  # min share as % of demand
        # Mix's demand fraction in %: proc * pct / 100
        mix_demand_pct = proc * r_pct / 100.0
        # Only filter if min share is material (>1% of demand at 2050 high)
        has_material_existing = row_min > MATERIALITY_THRESHOLD
        fails = has_material_existing & (mix_demand_pct < row_min)
        keep &= ~fails

    # Resource allocation floors: existing generation scaled to 2050 high demand growth.
    # Any mix allocating less than the floor for a resource wastes cheap existing generation.
    # Applied to all four resource types where the floor exceeds the materiality threshold.
    resource_cols = {'clean_firm': cf, 'solar': sol, 'wind': wnd, 'hydro': hyd}
    print("  Resource allocation floors (existing / 2050 high growth factor):")
    alloc_floor_removed = 0
    for rtype, r_col in resource_cols.items():
        floor_arr = np.zeros(len(ISOS), dtype=np.float64)
        for i, iso in enumerate(ISOS):
            existing = GRID_MIX_SHARES[iso].get(rtype, 0)
            growth_factor = (1 + DEMAND_GROWTH_HIGH[iso]) ** 25
            floor_val = existing / growth_factor
            if floor_val > MATERIALITY_THRESHOLD:
                floor_arr[i] = floor_val
        row_floor = floor_arr[iso_indices]
        fails = (row_floor > 0) & (r_col < row_floor)
        n_removed = (keep & fails).sum()
        if n_removed > 0:
            keep &= ~fails
            alloc_floor_removed += n_removed
        # Print summary per resource
        active_isos = [iso for i, iso in enumerate(ISOS) if floor_arr[i] > 0]
        if active_isos:
            floors_str = ", ".join(f"{iso}≥{floor_arr[ISOS.index(iso)]:.1f}" for iso in active_isos)
            print(f"    {rtype:>10}: {floors_str} (removed {n_removed:,})")
    if alloc_floor_removed > 0:
        print(f"  Allocation floor total: removed {alloc_floor_removed:,} additional rows")

    filtered = table.filter(pa.array(keep))
    removed = table.num_rows - filtered.num_rows

    # Per-ISO stats
    for iso in ISOS:
        iso_mask = np.array(iso_col) == iso
        iso_total = iso_mask.sum()
        iso_kept = (iso_mask & keep).sum()
        print(f"  {iso:>6}: {iso_total:>9,} → {iso_kept:>9,} "
              f"(removed {iso_total - iso_kept:>7,}, {(iso_total - iso_kept)/iso_total*100:.1f}%)")

    print(f"  Total: {table.num_rows:,} → {filtered.num_rows:,} (removed {removed:,})")
    return filtered


def step1_threshold_gate(table):
    """Step 1: Keep only target thresholds."""
    print("\nStep 1: Threshold gate")

    # Build OR filter for target thresholds
    threshold_col = table.column('threshold')
    mask = None
    for thr in TARGET_THRESHOLDS:
        eq = pc.equal(threshold_col, thr)
        mask = eq if mask is None else pc.or_(mask, eq)

    filtered = table.filter(mask)
    print(f"  {table.num_rows:,} → {filtered.num_rows:,} "
          f"(kept {len(TARGET_THRESHOLDS)} of {len(pc.unique(threshold_col).to_pylist())} thresholds)")
    return filtered


def step2_procurement_minimization(iso_thr_arrays):
    """
    Step 2: For each unique allocation (CF/Sol/Wnd/Hyd/Bat4/Bat8/LDES),
    keep only the row with the lowest procurement_pct.
    """
    n = len(iso_thr_arrays['clean_firm'])
    if n == 0:
        return np.array([], dtype=np.int64)

    cf = iso_thr_arrays['clean_firm']
    sol = iso_thr_arrays['solar']
    wnd = iso_thr_arrays['wind']
    hyd = iso_thr_arrays['hydro']
    bat = iso_thr_arrays['battery_dispatch_pct']
    bat8 = iso_thr_arrays['battery8_dispatch_pct']
    ldes = iso_thr_arrays['ldes_dispatch_pct']
    proc = iso_thr_arrays['procurement_pct']

    # Pack: CF*101^6 + Sol*101^5 + Wnd*101^4 + Hyd*101^3 + Bat*101^2 + Bat8*101 + LDES
    group_key = (cf.astype(np.int64) * (101**6) +
                 sol.astype(np.int64) * (101**5) +
                 wnd.astype(np.int64) * (101**4) +
                 hyd.astype(np.int64) * (101**3) +
                 bat.astype(np.int64) * (101**2) +
                 bat8.astype(np.int64) * 101 +
                 ldes.astype(np.int64))

    # Sort by group_key, then by procurement (ascending)
    sort_idx = np.lexsort((proc, group_key))
    sorted_keys = group_key[sort_idx]

    # Keep first occurrence of each group (lowest procurement)
    keep_mask = np.empty(n, dtype=np.bool_)
    keep_mask[0] = True
    keep_mask[1:] = sorted_keys[1:] != sorted_keys[:-1]

    return sort_idx[keep_mask]


def step3_strict_dominance(iso_thr_arrays, indices):
    """
    Step 3: Remove strictly dominated mixes.
    A mix A is dominated by mix B if B has ≤ procurement, ≤ battery4, ≤ battery8,
    ≤ LDES, and ≥ match score (with at least one strict inequality).
    O(n²) — only run for manageable group sizes.
    """
    n = len(indices)
    if n > DOMINANCE_MAX_GROUP:
        return indices

    if n <= 1:
        return indices

    proc = iso_thr_arrays['procurement_pct'][indices].astype(np.float64)
    bat = iso_thr_arrays['battery_dispatch_pct'][indices].astype(np.float64)
    bat8 = iso_thr_arrays['battery8_dispatch_pct'][indices].astype(np.float64)
    ldes = iso_thr_arrays['ldes_dispatch_pct'][indices].astype(np.float64)
    score = iso_thr_arrays['hourly_match_score'][indices].astype(np.float64)

    sort_order = np.argsort(proc)
    proc = proc[sort_order]
    bat = bat[sort_order]
    bat8 = bat8[sort_order]
    ldes = ldes[sort_order]
    score = score[sort_order]
    sorted_indices = indices[sort_order]

    dominated = np.zeros(n, dtype=np.bool_)

    for i in range(n):
        if dominated[i]:
            continue
        j_mask = (
            (proc[i] <= proc[i+1:]) &
            (bat[i] <= bat[i+1:]) &
            (bat8[i] <= bat8[i+1:]) &
            (ldes[i] <= ldes[i+1:]) &
            (score[i] >= score[i+1:])
        )
        strict = (
            (proc[i] < proc[i+1:]) |
            (bat[i] < bat[i+1:]) |
            (bat8[i] < bat8[i+1:]) |
            (ldes[i] < ldes[i+1:]) |
            (score[i] > score[i+1:])
        )
        dominated[i+1:] |= (j_mask & strict)

    return sorted_indices[~dominated]


def process_iso_threshold(table, iso, threshold):
    """Process a single (ISO, threshold) group through Steps 2-3."""
    mask = pc.and_(
        pc.equal(table.column('iso'), iso),
        pc.equal(table.column('threshold'), threshold)
    )
    subtable = table.filter(mask)
    n_raw = subtable.num_rows

    if n_raw == 0:
        return None, 0, 0, 0

    arrays = {
        'clean_firm': subtable.column('clean_firm').to_numpy(),
        'solar': subtable.column('solar').to_numpy(),
        'wind': subtable.column('wind').to_numpy(),
        'hydro': subtable.column('hydro').to_numpy(),
        'procurement_pct': subtable.column('procurement_pct').to_numpy(),
        'battery_dispatch_pct': subtable.column('battery_dispatch_pct').to_numpy(),
        'battery8_dispatch_pct': (subtable.column('battery8_dispatch_pct').to_numpy()
                                   if 'battery8_dispatch_pct' in subtable.column_names
                                   else np.zeros(n_raw, dtype=np.int64)),
        'ldes_dispatch_pct': subtable.column('ldes_dispatch_pct').to_numpy(),
        'hourly_match_score': subtable.column('hourly_match_score').to_numpy(),
    }

    step2_idx = step2_procurement_minimization(arrays)
    n_step2 = len(step2_idx)

    step3_idx = step3_strict_dominance(arrays, step2_idx)
    n_step3 = len(step3_idx)

    result = subtable.take(step3_idx)
    return result, n_raw, n_step2, n_step3


def main():
    print("=" * 70)
    print("  STEP 2: EFFICIENT FRONTIER (EF) EXTRACTION")
    print("  PFS → PFS post-EF")
    print("=" * 70)

    total_start = time.time()
    table = load_pfs()

    # Step 0: Existing generation utilization filter
    table = step0_existing_generation_filter(table)

    # Step 1: Threshold gate (all 13 thresholds kept since PFS has exactly these)
    table = step1_threshold_gate(table)

    # Process each (ISO, threshold) through Steps 2-3
    print("\nSteps 2-3: Procurement minimization + dominance removal")
    print(f"  {'ISO':>6}  {'Thr':>5}  {'Raw':>9}  {'Step2':>9}  {'Step3':>9}  {'Time':>6}")
    print("  " + "-" * 55)

    results = []
    total_raw = 0
    total_step2 = 0
    total_step3 = 0

    for iso in ISOS:
        for thr in TARGET_THRESHOLDS:
            t0 = time.time()
            result, n_raw, n_step2, n_step3 = process_iso_threshold(table, iso, thr)
            elapsed = time.time() - t0

            if result is not None and result.num_rows > 0:
                results.append(result)
                total_raw += n_raw
                total_step2 += n_step2
                total_step3 += n_step3

                print(f"  {iso:>6}  {thr:>4}%  {n_raw:>8,}  {n_step2:>8,}  "
                      f"{n_step3:>8,}  {elapsed:>5.1f}s")

    # Concatenate all results
    print(f"\n  Total: {total_raw:,} → {total_step2:,} (Step 2) → {total_step3:,} (Step 3)")
    if total_raw > 0:
        print(f"  Reduction: {(1 - total_step3/total_raw)*100:.1f}%")

    combined = pa.concat_tables(results)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    pq.write_table(combined, OUTPUT_PATH, compression='snappy')
    file_size = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)

    elapsed_total = time.time() - total_start
    print(f"\n  Output: {OUTPUT_PATH}")
    print(f"  Size: {file_size:.1f} MB ({combined.num_rows:,} rows)")
    print(f"  Total time: {elapsed_total:.0f}s")

    print("\n" + "=" * 70)
    print("  STEP 2 COMPLETE — PFS post-EF ready for Step 3 cost optimization")
    print("=" * 70)


if __name__ == '__main__':
    main()
