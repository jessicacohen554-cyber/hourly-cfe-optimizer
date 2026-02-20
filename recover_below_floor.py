#!/usr/bin/env python3
"""
One-time recovery script: Grab mixes below the existing clean floor from the
PFS cache and merge them into pfs_post_ef.parquet.

Background: Step 2 Phase 0 filtered out mixes that under-allocated existing
generation (hydro=0, low clean_firm, etc.). Those mixes are needed for the
new-build hourly matching analysis (Track 1) and cost-to-replace analysis
(Track 2). This script recovers them without re-running Step 1.

Logic:
  1. Load PFS cache (38.7M rows)
  2. Apply threshold gate (13 target thresholds)
  3. Apply existing clean floor filter → get the keep mask
  4. Take INVERTED mask (~keep) → below-floor mixes
  5. Run Pareto procurement on below-floor mixes per ISO
  6. Load existing pfs_post_ef.parquet
  7. Concatenate
  8. Re-run Pareto procurement on combined set per ISO
  9. Write back to pfs_post_ef.parquet

Can be deleted after merge is verified.
"""

import os
import sys
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

# Import step2 functions
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from step2_efficient_frontier import (
    ISOS, TARGET_THRESHOLDS, GRID_MIX_SHARES, DEMAND_GROWTH_HIGH,
    step2_pareto_procurement, process_iso,
)

PFS_PATH = os.path.join(SCRIPT_DIR, 'data', 'physics_cache_v4.parquet')
EF_PATH = os.path.join(SCRIPT_DIR, 'data', 'pfs_post_ef.parquet')
BACKUP_PATH = os.path.join(SCRIPT_DIR, 'data', 'pfs_post_ef_backup.parquet')

MATERIALITY_THRESHOLD = 1.0  # same as step2


def compute_floor_mask(table):
    """Compute the existing clean floor keep mask (same logic as step2 Phase 0).
    Returns: boolean numpy array — True = ABOVE floor (would be kept by step2).
    """
    cf = table.column('clean_firm').to_numpy()
    sol = table.column('solar').to_numpy()
    wnd = table.column('wind').to_numpy()
    hyd = table.column('hydro').to_numpy()
    proc = table.column('procurement_pct').to_numpy()
    iso_col = table.column('iso').to_pylist()

    iso_to_idx = {iso: i for i, iso in enumerate(ISOS)}
    iso_indices = np.array([iso_to_idx.get(s, 0) for s in iso_col])

    # Min shares at 2050 high growth
    min_shares = {}
    for iso in ISOS:
        growth_factor = (1 + DEMAND_GROWTH_HIGH[iso]) ** 25
        min_shares[iso] = {rtype: share / growth_factor
                           for rtype, share in GRID_MIX_SHARES[iso].items()}

    min_cf = np.array([min_shares[iso]['clean_firm'] for iso in ISOS])
    min_sol = np.array([min_shares[iso]['solar'] for iso in ISOS])
    min_wnd = np.array([min_shares[iso]['wind'] for iso in ISOS])
    min_hyd = np.array([min_shares[iso]['hydro'] for iso in ISOS])

    keep = np.ones(len(cf), dtype=np.bool_)

    # Demand-fraction check
    for r_pct, r_min in [(cf, min_cf), (sol, min_sol), (wnd, min_wnd), (hyd, min_hyd)]:
        row_min = r_min[iso_indices]
        mix_demand_pct = proc * r_pct / 100.0
        has_material_existing = row_min > MATERIALITY_THRESHOLD
        fails = has_material_existing & (mix_demand_pct < row_min)
        keep &= ~fails

    # Allocation floor check
    resource_cols = {'clean_firm': cf, 'solar': sol, 'wind': wnd, 'hydro': hyd}
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
        keep &= ~fails

    return keep


def apply_threshold_gate(table):
    """Keep only rows at target thresholds."""
    threshold_col = table.column('threshold')
    mask = None
    for thr in TARGET_THRESHOLDS:
        eq = pc.equal(threshold_col, thr)
        mask = eq if mask is None else pc.or_(mask, eq)
    return table.filter(mask)


def run_pareto_per_iso(table):
    """Run Pareto procurement selection per ISO, drop threshold column."""
    results = []
    total_raw = 0
    total_pareto = 0

    for iso in ISOS:
        t0 = time.time()
        result, n_raw, n_pareto = process_iso(table, iso)
        elapsed = time.time() - t0

        if result is not None and result.num_rows > 0:
            results.append(result)
            total_raw += n_raw
            total_pareto += n_pareto
            print(f"  {iso:>6}  {n_raw:>9,}  {n_pareto:>9,}  {elapsed:>5.1f}s")

    if not results:
        return None
    return pa.concat_tables(results), total_raw, total_pareto


def main():
    print("=" * 70)
    print("  BELOW-FLOOR MIX RECOVERY")
    print("  Recovering mixes filtered by Step 2 existing clean floor")
    print("=" * 70)
    total_start = time.time()

    # 1. Load PFS cache
    print(f"\nLoading PFS cache: {PFS_PATH}")
    pfs = pq.read_table(PFS_PATH)
    print(f"  Total PFS rows: {pfs.num_rows:,}")

    # 2. Apply threshold gate
    print("\nApplying threshold gate...")
    pfs_gated = apply_threshold_gate(pfs)
    print(f"  After threshold gate: {pfs_gated.num_rows:,}")
    del pfs  # free memory

    # 3. Compute floor mask
    print("\nComputing existing clean floor mask...")
    keep_mask = compute_floor_mask(pfs_gated)
    n_above = keep_mask.sum()
    n_below = (~keep_mask).sum()
    print(f"  Above floor (kept by step2): {n_above:,}")
    print(f"  Below floor (filtered out):  {n_below:,}")

    if n_below == 0:
        print("\nNo below-floor mixes found. Nothing to recover.")
        return

    # Per-ISO breakdown of below-floor mixes
    iso_col = np.array(pfs_gated.column('iso').to_pylist())
    for iso in ISOS:
        iso_mask = iso_col == iso
        iso_below = (~keep_mask & iso_mask).sum()
        iso_total = iso_mask.sum()
        print(f"  {iso:>6}: {iso_below:,} below floor of {iso_total:,} "
              f"({iso_below/max(iso_total,1)*100:.1f}%)")

    # 4. Extract below-floor mixes
    print("\nExtracting below-floor mixes...")
    below_floor = pfs_gated.filter(pa.array(~keep_mask))
    print(f"  Below-floor mixes: {below_floor.num_rows:,}")
    del pfs_gated  # free memory

    # 5. Run Pareto procurement on below-floor mixes
    print("\nPhase 2: Pareto procurement on below-floor mixes")
    print(f"  {'ISO':>6}  {'Raw':>9}  {'Pareto':>9}  {'Time':>6}")
    print("  " + "-" * 40)
    pareto_result = run_pareto_per_iso(below_floor)
    del below_floor

    if pareto_result is None:
        print("No Pareto-optimal below-floor mixes found.")
        return

    recovered, total_raw, total_pareto = pareto_result
    print(f"\n  Below-floor Pareto: {total_raw:,} → {total_pareto:,}")

    # 6. Load existing EF
    print(f"\nLoading existing EF: {EF_PATH}")
    existing_ef = pq.read_table(EF_PATH)
    print(f"  Existing EF rows: {existing_ef.num_rows:,}")

    # 7. Backup existing EF
    print(f"  Backing up to: {BACKUP_PATH}")
    pq.write_table(existing_ef, BACKUP_PATH, compression='snappy')

    # 8. Concatenate
    print("\nConcatenating existing EF + recovered below-floor mixes...")
    # Align columns
    target_cols = existing_ef.column_names
    recovered_aligned = []
    for col in target_cols:
        if col in recovered.column_names:
            recovered_aligned.append(recovered.column(col))
        elif col == 'pareto_type':
            recovered_aligned.append(pa.array(['below_floor'] * recovered.num_rows))
        else:
            recovered_aligned.append(pa.array(np.zeros(recovered.num_rows, dtype=np.int64)))
    recovered_table = pa.table(recovered_aligned, names=target_cols)
    combined = pa.concat_tables([existing_ef, recovered_table])
    print(f"  Combined: {combined.num_rows:,}")
    del existing_ef, recovered, recovered_table

    # 9. Re-run Pareto procurement on combined set
    print("\nPhase 2: Re-running Pareto procurement on combined set")
    print(f"  {'ISO':>6}  {'Raw':>9}  {'Pareto':>9}  {'Time':>6}")
    print("  " + "-" * 40)
    final_result = run_pareto_per_iso(combined)
    del combined

    if final_result is None:
        print("ERROR: Combined Pareto produced no results!")
        return

    final_table, total_raw, total_pareto = final_result
    print(f"\n  Final: {total_raw:,} → {total_pareto:,}")

    # 10. Write back
    pq.write_table(final_table, EF_PATH, compression='snappy')
    file_size = os.path.getsize(EF_PATH) / (1024 * 1024)
    elapsed = time.time() - total_start

    print(f"\n  Output: {EF_PATH}")
    print(f"  Size: {file_size:.1f} MB ({final_table.num_rows:,} rows)")
    print(f"  Total time: {elapsed:.0f}s")

    # Score distribution
    scores = final_table.column('hourly_match_score').to_numpy()
    for iso in ISOS:
        iso_mask = np.array(final_table.column('iso').to_pylist()) == iso
        iso_scores = scores[iso_mask]
        if len(iso_scores) > 0:
            n_hydro0 = 0
            hyd = final_table.column('hydro').to_numpy()
            n_hydro0 = ((hyd == 0) & iso_mask).sum()
            n_total = iso_mask.sum()
            print(f"  {iso}: {n_total:,} mixes ({n_hydro0:,} with hydro=0, "
                  f"{n_hydro0/max(n_total,1)*100:.1f}%)")

    print("\n" + "=" * 70)
    print("  RECOVERY COMPLETE — pfs_post_ef.parquet updated")
    print(f"  Backup at: {BACKUP_PATH}")
    print("=" * 70)


if __name__ == '__main__':
    main()
