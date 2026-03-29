#!/usr/bin/env python3
"""Step 1b: Score mix combinations in chunks → scored parquet database.

Reads the static mixes parquet from step1a ({ISO}_mixes.parquet), loads
EIA demand + generation profiles, and computes each combo's hourly match
score in memory-bounded chunks. Never allocates the full N × 8760 array.

Standard mode:
  Output: data/step1-pfs/{ISO}_coarse_cache.parquet
    Columns: clean_firm, solar, wind, hydro, [offshore_wind], [geothermal], score

Hybrid mode (--hybrid):
  Loads hybrid mixes from {ISO}_mixes.parquet (4 extra columns:
  solar_batt4, solar_batt8, wind_batt4, wind_batt8) and hybrid 8760
  profiles from data/hybrid_profiles/{ISO}_hybrid_profiles.npz.
  Output: data/step1-pfs/{ISO}_coarse_cache.parquet

Memory: Peak ~1.4 GiB (20K × 8760 × 8 bytes per scoring chunk).
  CAISO 5D: 1.6M combos scored in ~80 chunks, not all at once.

Usage:
  python scripts/step1_1b_score_mixes.py --iso CAISO
  python scripts/step1_1b_score_mixes.py --iso NYISO --chunk-size 10000
  python scripts/step1_1b_score_mixes.py --iso ALL --hybrid
"""

import argparse
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import step1_pfs_generator as s1

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("ERROR: pyarrow required. pip install pyarrow")
    sys.exit(1)


def _mixes_path(iso):
    """Path for the static mixes parquet from step1a."""
    return os.path.join(s1.STEP1_RAW_PFS_PARQUET_DIR, f'{iso}_mixes.parquet')


def _hybrid_mixes_path(iso):
    """Path for the hybrid mixes parquet from step1a --hybrid (same name as standard)."""
    return os.path.join(s1.STEP1_RAW_PFS_PARQUET_DIR, f'{iso}_mixes.parquet')


def _hybrid_cache_path(iso):
    """Path for the hybrid scored cache (same name as standard)."""
    return os.path.join(s1.STEP1_RAW_PFS_PARQUET_DIR, f'{iso}_coarse_cache.parquet')


def _hybrid_cache_part_path(iso, part_idx):
    """Path for a numbered part file of the hybrid scored cache."""
    return os.path.join(s1.STEP1_RAW_PFS_PARQUET_DIR,
                        f'{iso}_coarse_cache_part{part_idx:03d}.parquet')


def load_mixes(iso):
    """Load mixes from step1a parquet, or generate on-the-fly as fallback."""
    rtypes = s1.get_resource_types(iso)
    path = _mixes_path(iso)

    if os.path.exists(path):
        table = pq.read_table(path)
        combos = np.column_stack([table.column(rt).to_numpy() for rt in rtypes])
        print(f"  {iso}: Loaded {len(combos):,} mixes from {path}")
        return combos

    # Fallback: generate on-the-fly (step1a wasn't run separately)
    print(f"  {iso}: No mixes parquet found — generating on-the-fly")
    combos = s1.generate_resource_combos(iso, step=5)
    seeds = s1.get_seed_combos(iso)
    if len(seeds) > 0:
        combos = np.vstack([combos, seeds])
        combos = np.unique(combos, axis=0)
    print(f"  {iso}: Generated {len(combos):,} mixes ({len(rtypes)}D)")
    return combos


def load_hybrid_mixes(iso):
    """Load hybrid mixes from step1a --hybrid parquet."""
    rtypes = s1.get_resource_types(iso, include_hybrids=True)
    path = _hybrid_mixes_path(iso)

    if not os.path.exists(path):
        print(f"  ERROR: {path} not found. Run step1_1a --hybrid first.")
        sys.exit(1)

    table = pq.read_table(path)
    combos = np.column_stack([table.column(rt).to_numpy() for rt in rtypes])
    print(f"  {iso}: Loaded {len(combos):,} hybrid mixes from {path} ({len(rtypes)}D)")
    return combos


def score_and_save_hybrid_streaming(iso, demand_arr, supply_matrix,
                                    chunk_size=20000, max_file_mb=45):
    """Score hybrid mixes in streaming chunks and write directly to parquet.

    For large ISOs (NEISO 28M, CAISO 87M), loading all mixes into RAM + scoring
    exceeds the 7 GB GitHub Actions limit. This function reads parquet row groups,
    scores each chunk, and streams scored results to the output parquet.

    When max_file_mb > 0 and a single output file would exceed the limit,
    output is split into numbered part files:
      {ISO}_coarse_cache_part001.parquet, _part002.parquet, ...
    Small ISOs that fit in one file still produce the standard single file:
      {ISO}_coarse_cache.parquet

    Peak memory: chunk_size × 8760 × 8 bytes (scoring) + chunk_size × n_res × 8 (mixes).
    At chunk_size=20000: ~1.4 GB scoring + ~1.6 MB mixes = ~1.4 GB peak.
    """
    rtypes = s1.get_resource_types(iso, include_hybrids=True)
    n_res = len(rtypes)
    in_path = _hybrid_mixes_path(iso)

    if not os.path.exists(in_path):
        print(f"  ERROR: {in_path} not found. Run step1_1a --hybrid first.")
        sys.exit(1)

    os.makedirs(s1.STEP1_RAW_PFS_PARQUET_DIR, exist_ok=True)

    # Clean up any existing part files for this ISO
    import glob as _glob
    for old in _glob.glob(os.path.join(s1.STEP1_RAW_PFS_PARQUET_DIR,
                                       f'{iso}_coarse_cache_part*.parquet')):
        os.remove(old)
    single_path = _hybrid_cache_path(iso)
    if os.path.exists(single_path):
        os.remove(single_path)

    # Read parquet metadata to get total rows
    pf = pq.ParquetFile(in_path)
    total_rows = pf.metadata.num_rows
    n_row_groups = pf.metadata.num_row_groups

    mem_chunk_gb = chunk_size * 8760 * 8 / (1024**3)
    print(f"  {iso}: Streaming score — {total_rows:,} mixes in {n_row_groups} "
          f"row groups, chunk_size={chunk_size:,} ({mem_chunk_gb:.1f} GiB peak)",
          flush=True)

    # Output schema: resource columns + score
    out_schema = pa.schema([(rt, pa.float64()) for rt in rtypes] +
                           [('score', pa.float64())])

    max_file_bytes = max_file_mb * 1024 * 1024 if max_file_mb > 0 else float('inf')
    part_idx = 1
    current_path = _hybrid_cache_part_path(iso, part_idx)
    writer = pq.ParquetWriter(current_path, out_schema, compression='snappy')
    part_paths = [current_path]
    part_rows = 0  # rows in current part
    bytes_per_row = None  # estimated after first part closes

    total_scored = 0

    for rg_idx in range(n_row_groups):
        rg_table = pf.read_row_group(rg_idx, columns=rtypes)
        rg_combos = np.column_stack([rg_table.column(rt).to_numpy() for rt in rtypes])
        n_rg = len(rg_combos)

        # Score this row group in sub-chunks
        rg_scores = s1.batch_hourly_scores(demand_arr, supply_matrix, rg_combos,
                                           chunk_size=chunk_size)

        # Build output table for this row group
        out_data = {rt: rg_combos[:, i] for i, rt in enumerate(rtypes)}
        out_data['score'] = rg_scores
        writer.write_table(pa.table(out_data, schema=out_schema))

        total_scored += n_rg
        part_rows += n_rg
        print(f"    Row group {rg_idx + 1}/{n_row_groups}: "
              f"{n_rg:,} scored ({total_scored:,} total)", flush=True)

        # Estimate whether current part exceeds size limit
        # Use bytes_per_row from previous parts if available, else estimate
        # conservatively (~7 bytes/cell with snappy compression on float64)
        est_bytes_per_row = bytes_per_row if bytes_per_row else (n_res + 1) * 7
        est_size = part_rows * est_bytes_per_row

        if est_size >= max_file_bytes and rg_idx < n_row_groups - 1:
            # Close current part and start a new one
            writer.close()
            actual_size = os.path.getsize(current_path)
            bytes_per_row = actual_size / part_rows  # calibrate for next part
            size_mb = actual_size / (1024 * 1024)
            print(f"    Part {part_idx}: {size_mb:.1f} MB ({part_rows:,} rows) "
                  f"— rotating to next part", flush=True)
            part_idx += 1
            current_path = _hybrid_cache_part_path(iso, part_idx)
            part_paths.append(current_path)
            writer = pq.ParquetWriter(current_path, out_schema, compression='snappy')
            part_rows = 0

        # Free memory
        del rg_table, rg_combos, rg_scores, out_data

    writer.close()

    # If only one part and it's small enough, rename to single-file format
    if len(part_paths) == 1:
        final_path = _hybrid_cache_path(iso)
        os.rename(part_paths[0], final_path)
        size_mb = os.path.getsize(final_path) / (1024 * 1024)
        print(f"  {iso}: Hybrid scored database → {final_path} ({size_mb:.1f} MB)")
    else:
        total_mb = sum(os.path.getsize(p) / (1024 * 1024) for p in part_paths)
        print(f"  {iso}: Hybrid scored database → {len(part_paths)} parts "
              f"({total_mb:.1f} MB total)")
        for p in part_paths:
            sz = os.path.getsize(p) / (1024 * 1024)
            print(f"    {os.path.basename(p)}: {sz:.1f} MB")

    return total_scored


def score_mixes(iso, combos, demand_arr, supply_matrix, chunk_size=20000):
    """Score combos in memory-bounded chunks. Returns scores array (N,)."""
    n_combos = len(combos)
    mem_full_gb = n_combos * 8760 * 8 / (1024**3)
    mem_chunk_gb = min(chunk_size, n_combos) * 8760 * 8 / (1024**3)
    n_chunks = (n_combos + chunk_size - 1) // chunk_size

    print(f"  {iso}: Full array would be {mem_full_gb:.1f} GiB — scoring in "
          f"{n_chunks} chunks of {chunk_size:,} ({mem_chunk_gb:.1f} GiB peak)")

    return s1.batch_hourly_scores(demand_arr, supply_matrix, combos,
                                  chunk_size=chunk_size)


def save_scored(iso, combos, scores, hybrid=False):
    """Write scored mixes to cache parquet."""
    rtypes = s1.get_resource_types(iso, include_hybrids=hybrid)
    os.makedirs(s1.STEP1_RAW_PFS_PARQUET_DIR, exist_ok=True)

    data = {rt: combos[:, i] for i, rt in enumerate(rtypes)}
    data['score'] = scores

    table = pa.table(data)
    out_path = _hybrid_cache_path(iso) if hybrid else s1._coarse_cache_path(iso)
    pq.write_table(table, out_path, compression='snappy')

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    label = "Hybrid scored" if hybrid else "Scored"
    print(f"  {iso}: {label} database → {out_path} ({size_mb:.1f} MB)")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Step 1b: Score mix combinations in chunks → scored parquet.",
    )
    parser.add_argument(
        "--iso", required=True,
        help="ISO name or 'ALL' to run all ISOs",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=20000,
        help="Rows per scoring chunk (default 20000 ≈ 1.4 GiB peak)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rescore even if coarse cache already exists",
    )
    parser.add_argument(
        "--hybrid", action="store_true",
        help="Score hybrid mixes (from step1a --hybrid output)",
    )
    parser.add_argument(
        "--max-file-mb", type=int, default=45,
        help="Max output file size in MB before splitting into parts (default 45, 0=unlimited)",
    )
    args = parser.parse_args()

    isos = list(s1.ISOS) if args.iso.upper() == 'ALL' else [args.iso.upper()]

    for iso in isos:
        if iso not in s1.ISOS:
            print(f"ERROR: Unknown ISO '{iso}'. Valid: {', '.join(s1.ISOS)}")
            sys.exit(1)

    # Check for existing caches
    if not args.force:
        cache_fn = _hybrid_cache_path if args.hybrid else s1._coarse_cache_path
        skip = [iso for iso in isos if os.path.exists(cache_fn(iso))]
        if skip:
            label = "hybrid scored" if args.hybrid else "scored"
            print(f"Skipping ISOs with existing {label} cache: {', '.join(skip)} "
                  f"(use --force to rescore)")
            isos = [iso for iso in isos if iso not in skip]
            if not isos:
                print("Nothing to do.")
                return

    mode_label = "HYBRID " if args.hybrid else ""
    print("=" * 70)
    print(f"  Step 1b — Score {mode_label}Mix Combinations")
    print(f"  ISOs: {', '.join(isos)}")
    print(f"  Chunk size: {args.chunk_size:,}")
    if args.hybrid:
        print(f"  Hybrid profiles: data/hybrid_profiles/{{ISO}}_hybrid_profiles.npz")
    print("=" * 70)

    # Load EIA data once for all ISOs
    print("\nLoading EIA data...")
    demand_data, gen_profiles, _, _ = s1.load_data()

    # Pre-load hybrid profiles if needed
    hybrid_profiles_map = {}
    if args.hybrid:
        print("Loading hybrid profiles...")
        for iso in isos:
            hybrid_profiles_map[iso] = s1.load_hybrid_profiles(iso)
            n_types = len(hybrid_profiles_map[iso])
            print(f"  {iso}: {n_types} hybrid profile types loaded")

    for iso in isos:
        print(f"\n{'─' * 50}")
        print(f"  Processing {iso}")
        t0 = time.time()

        # Prepare supply profiles (base + hybrid if applicable)
        demand_norm = demand_data[iso]["normalized"]
        supply_profiles = s1.get_supply_profiles(iso, gen_profiles)

        if args.hybrid:
            demand_arr, supply_matrix = s1.prepare_numpy_profiles(
                iso, demand_norm, supply_profiles,
                include_hybrids=True,
                hybrid_profiles=hybrid_profiles_map[iso])

            # Use streaming scorer for hybrid (avoids loading all mixes into RAM)
            score_and_save_hybrid_streaming(
                iso, demand_arr, supply_matrix,
                chunk_size=args.chunk_size, max_file_mb=args.max_file_mb)
        else:
            demand_arr, supply_matrix = s1.prepare_numpy_profiles(
                iso, demand_norm, supply_profiles)

            combos = load_mixes(iso)

            # Verify dimensions match
            n_res_combos = combos.shape[1]
            n_res_profiles = supply_matrix.shape[0]
            if n_res_combos != n_res_profiles:
                print(f"  ERROR: Mix columns ({n_res_combos}) != profile rows "
                      f"({n_res_profiles}). Aborting.")
                sys.exit(1)

            scores = score_mixes(iso, combos, demand_arr, supply_matrix,
                                 chunk_size=args.chunk_size)
            save_scored(iso, combos, scores, hybrid=False)

        elapsed = time.time() - t0
        print(f"  {iso}: Done in {elapsed:.1f}s")

    suffix = " (hybrid)" if args.hybrid else ""
    print(f"\n{'=' * 70}")
    print(f"  All done. Scored databases{suffix} ready for downstream steps")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
