#!/usr/bin/env python3
"""Step 1b: Score mix combinations in chunks → scored parquet database.

Reads the static mixes parquet from step1a ({ISO}_mixes.parquet), loads
EIA demand + generation profiles, and computes each combo's hourly match
score in memory-bounded chunks. Never allocates the full N × 8760 array.

Standard mode:
  Output: data/step1-pfs/{ISO}_coarse_cache.parquet
    Columns: clean_firm, solar, wind, hydro, [offshore_wind], [geothermal], score

Hybrid mode (--hybrid):
  Loads hybrid mixes from {ISO}_hybrid_mixes.parquet (4 extra columns:
  solar_batt4, solar_batt8, wind_batt4, wind_batt8) and hybrid 8760
  profiles from data/hybrid_profiles/{ISO}_hybrid_profiles.npz.
  Output: data/step1-pfs/{ISO}_hybrid_coarse_cache.parquet

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
    """Path for the hybrid mixes parquet from step1a --hybrid."""
    return os.path.join(s1.STEP1_RAW_PFS_PARQUET_DIR, f'{iso}_hybrid_mixes.parquet')


def _hybrid_cache_path(iso):
    """Path for the hybrid scored cache."""
    return os.path.join(s1.STEP1_RAW_PFS_PARQUET_DIR, f'{iso}_hybrid_coarse_cache.parquet')


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

        # Load mixes
        if args.hybrid:
            combos = load_hybrid_mixes(iso)
        else:
            combos = load_mixes(iso)

        # Prepare supply profiles (base + hybrid if applicable)
        demand_norm = demand_data[iso]["normalized"]
        supply_profiles = s1.get_supply_profiles(iso, gen_profiles)

        if args.hybrid:
            demand_arr, supply_matrix = s1.prepare_numpy_profiles(
                iso, demand_norm, supply_profiles,
                include_hybrids=True,
                hybrid_profiles=hybrid_profiles_map[iso])
        else:
            demand_arr, supply_matrix = s1.prepare_numpy_profiles(
                iso, demand_norm, supply_profiles)

        # Verify dimensions match
        n_res_combos = combos.shape[1]
        n_res_profiles = supply_matrix.shape[0]
        if n_res_combos != n_res_profiles:
            print(f"  ERROR: Mix columns ({n_res_combos}) != profile rows "
                  f"({n_res_profiles}). Aborting.")
            sys.exit(1)

        # Score in chunks
        scores = score_mixes(iso, combos, demand_arr, supply_matrix,
                             chunk_size=args.chunk_size)

        # Save scored database
        save_scored(iso, combos, scores, hybrid=args.hybrid)

        elapsed = time.time() - t0
        print(f"  {iso}: Done in {elapsed:.1f}s")

    suffix = " (hybrid)" if args.hybrid else ""
    print(f"\n{'=' * 70}")
    print(f"  All done. Scored databases{suffix} ready for downstream steps")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
