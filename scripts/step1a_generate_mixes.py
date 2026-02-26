#!/usr/bin/env python3
"""Step 1a: Generate resource mix combinations → static parquet.

Pure combinatorics — no EIA data loading, no scoring. Produces every
resource fraction combination at 5% step for the specified ISO, plus
any seed combos from prior research.

Output: data/step1-pfs-parquets/{ISO}_mixes.parquet
  Columns: clean_firm, solar, wind, hydro, [geothermal]

Memory: Trivial (~64 MB for CAISO 1.6M × 5 float64 cols).

Usage:
  python scripts/step1a_generate_mixes.py --iso CAISO
  python scripts/step1a_generate_mixes.py --iso ALL
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


def mixes_path(iso):
    """Path for the static mixes parquet."""
    return os.path.join(s1.STEP1_RAW_PFS_PARQUET_DIR, f'{iso}_mixes.parquet')


def generate_mixes(iso):
    """Generate all coarse combos for an ISO.

    Returns combos array (N, n_res).
    """
    combos = s1.generate_resource_combos(iso, step=5)
    seeds = s1.get_seed_combos(iso)
    if len(seeds) > 0:
        combos = np.vstack([combos, seeds])
        combos = np.unique(combos, axis=0)
    return combos


def save_mixes(iso, combos):
    """Write mix combinations to parquet (no scores)."""
    rtypes = s1.get_resource_types(iso)
    os.makedirs(s1.STEP1_RAW_PFS_PARQUET_DIR, exist_ok=True)

    data = {rt: combos[:, i] for i, rt in enumerate(rtypes)}
    table = pa.table(data)
    out_path = mixes_path(iso)
    pq.write_table(table, out_path, compression='snappy')

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  {iso}: {len(combos):,} mixes → {out_path} ({size_mb:.1f} MB)")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Step 1a: Generate resource mix combinations → parquet.",
    )
    parser.add_argument(
        "--iso", required=True,
        help="ISO name or 'ALL' to run all ISOs",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate even if mixes parquet already exists",
    )
    args = parser.parse_args()

    isos = list(s1.ISOS) if args.iso.upper() == 'ALL' else [args.iso.upper()]

    for iso in isos:
        if iso not in s1.ISOS:
            print(f"ERROR: Unknown ISO '{iso}'. Valid: {', '.join(s1.ISOS)}")
            sys.exit(1)

    if not args.force:
        skip = [iso for iso in isos if os.path.exists(mixes_path(iso))]
        if skip:
            print(f"Skipping ISOs with existing mixes: {', '.join(skip)} "
                  f"(use --force to regenerate)")
            isos = [iso for iso in isos if iso not in skip]
            if not isos:
                print("Nothing to do.")
                return

    print("=" * 70)
    print(f"  Step 1a — Generate Mix Combinations")
    print(f"  ISOs: {', '.join(isos)}")
    print("=" * 70)

    for iso in isos:
        t0 = time.time()
        rtypes = s1.get_resource_types(iso)
        combos = generate_mixes(iso)
        save_mixes(iso, combos)
        elapsed = time.time() - t0
        print(f"  {iso}: {len(combos):,} combos ({len(rtypes)}D) in {elapsed:.1f}s")

    print(f"\n  Done. Mixes ready for step1b_score_mixes.py")


if __name__ == "__main__":
    main()
