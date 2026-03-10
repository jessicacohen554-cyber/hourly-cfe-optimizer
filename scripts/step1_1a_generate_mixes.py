#!/usr/bin/env python3
"""Step 1a: Generate resource mix combinations → static parquet.

Pure combinatorics — no EIA data loading, no scoring. Produces every
resource fraction combination at 5% step for the specified ISO, plus
any seed combos from prior research.

When prior windows are available (from step1_prior_windows.py), the search
space is narrowed to the union of prior EF bounds + 15pp buffer, plus 100
scout mixes to catch regime shifts. This typically saves ~30% of the
combinatorial space.

Output: data/step1-pfs-parquets/{ISO}_mixes.parquet
  Columns: clean_firm, solar, wind, hydro, [geothermal]

Usage:
  python scripts/step1a_generate_mixes.py --iso CAISO
  python scripts/step1a_generate_mixes.py --iso ALL
  python scripts/step1a_generate_mixes.py --iso PJM --use-prior-windows
"""

import argparse
import itertools
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

# Scout mix counts
N_RANDOM_SCOUTS = 50
N_CORNER_SCOUTS = 50


def mixes_path(iso):
    """Path for the static mixes parquet."""
    return os.path.join(s1.STEP1_RAW_PFS_PARQUET_DIR, f'{iso}_mixes.parquet')


def generate_mixes(iso, prior_windows=None):
    """Generate all coarse combos for an ISO.

    If prior_windows is provided, narrows the Cartesian product to union
    bounds and adds scout mixes outside the window.

    Returns combos array (N, n_res).
    """
    if prior_windows:
        combos = _generate_prior_informed(iso, prior_windows)
    else:
        combos = s1.generate_resource_combos(iso, step=5)

    seeds = s1.get_seed_combos(iso)
    if len(seeds) > 0:
        combos = np.vstack([combos, seeds])
        combos = np.unique(combos, axis=0)
    return combos


def _generate_prior_informed(iso, prior_windows):
    """Generate coarse grid narrowed by prior EF bounds + scout mixes."""
    rtypes = s1.get_resource_types(iso)
    n_res = len(rtypes)
    step = 5

    union_bounds = prior_windows.get('union_bounds', {})
    resource_caps = prior_windows.get('resource_caps', {})

    # Build ranges per resource using prior-informed bounds
    hydro_cap = int(s1.HYDRO_CAPS[iso] + s1.HYDRO_ADDER_PCT)
    ranges_informed = []
    ranges_full = []
    for rt in rtypes:
        if rt == 'hydro':
            cap = hydro_cap
        elif rt == 'geothermal':
            cap = s1.GEO_CAP_PCT
        elif rt == 'offshore_wind':
            cap = int(s1.OFFSHORE_WIND_CAP_PCT.get(iso, 0))
        else:
            cap = s1.RESOURCE_CAPS[rt]

        full_range = list(range(0, cap + 1, step))
        if cap not in full_range and cap > 0:
            full_range.append(cap)
        ranges_full.append(full_range)

        if rt in union_bounds:
            lo, hi = union_bounds[rt]
            informed_range = [v for v in full_range if lo <= v <= hi]
            # Always include 0 and cap boundaries
            if 0 not in informed_range:
                informed_range.insert(0, 0)
            ranges_informed.append(sorted(set(informed_range)))
        else:
            ranges_informed.append(full_range)

    # Generate narrowed Cartesian product
    combos_informed = np.array(list(itertools.product(*ranges_informed)),
                               dtype=np.float64)

    # Filter: sum > 0, within procurement cap
    if len(combos_informed) > 0:
        row_sums = combos_informed.sum(axis=1)
        combos_informed = combos_informed[
            (row_sums > 0) & (row_sums <= s1.TOTAL_PROCUREMENT_CAP)]

    # Generate scout mixes outside prior window
    scouts = _generate_scouts(iso, rtypes, ranges_informed, ranges_full)

    if len(scouts) > 0:
        combos = np.vstack([combos_informed, scouts])
        combos = np.unique(combos, axis=0)
    else:
        combos = combos_informed

    n_full = 1
    for r in ranges_full:
        n_full *= len(r)
    savings_pct = (1 - len(combos) / max(n_full, 1)) * 100
    print(f"  {iso}: Prior-informed: {len(combos):,} combos "
          f"(~{savings_pct:.0f}% narrower than full grid)")

    return combos


def _generate_scouts(iso, rtypes, ranges_informed, ranges_full):
    """Generate scout mixes outside the prior-informed window.

    50 random scouts in the outer region + 50 corner/extreme combos.
    """
    n_res = len(rtypes)
    scouts = []

    # Identify "outer region" — values in full range but NOT in informed range
    outer_values = []
    for i in range(n_res):
        informed_set = set(ranges_informed[i])
        outer = [v for v in ranges_full[i] if v not in informed_set]
        outer_values.append(outer)

    # Random scouts: sample from outer values mixed with informed values
    rng = np.random.RandomState(42)  # reproducible
    for _ in range(N_RANDOM_SCOUTS):
        scout = np.zeros(n_res, dtype=np.float64)
        # For each resource, 30% chance of picking an outer value
        for i in range(n_res):
            if outer_values[i] and rng.random() < 0.3:
                scout[i] = rng.choice(outer_values[i])
            else:
                scout[i] = rng.choice(ranges_informed[i])
        if scout.sum() > 0 and scout.sum() <= s1.TOTAL_PROCUREMENT_CAP:
            scouts.append(scout)

    # Corner scouts: extreme combos (max one resource at full cap, rest at 0)
    caps = [max(r) for r in ranges_full]
    for i in range(n_res):
        if caps[i] > 0:
            scout = np.zeros(n_res, dtype=np.float64)
            scout[i] = caps[i]
            if scout.sum() <= s1.TOTAL_PROCUREMENT_CAP:
                scouts.append(scout.copy())

    # Two-resource combos at cap
    for i in range(n_res):
        for j in range(i + 1, n_res):
            if caps[i] > 0 and caps[j] > 0:
                scout = np.zeros(n_res, dtype=np.float64)
                scout[i] = caps[i]
                scout[j] = caps[j]
                if scout.sum() <= s1.TOTAL_PROCUREMENT_CAP:
                    scouts.append(scout.copy())

    # Additional random corners
    n_remaining = N_CORNER_SCOUTS - len(scouts) + N_RANDOM_SCOUTS
    for _ in range(max(0, n_remaining)):
        scout = np.zeros(n_res, dtype=np.float64)
        n_active = rng.randint(1, min(4, n_res + 1))
        active_dims = rng.choice(n_res, size=n_active, replace=False)
        for d in active_dims:
            scout[d] = rng.choice(ranges_full[d])
        if scout.sum() > 0 and scout.sum() <= s1.TOTAL_PROCUREMENT_CAP:
            scouts.append(scout)

    if not scouts:
        return np.empty((0, n_res), dtype=np.float64)

    return np.array(scouts, dtype=np.float64)


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
    parser.add_argument(
        "--use-prior-windows", action="store_true",
        help="Use prior EF results to narrow search space",
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

    # Load prior windows if requested
    prior_windows_map = {}
    if args.use_prior_windows:
        from step1_prior_windows import load_prior_windows
        for iso in isos:
            pw = load_prior_windows(iso)
            if pw:
                prior_windows_map[iso] = pw
                print(f"  {iso}: Prior windows loaded")
            else:
                print(f"  {iso}: No prior windows — using full grid")

    print("=" * 70)
    print(f"  Step 1a — Generate Mix Combinations")
    print(f"  ISOs: {', '.join(isos)}")
    if prior_windows_map:
        print(f"  Prior-informed: {', '.join(prior_windows_map.keys())}")
    print("=" * 70)

    for iso in isos:
        t0 = time.time()
        rtypes = s1.get_resource_types(iso)
        pw = prior_windows_map.get(iso)
        combos = generate_mixes(iso, prior_windows=pw)
        save_mixes(iso, combos)
        elapsed = time.time() - t0
        print(f"  {iso}: {len(combos):,} combos ({len(rtypes)}D) in {elapsed:.1f}s")

    print(f"\n  Done. Mixes ready for step1b_score_mixes.py")


if __name__ == "__main__":
    main()
