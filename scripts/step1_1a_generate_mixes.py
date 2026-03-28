#!/usr/bin/env python3
"""Step 1a: Generate resource mix combinations → static parquet.

Pure combinatorics — no EIA data loading, no scoring. Produces every
resource fraction combination at 5% step for the specified ISO, plus
any seed combos from prior research.

When prior windows are available (from step1_prior_windows.py), the search
space is narrowed to the union of prior EF bounds + 15pp buffer, plus 100
scout mixes to catch regime shifts. This typically saves ~30% of the
combinatorial space.

Output: data/step1-pfs/{ISO}_mixes.parquet
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


def _generate_family_compositions(family_cap, hybrid_cap, step):
    """Generate all (standalone, batt4, batt8) compositions within a family budget.

    For each family total (0, step, 2*step, ..., family_cap), enumerate all
    3-way splits where standalone + batt4 + batt8 = total, and each hybrid
    type ≤ hybrid_cap.

    Returns list of (standalone, batt4, batt8) tuples.
    """
    compositions = []
    for total in range(0, family_cap + 1, step):
        for b4 in range(0, min(hybrid_cap, total) + 1, step):
            for b8 in range(0, min(hybrid_cap, total - b4) + 1, step):
                standalone = total - b4 - b8
                if standalone >= 0:
                    compositions.append((standalone, b4, b8))
    return compositions


def generate_hybrid_mixes(iso):
    """Generate coarse mix combos with hybrid family splits.

    Uses tightened per-ISO windows:
      - Solar family: (solar, solar_batt4, solar_batt8) compositions at 10% steps
      - Wind family: (wind, wind_batt4, wind_batt8) compositions at 10% steps
      - Clean firm: 10% steps within tightened per-ISO window
      - CCS: 5% steps within tightened per-ISO cap
      - Hydro: fixed at observed DG value (not a dimension)
      - OSW, Geo: 5% steps (unchanged)

    Returns combos array (N, n_res) with columns matching
    get_resource_types(iso, include_hybrids=True).
    """
    step_h = s1.HYBRID_FAMILY_STEP  # 10% for family splits
    h_cap = s1.HYBRID_MAX_PER_TYPE  # 40% max per hybrid type

    # ── Family compositions ──
    s_fam_cap = s1.SOLAR_FAMILY_CAP[iso]
    w_fam_cap = s1.WIND_FAMILY_CAP[iso]
    solar_comps = _generate_family_compositions(s_fam_cap, h_cap, step_h)
    wind_comps = _generate_family_compositions(w_fam_cap, h_cap, step_h)

    # ── Other dimensions ──
    cf_lo, cf_hi = s1.CF_WINDOW[iso]
    cf_step = s1.CF_COARSE_STEP  # 10%
    cf_vals = list(range(cf_lo, cf_hi + 1, cf_step))

    ccs_cap = s1.CCS_CAP[iso]
    ccs_vals = list(range(0, ccs_cap + 1, 5))

    hydro_val = s1.HYDRO_FIXED[iso]  # single fixed value

    osw_cap = int(s1.OFFSHORE_WIND_CAP_PCT.get(iso, 0)) if iso in s1.OFFSHORE_ISOS else 0
    osw_vals = list(range(0, osw_cap + 1, 5)) if osw_cap > 0 else [0]
    if osw_cap > 0 and osw_cap not in osw_vals:
        osw_vals.append(osw_cap)

    geo_vals = list(range(0, s1.GEO_CAP_PCT + 1, 5)) if iso == 'CAISO' else [0]

    # ── Build column order matching get_resource_types(iso, include_hybrids=True) ──
    # Base: clean_firm, solar, wind, hydro, [offshore_wind], [geothermal]
    # Hybrid: solar_batt4, solar_batt8, wind_batt4, wind_batt8
    rtypes = s1.get_resource_types(iso, include_hybrids=True)
    n_res = len(rtypes)

    # Column index map
    idx = {rt: i for i, rt in enumerate(rtypes)}

    print(f"  {iso}: Generating hybrid grid — "
          f"solar_fam={len(solar_comps)} splits (cap {s_fam_cap}%), "
          f"wind_fam={len(wind_comps)} splits (cap {w_fam_cap}%), "
          f"CF={len(cf_vals)} ({cf_lo}-{cf_hi}% @{cf_step}%), "
          f"CCS={len(ccs_vals)} (0-{ccs_cap}% @5%), "
          f"OSW={len(osw_vals)}, Geo={len(geo_vals)}")

    # Pre-compute raw grid size
    raw_size = (len(solar_comps) * len(wind_comps) * len(cf_vals) *
                len(ccs_vals) * len(osw_vals) * len(geo_vals))
    print(f"  {iso}: Raw Cartesian: {raw_size:,} — generating + filtering...")

    # ── Generate via chunked numpy to limit peak memory ──
    solar_arr = np.array(solar_comps, dtype=np.float64)  # (Ns, 3)
    wind_arr = np.array(wind_comps, dtype=np.float64)    # (Nw, 3)
    cf_arr = np.array(cf_vals, dtype=np.float64)
    ccs_arr = np.array(ccs_vals, dtype=np.float64)
    osw_arr = np.array(osw_vals, dtype=np.float64)
    geo_arr = np.array(geo_vals, dtype=np.float64)

    # Pre-compute the "other" grid (cf, ccs, osw, geo) once
    grids = np.meshgrid(cf_arr, ccs_arr, osw_arr, geo_arr, indexing='ij')
    other_combos = np.column_stack([g.ravel() for g in grids])
    other_sums = other_combos.sum(axis=1)
    n_other = len(other_combos)

    pcap = s1.TOTAL_PROCUREMENT_CAP

    # Two-pass: count first, then fill pre-allocated array
    # Pass 1: count valid combos
    total_count = 0
    sw_sums = []  # cache (si, wi) → sw_sum for valid pairs
    for si in range(len(solar_arr)):
        s_sum = solar_arr[si].sum()
        for wi in range(len(wind_arr)):
            w_sum = wind_arr[wi].sum()
            sw_sum = s_sum + w_sum + hydro_val
            if sw_sum + cf_arr[0] > pcap:
                continue
            n_keep = int(np.sum((sw_sum + other_sums > 0) &
                                (sw_sum + other_sums <= pcap)))
            if n_keep > 0:
                sw_sums.append((si, wi, sw_sum, n_keep))
                total_count += n_keep

    print(f"  {iso}: {total_count:,} valid combos from "
          f"{len(sw_sums):,} (solar,wind) pairs — allocating...")

    if total_count == 0:
        return np.empty((0, n_res), dtype=np.float64)

    # Pass 2: fill pre-allocated array
    combos = np.empty((total_count, n_res), dtype=np.float64)
    pos = 0
    for si, wi, sw_sum, n_keep in sw_sums:
        mask = (sw_sum + other_sums > 0) & (sw_sum + other_sums <= pcap)
        other_kept = other_combos[mask]

        end = pos + n_keep
        combos[pos:end, idx['clean_firm']] = other_kept[:, 0]
        combos[pos:end, idx['solar']] = solar_arr[si, 0]
        combos[pos:end, idx['wind']] = wind_arr[wi, 0]
        combos[pos:end, idx['hydro']] = hydro_val
        if 'offshore_wind' in idx:
            combos[pos:end, idx['offshore_wind']] = other_kept[:, 2]
        if 'geothermal' in idx:
            combos[pos:end, idx['geothermal']] = other_kept[:, 3]
        combos[pos:end, idx['solar_batt4']] = solar_arr[si, 1]
        combos[pos:end, idx['solar_batt8']] = solar_arr[si, 2]
        combos[pos:end, idx['wind_batt4']] = wind_arr[wi, 1]
        combos[pos:end, idx['wind_batt8']] = wind_arr[wi, 2]
        pos = end

    print(f"  {iso}: {len(combos):,} hybrid mixes after filtering "
          f"(procurement cap ≤ {pcap}%)")
    return combos


def hybrid_mixes_path(iso):
    """Path for the hybrid mixes parquet."""
    return os.path.join(s1.STEP1_RAW_PFS_PARQUET_DIR, f'{iso}_hybrid_mixes.parquet')


def save_hybrid_mixes(iso, combos):
    """Write hybrid mix combinations to parquet."""
    rtypes = s1.get_resource_types(iso, include_hybrids=True)
    os.makedirs(s1.STEP1_RAW_PFS_PARQUET_DIR, exist_ok=True)

    data = {rt: combos[:, i] for i, rt in enumerate(rtypes)}
    table = pa.table(data)
    out_path = hybrid_mixes_path(iso)
    pq.write_table(table, out_path, compression='snappy')

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  {iso}: {len(combos):,} hybrid mixes → {out_path} ({size_mb:.1f} MB)")
    return out_path


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
    parser.add_argument(
        "--hybrid", action="store_true",
        help="Generate hybrid co-located mixes (solar+batt, wind+batt families)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print grid sizes without generating (for QA/QC)",
    )
    args = parser.parse_args()

    isos = list(s1.ISOS) if args.iso.upper() == 'ALL' else [args.iso.upper()]

    for iso in isos:
        if iso not in s1.ISOS:
            print(f"ERROR: Unknown ISO '{iso}'. Valid: {', '.join(s1.ISOS)}")
            sys.exit(1)

    # ── Hybrid mode ──
    if args.hybrid:
        path_fn = hybrid_mixes_path
        if not args.force and not args.dry_run:
            skip = [iso for iso in isos if os.path.exists(path_fn(iso))]
            if skip:
                print(f"Skipping ISOs with existing hybrid mixes: {', '.join(skip)} "
                      f"(use --force to regenerate)")
                isos = [iso for iso in isos if iso not in skip]
                if not isos:
                    print("Nothing to do.")
                    return

        print("=" * 70)
        print(f"  Step 1a — Generate HYBRID Mix Combinations")
        print(f"  ISOs: {', '.join(isos)}")
        print(f"  Family step: {s1.HYBRID_FAMILY_STEP}%  "
              f"Hybrid cap: {s1.HYBRID_MAX_PER_TYPE}%  "
              f"CF step: {s1.CF_COARSE_STEP}%")
        print("=" * 70)

        for iso in isos:
            t0 = time.time()
            combos = generate_hybrid_mixes(iso)
            rtypes = s1.get_resource_types(iso, include_hybrids=True)
            if args.dry_run:
                print(f"  {iso}: DRY RUN — {len(combos):,} mixes ({len(rtypes)}D), "
                      f"columns: {rtypes}")
                # Print family composition counts for verification
                s_comps = _generate_family_compositions(
                    s1.SOLAR_FAMILY_CAP[iso], s1.HYBRID_MAX_PER_TYPE,
                    s1.HYBRID_FAMILY_STEP)
                w_comps = _generate_family_compositions(
                    s1.WIND_FAMILY_CAP[iso], s1.HYBRID_MAX_PER_TYPE,
                    s1.HYBRID_FAMILY_STEP)
                print(f"          Solar family: {len(s_comps)} compositions")
                print(f"          Wind family:  {len(w_comps)} compositions")
                # Sample some compositions
                for label, comps in [("Solar", s_comps[:5]), ("Wind", w_comps[:5])]:
                    for c in comps:
                        print(f"            {label}: stand={c[0]}% b4={c[1]}% b8={c[2]}%")
                continue

            save_hybrid_mixes(iso, combos)
            elapsed = time.time() - t0
            print(f"  {iso}: {len(combos):,} hybrid combos ({len(rtypes)}D) "
                  f"in {elapsed:.1f}s")

        print(f"\n  Done. Hybrid mixes generated.")
        return

    # ── Standard (non-hybrid) mode ──
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
