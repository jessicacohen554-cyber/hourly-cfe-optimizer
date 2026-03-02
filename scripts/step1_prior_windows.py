#!/usr/bin/env python3
"""Step 1 Prior Windows: Compute search windows from prior EF results.

Reads existing efficient frontier parquets and computes per-ISO, per-threshold
resource mix bounds with buffer. Used by step1a (coarse grid narrowing) and
step1c (zone-specific fine search windows).

Output: data/step1-pfs-parquets/{ISO}_prior_windows.json

Usage:
  python scripts/step1_prior_windows.py --iso CAISO
  python scripts/step1_prior_windows.py --iso ALL
"""

import argparse
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import step1_pfs_generator as s1

try:
    import pyarrow.parquet as pq
except ImportError:
    print("ERROR: pyarrow required. pip install pyarrow")
    sys.exit(1)


# Buffer added to prior [min, max] per resource (absolute percentage points).
# 8pp is generous — zone union across thresholds already widens ranges,
# and the coarse grid (5% step) provides additional coverage beyond EF bounds.
# Previous 15pp produced near-full-range bounds for most zones.
PRIOR_BUFFER_PP = 8

# Zone definitions: (name, score_low, score_high, thresholds_covered)
ZONES = [
    ('A', 0.45, 0.78, [50, 55, 60, 65, 70, 75]),
    ('B', 0.73, 0.93, [75, 80, 85, 87.5, 90]),
    ('C', 0.88, 1.00, [90, 92.5, 95, 97.5, 99, 99.5, 99.9, 99.99]),
]

# Additional thresholds below zone A (no fine search needed, coarse only)
LOW_THRESHOLDS = [10, 20, 30, 40]

EF_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'data', 'step2-ef-parquets')


def _ef_path(iso, threshold):
    """Path to an EF parquet file."""
    t_str = s1._normalize_threshold_str(threshold)
    return os.path.join(EF_DIR, f'step2_ef_{iso}_t{t_str}.parquet')


def compute_prior_windows(iso):
    """Compute search windows from prior EF results for an ISO.

    Returns dict with:
      - union_bounds: per-resource [min, max] across all thresholds
      - zone_{A,B,C}: per-zone resource bounds
      - per_threshold: per-threshold resource bounds
    """
    rtypes = s1.get_resource_types(iso)
    caps = {}
    for rt in rtypes:
        if rt == 'hydro':
            caps[rt] = int(s1.HYDRO_CAPS[iso] + s1.HYDRO_ADDER_PCT)
        elif rt == 'geothermal':
            caps[rt] = s1.GEO_CAP_PCT
        elif rt == 'offshore_wind':
            caps[rt] = int(s1.OFFSHORE_WIND_CAP_PCT.get(iso, 0))
        else:
            caps[rt] = s1.RESOURCE_CAPS[rt]

    per_threshold = {}
    all_mixes = []  # collect all EF mixes for union bounds

    for threshold in s1.THRESHOLDS:
        ef_path = _ef_path(iso, threshold)
        if not os.path.exists(ef_path):
            continue

        try:
            table = pq.read_table(ef_path)
        except Exception:
            continue

        if table.num_rows == 0:
            continue

        bounds = {}
        for rt in rtypes:
            if rt not in table.column_names:
                bounds[rt] = [0, caps[rt]]
                continue
            col = table.column(rt).to_numpy()
            lo = max(0, int(np.min(col)) - PRIOR_BUFFER_PP)
            hi = min(caps[rt], int(np.max(col)) + PRIOR_BUFFER_PP)
            bounds[rt] = [lo, hi]
            all_mixes.append((rt, col))

        per_threshold[str(threshold)] = bounds

    if not per_threshold:
        print(f"  {iso}: No prior EF results found — using full search space")
        return None

    # Union bounds across all thresholds
    union_bounds = {}
    for rt in rtypes:
        all_lo = min(pt[rt][0] for pt in per_threshold.values() if rt in pt)
        all_hi = max(pt[rt][1] for pt in per_threshold.values() if rt in pt)
        union_bounds[rt] = [all_lo, all_hi]

    # Zone-specific bounds (from thresholds covered by each zone)
    zone_bounds = {}
    for zone_name, z_lo, z_hi, z_thresholds in ZONES:
        zone_resource_bounds = {}
        zone_threshold_data = [per_threshold[str(t)] for t in z_thresholds
                               if str(t) in per_threshold]
        if not zone_threshold_data:
            zone_resource_bounds = dict(union_bounds)
        else:
            for rt in rtypes:
                rt_lo = min(td[rt][0] for td in zone_threshold_data if rt in td)
                rt_hi = max(td[rt][1] for td in zone_threshold_data if rt in td)
                zone_resource_bounds[rt] = [rt_lo, rt_hi]

        zone_bounds[f'zone_{zone_name}'] = {
            'score_range': [z_lo, z_hi],
            'thresholds': z_thresholds,
            'bounds': zone_resource_bounds,
        }

    result = {
        'iso': iso,
        'buffer_pp': PRIOR_BUFFER_PP,
        'union_bounds': union_bounds,
        'per_threshold': per_threshold,
        'resource_caps': caps,
    }
    result.update(zone_bounds)

    return result


def save_prior_windows(iso, windows):
    """Save prior windows to JSON."""
    os.makedirs(s1.STEP1_RAW_PFS_PARQUET_DIR, exist_ok=True)
    out_path = os.path.join(s1.STEP1_RAW_PFS_PARQUET_DIR,
                            f'{iso}_prior_windows.json')
    with open(out_path, 'w') as f:
        json.dump(windows, f, indent=2)
    print(f"  {iso}: Prior windows → {out_path}")
    return out_path


def load_prior_windows(iso):
    """Load prior windows from JSON. Returns None if not found."""
    path = os.path.join(s1.STEP1_RAW_PFS_PARQUET_DIR,
                        f'{iso}_prior_windows.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Compute search windows from prior EF results.")
    parser.add_argument("--iso", required=True,
                        help="ISO name or 'ALL'")
    args = parser.parse_args()

    isos = list(s1.ISOS) if args.iso.upper() == 'ALL' else [args.iso.upper()]

    for iso in isos:
        if iso not in s1.ISOS:
            print(f"ERROR: Unknown ISO '{iso}'")
            sys.exit(1)

        print(f"\n{iso}: Computing prior windows...")
        windows = compute_prior_windows(iso)
        if windows is None:
            print(f"  {iso}: No prior EF data available — skipped")
            continue

        save_prior_windows(iso, windows)

        # Summary
        ub = windows['union_bounds']
        print(f"  {iso}: Union bounds:")
        for rt, (lo, hi) in ub.items():
            cap = windows['resource_caps'][rt]
            narrowed = hi < cap
            marker = f" (narrowed from {cap})" if narrowed else ""
            print(f"    {rt}: [{lo}, {hi}]{marker}")

        for zone_key in ['zone_A', 'zone_B', 'zone_C']:
            if zone_key in windows:
                z = windows[zone_key]
                print(f"  {zone_key}: score [{z['score_range'][0]:.2f}, "
                      f"{z['score_range'][1]:.2f}], "
                      f"thresholds {z['thresholds']}")


if __name__ == "__main__":
    main()
