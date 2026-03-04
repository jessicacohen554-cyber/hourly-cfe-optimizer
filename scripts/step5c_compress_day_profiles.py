#!/usr/bin/env python3
"""
Step 5: Compressed Day Profile Generator
=========================================
Generates 24-hour representative day profiles for every unique resource mix
in the dashboard. Replays the 8760-hour physics (demand, generation, storage
dispatch) and compresses to hour-of-day annualized sums.

Pipeline position: Step 5 of 5
  Step 1 — PFS Generator (physics)
  Step 2 — Efficient Frontier extraction
  Step 3 — Cost optimization
  Step 4 — Post-processing (CO2, MAC, NEISO gas)
  Step 5 — Compressed day profiles (this file)

Input:
  - data/eia-930/eia_demand_profiles.json (8760 demand)
  - data/eia-930/eia_generation_profiles.json (solar, wind, hydro hourly)
  - dashboard/overprocure_results.json (all scenarios with resource mixes)

Output:
  - dashboard/compressed_day_profiles.json
    Keyed by ISO → threshold → mix_key → {demand, matched, surplus, charges, gap}
    Each array is 24 values in UTC (hour-of-day sums across 365 days, normalized)
    Chart displays as MWh (annual sum) = value * annual_demand_mwh
"""

import json
import os
import sys
import time
import numpy as np

# ============================================================================
# CONSTANTS
# ============================================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
sys.path.insert(0, os.path.dirname(DATA_DIR))
from parquet_io import load_from_parquets, find_input_dir

from dispatch_utils import (
    H, ISOS, RESOURCE_TYPES, CACHE_VERSION, DISPATCH_ORDER,
    load_common_data, get_supply_profiles, get_demand_profile,
    build_supply_matrix, reconstruct_hourly_dispatch,
    _archetype_key, load_dispatch_cache,
)

DATA_YEAR = '2025'


# ============================================================================
# DISPATCH — uses dispatch_utils cache (populated by step4_build_dispatch_cache)
# ============================================================================

def dispatch_from_cache(iso, mix, battery_pct, battery8_pct,
                        ldes_pct, h2_pct, demand_norm, gen_profiles, dispatch_cache):
    """Look up or compute dispatch and return compressed-day-format result dict.

    Tries the pre-built dispatch cache (step4_build_dispatch_cache) first. Falls back to live computation
    via reconstruct_hourly_dispatch(detailed=True) if cache miss.
    """
    resource_pcts = {
        'clean_firm': mix.get('clean_firm', 0),
        'solar': mix.get('solar', 0),
        'wind': mix.get('wind', 0),
        'ccs_ccgt': mix.get('ccs_ccgt', 0),
        'hydro': mix.get('hydro', 0),
    }
    key = _archetype_key(iso, resource_pcts, 100,
                         battery_pct, battery8_pct, ldes_pct)

    demand_arr = np.array(demand_norm[:H], dtype=np.float64)

    if dispatch_cache is not None and key in dispatch_cache:
        cached = dispatch_cache[key]
        matched = {}
        surplus = {}
        for rtype in RESOURCE_TYPES:
            mk = f'matched_{rtype}'
            sk = f'surplus_{rtype}'
            matched[rtype] = cached[mk] if mk in cached else np.zeros(H, dtype=np.float64)
            surplus[rtype] = cached[sk] if sk in cached else np.zeros(H, dtype=np.float64)

        return {
            'demand': demand_arr,
            'matched': matched,
            'surplus': surplus,
            'battery_matched': cached.get('battery4_profile', np.zeros(H, dtype=np.float64)),
            'battery_charge': cached.get('battery4_charge', np.zeros(H, dtype=np.float64)),
            'ldes_matched': cached.get('ldes_profile', np.zeros(H, dtype=np.float64)),
            'ldes_charge': cached.get('ldes_charge', np.zeros(H, dtype=np.float64)),
            'gap': cached.get('residual_demand', np.zeros(H, dtype=np.float64)),
        }

    # Cache miss — compute live
    supply_profiles = get_supply_profiles(iso, gen_profiles)
    supply_matrix = build_supply_matrix(supply_profiles)
    result = reconstruct_hourly_dispatch(
        demand_norm, supply_profiles, resource_pcts, 100,
        battery_pct, battery8_pct, ldes_pct,
        supply_matrix=supply_matrix, detailed=True,
        h2_dispatch_pct=h2_pct)

    matched = {}
    surplus = {}
    for rtype in RESOURCE_TYPES:
        matched[rtype] = result[f'matched_{rtype}']
        surplus[rtype] = result[f'surplus_{rtype}']

    return {
        'demand': demand_arr,
        'matched': matched,
        'surplus': surplus,
        'battery_matched': result['battery4_profile'],
        'battery_charge': result['battery4_charge'],
        'ldes_matched': result['ldes_profile'],
        'ldes_charge': result['ldes_charge'],
        'gap': result['residual_demand'],
    }


def compress_to_24h(result):
    """
    Compress 8760-hour dispatch result to 24 hour-of-day sums.

    Output values are normalized: each value is the sum across 365 days
    for that hour-of-day. The chart converts to MWh (annual sum) via:
        MWh = value * annual_demand_mwh

    All arrays in UTC (chart handles UTC → local rotation).
    """
    def sum_by_hod(arr):
        """Sum 8760 array by hour-of-day (0-23), producing 24 values."""
        a = np.asarray(arr[:H], dtype=np.float64)
        # Reshape to (365, 24) and sum across days — vectorized
        return a.reshape(365, 24).sum(axis=0).tolist()

    compressed = {
        'demand': sum_by_hod(result['demand']),
        'matched': {},
        'surplus': {},
        'battery_charge': sum_by_hod(result['battery_charge']),
        'ldes_charge': sum_by_hod(result['ldes_charge']),
        'gap': sum_by_hod(result['gap']),
    }

    for r in ['clean_firm', 'ccs_ccgt', 'solar', 'wind', 'hydro']:
        compressed['matched'][r] = sum_by_hod(result['matched'][r])
        compressed['surplus'][r] = sum_by_hod(result['surplus'][r])

    # Battery and LDES as matched resources
    compressed['matched']['battery'] = sum_by_hod(result['battery_matched'])
    compressed['matched']['ldes'] = sum_by_hod(result['ldes_matched'])

    return compressed


def round_arrays(compressed, decimals=5):
    """Round all arrays to save space in JSON output — numpy vectorized."""
    def r(arr):
        return np.round(arr, decimals).tolist() if isinstance(arr, np.ndarray) \
            else np.round(np.asarray(arr, dtype=np.float64), decimals).tolist()

    out = {
        'demand': r(compressed['demand']),
        'matched': {k: r(v) for k, v in compressed['matched'].items()},
        'surplus': {k: r(v) for k, v in compressed['surplus'].items()},
        'battery_charge': r(compressed['battery_charge']),
        'ldes_charge': r(compressed['ldes_charge']),
        'gap': r(compressed['gap']),
    }
    return out


# ============================================================================
# MIX KEY — unique identifier for a resource mix
# ============================================================================

def mix_key(mix, battery_pct, ldes_pct, h2_pct=0):
    """Generate a compact string key for a unique mix configuration.

    In v5.0, procurement_pct is always 100% (baked into resource percentages),
    so it's no longer part of the key.
    """
    cf = mix.get('clean_firm', 0)
    s = mix.get('solar', 0)
    w = mix.get('wind', 0)
    c = mix.get('ccs_ccgt', 0)
    h = mix.get('hydro', 0)
    return f"{cf}_{s}_{w}_{c}_{h}_{battery_pct}_{ldes_pct}_{h2_pct}"


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("Step 6: Compressed Day Profile Generator")
    print("=" * 70)
    print("Generates profiles for ALL feasible mixes (not just base-year optimal).")
    print("Uses dispatch cache from step4_build_dispatch_cache when available; falls back to live compute.\n")

    # Load hourly profiles via dispatch_utils (single source of truth)
    print("Loading data...")
    demand_data, gen_profiles, _, _ = load_common_data()

    # Load results — parquets preferred, JSON fallback
    input_dir = find_input_dir(ISOS)
    if input_dir:
        print(f"\nLoading from parquets: {input_dir}")
        results = load_from_parquets(input_dir, ISOS)
    elif os.path.exists('dashboard/overprocure_results.json'):
        print("\nLoading overprocure_results.json...")
        with open('dashboard/overprocure_results.json') as f:
            results = json.load(f)
    else:
        print("ERROR: No parquets or overprocure_results.json found")
        sys.exit(1)

    # Dashboard thresholds (same as shared-data.js FEASIBLE_MIXES)
    DASHBOARD_THRESHOLDS = ['50', '55', '60', '65', '70', '75', '80', '85', '87.5', '90', '92.5', '95', '97.5', '99', '99.5', '99.9', '99.99']

    output = {}
    total_mixes = 0
    total_computed = 0

    for iso in ISOS:
        print(f"\n{'='*50}")
        print(f"  {iso}")
        print(f"{'='*50}")

        iso_results = results['results'].get(iso, {})
        if not iso_results:
            print(f"  No results for {iso}, skipping")
            continue

        # Load dispatch cache from step4_build_dispatch_cache (v2 required for detailed fields)
        dispatch_cache = load_dispatch_cache(iso, require_version=CACHE_VERSION)
        if dispatch_cache:
            print(f"  Dispatch cache: {len(dispatch_cache)} archetypes (v{CACHE_VERSION})")
        else:
            print(f"  No v{CACHE_VERSION} cache — will compute live")

        demand_norm, _ = get_demand_profile(iso, demand_data)

        # Collect ALL unique mixes from feasible_mixes (not just scenario-selected)
        # These are the mixes that findOptimalMix can select under any cost/growth combo
        # Supports both columnar format (new: {col: [vals...]}) and row format (old: [{col: val}...])
        unique_mixes = {}  # mix_key → (mix_dict, batt, batt8, ldes, h2)

        thresholds = iso_results.get('thresholds', {})
        for t_str in DASHBOARD_THRESHOLDS:
            t_data = thresholds.get(t_str, {})
            fmixes = t_data.get('feasible_mixes', {})

            if isinstance(fmixes, dict) and 'clean_firm' in fmixes:
                # Columnar format: {clean_firm: [...], solar: [...], ...}
                n_mixes = len(fmixes['clean_firm'])
                for i in range(n_mixes):
                    rm = {
                        'clean_firm': fmixes['clean_firm'][i],
                        'solar': fmixes['solar'][i],
                        'wind': fmixes['wind'][i],
                        'ccs_ccgt': fmixes['ccs_ccgt'][i],
                        'hydro': fmixes['hydro'][i],
                    }
                    batt = fmixes.get('battery_dispatch_pct', [0] * n_mixes)[i]
                    batt8 = fmixes.get('battery8_dispatch_pct', [0] * n_mixes)[i]
                    ldes = fmixes.get('ldes_dispatch_pct', [0] * n_mixes)[i]
                    h2 = fmixes.get('h2_dispatch_pct', [0] * n_mixes)[i]
                    mk = mix_key(rm, batt, ldes, h2)
                    if mk not in unique_mixes:
                        unique_mixes[mk] = (rm, batt, batt8, ldes, h2)
            elif isinstance(fmixes, list):
                # Legacy row format: [{resource_mix: {...}, ...}, ...]
                for fm in fmixes:
                    rm = fm['resource_mix']
                    batt = fm.get('battery_dispatch_pct', 0)
                    batt8 = fm.get('battery8_dispatch_pct', 0)
                    ldes = fm.get('ldes_dispatch_pct', 0)
                    h2 = fm.get('h2_dispatch_pct', 0)
                    mk = mix_key(rm, batt, ldes, h2)
                    if mk not in unique_mixes:
                        unique_mixes[mk] = (rm, batt, batt8, ldes, h2)

            # Extract from scenarios (parquet format — winning mixes across cost combos)
            scenarios = t_data.get('scenarios', {})
            for sc_key, sc in scenarios.items():
                rm = sc.get('resource_mix', {})
                if not rm or 'clean_firm' not in rm:
                    continue
                batt = sc.get('battery_dispatch_pct', 0)
                batt8 = sc.get('battery8_dispatch_pct', 0)
                ldes = sc.get('ldes_dispatch_pct', 0)
                h2 = sc.get('h2_dispatch_pct', 0)
                mk = mix_key(rm, batt, ldes, h2)
                if mk not in unique_mixes:
                    unique_mixes[mk] = (rm, batt, batt8, ldes, h2)

        n_unique = len(unique_mixes)
        total_mixes += n_unique
        print(f"  {n_unique} unique mixes from feasible_mixes")

        # Dispatch and compress each unique mix
        iso_profiles = {}
        cache_hits = 0
        for i, (mk, (mix_dict, batt, batt8, ldes, h2)) in enumerate(unique_mixes.items()):
            result = dispatch_from_cache(
                iso, mix_dict, batt, batt8, ldes, h2,
                demand_norm, gen_profiles, dispatch_cache)
            compressed = compress_to_24h(result)
            iso_profiles[mk] = round_arrays(compressed)

            if (i + 1) % 500 == 0 or i == n_unique - 1:
                print(f"    Computed {i+1}/{n_unique} profiles")
            total_computed += 1

        output[iso] = {'profiles': iso_profiles}

    # Write output to dashboard (consumed by HTML pages)
    out_path = 'dashboard/compressed_day_profiles.json'
    print(f"\nWriting {out_path}...")
    with open(out_path, 'w') as f:
        json.dump(output, f, separators=(',', ':'))

    # Archive canonical copy to step5 results directory
    step5_dir = os.path.join(DATA_DIR, 'step5-post-processing')
    os.makedirs(step5_dir, exist_ok=True)
    step5_out = os.path.join(step5_dir, 'compressed_day_profiles.json')
    with open(step5_out, 'w') as f:
        json.dump(output, f, separators=(',', ':'))

    file_size = os.path.getsize(out_path) / 1024 / 1024
    elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print(f"  Done! {total_computed} profiles computed for {total_mixes} unique mixes")
    print(f"  Output: {out_path} ({file_size:.1f} MB)")
    print(f"  Archived: {step5_out}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
