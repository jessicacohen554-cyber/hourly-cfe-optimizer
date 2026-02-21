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
  - data/eia_demand_profiles.json (8760 demand)
  - data/eia_generation_profiles.json (solar, wind, hydro hourly)
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
# CONSTANTS (must match step1_pfs_generator.py)
# ============================================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
H = 8760
LEAP_FEB29_START = 1416
PROFILE_YEARS = ['2021', '2022', '2023', '2024', '2025']
DATA_YEAR = '2025'

ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']

# Storage parameters
BATTERY_EFFICIENCY = 0.85
BATTERY_DURATION_HOURS = 4
LDES_EFFICIENCY = 0.50
LDES_DURATION_HOURS = 100
LDES_WINDOW_DAYS = 7

# Nuclear parameters
NUCLEAR_SHARE_OF_CLEAN_FIRM = {
    'CAISO': 0.70, 'ERCOT': 1.0, 'PJM': 1.0, 'NYISO': 1.0, 'NEISO': 1.0,
    'MISO': 1.0, 'SPP': 1.0,
}
NUCLEAR_MONTHLY_CF = {
    'CAISO': {1: 0.94, 2: 0.94, 3: 0.85, 4: 0.75, 5: 0.80, 6: 0.99,
              7: 1.0, 8: 1.0, 9: 0.90, 10: 0.78, 11: 0.82, 12: 0.94},
    'ERCOT': {1: 1.0, 2: 1.0, 3: 0.90, 4: 0.80, 5: 0.89, 6: 0.97,
              7: 0.97, 8: 0.96, 9: 0.88, 10: 0.79, 11: 0.85, 12: 1.0},
    'PJM':   {1: 1.0, 2: 1.0, 3: 0.92, 4: 0.85, 5: 0.87, 6: 0.98,
              7: 0.99, 8: 0.97, 9: 0.93, 10: 0.89, 11: 0.91, 12: 1.0},
    'NYISO': {1: 1.0, 2: 1.0, 3: 0.88, 4: 0.78, 5: 0.81, 6: 0.95,
              7: 0.96, 8: 0.94, 9: 0.85, 10: 0.75, 11: 0.79, 12: 1.0},
    'NEISO': {1: 1.0, 2: 0.99, 3: 0.92, 4: 0.83, 5: 0.88, 6: 0.96,
              7: 0.97, 8: 0.95, 9: 0.88, 10: 0.82, 11: 0.85, 12: 1.0},
    'MISO':  {1: 1.0, 2: 1.0, 3: 0.92, 4: 0.84, 5: 0.87, 6: 0.98,
              7: 0.99, 8: 0.97, 9: 0.93, 10: 0.88, 11: 0.91, 12: 1.0},
    'SPP':   {1: 1.0, 2: 1.0, 3: 0.90, 4: 0.80, 5: 0.88, 6: 0.97,
              7: 0.97, 8: 0.96, 9: 0.88, 10: 0.80, 11: 0.85, 12: 1.0},
}

HYDRO_CAPS = {
    'CAISO': 9.5, 'ERCOT': 0.1, 'PJM': 1.8, 'NYISO': 15.9, 'NEISO': 4.4,
    'MISO': 1.6, 'SPP': 4.3,
}


# ============================================================================
# DATA LOADING (mirrors step1)
# ============================================================================

def _remove_leap_day(profile_8784):
    if len(profile_8784) <= H:
        return list(profile_8784[:H])
    return list(profile_8784[:LEAP_FEB29_START]) + list(profile_8784[LEAP_FEB29_START + 24:H + 24])


def _average_profiles(yearly_profiles):
    if not yearly_profiles:
        return [0.0] * H
    n = len(yearly_profiles)
    avg = [0.0] * H
    for profile in yearly_profiles:
        for h in range(H):
            avg[h] += profile[h]
    for h in range(H):
        avg[h] /= n
    return avg


def _validate_demand_profile(iso, year, profile):
    arr = np.array(profile[:H])
    for hod in range(24):
        vals = arr[hod::24]
        median = np.median(vals)
        if median > 0 and vals.max() > 100 * median:
            return False
    return True


def load_data():
    """Load demand and generation profiles (same logic as step1)."""
    print("Loading data...")
    with open(os.path.join(DATA_DIR, 'eia_demand_profiles.json')) as f:
        demand_raw = json.load(f)
    with open(os.path.join(DATA_DIR, 'eia_generation_profiles.json')) as f:
        gen_raw = json.load(f)

    # Average generation profiles
    gen_profiles = {}
    for iso in ISOS:
        iso_raw = gen_raw.get(iso, {})
        available_years = [y for y in PROFILE_YEARS if y in iso_raw]
        if not available_years:
            raise ValueError(f"No generation profile years found for {iso}")
        all_rtypes = set()
        for y in available_years:
            all_rtypes.update(iso_raw[y].keys())
        gen_profiles[iso] = {}
        for rtype in all_rtypes:
            yearly = []
            for y in available_years:
                raw = iso_raw[y].get(rtype)
                if raw is None:
                    continue
                if len(raw) > H:
                    raw = _remove_leap_day(raw)
                else:
                    raw = list(raw[:H])
                yearly.append(raw)
            if yearly:
                gen_profiles[iso][rtype] = _average_profiles(yearly)
        print(f"  {iso}: gen profiles averaged over {len(available_years)} years")

    # Average demand profiles
    demand_data = {}
    for iso in ISOS:
        iso_data = demand_raw.get(iso, {})
        year_keys = [k for k in iso_data.keys() if k.isdigit()]
        if not year_keys and 'normalized' in iso_data:
            demand_data[iso] = iso_data
            continue

        available_years = [y for y in PROFILE_YEARS if y in iso_data]
        if not available_years:
            raise ValueError(f"No demand data years found for {iso}")
        yearly_norms = []
        for y in available_years:
            raw = iso_data[y].get('normalized', [])
            if len(raw) > H:
                raw = _remove_leap_day(raw)
            else:
                raw = list(raw[:H])
            if _validate_demand_profile(iso, y, raw):
                yearly_norms.append(raw)
        if not yearly_norms:
            raise ValueError(f"All demand data years excluded for {iso}")

        avg_norm = _average_profiles(yearly_norms)
        actuals_year = DATA_YEAR if DATA_YEAR in iso_data else available_years[-1]
        demand_data[iso] = {
            'normalized': avg_norm,
            'total_annual_mwh': iso_data[actuals_year]['total_annual_mwh'],
            'peak_mw': iso_data[actuals_year]['peak_mw'],
        }
        print(f"  {iso}: demand shape averaged, scalars from {actuals_year}")

    return demand_data, gen_profiles


def get_supply_profiles(iso, gen_profiles):
    """Build per-resource 8760 profiles (matches step1 exactly)."""
    profiles = {}

    # Clean firm = nuclear seasonal-derated + geothermal flat blend
    nuc_share = NUCLEAR_SHARE_OF_CLEAN_FIRM.get(iso, 1.0)
    geo_share = 1.0 - nuc_share
    monthly_cf = NUCLEAR_MONTHLY_CF.get(iso, {m: 1.0 for m in range(1, 13)})
    cf_profile = []
    month_hours = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
    hour = 0
    for month_idx, hours_in_month in enumerate(month_hours):
        month_num = month_idx + 1
        nuc_cf = monthly_cf.get(month_num, 1.0)
        blended = nuc_share * nuc_cf + geo_share * 1.0
        for _ in range(hours_in_month):
            if hour < H:
                cf_profile.append(blended / H)
                hour += 1
    while len(cf_profile) < H:
        cf_profile.append(1.0 / H)
    profiles['clean_firm'] = cf_profile[:H]

    # CCS-CCGT = flat baseload (1/H per hour)
    profiles['ccs_ccgt'] = [1.0 / H] * H

    # Solar with DST-aware nighttime correction
    if iso == 'NYISO':
        p = gen_profiles[iso].get('solar_proxy')
        if not p:
            p = gen_profiles['NEISO'].get('solar')
        solar_raw = list(p[:H])
    else:
        solar_raw = list(gen_profiles[iso].get('solar', [0.0] * H)[:H])

    STD_UTC_OFFSETS = {'CAISO': 8, 'ERCOT': 6, 'PJM': 5, 'NYISO': 5, 'NEISO': 5, 'MISO': 6, 'SPP': 6}
    DST_START_DAY, DST_END_DAY = 69, 307
    local_start, local_end = 6, 19
    std_off = STD_UTC_OFFSETS.get(iso, 5)
    for day in range(H // 24):
        ds = day * 24
        is_dst = DST_START_DAY <= day < DST_END_DAY
        utc_off = std_off - (1 if is_dst else 0)
        utc_start = (local_start + utc_off) % 24
        utc_end = (local_end + utc_off) % 24
        for h_utc in range(24):
            idx = ds + h_utc
            if idx < len(solar_raw):
                if utc_start <= utc_end:
                    is_daylight = utc_start <= h_utc <= utc_end
                else:
                    is_daylight = h_utc >= utc_start or h_utc <= utc_end
                if not is_daylight:
                    solar_raw[idx] = 0.0
    profiles['solar'] = solar_raw

    # Wind
    profiles['wind'] = gen_profiles[iso].get('wind', [0.0] * H)[:H]

    # Hydro
    profiles['hydro'] = gen_profiles[iso].get('hydro', [0.0] * H)[:H]

    # Ensure all profiles are exactly H hours, no negatives
    for rtype in ['clean_firm', 'ccs_ccgt', 'solar', 'wind', 'hydro']:
        p = profiles[rtype]
        if len(p) > H:
            p = p[:H]
        elif len(p) < H:
            p = p + [0.0] * (H - len(p))
        profiles[rtype] = [max(0.0, v) for v in p]

    return profiles


# ============================================================================
# HOURLY DISPATCH SIMULATION
# ============================================================================

def simulate_hourly_dispatch(demand_norm, profiles, mix, procurement_pct,
                             battery_pct, ldes_pct):
    """
    Replay 8760-hour physics with per-resource tracking.

    Args:
        demand_norm: normalized demand [8760], sums to ~1.0
        profiles: dict of resource → [8760] normalized profiles
        mix: dict {clean_firm, ccs_ccgt, solar, wind, hydro} → % of mix
        procurement_pct: procurement as % of demand
        battery_pct: battery dispatch % of demand
        ldes_pct: LDES dispatch % of demand

    Returns:
        dict with per-resource matched, surplus, charging, gap arrays [8760]
    """
    demand = np.array(demand_norm[:H], dtype=np.float64)
    proc = procurement_pct / 100.0

    # Per-resource generation arrays (normalized)
    resources = ['clean_firm', 'ccs_ccgt', 'solar', 'wind', 'hydro']
    gen = {}
    for r in resources:
        pct = mix.get(r, 0) / 100.0
        profile = np.array(profiles[r][:H], dtype=np.float64)
        gen[r] = pct * proc * profile

    # Total generation
    total_gen = sum(gen[r] for r in resources)

    # Merit-order dispatch against demand (determines per-resource matched)
    # Order: clean_firm → ccs_ccgt → hydro → wind → solar
    # (baseload first, then must-take variable, then solar last as most curtailable)
    dispatch_order = ['clean_firm', 'ccs_ccgt', 'hydro', 'wind', 'solar']
    matched = {r: np.zeros(H) for r in resources}
    surplus = {r: np.zeros(H) for r in resources}
    remaining_demand = demand.copy()

    for r in dispatch_order:
        dispatched = np.minimum(gen[r], remaining_demand)
        matched[r] = dispatched
        surplus[r] = gen[r] - dispatched
        remaining_demand = remaining_demand - dispatched
        remaining_demand = np.maximum(remaining_demand, 0)

    # Post-dispatch residuals
    total_surplus = sum(surplus[r] for r in resources)
    gap = remaining_demand.copy()

    # Battery dispatch (daily cycle)
    battery_matched = np.zeros(H)
    battery_charge_arr = np.zeros(H)

    if battery_pct > 0:
        # Battery capacity: dispatch_pct × annual demand / 8760 gives avg MW
        # Duration × power = capacity. Power = demand_avg × dispatch_pct/100
        # For 4hr battery: capacity_fraction = 4 * (battery_pct/100) / H
        batt_power = (battery_pct / 100.0) * np.mean(demand)
        batt_capacity = batt_power * BATTERY_DURATION_HOURS
        batt_eff = BATTERY_EFFICIENCY

        for day in range(365):
            ds = day * 24
            de = ds + 24

            # Charge phase: absorb surplus
            stored = 0.0
            for h in range(ds, de):
                s = total_surplus[h]
                if s > 0 and stored < batt_capacity:
                    charge = min(s, batt_power, batt_capacity - stored)
                    stored += charge
                    battery_charge_arr[h] = charge

            # Discharge phase: fill gap
            available = stored * batt_eff
            for h in range(ds, de):
                g = gap[h]
                if g > 0 and available > 0:
                    discharge = min(g, batt_power, available)
                    battery_matched[h] = discharge
                    available -= discharge
                    gap[h] -= discharge

        # Update surplus after battery charging
        total_surplus_post_batt = total_surplus - battery_charge_arr
        total_surplus_post_batt = np.maximum(total_surplus_post_batt, 0)
    else:
        total_surplus_post_batt = total_surplus.copy()

    # LDES dispatch (multi-day rolling window)
    ldes_matched = np.zeros(H)
    ldes_charge_arr = np.zeros(H)

    if ldes_pct > 0:
        ldes_power = (ldes_pct / 100.0) * np.mean(demand)
        ldes_capacity = ldes_power * LDES_DURATION_HOURS
        ldes_eff = LDES_EFFICIENCY
        ldes_window_hours = LDES_WINDOW_DAYS * 24

        soc = 0.0
        n_windows = (H + ldes_window_hours - 1) // ldes_window_hours
        for w in range(n_windows):
            ws = w * ldes_window_hours
            we = min(ws + ldes_window_hours, H)

            # Charge phase
            for h in range(ws, we):
                s = total_surplus_post_batt[h]
                if s > 0 and soc < ldes_capacity:
                    charge = min(s, ldes_power, ldes_capacity - soc)
                    ldes_charge_arr[h] = charge
                    soc += charge

            # Discharge phase
            for h in range(ws, we):
                g = gap[h]
                if g > 0 and soc > 0:
                    available_e = soc * ldes_eff
                    discharge = min(g, ldes_power, available_e)
                    ldes_matched[h] = discharge
                    soc -= discharge / ldes_eff
                    gap[h] -= discharge

    return {
        'demand': demand,
        'matched': matched,           # {resource: [8760]}
        'surplus': surplus,            # {resource: [8760]}
        'battery_matched': battery_matched,
        'battery_charge': battery_charge_arr,
        'ldes_matched': ldes_matched,
        'ldes_charge': ldes_charge_arr,
        'gap': gap,
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
        a = np.array(arr[:H])
        return [float(a[h::24].sum()) for h in range(24)]

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
    """Round all arrays to save space in JSON output."""
    def r(arr):
        return [round(v, decimals) for v in arr]

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

def mix_key(mix, procurement_pct, battery_pct, ldes_pct):
    """Generate a compact string key for a unique mix configuration."""
    cf = mix.get('clean_firm', 0)
    s = mix.get('solar', 0)
    w = mix.get('wind', 0)
    c = mix.get('ccs_ccgt', 0)
    h = mix.get('hydro', 0)
    return f"{cf}_{s}_{w}_{c}_{h}_{procurement_pct}_{battery_pct}_{ldes_pct}"


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("Step 5: Compressed Day Profile Generator")
    print("=" * 70)
    print("Generates profiles for ALL feasible mixes (not just base-year optimal).")
    print("The client-side repricing can select different mixes as demand grows,")
    print("so all feasible_mixes need pre-computed profiles.\n")

    # Load hourly profiles
    demand_data, gen_profiles = load_data()

    # Load results to get ALL feasible mixes
    print("\nLoading overprocure_results.json...")
    with open('dashboard/overprocure_results.json') as f:
        results = json.load(f)

    # Dashboard thresholds (same as shared-data.js FEASIBLE_MIXES)
    DASHBOARD_THRESHOLDS = ['50', '60', '70', '75', '80', '85', '87.5', '90', '92.5', '95', '97.5', '99', '100']

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

        # Build supply profiles for this ISO
        profiles = get_supply_profiles(iso, gen_profiles)
        demand_norm = demand_data[iso]['normalized']

        # Collect ALL unique mixes from feasible_mixes (not just scenario-selected)
        # These are the mixes that findOptimalMix can select under any cost/growth combo
        # Supports both columnar format (new: {col: [vals...]}) and row format (old: [{col: val}...])
        unique_mixes = {}  # mix_key → (mix_dict, proc, batt, ldes)

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
                    proc = fmixes['procurement_pct'][i]
                    batt = fmixes.get('battery_dispatch_pct', [0] * n_mixes)[i]
                    ldes = fmixes.get('ldes_dispatch_pct', [0] * n_mixes)[i]
                    mk = mix_key(rm, proc, batt, ldes)
                    if mk not in unique_mixes:
                        unique_mixes[mk] = (rm, proc, batt, ldes)
            elif isinstance(fmixes, list):
                # Legacy row format: [{resource_mix: {...}, ...}, ...]
                for fm in fmixes:
                    rm = fm['resource_mix']
                    proc = fm['procurement_pct']
                    batt = fm.get('battery_dispatch_pct', 0)
                    ldes = fm.get('ldes_dispatch_pct', 0)
                    mk = mix_key(rm, proc, batt, ldes)
                    if mk not in unique_mixes:
                        unique_mixes[mk] = (rm, proc, batt, ldes)

        n_unique = len(unique_mixes)
        total_mixes += n_unique
        print(f"  {n_unique} unique mixes from feasible_mixes")

        # Simulate and compress each unique mix
        iso_profiles = {}
        for i, (mk, (mix_dict, proc, batt, ldes)) in enumerate(unique_mixes.items()):
            result = simulate_hourly_dispatch(
                demand_norm, profiles, mix_dict, proc, batt, ldes
            )
            compressed = compress_to_24h(result)
            iso_profiles[mk] = round_arrays(compressed)

            if (i + 1) % 500 == 0 or i == n_unique - 1:
                print(f"    Computed {i+1}/{n_unique} profiles")
            total_computed += 1

        output[iso] = {'profiles': iso_profiles}

    # Write output
    out_path = 'dashboard/compressed_day_profiles.json'
    print(f"\nWriting {out_path}...")
    with open(out_path, 'w') as f:
        json.dump(output, f, separators=(',', ':'))

    file_size = os.path.getsize(out_path) / 1024 / 1024
    elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print(f"  Done! {total_computed} profiles computed for {total_mixes} unique mixes")
    print(f"  Output: {out_path} ({file_size:.1f} MB)")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
