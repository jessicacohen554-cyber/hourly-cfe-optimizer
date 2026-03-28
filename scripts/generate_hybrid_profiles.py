#!/usr/bin/env python3
"""
Generate 8760 Hybrid Profiles for Solar+Battery and Wind+Battery Resources
==========================================================================

Produces pre-computed normalized 8760 profiles for 4 hybrid resource types
per ISO, ready to feed into Step 1 (PFS generation) and Step 2 (cost optimization).

Hybrid types:
  - solar_batt4: Solar + 4hr battery (DC:AC clipping recovery)
  - solar_batt8: Solar + 8hr battery (extended clipping + deeper shift)
  - wind_batt4:  Wind + 4hr battery (short temporal shifting)
  - wind_batt8:  Wind + 8hr battery (deep overnight-to-peak shifting)

Dispatch rules:
  - Solar hybrids: battery charges ONLY from clipped energy (DC > AC interconnection)
  - Wind hybrids: battery charges ONLY from off-peak surplus (no grid charging)
  - Both: discharge during top-N net-peak hours per day (N = battery duration)
  - Combined output capped at interconnection capacity
  - 85% round-trip efficiency (sqrt split: charge × discharge)

Output: data/hybrid_profiles/{ISO}_hybrid_profiles.npz
        Each file contains 4 arrays of shape (8760,), normalized to sum ~1.0

Usage:
  python scripts/generate_hybrid_profiles.py [--iso CAISO] [--plot]
"""

import argparse
import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from eia_data_io import load_generation_profiles, load_demand_profiles
from pipeline_config import ISOS, H, REGIONAL_DEMAND_TWH

# ============================================================================
# CONSTANTS
# ============================================================================

DATA_YEAR = '2025'

# DC:AC ratios — validated by scripts/validate_dcac_ratios.py
DC_AC_RATIOS = {
    'CAISO': 1.35,
    'ERCOT': 1.50,
    'PJM':   1.50,
    'NYISO': 1.50,
    'NEISO': 1.50,
    'MISO':  1.50,
    'SPP':   1.50,
}

# Battery parameters
BATTERY_RTE = 0.85
CHARGE_EFF = np.sqrt(BATTERY_RTE)   # ~0.922
DISCHARGE_EFF = np.sqrt(BATTERY_RTE) # ~0.922

# Wind battery MW ratio (fraction of wind nameplate)
# Conservative initial values — to be refined by wind profile analysis
WIND_BATT_MW_RATIO = {
    'CAISO': 0.30,
    'ERCOT': 0.40,
    'PJM':   0.30,
    'NYISO': 0.25,
    'NEISO': 0.25,
    'MISO':  0.35,
    'SPP':   0.40,
}

OUTPUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'data', 'hybrid_profiles')


# ============================================================================
# SOLAR HYBRID DISPATCH
# ============================================================================

def compute_solar_hybrid(solar_profile, demand_profile, dc_ac_ratio,
                         battery_hours, iso):
    """
    Compute 8760 hybrid output for solar + co-located battery.

    Args:
        solar_profile: 8760 array, normalized energy-share (sum ≈ 1.0)
        demand_profile: 8760 array, normalized demand (sum ≈ 1.0)
        dc_ac_ratio: e.g. 1.35 for CAISO
        battery_hours: 4 or 8
        iso: ISO name (for logging)

    Returns:
        8760 array of hybrid output, normalized to sum ≈ 1.0
    """
    solar = np.array(solar_profile, dtype=np.float64)
    demand = np.array(demand_profile, dtype=np.float64)

    # Normalize solar to capacity-factor space (peak = 1.0 = AC interconnection)
    peak = np.max(solar)
    if peak <= 0:
        return solar  # No solar generation — return zeros
    solar_cf = solar / peak

    # DC generation with overbuild
    dc_gen = solar_cf * dc_ac_ratio
    ac_cap = 1.0  # Interconnection limit (normalized)

    # Battery specs (normalized to AC interconnection = 1.0)
    batt_capacity = float(battery_hours) * ac_cap  # MWh
    power_rating = ac_cap  # MW (can charge/discharge at full interconnection rate)

    # Compute net load for discharge targeting
    # Use total renewable (solar + wind proxy) minus demand for net-peak
    # Simplification: use demand alone since we don't have other renewables here
    # Net peak = hours with highest demand relative to solar output
    net_load = demand - solar_cf  # Higher = more need, good discharge target

    # Output array
    hybrid_output = np.minimum(dc_gen, ac_cap).copy()  # Start with clipped solar
    clipped = np.maximum(dc_gen - ac_cap, 0.0)

    n_days = H // 24
    soc = 0.0

    for day in range(n_days):
        h_start = day * 24
        h_end = min(h_start + 24, H)

        # --- Charge pass: absorb clipped energy ---
        daily_charged = 0.0
        for h in range(h_start, h_end):
            if clipped[h] > 0 and soc < batt_capacity:
                available = clipped[h]
                headroom = (batt_capacity - soc) / CHARGE_EFF
                charge = min(available, headroom, power_rating)
                soc += charge * CHARGE_EFF
                soc = min(soc, batt_capacity)
                daily_charged += charge

        # --- Discharge pass: top N net-peak hours ---
        if soc > 0 and daily_charged > 0:
            # Rank hours by net load (descending) — discharge into highest net-load hours
            day_hours = np.arange(h_start, h_end)
            day_net_load = net_load[h_start:h_end]
            # Only discharge during hours when solar is low (evening/night)
            peak_order = np.argsort(-day_net_load)
            discharge_count = 0

            for rank_idx in peak_order:
                if discharge_count >= battery_hours:
                    break
                h = h_start + rank_idx
                if soc <= 0:
                    break

                # How much room under the interconnection cap?
                headroom = max(ac_cap - hybrid_output[h], 0.0)
                if headroom <= 0:
                    continue

                available_energy = soc  # Raw SOC
                max_discharge = min(power_rating, available_energy, headroom / DISCHARGE_EFF)
                if max_discharge <= 0:
                    continue

                energy_out = max_discharge * DISCHARGE_EFF
                hybrid_output[h] += energy_out
                hybrid_output[h] = min(hybrid_output[h], ac_cap)
                soc -= max_discharge
                discharge_count += 1

    # Re-normalize to sum ≈ 1.0 (energy-share space, matching pipeline convention)
    total = np.sum(hybrid_output)
    if total > 0:
        hybrid_output = hybrid_output / total

    return hybrid_output


# ============================================================================
# WIND HYBRID DISPATCH
# ============================================================================

def compute_wind_hybrid(wind_profile, demand_profile, batt_mw_ratio,
                        battery_hours, iso):
    """
    Compute 8760 hybrid output for wind + co-located battery.

    Args:
        wind_profile: 8760 array, normalized energy-share (sum ≈ 1.0)
        demand_profile: 8760 array, normalized demand (sum ≈ 1.0)
        batt_mw_ratio: battery MW as fraction of wind nameplate (0.25-0.40)
        battery_hours: 4 or 8
        iso: ISO name (for logging)

    Returns:
        8760 array of hybrid output, normalized to sum ≈ 1.0
    """
    wind = np.array(wind_profile, dtype=np.float64)
    demand = np.array(demand_profile, dtype=np.float64)

    # Normalize wind to capacity-factor space (peak = 1.0 = wind nameplate)
    peak = np.max(wind)
    if peak <= 0:
        return wind
    wind_cf = wind / peak

    # Interconnection = wind nameplate (no overbuild for wind)
    interconnect_cap = 1.0

    # Battery specs (sized as fraction of wind nameplate)
    power_rating = batt_mw_ratio  # MW (e.g., 0.30 = 30% of wind nameplate)
    batt_capacity = power_rating * float(battery_hours)  # MWh

    # Net load for peak/off-peak classification
    # Normalize demand to same scale as wind_cf for meaningful comparison
    demand_scaled = demand / np.max(demand) if np.max(demand) > 0 else demand
    net_load = demand_scaled - wind_cf

    # Output array — wind goes directly to grid (no clipping)
    hybrid_output = wind_cf.copy()

    n_days = H // 24
    soc = 0.0

    for day in range(n_days):
        h_start = day * 24
        h_end = min(h_start + 24, H)

        # --- Identify off-peak hours (low net load = wind surplus) ---
        day_net_load = net_load[h_start:h_end]

        # Charge during hours when wind exceeds demand signal (negative net load)
        # Sort by net load ascending — charge during lowest net-load (highest surplus) hours
        hour_order = np.argsort(day_net_load)

        # --- Charge pass: absorb off-peak wind surplus ---
        daily_charged = 0.0
        for rank_idx in hour_order:
            h = h_start + rank_idx
            if soc >= batt_capacity:
                break

            # Only charge when wind is generating surplus (net_load < 0 or wind_cf > threshold)
            surplus = max(wind_cf[h] - demand_scaled[h], 0.0)
            if surplus <= 0:
                continue  # No surplus this hour

            available = min(surplus, power_rating)
            headroom = (batt_capacity - soc) / CHARGE_EFF
            charge = min(available, headroom)
            if charge <= 0:
                continue

            # Battery absorbs surplus — reduce grid output accordingly
            hybrid_output[h] -= charge  # Less wind goes to grid (diverted to battery)
            hybrid_output[h] = max(hybrid_output[h], 0.0)
            soc += charge * CHARGE_EFF
            soc = min(soc, batt_capacity)
            daily_charged += charge

        # --- Discharge pass: top N net-peak hours ---
        if soc > 0 and daily_charged > 0:
            peak_order = np.argsort(-day_net_load)
            discharge_count = 0

            for rank_idx in peak_order:
                if discharge_count >= battery_hours:
                    break
                h = h_start + rank_idx
                if soc <= 0:
                    break

                headroom = max(interconnect_cap - hybrid_output[h], 0.0)
                if headroom <= 0:
                    continue

                available_energy = soc
                max_discharge = min(power_rating, available_energy, headroom / DISCHARGE_EFF)
                if max_discharge <= 0:
                    continue

                energy_out = max_discharge * DISCHARGE_EFF
                hybrid_output[h] += energy_out
                hybrid_output[h] = min(hybrid_output[h], interconnect_cap)
                soc -= max_discharge
                discharge_count += 1

    # Re-normalize to sum ≈ 1.0
    total = np.sum(hybrid_output)
    if total > 0:
        hybrid_output = hybrid_output / total

    return hybrid_output


# ============================================================================
# DST-AWARE SOLAR CORRECTION (matches dispatch_utils.py)
# ============================================================================

def apply_dst_correction(solar_raw, iso):
    """Apply DST-aware nighttime zeroing, matching dispatch_utils.py logic."""
    solar = np.array(solar_raw[:H], dtype=np.float64)
    STD_UTC_OFFSETS = {
        'CAISO': 8, 'ERCOT': 6, 'PJM': 5,
        'NYISO': 5, 'NEISO': 5, 'MISO': 6, 'SPP': 6
    }
    DST_START_DAY, DST_END_DAY = 69, 307
    local_start, local_end = 6, 19
    std_off = STD_UTC_OFFSETS.get(iso, 5)

    hour_indices = np.arange(H)
    days = hour_indices // 24
    hours = hour_indices % 24
    is_dst = (days >= DST_START_DAY) & (days < DST_END_DAY)
    utc_off = std_off - is_dst.astype(np.int64)
    utc_start = (local_start + utc_off) % 24
    utc_end = (local_end + utc_off) % 24
    no_wrap = utc_start <= utc_end
    is_daylight = np.where(no_wrap,
                           (hours >= utc_start) & (hours <= utc_end),
                           (hours >= utc_start) | (hours <= utc_end))
    solar[~is_daylight] = 0.0
    return solar


# ============================================================================
# MAIN
# ============================================================================

def generate_profiles_for_iso(iso, gen_profiles, demand_data):
    """Generate all 4 hybrid profiles for one ISO."""
    # --- Load solar profile ---
    iso_data = gen_profiles.get(iso, {})
    year_data = iso_data.get(DATA_YEAR, iso_data.get('2024', {}))
    if not isinstance(year_data, dict):
        year_data = {}

    if iso == 'NYISO':
        solar_raw = year_data.get('solar_proxy', year_data.get('solar', [0.0] * H))
    else:
        solar_raw = year_data.get('solar', [0.0] * H)

    solar_raw = np.array(solar_raw[:H], dtype=np.float64)
    solar_raw = np.nan_to_num(solar_raw, nan=0.0)
    solar_raw[solar_raw < 0] = 0.0
    solar_profile = apply_dst_correction(solar_raw, iso)

    # Normalize solar to energy-share (sum ≈ 1.0) for consistency
    s_total = np.sum(solar_profile)
    if s_total > 0:
        solar_profile = solar_profile / s_total

    # --- Load wind profile ---
    wind_raw = year_data.get('wind', [0.0] * H)
    wind_profile = np.array(wind_raw[:H], dtype=np.float64)
    wind_profile = np.nan_to_num(wind_profile, nan=0.0)
    wind_profile[wind_profile < 0] = 0.0
    w_total = np.sum(wind_profile)
    if w_total > 0:
        wind_profile = wind_profile / w_total

    # --- Load demand profile ---
    iso_demand = demand_data.get(iso, {})
    year_demand = iso_demand.get(DATA_YEAR, iso_demand.get('2024', {}))
    if isinstance(year_demand, dict):
        demand_norm = np.array(year_demand.get('normalized', [0.0] * H)[:H], dtype=np.float64)
    else:
        demand_norm = np.zeros(H, dtype=np.float64)
    # Normalize demand to sum ≈ 1.0
    d_total = np.sum(demand_norm)
    if d_total > 0:
        demand_norm = demand_norm / d_total

    dc_ac = DC_AC_RATIOS[iso]
    batt_mw = WIND_BATT_MW_RATIO[iso]

    # --- Generate 4 hybrid profiles ---
    results = {}

    print(f"  {iso}: solar_batt4 (DC:AC={dc_ac:.2f}, 4hr)...")
    results['solar_batt4'] = compute_solar_hybrid(
        solar_profile, demand_norm, dc_ac, battery_hours=4, iso=iso)

    print(f"  {iso}: solar_batt8 (DC:AC={dc_ac:.2f}, 8hr)...")
    results['solar_batt8'] = compute_solar_hybrid(
        solar_profile, demand_norm, dc_ac, battery_hours=8, iso=iso)

    print(f"  {iso}: wind_batt4 (MW ratio={batt_mw:.2f}, 4hr)...")
    results['wind_batt4'] = compute_wind_hybrid(
        wind_profile, demand_norm, batt_mw, battery_hours=4, iso=iso)

    print(f"  {iso}: wind_batt8 (MW ratio={batt_mw:.2f}, 8hr)...")
    results['wind_batt8'] = compute_wind_hybrid(
        wind_profile, demand_norm, batt_mw, battery_hours=8, iso=iso)

    # --- Diagnostics ---
    for name, profile in results.items():
        cf_equiv = np.max(profile) / np.mean(profile) if np.mean(profile) > 0 else 0
        peak_hour = np.argmax(profile)
        print(f"    {name}: sum={np.sum(profile):.4f}, peak_hour={peak_hour}, "
              f"peak/mean={cf_equiv:.1f}x, nonzero_hours={np.sum(profile > 1e-8)}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Generate hybrid 8760 profiles')
    parser.add_argument('--iso', type=str, help='Single ISO (default: all)')
    args = parser.parse_args()

    isos = [args.iso] if args.iso else ISOS

    print("Loading EIA-930 profiles...")
    gen_profiles = load_generation_profiles()
    demand_data = load_demand_profiles()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nGenerating hybrid profiles for {len(isos)} ISOs...")
    print(f"Output: {OUTPUT_DIR}/\n")

    all_stats = {}

    for iso in isos:
        print(f"\n{'='*60}")
        print(f"  {iso}")
        print(f"{'='*60}")

        profiles = generate_profiles_for_iso(iso, gen_profiles, demand_data)

        # Save as NPZ
        out_path = os.path.join(OUTPUT_DIR, f'{iso}_hybrid_profiles.npz')
        np.savez_compressed(out_path, **profiles)
        print(f"  Saved: {out_path}")

        # Collect stats for summary
        all_stats[iso] = {}
        for name, profile in profiles.items():
            parent = 'solar' if 'solar' in name else 'wind'
            # Compare to parent profile energy distribution
            all_stats[iso][name] = {
                'nonzero_hours': int(np.sum(profile > 1e-8)),
                'peak_hour': int(np.argmax(profile)),
                'peak_value': float(np.max(profile)),
                'mean_value': float(np.mean(profile)),
            }

    # --- Summary table ---
    print(f"\n\n{'='*100}")
    print(f"  HYBRID PROFILE SUMMARY")
    print(f"{'='*100}")
    print(f"\n  {'ISO':<8} {'Type':<14} {'Active Hrs':>10} {'Peak Hour':>10} {'Peak/Mean':>10}")
    print(f"  {'-'*8} {'-'*14} {'-'*10} {'-'*10} {'-'*10}")

    for iso in isos:
        if iso not in all_stats:
            continue
        for name in ['solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8']:
            s = all_stats[iso].get(name, {})
            peak_mean = s['peak_value'] / s['mean_value'] if s.get('mean_value', 0) > 0 else 0
            print(f"  {iso:<8} {name:<14} {s.get('nonzero_hours', 0):>10} "
                  f"{s.get('peak_hour', 0):>10} {peak_mean:>10.1f}x")
        print()

    print(f"\nDone. Profiles saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
