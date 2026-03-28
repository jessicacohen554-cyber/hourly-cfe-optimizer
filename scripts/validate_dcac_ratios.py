#!/usr/bin/env python3
"""
Validate DC:AC ratios for solar+battery hybrids against actual EIA-930 solar profiles.

For each ISO and DC:AC ratio, computes:
  - Clipping hours and clipped energy
  - 4hr battery (85% RTE) recovery of clipped energy
  - Effective hybrid capacity factor vs. solar-only CF
  - Recommended optimal DC:AC ratio per ISO
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from eia_data_io import load_generation_profiles

ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']
DC_AC_RATIOS = [1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.5]
BATTERY_HOURS = 4
BATTERY_RTE = 0.85
CHARGE_EFF = np.sqrt(BATTERY_RTE)  # ~0.922
DISCHARGE_EFF = np.sqrt(BATTERY_RTE)  # ~0.922
# Evening peak proxy hours (16:00-19:59, 0-indexed)
PEAK_HOURS = [16, 17, 18, 19]
HOURS_PER_YEAR = 8760


def analyze_dcac(solar_profile: np.ndarray, dc_ac_ratio: float):
    """
    Analyze a DC:AC ratio against a normalized solar profile.

    solar_profile: 8760 array, normalized [0,1] representing DC capacity factor.
    dc_ac_ratio: e.g. 1.3 means 1.3 MW_DC per 1.0 MW_AC inverter.

    The AC interconnection clips at 1.0 MW_AC. With DC:AC=1.3 and profile
    normalized to DC nameplate=1.0, the AC clip threshold is 1/1.3.
    But we're oversizing DC, so effective generation = profile * dc_ac_ratio,
    clipped at 1.0 AC.
    """
    n = len(solar_profile)
    assert n == HOURS_PER_YEAR, f"Expected {HOURS_PER_YEAR} hours, got {n}"

    # Effective DC generation (scaled by oversize ratio), clipped at AC rating (1.0)
    dc_gen = solar_profile * dc_ac_ratio
    ac_output_no_battery = np.minimum(dc_gen, 1.0)
    clipped_energy_per_hour = np.maximum(dc_gen - 1.0, 0.0)

    clipping_hours = int(np.sum(clipped_energy_per_hour > 0))
    total_clipped = float(np.sum(clipped_energy_per_hour))

    # --- Battery simulation ---
    # Battery capacity in MWh (at AC rating = 1.0 MW_AC × 4 hours)
    batt_cap = BATTERY_HOURS * 1.0  # 4 MWh per MW_AC
    soc = 0.0  # state of charge in MWh
    total_recovered = 0.0
    total_charged = 0.0

    # Vectorize day-by-day for speed
    n_days = HOURS_PER_YEAR // 24
    ac_output_hybrid = ac_output_no_battery.copy()

    for day in range(n_days):
        h_start = day * 24
        h_end = h_start + 24

        # Charge phase: absorb clipped energy during daylight
        for h in range(h_start, h_end):
            if clipped_energy_per_hour[h] > 0 and soc < batt_cap:
                available = clipped_energy_per_hour[h]
                can_store = batt_cap - soc
                charge = min(available, can_store)
                soc += charge * CHARGE_EFF
                soc = min(soc, batt_cap)
                total_charged += charge

        # Discharge phase: evening peak hours
        for h_offset in PEAK_HOURS:
            h = h_start + h_offset
            if h < HOURS_PER_YEAR and soc > 0:
                # Discharge up to 1 MW_AC for 1 hour (but only fill gap to 1.0)
                headroom = max(1.0 - ac_output_hybrid[h], 0.0)
                discharge = min(soc, headroom, 1.0)
                energy_out = discharge * DISCHARGE_EFF
                ac_output_hybrid[h] += energy_out
                ac_output_hybrid[h] = min(ac_output_hybrid[h], 1.0)
                total_recovered += energy_out
                soc -= discharge

    # Base solar CF (no oversize, no battery)
    base_cf = float(np.sum(solar_profile)) / HOURS_PER_YEAR
    # Solar with oversize but no battery (clipped at AC)
    oversize_cf = float(np.sum(ac_output_no_battery)) / HOURS_PER_YEAR
    # Hybrid CF (oversize + battery recovering clips)
    hybrid_cf = float(np.sum(ac_output_hybrid)) / HOURS_PER_YEAR

    return {
        'dc_ac': dc_ac_ratio,
        'base_cf': base_cf,
        'oversize_cf': oversize_cf,
        'hybrid_cf': hybrid_cf,
        'clipping_hours': clipping_hours,
        'total_clipped_mwh': total_clipped,
        'total_charged_mwh': total_charged,
        'total_recovered_mwh': total_recovered,
        'cf_gain_oversize': oversize_cf - base_cf,
        'cf_gain_hybrid': hybrid_cf - base_cf,
    }


def main():
    print("Loading EIA-930 generation profiles...")
    profiles = load_generation_profiles()

    # Use latest available year
    target_year = '2024'

    print(f"\n{'='*120}")
    print(f"  DC:AC RATIO VALIDATION — Solar+Battery Hybrids vs. EIA-930 Profiles ({target_year})")
    print(f"  Battery: {BATTERY_HOURS}hr Li-ion, {BATTERY_RTE*100:.0f}% RTE, discharge during hours {PEAK_HOURS}")
    print(f"{'='*120}\n")

    all_results = {}

    for iso in ISOS:
        if iso not in profiles:
            print(f"  {iso}: no profile data, skipping")
            continue

        year_data = profiles[iso].get(target_year)
        if year_data is None:
            # Fall back to latest available year
            available = sorted(profiles[iso].keys())
            if not available:
                print(f"  {iso}: no year data, skipping")
                continue
            target = available[-1]
            year_data = profiles[iso][target]
            print(f"  {iso}: using {target} (2024 not available)")

        solar = year_data.get('solar')
        if solar is None:
            print(f"  {iso}: no solar profile, skipping")
            continue

        solar_arr = np.array(solar[:HOURS_PER_YEAR], dtype=np.float64)
        # Replace NaN with 0
        solar_arr = np.nan_to_num(solar_arr, nan=0.0)

        base_cf = float(np.sum(solar_arr)) / HOURS_PER_YEAR

        print(f"  {iso}  (base solar CF = {base_cf:.4f} = {base_cf*100:.1f}%)")
        print(f"  {'DC:AC':>6}  {'Oversize CF':>11}  {'Hybrid CF':>10}  {'CF Gain':>8}  "
              f"{'Clip Hrs':>8}  {'Clipped MWh':>11}  {'Recovered MWh':>13}  {'Recovery %':>10}")
        print(f"  {'-'*6}  {'-'*11}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*11}  {'-'*13}  {'-'*10}")

        iso_results = []
        for ratio in DC_AC_RATIOS:
            r = analyze_dcac(solar_arr, ratio)
            iso_results.append(r)
            recovery_pct = (r['total_recovered_mwh'] / r['total_clipped_mwh'] * 100) if r['total_clipped_mwh'] > 0 else 0
            print(f"  {ratio:>6.2f}  {r['oversize_cf']*100:>10.2f}%  {r['hybrid_cf']*100:>9.2f}%  "
                  f"{r['cf_gain_hybrid']*100:>+7.2f}%  {r['clipping_hours']:>8d}  "
                  f"{r['total_clipped_mwh']:>11.1f}  {r['total_recovered_mwh']:>13.1f}  {recovery_pct:>9.1f}%")

        all_results[iso] = iso_results
        print()

    # --- Marginal analysis & recommendations ---
    print(f"\n{'='*120}")
    print(f"  MARGINAL CF GAIN PER 0.05 DC:AC INCREMENT (hybrid)")
    print(f"{'='*120}\n")

    header = f"  {'DC:AC':>6}"
    for iso in ISOS:
        header += f"  {iso:>8}"
    print(header)
    print(f"  {'-'*6}" + f"  {'-'*8}" * len(ISOS))

    for i in range(1, len(DC_AC_RATIOS)):
        row = f"  {DC_AC_RATIOS[i]:>6.2f}"
        for iso in ISOS:
            if iso in all_results:
                delta = all_results[iso][i]['hybrid_cf'] - all_results[iso][i-1]['hybrid_cf']
                row += f"  {delta*100:>+7.3f}%"
            else:
                row += f"  {'N/A':>8}"
        print(row)

    print(f"\n{'='*120}")
    print(f"  OPTIMAL DC:AC RATIO RECOMMENDATION PER ISO")
    print(f"  (Selected where marginal hybrid CF gain drops below 50% of initial marginal gain)")
    print(f"{'='*120}\n")

    for iso in ISOS:
        if iso not in all_results:
            continue
        results = all_results[iso]

        # Compute marginal gains
        marginals = []
        for i in range(1, len(results)):
            delta_cf = results[i]['hybrid_cf'] - results[i-1]['hybrid_cf']
            delta_ratio = DC_AC_RATIOS[i] - DC_AC_RATIOS[i-1]
            marginals.append(delta_cf / delta_ratio)  # CF gain per unit DC:AC

        if not marginals:
            continue

        initial_marginal = marginals[0]
        # Find where marginal drops below 50% of initial
        optimal_idx = len(DC_AC_RATIOS) - 1  # default to highest
        for i, m in enumerate(marginals):
            if m < 0.5 * initial_marginal:
                optimal_idx = i  # the ratio BEFORE this drop
                break

        optimal_ratio = DC_AC_RATIOS[optimal_idx]
        r = results[optimal_idx]
        print(f"  {iso:>6}: DC:AC = {optimal_ratio:.2f}  |  "
              f"Hybrid CF = {r['hybrid_cf']*100:.2f}%  |  "
              f"CF gain = {r['cf_gain_hybrid']*100:>+.2f}%  |  "
              f"Clipped = {r['total_clipped_mwh']:.0f} MWh  |  "
              f"Recovered = {r['total_recovered_mwh']:.0f} MWh")

    print()


if __name__ == '__main__':
    main()
