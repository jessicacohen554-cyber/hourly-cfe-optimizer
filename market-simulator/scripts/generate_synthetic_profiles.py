#!/usr/bin/env python3
"""
Generate synthetic EIA-style demand and generation profiles for testing.
Uses realistic shapes for ERCOT (and other ISOs) based on published statistics.

This is a TESTING FALLBACK — production runs should use actual EIA 930 data
from the Step 0 data acquisition pipeline.
"""

import json
import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from pipeline_config import ISOS, REGIONAL_DEMAND_TWH, H

# Realistic annual demand (TWh) and peak/min MW ratios per ISO
ISO_PROFILES = {
    'CAISO': {'peak_ratio': 1.35, 'min_ratio': 0.55, 'summer_peak': True, 'solar_cf': 0.27, 'wind_cf': 0.22},
    'ERCOT': {'peak_ratio': 1.50, 'min_ratio': 0.45, 'summer_peak': True, 'solar_cf': 0.24, 'wind_cf': 0.35},
    'PJM':   {'peak_ratio': 1.30, 'min_ratio': 0.55, 'summer_peak': True, 'solar_cf': 0.18, 'wind_cf': 0.28},
    'NYISO': {'peak_ratio': 1.35, 'min_ratio': 0.50, 'summer_peak': True, 'solar_cf': 0.16, 'wind_cf': 0.25},
    'NEISO': {'peak_ratio': 1.30, 'min_ratio': 0.55, 'summer_peak': False, 'solar_cf': 0.17, 'wind_cf': 0.30},
    'MISO':  {'peak_ratio': 1.40, 'min_ratio': 0.50, 'summer_peak': True, 'solar_cf': 0.20, 'wind_cf': 0.35},
    'SPP':   {'peak_ratio': 1.35, 'min_ratio': 0.50, 'summer_peak': True, 'solar_cf': 0.22, 'wind_cf': 0.42},
}

def generate_demand_profile(iso, year='2024'):
    """Generate a realistic 8760-hour demand profile."""
    params = ISO_PROFILES.get(iso, ISO_PROFILES['PJM'])
    total_twh = REGIONAL_DEMAND_TWH[iso]
    avg_mw = total_twh * 1e6 / H
    peak_mw = avg_mw * params['peak_ratio']
    min_mw = avg_mw * params['min_ratio']

    hours = np.arange(H)
    day_of_year = hours // 24
    hour_of_day = hours % 24

    # Seasonal component (peaks in summer or winter)
    if params['summer_peak']:
        # Summer peak around day 200 (July)
        seasonal = 0.15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    else:
        # Winter peak around day 15 (January) — NEISO heating load
        seasonal = 0.10 * np.cos(2 * np.pi * day_of_year / 365)

    # Daily pattern — peaks 14:00-18:00, min 03:00-05:00
    daily = (0.10 * np.sin(np.pi * (hour_of_day - 6) / 12) *
             np.where((hour_of_day >= 6) & (hour_of_day <= 22), 1.0, 0.3))

    # Weekend reduction (~5% lower)
    day_of_week = day_of_year % 7
    weekend = np.where((day_of_week == 5) | (day_of_week == 6), -0.05, 0.0)

    profile = avg_mw * (1.0 + seasonal + daily + weekend)

    # Add slight noise
    rng = np.random.default_rng(42 + hash(iso) % 1000)
    noise = rng.normal(0, avg_mw * 0.02, H)
    profile = np.clip(profile + noise, min_mw, peak_mw)

    # Normalize to sum to 1.0 (for dispatch_utils compatibility)
    total_mwh = float(np.sum(profile))
    normalized = profile / total_mwh

    return {
        'raw_mw': profile.tolist(),
        'normalized': normalized.tolist(),
        'total_annual_mwh': total_mwh,
        'peak_mw': float(np.max(profile)),
        'min_mw': float(np.min(profile)),
        'avg_mw': float(np.mean(profile)),
    }


def generate_gen_profile(iso, resource, year='2024'):
    """Generate a realistic 8760-hour generation shape profile."""
    params = ISO_PROFILES.get(iso, ISO_PROFILES['PJM'])
    hours = np.arange(H)
    day_of_year = hours // 24
    hour_of_day = hours % 24

    if resource == 'solar':
        # Solar: daytime bell curve, seasonal variation
        solar_hour = np.clip((hour_of_day - 6) / 12.0, 0, 1)
        daily = np.sin(np.pi * solar_hour) * np.where((hour_of_day >= 6) & (hour_of_day <= 18), 1.0, 0.0)
        seasonal = 0.7 + 0.3 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        profile = daily * seasonal
        # Scale to match CF
        if np.mean(profile) > 0:
            profile *= params['solar_cf'] / np.mean(profile)

    elif resource in ('wind', 'offshore_wind'):
        # Wind: more at night, stronger in spring/fall
        rng = np.random.default_rng(42 + hash(iso + resource) % 1000)
        base = rng.beta(2, 3, H)  # Right-skewed distribution
        # Seasonal: more wind in spring/winter
        seasonal = 1.0 + 0.3 * np.cos(2 * np.pi * (day_of_year - 60) / 365)
        # Diurnal: slight nighttime bias
        diurnal = 1.0 + 0.1 * np.cos(2 * np.pi * hour_of_day / 24)
        profile = base * seasonal * diurnal
        cf = params['wind_cf'] if resource == 'wind' else params['wind_cf'] * 0.9
        if np.mean(profile) > 0:
            profile *= cf / np.mean(profile)

    elif resource in ('clean_firm', 'ccs_ccgt'):
        # Flat baseload with seasonal derate
        seasonal_derate = 1.0 - 0.08 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
        profile = 0.93 * seasonal_derate  # 93% CF for nuclear

    elif resource == 'hydro':
        # Hydro: seasonal (spring snowmelt peak)
        seasonal = 0.8 + 0.4 * np.exp(-((day_of_year - 120) ** 2) / 2000)
        # Daily dispatch flexibility
        daily = 0.8 + 0.2 * np.sin(np.pi * (hour_of_day - 6) / 12) * np.where(
            (hour_of_day >= 6) & (hour_of_day <= 22), 1.0, 0.5)
        profile = seasonal * daily * 0.4  # ~40% annual CF

    elif resource == 'geothermal':
        # Flat year-round
        profile = np.ones(H) * 0.90

    else:
        profile = np.ones(H) * 0.5

    return profile.tolist()


def generate_fossil_mix(iso, year='2024'):
    """Generate synthetic fossil fuel mix (coal/gas/oil shares per hour)."""
    from lmp_engine import FOSSIL_CAPACITY_SHARES
    shares = FOSSIL_CAPACITY_SHARES.get(iso, {'coal_steam': 0.2, 'gas_ccgt': 0.5, 'gas_ct': 0.25, 'oil_ct': 0.05})

    coal_share = shares.get('coal_steam', 0) + shares.get('coal', 0)
    gas_share = shares.get('gas_ccgt', 0) + shares.get('gas_ct', 0)
    oil_share = shares.get('oil_ct', 0) + shares.get('oil', 0)

    return {
        'hours': list(range(1, H + 1)),
        'coal_share': [coal_share] * H,
        'gas_share': [gas_share] * H,
        'oil_share': [oil_share] * H,
    }


def generate_all_profiles(output_dir=None, isos=None):
    """Generate all synthetic profiles and save as JSON files."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(SCRIPT_DIR), 'data', 'profiles')
    os.makedirs(output_dir, exist_ok=True)

    if isos is None:
        isos = list(ISOS)

    year = '2024'

    # Demand profiles
    demand_data = {}
    for iso in isos:
        demand_data[iso] = {year: generate_demand_profile(iso, year)}
    with open(os.path.join(output_dir, 'eia_demand_profiles.json'), 'w') as f:
        json.dump(demand_data, f)
    print(f"  Demand profiles: {len(isos)} ISOs")

    # Generation profiles
    gen_data = {}
    resources = ['solar', 'wind', 'offshore_wind', 'clean_firm', 'ccs_ccgt', 'hydro', 'geothermal']
    for iso in isos:
        gen_data[iso] = {}
        for res in resources:
            gen_data[iso][res] = {year: generate_gen_profile(iso, res, year)}
    with open(os.path.join(output_dir, 'eia_gen_profiles.json'), 'w') as f:
        json.dump(gen_data, f)
    print(f"  Generation profiles: {len(isos)} ISOs × {len(resources)} resources")

    # Fossil mix
    fossil_data = {}
    for iso in isos:
        fossil_data[iso] = {year: generate_fossil_mix(iso, year)}
    with open(os.path.join(output_dir, 'eia_fossil_mix.json'), 'w') as f:
        json.dump(fossil_data, f)
    print(f"  Fossil mix profiles: {len(isos)} ISOs")

    # Emission rates (eGRID-derived)
    emission_rates = {
        'CAISO': {'co2_rate': 0.35}, 'ERCOT': {'co2_rate': 0.42},
        'PJM': {'co2_rate': 0.55}, 'NYISO': {'co2_rate': 0.28},
        'NEISO': {'co2_rate': 0.30}, 'MISO': {'co2_rate': 0.62},
        'SPP': {'co2_rate': 0.48},
    }
    with open(os.path.join(output_dir, 'egrid_emission_rates.json'), 'w') as f:
        json.dump(emission_rates, f, indent=2)
    print(f"  Emission rates: {len(isos)} ISOs")

    print(f"\nAll profiles saved to {output_dir}")
    return output_dir


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate synthetic EIA profiles for testing')
    parser.add_argument('--isos', nargs='+', default=None)
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()
    generate_all_profiles(args.output_dir, args.isos)
