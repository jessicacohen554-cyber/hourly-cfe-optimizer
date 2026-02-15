#!/usr/bin/env python3
"""
Over-Procurement Optimization Engine — Advanced Sensitivity Model
==================================================================
For each ISO region (CAISO, ERCOT, PJM, NYISO, NEISO), sweep over-procurement
levels and find the optimal mix of 5 resource types (clean_firm, solar, wind,
ccs_ccgt, hydro) + 2 storage types (battery 4hr Li-ion + LDES 100hr iron-air)
to maximize hourly matching at minimum cost.

Three-phase refinement: coarse -> medium -> fine (adapted for 5D)

Resource types:
  - Clean Firm: seasonally-derated baseload — nuclear/geothermal
  - Solar: EIA 2021-2025 averaged hourly profile (DST-aware nighttime zeroing)
  - Wind: EIA 2021-2025 averaged hourly profile
  - CCS-CCGT: flat baseload (1/8760 per hour) — dispatchable, 95% capture
  - Hydro: EIA 2021-2025 averaged hourly profile (capped by region, existing only)

Storage:
  - Battery: 4hr Li-ion, 85% RTE, daily-cycle greedy dispatch
  - LDES: 100hr iron-air, 50% RTE, 7-day rolling window multi-day dispatch
"""

import json
import os
import sys
import time
import numpy as np
from multiprocessing import Pool

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
DATA_YEAR = '2025'        # Actuals year for total MWh, grid mix, hydro caps
PROFILE_YEARS = ['2021', '2022', '2023', '2024', '2025']  # Years to average for shapes
H = 8760
LEAP_FEB29_START = 1416   # Hour index where Feb 29 starts in a leap year (744+672)

# ══════════════════════════════════════════════════════════════════════════════
# STORAGE CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

BATTERY_EFFICIENCY = 0.85
BATTERY_DURATION_HOURS = 4

LDES_EFFICIENCY = 0.50
LDES_DURATION_HOURS = 100
LDES_WINDOW_DAYS = 7

# ══════════════════════════════════════════════════════════════════════════════
# NUCLEAR SEASONAL DERATE (from EIA 2021-2025 analysis)
# ══════════════════════════════════════════════════════════════════════════════

# Nuclear share of clean firm generation by ISO
# CAISO: ~1 GW geothermal + Diablo Canyon (~2.3 GW) → ~70% nuclear
# All others: clean firm is effectively 100% nuclear
NUCLEAR_SHARE_OF_CLEAN_FIRM = {
    'CAISO': 0.70,  # ~2.3 GW nuclear / (~2.3 GW nuclear + ~1 GW geothermal)
    'ERCOT': 1.0,   # South Texas Project only
    'PJM': 1.0,     # Large nuclear fleet (Limerick, Peach Bottom, etc.)
    'NYISO': 1.0,   # Nine Mile Point, Ginna, FitzPatrick
    'NEISO': 1.0,   # Millstone, Seabrook
}

# Seasonal capacity factors relative to peak (from 5-year EIA average)
# Applied to nuclear portion only. Geothermal stays flat.
# Months: 1=Jan, 12=Dec
NUCLEAR_MONTHLY_CF = {  # relative to nameplate (where winter peak = 1.0)
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
}

# ══════════════════════════════════════════════════════════════════════════════
# REGIONAL CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

HYDRO_CAPS = {  # 2025 actual hydro share of demand from EIA data
    'CAISO': 9.5,   # 5yr range: 5.2% (drought 2021) - 11.2% (wet 2023)
    'ERCOT': 0.1,   # Minimal hydro in Texas
    'PJM': 1.8,     # 5yr range: 1.9% - 2.1%
    'NYISO': 15.9,  # 5yr range: 15.9% - 18.3%
    'NEISO': 4.4,   # 5yr range: 4.5% - 7.8%
}

ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO']

ISO_LABELS = {
    'CAISO': 'CAISO (California)',
    'ERCOT': 'ERCOT (Texas)',
    'PJM': 'PJM (Mid-Atlantic)',
    'NYISO': 'NYISO (New York)',
    'NEISO': 'NEISO (New England)',
}

# 5 optimization resource types (sum to 100%)
RESOURCE_TYPES = ['clean_firm', 'solar', 'wind', 'ccs_ccgt', 'hydro']

# 10 target thresholds — key inflection points with 2.5% granularity in steep zone
THRESHOLDS = [75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 100]

# Adaptive procurement bounds by threshold — avoids wasting compute searching
# procurement levels that can't possibly be optimal for a given threshold
PROCUREMENT_BOUNDS = {
    75:   (75, 105),
    80:   (75, 110),
    85:   (80, 110),
    87.5: (87, 130),
    90:   (90, 140),
    92.5: (92, 150),
    95:   (95, 175),
    97.5: (100, 200),
    99:   (100, 200),
    100:  (100, 200),
}

# ══════════════════════════════════════════════════════════════════════════════
# COST TABLES — Medium values used by optimizer; full L/M/H output for dashboard
# ══════════════════════════════════════════════════════════════════════════════

# Wholesale electricity prices ($/MWh) — 2025 averages from FERC/ISO market reports
WHOLESALE_PRICES = {
    'CAISO': 30,
    'ERCOT': 27,
    'PJM':   34,
    'NYISO': 42,
    'NEISO': 41,
}

# Medium LCOE values used by the optimizer for mix optimization
REGIONAL_LCOE = {  # Medium level — blended uprate+new-build for clean_firm
    'CAISO': {'clean_firm': 79, 'solar': 60, 'wind': 73, 'ccs_ccgt': 86, 'hydro': 0, 'battery': 102, 'ldes': 180},
    'ERCOT': {'clean_firm': 79, 'solar': 54, 'wind': 40, 'ccs_ccgt': 71, 'hydro': 0, 'battery': 92, 'ldes': 155},
    'PJM':   {'clean_firm': 68, 'solar': 65, 'wind': 62, 'ccs_ccgt': 79, 'hydro': 0, 'battery': 98, 'ldes': 170},
    'NYISO': {'clean_firm': 86, 'solar': 92, 'wind': 81, 'ccs_ccgt': 99, 'hydro': 0, 'battery': 108, 'ldes': 200},
    'NEISO': {'clean_firm': 92, 'solar': 82, 'wind': 73, 'ccs_ccgt': 96, 'hydro': 0, 'battery': 105, 'ldes': 190},
}

# Transmission adders at Medium level (used by optimizer)
TRANSMISSION_ADDERS = {
    'CAISO': {'wind': 8, 'solar': 3, 'clean_firm': 3, 'ccs_ccgt': 2, 'battery': 1, 'ldes': 2, 'hydro': 0},
    'ERCOT': {'wind': 6, 'solar': 3, 'clean_firm': 2, 'ccs_ccgt': 2, 'battery': 1, 'ldes': 2, 'hydro': 0},
    'PJM':   {'wind': 10, 'solar': 5, 'clean_firm': 3, 'ccs_ccgt': 3, 'battery': 1, 'ldes': 3, 'hydro': 0},
    'NYISO': {'wind': 14, 'solar': 7, 'clean_firm': 5, 'ccs_ccgt': 4, 'battery': 2, 'ldes': 4, 'hydro': 0},
    'NEISO': {'wind': 12, 'solar': 6, 'clean_firm': 4, 'ccs_ccgt': 3, 'battery': 2, 'ldes': 3, 'hydro': 0},
}

# Full L/M/H cost tables for dashboard output
FULL_LCOE_TABLES = {
    'solar': {
        'Low':    {'CAISO': 45, 'ERCOT': 40, 'PJM': 50, 'NYISO': 70, 'NEISO': 62},
        'Medium': {'CAISO': 60, 'ERCOT': 54, 'PJM': 65, 'NYISO': 92, 'NEISO': 82},
        'High':   {'CAISO': 78, 'ERCOT': 70, 'PJM': 85, 'NYISO': 120, 'NEISO': 107},
    },
    'wind': {
        'Low':    {'CAISO': 55, 'ERCOT': 30, 'PJM': 47, 'NYISO': 61, 'NEISO': 55},
        'Medium': {'CAISO': 73, 'ERCOT': 40, 'PJM': 62, 'NYISO': 81, 'NEISO': 73},
        'High':   {'CAISO': 95, 'ERCOT': 52, 'PJM': 81, 'NYISO': 105, 'NEISO': 95},
    },
    'clean_firm': {  # Blended uprate + new-build LCOE
        'Low':    {'CAISO': 58, 'ERCOT': 56, 'PJM': 48, 'NYISO': 64, 'NEISO': 69},
        'Medium': {'CAISO': 79, 'ERCOT': 79, 'PJM': 68, 'NYISO': 86, 'NEISO': 92},
        'High':   {'CAISO': 115, 'ERCOT': 115, 'PJM': 108, 'NYISO': 136, 'NEISO': 143},
    },
    'ccs_ccgt': {
        'Low':    {'CAISO': 58, 'ERCOT': 52, 'PJM': 62, 'NYISO': 78, 'NEISO': 75},
        'Medium': {'CAISO': 86, 'ERCOT': 71, 'PJM': 79, 'NYISO': 99, 'NEISO': 96},
        'High':   {'CAISO': 115, 'ERCOT': 92, 'PJM': 102, 'NYISO': 128, 'NEISO': 122},
    },
    'battery': {
        'Low':    {'CAISO': 77, 'ERCOT': 69, 'PJM': 74, 'NYISO': 81, 'NEISO': 79},
        'Medium': {'CAISO': 102, 'ERCOT': 92, 'PJM': 98, 'NYISO': 108, 'NEISO': 105},
        'High':   {'CAISO': 133, 'ERCOT': 120, 'PJM': 127, 'NYISO': 140, 'NEISO': 137},
    },
    'ldes': {
        'Low':    {'CAISO': 135, 'ERCOT': 116, 'PJM': 128, 'NYISO': 150, 'NEISO': 143},
        'Medium': {'CAISO': 180, 'ERCOT': 155, 'PJM': 170, 'NYISO': 200, 'NEISO': 190},
        'High':   {'CAISO': 234, 'ERCOT': 202, 'PJM': 221, 'NYISO': 260, 'NEISO': 247},
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# PAIRED TOGGLE GROUPS — 5 groups replacing 10 individual toggles
# Dashboard toggles move all member variables in lockstep (all L, all M, or all H)
# ══════════════════════════════════════════════════════════════════════════════

PAIRED_TOGGLE_GROUPS = {
    'renewable_generation': {
        'label': 'Renewable Generation Cost',
        'options': ['Low', 'Medium', 'High'],
        'members': ['solar', 'wind'],
        'description': 'Solar and wind LCOE move together — driven by similar manufacturing scale, supply chain, and installation labor factors',
    },
    'firm_generation': {
        'label': 'Firm Generation Cost',
        'options': ['Low', 'Medium', 'High'],
        'members': ['clean_firm', 'ccs_ccgt'],
        'description': 'Clean firm (nuclear/geothermal) and CCS-CCGT share capital-intensive, long-lead-time cost structures',
    },
    'storage': {
        'label': 'Storage Cost',
        'options': ['Low', 'Medium', 'High'],
        'members': ['battery', 'ldes'],
        'description': 'Battery and LDES costs share manufacturing/materials cost drivers across storage technologies',
    },
    'fossil_fuel': {
        'label': 'Fossil Fuel Price',
        'options': ['Low', 'Medium', 'High'],
        'members': ['natural_gas', 'coal', 'oil'],
        'description': 'Gas, coal, and oil prices are correlated through energy commodity markets and macro conditions',
    },
    'transmission': {
        'label': 'Transmission Cost',
        'options': ['None', 'Low', 'Medium', 'High'],
        'members': ['wind', 'solar', 'clean_firm', 'ccs_ccgt', 'battery', 'ldes'],
        'description': 'Grid interconnection and transmission infrastructure costs affect all new-build resources within a region',
    },
}

# Full transmission adder tables (None/Low/Medium/High) for dashboard
FULL_TRANSMISSION_TABLES = {
    'wind': {
        'None':   {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 0, 'NEISO': 0},
        'Low':    {'CAISO': 4, 'ERCOT': 3, 'PJM': 5, 'NYISO': 7, 'NEISO': 6},
        'Medium': {'CAISO': 8, 'ERCOT': 6, 'PJM': 10, 'NYISO': 14, 'NEISO': 12},
        'High':   {'CAISO': 14, 'ERCOT': 10, 'PJM': 18, 'NYISO': 22, 'NEISO': 20},
    },
    'solar': {
        'None':   {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 0, 'NEISO': 0},
        'Low':    {'CAISO': 1, 'ERCOT': 1, 'PJM': 2, 'NYISO': 3, 'NEISO': 3},
        'Medium': {'CAISO': 3, 'ERCOT': 3, 'PJM': 5, 'NYISO': 7, 'NEISO': 6},
        'High':   {'CAISO': 6, 'ERCOT': 5, 'PJM': 9, 'NYISO': 12, 'NEISO': 10},
    },
    'clean_firm': {
        'None':   {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 0, 'NEISO': 0},
        'Low':    {'CAISO': 1, 'ERCOT': 1, 'PJM': 1, 'NYISO': 2, 'NEISO': 2},
        'Medium': {'CAISO': 3, 'ERCOT': 2, 'PJM': 3, 'NYISO': 5, 'NEISO': 4},
        'High':   {'CAISO': 6, 'ERCOT': 4, 'PJM': 6, 'NYISO': 9, 'NEISO': 7},
    },
    'ccs_ccgt': {
        'None':   {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 0, 'NEISO': 0},
        'Low':    {'CAISO': 1, 'ERCOT': 1, 'PJM': 1, 'NYISO': 2, 'NEISO': 2},
        'Medium': {'CAISO': 2, 'ERCOT': 2, 'PJM': 3, 'NYISO': 4, 'NEISO': 3},
        'High':   {'CAISO': 4, 'ERCOT': 3, 'PJM': 5, 'NYISO': 7, 'NEISO': 6},
    },
    'battery': {
        'None':   {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 0, 'NEISO': 0},
        'Low':    {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 1, 'NEISO': 1},
        'Medium': {'CAISO': 1, 'ERCOT': 1, 'PJM': 1, 'NYISO': 2, 'NEISO': 2},
        'High':   {'CAISO': 2, 'ERCOT': 2, 'PJM': 3, 'NYISO': 4, 'NEISO': 3},
    },
    'ldes': {
        'None':   {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 0, 'NEISO': 0},
        'Low':    {'CAISO': 1, 'ERCOT': 1, 'PJM': 1, 'NYISO': 2, 'NEISO': 2},
        'Medium': {'CAISO': 2, 'ERCOT': 2, 'PJM': 3, 'NYISO': 4, 'NEISO': 3},
        'High':   {'CAISO': 4, 'ERCOT': 3, 'PJM': 5, 'NYISO': 7, 'NEISO': 6},
    },
    'hydro': {
        'None':   {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 0, 'NEISO': 0},
        'Low':    {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 0, 'NEISO': 0},
        'Medium': {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 0, 'NEISO': 0},
        'High':   {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 0, 'NEISO': 0},
    },
}

# Fuel prices (L/M/H) for dashboard
FUEL_PRICES = {
    'natural_gas': {'Low': 2.00, 'Medium': 3.50, 'High': 6.00},
    'coal':        {'Low': 1.80, 'Medium': 2.50, 'High': 4.00},
    'oil':         {'Low': 55,   'Medium': 75,   'High': 110},
}

# Existing clean grid mix shares (% of total generation) from EIA-930 2025 data
# Resources up to these shares priced at wholesale; above = new-build LCOE
# CCS-CCGT has no existing share (entirely new-build)
GRID_MIX_SHARES = {
    'CAISO': {'clean_firm': 7.9, 'solar': 22.3, 'wind': 8.8, 'ccs_ccgt': 0, 'hydro': 9.5},
    'ERCOT': {'clean_firm': 8.6, 'solar': 13.8, 'wind': 23.6, 'ccs_ccgt': 0, 'hydro': 0.1},
    'PJM':   {'clean_firm': 32.1, 'solar': 2.9, 'wind': 3.8, 'ccs_ccgt': 0, 'hydro': 1.8},
    'NYISO': {'clean_firm': 18.4, 'solar': 0.0, 'wind': 4.7, 'ccs_ccgt': 0, 'hydro': 15.9},
    'NEISO': {'clean_firm': 23.8, 'solar': 1.4, 'wind': 3.9, 'ccs_ccgt': 0, 'hydro': 4.4},
}

# CCS-CCGT residual emission rate (tCO2/MWh) after 95% capture
CCS_RESIDUAL_EMISSION_RATE = 0.0185  # 95% capture from ~0.37 tCO2/MWh CCGT


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _remove_leap_day(profile_8784):
    """Remove Feb 29 hours (indices 1416-1439) from a leap year 8784-hour profile."""
    if len(profile_8784) <= H:
        return list(profile_8784[:H])
    return list(profile_8784[:LEAP_FEB29_START]) + list(profile_8784[LEAP_FEB29_START + 24:H + 24])


def _validate_demand_profile(iso, year, profile):
    """Detect corrupt demand profiles with statistical outlier detection.

    Returns True if profile is valid, False if it contains anomalous data.
    Known issue: PJM 2021 has a corrupt day (Oct 19) where hours 3-5 are
    ~20,000x normal values, likely an EIA unit conversion error.
    """
    arr = np.array(profile[:H])
    for hod in range(24):
        vals = arr[hod::24]
        median = np.median(vals)
        if median > 0 and vals.max() > 100 * median:
            print(f"  WARNING: {iso} {year} demand excluded — Hour {hod} outlier "
                  f"({vals.max():.4f} vs median {median:.6f}, "
                  f"{vals.max()/median:.0f}x)")
            return False
    return True


def _qa_qc_profiles(demand_data, gen_profiles):
    """QA/QC checkpoint: validate that all profiles make physical sense.

    Checks run after multi-year averaging, on final profiles used by optimizer.
    Failures are printed as warnings. Fatal issues raise ValueError.
    """
    print("\n  QA/QC: Validating profile shapes...")
    issues = []

    for iso in ISOS:
        # ── Demand shape checks ──
        norm = demand_data[iso]['normalized']
        arr = np.array(norm[:H])

        # 1. Sum should be ~1.0 (normalized)
        total = arr.sum()
        if abs(total - 1.0) > 0.01:
            issues.append(f"  FAIL: {iso} demand sum = {total:.4f} (expected ~1.0)")

        # 2. No negative values
        if arr.min() < 0:
            issues.append(f"  FAIL: {iso} demand has negative values (min={arr.min():.6f})")

        # 3. Diurnal pattern: daytime (hours 8-20) should be higher than nighttime (0-5)
        day_mean = np.mean([arr[h::24].mean() for h in range(8, 21)])
        night_mean = np.mean([arr[h::24].mean() for h in range(0, 6)])
        if night_mean > day_mean * 1.2:
            issues.append(f"  WARN: {iso} demand night > day by >20% "
                          f"(night={night_mean:.6f}, day={day_mean:.6f})")

        # 4. No single hour-of-day should exceed 3x the mean (after averaging)
        hod_sums = [arr[h::24].sum() for h in range(24)]
        mean_hod = np.mean(hod_sums)
        for h, s in enumerate(hod_sums):
            if s > 3 * mean_hod:
                issues.append(f"  FAIL: {iso} demand hour {h} = {s:.4f} "
                              f"({s/mean_hod:.1f}x mean — likely corrupt data)")

        # 5. Seasonal variation: summer/winter peaks should exist
        summer_mean = arr[3624:5832].mean()  # Jun-Aug approx
        winter_mean = np.concatenate([arr[:1416], arr[7296:]]).mean()  # Jan-Feb + Nov-Dec
        shoulder_mean = np.concatenate([arr[1416:3624], arr[5832:7296]]).mean()  # Mar-May, Sep-Oct
        print(f"    {iso} demand: summer={summer_mean:.6f} winter={winter_mean:.6f} "
              f"shoulder={shoulder_mean:.6f} (ratio={max(summer_mean,winter_mean)/shoulder_mean:.2f}x)")

        # ── Generation profile checks ──
        iso_gen = gen_profiles.get(iso, {})

        # Solar: should be zero at night (hours 22-5 local, roughly 3-10 UTC for US)
        solar_key = 'solar' if 'solar' in iso_gen else 'solar_proxy'
        if solar_key in iso_gen:
            solar = np.array(iso_gen[solar_key][:H])
            # Night hours (UTC 4-10 for eastern, 7-13 for pacific) — use conservative check
            night_total = sum(solar[h::24].sum() for h in range(2, 8))
            day_total = sum(solar[h::24].sum() for h in range(14, 22))
            if night_total > day_total * 0.1:
                issues.append(f"  WARN: {iso} solar night generation > 10% of daytime")

        # Wind: should have non-zero generation across most hours
        if 'wind' in iso_gen:
            wind = np.array(iso_gen['wind'][:H])
            zero_hours = (wind == 0).sum()
            if zero_hours > H * 0.3:
                issues.append(f"  WARN: {iso} wind has {zero_hours} zero hours ({zero_hours/H*100:.0f}%)")

    if issues:
        print("\n  QA/QC Results:")
        for issue in issues:
            print(f"    {issue}")
        # Fatal issues (FAIL) raise; warnings (WARN) continue
        fatal = [i for i in issues if 'FAIL' in i]
        if fatal:
            raise ValueError(f"QA/QC failed with {len(fatal)} fatal issues — fix data before running")
    else:
        print("    All profiles passed QA/QC checks")
    print()


def _average_profiles(yearly_profiles):
    """Element-wise average of multiple 8760-hour profiles."""
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


def load_data():
    """Load demand profiles, generation profiles, emission rates, and fossil mix.

    Data split (per SPEC.md §19.4):
      - Profile SHAPES: 5-year average (2021-2025) for weather smoothing
      - Scalar quantities: 2025 actuals (total_annual_mwh, peak_mw, grid mix, hydro caps)
    Leap year 2024 handled by removing Feb 29 before averaging.
    """
    print("Loading data...")

    with open(os.path.join(DATA_DIR, 'eia_demand_profiles.json')) as f:
        demand_raw = json.load(f)

    with open(os.path.join(DATA_DIR, 'eia_generation_profiles.json')) as f:
        gen_raw = json.load(f)

    with open(os.path.join(DATA_DIR, 'egrid_emission_rates.json')) as f:
        emission_rates = json.load(f)

    with open(os.path.join(DATA_DIR, 'eia_fossil_mix.json')) as f:
        fossil_mix = json.load(f)

    # ── Average generation profiles across PROFILE_YEARS ──
    gen_profiles = {}
    for iso in ISOS:
        iso_raw = gen_raw.get(iso, {})
        available_years = [y for y in PROFILE_YEARS if y in iso_raw]
        if not available_years:
            raise ValueError(f"No generation profile years found for {iso}")

        # Collect all resource types across years
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
                # Handle leap year 2024: remove Feb 29 to get 8760
                if len(raw) > H:
                    raw = _remove_leap_day(raw)
                else:
                    raw = list(raw[:H])
                yearly.append(raw)
            if yearly:
                gen_profiles[iso][rtype] = _average_profiles(yearly)

        n_yrs = len(available_years)
        print(f"  {iso}: gen profiles averaged over {n_yrs} years ({', '.join(available_years)})")

    # ── Average demand profiles; use 2025 actuals for scalars ──
    demand_data = {}
    for iso in ISOS:
        iso_data = demand_raw.get(iso, {})

        # Check if year-keyed
        year_keys = [k for k in iso_data.keys() if k.isdigit()]
        if not year_keys and 'normalized' in iso_data:
            # Old format: single year, use as-is
            demand_data[iso] = iso_data
            continue

        # Average normalized shape across available PROFILE_YEARS
        # Exclude years with corrupt data (detected via statistical outlier check)
        available_years = [y for y in PROFILE_YEARS if y in iso_data]
        if not available_years:
            raise ValueError(f"No demand data years found for {iso}")

        yearly_norms = []
        valid_years = []
        excluded_years = []
        for y in available_years:
            raw = iso_data[y].get('normalized', [])
            if len(raw) > H:
                raw = _remove_leap_day(raw)
            else:
                raw = list(raw[:H])
            if _validate_demand_profile(iso, y, raw):
                yearly_norms.append(raw)
                valid_years.append(y)
            else:
                excluded_years.append(y)

        if not yearly_norms:
            raise ValueError(f"All demand data years excluded for {iso} — check source data")

        avg_norm = _average_profiles(yearly_norms)

        # Use 2025 actuals for scalar quantities
        actuals_year = DATA_YEAR if DATA_YEAR in iso_data else available_years[-1]
        if actuals_year != DATA_YEAR:
            print(f"  Warning: {iso} demand using {actuals_year} actuals (2025 not found)")

        demand_data[iso] = {
            'normalized': avg_norm,
            'total_annual_mwh': iso_data[actuals_year]['total_annual_mwh'],
            'peak_mw': iso_data[actuals_year]['peak_mw'],
        }
        excluded_msg = f" (excluded: {', '.join(excluded_years)} — data quality)" if excluded_years else ""
        print(f"  {iso}: demand shape averaged over {len(valid_years)} years "
              f"({', '.join(valid_years)}){excluded_msg}, scalars from {actuals_year}")

    print("  Data loaded.")

    # QA/QC checkpoint: validate all profiles make physical sense
    _qa_qc_profiles(demand_data, gen_profiles)

    return demand_data, gen_profiles, emission_rates, fossil_mix


def get_supply_profiles(iso, gen_profiles):
    """Get generation shape profiles for the 5 resource types."""
    profiles = {}

    # Clean firm = seasonally-derated baseload
    # Nuclear portion gets spring/fall derate; geothermal stays flat
    nuc_share = NUCLEAR_SHARE_OF_CLEAN_FIRM.get(iso, 1.0)
    geo_share = 1.0 - nuc_share
    monthly_cf = NUCLEAR_MONTHLY_CF.get(iso, {m: 1.0 for m in range(1, 13)})
    cf_profile = []
    # Build 8760 profile: each hour gets its month's derate
    # Month boundaries for non-leap year (H=8760)
    month_hours = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
    hour = 0
    for month_idx, hours_in_month in enumerate(month_hours):
        month_num = month_idx + 1  # 1-indexed
        nuc_cf = monthly_cf.get(month_num, 1.0)
        # Blended CF: nuclear derated + geothermal flat
        blended = nuc_share * nuc_cf + geo_share * 1.0
        for _ in range(hours_in_month):
            if hour < H:
                cf_profile.append(blended / H)
                hour += 1
    # Fill any remaining hours (shouldn't happen, but safety)
    while len(cf_profile) < H:
        cf_profile.append(1.0 / H)
    profiles['clean_firm'] = cf_profile[:H]

    # Solar (with DST-aware nighttime correction)
    if iso == 'NYISO':
        p = gen_profiles[iso].get('solar_proxy')
        if not p:
            p = gen_profiles['NEISO'].get('solar')
        solar_raw = list(p[:H])
    else:
        solar_raw = list(gen_profiles[iso].get('solar', [0.0] * H)[:H])

    # Zero out nighttime solar — DST-aware UTC conversion
    # EIA data hours are sequential UTC. Convert local daylight windows to UTC,
    # adjusting the offset during DST months (March–November).
    # Standard UTC offsets (hours ahead of local): PST=8, CST=6, EST=5
    STD_UTC_OFFSETS = {'CAISO': 8, 'ERCOT': 6, 'PJM': 5, 'NYISO': 5, 'NEISO': 5}
    # DST boundaries: 2nd Sunday of March (~day 69) to 1st Sunday of Nov (~day 307)
    # Representative across 2021-2025 (actual dates: Mar 9-14, Nov 2-7)
    DST_START_DAY = 69    # ~March 10
    DST_END_DAY = 307     # ~November 3
    local_start, local_end = 6, 19  # 6am-7pm local prevailing time
    std_off = STD_UTC_OFFSETS.get(iso, 5)

    for day in range(H // 24):
        ds = day * 24
        # During DST, clocks spring forward → UTC offset is 1 less
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

    # CCS-CCGT = flat baseload (same shape as clean firm)
    profiles['ccs_ccgt'] = [1.0 / H] * H

    # Hydro
    profiles['hydro'] = gen_profiles[iso].get('hydro', [0.0] * H)[:H]

    # Ensure all profiles are exactly H hours with no negative values
    for rtype in RESOURCE_TYPES:
        if len(profiles[rtype]) > H:
            profiles[rtype] = profiles[rtype][:H]
        elif len(profiles[rtype]) < H:
            profiles[rtype] = profiles[rtype] + [0.0] * (H - len(profiles[rtype]))
        # Clamp negatives (floating point noise, pumped hydro charging)
        profiles[rtype] = [max(0.0, v) for v in profiles[rtype]]

    return profiles


def find_anomaly_hours(iso, gen_profiles):
    """Find hours where all gen types report zero (EIA data gaps)."""
    types = [t for t in gen_profiles[iso].keys() if t != 'solar_proxy']
    anomalies = set()
    for h in range(H):
        if all(gen_profiles[iso][t][h] == 0.0 for t in types):
            anomalies.add(h)
    return anomalies


# ══════════════════════════════════════════════════════════════════════════════
# NUMPY PROFILE PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def prepare_numpy_profiles(demand_norm, supply_profiles):
    """Convert profiles to numpy arrays for fast vectorized computation."""
    demand_arr = np.array(demand_norm[:H], dtype=np.float64)
    supply_arrs = {}
    for rtype in RESOURCE_TYPES:
        supply_arrs[rtype] = np.array(supply_profiles[rtype][:H], dtype=np.float64)
    # Pre-build supply matrix: shape (5, 8760) for [cf, solar, wind, ccs_ccgt, hydro]
    supply_matrix = np.stack([supply_arrs[rt] for rt in RESOURCE_TYPES])  # (5, 8760)
    return demand_arr, supply_arrs, supply_matrix


# ══════════════════════════════════════════════════════════════════════════════
# FAST SCORING FUNCTIONS (numpy-accelerated)
# ══════════════════════════════════════════════════════════════════════════════

def fast_hourly_score(demand_arr, supply_matrix, mix_fractions, procurement_factor):
    """
    Ultra-fast hourly matching score using numpy vectorized ops.
    mix_fractions: array of [cf, solar, wind, ccs_ccgt, hydro] as fractions (sum to 1.0)
    Returns: matching score (0-1), where demand sums to 1.0
    """
    supply = procurement_factor * np.dot(mix_fractions, supply_matrix)
    matched = np.minimum(demand_arr, supply)
    return matched.sum()


def fast_score_with_battery(demand_arr, supply_matrix, mix_fractions, procurement_factor,
                            battery_dispatch_pct):
    """
    Fast scoring with capacity-constrained battery storage (4hr Li-ion, 85% RTE).

    battery_dispatch_pct maps to a CAPACITY, not a guaranteed dispatch amount.
    Capacity (MWh) = battery_dispatch_pct / 100.0 (normalized to demand sum ~1.0).
    Power rating (MW) = capacity / 4hr.

    Each day: charge from available surplus up to capacity, discharge to gaps.
    Days with low surplus → partial cycle → less dispatch.
    Actual annual dispatch varies based on surplus availability.

    Returns matching score (0-1).
    """
    supply = procurement_factor * np.dot(mix_fractions, supply_matrix)
    surplus = np.maximum(0.0, supply - demand_arr)
    gap = np.maximum(0.0, demand_arr - supply)
    base_matched = np.minimum(demand_arr, supply)

    if battery_dispatch_pct <= 0:
        return base_matched.sum()

    # Capacity-based sizing: pct maps to energy capacity
    capacity_mwh = battery_dispatch_pct / 100.0  # normalized energy capacity
    power_rating = capacity_mwh / BATTERY_DURATION_HOURS

    total_dispatched = 0.0
    num_days = H // 24
    for day in range(num_days):
        ds = day * 24
        de = ds + 24

        day_surplus = surplus[ds:de]
        day_gap = gap[ds:de]

        total_surplus = day_surplus.sum()
        total_gap = day_gap.sum()

        # Charge limited by: available surplus, capacity, power rating × hours
        max_charge = min(total_surplus, capacity_mwh)
        max_from_charge = max_charge * BATTERY_EFFICIENCY
        # Dispatch limited by: charged energy × RTE, gap, capacity
        actual_dispatch = min(max_from_charge, total_gap, capacity_mwh)
        if actual_dispatch <= 0:
            continue

        required_charge = actual_dispatch / BATTERY_EFFICIENCY

        # Distribute charge (greedily, largest surplus first)
        sorted_idx = np.argsort(-day_surplus)
        pos_mask = day_surplus[sorted_idx] > 0
        sorted_idx = sorted_idx[pos_mask]
        remaining_charge = required_charge
        for idx in sorted_idx:
            if remaining_charge <= 1e-12:
                break
            amt = min(float(day_surplus[idx]), power_rating, remaining_charge)
            remaining_charge -= amt

        actual_charge = required_charge - remaining_charge
        ach_dispatch = min(actual_dispatch, actual_charge * BATTERY_EFFICIENCY)

        # Distribute dispatch (greedily, largest gap first)
        sorted_idx = np.argsort(-day_gap)
        pos_mask = day_gap[sorted_idx] > 0
        sorted_idx = sorted_idx[pos_mask]
        remaining_dispatch = ach_dispatch
        for idx in sorted_idx:
            if remaining_dispatch <= 1e-12:
                break
            amt = min(float(day_gap[idx]), power_rating, remaining_dispatch)
            total_dispatched += amt
            remaining_dispatch -= amt

    return base_matched.sum() + total_dispatched


def compute_ldes_dispatch(demand_arr, supply_arr_total, ldes_dispatch_pct=10):
    """
    LDES dispatch algorithm: 100hr iron-air, 50% RTE, 7-day rolling window.
    Capacity-constrained with dynamic sizing.

    ldes_dispatch_pct maps to CAPACITY (not guaranteed dispatch).
    Capacity (MWh) = ldes_dispatch_pct / 100.0 (normalized to demand).
    Power rating (MW) = capacity / 100hr.

    Args:
        demand_arr: numpy array (H,) of normalized demand
        supply_arr_total: numpy array (H,) of total supply after resource mix
        ldes_dispatch_pct: capacity sizing parameter (% of normalized demand)

    Returns:
        ldes_dispatch: numpy array (H,) of LDES dispatch amounts (added to matched)
        ldes_charge: numpy array (H,) of LDES charge amounts (absorbed from surplus)
        total_dispatched: float total energy dispatched
    """
    surplus = np.maximum(0.0, supply_arr_total - demand_arr)
    gap = np.maximum(0.0, demand_arr - supply_arr_total)

    ldes_dispatch = np.zeros(H, dtype=np.float64)
    ldes_charge = np.zeros(H, dtype=np.float64)

    # Dynamic capacity sizing: scales with ldes_dispatch_pct
    ldes_energy_capacity = ldes_dispatch_pct / 100.0  # normalized energy capacity
    ldes_power_rating = ldes_energy_capacity / LDES_DURATION_HOURS

    state_of_charge = 0.0
    window_hours = LDES_WINDOW_DAYS * 24

    # Process in 7-day rolling windows
    num_windows = (H + window_hours - 1) // window_hours
    for w in range(num_windows):
        w_start = w * window_hours
        w_end = min(w_start + window_hours, H)

        # Identify surplus and deficit hours in this window
        w_surplus = surplus[w_start:w_end].copy()
        w_gap = gap[w_start:w_end].copy()

        # Phase 1: Charge during surplus hours (largest surpluses first)
        surplus_indices = np.argsort(-w_surplus)
        # Filter to only positive-surplus hours for faster iteration
        pos_mask = w_surplus[surplus_indices] > 0
        surplus_indices = surplus_indices[pos_mask]
        for idx in surplus_indices:
            space = ldes_energy_capacity - state_of_charge
            if space <= 1e-12:
                break
            charge_amt = min(float(w_surplus[idx]), ldes_power_rating, space)
            ldes_charge[w_start + idx] = charge_amt
            state_of_charge += charge_amt

        # Phase 2: Discharge during deficit hours (largest gaps first)
        gap_indices = np.argsort(-w_gap)
        pos_mask = w_gap[gap_indices] > 0
        gap_indices = gap_indices[pos_mask]
        for idx in gap_indices:
            available = state_of_charge * LDES_EFFICIENCY
            if available <= 1e-12:
                break
            dispatch_amt = min(float(w_gap[idx]), ldes_power_rating, available)
            ldes_dispatch[w_start + idx] = dispatch_amt
            state_of_charge -= dispatch_amt / LDES_EFFICIENCY
            state_of_charge = max(0.0, state_of_charge)

    total_dispatched = ldes_dispatch.sum()
    return ldes_dispatch, ldes_charge, total_dispatched


def fast_score_with_ldes(demand_arr, supply_matrix, mix_fractions, procurement_factor,
                         ldes_dispatch_pct):
    """
    Fast scoring with LDES only (no battery). Used in sweep/optimization.
    ldes_dispatch_pct maps to LDES capacity (dynamic sizing).
    Returns matching score (0-1).
    """
    supply = procurement_factor * np.dot(mix_fractions, supply_matrix)

    if ldes_dispatch_pct <= 0:
        matched = np.minimum(demand_arr, supply)
        return matched.sum()

    base_matched = np.minimum(demand_arr, supply)
    ldes_dispatch_arr, _, total_dispatched = compute_ldes_dispatch(
        demand_arr, supply, ldes_dispatch_pct=ldes_dispatch_pct)

    # Dispatch is already capacity-constrained; just cap at remaining gap
    gap = np.maximum(0.0, demand_arr - supply)
    capped_dispatch = np.minimum(ldes_dispatch_arr, gap)

    return base_matched.sum() + capped_dispatch.sum()


def fast_score_with_both_storage(demand_arr, supply_matrix, mix_fractions, procurement_factor,
                                 battery_dispatch_pct, ldes_dispatch_pct):
    """
    Fast scoring with both capacity-constrained battery (daily) and LDES (multi-day).
    Battery runs first on daily cycle, LDES fills remaining multi-day gaps.
    Both use capacity-based sizing where dispatch_pct maps to built capacity.
    Returns matching score (0-1).
    """
    supply = procurement_factor * np.dot(mix_fractions, supply_matrix)
    surplus = np.maximum(0.0, supply - demand_arr)
    gap = np.maximum(0.0, demand_arr - supply)
    base_matched = np.minimum(demand_arr, supply)

    total_dispatched = 0.0
    residual_gap = gap.copy()
    residual_surplus = surplus.copy()

    # Phase 1: Battery capacity-constrained daily-cycle dispatch
    if battery_dispatch_pct > 0:
        batt_capacity = battery_dispatch_pct / 100.0  # normalized energy capacity
        batt_power_rating = batt_capacity / BATTERY_DURATION_HOURS
        num_days = H // 24

        for day in range(num_days):
            ds = day * 24
            de = ds + 24

            day_surplus = residual_surplus[ds:de]
            day_gap = residual_gap[ds:de]

            total_surplus_day = day_surplus.sum()
            total_gap_day = day_gap.sum()

            # Capacity-constrained: charge limited by surplus and capacity
            max_charge = min(total_surplus_day, batt_capacity)
            max_from_charge = max_charge * BATTERY_EFFICIENCY
            actual_dispatch = min(max_from_charge, total_gap_day, batt_capacity)
            if actual_dispatch <= 0:
                continue

            required_charge = actual_dispatch / BATTERY_EFFICIENCY

            # Charge from largest surpluses
            sorted_idx = np.argsort(-day_surplus)
            pos_mask = day_surplus[sorted_idx] > 0
            sorted_idx = sorted_idx[pos_mask]
            remaining_charge = required_charge
            for idx in sorted_idx:
                if remaining_charge <= 1e-12:
                    break
                amt = min(float(day_surplus[idx]), batt_power_rating, remaining_charge)
                residual_surplus[ds + idx] -= amt
                remaining_charge -= amt

            actual_charge = required_charge - remaining_charge
            ach_dispatch = min(actual_dispatch, actual_charge * BATTERY_EFFICIENCY)

            # Dispatch to largest gaps
            sorted_idx = np.argsort(-day_gap)
            pos_mask = day_gap[sorted_idx] > 0
            sorted_idx = sorted_idx[pos_mask]
            remaining_dispatch = ach_dispatch
            for idx in sorted_idx:
                if remaining_dispatch <= 1e-12:
                    break
                amt = min(float(day_gap[idx]), batt_power_rating, remaining_dispatch)
                residual_gap[ds + idx] -= amt
                total_dispatched += amt
                remaining_dispatch -= amt

    # Phase 2: LDES capacity-constrained multi-day dispatch on remaining gaps
    if ldes_dispatch_pct > 0:
        ldes_energy_capacity = ldes_dispatch_pct / 100.0  # dynamic capacity sizing
        ldes_power_rating = ldes_energy_capacity / LDES_DURATION_HOURS

        state_of_charge = 0.0
        window_hours = LDES_WINDOW_DAYS * 24

        num_windows = (H + window_hours - 1) // window_hours
        for w in range(num_windows):
            w_start = w * window_hours
            w_end = min(w_start + window_hours, H)

            w_surplus = residual_surplus[w_start:w_end]
            w_gap = residual_gap[w_start:w_end]

            # Charge
            surplus_indices = np.argsort(-w_surplus)
            pos_mask = w_surplus[surplus_indices] > 0
            surplus_indices = surplus_indices[pos_mask]
            for idx in surplus_indices:
                space = ldes_energy_capacity - state_of_charge
                if space <= 1e-12:
                    break
                charge_amt = min(float(w_surplus[idx]), ldes_power_rating, space)
                state_of_charge += charge_amt

            # Discharge
            gap_indices = np.argsort(-w_gap)
            pos_mask = w_gap[gap_indices] > 0
            gap_indices = gap_indices[pos_mask]
            for idx in gap_indices:
                available = state_of_charge * LDES_EFFICIENCY
                if available <= 1e-12:
                    break
                dispatch_amt = min(float(w_gap[idx]), ldes_power_rating, available)
                if dispatch_amt > 0:
                    total_dispatched += dispatch_amt
                    state_of_charge -= dispatch_amt / LDES_EFFICIENCY
                    state_of_charge = max(0.0, state_of_charge)

    return base_matched.sum() + total_dispatched


# ══════════════════════════════════════════════════════════════════════════════
# DETAILED DISPATCH COMPUTATION (for final results)
# ══════════════════════════════════════════════════════════════════════════════

def compute_hourly_matching_detailed(demand_norm, supply_profiles, resource_pcts, procurement_pct,
                                     battery_dispatch_pct=0, ldes_dispatch_pct=0):
    """
    Compute detailed hourly matching with both storage types.
    Returns: (score, hourly_detail, battery_dispatch_profile, battery_charge_profile,
              ldes_dispatch_profile, ldes_charge_profile)
    """
    procurement_factor = procurement_pct / 100.0

    hourly_detail = []
    supply_total = np.zeros(H, dtype=np.float64)

    for h in range(H):
        demand_h = demand_norm[h]
        supply_h = 0.0
        for rtype, pct in resource_pcts.items():
            if pct <= 0:
                continue
            supply_h += procurement_factor * (pct / 100.0) * supply_profiles[rtype][h]

        supply_total[h] = supply_h
        matched_h = min(demand_h, supply_h)
        surplus_h = max(0.0, supply_h - demand_h)
        gap_h = max(0.0, demand_h - supply_h)

        hourly_detail.append({
            'demand': demand_h,
            'supply': supply_h,
            'matched': matched_h,
            'surplus': surplus_h,
            'gap': gap_h,
        })

    demand_arr = np.array([d['demand'] for d in hourly_detail], dtype=np.float64)
    surplus_arr = np.array([d['surplus'] for d in hourly_detail], dtype=np.float64)
    gap_arr = np.array([d['gap'] for d in hourly_detail], dtype=np.float64)

    battery_dispatch_profile = np.zeros(H, dtype=np.float64)
    battery_charge_profile = np.zeros(H, dtype=np.float64)
    ldes_dispatch_profile = np.zeros(H, dtype=np.float64)
    ldes_charge_profile = np.zeros(H, dtype=np.float64)

    residual_surplus = surplus_arr.copy()
    residual_gap = gap_arr.copy()

    # Phase 1: Battery capacity-constrained daily-cycle dispatch
    if battery_dispatch_pct > 0:
        batt_capacity = battery_dispatch_pct / 100.0  # normalized energy capacity
        batt_power_rating = batt_capacity / BATTERY_DURATION_HOURS
        num_days = H // 24

        for day in range(num_days):
            ds = day * 24
            de = ds + 24

            day_surplus = residual_surplus[ds:de]
            day_gap = residual_gap[ds:de]

            total_surplus_day = day_surplus.sum()
            total_gap_day = day_gap.sum()

            # Capacity-constrained: charge limited by surplus and capacity
            max_charge = min(total_surplus_day, batt_capacity)
            max_from_charge = max_charge * BATTERY_EFFICIENCY
            actual_dispatch = min(max_from_charge, total_gap_day, batt_capacity)
            if actual_dispatch <= 0:
                continue

            required_charge = actual_dispatch / BATTERY_EFFICIENCY

            # Charge from largest surpluses
            sorted_idx = np.argsort(-day_surplus)
            remaining_charge = required_charge
            for idx in sorted_idx:
                if remaining_charge <= 0 or day_surplus[idx] <= 0:
                    break
                amt = min(float(day_surplus[idx]), batt_power_rating, remaining_charge)
                battery_charge_profile[ds + idx] = amt
                residual_surplus[ds + idx] -= amt
                remaining_charge -= amt

            actual_charge = required_charge - remaining_charge
            ach_dispatch = min(actual_dispatch, actual_charge * BATTERY_EFFICIENCY)

            # Dispatch to largest gaps
            sorted_idx = np.argsort(-day_gap)
            remaining_dispatch = ach_dispatch
            for idx in sorted_idx:
                if remaining_dispatch <= 0 or day_gap[idx] <= 0:
                    break
                amt = min(float(day_gap[idx]), batt_power_rating, remaining_dispatch)
                battery_dispatch_profile[ds + idx] = amt
                residual_gap[ds + idx] -= amt
                remaining_dispatch -= amt

    # Phase 2: LDES capacity-constrained multi-day dispatch on remaining gaps
    if ldes_dispatch_pct > 0:
        ldes_energy_capacity = ldes_dispatch_pct / 100.0  # dynamic capacity sizing
        ldes_power_rating = ldes_energy_capacity / LDES_DURATION_HOURS

        state_of_charge = 0.0
        window_hours = LDES_WINDOW_DAYS * 24

        num_windows = (H + window_hours - 1) // window_hours
        for w in range(num_windows):
            w_start = w * window_hours
            w_end = min(w_start + window_hours, H)

            w_surplus = residual_surplus[w_start:w_end]
            w_gap = residual_gap[w_start:w_end]

            # Charge from surplus
            surplus_indices = np.argsort(-w_surplus)
            for idx in surplus_indices:
                if w_surplus[idx] <= 0:
                    break
                space = ldes_energy_capacity - state_of_charge
                if space <= 0:
                    break
                charge_amt = min(float(w_surplus[idx]), ldes_power_rating, space)
                if charge_amt > 0:
                    ldes_charge_profile[w_start + idx] = charge_amt
                    residual_surplus[w_start + idx] -= charge_amt
                    state_of_charge += charge_amt

            # Discharge to gaps
            gap_indices = np.argsort(-w_gap)
            for idx in gap_indices:
                if w_gap[idx] <= 0:
                    break
                available = state_of_charge * LDES_EFFICIENCY
                if available <= 1e-12:
                    break
                dispatch_amt = min(float(w_gap[idx]), ldes_power_rating, available)
                if dispatch_amt > 0:
                    ldes_dispatch_profile[w_start + idx] = dispatch_amt
                    residual_gap[w_start + idx] -= dispatch_amt
                    state_of_charge -= dispatch_amt / LDES_EFFICIENCY
                    state_of_charge = max(0.0, state_of_charge)

    # Compute final score
    total_matched = 0.0
    total_demand = 0.0
    for h in range(H):
        d = hourly_detail[h]
        batt_disp = battery_dispatch_profile[h]
        ldes_disp = ldes_dispatch_profile[h]
        new_matched = d['matched'] + min(d['gap'], batt_disp + ldes_disp)
        total_matched += new_matched
        total_demand += d['demand']

    score = total_matched / total_demand if total_demand > 0 else 0.0
    return (score, hourly_detail,
            battery_dispatch_profile, battery_charge_profile,
            ldes_dispatch_profile, ldes_charge_profile)


# ══════════════════════════════════════════════════════════════════════════════
# 5D COMBINATION GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

# Edge case seed mixes — injected into Phase 1 to ensure extreme but potentially
# optimal mixes survive coarse pruning. Under low-cost renewables, a solar-dominant
# mix with heavy storage may be globally cheapest for <90% targets. Under low-cost
# firm generation, clean firm or CCS-dominant mixes may win. The coarse grid
# generates these combos but may not rank them highly enough to survive to Phase 2/3
# under all 324 cost scenarios. Seeds guarantee they're always evaluated.
#
# Format: {clean_firm, solar, wind, ccs_ccgt, hydro} — must sum to 100%
# Hydro-dependent seeds filtered at runtime by regional hydro cap.
EDGE_CASE_SEEDS = [
    # Low-cost renewable: solar-dominant
    {'clean_firm': 10, 'solar': 70, 'wind': 10, 'ccs_ccgt': 0, 'hydro': 10},
    {'clean_firm': 5,  'solar': 75, 'wind': 10, 'ccs_ccgt': 0, 'hydro': 10},
    {'clean_firm': 10, 'solar': 60, 'wind': 20, 'ccs_ccgt': 0, 'hydro': 10},
    # Low-cost renewable: wind-dominant
    {'clean_firm': 10, 'solar': 10, 'wind': 70, 'ccs_ccgt': 0, 'hydro': 10},
    {'clean_firm': 5,  'solar': 10, 'wind': 75, 'ccs_ccgt': 0, 'hydro': 10},
    {'clean_firm': 10, 'solar': 20, 'wind': 60, 'ccs_ccgt': 0, 'hydro': 10},
    # Low-cost renewable: balanced solar+wind
    {'clean_firm': 10, 'solar': 40, 'wind': 40, 'ccs_ccgt': 0, 'hydro': 10},
    {'clean_firm': 5,  'solar': 45, 'wind': 45, 'ccs_ccgt': 0, 'hydro': 5},
    # Low-cost clean firm: nuclear/geothermal dominant
    {'clean_firm': 70, 'solar': 10, 'wind': 10, 'ccs_ccgt': 0, 'hydro': 10},
    {'clean_firm': 80, 'solar': 10, 'wind': 10, 'ccs_ccgt': 0, 'hydro': 0},
    {'clean_firm': 60, 'solar': 15, 'wind': 15, 'ccs_ccgt': 0, 'hydro': 10},
    # Low-cost firm: combined clean firm + CCS
    {'clean_firm': 40, 'solar': 10, 'wind': 10, 'ccs_ccgt': 30, 'hydro': 10},
    {'clean_firm': 30, 'solar': 10, 'wind': 10, 'ccs_ccgt': 40, 'hydro': 10},
    # CCS-dominant (cheap firm gen + favorable geology)
    {'clean_firm': 20, 'solar': 15, 'wind': 15, 'ccs_ccgt': 50, 'hydro': 0},
    {'clean_firm': 10, 'solar': 10, 'wind': 10, 'ccs_ccgt': 60, 'hydro': 10},
    # High-hydro regions (NYISO, CAISO, NEISO) — hydro + firm
    {'clean_firm': 30, 'solar': 10, 'wind': 10, 'ccs_ccgt': 10, 'hydro': 40},
    {'clean_firm': 20, 'solar': 20, 'wind': 10, 'ccs_ccgt': 10, 'hydro': 40},
]


def generate_combinations(hydro_cap, step=5, max_single=80, min_dispatchable=0):
    """
    Generate all valid resource mix combinations that sum to 100%.
    Resources: clean_firm, solar, wind, ccs_ccgt, hydro
    Hydro capped by region. No single resource exceeds max_single%.
    min_dispatchable: floor on (clean_firm + ccs_ccgt) %, derived from prior threshold.
    """
    combos = []
    for cf in range(0, min(max_single + 1, 101), step):
        for sol in range(0, min(max_single + 1, 101 - cf), step):
            for wnd in range(0, min(max_single + 1, 101 - cf - sol), step):
                # Early prune: max possible ccs is (100 - cf - sol - wnd),
                # so if cf + max_ccs < floor, skip entire wnd iteration
                max_ccs = 100 - cf - sol - wnd
                if cf + max_ccs < min_dispatchable:
                    continue
                for ccs in range(0, min(max_single + 1, 101 - cf - sol - wnd), step):
                    if cf + ccs < min_dispatchable:
                        continue
                    hyd = 100 - cf - sol - wnd - ccs
                    if hyd >= 0 and hyd <= hydro_cap and hyd <= max_single:
                        combos.append({
                            'clean_firm': cf, 'solar': sol, 'wind': wnd,
                            'ccs_ccgt': ccs, 'hydro': hyd
                        })
    return combos


def get_seed_combos(hydro_cap, min_dispatchable=0):
    """
    Return edge case seed mixes valid for this region's hydro cap.
    Filters out seeds where hydro exceeds regional cap or dispatchable < floor.
    """
    valid = []
    seen = set()
    for seed in EDGE_CASE_SEEDS:
        if seed['hydro'] > hydro_cap:
            continue
        if seed['clean_firm'] + seed['ccs_ccgt'] < min_dispatchable:
            continue
        key = tuple(seed[rt] for rt in RESOURCE_TYPES)
        if key not in seen:
            seen.add(key)
            valid.append(dict(seed))
    return valid


def generate_combinations_around(base_combo, hydro_cap, step=1, radius=2, min_dispatchable=0):
    """
    Generate combinations in a neighborhood around base_combo with given step and radius.
    min_dispatchable: floor on (clean_firm + ccs_ccgt) %, derived from prior threshold.
    """
    combos = []
    seen = set()
    ranges = {}
    for rtype in RESOURCE_TYPES:
        base = int(base_combo[rtype])
        cap = int(hydro_cap) if rtype == 'hydro' else 100
        low = max(0, base - radius * step)
        high = min(cap, base + radius * step)
        ranges[rtype] = list(range(low, high + 1, step))

    for cf in ranges['clean_firm']:
        for sol in ranges['solar']:
            for wnd in ranges['wind']:
                for ccs in ranges['ccs_ccgt']:
                    if cf + ccs < min_dispatchable:
                        continue
                    hyd = 100 - cf - sol - wnd - ccs
                    if hyd < 0 or hyd > hydro_cap:
                        continue
                    key = (cf, sol, wnd, ccs, hyd)
                    if key in seen:
                        continue
                    seen.add(key)
                    combos.append({
                        'clean_firm': cf, 'solar': sol, 'wind': wnd,
                        'ccs_ccgt': ccs, 'hydro': hyd
                    })
    return combos


# ══════════════════════════════════════════════════════════════════════════════
# STORAGE OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════════

def find_optimal_storage(demand_arr, supply_matrix, mix_fractions, procurement_factor, hydro_cap):
    """
    Find the optimal battery + LDES dispatch percentages that maximize hourly matching.
    Sweep battery from 0-30% and LDES from 0-20%, then refine.
    """
    best_score = fast_hourly_score(demand_arr, supply_matrix, mix_fractions, procurement_factor)
    best_batt = 0
    best_ldes = 0

    # Coarse sweep
    for bp in range(0, 35, 5):
        for lp in range(0, 25, 5):
            if bp == 0 and lp == 0:
                continue
            score = fast_score_with_both_storage(
                demand_arr, supply_matrix, mix_fractions, procurement_factor, bp, lp
            )
            if score > best_score:
                best_score = score
                best_batt = bp
                best_ldes = lp

    # Fine sweep around best
    fine_best_score = best_score
    fine_best_batt = best_batt
    fine_best_ldes = best_ldes
    for bp in range(max(0, best_batt - 4), best_batt + 5, 2):
        for lp in range(max(0, best_ldes - 4), best_ldes + 5, 2):
            score = fast_score_with_both_storage(
                demand_arr, supply_matrix, mix_fractions, procurement_factor, bp, lp
            )
            if score > fine_best_score:
                fine_best_score = score
                fine_best_batt = bp
                fine_best_ldes = lp

    return fine_best_score, fine_best_batt, fine_best_ldes


# ══════════════════════════════════════════════════════════════════════════════
# CO2 ABATEMENT
# ══════════════════════════════════════════════════════════════════════════════

def compute_co2_abatement(iso, resource_pcts, procurement_pct, hourly_match_score,
                          battery_dispatch_pct, ldes_dispatch_pct,
                          emission_rates, demand_total_mwh,
                          demand_norm=None, supply_profiles=None, fossil_mix=None):
    """
    Compute CO2 abatement using hourly fossil-fuel emission rates.

    Methodology:
      1. Build hourly emission rate from eGRID per-fuel rates × EIA hourly fossil mix
         emission_rate[h] = coal_share[h]×coal_rate + gas_share[h]×gas_rate + oil_share[h]×oil_rate
      2. Reconstruct hourly clean supply → fossil displacement at each hour
      3. CO2_abated = Σ_h fossil_displaced[h] × emission_rate[h]
      4. CCS-CCGT: partial credit (rate[h] - 0.0185 tCO2/MWh residual, 95% capture)

    Falls back to flat regional rate if hourly data not provided (backward compat).
    """
    regional_data = emission_rates.get(iso, {})

    # Per-fuel emission rates from eGRID (lb/MWh → metric tons/MWh)
    coal_rate = regional_data.get('coal_co2_lb_per_mwh', 0.0) / 2204.62
    gas_rate = regional_data.get('gas_co2_lb_per_mwh', 0.0) / 2204.62
    oil_rate = regional_data.get('oil_co2_lb_per_mwh', 0.0) / 2204.62

    # ── Hourly emission rate calculation ──
    use_hourly = (demand_norm is not None and supply_profiles is not None
                  and fossil_mix is not None)

    if use_hourly:
        # Build hourly emission rates from fossil mix shares
        iso_fossil = fossil_mix.get(iso, {})
        year_data = iso_fossil.get(DATA_YEAR, iso_fossil.get('2024', {}))

        coal_shares = np.array(year_data.get('coal_share', [0.0] * H)[:H], dtype=np.float64)
        gas_shares = np.array(year_data.get('gas_share', [1.0] * H)[:H], dtype=np.float64)
        oil_shares = np.array(year_data.get('oil_share', [0.0] * H)[:H], dtype=np.float64)

        # Pad to H if shorter
        for arr_name in ['coal_shares', 'gas_shares', 'oil_shares']:
            arr = locals()[arr_name]
            if len(arr) < H:
                locals()[arr_name] = np.pad(arr, (0, H - len(arr)), mode='edge')

        hourly_rates = coal_shares * coal_rate + gas_shares * gas_rate + oil_shares * oil_rate

        # Reconstruct hourly clean supply and fossil displacement
        procurement_factor = procurement_pct / 100.0
        demand_arr = np.array(demand_norm[:H], dtype=np.float64)
        supply_total = np.zeros(H, dtype=np.float64)
        ccs_supply = np.zeros(H, dtype=np.float64)

        for rtype in RESOURCE_TYPES:
            pct = resource_pcts.get(rtype, 0)
            if pct <= 0:
                continue
            profile = np.array(supply_profiles[rtype][:H], dtype=np.float64)
            contribution = procurement_factor * (pct / 100.0) * profile
            supply_total += contribution
            if rtype == 'ccs_ccgt':
                ccs_supply = contribution.copy()

        # Direct generation displacement (no storage)
        fossil_displaced = np.minimum(demand_arr, supply_total)
        matched_mwh = np.sum(fossil_displaced) * demand_total_mwh

        # Non-CCS clean displacement
        ccs_matched = np.minimum(fossil_displaced, ccs_supply)
        non_ccs_matched = fossil_displaced - ccs_matched

        # CO₂ from non-CCS: full credit at hourly rate
        co2_clean = np.sum(non_ccs_matched * hourly_rates) * demand_total_mwh
        # CO₂ from CCS: partial credit (95% capture)
        ccs_credit = np.maximum(0.0, hourly_rates - CCS_RESIDUAL_EMISSION_RATE)
        co2_ccs = np.sum(ccs_matched * ccs_credit) * demand_total_mwh

        # ── Storage CO₂: hourly dispatch attribution with charge-side netting ──
        # Run actual storage dispatch to get hourly charge/discharge profiles
        surplus = np.maximum(0.0, supply_total - demand_arr)
        gap = np.maximum(0.0, demand_arr - supply_total)
        residual_surplus = surplus.copy()
        residual_gap = gap.copy()

        batt_dispatch_hrs = np.zeros(H, dtype=np.float64)
        batt_charge_hrs = np.zeros(H, dtype=np.float64)
        ldes_dispatch_hrs = np.zeros(H, dtype=np.float64)
        ldes_charge_hrs = np.zeros(H, dtype=np.float64)

        # Battery dispatch (capacity-constrained)
        if battery_dispatch_pct > 0:
            batt_capacity = battery_dispatch_pct / 100.0
            batt_pr = batt_capacity / BATTERY_DURATION_HOURS
            num_days = H // 24
            for day in range(num_days):
                ds, de = day * 24, (day + 1) * 24
                ds_surplus = residual_surplus[ds:de]
                ds_gap = residual_gap[ds:de]
                max_charge = min(ds_surplus.sum(), batt_capacity)
                max_dispatch = min(max_charge * BATTERY_EFFICIENCY, ds_gap.sum(), batt_capacity)
                if max_dispatch <= 0:
                    continue
                req_charge = max_dispatch / BATTERY_EFFICIENCY
                for idx in np.argsort(-ds_surplus):
                    if req_charge <= 0 or ds_surplus[idx] <= 0:
                        break
                    amt = min(float(ds_surplus[idx]), batt_pr, req_charge)
                    batt_charge_hrs[ds + idx] = amt
                    residual_surplus[ds + idx] -= amt
                    req_charge -= amt
                actual_charge_done = (max_dispatch / BATTERY_EFFICIENCY) - req_charge
                actual_batt_dispatch = min(max_dispatch, actual_charge_done * BATTERY_EFFICIENCY)
                rem = actual_batt_dispatch
                for idx in np.argsort(-ds_gap):
                    if rem <= 0 or ds_gap[idx] <= 0:
                        break
                    amt = min(float(ds_gap[idx]), batt_pr, rem)
                    batt_dispatch_hrs[ds + idx] = amt
                    residual_gap[ds + idx] -= amt
                    rem -= amt

        # LDES dispatch (capacity-constrained, dynamic sizing)
        if ldes_dispatch_pct > 0:
            ldes_cap = ldes_dispatch_pct / 100.0
            ldes_pr = ldes_cap / LDES_DURATION_HOURS
            soc = 0.0
            wh = LDES_WINDOW_DAYS * 24
            for w in range((H + wh - 1) // wh):
                ws, we = w * wh, min((w + 1) * wh, H)
                ws_surplus = residual_surplus[ws:we]
                ws_gap = residual_gap[ws:we]
                for idx in np.argsort(-ws_surplus):
                    if ws_surplus[idx] <= 0:
                        break
                    space = ldes_cap - soc
                    if space <= 0:
                        break
                    amt = min(float(ws_surplus[idx]), ldes_pr, space)
                    if amt > 0:
                        ldes_charge_hrs[ws + idx] = amt
                        residual_surplus[ws + idx] -= amt
                        soc += amt
                for idx in np.argsort(-ws_gap):
                    if ws_gap[idx] <= 0:
                        break
                    avail = soc * LDES_EFFICIENCY
                    if avail <= 1e-12:
                        break
                    amt = min(float(ws_gap[idx]), ldes_pr, avail)
                    if amt > 0:
                        ldes_dispatch_hrs[ws + idx] = amt
                        soc -= amt / LDES_EFFICIENCY
                        soc = max(0.0, soc)

        # CO₂ from storage dispatch (hourly attribution at discharge hours' rates)
        storage_dispatch_total = batt_dispatch_hrs + ldes_dispatch_hrs
        co2_storage_dispatch = np.sum(storage_dispatch_total * hourly_rates) * demand_total_mwh

        # Charge-side emissions netting: charging draws energy that may have fossil
        # on the margin. Net these against discharge abatement.
        batt_charge_emissions = np.sum(batt_charge_hrs * hourly_rates) * demand_total_mwh
        ldes_charge_emissions = np.sum(ldes_charge_hrs * hourly_rates) * demand_total_mwh

        # Net storage abatement = dispatch abatement - charge emissions
        co2_storage = co2_storage_dispatch - batt_charge_emissions - ldes_charge_emissions

        storage_dispatch_mwh = float(np.sum(storage_dispatch_total)) * demand_total_mwh
        total_abated = co2_clean + co2_ccs + co2_storage
        matched_mwh_total = matched_mwh + storage_dispatch_mwh
        co2_rate = total_abated / matched_mwh_total if matched_mwh_total > 0 else 0
        weighted_avg_rate = np.average(hourly_rates, weights=fossil_displaced) \
            if np.sum(fossil_displaced) > 0 else np.mean(hourly_rates)

        return {
            'marginal_emission_rate_tons_per_mwh': round(float(weighted_avg_rate), 4),
            'hourly_emission_rate_avg': round(float(np.mean(hourly_rates)), 4),
            'hourly_emission_rate_min': round(float(np.min(hourly_rates)), 4),
            'hourly_emission_rate_max': round(float(np.max(hourly_rates)), 4),
            'total_co2_abated_tons': round(float(total_abated), 0),
            'co2_rate_per_mwh': round(float(co2_rate), 4),
            'co2_storage_net_tons': round(float(co2_storage), 0),
            'co2_storage_dispatch_tons': round(float(co2_storage_dispatch), 0),
            'co2_storage_charge_emissions_tons': round(float(batt_charge_emissions + ldes_charge_emissions), 0),
            'methodology': 'hourly_dispatch_attribution_with_charge_netting',
        }

    # ── Fallback: flat regional rate (backward compatibility) ──
    fossil_co2_lb_per_mwh = regional_data.get('fossil_co2_lb_per_mwh', 900.0)
    marginal_rate_tons = fossil_co2_lb_per_mwh / 2204.62

    procurement_factor = procurement_pct / 100.0
    matched_fraction = hourly_match_score / 100.0
    matched_mwh = demand_total_mwh * matched_fraction

    resource_co2 = {}
    total_abated = 0.0

    for rtype in RESOURCE_TYPES:
        pct = resource_pcts.get(rtype, 0)
        if pct <= 0:
            resource_co2[rtype] = {'mwh': 0, 'co2_abated_tons': 0}
            continue

        resource_mwh = matched_mwh * (pct / 100.0)

        if rtype == 'ccs_ccgt':
            abated = resource_mwh * (marginal_rate_tons - CCS_RESIDUAL_EMISSION_RATE)
        else:
            abated = resource_mwh * marginal_rate_tons

        abated = max(0, abated)
        resource_co2[rtype] = {
            'mwh': round(resource_mwh, 0),
            'co2_abated_tons': round(abated, 0),
        }
        total_abated += abated

    storage_mwh = demand_total_mwh * ((battery_dispatch_pct + ldes_dispatch_pct) / 100.0)
    storage_abated = storage_mwh * marginal_rate_tons
    resource_co2['battery'] = {
        'mwh': round(demand_total_mwh * battery_dispatch_pct / 100.0, 0),
        'co2_abated_tons': round(demand_total_mwh * battery_dispatch_pct / 100.0 * marginal_rate_tons, 0),
    }
    resource_co2['ldes'] = {
        'mwh': round(demand_total_mwh * ldes_dispatch_pct / 100.0, 0),
        'co2_abated_tons': round(demand_total_mwh * ldes_dispatch_pct / 100.0 * marginal_rate_tons, 0),
    }
    total_abated += storage_abated

    co2_rate = total_abated / matched_mwh if matched_mwh > 0 else 0

    return {
        'marginal_emission_rate_tons_per_mwh': round(marginal_rate_tons, 4),
        'total_co2_abated_tons': round(total_abated, 0),
        'co2_rate_per_mwh': round(co2_rate, 4),
        'resource_breakdown': resource_co2,
        'methodology': 'flat_regional_rate',
    }


# ══════════════════════════════════════════════════════════════════════════════
# COST COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_costs(iso, resource_pcts, procurement_pct, battery_dispatch_pct, ldes_dispatch_pct,
                  hourly_match_score, demand_norm, supply_profiles):
    """
    Compute blended cost of energy and incremental cost above baseline.

    Cost model:
      - Resources up to existing grid mix share -> priced at wholesale market rate
      - Resources above grid mix share -> priced at new-build LCOE + transmission adder
      - CCS-CCGT: no existing share, all new-build priced
      - Hydro: always at wholesale (existing resource, no new-build)
      - Battery storage -> priced at battery LCOS + tx adder for dispatched MWh
      - LDES storage -> priced at LDES LCOS + tx adder for dispatched MWh

    Returns dict with cost metrics in $/MWh (per MWh of demand served).
    """
    wholesale = WHOLESALE_PRICES[iso]
    lcoe = REGIONAL_LCOE[iso]
    tx = TRANSMISSION_ADDERS[iso]
    grid_shares = GRID_MIX_SHARES[iso]

    procurement_factor = procurement_pct / 100.0

    resource_costs = {}
    total_cost_per_demand = 0.0

    for rtype in RESOURCE_TYPES:
        pct = resource_pcts.get(rtype, 0)
        if pct <= 0:
            resource_costs[rtype] = {'existing_share': 0, 'new_share': 0, 'cost': 0}
            continue

        resource_fraction = procurement_factor * (pct / 100.0)
        resource_pct_of_demand = resource_fraction * 100.0

        existing_share = grid_shares.get(rtype, 0)
        existing_pct = min(resource_pct_of_demand, existing_share)
        new_pct = max(0, resource_pct_of_demand - existing_share)

        if rtype == 'hydro':
            # Hydro: always at wholesale, no new-build, no transmission adder
            cost_per_demand = resource_pct_of_demand / 100.0 * wholesale
        else:
            # Existing portion at wholesale, new-build at LCOE + transmission
            new_build_cost = lcoe.get(rtype, 0) + tx.get(rtype, 0)
            cost_per_demand = (existing_pct / 100.0 * wholesale) + (new_pct / 100.0 * new_build_cost)

        resource_costs[rtype] = {
            'total_pct_of_demand': round(resource_pct_of_demand, 1),
            'existing_pct': round(existing_pct, 1),
            'new_pct': round(new_pct, 1),
            'cost_per_demand_mwh': round(cost_per_demand, 2),
        }
        total_cost_per_demand += cost_per_demand

    # Battery storage cost: pay for dispatched MWh at battery LCOS + tx
    battery_cost_rate = lcoe['battery'] + tx.get('battery', 0)
    battery_cost_per_demand = (battery_dispatch_pct / 100.0) * battery_cost_rate
    resource_costs['battery'] = {
        'dispatch_pct': round(battery_dispatch_pct, 1),
        'cost_per_demand_mwh': round(battery_cost_per_demand, 2),
    }
    total_cost_per_demand += battery_cost_per_demand

    # LDES storage cost: pay for dispatched MWh at LDES LCOS + tx
    ldes_cost_rate = lcoe['ldes'] + tx.get('ldes', 0)
    ldes_cost_per_demand = (ldes_dispatch_pct / 100.0) * ldes_cost_rate
    resource_costs['ldes'] = {
        'dispatch_pct': round(ldes_dispatch_pct, 1),
        'cost_per_demand_mwh': round(ldes_cost_per_demand, 2),
    }
    total_cost_per_demand += ldes_cost_per_demand

    # Effective cost per useful MWh (accounting for curtailment)
    matched_fraction = hourly_match_score / 100.0 if hourly_match_score > 0 else 1.0
    effective_cost_per_useful_mwh = total_cost_per_demand / matched_fraction

    baseline_cost = wholesale
    incremental = effective_cost_per_useful_mwh - baseline_cost

    return {
        'resource_costs': resource_costs,
        'total_cost_per_demand_mwh': round(total_cost_per_demand, 2),
        'effective_cost_per_useful_mwh': round(effective_cost_per_useful_mwh, 2),
        'baseline_wholesale_cost': wholesale,
        'incremental_above_baseline': round(incremental, 2),
        'curtailment_pct': round((procurement_factor - matched_fraction) / procurement_factor * 100, 1)
            if procurement_factor > 0 else 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERIZED COST COMPUTATION (for paired toggle sensitivity scenarios)
# ══════════════════════════════════════════════════════════════════════════════

# Wholesale price adjustments by region and fossil fuel price level
# Based on regional fossil fuel generation share and fuel price sensitivity
# Gas heat rate ~7 MMBtu/MWh × price delta, weighted by regional fossil share
WHOLESALE_FUEL_ADJUSTMENTS = {
    'CAISO': {'Low': -5, 'Medium': 0, 'High': 10},   # ~40% gas generation
    'ERCOT': {'Low': -7, 'Medium': 0, 'High': 12},   # ~50% gas, most sensitive
    'PJM':   {'Low': -6, 'Medium': 0, 'High': 11},   # ~40% gas + coal mix
    'NYISO': {'Low': -4, 'Medium': 0, 'High': 8},    # ~35% gas, more nuclear
    'NEISO': {'Low': -4, 'Medium': 0, 'High': 8},    # ~35% gas, more nuclear
}


def compute_costs_parameterized(iso, resource_pcts, procurement_pct, battery_dispatch_pct,
                                 ldes_dispatch_pct, hourly_match_score,
                                 renewable_gen_level, firm_gen_level, storage_level,
                                 fossil_fuel_level, transmission_level):
    """
    Compute costs for a specific paired toggle scenario on a cached resource mix.

    Instead of reading from global Medium constants, uses the full L/M/H cost tables
    mapped through the paired toggle groups.
    """
    # Map paired toggle levels to individual resource LCOEs
    lcoe_map = {
        'solar': FULL_LCOE_TABLES['solar'][renewable_gen_level][iso],
        'wind': FULL_LCOE_TABLES['wind'][renewable_gen_level][iso],
        'clean_firm': FULL_LCOE_TABLES['clean_firm'][firm_gen_level][iso],
        'ccs_ccgt': FULL_LCOE_TABLES['ccs_ccgt'][firm_gen_level][iso],
        'battery': FULL_LCOE_TABLES['battery'][storage_level][iso],
        'ldes': FULL_LCOE_TABLES['ldes'][storage_level][iso],
        'hydro': 0,
    }

    # Map transmission level to per-resource adders
    tx_map = {rtype: FULL_TRANSMISSION_TABLES[rtype][transmission_level][iso]
              for rtype in FULL_TRANSMISSION_TABLES}

    # Adjust wholesale price based on fossil fuel level
    wholesale = WHOLESALE_PRICES[iso] + WHOLESALE_FUEL_ADJUSTMENTS[iso][fossil_fuel_level]
    wholesale = max(5, wholesale)  # Floor at $5/MWh

    grid_shares = GRID_MIX_SHARES[iso]
    procurement_factor = procurement_pct / 100.0

    total_cost_per_demand = 0.0

    for rtype in RESOURCE_TYPES:
        pct = resource_pcts.get(rtype, 0)
        if pct <= 0:
            continue

        resource_fraction = procurement_factor * (pct / 100.0)
        resource_pct_of_demand = resource_fraction * 100.0

        existing_share = grid_shares.get(rtype, 0)
        existing_pct = min(resource_pct_of_demand, existing_share)
        new_pct = max(0, resource_pct_of_demand - existing_share)

        if rtype == 'hydro':
            cost_per_demand = resource_pct_of_demand / 100.0 * wholesale
        else:
            new_build_cost = lcoe_map.get(rtype, 0) + tx_map.get(rtype, 0)
            cost_per_demand = (existing_pct / 100.0 * wholesale) + (new_pct / 100.0 * new_build_cost)

        total_cost_per_demand += cost_per_demand

    # Battery storage cost
    battery_cost_rate = lcoe_map['battery'] + tx_map.get('battery', 0)
    total_cost_per_demand += (battery_dispatch_pct / 100.0) * battery_cost_rate

    # LDES storage cost
    ldes_cost_rate = lcoe_map['ldes'] + tx_map.get('ldes', 0)
    total_cost_per_demand += (ldes_dispatch_pct / 100.0) * ldes_cost_rate

    # Effective cost per useful MWh
    matched_fraction = hourly_match_score / 100.0 if hourly_match_score > 0 else 1.0
    effective_cost = total_cost_per_demand / matched_fraction
    incremental = effective_cost - wholesale

    return {
        'effective_cost': round(effective_cost, 2),
        'incremental': round(incremental, 2),
        'total_cost': round(total_cost_per_demand, 2),
        'wholesale': round(wholesale, 2),
    }


def precompute_sensitivity_costs(iso, result, emission_rates, demand_total_mwh):
    """
    Pre-compute costs for all 324 paired toggle combinations on a cached resource mix.

    Returns dict keyed by scenario string: "{renew}_{firm}_{storage}_{fuel}_{tx}"
    where each level is L/M/H (or N for None on transmission).
    """
    resource_pcts = result['resource_mix']
    procurement_pct = result['procurement_pct']
    battery_pct = result['battery_dispatch_pct']
    ldes_pct = result['ldes_dispatch_pct']
    match_score = result['hourly_match_score']

    level_keys = {'Low': 'L', 'Medium': 'M', 'High': 'H', 'None': 'N'}
    gen_levels = ['Low', 'Medium', 'High']
    tx_levels = ['None', 'Low', 'Medium', 'High']

    scenarios = {}

    for renew in gen_levels:
        for firm in gen_levels:
            for storage in gen_levels:
                for fuel in gen_levels:
                    for tx in tx_levels:
                        key = f"{level_keys[renew]}{level_keys[firm]}{level_keys[storage]}_{level_keys[fuel]}_{level_keys[tx]}"
                        cost_data = compute_costs_parameterized(
                            iso, resource_pcts, procurement_pct,
                            battery_pct, ldes_pct, match_score,
                            renew, firm, storage, fuel, tx
                        )
                        scenarios[key] = cost_data

    return scenarios


# ══════════════════════════════════════════════════════════════════════════════
# PEAK GAP COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_peak_gap(demand_norm, supply_profiles, resource_pcts, procurement_pct,
                     battery_dispatch_pct, ldes_dispatch_pct, anomaly_hours):
    """Compute peak single-hour gap % after both storage types."""
    (score, hourly_detail,
     batt_dispatch, batt_charge,
     ldes_dispatch, ldes_charge) = compute_hourly_matching_detailed(
        demand_norm, supply_profiles, resource_pcts, procurement_pct,
        battery_dispatch_pct, ldes_dispatch_pct
    )
    peak_gap = 0.0
    for h in range(H):
        if h in anomaly_hours:
            continue
        d = hourly_detail[h]
        disp = batt_dispatch[h] + ldes_dispatch[h]
        residual_gap = max(0, d['gap'] - disp)
        if d['demand'] > 0:
            gap_pct = (residual_gap / d['demand']) * 100
            if gap_pct > peak_gap:
                peak_gap = gap_pct
    return round(peak_gap, 1)


# ══════════════════════════════════════════════════════════════════════════════
# COMPRESSED DAY PROFILE
# ══════════════════════════════════════════════════════════════════════════════

def compute_compressed_day(demand_norm, supply_profiles, resource_pcts, procurement_pct,
                           battery_dispatch_pct, ldes_dispatch_pct):
    """Build compressed day profile for visualization with all 7 resource types."""
    procurement_factor = procurement_pct / 100.0

    (_, hourly_detail,
     batt_dispatch_profile, batt_charge_profile,
     ldes_dispatch_profile, ldes_charge_profile) = compute_hourly_matching_detailed(
        demand_norm, supply_profiles, resource_pcts, procurement_pct,
        battery_dispatch_pct, ldes_dispatch_pct
    )

    demand_sums = [0.0] * 24
    supply_by_type = {rt: [0.0]*24 for rt in RESOURCE_TYPES}
    gap_sums = [0.0] * 24
    surplus_sums = [0.0] * 24
    batt_dispatch_sums = [0.0] * 24
    batt_charge_sums = [0.0] * 24
    ldes_dispatch_sums = [0.0] * 24
    ldes_charge_sums = [0.0] * 24

    for h in range(H):
        hod = h % 24
        d = hourly_detail[h]
        demand_sums[hod] += d['demand']

        for rtype, pct in resource_pcts.items():
            if pct <= 0:
                continue
            type_supply = procurement_factor * (pct / 100.0) * supply_profiles[rtype][h]
            supply_by_type[rtype][hod] += type_supply

        batt_dispatch_sums[hod] += batt_dispatch_profile[h]
        batt_charge_sums[hod] += batt_charge_profile[h]
        ldes_dispatch_sums[hod] += ldes_dispatch_profile[h]
        ldes_charge_sums[hod] += ldes_charge_profile[h]

    # Match supply to demand per hour-of-day
    # Match order: clean_firm, ccs_ccgt, hydro, wind, solar (baseload first, then variable)
    match_order = ['clean_firm', 'ccs_ccgt', 'hydro', 'wind', 'solar']
    cut_order = list(reversed(match_order))  # solar curtailed first

    matched_by_type = {rt: [0.0]*24 for rt in RESOURCE_TYPES}
    surplus_by_type = {rt: [0.0]*24 for rt in RESOURCE_TYPES}
    matched_by_type['battery'] = [0.0]*24
    matched_by_type['ldes'] = [0.0]*24

    for hod in range(24):
        remaining = demand_sums[hod]

        # Match generation resources
        for rtype in match_order:
            avail = supply_by_type[rtype][hod]
            matched = min(remaining, avail)
            matched_by_type[rtype][hod] = matched
            surplus_by_type[rtype][hod] = avail - matched
            remaining -= matched

        # Battery dispatch fills remaining gap
        batt_disp = min(remaining, batt_dispatch_sums[hod])
        matched_by_type['battery'][hod] = batt_disp
        remaining -= batt_disp

        # LDES dispatch fills further remaining gap
        ldes_disp = min(remaining, ldes_dispatch_sums[hod])
        matched_by_type['ldes'][hod] = ldes_disp
        remaining -= ldes_disp

        gap_sums[hod] = max(0, remaining)

        # Reduce surplus by battery charging
        rem_charge = batt_charge_sums[hod]
        for rtype in cut_order:
            if rem_charge <= 0:
                break
            absorb = min(surplus_by_type[rtype][hod], rem_charge)
            surplus_by_type[rtype][hod] -= absorb
            rem_charge -= absorb

        # Reduce surplus by LDES charging
        rem_charge = ldes_charge_sums[hod]
        for rtype in cut_order:
            if rem_charge <= 0:
                break
            absorb = min(surplus_by_type[rtype][hod], rem_charge)
            surplus_by_type[rtype][hod] -= absorb
            rem_charge -= absorb

    # Total surplus per hour-of-day
    for hod in range(24):
        surplus_sums[hod] = sum(surplus_by_type[rt][hod] for rt in RESOURCE_TYPES)

    all_match_types = list(RESOURCE_TYPES) + ['battery', 'ldes']

    return {
        'demand': [round(v, 4) for v in demand_sums],
        'matched': {rt: [round(v, 4) for v in matched_by_type[rt]] for rt in all_match_types},
        'surplus': {rt: [round(v, 4) for v in surplus_by_type[rt]] for rt in RESOURCE_TYPES},
        'gap': [round(v, 4) for v in gap_sums],
        'battery_charge': [round(v, 4) for v in batt_charge_sums],
        'ldes_charge': [round(v, 4) for v in ldes_charge_sums],
        'total_surplus': [round(v, 4) for v in surplus_sums],
    }


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZATION AT PROCUREMENT LEVEL (for sweep chart)
# ══════════════════════════════════════════════════════════════════════════════

def optimize_at_procurement_level(iso, demand_norm, supply_profiles, procurement_pct, hydro_cap,
                                  np_profiles=None):
    """
    Find the resource mix that maximizes hourly matching at a given procurement level.
    Two-phase: coarse (5%) -> fine (1%), then optimize both storage types.
    """
    pf = procurement_pct / 100.0

    if np_profiles:
        demand_arr, _, supply_matrix = np_profiles

        # Coarse search with 10% step for 5D
        combos = generate_combinations(hydro_cap, step=10)
        best_score = -1
        best_combo = None

        for combo in combos:
            mix_fracs = np.array([combo[rt] / 100.0 for rt in RESOURCE_TYPES])
            score = fast_hourly_score(demand_arr, supply_matrix, mix_fracs, pf)
            if score > best_score:
                best_score = score
                best_combo = combo

        # Medium refinement (5% step, radius 10)
        if best_combo:
            combos_med = generate_combinations_around(best_combo, hydro_cap, step=5, radius=2)
            for combo in combos_med:
                mix_fracs = np.array([combo[rt] / 100.0 for rt in RESOURCE_TYPES])
                score = fast_hourly_score(demand_arr, supply_matrix, mix_fracs, pf)
                if score > best_score:
                    best_score = score
                    best_combo = combo

        # Fine refinement (1% step, radius 3)
        if best_combo:
            combos_fine = generate_combinations_around(best_combo, hydro_cap, step=1, radius=3)
            for combo in combos_fine:
                mix_fracs = np.array([combo[rt] / 100.0 for rt in RESOURCE_TYPES])
                score = fast_hourly_score(demand_arr, supply_matrix, mix_fracs, pf)
                if score > best_score:
                    best_score = score
                    best_combo = combo

        # Optimize storage
        mix_fracs = np.array([best_combo[rt] / 100.0 for rt in RESOURCE_TYPES])
        best_score_ws, best_batt, best_ldes = find_optimal_storage(
            demand_arr, supply_matrix, mix_fracs, pf, hydro_cap
        )
    else:
        combos = generate_combinations(hydro_cap, step=10)
        best_score = -1
        best_combo = None

        demand_arr = np.array(demand_norm[:H], dtype=np.float64)
        supply_matrix = np.stack([np.array(supply_profiles[rt][:H], dtype=np.float64) for rt in RESOURCE_TYPES])

        for combo in combos:
            mix_fracs = np.array([combo[rt] / 100.0 for rt in RESOURCE_TYPES])
            score = fast_hourly_score(demand_arr, supply_matrix, mix_fracs, pf)
            if score > best_score:
                best_score = score
                best_combo = combo

        if best_combo:
            combos_fine = generate_combinations_around(best_combo, hydro_cap, step=1, radius=3)
            for combo in combos_fine:
                mix_fracs = np.array([combo[rt] / 100.0 for rt in RESOURCE_TYPES])
                score = fast_hourly_score(demand_arr, supply_matrix, mix_fracs, pf)
                if score > best_score:
                    best_score = score
                    best_combo = combo

        mix_fracs = np.array([best_combo[rt] / 100.0 for rt in RESOURCE_TYPES])
        best_score_ws, best_batt, best_ldes = find_optimal_storage(
            demand_arr, supply_matrix, mix_fracs, pf, hydro_cap
        )

    return {
        'procurement_pct': procurement_pct,
        'resource_mix': best_combo,
        'battery_dispatch_pct': round(best_batt, 1),
        'ldes_dispatch_pct': round(best_ldes, 1),
        'hourly_match_score': round(best_score_ws * 100, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# THRESHOLD OPTIMIZATION (3-phase: coarse -> medium -> fine)
# ══════════════════════════════════════════════════════════════════════════════

def optimize_for_threshold(iso, demand_norm, supply_profiles, threshold, hydro_cap,
                           emission_rates, demand_total_mwh,
                           cost_levels=None, score_cache=None,
                           resweep=False, seed_mixes=None,
                           procurement_bounds_override=None,
                           warm_start_result=None,
                           min_dispatchable=0):
    """
    CO-OPTIMIZE cost and matching simultaneously for a given threshold target.
    Search across procurement levels AND resource mixes to find the CHEAPEST
    combination that meets or exceeds the threshold.

    3-phase approach adapted for 5D resource space:
      Phase 1: Coarse scan (10% mix steps, 10% procurement steps)
      Phase 2: Medium refinement (5% mix steps, 5% procurement steps)
      Phase 3: Fine-tune (1% mix, 2% procurement, refined storage)

    When warm_start_result is provided, Phase 1 is replaced with a targeted
    evaluation of the warm-start mix + edge-case seeds at all procurement levels.
    This preserves the co-optimization guarantee (Phase 2/3 still search the full
    neighborhood) while eliminating the expensive coarse grid scan. The warm-start
    mix acts as a high-quality initial guess — the refinement phases will find any
    nearby solution that's cheaper under this scenario's cost function.

    Args:
        cost_levels: Optional tuple (renewable_gen_level, firm_gen_level, storage_level,
                     fossil_fuel_level, transmission_level). If None, uses Medium costs.
        score_cache: Optional dict for caching matching scores across cost scenarios.
                     Matching scores are physics-based and cost-independent, so caching
                     them across cost scenarios is scientifically correct (not a shortcut).
        resweep: If True, use broader search parameters (finer Phase 1 step, more
                 Phase 2/3 candidates). Triggered by monotonicity violations.
        seed_mixes: Optional list of resource mix dicts to inject as seeds in Phase 1.
                    Used during re-sweep to seed with mixes that achieved better cost
                    at a higher threshold.
        procurement_bounds_override: Optional (min, max) tuple to override default
                                     procurement bounds. Used during re-sweep to expand
                                     the search range.
        warm_start_result: Optional dict from a prior scenario's optimization (e.g.,
                          Medium). When provided, replaces Phase 1 coarse grid with
                          targeted evaluation of the warm-start mix + edge-case seeds.
                          Phase 2/3 refinement still runs fully to find the cost-optimal
                          solution under this scenario's cost function.

    Returns: best_result dict or None
    """
    target = threshold / 100.0
    # Adaptive procurement bounds per threshold (use override during re-sweep)
    if procurement_bounds_override:
        proc_min, proc_max = procurement_bounds_override
    else:
        proc_min, proc_max = PROCUREMENT_BOUNDS.get(threshold, (70, 310))
    storage_threshold = max(0.5, target - 0.10)  # Wider net to catch storage-dependent optima
    demand_arr, supply_arrs, supply_matrix = prepare_numpy_profiles(demand_norm, supply_profiles)

    if score_cache is None:
        score_cache = {}

    # ---- Cached scoring wrappers ----
    # Matching scores are physics (hourly profile alignment) — cost-independent.
    # Caching them across cost scenarios avoids redundant computation without
    # compromising scientific rigor.

    def c_hourly_score(mix_fracs_tuple, pf):
        key = ('h', mix_fracs_tuple, round(pf, 4))
        if key not in score_cache:
            score_cache[key] = fast_hourly_score(demand_arr, supply_matrix,
                                                  np.array(mix_fracs_tuple), pf)
        return score_cache[key]

    def c_battery_score(mix_fracs_tuple, pf, bp):
        key = ('b', mix_fracs_tuple, round(pf, 4), bp)
        if key not in score_cache:
            score_cache[key] = fast_score_with_battery(demand_arr, supply_matrix,
                                                        np.array(mix_fracs_tuple), pf, bp)
        return score_cache[key]

    def c_both_score(mix_fracs_tuple, pf, bp, lp):
        key = ('bl', mix_fracs_tuple, round(pf, 4), bp, lp)
        if key not in score_cache:
            score_cache[key] = fast_score_with_both_storage(demand_arr, supply_matrix,
                                                              np.array(mix_fracs_tuple), pf, bp, lp)
        return score_cache[key]

    # ---- Cost function: uses scenario-specific costs ----
    best_result = None
    best_cost = float('inf')

    def eval_cost(combo, proc, bp, lp, score):
        """Evaluate cost using the scenario's cost function. Cost drives mix selection."""
        if cost_levels:
            r_gen, f_gen, stor, fuel, tx = cost_levels
            cost_data = compute_costs_parameterized(
                iso, combo, proc, bp, lp, score * 100,
                r_gen, f_gen, stor, fuel, tx
            )
            return cost_data['effective_cost']
        else:
            cost_data = compute_costs(iso, combo, proc, bp, lp, score * 100,
                                       demand_norm, supply_profiles)
            return cost_data['effective_cost_per_useful_mwh']

    def update_best(combo, proc, bp, lp, score):
        """Update best result if this is the cheapest so far under current cost scenario."""
        nonlocal best_result, best_cost
        cost = eval_cost(combo, proc, bp, lp, score)
        if cost < best_cost:
            best_cost = cost
            best_result = {
                'procurement_pct': proc,
                'resource_mix': dict(combo),
                'battery_dispatch_pct': round(bp, 1),
                'ldes_dispatch_pct': round(lp, 1),
                'hourly_match_score': round(score * 100, 1),
            }
        return cost

    # ---- Phase 1: Coarse scan (or warm-start shortcut) ----
    candidates = []

    if warm_start_result and not resweep:
        # WARM-START MODE: Skip full coarse grid. Instead, evaluate the warm-start
        # mix + edge-case seeds at all procurement levels. This is scientifically
        # valid because:
        #   1. Matching scores are physics-based (cost-independent) — same mix at
        #      same procurement produces the same hourly match regardless of costs.
        #   2. Phase 2/3 still searches the full 5% and 1% neighborhoods around
        #      the best candidates, so any mix that differs from warm-start under
        #      this cost function will be discovered during refinement.
        #   3. Edge-case seeds ensure extreme mixes (100% solar, 100% wind, etc.)
        #      are always evaluated, catching cases where cost assumptions make
        #      a radically different mix optimal.
        ws_mix = warm_start_result['resource_mix']
        ws_proc = warm_start_result['procurement_pct']
        ws_bp = warm_start_result.get('battery_dispatch_pct', 0)
        ws_lp = warm_start_result.get('ldes_dispatch_pct', 0)

        # Build targeted combo list: warm-start mix + 5% neighborhood + edge seeds
        ws_combos = [dict(ws_mix)]
        ws_neighborhood = generate_combinations_around(ws_mix, hydro_cap, step=5, radius=2, min_dispatchable=min_dispatchable)
        existing_set = set()
        existing_set.add(tuple(ws_mix[rt] for rt in RESOURCE_TYPES))
        for nc in ws_neighborhood:
            key = tuple(nc[rt] for rt in RESOURCE_TYPES)
            if key not in existing_set:
                existing_set.add(key)
                ws_combos.append(nc)
        # Always include edge-case seeds
        seeds = get_seed_combos(hydro_cap, min_dispatchable=min_dispatchable)
        for seed in seeds:
            key = tuple(seed[rt] for rt in RESOURCE_TYPES)
            if key not in existing_set:
                existing_set.add(key)
                ws_combos.append(seed)
        # Include any explicit seed_mixes
        if seed_mixes:
            for smix in seed_mixes:
                key = tuple(smix[rt] for rt in RESOURCE_TYPES)
                if key not in existing_set:
                    existing_set.add(key)
                    ws_combos.append(dict(smix))

        for procurement_pct in range(proc_min, proc_max + 1, 5):
            pf = procurement_pct / 100.0
            for combo in ws_combos:
                mix_fracs = tuple(combo[rt] / 100.0 for rt in RESOURCE_TYPES)
                score = c_hourly_score(mix_fracs, pf)

                if score >= target:
                    cost = update_best(combo, procurement_pct, 0, 0, score)
                    candidates.append((cost, combo, score, 0, 0, procurement_pct))
                elif score >= storage_threshold:
                    for bp in [5, 10, 15, 20]:
                        score_ws = c_battery_score(mix_fracs, pf, bp)
                        if score_ws >= target:
                            cost = update_best(combo, procurement_pct, bp, 0, score_ws)
                            candidates.append((cost, combo, score_ws, bp, 0, procurement_pct))
                            break
                    for lp in [5, 10, 15, 20]:
                        score_ws = c_both_score(mix_fracs, pf, 0, lp)
                        if score_ws >= target:
                            cost = update_best(combo, procurement_pct, 0, lp, score_ws)
                            candidates.append((cost, combo, score_ws, 0, lp, procurement_pct))
                            break
                    for bp in [5, 10, 15, 20]:
                        for lp in [5, 10, 15, 20]:
                            score_ws = c_both_score(mix_fracs, pf, bp, lp)
                            if score_ws >= target:
                                cost = update_best(combo, procurement_pct, bp, lp, score_ws)
                                candidates.append((cost, combo, score_ws, bp, lp, procurement_pct))
                                break
                        else:
                            continue
                        break

        if not candidates:
            # Warm-start failed to find any feasible solution — fall back to full Phase 1
            warm_start_result = None  # Clear so we don't re-enter this branch

    if not candidates:
        # FULL PHASE 1: Coarse grid scan (used for Medium scenario, resweep, or warm-start fallback)
        phase1_step = 5 if resweep else 10
        combos_10 = generate_combinations(hydro_cap, step=phase1_step, min_dispatchable=min_dispatchable)
        # Inject edge case seeds to guarantee extreme mixes survive pruning
        seeds = get_seed_combos(hydro_cap, min_dispatchable=min_dispatchable)
        seed_set = set(tuple(s[rt] for rt in RESOURCE_TYPES) for s in seeds)
        existing_set = set(tuple(c[rt] for rt in RESOURCE_TYPES) for c in combos_10)
        for seed in seeds:
            if tuple(seed[rt] for rt in RESOURCE_TYPES) not in existing_set:
                combos_10.append(seed)
        # Inject re-sweep seed mixes (from higher thresholds that achieved better cost)
        if seed_mixes:
            for smix in seed_mixes:
                key = tuple(smix[rt] for rt in RESOURCE_TYPES)
                if key not in existing_set:
                    existing_set.add(key)
                    combos_10.append(dict(smix))

        for procurement_pct in range(proc_min, proc_max + 1, 10):
            pf = procurement_pct / 100.0
            for combo in combos_10:
                mix_fracs = tuple(combo[rt] / 100.0 for rt in RESOURCE_TYPES)
                score = c_hourly_score(mix_fracs, pf)

                if score >= target:
                    cost = update_best(combo, procurement_pct, 0, 0, score)
                    candidates.append((cost, combo, score, 0, 0, procurement_pct))
                elif score >= storage_threshold:
                    # Phase 1 uses coarse storage grid (2×2) — Phase 2/3 refine
                    # Battery only
                    for bp in [10, 20]:
                        score_ws = c_battery_score(mix_fracs, pf, bp)
                        if score_ws >= target:
                            cost = update_best(combo, procurement_pct, bp, 0, score_ws)
                            candidates.append((cost, combo, score_ws, bp, 0, procurement_pct))
                            break
                    # LDES only (wind-heavy mixes benefit from multi-day shifting)
                    for lp in [10, 20]:
                        score_ws = c_both_score(mix_fracs, pf, 0, lp)
                        if score_ws >= target:
                            cost = update_best(combo, procurement_pct, 0, lp, score_ws)
                            candidates.append((cost, combo, score_ws, 0, lp, procurement_pct))
                            break
                    # Combined battery + LDES
                    for bp in [10, 20]:
                        for lp in [10, 20]:
                            score_ws = c_both_score(mix_fracs, pf, bp, lp)
                            if score_ws >= target:
                                cost = update_best(combo, procurement_pct, bp, lp, score_ws)
                                candidates.append((cost, combo, score_ws, bp, lp, procurement_pct))
                                break
                        else:
                            continue
                        break

    if not candidates:
        return None

    # Select top candidates for refinement — ranked by THIS scenario's cost function
    # Re-sweep uses wider cost filter and more candidates for broader exploration
    phase2_cost_mult = 2.00 if resweep else 1.50
    phase2_top_n = 30 if resweep else 10
    candidates.sort(key=lambda x: x[0])
    top = [c for c in candidates if c[0] <= best_cost * phase2_cost_mult][:phase2_top_n]

    # ---- Phase 2: 5% refinement around top candidates ----
    # Early termination: if best_cost hasn't improved after evaluating 3 full
    # candidate neighborhoods, remaining candidates are unlikely to help.
    phase2 = []
    seen = set()
    p2_candidates_since_improvement = 0
    P2_STALE_LIMIT = 4 if not resweep else 8  # More patience during resweep
    p2_cost_before = best_cost
    for _, combo, _, bp_base, lp_base, proc in top:
        if p2_candidates_since_improvement >= P2_STALE_LIMIT:
            break
        candidate_improved = False
        for p_d in [-5, 0, 5]:
            p = proc + p_d
            if p < proc_min or p > proc_max:
                continue
            pf = p / 100.0

            neighborhood = generate_combinations_around(combo, hydro_cap, step=5, radius=1, min_dispatchable=min_dispatchable)
            for rcombo in neighborhood:
                key = (rcombo['clean_firm'], rcombo['solar'], rcombo['wind'],
                       rcombo['ccs_ccgt'], rcombo['hydro'], p)
                if key in seen:
                    continue
                seen.add(key)

                mix_fracs = tuple(rcombo[rt] / 100.0 for rt in RESOURCE_TYPES)

                score = c_hourly_score(mix_fracs, pf)
                best_bp = 0
                best_lp = 0

                if score < target:
                    for bp in [5, 10, 15, 20, 25]:
                        score_ws = c_battery_score(mix_fracs, pf, bp)
                        if score_ws >= target:
                            score = score_ws
                            best_bp = bp
                            break

                if score < target:
                    # Try LDES only
                    for lp in [5, 10, 15, 20]:
                        score_ws = c_both_score(mix_fracs, pf, 0, lp)
                        if score_ws >= target:
                            score = score_ws
                            best_bp = 0
                            best_lp = lp
                            break

                if score < target:
                    for bp in [5, 10, 15, 20]:
                        for lp in [5, 10, 15, 20]:
                            score_ws = c_both_score(mix_fracs, pf, bp, lp)
                            if score_ws >= target:
                                score = score_ws
                                best_bp = bp
                                best_lp = lp
                                break
                        else:
                            continue
                        break

                if score >= target:
                    cost = update_best(rcombo, p, best_bp, best_lp, score)
                    phase2.append((cost, rcombo, score, best_bp, best_lp, p))
                    if best_cost < p2_cost_before:
                        candidate_improved = True
                        p2_cost_before = best_cost

        if candidate_improved:
            p2_candidates_since_improvement = 0
        else:
            p2_candidates_since_improvement += 1

    # ---- Phase 3: Fine-tune (1% mix, 2% procurement, refined storage) ----
    # Re-sweep uses wider filter and more finalists to avoid pruning the true optimum
    phase3_cost_mult = 1.20 if resweep else 1.10
    phase3_top_n = 15 if resweep else 8
    all_phase2 = phase2 if phase2 else top
    all_phase2.sort(key=lambda x: x[0])
    finalists = [c for c in all_phase2 if c[0] <= best_cost * phase3_cost_mult][:phase3_top_n]

    seen2 = set()
    phase3_radius = 2 if resweep else 1
    p3_finalists_since_improvement = 0
    P3_STALE_LIMIT = 3 if not resweep else 6
    p3_cost_before = best_cost
    for _, combo, _, bp_base, lp_base, proc in finalists:
        if p3_finalists_since_improvement >= P3_STALE_LIMIT:
            break
        finalist_improved = False
        for p_d in range(-2, 3):
            p = proc + p_d
            if p < proc_min or p > proc_max:
                continue
            pf = p / 100.0

            fine_combos = generate_combinations_around(combo, hydro_cap, step=1, radius=phase3_radius, min_dispatchable=min_dispatchable)
            for rcombo in fine_combos:
                key = (rcombo['clean_firm'], rcombo['solar'], rcombo['wind'],
                       rcombo['ccs_ccgt'], rcombo['hydro'], p)
                if key in seen2:
                    continue
                seen2.add(key)

                mix_fracs = tuple(rcombo[rt] / 100.0 for rt in RESOURCE_TYPES)

                score = c_hourly_score(mix_fracs, pf)
                best_bp = 0
                best_lp = 0
                best_score_here = score

                for bp in range(2, 22, 2):
                    score_ws = c_battery_score(mix_fracs, pf, bp)
                    if score_ws > best_score_here:
                        best_score_here = score_ws
                        best_bp = bp
                    if score_ws >= target and best_bp == bp:
                        break

                if best_score_here < target:
                    for lp in range(2, 22, 2):
                        score_ws = c_both_score(mix_fracs, pf, bp, lp)
                        if score_ws > best_score_here:
                            best_score_here = score_ws
                            best_lp = lp
                        if score_ws >= target and best_lp == lp:
                            break
                elif best_score_here >= target:
                    for lp in range(2, 12, 2):
                        score_ws = c_both_score(mix_fracs, pf, best_bp, lp)
                        if score_ws > best_score_here:
                            best_score_here = score_ws
                            best_lp = lp

                if best_score_here >= target:
                    update_best(rcombo, p, best_bp, best_lp, best_score_here)
                    if best_cost < p3_cost_before:
                        finalist_improved = True
                        p3_cost_before = best_cost

        if finalist_improved:
            p3_finalists_since_improvement = 0
        else:
            p3_finalists_since_improvement += 1

    return best_result


# ══════════════════════════════════════════════════════════════════════════════
# PER-ISO WORKER (for multiprocessing)
# ══════════════════════════════════════════════════════════════════════════════

def generate_all_cost_scenarios():
    """
    Generate all 324 paired toggle cost scenarios.
    Returns list of (scenario_key, cost_levels_tuple) pairs.
    """
    level_keys = {'Low': 'L', 'Medium': 'M', 'High': 'H', 'None': 'N'}
    gen_levels = ['Low', 'Medium', 'High']
    tx_levels = ['None', 'Low', 'Medium', 'High']

    scenarios = []
    for renew in gen_levels:
        for firm in gen_levels:
            for storage in gen_levels:
                for fuel in gen_levels:
                    for tx in tx_levels:
                        key = f"{level_keys[renew]}{level_keys[firm]}{level_keys[storage]}_{level_keys[fuel]}_{level_keys[tx]}"
                        levels = (renew, firm, storage, fuel, tx)
                        scenarios.append((key, levels))
    return scenarios


ALL_COST_SCENARIOS = generate_all_cost_scenarios()
# Dict for O(1) lookup of cost_levels by scenario key (used during re-sweep)
COST_SCENARIO_MAP = {key: levels for key, levels in ALL_COST_SCENARIOS}


# ── Scenario pruning for lower thresholds ──
# At 75-92.5%, physics dominates cost — only ~14 unique mixes serve 324 scenarios.
# Run 54 representative scenarios (corners + stratified) to discover all archetypes,
# then cross-pollinate to fill in the remaining 270 scenarios.
# At 95-100%, run full 324 for academic precision in the cost-sensitive zone.
PRUNING_THRESHOLD_CUTOFF = 100  # Prune all thresholds — empirically, 44 reps discover all archetypes

def _build_representative_scenarios():
    """Select ~30 representative scenarios covering cost space extremes.

    Reduced from 44 by removing:
    - Single-vary Medium scenarios (close to MMM_M_M, rarely discover unique archetypes)
    - Redundant cross-axis Tx variants (corners + Tx group already cover those axes)
    - Diagonals that overlap with corners
    Cross-pollination fills the remaining ~294 scenarios from discovered archetypes.
    """
    reps = set()
    lk = {'Low': 'L', 'Medium': 'M', 'High': 'H', 'None': 'N'}

    # 1. All 16 corners (L/H for ren/firm/stor × L/H fuel × Med Tx)
    #    These are the most important — they span the full cost hypercube
    for r in ['Low', 'High']:
        for f in ['Low', 'High']:
            for s in ['Low', 'High']:
                for fu in ['Low', 'High']:
                    reps.add(f"{lk[r]}{lk[f]}{lk[s]}_{lk[fu]}_M")

    # 2. Tx levels at Medium everything else (N/L/M/H) = 4
    for tx in ['None', 'Low', 'Medium', 'High']:
        reps.add(f"MMM_M_{lk[tx]}")

    # 3. Key cross-axis extremes at Med Tx only (corners already have L/H combos)
    reps.add('HLL_L_M')  # High renewables only (already in corners)
    reps.add('LHL_L_M')  # High firm only (already in corners)
    reps.add('LLH_L_M')  # High storage only (already in corners)
    reps.add('LLL_H_M')  # High fuel only

    # 4. Extreme Tx combos with non-Medium cost axes (capture Tx interaction effects)
    reps.add('HLL_L_N')  # High VRE, no transmission
    reps.add('HLL_L_H')  # High VRE, high transmission
    reps.add('LHL_L_N')  # High firm, no transmission

    # 5. Key diagonals (non-redundant with corners)
    reps.add('HHH_H_H')  # All high
    reps.add('LLL_L_L')  # All low
    reps.add('HHH_H_N')  # All high, no Tx
    reps.add('LLL_L_N')  # All low, no Tx
    reps.add('LMH_M_H')  # Mixed: low ren, high storage, high Tx
    reps.add('MHL_H_N')  # Mixed: high firm, high fuel, no Tx

    return reps

REPRESENTATIVE_SCENARIOS = _build_representative_scenarios()


CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'checkpoints')

# Module-level state for incremental dashboard saves
_dashboard_config = None  # Set by main() before ISO loop
_output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard', 'overprocure_results.json')


def save_results_incremental(iso, iso_results):
    """Merge current ISO results into dashboard JSON, preserving other ISOs.

    Loads existing overprocure_results.json (if any), updates/inserts this ISO's
    data, and saves back. Old ISO results are kept until overridden by new data.
    """
    global _dashboard_config, _output_path

    if _dashboard_config is None:
        return  # Config not yet initialized — skip

    # Load existing results file (preserve other ISOs)
    existing = None
    if os.path.exists(_output_path):
        try:
            with open(_output_path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing = None

    if existing is None:
        existing = {'config': _dashboard_config, 'results': {}}

    # Always use latest config
    existing['config'] = _dashboard_config

    # Merge this ISO's results (overwrites previous data for this ISO)
    existing['results'][iso] = iso_results

    os.makedirs(os.path.dirname(_output_path), exist_ok=True)
    with open(_output_path, 'w') as f:
        json.dump(existing, f)

    n_thresholds = len(iso_results.get('thresholds', {}))
    print(f"    [dashboard] Updated {_output_path.split('/')[-1]}: {iso} "
          f"({n_thresholds}/10 thresholds) — "
          f"{os.path.getsize(_output_path)/1024:.0f} KB total "
          f"({len(existing['results'])}/{len(ISOS)} ISOs)")


def save_score_cache(iso, threshold, score_cache):
    """Persist score cache to disk for warm restart.

    Keys are tuples of (type, mix_fracs_tuple, pf, [bp], [lp]).
    Values are floats. Convert keys to JSON-safe strings.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    cache_path = os.path.join(CHECKPOINT_DIR, f'{iso}_cache_{threshold}.json')
    # Convert tuple keys to string keys for JSON
    serializable = {str(k): v for k, v in score_cache.items()}
    with open(cache_path, 'w') as f:
        json.dump(serializable, f)
    print(f"    [cache] Saved {iso} {threshold}% score cache: {len(score_cache)} entries "
          f"({os.path.getsize(cache_path)/1024/1024:.1f} MB)")


def load_score_cache(iso, threshold):
    """Load persisted score cache for warm restart.

    Returns dict with original tuple keys, or empty dict if not found.
    """
    cache_path = os.path.join(CHECKPOINT_DIR, f'{iso}_cache_{threshold}.json')
    if not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path) as f:
            data = json.load(f)
        # Convert string keys back to tuples using eval (safe: controlled data)
        cache = {}
        for k_str, v in data.items():
            cache[eval(k_str)] = v
        print(f"    [cache] Loaded {iso} {threshold}% score cache: {len(cache)} entries")
        return cache
    except Exception as e:
        print(f"    [cache] Failed to load {iso} {threshold}%: {e}")
        return {}


def save_checkpoint(iso, iso_results, phase='threshold', partial_threshold=None):
    """Save incremental checkpoint for an ISO's results.

    partial_threshold: optional dict with {threshold, scenarios} for mid-threshold saves.
    This allows resuming from within a threshold's 324-scenario loop.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f'{iso}_checkpoint.json')
    payload = {'iso': iso, 'phase': phase, 'results': iso_results}
    if partial_threshold:
        payload['partial_threshold'] = partial_threshold
    with open(ckpt_path, 'w') as f:
        json.dump(payload, f)
    print(f"    [checkpoint] Saved {iso} ({phase}) → {os.path.getsize(ckpt_path)/1024:.0f} KB")


def load_checkpoint(iso):
    """Load existing checkpoint for an ISO.

    Returns (iso_results, completed_thresholds, partial_threshold) or None.
    partial_threshold is a dict {threshold, scenarios} if a threshold was mid-progress.
    """
    ckpt_path = os.path.join(CHECKPOINT_DIR, f'{iso}_checkpoint.json')
    if not os.path.exists(ckpt_path):
        return None
    try:
        with open(ckpt_path) as f:
            data = json.load(f)
        iso_results = data['results']
        completed = set(iso_results.get('thresholds', {}).keys())
        phase = data.get('phase', 'unknown')
        partial = data.get('partial_threshold', None)
        partial_msg = ''
        if partial:
            partial_msg = f", partial {partial['threshold']}% ({len(partial['scenarios'])}/324 scenarios)"
        print(f"    [checkpoint] Resuming {iso} from {phase}: "
              f"{len(completed)} thresholds done{partial_msg}")
        return iso_results, completed, partial
    except Exception as e:
        print(f"    [checkpoint] Failed to load {iso}: {e}")
        return None


def clear_checkpoint(iso):
    """Remove checkpoint after ISO is fully complete."""
    ckpt_path = os.path.join(CHECKPOINT_DIR, f'{iso}_checkpoint.json')
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)


def process_iso(args):
    """
    Process a single ISO region. Called sequentially.

    For each threshold:
      1. Run full 3-phase co-optimization for all 324 cost scenarios
      2. Each scenario uses its own cost function → different costs produce different optimal mixes
      3. Matching cache shared across scenarios within a threshold (physics reuse only)
      4. Phase 2/3 refinement neighborhoods are cost-driven (different per scenario)
      5. Post-hoc monotonicity re-sweep: if cost(T_lower) > cost(T_higher), re-sweep
         T_lower with broader parameters (5% Phase 1 step, expanded procurement range,
         seed mixes from T_higher) to find the true optimum — up to 2 rounds

    Saves checkpoint after each threshold so progress is never lost.
    Resumes from checkpoint if one exists for this ISO.
    """
    # Unpack args — supports both old (4-tuple) and new (5-tuple) format
    if len(args) == 5:
        iso, demand_data, gen_profiles, emission_rates, fossil_mix = args
    else:
        iso, demand_data, gen_profiles, emission_rates = args
        fossil_mix = None

    print(f"\n{'='*70}")
    print(f"  {ISO_LABELS[iso]}")
    print(f"{'='*70}")

    demand_norm = demand_data[iso]['normalized'][:H]
    supply_profiles = get_supply_profiles(iso, gen_profiles)
    hydro_cap = HYDRO_CAPS[iso]
    anomaly_hours = find_anomaly_hours(iso, gen_profiles)
    np_profiles = prepare_numpy_profiles(demand_norm, supply_profiles)
    demand_total_mwh = demand_data[iso]['total_annual_mwh']

    # ── Check for existing checkpoint to resume from ──
    checkpoint = load_checkpoint(iso)
    completed_thresholds = set()
    partial_threshold_data = None
    if checkpoint:
        iso_results, completed_thresholds, partial_threshold_data = checkpoint
        # Re-derive sweep_results dict from stored sweep list (if any)
        if iso_results.get('sweep'):
            print(f"  Resuming from checkpoint: sweep done, "
                  f"{len(completed_thresholds)} thresholds complete")
        else:
            print(f"  Checkpoint found but no sweep — starting fresh")
            checkpoint = None

    if not checkpoint:
        iso_results = {
            'iso': iso,
            'label': ISO_LABELS[iso],
            'annual_demand_mwh': demand_total_mwh,
            'peak_demand_mw': demand_data[iso]['peak_mw'],
            'sweep': [],
            'thresholds': {},
        }

    # ---- SWEEP: max-matching runs for the sweep chart visualization ----
    if not iso_results.get('sweep'):
        print(f"  Sweep: max-matching at key procurement levels...")
        sweep_results = {}
        for proc_pct in list(range(70, 130, 10)) + list(range(140, 520, 20)):
            result = optimize_at_procurement_level(
                iso, demand_norm, supply_profiles, proc_pct, hydro_cap,
                np_profiles=np_profiles
            )
            sweep_results[proc_pct] = result
            score = result['hourly_match_score']
            print(f"    {proc_pct}%: {score}% match")
            if score >= 99.95 and proc_pct >= 140:
                break
    else:
        print(f"  Sweep: loaded from checkpoint ({len(iso_results['sweep'])} points)")
        sweep_results = None  # Already in iso_results['sweep']

    # Build sweep with costs, peak gap, and CO2 (skip if loaded from checkpoint)
    if sweep_results is not None:
        for p in sorted(sweep_results.keys()):
            r = sweep_results[p]
            peak_gap = compute_peak_gap(
                demand_norm, supply_profiles, r['resource_mix'],
                r['procurement_pct'], r['battery_dispatch_pct'], r['ldes_dispatch_pct'],
                anomaly_hours
            )
            r['peak_gap_pct'] = peak_gap
            costs = compute_costs(
                iso, r['resource_mix'], r['procurement_pct'],
                r['battery_dispatch_pct'], r['ldes_dispatch_pct'],
                r['hourly_match_score'],
                demand_norm, supply_profiles
            )
            r['costs'] = costs
            co2 = compute_co2_abatement(
                iso, r['resource_mix'], r['procurement_pct'], r['hourly_match_score'],
                r['battery_dispatch_pct'], r['ldes_dispatch_pct'],
                emission_rates, demand_total_mwh,
                demand_norm=demand_norm, supply_profiles=supply_profiles,
                fossil_mix=fossil_mix
            )
            r['co2_abated'] = co2
            iso_results['sweep'].append(r)
        save_checkpoint(iso, iso_results, phase='sweep')

    # ---- THRESHOLDS: co-optimize cost + matching for ALL 324 cost scenarios ----
    remaining = [t for t in THRESHOLDS if str(t) not in completed_thresholds]
    print(f"\n  Cost-optimizing thresholds (324 scenarios each)... "
          f"[{len(THRESHOLDS) - len(remaining)} done, {len(remaining)} remaining]")

    # Each threshold optimized independently; monotonicity enforced via
    # post-hoc re-sweep with broader parameters (not by result replacement)

    INTRA_THRESHOLD_CHECKPOINT_INTERVAL = 1  # Save every scenario — negligible I/O overhead (<0.1%)

    # ── CROSS-THRESHOLD SCORE CACHE ──
    # Matching scores are physics-based (cost-independent, threshold-independent).
    # A score for (mix, procurement, storage) is identical at 75% and 100% thresholds.
    # Share one cumulative cache across all thresholds to avoid recomputing ~50MB+
    # of scores per threshold. Load from the highest completed threshold's cache
    # (largest), then merge any partial cache for the current threshold.
    cumulative_score_cache = {}
    for t in reversed(THRESHOLDS):
        t_cache = load_score_cache(iso, t)
        if t_cache:
            cumulative_score_cache.update(t_cache)
            break  # Highest available has most entries; earlier ones are subsets
    # Also merge the partial threshold's cache if it exists and differs
    if partial_threshold_data:
        partial_t = partial_threshold_data.get('threshold')
        if partial_t and str(partial_t) not in completed_thresholds:
            partial_cache = load_score_cache(iso, partial_t)
            if partial_cache:
                cumulative_score_cache.update(partial_cache)
    if cumulative_score_cache:
        print(f"    [cache] Cumulative score cache: {len(cumulative_score_cache)} entries "
              f"(shared across thresholds)")

    # ── DISPATCHABLE FLOOR ──
    # At higher thresholds, dispatchable capacity (clean_firm + ccs_ccgt) must be
    # non-decreasing. Compute min(CF+CCS) across all scenarios from the previous
    # completed threshold and use (min - 10%) as a floor, pruning the search space.
    # 10% headroom preserves ability for costs to shift CF↔CCS split.
    dispatchable_floor = 0
    sorted_thresholds = sorted(THRESHOLDS)
    for t in sorted_thresholds:
        t_str = str(t)
        if t_str in completed_thresholds and t_str in iso_results.get('thresholds', {}):
            t_data = iso_results['thresholds'][t_str]
            scenarios = t_data.get('scenarios', {})
            if scenarios:
                min_disp = min(
                    s['resource_mix']['clean_firm'] + s['resource_mix']['ccs_ccgt']
                    for s in scenarios.values() if 'resource_mix' in s
                )
                dispatchable_floor = max(dispatchable_floor, min_disp - 10)
    dispatchable_floor = max(0, dispatchable_floor)
    if dispatchable_floor > 0:
        print(f"    [floor] Dispatchable floor (CF+CCS): {dispatchable_floor}% "
              f"(from prior thresholds, with 10% headroom)")

    for threshold in THRESHOLDS:
        t_str = str(threshold)
        if t_str in completed_thresholds:
            print(f"    {threshold}%: loaded from checkpoint — skipping")
            continue

        t_start = time.time()
        # Use cumulative cross-threshold cache (scores are physics-based, threshold-independent)
        score_cache = cumulative_score_cache
        threshold_scenarios = {}
        medium_result = None

        # Resume from partial checkpoint if available for this threshold
        partial_done = set()
        if partial_threshold_data and str(partial_threshold_data.get('threshold')) == t_str:
            threshold_scenarios = partial_threshold_data.get('scenarios', {})
            partial_done = set(threshold_scenarios.keys())
            print(f"    {threshold}%: resuming from partial checkpoint "
                  f"({len(partial_done)}/324 scenarios done)")
            # Clear partial data so it's not reused for next threshold
            partial_threshold_data = None

        # Determine if pruning applies for this threshold
        use_pruning = threshold <= PRUNING_THRESHOLD_CUTOFF
        if use_pruning:
            active_scenarios = [(k, v) for k, v in ALL_COST_SCENARIOS
                                if k in REPRESENTATIVE_SCENARIOS]
            total_active = len(active_scenarios)
            print(f"      (pruned: {total_active} representative scenarios, "
                  f"filling {324 - total_active} via cross-pollination)")
        else:
            active_scenarios = ALL_COST_SCENARIOS
            total_active = 324

        opt_count = 0
        cache_saved_this_run = False  # Track whether cache has been saved since start/resume

        # ── WARM-START: Run Medium (MMM_M_M) first to get warm-start seed ──
        # Medium is the central cost scenario. Its optimal mix serves as a high-quality
        # starting point for all other scenarios, eliminating the expensive Phase 1
        # coarse grid scan (270 combos × procurement levels) for 43/44 scenarios.
        # Phase 2/3 refinement still runs fully for each scenario, preserving the
        # co-optimization guarantee: different costs still produce different optimal mixes.
        medium_key = 'MMM_M_M'

        # Check if Medium was already completed (from checkpoint or not in active set)
        if medium_key in partial_done:
            medium_result = threshold_scenarios.get(medium_key)
        elif any(k == medium_key for k, _ in active_scenarios):
            # Run Medium with full 3-phase optimization (no warm-start)
            medium_cost_levels = COST_SCENARIO_MAP[medium_key]
            medium_result = optimize_for_threshold(
                iso, demand_norm, supply_profiles, threshold, hydro_cap,
                emission_rates, demand_total_mwh,
                cost_levels=medium_cost_levels, score_cache=score_cache,
                min_dispatchable=dispatchable_floor
            )
            if medium_result:
                r_gen, f_gen, stor, fuel, tx = medium_cost_levels
                cost_data = compute_costs_parameterized(
                    iso, medium_result['resource_mix'], medium_result['procurement_pct'],
                    medium_result['battery_dispatch_pct'], medium_result['ldes_dispatch_pct'],
                    medium_result['hourly_match_score'],
                    r_gen, f_gen, stor, fuel, tx
                )
                medium_result['costs'] = cost_data
                threshold_scenarios[medium_key] = medium_result

            opt_count += 1
            save_checkpoint(iso, iso_results,
                phase=f'threshold-{threshold}-partial-{len(threshold_scenarios)}of{total_active}',
                partial_threshold={'threshold': threshold, 'scenarios': threshold_scenarios})
            save_score_cache(iso, threshold, score_cache)
            cache_saved_this_run = True

        # ── Run extreme-archetype scenarios with full Phase 1 to capture diverse mixes ──
        # These scenarios represent opposite corners of the cost space where the optimal
        # mix is most likely to diverge from Medium. Running them with full Phase 1
        # ensures we discover all major mix archetypes before warm-starting the rest.
        EXTREME_ARCHETYPES = [
            'HLL_L_N',  # High renewables, low firm, low storage, low fuel, no transmission
            'LHL_L_M',  # Low renewables, high firm, low storage, low fuel, med transmission
            'LLH_H_M',  # Low renewables, low firm, high storage, high fuel, med transmission
            'HHH_H_H',  # All high — maximum cost pressure
        ]

        # Run archetypes that haven't been completed yet
        for archetype_key in EXTREME_ARCHETYPES:
            if archetype_key in partial_done or archetype_key in threshold_scenarios:
                continue
            if archetype_key not in COST_SCENARIO_MAP:
                continue
            arch_cost_levels = COST_SCENARIO_MAP[archetype_key]
            arch_result = optimize_for_threshold(
                iso, demand_norm, supply_profiles, threshold, hydro_cap,
                emission_rates, demand_total_mwh,
                cost_levels=arch_cost_levels, score_cache=score_cache,
                min_dispatchable=dispatchable_floor
            )
            if arch_result:
                r_gen, f_gen, stor, fuel, tx = arch_cost_levels
                cost_data = compute_costs_parameterized(
                    iso, arch_result['resource_mix'], arch_result['procurement_pct'],
                    arch_result['battery_dispatch_pct'], arch_result['ldes_dispatch_pct'],
                    arch_result['hourly_match_score'],
                    r_gen, f_gen, stor, fuel, tx
                )
                arch_result['costs'] = cost_data
                threshold_scenarios[archetype_key] = arch_result

            opt_count += 1
            if opt_count % INTRA_THRESHOLD_CHECKPOINT_INTERVAL == 0:
                save_checkpoint(iso, iso_results,
                    phase=f'threshold-{threshold}-partial-{len(threshold_scenarios)}of{total_active}',
                    partial_threshold={'threshold': threshold, 'scenarios': threshold_scenarios})
                if not cache_saved_this_run or opt_count % 10 == 0:
                    save_score_cache(iso, threshold, score_cache)
                    cache_saved_this_run = True

        # Collect all diverse warm-start seeds from completed scenarios
        # These represent the full range of discovered mix archetypes
        warm_start_seeds = []
        seen_mix_archetypes = set()
        for sk, res in threshold_scenarios.items():
            mix = res['resource_mix']
            # Round to 5% to group similar mixes into archetypes
            archetype = tuple(round(mix[rt] / 5) * 5 for rt in RESOURCE_TYPES)
            if archetype not in seen_mix_archetypes:
                seen_mix_archetypes.add(archetype)
                warm_start_seeds.append(res)

        n_seeds = len(warm_start_seeds)
        n_archetypes_run = len([k for k in EXTREME_ARCHETYPES if k in threshold_scenarios])
        print(f"      Warm-start: {n_seeds} diverse seed mixes from Medium + "
              f"{n_archetypes_run} archetypes")

        # ── Run remaining scenarios with warm-start from diverse seeds ──
        warm_start_count = 0
        full_phase1_count = 0
        for s_idx, (scenario_key, cost_levels) in enumerate(active_scenarios):
            if scenario_key in partial_done:
                continue  # Already completed in previous session
            if scenario_key == medium_key and medium_key not in partial_done:
                continue  # Already run above
            if scenario_key in threshold_scenarios:
                continue  # Already run as archetype

            # Use Medium result as primary warm-start, with all seeds as seed_mixes
            ws = medium_result if medium_result else None
            # Pass all diverse seed mixes to inject into warm-start Phase 1
            extra_seeds = [r['resource_mix'] for r in warm_start_seeds if r is not ws]
            result = optimize_for_threshold(
                iso, demand_norm, supply_profiles, threshold, hydro_cap,
                emission_rates, demand_total_mwh,
                cost_levels=cost_levels, score_cache=score_cache,
                warm_start_result=ws,
                seed_mixes=extra_seeds if ws else None,
                min_dispatchable=dispatchable_floor
            )
            if ws:
                warm_start_count += 1
            else:
                full_phase1_count += 1

            # Dynamically add newly-discovered archetypes to the seed pool
            if result:
                mix = result['resource_mix']
                archetype = tuple(round(mix[rt] / 5) * 5 for rt in RESOURCE_TYPES)
                if archetype not in seen_mix_archetypes:
                    seen_mix_archetypes.add(archetype)
                    warm_start_seeds.append(result)

            if result:
                # Compute cost for this scenario
                r_gen, f_gen, stor, fuel, tx = cost_levels
                cost_data = compute_costs_parameterized(
                    iso, result['resource_mix'], result['procurement_pct'],
                    result['battery_dispatch_pct'], result['ldes_dispatch_pct'],
                    result['hourly_match_score'],
                    r_gen, f_gen, stor, fuel, tx
                )
                result['costs'] = cost_data

                threshold_scenarios[scenario_key] = result

            opt_count += 1
            # Intra-threshold checkpoint: save every scenario
            if opt_count > 0 and opt_count % INTRA_THRESHOLD_CHECKPOINT_INTERVAL == 0:
                save_checkpoint(iso, iso_results,
                    phase=f'threshold-{threshold}-partial-{len(threshold_scenarios)}of{total_active}',
                    partial_threshold={'threshold': threshold, 'scenarios': threshold_scenarios})
                # Score cache save cadence: 1st scenario after start/resume, then every 10
                if not cache_saved_this_run or opt_count % 10 == 0:
                    save_score_cache(iso, threshold, score_cache)
                    cache_saved_this_run = True

        # ── Adaptive resampling: if unique mixes > 50% of scenarios run, expand ──
        # Target: unique_mixes < 50% of total scenarios run
        # Expansion: add unrun scenarios (midpoints) until ratio drops below 50%
        if use_pruning:
            UNIQUENESS_THRESHOLD = 0.50  # Max ratio of unique mixes to scenarios run
            MAX_RESAMPLE_ROUNDS = 5
            already_run = set(threshold_scenarios.keys())
            total_run = len(already_run)

            for resample_round in range(MAX_RESAMPLE_ROUNDS):
                unique_mixes = set()
                for res in threshold_scenarios.values():
                    m = res['resource_mix']
                    unique_mixes.add((m['clean_firm'], m['solar'], m['wind'],
                                     m['ccs_ccgt'], m['hydro'],
                                     res['procurement_pct'],
                                     res['battery_dispatch_pct'],
                                     res['ldes_dispatch_pct']))
                ratio = len(unique_mixes) / total_run if total_run > 0 else 0

                if ratio <= UNIQUENESS_THRESHOLD:
                    if resample_round > 0:
                        print(f"      Resampling converged: {len(unique_mixes)} unique mixes "
                              f"/ {total_run} scenarios = {ratio:.0%}")
                    break

                # Need more scenarios — pick unrun ones from ALL_COST_SCENARIOS
                unrun = [(k, v) for k, v in ALL_COST_SCENARIOS if k not in already_run]
                # Scale: add enough to bring ratio below threshold
                # target_total = unique_mixes / 0.50 → need (target - current) more
                target_total = int(len(unique_mixes) / UNIQUENESS_THRESHOLD) + 1
                add_count = min(target_total - total_run, len(unrun))
                # Spread evenly across unrun list (midpoints)
                step = max(1, len(unrun) // add_count) if add_count > 0 else 1
                to_add = [unrun[i] for i in range(0, len(unrun), step)][:add_count]

                print(f"      RESAMPLE round {resample_round + 1}: "
                      f"{len(unique_mixes)} unique / {total_run} run = {ratio:.0%} > 50% — "
                      f"adding {len(to_add)} midpoint scenarios")

                for scenario_key, cost_levels in to_add:
                    result = optimize_for_threshold(
                        iso, demand_norm, supply_profiles, threshold, hydro_cap,
                        emission_rates, demand_total_mwh,
                        cost_levels=cost_levels, score_cache=score_cache,
                        min_dispatchable=dispatchable_floor
                    )
                    if result:
                        r_gen, f_gen, stor, fuel, tx = cost_levels
                        cost_data = compute_costs_parameterized(
                            iso, result['resource_mix'], result['procurement_pct'],
                            result['battery_dispatch_pct'], result['ldes_dispatch_pct'],
                            result['hourly_match_score'],
                            r_gen, f_gen, stor, fuel, tx
                        )
                        result['costs'] = cost_data
                        threshold_scenarios[scenario_key] = result
                        if scenario_key == 'MMM_M_M':
                            medium_result = result
                    already_run.add(scenario_key)
                    total_run += 1
                    opt_count += 1
                    if opt_count % INTRA_THRESHOLD_CHECKPOINT_INTERVAL == 0:
                        save_checkpoint(iso, iso_results,
                            phase=f'threshold-{threshold}-resample-{len(threshold_scenarios)}',
                            partial_threshold={'threshold': threshold, 'scenarios': threshold_scenarios})

        # Log Medium scenario progress + warm-start stats
        t_elapsed = time.time() - t_start
        cache_size = len(score_cache)
        opt_label = f"{opt_count} optimized" if use_pruning else f"{len(threshold_scenarios)}/324"
        ws_label = f" (warm-start: {warm_start_count}, full: {full_phase1_count + 1})" if warm_start_count > 0 else ""
        if medium_result:
            mix = medium_result['resource_mix']
            print(f"    {threshold}%: {opt_label} scenarios{ws_label}, "
                  f"cache={cache_size}, {t_elapsed:.1f}s | "
                  f"Medium: CF{mix['clean_firm']}/Sol{mix['solar']}/Wnd{mix['wind']}"
                  f"/CCS{mix['ccs_ccgt']}/Hyd{mix['hydro']} "
                  f"batt={medium_result['battery_dispatch_pct']}% "
                  f"ldes={medium_result['ldes_dispatch_pct']}%"
                  f" | {len(seen_mix_archetypes)} unique archetypes discovered")
        else:
            print(f"    {threshold}%: {opt_label} scenarios{ws_label}, "
                  f"cache={cache_size}, {t_elapsed:.1f}s")

        # ── Cross-pollination: evaluate all discovered mixes for all scenarios ──
        # A mix found optimal for one cost scenario may be cheaper for another
        unique_results = []
        seen_mix_keys = set()
        for sk, res in threshold_scenarios.items():
            mix = res['resource_mix']
            mk = (mix['clean_firm'], mix['solar'], mix['wind'],
                  mix['ccs_ccgt'], mix['hydro'],
                  res['procurement_pct'],
                  res['battery_dispatch_pct'],
                  res['ldes_dispatch_pct'])
            if mk not in seen_mix_keys:
                seen_mix_keys.add(mk)
                unique_results.append(res)

        cross_fixes = 0
        for scenario_key, cost_levels in ALL_COST_SCENARIOS:
            r_gen, f_gen, stor, fuel, tx = cost_levels
            current_cost = float('inf')
            if scenario_key in threshold_scenarios:
                current = threshold_scenarios[scenario_key]
                if 'costs' in current:
                    current_cost = current['costs']['effective_cost']

            for candidate in unique_results:
                if candidate['hourly_match_score'] < threshold:
                    continue
                cand_cost_data = compute_costs_parameterized(
                    iso, candidate['resource_mix'], candidate['procurement_pct'],
                    candidate['battery_dispatch_pct'], candidate['ldes_dispatch_pct'],
                    candidate['hourly_match_score'],
                    r_gen, f_gen, stor, fuel, tx
                )
                if cand_cost_data['effective_cost'] < current_cost:
                    threshold_scenarios[scenario_key] = dict(candidate)
                    threshold_scenarios[scenario_key]['costs'] = cand_cost_data
                    current_cost = cand_cost_data['effective_cost']
                    cross_fixes += 1

        total_after = len(threshold_scenarios)
        filled_count = total_after - (opt_count + len(partial_done))
        if cross_fixes > 0 or filled_count > 0:
            parts = []
            if cross_fixes > 0:
                parts.append(f"{cross_fixes} improvements")
            if use_pruning and filled_count > 0:
                parts.append(f"{filled_count} filled via cross-pollination")
            parts.append(f"{len(unique_results)} unique mixes")
            print(f"      Cross-pollination: {', '.join(parts)} → {total_after}/324 total")

        # Store all scenarios for this threshold
        # Medium result gets full detail (compressed day, peak gap, CO2)
        medium_key = 'MMM_M_M'
        if medium_key in threshold_scenarios:
            med = threshold_scenarios[medium_key]
            peak_gap = compute_peak_gap(
                demand_norm, supply_profiles, med['resource_mix'],
                med['procurement_pct'], med['battery_dispatch_pct'],
                med['ldes_dispatch_pct'], anomaly_hours
            )
            med['peak_gap_pct'] = peak_gap

            # Full costs at Medium
            full_costs = compute_costs(
                iso, med['resource_mix'], med['procurement_pct'],
                med['battery_dispatch_pct'], med['ldes_dispatch_pct'],
                med['hourly_match_score'],
                demand_norm, supply_profiles
            )
            med['costs_detail'] = full_costs

            co2 = compute_co2_abatement(
                iso, med['resource_mix'], med['procurement_pct'],
                med['hourly_match_score'],
                med['battery_dispatch_pct'], med['ldes_dispatch_pct'],
                emission_rates, demand_total_mwh,
                demand_norm=demand_norm, supply_profiles=supply_profiles,
                fossil_mix=fossil_mix
            )
            med['co2_abated'] = co2

            cdp = compute_compressed_day(
                demand_norm, supply_profiles, med['resource_mix'],
                med['procurement_pct'],
                med['battery_dispatch_pct'], med['ldes_dispatch_pct']
            )
            med['compressed_day'] = cdp

        iso_results['thresholds'][str(threshold)] = {
            'scenarios': threshold_scenarios,
            'scenario_count': len(threshold_scenarios),
        }
        # Checkpoint after each threshold — never lose more than one threshold's work
        save_checkpoint(iso, iso_results, phase=f'threshold-{threshold}')
        # Persist score cache for warm restart on interruption
        save_score_cache(iso, threshold, score_cache)
        # Incremental dashboard update — push each threshold to results JSON
        save_results_incremental(iso, iso_results)

        # Update dispatchable floor for next threshold
        if threshold_scenarios:
            min_disp = min(
                s['resource_mix']['clean_firm'] + s['resource_mix']['ccs_ccgt']
                for s in threshold_scenarios.values() if 'resource_mix' in s
            )
            new_floor = max(0, min_disp - 10)
            if new_floor > dispatchable_floor:
                dispatchable_floor = new_floor
                print(f"    [floor] Updated dispatchable floor: {dispatchable_floor}% "
                      f"(min CF+CCS={min_disp}% at {threshold}%, minus 10% headroom)")

    # ---- MONOTONICITY CORRECTION ----
    # For each cost scenario, cost must be non-decreasing across thresholds.
    # If cost(T_lower) > cost(T_higher), the search missed a better solution at T_lower.
    #
    # Two-phase approach (warm-start + self-learning):
    #   Phase 1: Global cross-threshold pollination — evaluate ALL unique mixes from
    #            ALL higher thresholds against violated scenarios (cost math only, no
    #            optimization). This resolves most violations in milliseconds.
    #   Phase 2: Targeted re-sweep — only for scenarios that Phase 1 couldn't fix.
    #            Uses normal search parameters (not 4x broader) since seed mixes from
    #            higher thresholds are already close to optimal. Each round's winning
    #            mixes are added to the seed library for subsequent rounds.
    MAX_RESWEEP_ROUNDS = 2
    sorted_thresholds = sorted(THRESHOLDS)

    # Build set of all scenario keys present across thresholds
    checked_scenarios = set()
    for t_idx in range(len(sorted_thresholds)):
        t_str = str(sorted_thresholds[t_idx])
        if t_str not in iso_results['thresholds']:
            continue
        for sk in iso_results['thresholds'][t_str]['scenarios']:
            checked_scenarios.add(sk)

    # ── Collect ALL unique mixes across ALL thresholds (for global cross-pollination) ──
    all_threshold_mixes = {}  # threshold -> list of unique (result, mix_key) pairs
    all_unique_results = []   # flat list of all unique results across all thresholds
    global_seen_keys = set()
    for t in sorted_thresholds:
        t_str = str(t)
        if t_str not in iso_results['thresholds']:
            continue
        t_mixes = []
        for sk, res in iso_results['thresholds'][t_str]['scenarios'].items():
            if 'resource_mix' not in res:
                continue
            mix = res['resource_mix']
            mk = (mix['clean_firm'], mix['solar'], mix['wind'],
                  mix['ccs_ccgt'], mix['hydro'],
                  res['procurement_pct'],
                  res['battery_dispatch_pct'],
                  res['ldes_dispatch_pct'])
            if mk not in global_seen_keys:
                global_seen_keys.add(mk)
                t_mixes.append(res)
                all_unique_results.append(res)
        all_threshold_mixes[t] = t_mixes
    print(f"  Monotonicity: {len(all_unique_results)} unique mixes across "
          f"{len(all_threshold_mixes)} thresholds")

    # Self-learning seed library: accumulates winning mixes across rounds
    winning_seed_library = []
    winning_seed_keys = set()

    for resweep_round in range(MAX_RESWEEP_ROUNDS + 1):  # +1: round 0 is cross-pollination only
        # Detect monotonicity violations: {threshold: {scenario_key: better_threshold}}
        violations = {}
        total_violations = 0
        for scenario_key in checked_scenarios:
            prev_cost = None
            prev_t = None
            for threshold in sorted_thresholds:
                t_str = str(threshold)
                if t_str not in iso_results['thresholds']:
                    continue
                scenarios = iso_results['thresholds'][t_str]['scenarios']
                if scenario_key not in scenarios:
                    continue
                result = scenarios[scenario_key]
                if 'costs' not in result:
                    continue
                cost = result['costs']['effective_cost']
                if prev_cost is not None and cost < prev_cost - 0.01:
                    total_violations += 1
                    if prev_t not in violations:
                        violations[prev_t] = {}
                    violations[prev_t][scenario_key] = threshold
                prev_cost = cost
                prev_t = threshold

        if not violations:
            if resweep_round == 0:
                print(f"  Monotonicity check passed for all scenarios")
            else:
                print(f"  Monotonicity resolved after round {resweep_round} "
                      f"({len(winning_seed_library)} winning mixes learned)")
            break

        # ── Phase 1: Global cross-threshold pollination (cost math only) ──
        phase1_fixes = 0
        for viol_threshold in sorted(violations.keys()):
            violated_scenarios = violations[viol_threshold]
            t_str = str(viol_threshold)
            threshold_scenarios = iso_results['thresholds'][t_str]['scenarios']

            # Collect candidates from ALL higher thresholds + winning seed library
            candidates = []
            for t in sorted_thresholds:
                if t > viol_threshold and t in all_threshold_mixes:
                    candidates.extend(all_threshold_mixes[t])
            candidates.extend(winning_seed_library)

            for scenario_key in list(violated_scenarios.keys()):
                cost_levels = COST_SCENARIO_MAP[scenario_key]
                r_gen, f_gen, stor, fuel, tx = cost_levels
                current = threshold_scenarios.get(scenario_key)
                current_cost = current['costs']['effective_cost'] if current and 'costs' in current else float('inf')
                fixed = False

                for candidate in candidates:
                    if candidate['hourly_match_score'] < viol_threshold:
                        continue
                    cand_cost_data = compute_costs_parameterized(
                        iso, candidate['resource_mix'], candidate['procurement_pct'],
                        candidate['battery_dispatch_pct'], candidate['ldes_dispatch_pct'],
                        candidate['hourly_match_score'],
                        r_gen, f_gen, stor, fuel, tx
                    )
                    if cand_cost_data['effective_cost'] < current_cost:
                        threshold_scenarios[scenario_key] = dict(candidate)
                        threshold_scenarios[scenario_key]['costs'] = cand_cost_data
                        current_cost = cand_cost_data['effective_cost']
                        fixed = True

                if fixed:
                    phase1_fixes += 1
                    del violated_scenarios[scenario_key]

            # Remove threshold from violations if all fixed
            if not violated_scenarios:
                del violations[viol_threshold]

        # Re-count remaining violations after Phase 1
        remaining_violations = sum(len(v) for v in violations.values())
        print(f"  Monotonicity round {resweep_round + 1}: {total_violations} violations, "
              f"{phase1_fixes} fixed via cross-threshold pollination, "
              f"{remaining_violations} remaining")

        if not violations:
            break

        # Round 0 is cross-pollination only — no re-optimization
        if resweep_round == 0 and remaining_violations > 0:
            print(f"    {remaining_violations} violations need targeted re-sweep")

        # ── Phase 2: Targeted re-sweep (only for remaining violations) ──
        if resweep_round > 0:
            for viol_threshold in sorted(violations.keys()):
                violated_scenarios = violations[viol_threshold]
                t_str = str(viol_threshold)

                # Collect seed mixes: higher threshold mixes + winning seed library
                seed_mixes_for_resweep = []
                seen_seeds = set()
                # Seeds from the specific better thresholds
                for sk, better_t in violated_scenarios.items():
                    better_result = iso_results['thresholds'][str(better_t)]['scenarios'].get(sk)
                    if better_result and 'resource_mix' in better_result:
                        key = tuple(better_result['resource_mix'][rt] for rt in RESOURCE_TYPES)
                        if key not in seen_seeds:
                            seen_seeds.add(key)
                            seed_mixes_for_resweep.append(dict(better_result['resource_mix']))
                # Add winning seeds from previous rounds
                for ws in winning_seed_library:
                    if 'resource_mix' in ws:
                        key = tuple(ws['resource_mix'][rt] for rt in RESOURCE_TYPES)
                        if key not in seen_seeds:
                            seen_seeds.add(key)
                            seed_mixes_for_resweep.append(dict(ws['resource_mix']))

                # Use normal procurement bounds (not 4x expanded) — seeds are warm
                default_min, default_max = PROCUREMENT_BOUNDS.get(viol_threshold, (70, 310))
                expanded_min = max(60, default_min - 10)
                expanded_max = min(400, default_max + 20)

                print(f"    Re-sweeping {viol_threshold}%: {len(violated_scenarios)} scenarios, "
                      f"{len(seed_mixes_for_resweep)} seed mixes "
                      f"(incl {len(winning_seed_library)} learned)")

                resweep_cache = dict(cumulative_score_cache)
                resweep_fixes = 0
                threshold_scenarios = iso_results['thresholds'][t_str]['scenarios']

                for scenario_key in violated_scenarios:
                    cost_levels = COST_SCENARIO_MAP[scenario_key]
                    result = optimize_for_threshold(
                        iso, demand_norm, supply_profiles, viol_threshold, hydro_cap,
                        emission_rates, demand_total_mwh,
                        cost_levels=cost_levels, score_cache=resweep_cache,
                        resweep=False, seed_mixes=seed_mixes_for_resweep,
                        procurement_bounds_override=(expanded_min, expanded_max),
                        min_dispatchable=dispatchable_floor
                    )

                    if result:
                        r_gen, f_gen, stor, fuel, tx = cost_levels
                        cost_data = compute_costs_parameterized(
                            iso, result['resource_mix'], result['procurement_pct'],
                            result['battery_dispatch_pct'], result['ldes_dispatch_pct'],
                            result['hourly_match_score'],
                            r_gen, f_gen, stor, fuel, tx
                        )
                        result['costs'] = cost_data

                        current = threshold_scenarios.get(scenario_key)
                        if current is None or cost_data['effective_cost'] < current['costs']['effective_cost']:
                            threshold_scenarios[scenario_key] = result
                            resweep_fixes += 1
                            # Add winning mix to seed library (self-learning)
                            mix = result['resource_mix']
                            mk = (mix['clean_firm'], mix['solar'], mix['wind'],
                                  mix['ccs_ccgt'], mix['hydro'],
                                  result['procurement_pct'],
                                  result['battery_dispatch_pct'],
                                  result['ldes_dispatch_pct'])
                            if mk not in winning_seed_keys:
                                winning_seed_keys.add(mk)
                                winning_seed_library.append(result)

                # Cross-pollination within re-swept threshold
                unique_results = []
                seen_mix_keys = set()
                for sk, res in threshold_scenarios.items():
                    mix = res['resource_mix']
                    mk = (mix['clean_firm'], mix['solar'], mix['wind'],
                          mix['ccs_ccgt'], mix['hydro'],
                          res['procurement_pct'],
                          res['battery_dispatch_pct'],
                          res['ldes_dispatch_pct'])
                    if mk not in seen_mix_keys:
                        seen_mix_keys.add(mk)
                        unique_results.append(res)

                cross_fixes = 0
                for scenario_key in violated_scenarios:
                    cost_levels = COST_SCENARIO_MAP[scenario_key]
                    r_gen, f_gen, stor, fuel, tx = cost_levels
                    current = threshold_scenarios.get(scenario_key)
                    current_cost = current['costs']['effective_cost'] if current and 'costs' in current else float('inf')

                    for candidate in unique_results:
                        if candidate['hourly_match_score'] < viol_threshold:
                            continue
                        cand_cost_data = compute_costs_parameterized(
                            iso, candidate['resource_mix'], candidate['procurement_pct'],
                            candidate['battery_dispatch_pct'], candidate['ldes_dispatch_pct'],
                            candidate['hourly_match_score'],
                            r_gen, f_gen, stor, fuel, tx
                        )
                        if cand_cost_data['effective_cost'] < current_cost:
                            threshold_scenarios[scenario_key] = dict(candidate)
                            threshold_scenarios[scenario_key]['costs'] = cand_cost_data
                            current_cost = cand_cost_data['effective_cost']
                            cross_fixes += 1

                # Update Medium scenario detail if it was re-swept
                medium_key = 'MMM_M_M'
                if medium_key in violated_scenarios and medium_key in threshold_scenarios:
                    med = threshold_scenarios[medium_key]
                    peak_gap = compute_peak_gap(
                        demand_norm, supply_profiles, med['resource_mix'],
                        med['procurement_pct'], med['battery_dispatch_pct'],
                        med['ldes_dispatch_pct'], anomaly_hours
                    )
                    med['peak_gap_pct'] = peak_gap
                    full_costs = compute_costs(
                        iso, med['resource_mix'], med['procurement_pct'],
                        med['battery_dispatch_pct'], med['ldes_dispatch_pct'],
                        med['hourly_match_score'],
                        demand_norm, supply_profiles
                    )
                    med['costs_detail'] = full_costs
                    co2 = compute_co2_abatement(
                        iso, med['resource_mix'], med['procurement_pct'],
                        med['hourly_match_score'],
                        med['battery_dispatch_pct'], med['ldes_dispatch_pct'],
                        emission_rates, demand_total_mwh,
                        demand_norm=demand_norm, supply_profiles=supply_profiles,
                        fossil_mix=fossil_mix
                    )
                    med['co2_abated'] = co2
                    cdp = compute_compressed_day(
                        demand_norm, supply_profiles, med['resource_mix'],
                        med['procurement_pct'],
                        med['battery_dispatch_pct'], med['ldes_dispatch_pct']
                    )
                    med['compressed_day'] = cdp

                print(f"      Fixed {resweep_fixes} via re-sweep, "
                      f"{cross_fixes} via cross-pollination")

    else:
        # Exhausted MAX_RESWEEP_ROUNDS — report remaining violations
        remaining = 0
        for scenario_key in checked_scenarios:
            prev_cost = None
            prev_t = None
            for threshold in sorted_thresholds:
                t_str = str(threshold)
                if t_str not in iso_results['thresholds']:
                    continue
                scenarios = iso_results['thresholds'][t_str]['scenarios']
                if scenario_key not in scenarios:
                    continue
                result = scenarios[scenario_key]
                if 'costs' not in result:
                    continue
                cost = result['costs']['effective_cost']
                if prev_cost is not None and cost < prev_cost - 0.01:
                    remaining += 1
                prev_cost = cost
                prev_t = threshold
        if remaining > 0:
            print(f"  WARNING: {remaining} monotonicity violations remain after "
                  f"{MAX_RESWEEP_ROUNDS + 1} rounds (search space exhausted)")
        else:
            print(f"  All monotonicity violations resolved after {MAX_RESWEEP_ROUNDS + 1} rounds")

    # Final incremental save after monotonicity correction
    save_results_incremental(iso, iso_results)

    # ISO complete — clear checkpoint (full results saved in main())
    clear_checkpoint(iso)
    return iso, iso_results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import sys
    # Support single-ISO mode: python optimize_overprocure.py --iso PJM
    target_isos = None
    if '--iso' in sys.argv:
        idx = sys.argv.index('--iso')
        if idx + 1 < len(sys.argv):
            target_isos = [iso.strip() for iso in sys.argv[idx + 1].split(',')]
            invalid = [iso for iso in target_isos if iso not in ISOS]
            if invalid:
                print(f"ERROR: Unknown ISO(s): {invalid}. Valid: {ISOS}")
                sys.exit(1)
            print(f"  Single-ISO mode: running {', '.join(target_isos)} only")

    start_time = time.time()
    demand_data, gen_profiles, emission_rates, fossil_mix = load_data()

    # Build config section with ALL cost tables for dashboard recalculation
    config = {
        'data_year': DATA_YEAR,
        'battery_duration': '4h',
        'battery_efficiency': BATTERY_EFFICIENCY,
        'ldes_duration': '100h',
        'ldes_efficiency': LDES_EFFICIENCY,
        'ldes_window_days': LDES_WINDOW_DAYS,
        'hydro_caps': HYDRO_CAPS,
        'resource_types': RESOURCE_TYPES,
        'thresholds': THRESHOLDS,
        'procurement_bounds': {str(k): list(v) for k, v in PROCUREMENT_BOUNDS.items()},
        'total_scenarios_per_threshold': len(ALL_COST_SCENARIOS),
        'wholesale_prices': WHOLESALE_PRICES,
        'regional_lcoe': REGIONAL_LCOE,
        'grid_mix_shares': GRID_MIX_SHARES,
        'ccs_residual_emission_rate': CCS_RESIDUAL_EMISSION_RATE,
        # Paired toggle group definitions for dashboard controls
        'paired_toggle_groups': PAIRED_TOGGLE_GROUPS,
        # Full L/M/H tables for dashboard cost recalculation
        'lcoe_tables': FULL_LCOE_TABLES,
        'transmission_tables': FULL_TRANSMISSION_TABLES,
        'fuel_prices': FUEL_PRICES,
        # Emission rates from eGRID (for CO2 calculations)
        'emission_rates': {
            iso: {
                'fossil_co2_lb_per_mwh': emission_rates[iso]['fossil_co2_lb_per_mwh'],
                'gas_co2_lb_per_mwh': emission_rates[iso]['gas_co2_lb_per_mwh'],
                'coal_co2_lb_per_mwh': emission_rates[iso]['coal_co2_lb_per_mwh'],
                'oil_co2_lb_per_mwh': emission_rates[iso]['oil_co2_lb_per_mwh'],
                'total_co2_lb_per_mwh': emission_rates[iso]['total_co2_lb_per_mwh'],
            }
            for iso in ISOS
        },
        # Grid mix shares for baseline display
        'grid_mix_shares_display': GRID_MIX_SHARES,
        # Regional fossil fuel mix shares (for fuel price sensitivity)
        'fossil_mix': {
            iso: {
                'coal_share': fossil_mix[iso][DATA_YEAR]['coal_share'][:24] if DATA_YEAR in fossil_mix.get(iso, {}) else [],
                'gas_share': fossil_mix[iso][DATA_YEAR]['gas_share'][:24] if DATA_YEAR in fossil_mix.get(iso, {}) else [],
                'oil_share': fossil_mix[iso][DATA_YEAR]['oil_share'][:24] if DATA_YEAR in fossil_mix.get(iso, {}) else [],
            }
            for iso in ISOS
        },
    }

    all_results = {
        'config': config,
        'results': {},
    }

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard', 'overprocure_results.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Set module-level state for incremental per-threshold dashboard saves
    global _dashboard_config, _output_path
    _dashboard_config = config
    _output_path = output_path

    # Load existing results to preserve old ISOs until overridden
    if os.path.exists(output_path):
        try:
            with open(output_path) as f:
                existing = json.load(f)
            all_results['results'] = existing.get('results', {})
            print(f"  Loaded existing results: {len(all_results['results'])} ISOs preserved")
        except (json.JSONDecodeError, IOError):
            pass

    # Run ISOs sequentially to avoid memory pressure and enable incremental saves.
    # Each ISO does 10 thresholds × 324 scenarios — heavy compute per ISO.
    # Per-threshold incremental saves happen inside process_iso() via save_results_incremental().
    run_isos = target_isos if target_isos else ISOS
    for iso in run_isos:
        iso_start = time.time()
        args = (iso, demand_data, gen_profiles, emission_rates, fossil_mix)
        iso_name, iso_results = process_iso(args)
        all_results['results'][iso_name] = iso_results
        iso_elapsed = time.time() - iso_start
        print(f"\n  {iso_name} completed in {iso_elapsed:.0f}s")

        # Final ISO-level save (redundant with per-threshold saves, but ensures completeness)
        with open(output_path, 'w') as f:
            json.dump(all_results, f)
        print(f"  Saved: {os.path.getsize(output_path) / 1024:.0f} KB "
              f"({len(all_results['results'])}/{len(ISOS)} ISOs)")

    # Save cached results file (reusable input for future projects)
    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'optimizer_cache.json')
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    try:
        import subprocess
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                           cwd=os.path.dirname(os.path.abspath(__file__)),
                                           stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_hash = 'unknown'

    from datetime import datetime, timezone
    cache_data = {
        'metadata': {
            'created_at': datetime.now(timezone.utc).isoformat(),
            'optimizer_version': '3.1-paired-toggles-resweep',
            'git_commit': git_hash,
            'runtime_seconds': round(time.time() - start_time, 1),
            'description': 'Full co-optimized results: 10 thresholds x 324 paired-toggle scenarios x 5 ISOs',
            'thresholds': THRESHOLDS,
            'isos': ISOS,
            'scenarios_per_threshold': len(ALL_COST_SCENARIOS),
            'total_optimizations': len(THRESHOLDS) * len(ALL_COST_SCENARIOS) * len(ISOS),
        },
        'config': config,
        'results': all_results['results'],
    }
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f, indent=2)
    print(f"  Cached results: {cache_path} ({os.path.getsize(cache_path) / 1024:.0f} KB)")

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  Complete in {elapsed:.0f}s. Saved to: {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1024:.0f} KB")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
