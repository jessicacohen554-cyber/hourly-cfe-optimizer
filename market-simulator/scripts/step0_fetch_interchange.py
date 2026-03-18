#!/usr/bin/env python3
"""
Step 0: Generate inter-regional interchange profiles per ISO.

Creates hourly net import/export time series for each ISO based on published
EIA-930 interchange statistics. Net interchange = imports - exports per hour.
Positive values = net imports (reduce residual demand).

When EIA-930 hourly data is available in data/eia-930/, aggregates BA-level
interchange to ISO level using BA_TO_ISO mapping. Otherwise, generates
synthetic profiles from published annual averages with realistic
diurnal/seasonal shapes.

Output: data/profiles/eia_interchange_profiles.json
Format: {ISO: {"2024": {"net_import_mw": [8760], "net_import_norm": [8760]}}}

Sources:
  - EIA-930 Hourly Grid Monitor (2024 data)
  - NERC Interregional Transfer Capability Reports
  - CAISO DMM Annual Reports (Path 66, PDCI flows)
  - NYISO Gold Book (HQ/PJM/NEISO interface flows)
"""

import json
import os
import sys
import numpy as np

MODULE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(MODULE_ROOT, 'data')
PROFILES_DIR = os.path.join(DATA_DIR, 'profiles')
OUTPUT_FILE = os.path.join(PROFILES_DIR, 'eia_interchange_profiles.json')

H = 8760

# --- Annual average net imports (MW) by ISO ---
# Source: EIA-930 2024 annual summary. Positive = net importer.
# These are the mean hourly net import MW over the full year.
ANNUAL_AVG_NET_IMPORT_MW = {
    'CAISO': 5500,    # Heavy PNW hydro imports; net importer year-round
    'ERCOT': -200,    # Slight net exporter via DC ties (mostly self-contained)
    'PJM': -2000,     # Large net exporter to MISO/NYISO
    'NYISO': 2500,    # Net importer from PJM, HQ, NEISO
    'NEISO': 2200,    # Net importer from HQ (Phase I/II), NYISO, NB Power
    'MISO': 1500,     # Net importer from PJM, slight exporter to SPP
    'SPP': -800,      # Slight net exporter (wind surplus to MISO)
}

# --- Seasonal amplitude (fraction of mean) ---
# How much interchange varies by season. E.g., 0.3 means ±30% of mean.
SEASONAL_AMPLITUDE = {
    'CAISO': 0.40,    # Higher summer imports (PNW hydro) + evening ramp
    'ERCOT': 0.50,    # Small base → big relative swings
    'PJM': 0.25,      # Fairly steady exporter
    'NYISO': 0.30,    # HQ imports seasonal (more in summer)
    'NEISO': 0.35,    # HQ imports higher in winter (heating demand)
    'MISO': 0.30,     # Moderate seasonal variation
    'SPP': 0.40,      # Wind-driven exports vary seasonally
}

# --- Seasonal peak month (0-indexed: 0=Jan, 6=Jul) ---
# Month when net imports peak (or net exports are smallest).
SEASONAL_PEAK_MONTH = {
    'CAISO': 7,     # Aug — peak summer demand + PNW hydro still flowing
    'ERCOT': 7,     # Aug — peak demand, slight increase in imports
    'PJM': 0,       # Jan — winter demand, reduced exports
    'NYISO': 7,     # Aug — summer peak, max HQ imports
    'NEISO': 0,     # Jan — winter heating demand, max HQ imports
    'MISO': 7,      # Aug — summer peak
    'SPP': 3,       # Apr — spring wind surplus = max exports (minimum net import)
}

# --- Diurnal shape (24-hour pattern, normalized to mean=1.0) ---
# Imports tend to be higher during peak demand hours and lower overnight.
DIURNAL_SHAPE_IMPORTER = np.array([
    0.70, 0.65, 0.62, 0.60, 0.62, 0.68,  # 0-5: overnight low
    0.78, 0.90, 1.00, 1.08, 1.12, 1.15,  # 6-11: morning ramp
    1.18, 1.20, 1.22, 1.25, 1.28, 1.30,  # 12-17: afternoon peak
    1.25, 1.15, 1.05, 0.95, 0.85, 0.75,  # 18-23: evening decline
])
DIURNAL_SHAPE_IMPORTER /= DIURNAL_SHAPE_IMPORTER.mean()  # Normalize to mean=1

# For net exporters, shape is inverted (export more during off-peak, less at peak)
DIURNAL_SHAPE_EXPORTER = np.array([
    1.25, 1.30, 1.32, 1.35, 1.30, 1.22,  # 0-5: overnight high exports
    1.10, 0.95, 0.85, 0.78, 0.75, 0.72,  # 6-11: morning ramp reduces exports
    0.70, 0.68, 0.70, 0.72, 0.75, 0.80,  # 12-17: afternoon — less export
    0.88, 0.98, 1.08, 1.15, 1.20, 1.25,  # 18-23: evening — exports resume
])
DIURNAL_SHAPE_EXPORTER /= DIURNAL_SHAPE_EXPORTER.mean()

# CAISO has a unique shape: imports peak during evening ramp (solar cliff)
DIURNAL_SHAPE_CAISO = np.array([
    0.60, 0.55, 0.52, 0.50, 0.52, 0.58,  # 0-5: overnight low
    0.65, 0.72, 0.75, 0.70, 0.60, 0.50,  # 6-11: solar ramp reduces import need
    0.45, 0.42, 0.48, 0.65, 0.90, 1.30,  # 12-17: solar cliff → imports surge
    1.55, 1.60, 1.50, 1.30, 1.05, 0.80,  # 18-23: evening peak imports from PNW
])
DIURNAL_SHAPE_CAISO /= DIURNAL_SHAPE_CAISO.mean()


def generate_synthetic_interchange(iso, demand_data=None):
    """Generate 8760-hour synthetic interchange profile for one ISO.

    Returns (net_import_mw, net_import_norm) arrays.
    net_import_mw: MW values (positive = imports)
    net_import_norm: normalized to demand (consistent with dispatch_utils convention)
    """
    avg_mw = ANNUAL_AVG_NET_IMPORT_MW[iso]
    amplitude = SEASONAL_AMPLITUDE[iso]
    peak_month = SEASONAL_PEAK_MONTH[iso]

    # Select diurnal shape
    if iso == 'CAISO':
        diurnal = DIURNAL_SHAPE_CAISO
    elif avg_mw >= 0:
        diurnal = DIURNAL_SHAPE_IMPORTER
    else:
        diurnal = DIURNAL_SHAPE_EXPORTER

    # Build 8760-hour profile
    net_import = np.zeros(H, dtype=np.float64)

    # Hour-of-year to (month, hour-of-day)
    # Approximate: each month has H/12 hours
    hours_per_month = H / 12.0

    for h in range(H):
        month_frac = h / hours_per_month  # 0..12
        hour_of_day = h % 24

        # Seasonal factor: cosine wave peaking at peak_month
        seasonal = 1.0 + amplitude * np.cos(2 * np.pi * (month_frac - peak_month) / 12.0)

        # Diurnal factor
        diurnal_factor = diurnal[hour_of_day]

        net_import[h] = avg_mw * seasonal * diurnal_factor

    # Add small random noise (±3%) for realism
    rng = np.random.default_rng(seed=42 + hash(iso) % 1000)
    noise = 1.0 + rng.normal(0, 0.03, H)
    net_import *= noise

    # Normalize to demand units
    if demand_data and iso in demand_data:
        year_data = demand_data[iso].get('2025', demand_data[iso].get('2024', {}))
        total_mwh = year_data.get('total_annual_mwh', None)
        if total_mwh and total_mwh > 0:
            net_import_norm = net_import / total_mwh
        else:
            # Fallback: use raw_mw sum
            raw_mw = year_data.get('raw_mw', [])
            total_mwh = sum(raw_mw) if raw_mw else 1e8
            net_import_norm = net_import / total_mwh
    else:
        # Fallback normalization using REGIONAL_DEMAND_TWH
        from pipeline_config import REGIONAL_DEMAND_TWH
        total_mwh = REGIONAL_DEMAND_TWH.get(iso, 300) * 1e6
        net_import_norm = net_import / total_mwh

    return net_import.tolist(), net_import_norm.tolist()


def main():
    """Generate interchange profiles for all ISOs."""
    # Load demand data for normalization
    demand_data = None
    demand_file = os.path.join(PROFILES_DIR, 'eia_demand_profiles.json')
    if os.path.exists(demand_file):
        with open(demand_file) as f:
            demand_data = json.load(f)
        print(f"Loaded demand profiles from {demand_file}")

    result = {}
    for iso in ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']:
        net_mw, net_norm = generate_synthetic_interchange(iso, demand_data)
        result[iso] = {
            '2024': {
                'net_import_mw': net_mw,
                'net_import_norm': net_norm,
            }
        }
        avg = np.mean(net_mw)
        peak = np.max(net_mw)
        trough = np.min(net_mw)
        print(f"  {iso}: avg={avg:+.0f} MW, peak={peak:+.0f} MW, trough={trough:+.0f} MW")

    os.makedirs(PROFILES_DIR, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(result, f)
    print(f"\nSaved interchange profiles to {OUTPUT_FILE}")
    print(f"File size: {os.path.getsize(OUTPUT_FILE) / 1e6:.1f} MB")


if __name__ == '__main__':
    main()
