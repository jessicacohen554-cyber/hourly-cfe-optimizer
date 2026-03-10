#!/usr/bin/env python3
"""
Parquet-based I/O for EIA 930 consolidated data files.

Reads from parquet files and returns the same nested dict structures
that the pipeline expects (matching the legacy JSON format).

Files:
  - eia_generation_profiles.parquet  (iso, year, fuel, hour, value)
  - eia_demand_profiles.parquet      (iso, year, hour, raw_mw, normalized)
  - eia_demand_meta.parquet          (iso, year, total_annual_mwh, peak_mw, min_mw, avg_mw)
  - eia_fossil_mix.parquet           (iso, year, hour, coal_share, gas_share, oil_share)
"""

import os
import json

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
EIA_DIR = os.path.join(DATA_DIR, 'eia-930')


def _try_parquet(name):
    """Check if parquet version exists."""
    path = os.path.join(EIA_DIR, name + '.parquet')
    return path if os.path.exists(path) else None


def _try_json(name):
    """Check if JSON version exists."""
    path = os.path.join(EIA_DIR, name + '.json')
    return path if os.path.exists(path) else None


def load_generation_profiles():
    """Load generation profiles → {iso: {year_str: {fuel: [8760 floats]}}}."""
    pq = _try_parquet('eia_generation_profiles')
    if pq:
        import pandas as pd
        df = pd.read_parquet(pq)
        result = {}
        for iso in df['iso'].unique():
            result[iso] = {}
            iso_df = df[df['iso'] == iso]
            for year in iso_df['year'].unique():
                year_df = iso_df[iso_df['year'] == year].sort_values('hour')
                result[iso][str(year)] = {}
                for fuel in year_df['fuel'].unique():
                    vals = year_df[year_df['fuel'] == fuel]['value'].tolist()
                    result[iso][str(year)][fuel] = vals
        return result

    jp = _try_json('eia_generation_profiles')
    if jp:
        with open(jp) as f:
            return json.load(f)

    raise FileNotFoundError(
        f"No generation profiles found. Checked:\n  {EIA_DIR}/eia_generation_profiles.parquet\n  {EIA_DIR}/eia_generation_profiles.json")


def load_demand_profiles():
    """Load demand profiles → {iso: {year_str: {raw_mw: [...], normalized: [...], total_annual_mwh, peak_mw, min_mw, avg_mw}}}."""
    pq = _try_parquet('eia_demand_profiles')
    meta_pq = _try_parquet('eia_demand_meta')
    if pq:
        import pandas as pd
        df = pd.read_parquet(pq)
        meta_df = pd.read_parquet(meta_pq) if meta_pq else None

        result = {}
        for iso in df['iso'].unique():
            result[iso] = {}
            iso_df = df[df['iso'] == iso]
            for year in iso_df['year'].unique():
                year_df = iso_df[iso_df['year'] == year].sort_values('hour')
                entry = {
                    'raw_mw': year_df['raw_mw'].tolist(),
                    'normalized': year_df['normalized'].tolist(),
                }
                # Add metadata
                if meta_df is not None:
                    meta_row = meta_df[(meta_df['iso'] == iso) & (meta_df['year'] == year)]
                    if len(meta_row) > 0:
                        row = meta_row.iloc[0]
                        entry['total_annual_mwh'] = float(row['total_annual_mwh'])
                        entry['peak_mw'] = float(row['peak_mw'])
                        entry['min_mw'] = float(row['min_mw'])
                        entry['avg_mw'] = float(row['avg_mw'])
                result[iso][str(year)] = entry
        return result

    jp = _try_json('eia_demand_profiles')
    if jp:
        with open(jp) as f:
            return json.load(f)

    raise FileNotFoundError(
        f"No demand profiles found. Checked:\n  {EIA_DIR}/eia_demand_profiles.parquet\n  {EIA_DIR}/eia_demand_profiles.json")


def load_fossil_mix():
    """Load fossil mix → {iso: {year_str: {hours: [...], coal_share: [...], gas_share: [...], oil_share: [...]}}}."""
    pq = _try_parquet('eia_fossil_mix')
    if pq:
        import pandas as pd
        df = pd.read_parquet(pq)
        result = {}
        for iso in df['iso'].unique():
            result[iso] = {}
            iso_df = df[df['iso'] == iso]
            for year in iso_df['year'].unique():
                year_df = iso_df[iso_df['year'] == year].sort_values('hour')
                result[iso][str(year)] = {
                    'hours': list(range(1, len(year_df) + 1)),
                    'coal_share': year_df['coal_share'].tolist(),
                    'gas_share': year_df['gas_share'].tolist(),
                    'oil_share': year_df['oil_share'].tolist(),
                }
        return result

    jp = _try_json('eia_fossil_mix')
    if jp:
        with open(jp) as f:
            return json.load(f)

    raise FileNotFoundError(
        f"No fossil mix found. Checked:\n  {EIA_DIR}/eia_fossil_mix.parquet\n  {EIA_DIR}/eia_fossil_mix.json")
