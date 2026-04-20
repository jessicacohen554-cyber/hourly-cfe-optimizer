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

# Centralized path resolution — supports frozen (PyInstaller) and dev modes
try:
    from paths import MODULE_ROOT, get_data_search_dirs, resolve_data_path
    _DATA_SEARCH_DIRS = get_data_search_dirs(["profiles", "eia-930"])
except ImportError:
    MODULE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _DATA_SEARCH_DIRS = None

    def resolve_data_path(rel):
        from pathlib import Path
        return Path(MODULE_ROOT) / "data" / rel

DATA_DIR = os.path.join(str(MODULE_ROOT), 'data')

# Search local directories for EIA data
# 1. market-simulator/data/profiles/ (synthetic or generated)
# 2. market-simulator/data/eia-930/ (standard EIA 930 location)
if _DATA_SEARCH_DIRS:
    EIA_SEARCH_DIRS = [str(d) for d in _DATA_SEARCH_DIRS]
else:
    EIA_SEARCH_DIRS = [
        os.path.join(DATA_DIR, 'profiles'),
        os.path.join(DATA_DIR, 'eia-930'),
    ]
EIA_DIR = os.path.join(DATA_DIR, 'eia-930')  # Default for error messages


def _try_parquet(name):
    """Check if parquet version exists in any search directory."""
    for search_dir in EIA_SEARCH_DIRS:
        path = os.path.join(search_dir, name + '.parquet')
        if os.path.exists(path):
            return path
    return None


def _try_json(name):
    """Check if JSON version exists in any search directory."""
    for search_dir in EIA_SEARCH_DIRS:
        path = os.path.join(search_dir, name + '.json')
        if os.path.exists(path):
            return path
    return None


def load_generation_profiles():
    """Load generation profiles → {iso: {year_str: {fuel: [8760 floats]}}}."""
    pq = _try_parquet('eia_generation_profiles')
    if pq:
        import pandas as pd
        print(f"[eia-io] load_generation_profiles: reading {pq}", flush=True)
        df = pd.read_parquet(pq)
        df = df.sort_values(['iso', 'year', 'fuel', 'hour'])
        series = df.groupby(['iso', 'year', 'fuel'])['value'].apply(lambda x: x.tolist())
        result = {}
        for (iso, year, fuel), vals in series.items():
            result.setdefault(iso, {}).setdefault(str(year), {})[fuel] = vals
        print("[eia-io] load_generation_profiles: done", flush=True)
        return result

    jp = _try_json('eia_generation_profiles')
    if jp is None:
        jp = _try_json('eia_gen_profiles')  # synthetic profile short name
    if jp:
        with open(jp) as f:
            return json.load(f)

    raise FileNotFoundError(
        f"No generation profiles found. Checked:\n  {EIA_DIR}/eia_generation_profiles.parquet\n  {EIA_DIR}/eia_generation_profiles.json\n  {EIA_DIR}/eia_gen_profiles.json")


def load_demand_profiles():
    """Load demand profiles → {iso: {year_str: {raw_mw: [...], normalized: [...], total_annual_mwh, peak_mw, min_mw, avg_mw}}}."""
    pq = _try_parquet('eia_demand_profiles')
    meta_pq = _try_parquet('eia_demand_meta')
    if pq:
        import pandas as pd
        print(f"[eia-io] load_demand_profiles: reading {pq}", flush=True)
        df = pd.read_parquet(pq)
        meta_df = pd.read_parquet(meta_pq) if meta_pq else None

        df = df.sort_values(['iso', 'year', 'hour'])
        raw_series = df.groupby(['iso', 'year'])['raw_mw'].apply(lambda x: x.tolist())
        norm_series = df.groupby(['iso', 'year'])['normalized'].apply(lambda x: x.tolist())
        if meta_df is not None:
            meta_df = meta_df.set_index(['iso', 'year'])
        result = {}
        for (iso, year), raw_mw in raw_series.items():
            entry = {
                'raw_mw': raw_mw,
                'normalized': norm_series[(iso, year)],
            }
            if meta_df is not None and (iso, year) in meta_df.index:
                row = meta_df.loc[(iso, year)]
                entry['total_annual_mwh'] = float(row['total_annual_mwh'])
                entry['peak_mw'] = float(row['peak_mw'])
                entry['min_mw'] = float(row['min_mw'])
                entry['avg_mw'] = float(row['avg_mw'])
            result.setdefault(iso, {})[str(year)] = entry
        print("[eia-io] load_demand_profiles: done", flush=True)
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
        print(f"[eia-io] load_fossil_mix: reading {pq}", flush=True)
        df = pd.read_parquet(pq)
        df = df.sort_values(['iso', 'year', 'hour'])
        coal = df.groupby(['iso', 'year'])['coal_share'].apply(lambda x: x.tolist())
        gas = df.groupby(['iso', 'year'])['gas_share'].apply(lambda x: x.tolist())
        oil = df.groupby(['iso', 'year'])['oil_share'].apply(lambda x: x.tolist())
        result = {}
        for (iso, year), coal_vals in coal.items():
            n = len(coal_vals)
            result.setdefault(iso, {})[str(year)] = {
                'hours': list(range(1, n + 1)),
                'coal_share': coal_vals,
                'gas_share': gas[(iso, year)],
                'oil_share': oil[(iso, year)],
            }
        print("[eia-io] load_fossil_mix: done", flush=True)
        return result

    jp = _try_json('eia_fossil_mix')
    if jp:
        with open(jp) as f:
            return json.load(f)

    raise FileNotFoundError(
        f"No fossil mix found. Checked:\n  {EIA_DIR}/eia_fossil_mix.parquet\n  {EIA_DIR}/eia_fossil_mix.json")


def load_interchange_profiles():
    """Load interchange profiles → {iso: {year_str: {net_import_mw: [...], net_import_norm: [...]}}}.

    Returns empty dict (not an error) if no data found — backward compatible
    with copper-plate (no interchange) behavior.
    """
    pq = _try_parquet('eia_interchange_profiles')
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
                    'net_import_mw': year_df['net_import_mw'].tolist(),
                    'net_import_norm': year_df['net_import_norm'].tolist(),
                }
        return result

    jp = _try_json('eia_interchange_profiles')
    if jp:
        with open(jp) as f:
            return json.load(f)

    # No pre-built file → generate on first load
    try:
        from step0_fetch_interchange import main as generate_interchange
        generate_interchange()
        # Retry after generation
        jp = _try_json('eia_interchange_profiles')
        if jp:
            with open(jp) as f:
                return json.load(f)
    except Exception:
        pass

    # No interchange data → return empty dict (copper-plate fallback)
    return {}
