# Hybrid Integration — Steps 3 & 4 Session Prompts

These prompts continue the hybrid integration from `hybrid-session-prompts.md` (Sessions A–F). They cover Steps 3A, 3B, and all Step 4 scripts.

## Dependencies
```
Sessions A–E (complete) → Session G (dispatch_utils foundation)
                              → Sessions H, I (parallel: 3A cache, 3B MAC queue)
                                  → Sessions J, K (parallel: Step 4 analytics)
```

---

## Session G: Foundation — `dispatch_utils.py` Core Hybrid Dispatch

### Task
Update the core dispatch engine in `dispatch_utils.py` so that `reconstruct_hourly_dispatch()`, `_compute_per_resource_dispatch()`, `_archetype_key()`, and `get_supply_profiles()` all support hybrid resource types. This is the foundation that Steps 3A, 3B, and all Step 4 scripts depend on.

### Branch
Develop on branch `claude/integrate-hybrid-resources-2pk5J`

### Prerequisites
Session A must be complete (pipeline_config.py has `HYBRID_TYPES`, `HYBRID_DC_AC_RATIOS`, `get_resource_cols()`, etc.).

### Background

**Current state of dispatch_utils.py hybrid support:**
- `RESOURCE_TYPES_HYBRID` is defined (line 75) as `RESOURCE_TYPES + HYBRID_TYPES`
- `build_supply_matrix()` (line 605) already accepts an optional `resource_types` parameter
- But `reconstruct_hourly_dispatch()` (line 669) **hardcodes `RESOURCE_TYPES`** when building `mix_weights` from the supply_matrix
- `_compute_per_resource_dispatch()` (line 558) **hardcodes `RESOURCE_TYPES`** for the per-resource arrays
- `_archetype_key()` (line 925) **hardcodes 6 base resources** in the hash parts
- `get_supply_profiles()` (line 115) only returns the 6 base resource profiles — no hybrid profiles

**Why this matters:** Even though `build_supply_matrix()` can accept hybrid resource types, the dispatch functions that consume it ignore the extra columns. Hybrid generation would be invisible to the dispatch engine, causing undercount of clean supply and incorrect storage sizing.

**Design principle:** Hybrid profiles are pre-computed 8760-hour shapes (stored in `data/hybrid_profiles/{ISO}_hybrid_profiles.npz`). They represent the combined output of the renewable + co-located battery after internal dispatch. The dispatch engine treats them identically to any other renewable profile — no special storage logic needed because the co-located battery dispatch is already baked into the profile.

### Files to Modify

**`scripts/dispatch_utils.py`** (~1234 lines):

#### 1. Update `get_supply_profiles()` (line 115) — Add hybrid profile loading

Currently returns profiles for 6 base resources only. Add optional hybrid profile loading:

```python
def get_supply_profiles(iso, gen_profiles, include_hybrids=False, hybrid_profiles=None):
    """Get generation shape profiles with nuclear seasonal derate.

    Args:
        include_hybrids: If True, include hybrid co-located profiles.
        hybrid_profiles: Pre-loaded hybrid profile dict from load_hybrid_profiles().
            If None and include_hybrids=True, loads from data/hybrid_profiles/.
    """
    profiles = {}
    # ... existing base resource profile logic (unchanged) ...

    if include_hybrids:
        if hybrid_profiles is None:
            hybrid_profiles = _load_hybrid_profiles(iso)
        for htype in HYBRID_TYPES:
            if htype in hybrid_profiles:
                # Hybrid profiles are already normalized (sum=1.0 over 8760h)
                profiles[htype] = hybrid_profiles[htype]
            else:
                profiles[htype] = np.zeros(H, dtype=np.float64)

    return profiles
```

Add a private helper to load hybrid NPZ files:
```python
def _load_hybrid_profiles(iso):
    """Load pre-computed hybrid 8760 profiles from data/hybrid_profiles/."""
    npz_path = os.path.join(DATA_DIR, 'hybrid_profiles', f'{iso}_hybrid_profiles.npz')
    if not os.path.exists(npz_path):
        print(f"  WARNING: No hybrid profiles at {npz_path}")
        return {}
    data = np.load(npz_path)
    result = {}
    for htype in HYBRID_TYPES:
        key = htype  # or htype + '_profile' — check actual NPZ key names
        if key in data:
            arr = data[key].astype(np.float64)
            # Normalize to sum=1.0 (same convention as base profiles)
            total = arr.sum()
            if total > 0:
                result[htype] = arr / total
            else:
                result[htype] = np.zeros(H, dtype=np.float64)
    return result
```

**IMPORTANT:** Before implementing, read the actual NPZ files to check key names:
```python
python -c "import numpy as np; d = np.load('data/hybrid_profiles/CAISO_hybrid_profiles.npz'); print(list(d.keys()))"
```

#### 2. Update `reconstruct_hourly_dispatch()` (line 625) — Dynamic resource types

**Current code (line 668-674):**
```python
if supply_matrix is not None:
    mix_weights = np.array([resource_pcts.get(rt, 0) / 100.0 for rt in RESOURCE_TYPES],
                           dtype=np.float64)
    supply_total = procurement_factor * (mix_weights @ supply_matrix)
    ccs_idx = RESOURCE_TYPES.index('ccs_ccgt')
```

**Problem:** Uses `RESOURCE_TYPES` (6 elements) even when `supply_matrix` has 10 rows (6 base + 4 hybrid). The matrix multiply produces wrong results because the weight vector is too short.

**Fix:** Accept an optional `resource_types` parameter that defaults to `RESOURCE_TYPES`. When a wider supply_matrix is passed (from hybrid-aware callers), the matching resource_types list must also be passed:

```python
def reconstruct_hourly_dispatch(demand_norm, supply_profiles, resource_pcts,
                                 procurement_pct=100, battery_dispatch_pct=0,
                                 battery8_dispatch_pct=0, ldes_dispatch_pct=0,
                                 supply_matrix=None, detailed=False,
                                 h2_dispatch_pct=0, resource_types=None):
    """...(existing docstring + note about resource_types)..."""
    rtypes = resource_types if resource_types is not None else RESOURCE_TYPES
    procurement_factor = procurement_pct / 100.0
    demand_arr = np.array(demand_norm[:H], dtype=np.float64)

    supply_total = np.zeros(H, dtype=np.float64)
    ccs_supply = np.zeros(H, dtype=np.float64)

    if supply_matrix is not None:
        mix_weights = np.array([resource_pcts.get(rt, 0) / 100.0 for rt in rtypes],
                               dtype=np.float64)
        supply_total = procurement_factor * (mix_weights @ supply_matrix)
        if 'ccs_ccgt' in rtypes:
            ccs_idx = rtypes.index('ccs_ccgt')
            if mix_weights[ccs_idx] > 0:
                ccs_supply = procurement_factor * mix_weights[ccs_idx] * supply_matrix[ccs_idx]
    else:
        for rtype in rtypes:
            pct = resource_pcts.get(rtype, 0)
            if pct <= 0:
                continue
            profile = np.array(supply_profiles[rtype][:H], dtype=np.float64)
            contribution = procurement_factor * (pct / 100.0) * profile
            supply_total += contribution
            if rtype == 'ccs_ccgt':
                ccs_supply = contribution.copy()
    # ... rest of function unchanged (storage dispatch, result dict) ...
```

Also update the `detailed` block (line 738-742) to pass `rtypes`:
```python
if detailed:
    matched, surplus = _compute_per_resource_dispatch(
        demand_arr, supply_profiles, resource_pcts, procurement_factor,
        supply_matrix, resource_types=rtypes)
    for rtype in rtypes:
        result[f'matched_{rtype}'] = matched[rtype]
        result[f'surplus_{rtype}'] = surplus[rtype]
```

#### 3. Update `_compute_per_resource_dispatch()` (line 544) — Dynamic resource types

**Current code (line 558):**
```python
resource_pcts_arr = np.array([resource_pcts.get(rt, 0) for rt in RESOURCE_TYPES],
                              dtype=np.float64)
```

**Fix:** Accept `resource_types` parameter:
```python
def _compute_per_resource_dispatch(demand_arr, supply_profiles, resource_pcts,
                                    procurement_factor, supply_matrix=None,
                                    resource_types=None):
    rtypes = resource_types if resource_types is not None else RESOURCE_TYPES

    if supply_matrix is None:
        supply_matrix = build_supply_matrix(supply_profiles, resource_types=rtypes)

    resource_pcts_arr = np.array([resource_pcts.get(rt, 0) for rt in rtypes],
                                  dtype=np.float64)

    # Update dispatch order indices for the current rtypes list
    dispatch_order_indices = _get_dispatch_order_indices(rtypes)

    matched_arr, surplus_arr = _per_resource_dispatch_njit(
        demand_arr, supply_matrix, resource_pcts_arr,
        procurement_factor, dispatch_order_indices)

    matched = {rtypes[i]: matched_arr[i] for i in range(len(rtypes))}
    surplus = {rtypes[i]: surplus_arr[i] for i in range(len(rtypes))}
    return matched, surplus
```

**Dispatch order for hybrids:** The existing `_DISPATCH_ORDER_INDICES` maps the merit-order (CF → CCS → hydro → wind → solar) to array indices. Hybrid resources should dispatch in the same tier as their parent renewable:
- `solar_batt4`, `solar_batt8` → dispatch with solar (lowest priority, curtailed first)
- `wind_batt4`, `wind_batt8` → dispatch with wind

Add a helper function:
```python
def _get_dispatch_order_indices(rtypes):
    """Build dispatch order index array for given resource types.

    Merit order: clean_firm → ccs_ccgt → geothermal → hydro → offshore_wind →
                 wind → wind_batt4 → wind_batt8 → solar → solar_batt4 → solar_batt8
    """
    order = ['clean_firm', 'ccs_ccgt', 'geothermal', 'hydro', 'offshore_wind',
             'wind', 'wind_batt4', 'wind_batt8',
             'solar', 'solar_batt4', 'solar_batt8']
    indices = []
    for rtype in order:
        if rtype in rtypes:
            indices.append(rtypes.index(rtype))
    # Add any remaining types not in the explicit order
    for i, rtype in enumerate(rtypes):
        if i not in indices:
            indices.append(i)
    return np.array(indices, dtype=np.int64)
```

**Check `_DISPATCH_ORDER_INDICES`**: Read the existing definition (search for it near line 67-73 or wherever it's defined) to understand the current format, then extend it.

#### 4. Update `_archetype_key()` (line 925) — Include hybrid resources in hash

**Current code:**
```python
parts = [
    iso,
    str(float(resource_pcts.get('clean_firm', 0))),
    str(float(resource_pcts.get('solar', 0))),
    str(float(resource_pcts.get('wind', 0))),
    str(float(resource_pcts.get('offshore_wind', 0))),
    str(float(resource_pcts.get('ccs_ccgt', 0))),
    str(float(resource_pcts.get('hydro', 0))),
    str(float(procurement_pct)),
    str(float(battery_dispatch_pct)),
    str(float(battery8_dispatch_pct)),
    str(float(ldes_dispatch_pct)),
]
```

**Fix:** Add hybrid resource pcts to the hash. Use `.get(key, 0)` so non-hybrid mixes produce the same hash as before (backward compatible):
```python
parts = [
    iso,
    str(float(resource_pcts.get('clean_firm', 0))),
    str(float(resource_pcts.get('solar', 0))),
    str(float(resource_pcts.get('wind', 0))),
    str(float(resource_pcts.get('offshore_wind', 0))),
    str(float(resource_pcts.get('ccs_ccgt', 0))),
    str(float(resource_pcts.get('hydro', 0))),
    # Hybrid resources (0 for non-hybrid mixes — backward compatible)
    str(float(resource_pcts.get('solar_batt4', 0))),
    str(float(resource_pcts.get('solar_batt8', 0))),
    str(float(resource_pcts.get('wind_batt4', 0))),
    str(float(resource_pcts.get('wind_batt8', 0))),
    str(float(procurement_pct)),
    str(float(battery_dispatch_pct)),
    str(float(battery8_dispatch_pct)),
    str(float(ldes_dispatch_pct)),
]
```

**CRITICAL — Backward compatibility concern:** Adding new parts changes the hash for ALL mixes (even non-hybrid ones with 0 hybrid pcts), because the string `|0.0|0.0|0.0|0.0` is now inserted before the storage parts. This means **existing dispatch caches become invalid**.

Two options:
- **Option A (recommended):** Only append hybrid parts when any hybrid pct > 0. Non-hybrid mixes produce identical hashes to before. Hybrid mixes get unique hashes.
  ```python
  # After base parts...
  has_hybrids = any(resource_pcts.get(ht, 0) > 0 for ht in HYBRID_TYPES)
  if has_hybrids:
      for ht in HYBRID_TYPES:
          parts.append(str(float(resource_pcts.get(ht, 0))))
  # Then storage parts...
  ```
- **Option B:** Bump `CACHE_VERSION` and regenerate all caches. Cleaner but requires full re-run.

**Ask the user** which approach they prefer. Present both with pros/cons.

### Verification
1. Import check: `python -c "from dispatch_utils import reconstruct_hourly_dispatch, get_supply_profiles, _archetype_key, RESOURCE_TYPES_HYBRID"`
2. Profile loading: `python -c "from dispatch_utils import get_supply_profiles, load_common_data; d,g,_,_ = load_common_data(); p = get_supply_profiles('CAISO', g, include_hybrids=True); print([k for k in p.keys()])"`
3. Archetype key backward compat: verify that `_archetype_key('CAISO', {'clean_firm': 10, 'solar': 20, 'wind': 30, 'offshore_wind': 0, 'ccs_ccgt': 5, 'hydro': 10}, 100, 5, 0, 0)` produces the same hash as before (if using Option A).
4. Dispatch test: build a supply_matrix with hybrid profiles and verify `reconstruct_hourly_dispatch()` uses all 10 rows.

### Commit
Commit with message: "Add hybrid resource support to dispatch_utils core functions — dynamic resource types in dispatch, archetype key, and supply profiles"

---

## Session H: Step 3A — Dispatch Cache Hybrid Integration

### Task
Update `step3a_build_dispatch_cache.py` to extract, dispatch, cache, and manifest hybrid resource columns. After this session, the dispatch cache will contain per-resource matched/surplus profiles for all 10 resource types (6 base + 4 hybrid), and all downstream Step 4 scripts can consume hybrid data.

### Branch
Develop on branch `claude/integrate-hybrid-resources-2pk5J`

### Prerequisites
- Session A complete (pipeline_config.py hybrid constants)
- Session G complete (dispatch_utils.py core hybrid dispatch)

### Background

**What step3a does:** Reads unique resource mixes from Step 2.2 parquets, runs `reconstruct_hourly_dispatch(detailed=True)` for each, and saves the 8760-hour per-resource profiles to a parquet-based dispatch cache. It also enriches step3 parquets with actual dispatch share columns and builds an annual manifest.

**What needs to change:** The script has hardcoded `MIX_COLUMNS` (line 54-58), hardcoded resource extraction in `extract_unique_mixes()` (lines 86-103), and hardcoded resource loops in `enrich_parquets_with_dispatch_shares()` and `build_annual_manifest()`. All of these need to dynamically include hybrid columns when present.

**Key design point:** Hybrid detection is automatic — if the Step 2.2 input parquet has `solar_batt4` columns, the script runs in hybrid mode. No CLI flag needed.

### Files to Modify

**`scripts/step3a_build_dispatch_cache.py`** (564 lines):

#### 1. Update imports (line 44-51)

Add hybrid-aware imports:
```python
from dispatch_utils import (
    ISOS, RESOURCE_TYPES, RESOURCE_TYPES_HYBRID, CACHE_VERSION, H,
    DISPATCH_CACHE_DIR, HYBRID_TYPES,
    load_common_data, get_supply_profiles, get_demand_profile,
    build_supply_matrix, reconstruct_hourly_dispatch,
    _archetype_key, load_dispatch_cache, save_dispatch_cache,
)
```

#### 2. Update `MIX_COLUMNS` (line 54-58)

Currently hardcoded to 6 base resources + 4 storage columns. Make it dynamic:
```python
MIX_COLUMNS_BASE = [
    'mix_clean_firm', 'mix_solar', 'mix_wind', 'mix_offshore_wind', 'mix_ccs_ccgt', 'mix_hydro',
    'battery_dispatch_pct', 'battery8_dispatch_pct',
    'ldes_dispatch_pct', 'h2_dispatch_pct',
]
MIX_COLUMNS_HYBRID = [f'mix_{ht}' for ht in HYBRID_TYPES]
# e.g. ['mix_solar_batt4', 'mix_solar_batt8', 'mix_wind_batt4', 'mix_wind_batt8']
```

**IMPORTANT:** Check what column naming convention Step 2.2 uses for hybrid columns. It might be `mix_solar_batt4` (with `mix_` prefix) or `solar_batt4` (without prefix). Read a Step 2.2 output parquet schema to confirm:
```python
python -c "import pyarrow.parquet as pq; s = pq.read_schema('data/step2.2-cost/step_2_2a_CAISO.parquet'); print([c for c in s.names if 'batt' in c or 'hybrid' in c])"
```

#### 3. Update `extract_unique_mixes()` (line 61-111)

Currently extracts 6 hardcoded resource columns into a `resource_pcts` dict. Extend to detect and include hybrid columns:

```python
def extract_unique_mixes(iso, input_dir):
    path = find_parquet(input_dir, iso)
    if not path:
        return [], False  # Return has_hybrids flag

    avail_cols = pq.read_schema(path).names

    # Detect hybrid columns
    has_hybrids = any(f'mix_{ht}' in avail_cols or ht in avail_cols for ht in HYBRID_TYPES)

    # Build column list dynamically
    mix_cols = list(MIX_COLUMNS_BASE)
    if has_hybrids:
        mix_cols.extend(MIX_COLUMNS_HYBRID)

    read_cols = [c for c in mix_cols if c in avail_cols]
    df = pd.read_parquet(path, columns=read_cols)
    for c in mix_cols:
        if c not in df.columns:
            df[c] = 0
    unique = df.drop_duplicates()

    # Vectorized extraction — base resources
    cf = unique['mix_clean_firm'].to_numpy(dtype=np.float64)
    sol = unique['mix_solar'].to_numpy(dtype=np.float64)
    wnd = unique['mix_wind'].to_numpy(dtype=np.float64)
    osw = unique['mix_offshore_wind'].to_numpy(dtype=np.float64)
    ccs = unique['mix_ccs_ccgt'].to_numpy(dtype=np.float64)
    hyd = unique['mix_hydro'].to_numpy(dtype=np.float64)
    bat = unique['battery_dispatch_pct'].to_numpy(dtype=np.float64)
    bat8 = unique['battery8_dispatch_pct'].to_numpy(dtype=np.float64)
    ldes = unique['ldes_dispatch_pct'].to_numpy(dtype=np.float64)
    h2 = unique['h2_dispatch_pct'].to_numpy(dtype=np.float64)

    # Hybrid columns (zeros if not present)
    hybrid_arrs = {}
    if has_hybrids:
        for ht in HYBRID_TYPES:
            col = f'mix_{ht}' if f'mix_{ht}' in unique.columns else ht
            hybrid_arrs[ht] = unique[col].to_numpy(dtype=np.float64) if col in unique.columns else np.zeros(len(unique))

    n = len(unique)
    mixes = [None] * n
    for i in range(n):
        rp = {
            'clean_firm': cf[i], 'solar': sol[i], 'wind': wnd[i],
            'offshore_wind': osw[i], 'ccs_ccgt': ccs[i], 'hydro': hyd[i],
        }
        if has_hybrids:
            for ht in HYBRID_TYPES:
                rp[ht] = hybrid_arrs[ht][i]

        mixes[i] = {
            'resource_pcts': rp,
            'battery_dispatch_pct': bat[i],
            'battery8_dispatch_pct': bat8[i],
            'ldes_dispatch_pct': ldes[i],
            'h2_dispatch_pct': h2[i],
        }

    return mixes, has_hybrids
```

**Note:** The return signature changes from `list` to `(list, bool)`. Update all callers of `extract_unique_mixes()`.

#### 4. Update `build_cache_for_iso()` (line 114-166)

Pass hybrid-aware supply profiles and resource types to the dispatch engine:

```python
def build_cache_for_iso(iso, unique_mixes, demand_data, gen_profiles,
                         existing_cache=None, force=False, has_hybrids=False):
    rtypes = RESOURCE_TYPES_HYBRID if has_hybrids else RESOURCE_TYPES
    supply_profiles = get_supply_profiles(iso, gen_profiles, include_hybrids=has_hybrids)
    supply_matrix = build_supply_matrix(supply_profiles, resource_types=rtypes)
    demand_norm, total_mwh = get_demand_profile(iso, demand_data)

    cache = {}
    computed = 0
    skipped = 0

    for mix_info in unique_mixes:
        rp = mix_info['resource_pcts']
        key = _archetype_key(
            iso, rp, 100,
            mix_info['battery_dispatch_pct'],
            mix_info['battery8_dispatch_pct'],
            mix_info['ldes_dispatch_pct'],
        )

        if not force and key in cache:
            skipped += 1
            continue

        result = reconstruct_hourly_dispatch(
            demand_norm, supply_profiles, rp,
            100,
            mix_info['battery_dispatch_pct'],
            mix_info['battery8_dispatch_pct'],
            mix_info['ldes_dispatch_pct'],
            supply_matrix=supply_matrix,
            detailed=True,
            h2_dispatch_pct=mix_info['h2_dispatch_pct'],
            resource_types=rtypes,  # NEW — pass dynamic resource types
        )

        cache[key] = {k: v for k, v in result.items()}
        computed += 1

    return cache, computed, skipped
```

#### 5. Update `enrich_parquets_with_dispatch_shares()` (line 169-261)

Add hybrid resource columns to the archetype key construction. The function builds `rp` dicts from parquet columns — these need hybrid entries:

In the loop (line 210-213), extend the `rp` dict:
```python
rp = {
    'clean_firm': cf[i], 'solar': sol[i], 'wind': wnd[i],
    'offshore_wind': osw[i], 'ccs_ccgt': ccs[i], 'hydro': hyd[i],
}
if has_hybrids:
    for ht in HYBRID_TYPES:
        rp[ht] = hybrid_col_arrays[ht][i]
```

Also extract hybrid column arrays at the top of the function (after line 202):
```python
hybrid_col_arrays = {}
if has_hybrids:
    for ht in HYBRID_TYPES:
        col = f'mix_{ht}' if f'mix_{ht}' in df.columns else ht
        hybrid_col_arrays[ht] = df[col].to_numpy(dtype=np.float64) if col in df.columns else np.zeros(n)
```

The `has_hybrids` parameter needs to be passed in from `main()`.

#### 6. Update `_enrich_single_parquet()` (line 264-322)

Same pattern as `enrich_parquets_with_dispatch_shares()` — extend the `rp` dict with hybrid columns when present:

```python
# Detect hybrids in this parquet
has_hybrids_local = any(ht in df.columns or f'mix_{ht}' in df.columns for ht in HYBRID_TYPES)
hybrid_col_arrays = {}
if has_hybrids_local:
    for ht in HYBRID_TYPES:
        col = f'mix_{ht}' if f'mix_{ht}' in df.columns else (ht if ht in df.columns else None)
        hybrid_col_arrays[ht] = df[col].to_numpy(dtype=np.float64) if col else np.zeros(n)
```

Then in the per-row loop, extend `rp` with hybrid values.

#### 7. Update `build_annual_manifest()` (line 325-415)

Two changes needed:

**a) Add hybrid resource columns to manifest rows (line 353-358):**
```python
# Existing base resources
row['mix_clean_firm'] = rp.get('clean_firm', 0)
row['mix_solar'] = rp.get('solar', 0)
# ... etc ...

# Hybrid resources (if present)
for ht in HYBRID_TYPES:
    if rp.get(ht, 0) > 0 or has_hybrids:
        row[f'mix_{ht}'] = rp.get(ht, 0)
```

**b) Add hybrid dispatch/surplus to the resource loop (line 370-374):**
```python
resources_to_iterate = ['clean_firm', 'ccs_ccgt', 'solar', 'wind', 'offshore_wind', 'hydro']
if has_hybrids:
    resources_to_iterate.extend(HYBRID_TYPES)

for resource in resources_to_iterate:
    matched = entry.get(f'matched_{resource}', np.zeros(H))
    surplus = entry.get(f'surplus_{resource}', np.zeros(H))
    row[f'{resource}_dispatch_pct'] = float(np.sum(matched)) * 100
    row[f'{resource}_surplus_pct'] = float(np.sum(surplus)) * 100
```

#### 8. Update `main()` (line 481-564)

Thread `has_hybrids` through the call chain:

```python
# In the per-ISO loop:
unique_mixes, has_hybrids = extract_unique_mixes(iso, input_dir)  # Updated return
if has_hybrids:
    print(f"    Hybrid mode: detected hybrid resource columns")

# Pass to build_cache_for_iso
cache, computed, skipped = build_cache_for_iso(
    iso, unique_mixes, demand_data, gen_profiles,
    force=True, has_hybrids=has_hybrids)

# Pass to enrich
enrich_parquets_with_dispatch_shares(iso, input_dir, cache, has_hybrids=has_hybrids)

# Pass to manifest
build_annual_manifest(iso, unique_mixes, cache, has_hybrids=has_hybrids)
```

Update function signatures to accept `has_hybrids` where needed.

### Critical Pitfalls

1. **Column naming convention**: Step 2.2 parquets may use `mix_solar_batt4` or `solar_batt4`. Check both. The code must handle either convention gracefully.

2. **Dispatch cache format**: `save_dispatch_cache()` in dispatch_utils stores dict entries as list columns in parquet. The new `matched_solar_batt4`, `surplus_solar_batt4`, etc. keys will automatically be saved IF they appear in the result dict. Verify `save_dispatch_cache()` iterates over all dict keys (not a hardcoded list).

3. **Non-hybrid backward compatibility**: When no hybrid columns exist in input, the script must produce identical output to before. Zero hybrid pcts should not change archetype keys (see Session G Option A).

4. **NPZ profile normalization**: Hybrid profiles from `data/hybrid_profiles/` must be normalized to sum=1.0 (same as base profiles). If `get_supply_profiles()` handles this in Session G, no extra work needed here.

### Verification
1. Run on one ISO: `python scripts/step3a_build_dispatch_cache.py --iso SPP`
2. Check cache has hybrid entries: `python -c "from dispatch_utils import load_dispatch_cache; c = load_dispatch_cache('SPP'); k = list(c.keys())[0]; print([x for x in c[k].keys() if 'batt' in x])"`
3. Check manifest has hybrid columns: `python -c "import pandas as pd; df = pd.read_parquet('data/step3-dispatch/SPP_annual_manifest.parquet'); print([c for c in df.columns if 'batt' in c and 'battery' not in c])"`
4. Verify non-hybrid ISOs still work: run on an ISO without hybrid columns and check output is unchanged.

### Commit
Commit with message: "Add hybrid resource support to Step 3A dispatch cache — extract, dispatch, cache, and manifest hybrid columns"
