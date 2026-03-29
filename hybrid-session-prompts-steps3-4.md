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

---

## Session I: Step 3B — MAC Queue Hybrid Integration

### Task
Update `step3b_mac_queue.py` to include hybrid co-located resources in the path-dependent MAC deployment queue. This means hybrid resources can be selected as part of cost-optimal mixes at each threshold, with proper LCOE pricing using component-additive costs (Session A constants).

### Branch
Develop on branch `claude/integrate-hybrid-resources-2pk5J`

### Prerequisites
- Session A complete (pipeline_config.py: `HYBRID_TYPES`, `get_hybrid_lcoe()`, `get_hybrid_tx()`)
- Session G complete (dispatch_utils.py: hybrid-aware `reconstruct_hourly_dispatch()`, `get_supply_profiles()`)

### Background

**What step3b does:** Path-dependent deployment queue that finds the cheapest $/mtCO₂ avoided at each SBTi threshold. For each ISO × price sensitivity × demand growth, it:
1. Loads PFS archetypes from Step 1 parquets
2. Filters to mixes that respect the ratcheting floor
3. Scores MAC = new_build_cost / CO₂_avoided for all candidates
4. Picks the winner, locks its resources as the new floor, repeats

**What needs to change:**
- `RESOURCE_COLS` (line 113) is hardcoded to 6 base resources — must include hybrids
- `compute_new_build_cost()` (line 325) has per-resource cost blocks for solar, wind, offshore, etc. — needs hybrid cost blocks using `get_hybrid_lcoe()` + `get_hybrid_tx()`
- CCS residual calculation (line 467) sums only base resources — must include hybrids
- `load_pfs_for_threshold()` (line 216) reads PFS parquets with `MIX_COLS` — must include hybrid columns
- `batch_score_mixes()` (line 1159) has vectorized cost for base resources — needs hybrid cost arrays
- `filter_and_sample()` (line 663) applies floor constraints — must handle hybrid resource floors
- Supply matrix construction in `run_pathway()` / `run_iso()` — must include hybrid profiles
- `_archetype_key()` calls in CO₂ dispatch — already handled by Session G

**Key design point:** All hybrid procurement is new-build. There is no "existing hybrid" capacity in the 2025 snapshot. Hybrid resources compete with standalone renewables on a levelized cost basis — the optimizer picks whichever mix minimizes MAC.

### Files to Modify

**`scripts/step3b_mac_queue.py`** (~2227 lines):

#### 1. Update `RESOURCE_COLS` and `MIX_COLS` (line 113-115)

```python
from pipeline_config import HYBRID_TYPES
# ...
RESOURCE_COLS = ['clean_firm', 'solar', 'wind', 'hydro', 'offshore_wind', 'geothermal']
HYBRID_COLS = list(HYBRID_TYPES)  # ['solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8']
STORAGE_COLS = ['battery_dispatch_pct', 'battery8_dispatch_pct', 'ldes_dispatch_pct', 'h2_dispatch_pct']
MIX_COLS = RESOURCE_COLS + HYBRID_COLS + STORAGE_COLS + ['hourly_match_score']
```

**Note:** `RESOURCE_COLS` stays as-is (used for base resource iteration). `HYBRID_COLS` is separate. `MIX_COLS` is the superset used for parquet I/O.

#### 2. Update `load_pfs_for_threshold()` (line 216-299)

The function loads PFS parquets and selects columns via `MIX_COLS`. Since `MIX_COLS` now includes hybrid columns, they'll be loaded automatically. The key fix is handling the "fill missing columns with 0" logic (line 288-290):

```python
# Fill missing columns with 0 (includes hybrid cols if not in parquet)
for col in MIX_COLS:
    if col not in combined.columns:
        combined[col] = 0.0
```

This already works — hybrid columns will be filled with 0 if the PFS parquet doesn't have them (pre-hybrid data). **No code change needed here if MIX_COLS is updated.**

Also update the dedup key (line 293):
```python
dedup_cols = RESOURCE_COLS + HYBRID_COLS + STORAGE_COLS
```

#### 3. Update `compute_new_build_cost()` (line 325-530)

Add hybrid resource cost blocks after the existing solar/wind/offshore blocks (~line 398):

```python
# --- Hybrid Solar+Battery (all new-build, component-additive LCOE) ---
from pipeline_config import get_hybrid_lcoe, get_hybrid_tx

for ht in HYBRID_TYPES:
    ht_pct = mix_pct.get(ht, 0.0)
    if ht_pct <= 0:
        continue
    ht_twh = ht_pct / 100.0 * demand_twh
    existing_ht_twh = floor_twh.get(ht, 0.0)
    ht_new_twh = max(0, ht_twh - existing_ht_twh)
    if ht_new_twh > 0:
        # Component-additive LCOE: renewable + ITC-discounted battery + TX
        ht_lcoe = get_hybrid_lcoe(ht, ren_name, batt_name, iso)
        ht_tx = get_hybrid_tx(ht, tx_name, iso)
        ht_total = ht_lcoe + ht_tx
        # Apply learning curve on battery component (solar/wind already at scale)
        # Battery learning curve uses same FOAK/NOAK as standalone
        # For simplicity in MAC queue, use static LCOE (learning curves are small for batteries)
        ht_cost = ht_new_twh * 1e6 * ht_total
        total_cost += ht_cost
        breakdown[ht] = {'new_twh': ht_new_twh, 'lcoe': ht_total, 'cost': ht_cost}
```

**Learning curves for hybrids:** The MAC queue applies learning curves to technologies with FOAK/NOAK trajectories. Solar and wind are already at scale (no FOAK). The co-located battery component could use a learning curve, but batteries are also nearly at scale. **Recommendation:** Use static hybrid LCOE from `get_hybrid_lcoe()` (no learning curve). If the user wants learning curves on the battery component, that can be added later.

#### 4. Update CCS residual calculation (line 467-471)

Currently:
```python
ccs_pct = 100.0 - (mix_pct.get('clean_firm', 0) + mix_pct.get('solar', 0) +
                    mix_pct.get('wind', 0) + mix_pct.get('hydro', 0) +
                    mix_pct.get('offshore_wind', 0) + mix_pct.get('geothermal', 0))
```

Add hybrid resources to the sum:
```python
explicit_pct = sum(mix_pct.get(r, 0) for r in RESOURCE_COLS)
hybrid_pct = sum(mix_pct.get(ht, 0) for ht in HYBRID_COLS)
ccs_pct = max(0, 100.0 - explicit_pct - hybrid_pct)
```

#### 5. Update `batch_score_mixes()` (line 1159-1250)

This is the vectorized Phase 1 scorer. Add hybrid cost computation:

**a) Extract hybrid resource arrays (after line 1184):**
```python
# Hybrid resource percentages
hybrid_pcts = {}
for ht in HYBRID_COLS:
    if ht in filtered_df.columns:
        hybrid_pcts[ht] = filtered_df[ht].values.astype(np.float64)
    else:
        hybrid_pcts[ht] = np.zeros(N)
```

**b) Precompute hybrid LCOEs in `_precompute_nb_cost_params()` (find this function):**
```python
# In _precompute_nb_cost_params():
for ht in HYBRID_COLS:
    ht_lcoe = get_hybrid_lcoe(ht, sens['ren'], sens['batt'], iso)
    ht_tx = get_hybrid_tx(ht, sens['tx'], iso)
    params[f'{ht}_lcoe'] = ht_lcoe + ht_tx
```

**c) Add hybrid cost to the vectorized nb_cost sum (after line 1242):**
```python
# Hybrid new-build cost (all hybrid is new-build, no existing floor)
for ht in HYBRID_COLS:
    ht_pct = hybrid_pcts[ht]
    ht_new_twh = np.maximum(0, ht_pct / 100.0 * demand_twh)  # No existing hybrid
    nb_cost += ht_new_twh * 1e6 * params[f'{ht}_lcoe']
```

**Note:** If there IS a floor for hybrid resources (from ratcheting), subtract it:
```python
existing_ht = deployed_twh.get(ht, 0.0)
ht_new_twh = np.maximum(0, ht_pct / 100.0 * demand_twh - existing_ht)
```

#### 6. Update `filter_and_sample()` (line 663-723)

The floor constraint logic needs to handle hybrid resources. Currently, the floor is enforced per-resource: `mix[resource] >= floor[resource]`. With hybrids:

```python
# After existing floor checks, add hybrid floor enforcement:
for ht in HYBRID_COLS:
    ht_floor = floor_pct.get(ht, 0)
    if ht_floor > 0 and ht in df.columns:
        df = df[df[ht] >= ht_floor - 0.5]  # 0.5% tolerance
```

#### 7. Update `phase2_refine()` (line 725-777)

Phase 2 perturbs the top candidates. Currently perturbs `clean_firm`, `solar`, `wind`. Add hybrid perturbation:

```python
# Add hybrid resources to perturbation dimensions
for ht in HYBRID_COLS:
    if top_mixes[ht].max() > 0:  # Only perturb if any candidate uses this hybrid
        perturbation_dims.append(ht)
```

Read the function to see how `perturbation_dims` is structured and extend accordingly.

#### 8. Update supply matrix construction in `run_pathway()` / `run_iso()`

Find where `get_supply_profiles()` and `build_supply_matrix()` are called. Pass `include_hybrids=True`:

```python
# In run_iso() or run_pathway():
supply_profiles = get_supply_profiles(iso, gen_profiles, include_hybrids=True)
rtypes = RESOURCE_TYPES_HYBRID  # from dispatch_utils
supply_matrix = build_supply_matrix(supply_profiles, resource_types=rtypes)
```

This ensures the hourly dispatch scoring (Phase 3) accounts for hybrid generation.

#### 9. Update winner extraction and floor ratcheting

When the winner is selected and its resources become the new floor, hybrid resources must be included:

Search for where `floor_twh` is updated (likely in `optimize_threshold()` or `run_pathway()`). Ensure hybrid resource TWh are added to the floor:

```python
# After selecting winner:
for ht in HYBRID_COLS:
    if winner.get(ht, 0) > 0:
        winner_ht_twh = winner[ht] / 100.0 * demand_twh
        floor_twh[ht] = max(floor_twh.get(ht, 0), winner_ht_twh)
```

#### 10. Update output parquet schema

The output parquet (`mac_queue_{ISO}.parquet`) must include hybrid columns. Search for where the DataFrame is constructed for output and ensure hybrid columns are included:

```python
# When building output rows:
for ht in HYBRID_COLS:
    row[ht] = winner_mix.get(ht, 0)
```

### Critical Pitfalls

1. **Family cap enforcement**: Hybrid resources are subject to family caps (`SOLAR_FAMILY_CAP`, `WIND_FAMILY_CAP`). In the MAC queue, this means `solar + solar_batt4 + solar_batt8` must not exceed the solar family cap for the ISO. Import these caps from `step1_pfs_generator` and enforce in `filter_and_sample()`.

2. **CCS residual displacement**: Hybrids reduce the CCS residual (more explicit clean supply → less fossil needed). This is correct behavior — make sure the CCS residual calculation includes hybrid pcts.

3. **Double-counting storage**: Hybrid co-located batteries are already in the 8760 profile. The standalone storage columns (`battery_dispatch_pct`, etc.) are separate grid-level storage. These must NOT be confused. The cost model already handles this correctly IF hybrid costs use `get_hybrid_lcoe()` (which includes battery LCOS) and standalone storage uses the existing annualized tables.

4. **Backward compatibility**: If PFS parquets don't have hybrid columns (pre-hybrid runs), the script should work identically to before. The `MIX_COLS` fill-with-zero logic handles this.

5. **Combinatorial explosion in Phase 2**: Adding 4 hybrid dimensions to perturbation can increase candidates significantly. Consider only perturbing hybrid dimensions when the top candidates actually use hybrids (conditional perturbation).

### Verification
1. Syntax check: `python -c "import py_compile; py_compile.compile('scripts/step3b_mac_queue.py')"`
2. Run on one ISO with one pathway: `python scripts/step3b_mac_queue.py --iso SPP --sensitivities all_med --growth Medium`
3. Check output: `python -c "import pandas as pd; df = pd.read_parquet('data/step3-dispatch/mac_queue/mac_queue_SPP.parquet'); print([c for c in df.columns if 'batt' in c and 'battery' not in c])"`
4. Verify hybrid resources appear in winning mixes at some thresholds
5. Verify CCS residual is lower when hybrids are present

### Commit
Commit with message: "Add hybrid resource support to Step 3B MAC queue — hybrid LCOE pricing, floor ratcheting, and vectorized scoring"

---

## Session J: Step 4.1A + 4.1B — Fossil Dispatch, LMP, and Day Profiles

### Task
Update `step4_1a_fossil_dispatch.py` and `step4_1b_compress_day_profiles.py` to handle hybrid resource columns flowing through the dispatch cache. These scripts consume the dispatch cache built in Step 3A and must construct archetype keys and resource dicts that include hybrid resources.

### Branch
Develop on branch `claude/integrate-hybrid-resources-2pk5J`

### Prerequisites
- Session G complete (dispatch_utils.py: hybrid-aware `_archetype_key()`, `reconstruct_hourly_dispatch()`)
- Session H complete (Step 3A: dispatch cache includes hybrid matched/surplus profiles)

### Background

**Step 4.1A** (`step4_1a_fossil_dispatch.py`, ~362 lines) reads the dispatch cache and computes CO₂ and LMP for each archetype. It builds `resource_pcts` dicts from parquet columns, uses `_archetype_key()` to look up cached dispatch results, and iterates over `RESOURCE_TYPES` for capacity revenue calculation. The dispatch cache now contains `matched_solar_batt4`, `surplus_solar_batt4`, etc. — these need to flow through.

**Step 4.1B** (`step4_1b_compress_day_profiles.py`, ~320 lines) reads the dispatch cache and compresses 8760-hour profiles to 24-hour representative days for dashboard visualization. It constructs `resource_pcts` dicts with hardcoded 6-resource keys, iterates over `RESOURCE_TYPES` for matched/surplus extraction, and generates mix keys without hybrid dimensions.

**Key insight:** Both scripts primarily need changes to I/O and key construction — the core computation logic (CO₂, LMP, compression) doesn't care what resources exist, it just works with the profiles in the cache. The main work is ensuring hybrid resources are included in:
1. Parquet column extraction → `resource_pcts` dict construction
2. Archetype key computation (so cache lookups succeed)
3. Per-resource loops for matched/surplus extraction
4. Output schema (so hybrid data reaches the dashboard)

### Files to Modify

#### `scripts/step4_1a_fossil_dispatch.py` (~362 lines)

##### 1. Update imports (line 44-58)

Add hybrid-aware imports:
```python
from dispatch_utils import (
    H, ISOS, RESOURCE_TYPES, RESOURCE_TYPES_HYBRID, HYBRID_TYPES,
    # ... existing imports ...
)
```

##### 2. Update `compute_capacity_market_revenue()` (line 79-145)

Add hybrid resources to the capacity revenue calculation. Hybrid resources get higher ELCC than standalone solar/wind (batteries shift generation to peak hours):

**a) Add hybrid resources to the per-resource loop (line 102-103):**
```python
for res in ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt',
            'hydro', 'battery', 'battery8', 'ldes', 'h2', 'geothermal',
            'solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8']:
```

**b) Add hybrid resources to VRE category (line 113):**
```python
vre_res = ['solar', 'wind', 'offshore_wind', 'solar_batt4', 'solar_batt8',
           'wind_batt4', 'wind_batt8']
```

**Note:** `PEAK_CAPACITY_CREDITS` in pipeline_config must have entries for hybrid types (Session A should have added these). Verify: `python -c "from pipeline_config import PEAK_CAPACITY_CREDITS; print({k: v for k, v in PEAK_CAPACITY_CREDITS.items() if 'batt' in k})"`

##### 3. Update `run_fossil_dispatch_for_iso()` (line 194-338)

**a) Update `mix_cols` (line 237-239):**
```python
mix_cols = ['mix_clean_firm', 'mix_solar', 'mix_wind', 'mix_offshore_wind',
            'mix_ccs_ccgt', 'mix_hydro', 'battery_dispatch_pct',
            'battery8_dispatch_pct', 'ldes_dispatch_pct']

# Detect hybrids from parquet schema
if co_path:
    avail_cols = pq.read_schema(co_path).names
    has_hybrids = any(f'mix_{ht}' in avail_cols or ht in avail_cols for ht in HYBRID_TYPES)
    if has_hybrids:
        for ht in HYBRID_TYPES:
            col = f'mix_{ht}' if f'mix_{ht}' in avail_cols else ht
            if col in avail_cols:
                mix_cols.append(col)
```

**b) Update `resource_pcts` dict construction (line 243-250):**
```python
rp = {
    'clean_firm': float(row['mix_clean_firm']),
    'solar': float(row['mix_solar']),
    'wind': float(row['mix_wind']),
    'offshore_wind': float(row.get('mix_offshore_wind', 0)),
    'ccs_ccgt': float(row['mix_ccs_ccgt']),
    'hydro': float(row['mix_hydro']),
}
# Add hybrid resources if present
for ht in HYBRID_TYPES:
    col = f'mix_{ht}' if f'mix_{ht}' in row.index else ht
    if col in row.index:
        rp[ht] = float(row.get(col, 0))
```

This ensures `_archetype_key()` (updated in Session G) produces the correct hash including hybrid pcts.

##### 4. Update CO₂ result metadata

In the result dict construction, include hybrid resource shares for downstream analysis:
```python
co2_result = {
    # ... existing fields ...
}
# Add hybrid resource shares if present
for ht in HYBRID_TYPES:
    if rp.get(ht, 0) > 0:
        co2_result[f'mix_{ht}'] = rp[ht]
```

---

#### `scripts/step4_1b_compress_day_profiles.py` (~320 lines)

##### 1. Update imports (line 46-51)

```python
from dispatch_utils import (
    H, ISOS, RESOURCE_TYPES, RESOURCE_TYPES_HYBRID, HYBRID_TYPES,
    CACHE_VERSION, DISPATCH_ORDER,
    load_common_data, get_supply_profiles, get_demand_profile,
    build_supply_matrix, reconstruct_hourly_dispatch,
    _archetype_key, load_dispatch_cache,
)
```

##### 2. Update `dispatch_from_cache()` (line 60-107)

**a) Extend `resource_pcts` dict (line 67-74):**
```python
resource_pcts = {
    'clean_firm': mix.get('clean_firm', 0),
    'solar': mix.get('solar', 0),
    'wind': mix.get('wind', 0),
    'offshore_wind': mix.get('offshore_wind', 0),
    'ccs_ccgt': mix.get('ccs_ccgt', 0),
    'hydro': mix.get('hydro', 0),
}
# Add hybrid resources if present in mix dict
for ht in HYBRID_TYPES:
    if ht in mix and mix[ht] > 0:
        resource_pcts[ht] = mix[ht]
```

**b) Extend per-resource matched/surplus extraction (line 84-88):**
```python
# Determine which resource types are in this cache entry
rtypes = list(RESOURCE_TYPES)
for ht in HYBRID_TYPES:
    if f'matched_{ht}' in cached:
        rtypes.append(ht)

matched = {}
surplus = {}
for rtype in rtypes:
    mk = f'matched_{rtype}'
    sk = f'surplus_{rtype}'
    matched[rtype] = cached[mk] if mk in cached else np.zeros(H, dtype=np.float64)
    surplus[rtype] = cached[sk] if sk in cached else np.zeros(H, dtype=np.float64)
```

##### 3. Update `compress_to_24h()` (line 110-147)

Extend the resource loop (line 137-139):
```python
base_resources = ['clean_firm', 'ccs_ccgt', 'solar', 'wind', 'offshore_wind', 'hydro']
for r in base_resources:
    compressed['matched'][r] = sum_by_hod(result['matched'][r])
    compressed['surplus'][r] = sum_by_hod(result['surplus'][r])

# Add hybrid resources if present
for ht in HYBRID_TYPES:
    if ht in result['matched']:
        compressed['matched'][ht] = sum_by_hod(result['matched'][ht])
        compressed['surplus'][ht] = sum_by_hod(result['surplus'][ht])
```

##### 4. Update `mix_key()` (line 173-184)

Include hybrid dimensions in the key so different hybrid mixes get different entries:
```python
def mix_key(mix, battery_pct, ldes_pct, h2_pct=0):
    cf = mix.get('clean_firm', 0)
    s = mix.get('solar', 0)
    w = mix.get('wind', 0)
    c = mix.get('ccs_ccgt', 0)
    h = mix.get('hydro', 0)
    key = f"{cf}_{s}_{w}_{c}_{h}_{battery_pct}_{ldes_pct}_{h2_pct}"
    # Append hybrid values if any are non-zero
    hybrid_vals = [mix.get(ht, 0) for ht in HYBRID_TYPES]
    if any(v > 0 for v in hybrid_vals):
        key += '_' + '_'.join(str(v) for v in hybrid_vals)
    return key
```

##### 5. Update `main()` mix extraction (line 252-294)

When extracting resource mixes from feasible_mixes (both columnar and row formats), include hybrid columns:

**Columnar format (line 256-261):**
```python
rm = {
    'clean_firm': fmixes['clean_firm'][i],
    'solar': fmixes['solar'][i],
    'wind': fmixes['wind'][i],
    'ccs_ccgt': fmixes['ccs_ccgt'][i],
    'hydro': fmixes['hydro'][i],
}
# Add hybrid resources if present in feasible_mixes
for ht in HYBRID_TYPES:
    if ht in fmixes:
        rm[ht] = fmixes[ht][i]
```

**Row format (line 272-273):**
```python
rm = fm['resource_mix']
# Hybrid resources will already be in resource_mix dict if present
```

**Scenario format (line 285-286):**
```python
rm = sc.get('resource_mix', {})
# Same — hybrid resources included if present
```

### Critical Pitfalls

1. **Archetype key consistency**: The `resource_pcts` dict passed to `_archetype_key()` must include the same hybrid resource values as when the cache was built in Step 3A. If Step 3A used `mix_solar_batt4` as the column name but Step 4 reads `solar_batt4`, the dict key must be `'solar_batt4'` (not `'mix_solar_batt4'`). Standardize on the resource name without the `mix_` prefix for `resource_pcts` dicts.

2. **Cache miss on hybrid mixes**: If the dispatch cache was built before hybrid integration, it won't have hybrid archetype keys. The scripts should handle this gracefully (skip with warning, not crash).

3. **Dashboard compatibility**: `compressed_day_profiles.json` is consumed by dashboard JavaScript. Adding hybrid matched/surplus entries changes the JSON schema. The dashboard chart code will need to handle new resource keys — but that's a separate session. For now, just ensure the data is there.

4. **`step4_1a_augment_capacity_rev.py`**: This secondary script also constructs `resource_pcts` dicts. Apply the same hybrid extension pattern (read file first to identify exact lines).

### Verification
1. Run step4_1a on one ISO: `python scripts/step4_1a_fossil_dispatch.py --iso SPP`
2. Check CO₂ output includes hybrid metadata: `python -c "import pandas as pd; df = pd.read_parquet('data/step4-analysis/co2_results/SPP_co2.parquet'); print([c for c in df.columns if 'batt' in c])"`
3. Run step4_1b: `python scripts/step4_1b_compress_day_profiles.py`
4. Check compressed profiles include hybrid resources: `python -c "import json; d = json.load(open('dashboard/compressed_day_profiles.json')); iso = list(d.keys())[0]; mk = list(d[iso]['profiles'].keys())[0]; print(list(d[iso]['profiles'][mk]['matched'].keys()))"`
5. Verify non-hybrid ISOs produce identical output to before.

### Commit
Commit with message: "Add hybrid resource support to Step 4.1A fossil dispatch and Step 4.1B compressed day profiles"
