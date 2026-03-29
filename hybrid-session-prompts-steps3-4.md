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
