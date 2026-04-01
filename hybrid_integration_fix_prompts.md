# Hybrid Resource Integration — Downstream Fix Prompts

**Generated**: 2026-04-01
**Context**: QA/QC of hybrid resource (solar_batt4, solar_batt8, wind_batt4, wind_batt8) integration across the 8-step pipeline. Step 1 is clean. These prompts fix downstream bugs where hardcoded resource lists silently drop hybrids.

**Usage**: Copy/paste each prompt into a fresh Claude Code session. Each is self-contained and independent — run in any order. Commit after each fix.

---

## Prompt 1: Fix step2_2b_track_nb_ctr.py — Hybrid Arrays Missing from Demand Growth Output (CRITICAL)

```
## Task: Fix hybrid resource exclusion in step2_2b_track_nb_ctr.py

### Bug Description
`scripts/step2_2b_track_nb_ctr.py` generates demand growth parquets (`track_demand_growth.parquet`) but completely excludes hybrid resources. Three issues:

1. **Lines 187-197**: Only retrieves base resource arrays from the `arrays` parameter. Never retrieves `solar_batt4`, `solar_batt8`, `wind_batt4`, `wind_batt8`.
2. **Lines 240, 273**: CCS-CCGT residual = `100 - (cf + sol + wnd + hyd + osw + geo)` — does NOT subtract hybrid percentages. This inflates CCS-CCGT values when hybrids are present.
3. **Lines 228-247 (batch path), 260-279 (single path)**: Output row dicts don't include `mix_solar_batt4`, `mix_solar_batt8`, `mix_wind_batt4`, `mix_wind_batt8` columns.

### Impact
- `track_demand_growth.parquet` has zero hybrid values and inflated CCS-CCGT percentages
- Corrupts downstream: step5 procurement strategies and step7 dashboard data extraction

### Fix Required

**A) Add hybrid array retrieval after line 192** (add 4 lines after `arr_geo`):
```python
arr_sb4 = arrays.get('solar_batt4', np.zeros(len(arr_cf), dtype=np.int64))
arr_sb8 = arrays.get('solar_batt8', np.zeros(len(arr_cf), dtype=np.int64))
arr_wb4 = arrays.get('wind_batt4', np.zeros(len(arr_cf), dtype=np.int64))
arr_wb8 = arrays.get('wind_batt8', np.zeros(len(arr_cf), dtype=np.int64))
```

**B) In the batch path (around line 210-216)**, add batch indexing for hybrids:
```python
sb4_batch = arr_sb4[idxs].astype(int)
sb8_batch = arr_sb8[idxs].astype(int)
wb4_batch = arr_wb4[idxs].astype(int)
wb8_batch = arr_wb8[idxs].astype(int)
```

**C) Fix CCS residual on line 240** (batch path) to subtract hybrids:
```python
'mix_ccs_ccgt': max(0, 100 - (cf + sol + wnd + hyd + osw + geo + int(sb4_batch[i]) + int(sb8_batch[i]) + int(wb4_batch[i]) + int(wb8_batch[i]))),
```

**D) Add hybrid columns to the batch row dict** (after `'mix_geothermal': geo,` around line 239):
```python
'mix_solar_batt4': int(sb4_batch[i]),
'mix_solar_batt8': int(sb8_batch[i]),
'mix_wind_batt4': int(wb4_batch[i]),
'mix_wind_batt8': int(wb8_batch[i]),
```

**E) Fix the single-item path (lines 260-279)** with the same pattern:
- Add hybrid lookups: `sb4 = int(arr_sb4[mix_idx])`, etc.
- Fix CCS residual on line 273 to subtract hybrids
- Add `row['mix_solar_batt4'] = sb4`, etc.

### Verification
1. `python -c "import py_compile; py_compile.compile('scripts/step2_2b_track_nb_ctr.py')"`
2. Grep the file to confirm all 4 hybrid columns appear in both batch and single paths
3. Check that `mix_ccs_ccgt` formula subtracts all hybrid values in both paths
```

---

## Prompt 2: Fix step7_1a_generate_shared_data.py — Manifest Key Mismatch (CRITICAL)

```
## Task: Fix hybrid exclusion in step7_1a manifest archetype key lookup

### Bug Description
`scripts/step7_1a_generate_shared_data.py` line 528 constructs an archetype key for manifest dispatch lookup, but excludes hybrid resources:

```python
rp = {k: rm.get(k, 0) for k in ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro']}
akey = _archetype_key(iso, rp, 100, bat_cap, bat8_cap, ldes_cap)
```

The `_archetype_key()` function in `dispatch_utils.py` (line 1027) appends hybrid parts to the hash when ANY hybrid pct > 0. So when a mix has non-zero hybrid values:
- The manifest was built WITH hybrid parts in the key
- But the lookup here builds a key WITHOUT hybrid parts
- Result: key mismatch → lookup fails → dispatch values silently default to 0

The demand growth path at lines 490-493 correctly includes hybrids in the resource dict. Only the fallback/manifest lookup path at line 528 is broken.

### Fix Required

**Line 528**: Add the 4 hybrid types to the resource pcts dict:
```python
rp = {k: rm.get(k, 0) for k in ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro', 'solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8']}
```

Alternatively, import HYBRID_TYPES from pipeline_config and use:
```python
rp = {k: rm.get(k, 0) for k in ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro'] + HYBRID_TYPES}
```

### Verification
1. `python -c "import py_compile; py_compile.compile('scripts/step7_1a_generate_shared_data.py')"`
2. Grep the file for `_archetype_key` calls — confirm ALL of them include hybrid resources in the `rp` dict
3. Check that HYBRID_TYPES is imported if using the dynamic approach
```

---

## Prompt 3: Fix scenario_common.py — Hardcoded RESOURCES List (MEDIUM)

```
## Task: Add hybrid resources to scenario_common.py RESOURCES tracking

### Bug Description
`scripts/scenario_common.py` line 58 defines:
```python
RESOURCES = ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro']
```

This list is used in several places that silently exclude hybrid resource TWh:

1. **Line 505**: `for res, pct in zip(RESOURCES, [...])` — hybrid costs excluded from scenario cost evaluation
2. **Line 1393**: `aug['resource_twh'] = {res: augmented.get(res, 0) for res in RESOURCES}` — hybrid TWh excluded from augmented resource tracking
3. **Lines 1652, 1731, 1736**: Iteration over RESOURCES misses hybrids in delta calculations and stranding analysis

### Fix Required

**A) Add a RESOURCES_WITH_HYBRIDS constant** near line 58:
```python
from pipeline_config import HYBRID_TYPES
RESOURCES = ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro']
RESOURCES_WITH_HYBRIDS = RESOURCES + list(HYBRID_TYPES)
```

**B) Update each usage site** — evaluate each one carefully:

- **Line 505** (`zip(RESOURCES, ...)`): This zips with positional cost coefficients. If the cost function already handles hybrid costs separately (check the calling context), this may be intentional. If hybrids need their own cost coefficients, extend both the RESOURCES list and the coefficient list.

- **Line 1393** (`aug['resource_twh']`): Change to `RESOURCES_WITH_HYBRIDS` so hybrid TWh is tracked in augmented results.

- **Lines 1652, 1731, 1736** (delta/stranding iteration): Change to `RESOURCES_WITH_HYBRIDS` so hybrid resources appear in resource deltas and stranding analysis.

**Important**: Read the full context around each usage before changing. Some may intentionally use base-only resources (e.g., if hybrid TWh is already folded into solar/wind family totals). Document any intentional exclusions with a comment.

### Verification
1. `python -c "import py_compile; py_compile.compile('scripts/scenario_common.py')"`
2. Grep for `\bRESOURCES\b` in the file to confirm every usage is either updated or has a comment explaining why it uses base-only
3. Check that HYBRID_TYPES import exists
```

---

## Prompt 4: Fix step2_2a_cost_optimization.py — Metadata RESOURCE_TYPES (LOW)

```
## Task: Update step2_2a metadata to include hybrid resource types

### Bug Description
`scripts/step2_2a_cost_optimization.py` line 110 defines a local:
```python
RESOURCE_TYPES = ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro']
```

This is used ONLY for metadata output at lines 2037 and 2299:
```python
'resource_types': RESOURCE_TYPES,
```

The actual cost computation functions correctly handle all 4 hybrids (confirmed — they retrieve hybrid arrays and compute component-additive LCOE). This is purely a metadata completeness issue.

### Fix Required

**Line 110**: Add hybrids to the metadata list:
```python
RESOURCE_TYPES = ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro',
                  'solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8']
```

Or import from pipeline_config:
```python
from pipeline_config import HYBRID_TYPES
RESOURCE_TYPES = ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro'] + list(HYBRID_TYPES)
```

### Verification
1. `python -c "import py_compile; py_compile.compile('scripts/step2_2a_cost_optimization.py')"`
2. Grep for `RESOURCE_TYPES` in the file — confirm it now includes hybrids
3. Confirm no other code paths use this local RESOURCE_TYPES for computation (they don't — verified)
```

---

## Prompt 5: Full Integration Test — Validate Hybrids Flow Step 1 to Step 7

```
## Task: Write and run a hybrid resource integration test

### Objective
Validate that hybrid resources flow correctly through the entire pipeline by checking parquet schemas and data values at each stage.

### Test Script
Create a temporary test script `scripts/test_hybrid_integration.py` that:

1. **Step 1 output check**: Read `data/step1-pfs/` parquets for one ISO (e.g., ERCOT). Verify columns `solar_batt4`, `solar_batt8`, `wind_batt4`, `wind_batt8` exist and have non-zero values.

2. **Step 2.1 output check**: Read `data/step2.1-ef/` parquets. Verify hybrid columns are present. Check that at least some EF mixes have non-zero hybrid values.

3. **Step 2.2 output check**: Read `data/step2.2-cost/` parquets. Verify hybrid columns present and that `mix_ccs_ccgt` = `100 - sum(all resources including hybrids)` for rows with non-zero hybrids.

4. **Step 3a dispatch cache check**: Read `data/step3-dispatch/` NPZ files. Verify hybrid resources appear in the dispatch manifest.

5. **Step 7 shared-data check**: Read `dashboard/js/shared-data.js`. Parse the JSON and verify hybrid resource keys appear in the data structure.

6. **Profile consistency check**: Load hybrid profiles from `data/hybrid_profiles/{ISO}_hybrid_profiles.npz`. Verify:
   - All 4 types present
   - Each sums to ~1.0 (energy-share normalized)
   - No negative values
   - Shape is (8760,)

### Run
```bash
python scripts/test_hybrid_integration.py
```

Report pass/fail for each check. Delete the test script after running.
```

---

## Prompt 6: Codebase-Wide Sweep — Find All Hardcoded Resource Lists

```
## Task: Sweep entire codebase for hardcoded resource lists that may miss hybrids

### Objective
Find every hardcoded resource list in Python scripts that could silently exclude hybrid resources, and classify each as:
- **BUG**: Needs fix (hybrids should be included)
- **INTENTIONAL**: Correctly uses base-only (e.g., existing generation tracking where hybrids don't apply)
- **N/A**: Not related to resource mix tracking

### Search Patterns
Run these greps across `scripts/`:

1. `grep -rn "RESOURCE_TYPES\s*=" scripts/` — local RESOURCE_TYPES definitions
2. `grep -rn "RESOURCES\s*=" scripts/` — local RESOURCES definitions  
3. `grep -rn "\['clean_firm'.*'hydro'\]" scripts/` — inline hardcoded lists
4. `grep -rn "'clean_firm'.*'solar'.*'wind'" scripts/` — partial inline lists
5. `grep -rn "for.*res.*in.*\[.*clean_firm" scripts/` — loops over hardcoded resource lists
6. `grep -rn "\.get(k, 0) for k in \[" scripts/` — dict comprehensions with hardcoded keys

### For Each Match
1. Read the surrounding context (10 lines above/below)
2. Determine if hybrids should be included
3. Classify as BUG / INTENTIONAL / N/A
4. For BUGs: note the fix needed

### Expected Known Results
- `step2_2b_track_nb_ctr.py:187-197` — BUG (Prompt 1)
- `step7_1a_generate_shared_data.py:528` — BUG (Prompt 2)
- `scenario_common.py:58` — BUG (Prompt 3)
- `step2_2a_cost_optimization.py:110` — BUG/LOW (Prompt 4)
- `dispatch_utils.py:74` — INTENTIONAL (RESOURCE_TYPES is base; RESOURCE_TYPES_HYBRID exists separately)
- `step2_1_efficient_frontier.py` — OK (uses get_resource_cols with include_hybrids flag)

### Deliverable
A table of all matches with classification and fix status. Add any NEW bugs found to the fix queue.
```

---

## Execution Order Recommendation

1. **Prompt 6 first** (sweep) — may surface additional bugs not yet found
2. **Prompt 1** (step2_2b) — highest impact, fixes data corruption
3. **Prompt 2** (step7_1a) — fixes dashboard data extraction
4. **Prompt 3** (scenario_common) — fixes MAC/scenario tracking
5. **Prompt 4** (step2_2a) — metadata cleanup
6. **Prompt 5** (integration test) — validates all fixes work end-to-end

After all fixes: re-run Steps 2.2b → 3a → 4 → 5 → 7 to regenerate downstream data with correct hybrid values.
