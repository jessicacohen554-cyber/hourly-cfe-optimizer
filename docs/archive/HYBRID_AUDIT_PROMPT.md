# Final Audit: Hybrid Resource Inclusion Across Pipeline

## Context

We just completed a major fix pass for 4 systematic bugs that caused hybrid co-located resources (solar_batt4, solar_batt8, wind_batt4, wind_batt8) to be partially or fully invisible throughout the pipeline. The fixes landed in commit `f336d72` on branch `claude/fix-hybrid-pipeline-bugs-RgxaR` across 8 files. This audit is the verification pass to confirm nothing was missed and no regressions were introduced.

## Background: The 4 Bug Classes

1. **Bug 1 — Zero energy revenue**: `get_supply_profiles()` called without `include_hybrids=True` means hybrid 8760 profiles aren't loaded into the supply dict. Downstream dispatch sees zero generation from hybrids even when the mix allocates capacity to them.

2. **Bug 2 — Ghost capacity**: `reconstruct_hourly_dispatch()` called without `resource_types=RESOURCE_TYPES_HYBRID` means dispatch only reads the base 6 resource columns from the supply matrix, ignoring hybrid columns entirely. Hybrids appear in the mix but contribute nothing to hourly matching.

3. **Bug 3 — Oversized fossil fleet**: `clean_peak_mw` calculations that skip hybrid capacity credits undercount clean peak MW. This inflates the residual peak that gas must cover, leading to oversized gas backup fleets and overstated gas costs.

4. **Bug 4 — Overstated gas backup**: `build_supply_matrix()` called without `resource_types=RESOURCE_TYPES_HYBRID` creates a matrix with only 6 rows (base resources) even when supply_profiles contains 10 entries (base + 4 hybrids). When this undersized matrix is passed to `reconstruct_hourly_dispatch` with `resource_types=RESOURCE_TYPES_HYBRID`, shape mismatches occur or hybrid rows silently read as zeros.

## What Was Fixed (for reference, do not re-fix these)

These files were modified in the fix commit — verify they are correct, don't re-apply:
- `scripts/step3b_mac_queue.py` — 5 call sites
- `scripts/lmp_engine.py` — import + 3 call sites
- `scripts/calibrate_lmp_model.py` — import + 2 call sites
- `scripts/procurement_utils.py` — import + 5 call sites
- `scripts/scenario_common.py` — import + 4 call sites
- `scripts/step7_1e_dispatch_deployment.py` — import + 3 call sites
- `scripts/step7_1g_extract_use_case_data.py` — import + 5 call sites
- `scripts/reexport_scenario_a.py` — import + 1 call site

## Audit Instructions

### Phase 1: Verify the fixes are correct

For each of the 8 files listed above, verify:

1. **Every `get_supply_profiles()` call** that loads profiles for dispatch/capacity/gas-backup purposes has `include_hybrids=True`. Exception: calls that intentionally load base-only profiles for non-dispatch purposes (e.g., extracting just solar/wind shapes for weighting) — flag these and explain why they're safe.

2. **Every `build_supply_matrix()` call** that follows an `include_hybrids=True` profile load also passes `resource_types=RESOURCE_TYPES_HYBRID`. A mismatch (hybrids in profiles, base-only matrix) is Bug 4.

3. **Every `reconstruct_hourly_dispatch()` call** that receives a hybrid-aware `supply_matrix` also passes `resource_types=RESOURCE_TYPES_HYBRID`. The resource_types parameter MUST match the supply_matrix shape.

4. **Every `resource_pcts` dict** passed to `reconstruct_hourly_dispatch()` includes hybrid keys when the mix data contains them. Check that hybrid percentages are extracted from the source mix dict (e.g., `for ht in HYBRID_COLS: resource_pcts[ht] = mix.get(ht, 0)`). A resource_pcts with only base 6 keys + resource_types=RESOURCE_TYPES_HYBRID is technically safe (hybrids default to 0%) but may indicate a missed extraction.

5. **CCS residual calculations** (pattern: `ccs_pct = max(0, 100.0 - explicit_sum)`) include hybrid percentages in `explicit_sum`. If hybrids are omitted, CCS is overestimated, which shifts dispatch and cost calculations.

6. **`clean_peak_mw` calculations** include hybrid capacity credits via `PEAK_CAPACITY_CREDITS[hybrid_type]`. The correct pattern is in `scenario_common.py` lines 479-490 and `step2_2a_cost_optimization.py` lines 406-416.

7. **Baseline dispatch calls** (using `GRID_MIX_SHARES` for 2025 grid) have hybrid keys set to 0 via `.setdefault()` and still pass `resource_types=RESOURCE_TYPES_HYBRID` for shape consistency with the supply_matrix.

### Phase 2: Scan for any remaining unhybridized call sites

Search the entire `scripts/` directory (excluding `step1_*.py` which use their own internal `get_supply_profiles` from `step1_pfs_generator.py`, not from `dispatch_utils`):

1. **`get_supply_profiles(iso, gen_profiles)` without `include_hybrids=True`**: Flag any call in a non-Step-1 script that loads profiles without hybrids and then uses them for dispatch, capacity, or gas backup calculations. Calls that only extract base resource shapes for weighting/normalization are acceptable — annotate why.

2. **`build_supply_matrix(supply_profiles)` without `resource_types=RESOURCE_TYPES_HYBRID`**: Flag any call where the supply_profiles dict was loaded with `include_hybrids=True` but the matrix build omits the resource_types parameter. This silently drops hybrid columns.

3. **`reconstruct_hourly_dispatch(...)` without `resource_types=...`**: Flag any call in a non-Step-1 script that dispatches without specifying resource_types. Calls in `step3a_build_dispatch_cache.py` and `step6_1_smartargets.py` already use conditional hybrid logic (`resource_types=RESOURCE_TYPES_HYBRID if has_hybrids else RESOURCE_TYPES`) — verify these are still correct.

4. **`clean_peak_mw` or `gas_backup` calculations**: Search for patterns like `PEAK_CAPACITY_CREDITS`, `clean_peak`, `gas_needed`, `gas_backup`, `residual_peak`, `ra_peak - clean_peak`. Verify each calculation accounts for hybrid capacity credits. Key files: `lmp_engine.py`, `step2_2a_cost_optimization.py`, `scenario_common.py`, `step7_1e_dispatch_deployment.py`.

### Phase 3: Check the market-simulator directory

The `market-simulator/scripts/` directory contains its own copies of several pipeline files with identical bugs. These were noted as out-of-scope in the fix commit but should be audited for completeness:

- `market-simulator/scripts/lmp_engine.py`
- `market-simulator/scripts/procurement_utils.py`
- `market-simulator/scripts/scenario_common.py`
- `market-simulator/scripts/market_simulation.py`
- `market-simulator/scripts/dispatch_utils.py`

For each, check the same 4 bug classes. Flag any unhybridized call sites with file, line number, function name, and which bug class applies.

### Phase 4: Verify no shape mismatches

The most dangerous failure mode is a supply_matrix built with one set of resource_types being passed to reconstruct_hourly_dispatch with a different resource_types. This causes silent data corruption (wrong resource mapped to wrong profile row).

For every code path where `build_supply_matrix` and `reconstruct_hourly_dispatch` are both called:
- Confirm the `resource_types` parameter is identical in both calls
- Or confirm the supply_matrix is rebuilt with matching resource_types before each dispatch call

### Phase 5: Test file audit

Check `scripts/tests/test_dispatch.py` and `scripts/tests/test_energy_conservation.py`:
- Do tests exercise hybrid dispatch paths?
- Are tests passing with the new changes?
- If tests don't cover hybrids, flag this as a gap (don't write new tests, just note the gap)

## Output Format

Produce a structured report:

```
## Audit Results

### Fixed Files Verification (Phase 1)
For each of the 8 files:
- [ ] File: filename.py
  - All get_supply_profiles calls: [OK/ISSUE: description]
  - All build_supply_matrix calls: [OK/ISSUE: description]
  - All reconstruct_hourly_dispatch calls: [OK/ISSUE: description]
  - resource_pcts hybrid extraction: [OK/ISSUE: description]
  - CCS residual calc: [OK/N/A]
  - clean_peak_mw calc: [OK/N/A]
  - Baseline dispatch shape consistency: [OK/N/A]

### Remaining Unhybridized Call Sites (Phase 2)
- [ ] get_supply_profiles: [count] remaining without include_hybrids
  - List each with file:line and justification (safe/needs fix)
- [ ] build_supply_matrix: [count] remaining without resource_types
  - List each with file:line and justification
- [ ] reconstruct_hourly_dispatch: [count] remaining without resource_types
  - List each with file:line and justification
- [ ] clean_peak/gas_backup calcs: [count] remaining without hybrid credits
  - List each with file:line and justification

### Market-Simulator Audit (Phase 3)
For each file:
- Bug 1 instances: [count] (list file:line)
- Bug 2 instances: [count] (list file:line)
- Bug 3 instances: [count] (list file:line)
- Bug 4 instances: [count] (list file:line)

### Shape Mismatch Check (Phase 4)
- [count] code paths checked, [count] mismatches found
- List any mismatches

### Test Coverage (Phase 5)
- Hybrid dispatch tested: [Yes/No]
- Tests passing: [Yes/No/Not run]
- Coverage gaps: [list]

### Summary
- Total remaining bugs found: [count]
- Critical (data corruption risk): [count]
- Moderate (incorrect results): [count]
- Low (cosmetic/non-impacting): [count]
```

## Key Constants Reference

```python
RESOURCE_TYPES = ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro']
HYBRID_TYPES = ['solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8']
RESOURCE_TYPES_HYBRID = RESOURCE_TYPES + HYBRID_TYPES  # 10 types
HYBRID_COLS = list(HYBRID_TYPES)  # alias used in step3b
```

## Key Function Signatures

```python
# dispatch_utils.py
def get_supply_profiles(iso, gen_profiles, include_hybrids=False, hybrid_profiles=None)
def build_supply_matrix(supply_profiles, resource_types=None)  # None → RESOURCE_TYPES (base 6)
def reconstruct_hourly_dispatch(demand_norm, supply_profiles, resource_pcts,
                                 procurement_pct=100, battery_dispatch_pct=0,
                                 battery8_dispatch_pct=0, ldes_dispatch_pct=0,
                                 supply_matrix=None, detailed=False,
                                 h2_dispatch_pct=0, resource_types=None)  # None → RESOURCE_TYPES
```

## Files Known to Be Correct (Skip These)

- `scripts/step1_*.py` — Use their own `s1.get_supply_profiles()` from `step1_pfs_generator.py`, not `dispatch_utils`. Step 1 handles hybrids via its own 8-10D grid dimensions.
- `scripts/step3a_build_dispatch_cache.py` — Already uses conditional `resource_types=RESOURCE_TYPES_HYBRID if has_hybrids else RESOURCE_TYPES`
- `scripts/step6_1_smartargets.py` — Already uses conditional hybrid logic
- `scripts/step2_2a_cost_optimization.py` — clean_peak_mw already includes hybrids (lines 406-416), no dispatch_utils dispatch calls
- `scripts/step4_1b_compress_day_profiles.py` — Cache-only reader (no live dispatch fallback), reads from step3a cache which is hybrid-aware
- `scripts/dispatch_utils.py` — Definition file, not a consumer
- `scripts/generate_spec_doc.py` — Documentation generator, mentions functions in text only
