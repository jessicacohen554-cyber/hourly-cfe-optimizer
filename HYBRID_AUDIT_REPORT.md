# Hybrid Resource Audit Report

**Date:** 2026-04-08
**Branch:** `claude/hybrid-audit-review-vAlfK`
**Scope:** Verification of hybrid resource (solar_batt4/8, wind_batt4/8) fix commit `f336d72` + scan for remaining bugs

---

## Audit Results

### Phase 1: Fixed Files Verification

All 8 files from the fix commit verified. Each call site checked for 7 criteria.

#### File 1: `scripts/step3b_mac_queue.py`
- All `get_supply_profiles` calls: **OK** — Lines 1915, 1978, 2255 all pass `include_hybrids=True`
- All `build_supply_matrix` calls: **OK** — Lines 1916, 1979, 2256 all pass `resource_types=RESOURCE_TYPES_HYBRID`
- All `reconstruct_hourly_dispatch` calls: **OK** — Lines 740, 1686, 2129, 2176 all pass `resource_types=RESOURCE_TYPES_HYBRID`
- `resource_pcts` hybrid extraction: **OK** — Lines 724-725, 1682-1685, 2166-2168 loop over `HYBRID_COLS`
- CCS residual calc: **OK** — Lines 639-641 include `hybrid_pct` in `explicit_sum`; lines 728-732 same pattern
- `clean_peak_mw` calc: **N/A** — Uses dispatch-based gas backup (8760 dispatch) rather than static ELCC
- Baseline dispatch shape consistency: **OK** — Lines 1837-1846, 2125-2128 use `.setdefault()` + `RESOURCE_TYPES_HYBRID`

#### File 2: `scripts/lmp_engine.py`
- All `get_supply_profiles` calls: **OK** — Lines 1402, 1517 pass `include_hybrids=True`
- All `build_supply_matrix` calls: **OK** — No direct calls (supply_matrix passed from caller or None)
- All `reconstruct_hourly_dispatch` calls: **OK** — Lines 1840, 1963-1969 pass `resource_types=RESOURCE_TYPES_HYBRID`
- `resource_pcts` hybrid extraction: **OK** — Lines 1813-1816, 1945-1950 extract hybrid keys from resource_mix
- CCS residual calc: **N/A** — LMP engine doesn't compute CCS residual
- `clean_peak_mw` calc: **OK** — `_compute_clean_peak_mw` (lines 356-384) includes all 4 hybrid types at lines 375-378
- Baseline dispatch shape consistency: **OK** — Lines 1818-1819, 1952-1953 `.setdefault()` for HYBRID_TYPES

#### File 3: `scripts/calibrate_lmp_model.py`
- All `get_supply_profiles` calls: **OK** — Uses `include_hybrids=True`
- All `build_supply_matrix` calls: **OK** — No direct calls
- All `reconstruct_hourly_dispatch` calls: **OK** — Passes `resource_types=RESOURCE_TYPES_HYBRID`
- `resource_pcts` hybrid extraction: **OK** — Extracts hybrid keys from resource_mix
- CCS residual calc: **N/A**
- `clean_peak_mw` calc: **N/A**
- Baseline dispatch shape consistency: **OK**

#### File 4: `scripts/procurement_utils.py`
- All `get_supply_profiles` calls: **OK** — Line 1391 passes `include_hybrids=True`
- All `build_supply_matrix` calls: **OK** — Line 1391 passes `resource_types=RESOURCE_TYPES_HYBRID`
- All `reconstruct_hourly_dispatch` calls: **OK** — Line 1468 passes `resource_types=RESOURCE_TYPES_HYBRID`
- `resource_pcts` hybrid extraction: **OK** — Lines 1423-1426 loop `HYBRID_TYPES` from EF data
- CCS residual calc: **N/A**
- `clean_peak_mw` calc: **N/A**
- Baseline dispatch shape consistency: **N/A**

#### File 5: `scripts/scenario_common.py`
- All `get_supply_profiles` calls: **OK** — Line 1530 passes `include_hybrids=True`; line 1594 passes `include_hybrids=has_hybrids`
- All `build_supply_matrix` calls: **OK** — Line 428 passes `resource_types=RESOURCE_TYPES_HYBRID`
- All `reconstruct_hourly_dispatch` calls: **OK** — Lines 437, 455 pass `resource_types=RESOURCE_TYPES_HYBRID`
- `resource_pcts` hybrid extraction: **OK** — Lines 435-436 `resource_pcts.update(hybrid_pcts)`; lines 453-454 `.setdefault()` for HYBRID_TYPES
- CCS residual calc: **OK** — Line 340 includes hybrid percentages
- `clean_peak_mw` calc: **OK** — Vectorized function uses `p[40]-p[43]` for hybrid credits; fallback at lines 490-503 includes hybrids
- Baseline dispatch shape consistency: **OK** — Lines 452-458 `.setdefault()` + `RESOURCE_TYPES_HYBRID`

#### File 6: `scripts/step7_1e_dispatch_deployment.py`
- All `get_supply_profiles` calls: **OK** — Line 330 passes `include_hybrids=True`
- All `build_supply_matrix` calls: **OK** — Line 330 passes `resource_types=RESOURCE_TYPES_HYBRID`
- All `reconstruct_hourly_dispatch` calls: **OK** — Lines 109-118, 169-178 pass `resource_types=RESOURCE_TYPES_HYBRID`
- `resource_pcts` hybrid extraction: **OK**
- CCS residual calc: **N/A**
- `clean_peak_mw` calc: **N/A** — Uses dispatch-based metrics
- Baseline dispatch shape consistency: **OK** — `resource_types=RESOURCE_TYPES_HYBRID` on all paths

#### File 7: `scripts/step7_1g_extract_use_case_data.py`
- All `get_supply_profiles` calls: **OK** — Lines 501, 560 pass `include_hybrids=True`
- All `build_supply_matrix` calls: **OK** — Lines 501, 560 pass `resource_types=RESOURCE_TYPES_HYBRID`
- All `reconstruct_hourly_dispatch` calls: **OK** — Lines 538, 580 pass `resource_types=RESOURCE_TYPES_HYBRID`
- `resource_pcts` hybrid extraction: **OK**
- CCS residual calc: **N/A**
- `clean_peak_mw` calc: **N/A**
- Baseline dispatch shape consistency: **OK**

#### File 8: `scripts/reexport_scenario_a.py`
- All `get_supply_profiles` calls: **OK** — Passes `include_hybrids=True`
- All `build_supply_matrix` calls: **OK** — Passes `resource_types=RESOURCE_TYPES_HYBRID`
- All `reconstruct_hourly_dispatch` calls: **N/A** — No direct dispatch calls
- `resource_pcts` hybrid extraction: **N/A**
- CCS residual calc: **N/A**
- `clean_peak_mw` calc: **N/A**
- Baseline dispatch shape consistency: **N/A**

---

### Phase 2: Remaining Unhybridized Call Sites (scripts/)

#### Bug Class 1: `get_supply_profiles()` without `include_hybrids=True`

| # | File:Line | Function | Verdict |
|---|-----------|----------|---------|
| 1 | `validate_weather_uncertainty.py:109` | `build_year_profiles()` | **SAFE** — Uses `step1_pfs_generator.get_supply_profiles` (Step 1 API). Base-only re-scoring for weather variance isolation. |
| 2 | `validate_weather_uncertainty.py:235` | main loop | **SAFE** — Same Step 1 API, consistent base-only. |
| 3 | `step2_1b_augment_thin_ef.py:472` | `main()` | **SAFE** — Uses Step 1 API. Hybrids loaded separately via `s1.load_hybrid_profiles()` and merged in `prepare_numpy_profiles(include_hybrids=True)`. |
| 4 | `test_storage_validation.py:17` | top-level | **SAFE** — Step 1 API, deliberate base-only scoring for storage validation. |

**0 bugs found** — all are intentional base-only loads via Step 1 API.

#### Bug Class 2: `build_supply_matrix()` without `resource_types=RESOURCE_TYPES_HYBRID`

| # | File:Line | Function | Verdict |
|---|-----------|----------|---------|
| 1 | `tests/test_dispatch.py:218` | `test_clean_firm_dispatched_before_solar()` | **SAFE** — Test fixture has base 6 resources only. |

**0 bugs found.**

#### Bug Class 3: `reconstruct_hourly_dispatch()` without `resource_types=`

| # | File:Line | Function | Verdict |
|---|-----------|----------|---------|
| 1 | `tests/test_dispatch.py:254,285,328,350,371` | Various test methods | **SAFE** — All use base-only fixtures and `resource_pcts`. |
| 2 | `tests/test_energy_conservation.py:48,74,152,203` | Various test methods | **SAFE** — Base-only fixtures. |

**0 bugs found** — all are test files with base-only fixtures (flagged as test coverage gap in Phase 5).

#### Bug Class 4: `clean_peak`/`gas_backup` calculations

| # | File:Line | Function | Verdict |
|---|-----------|----------|---------|
| 1 | `step2_2b_track_nb_ctr.py:797-811` | `peak_coeff` computation | **OK** — Lines 808-811 explicitly include `solar_batt4`, `solar_batt8`, `wind_batt4`, `wind_batt8` capacity credits. |
| 2 | `step6_2b_nuclear_retirement.py:384` | gas_backup_p50 | **SAFE** — Reads `gas_gas_backup_needed_mw` from step2.2 parquets (passthrough). |
| 3 | `step4_1a_fossil_dispatch.py:106` | imports only | **SAFE** — Imports `PEAK_CAPACITY_CREDITS` but never computes clean_peak directly; delegates to lmp_engine. |
| 4 | `step7_1a_generate_shared_data.py:718+` | gas_backup | **SAFE** — Reads computed values from step2.2 parquets (passthrough). |

**0 bugs found.**

#### NEW BUG FOUND: `get_or_compute_dispatch()` missing `resource_types` pass-through

**`scripts/dispatch_utils.py:1170-1204`** — `get_or_compute_dispatch()` function:
- Does NOT accept a `resource_types` parameter
- Does NOT accept an `h2_dispatch_pct` parameter
- Calls `reconstruct_hourly_dispatch()` at line 1195-1199 without `resource_types`, defaulting to base 6

**Callers affected:**

| # | File:Line | Context | Impact |
|---|-----------|---------|--------|
| 1 | `lmp_engine.py:1442` | `run_lmp_for_iso()` — supply_profiles loaded with `include_hybrids=True` at line 1402, but `get_or_compute_dispatch` dispatches base-only | **MODERATE** — LMP calculations don't account for hybrid generation in dispatch residual. Hybrid energy appears as 0, inflating residual demand and LMP. Affects step4 LMP outputs. |
| 2 | `scenario_common.py:1533` | `_get_dispatch_co2_for_mix()` — supply_profiles loaded with `include_hybrids=True` at line 1530, but `get_or_compute_dispatch` dispatches base-only | **MODERATE** — CO2 displacement calculations ignore hybrid generation. Affects step2.2 CO2 outputs. |

**Compound issue at `scenario_common.py:549-550`:** `compute_mix_cost()` returns `resource_pct` with only base 6 keys (no hybrids). When `_get_dispatch_co2_for_mix` reads this at line 1523, even if `get_or_compute_dispatch` supported `resource_types`, the hybrid percentages wouldn't be there.

**Severity: MODERATE** — Affects LMP and CO2 analysis outputs (Step 4 secondary analysis), NOT the core cost optimization or resource mix selection (which correctly uses `resource_types=RESOURCE_TYPES_HYBRID` in `compute_mix_cost`).

---

### Phase 3: Market-Simulator Audit

All 5 files exist in `market-simulator/scripts/`. The market-simulator uses a different API: `include_hybrids=True/False` flag instead of `resource_types=` parameter. `build_supply_matrix()` has no `resource_types` parameter and always builds a (6, H) matrix.

#### Bug 1 — Zero energy revenue (`get_supply_profiles` without `include_hybrids`)

| # | File:Line | Function | Verdict |
|---|-----------|----------|---------|
| 1 | `procurement_utils.py:553` | `get_merchant_clean_hourly_shape()` | **SAFE** — Existing fleet shape, no hybrids in existing grid |
| 2 | `procurement_utils.py:640` | `get_sss_hourly_shape()` | **SAFE** — SSS fleet shape (nuclear+hydro), no hybrids |
| 3 | `procurement_utils.py:686` | `get_existing_clean_hourly_shape()` | **SAFE** — Existing clean generation, no hybrids |

**0 actionable bugs** — all are intentional base-only loads for existing fleet shapes.

#### Bug 2 — Ghost capacity (`reconstruct_hourly_dispatch` without hybrids)

**0 instances found** — All `reconstruct_hourly_dispatch` calls properly pass `include_hybrids=_has_hybrids` or `include_hybrids=has_hybrids`:
- `lmp_engine.py:1840,1963` — correct
- `scenario_common.py:457,473` — correct
- `procurement_utils.py:1448` — correct
- `market_simulation.py:667,4949` — correct
- `dispatch_utils.py:1544` — correct (pass-through)

**Latent issue:** `_compute_per_resource_dispatch()` (dispatch_utils.py:966) always uses `RESOURCE_TYPES` (base 6). If a caller passes `include_hybrids=True` AND `detailed=True`, per-resource matched/surplus arrays would exclude hybrids. No current caller triggers this.

#### Bug 3 — Oversized fossil fleet (missing hybrid capacity credits)

| # | File:Line | Function | Detail |
|---|-----------|----------|--------|
| **1** | `scenario_common.py:1523` | `build_augmented_result()` | `aug['resource_twh']` built from `RESOURCES` (base 6 only). Hybrid TWh excluded from `recompute_gas_backup()` at line 1537, causing `clean_peak_mw` to undercount hybrid capacity credits. |

**1 bug found.**

#### Bug 4 — Overstated gas backup (supply_matrix shape mismatch)

| # | File:Line | Function | Detail |
|---|-----------|----------|--------|
| **1** | `procurement_utils.py:1370` | `_compute_hms_scores()` | `build_supply_matrix(supply_profiles)` builds (6, H) matrix even when `supply_profiles` loaded with `include_hybrids=True` (line 1369). Downstream `reconstruct_hourly_dispatch` with `include_hybrids=True` (line 1456) creates (10,) mix_weights @ (6, H) matrix = **dimension mismatch crash**. |
| **2** | `scenario_common.py:445` | `compute_mix_cost()` | `_bsm(supply_profiles)` builds (6, H) matrix. `reconstruct_hourly_dispatch` with `include_hybrids=True` (line 463) = **dimension mismatch crash**. |

**2 bugs found** — Both would cause runtime crashes when `include_hybrids=True` and `supply_matrix` is pre-built.

#### Market-Simulator Summary

| Bug Class | Count | Severity |
|-----------|-------|----------|
| Bug 1 (zero energy) | 0 actionable | — |
| Bug 2 (ghost capacity) | 0 + 1 latent | Low (latent) |
| Bug 3 (oversized fossil) | **1** | Moderate |
| Bug 4 (shape mismatch) | **2** | **Critical** (crash) |

---

### Phase 4: Shape Mismatch Check (build_supply_matrix ↔ reconstruct_hourly_dispatch)

#### scripts/ (main pipeline) — 8 code paths checked

| # | File | build_supply_matrix | reconstruct_hourly_dispatch | Match? |
|---|------|--------------------|-----------------------------|--------|
| 1 | `step3a_build_dispatch_cache.py` | `resource_types=rtypes` (line 168) | `resource_types=rtypes` (line 200) | **MATCH** |
| 2 | `scenario_common.py` (compute_mix_cost) | `RESOURCE_TYPES_HYBRID` (line 428) | `RESOURCE_TYPES_HYBRID` (lines 443, 458) | **MATCH** |
| 3 | `procurement_utils.py` (_compute_hms_scores) | `RESOURCE_TYPES_HYBRID` (line 1391) | `RESOURCE_TYPES_HYBRID` (line 1476) | **MATCH** |
| 4 | `step7_1e_dispatch_deployment.py` | `RESOURCE_TYPES_HYBRID` (line 330) | `RESOURCE_TYPES_HYBRID` (lines 117, 177) | **MATCH** |
| 5 | `step3b_mac_queue.py` (run_iso) | `RESOURCE_TYPES_HYBRID` (line 1916) | `RESOURCE_TYPES_HYBRID` (lines 748, 1694) | **MATCH** |
| 6 | `step3b_mac_queue.py` (_build_consequential_queue) | `RESOURCE_TYPES_HYBRID` (line 1979) | N/A (no dispatch here) | **MATCH** |
| 7 | `step3b_mac_queue.py` (_build_nbctr_queue) | `RESOURCE_TYPES_HYBRID` (line 2256) | `RESOURCE_TYPES_HYBRID` (lines 2135, 2184) | **MATCH** |
| 8 | `step7_1g_extract_use_case_data.py` | `RESOURCE_TYPES_HYBRID` (lines 501, 560) | `RESOURCE_TYPES_HYBRID` (lines 538, 580) | **MATCH** |

**8 code paths checked, 0 mismatches in scripts/.**

#### market-simulator/ — 2 code paths checked

| # | File | build_supply_matrix | reconstruct_hourly_dispatch | Match? |
|---|------|--------------------|-----------------------------|--------|
| 1 | `procurement_utils.py` | Always (6, H) (line 1370) | `include_hybrids=_has_hybrids` → (10,) weights when True (line 1456) | **MISMATCH** |
| 2 | `scenario_common.py` | Always (6, H) (line 445) | `include_hybrids=has_hybrids` → (10,) weights when True (line 463) | **MISMATCH** |

**2 code paths checked, 2 mismatches in market-simulator.** Root cause: `build_supply_matrix()` in the market-simulator fork has no `resource_types` parameter and always creates (6, H) matrices.

---

### Phase 5: Test Coverage

- **Hybrid dispatch tested:** No
- **Tests passing:** Not run (audit scope = read-only)
- **Coverage gaps:**
  1. No test for `get_supply_profiles(include_hybrids=True)`
  2. No test for `build_supply_matrix(resource_types=RESOURCE_TYPES_HYBRID)` → (10, 8760) shape
  3. No test for `reconstruct_hourly_dispatch` with hybrid `resource_pcts` + `resource_types=RESOURCE_TYPES_HYBRID`
  4. No test for hybrid dispatch order (10-element extended merit order)
  5. No test for `resource_pcts.get(rt, 0)` hybrid key defaulting
  6. No test for energy conservation with hybrid resources
  7. No test for `_mix_cache_key` hybrid backward compatibility
  8. No test for `HYBRID_TYPES` constant validation in `test_constants.py`
  9. No hybrid profiles in `conftest.py` fixtures (`synthetic_profiles` has base 6 only)
  10. No test for `_compute_per_resource_dispatch` with hybrid resource types

The **entire hybrid dispatch code path** — from profile loading to matrix construction to merit-order dispatch to full reconstruction — has **zero test coverage**.

---

## Summary

### scripts/ (main pipeline)

| Category | Count | Severity |
|----------|-------|----------|
| Fixed files verified clean | 8/8 | — |
| Remaining unhybridized calls (safe) | 8 | N/A (all safe) |
| **NEW: `get_or_compute_dispatch` missing `resource_types`** | **1 function, 2 callers** | **Moderate** |
| **NEW: `compute_mix_cost` `resource_pct` missing hybrid keys** | **1** | **Moderate** |
| Shape mismatches | 0 | — |
| Test coverage gaps | 10 | Low (no correctness risk, but no regression safety net) |

### market-simulator/

| Category | Count | Severity |
|----------|-------|----------|
| Bug 3 (oversized fossil) | **1** | Moderate |
| Bug 4 (shape mismatch → crash) | **2** | **Critical** |
| Latent issue (detailed dispatch) | 1 | Low |

### Total Remaining Bugs Found

- **Critical (data corruption / crash risk):** 2 (market-simulator shape mismatches)
- **Moderate (incorrect results):** 4 (1× `get_or_compute_dispatch` + 1× `resource_pct` in scripts/; 1× oversized fossil + 1× latent in market-simulator)
- **Low (test gaps, no immediate impact):** 10 test coverage gaps

### Recommended Fixes (Priority Order)

1. **`dispatch_utils.py:get_or_compute_dispatch`** — Add `resource_types=None` and `h2_dispatch_pct=0` parameters, pass through to `reconstruct_hourly_dispatch`.
2. **`scenario_common.py:549-550`** — Add hybrid keys to `resource_pct` dict (same pattern as `resource_pcts.update(hybrid_pcts)` at line 436).
3. **`market-simulator/scripts/dispatch_utils.py:build_supply_matrix`** — Add `resource_types=None` parameter matching the scripts/ API.
4. **`market-simulator/scripts/scenario_common.py:1523`** — Use `RESOURCES_WITH_HYBRIDS` instead of `RESOURCES`.
5. **Test coverage** — Add hybrid dispatch test fixtures and tests (lower priority, but needed for regression safety).
