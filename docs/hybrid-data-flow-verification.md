# Hybrid Data Flow Verification Report

**Date:** 2026-04-06
**Session:** End-to-End Data Flow Verification (Session 4)
**Scope:** Trace hybrid resource data from Step 3 dispatch cache through Step 7 dashboard output

## Compatibility Table

| # | Check | Status | Details |
|---|-------|--------|---------|
| 1 | Step 3 dispatch cache schema | PASS | `CAISO_dispatch_cache.parquet` contains `matched_solar_batt4/8`, `surplus_solar_batt4/8`, `matched_wind_batt4/8`, `surplus_wind_batt4/8`, `battery4_charge`, `battery8_charge`. |
| 2 | Step 3 annual manifest | PASS | `mix_solar_batt4/8`, `mix_wind_batt4/8` + dispatch/surplus pct columns present for all 4 hybrid types. |
| 3 | Step 3B MAC queue hybrid columns | PASS | `winner_*`, `new_build_*`, `floor_*`, `deployed_*` columns present for all 4 hybrid types. |
| 4 | Step 3B MAC queue — CAISO | WARN | No `mac_queue_CAISO.parquet`. All other 6 ISOs present. Likely intentional (geothermal dimension). |
| 5 | Step 5 strategy output | WARN | `data/step5-scenarios/` is empty. Not yet computed — not data loss. |
| 6 | Step 7 shared-data.js — hybrid keys | **FAIL** | 0 occurrences of `solar_batt4/8` or `wind_batt4/8` in `shared-data.js`. `MIX_RESOURCES` missing all 4 hybrid types. The `step7_1a` script writes them (line 848) but the file is stale. **Re-run step7_1a needed.** |
| 7 | Step 7.1H hybrid-analysis-data.js — NYISO | **WARN** | File exists but excludes NYISO. `ISOS` list at line 25 of `step7_1h_extract_hybrid_data.py` omits NYISO despite NYISO having 28,530 non-zero `mix_solar_batt4` rows in step2.2 parquets. **Add NYISO to ISOS list and re-run.** |
| 8 | Hybrid profile NPZ files | PASS | All 7 ISOs have NPZ files with 4 keys each (`solar_batt4/8`, `wind_batt4/8`), shape `(8760,)`, valid values. |
| 9 | dispatch_utils.py — hybrid loading | PASS | `_load_hybrid_profiles()` loads NPZ correctly. `get_supply_profiles()` injects hybrids when `include_hybrids=True`. Curtailment order correct. |
| 10 | scenario_common.py — RESOURCES_WITH_HYBRIDS | PASS | Line 60: `RESOURCES_WITH_HYBRIDS = RESOURCES + list(HYBRID_TYPES)`. |
| 11 | Backward compatibility (.get defaults) | PASS | All scripts use `.get('mix_solar_batt4', 0)` pattern. No hard lookups that crash on missing hybrid columns. |

## Issues Requiring Action

### 1. FAIL: shared-data.js is stale (no hybrid resource keys)

**Impact:** Dashboard cannot display hybrid resource data (solar+batt, wind+batt) until step7_1a is re-run.

**Current state:** `MIX_RESOURCES` = `['clean_firm', 'geothermal', 'hydro', 'ccs_ccgt', 'offshore_wind', 'wind', 'solar', 'battery', 'battery8', 'ldes', 'h2']`

**Expected after re-run:** `MIX_RESOURCES` includes `'wind_batt4', 'wind_batt8', 'solar_batt4', 'solar_batt8'`

**Fix:** Run `python3 scripts/step7_1a_generate_shared_data.py`

### 2. WARN: NYISO excluded from step7_1h_extract_hybrid_data.py

**Impact:** `hybrid-analysis-data.js` has no NYISO data, despite NYISO having significant hybrid adoption in step2.2 results.

**Evidence:** NYISO step2.2 parquet has 28,530 non-zero `mix_solar_batt4` rows out of 116,640 total (24.5% adoption rate).

**Fix:** Add `'NYISO'` to `ISOS` list at line 25 of `step7_1h_extract_hybrid_data.py` and re-run.

## Data Flow Summary

```
Step 1 (PFS) → Step 2.2 (Cost Opt) → Step 3 (Dispatch Cache) → Step 4 (Analysis) → Step 7 (Dashboard JS)
     ✓              ✓                      ✓                       ✓                   ✗ (stale)
```

Hybrid columns flow correctly through Steps 1→4. The only breakage is at the Step 7 output layer where JS files haven't been regenerated since hybrids were integrated.

No silent data loss at handoff boundaries between steps. The `.get(col, 0)` default pattern ensures backward compatibility at all read points.
