# Hybrid Resource Pipeline Integration Playbook

## Context

Hybrid co-located resources (`solar_batt4`, `solar_batt8`, `wind_batt4`, `wind_batt8`) were added to steps 1 & 2 of the pipeline. This playbook assesses compatibility across steps 3-7 and provides individual session prompts to ensure end-to-end integration.

The canonical definition lives in `scripts/pipeline_config.py:58`:
```python
HYBRID_TYPES = ['solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8']
```

---

## Executive Summary

**The pipeline is broadly compatible.** Hybrid integration was done comprehensively across 27+ scripts. Four categories of remaining work:

1. **One confirmed bug**: `step4_2c_analyze_tracks.py` loads hybrid data but silently drops it from report output
2. **Two missing integrations**:
   - `step7_1h_extract_strategy_comparison.py` has zero hybrid references -- VRE totals are understated in strategy comparison dashboard
   - `procurement_utils.py` `build_newbuild_only_tranches()` has no hybrid tranches -- Strategies 3A & 3C in step5_2d underprice hybrids by routing them through generic VRE tranche
3. **Hardcoded resource lists in 5 scripts** create maintenance risk for future hybrid type additions
4. **One fragile passthrough**: `step7_1e_dispatch_deployment.py` `_map_resource()` has no explicit hybrid mappings (currently works via fallthrough but is brittle)

---

## Compatibility Matrix

| Script | Step | Status | Risk | Notes |
|--------|------|--------|------|-------|
| `step3a_build_dispatch_cache.py` | 3A | FULLY COMPATIBLE | None | Dynamic detection, vectorized extraction |
| `step3b_mac_queue.py` | 3B | FULLY COMPATIBLE | None | HYBRID_TYPES import, family caps, cost calc |
| `step4_1a_fossil_dispatch.py` | 4.1A | IMPLICIT OK | None | Dynamic `mix_*` column detection |
| `step4_1a_augment_capacity_rev.py` | 4.1A | OK | **Maintenance** | `VRE_RES` hardcoded (lines 33-34) |
| `step4_1b_compress_day_profiles.py` | 4.1B | OK | None | HYBRID_TYPES import |
| `step4_1c_compute_mac_stats.py` | 4.1C | N/A | None | Scalar operations only |
| `step4_1d_compute_optimal_targets.py` | 4.1D | OK | **Maintenance** | Local RESOURCES list (line 113) |
| `step4_1e_export_tracks.py` | 4.1E | OK | None | Dynamic with defaults |
| `step4_2a_extract_resource_density.py` | 4.2A | OK | None | Dynamic column checking |
| `step4_2b_analyze_storage.py` | 4.2B | OK | **Maintenance** | RESOURCES + MIX_COL_MAP hardcoded (lines 83-89) |
| `step4_2c_analyze_tracks.py` | 4.2C | **BUG** | **Critical** | Data loaded but NOT printed (lines 278-290) |
| `step5_2b_strategy_consequential.py` | 5.2B | OK | None | HYBRID_TYPES import |
| `step5_2c_strategy_hourly.py` | 5.2C | OK | Low | Inline fallback fractions include hybrids |
| `step5_2d_strategy_annual.py` | 5.2D | **ASYMMETRIC** | **Critical** | 3B/3D handle hybrids; 3A/3C missing hybrid tranches |
| `step5_2e_wrights_law_curves.py` | 5.2E | OK | None | HYBRID_TYPES import |
| `procurement_utils.py` | Shared | **PARTIAL** | **Critical** | get_hybrid_lcoe/tx OK; `build_newbuild_only_tranches()` missing hybrids |
| `step6_1_smartargets.py` | 6.1 | OK | **Maintenance** | Hardcoded in 5 locations |
| `step6_1b_dashboard_data.py` | 6.1B | OK | None | Dynamic column extraction |
| `step6_2a_ipp_smartargets.py` | 6.2A | OK | None | Detailed hybrid parameters |
| `step6_2b_nuclear_retirement.py` | 6.2B | N/A | None | Nuclear economics only |
| `step7_1a_generate_shared_data.py` | 7.1A | OK | None | Very explicit: labels, colors, caps |
| `step7_1b_extract_deployment_data.py` | 7.1B | OK | None | Hybrid type mapping |
| `step7_1c_generate_foak_noak.py` | 7.1C | OK | None | HYBRID_TECHS with FOAK/NOAK |
| `step7_1e_dispatch_deployment.py` | 7.1E | **FRAGILE** | Low | Hybrid CF + capacity credits OK; `_map_resource()` missing explicit mappings |
| `step7_1f_extract_hourly_comparison.py` | 7.1F | OK | None | Resources list includes all 4 |
| `step7_1g_extract_use_case_data.py` | 7.1G | OK | None | Mix columns with defaults |
| `step7_1h_extract_hybrid_data.py` | 7.1H | DEDICATED | None | Entire script for hybrid dashboard |
| `step7_1h_extract_strategy_comparison.py` | 7.1H | **MISSING** | **Critical** | categorize_resources() ignores hybrids |
| `step7_2_extract_no_regrets.py` | 7.2 | OK | None | Resources list includes all 4 |

---

## Priority Order

| Priority | Session | Effort | Impact |
|----------|---------|--------|--------|
| 1 (Critical) | Session 1: Strategy comparison hybrid gap | Small | VRE totals wrong in dashboard |
| 2 (Critical) | Session 2: Procurement tranches missing hybrids | Medium | Strategies 3A/3C underprice hybrids |
| 3 (Critical) | Session 3: Track analysis report bug | Small | Report output incomplete |
| 4 (Medium) | Session 4: E2E data flow verification | Medium | Pipeline correctness confidence |
| 5 (Medium) | Session 5: Step 4 hardcoded lists | Small | Maintenance risk reduction |
| 6 (Medium) | Session 6: Step 6 hardcoded lists | Small | Maintenance risk reduction |
| 7 (Low) | Session 7: Dashboard rendering audit + step7_1e mapping | Medium | UI correctness verification |

---

## Session Prompts

---

### Session 1: Add Hybrid Support to step7_1h Strategy Comparison

**Goal**: Fix missing hybrid integration in strategy comparison extractor. VRE totals are currently understated because hybrid TWh falls through uncounted.

**Prompt**:

```
Read scripts/step7_1h_extract_strategy_comparison.py. This script has ZERO references to hybrid
resource types (solar_batt4, solar_batt8, wind_batt4, wind_batt8). The categorize_resources()
function (~lines 78-118) only recognizes solar, wind, offshore_wind as VRE. Hybrid TWh falls
through uncounted, understating VRE totals in the strategy comparison dashboard.

Fix:
1. Import HYBRID_TYPES from pipeline_config.
2. In categorize_resources(), add hybrid types to the VRE branch. Cleanest approach:
   `if rname in ("solar", "wind", "offshore_wind") or rname in HYBRID_TYPES:`
3. Note: line ~94 already normalizes with .replace("-", "_"), so naming should be fine.
4. Verify upstream: check step5_2c_strategy_hourly.py output to confirm hybrid keys exist
   in the resource_mix dicts that this script consumes.

Reference pattern: see how step7_1h_extract_hybrid_data.py (at repo root) handles hybrid data.
Also check: the ISOS list at line 26 of step7_1h_extract_hybrid_data.py excludes NYISO -- verify
this is intentional.

Verify: run `grep -n "solar\|wind\|hybrid\|batt" scripts/step7_1h_extract_strategy_comparison.py`
before and after to confirm coverage. Commit to branch.
```

**Key files**:
- `scripts/step7_1h_extract_strategy_comparison.py` -- categorize_resources() ~lines 78-118
- `scripts/pipeline_config.py` -- HYBRID_TYPES at line 58
- `scripts/step5_2c_strategy_hourly.py` -- upstream data producer
- `step7_1h_extract_hybrid_data.py` -- reference pattern (repo root)

---

### Session 2: Add Hybrid Tranches to Procurement Utils (Strategies 3A/3C)

**Goal**: Fix missing hybrid resource tranches in `build_newbuild_only_tranches()` and `build_procurement_tranches()`. Without this, Strategies 3A and 3C in step5_2d underprice hybrids by routing them through the generic `new_build_vre` tranche (priced at min of solar/wind PPA), ignoring the battery cost component.

**Prompt**:

```
Read scripts/procurement_utils.py. Two functions are missing hybrid resource tranches:

1. build_newbuild_only_tranches() (~lines 934-988):
   - Currently builds tranches: uprate -> new_build_vre (solar+wind averaged) -> new_build_firm
   - Missing: dedicated tranches for solar_batt4, solar_batt8, wind_batt4, wind_batt8
   - Hybrids have HIGHER costs than base solar/wind (include battery LCOS component)
   - They should NOT be lumped into the generic VRE tranche

2. build_procurement_tranches() (~lines 847-928):
   - Same issue -- check if hybrid tranches are needed here too

Contrast with correct handling:
- step5_2d_strategy_annual.py Strategies 3B (line ~221) and 3D (line ~495) MANUALLY iterate
  over HYBRID_TYPES to build proper hybrid tranches. Strategies 3A and 3C use the tranche
  builder functions and thus miss hybrids.
- get_resource_ppa_price() (~lines 1091-1134) already correctly handles hybrid pricing
  via get_hybrid_lcoe() + get_hybrid_tx()

Fix:
1. Import HYBRID_TYPES from pipeline_config (if not already imported).
2. In build_newbuild_only_tranches(), after the VRE tranche, add one tranche per hybrid type
   priced via get_resource_ppa_price(). Insert in merit order (cheapest first).
3. Apply same pattern to build_procurement_tranches() if applicable.
4. Verify step5_2d Strategies 3A and 3C will now pick up hybrid tranches automatically.

Verify: Run a dry comparison -- print the tranche list before and after for one ISO/scenario
to confirm hybrids appear at correct price points. Commit to branch.
```

**Key files**:
- `scripts/procurement_utils.py` -- `build_newbuild_only_tranches()` lines 934-988, `build_procurement_tranches()` lines 847-928
- `scripts/step5_2d_strategy_annual.py` -- Strategies 3B/3D as reference for correct handling
- `scripts/pipeline_config.py` -- `get_hybrid_lcoe()`, `get_hybrid_tx()`

---

### Session 3: Fix step4_2c Report Output Bug

**Goal**: Fix confirmed bug where hybrid resource data is loaded into RESOURCES but excluded from the printed report output.

**Prompt**:

```
Read scripts/step4_2c_analyze_tracks.py. At line 31-32, RESOURCES includes all 4 hybrid types
(solar_batt4, solar_batt8, wind_batt4, wind_batt8). However, the print-format report at lines
~253-290 only formats 6 columns (CF, Sol, Wnd, OSW, CCS, Hyd). The delta dict at line ~286
computes values for ALL resources including hybrids, but lines ~287-290 only print the base 6.

Fix:
1. Add hybrid columns to the header line (~line 253). Use abbreviated labels: SB4, SB8, WB4, WB8.
2. Add hybrid values to baseline print, newbuild print, and delta print lines.
3. Widen the separator line accordingly.
4. Replace the hardcoded RESOURCES list with:
   `from pipeline_config import HYBRID_TYPES` and
   `RESOURCES = ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro'] + list(HYBRID_TYPES)`

Verify: grep for every element in RESOURCES and confirm it appears in the print format strings.
Commit to branch.
```

**Key files**:
- `scripts/step4_2c_analyze_tracks.py` -- RESOURCES at lines 31-32, print format at lines 253-290
- `scripts/pipeline_config.py` -- HYBRID_TYPES import pattern

---

### Session 4: End-to-End Data Flow Verification

**Goal**: Trace hybrid data from step 3 dispatch cache through step 7 dashboard output. Verify no silent data loss at handoff boundaries.

**Prompt**:

```
This is a READ-ONLY verification session. Do NOT modify code -- only inspect and report.

Trace hybrid resource data flow across the full pipeline:

1. Step 3 output schema: If parquets exist in data/step3-dispatch/, inspect one:
   python3 -c "import pandas as pd; df = pd.read_parquet('data/step3-dispatch/CAISO_dispatch_manifest.parquet'); print([c for c in df.columns if 'batt' in c or 'hybrid' in c.lower()])"
   Expected: mix_solar_batt4, mix_solar_batt8, mix_wind_batt4, mix_wind_batt8 columns

2. Step 3B MAC queue output: Check data/step3-dispatch/ for MAC parquets with hybrid columns.

3. Step 5 strategy output: Check data/step5-scenarios/ for hybrid keys in strategy structures.

4. Step 7 dashboard JS: Check dashboard/js/shared-data.js for hybrid entries.

5. Hybrid profile NPZ files: Verify data/hybrid_profiles/ exists and contains loadable NPZ files:
   python3 -c "import numpy as np; d = np.load('data/hybrid_profiles/CAISO_hybrid_profiles.npz'); print(list(d.keys()))"

6. Cross-step backward compatibility: Verify scripts that read parquets from prior steps handle
   missing hybrid columns gracefully (look for .get(col, 0) or default=0 patterns).

7. Check step7_1h_extract_hybrid_data.py (at repo root): its ISOS list excludes NYISO --
   determine if this is intentional by checking if NYISO has step2.2 parquets.

Report findings as a compatibility table with PASS/FAIL/WARN per check.
```

**Key files**:
- `data/step3-dispatch/` -- dispatch cache parquets
- `data/hybrid_profiles/` -- hybrid profile NPZ files
- `data/step5-scenarios/` -- strategy output data
- `dashboard/js/shared-data.js` -- step 7 JS output
- `scripts/dispatch_utils.py` -- _load_hybrid_profiles() function
- `scripts/scenario_common.py` -- RESOURCES_WITH_HYBRIDS at line 60

---

### Session 5: Consolidate Hardcoded Lists in Step 4

**Goal**: Replace hardcoded hybrid resource lists with HYBRID_TYPES imports in 3 step-4 scripts to reduce maintenance risk.

**Prompt**:

```
Three step-4 scripts hardcode hybrid resource lists instead of importing HYBRID_TYPES from
pipeline_config. Refactor each to use the canonical source of truth.

1. scripts/step4_1a_augment_capacity_rev.py (lines 33-34):
   - VRE_RES hardcodes all 4 hybrids inline
   - Add `from pipeline_config import HYBRID_TYPES`
   - Replace with: `VRE_RES = ['solar', 'wind', 'offshore_wind'] + list(HYBRID_TYPES)`

2. scripts/step4_2b_analyze_storage.py (lines 83-89):
   - RESOURCES and MIX_COL_MAP both hardcode all 4 hybrids
   - Import HYBRID_TYPES (check existing import pattern at top of file)
   - Construct RESOURCES and MIX_COL_MAP dynamically

3. scripts/step4_1d_compute_optimal_targets.py (line 113):
   - RESOURCES list includes hybrids inline (already imports HYBRID_TYPES at line 72)
   - Replace: `RESOURCES = [...base...] + list(HYBRID_TYPES)`

Reference: scripts/step4_1b_compress_day_profiles.py is a good example of the import pattern.

Verify: After each change, run `grep -c "solar_batt4" <file>` -- literal string should only appear
in comments/docstrings, not in list definitions. Check no import cycles introduced.
Commit to branch.
```

**Key files**:
- `scripts/step4_1a_augment_capacity_rev.py` -- VRE_RES at lines 33-34
- `scripts/step4_2b_analyze_storage.py` -- RESOURCES at lines 83-84, MIX_COL_MAP at lines 86-89
- `scripts/step4_1d_compute_optimal_targets.py` -- RESOURCES at line 113
- `scripts/step4_1b_compress_day_profiles.py` -- reference pattern for import
- `scripts/pipeline_config.py` -- HYBRID_TYPES at line 58

---

### Session 6: Consolidate Hardcoded Lists in Step 6 (SMARTargets)

**Goal**: Replace 5 inline hybrid type lists in step6_1_smartargets.py with HYBRID_TYPES references.

**Prompt**:

```
scripts/step6_1_smartargets.py has 5 locations where hybrid types are listed inline instead of
using HYBRID_TYPES from pipeline_config.

Specific locations:
1. Line ~255: REC_ELIGIBLE set literal includes all 4 hybrids ->
   replace with: `{'solar', 'wind', 'offshore_wind', 'hydro', 'geothermal'} | set(HYBRID_TYPES)`

2. Lines ~1030-1031: PPA category check `if res in ('solar', 'wind', 'offshore_wind',
   'solar_batt4', ...)` -> replace with: `if res in ('solar', 'wind', 'offshore_wind') or res in HYBRID_TYPES`

3. Lines ~1133-1137: Hybrid resource extraction loop uses inline list -> replace with `HYBRID_TYPES`

4. Lines ~2074-2076: Parquet export resource list -> construct from base list + HYBRID_TYPES

5. Lines ~2341-2343: Same pattern as #4 in batch export -> construct dynamically

Verify HYBRID_TYPES is already imported (check existing pipeline_config imports at top of file).
After refactoring: `grep -c "solar_batt4" scripts/step6_1_smartargets.py` should show only
comments/docstrings, not list definitions. Verify parquet output schema unchanged.
Commit to branch.
```

**Key files**:
- `scripts/step6_1_smartargets.py` -- 5 locations identified above
- `scripts/pipeline_config.py` -- HYBRID_TYPES source of truth

---

### Session 7: Dashboard-Side Hybrid Rendering Audit + step7_1e Mapping

**Goal**: Verify dashboard HTML/JS correctly consumes and renders hybrid data from step 7 outputs. Also fix the fragile `_map_resource()` passthrough in step7_1e.

**Prompt**:

```
Two tasks in this session:

TASK A: Audit the dashboard consumer side for hybrid resource rendering. READ-ONLY verification.

1. Check dashboard/js/chart-colors.js for hybrid resource color definitions:
   - RESOURCE_COLORS.solarBatt4, solarBatt8, windBatt4, windBatt8 should exist
   - Verify they match the CSS variables in dashboard/styles/shared.css

2. Check dashboard/js/shared-data.js for:
   - MIX_RESOURCES array includes all 4 hybrid types
   - RESOURCE_LABELS has display names for hybrids
   - RESOURCE_STYLES has fill/border colors for hybrids
   - RESOURCE_CAPS notes hybrids (no separate physical cap)

3. Check dashboard HTML files that render resource mix charts:
   - dashboard/dashboard.html -- does the stacked area/bar handle 14+ resource types?
   - dashboard/index.html -- scrollytell resource visualizations

4. Check dashboard/js/strategy-comparison-data.js (if it exists) -- will it show correct VRE
   totals after Session 1 fixes?

5. Verify step7_1h_extract_hybrid_data.py output: dashboard/js/hybrid-analysis-data.js --
   confirm the JS is syntactically valid and consumed by a dashboard page.

Report: which dashboard pages properly render hybrid data, which might truncate or ignore it.

TASK B: Fix step7_1e_dispatch_deployment.py _map_resource() fragile passthrough.

Read scripts/step7_1e_dispatch_deployment.py. The _map_resource() function (~lines 280-294) maps
deployment resource names to dispatch utility names. It has no explicit entries for hybrid types
(solar_batt4, solar_batt8, wind_batt4, wind_batt8). They currently fall through to the default
passthrough (line ~294), which happens to work but is fragile.

Fix: Add explicit mappings for all 4 hybrid types. Check whether dispatch_utils.py handles
'solar_batt4' natively (it does -- they're in RESOURCE_TYPES_HYBRID). So the mapping should be
identity: 'solar_batt4' -> 'solar_batt4'. Add them explicitly for documentation and safety.

Verify: grep for _map_resource in the file and confirm all resource types have explicit entries.
```

**Key files**:
- `dashboard/js/chart-colors.js` -- RESOURCE_COLORS definitions
- `dashboard/styles/shared.css` -- CSS color variables
- `dashboard/js/shared-data.js` -- MIX_RESOURCES, RESOURCE_LABELS, RESOURCE_STYLES
- `dashboard/dashboard.html` -- main optimizer dashboard
- `dashboard/js/hybrid-analysis-data.js` -- hybrid-specific dashboard data
- `step7_1h_extract_hybrid_data.py` -- hybrid data extractor (repo root)
- `scripts/step7_1e_dispatch_deployment.py` -- `_map_resource()` at lines 280-294

---

## Critical Files Summary

| File | Role |
|------|------|
| `scripts/pipeline_config.py` | Canonical HYBRID_TYPES source of truth (line 58) |
| `scripts/dispatch_utils.py` | Core dispatch with hybrid support (RESOURCE_TYPES_HYBRID) |
| `scripts/scenario_common.py` | RESOURCES_WITH_HYBRIDS shared constant |
| `scripts/step7_1h_extract_strategy_comparison.py` | **Zero hybrid refs -- Session 1** |
| `scripts/procurement_utils.py` | **Missing hybrid tranches in tranche builders -- Session 2** |
| `scripts/step5_2d_strategy_annual.py` | Strategies 3A/3C affected by Session 2 fix |
| `scripts/step4_2c_analyze_tracks.py` | **Report output bug -- Session 3** |
| `scripts/step6_1_smartargets.py` | 5 hardcoded locations -- Session 6 |
| `scripts/step4_2b_analyze_storage.py` | Hardcoded lists -- Session 5 |
| `scripts/step4_1a_augment_capacity_rev.py` | Hardcoded VRE_RES -- Session 5 |
| `scripts/step7_1e_dispatch_deployment.py` | Fragile _map_resource() -- Session 7 |
