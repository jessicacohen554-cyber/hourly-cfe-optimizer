# New-Build Fossil Integration Audit & Implementation Prompts

> **Created**: 2026-03-19
> **Scope**: Audit of downstream impacts after `apply_economic_new_build()` was added to the market-simulator dispatch engine
> **Status**: Audit complete — implementation prompts ready for execution
> **Last Updated**: 2026-03-19 — Updated to reflect fleet config sidebar + client-side dispatch engine (commit `66d5ca7`)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [ORDC Scarcity Pricing Analysis](#2-ordc-scarcity-pricing-analysis)
3. [Compatibility Matrix](#3-compatibility-matrix)
4. [Implementation Prompts](#4-implementation-prompts)
5. [Testing Schema](#5-testing-schema)
6. [Risk Register](#6-risk-register)

---

## 1. Executive Summary

### What Changed

New-build fossil capability was added to the market-simulator dispatch engine via `apply_economic_new_build()` in `market_simulation.py` (lines 1054–1265). This function evaluates and applies economic new-build fossil capacity (CCGT, CT, coal) based on two triggers:

1. **Reserve adequacy (RA)**: Reserve margin falls below target after retirements + clean deployment
2. **Economic viability**: LMP levels support positive margins for new plants above minimum CF threshold

### Impact on Parametric Sweep

The addition of `new_fossil_cost_level` (Low/Medium/High) as the 6th sweep dimension expanded the scenario count:

| Dimension | Levels | Count |
|-----------|--------|-------|
| Demand growth | 3 | 3 |
| Price sensitivity | 5 | 15 |
| PPA level | 3 | 45 |
| Gas friction | 3 | 135 |
| Queue pace | 3 | 405 |
| **New fossil cost** | **3** | **1,215** |

Total simulations: **1,215 scenarios × 7 ISOs × 6 years = 51,030 ISO-year runs**

### New Output Fields

The sweep now produces these additional columns in the flat parquet:

| Column | Type | Description |
|--------|------|-------------|
| `nb_{fuel}_mw` | float | New-build capacity by fuel type (e.g., `nb_gas_ccgt_mw`, `nb_gas_ct_mw`) |
| `total_new_fossil_mw` | float | Total new fossil MW built in that year |
| `gas_built_gw` | float | Cumulative gas-type builds (GW) |
| `fossil_built_gw` | float | Cumulative all-fossil builds (GW) |

### Scope of Impact

The new-build fossil capability touches dispatch ordering, LMP formation, and scenario output schema. Several downstream modules were built before this capability existed and need updates:

- **`fleet_dispatch.py`** — Does not ingest new fossil columns; fleet emissions ignore grid-level new fossil builds
- **`run_sweep_405.py`** — Filename and docstrings say "405" but script runs 1,215 scenarios
- **`constellation_scenarios.json`** — Fleet-level `add_plant` actions don't interact with grid-level `apply_economic_new_build()`. Now more urgent: the fleet config sidebar enables interactive plant additions, compounding the double-counting risk.
- **`results/sweep_405/`** — Results are pre-ORDC fix AND pre-new-fossil (doubly stale)
- **`backend/main.py`** — Already correctly points to `results/sweep_1215/` (no change needed)
- **`frontend/js/fleet-dispatch-engine.js`** *(NEW)* — Client-side JS port of `fleet_dispatch.py`. Matches Python constants exactly (heat rates, emission factors, CCS ramp). Validated: baseline 2030 P50 = 32.81 Mt vs server 32.8062 Mt. However, does **not** ingest `nb_*` or `total_new_fossil_mw` columns — same gap as Python version.
- **`frontend/js/fleet-sidebar.js`** *(NEW)* — Fleet config sidebar for interactive plant modifications (Operating/Retired/CCS Retrofit), capacity editing, new plant addition, and instant browser-side dispatch recalculation via `FleetDispatchEngine`. localStorage persistence for custom scenarios.
- **`frontend/js/fleet-scenarios.js`** *(MODIFIED)* — Exposed `FLEET_SCENARIOS_API.addScenario()` for sidebar integration; all 4 charts (fan, waterfall, gen mix, fuel emissions) update instantly with custom fleet configurations.

---

## 2. ORDC Scarcity Pricing Analysis

### Verdict: KEEP the fix, tune parameters post-sweep

The exponential-knee ORDC fix (commit `463ada5`) is **correct and must be retained**. Reverting it would break the new-fossil feedback loop and produce unrealistic market outcomes.

### Background

The Operating Reserve Demand Curve (ORDC) determines scarcity pricing — the adder to LMP when reserve margins are thin. The original implementation used a sigmoid curve that produced pathological behavior: average LMPs of $200+ across all scenarios, far exceeding real-world ISOs where average LMPs are typically $25–50/MWh.

The exponential-knee fix replaced the sigmoid with a curve that:
- Produces **realistic baseline LMPs** ($26–42/MWh average depending on ISO)
- Still generates **scarcity spikes** when reserves drop below critical thresholds
- Has tunable `knee_mw` and `cap` parameters per ISO

### Why New-Build Fossil Makes ORDC More Important

New-build fossil creates a **stabilizing feedback loop** that depends on correct ORDC behavior:

```
Low reserves → ORDC scarcity spikes → New fossil becomes economically viable
    → New fossil builds → Reserves restored → ORDC calms down
```

With the buggy sigmoid (pre-fix):
- Permanently elevated LMPs ($200+) would make new fossil **always** viable regardless of reserves
- Result: runaway fossil builds in every scenario — unrealistic and scientifically indefensible

With the exponential-knee fix:
- New fossil builds only trigger when scarcity is genuine
- Once built, new capacity suppresses scarcity pricing — self-correcting equilibrium
- Different `new_fossil_cost_level` settings (L/M/H) shift the viability threshold appropriately

### Potential Tuning Needed

With new fossil adding supply capacity, the scarcity hour distribution will shift. After running the 1,215 sweep:

1. **Extract `ordc_scarcity_hours` distribution** across all scenarios
2. **Compare with/without new fossil** — scarcity hours should decrease when new fossil builds occur (expected)
3. **Verify ORDC_PARAMS caps** — particularly for ERCOT where the energy-only market makes ORDC most impactful
4. **Check for over-suppression** — if new fossil eliminates ALL scarcity hours, the `knee_mw` thresholds may be too aggressive (new builds arrive too easily)

The calibration check is captured as **Prompt 4** in the implementation plan below.

---

## 3. Compatibility Matrix

File-by-file assessment of new-build fossil integration status across the codebase.

### Legend

| Symbol | Meaning |
|--------|---------|
| :white_check_mark: | Fully compatible — no changes needed |
| :warning: | Partially compatible — cosmetic or documentation updates needed |
| :x: | Incompatible — functional changes required |

### Assessment

#### `scripts/run_sweep_405.py` — :warning: Cosmetic Updates Needed

| Aspect | Status | Detail |
|--------|--------|--------|
| Sweep logic | :white_check_mark: | Correctly iterates 1,215 scenarios with `new_fossil_cost_level` as 6th dimension |
| Parquet output | :white_check_mark: | Exports `nb_{fuel}_mw`, `total_new_fossil_mw`, `gas_built_gw`, `fossil_built_gw` correctly |
| Aggregation | :white_check_mark: | Percentile bands computed across full 1,215 sweep |
| **Filename** | :warning: | File is named `run_sweep_405.py` but runs 1,215 scenarios — misleading |
| **Docstring** | :warning: | Line 2: says "405" in several places; line 10 CLI example uses `sweep_405` paths |
| **Output paths** | :warning: | Default `--output-dir` is `../results/sweep_405` — should be `sweep_1215` |

**Fix**: Update docstrings, comments, and default output path references. Filename rename is optional (tracked separately).

---

#### `scripts/fleet_dispatch.py` — :x: Functional Changes Required

| Aspect | Status | Detail |
|--------|--------|--------|
| Sweep loading | :white_check_mark: | `load_sweep()` reads any parquet with `iso`, `year`, `scenario` columns — auto-scales |
| **Docstring** | :warning: | Lines 5, 9, 14, 20: all say "405-scenario" — needs updating to "1,215-scenario" |
| **`load_sweep()` docstring** | :warning: | Line 136: says "405-sweep flat parquet" |
| **New fossil columns** | :x: | Zero references to `nb_*`, `total_new_fossil_mw`, `gas_built_gw`, or `fossil_built_gw` |
| **Fleet emissions** | :x: | Ignores grid-level new fossil builds — underestimates fleet generation impacts |

**Impact**: Fleet emissions calculations are incomplete. When grid-level new fossil builds occur (especially new gas), they increase supply and depress margins for incumbent fleet plants. Without ingesting these columns, fleet dispatch overestimates CF/margins for existing plants in high-new-build scenarios.

**Fix needed**:
1. Ingest new fossil columns from sweep parquet
2. Adjust incumbent plant CF/margins based on new fossil supply additions (new gas entrants compress margins)
3. Include new fossil generation in total grid supply when computing fleet plant dispatch fractions

---

#### `fleet_scenarios/constellation_scenarios.json` — :x: Reconciliation Required

| Aspect | Status | Detail |
|--------|--------|--------|
| Schema | :white_check_mark: | Phase 3.1 schema is valid and well-structured |
| `add_plant` actions | :x: | "CCS + New Gas" scenario adds static `add_plant` entries that don't interact with grid-level `apply_economic_new_build()` |
| Targets | :white_check_mark: | SBTi, AT Power NZ, custom targets — unaffected |
| Base fleet | :white_check_mark: | Plant data unaffected |

**Impact**: Two independent "new gas" mechanisms exist:
1. **Grid-level** (`apply_economic_new_build()`): Market-driven, triggered by reserve/economic conditions, varies by scenario
2. **Fleet-level** (`add_plant` in constellation_scenarios.json): User-defined, static, applied regardless of market conditions

These mechanisms are unaware of each other. In a scenario where both trigger, the combined new gas capacity could be unrealistically high (double-counting). Conversely, fleet-level additions that overlap with grid-level builds don't net out.

> **Update (2026-03-19):** The fleet config sidebar (`fleet-sidebar.js`) now enables **interactive** plant additions, making this double-counting risk more visible and immediate. Users can add gas plants via the sidebar while the sweep already includes grid-level new builds from `apply_economic_new_build()`. Reconciliation (Prompt 3) is now more urgent.

**Fix needed**: Define interaction rules — see Prompt 3.

---

#### ~~`docs/Fleet_Scenarios_Viz_Design.md`~~ — Deleted (Implemented)

> **Status update (2026-03-19):** This design doc was fully implemented as the live `fleet-scenarios.html` page with sidebar and then deleted (commit `eff7c4c`). No longer a downstream artifact. The remaining gap — grid-level `new_fossil_cost_level` L/M/H toggle in the controls strip — is tracked below in Prompt 7's status note.

---

#### `frontend/js/fleet-dispatch-engine.js` — :warning: Partially Compatible (NEW)

| Aspect | Status | Detail |
|--------|--------|--------|
| Dispatch logic | :white_check_mark: | Correct JS port of `fleet_dispatch.py` — efficiency ratios, CCS ramp schedule (0/2/5/8yr), economic retirement, year-aware masks all match Python |
| Constants | :white_check_mark: | Heat rates (CCGT 7.0, CT 10.5, Coal 10.0, Oil 10.5), emission factors, CO₂ rates match Python exactly |
| Economic retirement | :white_check_mark: | `margin < 0 → CF = 0` prevents negative-margin dispatch (mitigates Risk R4 client-side) |
| Validation | :white_check_mark: | Baseline 2030 P50 = 32.81 Mt matches server-computed 32.8062 Mt |
| **New fossil columns** | :x: | Does not read `nb_*`, `total_new_fossil_mw`, `gas_built_gw`, `fossil_built_gw` from sweep dispatch data |

**Impact**: Same gap as Python `fleet_dispatch.py` — grid-level new fossil builds are invisible to the client-side engine. When Prompt 2 is implemented for Python, the sweep dispatch data JSON extraction must also be updated to include new fossil columns so the JS engine can consume them.

---

#### `frontend/js/fleet-sidebar.js` — :warning: Partially Compatible (NEW)

| Aspect | Status | Detail |
|--------|--------|--------|
| Plant modifications | :white_check_mark: | Operating → Retired → CCS Retrofit status changes, capacity editing, year_online inputs |
| Add plant | :white_check_mark: | New gas/coal plants with fuel type, capacity, heat rate, year online |
| Client-side dispatch | :white_check_mark: | Calls `FleetDispatchEngine.computeFleetDispatch()` and injects results via `FLEET_SCENARIOS_API.addScenario()` — all 4 charts update instantly |
| localStorage | :white_check_mark: | Save/load/delete custom fleet scenarios |
| **Grid-level new fossil awareness** | :x: | User-added plants via sidebar are independent of grid-level `apply_economic_new_build()` results — same double-counting risk as `constellation_scenarios.json` (see Prompt 3) |

**Impact**: The sidebar enables interactive fleet-level plant additions while the sweep already includes grid-level new fossil builds. This makes the reconciliation rule (Prompt 3) more urgent — users can now trigger the overlap interactively without any warning.

---

#### `backend/main.py` — :white_check_mark: Already Compatible

| Aspect | Status | Detail |
|--------|--------|--------|
| `SWEEP_CACHE_DIR` | :white_check_mark: | Line 1348: Points to `results/sweep_1215/` (correct) |
| API endpoints | :white_check_mark: | `/api/sweep-cached/status` checks for `sweep_1215_*` files |
| Parquet reader | :white_check_mark: | Generic — reads whatever columns exist in the parquet |
| **Sweep dimensions** | :warning: | Line 1366: `sweep_dimensions` dict may not include `new_fossil_cost_level` — needs verification |

**Fix**: Verify `sweep_dimensions` metadata includes the new fossil cost dimension.

---

#### `results/sweep_405/` — :x: Doubly Stale

| Aspect | Status | Detail |
|--------|--------|--------|
| ORDC model | :x: | Generated with buggy sigmoid ORDC — pre-fix (commit `463ada5`) |
| New-build fossil | :x: | Generated without `new_fossil_cost_level` dimension — only 405 scenarios |
| Schema | :x: | Missing `nb_*`, `total_new_fossil_mw`, `gas_built_gw`, `fossil_built_gw` columns |

**Impact**: Any analysis consuming `sweep_405/` data produces invalid results. This includes:
- `fleet_results.json` — derived from stale sweep
- `fleet_scenario_results.json` — derived from stale sweep
- `fleet_validation.json` — derived from stale sweep
- API endpoints (`/api/sweep-results`, `/api/sweep-aggregates`) — will 404 until sweep_1215/ exists

**Fix**: Run fresh 1,215 sweep (Prompt 5), archive `sweep_405/` for reference, regenerate all downstream artifacts.

---

## 4. Implementation Prompts

Ordered, self-contained prompts for integrating new-build fossil across the codebase. Execute in sequence — each prompt builds on the prior one.

### Prompt 1: Code Cleanup — Update "405" References to "1,215"

**Goal**: Eliminate misleading "405" references in docstrings and comments across the codebase.

**Files to modify**:
- `scripts/run_sweep_405.py`
- `scripts/fleet_dispatch.py`

**Changes for `run_sweep_405.py`**:
1. Line 2-16 (module docstring): Update scenario count description. The docstring already correctly states the 6 dimensions at line 4-5, but the usage examples and output paths reference `sweep_405`:
   - Line 10: `python run_sweep_405.py [--isos CAISO PJM] [--output-dir ../results/sweep_405]` → update output dir to `../results/sweep_1215`
   - Lines 13-15: Output file names `sweep_405/*` → `sweep_1215/*`
2. Search for any other "405" string literals in the file (variable names, path defaults, comments)
3. Do NOT rename the file itself (that's a separate decision with import/CI implications)

**Changes for `fleet_dispatch.py`**:
1. Line 5: "Maps grid-level **405-scenario** sweep results" → "Maps grid-level **1,215-scenario** sweep results"
2. Line 9: "no Python for-loops over the **405** scenarios" → "no Python for-loops over the **1,215** scenarios"
3. Lines 14, 20: CLI usage examples — update `sweep_405` paths to `sweep_1215`
4. Line 136: `load_sweep()` docstring — "Load the **405-sweep** flat parquet" → "Load the **1,215-sweep** flat parquet"

**Verification**: `grep -rn "405" scripts/run_sweep_405.py scripts/fleet_dispatch.py` should return only the filename itself (unavoidable) and no stale scenario-count references.

---

### Prompt 2: Fleet Dispatch — Ingest New Fossil Build Data

> **Status (2026-03-19): Not addressed.** Python `fleet_dispatch.py` was not modified. The JS port (`fleet-dispatch-engine.js`) also does not ingest new fossil columns — it reads `{fuel}_cf` and `{fuel}_margin` from pre-extracted sweep dispatch JSON, not raw parquet. This prompt now applies to **both** the Python script AND the JSON extraction pipeline (`export_sweep_dispatch_data.py` / `/api/sweep-dispatch-data`) that feeds the JS engine.

**Goal**: Update `fleet_dispatch.py` to read and use new-build fossil columns from the sweep parquet, so fleet emissions correctly account for grid-level new fossil builds.

**Background**: When `apply_economic_new_build()` adds new gas capacity to the grid, it increases total supply and depresses LMPs and incumbent margins. Currently, `fleet_dispatch.py` only reads `ge_{fuel}_cf` and `ge_{fuel}_margin_mwh` — these already reflect the post-new-build equilibrium. However, the fleet module doesn't know **why** margins changed or **how much** new fossil exists, which limits its ability to:
- Report new-build fossil generation alongside fleet generation
- Adjust plant-level dispatch for competitive pressure from new entrants
- Provide new-fossil context in fleet scenario output

**Files to modify**: `scripts/fleet_dispatch.py`

**Implementation**:

1. **Expand `load_sweep()` to validate new fossil columns exist** (lines 135-142):
   ```python
   def load_sweep(sweep_path: str) -> pd.DataFrame:
       """Load the 1,215-sweep flat parquet."""
       df = pd.read_parquet(sweep_path)
       required = {"iso", "year", "scenario"}
       missing = required - set(df.columns)
       if missing:
           raise ValueError(f"Sweep parquet missing columns: {missing}")
       # Warn if new fossil columns are absent (pre-expansion parquet)
       new_fossil_cols = {"total_new_fossil_mw", "gas_built_gw", "fossil_built_gw"}
       if not new_fossil_cols.issubset(df.columns):
           import warnings
           warnings.warn(
               f"Sweep parquet missing new-fossil columns: {new_fossil_cols - set(df.columns)}. "
               "Fleet dispatch will not account for grid-level new fossil builds."
           )
       return df
   ```

2. **Add new fossil data to fleet dispatch output** — in the per-scenario results dict, include:
   - `grid_new_fossil_mw`: Total new fossil MW built on the grid (from `total_new_fossil_mw`)
   - `grid_gas_built_gw`: Cumulative gas builds (from `gas_built_gw`)
   - `grid_fossil_built_gw`: Cumulative all-fossil builds (from `fossil_built_gw`)
   - Per-fuel new builds: `nb_{fuel}_mw` columns

3. **Adjust incumbent fleet plant CF for competitive pressure** (optional, defensible either way):
   - When `total_new_fossil_mw > 0`, apply a capacity-weighted CF depression to incumbent gas plants
   - Rationale: New gas competes for the same dispatch hours, pushing existing plants down the merit order
   - Conservative approach: Flag the data in output, let downstream analysis apply the adjustment
   - Aggressive approach: Reduce incumbent gas CF by `new_gas_mw / (existing_gas_mw + new_gas_mw)` fraction
   - **Recommendation**: Start conservative (flag only), add adjustment as a follow-up if analysis shows material impact

4. **Update output schema** — add new fossil fields to `fleet_results.json` and `fleet_scenario_results.json`:
   ```json
   {
     "scenario": "baseline",
     "year": 2030,
     "iso": "PJM",
     "fleet_emissions_mt": 12.5,
     "grid_new_fossil_mw": 2400,
     "grid_gas_built_gw": 2.1,
     "grid_fossil_built_gw": 2.4,
     "nb_gas_ccgt_mw": 1800,
     "nb_gas_ct_mw": 600,
     "nb_coal_mw": 0
   }
   ```

**Verification**:
- Load a sweep parquet with new fossil columns → confirm no warnings
- Load a pre-expansion parquet (sweep_405) → confirm warning emitted, no crash
- Check output JSON includes `grid_new_fossil_mw` and `nb_*` fields

---

### Prompt 3: Fleet Scenario Reconciliation — Define Interaction Rules

**Goal**: Reconcile the two independent "new fossil" mechanisms so they don't double-count or conflict.

**The two mechanisms**:
1. **Grid-level** (`apply_economic_new_build()` in `market_simulation.py`): Market-driven. Triggered by reserve/economic conditions. Varies by scenario. Produces `nb_{fuel}_mw` in sweep parquet.
2. **Fleet-level** (`add_plant` in `constellation_scenarios.json`): User-defined. Static. Applied regardless of market conditions. Currently used in "CCS + New Gas" scenario.

**Recommended interaction rule**: **Fleet `add_plant` is additive to grid new-build** (Option A from the original plan).

**Rationale**:
- Grid-level new-build represents **market-driven** capacity that any developer might build (generic)
- Fleet-level `add_plant` represents **company-specific** capacity decisions (e.g., Constellation decides to build a specific plant at a specific site)
- These are conceptually distinct — a company's decision to build doesn't prevent the market from also building
- The alternative (fleet replaces grid) would mean company decisions negate market forces — less realistic
- If combined capacity seems unrealistically high, that's a signal the scenario assumptions need tuning, not that the interaction model is wrong

**Implementation**:

1. **Document the interaction rule** in `constellation_scenarios.json`:
   Add a top-level field:
   ```json
   "new_build_interaction": {
     "rule": "additive",
     "description": "Fleet-level add_plant actions are additive to grid-level new fossil builds from apply_economic_new_build(). Fleet additions represent company-specific decisions; grid builds represent market-driven capacity. Both can coexist."
   }
   ```

2. **Update `fleet_dispatch.py`** to report both sources separately:
   - `grid_new_fossil_mw` — from sweep parquet (market-driven)
   - `fleet_new_fossil_mw` — from `add_plant` actions (company-specific)
   - `total_new_fossil_mw` — sum of both (for total impact analysis)

3. **Add a note in the fleet dispatch output** when both mechanisms produce new gas in the same ISO/year:
   ```json
   "new_fossil_overlap_note": "Both grid-level (2,400 MW) and fleet-level (800 MW) new gas in PJM 2035. Combined 3,200 MW is additive per interaction rule."
   ```

**Verification**:
- Run fleet dispatch on a scenario with both grid and fleet new gas → confirm both appear in output
- Confirm `total_new_fossil_mw = grid_new_fossil_mw + fleet_new_fossil_mw`

---

### Prompt 4: ORDC Calibration Check

**Goal**: After running the 1,215 sweep, validate that ORDC scarcity pricing remains calibrated with new fossil builds active.

**Prerequisites**: Prompt 5 (fresh sweep) must complete first. This prompt is ordered before Prompt 5 for logical grouping but executes after.

**Analysis steps**:

1. **Extract scarcity hour distribution**:
   ```python
   import pandas as pd
   df = pd.read_parquet("results/sweep_1215/sweep_1215_flat.parquet")

   # Scarcity hours by ISO and year
   scarcity = df.groupby(["iso", "year"])["ordc_scarcity_hours"].describe()

   # Scarcity hours by new fossil build level
   # (new_fossil_cost_level is embedded in scenario string — parse it)
   df["has_new_fossil"] = df["total_new_fossil_mw"] > 0
   compare = df.groupby(["iso", "has_new_fossil"])["ordc_scarcity_hours"].describe()
   ```

2. **Expected behavior**:
   - Scenarios with new fossil builds should have **fewer** scarcity hours (new supply relieves reserves)
   - The reduction should be **moderate, not total** — some scarcity should remain even with new builds
   - ERCOT should show the largest ORDC impact (energy-only market, no capacity market to cushion)

3. **Red flags to watch for**:
   - **Zero scarcity hours in >50% of scenarios** → ORDC caps may be too low or knee_mw too aggressive
   - **No difference with/without new fossil** → new builds may not be large enough to matter, or ORDC parameters are too loose
   - **Scarcity hours >2,000 in any scenario** → something is broken (8,760 hours/year, >23% scarcity is extreme)

4. **Calibration targets** (approximate, based on real-world data):
   | ISO | Expected avg scarcity hours | Acceptable range |
   |-----|---------------------------|-----------------|
   | ERCOT | 100–400 | 50–800 |
   | CAISO | 50–200 | 20–500 |
   | PJM | 30–150 | 10–400 |
   | NYISO | 40–200 | 15–500 |
   | NEISO | 30–150 | 10–400 |
   | MISO | 50–250 | 20–600 |
   | SPP | 40–200 | 15–500 |

5. **If recalibration needed**: Adjust `ORDC_PARAMS` in `market_simulation.py` — specifically `knee_mw` (reserve level where scarcity pricing kicks in) and `cap` (maximum scarcity adder in $/MWh). Document changes and rationale.

**Output**: Summary table of scarcity hour statistics by ISO × new-fossil-present. Recommendation on whether ORDC parameters need adjustment.

---

### Prompt 5: Run Fresh 1,215-Scenario Sweep

**Goal**: Generate fresh sweep results with both the ORDC fix AND new-build fossil active.

**Prerequisites**: Prompts 1-2 should be complete (code cleanup + fleet dispatch updates). Prompt 3 (reconciliation rules) should be decided but doesn't block the sweep itself.

**Steps**:

1. **Archive stale results**:
   ```bash
   mv results/sweep_405/ results/sweep_405_archived_$(date +%Y%m%d)/
   ```

2. **Run the sweep**:
   ```bash
   cd market-simulator/scripts
   python run_sweep_405.py --output-dir ../results/sweep_1215
   ```
   Expected runtime: depends on hardware. ~51,030 simulations.

3. **Validate output**:
   - Check row count: `1,215 × 7 × 6 = 51,030` rows in parquet
   - Check new fossil columns exist: `nb_gas_ccgt_mw`, `nb_gas_ct_mw`, `total_new_fossil_mw`, `gas_built_gw`, `fossil_built_gw`
   - Check LMP sanity: avg LMP should be $25–50/MWh, no $0 or $9999 outliers
   - Check new fossil builds: `total_new_fossil_mw > 0` in at least some scenarios (confirms the feature is active)
   - Check aggregates JSON generated

4. **Commit results immediately**:
   ```bash
   git add results/sweep_1215/
   git commit -m "Bank 1,215-scenario sweep results (ORDC fix + new-build fossil)"
   git push
   ```

**Output**: `results/sweep_1215/` containing:
- `market_simulation_results.json` — full nested JSON
- `sweep_1215_flat.parquet` — flat table for analysis
- `sweep_1215_aggregates.json` — P10/P50/P90 percentile bands

---

### Prompt 6: Re-run Fleet Dispatch Against Fresh Sweep

> **Status (2026-03-19): Not addressed** (blocked on Prompts 2 + 5). After this re-run completes, the sweep dispatch data JSON (`sweep_dispatch_data.json`) served to the browser must also be regenerated via `export_sweep_dispatch_data.py` to include new fossil columns, so the client-side fleet sidebar benefits from the updated data.

**Goal**: Regenerate fleet results using the updated `fleet_dispatch.py` (from Prompt 2) against the fresh 1,215-scenario sweep (from Prompt 5).

**Prerequisites**: Prompts 2 and 5 must be complete.

**Steps**:

1. **Single-fleet mode** (Phase 2.1):
   ```bash
   cd market-simulator/scripts
   python fleet_dispatch.py \
       --fleet ../fleet_scenarios/sample_fleet.json \
       --sweep ../results/sweep_1215/sweep_1215_flat.parquet \
       --output ../results/fleet_results.json
   ```

2. **Multi-scenario mode** (Phase 3.1):
   ```bash
   python fleet_dispatch.py \
       --scenarios ../fleet_scenarios/constellation_scenarios.json \
       --sweep ../results/sweep_1215/sweep_1215_flat.parquet \
       --output ../results/fleet_scenario_results.json
   ```

3. **Validate output**:
   - `fleet_results.json`: Contains emissions for all 1,215 × 7 × 6 combinations
   - `fleet_scenario_results.json`: Contains emissions per named scenario × grid scenario × ISO × year
   - New fossil fields present: `grid_new_fossil_mw`, `nb_gas_ccgt_mw`, etc.
   - Emissions values are physically reasonable (positive, non-zero for fossil plants, zero for retired plants)

4. **Commit**:
   ```bash
   git add results/fleet_results.json results/fleet_scenario_results.json
   git commit -m "Regenerate fleet results against 1,215-scenario sweep with new fossil data"
   git push
   ```

---

### Prompt 7: Fleet Scenarios Viz Design Update

> **Status (2026-03-19): Complete.** `Fleet_Scenarios_Viz_Design.md` was fully implemented as the live `fleet-scenarios.html` page with fleet config sidebar (commit `66d5ca7`) and then deleted (commit `eff7c4c`). The grid-level `new_fossil_cost_level` L/M/H toggle has been added to the controls strip, filtering the fan chart envelope data by sweep dimension. Backend (`fleet_dispatch.py`) computes per-level envelopes from scenario IDs.
>
> **Implementation**: Toggle added to controls strip (All/Low/Med/High). Selecting a level filters the P10/P50/P90 fan chart bands to show only scenarios at that fossil cost level. `fleet_dispatch.py` parses `new_fossil_cost_level` from scenario IDs (last token: L/M/H) and groups envelope computation by level. Sample JSON updated with per-level envelope data.

**Original goal** *(superseded)*: Update `Fleet_Scenarios_Viz_Design.md` to reflect the 1,215-scenario sweep and new fossil dimension.

**Revised goal**: Add `new_fossil_cost_level` L/M/H toggle to the fleet-scenarios page controls strip, filtering the sweep data served to charts. The sidebar and all 4 charts are already implemented.

---

### Prompt 8: Constellation Scenarios Update

> **Status (2026-03-19): Not addressed.** JSON not modified. Now more urgent — the fleet config sidebar enables interactive plant additions that compound the double-counting risk without any reconciliation rule or user warning.

**Goal**: Update `constellation_scenarios.json` to account for the new fossil dimension and document the interaction rule.

**File to modify**: `fleet_scenarios/constellation_scenarios.json`

**Changes**:

1. **Add `new_build_interaction` field** (top-level, after `metadata`):
   ```json
   "new_build_interaction": {
     "rule": "additive",
     "description": "Fleet-level add_plant actions are additive to grid-level new fossil builds from apply_economic_new_build(). Fleet additions represent company-specific decisions; grid builds represent market-driven capacity.",
     "reporting": "Both sources reported separately in output: grid_new_fossil_mw (market) + fleet_new_fossil_mw (company) = total_new_fossil_mw"
   }
   ```

2. **Update "CCS + New Gas" scenario description** to clarify interaction:
   - Current: Scenario has static `add_plant` gas entries
   - Updated: Add a `notes` field explaining that these fleet-level additions are **on top of** any grid-level new fossil builds in the sweep
   - Example:
     ```json
     {
       "description": "CCS retrofits on existing coal + new gas CCGT additions. Fleet-level gas additions are additive to grid-level new fossil builds (see new_build_interaction rule).",
       "modifications": [...]
     }
     ```

3. **Update `$schema_version`** to `"2.1"` to reflect the schema extension.

4. **Update `$schema_description`** to mention new-build interaction rules.

**Verification**:
- JSON validates (no syntax errors)
- `new_build_interaction` field is parseable by `fleet_dispatch.py`
- "CCS + New Gas" scenario description is clear about additive behavior

---

### Prompt 9: Backend API Alignment

> **Status (2026-03-19): Partially addressed.** The `/api/fleet-scenarios-config` and `/api/sweep-dispatch-data` endpoints were created to serve the fleet sidebar. The sweep dispatch data JSON (`sweep_dispatch_data.json`) is generated by `export_sweep_dispatch_data.py`. However, new fossil columns (`nb_*`, `total_new_fossil_mw`) are not included in the dispatch data JSON — the JS engine cannot see grid-level new fossil builds.

**Goal**: Ensure backend API endpoints serve data from the correct `results/sweep_1215/` directory and expose new fossil columns.

**File to verify**: `backend/main.py`

**Checks**:

1. **`SWEEP_CACHE_DIR`** (line 1348): Already points to `results/sweep_1215/` — **no change needed**.

2. **`sweep_dimensions` metadata** (around line 1366): Verify it includes `new_fossil_cost_level`:
   ```python
   "sweep_dimensions": {
       "demand_growth": list(DEMAND_GROWTH_LEVELS),
       "price_sensitivity": list(PRICE_SENSITIVITIES.keys()),
       "ppa_level": list(PPA_LEVELS),
       "gas_friction": list(GAS_FRICTION_LEVELS.keys()),
       "queue_pace": list(QUEUE_PACE_LEVELS.keys()),
       "new_fossil_cost": ["Low", "Medium", "High"],  # <-- verify this exists
   }
   ```
   If missing, add it.

3. **Parquet schema validation** — when serving `/api/sweep-results`, verify the response includes new fossil columns:
   - `total_new_fossil_mw`
   - `gas_built_gw`
   - `fossil_built_gw`
   - `nb_{fuel}_mw` columns

4. **Aggregation endpoints** — `/api/sweep-aggregates` should compute P10/P50/P90 for new fossil columns too. Verify the aggregation logic doesn't hardcode column lists that exclude new fossil fields.

5. **Fleet results endpoint** — if `/api/fleet-results` exists, verify it reads from the updated `fleet_results.json` / `fleet_scenario_results.json` that include new fossil data.

**Verification**:
- Start backend: `cd market-simulator/backend && python main.py`
- `curl localhost:8000/api/sweep-cached/status` → should show `available: true` after sweep runs
- `curl localhost:8000/api/sweep-results?iso=PJM&year=2030` → response should include `total_new_fossil_mw`

---

### Prompt 10: End-to-End Integration Test

**Goal**: Full pipeline validation from sweep through to dashboard API.

**Test matrix**:

| Check | Expected | Pass criteria |
|-------|----------|---------------|
| Parquet row count | 51,030 | `len(df) == 1215 * 7 * 6` |
| New fossil columns exist | All present | `{'nb_gas_ccgt_mw', 'nb_gas_ct_mw', 'total_new_fossil_mw', 'gas_built_gw', 'fossil_built_gw'} ⊆ df.columns` |
| New fossil builds non-zero | At least some scenarios build fossil | `(df['total_new_fossil_mw'] > 0).any()` |
| LMP range | $5–$300/MWh | `5 ≤ df['avg_lmp'].min()` and `df['avg_lmp'].max() ≤ 300` |
| No $0 LMP | No zero LMPs | `(df['avg_lmp'] == 0).sum() == 0` |
| No $9999 LMP | No absurd outliers | `df['avg_lmp'].max() < 1000` |
| Scarcity hours reasonable | 0–2000 per scenario | `df['ordc_scarcity_hours'].max() < 2000` |
| Fleet emissions populated | All scenarios have fleet data | `len(fleet_results) == expected` |
| Fleet new fossil fields | Present in output | `'grid_new_fossil_mw' in fleet_results[0]` |
| API responds | 200 OK | All endpoints return valid JSON |
| Aggregates generated | P10/P50/P90 bands | Aggregates JSON has expected structure |

**Execution**:

```python
import pandas as pd
import json

# 1. Parquet validation
df = pd.read_parquet("results/sweep_1215/sweep_1215_flat.parquet")
assert len(df) == 51030, f"Expected 51,030 rows, got {len(df)}"

new_fossil_cols = {"nb_gas_ccgt_mw", "nb_gas_ct_mw", "total_new_fossil_mw",
                   "gas_built_gw", "fossil_built_gw"}
assert new_fossil_cols.issubset(df.columns), f"Missing: {new_fossil_cols - set(df.columns)}"

assert (df["total_new_fossil_mw"] > 0).any(), "No new fossil builds in any scenario"
assert df["avg_lmp"].between(1, 500).all(), f"LMP outliers: min={df['avg_lmp'].min()}, max={df['avg_lmp'].max()}"
assert df["ordc_scarcity_hours"].max() < 4000, f"Extreme scarcity: {df['ordc_scarcity_hours'].max()}"

# 2. Fleet results validation
with open("results/fleet_scenario_results.json") as f:
    fleet = json.load(f)
# Check structure and new fossil fields present

# 3. Aggregates validation
with open("results/sweep_1215/sweep_1215_aggregates.json") as f:
    agg = json.load(f)
# Check P10/P50/P90 bands exist for all ISOs

print("All integration tests passed.")
```

**If any test fails**: Stop, diagnose root cause, fix, and re-run from the appropriate prompt. Do not proceed to dashboard work with invalid data.

---

## 5. Testing Schema

### Pre-Sweep Validation (Before Prompt 5)

| Test | Command | Pass criteria |
|------|---------|---------------|
| Syntax check | `python -c "import py_compile; py_compile.compile('scripts/run_sweep_405.py')"` | No errors |
| Import check | `python -c "from market_simulation import apply_economic_new_build"` | Imports successfully |
| Fleet dispatch syntax | `python -c "import py_compile; py_compile.compile('scripts/fleet_dispatch.py')"` | No errors |
| Stale refs eliminated | `grep -rn '"405-scenario"' scripts/` | No matches (filename refs OK) |

### Post-Sweep Validation (After Prompt 5)

| Test | Command | Pass criteria |
|------|---------|---------------|
| Row count | `python -c "import pandas as pd; df=pd.read_parquet('results/sweep_1215/sweep_1215_flat.parquet'); print(len(df))"` | 51,030 |
| Schema check | See Prompt 10 assertions | All pass |
| LMP sanity | `df['avg_lmp'].describe()` | Mean $25–50, max < $500 |
| Scarcity check | `df['ordc_scarcity_hours'].describe()` | Median > 0, max < 2000 |
| New fossil active | `(df['total_new_fossil_mw'] > 0).mean()` | > 5% of scenarios |

### Post-Fleet Validation (After Prompt 6)

| Test | Command | Pass criteria |
|------|---------|---------------|
| Fleet results exist | `ls results/fleet_results.json results/fleet_scenario_results.json` | Both exist |
| JSON valid | `python -c "import json; json.load(open('results/fleet_results.json'))"` | No errors |
| New fossil fields | Check for `grid_new_fossil_mw` in output | Present |
| Emissions non-negative | All emissions values ≥ 0 | No negatives |

### API Validation (After Prompt 9)

| Test | Command | Pass criteria |
|------|---------|---------------|
| Status endpoint | `curl -s localhost:8000/api/sweep-cached/status \| jq .available` | `true` |
| Sweep results | `curl -s localhost:8000/api/sweep-results?iso=PJM&year=2030` | 200 OK, JSON valid |
| New fossil in API | Check response includes `total_new_fossil_mw` | Present |

---

## 6. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| R1 | Sweep runtime exceeds expectations | Medium | High (blocks all downstream) | Monitor progress, use `--isos` flag to test single ISO first before full sweep |
| R2 | New fossil builds in every scenario (over-triggering) | Low | Medium | Check `(df['total_new_fossil_mw'] > 0).mean()` — if > 80%, new fossil thresholds may be too permissive |
| R3 | New fossil builds in no scenario (under-triggering) | Low | Medium | Check `(df['total_new_fossil_mw'] > 0).any()` — if False, cost assumptions may be too restrictive or ORDC doesn't create enough scarcity |
| R4 | Fleet dispatch CF adjustment causes negative margins | Medium | Low | Start conservative (flag only, no adjustment) per Prompt 2 recommendation. **Update:** Client-side economic retirement (`margin < 0 → CF = 0`) in `fleet-dispatch-engine.js` mitigates this for browser-side dispatch. Python-side risk unchanged. |
| R5 | Constellation scenarios JSON schema break | Low | Medium | Validate JSON after edits, run fleet dispatch with `--dry-run` if available |
| R6 | ORDC recalibration needed post-new-fossil | Medium | High (re-run sweep) | Run Prompt 4 analysis first, make targeted ORDC_PARAMS adjustments if needed, then re-sweep only if changes are material |
| R7 | Stale sweep_405/ results accidentally consumed | Medium | High (invalid analysis) | Archive immediately (Prompt 5 step 1), grep codebase for `sweep_405` path references |
| R8 | Dashboard JS hardcodes 405 scenario count | Low | Low | Search frontend/ for "405" string literals after sweep completes. **Update:** `fleet-dispatch-engine.js` is scenario-count agnostic — reads whatever arrays exist in sweep dispatch data. No "405" references found in JS engine. |
| R9 | Git merge conflicts on fleet_dispatch.py | Low | Low | Work on feature branch, resolve conflicts before merging to main |
| R10 | Additive interaction rule produces unrealistic combined capacity | Medium | Medium | Monitor `total_new_fossil_mw` (grid + fleet) — if combined exceeds 50% of existing capacity in any ISO, flag for review |
| R11 | Client-side / server-side dispatch divergence | Medium | Medium | JS engine (`fleet-dispatch-engine.js`) and Python `fleet_dispatch.py` could drift apart as one is updated without the other. Keep `§4.6.1` of Model Methodology Spec as canonical reference; update both engines in lockstep. |

### Dependency Chain

```
Prompt 1 (Cleanup) ─────────────────────────────────────┐
Prompt 2 (Fleet Dispatch) ──────────────────────────────┤
Prompt 3 (Reconciliation) ─────────────────────────────┤
                                                        ▼
                                              Prompt 5 (Run Sweep)
                                                        │
                                           ┌────────────┼────────────┐
                                           ▼            ▼            ▼
                                    Prompt 4      Prompt 6      Prompt 9
                                   (ORDC Cal)   (Fleet Re-run)  (API Align)
                                                        │
                                                        ▼
                                                  Prompt 7 + 8
                                              (Viz + Scenarios)
                                                        │
                                                        ▼
                                                  Prompt 10
                                                (E2E Integration)
```

Prompts 1–3 can run in parallel. Prompt 5 depends on 1–2. Prompts 4, 6, 9 depend on 5 (can run in parallel). Prompt 7 is largely complete (sidebar implemented); remaining grid-level toggle can run anytime. Prompt 8 can run anytime. Prompt 10 is the final gate.

> **Note (2026-03-19):** Prompts 2 and 9 now have a JS-side component: when implementing new fossil column ingestion (Prompt 2), also update `export_sweep_dispatch_data.py` to include those columns in the browser-served JSON. When verifying API alignment (Prompt 9), also verify the `/api/sweep-dispatch-data` endpoint includes new fossil data.
