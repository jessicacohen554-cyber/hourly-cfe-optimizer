# New-Build Fossil Integration Audit & Implementation Prompts

> **Created**: 2026-03-19
> **Scope**: Audit of downstream impacts after `apply_economic_new_build()` was added to the market-simulator dispatch engine
> **Status**: Audit complete — implementation prompts ready for execution

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
- **`constellation_scenarios.json`** — Fleet-level `add_plant` actions don't interact with grid-level `apply_economic_new_build()`
- **`Fleet_Scenarios_Viz_Design.md`** — Assumes 405 scenarios; no new fossil dimension in controls
- **`results/sweep_405/`** — Results are pre-ORDC fix AND pre-new-fossil (doubly stale)
- **`backend/main.py`** — Already correctly points to `results/sweep_1215/` (no change needed)

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

**Fix needed**: Define interaction rules — see Prompt 3.

---

#### `docs/Fleet_Scenarios_Viz_Design.md` — :warning: Design Updates Needed

| Aspect | Status | Detail |
|--------|--------|--------|
| Page layout | :white_check_mark: | Unaffected |
| Chart types | :white_check_mark: | Emissions fan, waterfall, generation mix — still valid |
| **Controls strip** | :warning: | No `new_fossil_cost_level` toggle — missing the 6th dimension |
| **Scenario count** | :warning: | Implicitly assumes 405 scenarios (references `fleet_scenario_results.json` from pre-expansion) |
| **Secondary charts** | :warning: | No new-fossil build waterfall or new-fossil generation in mix charts |
| **Emission accounting** | :warning: | Doesn't account for new-build fossil generation in emissions by fuel chart |

**Fix needed**: Update design doc to include new fossil dimension in controls, charts, and emission accounting.

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
