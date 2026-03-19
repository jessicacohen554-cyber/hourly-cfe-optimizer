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
