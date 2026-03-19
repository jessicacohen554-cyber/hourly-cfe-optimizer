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
