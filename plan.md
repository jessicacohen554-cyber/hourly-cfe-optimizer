# Plan: Per-Threshold Resource Grid Caps for Step 1

## Problem

The Step 1 grid search uses **global** resource caps for all thresholds:
- Wind: 0–250% for every threshold
- Clean firm: 0–120% for every threshold
- Solar: 0–100% for every threshold

But the EF analysis shows these caps are only needed at the highest thresholds. At t50, wind maxes at 55% across all ISOs — the grid wastes 80% of compute exploring wind 60–250% where no EF-worthy solution exists.

**Impact estimate**: ERCOT alone has 70% duplicate/wasted mixes. The compute savings from per-threshold caps would be multiplicative — fewer combos generated × fewer scored × fewer stored.

## EF-Derived Resource Ranges (Observed Max Across All ISOs)

| Threshold | Clean Firm Max | Wind Max | Solar Max |
|-----------|---------------|----------|-----------|
| 10–40     | 55            | 50       | 70        |
| 50        | 60            | 55       | 94        |
| 55        | 65            | 60       | 99        |
| 60        | 70            | 67       | 99        |
| 65        | 75            | 75       | 100       |
| 70        | 82            | 80       | 100       |
| 75        | 89            | 100      | 100       |
| 80        | 95            | 120      | 100       |
| 85        | 96            | 133      | 100       |
| 87.5      | 102           | 150      | 100       |
| 90        | 107           | 174      | 100       |
| 92.5      | 117           | 210      | 100       |
| 95+       | 120           | 250      | 100       |

## Proposed Per-Threshold Caps (With ~25% Buffer)

Adding 25% relative buffer (minimum +10 absolute) to ensure no edge-case EF solutions are missed. Caps are rounded up to nearest 5 for clean grid steps.

| Threshold | Clean Firm Cap | Wind Cap | Solar Cap | Notes |
|-----------|---------------|----------|-----------|-------|
| 10–40     | 70            | 65       | 90        | Tight — very few EF mixes here |
| 50        | 75            | 70       | 100       | |
| 55        | 80            | 75       | 100       | Solar hits cap |
| 60        | 90            | 85       | 100       | |
| 65        | 95            | 95       | 100       | |
| 70        | 100           | 100      | 100       | |
| 75        | 110           | 125      | 100       | Wind starts climbing |
| 80        | 120           | 150      | 100       | Clean firm at max |
| 85        | 120           | 170      | 100       | |
| 87.5      | 120           | 190      | 100       | |
| 90        | 120           | 220      | 100       | |
| 92.5      | 120           | 250      | 100       | Full range needed |
| 95–99.99  | 120           | 250      | 100       | Full range |

**Hydro/Geothermal**: Already per-ISO capped. No change needed.
**Storage**: Already threshold-dependent (H2 ≥95% only, storage ≥65% only in 1d). No change.

## Estimated Compute Savings

The grid is a Cartesian product. Reducing wind from 250→70 at t50 (5% step) cuts wind dimension from 51 to 15 points — a **3.4× reduction** in combo count for that threshold. Combined with clean_firm reduction (25→16 points), that's ~**5× fewer combos** at t50.

Thresholds ≤90 see the biggest savings. High thresholds (92.5+) are unchanged.

Since low thresholds run fastest anyway (fewer combos pass viability), the **wall-clock** savings are moderate but the **total mix count** and **storage size** savings are substantial.

## Implementation Approach

### Option A: Lookup Table in `step1_pfs_generator.py` (Recommended)
- Add a `THRESHOLD_CAPS` dict mapping each threshold → `{clean_firm, solar, wind}` caps
- Modify `generate_resource_combos()` to accept per-threshold caps instead of global `RESOURCE_CAPS`
- The adaptive grid refinement already works per-threshold — it just needs tighter bounds
- Minimal code change (~30 lines), fully backward-compatible
- **Risk**: If the EF changes materially in a future run (unlikely for well-established thresholds), caps may need updating. The 25% buffer mitigates this.

### Option B: Data-Driven Caps from Previous EF
- Before Step 1 runs, read existing Step 2 EF parquets to derive caps dynamically
- Pro: Self-updating, no hardcoded table
- Con: Circular dependency (Step 1 depends on Step 2 output from a prior run). First run needs fallback to global caps. More complex code.

### Option C: Prune After Generation (Already Done)
- The `prune_duplicate_mixes.py` script already handles post-hoc deduplication
- This saves storage but NOT compute — Step 1 still generates and scores all the wasted combos
- Complementary to A or B, not a replacement

### Recommendation: Option A
Simple, predictable, no circular dependency. The caps are derived from a mature dataset (7 ISOs × 21 thresholds × millions of EF mixes) and are unlikely to shift materially. The 25% buffer provides safety margin.

## Files to Modify

1. **`scripts/step1_pfs_generator.py`** — Add `THRESHOLD_CAPS` lookup, modify `generate_resource_combos()` and `run_phase1a_coarse()` / `run_phase1b_refine()` to use per-threshold caps
2. **`scripts/step1a_generate_mixes.py`** — If the modular path also hardcodes global caps, update there too
3. **SPEC.md** — Document the per-threshold cap decision

## What This Does NOT Change

- Storage dispatch logic (already per-threshold)
- Green H2 activation (already ≥95% only)
- Hydro/geothermal caps (already per-ISO)
- Step 2/3/4+ downstream — they consume whatever Step 1 produces
- The prune workflow — still useful as a safety net
