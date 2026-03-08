# Procurement Deployment Matrix Overhaul — Implementation Plan

## Summary
Major overhaul of the procurement deployment visualization to show **grid-level impact** (what happens to each ISO's grid) instead of buyer-level procurement. 9 strategies × 7 ISOs matrix with dispatch-integrated physics, shared queue depletion, and precise resource-type balls.

## Phase 1: Backend — Shared Queue & Strategy Updates (Python)

### 1A. Fix Strategy 1A/1B shared queue depletion (`step8a_strategy_consequential.py`)
**Problem**: Currently each buyer ISO walks the deployment queue independently — the same MISO wind tranche is consumed by every buyer, double-counting capacity.
**Fix**: Create a `compute_all_isos_shared_queue()` function that:
1. Builds the deployment queue once (sorted by $/tCO₂ for 1A, fossil-only for 1B)
2. For each participation × threshold combo, allocates tranches across ALL 7 buyer ISOs simultaneously
3. As one buyer ISO consumes a tranche, it's depleted from the pool for all other ISOs
4. The allocation order within a tranche follows proportional buyer demand (largest buyer gets first draw, or pro-rata)
5. Output format unchanged but cross-ISO flows now reflect shared depletion

### 1B. Fix Strategy 3B prioritization (`step8c_strategy_annual.py`)
**Problem**: 3B may not be producing enough cross-regional ERCOT flow.
**Fix**: Verify and enforce that 3B sorts by cheapest $/MWh (not $/tCO₂). Ensure ERCOT wind/solar appears prominently in cross-regional purchases by other ISOs. Check the cross-regional pool construction and sort order.

### 1C. Add Strategy 3C and 3D to extraction pipeline
- 3C (same-ISO, no additionality) — already coded in `step8c_strategy_annual.py`
- 3D (cross-regional, no additionality) — already coded
- Add both to `STRATEGY_SOURCES` in `step9c_extract_deployment_data.py`
- Add extractors for 3C (same format as 2C/3A existing+new) and 3D (same format as 3B cross-regional)

## Phase 2: Backend — Dispatch Integration (New Pipeline Step)

### 2A. Create `step9d_dispatch_deployment.py`
New script that computes dispatch-level grid impact for every strategy × ISO × participation × threshold:

1. **For each combination**, take the procurement output (resource mix by ISO)
2. **Compute the resulting grid mix**: baseline grid + procurement additions
3. **Run through dispatch**: Use `dispatch_utils.reconstruct_hourly_dispatch()` or simplified version to get 8,760-hour profile
4. **Compute CO₂**: Use `dispatch_utils.compute_co2_from_dispatch()` for actual fossil displacement
5. **Compute gas backup**: Using step3a approach — peak demand minus firm clean capacity = required gas backup MW
6. **Compute curtailment**: Total generation minus total demand = curtailed energy
7. **Compute hourly match score**: Hours where clean ≥ demand / 8760

**Output fields added to deployment-data.js records:**
- `hms`: hourly match score (%)
- `curt`: curtailment TWh
- `curtPct`: curtailment as % of clean generation
- `gasBackupGw`: gas backup capacity needed (GW)
- `gridClean`: total grid clean % after procurement
- `baseline`: baseline grid state (at 0% participation) for each ISO

### 2B. Add grid baseline data to deployment-data.js
New top-level field `gridBaseline` with each ISO's existing clean mix:
```json
{
  "gridBaseline": {
    "CAISO": {"solar": 50.0, "wind": 19.7, "hydro": 21.3, "clean_firm": 17.7},
    ...
  }
}
```
This enables the 0% participation state where all jars show identical grid mix per ISO.

## Phase 3: Frontend — Jar Animation Overhaul (`jar-animation.js`)

### 3A. Add 3C and 3D strategies
- Update `STRATEGIES` array: `['1A', '1B', '2A', '2B', '2C', '3A', '3B', '3C', '3D']`
- Add labels: `'3C': 'Annual\n(Same, No Add.)'`, `'3D': 'Annual\n(Cross, No Add.)'`
- Matrix becomes 9 rows × 7 columns

### 3B. Grid-level paradigm shift
- **At 0% participation**: All jars show baseline grid mix per ISO (from `gridBaseline` data)
  - Use solid fill for existing grid clean resources
  - Ball count = grid clean % of total load (e.g., CAISO ~48.5% = ~49 balls)
- **As participation increases**: Jars show grid transformation:
  - Existing grid clean stays (solid fill)
  - Existing clean claimed by buyers (transparent fill + saturated outline) — for 2C, 3A, 3C
  - New clean added (transparent fill only)
  - Cross-ISO resources flowing OUT of this ISO shown with glow

### 3C. Glow semantics flip (point 5)
- **Current**: Glow on balls in buyer's jar showing source ISO
- **New**: Glow on balls in SOURCE ISO's jar showing that this resource serves external buyers
- When MISO wind serves PJM buyers in Strategy 1A, the MISO jar's wind balls glow with PJM's color
- Multiple buyer ISOs → cycle colors or use a blended glow

### 3D. Curtailment overflow (point 4.1)
- Balls that represent curtailed energy render ABOVE the jar rim
- Use a distinct visual: same resource color but with a striped/dashed pattern or reduced opacity
- Modify `_packRows()` to allow placing balls above `rimY`
- Modify `draw()` to NOT clip balls above the rim (remove or expand clip region)

### 3E. Ball rendering tiers (refined from point 4)
1. **Grid baseline (at 0% participation)**: Solid fill — existing grid clean that's just "there"
2. **Existing claimed (non-SSS, strategies 2C/3A/3C)**: Transparent fill + saturated outline
3. **New build**: Transparent fill only (current Tier 3 behavior)
4. **Curtailed**: Above rim, with reduced opacity or striped pattern

### 3F. Precise resource types (point 2)
- Replace aggregated "firm clean" / "VRE" with specific types: solar, wind, nuclear, CCS, hydro, battery, LDES, geothermal, offshore wind
- The Ball class already supports this via `getResourceColor()` — just need the data to provide precise types
- Update `getResourceLabel()` with any missing types

## Phase 4: Frontend — HTML Updates (`procurement_deployment.html`)

### 4A. Delete narrative sections
Remove everything from "Deployment Divergence" (Section 1, line 443) through Section 6 "Which Strategy Builds Clean Grids?" (line 561). Keep:
- Header
- Controls bar
- Legend
- Jar matrix
- Tooltip
- Aggregate charts (procurement + CO₂)
- Footer

### 4B. Update legend
- Remove "Free credits excluded" note (we're now showing grid baseline)
- Add "Grid baseline (solid)" tier to legend
- Update glow description: "Glow = resource serving external ISO buyers"
- Add curtailment indicator to legend

### 4C. Update strategy lists and colors
- Add 3C and 3D to `STRAT_COLORS`, `STRAT_LABELS`, `STRATEGIES_LIST`
- Update aggregate charts to show 9 strategies

### 4D. Update slider defaults
- Consider adding 0% to participation levels for baseline state
- Or handle 0% as a special case in the grid (show baseline when no data available)

## Phase 5: Re-run Pipeline & Generate Data

### 5A. Re-run step 8a with shared queue
```bash
python scripts/step8a_strategy_consequential.py
```

### 5B. Re-run step 8c (if 3B prioritization changed)
```bash
python scripts/step8c_strategy_annual.py
```

### 5C. Run dispatch integration
```bash
python scripts/step9d_dispatch_deployment.py
```

### 5D. Re-run step 9c with new strategies + dispatch data
```bash
python scripts/step9c_extract_deployment_data.py
```

## Execution Order
1. Phase 1A (shared queue) — can start immediately
2. Phase 1B (3B fix) — parallel with 1A
3. Phase 1C (add 3C/3D extraction) — parallel with 1A/1B
4. Phase 5A+5B (re-run strategies) — after 1A+1B complete
5. Phase 2A+2B (dispatch integration) — after 5A+5B complete
6. Phase 5C+5D (run dispatch + extract) — after 2A complete
7. Phase 3 (jar animation) — can start in parallel with Phase 2
8. Phase 4 (HTML) — can start in parallel with Phase 2
9. Final QA — after all phases complete

## Key Risk: Compute Time
- Step 8a/8c re-runs: seconds to minutes (these are cheap)
- Step 9d dispatch integration: ~4000 combos × 8760 hours — could be minutes to an hour depending on approach
  - Optimization: Use simplified dispatch model (not full Step 4 cache) for grid-level metrics
  - Fallback: Pre-compute for key participation × threshold combos, interpolate
