# Fix Plan: Marginal MAC Monotonicity

## Problem Statement

Marginal MAC (stepwise cost per incremental ton CO2 abated between consecutive thresholds) is wildly non-monotonic. Current `MARGINAL_MAC_DATA` in `shared-data.js`:

```
CAISO:  [null, 214, 116, 475, 138, 290, 305, 347, 340]
ERCOT:  [null,  78, 112, 118, 110, 154, 192, 208, 266]
PJM:    [null,  62,  63,  82, 105, 150, 339, 215, 514]
NYISO:  [null, 117,  74,  98, 155, 260, 310, 449, 399]
NEISO:  [null, 141, 166, 108, 170, 329, 536, 491, 677]
```

Every ISO has at least 2-3 monotonicity violations. The `MAC_STEPWISE_FAN` percentile data is even worse, with oscillations of 10x+ between adjacent steps.

## Root Cause: Resource Reshuffling Across Thresholds

The optimizer independently optimizes each threshold's resource portfolio. At threshold N, it finds the globally cheapest mix to hit N% hourly matching. At threshold N+1, it independently finds a different cheapest mix. These are two different optimization problems producing fundamentally different resource mixes.

When computing marginal MAC as `(cost[N+1] - cost[N]) / (CO2[N+1] - CO2[N])`, we're comparing two independently optimized systems — not measuring the incremental cost of adding to an existing portfolio. The "delta" is meaningless when the portfolio reshuffled.

**Specific manifestations:**
- **Small ΔCO2 denominators**: When CO2 barely changes between thresholds, step MAC spikes to thousands $/ton
- **Negative ΔCO2**: Higher threshold can mean LESS CO2 abated (optimizer swaps generation for storage)
- **Cost reversals**: Occasionally a higher threshold finds a cheaper solution (better local minimum)

## Strategy: Two-Zone Marginal MAC

The key insight: **grid decarbonization holds to ~92.5% no matter what**. The economics clearly support going from 75% to 90% — granular stepwise anomalies in that range are noise from independent optimization, not real economic signals.

### Zone 1: 75% → 90% — Single Aggregate Marginal MAC
- **One number per (ISO, scenario)**: `MAC_75_to_90 = (cost[90%] - cost[75%]) × demand / (CO2[90%] - CO2[75%])`
- This is the "grid backbone" marginal cost — what it costs per ton to go from 75% to 90% matching
- Resource reshuffling between 75/80/85/87.5 is irrelevant because the destination (90%) is what matters
- No monotonicity issue because there's only one value

### Zone 2: 90% → 100% — Granular Checkpoints with Enforced Monotonicity
- **5 stepwise values**: 90→92.5, 92.5→95, 95→97.5, 97.5→99, 99→100
- Each step: `MAC_step = (cost[t+1] - cost[t]) × demand / (CO2[t+1] - CO2[t])`
- **Enforced monotonicity**: `step_mac[t] = max(raw_step_mac[t], step_mac[t-1])`
- The Zone 1 aggregate MAC serves as the floor for the first Zone 2 step (90→92.5)
- **Convex hull as backup**: For scenarios where raw stepwise is severely non-monotonic (>50% drop between steps), use the convex hull slope instead of the running-max correction. This prevents the running max from holding an artificially high spike across many steps.

### Result: 6-Value Marginal MAC Curve

```
[MAC_75→90, MAC_90→92.5, MAC_92.5→95, MAC_95→97.5, MAC_97.5→99, MAC_99→100]
```

Guaranteed non-decreasing. Tells the economic story clearly:
1. How much does the grid backbone cost per ton?
2. How does marginal cost escalate in the last mile?

## Implementation Steps

### Step 1: New Marginal MAC Computation (`compute_mac_stats.py`)

Add new function `compute_two_zone_marginal_mac()`:

```python
def compute_two_zone_marginal_mac(iso_data, scenario_key):
    """
    Two-zone marginal MAC:
      Zone 1 (75→90%): single aggregate MAC
      Zone 2 (90→100%): granular steps with enforced monotonicity
    Returns: 6-element list [agg_75_90, step_90_92.5, step_92.5_95, step_95_97.5, step_97.5_99, step_99_100]
    """
    demand_mwh = iso_data.get('annual_demand_mwh', 1)

    # Zone 1: aggregate 75→90
    cost_75 = get_incremental_cost(iso_data, '75', scenario_key)
    cost_90 = get_incremental_cost(iso_data, '90', scenario_key)
    co2_75  = get_co2_abated(iso_data, '75', scenario_key)
    co2_90  = get_co2_abated(iso_data, '90', scenario_key)

    if co2_90 > co2_75 and cost_90 >= cost_75:
        zone1_mac = ((cost_90 - cost_75) * demand_mwh) / (co2_90 - co2_75)
    else:
        zone1_mac = None  # Edge case: flag for review

    # Zone 2: granular steps with monotonicity floor
    zone2_thresholds = [90, 92.5, 95, 97.5, 99, 100]
    zone2_steps = []
    running_floor = zone1_mac or 0  # Zone 1 MAC is the floor

    for i in range(1, len(zone2_thresholds)):
        t_prev = str(zone2_thresholds[i-1])
        t_curr = str(zone2_thresholds[i])

        cost_prev = get_incremental_cost(iso_data, t_prev, scenario_key)
        cost_curr = get_incremental_cost(iso_data, t_curr, scenario_key)
        co2_prev  = get_co2_abated(iso_data, t_prev, scenario_key)
        co2_curr  = get_co2_abated(iso_data, t_curr, scenario_key)

        if co2_curr > co2_prev and cost_curr >= cost_prev:
            raw_mac = ((cost_curr - cost_prev) * demand_mwh) / (co2_curr - co2_prev)
        elif co2_curr <= co2_prev:
            # CO2 plateau/decrease — use convex hull interpolation
            raw_mac = interpolate_from_surrounding_steps(...)
        else:
            raw_mac = running_floor  # Cost decrease → hold floor

        # Enforce monotonicity
        corrected_mac = max(raw_mac, running_floor)
        running_floor = corrected_mac
        zone2_steps.append(round(corrected_mac, 1))

    return [round(zone1_mac, 1)] + zone2_steps
```

### Step 2: Convex Hull for Edge Cases

For scenarios where ΔCO2 ≤ 0 at a step (CO2 decreased or plateaued due to reshuffling), we can't compute a raw step MAC. Instead of inserting `null`, use the convex hull of the (CO2, cost) frontier across all 10 thresholds to interpolate what the marginal MAC "should" be at that step.

```python
def lower_convex_hull(points):
    """Andrew's monotone chain algorithm for lower hull of (x, y) points.
    Points must be sorted by x. Returns subset of points on lower hull.
    Slopes between consecutive hull vertices are non-decreasing."""
    points = sorted(points)
    hull = []
    for p in points:
        while len(hull) >= 2 and cross(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)
    return hull

def cross(O, A, B):
    return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0])
```

The hull slopes provide the "true" efficient-frontier marginal MAC where raw data is unreliable.

### Step 3: Fix Fan Chart Percentiles

**Current bug in `compute_stepwise_fan()`**: P10/P50/P90 taken independently at each step → "percentile crossing" where different scenarios contribute to different steps.

**Fix — consistent scenario ranking**:
1. For each of 324 scenarios, compute its Zone 1 aggregate MAC + Zone 2 stepwise MACs (corrected)
2. Rank scenarios by their **total portfolio cost at 99%** (stable ranking metric)
3. Select scenarios at P10/P50/P90 of this ranking
4. Use those specific scenarios' full 6-value marginal MAC curves
5. Each percentile curve comes from a coherent cost scenario

### Step 4: Update Data Structure

New `MARGINAL_MAC_DATA` format in `shared-data.js`:

```javascript
const MARGINAL_MAC_DATA = {
    // Labels for the 6-value curve
    labels: ['75→90%', '90→92.5%', '92.5→95%', '95→97.5%', '97.5→99%', '99→100%'],

    medium: {
        CAISO:  [mac1, mac2, mac3, mac4, mac5, mac6],  // 6 values, non-decreasing
        ERCOT:  [...],
        PJM:    [...],
        NYISO:  [...],
        NEISO:  [...]
    },
    low: { ... },   // P10 from consistent scenario ranking
    high: { ... }   // P90 from consistent scenario ranking
};
```

### Step 5: Update Dashboard Visualizations

**`abatement_comparison.html`** (main consumer of marginal MAC):
- Bar chart (line 531-602): Update to use 6 bars instead of 9
  - `STEP_LABELS` → `['75→90%', '90→92.5%', '92.5→95%', '95→97.5%', '97.5→99%', '99→100%']`
  - Index references change from `marg[iso][si + 1]` to `marg[iso][si]`
- Insight text (line 596-601): Update zone descriptions
- Region cards (line 683-690): Update min/max step references
- Crossover analysis (line 819-828): Update threshold iteration

**`dashboard.html`** (line 3649): Update prose about stepwise MAC

**`research_paper.html`** (lines 736, 1882, 2025): Update marginal MAC narrative to describe two-zone methodology

**`abatement_dashboard.html`**: Update envelope/path-constrained charts if they show stepwise data

### Step 6: Fix Path-Constrained Reference

`compute_path_constrained_mac()` should also use the two-zone structure and enforce stepwise monotonicity. Add running-max on step_mac values.

### Step 7: Update SPEC.md

Add section documenting:
- Two-zone marginal MAC methodology
- Rationale: grid backbone holds to ~92.5%, so sub-90% granularity is noise
- Convex hull edge case handling
- Consistent percentile ranking
- Enforce monotonicity for 90-100% zone

### Step 8: Re-run `compute_mac_stats.py`

After code changes:
1. `python compute_mac_stats.py` → regenerates `mac-stats-data.js` and `mac_stats.json`
2. Manually update `shared-data.js` with new `MARGINAL_MAC_DATA` format
3. Verify all dashboard pages render correctly

## Files to Modify

| # | File | Change |
|---|------|--------|
| 1 | `compute_mac_stats.py` | Add `compute_two_zone_marginal_mac()`, `lower_convex_hull()`, fix `compute_stepwise_fan()` for consistent ranking, fix `compute_path_constrained_mac()` |
| 2 | `dashboard/js/mac-stats-data.js` | Auto-regenerated |
| 3 | `dashboard/js/shared-data.js` | Restructure `MARGINAL_MAC_DATA` (6-value format), update `CROSSOVER_SUMMARY` |
| 4 | `dashboard/abatement_comparison.html` | Update bar chart to 6 bars, fix STEP_LABELS, update insight text, update region cards |
| 5 | `dashboard/dashboard.html` | Update marginal MAC prose |
| 6 | `dashboard/research_paper.html` | Update marginal MAC narrative |
| 7 | `dashboard/abatement_dashboard.html` | Update envelope/path charts if applicable |
| 8 | `SPEC.md` | Document two-zone methodology decision |

## Verification Criteria

1. **Monotonicity**: All 6 values non-decreasing for every ISO and every scenario
2. **Zone 1 sanity**: Aggregate 75→90% MAC should be $20-150/ton across ISOs (literature range)
3. **Zone 2 escalation**: Clear hockey stick — last step (99→100%) should be 3-15× the Zone 1 value
4. **Fan coherence**: P10 ≤ P50 ≤ P90 at every step (consistent scenario ranking prevents crossing)
5. **No nulls**: All 6 values computable for Medium scenario in all ISOs
6. **Literature alignment**: Marginal MAC at 99→100% consistent with NREL ($930/ton), Riepin & Brown findings
7. **Dashboard renders**: All charts display correctly with new 6-bar format

## Integrity Safeguards

- **Raw data preserved**: Keep raw 9-step marginal MAC alongside corrected 6-value version for transparency
- **Methodology transparent**: Tooltip/methodology page explains two-zone approach
- **Convex hull documented**: When hull interpolation is used (ΔCO2 ≤ 0 edge cases), flag in output data
- **No fabrication**: If a scenario genuinely can't produce a computable MAC at a step, it's excluded from percentile computation rather than assigned a fake value
