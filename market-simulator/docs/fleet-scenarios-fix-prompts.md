# Fleet Scenarios Dashboard — Fix Prompts

## Overview
These prompts fix the fleet scenarios dashboard at `dashboard/fleet_scenarios.html` (the live beta page). Each prompt is a self-contained session. Execute in order.

**Source of truth for UI**: `dashboard/fleet_scenarios.html` and `dashboard/js/fleet-*.js` — these have the latest features (uprate dropdown, CCS panel, etc.). The `market-simulator/frontend/` versions are stale.

**Source of truth for calculations**: Changes to calculation logic should be made in `dashboard/js/fleet-dispatch-engine.js`. After stabilizing, sync back to `market-simulator/frontend/js/`.

**Source of truth for data**: `market-simulator/frontend/data/` (sweep_dispatch_data.json, constellation_scenarios.json, fleet_scenario_results_sample.json). Dashboard copies live at `dashboard/data/`.

---

## Root Cause Summary (Read This First)

Every custom fleet configuration increases emissions against baseline. This happens because **two different calculation engines produce two different baselines**:

1. **Displayed baseline fan chart**: Loaded from `fleet_scenario_results_sample.json` — precomputed with synthetic ±12% spread around hardcoded fleet-level TWh constants. Produces ~23 Mt CO₂ baseline.

2. **Custom scenario (sidebar recalculate)**: Computed by `fleet-dispatch-engine.js` dispatching each plant individually against `sweep_dispatch_data.json` (1,215 sweep scenarios). Produces ~30-40 Mt CO₂ for the same unmodified fleet.

**Result**: The waterfall chart (fleet-scenarios.js:996-1049) compares `DATA.scenarios.baseline.plant_detail[yr]` (from precomputed JSON) vs `customScenario.plant_detail[yr]` (from dispatch engine). Since these engines compute different emission values for the SAME unmodified plants, phantom deltas appear everywhere:
- Morgan Energy Center: precomputed says ~1.0 Mt, engine says ~2.83 Mt → phantom +1.83 Mt
- Hillabee: precomputed says ~0.5 Mt, engine says ~2.02 Mt → phantom +1.52 Mt

**The fix**: Both baseline AND custom must use the same `fleet-dispatch-engine.js` calculation path. On page load, compute the baseline using the dispatch engine with an unmodified fleet, then use that as the comparison target.

### Other Issues
- **Accordion collapse** (Issue 1): `renderFleetList()` in fleet-sidebar.js rebuilds full innerHTML on every status change, destroying accordion open/closed DOM state.
- **Nuclear uprates decrease gen** (Issue 2): The uprate UI and dispatch engine code exist in `dashboard/js/` (fleet-sidebar.js:392-396 has uprate dropdown + MW input; fleet-dispatch-engine.js:265,283-284 adds `_uprate_mw` to capacity). But the uprate appears to decrease generation because the waterfall compares against the WRONG baseline (precomputed JSON vs dispatch engine). Once the baseline is unified (Prompt 2), uprates should work correctly. Verify after Prompt 2.
- **No "Default Market" option** (Issue 3): Plants only have Operating/Retired/CCS. Need an option that follows the sweep's economic retirement logic naturally, so plants can retire economically without being forced to operate.
- **Missing 2023 baseline chart** (Issue 4): Only market trajectory delta exists. Need a chart showing delta vs actual 2023 eGRID baseline values.

---

## Prompt 1: Fix Accordion State Preservation

### Context
In `dashboard/js/fleet-sidebar.js`, every time a user changes a plant's status dropdown (e.g., Operating → CCS Retrofit), `onStatusChange()` at line 492 calls `renderFleetList()`. This function rebuilds the entire sidebar HTML via `els.fleetList.innerHTML = html` (line 411), which destroys all DOM state including which accordion groups are open. After rebuild, the auto-collapse logic (lines 424-433) collapses non-fossil categories by default. The user has to re-open the accordion every single time they make any change.

### Task
Fix the accordion in `dashboard/js/fleet-sidebar.js` so that open groups stay open across re-renders:

1. Add a module-level `Set` variable to track which accordion groups are currently open:
   ```javascript
   var openGroups = new Set();
   ```

2. At the TOP of `renderFleetList()` (before innerHTML rebuild), snapshot which groups are currently open by reading DOM state:
   ```javascript
   // Snapshot open groups before DOM rebuild
   var prevOpen = new Set();
   els.fleetList.querySelectorAll('.sb-iso-body').forEach(function(body) {
       if (!body.classList.contains('collapsed')) {
           prevOpen.add(body.dataset.isoBody);
       }
   });
   if (prevOpen.size > 0) openGroups = prevOpen;
   ```

3. After the innerHTML rebuild and event binding (around line 413-434), replace the auto-collapse logic. Instead of always collapsing non-fossil:
   - If `openGroups` has entries, restore those exact groups as open and collapse everything else
   - If `openGroups` is empty (first render), use the current default behavior (fossil open, others collapsed)

4. When an accordion header is clicked, update `openGroups` to reflect the new state.

### Verification
- Open the page, expand a nuclear ISO group
- Change a fossil plant's status dropdown
- The nuclear group should remain open
- Open multiple groups across categories, make changes — all stay open
- Refresh the page — default collapse behavior applies (fossil open, others collapsed)

### Files Changed
- `dashboard/js/fleet-sidebar.js` only (~25 lines changed)

### After Completing
```bash
# Sync to market-simulator
cp dashboard/js/fleet-sidebar.js market-simulator/frontend/js/fleet-sidebar.js
# Commit
git add dashboard/js/fleet-sidebar.js market-simulator/frontend/js/fleet-sidebar.js
git commit -m "Fix accordion state: preserve open groups across sidebar re-renders"
git push -u origin claude/fix-fleet-scenarios-calcs-AYeYh
```

---

## Prompt 2: Unify Baseline & Custom Calculation Engines (Critical Fix)

### Context
This is the root cause of ALL calculation bugs. The baseline fan chart loads from `fleet_scenario_results_sample.json` (precomputed with synthetic ±12% spread, ~23 Mt CO₂). Custom scenarios are computed by `fleet-dispatch-engine.js` against `sweep_dispatch_data.json` (~30-40 Mt CO₂ for the same fleet). When the waterfall chart compares these two different baselines, every plant shows phantom emission changes even when the user hasn't touched it.

**The fix**: On page load, compute the baseline using the SAME `fleet-dispatch-engine.js` dispatch engine with an unmodified fleet. Store this as the baseline for all comparisons. The precomputed JSON becomes a fallback only.

### Task
Modify `dashboard/js/fleet-scenarios.js` and `dashboard/js/fleet-sidebar.js` so that both baseline and custom scenarios use the same calculation engine:

#### Step 1: Compute baseline on load (fleet-sidebar.js)

In `fleet-sidebar.js`, after both fleet data AND sweep data have loaded (in the `onAllLoaded()` function), compute the baseline automatically:

```javascript
function onAllLoaded() {
    // ... existing status message code ...

    // AUTO-COMPUTE BASELINE using same engine as custom scenarios
    if (sweepData && baseFleet.length > 0) {
        computeAndSetBaseline();
    }
}

function computeAndSetBaseline() {
    // Use unmodified base fleet (no _action flags) — this IS the market trajectory
    var unmodifiedFleet = baseFleet.map(function(p) {
        return Object.assign({}, p, {
            _action: null,
            _year_online: null,
            _ccs_target_rate: 0,
            _uprate_mw: 0
        });
    });

    var result = FleetDispatchEngine.computeFleetDispatch(unmodifiedFleet, sweepData, {
        ccs_derate_pct: 0,
        ccs_capture_rate_pct: 0,
        ccs_cf_pct: 85
    });

    // Push to chart system as the BASELINE (replacing precomputed JSON baseline)
    if (window.FLEET_SCENARIOS_API && window.FLEET_SCENARIOS_API.setComputedBaseline) {
        window.FLEET_SCENARIOS_API.setComputedBaseline({
            envelope: result.envelope,
            plant_detail: result.plant_detail,
            generation_by_fuel: result.generation_by_fuel,
            emissions_by_fuel: result.emissions_by_fuel,
            fleet_summary: result.fleet_summary
        });
    }
}
```

#### Step 2: Accept computed baseline in chart system (fleet-scenarios.js)

In `fleet-scenarios.js`, add a `setComputedBaseline()` method to `FLEET_SCENARIOS_API` that replaces `DATA.scenarios.baseline` with the engine-computed result:

```javascript
// In the FLEET_SCENARIOS_API object:
setComputedBaseline: function(baselineData) {
    if (!DATA) return;
    // Replace the precomputed baseline with engine-computed baseline
    DATA.scenarios.baseline = baselineData;
    // Rebuild all charts with the new baseline
    updateAllCharts();
}
```

#### Step 3: Ensure baseline identity

The key principle: **baseline and custom must be identical until the first intervention year.** Since both now use the same engine with the same sweep data, an unmodified custom fleet will produce EXACTLY the same numbers as the baseline. The waterfall will show zero delta for all untouched plants.

#### Step 4: Handle the loading sequence

The current loading sequence is:
1. `fleet-scenarios.js` loads `fleet_scenario_results_sample.json` → sets `DATA` → builds charts
2. `fleet-sidebar.js` loads fleet config + sweep data → renders sidebar

After this change:
1. `fleet-scenarios.js` loads `fleet_scenario_results_sample.json` → sets `DATA` → builds charts (initial render with precomputed data for fast first paint)
2. `fleet-sidebar.js` loads fleet config + sweep data → **computes baseline via dispatch engine** → calls `setComputedBaseline()` → charts update with engine-computed baseline
3. User edits fleet → clicks Recalculate → custom scenario computed with same engine → deltas are real

This means there's a brief flash of the old baseline before the engine-computed one replaces it. That's fine — it's a loading state.

### Verification
1. **No phantom deltas**: Open sidebar, make NO changes, click Recalculate. The waterfall should show zero delta for ALL plants (or very close to zero — tiny floating point differences OK).
2. **CCS reduces emissions**: Set Colorado Bend II to CCS Retrofit (2028), 15% derate, 80% CF, 95% capture. Recalculate. The waterfall should show Colorado Bend II with a NEGATIVE delta (emission reduction).
3. **Nuclear uprate increases gen**: Set Limerick to Uprate, +343 MW, year 2030. Recalculate. Nuclear generation should INCREASE by ~2.76 TWh. No phantom changes at Morgan or Hillabee.
4. **Unrelated plants untouched**: When modifying only one plant, all other plants should show ~0 delta in the waterfall.

### Files Changed
- `dashboard/js/fleet-sidebar.js` — add `computeAndSetBaseline()`, call from `onAllLoaded()`
- `dashboard/js/fleet-scenarios.js` — add `setComputedBaseline()` to API, update `updateAllCharts()`

### After Completing
```bash
# Sync to market-simulator
cp dashboard/js/fleet-sidebar.js market-simulator/frontend/js/fleet-sidebar.js
cp dashboard/js/fleet-scenarios.js market-simulator/frontend/js/fleet-scenarios.js
# Commit
git add dashboard/js/fleet-sidebar.js dashboard/js/fleet-scenarios.js \
    market-simulator/frontend/js/fleet-sidebar.js market-simulator/frontend/js/fleet-scenarios.js
git commit -m "Unify baseline and custom calculation engines — fix phantom emission deltas

Both baseline and custom scenarios now use fleet-dispatch-engine.js with
sweep_dispatch_data.json. Eliminates phantom deltas from baseline mismatch
(precomputed ~23Mt vs engine ~30-40Mt). Waterfall now shows real deltas only."
git push -u origin claude/fix-fleet-scenarios-calcs-AYeYh
```

---

*Prompts 3-5 will be added in subsequent commits.*
