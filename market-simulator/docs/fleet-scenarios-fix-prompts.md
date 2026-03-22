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
- **Nuclear uprates decrease gen** (Issue 2): The uprate option exists in the sidebar UI but the dispatch engine doesn't read `_uprate_mw` — it uses only `p.capacity_mw`.
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

*Prompts 2-5 will be added in subsequent commits.*
