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
- **Nuclear uprates decrease gen** (Issue 2): The uprate UI and dispatch engine code exist in `dashboard/js/` (fleet-sidebar.js:392-396 has uprate dropdown + MW input; fleet-dispatch-engine.js:265,283-284 adds `_uprate_mw` to capacity). The uprate appears to decrease generation because the waterfall compares against the WRONG baseline (precomputed JSON vs dispatch engine). Once the baseline is unified (Prompt 2), uprates should work correctly. **Important context**: No nuclear plants should retire before 2032 under any scenario due to 45U production tax credits, which are built into the 1215 sweep. Nuclear gen decreasing before 2032 is a clear sign of the baseline mismatch, not a real economic signal. Verify after Prompt 2.
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

## Prompt 3: Add "Default Market" Status Option

### Context
Currently the status dropdown for each plant has: Operating, Retired, CCS Retrofit (fossil CCS-eligible), Uprate (nuclear). The problem: selecting "Operating" forces the plant to run at its sweep CF even if it would economically retire. There's no way to say "let the market decide" — which is what the baseline trajectory represents.

The user needs a "Default Market" option that is the AUTO-SELECTED DEFAULT for all plants. This means: the plant follows whatever the 1,215-scenario sweep says — if it's economic it runs, if not it retires. "Operating" should mean "force this plant to operate regardless of economics." "Retired" means "force retire regardless."

### Task

#### Step 1: Add "Default Market" to dropdowns (fleet-sidebar.js)

In `renderFleetList()`, change the status `<select>` for both fossil and non-fossil plants:

**For fossil plants** (currently around line 366-372):
```html
<option value="default_market" [selected if no _action]>Default Market</option>
<option value="operating" [selected if _action === 'operating_override']>Operating (forced)</option>
<option value="retire" [selected if _action === 'retire']>Retired</option>
<option value="ccs_retrofit" [selected if _action === 'ccs_retrofit']>CCS Retrofit</option>  <!-- CCS-eligible only -->
```

**For non-fossil plants** (currently around line 390-393):
```html
<option value="default_market" [selected if no _action]>Default Market</option>
<option value="operating" [selected if _action === 'operating_override']>Operating (forced)</option>
<option value="retire" [selected if _action === 'retire']>Retired</option>
<option value="uprate" [selected if _action === 'uprate']>Uprate</option>  <!-- nuclear only -->
```

#### Step 2: Handle "Default Market" in dispatch engine (fleet-dispatch-engine.js)

In the fossil plant dispatch loop (second pass), the current logic at line 374 already applies economic retirement:
```javascript
if (yearHasData[yi3] && margin < 0) adjustedCf = 0;
```

This needs to be conditional on the plant's action:
- `_action === null` or `_action === 'default_market'`: Apply economic retirement as-is (margin < 0 → CF = 0). This IS the market trajectory.
- `_action === 'operating_override'`: Skip economic retirement — force the plant to run at its sweep CF regardless of margin.
- `_action === 'retire'`: Force CF = 0 from `_year_online` onward (already works).

```javascript
// Economic retirement — only for default_market plants
var isDefaultMarket = !action || action === 'default_market';
if (isDefaultMarket && yearHasData[yi3] && margin < 0) {
    adjustedCf = 0;
}
// Forced operating: skip economic retirement entirely (plant runs at sweep CF)
// No change needed — just don't zero out the CF
```

#### Step 3: Update onStatusChange (fleet-sidebar.js)

When a plant is set to "Default Market", clear all action flags:
```javascript
if (newStatus === 'default_market') {
    p._action = null;  // null = default market behavior
    p._year_online = null;
    p._ccs_target_rate = 0;
    p._uprate_mw = 0;
}
if (newStatus === 'operating') {
    p._action = 'operating_override';  // New: distinguishes forced-operate from default
}
```

#### Step 4: Baseline identity guarantee

Since baseline is computed with all plants at `_action = null` (Prompt 2), and "Default Market" also sets `_action = null`, a fleet with all plants on "Default Market" will produce EXACTLY the same result as baseline. No divergence until the user changes something.

### Verification
1. Load page — all plants should show "Default Market" selected by default
2. Change nothing, click Recalculate — zero delta everywhere (same as baseline)
3. Change one plant to "Operating (forced)" where it would economically retire — that plant's emissions increase (it's now forced to run), all other plants unchanged
4. Change one plant to "Retired" — that plant's emissions drop to zero from the retirement year
5. Baseline and custom trajectories are identical until the first year where any intervention occurs

### Files Changed
- `dashboard/js/fleet-sidebar.js` — dropdown options, `onStatusChange()` handler
- `dashboard/js/fleet-dispatch-engine.js` — conditional economic retirement logic

### After Completing
```bash
cp dashboard/js/fleet-sidebar.js market-simulator/frontend/js/fleet-sidebar.js
cp dashboard/js/fleet-dispatch-engine.js market-simulator/frontend/js/fleet-dispatch-engine.js
git add dashboard/js/fleet-sidebar.js dashboard/js/fleet-dispatch-engine.js \
    market-simulator/frontend/js/fleet-sidebar.js market-simulator/frontend/js/fleet-dispatch-engine.js
git commit -m "Add Default Market status option — plants follow sweep economics by default

Default Market is the auto-selected default. Plants follow 1215 sweep
economic retirement logic. Operating (forced) overrides retirement.
Guarantees baseline identity when no interventions are applied."
git push -u origin claude/fix-fleet-scenarios-calcs-AYeYh
```

---

## Prompt 4: Add Emissions Delta vs 2023 Baseline Chart + Rename Existing

### Context
The existing waterfall chart shows "Emissions Delta vs. Baseline — {year}" comparing custom scenario to market trajectory baseline. The user also needs a chart comparing against the actual 2023 eGRID baseline — this answers "how much have I reduced from where I started?" vs "how much have I reduced from where the market would take me?"

### Task

#### Step 1: Rename existing waterfall chart

In `dashboard/fleet_scenarios.html`, rename the waterfall chart heading:
- From: `Emissions Delta vs. Baseline — {year}`
- To: `Emissions Delta vs Market Trajectory Baseline — {year}`

In `dashboard/js/fleet-scenarios.js`, update the `updateWaterfallChart()` function (around line 1000):
```javascript
document.getElementById('waterfallTitle').textContent =
    'Emissions Delta vs Market Trajectory Baseline — ' + yr;
```

#### Step 2: Add new chart HTML (fleet_scenarios.html)

Add a new chart card AFTER the existing waterfall chart card (after its closing `</div>`). Place it before the generation mix chart:

```html
<div class="chart-card">
    <h3 id="waterfall2023Title">Emissions Delta vs 2023 Baseline</h3>
    <div class="chart-wrap" style="height:400px;">
        <canvas id="waterfall2023Chart"></canvas>
    </div>
</div>
```

#### Step 3: Build the 2023 baseline chart (fleet-scenarios.js)

Add a new chart variable and build/update functions. The 2023 baseline data comes from the engine-computed baseline's `plant_detail['2023']` — this represents actual eGRID 2023 emissions per plant (since the dispatch engine uses static fallback CFs for 2023 that match eGRID actuals).

```javascript
var waterfall2023Chart = null;

function buildWaterfall2023Chart() {
    var ctx = document.getElementById('waterfall2023Chart').getContext('2d');
    waterfall2023Chart = new Chart(ctx, {
        type: 'bar',
        data: { labels: [], datasets: [] },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            animation: { duration: 300 },
            scales: {
                x: {
                    title: { display: true, text: 'Emissions Delta vs 2023 (Mt CO₂)', font: { size: 12 } },
                    grid: { color: '#E0E6EF' }
                },
                y: { ticks: { font: { size: 11 } }, grid: { display: false } }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(ctx) {
                            var v = ctx.raw;
                            return (v > 0 ? '+' : '') + v.toFixed(2) + ' Mt CO₂ vs 2023';
                        }
                    }
                }
            }
        }
    });
    updateWaterfall2023Chart();
}
```

The `updateWaterfall2023Chart()` function compares the CUSTOM scenario's plant_detail at selectedYear against the baseline's plant_detail at 2023:

```javascript
function updateWaterfall2023Chart() {
    if (!waterfall2023Chart || !DATA) return;

    var yr = String(nearestYear(selectedYear, getYears()));
    document.getElementById('waterfall2023Title').textContent =
        'Emissions Delta vs 2023 Baseline — ' + yr;

    // Use custom scenario if available, otherwise use baseline at selectedYear
    var scenarioToCompare = customScenario || DATA.scenarios.baseline;
    var scPlants = scenarioToCompare.plant_detail ? scenarioToCompare.plant_detail[yr] : null;

    // 2023 baseline from engine-computed baseline
    var base2023Plants = DATA.scenarios.baseline.plant_detail ?
        DATA.scenarios.baseline.plant_detail['2023'] : null;

    if (!base2023Plants || !scPlants) {
        waterfall2023Chart.data.labels = ['No 2023 data'];
        waterfall2023Chart.data.datasets = [{ data: [0], backgroundColor: '#ddd' }];
        waterfall2023Chart.update('active');
        return;
    }

    // Build delta: scenario[yr] emissions - baseline[2023] emissions per plant
    var baseMap = {};
    base2023Plants.forEach(function(p) { baseMap[p.orispl] = p; });

    var deltas = [];
    scPlants.forEach(function(p) {
        var baseP = baseMap[p.orispl];
        var baseE = baseP ? baseP.emissions_mt : 0;
        var delta = baseE - p.emissions_mt;  // positive = reduction from 2023
        if (Math.abs(delta) > 0.001) {
            deltas.push({ name: p.name, delta: delta });
        }
    });

    deltas.sort(function(a, b) { return Math.abs(b.delta) - Math.abs(a.delta); });

    // Truncate to top 15 + Other
    var other = 0;
    if (deltas.length > 15) {
        for (var i = 15; i < deltas.length; i++) other += deltas[i].delta;
        deltas = deltas.slice(0, 15);
        if (Math.abs(other) > 0.001) deltas.push({ name: 'Other', delta: other });
    }

    var labels = deltas.map(function(d) { return d.name; });
    var values = deltas.map(function(d) { return d.delta; });
    var colors = values.map(function(v) { return v >= 0 ? '#6BA543' : '#DC2626'; });

    waterfall2023Chart.data.labels = labels;
    waterfall2023Chart.data.datasets = [{
        data: values,
        backgroundColor: colors,
        borderRadius: 3
    }];
    waterfall2023Chart.update('active');
}
```

#### Step 4: Wire into existing update flow

- Call `buildWaterfall2023Chart()` in `onDataLoaded()` after building the existing waterfall
- Call `updateWaterfall2023Chart()` in `updateAllCharts()` and whenever the year slider changes
- Call `updateWaterfall2023Chart()` in `setCustomScenario()` after setting `customScenario`

### Verification
1. Load page — both waterfall charts render. The 2023 chart shows deltas between market trajectory at selectedYear vs 2023 actuals.
2. Slide the year slider — both charts update title and data for the selected year.
3. Apply CCS to Colorado Bend II — the 2023 chart shows a large negative delta (reduction from 2023 baseline). The market trajectory chart shows the delta vs what would have happened without intervention.
4. At year 2023, the 2023 chart should show ~0 delta for all plants (comparing 2023 to itself).

### Files Changed
- `dashboard/fleet_scenarios.html` — add new chart card HTML
- `dashboard/js/fleet-scenarios.js` — add chart build/update functions, wire into update flow

### After Completing
```bash
cp dashboard/fleet_scenarios.html market-simulator/frontend/fleet-scenarios.html
cp dashboard/js/fleet-scenarios.js market-simulator/frontend/js/fleet-scenarios.js
git add dashboard/fleet_scenarios.html dashboard/js/fleet-scenarios.js \
    market-simulator/frontend/fleet-scenarios.html market-simulator/frontend/js/fleet-scenarios.js
git commit -m "Add Emissions Delta vs 2023 Baseline chart, rename existing waterfall

Two waterfall charts: (1) Delta vs Market Trajectory Baseline shows impact of
interventions against probabilistic market sweep. (2) Delta vs 2023 Baseline
shows total reduction from actual eGRID 2023 starting point."
git push -u origin claude/fix-fleet-scenarios-calcs-AYeYh
```

---

## Prompt 5: End-to-End QA & Verification

### Context
Prompts 1-4 fix the accordion UI, unify the calculation engine, add Default Market status, and add the 2023 baseline chart. This prompt verifies everything works end-to-end with specific test cases the user provided.

### Task

#### Test Case 1: Colorado Bend II CCS Retrofit
User's expected calculation:
- 2023 eGRID: 73% CF, 2.828 Mt CO₂
- With CCS: 80% CF, 15% derate, 95% capture rate
- Gross gen at 80% CF: 1,210 MW × 0.80 × 8,760 = 8,479,680 MWh
- Net gen (after 15% derate): 8,479,680 × 0.85 = 7,207,728 MWh
- Baseline CO₂ rate: 2.828 Mt / (1,210 × 0.73 × 8,760 MWh) = ~0.366 t/MWh
- Gross emissions: 8,479,680 × 0.366 / 1e6 = 3.104 Mt
- After 95% capture: 3.104 × 0.05 = 0.155 Mt
- **Reduction from 2023 baseline: 2.828 - 0.155 = 2.673 Mt (94.5% reduction)**

Steps:
1. Open dashboard/fleet_scenarios.html
2. Open sidebar, find Colorado Bend II (ERCOT, gas_ccgt)
3. Set CCS panel: Derate = 15%, Capture Rate = 95%, CF = 80%
4. Set Colorado Bend II status to "CCS Retrofit", year 2028
5. Click Recalculate
6. Check "Emissions Delta vs 2023 Baseline" chart at year 2030 — Colorado Bend II should show ~-2.67 Mt
7. Check "Emissions Delta vs Market Trajectory Baseline" chart — Colorado Bend II should show a negative delta (reduction vs what the sweep says it would emit without CCS)
8. **Critical**: No other plants should show significant deltas unless they're economically retiring under the sweep

#### Test Case 2: Limerick Nuclear Uprate
User's expected calculation:
- Limerick (PJM, nuclear): add +343 MW uprate in 2030
- At 92% CF: 343 × 0.92 × 8,760 / 1e6 = 2.764 TWh additional clean generation
- Total nuclear fleet should INCREASE by ~2.76 TWh (from ~187 TWh baseline)
- Zero emission impact (nuclear = 0 emissions)

Steps:
1. Set Limerick to "Uprate", +343 MW, year 2030
2. Click Recalculate
3. Verify nuclear generation at 2030 is ~190 TWh (baseline + 2.76), NOT 171 TWh
4. **Critical**: Morgan Energy Center (Alabama, ERCOT) and Hillabee (Alabama, ERCOT) should show ZERO delta — they're completely unrelated to Limerick

#### Test Case 3: Baseline Identity
Steps:
1. Open sidebar, change NOTHING
2. Click Recalculate
3. Both waterfall charts should show zero or near-zero deltas for ALL plants
4. The fan chart custom scenario line should overlap exactly with the baseline

#### Test Case 4: Default Market Behavior
Steps:
1. Set all plants to "Default Market" (should be default)
2. Click Recalculate — identical to baseline
3. Change one plant to "Operating (forced)" that would normally retire economically
4. That plant shows increased emissions; all others unchanged
5. Change it to "Retired" — that plant shows zero emissions from retirement year

#### Test Case 5: Accordion Persistence
Steps:
1. Expand Nuclear > PJM group
2. Change a fossil plant status in a different group
3. Nuclear > PJM should remain open
4. Expand 3 groups across categories, make changes — all stay open

### Debug Checklist (if tests fail)
- Open browser console, check for errors
- After Recalculate, log `customScenario.envelope['2030']` and `DATA.scenarios.baseline.envelope['2030']` — if baseline is still from precomputed JSON (~23 Mt), the engine baseline wasn't set correctly
- Check `DATA.scenarios.baseline.plant_detail['2023']` has Colorado Bend II with ~2.83 Mt — if not, the 2023 static CF fallback isn't working
- If uprates show wrong gen, check `FleetSidebar.getFleetPlants()` for the plant — `_action` should be `'uprate'` and `_uprate_mw` should be 343

### Files Changed
- None (QA only) — or minor fixes found during testing

### After Completing
```bash
git add -A
git commit -m "QA verification complete — all fleet scenario calculations verified

Tested: CCS retrofit (Colorado Bend II), nuclear uprate (Limerick),
baseline identity, Default Market behavior, accordion persistence."
git push -u origin claude/fix-fleet-scenarios-calcs-AYeYh
```
