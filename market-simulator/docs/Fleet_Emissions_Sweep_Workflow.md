# Fleet Emissions Sweep → `emissions_dashboard.html` Workflow

> **Purpose**: Step-by-step implementation prompts to build a fleet-level emissions sweep pipeline that feeds the interactive `emissions_dashboard.html` (Emissions Trajectory Explorer), NOT the simpler `emissions.html` in `frontend/`.
>
> **Key distinction**: `frontend/emissions.html` is a basic CCS dispatch page that calls the backend API in real-time. `dashboard/emissions_dashboard.html` is a standalone pre-computed explorer with P10/P50/P90 fan bands, interactive plant CCS toggles, scenario toggles, and trajectory targets — all powered by a pre-generated JS data file (`js/emissions-dashboard-data.js`).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Step A: Fleet Baseline Extraction                          │
│  scripts/fleet_model.py + data/epa-campd/                   │
│  → constellation_fleet.json (39 CCGT plants, equity shares) │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Step B: Market Scenario Sweep                              │
│  scripts/market_simulation.py                               │
│  540 scenarios × 7 ISOs × 25-year trajectory                │
│  → year_results[] per scenario                              │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Step C: Fleet Dispatch + CCS Emissions                     │
│  scripts/constellation_dispatch_integrated.py               │
│  Per-plant dispatch → baseline CO2, CCS residual, delta     │
│  → fan_bands{p10,p50,p90}, per-plant trajectories           │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Step D: Data Export for Dashboard                          │
│  New script: generate_emissions_dashboard_data.py           │
│  → dashboard/js/emissions-dashboard-data.js                 │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Step E: Dashboard Rendering                                │
│  dashboard/emissions_dashboard.html                         │
│  Chart.js fan chart + plant toggles + scenario controls     │
└─────────────────────────────────────────────────────────────┘
```

---

## Prompt 1: Extract & Validate Fleet Baseline Data

**Goal**: Ensure `constellation_fleet.json` has complete, validated plant data for all 39 CCGT units.

**Context files to read first**:
- `scripts/fleet_model.py` — EIA-860/923 plant data integration
- `scripts/constellation_dispatch_integrated.py:37-68` — `load_ccs_plants_from_file()`
- `data/` — check for existing `constellation_fleet.json`

**Prompt**:

```
Read scripts/fleet_model.py and scripts/constellation_dispatch_integrated.py (lines 37-68).

Verify that constellation_fleet.json exists and contains all 39 Constellation CCGT plants with these required fields per plant:
- orispl (EIA plant ID)
- name
- iso (CAISO|ERCOT|PJM|NYISO|NEISO|MISO|SPP)
- capacity_mw
- co2_rate (tCO2/MWh, should be ~0.37 for CCGT)
- equity_pct (Constellation's ownership share, 0-1)
- ccs_eligible (boolean)
- heat_rate (MMBtu/MWh, should be ~7.0 for CCGT)

If any fields are missing or the file doesn't exist, generate it from EIA-860/923 data using fleet_model.py.

Validate:
1. Sum of (capacity_mw × equity_pct) across all plants = total equity capacity
2. All co2_rate values are in [0.30, 0.45] range (CCGT expected)
3. All heat_rate values are in [6.5, 8.0] range
4. ISO assignments match EIA plant locations
5. No duplicate orispl values

Output: validated constellation_fleet.json + summary table (plant count by ISO, total equity MW, total baseline MMt CO2/yr)
```

---

## Prompt 2: Run Market Scenario Sweep

**Goal**: Generate 540 market scenarios (fuel price × carbon price × clean penetration × demand growth combinations) across all ISOs, producing year_results for each.

**Context files to read first**:
- `scripts/market_simulation.py` — core simulation loop (lines ~533-982)
- `scripts/pipeline_config.py` — scenario parameter ranges
- `scripts/lmp_engine.py` — merit-order dispatch

**Prompt**:

```
Read scripts/market_simulation.py (focus on run_market_simulation function and the scenario parameter grid).

The sweep needs to produce year_results for 540 scenarios. Each scenario is a combination of:
- Fuel prices: Low / Medium / High (from pipeline_config.py)
- Carbon price trajectory: None / Low / Medium / High / Very High
- Clean energy penetration growth rate: Slow / Medium / Fast
- Demand growth: Low / Medium / High

For each scenario × ISO combination:
1. Run run_market_simulation() to get 25-year trajectory (2025-2050)
2. Collect year_results: [{year, clean_pct, avg_lmp, iso, fossil_fraction, ...}]
3. Store results keyed by (scenario_id, iso)

Save output to data/results/emissions_sweep_results.json with structure:
{
    "scenarios": {
        "scenario_001": {
            "params": {"fuel": "Low", "carbon": "Medium", ...},
            "iso_results": {
                "PJM": {"year_results": [...]},
                "ERCOT": {"year_results": [...]},
                ...
            }
        },
        ...
    },
    "metadata": {"n_scenarios": 540, "years": [2025,...,2050], "generated_at": "..."}
}

If 540 full runs are too slow, implement a Latin Hypercube sampling approach to get representative coverage with fewer runs, then interpolate. Document the sampling strategy.

IMPORTANT: Use vectorized operations (numpy/numba) for the inner dispatch loop — no Python for-loops over 8760 hours.
```

---

## Prompt 3: Compute Fleet Dispatch & CCS Emissions Per Scenario

**Goal**: For each of the 540 scenarios, run the Constellation fleet dispatch model to get per-plant emissions trajectories with and without CCS.

**Context files to read first**:
- `scripts/constellation_dispatch_integrated.py` — full file, especially:
  - `dispatch_plant()` (lines 71-142) — merit-order positioning + CCS math
  - `run_dispatch_from_sim_results()` (lines 154-310) — fleet orchestration
- `scripts/pipeline_config.py:415-850` — CCS constants (caps, rates, 45Q)
- `scripts/lmp_engine.py:53-76` — heat rates and CO2 rates

**Prompt**:

```
Read scripts/constellation_dispatch_integrated.py completely.

For each of the 540 scenarios from Prompt 2's output (data/results/emissions_sweep_results.json):

1. Call run_dispatch_from_sim_results(year_results, fleet_overrides=None) for the "no CCS" baseline
2. For each possible CCS retrofit combination (or a representative subset):
   - Set fleet_overrides = {plant_orispl: "CCS Retrofit"} for selected plants
   - Re-run dispatch to get CCS-adjusted emissions

Key CCS parameters (from pipeline_config.py):
- CCS capture rate: 95% (5% residual)
- Heat rate penalty: 14% (×1.14)
- CCS capacity factor: 80%
- CO2 rate: 0.37 tCO2/MWh (CCGT baseline)
- CCS residual: capacity_mw × 0.80 × 8760 × 0.37 × 1.14 × 0.05 × equity_pct / 1e6

Collect per scenario:
- fan_bands: {p10: [...], p50: [...], p90: [...]} (percentiles across scenarios for each year)
- per_plant: [{orispl, name, iso, capacity_mw, baseline_co2_mmt, ccs_residual_mmt, ccs_delta_mmt, trajectory_by_year: [...]}]
- trajectories: {at_power_nz: [...], sbti_15c: [...]} (reference target paths)

Save to data/results/fleet_emissions_sweep.json

IMPORTANT: The P10/P50/P90 bands should be computed ACROSS scenarios for each year — i.e., for year 2030, take the 10th/50th/90th percentile of fleet emissions across all 540 scenarios. This captures uncertainty from market conditions, not just CCS adoption.
```

---

## Prompt 4: Build the Data Export Script

**Goal**: Create `generate_emissions_dashboard_data.py` that transforms the sweep results into the JS data file consumed by `emissions_dashboard.html`.

**Context files to read first**:
- `dashboard/emissions_dashboard.html` — full file (understand what data the JS expects)
- `dashboard/js/emissions-dashboard-data.js` — existing data file structure (if it exists)
- `data/results/fleet_emissions_sweep.json` — output from Prompt 3

**Prompt**:

```
Read dashboard/emissions_dashboard.html completely to understand what data structures the page's JavaScript expects.

The page uses a JS data file (emissions-dashboard-data.js) that defines window.EMISSIONS_DATA or similar global. Inspect the existing file to understand the exact schema.

Create scripts/generate_emissions_dashboard_data.py that:

1. Reads data/results/fleet_emissions_sweep.json
2. Transforms it into the exact JS data structure the dashboard expects
3. Writes dashboard/js/emissions-dashboard-data.js

The output JS must include:
- FLEET_PLANTS: array of {orispl, name, iso, capacity_mw, equity_pct, baseline_mmt, ccs_eligible}
- SCENARIO_PARAMS: {fuel_levels, carbon_levels, clean_growth_rates, demand_levels}
- FAN_BANDS: nested by scenario_key → {p10: [by_year], p50: [by_year], p90: [by_year]}
- PLANT_TRAJECTORIES: nested by orispl → scenario_key → {baseline: [by_year], ccs: [by_year]}
- YEAR_LABELS: ["2025", "2026", ..., "2050"]
- REFERENCE_TRAJECTORIES: {at_power_nz: {years, values}, sbti_15c: {years, values}}
- CCS_DELTAS: by orispl → {baseline_mmt, ccs_residual_mmt, delta_mmt}

The script should:
- Validate all plants have complete data before writing
- Log summary stats (total baseline MMt, max CCS reduction, scenario count)
- Format numbers to 4 decimal places to keep file size manageable
- Wrap output in: window.EMISSIONS_DATA = { ... };

Run the script and verify the output file loads without JS errors.
```

---

## Prompt 5: Wire Up Dashboard Interactivity

**Goal**: Ensure `emissions_dashboard.html` correctly renders all charts and controls from the pre-computed data.

**Context files to read first**:
- `dashboard/emissions_dashboard.html` — full file
- `dashboard/js/emissions-dashboard-data.js` — the data file from Prompt 4
- `dashboard/styles/ceg-report.css` — styling

**Prompt**:

```
Read dashboard/emissions_dashboard.html completely.

Verify that the page's inline JavaScript correctly:

1. HEADLINE STATS (4 cards):
   - Baseline (MMt CO2): sum of all plant baselines at equity share
   - P50 at 2050: median fleet emissions in final year across scenarios
   - CCS Reduction: total delta from currently-selected CCS plants
   - Gap to AT Target 2050: P50_2050 minus AT Power NZ target at 2050

2. FAN CHART (main visualization):
   - Chart.js line chart with P10/P50/P90 bands (filled area for P10-P90)
   - P50 as solid line, P10/P90 as light fill boundaries
   - AT Power NZ reference trajectory (dashed red line)
   - SBTi 1.5°C reference trajectory (dashed amber line)
   - X-axis: years (2025-2050), Y-axis: MMt CO2
   - Updates dynamically when scenario toggles or plant CCS selections change

3. PLANT SELECTOR (checklist panel):
   - Grouped by ISO with "Select All" / "Clear" per ISO group
   - Each plant shows: checkbox, name, baseline CO2 (MMt)
   - Checking a plant = "retrofit to CCS" → recalculates fan bands with that plant's emissions reduced
   - "CCS All" / "CCS None" bulk action buttons

4. SCENARIO TOGGLES:
   - Fuel price: Low / Medium / High
   - Other toggles as present in the controls bar
   - Changing a toggle re-filters the scenario set and recomputes P10/P50/P90

5. DELTA BAR CHART (below fan chart):
   - Horizontal bars showing CCS delta per selected plant
   - Sorted by delta magnitude (largest reduction first)

6. All charts use Chart.js (not Plotly) with responsive:true, maintainAspectRatio:false

Fix any issues where:
- Charts reference data keys that don't match emissions-dashboard-data.js
- Toggle event listeners aren't wired up
- Fan band recalculation doesn't trigger on plant selection change
- Headline stats don't update reactively

Test by opening the page and verifying all 4 headline cards show numbers, the fan chart renders, and toggling a plant CCS checkbox updates the visualization.
```

---

## Prompt 6: Add AT & SBTi Trajectory Targets

**Goal**: Ensure reference decarbonization trajectories are correctly computed and displayed.

**Context files to read first**:
- `scripts/constellation_dispatch_integrated.py:161-225` — trajectory computation in `run_dispatch_from_sim_results`
- `dashboard/emissions_dashboard.html` — where trajectories are rendered

**Prompt**:

```
Read scripts/constellation_dispatch_integrated.py lines 161-225 to understand how AT Power NZ and SBTi 1.5°C trajectories are computed.

Verify or implement:

1. AT POWER NZ TRAJECTORY:
   - Constellation's stated Power NZ target pathway
   - Should show year-by-year emissions reduction target from baseline to near-zero by 2050
   - Rendered as dashed red line on fan chart
   - Must use actual AT commitment milestones (not linear interpolation unless no milestones available)

2. SBTi 1.5°C TRAJECTORY:
   - Science Based Targets initiative 1.5°C-aligned pathway for power sector
   - ~4.2% annual linear reduction from baseline (SBTi power sector guidance)
   - Rendered as dashed amber line on fan chart

3. GAP ANALYSIS:
   - For headline card "Gap to AT Target 2050": compute P50_emissions_2050 - AT_target_2050
   - Positive = above target (red), Negative = below target (green)
   - Update dynamically when CCS plant selections change

4. DATA EXPORT:
   - Ensure generate_emissions_dashboard_data.py includes both trajectories in REFERENCE_TRAJECTORIES
   - Format: {at_power_nz: {years: [...], values_mmt: [...]}, sbti_15c: {years: [...], values_mmt: [...]}}

Both trajectories should be equity-adjusted (same basis as fleet emissions) for apples-to-apples comparison on the fan chart.
```

---

## Prompt 7: Validation & QA

**Goal**: End-to-end validation that the pipeline produces correct results and the dashboard renders properly.

**Prompt**:

```
Run the full validation sweep:

1. DATA INTEGRITY:
   - Verify fleet baseline CO2 = sum of (capacity_mw × CF × 8760 × co2_rate × equity_pct / 1e6) across all plants
   - Check P10 ≤ P50 ≤ P90 for every year (fan bands must never cross)
   - Verify CCS delta is always positive (CCS should always reduce emissions)
   - Check that CCS residual = baseline × (1.14 × 0.05 / CF_ratio) approximately
   - Verify total CCS reduction ≤ total baseline (can't abate more than you emit)

2. SCENARIO CONSISTENCY:
   - Higher fuel prices → higher avg_lmp → different dispatch → different emissions
   - Higher carbon prices → coal/oil retirement → lower fleet emissions
   - Faster clean penetration → lower fossil fraction → lower fleet emissions
   - P10-P90 spread should be meaningful (>10% of P50) — if too tight, scenario diversity is insufficient

3. DASHBOARD RENDERING:
   - All 4 headline cards show numeric values (not "—" or NaN)
   - Fan chart renders with visible P10-P90 shaded area
   - Both reference trajectories visible as dashed lines
   - Plant selector shows all 39 plants grouped by ISO
   - Selecting/deselecting CCS plants updates fan chart and headline stats
   - Delta bar chart updates when plant selection changes
   - Scenario toggles filter and recompute correctly

4. MOBILE COMPATIBILITY:
   - Charts readable at 375px viewport width
   - Plant selector scrollable, not overflowing
   - Toggle buttons have 44px min tap targets
   - No horizontal scroll on any viewport

5. CROSS-CHECKS:
   - Compare dashboard P50 baseline with constellation_dispatch_integrated.py standalone calculation
   - Verify emissions-dashboard-data.js file size is reasonable (<2MB)
   - Check for console errors in browser dev tools

Report pass/fail for each check with specific values where applicable.
```

---

## File Reference

| File | Role |
|------|------|
| `scripts/constellation_dispatch_integrated.py` | CCS dispatch engine (318 lines) |
| `scripts/pipeline_config.py` | All CCS constants (lines 415-850, 1256-1315) |
| `scripts/lmp_engine.py` | Merit-order fossil dispatch + LMP |
| `scripts/market_simulation.py` | Core simulation loop (3300 lines) |
| `scripts/fleet_model.py` | EIA-860/923 plant data (1257 lines) |
| `dashboard/emissions_dashboard.html` | **Target dashboard** — interactive explorer |
| `dashboard/js/emissions-dashboard-data.js` | Pre-computed data for dashboard |
| `frontend/emissions.html` | Simpler real-time CCS page (NOT this workflow's target) |
| `frontend/js/emissions.js` | Frontend CCS logic (real-time API calls) |

## CCS Constants Quick Reference

| Constant | Value | Source |
|----------|-------|--------|
| CO2 rate (CCGT) | 0.37 tCO2/MWh | EPA eGRID 2022 |
| CCS capture rate | 95% | pipeline_config.py |
| CCS heat rate penalty | 14% (×1.14) | pipeline_config.py |
| CCS capacity factor | 80% | pipeline_config.py |
| CCS residual rate | 0.037 tCO2/MWh | 10% × 0.37 |
| 45Q credit | $27.5/MWh | pipeline_config.py:840 |
| CCS LCOE (Med, 45Q on) | $86/MWh | pipeline_config.py:1258 |
| CCS LCOE (Med, 45Q off) | $113.5/MWh | pipeline_config.py:1264 |
| Regional CCS caps | 0.8–8.5 TWh/yr | pipeline_config.py:419-424 |
