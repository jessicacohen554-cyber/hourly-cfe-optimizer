##USER REVIEW & EXPECTATIONS 
#Issue -- fleet_scenario.html page is not functioning correctly… 

#Context / new changes: 
    - I have uploaded a new market-simulator/data/CEG_fleet_rosetta.csv which includes constellation owned capacity by plant. Plant naming convention may make cross referencing renewable assets difficult but that is OK. Fossil assets, especially with CAMPD data and facility IDs. This should serve as the primary source of truth on the CEG fleet. Capacity attributed to constellation in this dataset is based on equity share ownership, which is how emissions should be derived based on campd data and then for other fossil plants, derived from EIA 923 data x generic EPA emission rates for each fuel if 923 does not provide CO2 data
    - I have run a new set of 1215 sweep data available here market-simulator/data/results that will now include annual data instead of 5 year increments. It should be leveraged for this. this shoild be avaiable to read from when we are ready to execute prolpts. 
#Deliverable: ultimately I want this integrated into the market-simulator UI seamlessly but in the interim to test and troubleshoot, I want out to use the data / scripts / etc. to produce a publishable version of the fleet_scenario.html file available via the main dashboard site link, but NOT linked in Nav or searchable. 

#The vision: use market trajectory 1215 sweep data to understand probabilistic dispatch and generation stars for a baseline, p50 with confidence intervals case to understand emissions trajectory of the fleet without any intervention. Then enable the user to layer in decarbonization scenarios on top of that using CCs retrofitting on CCGT assets based on actual capacity, capacity derate, capacity factor, and carbon capture assumptions. Captured carbon is treated as reduced emissions. Retired plants would immediately drop to 0 emissions in the fleet scenario even if they were still economic under some scenarios of the market sweep. The vision is that the dashboard  shows what’s achievable with intervention on top of a baseline market trajectory. 

#Section 1 
- draft understanding of emissions calculations. First: i need to you to give me a step by step understanding of how you are calculating emissions from the constellation fleet in market-simulator/frontend/fleet-scenarios.html and based on my questions below, deduce ways to resolve the inconsistencies and series issues occurring with the dashboard. 

    1. What scripts and datasets does it rely on for upstream values? 
        1.1 Note the actual baseline case that displays on the fleet scenario UI should always be available to select on the fan chart and should be built purely on market trajectory 1215 sweep. 
    2. how are constellation fleet emissions derived from the market sweep results?
        2.1 is it plant specific dispatch And if so which data sources are you using to cross-reference constellation plants to dispatch data?
    3. what input data files are being used to identify and query the constellation fleet for the sidebar on the page? 
        3.1 are those aligned precisely to the same data files being used per my question in 2 re: derivation of fleet emissions from market sweep results?  
        3.2 Where do the capacity mw in the sidebar come from and why are so many 800? 
    4. How are you calculating emissions based on CCS retrofit and what variables are you using for derate, capture rate, and capacity factor? 
        4.1 right now I’m seeing huge emissions increases when I install CCS which doesn’t make sense… you’re not actually applying the capture as a reduction. 
        4.2 we should add a panel on top of fleet configuration sidebar that allows us to set capacity derate %, capacity factor % and carbon capture % for all CCS retrofit selections, which can default, reset in different scenario runs. 

#section 2
- Draft a plan to align to context /new changes and end-goal vision in terms of code / script changes, frontend uI updates, ensuring single source of truth across ALL scripts for CEG fleet assets and owner capacity: simulator/data/CEG_fleet_rosetta.csv. See above for other major edits. 

#section 3
- Draft a series of implementable prompts to achieve vision that can be leveraged in individual sessions to get to an intermediate state of dashboard/fleet_scenarios.html for live beta testing and edits 

## CLAUDE RESPONSE

### Vision Understanding
Publish `dashboard/fleet_scenarios.html` with CEG branding as an unlisted page (no nav link, `noindex` meta tag) for live beta testing. The page shows a probabilistic emissions fan chart (P10/P50/P90 from 1215 sweep) as the baseline market trajectory, with user-layered decarbonization scenarios (CCS retrofits with adjustable derate/capture, early retirements) overlaid via the sidebar editor. Retired plants drop to zero immediately. CCS reduces emissions via captured carbon. The dashboard tells the story: "here's what happens without intervention, and here's what's achievable with it."

---

### Section 1 Responses

#### 1. Scripts and datasets fleet-scenarios.html relies on

**Scripts (pipeline order):**
1. `market-simulator/scripts/build_fleet_scenario_data.py` — parses Rosetta CSV → generates `fleet_scenario_results_sample.json` with P10/P50/P90 envelopes
   - **PROBLEM:** Uses hardcoded fleet-level generation constants (e.g., `GAS_CCGT_GENERATION_TWH = 50.0`, line 57), NOT actual per-plant capacities from Rosetta
   - The ±12% spread is synthetic (line 299-300: `spread = 0.12`), NOT derived from the 1215 sweep
2. `market-simulator/frontend/js/fleet-dispatch-engine.js` — client-side dispatch engine that computes per-plant emissions across 1215 sweep scenarios. Used for sidebar recalculation ONLY
3. `market-simulator/frontend/js/fleet-sidebar.js` — loads fleet from `/api/fleet-scenarios-config` → fallback to `/api/fleet-config` → builds sidebar plant list
4. `market-simulator/frontend/js/fleet-scenarios.js` — chart rendering (fan, waterfall, gen mix, fuel emissions)

**Datasets:**
- `CEG_fleet_rosetta.csv` (141 plants, actual capacity, CCS eligibility, equity shares) — **intended** single source of truth, but currently underutilized
- `fleet_scenario_results_sample.json` — precomputed baseline scenarios (from hardcoded values, NOT from sweep)
- `sweep_dispatch_data.json` — 1215 scenarios × 6 years × 7 ISOs, per-fuel-type CF and margin arrays
- `market_simulation_results.json` — raw sweep output

**Per 1.1:** The baseline fan chart currently does NOT come from the 1215 sweep. It comes from `fleet_scenario_results_sample.json` which has ±12% synthetic spreads. **This is the core issue** — the "baseline" displayed and the "custom" computed by the dispatch engine use completely different calculation methodologies.

#### 2. How are Constellation fleet emissions derived from the market sweep?

**Current flow:** They are NOT properly derived from the sweep for the baseline:

- **Baseline fan:** `build_fleet_scenario_data.py` computes emissions using hardcoded fleet-level generation estimates (`GAS_CCGT_GENERATION_TWH = 50.0`) × emission rates (`EMISSION_RATE['gas_ccgt'] = 0.434 Mt/TWh`). Result: ~23.2 Mt CO₂ baseline. Fan spread is ±12% synthetic.
- **Custom scenarios (sidebar recalc):** `fleet-dispatch-engine.js` dispatches each plant individually against the 1215 sweep scenarios. Per plant: `genMwh = capMW × sweepCF × 8760`, `emisMt = genMwh × co2_rate / 1e6`. Real P10/P50/P90 from scenario distribution.

#### 2.1 Is it plant-specific dispatch?

**For custom scenarios (sidebar): YES** — `fleet-dispatch-engine.js` dispatches each plant individually. It reads the plant's ISO, looks up `sweep_dispatch_data.json[iso][fuel_cf][scenario][year]` to get capacity factor, applies efficiency ratio based on plant heat rate vs. reference, then computes generation and emissions.

**For baseline: NO** — uses fleet-aggregate hardcoded TWh values, not per-plant dispatch.

**Cross-reference:** Plants are mapped to ISOs via the `iso` field in the fleet config. The sweep data has per-ISO, per-fuel-type CF arrays. There is NO plant-specific dispatch (no CAMPD/ORISPL matching to individual unit dispatch) — it's fuel-type-level CF applied to each plant's capacity.

#### 3. What data files are used for the sidebar?

The sidebar (`fleet-sidebar.js` line 85-108) tries these sources in order:
1. `/api/fleet-scenarios-config` → `data.base_fleet[]`
2. Fallback: `/api/fleet-config` → maps to internal format
3. These API endpoints serve from `constellation_scenarios.json` or `constellation_fleet.json`

#### 3.1 Are sidebar data files aligned with emissions derivation?

**NO — this is the root cause of the CCS bug.** The baseline fan uses `fleet_scenario_results_sample.json` (hardcoded ~23.2 Mt). The sidebar recalculation uses `fleet-dispatch-engine.js` with sweep data. These produce DIFFERENT baseline emission levels. When a user applies CCS and recalculates, the custom scenario is compared against the wrong baseline.

#### 3.2 Where do the 800 MW values come from?

`build_fleet_scenario_data.py` line 193-206, function `estimate_plant_capacity()`:
```python
capacity_estimates = {
    'gas_ccgt': {'Combined Cycle': 800},  # hardcoded default
    'gas_ct': {'Combustion Turbine': 200, 'Gas Turbine': 150},
}
defaults = {'gas_ccgt': 800, 'gas_ct': 200, ...}
```

The Rosetta CSV HAS actual capacity in column `Constellation Owned Capacity (MW)` (e.g., Baytown = 852 MW, Colorado Bend II = 1,210 MW, Delta = 858 MW) but `parse_rosetta()` **never reads this column**. Every CCGT gets the 800 MW default.

#### 4. CCS emissions calculation

##### 4.1 Why CCS appears to increase emissions

The math in `fleet-dispatch-engine.js` (lines 347-355) is actually correct for per-plant CCS:
```javascript
effectiveCO2 = baseCO2 * hrPenalty * (1.0 - capture)
// = 0.37 * 1.14 * 0.05 = 0.021 t/MWh (with 95% capture)
genMwh = genMwh / hrPenalty  // reduces generation by 14%
```

**The bug is the baseline mismatch:** When a user selects CCS and clicks recalculate, the dispatch engine computes a FULL fleet dispatch using sweep data. This produces a baseline of ~30-40 Mt (depending on sweep scenarios), which is HIGHER than the displayed baseline fan of ~23.2 Mt (from hardcoded sample JSON). So even though CCS reduces emissions from the engine's perspective, the absolute value still appears higher than the artificial baseline on the chart.

**Root cause:** Two different calculation engines producing two different baselines. The displayed baseline and recalculated scenario are apples-to-oranges.

**Fix:** Make both baseline AND custom scenarios use the same dispatch engine with the same fleet data.

##### 4.2 CCS parameter panel (decided)

Add panel at top of fleet sidebar with adjustable sliders:
- **Capacity derate %** (0-30%, default 14%) — reduces net output capacity
- **Carbon capture rate %** (50-99%, default 90%) — fraction of CO₂ captured
- These apply as defaults to ALL plants set to "CCS Retrofit"
- Per-plant overrides still possible

---

### Section 2: Implementation Plan

#### Decisions Log
| Decision | Choice | Rationale |
|----------|--------|-----------|
| Fleet capacity source | `Constellation Owned Capacity (MW)` from Rosetta CSV | Actual equity-weighted capacity, not hardcoded estimates |
| CCS capacity handling | New sidebar derate slider | User wants adjustable param, not static from Rosetta `Available CCS Capacity` column |
| Baseline compute strategy | Precompute (Python) + live recalc (JS) | Fast initial render from precomputed JSON; dispatch engine for custom scenarios |
| Time resolution | Rebuild sweep_dispatch_data.json with annual years (2023-2050) | User has new annual sweep data |
| CCS parameters | Fully adjustable: derate % + capture rate % | Derate replaces heat rate penalty concept |
| Dashboard branding | CEG branding (CEG-style.css) | Keep market-simulator look, not dashboard shared.css |

#### Phase 0: Data Alignment (Foundation)
- **0a (unblocked):** Fix `parse_rosetta()` to read actual capacity from Rosetta CSV
- **0b (BLOCKED on 1215 sweep):** Rebuild `sweep_dispatch_data.json` with annual resolution

#### Phase 1: Fix Baseline Fan
- Rewrite `build_fleet_scenario_data.py` to dispatch all 1215 scenarios × years using actual per-plant capacity
- Output real P10/P25/P50/P75/P90 envelopes (replacing ±12% synthetic spread)

#### Phase 2: Fix CCS & Add Parameter Panel
- Add CCS derate % and capture rate % sliders to sidebar
- Update dispatch engine to use sidebar params instead of hardcoded values
- Verify CCS correctly reduces emissions

#### Phase 3: Publish to Dashboard
- Port to `dashboard/fleet_scenarios.html` with CEG branding
- Self-contained (no API dependencies), unlisted, `noindex`

#### Phase 4: QA & Verification
- Cross-check P50 against Constellation 10-K (~22 Mt Scope 1)
- Verify all 4 charts, mobile responsiveness, no console errors

---

### Section 3: Implementable Session Prompts

#### Prompt 1: Data Foundation (can start immediately — Phase 0a)
```
Read CEG_fleet_rosetta.csv and rebuild parse_rosetta() in
market-simulator/scripts/build_fleet_scenario_data.py to:
1. Read 'Constellation Owned Capacity (MW)' column for actual plant capacity
2. Read 'Available CCS Capacity' column (informational)
3. Remove estimate_plant_capacity() function and all hardcoded capacity defaults
4. Use actual capacity × fuel-specific CF for generation estimates
5. Run script and verify output matches expected fleet totals (~13 GW CCGT,
   ~22 GW total, ~190 TWh nuclear)

Also: rebuild sweep_dispatch_data.json with annual resolution (2023-2050) from
the 1215 sweep results at market-simulator/data/results/ (may be blocked if sweep
still running — do Rosetta fix first)
```

#### Prompt 2: Fix Baseline Fan (blocked on 1215 sweep completion)
```
Rewrite build_fleet_scenario_data.py to compute baseline P10/P50/P90 envelope
from 1215 sweep scenarios:
1. Load sweep_dispatch_data.json (annual resolution from Prompt 1)
2. Load fleet from Rosetta CSV with actual capacities
3. For each of 1215 scenarios × each year: dispatch each fossil plant using
   sweep CF[scenario][year] × actual capacity × equity share. Non-fossil uses
   static CF.
4. Sum fleet emissions per scenario-year. Compute P10/P25/P50/P75/P90
   percentiles across 1215 scenarios.
5. Output to fleet_scenario_results_sample.json with real envelopes
   (replacing synthetic ±12% spread)
6. Run and verify P50 baseline emissions are reasonable (~20-25 Mt CO₂)
```

#### Prompt 3: Fix CCS & Add Parameter Panel (can start after Prompt 2)
```
Fix CCS emissions bug and add global CCS parameter panel:
1. In fleet-sidebar.js: Add CCS parameter panel at top of sidebar with:
   - Capacity derate % slider (0-30%, default 14%)
   - Carbon capture rate % slider (50-99%, default 90%)
   - "Apply to All CCS Plants" button
2. In fleet-dispatch-engine.js: Accept derate % and capture rate from sidebar
   - Derate reduces net capacity: netCap = cap × (1 - derate/100)
   - Capture reduces CO₂: effectiveCO₂ = baseCO₂ × (1 - capture/100)
3. Verify: Toggle a plant to CCS → recalculate → emissions should decrease
4. Verify: Baseline fan and custom scenario start from same 2023 value
```

#### Prompt 4: Publish to Dashboard (can start after Prompt 3)
```
Create dashboard/fleet_scenarios.html as a publishable beta page:
1. Port fleet-scenarios.html from market-simulator/frontend/ to dashboard/
2. Keep CEG branding (CEG-style.css, simulator.css) — copy to dashboard/styles/
3. Bundle data as inline JS const or copy to dashboard/data/
4. Add <meta name="robots" content="noindex, nofollow">
5. Do NOT add to nav.js navigation
6. Verify page loads at /dashboard/fleet_scenarios.html with all 4 charts
7. Verify sidebar fleet editor works with recalculate
8. Test mobile responsiveness
```

#### Prompt 5: End-to-End Verification (final)
```
Full QA sweep on dashboard/fleet_scenarios.html:
1. Verify baseline fan P10/P50/P90 values are from 1215 sweep (not ±12%)
2. Verify sidebar shows actual plant capacities from Rosetta (not 800 MW)
3. Verify CCS retrofit reduces emissions on fan chart
4. Verify CCS parameter sliders work (derate + capture rate)
5. Verify all 4 charts render correctly
6. Verify target overlays (SBTi 1.5°C, AT Power NZ) display
7. Mobile test at 320px, 375px, 768px
8. No console errors
```

---

### Critical Files Reference

| File | Role | Modification |
|------|------|-------------|
| `market-simulator/data/CEG_fleet_rosetta.csv` | Fleet source of truth | READ ONLY |
| `market-simulator/scripts/build_fleet_scenario_data.py` | Baseline builder | Major rewrite |
| `market-simulator/frontend/data/sweep_dispatch_data.json` | Sweep dispatch data | Rebuild (annual) |
| `market-simulator/frontend/data/fleet_scenario_results_sample.json` | Precomputed baseline | Regenerated |
| `market-simulator/frontend/js/fleet-dispatch-engine.js` | Client dispatch engine | CCS param changes |
| `market-simulator/frontend/js/fleet-sidebar.js` | Fleet editor UI | Add CCS panel |
| `market-simulator/frontend/js/fleet-scenarios.js` | Chart rendering | Pass CCS params |
| `market-simulator/frontend/fleet-scenarios.html` | Source page | Reference |
| `dashboard/fleet_scenarios.html` | Published beta page | NEW |

### Verification Criteria
1. **Baseline accuracy:** P50 fleet emissions ~20-25 Mt CO₂ in 2023 (cross-check: Constellation 10-K ~22 Mt Scope 1)
2. **CCS reduction:** CCS on all eligible CCGTs with 90% capture → ~40-60% reduction from fossil baseline
3. **Capacity alignment:** Sum of sidebar CCGT capacities matches Rosetta totals (~13 GW)
4. **Fan spread:** P10-P90 driven by actual market scenario variation (not flat ±12%)
5. **Dashboard page:** Loads without backend API, all charts render, sidebar works

