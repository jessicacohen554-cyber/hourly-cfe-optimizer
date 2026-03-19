# Fleet Scenario Playbook: Parametric Sweeps → Qualified Targets → CCS Retrofit Analysis

> **Purpose**: Step-by-step guide for using the market-simulator to produce absolute emissions reduction targets with fleet-level dispatch, then layering CCS retrofit / new gas / retirement scenarios to generate P10/P50/P90 probability envelopes.

> **Phased approach** — each phase is self-contained and builds on the prior. No need to do everything at once.

---

## Architecture Decision: Front-End Fleet Config vs. Post-Hoc Module

**Recommendation: Hybrid — configure fleet up front, analyze CCS/retirement as post-processing.**

The market-simulator already has two relevant systems:

| Capability | Already Built? | Where |
|------------|---------------|-------|
| 1,215-scenario parametric sweep | Yes | `market_simulation.py:build_market_scenarios()` |
| Per-ISO annual dispatch + emissions | Yes | `market_simulation.py:run_market_simulation()` |
| Plant-level fleet config (toggle on/off) | Yes (frontend) | `frontend/fleet-config.html` + `backend/main.py` |
| Generator-level economics (margin, CF, retirement) | Yes | Output `generator_economics` dict per year |
| CCS breakeven carbon price | Yes | `market_simulation.py:1433-1480` |
| CCS as a deployable resource in merit-order | Yes | LCOE tables, queue caps, 45Q toggle |
| **Per-plant CCS retrofit toggle** | **Partial** | Emissions dashboard does client-side delta — not fed back into dispatch |
| **"With new gas" fleet variant** | **Partial** | Emissions dashboard has static overlay — not integrated into dispatch |
| **P10/P50/P90 envelope from sweep** | **Yes** | `constellation_dispatch.py:265-285` |

**What's missing**: A module that takes the 1,215 sweep results, lets you define fleet scenarios (CCS on plant X, retire plant Y, add 600 MW new gas in PJM), recomputes emissions through the dispatch model, and produces probability envelopes for each fleet scenario side by side.

The market-simulator's dispatch engine already handles CCS deployment at the grid level. What you want is *company-fleet-level* CCS/retirement/new-gas toggling applied *on top of* grid-level market trajectories. That's an overlay module, not a rewrite.

---

## Phase 1: Run the 1,215-Scenario Parametric Sweep

**Goal**: Generate the base market trajectories across all 7 ISOs.

### Prompt 1.1 — Verify sweep configuration

```
Review market_simulation.py:build_market_scenarios() and confirm:
1. How many scenarios does it produce? (Expected: 1,215 = 3×5×3×3×3×3)
2. What are the swept dimensions and their levels?
3. What years are simulated? (Expected: 2023, 2030, 2035, 2040, 2045, 2050)
4. What's the output schema per scenario/ISO/year?

Don't change anything — just confirm the configuration matches expectations
and flag any gaps.
```

### Prompt 1.2 — Run the sweep (all ISOs)

```
Run the full 1,215-scenario parametric sweep across all 7 ISOs
(CAISO, ERCOT, PJM, NYISO, NEISO, MISO, SPP).

Use market_simulation.py's batch mode. Save results to
market-simulator/results/sweep_1215/.

For each ISO × scenario × year, I need:
- clean_pct, demand_twh, emissions_mt, emission_rate_tco2_mwh
- avg_lmp, lmp_p90
- resource_mix_twh (full breakdown)
- generator_economics (per fuel type: capacity_mw, cf, margin_mwh)
- economic_retirements_mw
- emissions_by_fuel (coal, gas_ccgt, gas_ct, oil)
- nuclear_revenue breakdown
- ccs_breakeven

Save as both JSON and parquet for downstream consumption.
```

### Prompt 1.3 — Validate sweep results

```
Load the sweep results from market-simulator/results/sweep_1215/ and run
validation checks:

1. All 1,215 scenarios × 7 ISOs × 6 years = 51,030 year-results present?
2. Emissions monotonically decrease within each scenario as clean_pct rises?
3. No negative emissions_mt or clean_pct > 100%?
4. Generator economics: coal CF declining over time in most scenarios?
5. LMP range reasonable? ($20-200/MWh, no $0 or $9999 outliers)

Report any anomalies. Don't fix anything yet — just flag issues.
```

---

## Phase 2: Extract Fleet-Level Dispatch & Emissions by Year

**Goal**: Map grid-level market outcomes to a specific company's fleet.

### Prompt 2.1 — Define fleet input format

```
Design a fleet input schema (JSON) that I can use to define any company's
fossil fleet. Each plant needs:

- orispl (EIA plant ID)
- name
- iso (which ISO the plant operates in)
- capacity_mw (owned/equity-adjusted)
- fuel_type (gas_ccgt, gas_ct, coal_steam, oil_ct)
- heat_rate_mmbtu_mwh (plant-specific)
- co2_rate_t_mwh (derived from heat rate × emission factor)
- equity_share (0-1, for partial ownership)
- status: "operating" | "retired" | "ccs_retrofit"
- ccs_capture_rate (default 0.95 if status=ccs_retrofit)
- ccs_heat_rate_penalty (default 1.14 if status=ccs_retrofit)

Write a sample fleet file for 5 plants across 2 ISOs so I can see the format.
Save as market-simulator/fleet_scenarios/sample_fleet.json.

Don't build the processing logic yet — just the schema.
```

### Prompt 2.2 — Build fleet dispatch mapper

```
Build a module market-simulator/scripts/fleet_dispatch.py that:

1. Loads a fleet JSON file (from Phase 2.1 schema)
2. Loads the 1,215-sweep results (from Phase 1)
3. For each scenario × year:
   a. Look up the ISO-level generator_economics for each plant's fuel_type
   b. Apply plant-specific heat rate to get plant-level:
      - capacity_factor (from fleet-avg CF, adjusted by heat rate efficiency bin)
      - generation_mwh = capacity_mw × CF × 8760
      - emissions_mt = generation_mwh × co2_rate_t_mwh / 1e6
   c. Apply economic retirement logic: if the plant's fuel type has negative
      margins in that scenario/year, retire it (zero gen, zero emissions)
   d. Sum across all plants → fleet_emissions_mt per scenario per year
4. Compute P10/P50/P90 across the 1,215 scenarios for each year
5. Output: fleet_results.json with per-year P10/P50/P90 emissions envelopes

Key design choice: Use the market-level generator_economics as the dispatch
signal, but apply plant-specific heat rates to differentiate efficient vs.
inefficient units within the same fuel type.

Vectorize all loops — no Python for-loops over the 1,215 scenarios.
```

### Prompt 2.3 — Validate fleet dispatch against emissions dashboard

```
Run fleet_dispatch.py with the Constellation fleet (39 plants from
dashboard/js/ccs-proximity-data.js) and compare the P10/P50/P90 envelopes
against the existing emissions-dashboard-data.js values.

They won't match exactly (different dispatch model), but they should be
in the same ballpark (within 20%). Flag any plants or years where the
divergence is large and explain why.
```

---

## Phase 3: Fleet Scenario Variants (CCS / Retirement / New Gas)

**Goal**: Define multiple fleet configurations and compare their emissions trajectories.

### Prompt 3.1 — Fleet scenario definition format

```
Extend the fleet JSON schema to support multiple named scenarios. Each
scenario is a copy of the base fleet with modifications:

{
  "base_fleet": [...],  // all plants in default "operating" state
  "scenarios": {
    "baseline": {
      "description": "Status quo — no CCS, no new gas, no early retirements",
      "modifications": []
    },
    "ccs_only": {
      "description": "CCS retrofit on selected plants, no new gas",
      "modifications": [
        {"orispl": 55XXX, "action": "ccs_retrofit", "year_online": 2030},
        {"orispl": 55YYY, "action": "ccs_retrofit", "year_online": 2032}
      ]
    },
    "ccs_plus_new_gas": {
      "description": "CCS on existing + add 1200 MW new CCGT in PJM",
      "modifications": [
        {"orispl": 55XXX, "action": "ccs_retrofit", "year_online": 2030},
        {"action": "add_plant", "name": "New PJM CCGT", "iso": "PJM",
         "capacity_mw": 600, "fuel_type": "gas_ccgt",
         "heat_rate_mmbtu_mwh": 6.3, "year_online": 2028}
      ]
    },
    "retire_coal_ccs_gas": {
      "description": "Retire all coal by 2030, CCS on gas fleet",
      "modifications": [
        {"fuel_type": "coal_steam", "action": "retire", "year_online": 2030},
        {"fuel_type": "gas_ccgt", "action": "ccs_retrofit", "year_online": 2035}
      ]
    }
  }
}

Save the schema and a sample file. Don't build processing yet.
```

### Prompt 3.2 — Fleet scenario runner

```
Extend fleet_dispatch.py to process multiple fleet scenarios:

1. Load the fleet scenario file (Phase 3.1 format)
2. For each named scenario:
   a. Apply modifications to the base fleet (CCS, retirement, new plants)
   b. Respect year_online — modifications only take effect in that year
   c. CCS ramp: 0% capture in year_online, 30% by +2yr, 70% by +5yr, 100% by +8yr
   d. Run the fleet dispatch mapper (Phase 2.2) with modified fleet
   e. Output P10/P50/P90 envelopes for this scenario
3. Combine all scenarios into a single output file:
   fleet_scenario_results.json with structure:
   {
     "scenario_name": {
       "description": "...",
       "envelope": {
         "2023": {"p10": X, "p50": Y, "p90": Z},
         "2030": {...}, ...
       },
       "plant_detail": {
         "2030": [
           {"name": "Plant A", "gen_mwh": ..., "emissions_mt": ..., "status": "ccs_retrofit"},
           ...
         ]
       }
     }
   }

Vectorize: process all 1,215 scenarios in numpy, not Python loops.
```

### Prompt 3.3 — Target overlay (SBTi / AT Power NZ)

```
Add target trajectory overlays to fleet_scenario_results.json:

1. SBTi 1.5°C pathway: -4.2% annual linear reduction from 2023 baseline
2. AT Power NZ (Absolute Target Power Net Zero): whatever the company's
   committed target is (make this configurable in the fleet JSON)
3. Custom target: allow user to specify arbitrary {year: emissions_mt} points

For each scenario, compute:
- gap_to_target_mt per year (positive = above target, negative = below)
- year_of_target_achievement (first year P50 crosses below target, if ever)
- probability_of_meeting_target per year (% of 1,215 scenarios below target)

Add these to the output JSON.
```

---

## Phase 4: Visualization — Probability Envelope Dashboard

**Goal**: Interactive HTML page showing side-by-side fleet scenario envelopes.

### Prompt 4.1 — Chart architecture

```
Design (don't build yet) the visualization for fleet scenario comparison.

Main chart: Emissions Fan Chart (like emissions_dashboard.html but better)
- X-axis: years (2023–2050)
- Y-axis: fleet emissions (Mt CO2)
- For each scenario: P10/P50/P90 band (shaded) + P50 line
- Target overlays: SBTi 1.5°C (dashed), AT Power NZ (solid), custom
- Color-code scenarios: baseline (gray), CCS-only (blue), CCS+gas (orange),
  retire+CCS (green)
- Toggle scenarios on/off via checkboxes
- Hover tooltip: exact P10/P50/P90 values + gap-to-target

Secondary charts (below main):
1. Plant-level waterfall: which plants contribute how much to the delta
   between baseline and selected scenario (at a chosen year)
2. Generation mix bar chart: fleet TWh by fuel type per year per scenario
3. Emissions by fuel: stacked area showing coal/gas/oil emissions over time

Controls:
- Scenario checkboxes (toggle visibility)
- Year slider (for waterfall chart)
- Target selector (SBTi / AT Power NZ / custom)

Write the design spec as a markdown section. Use the shared.css design system
from the main dashboard. Don't write any HTML/JS yet.
```

### Prompt 4.2 — Build the visualization page

```
Build market-simulator/frontend/fleet-scenarios.html using the design spec
from 4.1.

Requirements:
- Load fleet_scenario_results.json via fetch()
- Use Chart.js for all charts
- Use the market-simulator's existing chart-colors.js for resource/ISO colors
- Follow the shared CSS design system (cards, grids, toggles)
- Mobile responsive (320px+, 44px tap targets)
- P10/P50/P90 bands as filled areas with 20% opacity
- Smooth transitions when toggling scenarios on/off
- Waterfall chart updates when year slider changes

This page lives in the market-simulator frontend, not the main dashboard.
It consumes market-simulator sweep results, not the hourly-cfe-optimizer
pipeline outputs.
```

### Prompt 4.3 — Add interactive plant toggling

```
Add a sidebar or modal to fleet-scenarios.html that lets me:

1. See all plants in the fleet (grouped by ISO)
2. Toggle each plant between: Operating / Retired / CCS Retrofit
3. Set year_online for retirements and CCS retrofits
4. Add a new plant (name, ISO, capacity, heat rate, fuel type, year online)
5. Name and save the current configuration as a new scenario
6. Click "Recalculate" to rerun fleet_dispatch against the 1,215 sweep results
   and update all charts in real-time

The recalculation should happen client-side in JavaScript (the 1,215 sweep
results are pre-loaded — the per-plant math is simple enough for the browser).

No backend round-trip needed for the fleet overlay math. The heavy compute
(1,215 market scenarios) is already done.
```

---

## Phase 5: Batch Runner for Multi-Company Analysis

**Goal**: Run the same analysis for multiple companies / fleet configurations from a batch file.

### Prompt 5.1 — Batch input format

```
Design a batch input file (CSV or JSON) that lets me queue up multiple
fleet scenario runs:

Each row/entry:
- company_name
- fleet_file_path (path to fleet JSON from Phase 2.1)
- scenario_file_path (path to scenario definitions from Phase 3.1)
- target_type (sbti_15 | custom)
- target_params (if custom: {year: emissions} pairs)
- output_dir

Write a batch runner script that processes each entry sequentially,
generates fleet_scenario_results.json for each company, and produces
a summary CSV with:
- company | scenario | 2030_p50 | 2040_p50 | 2050_p50 | target_met_year | gap_2030 | gap_2050
```

### Prompt 5.2 — Multi-company comparison view

```
Build a comparison page that loads results from multiple batch runs and shows:

1. Table: company × scenario matrix with P50 emissions at 2030/2040/2050
2. Chart: overlay P50 lines from different companies (same scenario)
   or different scenarios (same company)
3. Heatmap: probability of meeting SBTi target by company × year

This is a management-level summary view — "which portfolio companies
are on track and which aren't?"
```

---

## Phase Summary

| Phase | What You Get | Depends On |
|-------|-------------|------------|
| **1** | 1,215-scenario sweep results (grid-level market trajectories) | Market-simulator QA complete |
| **2** | Fleet-level dispatch mapper → company emissions by year | Phase 1 results |
| **3** | Multi-scenario fleet configs (CCS/retire/new gas) with envelopes | Phase 2 module |
| **4** | Interactive visualization dashboard | Phase 3 results |
| **5** | Batch runner for multi-company portfolio analysis | Phase 3 module |

Each phase produces a usable artifact. You can stop after any phase and have something valuable.

---

## Key Design Decisions Still Open

1. **Heat rate differentiation**: Use plant-specific heat rates from EIA 860/923 (more accurate) or bin into efficient/avg/old categories (simpler, matches existing fleet_model.py)?

2. **CCS ramp schedule**: Linear ramp (0→30→70→100% over 8 years) or step function (0% until online, then 95% immediately)?

3. **New gas dispatch assumption**: New CCGTs dispatch at fleet-average CF for gas_ccgt in that market scenario, or at a fixed high CF (e.g., 70%) since they'd be the most efficient units?

4. **Client-side vs. server-side recalc**: Phase 4.3 proposes client-side recalculation. With 1,215 scenarios × 6 years × 40 plants, that's ~292K simple multiplications — fast in JS. But if you want more sophisticated dispatch (e.g., plant-specific retirement thresholds), it needs server-side. Which approach?

5. **Scope**: Is this Constellation-specific or truly generic (any company, any fleet)? The existing emissions dashboard is Constellation-specific. A generic tool needs the fleet JSON input; a Constellation-specific tool can hardcode the 39-plant fleet.
