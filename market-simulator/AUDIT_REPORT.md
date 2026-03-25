# Third-Party Code Audit: Market Simulator

**Date:** 2026-03-25
**Scope:** All code within `/market-simulator/` — 76 files, ~41,800 lines (27,578 Python + 14,230 frontend)
**Method:** Code-only review. No documentation (README, SPEC.md, CLAUDE.md) was consulted.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Core Simulation Engine](#3-core-simulation-engine)
4. [Data Acquisition & I/O Layer](#4-data-acquisition--io-layer)
5. [LMP & Pricing Models](#5-lmp--pricing-models)
6. [Fleet Modeling & Dispatch](#6-fleet-modeling--dispatch)
7. [Parametric Sweep & Sensitivity Analysis](#7-parametric-sweep--sensitivity-analysis)
8. [Backend API](#8-backend-api)
9. [Frontend & Visualization](#9-frontend--visualization)
10. [Test Suite Assessment](#10-test-suite-assessment)
11. [Archive & Post-Processing Tools](#11-archive--post-processing-tools)
12. [Potential Use Cases](#12-potential-use-cases)
13. [Strengths & Weaknesses](#13-strengths--weaknesses)
14. [Appendix: File Inventory](#14-appendix-file-inventory)

---

## 1. Executive Summary

This codebase is a **full-stack hourly electricity market simulator** for the 7 major US Independent System Operators (CAISO, ERCOT, PJM, NYISO, NEISO, MISO, SPP). It models:

- **Merit-order fossil dispatch** with 5 heat-rate efficiency tiers per fuel type
- **Zonal transmission-constrained LMP** via analytical solvers and LP optimization
- **Plant-level economic retirement** with reliability floor constraints
- **CCS retrofit economics** with ramp schedules and 45Q tax credit modeling
- **Wright's Law learning curves** for technology cost trajectories (2026–2050)
- **1,215-scenario parametric sweeps** across 6 independent dimensions
- **Client-side fleet recalculation** for interactive what-if analysis

The tool produces hourly (8,760-point) dispatch, pricing, and emissions results across 25-year trajectories under thousands of cost/policy scenarios. It is delivered as both a **web application** (FastAPI + browser) and a **native desktop application** (PyWebView + PyInstaller).

**Scale:** 7 ISOs × 1,215 scenarios × 28 years = 238,140 simulation points per sweep, each requiring 8,760-hour merit-order dispatch.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     MARKET SIMULATOR                         │
├──────────────┬──────────────┬───────────────┬───────────────┤
│  Data Layer  │  Sim Engine  │  Backend API  │   Frontend    │
├──────────────┼──────────────┼───────────────┼───────────────┤
│ eia_data_io  │ market_sim   │ FastAPI       │ setup.html    │
│ fleet_model  │ lmp_engine   │ (38 routes)   │ results.html  │
│ fuel_prices  │ dispatch_utl │ models.py     │ fleet_*.html  │
│ pipeline_cfg │ fleet_dispch │ desktop_app   │ methodology   │
│ sweep_params │ zonal_lmp    │ paths.py      │ Chart.js      │
│ step0_*      │ sensitivity  │ start.py      │ Plotly        │
└──────────────┴──────────────┴───────────────┴───────────────┘
```

### Dependency Graph (Core Modules)

```
pipeline_config.py          ← Single source of truth (constants, cost tables)
    ↓
dispatch_utils.py           ← Hourly dispatch reconstruction, Numba-JIT storage loops
    ↓
lmp_engine.py               ← Merit-order LMP, scarcity pricing, 3-layer price model
    ↓
market_simulation.py        ← Profit-driven deployment, Wright's Law, 1,215-scenario sweep
    ↓
fleet_dispatch.py           ← Plant-level emissions, CCS ramp, vectorized (N_scenarios × N_years)
    ↓
main.py (FastAPI)           ← REST API orchestrating all above
```

---

## 3. Core Simulation Engine

### 3.1 pipeline_config.py — Central Configuration (Single Source of Truth)

**What it does:** Centralizes every shared constant, cost table, and schema for the entire pipeline. All other modules import from here — no module defines its own cost assumptions.

**Key constants discovered:**

| Category | Values |
|----------|--------|
| ISOs | 7 regions (CAISO, ERCOT, PJM, NYISO, NEISO, MISO, SPP) |
| Regional demand (2025) | CAISO 224 TWh, ERCOT 488, PJM 843, NYISO 152, NEISO 115, MISO 660, SPP 296 |
| Storage types | Battery 4hr (85% RTE), Battery 8hr (85% RTE), LDES 100hr iron-air (50% RTE), H2 1000hr (35% RTE) |
| ORDC params | Per-ISO VOLL ($2K–$5K/MWh), knee MW, lambda decay, caps ($200–$500/MWh) |
| Capacity markets | $0 (ERCOT/SPP energy-only) to $120/kW-yr (PJM RPM) |
| CCS caps | Regional geologic storage limits (NYISO/NEISO = 0; ERCOT = 200 TWh) |
| Zones | 2–5 transmission zones per ISO with transfer limits and BA mappings |

**Notable algorithm:** `compute_capacity_price(iso, reserve_margin_pct, clean_pct)` — endogenous capacity market pricing combining scarcity multiplier and clean penetration degradation (sigmoid S-curve). This means capacity prices respond dynamically to grid conditions rather than being static inputs.

### 3.2 market_simulation.py — The Core Simulator

**What it does:** Profit-driven market simulator where clean energy deployment is an *output* (emerges from profitability), not a mandated target. Resources deploy only when NPV ≥ 0 and stop when profitable deployment is exhausted or queue caps are hit.

**Deployment order:** clean_firm (baseload) → solar/wind (capture-rate dependent) → battery/LDES → hydro

**Key mechanisms:**
- **Wright's Law integration:** Cumulative deployment feeds learning curves → lower costs → more deployment (virtuous cycle)
- **Interconnection queue caps:** Per-ISO, per-technology annual MW limits based on LBNL "Queued Up 2024" completion rates (e.g., CAISO Medium = 4.5 GW/yr total)
- **Fuel price time-series:** Supports year-varying fuel prices from EIA AEO 2025 projections
- **Demand response:** Per-ISO DR parameters (max GW, trigger price, participation rate)

**Scenario matrix:** 1,215 combinations = 3 (demand) × 5 (price sensitivity bundles) × 3 (PPA) × 3 (gas friction) × 3 (queue capacity) × 3 (new fossil cost). CAISO adds geothermal → 17,496.

### 3.3 dispatch_utils.py — Hourly Dispatch Reconstruction

**What it does:** Reconstructs 8,760-hour dispatch profiles using sequential merit-order logic with Numba-JIT accelerated storage loops.

**Dispatch order:** clean_firm → CCS (baseload) → hydro → offshore_wind → wind → solar → battery → LDES → H2

**Storage dispatch algorithms (all Numba @njit accelerated):**
- **Battery:** Daily rolling window charge/discharge (4hr or 8hr duration, 85% RTE)
- **LDES:** 7-day rolling window (100hr iron-air, 50% RTE) — multi-day bridging
- **H2:** 30-day rolling window (1000hr, 35% RTE) — seasonal storage, ≥95% clean only

**Other notable features:**
- DST-aware solar profile correction (vectorized UTC offset, no per-hour loop)
- Nuclear seasonal capacity factor derate (monthly, varies by ISO)
- Weather-year sensitivity (2021–2025 distinct demand/renewable patterns)
- Dispatch cache (NPZ v4 format) — stores per-archetype 8,760-hour profiles to avoid recomputation

---

## 4. Data Acquisition & I/O Layer

### 4.1 eia_data_io.py — Profile Loading

**What it does:** Reads EIA 930 consolidated data from parquet or JSON. Returns nested dict structures for demand (8,760 MW), generation (per-fuel 8,760 shapes), fossil mix (hourly coal/gas/oil shares), and interchange (net import MW).

**Search path priority:** `market-simulator/data/profiles/` → `market-simulator/data/eia-930/` → parent project `data/`

**Auto-generation:** If interchange profiles are missing, automatically generates synthetic profiles via `step0_fetch_interchange.py` on first load.

### 4.2 fleet_model.py — Real Generator Inventory

**What it does:** Loads EIA 860 (generator inventory), EIA 923 (generation/fuel consumption), and EPA CAMPD (hourly emissions) data to build generator-level merit-order stacks with heat-rate binning.

**Key mechanics:**
- Fuzzy column matching handles EIA's inconsistent naming across data vintages
- Heat rates revealed from EIA 923 (gen ÷ fuel_MMBtu) when design heat rate unavailable
- Sanity bounds: 5.0 ≤ heat_rate ≤ 20.0 MMBtu/MWh (rejects outliers)
- BA-to-ISO mapping: ~40 Balancing Authority → ISO mappings (AEP/AP/ATSI → PJM, NSP/GRE → MISO, etc.)
- Bins generators into 5 heat-rate tiers: very_low (most efficient) → very_high (least efficient)

**Default heat rates (MMBtu/MWh):** Coal 10.0, CCGT 7.0, CT 10.5, Oil CT 10.5

### 4.3 fuel_price_projections.py — EIA AEO 2025 Fuel Prices

**What it does:** Provides year-varying fuel price projections (2023–2050) with Low/Medium/High cases from EIA Annual Energy Outlook 2025.

**Interpolation:** Linear between adjacent years when requested year not in dataset.

**Fallback prices ($/MMBtu):**

| Fuel | Low | Medium | High |
|------|-----|--------|------|
| Coal | 2.00 | 2.25 | 2.50 |
| Gas | 2.00 | 3.50 | 6.00 |
| Oil | 8.00 | 10.50 | 13.00 |

### 4.4 step0_parse_aeo_fuel_prices.py — AEO CSV Parser

**What it does:** Parses EIA AEO 2025 Table 1 CSVs (Reference, Low Oil, High Oil cases). Extracts Henry Hub gas ($/MMBtu), delivered coal ($/MMBtu), WTI oil ($/barrel ÷ 5.8 → $/MMBtu).

**Notable:** Handles column offset issues where commas in fuel names (e.g., "Coal, Delivered") shift CSV columns.

### 4.5 step0_fetch_interchange.py — Synthetic Interchange Profiles

**What it does:** Generates synthetic inter-regional import/export profiles based on EIA-930 annual averages, using a layered model:
1. Seasonal variation (cosine wave, peak month varies by ISO)
2. Diurnal shape (importers peak at demand peak; exporters at off-peak)
3. Gaussian noise (±3%, seeded by ISO hash for reproducibility)

**Annual net imports (MW avg):** CAISO +5,500 (net importer), ERCOT −200 (slight exporter), PJM −2,000 (exporter)

### 4.6 generate_synthetic_profiles.py — Testing Fallback

**What it does:** Generates realistic hourly demand and generation shape profiles for all 7 ISOs. Used only when real EIA data is unavailable (testing, first-run, CI/CD).

**Profile shapes:**
- Solar: daytime bell curve (CF 0.22–0.27 depending on ISO)
- Wind: beta-distributed + seasonal×diurnal (CF 0.22–0.38)
- Hydro: spring snowmelt peak
- Clean firm: flat baseload with seasonal nuclear derate
- Demand: seasonal + daily components, weekend reduction (−5%), noise

### 4.7 sweep_params_io.py — CSV Parameter Templates

**What it does:** Loads parametric sweep variable space from user-editable CSV files instead of hardcoding. Supports 5 axes (demand_growth, gas_friction, PPA, queue, new_fossil_cost) × price sensitivity bundles.

**Scenario count formula:** demand_growth_levels × price_sensitivities × PPA × gas_friction × queue × new_fossil_cost = 1,215

### 4.8 generate_plant_heat_rates.py — Plant Efficiency Data

**What it does:** Computes plant-specific thermal efficiency from EIA 923 and EPA CAMPD, with synthetic defaults for unmeasured plants.

**Priority chain:** EIA 923 (most reliable) → EPA CAMPD (fills gaps) → technology defaults

---

## 5. LMP & Pricing Models

### 5.1 lmp_engine.py — Synthetic Hourly LMP

**What it does:** Computes synthetic hourly Locational Marginal Price from merit-order fossil dispatch. This is the economic backbone of the simulator — it determines which generators run, what they earn, and whether they survive.

**Three-layer price model:**

1. **Merit-order dispatch:** Marginal cost = (heat_rate × fuel_price + VOM + CO₂_rate × CO₂_price + NOx/SOx adjustments) × (1 + ISO-specific adder). Units sorted cheapest-first; LMP set by marginal unit.

2. **Demand-quantile pricing:** Congestion/tightness adder on high-demand hours; negative pricing on low-demand hours with must-run surplus.

3. **Scarcity pricing:** Exponential LOLP-based adder when reserves fall below threshold. Uses ORDC (Operating Reserve Demand Curve) calibrated per-ISO.

**Heat-rate tier distribution (5 tiers per fuel type):**

| Tier | CCGT HR (MMBtu/MWh) | Capacity Fraction |
|------|---------------------|-------------------|
| very_low | 6.2 | 15–20% |
| low | 6.8 | 20–25% |
| medium | 7.5 | 25–30% |
| high | 8.1 | 20–25% |
| very_high | 9.0 | 10–25% |

**Fossil retirement model:** Coal and oil fully retire at ~60% clean energy penetration; below that, linear pro-rata retirement. Gas fleet sized by RA (Resource Adequacy) floor: residual_peak_MW / GAF (Gas Availability Factor, 0.82–0.88 by ISO).

**Cost-based offer adders:** 10% for PJM/NYISO/NEISO/CAISO (Manual 15 cost-based offers), 7% MISO, 0% ERCOT/SPP (energy-only markets).

**Confidence degradation:** LMP confidence factor decreases at high VRE penetration:
- ≤60% VRE → 1.0 (fully calibrated)
- 60–75% → 0.8 (moderate extrapolation)
- 75–90% → 0.6 (significant extrapolation)
- \>90% → 0.4 (beyond model validity)

### 5.2 zonal_lmp.py — Transmission-Constrained Zonal Pricing

**What it does:** Decomposes ISO-level LMP into 2–5 zones with constrained transmission interfaces. Solves hourly market clearing via either analytical dispatch or LP optimization.

**Three solver tiers:**
1. **1-zone vectorized:** `np.searchsorted()` over merit stack — all 8,760 hours at once
2. **2–3 zone analytical:** Algebraic 3-regime solution (unconstrained, A→B at limit, B→A at limit)
3. **N-zone LP:** `scipy.optimize.linprog` (HiGHS solver, ~0.5ms per 5-zone problem)

**Process:**
1. Build per-zone merit-order stacks
2. Solve copper-plate (unconstrained) as baseline
3. Check if any transfer limits are violated
4. If congested: re-solve with constraints → different zonal LMPs
5. Add scarcity pricing layer on top

**Output:** (n_zones × 8,760) LMP matrix + zonal stats (avg, peak, off-peak, P10/P90, spread vs system)

---

## 6. Fleet Modeling & Dispatch

### 6.1 fleet_dispatch.py — Plant-Level Emissions

**What it does:** Maps grid-level sweep results to plant-level emissions using equity shares, heat rates, and CCS ramp schedules. Fully vectorized across (N_scenarios × N_years) — no Python for-loops.

**CCS ramp schedule:**
| Years after online | Capture rate |
|-------------------|-------------|
| 0 | 0% |
| +2 | 30% |
| +5 | 70% |
| +8 | 100% |

**Efficiency adjustment:** Plant dispatch = grid_CF × min(reference_HR / plant_HR, 1.5)

**Emission factors (t CO₂/MMBtu):** Gas 0.05306, Coal 0.09552, Oil 0.07396

### 6.2 generate_constellation_scenarios.py — Fleet Scenario Builder

**What it does:** Parses a fleet rosetta CSV (plant inventory with ownership, capacity, fuel type, CCS eligibility) and generates 4 policy scenarios:
1. **Baseline:** No changes
2. **CCS top emitters:** CCS retrofit on largest CCGTs
3. **CCS + new gas:** CCS retrofits plus new CCGT construction
4. **Retire peakers + CCS baseload:** Retire oil/gas CTs, add CCS to baseload CCGTs

**Plant deduplication:** Key = (name, ISO, fuel_type, unit_detail). Stable hash IDs (900000 + hash % 100000) for plants without CAMPD identifiers.

### 6.3 build_fleet_scenario_data.py — Fleet Trajectory Builder

**What it does:** Builds 2023–2050 generation/emissions trajectories for an IPP fleet across 4 scenarios with P10/P50/P90 uncertainty bands derived from the 1,215-scenario sweep.

**Data priority hierarchy:**
1. User override (plant_emissions_overrides.json)
2. Fleet rosetta actuals (2023/2024)
3. EPA CAMPD actuals (2025)
4. Projected: year_factor × sweep CF scaling × capacity factor estimate

**Sweep-based uncertainty:** Loads P10/P50/P90 capacity factors from sweep parquet, uses as relative multipliers around point estimates — replaces naive linear decay with actual market dynamics.

**CCS ramp:** Quadratic: (year − 2028)/5, capped at 1.0 by 2033; 95% capture rate.

### 6.4 validate_plant_retirement.py — Retirement Model Validation

**What it does:** Compares two retirement approaches:
- **Plant-level:** Individual unit economics; retires worst-performing first; 15% zonal reserve margin floor
- **Fleet-fraction:** Retires percentage of all units of a type regardless of efficiency; 5% type minimum reserve

Tests both against synthetic 15-plant fleet and live PJM data under Low/Medium/High LMP scenarios.

---

## 7. Parametric Sweep & Sensitivity Analysis

### 7.1 run_sweep_1215.py — Full Parametric Sweep

**What it does:** Executes the full 1,215-scenario sweep across all ISOs and years, flattens results to parquet, and computes percentile aggregates.

**Output:** `sweep_1215_flat.parquet` — 1 row per (scenario × ISO × year), ~77 columns covering:
- Resource mix (% by type)
- Generator economics (CF, margin, capacity by fuel)
- Retirements (by fuel and heat-rate tier)
- New builds (fossil + clean)
- Emissions (by fuel), LMP, scarcity hours, RPS compliance, nuclear metrics, CCS breakeven

**Expected row count:** 476,280 (1,215 scenarios × 7 ISOs × 28 years, 2023–2050)

**Supports sharding:** `--shard K/N` for parallel execution across multiple workers.

### 7.2 sensitivity_analysis.py — Morris Method + ANOVA

**What it does:** Post-processing module that decomposes sweep results via two methods:

1. **Morris method (μ*, σ):** Finds scenario pairs differing in only one dimension, computes elementary effects. μ* = mean absolute effect (importance); σ = std of signed effects (non-linearity/interaction indicator).

2. **ANOVA variance decomposition:** First-order Sobol index approximation — fraction of output variance explained by each of the 6 sweep dimensions.

**6 sweep dimensions analyzed:** demand_growth, price_sensitivity, PPA_level, gas_friction, queue_capacity, new_fossil_cost

**4 output metrics:** clean_pct, cost_per_mwh, emissions_mt, avg_lmp

### 7.3 extract_iso_sweep_data.py — Dashboard Data Export

**What it does:** Converts sweep parquet into browser-ready JavaScript with envelopes, sensitivity decomposition, asset archetypes, and supply stacks.

**Computed outputs:**
- P10/P25/P50/P75/P90 envelopes for 77 metrics per ISO/year
- Variance contribution per dimension/metric (ANOVA)
- Archetype P10/P50/P90 per price bundle and demand level
- Median supply stacks per price bundle/year
- Outcome driver enrichment (which dimensions drive extreme outcomes)
- Fleet plant mapping (maps individual plants to sweep-derived metrics)

### 7.4 run_cv_simulations.py — Cross-Validation Baselines

**What it does:** Runs two reference scenarios (zero carbon price vs. EPA $51/ton carbon) for CAISO, PJM, ERCOT to generate cached comparison baselines.

### 7.5 procurement_utils.py — Procurement Strategy Utilities

**What it does:** Cross-cutting utilities for 3 procurement strategy families (Consequential, Hourly Matching, Annual). Implements Wright's Law cost trajectories, SSS (State-Sponsored Supply) allocation, and 25-year SBTi timeline modeling.

**Wright's Law formula:** `cost = FOAK × (cumulative_GW / reference_GW) ^ (−log₂(1 − learning_rate))`

**SSS allocation:** Two-component model:
- Fixed fleet: Nuclear + contracted hydro (constant TWh, e.g., PJM 95 TWh from IL ZEC 50, NJ ZEC 15, PA nuclear 30)
- RPS component: Scales with demand growth; 40% of new RPS build is SSS, 60% merchant

**Nuclear policy rolloff:** ZEC/CMC expirations shift plants from SSS (free) to merchant (priced) pools on hardcoded dates.

---

## 8. Backend API

### 8.1 main.py — FastAPI Backend (2,768 lines, 38 routes)

**What it does:** REST API orchestrating all simulation modes: single trajectory, parametric sweep, sensitivity analysis, and correlated IEA scenario bundles.

**Key endpoints:**

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/isos` | ISO metadata (demand, grid mix, fossil MW) |
| GET | `/api/defaults/{iso}` | Per-ISO defaults (fuel prices, heat rates, VOM) |
| POST | `/api/simulate` | Run single trajectory (one or more years) |
| POST | `/api/sweep` | Enqueue parametric sweep (async, returns job_id) |
| POST | `/api/sensitivity` | Single-parameter sensitivity sweep |
| POST | `/api/correlated-scenarios` | Run IEA scenario bundle (STEPS, APS, NZE, etc.) |
| GET | `/api/sweep-cached/aggregates` | Pre-computed percentile bands from sweep |
| GET | `/api/sweep-cached/sensitivity` | Cached sensitivity decomposition |
| POST | `/api/constellation-dispatch` | Fleet-level CCS dispatch optimization |
| GET | `/api/data-status` | Data source quality (synthetic vs. real) |
| GET | `/api/health` | Heartbeat |

**Background processing:** Sweeps run async in ThreadPoolExecutor with progress tracking. Each scenario requires full 8,760-hour merit-order dispatch + LMP + VRE curtailment + economic retirement.

**Data quality tracking:** Each result includes `data_quality: {synthetic_backed: bool, missing_sources: [...]}` and IPM validation triggers that flag when results exceed model validity bounds.

**5 correlated IEA scenarios:** IEA_STEPS ($0 carbon), IEA_APS ($51 EPA SCC), IEA_NZE ($185 Rennert et al.), HIGH_FRICTION, RAPID_TRANSITION

### 8.2 models.py — Pydantic Schemas (711 lines)

**What it does:** Defines all request/response data contracts with comprehensive validation.

**Notable validation rules:**
- Carbon price: 0–$1,000/ton
- LCOE bounds: 0–$500/MWh
- Fuel price bounds: 0–$100/MMBtu
- Year range: 2020–2100, must be monotonically ascending
- ISO validation: case-insensitive matching against 7 valid ISOs

**Response model hierarchy:** SimulationResponse contains YearResult[] → each with resource_mix_twh, zone_details[], fuel_bin_table[], plant_retirements[], ipm_triggers[], data_quality, confidence level.

### 8.3 desktop_app.py — Native Desktop Launcher (418 lines)

**What it does:** Wraps the FastAPI server in a native desktop window using PyWebView. Supports PyInstaller frozen bundles and development mode.

**Startup sequence:**
1. Detect PyInstaller bundle vs. dev mode
2. First-run setup: copy bundled profiles/templates to writable directory
3. Check heat rate staleness (regenerate if EIA data newer than cached rates)
4. Find free localhost port via socket binding
5. Start uvicorn in daemon thread (not main thread — no signal handlers)
6. Poll `/api/isos` until server ready (30s timeout)
7. Navigate PyWebView window to server URL

### 8.4 paths.py — 4-Tier Path Resolution

**What it does:** Resolves data file paths with priority chain: (1) User `MARKET_SIM_DATA_DIR` → (2) Bundled `MARKET_SIM_BUNDLE_DIR` → (3) Dev local → (4) Parent project `../data/`

---

## 9. Frontend & Visualization

### 9.1 Overview

**5 HTML pages, 7 JavaScript modules, 4 CSS files** = 14,230 lines of frontend code.

All charts render client-side via **Plotly** (results page) and **Chart.js** (fleet scenarios). Fleet dispatch recalculation runs entirely in-browser via `FleetDispatchEngine` — no server round-trip for what-if analysis.

### 9.2 setup.html + setup.js — Simulation Configuration

**Input form with:**
- Mode toggle: Trajectory 5-yr, Trajectory Annual, Market Sweep
- ISO selector with data-tier indicator (real vs. synthetic)
- 8+ cost sensitivity toggles (Low/Medium/High)
- Carbon price input with preset buttons ($0, $51, $100, $185)
- Demand growth, nuclear retirement threshold, custom fleet upload
- Heat rate adjustment table
- Sweep cache status indicator (shows if pre-computed results exist)

**Validation:** All numeric inputs range-checked, ISO required, mode required. Submit disabled until valid.

### 9.3 results.html + results.js — Simulation Output

**13+ visualizations:**
1. LMP & Capacity Revenue (time series)
2. Zonal Price Spreads (bar + table)
3. Supply Stack (stacked area by fuel)
4. System Emissions by Year (stacked bar)
5. Fuel Source & Heat Rate Table
6. Merit-Order Stack (generators sorted by marginal cost, LMP line overlay)
7. What Gets Built (donut chart)
8. Resource Economics / Capture Rates
9. CCS Breakeven (carbon price crossover)
10. Clean Cost Ladder ($/MWh vs cumulative GW)
11. Gas Fleet Efficiency Shift
12. Nuclear Revenue Stack (energy + capacity + PTC)
13. Sensitivity Matrix (fuel × carbon heatmap)

**IPM triggers:** High/medium severity alerts (stranded asset risk, demand spike, fuel price shock) with dismissal functionality.

**Export:** PNG (html2canvas) + CSV download.

### 9.4 fleet_scenarios.html + fleet-scenarios.js + fleet-sidebar.js — Fleet Analysis

**Interactive fleet editing tool:**
- Left sidebar (30vw, collapsible): plant list grouped by fuel type, inline-editable cells (capacity, retirement year, heat rate, CO₂ rate)
- Add/remove plants, CCS retrofit controls
- Save up to 8 scenarios in localStorage (v2 schema)
- Color-coded scenario tabs with load/delete/rename

**8 chart types:** Capacity by fuel, generation by fuel, intensity trajectory, top 10 emitters, CCGT tier dispatch, resource economics, scenario comparison, CCS retrofit IRR.

### 9.5 fleet-dispatch-engine.js — Client-Side Dispatch (724 lines)

**What it does:** Browser-side port of Python fleet_dispatch.py. Computes merit-order dispatch, CCS ramp, emissions, and economics without server calls.

**Logic:** Sort fossil plants by marginal cost → dispatch cheapest until demand met → plants above LMP line retire → CCS ramp applied → storage charge/discharge at peaks.

### 9.6 chart-colors.js — Color System

Canonical resource and ISO color constants. All Chart.js/Plotly datasets reference these — no hardcoded hex values in chart code.

### 9.7 shared-header.js — Animated SVG Banner

Injects animated SVG waveform overlay (3 sine wave curves + 2 heartbeat/EKG lines) into page headers. 12–20s animation cycles.

### 9.8 CSS Architecture

- **shared.css (950 lines):** Global design system — CSS variables, typography, spacing, grids, cards, buttons, toggles, tables, responsive breakpoints (1024/900/768/480/375/320px)
- **simulator.css (512 lines):** Setup form styling with collapsible sections, sliders, presets
- **results.css (511 lines):** KPI grid, confidence badges, generator table, IPM triggers, narrative box
- **CEG-style.css (950 lines):** Legacy duplicate of shared.css

---

## 10. Test Suite Assessment

### 10.1 Coverage

**21 test files, ~7,177 lines** covering unit tests, integration tests, end-to-end validation, and calibration verification.

### 10.2 Test Categories

| Category | Files | What's Tested |
|----------|-------|---------------|
| **Solver validation** | test_analytical_vs_lp, test_lp_codispatch, test_lp_batch_sim | Analytical vs LP LMP within $5/MWh (90% hours); energy conservation; storage co-optimization |
| **E2E integration** | test_e2e_integration, test_e2e_differentiation | 476,280-row parquet validation; input responsiveness (gas price → LMP, carbon → emissions) |
| **Calibration** | validate_ordc_calibration, test_scarcity_fix | ORDC scarcity hours within ISO-specific bands (ERCOT: 100–400 expected); new-build fossil feedback |
| **Economics** | test_storage_deployment, test_plant_retirement | Storage arbitrage $5–$500/kW-yr; H2 blocked below 95% clean; 15% reserve margin floor |
| **Peer review** | test_r9_integration, test_r9_e2e_sweep, test_r9_qa_qc | R1–R10 peer review recommendations (Wright's Law, curtailment, basis differentials, confidence intervals) |
| **Sensitivity** | test_sensitivity_analysis | Morris μ*/σ validation; ANOVA variance fractions sum ≤1.0; known-linear/quadratic test cases |
| **Scenarios** | test_correlated_scenarios | 5 IEA scenarios; carbon price monotonicity; distinct parameter combos |
| **Retirement** | test_plant_retirement, validate_fleet_dispatch | Plant-level vs fleet-fraction comparison; nuclear contract protection; reliability floor |
| **Data paths** | test_data_path_resolution | 4-tier path priority; env var overrides |
| **Parity** | test_fleet_scenario_parity | Byte-identical JS/data between dashboard and market-simulator frontends |
| **Confidence** | test_lmp_confidence | VRE penetration → confidence degradation (1.0 → 0.8 → 0.6 → 0.4) |
| **Diagnostics** | test_ercot_diagnostic, test_targeted_validation | ERCOT high-growth LMP diagnosis; worst-case ERCOT/NEISO bounds |

### 10.3 Key Assertion Thresholds

| Metric | Threshold | Source |
|--------|-----------|--------|
| LMP range | $5–$500/MWh | test_e2e_integration |
| Analytical vs LP | Within $5/MWh, 90% of hours | test_analytical_vs_lp |
| Energy conservation | discharge ≤ charge × RTE (±1e-4) | test_lp_codispatch |
| Scarcity hours ceiling | <4,000 annual | test_e2e_integration |
| Storage arbitrage | $5–$500/kW-yr | test_storage_deployment |
| Reserve margin floor | ≥15% zonal | test_plant_retirement |
| Stranded plant threshold | margin < −$5/MWh | test_plant_retirement |
| H2 minimum clean % | ≥95% | test_storage_deployment |
| Sweep row count | 476,280 | test_e2e_integration |
| Variance fractions sum | ≤1.0 | test_sensitivity_analysis |

---

## 11. Archive & Post-Processing Tools

### 11.1 constellation_dispatch_integrated.py

Per-plant CCS retrofit analysis. Dispatches each plant through merit-order pricing, computes CO₂ baseline and CCS capture deltas. Unit commitment logic: 50% minimum generation, 4-hour minimum up time, 2-hour minimum down time.

### 11.2 backtest_trajectory.py

Validates simulator predictions against observed 2020–2024 data. Computes direction accuracy, magnitude accuracy (MAE), rank ordering (Kendall-tau), and trend accuracy (slope within ±25%).

Uses scarcity-mode-aware tolerance bands: ±$5/MWh for ORDC years, ±$8/MWh for demand-quantile years.

### 11.3 build_tx_fleet_json.py

Synthesizes ERCOT generator fleet from EIA 860/923. Classifies 20+ unit types, bins CCGT by heat rate, aggregates by owner (top 30). Outputs JavaScript module for client-side portfolio analysis.

### 11.4 export_sweep_dispatch_data.py

Converts per-ISO sweep parquets into browser-friendly JSON (~27 MB). Per-fuel and per-tier capacity factors and margins as (n_scenarios × n_years) matrices.

### 11.5 Frontend (emissions.html, ipp-report.html)

CCS impact dashboard and fleet intelligence dashboard. Session storage handoff pattern between pages. Plotly visualizations: fan charts, per-plant CCS delta bars, fleet summary tables.

---

## 12. Potential Use Cases

Based solely on the code's capabilities (not documentation claims), this tool could serve:

### 12.1 Corporate Clean Energy Procurement

The simulator models hourly clean energy matching, annual matching, and consequential matching strategies with Wright's Law cost trajectories. The SBTi milestone timeline (2025→0% to 2050→99.99%) and procurement strategy comparison (Consequential, Hourly, Annual — each with 3–4 variants) suggests this is designed for corporations setting science-based clean energy targets. The 1,215-scenario sweep provides uncertainty bands on procurement costs.

### 12.2 IPP / Generator Fleet Strategy

The fleet scenario tool allows an IPP to:
- Model plant-level economics under market scenarios (which plants are profitable, which strand)
- Evaluate CCS retrofit ROI with realistic ramp schedules
- Assess nuclear revenue risk (energy + capacity + PTC stacking)
- Compare fleet transformation pathways (retire peakers vs. CCS baseload vs. new builds)
- The Constellation Energy-specific data (fleet rosetta, CAMPD IDs, equity shares) suggests this was built for or with a specific IPP client.

### 12.3 Regional Grid Decarbonization Analysis

The 7-ISO coverage with region-specific parameters (hydro caps, CCS geologic limits, capacity market designs, fossil fleet composition) enables comparative analysis of decarbonization pathways across US electricity markets. The tool can answer: "What does it take to reach 90% clean in PJM vs. ERCOT, and what does it cost?"

### 12.4 Policy Impact Assessment

The tool models:
- Carbon price effects on dispatch order and plant economics ($0–$1,000/ton range)
- 45Q tax credit impact on CCS viability
- RPS target trajectories (state-specific 2025–2050)
- Nuclear ZEC/CMC policy expirations
- Capacity market design effects (energy-only vs. capacity payments)
- Interconnection queue constraints on deployment speed

### 12.5 Electricity Market Price Forecasting

The LMP engine produces synthetic hourly prices calibrated against ISO-specific market structures (ORDC for ERCOT, cost-based offer adders for PJM, etc.). With confidence degradation at high VRE penetration, this acknowledges model limitations while still providing forward-looking price signals for investment decisions.

### 12.6 Storage Technology Valuation

The multi-technology storage model (4hr battery, 8hr battery, 100hr LDES, 1000hr H2) with technology-specific dispatch windows and efficiency losses can evaluate which storage technologies are profitable under different grid conditions and cost trajectories.

### 12.7 Academic / Research

The Morris method sensitivity analysis, ANOVA variance decomposition, backtesting framework, and confidence interval methodology suggest this is intended to meet academic rigor standards. The tool could support peer-reviewed research on grid decarbonization economics.

---

## 13. Strengths & Weaknesses

### Strengths

1. **Physically grounded dispatch:** 8,760-hour sequential merit-order dispatch captures hourly dynamics (duck curves, scarcity events, seasonal patterns) that capacity-factor-based models miss.

2. **Configuration centralization:** `pipeline_config.py` as single source of truth prevents constant drift between modules. Any cost table change propagates automatically.

3. **Vectorized compute:** No Python for-loops over large arrays. Numba @njit for storage dispatch, NumPy broadcasting for fleet emissions, `np.searchsorted` for merit-order. Fallback to pure NumPy when Numba unavailable.

4. **Multi-tier solver strategy:** Analytical zonal LMP for simple topologies (10–100× faster), LP for complex — automatically selected based on zone count.

5. **Comprehensive uncertainty quantification:** 1,215 scenarios with P10/P50/P90 envelopes, Morris/ANOVA sensitivity decomposition, and explicit confidence degradation at high VRE penetration. The tool knows where its model is unreliable.

6. **Client-side recalculation:** Fleet dispatch engine ported to JavaScript enables real-time what-if analysis without server round-trips. This is a significant UX advantage.

7. **Test suite depth:** 21 test files covering solver accuracy, energy conservation, calibration, peer review recommendations, and cross-model parity. Key thresholds are explicitly validated.

8. **Backtesting framework:** Validates against 2020–2024 observed data with scarcity-mode-aware tolerance bands. This is rare in forward-looking energy models.

9. **Data quality transparency:** Every result includes synthetic_backed flags and IPM triggers. The tool explicitly warns when its predictions are less reliable.

10. **Desktop + web delivery:** PyWebView desktop app with PyInstaller support alongside standard web deployment. Lowers barrier for non-technical users.

### Weaknesses & Observations

1. **Stylized vs. real stacks:** The primary LMP engine uses aggregated 5-tier heat-rate distributions, not actual generator bids. Real plant-level stacks are available (`fleet_model.py`) but appear to be a secondary path. This limits accuracy for specific plant-level price predictions.

2. **Static demand shapes:** Demand profiles are annual shapes scaled by growth factors. No weather-correlated demand spikes (polar vortex, heat dome) beyond the 5-year weather sensitivity. Extreme events may be underrepresented.

3. **No transmission build-out:** Transmission is modeled as fixed zones with static transfer limits. No endogenous transmission expansion — the grid topology doesn't evolve with clean energy deployment.

4. **CCS availability assumption:** CCS modeled as flat baseload with deterministic ramp schedule. No outage modeling, no CO₂ storage rate constraints beyond regional caps.

5. **Single-year snapshot extrapolated:** The physics (dispatch profiles, capacity factors) appear to come from a 2025 snapshot year, extrapolated via growth factors. Multi-year weather variability is limited to 5 historical years (2021–2025).

6. **IPP-specific data coupling:** Constellation Energy fleet data (`CEG_fleet_rosetta.csv`, CAMPD IDs, equity shares) is embedded in the codebase. This creates maintenance overhead if the fleet changes and limits portability to other IPPs without data substitution.

7. **Confidence zone thresholds are coarse:** 4 discrete confidence levels (1.0/0.8/0.6/0.4) at fixed VRE penetration boundaries. A continuous degradation function might be more appropriate.

8. **Duplicate CSS:** `CEG-style.css` is identical to `shared.css` — dead code adding 950 lines of maintenance burden.

9. **Large client-side data:** `sweep_dispatch_data.json` at ~27 MB is significant for browser loading. Could benefit from lazy loading or compression.

10. **No demand-side flexibility beyond DR:** The model includes demand response (GW caps, trigger prices) but no load flexibility, electrification profiles, or behind-the-meter resources.

---

## 14. Appendix: File Inventory

### Python Scripts (30 files, 27,578 lines)

| File | Lines | Category |
|------|-------|----------|
| `scripts/pipeline_config.py` | ~800 | Configuration |
| `scripts/market_simulation.py` | ~2,500 | Core engine |
| `scripts/dispatch_utils.py` | ~1,200 | Dispatch reconstruction |
| `scripts/lmp_engine.py` | ~1,500 | LMP pricing |
| `scripts/fleet_dispatch.py` | ~600 | Plant-level emissions |
| `scripts/fleet_model.py` | ~800 | EIA/EPA data loading |
| `scripts/zonal_lmp.py` | ~1,065 | Transmission-constrained LMP |
| `scripts/fuel_price_projections.py` | ~200 | AEO fuel prices |
| `scripts/sensitivity_analysis.py` | ~500 | Morris + ANOVA |
| `scripts/scenario_common.py` | ~600 | Shared scenario utilities |
| `scripts/procurement_utils.py` | ~800 | Wright's Law, SSS, SBTi |
| `scripts/generate_constellation_scenarios.py` | ~500 | Fleet scenario builder |
| `scripts/generate_plant_heat_rates.py` | ~300 | Plant efficiency |
| `scripts/generate_synthetic_profiles.py` | ~400 | Testing fallback profiles |
| `scripts/eia_data_io.py` | ~200 | Parquet/JSON I/O |
| `scripts/sweep_params_io.py` | ~269 | CSV parameter templates |
| `scripts/run_sweep_1215.py` | ~300 | Sweep orchestration |
| `scripts/run_cv_simulations.py` | ~200 | Cross-validation |
| `scripts/extract_iso_sweep_data.py` | ~500 | Dashboard data export |
| `scripts/build_fleet_scenario_data.py` | ~1,571 | Fleet trajectory builder |
| `scripts/validate_plant_retirement.py` | ~400 | Retirement validation |
| `scripts/step0_fetch_interchange.py` | ~198 | Interchange profiles |
| `scripts/step0_parse_aeo_fuel_prices.py` | ~211 | AEO CSV parser |
| `backend/main.py` | ~2,768 | FastAPI backend |
| `backend/models.py` | ~711 | Pydantic schemas |
| `desktop_app.py` | ~418 | Desktop launcher |
| `paths.py` | ~155 | Path resolution |
| `app-startup/start.py` | ~103 | Dev launcher |

### Test Files (21 files, 7,177 lines)

| File | What it tests |
|------|---------------|
| `test_analytical_vs_lp.py` | Analytical vs LP solver accuracy |
| `test_lp_batch_sim.py` | LP co-dispatch on real EF data |
| `test_lp_codispatch.py` | Multi-storage LP optimization |
| `test_e2e_integration.py` | Full sweep parquet → API validation |
| `test_e2e_differentiation.py` | Input responsiveness (8 scenarios) |
| `test_correlated_scenarios.py` | 5 IEA scenario definitions |
| `test_data_path_resolution.py` | 4-tier path priority |
| `test_lmp_confidence.py` | VRE confidence degradation |
| `test_ercot_diagnostic.py` | ERCOT high-growth LMP diagnosis |
| `test_scarcity_fix.py` | New-build fossil feedback |
| `test_targeted_validation.py` | Worst-case ERCOT/NEISO bounds |
| `test_r9_integration.py` | Peer review R1–R10 integration |
| `test_r9_e2e_sweep.py` | Mini-sweep exercising all R recs |
| `test_r9_qa_qc.py` | Unit tests for R1–R5, R7–R8, R10 |
| `test_sensitivity_analysis.py` | Morris + ANOVA validation |
| `test_storage_deployment.py` | Storage economics & H2 threshold |
| `test_plant_retirement.py` | Retirement with reliability floor |
| `test_fleet_scenario_parity.py` | Dashboard/simulator file parity |
| `validate_ordc_calibration.py` | ORDC scarcity hour calibration |
| `validate_cross_model.py` | Cross-model consistency |
| `validate_fleet_dispatch.py` | Fleet vs emissions dashboard parity |

### Frontend (14 files, 14,230 lines)

| File | Lines | Purpose |
|------|-------|---------|
| `frontend/setup.html` | 911 | Configuration form |
| `frontend/results.html` | 416 | Simulation output |
| `frontend/fleet_scenarios.html` | 781 | Fleet analysis |
| `frontend/methodology.html` | 1,066 | Technical documentation |
| `frontend/guide.html` | 646 | User guide |
| `frontend/js/setup.js` | 872 | Form controller |
| `frontend/js/results.js` | 1,573 | Results visualization |
| `frontend/js/fleet-scenarios.js` | 2,292 | Fleet analysis controller |
| `frontend/js/fleet-dispatch-engine.js` | 724 | Client-side dispatch |
| `frontend/js/fleet-sidebar.js` | 1,580 | Sidebar UI controller |
| `frontend/js/chart-colors.js` | 261 | Color constants |
| `frontend/js/shared-header.js` | 185 | SVG banner |
| `frontend/styles/shared.css` | 950 | Design system |
| `frontend/styles/simulator.css` | 512 | Setup styling |
| `frontend/styles/results.css` | 511 | Results styling |
| `frontend/styles/CEG-style.css` | 950 | Legacy duplicate |

### Archive (7 files)

| File | Purpose |
|------|---------|
| `archive/scripts/constellation_dispatch_integrated.py` | Per-plant CCS analysis |
| `archive/scripts/backtest_trajectory.py` | 2020–2024 backtesting |
| `archive/scripts/build_tx_fleet_json.py` | ERCOT fleet inventory |
| `archive/scripts/export_sweep_dispatch_data.py` | Sweep → browser JSON |
| `archive/emissions.html` + `emissions.js` | CCS impact dashboard |
| `archive/ipp-report.html` + `ipp-report.js` | Fleet intelligence dashboard |

---

*End of audit report.*

---
