# Interactive Projection Module — Build Guide

## Overview

A locally containerized Docker app that lets users adjust forward-looking assumptions (carbon prices, demand growth, learning curves, policy scenarios) and see how optimal resource mixes and costs shift in real time. Built on top of the existing hourly CFE optimizer's computed results.

**Deployment**: `docker compose up -d` → accessible at `http://localhost:8000`
**No cloud services, no auth, no build toolchain required.**

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Browser (any)                   │
│         http://localhost:8000                     │
│                                                   │
│  ┌─────────────┐  ┌──────────┐  ┌─────────────┐ │
│  │ Parameter UI │→ │ API call │→ │ Plotly.js   │ │
│  │ (sliders,   │  │ /api/    │  │ charts      │ │
│  │  toggles)   │  │ project  │  │ (live       │ │
│  │             │  │          │  │  update)    │ │
│  └─────────────┘  └────┬─────┘  └─────────────┘ │
└─────────────────────────┼───────────────────────┘
                          │ JSON
┌─────────────────────────┼───────────────────────┐
│              Docker Container                    │
│                                                   │
│  ┌──────────────────────┴──────────────────────┐ │
│  │           FastAPI (uvicorn)                  │ │
│  │                                              │ │
│  │  /api/project    POST  run projection        │ │
│  │  /api/scenarios  GET   list presets           │ │
│  │  /api/baseline   GET   current 2025 snapshot │ │
│  │  /api/isos       GET   available ISOs        │ │
│  │  /*              GET   static frontend files  │ │
│  └──────────────────────┬──────────────────────┘ │
│                          │                        │
│  ┌──────────────────────┴──────────────────────┐ │
│  │         Projection Engine (Python)           │ │
│  │                                              │ │
│  │  • Loads 2025 baseline from parquets         │ │
│  │  • Applies user assumptions forward          │ │
│  │  • LCOE learning curves (Wright's Law)       │ │
│  │  • Carbon price → dispatch cost shifts       │ │
│  │  • Demand growth → capacity scaling          │ │
│  │  • Returns year-by-year resource mix + cost  │ │
│  └──────────────────────┬──────────────────────┘ │
│                          │                        │
│  ┌──────────────────────┴──────────────────────┐ │
│  │     Fleet Model + Screening Engine           │ │
│  │                                              │ │
│  │  • EIA 860 unit-level fleet data             │ │
│  │  • Configurable heat rate bins (1-7, def 5)  │ │
│  │  • Asset-class survival probabilities        │ │
│  │  • New gas build trigger detection           │ │
│  │  • Probabilistic scenario sweep              │ │
│  └──────────────────────┬──────────────────────┘ │
│                          │                        │
│  ┌──────────────────────┴──────────────────────┐ │
│  │     /app/data/ (volume-mounted parquets)     │ │
│  │                                              │ │
│  │  step2.2-cost/   cost optimization results   │ │
│  │  step2.1-ef/     efficient frontier mixes    │ │
│  │  step1-pfs/      physics feasible space      │ │
│  │  step4-analysis/ MAC, LMP, CO2 results       │ │
│  │  step5-wrights/  learning curve data          │ │
│  │  eia-860/        unit-level fleet data        │ │
│  └──────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────┘
```

### Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| **Frontend** | Vanilla HTML/CSS/JS | Reuses existing design system (`shared.css`, `chart-colors.js`) |
| **Charts** | Plotly.js | Interactive, hover tooltips, responsive |
| **Backend** | FastAPI + Uvicorn | Python 3.11, serves API + static files |
| **Compute** | NumPy + Numba | Vectorized projections, sub-second response |
| **Data** | Parquet (PyArrow) | Volume-mounted from host, read-only |
| **Container** | Docker + docker-compose | Single container, no orchestration needed |
| **Auth** | None | Internal network access only |

### File Structure

```
interactive-module/
├── docker-compose.yml
├── Dockerfile
├── backend/
│   ├── main.py              # FastAPI app
│   ├── projection_engine.py # Core simulation logic
│   ├── data_loader.py       # Parquet loading + caching
│   ├── fleet_model.py       # EIA 860 fleet binning + heat rate tranching
│   ├── screening.py         # Asset-class survival sweep + new-build trigger
│   ├── models.py            # Pydantic request/response schemas
│   └── config.py            # Constants, defaults, LCOE tables
├── frontend/
│   ├── index.html           # Main projection UI
│   ├── screening.html       # Asset screening tool (or tab within index)
│   ├── styles/
│   │   ├── shared.css       # Copied from dashboard design system
│   │   └── projection.css   # Page-specific styles
│   └── js/
│       ├── app.js           # Main UI controller
│       ├── screening.js     # Screening tool UI + charts
│       ├── charts.js        # Plotly chart builders
│       ├── api.js           # API client
│       └── chart-colors.js  # Copied from dashboard
└── data/                    # Volume-mounted at runtime (not in image)
```

---

## Model Tiers: Haiku vs Sonnet vs Opus

Use the cheapest model that can handle each step. Escalate only when needed.

### Haiku ($0.25/MTok in, $1.25/MTok out) — Boilerplate & Repetitive Tasks

| Task | Why Haiku |
|------|-----------|
| Dockerfile + docker-compose.yml | Standard boilerplate, well-documented patterns |
| FastAPI route stubs (`main.py` skeleton) | CRUD-level routing, no domain logic |
| Pydantic models (`models.py`) | Schema definitions from a spec — mechanical |
| HTML page structure + CSS layout | Using existing design system classes — copy/adapt |
| Plotly chart scaffolding (empty charts, axis config) | Plotly API is well-known, no custom math |
| API client (`api.js` — fetch wrappers) | Trivial async fetch calls |
| `config.py` constants file | Transcribing tables from SPEC.md — no reasoning |
| README / documentation | Straightforward writing |
| Bug fixes with clear error messages | Stack trace → fix pattern |
| EIA 860 data cleaning + column mapping | Mechanical CSV→parquet conversion |
| Heat rate binning config defaults | Transcribing bin boundaries from analysis |

### Sonnet ($3/MTok in, $15/MTok out) — Core Application Logic

| Task | Why Sonnet |
|------|-----------|
| `data_loader.py` — parquet loading + memory caching | Multi-file I/O with schema awareness, caching strategy |
| `projection_engine.py` — main simulation loop | Wright's Law curves, demand scaling, carbon cost integration — requires understanding the domain model but follows established formulas |
| `app.js` — UI state management + chart updates | Complex event wiring, debouncing, parameter binding — needs architectural judgment |
| `charts.js` — interactive Plotly charts with dynamic data | Multi-trace updates, animation transitions, responsive config — moderate complexity |
| Connecting frontend ↔ API ↔ engine end-to-end | Integration work across 3 layers, debugging data flow |
| Slider/toggle UI with live preview | UX interaction design, throttling, visual feedback |
| Responsive layout + mobile optimization | CSS media queries, touch targets, chart resizing |
| Error handling + loading states | User-facing edge cases, timeout handling |
| Docker networking + volume mount debugging | Environment-specific troubleshooting |
| `fleet_model.py` — EIA 860 loading + capacity-weighted binning | Multi-file I/O, numpy aggregation, caching — follows established patterns |
| `screening.py` — scenario sweep + survival classification | Cartesian product sweep, threshold classification, probability aggregation — domain-informed but formulaic |
| `screening.js` — Plotly heatmap, fan charts, cascade visualization | Complex multi-chart page with interactive controls — moderate complexity |
| Heat rate bins slider with live re-binning | UI ↔ API ↔ engine round-trip with debouncing |

### Opus ($15/MTok in, $75/MTok out) — Complex Domain Logic & Debugging

| Task | Why Opus |
|------|-----------|
| Merit-order dispatch re-ranking under carbon prices | Requires understanding fuel-switching elasticity, regional fossil stacks, and how carbon costs cascade through dispatch order |
| Porting/adapting existing numba kernels for real-time use | Vectorization strategy, memory layout, JIT warmup — subtle performance bugs |
| Multi-dimensional sensitivity analysis (carbon × demand × learning curves) | Combinatorial parameter space, ensuring results are physically consistent |
| Architectural decisions when things don't fit | "Should the projection engine re-optimize or interpolate?" — requires understanding the tradeoff |
| Debugging scientifically wrong results | Numbers come out but are wrong — requires domain knowledge to spot implausible resource mixes |
| Carbon price → LMP feedback loop | Synthetic LMP shifts with carbon pricing, second-order effects on storage value |
| Validating projection outputs against known benchmarks | Cross-referencing NREL ATB, Lazard, EIA AEO — needs energy domain expertise |
| Debugging implausible fleet retirement cascades | Must know expected coal→gas→CCS switching order and at which carbon prices |
| Validating bin dispatch ordering against EIA 860 fleet data | Heat rate distributions, vintage correlations, regional fleet composition |
| New-build trigger logic with ELCC interactions | Capacity adequacy math with mixed clean+fossil fleet — subtle edge cases |

### Decision Rule

```
Start with Haiku for scaffolding (Phase 0)
     ↓
Switch to Sonnet for all application logic (Phases 1-3)
     ↓
Escalate to Opus ONLY when:
  - Results look wrong and you can't figure out why
  - Porting dispatch/optimization math from existing scripts
  - Making architectural decisions about simulation fidelity
  - Debugging numba/vectorization performance issues
  - Fleet retirement cascade doesn't match expected physical behavior
  - New-build trigger logic produces implausible years
```

---

## Build Phases & Copilot Prompts

### Phase 0: Project Scaffold (Haiku)

**Prompt 1 — Docker + FastAPI skeleton:**

> Create a Dockerized FastAPI project for an internal energy market projection tool:
>
> Files to create:
> - `Dockerfile` — Python 3.11-slim, installs: fastapi, uvicorn, pyarrow, pandas, numpy, numba, plotly. Copies `backend/` and `frontend/`. CMD runs uvicorn on 0.0.0.0:8000.
> - `docker-compose.yml` — builds from Dockerfile, maps port 8000:8000, volume-mounts `./data:/app/data:ro` (read-only).
> - `backend/main.py` — FastAPI app with 4 route stubs:
>   - `GET /api/isos` → returns list of 7 ISOs (CAISO, ERCOT, PJM, NYISO, NEISO, MISO, SPP)
>   - `GET /api/baseline/{iso}` → returns 2025 baseline data (stub: return empty dict)
>   - `POST /api/project` → accepts projection parameters, returns results (stub: return empty dict)
>   - `GET /api/scenarios` → returns preset scenario list (stub: return empty list)
>   - Static file mount: serves `frontend/` directory at root `/`
> - `backend/models.py` — Pydantic models:
>   - `ProjectionRequest`: iso (str), target_year (int, 2025-2050), carbon_price_usd (float, 0-300), demand_growth_pct (float, 0-5), cfe_target_pct (float, 50-99.99), cost_scenario (str: low/medium/high), learning_rate (float, 0.1-0.3)
>   - `ProjectionResponse`: years (list[int]), resource_mix (dict[str, list[float]]), total_cost (list[float]), lcoe_trajectories (dict[str, list[float]]), carbon_abated (list[float])
>   - `ScreeningRequest`: iso (str), heat_rate_bins (int, 1-7, default 5), carbon_price_range (object: min/max/step, default 0-300 step 25), demand_growth_range (object: min/max/step, default 0-5 step 1), cfe_targets (list[float], default [75, 85, 90, 95]), learning_rate (float, default 0.18)
>   - `ScreeningResponse`: bins (list[BinResult]), new_build_trigger (dict[str, int|null]), scenario_count (int), metadata (dict)
>   - `BinResult`: bin_id (int), fuel_type (str), heat_rate_range (tuple), capacity_mw (float), avg_age_years (float), p_operating (float), p_marginal (float), p_stranded (float), median_retirement_carbon_price (float|null), capacity_factor_p10 (float), capacity_factor_p50 (float), capacity_factor_p90 (float)
> - `backend/config.py` — Constants: ISO list, default LCOE values (solar: 31, onshore wind: 26, offshore wind: 53, nuclear: 88, ccs: 73, battery_4hr: 8.4, battery_8hr: 15.2, ldes: 35, geothermal: 52 — all $/MWh), Wright's Law default learning rate 0.18. Default heat rate bins: 5 (calibrated from EIA 860 fleet distribution).
>
> Keep everything minimal — stubs only. No real logic yet.

**Prompt 2 — Frontend skeleton:**

> Create the frontend HTML page for the projection tool. Use this exact design system (it matches our existing dashboard):
>
> `frontend/index.html`:
> - Include Google Fonts (Plus Jakarta Sans + DM Sans)
> - Include Plotly.js from CDN
> - Page header with class "header": title "CFE Projection Explorer", subtitle "Forward-looking resource mix & cost projections"
> - Parameter panel (left sidebar on desktop, top on mobile):
>   - ISO selector: 7 buttons (CAISO, ERCOT, PJM, NYISO, NEISO, MISO, SPP), use `.iso-selector` + `.iso-btn` classes
>   - Target year slider: 2025-2050, step 5, default 2035
>   - CFE target slider: 50-99.99%, step 2.5, default 90
>   - Carbon price slider: $0-300/ton, step 10, default 0
>   - Demand growth slider: 0-5%/yr, step 0.5, default 2
>   - Cost scenario toggle: Low/Medium/High buttons
>   - Learning rate slider: 10-30%, step 2, default 18
>   - "Run Projection" button, primary styled
> - Chart area (right/main on desktop, below on mobile):
>   - 4 Plotly chart divs in a 2x2 grid:
>     1. "Resource Mix Over Time" (stacked area)
>     2. "LCOE Trajectories" (multi-line)
>     3. "Annual Cost" (bar + line overlay)
>     4. "Cumulative Carbon Abated" (area)
>   - Each in a `.chart-panel` container with `.chart-container-lg` sizing
> - Use CSS variables: `--font-heading`, `--font-body`, `--bg-page`, `--navy`, spacing vars
> - Mobile responsive: sidebar collapses to top stack below 768px
>
> `frontend/js/api.js`:
> - `fetchBaseline(iso)` → GET /api/baseline/{iso}
> - `runProjection(params)` → POST /api/project with JSON body
> - `fetchScenarios()` → GET /api/scenarios
> - All return parsed JSON, handle errors with console.error
>
> `frontend/js/app.js`:
> - On page load: select CAISO, fetch baseline, initialize empty charts
> - On ISO button click: update active state, fetch baseline
> - On "Run Projection" click: gather all params, call runProjection, update charts
> - Debounce slider changes (300ms) for auto-update option
>
> `frontend/js/charts.js`:
> - `initCharts()` — create 4 empty Plotly charts with proper layout config
> - `updateResourceMix(data)` — stacked area chart, one trace per resource
> - `updateLCOE(data)` — line chart per technology
> - `updateCost(data)` — bar chart for annual, line overlay for cumulative
> - `updateCarbon(data)` — filled area chart
> - Use these colors (match our existing system): Solar #F59E0B, Wind #22C55E, Hydro #0EA5E9, Nuclear #6366F1, CCS #64748B, Battery #06B6D4, LDES #E91E63, Geothermal #D97706
> - All charts: responsive, white background, Plus Jakarta Sans font

---

### Phase 1: Data Layer (Sonnet)

**Prompt 3 — Parquet loader:**

> Implement `backend/data_loader.py` to load our optimizer's precomputed results:
>
> The data lives in `/app/data/` (Docker volume mount). Key files:
> - `data/step2.2-cost/` — parquet files per ISO with columns: threshold, solar_pct, wind_pct, clean_firm_pct, hydro_pct, battery_4hr_mw, battery_8hr_mw, ldes_mw, h2_mw, total_cost_mwh, lcoe_by_resource (nested), scenario fields (renewable_cost, firm_cost, storage_cost, fossil_cost, transmission_cost, ccs_level, fortyfiveq)
> - `data/step2.1-ef/` — efficient frontier parquets per ISO with physics-optimal resource mixes
> - `data/step5-wrights/` — Wright's Law learning curve parameters
>
> Implement:
> - `DataLoader` class with lazy loading + in-memory LRU cache
> - `get_baseline(iso, threshold, scenario)` → returns the 2025 optimized mix for given params
> - `get_efficient_frontier(iso)` → returns all non-dominated mixes
> - `get_cost_scenarios(iso, threshold)` → returns low/medium/high cost envelopes
> - Load parquets with pyarrow, convert to pandas DataFrames, cache in memory
> - Singleton pattern — one DataLoader instance shared across requests
>
> Handle missing files gracefully (log warning, return empty DataFrame). The container starts once and serves many requests — optimize for repeated reads.

**Prompt 4 — Wire up baseline endpoint:**

> Connect the `GET /api/baseline/{iso}` endpoint in `main.py` to the DataLoader:
>
> - Accept query params: `threshold` (float, default 90), `scenario` (str, default "medium")
> - Call `data_loader.get_baseline(iso, threshold, scenario)`
> - Return JSON with: resource_mix (dict of resource → percentage), total_cost_mwh (float), lcoe_breakdown (dict of resource → $/MWh), threshold (float), scenario (str)
> - If ISO not found or data missing, return 404 with descriptive error
> - Add startup event that pre-loads data for all 7 ISOs at medium scenario to warm the cache

---

### Phase 1.5: Fleet Data Layer (Sonnet)

**Prompt 4.5 — EIA 860 fleet loader + heat rate binning:**

> Implement `backend/fleet_model.py` — loads EIA 860 unit-level generator data and bins the fossil fleet by heat rate for carbon price sensitivity analysis.
>
> **Data source**: EIA Form 860 (annual generator inventory) provides per-unit: plant_id, generator_id, capacity_mw, heat_rate_mmbtu_mwh, fuel_type (coal, gas_cc, gas_ct, oil_ct), online_year, planned_retirement_year, ISO/balancing_authority. Data files in `/app/data/eia-860/`.
>
> **Core class: `FleetModel`**
>
> Constructor: `FleetModel(iso: str, n_bins: int = 5)`
> - `n_bins` is user-configurable from 1 to 7. Default 5.
> - 1 bin = aggregate (current model behavior, single heat rate per fuel type)
> - 7 bins = maximum granularity (captures full fleet heterogeneity)
>
> **Binning logic**:
> - Load all generators for the given ISO from EIA 860 parquet
> - Separate by fuel type: gas_cc (combined cycle), gas_ct (combustion turbine), coal, oil
> - For each fuel type, sort units by heat rate ascending
> - Create N equal-**capacity** bins (not equal-count) — each bin holds ~1/N of total MW
> - Per bin, compute: weighted_avg_heat_rate, total_capacity_mw, capacity_weighted_avg_age, emission_rate_tco2_mwh (= heat_rate × fuel_emission_factor)
>
> **Default 5-bin structure for gas CC** (illustrative, actual breakpoints from EIA 860 data):
>
> | Bin | Label | Heat Rate Range | Typical Vintage | Example |
> |-----|-------|----------------|-----------------|---------|
> | 1 | Efficient (H-class) | <6.5 MMBtu/MWh | 2018+ | New 7HA.03 units |
> | 2 | Above-average | 6.5–7.0 | 2010-2017 | F/H-class mid-life |
> | 3 | Average | 7.0–7.5 | 2003-2009 | F-class fleet bulk |
> | 4 | Below-average | 7.5–8.5 | 1995-2002 | Early F-class, E-class |
> | 5 | Inefficient | >8.5 | Pre-1995 | Old steam-to-CC conversions |
>
> **Methods**:
> - `get_bins(fuel_type: str) → list[FleetBin]` — returns bins for given fuel
> - `get_merit_order(carbon_price: float) → list[FleetBin]` — all bins sorted by variable cost (fuel + carbon + VOM) ascending
> - `get_bin_marginal_cost(bin: FleetBin, fuel_price: float, carbon_price: float) → float` — per-bin MC
> - `rebin(n_bins: int)` — re-create bins with different granularity (live slider support)
>
> **Emission factors** (tCO2 per MMBtu):
> - Natural gas: 0.0531
> - Coal (bituminous): 0.0934
> - Oil (distillate): 0.0732
>
> So emission_rate_tco2_mwh = heat_rate_mmbtu_mwh × emission_factor_tco2_mmbtu.
>
> **Important**: The binning must be per-ISO, because fleet composition varies dramatically:
> - ERCOT: gas-heavy, newer fleet (lower avg heat rate)
> - PJM/MISO: mixed coal+gas, older fleet (higher avg heat rate)
> - CAISO: almost no coal, gas-only with some old steam units
> - NYISO/NEISO: gas-dominated, dual-fuel oil backup
>
> Use numpy for all aggregations. Cache the binned fleet on first load — rebinning is a lightweight array operation, not a re-read.

---

### Phase 2: Projection Engine (Sonnet, escalate to Opus if needed)

**Prompt 5 — Core projection logic:**

> Implement `backend/projection_engine.py` — the forward projection model:
>
> **Inputs** (from ProjectionRequest):
> - `baseline`: 2025 optimized resource mix from DataLoader
> - `target_year`: project to this year (2025-2050)
> - `carbon_price_usd`: $/ton CO2 (affects fossil dispatch cost, shifts optimal mix)
> - `demand_growth_pct`: annual demand growth rate
> - `cfe_target_pct`: target CFE matching percentage
> - `learning_rate`: Wright's Law learning rate for cost reduction
>
> **Projection logic** (year-by-year from 2025 to target_year):
>
> 1. **LCOE evolution**: Apply Wright's Law to each technology:
>    `lcoe(t) = lcoe_2025 * (cumulative_capacity(t) / cumulative_capacity(2025)) ^ (-learning_rate / ln(2))`
>    Starting LCOEs from config.py. Cumulative capacity doubles approximately:
>    - Solar: every 3 years
>    - Wind: every 4 years
>    - Battery: every 3 years
>    - Nuclear/CCS: every 10 years
>    - LDES: every 5 years
>
> 2. **Demand scaling**: `demand(t) = demand_2025 * (1 + growth_rate)^(t - 2025)`
>
> 3. **Carbon cost integration** (heat-rate-aware, uses `fleet_model.py`):
>    - Load fleet bins from `FleetModel(iso, n_bins=request.heat_rate_bins)`
>    - For each bin: `carbon_adder = carbon_price × bin.emission_rate_tco2_mwh`
>    - Variable cost per bin: `fuel_price × bin.heat_rate / 1000 + VOM + carbon_adder`
>    - **Key behavior**: At $50/ton carbon, efficient H-class CCGT (6.2 HR) pays $16.5/MWh carbon cost; inefficient old CCGT (8.5 HR) pays $22.6/MWh — a $6/MWh spread that determines who operates and who doesn't
>    - Bins whose variable cost exceeds the clean energy clearing price are dispatched less or retired
>    - This is what makes the model sensitive enough to differentiate old vs. new CCGTs
>
> 4. **Resource mix re-optimization**: For each projected year, find the cost-minimizing mix at the given CFE target using the updated LCOEs. Use the efficient frontier mixes as candidates — re-score them with projected costs and pick the cheapest that meets the target. This is a lookup + re-rank, NOT a full physics re-run.
>
> 5. **Carbon abatement**: Calculate displaced fossil generation and CO2 avoided vs. a no-policy baseline.
>
> **Output**: ProjectionResponse with year-by-year arrays for resource_mix, total_cost, lcoe_trajectories, carbon_abated.
>
> Use numpy vectorized operations throughout — no Python for-loops over data arrays. The engine must respond in <500ms for a 25-year projection.

**Prompt 6 — Wire up projection endpoint:**

> Connect `POST /api/project` to the projection engine:
>
> - Validate ProjectionRequest with Pydantic
> - Load baseline from DataLoader
> - Load efficient frontier mixes for the ISO
> - Run projection_engine.project(baseline, ef_mixes, request_params)
> - Return ProjectionResponse as JSON
> - Add request timing header (X-Compute-Time-Ms)
> - If projection takes >2s, log a warning (indicates vectorization issue)

---

### Phase 3: Interactive Frontend (Sonnet)

**Prompt 7 — Live chart updates:**

> Update the frontend to display real projection results:
>
> `frontend/js/charts.js` — implement all 4 chart update functions with Plotly:
>
> 1. **Resource Mix Over Time** (stacked area):
>    - X axis: years (2025 → target)
>    - Y axis: % of generation
>    - One trace per resource, stacked to 100%
>    - Use resource colors from chart-colors.js
>    - Hover: show resource name, %, and absolute GWh
>
> 2. **LCOE Trajectories** (multi-line):
>    - X axis: years
>    - Y axis: $/MWh
>    - One line per technology, dashed for projected vs. solid for historical
>    - Annotations: mark where solar crosses nuclear, where storage crosses gas peaker
>    - Add shaded band for current fossil cost range
>
> 3. **Annual System Cost** (bar + line):
>    - Bars: annual $/MWh by component (stacked: generation, storage, transmission, carbon cost)
>    - Line overlay: total blended cost
>    - Secondary Y axis: demand in TWh (line, right axis)
>
> 4. **Cumulative Carbon Abated** (area):
>    - Filled area: cumulative MtCO2 avoided
>    - Horizontal reference lines: DAC equivalent cost at $300/ton, $150/ton
>    - Annotation: "break-even year" where abatement value exceeds clean energy premium
>
> All charts: `Plotly.react()` for smooth transitions (not `newPlot`), responsive resize on window change, export button (Plotly modebar), consistent font (Plus Jakarta Sans).
>
> `frontend/js/app.js` — update to:
> - Show loading spinner during API call
> - Auto-run projection on any parameter change (debounced 500ms)
> - Display headline stats above charts: total cost delta vs. fossil baseline, total carbon abated, dominant resource, crossover year
> - Animate stat counters on update

**Prompt 8 — Scenario presets & comparison:**

> Add scenario preset functionality:
>
> Backend (`main.py` + new `scenarios.py`):
> - Define 5 preset scenarios:
>   1. "Current Policy" — no carbon price, 2% demand growth, medium costs
>   2. "Moderate Carbon Tax" — $75/ton by 2035, 2% growth, medium costs
>   3. "Aggressive Decarbonization" — $150/ton by 2030, 3% growth, low renewable costs
>   4. "High Demand (AI/Data Centers)" — no carbon price, 5% growth, medium costs
>   5. "Technology Breakthrough" — no carbon price, 2% growth, 25% learning rate
> - `GET /api/scenarios` returns the full list with parameter values
>
> Frontend:
> - Dropdown or button row above the parameter panel to select presets
> - Selecting a preset fills in all sliders/toggles to match
> - "Compare" mode: run 2-3 scenarios side-by-side
>   - Split each chart into small multiples or overlay with different line styles
>   - Legend distinguishes scenarios by color + line style
>   - Stat cards show delta between scenarios
> - "Custom" option: user adjusts any param freely (clears preset selection)

---

### Phase 4: Carbon Price Deep-Dive (Sonnet → Opus for dispatch logic)

**Prompt 9 — Carbon price impact module (start with Sonnet, escalate if needed):**

> Add a dedicated carbon price analysis panel (new tab or expandable section):
>
> **Backend** — new endpoint `POST /api/carbon-analysis`:
> - Input: iso, cfe_target, carbon_price_range (min, max, step)
> - Run projection across the full carbon price range
> - Return: for each price point, the optimal mix, total cost, and marginal abatement cost
>
> **Frontend** — new chart section "Carbon Price Impact":
> 1. **MAC Curve**: marginal abatement cost vs. carbon price (shows diminishing returns)
> 2. **Resource Switching**: stacked area showing how mix shifts as carbon price increases (gas → CCS → nuclear → renewables+storage)
> 3. **Cost Parity Map**: heatmap of carbon_price × cfe_target showing where clean energy is cheaper than fossil (the "green zone")
> 4. **Reference lines**: EU ETS current (~€60), US SCC ($51 EPA / $185 Rennert), IEA Net Zero ($250 by 2050)
>
> This is the most domain-intensive piece. If Sonnet produces results that look physically implausible (e.g., coal increasing with carbon prices, or costs decreasing without technology improvement), escalate to Opus for the dispatch re-ranking logic.

**Prompt 10 (Opus — only if needed) — Dispatch re-ranking under carbon prices:**

> The carbon price analysis is producing implausible results. Fix the merit-order dispatch logic:
>
> The issue is in `projection_engine.py` where carbon costs are applied to the fossil stack. The correct logic:
>
> 1. Each fossil unit has a variable cost: fuel_cost + VOM + (carbon_price × emission_rate)
> 2. Emission rates by fuel: gas_ccgt=0.35, gas_ct=0.50, coal=0.95, oil=0.75 tCO2/MWh
> 3. As carbon price rises, the dispatch order changes: cheap coal gets expensive, efficient gas moves up
> 4. At ~$40/ton: gas CCGT beats coal on variable cost → coal retires first
> 5. At ~$100/ton: CCS-CCGT beats unabated gas → CCS deployment accelerates
> 6. At ~$150/ton: new nuclear beats gas on LCOE → nuclear deployment begins
> 7. Storage value increases with carbon price because it displaces marginal fossil
>
> The re-ranking must respect:
> - Regional fossil fleet composition (ERCOT is gas-heavy, PJM/MISO have coal)
> - Existing clean generation can't be "un-built" — it's a floor
> - CCS has a capture rate of 90%, so residual emissions remain
> - Carbon cost applies to NET emissions (after CCS capture)
>
> Use numpy vectorized operations. The existing `lmp_engine.py` in the main repo has the merit-order dispatch logic — adapt it for the projection context.

---

### Phase 5: Asset Screening Tool (Sonnet → Opus for dispatch validation)

This is the core **probabilistic screening tool** — a proxy for IPM / capacity expansion models that gives directionally correct probabilities for asset-class viability under different futures.

**Prompt 11 — Screening engine (Sonnet):**

> Implement `backend/screening.py` — the probabilistic asset screening engine.
>
> **Purpose**: Sweep a multi-dimensional scenario space and compute, for each heat-rate bin in the fossil fleet, the probability of being operating, marginal, or stranded. Also detect when new gas capacity is needed.
>
> **Endpoint**: `POST /api/screen`
> - Input: `ScreeningRequest` (iso, heat_rate_bins, carbon_price_range, demand_growth_range, cfe_targets, learning_rate)
> - Output: `ScreeningResponse` (bins with survival probabilities, new-build triggers, scenario count)
>
> **Screening logic**:
>
> 1. **Build fleet**: `FleetModel(iso, n_bins=request.heat_rate_bins)` → get all bins for gas_cc, gas_ct, coal, oil
>
> 2. **Generate scenario grid**: Cartesian product of:
>    - Carbon price: range(min, max, step) — e.g., 0, 25, 50, ..., 300 (13 values)
>    - Demand growth: range(min, max, step) — e.g., 0, 1, 2, 3, 4, 5 (6 values)
>    - CFE targets: list — e.g., [75, 85, 90, 95] (4 values)
>    - Total: ~312 scenarios per default config (fast enough for real-time)
>
> 3. **For each scenario**: Run projection (reuse `projection_engine.project()`), then:
>    a. Compute market clearing price from the clean energy mix cost
>    b. For each fossil bin, compute variable cost at that scenario's carbon price
>    c. **Classify**:
>       - **Operating**: bin variable cost < clearing price AND bin dispatches >50% of hours
>       - **Marginal**: bin variable cost within ±10% of clearing price OR dispatches 10-50% of hours
>       - **Stranded**: bin variable cost > clearing price for >90% of hours (economically unviable)
>
> 4. **Aggregate across scenarios**: For each bin:
>    - `p_operating` = fraction of scenarios where bin is "operating"
>    - `p_marginal` = fraction where "marginal"
>    - `p_stranded` = fraction where "stranded"
>    - `median_retirement_carbon_price` = carbon price at which P(stranded) crosses 50%
>    - `capacity_factor_p10/p50/p90` = percentiles of dispatch CF across all scenarios
>
> 5. **New gas build trigger**: For each scenario, check:
>    - `residual_peak_mw = peak_demand × (1 + demand_growth)^years × (1 + 0.15 RA margin) - clean_firm_mw × ELCC`
>    - `surviving_fossil_mw = sum(bin.capacity for non-stranded bins)`
>    - `deficit = residual_peak_mw - surviving_fossil_mw × GAF`
>    - If `deficit > 0`: new gas needed. Record the year + MW needed.
>    - Output: `new_build_trigger[scenario_label] = {year: int, mw_needed: float}` or null
>    - Aggregate: P10/P50/P90 of trigger year across scenarios
>
> **Performance**: The scenario grid is small (~300 combos) and each projection is <500ms, so total sweep <2min. Use `concurrent.futures.ThreadPoolExecutor` or vectorize the sweep with numpy broadcasting if possible.
>
> **Key insight this enables**: "Under 78% of plausible scenarios, CCGTs with heat rates above 8.0 are economically stranded by $75/ton carbon. New gas is needed in PJM by 2031 under 60% of high-demand scenarios." — This is exactly the kind of directional, probabilistic output that screens which scenarios deserve a full IPM run.

**Prompt 12 — Screening UI (Sonnet):**

> Create `frontend/screening.html` (or a tab within `index.html`) for the asset screening tool:
>
> **Parameter panel**:
> - ISO selector (same as projection page)
> - **Heat rate bins slider**: 1–7, integer steps, default 5. Label shows current value + description:
>   - 1: "Aggregate (single class per fuel)"
>   - 2-3: "Coarse (old vs. new)"
>   - 4-5: "Standard (recommended)"
>   - 6-7: "Fine (maximum granularity)"
> - Carbon price range: min/max/step sliders (default 0-300 step 25)
> - Demand growth range: min/max sliders (default 0-5%)
> - CFE target checkboxes: 75%, 85%, 90%, 95% (multi-select, all on by default)
> - "Run Screening" button
>
> **Charts** (5 panels in a responsive grid):
>
> 1. **Asset Survival Heatmap** (Plotly heatmap):
>    - X axis: carbon price ($/ton)
>    - Y axis: fleet bins (labeled by fuel + heat rate range, e.g., "Gas CC 7.0-7.5")
>    - Color: P(stranded) from 0% (green) → 50% (yellow) → 100% (red)
>    - Hover: shows P(operating), P(marginal), P(stranded), capacity MW
>    - **This is the headline chart** — instantly shows which assets survive which carbon prices
>
> 2. **Fleet Retirement Cascade** (stacked area):
>    - X axis: carbon price ($/ton, ascending)
>    - Y axis: capacity MW
>    - Stacked areas: one per bin, colored by fuel type + opacity by efficiency
>    - As carbon price rises, inefficient bins peel off the top → visual cascade
>    - Annotation: mark where coal fully retires, where first CCGT bin retires
>
> 3. **New Gas Build Trigger** (scatter + fan):
>    - X axis: demand growth rate (%)
>    - Y axis: year new gas needed
>    - Fan chart: P10/P50/P90 bands across carbon price scenarios
>    - Horizontal reference: "current planned additions" line
>    - Key question answered: "When do we NEED new gas, given these assumptions?"
>
> 4. **Bin-Level Dispatch Stack** (interactive bar chart):
>    - Dropdown: select a specific carbon price point
>    - Stacked bars: 8,760 hours compressed to 24-hour representative day
>    - Each bin shown as a separate color in the stack
>    - Shows: which bins run in peak vs. baseload vs. not at all
>    - Slider to scrub across carbon prices and watch the stack shift
>
> 5. **Capacity Factor Fan Chart** (line + confidence bands):
>    - One subplot per fuel type (gas CC, gas CT, coal)
>    - X axis: carbon price
>    - Y axis: capacity factor (0-100%)
>    - P10/P50/P90 bands per bin
>    - Shows the gradual decline in utilization as carbon price rises
>    - Efficient bins maintain CF longer; inefficient bins crash first
>
> **Headline stats bar** (above charts):
> - "X% of [ISO] gas fleet stranded at $Y/ton carbon"
> - "New gas needed by [year] under [X]% of scenarios"
> - "Most resilient asset: [bin label] — operates in [X]% of scenarios"
> - "Coal fully retired at $[X]/ton in [X]% of scenarios"
>
> Use Plotly for all charts. Same design system (shared.css, chart-colors.js). Mobile responsive — charts stack vertically below 768px. Heat rate bins slider triggers live re-run via debounced API call.

**Prompt 13 (Opus — only if needed) — Validating screening results:**

> The screening tool's retirement cascade doesn't match expected physical behavior. Validate and fix:
>
> **Expected behavior by carbon price** (for a typical PJM-like ISO):
> - $0/ton: All fossil operates by merit order (coal baseload where cheap, gas CC mid-merit, gas CT peaker)
> - $25/ton: Marginal coal plants begin retiring; efficient gas CC displaces some coal
> - $40-50/ton: Most coal uneconomic; gas CC becomes baseload fossil; inefficient gas CT rarely dispatches
> - $75/ton: Inefficient gas CC (HR >8.0) begins retiring; efficient gas CC + CCS competitive
> - $100-125/ton: Only efficient gas CC (HR <7.0) survives as fossil; CCS-CCGT becomes cheaper than unabated
> - $150+/ton: Only the most efficient gas CC peaking fleet remains; clean energy + storage handles most hours
>
> **Common bugs to check**:
> - Emission rate calculation: must use heat_rate × fuel_emission_factor, NOT a flat rate
> - Gas CT should retire BEFORE gas CC at same carbon price (higher heat rate)
> - Coal retirement should be monotonic — once stranded at $X, stays stranded at $X+1
> - New-build trigger should account for ELCC of clean resources replacing fossil capacity value
> - Ensure bins are capacity-weighted, not count-weighted (a few large efficient plants ≠ many small old ones)
>
> Cross-reference against:
> - EIA AEO 2024 coal retirement projections under carbon price scenarios
> - NREL Standard Scenarios (ReEDS) gas fleet utilization curves
> - PJM/MISO capacity auction clearing prices for context on capacity value

---

## Running It

### Build & Start

```bash
cd interactive-module/
docker compose up -d --build
```

### Access

```
http://localhost:8000
```

### Development (live reload)

```bash
# Mount source code for hot reload
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

`docker-compose.dev.yml`:
```yaml
services:
  app:
    volumes:
      - ./backend:/app/backend
      - ./frontend:/app/frontend
    command: uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### Share on Network

```bash
# Find your IP
hostname -I

# Others access via:
http://YOUR_IP:8000
```

---

## Copilot Workflow Tips

### Session Structure

1. **Start each session** by reading this doc + checking what exists
2. **One phase per session** is a good pace — scaffold, data, engine, frontend
3. **Test incrementally** — `docker compose up --build` after each phase to verify
4. **Commit after each working state** — don't accumulate uncommitted changes

### When to Switch Models

| Signal | Action |
|--------|--------|
| Writing boilerplate, HTML, CSS, config | Stay on Haiku |
| Wiring logic, building features, debugging integration | Use Sonnet |
| Results look wrong, need domain expertise | Switch to Opus |
| Back to boilerplate after Opus fix | Drop back to Haiku/Sonnet |

### Effective Prompting Patterns

- **Give the full file path** — "Edit `backend/projection_engine.py`" not "edit the engine"
- **Reference existing code** — "Adapt the merit-order logic from `lmp_engine.py` lines 45-120"
- **Specify the contract** — "Input: ProjectionRequest. Output: ProjectionResponse. Must respond in <500ms."
- **Show the shape of data** — paste a sample parquet schema or JSON response
- **State what's wrong** — "Solar percentage increases to 200% at high carbon prices" not "the chart looks weird"

### Debugging Checklist

```
□ Container builds?          docker compose build
□ Container starts?          docker compose up (check logs)
□ Data mounted?              docker compose exec app ls /app/data/
□ API responds?              curl http://localhost:8000/api/isos
□ Baseline loads?            curl http://localhost:8000/api/baseline/CAISO
□ Projection runs?           curl -X POST http://localhost:8000/api/project -H "Content-Type: application/json" -d '{"iso":"CAISO","target_year":2035,"carbon_price_usd":50,"demand_growth_pct":2,"cfe_target_pct":90,"cost_scenario":"medium","learning_rate":0.18}'
□ Frontend renders?          Open browser, check console for errors
□ Charts populate?           Run a projection, verify all 4 charts update
```

---

## Data Dependencies

The projection module reads from the main optimizer's output. Before first run, ensure these exist:

```
data/
├── step2.2-cost/          # Required — cost optimization results
│   ├── CAISO_cost_results.parquet
│   ├── ERCOT_cost_results.parquet
│   └── ... (7 ISOs)
├── step2.1-ef/            # Required — efficient frontier mixes
│   ├── CAISO_ef.parquet
│   └── ...
├── eia-860/               # Required for screening — unit-level fleet data
│   ├── generators.parquet # EIA 860 generator inventory (plant_id, cap_mw, heat_rate, fuel, online_year, iso)
│   └── README.md          # Data dictionary + vintage notes
├── step1-pfs/             # Optional — raw physics (only if re-optimizing)
└── step5-wrights/         # Optional — learning curve fits (fallback to defaults)
```

**Data tiers**:
- **Projection tool** (Phases 0-4): Needs `step2.2-cost/` + `step2.1-ef/`. Works without EIA 860.
- **Screening tool** (Phase 5): Needs `eia-860/` for heat rate binning. Without it, falls back to aggregate heat rates from `config.py` (equivalent to 1-bin mode).

If you haven't run the optimizer pipeline, the module will start with default 2025 values from `config.py` and still produce projections — they just won't be grounded in ISO-specific optimized mixes.

### EIA 860 Data Preparation

The screening tool needs unit-level generator data from EIA Form 860. To prepare:

```bash
# Download from EIA (annual release, usually March):
# https://www.eia.gov/electricity/data/eia860/

# Required columns from 3_1_Generator_Y2024.xlsx:
# Plant Code, Generator ID, Nameplate Capacity (MW),
# Heat Rate (Btu/kWh), Energy Source Code, Operating Year,
# Planned Retirement Year, Balancing Authority Code

# Map Balancing Authority → ISO:
# CISO → CAISO, ERCO → ERCOT, PJM → PJM,
# NYIS → NYISO, ISNE → NEISO, MISO → MISO, SWPP → SPP

# Filter to: fuel_type IN (coal, gas_cc, gas_ct, oil)
# Export as parquet to data/eia-860/generators.parquet
```

A step0 script can automate this — or do it manually once per year.
