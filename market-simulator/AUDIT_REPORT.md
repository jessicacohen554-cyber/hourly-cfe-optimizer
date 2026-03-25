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
