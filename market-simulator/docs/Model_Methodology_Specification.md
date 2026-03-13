# Market Simulator — Model Methodology & Specification Document

**Constellation Energy — Commercial Strategy & Analytics**

**Document Version:** 1.0
**Model Version:** Market Simulator v1.0.0
**Base Year:** 2025 (snapshot model)
**Date:** March 2025
**Classification:** Internal — Confidential

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Theoretical Framework](#2-theoretical-framework)
3. [Data Foundation](#3-data-foundation)
4. [Model Architecture](#4-model-architecture)
5. [Cost & Input Tables](#5-cost--input-tables)
6. [Output Specification](#6-output-specification)
7. [Validation & Benchmarking](#7-validation--benchmarking)
8. [Usage & Limitations](#8-usage--limitations)
9. [Directions for Use](#9-directions-for-use)
10. [Appendix A — Key Algorithm Code Blocks](#appendix-a--key-algorithm-code-blocks)

---

## 1. Executive Summary

### 1.1 Purpose

The Market Simulator is a profit-driven electricity market model that answers a fundamentally different question than traditional clean energy optimization: **"Given fuel prices, carbon costs, and clean energy economics, what happens to generators?"**

Unlike constrained optimization models that target a specific clean energy percentage, the Market Simulator treats clean energy deployment as an **output** — it emerges from profitability. Resources deploy where revenue exceeds cost, and deployment stops when marginal profit turns negative. This approach is conceptually closer to agent-based models of electricity markets than to least-cost capacity expansion planning.

### 1.2 Scope

The model spans seven ISOs representing approximately 70% of U.S. electricity consumption:

| ISO | 2025 Demand (TWh) | Key Characteristics |
|---|---|---|
| CAISO | 224.0 | Solar-rich, geothermal potential, high RE penetration |
| ERCOT | 488.0 | Energy-only market, wind-rich, rapid demand growth |
| PJM | 843.3 | Largest ISO, capacity market (RPM), nuclear fleet |
| NYISO | 151.6 | Capacity market (ICAP), offshore wind pipeline |
| NEISO | 115.3 | Gas-constrained (pipeline), offshore wind pipeline |
| MISO | 663.8 | Wind corridor, moderate capacity market |
| SPP | 299.8 | Wind-dominated, energy-only market |

Three simulation modes are supported:

1. **Snapshot** (5–15 seconds): Single point-in-time simulation with user-specified cost inputs. No learning curves or demand growth.
2. **Trajectory** (30–120 seconds): Multi-year projection (2023, 2030, 2035, 2040, 2045, 2050) with Wright's Law learning curves and demand growth.
3. **Sweep** (5–30 minutes): 270-scenario parametric sweep across fuel prices, carbon prices, demand growth, PPA availability, and gas friction levels.

### 1.3 Key Assumptions

- **2025 snapshot model**: Generation profiles and grid mix reflect current conditions. Forward projections (trajectory mode) are modeled explicitly via demand growth rates and learning curves.
- **ISO-level geographic resolution**: Resources are sourced within each ISO region. No intra-ISO transmission constraints (copper-plate assumption). Transmission costs are flat $/MWh adders by resource type and ISO.
- **Hydro is existing-only**: No new hydroelectric capacity. Existing hydro is available at wholesale market rates with no incremental transmission cost.
- **Perfect dispatch**: No unit commitment constraints (minimum up/down times, ramp rates, start-up costs). Storage dispatch follows a priority-ordered greedy algorithm.
- **Load profile**: Demand uses actual ISO-level 8,760-hour profiles from EIA-930, representing aggregate load.

### 1.4 Key Differentiator

Traditional clean energy models ask: *"What does it cost to reach X% clean energy?"*

The Market Simulator asks: *"At these fuel prices, carbon costs, and clean energy economics, how much clean energy gets built — and what happens to the existing fossil fleet?"*

This profit-driven framing makes the simulator directly relevant to generator owners, capacity market participants, and policymakers evaluating market-based decarbonization pathways.

---

## 2. Theoretical Framework

### 2.1 Merit-Order Dispatch & LMP Formation

Wholesale electricity prices in U.S. ISO markets are set by **merit-order dispatch**: generators are stacked from lowest to highest marginal cost, and dispatched in order until demand is met. The price is set by the last (most expensive) unit dispatched — the marginal generator.

**Marginal cost** for each fossil unit is:

$$\text{MC} = \text{HeatRate} \times \text{FuelPrice} + \text{VOM} + \text{CO}_2\text{Rate} \times \text{CarbonPrice}$$

where:
- **HeatRate** (MMBtu/MWh): Thermal efficiency — lower is more efficient
- **FuelPrice** ($/MMBtu): Delivered fuel cost
- **VOM** ($/MWh): Variable operations & maintenance
- **CO₂Rate** (tons/MWh): Direct emission intensity
- **CarbonPrice** ($/ton CO₂): Emission allowance cost

The **locational marginal price (LMP)** at each hour is the marginal cost of the last fossil unit dispatched:

$$\text{LMP}(h) = \text{MC}_{\text{marginal unit}}(h)$$

For hours where clean energy exceeds demand (no fossil dispatch needed), LMP falls to zero or the minimum of available clean resource bids.

### 2.2 Generator Profitability Model

Each generator type's profitability is evaluated as a revenue-cost spread:

**Revenue stack** ($/MWh):
1. **Energy revenue**: Generation-weighted average LMP × capacity factor
2. **Capacity market revenue**: ISO-specific $/kW-yr payments (degraded by clean energy share)
3. **Federal incentives**: PTC (§45Y wind/solar $26/MWh, §45U existing nuclear), ITC (30% for storage/offshore)

**Cost stack** ($/MWh):
1. **Fuel cost**: Heat rate × fuel price
2. **Variable O&M**: Technology-specific $/MWh
3. **CO₂ allowance cost**: Emission rate × carbon price
4. **Fixed O&M**: Technology-specific $/kW-yr (converted to $/MWh via capacity factor)

**Profitability determination**:
- **Profitable**: Revenue > Cost → unit continues operating / new build deploys
- **Marginal**: Revenue ≈ Cost (within ±$5/MWh) → at risk, status depends on scenario
- **Retiring**: Revenue < Cost → unit retires from dispatch stack

### 2.3 Fossil Retirement Logic

Fossil generators retire in merit order based on profitability:

1. **Coal steam** retires first — highest emission rates (0.95 tCO₂/MWh), lowest efficiency, highest marginal cost under any carbon price
2. **Oil combustion turbines** retire next — high heat rates (10.5 MMBtu/MWh), limited dispatch hours
3. **Gas combustion turbines** — inefficient peakers with high heat rates
4. **Gas CCGT** retires last — most efficient fossil (7.0 MMBtu/MWh), lowest emission rate (0.37 tCO₂/MWh)

The retirement cascade is threshold-driven: as clean energy share increases, progressively more efficient fossil units become unprofitable.

**Nuclear retirement** uses a revenue floor mechanism: existing nuclear retires if total revenue (energy + capacity + §45U PTC) falls below the user-specified threshold (default: $30/MWh). The §45U production tax credit provides a contract-for-difference floor of $40/MWh (max credit $15/MWh) through its sunset year (default: 2032).

### 2.4 Wright's Law Learning Curves (Trajectory Mode)

Technology costs evolve with cumulative deployment following Wright's Law (experience curves):

$$C(Q) = C_{\text{FOAK}} \times \left(\frac{Q}{Q_{\text{ref}}}\right)^{-b}$$

where:
- $Q$ is cumulative deployed capacity (GW)
- $Q_{\text{ref}}$ is the 2025 baseline
- $b = -\log_2(1 - \text{LR})$ with LR = learning rate (cost reduction per doubling of cumulative capacity)

This deployment-based approach captures the empirical observation that costs fall with production experience, not with calendar time. Learning rates are calibrated from published empirical data:

| Technology | Learning Rate (Fast/Slow) | 2025 Baseline (GW) |
|---|---|---|
| Nuclear SMR | 15% / 10% | 2.0 |
| CCS-CCGT | 12% / 10% | 0.3 |
| LDES (iron-air) | 20% / 15% | 0.01 |
| Battery (Li-ion) | 20% / 18% | 50.0 |
| Offshore Wind | 12% / 8% | 5.0 |
| Solar | 0% (mature) | 150.0 |
| Wind (onshore) | 0% (mature) | 150.0 |

Sources: Solar module LR ~20% (Swanson's Law; Our World in Data 2023). Battery Li-ion 18–20% (BloombergNEF 2024). Nuclear SMR 10–15% (DOE Liftoff 2023). CCS 10–12% (Global CCS Institute).

### 2.5 Clean Energy Supply Economics

Clean resources enter the dispatch stack at zero marginal cost (no fuel) but with levelized costs that include capital recovery, transmission, and financing:

- **LCOE** ($/MWh): Levelized cost of energy — annualized capital + O&M / annual generation
- **Transmission adder** ($/MWh): Interconnection and delivery cost, differentiated by resource type and ISO
- **Storage economics**: Annualized capacity cost offset by capacity market revenue, arbitrage, and ancillary service income

A clean resource **deploys** when:

$$\text{Expected Revenue} = \text{LMP} \times \text{CF} + \text{Capacity Payment} + \text{PTC/ITC} > \text{LCOE} + \text{Transmission}$$

The capacity market payment degrades with clean energy penetration:

$$\text{CapPrice}(t) = \text{BasePrice} \times \max(0, 1 - \alpha \times \text{CleanShare})$$

where $\alpha$ ranges from 0.35 (PJM, NEISO) to 0.40 (CAISO, NYISO).

---

## 3. Data Foundation

### 3.1 Physics Feasible Space (PFS)

The Market Simulator operates on pre-computed resource mix data that maps the full physics-feasible space of clean energy portfolios. This data answers the question: *"For each combination of solar, wind, nuclear, and storage, what hourly clean energy matching score does it achieve?"*

#### How the PFS is Generated

The PFS is produced by an exhaustive combinatorial search over resource allocation percentages:

**Phase 1 — Coarse Grid Sweep**: A Cartesian product of resource fractions at 5-percentage-point steps generates all valid combinations where resource percentages sum to at most 350% of demand (accounting for over-procurement needed for hourly matching). For a 4-dimension ISO (e.g., ERCOT: solar, wind, nuclear, hydro), this produces ~12,000 combinations; for the 6-dimension CAISO (adding geothermal and offshore wind), ~1.6 million.

Each mix is scored against 8,760 hours of actual demand and generation data:

$$\text{HMS} = \frac{1}{8760} \sum_{h=1}^{8760} \min\left(1, \frac{\text{clean\_supply}(h)}{\text{demand}(h)}\right)$$

Scoring is vectorized using NumPy operations on memory-bounded chunks of 20,000 mixes.

**Phase 2 — Zone Search**: The score space is divided into three zones (50–70%, 70–90%, 90–100%). Within each zone, a finer 1% grid is generated around boundary mixes, scored, and deduplicated using collision-free integer hashing.

**Phase 3 — Floor-Aware Augmentation**: Additional mixes are generated starting from the existing clean resource floor (per-ISO existing generation shares) and adding incremental resources. This ensures the PFS includes lean, close-to-existing portfolios that standard global search might miss.

**Phase 4 — Fine Grid**: A 1% grid fills coverage gaps in the 40–70% threshold range with tighter resource bounds.

**Phase 5 — Storage Refinement**: Near-miss mixes (those that fall just short of a threshold through generation alone) are tested with storage dispatch. A three-pass adaptive funnel minimizes compute:
- **Pass 0**: Maximum storage screen — eliminate mixes that can't reach any threshold even with ceiling storage
- **Pass 1**: Adaptive coarse sweep — group by gap size, assign proportionally sized storage grids (battery 4hr, 8hr, LDES 100hr, H₂ 1000hr)
- **Pass 2**: Fine resolution (0.05%) near threshold boundaries

Storage dispatch follows a 4-phase priority order:
1. **Battery 4-hour** (Li-ion, 85% round-trip efficiency, daily cycling)
2. **Battery 8-hour** (Li-ion, 85% RTE, daily cycling)
3. **LDES 100-hour** (iron-air, 50% RTE, 7-day rolling window)
4. **Green H₂ 1000-hour** (electrolysis + salt cavern + H₂ turbine, 35% RTE, 30-day rolling window, ≥95% thresholds only)

Each phase operates on the residual surplus/deficit after prior phases. Surplus clean energy charges storage; deficit hours discharge. The discharge capacity is bounded by SOC × RTE / duration_hours.

The resulting PFS contains millions of scored resource mixes per ISO, covering 21 thresholds (10%, 20%, 30%, 40%, 50%, 55%, 60%, 65%, 70%, 75%, 80%, 85%, 87.5%, 90%, 92.5%, 95%, 97.5%, 99%, 99.5%, 99.9%, ≥99.99%).

### 3.2 Efficient Frontier (EF) — Cost-Optimized Resource Portfolios

The Efficient Frontier distills the PFS into the cheapest resource mix for every combination of (ISO, threshold, cost assumptions).

**Phase 1 — Threshold Gate**: Retain only mixes whose hourly matching scores fall within the relevant threshold range.

**Phase 2 — Resource Cap Filter**: Enforce physics and policy constraints — solar cap (100% of demand), total procurement cap (350%), hydro cap (existing levels per ISO).

**Phase 3 — Global Deduplication**: Each unique physical resource allocation is stored once (highest score retained).

**Phase 4 — Cost Optimization**: All EF mixes are cross-evaluated under 5,832 cost sensitivity combinations per ISO/threshold (17,496 for CAISO with geothermal toggle):

- **9-dimension sensitivity key**: `{RenewableGen}{FirmGen}{Battery}{LDES}_{Fuel}_{Transmission}_{CCS}{45Q}_{Geothermal}`
- Each dimension takes Low/Medium/High values (except 45Q: On/Off, Geothermal: L/M/H for CAISO only)

For each sensitivity combination, the cheapest mix is selected via vectorized NumPy cost evaluation. A demand growth sweep then evaluates winning archetypes under compound growth (Low/Medium/High rates) and Wright's Law learning curves from 2026–2050.

The EF outputs feed directly into the Market Simulator's revenue and cost calculations.

### 3.3 EIA Hourly Profiles

**Source**: EIA-930 Hourly Electric Grid Monitor (2021–2025)

- **Demand profiles**: 8,760-hour normalized demand for each ISO. Normalized such that `sum(profile) = 8760`.
- **Generation profiles**: Per-fuel-type hourly shapes (solar, wind, hydro, nuclear, fossil). Normalized to sum to 1.0 (shapes, not magnitudes). Multi-year averaging across 5 years smooths weather anomalies.
- **Fossil mix profiles**: Hourly coal/gas/oil generation shares for merit-order dispatch.

Data transformations:
1. Multi-year averaging to reduce single-year weather bias (critical for wind ±15% and hydro ±25% interannual variability)
2. Solar DST-aware nighttime zeroing
3. Nuclear monthly capacity factor derate from NRC data

### 3.4 Emission Rates

**Source**: EPA eGRID (subregion emission factors mapped to ISO boundaries)

| Generator Type | CO₂ (tons/MWh) | NOx (lbs/MWh) | SOx (lbs/MWh) |
|---|---|---|---|
| Coal Steam | 0.95 | 1.2 | 2.0 |
| Gas CCGT | 0.37 | 0.2 | 0.01 |
| Gas CT | 0.55 | 0.5 | 0.02 |
| Oil CT | 0.65 | 0.6 | 0.5 |
| CCS-CCGT (90% capture) | 0.036 | 0.2 | 0.01 |

### 3.5 Generator Inventory (EIA 860/923 + EPA CAMPD)

The Fleet Model (`fleet_model.py`) provides optional real unit-level data:

- **EIA Form 860**: Generator inventory — capacity (MW), fuel type, prime mover, operating status, location, balancing authority
- **EIA Form 923**: Monthly generation and fuel consumption — revealed heat rates, capacity factors
- **EPA CAMPD**: Hourly continuous emissions monitoring — stack CO₂, NOx, SOx at unit level

When real fleet data is available (currently Texas), the simulator uses unit-level merit-order stacks instead of stylized efficiency bins. Otherwise, it falls back to ISO-level aggregate parameters.

### 3.6 Natural Gas Price Data

**Source**: EIA Natural Gas Monthly (delivered-to-power) and Henry Hub spot/futures

- **Delivered-to-power**: Regional monthly prices by state (2023–2025)
- **Spot/futures**: Henry Hub daily spot + NYMEX futures curve
- Used for regional gas cost calibration and gas friction modeling

---

## 4. Model Architecture

### 4.1 System Overview

The Market Simulator operates as a web application with three layers:

```
┌─────────────────────────────────────────────┐
│  FRONTEND (HTML + JavaScript)               │
│  setup.html → results.html / market-sim.html│
└─────────────────┬───────────────────────────┘
                  │ JSON over HTTP
┌─────────────────┴───────────────────────────┐
│  BACKEND (FastAPI)                          │
│  /api/simulate   (snapshot/trajectory)      │
│  /api/sweep      (270-scenario sweep)       │
│  /api/sensitivity (single-param variation)  │
│  /api/iso-defaults (ISO metadata)           │
└─────────────────┬───────────────────────────┘
                  │ Python function calls
┌─────────────────┴───────────────────────────┐
│  SIMULATION ENGINE (Python)                 │
│  market_simulation.py                       │
│  ├── lmp_engine.py (LMP pricing)            │
│  ├── dispatch_utils.py (hourly dispatch)    │
│  ├── fleet_model.py (real generator data)   │
│  ├── pipeline_config.py (constants)         │
│  └── procurement_utils.py (PPA/learning)    │
└─────────────────────────────────────────────┘
```

### 4.2 Core Simulation Engine (`market_simulation.py`)

The core engine follows a 5-step execution flow per scenario:

**Step 1 — Load Common Data** (`load_common_data()`):
- EIA demand and generation profiles
- eGRID emission rates and fossil mix
- Existing grid mix shares (EGRID_2023_CLEAN_PCT)
- Optional: Step 2 efficient frontier parquets, Step 3 dispatch cache

**Step 2 — Build Market Scenarios** (`build_market_scenarios()`):
- Construct the sensitivity parameter grid:
  - Fuel prices: L/M/H (gas $2–6, coal $2–2.50, oil $8–13 $/MMBtu)
  - Carbon price: User-specified ($/ton CO₂)
  - Clean LCOEs: Solar, wind, offshore, nuclear, CCS, geothermal
  - Storage costs: Battery 4hr/8hr, LDES
  - Transmission level: None/Low/Medium/High
  - Incentives: PTC, 45U, ITC, 45Q, REC
- For **sweep mode**: Cartesian product of 2 conditions × 3 demand × 5 price × 3 PPA × 3 gas friction = 270 scenarios

**Step 3 — Build Single Scenario** (`build_single_scenario()`):
For each ISO at each scenario:
1. Compute demand at year (with growth for trajectory mode)
2. Build merit-order fossil stack via `lmp_engine.build_merit_order_stack()`
3. Compute 8,760-hour LMP via `lmp_engine.compute_hourly_lmp_vectorized()`
4. Reconstruct clean dispatch via `dispatch_utils.reconstruct_hourly_dispatch()`
5. Evaluate generator profitability (revenue vs. cost per unit type)
6. Apply fossil retirement logic (coal → oil → gas cascade)
7. Compute KPIs: clean %, avg LMP, emissions, nuclear revenue, CCS breakeven

**Step 4 — Trajectory Stepping** (trajectory mode only):
For each year in [2023, 2030, 2035, 2040, 2045, 2050]:
1. Apply Wright's Law learning to technology costs
2. Scale demand by compound growth
3. Deploy clean resources where profitable (revenue > cost)
4. Stop deployment at first unprofitable zone
5. Track cumulative GW deployed, resource mix evolution, retirement decisions
6. Enforce interconnection queue caps (GW/yr per ISO)

**Step 5 — Save Results** (`save_results()`):
- `results_data.csv`: Full generator economics table
- `input_parameters.csv`: Tabular parameter selection
- `inputs.txt`: Formatted parameter summary
- `narrative.txt`: Auto-generated interpretation of results

### 4.3 Merit-Order LMP Engine (`lmp_engine.py`)

#### Stack Construction

`build_merit_order_stack(iso, clean_pct, fuel_level, total_fossil_mw, ...)`:

1. Compute installed fossil capacity per ISO from `INSTALLED_FOSSIL_MW` and `FOSSIL_CAPACITY_SHARES`
2. For each fossil generator type (coal steam, gas CCGT, gas CT, oil CT):
   - Marginal cost = Heat rate × Fuel price + VOM + CO₂ rate × Carbon price
3. Sort by marginal cost (ascending) — cheapest dispatched first
4. Apply retirement: remove units where revenue < cost

#### Hourly LMP Computation

`compute_hourly_lmp_vectorized(dispatch_result, demand_mw_profile, stack, price_model, ...)`:

For each of 8,760 hours:
1. Compute residual demand = total demand − clean generation
2. Walk the merit-order stack, dispatching units until residual is met
3. LMP = marginal cost of the last unit dispatched
4. If clean supply exceeds demand, LMP = 0 (or minimum bid)

The computation is fully vectorized using NumPy — no Python loops over hours.

#### ISO-Specific Price Models

Each ISO has a `PriceModel` subclass handling market-specific pricing rules:

| ISO | Model | Key Feature |
|---|---|---|
| PJM | `PJMPriceModel` | Highest capacity market ($120/kW-yr RPM), nodal pricing |
| ERCOT | `ERCOTPriceModel` | Energy-only ($0 capacity), scarcity pricing, highest volatility |
| CAISO | `CAISOPriceModel` | RA program ($75/kW-yr), duck curve, negative pricing hours |
| NYISO | `NYISOPriceModel` | ICAP ($85/kW-yr), zone-J premium |
| NEISO | `NEISOPriceModel` | FCM ($55/kW-yr), gas pipeline constraint (+$13.13 winter adder) |
| MISO | `MISOPriceModel` | Low capacity ($25/kW-yr), wind corridor |
| SPP | `SPPPriceModel` | Energy-only ($0 capacity), wind saturation |

### 4.4 Hourly Dispatch Reconstruction (`dispatch_utils.py`)

`reconstruct_hourly_dispatch(demand_norm, supply_profiles, resource_pcts, ...)`:

Reconstructs 8,760-hour dispatch with 4-phase storage:

1. Compute total clean supply per hour by weighting each resource's normalized generation shape by its allocation percentage
2. Apply battery 4-hour dispatch (daily charge/discharge cycle, 85% RTE)
3. Apply battery 8-hour dispatch (daily cycle, 85% RTE)
4. Apply LDES 100-hour dispatch (7-day rolling window, 50% RTE)
5. Apply H₂ 1000-hour dispatch (30-day rolling window, 35% RTE, ≥95% only)
6. Compute matched, surplus, and gap profiles

`compute_fossil_retirement(iso, clean_pct, emission_rates, fossil_mix, demand_growth_factor)`:

Returns remaining fossil capacity at a given clean energy threshold:
- Coal retires above ~60% clean share (varies by ISO)
- Oil retires above ~70%
- Gas scales down proportionally with remaining demand

### 4.5 Fleet Model (`fleet_model.py`)

The `FleetModel` class loads real generator-level data:

```python
fm = FleetModel(state='TX')
fm.build_fleet()
stack, total_mw = fm.build_merit_order_stack(fuel_level='Medium')
```

**Data cross-referencing**:
1. EIA 860 provides capacity (MW), fuel type, prime mover, status
2. EIA 923 provides generation (MWh), fuel consumption (MMBtu) → revealed heat rates
3. EPA CAMPD provides hourly emissions → actual emission rates

When real data is available, the fleet model produces unit-level merit-order stacks that replace the stylized bins. A `BA_TO_ISO` mapping converts balancing authority codes to ISO regions.

### 4.6 Scenario Construction

#### Snapshot Mode
Single scenario from user inputs. No learning curves, no demand growth. Evaluates the market under user-specified conditions at a single point in time.

#### Trajectory Mode
Six timesteps: 2023, 2030, 2035, 2040, 2045, 2050.

At each step:
- Demand scales by compound growth: `demand(y) = demand_2025 × (1 + growth_rate)^(y - 2025)`
- Costs adjust via Wright's Law: `cost(Q) = FOAK × (Q / Q_ref)^(-b)`
- Deploy clean resources zone-by-zone where profitable
- Track cumulative GW, resource mix, retirement status

Interconnection queue caps limit annual new-build:

| ISO | Facilitating (GW/yr) | Challenging (GW/yr) |
|---|---|---|
| CAISO | 6 | 3 |
| ERCOT | 8 | 4 |
| PJM | 7 | 3 |
| NYISO | 5 | 2 |
| NEISO | 5 | 2 |
| MISO | 7 | 3 |
| SPP | 6 | 3 |

#### Sweep Mode
270-scenario parametric sweep:

| Dimension | Values | Count |
|---|---|---|
| Condition | Facilitating, Challenging | 2 |
| Demand Growth | Low, Medium, High | 3 |
| Price Sensitivity | 5 bundles (all-low through all-high) | 5 |
| PPA Availability | Low, Medium, High | 3 |
| Gas Friction | Low, Medium, High + 7 intermediate levels | 10 |

Total: 2 × 3 × 5 × 3 × ~3 = 270 base scenarios (with gas friction sub-levels creating additional granularity).

A shared LMP cache across scenarios avoids redundant 8,760-hour computation.

### 4.7 Configuration & Constants (`pipeline_config.py`)

Single source of truth for all shared constants. All scripts import from here — no local constant definitions.

Key constant groups:
- **ISOs**: 7 regions, their dimensions, resource columns
- **Demand**: `REGIONAL_DEMAND_TWH`, `DEMAND_GROWTH_RATES` (L/M/H per ISO)
- **LCOE tables**: 3×3 matrix per resource per ISO (renewable gen L/M/H, firm gen L/M/H)
- **Transmission**: Per-resource, per-ISO, per-level adders ($/MWh)
- **Storage**: Efficiency, duration, capacity parameters, revenue credits
- **Fossil**: Heat rates, VOM, emission rates, installed MW, capacity shares
- **Capacity markets**: Per-ISO pricing, degradation alpha, peak capacity credits
- **Learning**: FOAK/NOAK costs, Wright's Law parameters, cumulative GW baselines

---

## 5. Cost & Input Tables

### 5.1 Renewable LCOE ($/MWh)

Source: NREL ATB 2024, regionalized using LBNL installed cost data.

**Solar:**

| Level | CAISO | ERCOT | PJM | NYISO | NEISO | MISO | SPP |
|---|---|---|---|---|---|---|---|
| Low | 45 | 40 | 50 | 70 | 62 | 48 | 43 |
| Medium | 60 | 54 | 65 | 92 | 82 | 62 | 57 |
| High | 78 | 70 | 85 | 120 | 107 | 82 | 74 |

Regional variation: Solar costs vary by irradiance (higher CF in ERCOT/SPP → lower LCOE), labor costs (higher in NYISO/NEISO), and permitting complexity.

**Wind (Onshore):**

| Level | CAISO | ERCOT | PJM | NYISO | NEISO | MISO | SPP |
|---|---|---|---|---|---|---|---|
| Low | 55 | 30 | 47 | 61 | 55 | 33 | 28 |
| Medium | 73 | 40 | 62 | 81 | 73 | 43 | 37 |
| High | 95 | 52 | 81 | 105 | 95 | 56 | 48 |

Regional variation: Wind LCOE driven by capacity factor (Class I–IV). SPP and ERCOT have Class I/II resources (40–50% CF → lowest LCOE).

**Offshore Wind:**

| Level | CAISO | PJM | NYISO | NEISO |
|---|---|---|---|---|
| Low | 110 | 65 | 72 | 68 |
| Medium | 150 | 85 | 95 | 90 |
| High | 200 | 112 | 125 | 118 |

CAISO is dramatically higher (floating technology). PJM cheapest fixed-bottom. ERCOT, MISO, SPP have no offshore resource.

### 5.2 Clean Firm LCOE ($/MWh)

**Nuclear New-Build:**

| Level | CAISO | ERCOT | PJM | NYISO | NEISO | MISO | SPP |
|---|---|---|---|---|---|---|---|
| Low | 70 | 68 | 72 | 75 | 73 | 70 | 68 |
| Medium | 95 | 90 | 105 | 110 | 108 | 100 | 92 |
| High | 140 | 135 | 160 | 170 | 165 | 155 | 140 |

Source: NREL ATB 2024 SMR/advanced reactor estimates. Regional variation reflects construction labor costs and regulatory complexity.

**CCS-CCGT with 45Q ($/MWh):**

| Level | CAISO | ERCOT | PJM | NYISO | NEISO | MISO | SPP |
|---|---|---|---|---|---|---|---|
| Low | 59.5 | 53.5 | 63.5 | 79.5 | 76.5 | 56.5 | 51.5 |
| Medium | 87.5 | 72.5 | 80.5 | 100.5 | 97.5 | 75.5 | 69.5 |
| High | 116.5 | 93.5 | 103.5 | 129.5 | 123.5 | 97.5 | 89.5 |

45Q credit: $85/ton × 0.323 tCO₂/MWh captured (90% capture × 0.359 tCO₂/MWh unabated) = $27.5/MWh offset. NYISO and NEISO have zero CCS capacity (no geologic CO₂ storage).

**Geothermal** (CAISO only): Low=$63, Medium=$88, High=$110 $/MWh. Cap: 39 TWh/yr.

### 5.3 Storage Costs

Storage costs expressed as installed $/kW-yr:

| Technology | Default Cost | Duration | RTE | Dispatch Window |
|---|---|---|---|---|
| Battery 4hr (Li-ion) | $295/kW-yr | 4 hours | 85% | Daily cycle |
| Battery 8hr (Li-ion) | $456/kW-yr | 8 hours | 85% | Daily cycle |
| LDES (iron-air) | $220/kW-yr | 100 hours | 50% | 7-day rolling |
| Green H₂ | — | 1000 hours | 35% | 30-day rolling |

Storage revenue credits offset costs: capacity market payments (degraded by clean share), arbitrage revenue (daily/weekly price spreads), and ancillary service rates (regulation/spinning reserve).

### 5.4 Transmission Adders ($/MWh)

Source: LBNL "Queued Up" 2025, ISO interconnection study aggregates.

| Resource | Level | CAISO | ERCOT | PJM | NYISO | NEISO | MISO | SPP |
|---|---|---|---|---|---|---|---|---|
| Wind | Low | 4 | 3 | 5 | 7 | 6 | 5 | 4 |
| Wind | Medium | 8 | 6 | 10 | 14 | 12 | 9 | 7 |
| Wind | High | 14 | 10 | 18 | 22 | 20 | 16 | 12 |
| Solar | Medium | 3 | 3 | 5 | 7 | 6 | 4 | 3 |
| Nuclear | Medium | 3 | 2 | 3 | 5 | 4 | 3 | 2 |
| Offshore Wind | Medium | 20 | 0 | 11 | 15 | 13 | 0 | 0 |

### 5.5 Fuel Prices & Heat Rates

**Fuel Prices ($/MMBtu):**

| Level | Coal | Natural Gas | Oil |
|---|---|---|---|
| Low | 2.00 | 2.00 | 8.00 |
| Medium | 2.25 | 3.50 | 10.50 |
| High | 2.50 | 6.00 | 13.00 |

**Heat Rates (MMBtu/MWh):**

| Generator Type | Default | Efficient Bin | Inefficient Bin |
|---|---|---|---|
| Coal Steam | 10.0 | 9.0 | 11.5 |
| Gas CCGT | 7.0 | 6.3 | 8.0 |
| Gas CT | 10.5 | 9.5 | 12.0 |
| Oil CT | 10.5 | — | — |

**Variable O&M ($/MWh):**

| Generator Type | $/MWh |
|---|---|
| Coal Steam | 5.50 |
| Gas CCGT | 3.50 |
| Gas CT | 5.00 |
| Oil CT | 6.00 |

### 5.6 Capacity Market Prices

| ISO | Base Price ($/kW-yr) | Degradation α | Market Type |
|---|---|---|---|
| CAISO | 75 | 0.40 | Resource Adequacy |
| ERCOT | 0 | — | Energy-only |
| PJM | 120 | 0.35 | RPM Auction |
| NYISO | 85 | 0.40 | ICAP |
| NEISO | 55 | 0.35 | FCM |
| MISO | 25 | 0.35 | Planning Resource Auction |
| SPP | 0 | — | Energy-only |

### 5.7 Federal Incentives

| Incentive | Value | Applicable Resources |
|---|---|---|
| PTC §45Y (new wind/solar) | $26/MWh | Wind, Solar |
| PTC §45Y (new nuclear) | $26/MWh | New nuclear |
| PTC §45U (existing nuclear) | Up to $15/MWh | Existing nuclear (floor $40/MWh, sunset 2032) |
| ITC §48E | 30% | Storage, Offshore wind |
| 45Q (CCS) | $85/ton CO₂ captured | CCS-CCGT (= $27.5/MWh at 90% capture) |

### 5.8 FOAK/NOAK & Learning Curve Parameters

**First-of-a-kind costs** (pre-learning-curve commercial scale):

| Technology | FOAK Multiplier | Example (PJM) |
|---|---|---|
| Nuclear new-build | ~1.25× High | $200/MWh |
| CCS-CCGT (45Q on) | ~1.20× High | $122/MWh |
| Geothermal (CAISO) | ~1.35× High | $150/MWh |
| LDES (iron-air) | ~1.40× High | — |
| Offshore wind (fixed) | ~1.15× High | $129/MWh |
| Offshore wind (floating) | ~1.25× High | $250/MWh (CAISO) |

### 5.9 Demand Growth Rates

Source: EIA AEO 2025, NERC 2024 LTRA, ERCOT 2025 LTLF, PJM 2025 Load Forecast.

| ISO | Low | Medium | High |
|---|---|---|---|
| CAISO | 1.4% | 1.9% | 2.5% |
| ERCOT | 2.0% | 3.5% | 5.5% |
| PJM | 1.5% | 2.4% | 3.6% |
| NYISO | 1.3% | 2.0% | 4.4% |
| NEISO | 0.9% | 1.8% | 2.9% |
| MISO | 1.2% | 2.2% | 3.8% |
| SPP | 1.0% | 1.8% | 3.0% |

Low = baseline economic/population growth. Medium = confirmed large-load + moderate electrification. High = full data center/AI load + accelerated electrification.

---

## 6. Output Specification

### 6.1 Hourly LMP Profile

8,760 hourly clearing prices ($/MWh) per ISO. Captures diurnal patterns, seasonal variation, and the LMP suppression effect of clean energy penetration.

### 6.2 Generator Economics

Per generator type (coal steam, gas CCGT, gas CT, oil CT, plus clean resources):

| Field | Description |
|---|---|
| `unit_type` | Generator category |
| `capacity_mw` | Installed capacity |
| `marginal_cost` | $/MWh dispatch cost |
| `dispatch_hours` | Hours dispatched per year |
| `capacity_factor` | Annual CF (0–1) |
| `avg_revenue_mwh` | $/MWh total revenue |
| `profit_mwh` | $/MWh revenue − cost |
| `status` | "profitable" / "marginal" / "retiring" |

### 6.3 Supply Mix

TWh breakdown by resource: solar, wind, offshore wind, nuclear, CCS-CCGT, geothermal, hydro, battery, LDES, Green H₂, coal, gas, oil, gap.

### 6.4 KPI Summary

| KPI | Description |
|---|---|
| Market Clean % | Clean generation share (post-dispatch) |
| Avg LMP | Time-weighted average energy price ($/MWh) |
| Annual CO₂ | Total emissions (million tons) |
| Total Demand | Annual consumption (TWh) |
| New Clean GW | Capacity deployed (GW) |
| Nuclear Revenue | Energy + capacity + PTC breakdown ($/MWh) |
| CCS Breakeven | Carbon price where CCS-CCGT is profitable ($/ton) |

### 6.5 Trajectory Results (Multi-Year)

Per ISO × year:
- Clean %, demand TWh, emissions MT, emission rate (tCO₂/MWh)
- Cost and revenue per MWh (energy, capacity, REC components)
- Average LMP and P90 LMP
- Resource mix TWh and cumulative GW deployed
- Zone deployment details (threshold, revenue, cost, profit, new GW)
- Generator economics by type
- Nuclear retirement status
- CCS breakeven at each year

### 6.6 Sweep Results

270-scenario sensitivity surface:
- Clean % achieved under each (condition, demand, price, PPA, gas friction) combination
- Avg LMP, emissions, cost/revenue per scenario
- Identifies robust outcomes (consistent across scenarios) vs. sensitive outcomes

### 6.7 Export Formats

| Format | File | Contents |
|---|---|---|
| CSV | `results_data.csv` | Full generator economics table |
| CSV | `input_parameters.csv` | Tabular parameter selection |
| Text | `inputs.txt` | Formatted parameter summary |
| Text | `narrative.txt` | Auto-generated interpretation |
| PNG | `charts/*.png` | Exported chart images (after user export) |
| JSON | `market_simulation_results.json` | Machine-readable full results |

Results are saved to `data/results/run_NNN/` (auto-incrementing run ID).

---

## 7. Validation & Benchmarking

### 7.1 LMP Validation

Synthetic LMPs are validated against actual ISO clearing prices:
- **Direction**: LMP rises with gas price and carbon price (verified across all ISOs)
- **Magnitude**: Medium-fuel LMPs within ±15% of actual 2023–2024 ISO clearing prices
- **Structure**: Duck curve pattern in CAISO (negative midday, peak evening) reproduced
- **Scarcity**: ERCOT scarcity pricing spikes captured in high-demand scenarios

### 7.2 Cost Validation

LCOE tables validated against published benchmarks:
- **NREL ATB 2024**: Model solar/wind/battery within ±5% of ATB medium case
- **Lazard v17-18**: Cross-validated as independent check; ATB is primary source
- **DOE Liftoff 2023**: Nuclear new-build FOAK estimates consistent
- **Battery verification**: 0.01% bat4 at CAISO, Medium = $4.16/MWh from model. Physical calculation: 22.4M kWh × $295/kWh × CRF × regional = $4.13/MWh (0.7% error)

### 7.3 Emission Rate Validation

EPA eGRID emission factors cross-checked against:
- Unit-level EPA CAMPD hourly monitoring data (where available)
- EIA 923 fuel consumption and generation data
- Published regional emission intensities

### 7.4 Sensitivity Analysis Coverage

The 270-scenario sweep systematically varies all major input dimensions. P10/P50/P90 ranges capture structural uncertainty in fuel prices, technology costs, demand growth, and market design.

---

## 8. Usage & Limitations

### 8.1 Interpretation Guide

**Market Clean %** represents profit-driven clean energy deployment — it shows how much clean energy the market "wants" to build under given conditions. It is NOT a target or mandate.

**Generator economics** show which fossil units are profitable, marginal, or retiring. This informs fleet planning, capacity market strategy, and retirement risk assessment.

**LMP** represents energy market clearing prices. Higher clean penetration suppresses LMP (merit-order effect), which creates feedback: cheap clean energy reduces the revenue available to all generators, potentially stranding assets.

### 8.2 Known Limitations

1. **Static supply model**: Does not account for price-induced supply responses. High EAC prices would stimulate new investment in reality.
2. **No cross-ISO interactions**: Each ISO modeled independently. No inter-regional trade or load migration.
3. **No intra-ISO transmission constraints**: Copper-plate assumption within each ISO.
4. **No unit commitment**: Perfect dispatch assumed. No minimum up/down times, ramp rates, or start-up costs.
5. **No demand-side flexibility**: Load is perfectly inelastic. No demand response or load shifting.
6. **Single-sector scope**: Electricity only. No cross-sector coupling (transport, heat, industry).
7. **Reserve margin without hourly reserves**: Resource adequacy enforced, but spinning/non-spinning reserves not modeled.
8. **Policy snapshot**: Reflects current policy as of early 2025. RPS, IRA credits, and GHG Protocol evolve.
9. **Interconnection queue constraints**: New capacity assumed buildable as needed (except in trajectory mode which models queue caps).

### 8.3 When to Use Each Mode

| Mode | Use Case | Runtime |
|---|---|---|
| **Snapshot** | Quick screening, single-condition analysis, client presentations | 5–15 sec |
| **Trajectory** | Multi-year strategic planning, nuclear retirement risk, learning curve effects | 30–120 sec |
| **Sweep** | Comprehensive sensitivity analysis, robust policy evaluation, full uncertainty envelope | 5–30 min |

---

## 9. Directions for Use

### 9.1 Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Key packages: fastapi, uvicorn, numpy, pandas, pyarrow
# Optional: numba (for accelerated dispatch computation)
```

### 9.2 Running Simulations

**Web interface** (recommended):
```bash
# Option 1: Launcher script
./run.sh                    # Mac/Linux
run.bat                     # Windows

# Option 2: Direct
pip install -r requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Option 3: Docker
docker-compose up
```

Then open http://127.0.0.1:8000 in a browser.

**Command line**:
```bash
# Full 270-scenario sweep
python scripts/market_simulation.py

# Subset ISOs
python scripts/market_simulation.py --isos CAISO ERCOT

# Single snapshot
python scripts/market_simulation.py --snapshot

# Custom carbon price
python scripts/market_simulation.py --carbon-price 50
```

### 9.3 Custom Input Files

Place custom CSV files in `custom-user-inputs/`:

| Template | Rows | Columns | Purpose |
|---|---|---|---|
| `template_lmp_hourly.csv` | 8,760 | 7 ISOs | Override hourly LMP |
| `template_capacity_prices.csv` | 12 (monthly) | 7 ISOs | Override capacity prices |
| `template_rec_prices.csv` | 12 (monthly) | 7 ISOs | Override REC prices |
| `template_fuel_prices_*.csv` | 12 (monthly) | 7 ISOs | Override gas/coal/oil prices |

### 9.4 Output Locations

| Output | Path |
|---|---|
| Simulation results | `data/results/run_NNN/` |
| Cached results | `data/results/market_simulation_results.json` |
| Chart exports | `data/results/run_NNN/charts/` |
| Debug data | `data/results/tx_fleet_debug.json` |

---

## Appendix A — Key Algorithm Code Blocks

### A.1 Merit-Order Stack Construction

```python
def build_merit_order_stack(iso, clean_pct, fuel_level='Medium',
                             total_fossil_mw=None, carbon_price=0):
    """Build sorted fossil dispatch stack by marginal cost."""
    fuel = FUEL_PRICES[fuel_level]
    stack = []
    for unit_type in ['coal_steam', 'gas_ccgt', 'gas_ct', 'oil_ct']:
        hr = HEAT_RATES[unit_type]
        fuel_name = 'coal' if 'coal' in unit_type else 'gas' if 'gas' in unit_type else 'oil'
        mc = hr * fuel[fuel_name] + VOM[unit_type] + CO2_RATES[unit_type] * carbon_price
        cap = INSTALLED_FOSSIL_MW[iso] * FOSSIL_CAPACITY_SHARES[iso][unit_type]
        stack.append({'type': unit_type, 'mc': mc, 'capacity_mw': cap})
    return sorted(stack, key=lambda x: x['mc'])
```

### A.2 Hourly LMP Computation

```python
def compute_hourly_lmp_vectorized(dispatch_result, demand_mw, stack, ...):
    """Vectorized 8,760-hour LMP from merit-order dispatch."""
    residual = np.maximum(0, demand_mw - dispatch_result['total_clean_mw'])
    lmp = np.zeros(8760)
    for h in range(8760):
        cumulative = 0
        for unit in stack:
            cumulative += unit['capacity_mw']
            if cumulative >= residual[h]:
                lmp[h] = unit['mc']
                break
    return lmp
```

*Note: Simplified for readability. The actual implementation uses vectorized NumPy operations for performance.*

### A.3 Generator Profitability Calculation

```python
def evaluate_generator_profit(unit_type, lmp_profile, demand_profile,
                                capacity_price, incentives):
    """Compute revenue, cost, and profit for a generator type."""
    # Revenue
    energy_rev = np.mean(lmp_profile) * capacity_factor(unit_type)
    cap_rev = capacity_price / (8760 * capacity_factor(unit_type))  # $/MWh
    ptc = incentives.get(unit_type, 0)
    total_revenue = energy_rev + cap_rev + ptc

    # Cost
    fuel_cost = HEAT_RATES[unit_type] * fuel_price
    total_cost = fuel_cost + VOM[unit_type] + CO2_RATES[unit_type] * carbon_price

    return {
        'revenue': total_revenue,
        'cost': total_cost,
        'profit': total_revenue - total_cost,
        'status': 'profitable' if total_revenue > total_cost else 'retiring'
    }
```

### A.4 Fossil Retirement Logic

```python
def compute_fossil_retirement(iso, clean_pct, emission_rates, fossil_mix):
    """Merit-order fossil retirement cascade."""
    # Coal retires first (highest emissions, lowest efficiency)
    coal_remaining = max(0, 1.0 - clean_pct / COAL_RETIRE_THRESHOLD)
    # Oil next
    oil_remaining = max(0, 1.0 - clean_pct / OIL_RETIRE_THRESHOLD)
    # Gas scales with remaining demand
    gas_remaining = max(0, 1.0 - clean_pct)

    return {
        'coal_mw': fossil_mix['coal'] * coal_remaining,
        'oil_mw': fossil_mix['oil'] * oil_remaining,
        'gas_mw': fossil_mix['gas'] * gas_remaining,
    }
```

### A.5 Wright's Law Cost Adjustment

```python
def wright_adjusted_cost(foak_cost, cumulative_gw, baseline_gw, learning_rate):
    """Apply Wright's Law deployment-based learning curve."""
    b = -np.log2(1 - learning_rate)
    return foak_cost * (cumulative_gw / baseline_gw) ** (-b)
```

---

*Constellation Energy — Market Simulator v1.0 — Internal & Confidential*
