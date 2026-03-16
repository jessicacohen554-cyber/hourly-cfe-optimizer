# Hourly CFE Optimizer — Model Methodology & Specification Document

**Constellation Energy — Commercial Strategy & Analytics**

**Document Version:** 1.0  
**Model Version:** Pipeline v1.0.0  
**Base Year:** 2025 (snapshot model)  
**Date:** June 2025  
**Classification:** Internal — Confidential

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Introduction](#2-introduction)
3. [Theoretical Framework](#3-theoretical-framework)
4. [Model Architecture](#4-model-architecture)
   - 4.1 [Data Engineering & Pipeline](#41-data-engineering--pipeline)
   - 4.2 [Cost & Input Tables](#42-cost--input-tables)
   - 4.3 [Algorithm Selection & Core Mathematical Functions](#43-algorithm-selection--core-mathematical-functions)
5. [Validation Results](#5-validation-results)
6. [Usage & Limitations](#6-usage--limitations)
7. [Directions for Use](#7-directions-for-use)
8. [Appendix A — Key Algorithm Code Blocks](#appendix-a--key-algorithm-code-blocks)

---

## 1. Executive Summary

### 1.1 Purpose

The Hourly CFE Optimizer is a computational model that determines the least-cost portfolio of clean energy resources required to achieve any specified level of hourly clean energy matching (10%–99.99%) across seven major U.S. ISO/RTO regions. Unlike traditional annual procurement models, this optimizer evaluates resource mixes against 8,760 hours of actual demand and generation data, capturing the temporal mismatch between variable renewable generation and load that drives procurement costs at high clean energy targets.

### 1.2 Scope

The model spans seven ISOs representing approximately 70% of U.S. electricity consumption:

| ISO | 2025 Demand (TWh) | Dimensions | Key Resources |
|---|---|---|---|
| CAISO | 224.0 | 6D | Solar, wind, nuclear, offshore wind, geothermal |
| ERCOT | 488.0 | 4D | Solar, wind, nuclear |
| PJM | 843.3 | 5D | Nuclear, wind, solar, offshore wind |
| NYISO | 151.6 | 5D | Nuclear, hydro, wind, offshore wind |
| NEISO | 115.3 | 5D | Nuclear, hydro, wind, offshore wind |
| MISO | 663.8 | 4D | Nuclear, wind, solar |
| SPP | 299.8 | 4D | Wind, nuclear, solar |

The pipeline produces three categories of output:

1. **Cost-optimized resource portfolios** (Steps 1–2): For each (ISO, threshold, cost scenario), the minimum-cost resource mix that achieves the target hourly matching score, evaluated across up to 17,496 cost sensitivity combinations.

2. **Marginal abatement cost (MAC) queues** (Step 3): Path-dependent deployment sequences optimized for cheapest $/tCO₂ avoided, with resource lock-in and clean firm technology tranching.

3. **Market simulation trajectories** (Step 6 — SMARTargets): Forward-looking deployment simulations from 2023–2050 under reference, aspirational, and parametric emission reduction scenarios, incorporating Wright's Law learning curves, REC pricing, capacity markets, and LMP-driven revenue models.

### 1.3 Key Assumptions

- **2025 snapshot model**: Generation profiles and grid mix reflect current conditions. Forward projections are modeled explicitly via demand growth rates and learning curves where applicable (Steps 2.2b, 3b, 6.1).
- **ISO-level geographic resolution**: Resources are sourced within each ISO/RTO region. No intra-ISO transmission constraints or nodal pricing (copper-plate assumption). Transmission costs are modeled as flat $/MWh adders differentiated by resource type and ISO.
- **Hydro is existing-only**: No new hydroelectric capacity is modeled. Existing hydro is available at wholesale market rates with no incremental transmission cost.
- **No incrementality requirement**: Buyers can claim existing clean generation via EAC procurement in the baseline track.
- **Perfect dispatch**: No unit commitment constraints (minimum up/down times, ramp rates, start-up costs). Storage dispatch follows a priority-ordered greedy algorithm (battery4 → battery8 → LDES → H₂).
- **Load profile**: Demand is modeled using actual ISO-level 8,760-hour profiles from EIA-930, representing aggregate load (not a single buyer's load shape).

> *[Insert screenshot of `docs/architecture-high-level.html` here — High-Level System Architecture Diagram]*

---

## 2. Introduction

### 2.1 Document Purpose

This document provides a comprehensive specification of the Hourly CFE Optimizer model as implemented in the production codebase. It serves as:

1. **A technical manual** for team members who will maintain, extend, or audit the model.
2. **A methodology reference** establishing the analytical basis, data provenance, algorithmic choices, and validation results.
3. **A traceability record** documenting how each pipeline step's outputs flow into downstream consumers.

### 2.2 Document Scope

This specification covers pipeline Steps 0–6 (data ingestion through market simulation). Step 7 (dashboard data extraction) is excluded as it is a presentation layer subject to change and does not affect model outputs.

### 2.3 How to Read This Document

- **Section 3** establishes the theoretical framework and analytical basis.
- **Section 4** provides the detailed model architecture, proceeding step-by-step through the pipeline with explicit references to functions, parameters, and data flows. Code blocks for key algorithms are collected in **Appendix A** and referenced inline.
- **Section 5** covers validation, sensitivity analysis, and robustness checks.
- **Section 6** documents limitations and edge cases.
- **Section 7** provides practical usage instructions.

---

## 3. Theoretical Framework

### 3.1 The 8,760-Hour Matching Problem

The central analytical challenge is the temporal mismatch between variable renewable generation and electricity demand. Annual procurement accounting (e.g., purchasing enough RECs to equal annual consumption) masks this mismatch — a buyer may claim 100% clean energy annually while consuming fossil-generated power during nights, winters, and low-wind periods.

The emerging GHG Protocol Scope 2 revision (October 2025 first consultation draft) and SBTi Power Sector v2 framework (September 2025 draft) are moving toward hourly temporal matching requirements. This model evaluates the cost and resource implications of this transition.

**Hourly matching score** is defined as:

$$\text{HMS} = \frac{1}{8760} \sum_{h=1}^{8760} \min\left(1, \frac{\text{clean\_supply}(h)}{\text{demand}(h)}\right)$$

This score represents the fraction of demand met by clean energy in every hour, averaged across the year. A score of 95% means that in the average hour, 95% of demand is met by temporally coincident clean generation. This is strictly more demanding than annual matching, which would count surplus solar in one hour against deficits at night.

### 3.2 Resource Mix Optimization via Exhaustive Search

Unlike linear programming (LP) or mixed-integer programming (MIP) approaches used by capacity expansion models such as GenX[^genx] or EPRI's US-REGEN[^regen], this model uses an **exhaustive combinatorial search** followed by cost-based selection. This approach was chosen because:

1. The hourly matching constraint is non-convex (due to the `min(1, ...)` operator), making LP relaxations unreliable.
2. The search space, while large (~1.6M combinations for 6D CAISO), is tractable with vectorized NumPy operations and memory-bounded chunking.
3. Exhaustive search guarantees global optimality within the grid resolution — no feasible mix is missed due to solver heuristics or starting-point sensitivity.

[^genx]: Jenkins, J.D., et al. "GenX: A Configurable Power System Capacity Expansion Model." MIT Energy Initiative, 2017.
[^regen]: EPRI. "US-REGEN: US Regional Economy, Greenhouse Gas, and Energy Model." EPRI Technical Report, 2023.

The trade-off is grid resolution: at 5% step (coarse) and 1% step (fine), the model cannot find optima between grid points. This is mitigated by the multi-phase search (coarse → zone → fine → storage) that progressively narrows the search space.

### 3.3 Marginal Abatement Cost (MAC) Framework

The MAC queue (Step 3b) adopts a **path-dependent sequential deployment** framework where each threshold's resource deployment is constrained by ("ratcheted to") prior deployments. This reflects real-world dynamics where infrastructure, once built, remains in service.

MAC is computed as:

$$\text{MAC}_t = \frac{\Delta\text{NewBuildCost}_{t-1 \to t}}{\Delta\text{CO}_2\text{Avoided}_{t-1 \to t}}$$

where costs include only new-build LCOE and transmission (no gas backup, no wholesale revenue), and CO₂ avoided is computed from hourly dispatch-based fossil displacement with merit-order retirement (coal → oil → gas). This aligns with the World Bank's marginal abatement cost curve methodology[^worldbank_mac] and is comparable to the incremental cost approach used in EPA's Integrated Planning Model (IPM)[^epa_ipm].

[^worldbank_mac]: World Bank. "State and Trends of Carbon Pricing 2024." Washington, DC, 2024.
[^epa_ipm]: EPA. "Documentation for EPA's Power Sector Modeling Platform v6." Office of Air and Radiation, 2023.

### 3.4 Market Simulation (SMARTargets)

Step 6 employs a **profit-driven deployment simulation** where clean energy resources are deployed wherever revenue exceeds cost, and deployment stops when marginal profit turns negative. This differs from optimization-based capacity expansion models in that:

- CFE level is an **output** (emerges from profitability), not an input constraint.
- Revenue is computed from endogenous LMP (hourly merit-order pricing), capacity markets, and scarcity-driven REC pricing.
- Cost includes Wright's Law deployment-based learning curves, not time-based cost assumptions.
- Emission constraints (AT scenarios) layer on mandated deployment with explicit subsidy tracking.

This approach is conceptually closer to agent-based models of electricity markets than to traditional least-cost capacity expansion.

### 3.5 Wright's Law Learning Curves

Technology cost reductions follow Wright's Law (experience curves) rather than time-based projections:

$$C(Q) = C_{\text{FOAK}} \times \left(\frac{Q}{Q_{\text{ref}}}\right)^{-b}$$

where $Q$ is cumulative deployed capacity (GW), $Q_{\text{ref}}$ is the 2025 baseline, and $b = -\log_2(1 - \text{LR})$ with LR being the learning rate (cost reduction per doubling of cumulative capacity). Technology-specific learning rates are sourced from published empirical literature (see Section 4.2.6).

---

## 4. Model Architecture

> *[Insert screenshot of `docs/architecture-detailed.html` here — Detailed Pipeline Architecture Diagram]*

### 4.0 Shared Analytical Modules

Before describing the pipeline steps, we document the shared modules that multiple steps depend on.

#### 4.0.1 `dispatch_utils.py` — Hourly Dispatch Reconstruction

This module provides the canonical dispatch algorithm used throughout the pipeline. All storage types carry state-of-charge (SOC) across window boundaries and apply round-trip efficiency per discharge event. The 4-phase storage dispatch order is:

1. **Battery 4-hour** (Li-ion, 85% RTE, daily cycling)
2. **Battery 8-hour** (Li-ion, 85% RTE, daily cycling)
3. **LDES 100-hour** (iron-air, 50% RTE, 7-day rolling window)
4. **H₂ 1000-hour** (electrolysis + salt cavern + H₂ turbine, 35% RTE, 30-day rolling window)

Each phase operates on the residual surplus/deficit after prior phases. Surplus clean energy charges storage; deficit hours discharge. The discharge capacity is bounded by `SOC × RTE / duration_hours`. See **Appendix A, Code Block 1** for the core dispatch function.

**Key design decision**: Storage capacity in Steps 1–2.1 is expressed as a percentage of annual demand (energy capacity coefficient). This dimensionless unit enables cross-ISO comparisons and simplifies the combinatorial search. In Step 3, capacity is translated back to physical units for dispatch:

```
MW_capacity = coefficient × annual_MWh / duration_hours
```

This ensures dispatch fidelity matches the capacity model used in cost optimization. The translation is implemented in `reconstruct_hourly_dispatch()` (see Appendix A, Code Block 1).

#### 4.0.2 `pipeline_config.py` — Single Source of Truth

All constants, cost tables, resource caps, and parameters are defined in a single configuration module. Downstream scripts import from here — no local constant definitions are permitted. This prevents the class of bugs where separate scripts disagree on shared values (which historically occurred with `CCS_CAP_TWH`).

#### 4.0.3 `lmp_engine.py` — Merit-Order LMP Pricing

Constructs a fossil merit-order stack (coal → oil → gas units with heat rate curves and emission costs) and computes hourly locational marginal prices (LMP) as the marginal cost of the last fossil unit dispatched in each hour. Used by Steps 4.1a (CO₂/LMP analytics) and 6.1 (SMARTargets revenue model).

#### 4.0.4 `procurement_utils.py` — Buyer-Level Procurement Logic

Provides PPA pricing, RPS target calculations, learning-curve-adjusted LCOEs, SSS (Supplier Specific Sourcing) allocation, and strategy-specific procurement logic for Step 5 buyer scenarios.

### 4.1 Data Engineering & Pipeline

#### 4.1.1 Data Provenance & Cleaning

**Primary data sources:**

| Source | Files | Purpose | Cleaning |
|---|---|---|---|
| **EIA-930 Hourly Electric Grid Monitor** | `eia_generation_profiles.parquet`, `eia_demand_profiles.parquet` | 8,760-hour demand and generation profiles (2021–2025) | Multi-year averaging (5 years) to smooth weather anomalies. DST-aware solar nighttime zeroing. Nuclear monthly CF derate from NRC data. |
| **EIA Form 860/923** | Embedded in grid mix calibration | Generator-level capacity and net generation | Cross-validated against EIA-930 totals. |
| **EPA eGRID** | `egrid_emission_rates.json` | Subregion emission factors by fuel type | Mapped from eGRID subregions to ISO boundaries via BA-to-ISO mapping. |
| **EIA AEO 2025** | Embedded in `pipeline_config.py` | Demand growth rates (Low/Med/High) | Regionalized from national cases using NERC LTRA 2024, ERCOT LTLF 2025, PJM 2025 Load Forecast. |
| **NREL ATB 2024** | Embedded in `pipeline_config.py` | LCOE tables, storage cost projections | 2024 USD. Regional multipliers applied to national ATB estimates. |
| **Lazard LCOE v17-18** | Embedded in `pipeline_config.py` | Cross-validation of LCOE tables | Used as independent check; ATB is primary source. |
| **LBNL "Queued Up" 2025** | Embedded in transmission adder tables | Interconnection queue data, transmission cost estimates | ISO-level aggregation of project-specific transmission costs. |

**Data transformation pipeline** (`eia_data_io.py`):

1. **Parquet ingestion**: Reads EIA-930 data from parquet files (converted from original CSV/JSON).
2. **Generation profile normalization**: Each fuel type's 8,760-hour profile is normalized to sum to 1.0 (representing shape, not magnitude). Resource allocation percentages then scale these shapes to actual TWh.
3. **Multi-year averaging**: Generation profiles are averaged across available years (2021–2025) to reduce single-year weather bias. This is critical for wind (±15% interannual variability) and hydro (±25%).
4. **Demand profile normalization**: Hourly demand is normalized such that `sum(normalized) = 8760` (i.e., each hour's value represents its share of annual demand × 8760).
5. **Fossil mix profiles**: Hourly coal/gas/oil generation shares for merit-order dispatch.

**Where time-series vs. static data are used:**

- **Steps 1–2.1**: Use **static 2025** demand and generation profiles. Resource mixes are evaluated against the 2025 hourly shape only.
- **Step 2.2 (Phase 2 — demand growth sweep)**: Applies **time-varying demand** by year using growth rates from `DEMAND_GROWTH_RATES`. Generation shapes remain 2025 — the assumption is that the *shape* of solar/wind generation is stable (driven by geography/weather), while the *magnitude* scales with deployment.
- **Step 3b (MAC queue)**: Uses **time-varying demand** per SBTi year mapping (`THRESHOLD_TARGET_YEARS`). Each threshold maps to a target achievement year, and demand at that year is computed using compound growth.
- **Step 6 (SMARTargets)**: Uses **time-varying demand** explicitly per simulation year (2023, 2030, 2035, 2040, 2045, 2050). LMP computation uses demand-scaled MW profiles.

### 4.2 Cost & Input Tables

#### 4.2.1 Renewable LCOE ($/MWh)

Source: NREL ATB 2024, regionalized using LBNL installed cost data.

**Solar LCOE ($/MWh):**

| Level | CAISO | ERCOT | PJM | NYISO | NEISO | MISO | SPP |
|---|---|---|---|---|---|---|---|
| Low | 45 | 40 | 50 | 70 | 62 | 48 | 43 |
| Medium | 60 | 54 | 65 | 92 | 82 | 62 | 57 |
| High | 78 | 70 | 85 | 120 | 107 | 82 | 74 |

**Regional adjustment rationale**: Solar costs vary by irradiance (higher CF in ERCOT/SPP → lower LCOE), labor costs (higher in NYISO/NEISO), and permitting complexity. NYISO is 1.75× ERCOT due to NYC-area construction costs and limited greenfield sites. ATB national median is scaled using LBNL's regional installed cost multipliers.

**Wind LCOE ($/MWh):**

| Level | CAISO | ERCOT | PJM | NYISO | NEISO | MISO | SPP |
|---|---|---|---|---|---|---|---|
| Low | 55 | 30 | 47 | 61 | 55 | 33 | 28 |
| Medium | 73 | 40 | 62 | 81 | 73 | 43 | 37 |
| High | 95 | 52 | 81 | 105 | 95 | 56 | 48 |

**Regional adjustment rationale**: Wind LCOE is driven by capacity factor (Class I–IV wind resources). SPP and ERCOT have Class I/II resources (40–50% CF → lowest LCOE). CAISO and NEISO have Class III/IV (25–35% CF → highest LCOE). PJM is intermediate (mix of Appalachian ridge and offshore-adjacent).

**Offshore Wind LCOE ($/MWh):**

| Level | CAISO | PJM | NYISO | NEISO |
|---|---|---|---|---|
| Low | 110 | 65 | 72 | 68 |
| Medium | 150 | 85 | 95 | 90 |
| High | 200 | 112 | 125 | 118 |

CAISO is dramatically higher due to floating technology (no U.S. commercial experience). PJM is cheapest fixed-bottom (shallowest water, NJ 7.5 GW mandate). ERCOT, MISO, SPP have no offshore resource (set to 0).

#### 4.2.2 Clean Firm LCOE ($/MWh)

**Nuclear New-Build LCOE:**

| Level | CAISO | ERCOT | PJM | NYISO | NEISO | MISO | SPP |
|---|---|---|---|---|---|---|---|
| Low | 70 | 68 | 72 | 75 | 73 | 70 | 68 |
| Medium | 95 | 90 | 105 | 110 | 108 | 100 | 92 |
| High | 140 | 135 | 160 | 170 | 165 | 155 | 140 |

Source: NREL ATB 2024 SMR/advanced reactor estimates. Regional variation reflects construction labor costs and regulatory complexity. PJM/NYISO/NEISO are highest due to Northeast construction cost premiums and NRC licensing complexity.

**Nuclear Uprate LCOE**: Low=$15, Medium=$25, High=$40 $/MWh. Capped per ISO based on existing fleet capacity (see `UPRATE_CAP_TWH` in `pipeline_config.py`).

**CCS-CCGT LCOE with 45Q ($/MWh):**

| Level | CAISO | ERCOT | PJM | NYISO | NEISO | MISO | SPP |
|---|---|---|---|---|---|---|---|
| Low | 59.5 | 53.5 | 63.5 | 79.5 | 76.5 | 56.5 | 51.5 |
| Medium | 87.5 | 72.5 | 80.5 | 100.5 | 97.5 | 75.5 | 69.5 |
| High | 116.5 | 93.5 | 103.5 | 129.5 | 123.5 | 97.5 | 89.5 |

45Q credit: $85/ton × 0.323 tCO₂/MWh captured (90% capture × 0.359 tCO₂/MWh unabated) = $27.5/MWh offset. NYISO and NEISO have **zero CCS capacity** due to crystalline bedrock (no geologic CO₂ storage).

**Geothermal LCOE** (CAISO only): Low=$63, Medium=$88, High=$110 $/MWh. Cap: 39 TWh/yr. Source: USGS assessment + Fervo EGS potential.

#### 4.2.3 Storage Costs

Storage costs are expressed as **annualized capacity cost per % of annual demand** — not LCOS. This matches the coefficient model used in Step 1 where storage capacity is parameterized as a fraction of annual demand.

The conversion formula:

```
price = CAPEX_per_kWh × (CRF + FOM_rate) × 1000 × regional_multiplier
```

where CRF = 0.1019 (8% WACC, 20-year life), FOM_rate = 2.5% of CAPEX($/kW) per NREL ATB.

**Battery 4-hour (Li-ion) — sample values ($/% of annual demand):**

| Level | CAISO | ERCOT | PJM |
|---|---|---|---|
| Low | 33,814 | 30,485 | 32,412 |
| Medium | 41,610 | 37,405 | 39,858 |
| High | 52,823 | 47,567 | 50,633 |

**Verification**: 0.01% bat4 at CAISO (224 TWh) = 22,400 MWh. Cost = 0.0001 × 41,610 = $4.16/MWh. Physical: 22.4M kWh × $295/kWh × 0.127 × 1.11 = $924M/yr ÷ 224 TWh = $4.13/MWh. ✓

LDES (100-hour iron-air) and H₂ (1000-hour green hydrogen) follow the same framework with technology-specific CAPEX inputs (Form Energy specifications for LDES, Hydrogen Council 2024 for H₂). Round-trip efficiencies: battery 85%, LDES 50%, H₂ 35%.

#### 4.2.4 Transmission Adders ($/MWh)

Source: LBNL "Queued Up" 2025, ISO interconnection study aggregates.

| Resource | Level | CAISO | ERCOT | PJM | NYISO | NEISO | MISO | SPP |
|---|---|---|---|---|---|---|---|---|
| Wind | Low | 4 | 3 | 5 | 7 | 6 | 5 | 4 |
| Wind | Medium | 8 | 6 | 10 | 14 | 12 | 9 | 7 |
| Wind | High | 14 | 10 | 18 | 22 | 20 | 16 | 12 |
| Solar | Medium | 3 | 3 | 5 | 7 | 6 | 4 | 3 |
| Nuclear | Medium | 3 | 2 | 3 | 5 | 4 | 3 | 2 |
| Offshore Wind | Medium | 20 | 0 | 11 | 15 | 13 | 0 | 0 |

Offshore wind transmission is higher due to submarine cable + offshore substation costs ($1–3M/km, 20–80 km to shore). CAISO floating offshore is highest ($20/MWh at Medium) due to deeper water and longer cable runs. Storage has no separate transmission adder — regional variation is baked into annualized capacity costs.

**NEISO gas constraint**: No explicit gas constraint adder is modeled. Instead, NEISO's higher wholesale price ($41/MWh vs. $27 for ERCOT) and higher LCOE inputs reflect the constrained gas pipeline capacity in New England. This is a limitation — an explicit seasonal gas constraint (winter pricing spikes) would improve accuracy.

#### 4.2.5 Gas Backup Costs

**Existing gas FOM ($/kW-yr):** Ranges from $13 (ERCOT, SPP) to $17 (NYISO). Represents fixed O&M for maintaining existing gas capacity as backup.

**New gas CCGT LCOE ($/MWh):** Low=$45, Medium=$55, High=$65. Source: Lazard v17-18, EIA AEO 2024, NREL ATB 2024.

**Capacity market prices ($/kW-yr):** PJM=$120 (highest, RPM auction), CAISO=$75 (RA program), NYISO=$85 (ICAP), NEISO=$55 (FCM), MISO=$25. ERCOT and SPP are energy-only markets ($0 capacity payment).

**Capacity market degradation**: As clean share rises, capacity prices fall: `cap_price(t) = base_price × max(0, 1 − α × clean_share)`. Alpha ranges from 0.35 (PJM, NEISO) to 0.40 (CAISO, NYISO).

#### 4.2.6 FOAK/NOAK Costs & Learning Curves

**First-of-a-kind (FOAK) costs** represent pre-learning-curve commercial-scale project costs:

| Technology | Multiplier over High | Example: PJM FOAK |
|---|---|---|
| Nuclear new-build | ~1.25× | $200/MWh |
| CCS-CCGT (45Q on) | ~1.20× | $122/MWh |
| Geothermal (CAISO) | ~1.35× | $150/MWh |
| LDES | ~1.40× | $11,563/unit |
| Offshore wind (fixed) | ~1.15× | $129/MWh (PJM) |
| Offshore wind (floating) | ~1.25× | $250/MWh (CAISO) |

**Wright's Law learning parameters** (Step 6):

| Technology | Learning Rate (Fast/Slow) | Cumulative GW Baseline (2025) |
|---|---|---|
| Nuclear SMR | 15% / 10% | 2.0 GW |
| CCS | 12% / 10% | 0.3 GW |
| LDES (iron-air) | 20% / 15% | 0.01 GW |
| Battery (Li-ion) | 20% / 18% | 50.0 GW |
| Offshore wind | 12% / 8% | 5.0 GW |
| Solar | 0% / 0% (mature) | 150.0 GW |
| Wind (onshore) | 0% / 0% (mature) | 150.0 GW |

Sources: Solar module LR ~20% (Swanson's Law; Our World in Data 2023). Battery Li-ion 18–20% (BloombergNEF 2024). Nuclear SMR 10–15% estimated (DOE Liftoff 2023). CCS 10–12% (Global CCS Institute). Note: Constant LR assumption is a simplification — recent literature (ScienceDirect Oct 2025) finds stepwise LR changes for 58/87 technologies studied.

#### 4.2.7 Demand Growth Rates

Source: EIA AEO 2025 (Reference + High/Low Economic Growth), NERC 2024 LTRA, ERCOT 2025 LTLF, PJM 2025 Load Forecast, Grid Strategies 2025.

| ISO | Low | Medium | High |
|---|---|---|---|
| CAISO | 1.4% | 1.9% | 2.5% |
| ERCOT | 2.0% | 3.5% | 5.5% |
| PJM | 1.5% | 2.4% | 3.6% |
| NYISO | 1.3% | 2.0% | 4.4% |
| NEISO | 0.9% | 1.8% | 2.9% |
| MISO | 1.2% | 2.2% | 3.8% |
| SPP | 1.0% | 1.8% | 3.0% |

Low = baseline economic/population growth. Medium = confirmed large-load requests + moderate electrification. High = full data center/AI load growth + accelerated electrification.

### 4.3 Algorithm Selection & Core Mathematical Functions

#### 4.3.1 STEP 1.1a/b — Coarse Grid Sweep & Scoring

**Script**: `step1_1a_generate_mixes.py`, `step1_1b_score_mixes.py`

**Algorithm**: Cartesian product of resource fractions at 5-percentage-point step for each ISO's resource dimensions. For a 4D ISO (ERCOT), this produces ~12,000 combinations; for 6D CAISO, ~1.6 million.

**Function**: `generate_resource_combos(iso, step=5)` in `step1_pfs_generator.py` generates all valid combinations where resource percentages sum to at most the total procurement cap (350%). Hydro is capped at existing levels per ISO.

**Scoring**: `batch_hourly_scores()` in `step1_pfs_generator.py` performs vectorized 8,760-hour matching. For each mix, the clean supply profile is constructed by weighting each resource's normalized generation shape by its allocation percentage, then computing the hourly matching score (see Section 3.1). Scoring is done in memory-bounded chunks of 20,000 mixes to keep peak memory at ~1.4 GiB.

**Prior windows**: When available from prior runs (via `step1_prior_windows.py`), the Cartesian product is narrowed to the union of prior EF bounds ± 15 percentage points, plus 100 scout mixes to detect regime shifts. This typically reduces the search space by ~30%.

**Output**: `{ISO}_coarse_cache.parquet` — each row is a resource mix with its hourly matching score.

#### 4.3.2 STEP 1.2 — Zone-Based Fine Search

**Script**: `step1_2_zone_search.py`

**Algorithm**: Divides the score space into three zones based on score ranges:
- **Zone A** (50–70%): Coarse grid typically sufficient; moderate fine-grid expansion.
- **Zone B** (70–90%): Inflection zone where resource mix diversity matters most.
- **Zone C** (90–100%): Last-mile zone requiring precise resource balancing.

For each zone:
1. Identify coarse boundary mixes within the zone's score range.
2. Compute zone-specific resource windows (bounds on each resource dimension).
3. Generate a 1% fine grid within the zone bounds (step size controlled by `FINE_STEP = 1`).
4. Deduplicate against a global hash set — no mix is scored twice across zones.
5. Score all new mixes via `batch_hourly_scores()`.
6. Assign scored mixes to relevant thresholds and apply dominance filtering.

**Global deduplication** uses a collision-free integer hash: `key = Σ(round(resource_i) × 301^i)` for up to 7 resource dimensions. This avoids string hashing overhead while guaranteeing uniqueness for resource values in [0, 300].

**Safety caps**: `MAX_FINE_ARCHETYPES = 2,000` (4D ISOs), `MAX_FINE_ARCHETYPES_5D = 500`. If the fine grid exceeds 10M combinations, the search falls back to archetype-based expansion around boundary mixes.

**Output**: `{ISO}_t{T}_raw_pfs.parquet` (per-threshold feasible mixes), `{ISO}_near_miss.parquet` (union near-miss list for Step 1.5).

#### 4.3.3 STEP 1.3 — Floor-Aware PFS

**Script**: `step1_3_floor_aware_pfs.py`

**Algorithm**: Generates resource mixes that start from the **existing clean resource floor** (per `GRID_MIX_SHARES` in `pipeline_config.py`) and add incremental resources. This is critical for MAC accuracy because the standard PFS (Step 1.2) generates full portfolios that may include large resource allocations unrelated to the existing grid.

**Grid generation**:
- Solar additions: 0–80% above existing (2% step)
- Wind additions: 0–80% above existing (2% step)
- Clean firm additions: 0–40% above existing (2% step)
- Hydro: fixed at existing (capped, $0 — no new hydro)
- Offshore wind: 0–30% (5% step, offshore ISOs only)
- Geothermal: 0–20% (5% step, CAISO only)

**Target thresholds**: 50–70%. Produces mixes representing the cheapest incremental build above existing infrastructure.

**Output**: `{ISO}_t{T}_floor_pfs.parquet` (per threshold).

#### 4.3.4 STEP 1.4 — Fine Grid PFS

**Script**: `step1_4_fine_grid_pfs.py`

**Algorithm**: Fills coverage gap in the standard PFS for 40–70% thresholds using a finer 1% grid (vs. Step 1.3's 2% grid). Same floor-aware approach with slightly tighter resource bounds: solar 0–60%, wind 0–60%, clean firm 0–30% additions.

**Output**: `{ISO}_t{T}_fine_pfs.parquet`.

#### 4.3.5 STEP 1.5 — Storage Refinement ★

**Script**: `step1_5_storage_refinement.py`

This step identifies resource mixes that fail to meet target thresholds through generation alone but can reach them with storage dispatch. It is the most computationally intensive sub-step and uses a **three-pass adaptive funnel** to minimize compute while preserving accuracy.

**Pass 0 — Maximum Screen** (~40s per ISO):
Score each near-miss mix with ceiling storage levels (battery4=0.10%, battery8=0.15%, LDES=1.0%, H₂=1.0% of annual demand). Eliminates mixes that cannot reach any active threshold even with maximum storage. This is a single combination per mix — fast screening.

**Pass 1 — Adaptive Coarse Sweep**:
Group surviving mixes by gap-to-threshold into buckets (0–5pp, 5–10pp, 10–25pp, 25–50pp). Each bucket receives a right-sized storage grid:
- Battery 4hr: [0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.10]%
- Battery 8hr: [0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.15]%
- LDES: [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]%
- H₂: [0, 0.3, 1.0]%

The Cartesian product of these grids across 4 storage dimensions creates the full parametric sweep. Upper limit per threshold: the number of surviving near-miss mixes × the storage grid size for their gap bucket. See **Appendix A, Code Block 2** for the storage sweep function.

**Pass 2 — Fine Targeted** (0.05% resolution):
For mixes near each threshold's storage-enhanced boundary, refine storage parameters at fine resolution.

**Dominance filter**: After each pass, mixes where **all** resource and storage allocations exceed those of a mix that already achieves the threshold are eliminated. This prevents dominated solutions from propagating into Step 2.

**Floor/fine augmentation**: Loads floor_pfs (Step 1.3) and fine_pfs (Step 1.4) parquets. Filters to mixes within the near-miss window, deduplicates, and runs the storage sweep. This exposes lean floor+storage combinations that the standard near-miss pool (biased toward high-procurement mixes) would miss.

**Output**: `{ISO}_t{T}_storage.parquet` (per threshold, auto-batched if large).

#### 4.3.6 STEP 2.1 — Efficient Frontier Extraction

**Script**: `step2_1_efficient_frontier.py`

**Algorithm**: Three-phase reduction from the union of all Step 1 outputs.

**Phase 1 — Threshold Gate**: Keep only rows whose hourly matching scores fall within the relevant threshold range. A mix with score 87.3% is relevant for thresholds ≤ 87.5%.

**Phase 2 — Resource Cap Filter**: Enforce physics and policy constraints:
- Solar cap: 100% of demand
- Total procurement cap: 350% of demand (accounts for over-procurement needed for hourly matching)
- Hydro cap: Match existing levels per ISO (filters out Step 1's +10pp exploratory hydro mixes)

**Phase 3 — Global Deduplication**: Drop the threshold column. For each unique resource allocation (ISO/CF/Sol/Wnd/Hyd/Geo/Bat/Bat8/LDES), keep only the row with the highest hourly_match_score. Each unique physical configuration is stored **once**.

**Design decision**: No dominance removal across different resource mixes is performed at this stage. Different resource mixes at the same score/storage can have very different costs under different LCOE assumptions — removing them risks losing true cost optimums. Cost-based selection happens in Step 2.2.

Output is partitioned into non-overlapping threshold bands based on score. Each mix appears in exactly one file. Step 2.2 loads bands ≥ its target threshold to reconstruct the qualifying set.

**Output**: `step_2_1_EF_{ISO}_{T}.parquet` (per ISO × threshold band).

**Traceability**: Step 2.1 outputs feed directly into → **Step 2.2a** (cost optimization) and **Step 2.2b** (track evaluation).

#### 4.3.7 STEP 2.2a — Cost Optimization

**Script**: `step2_2a_cost_optimization.py`

**Algorithm**: For each (ISO, threshold):

1. Load all EF mixes with score ≥ threshold (loading from multiple band files).
2. For each of the sensitivity combinations:
   - Non-CAISO: 3³ × 3 × 3 × 3 × 2 × 3 = 5,832 combos
   - CAISO: 5,832 × 3 (geothermal) = 17,496 combos
   - **9-dimension sensitivity key**: `{Ren}{Firm}{Batt}{LDES}_{Fuel}_{Tx}_{CCS}{45Q}_{Geo}`
3. **Vectorized cost evaluation**: All mixes are evaluated simultaneously using NumPy broadcasting. For each mix, cost = Σ(resource_pct × LCOE × demand + tx_adder) + storage_cost + gas_backup_cost − wholesale_revenue.
4. Select the cheapest mix per sensitivity combo.

**Phase 2 — Demand Growth Sweep**: Extract unique winning archetypes (deduplicated winning mixes across all sensitivities). For each (year, demand growth level), evaluate archetypes with:
- Demand scaled by compound growth rate
- Costs adjusted via Wright's Law learning curves (`year_adjusted_cost()` in `pipeline_config.py`)

**Gas backup model**: At each clean energy threshold, residual demand must be met by gas (existing + new-build). Resource adequacy margin is enforced. Existing gas FOM is charged on used capacity; new gas CCGT capital is charged on new-build capacity. Peak capacity credits for clean resources (ELCC-based) reduce gas backup requirements.

**Output**: `step_2_2a_CO_{ISO}.parquet` — one row per (ISO, threshold, sensitivity, year, growth level) with the winning mix, its cost, and all resource allocations.

**Traceability**: Step 2.2a outputs feed into → **Step 3a** (dispatch cache for winning mixes), **Step 3b** (MAC queue optimization), **Step 4** (all derived analytics), and **Step 6** (SMARTargets reads step3 data for resource mixes).

#### 4.3.8 STEP 2.2b — Track NB/CTR

**Script**: `step2_2b_track_nb_ctr.py`

Evaluates two alternative procurement tracks:

- **New-Build (NB) Track**: Only new resources count (no existing clean credit). Represents a buyer building an entirely new clean portfolio.
- **Cost-to-Replace (CTR) Track**: Evaluates the cost of replacing existing nuclear fleet with new clean resources. Tests the incremental cost of substituting firm clean generation.

Each track runs the same demand growth sweep and learning curve logic as Step 2.2a Phase 2, producing P10/P50/P90 cost envelopes.

**Output**: `track_scenarios.parquet`.

**Traceability**: → **Step 4.1e** (export tracks) → **Step 4.2c** (analyze tracks) → dashboard.

#### 4.3.9 STEP 3a — Dispatch Cache

**Script**: `step3a_build_dispatch_cache.py`

Pre-computes full 8,760-hour dispatch for every unique resource mix from Step 2.2. Uses `dispatch_utils.reconstruct_hourly_dispatch(detailed=True)` to produce per-resource matched/surplus breakdowns and storage charge/discharge profiles.

**Battery capacity translation**: Step 1–2.1 storage is in % of annual demand (energy capacity coefficient). Step 3 translates back to physical dispatch units:

```
MW_capacity = (battery_dispatch_pct / 100) × total_annual_MWh / duration_hours
```

For example, 0.01% battery at CAISO (224 TWh = 224,039,000 MWh): MW = 0.0001 × 224,039,000 / 4 = 5,601 MW, which is comparable to CAISO's real 10 GW+ storage fleet.

**Archetype key**: Each unique mix is identified by a string key encoding all resource and storage percentages (e.g., `"cf15_s25_w10_h5_b0.01_l0.05"`). This key is used for cache lookup throughout Steps 4–6.

**Output**: `{ISO}_dispatch_cache.parquet`, `{ISO}_annual_manifest.parquet`.

**Traceability**: → **Step 4.1a** (CO₂ + LMP), **Step 4.1b** (day profiles), **Step 4.2b** (storage analysis), **Step 5.2a** (scenario comparison dispatch lookup).

#### 4.3.10 STEP 3b — MAC Queue

**Script**: `step3b_mac_queue.py`

**Algorithm**: Path-dependent deployment queue optimizing for cheapest $/tCO₂ avoided.

For each ISO × price_sensitivity (5) × demand_growth (3) = 15 pathways:

1. **Floor initialization**: Start with existing clean resources (`GRID_MIX_SHARES`).
2. **At each threshold T** (mapped to SBTi year via `THRESHOLD_TARGET_YEARS`):
   a. Compute demand at year Y with compound growth.
   b. Dispatch floor resources → compute path-dependent CO₂ baseline.
   c. Sample archetypes from PFS (raw, fine, floor, storage) that **respect the floor** (no resource can be lower than the prior threshold's locked-in value).
   d. Score: `new_build_cost / CO₂_avoided` ($/tCO₂). Only new-build cost is counted — existing resources at $0.
   e. Winner = argmin(MAC) with overshoot constraint ≤ 1%.
   f. Phase 2: refine around best archetypes at finer grid.
   g. **Ratchet**: Lock winner's resource allocations as the new floor for the next threshold.

**Clean firm tranching** (merit-order within the "clean firm" category):
1. **Tranche 1**: Nuclear uprates (existing fleet, cheapest, capped per ISO)
2. **Tranche 2**: Geothermal (CAISO only, capped at 39 TWh minus existing)
3. **Tranche 3**: min(nuclear new-build, CCS) — CCS capped per ISO

This is implemented in `compute_clean_firm_tranches()` in `pipeline_config.py` (see **Appendix A, Code Block 3**).

**CO₂ model**: Merit-order fossil retirement: coal retires first (>90% threshold), then oil, then gas. Emission rates are from EPA eGRID. CCS has a residual emission rate of 0.036 tCO₂/MWh (10% of unabated CCGT at 90% capture).

**Consequential queue export**: Results are sorted by marginal MAC across all ISOs, producing a cross-regional deployment queue (`consequential_queue.json`) that shows the cheapest order to decarbonize across the entire US.

**Output**: `mac_queue_{ISO}.parquet`, `mac_queue_summary.json`, `consequential_queue.json`, `consequential_queue_scenario_a.json`, `scenario_a_{ISO}.json`.

**Traceability**: → **Step 5.2a** (scenario comparison reads scenario_a files and queue), → **Step 5.2b** (strategy 1 uses MAC queue ordering).

#### 4.3.11 STEP 4 — Derived Analytics

**Step 4.1a — Fossil Dispatch (CO₂ + LMP)**
`step4_1a_fossil_dispatch.py`

Reads clean dispatch from Step 3a cache. Computes `residual_demand[h] = max(0, demand[h] − clean[h])`. Fills residual with merit-order fossil stack (coal → oil → gas). CO₂ = Σ(fossil_dispatch[h] × emission_rate). LMP = marginal cost of last fossil unit dispatched per hour. Also computes capacity market revenue with ELCC-degraded pricing.

**Step 4.1b — Compressed Day Profiles**
`step4_1b_compress_day_profiles.py`

Replays 8,760-hour dispatch → compresses to 24-hour representative day (hour-of-day annualized sums). Used for dashboard visualization.

**Step 4.1c — MAC Statistics**
`step4_1c_compute_mac_stats.py`

Computes marginal MAC = d(TotalCost)/d(CO₂) using PCHIP monotone cubic splines between discrete thresholds. Crosses 3 grid cost × 3 DAC cost scenarios = 9 crossover points. Produces no-regrets analysis (minimum resource shares across all scenarios).

**Step 4.1d — Optimal Targets**
`step4_1d_compute_optimal_targets.py`

Identifies where marginal grid abatement cost equals DAC cost → the economically optimal clean energy target per ISO. Produces range [min crossover, max crossover].

**Step 4.1e — Export Tracks** → **Step 4.2c — Analyze Tracks**
Dependency chain: Step 2.2b produces track parquets → Step 4.1e exports to JSON → Step 4.2c computes P10/P50/P90 cost envelopes and resource mix differentials → dashboard.

**Step 4.1f — Building Blocks**
Extracts resource mix compositions and 24-hour generation shapes using Step 1's exact profiles.

**Step 4.2a — Resource Density**
Reads Step 2.2 results for thresholds within each ISO's optimal crossover range. Computes new-build TWh per resource per cost scenario for strip-plot visualization.

**Step 4.2b — Storage Analysis**
Extracts storage dispatch metrics from Step 3a dispatch cache. Analyzes battery charge/discharge timing, LDES multi-day bridging, H₂ seasonal storage, and how storage need scales with threshold.

#### 4.3.12 STEP 5 — Procurement Strategy Evaluation

**Step 5.1 — Scenario B (Hourly Matching)**
`step5_1_scenario_hourly.py`

Models GHG Protocol hourly Scope 2 incentive-driven grid buildout. Four supply pools:
1. **SSS** ($0): Policy-supported nuclear ZECs, public hydro, RPS mandates
2. **Contracted** (excluded): Locked via hyperscaler PPAs
3. **Existing merchant**: Solar/wind/merchant nuclear at EAC premium ($3–5/MWh)
4. **New-build**: Investment signal from hourly matching gap

Learning curve: Scenario B (accelerated) — FOAK→NOAK by 2040, reflecting early investment in firm clean.

**Output**: `scenario_b_{ISO}.json`

**Step 5.2a — Scenario Comparison**
`step5_2a_scenario_comparison.py`

Reads Scenario A from Step 3b (`data/step3-dispatch/mac_queue/scenario_a_{ISO}.json`) and Scenario B from Step 5.1 (`data/step5-scenarios/scenario_b_{ISO}.json`). Produces side-by-side comparison: resource mix, cost, gas backup, MAC trajectories, stranding analysis, domino sequence (cross-ISO MAC ladder).

**Step 5.2b — Strategy 1 (Consequential Netting)**
`step5_2b_strategy_consequential.py`

Three variants: 1A grid-average baseline, 1B fossil-average, 1C marginal. Cross-regional procurement (cheapest $/tCO₂ anywhere). Scenario A learning curve (delayed).

**Step 5.2c — Strategy 2 (Hourly Matching)**
`step5_2c_strategy_hourly.py`

Three variants: 2A 100% new-build, 2B grid baseline, 2C SSS+premium tranches. Same-ISO, hourly temporal matching. Scenario B learning (accelerated).

**Step 5.2d — Strategy 3 (Annual Bundled)**: Annual matching with bundled RECs.

**Step 5.2e — Wright's Law Curves**: Exports FOAK→NOAK trajectories for dashboard.

#### 4.3.13 STEP 6 — SMARTargets Market Simulation

**Script**: `step6_1_smartargets.py`

**Architecture**: Market-driven simulation where clean energy resources deploy where profitable and stop when marginal profit ≤ 0. CFE level is an output, not an input.

**Scenario classes**:
- **R1/R2 (Reference)**: Pure profit-driven. R1=Facilitating (fast learning, large queue, low costs), R2=Challenging (slow learning, small queue, high costs).
- **AT1–AT4 (Aspirational Transition)**: Profit-driven first, then mandated deployment to meet emission constraint. Power NZ and Economy NZ trajectories. Facilitating variants include DAC backstop; Challenging variants must overbuild grid.
- **QT1–QT4 (Quick Transition)**: Parametric sweep across 19 emission reduction targets (5%–95% in 5% steps).

**Revenue model** (per zone/threshold step):
1. **Energy revenue**: Hourly LMP from `lmp_engine.py` × resource generation profile (generation-weighted LMP).
2. **Capacity market revenue**: ISO-specific prices degraded by clean share (ELCC-based capacity credits).
3. **REC revenue**: Scarcity-driven compliance REC pricing: `price = ACP × (1 − exp(−k × gap%))` where gap = RPS target − eligible clean %. Calibrated per ISO against 2025 observed compliance REC prices. Voluntary corporate demand adder amplifies scarcity.

**Cost model** (per zone/threshold step):
1. **LCOE**: From `pipeline_config.py` tables, adjusted by Wright's Law deployment-based learning.
2. **Transmission**: Per-resource, per-ISO adders.
3. **PPA discount**: Regional market depth scaling (ERCOT=1.0 → SPP=0.50).

**Stepping loop**: For each year (2023, 2030, 2035, 2040, 2045, 2050):
1. 2023 injects actual eGRID baseline (identical for all scenarios).
2. For each ISO: iterate through candidate thresholds above current clean%. Compute revenue and cost at each threshold step. Deploy if profit > 0; stop at first unprofitable step.
3. For constrained scenarios (AT/QT): after profit-driven deployment, enforce emission cap via mandated deployment at a loss (tracked as subsidy), DAC backstop (facilitating), or queue overshoot premium (challenging).
4. Queue cap: Annual interconnection limit per ISO (e.g., CAISO: 6 GW facilitating, 3 GW challenging).

**Parametric sweep mode**: 270 scenarios = 2 conditions × 3 demand × 5 price × 3 PPA × 3 gas friction. Shared LMP cache across scenarios to avoid redundant 8,760-hour computation.

**Output**: `smartargets_{scenario}.parquet`, `sweep_{type}_{ISO}.parquet`.

---

## 5. Validation Results

### 5.1 Physics Validation

- **Hourly matching scores**: Verified that existing grid mixes produce scores consistent with eGRID clean energy shares (within ±2pp). CAISO existing clean (48.5%) produces HMS ~48.0%.
- **Storage dispatch conservation**: Verified that total energy out of storage ≤ total energy in × RTE for all storage types across all mixes. SOC is non-negative at all hours.
- **Hydro cap enforcement**: No feasible mix exceeds ISO-specific hydro caps in the cost optimization stage (Step 2.1 Phase 2 filter).

### 5.2 Cost Validation

- **Battery cost cross-check**: 0.01% bat4 at CAISO, Medium = $4.16/MWh from model. Physical calculation: 22.4M kWh × $295/kWh × CRF × regional = $4.13/MWh. Error: 0.7%.
- **LCOE benchmarks**: Model LCOE tables validated against NREL ATB 2024 medium case within ±5% for solar, wind, battery. Nuclear new-build validated against DOE Liftoff 2023 estimates.

### 5.3 Sensitivity Analysis

The 9-dimensional sensitivity sweep (5,832–17,496 combinations per threshold) systematically varies:

1. **Renewable LCOE** (L/M/H) — tests solar/wind cost uncertainty
2. **Firm clean LCOE** (L/M/H) — tests nuclear/CCS cost uncertainty
3. **Battery cost** (L/M/H) — tests storage cost uncertainty
4. **LDES cost** (L/M/H) — tests long-duration storage
5. **Fuel prices** (L/M/H) — tests gas/coal price scenarios
6. **Transmission** (L/M/H) — tests interconnection cost uncertainty
7. **CCS availability** (L/M/H) — tests CCS cost with/without 45Q
8. **45Q credit** (on/off) — tests IRA tax credit policy
9. **Geothermal** (L/M/H, CAISO only) — tests EGS development costs

P10/P50/P90 cost envelopes from Step 4.2c capture the full uncertainty range.

### 5.4 MAC Validation

MAC curves are compared against published estimates:
- $0–50/tCO₂ for first 50–70% clean (wind/solar in wind-rich ISOs) — consistent with BNEF Global LCOE tracker.
- $100–300/tCO₂ for 90–95% clean (firm clean deployment) — consistent with Princeton Net Zero America estimates.
- $500+ for 99%+ (last-mile firm + storage) — consistent with DOE Liftoff report estimates for deep decarbonization.

---

## 6. Usage & Limitations

### 6.1 How to Interpret Output

**Cost results** represent the theoretical minimum-cost resource portfolio under specified assumptions. Real-world procurement involves additional factors — developer availability, interconnection timelines, contract structure, counterparty risk — that may shift actual portfolios from the modeled optimum.

**MAC results** show the incremental cost of each additional unit of CO₂ abatement. The "no regrets" zone identifies clean energy investments that are cheaper than DAC under all cost scenarios.

**SMARTargets results** show market-driven deployment trajectories, not mandated or optimized pathways. They answer: "What would the grid look like if clean energy deployed wherever it was profitable?"

### 6.2 Known Limitations

1. **Static supply model**: Does not account for price-induced supply responses. High EAC prices would stimulate new clean energy investment in reality.

2. **No cross-ISO interactions**: Each ISO modeled independently. Scarcity in one region could drive inter-regional EAC trade or load migration.

3. **No intra-ISO transmission constraints**: Copper-plate assumption. Production models like GenX (DC optimal power flow) and EPRI's US-REGEN (zonal pipe-and-bubble) capture congestion-driven price separation within ISOs.

4. **No unit commitment constraints**: Perfect dispatch assumed. No minimum up/down times, ramp rate limits, start-up costs, or minimum stable output for thermal generators. This may understate integration costs at high renewable penetration.

5. **Reserve margin without hourly reserves**: Resource adequacy margin is enforced, but spinning, non-spinning, and regulation reserves are not modeled as hourly dispatch constraints.

6. **No demand-side flexibility**: Load is perfectly inelastic. No demand response, load shifting, or flexible consumption modeled.

7. **Single-sector scope**: Electricity-only. Cross-sector interactions not captured.

8. **Policy evolution**: Reflects current policy as of early 2025. RPS targets, federal tax credits, and GHG Protocol requirements continue to evolve.

9. **Interconnection queue constraints**: New capacity assumed buildable as needed (except in SMARTargets which models queue caps).

### 6.3 Future Enhancements

**Tier 1 — High Impact, Feasible**: Zonal transmission modeling (pipe-and-bubble, 2–5 zones per ISO), operating reserve & unit commitment, demand-side flexibility. Informed by EPRI US-REGEN and GenX.jl architectures.

**Tier 2 — Medium Impact**: Cross-ISO EAC trade, endogenous capacity retirement, flexible CCS dispatch.

**Tier 3 — Aspirational**: DC optimal power flow, multi-stage pathway optimization, multi-sector energy integration.

---

## 7. Directions for Use

### 7.1 Environment Setup

```bash
pip install numpy numba pyarrow pandas
# Verify Numba:
python3 -c "from numba import njit; print('Numba OK')"
```

### 7.2 Running the Pipeline

The pipeline runs as a directed acyclic graph (DAG). Sub-step notation: numbers = sequential, letters = parallel.

```bash
# Step 1: Physics Feasible Space (~3-8 hours)
python scripts/step1_1a_generate_mixes.py --iso ALL
python scripts/step1_1b_score_mixes.py --iso ALL
python scripts/step1_2_zone_search.py --iso ALL
python scripts/step1_3_floor_aware_pfs.py --iso ALL
python scripts/step1_4_fine_grid_pfs.py --iso ALL
python scripts/step1_5_storage_refinement.py --iso ALL

# Step 2: Optimization (~15-30 min)
python scripts/step2_1_efficient_frontier.py --iso ALL
python scripts/step2_2a_cost_optimization.py --iso ALL
python scripts/step2_2b_track_nb_ctr.py --iso ALL

# Step 3: Dispatch & MAC (~10-30 min)
python scripts/step3a_build_dispatch_cache.py
python scripts/step3b_mac_queue.py

# Step 4: Analytics (~5-10 min, all parallel)
python scripts/step4_1a_fossil_dispatch.py
python scripts/step4_1b_compress_day_profiles.py
python scripts/step4_1c_compute_mac_stats.py
python scripts/step4_1d_compute_optimal_targets.py
python scripts/step4_1e_export_tracks.py
python scripts/step4_1f_extract_building_blocks.py

# Step 5: Scenarios (~10-20 min)
python scripts/step5_1_scenario_hourly.py
python scripts/step5_2a_scenario_comparison.py
python scripts/step5_2b_strategy_consequential.py
python scripts/step5_2c_strategy_hourly.py
python scripts/step5_2d_strategy_annual.py

# Step 6: SMARTargets (20-60 min; sweep: hours)
python scripts/step6_1_smartargets.py
python scripts/step6_1_smartargets.py --sweep reference
```

### 7.3 Single-ISO Runs

Most scripts support `--iso` flags for running a single ISO:

```bash
python scripts/step1_1a_generate_mixes.py --iso PJM
python scripts/step2_2a_cost_optimization.py --iso CAISO
python scripts/step6_1_smartargets.py --scenarios R1 --isos ERCOT
```

### 7.4 Output Locations

| Step | Output Directory | Format |
|---|---|---|
| 0 | `data/eia-930/` | Parquet |
| 1 | `data/step1-pfs/` | Parquet |
| 2.1 | `data/step2.1-ef/` | Parquet |
| 2.2 | `data/step2.2-cost/` | Parquet |
| 3 | `data/step3-dispatch/` | Parquet + JSON |
| 4 | `data/step4-analysis/` | Parquet + JSON |
| 5 | `data/step5-scenarios/` | JSON |
| 6 | `data/step6-smartargets/` | Parquet + JSON |

---

## Appendix A — Key Algorithm Code Blocks

### Code Block 1: Hourly Dispatch Reconstruction (`dispatch_utils.py`)

The core 4-phase storage dispatch function. All storage types carry SOC across window boundaries with round-trip efficiency per discharge event. This is the single source of truth for dispatch logic — used by Steps 1, 3, 4, and 6.

```python
def reconstruct_hourly_dispatch(demand_norm, supply_profiles, resource_pcts,
                                 procurement_pct, battery_dispatch_pct=0,
                                 battery8_dispatch_pct=0, ldes_dispatch_pct=0,
                                 h2_dispatch_pct=0, detailed=False):
    """Reconstruct 8760-hour dispatch with 4-phase storage."""
    H = 8760
    total_clean = np.zeros(H)
    for res, pct in resource_pcts.items():
        if pct > 0 and res in supply_profiles:
            total_clean += np.array(supply_profiles[res]) * (pct / 100.0)
    
    # Phase 1: Battery 4hr
    if battery_dispatch_pct > 0:
        total_clean = _apply_storage(total_clean, demand_norm,
                                      battery_dispatch_pct, BATTERY_EFFICIENCY,
                                      BATTERY_DURATION_HOURS, window_days=1)
    # Phase 2: Battery 8hr
    if battery8_dispatch_pct > 0:
        total_clean = _apply_storage(total_clean, demand_norm,
                                      battery8_dispatch_pct, BATTERY8_EFFICIENCY,
                                      BATTERY8_DURATION_HOURS, window_days=1)
    # Phase 3: LDES 100hr
    if ldes_dispatch_pct > 0:
        total_clean = _apply_storage(total_clean, demand_norm,
                                      ldes_dispatch_pct, LDES_EFFICIENCY,
                                      LDES_DURATION_HOURS, window_days=7)
    # Phase 4: H2 1000hr
    if h2_dispatch_pct > 0:
        total_clean = _apply_storage(total_clean, demand_norm,
                                      h2_dispatch_pct, H2_EFFICIENCY,
                                      H2_DURATION_HOURS, window_days=30)
    
    matched = np.minimum(total_clean, demand_norm * (procurement_pct / 100.0))
    surplus = total_clean - matched
    gap = demand_norm * (procurement_pct / 100.0) - matched
    
    return {'matched': matched, 'surplus': surplus, 'gap': gap,
            'total_clean': total_clean}
```

*Note: Simplified for readability. See `dispatch_utils.py` for the full implementation including detailed per-resource breakdowns and Numba JIT acceleration.*

### Code Block 2: Storage Sweep — Pass 1 Adaptive Grid (`step1_5_storage_refinement.py`)

The adaptive coarse sweep groups near-miss mixes by gap size and assigns proportionally sized storage grids:

```python
# Gap bucket boundaries (percentage points)
GAP_BUCKET_PP = [5, 10, 25, 50]

# Full storage grids (% of annual demand)
FULL_BAT4 = [0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.10]
FULL_BAT8 = [0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.15]
FULL_LDES = [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
FULL_H2   = [0, 0.3, 1.0]

# For each gap bucket, cap storage at ~3x the max gap
# Small gap (0-5pp): only needs small storage → prune large levels
# Large gap (25-50pp): needs full grid including heavy storage
```

### Code Block 3: Clean Firm Tranching (`pipeline_config.py`)

Merit-order assignment within the "clean firm" resource category:

```python
def compute_clean_firm_tranches(new_cf_twh, iso, firm_lev, ccs_lev, q45,
                                 tx_name='Medium', geo_lev=None,
                                 geo_physics_new_twh=0):
    """Split new clean firm TWh into cost-ordered tranches."""
    # Tranche 1: Nuclear uprates (cheapest, capped)
    uprate_cap = UPRATE_CAP_TWH.get(iso, 0)
    uprate_twh = min(new_cf_twh, uprate_cap)
    remaining = new_cf_twh - uprate_twh
    
    # Tranche 2: Geothermal (CAISO only, capped)
    geo_twh = 0
    if iso == 'CAISO' and geo_lev and remaining > 0:
        geo_cap = GEOTHERMAL_CAP_TWH - geo_physics_new_twh
        geo_twh = min(remaining, max(0, geo_cap))
        remaining -= geo_twh
    
    # Tranche 3: min(nuclear new-build, CCS) — CCS capped per ISO
    ccs_cap = CCS_CAP_TWH.get(iso, 0)
    nuclear_lcoe = NUCLEAR_NEWBUILD_LCOE[firm_lev][iso] + get_tx('clean_firm', tx_name, iso)
    ccs_lcoe = (CCS_LCOE_45Q_ON if q45 == '1' else CCS_LCOE_45Q_OFF)[ccs_lev][iso]
    ccs_lcoe += get_tx('ccs_ccgt', tx_name, iso)
    
    if remaining > 0:
        if ccs_lcoe < nuclear_lcoe and ccs_cap > 0:
            ccs_twh = min(remaining, ccs_cap)
            nuclear_twh = remaining - ccs_twh
        else:
            nuclear_twh = remaining
            ccs_twh = 0
    else:
        nuclear_twh = ccs_twh = 0
    
    return {
        'uprate_twh': uprate_twh,
        'geo_twh': geo_twh,
        'nuclear_newbuild_twh': nuclear_twh,
        'ccs_tranche_twh': ccs_twh,
    }
```

### Code Block 4: REC Scarcity Pricing (`step6_1_smartargets.py`)

Compliance REC pricing driven by supply-demand gap:

```python
def compute_rec_price(iso, eligible_pct, year):
    """Scarcity-driven compliance REC price ($/MWh)."""
    acp = ACP_RATES[iso]  # Alternative Compliance Payment cap
    if acp <= 0:
        return VOLUNTARY_REC_FLOOR[iso]  # No RPS (ERCOT)
    
    # Effective demand = RPS mandate + voluntary corporate procurement
    rps_target = get_rps_target_at_year(iso, year)
    vol_adder = VOLUNTARY_DEMAND_ADDER[iso]
    eff_target_pct = (rps_target + vol_adder) * 100.0
    gap = eff_target_pct - eligible_pct  # positive = scarcity
    
    floor = VOLUNTARY_REC_FLOOR[iso]
    k = REC_SCARCITY_K[iso]  # Calibrated per ISO to match 2025 observed
    
    if gap > 0:
        # Scarcity: price ramps toward ACP
        price = acp * (1.0 - np.exp(-k * gap))
    else:
        # Surplus: price decays toward voluntary floor
        price = floor + (compliance_2025 - floor) * np.exp(0.20 * gap)
    
    return max(floor, min(acp, price))
```

### Code Block 5: Wright's Law Learning (`step6_1_smartargets.py`)

Deployment-based cost reduction:

```python
def wright_cost(foak_cost, noak_floor, cumulative_gw, reference_gw, learning_rate):
    """Wright's Law: cost = FOAK × (Q/Q_ref)^(-b), floored at NOAK."""
    if cumulative_gw <= reference_gw or learning_rate <= 0:
        return foak_cost
    exponent = -np.log2(1.0 - learning_rate)
    cost = foak_cost * (cumulative_gw / reference_gw) ** (-exponent)
    return max(noak_floor, cost)
```

---

*End of Document*

*Constellation Energy — Commercial Strategy & Analytics*
*Model Version 1.0.0 | Pipeline v1.0.0 | Base Year 2025*
