# Advanced Sensitivity Model — Complete Specification

> **Authoritative reference for all design decisions.** If a future session needs context, read this file first.
> Finalized: 2025-02-13. Derived from full conversation history.

---

## 1. Model Framework

- **2025 snapshot model** — all data, profiles, costs, grid mix shares reflect fixed 2025 actuals
- **No demand growth projections** — point-in-time scenario analysis only
- **Grid mix baseline** = actual 2025 regional shares, priced at wholesale, selectable as reference scenario (fixed, not adjustable by user)
- **Regions**: CAISO, ERCOT, PJM, NYISO, NEISO
- **Repo**: `jessicacohen554-cyber/hourly-cfe-optimizer`
- **Dev branch**: `claude/enhance-optimizer-model-IqSpe` (all advanced model work on this branch)

---

## 2. Resources (7 total)

| # | Resource | Profile Type | New-Build? | Cost Toggle? | Transmission Adder? |
|---|---|---|---|---|---|
| 1 | **Clean Firm** (nuclear/geothermal) | Flat baseload (1/8760) | Yes | Low/Med/High (regional) | Yes (regional) |
| 2 | **Solar** | EIA 2025 hourly regional | Yes | Low/Med/High (regional) | Yes (regional) |
| 3 | **Wind** | EIA 2025 hourly regional | Yes | Low/Med/High (regional) | Yes (regional) |
| 4 | **CCS-CCGT** | Dispatchable baseload (flat) | Yes | Low/Med/High (regional) | Yes (regional) |
| 5 | **Hydro** | EIA 2025 hourly regional | **No** — capped at existing | **No** — wholesale only | **No** — always $0 |
| 6 | **Battery** (4hr Li-ion) | Daily cycle dispatch | Yes | Low/Med/High (regional) | Yes (regional) |
| 7 | **LDES** (100hr iron-air) | Multi-day/seasonal dispatch | Yes | Low/Med/High (regional) | Yes (regional) |

### Key resource decisions:
- **H2 storage excluded** (explicitly out of scope)
- **Hydro**: Existing only, capped at regional capacity, wholesale priced, no new-build tier, $0 transmission
- **CCS-CCGT**: 90% capture rate, residual ~0.037 tCO2/MWh, 45Q ($85/ton = ~$29/MWh offset) baked into LCOE, fuel cost linked to gas price toggle
- **LDES**: 100-hour iron-air, 50% round-trip efficiency, new multi-day dispatch algorithm
- **Battery**: 4-hour Li-ion, 85% round-trip efficiency, existing daily-cycle greedy dispatch preserved

---

## 3. Thresholds (10 total — reduced from 18)

```
75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 100
```

- 5% intervals from 75-85 (captures broad trend)
- 2.5% intervals from 87.5-97.5 (captures steep cost inflection zone)
- 99% and 100% anchor the extreme end
- Reduced from 18 to 10 while adding granularity in the inflection zone
- Key inflection behavior (CCS/LDES entering mix, storage costs spiking) captured at 90-97.5
- Dashboard interpolates smoothly between these anchor points for abatement curves

---

## 4. Dashboard Controls (7 total — paired toggles)

### Preserved (2):
1. **Region/ISO select** (CAISO, ERCOT, PJM, NYISO, NEISO)
2. **Threshold select** (10 values: 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 100)

### Paired sensitivity toggles (5 groups replacing 10 individual):

The 10 individual cost toggles are **paired into 5 groups** where related variables move in lockstep (all Low, all Medium, or all High together). This reduces dashboard complexity and scenario count while preserving meaningful sensitivity analysis.

| # | Paired Toggle | Options | Member Variables | Affects |
|---|---|---|---|---|
| 3 | **Renewable Generation Cost** | Low / Medium / High | Solar LCOE + Wind LCOE | Both solar and wind generation costs (regional) |
| 4 | **Firm Generation Cost** | Low / Medium / High | Clean Firm LCOE + CCS-CCGT LCOE | Both firm dispatchable resource costs (regional) |
| 5 | **Storage Cost** | Low / Medium / High | Battery LCOS + LDES LCOS | Both storage technology costs (regional) |
| 6 | **Fossil Fuel Price** | Low / Medium / High | Gas + Coal + Oil prices | Wholesale electricity price + CCS fuel cost + emission rates |
| 7 | **Transmission Cost** | None / Low / Medium / High | All resource transmission adders | Transmission adders on all new-build resources (regional) |

**Pairing rationale**:
- **Renewable Gen**: Solar and wind costs are driven by similar factors (manufacturing scale, supply chain, installation labor) — they tend to move together directionally
- **Firm Gen**: Clean firm (nuclear/geothermal) and CCS-CCGT share capital-intensive, long-lead-time cost structures
- **Storage**: Battery and LDES costs share manufacturing/materials cost drivers (lithium, iron, electrolyte supply chains)
- **Fossil Fuel**: Gas, coal, and oil prices are correlated through energy commodity markets and macro conditions
- **Transmission**: Infrastructure costs affect all new-build resources similarly within a region

**Scenario count**: 3 × 3 × 3 × 3 × 4 = **324 cost scenarios** per region per threshold (vs 59,049 if all 10 toggles independent)

**NOTE**: All toggles use **Low / Medium / High** naming consistently (never "Base" or "Baseline").

**Optimizer approach**: Resource mix co-optimized with costs for EVERY scenario. Different cost assumptions produce different optimal resource mixes — this is the core scientific contribution. 324 scenarios × 10 thresholds × 5 regions = 16,200 independent co-optimizations, with matching score cache shared across cost scenarios (physics reuse, not a shortcut).
## 3. Thresholds (18 total)

```
75, 80, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100
```

- 5% intervals: 75-85
- 1% intervals: 85-100 (inflection zone where CCS/LDES enter optimal mix)
- Critical for smooth abatement cost curves

---

## 4. Dashboard Controls (12 total)

### Preserved (2):
1. **Region/ISO select** (CAISO, ERCOT, PJM, NYISO, NEISO)
2. **Threshold select** (expanded from 7 to 18 values)

### New sensitivity toggles (10):

| # | Toggle | Options | Affects |
|---|---|---|---|
| 3 | Solar Generation Cost | Low / Medium / High | Solar LCOE (regional) |
| 4 | Wind Generation Cost | Low / Medium / High | Wind LCOE (regional) |
| 5 | Clean Firm Generation Cost | Low / Medium / High | CF LCOE (regional — geothermal blend) |
| 6 | CCS-CCGT Generation Cost | Low / Medium / High | CCS LCOE (regional — Class VI, 45Q, fuel) |
| 7 | Battery Storage Cost | Low / Medium / High | Battery LCOS (regional) |
| 8 | LDES Storage Cost | Low / Medium / High | LDES LCOS (regional) |
| 9 | Transmission Cost | None / Low / Medium / High | Resource+region adders on new-build |
| 10 | Natural Gas Price | Low / Medium / High | Wholesale price + CCS fuel cost + emission rate |
| 11 | Coal Price | Low / Medium / High | Wholesale price + emission rate |
| 12 | Oil Price | Low / Medium / High | Wholesale price + emission rate |

**NOTE**: All toggles use **Low / Medium / High** naming consistently (never "Base" or "Baseline").

**Replaces**: Old single Clean Firm cost dropdown ($90/$120/$150)

**Grid mix baseline**: Selectable as reference scenario (not a toggle — fixed 2025 actual at wholesale)

---

## 5. Complete Cost Tables

### 5.1 Solar LCOE ($/MWh)

| Level | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|
| Low | $45 | $40 | $50 | $70 | $62 |
| Medium | $60 | $54 | $65 | $92 | $82 |
| High | $78 | $70 | $85 | $120 | $107 |

### 5.2 Wind LCOE ($/MWh)

| Level | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|
| Low | $55 | $30 | $47 | $61 | $55 |
| Medium | $73 | $40 | $62 | $81 | $73 |
| High | $95 | $52 | $81 | $105 | $95 |

### 5.3 Clean Firm LCOE ($/MWh) — Regionalized by Geothermal Access

| Level | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|
| Low | $58 | $63 | $72 | $75 | $73 |
| Medium | $78 | $85 | $93 | $98 | $96 |
| High | $110 | $120 | $140 | $150 | $145 |

*CAISO lowest (Salton Sea, Imperial Valley, The Geysers geothermal). NYISO highest (nuclear-only, zero geothermal potential).*

### 5.4 CCS-CCGT LCOE ($/MWh, net of 45Q)

| Level | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|
| Low | $58 | $52 | $62 | $78 | $75 |
| Medium | $86 | $71 | $79 | $99 | $96 |
| High | $115 | $92 | $102 | $128 | $122 |

*ERCOT lowest (Gulf Coast Class VI wells, abundant geology, cheap gas, shortest CO2 transport). NYISO highest (no suitable sequestration geology, longest transport, highest permitting burden).*

**CCS-CCGT cost buildup**:
- Capture cost: ~$30-40/MWh (technology-dependent, relatively uniform)
- CO2 transport: $2-20/MWh (regional — distance to Class VI well)
- CO2 storage: $5-15/MWh (regional — geology, well costs)
- Fuel cost: Heat rate × gas price (responds to gas toggle)
- 45Q offset: -$29/MWh ($85/ton × 0.34 tCO2/MWh)
- Capture rate: 90%
- Residual emissions: ~0.037 tCO2/MWh

### 5.5 Battery LCOS ($/MWh) — Regionalized

| Level | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|
| Low | $77 | $69 | $74 | $81 | $79 |
| Medium | $102 | $92 | $98 | $108 | $105 |
| High | $133 | $120 | $127 | $140 | $137 |

*ERCOT lowest (low labor, fast permitting, flat land). NYISO highest (highest labor, most constrained siting).*

### 5.6 LDES LCOS ($/MWh, 100hr iron-air) — Regionalized

| Level | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|
| Low | $135 | $116 | $128 | $150 | $143 |
| Medium | $180 | $155 | $170 | $200 | $190 |
| High | $234 | $202 | $221 | $260 | $247 |

*ERCOT lowest (Gulf Coast geology for compressed air variants, low labor). NYISO highest (expensive labor, constrained siting, limited geology).*

### 5.7 Transmission Adders ($/MWh, new-build only)

| Resource | Level | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|---|
| **Wind** | Low | $4 | $3 | $5 | $7 | $6 |
| | Medium | $8 | $6 | $10 | $14 | $12 |
| | High | $14 | $10 | $18 | $22 | $20 |
| **Solar** | Low | $1 | $1 | $2 | $3 | $3 |
| | Medium | $3 | $3 | $5 | $7 | $6 |
| | High | $6 | $5 | $9 | $12 | $10 |
| **Clean Firm** | Low | $1 | $1 | $1 | $2 | $2 |
| | Medium | $3 | $2 | $3 | $5 | $4 |
| | High | $6 | $4 | $6 | $9 | $7 |
| **CCS-CCGT** | Low | $1 | $1 | $1 | $2 | $2 |
| | Medium | $2 | $2 | $3 | $4 | $3 |
| | High | $4 | $3 | $5 | $7 | $6 |
| **Battery** | Low | $0 | $0 | $0 | $1 | $1 |
| | Medium | $1 | $1 | $1 | $2 | $2 |
| | High | $2 | $2 | $3 | $4 | $3 |
| **LDES** | Low | $1 | $1 | $1 | $2 | $2 |
| | Medium | $2 | $2 | $3 | $4 | $3 |
| | High | $4 | $3 | $5 | $7 | $6 |
| **Hydro** | All | $0 | $0 | $0 | $0 | $0 |

*ERCOT lowest (CREZ buildout, less congestion). NYISO highest (constrained corridors, siting opposition). Sources: LBNL "Queued Up", MISO/SPP interconnection data.*

### 5.8 Fuel Prices

| Fuel | Low | Medium | High |
|---|---|---|---|
| Natural Gas | $2.00/MMBtu | $3.50/MMBtu | $6.00/MMBtu |
| Coal | $1.80/MMBtu | $2.50/MMBtu | $4.00/MMBtu |
| Oil | $55/bbl | $75/bbl | $110/bbl |

### 5.9 Fuel Price → Wholesale + Emission Rate Impact

**Wholesale**: Shifts based on regional 2025 fossil fuel mix composition. Uses **hourly wholesale price profiles** from EIA 2025 data (not flat averages).

**Emission rate — Regional fuel-switching elasticity**:

| Region | Coal Fleet Status | Switching Elasticity | Rationale |
|---|---|---|---|
| ERCOT | Largely retired (~10GW remaining) | **Low** | Limited coal to switch to; gas price barely shifts emission rate |
| PJM | Substantial remaining (~45GW) | **High** | Gas price ↑ drives meaningful coal resurgence, emission rate jumps |
| CAISO | Near zero | **Very low** | Almost no coal option |
| NYISO | Minimal | **Low** | Small effect |
| NEISO | Minimal (retiring) | **Low** | Small effect |

---

## 6. Storage Algorithms

### 6.1 Battery (4hr Li-ion) — EXISTING algorithm, unchanged
1. Calculate per-day dispatch target from total annual %
2. Identify daily surplus hours (supply > demand) and gap hours (demand > supply)
3. Charge from largest surpluses up to power rating (capacity/4hr), up to required charge amount
4. Discharge to largest gaps up to power rating, up to charged energy × 85% efficiency
5. Repeat for each of 365 days

### 6.2 LDES (100hr iron-air) — NEW algorithm
1. **Rolling 7-day window** (fine phase: full-year optimization)
2. Identify sustained multi-day surplus periods (spring wind runs, long sunny stretches)
3. Charge during surplus periods up to power rating (capacity/100hr), respecting energy capacity
4. Identify sustained multi-day deficit periods (winter evening doldrums, cloudy windless stretches)
5. Discharge during deficit periods up to power rating, up to stored energy × 50% efficiency
6. Seasonal shifting: captures week-to-week and seasonal patterns batteries cannot

---

## 7. CO2 & Abatement

### 7.1 CO2 Emissions Abated
- Each clean MWh displaces fossil generation at **regional marginal emission rate** (eGRID)
- Emission rate shifts with fuel price toggles using **regional fuel-switching elasticity**
- CCS-CCGT gets **partial credit**: 90% capture → displaces ~0.037 tCO2/MWh residual (vs ~0.37 unabated CCGT)
- New **metric tile** on dashboard showing total CO2 abated (tons) for selected scenario

### 7.2 Abatement Cost Curves (2 new charts)
- **Average Cost of Abatement**: Total incremental cost / Total CO2 abated = **$/ton CO2**
- **Marginal Cost of Abatement**: (Cost_{X+1%} − Cost_{X%}) / (CO2_{X+1%} − CO2_{X%}) = **$/ton CO2**
- **X-axis**: 75% to 100%, **linear numeric scale** (proportional spacing — distance from 85→90 equals 75→80)
- Both curves respond dynamically to **all 10 sensitivity toggles**
- 1% intervals from 85% provide smooth curve in the inflection zone
- Marginal curve shows hockey-stick shape: cheap early tons, expensive last tons

---

## 8. Dashboard Layout

### 8.1 Existing visuals — ALL PRESERVED
- Donut chart (resource mix) → updated to show 7 resources
- Compressed day profile (stacked area) → updated with Battery/LDES/CCS-CCGT split
- Peak capacity panel → updated with new resources
- Cost breakdown panel → updated with all resources
- Metric tiles: match score, procurement level, blended cost, curtailment → preserved
- Key finding box → preserved
- Scrollytelling narrative sections → preserved

### 8.2 New additions (layered on top)
- **CO2 abated metric tile** — tons of CO2 displaced for selected scenario
- **Average abatement cost curve chart** — $/ton across 75-100% thresholds
- **Marginal abatement cost curve chart** — incremental $/ton at each threshold step
- **"What You Need Depends on What You Have" panel**:
  - Starting point: grid mix baseline hourly match score
  - Target: selected threshold
  - Gap: target − baseline
  - Incremental resources needed to close gap
  - Incremental $/MWh above wholesale
  - CO2 impact of closing gap
  - Dynamic regional insight text (e.g., "NYISO's strong nuclear fleet means 40% less incremental clean firm needed vs. ERCOT")
- **Sensitivity toggle panel** (10 new toggles in control area)

### 8.3 Chart axis rules
- Abatement curves: **Linear numeric x-axis** (not categorical). 1 percentage point = same pixel distance everywhere.
- Data points at 75, 80, 85, 86, 87, ..., 100. Longer line segments between sparse points (75→80→85) are honest about lower granularity there.

---

## 9. Existing Grid Mix (2025 Actuals)

### Grid Mix Shares (% of generation):
| Resource | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|
| Clean Firm | 7.9% | 8.6% | 32.1% | 18.4% | 23.8% |
| Solar | 22.3% | 13.8% | 2.9% | 0% | 1.4% |
| Wind | 8.8% | 23.6% | 3.8% | 4.7% | 3.9% |
| Hydro | 9.5% | 0.1% | 1.8% | 15.9% | 4.4% |

### Hydro Caps (existing capacity):
| Region | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|
| Cap (GW) | 30 | 5 | 15 | 40 | 30 |

### Wholesale Market Prices (2025 hourly profiles from EIA, not flat averages):
- Average reference points: CAISO ~$30, ERCOT ~$27, PJM ~$34, NYISO ~$42, NEISO ~$41
- Actual hourly data used for storage arbitrage, deficit-hour costing, curtailment economics

---

## 10. Two-Tier Pricing Model (Preserved)

For each resource:
- **Existing share** (up to grid mix %) → priced at wholesale
- **New-build share** (above grid mix %) → priced at LCOE + transmission adder
- **Hydro**: Always wholesale (existing only, no new-build tier)
- **CCS-CCGT**: No existing share (new resource) → all new-build priced

---

## 11. Performance Optimizations

- **Parallelization**: All 5 ISOs run concurrently via multiprocessing pool
- **Caching**: Matching scores cached across 324 cost scenarios per threshold (physics reuse — cost-independent)
- **Cross-pollination**: After all 324 scenarios run per threshold, every unique mix re-evaluated against all scenarios
- **10 thresholds × 5 regions × 324 scenarios × 3 phases** — parallelization essential for reasonable runtime

### 11.1 Adaptive Procurement Bounds

Compressed procurement sweep ranges per threshold to avoid wasting compute on levels that can't be optimal:

| Threshold | Min% | Max% | Rationale |
|-----------|------|------|-----------|
| 75% | 75 | 105 | Easy target, minimal overprocurement needed |
| 80% | 75 | 110 | Slight headroom |
| 85% | 80 | 110 | Still achievable without heavy overprocurement |
| 87.5% | 87 | 130 | Storage helps close gap, not raw procurement |
| 90% | 90 | 140 | Moderate overprocurement + storage |
| 92.5% | 92 | 150 | Entering inflection zone |
| 95% | 95 | 175 | Significant overprocurement may be needed |
| 97.5% | 100 | 200 | Heavy overprocurement territory |
| 99% | 100 | 220 | Near-perfect matching |
| 100% | 100 | 200 | If 2× procurement can't hit 100%, it's not cost-viable |

### 11.2 Edge Case Seed Mixes

17 forced seed mixes injected into Phase 1 coarse scan to guarantee extreme-but-potentially-optimal mixes survive pruning. Categories:

- **Solar-dominant** (70-75%): captures low-cost renewable scenarios where massive solar + storage is cheapest
- **Wind-dominant** (70-75%): captures ERCOT-like regions where wind + LDES dominates
- **Balanced renewable** (40/40 solar/wind): diversified variable generation
- **Clean firm dominant** (60-80%): captures low-cost nuclear/geothermal scenarios
- **Combined firm** (CF 30-40% + CCS 30-40%): dual-firm strategy
- **CCS-dominant** (50-60%): regions with favorable geology (ERCOT Gulf Coast)
- **High-hydro** (40% hydro): NYISO, CAISO, NEISO where existing hydro fleet is large

Seeds filtered at runtime by regional hydro cap. Adds ~10-15 combos to the ~280 coarse grid combos per region — negligible compute cost, significant coverage improvement.

### 11.3 Monotonicity Re-Sweep Mechanism

**Problem**: The 3-phase heuristic search (coarse → medium → fine) can miss the global optimum at a lower threshold, producing a non-monotonic cost curve where cost(T_lower) > cost(T_higher). This is a diagnostic signal, not a valid result — achieving a lower CFE target should never cost more than achieving a higher one.

**Solution**: Post-hoc re-sweep with broader parameters. After all thresholds are initially optimized:

1. **Detection**: For each cost scenario, check that cost is non-decreasing across thresholds. Tolerance: $0.01/MWh (allows floating-point rounding).
2. **Collection**: Group all violations by threshold — `{threshold: {scenario_key: better_threshold}}`.
3. **Re-sweep** (up to 2 rounds): For each violated threshold:
   - **Seed injection**: Collect winning mixes from the thresholds that achieved better cost. These become Phase 1 seeds, guaranteeing the re-sweep explores the region of solution space that worked at the higher target.
   - **Broader Phase 1**: Use 5% step instead of 10% (~7-14× more combos), exploring the space more densely.
   - **Expanded procurement bounds**: Default bounds widened by -20% (min) and +30% (max) to search outside the assumed-optimal range.
   - **More Phase 2 candidates**: Top 30 instead of 20, with 2.0× cost filter (vs. 1.5×).
   - **More Phase 3 finalists**: Top 15 instead of 8, with 1.2× cost filter (vs. 1.1×).
   - Cross-pollination within re-swept threshold after re-optimization.
4. **Verification**: Re-check monotonicity after each round. If all violations resolved, stop early.
5. **Acceptance**: After 2 rounds, accept remaining violations with a warning (search space exhausted).

**Design rationale**: This approach finds the *true* optimum rather than masking the problem by pulling down from a higher threshold's result. The re-sweep is targeted (only violated scenarios) and seeded (with known-good mixes from higher thresholds), so it's both rigorous and compute-efficient.

**Compute overhead**: Typically affects 5-15% of scenarios at 1-3 thresholds per ISO. Phase 1 at 5% step generates ~2,000-5,000 combos (vs. ~280 at 10% step), but scoring is cached. Net overhead: ~10-20% of total runtime.

---

## 12. Methodology Documentation Checklist

The methodology page must include:
- [ ] All generation cost tables (Sections 5.1-5.4) with sources (NREL ATB 2024, Lazard, LBNL)
- [ ] All storage cost tables (Sections 5.5-5.6) with sources
- [ ] Complete transmission adder table (Section 5.7) with sources (LBNL "Queued Up", MISO/SPP)
- [ ] Fuel price scenario table (Section 5.8) with sources (EIA AEO, Henry Hub)
- [ ] Fuel → wholesale + emission rate linkage methodology (Section 5.9)
- [ ] Regional fuel-switching elasticity rationale
- [ ] CCS-CCGT complete cost buildup (capture + transport + storage + fuel − 45Q)
- [ ] Class VI well availability rationale by region
- [ ] 45Q tax credit mechanics and levelized impact
- [ ] LDES vs battery technology specs, efficiency, duration
- [ ] LDES dispatch algorithm description
- [ ] Battery dispatch algorithm description (preserved)
- [ ] CO2 emission factor methodology (eGRID marginal rates)
- [ ] CCS-CCGT partial credit methodology (90% capture)
- [ ] Abatement curve formulas (average and marginal)
- [ ] Hydro treatment rationale (existing-only, wholesale, regional caps)
- [ ] Two-tier pricing explanation
- [ ] Hourly wholesale price profile methodology
- [ ] Grid mix baseline methodology
- [ ] All cited sources: LBNL, NREL ATB 2024, Lazard, EIA AEO, eGRID, MISO/SPP, FERC/ISO reports

---

## 13. Regional Deep-Dive Pages (5 new HTML pages)

Each region gets a dedicated deep-dive page linked from the main dashboard top navigation.

### Structure
- **File naming**: `dashboard/region_caiso.html`, `region_ercot.html`, etc.
- **Navigation**: Top bar on main dashboard links to each region page; each region page links back to main and to other regions
- **Format**: Scrollytelling narrative matching main dashboard visual identity
- **Content**: In-depth exploration of what deep decarbonization looks like for that region under different sensitivity scenarios

### Default Cost Scenario for Static Pages
- **Homepage (index.html)** and **Regional Deep-Dive pages**: All figures and narrative use **Medium cost sensitivities** (all 5 toggle groups at Medium) unless a figure is explicitly designed to show Low/Medium/High ranges for comparison purposes.
- **Dashboard (dashboard.html)**: Interactive — user controls sensitivities via toggles.
- This ensures consistency across static narrative pages and reserves L/M/H range displays for intentional comparison figures (e.g., cost sensitivity deep-dive section #4 below).

### Per-Region Content Sections
1. **Region Overview** — grid composition, geography, market structure, current clean energy share
2. **Baseline Analysis** — existing grid hourly match score, seasonal patterns, strengths/weaknesses
3. **Decarbonization Pathway** — how optimal resource mix evolves from 75% → 100% threshold
4. **Cost Sensitivity Deep-Dive** — how L/M/H scenarios on key resources (regional priority resources) change the cost picture (this section explicitly shows ranges)
5. **Storage Role** — battery vs LDES contributions at different thresholds, dispatch patterns
6. **CO2 Abatement Profile** — regional abatement curves with commentary on inflection points
7. **Key Regional Insights** — unique factors (e.g., ERCOT's wind dominance, NYISO's nuclear fleet, CAISO's solar+storage, PJM's coal fleet switching dynamics)
8. **Comparison to National Context** — how this region compares to others

### Regional Priority Resources
- **CAISO**: Solar + battery, geothermal clean firm
- **ERCOT**: Wind + CCS-CCGT (Gulf Coast geology), low-cost solar
- **PJM**: Nuclear clean firm, coal→gas switching dynamics, wind
- **NYISO**: Nuclear fleet, hydro, limited renewables siting
- **NEISO**: Offshore wind potential, nuclear, limited solar

---

## 14. Research Paper PDF

### Format
- Generated as PDF (via HTML→PDF or direct HTML print stylesheet)
- Academic paper structure with executive summary
- Includes all regional deep-dive content as paper sections
- Includes full methodology detail (more than the methodology HTML page)

### Paper Structure
1. **Executive Summary** — key findings across all regions
2. **Introduction** — hourly CFE matching problem, why annual matching isn't enough
3. **Methodology** — full model description, all cost tables, algorithms, data sources
4. **National Results** — overview across all 5 regions, comparison charts
5. **Regional Deep-Dives** (5 sections, one per region — content from deep-dive pages)
6. **Sensitivity Analysis** — how key assumptions drive results
7. **Policy Implications** — what this means for procurement strategy
8. **Appendix** — full data tables, source citations, technical specifications

### Audience
- **Primary**: Business professionals with minimal energy domain knowledge
- **Secondary**: Academic/policy reviewers (must withstand scrutiny)
- Accessible first, rigorous underneath

---

## 15. Abatement Cost Comparison Page (NEW)

### Concept
A "Liebreich ladder for grid decarbonization" — analyzing when/where/under what conditions pushing grid decarbonization % is no longer cost-effective compared to alternative mitigation and carbon removal options. Linked from dashboard navigation.

**File**: `dashboard/abatement_comparison.html`

**Core Question**: "Should we focus the next marginal dollar on the last 5% of PJM grid decarbonization, sustainable aviation fuel, or direct air capture?"

### Analysis Framework

**Y-axis**: Cost of carbon abatement ($/ton CO2)
**X-axis**: Cumulative abatement potential or grid % target

**Grid Decarbonization Curves** (from our model):
- Regional marginal abatement cost curves (75-100%) for each ISO under L/M/H sensitivities
- Show hockey-stick inflection where costs spike (typically 95-100%)
- Each region's curve under different sensitivity scenarios

**Comparison Benchmarks** (horizontal lines/bands on same chart):
| Mitigation Option | Low $/ton | Medium $/ton | High $/ton | Source |
|---|---|---|---|---|
| Energy efficiency (buildings) | $0 | $20 | $50 | IEA, McKinsey |
| Industrial electrification | $30 | $75 | $150 | IEA |
| Sustainable Aviation Fuel (SAF) | $150 | $250 | $400 | ICCT, BloombergNEF |
| Green hydrogen (industrial) | $100 | $200 | $350 | Liebreich, BNEF |
| BECCS | $100 | $175 | $300 | IPCC AR6 |
| Direct Air Capture (DAC) | $250 | $400 | $600+ | Carbon Engineering, Climeworks |
| Enhanced weathering | $50 | $125 | $200 | IPCC |
| Carbon credits (voluntary market) | $10 | $50 | $150 | Ecosystem Marketplace |
| EU ETS carbon price (2024-2025) | $60 | $80 | $100 | EMBER |

### Key Analytical Sections

1. **The Grid Decarbonization Curve** — Our model's regional marginal abatement curves plotted together. Where does each region's curve cross the DAC line? The SAF line?

2. **The Inflection Point Analysis** — For each region × sensitivity scenario, identify the % threshold where grid decarbonization costs exceed:
   - The social cost of carbon ($51/ton EPA, $185/ton Rennert et al.)
   - DAC costs ($250-600/ton)
   - SAF costs ($150-400/ton)
   - Voluntary carbon market prices ($10-150/ton)

3. **The Liebreich-Style Ladder** — Rank all mitigation options by cost-effectiveness at each grid % level. At 85% grid target, what's cheaper? At 95%? At 99%?

4. **Regional Divergence** — Some regions (ERCOT with cheap wind) stay cost-competitive deep into high %'s. Others (NYISO) become expensive earlier. Map the crossover points.

5. **The Net-Zero Pathway** — Given that residual emissions exist at any grid %, what's the optimal split between:
   - Pushing grid % higher (expensive past inflection)
   - Investing in DAC for residual emissions
   - Investing in other sectors (SAF, industrial) for cross-sector abatement

6. **Sensitivity Scenarios** — How do different cost assumptions shift the inflection points? Under low DAC cost assumptions, the crossover happens earlier. Under high renewable cost assumptions, same.

### Path-Dependency & Retroactive Cost Modeling (Under Development)

**Problem**: The optimizer independently optimizes each threshold. The 85% mix and 95% mix may differ fundamentally — heavy solar at 85%, heavy clean firm at 95%. Building the 85%-optimal mix then upgrading to 95% would strand solar assets and cost more than building toward 95% from the start.

**Proposed Approach**: Model backwards from the inflection point where the optimal grid solution's LCOE crosses a benchmark price:
- **Primary benchmark**: DAC cost projected to the target year
  - 2025: $400-600/ton → grid dominates through ~97%
  - 2035: $250-350/ton → grid dominates through ~93-95%
  - 2045: $150-250/ton → grid dominates through ~90-93%
- At the crossover threshold, the optimal mix is fixed. Then model the build-up path from lower thresholds using the cheapest-first resource ordering that converges to the crossover mix.
- DAC learning curve: ~15-20% cost reduction per doubling of deployment (ETH Zurich/Climeworks data)
- This creates a **declining optimal grid target over time** as removal costs fall — counterintuitive but logical.

**Status**: Waiting for optimizer results to analyze mix divergence between thresholds. If divergence is small (resources are additive), the current independent optimization is sufficient. If large (mix pivots between thresholds), path-constrained modeling is needed.

### DAC-VRE Co-Optimization Insight (Under Development)

**Core insight**: DAC is a flexible load that can absorb curtailed renewable energy. At high grid targets, significant curtailment occurs — this energy is nearly free ($0-5/MWh). DAC facilities co-located with sequestration geology (Class VI wells) can use curtailed power to remove CO₂ at dramatically reduced costs.

**Regional specialization model**:
- **ERCOT/CAISO**: Push grid to 95-97% (cheap wind/solar), operate DAC on curtailed surplus. Gulf Coast & Imperial Valley have Class VI well capacity.
- **PJM**: Push grid to 93-95%, buy ERCOT/regional DAC credits for residual emissions.
- **NYISO/NEISO**: Push grid to 90-92% (expensive beyond), heavy DAC credit procurement from regions with cheaper removal.

**DAC cost with curtailed power**: If energy is the #1 DAC cost driver (~60% of total), curtailed power at $0-5/MWh could reduce DAC from $400-600/ton to $150-250/ton — making it competitive with grid decarbonization costs above 93% in most regions.

**Analysis needed**:
1. From optimizer results: quantify curtailed MWh at each threshold × region
2. Estimate DAC capacity supportable by curtailed energy (assume 2 MWh/ton)
3. Derive DAC marginal cost curve as a function of curtailment availability
4. Compare DAC-on-curtailment cost to grid MAC at each threshold
5. Find the optimal regional grid target + DAC allocation

**Why this matters**: This reframes the "100% clean grid" question. If DAC-on-curtailment is cheaper than the last 5-10% of grid matching, the rational strategy is to overbuild VRE (creating more curtailment) and co-locate DAC — achieving net-zero at lower total cost than pure grid matching.

**DAC CapEx vs. curtailment capacity factor**:
- Running DAC only on curtailed hours (15-30% CF) increases amortized CapEx/ton by 3-6x
- Energy savings ($120-170/ton) partially offset but don't fully compensate
- **Optimal: Hybrid model** — DAC runs baseload at 70-80% CF (grid power), with curtailed hours as supplemental free energy
- This spreads CapEx over sufficient tons while capturing curtailment energy savings
- The MAC curves for DAC typically assume ~90% CF with market-rate power; our model should account for the CF impact on per-ton costs

**Abatement page section**: Dedicated section with narrative walkthrough + findings in the Key Insights panel at top.

**Implementation scope**: Supplementary analysis for the **abatement page only** — not the main dashboard.
- Run supplementary optimizer scenarios for ERCOT and CAISO (both have Class VI well capacity — Gulf Coast and Central Valley/Salton Sea respectively)
- Model allows monetizing curtailed energy via DAC-VRE co-location
- These regions push to 100%+ procurement with excess curtailment → DAC
- DAC credits offset residual emissions in NYISO/NEISO/PJM at high targets where grid costs are steep
- Produces a "cross-regional portfolio" where cheap-DAC regions export removal credits to expensive-grid regions

### Visual Design
- Large interactive chart: Regional MAC curves overlaid with benchmark bands
- Horizontal benchmark lines clearly labeled with color-coded bands
- Inflection point callouts where curves cross benchmarks
- Toggle: Region selector, sensitivity scenario
- Scrollytelling narrative explaining the analysis

### Audience
- Same business professional audience
- Build the case: "Here's when clean energy procurement stops being the cheapest path and alternatives become more efficient"
- Frame as strategic portfolio optimization, not just grid optimization

---

## 15b. Methodology HTML Page (Trimmed)

- **Keep**: Technical methodology, model specifications, algorithm descriptions
- **Remove**: Deep narrative content (moved to PDF paper and regional pages)
- **Purpose**: Quick-reference technical documentation for the model
- **Links to**: PDF paper for full methodology and analysis

---

## 16. Header Banner & Navigation

### Banner Placement
- **Main dashboard**: Banner appears ABOVE intro text (not below)
- **All pages** (dashboard, regional deep-dives, methodology): Same header banner styling
- Banner includes page-specific name + tagline

### Per-Page Banner Content
| Page | Title | Tagline |
|---|---|---|
| Main Dashboard | Hourly CFE Optimizer | Advanced Sensitivity Model |
| CAISO Deep Dive | CAISO Deep Dive | California's Path to 24/7 Clean Energy |
| ERCOT Deep Dive | ERCOT Deep Dive | Texas Grid Decarbonization Analysis |
| PJM Deep Dive | PJM Deep Dive | Mid-Atlantic Clean Energy Transition |
| NYISO Deep Dive | NYISO Deep Dive | New York's Hourly Matching Challenge |
| NEISO Deep Dive | NEISO Deep Dive | New England's Decarbonization Pathway |
| Methodology | Technical Methodology | Model Specifications & Data Sources |

### Navigation
- Top navigation bar on ALL pages
- Links: Dashboard | CAISO | ERCOT | PJM | NYISO | NEISO | Methodology | Paper (PDF)
- Current page highlighted in nav
- Mobile: collapsible/hamburger nav

---

## 17. Audience & UX Guidelines

### Dashboard (business professional audience)
- Layer in explanations for model elements, figures, toggles
- Tooltips or info icons (ⓘ) on each control explaining what it does and why it matters
- Chart titles that tell the story, not just label the axis
- Key finding boxes that translate numbers into business implications
- Assume reader does NOT know what LCOE, LCOS, capacity factor, or hourly matching mean

### Regional Deep-Dive Pages
- Written for a reader encountering the topic for the first time
- Build understanding progressively (scrollytelling)
- Lead with "so what" before diving into "how"
- Use analogies and real-world comparisons where helpful

### Research Paper / Methodology
- More technical depth acceptable
- Must still be accessible to first-time readers
- Withstand academic scrutiny: cite sources, show methodology, acknowledge limitations
- Full cost table transparency

---

## 17. QA/QC Requirements

### Optimizer Results QA (after first region completes)
- Validate hourly match scores against expected ranges from existing research
- Check that resource mixes make directional sense (e.g., wind-heavy in ERCOT, nuclear-heavy in PJM)
- Verify cost figures fall within published LCOE/LCOS ranges (NREL ATB, Lazard)
- Confirm CO2 abatement numbers are physically reasonable (tons displaced per MWh)
- Check that higher thresholds always cost more than lower ones (monotonicity)
- Verify storage dispatch increases with threshold (more storage needed at higher targets)

### Dashboard HTML QA
- Visual consistency: fonts, colors, spacing, alignment across all sections
- All toggles functional and responsive
- Chart rendering correct with proper labels, legends, axes
- Metric tiles update correctly when controls change
- No broken layouts at any control combination
- Clean/crisp visual identity — no cluttered elements

### Mobile Compatibility
- All figures render with readable text on mobile screens (320px-768px)
- Touch-friendly toggle controls (minimum 44px tap targets)
- Charts scale properly (responsive Canvas/Chart.js)
- Scrollytelling sections work on touch scroll
- No horizontal overflow or text truncation
- Navigation accessible on mobile (hamburger or stacked)
- Test at: 320px (small phone), 375px (iPhone), 768px (tablet)

### Pre-Push Checklist
- [ ] Optimizer results QA passed for all 5 regions
- [ ] All dashboard controls functional
- [ ] All charts render correctly
- [ ] Mobile compatibility verified
- [ ] Regional deep-dive pages complete and linked
- [ ] Research paper PDF generated
- [ ] Methodology page trimmed
- [ ] No console errors in browser
- [ ] Standalone HTML builds successfully
- [ ] All text readable at all viewport sizes

---

## 18. Summary Counts

| Item | Count |
|---|---|
| Resources | 7 |
| Thresholds | 10 (reduced from 18) |
| Regions | 5 |
| Dashboard controls | 7 (2 existing + 5 paired toggles) |
| Paired toggle groups | 5 (from 10 individual toggles) |
| Cost scenarios per region/threshold | 324 (3×3×3×3×4) — each independently co-optimized |
| New chart types | 2 (average + marginal abatement curves) |
| New panels | 1 ("What You Need" panel) |
| New metric tiles | 1 (CO2 abated) |
| Cost tables in methodology | ~15 |
| Sensitivity toggles | 10 (all Low/Medium/High except Transmission which adds None) |
| Regional deep-dive pages | 5 (one per region) |
| Research paper sections | 8 (including 5 regional deep-dives) |
| QA checkpoints | 3 (optimizer, HTML, mobile) |

---

## 19. Model Limitations & Simplifying Assumptions

This section documents known simplifying assumptions for transparency and academic rigor. These should be acknowledged in the research paper and methodology page.

### 19.1 Static LDES LCOS (Utilization-Independent)

**Assumption**: LDES (100hr iron-air) uses a static LCOS ($/MWh) from published cost tables at assumed cycling frequency, regardless of the scenario's realized dispatch utilization.

**Why this matters**: LDES is extremely capital-intensive (~$5,000-10,000/kW installed at 100hr duration). The LCOS is dominated by capital recovery, so it is highly sensitive to utilization. A scenario where LDES cycles 50 times/year has a dramatically lower effective LCOS than one where it cycles 5 times/year — yet both use the same $/MWh in the model.

**Impact**: In scenarios with low LDES utilization (e.g., solar-dominant mixes with limited multi-day surplus), the model may understate the true cost of LDES. In scenarios with high utilization (wind-dominant mixes with abundant multi-day surplus to time-shift), the model may overstate LDES costs.

**Justification**: This approach is consistent with standard practice in published energy models (NREL ATB, Lazard LCOS). These sources quote LCOS at assumed utilization rates, and most capacity expansion models use static cost inputs without feedback from dispatch results. Implementing utilization-dependent LCOS would create a cost ↔ dispatch feedback loop (cost depends on dispatch, which depends on mix, which depends on cost) that, while convergent, adds significant methodological complexity. The same limitation applies to CCS-CCGT capacity factor effects on LCOE, though to a lesser degree given CCS's lower capital intensity per kW.

**Mitigation**: The optimizer's resource mix co-optimization partially self-corrects for this — it won't allocate large LDES shares in mixes that don't produce sufficient multi-day surplus to fill it, because the matching score won't benefit enough to justify the cost. The limitation is most relevant at the margin, where small LDES allocations face the highest effective cost per useful MWh.

### 19.2 CCS-CCGT at Assumed Baseload Capacity Factor

**Assumption**: CCS-CCGT LCOE reflects assumed high-capacity-factor baseload operation. In practice, CCS plants in a high-renewable grid might operate at lower capacity factors, increasing their effective LCOE.

**Impact**: Similar to LDES, the model may understate CCS-CCGT costs in scenarios where it operates at low utilization. However, since the optimizer models CCS as flat baseload (1/8760 profile), allocated CCS capacity runs at 100% CF by construction. The limitation applies to whether that assumption reflects real-world operations in a grid with significant renewable penetration.

**Mitigation**: The firm generation cost toggle (Low/Medium/High) provides sensitivity analysis around the LCOE assumption. High firm generation costs can be interpreted as a proxy for reduced capacity factor economics.
