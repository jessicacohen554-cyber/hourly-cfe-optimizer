# Independent Technical Audit: Market Simulation Screening Tool

**Reviewer perspective**: Third-party energy systems modeler with experience in production-grade capacity expansion and dispatch modeling (GenX, ReEDS, PLEXOS, IPM, Aurora, AMP).

**Date**: March 2026

**Scope**: Code-level review of `market-simulator/` — Python simulation engine, LMP engine, zonal decomposition, fleet model, and web frontend. This audit evaluates whether the tool produces results that are directionally meaningful, identifies methodological limitations, and assesses where (if anywhere) it fits in the landscape of electricity market modeling.

---

## Executive Summary

This tool occupies an unusual position: it is simultaneously more ambitious and less rigorous than it needs to be. It attempts to simulate multi-decade market trajectories across seven ISOs with merit-order dispatch, plant-level economics, zonal LMP decomposition, learning curves, capacity market degradation, RPS compliance, demand response, and CCS retrofit analysis — a scope that would be impressive in a production model with proper foundations, but is built on a dispatch engine that lacks the physical and economic constraints necessary to produce trustworthy results at the resolution it claims.

**Bottom line**: The tool is not useless, but it is not what it presents itself as. It is a *parametric sensitivity explorer* — useful for understanding which variables matter most and in what direction — wrapped in the UI of a *market simulator* that implies quantitative precision it cannot deliver. Used honestly for directional screening and stakeholder education, it has value. Used as a basis for investment decisions, fleet retirement timing, or policy design, it would produce misleading results.

The tool should not be scrapped. It should be reclassified, its limitations made explicit to users, and a subset of its most problematic methodological choices corrected if it is to be taken seriously even as a screening tool.

---

## 1. What This Tool Actually Is (vs. What It Implies)

### What the UI Suggests
The web interface presents detailed KPI cards (Clean %, Avg LMP to the cent, Emissions in MT), year-by-year trajectory charts with confidence bands, plant-level profit/loss tables, and zonal LMP decomposition — all of which imply a production-quality market simulation producing quantitatively reliable results.

### What the Code Does
The simulation engine (`market_simulation.py`) performs the following per year:

1. Loads pre-computed resource mixes from Step 2.2 parquets (or generates synthetic ones if absent)
2. Builds a merit-order stack with 4 aggregated fossil unit types using fleet-average heat rates
3. Dispatches fossil generation against residual demand (total demand minus clean supply)
4. Computes LMP as the marginal cost of the last dispatched unit, with post-hoc statistical adjustments
5. Evaluates clean resource deployment via LCOE merit-order (deploy if revenue > cost, stop if not)
6. Applies economic retirement as a fractional capacity reduction based on margin thresholds

This is a **reduced-form screening model**, not a market simulator. The distinction matters.

---

## 2. Methodological Criticisms

### 2.1 Dispatch Model: Merit-Order Without Physical Constraints

The dispatch engine (`lmp_engine.py:284-420`) builds a merit-order stack from four aggregated fossil unit types and dispatches them against hourly residual demand. This is the textbook "economics 101" dispatch model — useful for pedagogy, inadequate for simulation.

**What's missing:**

- **No network constraints.** The "zonal LMP" module (`zonal_lmp.py`) solves an LP with inter-zonal transfer limits, but the zone definitions are hardcoded approximations (e.g., PJM decomposed into 5 zones with fixed demand shares), and the fossil stack is split proportionally by demand share rather than by actual plant location (`lmp_engine.py:376-399`, `_split_stack_by_zone`). This produces zonal price differentials that are artifacts of the decomposition method, not reflections of actual transmission congestion.

- **No ramp rate constraints.** Units can change output from 0 to full capacity between hours with no penalty beyond the start-up cost tracked in the post-hoc UC module. Real CCGTs ramp at 8-15 MW/min; coal at 1-3 MW/min. During steep morning ramps and evening solar cliffs, this omission materially affects which units are marginal and therefore what the LMP is.

- **Unit commitment is post-hoc, not co-optimized.** The UC state machine (`market_simulation.py:533-606`) applies min-up/min-down constraints *after* the merit-order dispatch decision, adjusting committed hours but not feeding back into the dispatch or pricing solution. In real markets, UC constraints raise LMP during shoulder hours (units forced to run below cost create uplift payments) and create price spikes during tight periods. The model misses this entirely.

- **No must-run constraints beyond a config flag.** Nuclear is assumed flat baseload at 93% CF, but the model doesn't enforce must-run status for coal units with take-or-pay fuel contracts, self-scheduled QFs, or RMR units — all of which affect merit-order pricing.

### 2.2 LMP Model: Statistical Overlays on a Structural Gap

The LMP calculation (`lmp_engine.py`, `compute_hourly_lmp_vectorized`) determines price as the marginal cost of the last dispatched fossil unit, then applies demand-quantile adjustments via `PriceModel`:

- **High-demand congestion adder**: Quadratic function of demand percentile rank above the 85th percentile
- **Scarcity tail**: Linear adder above the 97th percentile
- **Low-demand depression**: Linear reduction below the 15th percentile, amplified when VRE penetration exceeds 25%

These are calibration overlays designed to make the output *look like* real LMP distributions. They are not derived from physical constraints — they are curve-fitting parameters applied to a structurally incomplete dispatch model. The `_apply_pricing_layers` function in `zonal_lmp.py:401-450` applies these same statistical adjustments per zone, compounding the approximation.

**The fundamental problem**: Real LMP volatility comes from transmission congestion, unit commitment constraints, operating reserve requirements, and scarcity pricing mechanisms (ORDC in ERCOT, capacity performance penalties in PJM). This model has none of these. The statistical overlays can match historical LMP *distributions* but cannot predict how LMP *responds to structural changes* in the generation mix — which is the entire purpose of a forward-looking simulation.

### 2.3 Capacity Expansion: LCOE Merit-Order Is Not Optimization

The deployment model (`market_simulation.py:1421-1638`, `compute_market_deployment`) ranks clean resources by LCOE and deploys them in order as long as revenue exceeds cost, subject to an annual interconnection queue cap.

**Critical flaws:**

- **No system value feedback.** When solar is deployed, it suppresses midday LMP (the "solar duck curve" / cannibalization effect). This model deploys solar at the *current* average LMP and does not re-compute LMP after deployment. In reality, the marginal value of solar collapses with penetration — this is the single most important dynamic in VRE deployment economics, and it is entirely absent.

  The relevant code (`market_simulation.py:1518`): `total_revenue = base_energy_rev + capacity_rev + rec_rev` uses a static `avg_lmp` as `base_energy_rev` for all resources, regardless of their temporal generation profile. Solar and wind receive the same energy revenue per MWh, which is flatly wrong at any significant penetration level.

- **No storage co-optimization with deployment.** Storage parameters (battery %, LDES %) are read from pre-computed parquets or hardcoded profiles. The deployment model does not decide how much storage to build — it deploys generation resources and hopes the pre-computed storage allocation is appropriate. In production capacity expansion models (GenX, ReEDS), storage is endogenously co-optimized with generation because its value depends entirely on the generation mix.

- **Queue cap as the binding constraint on deployment pace.** The model uses LBNL queue completion rate data (`market_simulation.py:96-109`) as the primary bottleneck on new builds. This is a reasonable real-world constraint, but the model treats it as a hard GW/year cap applied uniformly across all technologies. In practice, solar and wind have much higher queue completion rates than nuclear or offshore wind. The undifferentiated cap biases results toward slower deployment.

### 2.4 Retirement Model: Heuristic, Not Economic

Economic retirement (`market_simulation.py:857-954`) uses a margin-threshold approach:

```python
if margin < -5:
    retire_frac = 0.20 + 0.70 * ((loss_depth - 5) / 25)
```

This retires a *fraction* of capacity within a unit type based on a continuous function of operating margin. Real retirement decisions are binary (a plant either closes or doesn't) and depend on factors this model ignores:

- **Fixed cost recovery**: A plant with negative operating margin may stay open if it has a capacity obligation, below-market fuel contract, or regulatory mandate
- **Decommissioning costs**: Retirement has costs (environmental remediation, labor obligations) that make "keep losing money" rational in some cases
- **Optionality**: A plant that is unprofitable today but expects fuel prices to rise may stay open
- **State/regulatory intervention**: Illinois, New York, and New Jersey have all intervened to prevent economically-driven retirements of generators deemed reliability-critical

The 5% reliability floor (`market_simulation.py:906-907`) is a rough substitute for proper resource adequacy modeling.

### 2.5 Nuclear Revenue and Retirement: Oversimplified 45U Model

The nuclear revenue stack (`market_simulation.py:985-1029`) models 45U as a contract-for-difference with a floor price and sunset year. This is a reasonable first-order approximation, but:

- The model applies a single nuclear CF (93%) and a single retirement threshold ($30/MWh) across all nuclear plants. In reality, single-unit sites have higher per-MWh fixed costs than multi-unit sites, and merchant plants face different economics than rate-based plants.
- The capacity market degradation S-curve (`market_simulation.py:957-982`) is an interesting modeling choice but is applied identically to nuclear and all other resources, when in practice nuclear's ELCC (effective load carrying capability) holds up better under decarbonization than intermittent resources.

### 2.6 Synthetic Data Fallback: Invalidates Results

When Step 2.2 parquets are absent, the model falls back to `_generate_synthetic_step3_data()` (`market_simulation.py:1719-1802`), which fabricates resource mix profiles using hardcoded linear ramp patterns per ISO. These are not calibrated to any physical model — they are guesses about what the resource mix "should look like" at different clean energy percentages.

Any results generated from this fallback path are not screening-quality — they are illustrative at best. The UI provides no indication to the user that synthetic data is being used.

### 2.7 Demand Response: Ad-Hoc Price Elasticity

The DR implementation (`market_simulation.py:476-508`) applies a post-hoc LMP reduction proportional to curtailed demand with a hardcoded elasticity multiplier of 3.0:

```python
price_reduction = min(demand_reduction_pct * 3.0, 0.5)
hourly_lmp[h_idx] *= (1.0 - price_reduction)
```

This is a Python `for` loop over individual hours (violating the codebase's own vectorization mandate in CLAUDE.md) that applies an arbitrary price-demand elasticity. Real DR programs have participation caps, notification requirements, performance penalties, and baseline calculation disputes that this model ignores. More fundamentally, DR should be co-dispatched with supply resources, not applied as a post-processing step.

---

## 3. What the Tool Does Well

Credit where due — several elements are well-executed:

- **Parameter sourcing is thorough.** Heat rates from PJM SOM 2024, emission rates from EPA eGRID 2022 and CAMPD 2023, fuel prices from EIA, capacity market prices from actual RPM/ICAP/FCM auction results. The constants in `lmp_engine.py` and `pipeline_config.py` are well-documented with source citations. This is better than many academic models.

- **Revenue decomposition is comprehensive.** The model tracks energy revenue, capacity revenue, REC revenue, and PTC/ITC effects separately per resource. This multi-stream revenue model (`market_simulation.py:1032-1158`) is exactly the right framework for understanding clean energy economics — it's just applied to a dispatch model that can't produce reliable energy revenue estimates.

- **Wright's Law implementation is reasonable.** The learning curve model (`market_simulation.py:342-368`) with technology-specific learning rates, background deployment, and FOAK/NOAK cost floors is a credible approach. The parameterization (20% learning for batteries, 15% for nuclear, 0% for mature solar/wind) aligns with published literature.

- **CCS breakeven analysis is useful.** The retrofit breakeven calculation (`market_simulation.py:1339-1396`) — finding the carbon price where CCS becomes cheaper than unabated CCGT — is a clean analytical calculation that doesn't depend on the dispatch model's limitations. This section produces genuinely useful results.

- **Plant-level data integration is real.** When EIA-860/923/EPA CAMPD data is available, the `fleet_model.py` module builds per-plant merit-order stacks with actual heat rates, vintage-adjusted UC parameters, and plant-specific emission factors. This is a meaningful upgrade over the aggregated 4-unit-type default.

- **Interconnection queue constraints are grounded.** Using LBNL queue completion rate data as a deployment pace constraint is a pragmatic modeling choice that many academic models ignore entirely.

- **The code is clean and well-organized.** Clear separation of concerns (dispatch engine, LMP engine, fleet model, zonal solver), comprehensive docstrings, proper use of NumPy vectorization in the core dispatch paths, and Numba JIT for the UC state machine. The codebase is substantially more readable than many production energy models.

---

## 4. Positioning in the Modeling Landscape

### 4.1 What Production Models Do That This Tool Doesn't

| Capability | GenX | ReEDS | PLEXOS | IPM | Aurora | AMP | This Tool |
|-----------|------|-------|--------|-----|--------|-----|-----------|
| Co-optimized capacity expansion + dispatch | Yes | Yes | Yes | Yes | Yes | Yes | No |
| Nodal/zonal transmission with power flow | Yes | Zonal | Yes | Yes | Yes | Yes | Heuristic |
| Co-optimized unit commitment | Yes | No* | Yes | Yes | Yes | Partial | Post-hoc |
| Storage co-optimization | Yes | Yes | Yes | Yes | Yes | Yes | Exogenous |
| Renewable curtailment / cannibalization | Yes | Yes | Yes | Yes | Yes | Yes | No |
| Probabilistic reliability (LOLE/EUE) | Optional | REPRA | Yes | Yes | Yes | Partial | No |
| Inter-temporal build constraints | Yes | Yes | Yes | Yes | Yes | Yes | No |
| Multi-year with path dependence | Yes | Yes | Partial | Yes | Yes | Yes | Partial** |

\* ReEDS uses a capacity credit approach with reduced-form dispatch, not full UC.
\** Learning curves create path dependence, but each year's dispatch is independent.

### 4.2 Where This Tool Sits

This tool is closest in category to a **corporate energy screening tool** — similar to what internal strategy teams at utilities or large energy consumers build for directional analysis. It is not comparable to:

- **Production planning models** (GenX, ReEDS, IPM): These solve optimization problems. This tool evaluates pre-determined scenarios.
- **Detailed dispatch models** (PLEXOS, Aurora): These solve security-constrained unit commitment with detailed network models. This tool does merit-order dispatch with statistical LMP adjustments.
- **Integrated assessment models** (GCAM, NEMS): These model economy-wide energy-climate interactions. This tool focuses on the electricity sector.

The closest analogs are:
- **NREL's Cambium** (reduced-form projections from ReEDS runs) — but Cambium is backed by a proper capacity expansion model
- **Bloomberg NEF's market outlook models** — scenario-based, reduced-form, directional
- **Internal utility "what-if" spreadsheets** — but with much better parameter sourcing and more dimensions

### 4.3 The Fundamental Tension

The tool's core innovation — "clean energy deployment as an output of profitability, not a mandated target" — is a genuinely interesting framing. Most capacity expansion models treat clean energy targets as constraints and find the least-cost way to meet them. This tool inverts the question: *given market conditions, how much clean energy gets built on its own?*

That's a legitimate question. But answering it correctly requires a dispatch model that can compute the actual market revenue each resource earns at different penetration levels, accounting for temporal value, curtailment, and system interactions. This model cannot do that — it uses average LMP as energy revenue for all resources, which breaks down completely above ~40% VRE penetration where temporal value divergence is the dominant economic signal.

---

## 5. Valid Use Cases

Despite the limitations above, the tool can produce useful results in specific contexts:

1. **Directional sensitivity analysis**: "Does higher gas prices accelerate or decelerate clean energy deployment?" — the model will get the *direction* right even if the *magnitude* is wrong.

2. **CCS retrofit economics**: The breakeven carbon price analysis is a straightforward calculation that doesn't depend on the dispatch model's limitations.

3. **Nuclear retirement risk screening**: "At what combination of gas price + capacity market price + PTC does nuclear become uneconomic?" — reasonable screening question, and the multi-stream revenue model is well-suited to it.

4. **Stakeholder education**: The interactive UI with parameterized inputs helps non-technical stakeholders understand which variables drive electricity market outcomes. This is genuinely valuable — energy system literacy is low among corporate decision-makers.

5. **Scenario comparison (relative, not absolute)**: "Scenario A produces 15% more clean energy than Scenario B" — the relative ranking of scenarios is more reliable than the absolute clean% in any single scenario.

6. **First-order fleet economics**: When using real EIA/EPA plant data, the per-plant profitability assessment under different fuel and carbon price assumptions is useful directional analysis.

### Invalid Use Cases

1. **Absolute LMP forecasting**: The LMP values should not be taken at face value. They are produced by a model without transmission constraints, proper UC, or scarcity pricing.

2. **Optimal resource portfolio design**: The model cannot determine optimal resource mixes because it doesn't account for curtailment, storage co-optimization, or system value.

3. **Retirement timing**: The heuristic retirement model cannot predict when specific plants will retire.

4. **Policy impact quantification**: "A $50/ton carbon price reduces emissions by X MT" — the model will get the direction right but the magnitude is unreliable because it doesn't model fuel switching, dispatch changes, or capacity investment responses correctly.

---

## 6. Recommendations

### 6.1 If the Goal Is a Respectable Screening Tool (Moderate Effort)

These changes would make the tool defensible as a screening model without requiring a ground-up rebuild:

1. **Add VRE cannibalization feedback.** After deploying solar/wind, re-compute their energy revenue using time-matched LMP × generation profiles rather than average LMP. This is the single highest-impact improvement — a few dozen lines of code that would fix the most egregious bias in the deployment model. The infrastructure already exists: `compute_energy_revenue_by_resource` (`market_simulation.py:1032-1048`) computes time-weighted revenue per resource but the deployment model doesn't use it.

2. **Co-optimize storage with generation.** The storage dispatch LP in `dispatch_utils.py` already exists — wire it into the deployment model so storage capacity is determined endogenously rather than read from parquets.

3. **Make the synthetic fallback path visually distinct.** When running without Step 2.2 parquets, the UI should display a prominent warning that results are illustrative only. Currently there is no indication.

4. **Replace the demand-quantile LMP overlays with calibrated scarcity pricing.** Use an ORDC-style curve (price rises as reserves fall below target) rather than percentile-based statistical adjustments. This would make LMP respond structurally to generation mix changes rather than following a fixed distribution shape.

5. **Differentiate queue caps by technology.** Solar/wind should have higher completion rates than nuclear/offshore wind, per the LBNL data the tool already cites.

6. **Add explicit confidence intervals / uncertainty bands.** The trajectory mode has confidence zone labels ("Calibrated", "Moderate Extrapolation", "High Uncertainty") — extend this to show fan charts on key outputs so users see the uncertainty inherent in the projections.

### 6.2 If the Goal Is a Production-Quality Model (Major Effort)

To be comparable to GenX, ReEDS, or Aurora, the tool would need:

1. **A proper optimization formulation.** Replace the sequential merit-order → deploy → retire pipeline with a co-optimized LP/MIP that simultaneously determines capacity additions, retirements, and dispatch. This is a fundamental architectural change.

2. **A real transmission model.** At minimum, a properly calibrated zonal model with historically-derived transfer limits and generation-to-zone mapping from actual plant locations. Ideally, a DC power flow approximation.

3. **Co-optimized unit commitment.** Either a full MIP formulation or a Lagrangian relaxation approach that feeds UC costs back into LMP.

4. **Probabilistic reliability assessment.** Replace the heuristic RA floor with LOLE/EUE calculations that properly account for correlated outages and weather-dependent renewable generation.

5. **Inter-temporal investment constraints.** Construction lead times, financing constraints, learning-by-doing feedbacks, and path-dependent technology lock-in.

This would essentially mean rewriting the simulation engine from scratch. The frontend, data pipeline, and parameter library could be preserved, but the core modeling would be new.

### 6.3 What I Would Actually Recommend

**Keep the tool. Rename it. Fix the biggest bias. Be honest about what it does.**

- Rename from "Market Simulator" to something like "Market Screening Tool" or "Directional Market Explorer." The word "simulator" implies a level of physical fidelity that this tool does not have.
- Implement recommendation 6.1.1 (VRE cannibalization feedback) — this is the single change with the highest impact-to-effort ratio.
- Add a methodology disclosure to the UI that honestly states what the model does and doesn't do.
- Position it as a complement to production models, not a substitute. Its strengths (speed, interactivity, comprehensive parameterization, stakeholder accessibility) are genuinely valuable in the phases of analysis that precede a full capacity expansion study.

The energy modeling landscape has room for fast, interactive screening tools — the gap between "back of envelope" and "6-month GenX study" is large, and tools that help stakeholders develop intuition about which variables matter are genuinely useful. This tool can fill that gap if it's honest about what it is.

---

## 7. Summary Assessment

| Dimension | Rating | Notes |
|-----------|--------|-------|
| **Code quality** | Good | Clean, well-documented, properly vectorized |
| **Data sourcing** | Good | Thorough use of public data (PJM SOM, EIA, eGRID, EPA CAMPD) |
| **Dispatch fidelity** | Poor | Merit-order without UC co-optimization, no ramping, no reserves |
| **LMP accuracy** | Poor | Statistical overlays on structurally incomplete dispatch |
| **Capacity expansion** | Poor | No curtailment feedback, no storage co-optimization |
| **Transmission modeling** | Poor | Pipe-and-bubble with proportional stack splitting |
| **Retirement modeling** | Fair | Directionally correct but heuristic |
| **Revenue model** | Good | Multi-stream (energy + capacity + REC + PTC/ITC) |
| **Learning curves** | Good | Well-parameterized Wright's Law with FOAK/NOAK bounds |
| **Scenario coverage** | Good | 270+ parametric scenarios across 9 dimensions |
| **Interactivity / UX** | Good | Clean web UI with parameterized inputs and Plotly charts |
| **Directional accuracy** | Fair | Gets directions right, magnitudes unreliable |
| **Appropriate for investment decisions** | No | |
| **Appropriate for directional screening** | Yes, with caveats | |
| **Appropriate for stakeholder education** | Yes | |

**Overall**: A well-built tool with a weak engine. The parameter library, revenue model, and interactive UI are genuine strengths. The dispatch and capacity expansion methodology would not survive peer review as a "market simulation" but is acceptable as a screening/exploration tool if labeled and used appropriately. The single most impactful improvement would be adding VRE value cannibalization feedback to the deployment model. Without it, the tool systematically overestimates clean energy deployment at high penetration levels — the exact regime where its results are most frequently consulted.
