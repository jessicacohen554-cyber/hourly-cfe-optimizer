# Peer Review: Market Simulator
## A Critical Assessment of Model Validity, Gaps, and Improvement Pathways

**Review Date**: March 2026
**Reviewer**: Independent Technical Review (Claude Code)
**Codebase Version**: Current `main` branch
**Scope**: Full model validity assessment across 8 dimensions

---

## Table of Contents

1. [Assessment Framework](#1-assessment-framework)
2. [Model Validity Assessment](#2-model-validity-assessment)
   - 2.1 [Theoretical Framework](#21-theoretical-framework)
   - 2.2 [LMP Formation](#22-lmp-formation)
   - 2.3 [VRE Integration](#23-vre-integration)
   - 2.4 [Storage Dispatch](#24-storage-dispatch)
   - 2.5 [Retirement & Deployment Logic](#25-retirement--deployment-logic)
   - 2.6 [Data Foundation & Calibration](#26-data-foundation--calibration)
   - 2.7 [Scenario Construction](#27-scenario-construction)
   - 2.8 [Uncertainty Communication](#28-uncertainty-communication)
3. [Identified Gaps](#3-identified-gaps)
4. [Recommendations & Implementation Prompts](#4-recommendations--implementation-prompts)

---

## 1. Assessment Framework

### What Was Reviewed

| File | Lines | Role |
|------|-------|------|
| `scripts/market_simulation.py` | ~3,350 | Core engine: dispatch, deployment, retirement, trajectory |
| `scripts/lmp_engine.py` | ~1,955 | Merit-order LMP formation, ISO price models, ORDC |
| `scripts/pipeline_config.py` | ~998 | Constants, cost tables, regional parameters |
| `scripts/dispatch_utils.py` | ~1,536 | Hourly dispatch reconstruction, storage loops, caching |
| `scripts/zonal_lmp.py` | ~1,070 | Pipe-and-bubble LP zonal decomposition |
| `docs/Model_Methodology_Specification.md` | ~80pp | Methodology documentation |
| `backend/main.py` | ~667 | FastAPI backend, request mapping |
| `backend/models.py` | ~420 | Pydantic schemas, validation |

### How It Was Reviewed

Each dimension was assessed on three axes:

- **Theoretical soundness**: Does the model capture the right economic/physical mechanisms?
- **Implementation fidelity**: Does the code faithfully implement what's described?
- **Fitness for purpose**: Is the model appropriate for its stated use case (profit-driven electricity market screening)?

Rating scale: **Strong** / **Adequate** / **Weak** / **Critical Gap**

### Intended Use Case

The market simulator is a **profit-driven, agent-based generator/investor dispatch model** designed to screen clean energy deployment trajectories across 7 US ISOs under 1,215+ parametric scenarios. It answers: "What gets built, what retires, and at what cost — driven by market economics rather than policy mandates?"

This is fundamentally different from a constrained optimization model (like the hourly CFE optimizer in the parent repo). The simulator lets clean percentage emerge as an *output* of market forces, not an input target. This distinction is critical to evaluating model validity.

---

## 2. Model Validity Assessment

### 2.1 Theoretical Framework

**Rating: Strong**

The model's theoretical foundation is economically coherent and well-suited for its screening purpose.

**Strengths:**

- **Merit-order dispatch is the right abstraction.** The model constructs a fossil merit-order stack sorted by variable cost (coal → oil → gas CT → gas CCGT), dispatches against residual demand after clean supply, and prices at the marginal unit. This is the standard wholesale market clearing mechanism used by all US ISOs. The implementation in `lmp_engine.py` (lines 212–255) correctly decomposes marginal cost into heat rate × fuel price + VOM + emission costs.

- **Profitability-driven deployment is the correct framework for market screening.** Rather than optimizing to a target, the model deploys resources in LCOE merit order when total revenue (energy + capacity + RECs) exceeds LCOE (`market_simulation.py` lines 2058–2260). This captures the actual investment decision framework: private capital deploys when returns are positive.

- **Revenue stacking reflects real-world clean energy economics.** Revenue has three components: (1) temporal energy revenue (profile-weighted hourly LMP), (2) capacity market revenue (ELCC × capacity price), and (3) REC/CES revenue (scarcity-driven ACP pricing). All three are endogenous to the simulation state, creating realistic feedback loops.

- **Wright's Law learning curves for technology cost reduction** are well-established in energy economics (Way et al. 2022, IRENA 2023). The implementation uses pre-2025 cumulative GW as the starting point and applies standard learning rates.

**Limitations:**

- **Wright's Law learning is static within a simulation run** — new deployments during the trajectory don't update cost curves for subsequent years. Acceptable for <10-year horizons but introduces bias in 25-year trajectories where cumulative deployment could halve costs.

- **No explicit financing cost modeling.** LCOE implicitly includes a WACC assumption, but there's no sensitivity to interest rates, project risk premiums, or PPA contract structures beyond a simple discount factor.

- **No explicit fuel hedging or forward contracting.** Gas prices vary by scenario (Low/Medium/High) but are constant within a year — no intra-year volatility or hedging costs.

---

### 2.2 LMP Formation

**Rating: Strong**

The LMP engine is the most technically sophisticated component, with ISO-specific calibration that few screening models attempt.

**Strengths:**

- **Multi-layer pricing architecture.** LMP is constructed from four additive layers (`lmp_engine.py`):
  1. Merit-order marginal cost (lines 1120–1196): `np.searchsorted()` on cumulative capacity for O(H log n) marginal unit identification
  2. Demand-quantile adjustments (lines 1221–1350): congestion adders, must-run depression, negative pricing
  3. ORDC scarcity pricing (lines 622–643): reserve-based exponential adder (LOLP = exp(−λ × reserves))
  4. Demand response dampening (lines 1352–1382): curtailment at $200–300/MWh threshold

- **7 ISO-specific price models** calibrated to 2024 State of the Market reports (PJM SOM, ERCOT PUCT, CAISO DMM, etc.). Each ISO has distinct VOLL caps ($400 NEISO to $5,000 ERCOT), floor prices (−$20 to −$60), and demand-quantile parameters. This level of regional specificity is uncommon in screening models.

- **Load-dependent heat-rate ramp** (lines 1150–1155): `1.0 + 0.15 × (position_in_band)^1.5` captures 6–20% heat rate variation across a generator's operating range. This replaces flat step-function pricing with realistic intra-band price gradation — a meaningful improvement over many production cost models.

- **NEISO winter gas pipeline constraint** (+$13.13/MWh CCS adder, Dec–Feb, lines 1384–1411): Correctly models the real regional bottleneck in New England gas supply during winter peaks.

- **Zonal decomposition option** (`zonal_lmp.py`): Full pipe-and-bubble LP with inter-zonal transfer limits, analytical solvers for 2–5 zone problems (100× faster than per-hour LP), and dual-variable extraction for zonal LMPs. ERCOT 4-zone and PJM 5-zone topologies are properly configured.

**Limitations:**

- **Demand-quantile layer is empirically calibrated, not physics-derived.** The quadratic ramp coefficients, scarcity tail exponents, and mid-low compression factors are fit to historical outcomes rather than derived from first principles. This works well for interpolation but may not extrapolate reliably to unprecedented VRE penetration levels.

- **Unit commitment is simplified.** Min-up/min-down constraints exist (`market_simulation.py` lines 645–687) but cycling costs and startup times are approximations. For a screening model, this is adequate — but users should be aware that short-run marginal cost curves may differ from production-grade UC models by ±5–10%.

- **Zonal LMP is lossless DC.** No I²R transmission losses, no reactive power, no voltage constraints. Transfer limits are symmetric bidirectional (real grids have directional ratings). These simplifications are standard for screening but could matter in congestion-heavy corridors.

---

### 2.3 VRE Integration

**Rating: Adequate**

VRE integration captures the most important economic mechanism (temporal cannibalization) but lacks spatial granularity.

**Strengths:**

- **Temporal cannibalization is correctly modeled.** Per-resource energy revenue is computed as profile-weighted average hourly LMP (`market_simulation.py` lines 1358–1374). As solar/wind penetration increases, generation shifts to low-price hours, reducing capture rates. This is the primary mechanism through which VRE deployment self-limits in market-driven models.

- **VRE penetration scaling in LMP engine** (`lmp_engine.py` lines 1284–1294): Floor depth and negative-price frequency scale linearly with VRE above a 25% baseline, calibrated against CAISO DMM 2019–2024 data showing negative hours increased ~40% as solar went from 20% to 35%.

- **Residual demand formulation** (`market_simulation.py` line 749): `residual = 1.0 − (clean_pct/100) × VRE_profiles` correctly captures the non-linear interaction between VRE shape and fossil dispatch requirements.

**Limitations:**

- **No spatial basis differential for VRE.** All VRE within an ISO receives the system-average (or demand-weighted zonal) LMP. In reality, wind farms in West Texas face persistent basis risk vs. ERCOT Houston load. Solar in CAISO Zone SP15 captures different prices than NP15. This omission likely overestimates VRE revenue in congested ISOs.

- **No curtailment feedback on deployment economics.** While curtailment is tracked in dispatch reconstruction, it doesn't explicitly reduce the LCOE-vs-revenue comparison for marginal VRE projects. High curtailment should reduce effective capacity factor and thus increase effective LCOE.

- **Single weather year as default.** VRE generation profiles come from a single year (2025 default) with optional 2021–2025 sensitivity. No ensemble or synthetic weather year generation, which means tail-risk VRE underperformance isn't captured.

---

### 2.4 Storage Dispatch

**Rating: Weak**

Storage is the most significant modeling gap. The dispatch is either a greedy heuristic or an opt-in LP, and the market simulation engine largely treats storage as a static deployment ramp rather than an economically co-optimized asset.

**Strengths:**

- **Dispatch utilities include both greedy and LP co-optimization paths** (`dispatch_utils.py`):
  - Greedy sequential: Numba `@njit` battery loop (lines 268–311) and LDES loop (lines 315–365) with rolling-window charge/discharge
  - LP co-optimization: `_solve_storage_window_lp()` (lines 589–721) via scipy.linprog with SOC constraints and multi-storage coordination
  - March 2026 bug fix corrected double-counting where LDES and battery consumed the same surplus MWh

- **Technology differentiation is correct**: Battery 4hr (85% RTE, daily cycle), Battery 8hr (85% RTE), LDES 100hr iron-air (50% RTE, 7-day window), Green H₂ 1000hr (35% RTE, 30-day window, ≥95% only). Parameters match industry consensus (NREL ATB 2024, LDES Council).

- **LP co-dispatch** (`co_dispatch_storage_lp`, lines 724–833) handles multi-storage coordination with graceful fallback to greedy on infeasibility.

**Critical Weaknesses:**

- **Market simulation engine bypasses co-optimization entirely.** In `market_simulation.py` (lines 2565–2578), storage deployment is a **static ramp function** of clean percentage:
  ```
  Battery 4hr: min(15%, progress × 12%)
  Battery 8hr: min(8%, max(0, progress − 0.3) × 10%)
  LDES: min(5%, max(0, progress − 0.5) × 8%)
  ```
  These are placeholders, not economics-driven deployment. Storage capacity is not optimized against arbitrage revenue, capacity market revenue, or ancillary services — it's simply ramped as a function of trajectory progress.

- **No hourly charge/discharge optimization in the market engine.** Storage is treated as a demand-side reduction (same as VRE) rather than a time-shifting asset. There's no price-taking storage dispatch model that would charge during low-LMP hours and discharge during high-LMP hours. This means storage revenue is not endogenous to the simulation.

- **Revenue assumption is static** (line 2154): `capacity_credit × capacity_price / (CF × 8.760)` converted to $/MWh. This ignores arbitrage revenue, which is the primary value stream for merchant storage. For a 4hr battery in ERCOT (2024), arbitrage revenue was ~$80–120/kW-yr — comparable to or exceeding capacity revenue.

- **No round-trip efficiency impact on LMP.** Storage charging adds load; storage discharging reduces residual demand. Neither effect is reflected in the LMP calculation, which sees only the net clean percentage. At high storage penetration (>10 GW in ERCOT), charge-induced demand could meaningfully shift overnight LMPs.

---

### 2.5 Retirement & Deployment Logic

**Rating: Adequate**

The retirement cascade and deployment merit order are economically grounded but make simplifying assumptions that users should understand.

**Strengths:**

- **Retirement is margin-driven with graduated response** (`market_simulation.py` lines 969–1067):
  - Stranded (margin < −$5/MWh): 20–90% retirement (scaled by loss depth)
  - At-risk (margin < +$2/MWh): ~10% per year
  - Profitable (margin ≥ +$2/MWh): retained
  - Reliability floor: never retire below 5% of original capacity per unit type

- **Resource adequacy backstop for new fossil** (`apply_economic_new_build`, lines 1069–1280): If reserve margin falls below 15% target after retirements + clean deployment, the model builds cheapest dispatchable capacity (CCGT or CT). This prevents the model from producing physically infeasible grids with insufficient firm capacity.

- **Technology-differentiated interconnection queues** (lines 2194–2256): Per-technology budgets with a 20% flexible pool that can overflow between technologies. Calibrated to LBNL interconnection queue data. This is a meaningful constraint that many models omit.

- **RPS compliance via ACP pricing** (lines 1405–1427): Scarcity-driven exponential REC pricing based on gap to RPS target. Creates a price signal for compliance-driven deployment beyond pure market economics.

**Limitations:**

- **Retirement thresholds are deterministic.** Real-world retirement decisions involve regulatory proceedings (state utility commissions), labor agreements, environmental remediation costs, and political factors. The margin-based model captures the economic signal but not the institutional friction that keeps uneconomic plants running for years.

- **No plant-level heterogeneity in retirement.** The model retires a *fraction* of capacity by unit type, not individual plants. A fleet with one highly profitable CCGT and one deeply uneconomic one retires a fraction of both rather than closing only the unprofitable unit.

- **New fossil CAPEX is static** — no technology learning for CCGT/CT. Gas prices float via fuel sensitivity, but construction costs don't change. For 25-year trajectories, this may underestimate future gas plant costs (labor, permitting, carbon risk premium).

- **Nuclear retirement is binary.** After a trigger year, nuclear either stays fully online or exits entirely (line 3068). No partial fleet retirement or plant-by-plant assessment. Given Constellation, Vistra, and NextEra's plant economics vary substantially, this is a significant simplification.

---

### 2.6 Data Foundation & Calibration

**Rating: Strong**

The data foundation is comprehensive and well-sourced, with appropriate regional specificity.

**Strengths:**

- **EIA data backbone**: Demand profiles from EIA-930, generation from EIA-923, fleet inventory from EIA-860. These are the standard authoritative sources for US electricity data.

- **eGRID emission factors**: CO₂/NOx/SOx rates from EPA eGRID 2022 + CAMPD 2023. Unit-type variability in emission rates (not just fleet averages) is correctly modeled.

- **ISO-specific calibration targets** from 2024 State of the Market reports:
  - PJM: avg $34.70/MWh, P90 $55, 200 negative hours (SOM 2024)
  - ERCOT: avg $26/MWh (Modo Energy 2024)
  - CAISO: avg $38/MWh, extreme duck curve (DMM 2024)
  - NYISO: avg $42/MWh (Potomac Economics 2024)
  - NEISO: avg $39.50/MWh + winter gas adder (EMM 2024)
  - MISO: avg $31/MWh, 35% coal (Potomac 2024)
  - SPP: avg $26.18/MWh, 37% wind (MMU 2024)

- **Pipeline configuration as single source of truth** (`pipeline_config.py`): All constants imported from one module, preventing parameter drift between scripts. This is good engineering practice.

- **Plant-level merit order option** (`lmp_engine.py` lines 442–553): Can load actual EIA 860 generator data with per-unit heat rates, avoiding aggregation error from fleet-average assumptions.

**Limitations:**

- **Capacity market prices are annual snapshots** (`pipeline_config.py` lines 569–576): $/kW-yr values calibrated to recent auctions but don't evolve with the simulation. In a 25-year trajectory, capacity prices should respond to changing reserve margins.

- **Demand response parameters are 2023–2024 vintage** (lines 543–550): DR trigger prices, participation rates, and max GW are static. DR programs are expanding rapidly — FERC Order 2222 alone could double distributed DR participation by 2030.

- **No EIA data freshness tracking.** The model uses whatever data is in `data/`, but there's no automated validation that profiles match the simulation year or flag when data is stale.

---

### 2.7 Scenario Construction

**Rating: Strong**

The 1,215-scenario sweep is well-designed for screening, with meaningful dimensionality and interpretable parameter combinations.

**Strengths:**

- **Nine sweep dimensions** capture the most policy-relevant uncertainties:
  1. Demand growth (3 levels)
  2. Price sensitivity (5 named combos mapping to LCOE/fuel/transmission)
  3. PPA discount level (3)
  4. Gas friction (3: 0.3/0.7/1.0)
  5. Queue cap (3)
  6. New fossil cost (3)
  - Total: 3 × 5 × 3 × 3 × 3 × 3 = **1,215 scenarios** per ISO

- **Named price combos** (all_low, all_med, all_high, high_vre_low_firm, high_firm_low_vre) are more interpretable than exhaustive L/M/H grids across individual parameters. They capture correlated cost movements (e.g., if renewables are cheap, storage likely is too).

- **Gas friction parameter** is a thoughtful addition — it modulates gas scarcity pricing intensity, capturing uncertainty about pipeline constraints, LNG competition, and methane regulation.

- **Single-scenario override** (`build_single_scenario`, lines 2658–2687) allows custom exploration without running the full sweep. Supports custom LCOEs, PTC extensions, CCS credit overrides, etc.

**Limitations:**

- **No correlated scenario construction.** Demand growth, fuel prices, and renewable costs are swept independently, but in reality they're correlated. High demand growth → higher gas prices → faster renewable deployment → lower renewable costs via learning. An internally consistent scenario framework (e.g., IEA WEO scenarios) would improve realism.

- **No tail-risk scenarios.** The sweep covers L/M/H bands but not extreme outcomes: prolonged gas shortage, major nuclear accident affecting fleet policy, breakthrough in fusion or advanced geothermal, or federal carbon tax. These would test model robustness at the boundaries.

- **Binary 45Q toggle** (On/Off) doesn't capture policy uncertainty about credit value, qualification criteria, or phase-out schedules. A graduated 45Q sensitivity (50%/70%/100% realization probability) would be more informative.

---

### 2.8 Uncertainty Communication

**Rating: Weak**

The model generates 1,215 scenarios but provides limited tools for communicating parametric uncertainty to users.

**Strengths:**

- **Scenario sweep enables distributional analysis.** With 1,215 outcomes per ISO, users can construct percentile bands (P10/P50/P90) for any metric. The infrastructure for uncertainty quantification exists.

- **Data tier tracking** (`models.py` SimulationResponse): Multi-tier quality labels signal which results use calibrated data vs. synthetic fallback.

- **IPM trigger indicators** (models.py lines 300–309): Flag when simulation results hit boundaries where a production-grade model would give different answers. This is an honest and valuable meta-uncertainty signal.

**Critical Weaknesses:**

- **No confidence intervals on outputs.** Individual scenario results are point estimates. The model doesn't report P10/P50/P90 bands for clean deployment, cost, or emissions at each year — the user must compute these from raw sweep results.

- **No Monte Carlo or bootstrap.** Each scenario runs once deterministically. No stochastic weather years, no random draws from cost distributions, no parametric uncertainty propagation. The 1,215 scenarios are a structured grid, not a probability-weighted sample.

- **No scenario probability weights.** All 1,215 scenarios are implicitly equiprobable. In reality, "all_med" is more likely than "all_low" or "all_high." Without weights, summary statistics over the sweep are biased toward extreme scenarios that dominate the tails.

- **Synthetic data fallback is silent** (`market_simulation.py` line 2503+): When parquet data is absent, the model silently generates illustrative ramp patterns for storage. Production code should error or at minimum flag results as synthetic-backed. Users could unknowingly make decisions based on placeholder data.

- **No documentation of model limitations in API responses.** The methodology doc covers limitations, but the API response doesn't include a `caveats` or `limitations` field that would travel with the results.

---

## 3. Identified Gaps

### 3.1 Critical Gaps

These gaps could produce materially misleading results if not addressed.

| # | Gap | Location | Impact |
|---|-----|----------|--------|
| C1 | **Storage dispatch is a static ramp stub in the market engine** | `market_simulation.py` lines 2565–2578 | Storage deployment is not economics-driven. Capacity is assigned as a fixed function of clean%, ignoring arbitrage revenue, capacity value, and ancillary services. Results at >60% clean are unreliable because storage economics dominate the marginal deployment decision. |
| C2 | **No hourly storage charge/discharge in market LMP** | `market_simulation.py` — absent | Storage charging adds load (raises off-peak LMP) and discharging reduces peak residual demand (lowers peak LMP). Neither effect is modeled. At >10 GW storage penetration, this could shift LMPs by $5–15/MWh in peak/off-peak spread. |
| C3 | **Synthetic data fallback is silent** | `market_simulation.py` line 2503+ | When PFS parquets are missing, the model generates illustrative data without warning. Users could make investment decisions based on placeholder ramp patterns. Should either error or prominently flag results as "synthetic-backed." |
| C4 | **No VRE spatial basis differential** | `market_simulation.py` lines 1358–1374 | All VRE within an ISO receives system-average LMP. In congested ISOs (ERCOT West→Houston, CAISO SP15→NP15), actual basis differentials are −$5 to −$20/MWh. Overestimates VRE revenue and thus deployment pace. |
| C5 | **Wright's Law learning curves are static within a run** | `market_simulation.py` — deployment loop | Cumulative GW is set at simulation start and not updated as the model deploys new capacity. In a 25-year trajectory deploying 200+ GW of solar, this could overestimate costs by 15–30% in later years. |

### 3.2 Moderate Gaps

These gaps introduce systematic bias but may be acceptable for screening purposes if documented.

| # | Gap | Location | Impact |
|---|-----|----------|--------|
| M1 | **Retirement is fleet-fraction, not plant-level** | `market_simulation.py` lines 969–1067 | Retires a percentage of capacity by unit type rather than individual plants. Over-retires profitable units and under-retires uneconomic ones. |
| M2 | **Nuclear retirement is binary** | `market_simulation.py` line 3068 | Entire nuclear fleet either stays or exits at trigger year. No plant-by-plant assessment. Given 30+ GW US nuclear fleet with highly variable economics, this is a significant simplification. |
| M3 | **Capacity market prices are static** | `pipeline_config.py` lines 569–576 | $/kW-yr values don't respond to changing reserve margins or clean penetration. In a trajectory where 50 GW of coal retires, capacity prices should spike. |
| M4 | **Demand response parameters frozen at 2023–2024** | `pipeline_config.py` lines 543–550 | DR trigger prices, participation rates, and max GW don't evolve. FERC Order 2222 and state programs are expanding DR significantly. |
| M5 | **No correlated scenario construction** | `market_simulation.py` lines 2607–2655 | Demand, fuel, and technology costs are swept independently. In reality, high demand growth correlates with higher gas prices and faster renewable cost decline. |
| M6 | **Demand-quantile pricing layer is empirical, not physical** | `lmp_engine.py` lines 1221–1350 | Calibrated to historical data. May not extrapolate reliably to unprecedented VRE penetration (>60%) where price formation fundamentally changes. |
| M7 | **No curtailment feedback on VRE deployment economics** | `market_simulation.py` — deployment loop | Curtailment is tracked but doesn't reduce effective capacity factor or increase effective LCOE for marginal VRE projects. |
| M8 | **Zonal LMP clean supply not zone-allocated** | `zonal_lmp.py` line 1458 (TODO) | Clean supply is split proportionally rather than by actual zonal location. Overstates clean supply in fossil-heavy zones and understates it in VRE-rich zones. |
| M9 | **No financing cost sensitivity** | `market_simulation.py` — deployment loop | LCOE implicitly includes WACC but there's no toggle for interest rate environments. A 200bp rate increase can shift solar LCOE by $5–8/MWh. |
| M10 | **REC price model not validated** | `market_simulation.py` lines 1405–1427 | Scarcity-driven exponential with per-ISO k_scarcity. High sensitivity to RPS target assumptions. Not benchmarked against actual REC market prices. |

### 3.3 Minor Gaps

These are quality-of-life improvements that would strengthen the model but don't affect core results.

| # | Gap | Location | Impact |
|---|-----|----------|--------|
| m1 | **No input validation for years list monotonicity** | `backend/main.py` | Non-monotonic years (e.g., [2025, 2020, 2030]) will fail silently downstream rather than returning a clear error. |
| m2 | **No bounds check on custom cost overrides** | `backend/main.py` lines 381–524 | Negative LCOE or extreme values accepted without warning. |
| m3 | **Methodology doc missing algorithmic detail** | `docs/Model_Methodology_Specification.md` | ORDC equation, pipe-and-bubble LP formulation, and VRE cannibalization algorithm are described conceptually but not with equations. |
| m4 | **Demand elasticity hardcoded at 3× leverage** | `market_simulation.py` line 600 | No sensitivity to actual demand elasticity. DR response is identical across ISOs despite different program structures. |
| m5 | **LMP caching granularity at 5 GW** | `market_simulation.py` lines 2973–2980 | Buckets fossil capacity to 5 GW increments for LMP reuse. In ISOs with <20 GW fossil (NEISO, SPP), this is >25% granularity — potentially significant. |
| m6 | **Capacity degradation sigmoid parameters are point estimates** | `market_simulation.py` lines 1296–1308 | Midpoint, k, and floor of the capacity price degradation S-curve are fixed per ISO. No sensitivity to auction design changes. |
| m7 | **No API-level result caveats** | `backend/models.py` | SimulationResponse lacks a `limitations` or `data_quality` field that would travel with results to the frontend. |
| m8 | **CORS allows all origins** | `backend/main.py` line 111–123 | `allow_origins=["*"]` with credentials — acceptable for development but should be restricted in production. |

---

## 4. Recommendations & Implementation Prompts

Each recommendation includes a severity tag, estimated effort, and a ready-to-execute Claude Code prompt.

---

### R1. Economics-Driven Storage Deployment (Critical — C1, C2)

**Problem**: Storage deployment is a hardcoded ramp function of clean%, not driven by market economics.

**Recommendation**: Replace the static ramp with a price-taking storage dispatch model. For each candidate storage technology, compute arbitrage revenue from the hourly LMP profile (charge at bottom quartile, discharge at top quartile), add capacity market revenue and ancillary services, compare total revenue against annualized storage cost, and deploy only if profitable.

**Estimated Effort**: 2–3 sessions

**Claude Code Prompt**:
```
In market-simulator/scripts/market_simulation.py, replace the static storage
ramp (lines 2565–2578) with an economics-driven storage deployment model:

1. After computing hourly LMP for the current year (compute_lmp_at_threshold),
   run a price-taking storage dispatch for each candidate technology (battery
   4hr, battery 8hr, LDES, Green H2):
   - Use the existing _battery_loop / _ldes_loop from dispatch_utils.py
   - Input: hourly_lmp profile, storage parameters from pipeline_config
   - Compute annual arbitrage revenue = sum((discharge_price - charge_price) *
     MWh * RTE)
   - Add capacity revenue from pipeline_config.CAPACITY_MARKET_PRICES
   - Add ancillary services from compute_storage_revenue_credit()

2. Compare total revenue against annualized storage LCOE (from pipeline_config
   STORAGE_PARAMS). Deploy capacity where revenue > LCOE, subject to queue cap.

3. Feed deployed storage back into LMP recalculation: storage charging adds
   to demand in low-price hours, discharging reduces residual demand in high-
   price hours. This requires a second LMP pass (iterate until convergence or
   cap at 2 iterations).

4. Update the synthetic data fallback to raise an explicit warning when
   parquets are missing rather than silently generating ramp data.

Key files: market_simulation.py (deployment loop), dispatch_utils.py (storage
kernels), pipeline_config.py (storage parameters and costs).
```

---

### R2. Endogenous Wright's Law Learning (Critical — C5)

**Problem**: Cumulative GW is frozen at simulation start. Multi-decade trajectories understate cost reduction from within-run deployments.

**Recommendation**: Update cumulative installed capacity at the end of each simulation year and recompute LCOE for the next year using the updated learning curve position.

**Estimated Effort**: 1 session

**Claude Code Prompt**:
```
In market-simulator/scripts/market_simulation.py, make Wright's Law learning
curves endogenous to the simulation trajectory:

1. In run_market_simulation() (around line 2728), initialize a cumulative_gw
   dict tracking installed capacity by technology (solar, wind, offshore_wind,
   battery, etc.) using the existing 2025 baseline values.

2. At the end of each simulation year (after compute_market_deployment), add
   newly deployed GW to cumulative_gw for each technology.

3. Before the LCOE merit-order in the next year's deployment loop, recompute
   LCOE using the updated cumulative_gw:
   LCOE_new = LCOE_initial * (cumulative_gw_new / cumulative_gw_baseline) ^ log2(learning_rate)

4. Store the year-over-year LCOE trajectory in the YearResult for
   visualization (add lcoe_trajectory field to models.py YearResult).

5. Add a toggle (ENDOGENOUS_LEARNING = True/False) in pipeline_config.py so
   the static behavior is preserved as a comparison mode.
```

---

### R3. Silent Synthetic Fallback → Explicit Error/Warning (Critical — C3)

**Problem**: Missing parquet data triggers silent generation of illustrative ramp patterns.

**Recommendation**: Replace silent fallback with explicit error in production mode and a prominently flagged warning in development mode.

**Estimated Effort**: 0.5 session

**Claude Code Prompt**:
```
In market-simulator/scripts/market_simulation.py, fix the silent synthetic
data fallback (around line 2503):

1. Add a SYNTHETIC_DATA_MODE flag to pipeline_config.py with values:
   - "error": raise RuntimeError when parquets are missing (production default)
   - "warn": generate synthetic data but add a prominent warning to results
   - "silent": current behavior (for backward compatibility only)

2. When mode is "warn", add a data_quality field to every YearResult:
   data_quality = {"synthetic_backed": True, "missing_sources": [...]}

3. In backend/models.py, add data_quality: Optional[dict] to YearResult and
   SimulationResponse.

4. In the frontend, display a yellow banner when any result has
   synthetic_backed=True: "⚠️ Results use synthetic placeholder data. Run
   the full pipeline (Steps 1-2) for calibrated results."

5. Log a WARNING-level message listing which parquets were missing and what
   synthetic data was generated.
```

---

### R4. VRE Basis Differential by Zone (Critical — C4)

**Problem**: VRE revenue uses system-average LMP rather than zone-specific prices.

**Recommendation**: When zonal LMP is active, compute per-resource energy revenue using the LMP of the zone where that resource type is predominantly located.

**Estimated Effort**: 1 session

**Claude Code Prompt**:
```
In market-simulator/scripts/market_simulation.py, add VRE basis differentials:

1. In pipeline_config.py, add VRE_PRIMARY_ZONE mapping per ISO:
   VRE_PRIMARY_ZONE = {
       "ERCOT": {"solar": "West", "wind": "West", "offshore_wind": "Coast"},
       "CAISO": {"solar": "SP15", "wind": "SP15"},
       "PJM": {"solar": "West", "wind": "West", "offshore_wind": "MAAC"},
       ...
   }

2. In compute_energy_revenue_by_resource() (lines 1358–1374), when zonal LMP
   data is available:
   - Look up the primary zone for each resource type
   - Use that zone's hourly LMP instead of system-average LMP
   - Compute profile-weighted average: revenue = mean(profile * zonal_lmp)

3. Fall back to system-average LMP when zonal data is unavailable (copper-
   plate mode).

4. Add a basis_differential field to ZoneDetail in models.py showing the
   $/MWh difference between resource zone LMP and system LMP.

Key files: market_simulation.py (revenue calculation), zonal_lmp.py (zonal
LMP data), pipeline_config.py (zone mapping).
```

---

### R5. Confidence Intervals on Sweep Results (Moderate — Uncertainty)

**Problem**: 1,215 scenarios produce point estimates with no distributional summary.

**Recommendation**: Add post-processing to compute P10/P50/P90 bands across the scenario sweep for key metrics and include scenario probability weights.

**Estimated Effort**: 1 session

**Claude Code Prompt**:
```
In market-simulator/scripts/market_simulation.py, add uncertainty
quantification to the sweep results:

1. After run_full_sweep() completes, add a post-processing function
   compute_sweep_uncertainty() that:
   - Groups results by (ISO, year)
   - For each group, computes P10/P25/P50/P75/P90 across scenarios for:
     clean_pct, total_cost_per_mwh, emissions_mt, deployed_gw, lmp_avg
   - Returns a dict: {iso: {year: {metric: {p10, p25, p50, p75, p90}}}}

2. Add optional scenario probability weights in pipeline_config.py:
   SCENARIO_WEIGHTS = {
       "demand": {"Low": 0.2, "Medium": 0.6, "High": 0.2},
       "price": {"all_low": 0.1, "all_med": 0.5, "all_high": 0.1,
                 "high_vre_low_firm": 0.15, "high_firm_low_vre": 0.15},
       ...
   }
   Combined weight = product of individual dimension weights.

3. In backend/models.py, add UncertaintyBands to SimulationResponse:
   class UncertaintyBands(BaseModel):
       metric: str
       p10: float
       p25: float
       p50: float
       p75: float
       p90: float

4. Include weighted and unweighted percentiles in the response.
```

---

### R6. Plant-Level Retirement (Moderate — M1, M2)

**Problem**: Retirement operates on fleet fractions, not individual plant economics.

**Recommendation**: Use the plant-level merit order from `lmp_engine.py` to evaluate individual plant profitability and retire the specific units that are uneconomic.

**Estimated Effort**: 2 sessions

**Claude Code Prompt**:
```
In market-simulator/scripts/market_simulation.py, replace fleet-fraction
retirement with plant-level economics:

1. Use build_plant_level_merit_order() from lmp_engine.py (lines 442–553)
   to load individual plant data (EIA 860 + heat rates + VOM).

2. In apply_economic_retirement() (lines 969–1067):
   - For each plant in the fleet, compute plant-specific margin:
     margin = (hourly_lmp * dispatch_hours * CF) - (heat_rate * fuel_price +
     VOM + CO2_rate * CO2_price) * dispatched_MWh - fixed_costs
   - Rank plants by margin (worst first)
   - Retire plants with margin < threshold (current -$5/MWh rule)
   - Preserve reliability floor per zone, not per unit type

3. For nuclear: evaluate each plant's contract revenue vs. operating cost.
   Plants with below-market offtake contracts (e.g., Comanche Peak at
   $35/MWh) may be stranded while others remain profitable. Use the
   NUCLEAR_OFFTAKE_CONTRACTS dict in pipeline_config.py.

4. Store retired plant IDs in state for tracking across years.

Key files: market_simulation.py (retirement logic), lmp_engine.py (plant-level
stack), pipeline_config.py (nuclear contracts).
```

---

### R7. Endogenous Capacity Market Prices (Moderate — M3)

**Problem**: Capacity prices are static $/kW-yr values that don't respond to reserve margin changes.

**Recommendation**: Make capacity prices a function of the current reserve margin, using the existing sigmoid degradation curve but in reverse — as reserves tighten, prices should increase.

**Estimated Effort**: 1 session

**Claude Code Prompt**:
```
In market-simulator/scripts/pipeline_config.py and market_simulation.py,
make capacity market prices endogenous:

1. In pipeline_config.py, add a capacity price curve function:
   def compute_capacity_price(iso, reserve_margin_pct, clean_pct):
       base_price = CAPACITY_MARKET_PRICES[iso]
       # Scarcity: price increases as reserve margin falls below target
       if reserve_margin_pct < 15:
           scarcity_mult = 1.0 + 2.0 * (15 - reserve_margin_pct) / 15
       else:
           scarcity_mult = 1.0
       # Clean penetration degradation (existing sigmoid)
       clean_mult = capacity_degradation_factor(iso, clean_pct)
       return base_price * scarcity_mult * clean_mult

2. In market_simulation.py, replace static CAPACITY_MARKET_PRICES[iso]
   lookups with compute_capacity_price(iso, current_reserve_margin,
   current_clean_pct).

3. This creates a feedback loop: fossil retirements → lower reserves →
   higher capacity prices → new fossil builds become economic → reserves
   recover → prices stabilize.

4. Add capacity_price_trajectory to YearResult for visualization.
```

---

### R8. Input Validation Hardening (Minor — m1, m2, m7)

**Problem**: API accepts invalid inputs without clear errors.

**Recommendation**: Add Pydantic validators and a dry-run validation endpoint.

**Estimated Effort**: 0.5 session

**Claude Code Prompt**:
```
In market-simulator/backend/models.py and main.py, add input validation:

1. In models.py SimulationRequest, add Pydantic validators:
   @validator('iso')
   def validate_iso(cls, v):
       if v not in ISOS:
           raise ValueError(f"ISO must be one of {ISOS}")
       return v

   @validator('years')
   def validate_years_monotonic(cls, v):
       if v and v != sorted(v):
           raise ValueError("Years must be in ascending order")
       return v

   @root_validator
   def validate_cost_bounds(cls, values):
       for field in ['solar', 'wind', 'nuclear', 'ccs']:
           lcoe = getattr(values.get('clean_lcoes', CleanLCOEs()), field, None)
           if lcoe is not None and (lcoe < 0 or lcoe > 500):
               raise ValueError(f"{field} LCOE must be 0-500 $/MWh")
       return values

2. In main.py, add a dry-run validation endpoint:
   @app.post("/api/validate-request")
   async def validate_request(req: SimulationRequest):
       conditions = _map_request_to_conditions(req)
       # Check data availability
       missing = check_required_parquets(conditions)
       return {"valid": len(missing) == 0, "missing_data": missing,
               "scenario_count": compute_scenario_count(conditions)}

3. Add a result_caveats field to SimulationResponse listing any limitations
   relevant to this specific run (e.g., "Zonal LMP not available for SPP",
   "Storage deployment uses static ramp — see C1").
```

---

### R9. Methodology Documentation Completeness (Minor — m3)

**Problem**: Methodology doc describes *what* is modeled but not *how* — missing algorithmic equations.

**Recommendation**: Add mathematical formulations for the three most complex subsystems.

**Estimated Effort**: 1 session

**Claude Code Prompt**:
```
In market-simulator/docs/Model_Methodology_Specification.md, add algorithmic
detail for three key subsystems:

1. **ORDC Scarcity Pricing** (add to §4.4.6):
   - Equation: LOLP(h) = exp(-λ × max(0, reserves(h) - knee_mw))
   - Adder(h) = min(cap, VOLL × LOLP(h))
   - Table of per-ISO parameters from pipeline_config.ORDC_PARAMS
   - Reference: lmp_engine.py compute_ordc_adder() lines 622–643

2. **Pipe-and-Bubble LP Formulation** (add to §4.4.1):
   - Decision variables: g[z][u] (generation), f_pos[i]/f_neg[i] (flows)
   - Objective: minimize sum(mc[u] × g[u])
   - Constraints: zonal balance, capacity bounds, transfer limits
   - Dual extraction: zonal LMP = shadow price on balance constraint
   - Reference: zonal_lmp.py lines 549–633

3. **VRE Cannibalization / Capture Rate** (add to §4.4.7):
   - Revenue(r) = mean(profile(r,h) × LMP(h)) for h in 1..8760
   - Capture rate = Revenue(r) / mean(LMP)
   - VRE floor scaling: 1.0 + 1.5 × max(0, vre_pct - 0.25), capped at 2×
   - Reference: lmp_engine.py lines 1284–1294, market_simulation.py
     lines 1358–1374
```

---

### R10. Curtailment Feedback on VRE Economics (Moderate — M7)

**Problem**: Curtailment is tracked but doesn't affect the deployment merit order.

**Recommendation**: Reduce effective capacity factor for marginal VRE projects based on system curtailment rates at the current clean percentage.

**Estimated Effort**: 0.5 session

**Claude Code Prompt**:
```
In market-simulator/scripts/market_simulation.py, add curtailment feedback
to VRE deployment economics:

1. In compute_lmp_at_threshold(), track curtailment_rate = curtailed_mwh /
   total_vre_generation_mwh for the current dispatch.

2. In compute_market_deployment() (lines 2058–2260), before the LCOE
   merit-order comparison:
   - For solar and wind, compute effective_cf = base_cf * (1 - curtailment_rate)
   - Adjust effective LCOE: lcoe_effective = lcoe_base / (1 - curtailment_rate)
   - This makes marginal VRE more expensive as curtailment rises

3. Store curtailment_rate in YearResult for trend visualization.

4. This creates a natural saturation point: as VRE deployment increases →
   curtailment rises → effective LCOE increases → VRE becomes less
   competitive → deployment slows. Currently this feedback loop is missing,
   allowing unrealistic VRE overdeployment.
```

---

## Appendix: Summary Scorecard

| Dimension | Rating | Key Strength | Primary Gap |
|-----------|--------|-------------|-------------|
| Theoretical Framework | **Strong** | Merit-order + profitability-driven deployment | Static learning curves |
| LMP Formation | **Strong** | 7 ISO-specific models, ORDC + demand-quantile | Empirical extrapolation risk |
| VRE Integration | **Adequate** | Temporal cannibalization modeled | No spatial basis differential |
| Storage Dispatch | **Weak** | LP co-optimization exists in dispatch_utils | Not wired into market engine |
| Retirement/Deployment | **Adequate** | Margin-driven + RA backstop | Fleet-fraction, not plant-level |
| Data & Calibration | **Strong** | EIA/eGRID/SOM sources, per-ISO calibration | Static capacity prices |
| Scenario Construction | **Strong** | 1,215 scenarios, 9 dimensions | No correlation structure |
| Uncertainty Communication | **Weak** | IPM triggers, data tiers | No confidence intervals |

**Overall Assessment**: The market simulator is a well-engineered screening tool with a sound economic foundation. Its LMP engine and scenario sweep are notably sophisticated for a screening model. The critical gap is storage dispatch — the model's weakest link is precisely the technology class whose economics will most determine market outcomes in the 2025–2050 period. Addressing C1/C2 (storage economics) would meaningfully improve model validity. The remaining moderate gaps (plant-level retirement, endogenous capacity prices, correlated scenarios) are worth addressing but don't undermine the model's usefulness for its stated screening purpose.

---

*End of Peer Review*
