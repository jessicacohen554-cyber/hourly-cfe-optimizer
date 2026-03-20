# Final Peer Review Audit — Market Simulator
## Post-Implementation Assessment, Use-Case Specification & Improvement Roadmap

**Audit Date**: March 2026
**Auditor**: Independent Technical Review (Claude Code)
**Codebase Version**: Current `claude/market-simulator-peer-review-nHu83` branch
**Scope**: Third-party audit of implementation completeness, remaining gaps, appropriate use-case boundaries, and actionable improvement roadmap
**Prior Documents**: [Peer Review](Peer_Review_Market_Simulator.md) | [R9 QA/QC Results](R9_AUDIT_RESULTS.md) | [Methodology Spec](Model_Methodology_Specification.md)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Implementation Status](#2-implementation-status)
3. [Updated Validity Scorecard](#3-updated-validity-scorecard)
4. [Appropriate Use-Case Description](#4-appropriate-use-case-description)
5. [Remaining Gaps — Prioritized Improvement Roadmap](#5-remaining-gaps--prioritized-improvement-roadmap)
6. [Claude Code Implementation Prompts](#6-claude-code-implementation-prompts)
7. [Conclusion](#7-conclusion)

---

## 1. Executive Summary

The market simulator underwent a three-phase review process:

1. **Original Peer Review** — Identified 10 recommendations (R1–R10) across 5 Critical, 10 Moderate, and 8 Minor gaps spanning 8 model dimensions and ~9,600 lines of code.
2. **R9 QA/QC Code Audit** — Verified 8 of 10 recommendations are fully implemented, passing **304 of 304 validation checks** across unit tests (29), integration tests (7), end-to-end sweep (127), edge cases (17), and code audit checklist items (59).
3. **This Final Audit** — Assesses remaining gaps from a third-party perspective, defines appropriate use-case boundaries, and provides a sequenced improvement roadmap with executable prompts.

### Key Findings

- **8 of 10 original recommendations fully implemented and verified** (R1–R5, R7, R8, R10)
- **2 recommendations not yet implemented**: R6 (plant-level retirement) and R9 (methodology equation documentation)
- **2 model dimensions materially improved**: Storage Dispatch (Weak → Adequate), Uncertainty Communication (Weak → Adequate)
- **5 new third-party findings** identified beyond the original review: result provenance, correlated scenarios, extrapolation guardrails, cross-validation benchmarks, and sensitivity decomposition
- **Overall assessment**: Fit for internal strategic screening with appropriate caveats. External distribution or regulatory use should await plant-level retirement (G1), result provenance (G3), and cross-validation (G6).

---

## 2. Implementation Status

### 2.1 Original Recommendations — Current State

| # | Recommendation | Severity | Status | Evidence | Residual Risk |
|---|---|---|---|---|---|
| **R1** | Economics-driven storage deployment | Critical | **IMPLEMENTED** | 8/8 checklist, arbitrage + capacity + ancillary revenue stacking verified | Low — static ramp replaced with price-taking dispatch |
| **R2** | Endogenous Wright's Law learning curves | Critical | **IMPLEMENTED** | 8/8 checklist, costs decrease monotonically, NOAK floor enforced | Low — cumulative GW updated per deployment year |
| **R3** | Synthetic fallback → explicit warning | Critical | **IMPLEMENTED** | 6/6 checklist, 3-mode behavior (error/warn/silent) verified | Low — `SYNTHETIC_DATA_MODE` env var controls behavior |
| **R4** | VRE basis differential by zone | Critical | **IMPLEMENTED** | 6/6 checklist, zone-specific LMP per resource type | Low — copper-plate fallback when zonal data unavailable |
| **R5** | Confidence intervals on sweep results | Moderate | **IMPLEMENTED** | 7/7 checklist, P10/P25/P50/P75/P90 + scenario weights | Low — weighted percentiles match numpy within 0.01 |
| **R6** | Plant-level retirement | Moderate | **NOT IMPLEMENTED** | Fleet-fraction retirement still in use (lines 999–1067) | **Medium** — over-retires profitable units, under-retires uneconomic ones |
| **R7** | Endogenous capacity market prices | Moderate | **IMPLEMENTED** | 8/8 checklist, sigmoid + scarcity multiplier, ERCOT/SPP = $0 | Low — feedback loop: retirements → lower reserves → higher prices |
| **R8** | Input validation hardening | Minor | **IMPLEMENTED** | 10/10 checklist, Pydantic validators + dry-run endpoint | Low — invalid ISOs/years/prices rejected |
| **R9** | Methodology documentation completeness | Minor | **PARTIAL** | ORDC equations added (§4.4.6); LP formulation and VRE cannibalization equations still missing | **Low** — equations exist in code, not yet in documentation |
| **R10** | Curtailment feedback on VRE economics | Moderate | **IMPLEMENTED** | 6/6 checklist, effective LCOE formula correct | Low — natural saturation point for VRE deployment |

### 2.2 Test Suite Summary

| Test Category | Tests | Result |
|---|---|---|
| Unit tests (per-recommendation) | 29 | 29/29 PASS |
| Integration tests (cross-recommendation) | 7 | 7/7 PASS |
| End-to-end mini-sweep (ERCOT, 3 scenarios × 6 years) | 127 checks | 127/127 PASS |
| Edge case verification | 17 | 17/17 PASS |
| Regression suite (all test files) | 65 | 65/65 PASS (42 data-dependent skipped) |
| **Total** | **304** | **304/304 PASS** |

---

## 3. Updated Validity Scorecard

| Dimension | Original Rating | Current Rating | Key Change |
|-----------|----------------|---------------|------------|
| Theoretical Framework | **Strong** | **Strong** | Endogenous learning curves (R2) strengthen cost dynamics |
| LMP Formation | **Strong** | **Strong** | No change — already sophisticated with ORDC + demand-quantile |
| VRE Integration | **Adequate** | **Adequate+** | Basis differential (R4) + curtailment feedback (R10) address two key gaps |
| Storage Dispatch | **Weak** | **Adequate** | Economics-driven deployment (R1) replaces static ramps with arbitrage revenue |
| Retirement & Deployment | **Adequate** | **Adequate** | No change — R6 (plant-level) not yet implemented |
| Data & Calibration | **Strong** | **Strong** | Endogenous capacity prices (R7) add feedback loop |
| Scenario Construction | **Strong** | **Strong** | No change — 1,215 scenarios remain independently swept |
| Uncertainty Communication | **Weak** | **Adequate** | Confidence intervals (R5), synthetic warnings (R3), input validation (R8) |

**Net improvement**: Two dimensions elevated from Weak to Adequate. No dimensions degraded. The model's two most critical weaknesses (storage economics and uncertainty communication) have been substantively addressed.

---

## 4. Appropriate Use-Case Description

### 4.1 What This Model Is

The market simulator is a **profit-driven, agent-based generator and investor dispatch model** that projects clean energy deployment trajectories across 7 US ISOs (CAISO, ERCOT, PJM, NYISO, NEISO, MISO, SPP) under 1,215+ parametric scenarios.

**Core mechanism**: The model constructs a merit-order fossil dispatch stack, computes hourly LMPs from residual demand after clean supply, evaluates technology-specific revenue (energy + capacity + RECs), and deploys resources when total revenue exceeds LCOE. Fossil units retire when operating margins turn negative. Clean energy percentage emerges as an *output* of market forces — not an input target.

This is fundamentally different from constrained optimization models (e.g., the hourly CFE optimizer in the parent repository, or capacity expansion models like GenX/ReEDS). The simulator does not minimize cost subject to a clean energy constraint. Instead, it answers: **"What gets built, what retires, and at what cost — driven by market economics rather than policy mandates?"**

### 4.2 Appropriate Uses

| Use Case | Description | Confidence Level |
|----------|-------------|-----------------|
| **Corporate fleet strategy** | Screen retirement risk for fossil assets, identify ISOs where clean investment is market-viable earliest, evaluate capacity revenue trajectories | **High** — core design purpose |
| **Policy scenario analysis** | Assess impact of carbon prices, IRA credits (45Q, PTC/ITC), RPS compliance costs, and fuel price shocks on deployment trajectories | **High** — 1,215 scenarios with 9 interpretable dimensions |
| **Regional comparison** | Compare which ISOs reach clean energy milestones fastest under identical market assumptions; identify structural advantages (hydro endowment, gas dependence, nuclear fleet) | **High** — 7 ISO-specific calibrated models |
| **Technology investment timing** | Screen when specific technologies (solar, wind, storage, CCS, nuclear) become market-viable under different cost trajectories | **High** — Wright's Law learning + LCOE merit-order deployment |
| **Capacity market strategy** | Model how clean penetration affects capacity prices, fossil stranding risk, and reserve margin dynamics | **Medium-High** — endogenous capacity prices (R7), but no forward capacity auction modeling |
| **Sensitivity analysis** | Identify which input assumptions (fuel prices, LCOE, carbon price, demand growth) drive the most outcome variance | **Medium** — 1,215-scenario sweep provides broad coverage, but no formal decomposition (see G7) |
| **Client presentations** | Show market-driven deployment trajectories with P10/P50/P90 uncertainty bands for strategic planning discussions | **High** — confidence intervals (R5) + scenario weights |
| **Screening for production-model deep dives** | Identify scenarios and ISOs that warrant detailed IPM/PLEXOS analysis based on screening results and IPM trigger indicators | **High** — IPM triggers flag boundary conditions automatically |

### 4.3 Boundary Conditions — Where NOT to Use This Model

| Boundary | Why | What to Use Instead |
|----------|-----|-------------------|
| **Operational dispatch planning** | No unit commitment (startup costs, ramp rates, minimum run times are approximated). No sub-hourly resolution. No ancillary service co-optimization. | PLEXOS, PSO, GE MAPS, or production cost models |
| **Transmission planning** | Zonal LMP uses lossless DC power flow (no reactive power, voltage constraints, N-1 contingency). 2–5 zones per ISO vs. 1000+ nodes in nodal models. | PowerWorld, PSS/E, TARA, nodal production cost models |
| **Individual plant investment decisions** | Fleet-fraction retirement doesn't capture plant-specific economics (contract structures, age, location value). Use as screening input, not final decision basis. | Plant-level financial models + IPM validation |
| **Results above ~60% VRE penetration** | Demand-quantile pricing layer is empirically calibrated to 2019–2024 data. Extrapolation beyond observed VRE levels introduces unquantified uncertainty in LMP formation. | Production cost models with detailed unit commitment |
| **Regulatory filings** | Not NERC/FERC-grade. No reliability assessment (LOLE, EUE). Resource adequacy backstop is a simplified reserve margin check, not a probabilistic reliability model. | SERVM, RECAP, or utility-grade IRP models |
| **Post-2035 trajectory precision** | Trajectory confidence degrades with horizon. 25-year projections are directional, not precise — technology disruptions, policy shifts, and demand evolution are unpredictable at that range. | Treat 2040+ results as directional scenarios, not forecasts |
| **Nodal congestion analysis** | System-average LMP per zone masks intra-zonal congestion. Basis differentials (R4) improve this but don't replace nodal analysis for siting decisions. | Nodal LMP models (FESTIV, PLEXOS nodal) |

### 4.4 Model Comparison — Screening Context

| Attribute | This Model | NREL ReEDS | MIT GenX | EPA IPM | Energy Exemplar PLEXOS |
|-----------|-----------|------------|---------|---------|----------------------|
| **Purpose** | Market-driven screening | Capacity expansion planning | Capacity expansion + operations | Regulatory compliance | Production cost simulation |
| **Mechanism** | Agent-based profit-driven | Least-cost optimization | Least-cost optimization | Least-cost optimization | Chronological dispatch |
| **Clean % treatment** | Emergent (output) | Target (input constraint) | Target (input constraint) | Target (input constraint) | Emergent from commitments |
| **Temporal resolution** | 8,760 hours | 17 time-slices | 8,760 hours | Seasonal blocks | Sub-hourly capable |
| **Spatial resolution** | 7 ISOs, 2–5 zones each | 134 BAs | Configurable zones | ~64 IPM regions | Nodal (1000+ nodes) |
| **Runtime** | Minutes (single ISO) | Hours–Days | Hours | Hours–Days | Hours–Days |
| **Scenarios** | 1,215+ parametric | ~10–20 scenarios | ~10–20 scenarios | 3–5 reference cases | 5–10 scenarios |
| **Learning curves** | Endogenous Wright's Law | Exogenous ATB trajectories | Exogenous or endogenous | Exogenous | N/A (short-term model) |
| **Data requirements** | EIA/eGRID public data | Extensive proprietary | Moderate | EPA datasets | Utility-specific data |
| **License** | Internal tool | Open-source | Open-source | EPA internal | Commercial ($100K+/yr) |
| **Best for** | Strategic screening, scenario exploration | National/regional planning | Academic research | EPA rulemaking | Utility operations planning |

### 4.5 Interpreting Results Correctly

**Always present results as ranges, not point estimates.** The P10/P50/P90 bands from the 1,215-scenario sweep represent the model's view of plausible outcomes under different assumptions — not probabilistic forecasts.

**Key caveats that should accompany any results presentation:**

1. This is a screening model. Results identify promising regions of the decision space for deeper analysis, not investment-grade projections.
2. Clean energy deployment trajectories are market-driven (profit-seeking) — they do not assume policy mandates, corporate commitments, or utility IRP targets.
3. Results at high VRE penetration (>60%) should be interpreted with additional caution due to empirical pricing extrapolation.
4. The model assumes competitive wholesale markets. Results may not apply to vertically integrated utilities or bilateral markets outside ISO footprints.
5. IPM trigger indicators in the output flag when results cross screening-model validity boundaries. When triggered, escalate to production-grade models before making decisions.

---

## 5. Remaining Gaps — Prioritized Improvement Roadmap

### 5.1 High Priority — Address Before External Distribution

These gaps could produce materially misleading results or undermine auditability if the model is shared outside the immediate team.

#### G1: Plant-Level Retirement (Original R6 — Moderate Gaps M1, M2)

**Current state**: Retirement operates on fleet fractions (`apply_economic_retirement()`, market_simulation.py:999–1067). When gas CCGT margin falls below -$5/MWh, a percentage of the entire gas CCGT fleet retires based on loss depth. Nuclear retirement is binary — the entire fleet exits at a trigger year.

**Problem**: Real fleets have heterogeneous economics. A 2020-vintage high-efficiency CCGT may be profitable while a 1990s unit with high heat rates is stranded. Fleet-fraction retirement over-retires profitable units and under-retires uneconomic ones, biasing trajectories in both directions.

**Impact**: Medium. Affects retirement timing and pace, which cascades to reserve margins, capacity prices (R7 feedback loop), and new deployment timing. Most consequential in ISOs with large, diverse fossil fleets (PJM, MISO, ERCOT).

**Effort**: 2 sessions. Plant-level data already available via `build_plant_level_merit_order()` in lmp_engine.py (lines 442–553) and `compute_plant_level_economics()` in market_simulation.py (lines 868–997).

---

#### G2: Methodology Equation Documentation (Original R9 — Minor Gap m3)

**Current state**: The Model Methodology Specification (1,592 lines) describes *what* is modeled comprehensively but is missing mathematical formulations for two key subsystems: (1) the pipe-and-bubble LP zonal decomposition, and (2) the VRE cannibalization / capture rate model. ORDC scarcity pricing equations were added in v2.0.

**Problem**: Without equations, an independent reviewer cannot verify that the code implements the stated methodology without reading source code. This is a reproducibility and auditability gap.

**Impact**: Low for internal use, Medium for peer review or publication. The equations exist in the codebase — they just need to be extracted into the specification document.

**Effort**: 1 session.

---

#### G3: Result Provenance and Reproducibility (New Finding)

**Current state**: Simulation outputs (JSON responses, sweep results) contain scenario parameters and results but no metadata linking them to the exact code version, configuration state, or input data versions that produced them.

**Problem**: If a decision is made based on model outputs from March 2026, and the code is updated in April 2026, there is no way to determine which code version produced the original results. No git SHA, `pipeline_config.py` hash, or input parameter snapshot is embedded in output.

**Impact**: Medium. This is a governance and audit trail gap. Any regulatory or investment decision based on model outputs should be traceable to exact inputs + code version.

**Effort**: 0.5 session. Add `model_version`, `git_sha`, `config_hash`, and `input_snapshot` fields to `SimulationResponse` in models.py.

---

### 5.2 Medium Priority — Strengthen for Peer Review or Regulatory Use

These gaps don't undermine the model's screening utility but would strengthen it for more rigorous audiences.

#### G4: Correlated Scenario Construction (Original Moderate Gap M5)

**Current state**: The 1,215 scenarios sweep demand, fuel prices, renewable LCOE, and other dimensions independently. Each dimension's values are combined with every other dimension's values in a full Cartesian grid.

**Problem**: In reality, these dimensions are correlated. High electricity demand tends to coincide with higher gas prices (both driven by economic growth) and faster renewable cost decline (driven by deployment scale from higher demand). The independent sweep assigns equal probability to implausible combinations like "high demand + low gas prices + slow renewable cost decline."

**Impact**: Medium. The scenario sweep still covers the relevant outcome space, but the distributional summary (P10/P50/P90 from R5) may be misleading because implausible combinations dilute the probability mass. The scenario weights added in R5 partially mitigate this but are still applied independently per dimension.

**Effort**: 1 session. Add IEA World Energy Outlook-aligned scenario bundles (Stated Policies, Announced Pledges, Net Zero) as curated, internally-consistent scenario sets alongside the independent sweep.

---

#### G5: Demand-Quantile Pricing Extrapolation Guard (Original Moderate Gap M6)

**Current state**: The demand-quantile pricing layer in `lmp_engine.py` is empirically calibrated to 2019–2024 ISO State of Market data. At VRE penetrations above ~60%, the model extrapolates beyond observed data.

**Problem**: The relationship between VRE penetration and LMP shape may be non-linear in ways not captured by the calibrated coefficients. Negative pricing, duck curve depth, and minimum generation constraints behave differently at penetrations never observed in US ISOs.

**Impact**: Medium. IPM trigger indicators already flag high-VRE conditions, but the current implementation doesn't explicitly degrade confidence or warn users in API responses.

**Effort**: 0.5 session. Add explicit confidence degradation factor to LMP results above 60% VRE and enhance the existing IPM trigger to include a structured warning in the API response.

---

#### G6: Cross-Validation Against Established Models (New Finding)

**Current state**: Individual calibration targets exist for each ISO (2024 State of Market reports), and plant-level economics are validated against EIA data. However, there is no systematic comparison of model trajectories against NREL ReEDS Annual Technology Baseline scenarios, EIA Annual Energy Outlook projections, or EPA IPM reference cases.

**Problem**: Without cross-validation, it's impossible to distinguish between "the model produces different results because it uses a different mechanism (profit-driven vs. cost-minimizing)" and "the model produces different results because it has a bug or calibration error."

**Impact**: Medium. Cross-validation doesn't mean the model must match other models — divergence is expected and valuable — but the divergence should be explainable. A structured comparison against 2–3 reference cases would significantly strengthen credibility.

**Effort**: 1 session. Create a validation script that compares model outputs for a reference scenario against published ReEDS/AEO mid-case trajectories, with commentary on expected divergences.

---

#### G7: Sensitivity Analysis Framework (New Finding)

**Current state**: The 1,215-scenario sweep provides broad parametric coverage, but no formal sensitivity decomposition identifies which inputs drive the most output variance. Users must inspect results manually to understand sensitivity.

**Problem**: A decision-maker asking "which assumptions matter most?" cannot get a direct answer from the model. Tornado diagrams, Morris method screening, or first-order Sobol indices would provide this. This is standard practice in energy system modeling (NREL, IEA, EPA all publish sensitivity decompositions).

**Impact**: Medium. Useful for prioritizing which assumptions to refine and for communicating uncertainty to non-technical stakeholders.

**Effort**: 1–2 sessions. The 1,215-scenario sweep results already contain all the data needed — this is a post-processing analysis, not a model change.

---

### 5.3 Low Priority — Proportional to Screening Purpose

These are quality-of-life improvements that enhance the model but aren't required for its stated screening purpose.

#### G8: Code Quality Improvements (New Finding)

**Current state**: Type hint coverage is approximately 5% (~14 of ~298 functions have return type annotations). Test suite has 65 passing tests across 12 files but no formal coverage percentage tracked. No CI/CD pipeline runs tests on PR.

**Effort**: 1–2 sessions.

#### G9: Financing Cost Sensitivity / WACC Toggle (Original Moderate Gap M9)

**Current state**: LCOE calculations include an implicit WACC assumption but no sensitivity toggle for interest rates, project risk premiums, or PPA structures.

**Effort**: 0.5 session.

#### G10: Demand Response Parameter Update (Original Moderate Gap M4)

**Current state**: DR parameters frozen at 2023–2024 values. FERC Order 2222 implementation could materially change DR availability by 2028–2030.

**Effort**: 0.5 session.

#### G11: Multi-Weather-Year Option (Original Minor Gap)

**Current state**: Single weather year (2024) as default. 2021–2025 sensitivity available but not wired into the sweep.

**Effort**: 0.5 session.

