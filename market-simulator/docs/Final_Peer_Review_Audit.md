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

---

## 6. Claude Code Implementation Prompts

The following prompts are designed to be executed sequentially, each completable in a single Claude Code session. Dependencies are noted — prompts without dependencies can be reordered freely.

### Prompt 1: G3 — Result Provenance (0.5 session, no dependencies)

```
In market-simulator/backend/models.py and market-simulator/scripts/market_simulation.py,
add result provenance metadata to all simulation outputs:

1. In models.py, add a ProvenanceMetadata model:
   class ProvenanceMetadata(BaseModel):
       model_version: str          # e.g., "2.1.0"
       git_sha: str                # short SHA of current commit
       git_branch: str             # current branch name
       config_hash: str            # SHA-256 of pipeline_config.py contents
       run_timestamp: str          # ISO 8601 UTC timestamp
       python_version: str         # sys.version
       input_snapshot: dict        # full request parameters as submitted

2. Add a provenance field to SimulationResponse:
   provenance: Optional[ProvenanceMetadata] = None

3. In market_simulation.py, add a helper function build_provenance_metadata()
   that:
   - Reads git SHA via subprocess: git rev-parse --short HEAD
   - Reads git branch via: git rev-parse --abbrev-ref HEAD
   - Computes SHA-256 hash of pipeline_config.py file contents
   - Captures current timestamp and Python version
   - Accepts the input request dict as a parameter
   - Returns a ProvenanceMetadata instance

4. Call build_provenance_metadata() at the start of run_single_simulation()
   and run_full_sweep(), attaching it to the response.

5. Add a test in scripts/tests/test_r9_qa_qc.py:
   - TestProvenance class with tests verifying:
     - git_sha is non-empty string
     - config_hash is 64-char hex string
     - run_timestamp is valid ISO 8601
     - input_snapshot round-trips correctly

Key files: backend/models.py, scripts/market_simulation.py, scripts/pipeline_config.py
```

---

### Prompt 2: G2 — Methodology Equation Documentation (1 session, no dependencies)

```
In market-simulator/docs/Model_Methodology_Specification.md, add mathematical
formulations for two subsystems that currently lack equations:

1. **Pipe-and-Bubble LP Formulation** (add to §4.4.1 Zonal LMP Decomposition):

   Reference implementation: zonal_lmp.py lines 549–633

   Document the following:
   - Decision variables: g[z][u] (generation per zone per unit), f_pos[i]/f_neg[i]
     (positive/negative flow on each interface)
   - Objective: minimize Σ(mc[u] × g[z][u]) over all zones z and units u
   - Constraints:
     a. Zonal power balance: Σ g[z][u] + Σ f_in[z] - Σ f_out[z] = demand[z]
     b. Generation bounds: 0 ≤ g[z][u] ≤ capacity[z][u]
     c. Transfer limits: -limit[i] ≤ f[i] ≤ limit[i]
   - Dual extraction: zonal LMP[z] = shadow price (dual variable) on balance
     constraint for zone z
   - Note the analytical solver path (closed-form for 2-zone cases) vs. scipy
     linprog for >2 zones

2. **VRE Cannibalization / Capture Rate Model** (add to §4.4.7):

   Reference implementation: lmp_engine.py lines 1284–1294,
   market_simulation.py lines 1358–1374

   Document:
   - Energy revenue: Revenue(r) = mean(profile(r,h) × LMP(h)) for h ∈ {1..8760}
   - Capture rate: CR(r) = Revenue(r) / mean(LMP)
   - VRE penetration floor scaling: floor_mult = 1.0 + 1.5 × max(0, vre_pct − 0.25),
     capped at 2.0
   - Interaction with basis differential (R4): zone-specific LMP replaces system
     LMP when zonal data available
   - Interaction with curtailment feedback (R10): effective_lcoe = lcoe / (1 − curtailment_rate)

3. Add a cross-reference table at the end of each equation section mapping
   equation variables to code variables and line numbers.

Key files: docs/Model_Methodology_Specification.md, scripts/zonal_lmp.py,
scripts/lmp_engine.py, scripts/market_simulation.py
```

---

### Prompt 3: G1 — Plant-Level Retirement (2 sessions, no dependencies)

```
In market-simulator/scripts/market_simulation.py, replace fleet-fraction
retirement with plant-level economics. This is a 2-session task.

SESSION 1 — Plant-Level Retirement Engine:

1. The plant-level merit order already exists: build_plant_level_merit_order()
   in lmp_engine.py (lines 442–553) loads EIA 860 plant data with capacity,
   heat rates, VOM, fuel type, and location. compute_plant_level_economics()
   in market_simulation.py (lines 868–997) already computes per-plant margins
   and classifies plants as stranded/at_risk/operating.

2. Refactor apply_economic_retirement() (lines 999–1067) to:
   a. Accept plant_economics (output of compute_plant_level_economics) instead
      of fleet-level gen_econ
   b. Sort plants by margin (worst first)
   c. Retire individual plants with margin < -$5/MWh (current threshold)
   d. Track retired plant IDs in state['retired_plants'] for inter-year persistence
   e. Enforce reliability floor per zone, not per unit type: maintain minimum
      reserve margin of 15% based on zonal peak demand
   f. Return list of retired plant dicts (ID, capacity, type, margin, zone)

3. For nuclear: use NUCLEAR_OFFTAKE_CONTRACTS dict from pipeline_config.py
   to evaluate each plant's contract revenue vs. operating cost individually.
   Plants with below-market offtake contracts are at higher stranding risk than
   those with market-rate or regulated-rate contracts. Do NOT retire the entire
   nuclear fleet at a single trigger year.

4. Add plant_retirements field to YearResult in models.py:
   plant_retirements: List[dict] = []  # [{plant_id, capacity_mw, unit_type, margin, iso, zone}]

5. Add unit tests:
   - Test that high-margin plants survive while low-margin plants retire
   - Test reliability floor prevents over-retirement
   - Test nuclear plants retire individually based on contract economics
   - Test retired plant IDs persist across simulation years

SESSION 2 — Integration and Validation:

6. Wire plant-level retirement into the main simulation loop
   (run_single_simulation), replacing the fleet-fraction call.

7. Run a single-ISO (PJM) validation comparing fleet-fraction vs. plant-level
   retirement trajectories across 3 scenarios. Document the difference in
   retirement timing, reserve margins, and capacity price trajectories.

8. Update the existing regression tests to work with the new retirement
   interface.

Key files: market_simulation.py (lines 868–1067, 2190–2260),
lmp_engine.py (lines 442–553), pipeline_config.py (NUCLEAR_OFFTAKE_CONTRACTS),
backend/models.py (YearResult)
```

---

### Prompt 4: G5 — Demand-Quantile Extrapolation Guard (0.5 session, no dependencies)

```
In market-simulator/scripts/lmp_engine.py and market_simulation.py, add explicit
confidence degradation for LMP results above 60% VRE penetration:

1. In lmp_engine.py, after compute_hourly_lmp_vectorized() returns LMP array,
   add a confidence_factor field to the result:
   - If vre_penetration <= 0.50: confidence = 1.0 (fully calibrated)
   - If 0.50 < vre_penetration <= 0.60: confidence = 1.0 (within calibration)
   - If 0.60 < vre_penetration <= 0.75: confidence = 0.8 (moderate extrapolation)
   - If 0.75 < vre_penetration <= 0.90: confidence = 0.6 (significant extrapolation)
   - If vre_penetration > 0.90: confidence = 0.4 (beyond model validity)

2. In market_simulation.py, when VRE > 60%, add to the existing IPM trigger
   system:
   - ipm_triggers.append({
       "type": "lmp_extrapolation",
       "severity": "warning" if vre < 0.75 else "critical",
       "message": f"VRE penetration {vre*100:.0f}% exceeds calibration range...",
       "recommendation": "Validate LMP results with production cost model"
     })

3. Add lmp_confidence_factor to YearResult in models.py.

4. Add a test verifying confidence degrades correctly at each VRE bracket.

Key files: lmp_engine.py, market_simulation.py, backend/models.py
```

---

### Prompt 5: G6 — Cross-Validation Benchmarks (1 session, run after G1 if plant retirement implemented)

```
Create market-simulator/scripts/tests/validate_cross_model.py — a validation
script that compares model outputs against published reference-case trajectories:

1. Define reference data from publicly available sources:
   a. EIA AEO 2025 Reference Case — renewable share trajectory 2025–2050
      (Table 8 "Electricity Generation by Fuel")
   b. NREL ReEDS Standard Scenarios 2024 — Mid-case clean share + capacity additions
   c. EPA IPM v6 Reference Case — coal retirement schedule + gas build trajectory

   Hard-code these as dicts in the script (small data, publicly available,
   won't change). Include source citations.

2. Run the market simulator for a "reference-equivalent" scenario:
   - Medium demand, Medium LCOE, Medium gas price, Medium carbon ($0/ton for
     AEO comparison, $51/ton for EPA comparison)
   - ERCOT + PJM + CAISO (3 largest ISOs)
   - 2025–2050 trajectory

3. For each ISO × reference model pair, compute:
   - Clean share divergence at 2030, 2035, 2040
   - Capacity addition rate divergence (GW/year)
   - Coal retirement timing divergence (year of <5% coal share)

4. Generate a structured comparison table and narrative:
   - Expected divergences (this model is profit-driven, others are cost-minimizing)
   - Unexplained divergences (potential calibration issues)
   - Document the comparison as market-simulator/docs/Cross_Validation_Results.md

5. Add to the test suite as a non-blocking validation (warnings, not failures)
   since divergence from optimization models is expected and intentional.

Key files: New script + new doc. References: market_simulation.py (run_single_simulation),
pipeline_config.py (scenario parameters)
```

---

### Prompt 6: G4 — Correlated Scenario Construction (1 session, no dependencies)

```
In market-simulator/scripts/pipeline_config.py and market_simulation.py, add
IEA-aligned correlated scenario bundles alongside the existing independent sweep:

1. In pipeline_config.py, add CORRELATED_SCENARIOS dict:
   CORRELATED_SCENARIOS = {
       "IEA_STEPS": {  # Stated Policies
           "description": "Current policies continue, moderate ambition",
           "demand_growth": "Medium",
           "gas_price": "Medium",
           "renewable_lcoe": "Medium",
           "carbon_price": 0,
           "learning_rate": "Medium",
           "45q": True,
       },
       "IEA_APS": {  # Announced Pledges
           "description": "All announced national commitments implemented",
           "demand_growth": "Medium",
           "gas_price": "Medium",
           "renewable_lcoe": "Low",  # faster cost decline
           "carbon_price": 51,       # EPA SCC
           "learning_rate": "Fast",
           "45q": True,
       },
       "IEA_NZE": {  # Net Zero by 2050
           "description": "1.5C-aligned pathway",
           "demand_growth": "High",  # electrification drives demand
           "gas_price": "High",      # carbon costs embedded
           "renewable_lcoe": "Low",
           "carbon_price": 185,      # Rennert et al.
           "learning_rate": "Fast",
           "45q": True,
       },
       "HIGH_FRICTION": {  # Stress test
           "description": "Regulatory/permitting delays + high costs",
           "demand_growth": "High",
           "gas_price": "Low",       # cheap gas delays transition
           "renewable_lcoe": "High",
           "carbon_price": 0,
           "learning_rate": "Slow",
           "45q": False,
       },
       "RAPID_TRANSITION": {  # Bull case for clean energy
           "description": "Technology breakthroughs + strong policy",
           "demand_growth": "High",
           "gas_price": "High",
           "renewable_lcoe": "Low",
           "carbon_price": 100,
           "learning_rate": "Fast",
           "45q": True,
       },
   }

2. In market_simulation.py, add run_correlated_scenarios() function that:
   - Accepts an ISO and list of scenario names (default: all 5)
   - Maps each scenario to the appropriate parameter combination
   - Runs simulations and returns results keyed by scenario name
   - These are NOT added to the independent sweep — they run separately

3. In backend/main.py, add an endpoint:
   @app.post("/api/correlated-scenarios")
   async def run_correlated(iso: str, scenarios: List[str] = None):
       ...

4. In backend/models.py, add CorrelatedScenarioResponse with scenario metadata.

5. Add tests verifying each scenario maps to valid parameter combinations and
   produces distinct trajectories.

Key files: pipeline_config.py, market_simulation.py, backend/main.py, backend/models.py
```

---

### Prompt 7: G7 — Sensitivity Analysis Framework (1–2 sessions, run after G4)

```
Create market-simulator/scripts/sensitivity_analysis.py — a post-processing
module that decomposes output variance across the 1,215-scenario sweep:

SESSION 1 — Morris Method Screening:

1. Implement Morris method (elementary effects) screening:
   - For each input dimension (demand, gas_price, renewable_lcoe, carbon_price,
     learning_rate, 45q), compute the mean and standard deviation of elementary
     effects on key outputs (clean_pct, total_cost_per_mwh, emissions_mt)
   - This uses existing sweep results — no new simulations needed
   - Read sweep results from the API or from cached JSON

2. Generate a Morris plot (mean vs. std of elementary effects) per ISO:
   - High mean + high std = non-linear, important
   - High mean + low std = linear, important
   - Low mean + low std = unimportant
   - Output as JSON data suitable for Chart.js visualization

3. Implement first-order variance decomposition (one-at-a-time):
   - For each dimension, compute fraction of total output variance explained
   - Output as tornado diagram data (sorted bar chart of variance fractions)

SESSION 2 — Integration:

4. Add a tornado_data field to the sweep uncertainty response (from R5)
   containing the variance decomposition results.

5. Create market-simulator/docs/Sensitivity_Analysis_Results.md documenting:
   - Which inputs matter most per ISO
   - Non-linear interactions identified
   - Recommendations for where to invest in better input data

6. Add tests verifying variance fractions sum to <= 1.0 and Morris elementary
   effects are computed correctly on a known test case.

Key files: New script. References: market_simulation.py (sweep results),
pipeline_config.py (scenario dimensions), backend/models.py (response models)
```

---

### Prompt 8: Documentation & Architecture Updates (1–2 sessions, run AFTER Prompts 1–7)

This prompt ensures all user-facing documentation, methodology specifications, frontend pages, and architecture diagrams reflect every change implemented in Prompts 1–7. Run this last — it captures the final state.

```
After implementing Prompts 1–7 (G3 provenance, G2 methodology equations, G1 plant-level
retirement, G5 extrapolation guard, G6 cross-validation, G4 correlated scenarios,
G7 sensitivity analysis), update ALL documentation artifacts to reflect the new
capabilities. This is a documentation-only prompt — no model code changes.

TARGET FILES (6 documents):
  1. market-simulator/USER_MANUAL.md (573 lines)
  2. market-simulator/docs/Model_Methodology_Specification.md (1,592 lines)
  3. market-simulator/frontend/guide.html (544 lines)
  4. market-simulator/frontend/methodology.html (526 lines)
  5. market-simulator/docs/architecture-high-level.html (318 lines)
  6. market-simulator/docs/architecture-detailed.html (584 lines)

─────────────────────────────────────────────────
PART A — USER_MANUAL.md
─────────────────────────────────────────────────

Update the user manual to document all new features and API endpoints:

1. **New API Endpoints section** — Add documentation for:
   - POST /api/correlated-scenarios (from G4): parameters, example request/response,
     available scenario names (IEA_STEPS, IEA_APS, IEA_NZE, HIGH_FRICTION, RAPID_TRANSITION)
   - POST /api/validate-request: already exists but verify it's documented

2. **Result Provenance section** (from G3): Explain the new provenance metadata
   fields in simulation responses (model_version, git_sha, config_hash,
   run_timestamp, input_snapshot). Explain how users can use this for audit trails
   and result reproducibility.

3. **Plant-Level Retirement section** (from G1): Document the new plant_retirements
   field in year results. Explain:
   - How individual plant economics drive retirement (vs. old fleet-fraction approach)
   - The reliability floor (15% reserve margin per zone)
   - Nuclear retirement is now per-plant based on contract economics
   - How retired_plants persist across simulation years

4. **LMP Confidence & Extrapolation Warnings section** (from G5): Document:
   - The lmp_confidence_factor field in year results
   - Confidence degradation brackets (≤50%=1.0, 50-60%=1.0, 60-75%=0.8,
     75-90%=0.6, >90%=0.4)
   - IPM trigger: "lmp_extrapolation" type with severity levels
   - How users should interpret results at high VRE penetration

5. **Correlated Scenarios section** (from G4): Document:
   - The 5 IEA-aligned scenario bundles and what each represents
   - How correlated scenarios differ from the independent 1,215 sweep
   - When to use correlated scenarios vs. independent sweep

6. **Sensitivity Analysis section** (from G7): Document:
   - What Morris method screening tells users (which inputs matter most)
   - How to interpret tornado diagrams and variance decomposition
   - The tornado_data field in sweep uncertainty results

7. **Cross-Validation section** (from G6): Document:
   - What the cross-validation script compares against (AEO, ReEDS, EPA IPM)
   - How to run validate_cross_model.py
   - How to interpret divergences (expected vs. unexplained)

8. **Directory Structure**: Update the directory tree to include:
   - scripts/sensitivity_analysis.py
   - scripts/tests/validate_cross_model.py
   - docs/Cross_Validation_Results.md
   - docs/Sensitivity_Analysis_Results.md

─────────────────────────────────────────────────
PART B — Model_Methodology_Specification.md
─────────────────────────────────────────────────

Note: Prompt 2 (G2) already adds LP formulation and VRE cannibalization equations.
This prompt adds the broader methodology updates needed from the other prompts.

1. **Plant-Level Retirement** (from G1): Add or update the retirement methodology
   section to document:
   - Per-plant margin calculation: margin = energy_revenue + capacity_revenue
     + ancillary_revenue - variable_cost - fixed_cost
   - Retirement threshold: margin < -$5/MWh (configurable)
   - Retirement ordering: plants sorted by margin, worst-first
   - Reliability floor constraint: minimum 15% reserve margin per zone based
     on zonal peak demand
   - Nuclear: per-plant evaluation using NUCLEAR_OFFTAKE_CONTRACTS, not
     fleet-wide binary trigger
   - State persistence: retired plant IDs tracked across simulation years in
     state['retired_plants']

2. **Result Provenance** (from G3): Add a section on output metadata and
   reproducibility. Document the ProvenanceMetadata schema and how it enables
   audit trails.

3. **Extrapolation Guard** (from G5): Add to the LMP formation section:
   - Confidence degradation formula at VRE > 60%
   - IPM trigger integration for lmp_extrapolation warnings
   - Calibration range boundaries (2019–2024 ISO State of Market data)

4. **Correlated Scenarios** (from G4): Add a section describing:
   - The 5 IEA-aligned scenario bundles with parameter mappings
   - Rationale: why correlated scenarios complement independent sweep
   - Relationship to R5 confidence intervals (weighted percentiles)

5. **Sensitivity Decomposition** (from G7): Add a section describing:
   - Morris method elementary effects computation
   - First-order variance decomposition methodology
   - How sensitivity results are aggregated per ISO

6. **Cross-Validation Framework** (from G6): Add a section describing:
   - Reference models and data sources (AEO 2025, ReEDS 2024, EPA IPM v6)
   - Comparison metrics (clean share, capacity additions, coal retirement timing)
   - Expected divergences due to mechanism difference (profit-driven vs.
     cost-minimizing)

7. **Version History**: Add entries for all G1–G7 implementations with dates.

─────────────────────────────────────────────────
PART C — guide.html (Frontend Guide Page)
─────────────────────────────────────────────────

Update the user-facing guide page to reflect new capabilities:

1. **New Features callout**: Add a "What's New" or "Recent Enhancements" section
   near the top highlighting:
   - Plant-level retirement (more realistic fleet economics)
   - Correlated scenario bundles (IEA-aligned analysis)
   - Sensitivity analysis (identify which assumptions matter most)
   - Result provenance (audit trail for every simulation)
   - LMP confidence indicators (transparency at high VRE)

2. **Updated workflow steps**: If the guide has a step-by-step workflow, add:
   - Step for selecting correlated scenarios vs. independent sweep
   - Note about checking lmp_confidence_factor in high-VRE results
   - Note about provenance metadata in results JSON

3. **Interpreting Results section**: Add guidance on:
   - plant_retirements field in year results
   - lmp_confidence_factor interpretation
   - tornado_data for sensitivity analysis
   - How to use provenance metadata for result traceability

4. **Data & Methodology link**: Ensure the guide links to the updated
   methodology.html page.

─────────────────────────────────────────────────
PART D — methodology.html (Frontend Methodology Page)
─────────────────────────────────────────────────

Update the technical methodology page to reflect all model changes:

1. **Retirement Mechanism section**: Update from fleet-fraction description to
   plant-level retirement. Include:
   - Brief description of per-plant margin calculation
   - Reliability floor constraint
   - Nuclear per-plant evaluation
   - Link to full equations in Model_Methodology_Specification.md

2. **LMP Formation section**: Add:
   - Confidence degradation at high VRE penetration
   - Calibration range disclosure (2019–2024)
   - IPM trigger integration

3. **Scenario Construction section**: Add:
   - Correlated scenario bundles table (5 scenarios with parameter mappings)
   - How correlated scenarios complement the independent sweep
   - Sensitivity analysis methodology (Morris method, variance decomposition)

4. **Model Comparison table**: Update the comparison table (if present) to
   reflect new capabilities:
   - Plant-level retirement: ✓ (was ✗)
   - Sensitivity decomposition: ✓ (was ✗)
   - Correlated scenarios: ✓ (was ✗)
   - Result provenance: ✓ (was ✗)

5. **Known Limitations section**: Update to reflect which limitations have been
   addressed (G1–G7) and which remain (G8–G11).

6. **Cross-Validation section**: Add summary of cross-validation results from
   docs/Cross_Validation_Results.md with link to full document.

─────────────────────────────────────────────────
PART E — Architecture Diagrams
─────────────────────────────────────────────────

Update both architecture HTML documents to reflect structural changes:

1. **architecture-high-level.html** (318 lines):
   - Add "Provenance Layer" box showing metadata injection into responses
   - Update "Retirement Module" label from "Fleet-Fraction" to "Plant-Level"
   - Add "Sensitivity Analysis" post-processing module in the pipeline flow
   - Add "Correlated Scenarios" as an input path alongside "Parametric Sweep"
   - Add "Cross-Validation" as an output/validation step
   - Add "Confidence Guard" annotation on the LMP engine box

2. **architecture-detailed.html** (584 lines):
   - Update the Retirement section with plant-level flow:
     compute_plant_level_economics() → sort by margin → retire worst-first
     → check reliability floor → persist retired_plants
   - Add ProvenanceMetadata to the data model section
   - Add sensitivity_analysis.py to the scripts section
   - Add validate_cross_model.py to the validation section
   - Add /api/correlated-scenarios to the API endpoint listing
   - Update data flow arrows to show:
     - ProvenanceMetadata attached at simulation start
     - lmp_confidence_factor computed in LMP engine
     - plant_retirements returned in YearResult
     - tornado_data returned in sweep response

─────────────────────────────────────────────────
VERIFICATION CHECKLIST
─────────────────────────────────────────────────

After all updates, verify:
□ USER_MANUAL.md documents all 7 new features (G1–G7)
□ USER_MANUAL.md directory tree includes new files
□ Model_Methodology_Specification.md has methodology sections for all 7 features
□ Model_Methodology_Specification.md version history updated
□ guide.html mentions all new capabilities
□ guide.html workflow steps reflect new options
□ methodology.html retirement section updated to plant-level
□ methodology.html scenario section includes correlated bundles
□ methodology.html limitations section reflects current gap status
□ architecture-high-level.html shows provenance, plant-level retirement,
  sensitivity analysis, correlated scenarios, confidence guard
□ architecture-detailed.html shows updated data models, API endpoints,
  script inventory, and data flow
□ All internal cross-references between documents are consistent
□ No stale references to "fleet-fraction retirement" remain in any document
□ No stale references to missing features (sensitivity, correlated scenarios) as
  "not yet implemented"

Key files:
  USER_MANUAL.md
  docs/Model_Methodology_Specification.md
  frontend/guide.html
  frontend/methodology.html
  docs/architecture-high-level.html
  docs/architecture-detailed.html
```

---

## 7. Conclusion

### 7.1 Current Fitness Assessment

The market simulator has materially improved since the original peer review. The implementation of 8 of 10 recommendations — verified by 304 independent checks — addressed the two most critical weaknesses:

1. **Storage dispatch** moved from static ramps to economics-driven deployment with arbitrage revenue, capacity market credits, and technology-differentiated dispatch. This directly addresses the gap that was identified as the model's weakest link.

2. **Uncertainty communication** moved from point estimates to P10/P50/P90 bands with scenario probability weights, synthetic data warnings, and input validation. Users now receive structured uncertainty information rather than single-scenario results.

### 7.2 Readiness Assessment

| Use Context | Ready? | Prerequisites |
|---|---|---|
| Internal strategic screening | **Yes** | Use P10/P50/P90 ranges, note IPM triggers |
| Client presentations | **Yes, with caveats** | Include Section 4.5 interpretation guidance |
| Peer review / publication | **Not yet** | Complete G1 (plant-level retirement), G2 (methodology equations), G6 (cross-validation) |
| Regulatory filing support | **Not yet** | Complete G1, G3 (provenance), G5 (extrapolation guards), G6, G7 (sensitivity) |
| Investment-grade decisions | **Never standalone** | Always validate screening results with production cost models (PLEXOS, IPM) |

### 7.3 Recommended Implementation Sequence

For teams with limited capacity, the highest-impact improvements in order:

1. **G3 (Result Provenance)** — 0.5 session, immediate governance benefit
2. **G2 (Methodology Equations)** — 1 session, documentation-only, enables peer review
3. **G1 (Plant-Level Retirement)** — 2 sessions, largest model improvement remaining
4. **G5 (Extrapolation Guard)** — 0.5 session, protective guardrail
5. **G6 (Cross-Validation)** — 1 session, credibility for external audiences
6. **G4 (Correlated Scenarios)** — 1 session, better distributional analysis
7. **G7 (Sensitivity Analysis)** — 1–2 sessions, decision-support enhancement
8. **Documentation & Architecture Updates** — 1–2 sessions, run AFTER all code changes. Updates USER_MANUAL.md, Model_Methodology_Specification.md, guide.html, methodology.html, and both architecture diagrams to reflect everything implemented in Prompts 1–7.

**Total estimated effort**: 8.5–10 sessions for all high and medium priority items plus documentation. The 4 low-priority items (G8–G11) add 2.5–3.5 sessions but are optional for the model's stated screening purpose.

### 7.4 Final Assessment

The market simulator is a well-engineered screening tool that has demonstrably improved through systematic peer review and implementation. Its profit-driven, agent-based approach fills a genuine gap in the modeling landscape — most available tools are cost-minimizing capacity expansion models that treat clean energy targets as constraints rather than emergent outcomes.

The remaining gaps are proportional: plant-level retirement (G1) would bring the model from adequate to strong in its weakest remaining dimension, while provenance (G3) and cross-validation (G6) are standard practices for any model used in decision-making contexts. None of the remaining gaps undermine the model's core screening utility when used within its stated boundary conditions.

---

*End of Final Peer Review Audit*
