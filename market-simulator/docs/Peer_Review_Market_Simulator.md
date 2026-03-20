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
