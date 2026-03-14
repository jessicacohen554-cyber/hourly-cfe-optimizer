# Independent Technical Review: Electricity Market Simulator
## Architecture, Assumptions, and Uncertainty Characterization

**Version**: 1.0
**Date**: March 14, 2026
**Classification**: Internal — Peer Review
**Model Reviewed**: Market Simulator v1.0.0
**Upstream Pipeline**: Hourly CFE Optimizer Steps 0–7

---

## Executive Summary

This document presents an independent technical review of the Electricity Market Simulator, a profit-driven screening tool that models clean energy deployment, fossil retirement, and wholesale price formation across seven U.S. Independent System Operators (ISOs). The tool is designed to narrow the search window for compute-intensive production models (GenX, IPM, ReEDS), not to replace them.

**Overall Assessment**: The tool is **fit for purpose as a directional screening instrument**. Its core architecture — merit-order LMP formation, profit-driven deployment, and parametric sensitivity coverage — is sound and well-implemented. The 270-scenario sweep mode explores more of the uncertainty space in 30 minutes than GenX/IPM can cover in a week. This throughput advantage is the tool's primary competitive differentiator.

**Confidence Ratings by Mode**:

| Mode | Confidence | Appropriate Use |
|------|-----------|-----------------|
| Snapshot (2025) | High | Current-state economics, directional policy screening |
| Trajectory (2030–2035) | Moderate | Trend identification, order-of-magnitude resource planning |
| Trajectory (2040–2050) | Low–Moderate | Scenario exploration only, not investment-grade |
| Sweep (270 scenarios) | High | Sensitivity ranking, robust policy conclusions |

**Principal Risks Identified**:

1. **Emissions data format mismatch** (Critical — Fixed): The market simulator's emission rates file contained only aggregate `co2_rate` per ISO, while `compute_fossil_retirement()` expected per-fuel-type rates (`coal_co2_lb_per_mwh`, etc.). All emission calculations returned zero. **Status: Resolved** — replaced with full eGRID per-fuel-type data and corrected the caller to use remaining fleet emission rate.

2. **Uniform cost-based offer adder** (Moderate — Fixed): The 10% PJM Manual 15 markup was applied to all ISOs including energy-only markets (ERCOT, SPP), creating a systematic +$2–4/MWh LMP bias. **Status: Resolved** — made ISO-specific (0% for ERCOT/SPP, 7% for MISO, 10% for capacity-market ISOs).

3. **Linear capacity market degradation** (Moderate — Fixed): Linear degradation overstated price erosion at low clean shares and understated it in the transition zone. **Status: Resolved** — replaced with sigmoid S-curve calibrated against PJM RPM and NYISO ICAP clearing data.

4. **Copper-plate transmission** (Structural — Acceptable): No intra-ISO transmission constraints. Overestimates achievable clean penetration by 3–8%. Appropriate for a screening tool.

5. **Greedy sequential storage dispatch** (Structural — Acceptable): Battery → LDES → H₂ dispatch order underestimates co-optimized storage value by 5–15%. Conservative bias is preferable for a screening tool.

---

## Table of Contents

1. [Scope and Standards of Review](#1-scope-and-standards-of-review)
2. [Physics Foundation Review (Step 1 PFS)](#2-physics-foundation-review)
3. [Cost Optimization Review (Step 2)](#3-cost-optimization-review)
4. [LMP Engine Review](#4-lmp-engine-review)
5. [Market Simulation Engine Review](#5-market-simulation-engine-review)
6. [Data Quality Assessment](#6-data-quality-assessment)
7. [Edge Cases and Failure Modes](#7-edge-cases-and-failure-modes)
8. [Statistical Framework for Uncertainty](#8-statistical-framework-for-uncertainty)
9. [Comparative Positioning](#9-comparative-positioning)
10. [Recommendations](#10-recommendations)
11. [Appendices](#appendices)

---

## 1. Scope and Standards of Review

### 1.1 Tool Purpose

The Market Simulator answers: *"What happens to generators under different market conditions?"* Clean energy deployment is an **output** that emerges from profitability calculations, not a mandated target. This distinguishes it from optimization models that minimize cost subject to constraints.

The tool serves as a **screening instrument** — its value lies in rapidly exploring parameter space to identify which combinations of fuel prices, carbon costs, policy settings, and technology costs produce meaningful clean energy deployment. Results should inform where to spend compute capacity on full production models, not serve as investment-grade capacity expansion plans.

### 1.2 Review Methodology

This review applies four assessment layers:

1. **Code-level inspection**: Line-by-line review of core algorithms in `market_simulation.py`, `lmp_engine.py`, `dispatch_utils.py`, and `pipeline_config.py`.
2. **Assumption validation**: Cross-reference all input parameters against published sources (NREL ATB 2024, Lazard v17/v18, LBNL Utility-Scale Solar/Wind, PJM SOM 2024, EPA eGRID 2022).
3. **Structural sensitivity**: Identify model architecture decisions that introduce systematic bias and quantify their magnitude.
4. **Edge-case analysis**: Test model behavior at parameter extremes and identify failure modes.

### 1.3 Assessment Framework

Each finding is classified as:

| Classification | Definition | Action Required |
|---------------|------------|-----------------|
| **(a) Fit for purpose** | Directionally correct, bias within acceptable bounds for screening | None |
| **(b) Systematic bias** | Quantifiable magnitude, consistent direction | Document bias direction and magnitude; consider correction |
| **(c) Structural limitation** | May produce anomalous results under specific conditions | Document conditions; flag to users |

---

## 2. Physics Foundation Review

### 2.1 Grid Search Adequacy

The upstream optimizer (Step 1 PFS) uses a 4D/5D/6D adaptive grid search over clean_firm, solar, wind, and hydro dimensions (CAISO adds geothermal as 5th dimension). The search proceeds through three resolutions:

- **Coarse grid**: 5% steps across all dimensions
- **Zone search**: 1% steps around promising regions identified in coarse
- **Fine grid**: Sub-percent refinement in the 40–75% threshold range

**Assessment (a)**: The coarse-to-fine funnel is computationally efficient. At 5% initial resolution, the maximum gap error for any resource dimension is 2.5 percentage points of demand. For PJM (843 TWh), this translates to ~21 TWh resource allocation uncertainty, or ~$1–3/MWh cost impact depending on the LCOE differential between adjacent resources. This is within acceptable bounds for screening.

**Note**: Thresholds 10–40% use coarse grid only (no zone search or storage refinement). This is adequate given the flat cost surfaces in that range — the marginal cost of additional clean energy is nearly constant when displacing the cheapest fossil units.

### 2.2 Storage Dispatch Model

Storage dispatch follows a greedy sequential hierarchy:

| Priority | Type | Duration | RTE | Window | Dispatch Logic |
|----------|------|----------|-----|--------|---------------|
| 1 | Battery 4hr | 4 hours | 85% | Daily | Charge from surplus, discharge to deficit |
| 2 | Battery 8hr | 8 hours | 85% | Daily | Same, extended range |
| 3 | LDES (iron-air) | 100 hours | 50% | 7-day rolling | Multi-day bridging |
| 4 | Green H₂ | 1000 hours | 35% | 30-day rolling | Seasonal storage, ≥95% thresholds only |

**Assessment (b)**: Greedy sequential dispatch underestimates storage value by 5–15% compared to LP-optimized co-dispatch (NREL storage co-optimization studies show 8–12% improvement from joint dispatch). This is a **conservative bias** — the model overstates the resource procurement needed to achieve a given clean energy threshold, which is the safer direction for a screening tool.

**Specific concern**: The dispatch order is fixed regardless of the relative economics of each storage type. In scenarios where LDES is significantly cheaper than battery (e.g., at Low storage costs), the greedy approach may over-utilize batteries and under-utilize LDES, distorting the cost picture.

### 2.3 Copper-Plate Assumption

The model treats each ISO as a single bus with no internal transmission constraints. All generation within an ISO can reach any load within that ISO without congestion costs.

**Assessment (b)**: This is the most significant structural simplification. Impact varies by ISO:

| ISO | Impact | Reason |
|-----|--------|--------|
| PJM | High (+5–8% bias) | 300,000 sq mi footprint, AEP-East/West interface constraints, MAAC/ATSI transfer limits |
| MISO | High (+5–8% bias) | North-south thermal limits, MISO South import constraints |
| ERCOT | Moderate (+3–5%) | West Texas wind export constraints to load centers |
| CAISO | Low–Moderate (+2–4%) | Compact geography, well-developed Path 15/26 |
| NYISO | Low (+1–3%) | Compact geography, though NYC load pocket creates locational price separation |
| NEISO | Low (+1–3%) | Well-interconnected, though Maine wind export constraints exist |
| SPP | Moderate (+3–5%) | Wide geography, known wind curtailment in Oklahoma panhandle |

**Direction**: Overestimates achievable clean energy penetration. Underestimates LMP spatial variance.

**Appropriateness**: Explicitly acceptable for a screening tool. Adding transmission network modeling would increase compute by 10–100× for marginal precision improvement at the portfolio level.

### 2.4 No Unit Commitment

The model does not enforce minimum up/down times, ramp rates, or start-up costs for thermal generators.

**Assessment (b)**: Overestimates dispatchability of the fossil fleet by 5–10% during fast ramp transitions. Most impactful for:
- **Coal** (minimum up time 8–24 hours, minimum down time 4–8 hours)
- **Older gas CT** (minimum down time 2–4 hours)
- **Gas CCGT** (4–8 hour ramp-up to full output)

Partially mitigated by the GAF deration factors (12–18% by ISO), which implicitly capture some forced outage and availability constraints, though they do not model temporal correlation of outages.

### 2.5 Demand Profiles

The model uses multi-year averaged EIA-930 hourly profiles. This smooths interannual weather variability but eliminates extreme weather events (polar vortices, heat domes).

**Assessment (b)**:

| Variable | Interannual Variability | Impact on Optimal Mix |
|----------|------------------------|----------------------|
| Wind generation | ±15% | ±2–3% of demand |
| Solar generation | ±8% | ±1–2% of demand |
| Hydro generation | ±25% | ±1–3% of demand (ISO-dependent) |
| Peak demand | ±10% | Shifts capacity adequacy economics |

A single extreme weather year can shift the optimal resource mix by 3–5% of demand. The multi-year average produces a "typical" year that may not exist in reality.

**Recommendation**: Report key outputs with weather uncertainty bands derived from individual historical years (P10/P50/P90).

---

## 3. Cost Optimization Review

### 3.1 LCOE Table Validation

All LCOE inputs were cross-referenced against NREL ATB 2024, Lazard v17/v18, and LBNL reports.

**Solar and Wind**: Regional LCOE values are within ±5% of ATB 2024 moderate projections after adjusting for transmission. The regional variation (e.g., ERCOT solar $54/MWh vs. NYISO $92/MWh) correctly reflects irradiance/wind resource differences.

**Nuclear New-Build**: Range of $68–$170/MWh spans ATB moderate ($92/MWh) through conservative ($140/MWh), with the High sensitivity exceeding ATB conservative. This appropriately captures the extreme uncertainty in nuclear new-build costs.

**CCS-CCGT**: Medium values ($69.5–$100.5/MWh with 45Q) are within ±10% of industry estimates. However, the model assumes 90% CO₂ capture rate for the 45Q credit calculation. Real-world CCS projects have achieved 60–80% sustained capture (Boundary Dam: ~65% average, Petra Nova: ~70% before shutdown). This inflates the 45Q credit value by $4–9/MWh compared to demonstrated performance.

**Assessment (b)**: CCS LCOE is systematically optimistic by $4–9/MWh due to the 90% capture assumption. This overstates CCS competitiveness relative to nuclear and renewables+storage pathways.

### 3.2 45Q Credit Calculation

The 45Q credit is computed as:

```
$85/ton × 0.323 tCO₂/MWh = $27.5/MWh
where 0.323 = 0.90 (capture rate) × 0.359 tCO₂/MWh (unabated emission rate)
```

**Assessment (a)**: Mathematically correct given the 90% capture assumption. The arithmetic is verified.

**Risks not modeled**:
- 45Q requires 12-year credit period with geological sequestration certification
- IRS compliance and recapture risk
- Class VI well permitting delays (average 2–4 years)

**Recommendation**: Include a "45Q realization probability" sensitivity (70%/85%/100% of full credit) to capture execution risk.

### 3.3 Transmission Adders

Transmission costs are modeled as flat $/MWh adders by resource type and ISO, ranging from $0 (None) to $22/MWh (NYISO wind, High).

**Source**: LBNL "Queued Up" 2025 interconnection cost data.

**Assessment (a)**: Flat adders are appropriate for a screening tool. They represent fleet-average interconnection costs.

**Limitation**: Real transmission costs are highly site-specific (line length, terrain, existing capacity, affected system upgrades). Projects in constrained areas may face 2–3× the average cost. For PJM wind at Medium ($10/MWh), individual project costs range from $3/MWh (near existing substations) to $30/MWh (requiring new 500kV lines).

**Impact**: Transmission is 10–25% of delivered cost for wind, 5–15% for solar, 3–8% for nuclear. A 50% error in the transmission adder translates to 5–12% error in delivered cost for wind — material but within screening tolerance.

### 3.4 Storage Revenue Credits

Storage economics include a 70% revenue stacking factor applied to the sum of capacity, ancillary service, and energy arbitrage revenues.

**Assessment (a)**: The 70% stacking factor falls within the Lazard LCOS 2024 range (60–80%). This is a reasonable central estimate that captures the real-world difficulty of simultaneously monetizing all value streams.

The capacity degradation alpha (now sigmoid, 0.35–0.40 per ISO) captures the empirical phenomenon of capacity value erosion at high clean penetration. This is a strong modeling choice backed by observed PJM RPM clearing price trends.

### 3.5 Sensitivity Architecture

The model evaluates 5,832 cost scenarios per ISO/threshold (17,496 for CAISO with geothermal toggle), constructed from:

| Toggle | Levels | Values |
|--------|--------|--------|
| Renewable Gen | 3 | Low/Medium/High LCOE |
| Firm Gen | 3 | Low/Medium/High nuclear + CCS LCOE |
| Storage | 3 | Low/Medium/High battery + LDES LCOE |
| Fossil Fuel | 3 | Low/Medium/High coal + gas + oil prices |
| Transmission | 2 | Medium/High (None and Low are separate modes) |
| CCS 45Q | 3 | Low/Medium/High (with On/Off) |
| Geothermal | 4 | Low/Medium/High + None (CAISO only) |

**Assessment (a)**: Excellent coverage of input uncertainty space. The paired toggle design (correlated cost movements within categories) is a defensible rigor-compute tradeoff that reflects real-world cost correlations (solar and wind costs are correlated via shared supply chain and labor markets).

**Gap**: No demand elasticity. Load is perfectly inelastic across all scenarios. In reality, high prices induce some demand response (industrial load curtailment, behavioral changes). This omission overstates scarcity pricing frequency by ~10–20% during extreme hours.

---

## 4. LMP Engine Review

### 4.1 Merit-Order Stack Construction

The fossil fleet is modeled with four unit types, each characterized by heat rate, VOM, and emission rates:

| Unit Type | Heat Rate (MMBtu/MWh) | VOM ($/MWh) | CO₂ (t/MWh) | Source |
|-----------|----------------------|-------------|-------------|--------|
| Coal Steam | 10.0 | 5.50 | 0.95 | EPA eGRID 2022, PJM SOM 2024 |
| Gas CCGT | 7.0 | 3.50 | 0.37 | EPA eGRID 2022, PJM SOM 2024 |
| Gas CT | 10.5 | 5.00 | 0.55 | EPA eGRID 2022 |
| Oil CT | 10.5 | 6.00 | 0.65 | EPA eGRID 2022 |

**Assessment (a)**: Heat rates and emission rates are consistent with published eGRID values. The fleet-average approach (one heat rate per type) is appropriate for ISO-level screening. Individual unit heat rates vary ±15% around these averages, which translates to ±$2–4/MWh marginal cost variation within each unit type.

### 4.2 Cost-Based Offer Adder (FIXED)

**Previous state**: A uniform 10% cost-based offer adder (PJM Manual 15) was applied to all ISOs, including energy-only markets (ERCOT, SPP). This created a systematic positive bias of +$2–4/MWh for ISOs where generators submit competitive offers rather than cost-based offers.

**Current state (post-fix)**: ISO-specific adders based on market structure:

| ISO | Adder | Market Type | Rationale |
|-----|-------|-------------|-----------|
| PJM | 10% | RPM capacity market | PJM Manual 15 cost-based offer rule |
| NYISO | 10% | ICAP capacity market | NYISO OATT cost-based rules |
| NEISO | 10% | FCM capacity market | ISO-NE Manual for Market Operations |
| CAISO | 10% | Resource Adequacy | Cost-based offer rules similar to PJM |
| MISO | 7% | PRA capacity market | Module C energy offer rules, lower effective markup |
| ERCOT | 0% | Energy-only | Competitive offers, no regulatory markup |
| SPP | 0% | Energy-only | Competitive offers, no regulatory markup |

**Implementation**: `lmp_engine.py`, `COST_BASED_ADDERS` dict (line 127), threaded through `compute_marginal_costs()` via new `iso` parameter.

### 4.3 Demand-Quantile Pricing Layer

The LMP engine uses a three-layer pricing model:

1. **Base layer**: Merit-order marginal cost from fossil stack
2. **Demand-quantile layer**: Modifies prices based on demand rank
   - High-demand hours (>80th percentile): congestion adder up to $60/MWh
   - Extreme demand (>97th percentile): scarcity tail up to $500/MWh
   - Low-demand hours (<15th percentile): negative pricing floor at −$25/MWh
3. **ISO-specific price model**: Custom `PriceModel` subclass per ISO with calibrated parameters

Each ISO has a tailored price model:

| ISO | Model Class | Key Feature |
|-----|------------|-------------|
| PJM | `PJMPriceModel` | Penalty factor scarcity ($2,000 cap), moderate negative prices |
| ERCOT | `ERCOTPriceModel` | ORDC with VOLL=$5,000/MWh, higher volatility |
| CAISO | `CAISOPriceModel` | Aggressive negative pricing (duck curve), $-65/MWh floor |
| NYISO | `NYISOPriceModel` | Tight geography, congestion-driven |
| NEISO | `NEISOPriceModel` | Winter gas pipeline constraint, seasonal premium |
| MISO | `MISOPriceModel` | Coal-heavy fleet, wind congestion |
| SPP | `SPPPriceModel` | Closer to energy-only behavior |

**Assessment (a)**: The demand-quantile approach is a creative solution for capturing real-world price distribution features that a single-bus merit-order model cannot. The calibration against published ISO SOM statistics is well-documented.

**Structural concern (b)**: The demand-quantile layer is applied AFTER the merit-order layer, meaning it modifies prices based on demand rank rather than economic fundamentals. A high-demand hour where the marginal unit is a cheap CCGT gets the same proportional adder as a high-demand hour where the marginal unit is an expensive CT. This can produce price distributions that don't match the economic logic of the underlying dispatch.

For snapshot mode (2024/2025 calibration), this is acceptable — the parameters are tuned to match observed statistics. For trajectory mode beyond 2035, the assumption that demand-quantile relationships remain stable as the generation mix evolves is increasingly unreliable.

### 4.4 Scarcity Pricing

**PJM**: Penalty factor regime with $2,000/MWh cap — consistent with PJM tariff.

**ERCOT**: ORDC with VOLL = $5,000/MWh and exponential LOLP curve. The simplified exponential model with λ = 0.004 understates scarcity pricing during summer peaks and overstates it during shoulder months compared to ERCOT's actual ORDC curve (which has seasonal variation and separate spinning/non-spinning reserve parameters).

**Assessment (a)**: Scarcity events are rare (<100 hours/year) and their pricing impact averages to <$1–3/MWh on annual LMP. The simplification is appropriate for a screening tool.

### 4.5 Negative Pricing

Negative price floors vary by ISO ($-25 to $-65/MWh). CAISO has the deepest negative pricing ($-65/MWh floor, estimated 600 negative hours at current solar penetration).

**Assessment (b)**: The model produces negative prices based on demand-quantile rank rather than actual curtailment economics. Real negative prices result from:
- Must-run obligations (nuclear, combined heat and power)
- PTC incentives (negative marginal cost of −$26/MWh for PTC-eligible wind)
- Transmission constraints forcing local oversupply

None of these mechanisms are directly modeled. The calibrated parameters approximate the aggregate effect, but the model cannot predict how negative pricing frequency changes with increasing VRE penetration — a significant limitation for trajectory mode.

### 4.6 Price Formation Under High Clean Penetration

At >90% clean penetration, the fossil fleet becomes so small that LMP is dominated by:
- Scarcity pricing during non-clean-generation hours
- Zero/negative pricing during clean surplus hours

The model's demand-quantile approach may produce unrealistically smooth price transitions rather than the cliff effects observed in markets with rapid renewable deployment (e.g., South Australia's experience above 60% VRE).

**Assessment (c)**: At extreme clean penetration (>90%), the model's LMP outputs should be treated as directional only, not relied upon for specific price projections.

---

## 5. Market Simulation Engine Review

### 5.1 Profit-Driven Deployment Logic

Clean resources deploy where revenue exceeds cost, stopping at the first unprofitable zone. The simulation walks through thresholds from the current clean share (baseline grid mix) upward:

1. At each threshold, compute blended revenue (energy + capacity + REC)
2. Compare to blended cost (LCOE + transmission + PPA premium)
3. If profitable, deploy; if not, stop

**Assessment (a)**: Conceptually sound. The zone-by-zone approach introduces a path-dependency bias — the model cannot discover a non-monotonic optimal path (e.g., skip from 60% to 80% if 65–75% is unprofitable but 80% is profitable due to a different resource mix). In practice, this bias is small (<3% of demand) because efficient frontier mixes are monotonically increasing in total procurement.

### 5.2 Fossil Retirement Cascade

Retirement follows merit-order displacement: coal → oil → gas.

| Fuel | CO₂ Rate (t/MWh) | Retirement Order | Full Retirement Threshold |
|------|------------------|-----------------|--------------------------|
| Coal | 0.95 | First | ≥70% clean share |
| Oil | 0.65 | Second | ≥70% clean share |
| Gas CT | 0.55 | Third | Last to retire |
| Gas CCGT | 0.37 | Fourth | Most efficient, persists longest |

**Assessment (a)**: Correct in principle. Coal retires first as the highest-emitting, least-efficient fuel type.

**Limitation (c)**: Retirement is purely economic. The model does not account for:
- State-level coal plant preservation mandates (e.g., West Virginia in PJM)
- Reliability-must-run (RMR) contracts that keep uneconomic plants operating
- EPA 111(d) compliance deadlines that may force earlier retirement
- Political/regulatory constraints on gas plant construction

**Impact**: In states with regulatory constraints, actual retirement may lag economic retirement by 5–10 years. This is a known limitation appropriate for a market-driven screening tool.

### 5.3 Nuclear Revenue and 45U PTC

The model implements a contract-for-difference (CfD) floor mechanism for existing nuclear:

```
Floor Price: $40/MWh (escalation-adjustable)
Max Credit: $15/MWh
Sunset: 2032

If (energy_rev + capacity_rev) < floor_price:
    ptc = min(max_credit, floor_price - (energy_rev + capacity_rev))
```

**Assessment (a)**: Correctly captures the IRA §45U structure. The CfD mechanism ensures nuclear plants receive at minimum $40/MWh in combined revenue, with the PTC filling the gap up to $15/MWh.

**Minor note**: The default escalation parameter (0%) could be set to 2–3% to reflect inflation adjustment provisions in the actual statute. This would modestly increase nuclear viability in trajectory mode.

### 5.4 Capacity Market Degradation (FIXED)

**Previous state**: Linear degradation (`cap_price = base × max(0, 1 - α × clean_share)`) overstated price erosion at low clean shares and understated it in the transition zone.

**Current state (post-fix)**: Sigmoid S-curve degradation:

```
cap_price = base × max(floor, 1 - max_degrade / (1 + exp(-k × (clean_share - midpoint))))
```

Parameters by ISO:

| ISO | max_degrade | midpoint | k | floor | Base ($/kW-yr) |
|-----|------------|----------|---|-------|---------------|
| PJM | 0.80 | 0.50 | 8 | 0.15 | 120 |
| NYISO | 0.85 | 0.45 | 10 | 0.10 | 85 |
| CAISO | 0.85 | 0.55 | 10 | 0.10 | 75 |
| NEISO | 0.80 | 0.50 | 8 | 0.15 | 55 |
| MISO | 0.0 | — | — | — | 25 |
| ERCOT | 0.0 | — | — | — | 0 |
| SPP | 0.0 | — | — | — | 0 |

**Assessment (a)**: The S-curve better matches observed PJM RPM clearing price behavior, where prices are sticky below 40% clean share, decline steeply through 50–70%, and flatten near a floor at high penetration.

**Residual uncertainty**: ±30% around the model's capacity revenue estimate, driven by structural uncertainty in the degradation functional form and the inherent unpredictability of administrative capacity market parameters (CONE, Net CONE, MOPR).

### 5.5 Emissions Accounting (FIXED)

**Previous state**: Two bugs produced zero emissions across all ISOs:

1. **Data format mismatch**: The emission rates file contained `{"ERCOT": {"co2_rate": 0.42}}` but `compute_fossil_retirement()` looked for `coal_co2_lb_per_mwh`, `gas_co2_lb_per_mwh`, `oil_co2_lb_per_mwh`. All `.get()` calls defaulted to 0.0.

2. **Rate type confusion**: The caller used `displaced_rate` (emission rate of retired units) as the emission rate for the remaining fossil fleet. Should use `remaining_rate` for actual emission calculations.

**Current state (post-fix)**:
- Emission rates file replaced with full eGRID per-fuel-type data (coal: ~2,200 lb CO₂/MWh, gas CCGT: ~867 lb/MWh, oil CT: ~2,894 lb/MWh)
- Caller updated to use `retirement_info['remaining_rate_tco2_mwh']`

**Verification**: ERCOT at 46.1% clean now shows 148.2 Mt CO₂ (was 0.0). PJM at 40.6% shows 284.3 Mt (was 0.0).

### 5.6 Wright's Law Learning Curves (Trajectory Mode)

Learning rates by technology:

| Technology | Fast LR | Slow LR | 2025 Baseline (GW) |
|-----------|---------|---------|---------------------|
| Nuclear SMR | 15% | 10% | 2.0 |
| CCS-CCGT | 12% | 10% | 0.3 |
| Battery 4hr/8hr | 20% | 18% | 50.0 |
| LDES (iron-air) | 20% | 15% | 0.01 |
| Offshore Wind | 12% | 8% | 5.0 |
| Green H₂ | 18% | 12% | 0.1 |
| Solar | 0% | 0% | 150.0 (mature) |
| Wind (onshore) | 0% | 0% | 150.0 (mature) |

**Assessment (b)**:
- Solar and wind at 0% learning is conservative but defensible — cost reductions over 2025–2050 will be incremental relative to historical rates.
- Nuclear SMR at 10–15% learning is highly uncertain. No SMR has completed a full commercial learning curve. The 2.0 GW 2025 baseline is optimistic (NuScale's UAMPS project was cancelled; only small demonstrations exist).
- Battery learning rate (18–20%) is consistent with BNEF and historical Li-ion cost trajectories.

**Confidence intervals for 2050 costs**: Nuclear ±50%, CCS ±40%, LDES ±60%, Battery ±20%.

### 5.7 Interconnection Queue Caps

| ISO | Facilitating (GW/yr) | Challenging (GW/yr) |
|-----|---------------------|---------------------|
| CAISO | 6 | 3 |
| ERCOT | 8 | 4 |
| PJM | 7 | 3 |
| NYISO | 5 | 2 |
| NEISO | 5 | 2 |
| MISO | 7 | 3 |
| SPP | 6 | 3 |

**Assessment (a)**: Consistent with LBNL "Queued Up 2024" queue completion rates. Historical completion rates are lower: only 20–25% of queued capacity reaches commercial operation, with 5–7 year average development times. The Facilitating/Challenging framing appropriately brackets the range.

---

## 6. Data Quality Assessment

### 6.1 EIA-930 Hourly Profiles

Multi-year averaged (5-year) demand and generation profiles from EIA-930.

**Assessment (a)**: Appropriate for screening. DST-aware solar zeroing and nuclear monthly derate are correct data transformations.

**Note**: EIA-930 data quality varies by ISO. MISO and SPP have known reporting inconsistencies in 2021–2022 data. Cross-validation with ISO-published generation reports is recommended.

### 6.2 eGRID Emission Factors

2022 vintage. Per-fuel-type CO₂ rates (coal: 0.95 tCO₂/MWh, gas CCGT: 0.37 tCO₂/MWh) are consistent with published eGRID subregion values.

**Assessment (a)**: The `fleet_model.py` validation against EPA CAMPD hourly data is a strong quality control mechanism. Currently available only for Texas (EIA 860/923 data loaded for TX). Extending to additional states would strengthen confidence in fleet characterization.

### 6.3 Fossil Capacity Inventory

**INSTALLED_FOSSIL_MW** and **FOSSIL_CAPACITY_SHARES** by ISO are well-sourced from PJM SOM 2024 and EIA 860M.

| ISO | Total Fossil (MW) | Coal Share | Gas CCGT | Gas CT | Oil CT |
|-----|-------------------|-----------|----------|--------|--------|
| CAISO | 47,000 | 0% | 55% | 40% | 5% |
| ERCOT | 80,000 | 22% | 50% | 28% | 0% |
| PJM | 127,800 | 29% | 37% | 31% | 3% |
| NYISO | 28,000 | 0% | 45% | 50% | 5% |
| NEISO | 16,000 | 0% | 52% | 42% | 6% |
| MISO | 105,000 | 34% | 38% | 23% | 1% |
| SPP | 58,000 | 35% | 42% | 27% | 1% |

**Gap (b)**: No systematic accounting of planned retirements beyond PJM (Brandon Shores, Wagner, Indian River adjustments are documented). MISO has significant coal retirement announcements (Ameren Rush Island, Xcel Sherco) not reflected in the static inventory. This may overstate MISO's coal generation by 5–10 TWh.

### 6.4 Coal and Oil Capacity Caps

Fossil generation caps by fuel type (TWh/yr, 2025 baseline):

| ISO | Coal Cap (TWh) | Oil Cap (TWh) |
|-----|---------------|--------------|
| CAISO | 0.00 | 0.60 |
| ERCOT | 67.58 | 0.00 |
| PJM | 139.09 | 4.59 |
| NYISO | 0.00 | 0.15 |
| NEISO | 0.31 | 1.29 |
| MISO | 125.0 | 0.50 |
| SPP | 42.0 | 0.20 |

**Source**: EIA-923 2023 actuals, adjusted for announced retirements through 2025.

**Assessment (a)**: Reasonable baseline, though see MISO gap noted above.

---

## 7. Edge Cases and Failure Modes

### 7.1 ERCOT at High Solar Penetration

ERCOT has no capacity market. At high solar penetration (>40% energy share), the model may understate negative pricing frequency because the demand-quantile approach — calibrated to 2024 data (~14% solar) — does not scale the negative pricing mechanism with increasing solar share.

**Potential anomaly**: Revenue collapse for existing nuclear in ERCOT. With no capacity payments and suppressed LMP from solar, nuclear revenue could fall below variable costs, triggering retirement in the model. In reality, Comanche Peak has long-term offtake contracts (not modeled) that would prevent economic retirement.

### 7.2 PJM Winter Gas Constraints

Winter Storm Elliott (2022) exposed PJM's gas fleet vulnerability: 46 GW of forced outages from correlated gas supply failures. The GAF of 0.82 (18% deration) partially captures availability reduction but does not model temporal correlation — the model assumes outages are uniformly distributed rather than concentrated during extreme cold events.

**Impact on trajectory mode**: At high clean penetration with gas as the marginal backup, correlated gas outages during polar vortex events could produce reliability shortfalls not visible in the model's hourly dispatch.

### 7.3 NEISO Winter Gas Pipeline

The model applies a +$13.13/MWh CCS cost adder for NEISO to capture the Algonquin pipeline constraint premium. However, the same constraint affects gas CCGT dispatch costs during winter peaks — gas delivered prices in NEISO regularly spike to $15–25/MMBtu during January cold snaps. This asymmetry may understate NEISO winter LMP by $10–20/MWh during peak periods.

### 7.4 CAISO Duck Curve Extremes

At >35% solar penetration (current CAISO level), the 3-hour evening ramp from solar peak to sunset (~15 GW in 3 hours) creates extreme price volatility. The demand-quantile approach captures the statistical distribution but not the temporal concentration of price movement. Trajectory mode with higher solar penetration will amplify this effect.

### 7.5 Sweep Mode at Extreme Parameter Combinations

The 270-scenario sweep covers 2 conditions × 3 demand × 5 price × 3 PPA × 3 gas friction. At extreme corners (e.g., Challenging + High demand + All-high prices + Low PPA + High gas friction), the model may produce zero clean deployment — this is a valid result, not a failure mode, but should be flagged to users as a "no investment" scenario.

### 7.6 >90% Clean Penetration

At very high clean shares, the fossil fleet is so small that:
- LMP is bimodal: near-zero during clean surplus, extreme during deficits
- Capacity market clearing prices become administratively determined
- The linear interpolation between thresholds may miss sharp non-linearities

The model's sigmoid capacity degradation and demand-quantile pricing become increasingly approximate above 90% clean share. Results in this range should be treated as directional only.

---

## 8. Statistical Framework for Uncertainty

### 8.1 Uncertainty Taxonomy

| Category | Description | Examples | Quantification |
|----------|-------------|----------|----------------|
| **Parametric** | Input value uncertainty | LCOE, fuel prices, capacity costs | L/M/H sensitivity ranges |
| **Structural** | Model architecture limitations | Copper-plate, no UC, greedy storage | Comparison to reference models |
| **Scenario** | Future-state uncertainty | Demand growth, policy changes, tech breakthroughs | Sweep mode coverage |

### 8.2 Confidence Intervals by Output

| Output | Snapshot | Trajectory 2035 | Trajectory 2050 |
|--------|---------|-----------------|-----------------|
| Clean energy % | ±5 pp | ±10 pp | ±20 pp |
| Average LMP | ±15% | ±25% | ±40% |
| Annual CO₂ | ±20% | ±30% | ±50% |
| Nuclear retirement decision | Directionally reliable | Scenario-dependent | Low confidence |
| CCS breakeven carbon price | ±$10/ton | ±$20/ton | ±$30/ton |
| Resource mix composition | Directionally reliable | Order-of-magnitude | Qualitative only |

### 8.3 Directional Confidence Assessment

Despite wide individual uncertainty ranges, the tool is highly reliable for directional conclusions:

| Question | Confidence | Evidence |
|----------|-----------|---------|
| Does higher gas price increase clean deployment? | >95% | Monotonic across all 270 scenarios |
| Does carbon pricing accelerate coal retirement? | >99% | Coal economics are unambiguously worse with any carbon price |
| Is nuclear retirement risk higher in energy-only markets? | >90% | ERCOT/SPP nuclear revenue consistently lower than PJM/NYISO |
| Does CCS require carbon pricing to be competitive? | >95% | Without 45Q at high gas prices, CCS LCOE exceeds gas CCGT |
| Does storage reduce the cost of high clean targets? | >99% | Storage smooths VRE intermittency at lower cost than firm clean |

### 8.4 Systematic Bias Summary

| Assumption | Bias Direction | Magnitude | Impact on Key Outputs |
|-----------|---------------|-----------|----------------------|
| Copper-plate transmission | Optimistic | +3–8% clean share | Overstates achievable penetration |
| No unit commitment | Optimistic | +5–10% fossil availability | Understates retirement pressure |
| Greedy storage dispatch | Conservative | −5–15% storage utilization | Overstates clean firm needs |
| 90% CCS capture rate | Optimistic | −$4–9/MWh CCS cost | Overstates CCS competitiveness |
| Sigmoid capacity degradation | Neutral | ±30% capacity revenue | Balanced at mid-range |
| Static demand | Conservative | −20–40% price variance | Understates scarcity frequency |
| Multi-year weather averaging | Mixed | ±3–5% optimal mix | Smooths weather extremes |

**Net direction**: The model is slightly optimistic about clean energy deployment potential (copper-plate and no-UC effects dominate) and slightly conservative about storage economics (greedy dispatch). These partially offset each other.

---

## 9. Comparative Positioning

| Feature | This Tool | GenX | IPM | ReEDS |
|---------|-----------|------|-----|-------|
| Geographic resolution | ISO-level (7) | Nodal | Nodal | ReEDS zones (~130) |
| Temporal resolution | 8,760 hourly | Representative periods | Seasonal blocks | 17 time slices |
| Transmission | Copper-plate | Full network | Full network | Pipeline |
| Unit commitment | None | Optional | Full | None |
| Storage optimization | Greedy sequential | Co-optimized LP | Co-optimized LP | Simplified |
| Scenario coverage | 270 in 30 min | 1 in 2–12 hours | 1 in 4–24 hours | ~10 in days |
| Runtime (single scenario) | 5–15 seconds | 2–12 hours | 4–24 hours | 1–4 hours |
| Cost | $0 (local compute) | Licensed software | Licensed software | NREL tool |

**Competitive advantage**: The tool explores more of the uncertainty space in 30 minutes than GenX/IPM can in a week. This makes it well-suited for **screening and directional analysis** — identifying which parameter combinations warrant deeper investigation with production-grade models.

**When to escalate**: Use GenX/IPM/ReEDS when:
- Facility-siting decisions require nodal analysis
- Transmission expansion is a decision variable
- Unit commitment constraints materially affect outcomes
- Results must withstand regulatory/compliance scrutiny

---

## 10. Recommendations

### 10.1 Critical Fixes (Completed in This Review)

| # | Issue | Status | Impact |
|---|-------|--------|--------|
| 1 | Emission rates data format mismatch (all emissions = 0) | **Fixed** | All ISOs now compute non-zero emissions |
| 2 | Uniform 10% cost-based offer adder | **Fixed** | ERCOT/SPP LMP reduced by $2–4/MWh |
| 3 | Linear capacity market degradation | **Fixed** | S-curve matches real auction behavior |

### 10.2 High-Priority Improvements

| # | Recommendation | Rationale | Status |
|---|---------------|-----------|--------|
| 4 | Add explicit uncertainty bands to all outputs (P10/P50/P90 from sweep mode) | Users need to understand result confidence, not just point estimates | **Fixed** — `aggregate_sweep_percentiles()` computes P10/P50/P90/mean/std across all 270 sweep scenarios for 12 scalar metrics, boolean metrics (% of scenarios), per-resource mix TWh, nuclear revenue, and zone deployment count. Results injected as `_aggregates` key in output JSON. |
| 5 | Implement weather-year sensitivity (use individual historical years) | Multi-year average smooths extremes that drive capacity adequacy decisions | **Fixed** — `get_demand_profile()` and `get_supply_profiles()` accept `weather_year` parameter. 5 years available (2021-2025) from EIA-930 parquets. `run_full_sweep()` accepts `weather_years` list for multi-year sweep. CLI: `--weather-year all` runs 5× sweep (1,350 scenarios). |
| 6 | Add "45Q realization probability" parameter (70%/85%/100%) | CCS economics are highly sensitive to 45Q execution risk | **Fixed** — `CCS_45Q_REALIZATION_PROB` parameter added to `pipeline_config.py` with 3 levels (0.70/0.85/1.00). `compute_clean_firm_tranches()` accepts `q45_realization` parameter that applies a credit haircut to CCS LCOE. |
| 7 | Scale negative pricing with VRE penetration in trajectory mode | Current calibration is fixed to 2024 VRE levels | **Fixed** — `compute_hourly_lmp_vectorized()` accepts `vre_penetration` parameter. At >25% VRE, negative price floor deepens (1.5× scaling per 10pp above baseline) and negative-price band widens (1.0× per 10pp). Capped at 2× to prevent extremes. Calibrated against CAISO DMM 2019-2024 negative price hour trends. |

### 10.3 Medium-Priority Improvements

| # | Recommendation | Rationale | Status |
|---|---------------|-----------|--------|
| 8 | Add demand elasticity for extreme price events (>$200/MWh) | Overstates scarcity pricing by 10–20% | Open |
| 9 | Extend fleet_model.py real generator data beyond Texas | Strengthens heat rate and emission rate validation | Open — blocked on EIA 860/923 multi-state data acquisition |
| 10 | Add MISO planned coal retirement adjustments | Current coal cap overstated by 5–10 TWh | **Fixed** — `COAL_CAP_TWH['MISO']` reduced from 125.0 to 112.0 TWh. Accounts for Rush Island (retired 2024), Sherco Unit 2 (retired 2023), Campbell 1-3 (retiring 2025), Belle River (retiring 2028-29). Sourced from EIA-860M Dec 2024. |
| 11 | Model nuclear offtake contracts (ERCOT) | Prevents false retirement signal for contracted plants | **Fixed** — `NUCLEAR_OFFTAKE_CONTRACTS` added to `pipeline_config.py` with ERCOT Comanche Peak data (2.3 GW, $35/MWh floor, contract through 2045). Nuclear retirement check in `market_simulation.py` skips retirement for contracted plants within contract period. |

### 10.4 Documentation Improvements

| # | Recommendation | Rationale | Status |
|---|---------------|-----------|--------|
| 12 | Add source citations for interconnection queue caps | Currently undocumented in code comments | **Fixed** — Detailed per-ISO citations added to `QUEUE_CAP_GW` in `market_simulation.py`. Sources: LBNL "Queued Up 2024" (Rand et al.), queue sizes and completion rates per ISO, cross-validated against Princeton REPEAT and Rhodium Clean Investment Monitor. |
| 13 | Publish demand-quantile pricing calibration methodology as technical note | Key innovation deserves standalone documentation | **Fixed** — `docs/Demand_Quantile_Pricing_Methodology.md`: comprehensive technical note covering architecture (4 pricing layers), all parameter definitions with formulas, ISO-specific parameter table (v11.3), calibration process (v11.0→v11.3 iteration), VRE scaling extension, and known limitations. |
| 14 | Document synthetic LMP validation against actual ISO clearing prices (all 7 ISOs) | Currently available for PJM only | **Fixed** — `docs/LMP_Validation_Results.md`: all 7 ISOs documented with calibration targets, sources, ISO-specific notes, accuracy metrics (avg ±8%, P50 ±8%, P90 ±15%), confidence ratings (High for PJM/ERCOT, Medium-High for CAISO/MISO/SPP, Medium for NYISO/NEISO), and known biases. |

---

## Appendices

### Appendix A: LCOE Cross-Reference Table

**Solar LCOE ($/MWh) — Model vs. NREL ATB 2024 Moderate**

| ISO | Model (Med) | ATB 2024 (adj.) | Delta | Notes |
|-----|------------|----------------|-------|-------|
| CAISO | 60 | 58 | +3% | Higher CA labor/permitting costs |
| ERCOT | 54 | 52 | +4% | TX irradiance advantage |
| PJM | 65 | 63 | +3% | Mid-Atlantic resource |
| NYISO | 92 | 88 | +5% | NY labor costs, lower irradiance |
| NEISO | 82 | 79 | +4% | NE resource quality |
| MISO | 62 | 60 | +3% | Midwest average |
| SPP | 57 | 55 | +4% | Southern plains |

**Wind LCOE ($/MWh) — Model vs. NREL ATB 2024 Moderate**

| ISO | Model (Med) | ATB 2024 (adj.) | Delta |
|-----|------------|----------------|-------|
| CAISO | 73 | 70 | +4% |
| ERCOT | 40 | 38 | +5% |
| PJM | 62 | 59 | +5% |
| NYISO | 81 | 77 | +5% |
| NEISO | 73 | 69 | +6% |
| MISO | 43 | 41 | +5% |
| SPP | 37 | 35 | +6% |

Model values are consistently 3–6% above ATB moderate, which reflects a modest conservatism in LCOE inputs. This is appropriate for a screening tool.

### Appendix B: Gas Availability Factors (GAF)

| ISO | GAF | Effective Deration | Basis |
|-----|-----|-------------------|-------|
| CAISO | 0.88 | 12% | Drought risk, gas supply constraints |
| ERCOT | 0.83 | 17% | Winter Storm Uri/Elliott experience |
| PJM | 0.82 | 18% | Largest deration — gas pipeline + weather exposure |
| NYISO | 0.82 | 18% | NYC dual-fuel constraints |
| NEISO | 0.85 | 15% | Algonquin pipeline bottleneck |
| MISO | 0.83 | 17% | Continental climate exposure |
| SPP | 0.84 | 16% | Moderate weather risk |

GAF combines forced outage rates, planned maintenance, and weather-related availability into a single annual availability factor. It does not capture temporal correlation of outages during extreme events.

### Appendix C: Complete Sensitivity Dimensions

**Sweep Mode (270 scenarios)**:

| Dimension | Levels | Values |
|-----------|--------|--------|
| Grid Condition | 2 | Facilitating, Challenging |
| Demand Growth | 3 | Low (0.5%/yr), Medium (1.5%/yr), High (2.5%/yr) |
| Fuel Prices | 5 | Very Low, Low, Medium, High, Very High |
| PPA Availability | 3 | Low, Medium, High |
| Gas Friction | 3 | Low, Medium, High |

Total: 2 × 3 × 5 × 3 × 3 = 270 scenarios

**Cost Sensitivity (per threshold per ISO)**:

| Toggle | Levels | Count |
|--------|--------|-------|
| Renewable Gen | 3 | L/M/H |
| Firm Gen | 3 | L/M/H |
| Storage | 3 | L/M/H |
| Fossil Fuel | 3 | L/M/H |
| Transmission | 2 | M/H |
| CCS 45Q | 3 | L/M/H with On/Off |
| Geothermal (CAISO only) | 4 | N/L/M/H |

Total: 3⁴ × 2 × 3 = 5,832 (non-CAISO), 5,832 × 3 = 17,496 (CAISO)

### Appendix D: Glossary

| Term | Definition |
|------|-----------|
| **CFE** | Carbon-Free Energy — hourly matching of clean generation to demand |
| **CfD** | Contract for Difference — price floor mechanism (used for nuclear 45U) |
| **ELCC** | Effective Load Carrying Capability — capacity credit for reliability |
| **FOAK** | First-of-a-Kind — initial commercial-scale project costs |
| **GAF** | Gas Availability Factor — annual deration for gas fleet reliability |
| **LMP** | Locational Marginal Price — wholesale electricity price at a node |
| **LDES** | Long-Duration Energy Storage — 100+ hour duration technologies |
| **NOAK** | Nth-of-a-Kind — mature technology costs after learning curve |
| **ORDC** | Operating Reserve Demand Curve — ERCOT's scarcity pricing mechanism |
| **PFS** | Physics Feasible Space — set of resource mixes that achieve a given threshold |
| **PPA** | Power Purchase Agreement — long-term offtake contract |
| **RPM** | Reliability Pricing Model — PJM's capacity market auction |
| **RTE** | Round-Trip Efficiency — energy out / energy in for storage |
| **VRE** | Variable Renewable Energy — solar and wind generation |

---

*End of Independent Technical Review*
