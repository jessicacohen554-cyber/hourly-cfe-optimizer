# Independent Expert Review: Market Simulation Screening Tool

## A Critical Assessment of Architecture, Assumptions, and Fitness for Purpose

**Reviewer Posture**: Third-party expert in energy systems modeling, power price forecasting, and clean energy policy design. This review assesses the tool's validity as a screening instrument for directionally correct market outcomes under user-defined inputs. The review is skeptical but constructive — I want this tool to succeed, but only if the claims match the capabilities.

**Model Reviewed**: Market Simulator v1.1.0 (March 2026)
**Codebase**: `market-simulator/` directory — FastAPI backend, 12 Python simulation modules, 6 HTML frontend pages
**Scope**: 7 U.S. ISOs (CAISO, ERCOT, PJM, NYISO, NEISO, MISO, SPP), profit-driven deployment model

---

## 1. The Right Question, Asked the Right Way

Before cataloging what's wrong, I want to acknowledge what this tool gets right at a conceptual level — because the framing matters.

Most capacity expansion models (GenX, ReEDS, IPM) ask: *"What's the least-cost pathway to meet a clean energy target?"* This tool asks a fundamentally different question: *"Given these market conditions, what do generators actually do?"* Clean energy deployment is an **output** — an emergent property of profitability — not a constraint. That's a commercially relevant framing that most production models don't serve well.

For an IPP commercial strategy team trying to answer "will my gas fleet survive a $50/ton carbon price?" or "at what gas price does CCS pencil out?", a constraint-minimization model is the wrong tool. You need something that models market dynamics, not optimization targets. This tool occupies that niche.

The 270-scenario parametric sweep completing in under 30 minutes is genuinely valuable. Running 270 scenarios in GenX or PLEXOS would take days to weeks of setup and compute. That throughput advantage — the ability to explore parameter space rapidly — is the tool's primary competitive moat. Don't underestimate it.

---

## 2. What the Implementation Gets Right

### 2.1 Plant-Level Merit-Order Dispatch

The two-tier architecture is pragmatic and well-executed:

- **Tier 1** (fleet-average): Four aggregated unit types with representative heat rates (coal 10.0, CCGT 7.0, CT 10.5, oil 10.5 MMBtu/MWh). Fast, suitable for parametric sweeps.
- **Tier 2** (plant-level): Per-generator heat rates from EIA Form 860, with `build_plant_level_merit_order()` sorting hundreds of individual plants by their actual marginal costs. This is meaningfully better than fleet-average screening — real CCGT heat rates range from 6.3 to 8.2 MMBtu/MWh, and that spread creates a $5–12/MWh marginal cost differential that determines which specific plants survive and which don't.

The automatic fallback between tiers based on data availability is a good design choice. Most screening tools force users into one mode. Having both modes with identical output formats is practical engineering.

### 2.2 Unit Commitment Constraints

The implementation includes vintage-adjusted unit commitment parameters (`market_simulation.py:246-301`):

| Unit Type | Min Up (hrs) | Min Down (hrs) | Start Cost ($/MW) | Min Gen (%) |
|-----------|-------------|----------------|-------------------|-------------|
| Coal Steam | 24 | 12 | $150 | 40% |
| Gas CCGT (pre-2005) | 4 | 2 | $35 | 50% |
| Gas CCGT (2005-2014) | 4 | 2 | $30 | 38% |
| Gas CCGT (2015+) | 3 | 1 | $25 | 30% |
| Gas CT | 1 | 1 | $15 | 20% |

The vintage differentiation for CCGTs is a detail most screening tools skip — newer H-class single-shaft turbines (GE 7HA, Siemens 9000HL) genuinely do turn down to 30%, while older 2x1 multi-shaft configurations need 50%. Getting this right matters for coal retirement timing: when coal operates at 40% minimum generation for 24 hours, it generates expensive power that wouldn't exist in a perfect-dispatch model.

**Important caveat**: This UC is applied as a post-processing step to merit-order dispatch results, not co-optimized with dispatch decisions (see Section 3.6). This distinction matters and should be documented more prominently.

### 2.3 Wright's Law Learning Curves

Endogenous cost reduction via deployment-based learning (`wright_cost()`) with technology-specific learning rates and background (rest-of-world) deployment curves is a strong inclusion. Most screening tools treat future costs as exogenous scenarios. Having costs respond to cumulative deployment creates realistic feedback loops: as batteries deploy, they get cheaper, which drives more deployment. The FOAK-to-NOAK trajectory for nuclear (10–15% learning rate) and LDES (15–20%) reflects the genuine uncertainty in these technologies.

### 2.4 Calibrated LMP Engine

The per-ISO price models calibrated against State of the Market reports (PJM SOM 2024, ERCOT Modo Energy, CAISO DMM, etc.) with documented validation targets and error bounds is above-average for a screening tool. Average LMP error of ±$2–4/MWh (±8%) against published data is adequate for the tool's stated purpose.

### 2.5 Nuclear Revenue Modeling

The 45U PTC contract-for-difference implementation — $40/MWh floor, $15/MWh max credit, 2032 sunset — correctly captures the IRA structure. For a tool targeting IPP fleet planning, getting nuclear retirement economics right is essential, and this implementation does.

---

## 3. Structural Critiques

### 3.1 Transmission: The Copper-Plate Problem

**Severity: High — This is the primary structural limitation.**

Each ISO is modeled as a single bus. All generation can reach all load without congestion. This is the most consequential simplification in the model, and its impact varies dramatically by ISO and by the question being asked.

**Where it matters most:**

| ISO | Key Constraint | System-Avg vs. Zonal Spread | Impact on Screening |
|-----|---------------|----------------------------|---------------------|
| PJM | AEP-East/West interface, MAAC/ATSI transfer limits | $10–20/MWh | High — western PJM plants face structurally different prices than eastern |
| MISO | North-South thermal limits (3–5 GW transfer cap) | $8–15/MWh | High — MISO South import constraints affect coal retirement timing |
| ERCOT | West Texas wind export to Dallas/Houston load centers | $5–25/MWh (extreme in wind events) | Moderate-High — wind curtailment economics are entirely locational |
| NYISO | NYC load pocket (Zone J) vs. upstate (Zones A-F) | $15–35/MWh | Moderate — compact geography but extreme zonal separation |
| CAISO | Path 15/26, but relatively well-interconnected | $3–8/MWh | Low-Moderate |
| NEISO | Maine wind exports, CT/RI interface | $3–8/MWh | Low-Moderate |
| SPP | Oklahoma panhandle wind congestion | $5–15/MWh | Moderate |

The internal peer review estimates +3–8% bias on achievable clean penetration. I'd argue this understates the problem for asset-specific questions. When a plant in western PJM faces an LMP of $22/MWh while the system average is $35/MWh, the copper-plate model shows the plant as profitable when it's actually stranded. That's not a 3–8% bias — it's a qualitatively wrong answer for that specific asset.

**Can zonal resolution be added without becoming PLEXOS?**

Yes. There's a well-established middle ground between copper-plate and full nodal modeling:

1. **Pipe-and-bubble zonal decomposition** (GenX approach): Define 3–5 zones per ISO with aggregate transfer limits between them. PJM publishes transmission zone data; MISO publishes North/Central/South transfer capabilities. This adds one `scipy.optimize.linprog` call per hour to enforce flow constraints. Compute cost: ~2–5x current runtime. Accuracy gain: captures 60–80% of congestion effects.

2. **Post-hoc zonal price adjustment**: Apply historical basis differentials from FTR/CRR auction clearing data to system-average LMP. CAISO CRR data and PJM FTR data are publicly available. This is cheap (lookup table) and captures static congestion patterns, though it misses how congestion changes with fleet evolution.

3. **Congestion adders from SOM data**: PJM's State of the Market report publishes $3.01/MWh average system-wide congestion for 2024. This could be decomposed by zone and applied as a location-specific price adjustment.

**Is zonal necessary for directional screening?**

It depends entirely on the question:

- *"Will coal retire in MISO under a $50/ton carbon price?"* — System-average is fine. The answer is directionally robust to congestion effects.
- *"Will Plant X in Zone 5 of MISO survive?"* — Zonal prices are essential. Zone 5 cleared at $719/MW-day in capacity while other zones cleared at $30/MW-day. System average is meaningless here.
- *"What carbon price makes CCS profitable in PJM?"* — System-average is adequate for CCS breakeven (it's a region-wide cost question), but the answer for a specific retrofit site needs local LMP.

**Recommendation**: For the tool's current use case (fleet-level screening), copper-plate is defensible but should be flagged more aggressively in the UI. For the plant-level Tier 2 mode, the absence of zonal pricing undermines the granularity the feature promises. At minimum, implement Option 2 (post-hoc zonal adjustment) for plant-level results.

### 3.2 Storage Dispatch: Greedy Sequential, Not Co-Optimized

Storage dispatches in a fixed priority order: Battery 4hr → Battery 8hr → LDES → Green H₂. Each storage type operates on the residuals left by the previous one. This is a greedy heuristic, not an economic optimization.

**What this misses:**

- **Relative economics**: If LDES is significantly cheaper than battery in a given scenario (Low storage costs), the greedy order still dispatches battery first. An LP co-dispatch would allocate storage capacity to maximize total value.
- **Price-responsive dispatch**: Real battery operators arbitrage LMP spreads — charge at $10/MWh, discharge at $60/MWh. The model's surplus/deficit dispatch captures some of this implicitly, but misses strategic bidding behavior and the feedback loop where storage flattens the LMP spread it's arbitraging.
- **Inter-temporal coupling**: The 24-hour battery window and 7-day LDES window are fixed. Real operators adapt their dispatch horizon to market conditions — a battery might hold charge through a 36-hour wind lull if the forecast warrants it.

NREL benchmarks suggest LP co-optimized storage dispatch captures 8–15% more value than greedy sequential approaches. For a screening tool, the conservative bias (understating storage value → overstating required procurement) is arguably the safer direction of error. But it means the model systematically undervalues storage-heavy pathways, which could cause users to dismiss viable high-storage portfolios.

**Fix**: Replace the sequential loop with a single LP co-dispatch using `scipy.optimize.linprog`. Objective: minimize total residual demand (or maximize arbitrage revenue). Constraints: SOC limits, power ratings, RTE losses, inter-temporal energy balance. This is a well-solved problem and adds minimal compute overhead for a single ISO-year.

### 3.3 The Demand-Quantile Pricing Layer

This is the model's most creative feature and also its most epistemologically fragile.

The LMP engine applies a three-layer pricing model:
1. Merit-order marginal cost (physics-based)
2. Demand-quantile adjustments (statistically calibrated)
3. ISO-specific scarcity/negative pricing (parameter-tuned)

Layer 1 is standard and defensible. Layers 2 and 3 are calibrated curve-fitting — they reproduce observed 2024 price distributions by mapping demand rank to price adders. This works because the parameters were tuned to match published SOM statistics.

**The problem is extrapolation.** When the model projects forward to 2040 with 75% clean energy, the demand-quantile relationships calibrated in 2024 (at 20–40% clean) may not hold. Specifically:

- Negative pricing frequency is driven by VRE curtailment economics and must-run obligations, not demand rank. At 60% solar penetration in CAISO, midday negative pricing occurs regardless of whether it's a "low-demand" hour by historical standards.
- Scarcity pricing at high clean penetration is dominated by multi-day weather events (wind droughts, cloudy weeks), not the seasonal demand peaks that drive scarcity in 2024.
- Capacity market clearing prices under high clean penetration follow different dynamics than the sigmoid degradation curves calibrated to 2020–2024 RPM data.

The model's own documentation acknowledges these limits (LMP Validation Results, Section 5), but the user-facing materials don't communicate this loudly enough. A user running a trajectory to 2050 might reasonably assume the LMP outputs are as reliable as the 2025 snapshot, when in reality confidence degrades significantly beyond 2035.

**Recommendation**: Add explicit confidence bands to trajectory outputs that widen with projection horizon. Display a visual indicator (color-coded timeline or confidence rating) on the results page showing where the model transitions from "calibrated" to "extrapolated."

### 3.4 Inter-Regional Trade

Each ISO is modeled in complete isolation. No imports, no exports, no cross-border flows.

This is a material omission:

| Interconnection | Typical Flow | Impact |
|----------------|-------------|--------|
| MISO → PJM | 5–15 GW hourly | Suppresses PJM LMP by $2–5/MWh; supports MISO exports |
| Pacific NW → CAISO | 5–8 GW (hydro-heavy) | Provides 15–20% of CAISO supply; affects evening ramp pricing |
| Hydro-Québec → NEISO | 2–4 GW | Critical for NEISO winter reliability; affects gas dispatch |
| PJM → NYISO | 2–5 GW | NYC load pocket served partly by PJM imports |
| MISO → SPP | 1–3 GW (bidirectional) | Wind energy exchange affects both ISOs' curtailment patterns |

Ignoring inter-regional trade means the model overstates each ISO's self-sufficiency requirements. NEISO's 16 GW fossil fleet looks oversized until you realize 4 GW of that demand is met by Quebec hydro imports. Similarly, CAISO's resource adequacy picture changes dramatically when you remove Pacific NW hydro imports.

**Minimum viable fix**: Add exogenous net import/export profiles per ISO, derived from EIA-930 interchange data. This doesn't require modeling the interconnection economics — just applying a fixed hourly import/export time series that reduces residual demand. It's a 50-line code change that materially improves resource adequacy calculations.

### 3.5 Demand Response

Load is modeled as nearly perfectly inelastic, with a logarithmic dampening factor that only activates above $200–300/MWh. Real markets have substantial demand-side flexibility:

- **PJM**: ~10 GW of registered demand response (~6% of peak)
- **ERCOT**: ~5 GW emergency and economic DR (~6% of peak)
- **MISO**: ~8 GW LMR (Load Modifying Resources)
- **NYISO**: ~1.5 GW EDRP + ICAP/SCR programs

Missing demand response has two effects:
1. **Overstates scarcity pricing**: When load can't shed, prices spike higher and longer than they would with DR participation.
2. **Understates high-clean viability**: DR reduces peak residual demand, which reduces the fossil capacity needed for reliability — a key enabler of high clean penetration.

For a screening tool, this creates a mild pessimistic bias on clean energy deployment. That's acceptable but should be documented. For the nuclear retirement use case specifically, missing DR overstates LMP during scarcity hours, which makes nuclear look slightly more viable than it is — a small but noteworthy optimistic bias for that particular output.

### 3.6 Unit Commitment: Implemented But Post-Hoc

Credit where due: the model implements min up/down times, start-up costs, minimum stable generation, and vintage-adjusted CCGT parameters. This is better than most screening tools.

However, the UC is applied as a post-processing correction to merit-order dispatch results (`apply_unit_commitment()` at line 417), not integrated into the dispatch optimization. In practice:

1. Merit-order dispatch decides which units run each hour based on marginal cost
2. UC constraints then modify the commitment schedule — extending run times to meet minimum up requirements, preventing shutdowns that violate minimum down times
3. This can create hours where a unit is committed (forced on by min-up) but has no economic demand to serve

Real SCUC (Security-Constrained Unit Commitment), as implemented in ISO market clearing engines, co-optimizes commitment and dispatch simultaneously. The difference is typically 2–5% of total production cost (per FERC Technical Conference proceedings on market design).

For screening purposes, post-hoc UC is a reasonable approximation. It captures the first-order effects (coal can't cycle daily, start costs are real) without the computational complexity of mixed-integer programming. But results should be presented with the caveat that dispatch costs may be understated by 2–5% relative to a co-optimized solution.

---

## 4. Comparative Positioning

How does this tool compare to production-grade alternatives?

| Dimension | Market Simulator | GenX (MIT) | PLEXOS (Energy Exemplar) | Aurora (Energy Aspects) | AMP (Ascend) | ReEDS (NREL) |
|-----------|-----------------|------------|--------------------------|------------------------|--------------|--------------|
| **Primary Question** | "What happens to generators?" | "Least-cost pathway to target" | "Detailed market clearing" | "Price forecasting" | "Asset valuation" | "Long-term capacity expansion" |
| **Transmission** | Copper-plate | Pipe-and-bubble | Full nodal (AC/DC) | Zonal | Zonal/nodal | Zonal (134 BAs) |
| **Unit Commitment** | Post-hoc (min up/down, starts) | Optional (relaxed MIP) | Full SCUC | Simplified | Full SCUC | None |
| **Storage Dispatch** | Greedy sequential | LP co-optimized | LP co-optimized | Heuristic + LP | LP co-optimized | LP |
| **Plant-Level** | Yes (EIA 860) | No (aggregate) | Yes | Partial | Yes | No (aggregate) |
| **Demand Response** | Minimal (>$200/MWh only) | Exogenous | Endogenous | Limited | Endogenous | Exogenous |
| **Inter-Regional** | None | Yes (transfer limits) | Yes (full network) | Yes | Yes | Yes |
| **Learning Curves** | Endogenous (Wright's Law) | Exogenous | None | Exogenous | None | Exogenous |
| **Scenario Throughput** | 270 in 30 min | 3–10 in days | 3–10 in days | 10–50 in hours | 5–20 in hours | 5–20 in hours |
| **Setup Time** | Minutes (web UI) | Days–weeks | Weeks–months | Days–weeks | Days–weeks | Days |
| **License Cost** | Free (internal) | Free (open-source) | $50K–$500K/yr | $100K+/yr | Custom | Free (open-source) |
| **Regulatory Acceptance** | No | Growing | Established | Established | Established | Established |

**Key takeaway**: This tool sacrifices transmission fidelity, co-optimized dispatch, and inter-regional trade for 10–100x faster scenario throughput and dramatically lower setup costs. That's a legitimate tradeoff for screening. It's not a legitimate tradeoff for investment-grade analysis.

---

## 5. Viable Use Cases

If I had to use this tool, here's where I'd deploy it with some confidence:

### 5.1 High Confidence (Directionally Reliable)

- **Parametric policy screening**: "At what carbon price does coal become uneconomic in each ISO?" The sweep mode gives robust answers across 270 scenarios. The rank ordering of ISOs by coal retirement sensitivity is reliable even if the exact $/ton threshold has ±$5–10 uncertainty.

- **IPP fleet profitability triage**: "Which of my 200 plants are at risk under $4/MMBtu gas and $30/ton carbon?" The plant-level Tier 2 mode with real heat rates gives actionable rankings. The absolute profit numbers need a zonal adjustment, but the relative ordering (which plants are most/least vulnerable) is useful.

- **Nuclear retirement risk assessment**: The 45U PTC CfD model correctly identifies when existing nuclear revenue drops below operating cost thresholds. The primary uncertainty (LMP at future clean penetration levels) is appropriately captured by the sweep mode's P10/P50/P90 bands.

- **CCS breakeven carbon pricing**: "What carbon price makes CCS-CCGT economically competitive in PJM?" This is a cost comparison question where the copper-plate limitation matters less — CCS economics are driven by carbon price and gas price, not locational congestion.

- **Pre-screening for production models**: Narrow the parameter space before running expensive GenX/PLEXOS analyses. Instead of running 50 GenX scenarios, use this tool to identify the 5–8 most interesting parameter combinations, then run GenX on those.

### 5.2 Moderate Confidence (Use with Caveats)

- **Corporate procurement guidance**: "Which ISOs offer the best clean energy economics for a data center?" Directionally correct, but the copper-plate assumption means the answer could change with specific site selection within an ISO.

- **Trajectory mode to 2035**: Wright's Law learning curves and demand growth produce reasonable trend lines. The demand-quantile LMP layer is increasingly unreliable beyond 2035, so treat the trajectory as a scenario narrative, not a forecast.

- **Storage deployment economics**: The greedy dispatch understates storage value by 8–15%. If the model says storage is economically viable, it almost certainly is. If it says storage is marginal, an LP co-dispatch might flip the answer.

### 5.3 Low Confidence (Screening Only, Do Not Rely)

- **Trajectory mode beyond 2040**: Too many compounding uncertainties. The demand-quantile layer, capacity market degradation curves, and learning curves are all extrapolating far from calibration data.

- **Plant-specific revenue projections**: Without zonal pricing, plant-level revenue numbers can be off by $10–20/MWh from reality. Use for relative ranking only, not absolute valuation.

- **High clean penetration (>85%)**: The LMP engine's behavior in this regime is dominated by the demand-quantile layer's extrapolation, not physics. Real market dynamics at >85% clean involve storage saturation effects, curtailment cascades, and reliability events that the model doesn't capture well.

---

## 6. Where This Tool Should NOT Be Used

To be unambiguous:

1. **Facility siting or asset valuation**: Needs nodal or at minimum zonal price resolution. A copper-plate model cannot distinguish between a plant in a congestion-constrained pocket and one at a transmission hub.

2. **Revenue projections for project finance**: Lenders and tax equity investors require P99 revenue estimates with auditable dispatch models. This tool's LMP outputs are not bankable.

3. **Regulatory filings or IRP proceedings**: Requires auditable co-optimized dispatch with SCUC, network constraints, and established model pedigree. No regulator will accept results from a tool without transmission modeling.

4. **Transmission planning**: The model has no network representation whatsoever.

5. **Hour-ahead or day-ahead price forecasting**: The demand-quantile approach is calibrated to annual statistics, not temporal precision.

---

## 7. Prioritized Recommendations

If I were advising the development team, here's what I'd prioritize:

### Priority 1: Simplified Zonal Decomposition
**Effort**: Medium (2–4 weeks)
**Impact**: High
**Approach**: 3–5 zones per ISO using published transfer limits. PJM (Western, AEP, MAAC, EMAAC, SWMAAC), MISO (North, Central, South), ERCOT (West, North, South, Houston). Zone-level demand/supply profiles from EIA-930 subregion data. Inter-zonal transfer limits from ISO planning reports. Single LP per hour to clear zonal markets.
**Why**: Unlocks the plant-level Tier 2 mode's full potential. Without zonal prices, plant-level economics are decorative rather than actionable.

### Priority 2: LP Storage Co-Dispatch
**Effort**: Low (1–2 weeks)
**Impact**: Medium
**Approach**: Replace greedy sequential loop with `scipy.optimize.linprog` co-dispatching all storage types simultaneously. Objective: minimize residual demand (or maximize arbitrage value). SOC constraints per type, shared surplus/deficit pool.
**Why**: Removes the systematic bias against storage-heavy portfolios. Relatively easy implementation for meaningful accuracy gain.

### Priority 3: Exogenous Inter-Regional Flows
**Effort**: Low (1 week)
**Impact**: Medium
**Approach**: Add hourly net import/export profiles per ISO from EIA-930 interchange data. Apply as a demand adjustment (imports reduce residual demand, exports increase it). No need to model the interconnection economics.
**Why**: Corrects resource adequacy calculations for ISOs with significant imports (NEISO, NYISO, CAISO). Minimal code change, high impact on RA-dependent results.

### Priority 4: Demand Response
**Effort**: Low (1 week)
**Impact**: Low-Medium
**Approach**: Price-elastic demand curtailment: when LMP exceeds a threshold (varies by ISO), shed 5–10% of load. PJM: 10 GW DR at $100+/MWh trigger. ERCOT: 5 GW at $200+/MWh.
**Why**: Reduces unrealistic scarcity pricing spikes, improves high-clean-penetration scenario realism.

### Priority 5: Trajectory Mode Backtesting
**Effort**: Medium (2–3 weeks)
**Impact**: High (for credibility)
**Approach**: Run the model starting from 2020 conditions and compare predicted 2020–2024 outcomes (coal retirements, LMP evolution, clean deployment) against observed data. This is the most convincing validation possible.
**Why**: The model has forward-looking validation (calibrated to 2024 SOM data), but no backward-looking validation. Demonstrating that the model would have correctly predicted 2020–2024 trends would substantially strengthen confidence in near-term trajectory outputs.

### Priority 6: Confidence Visualization
**Effort**: Low (1 week)
**Impact**: Medium (for user trust)
**Approach**: Add a visual confidence indicator to trajectory results — green (calibrated, 2025–2030), yellow (moderate extrapolation, 2030–2040), red (high uncertainty, 2040+). Display alongside the LMP and deployment charts.
**Why**: Prevents users from over-interpreting long-horizon projections. Sets appropriate expectations without undermining the tool's near-term value.

---

## 8. Bottom Line Assessment

**Don't scrap this tool.** The speed advantage is real, the question it asks is commercially relevant, and the implementation quality — plant-level heat rates, vintage-adjusted UC, calibrated per-ISO price models, Wright's Law learning curves — is substantially above the typical "spreadsheet model with a UI" that passes for a screening tool in many commercial settings.

**But be precise about what it is.** It's a fast market screening heuristic. The word "simulator" implies a level of physical fidelity that the copper-plate transmission model, greedy storage dispatch, and demand-quantile price formation don't deliver. Consider renaming to "Market Screening Tool" or "Market Economics Explorer" — something that sets expectations correctly.

**The internal peer review is directionally correct but too generous.** "Fit for purpose" is accurate for portfolio-level directional screening. It understates the limitations for plant-level analysis (where the copper-plate assumption creates qualitatively wrong answers for specific assets) and for trajectory mode beyond 2035 (where the demand-quantile layer is extrapolating well outside its calibration domain).

**With targeted improvements, this tool could graduate from "interesting prototype" to "credible screening instrument."** The three highest-impact additions — simplified zonal decomposition, LP storage co-dispatch, and exogenous inter-regional flows — would collectively address ~70% of the structural gap between this tool and a production screening model. None requires the computational complexity of full SCUC or nodal market clearing. All are well-solved problems with off-the-shelf implementations.

The tool's unique value proposition — rapid parametric exploration with endogenous learning curves and plant-level economics — is worth preserving and strengthening. But the gap between "directionally correct for portfolio questions" and "reliable for asset-specific decisions" is wider than the current documentation suggests. Close that gap honestly, either by adding zonal resolution or by restricting the tool's stated use cases to portfolio-level analysis.

I'd use this tool. I'd use it to pre-screen before running PLEXOS. I'd use it to brief stakeholders on directional trends. I'd use it to stress-test nuclear and CCS economics. I would not use it to make a $500M investment decision, and I would caution anyone who tried.

---

*Review prepared March 2026. Opinions expressed are based on code inspection, methodology documentation review, and comparison to established energy systems modeling standards (FERC Technical Conference proceedings, NREL ATB validation protocols, IEEE Power & Energy Society best practices for production cost modeling).*
