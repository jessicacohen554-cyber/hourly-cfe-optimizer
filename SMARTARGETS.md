# SMARTargets Module — Market Simulation of Clean Energy Deployment

> **Status**: Design phase — not yet implemented.
> **Last updated**: 2026-03-06 (added AT trajectory, 2023 baseline emissions, constraint-based optimization).

## Core Concept

A **market simulation** that answers: *"Under real-world market conditions, what clean energy resources would rational actors actually deploy, and how clean does the grid get?"*

This is fundamentally different from the existing pipeline:

| | Existing Pipeline (Steps 1-3) | SMARTargets |
|---|---|---|
| **Question** | "What's the cheapest way to reach X% clean?" | "What gets built when developers chase profit?" |
| **Input** | CFE target (50%, 90%, 99%...) | Market conditions (prices, policy, learning) |
| **Output** | Optimal resource mix + cost | Emergent CFE level + resource mix + market dynamics |
| **Objective** | Minimize system cost | Maximize developer profit (revenue - cost) |
| **Who decides** | Central planner | Decentralized market actors |
| **Target** | Exogenous (user picks it) | **Endogenous** (emerges from profitability) |

The CFE target is an **output**, not an input. The model deploys whatever's profitable at each step — the grid gets as clean as the market incentives allow. Under a $0 carbon price with high LCOE, it might plateau at 60%. Under a $100 carbon price with learning curves, it might reach 95%+.

**Key market dynamics captured**:
- **Revenue cannibalization**: Solar suppresses its own prices (duck curve). Adding more solar makes the *next* MW of solar less profitable.
- **Scarcity value**: As fossil retires, remaining dispatchable capacity earns premium prices. This pulls in clean firm (nuclear, CCS, LDES) even at high LCOE.
- **Learning curves**: Early deployment (even if marginally profitable) reduces costs for later deployment, potentially unlocking a cascade.
- **Capacity market signals**: Resources with high ELCC earn capacity payments that improve economics, but these payments degrade as clean supply grows.

## Architecture

### What We Borrow (Existing Modules)

**Primary architecture: Step 5D's zone-delta structure** — not Step 7A's full-mix optimizer.

Step 5D (`compute_zone_metrics` / `build_consequential_queue`) works in *incremental zone transitions* (50→55%, 55→60%, etc.), computing delta cost and delta CO2 per zone. This is the right foundation because SMARTargets asks "is the *next increment* profitable?" — an inherently incremental question.

Step 7A (`_forward_step_optimization`) selects the best *absolute mix* at each threshold — it answers "what's cheapest to reach X%?" which is a different question entirely. Step 7A is not obsolete (it's still needed for procurement strategy pages) but it's not the right architecture for a market simulation.

| Component | Source | What We Use |
|-----------|--------|-------------|
| **Zone-delta structure** | `step5d::compute_zone_metrics()` | Incremental cost/resource/CO2 deltas per zone transition. Add revenue delta for profit calculation. |
| **Sequential zone stepping** | `step5d` / `scenario_common::build_consequential_queue()` | Process zones in order, each step building on prior. Already supports `per_iso_sequential` and `global_merit_order` sequencing. |
| **Dispatch-based CO2 per zone** | `step5d::get_dispatch_co2()` | 8,760-hour CO2 accounting at each threshold boundary — needed for carbon price revenue. |
| **Cross-regional sequencing** | `scenario_common::build_consequential_queue(sequencing='global_merit_order')` | Capital flows to highest-profit zone across ISOs (re-rank by profit instead of MAC). |
| **8,760-hour dispatch** | `dispatch_utils.py` / Step 4 cache | Hourly supply-demand matching for LMP computation. |
| **LMP merit-order engine** | `step5b::build_merit_order_stack()` + `PriceModel` subclasses | Synthetic hourly LMP from fossil stack composition at each threshold. |
| **Scarcity pricing** | `step5b` ISO-specific models | Reserve-based exponential adder (ERCOT ORDC, PJM RPM-calibrated). |
| **Wright's Law learning curves** | `step8d` + `procurement_utils.py` | FOAK→NOAK cost reduction as global cumulative GW deployed increases. |
| **Capacity market revenue** | `pipeline_config.py::CAPACITY_MARKET_PRICES` | $/kW-yr by ISO (zero for energy-only markets: ERCOT, SPP). |
| **Storage revenue stacking** | `pipeline_config.py::compute_storage_revenue_credit()` | Arbitrage + capacity + ancillary with 70% co-optimization efficiency. |
| **Gas backup tracking** | `step5d` zone metrics | Already tracks `gas_backup_mw_start/end` and `delta_gas_mw` per zone. |

### What We Do NOT Borrow

| Component | Source | Why Not |
|-----------|--------|---------|
| **MAC ranking** | `step5d::build_consequential_queue()` | Queue ranks by cheapest $/tCO2. We re-rank by highest profit. Same structure, different sort key. |
| **Step 7A full-mix optimizer** | `step7a::_forward_step_optimization()` | Selects best absolute mix per threshold. Wrong framing — we need incremental profitability, not system-optimal mix. Step 7A answers "what should a planner build?" — we answer "what would the market build?" |
| **Step 7A EF/PFS filtering** | `scenario_common::batch_filter_floor()` | Not needed — step 5d's zone structure already handles monotonicity via zone ordering. |

### What's New (~30-50 Lines of Glue Code)

| Component | Description |
|-----------|-------------|
| **Revenue calculator** | For each zone transition: compute hourly LMP revenue from `step5b` engine, add capacity market payments, add storage revenue credits. Extends `zone_metrics` with `delta_revenue_per_mwh`. ~20 lines. |
| **Profit metric** | `delta_profit = delta_revenue - delta_cost` per zone. Replace MAC sort key with profit sort key in `build_consequential_queue`. ~10 lines. |
| **Stopping rule** | When `delta_profit < 0` for all remaining zones across all ISOs, stop. Record current clean level as market equilibrium. ~5 lines. |
| **Price feedback** | After selecting a zone, recompute LMP for the new fossil stack. Natural sequential feedback — already how step 5d processes zones. |
| **Capacity price degradation** | `capacity_price(t) = base_price × max(0, 1 - α × clean_share(t))`. α calibrated per ISO. ~5 lines. |
| **Carbon price lever** | Expose `co2_level` parameter from existing `compute_marginal_costs()` as a scenario axis. Already in step5b — just thread through. |

## Sequential Algorithm

The model steps through increasing clean percentages — but these aren't *targets*. They're measurement points. At each step, the model asks "is it profitable to go from here to the next level?" If yes, deploy. If no, stop — that's the market equilibrium.

```
For each scenario (15-20 market condition combos):
  Initialize:
    cumulative_gw = {resource: global_installed_2025}  # for learning curves
    deployed_state = {iso: existing_clean_floor}       # per-ISO current state

  Build zone list:
    - ISO-aware zones from step5d (e.g., MISO gets 30→40, 40→50, 50→55, ...)
    - Each zone = one threshold increment with delta_cost, delta_resources, delta_co2

  For each zone (sequenced by profit, per-ISO or global):

    1. COST DELTA: Incremental LCOE for new resources in this zone  [step5d pattern]
       - delta_cost = zone_end_cost - zone_start_cost               [already computed]
       - Apply Wright's Law reduction based on cumulative_gw         [procurement_utils]
       - Global cumulative: all ISO deployments pool together

    2. REVENUE DELTA: Market revenue earned by new resources         [NEW]
       a. LMP at this clean level from fossil stack                  [step5b.build_merit_order_stack]
          - Fossil stack shrinks as clean rises → prices change
          - Scarcity pricing kicks in when reserves thin
       b. Hourly energy revenue for incremental resources            [dot product, vectorized]
          - Revenue depends on WHICH resources are added (solar vs firm)
          - Solar earns daytime LMP (suppressed by cannibalization)
          - Clean firm earns baseload LMP (premium during scarcity)
          - Storage earns arbitrage spread
       c. Capacity market payment for new capacity (degrading)
       d. Ancillary service revenue
       e. Carbon credit revenue (if carbon price > $0)               [step5d CO2 data]
       f. REC revenue from state RPS compliance                      [NEW — Decided]
          - Resources in RPS states earn REC $/MWh on eligible generation
          - REC price != ACP — ACPs are the penalty cap, RECs trade below
          - REC revenue is additive to energy revenue for eligible resources
          - NOTE: RPS mandates do NOT force builds directly in the model;
            REC revenue makes resources more profitable, which may trigger
            builds via the standard profit criterion. ACP < REC means
            some obligated entities pay the penalty rather than build,
            so RPS doesn't guarantee full compliance.
          - Interconnection queue delays throttle the RATE of new builds
            (modeled as max annual GW deployment cap per ISO, not RPS-specific)
       g. Federal tax credits (Decided — included in effective LCOE):
          - **45Y (Clean Electricity PTC)**: Technology-neutral PTC for
            NEW builds — solar, wind, AND nuclear uprates. $26/MWh (2024$),
            inflation-adjusted. Applies to projects placed in service after
            2024. Replaces 45/48 for new projects.
          - **45U (Nuclear PTC)**: Existing nuclear production credit,
            $15/MWh, through 2032. Keeps existing nuclear competitive
            when market LMPs are low. Critical for nuclear retirement
            calculations — without 45U, some existing nuclear may be
            uneconomic pre-2032 but viable with the credit.
          - **45Q (CCS credit)**: Already modeled — $85/ton for power
            sector CCS. Reflected as LCOE offset in CCS-CCGT cost.
          - Implementation: Credits reduce effective LCOE (cost side),
            not modeled as revenue. Net effect is the same for profit
            calculation, but conceptually cleaner.

    3. PROFIT: delta_profit = delta_revenue - delta_cost             [NEW]
       - If delta_profit > 0: DEPLOY — but check portfolio stop first
       - If delta_profit ≤ 0: SKIP / STOP (market stop)
         → But check: might LATER zones be profitable?
           (e.g., learning from other ISOs lowers cost, or
            further fossil retirement raises scarcity value)

    3b. PORTFOLIO STOP (DAC crossover):                              [step6b pattern]
       - marginal_mac = delta_cost / delta_co2_avoided              [$/tCO2]
       - If marginal_mac > dac_cost_per_ton: STOP — buy DAC instead
         → Grid decarbonization beyond this point is more expensive
           per ton than direct air capture. Rational economy-wide
           portfolio uses DAC/removals for remaining emissions.
       - DAC cost is a scenario axis (Low/Medium/High from step6b)
       - This is the KEY INSIGHT: 99.99% clean grids may not make
         economic sense when DAC can offset remaining fossil cheaper.
       - The optimal grid clean level is WHERE marginal MAC = DAC cost
         (already computed in step6b as the PCHIP crossover point)

    4. LEARN (if deployed):                                          [step8d pattern]
       cumulative_gw[resource] += new_capacity_gw
       # Deployment lowers future costs for ALL ISOs (global learning)

    5. PRICE UPDATE (automatic):
       - Next zone's LMP reflects updated fossil stack
       - Scarcity value shifts → changes which resources are profitable
       - Cannibalization shifts → changes VRE revenue

  Terminate when:
    - MARKET STOP: All remaining zones have delta_profit ≤ 0
      AND no learning could flip any zone positive, OR
    - PORTFOLIO STOP: marginal MAC exceeds DAC cost for all remaining
      zones — cheaper to remove residual emissions via DAC than to
      clean the grid further
    - Whichever binds first determines the market-equilibrium clean level
    - Record per-ISO: clean level, residual emissions, DAC needed for net-zero
```

**The stopping point IS the result.** Each scenario produces a different market-equilibrium clean level per ISO. The dashboard shows how market conditions determine where deployment stalls or accelerates.

### Economy-Wide Portfolio Framing (Decided)

Grid decarbonization doesn't happen in a vacuum. The right question isn't "how do we get the grid to 100% clean?" — it's "how clean should the grid get before economy-wide portfolio optimization says to spend the next dollar elsewhere?"

**The existing pipeline already answers this.** Step 6B computes the optimal CFE target per ISO as the point where marginal MAC crosses DAC cost (PCHIP spline interpolation). The 3×3 grid-cost × DAC-scenario matrix gives 9 crossover points per ISO. This is exactly the "fuller option" model.

**How it connects to SMARTargets:**
- The **market stop** tells you where deployment stalls without policy intervention
- The **portfolio stop** tells you where a rational net-zero strategy switches from grid clean to DAC/removals
- The gap between them is the policy design space: how much policy intervention is needed to push the market from its natural equilibrium to the portfolio-optimal clean level?
- If market stop > portfolio stop: the market naturally overshoots — no intervention needed (unlikely but possible with very low clean LCOE + high carbon price)
- If market stop < portfolio stop: policy needed to close the gap (likely the common case)

**Net-zero pathways always include DAC for residual emissions.** A 90% clean grid + DAC for the remaining 10% may be cheaper than a 99.99% clean grid. The model shows exactly where that crossover lives per ISO, per scenario.

**DAC cost as scenario axis**: Low ($150/ton) / Medium ($250/ton) / High ($400/ton) — same levels as Step 6B. As DAC costs fall (learning), the optimal grid clean level *decreases* — you don't need to push the grid as far when removals are cheap.

**Key difference from step 5d**: Step 5d ranks zones by MAC ($/tCO2) — this ranks by profit ($/MWh). Same zone structure, different objective. Step 5d asks "where's abatement cheapest?" — SMARTargets asks "where do developers make money?"

## 2023 Baseline Emissions

The AT trajectory measures absolute CO₂ reductions from a **2023 baseline year**. Two separate data sources serve two different modeling purposes:

### ISO-Level Baseline (eGRID — for Regional AT Modeling)

**Source**: EPA eGRID 2023 detailed data download, plant-level sheet (PLNT23).

**Method**: Sum `PLCO2AN` (annual CO₂ emissions, short tons) for all plants mapped to each ISO via BA code (`BACODE`). Convert short tons → metric tons (÷ 1.10231). This gives total annual power sector CO₂ emissions per ISO in metric tons.

**Purpose**: Used as the denominator for S1–S4 AT trajectory constraints. The emission cap for each year is a percentage of this baseline.

**Processing**: `scripts/step0_extract_egrid_baselines.py` — reads the plant-level sheet, sums `PLCO2AN` by BA code, converts to metric tons. Output: `data/egrid_2023_baseline_emissions.json` with per-ISO totals and AT trajectory caps.

**Data file**: `data/egrid_2023_baseline_emissions.json` (extracted from `data/egrid2023_data_rev2 2.xlsx`, BA23 sheet, column `BACO2AN`).

| ISO | 2023 Baseline CO₂ (M metric tons) | 2023 Gen (TWh) | Source |
|-----|-----------------------------------:|---------------:|--------|
| CAISO | 31.38 | 187.3 | eGRID 2023 BA23 |
| ERCOT | 157.46 | 472.9 | eGRID 2023 BA23 |
| PJM | 267.32 | 823.0 | eGRID 2023 BA23 |
| NYISO | 28.19 | 129.7 | eGRID 2023 BA23 |
| NEISO | 25.08 | 102.1 | eGRID 2023 BA23 |
| MISO | 290.40 | 652.0 | eGRID 2023 BA23 |
| SPP | 110.51 | 279.5 | eGRID 2023 BA23 |
| **TOTAL** | **910.34** | **2,646.5** | |

**AT Trajectory Emission Caps (M metric tons)**:

| ISO | 2030 (−57%) | 2035 (−82%) | 2040 (−88%) | 2045 (−94%) | 2050 (−100%) |
|-----|------------:|------------:|------------:|------------:|-------------:|
| CAISO | 13.49 | 5.65 | 3.77 | 1.88 | 0 |
| ERCOT | 67.71 | 28.34 | 18.90 | 9.45 | 0 |
| PJM | 114.95 | 48.12 | 32.08 | 16.04 | 0 |
| NYISO | 12.12 | 5.07 | 3.38 | 1.69 | 0 |
| NEISO | 10.78 | 4.51 | 3.01 | 1.50 | 0 |
| MISO | 124.87 | 52.27 | 34.85 | 17.42 | 0 |
| SPP | 47.52 | 19.89 | 13.26 | 6.63 | 0 |

**Note**: Our model uses 2025 generation profiles for physics simulation, but the emissions baseline for AT trajectory measurement is 2023. This is intentional — targets are measured against a fixed historical baseline, not a moving one.

### Company-Level Baseline (Sustainability/CDP/10-K — for IPP QT Modeling)

**Source**: Company self-reported data, NOT eGRID plant-to-parent mapping (which is unreliable for complex corporate structures).

**Data sources** (in order of preference):
1. **CDP Climate disclosures** — CO₂ from power generation specifically (Scope 1, category-level)
2. **Sustainability reports** — 2023 Scope 1 emissions from electricity generation
3. **10-K filings (SEC)** — generation by fuel type, capacity by region/ISO, emissions reporting
4. **S&P Global** — fleet capacity and generation data mapped to ISOs
5. **Company websites** — investor relations, fleet composition pages

**Purpose**: Establishes each IPP's 2023 CO₂ baseline for QT target-setting. The IPP's QT is tested as a percentage reduction from this company-specific baseline.

**Constellation special case**: Constellation acquired Calpine in January 2026. The pro-forma 2023 baseline for the combined entity is:
```
Constellation 2023 baseline = Constellation 2023 Scope 1 + Calpine 2023 Scope 1
                              - plants divested as acquisition condition
```
Research needed: identify which plants Constellation is being forced to divest and their 2023 CO₂ contribution.

**Fleet-to-ISO mapping**: Use 10-K and S&P data to map each company's generation assets to ISOs. This enables scaling regional AT constraints to company-level: "If ERCOT must reduce CO₂ by 57% by 2030, what does that imply for an IPP with X% of ERCOT generation?"

## AT Reduction Trajectory (Decided)

The Aspirational Target trajectory defines **absolute CO₂ metric ton caps** on power sector emissions, measured against the 2023 baseline:

| Year | Reduction from 2023 | Remaining Emissions | Cap Formula |
|------|---------------------|---------------------|-------------|
| 2023 | 0% (baseline) | 100% | `baseline_2023` |
| 2030 | **−57%** | 43% | `baseline_2023 × 0.43` |
| 2035 | **−82%** | 18% | `baseline_2023 × 0.18` |
| 2040 | **−88%** | 12% | `baseline_2023 × 0.12` |
| 2045 | **−94%** | 6% | `baseline_2023 × 0.06` |
| 2050 | **−100%** | 0% | `0` |

### These Are Constraints, Not Goals

The emission cap is a **hard constraint** — the system cannot exceed it. The optimization question becomes: *"What is the least-cost resource mix that keeps emissions at or below the cap in each year?"*

This is fundamentally different from "minimize cost to reach X% clean energy." The constraint is on CO₂ tons, not clean percentage. The required clean % is an output that depends on:
- The cap (from the trajectory above)
- Projected demand (which grows over time)
- The emission intensity of the remaining fossil fleet (which shifts as coal → gas retirement occurs)

### Demand Growth Makes Later Targets Harder

Because the caps are **absolute** (metric tons, not intensity), demand growth directly increases the difficulty:

```
Example — ISO with 100M tCO2 baseline, 2% annual demand growth:

2030: Cap = 43M tCO2. Demand grew ~15%. Need ~50% clean to hit cap.
2035: Cap = 18M tCO2. Demand grew ~26%. Need ~78% clean to hit cap.
2040: Cap = 12M tCO2. Demand grew ~37%. Need ~88% clean to hit cap.

Achieving -57% in 2030 does NOT scale forward — if demand grew 10% between
2030 and 2035 and you added no new clean, emissions would INCREASE (more
demand served by same fossil fleet). The 2035 target requires additional
new-build clean capacity on top of what was deployed for 2030.
```

This creates an accelerating buildout requirement: each milestone demands not just maintaining the prior clean level, but adding enough new clean capacity to (a) serve demand growth AND (b) further reduce the fossil share.

### Same Trajectory for All AT Scenarios

S1, S2, S3, and S4 all use the same −57/−82/−88/−94/−100% trajectory. The scenarios differ in **how hard and expensive** it is to meet the trajectory, not in the trajectory itself:
- **S1/S3** (Facilitating): Lower LCOE, faster learning, reformed queues → cheaper path to the same caps
- **S2/S4** (Challenging): Higher LCOE, slower learning, constrained queues → more expensive path
- **S3/S4** (Economy-wide): Higher demand from mandatory electrification → higher required clean % for the same absolute cap

## Constraint-Based Optimization Approach (Decided)

### No Carbon Price Iteration Needed

The existing pipeline can handle emission caps directly — no need to build a carbon price in and iterate to find equilibrium. The approach:

```
For each year in the trajectory (2030, 2035, 2040, 2045, 2050):

  1. COMPUTE EMISSION CAP
     cap_tCO2 = baseline_2023_tCO2[iso] × (1 - reduction_pct[year])

  2. COMPUTE PROJECTED DEMAND
     demand_TWh = demand_2023_TWh[iso] × (1 + growth_rate) ^ (year - 2023)

  3. INVERT CO₂ MODEL → REQUIRED CLEAN %
     The merit-order retirement model (step5a) maps clean% → CO₂.
     Invert: find minimum clean% where fossil_CO₂ ≤ cap_tCO2.
     This accounts for coal→gas fuel switching as clean% rises.

  4. FIND LEAST-COST MIX AT REQUIRED CLEAN %
     Use existing Steps 1-3 pipeline: PFS → EF → cost optimization
     at the required clean% threshold. This gives the cheapest resource
     mix that satisfies the emission constraint.

  5. APPLY LEARNING CURVES (Year-Over-Year)
     Costs in 2035 reflect cumulative deployment through 2030.
     Wright's Law: cost(year) = f(cumulative_GW_deployed_through_prior_years)
     Earlier deployment → lower costs for later milestones.

  6. RATCHET CONSTRAINT
     Capacity built for 2030 doesn't retire — it carries forward.
     2035 only needs INCREMENTAL clean capacity above the 2030 fleet.
     But demand growth means the increment may be substantial.
```

### Shadow Carbon Price as Output

After finding the least-cost constrained solution, the **marginal cost of the last unit of abatement** is the implied carbon price — what a carbon price would need to be to make the market naturally hit this target. This is reported as an insight:

```
shadow_carbon_price[year] = marginal_MAC at the required clean%
                          = (incremental_cost / incremental_CO₂_avoided)
                          at the emission cap boundary
```

This answers the S1-S4 question "what carbon price or equivalent forcing is needed?" without requiring an iterative solver. The shadow price falls out naturally from the constrained optimization.

### Why This Works Without Iteration

The key insight: our model already parameterizes by clean %, and step5a already maps clean % → CO₂. The emission constraint just adds one step — converting the CO₂ cap to a required clean % — before feeding into the existing least-cost optimization. No equilibrium search needed because:
- The CO₂-to-clean% mapping is monotonic (higher clean% = lower CO₂)
- The cost optimization at a given clean% is already solved (Steps 1-3)
- The only new computation is the inversion step, which is a simple 1D search

## Revenue Model Detail

### Energy Revenue (Hourly LMP × Generation Profile)

For each resource `r` in a candidate mix:
```
energy_revenue_r = sum(generation_profile_r[h] × lmp[h] for h in 0..8759) / demand_mwh
```

The generation profiles already exist in the dispatch cache (Step 4). The LMP array comes from step5b's `compute_hourly_lmp_vectorized()`. This is a dot product — fully vectorized.

**Key insight**: Resources that generate during high-price hours (clean firm, storage discharge) earn more per MWh than resources that generate during low-price hours (midday solar in CAISO). This is the *value* signal that MAC-based optimization misses — a resource can have low LCOE but also low revenue if it generates during hours when prices are suppressed by its own output (the "solar duck curve" / cannibalization effect).

### Capacity Revenue (Degrading with Clean Penetration)

```python
capacity_rev_per_kw_yr = CAPACITY_MARKET_PRICES[iso] × max(0, 1 - alpha × clean_share)
capacity_rev_per_mwh = capacity_rev_per_kw_yr × resource_elcc × 1000 / 8760
```

- `alpha` calibrated per ISO from historical auction trends (~0.3-0.5)
- ERCOT and SPP = $0 (energy-only markets)
- Resources with high ELCC (nuclear, CCS, storage) earn more capacity revenue
- As clean share rises, capacity clearing prices fall (more supply in auction)

### Storage Revenue (Already Implemented)

`pipeline_config.py::compute_storage_revenue_credit()` already computes:
```
credit = capacity_payment + (arbitrage + ancillary) × 0.70 stacking_factor
```
This converts to $/MWh via duration. Can be used directly.

### Carbon Price Revenue (Optional Lever)

```python
carbon_rev_per_mwh = carbon_price × avoided_emission_rate  # $/MWh
```
- `carbon_price`: scenario axis ($0, $50, $100, $185)
- `avoided_emission_rate`: from dispatch-based CO2 model (step5a)
- Only relevant if a carbon price or ETS exists

### REC Revenue from State RPS Compliance (Decided)

```python
rec_revenue_per_mwh = rec_price[iso] × rps_eligible[resource]  # $/MWh
```

- **REC price by ISO** — reflects actual RPS compliance market prices, NOT ACPs. ACP is the penalty ceiling; RECs trade below ACP because some entities pay the penalty rather than procure RECs.
- **Key nuance**: RPS mandates do NOT force builds in our model. Instead, REC revenue makes clean resources more profitable. If `energy_LMP + REC_price + capacity_rev > annualized_LCOE`, a developer builds. If not, the resource isn't built and some load-serving entities pay the ACP instead.
- **Eligible resources**: Solar, wind (onshore + offshore), hydro, geothermal. Nuclear eligibility varies by state — most state RPS programs exclude existing nuclear. New nuclear may qualify under some clean energy standards (e.g., NY CLCPA).
- **Interconnection queue delays**: Modeled as a max annual GW deployment rate cap per ISO, not RPS-specific. This constraint throttles the *rate* of new builds regardless of profitability.
- **REC prices to research**: Need ISO-level REC price data (SREC, Class I, etc.) — varies significantly by state/market. This is an open data need.

### Federal Tax Credits (Decided)

| Credit | Type | Rate | Applies To | Duration | Effect |
|--------|------|------|------------|----------|--------|
| **45Y** | New-build PTC | ~$26/MWh (2024$) | Solar, wind, **nuclear uprates**, other zero-emission | Post-2024 projects | Reduces effective LCOE for ALL new clean builds |
| **45U** | Existing nuclear PTC | $15/MWh | Existing nuclear plants | Through 2032 | Keeps existing nuclear competitive when LMPs are low |
| **45Q** | CCS credit | $85/ton | Power sector CCS | Already modeled | LCOE offset for CCS-CCGT |

**Implementation**: All credits modeled as LCOE reductions (cost side), not as revenue. For the profit calculation `revenue - cost`, reducing cost by $X is equivalent to increasing revenue by $X. Conceptually cleaner to keep credits on the cost side.

**45U sunset (2032) is important**: After 2032, existing nuclear loses $15/MWh of support. In ISOs where nuclear is marginally economic, this could trigger retirements post-2032. The model should handle this as a time-dependent LCOE adjustment — 45U applies in simulation years ≤2032, then drops off.

## Price Feedback Mechanism

The feedback is **implicit and free** — no iterative equilibrium solver needed:

1. At threshold t=50%, fossil stack is large → LMP is low → renewables may not be very profitable
2. At t=75%, significant fossil has retired → LMP rises (less supply) → clean firm becomes more profitable
3. At t=95%, fossil is scarce → scarcity pricing spikes → dispatchable clean (nuclear, CCS, LDES) earns premium
4. At t=99%, extreme scarcity → only firm/storage resources profitable enough to justify deployment

This happens naturally because `build_merit_order_stack()` takes `clean_pct` as input and sizes the fossil fleet accordingly. Each threshold step gets a different fossil stack → different LMP distribution → different revenue profile → different profit-maximizing mix.

**The sequential structure IS the feedback loop.** No iteration needed.

## Reference Case Design (Decided)

### Core Principle: Market-Driven Build-Out

R1/R2 are NOT "freeze existing clean + add demand." They are **full market simulations** — developers build whatever is profitable (clean OR fossil), and existing units retire when they can't cover costs. The reference case answers: *"What does the market build on its own, without clean energy policy?"*

### New-Build Options
- **Fossil**: Gas CCGT (baseload) and gas CT (peakers) only. **No new coal** — new coal LCOE ($65-90/MWh) is not competitive with new gas CCGT ($45-65/MWh) regardless of political environment. Economically unrealistic in any 2025+ scenario.
- **Clean**: Solar, onshore wind, offshore wind, battery (4hr/8hr), nuclear, CCS-CCGT, LDES, Green H2, geothermal (CAISO only). Same resource set as existing pipeline.
- **Build criterion**: Resource is built when `annual_revenue > annualized_LCOE` (positive profit at market prices).

### Retirement Logic
- **Economic retirement**: Existing unit retires when `annual_revenue < fixed_O&M`. Revenue based on dispatch hours × LMP at those hours.
- **Natural retirement order**: Coal first (highest fixed costs, displaced by cheaper gas in merit order), then older/less efficient gas as clean penetration suppresses prices.
- **Wind/solar effectively never retire** — near-zero marginal cost AND near-zero fixed O&M means they always run when available regardless of price. They're always merit-order competitive.
- **Nuclear is the interesting case** — fixed O&M ~$30/MWh means it needs decent capacity factors and prices. If LMPs crash from solar oversupply, a nuclear plant could theoretically become uneconomic and retire. **Exception: CAISO/Diablo Canyon** — California's Clean Energy Standard effectively mandates Diablo Canyon stays online indefinitely regardless of market economics. In the reference case (current policy), Diablo Canyon is policy-protected from retirement. Other ISOs' nuclear fleets face market retirement risk.
- **Hydro** — existing only (no new-build), very low O&M, effectively never retires.

### Existing Clean Viability
- The model validates that existing 2025 clean resources remain economically viable under reference case pricing.
- If an existing clean resource can't cover fixed O&M, it retires — this is a real risk for nuclear under high-solar scenarios.
- This validation is important: it confirms the "starting fleet" is stable before layering on policy scenarios.

### R1 vs R2

See full scenario matrix below (§ Scenario Matrix). R1/R2 are both "no carbon constraint" reference cases differing on whether market conditions facilitate or challenge clean deployment.

| | R1: Facilitating | R2: Challenging |
|---|---|---|
| **Starting fleet** | 2025 existing (all resources) | 2025 existing (all resources) |
| **Demand growth** | Medium (~1-1.5%/yr) | High (~2-3%/yr, AI/electrification) |
| **Carbon price** | $0 | $0 |
| **Policy** | Current (45Y/45U/45Q + REC revenue) | Current (same — 45Q stays intact) |
| **Interconnection queues** | Reformed (~5-8 GW/yr/ISO) | Constrained (~2-4 GW/yr/ISO) |
| **Learning curves** | Fast (high learning rate per doubling) | Slow (low learning rate per doubling) |
| **Starting LCOE** | Low (optimistic 2025 costs) | High (pessimistic 2025 costs) |
| **Key question** | How clean does the grid get when everything goes right but there's no carbon price? | How clean does the grid get when it's an uphill battle? |

### Cost Baseline: Track 1 (ECF) Across All Scenarios (Decided)

**All SMARTargets scenarios use Track 1 (Existing Clean Floor) as the cost baseline.** This is actual market modeling — the real world starts with an existing fleet that has sunk capital costs.

- **Existing assets**: Judged on fixed O&M economics (sunk capital), not greenfield LCOE. An existing nuclear plant doesn't need to recover its construction cost — it needs `revenue ≥ fixed_O&M` to stay online.
- **New builds**: Evaluated at full annualized LCOE (greenfield economics). A developer builds when `revenue > annualized_LCOE_newbuild`.
- **Track 2 (NB) and Track 3 (CTR) have NO role here.** Track 2 simulates building a new grid from scratch (greenfield) — purely theoretical. Track 3 is a nuclear-retirement counterfactual — also purely theoretical. Neither reflects actual market conditions. They exist in the pipeline as analytical exercises but are irrelevant to market simulation.
- **Source data**: Step 3A (Track 1 baseline) parquets only. Step 3B outputs are not used by SMARTargets.

### Reference Case Logic (Per Step)
```
For each step in the simulation:
  1. Compute LMPs from current fleet (merit-order dispatch)
  2. For each EXISTING unit (Track 1 economics):
     - Compute annual revenue (dispatch hours × LMP)
     - Add 45U credit (nuclear only, through 2032)
     - If revenue < fixed_O&M → RETIRE
  3. For each potential NEW BUILD (gas CCGT, gas CT, solar, wind, battery, etc.):
     - Compute expected revenue at current LMPs
     - Add REC revenue if RPS-eligible
     - If revenue > annualized LCOE (after 45Y/45Q credits) → BUILD
  4. Update fleet composition (retirements + new builds)
  5. Recompute LMPs with updated fleet → next step
```

## Endogenous Wright's Law Learning Curves (Decided)

### Core Principle: New Technology Frontier, Not Legacy Fleet

Learning curves for nuclear start from the **new nuclear frontier** — SMRs, advanced reactors, new AP1000s — NOT from the ~440 GW of legacy reactors built last century. These are fundamentally different technologies with different supply chains, manufacturing processes, and cost structures. The legacy fleet's cumulative experience doesn't transfer to SMR learning.

**One learning pool per technology class (Decided).** All new nuclear designs (BWRX-300, NuScale, AP1000, Natrium, Xe-100) contribute to a single "new nuclear" learning pool. Supply chain, regulatory, construction management, and workforce learnings are shared across designs even if reactor architectures differ. Same principle applies to all other technology classes — one pool each for CCS, LDES, etc. Keep it simple.

**Starting cumulative GW baselines** (the denominator for doubling calculations):

| Technology | 2025 Cumulative GW | Rationale | First Doubling At |
|-----------|-------------------|-----------|-------------------|
| **New nuclear** (SMR/advanced/AP1000) | ~2 GW | Vogtle 3&4 (2.2 GW) — only modern US nuclear. Global: Barakah, Olkiluoto, Hinkley = ~10 GW but different designs. Use US-only as conservative start. | ~4 GW |
| **CCS-CCGT** | ~0.3 GW | Boundary Dam, Petra Nova (both partial). Near-zero at power-sector scale. Effectively greenfield. | ~0.6 GW |
| **LDES (iron-air)** | ~0.01 GW | Form Energy pilot scale only. Pre-commercial. | ~0.02 GW |
| **Green H2** | ~0.1 GW | Electrolysis at scale barely exists. | ~0.2 GW |
| **Geothermal (enhanced)** | ~0.05 GW | Fervo pilot. Next-gen EGS is new frontier. | ~0.1 GW |
| **Battery (Li-ion)** | ~50 GW (US grid) | Already at manufacturing scale — learning curve is shallow/mature. | ~100 GW |
| **Solar** | ~150 GW (US) | Mature. Static cost in model. | N/A |
| **Wind (onshore)** | ~150 GW (US) | Mature. Static cost in model. | N/A |

### The 7-ISO Scope Problem (Decided)

We model 7 ISOs (~65% of US generation), not the whole country or world. This creates a learning curve attribution challenge:

**Problem**: If PJM builds 2 GW of new nuclear and ERCOT builds 1 GW, that's 3 GW within our model. But in reality, global deployment (Korea, France, UK, Canada, rest-of-US) also contributes to learning. Our 7 ISOs aren't the whole market.

**Solution — Exogenous background learning + endogenous model learning**:

```
effective_cumulative_gw = model_deployed_gw + background_learning_gw(year)
```

- **`model_deployed_gw`**: Endogenous — cumulative GW built within the 7 ISOs during the simulation. This is what the model controls.
- **`background_learning_gw(year)`**: Exogenous — assumed rest-of-world/rest-of-US deployment trajectory. A scenario parameter (Fast/Medium/Slow global adoption) that adds learning from outside our model boundary.
- **Effect**: Even if our 7 ISOs build zero new nuclear, costs still decline (slowly) from global deployment. But our ISOs' own deployment accelerates the decline.

**Background learning scenarios**:

| Scenario | Global new nuclear by 2035 | Global new nuclear by 2050 | Notes |
|----------|--------------------------|--------------------------|-------|
| **Slow** | +5 GW | +20 GW | Only committed projects (Sizewell C, a few SMR demos) |
| **Medium** | +15 GW | +60 GW | IEA NZE-aligned, moderate SMR ramp |
| **Fast** | +30 GW | +150 GW | Aggressive SMR/advanced nuclear buildout (US + global) |

Same pattern applies to CCS, LDES, etc. — each technology has its own background trajectory.

**Why this works**: The model's endogenous deployment *adds to* the global background. Even if our 7 ISOs are slow to add new nuclear, the background learning from TVA/non-ISO/global deployment still drives costs down. An ISO that waits still benefits from external cost decline — it just doesn't contribute to accelerating it.

### US New Nuclear Pipeline — ISO vs Non-ISO (Research, Jan 2026)

The near-term US new nuclear pipeline is roughly **40/60 in-model vs out-of-model**:

**Within our 7 ISOs (~5-6 GW):**
| Project | ISO | Capacity | Timeline |
|---------|-----|----------|----------|
| Fermi AP1000s (×4) | ERCOT (Carson County, TX) | 4.4 GW | 2032-2036 |
| X-energy Xe-100 | ERCOT (Calhoun County, TX) | 320 MW | Mid-2030s |
| Holtec SMR-300 (×2) | MISO (Palisades, MI) | 600 MW | Mid-2030s |
| Meta uprates (Perry, Davis-Besse, Beaver Valley) | PJM | ~500 MW uprate | Near-term |

**Outside our model boundary (~7+ GW):**
| Project | Territory | Capacity | Timeline |
|---------|-----------|----------|----------|
| TVA BWRX-300 | Non-ISO (TVA, TN) | 300 MW+ | 2030s |
| ENTRA1/NuScale | Non-ISO (TVA 7-state region) | Up to 6 GW | 2030s-2040s |
| Kairos Hermes 2 | Non-ISO (TVA, TN) | ~100 MW | Late 2020s |
| TerraPower Natrium | Non-ISO (Wyoming) | 345 MW | ~2030 |
| TerraPower/Meta (×8) | TBD | ~2.8 GW | 2032+ |

**Key insight**: TVA territory (non-ISO Southeast) is the epicenter of US SMR deployment. The ENTRA1/NuScale 6 GW pipeline alone exceeds total in-model new nuclear. This learning spillover — TVA builds drive costs down for ERCOT/MISO/PJM — is exactly why background learning matters. Even if our modeled ISOs are slow to build nuclear, the TVA pipeline provides a substantial learning tailwind.

**ERCOT dominates in-model nuclear**: Fermi's 4× AP1000 project (4.4 GW) is the single largest new nuclear project in any ISO. Texas's pro-nuclear policy environment (TANEO, streamlined siting) makes ERCOT the likely first-mover within our model boundary.

*Sources: [DOE SMR Selections](https://www.energy.gov/articles/energy-department-selects-tva-and-holtec-advance-deployment-us-small-modular-reactors), [NRC Advanced Reactor Highlights 2025](https://www.nrc.gov/reactors/new-reactors/advanced/highlights/2025), [World Nuclear Association — USA](https://world-nuclear.org/information-library/country-profiles/countries-t-z/usa-nuclear-power), [Spencer Fane Nuclear Overview](https://www.spencerfane.com/insight/nuclear-power-in-the-u-s-part-2-technologies-and-projects-that-are-shaping-the-industry/)*

### Wright's Law Formula

Classic experience curve, not time-based:

```python
cost(cumulative_gw) = FOAK × (cumulative_gw / reference_gw) ^ (-learning_exponent)
# Capped at NOAK floor: cost = max(NOAK, computed_cost)
```

**Learning rates** (cost reduction per doubling of cumulative capacity):

| Technology | Learning Rate | Source | Implied Exponent |
|-----------|--------------|--------|-----------------|
| **New nuclear** | 10-15% | Conservative — historical nuclear was ~0%, but SMR modular manufacturing should unlock learning. NuScale/X-energy/Kairos projections. | 0.15-0.23 |
| **CCS-CCGT** | 10-12% | GCCSI literature. Carbon capture is the learning component; CCGT is mature. | 0.15-0.18 |
| **LDES (iron-air)** | 15-20% | Analogous to early battery learning. Manufacturing-dominated cost. | 0.23-0.32 |
| **Green H2** | 12-18% | Electrolyzer learning. IRENA projections. | 0.18-0.29 |
| **Geothermal (EGS)** | 15-20% | Drilling cost learning. Fervo/Quaise projections. | 0.23-0.32 |
| **Battery** | 18-20% | Well-established. BloombergNEF, NREL. Already reflected in shallow cost decline from current levels. | 0.29-0.32 |

`learning_exponent = -log2(1 - learning_rate)`. E.g., 15% learning rate → exponent = 0.234.

**NOAK floor**: Each technology's Low LCOE from existing tables = the NOAK floor. Learning can't reduce cost below the most optimistic long-run estimate.

### Supersedes Time-Based Learning

This endogenous formulation **supersedes** the time-based `learning_fraction(year, foak_start, noak_year)` approach used in Step 3's demand growth sweep and the procurement strategy comparison. Those used calendar-year-based concave ramps. SMARTargets uses deployment-based Wright's Law — costs fall because capacity was built, not because the calendar advanced.

The existing Step 3 time-based curves remain valid for their original purpose (SBTi milestone pricing under assumed deployment schedules). SMARTargets just uses a different, more fundamental mechanism.

## Scenario Matrix

Each scenario represents a different set of **market conditions** — not a different target. The question each scenario answers: *"How clean does the grid get when the market looks like this?"*

### Scenario Condition Definitions

Before the matrix, here's what each condition column means:

| Condition | Facilitating | Challenging |
|-----------|-------------|-------------|
| **Demand growth** | Medium (~1-1.5%/yr) | High (~2-3%/yr, AI + electrification) |
| **Interconnection queues** | Reformed (~5-8 GW/yr/ISO) — FERC Order 2023 reforms, cluster studies, faster processing | Constrained (~2-4 GW/yr/ISO) — status quo, ~80% dropout, 5-yr avg wait |
| **Learning curves** | Fast — high learning rate per doubling, strong manufacturing scale-up | Slow — low learning rate, supply chain bottlenecks, regulatory friction |
| **Starting LCOE** | Low — optimistic 2025 cost starting point across all clean resources | High — pessimistic 2025 costs, supply chain premiums, tariff exposure |
| **New gas buildability** | Moderate friction — siting opposition, ESG lending constraints slow new gas | Low friction — gas builds freely where profitable |

| Condition | Fuller Technology Options | More Limited Technology Options |
|-----------|--------------------------|-------------------------------|
| **DAC availability** | Available at scenario cost (L/M/H) — grid can lean on DAC at crossover | Not available / very expensive ($600+/ton) — grid must decarbonize further before offsets |
| **CCS viability** | CCS-CCGT viable, 45Q intact, CO₂ transport infrastructure builds out | CCS limited — storage site constraints, public opposition, 45Q expires |
| **Storage breadth** | Full suite: 4hr/8hr Li-ion, LDES (iron-air), Green H₂ | Battery only (4hr/8hr). LDES/H₂ remain pre-commercial |
| **Nuclear pathway** | SMR/advanced reactors reach commercial scale. NRC streamlines licensing | Nuclear stalls — cost overruns, regulatory delays, public opposition |

### Full Scenario Matrix

| ID | Target Type | Decarbonization Incentive | Economy-Wide NZ? | Decarb Technology Options | Conditions | Carbon Price | Policy Regime |
|----|------------|--------------------------|-------------------|--------------------------|------------|-------------|---------------|
| **R1** | Reference | No CO₂ constraint | No | N/A | Facilitating | $0 | Current (45Y/45U/45Q + REC) |
| **R2** | Reference | No CO₂ constraint | No | N/A | Challenging | $0 | Current (45Q intact) |
| **S1** | Aspirational (AT) | −57/−82/−88/−94/−100% trajectory (2023 baseline) | No | Fuller | Facilitating | Yes (shadow price emerges from constraint) | Current + emission cap mechanism |
| **S2** | Aspirational (AT) | −57/−82/−88/−94/−100% trajectory (2023 baseline) | No | More Limited | Challenging | Yes (same mechanism) | Current + emission cap mechanism |
| **S3** | Aspirational (AT) | −57/−82/−88/−94/−100% trajectory (2023 baseline) | Yes | Fuller | Facilitating | Yes | Current + economy-wide emission caps |
| **S4** | Aspirational (AT) | −57/−82/−88/−94/−100% trajectory (2023 baseline) | Yes | More Limited | Challenging | Yes | Current + economy-wide emission caps |
| **Q1** | Qualified (QT) | No new constraint (emergent from market) | No | Fuller | Facilitating | $0 | Current (same as R1) |
| **Q2** | Qualified (QT) | No new constraint (emergent from market) | No | More Limited | Challenging | $0 | Current (same as R2) |

### Detailed Scenario Conditions

#### R1 — Reference: Facilitating (No Carbon Constraint)
*"Everything goes right, but no one's pushing for decarbonization."*

- **Demand**: Medium growth (~1-1.5%/yr)
- **Interconnection**: Reformed queues (~5-8 GW/yr/ISO)
- **Learning**: Fast (high rates per doubling)
- **Starting LCOE**: Low across all clean resources
- **Policy**: Current — 45Y (new-build PTC), 45U (existing nuclear through 2032), 45Q (CCS), REC revenue
- **Carbon price**: $0
- **New gas**: Moderate friction (ESG lending, siting opposition slow new gas)
- **Key question**: How clean does the grid get on economics alone when the wind is at your back?

#### R2 — Reference: Challenging (No Carbon Constraint)
*"Everything's an uphill battle, and no one's pushing either."*

- **Demand**: High growth (~2-3%/yr — AI, data centers, electrification pressure)
- **Interconnection**: Constrained queues (~2-4 GW/yr/ISO, status quo)
- **Learning**: Slow (low rates, bottlenecks)
- **Starting LCOE**: High across clean resources (tariffs, supply chain premiums)
- **Policy**: Current (45Q stays intact — CCS still gets its credit)
- **Carbon price**: $0
- **New gas**: Low friction (gas builds freely where profitable)
- **Key question**: Does high demand + slow clean deployment = extended fossil dominance?

#### S1 — Aspirational: Emission-Capped Trajectory, Fuller Options, Facilitating
*"Full commitment to the AT trajectory with every tool available."*

- **Emission constraint**: −57% by 2030, −82% by 2035, −88% by 2040, −94% by 2045, −100% by 2050 (absolute CO₂ metric tons vs 2023 eGRID baseline)
- **Demand**: Medium growth (~1-1.5%/yr)
- **Interconnection**: Reformed queues
- **Learning**: Fast
- **Starting LCOE**: Low
- **DAC**: Available — leveraged at the crossover point (from optimal target exercise). Grid pushes clean until marginal MAC > DAC cost, then DAC handles residual.
- **CCS**: Viable, 45Q intact, CO₂ infrastructure builds out
- **Nuclear**: SMR/advanced pathway open
- **Shadow carbon price**: Emerges from the constrained optimization — the marginal cost of the last unit of abatement at each milestone IS the implied carbon price needed to make the market hit the target.
- **Key question**: With full toolkit and tailwinds, what's the least-cost path to the AT trajectory? What shadow carbon price does each milestone imply?

#### S2 — Aspirational: Emission-Capped Trajectory, Limited Options, Challenging
*"Same trajectory mandate, but the toolkit is constrained and the headwinds are real."*

- **Emission constraint**: Same −57/−82/−88/−94/−100% trajectory as S1
- **Demand**: High growth (~2-3%/yr) — makes absolute caps harder to meet (more demand, same cap)
- **Interconnection**: Constrained queues (status quo)
- **Learning**: Slow
- **Starting LCOE**: High
- **DAC**: Not available (or prohibitively expensive, $600+/ton) — grid must decarbonize deeper because offsets aren't an option
- **CCS**: Limited — storage site constraints, pipeline opposition, 45Q at risk
- **Nuclear**: Stalls — cost overruns, NRC delays
- **Shadow carbon price**: Higher than S1 — headwinds mean each milestone costs more per ton
- **Key question**: How much more expensive is the AT trajectory when you can't lean on DAC, nuclear stalls, and deployment is slow? The shadow carbon price gap between S1 and S2 IS the cost of adverse conditions.

#### S3 — Aspirational: Economy-Wide NZ, Fuller Options, Facilitating
*"The whole economy goes net-zero. The grid is the backbone."*

**Same AT trajectory** (−57/−82/−88/−94/−100%) for the power sector. The additional difficulty comes from three mechanisms, not a stricter cap:

1. **Demand growth is mandatory and higher** (~2-3%/yr even in "facilitating" conditions) — economy-wide NZ requires electrifying transport (EVs), buildings (heat pumps), and industry (electric furnaces). This isn't optional growth — it's policy-mandated electrification. Both S3 and S4 use high demand regardless of facilitating/challenging. Higher demand + same absolute cap = higher required clean %.

2. **DAC budget is shared across the whole economy** — In S1, the grid uses DAC for its residual emissions and that's it. In S3, industry (steel, cement), transport (aviation, shipping), and agriculture also need DAC/removals for *their* residual emissions. The grid's "share" of available DAC capacity shrinks. Effect: the grid may need to push to a *higher* clean % before the portfolio stop binds, because the DAC that would otherwise cover grid residuals is allocated to harder-to-abate sectors.

3. **Green H₂ production load** — Economy-wide NZ requires green hydrogen for steel, chemicals, shipping fuel. H₂ electrolysis consumes clean electricity. This adds ~20-30% electricity demand on top of direct electrification. The grid has to be bigger AND cleaner.

- **Emission constraint**: Same −57/−82/−88/−94/−100% trajectory
- **Demand**: High (~2-3%/yr — mandatory electrification even under facilitating conditions)
- **Interconnection**: Reformed queues
- **Learning**: Fast (economy-wide deployment accelerates learning across sectors)
- **Starting LCOE**: Low
- **DAC**: Available, but shared — grid gets a fraction of total DAC capacity
- **H₂ load**: Significant — electrolysis demand on top of direct load
- **Carbon price**: Economy-wide carbon price (shadow price likely higher than S1 due to demand)
- **Key question**: When the grid has to power everything AND compete for DAC, how much more does each milestone cost vs S1?

#### S4 — Aspirational: Economy-Wide NZ, Limited Options, Challenging
*"The hardest scenario. Everything electrifies, nothing's easy."*

Same AT trajectory (−57/−82/−88/−94/−100%) and economy-wide mechanisms as S3, but under challenging conditions:

- **Emission constraint**: Same −57/−82/−88/−94/−100% trajectory
- **Demand**: Very high (~3-4%/yr — aggressive electrification + AI + H₂ under constrained grid)
- **Interconnection**: Constrained queues (the worst combo — massive demand + can't build fast enough)
- **Learning**: Slow
- **Starting LCOE**: High
- **DAC**: Not available — grid must go nearly all the way without offsets
- **H₂ load**: Same or higher (no alternative to green H₂ when economy-wide NZ is mandated)
- **CCS/Nuclear**: Limited
- **Key question**: Is the trajectory even feasible? At what milestone does cost become prohibitive? The shadow carbon price at each milestone reveals where the constraint binds hardest.

#### Q1 — Qualified Target: Best Achievable Without New Enabling Conditions, Facilitating
*"What can we credibly commit to today, without waiting for new policy?"*

**What QTs actually mean (per SMARTargets methodology):**
- QTs represent the **best a company can achieve given current/realistic conditions WITHOUT new enabling conditions** (no new carbon price, no new CES, no new mandates beyond what exists today)
- QTs use the **same policy environment as R1/R2** — current policy only. No additional forcing mechanism.
- The QT level is determined by the model: run the market simulation under current-policy conditions and find the maximum clean level where deployment is still economically rational (profitable or cost-effective to commit to)
- The **gap between QT and AT** is the key output — it defines the "enabling conditions gap" the company must report: "We can commit to X% (QT). To reach NZ (AT), we would need: interconnection reform, CCS viability, carbon pricing, nuclear pathway, etc."
- **QT emissions reduction ranges are tested at 5% intervals** (e.g., 50%, 55%, 60%, ..., 95%) to identify the precise crossover point where further reductions become uneconomic.

**QT Determination for Vertically Integrated Utilities:**
1. Run market simulation under Q1 (facilitating) and Q2 (challenging) conditions
2. The model produces an emergent clean level per ISO — same as R1/R2, but now the company asks: "Can I credibly commit to this level?"
3. The QT is typically at or slightly below the R1/R2 market equilibrium — it's what the market would do anyway, formalized as a corporate commitment
4. If R1 produces 65% clean and R2 produces 45% clean, the QT range is 45-65% depending on risk appetite
5. The company balances emissions reductions against **consumer rates** — the QT is the strictest target achievable without unacceptable rate increases
6. The company then reports: "Our QT is X%. To reach the AT (NZ by 2050), we need [specific enabling conditions]."

**QT Determination for Independent Power Producers (IPPs):**

For an IPP, the logical premise shifts from system-wide ratepayer impact to **competitive market survival**:

1. **Company-level baseline**: Use 2023 Scope 1 emissions from sustainability reports / CDP disclosures (NOT eGRID plant mapping). For Constellation: pro-forma = Constellation 2023 + Calpine 2023 − acquisition divestitures. Fleet-to-ISO mapping from 10-K filings, S&P Global, and company investor relations.
2. **Regional view**: Because an IPP lacks a captive rate base, forcing the model to hit a strict emissions constraint simulates the entire regional wholesale market's least-cost response, not just the utility's internal system.
3. **Testing market viability**: The model outputs how wholesale electricity prices, new capacity needs, and retirement schedules must shift across the grid to meet that forced target.
4. **Scaling to the portfolio**: The IPP scales these regional, macro-level market shifts down to evaluate the viability of its own specific assets.
5. **Setting the target**: The IPP's QT becomes the strictest reduction threshold their specific fleet can align with while remaining profitable and competitive in that future wholesale market.

Instead of balancing reductions against consumer rates, the IPP balances reductions against **asset stranding and merchant risk**. The QT is the point where further reductions would strand existing assets or make the portfolio uncompetitive.

**QT-to-AT gap reporting**: The IPP's QT (e.g., −40% by 2030) vs the AT trajectory (−57% by 2030) defines the enabling conditions gap. The company reports: "We can commit to −40%. To reach −57%, we need: [carbon pricing at $X/ton, interconnection reform, CCS viability, etc.]" The shadow carbon price from S1/S2 quantifies exactly what policy signal is needed to close the gap.

- **Demand**: Medium growth (same as R1)
- **Interconnection**: Reformed queues (same as R1)
- **Learning**: Fast (same as R1)
- **Starting LCOE**: Low (same as R1)
- **Carbon price**: $0 (same as R1 — no new forcing mechanism)
- **Policy**: Current only (45Y/45U/45Q + REC revenue — same as R1)
- **All technology options available** but no policy mandate to deploy them
- **Key question**: Under best-case market conditions with NO new policy, what's the highest clean level a company can credibly commit to? This IS the QT under facilitating conditions.

#### Q2 — Qualified Target: Best Achievable Without New Enabling Conditions, Challenging
*"What can we commit to when the market works against us?"*

- **Demand**: High growth (same as R2)
- **Interconnection**: Constrained queues (same as R2)
- **Learning**: Slow (same as R2)
- **Starting LCOE**: High (same as R2)
- **Carbon price**: $0 (same as R2)
- **Policy**: Current only (same as R2)
- **Technology**: Limited (same technology constraints as R2 challenging conditions — nuclear stalls, LDES pre-commercial, CCS limited)
- **Key question**: Under worst-case market conditions with NO new policy, what's the best a company can commit to? This is the QT floor — the minimum credible commitment even when everything's hard.

**The QT range (Q1-Q2) vs the AT-QT gap:**
- **Q1 result** = upper bound of credible commitment (best conditions, no new policy)
- **Q2 result** = lower bound of credible commitment (worst conditions, no new policy)
- **AT (S1/S2)** = what the company WOULD achieve if NZ policy existed
- **AT minus QT** = the enabling conditions gap. This gap is what the company reports alongside its target: "We need X, Y, Z conditions to close this gap."

### Scenario Comparison: Key Differentiators

| Dimension | R1/R2 | S1/S2 | S3/S4 | Q1/Q2 |
|-----------|-------|-------|-------|-------|
| **CO₂ target** | None (emergent) | −57/−82/−88/−94/−100% caps (2023 baseline) | Same caps (within economy-wide NZ) | None (emergent — same as R) |
| **Carbon price** | $0 | Yes | Yes (economy-wide, likely higher) | $0 (same as R) |
| **Demand driver** | Market-driven | Market-driven | **Mandatory electrification** | Market-driven |
| **DAC role** | N/A | Grid residuals only | **Shared with industry/transport** | Minimal (MAC < DAC) |
| **H₂ load** | Negligible | Small | **Large (industry demand)** | Negligible |
| **Gas end-state** | Whatever's profitable | Near-zero | Near-zero | Whatever's profitable (same as R) |
| **Last-mile resources** | Only if profitable | Required (LDES, nuclear) | Required + more of them | Only if profitable (same as R) |
| **IPP relevance** | What the market does | Long-term strategic | Strategic + electrification | **What we can commit to today** |

### Interconnection Queue Constraint (Decided — Bundled with Facilitating/Challenging)

Interconnection queue friction is now **bundled into the Facilitating vs Challenging condition axis** rather than being a separate scenario dimension. This keeps the scenario count manageable while capturing the key dynamic:

- **Facilitating** = Reformed queues (~5-8 GW/yr/ISO). FERC Order 2023 reforms, cluster studies, faster processing.
- **Challenging** = Constrained queues (~2-4 GW/yr/ISO). Status quo — ~80% dropout, 5-year avg wait (LBNL data).

**Modeling approach**:
```
For each ISO, each simulation step:
  desired_new_build_gw = sum of all profitable projects
  actual_new_build_gw = min(desired_new_build_gw, max_annual_gw[iso])
  # Queue prioritization: highest-profit-margin projects connect first
```

This means every Challenging scenario (R2, S2, S4, Q2) faces queue constraints while every Facilitating scenario (R1, S1, S3, Q1) benefits from reform. The queue constraint compounds with high demand and slow learning to make Challenging scenarios meaningfully harder.

## Compute Budget

With 15-20 scenarios × 17 thresholds × 7 ISOs:
- **Mix filtering + cost eval**: ~2,400 calls to existing vectorized batch functions. Seconds.
- **LMP computation**: ~2,400 calls to `build_merit_order_stack` + price model. Each is <1ms. Seconds total.
- **Hourly revenue**: 2,400 dot products of (8760,) arrays. Milliseconds each.
- **Total**: Well under 60 seconds for the full sweep. No CI/CD needed — runs locally.

## Output

TBD — likely:
- `data/step5-post-processing/smartargets/` — per-scenario deployment trajectories
- `dashboard/js/smartargets-data.js` — pre-computed for dashboard
- Dashboard page: `dashboard/smartargets.html`

Each scenario result contains, per ISO × threshold:
- Optimal resource mix (TWh by resource)
- LCOE breakdown by resource
- Revenue breakdown (energy, capacity, ancillary, carbon)
- Profit per MWh
- Cumulative GW deployed (for learning curve tracking)
- LMP distribution stats (P10/P50/P90, scarcity hours)

## Dependencies

```
Step 3 (cost-opt parquets) ──┐
Step 4 (dispatch cache) ─────┤
Step 5d (zone metrics) ──────├── SMARTargets module ──→ smartargets output
pipeline_config.py ──────────┤
step5b LMP engine ───────────┘
```

No upstream changes needed. All inputs already exist.

## Data Tables (From pipeline_config.py — Single Source of Truth)

These values already exist in `scripts/pipeline_config.py` and are referenced here for completeness. A fresh session building SMARTargets should import from `pipeline_config`, not hardcode.

### Regional Demand (2025 Baseline, TWh)

| ISO | Demand (TWh) | Source |
|-----|-------------|--------|
| CAISO | 224.0 | EIA-930, 2024 annualized + 2025 growth |
| ERCOT | 488.0 | EIA-930 |
| PJM | 843.3 | EIA-930 |
| NYISO | 151.6 | EIA-930 |
| NEISO | 115.3 | EIA-930 |
| MISO | 660.0 | EIA-930 |
| SPP | 296.0 | EIA-930 |

**Source**: `pipeline_config.py::REGIONAL_DEMAND_TWH` (line 96)

### Demand Growth Rates (Annual %)

| ISO | Low | Medium | High |
|-----|-----|--------|------|
| CAISO | 1.4% | 1.9% | 2.5% |
| ERCOT | 2.0% | 3.5% | 5.5% |
| PJM | 1.5% | 2.4% | 3.6% |
| NYISO | 1.3% | 2.0% | 4.4% |
| NEISO | 0.9% | 1.8% | 2.9% |
| MISO | 1.2% | 2.2% | 3.8% |
| SPP | 1.0% | 1.8% | 3.0% |

**Scenario mapping**: R1/S1/Q1 (Facilitating) = Medium growth. R2/S2/Q2 (Challenging) = High growth. S3 (Economy-wide, Facilitating) = High (mandatory electrification). S4 (Economy-wide, Challenging) = High+ (~3-4%/yr).

**Source**: `pipeline_config.py::DEMAND_GROWTH_RATES` (line 1005)

### Capacity Market Prices ($/kW-yr)

| ISO | Price | Notes |
|-----|-------|-------|
| CAISO | $75 | RA program, system-wide avg |
| ERCOT | **$0** | Energy-only market |
| PJM | $120 | RPM BRA clearing |
| NYISO | $85 | ICAP monthly spot, annualized |
| NEISO | $55 | FCM FCA-19 clearing |
| MISO | $25 | PRA Zone 1-7 avg |
| SPP | **$0** | Energy-only market |

**Source**: `pipeline_config.py::CAPACITY_MARKET_PRICES` (line 240)

### Peak Capacity Credits (ELCC, % of Nameplate)

| Resource | Credit | Notes |
|----------|--------|-------|
| Clean firm (nuclear) | 100% | Baseload, always available |
| Solar | 30% | Daytime only, no evening peak |
| Wind (onshore) | 10% | Intermittent, low correlation with peak |
| Offshore wind | 25% | More consistent than onshore |
| CCS-CCGT | 90% | Near-baseload with some capture downtime |
| Hydro | 50% | Seasonal/constrained |
| Battery (4hr/8hr) | 95% | Dispatchable during peak |
| LDES (iron-air) | 90% | Long-duration dispatch |
| Green H₂ | 85% | Dispatchable |

**Source**: `pipeline_config.py::PEAK_CAPACITY_CREDITS` (line 411)

### Fixed O&M Costs ($/kW-yr) — For Retirement Economics

| Resource | Fixed O&M ($/kW-yr) | Source | Notes |
|----------|-------------------:|--------|-------|
| **Existing nuclear** | ~$30/MWh (~$26/kW-yr at 93% CF) | NREL ATB 2024 | Key retirement threshold — needs revenue > FOM to stay online |
| **Existing gas CCGT** | $13-17/kW-yr (varies by ISO) | `pipeline_config.py` line 866 | CAISO $16, ERCOT $13, PJM $14, NYISO $17, NEISO $15, MISO $14, SPP $13 |
| **Existing coal** | $40-50/kW-yr | EIA AEO 2023 | High FOM → first to retire when revenue drops |
| **Solar/wind** | $8-15/kW-yr | NREL ATB 2024 | Near-zero marginal cost → effectively never retire |
| **Hydro** | $15-20/kW-yr | NREL ATB 2024 | Low FOM → effectively never retire |

**Retirement criterion**: `annual_revenue < fixed_O&M × capacity_kw` → unit retires.

**Note**: Nuclear FOM is the critical value — it determines the LMP threshold below which existing nuclear retires. At $26/kW-yr and 93% CF, nuclear needs ~$30/MWh average revenue. With 45U credit ($15/MWh through 2032), the effective threshold drops to ~$15/MWh. Post-2032, the full $30/MWh threshold applies.

### Wholesale Prices ($/MWh, 2024 Weighted Avg DA LMP)

| ISO | Price |
|-----|------:|
| CAISO | $30 |
| ERCOT | $27 |
| PJM | $34 |
| NYISO | $42 |
| NEISO | $41 |
| MISO | $30 |
| SPP | $25 |

**Source**: `pipeline_config.py::WHOLESALE_PRICES` (line 127)

## Emissions Fan Output Specification

### What the Dashboard Shows

The **emissions fan** visualizes how the 8 scenarios produce different emission trajectories over 2025-2050, creating a fan-shaped uncertainty range:

```
CO₂ Emissions (M metric tons)
│
│  ████████ ← 2023 Baseline (100%)
│  ████████
│   ██████  ← R2 (worst case — high demand, no constraint)
│    █████  ← R1 (market only — facilitating)
│     ████  ← Q2 (qualified target floor)
│      ███  ← Q1 (qualified target ceiling)
│       ██  ← S2/S4 (AT trajectory — challenging)
│        █  ← S1/S3 (AT trajectory — facilitating)
│        ·  ← 0 (100% reduction by 2050)
├───┬───┬───┬───┬───┬───→ Year
  2023 2030 2035 2040 2045 2050
```

### Per-ISO Output Data Structure

```json
{
  "iso": "ERCOT",
  "baseline_2023_mt": 162500000,
  "scenarios": {
    "R1": {
      "trajectory": {
        "2025": {"emissions_mt": 158000000, "clean_pct": 48.5, "cost_per_mwh": 42.1},
        "2030": {"emissions_mt": 120000000, "clean_pct": 58.3, "cost_per_mwh": 39.8},
        "2035": {"emissions_mt": 95000000, "clean_pct": 65.1, "cost_per_mwh": 38.2},
        "2040": {"emissions_mt": 78000000, "clean_pct": 70.4, "cost_per_mwh": 37.5},
        "2045": {"emissions_mt": 65000000, "clean_pct": 74.2, "cost_per_mwh": 37.1},
        "2050": {"emissions_mt": 55000000, "clean_pct": 77.8, "cost_per_mwh": 36.8}
      },
      "resource_mix_2050": {"solar_twh": 180, "wind_twh": 150, "nuclear_twh": 85, ...},
      "market_stop_pct": 77.8,
      "shadow_carbon_price": null
    },
    "S1": {
      "trajectory": {
        "2025": {"emissions_mt": 158000000, "clean_pct": 48.5, "cost_per_mwh": 42.1},
        "2030": {"emissions_mt": 69875000, "clean_pct": 72.1, "cost_per_mwh": 48.3, "cap_mt": 69875000},
        "2035": {"emissions_mt": 29250000, "clean_pct": 89.5, "cost_per_mwh": 55.7, "cap_mt": 29250000},
        "2040": {"emissions_mt": 19500000, "clean_pct": 93.8, "cost_per_mwh": 52.1, "cap_mt": 19500000},
        "2045": {"emissions_mt": 9750000, "clean_pct": 97.1, "cost_per_mwh": 58.4, "cap_mt": 9750000},
        "2050": {"emissions_mt": 0, "clean_pct": 100.0, "cost_per_mwh": 65.2, "cap_mt": 0}
      },
      "resource_mix_2050": {"solar_twh": 250, "wind_twh": 200, "nuclear_twh": 120, ...},
      "shadow_carbon_price": {"2030": 45, "2035": 85, "2040": 110, "2045": 180, "2050": 320}
    }
  }
}
```

### Dashboard Visualization Components

1. **Emissions Fan Chart** (primary) — All 8 scenarios × 7 ISOs, with AT trajectory milestones as constraint markers. Shaded bands between S1-S2 and Q1-Q2.
2. **Shadow Carbon Price Chart** — S1-S4 implied carbon price at each milestone. Shows policy cost gap.
3. **Resource Mix Evolution** — Stacked area showing resource mix change over 2025-2050 per scenario.
4. **QT-AT Gap Table** — Per-ISO: QT range (Q1-Q2), AT trajectory, enabling conditions needed.
5. **Cost Comparison** — $/MWh cost at each milestone across scenarios. S2 vs S1 spread = cost of adverse conditions.

## Open Questions

1. **Scenario axis definitions** — what are the 15-20 scenarios? What story should the SMARTargets dashboard page tell? R1/R2 reference case is designed. R3+ scenarios (carbon price, policy, learning) still TBD.
2. **Cross-regional capital flow** — V1 is per-ISO. Should V2 allow developers to deploy wherever profit is highest across ISOs? (Global merit order by profit, not MAC.)
3. **Demand growth rate for R2** — what growth trajectory? EIA AEO reference case (~1%/yr)? AI/electrification-driven (~2-3%/yr)? Multiple rates as sub-scenarios?
4. **Within-threshold cannibalization** — the LMP engine captures *between-threshold* cannibalization (each step sees updated fossil stack). But within a single threshold, adding 10 GW of solar suppresses solar-hour prices more than adding 1 GW. Should we model this, or is the 5% step size granular enough?
5. **PPA vs merchant revenue** — current design assumes merchant (spot LMP). Real developers sign PPAs at a discount to expected spot. Does this matter for the simulation, or is merchant a reasonable proxy?
6. ~~**Existing asset economics**~~ — **RESOLVED**: Yes, model existing clean viability. Retire if revenue < fixed O&M. Wind/solar effectively never retire (near-zero O&M). Nuclear is the key risk case.
7. **Stopping rule nuance** — **PARTIALLY RESOLVED**: RPS mandates don't force builds directly — REC revenue makes resources more profitable, which may trigger builds via the standard profit criterion. ACP payments (penalty < REC price) mean some entities pay the compliance penalty rather than build, so RPS doesn't guarantee full compliance. Federal credits (45Y/45U/45Q) are modeled as LCOE reductions, not mandates. Remaining question: should there be a "policy mandate" scenario axis where RPS IS a hard floor (forcing builds even at a loss)?
8. **What is the relationship between SMARTargets and the existing SBTi/procurement strategy pages?** SMARTargets answers "what would the market do" — the existing pages answer "what should a buyer do." How do they connect on the dashboard?
9. ~~**Fixed O&M data source**~~ — **RESOLVED**: See Data Tables section above. Gas CCGT FOM from `pipeline_config.py` (varies by ISO, $13-17/kW-yr). Nuclear ~$26/kW-yr (NREL ATB 2024). Coal $40-50/kW-yr (EIA AEO). Solar/wind/hydro near-zero — effectively never retire.
10. **New gas build constraints** — are there siting/permitting constraints on new gas, or do we assume unlimited new gas can be built if profitable? (Affects how quickly gas fills demand growth in R2.)
