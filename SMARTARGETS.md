# SMARTargets Module — Market Simulation of Clean Energy Deployment

> **Status**: Design phase — not yet implemented.
> **Last updated**: 2026-03-06.

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
| **S1** | Aspirational (AT) | Power sector NZ CO₂ by 2050 | No | Fuller | Facilitating | Yes (carbon price or equivalent forcing) | Current + carbon pricing mechanism |
| **S2** | Aspirational (AT) | Power sector NZ CO₂ by 2050 | No | More Limited | Challenging | Yes (same mechanism) | Current + carbon pricing mechanism |
| **S3** | Aspirational (AT) | Power sector NZ CO₂ by 2050 | Yes | Fuller | Facilitating | Yes | Current + economy-wide carbon pricing |
| **S4** | Aspirational (AT) | Power sector NZ CO₂ by 2050 | Yes | More Limited | Challenging | Yes | Current + economy-wide carbon pricing |
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

#### S1 — Aspirational: Power Sector NZ, Fuller Options, Facilitating
*"Full commitment to power sector net-zero with every tool available."*

- **Demand**: Medium growth (~1-1.5%/yr)
- **Interconnection**: Reformed queues
- **Learning**: Fast
- **Starting LCOE**: Low
- **DAC**: Available — leveraged at the crossover point (from optimal target exercise). Grid pushes clean until marginal MAC > DAC cost, then DAC handles residual.
- **CCS**: Viable, 45Q intact, CO₂ infrastructure builds out
- **Nuclear**: SMR/advanced pathway open
- **Carbon price or equivalent forcing**: The mechanism that makes NZ happen. Could be carbon price ($50-100 range), cap-and-trade, or CES mandate. Effect: makes clean deployment more profitable AND fossil generation more expensive.
- **Key question**: With full toolkit and tailwinds, what does the optimal NZ grid look like? Where does the DAC crossover land?

#### S2 — Aspirational: Power Sector NZ, Limited Options, Challenging
*"Same NZ mandate, but the toolkit is constrained and the headwinds are real."*

- **Demand**: High growth (~2-3%/yr)
- **Interconnection**: Constrained queues (status quo)
- **Learning**: Slow
- **Starting LCOE**: High
- **DAC**: Not available (or prohibitively expensive, $600+/ton) — grid must decarbonize deeper because offsets aren't an option
- **CCS**: Limited — storage site constraints, pipeline opposition, 45Q at risk
- **Nuclear**: Stalls — cost overruns, NRC delays
- **Carbon price or equivalent forcing**: Same mechanism as S1, but biting harder because it has to push through headwinds
- **Key question**: How much more expensive is NZ when you can't lean on DAC, nuclear stalls, and deployment is slow? This is the "hard mode" scenario — NZ is required but the path is brutal.

#### S3 — Aspirational: Economy-Wide NZ, Fuller Options, Facilitating
*"The whole economy goes net-zero. The grid is the backbone."*

**How S3 differs from S1 — three mechanisms, not just more demand:**

1. **Demand growth is mandatory and higher** (~2-3%/yr even in "facilitating" conditions) — economy-wide NZ requires electrifying transport (EVs), buildings (heat pumps), and industry (electric furnaces). This isn't optional growth — it's policy-mandated electrification. Both S3 and S4 use high demand regardless of facilitating/challenging.

2. **DAC budget is shared across the whole economy** — In S1, the grid uses DAC for its residual emissions and that's it. In S3, industry (steel, cement), transport (aviation, shipping), and agriculture also need DAC/removals for *their* residual emissions. The grid's "share" of available DAC capacity shrinks. Effect: the grid may need to push to a *higher* clean % before the portfolio stop binds, because the DAC that would otherwise cover grid residuals is allocated to harder-to-abate sectors.

3. **Green H₂ production load** — Economy-wide NZ requires green hydrogen for steel, chemicals, shipping fuel. H₂ electrolysis consumes clean electricity. This adds ~20-30% electricity demand on top of direct electrification. The grid has to be bigger AND cleaner.

- **Demand**: High (~2-3%/yr — mandatory electrification even under facilitating conditions)
- **Interconnection**: Reformed queues
- **Learning**: Fast (economy-wide deployment accelerates learning across sectors)
- **Starting LCOE**: Low
- **DAC**: Available, but shared — grid gets a fraction of total DAC capacity
- **H₂ load**: Significant — electrolysis demand on top of direct load
- **Carbon price**: Economy-wide carbon price (higher signal than power-only)
- **Key question**: When the grid has to power everything AND compete for DAC, how much harder is decarbonization?

#### S4 — Aspirational: Economy-Wide NZ, Limited Options, Challenging
*"The hardest scenario. Everything electrifies, nothing's easy."*

Same economy-wide mechanisms as S3, but under challenging conditions:

- **Demand**: Very high (~3-4%/yr — aggressive electrification + AI + H₂ under constrained grid)
- **Interconnection**: Constrained queues (the worst combo — massive demand + can't build fast enough)
- **Learning**: Slow
- **Starting LCOE**: High
- **DAC**: Not available — grid must go nearly all the way without offsets
- **H₂ load**: Same or higher (no alternative to green H₂ when economy-wide NZ is mandated)
- **CCS/Nuclear**: Limited
- **Key question**: Is this even feasible? What's the cost premium vs S3? This scenario tests whether economy-wide NZ is physically achievable under adverse conditions, or if something has to give.

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

1. **Regional view**: Because an IPP lacks a captive rate base, forcing the model to hit a strict emissions constraint simulates the entire regional wholesale market's least-cost response, not just the utility's internal system.
2. **Testing market viability**: The model outputs how wholesale electricity prices, new capacity needs, and retirement schedules must shift across the grid to meet that forced target.
3. **Scaling to the portfolio**: The IPP scales these regional, macro-level market shifts down to evaluate the viability of its own specific assets.
4. **Setting the target**: The IPP's QT becomes the strictest reduction threshold their specific fleet can align with while remaining profitable and competitive in that future wholesale market.

Instead of balancing reductions against consumer rates, the IPP balances reductions against **asset stranding and merchant risk**. The QT is the point where further reductions would strand existing assets or make the portfolio uncompetitive.

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
| **CO₂ target** | None (emergent) | Power sector NZ | Power sector NZ (within economy-wide NZ) | None (emergent — same as R) |
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

## Open Questions

1. **Scenario axis definitions** — what are the 15-20 scenarios? What story should the SMARTargets dashboard page tell? R1/R2 reference case is designed. R3+ scenarios (carbon price, policy, learning) still TBD.
2. **Cross-regional capital flow** — V1 is per-ISO. Should V2 allow developers to deploy wherever profit is highest across ISOs? (Global merit order by profit, not MAC.)
3. **Demand growth rate for R2** — what growth trajectory? EIA AEO reference case (~1%/yr)? AI/electrification-driven (~2-3%/yr)? Multiple rates as sub-scenarios?
4. **Within-threshold cannibalization** — the LMP engine captures *between-threshold* cannibalization (each step sees updated fossil stack). But within a single threshold, adding 10 GW of solar suppresses solar-hour prices more than adding 1 GW. Should we model this, or is the 5% step size granular enough?
5. **PPA vs merchant revenue** — current design assumes merchant (spot LMP). Real developers sign PPAs at a discount to expected spot. Does this matter for the simulation, or is merchant a reasonable proxy?
6. ~~**Existing asset economics**~~ — **RESOLVED**: Yes, model existing clean viability. Retire if revenue < fixed O&M. Wind/solar effectively never retire (near-zero O&M). Nuclear is the key risk case.
7. **Stopping rule nuance** — **PARTIALLY RESOLVED**: RPS mandates don't force builds directly — REC revenue makes resources more profitable, which may trigger builds via the standard profit criterion. ACP payments (penalty < REC price) mean some entities pay the compliance penalty rather than build, so RPS doesn't guarantee full compliance. Federal credits (45Y/45U/45Q) are modeled as LCOE reductions, not mandates. Remaining question: should there be a "policy mandate" scenario axis where RPS IS a hard floor (forcing builds even at a loss)?
8. **What is the relationship between SMARTargets and the existing SBTi/procurement strategy pages?** SMARTargets answers "what would the market do" — the existing pages answer "what should a buyer do." How do they connect on the dashboard?
9. **Fixed O&M data source** — need $/kW-yr fixed O&M by resource type and vintage for retirement calculations. EIA AEO or NREL ATB?
10. **New gas build constraints** — are there siting/permitting constraints on new gas, or do we assume unlimited new gas can be built if profitable? (Affects how quickly gas fills demand growth in R2.)
