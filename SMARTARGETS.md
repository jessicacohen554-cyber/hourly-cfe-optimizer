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

    3. PROFIT: delta_profit = delta_revenue - delta_cost             [NEW]
       - If delta_profit > 0: DEPLOY THIS ZONE
         → Lock in resources, advance to next zone for this ISO
       - If delta_profit ≤ 0: SKIP / STOP
         → This zone is unprofitable under current conditions
         → But check: might LATER zones be profitable?
           (e.g., learning from other ISOs lowers cost, or
            further fossil retirement raises scarcity value)

    4. LEARN (if deployed):                                          [step8d pattern]
       cumulative_gw[resource] += new_capacity_gw
       # Deployment lowers future costs for ALL ISOs (global learning)

    5. PRICE UPDATE (automatic):
       - Next zone's LMP reflects updated fossil stack
       - Scarcity value shifts → changes which resources are profitable
       - Cannibalization shifts → changes VRE revenue

  Terminate when:
    - All remaining zones across all ISOs have delta_profit ≤ 0
    - AND no learning-driven cost reduction could flip any zone positive
    - Record per-ISO clean level as "market equilibrium outcome"
```

**The stopping point IS the result.** Each scenario produces a different market-equilibrium clean level per ISO. The dashboard shows how market conditions determine where deployment stalls or accelerates.

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
- **Nuclear is the interesting case** — fixed O&M ~$30/MWh means it needs decent capacity factors and prices. If LMPs crash from solar oversupply, a nuclear plant could theoretically become uneconomic and retire.
- **Hydro** — existing only (no new-build), very low O&M, effectively never retires.

### Existing Clean Viability
- The model validates that existing 2025 clean resources remain economically viable under reference case pricing.
- If an existing clean resource can't cover fixed O&M, it retires — this is a real risk for nuclear under high-solar scenarios.
- This validation is important: it confirms the "starting fleet" is stable before layering on policy scenarios.

### R1 vs R2

| | R1: Current Market | R2: Current Market + Demand Growth |
|---|---|---|
| **Starting fleet** | 2025 existing (all resources) | 2025 existing (all resources) |
| **Demand** | Flat (2025 levels) | Growth trajectory (TBD rate) |
| **Carbon price** | $0 | $0 |
| **Policy** | Current (existing ITC/PTC, 45Q) | Current |
| **New builds** | Whatever's profitable at market LMPs | More capacity needed → more new builds |
| **Key question** | What does the market do on its own? | Does demand growth extend fossil life or pull in clean? |

### Reference Case Logic (Per Step)
```
For each step in the simulation:
  1. Compute LMPs from current fleet (merit-order dispatch)
  2. For each EXISTING unit:
     - Compute annual revenue (dispatch hours × LMP)
     - If revenue < fixed_O&M → RETIRE
  3. For each potential NEW BUILD (gas CCGT, gas CT, solar, wind, battery, etc.):
     - Compute expected revenue at current LMPs
     - If revenue > annualized LCOE → BUILD (developer would invest)
  4. Update fleet composition (retirements + new builds)
  5. Recompute LMPs with updated fleet → next step
```

## Scenario Axes (TBD — User to Define)

Each scenario represents a different set of **market conditions** — not a different target. The question each scenario answers: *"How clean does the grid get when the market looks like this?"*

R1/R2 are the reference cases (described above). R3+ layer on policy/price interventions:

Candidate axes (not finalized):

| Axis | Options | What It Changes |
|------|---------|-----------------|
| **Carbon price** | $0 (R1/R2) / $50 / $100 / $185 | Raises fossil marginal cost → raises LMP → improves clean revenue. Direct: makes every clean MWh worth `carbon_price × displaced_rate` more. |
| **Learning speed** | Fast / Medium / Slow | How quickly early deployment drives FOAK→NOAK cost reduction. Fast learning = early deployment unlocks cascade. |
| **Starting LCOE** | Low / Medium / High | Where technology costs begin (2025 starting point). Low = some resources already profitable at t=0. |
| **Participation level** | TBD | How many corporate buyers / utilities participate. Affects cumulative GW → learning curve speed. |
| **Policy regime** | TBD | ITC/PTC, 45Q, state RPS mandates. Changes effective cost floor. |
| **Capacity market reform** | TBD | ELCC-based vs flat capacity payments. Affects dispatchable premium. |

**Not yet decided**: Which axes beyond carbon price, how many levels, which combinations. R1/R2 reference case design is locked. R3+ scenario design determines what story the dashboard tells.

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
7. **Stopping rule nuance** — when profit goes negative, is that truly "stop"? Or might developers accept marginal losses if mandated (RPS), subsidized (PTC), or pursuing portfolio strategy? The model as designed is pure-market — no policy mandates forcing unprofitable deployment.
8. **What is the relationship between SMARTargets and the existing SBTi/procurement strategy pages?** SMARTargets answers "what would the market do" — the existing pages answer "what should a buyer do." How do they connect on the dashboard?
9. **Fixed O&M data source** — need $/kW-yr fixed O&M by resource type and vintage for retirement calculations. EIA AEO or NREL ATB?
10. **New gas build constraints** — are there siting/permitting constraints on new gas, or do we assume unlimited new gas can be built if profitable? (Affects how quickly gas fills demand growth in R2.)
