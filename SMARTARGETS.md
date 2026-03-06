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

| Component | Source | What We Use |
|-----------|--------|-------------|
| **Floor ratchet** | `step7a` / `scenario_common.py` | Sequential deployment with per-resource TWh floors. Once deployed, capacity can't be removed. |
| **Sequential threshold stepping** | `step7a::_forward_step_optimization()` | Process thresholds 50% → 99.99% in order, each step constrained by prior. |
| **EF/PFS mix filtering** | `scenario_common.py::batch_filter_floor()` | Vectorized floor-compatible mix selection from efficient frontier. |
| **8,760-hour dispatch** | `dispatch_utils.py` / Step 4 cache | Hourly supply-demand matching for LMP computation. |
| **LMP merit-order engine** | `step5b::build_merit_order_stack()` + `PriceModel` subclasses | Synthetic hourly LMP from fossil stack composition at each threshold. |
| **Scarcity pricing** | `step5b` ISO-specific models | Reserve-based exponential adder (ERCOT ORDC, PJM RPM-calibrated). |
| **Wright's Law learning curves** | `step8d` + `procurement_utils.py` | FOAK→NOAK cost reduction as global cumulative GW deployed increases. |
| **Capacity market revenue** | `pipeline_config.py::CAPACITY_MARKET_PRICES` | $/kW-yr by ISO (zero for energy-only markets: ERCOT, SPP). |
| **Storage revenue stacking** | `pipeline_config.py::compute_storage_revenue_credit()` | Arbitrage + capacity + ancillary with 70% co-optimization efficiency. |
| **Gas backup recomputation** | `scenario_common.py` | RA-aware gas MW recalculation after floor augmentation. |

### What We Do NOT Borrow

| Component | Source | Why Not |
|-----------|--------|---------|
| **MAC optimization** | `step5d`, `step6a` | SMARTargets optimizes profit, not $/tCO2. MAC is irrelevant to the objective function. |
| **Consequential queue ordering** | `step5d::build_consequential_queue()` | Queue ranks by cheapest abatement. We rank by highest profit. |
| **Cross-regional netting** | `step5d` | May add later, but V1 is per-ISO. |

### What's New (~30-50 Lines of Glue Code)

| Component | Description |
|-----------|-------------|
| **Revenue calculator** | For each candidate mix at each threshold: compute hourly LMP revenue from the existing `step5b` engine, add capacity market payments, add storage revenue credits. ~20 lines. |
| **Profit objective** | `profit = revenue_per_mwh - lcoe_per_mwh` for each resource in the mix. Replace `np.argmin(costs)` with `np.argmax(profit)` in the selection step. ~10 lines. |
| **Price feedback loop** | After selecting a mix, recompute LMP for the *new* fossil stack (since clean deployment changes the merit order). Feed updated prices into revenue calc for next threshold step. Natural sequential feedback — no iterative equilibrium needed. |
| **Capacity price degradation** | `capacity_price(t) = base_price × max(0, 1 - α × clean_share(t))`. α calibrated per ISO. ~5 lines. |
| **Carbon price lever** | Expose `co2_level` parameter from existing `compute_marginal_costs()` as a scenario axis. Already implemented in step5b — just needs to be threaded through. |

## Sequential Algorithm

The model steps through increasing clean percentages — but these aren't *targets*. They're measurement points. At each step, the model asks "is it profitable to go from here to the next level?" If yes, deploy. If no, stop — that's the market equilibrium.

```
For each scenario (15-20 market condition combos):
  Initialize:
    floor_twh = 2025 existing clean by resource (from GRID_MIX_SHARES)
    cumulative_gw = {resource: global_installed_2025}  # for learning curves
    current_clean_pct = existing_clean_floor  # ~30-48% depending on ISO

  For each threshold step t (50%, 55%, ... 99.99%):

    1. FILTER: Get feasible mixes at this clean level        [scenario_common.batch_filter_floor]
       - Must meet floor ratchet (can't un-deploy)           [step7a pattern]
       - If EF exhausted, PFS fallback with floor window     [scenario_common._filter_pfs_by_floor_window]

    2. COST: Compute all-in LCOE for each candidate mix      [scenario_common.batch_compute_total_costs]
       - Apply Wright's Law cost reduction from cumulative_gw [procurement_utils.learning_fraction]
       - Global cumulative: all ISO deployments pool together

    3. REVENUE: For each candidate mix, compute market revenue [NEW]
       a. LMP engine on fossil stack at this clean level      [step5b.build_merit_order_stack + PriceModel]
          - Fossil stack naturally shrinks as clean rises
          - Scarcity pricing kicks in when reserves thin
       b. Hourly energy revenue = generation_profile · lmp    [dot product, vectorized]
          - Solar earns daytime LMP (suppressed by own output)
          - Wind earns wind-hour LMP
          - Clean firm earns baseload (all-hours) LMP
          - Storage earns arbitrage spread
       c. Capacity market payment (degrading with clean share)
       d. Ancillary service revenue
       e. Carbon credit revenue (if carbon price > $0)

    4. PROFIT: profit_per_mwh = revenue_per_mwh - lcoe_per_mwh  [NEW]
       - If best mix is profitable (profit > 0): DEPLOY IT
         → Lock in resources, advance to next step
       - If NO mix is profitable (all profit < 0): STOP
         → This is the market equilibrium clean level for this scenario
         → Record current_clean_pct as the "market outcome"

    5. LOCK IN (if deployed):                                 [step7a pattern]
       floor_twh[resource] = max(floor_twh[resource], deployed_twh[resource])

    6. LEARN (if deployed):                                   [step8d pattern]
       cumulative_gw[resource] += new_capacity_gw
       # Deployment lowers future costs → may unlock next step

    7. PRICE UPDATE (automatic):
       - Next step's LMP reflects smaller fossil stack
       - Scarcity value rises → dispatchable resources more attractive
       - But cannibalization also rises → VRE revenue falls
```

**The stopping point IS the result.** Each scenario produces a different market-equilibrium clean level. The dashboard shows how market conditions determine where deployment stalls or accelerates.

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

## Scenario Axes (TBD — User to Define)

Each scenario represents a different set of **market conditions** — not a different target. The question each scenario answers: *"How clean does the grid get when the market looks like this?"*

Candidate axes (not finalized):

| Axis | Options | What It Changes |
|------|---------|-----------------|
| **Carbon price** | $0 / $50 / $100 / $185 | Raises fossil marginal cost → raises LMP → improves clean revenue. Direct: makes every clean MWh worth `carbon_price × displaced_rate` more. |
| **Learning speed** | Fast / Medium / Slow | How quickly early deployment drives FOAK→NOAK cost reduction. Fast learning = early deployment unlocks cascade. |
| **Starting LCOE** | Low / Medium / High | Where technology costs begin (2025 starting point). Low = some resources already profitable at t=0. |
| **Participation level** | TBD | How many corporate buyers / utilities participate. Affects cumulative GW → learning curve speed. |
| **Policy regime** | TBD | ITC/PTC, 45Q, state RPS mandates. Changes effective cost floor. |
| **Capacity market reform** | TBD | ELCC-based vs flat capacity payments. Affects dispatchable premium. |

**Not yet decided**: Which axes, how many levels, which combinations. The scenario design determines what story the dashboard tells. Need to think through what the SMARTargets page should show before committing to axes.

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
Step 2 (EF parquets) ──┐
Step 4 (dispatch cache) ├── SMARTargets module ──→ smartargets output
pipeline_config.py ─────┤
step5b LMP engine ──────┘
```

No upstream changes needed. All inputs already exist.

## Open Questions

1. **Scenario axis definitions** — what are the 15-20 scenarios? What story should the SMARTargets dashboard page tell? This is the critical design question.
2. **Cross-regional capital flow** — V1 is per-ISO. Should V2 allow developers to deploy wherever profit is highest across ISOs? (Global merit order by profit, not MAC.)
3. **Demand growth** — existing model ties demand growth to SBTi timeline (threshold = year). In SMARTargets, the threshold is an *output* not a year. How should demand grow? Calendar-based? Or tied to deployment pace?
4. **Within-threshold cannibalization** — the LMP engine captures *between-threshold* cannibalization (each step sees updated fossil stack). But within a single threshold, adding 10 GW of solar suppresses solar-hour prices more than adding 1 GW. Should we model this, or is the 5% step size granular enough?
5. **PPA vs merchant revenue** — current design assumes merchant (spot LMP). Real developers sign PPAs at a discount to expected spot. Does this matter for the simulation, or is merchant a reasonable proxy?
6. **Existing asset economics** — do 2025 existing clean resources earn revenue (validating they'd "survive" in the new market), or do we only track incremental new-build profitability?
7. **Stopping rule nuance** — when profit goes negative, is that truly "stop"? Or might developers accept marginal losses if mandated (RPS), subsidized (PTC), or pursuing portfolio strategy? The model as designed is pure-market — no policy mandates forcing unprofitable deployment.
8. **What is the relationship between SMARTargets and the existing SBTi/procurement strategy pages?** SMARTargets answers "what would the market do" — the existing pages answer "what should a buyer do." How do they connect on the dashboard?
