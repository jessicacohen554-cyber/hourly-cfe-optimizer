# SMARTargets Module — Sequential Market-Based Resource Allocation

> **Status**: Design phase — not yet implemented.
> **Last updated**: 2026-03-06.

## Core Concept

A **profit-maximizing sequential deployment model** that determines optimal clean energy resource mixes by maximizing `revenue - cost` (not minimizing marginal abatement cost). At each threshold step, the model asks: *"What resource portfolio maximizes profit given current market prices, locked-in prior deployments, and learning-curve-adjusted costs?"*

This inverts the existing pipeline's logic:
- **Existing pipeline (Steps 1-3, 7A)**: Minimizes cost to achieve a CFE target. MAC is a derived metric.
- **SMARTargets**: Maximizes profit (market revenue minus LCOE) subject to floor ratchets. The CFE target emerges from what's profitable, not the other way around.

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

```
For each scenario (15-20 combos):
  Initialize:
    floor_twh = 2025 existing clean by resource (from GRID_MIX_SHARES)
    cumulative_gw = {resource: global_installed_2025}  # for learning curves

  For each threshold step t (50%, 55%, ... 99.99%):

    1. FILTER: Get EF mixes meeting floor ratchet           [scenario_common.batch_filter_floor]
       - If exhausted, PFS fallback with floor window       [scenario_common._filter_pfs_by_floor_window]

    2. COST: Compute LCOE for each candidate mix             [scenario_common.batch_compute_total_costs]
       - Apply Wright's Law reduction based on cumulative_gw [procurement_utils.learning_fraction]
       - Global cumulative: all ISOs contribute to same curve

    3. REVENUE: For each candidate mix, compute revenue      [NEW]
       a. Run LMP engine on fossil stack at threshold t      [step5b.build_merit_order_stack + PriceModel]
          - Fossil stack naturally shrinks as clean_pct rises
          - Scarcity pricing kicks in when reserves thin
       b. Compute hourly energy revenue from 8,760 LMP prices
          - Each resource earns its profile-weighted LMP
          - Solar earns daytime LMP, wind earns wind-hour LMP
          - Clean firm earns baseload (all-hours) LMP
          - Storage earns arbitrage spread
       c. Add capacity market payment (degraded by clean share)
          - capacity_rev = base_price × max(0, 1 - α × clean_pct/100) × ELCC
       d. Add ancillary service revenue (static or VRE-scaled)
       e. Optionally add carbon credit revenue (carbon_price × avoided_tons)

    4. PROFIT: profit_per_mwh = revenue_per_mwh - lcoe_per_mwh  [NEW]
       - Select mix with highest profit (argmax, not argmin)
       - If no mix is profitable, select least-loss mix (most negative profit closest to zero)

    5. LOCK IN: Update floor ratchet                         [step7a pattern]
       floor_twh[resource] = max(floor_twh[resource], deployed_twh[resource])

    6. LEARN: Update cumulative GW deployed                  [step8d pattern]
       cumulative_gw[resource] += new_capacity_gw
       # Global: same cumulative_gw dict shared across ISOs

    7. PRICE UPDATE: LMP for next step reflects new fossil stack  [natural — step 3a uses new clean_pct]
       - No explicit feedback loop needed
       - Merit order stack changes because fossil fleet shrinks
       - Scarcity pricing changes because reserve margin changes
```

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

The user hasn't finalized the 15-20 scenario definitions yet. Candidate axes:

| Axis | Options | Notes |
|------|---------|-------|
| **Carbon price** | $0 / $50 / $100 / $185 | EPA SCC, EU ETS, Rennert |
| **Learning speed** | Fast / Medium / Slow | Wright's Law FOAK→NOAK timeline |
| **Starting cost level** | Low / Medium / High | Which LCOE table row to start from |
| **Participation level** | TBD | How many buyers participate (affects cumulative GW for learning) |
| **Policy regime** | TBD | ITC/PTC extensions, 45Q, state mandates |

Combinatorics: 4 × 3 × 3 = 36 (too many) → curate to 15-20 representative combos.

**Open question**: What goes on the SMARTargets dashboard page? The scenario axes need to tell a story — "under what conditions does X resource become profitable?" is the core question.

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

1. **Scenario axis definitions** — what are the 15-20 scenarios? User needs to think through what the SMARTargets page should show.
2. **Cross-regional deployment** — V1 is per-ISO. Should V2 allow capital to flow to highest-profit ISO first (global merit order by profit)?
3. **Demand growth** — existing model grows demand with threshold (SBTi timeline). Same here, or different growth assumptions?
4. **Revenue cannibalization** — as solar/wind deploy, they suppress their own LMP hours. The LMP engine captures this implicitly (more clean = lower residual demand = lower prices during those hours). But should we model *within-threshold* cannibalization (diminishing returns as you add more of the same resource at a single threshold)?
5. **PPA vs merchant** — is the revenue model based on merchant (spot LMP) revenue, or does it assume PPA contracts? PPAs would smooth revenue but at a discount. Current design assumes merchant.
6. **Existing asset revenue** — do existing clean resources (2025 fleet) earn revenue in this model, or only new-build? Affects profit calculation for incremental deployment.
