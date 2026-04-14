# Reliability Tax — Methodology

This file is the authoritative record of every methodology decision made for the Reliability Tax sub-project. It is written-first (per the documentation-first rule) and updated immediately whenever a decision is approved. Code must conform to this document, not the other way around.

For the locked invariants (endpoint targets, planning horizon, pathway list, clean firm bucket, cost basis, ISO scope, stranding scope, demand growth) see `reliability_tax/README.md` and `SPEC.md` §24.1. This file does **not** restate them — it covers only the implementation-level decisions that build on top of those invariants.

---

## Decision log

All 10 decision cards below were approved by the user in Prompt 2A on 2026-04-13. The user opted to accept the recommended option on every card.

### Card 1 — Optimizer architecture: **EF-rung-chain**

The pathway optimizer treats the existing Step 2.1 efficient frontier as a pre-solved menu of capacity snapshots, one per CFE threshold rung (50, 55, 60, 65, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 99.5, 99.9 — the 16 active rungs from `scripts/pipeline_config.THRESHOLDS`). The optimizer does **not** solve a per-year LP or full-horizon MILP. Instead, for each year 2025–2050:

1. The pathway's per-year CFE target is read from the SBTi ladder (see Card 7 for Pathway 3 deviations).
2. The optimizer identifies the EF rung whose stored CFE% matches that target.
3. The capacity vector is linearly interpolated between the previous year's capacity and the target rung's capacity vector.
4. Wright's Law is applied to capex year by year to reprice newly-added capacity (existing capacity carries its as-built capex).
5. Cost components (capex, fixed O&M, variable O&M, fuel) are computed for that year's installed base and the marginal MW added.

Pathway 1 (VRE + batteries only) is restricted to EF rungs with `clean_firm_twh == 0`. Pathway 3 follows an interpolated trajectory toward the 99.9% endpoint mix instead of the SBTi ladder (Card 7). Pathways 2a/2b follow Pathway 1's restriction until their pivot, then switch to the unrestricted EF menu.

**Why not LP/MILP**: Step 2.1 already solved the snapshot endpoint-cost optimization at every threshold across the 5,832-scenario cost cube. Re-solving it per year per pathway would duplicate that work, require an LP solver, and add academic-debate surface area without a corresponding accuracy gain. The rung-chain approach is `O(years × pathways × ISOs)` and stays inside the existing methodology.

### Card 2 — CFE% computation: **EF-rung tagging with linear interpolation**

When the current capacity vector exactly matches an EF rung, the achieved CFE% is read from the parquet's stored value. For interpolated states between rungs (which is the common case during the year-by-year climb), the CFE% is linearly interpolated by the same capacity-fraction parameter used to interpolate the capacity vector itself.

Off-rail states (capacity vectors that don't lie on the rung-to-rung line) are not produced by the optimizer's interpolation logic, so this case does not arise in practice. The hourly dispatch reconstruction path (`dispatch_utils`) is not called by the pathway optimizer at all — it is reserved for the curtailment validation in Card 10.

### Card 3 — Retirements: **Battery-only 15-year rebuild**

Within the 2025–2050 horizon, only batteries (4hr Li-ion, 8hr Li-ion, 100hr LDES) are retired and rebuilt. Each battery cohort retires 15 years after installation and is rebuilt at the same nameplate MW with the then-current Wright's-adjusted capex. The rebuild is recorded as a new `mw_added` event in `capacity_by_year.parquet` and as a fresh capex line in `cost_by_year.parquet`.

Nuclear (60-year design life), CCS-CCGT (30-year), geothermal (30-year), solar PV (30-year), and wind (25-year) are all assumed to remain in service through 2050 once built. This is approximate for solar built in 2025 (which would technically retire in 2055, just outside the horizon) and for wind built in 2025 (which would retire in 2050 itself). Both are treated as in-service through 2050; the small over-count is documented here as a known minor optimism in the cost numbers.

A simple `age_year` column is carried per resource cohort in the optimizer's internal state to support the battery-rebuild bookkeeping. This adds no measurable runtime.

### Card 4 — Pathway 1 feasibility ceiling: **Best-feasible, recorded in manifest**

Pathway 1 walks the no-clean-firm subset of the EF ladder upward year by year. When the next required rung does not exist (because no `clean_firm_twh == 0` mix at that ISO can reach that CFE), the optimizer:

1. Stops adding new capacity beyond the highest reachable rung.
2. Records `feasibility_ceiling_cfe = highest_reachable_cfe` in the run manifest.
3. Continues year-by-year cost accounting at the ceiling capacity through 2050 (so NPV reflects the cost of holding the ceiling for the remaining years).
4. Tags the run manifest with `infeasible_beyond = endpoint_cfe`.

The four output parquets are still written for infeasible runs. The `cfe_by_year` table will show `achieved_cfe` plateauing at the ceiling for all post-ceiling years, with `target_cfe` continuing to climb. This matches the locked invariant: "Record ceiling, flag in manifest, do not force a solution."

### Card 5 — Pathway 2a pivot trigger: **First year achieved CFE ≥ 90%**

The pivot from VRE-only to clean-firm-allowed occurs in the first calendar year `Y` where the optimizer's interpolated capacity vector delivers `achieved_cfe >= 0.90`. Starting in year `Y+1`, the optimizer's allowed rung set expands from `{clean_firm_twh == 0}` to the full EF menu. Year `Y` itself remains a VRE-only year — the pivot decision happens at the end-of-year evaluation and takes effect the following year.

If the ISO is on a slow climb and Pathway 2a never reaches 90% before 2050, the run is treated as infeasible per Card 4 and no pivot is ever triggered.

### Card 6 — Pathway 2b pivot trigger: **Cheapest-available clean firm LCOE, finite-difference marginal**

The pivot occurs in the first year `Y` where:

```
marginal_$/CFE%_year_Y  >  cheapest_clean_firm_LCOE_year_Y
```

**Marginal $/CFE% (Card 6 part B.i)**: finite difference between the current EF rung and the next higher allowed rung on Pathway 2b's restricted (no-clean-firm) rail. Formally, if the optimizer is currently at rung `R_i` and would move to rung `R_{i+1}` next year:

```
marginal_$/CFE%_year_Y = (npv_cost_at_R_{i+1} - npv_cost_at_R_i) / (cfe_R_{i+1} - cfe_R_i)
```

with both NPVs evaluated at year-`Y` Wright's-adjusted capex.

**Cheapest available clean firm LCOE (Card 6 part A.i)**: the minimum over `{nuclear, CCS-CCGT, geothermal}` of the resource's year-`Y` Wright's-adjusted LCOE, subject to that resource being available in this ISO under existing constraints (geothermal CAISO-only via `pipeline_config.GEOTHERMAL_ISOS`, CCS subject to `pipeline_config.CCS_CAP_TWH`). Nuclear is assumed available in all ISOs.

After the pivot, Pathway 2b's allowed rung set expands to the full EF menu starting the following year, identical to Pathway 2a's post-pivot behavior.

### Card 7 — Pathway 3 schedule: **Endpoint-aim interpolation**

For a Pathway 3 run targeting endpoint `E ∈ {0.90, 0.95, 0.975, 0.99, 0.999}`:

1. Look up the Step 2.1 EF-optimal mix at rung `E` for this ISO. Call this the *target endpoint mix*.
2. Look up the 2025 baseline mix (the actual current installed capacity, from `pipeline_config` baseline tables — not zero).
3. For each year 2025 → 2050, linearly interpolate every resource's MW between the baseline and the target endpoint mix, scaled by `year_index / 25`.
4. Wright's Law reprices new additions year by year as they are placed.
5. The yearly CFE% is recomputed from the interpolated capacity vector via the rung-tagging method (Card 2). It is allowed to undershoot the SBTi ladder in early years — Pathway 3's defining feature is "head straight for the destination" rather than "follow SBTi milestones".

Pathway 3 is never infeasible (the endpoint mix is by construction the EF-optimal mix at that endpoint, which is feasible whenever Step 2.1 produced a result for that ISO × threshold).

### Card 8 — Output table schemas

The optimizer writes exactly four parquets per run, plus one manifest JSON. All paths are relative to `reliability_tax/results/`.

#### `capacity_by_year.parquet`
| Column | Type | Notes |
|---|---|---|
| `year` | int | 2025–2050 |
| `iso` | str | one of the 7 ISOs |
| `pathway` | str | `1`, `2a`, `2b`, `3` |
| `endpoint` | float | 0.90 / 0.95 / 0.975 / 0.99 / 0.999 |
| `resource` | str | solar, wind, offshore_wind, hydro, nuclear, ccs_ccgt, geothermal, battery_4hr, battery_8hr, ldes_100hr, new_gas |
| `mw_installed` | float | cumulative nameplate MW in service this year |
| `mw_added` | float | new MW added this year (incl. battery rebuilds) |
| `mw_retired` | float | MW retired this year (battery 15y only within horizon) |
| `wrights_capex_usd_per_kw` | float | year-Y Wright's-adjusted overnight capex applied to `mw_added` |

#### `cost_by_year.parquet`
| Column | Type | Notes |
|---|---|---|
| `year` | int | 2025–2050 |
| `iso` | str | |
| `pathway` | str | |
| `endpoint` | float | |
| `capex_usd_mm` | float | sum across resources of `mw_added * wrights_capex_usd_per_kw / 1000` |
| `fixed_om_usd_mm` | float | from `pipeline_config` fixed O&M tables × installed MW |
| `variable_om_usd_mm` | float | variable O&M × dispatched MWh per resource |
| `fuel_usd_mm` | float | non-zero only for CCS-CCGT and (if any) new_gas; uses `pipeline_config` fuel adjustments |
| `total_usd_mm` | float | sum of the four cost components |
| `discount_factor_7pct` | float | `(1.07)^-(year - 2025)` |

#### `cfe_by_year.parquet`
| Column | Type | Notes |
|---|---|---|
| `year` | int | |
| `iso` | str | |
| `pathway` | str | |
| `endpoint` | float | |
| `target_cfe` | float | SBTi ladder value or Pathway-3 schedule |
| `achieved_cfe` | float | from rung-tagging interpolation (Card 2) |
| `matched_twh` | float | clean energy delivered to load |
| `gap_twh` | float | unmet demand met by gap (i.e. fossil) |
| `curtailed_twh` | float | per Card 10 |
| `marginal_usd_per_cfe_pct` | float | finite-diff slope from previous year's rung to this year's rung |

#### `stranding_summary.parquet` (one row per run)
| Column | Type | Notes |
|---|---|---|
| `iso` | str | |
| `pathway` | str | |
| `endpoint` | float | |
| `feasibility_ceiling_cfe` | float | NaN if pathway reached endpoint |
| `new_gas_cf_2050` | float | new-build gas CF in 2050; NaN if pathway built no new gas |
| `vre_curtailment_pct_2050` | float | per Card 10 |
| `stranded_new_gas_gw` | float | new-build gas GW with CF<20% in 2050 |
| `stranded_vre_twh` | float | curtailed VRE TWh in 2050 if curtailment >30% |
| `stranded_new_tx_usd_mm` | float | new-build transmission $ written off (definition refined later) |
| `npv_undisc_usd_mm` | float | undiscounted cumulative 2025–2050 |
| `npv_5pct` | float | NPV @ 5% real |
| `npv_7pct` | float | NPV @ 7% real (objective) |
| `npv_9pct` | float | NPV @ 9% real |

#### `run_manifest.json`
Run metadata: git commit hash, input file hashes (Step 2.1 EF parquet, Step 5 Wright's curves), `feasibility_ceiling_cfe`, `infeasible_beyond` flag, optimizer version, runtime seconds, decision-card-snapshot hash.

### Card 9 — Toy case spec for unit tests

The toy world is a synthetic 3-year, 2-resource, 1-load environment with a hand-computable analytical optimum. It is implemented as a fixture (`toy_world` in `test_pathway_optimizer.py`) that injects synthetic cost tables, generation profiles, and load curves into the optimizer.

**World parameters**:
- Years: 2025, 2026, 2027 (3-year horizon).
- Endpoint target: `cfe >= 0.80` by end of 2027.
- Resources: two synthetic resources `flat` and `peaky`.
  - `flat` (clean-firm proxy): `lcoe = $100/MWh`, `cf = 1.00` (constant 100% of hours), `capex = $5,000/kW`, available unlimited.
  - `peaky` (VRE proxy): `lcoe = $30/MWh`, `cf = 0.50` (delivers in exactly half of hours), `capex = $1,000/kW`, available unlimited.
- Load: 100 MWh per year, constant, every hour.
- Discount rate: 7% real.
- Wright's Law: disabled (learning exponent = 0) so capex does not decline over the toy horizon.
- Storage: not present in the toy world.

**Analytical optimum at end of year 3 (target = 80% CFE)**:
- `peaky` covers 50% of hours by construction (when its CF lines up with load), contributing 50 MWh/year matched.
- `flat` must cover the additional 30 MWh/year to reach 80 MWh/year matched out of 100 MWh load.
- Required `peaky` nameplate MW: `100 / 8760 / 0.50 = 0.0228 MW` (delivers 100 MWh of generation potential, of which 50 MWh aligns with load).
- Required `flat` nameplate MW: `30 / 8760 = 0.00342 MW` (delivers exactly 30 MWh/year matched).
- Year-3 cost = capex amortized over remaining life + variable cost. Hand-computed reference NPV is asserted to within ±0.5%.

The optimizer's EF-rung-chain logic in the toy world is fed a synthetic rung table with three rungs at 50% / 65% / 80% CFE so the rung-chain machinery is exercised.

### Card 10 — Curtailment accounting: **Annual-totals analytical**

For each year `Y` and each VRE resource `R`:

```
potential_twh_R_Y    = installed_mw_R_Y * cf_R * 8760 / 1e6
matched_twh_R_Y      = from EF mix metadata (per-resource matched contribution)
charged_twh_R_Y      = from EF mix metadata (energy routed to storage)
curtailment_twh_R_Y  = potential_twh_R_Y - matched_twh_R_Y - charged_twh_R_Y
```

The aggregate VRE curtailment percentage is `sum(curtailment) / sum(potential)` across all VRE resources for the year. The 30% stranding test is `vre_curtailment_pct_2050 > 0.30`.

This avoids hourly recomputation. The EF mix metadata already separates matched / surplus (curtailed) / charge contributions per resource, so this is a near-zero-cost lookup.

---

## Implementation defaults (assumed unless flagged)

These are not numbered cards but are recorded here so they survive session handoff.

- **Discount factor convention**: 2025 = year 0, discount factor `(1 + r)^-(year - 2025)`. 2050 = year 25.
- **Cost basis**: every cost figure is pulled from `scripts/pipeline_config.py` without local overrides. Local cost overrides are forbidden in this sub-project.
- **Demand growth**: Medium (`pipeline_config.DEMAND_GROWTH_RATES['medium']`) by default. Section 2 sensitivity sweeps L/M/H per ISO.
- **Nuclear subsidy roll-off**: reuse the convention in `scripts/procurement_utils.py` unchanged.
- **Objective**: minimize NPV@7% of cumulative 2025–2050 total system cost subject to endpoint and pathway constraints. NPV@5% and NPV@9% are reported but not optimized.
- **Output count**: exactly four parquets per run (capacity, cost, cfe, stranding). The manifest JSON is metadata, not one of the four tables.
- **Script path**: `reliability_tax/scripts/step_2_3_pathway_optimizer.py`.
- **Test path**: `reliability_tax/tests/test_pathway_optimizer.py`, runnable via `pytest reliability_tax/tests/`.
- **Determinism**: every run is deterministic. No stochastic sampling, no random seeds, no Monte Carlo. Re-running the same `(iso, pathway, endpoint)` produces byte-identical parquets.

---

## What this methodology deliberately excludes

- **Market price formation**: no LMP modeling, no clearing prices, no merit-order dispatch *for cost evaluation*. Costs come from the LCOE / capex / O&M tables in `pipeline_config`.
- **Policy variables**: no IRA scenarios, no PTC/ITC sweeps, no carbon tax — Wright's Law and nuclear subsidy roll-off are the only time-varying policy mechanisms.
- **Demand elasticity**: demand grows exogenously per the L/M/H tables. No price-responsive demand.
- **Stochastic elements**: weather years, fuel-price uncertainty, technology-cost distributions — all out of scope. Single deterministic central case per run.
- **Behavioral factors beyond the literal pivot triggers in Cards 5 and 6**.

This is a **deterministic cost minimization under endpoint and pathway constraint**, not a market or behavioral simulation. The Step 6 mechanisms that model uncertainty, policy, or behavior are stripped.

---

## Status

- 2026-04-13 — Cards 1–10 approved by user; recommendations adopted on every card. METHODOLOGY.md created.
- Next: inheritance discovery (Step 2 of Prompt 2A) → user approval → write `step_2_3_pathway_optimizer.py` and `test_pathway_optimizer.py` → run unit tests → report results in chat.
