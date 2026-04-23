# Step 2.3 — Reliability-Tax Pathway Optimizer: Methodology

**Version:** v3 (post-refactor)
**Script:** `step_2_3_reliability_tax_optimized_py.py`
**Scope:** 7 ISOs × 5 pathways × 4 endpoints = 140 pathway runs (2025–2050)

-----

## 1. Purpose

Step 2.3 determines the least-cost trajectory for transitioning each ISO’s grid from its current generation mix to a clean-energy endpoint, subject to hourly reliability constraints. Its primary outputs are:

- A year-by-year resource buildout schedule (which resources are added, in what quantity, and at what locked-in LCOE).
- The **reliability tax** — a per-MWh cost adder representing the backup gas infrastructure that clean portfolios implicitly require.
- Gas fleet sizing, capacity factor, and stranded-asset exposure at 2050.
- Undiscounted and NPV total system costs at three discount rates (5%, 7%, 9%).

The model operates as a constrained myopic optimizer with optional foresight steering via hard ceiling constraints derived from the cost-optimal endpoint mix produced by Step 2.2A.

-----

## 2. Architecture Overview

The solver uses a **two-stage architecture** designed to make the combinatorial problem tractable without sacrificing hourly reliability fidelity.

### 2.1 Stage 1 — Precompute Worst-Hour Residuals

The candidate pool consists of millions of discrete energy-mix combinations (the “EF pool,” produced by Step 2.1), each defined by percentage shares across 11 resource columns and 4 storage-dispatch columns. Evaluating every candidate against 8,760 hourly dispatch profiles at every year-step would be computationally prohibitive.

Stage 1 solves this by precomputing a single reliability scalar per candidate:

1. **Archetype matching.** The Step 3 dispatch cache contains hourly generation profiles for approximately 63 representative grid archetypes. Each of the millions of EF candidates is matched to its nearest archetype using L1 distance on quantized mix-share vectors. This matching is accelerated by a Numba-JIT parallel kernel (`_l1_nn`) that processes the candidate axis with `prange`, avoiding the memory cost of materializing an `(N, 63, D)` broadcast tensor.
1. **Residual calculation.** For each archetype, the per-hour shortfall between demand (inflated by the resource adequacy margin) and total clean generation is computed. The **99.97th percentile** of this shortfall distribution becomes the archetype’s residual — a proxy for how much gas backup the mix requires. The 99.97th percentile corresponds to approximately the worst 3 hours of 8,760, aligning with standard “1-in-10 LOLE” reliability planning practice.
1. **Sidecar output.** Stage 1 writes a Parquet sidecar per ISO per threshold band containing `resid_norm_p9997` (fractional residual) and `clean_peak_hour_mw` (peak-hour clean capacity). Sidecars are cache-versioned against the dispatch cache; they rebuild automatically when the upstream cache changes.

### 2.2 Stage 2 — Year-by-Year Constrained Optimization

With precomputed residuals in hand, Stage 2 loops from 2025 through the endpoint year, selecting one winner candidate per year from the full pool. After the endpoint year, the winner is frozen (post-endpoint freeze). The solver processes candidates in **streaming chunks** of 2 million rows to cap peak memory at a few hundred MB rather than requiring tens of GB.

-----

## 3. Candidate Scoring

Each candidate is scored by its **incremental system cost** — the cost of building and operating the resources implied by that candidate’s mix, accounting for what has already been committed in prior years.

### 3.1 Sunk-Cost Accounting

Only the *incremental* TWh above the running-maximum floor for each resource are priced at the current year’s LCOE. Previously committed capacity is treated as sunk. Without this correction, re-evaluating the same solar capacity every year at that year’s LCOE would multiply-count costs and penalize early buildout.

### 3.2 Cost Components

For a candidate at year *t*, the total score is the sum of:

|Component                 |Calculation                                                                                                                                                 |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
|**Non-CF incremental**    |For each non-clean-firm resource *r*: `max(0, TWh_r(t) − floor_r) × delivered_LCOE_r(t)`, converted to USD. Delivered LCOE = base LCOE + transmission adder.|
|**CF tranche incremental**|Same logic applied per tranche (uprate, geothermal, new nuclear, CCS), using each tranche’s effective LCOE at year *t*.                                     |
|**Storage dispatch**      |`Σ (storage_dispatch_pct / 100) × demand_TWh × storage_LCOE`.                                                                                               |
|**Gas infrastructure**    |`fleet_MW × (annualized_capex + FOM) + unmatched_MWh × fuel_cost + existing_gas_FOM`.                                                                       |

### 3.3 Technology Cost Curves

Base LCOEs for solar, wind, batteries, and other technologies are loaded from `pipeline_config` lookup tables. Costs decline over time following **Wright’s Law learning curves** loaded from pre-computed JSON (`wrights_law_curves.json`). Each technology has a FOAK (first-of-a-kind) and NOAK (Nth-of-a-kind) cost, with the trajectory shaped by pathway-specific learning-rate parameters and NOAK-achievement windows.

Transmission adders are ISO- and resource-specific, drawn from `pipeline_config.get_tx()`.

-----

## 4. Clean-Firm Tranche Decomposition

“Clean firm” is not a single technology — it represents a family with very different costs and regional availability. The model decomposes each candidate’s total clean-firm share into five tranches applied in strict merit order:

1. **Existing** — Current nuclear and baseload hydro, capped at the ISO’s baseline clean-firm share. Cost: zero (already built).
1. **Uprates** — Capacity gains from existing nuclear uprates, capped at `UPRATE_CAP_TWH[iso]`. Cost: flat uprate LCOE.
1. **Geothermal** — Available in CAISO only, capped at `GEOTHERMAL_CAP_TWH`. Cost: FOAK→NOAK via Wright’s Law.
1. **New nuclear** — Unlimited above prior tranches. Cost: FOAK→NOAK via Wright’s Law + transmission.
1. **CCS (gas w/ carbon capture)** — Capped at `CCS_CAP_TWH[iso]`. Cost: FOAK→NOAK via Wright’s Law + transmission + NEISO gas adder where applicable.

New nuclear and CCS compete on cost: whichever has the lower effective LCOE in a given year is dispatched first, with the other absorbing the remainder.

**Pathway gating** controls tranche availability. Pathways 1 and 1a restrict candidates to existing + uprate capacity only (no new nuclear, no CCS). Pathways 2a, 2b, and 3 unlock all tranches.

The tranche decomposition is implemented once in `_tranche_merit_order` and called from the scoring kernel, floor updates, and the post-solve decomposition, ensuring a single source of truth.

-----

## 5. Constraints

Each candidate must pass a composite feasibility mask to be eligible. If no candidate passes, a **safety-net fallback** selects the global lowest-cost candidate with no constraints and logs a warning.

### 5.1 Monotone Ratchet

Once a resource reaches *X* TWh in year *Y*, no future year may select a candidate with less than *X* TWh of that resource (tolerance: 10⁻⁶ TWh). This applies per non-CF resource and per clean-firm tranche. The constraint prevents physically impossible “un-building” and ensures monotone capacity trajectories.

### 5.2 CFE Target Ramp

Annual clean-energy percentage targets follow an SBTi-aligned ramp with configurable waypoints (default: 50% by 2030, 70% by 2035, 90% by 2040, 95% by 2045). Between waypoints, targets are linearly interpolated. Candidates must achieve a `hourly_match_score` within 0.5 percentage points of the year’s target.

The waypoints are stored in `RunConfig.cfe_waypoints` and can be overridden per run.

### 5.3 Pathway Mask

For Pathways 1/1a, candidates are excluded if their clean-firm share exceeds existing + uprate capacity (plus a 0.5% tolerance) or if their CCS share is nonzero.

### 5.4 Tranche Feasibility

On Pathways 1/1a (where advanced tranches are unavailable), any candidate requiring clean-firm capacity beyond existing + uprate is marked infeasible.

### 5.5 Endpoint Ceiling Constraints (Foresight Mode)

When `solver_mode='foresight'`, the optimizer loads the cost-optimal endpoint resource shares from Step 2.2A and enforces them as **hard ceilings** on per-resource shares. This prevents the myopic solver from locking in a trajectory that overshoots any single resource relative to the known-optimal endpoint mix, without introducing tuning parameters.

-----

## 6. Gas Fleet Sizing and Stranding

### 6.1 Gas Sizing

For each candidate, the model computes new gas capacity needed:

1. Start with the precomputed worst-hour residual (the 99.97th-percentile shortfall).
1. Scale by annual demand (MWh) and divide by the ISO’s gas availability factor.
1. Subtract the 2025 baseline gas need and any existing gas capacity freed by fossil retirement.
1. Take the running maximum across years (gas, once built, stays).

### 6.2 Fossil Retirement

As clean energy displaces fossil generation, coal is retired first, then oil, then gas. The model tracks cumulative gas displacement to compute how much existing gas capacity is freed to serve as backup, reducing the need for new-build gas. The retirement function returns only the gas-displacement credit needed for the `existing_gas_vec` calculation.

### 6.3 Per-Vintage Stranding (Card 12 Rework)

Rather than assuming all gas is built in a single peak year, the model tracks **per-year gas vintages**. Each year that cumulative gas capacity increases, a new vintage is recorded with its MW increment and build year. At 2050, stranded capacity is allocated proportionally across vintages, and each vintage’s stranded capex is depreciated based on its age and a 25-year asset life:

```
stranded_capex_vintage = stranded_MW × overnight_capex × (remaining_life / asset_life)
```

This produces a more granular and realistic stranding estimate than the prior lump-build approach.

-----

## 7. Reliability Tax

The reliability tax quantifies the hidden infrastructure cost of intermittency. It sums four components over the full 2025–2050 horizon:

|Component                      |Description                                     |
|-------------------------------|------------------------------------------------|
|**New gas capex (annualized)** |`Σ active_fleet_MW(t) × annualized_capex_per_MW`|
|**New gas FOM**                |`Σ active_fleet_MW(t) × FOM_per_MW`             |
|**Existing gas FOM (carried)** |`existing_gas_MW × FOM × N_years`               |
|**VRE/storage overbuild capex**|Reserved for future use (currently 0)           |

The total is divided by total demand (MWh) over the horizon to produce a **$/MWh reliability-tax adder**.

VRE curtailment is computed for endpoint output metadata (via the dispatch cache’s matched vs. surplus energy per resource) but is **not priced** into costs, as curtailment economics are already captured through two channels: (1) LCOEs price the full nameplate share including curtailed MWh, and (2) higher curtailment lowers the hourly match score, which sizes a larger gas fleet.

-----

## 8. Solver Modes

### 8.1 Myopic (Default)

Selects the lowest-cost feasible candidate at each year-step, subject to the constraint mask. Computationally simple and deterministic.

### 8.2 Foresight

Identical to myopic but adds hard ceiling constraints from Step 2.2A’s cost-optimal endpoint mix. No tuning parameters (the prior quadratic-penalty foresight solver with its λ parameter was removed during the v3 refactor). The 2.2A dependency is preserved — the step remains the source of truth for endpoint resource allocation.

-----

## 9. Output Schema

Each pathway run produces a JSON file (`schema_version: 2`) containing:

|Section                          |Contents                                                            |
|---------------------------------|--------------------------------------------------------------------|
|`config`                         |ISO, pathway, endpoint, growth/cost scenario levers                 |
|`feasibility`                    |Physical feasibility flag and notes                                 |
|`headline`                       |Achieved CFE %, undiscounted cost, NPVs at 5/7/9%, endpoint year    |
|`tables.annual_buildout`         |Per-year new vintages with locked LCOE and gas sizing               |
|`tables.annual_cost`             |Per-year gross operating, gas capex/FOM/fuel, net cost, achieved CFE|
|`tables.endpoint_hourly_dispatch`|8,760-hour clean dispatch at endpoint mix                           |
|`tables.new_gas_fleet`           |Per-vintage gas build/strand detail                                 |
|`reliability_tax`                |Component breakdown, total USD, USD/MWh                             |
|`stranding_metadata`             |Fleet size, peak year, stranded MW and capex, per-vintage detail    |
|`vre_curtailment_at_endpoint`    |Per-resource curtailment fractions                                  |
|`endpoint_mix_pct`               |Final resource shares (%)                                           |
|`endpoint_storage_pct`           |Final storage dispatch shares (%)                                   |
|`terminal_ledger`                |Full vintage ledger (resource, COD year, TWh, locked LCOE)          |

-----

## 10. Key Parameters

|Parameter                   |Value                                     |Source                       |
|----------------------------|------------------------------------------|-----------------------------|
|Planning horizon            |2025–2050 (26 years)                      |Hardcoded                    |
|Hours per year              |8,760                                     |Hardcoded                    |
|Worst-hour percentile       |99.97                                     |Hardcoded (≈ worst 3 hours)  |
|Discount rates              |5%, 7%, 9%                                |Hardcoded                    |
|CCGT overnight capex        |$1,200/kW                                 |`CCGT_OVERNIGHT_CAPEX_USD_KW`|
|Gas asset life              |25 years                                  |`NEW_GAS_ASSET_LIFE_YEARS`   |
|Gas capacity factor (fossil)|0.45                                      |`_FOSSIL_CFS['gas']`         |
|Ratchet tolerance           |10⁻⁶ TWh                                  |`RATCHET_TOL`                |
|Streaming chunk size        |2,000,000 rows                            |`_POOL_CHUNK`                |
|Hourly match score floor    |10 pct below threshold                    |`SCORE_FLOOR_PCT`            |
|CFE waypoints (default)     |(2030,50), (2035,70), (2040,90), (2045,95)|`_DEFAULT_CFE_WAYPOINTS`     |
|EF band thresholds          |10–99.9 (20 bands)                        |`_EF_BAND_THRESHOLDS`        |
|Pathways                    |1, 1a, 2a, 2b, 3                          |`PATHWAYS`                   |
|Endpoints                   |90%, 95%, 99%, 99.9%                      |`ENDPOINT_TO_THRESHOLD`      |

-----

## 11. Data Dependencies

|Input                    |Source           |Format                                                                                |
|-------------------------|-----------------|--------------------------------------------------------------------------------------|
|EF candidate pool        |Step 2.1         |Parquet (partitioned by ISO × threshold band)                                         |
|Cost-optimal endpoint mix|Step 2.2A        |Parquet (one row per scenario)                                                        |
|Dispatch cache + manifest|Step 3           |Parquet (hourly profiles per archetype)                                               |
|Generation profiles      |EIA-930          |Parquet (solar, wind, hydro, nuclear, offshore wind, geothermal)                      |
|Demand profiles          |EIA-930          |Parquet (normalized hourly shape per ISO)                                             |
|Fossil fuel mix          |EIA-930          |Parquet (coal/gas/oil shares per ISO)                                                 |
|Hybrid profiles          |Pre-computed     |NPZ (solar+batt, wind+batt at 4h and 8h)                                              |
|Emission rates           |eGRID            |JSON                                                                                  |
|Wright’s Law curves      |Step 5           |JSON                                                                                  |
|Regional config          |`pipeline_config`|Python module (demand, peak MW, grid mix shares, LCOE tables, gas costs, growth rates)|

-----

## 12. Design Decisions (v3 Refactor Summary)

The v3 refactor addressed 17 design elements reviewed across the full codebase. The net result was approximately 220 lines removed, zero new tuning parameters, and several methodology improvements:

**Kept as-is (10 elements):** Two-stage architecture, Numba L1-NN, 99.97th-percentile rule, monotone ratchet, sunk-cost scoring, pathway gating, Wright’s Law curves, vintage ledger, streaming chunked argmin, post-endpoint freeze.

**Simplified (4 elements):** Tranche decomposition deduplicated to a single shared helper. CFE ramp waypoints made configurable via `RunConfig`. Gas stranding reworked from lump-build to per-year vintage tracking. Fossil retirement simplified to gas-credit-only (coal/oil displacement tracking removed as dead code).

**Removed (3 elements):** The 4-tier cascade fallback was replaced with a single feasibility mask plus a safety-net global argmin. The foresight solver’s quadratic penalty was replaced with hard ceiling constraints from Step 2.2A, eliminating the λ tuning parameter. VRE curtailment pricing was removed as double-counting (curtailment cost is already embedded in LCOE and in gas-fleet sizing via lower match scores).

-----

## 13. Execution

```bash
# Single run
python step_2_3_reliability_tax_optimized_py.py \
    --iso PJM --pathway 2a --endpoint 0.95

# Full sweep (all ISOs × pathways × endpoints)
python step_2_3_reliability_tax_optimized_py.py --all

# Filter to single ISO
ISO_FILTER=CAISO python step_2_3_reliability_tax_optimized_py.py --all
```

The `--all` mode parallelizes Stage 1 sidecar builds across ISO×threshold pairs, then parallelizes the full sweep across ISOs, using `multiprocessing.spawn` contexts. Output files land in `analysis/reliability-tax/data/{ISO}/pathway{P}_{ep_tag}.json`.