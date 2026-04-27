# Step 2.3 Rebuild — On-the-Fly Mix Builder

## Decisions & Methodology Spec

Version 3.3 | Updated 2026-04-26 | Status: draft

Changelog: v3.3 — Wright’s Law k bumped to 5 (compressed curve). Nuclear FOAK discounted to 80% of Vogtle-era cost. Dimensionality-scaled samples and beams for Pathway B. Fixed storage cost units ($/MW-yr not $/MWh).

-----

## Architecture Overview

**Current approach (being replaced):** Pre-compute millions of mixes in Step 1 → filter/deduplicate in Step 2.1 → augment sparse bands in Step 2.1b/2.1d → load entire pool in Step 2.3 → search for cheapest feasible mix per year.

**New approach:** Two-stage on-the-fly generation. Stage 1 uses existing EF parquets for the first threshold. Stage 2 generates mixes on the fly for all subsequent thresholds.

**Two-stage solve:**

1. **Stage 1 — First threshold:** Load existing Step 2.1 EF parquets for the first threshold band above baseline CFE. Score, filter, cluster into archetypes. Pick the N best (cheapest per archetype, meaningfully different). Default N=5, adjustable.
1. **Stage 2 — Trajectory:** For each of those N seed mixes, build an independent on-the-fly pathway through all remaining thresholds to 99.9%. At each year, generate candidates above current ratchet floors, score with fused Numba kernel, pick cheapest in the target CFE band. Write results at every threshold crossed.

**Two pathways per ISO:**

- **Pathway A:** VRE + hybrids + storage only. No offshore wind, no geothermal, no CCS, no clean_firm expansion (no uprates, no new nuclear). Existing clean_firm baseline frozen.
- **Pathway B:** All resources (current P3). Nuclear new-build, CCS, offshore wind, geothermal (CAISO only), uprates — full resource set with Wright’s Law learning curves.

**Endpoint:** 99.9% CFE only (target year 2050).

-----

## Locked Decisions

### 1. Timeline ✅

|Parameter           |Value                |
|--------------------|---------------------|
|Base year           |2025                 |
|End year            |2050                 |
|Yearly steps        |26 (2025–2050)       |
|Hours per year      |8,760 (no leap years)|
|Endpoint            |99.9% CFE            |
|Endpoint target year|2050                 |

### 2. CFE Trajectory ✅

**Waypoint ramp (linear interpolation between):**

|Year|CFE target                     |
|----|-------------------------------|
|2025|Baseline clean % (ISO-specific)|
|2030|50%                            |
|2035|70%                            |
|2040|90%                            |
|2045|95%                            |
|2050|99.9%                          |

The solver follows this ramp — each year’s target CFE is determined by linear interpolation between waypoints.

**Results written at each of the 20 EF band thresholds when crossed:**
10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 99.5, 99.9

### 3. Demand Model ✅

- **Growth levels:** Low, Medium, High parameterized. Default: Medium.
- **Growth rates (compound annual):**

|ISO  |Low |Medium|High|
|-----|----|------|----|
|CAISO|1.4%|1.9%  |2.5%|
|ERCOT|2.0%|3.5%  |5.5%|
|PJM  |1.5%|2.4%  |3.6%|
|NYISO|1.3%|2.0%  |4.4%|
|NEISO|0.9%|1.8%  |2.9%|
|MISO |1.2%|2.2%  |3.8%|
|SPP  |1.0%|1.8%  |3.0%|

- **Demand profile:** 8,760-hour normalized shape from EIA-930 (latest year), scaled by `baseline_TWh × (1 + rate)^year_offset`.
- **Peak demand:** grows at same compound rate as energy demand.
- **Resource adequacy margin:** 15%.

### 4. Supply Profiles ✅

All profiles sourced identically to current script:

|Resource     |Profile source                                      |
|-------------|----------------------------------------------------|
|clean_firm   |Nuclear hourly shape from EIA-930 (effectively flat)|
|solar        |EIA-930 solar CF profile                            |
|wind         |EIA-930 wind CF profile                             |
|hydro        |EIA-930 hydro shape (seasonal)                      |
|offshore_wind|EIA-930 offshore wind profile                       |
|geothermal   |Flat (1/8760)                                       |
|ccs_ccgt     |Flat (1/8760)                                       |
|solar_batt4  |Pre-computed hybrid .npz per ISO                    |
|solar_batt8  |Pre-computed hybrid .npz per ISO                    |
|wind_batt4   |Pre-computed hybrid .npz per ISO                    |
|wind_batt8   |Pre-computed hybrid .npz per ISO                    |

**Standalone storage dispatch:** battery (4h), battery8 (8h), LDES (100h), H2 (1000h) — layered on top of resource mix.

### 5. Resource Set by Pathway ✅

**Pathway A (VRE + hybrids + storage, no offshore):**

|Resource                |Available?                           |
|------------------------|-------------------------------------|
|solar                   |✅                                    |
|wind                    |✅                                    |
|hydro                   |✅ (baseline share, grows with demand)|
|solar_batt4             |✅                                    |
|solar_batt8             |✅                                    |
|wind_batt4              |✅                                    |
|wind_batt8              |✅                                    |
|battery (standalone)    |✅                                    |
|battery8 (standalone)   |✅                                    |
|LDES (standalone)       |✅                                    |
|H2 (standalone)         |✅                                    |
|offshore_wind           |❌                                    |
|geothermal              |❌                                    |
|ccs_ccgt                |❌                                    |
|clean_firm (new nuclear)|❌                                    |
|clean_firm (uprates)    |❌                                    |
|clean_firm (existing)   |Frozen at baseline                   |

**Pathway B (full P3):**

|Resource                 |Available?                                |
|-------------------------|------------------------------------------|
|All VRE, hybrids, storage|✅                                         |
|offshore_wind            |✅ (NYISO, NEISO, PJM, CAISO only)         |
|geothermal               |✅ (CAISO only)                            |
|ccs_ccgt                 |✅ (per-ISO TWh caps; zero for NYISO/NEISO)|
|clean_firm (new nuclear) |✅                                         |
|clean_firm (uprates)     |✅ (per-ISO TWh caps)                      |
|clean_firm (existing)    |✅                                         |

### 6. Per-ISO Resource Constraints ✅

**No caps on solar, wind, hybrids, or storage.** Candidates can overbuild freely; curtailment is priced in through the cost function.

**Caps retained:**

|Constraint        |Values                                                                        |
|------------------|------------------------------------------------------------------------------|
|Offshore wind ISOs|NYISO, NEISO, PJM, CAISO (Pathway B only)                                     |
|Geothermal        |CAISO only (Pathway B only)                                                   |
|Geothermal cap TWh|39.0                                                                          |
|CCS cap TWh       |CAISO 25, ERCOT 200, PJM 125, MISO 200, SPP 50, **NYISO 0, NEISO 0**          |
|Uprate cap TWh    |CAISO 1.45, ERCOT 1.70, PJM 20.18, NYISO 2.14, NEISO 2.21, MISO 7.57, SPP 0.76|

### 7. Ratchet Mechanism ✅

- All resource floors (absolute TWh) are monotonically non-decreasing.
- Applies to each resource dimension individually.
- In Pathway B, also applies to clean_firm tranches (uprate, geo, nuclear new, CCS).
- Tolerance: 0.01% of demand (`RATCHET_TOL_PCT = 0.01`).
- On-the-fly generator satisfies ratchet by construction (candidates start from current floors).

### 8. Gas Sizing / Reliability Tax ✅ (simplified)

**Simplified from current script. No vintage ledger, no merit-order retirement.**

- **Gas need per year** = `max(0, resid_p9997 × demand_TWh × 1e6 / GAF - existing_gas_MW)`. The residual is in normalized demand space (dn sums to 1.0); multiply by annual demand MWh to convert to MW. dm32 already includes RA margin — no double-counting.
- **Existing gas capacity** = fixed constant per ISO for the entire run (no retirement)
- **Worst hour metric** = 99.97th percentile of residual demand gap (normalized)
- **Peak gas year** = year with highest gas need across the trajectory
- **Stranding at any threshold** = peak_gas_MW − gas_need_MW at that threshold’s year

**Gas cost assumptions:**

|Parameter              |Source                                                |
|-----------------------|------------------------------------------------------|
|Overnight capex        |$1,200/kW                                             |
|Annualized capex       |Per-ISO (`NEW_CCGT_COST_KW_YR`)                       |
|FOM                    |Per-ISO (`NEW_CCGT_FOM_KW_YR`)                        |
|Existing gas FOM       |Per-ISO (`EXISTING_GAS_FOM_KW_YR`)                    |
|Fuel cost              |Wholesale price + fuel adjustment (per scenario level)|
|Gas CF                 |0.45                                                  |
|Gas availability factor|Per-ISO                                               |
|RA margin              |15%                                                   |

### 9. Cost Model ✅

**LCOE and TX adders:**

- Per-ISO LCOE tables from `pipeline_config.py`.
- TX adders per-ISO per-resource from `pipeline_config.py`.
- Cost level is a scenario dimension — can batch across Low/Medium/High for firm, CCS, TX, demand.
- Default: Medium for all dimensions.

**Learning curves:**

- **Pathway A:** No learning curves. VRE and hybrid LCOEs are flat (no expensive tech to learn down).
- **Pathway B:** Wright’s Law concave curve (exponential decay, not linear) for nuclear, CCS, offshore wind, geothermal. Same FOAK/NOAK endpoints and time windows as current pipeline, but concave shape:

```
cost(t) = NOAK + (FOAK - NOAK) × exp(-k × (t - t_start) / (t_noak - t_start))
```

Where `k = 5` (compressed curve — reaches ~99% of NOAK by t_noak). Nuclear FOAK is discounted to 80% of pipeline_config values (Vogtle was the literal FOAK; subsequent builds start lower).

**Pathway B learning windows (P3):**

|Tech                        |First NOAK year|NOAK year|
|----------------------------|---------------|---------|
|Nuclear                     |2030           |2035     |
|CCS                         |2030           |2035     |
|Offshore wind (fixed)       |2028           |2038     |
|Offshore wind (float, CAISO)|2031           |2042     |
|Geothermal                  |2030           |2035     |

**Scenario dimension keys (3 presets, not combinatorial):**

|Scenario             |ren|firm|batt|ldes|fuel|tx|ccs|geo|q45|
|---------------------|---|----|----|----|----|--|---|---|---|
|**base**             |M  |M   |M   |M   |M   |M |M  |M  |1  |
|**firm_high_vre_low**|L  |H   |L   |L   |M   |L |H  |H  |1  |
|**firm_low_vre_high**|H  |L   |H   |H   |M   |H |L  |L  |1  |

Gas (fuel) always Medium. 45Q always on. Geothermal grouped with firm/CCS.
Demand growth is a separate axis (Low/Medium/High), crossable via `--batch-demand`.
Both cost modes (1 and 2) run for each scenario × demand combo.

### 10. Clean Firm Tranche Decomposition (Pathway B only) ✅

Clean_firm demand is decomposed into 5 tranches consumed in merit order:

1. **Existing** — baseline nuclear, capped at current share × demand TWh
1. **Uprates** — per-ISO TWh cap, flat LCOE ($15/$25/$40 for L/M/H)
1. **Geothermal** — CAISO only, 39 TWh cap, Wright’s Law curve
1. **Nuclear new-build vs CCS** — whichever has lower delivered cost that year goes first; other fills remainder. CCS subject to per-ISO TWh cap (zero for NYISO/NEISO).

Each tranche has its own cost curve. Ratchet applies to each tranche independently.

### 11. Storage Dispatch Cost Model ✅

**Two layers:**

**Layer 1 — Hybrid-embedded storage:** Part of resource columns (solar_batt4/8, wind_batt4/8). Cost = base VRE LCOE + fixed adder ($30 solar, $25 wind). Dispatch pre-baked in hybrid profiles.

**Layer 2 — Standalone storage dispatch:** 4 types with net cost = LCOE − revenue credit. **Storage percentages are expressed as capacity (MW) as a percent of annual demand (MWh/8760).** E.g., `battery_dispatch_pct = 5.0` means battery capacity equal to 5% of average hourly demand.

|Type    |Duration|Efficiency|Window    |
|--------|--------|----------|----------|
|battery |4h      |85%       |24h       |
|battery8|8h      |85%       |48h       |
|LDES    |100h    |50%       |168h (7d) |
|H2      |1000h   |35%       |720h (30d)|

Revenue credits are per-ISO (from `STORAGE_REVENUE_CREDITS` in pipeline_config).

Storage dispatch percentages are generated as part of each candidate mix and scored through the fused Numba kernel (charge/discharge simulation across 8,760 hours).

### 12. Scoring Kernel ✅

Each candidate mix is scored inline via the fused Numba kernel. **Two outputs per mix:**

1. **hourly_match_score** = `sum(min(clean_supply_h, demand_h)) / sum(demand_h) × 100` — the CFE percentage. This is demand-weighted MWh matching, not hours matched.
1. **resid_norm_p9997** = 99.97th percentile of the normalized hourly residual gap — used for gas sizing.

**Dropped:** `clean_peak_hour_mw` (redundant with resid_norm_p9997).

The kernel computes both values in a single pass per mix with no intermediate arrays. Reuses the existing fused Numba kernel from `step2_3a_regenerate_peakclean.py`.

### 13. Output Format ✅

**One JSON file per run, tagged with:**

```
{ISO}__pathway{A|B}__{dim_key}__{YYYYMMDD_HHMMSS}.json
```

Where `dim_key` encodes the scenario combo (e.g., `firm_M__ccs_M__tx_Med__dg_Med`).

**File structure:**

```json
{
  "schema_version": "3.0",
  "iso": "NYISO",
  "pathway": "A",
  "dim_key": "firm_M__ccs_M__tx_Med__dg_Med",
  "timestamp": "20260426_143022",
  "config": { ... },
  "beam_index": 0,
  "beam_archetype": "solar-led",
  "threshold_snapshots": [
    {
      "threshold_pct": 50.0,
      "year_achieved": 2030,
      "achieved_cfe_pct": 50.3,
      "resource_mix_pct": { "solar": 25.1, "wind": 12.3, ... },
      "storage_dispatch_pct": { "battery": 2.1, ... },
      "gas_need_mw": 12450.0,
      "peak_gas_mw": 15200.0,
      "stranded_vs_peak_mw": 2750.0,
      "cost_to_date_undiscounted_usd": 1.23e9,
      "cost_to_date_npv_5pct_usd": 0.98e9,
      "cost_to_date_npv_7pct_usd": 0.87e9,
      "cost_to_date_npv_9pct_usd": 0.78e9,
      "curtailment_pct": { "solar": 0.12, "wind": 0.03, ... }
    },
    ...
  ]
}
```

Each beam (N=5 default) produces its own file. All threshold snapshots (up to 20) are nested in one file per beam.

### 14. On-the-Fly Candidate Generation ✅

**Stage 1 — First threshold (uses existing data):**

1. Load existing Step 2.1 EF parquets for the first threshold band above baseline CFE.
1. Score all mixes through the Numba kernel (including storage dispatch on every mix).
1. Filter to mixes in the target CFE band.
1. Cluster into archetypes by dominant resource family:
- Solar-led (solar family > 50% of clean supply)
- Wind-led (wind family > 50% of clean supply)
- Balanced VRE (no single family dominates)
- For Pathway B additionally: nuclear-heavy, offshore-heavy
1. Pick cheapest from each archetype, up to N (default 5).
1. If fewer archetypes than N, fill remaining slots with next-cheapest from the most populated archetype.
1. Stage 1 sample count is adjustable.

**Stage 2 — On-the-fly (all subsequent thresholds):**

For each of the N seed mixes, independently at each year:

**Step A — Compute marginal CFE yield per resource (recomputed every year):**

1. Start from current ratchet floor mix.
1. Add +1pp of each available resource individually. Score each perturbation through the Numba kernel.
1. Compute marginal CFE yield = `(perturbed_score - floor_score) / 1pp` for each resource.
1. These marginals capture the current state of the grid — solar at 80% penetration has much lower marginal yield than solar at 20% due to curtailment. Recomputing each year tracks this correctly.

**Step B — Size the search ceiling per resource:**

1. Determine the year’s CFE increment needed: `target_cfe - floor_cfe`.
1. For each resource, set ceiling = `floor + increment / marginal_yield × safety_factor`. Resource shares are in % of annual demand (same unit as EF parquets and pipeline_config grid mix shares). Resources with near-zero marginal yield get minimal headroom. Resources with high yield get more exploration space.
1. Clamp ceilings at hard caps where applicable (CCS cap, geo cap, uprate cap for Pathway B).

**Step C — Sample and score:**

1. Generate candidate mixes via Latin Hypercube sampling within the per-resource [floor, ceiling] bounds. Default ~5,000 candidates, adjustable.
1. Score ALL candidates through the fused Numba kernel, including standalone storage dispatch on every mix. Storage dispatch percentages are capacity as a percent of annual demand — same unit convention as the existing kernel and EF parquets. The storage kernel discovers resource+storage interactions naturally — no need for separate joint perturbation testing.
1. Filter to candidates in the valid CFE band: `[year_target, next_year_target)`. **The winning mix’s CFE cannot exceed the next year’s CFE target.** This enforces truly incremental year-by-year buildout — no leapfrogging.
1. If too few hits, resample with adjusted bounds (expand safety factor, shift ceiling range).
1. Pick cheapest feasible candidate as this year’s winner.
1. Update ratchet floors.
1. Write a threshold snapshot whenever the achieved CFE crosses a new EF band threshold.

**Key constraint — CFE ceiling per year:**
The winner’s `hourly_match_score` must satisfy:

```
year_cfe_target ≤ winner_cfe < next_year_cfe_target
```

This prevents the optimizer from jumping to a high-CFE mix in an early year and coasting. Every year must make an incremental step. This is critical for producing meaningful year-by-year buildout trajectories.

**No pre-computed pool, no peakclean sidecars, no augmentation scripts needed.**

### 15. Fossil Retirement ✅

**Dropped entirely.** No coal→oil→gas displacement sequence. Existing gas capacity is a fixed constant per ISO. Gas need is purely the residual gap at the worst hour minus existing capacity.

### 16. Hydro Treatment ✅

**Frozen at baseline for both pathways.** Hydro comes from existing dams — you can’t build more. Treated identically to clean_firm in Pathway A: not a decision variable for the optimizer.

### 17. Frozen Resource Convention ✅

**Frozen at baseline absolute TWh, not baseline share.** As demand grows, the frozen resource’s share of demand shrinks. Applies to:

- Hydro (both pathways): frozen at `baseline_share% × baseline_demand_TWh`
- Clean_firm existing (Pathway A): frozen at `baseline_share% × baseline_demand_TWh`

Example: NYISO hydro = 15.9% × 151.6 TWh = 24.1 TWh, fixed for all 26 years regardless of demand growth.

### 18. Ratchet Unit Convention ✅

**Ratchet binds on absolute TWh (physical capacity locked), not percentage share.**

When generating candidates for year Y+1, the ratchet floor for each resource is converted to % terms:

```
floor_pct_Y+1 = floor_TWh / demand_TWh_Y+1
```

Since demand grows, the same locked TWh corresponds to a slightly lower %. The solver CAN reduce a resource’s % share while honoring the absolute TWh ratchet — the physical capacity stays built, it just represents a smaller fraction of growing demand.

This is consistent with the real-world constraint: you don’t tear down capacity you built, but you don’t need to maintain its share of a growing pie.

### 19. LCOE Vintage Locking ✅

**Track each year’s incremental build at its vintage cost.** Simple list per beam — each year records the incremental TWh added per resource and the LCOE locked at that year’s price.

Annual cost = sum over all vintages of `vintage_TWh × vintage_LCOE`. For flat-cost resources (solar, wind, hybrids) this is trivial since LCOE doesn’t change. For learning-curve resources (Pathway B: nuclear, CCS, offshore, geothermal), early-year builds stay at their expensive FOAK-era cost even after NOAK is reached.

Format: list of `(year, resource, incremental_TWh, locked_LCOE)` tuples. Not a full gas-style ledger — no retirement, no asset life tracking.

### 20. Fallback Cascade ✅

When no candidate lands in the CFE band `[year_target, next_year_target)`:

1. **Resample at 2× candidates** (e.g., 10,000 instead of 5,000)
1. **Resample at 4× candidates** (e.g., 20,000)
1. **Flag infeasible and stop the beam.** This beam is dead — its trajectory is truncated at the last successful year. Surviving beams continue.

No band widening — the CFE ceiling constraint is hard.

### 21. Final Year Ceiling ✅

**2050 ceiling = 100%.** The final year’s valid CFE band is `[99.4, 100.0)`. No year 2051 exists, so no next-year target to constrain against.

### 22. Safety Factor for Ceiling Sizing ✅

**1.5× (tight).** `ceiling = floor + increment / marginal_yield × 1.5`. Accepts the risk of occasionally missing the band — the fallback cascade (Decision 20) handles that case. Keeps samples concentrated in the realistic region rather than wasting them on implausibly large builds.

### 23. Cost Accounting for Winner Selection ✅

**Two modes, run both and compare:**

- **Mode 1 — Incremental clean cost only:** Cost of all new resource capacity (VRE, hybrids, storage, nuclear, CCS, etc.) above the ratchet floors. Gas impact ignored. Favors the cheapest clean build regardless of reliability implications.
- **Mode 2 — Incremental clean cost minus gas savings:** Same as Mode 1, but subtracts the reduction in gas cost relative to the prior year. `net_cost = incremental_clean_cost - (prior_year_gas_cost - this_year_gas_cost)`. Favors candidates that balance clean build cost against gas savings.

Both modes run on the same candidate set — same mixes, same scores, different ranking. Each mode produces its own set of beams and output files.

Storage cost is part of incremental clean cost — if a candidate adds more standalone storage above the prior year’s floor, that cost is included.

### 24. Curtailment Computation ✅

**Compute curtailment at every threshold snapshot.** Requires one additional kernel call per snapshot (cheap). Curtailment per resource = `max(0, clean_supply_h - demand_h)` allocated pro-rata to each resource’s contribution at that hour, summed across 8,760 hours, divided by total resource generation.

### 25. LHS Sampling Dimensions ✅

Given frozen hydro and clean_firm decisions, the free dimensions for Latin Hypercube sampling are:

**Pathway A — 10 dimensions:**

- 6 resource shares: solar, wind, solar_batt4, solar_batt8, wind_batt4, wind_batt8
- 4 storage dispatch %: battery, battery8, LDES, H2

**Pathway B — 12 dimensions:**

- 6 resource shares: solar, wind, solar_batt4, solar_batt8, wind_batt4, wind_batt8
- 1 offshore_wind (where available)
- 1 clean_firm total (tranched for costing per Decision 10, but sampled as single variable)
- 4 storage dispatch %: battery, battery8, LDES, H2

Hydro and existing clean_firm (Pathway A) are fixed inputs, not sampled. CCS and geothermal are internal to the clean_firm tranche decomposition (Pathway B) — not separate sampling dimensions.

### 26. Baseline Clean Firm Floor ✅

**Both pathways enforce a minimum clean_firm floor at baseline TWh.** Existing nuclear doesn’t disappear.

- **Pathway A:** clean_firm frozen at baseline TWh (no growth). Share shrinks as demand grows.
- **Pathway B:** clean_firm can grow above baseline (nuclear new-build), but never drops below baseline TWh. The floor is `baseline_share% × baseline_demand_TWh / current_year_demand_TWh × 100` each year.

This ensures Pathway B starts from the same existing nuclear base as Pathway A and builds on top of it.

-----

## Changes from Current Step 2.3

|Feature                       |Current                                |Rebuild                                                                       |
|------------------------------|---------------------------------------|------------------------------------------------------------------------------|
|Mix source                    |Pre-computed pool (millions of mixes)  |Stage 1: EF parquets. Stage 2: marginal-yield-bounded LHS generation          |
|CFE band constraint           |Floor only (no ceiling)                |Floor AND ceiling: `[year_target, next_year_target)` — truly incremental      |
|Search bounds                 |Fixed caps per resource                |Marginal-yield-sized ceilings recomputed each year (1.5× safety factor)       |
|Storage scoring               |Pre-baked in pool                      |Numba kernel runs storage dispatch on every candidate                         |
|Hydro                         |Optimizable                            |Frozen at baseline TWh (both pathways)                                        |
|Frozen resources              |Frozen share (grows with demand)       |Frozen TWh (share shrinks with demand growth)                                 |
|Ratchet unit                  |Ambiguous                              |Explicit absolute TWh, converted to % via demand growth                       |
|LCOE tracking                 |Current-year pricing                   |Vintage-locked: each year’s build pays its build-year LCOE forever            |
|Winner selection              |Single cost mode                       |Dual mode: Mode 1 (incremental clean only) vs Mode 2 (clean minus gas savings)|
|Curtailment                   |Endpoint only                          |Every threshold snapshot                                                      |
|Fallback on miss              |Hold previous winner                   |Cascade: 2×, 4× resample, then flag infeasible and stop beam                  |
|Pathways                      |4 (P1, P2a, P2b, P3)                   |2 (A = P1 no offshore, B = P3)                                                |
|Endpoints                     |95% and 99.9%                          |99.9% only                                                                    |
|Learning curves               |Linear FOAK→NOAK                       |Wright’s Law concave (Pathway B); flat (Pathway A)                            |
|Solar/wind/hybrid/storage caps|Per-ISO caps enforced                  |**No caps**                                                                   |
|CCS/geo/uprate caps           |Per-ISO                                |**Kept**                                                                      |
|Gas sizing                    |Vintage ledger + merit-order retirement|Simple residual gap, fixed existing capacity                                  |
|Fossil retirement             |Coal→oil→gas displacement              |**Dropped**                                                                   |
|Stranding                     |Per-vintage tracking                   |Simple peak-year minus threshold-year delta                                   |
|Scoring outputs               |3 (hourly_match, resid, clean_peak_mw) |2 (hourly_match, resid) — clean_peak_mw dropped                               |
|Output format                 |One JSON per endpoint                  |One JSON per beam, snapshots at every threshold                               |
|Cost scenarios                |Single run                             |Batchable across firm/ccs/tx/demand dimensions                                |
|Stage 1 source                |Same pool as Stage 2                   |Existing Step 2.1 EF parquets                                                 |
|Beam width                    |Default 1, adjustable                  |Default 5, adjustable                                                         |

-----

## Baseline Grid Mix Shares (reference)

|ISO  |clean_firm|solar|wind|hydro|offshore|ccs|Total clean|
|-----|----------|-----|----|-----|--------|---|-----------|
|CAISO|7.9       |22.3 |8.8 |9.5  |0       |0  |48.5%      |
|ERCOT|8.6       |13.8 |23.6|0.1  |0       |0  |46.1%      |
|PJM  |32.1      |2.9  |3.8 |1.8  |0       |0  |40.6%      |
|NYISO|18.4      |0.0  |4.7 |15.9 |0       |0  |39.0%      |
|NEISO|23.8      |1.4  |3.9 |4.4  |0       |0  |33.5%      |
|MISO |13.1      |2.1  |14.5|1.6  |0       |0  |31.3%      |
|SPP  |5.2       |0.4  |37.1|4.3  |0       |0  |47.0%      |

## Key Cost Tables (Medium level, reference)

**VRE LCOE ($/MWh):**

|ISO  |Solar|Wind|
|-----|-----|----|
|CAISO|60   |73  |
|ERCOT|54   |40  |
|PJM  |65   |62  |
|NYISO|92   |81  |
|NEISO|82   |73  |
|MISO |62   |43  |
|SPP  |57   |37  |

**Hybrid adders:** solar+batt = solar LCOE + $30; wind+batt = wind LCOE + $25.

**Nuclear FOAK → NOAK (Pathway B, P3 Wright’s Law window 2030→2035):**

|ISO  |FOAK|NOAK (Low)|
|-----|----|----------|
|CAISO|175 |70        |
|ERCOT|169 |68        |
|PJM  |200 |72        |
|NYISO|212 |75        |
|NEISO|206 |73        |
|MISO |194 |70        |
|SPP  |175 |68        |

**Offshore wind FOAK/NOAK ($/MWh):**

|ISO  |FOAK|NOAK Med|
|-----|----|--------|
|CAISO|250 |72      |
|PJM  |129 |62      |
|NYISO|144 |65      |
|NEISO|136 |63      |

**CCS FOAK ($/MWh, Pathway B only):**

|ISO  |FOAK|Cap TWh|
|-----|----|-------|
|CAISO|138 |25     |
|ERCOT|110 |200    |
|PJM  |122 |125    |
|NYISO|154 |0      |
|NEISO|146 |0      |
|MISO |115 |200    |
|SPP  |106 |50     |

**TX adders (Medium, $/MWh):**

|ISO  |Solar|Wind|Nuclear|CCS|Offshore|
|-----|-----|----|-------|---|--------|
|CAISO|3    |8   |3      |2  |20      |
|ERCOT|3    |6   |2      |2  |0       |
|PJM  |5    |10  |3      |3  |11      |
|NYISO|7    |14  |5      |4  |15      |
|NEISO|6    |12  |4      |3  |13      |
|MISO |4    |9   |3      |2  |0       |
|SPP  |3    |7   |2      |2  |0       |

**Existing Gas (fixed for entire run):**

|ISO  |Capacity MW|GAF |Wholesale $/MWh|Peak MW|
|-----|-----------|----|---------------|-------|
|CAISO|37,000     |0.88|30             |43,860 |
|ERCOT|55,000     |0.83|27             |83,597 |
|PJM  |75,000     |0.82|34             |160,560|
|NYISO|18,000     |0.82|42             |31,857 |
|NEISO|14,000     |0.85|41             |25,898 |
|MISO |68,000     |0.84|30             |127,125|
|SPP  |32,000     |0.84|25             |54,368 |