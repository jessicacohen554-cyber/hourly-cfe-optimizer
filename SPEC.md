# Advanced Sensitivity Model — Complete Specification

> **Authoritative reference for all design decisions.** If a future session needs context, read this file first.
> Last updated: 2026-02-21.

## Current Status (Feb 21, 2026)

### LMP Module — v9 Calibration Complete (Feb 21, 2026)

**Completed:**
- `dispatch_utils.py` — shared dispatch module (constants, profiles, battery/LDES dispatch, fossil retirement, hourly dispatch cache)
- `compute_lmp_prices.py` — core LMP engine with:
  - Merit-order fossil stack: PJM Manual 15 cost-based offer formula (HR × fuel + VOM + CO2 + 10% adder)
  - Heat rates calibrated to PJM SOM 2024 benchmarks (CCGT 7.0, CT 10.5, coal 10.0 MMBtu/MWh)
  - CO2 allowance costs (RGGI-weighted: L/M/H = $3/$5.50/$14 per ton)
  - 10% adder per PJM market rules ($2.00/MWh contribution per SOM 2024)
  - VOM split: variable maintenance + variable operations (SOM 2024: $3.18 + $1.43 = $4.61 fleet avg)
  - Load-dependent heat rate ramp (15% quadratic) for within-band price variation
  - Demand-quantile pricing: congestion, scarcity tail, off-peak compression
  - ISO-specific price formation: PJM (RPM), ERCOT (ORDC), CAISO (RA), NYISO (ICAP), NEISO (FCM + winter gas)
  - Archetype deduplication: (mix, fuel_level, threshold) → ~7,800 unique per ISO
  - Dispatch cache: append-mode NPZ per ISO, shared with recompute_co2.py
- `calibrate_lmp_model.py` — validation framework with embedded PJM IMM/EIA reference data
- `recompute_co2.py` — refactored to import from dispatch_utils.py (identical behavior)

**PJM v9 Calibration Results (2024 baseline, Medium fuel/CO2):**

| Metric | Synthetic | Target | Delta | Status |
|---|---|---|---|---|
| Avg LMP | $36.69 | $34.70 | +6% | GOOD |
| Peak avg | $38.82 | $42.00 | -8% | GOOD |
| P10 | $20.00 | $18.00 | +$2 | FAIR |
| P25 | $24.37 | $23.00 | +$1.37 | GOOD |
| P50 | $31.88 | $30.00 | +$1.88 | GOOD |
| P75 | $38.96 | $42.00 | -$3 | GOOD |
| P90 | $50.21 | $55.00 | -$5 | GOOD |
| Scarcity hours | 102 | 100 | +2 | GOOD |
| Negative hours | 246 | 200 | +46 | FAIR |
| Volatility | $31.03 | $25.00 | +$6 | FAIR |

- Calibration report: "No major adjustments needed"
- Known limitation: off-peak avg ($34.75 vs $28) — no unit commitment/min-gen constraints

**Marginal Costs at Medium (with 10% adder + CO2):**
- Gas CCGT: $33.04/MWh (PJM 2024 RT avg: $33.74 — matches within 2%)
- Coal: $36.55/MWh
- Gas CT: $49.25/MWh
- Oil CT: $131.81/MWh

**Track Sweep Status:**
- CAISO: Complete (NB + replace, 12 thresholds × 209,952 scenarios each)
- ERCOT: NB partial (10/13 thresholds), replace not started
- PJM, NYISO, NEISO: Not started
- Checkpoint: `data/track_checkpoint.json` (partial results)
- Parquet export: `dashboard/track_scenarios.parquet` (CAISO only)

**Next Steps:**
- Finish track sweep (ERCOT + PJM + NYISO + NEISO)
- Run LMP model on full PJM ECF scenarios (all thresholds × fuel sensitivities)
- Optional: fetch PJM hourly LMP data from Data Miner 2 for distribution matching

### LMP Price Calculation & Existing vs New-Build Analysis (Feb 20, 2026)

**Goal**: Separate existing-generation vs new-build pricing to enable "with vs without existing" analysis. Shows cost of replacing nuclear/hydro, asset stranding risk, and true greenfield costs.

#### Decision 1: Hydro Treatment — Both Scenarios (1C)
- **With hydro**: Hydro included as existing wholesale-priced resource. Shows cost advantage of existing hydro and its value for hourly matching. Better view of cost to replace nuclear.
- **Without hydro**: Mixes with hydro=0 in the EF. Shows new-build requirement for hourly matching without hydro, procurement impact, and potential asset stranding if hydro is curtailed or unavailable.
- **Implementation**: Step 2 EF existing clean floor removed → hydro=0 mixes now available. Step 3 can filter to hydro=0 or hydro>0 mixes for comparison.

#### Decision 2: Nuclear Uprates — Both Scenarios
- **With uprates**: Tranche 1 (uprate) pricing preserved. Better for hourly matching since uprates are cheap dispatchable capacity.
- **Without uprates**: Tranche 1 disabled (uprate_cap=0). All new clean firm priced at tranche 2 (geothermal) / tranche 3 (new-build nuclear vs CCS). Provides better view of true new-build replacement cost.
- **Implementation**: `uprate_mode` flag in Step 3 cost function. When 'off', `uprate_cap=0` and all new CF flows directly to geothermal/new-build tranches.

#### Decision 3: Below-Floor Mix Recovery
- **Problem**: Step 2 Phase 0 (existing clean floor filter) removed mixes that under-allocated existing generation. This filtered out hydro=0 and low-clean-firm mixes needed for greenfield analysis.
- **Approach**: Temp recovery script reads PFS cache, inverts the floor filter, recovers below-floor mixes, runs Pareto procurement, and merges into existing `pfs_post_ef.parquet`.
- **Step 2 update**: Remove Phase 0 filter for future runs (kept as dead code for reference). No full PFS re-run needed — temp script recovers the delta.
- **Script**: `recover_below_floor.py` (one-time use, can be deleted after merge)

#### Decision 4: LMP Integration — Wholesale + LCOE Hybrid (4A)
- **Approach**: Current architecture preserved. Existing generation priced at regional wholesale (LMP-informed). New-build priced at LCOE + transmission adder.
- **No change to pricing engine structure** — the existing/new split already computed in Step 3 `price_mix_batch()`. This decision confirms the hybrid approach is correct and no full nodal LMP model is needed.

#### Decision 6: LMP Module Architecture (Feb 21, 2026, updated v9)
- **Pipeline position**: Downstream of Step 4, reads ECF base case from `overprocure_scenarios.parquet`
- **Cost-based offer formula** (PJM Manual 15): `MC = (Heat Rate × Fuel Price + VOM + CO2 Rate × CO2 Price) × (1 + 10% Adder)`
- **Heat rates** (PJM SOM 2024 benchmarks): Coal 10.0, CCGT 7.0, CT 10.5, Oil 10.5 MMBtu/MWh
- **VOM** (SOM 2024 decomposition: $3.18 maintenance + $1.43 operations): Coal $5.50, CCGT $3.50, CT $5.00, Oil $6.00 $/MWh
- **CO2 allowance** (RGGI-weighted for PJM): Low $3/ton, Medium $5.50/ton, High $14/ton. SOM 2024: $1.94/MWh contribution.
- **10% adder**: PJM market rules allow 10% markup above cost-based offers. SOM 2024: $2.00/MWh contribution.
- **Stack walk**: `np.searchsorted` step function with load-dependent heat rate ramp (15% quadratic)
- **Demand-quantile pricing**: High-demand congestion adder (P75+), scarcity tail (P95.5+), mid-low compression (P10-P70), negative pricing (P0-P10)
- **ISO price formation**: PJM RPM ($2K cap), ERCOT ORDC (VOLL×LOLP, $5K cap), CAISO RA (-$60 floor), NYISO ICAP, NEISO FCM (+$13.13 winter gas)
- **Installed capacity**: EIA 860 actuals (PJM 127.8 GW, ERCOT 80 GW, CAISO 47 GW, NYISO 28 GW, NEISO 16 GW)
- **Fuel prices**: Low/Medium/High sensitivity (coal $2.00-2.50, gas $2.00-6.00, oil $8.00-13.00 $/MMBtu)
- **Dispatch cache**: Shared with recompute_co2.py via dispatch_utils.py, append-mode NPZ per ISO
- **Calibration reference**: PJM IMM 2024 SOM: RT LW avg $33.74/MWh, total wholesale $55.54/MWh
- **LMP runs on ECF track only** (base case with existing clean floor). NB/CTR tracks are separate analysis, not priced through LMP.
- **Data sources**: PJM Manual 15 Rev. 47, Monitoring Analytics 2024 SOM, EIA Electric Power Annual Table 8.1, EPA eGRID 2022

#### Decision 5: Three Analysis Tracks (5C — Updated Feb 21, 2026)
Three distinct tracks with standardized naming:

**Track 1 — ECF (Existing Clean Floor)**: Base case
- The standard optimizer output with existing generation credited at wholesale
- Source: `overprocure_scenarios.parquet` (baseline results)
- LMP module runs on this track first
- Files/caches use `ecf_` prefix

**Track 2 — NB (New-Build)**: What hourly matching incentivizes
- Hydro: **excluded** (hydro=0 mixes only)
- All existing clean: **zeroed** (GRID_MIX_SHARES = 0 for CF, solar, wind, CCS)
- Uprates: **on** (uprate tranche active — cheapest new-build option)
- Purpose: What does hourly matching incentivize you to BUILD from scratch?
- Files/caches use `nb_` prefix

**Track 3 — CTR (Cost to Replace)**: True greenfield replacement cost
- Hydro: **included** (existing floor, wholesale-priced)
- All other existing clean: **zeroed** (CF, solar, wind, CCS all priced as new-build)
- Uprates: **off** (uprate_cap=0, no uprate tranche)
- Purpose: True greenfield cost of replacing all existing clean generation
- Files/caches use `ctr_` prefix

**Naming convention**: All file names, cache keys, code comments, and output fields use ECF/NB/CTR abbreviations consistently. File rename deferred until NB/CTR sweep completes.

**Output**: Data files only. No research paper update yet — discuss findings with user first, then write.
**Architecture**: Step 3 runs 2 additional passes per (ISO, threshold):
  - Pass "NB": filter EF to hydro=0, uprate tranche on → new-build hourly matching results
  - Pass "CTR": full EF (hydro≤existing), uprate tranche off → replacement cost results
  - Plus existing pass "ECF": full EF, all features on → current behavior (preserved)

### Columnar JSON Format for Feasible Mixes (Feb 19, 2026)

**Problem**: `feasible_mixes` in `overprocure_results.json` stored as array of dicts — each mix repeated key names (`resource_mix`, `clean_firm`, `solar`, etc.) across potentially 1.78M entries, inflating JSON from ~40 MB to ~312 MB.

**Decision**: Option 2 — Columnar format. Store as `{col_name: [values...]}` instead of `[{col_name: val}, ...]`.

**Format**:
```json
"feasible_mixes": {
  "clean_firm": [50, 60, ...],
  "solar": [25, 20, ...],
  "wind": [...],
  "ccs_ccgt": [...],
  "hydro": [...],
  "procurement_pct": [...],
  "hourly_match_score": [...],
  "battery_dispatch_pct": [...],
  "ldes_dispatch_pct": [...]
}
```

**Measured savings**: 81% reduction per threshold group (98K → 18K bytes for 510 mixes). Projected ~312 MB → ~40 MB at full 1.78M mix scale.

**Files changed**:
- `step3_cost_optimization.py` — writes columnar format
- `step5_compressed_day.py` — reads both columnar (new) and row (legacy) formats
- `generate_shared_data.py` — reads both columnar (new) and row (legacy) formats
- Dashboard JS (`shared-data.js`) — already used compact arrays `[cf, sol, wnd, ccs, hyd, proc, match, bat, ldes]`; no change needed

**Backward compat**: Step 5 and generate_shared_data both auto-detect format (`isinstance(fmixes, dict)` vs `isinstance(fmixes, list)`).

### Compressed Day Chart: Curtailment Stacking Fix (Feb 19, 2026)

**Problem**: Curtailment was anchored to the demand line and stacked upward, creating a visual gap between the top of matched generation and the bottom of curtailment. On mobile, this made it look like curtailment was floating disconnected from the generation it belongs to.

**Fix**: Curtailment now stacks from `matchedTotal` (top of generation stack) upward, keeping it flush against the generation area. The demand line cuts through as a visual boundary — area above demand = true curtailment, area between matchedTotal and demand = unmatched gap.

### New Toggle Architecture Decisions (Feb 19, 2026)

Three new Step 3 cost model changes — no Step 1 physics re-run needed:

1. **CCS separated from Firm Gen toggle** — CCS gets its own L/M/H toggle (maturity-based: L=mature/low capex, H=immature/high capex) plus a binary 45Q On/Off switch. 6 CCS cost states total (3×2).
2. **Geothermal toggle (CAISO only)** — L/M/H based on published data (NREL ATB, USGS, Lazard). 5 GW cap (~39 TWh/yr) from USGS identified hydrothermal. After cap, remaining clean firm filled by cheapest of nuclear new-build vs CCS (toggle-dependent). Non-CAISO ISOs have zero geothermal resource — toggle hidden.
3. **Nuclear new-build Low target = $70/MWh** — nth-of-a-kind SMR deployment target. Regional variation: $68-75/MWh at Low.
4. **CAISO clean firm merit order**: Existing → uprates → geothermal (capped) → cheapest of nuclear/CCS (toggle-dependent)
5. **Non-CAISO clean firm merit order**: Existing → uprates → cheapest of nuclear/CCS (toggle-dependent)

**Sensitivity space expanded**: 324 → 5,832 (non-CAISO) / 17,496 (CAISO) combos. All Step 3 arithmetic — minutes, not hours.

### Demand Growth in Resource Mix Pricing (Feb 19, 2026)

**Decision**: Demand growth dynamically scales resource mix pricing. Existing generation stays flat in absolute TWh — as demand grows, existing's share of grown demand shrinks, requiring more new-build.

**Approach**: 1C (affects real pricing) + 2C (target year + growth rate as parameters) + 3C (client-side repricing handles it).

**Mechanics**:
- `grownDemandTwh = baseDemandTwh × (1 + annualRate)^(targetYear − 2025)`
- Existing share rescaled: `existingPctGrown = existingPct × (baseDemandTwh / grownDemandTwh)`
- More new-build fills the gap → higher costs at longer horizons / higher growth rates
- Growth rates per ISO from DEMAND_GROWTH_RATES (L/M/H): CAISO 1.4-2.5%, ERCOT 2.0-5.5%, PJM 1.5-3.6%, NYISO 1.3-4.4%, NEISO 0.9-2.9%
- Target year from dashboard selectors (interim: 2027-2035, longterm: 2036-2050)
- `priceMix()` accepts optional `targetYear` and `growthRate` parameters; defaults to 2025/0 (no growth = current behavior)
- Base year fixed at 2025 (snapshot year)

**Implication**: Same physical feasible mix can cost significantly more at a 2040 target than at 2028, because more of the mix must be new-build to replace the shrunken existing share. This is correct — the resource mix needs to adapt dynamically to absolute TWh.

### v4.0 Fresh Rebuild — Decisions Locked (Feb 19, 2026)

Complete optimizer rebuild with new architecture. All 9 design decisions + 5 efficiency optimizations locked below.

#### Design Decisions

| # | Decision | Choice | Detail |
|---|----------|--------|--------|
| 1 | Grid search strategy | **1C — Adaptive** | Start at 5% step, identify promising regions, refine to 1%. Replaces 3-phase 10%→5%→1%. |
| 2 | Solution output | **2B — Pareto frontier** | 3-5 points per mix along procurement/storage tradeoff (not single-point optimal). |
| 3 | Procurement bounds | **3C — Threshold-adaptive** | Narrow bounds at low thresholds (e.g., 100-110% at 50%), wider at high (100-150% at 99-100%). |
| 4 | min_dispatchable constraint | **4B — Drop it** | No dispatchable floor. Let physics prove/disprove — constraint was potentially biasing results. |
| 5 | Thresholds | **5D — 13 total** | Current 10 + 50%, 60%, 70%. Full list: 50, 60, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 100. |
| 6 | CCS-CCGT resource | **6D — Collapse into Clean Firm** | Merge CCS into Clean Firm allocation. Reduces resource space from 5D to 4D. CCS retains its own cost profile and dispatch characteristics within the merged allocation — the optimizer determines sub-allocation internally. |
| 7 | Storage parameters | **7A — Keep current** | Battery: 4hr Li-ion, 85% RT, daily cycle. LDES: 100hr iron-air, 50% RT, 7-day window. |
| 8 | Output format | **8C — Both** | JSON (backward compat) + Parquet (analytics). |
| 9 | Numba acceleration | **9C — Optional** | Try Numba JIT, fall back to NumPy if install fails. |

#### Efficiency Architecture (Fresh Rebuild)

| ID | Optimization | Description | Expected Speedup |
|----|-------------|-------------|-----------------|
| A | ISO parallelism | Run all 5 ISOs in parallel on 16 cores (3 cores/ISO) | 4-5× |
| B | Vectorized battery dispatch | Replace Python `for day in range(365)` with NumPy reshape + vectorized ops | 3-5× on storage scoring |
| C | Batch mix evaluation | Evaluate batches of mixes simultaneously via matrix ops: `(N,4) @ (4,8760)` | 5-10× on grid search |
| D | Numba JIT (try/fallback) | Compile storage scoring to machine code; fall back to B+C if install fails | 10-50× on storage (if available) |
| F | Shared memory cache | `multiprocessing.shared_memory` for parallel ISO workers to share data | Enables A |

**Scope**: Step 1 only (physics). No cost model — the optimizer generates the feasible solution space (all viable resource mixes per threshold×ISO). Cost sensitivities (5,832 paired-toggle scenarios) applied in Step 3 cost optimization. This reduces from 21,060 cost-coupled optimizations to 65 physics-only sweeps (13 thresholds × 5 ISOs), each finding the Pareto frontier of feasible mixes.

**Projected runtime**: ~1-3 min with Numba (installed successfully). Down from multi-hour current architecture.

**Cost model**: NOT in scope for this rebuild. Cost model will be updated separately with dynamic functionality. This optimizer produces the physics-only feasible solution space that the cost model will consume.

#### What needs building (fresh rebuild — Step 1 physics only)
- [ ] New optimizer with 4D resource space (Clean Firm absorbs CCS)
- [ ] ISO parallel execution with shared memory (A+F)
- [ ] Vectorized scoring functions (B+C)
- [ ] Numba JIT with fallback (D)
- [ ] Pareto frontier output (3-5 points per threshold×ISO)
- [ ] 13-threshold sweep with adaptive procurement bounds
- [ ] JSON + Parquet dual output of feasible solution space

### 4-Step Pipeline Architecture

The optimizer runs as a 4-step pipeline. Each step is independent — only re-run the step whose inputs changed.

| Step | Script | Name | What It Does | When to Re-run |
|------|--------|------|-------------|---------------|
| **Step 1** | `step1_pfs_generator.py` | **PFS Generator** | Generates the Physics Feasible Space (PFS). Sweeps 4D resource mixes × procurement × battery × LDES, evaluates hourly generation vs. demand, computes match scores, curtailment, storage dispatch. Produces 21.4M physics-validated mixes across 5 ISOs × 13 thresholds. | Only if dispatch logic, generation curves, or demand curves change. |
| **Step 2** | `step2_efficient_frontier.py` | **Efficient Frontier (EF)** | Extracts the efficient frontier from the PFS. Filters existing generation utilization, minimizes procurement per allocation, removes strictly dominated mixes. Reduces 21.4M → ~1.8M rows. | Only if PFS changes or filtering criteria change. |
| **Step 3** | `step3_cost_optimization.py` | **Cost Optimization** | Vectorized cross-evaluation of all EF mixes under 5,832 sensitivity combos. Merit-order tranche pricing for clean firm. Extracts archetypes and sweeps demand growth scenarios (25 years × 3 growth rates). | When cost assumptions, tranche caps, LCOE tables, or sensitivity toggles change. |
| **Step 4** | `step4_postprocess.py` | **Post-Processing** | NEISO gas constraint, CCS vs LDES crossover analysis, CO₂ calculations, MAC calculations. Produces final corrected results for the dashboard. | When Step 3 outputs change, or when CO₂ methodology changes. |

**Key acronyms**:
- **PFS** — Physics Feasible Space: the full set of physically valid resource mixes (Step 1 output, `data/physics_cache_v4.parquet`)
- **EF** — Efficient Frontier: the reduced set of non-dominated mixes (Step 2 output, `data/pfs_post_ef.parquet`)

**Post-processing scripts** (run after Step 4):

| Script | Name | What It Does |
|--------|------|-------------|
| `recompute_co2.py` | **CO₂ Dispatch-Stack Model** | Merit-order fuel retirement (coal→oil→gas). Coal/oil capped at 2025 absolute TWh (no new build). Returns weighted average displaced emission rate for CO₂ abated. Demand-growth-aware. |
| `compute_mac_stats.py` | **MAC Statistics** | 6 MAC metrics: average fan (P10/P50/P90), stepwise marginal, monotonic envelope, path-constrained. ANOVA sensitivity decomposition. Crossover analysis vs DAC/SCC/ETS benchmarks. |
| `generate_shared_data.py` | **Dashboard Data** | Extracts all results into `dashboard/js/shared-data.js`. SBTi milestone mapping, DAC trajectories, LCOE tables for client-side repricing. |

**Key principle**: Step 1 is expensive (hours of compute). Step 2 takes ~40 seconds. Steps 3–4 + post-processing are cheap (minutes). Changing cost assumptions only requires Steps 3–4 + post-processing.

**Data contract**: Step 3 must NOT change existing columns in shared-data.js or overprocure_results.json. Add new columns/fields as needed. This prevents recoding existing figures and dashboards.

### What was accomplished
- [x] Homepage (`index.html`) — 4 charts rendering with real data, region toggle pills, narrative sections
- [x] Carbon Abatement Dashboard (`abatement_dashboard.html`) — 3 charts (MAC, portfolio, ladder) fully rendering with hardcoded illustrative data + 4 stress-test toggles
- [x] Navigation site-wide: Home | Cost Optimizer | Analysis (CO₂ Abatement Analysis) | Research (Paper, Methodology, Policy, About)
- [x] "Back to Home" button on all non-home pages
- [x] Chart styling QA/QC on working charts (borderRadius 6, no grid lines, axis borders)
- [x] Merged methodology into research paper (Appendix B with 7 sub-sections)
- [x] Tagline: "Most climate solutions depend on" across all pages
- [x] **CO₂ methodology fixed**: Hourly fossil-fuel emission rates (eGRID per-fuel × EIA hourly mix) replacing flat rate
- [x] **Post-optimizer pipeline**: `recompute_co2.py`, `analyze_results.py`, `run_post_optimizer.sh`
- [x] **Multi-year data infra**: `fetch_eia_multiyear.py` (2021-2025 EIA API + DST + averaging)
- [x] **Phase 3 re-optimizer**: `optimize_phase3_only.py` (±5% neighborhood refinement)
- [x] **DST fix script**: `fix_dst_profiles.py` (UTC → local prevailing time conversion)
- [x] **Optimizer checkpointing**: Saves after each threshold + resume from checkpoint on restart
- [x] **Sequential ISO processing**: Runs one ISO at a time with incremental result saves (avoids OOM)

### Needs work (awaiting optimizer results)
- [ ] **Dashboard abatement section (`dashboard.html`)** — 5 paired toggles work, 4 core charts work. Abatement cost section has placeholder divs awaiting optimizer results data.

### Optimizer code — ready for run (all completed)
- [x] **Hydro caps** — 2025 actuals: CAISO 9.5%, ERCOT 0.1%, PJM 1.8%, NYISO 15.9%, NEISO 4.4%
- [x] **5-year profile averaging** — gen + demand shapes averaged 2021-2025 (leap year handled)
- [x] **DST-aware solar nighttime correction** — 6am-7pm local prevailing time, UTC offset adjusts during DST
- [x] **Nuclear seasonal derate** — monthly CF factors × nuclear share of clean_firm
- [x] **Nuclear uprate LCOE blending** — regional uprate share blends cheap uprates with new-build
- [x] **CCS 95% capture rate** — residual 0.0185 tCO2/MWh
- [x] **Capacity-constrained storage** — battery + LDES dispatch_pct = built capacity
- [x] **CO2 hourly dispatch attribution** — charge-side emission netting
- [x] **1-scenario checkpointing** — zero compute loss on interruption
- [x] **Wholesale fuel adjustments** — documented in §5.9 with per-ISO $/MWh table
- [x] **SPEC.md ↔ code audit** — 150+ values verified, 0 discrepancies

### Needs work (awaiting optimizer results)
- [ ] Dashboard abatement section (`dashboard.html`) — placeholder divs awaiting results
- [ ] Site content gap closure: incomplete pages need optimizer data
- [ ] Update narratives + research paper with new results

### MAC Methodology Decision (Feb 16, 2026) — Option B: Hybrid Stats + Path-Constrained Reference

**Problem**: Independent threshold optimization produces non-monotonic marginal abatement costs (MACs). Each threshold finds its own globally optimal portfolio; the "marginal cost" between thresholds compares two independently optimized systems, not incremental resource additions. This creates phantom MAC spikes ($887/ton NYISO 95→97.5%) and even negative abatement (NEISO 95→97.5%, ERCOT 97.5→99%) when the optimizer reshuffles the portfolio.

**Decision**: Option B — Hybrid approach combining statistical post-processing of existing 16,200 results with a small set of path-constrained optimization runs.

**Components**:
1. **Monotonic envelope** (stats): Convex hull of (cost, CO2) per ISO — filters rebalancing noise
2. **MAC uncertainty fan** (stats): P10/P50/P90 across 5,832 scenarios at each threshold
3. **Sensitivity decomposition** (stats): ANOVA on which toggles drive MAC variance
4. **Path-constrained reference MAC** (50 targeted runs): Force each threshold's mix to build on previous — monotonic by construction. One run per threshold × 5 ISOs at Medium costs.
5. **Visualization**: Central monotonic reference curve inside P10-P90 fan, with DAC/SCC horizontal bands

**Methodology statement**: "Path-dependent marginal costs (central line) with uncertainty characterized via factorial sensitivity analysis across 5,832 cost scenarios (shaded band)."

**Literature basis**: Systems MAC / MAC 2.0 (Evolved Energy/EDF 2021), scenario ensemble approach (Deane et al. 2020), conservation supply curve methodology (Meier & Rosenfeld 1982). Full lit review: `research/mac_methodology_lit_review.md`.

**Reference docs**:
- `research/mac_methodology_lit_review.md` — Full literature review with 17 key citations
- `research/optimizer_statistical_methodology.md` — Search space analysis, global optimum capture probability

### Marginal MAC Monotonicity Fix (Feb 16, 2026) — Two-Zone Approach

**Problem**: Stepwise marginal MAC (Δcost/ΔCO2 between consecutive thresholds) is wildly non-monotonic due to resource reshuffling. Current data oscillates by 2-10x between adjacent steps (e.g., CAISO P50: 214→116→475→138→290→305→347→340). Root cause: independent threshold optimization produces different optimal portfolios at each threshold — the "delta" between them measures portfolio switching cost, not incremental resource addition cost.

**Key insight**: Grid decarbonization holds to ~92.5% in all regions regardless of cost assumptions. Sub-90% marginal MAC granularity is noise from optimization artifacts, not economically meaningful signals.

**Decision**: Two-zone marginal MAC structure:

**Zone 1 — Grid Backbone (75% → 90%): Single aggregate marginal MAC**
- One value per (ISO, scenario): `MAC = (cost[90%] - cost[75%]) × demand / (CO2[90%] - CO2[75%])`
- Represents the cost per ton of grid backbone decarbonization
- No monotonicity issue (single value)

**Zone 2 — Last Mile (90% → 100%): Granular checkpoints with enforced monotonicity**
- 5 stepwise values: 90→92.5%, 92.5→95%, 95→97.5%, 97.5→99%, 99→100%
- Enforced non-decreasing: `step_mac[t] = max(raw_step_mac[t], step_mac[t-1])`
- Zone 1 aggregate MAC serves as floor for first Zone 2 step
- Convex hull interpolation for edge cases where ΔCO2 ≤ 0

**Result**: 6-value marginal MAC curve per (ISO, scenario):
```
[MAC_75→90, MAC_90→92.5, MAC_92.5→95, MAC_95→97.5, MAC_97.5→99, MAC_99→100]
```

**Fan chart fix**: Consistent scenario ranking (rank by total cost at 99%, select P10/P50/P90 scenarios, use their full curves) instead of independent per-step percentiles that mix different scenarios.

**Implementation plan**: See `PLAN_marginal_mac_fix.md` for detailed implementation steps and file-by-file changes.

### Optimizer Statistical Properties (Feb 16, 2026)

**Search architecture**: 3-phase hierarchical grid search (10% → 5% → 1% resolution)

**Global optimum capture probability**: >99.9%
- Phase 1 coarse grid covers all 32 piecewise-linear regions of the cost function
- P(all regions sampled) ≈ 99.5% from grid alone, >99.9% with edge-case seeds
- Lipschitz gap bound (Nesterov 2003): <$0.01/MWh in mix dimensions at 1% resolution

**Maximum sub-optimality**: ~$2-4/MWh (~1-3% of typical $50-150/MWh total)
- Mix dimensions (1% steps): <$0.58/MWh
- Storage dimensions (2% steps): <$3.64/MWh — dominant error source
- Procurement (1% steps): <$1.00/MWh

**Why grid search, not LP**: Non-convex problem (hourly min() in matching score, nonlinear storage dispatch). Standard energy models (TIMES, ReEDS, GenX, PyPSA) use LP, but our hourly matching + greedy storage dispatch cannot be linearized without accuracy loss. Grid search is appropriate because evaluations are cheap (~0.1ms vectorized numpy) and dimensionality is low (5-7 DOF).

**Warm-start bias**: Non-Medium scenarios start from Medium optimum + ±17pp reach. 4 extreme archetypes get full exploration. Cross-pollination covers remaining risk.

### Pipeline when optimizer completes
1. Run `recompute_co2.py` → hourly CO₂ correction
2. Run `analyze_results.py` → monotonicity, literature alignment, VRE waste, DAC inputs
3. Update dashboards with real data, update narratives
4. Path-constrained MAC runs (50 targeted optimizations)
5. Statistical post-processing: envelope, fan, ANOVA
6. DAC-VRE analysis, resource mix analysis
7. Commit + push

### Pre-Run QA/QC Gate (Mandatory Before Every Optimizer Run)
**This gate exists because**: a previous run wasted 3+ hours of compute due to incorrect hydro caps that weren't caught before launch. Every optimizer run is expensive — never launch without verifying assumptions first.

Before launching `step1_pfs_generator.py`, the following must be verified:
1. All decisions from the current conversation implemented in optimizer code
2. All decisions captured in SPEC.md
3. No open questions that could change optimizer logic, cost tables, or methodology
4. Code passes syntax check (`python -c "import py_compile; py_compile.compile(...)"`)
5. **Full assumptions audit**: verify ALL key assumptions (hydro caps, cost tables, resource constraints, dispatch logic, procurement bounds, storage parameters) match SPEC.md and real-world data
6. **Dry-run test**: imports, constants, data loading, checkpoint save/load round-trip
7. **Checkpoint system verified**: save/load/resume works, interval set appropriately
8. Present user with summary of verified assumptions before starting
9. User explicitly approves the run

### Generator Analysis & Policy Page Decisions (Feb 15)
- [x] **Tone down Constellation-specific narrative** — generic archetypes (nuclear-led, coal-heavy, gas-dominant). Removed "unfairness" language. Applied across targets.html, index.html, fleet-analysis.html, policy.html.
- [x] **Add GHG Protocol Scope 2 revision context** — deep-dive on targets.html: 4 quality criteria (temporal, deliverability, incrementality, SSS), hourly premium economics, convergence with SBTi.
- [x] **Add EPRI SMARTargets context** — targets.html: AT/QT framework, Ceres criticism, investor credibility debate, "both/and" resolution.
- [x] **Add hourly RPS discussion** — targets.html: hourly RPS as policy frontier, convergence with GHG Protocol + SBTi, demand-side pull for clean firm.
- [x] **Policy page: RPS + corporate demand under hourly matching** — SSS baseline, corporate participation scenario table (10-50% × 5 ISOs), clean premium projections by ISO.
- [x] **EAC scarcity analysis REWRITE (Feb 15)** — corrected SSS framework + interactive dashboard
- [x] **SSS pro-rata derate** — corporate EAC demand is incremental above SSS baseline allocation, not gross
- [x] **Demand-proportional RPS** — clean supply = RPS target % × projected demand (not independent growth rate)
- [x] **Two-component SSS** — fixed-fleet (nuclear, hydro — constant TWh) + RPS (scales with demand)
- [x] **Diablo Canyon + NY nuclear as permanent fixed SSS** — state-supported indefinitely

### EAC Scarcity: Combined RPS + Voluntary Supply Stack Model (Feb 15, rev 2)
**Decision**: RPS mandates and voluntary corporate procurement compete for the same finite buildable clean capacity. Marginal cost is set by combined demand (RPS + voluntary) on the supply stack, not voluntary alone.

**Literature validation** (Xu et al. 2024, Joule / Princeton ZERO Lab): GenX capacity expansion model shows combined RPS + voluntary C&I demand on the same regional supply curve produces non-linear cost escalation as both compete for finite buildable capacity. Gillenwater (2008, Energy Policy): only when combined demand creates real scarcity does voluntary procurement drive new investment. Denholm et al. (NREL 2021): "last few percent" costs escalate exponentially.

**Previous approach (superseded, v1)**: RPS adder as economic gate — new clean only entered when LCOE < wholesale + RPS adder. This produced zero new build in all ISOs because the price signal never cleared any supply stack tier. Supply was frozen at 2025 levels.

**Current approach (v2) — two-track demand, unified supply stack**:

1. **RPS-mandated demand** (forced, regardless of economics):
   - `rps_new_need = max(0, rps_target × projected_demand - existing_total_clean)`
   - Regulators require this build — it happens whether or not LCOE < wholesale + adder
   - New RPS build splits into SSS vs merchant per `SSS_NEW_BUILD_FRACTION`

2. **Voluntary corporate demand** (additional, on top of RPS):
   - `corp_eac_demand = CI_share × participation × incremental_need × procurement_ratio`
   - Incremental need = `max(0, match_target% - sss_share_of_total%)`
   - SSS share grows over time as RPS mandates add clean capacity

3. **Combined demand on supply stack**:
   - `total_new_demand = rps_new_need + corp_eac_demand`
   - Walk up the supply stack (cheapest tier first) until total demand is met
   - RPS compliance absorbs cheap tiers first; corporates ride on top
   - **Marginal cost = LCOE of the tier where combined demand lands**
   - If combined demand exceeds total buildable capacity → scarcity pricing

4. **Scarcity classification**:
   - `demand_ratio = total_new_demand / total_buildable_capacity`
   - Bands: Abundant (<0.3), Adequate (<0.6), Tightening (<0.8), Scarce (<0.95), Critical (>0.95)

5. **Clean premium = marginal LCOE - wholesale price**
   - Reflects the REAL competition for clean resources — both mandated and voluntary

**Bug fixes in v2**:
- **SSS→non-SSS transfer**: When SSS policies expire (e.g., IL ZEC/CMC 2027), those TWh move to non-SSS merchant pool — not into the void. Plants don't disappear when subsidies end.
- **Annual resolution**: All years 2025–2050 (26 years, not just 6 milestone years)

**Supply stack per ISO** (static 2025 LCOEs, no decline curves — deliberate simplification):
- Resources ordered by LCOE from optimizer config
- Each tier has annual buildable TWh and cumulative max (from LBNL "Queued Up")
- No LCOE decline or wholesale escalation modeled — avoids overcomplication

**Procurement ratio** (theoretical, not optimizer-derived):
- 75%→0.80×, 90%→1.05×, 100%→1.45×
- Reflects temporal mismatch physics: higher match targets need more over-procurement

**What stays from v1**:
- SSS/non-SSS classification, two-component SSS (fixed + RPS), policy expirations
- C&I demand filter (62%), demand growth rates, participation scenarios
- Scarcity bands, supply stack LCOEs, committed hyperscaler pipeline
- Wholesale prices, RPS target trajectories

### EAC Scarcity: C&I Demand Filter (Feb 15)
**Decision**: Corporate EAC participation base = C&I (commercial + industrial) share of total demand, not total demand. Residential load does not participate in voluntary EAC procurement.

**C&I share**: ~62% of total demand (EIA 2024 national average: 38% residential, 36% commercial, 26% industrial). Applied as a flat multiplier across all ISOs and demand growth scenarios.

**RPS stays against total demand**: RPS mandates apply to total retail sales (including residential), so RPS/SSS calculations continue to use full demand. Only the voluntary corporate procurement base is filtered to C&I.

**Limitation (noted)**: C&I share held constant across demand growth scenarios. In practice, data center growth (classified as commercial by EIA) could shift C&I share higher over time, particularly in PJM and ERCOT. This simplification is acknowledged but not modeled.

### EAC Scarcity: Hyperscaler Committed Nuclear PPA Pipeline (Feb 15)
**Decision**: Model committed hyperscaler nuclear PPAs as a phased supply reduction rather than generic demand growth. Hyperscaler data center demand is disproportionately clean-energy-focused — these PPAs lock up specific clean generation that is no longer available for other corporate procurement.

**PJM committed pipeline**: ~4 GW nuclear PPAs committed by hyperscalers:
- Amazon-Talen: Susquehanna campus (~960 MW, operational)
- Microsoft-Constellation: TMI Unit 1 restart (~835 MW, targeting 2028)
- Other committed deals ramping through 2030

**Phasing** (cumulative GW online → TWh/yr at 90% CF):
- 2025: 1.0 GW → ~7.9 TWh (Susquehanna campus + early deals)
- 2027: 2.0 GW → ~15.8 TWh
- 2028: 3.0 GW → ~23.7 TWh (TMI restart)
- 2030: 4.0 GW → ~31.5 TWh (full pipeline)

**Implementation**: Subtracted from available non-SSS supply alongside existing corporate PPAs. Modeled as `COMMITTED_CLEAN_PIPELINE` with time-phased GW → TWh conversion. Applies only to PJM currently (can be extended to other ISOs as hyperscaler commitments are announced in those markets).

**Why supply reduction, not demand growth**: Generic demand growth is diluted by the C&I share (62%) and mixed across all electricity sources. Hyperscaler nuclear PPAs specifically target and lock up clean generation — modeling as supply reduction correctly captures that these MWh are spoken for by specific off-takers.

### Corrected SSS Framework (Feb 15)
**SSS = mandatory/non-bypassable procurement creating a financial relationship between customers and generation.** Determined by whether a policy acts upon the EAC:
- **RPS/CES mandates** — state renewable/clean energy standards that retire EACs on behalf of ratepayers
- **Public ownership** — municipal utilities, federal power agencies (NYPA, BPA, TVA, WAPA)
- **Vertically integrated / rate-base assets** — utility-owned generation in regulated territories (Dominion VA plants)
- **State nuclear programs** — ZEC, CMC, or CES programs that retire nuclear EACs (NY ZEC, IL ZEC/CMC, CT Millstone PPA)

**What is NOT SSS:**
- **45U Production Tax Credit** — does not act on EAC, designed to decrease at higher revenues, credit rolls off if clean premium increases
- **Merchant nuclear** — plants not in state programs are fair game for corporate procurement (LaSalle, Calvert Cliffs, Limerick, Peach Bottom)
- **Merchant renewables** — new-build wind/solar in ERCOT or other deregulated markets without RPS obligation
- **Corporate PPAs** — voluntary, not mandatory; reduce available supply but are not SSS

**SSS is temporal** — state programs expire:
- **IL ZEC/CMC**: expires mid-2027. Dresden, Braidwood, Byron, LaSalle, Clinton, Quad Cities (~94 TWh) shift from SSS to non-SSS/PPA
- **NJ ZEC**: expired June 2025. Salem + Hope Creek (~27 TWh) already non-SSS
- **NY ZEC**: extended through 2049. All 4 NYISO plants remain SSS
- **Diablo Canyon**: state extension through 2030, NRC renewal sought to 2045. Uncertain post-2030
- **CT Millstone PPA**: ~half of output under CT auction PPA. Remainder merchant

**Key implication**: Existing merchant nuclear is available for corporate procurement. Corporations CAN buy nighttime nuclear EACs in PJM. But data center PPAs (Amazon-Susquehanna, Meta-Vistra plants, Microsoft-Crane) are rapidly consuming this supply.

**National SSS estimates (2025):**
| ISO | Total Clean (TWh) | SSS (TWh) | Non-SSS (TWh) | SSS % |
|---|---|---|---|---|
| PJM | ~280 | ~150-180 | ~100-130 | ~57% |
| ERCOT | ~205 | ~20-25 | ~180-190 | ~12% |
| CAISO | ~172 | ~140-155 | ~17-32 | ~85% |
| NYISO | ~60 | ~49-55 | ~5-11 | ~85% |
| NEISO | ~50 | ~25-30 | ~15-20 | ~55% |

**Scarcity analysis parameters (expanded):**
- Corporate participation: 0-100% of ISO load
- Hourly match target: 75-100%
- Time horizons: 2025–2050 (annual, 26 years)
- Demand growth: Low/Med/High per ISO (from dashboard DEMAND_GROWTH_RATES)
- SSS supply evolves over time (policy expirations + new build from RPS mandates)
- Scarcity inflection = participation × match level where hourly demand > uncommitted non-SSS supply

**Interactive dashboard toggles:** Corporate participation (slider 0-100%), hourly match target (slider 75-100%), region selector (5 ISOs + national), demand growth (Low/Med/High), time horizon (2025-2050)

### Timezone / UTC Handling (Feb 15)
- **EIA hourly data**: Local time (NOT UTC), per EIA documentation. No offset needed during data loading.
- **Optimizer compressed_day**: New optimizer checkpoint outputs UTC-indexed arrays (h%24 from 0-8759 sequential UTC). Old pre-computed results were local time.
- **Dashboard fix applied**: CAISO 75%/80% rotated UTC→local (offset 8). All other ISOs were already local.
- **Future checkpoint merges**: Must apply UTC-8 rotation to CAISO compressed_day data from new optimizer run before merging into results JSON.
- **All other ISOs verified**: PJM, ERCOT, NYISO, NEISO show local-time profiles (solar 7-19, demand peaks 16-18). Issue was CAISO-specific from checkpoint merge.

### About Page (`about.html`) — Design Direction
- **Purpose**: Scrollytell explainer of the entire project scope and what it researches & explores
- **Narrative layers** (in order):
  1. System-level grid decarb economics (marginal dispatch, last-mile costs, hourly supply gaps)
  2. Power generation corporate targets and decarbonization efforts (fleet transition, nuclear, CCS, renewables)
  3. Voluntary corporate clean energy buyers (PPAs, 24/7 CFE, hourly matching, EAC demand)
  4. State and national policies (RPS, ITC/PTC, 45Q, capacity markets, mandates)
  5. Interconnected accounting & reporting frameworks (GHG Protocol Scope 2 revision, SBTi, EPRI SMARTargets)
  6. Global and national goals (Paris, IEA NZE, US targets, EU climate law)
- **Mind map visualization**: SVG-based infographic showing relationships between all six layers with:
  - Catalytic links (green dashed) — positive feedback loops accelerating decarbonization
  - Perverse incentive links (red dashed) — misaligned frameworks channeling dollars to paper compliance
  - Feedback loops (blue dashed) — systemic interdependencies
  - Animated node entrance + line drawing on scroll
- **Key themes**:
  - How frameworks can catalyze affordable/feasible decarbonization OR create perverse incentives (e.g., 45Q running gas at max CF, annual RECs hiding dirty hours, unbundled cross-region RECs)
  - Research gaps this project addresses: cost-as-variable co-optimization, regional variation, last-5% inflection zone, EAC scarcity quantification, policies evaluated against physical constraints
  - Novel insights produced: cost drives mix, inflection zone steeper than expected, region determines strategy, existing clean assets undervalued, 45Q perverse incentive, EAC scarcity already emerging
- **Standalone page** — no dependencies on other files, avoids merge conflicts with ongoing work on other branches

### LMP Price Calculation Module — Design Plan (Feb 20, 2026)

**Purpose**: Compute synthetic hourly LMP (Locational Marginal Prices) for each winning scenario by reconstructing 8760-hour dispatch and applying ISO-specific price formation models. Enables cost-of-energy analysis that accounts for how clean energy penetration reshapes wholesale electricity prices.

**Pipeline position**: Downstream of Step 4. Reads Step 3/4 outputs, writes to `data/lmp/`. No changes to Steps 1–4.

```
Step 1 (PFS) → Step 2 (EF) → Step 3 (Cost) → Step 4 (Postprocess)
                                                      ↓
                                          compute_lmp_prices.py    ← NEW
                                                      ↓
                              data/lmp/{ISO}_lmp.parquet   (per-ISO output)
                              data/lmp/{ISO}_archetypes.parquet
                              data/lmp/lmp_summary.json  (dashboard-ready)
```

#### Shared Architecture: `dispatch_utils.py`

Extracted from `recompute_co2.py` to create a single source of truth for dispatch reconstruction, fossil retirement, and profile loading. Both `recompute_co2.py` and `compute_lmp_prices.py` import from this module. Step 1 Numba JIT dispatch functions also consolidated here.

```
dispatch_utils.py (shared)
├── Constants: battery/LDES params, hydro caps, grid mix, coal/oil caps, base demand
├── get_supply_profiles(iso, gen_profiles)
├── reconstruct_hourly_dispatch(mix, demand, profiles)  ← battery + LDES dispatch
├── compute_fossil_retirement(iso, clean_pct, ...)      ← remaining capacity at threshold
└── load_common_data()                                  ← demand, gen profiles, emission rates, fossil mix

recompute_co2.py (imports dispatch_utils — refactored, identical behavior)
├── compute_dispatch_stack_emission_rate()
├── compute_co2_hourly()
└── recompute_all_co2()

compute_lmp_prices.py (imports dispatch_utils)
├── PriceModel classes (ISO-specific)
├── build_merit_order_stack()
├── compute_lmp_stats()
└── calibration framework
```

**Compatibility requirement**: After refactor, `recompute_co2.py` must produce bit-identical results. Verified via automated test.

#### Design Decisions (Locked)

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | Merit-order stack walk | `np.searchsorted` step-function | More accurate than `np.interp` linear interpolation — discrete units don't interpolate. Same performance. |
| 2 | Archetype dedup key | `(mix_tuple, fuel_level, threshold)` | Threshold affects fossil stack (retirement changes available capacity). ~7,800 unique calcs per ISO. Still fast with Numba (<30s). |
| 3 | Dispatch functions | Shared module (`dispatch_utils.py`) | Single source of truth. Extracted from `recompute_co2.py` + Step 1 Numba JIT. Compatibility test required. |
| 4 | Surplus pricing | Calibrated empirical curve from start | Parameterized with calibration targets; "reasonable defaults" before actual LMP data. Phase 2 tunes parameters, no refactoring. |
| 5 | Calibration LMP source | Day-ahead LMP | Cleaner, better for structural model. RT sensitivity via `rt_sensitivity_factor` parameter baked in. |

#### Data Flow

**Inputs (all existing)**:
- `data/eia_generation_profiles.json` — hourly solar/wind/hydro/nuclear shapes (8760)
- `data/eia_demand_profiles.json` — hourly demand shape (8760)
- `dashboard/overprocure_scenarios.parquet` (or `overprocure_results.json`) — winning mixes
- `data/eia_fossil_mix.json` — coal/gas/oil shares for stack construction
- `data/egrid_emission_rates.json` — heat rates for marginal cost derivation
- Step 3/4 constants — wholesale prices, fuel adjustments, gas capacity (imported at runtime, NOT hardcoded)

**New inputs (Phase 2 — calibration)**:
- `data/lmp/actual_lmp_PJM.json` — PJM Data Miner 2 API (Western Hub, 2021-2025)

**Outputs**:
- `data/lmp/{ISO}_lmp.parquet` — summary stats per (threshold, scenario), ~2 MB/ISO
- `data/lmp/{ISO}_archetypes.parquet` — 8760h profiles for unique archetypes, ~15-20 MB/ISO
- `data/lmp/{ISO}_checkpoint.json` — resume state (transient, deleted on completion)
- `data/lmp/lmp_summary.json` — dashboard-ready cross-ISO summary, <500 KB
- `dashboard/js/lmp-data.js` — client-side visualization data (Phase 4)

#### Hourly Dispatch Reconstruction

Reuses Step 1/Step 5 logic via shared `dispatch_utils.py`:
1. Build weighted supply curve: `Σ (mix_pct × profile)` for clean_firm, solar, wind, hydro
2. Apply procurement multiplier
3. Battery dispatch (4hr daily cycle, 85% RTE)
4. Battery8 dispatch (8hr daily cycle, 85% RTE)
5. LDES dispatch (100hr, 7-day window, 50% RTE)
6. Result: `residual_demand = demand - total_clean_supply` (8760 array; positive = fossil needed)

**Vectorization**: Base supply for N archetypes is a single matrix multiply `(N,4) @ (4,8760)`. Storage dispatch loops per-archetype (SOC state carries forward) — Numba JIT target.

#### Fossil Merit-Order Stack (Parameterized)

Heat rates (MMBtu/MWh) from EIA Electric Power Annual Table 8.1:
- Coal steam: 10.0, Gas CCGT: 6.4, Gas CT: 10.0, Oil CT: 10.5

Variable O&M ($/MWh) from EIA AEO / NREL ATB:
- Coal: $4.50, Gas CCGT: $2.00, Gas CT: $4.00, Oil CT: $5.00

Marginal cost = heat_rate × fuel_price + VOM. **Stack order determined by marginal cost** — fuel-switching aware:
- At Low gas ($2/MMBtu): gas CCGT MC = $14.80 < coal MC = $22.50 → gas dispatches first
- At High gas ($6/MMBtu): gas CCGT MC = $40.40 > coal MC = $26.50 → coal dispatches first

Fuel prices imported from Step 3/4 at runtime (L/M/H sensitivity). Fossil capacity from shared retirement model (threshold-dependent: coal retires first → oil → gas).

#### ISO-Specific Price Formation Models

Each ISO gets its own `PriceModel` class with calibratable parameters:

| ISO | Capacity Mechanism | Scarcity Model | Surplus Model | Key Parameters |
|---|---|---|---|---|
| **PJM** | RPM capacity market | Penalty factor → $2,000 cap | Moderate negative prices (coal/nuclear must-run) | `scarcity_cap=2000`, `floor=-30`, coal baseload min-gen |
| **ERCOT** | Energy-only | **ORDC** — smooth exponential adder (VOLL × LOLP curve) | Aggressive negative prices (no must-run obligation) | `ordc_cap=5000`, `ordc_shape`, `floor=-50` |
| **CAISO** | Resource Adequacy | Soft cap + RDRR mechanism | Most negative prices (solar duck curve) | `scarcity_cap=2000`, `floor=-60`, solar curtailment premium |
| **NYISO** | ICAP | Penalty factor → $2,000 cap | Similar to PJM but tighter geography | `scarcity_cap=2000`, `floor=-20` |
| **NEISO** | FCM | Penalty factor → $2,000 cap | Winter gas constraint creates seasonal scarcity | `scarcity_cap=2000`, `winter_gas_adder`, pipeline constraint |

**ERCOT ORDC** (unique — not a simple price cap): `adder = VOLL × LOLP(reserves)`, LOLP increases exponentially as reserves drop below ~3,000 MW. VOLL = $5,000/MWh (post-2023 reform).

**NEISO winter gas** (unique — already modeled in Step 4): Winter hours (Dec-Feb) get gas price spike due to pipeline constraints. Parameterized via existing `NEISO_CCS_GAS_ADDER = $13.13/MWh`.

**All surplus pricing uses calibrated empirical curves** — parameterized from the start with shape/floor/decay parameters that Phase 2 calibration tunes. Reasonable defaults used before actual LMP data.

**RT sensitivity**: `rt_sensitivity_factor` parameter scales volatility/spread to approximate real-time conditions from day-ahead calibration.

#### Output Schema

**Per-ISO stats: `data/lmp/{ISO}_lmp.parquet`**

| Column | Type | Description |
|---|---|---|
| `threshold` | float32 | Clean energy % |
| `scenario` | string | 9-dim key |
| `archetype_key` | string | Dedup key |
| `avg_lmp` | float32 | Time-weighted average $/MWh |
| `peak_avg_lmp` | float32 | Peak hours (7am-11pm weekdays) |
| `offpeak_avg_lmp` | float32 | Off-peak hours |
| `zero_price_hours` | int16 | Hours at $0 or below |
| `negative_price_hours` | int16 | Hours below $0 |
| `scarcity_hours` | int16 | Hours above scarcity threshold |
| `lmp_p10/p25/p50/p75/p90` | float32 | Percentiles |
| `price_volatility` | float32 | Std dev |
| `duck_curve_depth_mw` | float32 | Max surplus MW |
| `net_peak_price` | float32 | Price at max net demand hour |
| `fossil_revenue_mwh` | float32 | Avg $/MWh earned by remaining fossil |

**Per-ISO archetype profiles: `data/lmp/{ISO}_archetypes.parquet`**

| Column | Type |
|---|---|
| `archetype_key` | string |
| `threshold` | float32 |
| `fuel_level` | string |
| `hourly_lmp` | float32[8760] (list column) |
| `hourly_residual_mw` | float32[8760] (list column) |
| `hourly_marginal_unit` | uint8[8760] (list column) |

Size: ~15-20 MB/ISO compressed. Total all ISOs: ~75-100 MB.

#### Checkpointing & Session Resilience

- **Atomic checkpoint writes** (`os.replace` — POSIX atomic)
- **Append-mode parquet** (don't hold full ISO in memory)
- **Per-threshold flush** (max ~30s of lost work on crash)
- **Skip-if-exists** at ISO level (`--force` to override)
- **Per-ISO output files** (completing PJM doesn't risk ERCOT data)
- **Resume from checkpoint**: loads `{ISO}_checkpoint.json`, skips completed thresholds

#### Calibration Framework (Phase 2)

**Data source**: PJM Data Miner 2 API — Western Hub, DA LMP, 2021-2025. Free registration.

**Calibration targets (weighted)**:
- 40%: RMSE of hourly prices
- 15%: Error in annual mean price
- 15%: Error in zero/negative price hour count
- 15%: Error in P90 price (tail behavior)
- 15%: KS statistic of price duration curve shape

**Parameters calibrated**: floor_price, surplus_slope, stack price offsets, scarcity_shape.

**Validation**: Train 2021-2023, test 2024-2025. Cross-region sanity check: calibrate PJM, verify NYISO (similar market).

**Other ISO data sources** (Phase 3): ERCOT (`misportal.ercot.com`), CAISO (`oasis.caiso.com`), NYISO (`mis.nyiso.com`), NEISO (`isoexpress.iso-ne.com`). All free. `gridstatus` package wraps all APIs.

#### Implementation Phases

| Phase | Scope | Files | Size |
|---|---|---|---|
| **0** | Extract `dispatch_utils.py` from `recompute_co2.py` + compatibility test | `dispatch_utils.py`, `recompute_co2.py` | ~300 lines extracted |
| **1a** | Core engine — PJM only, Medium fuel, no calibration | `compute_lmp_prices.py` | ~400 lines |
| **1b** | Full fuel sensitivity sweep (L/M/H) + all thresholds + checkpointing | Same | +~150 lines |
| **1c** | All 5 ISOs with market-specific models | Same | +~200 lines |
| **2a** | PJM LMP data fetch | `fetch_pjm_lmp.py` | ~150 lines |
| **2b** | Calibration + validation | `calibrate_lmp_model.py` | ~200 lines |
| **3** | All-ISO calibration data fetch | Per-ISO fetch scripts | ~150 lines |
| **4** | Dashboard integration | `generate_shared_data.py` + JS | ~150 lines |

#### New Files

```
dispatch_utils.py              # Shared dispatch/retirement/profiles (~300 lines)
compute_lmp_prices.py          # Main LMP script (~750 lines)
fetch_pjm_lmp.py               # Phase 2: LMP data fetcher (~150 lines)
calibrate_lmp_model.py         # Phase 2: Parameter calibration (~200 lines)
data/lmp/                      # Output directory
  ├── {ISO}_lmp.parquet        # ~2 MB each
  ├── {ISO}_archetypes.parquet # ~15-20 MB each
  ├── {ISO}_checkpoint.json    # Transient
  ├── lmp_summary.json         # ~500 KB
  └── actual_lmp_*.json        # Phase 2: calibration data
```

### Open questions
- Path-dependent MAC visualization: may need alternative to MAC curve format
- ELCC: include in next run? Fixed or penetration-dependent?
- Multi-year re-run: Phase 1+3 hybrid recommended (~40% compute savings vs full)

---

## 1. Model Framework

- **2025 snapshot model** — all data, profiles, costs, grid mix shares reflect fixed 2025 actuals
- **No demand growth projections** — point-in-time scenario analysis only
- **Grid mix baseline** = actual 2025 regional shares, priced at wholesale, selectable as reference scenario (fixed, not adjustable by user)
- **Regions**: CAISO, ERCOT, PJM, NYISO, NEISO
- **Repo**: `jessicacohen554-cyber/hourly-cfe-optimizer`
- **Dev branch**: `claude/add-status-feature-7Jhew` (v4.0 rebuild)

---

## 2. Resources (6 total — v4.0 rebuild: CCS merged into Clean Firm)

| # | Resource | Profile Type | New-Build? | Cost Toggle? | Transmission Adder? |
|---|---|---|---|---|---|
| 1 | **Clean Firm** (nuclear/geothermal/CCS-CCGT) | Blended: seasonal-derated baseload (nuclear/geo) + flat baseload (CCS) | Yes | Low/Med/High (regional) | Yes (regional) |
| 2 | **Solar** | EIA 2025 hourly regional | Yes | Low/Med/High (regional) | Yes (regional) |
| 3 | **Wind** | EIA 2025 hourly regional | Yes | Low/Med/High (regional) | Yes (regional) |
| 4 | **Hydro** | EIA 2025 hourly regional | **No** — capped at existing | **No** — wholesale only | **No** — always $0 |
| 5 | **Battery** (4hr Li-ion) | Daily cycle dispatch | Yes | Low/Med/High (regional) | Yes (regional) |
| 6 | **LDES** (100hr iron-air) | Multi-day/seasonal dispatch | Yes | Low/Med/High (regional) | Yes (regional) |

### v4.0 Change: CCS-CCGT merged into Clean Firm (Decision 6D)
- **Rationale**: Reduces resource mix search space from 5D to 4D, dramatically cutting grid search combinatorics (~40-60% fewer combos). Both nuclear and CCS-CCGT are modeled as baseload (CCS runs flat due to 45Q incentives), making them functionally similar for dispatch purposes.
- **Implementation**: The optimizer allocates a single `clean_firm` percentage. Within that allocation, the sub-split between nuclear/geothermal and CCS-CCGT is determined by cost optimization — the cost model evaluates different sub-allocations and picks the cheapest blend. CCS retains its distinct cost profile (LCOE, 45Q offset, fuel linkage) and emission characteristics (95% capture, residual 0.0185 tCO2/MWh).
- **Dispatch profile**: Weighted blend of nuclear seasonal-derated profile and CCS flat profile, based on sub-allocation ratio.
- **Dashboard impact**: Results still report the nuclear/CCS sub-split for transparency.

### Key resource decisions:
- **H2 storage excluded** (explicitly out of scope)
- **Clean Firm nuclear derate**: Seasonal spring/fall derate applied to nuclear portion only (not geothermal or CCS). Reflects staggered refueling outages across the fleet (~18-24 day outages per plant every 18-24 months, distributed across spring/fall shoulders). Summer/winter: ~100% CF (nukes run full during peak demand seasons). Spring/fall: reduced CF based on observed EIA 2021-2025 nuclear generation vs. available capacity. Derive simplified flat seasonal percentages from actual data. Geothermal (relevant in CAISO) stays flat 1/8760.
- **Hydro**: Existing only, capped at regional capacity, wholesale priced, no new-build tier, $0 transmission
- **CCS-CCGT** (within Clean Firm): 95% capture rate, residual ~0.0185 tCO2/MWh, 45Q ($85/ton = ~$27.5/MWh offset) baked into LCOE, fuel cost linked to gas price toggle. **Modeled as flat baseload (not dispatchable) by design** — while CCS-CCGT is physically dispatchable, the 45Q tax credit ($85/ton for geologic storage) incentivizes running at maximum capacity factor to maximize capture credits. This is an economics-driven decision, not a physical constraint.
- **LDES**: 100-hour iron-air, 50% round-trip efficiency, capacity-constrained dispatch with dynamic capacity sizing. LCOS reflects actual utilization of built capacity. (Decision 7A — kept current.)
- **Battery**: 4-hour Li-ion, 85% round-trip efficiency, capacity-constrained daily-cycle dispatch. LCOS reflects actual utilization — oversized capacity that sits idle drives cost up. (Decision 7A — kept current.)

---

## 3. Thresholds (13 total — v4.0 rebuild: expanded from 10)

```
50, 60, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 100
```

- **50%, 60%, 70%** (v4.0 addition): Captures the easy-to-achieve baseline region where most mixes succeed. Provides context for "how cheap is partial decarbonization" and anchors the cost curve left side. These thresholds run fast (most mixes hit target, narrow procurement bounds).
- 5% intervals from 75-85 (captures broad trend)
- 2.5% intervals from 87.5-97.5 (captures steep cost inflection zone)
- 99% and 100% anchor the extreme end
- Key inflection behavior (CCS/LDES entering mix, storage costs spiking) captured at 90-97.5
- Dashboard interpolates smoothly between these anchor points for abatement curves

---

## 4. Dashboard Controls (7 total — paired toggles)

### Preserved (2):
1. **Region/ISO select** (CAISO, ERCOT, PJM, NYISO, NEISO)
2. **Threshold select** (10 values: 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 100)

### Sensitivity toggles (7 toggles + 1 binary switch):

Cost sensitivities are organized into 7 graduated toggles (L/M/H) plus one binary policy switch (45Q). CCS and Geothermal are separated from Firm Gen to allow independent sensitivity analysis of these distinct technologies.

| # | Toggle | Options | Controls | Affects |
|---|---|---|---|---|
| 3 | **Renewable Generation Cost** | Low / Medium / High | Solar LCOE + Wind LCOE | Both solar and wind generation costs (regional) |
| 4 | **Firm Generation Cost** | Low / Medium / High | Clean Firm (nuclear) LCOE — uprate + new-build | Nuclear uprate and new-build costs (regional) |
| 5 | **Storage Cost** | Low / Medium / High | Battery LCOS + LDES LCOS | Both storage technology costs (regional) |
| 6 | **CCS Cost** | Low / Medium / High | CCS-CCGT underlying cost (capex, transport, storage) | CCS technology maturity — L=mature/low capex, H=immature/high capex |
| 7 | **45Q Credit** | On / Off | $29/MWh 45Q tax credit offset on CCS LCOE | Binary policy switch — On=full 45Q offset, Off=no offset |
| 8 | **Fossil Fuel Price** | Low / Medium / High | Gas + Coal + Oil prices | Wholesale electricity price + CCS fuel cost + emission rates |
| 9 | **Transmission Cost** | None / Low / Medium / High | All resource transmission adders | Transmission adders on all new-build resources (regional) |
| 10 | **Geothermal Cost** | Low / Medium / High | Geothermal LCOE (CAISO only) | **CAISO only** — no geothermal resource in other ISOs |

**Toggle separation rationale**:
- **CCS separated from Firm Gen**: CCS has a distinct cost structure (capture + transport + storage + fuel) and policy dependency (45Q) that makes it independently variable from nuclear. Pairing them hides the 45Q sensitivity.
- **Geothermal separated and CAISO-only**: Geothermal is a regionally constrained resource — only CAISO has meaningful hydrothermal potential (5 GW cap from USGS identified resources). Other ISOs have zero geothermal potential for power generation. Toggle is hidden/disabled for non-CAISO regions.
- **45Q as binary switch**: The 45Q credit is a policy decision (exists or doesn't), not a cost spectrum. Keeping it binary allows clean analysis of "what if 45Q expires/isn't renewed."

**L/M/H maturity mapping for CCS**:
- **Low**: Mature CCS deployment — nth-of-a-kind plants, established Class VI wells, optimized CO₂ transport networks, low capex
- **Medium**: Mid-range — some learning curve benefits, moderate infrastructure availability
- **High**: Immature/early deployment — first-of-a-kind plants, new well permitting, long transport distances, high capex

**Scenario count**:
- Non-CAISO: 3×3×3×3×2×3×4 = **5,832 cost scenarios** per region per threshold
- CAISO: 5,832 × 3 = **17,496 cost scenarios** per threshold (includes geothermal toggle)
- Total: 17,496 + 5,832×4 = **40,824 scenarios** per threshold set
- All Step 3 (arithmetic on cached physics) — runs in minutes, not hours

**Sensitivity key format**:
- Non-CAISO: `RFSC_QFF_TX` (e.g., `MMMM_1M_M` = all Medium, 45Q on)
- CAISO: `RFSC_QFF_TX_G` (e.g., `MMMM_1M_M_M` = all Medium, 45Q on, Medium geo)
- Q = `1` (45Q on) or `0` (45Q off)

**NOTE**: All graduated toggles use **Low / Medium / High** naming consistently (never "Base" or "Baseline").

**Optimizer approach**: Resource mix co-optimized with costs for EVERY scenario. Different cost assumptions produce different optimal resource mixes — this is the core scientific contribution. Physics cached from Step 1; Step 3 cross-evaluates all EF mixes under each sensitivity combo to find the cheapest valid mix.

### 4.1 Warm-Start Optimization (Trifold Seed Strategy)

**Problem**: Full 3-phase co-optimization (Phase 1 coarse grid → Phase 2 medium refinement → Phase 3 fine-tune) takes 5-10 minutes per scenario. With 44 representative scenarios per threshold × 10 thresholds × 5 ISOs, full Phase 1 for every scenario is prohibitively slow.

**Solution**: Trifold warm-start seeding — run full 3-phase for 3 categories of scenarios, then warm-start the remainder with the discovered mix archetypes:

1. **Medium scenario** (`MMM_M_M`): Full 3-phase. The central cost case — most scenarios converge to similar mixes since physics dominates.
2. **Extreme archetype scenarios** (7 scenarios): Full 3-phase. These represent opposite corners of the cost space where the optimal mix is most likely to diverge from Medium:
   - `HLL_L_N` — High renewables, low firm/storage/fuel, no transmission (VRE-dominant)
   - `LHL_L_M` — High firm, low renewables (firm-dominant)
   - `LLH_H_M` — High storage, high fuel (storage-dominant)
   - `HHH_H_H` — All high (maximum cost pressure)
   - `LLL_L_L` — All low (minimum cost environment)
   - `HLL_L_H` — High renewables + high transmission (VRE with tx penalty)
   - `LHL_H_N` — High firm + high fuel, no transmission (fuel-stressed firm)
3. **All remaining scenarios**: Warm-started from the diverse seed pool discovered in steps 1-2. Skip Phase 1 coarse grid; start with seed mixes + their 5% neighborhoods + edge-case seeds, then run full Phase 2/3 refinement.

**Dynamic archetype discovery**: As warm-started scenarios find new mix archetypes (mixes that differ by >5% in any resource dimension from known archetypes), these are automatically added to the seed pool for subsequent scenarios. This ensures rare-but-valid mixes discovered mid-run are propagated forward.

**Fallback**: If warm-start fails to find any feasible solution for a scenario, it automatically falls back to full Phase 1 coarse grid search.

**Scientific validity**: This approach is equivalent to providing a smarter initial guess, not a shortcut:
- Hourly matching scores are physics-based and cost-independent. The score cache is shared across all scenarios (same mix + procurement → same physics).
- Phase 2 (5% neighborhood) and Phase 3 (1% neighborhood) refinement run identically regardless of whether warm-start or full Phase 1 was used.
- The only difference is the set of candidates entering Phase 2. Warm-start uses the discovered archetype pool + edge seeds instead of the full 270-combo coarse grid. Since most of those 270 combos evaluate to the same few optimal regions anyway (especially at lower thresholds), the archetype pool covers the same solution space more efficiently.
- Cross-pollination after all scenarios still evaluates every discovered mix against every cost scenario, catching any missed optimizations.
- Monotonicity re-sweep uses full Phase 1 (resweep=True disables warm-start), providing an additional safety net.

**Expected speedup**: ~3-5× per threshold (estimated reduction from ~5-10 min/scenario to ~1-3 min/scenario for warm-started cases).

**Risks and limitations**:
1. **Missed global optima at extreme cost corners**: If an extreme cost combination produces an optimal mix radically different from any archetype, warm-start's neighborhood search might not find it. **Mitigation**: The 7 extreme archetype scenarios cover the most divergent cost corners; dynamic archetype discovery catches emergent patterns; cross-pollination provides a second chance; monotonicity re-sweep with full Phase 1 provides a final safety net.
2. **Phase 2 neighborhood radius**: The 5% step with radius 2 covers ±10% in each resource dimension from the warm-start mix. Optimal mixes more than 10% away in any dimension from all seed archetypes would be missed. **Mitigation**: Edge-case seeds (100% solar, 100% wind, etc.) are always included regardless of warm-start. At observed convergence rates, ≤14 unique mixes typically serve 5,832 scenarios at lower thresholds, well within the archetype pool's coverage.
3. **Threshold-dependent risk**: Higher thresholds (95-100%) have more diverse optimal mixes across cost scenarios. **Mitigation**: The archetype pool grows dynamically; extreme scenarios are more likely to diverge at high thresholds, populating the pool with the right seeds.
4. **Not used during re-sweep**: Monotonicity re-sweep always uses full Phase 1 (warm_start_result is not passed when resweep=True). This is intentional — re-sweep needs the broadest possible search to resolve violations.

### 4.2 Scenario Pruning & Adaptive Resampling Pipeline

**Problem**: 5,832 cost scenarios × 13 thresholds × 5 ISOs = 378,780 co-optimizations. Even with warm-start, running all 5,832 per threshold is slow. Empirically, physics dominates at lower thresholds — only ~14 unique mixes serve all 5,832 scenarios.

**Solution**: 5-stage pipeline runs 44 representative scenarios, then fills the remaining ~5,788 via cross-pollination, with adaptive resampling as a safety net.

#### Stage 1: Medium Seed (1 scenario)
- Run `MMM_M_M` with full 3-phase optimization (no warm-start)
- Becomes the primary warm-start seed for all subsequent scenarios

#### Stage 2: Extreme Archetypes (7 scenarios)
- Run 7 corner scenarios with full Phase 1 (no warm-start): `HLL_L_N`, `LHL_L_M`, `LLH_H_M`, `HHH_H_H`, `LLL_L_L`, `HLL_L_H`, `LHL_H_N`
- These explore the most divergent regions of cost space to discover distinct mix archetypes

#### Stage 3: Remaining Representatives (~36 scenarios, totaling ~44)
- `_build_representative_scenarios()` generates a set of ~54 keys covering cost space corners, axis sweeps, and diagonals. After dedup, ~44 unique scenarios.
- The ~36 scenarios not already run as Medium/archetypes are warm-started from Medium + all diverse seed mixes discovered in Stages 1-2
- New archetypes discovered during this stage are dynamically added to the seed pool

#### Stage 4: Adaptive Resampling (if needed)
- After Stage 3, count unique resource mix archetypes found across the ~44 scenarios
- **Uniqueness threshold**: 50% — if unique mixes > 50% of scenarios run (i.e., >22 unique mixes from 44 scenarios), the representative set didn't adequately capture the diversity
- **Action**: Add midpoint scenarios from the unrun 280, spread evenly across cost space
- Target: enough additional scenarios to bring the ratio below 50%
- Up to 5 resampling rounds, each adding scenarios until convergence
- **If unique mixes ≤ 22**: Proceed directly — the 44 representatives captured the full archetype space

#### Stage 5: Cross-Pollination (fills remaining to 5,832)
- Collect all unique mixes discovered across Stages 1-4
- For ALL 5,832 scenarios (including the ~5,788 not directly optimized): evaluate every discovered mix under that scenario's cost function
- If a mix found optimal for scenario A is cheaper for scenario B than B's current best, assign it
- Result: all 5,832 scenarios have cost-optimal assignments, even the ~5,788 that were never directly optimized

**Why this works**: At lower thresholds, physics strongly constrains the feasible solution space — the same ~10-14 resource mixes are optimal across all 5,832 cost scenarios, just at different costs. Cross-pollination guarantees every scenario gets the cheapest-for-it mix from the full discovered set. Adaptive resampling is the safety net: if we're seeing more diversity than expected (>22 unique from 44), we add more direct optimizations to make sure we're not missing archetypes.

**Applies to all thresholds**: `PRUNING_THRESHOLD_CUTOFF = 100` — empirically, even at 95-100%, the archetype pool from 44 reps + resampling + cross-pollination captures the full solution space.

---

## 5. Complete Cost Tables

### 5.1 Solar LCOE ($/MWh)

| Level | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|
| Low | $45 | $40 | $50 | $70 | $62 |
| Medium | $60 | $54 | $65 | $92 | $82 |
| High | $78 | $70 | $85 | $120 | $107 |

### 5.2 Wind LCOE ($/MWh)

| Level | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|
| Low | $55 | $30 | $47 | $61 | $55 |
| Medium | $73 | $40 | $62 | $81 | $73 |
| High | $95 | $52 | $81 | $105 | $95 |

### 5.3 Clean Firm LCOE ($/MWh) — Merit-Order Tranche Model (Step 3)

Clean firm cost uses a **merit-order supply curve** with two tranches, filled cheapest-first. The effective LCOE depends on how much clean firm a scenario requires — small amounts are cheap (all uprates), large amounts are expensive (hitting new-build tranche). This is a Step 3 cost calculation applied to the Step 2 efficient frontier.

#### Tranche 1: Nuclear Uprates (Cheapest, Capped)

**Uprate LCOE** (incremental cost of adding capacity to existing plants):
| Level | LCOE ($/MWh) | Basis |
|---|---|---|
| Low | $15 | MUR-dominated (measurement recapture, minimal capital) |
| Medium | $25 | Typical MUR + stretch blend |
| High | $40 | Stretch/small EPU with equipment replacement |

*Sources: INL LWRS Program, NRC uprate database, NEI fleet data, Thunder Said Energy capex analysis, IRA §45Y PTC*

**Uprate cap** — 5% of existing nuclear capacity (midpoint estimate accounting for partial exhaustion of MUR/stretch potential; EPU potential remains but is higher-cost):

| Region | Existing Nuclear (GW) | Uprate Cap (GW) | Uprate Cap (TWh/yr @ 90% CF) |
|---|---|---|---|
| **CAISO** | 2.3 (Diablo Canyon) | 0.12 | 0.9 |
| **ERCOT** | 2.7 (South Texas Project) | 0.14 | 1.1 |
| **PJM** | 32.0 (largest US fleet) | 1.60 | 12.6 |
| **NYISO** | 3.4 (Nine Mile, FitzPatrick, Ginna) | 0.17 | 1.3 |
| **NEISO** | 3.5 (Millstone, Seabrook) | 0.18 | 1.4 |

*5% chosen as midpoint: NRC has approved ~8% fleet-wide historically, but MUR/stretch largely exhausted. Remaining potential is primarily EPU on ~27 of 94 reactors. DOE executive order targets ~3-5 GW; INL LWRS estimates 3-8% remaining. 5% balances optimism with exhaustion.*

#### Tranche 2: Geothermal (CAISO Only, Capped at 5 GW)

**CAISO only.** Geothermal fills before nuclear new-build, capped at 5 GW (~39 TWh/yr at 90% CF). Based on USGS identified hydrothermal resources (Salton Sea, Imperial Valley, The Geysers). Non-CAISO ISOs have zero geothermal potential for power generation (temperature gradients too low — see §5.4.3).

Geothermal LCOE controlled by **Geothermal Cost** toggle (CAISO only):

| Level | CAISO | Basis |
|---|---|---|
| Low | $63 | Mature hydrothermal flash (Lazard low-end, NREL ATB) |
| Medium | $88 | Blended hydrothermal flash + binary (NREL 2025 Market Report) |
| High | $110 | Binary plants + early EGS (NREL ATB conservative) |

*Sources: NREL ATB 2024, NREL 2025 US Geothermal Market Report, Lazard LCOE+ v18, USGS 2008 Assessment (FS 2008-3082), USGS 2025 Great Basin EGS Assessment.*

**Geothermal cap**: 5 GW = ~39 TWh/yr at 90% CF. Conservative bound using USGS identified hydrothermal only (excludes undiscovered and EGS). After geothermal cap is filled, remaining CAISO clean firm demand falls to Tranche 3 (nuclear new-build) or CCS, whichever is cheaper.

**Non-CAISO geothermal**: Zero. ERCOT has nascent EGS demos (Sage Geosystems) but no operating capacity. PJM/NYISO/NEISO have temperature gradients of 20-25°C/km — far below power generation thresholds. Toggle hidden/disabled for non-CAISO regions.

#### Tranche 3: Nuclear New-Build (Uncapped)

Nuclear new-build LCOE reflects advanced SMR/Gen IV technology. Controlled by **Firm Generation Cost** toggle. For CAISO, this tranche fills after geothermal cap is exhausted. For all other ISOs, this is the first new-build tranche after uprates.

| Level | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|
| Low | $70 | $68 | $72 | $75 | $73 |
| Medium | $95 | $90 | $105 | $110 | $108 |
| High | $140 | $135 | $160 | $170 | $165 |

*Low = nth-of-a-kind SMR deployment target ($70/MWh). Regional variation at Low is minimal (mature deployment compresses cost differences). Medium/High retain larger regional spreads reflecting siting, permitting, and labor differentials. ERCOT lowest (favorable siting/permitting). NYISO highest (siting constraints, labor costs).*

#### Merit-Order Cost Calculation (Step 3 Pipeline)

For each cached scenario's new clean firm demand (above existing grid share), the merit order fills cheapest-first. **CAISO has 4 tranches; other ISOs have 3.**

**Non-CAISO merit order:**
```
new_cf_twh = max(0, total_cf_pct - existing_cf_pct) / 100 × demand_twh
uprate_twh = min(new_cf_twh, uprate_cap_twh)
remaining = max(0, new_cf_twh - uprate_twh)
# Remaining filled by cheapest of: nuclear new-build vs CCS (toggle-dependent)
nuclear_price = NEWBUILD_LCOE[firm_level][iso] + tx_adder
ccs_price = CCS_LCOE[ccs_level][45q_state][iso] + tx_adder
# Each MWh goes to whichever is cheaper
```

**CAISO merit order (includes geothermal tranche):**
```
new_cf_twh = max(0, total_cf_pct - existing_cf_pct) / 100 × demand_twh
uprate_twh = min(new_cf_twh, uprate_cap_twh)
remaining_after_uprate = max(0, new_cf_twh - uprate_twh)
geo_twh = min(remaining_after_uprate, GEO_CAP_TWH)  # 39 TWh cap
remaining_after_geo = max(0, remaining_after_uprate - geo_twh)
# Remaining filled by cheapest of: nuclear new-build vs CCS (toggle-dependent)
```

At low clean firm demand → effective LCOE approaches uprate price ($25/MWh Medium).
At high clean firm demand → effective LCOE approaches new-build price ($88-110/MWh Medium).
The transition point (where uprate cap is exhausted) varies by region — PJM has the most uprate headroom.

**Replaces**: The previous fixed-blend model (§5.3 legacy: `uprate_share × uprate + (1-uprate_share) × new_build`) which applied the same effective LCOE regardless of quantity demanded. The tranche model makes clean firm cost quantity-dependent, which shifts optimal resource mixes at high thresholds.

#### Legacy Blended Values (Preserved for Reference)

Previous blended LCOE (still used in Step 1 physics optimization cache):
| Level | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|
| Low | $58 | $56 | $48 | $64 | $69 |
| Medium | $79 | $79 | $68 | $86 | $92 |
| High | $115 | $115 | $108 | $136 | $143 |

*These are what the Step 1 optimizer used. Step 3 reprices using the tranche model above.*

### 5.4 CCS-CCGT LCOE ($/MWh) — Separate Toggle with 45Q Switch

CCS cost is controlled by two independent toggles: **CCS Cost** (L/M/H maturity) and **45Q Credit** (On/Off).

#### 5.4.1 CCS LCOE with 45Q ON ($/MWh)

| Level | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|
| Low | $58 | $52 | $62 | $78 | $75 |
| Medium | $86 | $71 | $79 | $99 | $96 |
| High | $115 | $92 | $102 | $128 | $122 |

#### 5.4.2 CCS LCOE with 45Q OFF ($/MWh)

45Q OFF = add back $29/MWh offset. Same underlying capex/transport/storage assumptions.

| Level | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|
| Low | $87 | $81 | $91 | $107 | $104 |
| Medium | $115 | $100 | $108 | $128 | $125 |
| High | $144 | $121 | $131 | $157 | $151 |

*ERCOT lowest (Gulf Coast Class VI wells, abundant geology, cheap gas, shortest CO2 transport). NYISO highest (no suitable sequestration geology, longest transport, highest permitting burden).*

**L/M/H maturity mapping**:
- **Low**: Mature nth-of-a-kind CCS, established CO₂ infrastructure, low capex
- **Medium**: Mid-range deployment maturity
- **High**: Immature/early deployment, first-of-a-kind, high capex

**CCS-CCGT cost buildup**:
- Capture cost: ~$30-40/MWh (technology-dependent, relatively uniform)
- CO2 transport: $2-20/MWh (regional — distance to Class VI well)
- CO2 storage: $5-15/MWh (regional — geology, well costs)
- Fuel cost: Heat rate × gas price (responds to gas toggle)
- 45Q offset (when ON): -$29/MWh ($85/ton × 0.34 tCO2/MWh × 95% capture)
- Capture rate: 95%
- Residual emissions: ~0.0185 tCO2/MWh (= 0.37 × 0.05)

**45Q behavioral note**: With 45Q ON, CCS modeled as flat baseload (45Q incentivizes max CF to maximize capture credits). With 45Q OFF, CCS dispatch assumption unchanged in Step 3 (same cached physics), but the cost premium reflects the absence of the policy subsidy.

### 5.5 Battery LCOS ($/MWh) — Regionalized

| Level | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|
| Low | $77 | $69 | $74 | $81 | $79 |
| Medium | $102 | $92 | $98 | $108 | $105 |
| High | $133 | $120 | $127 | $140 | $137 |

*ERCOT lowest (low labor, fast permitting, flat land). NYISO highest (highest labor, most constrained siting).*

### 5.6 LDES LCOS ($/MWh, 100hr iron-air) — Regionalized

| Level | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|
| Low | $135 | $116 | $128 | $150 | $143 |
| Medium | $180 | $155 | $170 | $200 | $190 |
| High | $234 | $202 | $221 | $260 | $247 |

*ERCOT lowest (Gulf Coast geology for compressed air variants, low labor). NYISO highest (expensive labor, constrained siting, limited geology).*

### 5.7 Transmission Adders ($/MWh, new-build only)

| Resource | Level | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|---|
| **Wind** | Low | $4 | $3 | $5 | $7 | $6 |
| | Medium | $8 | $6 | $10 | $14 | $12 |
| | High | $14 | $10 | $18 | $22 | $20 |
| **Solar** | Low | $1 | $1 | $2 | $3 | $3 |
| | Medium | $3 | $3 | $5 | $7 | $6 |
| | High | $6 | $5 | $9 | $12 | $10 |
| **Clean Firm** | Low | $1 | $1 | $1 | $2 | $2 |
| | Medium | $3 | $2 | $3 | $5 | $4 |
| | High | $6 | $4 | $6 | $9 | $7 |
| **CCS-CCGT** | Low | $1 | $1 | $1 | $2 | $2 |
| | Medium | $2 | $2 | $3 | $4 | $3 |
| | High | $4 | $3 | $5 | $7 | $6 |
| **Battery** | Low | $0 | $0 | $0 | $1 | $1 |
| | Medium | $1 | $1 | $1 | $2 | $2 |
| | High | $2 | $2 | $3 | $4 | $3 |
| **LDES** | Low | $1 | $1 | $1 | $2 | $2 |
| | Medium | $2 | $2 | $3 | $4 | $3 |
| | High | $4 | $3 | $5 | $7 | $6 |
| **Hydro** | All | $0 | $0 | $0 | $0 | $0 |

*ERCOT lowest (CREZ buildout, less congestion). NYISO highest (constrained corridors, siting opposition). Sources: LBNL "Queued Up", MISO/SPP interconnection data.*

### 5.8 Fuel Prices

| Fuel | Low | Medium | High |
|---|---|---|---|
| Natural Gas | $2.00/MMBtu | $3.50/MMBtu | $6.00/MMBtu |
| Coal | $1.80/MMBtu | $2.50/MMBtu | $4.00/MMBtu |
| Oil | $55/bbl | $75/bbl | $110/bbl |

### 5.9 Fuel Price → Wholesale + Emission Rate Impact

**Wholesale**: Shifts based on regional 2025 fossil fuel mix composition. Uses **hourly wholesale price profiles** from EIA 2025 data (not flat averages).

**Wholesale fuel price adjustments** ($/MWh adder to base wholesale, by fossil fuel toggle level):

| Region | Low | Medium | High | Rationale |
|--------|-----|--------|------|-----------|
| CAISO  | -5  |   0    | +10  | ~40% gas generation |
| ERCOT  | -7  |   0    | +12  | ~50% gas, most sensitive to fuel prices |
| PJM    | -6  |   0    | +11  | ~40% gas + coal mix |
| NYISO  | -4  |   0    |  +8  | ~35% gas, more nuclear insulates from fuel |
| NEISO  | -4  |   0    |  +8  | ~35% gas, more nuclear insulates from fuel |

**Emission rate — Regional fuel-switching elasticity**:

| Region | Coal Fleet Status | Switching Elasticity | Rationale |
|---|---|---|---|
| ERCOT | Largely retired (~10GW remaining) | **Low** | Limited coal to switch to; gas price barely shifts emission rate |
| PJM | Substantial remaining (~45GW) | **High** | Gas price ↑ drives meaningful coal resurgence, emission rate jumps |
| CAISO | Near zero | **Very low** | Almost no coal option |
| NYISO | Minimal | **Low** | Small effect |
| NEISO | Minimal (retiring) | **Low** | Small effect |

---

## 6. Storage Algorithms

### 6.1 Battery (4hr Li-ion) — CAPACITY-CONSTRAINED dispatch

**Key principle**: Cost comes from capacity built. LCOS must reflect actual utilization — can't have huge redundant capacity that's barely used. The optimizer co-optimizes capacity size and dispatch.

1. `battery_dispatch_pct` maps to a **capacity** (MWh) and **power rating** (MW = capacity / 4hr)
2. Each day: charge from surplus hours up to min(available surplus, capacity), discharge to gap hours up to min(stored energy × 85% RTE, capacity)
3. Days with insufficient surplus → partial cycle → less dispatch that day
4. Annual MWh dispatched = sum of actual daily dispatches (variable, not uniform)
5. **Utilization factor** = actual annual cycles / 365 theoretical max cycles
6. **LCOS** = annualized capital cost of built capacity / actual MWh dispatched — underutilized capacity drives LCOS up, creating a natural cost penalty for oversizing
7. Optimizer finds the sweet spot: enough capacity to be useful at the target threshold, not so much that idle capacity inflates cost

### 6.2 LDES (100hr iron-air) — CAPACITY-CONSTRAINED dispatch with dynamic sizing

**Same capacity-constrained principle as battery.**

1. `ldes_dispatch_pct` maps to a **capacity** (MWh) that scales dynamically (not fixed at 1 day of demand) and **power rating** (MW = capacity / 100hr)
2. **Rolling 7-day window**: identify sustained multi-day surplus periods (spring wind, long sunny stretches) and deficit periods (winter evening doldrums, cloudy windless stretches)
3. Charge during surplus periods up to min(available surplus, power rating), respecting energy capacity
4. Discharge during deficit periods up to min(stored energy × 50% RTE, power rating)
5. State of charge carries over between windows
6. **Utilization factor** = actual annual energy throughput / (capacity × theoretical max cycles)
7. **LCOS** = annualized capital cost of built capacity / actual MWh dispatched — same utilization penalty as battery
8. Seasonal shifting: captures week-to-week and seasonal patterns batteries cannot

---

## 7. CO2 & Abatement

### 7.1 CO2 Emissions Abated — Dispatch-Stack Retirement Model

**Core assumption (Decision: Feb 19, 2026)**: As clean energy percentage grows, fossil fuels retire in merit order — coal first (dirtiest, most expensive), then oil, then gas. **Above 70% clean, all coal and oil have retired; only gas CCGT + clean remains.** This replaces the previous uniform hourly fossil mix model where coal/gas/oil shares were constant regardless of clean energy percentage.

**Validated by regional data**: Coal exhausts well before 70% clean in every ISO:
- CAISO: 0.0% coal (already gone at any threshold)
- ERCOT: 13.9% coal → exhausted at ~60% clean
- PJM: 16.5% coal → exhausted at ~57% clean
- NYISO: 0.0% coal (already gone)
- NEISO: 0.3% coal → exhausted at ~34% clean

**Merit-order retirement stack** (per ISO):
1. **Coal retires first** — highest emitter (~1.0-1.05 tCO₂/MWh). As clean % grows from baseline, each additional MWh of clean displaces coal until the regional coal fleet is fully retired.
2. **Oil retires second** — mid emitter (~0.82-1.31 tCO₂/MWh). After coal is gone, clean MWh displace oil. Oil shares are tiny (<1.1% of total gen in all ISOs), so this band is narrow.
3. **Gas retires last** — lowest fossil emitter (~0.38-0.41 tCO₂/MWh). Once coal and oil are gone (at or before 70% clean), all remaining fossil is gas CCGT. Every additional MWh of clean energy above this point displaces gas only.

**Calculation for a given clean energy threshold T%**:
```
baseline_clean = sum of existing clean shares (GRID_MIX_SHARES)
fossil_total = 100% - baseline_clean
coal_total = coal_share_of_fossil × fossil_total  (from EIA fossil mix data)
oil_total = oil_share_of_fossil × fossil_total
gas_total = gas_share_of_fossil × fossil_total

additional_clean = T% - baseline_clean  (new clean energy added)

# Merit-order displacement:
coal_displaced = min(additional_clean, coal_total)
remaining = additional_clean - coal_displaced
oil_displaced = min(remaining, oil_total)
remaining = remaining - oil_displaced
gas_displaced = min(remaining, gas_total)

# Emission rate of remaining fossil fleet at threshold T:
coal_remaining = coal_total - coal_displaced
oil_remaining = oil_total - oil_displaced
gas_remaining = gas_total - gas_displaced
fossil_remaining = coal_remaining + oil_remaining + gas_remaining

if fossil_remaining > 0:
    emission_rate = (coal_remaining × coal_rate + oil_remaining × oil_rate + gas_remaining × gas_rate) / fossil_remaining
else:
    emission_rate = 0  (100% clean)
```

**Above 70% clean**: Forced to gas-only emission rate regardless of stack calculation (simplifying assumption). `emission_rate = gas_rate` (~0.39 tCO₂/MWh). Fuel-switching elasticity (Section 5.9) is zeroed out above 70% — no coal to switch with.

**Per-fuel emission rates** (from eGRID 2023, static per region):
- `coal_rate[iso]` = eGRID coal CO₂ lb/MWh (e.g., ERCOT: 2325, PJM: 2216)
- `gas_rate[iso]` = eGRID gas CO₂ lb/MWh (e.g., ERCOT: 867, PJM: 867)
- `oil_rate[iso]` = eGRID oil CO₂ lb/MWh (e.g., ERCOT: 2894, PJM: 1919)

**CO₂ abated** (hourly resolution):
- For each hour h: `fossil_displaced[h] = clean_supply[h] − max(0, clean_supply[h] − demand[h])`
- `CO₂_abated = Σ_h fossil_displaced[h] × emission_rate_at_threshold`
- The emission rate is threshold-dependent (not hourly-variable anymore): at a given clean %, the fossil fleet composition is fixed by the retirement stack
- CCS-CCGT gets **partial credit**: 90% capture → residual ~0.037 tCO₂/MWh (vs ~0.39 unabated CCGT)

**Storage CO₂ attribution** (hourly dispatch tracking):
- Track exact hours each storage type (battery/LDES) dispatches into → use threshold-level emission rate for abatement credit
- Storage charging from surplus clean energy → charge emissions = 0
- Storage charging during hours when fossil is still marginal → charge has real emissions that reduce net abatement

**Impact vs previous model**:
- **Low thresholds (50-70%)**: Higher CO₂ abatement — first MWh of clean displaces coal (~1.0 tCO₂/MWh), not a blended average (~0.5 tCO₂/MWh)
- **High thresholds (>70%)**: Lower marginal CO₂ abatement — displacing gas only (~0.39 tCO₂/MWh), not a blended average
- **MAC at high thresholds increases** — same cost but less CO₂ per MWh displaced
- Fuel-switching elasticity irrelevant above 70% (no coal/oil to switch)

**Why this matters**: The previous uniform model assumed the fossil fleet composition stays constant as clean energy grows. In reality, coal plants are the first to retire (most expensive, most regulated, dirtiest). The dispatch-stack model correctly captures decreasing marginal emission reductions as the grid gets cleaner — the "easy" high-emission tons are abated first, and the last tons (displacing efficient gas) are the hardest.

**Absolute coal/oil caps — no new fossil build (Decision: Feb 19, 2026)**:

No new coal or oil capacity is built. Coal and oil generation are capped at their 2025 absolute TWh levels. As demand grows, only gas CCGT fills the gap — so coal/oil's share of total generation naturally declines, and the average fossil emission rate trends toward gas-only.

2025 caps (from EIA hourly data):

| ISO | Coal TWh | Oil TWh | Gas TWh | Coal Peak MW | Oil Peak MW |
|-----|----------|---------|---------|-------------|-------------|
| CAISO | 0.00 | 0.60 | 114.8 | 15 | 470 |
| ERCOT | 67.58 | 0.00 | 195.5 | 14,379 | 0 |
| PJM | 139.09 | 4.59 | 357.3 | 29,861 | 5,608 |
| NYISO | 0.00 | 0.15 | 92.3 | 0 | 1,948 |
| NEISO | 0.31 | 1.29 | 75.1 | 653 | 6,554 |

Effect: At 2025 base demand, caps equal actual generation (no change). Under demand growth scenarios, fossil fleet composition shifts:
```
grown_demand_twh = base_demand_twh × (1 + annual_rate)^(target_year − 2025)
grown_fossil_twh = grown_demand_twh × (1 − clean_pct/100)
coal_twh = min(COAL_CAP_TWH[iso], coal_cap)  # capped at 2025 level
oil_twh = min(OIL_CAP_TWH[iso], oil_cap)    # capped at 2025 level
gas_twh = grown_fossil_twh − coal_twh − oil_twh  # gas absorbs all growth
```
This means the merit-order retirement stack uses absolute TWh internally, not fixed percentages. PJM's 139 TWh of coal stays at 139 TWh even if demand doubles — its share of fossil drops from 28% to ~16%, pulling the average fossil rate toward gas.

**Data sources**:
- `data/egrid_emission_rates.json` — 2023 eGRID per-fuel CO₂ rates (lb/MWh) by region
- `data/eia_fossil_mix.json` — EIA hourly fossil fuel mix shares (coal/gas/oil) by ISO

**Implementation note**: CO₂ calculation is post-hoc (doesn't affect cost/matching optimization). The optimizer's resource mix and cost results are unaffected. CO₂ values can be recomputed on cached results.

**Bug fix (2026-02-16)**: The optimizer was applying marginal fossil emission rates to ALL storage charging hours, including hours with clean surplus (curtailment). Since storage in this model only charges from surplus clean energy, this incorrectly inflated charge emissions to ~21M tons (ERCOT 92.5%), making storage appear CO₂-neutral or negative. Fix: `charge_emission_rate = np.where(surplus > 0, 0.0, hourly_rates)` — zero rate when curtailment is occurring, marginal fossil rate otherwise. Post-processed `overprocure_results.json` and updated `MAC_DATA` in `shared-data.js`. CAISO MAC at 90% dropped from $122 to $98/ton; other regions with storage deployment similarly affected.

### 7.2 Demand Growth Counterfactual — New Gas at 350 kg/MWh (Decision: Feb 19, 2026)

**Problem**: Current CO₂ abatement only counts displaced existing grid emissions. But demand growth MWh that aren't served by clean energy would be met by new gas-fired generation. The counterfactual is that without clean procurement, those MWh produce emissions at **350 kg/MWh (0.35 tCO₂/MWh)** — the emission rate of a new CCGT.

**Formula**:
```
growth_mwh = base_demand × ((1 + annual_growth_rate)^(target_year − 2025) − 1) × 1,000,000
counterfactual_growth_emissions = growth_mwh × 0.35
total_co2_abated = existing_grid_displacement + counterfactual_growth_emissions
```

**Implementation**: Add growth counterfactual to `recompute_co2.py`. Growth rates from `step3_cost_optimization.py` DEMAND_GROWTH_RATES (CAISO 1.4–2.5%, ERCOT 2.0–5.5%, PJM 1.5–3.6%, NYISO 1.3–4.4%, NEISO 0.9–2.9%). New gas rate is 350 kg/MWh (representative CCGT heat rate ~6,400 BTU/kWh, pipeline gas). This is a post-hoc calculation — doesn't change resource mix or cost optimization.

### 7.3 SBTi Timeline-Indexed DAC Learning Curve (Decision: Feb 19, 2026)

**Approach**: Piecewise linear DAC cost projections from literature anchor points, overlaid on abatement charts where x-axis maps both clean energy threshold AND SBTi target year.

**SBTi Threshold-to-Year Mapping**:
| Year | SBTi Requirement | Optimizer Threshold |
|------|------------------|--------------------|
| 2025 | (today)          | Baseline           |
| 2030 | 50% hourly       | 50%                |
| 2035 | ~70% (interpolated) | 70%             |
| 2040 | 90% hourly       | 90%                |
| 2045 | ~95% (interpolated) | 95%             |
| 2050 | 100% (net-zero)  | 100%               |

**DAC Cost Trajectories ($/ton CO₂, net DACCS)**:

| Year | Optimistic | Central | Conservative |
|------|-----------|---------|-------------|
| 2025 | $400      | $600    | $800        |
| 2030 | $200      | $350    | $550        |
| 2035 | $150      | $275    | $450        |
| 2040 | $115      | $225    | $375        |
| 2045 | $90       | $200    | $325        |
| 2050 | $75       | $180    | $300        |

**Sources**: DOE Liftoff (2023), Sievert et al. (Joule 2024), IEA DAC (2022/2024), Fasihi et al. (J. Cleaner Prod. 2019), IEAGHG (2021/2024), Climeworks Gen 3, DOE Carbon Negative Shot, Kanyako & Craig (Earth's Future 2025), NAS (2019), Young et al. (One Earth 2023), Keith et al. (Joule 2018), Shayegh et al. (Frontiers in Climate 2021), Belfer Center (2023).

**Key assumptions by trajectory**:
- **Optimistic**: 15–20% learning rate, R&D breakthroughs, $<20/MWh renewables, 1+ GtCO₂/yr deployment by 2050
- **Central**: 10–12% learning rate, moderate policy support, $30–40/MWh renewables, 100–500 MtCO₂/yr by 2050
- **Conservative**: 5–8% learning rate, limited policy support, $40–60/MWh renewables, <100 MtCO₂/yr by 2050

**Visualization**: Abatement charts get dual x-axis (threshold % bottom, SBTi year top). DAC trajectory shown as 3 declining curves with shaded band. MAC curve intersections with DAC curves show the crossover points where grid decarbonization becomes more/less expensive than DAC at each milestone year.

All values are 2024 USD, net tons CO₂ removed (accounting for 5–12% lifecycle emissions). Full DACCS (capture + transport + storage + MRV).

### 7.2 Abatement Cost Curves (2 new charts)
- **Average Cost of Abatement**: Total incremental cost / Total CO2 abated = **$/ton CO2**
- **Marginal Cost of Abatement**: (Cost_{X+1%} − Cost_{X%}) / (CO2_{X+1%} − CO2_{X%}) = **$/ton CO2**
- **X-axis**: 75% to 100%, **linear numeric scale** (proportional spacing — distance from 85→90 equals 75→80)
- Both curves respond dynamically to **all 10 sensitivity toggles**
- 1% intervals from 85% provide smooth curve in the inflection zone
- Marginal curve shows hockey-stick shape: cheap early tons, expensive last tons

---

## 8. Dashboard Layout

### 8.1 Existing visuals — ALL PRESERVED
- Donut chart (resource mix) → updated to show 7 resources
- Compressed day profile (stacked area) → updated with Battery/LDES/CCS-CCGT split
- Peak capacity panel → updated with new resources
- Cost breakdown panel → updated with all resources
- Metric tiles: match score, procurement level, blended cost, curtailment → preserved
- Key finding box → preserved
- Scrollytelling narrative sections → preserved

### 8.2 New additions (layered on top)
- **CO2 abated metric tile** — tons of CO2 displaced for selected scenario
- **Average abatement cost curve chart** — $/ton across 75-100% thresholds
- **Marginal abatement cost curve chart** — incremental $/ton at each threshold step
- **"What You Need Depends on What You Have" panel**:
  - Starting point: grid mix baseline hourly match score
  - Target: selected threshold
  - Gap: target − baseline
  - Incremental resources needed to close gap
  - Incremental $/MWh above wholesale
  - CO2 impact of closing gap
  - Dynamic regional insight text (e.g., "NYISO's strong nuclear fleet means 40% less incremental clean firm needed vs. ERCOT")
- **Sensitivity toggle panel** (10 new toggles in control area)

### 8.3 Chart axis rules
- Abatement curves: **Linear numeric x-axis** (not categorical). 1 percentage point = same pixel distance everywhere.
- Data points at 75, 80, 85, 86, 87, ..., 100. Longer line segments between sparse points (75→80→85) are honest about lower granularity there.

---

## 9. Existing Grid Mix (2025 Actuals)

### Grid Mix Shares (% of generation):
| Resource | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|
| Clean Firm | 7.9% | 8.6% | 32.1% | 18.4% | 23.8% |
| Solar | 22.3% | 13.8% | 2.9% | 0% | 1.4% |
| Wind | 8.8% | 23.6% | 3.8% | 4.7% | 3.9% |
| Hydro | 9.5% | 0.1% | 1.8% | 15.9% | 4.4% |

### Hydro Caps (2025 actual share of demand, from EIA):
| Region | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|
| Cap (%) | 9.5 | 0.1 | 1.8 | 15.9 | 4.4 |
| 5yr range (%) | 5.2–11.2 | 0.07–0.12 | 1.9–2.1 | 15.9–18.3 | 4.5–7.8 |

**Notes**: Using 2025 actuals (not 5-year average) to match our 2025 snapshot model. CAISO hydro varies enormously by water year (2025 was above average). NYISO imports significant hydro from Quebec/Ontario.

### Wholesale Market Prices (2025 hourly profiles from EIA, not flat averages):
- Average reference points: CAISO ~$30, ERCOT ~$27, PJM ~$34, NYISO ~$42, NEISO ~$41
- Actual hourly data used for storage arbitrage, deficit-hour costing, curtailment economics

---

## 10. Two-Tier Pricing Model (Preserved)

For each resource:
- **Existing share** (up to grid mix %) → priced at wholesale
- **New-build share** (above grid mix %) → priced at LCOE + transmission adder
- **Hydro**: Always wholesale (existing only, no new-build tier)
- **CCS-CCGT** (within Clean Firm): No existing share (new resource) → all new-build priced

---

## 11. Performance Optimizations (v4.0 Rebuild)

### v4.0 Architecture (replaces v3.x sequential architecture)

- **Parallel ISO processing (A+F)**: All 5 ISOs run in parallel on 16 cores (~3 cores/ISO). Shared memory for cross-ISO data coordination. Replaces sequential processing.
- **Vectorized storage dispatch (B)**: Battery and LDES scoring use NumPy reshape/vectorized ops instead of Python day-loops. `surplus.reshape(365, 24)` for battery, vectorized rolling windows for LDES.
- **Batch mix evaluation (C)**: Grid search evaluates all combos in a single matrix multiply: `(N, 4) @ (4, 8760) = (N, 8760)`. Eliminates Python loop over individual mixes.
- **Numba JIT with fallback (D)**: Storage scoring functions compiled to machine code via Numba. If Numba unavailable, falls back to B+C (vectorized NumPy).
- **Checkpointing**: Saves after each threshold (13 per ISO); resumes from checkpoint on restart
- **Score caching**: Matching scores cached across 5,832 cost scenarios per threshold (physics reuse — cost-independent)
- **Cross-pollination**: After representative scenarios run per threshold, every unique mix re-evaluated against all scenarios
- **13 thresholds × 5 regions × 5,832 scenarios** — incremental saves essential for reliability

### 11.1 Adaptive Procurement Bounds (v4.0 — Decision 3C: Threshold-Adaptive)

Narrower bounds at low thresholds where targets are easily met; wider at high thresholds to allow extreme solar+storage or wind+storage outcomes:

| Threshold | Min% | Max% | Rationale |
|-----------|------|------|-----------|
| 50% | 50 | 150 | Easy target, modest headroom |
| 60% | 60 | 150 | Easy target |
| 70% | 70 | 175 | Moderate headroom |
| 75% | 75 | 200 | Allow 2× procurement for renewable-heavy mixes |
| 80% | 80 | 200 | Allow 2× procurement |
| 85% | 85 | 225 | Growing overprocurement for extreme renewables |
| 87.5% | 87 | 250 | 2.5× procurement |
| 90% | 90 | 250 | 2.5× procurement — cap for 90-99% per user direction |
| 92.5% | 92 | 250 | 2.5× procurement |
| 95% | 95 | 250 | 2.5× procurement |
| 97.5% | 100 | 250 | 2.5× procurement |
| 99% | 100 | 250 | 2.5× procurement |
| 100% | 100 | 500 | 5× procurement for perfect hourly matching |

**Resource ceiling**: `max_single=100` — any single resource can be up to 100% of the mix allocation. Combined with high procurement, this enables outcomes like 200% of demand from solar alone (100% solar mix at 200% procurement). This captures extreme solar+storage and wind+storage scenarios that may be cost-optimal at lower thresholds.

**Procurement step**: Adaptive based on range width — 2% step for narrow ranges (<100%), 5% for medium (100-200%), 10% for wide (>200%). Keeps runtime bounded while preserving resolution where it matters.

**Per-mix early stopping**: For each (mix, storage_config) combination, procurement is swept low→high and stops at the first level that clears the target. Lower procurement = lower cost, so the first-feasible procurement is always the cost-optimal choice for that mix. This prevents exploring the vast upper range of procurement bounds for mixes that achieve feasibility early.

**Cross-threshold pruning**: Thresholds are processed in ascending order (50% → 100%). After each threshold, the optimizer records which mixes were infeasible even at maximum procurement. These mixes are eliminated from all higher thresholds (if it can't hit 50%, it can't hit 85%). Additionally, each mix's minimum-feasible procurement from the previous threshold becomes the floor for the next threshold (no point starting below the level needed for a lower target). This dramatically narrows the search space for high thresholds.

**Persistent solution cache**: Results are accumulated in `data/physics_cache_v4.json` across runs. Each run merges new solutions with the existing cache — deduplicating by (mix, procurement, battery, ldes) key but never deleting previously found solutions. This means iterating on parameter bounds, procurement ceilings, or grid resolution adds to the feasible solution space without losing work from prior runs. The cost model in Step 3 always operates on the EF extracted in Step 2.

### 11.2 Edge Case Seed Mixes

Forced seed mixes injected into initial grid scan to guarantee extreme-but-potentially-optimal mixes survive pruning. Updated for 4D resource space (v4.0) with expanded extremes:

- **Pure solar** (95-100%): captures extreme solar+storage outcomes at high procurement. At 200%+ procurement, a 100% solar mix delivers 200%+ of demand from solar alone — paired with storage, this may be cost-optimal at lower thresholds.
- **Pure wind** (95-100%): same logic for wind-dominant regions (ERCOT)
- **Solar-dominant** (70-90%): traditional solar-heavy scenarios
- **Wind-dominant** (70-90%): traditional wind-heavy scenarios
- **Balanced renewable** (40/40 to 50/50 solar/wind): diversified variable generation
- **Clean firm dominant** (60-100%): captures scenarios where nuclear/CCS is cheapest, including pure-nuclear
- **Minimal/zero firm** (0-10% clean_firm): tests whether renewables + storage can carry all load
- **Zero-firm pure renewables** (0% clean_firm, 50/50 solar/wind): extreme test case

Seeds filtered at runtime by regional hydro cap. Adds ~30 combos to the ~441 adaptive grid combos per region — negligible compute cost, significant coverage improvement.

### 11.3 Monotonicity Re-Sweep Mechanism

**Problem**: The 3-phase heuristic search (coarse → medium → fine) can miss the global optimum at a lower threshold, producing a non-monotonic cost curve where cost(T_lower) > cost(T_higher). This is a diagnostic signal, not a valid result — achieving a lower CFE target should never cost more than achieving a higher one.

**Solution**: Post-hoc re-sweep with broader parameters. After all thresholds are initially optimized:

1. **Detection**: For each cost scenario, check that cost is non-decreasing across thresholds. Tolerance: $0.01/MWh (allows floating-point rounding).
2. **Collection**: Group all violations by threshold — `{threshold: {scenario_key: better_threshold}}`.
3. **Re-sweep** (up to 2 rounds): For each violated threshold:
   - **Seed injection**: Collect winning mixes from the thresholds that achieved better cost. These become Phase 1 seeds, guaranteeing the re-sweep explores the region of solution space that worked at the higher target.
   - **Broader Phase 1**: Use 5% step instead of 10% (~7-14× more combos), exploring the space more densely.
   - **Expanded procurement bounds**: Default bounds widened by -20% (min) and +30% (max) to search outside the assumed-optimal range.
   - **More Phase 2 candidates**: Top 30 instead of 20, with 2.0× cost filter (vs. 1.5×).
   - **More Phase 3 finalists**: Top 15 instead of 8, with 1.2× cost filter (vs. 1.1×).
   - Cross-pollination within re-swept threshold after re-optimization.
4. **Verification**: Re-check monotonicity after each round. If all violations resolved, stop early.
5. **Acceptance**: After 2 rounds, accept remaining violations with a warning (search space exhausted).

**Design rationale**: This approach finds the *true* optimum rather than masking the problem by pulling down from a higher threshold's result. The re-sweep is targeted (only violated scenarios) and seeded (with known-good mixes from higher thresholds), so it's both rigorous and compute-efficient.

**Compute overhead**: Typically affects 5-15% of scenarios at 1-3 thresholds per ISO. Phase 1 at 5% step generates ~2,000-5,000 combos (vs. ~280 at 10% step), but scoring is cached. Net overhead: ~10-20% of total runtime.

---

## 12. Methodology Documentation Checklist

The methodology page must include:
- [ ] All generation cost tables (Sections 5.1-5.4) with sources (NREL ATB 2024, Lazard, LBNL)
- [ ] All storage cost tables (Sections 5.5-5.6) with sources
- [ ] Complete transmission adder table (Section 5.7) with sources (LBNL "Queued Up", MISO/SPP)
- [ ] Fuel price scenario table (Section 5.8) with sources (EIA AEO, Henry Hub)
- [ ] Fuel → wholesale + emission rate linkage methodology (Section 5.9)
- [ ] Regional fuel-switching elasticity rationale
- [ ] CCS-CCGT complete cost buildup (capture + transport + storage + fuel − 45Q)
- [ ] Class VI well availability rationale by region
- [ ] 45Q tax credit mechanics and levelized impact
- [ ] LDES vs battery technology specs, efficiency, duration
- [ ] LDES dispatch algorithm description
- [ ] Battery dispatch algorithm description (preserved)
- [ ] CO2 emission factor methodology (eGRID marginal rates)
- [ ] CCS-CCGT partial credit methodology (90% capture)
- [ ] Abatement curve formulas (average and marginal)
- [ ] Hydro treatment rationale (existing-only, wholesale, regional caps)
- [ ] Two-tier pricing explanation
- [ ] Hourly wholesale price profile methodology
- [ ] Grid mix baseline methodology
- [ ] All cited sources: LBNL, NREL ATB 2024, Lazard, EIA AEO, eGRID, MISO/SPP, FERC/ISO reports

---

## 13. Regional Deep-Dive Pages (1 combined page)

All 5 regions covered in a single scrollytelling page with region selector.

### Structure
- **Status**: DELETED (Feb 19, 2026). Regional deep-dive content consolidated into research paper and homepage scrollytell.

### Default Cost Scenario for Static Pages
- **Homepage (index.html)** and **Regional Deep-Dive pages**: All figures and narrative use **Medium cost sensitivities** (all 5 toggle groups at Medium) unless a figure is explicitly designed to show Low/Medium/High ranges for comparison purposes.
- **Dashboard (dashboard.html)**: Interactive — user controls sensitivities via toggles.
- This ensures consistency across static narrative pages and reserves L/M/H range displays for intentional comparison figures (e.g., cost sensitivity deep-dive section #4 below).

### Per-Region Content Sections
1. **Region Overview** — grid composition, geography, market structure, current clean energy share
2. **Baseline Analysis** — existing grid hourly match score, seasonal patterns, strengths/weaknesses
3. **Decarbonization Pathway** — how optimal resource mix evolves from 75% → 100% threshold
4. **Cost Sensitivity Deep-Dive** — how L/M/H scenarios on key resources (regional priority resources) change the cost picture (this section explicitly shows ranges)
5. **Storage Role** — battery vs LDES contributions at different thresholds, dispatch patterns
6. **CO2 Abatement Profile** — regional abatement curves with commentary on inflection points
7. **Key Regional Insights** — unique factors (e.g., ERCOT's wind dominance, NYISO's nuclear fleet, CAISO's solar+storage, PJM's coal fleet switching dynamics)
8. **Comparison to National Context** — how this region compares to others

### Regional Priority Resources
- **CAISO**: Solar + battery, geothermal clean firm
- **ERCOT**: Wind + CCS-CCGT (Gulf Coast geology), low-cost solar
- **PJM**: Nuclear clean firm, coal→gas switching dynamics, wind
- **NYISO**: Nuclear fleet, hydro, limited renewables siting
- **NEISO**: Offshore wind potential, nuclear, limited solar

---

## 14. Research Paper PDF

### Format
- Generated as PDF (via HTML→PDF or direct HTML print stylesheet)
- Academic paper structure with executive summary
- Includes all regional deep-dive content as paper sections
- Includes full methodology detail (more than the methodology HTML page)

### Paper Structure
1. **Executive Summary** — key findings across all regions
2. **Introduction** — hourly CFE matching problem, why annual matching isn't enough
3. **Methodology** — full model description, all cost tables, algorithms, data sources
4. **National Results** — overview across all 5 regions, comparison charts
5. **Regional Deep-Dives** (5 sections, one per region — content from deep-dive pages)
6. **Sensitivity Analysis** — how key assumptions drive results
7. **Policy Implications** — what this means for procurement strategy
8. **Appendix** — full data tables, source citations, technical specifications

### Audience
- **Primary**: Business professionals with minimal energy domain knowledge
- **Secondary**: Academic/policy reviewers (must withstand scrutiny)
- Accessible first, rigorous underneath

---

## 15. Abatement Cost Comparison Page (NEW)

### Concept
A "Liebreich ladder for grid decarbonization" — analyzing when/where/under what conditions pushing grid decarbonization % is no longer cost-effective compared to alternative mitigation and carbon removal options. Linked from dashboard navigation.

**Status**: DELETED (Feb 19, 2026). Consolidated into `abatement_dashboard.html` (now "CO₂ Abatement Analysis").

**Core Question**: "Should we focus the next marginal dollar on the last 5% of PJM grid decarbonization, sustainable aviation fuel, or direct air capture?"

### Analysis Framework

**Y-axis**: Cost of carbon abatement ($/ton CO2)
**X-axis**: Cumulative abatement potential or grid % target

**Grid Decarbonization Curves** (from our model):
- Regional marginal abatement cost curves (75-100%) for each ISO under L/M/H sensitivities
- Show hockey-stick inflection where costs spike (typically 95-100%)
- Each region's curve under different sensitivity scenarios

**Comparison Benchmarks** (horizontal lines/bands on same chart):
| Mitigation Option | Low $/ton | Medium $/ton | High $/ton | Source |
|---|---|---|---|---|
| Energy efficiency (buildings) | $0 | $20 | $50 | IEA, McKinsey |
| Industrial electrification | $30 | $75 | $150 | IEA |
| Sustainable Aviation Fuel (SAF) | $150 | $250 | $400 | ICCT, BloombergNEF |
| Green hydrogen (industrial) | $100 | $200 | $350 | Liebreich, BNEF |
| BECCS | $100 | $175 | $300 | IPCC AR6 |
| Direct Air Capture (DAC) | $250 | $400 | $600+ | Carbon Engineering, Climeworks |
| Enhanced weathering | $50 | $125 | $200 | IPCC |
| Carbon credits (voluntary market) | $10 | $50 | $150 | Ecosystem Marketplace |
| EU ETS carbon price (2024-2025) | $60 | $80 | $100 | EMBER |

### Key Analytical Sections

1. **The Grid Decarbonization Curve** — Our model's regional marginal abatement curves plotted together. Where does each region's curve cross the DAC line? The SAF line?

2. **The Inflection Point Analysis** — For each region × sensitivity scenario, identify the % threshold where grid decarbonization costs exceed:
   - The social cost of carbon ($51/ton EPA, $185/ton Rennert et al.)
   - DAC costs ($250-600/ton)
   - SAF costs ($150-400/ton)
   - Voluntary carbon market prices ($10-150/ton)

3. **The Liebreich-Style Ladder** — Rank all mitigation options by cost-effectiveness at each grid % level. At 85% grid target, what's cheaper? At 95%? At 99%?

4. **Regional Divergence** — Some regions (ERCOT with cheap wind) stay cost-competitive deep into high %'s. Others (NYISO) become expensive earlier. Map the crossover points.

5. **The Net-Zero Pathway** — Given that residual emissions exist at any grid %, what's the optimal split between:
   - Pushing grid % higher (expensive past inflection)
   - Investing in DAC for residual emissions
   - Investing in other sectors (SAF, industrial) for cross-sector abatement

6. **Sensitivity Scenarios** — How do different cost assumptions shift the inflection points? Under low DAC cost assumptions, the crossover happens earlier. Under high renewable cost assumptions, same.

### Path-Dependency & Retroactive Cost Modeling (Under Development)

**Problem**: The optimizer independently optimizes each threshold. The 85% mix and 95% mix may differ fundamentally — heavy solar at 85%, heavy clean firm at 95%. Building the 85%-optimal mix then upgrading to 95% would strand solar assets and cost more than building toward 95% from the start.

**Proposed Approach**: Model backwards from the inflection point where the optimal grid solution's LCOE crosses a benchmark price:
- **Primary benchmark**: DAC cost projected to the target year
  - 2025: $400-600/ton → grid dominates through ~97%
  - 2035: $250-350/ton → grid dominates through ~93-95%
  - 2045: $150-250/ton → grid dominates through ~90-93%
- At the crossover threshold, the optimal mix is fixed. Then model the build-up path from lower thresholds using the cheapest-first resource ordering that converges to the crossover mix.
- DAC learning curve: ~15-20% cost reduction per doubling of deployment (ETH Zurich/Climeworks data)
- This creates a **declining optimal grid target over time** as removal costs fall — counterintuitive but logical.

**Status**: Waiting for optimizer results to analyze mix divergence between thresholds. If divergence is small (resources are additive), the current independent optimization is sufficient. If large (mix pivots between thresholds), path-constrained modeling is needed.

### DAC-VRE Co-Optimization Insight (Under Development)

**Core insight**: DAC is a flexible load that can absorb curtailed renewable energy. At high grid targets, significant curtailment occurs — this energy is nearly free ($0-5/MWh). DAC facilities co-located with sequestration geology (Class VI wells) can use curtailed power to remove CO₂ at dramatically reduced costs.

**Regional specialization model**:
- **ERCOT/CAISO**: Push grid to 95-97% (cheap wind/solar), operate DAC on curtailed surplus. Gulf Coast & Imperial Valley have Class VI well capacity.
- **PJM**: Push grid to 93-95%, buy ERCOT/regional DAC credits for residual emissions.
- **NYISO/NEISO**: Push grid to 90-92% (expensive beyond), heavy DAC credit procurement from regions with cheaper removal.

**DAC cost with curtailed power**: If energy is the #1 DAC cost driver (~60% of total), curtailed power at $0-5/MWh could reduce DAC from $400-600/ton to $150-250/ton — making it competitive with grid decarbonization costs above 93% in most regions.

**Analysis needed**:
1. From optimizer results: quantify curtailed MWh at each threshold × region
2. Estimate DAC capacity supportable by curtailed energy (assume 2 MWh/ton)
3. Derive DAC marginal cost curve as a function of curtailment availability
4. Compare DAC-on-curtailment cost to grid MAC at each threshold
5. Find the optimal regional grid target + DAC allocation

**Why this matters**: This reframes the "100% clean grid" question. If DAC-on-curtailment is cheaper than the last 5-10% of grid matching, the rational strategy is to overbuild VRE (creating more curtailment) and co-locate DAC — achieving net-zero at lower total cost than pure grid matching.

**DAC operating model: curtailment-only (20-30% CF)**:
- DAC as a **grid balancing tool** — runs only on otherwise-curtailed renewable energy
- This avoids adding demand that competes with other loads, which would drive up wholesale prices and congestion
- From a social good standpoint: DAC should not increase grid costs; it should absorb energy that would be wasted
- At 20-30% CF: CapEx/ton is 3-6x higher than full utilization, but energy cost is near-zero ($0-5/MWh)
- 2025 total: ~$835/ton (too expensive); **2040 projected: ~$360-410/ton** (competitive above 96-97% grid MAC)
- **2045-2050 projected: ~$280-320/ton** (competitive above 93-95% grid MAC in most regions)
- Standard DAC MAC curves assume ~90% CF — our model adjusts for curtailment-only operation
- Additional value not captured in per-ton cost: DAC provides grid stabilization by absorbing excess generation

**Abatement page section**: Dedicated section with narrative walkthrough + findings in the Key Insights panel at top.

**Implementation scope**: Supplementary analysis for the **abatement page only** — not the main dashboard.
- Run supplementary optimizer scenarios for ERCOT and CAISO (both have Class VI well capacity — Gulf Coast and Central Valley/Salton Sea respectively)
- Model allows monetizing curtailed energy via DAC-VRE co-location
- These regions push to 100%+ procurement with excess curtailment → DAC
- DAC credits offset residual emissions in NYISO/NEISO/PJM at high targets where grid costs are steep
- Produces a "cross-regional portfolio" where cheap-DAC regions export removal credits to expensive-grid regions

### Visual Design
- Large interactive chart: Regional MAC curves overlaid with benchmark bands
- Horizontal benchmark lines clearly labeled with color-coded bands
- Inflection point callouts where curves cross benchmarks
- Toggle: Region selector, sensitivity scenario
- Scrollytelling narrative explaining the analysis

### Audience
- Same business professional audience
- Build the case: "Here's when clean energy procurement stops being the cheapest path and alternatives become more efficient"
- Frame as strategic portfolio optimization, not just grid optimization

---

## 15b. Methodology & Research Paper (Merged)

- **research_paper.html** is now the single source of truth for methodology + research content
- **Appendix B** added with: B.1 Two-Tier Pricing Model, B.2 Generation Cost Tables, B.3 Storage Cost Tables, B.4 Transmission Adders, B.5 Sensitivity Toggle Pairing, B.6 CO₂ Emission Factor Methodology, B.7 Hydro Treatment
- **optimizer_methodology.html** preserved but removed from nav — all content consolidated
- Nav link: "Methodology & Paper" → research_paper.html
- Clickable table of contents at top of page

---

## 16. Header Banner & Navigation

### Banner Placement
- **Main dashboard**: Banner appears ABOVE intro text (not below)
- **All pages** (dashboard, regional deep-dives, methodology): Same header banner styling
- Banner includes page-specific name + tagline

### Per-Page Banner Content
| Page | Title | Tagline |
|---|---|---|
| Homepage (index.html) | The 8,760 Problem | Most climate solutions depend on a clean grid. But how clean is clean enough? |
| Cost Optimizer (dashboard.html) | Hourly CFE Optimizer | Advanced Sensitivity Model |
| CO₂ Abatement Analysis (abatement_dashboard.html) | CO₂ Abatement Analysis | Comparing Grid Decarbonization to Alternative Pathways |
| Methodology & Paper (research_paper.html) | Technical Methodology & Research Paper | Full Paper with Appendix B Cost Tables |

### Navigation (Updated Feb 14)
- Top navigation bar on ALL pages
- Links: Home | Cost Optimizer | Abatement Dashboard | Regional Deep Dives | CO₂ Abatement Summary | Methodology & Paper
- Current page highlighted in nav (nav-active class)
- Mobile: collapsible/hamburger nav
- "Back to Home" button at top of all non-home pages
- Methodology page (optimizer_methodology.html) still exists but removed from primary nav — content consolidated into research_paper.html Appendix B

---

## 17. Audience & UX Guidelines

### Dashboard (business professional audience)
- Layer in explanations for model elements, figures, toggles
- Tooltips or info icons (ⓘ) on each control explaining what it does and why it matters
- Chart titles that tell the story, not just label the axis
- Key finding boxes that translate numbers into business implications
- Assume reader does NOT know what LCOE, LCOS, capacity factor, or hourly matching mean

### Regional Deep-Dive Pages
- Written for a reader encountering the topic for the first time
- Build understanding progressively (scrollytelling)
- Lead with "so what" before diving into "how"
- Use analogies and real-world comparisons where helpful

### Research Paper / Methodology
- More technical depth acceptable
- Must still be accessible to first-time readers
- Withstand academic scrutiny: cite sources, show methodology, acknowledge limitations
- Full cost table transparency

---

## 17. QA/QC Requirements

### Optimizer Results QA (after first region completes)
- Validate hourly match scores against expected ranges from existing research
- Check that resource mixes make directional sense (e.g., wind-heavy in ERCOT, nuclear-heavy in PJM)
- Verify cost figures fall within published LCOE/LCOS ranges (NREL ATB, Lazard)
- Confirm CO2 abatement numbers are physically reasonable (tons displaced per MWh)
- Check that higher thresholds always cost more than lower ones (monotonicity)
- Verify storage dispatch increases with threshold (more storage needed at higher targets)

### Dashboard HTML QA
- Visual consistency: fonts, colors, spacing, alignment across all sections
- All toggles functional and responsive
- Chart rendering correct with proper labels, legends, axes
- Metric tiles update correctly when controls change
- No broken layouts at any control combination
- Clean/crisp visual identity — no cluttered elements

### Mobile Compatibility
- All figures render with readable text on mobile screens (320px-768px)
- Touch-friendly toggle controls (minimum 44px tap targets)
- Charts scale properly (responsive Canvas/Chart.js)
- Scrollytelling sections work on touch scroll
- No horizontal overflow or text truncation
- Navigation accessible on mobile (hamburger or stacked)
- Test at: 320px (small phone), 375px (iPhone), 768px (tablet)

### Pre-Push Checklist
- [ ] Optimizer results QA passed for all 5 regions
- [ ] All dashboard controls functional
- [ ] All charts render correctly
- [ ] Mobile compatibility verified
- [ ] Regional deep-dive pages complete and linked
- [ ] Research paper PDF generated
- [ ] Methodology page trimmed
- [ ] No console errors in browser
- [ ] Standalone HTML builds successfully
- [ ] All text readable at all viewport sizes

---

## 18. Summary Counts (v4.0)

| Item | Count |
|---|---|
| Resources (optimization dimensions) | 4 (clean_firm, solar, wind, hydro) — CCS merged into clean_firm |
| Resources (total modeled) | 6 (clean_firm incl. CCS, solar, wind, hydro, battery, LDES) |
| Thresholds | 13 (expanded from 10: added 50%, 60%, 70%) |
| Regions | 5 |
| Dashboard controls | 12 (2 existing + 7 graduated toggles + 1 binary + 2 region-conditional) |
| Sensitivity toggles | 7 graduated (L/M/H) + 1 binary (45Q On/Off) + 1 CAISO-only (Geothermal L/M/H) |
| Step 1 physics scenarios per region/threshold | 324 (3×3×3×3×4) — each independently co-optimized |
| Step 3 cost scenarios (non-CAISO) | 5,832 (3×3×3×3×2×3×4) per region/threshold |
| Step 3 cost scenarios (CAISO) | 17,496 (5,832 × 3 geothermal) per threshold |
| Total Step 3 evaluations | ~40,824 sensitivity combos × unique mixes per (region, threshold) |
| Pareto points per scenario | 3-5 (procurement/storage tradeoff frontier) |
| Regional deep-dive pages | 1 (combined, with region selector) |
| Research paper sections | 8 (including 5 regional deep-dives) |
| QA checkpoints | 3 (optimizer, HTML, mobile) |
| Output formats | 2 (JSON + Parquet) |

---

## 19. Model Limitations & Simplifying Assumptions

This section documents known simplifying assumptions for transparency and academic rigor. These should be acknowledged in the research paper and methodology page.

### 19.1 Static LDES LCOS (Utilization-Independent)

**Assumption**: LDES (100hr iron-air) uses a static LCOS ($/MWh) from published cost tables at assumed cycling frequency, regardless of the scenario's realized dispatch utilization.

**Why this matters**: LDES is extremely capital-intensive (~$5,000-10,000/kW installed at 100hr duration). The LCOS is dominated by capital recovery, so it is highly sensitive to utilization. A scenario where LDES cycles 50 times/year has a dramatically lower effective LCOS than one where it cycles 5 times/year — yet both use the same $/MWh in the model.

**Impact**: In scenarios with low LDES utilization (e.g., solar-dominant mixes with limited multi-day surplus), the model may understate the true cost of LDES. In scenarios with high utilization (wind-dominant mixes with abundant multi-day surplus to time-shift), the model may overstate LDES costs.

**Justification**: This approach is consistent with standard practice in published energy models (NREL ATB, Lazard LCOS). These sources quote LCOS at assumed utilization rates, and most capacity expansion models use static cost inputs without feedback from dispatch results. Implementing utilization-dependent LCOS would create a cost ↔ dispatch feedback loop (cost depends on dispatch, which depends on mix, which depends on cost) that, while convergent, adds significant methodological complexity. The same limitation applies to CCS-CCGT capacity factor effects on LCOE, though to a lesser degree given CCS's lower capital intensity per kW.

**Mitigation**: The optimizer's resource mix co-optimization partially self-corrects for this — it won't allocate large LDES shares in mixes that don't produce sufficient multi-day surplus to fill it, because the matching score won't benefit enough to justify the cost. The limitation is most relevant at the margin, where small LDES allocations face the highest effective cost per useful MWh.

### 19.2 CCS-CCGT at Assumed Baseload Capacity Factor

**Assumption**: CCS-CCGT LCOE reflects assumed high-capacity-factor baseload operation. In practice, CCS plants in a high-renewable grid might operate at lower capacity factors, increasing their effective LCOE.

**Impact**: Similar to LDES, the model may understate CCS-CCGT costs in scenarios where it operates at low utilization. However, since the optimizer models CCS as flat baseload (1/8760 profile), allocated CCS capacity runs at 100% CF by construction. The limitation applies to whether that assumption reflects real-world operations in a grid with significant renewable penetration.

**Mitigation**: The firm generation cost toggle (Low/Medium/High) provides sensitivity analysis around the LCOE assumption. High firm generation costs can be interpreted as a proxy for reduced capacity factor economics.

### 19.3 DST-Aware Solar Nighttime Correction (Fixed)

**Implementation**: Solar nighttime zeroing now accounts for Daylight Saving Time. The optimizer applies a 6am–7pm local prevailing time daylight window, converting to UTC using DST-adjusted offsets:
- **Standard time** (Nov–Mar): CAISO UTC+8, ERCOT UTC+6, PJM/NYISO/NEISO UTC+5
- **DST** (Mar–Nov, ~day 69–307): Offsets decrease by 1 (CAISO UTC+7, ERCOT UTC+5, PJM/NYISO/NEISO UTC+4)
- DST boundaries use representative dates across 2021–2025 (2nd Sunday of March ≈ day 69, 1st Sunday of November ≈ day 307)

**Optimizer matching**: Unaffected — `demand[h]` and `supply[h]` refer to the same physical UTC hour. DST adjustment only affects which hours get nighttime solar zeroing.

**If re-importing EIA data**: The `fetch_all_data.py` script stores all profiles in sequential UTC order. The DST correction is applied at profile loading time in the optimizer, NOT during data import. This means raw data files are always UTC and the DST logic lives only in `get_supply_profiles()`.

### 19.4 Multi-Year Data Usage (Implemented)

**Data split — what comes from where:**
- **2021-2025 average**: Hourly profile *shapes* for both generation (solar, wind, hydro, nuclear) and demand. Element-wise average across 5 years smooths single-year weather anomalies.
- **2025 actuals**: Total annual MWh (demand and generation), existing grid mix shares, hydro caps, peak demand. These anchor the model to current-year reality.
- **Solar nighttime correction**: Solar generation zeroed during nighttime hours using DST-aware local time windows (see §19.3).

**Leap year handling**: 2024 (8784 hours) is included by removing Feb 29 hours (indices 1416–1439) before averaging, preserving seasonal alignment with 8760-hour non-leap years.

**Data quality validation**: Demand profiles are validated at load time for statistical outliers using hour-of-day median comparison. Years where any hour-of-day has a maximum value exceeding 100× the median for that hour are excluded from the average. This catches EIA data entry errors (e.g., unit conversion errors that inflate individual hours by orders of magnitude). **Known exclusion**: PJM 2021 is excluded — October 19, 2021 hours 03:00-05:00 UTC contain demand values ~20,000× normal (0.31–0.44 of annual normalized demand concentrated in 3 hours), likely an EIA reporting error. PJM demand shape is averaged over 2022-2025 (4 years). All other ISO-year combinations pass validation. Raw data is preserved unmodified in `eia_demand_profiles.json` for auditability.

**Implementation in `load_data()`**:
- `_remove_leap_day(profile)`: Excises Feb 29 from 8784→8760
- `_validate_demand_profile(iso, year, profile)`: Detects corrupt years via hour-of-day outlier check
- `_average_profiles(yearly_profiles)`: Element-wise mean across years
- Generation profiles: `gen_profiles[iso][resource_type]` → direct access (no year key)
- Demand profiles: `demand_data[iso]['normalized']` uses averaged shape; `total_annual_mwh` and `peak_mw` from 2025

**Key constraint**: Profile shapes are weather-averaged; absolute quantities are 2025 actuals. This means the optimizer uses realistic hourly patterns (no single-year weather bias) scaled to actual 2025 generation levels.

**If re-importing EIA data**: The `fetch_all_data.py` script stores raw per-year profiles in `eia_generation_profiles.json` and `eia_demand_profiles.json`, each year-keyed. The 5-year averaging happens at optimizer load time, NOT during import. Raw data files preserve full per-year resolution for auditability.

### 19.5 NYISO Solar Proxy

**Status**: Working correctly. NYISO uses NEISO solar generation profile as proxy since NYISO lacks meaningful solar generation data in EIA 930. The proxy is stored in `eia_generation_profiles.json` as `solar_proxy` under NYISO and matches NEISO solar values exactly. The optimizer code (line 298-302) checks for `solar_proxy` first, falls back to NEISO solar.

---

## 20. Model Alignment and Differentiation vs. Existing Energy Models

This section documents how our model compares to established capacity expansion and procurement models, where we align with standard methodology, and where we deliberately diverge with justification.

### 20.1 Alignment with Standard Methodology

| Feature | Our Model | Industry Standard (GenX, ReEDS, SWITCH) | Alignment |
|---|---|---|---|
| **Hourly temporal resolution** | 8760 hours | 8760 hours (GenX), representative weeks (ReEDS), 12-288 time slices (SWITCH) | ✓ Matches GenX; exceeds ReEDS/SWITCH |
| **LCOS at reference utilization** | Static LCOS from NREL ATB/Lazard | Same — static cost inputs without dispatch feedback | ✓ Full alignment |
| **Solar/wind hourly profiles** | EIA 930 actual generation data, 5-year average | NREL ATB capacity factors, or NSRDB/WIND Toolkit | ✓ Comparable rigor; actual generation vs. modeled resource |
| **Two-tier pricing** | Existing capacity at wholesale; new-build at LCOE + transmission | Standard in procurement models (LevelTen, 3Degrees) | ✓ Full alignment |
| **Co-optimization of cost + mix** | Cost drives resource mix selection at every threshold | Standard in all capacity expansion models | ✓ Full alignment |
| **Regional granularity** | 5 ISOs (CAISO, ERCOT, PJM, NYISO, NEISO) | GenX: zonal; ReEDS: 134 BAs; SWITCH: load zones | ✓ Comparable scope for procurement analysis |

### 20.2 Deliberate Differentiations (with justification)

| Feature | Our Model | Standard Models | Why We Diverge |
|---|---|---|---|
| **CCS-CCGT as flat baseload** | Always-on, 100% CF | Dispatchable (ramps with system needs) | **45Q tax credit incentive**: $85/ton for captured CO₂ creates a strong economic incentive to maximize capacity factor regardless of grid need. The policy distortion means CCS would run baseload in practice, not dispatch. Standard models don't account for 45Q's perverse incentive structure. |
| **Nuclear seasonal derate** | Monthly flat derate from 5-year EIA data (spring/fall refueling) | Flat 90-93% annual CF (NREL ATB) or explicit outage scheduling (PLEXOS) | **Seasonal accuracy matters for hourly matching**: A flat annual CF misses the spring/fall refueling pattern where clean firm availability drops 15-20%. For hourly CFE procurement, this seasonal gap is exactly when storage or CCS must compensate. Our approach uses observed EIA data rather than assumed CF, and preserves high summer/winter availability when clean firm is most valuable. |
| **Storage capacity-constrained dispatch** | Capacity built = physical limit on daily/weekly dispatch | Varies: some use exogenous capacity, some co-optimize | **Prevents unrealistic dispatch**: The optimizer can't claim more storage dispatch than the built capacity allows. Days with insufficient surplus get partial cycles. This is more conservative than models that assume perfect foresight dispatch or exogenous capacity sizing. |
| **CO₂ hourly attribution with charge netting** | Track exact dispatch hours + net charge-side emissions | Flat marginal emission rate or annual average | **Prevents CO₂ overcounting**: Storage charging from fossil-marginal hours carries real emissions. Our approach credits storage abatement only for the net emission reduction, not the gross displacement. This is consistent with the GHG Protocol Scope 2 hourly matching framework. |
| **Hydro as existing-only** | Capped at 5-year average share, wholesale-priced, no new-build | Varies: some allow new-build hydro/pumped storage | **Reflects procurement reality**: New conventional hydro is effectively unavailable in the US (permitting, environmental constraints). Treating it as existing-only matches what a corporate buyer can actually procure. |
| **Procurement-focused objective** | Minimize $/MWh to achieve target CFE % | Minimize total system cost or maximize welfare | **Different question**: We're asking "what should a buyer procure?" not "what should the system build?" This means we don't model transmission expansion, retirement decisions, or inter-regional trade — we take the grid as-given and optimize the buyer's clean energy portfolio within it. |

### 20.3 Key Assumptions Where We Use Standard Values

- **Battery**: 4hr Li-ion, 85% RTE, daily-cycle dispatch → NREL ATB 2024 reference
- **LDES**: 100hr iron-air, 50% RTE → Form Energy published specs, NREL ATB storage module
- **CCS capture rate**: 95% → DOE/NETL reference for next-gen CCGT+CCS (conservative vs. 90% in older literature)
- **45Q offset**: $85/ton × 95% capture × ~0.37 tCO₂/MWh ≈ $29/MWh LCOE reduction → IRC §45Q(a)(3)(A)
- **Discount rate**: Implicit in LCOE tables (NREL ATB uses WACC by technology)
- **Transmission adders**: Regional, based on published interconnection queue data and MISO/PJM/CAISO tariff filings

### 20.4 What Our Model Does NOT Include (Scope Boundaries)

- **Transmission expansion or congestion** — we use existing interconnection costs
- **Retirement/entry decisions** — we take the existing grid as a given
- **Inter-regional trade / import-export** — each ISO is modeled as self-contained. Unmatched demand hours are assumed met by fossil generation priced at regional fossil cost sensitivities (coal/gas/oil). We do not consider interconnection or power flows across grid boundaries. This is a meaningful simplification for ISOs that rely on imports (e.g., CAISO imports from Pacific NW hydro, NYISO imports from Quebec/Ontario hydro). The effect is that our model may slightly overstate the difficulty of meeting high CFE thresholds in import-dependent regions.
- **Demand response or demand flexibility** — demand is fixed hourly profile
- **Hydrogen storage** — explicitly excluded (immature for grid-scale energy storage)
- **Multi-year capacity planning** — single 2025 snapshot, not a trajectory
- **Reliability/adequacy constraints (ELCC)** — under consideration (Section 21.1)
- **NEISO gas delivery constraints** — New England has well-documented natural gas pipeline constraints (Algonquin pipeline congestion during winter peaks). This creates winter gas price spikes that aren't captured by our flat L/M/H gas price sensitivity. Future iteration should model seasonal gas price multipliers or a NEISO-specific winter gas adder. See §21.3.
- **BECCS (Bioenergy with CCS)** — Not modeled in current version. Relevant for NEISO where high CCS shares (50%+ at 92.5%) suggest a natural use case. BECCS could offer negative emissions AND firm dispatchable generation. Future post-processing: derate CCS scenarios with a BECCS cost overlay to avoid full re-optimization. See §21.3.

---

## 21. Planned Enhancements

### 21.1 Capacity Reserve Margin / ELCC (Under Consideration)

**Concept**: Layer in a capacity reserve margin constraint using Effective Load Carrying Capability (ELCC) to ensure resource mixes maintain grid reliability.

**What ELCC does**: ELCC measures the firm capacity contribution of each resource type — how much peak demand it can reliably serve. Variable resources (solar, wind) have lower ELCC than their nameplate capacity because they may not generate during peak demand hours.

**Typical ELCC values** (from NREL/regional ISO studies):
| Resource | ELCC Range | Notes |
|---|---|---|
| Clean Firm (nuclear) | 90-95% | Near-firm, planned outages reduce |
| Solar | 30-70% | Varies by region, declines with penetration |
| Wind | 10-30% | Highly region-dependent |
| CCS-CCGT | 85-95% | Dispatchable, similar to CCGT |
| Battery (4hr) | 60-95% | Duration-limited; declines as peak broadens |
| LDES (100hr) | 85-95% | Long duration → high capacity value |

**Implementation approach**: Add a constraint to the optimizer that the ELCC-weighted capacity of the resource mix must meet a minimum reserve margin (e.g., 15% above peak demand). This would:
- Prevent resource mixes that meet hourly matching targets but lack capacity adequacy
- Penalize solar-heavy mixes at high thresholds (solar ELCC drops with penetration)
- Favor firm resources and storage at the margin
- Better reflect real planning constraints

**Complexity**: Moderate. The ELCC calculation is a post-hoc check on each candidate mix during optimization. The main challenge is ELCC values that decline with penetration (saturation effects), which creates non-linear constraints. A simplified version could use fixed ELCC percentages per resource type.

**Decision**: Under consideration — user to confirm whether to implement for next optimizer run.

### 21.2 Post-Processing from Cached Results (Decided — Future Iterations)

**Principle**: After monotonicity sweeps are complete, use cached results (resource mixes + score caches + optimizer_cache.json) for future post-processing instead of full re-optimization runs. This enables:
- Rapid sensitivity analysis (re-price existing mixes under new cost assumptions)
- BECCS overlays (derate CCS scenarios with BECCS costs)
- Gas constraint scenarios (apply winter price multipliers to existing mixes)
- Carbon price sensitivity (overlay SCC/ETS prices on existing results)

**How**: `optimizer_cache.json` stores the full co-optimized results for all 16,200 scenarios. `compute_costs_parameterized()` can re-price any cached mix in milliseconds. Only changes that fundamentally alter the optimization landscape (new resource types, new dispatch algorithms, new constraint structures) require full re-runs.

**Two-file architecture** (decided):
1. **`data/optimizer_cache.json`** — Raw, untouched optimizer output. Never modified after a run. This is the canonical record of what the optimizer produced. Includes all resource mixes, costs, scores, metadata.
2. **`dashboard/overprocure_results.json`** — Post-processed copy that feeds the live dashboard. Derived from the cache + any post-processing overlays (BECCS derating, gas adjustments, etc.). Can always be regenerated from the cache.

Any post-processing (cost overlays, BECCS, gas constraints, carbon pricing) operates on a copy of the cache data and writes to the dashboard results file. The raw cache is always preserved as ground truth.

### 21.3 NEISO Gas Pipeline Constraints + BECCS (Future Iteration)

**NEISO Gas Delivery Constraints**: New England has severe natural gas pipeline constraints, particularly on the Algonquin City Gate pipeline during winter peaks. Key literature:
- **ISO-NE Operational Fuel Security Analysis (2018)**: Documented reliability risk from winter gas constraints; gas generators unable to secure fuel during cold snaps
- **Algonquin basis differentials**: Winter spot gas prices in New England can spike to $20-30/MMBtu (vs. $3.50 Henry Hub medium), reflecting pipeline congestion
- **Grid-scale impact**: During 2017-2018 "bomb cyclone", New England gas prices exceeded $30/MMBtu, oil generation surged to 30%+ of total
- **Current model limitation**: Our flat L/M/H gas price sensitivity ($2/$3.50/$6 MMBtu) doesn't capture this seasonal volatility. The "High" gas scenario ($6) still understates winter peaks by 3-5x.

**Potential fix for future iteration**: Apply a NEISO-specific seasonal gas price multiplier (e.g., 3-5x during Dec-Feb) or model a winter gas constraint that caps gas-fired generation availability. This would increase the value of non-gas firm resources (nuclear, BECCS) and storage in NEISO.

**BECCS for NEISO**: Current optimizer shows NEISO needs 50%+ CCS at 92.5% matching. This creates a natural use case for BECCS (Bioenergy with CCS):
- BECCS provides firm dispatchable generation (like CCS-CCGT) PLUS negative emissions
- NEISO has significant forestry biomass resource (wood pellets, forestry residues)
- Cost estimate: ~$120-180/MWh LCOE (NREL ATB) — higher than CCS-CCGT but with carbon-negative value
- **Post-processing approach**: For scenarios with high CCS share (>25%), run a cost overlay replacing a fraction of CCS with BECCS pricing. Include negative emissions credit at SCC values ($51-185/tCO2). This avoids full re-optimization — just re-price cached mixes.

**Decision**: Implemented in post-processing (Feb 15, 2026). See §22.

---

## 22. Post-Processing Corrections & Overlays (Feb 15, 2026)

Applied to Step 3 cost optimization results via `step4_postprocess.py`. Corrected results written to `dashboard/overprocure_results.json`.

### 22.1 CO₂ Monotonicity Enforcement

**Problem**: CO₂ abatement is non-monotonic across thresholds in 4 of 5 ISOs. Higher hourly match targets can result in LESS CO₂ abated (up to -15.3M tons in ERCOT 90%→92.5%). Root cause: the optimizer minimizes cost, not CO₂. A cheaper mix at a higher threshold may procure less total clean energy (substituting storage for overprocurement), reducing total fossil displacement even as temporal matching improves.

**Fix**: Running-max constraint — `co2_corrected[t] = max(co2[t], co2[t-1])` across thresholds. Ensures abatement narrative never shows "paying more for less CO₂."

### 22.2 45Q Offset Correction

**Problem**: Model calculates 45Q credit as $85/ton × 0.34 tCO₂/MWh = $29/MWh. Correct calculation: $85 × 0.34 × 0.95 (captured only) = $27.5/MWh. Overstated by ~$1.5/MWh.

**Fix**: Adjust CCS LCOE by +$1.5/MWh across all scenarios. Negligible impact on results.

### 22.3 Without-45Q Toggle Layer

**Design**: Dashboard toggle "45Q Credit: On / Off" showing cost impact of removing the 45Q incentive from CCS-CCGT.

**Without-45Q CCS cost model**:
- Remove $27.5/MWh 45Q offset from CCS LCOE
- Model CCS as **dispatchable** (not baseload) — without 45Q, there's no perverse incentive to maximize captured CO₂ by running 24/7
- CCS LCOE becomes **capacity-factor-dependent**: at lower CF, capital recovery per MWh increases

**CCS LCOE decomposition** (from NETL Baseline Rev 4a):
- Capital recovery: 55% of LCOE (scales inversely with CF)
- Fixed O&M: 8% of LCOE (scales inversely with CF)
- Fuel: 30% of LCOE (constant per MWh)
- Variable O&M + T&S: 7% of LCOE (constant per MWh)
- Reference CF: 85% (NETL standard)

**CF-dependent formula**: `LCOE(CF) = LCOE_no45q × ((0.63 × 0.85 / CF_actual) + 0.37)`

**CCS vs LDES crossover**: At each region's Medium costs, the CF at which CCS-without-45Q equals LDES cost. Below this CF, LDES is cheaper. This determines whether CCS would ever be built without 45Q.

**Implementation**: For each cached scenario, recalculate costs assuming no 45Q. CCS mix share implies an effective CF that determines the dispatchable LCOE. Compare to what the cost would be if CCS share were replaced by LDES or additional clean firm.

### 22.4 NEISO Winter Gas Pipeline Constraint

**Problem**: NEISO has structural winter gas price spikes due to Algonquin Citygates pipeline congestion. Winter spot prices historically $15-30/MMBtu vs. ~$5-6/MMBtu annual average. The model's flat L/M/H gas sensitivity ($2/$3.50/$6 MMBtu) understates NEISO winter costs by 3-5×.

**Post-processing approach**:
- Winter months (Dec-Feb, ~25% of year): +$7.50/MMBtu above annual average (midpoint of $5-10 range)
- CCS fuel impact: 7 MMBtu/MWh heat rate × $7.50 × 0.25 = **+$13.13/MWh annualized CCS adder** for NEISO
- Wholesale impact: gas-on-margin × winter premium → **+$4/MWh annualized wholesale adder** for NEISO
- Applied to NEISO results only; all other ISOs unaffected

**Sources**: ISO-NE Operational Fuel Security Analysis (2018), Algonquin Citygates historical basis differentials, 2017-2018 bomb cyclone gas pricing data.

### 22.5 ERCOT Battery LCOS Low ($69/MWh) — Retained

**Finding**: The $69/MWh ERCOT Low battery LCOS lacks a peer-reviewed citation. It was set based on regional qualitative factors (low labor costs, fast permitting, flat terrain, minimal unionization, extensive solar co-location potential).

**Decision**: Retain $69/MWh. ERCOT is genuinely the lowest-cost US market for battery deployment. Lazard's national unsubsidized range ($115-$254) reflects high-cost assumptions (80% equity at 12% return) and diverse geographies. ERCOT-specific conditions (non-ERCOT interconnection queue, streamlined permitting, LFP oversupply benefiting Texas ports) justify costs below national averages. The Low case explicitly represents an optimistic-but-plausible scenario.

**Mitigation**: Document in research paper that regional battery cost differentiation is based on qualitative assessment of market conditions, not published regional cost studies. Note that all Low-case costs represent aggressive forward trajectories.

### 22.6 Post-Processing Peer Review Fixes (Feb 15, 2026)

**Findings from third-party code review:**

1. **`costs_detail` sync** — `fix_45q_offset()` was updating `scenario['costs']` but not `scenario['costs_detail']` for Medium scenarios (MMM_M_M), causing a data inconsistency where the dashboard's detail views showed stale pre-correction numbers. **Fixed**: Now syncs `effective_cost_per_useful_mwh`, `total_cost_per_demand_mwh`, `incremental_above_baseline`, and `baseline_wholesale_cost` between both dicts.

2. **Crossover edge-case comment** — When `rhs ≤ 0` (LDES variable cost alone exceeds LDES cost), the comment incorrectly stated "CCS always cheaper." **Fixed**: Corrected to "LDES always cheaper."

3. **Dead import** — `import copy` was unused. **Removed.**

### 22.7 Gas Availability Factor (GAF) — Resource Adequacy Deration (Feb 20, 2026)

**Problem**: The model assumed 100% gas availability at peak — if gas backup = 10,000 MW needed, exactly 10,000 MW was built. This contradicts all ISO practice and empirical evidence. Gas plants experience both independent forced outages (EFORd ~5-7%) and correlated failures during extreme weather events (Winter Storm Uri: 49% outage; Elliott: 24% outage). PJM's 2024 ELCC methodology rates gas CCGT at ~80% effective capacity.

**Fix**: Divide raw gas backup requirement by an ISO-specific Gas Availability Factor (GAF):
```
gas_needed_mw = max(0, ra_peak - clean_peak) / GAF
```

**ISO-specific GAF values** (applied in both Step 3 cost optimization and Step 4 post-processing):

| ISO | GAF | Deration | Rationale |
|-----|-----|----------|-----------|
| CAISO | 0.88 | 12% | Summer ambient derate + mechanical outages |
| ERCOT | 0.83 | 17% | Extreme weather both seasons, gas supply correlation |
| PJM | 0.82 | 18% | PJM ELCC data, Winter Storm Elliott evidence |
| NYISO | 0.82 | 18% | Pipeline constraints, winter gas competition |
| NEISO | 0.85 | 15% | Mechanical + weather only (pipeline handled separately) |

**NEISO note**: GAF captures only mechanical/weather unavailability. The pipeline capacity constraint is structurally different — an absolute MW ceiling, not a proportional derate — and is modeled separately (see §22.8).

**Sources**: PJM ELCC Class Ratings (2024/25), NERC GADS EFORd class averages, FERC Final Reports on Winter Storm Uri (2021) and Elliott (2022), Brattle Group VRR Curve Review (2025), UCS gas reliability analyses, ERCOT Aurora RA Assessment (2025).

**Impact on optimization**: GAF increases gas backup MW requirements by 12-18% across ISOs, which increases gas backup costs. This tilts cost-optimal mixes toward resources with higher peak capacity credits (clean firm, CCS, battery) and away from resources with low capacity credits (solar, wind) at high matching thresholds. The effect is modest at low thresholds (gas backup is small) and material at 95%+ (where gas backup costs are a significant fraction of total cost).

### 22.8 NEISO Pipeline Capacity Constraint — Informational Metric (Feb 20, 2026)

**Problem**: NEISO's gas constraint is an absolute physical ceiling (~4.5 BCF/day total pipeline capacity; ~1.5 BCF/day available for power generation after heating demand during winter peak), not a proportional deration. Building more gas plants doesn't help if the pipeline can't feed them. As demand grows, the constraint worsens (same pipeline, more load).

**Approach**: Compute as a downstream informational metric, NOT integrated into the optimization. For each NEISO scenario:
1. Compare gas backup MW (post-GAF) against pipeline-deliverable gas MW ceiling (8,300 MW)
2. If gas exceeds pipeline capacity: compute shortfall MW and annualized pipeline expansion cost
3. Output as `pipeline_constraint` sub-dict in gas_backup results

**Constants**:
- Pipeline-deliverable gas at peak: **8,300 MW** (1.5 BCF/day ÷ 7.5 MMBtu/MWh heat rate)
- Pipeline expansion cost: **$2,400/MW-yr** annualized ($150M/BCF-day, 30yr at 8% WACC)

**Source**: ISO-NE Gas Availability Study (2025), FERC pipeline project filings.

**Rationale for informational-only**: The pipeline constraint is structural and binary — scenarios either exceed the ceiling or don't. Baking it into the optimization would distort mix selection by treating a New England infrastructure policy question as an engineering parameter. Instead, it's presented as: "this scenario requires X MW of gas backup, but the pipeline can only deliver 8,300 MW — here's what closing that gap would cost."

4. **CCS CF estimation floor** (documented limitation) — The 0.20 minimum CF floor in `ccs_lcoe_dispatchable()` may understate no-45Q costs for small CCS shares (where actual CF might be 0.08-0.15). Without hourly dispatch data in the results JSON, we can't improve this in post-processing. Documented as a conservative (cost-understating) assumption.

5. **No-45Q mix bias** (documented limitation) — The no-45Q overlay reprices the same resource mix that was co-optimized WITH 45Q. This mix over-represents CCS, making the no-45Q cost a conservative upper bound. A true no-45Q re-optimization would substitute LDES/renewables for CCS, yielding lower costs.

### 22.7 100% Hourly Match Asymptote — Literature Review & Procurement Bounds

**Key literature findings:**
- NREL (Cole et al., 2021, Joule): Marginal abatement cost 99%→100% = **$930/ton** — 15× the average cost of the full 100% target. Nonlinear in all 22 sensitivities tested.
- Riepin & Brown (2024, Energy Strategy Reviews): 98% CFE = 54% premium over annual matching. 100% doubles costs again. With clean firm + LDES, 100% premium drops to just 15%.
- Peninsula Clean Energy MATCH Model (2023): 99%→100% requires **34% more supply**, +10% portfolio cost. 0%→99% costs only +2%.
- Budischak et al. (2013, J. Power Sources): Cost-optimal 99.9% requires ~280% nameplate capacity. "Least cost solutions yield seemingly-excessive generation capacity."
- WattTime: 100% hourly matching may require PPAs for **up to 400%** of annual consumption.

**Granularity consensus:** The 90-100% zone needs 2.5% resolution minimum. Our threshold set (90, 92.5, 95, 97.5, 99, 100) is well-aligned with literature practice.

**Procurement bound assessment:**
- Current bound: 200% of demand
- Actual usage at 99%: max 135% (CAISO), 130% (NYISO), 125% (NEISO), 123% (PJM), 118% (ERCOT)
- 100% threshold: 0 feasible scenarios found (all ISOs)
- Max hourly match achieved: 99.6% (PJM at 123% procurement)
- **Decision**: If rerunning for 100%, increase upper bound to **250%** based on literature support (Budischak 280%, WattTime 400%). The 200% bound is sufficient for ≤99% targets.

**Archetype diversity in cache:**
- 46–70 unique resource mix archetypes per ISO across all thresholds
- Only 4–14 unique mixes per threshold (massive redundancy across 5,832 scenarios)
- Cache comprehensively covers the feasible solution space — new constraint runs can seed from existing archetypes rather than cold-start
