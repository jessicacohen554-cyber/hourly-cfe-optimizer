# Advanced Sensitivity Model — Complete Specification

> **Authoritative reference for all design decisions.** If a future session needs context, read this file first.
> Last updated: 2026-02-14.

## Current Status (Feb 15, 2026)

### What was accomplished
- [x] Homepage (`index.html`) — 4 charts rendering with real data, region toggle pills, narrative sections
- [x] Carbon Abatement Dashboard (`abatement_dashboard.html`) — 3 charts (MAC, portfolio, ladder) fully rendering with hardcoded illustrative data + 4 stress-test toggles
- [x] Navigation site-wide: Home | Cost Optimizer | Abatement Dashboard | Regional Deep Dives | CO₂ Abatement Summary | Methodology & Paper
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

### Pipeline when optimizer completes
1. Run `recompute_co2.py` → hourly CO₂ correction
2. Run `analyze_results.py` → monotonicity, literature alignment, VRE waste, DAC inputs
3. Update dashboards with real data, update narratives
4. DAC-VRE analysis, resource mix analysis, path-dependent MAC
5. Commit + push

### Pre-Run QA/QC Gate (Mandatory Before Every Optimizer Run)
**This gate exists because**: a previous run wasted 3+ hours of compute due to incorrect hydro caps that weren't caught before launch. Every optimizer run is expensive — never launch without verifying assumptions first.

Before launching `optimize_overprocure.py`, the following must be verified:
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
- [ ] **Tone down Constellation-specific narrative** — make discussion generic about archetypes (nuclear-led, coal-heavy, gas-dominant, mixed fleet). Remove "unfairness" language. Analysis shouldn't sound biased.
- [ ] **Add GHG Protocol Scope 2 revision context** — hourly accounting proposed in Scope 2 revision creates demand-side pull for deep decarb vs annual accounting
- [ ] **Add EPRI SMARTargets context** — addresses SBTi criticisms, company-specific targets considering regional constraints
- [ ] **Add hourly RPS discussion** — how hourly RPS targets could catalyze generator deep decarbonization via demand-side pull
- [ ] **Policy page: RPS + corporate demand under hourly matching** — projected hourly clean premium cost curves in 5 ISOs
  - **Standard supply service baseline**: Nuclear, RPS, rate base clean gen, publicly owned assets = "standard supply service" per GHG Protocol Scope 2 revisions. These count toward corporate hourly matching on the front end.
  - **Corporate voluntary layer**: Corporate buyers add procurement on top of standard supply service baseline. Model at 10% intervals of corporate participation share.
  - **Corporate buyers compete with state RPS** for existing clean resources — both draw from same pool
  - Sources: CEBA, RE100, CDP, NREL voluntary procurement, Berkeley National Labs RPS data
- [ ] **EAC scarcity analysis by ISO** — core question: which ISOs see EAC scarcity that drives up hourly clean premiums?
  - **Assumption**: Corporate procurement must occur within the ISO where load is (no cross-ISO claims)
  - **No incrementality constraint assumed** on voluntary Scope 2 claims initially
  - **PJM**: No renewables left after RPS compliance currently. Excess nuclear after state programs, but data center PPAs (Microsoft-Constellation Crane, etc.) consuming nuclear surplus. If data center demand growth materializes → scarcity → higher EAC prices
  - **ERCOT**: Abundant wind+solar being added rapidly, no state RPS. May never see scarcity regardless of incrementality requirements
  - **CAISO, NYISO, NEISO**: Analyze each for supply-demand balance of clean EACs after RPS + data center growth
  - Key output: projected hourly clean premium cost curve at 10% corporate participation intervals × hourly matching targets

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
- **Dev branch**: `claude/enhance-optimizer-pairing-k0h9h`

---

## 2. Resources (7 total)

| # | Resource | Profile Type | New-Build? | Cost Toggle? | Transmission Adder? |
|---|---|---|---|---|---|
| 1 | **Clean Firm** (nuclear/geothermal) | Seasonal-derated baseload | Yes | Low/Med/High (regional) | Yes (regional) |
| 2 | **Solar** | EIA 2025 hourly regional | Yes | Low/Med/High (regional) | Yes (regional) |
| 3 | **Wind** | EIA 2025 hourly regional | Yes | Low/Med/High (regional) | Yes (regional) |
| 4 | **CCS-CCGT** | Dispatchable baseload (flat) | Yes | Low/Med/High (regional) | Yes (regional) |
| 5 | **Hydro** | EIA 2025 hourly regional | **No** — capped at existing | **No** — wholesale only | **No** — always $0 |
| 6 | **Battery** (4hr Li-ion) | Daily cycle dispatch | Yes | Low/Med/High (regional) | Yes (regional) |
| 7 | **LDES** (100hr iron-air) | Multi-day/seasonal dispatch | Yes | Low/Med/High (regional) | Yes (regional) |

### Key resource decisions:
- **H2 storage excluded** (explicitly out of scope)
- **Clean Firm nuclear derate**: Seasonal spring/fall derate applied to nuclear portion only (not geothermal). Reflects staggered refueling outages across the fleet (~18-24 day outages per plant every 18-24 months, distributed across spring/fall shoulders). Summer/winter: ~100% CF (nukes run full during peak demand seasons). Spring/fall: reduced CF based on observed EIA 2021-2025 nuclear generation vs. available capacity. Derive simplified flat seasonal percentages from actual data. Geothermal (relevant in CAISO) stays flat 1/8760.
- **Hydro**: Existing only, capped at regional capacity, wholesale priced, no new-build tier, $0 transmission
- **CCS-CCGT**: 95% capture rate, residual ~0.0185 tCO2/MWh, 45Q ($85/ton = ~$29/MWh offset) baked into LCOE, fuel cost linked to gas price toggle. **Modeled as flat baseload (not dispatchable) by design** — while CCS-CCGT is physically dispatchable, the 45Q tax credit ($85/ton for geologic storage) incentivizes running at maximum capacity factor to maximize capture credits. This is an economics-driven decision, not a physical constraint. The perverse policy incentive drives gas demand even when the grid doesn't need it. Not worth the compute to model dispatchability when the economics don't support it today.
- **LDES**: 100-hour iron-air, 50% round-trip efficiency, capacity-constrained dispatch with dynamic capacity sizing. LCOS reflects actual utilization of built capacity.
- **Battery**: 4-hour Li-ion, 85% round-trip efficiency, capacity-constrained daily-cycle dispatch. LCOS reflects actual utilization — oversized capacity that sits idle drives cost up.

---

## 3. Thresholds (10 total — reduced from 18)

```
75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 100
```

- 5% intervals from 75-85 (captures broad trend)
- 2.5% intervals from 87.5-97.5 (captures steep cost inflection zone)
- 99% and 100% anchor the extreme end
- Reduced from 18 to 10 while adding granularity in the inflection zone
- Key inflection behavior (CCS/LDES entering mix, storage costs spiking) captured at 90-97.5
- Dashboard interpolates smoothly between these anchor points for abatement curves

---

## 4. Dashboard Controls (7 total — paired toggles)

### Preserved (2):
1. **Region/ISO select** (CAISO, ERCOT, PJM, NYISO, NEISO)
2. **Threshold select** (10 values: 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 100)

### Paired sensitivity toggles (5 groups replacing 10 individual):

The 10 individual cost toggles are **paired into 5 groups** where related variables move in lockstep (all Low, all Medium, or all High together). This reduces dashboard complexity and scenario count while preserving meaningful sensitivity analysis.

| # | Paired Toggle | Options | Member Variables | Affects |
|---|---|---|---|---|
| 3 | **Renewable Generation Cost** | Low / Medium / High | Solar LCOE + Wind LCOE | Both solar and wind generation costs (regional) |
| 4 | **Firm Generation Cost** | Low / Medium / High | Clean Firm LCOE + CCS-CCGT LCOE | Both firm dispatchable resource costs (regional) |
| 5 | **Storage Cost** | Low / Medium / High | Battery LCOS + LDES LCOS | Both storage technology costs (regional) |
| 6 | **Fossil Fuel Price** | Low / Medium / High | Gas + Coal + Oil prices | Wholesale electricity price + CCS fuel cost + emission rates |
| 7 | **Transmission Cost** | None / Low / Medium / High | All resource transmission adders | Transmission adders on all new-build resources (regional) |

**Pairing rationale**:
- **Renewable Gen**: Solar and wind costs are driven by similar factors (manufacturing scale, supply chain, installation labor) — they tend to move together directionally
- **Firm Gen**: Clean firm (nuclear/geothermal) and CCS-CCGT share capital-intensive, long-lead-time cost structures
- **Storage**: Battery and LDES costs share manufacturing/materials cost drivers (lithium, iron, electrolyte supply chains)
- **Fossil Fuel**: Gas, coal, and oil prices are correlated through energy commodity markets and macro conditions
- **Transmission**: Infrastructure costs affect all new-build resources similarly within a region

**Scenario count**: 3 × 3 × 3 × 3 × 4 = **324 cost scenarios** per region per threshold (vs 59,049 if all 10 toggles independent)

**NOTE**: All toggles use **Low / Medium / High** naming consistently (never "Base" or "Baseline").

**Optimizer approach**: Resource mix co-optimized with costs for EVERY scenario. Different cost assumptions produce different optimal resource mixes — this is the core scientific contribution. 324 scenarios × 10 thresholds × 5 regions = 16,200 independent co-optimizations, with matching score cache shared across cost scenarios (physics reuse, not a shortcut).

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
2. **Phase 2 neighborhood radius**: The 5% step with radius 2 covers ±10% in each resource dimension from the warm-start mix. Optimal mixes more than 10% away in any dimension from all seed archetypes would be missed. **Mitigation**: Edge-case seeds (100% solar, 100% wind, etc.) are always included regardless of warm-start. At observed convergence rates, ≤14 unique mixes typically serve 324 scenarios at lower thresholds, well within the archetype pool's coverage.
3. **Threshold-dependent risk**: Higher thresholds (95-100%) have more diverse optimal mixes across cost scenarios. **Mitigation**: The archetype pool grows dynamically; extreme scenarios are more likely to diverge at high thresholds, populating the pool with the right seeds.
4. **Not used during re-sweep**: Monotonicity re-sweep always uses full Phase 1 (warm_start_result is not passed when resweep=True). This is intentional — re-sweep needs the broadest possible search to resolve violations.

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

### 5.3 Clean Firm LCOE ($/MWh) — Blended Uprate + New-Build, Regionalized

Clean firm LCOE is a **capacity-weighted blend** of nuclear uprate costs and new-build costs, reflecting that the first incremental clean firm capacity a buyer would procure comes from uprating existing plants (much cheaper) before requiring new-build SMRs.

**Nuclear uprate LCOE** (incremental cost of adding capacity to existing plants):
| Level | LCOE ($/MWh) | Basis |
|---|---|---|
| Low | $15 | MUR-dominated (measurement recapture, minimal capital) |
| Medium | $30 | Typical EPU blend (extended power uprate) |
| High | $55 | Large-scope EPU (major equipment replacement) |

*Sources: INL LWRS Program, NRC uprate database, NEI fleet data, Vistra/Meta 2026 PPA filings*

**Regional uprate share** (fraction of new clean firm that comes from uprates vs. new-build):
| Region | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|
| Uprate share | 15% | 25% | 50% | 30% | 20% |

- **PJM**: 50% — 32 GW nuclear fleet, largest uprate pool (Constellation ~1GW, Vistra/Meta ~433MW announced)
- **NYISO**: 30% — smaller fleet (3.4 GW) but FitzPatrick, Nine Mile Pt have uprate potential
- **ERCOT**: 25% — South Texas Project (2.7 GW), MUR/stretch potential
- **NEISO**: 20% — Millstone + Seabrook, limited fleet
- **CAISO**: 15% — Diablo Canyon only (2.3 GW), minimal uprate headroom + geothermal is the cheap option

**New-build component LCOE** (SMR/Gen III+ for capacity beyond uprates):
| Level | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|
| Low | $65 | $70 | $80 | $85 | $82 |
| Medium | $88 | $95 | $105 | $110 | $108 |
| High | $125 | $135 | $160 | $170 | $165 |

**Blended clean firm LCOE** (uprate_share × uprate + (1 - uprate_share) × new_build):
| Level | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|
| Low | $58 | $56 | $48 | $64 | $69 |
| Medium | $79 | $79 | $68 | $86 | $92 |
| High | $115 | $115 | $108 | $136 | $143 |

*CAISO benefits from geothermal (lower new-build component). PJM gets the biggest discount due to largest uprate pool. NYISO/NEISO remain expensive due to smaller fleets and higher new-build costs.*

**Why this matters**: Pure new-build clean firm LCOE ($93-150/MWh at Medium) overstates the cost of incremental clean firm capacity because it ignores the ~5 GW of uprate potential available at $15-55/MWh. The blended approach reflects the actual procurement cost curve a buyer would face.

### 5.4 CCS-CCGT LCOE ($/MWh, net of 45Q)

| Level | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|
| Low | $58 | $52 | $62 | $78 | $75 |
| Medium | $86 | $71 | $79 | $99 | $96 |
| High | $115 | $92 | $102 | $128 | $122 |

*ERCOT lowest (Gulf Coast Class VI wells, abundant geology, cheap gas, shortest CO2 transport). NYISO highest (no suitable sequestration geology, longest transport, highest permitting burden).*

**CCS-CCGT cost buildup**:
- Capture cost: ~$30-40/MWh (technology-dependent, relatively uniform)
- CO2 transport: $2-20/MWh (regional — distance to Class VI well)
- CO2 storage: $5-15/MWh (regional — geology, well costs)
- Fuel cost: Heat rate × gas price (responds to gas toggle)
- 45Q offset: -$29/MWh ($85/ton × 0.34 tCO2/MWh)
- Capture rate: 95%
- Residual emissions: ~0.0185 tCO2/MWh (= 0.37 × 0.05)

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

### 7.1 CO2 Emissions Abated — Hourly Fossil-Fuel Emission Rates

**Methodology**: Build hourly variable emission rates from eGRID 2023 per-fuel emission factors × EIA hourly fossil fuel mix shares.

**Step 1 — Per-fuel emission rates** (from eGRID, static per region):
- `coal_rate[iso]` = eGRID coal CO₂ lb/MWh (e.g., ERCOT: 2325, PJM: 2216)
- `gas_rate[iso]` = eGRID gas CO₂ lb/MWh (e.g., ERCOT: 867, PJM: 867)
- `oil_rate[iso]` = eGRID oil CO₂ lb/MWh (e.g., ERCOT: 2894, PJM: 1919)

**Step 2 — Hourly fossil mix** (from EIA hourly data, 8760 values per ISO):
- `coal_share[h]`, `gas_share[h]`, `oil_share[h]` — fraction of fossil generation from each fuel at hour h

**Step 3 — Hourly emission rate**:
```
emission_rate[h] = coal_share[h] × coal_rate + gas_share[h] × gas_rate + oil_share[h] × oil_rate
```
This produces a variable hourly emission rate that reflects the actual fossil fuel mix dispatched at each hour (coal-heavy night hours vs. gas-peaker daytime hours, seasonal variation, etc.)

**Step 4 — Fuel price sensitivity** (shifts fossil mix):
- **Low gas price** → more gas dispatch, less coal → lower emission rate
- **High gas price** → coal resurgence (where coal capacity exists) → higher emission rate
- Regional fuel-switching elasticity from Section 5.9 applied as shift factors to coal/gas shares
- ERCOT: Low elasticity (coal mostly retired); PJM: High elasticity (45GW coal remaining)

**Step 5 — CO₂ abated** (hourly resolution, with storage dispatch attribution):
- For each hour h: `fossil_displaced[h] = clean_supply[h] − max(0, clean_supply[h] − demand[h])`
- `CO₂_abated = Σ_h fossil_displaced[h] × emission_rate[h]`
- CCS-CCGT gets **partial credit**: 95% capture → residual ~0.0185 tCO₂/MWh (vs ~0.37 unabated CCGT)

**Step 6 — Storage CO₂ attribution** (hourly dispatch tracking):
- Track exact hours each storage type (battery/LDES) dispatches into → use those hours' specific emission rates for abatement credit
- **Net against charge-side emissions**: when storage charges during hours with nonzero fossil on the margin, the charging energy carries an emissions cost. `net_storage_abatement = Σ_discharge_hours(dispatch[h] × emission_rate[h]) − Σ_charge_hours(charge[h] × emission_rate[h] / RTE)`
- This replaces the previous gap-weighted average approximation with exact hourly attribution
- Storage charging from pure surplus clean energy (no fossil on margin) → charge emissions = 0
- Storage charging during hours when fossil is still marginal → charge has real emissions that reduce net abatement

**Why this matters**: Flat emission rates overcount abatement during low-emission gas-dominant hours and undercount during high-emission coal-dominant hours. The hourly approach captures the actual carbon intensity of displaced generation. The storage charge-side netting prevents overcounting — a battery that charges from gas-marginal hours and discharges to gas-marginal hours provides less net abatement than one charging from clean surplus hours.

**Data sources**:
- `data/egrid_emission_rates.json` — 2023 eGRID per-fuel CO₂ rates (lb/MWh) by region
- `data/eia_fossil_mix.json` — EIA hourly fossil fuel mix shares (coal/gas/oil) by ISO

**Implementation note**: CO₂ calculation is post-hoc (doesn't affect cost/matching optimization). The optimizer's resource mix and cost results are unaffected. CO₂ values can be recomputed on cached results.

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
- **CCS-CCGT**: No existing share (new resource) → all new-build priced

---

## 11. Performance Optimizations

- **Sequential ISO processing**: ISOs run one at a time to avoid OOM; results saved incrementally after each ISO
- **Checkpointing**: Saves after each threshold (10 per ISO); resumes from checkpoint on restart — never loses more than one threshold's work
- **Caching**: Matching scores cached across 324 cost scenarios per threshold (physics reuse — cost-independent)
- **Cross-pollination**: After all 324 scenarios run per threshold, every unique mix re-evaluated against all scenarios
- **10 thresholds × 5 regions × 324 scenarios × 3 phases** — incremental saves essential for reliability

### 11.1 Adaptive Procurement Bounds

Compressed procurement sweep ranges per threshold to avoid wasting compute on levels that can't be optimal:

| Threshold | Min% | Max% | Rationale |
|-----------|------|------|-----------|
| 75% | 75 | 105 | Easy target, minimal overprocurement needed |
| 80% | 75 | 110 | Slight headroom |
| 85% | 80 | 110 | Still achievable without heavy overprocurement |
| 87.5% | 87 | 130 | Storage helps close gap, not raw procurement |
| 90% | 90 | 140 | Moderate overprocurement + storage |
| 92.5% | 92 | 150 | Entering inflection zone |
| 95% | 95 | 175 | Significant overprocurement may be needed |
| 97.5% | 100 | 200 | Heavy overprocurement territory |
| 99% | 100 | 220 | Near-perfect matching |
| 100% | 100 | 200 | If 2× procurement can't hit 100%, it's not cost-viable |

### 11.2 Edge Case Seed Mixes

17 forced seed mixes injected into Phase 1 coarse scan to guarantee extreme-but-potentially-optimal mixes survive pruning. Categories:

- **Solar-dominant** (70-75%): captures low-cost renewable scenarios where massive solar + storage is cheapest
- **Wind-dominant** (70-75%): captures ERCOT-like regions where wind + LDES dominates
- **Balanced renewable** (40/40 solar/wind): diversified variable generation
- **Clean firm dominant** (60-80%): captures low-cost nuclear/geothermal scenarios
- **Combined firm** (CF 30-40% + CCS 30-40%): dual-firm strategy
- **CCS-dominant** (50-60%): regions with favorable geology (ERCOT Gulf Coast)
- **High-hydro** (40% hydro): NYISO, CAISO, NEISO where existing hydro fleet is large

Seeds filtered at runtime by regional hydro cap. Adds ~10-15 combos to the ~280 coarse grid combos per region — negligible compute cost, significant coverage improvement.

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
- **File**: `dashboard/region_deepdive.html` (single combined page)
- **Navigation**: Top nav "Regional Deep Dives" links here; region selector within the page
- **Format**: Scrollytelling narrative matching main dashboard visual identity
- **Content**: In-depth exploration of what deep decarbonization looks like for each region under different sensitivity scenarios

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

**File**: `dashboard/abatement_comparison.html`

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
| Abatement Dashboard (abatement_dashboard.html) | Carbon Abatement Dashboard | Interactive Abatement Cost & Portfolio Analysis |
| CO₂ Abatement Summary (abatement_comparison.html) | CO₂ Abatement Summary | Comparing Grid Decarbonization to Alternative Pathways |
| CAISO Deep Dive | CAISO Deep Dive | California's Path to 24/7 Clean Energy |
| ERCOT Deep Dive | ERCOT Deep Dive | Texas Grid Decarbonization Analysis |
| PJM Deep Dive | PJM Deep Dive | Mid-Atlantic Clean Energy Transition |
| NYISO Deep Dive | NYISO Deep Dive | New York's Hourly Matching Challenge |
| NEISO Deep Dive | NEISO Deep Dive | New England's Decarbonization Pathway |
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

## 18. Summary Counts

| Item | Count |
|---|---|
| Resources | 7 |
| Thresholds | 10 (reduced from 18) |
| Regions | 5 |
| Dashboard controls | 7 (2 existing + 5 paired toggles) |
| Paired toggle groups | 5 (from 10 individual toggles) |
| Cost scenarios per region/threshold | 324 (3×3×3×3×4) — each independently co-optimized |
| New chart types | 2 (average + marginal abatement curves) |
| New panels | 1 ("What You Need" panel) |
| New metric tiles | 1 (CO2 abated) |
| Cost tables in methodology | ~15 |
| Sensitivity toggles | 10 (all Low/Medium/High except Transmission which adds None) |
| Regional deep-dive pages | 1 (combined, with region selector) |
| Research paper sections | 8 (including 5 regional deep-dives) |
| QA checkpoints | 3 (optimizer, HTML, mobile) |

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

**Implementation in `load_data()`**:
- `_remove_leap_day(profile)`: Excises Feb 29 from 8784→8760
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
