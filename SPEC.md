# Advanced Sensitivity Model — Complete Specification

> **Authoritative reference for all design decisions.** If a future session needs context, read this file first.
> Last updated: 2026-03-02.

## Current Status (Mar 3, 2026)

### Offshore Wind Integration — Steps 4–7 + Dashboard (In Progress)

**Branch:** `claude/integrate-offshore-wind-SkQIr`

**Scope**: Thread offshore wind through Steps 4–7, update all dashboard pages, integrate resource caps (geothermal, CCS, offshore wind) into Scenario A/B, Procurement 1-3, Track 2 NB, Track 3 CTR. Simultaneously updating resource colors (nuclear → #6366F1, CCS → #64748B) and display order.

**Status:**
- [ ] Phase 1: Core infrastructure (dispatch_utils.py + scenario_common.py)
- [ ] Phase 2: Step 4 (gas/CCS adjustments)
- [ ] Phase 3: Step 5 (dispatch cache)
- [ ] Phase 4: Step 6 scripts (10+ files)
- [ ] Phase 5: Step 7 (shared data generation)
- [ ] Phase 6: Dashboard (JS + HTML + CSS, 12+ files)
- [ ] Phase 7: GitHub Actions workflows

---

## Previous Status (Mar 2, 2026)

### Step 1c/1d Workflow Fixes (Mar 2, 2026 — COMPLETED)

**Branch:** `claude/fix-workflow-batching-c1EFG`

**Problems Fixed:**
1. **Zone search getting stuck**: Removed problematic `step1_prior_windows.py` call from workflow that tried to compute from non-existent EF results. Scripts now gracefully handle missing prior windows.
2. **No threshold batching**: Added `--thresholds` parameter to both `step1c_zone_search.py` and `step1d_fine_storage.py` to allow running individual thresholds or custom subsets (e.g., `--thresholds "95,99"`).
3. **No per-threshold commits**: Added `git_commit_threshold_single()` function to step1c and updated both scripts to commit after each threshold completes (with retry logic).
4. **Near-miss parquets**: Verified step1c creates `{ISO}_near_miss.parquet` (union of all near-miss mixes across all thresholds) for step1d to consume.
5. **Graceful error handling**: Prior windows loading now catches exceptions and falls back to coarse-derived bounds.

**Changes:**
- `step1c_zone_search.py`: Added `--thresholds` arg, per-threshold commits, graceful prior windows loading
- `step1d_fine_storage.py`: Added `--thresholds` arg, filtering to requested thresholds only
- `.github/workflows/step1c-zone-search.yml`: Removed prior windows step, added `thresholds` input, updated script call
- `.github/workflows/step1d-fine-storage-v2.yml`: Added `thresholds` input, updated script call

**Result:** Both 1c and 1d now support per-threshold execution with auto-commits, preventing work loss on timeout.

---

### Streamlined PFS Architecture Redesign (Mar 2, 2026)

**Branch:** `claude/streamline-pfs-architecture-0RMKJ`

**Problem:** Current Step 1c/1d runs `optimize_threshold()` independently per threshold. A mix scoring 63% base gets storage-swept for t50, t55, t60, t65 independently — 4× redundant scoring. Fine 1% refinement around boundary archetypes duplicates across adjacent thresholds. Estimated 5-8× compute waste from per-threshold duplication.

**Architecture Decision: "Score Once, Bin by Threshold"**

A mix's hourly match score is a fixed physics property — it doesn't change per threshold. Score every unique (mix) and (mix, storage_config) tuple exactly once, then assign results to thresholds by their score.

**5-Script Pipeline (replacing old 1a/1b/1c/1d):**

| Script | Status | Purpose |
|--------|--------|---------|
| `step1_prior_windows.py` | NEW | Reads prior EF parquets → search window JSON |
| `step1a_generate_mixes.py` | MODIFIED | Prior-informed bounds + 100 scout mixes |
| `step1b_score_mixes.py` | MINIMAL CHANGES | Add caching layer |
| `step1c_zone_search.py` | NEW (replaces old 1c) | Score-band zone fine search with global dedup |
| `step1d_fine_storage.py` | NEW (replaces old 1d) | Two-pass: coarse global → fine targeted 0.05% |

**Decision 1: Prior-Informed Search Windows (Hard Windows + Scouts)**
- Load prior EF parquets, compute per-resource [min, max] per threshold
- Add 15pp absolute buffer: [max(min-15, 0), min(max+15, cap)]
- Union across thresholds → outer bounds for coarse grid
- 100 scout mixes (50 random outside window + 50 corner combos) to catch regime shifts
- If any scout scores near a boundary, dynamically expand the window
- Saves ~30% of coarse grid compute vs. full Cartesian

**Decision 2: Score-Band Zones (Fine Search Grouping)**
- 3 overlapping zones replace 15+ per-threshold fine searches:
  - Zone A: score [0.45, 0.78] → covers t50–t75
  - Zone B: score [0.73, 0.93] → covers t75–t90
  - Zone C: score [0.88, 1.00] → covers t90–t99.99
- Fine 1% grid generated per zone with zone-specific resource windows
- Global hash set prevents any mix from being scored twice across zones
- Each scored mix assigned to ALL thresholds where feasible or near-miss
- ~4× reduction in fine scoring operations

**Decision 3: Two-Pass Storage (Coarse Global → Fine Targeted)**
- Pass 1: Collect UNION of all near-miss mixes across ALL thresholds (unique set). Sweep storage at coarse resolution (bat4 0-6%, bat8 0-8%, LDES 0-25%, H2 0-25%). Score each (mix, storage) combo ONCE. Bin results to thresholds by score. ~8× reduction vs. per-threshold storage sweep.
- Pass 2: For each threshold, identify boundary mixes (score within [T-2pp, T+1pp] after Pass 1). Refine winning storage dims at 0.05% resolution ±0.5pp around Pass 1 winner (~21 values/dim). Importance-weighted subset if cross-product >1000. Gets 0.05% storage accuracy on frontier mixes.

**Decision 4: Full Rewrite of 1c/1d**
- Keep 1a (mix generation) and 1b (scoring kernel) largely intact
- Numba dispatch kernels preserved — proven, fast
- Old 1c/1d preserved as `step1c_build_pfs_legacy.py` / `step1d_storage_refinement_legacy.py`
- New `step1c_zone_search.py` + `step1d_fine_storage.py`

**Accuracy Targets:**
- Resource mix: 1% (fine grid step)
- Storage: 0.05% on frontier boundary mixes, 1% elsewhere

**Estimated Compute Savings:** ~5× total reduction in scoring operations. Same or better PFS quality due to finer storage resolution on frontier.

---

## Previous Status (Feb 28, 2026)

### MAC Formula: Pure LCOE / CO₂ Displaced — No Wholesale Offset (Mar 1, 2026)

**Decision confirmed:** MAC = pure deployment LCOE of new clean resources / CO₂ displaced by those resources. No wholesale electricity price, fuel cost, or system cost plays any role in the MAC numerator.

**MAC formula:**
```
MAC = (new_resource_lcoe_cost × annual_demand_mwh) / CO2_abated_by_new_capital
```

Where:
- **Cost numerator**: `cost_total_cost − gas_backup_cost` — the LCOE of NEW-BUILD clean resources only (solar, wind, clean firm, CCS, storage + transmission). Step 3 already prices existing resources at $0 (sunk fleet), so `cost_total_cost` contains only new-build costs. Gas backup (resource adequacy) is subtracted because it's system reliability, not abatement investment.
- **CO₂ denominator**: Baseline fossil emissions (existing clean only, 2025 demand) minus scenario fossil emissions (at threshold level). Uses merit-order dispatch model (coal → oil → gas retirement). Only counts CO₂ displaced by NEW capital.

**Critical: NO wholesale offset.** The prior code subtracted `existing_pct × wholesale_price` from `cost_total_cost` before computing MAC. This was a phantom double-subtraction — Step 3 already excludes existing resources from cost, so subtracting wholesale for them a second time drove MAC to $0 in wind-heavy ISOs (SPP, ERCOT).

**Sanity check (floor MAC, Medium sensitivity, no TX):**
- SPP wind: $37/MWh ÷ 1.021 tCO₂/MWh (coal) = **$36/tCO₂** — absolute floor
- SPP wind: $37/MWh ÷ 0.392 tCO₂/MWh (gas) = **$94/tCO₂** — once coal retired
- ERCOT wind: $40/MWh ÷ 1.055 tCO₂/MWh (coal) = **$38/tCO₂** — absolute floor
- ERCOT wind: $40/MWh ÷ 0.393 tCO₂/MWh (gas) = **$102/tCO₂** — once coal retired
- Values below ~$27/tCO₂ at any threshold are physically impossible.

**Bug fixed (Mar 1, 2026):** Removed `existing_pct × wholesale` subtraction from `add_mac_column()` and `compute_dg_mac()` in `step5_compute_mac_stats.py`. Also removed wholesale-related imports/constants that are no longer needed by MAC calculation.

---

### Scenario A Forward-Stepping Rewrite (Feb 28, 2026)

**Branch:** `claude/fix-scenario-a-resources-kDR92`

**Problem:** Old Scenario A forward-stepping evaluated ALL feasible EF mixes at each threshold, then penalized under-floor mixes with excess cost. This was conceptually messy — it priced solutions that should have been filtered out.

**Verification finding:** Scenario A's 50% starting point is NOT equivalent to Step 3's `LHLH_M_M_H1_X` key because (1) Step 6 applies demand growth (843→950 TWh for PJM at 2030), (2) uprate override ($25 vs $40/MWh), (3) different feasible mix pools (shared-data.js subset vs full Step 2 EF).

**New algorithm (filter-first with PFS fallback):**
1. At each threshold, convert prior-step deployed TWh into per-resource floor percentages (floor_twh / demand_twh at target year).
2. **Filter** EF mixes to only those meeting ALL per-resource floors (can't un-build deployed assets).
3. Price filtered mixes under scenario cost assumptions. Pick cheapest total_cost.
4. If floor eliminates all EF mixes → progressive PFS fallback:
   a. PFS within floor to floor+10% per resource (narrow window)
   b. PFS within floor to floor+250% per resource (wide window, 5% grid implicit)
   c. PFS floor-as-minimum-only (no upper bound)
   d. All PFS with under-floor cost carry-forward (absolute last resort)
5. If a PFS mix goes under on a resource to hit the threshold, carry that resource's floor cost forward (priced at newbuild LCOE).
6. Floor ratchets: `floor = max(prior_floor, deployed)` per resource. Per-resource, not aggregate — can't un-build.

**Key design decisions:**
- **Per-resource floor** (not aggregate): Each of solar, wind, clean_firm, CCS, hydro, battery, LDES individually >= prior deployed TWh. Most physically realistic — deployed panels/turbines/plants don't disappear.
- **Cheapest total_cost** (not cheapest $/MWh-CFE): Since EF mixes at threshold T already achieve >= T%, we pick the cheapest total cost, not normalized by match score. This naturally selects barely-above-threshold mixes at each step — correct for sequential procurement modeling.
- **Demand growth applied**: Floor TWh is absolute (fixed MW deployed). As demand grows, floor as a percentage of demand shrinks slightly — new solutions must still deploy at least as much absolute capacity.

**Files changed:** `scripts/step6_scenario_comparison.py` — rewrote `_forward_step_optimization()`, added `_load_pfs_mixes()`, `_filter_mixes_by_floor()`, `_filter_pfs_by_floor_window()`.

### Consequential Queue: MAC Formula + Threshold Fix (Feb 28, 2026)

**Changes:**

1. **MAC formula = newbuild LCOE / displaced emission rate** (replaces delta_cost/delta_co2).
   - Buyers using consequential accounting optimize on this metric: cheapest technology cost per tCO2 of their claim.
   - This is deliberately the *narrow buyer's metric* — it reflects what drives procurement decisions in practice.
   - The whole point of the analysis is to show that optimizing on this metric in isolation (without contextualizing system costs, gas backup needs, asset stranding, learning curves) yields adverse outcomes.
   - System costs (gas backup capacity, stranded assets, foregone learning) are tracked separately as comparative layers. They're real costs but they're NOT what drives the buyer's purchasing decision under consequential accounting.

2. **Thresholds filtered to >= 50%** (removed 10-40% from all consequential queue functions). Below 50% is pre-SBTi baseline — not relevant to the deployment queue analysis.

**Files changed:** `scripts/step5_consequential_deployment_queue.py`, `scripts/step6_scenario_comparison.py` — MAC formula in `compute_zone_metrics()` and queue builder.

---

### Corporate Procurement Strategy Simulation — Compute Architecture Phase

**Branch:** `claude/procurement-strategy-page-lmOiJ`

**Completed (Research & Design — Feb 27):**
- [x] Research: C&I load share by ISO (EIA data — national ~62%, ranges 52-67% by ISO)
- [x] Research: Corporate voluntary procurement market state (NREL 2024: ~315 TWh, ~13% C&I penetration)
- [x] Research: RPS/compliance/nuclear programs — clean energy already committed per ISO
- [x] Research: Grid avg vs fossil avg vs marginal emission rates per ISO (eGRID + VERACI-T)
- [x] Research: EAC scarcity by ISO (20x variation: ERCOT 130-160 TWh available vs NEISO 3-8 TWh)
- [x] Created `dashboard/procurement_research.html` — research page documenting findings
- [x] Design decisions captured (see §15 below)
- [x] Strategy 1C (marginal emission baseline): Include — material in MISO (+17%) and SPP (+22%)
- [x] Learning curve toggle: On/Off, mapped to Scenario A (Strategy 1/3) and Scenario B (Strategy 2) — see §15.10
- [x] Supply constraints: Show explicitly with infeasibility bands — see §15.12
- [x] Adverse effects of delayed hourly matching: 3 compounding effects documented — see §15.11
- [x] Participation slider defaults: Hyperscaler 5-6%, Other 7-8% — see §15.13
- [x] Cost-to-replace premium (Strategy 2C): Use existing Track 3 CTR values directly

**Completed (Compute Architecture — Feb 28):**
- [x] Step 6.5 compute architecture decisions — all 7 decision cards resolved (see §15.14)
- [x] Card 1 (Script structure): 1A — one script per strategy family + shared utils
- [x] Card 2 (Existing clean pricing): Dual toggle — 45U + CTR NOAK-based premiums
- [x] Card 3 (SSS baseline): 2B gets grid 8760 shape free; 2C gets SSS 8760 + existing-minus-SSS at premium + new-build
- [x] Card 4 (Participation model): 4B — independent annual + hourly translation + scarcity
- [x] Card 5 (LMP feedback): 5C — full 8760-hour LMP for all 7 ISOs
- [x] Card 6 (SBTi mapping): 6D — SBTi default + manual override
- [x] Card 7 (Output format): 7B — standalone procurement-strategy-data.js

- [x] PPA pricing model: Percentage premium (LCOE × (1 + pct)) — VRE 5/12/22%, Firm 12/22/38%, Uprate 10/20/35%

**Completed (LMP Infrastructure — Feb 28):**
- [x] Extend LMP model to all 7 ISOs: MISO + SPP price models added to `step5_compute_lmp_prices.py`
- [x] Calibration targets for all 7 ISOs in `calibrate_lmp_model.py` (2024 SOM data)
- [x] Extend `step0_fetch_lmp_2025.py` to support `--year 2024` and MISO/SPP
- [x] GitHub Actions workflows: `fetch-actual-lmp.yml` + `run-lmp-calibration.yml`

**Completed (Step 6.5 Compute Scripts — Feb 28):**
- [x] `step5_5_procurement_utils.py` — Shared utilities (SSS allocation, EAC pricing, LMP feedback, PPA premiums, learning curve, 25yr timeline)
- [x] `step5_5_strategy1_consequential.py` — Strategies 1A/1B/1C (cross-regional consequential netting)
- [x] `step5_5_strategy2_hourly.py` — Strategies 2A/2B/2C (hourly matching same-ISO)
- [x] `step5_5_strategy3_annual.py` — Strategies 3A/3B/3C/3D (annual matching 2×2 matrix)
- [x] GitHub Actions workflow: `run-procurement-strategies.yml` (quick/full mode, per-strategy or ALL)

**Next steps:**
- Run procurement strategy workflows via GitHub Actions to generate data
- Run LMP calibration workflow for all 7 ISOs
- Build interactive procurement comparison dashboard page (`procurement_comparison.html`)
- Wire up `procurement-strategy-data.js` to dashboard charts

---

## §15: Corporate Procurement Strategy Simulation

### §15.1 Overview

Extension of the optimizer to model how different GHG accounting policies and procurement strategies affect clean energy deployment, costs, and emissions at varying levels of corporate participation. Builds on existing hourly matching (Track 2 NB), cost-to-replace (Track 3 CTR), and consequential accounting (Scenario A/B) frameworks.

### §15.2 Strategy Taxonomy

**Strategy 1 — Consequential Cross-Regional Netting**
Buyers purchase cheapest $/tCO₂ clean energy anywhere in the US to "net" against location-based carbon emissions. Requires new build or nuclear uprates. No ISO boundary constraint.

| Variant | Emission Baseline | Description |
|---------|------------------|-------------|
| **1A** | Grid-average | Buyer's ISO grid-average emission rate (includes clean in denominator). Lowest bar. |
| **1B** | Fossil-average | Buyer's ISO fossil-only fleet average. Higher bar. |
| **1C** | Marginal emissions | Short-run marginal emission rate. Highest bar in coal-heavy ISOs (MISO +17%, SPP +22% vs fossil avg). Negligible difference in gas-dominated ISOs. |

**Strategy 2 — Hourly Matching (Same-ISO)**
Buyer matches load hour-by-hour within their own ISO. No cross-regional procurement. Highest verifiability, highest cost.

| Variant | Existing Clean Credit | Description |
|---------|----------------------|-------------|
| **2A** | None | 100% new build. Maximum additionality. Equivalent to existing Track 2 NB analysis. |
| **2B** | Grid baseline | Buyer takes credit for existing clean grid mix as hourly baseline, procures new build on top. Reduces cost in clean-grid ISOs. |
| **2C** | Pro-rata allocation + premium | Pro-rata share of RPS/nuclear/public utility clean allocated. Premium for existing clean to keep it online (cost-to-replace). New build on top. |

**Strategy 3 — Annual Matching**
Volumetric annual matching without hourly temporal constraint. 2×2 matrix: {Same-ISO, Cross-Regional} × {Additionality Required, No Additionality}.

| Variant | Boundary | Additionality | Description |
|---------|----------|---------------|-------------|
| **3A** | Same-ISO | New build required | Annual matching within buyer's ISO. Only new-build clean energy counts. Comparable to Strategy 2A but annual. |
| **3B** | Cross-regional | New build required | Annual matching from any US ISO. Only new-build clean counts. Comparable to Strategy 1 but annual volumetric rather than consequential netting. |
| **3C** | Same-ISO | No additionality | Annual matching within buyer's ISO. Existing clean counts (includes unbundled RECs from existing generators). |
| **3D** | Cross-regional | No additionality | Annual matching from any US ISO. Existing clean counts. Cheapest option — unbundled RECs from anywhere. This is the "status quo" for most corporate procurement today. |

**Cross-cutting toggle: FOAK-to-NOAK Learning Curves (On/Off)**
Toggle (not suffix) applicable to all strategies simultaneously. When On, each strategy's cost curve shifts based on cumulative clean firm deployment along its mapped trajectory (see §15.10). When Off, static Medium costs from existing optimizer.

### §15.10 Learning Curve Integration (Decided Feb 27)

**Toggle:** "Learning curve: On/Off" in interactive section. Applies to all strategies simultaneously.

**Strategy → Trajectory Mapping:**

| Strategy | Trajectory | Rationale |
|----------|------------|-----------|
| **Strategy 1** (Consequential) | **Scenario A** (delayed) | Chases cheap $/tCO₂ with VRE → no clean firm investment → FOAK when firm is finally needed. Learning period 2035-2047, never fully reaches NOAK. |
| **Strategy 2** (Hourly) | **Scenario B** (accelerated) | Hourly matching *forces* early clean firm + storage investment → accelerates Wright's Law learning → NOAK by 2040. Learning period 2030-2040. |
| **Strategy 3** (Annual) | **Scenario A** (delayed) | Annual flexibility lets buyers avoid firm clean (VRE + unbundled RECs satisfy annual targets) → same delayed investment dynamic as consequential. |

**SBTi Milestone Mapping:** (existing constants from `step6_generate_shared_data.py`)
- 2025: Today (0%) | 2030: SBTi 50% | 2035: SBTi ~70% | 2040: SBTi 90% | 2045: SBTi ~95% | 2050: Net-Zero (≥99.99%)

**Core argument:** Hourly matching incentivizes earlier corporate investment in clean firm, accelerating the learning curve, making the entire system cheaper on a net-zero trajectory. It is significantly more expensive to reach net zero by 2050 if you delay investment in firm clean. The three compounding adverse effects of delay are documented in §15.11.

**Implementation:** Uses existing `learning_fraction()` from `step6_scenario_comparison.py` (Scenario A: FOAK until 2035, learning 2035-2047; Scenario B: FOAK until 2030, learning 2030-2040, NOAK by 2040). Cost at each SBTi milestone = FOAK × (1 - learning_fraction) + NOAK × learning_fraction for clean firm resources.

### §15.11 Adverse Effects of Delayed Hourly Matching (Decided Feb 27)

Three compounding effects when strategies don't require hourly deliverable matching:

**1. Learning Curve Delay (§15.10)**
Consequential/annual strategies defer firm clean investment. When 90%+ targets require firm clean (2040 SBTi milestone), buyers using Strategy 1/3 face near-FOAK prices. Strategy 2 buyers have already driven costs to NOAK via early deployment.

**2. Stranded VRE Overbuild**
Cheap $/tCO₂ logic under annual/consequential accounting drives massive solar/wind procurement at low thresholds (50-70%). But at high thresholds (90%+), additional VRE has sharply diminishing returns — surplus solar during peak hours is already being curtailed. The VRE built at 60% to satisfy annual accounting doesn't deliver physical electrons during nighttime/low-wind hours when the grid actually needs them. This capacity may not be useful by the time you need a deeply decarbonized grid. Hourly matching forces buyers to confront the residual gap early → invests in resources that actually close it.

**3. Gas Lock-in from Missing Storage Signal**
Without hourly matching, there is no price signal to invest in storage (battery + LDES) for nighttime/low-wind hours. Gas fills that gap by default. Once gas capacity is built or retained, it creates political and economic inertia (stranded asset risk, workforce dependencies, pipeline contracts) to keep running it. Hourly matching creates direct demand for storage to cover every hour → displaces gas earlier → shorter gas plant lifetimes → less stranded fossil infrastructure. The longer gas is held, the more expensive the eventual retirement (stranded asset write-downs, decommissioning, workforce transition).

**Compounding on SBTi Timeline:**

| SBTi Milestone | Strategy 1/3 (Annual/Consequential) | Strategy 2 (Hourly) |
|---|---|---|
| 2030 (50%) | Cheap — lots of VRE, looks great on paper | Slightly more expensive — investing in firm + storage |
| 2035 (70%) | Still cheap — more VRE, gas fills gaps | Firm clean hitting learning curve, storage displacing gas |
| 2040 (90%) | **Wall** — VRE saturated, firm at FOAK, gas locked in | Firm at NOAK, storage mature, gas already retiring |
| 2050 (≥99.99%) | Scramble — paying FOAK for firm, retiring gas at huge cost, stranded VRE | Smooth glide — infrastructure already in place |

These effects should be modeled explicitly in the dashboard and presented as a key finding in the scrollytell narrative and research paper.

### §15.12 Supply Constraint Handling (Decided Feb 27)

**Approach:** Show constraints explicitly. When a strategy hits a physical supply ceiling in an ISO, display it as "infeasible above X% participation" with a red/hatched band on the chart.

**Constraint sources:**
- EAC scarcity (§15.7): NEISO has only ~3-8 TWh available for voluntary procurement. Same-ISO strategies (2, 3A, 3C) hit hard walls.
- Cross-regional strategies (1, 3B, 3D) can route around ISO-level constraints by procuring from surplus ISOs (ERCOT, SPP).
- Resource adequacy: High participation rates under hourly matching may exceed buildable capacity in constrained ISOs.

**Key finding (to highlight):** The existence of supply constraints is itself a major result — it demonstrates *why* cross-regional accounting matters and where same-ISO hourly matching faces physical limits. This should be a prominent element of the scrollytell narrative.

### §15.13 Participation Slider Defaults (Decided Feb 27)

- **Hyperscaler participation:** Default 5-6% of C&I load (current market share). Range 0-15%.
- **Other corporate participation:** Default 7-8% of C&I load (current mid-market). Range 0-40%.
- **Data center electricity:** ~130-150 TWh (2024), growing ~15-20%/yr. Data center share of C&I: ~5-6% today, projected ~8-10% by 2028.
- **Total current corporate procurement:** ~315 TWh (~13% of C&I), with 84% from tech/hyperscaler buyers.

### §15.14 Step 6.5 Procurement Strategy Compute Architecture (Decided Feb 28)

**Purpose:** Compute pipeline to model all 10 procurement strategy variants at varying participation levels, producing data for the procurement comparison dashboard page (§15.5).

#### §15.14.1 Script Architecture (Card 1 — Selected: 1A)

One script per strategy family + shared utility module:
- `scripts/step5_5_strategy1_consequential.py` — Strategies 1A, 1B, 1C
- `scripts/step5_5_strategy2_hourly.py` — Strategies 2A, 2B, 2C
- `scripts/step5_5_strategy3_annual.py` — Strategies 3A, 3B, 3C, 3D
- `scripts/step5_5_procurement_utils.py` — Shared utilities (SSS allocation, EAC pricing, LMP feedback, participation scaling)

Each script is independently runnable. Shared utils handle cross-cutting logic.

#### §15.14.2 Existing Clean Pricing — Dual Toggle (Card 2 — Decided)

Two independent premium mechanisms, each with its own dashboard toggle:

**Toggle (i): 45U-Based Clean Premium**
- Applies to **existing nuclear** in ISOs with 45U PTC eligibility
- Price = 45U credit value ($15/MWh, inflation-adjusted) + small margin (5%)
- Rationale: 45U provides a known revenue floor for existing nuclear. The premium represents what a buyer pays to "claim" the clean attribute of existing nuclear generation beyond what 45U already covers.
- L/M/H sensitivity on the margin above 45U

**Toggle (ii): CTR (Cost-to-Replace) NOAK-Based Premium**
- Uses existing Track 3 CTR values directly from `track_results.json`
- Premium = delta between CTR effective cost and ECF effective cost at each threshold
- Represents what it would cost to replace existing dispatchable clean if it retired
- NOAK-adjusted: applies learning curve discount to replacement cost (reflects that under Strategy 2C with sufficient participation, replacement would be at NOAK, not FOAK)
- L/M/H maps to FOAK/Mid-Learning/NOAK replacement cost

Both toggles can be on simultaneously (additive). Default: Toggle (i) On, Toggle (ii) On at Medium.

#### §15.14.3 Existing Clean Baseline Allocation (Card 3 — Decided Feb 28)

**Strategy 2B — Grid Baseline (Simple):**
- Buyer takes credit for **existing clean grid at 8760 hourly shape** as baseline
- Like Track 1 baseline — existing clean generation follows its actual hourly profile (nuclear flat 24/7, solar daytime, wind stochastic, hydro seasonal)
- Buyer procures **above** this baseline to reach target CFE threshold
- No premium paid — free-rides on existing clean being online
- Cheapest hourly variant but creates stranding risk (§15.5.2)

**Strategy 2C — SSS Allocation + Premium Tranches:**
- **Layer 1: SSS allocation** — Buyer receives pro-rata share of SSS (state-sponsored/contracted) clean within their ISO, following **8760 shape of SSS generation** (nuclear baseload, solar daytime, hydro seasonal — shaped by actual SSS resource mix per ISO)
- **Layer 2: Existing clean beyond SSS** — Remaining existing clean (total existing minus SSS) follows its **8760 hourly shape**. Buyer can procure from this pool at premium prices (see §15.14.4 below)
- **Layer 3: New-build** — Any remaining gap above existing is filled with new-build procurement at LCOE (or LCOE + PPA premium — see §15.14.4)
- Buyer procures **above SSS** allocation to reach target, drawing from Layer 2 (existing non-SSS at premium) then Layer 3 (new-build)

**Key distinction:** 2B gets existing clean for free (no payment signal). 2C pays for existing clean via explicit premiums, creating a revenue signal to prevent retirement.

#### §15.14.4 Procurement Cost Tranches for Strategy 2C (Decided Feb 28)

Tranche merit-order for 2C procurement above SSS:

| Tranche | Source | Price | Status |
|---------|--------|-------|--------|
| 1 | Existing nuclear (non-SSS) | 45U + 5% margin, or CTR delta | **Decided** (§15.14.2) |
| 2 | Nuclear uprates | Uprate LCOE × (1 + premium_pct) | **Decided** (see PPA pricing below) |
| 3 | Existing hydro/solar/wind (non-SSS) | EAC market proxy ($3-5/MWh) | **Decided** (from §Decision 5e) |
| 4 | New-build VRE (solar, wind) | LCOE × (1 + VRE_PPA_premium) | **Decided** (see PPA pricing below) |
| 5 | New-build clean firm (nuclear, CCS, geothermal) | LCOE × (1 + Firm_PPA_premium) | **Decided** (see PPA pricing below) |

**PPA Premium Model (Decided Feb 28):**

PPA prices are set by developer financial models (capital recovery + equity return + risk). The PPA-to-LCOE gap reflects the difference between LCOE's assumed WACC (6-8% real) and actual project financing costs (10-12% nominal equity + 5-7% debt), plus transaction costs. Empirically (LBNL PPA tracking), wind/solar PPAs run 10-25% above NREL ATB LCOE. The percentage model is used because developer returns scale with capital deployed.

`PPA_price = LCOE × (1 + premium_pct)`

| Resource Category | L Premium | M Premium | H Premium | Rationale |
|-------------------|----------|-----------|-----------|-----------|
| **VRE (solar, wind)** | +5% | +12% | +22% | Commodity market, many competing developers, lower risk |
| **Clean firm (nuclear, CCS, geothermal)** | +12% | +22% | +38% | Fewer projects, higher development risk, longer timelines |
| **Nuclear uprates** | +10% | +20% | +35% | Limited supply (~4.4 GW nationally), bilateral negotiation |

**L/M/H mapping**: Low = competitive market, ample supply, multiple bidders. Medium = balanced market, typical bilateral dynamics. High = constrained supply, limited developers, high demand for EACs.

**Example at Medium costs, PJM:**
- Solar LCOE $32 → PPA $36/MWh (+$4)
- Wind LCOE $38 → PPA $43/MWh (+$5)
- Uprate LCOE $25 → PPA $30/MWh (+$5)
- Nuclear new-build LCOE $105 → PPA $128/MWh (+$23)
- CCS LCOE $79 (45Q on) → PPA $96/MWh (+$17)

#### §15.14.5 Participation-to-CFE Target Mapping (Card 4 — Selected: 4B)

Independent annual demand-share model + hourly translation + scarcity feedback:

1. **Annual demand share:** At X% participation, buyer's annual demand = X% × ISO C&I load (TWh)
2. **Hourly translation:** Apply 8760-hour load shape to get hourly demand profile. Each hour, buyer needs (hourly_demand × CFE_target%) matched by clean generation.
3. **EAC scarcity model:** As cumulative demand for EACs increases (more participation), available supply tightens → price escalation. Uses existing `step5_compute_eac_scarcity.py` supply curves.

No "market clearing" between strategies — each strategy computed independently at each participation level.

#### §15.14.6 LMP Wholesale Price Feedback (Card 5 — Selected: 5C)

Full 8760-hour LMP model for **all 7 ISOs**. All 7 ISOs now have calibrated price model classes.

**Implementation status (Feb 28):**
1. PJM — fully calibrated (v10, PJMPriceModel), target $34.7/MWh ✓
2. ERCOT — ERCOTPriceModel, ORDC VOLL×LOLP $5K cap, target $26/MWh ✓
3. CAISO — CAISOPriceModel, RA + duck curve -$60 floor, target $38/MWh ✓
4. NYISO — NYISOPriceModel, ICAP + tight geography, target $42/MWh ✓
5. NEISO — NEISOPriceModel, FCM + winter gas $13.13/MWh, target $39.5/MWh ✓
6. MISO — MISOPriceModel, PRA, $3,500 VOLL, coal 35%, target $31/MWh ✓
7. SPP — SPPPriceModel, limited capacity market, wind 37%, target $26/MWh ✓

**Calibration data:** `calibrate_lmp_model.py` has `ISO_CALIBRATION_TARGETS` with 2024 SOM data for all ISOs. Sources: PJM IMM, Potomac Economics (MISO/NYISO), SPP MMU, ISO-NE EMM, CAISO DMM, ERCOT/Modo Energy.

**Actual LMP fetching:** `step0_fetch_lmp_2025.py` extended to support `--year 2024` and all 7 ISOs (MISO + SPP added via gridstatus). GitHub Actions workflow: `fetch-actual-lmp.yml`.

**Wholesale price degradation analysis (Decided Feb 28):**
- Run LMP model for all 7 ISOs at all 21 thresholds to produce price degradation curves
- Key output: avg LMP vs clean energy threshold (50%→99.99%) showing merit-order price depression
- Correlation with clean penetration demonstrates cannibalization effect
- Data feeds into both the wholesale price dashboard page AND the procurement strategy comparison
- Price degradation directly affects procurement cost: as more buyers adopt hourly matching, wholesale prices fall, making EACs relatively more expensive (the "stranding" effect documented in §15.11)

Each ISO requires calibration against actual LMP data. ISO-specific price formation rules from §Decision 6 (LMP Module) apply.

#### §15.14.7 SBTi Timeline + 25-Year Demand Growth (Card 6 — Selected: 6D, Extended Feb 28)

Default to SBTi milestone mapping (2030→50%, 2035→70%, 2040→90%, 2050→≥99.99%) with manual override slider for custom targets.

Uses existing constants from `step6_generate_shared_data.py` SBTI_MILESTONES.

**25-Year Demand Growth Dimension (Decided Feb 28):**
The procurement strategy page is built on the **25-year demand growth trajectory / SBTi timeline** already computed by the optimizer:
- Existing demand growth sweep: 25 years × 3 growth rates (L/M/H) per ISO (from Step 3)
- Each year maps to an SBTi milestone CFE target → determines how much clean procurement is needed
- Procurement cost trajectory: for each strategy, compute annual cost over 25 years as the CFE target ratchets up along the SBTi curve
- Strategy comparison becomes: "what is the total cost of getting from today's procurement to 99.99% by 2050, under each strategy?"
- Demand growth interacts with EAC scarcity (higher demand = more competition for same EAC supply = higher prices)
- Learning curve effects compound over the timeline (early adoption at FOAK → late adoption at NOAK)
- Key visualization: cumulative cost envelope (25 years × 10 strategies × L/M/H demand growth) showing when each strategy becomes optimal

#### §15.14.8 Output Format (Card 7 — Selected: 7B)

Standalone JS data file: `dashboard/js/procurement-strategy-data.js`
- Loaded only by `procurement_comparison.html`
- Contains all strategy comparison data (costs, resource mixes, CO₂, MAC, participation curves)
- Generated by a new step in the pipeline (after Step 6, before Step 7)
- Does NOT bloat shared-data.js

---

### §15.3 Participation Model (Decided Feb 27)

**Two national sliders:**
1. **Hyperscaler participation** (% of C&I load from top ~30 hyperscaler/tech buyers)
2. **All other corporate participation** (% of remaining C&I load from mid-market, Fortune 500 non-tech, etc.)

Both are national-level sliders (not per-ISO). Total corporate participation = hyperscaler_share × hyperscaler_pct + other_share × other_pct, applied uniformly across ISOs.

**Rationale:** Market is increasingly bifurcated (BNEF 2025: tech = 84% of deal activity, only 33 unique buyers). Modeling the two cohorts separately captures the structural difference between hyperscaler procurement capacity and mid-market adoption.

### §15.4 Target/Outcome Model (Decided Feb 27)

**Dual-mode:**

1. **Per-buyer emission reduction target:** Each participating buyer targets X% clean energy (using the CFE threshold slider, paralleling existing dashboard). The *calculation* of what constitutes "X% clean" differs by strategy:
   - Strategy 1: Cross-regional netting against emission baseline (grid-avg/fossil-avg/marginal)
   - Strategy 2: Hourly matching within ISO (with or without existing clean credit)
   - Strategy 3: Annual MWh matching (same-ISO or cross-regional)

2. **System-wide CO₂ reduction panel:** Separate interactive panel. User sets a system-wide CO₂ reduction target (e.g., "reduce US power sector emissions by 30%"), and the model backs into what corporate participation rate each strategy would need to achieve it. Shows required participation as a function of strategy choice.

### §15.5 Dashboard Page Design (Updated Feb 27)

**Hybrid scrollytell + interactive** with integrated tradeoff matrix, failure mode demonstration, and strategy horse race.

**Core thesis:** Every procurement strategy has failure modes at scale. The design details within each strategy family matter as much as the family choice itself. The page makes the reader uncomfortable about *all* the options, then shows which design choices minimize systemic risk.

**Key framing:** This is not a polemic against consequential — it's a rigorous demonstration that strategy choice is nuanced. The GHG Protocol debate is largely framed as "should we allow consequential?" when the real question is "which *version* of any matching approach actually works at scale?"

**Structure:**
1. **Scrollytell intro:** Explains the three strategy families (1: Consequential, 2: Hourly, 3: Annual), builds intuition about tradeoffs. Key message: "every strategy looks fine at 10% participation — the question is what happens at scale."
2. **Tradeoff matrix:** Summary table showing all 10 strategies × key metrics (cost, CO₂, build required, $/tCO₂) — embedded in scrollytell flow
3. **Failure mode demonstration charts (5 interactive):** Participation slider as unifying x-axis across all charts. All 10 strategies shown. As participation increases, watch each strategy's failure modes activate. Charts link to deep-dive research pages for full analysis.
4. **Strategy horse race:** Fixed outcome comparison + fixed budget comparison — embedded in scrollytell
5. **Interactive explorer:** Strategy selector, participation sliders, CFE threshold slider, ISO selector. Full exploratory mode.
6. **System-wide panel:** CO₂ reduction target → required participation by strategy

#### §15.5.1 Failure Mode Demonstration Charts (Decided Feb 27)

Five interactive charts, all sharing a **participation rate slider** (x-axis: 0-80% of C&I load). Each chart shows all 10 strategy variants as lines/areas. The participation slider is the unifying interaction — as you drag it from 5% to 80%, you watch every failure mode activate in sequence.

**Chart 1: Cost Trajectory Divergence**
- Y-axis: Effective $/MWh at the selected CFE threshold
- Shows all 10 strategies diverging as participation scales
- At low participation, strategies cluster. At high participation, massive spread.
- Key inflection: Where Strategy 2A (all new-build hourly) starts triggering wholesale erosion, converging cost-wise toward Strategy 1 failures despite different mechanism
- Data source: Scenario comparison trajectories + Step 3 cost optimization repriced per strategy

**Chart 2: Capital Allocation by ISO**
- Y-axis: % of total clean energy investment going to each ISO
- Shows geographic clustering: Strategy 1/3B/3D concentrate capital in coal-heavy ISOs (SPP, MISO). Strategy 2/3A/3C forced same-ISO.
- "Fair share" reference lines (proportional to ISO demand)
- Key inflection: 20-30% participation where gas grids drop below fair share under cross-regional strategies
- Data source: Deployment queue from consequential_queue.json, indexed by cumulative TWh as % of C&I load

**Chart 3: Wholesale Price Erosion & Existing Clean Stranding**
- Y-axis: Estimated wholesale LMP ($/MWh) + Section 45U strike price reference
- Shows merit-order effect: as clean penetration rises under each strategy, wholesale prices drop
- **Key insight for Strategy 2 debate:** 2A (all new-build) accelerates wholesale erosion locally — floods market with zero-marginal-cost gen. 2C (premium + new) mitigates via revenue floor for existing generators. 2B falls between.
- Annotation: "When LMP < $44 (45U strike) - operating costs → nuclear stranding begins"
- Links to: [Cost to Replace →] and [New Build Analysis →] for full regional breakdown
- Data source: LMP reconstruction from step5_compute_lmp_prices.py + Track 3 CTR data

**Chart 4: MAC Escalation (Marginal Abatement Cost)**
- Y-axis: $/tCO₂ for the marginal ton abated under each strategy
- Consequential (Strategy 1) starts cheap then hits a wall when coal exhausted
- Hourly (Strategy 2) starts higher but stays flatter — no saturation cliff
- Annual (Strategy 3) variants fall between, depending on boundary + additionality
- Key inflection: Coal exhaustion point (varies by ISO) where Strategy 1 MAC jumps to gas-displacement levels
- Data source: MAC stats from step5_compute_mac_stats.py + deployment queue MAC ordering

**Chart 5: Resource Mix Divergence**
- Y-axis: Stacked resource mix (clean firm, solar, wind, CCS, battery, LDES)
- Side-by-side or toggled comparison at selected threshold showing what gets built under each strategy
- Strategy 1/3 = VRE-heavy, no firm. Strategy 2A = firm + storage + VRE. Strategy 2C = existing + firm + VRE.
- Key insight: Strategy 2A and 2C build different mixes — 2A builds more new firm (drives learning), 2C preserves existing (prevents stranding). Different tradeoff.
- Data source: Scenario comparison resource trajectories + Step 3 resource mix data

#### §15.5.2 Strategy 2 Internal Debate (Decided Feb 27)

The page must demonstrate that the debate within hourly matching (2A vs 2B vs 2C) is as consequential as the debate between strategy families:

| Dimension | 2A (All New) | 2B (Grid Baseline) | 2C (Premium + New) |
|---|---|---|---|
| Wholesale erosion | **Accelerates** — new zero-marginal gen floods market | **Accelerates** — takes credit for existing without supporting it, still adds new | **Mitigates** — premium provides revenue floor for existing |
| Existing clean stranding | **Ignores** — no revenue signal, doesn't acknowledge existing clean exists | **Strands** — takes credit for existing generation without paying for it. Worst of both: claims the benefit while starving generators of revenue signal to stay online | **Addressed** — explicit clean premium keeps plants viable by paying them for being clean |
| Learning curves | **Maximum** — most new firm built earliest | **Moderate** — less new build needed (credited baseline reduces requirement) | **Moderate** — premium $ supports existing, new build on top |
| Cost trajectory | Highest near-term → lowest long-term (FOAK→NOAK) | Cheapest near-term (free-rides on existing), but vulnerable to replacement spike when unpaid existing retires | Higher near-term, avoids replacement spike — pays now to prevent paying more later |
| Firm investment signal | **Strong** — hourly constraint forces it | **Diluted** — baseline credit reduces urgency to build | **Strong** for both existing (premium) + new (hourly constraint) |
| Additionality | **Maximum** — 100% new build | **Problematic** — claims credit for existing without driving new investment or sustaining existing | **Transparent** — explicitly values existing clean (premium) and requires new build on top |

**Key question the charts must demonstrate:** At high participation, does 2A converge toward the same wholesale erosion failure mode as Strategy 1 — just locally instead of cross-regionally? Both flood the market with zero-marginal-cost generation without a mechanism to preserve existing clean. The geography is different but the wholesale destruction is the same.

**2C as the reference strategy — and its one failure mode:** Strategy 2C's failure mode is the inverse of every other strategy. Every other approach degrades as participation *increases* — signal degradation, saturation, wholesale erosion, stranding. 2C's risk is at *insufficient* scale: if participation stays too low, all you've done is pay a premium to keep existing nuclear and hydro running (valuable but not transformative). You never build enough new firm clean to push the learning curve from FOAK to NOAK. The premium keeps existing assets alive but doesn't drive the additionality needed for the NOAK ROI.

Above a critical mass threshold — the participation level where cumulative new-build firm clean investment under 2C is sufficient to trigger Wright's Law learning — 2C is the only strategy that both preserves existing clean AND drives NOAK economics. Below that threshold, it's just an expensive maintenance program for existing generation.

**The chart moment:** Every other strategy line degrades as the participation slider moves right. 2C's risk zone is on the *left* — shaded band below the critical mass threshold. Once past that threshold, 2C is structurally sound at any participation level. The page should make this visually obvious: 2C starts in a "needs more adoption" zone, crosses into "works at any scale," while every other strategy starts in "looks fine" and crosses into various failure modes.

The critical mass threshold is quantifiable via two key numbers:

**1. Critical mass threshold (% participation):** The participation level where *aggregate* new-build firm clean volume across all 7 ISOs under 2C exceeds the deployment needed for Wright's Law cost reductions (first doubling of installed capacity per technology). Critically, learning is **global, not regional** — a nuclear plant built in PJM drives NOAK for nuclear everywhere. LDES deployed in ERCOT brings down iron-air costs in NEISO. So the threshold is lower than a per-ISO calculation would suggest because all 7 ISOs contribute to the same global learning pool.

**2. Investment pool composition:** At each participation level, total 2C spend splits into:
- **Existing clean premiums** (keeping nuclear/hydro online — maintenance spend)
- **New-build capital** (firm clean + storage — learning curve spend)

The split varies dramatically by ISO. Nuclear-heavy ISOs (PJM 32%, NEISO 24%) allocate more to premiums; renewable-heavy ISOs (ERCOT, SPP) have less existing clean to maintain → higher share flows to new-build → they are the **learning curve workhorses** even though they're not where the premium payments concentrate.

**Step 8 computation target:** For each participation level (0-80% of C&I), compute:
- Total new-build firm TWh across all ISOs (the number that matters for Wright's Law)
- Premium spend vs new-build spend by ISO (shows where learning investment concentrates)
- Participation threshold where aggregate new-build hits first doubling → NOAK pricing activates
- Post-NOAK cost trajectory showing the strategy pays for itself

Data sources: Track 2 NB (new-build costs), Track 3 CTR (existing premium costs), learning curve parameters from `step6_scenario_comparison.py`, resource mix data from shared-data.js.

**Step 8 implementation: `scripts/step8_wrights_law_curves.py`** — COMPLETE. Vectorized numpy, no sequential loops. Output:
- `data/step8-wrights-law/wrights_law_curves.parquet` (4 KB, snappy compressed — 252 rows: 12 participation levels × 21 thresholds)
- `data/step8-wrights-law/wrights_law_curves.json` (8.4 KB — dashboard-ready figure data)

Key results at 90% CFE:
- **Critical mass threshold: 25% C&I participation** — where cumulative CCS-CCGT deployment exceeds 8 GW globally
- At 95% CFE: 10% participation; at 99.99%: 5% (higher thresholds drive more new-build per participant)
- PJM dominates new-build spend (largest demand, most CCS needed); SPP is 100% premium (all wind, zero new firm)
- Wright's Law gating: learning fraction = 0 below critical mass (maintenance mode), ramps via exponent 0.6 above it
- First-doubling thresholds: nuclear 5 GW, CCS 8 GW, LDES 3 GW, geothermal 2 GW (DOE Liftoff / INL SOAR calibrated)

#### §15.5.3 Cross-References to Deep Dive Pages

The procurement comparison page is a **hub**. At each failure mode inflection point, surface the relevant link:
- **Wholesale erosion / stranding → [Cost to Replace]** (`cost_to_replace.html`) — full regional replacement premium analysis
- **Wholesale erosion / stranding → [New Build Analysis]** (`new_build_analysis.html`) — supply ceiling, LMP feedback loop, 45U stranding threshold
- **Geographic clustering / saturation → [Consequential Vacuum]** (`consequential_vacuum.html`) — 5 failure modes deep dive with dispatch-based evidence
- **Learning curves / cost trajectory → [Scenario Comparison]** (`scenario_comparison.html`) — FOAK→NOAK dynamics, Scenario A vs B full trajectories
- **MAC escalation → [Abatement Dashboard]** (`abatement_dashboard.html`) — MAC fan charts, DAC crossover, optimal target analysis

Charts on this page show *that* something is happening at a participation threshold. Links take the reader to the page that explains *why* in depth.

#### §15.5.4 Failure Mode × Strategy Matrix (Reference)

| Failure Mode | 1A | 1B | 1C | 2A | 2B | 2C | 3A | 3B | 3C | 3D |
|---|---|---|---|---|---|---|---|---|---|---|
| Signal degradation | Worst | Bad | Less bad | Immune | Immune | Immune | Immune | Immune | Vulnerable | Vulnerable |
| Saturation (coal wall) | Yes | Yes | Yes | N/A | N/A | N/A | N/A | Yes | N/A | Yes |
| Fossil lock-in | Severe | Severe | Severe | Low | Low | Low | Partial | Partial | Severe | Severe |
| Geographic clustering | Core | Core | Core | Impossible | Impossible | Impossible | Impossible | Replicates | Impossible | Replicates |
| Wholesale erosion | Accelerates | Accelerates | Accelerates | **Accelerates** | **Accelerates** | **Mitigates** | Neutral | Accelerates | Accelerates | Accelerates |
| Existing clean stranding | No signal | No signal | No signal | **Ignores** | **Strands** (worst — claims credit without paying) | **Addressed** | Neutral | No signal | Free-rides | No signal |

**Critical nuance:** 2B is arguably *worse* than 2A for existing clean stranding. 2A ignores existing clean — doesn't help, doesn't harm. 2B actively takes credit for existing clean generation (reducing the buyer's procurement cost and requirement) without directing any revenue to those generators. It free-rides on existing clean being online while starving it of the payment signal needed to stay online. When the unpaid existing generation retires, 2B buyers face the same replacement cost spike as everyone else — but they've also reduced the market signal that could have prevented the retirement. Strategy 2C is the only hourly variant that explicitly pays existing generators for being clean, creating the revenue floor needed to prevent premature retirement.

### §15.5.5 Page Narrative Map & Figure Descriptions (Feb 27)

**File:** `dashboard/procurement_comparison.html` (new page — `procurement_research.html` is the content plan, kept separately)

---

#### ACT 1: "EVERY STRATEGY LOOKS FINE AT LOW ADOPTION" (Scrollytell)

**Purpose:** Build the reader's mental model of the three strategy families, then plant the seed that things break at scale.

**Section 1.1 — Opening Hook**
- **Text:** "Today, ~13% of US commercial & industrial electricity is covered by voluntary clean energy procurement. At that level, every strategy works. The question isn't which approach looks best at 13% — it's which ones survive at 40%, 60%, 80%."
- **Visual:** Animated counter showing current market: 315 TWh / 2,400 TWh C&I load = 13%. Simple, cinematic.
- No chart. Just the number landing with weight.

**Section 1.2 — The Three Families**
- **Text:** Brief intro to each strategy family (3 cards, scroll-triggered reveal):
  - **Consequential (Strategy 1):** "Buy the cheapest clean energy anywhere in the US. Net it against your emissions. Maximum flexibility, minimum cost."
  - **Hourly (Strategy 2):** "Match your load hour-by-hour within your own grid region. Most rigorous. Most expensive."
  - **Annual (Strategy 3):** "Match your annual consumption with clean energy certificates. The status quo for most buyers today."
- **Visual:** Three strategy family cards with icons. No chart yet. Clean, simple.

**Section 1.3 — The Variant Tree**
- **Text:** "But within each family, design choices matter enormously. There are 10 distinct strategy variants — and the differences within a family can be larger than the differences between families."
- **FIGURE 1: Strategy Taxonomy Tree**
  - **Type:** Interactive tree/org-chart diagram
  - **Description:** Visual hierarchy: 3 families → 10 variants. Each node shows: variant code (1A, 2C, etc.), one-line description, key distinguishing feature. Color-coded by family. Clicking a variant highlights it across all subsequent charts.
  - **Data source:** Static (strategy definitions from §15.2). No compute needed.
  - **Key design:** This becomes the "legend" for the rest of the page. Reader builds familiarity with the codes here so they can track them through subsequent figures.

---

#### ACT 2: "WHAT BREAKS, AND WHEN" (Scrollytell → Interactive)

**Purpose:** The core analytical payload. Five failure mode demonstrations, each building on the last. Shared participation slider ties them together.

**Transition text:** "Now drag the participation slider from 13% to 80% and watch what happens to each strategy."

**Global control: Participation Slider** — Sticky/floating, visible across all Act 2 charts. Range: 5–80% of C&I load. Default position: 13% (current market). Dragging it right is the primary interaction.

**Section 2.1 — Cost Trajectory Divergence**
- **Lead text:** "At low participation, all strategies cost roughly the same. At high participation, the spread is enormous."
- **FIGURE 2: Cost Divergence Fan**
  - **Type:** Multi-line chart (10 lines, one per strategy variant)
  - **X-axis:** Participation rate (5–80% of C&I load)
  - **Y-axis:** Effective cost ($/MWh) at 90% CFE threshold
  - **Behavior:** At 13%, lines cluster in a $30–50/MWh band. As slider moves right, lines diverge. Strategy 3D (status quo RECs) stays flat and cheap. Strategy 2A (all new-build hourly) rises steeply then curves. Strategy 1A (consequential, grid-avg) stays cheap until ~40% then jumps (coal exhaustion).
  - **Key annotation:** Vertical dashed line at current market (13%). Shaded "comfort zone" where strategies look similar.
  - **Data source:** For strategies 2A and 2C — **existing data**: `track_results.json` (newbuild = 2A, cost_to_replace = 2C) repriced at Medium costs gives $/MWh at each threshold. Map thresholds to participation via SBTi timeline. For 3D — near-zero (unbundled REC price). For 1A/1B/1C — derive from `consequential_queue.json` (deployment queue MAC × emission rate gives effective cost). **Strategies 2B, 3A, 3B, 3C need Step 8 compute** — show as dashed/estimated lines initially.
  - **Callout box:** "The cheap strategies aren't actually cheap — they're deferring costs to the future."

**Section 2.2 — Where the Money Goes (Geographic Clustering)**
- **Lead text:** "Cross-regional strategies chase the cheapest abatement. That concentrates investment in a few regions and starves others."
- **FIGURE 3: Capital Allocation Heatmap**
  - **Type:** Stacked bar chart or heatmap (7 ISOs × selected strategies)
  - **At each participation level:** Shows what % of total clean energy investment goes to each ISO
  - **Strategy 1 (consequential):** Capital clusters in SPP and MISO (coal-heavy, cheapest MAC). CAISO, NEISO, NYISO get almost nothing.
  - **Strategy 2 (hourly):** Capital distributed proportionally — each ISO serves its own load.
  - **"Fair share" reference:** Dashed lines showing demand-proportional allocation.
  - **Data source:** `consequential_queue.json` → `deployment_queue` entries have `iso` field. Sum `delta_cost_total_bn` by ISO at each cumulative step. For hourly: `track_results.json` — inherently same-ISO so allocation = demand share. **Available now for Strategy 1 and 2. Strategy 3 variants need Step 8.**
  - **Callout box:** "When SPP and MISO receive 60%+ of clean investment while serving 25% of demand, the other five regions are subsidizing their transition while getting none of their own."

**Section 2.3 — Wholesale Destruction & Nuclear Stranding**
- **Lead text:** "Every MWh of zero-marginal-cost generation added to a grid pushes wholesale prices down. That's great for consumers — until it kills the existing clean generation you're counting on."
- **FIGURE 4: Wholesale Price Erosion**
  - **Type:** Dual-axis line chart. Primary: LMP ($/MWh). Secondary: Nuclear operating cost reference line.
  - **X-axis:** Participation rate (5–80%)
  - **Lines:** LMP trajectory under Strategy 1 (cross-regional), Strategy 2A (all new, same-ISO), Strategy 2C (premium + new, same-ISO)
  - **Key feature:** Horizontal band at ~$44/MWh (45U strike price) with annotation: "Below this line, existing nuclear can't cover operating costs." When Strategy 2A's line crosses below this band, highlight it.
  - **Strategy 2C difference:** Its line stays higher because the premium mechanism acts as a revenue floor — you're paying existing clean to stay online rather than flooding the market with competing new zero-marginal gen.
  - **Data source:** `data/step5-post-processing/lmp/lmp_summary.json` has PJM LMP data. `scenario_comparison.json` has stranding analysis (`stranding_a`, `stranding_b`). **LMP currently computed for PJM only — show PJM as representative, note other ISOs forthcoming.** Track 3 CTR effective costs from `track_results.json` provide the "premium" price signal.
  - **Callout box:** "Strategy 2A and Strategy 1 both destroy wholesale prices — they just do it in different geographies. Strategy 2C is the only variant with a built-in mechanism to prevent it."

**Section 2.4 — The MAC Wall**
- **Lead text:** "Consequential strategies look cheap because they pick off the lowest-hanging fruit first. But that fruit runs out."
- **FIGURE 5: Marginal Abatement Cost Escalation**
  - **Type:** Line chart with shaded uncertainty bands
  - **X-axis:** Cumulative CO₂ abated (Mt) — maps to participation via deployment queue
  - **Y-axis:** Marginal $/tCO₂ for the next ton abated
  - **Strategy 1 (consequential):** Starts at ~$80-100/tCO₂ (coal displacement in SPP/MISO). Stays flat through ~200 Mt. Then **wall** when coal is exhausted → jumps to $300-500+/tCO₂ (gas displacement).
  - **Strategy 2 (hourly):** Starts higher (~$150-200/tCO₂) but stays **flatter** — no saturation cliff because you're always building the full stack (firm + storage + VRE) rather than cherry-picking.
  - **Horizontal reference bands:** DAC ($400-600), EU ETS ($60-100), EPA SCC ($190), Rennert SCC ($185).
  - **Data source:** `mac_stats.json` — `stepwise_fan` has P10/P50/P90 MAC by threshold for all 7 ISOs. `consequential_queue.json` → `deployment_queue` has `marginal_mac` per zone. `scenario_comparison.json` → `queue_a` and `queue_b` have MAC trajectories for both scenarios. **Fully available now.**
  - **Callout box:** "The coal wall is a cliff, not a hill. Once you've displaced all the coal, the next ton costs 3-5× more."

**Section 2.5 — What Gets Built**
- **Lead text:** "Different strategies build different grids. That matters more than the cost difference."
- **FIGURE 6: Resource Mix Comparison**
  - **Type:** Stacked area or grouped bar chart
  - **Comparison:** At a selected threshold (default 90%), show the resource mix under 3-4 key strategies side by side
  - **Strategy 1 (consequential):** Heavy VRE (solar + wind), minimal firm clean, no storage. Gas fills the gaps.
  - **Strategy 2A (all new hourly):** VRE + firm clean + battery + LDES. Balanced portfolio forced by hourly constraint.
  - **Strategy 2C (premium + new hourly):** Existing nuclear/hydro preserved + new firm + VRE + storage. Most diversified.
  - **Strategy 3D (status quo annual):** Unbundled RECs from existing — no new build at all. Cheapest but builds nothing.
  - **Color coding:** Solar=amber, Wind=blue, Clean Firm=green, CCS=teal, Hydro=cyan, Battery=purple, LDES=pink (per project standard)
  - **Data source:** `track_results.json` → `resource_mix` for newbuild (2A) and cost_to_replace (2C) at each threshold. `scenario_comparison.json` → `trajectories` → `pure_consequential` has `resource_twh` per threshold for Strategy 1. `shared-data.js` → `RESOURCE_MIX_DATA` has Medium-cost mix by ISO. **Available now for 1, 2A, 2C. 3D is trivially zero new-build.**
  - **Callout box:** "Hourly matching is the only approach that forces investment in the resources you actually need for a deeply decarbonized grid — firm clean generation and long-duration storage."

---

#### ACT 3: "THE DEBATE WITHIN HOURLY" (Scrollytell)

**Purpose:** Shift from family-level comparison to the 2A vs 2B vs 2C internal debate. This is the most nuanced section — it argues that the debate *within* hourly matching is as important as the debate *between* strategies.

**Transition text:** "If hourly matching is the most robust family, the next question is: which version? The differences are larger than you'd think."

**Section 3.1 — The Stranding Paradox**
- **Text:** "Strategy 2A (all new-build) sounds maximally additional. But it ignores the 380 TWh of existing nuclear and hydro already running on US grids. Strategy 2B claims credit for that existing clean without paying for it — the worst of both worlds. Strategy 2C pays existing generators a premium to stay online, then builds new on top."
- **FIGURE 7: Strategy 2 Internal Comparison Table**
  - **Type:** Animated comparison table (not a chart — a styled, scroll-triggered table)
  - **Rows:** 6 dimensions: Wholesale erosion, Existing clean stranding, Learning curves, Cost trajectory, Firm investment signal, Additionality
  - **Columns:** 2A, 2B, 2C — color-coded cells (green=good, amber=mixed, red=bad)
  - **Scroll animation:** Rows reveal one at a time as user scrolls. Each row highlights the "winner" and "loser."
  - **Data source:** Static (the §15.5.2 comparison table). No compute needed.
  - **Key moment:** When the "Existing clean stranding" row reveals, 2B's cell turns red with bold text: "Worst — claims credit without paying." This is the insight most readers won't expect.

**Section 3.2 — The 2C Critical Mass Question**
- **Text:** "Every other strategy degrades as adoption increases. Strategy 2C has the opposite problem — it needs *enough* adoption to work. Below a critical mass threshold, you're just paying premiums to keep existing plants alive. Above it, you've funded the learning curve that makes new clean firm affordable everywhere."
- **FIGURE 8: The 2C Threshold Diagram**
  - **Type:** Single-line chart with shaded zones
  - **X-axis:** Participation rate (5–80%)
  - **Y-axis:** Effective $/MWh for Strategy 2C (blended existing premium + new build)
  - **Key feature:** Two shaded zones:
    - **Left zone (red/amber):** "Maintenance mode" — below critical mass. Premium spend dominates. Not enough new-build volume to trigger learning.
    - **Right zone (green):** "Learning activated" — past critical mass. Aggregate new-build firm across all ISOs hits first Wright's Law doubling. NOAK pricing begins. Cost curve bends down.
  - **Vertical line:** Critical mass threshold (computed from aggregate new-build TWh needed for first doubling)
  - **Contrast overlay:** Faded lines for other strategies showing their degradation at high participation — 2C is the only one that *improves* past its threshold.
  - **Data source:** `track_results.json` cost_to_replace effective costs for 2C base. `scenario_comparison.json` trajectories for learning curve application. Critical mass threshold = **Step 8 compute needed** for exact number, but can estimate from existing resource mix data (how much new firm TWh at each participation level × learning curve parameters from `step6_scenario_comparison.py`).
  - **Callout box:** "The critical mass threshold is lower than you'd think — because learning is global. A nuclear plant built in PJM drives NOAK pricing for nuclear in NEISO. All 7 ISOs contribute to the same global learning pool."

**Section 3.3 — The Regional Role Map**
- **Text:** "Under Strategy 2C, different regions play different roles. Nuclear-heavy ISOs (PJM, NEISO) are the premium-payers — their spend keeps existing clean alive. Renewable-rich ISOs (ERCOT, SPP) are the learning-curve drivers — their spend deploys the new technologies. Both are essential."
- **FIGURE 9: Investment Pool Composition by ISO**
  - **Type:** Stacked bar chart (7 ISOs)
  - **Each bar split into:** Existing clean premium (gray/blue) vs. New-build capital (green/amber)
  - **At selected participation level** (linked to global slider)
  - **Key insight:** PJM's bar is 60-70% premium (keeping 32% nuclear fleet online). ERCOT's bar is 80%+ new-build (little existing clean to maintain). Both contribute to the same NOAK outcome.
  - **Data source:** `track_results.json` — newbuild (2A) gives new-build costs; cost_to_replace (2C) gives total including premium. Delta = premium portion. **Available now for 5 ISOs** (CAISO, ERCOT, NEISO, NYISO, PJM). MISO, SPP from `shared-data.js` resource mix data.
  - **Callout box:** "PJM free-rides on the learning curve that ERCOT is paying for — and that's the system working as designed."

---

#### ACT 4: "THE TIMELINE TRAP" (Scrollytell)

**Purpose:** Introduce the temporal dimension — SBTi milestones create a timeline that makes strategy choice path-dependent. What you choose at 50% determines what you face at 90%.

**Section 4.1 — The SBTi Ratchet**
- **Text:** "Corporate decarbonization isn't a static optimization — it's a 25-year ratchet. Science-based targets lock in progressively tighter commitments: 50% by 2030, 70% by 2035, 90% by 2040, 100% by 2050. What you build at 50% determines what you face at 90%."
- **FIGURE 10: The FOAK→NOAK Timeline**
  - **Type:** Dual-trajectory line chart with SBTi milestone markers
  - **X-axis:** Year (2025–2050), with SBTi milestone markers
  - **Y-axis:** Clean firm LCOE ($/MWh) — blended nuclear + CCS + LDES
  - **Line A (Strategy 1/3 — Consequential/Annual):** Stays at FOAK through 2035 (no investment). Learning starts 2035, compressed. Still near-FOAK at 2040 (the 90% milestone). Reaches NOAK only by ~2047.
  - **Line B (Strategy 2 — Hourly):** FOAK from 2025-2030. Learning 2030-2040. NOAK by 2040 — right when you need it for the 90% target.
  - **Shaded band:** Cost difference area between the two lines = "the learning curve premium" — what hourly matching costs upfront vs. what it saves long-term.
  - **SBTi markers:** Vertical dashed lines at 2030 (50%), 2035 (70%), 2040 (90%), 2050 (100%)
  - **Data source:** `step6_scenario_comparison.py` → `learning_fraction()` gives the curve shape. LCOE_TABLES from `shared-data.js` give FOAK (High) and NOAK (Low) costs. `scenario_comparison.json` → `trajectories` have `blended_new_lcoe` at each threshold/year. **Fully available now.**
  - **Callout box:** "By 2040, hourly matching has driven clean firm costs to NOAK. Consequential strategies are still paying near-FOAK — for the same technology, at the same time, because they delayed investment."

**Section 4.2 — Three Compounding Failures**
- **Text:** Brief narrative on the three adverse effects of delayed hourly matching (from §15.11):
  1. Learning curve delay → FOAK at 90%
  2. Stranded VRE overbuild → curtailed solar doesn't help at night
  3. Gas lock-in → no storage signal, gas fills by default
- **FIGURE 11: The Compounding Timeline**
  - **Type:** SBTi milestone comparison table (styled, scroll-animated)
  - **Rows:** 4 SBTi milestones (2030/50%, 2035/70%, 2040/90%, 2050/100%)
  - **Columns:** "Strategy 1/3 (Annual/Consequential)" vs "Strategy 2 (Hourly)"
  - **Cell content:** Status descriptor + cost indicator. E.g., 2040 row: Strategy 1/3 = "WALL — VRE saturated, firm at FOAK, gas locked in" (red). Strategy 2 = "Firm at NOAK, storage mature, gas retiring" (green).
  - **Data source:** Static text from §15.11 table, enriched with cost numbers from trajectory data. **Available now.**

---

#### ACT 5: "THE HORSE RACE" (Interactive)

**Purpose:** Direct comparison mode. Same outcome, which strategy gets there cheapest? Same budget, which strategy achieves the most?

**Section 5.1 — Fixed Outcome: "Get to 90% CFE. What does it cost?"**
- **FIGURE 12: Cost to Reach 90% by Strategy**
  - **Type:** Horizontal bar chart (10 strategies ranked by cost)
  - **Y-axis:** Strategy variants (labeled)
  - **X-axis:** Effective $/MWh to achieve 90% hourly CFE (or equivalent)
  - **Key insight:** Strategy 3D is "cheapest" but achieves nothing physical. Strategy 2A is most expensive but builds the most. Strategy 2C is moderate and sustainable.
  - **Annotations:** Each bar annotated with what it actually built (resource mix icons) and what's at risk (failure mode flag)
  - **Data source:** For 2A and 2C: `track_results.json` at threshold 90, Medium scenario. For Strategy 1: `scenario_comparison.json` trajectory at 90% threshold. **Available now for 1, 2A, 2C. Others need Step 8.**
  - **Toggle:** ISO selector (default: all-ISO weighted average)

**Section 5.2 — Fixed Budget: "Spend $60/MWh. What do you get?"**
- **FIGURE 13: Achievement at Fixed Budget**
  - **Type:** Horizontal bar chart (10 strategies ranked by CFE% achieved)
  - **X-axis:** CFE threshold achieved with $60/MWh budget
  - **Annotations:** Each bar annotated with CO₂ abated and resource mix
  - **Data source:** Interpolate from cost curves at each threshold. **Derivable from existing data for strategies with cost trajectories.**

---

#### ACT 6: "THE SYSTEM VIEW" (Interactive Panel)

**Purpose:** Flip the question. Instead of "what does it cost per buyer?" ask "what participation level does each strategy need to hit a system-wide CO₂ target?"

**Section 6.1 — Required Participation by Strategy**
- **FIGURE 14: Participation Required for 30% US Power Sector CO₂ Reduction**
  - **Type:** Horizontal bar chart or gauge visualization
  - **Y-axis:** Strategy variants
  - **X-axis:** Required C&I participation (% of load)
  - **Key insight:** Some strategies can't get there at any participation level (supply constraints, saturation). Others need implausibly high participation. A few are feasible at realistic levels.
  - **Infeasibility markers:** Strategies that hit physical supply ceilings before reaching the target show as "infeasible" with hatched bars.
  - **Data source:** Requires mapping strategy → CO₂ displaced at each participation level. `consequential_queue.json` has `co2_displaced_mt` per step for consequential. `co2_results.json` for dispatch-based emission reductions (currently PJM only). **Partial data — full computation is Step 8.**
  - **User control:** CO₂ reduction target slider (10–50% of US power sector emissions)

**Section 6.2 — Interactive Explorer (Full Controls)**
- **Text:** "Explore the full parameter space."
- **Controls:**
  - Strategy selector (checkbox — select multiple to compare)
  - Participation slider (5–80%)
  - CFE threshold slider (50–100%)
  - ISO selector (7 ISOs or all)
  - Learning curves toggle (On/Off)
- **FIGURE 15: Explorer Output Panel**
  - Multi-panel: Cost ($/MWh), Resource mix (stacked bar), CO₂ abated (bar), MAC (line), Gas capacity (bar)
  - Updates in real-time as controls change
  - **Data source:** All existing data files, combined. This is the "power user" interface. **Largely available for strategies with computed data.**

---

#### PAGE FOOTER

Cross-reference links to deep-dive pages:
- Cost to Replace → `cost_to_replace.html`
- New Build Analysis → `new_build_analysis.html`
- Consequential Vacuum → `consequential_vacuum.html`
- Scenario Comparison → `scenario_comparison.html`
- Abatement Dashboard → `abatement_dashboard.html`

---

#### DATA AVAILABILITY SUMMARY

| Figure | Strategies with Data Now | Strategies Needing Step 8 |
|--------|--------------------------|---------------------------|
| Fig 1 (Taxonomy tree) | All 10 (static) | — |
| Fig 2 (Cost divergence) | 1A-C (from queue), 2A, 2C, 3D (≈$0) | 2B, 3A, 3B, 3C |
| Fig 3 (Capital allocation) | 1 (queue), 2 (inherently same-ISO) | 3 variants |
| Fig 4 (Wholesale erosion) | 2A, 2C (PJM LMP data) | 1, 2B, 3 |
| Fig 5 (MAC escalation) | 1 (queue MACs), 2 (mac_stats) | 3 variants |
| Fig 6 (Resource mix) | 1 (trajectory), 2A, 2C, 3D (=nothing) | 2B, 3A, 3B, 3C |
| Fig 7 (2ABC table) | All (static) | — |
| Fig 8 (2C threshold) | 2C (track data + learning curve) | Critical mass exact point |
| Fig 9 (Regional roles) | 2C (track data, 5 ISOs) | MISO, SPP |
| Fig 10 (FOAK→NOAK) | Both (learning_fraction + LCOEs) | — |
| Fig 11 (Compounding table) | All (static + trajectory data) | — |
| Fig 12 (Horse race: cost) | 1, 2A, 2C | 2B, 3A-D |
| Fig 13 (Horse race: budget) | 1, 2A, 2C | 2B, 3A-D |
| Fig 14 (System CO₂) | Partial (consequential queue CO₂) | Most strategies |
| Fig 15 (Explorer) | 1, 2A, 2C | 2B, 3A-D |

**Bottom line:** Acts 1–4 (the scrollytell narrative) are ~80% buildable with existing data. Strategies 2A and 2C have the richest data. Strategy 1 has deployment queue data. The remaining strategies (2B, 3A-3D) need Step 8 compute for precise numbers but can be shown as estimated/dashed lines derived from the strategies we do have.

### §15.6 Emission Rate Data (Research, Feb 27)

| ISO | Grid Avg (tCO₂/MWh) | Fossil Avg | Marginal | Marginal vs Fossil |
|-----|---------------------|------------|----------|-------------------|
| CAISO | 0.168 | 0.392 | 0.397 | ≈ same |
| ERCOT | 0.333 | 0.535 | 0.526 | ≈ same |
| PJM | 0.325 | 0.539 | 0.573 | +6% |
| MISO | 0.354 | 0.567 | 0.663 | **+17%** |
| NYISO | 0.217 | 0.415 | 0.437 | +5% |
| NEISO | 0.246 | 0.387 | 0.425 | +10% |
| SPP | 0.340 | 0.544 | 0.665 | **+22%** |

Sources: EPA eGRID2023 (grid-avg, fossil-avg), VERACI-T/WattTime (marginal), Holland et al. 2022 PNAS.

### §15.7 EAC Scarcity by ISO (Research, Feb 27)

| ISO | Total Clean (TWh) | Committed (TWh) | Available for Voluntary (TWh) | REC Price Signal |
|-----|-------------------|-----------------|------------------------------|-----------------|
| CAISO | ~172 | ~158-164 | 10-20 | Moderate |
| ERCOT | ~200-205 | ~30-40 | **130-160** | Very Low ($1-5) |
| PJM | ~310-330 | ~230-290 | 50-80 | High ($35+) |
| MISO | ~200-215 | ~170-230 | 30-50 | Low-Moderate |
| NYISO | ~62-66 | ~47-63 | 5-15 | High ($20-35) |
| NEISO | ~43-45 | ~35-60 | **3-8** | Critical (~$40) |
| SPP | ~136 | ~65-90 | 40-60 | Very Low ($1-5) |

**Key finding:** 20x scarcity variation across ISOs. ERCOT has ~130-160 TWh unclaimed; NEISO has ~3-8 TWh. This directly drives the economics of cross-regional (Strategy 1/3B) vs same-ISO (Strategy 2/3A) strategies.

### §15.8 C&I Load Share (Research, Feb 27)

National C&I = ~62% of total US load (~2,400 of ~3,860 TWh). Range by ISO: 52-57% (NEISO) to 63-67% (ERCOT). Voluntary procurement currently covers ~13% of C&I load (~315 TWh/yr, NREL 2024).

### §15.9 Corporate Procurement Market (Research, Feb 27)

- Voluntary market: ~315 TWh (2024), 7.7% of total US demand
- Corporate PPAs: 28 GW signed in 2024, 29.5 GW in 2025 (BNEF)
- Concentration: Tech/data = 84% of deal activity; Big 4 hyperscalers = 49% of global activity
- Unique US buyers: Fell 51% YoY to 33 companies in 2025
- 41% of all US clean energy added since 2014 was corporate-procured (CEBA)
- Long-term contracts (PPAs + utility) overtook unbundled RECs in 2023 (~46% of volume)

---

## Current Status (Feb 26, 2026)

### Optimal CFE Targets + No-Regrets Investments (Feb 26, 2026)

**Completed:**
- Created `scripts/step5_compute_optimal_targets.py` — Step 6 post-processor computing optimal CFE target range per ISO
- Smooth marginal MAC via PCHIP spline derivatives on isotonic-corrected cost/CO₂ curves
- 3 grid cost tiers × 3 DAC scenarios = 9 crossover points → range per ISO
- L/M/H demand growth scenarios (scale-invariant for MAC %, but affects absolute resource quantities)
- No-regrets resource investment analysis: floor, consensus, and average resource investments within the crossover range
- Uses canonical CO₂ model from `dispatch_utils.compute_fossil_retirement()` (same as step5_compute_co2, step5_compute_mac_stats)
- Output: `data/step5-post-processing/optimal_targets.json` + `dashboard/js/optimal-target-data.js`
- Wired into `step6_generate_shared_data.py` → OPTIMAL_TARGETS constant in shared-data.js
- Added to Step 6 GitHub Actions workflow (parallel batch with MAC, LMP, compressed day, etc.)
- Added scipy to workflow dependencies
- **Gas cost separation (Feb 26)**: MAC uses `effective_cost` (clean procurement only), NOT `total_system_cost` (which includes gas backup). Gas backup is a system reliability cost, not an abatement cost. See §7.4.1.
- Added `CLEAN_COST_DATA` extraction to step7 (P10/P50/P90 of effective_cost across scenarios)
- Fixed step5_consequential_deployment_queue.py MAC to exclude gas backup cost

**Completed (Feb 27):**
- Revamped `abatement_dashboard.html` with optimal target tiles, ISO deep-dive, and no-regrets analysis
- Created `scripts/step6_extract_no_regrets.py` — extracts no-regrets resources from Step 3 DG parquets (5,832+ scenarios)
- Revised DAC cost trajectories upward across all 4 files (anchored to 2025 actual costs of $600-$1,500/tCO₂)
- **Wired smooth PCHIP MAC into dashboard** — `abatement_dashboard.html` now uses `OPTIMAL_TARGETS` smooth curves from `optimal-target-data.js` when available, falls back to `MAC_STEPWISE_FAN` stepwise data otherwise
- Dashboard data priority chain: `OPTIMAL_TARGETS` (smooth PCHIP) > `MAC_STEPWISE_FAN` (stepwise fallback)
- No-regrets data priority chain: `NO_REGRETS_DATA` (DG parquets, 5,832+ scenarios) > `OPTIMAL_TARGETS.no_regrets` (medium-cost) > client-side `RESOURCE_MIX_DATA` analysis (fallback)
- Crossover computation prefers pre-computed smooth crossover range from `OPTIMAL_TARGETS`, falls back to client-side stepwise computation
- Created placeholder `dashboard/js/optimal-target-data.js` — populated by running `step5_compute_optimal_targets.py`

**Next steps:**
- [ ] Run Step 6 workflow to generate `optimal_targets.json` + `optimal-target-data.js` (smooth MAC data)
- [ ] Generate DG parquets for remaining ISOs (ERCOT, PJM, NYISO, NEISO, MISO, SPP) — currently only CAISO
- [ ] Re-run `step6_extract_no_regrets.py` after all ISO DG data is available
- [ ] Wire no-regrets investment data into research paper narrative
- [ ] Add prominent gas capacity warning to consequential scenario dashboard

### Scenario Comparison Page Fixes (Feb 26, 2026)

**Completed:**
- Fixed FOAK→NOAK learning curve chart (inline data constants, was referencing missing shared-data.js vars)
- Moved target slider inline with ISO selector in sticky bar
- Redesigned metric tiles with heat-map styling (green/amber/red) and prominent conclusions
- Replaced infinite MAC bars with ⚠ symbol when no emissions displaced

### Pipeline Reorganization — Step 5/6/7 Split (Feb 26, 2026)

**Completed:**
- Reorganized post-processing pipeline by dispatch cache dependency:
  - **Step 5**: Dispatch cache build + cache-independent scripts (EAC scarcity, track export, track analysis)
  - **Step 6**: Dispatch-cache-dependent scripts (compressed day, CO2, MAC, LMP, consequential, scenarios)
  - **Step 7**: Generate shared data (dashboard aggregation)
- Refactored PP2 (consequential queue) to use dispatch cache for hourly emission accounting via `compute_co2_from_dispatch()`
- Refactored PP3 (scenario comparison) — both Scenario A and Scenario B now use dispatch-cache-based emission accounting
- Added `compute_co2_from_dispatch()` to `dispatch_utils.py` — shared hourly emission function using merit-order fuel displacement
- Integrated Track 2&3 workflow into Step 3 with track selector dropdown (Track 1 baseline / Track 2 NB / Track 3 CTR)
- Added `--track` flag to `step3_track_nb_ctr.py` (nb/ctr/both)
- Renamed all scripts from PP-numbered to step-numbered (see Pipeline Architecture below)
- Updated all internal references, docstrings, imports, workflow files

**Script renames:**
| Old name | New name |
|----------|----------|
| `step5_PP0_build_dispatch_cache.py` | `step4_build_dispatch_cache.py` |
| `step5_PP1_compressed_day.py` | `step5_compress_day_profiles.py` |
| `step5_PP2_consequential_queue.py` | `step5_consequential_deployment_queue.py` |
| `step6_scenario_comparison.py` | `step6_scenario_comparison.py` |
| `step5_compute_co2.py` | `step5_compute_co2.py` |
| `step5_compute_mac_stats.py` | `step5_compute_mac_stats.py` |
| `step5_compute_lmp_prices.py` | `step5_compute_lmp_prices.py` |
| `step5_PP7_compute_eac_scarcity.py` | `step5_compute_eac_scarcity.py` |
| `step5_PP8_export_track_results.py` | `step4_export_track_results.py` |
| `step5_PP9_analyze_tracks.py` | `step4_analyze_tracks.py` |
| `step6_generate_shared_data.py` | `step6_generate_shared_data.py` |

**Pipeline execution order:**
```
Step 4 → Step 5: dispatch cache build → Step 5: cache-independent (EAC, tracks) in parallel
                                       → Step 6: cache-dependent (CO2, MAC, LMP, compressed day, consequential, scenarios) in parallel
                                                → Step 7: generate shared data
```

### Pipeline Audit & Dispatch Consolidation (Feb 26, 2026)

**Completed:**
- Created `step4_build_dispatch_cache.py` — pre-computes 8,760-hour dispatch for all unique mixes across all ISOs. Versioned NPZ cache (v2) with per-resource matched/surplus + charge profiles.
- Extended `dispatch_utils.py` with `detailed=True` mode on `reconstruct_hourly_dispatch()`: adds per-resource merit-order breakdown (CF→CCS→hydro→wind→solar) and storage charge tracking.
- Added `DISPATCH_ORDER`, `CACHE_VERSION`, `_compute_per_resource_dispatch()`, `_battery_loop_detailed()`, `_ldes_loop_detailed()` to dispatch_utils.
- Fixed PP4 supply profile bug: was importing `get_supply_profiles_simple` (flat clean_firm, no DST correction) instead of canonical `get_supply_profiles`. CO₂ results on the dispatch path will now use correct nuclear seasonal derate + DST-corrected solar.
- Refactored `step5_compress_day_profiles.py`: removed ~170 lines of duplicate dispatch engine (battery/LDES loops, supply profiles, data loading). Now imports from dispatch_utils and reads from dispatch cache.
- Added cache-miss warning to `step5_compute_lmp_prices.py`.
- Removed dead code: `get_supply_profiles_simple` from dispatch_utils.
- Added version metadata support to `load/save_dispatch_cache()`.
- Updated all documentation: SPEC.md, CLAUDE.md, README.md, pipeline.html, optimizer_methodology.html.
- Added MISO and SPP ISO support across pipeline and dashboard.

**Pipeline execution order (legacy reference):**
```
Step 4 → dispatch cache → CO2/MAC/LMP/compressed day (read from cache, run in parallel)
```

### Step 1 PFS Rebuild — Two-Phase Adaptive Storage Sweep (Feb 21, 2026)

**Completed:**
- Energy-based battery caps (replaces power-based formula): `bat4_cap = max_daily_surplus`, `bat8_cap = max_2day_surplus`, `ldes_cap = max_7day_surplus`
- Two-phase adaptive storage sweep: Phase 1 coarse (0.25% steps) → analyze saturation → Phase 2 fine (0.05% steps) within saturation range
- Per-ISO/threshold output files: `data/step1-pfs-parquets/{ISO}_t{XX}_raw_pfs.parquet`
- ERCOT: 2,033,961 solutions (complete)
- CAISO: Running (Phase 2 in progress)

**In Progress:**
- CAISO two-phase run
- PJM, NYISO, NEISO queued

**Per-ISO Results:**
| ISO | Solutions | Phase 1 (coarse) | Phase 2 (fine) | bat4 sat | bat8 sat | Time |
|---|---|---|---|---|---|---|
| ERCOT | 2,033,961 | 226K | 1.8M | 0.75% | 1.50% | 11 min |
| CAISO | (running) | — | — | 1.00% | 2.00% | ~25 min |
| PJM | (queued) | — | — | — | — | — |
| NYISO | (queued) | — | — | — | — | — |
| NEISO | (queued) | — | — | — | — | — |

**Branch:** `claude/validate-battery-physics-bqZTr`

### LMP Module — v9 Calibration Complete (Feb 21, 2026)

**Completed:**
- `dispatch_utils.py` — shared dispatch module (constants, profiles, battery/LDES dispatch, fossil retirement, hourly dispatch cache)
- `step5_compute_lmp_prices.py` — core LMP engine with:
  - Merit-order fossil stack: PJM Manual 15 cost-based offer formula (HR × fuel + VOM + CO2 + 10% adder)
  - Heat rates calibrated to PJM SOM 2024 benchmarks (CCGT 7.0, CT 10.5, coal 10.0 MMBtu/MWh)
  - CO2 allowance costs (RGGI-weighted: L/M/H = $3/$5.50/$14 per ton)
  - 10% adder per PJM market rules ($2.00/MWh contribution per SOM 2024)
  - VOM split: variable maintenance + variable operations (SOM 2024: $3.18 + $1.43 = $4.61 fleet avg)
  - Load-dependent heat rate ramp (15% quadratic) for within-band price variation
  - Demand-quantile pricing: congestion, scarcity tail, off-peak compression
  - ISO-specific price formation: PJM (RPM), ERCOT (ORDC), CAISO (RA), NYISO (ICAP), NEISO (FCM + winter gas)
  - Archetype deduplication: (mix, fuel_level, threshold) → ~7,800 unique per ISO
  - Dispatch cache: append-mode NPZ per ISO, shared with step5_compute_co2.py
- `calibrate_lmp_model.py` — validation framework with embedded PJM IMM/EIA reference data
- `step5_compute_co2.py` — refactored to import from dispatch_utils.py (identical behavior)

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
- ERCOT: NB partial (10/21 thresholds), replace not started
- PJM, NYISO, NEISO: Not started
- Checkpoint: `data/track_checkpoint.json` (partial results)
- Parquet export: `dashboard/track_scenarios.parquet` (CAISO only)

**All-ISO Price Models (Added Feb 28):**
- All 7 ISOs now have calibrated price model classes (PJMPriceModel through SPPPriceModel)
- GAF (Gas Availability Factor) added for MISO (0.83) and SPP (0.84)
- `calibrate_lmp_model.py` updated with ISO_CALIBRATION_TARGETS for all 7 ISOs
- `--iso ALL` support in both `step5_compute_lmp_prices.py` and `calibrate_lmp_model.py`
- 2024 SOM calibration targets (avg LMP, percentiles, negative hours, scarcity):
  - SPP: $26/MWh (cheapest), ERCOT: $26, MISO: $31, PJM: $34.7, CAISO: $38, NEISO: $39.5, NYISO: $42

**Next Steps:**
- Fetch actual 2024 EIA hourly data for all ISOs and run calibration validation
- Tune demand-quantile parameters per ISO to match actual LMP distributions
- Run LMP model on full all-ISO ECF scenarios (all thresholds × fuel sensitivities)

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

#### Decision 4: Existing Clean = $0, New-Build = LCOE (Updated Feb 25, 2026)
- **Approach**: Existing clean generation priced at **$0** (sunk fleet — already built and operating, no cost to buyer). New-build priced at LCOE + transmission adder. Previous wholesale pricing of existing resources was distorting the optimizer toward wind-heavy mixes that underutilized the existing fleet.
- **Rationale**: For Track 1 baseline analysis, existing clean resources are constants on the grid. Charging them at wholesale ($25-42/MWh) penalized mixes that used more existing, causing the optimizer to prefer over-procuring cheap new-build wind while leaving free existing nuclear/solar unused.
- **Effect**: Optimizer now strongly prefers mixes that maximize use of existing fleet (free) before adding new-build. ERCOT 50% at 2030 now correctly shows ~13% CF / 22% solar / 65% wind at 55% procurement (using ~42 TWh of existing nuclear) instead of 5% CF / 86% wind at 65% procurement.
- **Implementation**: `coeff_matrix[:, _COL_WHOLESALE] = 0` in coefficient model; `total_cost += 0` for existing in price_mix_batch.

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

#### Decision 5: Three Analysis Tracks (5C — Updated Feb 24, 2026)
Three distinct tracks with standardized naming. Each track represents a different set of assumptions about what existing clean energy is credited vs. what must be procured new.

**Track 1 — ECF (Existing Clean Floor)**: Baseline case
- Cost optimization built on **2025 absolute existing clean generation** — the "existing clean floor"
- Existing clean TWh remain **constant** across all demand growth scenarios (absolute TWh steady; share of total generation declines over time as demand grows)
- All existing resources priced at **$0** (sunk fleet — no cost to buyer)
- New-build procurement optimized on top of the existing floor to hit each target threshold
- Source: `overprocure_scenarios.parquet` (baseline results)
- LMP module runs on this track
- Files/caches use `ecf_` prefix

**Track 2 — NB (New-Build / Greenfield)**: What hourly matching incentivizes to build
- **Ignores ALL existing clean energy** — no existing solar, wind, nuclear, CCS, or hydro credited
- Hydro: **excluded** (hydro=0 mixes only, not considered at all)
- Uprates: **on** (uprate tranche active — cheapest new-build option, part of greenfield procurement)
- All resources must be procured as new-build (except uprates which are allowed)
- Purpose: What does hourly matching incentivize you to BUILD from scratch?
- Files/caches use `nb_` prefix

**Track 3 — CTR (Cost to Replace Clean Firm)**: What it costs to replace existing dispatchable clean
- **Does NOT include** existing clean firm (nuclear) or uprates — these are what's being "replaced"
- **Does NOT include** existing CCS (near-zero in most ISOs, but conceptually part of what's replaced)
- Uprates: **off** (uprate_cap=0, no uprate tranche)
- Hydro: **included** (existing floor, $0 — sunk fleet)
- Existing solar: **included** (existing floor, $0 — sunk fleet)
- Existing wind: **included** (existing floor, $0 — sunk fleet)
- Purpose: True cost of replacing existing dispatchable clean generation (nuclear/CCS), while keeping existing renewables and hydro as the floor
- Files/caches use `ctr_` prefix

**Naming convention**: All file names, cache keys, code comments, and output fields use ECF/NB/CTR abbreviations consistently. File rename deferred until NB/CTR sweep completes.

**Output**: Data files only. No research paper update yet — discuss findings with user first, then write.
**Architecture**: Step 3 runs 2 additional passes per (ISO, threshold):
  - Pass "NB": filter EF to hydro=0, zero all existing clean, uprate tranche on → pure greenfield results
  - Pass "CTR": keep hydro + solar + wind floors, zero clean firm + CCS existing, uprate tranche off → replacement cost results
  - Plus existing pass "ECF": full EF, all features on → current behavior (preserved)

#### Decision 5b: Track-to-Page Assignment (Feb 24, 2026)
Each dashboard page uses a specific track (or combination) for its data and visualizations:

**Track 1 (ECF) exclusively:**
| Page | File | Notes |
|---|---|---|
| Home | `index.html` | Scrollytelling narrative, key findings |
| Grid Simulation | `dashboard.html` | Interactive optimizer with all sensitivity toggles |
| CO₂ Abatement | `abatement_dashboard.html` | MAC curves, abatement ladders |
| Wholesale Price Assessment | `lmp_trends.html` | **PJM only** — LMP trends and price impact |
| Fossil Fuel Deep Dive | `fossil_fuel_deepdive.html` | Fossil retirement, fuel switching |

**Track 2 (NB) + Track 3 (CTR) compared against Track 1 (ECF):**
| Page | File | Notes |
|---|---|---|
| Cost to Replace | `cost_to_replace.html` | CTR vs ECF — replacement cost of dispatchable clean |
| New Build Analysis | `new_build_analysis.html` | NB vs ECF — procurement strategy, EAC scarcity scaling, LMP feedback loop (**PJM only**) |

#### Decision 5e: New Build Analysis Page Design (Feb 25, 2026)

**Page**: `new_build_analysis.html` — "The Build-or-Buy Decision"
**Scope**: PJM only. Scrollytell hybrid (Sections 1–2 interactive, Section 3 narrative scroll).

**Section 1: The Procurement Choice** (interactive)
- Target slider: range input snapping to 11 PJM thresholds (50–92.5%)
- SBTi year label displayed alongside slider value
- Side-by-side comparison: Track 1 (ECF, "With Existing Clean") vs Track 2 (NB, "New Build Only")
- Each side shows: effective cost ($/MWh), resource mix stacked bar, P10/P50/P90 sweep bands
- EAC premium sliders with dynamic defaults:
  - Nuclear: auto = max(0, $33/MWh operating cost + 10% margin − LMP(threshold))
  - Hydro: $3/MWh default
  - Existing solar/wind: $5/MWh default (REC market proxy)
- New-build resources priced at LCOE; existing resources at wholesale + EAC premium
- "Additionality Premium" callout: $/MWh delta between tracks

**Section 2: The Scaling Curve** (interactive)
- X-axis: % of C&I load participating in hourly CFE (0–100%)
- Y-axis: effective cost $/MWh
- Two curves: "With Existing" (Track 1) and "New Build Only" (Track 2) converging
- PJM supply stack: 280 TWh total clean → 95 SSS-fixed → 70 RPS → 25 existing PPAs → ~90 TWh available
- Vertical marker at ~90 TWh showing where existing merchant supply exhausts
- L/M/H cost sensitivity bands (P10/P50/P90)
- As participation scales, existing supply exhausts → Track A cost rises toward Track B

**Section 3: The Economic Feedback Loop** (scrollytell narrative)
- Step 1: LMP decline chart — PJM LMP from $36.86 (50%) to $6.80 (99%), with nuclear operating cost line ($33/MWh)
- Step 2: The widening gap IS the clean premium — above ~80% threshold, LMP < nuclear cost
- Step 3: PJM clean supply waterfall — 280 TWh total → decomposition
- Step 4: IL ZEC/CMC 2027 expiry callout — 94 TWh at risk of retirement
- Step 5: 45U PTC evidence — $15/MWh production tax credit as bridge, but expires; without policy backstop, LMP decline threatens existing nuclear viability
- Step 6: The paradox — new-build-only procurement accelerates LMP decline → existing nuclear uneconomic → retirements → more new build needed at higher cost

**Section 4: Bottom Line** (static summary)
- Cost delta summary at key thresholds
- Policy implication: procurement strategy must account for systemic feedback effects
- Clean premiums as market signal for existing nuclear viability

**Data sources**:
- `track_results.json` (Track 2 NB + Track 3 CTR vs baseline)
- `shared-data.js` (ECF baseline: EFFECTIVE_COST_DATA, RESOURCE_MIX_DATA)
- `lmp_summary.json` (PJM LMP trajectory by threshold)
- EAC scarcity parameters inline (from step5_compute_eac_scarcity constants)
- Nuclear operating cost: $33/MWh (fuel ~$5.5 + fixed O&M ~$25 + VOM ~$2.5)

#### Decision 5c: Visual Differentiation — Existing vs New-Build vs Curtailment (Feb 24, 2026)
All resource mix figures across the entire site must visually differentiate existing vs new-build generation:

- **Existing resources**: Full saturation fill (solid, opaque colors from the standard palette)
- **New-build resources**: Transparent fill with full-saturation outline (same hue, reduced opacity interior, solid border)
- **Curtailment**: Transparent fill with diagonal cross-hatching (curtailment is almost always from new-build)

This applies to ALL charts showing resource mixes (stacked bars, stacked areas, waterfall, etc.) across all pages and all ISOs.

#### Decision 5d: Resource Differentiation & Color Palette (Feb 24, 2026)
All resource mix figures must differentiate ALL individual resources (not aggregate categories). The following resources must each have a distinct, consistent color across the entire site and all ISOs:

| Resource | Color | Hex | Notes |
|---|---|---|---|
| Nuclear | Indigo | `#6366F1` | Existing dispatchable clean (updated Mar 3, 2026 from #1E3A5F) |
| CCS-CCGT | Slate | `#64748B` | Carbon capture gas (updated Mar 3, 2026 from #0D9488) |
| Geothermal | Ochre/Brown | `#B45309` | CAISO only |
| Offshore Wind | Teal | `#009688` | Atlantic ISOs + CAISO (added Mar 3, 2026) |
| Wind (Onshore) | Green | `#22C55E` | Onshore wind |
| Solar | Amber | `#F59E0B` | Utility-scale PV |
| Hydro | Cyan | `#0EA5E9` | Existing only, wholesale-priced |
| LDES (100hr) | Pink | `#EC4899` | Iron-air, 50% RTE |
| Battery (4hr) | Purple | `#8B5CF6` | Li-ion, 85% RTE |
| Battery (8hr) | Light Purple | `#A78BFA` | Li-ion extended duration |
| Green H₂ | Emerald | `#10B981` | 1000hr, 35% RTE, ≥95% only |
| Curtailment | Gray + cross-hatch | `#D1D5DB` | Diagonal hatching pattern |

**Display order** (updated Mar 3, 2026): Nuclear → Geothermal → Hydro → CCS → Offshore Wind → Onshore Wind → Solar → Battery 4 → Battery 8 → LDES → H2

**Bar chart styling**: ALL bar charts across the entire site must have **rounded corners** (`borderRadius` in Chart.js).

**Consistency rule**: These colors and styling rules apply to every page, every ISO, every chart type. No page-specific overrides unless explicitly approved.

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
- `step6_generate_shared_data.py` — reads both columnar (new) and row (legacy) formats
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
| 3 | Procurement bounds | **3C — Threshold-adaptive** | Narrow bounds at low thresholds (e.g., 100-110% at 50%), wider at high (100-150% at 99-≥99.99%). |
| 4 | min_dispatchable constraint | **4B — Drop it** | No dispatchable floor. Let physics prove/disprove — constraint was potentially biasing results. |
| 5 | Thresholds | **5F — 21 total** | 10/20/30/40 (coarse only) + 50/55/60/65/70/75/80/85/87.5/90/92.5/95/97.5/99/99.5/99.9/≥99.99 (17 active, full pipeline). Top threshold is ≥99.99% (not 100%) — true 100% hourly matching is physically unreachable. Thresholds 10–40 are coarse-grid only (no fine zone search, no step1d storage). |
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

**Scope**: Step 1 only (physics). No cost model — the optimizer generates the feasible solution space (all viable resource mixes per threshold×ISO). Cost sensitivities (5,832 paired-toggle scenarios) applied in Step 3 cost optimization. This reduces from 25,872 cost-coupled optimizations to 147 physics-only sweeps (21 thresholds × 7 ISOs), each finding the Pareto frontier of feasible mixes.

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

### 7-Step Pipeline Architecture

The optimizer runs as a 7-step pipeline. Step 1 is expensive (hours). Steps 2–4 are cheap (seconds to minutes). Step 5 builds a dispatch cache. Step 6 runs parallel analytics on 10+ scripts. Step 7 exports dashboard data.

#### Core Pipeline (Steps 1–4)

| Step | Script(s) | What It Does | When to Re-run |
|------|-----------|-------------|---------------|
| **Step 0** | `step0_*.py` (8 scripts) | **Data Fetch/Prep** — EIA hourly profiles (multi-year 2021–2025), eGRID emissions, actual LMP data, DST/UTC fixes, MISO/SPP consolidation. | When source data updates. |
| **Step 1** | `step1_pfs_generator.py` (monolithic) or `step1a` → `step1b` → `step1c` → `step1d` (modular) | **PFS Generator** — 4D/5D adaptive grid search × procurement × storage. Two-phase storage sweep (coarse → fine). Output: `data/step1-pfs-parquets/` + `data/step1d-storage-parquets/`. | Only if dispatch logic, generation profiles, or demand curves change. |
| **Step 2** | `step2_efficient_frontier.py` + `step2_5_expand_ef_for_floors.py` | **Efficient Frontier** — Extracts non-dominated mixes from PFS. Reads both step1 and step1d parquets. Optional EF expansion for Scenario A floor constraints. Output: `data/step2-ef-parquets/`. | Only if PFS or filtering criteria change. |
| **Step 3** | `step3_cost_optimization.py` + `step3_track_nb_ctr.py` | **Cost Optimization** — Track 1 baseline: 5,832 sensitivity combos (17,496 CAISO). Track 2 (NB) + Track 3 (CTR). Merit-order tranche pricing. Demand growth with FOAK→NOAK learning curves. Output: `data/step3-cost-opt-parquets/`. | When cost assumptions, LCOE tables, or toggles change. |
| **Step 4** | `step4_gas_ccs_adjustement.py` | **Gas/CCS Adjustments** — NEISO winter gas constraint (+$13.13/MWh CCS adder), 45Q correction ($27.5/MWh), gas capacity backup & RA (15% margin), CCS vs LDES crossover. Output: `data/step4-gas-ccs-parquets/`. | When Step 3 outputs change. |

#### Step 1 Sub-Pipeline (Modular, for CI/CD)

| Script | What It Does |
|--------|-------------|
| `step1a_generate_mixes.py` | Generates all resource fraction combos (4D/5D grid). |
| `step1b_score_mixes.py` | Scores mixes against hourly demand → `coarse_cache.parquet`. |
| `step1c_build_pfs.py` | Mines PFS from scored DB per threshold. |
| `step1d_storage_refinement.py` | Fills storage exploration gaps (bat4 0.05–3%, bat8 0.1–4%, LDES 0.25–10%). Output: `data/step1d-storage-parquets/`. |

#### Step 5: Dispatch Cache + Independent Analysis

| Script | What It Does | Cache Dep? |
|--------|-------------|-----------|
| `step4_build_dispatch_cache.py` | **Run first.** Pre-computes 8,760-hour dispatch for all unique mixes. Versioned NPZ cache (v2) with per-resource matched/surplus + charge profiles. | Creates cache |
| `step4_export_track_results.py` | Exports track parquets (NB + CTR) to `track_results.json`. | None |
| `step4_analyze_tracks.py` | Track cost envelopes (P10/P50/P90), resource mix differentials. | None |

#### Step 6: Dispatch-Cache-Dependent Analysis (10 scripts, run in parallel)

| Script | What It Does |
|--------|-------------|
| `step5_compute_co2.py` | **CO₂ dispatch-stack model.** Merit-order fuel retirement (coal → oil → gas). Coal/oil capped at 2025 TWh. Demand-growth-aware. **Run before MAC stats.** |
| `step5_compute_mac_stats.py` | **MAC statistics.** 6 metrics: average fan (P10/P50/P90), stepwise marginal, monotonic envelope, path-constrained. ANOVA decomposition. Crossover vs DAC/SCC/ETS. |
| `step5_compute_lmp_prices.py` | **Synthetic LMP.** 8,760-hour dispatch; hourly LMP from merit-order fossil stack. All 7 ISOs with calibrated price models. |
| `step5_compute_optimal_targets.py` | **Optimal CFE targets.** Marginal MAC × DAC crossover via PCHIP spline. 3×3 grid-cost × DAC-scenario matrix. No-regrets resource analysis. |
| `step5_compress_day_profiles.py` | **Compressed day profiles.** 24-hour representative day from dispatch cache. Falls back to live compute on cache miss. |
| `step5_consequential_deployment_queue.py` | **Consequential queue.** Cross-regional deployment path under consequential accounting. Hourly emission accounting via dispatch cache. |
| `step5_scenario_a_consequential.py` | **Scenario A.** Forward-stepping consequential procurement with per-resource floor ratchets. PFS fallback on filter exhaustion. |
| `step5_scenario_b_hourly.py` | **Scenario B.** Hourly matching procurement strategy. |
| `step5_scenario_comparison.py` | **Scenario comparison.** Consequential vs. hourly matching — cost, emissions, resource mix differentials. |
| `step5_analyze_storage.py` | **Storage analysis.** Battery/LDES utilization, dispatch patterns, capacity factor analysis. |

#### Step 6.5: Corporate Procurement Strategy Simulation

| Script | What It Does |
|--------|-------------|
| `step5_5_procurement_utils.py` | **Shared utilities.** SSS allocation, EAC pricing, LMP feedback, PPA premiums, learning curves, 25-year timeline. |
| `step5_5_strategy1_consequential.py` | **Strategy 1 (A/B/C).** Cross-regional consequential netting under 3 emission baselines. |
| `step5_5_strategy2_hourly.py` | **Strategy 2 (A/B/C).** Hourly matching same-ISO with existing clean credit variants. |
| `step5_5_strategy3_annual.py` | **Strategy 3 (A/B/C/D).** Annual matching 2×2 matrix (new-only vs all-clean × additionality). |

#### Step 7: Dashboard Data Generation

| Script | What It Does |
|--------|-------------|
| `step6_generate_shared_data.py` | **Main export.** All results → `dashboard/js/shared-data.js`. SBTi mapping, DAC projections, LCOE tables for client-side repricing. Runs last. |
| `step6_extract_no_regrets.py` | **No-regrets analysis.** Optimal targets and no-regrets resource investments from crossover analysis. |

#### Shared Utility Modules

| Script | What It Does |
|--------|-------------|
| `dispatch_utils.py` | **Dispatch engine.** Single source of truth for dispatch reconstruction (`reconstruct_hourly_dispatch(detailed=True)`), supply profiles (`get_supply_profiles()`), fossil retirement, cache I/O (v2 NPZ). |
| `scenario_common.py` | **Scenario utilities.** Shared Scenario A/B logic: cost tables, demand growth, learning curves, EF/PFS loading. |
| `eia_data_io.py` | **EIA data I/O.** Standardized multi-year EIA generation/demand profile loading. |
| `calibrate_lmp_model.py` | **LMP calibration.** Validates synthetic LMP against actual ISO data (2024 SOM reports). |

#### Pipeline Execution Order

```
Step 0 (data fetch, optional)
  → Step 1a → 1b → 1c → 1d (PFS generation)
    → Step 2 (efficient frontier)
      → Step 3 (cost optimization, 3 tracks)
        → Step 4 (gas/CCS adjustments)
          → Step 5 (dispatch cache first → track export/analysis in parallel)
            → Step 6.0: CO₂ recompute (run before 6.1)
              → Steps 6.1–6.6 (cache-dependent analytics, run in parallel)
                → Step 7 (dashboard data export)
```

**Key principle**: Step 1 is expensive (hours). Step 2 ~40s. Steps 3–4 cheap (minutes). Changing cost assumptions → Steps 3–4 + post-processing only. Changing a single analysis → relevant Step 6 script + Step 7.

#### GitHub Actions Workflows

All workflows: `workflow_dispatch`, script selectors, ISO selectors. See `.github/workflows/README.md` for full docs.

| # | Workflow | What It Does |
|---|----------|-------------|
| 0 | `step0-fetch-lmp-data.yml` | Fetch actual DA hourly LMP |
| 1a | `step1a-scored-database.yml` | Generate + score resource combos |
| 1b | `step1b-build-pfs.yml` | Mine PFS per threshold |
| 1d | `step1d-storage-refinement.yml` | Fill storage gaps |
| 2 | `step2-efficient-frontier.yml` | EF filter + expansion |
| 3 | `step3-cost-optimization.yml` | Cost optimization (3 tracks) |
| 4 | `step4-gas-ccs.yml` | Gas/CCS post-processing |
| 5 | `step4-dispatch-cache.yml` | Dispatch cache + track analysis |
| 6.0–6.6 | `step6.X-*.yml` | Page-oriented analytics |
| 7 | `step6-generate-shared-data.yml` | Dashboard data export |

**Key acronyms**:
- **PFS** — Physics Feasible Space: all physically valid resource mixes (Step 1 output, `data/step1-pfs-parquets/`)
- **EF** — Efficient Frontier: non-dominated mixes optimal under some cost assumption (Step 2 output, `data/step2-ef-parquets/`)

**Data contract**: Step 3 must NOT change existing columns in shared-data.js or overprocure_results.json. Add new columns/fields only. This prevents recoding existing figures and dashboards.

### What was accomplished
- [x] Homepage (`index.html`) — 4 charts rendering with real data, region toggle pills, narrative sections
- [x] Carbon Abatement Dashboard (`abatement_dashboard.html`) — 3 charts (MAC, portfolio, ladder) fully rendering with hardcoded illustrative data + 4 stress-test toggles
- [x] Navigation site-wide: Home | Cost Optimizer | Analysis (CO₂ Abatement Analysis) | Research (Paper, Methodology, Policy, About)
- [x] "Back to Home" button on all non-home pages
- [x] Chart styling QA/QC on working charts (borderRadius 6, no grid lines, axis borders)
- [x] Merged methodology into research paper (Appendix B with 7 sub-sections)
- [x] Tagline: "Most climate solutions depend on" across all pages
- [x] **CO₂ methodology fixed**: Hourly fossil-fuel emission rates (eGRID per-fuel × EIA hourly mix) replacing flat rate
- [x] **Post-optimizer pipeline**: `step5_compute_co2.py`, `analyze_results.py`, `run_post_optimizer.sh`
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

**Zone 2 — Last Mile (90% → ≥99.99%): Granular checkpoints with enforced monotonicity**
- 5 stepwise values: 90→92.5%, 92.5→95%, 95→97.5%, 97.5→99%, 99→≥99.99%
- Enforced non-decreasing: `step_mac[t] = max(raw_step_mac[t], step_mac[t-1])`
- Zone 1 aggregate MAC serves as floor for first Zone 2 step
- Convex hull interpolation for edge cases where ΔCO2 ≤ 0

**Result**: 6-value marginal MAC curve per (ISO, scenario):
```
[MAC_75→90, MAC_90→92.5, MAC_92.5→95, MAC_95→97.5, MAC_97.5→99, MAC_99→≥99.99]
```

**Fan chart fix**: Consistent scenario ranking (rank by total cost at 99%, select P10/P50/P90 scenarios, use their full curves) instead of independent per-step percentiles that mix different scenarios.

**Implementation plan**: See `PLAN_marginal_mac_fix.md` for detailed implementation steps and file-by-file changes.

### MAC Formula Change (Feb 25, 2026) — Full Portfolio LCOE, Not Incremental Above Wholesale

**Previous formula**: `MAC = (cost_incremental × demand) / CO₂_abated` where `cost_incremental = effective_cost - wholesale_price`. This measured the premium of clean energy over grid wholesale power per ton abated.

**New formula**: `MAC = (cost_effective_cost × demand) / CO₂_abated`. Uses the full portfolio LCOE (effective cost) rather than the incremental cost above wholesale.

**Rationale**:
- MAC should measure the standalone cost-effectiveness of the clean portfolio per ton of CO₂ displaced, not a premium relative to a wholesale baseline
- Removes wholesale price sensitivity from MAC (previously, higher wholesale prices made MAC look artificially low)
- Aligns with standard MAC curve methodology: cost of the abatement action itself
- CO₂ abatement continues to use merit-order fossil retirement (coal → oil → gas) from the dispatch model in `dispatch_utils.py`

**Impact**: MAC values will be higher across the board (roughly 2× for regions where wholesale ≈ effective_cost/2). DAC/SCC/ETS benchmark comparisons remain unchanged. Crossover thresholds will shift. Dashboard narrative text must be updated after PP5 re-run.

**Files changed**: `scripts/step5_compute_mac_stats.py` — all MAC computation functions (`add_mac_column`, `compute_stepwise_fan`, `compute_monotonic_envelope`, `compute_path_constrained_mac`, demand-growth MAC) now use `cost_effective_cost` instead of `cost_incremental`. No changes to Steps 1-4 or parquet outputs needed — only PP5 re-run required.

### FOAK→NOAK Learning Curve — Scenario-Differentiated, Year-Based (Feb 25, 2026)

**Decision**: Replaced single learning curve with scenario-differentiated, year-based curves. Both scenarios now have learning curves, but with different start dates and durations reflecting deployment pace differences.

**Rationale**: INL data shows SOAK (2nd unit) achieves 15% reduction, units 2-4 another 5%, before gradual decline toward NOAK. NEA shows -18 to -25% for unit 2, -25 to -40% by unit 4, -35 to -55% by unit 8. DOE Liftoff projects NOAK by early 2030s for advanced designs, scaled achievement by 2035, full low-price stabilization by 2040. Learning occurs sequentially and in compressed timelines — construction learning transfers through engineering teams, supply chains, and regulatory streamlining before each unit completes. Westinghouse plans 10 new AP1000s by 2030; commercial SMRs expected operational by 2030.

**Scenario B (Hourly Matching — aggressive deployment):**
- FOAK starts: 2029 (first commercial SMR deployments)
- Learning period: 2028-2038 (10 years)
- NOAK achieved: 2038
- 2038-2050: Stable at NOAK (Low) pricing
- Shape: Concave (exponent 0.6) — steep front-end matching INL/NEA unit-doubling data, asymptotic tail
- Pre-2029: Pure FOAK (no learning, no SMR deployment yet)

**Scenario A (Pure Consequential — delayed deployment):**
- FOAK starts: 2036 (5-year delay due to less investment, slower regulatory pathway)
- Learning period: 2036-2048 (12 years — stretched due to fewer units built per year)
- NOAK achieved: 2048
- 2048-2050: Stable at NOAK (Low) pricing
- Shape: Concave (exponent 0.6) — same steep front-end physics, stretched across 12 years
- Pre-2036: Pure FOAK

**Implementation**: `learning_fraction(threshold, scenario='B')` now uses year-based lookup via SBTI_YEAR_MAP. Each scenario defines `foak_start_year` and `noak_year`. Fraction is 0 (pure FOAK) before start, concave ramp during learning period, 1.0 (full NOAK) after NOAK year. Both `_adjust_costs_with_learning` (Scenario B) and `_adjust_costs_no_learning` → `_adjust_costs_delayed_learning` (Scenario A) use the same interpolation machinery with different timeline parameters.

**Nuclear LCOE trajectory (PJM, H=$160 → L=$72):**

Scenario B:
| Year | Threshold | Fraction | LCOE |
|------|-----------|----------|------|
| 2030 | 50% | 0.38 | $126 |
| 2031 | 55% | 0.49 | $117 |
| 2033 | 60% | 0.66 | $102 |
| 2035 | 70% | 0.81 | $89 |
| 2037 | 80% | 0.94 | $77 |
| 2038 | 85% | 1.00 | $72 |
| 2040 | 90% | 1.00 | $72 |
| 2045 | 95% | 1.00 | $72 |
| 2050 | 100% | 1.00 | $72 |

Scenario A:
| Year | Threshold | Fraction | LCOE |
|------|-----------|----------|------|
| 2030 | 50% | 0.00 | $160 |
| 2035 | 70% | 0.00 | $160 |
| 2036 | 75% | 0.00 | $160 |
| 2037 | 80% | 0.23 | $140 |
| 2038 | 85% | 0.34 | $130 |
| 2040 | 90% | 0.52 | $114 |
| 2045 | 95% | 0.84 | $86 |
| 2048 | 97.5% | 1.00 | $72 |
| 2050 | 100% | 1.00 | $72 |

**Supersedes**: Previous single learning curve (concave ramp, no learning below 70%, same curve for both scenarios). Scenario A previously had NO learning curve (flat FOAK forever).

**Scope**: PP3 scenario comparison only. Step 3 cost optimization is NOT modified — it remains the 2025 snapshot with static LCOE tables.

**Files changed**: `scripts/step6_scenario_comparison.py` — `learning_fraction()`, `_adjust_costs_no_learning()` → `_adjust_costs_delayed_learning()`. PP3 re-run required.

### Compressed Day Chart — Curtailment Double-Count Fix (Feb 25, 2026)

**Decision**: Fix the compressed day chart to:
1. Show "Total Generation" line (total clean energy output before curtailment) instead of "Demand" line
2. Net out storage charging from displayed surplus to eliminate double-count

**Problem**: Per-resource surplus arrays included energy absorbed by storage (battery/LDES charging). That same energy was also shown as negative charging bars below the x-axis — double-counting the same energy.

**Fix**:
- Compute net curtailment factor: `(grossSurplus - totalCharging) / grossSurplus` per hour
- Apply proportionally to each resource's surplus so only TRUE curtailment (not stored) is shown as hatched area
- Replace "Demand" line with "Total Generation" = sum of primary resource matched + gross surplus (total clean output before curtailment/storage)

**Files changed**: `dashboard/dashboard.html` — compressed day chart rendering. No PP1 changes needed.

### Annual Cost + 25-Year NPV Metrics (Feb 25, 2026)

**Decision**: Add `annual_cost_billion` ($B/yr) and `npv_25yr_billion` ($B, 25-year NPV) to PP3 trajectory entries alongside existing $/MWh metrics.

**Formulas**:
- `annual_cost_billion = effective_cost ($/MWh) × demand_twh / 1000`
- `npv_25yr_billion = annual_cost × annuity_factor(5%, 25)` where annuity factor ≈ 14.09
- Uses 5% real WACC (standard utility/infrastructure discount rate)

**Rationale**: $/MWh is useful for comparison but doesn't convey scale. Annual $B shows the absolute commitment at each threshold. 25-year NPV shows the total investment required, useful for investment framing and policy cost-benefit analysis.

**Files changed**: `scripts/step6_scenario_comparison.py` — trajectory entry construction. PP3 re-run required.

### Gas Capacity Costs — Already Integrated (Feb 25, 2026)

**Status**: Gas backup capacity costs (both existing FOM and new CCGT build) are already fully integrated into total system cost in PP3's `compute_mix_cost()`. New CCGT costs range from $88-114/kW-yr by ISO. No additional changes needed.

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
1. Run `step5_compute_co2.py` → hourly CO₂ correction
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
- 75%→0.80×, 90%→1.05×, ≥99.99%→1.45×
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

**Pipeline position**: Downstream of Step 4. Reads Step 3/4 outputs, writes to `data/step5-post-processing/lmp/`. No changes to Steps 1–4.

```
Step 1 (PFS) → Step 2 (EF) → Step 3 (Cost) → Step 4 (Postprocess)
                                                      ↓
                                          step5_compute_lmp_prices.py
                                                      ↓
                              data/step5-post-processing/lmp/{ISO}_lmp.parquet   (per-ISO output)
                              data/step5-post-processing/lmp/{ISO}_archetypes.parquet
                              data/step5-post-processing/lmp/lmp_summary.json  (dashboard-ready)
```

#### Shared Architecture: `dispatch_utils.py`

Single source of truth for dispatch reconstruction, fossil retirement, and profile loading. All Step 5/6 scripts import from this module. Step 1 Numba JIT dispatch functions consolidated here.

```
dispatch_utils.py (shared)
├── Constants: battery/LDES params, hydro caps, grid mix, coal/oil caps, base demand
├── get_supply_profiles(iso, gen_profiles)           ← nuclear derate + DST correction
├── reconstruct_hourly_dispatch(..., detailed=False)  ← battery + LDES dispatch
│   └── detailed=True adds: per-resource matched/surplus, charge profiles
├── _compute_per_resource_dispatch(...)               ← merit-order: CF→CCS→hydro→wind→solar
├── _battery_loop_detailed / _ldes_loop_detailed      ← Numba loops with charge tracking
├── compute_fossil_retirement(iso, clean_pct, ...)    ← remaining capacity at threshold
├── compute_co2_from_dispatch(iso, dispatch, ...)     ← hourly merit-order emission accounting
├── load_common_data()                                ← demand, gen profiles, emission rates, fossil mix
├── Dispatch cache: load/save per-ISO NPZ, versioned (CACHE_VERSION=2)
└── DISPATCH_ORDER, CACHE_VERSION constants

step4_build_dispatch_cache.py (runs first — Step 5)
├── extract_unique_mixes(iso, input_dir)              ← reads step4/step3 parquets
├── build_cache_for_iso(iso, mixes, ...)              ← detailed=True for all mixes
└── Output: data/step5-post-processing/dispatch_cache/{ISO}_dispatch_cache.npz (v2)

step5_compress_day_profiles.py (Step 6 — reads from dispatch cache)
├── dispatch_from_cache(iso, mix, ...)                ← cache lookup → result format
├── compress_to_24h(result)                           ← 8760 → 24 hour-of-day sums
└── No duplicate dispatch engine — imports from dispatch_utils

step5_compute_co2.py (Step 6 — dispatch-stack emission model)
├── compute_dispatch_stack_emission_rate()
├── fast_co2_from_match_score()                       ← ~1000x faster, no dispatch needed
├── compute_co2_hourly()                              ← fallback dispatch path
└── recompute_all_co2()

step5_consequential_deployment_queue.py (Step 6 — uses dispatch cache for emissions)
├── get_dispatch_co2()                                ← dispatch-cache-based emission lookup
├── compute_marginal_displaced_rate_dispatch()         ← zone-boundary CO₂
└── extract_medium_scenarios() + consequential queue logic

step6_scenario_comparison.py (Step 6 — dispatch cache for both scenarios)
├── _get_dispatch_co2_for_mix()                       ← scenario mix emission lookup
├── build_consequential_queue()                       ← dispatch-based emissions
└── find_optimal_mixes() / find_optimal_mixes_sequential()

step5_compute_lmp_prices.py (Step 6 — imports dispatch_utils)
├── PriceModel classes (ISO-specific)
├── build_merit_order_stack()
├── compute_lmp_stats()
└── calibration framework
```

**Dispatch cache pipeline position**: `step4_build_dispatch_cache.py` runs after Step 4, before all Step 6 scripts. Pre-computes dispatch for all unique mixes across 7 ISOs. Cache is versioned (v2) to invalidate stale v1 caches.

**CO₂ recompute bug fix (Feb 2026)**: Was importing `get_supply_profiles_simple` (flat clean_firm, no DST correction) instead of the canonical `get_supply_profiles`. Fixed to use the same nuclear-derated, DST-corrected profiles as Step 1.

**Compressed day compatibility note**: Local dispatch engine replaced with canonical dispatch_utils engine. Total energy dispatched is identical; hourly distribution differs slightly.

#### Design Decisions (Locked)

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | Merit-order stack walk | `np.searchsorted` step-function | More accurate than `np.interp` linear interpolation — discrete units don't interpolate. Same performance. |
| 2 | Archetype dedup key | `(mix_tuple, fuel_level, threshold)` | Threshold affects fossil stack (retirement changes available capacity). ~7,800 unique calcs per ISO. Still fast with Numba (<30s). |
| 3 | Dispatch functions | Shared module (`dispatch_utils.py`) | Single source of truth. Extracted from `step5_compute_co2.py` + Step 1 Numba JIT. |
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
- `data/step5-post-processing/lmp/actual_lmp_PJM.json` — PJM Data Miner 2 API (Western Hub, 2021-2025)

**Outputs**:
- `data/step5-post-processing/lmp/{ISO}_lmp.parquet` — summary stats per (threshold, scenario), ~2 MB/ISO
- `data/step5-post-processing/lmp/{ISO}_archetypes.parquet` — 8760h profiles for unique archetypes, ~15-20 MB/ISO
- `data/step5-post-processing/lmp/{ISO}_checkpoint.json` — resume state (transient, deleted on completion)
- `data/step5-post-processing/lmp/lmp_summary.json` — dashboard-ready cross-ISO summary, <500 KB
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

**Per-ISO stats: `data/step5-post-processing/lmp/{ISO}_lmp.parquet`**

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

**Per-ISO archetype profiles: `data/step5-post-processing/lmp/{ISO}_archetypes.parquet`**

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
| **0** | Extract `dispatch_utils.py` from `step5_compute_co2.py` + compatibility test | `dispatch_utils.py`, `step5_compute_co2.py` | ~300 lines extracted |
| **1a** | Core engine — PJM only, Medium fuel, no calibration | `step5_compute_lmp_prices.py` | ~400 lines |
| **1b** | Full fuel sensitivity sweep (L/M/H) + all thresholds + checkpointing | Same | +~150 lines |
| **1c** | All 5 ISOs with market-specific models | Same | +~200 lines |
| **2a** | PJM LMP data fetch | `fetch_pjm_lmp.py` | ~150 lines |
| **2b** | Calibration + validation | `calibrate_lmp_model.py` | ~200 lines |
| **3** | All-ISO calibration data fetch | Per-ISO fetch scripts | ~150 lines |
| **4** | Dashboard integration | `step6_generate_shared_data.py` + JS | ~150 lines |

#### New Files

```
dispatch_utils.py              # Shared dispatch/retirement/profiles (~300 lines)
step5_compute_lmp_prices.py  # Main LMP script (~750 lines)
fetch_pjm_lmp.py               # Phase 2: LMP data fetcher (~150 lines)
calibrate_lmp_model.py         # Phase 2: Parameter calibration (~200 lines)
data/step5-post-processing/lmp/                      # Output directory
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
- **Regions**: CAISO, ERCOT, PJM, NYISO, NEISO, MISO, SPP (7 ISOs)
- **Repo**: `jessicacohen554-cyber/hourly-cfe-optimizer`

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
- **Green H2 seasonal storage** (added Feb 2026):
  - **Parameters**: 35% round-trip efficiency (electrolysis 70% × storage 95% × turbine 55%), 1000hr duration (~42 days at full power), 30-day rolling dispatch window
  - **Physics**: Dispatches as Phase 4 after battery4 → battery8 → LDES on post-LDES residual surplus/gap. Same window-based charge/discharge as LDES but with longer window and lower RTE.
  - **Sweep levels**: Only evaluated at ≥95% thresholds (too expensive for lower). Levels: [0, 1, 2, 5, 10, 20] % of demand.
  - **Cost**: LCOS-based, shares `ldes_lvl` sensitivity toggle. L=$185-230, M=$260-330, H=$365-460 $/MWh by ISO. Transmission adders: L=$2-3, M=$3-6, H=$5-10.
  - **Peak capacity credit**: 0.85 (dispatchable but slower ramp than gas/battery)
  - **Merit order rationale**: Battery → LDES → H2 is economically robust because (1) higher RTE storage should fill short gaps first to minimize surplus waste, (2) battery $/kW is lower than LDES for 4hr needs, (3) H2's only advantage is very cheap $/kWh (salt caverns) at multi-week timescales where LDES is prohibitively expensive.
- **CAISO geothermal as 5th physics dimension** (added Feb 2026):
  - CAISO uses 5D grid search: [clean_firm (nuclear/CCS only), solar, wind, hydro, geothermal] — each as independent % of demand (no sum constraint).
  - **Geothermal profile**: Flat year-round (1/8760 per hour). No seasonal derate — geothermal has no refueling outages.
  - **CAISO clean_firm profile**: Now purely nuclear with full seasonal derate (NUCLEAR_SHARE_OF_CLEAN_FIRM = 1.0 for CAISO). The 70/30 nuclear/geo blend is removed; geothermal physics are captured by the separate dimension.
  - **Geothermal cap**: (existing_geo_TWh + GEO_CAP_TWH) / CAISO_demand_TWh = (5.31 + 39.0) / 224.039 = 19.8% → capped at 20% in grid search.
  - **Non-CAISO ISOs**: Stay 4D. No geothermal resource.
  - **Rationale**: Geothermal has fundamentally different physics than nuclear/CCS (no seasonal derate, no outages). Lumping into clean_firm understated CAISO's winter/spring firm capacity.
- **Clean Firm nuclear derate**: Seasonal spring/fall derate applied to nuclear/CCS portion. Reflects staggered refueling outages for nuclear and scheduled maintenance for CCS-CCGT in shoulder months. Summer/winter: ~100% CF. Spring/fall: reduced CF based on observed EIA 2021-2025 patterns. CCS-CCGT aggregate fleet maintenance in shoulder months produces a similar derate pattern.
- **Hydro**: Existing only, capped at regional capacity, wholesale priced, no new-build tier, $0 transmission
- **CCS-CCGT** (within Clean Firm): 95% capture rate, residual ~0.0185 tCO2/MWh, 45Q ($85/ton = ~$27.5/MWh offset) baked into LCOE, fuel cost linked to gas price toggle. **Modeled as flat baseload (not dispatchable) by design** — while CCS-CCGT is physically dispatchable, the 45Q tax credit ($85/ton for geologic storage) incentivizes running at maximum capacity factor to maximize capture credits. This is an economics-driven decision, not a physical constraint.
- **LDES**: 100-hour iron-air, 50% round-trip efficiency, capacity-constrained dispatch with dynamic capacity sizing. LCOS reflects actual utilization of built capacity. (Decision 7A — kept current.)
- **Battery**: 4-hour Li-ion, 85% round-trip efficiency, capacity-constrained daily-cycle dispatch. LCOS reflects actual utilization — oversized capacity that sits idle drives cost up. (Decision 7A — kept current.)

---

## 3. Thresholds (21 total — v4.2: added 10/20/30/40 coarse low range + 99.5/99.9 last-mile)

```
10, 20, 30, 40 [coarse only], 50, 55, 60, 65, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 99.5, 99.9, ≥99.99
```

- **10%, 20%, 30%, 40%** (v4.2): Coarse-grid only — no fine zone search, no step1d storage refinement. Captures early adoption / RPS-range where mixes are easy to achieve and cost curves are flat.
- **50%, 55%, 60%, 65%, 70%**: Captures the easy-to-achieve baseline region where most mixes succeed. 5% granularity anchors the cost curve left side.
- 5% intervals from 75–85 (captures broad trend)
- 2.5% intervals from 87.5–97.5 (captures steep cost inflection zone)
- **99%, 99.5%, 99.9%, ≥99.99%** (v4.2 added 99.5/99.9): Last-mile granularity at the near-perfect end. True 100% is physically unreachable.
- Key inflection behavior (CCS/LDES entering mix, storage costs spiking) captured at 90–97.5
- Dashboard interpolates smoothly between anchor points for abatement curves

---

## 4. Dashboard Controls (7 total — paired toggles)

### Preserved (2):
1. **Region/ISO select** (CAISO, ERCOT, PJM, NYISO, NEISO)
2. **Threshold select** (10 values: 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, ≥99.99)

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

**Problem**: 5,832 cost scenarios × 21 thresholds × 5 ISOs = 611,280 co-optimizations (17 active thresholds for full cost optimization, 4 coarse thresholds for coarse pass only). Even with warm-start, running all 5,832 per threshold is slow. Empirically, physics dominates at lower thresholds — only ~14 unique mixes serve all 5,832 scenarios.

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

**Uprate cap** — 8% of existing nuclear capacity (includes MUR + stretch + good EPU opportunities):

| Region | Existing Nuclear (GW) | Uprate Cap (GW) | Uprate Cap (TWh/yr @ 90% CF) |
|---|---|---|---|
| **CAISO** | 2.3 (Diablo Canyon) | 0.18 | 1.5 |
| **ERCOT** | 2.7 (South Texas Project) | 0.22 | 1.7 |
| **PJM** | 32.0 (largest US fleet) | 2.56 | 20.2 |
| **NYISO** | 3.4 (Nine Mile, FitzPatrick, Ginna) | 0.27 | 2.1 |
| **NEISO** | 3.5 (Millstone, Seabrook) | 0.28 | 2.2 |
| **Total** | **43.9** | **3.51** | **27.7** |

*8% chosen: NRC has approved ~8% fleet-wide historically (MUR + stretch + EPU). Good EPU opportunities remain across ~27 of 94 reactors, particularly BWR plants. DOE executive order targets ~3-5 GW; INL LWRS estimates 3-8% remaining. 8% reflects full remaining potential including EPU deployment at $15-40/MWh — the cheapest new dispatchable clean capacity available.*

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

#### 5.4.3 Regional CCS Capacity Caps (TWh/yr)

CCS-CCGT allocation is capped per ISO based on geologic CO₂ storage availability, infrastructure, and regulatory feasibility — identical pattern to the geothermal cap (`GEO_CAP_TWH = 39.0`) for CAISO. The cap is enforced in Step 3 cost optimization (merit-order tranche logic) and will be propagated to Step 1 mix filtering in the next physics run.

```python
CCS_CAP_TWH = {
    'CAISO': 25.0,    # 11% of 224 TWh demand
    'ERCOT': 200.0,   # 41% of 488 TWh demand
    'PJM':   125.0,   # 15% of 843 TWh demand
    'NYISO': 0.0,     # Hard zero — no geologic storage
    'NEISO': 0.0,     # Hard zero — no geologic storage
    'MISO':  200.0,   # 30% of 660 TWh demand
    'SPP':   50.0,    # 17% of 296 TWh demand
}
```

**Regional justification:**

- **NYISO (0 TWh)**: No suitable onshore CO₂ storage geology. Newark Rift Basin assessed as "low potential" by USGS/NETL. Offshore Atlantic (Baltimore Canyon Trough) is decades from permitting. Zero Class VI well applications filed, state not pursuing primacy.
- **NEISO (0 TWh)**: Crystalline and metamorphic bedrock — zero identified CO₂ storage units in the USGS National Carbon Sequestration Database. No saline formations or depleted reservoirs anywhere in New England. Additionally constrained by winter gas pipeline bottleneck (Step 4 adder).
- **CAISO (25 TWh / 11%)**: Excellent geology (San Joaquin Basin 14–56 Gt, Sacramento Basin ~3 Gt) but SB 905 imposes strictest CCS regulatory framework in US. Zero operating CCS projects, zero CO₂ pipeline infrastructure in-state.
- **SPP (50 TWh / 17%)**: Good geology (Anadarko Basin, Arbuckle Group — 780 Mt P50 in KS) but Oklahoma induced seismicity from underground injection creates regulatory/social resistance. State pursuing but has not received Class VI primacy.
- **PJM (125 TWh / 15%)**: Stark east-west split. Western PJM (WV/OH/western PA) sits on Appalachian Basin (450–500 Gt theoretical); WV received Class VI primacy Jan 2025. Eastern PJM (DC/MD/VA/NJ/DE — majority of demand) has unsuitable Piedmont/Coastal Plain geology. No CO₂ transport infrastructure connecting east to west.
- **ERCOT (200 TWh / 41%)**: Best CCS region in US. Gulf Coast has 20+ Gt depleted offshore fields, 100s Gt offshore saline formations. TX received Class VI primacy Dec 2025 (64 apps from EPA). Denbury CO₂ pipeline network (900+ mi) is densest in US. Multiple storage hubs under development.
- **MISO (200 TWh / 30%)**: Mt. Simon Sandstone (12–172 Gt) is the most characterized formation in US with 2+ Mt successfully injected at ADM Decatur. ND has had primacy since 2018 with 3 active projects. Broadwing 400 MW CCS-CCGT (Google-backed, FID Q2 2026) would be first in US.

**Implementation (Step 3, March 2026)**: CCS cap enforced in two places within `price_mix_batch`:
1. **Implicit CCS residual** (`ccs_pct = 100 - sum(cf, sol, wnd, hyd)`): TWh capped at `CCS_CAP_TWH[iso]`. Excess priced as nuclear new-build at the firm gen toggle level.
2. **Tranche 3 CCS** (clean_firm overflow after uprate + geothermal): CCS headroom = `cap - residual_ccs_twh`. If CCS is cheaper than nuclear but headroom exhausted, overflow goes to nuclear new-build.
For NYISO/NEISO (cap=0), all CCS → nuclear automatically. Step 1 will filter `ccs_pct × demand_TWh ≤ CCS_CAP_TWH[iso]` in the next physics run (deferred to avoid re-running Step 1 this iteration).

*Sources: USGS National Carbon Sequestration Database (NATCARB), NETL Carbon Storage Atlas V (2015), DOE CarbonSAFE program status (2024–2025), EPA Class VI well permit tracker, California SB 905 (2022), Princeton Net-Zero America (2021), Global CCS Institute Status Report (2024), IEEFA CCS deployment analysis (2024).*

### 5.5 Battery Costs — NREL Component Model + Wright's Law Decline

**Updated March 2026.** Battery costs re-anchored to NREL ATB 2024 component model with Wright's Law learning curves for future cost decline.

**CAPEX derivation** — NREL ATB 2024 separates battery costs into energy ($/kWh) and power ($/kW) components. Total installed cost per kWh = Energy + Power/Duration. This gives the correct 4hr→8hr ratio (~14% cheaper for 8hr, because power electronics spread over 2× the energy capacity).

| Level | Energy ($/kWh) | Power ($/kW) | 4hr Total | 8hr Total | 8hr/4hr |
|---|---|---|---|---|---|
| Low | $170 | $280 | $240/kWh | $205/kWh | 85.4% |
| Medium | $210 | $340 | $295/kWh | $253/kWh | 85.6% |
| High | $270 | $420 | $375/kWh | $323/kWh | 86.0% |

*Low = aggressive LFP procurement + competitive BOS. Medium = typical US utility project (~$295/kWh vs NREL $334 benchmark — reflecting 2025 market reality below NREL's conservative bottom-up model). High = tariff-exposed, constrained interconnection.*

**Financial parameters**: WACC=8%, 20yr life, FOM=2.5% of CAPEX($/kW) per NREL (includes augmentation). Annualized = CAPEX × (CRF + 0.025) / 8760 × 1000 × regional_mult. Regional multipliers: ERCOT=1.00 (cheapest), CAISO=1.11, NYISO=1.18 (highest).

**Annualized capacity costs** ($/MWh-cap, 2025 starting values):

| Level | Type | CAISO | ERCOT | PJM | NYISO | NEISO | MISO | SPP |
|---|---|---|---|---|---|---|---|---|
| Low | 4hr | 3.86 | 3.48 | 3.70 | 4.08 | 3.97 | 3.62 | 3.52 |
| Medium | 4hr | 4.75 | 4.27 | 4.55 | 5.02 | 4.87 | 4.46 | 4.33 |
| High | 4hr | 6.03 | 5.43 | 5.78 | 6.38 | 6.20 | 5.66 | 5.50 |
| Low | 8hr | 3.30 | 2.97 | 3.16 | 3.49 | 3.39 | 3.10 | 3.01 |
| Medium | 8hr | 4.06 | 3.66 | 3.89 | 4.30 | 4.17 | 3.81 | 3.70 |
| High | 8hr | 5.19 | 4.67 | 4.97 | 5.49 | 5.33 | 4.87 | 4.73 |

**LCOS cross-check** (4hr Medium ERCOT, 365 cycles, 85% RTE): **$121/MWh**. Consistent with Lazard 2024 ($115-220/MWh range).

**Wright's Law learning curves** — Battery costs decline from 2025 starting values toward terminal NOAK floor. This is the reverse direction from other technologies (which start at FOAK and decline to NOAK): batteries are already at manufacturing scale, so 2025 IS the starting point. Curves calibrated to NREL 2050 cost projections.

Terminal NOAK ($/kWh): Low=50%, Medium=56%, High=80% of 2025 starting cost.
- Low 4hr: $120/kWh by 2042 | Med 4hr: $165/kWh by 2048 | High 4hr: $300/kWh by 2050
- Low 8hr: $102/kWh by 2040 | Med 8hr: $141/kWh by 2046 | High 8hr: $258/kWh by 2050

Learning curve exponent: 0.6 (concave ramp — steeper initially, asymptotic approach). 8hr reaches NOAK ~2yr faster than 4hr because cell costs (which decline faster) are a larger share of 8hr total cost.

**Trajectory (4hr Medium ERCOT):**
| Year | Wright's fraction | CAPEX | Annualized | LCOS (365 cyc) |
|---|---|---|---|---|
| 2025 | 0.00 | $295/kWh | $4.27/MWh-cap | $121/MWh |
| 2030 | 0.40 | $243/kWh | $3.52/MWh-cap | $99/MWh |
| 2035 | 0.61 | $216/kWh | $3.13/MWh-cap | $88/MWh |
| 2040 | 0.77 | $194/kWh | $2.82/MWh-cap | $79/MWh |
| 2048+ | 1.00 | $165/kWh | $2.39/MWh-cap | $67/MWh |

*Sources: [NREL ATB 2024](https://atb.nrel.gov/electricity/2024/utility-scale_battery_storage), [NREL Cost Projections 2025 Update](https://docs.nrel.gov/docs/fy25osti/93281.pdf), [Ember Battery Storage Costs](https://ember-energy.org/latest-insights/how-cheap-is-battery-storage/), Wright's Law learning rate literature.*

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

### 6.3 Storage Grid Refinement — Sub-Percent Granularity (Decision: Feb 21, 2026)

**Problem identified**: The original storage sweep grid `[0, 2, 5, 8, 10, 15, 20]` (% of annual demand) had a blind spot. Battery4 and Battery8 max SOC never exceeds ~1.0% of annual demand even under peak-stress conditions (high RE, low CF, max procurement, >90% targets). The jump from 0% → 2% skipped the entire range where batteries actually saturate, meaning the cost optimizer never tested right-sized battery configurations. This systematically overpriced storage (paying for 4-20× idle capacity) and biased the optimizer toward avoiding batteries when properly-sized batteries could be cost-competitive.

**Empirical saturation thresholds** (max SOC as % of annual demand, unconstrained capacity, high-RE stress mixes at 97.5-99% targets):

| ISO | Bat4 (4hr) 90% Sat | Bat8 (8hr) 90% Sat | LDES (100hr) |
|-----|---------------------|---------------------|--------------|
| CAISO | 0.577% | 0.577% | >50% (always saturated) |
| ERCOT | 0.663% | 0.663% | >50% |
| PJM | **1.155%** | **1.155%** | >50% |
| NYISO | 0.922% | 0.922% | >50% |
| NEISO | 0.975% | 0.975% | >50% |

**Root cause**: Battery daily surplus/gap is small relative to annual demand (~0.5% of annual demand on peak days). The 4hr/8hr durations provide sufficient power headroom that power rating never binds — only energy capacity matters. PJM is the binding case due to high-wind mixes at 99% creating larger daily swings.

**LDES is fundamentally different**: Multi-day accumulation over 7-day windows means LDES fills to capacity even at 20% of annual demand. LDES is capacity-hungry through the entire tested range. Fine granularity for LDES is about optimizing the marginal cost/benefit tradeoff, not finding saturation.

**Refined storage grids** — 0.1% intervals below max saturation, then coarser above:

```python
# Bat4: 0.1% intervals to 1.5% (covers PJM binding case + margin), then coarser
batt_levels  = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 2.5, 5, 10, 15, 20]  # 20 levels

# Bat8: identical physics (same surplus, duration only affects power limit which never binds)
batt8_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 2.5, 5, 10, 15, 20]  # 20 levels

# LDES: 0.5% intervals to 2.5% (marginal value optimization), then coarser
ldes_levels  = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 5, 8, 10, 15, 20]  # 11 levels
```

**Combo count**: 20 × 20 × 11 = **4,400** storage combos per mix (vs. old 7³ = 343 → **12.8×** increase).

**Implementation notes**:
- Storage levels change from integer to float → parquet schema uses `pa.float64()` for `battery_dispatch_pct`, `battery8_dispatch_pct`, `ldes_dispatch_pct`
- All downstream code (Step 2 EF, Step 3 cost, Step 4 postprocess, dashboard) must handle float storage values
- Existing cache with integer storage values must be preserved and converted on merge

### 6.4 Storage Sweep Optimizations — Batched Dispatch + Parallel Mixes (Feb 21, 2026)

The 12.8× increase in nominal storage combos is offset by a batched + parallel architecture:

#### 1. Batched Storage Dispatch (`_batch_storage_scores`)

Instead of calling `_score_with_all_storage` per combo (each recomputing bat4/bat8 dispatch from scratch), a single `_batch_storage_scores` call evaluates ALL bat4×bat8×LDES combos for a given (mix, procurement):

- **Bat4 dispatch residual reuse**: Computed once per bat4 level, reused across all bat8 levels
- **Bat8 dispatch residual reuse**: Computed once per (bat4, bat8) pair, reused across all LDES levels
- **LDES dispatch**: Only the innermost loop; runs on post-bat4+bat8 residual

This eliminates the redundant base dispatch recomputation that dominated the old triple-nested loop.

#### 2. Parallel Mix Screening (`_batch_mixes_storage_screen`)

Near-miss mixes are processed in batches of `MAX_MIX_BATCH = 100`. Each batch:
1. Pre-computes curtailment-MW caps for all mixes (fast: one 8760-hour pass per mix)
2. Gathers supply rows into a (N_batch, 8760) array
3. Calls `_batch_mixes_storage_screen` which uses **Numba `prange`** to distribute mixes across CPU cores
4. Each core runs `_batch_storage_scores` for its assigned mixes in parallel

This prevents large ISOs (NYISO with hydro_cap=15.9%, PJM) from stalling by parallelizing across mixes.

#### 3. Energy-Based Storage Cap (Per-Mix Physics Ceiling)

For each (mix, max_procurement), `_compute_storage_caps` computes the maximum surplus energy that could charge each storage type over its operational window:

- `bat4_cap = max_daily_surplus` — max energy surplus in any single day (4hr daily-cycle battery can't charge more than available daily curtailment)
- `bat8_cap = max_2day_surplus` — max energy surplus in any 2-day window (8hr battery uses 48hr dispatch window)
- `ldes_cap = max_7day_surplus` — max energy surplus in any 7-day window (100hr iron-air)

This is an **energy-based** ceiling, not power-based. A 4hr battery at 200% pure solar saturates at ~0.3% of annual demand capacity because the discharge-side gap (nighttime hours) limits useful capacity, not the charge-side surplus. The 0.3% capacity cycles 365×/year, delivering ~57% of annual demand throughput at 61% utilization.

Levels above the per-mix cap are auto-skipped in the storage sweep.

#### 4. Curtailment Frequency Filter

Daily-cycle batteries (bat4/bat8) need **≥ 150 surplus days** to justify capacity. Mixes with fewer surplus days skip battery combos entirely; only LDES is evaluated (which accumulates across multi-day windows).

### 6.5 Step 1D Storage Refinement Module (Decision: Mar 1, 2026)

**Problem**: Step 1C's coarse storage levels at <95% thresholds ([0,1,3] for bat4, [0,2,4] for bat8, [0,5,10] for LDES as % of annual demand) are too wide for the physical storage caps. Typical caps are bat4=0.2–0.5%, bat8=0.5–1.0%, LDES=1.0–3.0%. The first non-zero coarse level already exceeds the cap for most mixes, so the cap filtering skips ALL non-zero levels — effectively never exploring storage at <95% thresholds.

**Solution**: `step1d_storage_refinement.py` — a standalone module that reads the Step 1C coarse cache to identify candidate mixes and evaluates intermediate storage levels. No rerun of Steps 1A–1C.

**Storage levels for 65–92.5% thresholds** (full intermediate sweep):
```python
bat4:  [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]  # 14 levels
bat8:  [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]                 # 12 levels
LDES:  [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 5.0, 7.0, 10.0]          # 13 levels
H2:    [0]                                                                                # 1 level
# Total: 14 × 12 × 13 × 1 = 2,184 storage combos per mix
```

**Storage levels for ≥95% thresholds** (LDES intermediates only):
```python
bat4:  [0, 1, 3, 5]           # same as 1C (caps are larger at high procurement)
bat8:  [0, 2, 4, 6]           # same as 1C
LDES:  [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0]  # 13 levels (fills 0→5 gap)
H2:    [0, 5, 10, 20]         # same as 1C
# Total: 4 × 4 × 13 × 4 = 832 storage combos per mix
```

**Algorithm**:
1. Load coarse cache (no-storage scores) from Step 1B
2. For each threshold ≥65%: identify near-miss mixes (score >= max(target − 0.40, 0.50) AND score < target)
3. Compute physical storage caps per mix via `_batch_compute_storage_caps` (Numba parallel)
4. Batch-evaluate all storage combos via `_batch_mixes_storage_screen` (100 mixes/batch, Numba prange)
5. Cap filtering: skip levels exceeding per-mix physical cap (same as 1C §6.4.3)
6. Curtailment filter: batteries need ≥150 surplus days (same as 1C §6.4.4)
7. Output feasible solutions (score ≥ target AND at least one storage > 0) to new parquets

**Output**: `data/step1d-storage-parquets/{ISO}_t{XX}_storage_refined.parquet` — same schema as Step 1C PFS parquets, `pareto_type = 'storage_refined'`.

**Step 2 integration**: `step2_efficient_frontier.py` scans both `data/step1-pfs-parquets/` and `data/step1d-storage-parquets/`. Deduplication handles any overlap between 1C and 1D results (keeps max score per unique resource + storage key).

**Test results** (ERCOT t75): 3,303,625 new storage-enabled solutions found in 78s (vs 0 from Step 1C). Cap ranges confirmed: bat4=[0.00%, 0.34%, 0.80%], LDES=[0.00%, 2.10%, 5.15%].

#### 5. Bat8 Two-Day Dispatch Window

Battery 8hr dispatch uses 48hr (2-day) windows instead of daily 24hr windows, reflecting ~200 cycles/year operational pattern. This allows accumulating surplus across 2 days before discharging.

```python
batt8_window = 48  # 2-day window → ~183 windows/year → ~200 actual cycles
```

#### 6. Dominance Pruning (LDES Early Stop)

Within each (bat4, bat8) pair's LDES sweep, if increasing LDES doesn't reduce minimum feasible procurement, stop — higher LDES adds cost without reducing procurement.

#### 7. Infeasible Screening

The batch call at max procurement screens ALL combos in one shot. Combos infeasible at max procurement are skipped without binary search (biggest win — most combos are infeasible).

**Net architecture**: `_batch_mixes_storage_screen` (Numba prange across mixes) → `_batch_storage_scores` (residual reuse across storage combos) → post-filter by per-mix caps → binary search only feasible combos.

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

**Implementation**: Add growth counterfactual to `step5_compute_co2.py`. Growth rates from `step3_cost_optimization.py` DEMAND_GROWTH_RATES (CAISO 1.4–2.5%, ERCOT 2.0–5.5%, PJM 1.5–3.6%, NYISO 1.3–4.4%, NEISO 0.9–2.9%). New gas rate is 350 kg/MWh (representative CCGT heat rate ~6,400 BTU/kWh, pipeline gas). This is a post-hoc calculation — doesn't change resource mix or cost optimization.

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

**DAC Cost Trajectories ($/ton CO₂, net DACCS)** — *Revised Feb 27, 2026*:

| Year | Optimistic | Central | Conservative |
|------|-----------|---------|-------------|
| 2025 | $600      | $800    | $1,100      |
| 2030 | $350      | $500    | $750        |
| 2035 | $230      | $375    | $550        |
| 2040 | $175      | $300    | $450        |
| 2045 | $130      | $250    | $375        |
| 2050 | $100      | $200    | $300        |

**Revision rationale (Feb 27, 2026)**: Previous trajectories were too aggressive, particularly the optimistic scenario ($400 in 2025, $200 by 2030). Actual 2025 DAC costs are $600–$1,500/tCO₂ (Climeworks ~$1,000/ton, market average ~$600–$1,500 with subsidies). No credible source projects sub-$300 by 2030. Revised trajectories are anchored to:
- **2025 actuals**: Climeworks operational costs, CDR marketplace prices
- **2030–2035**: IEAGHG NOAK estimates ($194–$230 at 1 MtCO₂/yr, achievable "by as early as 2035"), Belfer Center projections ($400–$1,000 by 2030)
- **2040–2050**: Sievert et al. (Joule 2024) learning curves ($341/tCO₂ central at Gt scale), Climeworks roadmap (well below $500 by ~2040, $200–$250 towards 2050)

**Sources**: Climeworks (2024/2025 operational data), Sievert et al. (Joule 2024), IEAGHG (2021/2024), Belfer Center/Harvard (2023), DOE Liftoff (2023), IEA DAC (2022/2024), Fasihi et al. (J. Cleaner Prod. 2019), DOE Carbon Negative Shot, NAS (2019), Young et al. (One Earth 2023), Keith et al. (Joule 2018).

**Key assumptions by trajectory**:
- **Optimistic**: IEAGHG NOAK costs by 2035, strong learning rates, low-cost renewable energy, GtCO₂/yr scale by 2050
- **Central**: Belfer Center mid-range, Climeworks roadmap trajectory, moderate policy support, 100–500 MtCO₂/yr by 2050
- **Conservative**: Slow scale-up, limited policy, high energy costs, <100 MtCO₂/yr by 2050

**Visualization**: Abatement charts get dual x-axis (threshold % bottom, SBTi year top). DAC trajectory shown as 3 declining curves with shaded band. MAC curve intersections with DAC curves show the crossover points where grid decarbonization becomes more/less expensive than DAC at each milestone year.

All values are 2024 USD, net tons CO₂ removed (accounting for 5–12% lifecycle emissions). Full DACCS (capture + transport + storage + MRV).

### 7.2 Abatement Cost Curves (2 new charts)
- **Average Cost of Abatement**: Total incremental cost / Total CO2 abated = **$/ton CO2**
- **Marginal Cost of Abatement**: (Cost_{X+1%} − Cost_{X%}) / (CO2_{X+1%} − CO2_{X%}) = **$/ton CO2**
- **X-axis**: 75% to 100%, **linear numeric scale** (proportional spacing — distance from 85→90 equals 75→80)
- Both curves respond dynamically to **all 10 sensitivity toggles**
- 1% intervals from 85% provide smooth curve in the inflection zone
- Marginal curve shows hockey-stick shape: cheap early tons, expensive last tons

### 7.4 Optimal CFE Target per ISO — MAC × DAC Crossover (Decision: Feb 26, 2026)

**Goal**: For each ISO, identify the CFE threshold range where marginal grid decarbonization cost exceeds DAC — the "optimal target" beyond which buying offsets is cheaper than building more clean energy.

**Why stepwise MAC failed**: The existing stepwise MAC (Δcost/ΔCO₂ between adjacent thresholds) is wildly non-monotonic because:
1. Each threshold is independently optimized — the portfolio at 90% isn't built incrementally from the 87.5% portfolio
2. Coal retirement cliff at 70% causes a regime change in the CO₂ denominator
3. Fine threshold spacing (2.5% steps) amplifies small-denominator noise

**Solution — Option B: Smooth Marginal MAC from Cost Frontier**:
1. At each threshold, take the independently-optimized cheapest system (from Step 3)
2. Total cost premium ($M/yr = (system_cost - wholesale) × demand) and total CO₂ abated (Mt) form curves vs. threshold
3. Apply isotonic regression to enforce monotonicity (cost and CO₂ must be non-decreasing with threshold)
4. Fit monotone cubic splines (PCHIP) to the corrected curves
5. Marginal MAC = d(TotalCost)/d(CO₂) — the derivative of cost w.r.t. CO₂ along the spline
6. Cross with DAC cost trajectories to find crossover thresholds

**Crossover Range**: 3 grid cost tiers (L/M/H) × 3 DAC scenarios (optimistic/central/conservative) = 9 crossover points. The range = [min crossover, max crossover] across all 9 combinations. This captures: "between X% if DAC costs are low and clean energy costs are high, and Y% if DAC costs are high and clean energy costs are low."

**Option A: Target-Specific Analysis Within the Range**:
For each discrete threshold inside the crossover range (±1 step for context):
- Resource mix composition, system cost, total investment
- Comparison to DAC at the corresponding SBTi year
- Shows WHAT changes in the system as you stretch toward higher targets

**Demand Growth (L/M/H)**:
- Annual growth rates per ISO: CAISO 1.4/1.9/2.5%, ERCOT 2.0/3.5/5.5%, PJM 1.5/2.4/3.6%, NYISO 1.3/2.0/4.4%, NEISO 0.9/1.8/2.9%, MISO/SPP 2.0% (uniform)
- **Key finding**: Marginal MAC ($/tCO₂) is scale-invariant w.r.t. demand growth — both d(cost) and d(CO₂) scale by the same growth factor, so the ratio is unchanged. The crossover threshold % is the same regardless of demand growth.
- Demand growth DOES affect: total investment $M, total CO₂ abated, absolute resource quantities (TWh/GW). These are critical for the no-regrets analysis.

**No-Regrets Resource Investment Analysis**:
Within the crossover range, some resource investments are needed regardless of where the optimal target lands:
- **Floor**: minimum % share of each resource across all thresholds in the range — the absolute minimum you'd build regardless
- **Consensus**: resources that are non-zero at every threshold in the range — they show up across the board
- **Average**: expected investment level across the range
- All three scaled by L/M/H demand growth for absolute TWh quantities

**Implementation**: `scripts/step5_compute_optimal_targets.py` (Step 6 post-processor, runs in parallel with MAC/LMP/etc.)
- Outputs: `data/step5-post-processing/optimal_targets.json`, `dashboard/js/optimal-target-data.js`
- Consumed by: `step6_generate_shared_data.py` → OPTIMAL_TARGETS in shared-data.js
- Depends on: CLEAN_COST (L/M/H effective_cost, no gas backup), RESOURCE_MIX_DATA, emission rates, DAC trajectories
- No dispatch cache dependency — uses pre-computed Step 3 cost data

#### 7.4.1 Scenario Comparison MAC — PCHIP Spline Smoothing (Decision: Feb 28, 2026)

**Problem**: The scenario comparison page (`step6_scenario_comparison.py`) computes per-threshold stepwise MAC as `Δnew_build_cost / ΔCO₂_abated` between adjacent thresholds. Because each threshold is independently optimized, the marginal cost bounces wildly (e.g., CAISO: null, 278, 278, 278, 12560, 9999, 232, 1421, 140, 9999...). This produces an unreadable, non-monotonic MAC curve instead of the expected hockey-stick shape.

**Solution — PCHIP Spline + Isotonic Regression** (two-pass approach):
1. **Pass 1**: Collect cumulative `(CO₂_abated, new_build_cost)` data points at each threshold per ISO from dispatch-cache CO₂ and `new_build_cost_total`.
2. **Pass 2**: Fit a PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) monotone spline to the cumulative supply curve `cost = f(CO₂)`.
3. Take the PCHIP derivative at each data point → raw marginal MAC = d(cost)/d(CO₂).
4. Apply `scipy.optimize.isotonic_regression` to enforce non-decreasing marginal cost (isotonic constraint).
5. Replace raw stepwise MAC values with smoothed values in the trajectory output.

**Why PCHIP**: PCHIP preserves monotonicity of the interpolant and avoids the Runge phenomenon (wild oscillations) that plague polynomial/cubic spline fits. It produces smooth curves through the data while respecting the natural convexity of the abatement supply curve.

**Why isotonic regression**: Even after PCHIP smoothing, numerical derivatives can produce minor non-monotonicity at certain data points. Isotonic regression is the minimum-perturbation projection onto the non-decreasing constraint set — it enforces the hockey-stick shape without artificially inflating values.

**Result**: Smooth, monotonically non-decreasing marginal MAC curve that starts low (~$50-300/t) for easy decarbonization and rises steeply at high thresholds (>95%) — the expected hockey-stick shape.

**Files changed**: `scripts/step6_scenario_comparison.py` — `_build_trajectory()` function rewritten with two-pass PCHIP approach. Added `scipy.interpolate.PchipInterpolator` and `scipy.optimize.isotonic_regression` imports. No changes to downstream consumers — `stepwise_mac` field in trajectory output dicts remains the same interface.

**Regeneration required**: Run Step 6 workflow with `step6_scenario_comparison` to regenerate `dashboard/js/scenario-comparison-data.js`.

---

#### 7.4.2 Gas Cost Separation in MAC (Decision: Feb 26, 2026)

**Decision**: MAC calculations use `effective_cost` (clean procurement only). Gas backup capacity cost is excluded from MAC because it is a system reliability cost, not an abatement cost.

**Rationale**:
- The MAC answers: "how much does it cost to abate one more ton of CO₂ via clean energy procurement?"
- Gas backup capacity is needed for grid reliability regardless of CFE target — it keeps the lights on
- Including gas backup in the MAC conflates the abatement cost with the reliability cost, distorting the crossover with DAC
- `step5_compute_mac_stats.py` already correctly uses `cost_incremental` (= `effective_cost - wholesale`) for MAC
- `step6_scenario_comparison.py` already correctly subtracts gas_cost before computing MAC: `new_build_per_mwh = total_cost - gas_cost`

**What changed**:
- `step5_compute_optimal_targets.py`: `SYSTEM_COST` (total_system_cost incl. gas) → `CLEAN_COST` (effective_cost only)
  - Medium: exact `effective_cost` from EFFECTIVE_COST_DATA
  - Low/High: approximation = `SYSTEM_COST(P10/P90) - gas_backup_cost(medium scenario)`
- `step5_consequential_deployment_queue.py`: MAC formula stripped `+ start/end['gas_cost']`; gas cost tracked separately as `delta_gas_cost_per_mwh`
- `step6_generate_shared_data.py`: added `CLEAN_COST_DATA` extraction (P10/P50/P90 of `effective_cost` across scenarios)

**Gas capacity as educational warning**:
- Gas backup cost is NOT part of the MAC but IS a critical educational point
- Consequential scenario dashboard must prominently warn: "Chasing cheap carbon without understanding system needs means retaining or building new gas capacity — an unavoidable system cost"
- `GAS_BACKUP_COST` per threshold per ISO tracked in optimal targets output for dashboard overlay
- `gas_cost_per_mwh_end` and `delta_gas_cost_per_mwh` added to consequential queue output

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
- **Checkpointing**: Saves after each threshold (21 per ISO); resumes from checkpoint on restart
- **Score caching**: Matching scores cached across 5,832 cost scenarios per threshold (physics reuse — cost-independent)
- **Cross-pollination**: After representative scenarios run per threshold, every unique mix re-evaluated against all scenarios
- **21 thresholds × 5 regions × 5,832 scenarios** — incremental saves essential for reliability

### 11.1 Direct Resource Fractions (v5.0 — replaces procurement multiplier)

**Decision (Feb 2026)**: Procurement multiplier removed. Resource fractions are now expressed directly as % of annual demand. No sum-to-100% constraint. Total generation (sum of all fractions) is what "procurement" used to be, but implicit.

**Old approach (removed)**: Mix shape (clean_firm + solar + wind + hydro = 100%) × procurement multiplier (50–500%). This created redundant evaluations — e.g., `(50% solar, 50% wind) @ 200%` and `(100% solar, 0% wind)` at different procurement levels could produce similar supply profiles. The procurement dimension was an unnecessary indirection.

**New approach**: Each resource varies independently as % of demand:
| Resource | Range | Step (coarse) | Step (fine) | Cap logic |
|----------|-------|---------------|-------------|-----------|
| Clean Firm (nuclear/CCS) | 0–120% | 5% | 1% | Nuclear/CCS with seasonal derate; 120% allows surplus for storage |
| Solar | 0–250% | 5% | 1% | High values capture solar+storage strategies |
| Wind | 0–250% | 5% | 1% | High values capture wind+storage strategies |
| Hydro | 0–(cap+10%) | 5% | 1% | Regional cap + 10% adder for run-of-river innovation potential. Extra hydro beyond existing cap is physics-only — NOT priced in Step 3. |
| Geothermal (CAISO only) | 0–20% | 5% | 1% | (existing + potential) / demand |

**Two-phase architecture**:

**Phase 1a — One-time coarse sweep per ISO** (cached to `data/step1-pfs-parquets/{ISO}_coarse_cache.parquet`):
- Generate all resource fraction combos at 5% step
- Score each combo once: `supply[h] = sum(frac[r] * profile[r][h])`, `score = sum(min(demand[h], supply[h]))`
- Cache `(resource fractions, score)` — reusable across ALL thresholds
- Run once per ISO; subsequent threshold work reads from cache

**Per-threshold work** (reads from coarse cache):
- Filter: `score >= target` → feasible combos (no-storage)
- Filter: `score >= target - 0.40` → storage zone → storage sweep
- Fine refinement at 1% step around frontier combos
- Save per-threshold results to `{ISO}_t{XX}_raw_pfs.parquet`

**What this removes**:
- `procurement_pct` column from all parquets (step1 → step2 → step3 → dashboard)
- `PROCUREMENT_BOUNDS` dict
- `vectorized_procurement_sweep()` function
- Binary search on procurement in storage sweep
- Cross-threshold pruning logic (unnecessary — all scores known from single sweep)

**Cost formula change**: Step 3 simplifies from `resource_frac/100 × procurement/100 × LCOE` to `resource_frac/100 × LCOE × demand_TWh`. Resource fractions directly represent generation volume.

**Persistent solution cache**: Results accumulated in `data/step1-pfs-parquets/` as per-ISO/threshold parquet files. Deduplication by (resource fractions, storage levels) key — no procurement dimension.

### 11.2 Edge Case Seed Mixes

Forced seed combos injected into coarse sweep to guarantee extreme-but-potentially-optimal strategies are evaluated. Now expressed as direct % of demand (not mix fractions):

- **High solar + storage**: solar=200%, wind=0%, CF=0%. Relies entirely on solar surplus + storage.
- **High wind + storage**: solar=0%, wind=200%, CF=0%. Same for wind-dominant regions.
- **Balanced high renewable**: solar=125%, wind=125%, CF=0%. Diversified variable generation.
- **Clean firm dominant**: CF=100%, solar=0%, wind=0%. Pure baseload.
- **CF + moderate solar**: CF=60%, solar=80%, wind=20%. Firm backbone + solar.
- **CF + moderate wind**: CF=60%, solar=20%, wind=80%. Firm backbone + wind.
- **Minimal firm**: CF=10%, solar=120%, wind=120%. Almost pure renewables.

Seeds filtered at runtime by regional hydro cap and geothermal cap (CAISO). Negligible compute cost, significant coverage improvement.

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
   - DAC costs ($300-1,100/ton, trajectory-dependent)
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
- **Primary benchmark**: DAC cost projected to the target year (see §7.3 revised trajectories)
  - 2025: $600-1,100/ton → grid dominates through ~97%+
  - 2035: $230-550/ton → grid dominates through ~93-95%
  - 2045: $130-375/ton → grid dominates through ~90-93%
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

**DAC cost with curtailed power**: If energy is the #1 DAC cost driver (~60% of total), curtailed power at $0-5/MWh could reduce DAC from $600-1,100/ton to $250-450/ton — making it competitive with grid decarbonization costs above 93-95% in most regions.

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

### 19.6 Geothermal Scope: Conventional Hydrothermal Only (CAISO)

**Assumption**: Geothermal resource is modeled only for CAISO, using conventional hydrothermal potential (identified by USGS). Enhanced Geothermal Systems (EGS) and other advanced geothermal technologies (closed-loop, superhot rock) are excluded from all ISOs.

**Why this matters**: EGS could theoretically unlock GW-scale firm clean power in regions with no conventional hydrothermal resource (PJM, NYISO, MISO, etc.). DOE's Enhanced Geothermal Shot initiative targets 2035 for cost-competitive EGS, and projects like Fervo Energy's Utah pilot (2026) and DOE's FORGE site are advancing the technology. If EGS reaches commercial scale, the firm clean power landscape changes substantially for non-CAISO regions.

**Justification**: This is a 2025 snapshot model. Conventional hydrothermal is the only geothermal technology commercially deployed at scale today, and CAISO is the only modeled ISO with meaningful resource (5.3 TWh existing + ~39 TWh USGS identified = ~44 TWh potential, capped at 5 GW). Non-CAISO ISOs sit on geology unsuitable for conventional hydrothermal — Appalachian basement rock (PJM/NYISO/NEISO) with 20–25°C/km thermal gradients, deep sedimentary basins (MISO/SPP), or early-stage pilots (ERCOT — Sage Geosystems). None have commercial-scale geothermal in the 2025 timeframe.

**EGS exclusion rationale**: EGS commercial deployment timelines (DOE targets 2035, industry consensus >2030 for first utility-scale projects) fall outside this model's 2025 snapshot scope. Including speculative EGS capacity would require forward-looking assumptions about cost learning curves, drilling success rates, and induced seismicity risk that are inconsistent with the model's empirical, current-year methodology. EGS is noted as a potential model enhancement for post-2030 analysis (see §21).

**Impact**: The model may overstate the long-term cost of firm clean power for non-CAISO regions if EGS achieves cost targets. For the 2025 snapshot, this is immaterial — no EGS capacity exists to procure today.

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

### 21.4 Offshore Wind — New Resource Dimension (Planned, Mar 2, 2026)

**Rationale**: Onshore wind is 22–25% CF with strong diurnal swing. Offshore wind at ~47% CF with a flat diurnal profile fundamentally changes the cost curve for 80%+ thresholds — less storage needed to cover overnight gaps, less VRE overbuild required. This is material for Atlantic ISOs.

#### 21.4.1 Capacity Caps (TWh)

| ISO | Cap (TWh) | Rationale |
|-----|-----------|-----------|
| NYISO | 37 | 9 GW pipeline (Empire Wind, Sunrise Wind, etc.) |
| NEISO | 37 | 9 GW capacity (Vineyard Wind, Revolution Wind, etc.) |
| PJM | 30 | NJ 7.5 GW mandated + DE/MD/VA pipeline |
| CAISO | 20 | ~5 GW (Morro Bay + Humboldt WEAs) |
| ERCOT | 0 | No meaningful offshore resource |
| MISO | 0 | No meaningful offshore resource |
| SPP | 0 | No meaningful offshore resource |

#### 21.4.2 Profile Shape (8760)

**Data source**: NREL NOW-23 (National Offshore Wind dataset) — 2 km grid, hourly wind speeds, 2000–2020 (21 years).

**Reference turbine**: IEA 15 MW (150 m hub height, 240 m rotor diameter). Power curve CSV from NREL/IEA Wind Task 37 GitHub.

**Build process**:
1. Extract wind speeds at 140 m + 160 m from NOW-23 at lease area coordinates
2. Interpolate to 150 m hub height (linear between heights)
3. Apply IEA 15 MW power curve (wind speed → capacity factor)
4. Apply loss stack: wake (~10%) × electrical (~2.5%) × availability (~5%) = **net ~83.4% of gross**
5. Average across 5 years (2016–2020) — same methodology as EIA onshore profiles
6. Normalize profile to sum = 1.0 (consistent with all other generation profiles)

**Representative coordinates per ISO**:

| ISO | Location |
|-----|----------|
| NYISO | NY Bight (Empire Wind lease area) |
| NEISO | Vineyard Wind area (south of Martha's Vineyard) |
| PJM | NJ lease areas (Atlantic Shores / Ocean Wind) |
| CAISO | Morro Bay WEA |

#### 21.4.3 Variability Calibration

The 5-year average preserves:
- **Seasonal swing** (winter peak in NE, summer peak in CA)
- **Multi-day weather patterns** (storm cycles, lull periods)
- **Flat diurnal envelope** (no day/night swing like onshore — key differentiator)
- **Realistic zero-generation hours** (~20% of hours for offshore)

It smooths:
- **Interannual anomalies** (e.g., whether 2018 was unusually windy)

**Validation targets**: Compare resulting annual CF against South Fork Wind actual (46.4%) and NREL ATB projections (~49%). This is the **exact same treatment** as onshore wind and solar — the offshore profile naturally shows higher CF and flatter diurnal, but still has weather-driven gaps. The optimizer handles it as VRE, not firm.

#### 21.4.4 Integration in the Pipeline

**Step 0** — New `step0_fetch_offshore_wind.py`:
- Fetches NOW-23 data via NREL API (requires API key from developer.nrel.gov)
- Applies IEA 15 MW power curve + loss stack
- Generates normalized 8760 profiles per ISO
- Output: `data/offshore_wind_profiles/` (one parquet per ISO)

**Step 1** — `offshore_wind` as new resource dimension:
- NYISO, NEISO, PJM, CAISO: 5D grid search (clean_firm, solar, wind, **offshore_wind**, hydro) — analogous to CAISO's 5D with geothermal
- CAISO becomes 6D (clean_firm, solar, wind, offshore_wind, hydro, geothermal)
- ERCOT, MISO, SPP: remain 4D (no offshore resource)
- Offshore wind grid levels: [0, 5, 10, 15, 20, 25, 30]% of demand (capped by ISO TWh limits above)

**Step 3** — Cost tables:
- New `OFFSHORE_WIND_LCOE` table (L/M/H by ISO). Range: ~$80–150/MWh (higher than onshore $30–95 reflecting fixed-bottom/floating costs)
- New `OFFSHORE_WIND_TX` transmission adder table (submarine cable + grid interconnection)
- Shares `Renewable Gen` sensitivity toggle pairing with solar + onshore wind

**Dashboard** — Toggle and display:
- Offshore wind appears as a distinct resource in mix charts (new color in `chart-colors.js`)
- Capacity caps shown in methodology page

#### 21.4.5 Cost Tables (Finalized, Mar 2, 2026)

**LCOE tables ($/MWh)** — shares `Renewable Gen` sensitivity toggle with solar + onshore wind:

| Level | NYISO | NEISO | PJM | CAISO (floating) | Sources |
|-------|-------|-------|-----|---------|---------|
| Low | 72 | 68 | 65 | 110 | Lazard v17 low, NREL ATB Advanced |
| Medium | 95 | 90 | 85 | 150 | BNEF 2025, NREL ATB Moderate |
| High | 125 | 118 | 112 | 200 | NREL FORCE model, supply-chain stress |

Regional hierarchy: PJM cheapest (shallowest, closest to shore, NJ 7.5 GW pipeline), NEISO mid (Vineyard Wind precedent, 51% CF), NYISO most expensive East Coast (NY Bight permitting + Jones Act), CAISO dramatically higher (floating, no US commercial experience).

**Transmission tables ($/MWh)** — submarine cable + offshore substation:

| Level | NYISO | NEISO | PJM | CAISO |
|-------|-------|-------|-----|-------|
| None | 0 | 0 | 0 | 0 |
| Low | 8 | 7 | 6 | 10 |
| Medium | 15 | 13 | 11 | 20 |
| High | 25 | 22 | 18 | 35 |

**Wright's Law learning curves** — two separate curves for fixed-bottom vs floating:

Fixed-bottom (NYISO, NEISO, PJM):
- FOAK: 1.15× High (Vineyard Wind era, supply chain stress). NYISO $144, NEISO $136, PJM $129.
- NOAK: Low values (post-learning equilibrium). NYISO $52-88, NEISO $50-85, PJM $50-82.
- Learning rate: ~8.8% per capacity doubling (NREL ATB 2024 Moderate). Global base: 83 GW.
- Window: L=(2026,2034), M=(2028,2038), H=(2032,2045)

Floating (CAISO):
- FOAK: 1.25× High (pre-commercial, no US floating experience). $250/MWh.
- NOAK: DOE Wind Shot aligned. $55-100/MWh.
- Learning rate: ~11.5% per capacity doubling (NREL ATB Moderate). Global base: 0.3 GW (nascent). Multiple doublings ahead.
- Window: L=(2029,2037), M=(2031,2042), H=(2035,2050)

**Capacity factor**: NYISO 0.49, NEISO 0.51, PJM 0.48, CAISO 0.43 (from NOW-23 profiles).
**Peak capacity credit**: 0.25 (higher than onshore wind 0.10 — flatter profile, less correlated with system peak).

#### 21.4.6 NOW-23 API Details (Research Complete, Mar 2, 2026)

**Regional endpoints** — NOW-23 uses separate API endpoints per region:

| Region | API Path | ISOs Served |
|--------|----------|-------------|
| North Atlantic | `offshore-north-atlantic-download` | NEISO |
| Mid Atlantic | `offshore-mid-atlantic-download` | NYISO, PJM |
| California | `offshore-ca-download` | CAISO |

**Base URL**: `https://developer.nrel.gov/api/wind-toolkit/v2/wind/{endpoint}.{format}`

**Key parameters**:
- `api_key` — free from developer.nrel.gov/signup
- `wkt` — WKT geometry, e.g., `POINT(-74.5 39.5)` (longitude first!)
- `attributes` — e.g., `windspeed_140m,windspeed_160m`
- `names` — year, e.g., `2020`
- `interval` — `60` for hourly
- `utc` — `true`
- `email` — required for async requests

**Available heights**: 10m, 20m, 40m, 60m, 80m, 100m, 120m, **140m**, **160m**, 180m, 200m, 220m, 240m, 260m, 280m, 300m, 400m, 500m. Both 140m and 160m confirmed available — linear interpolation to 150m hub height.

**Coverage**: 2000–2020 (21 years) for Atlantic regions; 2000–2019 (20 years) for California (API returns 400 for 2020+).

**Rate limits**: CSV format = 10,000 requests/day, 1/second. Each request = 1 point × 1 year. 5 years × 4 ISOs = 20 requests total — well within limits.

**Bulk alternative**: AWS S3 at `s3://nrel-pds-wtk/` (no account needed). Also accessible via NREL's HSDS service with `h5pyd` or `NREL-rex` packages.

#### 21.4.7 IEA 15 MW Power Curve (Research Complete, Mar 2, 2026)

**Source**: `turbine-models` Python package (`pip install turbine-models`), file `Offshore/IEA_Reference_15MW_240.csv`. Also on GitHub: `github.com/NREL/turbine-models`.

**Key turbine specs**:
| Parameter | Value |
|-----------|-------|
| Rated Power | 15 MW |
| Rotor Diameter | 240 m |
| Hub Height | 150 m |
| Cut-in Wind Speed | 3 m/s |
| Rated Wind Speed | 10.59 m/s |
| Cut-out Wind Speed | 25 m/s |
| IEC Class | IB |
| Design Cp | 0.489 |

**Power curve summary** (59 data points from 3–25 m/s):
- 3 m/s: 70 kW (cut-in)
- 7 m/s: 4,339 kW
- 10 m/s: 12,661 kW
- 10.59 m/s: ~14,995 kW (rated)
- 10.6–25 m/s: 14,995 kW (constant, pitch-controlled)
- 25 m/s: 14,998 kW → cut-out

Full CSV with 59 wind speed × power × Cp × thrust data points available in the installed package.

#### 21.4.8 Steps 4–7 + Dashboard Integration (Decided, Mar 3, 2026)

**Resource display order** (user-confirmed): Nuclear → Geothermal → Hydro → CCS → Offshore Wind → Onshore Wind → Solar → Battery 4 → Battery 8 → LDES → H2. Internal `RESOURCE_TYPES` in dispatch_utils.py keeps processing order; display order applied at presentation layer (Step 7 + dashboard JS).

**Color palette updates** (user-confirmed):
| Resource | Old Color | New Color | Hex |
|----------|-----------|-----------|-----|
| Nuclear (clean_firm) | Dark Navy `#1E3A5F` | Indigo 500 | `#6366F1` |
| CCS-CCGT | Cyan `#0891B2` | Slate | `#64748B` |
| Offshore Wind | (new) | Material Teal | `#009688` |
| Geothermal | Green `#10B981` | Ochre | `#B45309` |

**FEASIBLE_MIXES positional array**: 12 elements in display order:
`[clean_firm, geothermal, hydro, ccs_ccgt, offshore_wind, wind, solar, score, bat4, bat8, ldes, h2]`

**Resource cap integration scope** (user-confirmed): Geothermal, CCS, and offshore wind TWh caps propagated into:
- Scenario A (step5_scenario_a_consequential.py) — floor ratchet respects caps as upper bounds
- Scenario B (step5_scenario_b_hourly.py) — hourly matching cap enforcement
- Scenario comparison (step5_scenario_comparison.py)
- Procurement Strategies 1–3 (step6_5_strategy1/2/3)
- Track 2 New-Build (step3_track_nb_ctr.py)
- Track 3 Cost-to-Replace (step3_track_nb_ctr.py)

**CCS cap table** (geological sequestration storage, from USGS/NETL):
| ISO | Cap (TWh) | Rationale |
|-----|-----------|-----------|
| ERCOT | 85 | Gulf Coast saline formations + depleted O&G reservoirs |
| PJM | 120 | Appalachian Basin + Midcontinent saline formations |
| MISO | 95 | Illinois Basin + Gulf Coast formations |
| SPP | 110 | Anadarko Basin + Permian Basin saline formations |
| NYISO | 15 | Limited NY/NJ offshore saline capacity |
| NEISO | 10 | Very limited NE geological storage |
| CAISO | 0 | No significant identified storage (seismic risk) |

**Dashboard cap export**: `RESOURCE_CAPS` JS constant in shared-data.js containing all three cap dicts (offshore_wind, ccs_ccgt, geothermal).

**dispatch_utils.py changes**:
- `RESOURCE_TYPES` expanded from 5 → 6 (add `offshore_wind`)
- `OFFSHORE_ISOS`, `OFFSHORE_WIND_CAP_TWH`, `CCS_CAP_TWH`, `GEOTHERMAL_CAP_TWH` constants
- `get_supply_profiles()` loads offshore wind profile (zeros for non-offshore ISOs)
- `build_supply_matrix()` builds (6, H) matrix
- `reconstruct_hourly_dispatch()` adds matched/surplus offshore_wind arrays
- `CACHE_VERSION` → v3 (old v2 caches rebuilt)

**Backward compat**: All parquet loading defaults `mix_offshore_wind` to 0 when column is missing.

#### 21.4.8 Implementation Approach (Decided)

**Recommended approach** (simplest, no HSDS complexity):
1. Use developer.nrel.gov CSV endpoint — one request per point per year
2. Request `windspeed_140m` + `windspeed_160m` at `interval=60`
3. Linear interpolation to 150m hub height
4. Apply IEA 15MW power curve via `numpy.interp()` on the 59-point CSV
5. Apply loss stack (wake 10% × electrical 2.5% × availability 5% = net 83.4%)
6. Average 5 years (2016–2020), normalize to sum = 1.0

**Python packages needed**: `requests`, `numpy`, `pandas` (all already in the project). The `turbine-models` package provides the power curve CSV but isn't needed at runtime — we'll embed the 59-point curve directly in the script.

#### 21.4.9 Blockers

1. **NREL API key** required for NOW-23 data access — sign up at developer.nrel.gov/signup
2. **Cost table finalization** — need NREL ATB 2024 offshore wind LCOE by region + Lazard cross-check
3. **Step 1 compute impact** — adding a 5th/6th dimension increases grid search combinatorics significantly. May need aggressive pruning or adaptive grid for offshore ISOs.

#### 21.4.10 Why This Matters

Offshore wind at 47% CF with flat diurnal is a **qualitatively different resource** from onshore wind:
- Onshore: low CF, strong diurnal → needs massive overbuild + storage for 80%+ matching
- Offshore: high CF, flat diurnal → approaches dispatchable VRE characteristics
- At 90%+ thresholds, offshore wind could displace significant clean firm / storage need in Atlantic ISOs
- The cost question is whether the $80–150/MWh LCOE premium over onshore ($30–95) is offset by reduced storage and overbuild needs — this is exactly what the optimizer will answer

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

### 22.9 Delta RA Approach — Calibrated Gas Backup (Feb 25, 2026)

**Problem**: The previous RA formula `gas_needed = max(0, ra_peak - clean_peak) / GAF` computed clean_peak from energy allocations (`proc * pct / 100 * avg_demand_mw * CC`), which conflates average generation MW with installed capacity. This systematically underestimated clean peak contribution (by 2-4x for solar/wind) because it didn't convert energy → installed MW using capacity factors. Result: ERCOT 2025 showed ~94 GW gas needed when only 55 GW exists.

**Fix**: Delta RA approach calibrated to 2025 reality:
1. At base year (2025): `gas = EXISTING_GAS_CAPACITY_MW` (calibrated to actual installed fleet)
2. Compute `EXISTING_CLEAN_PEAK_MW` from 2025 fleet using `avg_mw / capacity_factor * capacity_credit`
3. Compute `GAS_RAW_2025 = max(0, RA_peak - EXISTING_CLEAN_PEAK_MW) / GAF` as theoretical baseline
4. For any scenario: `gas_raw = max(0, ra_peak_grown - total_clean_peak) / GAF`
5. `gas_delta = gas_raw - GAS_RAW_2025`
6. `total_gas = max(0, EXISTING_GAS + gas_delta)`

**New-build peak uses capacity factor conversion**: `installed_mw = avg_generation_mw / CF[resource][iso]` then `peak_mw = installed_mw * CC[resource]`. This properly accounts for wind's low CF (high installed per MWh) and solar's low CF.

**Resource capacity factors (EIA Form 923, eGRID 2022-2024)**:
| Resource | CAISO | ERCOT | PJM | NYISO | NEISO |
|----------|-------|-------|-----|-------|-------|
| Nuclear  | 0.90  | 0.93  | 0.93| 0.90  | 0.90  |
| Solar    | 0.28  | 0.24  | 0.17| 0.15  | 0.15  |
| Wind     | 0.25  | 0.38  | 0.30| 0.28  | 0.30  |
| CCS-CCGT | 0.85  | 0.85  | 0.85| 0.85  | 0.85  |
| Hydro    | 0.40  | 0.30  | 0.35| 0.40  | 0.40  |

**Peak demand growth**: Peak scales with demand growth factor (`peak_grown = PEAK_2025 * gf`).

**Result**: ERCOT 50% at 2030 Medium growth now shows ~71 GW gas (down from >100 GW), consistent with real-world expectations of 65-70 GW for modest clean energy expansion from a 46% clean baseline with ~53 GW existing gas.

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

### 22.7 ≥99.99% Hourly Match Asymptote — Literature Review & Procurement Bounds

**Decision (Feb 2026):** Top threshold lowered from 100% to ≥99.99%. True 100% hourly matching is physically unreachable due to float precision and dispatch constraints. The effective gate maps ≥99.99% → 99.5% to capture near-perfect mixes. This makes the threshold honest — we label what we can actually achieve.

**Key literature findings:**
- NREL (Cole et al., 2021, Joule): Marginal abatement cost 99%→100% = **$930/ton** — 15× the average cost of the full 100% target. Nonlinear in all 22 sensitivities tested.
- Riepin & Brown (2024, Energy Strategy Reviews): 98% CFE = 54% premium over annual matching. 100% doubles costs again. With clean firm + LDES, 100% premium drops to just 15%.
- Peninsula Clean Energy MATCH Model (2023): 99%→100% requires **34% more supply**, +10% portfolio cost. 0%→99% costs only +2%.
- Budischak et al. (2013, J. Power Sources): Cost-optimal 99.9% requires ~280% nameplate capacity. "Least cost solutions yield seemingly-excessive generation capacity."
- WattTime: 100% hourly matching may require PPAs for **up to 400%** of annual consumption.

**Granularity consensus:** The 90–≥99.99% zone needs 2.5% resolution minimum. Our threshold set (90, 92.5, 95, 97.5, 99, ≥99.99) is well-aligned with literature practice.

**Procurement bound assessment:**
- Current bound: 200% of demand
- Actual usage at 99%: max 135% (CAISO), 130% (NYISO), 125% (NEISO), 123% (PJM), 118% (ERCOT)
- ≥99.99% threshold: 0 feasible scenarios found (all ISOs) at 200% bound
- Max hourly match achieved: 99.6% (PJM at 123% procurement)
- **Decision**: If rerunning for ≥99.99%, increase upper bound to **250%** based on literature support (Budischak 280%, WattTime 400%). The 200% bound is sufficient for ≤99% targets.

**Archetype diversity in cache:**
- 46–70 unique resource mix archetypes per ISO across all thresholds
- Only 4–14 unique mixes per threshold (massive redundancy across 5,832 scenarios)
- Cache comprehensively covers the feasible solution space — new constraint runs can seed from existing archetypes rather than cold-start

### 22.9 Step 1 PFS Improvement Opportunities (Feb 21, 2026)

**Constraint: No changes may sacrifice the ability to find the full PFS.** All improvements below are backward-compatible — they improve speed and/or coverage without changing the feasible space definition or dispatch physics.

**Post-process script**: `postprocess_storage_resweep.py` — standalone Numba parallel re-sweep that runs between Step 1 and Step 2. Demonstrates patterns 1, 4, and 8 below. Uses `@njit(parallel=True)` with `prange` to batch-evaluate near-miss mixes across CPU cores. Checkpoints to `data/resweep_checkpoints/resweep_progress.parquet` after each ISO×threshold.

#### 1. Numba Parallel Storage Sweep (High Impact — 4-8× speedup)
**Current**: Step 1 Phase 1b evaluates storage combos sequentially per mix — one mix at a time through 342 storage configs × N procurement levels.
**Improvement**: Wrap `_score_with_all_storage` in `@njit(parallel=True)` with `prange` across mixes. All near-miss mixes at a single (procurement, storage) config are evaluated simultaneously across CPU cores.
**Pattern**: `batch_score_storage()` in `postprocess_storage_resweep.py` — takes `(demand, supply_rows[N, 8760], procurement, N, storage_params...)`, returns `scores[N]` via `prange`.
**Impact**: Phase 1b is 60-80% of Step 1 runtime. Multi-core parallel cuts this proportional to available cores (typically 4-8× on modern machines).

#### 2. Consistent Scoring Metric (Correctness)
**Current**: Two different scoring metrics used within Step 1:
- Phase 1a (no storage): `np.sum(np.minimum(supply / demand, 1.0)) / H` — hourly average match fraction (weights all hours equally)
- Phase 1b (with storage): `sum(min(demand[h], supply[h]))` — total energy match fraction (weights by demand magnitude)
**Problem**: These produce different scores for the same mix. A mix could pass the no-storage check but fail the storage check (or vice versa) at the same threshold. The PFS mixes `hourly_match_score` column contains a mix of both metrics.
**Fix**: Unify to the energy metric (`sum(min(demand, supply))`) everywhere. The energy metric is more physically meaningful — it answers "what fraction of total demand energy is met?" rather than "what fraction of hours have some matching?"
**Risk**: None to PFS completeness. May change which mixes are classified as near-miss in Phase 1a, but the `batch_score_no_storage()` kernel in the post-process script shows how to do this efficiently.

#### 3. Wider Near-Miss Window (Coverage)
**Current**: Step 1 uses 15% near-miss window (`target - 0.15`).
**Improvement**: Expand to 25%. The post-process re-sweep with 25% window found 471K+ new solutions at CAISO 50% alone — mixes that scored 25-50% without storage but reached 50%+ with storage.
**Trade-off**: More near-miss mixes → more storage evaluations → longer runtime. With parallel kernels (improvement 1), the marginal cost is acceptable.

#### 4. Procurement Binary Search (2-5× faster per mix)
**Current**: Linear sweep of procurement levels — typically 30-50 evaluations per (mix, storage) combo with early stopping.
**Improvement**: Binary search for minimum feasible procurement: O(log₂ N) ≈ 5-6 evaluations instead of O(N) ≈ 30.
**Prerequisite**: Score is monotonically increasing with procurement (true by construction — more supply always helps).
**Integration**: Use batch evaluation at max procurement first to identify feasible mixes, then binary search per mix for the minimum.

#### 5. Full Storage Grid on Phase 2 Refinement (Coverage)
**Current**: Phase 2 (1% resolution refinement) only tries `[2, 5, 10]` for battery4 and battery8 levels.
**Improvement**: Use the full `[0, 2, 5, 8, 10, 15, 20]` grid, matching Phase 1b. Catches refinement mixes that need higher storage levels (15-20%) to become feasible.
**Cost**: Modest — refinement mixes are few (neighborhoods of Phase 1 archetypes), so the extra storage combos add seconds, not minutes.

#### 6. Two-Phase Adaptive Storage Sweep (Feb 21, 2026) — IMPLEMENTED

**Replaces** the previously rejected "adaptive storage tiers" approach. The two-phase approach does NOT skip storage levels — it sweeps ALL levels in both phases, differing only in granularity.

**Phase 1 — Coarse sweep (0.25% steps):** Identifies saturation range per ISO.
- bat4: `[0, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 2.5]`
- bat8: `[0, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]`
- LDES: `[0, 0.5, 1.0, 1.5, 2.0, 2.5, 5, 8, 10]`
- After Phase 1, analyzes max used levels across all thresholds to find saturation point.

**Phase 2 — Fine sweep (0.05% steps):** Fills in granularity within the saturation range.
- bat4: `[0, 0.05, 0.10, ..., max_bat4 + 0.25%]` — e.g., ERCOT: 21 levels (0-1.00%)
- bat8: `[0, 0.05, 0.10, ..., max_bat8 + 0.25%]` — e.g., ERCOT: 36 levels (0-1.75%)
- LDES: Same coarse levels (less sensitive to granularity)
- Phase 1 coarse solutions merged with Phase 2 fine solutions (deduped by full key).

**Results per ISO:**
- ERCOT: 2,033,961 solutions (Phase 1: 226K coarse → Phase 2: 1.8M fine), 11 min
- CAISO: ~1.8M solutions (saturation: bat4=1.00%, bat8=2.00%), ~25 min
- Output: Per-ISO/threshold parquet files (`data/step1-pfs-parquets/{ISO}_t{XX}_raw_pfs.parquet`)

**Scientific rigor preserved:** Every level from 0 to saturation+margin is swept at 0.05% resolution. No levels are skipped or short-circuited. The coarse phase just identifies WHERE the fine sweep should focus.

#### 7. Cross-Threshold Solution Injection (Coverage)
**Current**: Step 1 uses "cross-threshold pollination" to track proven-feasible mixes and skip their storage sweep at higher thresholds. But doesn't inject these solutions — just avoids redundant work.
**Improvement**: Inject known-feasible (mix, storage) configs from lower thresholds as seeds for higher thresholds' procurement sweep. A mix feasible at 85% with battery=10 is likely feasible at 87.5% with battery=15 or higher procurement — the seed gives a starting point for the procurement search.

#### 8. Vectorized Phase 1a Procurement (Minor)
**Current**: Phase 1a loops over procurement levels sequentially, computing vectorized scores at each level.
**Improvement**: Batch multiple procurement levels into a single evaluation: `supply_batch[N, P, H] = supply_rows[:, None, :] * proc_array[None, :, None]`. Score all (mix × procurement) combinations in one vectorized operation.
**Caveat**: Memory-intensive — N_mixes × N_procurement × 8760 × 8 bytes. For 20K mixes × 50 proc levels = 70 GB. Only viable with chunking or for small grids.

#### Priority Ranking
| # | Improvement | Impact | Effort | Risk to PFS |
|---|---|---|---|---|
| 1 | Numba parallel storage | 4-8× speedup | Low (pattern exists) | None |
| 2 | Consistent scoring metric | Correctness | Low | None |
| 3 | Wider near-miss window | More solutions | Low | None |
| 4 | Binary search procurement | 2-5× per-mix speedup | Medium | None |
| 5 | Full refinement storage grid | More solutions | Low | None |
| 6 | Two-phase adaptive sweep | 0.05% granularity | Medium | None — **IMPLEMENTED** |
| 7 | Cross-threshold injection | More solutions | Medium | None |
| 8 | Vectorized Phase 1a | Minor speedup | Low | None (memory) |

**Implementation path**: Improvements 1-3 can be applied to `step1_pfs_generator.py` directly by copying kernels from `postprocess_storage_resweep.py`. Improvements 4-7 require refactoring the `optimize_threshold()` function. None require re-running from scratch — all are refinements to existing logic.

#### Implemented (Feb 21, 2026)

| # | Improvement | Status | Notes |
|---|---|---|---|
| 1 | Numba parallel storage | **Done** | `_batch_score_storage()` and `_batch_score_no_storage()` added with `prange`. JIT warmup includes batch kernels. |
| 2 | Consistent scoring metric | **Done** | Phase 1a and Phase 2 now use `sum(min(demand, supply))` (total matched energy), consistent with `_score_with_all_storage` base_matched. Previously used `sum(min(supply/demand, 1.0)) / H` (hourly average fraction) — different metric. |
| 5 | Full refinement storage grid | **Done** | Phase 2 now uses full `batt_levels × batt8_levels × ldes_levels` grid (matching Phase 1b) instead of limited `[2, 5, 10]` for battery4/battery8 only. LDES now modeled in Phase 2. |
| 3 | Wider near-miss window | **Done** | Phase 1a: 15% → 25%. Phase 2: 10% → 15%. More near-miss mixes enter storage testing → more feasible solutions found. |
| 4 | Binary search procurement | **Done** | All phases (1a, 1b, Phase 2) now use binary search O(log₂ N) instead of linear sweep O(N). Phase 1b also checks max procurement first and skips infeasible (mix, storage) combos entirely. |
| 8 | Vectorized Phase 1a procurement | **Done** | `batch_hourly_scores()` classifies all mixes at max procurement in one matrix multiply. Only feasible mixes enter per-mix binary search. Eliminates per-mix × per-proc scoring loop. |

**Code cleanup (Feb 21, 2026)**:
- **Removed 3 redundant scoring functions**: `_score_hourly`, `_score_with_battery`, `_score_with_both_storage` — all subsets of `_score_with_all_storage` (passing 0 for unused storage types skips those phases via capacity guards)
- **Removed redundant Phase 1b battery4-only loop**: Was a separate sweep before the full triple storage loop. Now the full triple loop covers all non-zero storage combos including battery4-only (no `b8p == 0 and lp == 0` skip)
- **Vectorized `_average_profiles()`**: Replaced nested Python loop (O(N×8760)) with `np.mean(np.array(profiles), axis=0)`
- **Vectorized `get_supply_profiles()` clean_firm**: Replaced hour-by-hour Python loop with `np.repeat(month_cfs, month_hours)`
- **Vectorized `get_supply_profiles()` solar DST correction**: Replaced 365×24 nested Python loop with numpy boolean mask across all 8760 hours
- **Vectorized `get_supply_profiles()` post-processing**: Replaced list comprehension `[max(0, v) for v in p]` with `np.maximum(arr, 0.0, out=arr)`
- **Vectorized `generate_4d_combos()`**: Replaced triple-nested Python loop with `np.meshgrid` + vectorized filter

| 7 | Cross-threshold solution injection | **Done** | Solutions from lower thresholds that score >= current threshold are injected directly into higher threshold candidate lists. Combined with existing `cross_skip` (Phase 1b skips re-testing known-feasible mixes), this avoids redundant computation while ensuring all qualifying solutions propagate upward. Done-threshold parquets load full solutions for seeding. |

**Rejected**:
- ~~Item 6: Adaptive storage tiers~~ — **Deleted from consideration**. Would skip higher-tier storage configs for mixes already feasible with simpler configs, undermining scientific rigor by missing storage diversity needed for Step 3 cost optimization.

### Clean Firm FOAK→NOAK Learning Curves in Step 3 Demand Growth (Mar 1, 2026)

**Decision**: Integrate Wright's Law FOAK→NOAK learning curves directly into Step 3 cost optimization (Phase 2 — demand growth sweep). Each threshold's SBTi target year determines the learning-curve-adjusted cost for new-build clean firm technologies. This replaces the static-cost model where all demand growth years used the same 2025 LCOE snapshot.

**Problem**: Step 3 currently uses identical LCOEs for all years (2025-2050). A buyer in 2030 faces FOAK pricing for nuclear/CCS/LDES, but the optimizer prices it at NOAK. This systematically underprices new-build clean firm in early years, making storage non-competitive at any threshold — inconsistent with real-world storage deployment.

**Scope**: Phase 2 (demand growth sweep) only. Phase 1 (base year 2025) remains at static L/M/H costs.

**Design choices**:
1. **Step 3 integration** (not post-hoc repricing) — learning curves must be inside the optimization to change which mixes are selected, not just reprice fixed mixes.
2. **Paired adoption speed + NOAK optimism** (3 combos, not 6) — each technology's L/M/H toggle controls both NOAK endpoint and adoption speed. Correlated in reality: fast deployment → more learning → lower NOAK. Avoids scenario count explosion.
3. **Technologies with learning curves**: Nuclear new-build, CCS-CCGT, LDES (100hr iron-air), Green H2, Geothermal (CAISO only), Battery 4hr, Battery 8hr (shallow curves, see Storage Cost Fix section below). Solar/wind already mature — static costs.
4. **Uprates unchanged** — $15/25/40 (L/M/H), no learning curve. Existing fleet, sunk cost.

**FOAK Cost Tables** (first-of-a-kind, pre-learning, single value per ISO — same for all L/M/H):

Nuclear new-build FOAK ($/MWh) — ~1.25× High (Vogtle-era pricing):
| ISO | FOAK |
|-----|------|
| CAISO | 175 |
| ERCOT | 169 |
| PJM | 200 |
| NYISO | 212 |
| NEISO | 206 |
| MISO | 194 |
| SPP | 175 |

CCS-CCGT FOAK 45Q ON ($/MWh) — ~1.20× High:
| ISO | FOAK |
|-----|------|
| CAISO | 138 |
| ERCOT | 110 |
| PJM | 122 |
| NYISO | 154 |
| NEISO | 146 |
| MISO | 115 |
| SPP | 106 |

CCS-CCGT FOAK 45Q OFF ($/MWh) — ~1.20× High:
| ISO | FOAK |
|-----|------|
| CAISO | 173 |
| ERCOT | 145 |
| PJM | 157 |
| NYISO | 188 |
| NEISO | 181 |
| MISO | 150 |
| SPP | 140 |

Geothermal FOAK (CAISO only): $150/MWh (~1.35× High)

LDES FOAK ($/MWh-cap, annualized capacity cost) — ~1.40× High:
| ISO | FOAK |
|-----|------|
| CAISO | 1.40 |
| ERCOT | 1.20 |
| PJM | 1.32 |
| NYISO | 1.55 |
| NEISO | 1.48 |
| MISO | 1.26 |
| SPP | 1.23 |

Green H2 FOAK ($/MWh-cap, annualized capacity cost) — ~1.30× High:
| ISO | FOAK |
|-----|------|
| CAISO | 5.32 |
| ERCOT | 4.69 |
| PJM | 5.04 |
| NYISO | 5.85 |
| NEISO | 5.58 |
| MISO | 4.82 |
| SPP | 4.60 |

**Learning Curve Parameters** (per toggle level):

| Level | Adoption | FOAK Start | NOAK Year | Duration | NOAK Endpoint | Wright's Law Exponent |
|-------|----------|------------|-----------|----------|---------------|----------------------|
| L (Optimistic/Fast) | Fast | 2028 | 2036 | 8 years | Low cost table | 0.6 |
| M (Central) | Central | 2030 | 2040 | 10 years | Medium cost table | 0.6 |
| H (Pessimistic/Slow) | Slow | 2036 | 2048 | 12 years | High cost table | 0.6 |

**Unified timelines for all clean firm technologies** (simplified from prior per-technology overrides — now all clean firm techs share the same L/M/H learning schedule):

| Technology | Toggle | L FOAK→NOAK | M FOAK→NOAK | H FOAK→NOAK |
|------------|--------|-------------|-------------|-------------|
| Nuclear new-build | Firm | 2028→2036 | 2030→2040 | 2036→2048 |
| CCS-CCGT | CCS | 2028→2036 | 2030→2040 | 2036→2048 |
| Geothermal | Geo | 2028→2036 | 2030→2040 | 2036→2048 |
| LDES | LDES | 2028→2036 | 2030→2040 | 2036→2048 |
| Green H2 | LDES | 2028→2036 | 2030→2040 | 2036→2048 |
| Battery 4hr | Batt | 2025→2030 | 2026→2032 | 2027→2035 |
| Battery 8hr | Batt | 2025→2030 | 2026→2032 | 2027→2035 |

**Year-adjusted cost formula**: `cost(year) = FOAK × (1 - frac) + NOAK × frac` where `frac = learning_fraction(year, foak_start, noak_year)`.

**`learning_fraction(year, foak_start, noak_year)`**:
- Before `foak_start`: 0.0 (pure FOAK)
- After `noak_year`: 1.0 (full NOAK)
- During learning: `((year - foak_start) / (noak_year - foak_start)) ** 0.6`
- Exponent 0.6 produces Wright's Law concave ramp: steep initial drop (first 40% of cost reduction in first 30% of timeline), then asymptotic approach to NOAK.

**Example impact (PJM, 70% threshold = year 2035, Firm=M, CCS=M)**:
- Nuclear FOAK=$200, NOAK_M=$105, frac=`((2035-2030)/(2040-2030))^0.6 = 0.50^0.6 ≈ 0.66` → year cost = $200×0.34 + $105×0.66 = **$137/MWh**
- CCS FOAK 45Q ON=$122, NOAK_M=$79, frac=`((2035-2030)/(2040-2030))^0.6 = 0.50^0.6 ≈ 0.66` → year cost = $122×0.34 + $79×0.66 = **$93/MWh**
- Battery 4hr (static): **$98/MWh**
- **Battery now competitive with nuclear new-build at 70% threshold**

**Implementation**: Modify `precompute_all_prices()` in `step3_cost_optimization.py` to accept optional `target_year` parameter. In Phase 2, compute a year-specific price matrix for each unique DG year (14 matrices × ~0.01s each = negligible overhead). No new scenario dimensions — learning curves are embedded in existing L/M/H toggles.

**Supersedes**: Line 1483 of this file ("Scope: PP3 scenario comparison only. Step 3 cost optimization is NOT modified"). Step 3 DG sweep now uses learning curves. Step 6 scenario comparison curves remain unchanged (their own timeline parameters are for the A/B strategy comparison, not the core optimization).

**No compute cost increase**: Same 5,832/17,496 sensitivity combos per threshold. Only the price lookup changes for DG years. Phase 1 (base year) unchanged. Step 1/2 not affected.

### Storage Cost Fix: LCOS → Annualized Capacity Cost (Mar 1, 2026)

**Bug**: Step 3 priced storage as `bat_pct / 100.0 × LCOS`, where `bat_pct/100` is a normalized energy capacity parameter (energy capacity as fraction of avg hourly demand MWh). LCOS is $/MWh of *discharged* energy. The product gives the wrong units — it treats the capacity sizing parameter as an annual dispatch fraction, overpricing storage by 10-50× depending on utilization assumptions. Example: bat_pct=3, LCOS=$102 → $3.06/MWh, but actual annual cost of a 900 MW/3.6 GWh battery for CAISO is ~$0.28/MWh of demand.

**Fix**: Replace LCOS tables with annualized capacity cost per MWh-cap, matching the coefficient model:
```
price = 1000 × (CAPEX_kWh × CRF + FOM_kW / duration) / 8760 × regional_mult
```
Now `bat_pct / 100.0 × price` directly gives the annual fixed cost of that storage capacity as a fraction of total demand cost ($/MWh of demand). Eliminates cycling assumptions — prices pure capacity, not utilization.

**Financial parameters**:
- WACC: 8%
- Battery lifetime: 20 years → CRF = 0.10185
- LDES lifetime: 25 years → CRF = 0.09368
- H2 lifetime: 20 years → CRF = 0.10185

**CAPEX per kWh** (NREL ATB 2024):
| Technology | Low | Medium | High | Duration | FOM ($/kW-yr) |
|-----------|-----|--------|------|----------|---------------|
| Battery 4hr | $115 | $140 | $165 | 4 hr | $25 |
| Battery 8hr | $95 | $120 | $145 | 8 hr | $25 |
| LDES (iron-air) | $30 | $50 | $80 | 100 hr | $5 |
| Green H2 | $150 | $220 | $310 | 168 hr | $8 |

**Regional multipliers**: Derived from existing LCOS ratio (normalize to ERCOT=1.0), baked into capacity prices. TX adders set to $0 for all storage types.

**LCOS cross-check** (validates capacity prices against known LCOS benchmarks):
- Battery 4hr Med @ 365 cycles/yr = $121/MWh (Lazard 2024 range: $115-220)
- Battery 8hr Med @ 300 cycles/yr = $107/MWh
- LDES Med @ 50 cycles/yr = $95/MWh

**Battery learning curves** (updated March 2026 — Wright's Law with NREL-calibrated trajectories):
- Direction: LCOE_TABLES (2025 starting) → NOAK_BATTERY (terminal floor). Reverse of other techs.
- NOAK fractions: Low=50%, Med=56%, High=80% of starting cost. Calibrated to NREL 2050 projections.
- Timelines: bat4 L=(2025,2042), M=(2025,2048), H=(2025,2050). bat8 2yr faster.
- Exponent: 0.6 (concave ramp). Net effect: meaningful decline over 20+ year horizon, not the old shallow 5yr curve.
- See §5.5 for full trajectory table.

**Storage FOAK tables** ($/MWh-cap):
- Battery 4hr/8hr: FOAK = High (no premium — batteries at scale). Not used by learning curves (batteries use LCOE→NOAK direction).
- LDES: 1.40× High capacity cost per ISO
- H2: 1.30× High capacity cost per ISO

**Propagated to**: `step3_cost_optimization.py`. Other files (`scenario_common.py`, `step6_scenario_comparison.py`) may need separate update if they have independent LCOE_TABLES copies.

**Supersedes**: Prior LCOS values in LCOE_TABLES for battery, battery8, ldes, h2. Line 3764 of this file updated — batteries now have learning curves too (previously listed as "already mature — static costs").
