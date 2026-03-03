# Hourly CFE Cost Optimizer

An optimization engine and interactive dashboard for analyzing **hourly clean energy procurement strategies** across seven US ISO regions. The optimizer finds the least-cost mix of clean firm (nuclear/geothermal), solar, wind, hydro, CCS-CCGT, battery, and LDES to meet hourly demand at various Clean Energy Matching (CEM) targets (50%–≥99.99%), then visualizes results in a scrollytelling dashboard with 20+ interactive pages.

## Regions Covered

| ISO | Region | Notes |
|-----|--------|-------|
| CAISO | California | Includes geothermal as 5th physics dimension |
| ERCOT | Texas | |
| PJM | Mid-Atlantic | |
| NYISO | New York | |
| NEISO | New England | Winter gas pipeline constraint modeled |
| MISO | Midwest | |
| SPP | South-Central | |

## Quick Start

```bash
cd dashboard
python -m http.server 8000
# Open http://localhost:8000/index.html
```

## 7-Step Pipeline

The optimizer runs as a 7-step pipeline. Step 1 is expensive (hours). Steps 2–4 are cheap (seconds to minutes). Step 5 builds a dispatch cache. Step 6 runs parallel analytics. Step 7 exports dashboard data.

### Core Pipeline (Steps 1–4)

| Step | Script(s) | What It Does | When to Re-run |
|------|-----------|-------------|---------------|
| **Step 0** | `step0_*.py` (8 scripts) | **Data Fetch/Prep** — EIA hourly profiles, eGRID emissions, LMP data, DST/UTC fixes, MISO/SPP consolidation. | When source data updates. |
| **Step 1** | `step1_pfs_generator.py` (monolithic) or `step1a` → `step1b` → `step1c` → `step1d` (modular) | **PFS Generator** — Generates the Physics Feasible Space. 4D/5D adaptive grid search × procurement × battery × LDES. Two-phase storage sweep (coarse → fine). Output: `data/step1-pfs-parquets/` + `data/step1d-storage-parquets/`. | Only if dispatch logic, generation profiles, or demand curves change. |
| **Step 2** | `step2_efficient_frontier.py` + `step2_5_expand_ef_for_floors.py` | **Efficient Frontier** — Extracts non-dominated mixes from PFS. Reads both step1 and step1d parquets. Optional EF expansion for Scenario A floor constraints. Output: `data/step2-ef-parquets/`. | Only if PFS or filtering criteria change. |
| **Step 3** | `step3_cost_optimization.py` + `step3_track_nb_ctr.py` | **Cost Optimization** — Track 1 baseline: vectorized cross-eval of all EF mixes under 5,832 sensitivity combos (17,496 CAISO). Track 2 (newbuild) + Track 3 (cost-to-replace). Merit-order tranche pricing. Demand growth sweep with FOAK→NOAK learning curves. Output: `data/step3-cost-opt-parquets/`. | When cost assumptions, LCOE tables, or sensitivity toggles change. |
| **Step 4** | `step4_gas_ccs_adjustement.py` | **Gas/CCS Adjustments** — NEISO winter gas pipeline constraint (+$13.13/MWh CCS adder), 45Q correction ($27.5/MWh), gas capacity backup & resource adequacy (15% RA margin), CCS vs LDES crossover. Output: `data/step4-gas-ccs-parquets/`. | When Step 3 outputs change. |

### Step 1 Sub-Pipeline (Modular)

The monolithic `step1_pfs_generator.py` has been decomposed into modular scripts for CI/CD:

| Script | What It Does |
|--------|-------------|
| `step1a_generate_mixes.py` | Generates all resource fraction combos (4D/5D grid). |
| `step1b_score_mixes.py` | Scores mixes against hourly demand profiles → `coarse_cache.parquet`. |
| `step1c_build_pfs.py` | Mines Physics Feasible Space from scored DB per threshold. |
| `step1d_storage_refinement.py` | Fills storage exploration gaps — intermediate battery/LDES levels that 1c's coarse grid missed. Output: `data/step1d-storage-parquets/`. |

### Step 5: Dispatch Cache + Independent Analysis

| Script | What It Does | Cache Dependency |
|--------|-------------|-----------------|
| `step4_build_dispatch_cache.py` | **Run first.** Pre-computes 8,760-hour dispatch for all unique mixes. Versioned NPZ cache (v2) with per-resource matched/surplus + charge profiles. | Creates cache |
| `step5_export_track_results.py` | Exports track parquets (NB + CTR) to `track_results.json` for dashboard. | None |
| `step5_analyze_tracks.py` | Track result analysis: cost envelopes (P10/P50/P90), resource mix differentials. | None |

### Step 6: Dispatch-Cache-Dependent Analysis

All Step 6 scripts read from the Step 5 dispatch cache. They can run in parallel after the cache is built.

| Script | What It Does |
|--------|-------------|
| `step5_compute_co2.py` | **CO₂ dispatch-stack model.** Merit-order fuel retirement (coal → oil → gas). Coal/oil capped at 2025 TWh. Demand-growth-aware. |
| `step5_compute_mac_stats.py` | **MAC statistics.** 6 metrics: average fan (P10/P50/P90), stepwise marginal, monotonic envelope, path-constrained. ANOVA sensitivity decomposition. Crossover vs DAC/SCC/ETS. |
| `step5_compute_lmp_prices.py` | **Synthetic LMP.** 8,760-hour dispatch reconstruction; hourly LMP from merit-order fossil stack. All 7 ISOs with calibrated price models. |
| `step5_compute_optimal_targets.py` | **Optimal CFE targets.** Marginal MAC × DAC crossover via PCHIP spline. 3×3 grid-cost × DAC-scenario matrix. No-regrets resource analysis. |
| `step5_compress_day_profiles.py` | **Compressed day profiles.** 24-hour representative day from dispatch cache. Falls back to live compute on cache miss. |
| `step5_consequential_deployment_queue.py` | **Consequential queue.** Cross-regional deployment path under consequential accounting. Hourly emission accounting via dispatch cache. |
| `step5_scenario_a_consequential.py` | **Scenario A.** Forward-stepping consequential procurement with per-resource floor ratchets. PFS fallback on filter exhaustion. |
| `step5_scenario_b_hourly.py` | **Scenario B.** Hourly matching procurement strategy comparison. |
| `step5_scenario_comparison.py` | **Scenario comparison.** Consequential vs. hourly matching — cost, emissions, resource mix differentials. |
| `step5_analyze_storage.py` | **Storage analysis.** Battery/LDES utilization, dispatch patterns, capacity factor analysis. |

### Step 6.5: Corporate Procurement Strategy Simulation

| Script | What It Does |
|--------|-------------|
| `step5_5_procurement_utils.py` | **Shared utilities.** SSS allocation, EAC pricing, LMP feedback, PPA premiums, learning curves, 25-year timeline. |
| `step5_5_strategy1_consequential.py` | **Strategy 1 (A/B/C).** Cross-regional consequential netting under 3 emission baselines (grid-avg, fossil-avg, marginal). |
| `step5_5_strategy2_hourly.py` | **Strategy 2 (A/B/C).** Hourly matching same-ISO with existing clean credit variants. |
| `step5_5_strategy3_annual.py` | **Strategy 3 (A/B/C/D).** Annual matching 2×2 matrix (new-only vs all-clean × additionality). |

### Step 7: Dashboard Data Generation

| Script | What It Does |
|--------|-------------|
| `step6_generate_shared_data.py` | **Main export.** Extracts all results into `dashboard/js/shared-data.js`. SBTi milestone mapping, DAC trajectory projections, LCOE/transmission tables for client-side repricing. Aggregates all Step 5/6 outputs. Runs last. |
| `step6_extract_no_regrets.py` | **No-regrets analysis.** Extracts optimal targets and no-regrets resource investments from crossover analysis. |

### Shared Utilities

| Script | What It Does |
|--------|-------------|
| `dispatch_utils.py` | **Dispatch engine.** Single source of truth for dispatch reconstruction, supply profiles, fossil retirement, cache I/O. |
| `scenario_common.py` | **Scenario utilities.** Shared logic for Scenario A/B: cost tables, demand growth, learning curves, EF/PFS loading. |
| `eia_data_io.py` | **EIA data I/O.** Standardized loading of multi-year EIA generation/demand profiles. |
| `calibrate_lmp_model.py` | **LMP calibration.** Validates synthetic LMP model against actual ISO data (2024 SOM reports). |

### Pipeline Execution Order

```
Step 0 (data fetch, optional)
  → Step 1a → 1b → 1c → 1d (PFS generation)
    → Step 2 (efficient frontier)
      → Step 3 (cost optimization)
        → Step 4 (gas/CCS adjustments)
          → Step 5 (dispatch cache first, then track export/analysis in parallel)
            → Step 6.0 (CO₂ recompute — run before 6.1)
              → Steps 6.1-6.6 (all cache-dependent analytics, can run in parallel)
                → Step 7 (dashboard data export)
```

**Key principle**: Step 1 is expensive (hours). Steps 2–4 are cheap (seconds to minutes). Changing cost assumptions only requires Steps 3–4 + post-processing. Changing methodology for a single analysis page only requires the relevant Step 6 script + Step 7.

### GitHub Actions Workflows

All workflows are triggered manually via `workflow_dispatch`. Each creates a fresh branch, commits results, and opens a PR.

| # | Workflow | What It Does |
|---|----------|-------------|
| 0 | `step0-fetch-lmp-data.yml` | Fetch actual DA hourly LMP from ISO APIs |
| 1a | `step1a-scored-database.yml` | Generate + score resource fraction combos |
| 1b | `step1b-build-pfs.yml` | Mine PFS from scored database per threshold |
| 1d | `step1d-storage-refinement.yml` | Fill storage exploration gaps |
| 2 | `step2-efficient-frontier.yml` | EF filter + optional EF expansion for Scenario A |
| 2.5 | `step2.5-expand-ef.yml` | Standalone EF expansion for floor constraints |
| 3 | `step3-cost-optimization.yml` | Vectorized cost optimization (3 tracks) |
| 4 | `step4-gas-ccs.yml` | NEISO gas constraint, 45Q, resource adequacy |
| 5 | `step4-dispatch-cache.yml` | Dispatch cache build + track export/analysis |
| 5.2 | `step4.2-track-analysis.yml` | Track cost envelope analysis |
| 6.0 | `step5.0-compute-co2.yml` | Dispatch-stack CO₂ (shared, run before 6.1) |
| 6.1 | `step5.1-update-mac-page.yml` | MAC stats → Optimal targets → Shared data |
| 6.2 | `step5.2-update-lmp-page.yml` | Compute synthetic LMP → Extract dashboard JS |
| 6.3 | `step5.3-update-scenarios-page.yml` | Scenario A → B → Compare |
| 6.4 | `step5.4-procurement-strategies.yml` | 10 strategy variants → combined dashboard JS |
| 6.5 | `step5.5-supplemental-analytics.yml` | Compressed day + consequential queue |
| 6.6 | `step5.6-update-optimizer-dashboard.yml` | Compressed day + shared-data.js for dashboard |
| 7 | `step6-generate-shared-data.yml` | Consolidate all results into shared-data.js |

**Common workflow patterns:**
- Update Optimizer Dashboard: `Step 6.6`
- Update Abatement page: `Step 6.0` (if CO₂ stale) → `Step 6.1`
- Update LMP page: `Step 6.2`
- Cost assumptions changed: `Step 3 → 4 → 5 → 6.0 → 6.1+6.2+6.3+6.6 (parallel) → 7`
- Full pipeline: `Step 0 → 1a → 1b → 1d → 2 → 3 → 4 → 5 → 6.0 → 6.1-6.6 → 7`

## Project Structure

```
hourly-cfe-optimizer/
├── scripts/
│   ├── step0_*.py (8 scripts)              # Data fetch/prep
│   ├── step1_pfs_generator.py              # Monolithic PFS generator (legacy)
│   ├── step1a_generate_mixes.py            # Modular: generate resource combos
│   ├── step1b_score_mixes.py               # Modular: score mixes vs demand
│   ├── step1c_build_pfs.py                 # Modular: build PFS per threshold
│   ├── step1d_storage_refinement.py        # Modular: fill storage gaps
│   ├── step2_efficient_frontier.py         # Efficient frontier extraction
│   ├── step2_5_expand_ef_for_floors.py     # EF expansion for scenario floors
│   ├── step3_cost_optimization.py          # Cost optimization (5,832 combos)
│   ├── step3_track_nb_ctr.py              # Track 2 (NB) + Track 3 (CTR)
│   ├── step4_gas_ccs_adjustement.py        # Gas/CCS post-processing
│   ├── step4_build_dispatch_cache.py       # Dispatch cache (NPZ v2)
│   ├── step5_export_track_results.py       # Track result export
│   ├── step5_analyze_tracks.py             # Track cost envelope analysis
│   ├── step5_compute_co2.py              # CO₂ dispatch model
│   ├── step5_compute_mac_stats.py          # MAC statistics + ANOVA
│   ├── step5_compute_lmp_prices.py         # Synthetic LMP (7 ISOs)
│   ├── step5_compute_optimal_targets.py    # Optimal CFE targets
│   ├── step5_compress_day_profiles.py             # 24-hr representative profiles
│   ├── step5_consequential_deployment_queue.py        # Cross-regional deployment queue
│   ├── step5_scenario_a_consequential.py                 # Scenario A (consequential)
│   ├── step5_scenario_b_hourly.py                 # Scenario B (hourly matching)
│   ├── step5_scenario_comparison.py           # A vs B comparison
│   ├── step5_analyze_storage.py            # Storage dispatch analysis
│   ├── step5_5_procurement_utils.py        # Procurement strategy utilities
│   ├── step5_5_strategy1_consequential.py  # Strategy 1: cross-regional netting
│   ├── step5_5_strategy2_hourly.py         # Strategy 2: hourly matching
│   ├── step5_5_strategy3_annual.py         # Strategy 3: annual matching
│   ├── step6_generate_shared_data.py       # Dashboard data export
│   ├── step6_extract_no_regrets.py         # No-regrets analysis
│   ├── dispatch_utils.py                   # Shared dispatch engine
│   ├── scenario_common.py                  # Shared scenario utilities
│   ├── eia_data_io.py                      # EIA data I/O
│   ├── calibrate_lmp_model.py              # LMP model calibration
│   └── ...                                 # Additional utilities
├── data/
│   ├── step1-pfs-parquets/                 # PFS per-ISO/threshold (Step 1 output)
│   ├── step1d-storage-parquets/            # Storage refinement (Step 1d output)
│   ├── step2-ef-parquets/                  # Efficient frontier (Step 2 output)
│   ├── step3-cost-opt-parquets/            # Cost optimization (Step 3 output)
│   ├── step4-gas-ccs-parquets/             # Corrected results (Step 4 output)
│   ├── step5-post-processing/              # All post-processing outputs
│   │   ├── dispatch_cache/                 # Per-ISO NPZ dispatch cache
│   │   ├── co2_results/                    # CO₂ recomputation results
│   │   ├── lmp/                            # Synthetic LMP data
│   │   ├── mac_stats.json                  # MAC statistics
│   │   ├── optimal_targets.json            # Optimal CFE targets
│   │   ├── scenario_comparison.json        # A vs B comparison
│   │   ├── track_results.json              # Track export
│   │   └── ...
│   ├── eia-930/                            # Source EIA hourly profiles
│   └── ...                                 # Additional data files
├── dashboard/
│   ├── index.html                          # Homepage (scrollytelling)
│   ├── dashboard.html                      # Interactive cost optimizer
│   ├── abatement_dashboard.html            # CO₂ abatement analysis
│   ├── scenario_comparison.html            # Consequential vs hourly matching
│   ├── storage_analysis.html               # Storage dispatch analysis
│   ├── lmp_trends.html                     # LMP trend analysis
│   ├── new_build_analysis.html             # New-build track analysis
│   ├── cost_to_replace.html                # Cost-to-replace analysis
│   ├── consequential_accounting.html       # Consequential accounting deep-dive
│   ├── procurement_strategies.html         # Procurement strategy comparison
│   ├── research_paper.html                 # Standalone research paper
│   ├── optimizer_methodology.html          # Technical methodology
│   ├── pipeline.html                       # Pipeline architecture
│   ├── about.html                          # About the project
│   ├── js/shared-data.js                   # Dashboard data (Step 7 output)
│   └── ...                                 # Additional pages
├── .github/workflows/                      # CI/CD (22 workflow files)
├── SPEC.md                                 # Complete specification document
├── CLAUDE.md                               # Claude Code session instructions
└── README.md
```

## Resource Types

| Resource | Profile | New-Build? | Key Parameters |
|----------|---------|-----------|----------------|
| **Clean Firm** (nuclear + geothermal) | Seasonal-derated baseload | Yes | Regional uprate/new-build LCOE, monthly CF factors |
| **Solar** | EIA regional hourly, DST-corrected | Yes | Regional LCOE + transmission |
| **Wind** | EIA regional hourly | Yes | Regional LCOE + transmission |
| **Hydro** | EIA regional hourly | No (capped at existing) | Wholesale-priced, $0 transmission |
| **CCS-CCGT** | Flat baseload (45Q-incentivized max CF) | Yes | 95% capture, 45Q offset, fuel-linked |
| **Battery 4hr** | Daily-cycle dispatch, 85% RTE | Yes | Capacity-constrained, annualized $/MWh-cap |
| **Battery 8hr** | Daily-cycle dispatch, 85% RTE | Yes | Capacity-constrained, annualized $/MWh-cap |
| **LDES** (100hr iron-air) | 7-day rolling window, 50% RTE | Yes | Multi-day dispatch, annualized $/MWh-cap |
| **Green H2** (seasonal) | 30-day rolling window, 35% RTE | Yes | ≥95% thresholds only, 1000hr duration |
| **Geothermal** (CAISO only) | Flat year-round | Yes | 5th physics dimension, 39 TWh cap |

## Sensitivity Toggles

| Toggle | Options | Description |
|--------|---------|-------------|
| Renewable Gen | Low / Medium / High | Solar + wind LCOE |
| Firm Gen | Low / Medium / High | Nuclear uprate + new-build LCOE |
| Storage | Low / Medium / High | Battery + LDES cost |
| CCS | Low / Medium / High | CCS-CCGT LCOE (independent of Firm Gen) |
| 45Q | On / Off | Federal 45Q tax credit ($27.5/MWh offset) |
| Fossil Fuel | Low / Medium / High | Gas prices → wholesale + CCS fuel + emission rates |
| Transmission | None / Low / Medium / High | Interconnection costs per resource |
| Geothermal | Low / Medium / High | CAISO only, capped at 39 TWh |

**Scenario count**: 3×3×3×3×2×3×4 = **5,832** per region/threshold (non-CAISO). **17,496** for CAISO (×3 geothermal).

## Key Acronyms

- **PFS** — Physics Feasible Space: all physically valid resource mixes (Step 1 output)
- **EF** — Efficient Frontier: non-dominated mixes optimal under some cost assumption (Step 2 output)
- **MAC** — Marginal Abatement Cost: $/tCO₂ of incremental clean energy deployment
- **CEM** — Clean Energy Matching: hourly demand-weighted clean energy score (%)
- **NB** — New-Build track: greenfield cost analysis
- **CTR** — Cost-to-Replace track: replacement cost for existing generation
- **FOAK/NOAK** — First/Nth-of-a-Kind: learning curve pricing for new technologies

## Data Sources

- **EIA-930**: Hourly grid generation data (2021–2025, multi-year averaged)
- **NREL ATB 2024**: LCOE estimates for solar, wind, nuclear, battery, LDES
- **LBNL**: Utility-Scale Solar 2024, Wind Market Report 2024
- **FERC/ISO**: Wholesale electricity price averages (2023–2024)
- **eGRID**: Emission rate data per fuel type
- **USGS**: Geothermal resource assessments (CAISO)
- **PJM/ERCOT/NYISO SOM 2024**: LMP calibration data

## License

MIT License. See [LICENSE](LICENSE) for details.
