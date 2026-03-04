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

## 10-Step Pipeline

The optimizer runs as a 10-step pipeline (Steps 0–9). Step 1 is expensive (hours). Steps 2–9 are cheap (seconds to minutes). Each step maps to a clear dependency layer.

### Core Pipeline (Steps 0–4)

| Step | Script(s) | What It Does | When to Re-run |
|------|-----------|-------------|---------------|
| **Step 0** | `step0_*.py` (8 scripts) | **Data Acquisition** — EIA hourly profiles (multi-year 2021–2025), eGRID emissions, actual LMP data, offshore wind (NREL NOW-23), UTC fixes, MISO/SPP consolidation. Annual cadence. | When source data updates. |
| **Step 1** | `step1_pfs_generator.py` (monolithic) or 1A+1B+1C (modular) | **Physics Feasible Space** — 4D/5D adaptive grid search × procurement × storage. Output: `data/step1-pfs-parquets/` + `data/step1d-storage-parquets/`. | Only if dispatch logic, generation profiles, or demand curves change. |
| **Step 2** | `step2_efficient_frontier.py` | **Efficient Frontier** — Extracts non-dominated mixes from PFS. Reads both step1 and step1d parquets. Output: `data/step2-ef-parquets/`. | Only if PFS or filtering criteria change. |
| **Step 3** | `step3a_cost_optimization.py` + `step3b_track_nb_ctr.py` | **Cost Optimization** — 3A: Track 1 baseline (5,832 combos, 17,496 CAISO). 3B: Track 2 (NB) + Track 3 (CTR). Output: `data/step3-cost-opt-parquets/`. | When cost assumptions change. |
| **Step 4** | `step4_build_dispatch_cache.py` | **Dispatch Cache** — 8,760-hour dispatch for all unique mixes. NPZ v2. | After Step 3. |

### Step 1 Sub-Pipeline (Modular, for CI/CD)

| Sub-step | Script(s) | What It Does |
|----------|-----------|-------------|
| **1A: Generate+Score** | `step1a_generate_mixes.py` → `step1b_score_mixes.py` | Generates resource fraction combos (4D/5D grid), scores against hourly demand. |
| **1B: Zone Search** | `step1b_zone_search.py` | Fine-zone targeted search per threshold. |
| **1C: Storage Refinement** | `step1c_storage_refinement.py` | Fills storage gaps (near-miss + lean-mix strategies). Output: `data/step1d-storage-parquets/`. |

Utilities: `step1_pfs_generator.py` (monolithic), `step1_prior_windows.py` (search windows from prior EF).

### Steps 5–9: Analysis & Dashboard

#### Step 5: Core Analysis (6 scripts, all parallel after Step 4)

| Script | What It Does | Dependencies |
|--------|-------------|-------------|
| `step5a_compute_co2.py` | CO₂ dispatch-stack model. Merit-order fuel retirement. | Step 4. **Run before Step 6A.** |
| `step5b_compute_lmp_prices.py` | Synthetic hourly LMP from merit-order fossil stack. All 7 ISOs. | Step 4 |
| `step5c_compress_day_profiles.py` | 24-hour representative day profiles. | Step 4 |
| `step5d_deployment_queue.py` | Cross-regional consequential deployment path. | Step 4 |
| `step5e_export_tracks.py` | Track export (NB + CTR) to `track_results.json`. | Step 3 |
| `step5f_analyze_storage.py` | Battery/LDES utilization, dispatch patterns, capacity factors. | Step 4 + Step 3 |

#### Step 6: Derived Analytics (3 scripts)

| Script | What It Does | Dependencies |
|--------|-------------|-------------|
| `step6a_compute_mac_stats.py` | MAC statistics, ANOVA decomposition, DAC/SCC/ETS crossover. | Step 5A |
| `step6b_compute_optimal_targets.py` | Optimal CFE targets via MAC × DAC crossover. No-regrets analysis. | Step 3 |
| `step6c_analyze_tracks.py` | Track cost envelopes (P10/P50/P90), resource mix differentials. | Step 5E |

#### Step 7: Scenario Comparison (3 scripts)

| Script | What It Does | Dependencies |
|--------|-------------|-------------|
| `step7a_scenario_consequential.py` | Consequential procurement with per-resource floor ratchets. | Step 2 |
| `step7b_scenario_hourly.py` | Hourly matching procurement strategy. | Step 6B |
| `step7c_scenario_comparison.py` | Consequential vs. hourly — cost, emissions, resource mix. | Steps 7A + 7B |

#### Step 8: Procurement Strategies (4 scripts + 1 utility)

| Script | What It Does | Dependencies |
|--------|-------------|-------------|
| `procurement_utils.py` | Shared utilities: SSS, EAC, LMP feedback, PPA, learning curves. | — |
| `step8a_strategy_consequential.py` | Strategy 1 (A/B/C): cross-regional consequential netting. | Step 5D |
| `step8b_strategy_hourly.py` | Strategy 2 (A/B/C): hourly matching same-ISO. | — |
| `step8c_strategy_annual.py` | Strategy 3 (A/B/C/D): annual matching 2×2 matrix. | — |
| `step8d_wrights_law_curves.py` | Wright's Law learning curves & critical mass threshold. | — |

#### Step 9: Dashboard Data Generation

| Script | What It Does | Dependencies |
|--------|-------------|-------------|
| `step9a_generate_shared_data.py` | All results → `dashboard/js/shared-data.js`. **Runs last.** | All upstream |
| `step9b_extract_no_regrets.py` | No-regrets resource investments from crossover analysis. | Step 6B |

### Shared Utilities

| Script | What It Does |
|--------|-------------|
| `pipeline_config.py` | Single source of truth for all shared constants. |
| `dispatch_utils.py` | Dispatch reconstruction, supply profiles, fossil retirement, cache I/O. |
| `scenario_common.py` | Shared Scenario A/B logic: cost tables, demand growth, EF/PFS loading. |
| `eia_data_io.py` | Standardized multi-year EIA generation/demand profile loading. |
| `calibrate_lmp_model.py` | Validates synthetic LMP against actual ISO data. |

### Pipeline Execution Order

```
Step 0 (data acquisition, annual)
  → Step 1A (generate+score) → 1B (zone search) → 1C (storage refinement)
    → Step 2 (efficient frontier)
      → Step 3 (cost optimization)
        → Step 4 (dispatch cache)
          → Step 5 (5A–5F core analysis, all parallel)
            → Step 6 (derived analytics)
              → Step 7 (scenario comparison) + Step 8 (procurement strategies)
                → Step 9 (dashboard data export)
```

**Key principle**: Step 1 is expensive (hours). Steps 2–9 are cheap. Changing cost assumptions → Steps 3 + downstream. Changing a single analysis → relevant step + Step 9.

### GitHub Actions Workflows

All workflows: `workflow_dispatch`, ISO selectors. See `.github/workflows/README.md` for full docs.

| Step | Workflow | What It Does |
|------|----------|-------------|
| 0 | `step0-fetch-lmp-data.yml` | Fetch actual DA hourly LMP |
| 0 | `step0-fetch-offshore-wind.yml` | Fetch NREL NOW-23 offshore wind profiles |
| 1A | `step1a-scored-database.yml` | Generate + score resource combos |
| 1B | `step1b-zone-search.yml` | Zone-specific fine search |
| 1C | `step1c-storage-refinement.yml` | Fill storage exploration gaps |
| 2 | `step2-efficient-frontier.yml` | EF extraction |
| 3 | `step3-cost-optimization.yml` | Cost optimization (3 tracks) |
| 4 | `step4-dispatch-cache.yml` | Dispatch cache build |
| 5A | `step5a-compute-co2.yml` | CO₂ dispatch model |
| 5B | `step5b-compute-lmp.yml` | Synthetic LMP |
| 5C+D | `step5cd-supplemental.yml` | Day profiles + deployment queue |
| 5E | `step5e-track-analysis.yml` | Track export + analysis |
| 5F | `step5f-storage-analysis.yml` | Storage dispatch analysis |
| 6 | `step6-derived-analytics.yml` | MAC stats + optimal targets |
| 7 | `step7-scenario-comparison.yml` | Scenarios A → B → comparison |
| 8 | `step8-procurement-strategies.yml` | All procurement strategies |
| 9 | `step9-generate-shared-data.yml` | Dashboard data export |

**Common workflow patterns:**
- Update Optimizer Dashboard: `Step 5C` (day profiles) → `Step 9`
- Update Abatement page: `Step 5A` (if CO₂ stale) → `Step 6`
- Update LMP page: `Step 5B`
- Cost assumptions changed: `Step 3 → 4 → 5 (parallel) → 6 → 7+8 (parallel) → 9`
- Full pipeline: `Step 0 → 1A → 1B → 1C → 2 → 3 → 4 → 5 → 6 → 7+8 → 9`

## Project Structure

```
hourly-cfe-optimizer/
├── scripts/
│   ├── step0_*.py (8 scripts)              # Data acquisition (annual cadence)
│   ├── step1_pfs_generator.py              # Monolithic PFS generator
│   ├── step1_prior_windows.py              # Search window computation
│   ├── step1a_generate_mixes.py            # Step 1A: generate resource combos
│   ├── step1b_score_mixes.py               # Step 1A: score mixes vs demand
│   ├── step1b_zone_search.py               # Step 1B: zone-specific fine search
│   ├── step1c_storage_refinement.py        # Step 1C: fill storage gaps
│   ├── step2_efficient_frontier.py         # Step 2: efficient frontier
│   ├── step3a_cost_optimization.py         # Step 3A: baseline cost optimization
│   ├── step3b_track_nb_ctr.py              # Step 3B: NB + CTR tracks
│   ├── step4_build_dispatch_cache.py       # Step 4: dispatch cache (NPZ v2)
│   ├── step5a_compute_co2.py               # Step 5A: CO₂ dispatch model
│   ├── step5b_compute_lmp_prices.py        # Step 5B: synthetic LMP (7 ISOs)
│   ├── step5c_compress_day_profiles.py     # Step 5C: 24-hr representative profiles
│   ├── step5d_deployment_queue.py          # Step 5D: cross-regional deployment
│   ├── step5e_export_tracks.py             # Step 5E: track result export
│   ├── step5f_analyze_storage.py           # Step 5F: storage dispatch analysis
│   ├── step6a_compute_mac_stats.py         # Step 6A: MAC statistics + ANOVA
│   ├── step6b_compute_optimal_targets.py   # Step 6B: optimal CFE targets
│   ├── step6c_analyze_tracks.py            # Step 6C: track cost envelopes
│   ├── step7a_scenario_consequential.py    # Step 7A: Scenario A
│   ├── step7b_scenario_hourly.py           # Step 7B: Scenario B
│   ├── step7c_scenario_comparison.py       # Step 7C: A vs B comparison
│   ├── procurement_utils.py               # Step 8 shared utilities
│   ├── step8a_strategy_consequential.py    # Step 8A: cross-regional netting
│   ├── step8b_strategy_hourly.py           # Step 8B: hourly matching
│   ├── step8c_strategy_annual.py           # Step 8C: annual matching
│   ├── step8d_wrights_law_curves.py        # Step 8D: Wright's Law curves
│   ├── step9a_generate_shared_data.py      # Step 9A: dashboard data export
│   ├── step9b_extract_no_regrets.py        # Step 9B: no-regrets analysis
│   ├── pipeline_config.py                  # Shared constants (LCOE, fuel, storage)
│   ├── dispatch_utils.py                   # Dispatch engine
│   ├── scenario_common.py                  # Scenario A/B shared logic
│   ├── eia_data_io.py                      # EIA data I/O
│   ├── calibrate_lmp_model.py              # LMP model calibration
│   └── ...                                 # Additional utilities
├── data/
│   ├── step1-pfs-parquets/                 # PFS per-ISO/threshold (Step 1)
│   ├── step1d-storage-parquets/            # Storage refinement (Step 1C)
│   ├── step2-ef-parquets/                  # Efficient frontier (Step 2)
│   ├── step3-cost-opt-parquets/            # Cost optimization (Step 3)
│   ├── step5-post-processing/              # All post-processing outputs
│   │   ├── dispatch_cache/                 # Per-ISO NPZ dispatch cache (Step 4)
│   │   ├── co2_results/                    # CO₂ results (Step 5A)
│   │   ├── lmp/                            # Synthetic LMP (Step 5B)
│   │   ├── mac_stats.json                  # MAC statistics (Step 6A)
│   │   ├── optimal_targets.json            # Optimal targets (Step 6B)
│   │   ├── scenario_comparison.json        # A vs B (Step 7C)
│   │   ├── track_results.json              # Track export (Step 5E)
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
│   ├── js/shared-data.js                   # Dashboard data (Step 9 output)
│   └── ...                                 # Additional pages
├── .github/workflows/                      # CI/CD (~22 workflow files)
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
