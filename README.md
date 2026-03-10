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

## 8-Step Pipeline (Steps 0–7)

The optimizer runs as an 8-step pipeline. Step 1 is expensive (hours). Steps 2–7 are cheap (seconds to minutes).

**Naming convention**: `.1/.2/.3` = sequential sub-steps. `A/B/C` = parallel scripts.

### Pipeline Execution Order

```
Step 0 (data acquisition, annual)
  → Step 1 (1.1 → 1.2 → 1.3 → 1.4 → 1.5: physics PFS)
    → Step 2 (2.1 EF → 2.2A+2.2B cost optimization)
      → Step 3A (dispatch cache) ∥ Step 3B (MAC queue, needs Step 1 only)
        → Step 4.1 (4.1A–4.1F: core analysis, all parallel)
          → Step 4.2 (4.2A–4.2C: derived analysis, all parallel)
            → Step 5 (scenarios + procurement)
            → Step 6 (SMARTargets + IPP + nuclear)
              → Step 7 (dashboard data aggregation)
```

### Step 0: Data Acquisition
`step0_*.py` (9 scripts, annual) — EIA, eGRID, LMP, offshore wind, MISO/SPP.

### Step 1: Physics Feasible Space (sequential: 1.1 → 1.5)

| Sub-step | Script(s) | What It Does |
|----------|-----------|-------------|
| **1.1** | `step1_1a_generate_mixes.py` → `step1_1b_score_mixes.py` | Generate + score resource combos (4D/5D grid). |
| **1.2** | `step1_2_zone_search.py` | Zone-specific fine search per threshold. |
| **1.3** | `step1_3_floor_aware_pfs.py` | Floor-aware PFS augmentation (50-80%). |
| **1.4** | `step1_4_fine_grid_pfs.py` | Fine-grid PFS (40-75%). |
| **1.5** | `step1_5_storage_refinement.py` | Storage dispatch refinement. |

Utilities: `step1_pfs_generator.py` (monolithic), `step1_prior_windows.py` (search windows).

### Step 2: Optimization (sequential: 2.1 → 2.2)

| Sub-step | Script(s) | What It Does |
|----------|-----------|-------------|
| **2.1** | `step2_1_efficient_frontier.py` | Non-dominated mix extraction. Output: `data/step2-ef-parquets/`. |
| **2.2A** | `step2_2a_cost_optimization.py` | Track 1 baseline (5,832 combos). ─┐ parallel |
| **2.2B** | `step2_2b_track_nb_ctr.py` | Track 2 (NB) + Track 3 (CTR). ─┘ |

Output: `data/step3-cost-opt-parquets/`.

### Step 3: Caches (parallel: 3A ∥ 3B)

| Script | What It Does | Dependencies |
|--------|-------------|-------------|
| `step3a_build_dispatch_cache.py` | 8,760-hour dispatch cache (NPZ v2). | Step 2.2 |
| `step3b_mac_queue.py` | Path-dependent MAC queue. | Step 1 only |

### Step 4: Analysis (4.1 parallel → 4.2 parallel)

| Script | What It Does | Deps |
|--------|-------------|------|
| `step4_1a_fossil_dispatch.py` | CO₂ + LMP fossil dispatch. | 3A |
| `step4_1b_compress_day_profiles.py` | 24-hr day profiles. | 3A |
| `step4_1c_compute_mac_stats.py` | MAC stats + ANOVA. | 2.2 |
| `step4_1d_compute_optimal_targets.py` | Optimal CFE targets. | 2.2 |
| `step4_1e_export_tracks.py` | Track export → JSON. | 2.2 |
| `step4_1f_extract_building_blocks.py` | Building blocks. | 2.2 |
| `step4_2a_extract_resource_density.py` | Resource density. | 4.1D |
| `step4_2b_analyze_storage.py` | Storage analysis. | 2.2 + 4.1B |
| `step4_2c_analyze_tracks.py` | Track cost envelopes. | 4.1E |

### Step 5: Scenarios & Procurement (5.1 → 5.2 parallel)

| Script | What It Does | Deps |
|--------|-------------|------|
| `step5_1_scenario_hourly.py` | Hourly matching scenario. | 4.1D |
| `step5_2a_scenario_comparison.py` | Consequential vs. hourly. | 3B + 5.1 |
| `step5_2b_strategy_consequential.py` | Cross-regional netting. | 3B |
| `step5_2c_strategy_hourly.py` | Hourly matching strategy. | 5.1 |
| `step5_2d_strategy_annual.py` | Annual matching 2×2 matrix. | 2.2 |
| `step5_2e_wrights_law_curves.py` | Wright's Law curves. | — |

### Step 6: Policy Analysis (6.1 → 6.2 parallel)

| Script | What It Does | Deps |
|--------|-------------|------|
| `step6_1_smartargets.py` | Regional SMARTargets. | 2.2 + 4.1A |
| `step6_2a_ipp_smartargets.py` | IPP fleet modeling. | 6.1 |
| `step6_2b_nuclear_retirement.py` | Nuclear stranding risk. | 6.1 + 2.2 |

### Step 7: Dashboard Aggregation (7.1 parallel → 7.2)

| Script | What It Does | Deps |
|--------|-------------|------|
| `step7_1a_generate_shared_data.py` | Main export → `shared-data.js`. | All upstream |
| `step7_2_extract_no_regrets.py` | No-regrets analysis. | 7.1A |

### Shared Utilities

`pipeline_config.py`, `dispatch_utils.py`, `scenario_common.py`, `eia_data_io.py`, `calibrate_lmp_model.py`, `procurement_utils.py`

### GitHub Actions Workflows

All workflows: `workflow_dispatch`, ISO selectors. See `.github/workflows/README.md`.

| Step | Workflow | What It Does |
|------|----------|-------------|
| 0 | `step0-fetch-lmp-data.yml` | Fetch actual DA hourly LMP |
| 0 | `step0-fetch-offshore-wind.yml` | Fetch offshore wind profiles |
| 1.1 | `step1-1-scored-database.yml` | Generate + score resource combos |
| 1.2–1.3 | `step1-2-3-zone-floor.yml` | Zone search + floor PFS |
| 1.4–1.5 | `step1-4-5-fine-storage.yml` | Fine grid + storage refinement |
| 2.1 | `step2-1-efficient-frontier.yml` | EF extraction |
| 2.2 | `step2-2-cost-optimization.yml` | Cost optimization (3 tracks) |
| 3A | `step3a-dispatch-cache.yml` | Dispatch cache |
| 3B | `step3b-mac-queue.yml` | MAC queue |
| 4 | `step4-derived-analytics.yml` | MAC + targets + building blocks |
| 4 | `step4-tracks.yml` | Track export + analysis |
| 4.1B | `step4-1b-day-profiles.yml` | Day profiles |
| 4.2B | `step4-2b-storage-analysis.yml` | Storage analysis |
| 5 | `step5-scenarios.yml` | Scenario comparison |
| 5 | `step5-procurement.yml` | Procurement strategies |
| 6.1 | `step6-1-smartargets-*.yml` | SMARTargets (4 scenario workflows) |
| 7 | `step7-dashboard-data.yml` | Dashboard data export |

**Common patterns:**
- Cost assumptions changed: `Step 2.2 → 3 → 4 → 5+6 → 7`
- Full pipeline: `Step 0 → 1.1 → 1.2 → 1.3 → 1.4 → 1.5 → 2.1 → 2.2 → 3 → 4 → 5+6 → 7`

## Project Structure

```
hourly-cfe-optimizer/
├── scripts/
│   ├── step0_*.py (9 scripts)              # Data acquisition (annual)
│   ├── step1_pfs_generator.py              # Monolithic PFS generator
│   ├── step1_prior_windows.py              # Search window computation
│   ├── step1_1a_generate_mixes.py          # Step 1.1: generate resource combos
│   ├── step1_1b_score_mixes.py             # Step 1.1: score mixes vs demand
│   ├── step1_2_zone_search.py              # Step 1.2: zone-specific fine search
│   ├── step1_3_floor_aware_pfs.py          # Step 1.3: floor-aware PFS
│   ├── step1_4_fine_grid_pfs.py            # Step 1.4: fine-grid PFS
│   ├── step1_5_storage_refinement.py       # Step 1.5: storage refinement
│   ├── step2_1_efficient_frontier.py       # Step 2.1: efficient frontier
│   ├── step2_2a_cost_optimization.py       # Step 2.2A: baseline cost optimization
│   ├── step2_2b_track_nb_ctr.py            # Step 2.2B: NB + CTR tracks
│   ├── step3a_build_dispatch_cache.py      # Step 3A: dispatch cache (NPZ v2)
│   ├── step3b_mac_queue.py                 # Step 3B: MAC queue
│   ├── step4_1a_fossil_dispatch.py         # Step 4.1A: CO₂ + LMP
│   ├── step4_1b_compress_day_profiles.py   # Step 4.1B: day profiles
│   ├── step4_1c_compute_mac_stats.py       # Step 4.1C: MAC statistics
│   ├── step4_1d_compute_optimal_targets.py # Step 4.1D: optimal CFE targets
│   ├── step4_1e_export_tracks.py           # Step 4.1E: track export
│   ├── step4_1f_extract_building_blocks.py # Step 4.1F: building blocks
│   ├── step4_2a_extract_resource_density.py# Step 4.2A: resource density
│   ├── step4_2b_analyze_storage.py         # Step 4.2B: storage analysis
│   ├── step4_2c_analyze_tracks.py          # Step 4.2C: track cost envelopes
│   ├── step5_1_scenario_hourly.py          # Step 5.1: hourly matching scenario
│   ├── step5_2a_scenario_comparison.py     # Step 5.2A: scenario comparison
│   ├── step5_2b_strategy_consequential.py  # Step 5.2B: consequential netting
│   ├── step5_2c_strategy_hourly.py         # Step 5.2C: hourly strategy
│   ├── step5_2d_strategy_annual.py         # Step 5.2D: annual strategy
│   ├── step5_2e_wrights_law_curves.py      # Step 5.2E: Wright's Law curves
│   ├── step6_1_smartargets.py              # Step 6.1: regional SMARTargets
│   ├── step6_2a_ipp_smartargets.py         # Step 6.2A: IPP modeling
│   ├── step6_2b_nuclear_retirement.py      # Step 6.2B: nuclear retirement
│   ├── step7_1a_generate_shared_data.py    # Step 7.1A: dashboard data export
│   ├── step7_2_extract_no_regrets.py       # Step 7.2: no-regrets analysis
│   ├── pipeline_config.py                  # Shared constants (LCOE, fuel, storage)
│   ├── dispatch_utils.py                   # Dispatch engine
│   ├── scenario_common.py                  # Scenario A/B shared logic
│   ├── procurement_utils.py               # Procurement shared utilities
│   └── ...                                 # Additional utilities
├── data/
│   ├── step1-pfs-parquets/                 # Physics (Step 1)
│   ├── step2-ef-parquets/                  # Efficient frontier (Step 2.1)
│   ├── step3-cost-opt-parquets/            # Cost optimization (Step 2.2)
│   ├── step4-dispatch-cache/               # Dispatch cache (Step 3A)
│   ├── step5-post-processing/              # Analysis outputs (Steps 3B, 4–6)
│   ├── step10-smartargets/                 # SMARTargets (Step 6)
│   ├── step12-nuclear-retirement/          # Nuclear analysis (Step 6.2B)
│   └── ...                                 # Additional data files
├── dashboard/
│   ├── index.html                          # Homepage (scrollytelling)
│   ├── dashboard.html                      # Interactive cost optimizer
│   ├── js/shared-data.js                   # Dashboard data (Step 7 output)
│   └── ...                                 # 20+ interactive pages
├── .github/workflows/                      # CI/CD (~21 workflow files)
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
