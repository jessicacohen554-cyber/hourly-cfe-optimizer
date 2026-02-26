# Hourly CFE Cost Optimizer

An interactive dashboard and optimization engine for analyzing **hourly clean energy procurement strategies** across seven major US ISO regions.

The optimizer finds the least-cost mix of clean firm (nuclear/geothermal), solar, wind, hydro, CCS-CCGT, battery, and LDES to meet hourly demand at various Clean Energy Matching (CEM) targets (50%–100%), then visualizes the results in a scrollytelling dashboard.

## Regions Covered

| ISO | Region |
|-----|--------|
| CAISO | California |
| ERCOT | Texas |
| PJM | Mid-Atlantic |
| NYISO | New York |
| NEISO | New England |
| MISO | Midwest |
| SPP | South-Central |

## Quick Start

**To view the dashboard:**

```bash
cd dashboard
python -m http.server 8000
# Open http://localhost:8000/index.html
```

## 6-Step Optimizer Pipeline

The optimizer runs as a 6-step pipeline. Steps 1–4 run sequentially. Step 5 post-processors run in parallel (PP0 first, then PP1–PP6+). Step 6 exports data for the dashboard.

| Step | Script | What It Does | When to Re-run |
|------|--------|-------------|---------------|
| **Step 1** | `scripts/step1_pfs_generator.py` | **PFS Generator** — Generates the Physics Feasible Space (PFS). 4D adaptive grid search (clean firm, solar, wind, hydro) × procurement × battery × LDES. Produces physics-validated resource mixes across 15 thresholds × 7 ISOs. | Only if dispatch logic, generation profiles, or demand curves change. |
| **Step 2** | `scripts/step2_efficient_frontier.py` | **Efficient Frontier (EF)** — Extracts the efficient frontier from the PFS. Filters existing generation utilization, minimizes procurement, removes strictly dominated mixes. | Only if PFS changes or filtering criteria change. |
| **Step 3** | `scripts/step3_cost_optimization.py` | **Cost Optimization** — Vectorized cross-evaluation of all EF mixes under 5,832 sensitivity combos (17,496 for CAISO). Merit-order tranche pricing for clean firm. Demand growth sweep (25 years × 3 growth rates). | When cost assumptions, LCOE tables, or sensitivity toggles change. |
| **Step 4** | `scripts/step4_gas_ccs_adjustement.py` | **Post-Processing** — NEISO gas constraint (+$13.13/MWh CCS adder), 45Q correction, gas capacity backup & resource adequacy (15% RA margin), CCS vs LDES crossover. | When Step 3 outputs change. |
| **Step 5** | `scripts/step5_PP*.py` | **Post-Processing Suite** — 6+ specialized scripts (see below). PP0 builds shared dispatch cache; PP1–PP6 run in parallel. | When Step 4 outputs change. |
| **Step 6** | `scripts/step6_generate_shared_data.py` | **Dashboard Data Export** — Extracts all results into `dashboard/js/shared-data.js`. SBTi mapping, DAC projections, LCOE/transmission tables for client-side repricing. | Runs last. |

**Key principle**: Step 1 is expensive (hours of compute). Steps 2–4 are cheap (seconds to minutes). Changing cost assumptions only requires Steps 3–4 + post-processing.

### Step 5: Post-Processing Scripts

| Script | What It Does |
|--------|-------------|
| `scripts/step5_PP0_build_dispatch_cache.py` | **Run first.** Pre-computes 8,760-hour dispatch for all unique mixes across all ISOs. Populates versioned NPZ cache (v2) with per-resource matched/surplus + charge profiles. |
| `scripts/step5_PP1_compressed_day.py` | 24-hour representative day profiles. Reads from PP0 dispatch cache; falls back to live compute. |
| `scripts/step5_PP2_consequential_queue.py` | Cross-regional deployment path under consequential accounting; merit-order fuel retirement. |
| `scripts/step5_PP3_scenario_comparison.py` | Consequential vs. hourly matching strategy comparison. |
| `scripts/step5_PP4_recompute_co2.py` | Dispatch-stack emission model. Merit-order fuel retirement (coal→oil→gas). Coal/oil capped at 2025 absolute TWh. Uses canonical `get_supply_profiles` (nuclear derate + DST correction). |
| `scripts/step5_PP5_compute_mac_stats.py` | MAC statistics: P10/P50/P90 fan, stepwise marginal, ANOVA decomposition, crossover analysis. |
| `scripts/step5_PP6_compute_lmp_prices.py` | Synthetic hourly LMP from merit-order fossil stack. ISO-specific price formation models (PJM RPM, ERCOT ORDC, etc.). |
| `scripts/step5_PP7_compute_eac_scarcity.py` | EAC supply scarcity analysis under RPS + voluntary demand. |
| `scripts/step5_PP8_export_track_results.py` | Exports track parquets (NB + CTR) to `track_results.json` for dashboard. |
| `scripts/step5_PP9_analyze_tracks.py` | Track result analysis: cost envelopes (P10/P50/P90), resource mix differentials. |

### Shared Dispatch Engine: `dispatch_utils.py`

Single source of truth for dispatch reconstruction, supply profiles, and fossil retirement. All PP scripts import from this module.

Key functions:
- `reconstruct_hourly_dispatch(detailed=True)` — 8,760-hour supply/demand matching with optional per-resource breakdown
- `get_supply_profiles()` — Nuclear seasonal derate + DST-corrected solar profiles
- `build_supply_matrix()` — Pre-built (5, 8760) numpy array per ISO
- `load/save_dispatch_cache(version=2)` — Versioned NPZ per ISO, shared across PP0/PP1/PP4/PP6
- `compute_fossil_retirement()` — Merit-order emission rate at threshold

### Running the Pipeline

```bash
# Requires Python 3.10+, numpy, numba, pyarrow, pandas
pip install numpy numba pyarrow pandas

# Step 1: Generate PFS (expensive — only if physics change)
python scripts/step1_pfs_generator.py

# Step 2: Extract efficient frontier (~40s)
python scripts/step2_efficient_frontier.py

# Step 3: Cost optimization (~3 min)
python scripts/step3_cost_optimization.py

# Step 4: Post-processing (~seconds)
python scripts/step4_gas_ccs_adjustement.py

# Step 5: Post-processing suite
python scripts/step5_PP0_build_dispatch_cache.py   # Run first — builds shared cache
python scripts/step5_PP1_compressed_day.py          # Reads from cache
python scripts/step5_PP4_recompute_co2.py           # Reads from cache
python scripts/step5_PP5_compute_mac_stats.py
python scripts/step5_PP6_compute_lmp_prices.py      # Reads from cache

# Step 6: Generate dashboard data
python scripts/step6_generate_shared_data.py
```

### Workflow Output Branches (GitHub Actions)

The workflows now write outputs directly to branches (instead of only uploading zip artifacts), so results are versioned in Git and can be merged into `master`:

- `run-step1-5-convert-checkpoints.yml` → `step1-5-raw-pfs-parquets`
  - Writes: `data/step1-pfs-parquets/`
- `run-step1-full-pfs.yml` → `step1-full-pfs-results`
  - Writes: `data/step1-pfs-parquets/`, `data/checkpoints_v4/`, `scripts/dashboard/`
- `run-step2-efficient-frontier.yml` → `step2-ef-parquets-results`
  - Reads from: `data/step1-pfs-parquets/`
  - Writes: `data/step2-ef-parquets/step2_ef_<ISO>.parquet` (per ISO), `data/pfs_post_ef.parquet` (merged compatibility copy)
- `run-iso-thresholds.yml` → `iso-threshold-<ISO>-<THRESHOLD>`
  - Example: `iso-threshold-NYISO-95`
  - Includes a tranche selector (`all`, or tranche `1..10`) and `max_mixes_per_tranche` control for checkpoint-sized runs.
  - Writes: `data/checkpoints/`, `data/checkpoints_v4/`, `data/step1-pfs-parquets/`, `scripts/dashboard/`
- `run-step1-nyiso100-tranches.yml` → `iso-threshold-NYISO-100`
  - Dedicated NYISO 100 workflow with tranche selector (`all`, `1..6`).
  - Direction: run tranches 1→6 sequentially for full NYISO 100 completion.
- `run-step1-caiso100-tranches.yml` → `iso-threshold-CAISO-100`
  - Dedicated CAISO 100 workflow with tranche selector (`all`, `1..5`).
  - Direction: run tranches 1→5 sequentially for full CAISO 100 completion.

These branches are safe staging branches for workflow outputs; open PRs from them into `master` when ready.

### Key Acronyms

- **PFS** — Physics Feasible Space: the full set of physically valid resource mixes (Step 1 output)
- **EF** — Efficient Frontier: the reduced set of non-dominated mixes that could be optimal under any cost assumption (Step 2 output)

## Project Structure

```
hourly-cfe-optimizer/
├── scripts/
│   ├── step1_pfs_generator.py            # Step 1: PFS generator (physics)
│   ├── step2_efficient_frontier.py       # Step 2: Efficient frontier extraction
│   ├── step3_cost_optimization.py        # Step 3: Cost optimization
│   ├── step3_track_nb_ctr.py             # Step 3: NB + CTR track optimization
│   ├── step4_gas_ccs_adjustement.py      # Step 4: Gas/CCS post-processing
│   ├── step5_PP0_build_dispatch_cache.py # PP0: Dispatch cache builder (run first)
│   ├── step5_PP1_compressed_day.py       # PP1: Compressed day profiles
│   ├── step5_PP2_consequential_queue.py  # PP2: Consequential queue
│   ├── step5_PP3_scenario_comparison.py  # PP3: Scenario comparison
│   ├── step5_PP4_recompute_co2.py        # PP4: CO₂ dispatch model
│   ├── step5_PP5_compute_mac_stats.py    # PP5: MAC statistics
│   ├── step5_PP6_compute_lmp_prices.py   # PP6: LMP pricing
│   ├── step5_PP7_compute_eac_scarcity.py # PP7: EAC scarcity
│   ├── step5_PP8_export_track_results.py # PP8: Track export
│   ├── step5_PP9_analyze_tracks.py       # PP9: Track analysis
│   ├── step6_generate_shared_data.py     # Step 6: Dashboard data export
│   └── dispatch_utils.py                 # Shared dispatch engine
├── data/
│   ├── step1-pfs-parquets/               # PFS per-ISO/threshold (Step 1 output)
│   ├── step2-ef-parquets/                # Efficient frontier parquets (Step 2 output)
│   ├── step3-cost-opt-parquets/          # Cost optimization results (Step 3 output)
│   ├── step4-gas-ccs-parquets/           # Final corrected results (Step 4 output)
│   ├── step5-post-processing/            # Post-processing outputs
│   │   ├── dispatch_cache/               # Per-ISO NPZ dispatch cache (PP0 output)
│   │   ├── lmp/                          # LMP analysis results (PP6 output)
│   │   └── ...
│   ├── EIA 930 Data/                     # Source EIA profiles
│   ├── egrid_emission_rates.json
│   └── ...
├── dashboard/
│   ├── index.html                        # Homepage (scrollytelling)
│   ├── dashboard.html                    # Interactive cost optimizer
│   ├── abatement_dashboard.html          # CO₂ Abatement Analysis
│   ├── research_paper.html               # Standalone research paper
│   ├── optimizer_methodology.html        # Methodology documentation
│   ├── pipeline.html                     # Pipeline architecture page
│   ├── about.html                        # About the project
│   ├── js/shared-data.js                 # Dashboard data (Step 6 output)
│   └── compressed_day_profiles.json      # PP1 output
├── parquet_io.py                         # Shared parquet I/O utilities
├── SPEC.md                               # Complete specification document
├── CLAUDE.md                             # Claude Code session instructions
└── README.md
```

## Resource Types

- **Clean Firm**: Nuclear (seasonal-derated) + geothermal (CAISO only, capped at 5 GW)
- **Solar**: EIA regional hourly profile, DST-corrected nighttime zeroing
- **Wind**: EIA regional hourly profile
- **Hydro**: EIA regional hourly profile (capped by region, existing only, wholesale-priced)
- **CCS-CCGT**: Implicit 5th resource (100% − sum of above four), flat baseload profile, 45Q offset in LCOE
- **Battery**: 4hr Li-ion, 85% round-trip efficiency, daily-cycle dispatch
- **Battery (8hr)**: 8hr Li-ion, 85% round-trip efficiency, daily-cycle dispatch
- **LDES**: 100hr iron-air, 50% round-trip efficiency, 7-day rolling window dispatch

## Sensitivity Toggles (5,832 combos)

| Toggle | Options | Description |
|--------|---------|-------------|
| Renewable Gen | Low / Medium / High | Solar + wind LCOE |
| Firm Gen | Low / Medium / High | Nuclear new-build + uprate LCOE |
| Storage | Low / Medium / High | Battery + LDES cost |
| Fossil Fuel | Low / Medium / High | Gas prices (affects wholesale + CCS fuel) |
| Transmission | None / Low / Medium / High | Interconnection costs per resource |
| CCS | Low / Medium / High | CCS-CCGT LCOE (default: follows Firm Gen) |
| 45Q | On / Off | Federal 45Q tax credit ($27.5/MWh offset) |
| Geothermal | Low / Medium / High | CAISO-only, capped at 39 TWh |

**Total**: 3×3×3×3×4×3×2 = 1,944 (non-CAISO base) × 3 (geothermal, CAISO) = 5,832 / 17,496

Plus client-side toggles: CCS L/M/H, 45Q On/Off, Geothermal L/M/H (CAISO only), Demand Growth (year + rate).

## Data Sources

- **EIA-930**: Hourly grid generation data (2021-2025, multi-year averaged)
- **NREL ATB 2024**: LCOE estimates for solar, wind, nuclear, battery, LDES
- **LBNL**: Utility-Scale Solar 2024, Wind Market Report 2024
- **FERC/ISO**: Wholesale electricity price averages (2023-2024)
- **eGRID**: Emission rate data
- **USGS**: Geothermal resource assessments
- **PJM SOM 2024**: LMP calibration data (merit-order stack, price distribution)

## License

MIT License. See [LICENSE](LICENSE) for details.
