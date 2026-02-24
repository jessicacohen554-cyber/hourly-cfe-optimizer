# Hourly CFE Cost Optimizer

An interactive dashboard and optimization engine for analyzing **hourly clean energy procurement strategies** across five major US ISO regions.

The optimizer finds the least-cost mix of clean firm (nuclear/geothermal), solar, wind, hydro, CCS-CCGT, battery, and LDES to meet hourly demand at various Clean Energy Matching (CEM) targets (50%–100%), then visualizes the results in a scrollytelling dashboard.

## Regions Covered

| ISO | Region |
|-----|--------|
| CAISO | California |
| ERCOT | Texas |
| PJM | Mid-Atlantic |
| NYISO | New York |
| NEISO | New England |

## Quick Start

**To view the dashboard:**

```bash
cd dashboard
python -m http.server 8000
# Open http://localhost:8000/index.html
```

## 4-Step Optimizer Pipeline

The optimizer runs as a 4-step pipeline. Each step is independent — only re-run the step whose inputs changed.

| Step | Script | What It Does | When to Re-run |
|------|--------|-------------|---------------|
| **Step 1** | `scripts/step1_pfs_generator.py` | **PFS Generator** — Generates the Physics Feasible Space (PFS). Sweeps 4D resource mixes (clean firm, solar, wind, hydro) × procurement × battery × LDES, evaluates hourly generation vs. demand, computes match scores. Produces 21.4M physics-validated mixes. | Only if dispatch logic, generation profiles, or demand curves change. |
| **Step 2** | `scripts/step2_efficient_frontier.py` | **Efficient Frontier (EF)** — Extracts the efficient frontier from the PFS. Filters existing generation utilization, minimizes procurement, removes strictly dominated mixes. Reduces 21.4M → ~1.8M rows. | Only if PFS changes or filtering criteria change. |
| **Step 3** | `scripts/step3_cost_optimization.py` | **Cost Optimization** — Vectorized cross-evaluation of all EF mixes under 5,832 sensitivity combos. Merit-order tranche pricing for clean firm. Extracts archetypes and sweeps demand growth scenarios. | When cost assumptions, LCOE tables, or sensitivity toggles change. |
| **Step 4** | `scripts/step4_gas_ccs_adjustement.py` | **Post-Processing** — NEISO gas constraint, CCS vs LDES crossover analysis, CO₂/MAC calculations. Produces final corrected results. | When Step 3 outputs change. |

**Key principle**: Step 1 is expensive (hours of compute). Step 2 takes ~40 seconds. Steps 3–4 + post-processing are cheap (minutes). Changing cost assumptions only requires Steps 3–4 + post-processing.

### Post-Processing Scripts

| Script | What It Does |
|--------|-------------|
| `scripts/step5_PP4_recompute_co2.py` | Dispatch-stack emission model. Merit-order fuel retirement (coal→oil→gas). Coal/oil capped at 2025 absolute TWh. |
| `scripts/step5_PP5_compute_mac_stats.py` | MAC statistics: P10/P50/P90 fan, stepwise marginal, ANOVA decomposition, crossover analysis. |
| `scripts/step6_generate_shared_data.py` | Generates `dashboard/js/shared-data.js` with all dashboard data constants. |

### Running the Pipeline

```bash
# Requires Python 3.10+, numpy, pyarrow
pip install numpy pyarrow

# Step 1: Generate PFS (expensive — only if physics change)
python scripts/step1_pfs_generator.py

# Step 2: Extract efficient frontier (~40s)
python scripts/step2_efficient_frontier.py

# Step 3: Cost optimization (~3 min)
python scripts/step3_cost_optimization.py

# Step 4: Post-processing (~seconds)
python scripts/step4_gas_ccs_adjustement.py

# Step 5: Post-processing — CO₂, MAC stats
python scripts/step5_PP4_recompute_co2.py
python scripts/step5_PP5_compute_mac_stats.py

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
├── scripts/step1_pfs_generator.py        # Step 1: PFS generator (physics)
├── scripts/step2_efficient_frontier.py   # Step 2: Efficient frontier extraction
├── scripts/step3_cost_optimization.py    # Step 3: Cost optimization
├── scripts/step4_gas_ccs_adjustement.py          # Step 4: Post-processing
├── data/
│   ├── step1-pfs-parquets/       # PFS per-ISO/threshold (Step 1 output)
│   ├── pfs_post_ef.parquet       # PFS post-EF (1.8M rows, Step 2 output)
│   ├── EIA 930 Data/
│   │   ├── eia_demand_profiles.json
│   │   ├── eia_generation_profiles.json
│   │   ├── eia_hourly_*.json     # Per-ISO hourly generation data
│   │   └── eia_demand_*.json     # Per-ISO hourly demand data
│   ├── egrid_emission_rates.json
│   └── ...
├── dashboard/
│   ├── index.html                # Homepage (scrollytelling)
│   ├── dashboard.html            # Interactive cost optimizer
│   ├── abatement_dashboard.html  # CO₂ Abatement Analysis
│   ├── research_paper.html       # Standalone research paper
│   ├── optimizer_methodology.html # Methodology documentation
│   ├── overprocure_results.json  # Step 3 output (optimization results)
│   └── demand_growth_results.json # Step 3 output (demand growth sweep)
└── README.md
```

## Resource Types

- **Clean Firm**: Nuclear (seasonal-derated) + geothermal (CAISO only, capped at 5 GW)
- **Solar**: EIA regional hourly profile
- **Wind**: EIA regional hourly profile
- **Hydro**: EIA regional hourly profile (capped by region, existing only, wholesale-priced)
- **CCS-CCGT**: Implicit 5th resource (100% − sum of above four), flat baseload profile
- **Battery**: 4hr Li-ion, 85% round-trip efficiency, daily-cycle dispatch
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

- **EIA-930**: Hourly grid generation data (2024-2025)
- **NREL ATB 2024**: LCOE estimates for solar, wind, nuclear, battery, LDES
- **LBNL**: Utility-Scale Solar 2024, Wind Market Report 2024
- **FERC/ISO**: Wholesale electricity price averages (2023-2024)
- **eGRID**: Emission rate data
- **USGS**: Geothermal resource assessments

## License

MIT License. See [LICENSE](LICENSE) for details.
