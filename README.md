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
| **Step 1** | `step1_pfs_generator.py` | **PFS Generator** — Generates the Physics Feasible Space (PFS). Sweeps 4D resource mixes (clean firm, solar, wind, hydro) × procurement × battery × LDES, evaluates hourly generation vs. demand, computes match scores. Produces 21.4M physics-validated mixes. | Only if dispatch logic, generation profiles, or demand curves change. |
| **Step 2** | `step2_efficient_frontier.py` | **Efficient Frontier (EF)** — Extracts the efficient frontier from the PFS. Filters existing generation utilization, minimizes procurement, removes strictly dominated mixes. Reduces 21.4M → ~1.8M rows. | Only if PFS changes or filtering criteria change. |
| **Step 3** | `step3_cost_optimization.py` | **Cost Optimization** — Vectorized cross-evaluation of all EF mixes under 5,832 sensitivity combos. Merit-order tranche pricing for clean firm. Extracts archetypes and sweeps demand growth scenarios. | When cost assumptions, LCOE tables, or sensitivity toggles change. |
| **Step 4** | `step4_postprocess.py` | **Post-Processing** — NEISO gas constraint, CCS vs LDES crossover analysis, CO₂/MAC calculations. Produces final corrected results. | When Step 3 outputs change. |

**Key principle**: Step 1 is expensive (hours of compute). Step 2 takes ~40 seconds. Steps 3–4 + post-processing are cheap (minutes). Changing cost assumptions only requires Steps 3–4 + post-processing.

### Post-Processing Scripts

| Script | What It Does |
|--------|-------------|
| `recompute_co2.py` | Dispatch-stack emission model. Merit-order fuel retirement (coal→oil→gas). Coal/oil capped at 2025 absolute TWh. |
| `compute_mac_stats.py` | MAC statistics: P10/P50/P90 fan, stepwise marginal, ANOVA decomposition, crossover analysis. |
| `generate_shared_data.py` | Generates `dashboard/js/shared-data.js` with all dashboard data constants. |

### Running the Pipeline

```bash
# Requires Python 3.10+, numpy, pyarrow
pip install numpy pyarrow

# Step 1: Generate PFS (expensive — only if physics change)
python step1_pfs_generator.py

# Step 2: Extract efficient frontier (~40s)
python step2_efficient_frontier.py

# Step 3: Cost optimization (~3 min)
python step3_cost_optimization.py

# Step 4: Post-processing (~seconds)
python step4_postprocess.py

# Post-processing: CO₂, MAC stats, dashboard data
python recompute_co2.py
python compute_mac_stats.py
python generate_shared_data.py
```

### Key Acronyms

- **PFS** — Physics Feasible Space: the full set of physically valid resource mixes (Step 1 output)
- **EF** — Efficient Frontier: the reduced set of non-dominated mixes that could be optimal under any cost assumption (Step 2 output)

## Project Structure

```
hourly-cfe-optimizer/
├── step1_pfs_generator.py        # Step 1: PFS generator (physics)
├── step2_efficient_frontier.py   # Step 2: Efficient frontier extraction
├── step3_cost_optimization.py    # Step 3: Cost optimization
├── step4_postprocess.py          # Step 4: Post-processing
├── data/
│   ├── physics_cache_v4.parquet  # PFS (21.4M rows, Step 1 output)
│   ├── pfs_post_ef.parquet       # PFS post-EF (1.8M rows, Step 2 output)
│   ├── eia_demand_profiles.json
│   ├── eia_generation_profiles.json
│   ├── eia_hourly_*.json         # Per-ISO hourly data (2024-2025)
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
