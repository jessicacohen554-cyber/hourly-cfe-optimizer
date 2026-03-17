# Market Simulator — Data Directory Setup

This folder holds all input data for the market simulator. The tool runs with
built-in synthetic profiles by default, but for production-quality results you
need to drop in real data from EIA, EPA CAMPD, and the hourly-cfe-optimizer
pipeline.

## Quick Start

1. Drop your data files into the directories described below
2. Run `run.bat` (Windows) or `bash run.sh` (Mac/Linux)
3. The server starts at http://127.0.0.1:8000

The simulator will automatically detect and use whatever data files are present.
Missing data falls back to built-in synthetic profiles.


## Directory Structure

```
market-simulator/
└── data/
    ├── profiles/                       ← REQUIRED (ships with tool)
    │   ├── eia_demand_profiles.json    # 8760-hour demand by ISO
    │   ├── eia_gen_profiles.json       # 8760-hour generation by ISO + resource
    │   ├── eia_fossil_mix.json         # Fossil fuel mix shares by ISO
    │   └── egrid_emission_rates.json   # eGRID CO2 rates by ISO
    │
    ├── constellation_fleet.json        ← REQUIRED (ships with tool)
    │                                   # 202 Constellation/Calpine plants
    │
    ├── step2.1-ef/                     ← OPTIONAL (drop in from pipeline)
    │   └── step_2_1_ef_{ISO}.parquet   # Efficient frontier mixes per ISO
    │
    ├── step2.2-cost/                   ← OPTIONAL (drop in from pipeline)
    │   └── step_2_2a_CO_{ISO}.parquet  # Cost-optimized results per ISO
    │
    ├── eia-860/                        ← OPTIONAL (enhances plant-level detail)
    │   ├── TX/
    │   │   └── eia860_TX.json
    │   ├── IL/
    │   │   └── eia860_IL.json
    │   └── {STATE}/                    # One subfolder per state
    │       └── eia860_{STATE}.json     # EIA-860 plant characteristics
    │
    ├── eia-923/                        ← OPTIONAL (enhances generation data)
    │   ├── TX/
    │   │   ├── eia923_TX_2023-01.json
    │   │   ├── eia923_TX_2023-02.json
    │   │   └── ...                     # Monthly files, 2023-01 through 2025-12
    │   └── {STATE}/
    │       └── eia923_{STATE}_{YYYY-MM}.json
    │
    ├── epa-campd/                      ← OPTIONAL (enhances emissions data)
    │   ├── TX/
    │   │   ├── TX_2023-01.json
    │   │   ├── TX_2023-02.json
    │   │   └── ...                     # Monthly files
    │   └── {STATE}/
    │       └── {STATE}_{YYYY-MM}.json
    │
    ├── eia-natgas-prices/              ← OPTIONAL (enhances fuel price data)
    │   ├── delivered-to-power/
    │   │   └── natgas_delivered_to_power_2023_2025.json
    │   └── spot-futures/
    │       └── natgas_spot_futures_{YYYY}.json
    │
    └── results/                        ← AUTO-GENERATED (simulation outputs)
        └── market_simulation_results.json
```


## What Each Dataset Does

### Required (ships with tool — don't delete)

| File | Purpose |
|------|---------|
| `profiles/eia_demand_profiles.json` | 8760-hour demand curves per ISO (TWh → MW) |
| `profiles/eia_gen_profiles.json` | Solar/wind/hydro generation profiles per ISO |
| `profiles/eia_fossil_mix.json` | Coal/gas/oil capacity shares per ISO |
| `profiles/egrid_emission_rates.json` | CO2 emission rates (tons/MWh) per ISO |
| `constellation_fleet.json` | 202 Constellation/Calpine plants with capacity, ISO, fuel type, CCS eligibility |

### Optional — Pipeline Outputs (drop in for enhanced results)

| Directory | Source | Effect |
|-----------|--------|--------|
| `step2.1-ef/` | Step 2.1 efficient frontier | Enables resource mix optimization overlays |
| `step2.2-cost/` | Step 2.2 cost optimization | Enables cost scenario analysis across 5,832 sensitivity combos |

### Optional — Raw Government Data (drop in for full fidelity)

| Directory | Source | Effect |
|-----------|--------|--------|
| `eia-860/` | EIA Form 860 | Real plant characteristics (capacity, fuel, location, age) |
| `eia-923/` | EIA Form 923 | Real monthly generation and fuel consumption by plant |
| `epa-campd/` | EPA CAMPD | Real hourly emissions (CO2, NOx, SOx) by plant |
| `eia-natgas-prices/` | EIA natural gas | Real historical gas prices for fuel cost calibration |


## File Naming Conventions

### State Abbreviations (used in subdirectory names)

The tool uses **2-letter US state abbreviations** as subdirectory names. Each
state maps to one or more ISOs internally. The key states for the 7 ISOs:

| ISO | Key States |
|-----|-----------|
| ERCOT | TX |
| PJM | PA, NJ, DE, MD, VA |
| CAISO | CA |
| NYISO | NY |
| NEISO | MA, ME, NH |
| MISO | IL, AR, AL |
| SPP | AR, AZ, OR |

### EIA-923 Monthly Files

Format: `eia923_{STATE}_{YYYY-MM}.json`

Example: `eia923_TX_2024-06.json`

Should contain monthly generation records with fields like:
`plant_id`, `plant_name`, `fuel_type`, `net_generation_mwh`, `year`, `month`

### EPA CAMPD Monthly Files

Format: `{STATE}_{YYYY-MM}.json`

Example: `TX_2024-06.json`

Should contain hourly emissions records with fields like:
`orispl_code`, `unit_id`, `op_date`, `op_hour`, `co2_mass_tons`, `nox_mass_lbs`, `so2_mass_lbs`, `gross_load_mw`

### EIA-860 Plant Files

Format: `eia860_{STATE}.json`

Example: `eia860_TX.json`

Should contain plant characteristic records with fields like:
`plant_id`, `plant_name`, `state`, `capacity_mw`, `fuel_type`, `year_built`, `status`

### Step 2.1/2.2 Parquets

These come directly from the hourly-cfe-optimizer pipeline. Just copy the
parquet files from `data/step2.1-ef/` and `data/step2.2-cost/` into the
corresponding directories here.

Naming: `step_2_1_ef_{ISO}.parquet` and `step_2_2a_CO_{ISO}.parquet`
where `{ISO}` is one of: CAISO, ERCOT, PJM, NYISO, NEISO, MISO, SPP


## Minimal vs Full Data

| Mode | What You Need | What You Get |
|------|---------------|--------------|
| **Minimal** (demo) | Nothing extra — built-in synthetic profiles | Full UI with realistic shapes, synthetic plant data |
| **Pipeline only** | Drop in step2.1 + step2.2 parquets | Real optimizer results + cost scenarios |
| **Full fidelity** | All of the above + EIA-923 + EPA CAMPD | Real plant-level dispatch, emissions, CCS analysis |
