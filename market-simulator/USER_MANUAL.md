# Market Simulation Screening Tool — User Manual

## Quick Start

### Option 1: Double-Click Launcher (Recommended)

**Windows:**
Double-click `run.bat`. This calls `python app-startup\start.py`, which installs dependencies and launches the server.

**Mac/Linux:**
```bash
cd app-startup
python start.py
```

### Option 2: Manual

```bash
pip install -r app-startup/requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
Then open http://127.0.0.1:8000 in your browser.

---

## Directory Structure

```
market-simulator/
├── USER_MANUAL.md              # This file
├── run.bat                     # Windows launcher
├── app-startup/                # Startup files
│   ├── start.py                # Application launcher script
│   ├── requirements.txt        # Python dependencies
│   └── run.sh                  # Mac/Linux launcher
├── backend/                    # FastAPI backend
├── frontend/                   # HTML/CSS/JS frontend
│   ├── brand-assets/           # Logos and brand imagery
│   └── CONSTELLATION_STYLE_GUIDE.md  # UI style reference
├── custom-user-inputs/         # User-provided override CSVs
├── data/                       # Model input data (EIA, eGRID, etc.)
├── results/                    # Simulation output directories
└── scripts/                    # Utility scripts
```

---

## System Requirements

- **Python 3.9+** (3.11 recommended)
- **pip** (comes with Python)
- **Modern web browser** (Chrome, Firefox, Edge, Safari)
- **No other software required** — all dependencies install automatically via pip

---

## Tool Overview

The Market Simulation Screening Tool models electricity market economics across 7 US ISOs. Given market preconditions (fuel prices, carbon costs, clean energy LCOEs, incentives), it computes:

- Merit-order dispatch and hourly LMPs
- Generator profitability (which units retire vs. stay profitable)
- Clean energy deployment economics (what can be built profitably)
- Nuclear retirement risk under different revenue assumptions
- CCS breakeven carbon price analysis
- Plant-level economics for every generator in the EIA 860 fleet

### Three-Page Flow

1. **Guide** (`/`) — Instructions and tool overview
2. **Setup** (`/setup`) — Input parameters and run simulation
3. **Results** (`/results`) — Charts, tables, and analysis

### User Interface

The tool features a blue navigation header (#2372B9) with white text and the Constellation Energy logo. All input fields include info-hover tooltips that explain what each parameter does and why it matters. Section headers use larger eyebrow typography for clear visual grouping.

ISO selection is single-select — choose one ISO per simulation run.

---

## Setup Page Layout

The Setup page organizes inputs into 5 rows:

### Row 1: Mode / Region / Nuclear Threshold

| Parameter | Options | Description |
|-----------|---------|-------------|
| Simulation Mode | Snapshot / Trajectory / Sweep | See Simulation Mode table below |
| ISO Region | Single-select from 7 ISOs | CAISO, ERCOT, PJM, NYISO, NEISO, MISO, SPP |
| Nuclear Retirement Threshold | $10–60/MWh | Revenue floor below which nuclear retires |

### Row 2: Fuel Prices / Emissions

| Parameter | Options | Description |
|-----------|---------|-------------|
| Gas Price | Low / Medium / High / Custom | Natural gas $/MMBtu |
| Coal Price | Low / Medium / High / Custom | Coal $/MMBtu |
| Oil Price | Low / Medium / High / Custom | Oil $/MMBtu |
| Carbon Price | Presets or custom $/ton | CO2 emission cost adder |

### Row 3: Resource LCOEs

Levelized costs for new-build resources, including storage, fossil new-build, and transmission overrides.

| Resource | Low | Medium | High |
|----------|-----|--------|------|
| Solar | $24.5 | $31.8 | $42.0 |
| Wind | $28.0 | $34.5 | $46.0 |
| Offshore Wind | $68.0 | $82.0 | $105.0 |
| Nuclear | $72.0 | $89.0 | $115.0 |
| CCS-CCGT | $72.0 | $86.0 | $110.0 |
| Battery 4hr | $220 | $295 | $380 ($/kW-yr) |
| Battery 8hr | $350 | $456 | $580 ($/kW-yr) |
| LDES 100hr | $150 | $220 | $300 ($/kW-yr) |
| Fossil New-Build | Configurable | LCOE for new gas plants |
| Transmission | None / Low / Med / High | Transmission cost adder override |

### Row 4: Incentives

| Parameter | Default | Description |
|-----------|---------|-------------|
| PTC Wind/Solar | $26/MWh | Production Tax Credit |
| PTC New Nuclear | $26/MWh | 45Y new nuclear credit |
| 45U Max Credit | $15/MWh | Existing nuclear CfD cap |
| 45U Floor | $40/MWh | Revenue floor for existing nuclear |
| ITC | 30% | Investment Tax Credit (storage, offshore) |
| 45Q | On / Off | CCS carbon capture tax credit |
| PPA Availability | Low / Med / High | Clean energy offtake economics |

### Row 5: Market Parameters

| Parameter | Options | Description |
|-----------|---------|-------------|
| Demand Growth | Low (0.5%) / Med (1.5%) / High (2.5%) / Custom | Annual load growth rate |
| Gas Friction | Low / Med / High | ESG/permitting headwinds for new gas |

---

## Input Parameters Reference

### Simulation Mode

| Mode | Description | Speed |
|------|-------------|-------|
| **Snapshot** | Single point-in-time simulation | 5–15 seconds |
| **Trajectory** | Multi-year projection with learning curves (2030–2050) | 30–120 seconds |
| **Sweep** | 270-scenario parametric sweep | 5–30 minutes |

### Fuel Prices ($/MMBtu)

| Fuel | Low | Medium | High | What It Affects |
|------|-----|--------|------|-----------------|
| Gas | $2.00 | $3.50 | $6.00 | CCGT and CT marginal costs |
| Coal | $2.00 | $2.25 | $2.50 | Coal steam marginal costs |
| Oil | $8.00 | $10.50 | $13.00 | Oil CT peaker costs (scarcity pricing) |

### Carbon Price ($/ton CO2)

| Preset | Value | Context |
|--------|-------|---------|
| None | $0 | No carbon pricing |
| RGGI | $5.50 | Current RGGI allowance price |
| Full RGGI | $14.00 | RGGI with broader participation |
| EPA SCC | $51.00 | EPA Social Cost of Carbon |
| EU ETS | $100.00 | EU Emissions Trading System range |
| Rennert | $185.00 | Academic estimate (Rennert et al.) |

---

## Custom Data Inputs

Override model defaults with your own price data:

1. Navigate to the `custom-user-inputs/` folder
2. Copy a template (e.g., `template_lmp_hourly.csv` → `lmp_hourly.csv`)
3. Replace values with your data
4. Enable the corresponding toggle on the Setup page

### Template Files

| File | Rows | Columns | Units |
|------|------|---------|-------|
| `template_lmp_hourly.csv` | 8760 | 7 ISOs | $/MWh |
| `template_capacity_prices.csv` | 12 | 7 ISOs | $/MW-day |
| `template_rec_prices.csv` | 12 | 7 ISOs | $/MWh |
| `template_fuel_prices_gas.csv` | 12 | 7 ISOs | $/MMBtu |
| `template_fuel_prices_coal.csv` | 12 | 7 ISOs | $/MMBtu |
| `template_fuel_prices_oil.csv` | 12 | 7 ISOs | $/MMBtu |

### Validation Rules

- Headers must include: CAISO, ERCOT, PJM, NYISO, NEISO, MISO, SPP
- No blank cells or NaN values
- Correct row count (8760 for hourly, 12 for monthly)
- Numeric values only

---

## Understanding Results

### KPI Cards

| Metric | Description |
|--------|-------------|
| Market Clean % | Clean generation share after economic dispatch |
| Avg LMP | Time-weighted average energy price |
| Emissions | Annual CO2 from fossil fleet |
| Demand | Total annual load |
| Nuclear Rev | Nuclear total revenue stack (energy + capacity + PTC) |

### Key Charts

| Chart | What It Shows |
|-------|---------------|
| LMP & Capacity Revenue | Price profile (hourly or yearly trajectory) |
| Supply Stack | Generation mix by fuel (TWh) |
| Merit-Order Stack | Fossil generators sorted by cost, with LMP cutoff |
| Generator Economics | Per-unit dispatch, revenue, and profitability |
| Profitability | Revenue vs. cost per resource (trajectory mode) |
| CCS Breakeven | Carbon price at which CCS beats unabated gas |
| Nuclear Revenue Stack | Energy + capacity + PTC breakdown |

### Status Indicators

| Status | Color | Meaning |
|--------|-------|---------|
| Profitable | Green | Margin > $5/MWh — economically viable |
| Marginal | Yellow | Margin $-5 to $5/MWh — at risk |
| Retiring | Red | Margin < $-5/MWh — uneconomic |

---

## Plant-Level Results

Each simulation produces a `detailed_plant_results.csv` file with per-plant economics based on the EIA 860 fleet database. This provides granular, plant-by-plant analysis of the simulation outcomes.

### Plant-Level CSV Columns

| Column | Description |
|--------|-------------|
| entity | Plant owner/operator name |
| plant_name | EIA-registered plant name |
| capacity_mw | Nameplate capacity (MW) |
| heat_rate | Plant heat rate (Btu/kWh) |
| latitude | Plant latitude |
| longitude | Plant longitude |
| capacity_factor | Annual capacity factor (0–1) |
| generation_mwh | Annual generation (MWh) |
| fuel_consumed | Annual fuel consumption (MMBtu) |
| co2_tons | Annual CO2 emissions (short tons) |
| nox_tons | Annual NOx emissions (short tons) |
| sox_tons | Annual SOx emissions (short tons) |
| revenue | Annual revenue ($) |
| vom_cost | Annual variable O&M cost ($) |
| fuel_cost | Annual fuel cost ($) |
| profit | Annual profit (revenue minus costs, $) |
| status | Plant economic status (see below) |

### Plant Status Categories

| Status | Meaning |
|--------|---------|
| **operating** | Plant is profitable and continues normal operations |
| **at_risk** | Plant is marginally economic — small market shifts could push it into retirement |
| **stranded** | Plant is uneconomic under the simulated conditions and would likely retire |

These status categories enable identification of which specific plants face economic pressure under different market scenarios, supporting fleet transition planning and stranded asset analysis.

---

## Results Directory

Every simulation run saves to `results/run_NNN_{ISO}/`:

```
results/
├── run_001_PJM/
│   ├── inputs.txt                  # All user inputs (formatted text)
│   ├── narrative.txt               # Auto-generated results interpretation
│   ├── results_data.csv            # Full result data (generator economics, supply mix)
│   ├── detailed_plant_results.csv  # Per-plant EIA 860 economics
│   ├── input_parameters.csv        # Tabular copy of parameter selections
│   ├── charts/                     # PNG images (after clicking "Export Charts")
│   │   ├── chartLMPTimeSeries.png
│   │   ├── chartSupplyStack.png
│   │   └── ...
│   └── custom_*.csv                # Copies of custom input files (if used)
├── run_002_CAISO/
│   └── ...
```

### Exporting Charts

Click "Export Charts" on the results page to save PNG images of all charts to the run's `charts/` folder.

### Downloading Data

Click "Download CSV" to download the full results as a CSV file.

---

## Troubleshooting

### "Python not found"
Install Python 3.9+ from https://www.python.org/downloads/. Ensure it's on your system PATH.

### "Module not found" errors
Run `pip install -r app-startup/requirements.txt` manually from the market-simulator directory.

### Port 8000 already in use
Another application is using port 8000. Either stop it or edit the PORT variable in `run.bat` or `app-startup/start.py`.

### "No simulation results" on Results page
You need to run a simulation from the Setup page first. Results are stored in the browser's session storage — they clear when you close the tab.

### Charts not rendering
If Plotly.js fails to load from the local file, ensure you have internet access (CDN fallback) or run the dependency download step first.

### Custom input file rejected
Check that your CSV has exactly the right number of rows and all 7 ISO columns. See the `readme.txt` in `custom-user-inputs/` for format details.

---

## Technical Notes

- **Backend**: FastAPI (Python) serving REST API + static HTML
- **Frontend**: Vanilla HTML/CSS/JS with Plotly.js for charts; Constellation Energy branding
- **Model**: Merit-order dispatch with hourly LMP computation
- **Data**: EIA 860 plant fleet, EIA demand profiles, eGRID emission rates, fossil fleet data
- The tool runs entirely locally — no data is sent to external servers
- All input fields include info-hover tooltips with parameter explanations
