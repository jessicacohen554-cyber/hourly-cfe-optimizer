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
- **ORDC scarcity pricing** — Operating Reserve Demand Curve pricing replaces statistical demand-quantile overlays with structural reserve-based pricing: `price = marginal_cost + VOLL × LOLP(reserves)`. ISO-calibrated VOLL ($2,000–$5,000/MWh) and reserve targets.
- **VRE cannibalization feedback** — Solar and wind energy revenue uses time-matched LMP (capture rates), not average LMP. Capture rates decline as VRE penetration increases (duck curve effect). ORDC scarcity hours create a revenue floor preventing total collapse.
- **Zonal LMP decomposition** — pipe-and-bubble zonal model (2–5 zones/ISO) with ORDC-integrated scarcity pricing per zone. Captures intra-ISO transmission congestion.
- **LP storage co-dispatch** — simultaneous optimization of battery 4hr/8hr, LDES 100hr, and green H₂ via linear programming
- **Inter-regional flows** — hourly import/export profiles reduce self-sufficiency requirements (NEISO HQ hydro, CAISO PNW imports)
- **Demand response** — price-elastic load curtailment when LMP exceeds ISO-specific trigger prices. ORDC-linked dynamic triggers available.
- **IPM trigger indicators** — automated flags when results cross thresholds where screening approximations become unreliable. 6 trigger types with Medium/High severity recommend production-model validation.
- **Tech-differentiated queue caps** — per-technology interconnection limits (solar, wind, offshore, nuclear, CCS) based on LBNL "Queued Up 2024" completion rates
- **Data tier warnings** — color-coded indicators show data quality (synthetic vs. EIA-930 vs. pipeline parquets vs. full plant-level) so users understand result reliability
- **Confidence visualization** — trajectory results show calibrated/extrapolated/uncertain zones with widening uncertainty bands
- **Trajectory backtesting** — backward-looking validation against 2020–2024 observed outcomes

### Six-Page Flow

1. **Guide** (`/`) — Instructions, tool overview, and data setup documentation
2. **Setup** (`/setup`) — Input parameters and run simulation
3. **Fleet Config** (`/fleet-config`) — Constellation/Calpine asset configuration (toggle plants between Operating / Retired / CCS Retrofit)
4. **Results** (`/results`) — Charts, tables, and analysis
5. **CCS Emissions** (`/emissions`) — CCS emissions dashboard with plant-level dispatch and capture analysis
6. **IPP Report** (`/ipp-report`) — Constellation fleet transition report with 9-section executive summary

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

### Row 6: Advanced Options

| Parameter | Options | Description |
|-----------|---------|-------------|
| Inter-Regional Imports | On / Off | Include historical cross-border power flows from EIA-930 data. On = imports reduce residual demand and fossil fleet sizing. Off = copper-plate isolation (each ISO self-sufficient). |
| Demand Response | Off / Low / Med / High | Price-elastic load curtailment. Off = inelastic demand. Low = 50% registered DR participation with higher trigger prices. Medium = 70% participation (default DR values). High = 90% participation with lower trigger prices. ORDC-linked mode uses dynamic triggers based on reserve scarcity. |
| Scarcity Pricing Mode | ORDC / Demand-Quantile | **ORDC** (default): Structural reserve-based pricing using LOLP sigmoid and ISO-calibrated VOLL. Produces realistic scarcity spikes tied to reserve margins. **Demand-Quantile**: Statistical overlay based on demand percentile — simpler but less physically grounded. |

---

## Input Parameters Reference

### Simulation Mode

| Mode | Description | Speed |
|------|-------------|-------|
| **Snapshot** | Single point-in-time simulation | 5–15 seconds |
| **Trajectory** | Multi-year projection with learning curves (2030–2050) | 30–120 seconds |
| **Sweep** | 1,215-scenario parametric sweep | 15–90 minutes |

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

### New-Build Fossil Parameters

| Parameter | Default | Valid Range | Description |
|-----------|---------|-------------|-------------|
| **New Fossil Cost Level** | Medium | Low / Medium / High | CAPEX tier for new-build fossil (swept in parametric mode) |
| **New Fossil Builds** | On | On / Off | Enable/disable new fossil construction |
| **Min CF — CCGT** | 30% | 5–90% | Minimum capacity factor for new CCGT investment |
| **Min CF — CT** | 5% | 1–50% | Minimum capacity factor for new peaker CT investment |
| **Min CF — Coal** | 60% | 20–90% | Minimum capacity factor for new coal investment |

In single trajectory mode, users can also override per-type CAPEX ($/kW-yr) via the API `new_fossil_capex_override` field. Coal new-build is blocked in CAISO, NYISO, and NEISO.

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
| LMP & Capacity Revenue | Price profile (hourly or yearly trajectory). Confidence zone background bands in trajectory mode. |
| Supply Stack | Generation mix by fuel (TWh). Confidence zone annotations label calibrated vs. extrapolated ranges. |
| Merit-Order Stack | Fossil generators sorted by cost, with LMP cutoff |
| Generator Economics | Per-unit dispatch, revenue, and profitability |
| Profitability | Revenue vs. cost per resource (trajectory mode) |
| CCS Breakeven | Carbon price at which CCS beats unabated gas |
| Nuclear Revenue Stack | Energy + capacity + PTC breakdown |
| Zonal LMP Spread | Per-zone average LMP and price spread vs system average (when fleet data available) |

### Demand Response Metrics

When DR is enabled (Low/Med/High), four KPI cards appear below the main results:

| Metric | Description |
|--------|-------------|
| DR Curtailed (GWh/yr) | Total demand response energy curtailed annually |
| DR Peak (GW) | Maximum simultaneous demand response activation |
| DR Hours/yr | Number of hours demand response was triggered |
| DR Avg Price ($/MWh) | Average LMP during demand response events |

### Confidence Indicators (Trajectory Mode)

Trajectory results include visual confidence indicators:

| Zone | Years | Color | Meaning |
|------|-------|-------|---------|
| Calibrated | 2025–2030 | Green | Based on calibrated 2024 market data and near-term policy |
| Moderate Extrapolation | 2030–2040 | Yellow | Technology costs and market structure may diverge from calibration |
| High Uncertainty | 2040–2060 | Red | Multiple compounding uncertainties — scenario exploration, not forecast |

These appear as colored background bands on trajectory charts and as a badge on year-level KPI cards.

### VRE Capture Rates

When VRE cannibalization feedback is active, results include per-resource capture rates:

| Metric | Description |
|--------|-------------|
| Solar Capture Rate | Ratio of solar generation-weighted LMP to time-average LMP. Falls below 1.0 as solar penetration increases (duck curve). Typical range: 0.6–1.0. |
| Wind Capture Rate | Ratio of wind generation-weighted LMP to time-average LMP. Generally closer to 1.0 than solar due to more even diurnal profile. |

Capture rates below 0.7 indicate severe revenue depression — the model automatically dampens further VRE deployment when this occurs.

### IPM Trigger Indicators

When simulation results cross thresholds where the screening model's approximations become unreliable, trigger cards appear on the results page:

| Trigger | Medium Threshold | High Threshold | What to Do |
|---------|-----------------|----------------|------------|
| VRE Cannibalization | >40% VRE share | >60% VRE share | Run production dispatch with curtailment (PLEXOS, GenX) |
| Tight RA Margin | <10% reserve margin | <5% reserve margin | Run UC-constrained dispatch (IPM, PLEXOS) |
| High Congestion | >2× queue completion | >3× queue completion | Run zonal/nodal dispatch model |
| Storage Dominance | >15% of energy served | >25% of energy served | Run co-optimized storage dispatch (GenX) |
| Retirement Cascade | >20% fleet retired/period | >35% fleet retired/period | Run plant-level retirement model (IPM) |
| Nuclear at Risk | Revenue within $5/MWh of retirement cliff | Always high | Run plant-level nuclear economics |

- **High severity** triggers appear as red-bordered cards with "Production Modeling Recommended" header.
- **Medium severity** triggers appear as amber-bordered cards.
- Triggers can be dismissed via the close button (stored in session).

### Data Tier Indicators

The Setup page shows color-coded data quality indicators:

| Indicator | Green | Amber | Red |
|-----------|-------|-------|-----|
| Resource Mix | Pipeline parquets loaded | — | Synthetic data (illustrative only) |
| Interchange | EIA-930 profiles loaded | — | Not loaded |
| Zonal Config | Validated zone assignments | Hardcoded defaults | — |
| Fleet Data | Plant-level EIA 860 | Aggregated fleet-average | — |
| DR Params | Calibrated ISO-specific | Default values | — |

When running on synthetic data (Minimal tier), results are illustrative only — suitable for UI exploration but not for screening decisions. The results page shows a warning banner.

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

## Fleet Configuration Page

The Fleet Config page (`/fleet-config`) lets you modify the status of individual Constellation/Calpine plants before running a simulation.

### Plant Statuses

| Status | Effect on Simulation |
|--------|---------------------|
| **Operating** (default) | Plant dispatches normally in merit order |
| **Retired** | Plant capacity zeroed — removed from dispatch |
| **CCS Retrofit** | 95% CO2 capture, 14% heat rate penalty, 80% CF floor (available for 39 eligible CCGT plants only) |

### Page Layout

- Plants grouped by ISO → fuel type (expandable sections)
- Fossil sections (coal, gas CCGT, gas CT, oil) expanded by default
- Clean sections (nuclear, solar, wind, hydro) collapsed by default
- Search/filter bar for finding specific plants
- Bulk controls: "Set All Coal to Retired", "Set All CCGT to CCS Retrofit", "Reset to Defaults"

Fleet selections persist in browser localStorage and are included in the simulation request when you run from the Setup page. The Setup page shows a summary badge: "X plants modified (Y retired, Z CCS retrofit)".

---

## CCS Emissions Dashboard

The Emissions page (`/emissions`) displays CCS analysis results after running a simulation:

- **Plant selector**: Choose individual CCGT plants or view fleet aggregate
- **Fan bands**: P10/P50/P90 dispatch ranges across sweep scenarios
- **CCS delta chart**: Emissions reduction from CCS retrofit per plant per year
- **Stacked emissions**: System-wide CO2 by fuel type over time

This page uses market-sim dispatch results directly (not external parquets).

---

## IPP Report Page

The IPP Report (`/ipp-report`) generates a Constellation fleet transition executive summary with 9 sections covering fleet economics, CCS candidacy, retirement risk, and transition pathways.

---

## Plant-Level Data Setup

The simulator uses a **two-tier dispatch and emission model**:

### Tier 1 — Fleet-Average (Default, No Extra Data Needed)
Without plant-level data, the model uses aggregate parameters per unit type:

| Unit Type | Default Heat Rate (MMBtu/MWh) | CO2 Rate (tons/MWh) |
|-----------|-------------------------------|---------------------|
| Coal Steam | 10.0 | 0.95 |
| Gas CCGT | 7.0 | 0.37 |
| Gas CT | 10.5 | 0.55 |
| Oil CT | 10.5 | 0.65 |

### Tier 2 — Plant-Level (With EIA/EPA Data)
When plant-level data files are present, the model uses per-generator heat rates, vintage-adjusted unit commitment, and actual emission factors:

- Newer CCGTs (2015+): 30% minimum generation, lower start costs
- Older CCGTs (pre-2005): 50% minimum generation, higher start costs
- Per-plant heat rates from EIA 860/923 replace fleet averages
- Actual emission rates from EPA CAMPD replace eGRID averages

### Directory Setup for Plant-Level Data

To enable Tier 2 plant-level modeling, drop data files into the following directories under `market-simulator/data/`:

#### EIA Form 860 — Plant Characteristics
```
data/eia-860/{STATE}/eia860_{STATE}.json
```
**Example**: `data/eia-860/TX/eia860_TX.json`

**Required fields per record:**
- `plant_id` (int) — EIA plant ID (ORISPL code)
- `plant_name` (string) — Plant name
- `state` (string) — 2-letter state abbreviation
- `capacity_mw` (float) — Nameplate capacity in MW
- `fuel_type` (string) — EIA fuel type code (NG, BIT, SUB, DFO, etc.)
- `prime_mover` (string) — EIA prime mover code (CA, CT, ST, GT, etc.)
- `status` (string) — Operating status (OP = operating)
- `balancing_authority_code` (string) — BA code (ERCO, CISO, PJM, etc.)
- `heat_rate` (float) — Heat rate in Btu/kWh
- `online_year` (int) — Year unit came online

#### EIA Form 923 — Monthly Generation
```
data/eia-923/{STATE}/eia923_{STATE}_{YYYY-MM}.json
```
**Example**: `data/eia-923/TX/eia923_TX_2024-06.json`

**Required fields per record:**
- `plant_id` (int) — EIA plant ID
- `plant_name` (string)
- `fuel_type` (string)
- `net_generation_mwh` (float)
- `year` (int), `month` (int)

#### EPA CAMPD — Hourly Emissions
```
data/epa-campd/{STATE}/{STATE}_{YYYY-MM}.json
```
**Example**: `data/epa-campd/TX/TX_2024-06.json`

**Required fields per record:**
- `orispl_code` (int) — ORISPL plant code (matches EIA plant_id)
- `unit_id` (string)
- `op_date` (string), `op_hour` (int)
- `co2_mass_tons` (float)
- `nox_mass_lbs` (float), `so2_mass_lbs` (float)
- `gross_load_mw` (float)

### Key States per ISO

| ISO | Key States | BA Codes |
|-----|-----------|----------|
| ERCOT | TX | ERCO |
| PJM | PA, NJ, DE, MD, VA, OH, WV, IL, IN | PJM, AEP, COMED, DOM, DUK (partial) |
| CAISO | CA | CISO |
| NYISO | NY | NYIS |
| NEISO | MA, ME, NH, CT, RI, VT | ISNE |
| MISO | IL, IN, MI, WI, MN, IA, MO, AR, MS, LA | MISO, ALTE, CONS, NSP |
| SPP | AR, KS, OK, NE, NM (partial) | SWPP, KCPL, OKGE, SPS |

The model automatically maps plants to ISOs via their `balancing_authority_code` field using an internal BA-to-ISO mapping dictionary. You do not need to organize files by ISO — just drop state-level files and the model handles the mapping.

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
- **Model**: Merit-order dispatch with hourly LMP computation, ORDC scarcity pricing, VRE cannibalization feedback, zonal decomposition (scipy LP), LP storage co-dispatch
- **ORDC pricing**: `LOLP = 1/(1+exp(k×(reserves−target)))`, adder = VOLL × LOLP. Per-ISO parameters: ERCOT VOLL=$5,000, PJM=$3,700, MISO=$3,500, NYISO=$2,500, others=$2,000.
- **VRE cannibalization**: Per-resource capture rates computed from generation-weighted LMP. Sigmoid depression with ORDC floor on scarcity hours (>$50/MWh adder).
- **Storage dispatch**: Rolling-window LP co-optimization (scipy.optimize.linprog with HiGHS solver) for battery 4hr/8hr, LDES 100hr, green H₂ 1000hr. Sparse matrix construction for efficient constraint building. Falls back to greedy sequential dispatch if LP infeasible.
- **Zonal LMP**: Pipe-and-bubble model with 2–5 zones per ISO. ORDC integrated into zonal path. Inter-zonal transfer limits from ISO planning reports. LP solver per hour with dual prices as zonal marginal prices. Requires EIA-860 plant data for zone assignment.
- **Inter-regional flows**: Hourly net import profiles from EIA-930 data, scaled by demand growth but capped at firm import limits (CAISO 8 GW, PJM 5 GW, NEISO 3.5 GW, etc.)
- **Demand response**: Price-elastic curtailment with ISO-specific trigger prices and DR capacity from FERC Form 714 / ISO registrations. ORDC-linked dynamic trigger mode. Fully vectorized (numpy boolean masks).
- **IPM triggers**: 6 automated threshold checks flag when screening approximations are unreliable. Negligible compute overhead.
- **Tech queue caps**: Per-technology GW/yr caps (solar, wind, offshore, nuclear, CCS, geothermal) from LBNL "Queued Up 2024". 20% flex pool across technologies.
- **Confidence zones**: Year-based confidence classification (Calibrated 2025–2030, Moderate 2030–2040, High Uncertainty 2040+) with synthetic uncertainty bands on trajectory outputs.
- **Backtesting**: Backward-looking validation from 2020 conditions with actual historical fuel prices (EIA Henry Hub). Computes direction accuracy, magnitude error, rank ordering, and trend slope metrics. ORDC for 2023+ years, demand-quantile for 2020–2022.
- **Data**: EIA 860 plant fleet, EIA demand profiles, eGRID emission rates, fossil fleet data, EIA-930 interchange profiles
- The tool runs entirely locally — no data is sent to external servers
- All input fields include info-hover tooltips with parameter explanations
