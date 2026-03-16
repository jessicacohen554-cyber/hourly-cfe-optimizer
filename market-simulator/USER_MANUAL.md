# Market Simulation Tool — User Manual

## Quick Start

### Option 1: Double-Click Launcher (Recommended)

**Mac/Linux:**
```
./run.sh
```
Or double-click `run.sh` in Finder/file manager.

**Windows:**
Double-click `run.bat`.

### Option 2: Python Script
```
python start.py
```

### Option 3: Manual
```bash
pip install -r requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
Then open http://127.0.0.1:8000 in your browser.

---

## System Requirements

- **Python 3.9+** (3.11 recommended)
- **pip** (comes with Python)
- **Modern web browser** (Chrome, Firefox, Edge, Safari)
- **No other software required** — all dependencies install automatically via pip

---

## Tool Overview

The Market Simulation Tool models electricity market economics across 7 US ISOs. Given market preconditions (fuel prices, carbon costs, clean energy LCOEs, incentives), it computes:

- Merit-order dispatch and hourly LMPs
- Generator profitability (which units retire vs. stay profitable)
- Clean energy deployment economics (what can be built profitably)
- Nuclear retirement risk under different revenue assumptions
- CCS breakeven carbon price analysis

### Three-Page Flow

1. **Guide** (`/`) — Instructions and tool overview
2. **Setup** (`/setup`) — Input parameters and run simulation
3. **Results** (`/results`) — Charts, tables, and analysis

---

## Input Parameters Reference

### Simulation Mode
| Mode | Description | Speed |
|------|-------------|-------|
| **Snapshot** | Single point-in-time simulation | 5-15 seconds |
| **Trajectory** | Multi-year projection with learning curves (2030-2050) | 30-120 seconds |
| **Sweep** | 270-scenario parametric sweep | 5-30 minutes |

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

### Clean Energy LCOEs ($/MWh)
Levelized cost for new-build resources. Lower = cheaper to deploy.

| Resource | Low | Medium | High |
|----------|-----|--------|------|
| Solar | $24.5 | $31.8 | $42.0 |
| Wind | $28.0 | $34.5 | $46.0 |
| Offshore Wind | $68.0 | $82.0 | $105.0 |
| Nuclear | $72.0 | $89.0 | $115.0 |
| CCS-CCGT | $72.0 | $86.0 | $110.0 |

### Federal Incentives
| Parameter | Default | Description |
|-----------|---------|-------------|
| PTC Wind/Solar | $26/MWh | Production Tax Credit |
| PTC New Nuclear | $26/MWh | 45Y new nuclear credit |
| 45U Max Credit | $15/MWh | Existing nuclear CfD cap |
| 45U Floor | $40/MWh | Revenue floor for existing nuclear |
| ITC | 30% | Investment Tax Credit (storage, offshore) |

### Storage Costs ($/kW-yr)
| Technology | Low | Medium | High | Computed LCOS |
|------------|-----|--------|------|---------------|
| Battery 4hr | $220 | $295 | $380 | ~$238/MWh |
| Battery 8hr | $350 | $456 | $580 | ~$224/MWh |
| LDES 100hr | $150 | $220 | $300 | ~$440/MWh |

### Demand Growth
| Level | Rate | Custom |
|-------|------|--------|
| Low | 0.5%/year | Enter any % |
| Medium | 1.5%/year | |
| High | 2.5%/year | |

### Other Parameters
| Parameter | Options | Description |
|-----------|---------|-------------|
| Transmission | None / Low / Med / High | Transmission cost adder |
| 45Q | On / Off | CCS tax credit toggle |
| PPA Availability | Low / Med / High | Clean energy offtake economics |
| Gas Friction | Low / Med / High | ESG/permitting headwinds for new gas |
| Nuclear Retirement | $10-60/MWh | Revenue floor below which nuclear retires |

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

## Results Directory

Every simulation run saves to `results/run_NNN/`:

```
results/
├── run_001/
│   ├── inputs.txt            # All user inputs (formatted text)
│   ├── narrative.txt         # Auto-generated results interpretation
│   ├── results_data.csv      # Full result data (generator economics, supply mix)
│   ├── input_parameters.csv  # Tabular copy of parameter selections
│   ├── charts/               # PNG images (after clicking "Export Charts")
│   │   ├── chartLMPTimeSeries.png
│   │   ├── chartSupplyStack.png
│   │   └── ...
│   └── custom_*.csv          # Copies of custom input files (if used)
├── run_002/
│   └── ...
```

### Exporting Charts
Click "Export Charts" on the results page to save PNG images of all charts to the run's `charts/` folder.

### Downloading Data
Click "Download CSV" to download the full results as a JSON file.

---

## Troubleshooting

### "Python not found"
Install Python 3.9+ from https://www.python.org/downloads/. Ensure it's on your system PATH.

### "Module not found" errors
Run `pip install -r requirements.txt` manually from the market-simulator directory.

### Port 8000 already in use
Another application is using port 8000. Either stop it or edit the PORT variable in `run.sh`/`run.bat`/`start.py`.

### "No simulation results" on Results page
You need to run a simulation from the Setup page first. Results are stored in the browser's session storage — they clear when you close the tab.

### Charts not rendering
If Plotly.js fails to load from the local file, ensure you have internet access (CDN fallback) or run the download step from `run.sh` first.

### Custom input file rejected
Check that your CSV has exactly the right number of rows and all 7 ISO columns. See the `readme.txt` in `custom-user-inputs/` for format details.

---

## Technical Notes

- **Backend**: FastAPI (Python) serving REST API + static HTML
- **Frontend**: Vanilla HTML/CSS/JS with Plotly.js for charts
- **Model**: Merit-order dispatch with hourly LMP computation
- **Data**: EIA demand profiles, eGRID emission rates, fossil fleet data
- The tool runs entirely locally — no data is sent to external servers
