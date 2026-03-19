# Running the Parametric Sweep Locally (with Full Plant Data)

This guide replicates what the GitHub Actions workflow does, but on a local machine with full access to EIA 860, EIA 923, and EPA CAMPD data.

## Prerequisites

### 1. Python Environment

```bash
python3 --version  # Requires 3.9+
pip install numpy numba pandas pyarrow scipy fastapi uvicorn openpyxl jinja2 pydantic
```

Verify Numba:
```bash
python3 -c "from numba import njit; print('Numba OK')"
```

### 2. Clone the Repo

```bash
git clone https://github.com/jessicacohen554-cyber/hourly-cfe-optimizer.git
cd hourly-cfe-optimizer
```

---

## Step 1: Populate Plant Data

The simulator expects plant-level data in `market-simulator/data/`. Without it, the LMP engine falls back to stylized fleet-average defaults.

### Directory Structure

```
market-simulator/data/
├── eia-860/
│   ├── TX/
│   │   └── eia860_TX.json
│   ├── CA/
│   │   └── eia860_CA.json
│   ├── PA/
│   │   └── eia860_PA.json
│   └── {STATE}/
│       └── eia860_{STATE}.json
│
├── eia-923/
│   ├── TX/
│   │   ├── eia923_TX_2023-01.json
│   │   ├── eia923_TX_2023-02.json
│   │   └── ...  (monthly files through 2025-12)
│   └── {STATE}/
│       └── eia923_{STATE}_{YYYY-MM}.json
│
├── epa-campd/
│   ├── TX/
│   │   ├── TX_2023-01.json
│   │   ├── TX_2023-02.json
│   │   └── ...
│   └── {STATE}/
│       └── {STATE}_{YYYY-MM}.json
│
├── eia-natgas-prices/          (optional — enhances fuel cost calibration)
│   ├── delivered-to-power/
│   │   └── natgas_delivered_to_power_2023_2025.json
│   └── spot-futures/
│       └── natgas_spot_futures_{YYYY}.json
```

### Key States by ISO

You don't need every state — prioritize these:

| ISO | Must-Have States | Notes |
|-----|-----------------|-------|
| ERCOT | TX | Single-state ISO |
| CAISO | CA | Single-state ISO |
| PJM | PA, OH, VA, WV, NJ, DE, MD, IL, IN | Largest footprint |
| NYISO | NY | Single-state ISO |
| NEISO | MA, ME, NH, CT, RI, VT | 6 New England states |
| MISO | IL, IN, MI, WI, MN, IA, MO, AR, MS, LA | 10+ states |
| SPP | KS, OK, AR, NE, NM | South-central |

Plants are auto-assigned to ISOs via `balancing_authority_code` in EIA 860 data.

### EIA-860 JSON Format

Each record needs:
```json
{
  "plant_id": 12345,
  "plant_name": "Example Plant",
  "state": "TX",
  "capacity_mw": 500.0,
  "fuel_type": "NG",
  "prime_mover": "CA",
  "status": "OP",
  "balancing_authority_code": "ERCO",
  "heat_rate": 7200.0,
  "online_year": 2005
}
```

Fuel types: `NG` (gas), `BIT`/`SUB`/`LIG`/`ANT` (coal), `DFO`/`RFO` (oil)
Prime movers: `CA`/`CS`/`CT`/`CC` → gas_ccgt; `GT`/`IC`/`OT` → gas_ct; `ST` → resolved by fuel

### EIA-923 JSON Format

Monthly generation + fuel consumption per plant:
```json
{
  "plant_id": 12345,
  "plant_name": "Example Plant",
  "fuel_type": "NG",
  "net_generation_mwh": 350000.0,
  "fuel_mmbtu": 2450000.0,
  "year": 2024,
  "month": 6
}
```

### EPA CAMPD JSON Format

Hourly emissions monitoring per plant-unit:
```json
{
  "orispl_code": 12345,
  "unit_id": "1",
  "op_date": "2024-06-15",
  "op_hour": 13,
  "co2_mass_tons": 450.0,
  "nox_mass_lbs": 18.0,
  "so2_mass_lbs": 2.5,
  "gross_load_mw": 480.0,
  "heat_input": 3350.0
}
```

### Heat Rate Resolution Order

The model uses the most accurate heat rate available:
1. **EIA-923 revealed**: `fuel_mmbtu / net_generation_mwh` (preferred)
2. **EPA CAMPD measured**: `heat_input / gross_load` (hourly)
3. **EIA-860 design**: `heat_rate` field (least accurate)
4. **Fallback defaults**: coal 10.0, gas_ccgt 7.0, gas_ct 10.5, oil_ct 10.5 MMBtu/MWh

---

## Step 2: Generate Synthetic Profiles (if needed)

The base profiles ship with the repo, but regenerate if missing:

```bash
cd market-simulator

export PYTHONPATH="$(pwd):$(pwd)/backend:$(pwd)/scripts"

# Generate demand + generation profiles
python3 scripts/generate_synthetic_profiles.py

# Generate plant-specific heat rates (uses EIA-923 + CAMPD if available)
python3 scripts/generate_plant_heat_rates.py
```

This creates:
- `data/profiles/eia_demand_profiles.json`
- `data/profiles/eia_gen_profiles.json`
- `data/profiles/eia_fossil_mix.json`
- `data/profiles/eia_interchange_profiles.json`
- `data/plant_heat_rates.json`

---

## Step 3: Copy Pipeline Parquets (if available)

If you have results from the hourly-cfe-optimizer pipeline (Steps 1-2):

```bash
# From the repo root:
cp data/step2.1-ef/step_2_1_ef_*.parquet market-simulator/data/step2.1-ef/
cp data/step2.2-cost/step_2_2a_CO_*.parquet market-simulator/data/step2.2-cost/
```

Without these, the simulator generates synthetic resource mixes (good for screening, not publication-quality).

---

## Step 4: Run the Parametric Sweep

### Set PYTHONPATH

```bash
cd market-simulator
export PYTHONPATH="$(pwd):$(pwd)/backend:$(pwd)/scripts"
```

### Option A: Full 405-Scenario Sweep (all ISOs)

This is what the GitHub Action does. Runs 3 demand × 5 price × 3 PPA × 3 gas friction × 3 queue = 405 scenarios × 7 ISOs.

```bash
python3 scripts/market_simulation.py
```

**Runtime**: ~30-90 min depending on hardware and whether Numba is installed.

**Output**: `market-simulator/data/results/market_simulation_results.json`

### Option B: Full Sweep, Specific ISOs

```bash
# Single ISO
python3 scripts/market_simulation.py --isos ERCOT

# Multiple ISOs
python3 scripts/market_simulation.py --isos CAISO ERCOT PJM
```

### Option C: Single Scenario (fastest — for testing)

```bash
python3 scripts/market_simulation.py --single
```

Runs one Medium-defaults scenario across all ISOs. ~30s.

### Option D: Single Scenario with Overrides

```bash
python3 scripts/market_simulation.py --single \
  --fuel-level High \
  --lcoe-level Low \
  --carbon-price 50 \
  --nuclear-retirement 35
```

### Option E: Weather-Year Sensitivity

```bash
# Single weather year
python3 scripts/market_simulation.py --weather-year 2023

# Multiple weather years (multiplies scenario count)
python3 scripts/market_simulation.py --weather-year "2021,2023,2025"

# All 5 weather years (5× runtime)
python3 scripts/market_simulation.py --weather-year all
```

### Option F: Custom Output Directory

```bash
python3 scripts/market_simulation.py --output-dir /path/to/my/results
```

### Option G: Snapshot Mode (single-year, no trajectory)

```bash
python3 scripts/market_simulation.py --snapshot
```

---

## Step 5: Run the Backtest (Optional)

After the sweep, validate against historical data:

```bash
python3 scripts/backtest_trajectory.py
```

---

## Step 6: Copy Results Back to Pipeline

If you want results available to the dashboard and downstream steps:

```bash
# From the market-simulator directory:
cp data/results/market_simulation_results.json ../data/market_simulation_results.json
```

---

## Sweep Parameter Space

The full sweep explores:

| Dimension | Levels | Values |
|-----------|--------|--------|
| Demand growth | 3 | Low / Medium / High |
| Price sensitivity | 5 | all_low / all_med / all_high / high_vre_low_firm / high_firm_low_vre |
| PPA premium | 3 | Low / Medium / High |
| Gas friction | 3 | Low (0.3) / Medium (0.7) / High (1.0) |
| Queue cap | 3 | Low / Medium / High |

**Total**: 3 × 5 × 3 × 3 × 3 = **405 scenarios per ISO**

Each scenario simulates a 25-year trajectory (2025→2050) with annual year steps by default. Simulation includes:
- Wright's Law learning curves (FOAK→NOAK)
- Tech-differentiated interconnection queue caps
- Merit-order fossil dispatch + synthetic LMP
- Nuclear retirement economics
- RPS/CES compliance + REC pricing
- Capacity market revenue
- PPA premium scaling

### Per-scenario outputs (per ISO, per year):
- `clean_pct` — Clean energy share (%)
- `demand_twh` — Total demand
- `emissions_mt` — CO₂ emissions (metric tons)
- `cost_per_mwh` — System cost
- `avg_lmp` — Average LMP
- Per-resource deployment (GW), generation (TWh), revenue breakdown
- Plant-level economics (if EIA 860/923/CAMPD data available)

### Aggregated outputs:
- P10/P50/P90 uncertainty bands across all scenarios
- Per-ISO, per-year percentile distributions

---

## Data Tier Summary

| Mode | Data Needed | What You Get |
|------|-------------|--------------|
| **Minimal** | Nothing extra | Synthetic profiles, fleet-average dispatch |
| **Pipeline** | + step2.1/2.2 parquets | Real optimizer resource mixes + cost scenarios |
| **Plant-level** | + EIA-860 | Real plant characteristics, vintage-adjusted dispatch |
| **Full fidelity** | + EIA-923 + EPA CAMPD | Real heat rates, per-plant economics, hourly emissions |

---

## Troubleshooting

**"No module named 'pipeline_config'"**: Set `PYTHONPATH` as shown above.

**Slow runtime (10×+ expected)**: Numba isn't installed or JIT compilation hasn't cached yet. First run is slower due to JIT warm-up.

**"No parquet files found"**: The simulator falls back to synthetic data automatically. Results are still valid for screening but won't reflect real optimizer physics.

**Memory issues**: The full sweep holds all 405 scenario results in memory. For constrained systems, run one ISO at a time with `--isos`.
