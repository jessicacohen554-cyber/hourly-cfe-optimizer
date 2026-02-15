# Hourly CFE Cost Optimizer

An interactive dashboard and optimization engine for analyzing **hourly clean energy procurement strategies** across five major US ISO regions.

The optimizer finds the least-cost mix of clean firm (nuclear/geothermal), solar, wind, hydro, and battery storage to meet hourly demand at various Clean Energy Matching (CEM) targets (75%--100%), then visualizes the results in a scrollytelling dashboard.

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

## Project Structure

```
hourly-cfe-optimizer/
+-- optimize_overprocure.py       # Optimization engine
+-- data/                          # EIA hourly generation & demand profiles
|   +-- eia_demand_profiles.json
|   +-- eia_generation_profiles.json
|   +-- eia_hourly_*.json          # Per-ISO hourly data (2024-2025)
|   +-- egrid_emission_rates.json
|   +-- ...
+-- dashboard/
    +-- dashboard.html             # Interactive dashboard
    +-- overprocure_results.json   # Pre-computed optimization results
    +-- optimizer_methodology.html # Detailed methodology documentation
    +-- build_standalone.py        # Builds standalone from dashboard.html
```

## Running the Optimizer

To regenerate results from scratch (requires Python 3.10+ and NumPy):

```bash
pip install numpy
python optimize_overprocure.py
```

This runs a three-phase sweep (coarse -> medium -> fine) across procurement levels for each ISO, finding optimal resource mixes at each CEM threshold. Results are written to `dashboard/overprocure_results.json`.

Runtime is typically 2-5 minutes depending on hardware.

## How It Works

### Optimization Engine

For each ISO region, the engine:

1. Loads real hourly demand shapes and generation profiles from EIA data
2. Sweeps over-procurement levels from 100% to 200%+ of annual demand
3. At each level, finds the optimal mix of 4 resource types + storage that maximizes hourly matching score
4. Uses three-phase refinement (20% coarse, 5% medium, 1% fine grid) for efficiency
5. Applies regional LCOE pricing with wholesale/new-build cost tiers

### Resource Types

- **Clean Firm**: Flat baseload profile (nuclear, geothermal) -- $90/MWh new-build
- **Solar**: EIA regional hourly profile -- $54-92/MWh by region
- **Wind**: EIA regional hourly profile -- $40-81/MWh by region
- **Hydro**: EIA regional hourly profile (capped by region) -- priced at wholesale
- **Storage**: 4-hour Li-ion, 85% round-trip efficiency -- $100/MWh LCOS

### Dashboard Features

- **Compressed Day Profile**: 24-hour average pattern showing how each resource contributes
- **Resource Procurement Mix**: Donut chart of optimal resource allocation
- **Peak Hour Capacity Need**: Visual comparison of demand, clean coverage, and backup needs
- **Cost Breakdown**: Per-resource cost analysis with wholesale vs. new-build tiers
- **Scrollytelling Narrative**: Guided story exploring how optimal mixes evolve across matching targets

## Data Sources

- **EIA-930**: Hourly grid generation data (2024-2025)
- **LBNL**: Utility-Scale Solar 2024, Wind Market Report 2024 (LCOE estimates)
- **FERC/ISO**: Wholesale electricity price averages (2023-2024)
- **eGRID**: Emission rate data

## Rebuilding the Standalone

After modifying `dashboard.html`, regenerate the self-contained version:

```bash
cd dashboard
python build_standalone.py
```

This inlines Chart.js, the annotation plugin, and the JSON data into a single HTML file. If the existing standalone is present, it extracts the libraries from it; otherwise it downloads them from CDN.

## License

MIT License. See [LICENSE](LICENSE) for details.
