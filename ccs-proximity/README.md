# CCS Proximity Analysis

Spatial viability scoring for CCUS (Carbon Capture, Utilization, and Storage) retrofitting of US fossil power plants.

## What It Does

Cross-references three federal datasets to rank every US fossil plant by its suitability for carbon capture retrofit:

1. **EPA eGRID 2023** — Plant locations, CO2 emissions, fuel type, capacity
2. **NETL NATCARB** — CO2 pipeline routes and geologic saline storage formations
3. **EPA/CATF Class VI Wells** — Active and pending CO2 injection well permits

## Viability Score (0-100)

Each plant receives a composite score based on:

| Factor | Weight | Logic |
|--------|--------|-------|
| Annual CO2 emissions | 40% | Higher emissions = more impactful retrofit (log-scaled) |
| Distance to nearest CO2 pipeline | 25% | Closer = cheaper transport (capped at 200km) |
| Distance to nearest Class VI well | 20% | Closer = proven storage access (capped at 300km) |
| Overlaps geologic storage formation | 15% | Binary — plant sits atop viable geology |

## Quick Start

```bash
pip install -r requirements.txt
python ccs_proximity_analysis.py
```

All data is downloaded automatically on first run. Subsequent runs use cached files in `data/`.

## Output

| File | Description |
|------|-------------|
| `output/ccus_viability_ranking.csv` | Full ranked table — every fossil plant with all metrics |
| `output/ccus_summary_by_state.csv` | Aggregated scores by state |
| `output/ccus_summary_by_iso.csv` | Aggregated scores by ISO/RTO |
| `output/ccus_viability_map.png` | CONUS map with plants colored by viability score |

## Data Sources

- **EPA eGRID 2023**: https://www.epa.gov/egrid
- **NETL NATCARB / Carbon Storage Atlas**: https://netl.doe.gov/carbon-management/carbon-storage/atlas-data
- **NETL ArcGIS REST API**: https://arcgis.netl.doe.gov/server/rest/services
- **EPA UIC Class VI**: https://www.epa.gov/uic/current-class-vi-projects-under-review-epa
- **CATF Class VI Wells Tracker**: https://www.catf.us/classviwellsmap/

## CRS

All spatial analysis uses **EPSG:5070** (NAD83 / Conus Albers Equal Area) for accurate distance calculations across CONUS. Output coordinates are in WGS84 (EPSG:4326).

## Fallback Data

If NETL/EPA APIs are unavailable (auth required, downtime), the script falls back to curated reference datasets:
- **CO2 Pipelines**: 14 major US pipelines (operational + proposed) from PHMSA/DOE data
- **Saline Formations**: 10 major storage basins from NETL Carbon Storage Atlas (5th ed.)
- **Class VI Wells**: 22 wells from EPA permit documents and CATF tracker

These fallbacks are clearly labeled in the source and provide reasonable coverage for screening-level analysis.
