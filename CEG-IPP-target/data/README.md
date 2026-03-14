# Data Dictionary

## ipp_fleet_config.csv
One row per plant (~195 rows across 7 companies).

| Column | Type | Description |
|--------|------|-------------|
| company_id | string | Company identifier (vistra, constellation, nrg, nextera, talen, pseg, aes) |
| company_name | string | Full company name |
| short_name | string | Display name |
| co2_2023_mt | float | Company-level CO₂ emissions 2023 (Mt) |
| co2_2024_mt | float | Company-level CO₂ emissions 2024 (Mt) |
| intensity_kg | float | Company-level emissions intensity (kg CO₂/MWh) |
| company_gen_twh | float | Company-level total generation (TWh) |
| company_cap_gw | float | Company-level total capacity (GW) |
| target | string | Company climate target statement |
| plant_name | string | Individual plant name |
| iso | string | ISO region (CAISO, ERCOT, PJM, NYISO, NEISO, MISO, SPP) |
| fuel | string | Fuel type (nuclear, coal, gas_ccgt, gas_peaker, solar, wind, battery, hydro, geothermal, oil) |
| cap_mw | float | Plant nameplate capacity (MW) |
| gen_twh | float | Plant annual generation (TWh) |
| co2_mt | float | Plant annual CO₂ emissions (Mt) |
| heat_rate | float | Plant heat rate (BTU/kWh), null for clean resources |
| retire_by | float | Planned retirement year, null if none |

## ipp_fleet_companies.csv
One row per company (7 rows). Company-level summary with per-fuel MW breakdowns.

| Column | Type | Description |
|--------|------|-------------|
| company_id | string | Company identifier |
| company_name | string | Full name |
| short_name | string | Display name |
| co2_2023_mt | float | CO₂ 2023 (Mt) |
| co2_2024_mt | float | CO₂ 2024 (Mt) |
| intensity_kg | float | Emissions intensity (kg CO₂/MWh) |
| gen_twh | float | Total generation (TWh) |
| cap_gw | float | Total capacity (GW) |
| target | string | Climate target |
| nuclear_mw | float | Nuclear capacity (MW) |
| coal_mw | float | Coal capacity (MW) |
| gas_ccgt_mw | float | Gas CCGT capacity (MW) |
| gas_peaker_mw | float | Gas peaker capacity (MW) |
| solar_mw | float | Solar capacity (MW) |
| wind_mw | float | Wind capacity (MW) |
| battery_mw | float | Battery capacity (MW) |
| hydro_mw | float | Hydro capacity (MW) |
| geothermal_mw | float | Geothermal capacity (MW) |
| oil_mw | float | Oil capacity (MW) |
| iso_count | int | Number of ISO regions |
| primary_iso | string | ISO with largest capacity |

## ipp_sweep_results.csv
Scenario sweep results from the SMARTargets IPP module.

| Column | Type | Description |
|--------|------|-------------|
| scenario | string | Scenario identifier |
| sweep_type | string | Sweep type (reference, power_nz, economy_nz) |
| conditions | string | Condition code |
| demand_growth | string | Demand growth level (Low, Medium, High) |
| price_sens | string | Price sensitivity level |
| ppa_level | string | PPA coverage level (Low, Medium, High) |
| gas_friction | float | Gas transition friction (0.3, 0.7, 1.0) |
| year | int | Projection year |
| fleet_emissions_mt | float | Fleet CO₂ emissions (Mt) |
| fleet_revenue_m | float | Fleet revenue ($M) |
| fleet_cost_m | float | Fleet costs ($M) |
| fleet_profit_m | float | Fleet profit ($M) |
| operating_mw | float | Operating capacity (MW) |
| stranded_mw | float | Stranded capacity (MW) |
| retired_mw | float | Retired capacity (MW) |
| company | string | Company identifier |

## pipeline_constants.csv
Key-value pairs for model constants.

| Column | Type | Description |
|--------|------|-------------|
| key | string | Constant name (e.g., wholesale_price_CAISO) |
| value | float | Constant value |
| unit | string | Unit of measurement |
| source | string | Source module/reference |
