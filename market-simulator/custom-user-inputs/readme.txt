================================================================================
CUSTOM USER INPUTS — README
================================================================================

This folder contains CSV templates for overriding the model's built-in price
curves with your own data. To use a custom file:

1. Copy the template (e.g., template_lmp_hourly.csv → lmp_hourly.csv)
2. Replace the values with your data, keeping column headers unchanged
3. Save in this folder
4. On the Setup page, enable the "Use custom file" toggle for that input

The model auto-detects files by looking for the non-template filename
(without the "template_" prefix).

================================================================================
FILE FORMAT SPECIFICATIONS
================================================================================

FILE: lmp_hourly.csv
  Rows: 8760 (one per hour of the year)
  Columns: hour, CAISO, ERCOT, PJM, NYISO, NEISO, MISO, SPP
  Units: $/MWh
  Notes: hour column is 1-8760. Leave ISOs you don't need unchanged.

FILE: capacity_prices.csv
  Rows: 12 (one per month, Jan-Dec)
  Columns: month, CAISO, ERCOT, PJM, NYISO, NEISO, MISO, SPP
  Units: $/MW-day
  Notes: Energy-only markets (ERCOT, SPP) default to $0. Set non-zero
         values to model capacity payments (e.g., reliability credits).

FILE: rec_prices.csv
  Rows: 12 (one per month, Jan-Dec)
  Columns: month, CAISO, ERCOT, PJM, NYISO, NEISO, MISO, SPP
  Units: $/MWh
  Notes: REC prices by ISO. Set to 0 for ISOs without RPS.

FILE: fuel_prices_gas.csv
FILE: fuel_prices_coal.csv
FILE: fuel_prices_oil.csv
  Rows: 12 (one per month, Jan-Dec)
  Columns: month, CAISO, ERCOT, PJM, NYISO, NEISO, MISO, SPP
  Units: $/MMBtu
  Notes: Monthly fuel prices by region. The model converts monthly to
         hourly internally (flat within each month).

================================================================================
VALIDATION RULES
================================================================================

- Column headers must match exactly (case-sensitive ISO names)
- No blank cells or NaN values allowed
- Numeric values only (no currency symbols, commas, or text)
- LMP file: exactly 8760 data rows (plus header)
- Monthly files: exactly 12 data rows (plus header)
- Negative values are allowed for LMP (curtailment hours)
- Negative values are NOT allowed for fuel prices, capacity, or REC prices

If validation fails, the model will display an error message on the Setup
page and fall back to built-in defaults.

================================================================================
