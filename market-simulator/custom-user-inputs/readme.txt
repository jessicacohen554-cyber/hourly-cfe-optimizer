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

--------------------------------------------------------------------------------
TIME-SERIES FORMAT (fuel, capacity, REC files)
--------------------------------------------------------------------------------

These files support TWO formats:

  FORMAT 1 — Legacy (static, 12 rows):
    Columns: month, CAISO, ERCOT, PJM, NYISO, NEISO, MISO, SPP
    Rows: 12 (one per month, Jan-Dec)
    Behavior: Same prices applied to ALL simulation years.

  FORMAT 2 — Time-series (annual × monthly × zonal):
    Columns: year, month, zone, CAISO, ERCOT, PJM, NYISO, NEISO, MISO, SPP
    Rows: n_years × 12 × n_zones
    Behavior: Year-specific prices used in trajectory runs. The simulator
              looks up the exact year; if a year isn't in the file, the
              nearest available year is used.

    The 'zone' column is OPTIONAL:
      - Leave blank or omit entirely for system-wide (ISO-level) prices
      - Populate with zone names (e.g., "north", "south", "constrained")
        for sub-ISO granularity
      - If zones are used, each year×zone must have exactly 12 months

    Example (annual, no zones — 432 rows for 2025-2060):
      year,month,zone,CAISO,ERCOT,PJM,NYISO,NEISO,MISO,SPP
      2025,1,,3.50,3.50,3.50,4.00,4.50,3.25,3.25
      2025,2,,3.50,3.50,3.50,4.00,4.50,3.25,3.25
      ...
      2060,12,,4.20,4.00,4.10,5.00,5.50,3.80,3.80

    Example (annual + zonal):
      year,month,zone,CAISO,ERCOT,PJM,NYISO,NEISO,MISO,SPP
      2025,1,system,3.50,3.50,3.50,4.00,4.50,3.25,3.25
      2025,1,constrained,3.50,3.50,3.50,4.00,6.00,3.25,3.25
      2025,2,system,3.50,3.50,3.50,4.00,4.50,3.25,3.25
      2025,2,constrained,3.50,3.50,3.50,4.00,6.00,3.25,3.25

--------------------------------------------------------------------------------

FILE: fuel_prices_gas.csv
  Units: $/MMBtu
  Default template: Henry Hub Medium projection with regional basis
  differentials (NEISO +$1.00, NYISO +$0.50), seasonal shape, ~1.5%/yr
  annual escalation from 2025-2060.

FILE: fuel_prices_coal.csv
  Units: $/MMBtu
  Default template: Flat/declining trajectory (~-0.3%/yr) reflecting
  retirement economics, floor at 85% of 2025 price.

FILE: fuel_prices_oil.csv
  Units: $/MMBtu
  Default template: Flat trajectory with minimal escalation (~0.5%/yr).
  Peaker fuel, rarely dispatched.

FILE: capacity_prices.csv
  Units: $/MW-day
  Default template: Year-over-year growth (~2%/yr) reflecting tightening
  capacity markets. Energy-only markets (ERCOT, SPP) default to $0.

FILE: rec_prices.csv
  Units: $/MWh
  Default template: Declining trajectory (~-2%/yr, floor 30% of 2025 price)
  reflecting REC market saturation. Set to 0 for ISOs without RPS.

================================================================================
VALIDATION RULES
================================================================================

- Column headers must match exactly (case-sensitive ISO names)
- No blank cells or NaN values allowed in ISO columns
- Numeric values only (no currency symbols, commas, or text)
- Negative values are allowed for LMP (curtailment hours)
- Negative values are NOT allowed for fuel prices, capacity, or REC prices

Legacy format (no 'year' column):
- Monthly files: exactly 12 data rows (plus header)
- LMP file: exactly 8760 data rows (plus header)

Time-series format ('year' column present):
- Each year must have exactly 12 months (1-12)
- If zones are used, each year×zone combination must have 12 months
- Total rows must equal: n_years × 12 × n_zones
- The 'year' column must contain integer year values
- The 'month' column must contain values 1-12

BACKWARD COMPATIBILITY:
- Old 12-row files (without 'year' column) continue to work unchanged
- The model auto-detects the format based on whether 'year' column exists

If validation fails, the model will display an error message on the Setup
page and fall back to built-in defaults.

================================================================================
