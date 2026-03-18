# Implementation Prompts for Market Simulator Improvements

**Source**: Recommendations from `Third_Party_Expert_Review.md`
**Purpose**: Copy-paste ready prompts for future Claude Code sessions. Each prompt is self-contained with full context so the session can execute without needing to re-read the review document.

---

## Prompt 1: Simplified Zonal Decomposition

> **Task**: Add simplified zonal transmission modeling to the market simulator's LMP engine and dispatch system.
>
> **Context**: The market simulator in `market-simulator/` currently models each ISO as a single bus (copper-plate) — all generation can reach all load with no congestion. This is the tool's #1 structural limitation. Plant-level results (Tier 2 mode using EIA 860 data) promise per-generator economics, but without zonal prices, plants in congestion-constrained pockets get the same LMP as plants at transmission hubs. Real zonal price spreads are $10–35/MWh in PJM, $8–15/MWh in MISO, $5–25/MWh in ERCOT.
>
> **What to build**: A pipe-and-bubble zonal decomposition with 3–5 zones per ISO and inter-zonal transfer limits. This is the same approach GenX uses — simpler than full nodal (PLEXOS) but captures 60–80% of congestion effects.
>
> **Proposed zone definitions**:
> - **PJM** (5 zones): Western (AEP/APS), AEP-East, MAAC (Mid-Atlantic), EMAAC (Eastern Mid-Atlantic), SWMAAC (Baltimore/DC)
> - **MISO** (3 zones): North (MN/WI/IA/ND/SD), Central (IL/IN/MI), South (LA/MS/AR/TX)
> - **ERCOT** (4 zones): West (wind corridor), North (Dallas), South (San Antonio/Austin), Houston/Coast
> - **NYISO** (3 zones): Upstate (Zones A-F), NYC (Zone J), Long Island (Zone K)
> - **NEISO** (2 zones): Northern (ME/NH/VT), Southern (MA/CT/RI)
> - **CAISO** (2 zones): NP15 (Northern), SP15 (Southern)
> - **SPP** (2 zones): North (KS/NE), South (OK/TX panhandle)
>
> **Data sources for transfer limits**: ISO planning reports publish inter-zonal transfer capabilities. PJM RTEP, MISO MTEP, ERCOT CDR. Also use published SOM congestion data to calibrate zonal price differentials.
>
> **Implementation approach**:
> 1. Add a `ZONE_CONFIG` dict in `pipeline_config.py` defining zones per ISO with: zone names, transfer limits (MW) between adjacent zones, and demand/supply share per zone.
> 2. Add zone-level demand profiles — split EIA-930 ISO-level demand into zones using subregional data or published load zone shares (PJM publishes load by transmission zone).
> 3. Add zone-level generation profiles — assign existing plants to zones by latitude/longitude (EIA 860 has coordinates) or by balancing authority mapping.
> 4. Modify `compute_hourly_lmp_vectorized()` in `lmp_engine.py` to clear a zonal market: for each hour, solve a simple LP that minimizes total generation cost subject to zone-level demand balance and inter-zonal transfer limits. Use `scipy.optimize.linprog`. The marginal unit per zone sets the zonal LMP.
> 5. Modify `compute_plant_level_economics()` in `market_simulation.py` to use zonal LMP (based on plant location) instead of system-average LMP for revenue calculations.
> 6. Add a fallback: if zone data is unavailable for an ISO, fall back to current copper-plate behavior. The two-tier approach should extend to transmission too.
> 7. Update the results page (`frontend/results.html`) to show zonal LMP differences — at minimum a zone-level price spread indicator.
>
> **Key files to modify**:
> - `scripts/pipeline_config.py` — zone config constants
> - `scripts/lmp_engine.py` — `build_merit_order_stack()` (line 281), `compute_hourly_lmp_vectorized()` (line 1009)
> - `scripts/market_simulation.py` — `compute_plant_level_economics()` (line 596), `compute_generator_economics()` (line 479)
> - `scripts/fleet_model.py` — plant-to-zone assignment
> - `frontend/js/results.js` — zonal price display
>
> **Performance constraint**: The current sweep mode (270 scenarios × 7 ISOs) completes in ~30 minutes. Zonal LP adds one `linprog` call per hour × 8,760 hours per scenario-ISO, but `linprog` for a 3–5 zone problem is <1ms. Target: <2x runtime increase.
>
> **Validation**: Compare zonal LMP spreads against published SOM congestion data. PJM: Western vs EMAAC spread should be ~$8–15/MWh at medium fuel prices. MISO: North-South spread should be ~$5–10/MWh. ERCOT: West-Rest spread should be ~$5–15/MWh with high wind penetration.
>
> **Read these files first**: `scripts/lmp_engine.py`, `scripts/market_simulation.py`, `scripts/pipeline_config.py`, `scripts/fleet_model.py`, `docs/Third_Party_Expert_Review.md` (Section 3.1 and 7.1).

---

## Prompt 2: LP Storage Co-Dispatch

> **Task**: Replace the greedy sequential storage dispatch in `dispatch_utils.py` with LP co-optimized dispatch using `scipy.optimize.linprog`.
>
> **Context**: The market simulator dispatches storage in a fixed priority order: Battery 4hr → Battery 8hr → LDES 100hr → Green H₂ 1000hr. Each type operates on the residuals left by the previous one (`dispatch_utils.py`, lines 638–750). This greedy approach underestimates total storage value by 8–15% vs LP co-dispatch (per NREL storage co-optimization benchmarks). The fixed order also ignores relative economics — if LDES is cheaper than battery in a given scenario, it should dispatch first. Real batteries arbitrage LMP spreads; the current model only dispatches against surplus/deficit.
>
> **What to build**: A single LP co-dispatch that simultaneously optimizes all storage types for minimum residual demand (or maximum arbitrage revenue).
>
> **Implementation approach**:
> 1. Keep the existing greedy dispatch functions (`_battery_loop`, `_ldes_loop`) as a fallback and for comparison/validation.
> 2. Add a new function `co_dispatch_storage_lp()` in `dispatch_utils.py` that takes:
>    - `residual_surplus[H]` and `residual_gap[H]` arrays (from clean generation minus demand)
>    - Storage parameters for each type: capacity (MWh), power rating (MW), round-trip efficiency, dispatch window
>    - Optionally: `hourly_lmp[H]` for price-responsive dispatch (maximize arbitrage revenue instead of just filling gaps)
> 3. **LP formulation** (per dispatch window — daily for battery, weekly for LDES, monthly for H₂):
>    - **Decision variables**: `charge[type, hour]`, `discharge[type, hour]` for each storage type
>    - **Objective**: Minimize `sum(residual_gap[h] - sum_types(discharge[type,h]))` over all hours. Or if LMP available: Maximize `sum(lmp[h] * discharge[type,h] - lmp[h] * charge[type,h])` (arbitrage revenue).
>    - **Constraints per type**:
>      - `0 <= charge[type,h] <= power_rating[type]`
>      - `0 <= discharge[type,h] <= power_rating[type]`
>      - `0 <= soc[type,h] <= capacity[type]` (state of charge)
>      - `soc[type,h] = soc[type,h-1] + charge[type,h]*rte[type] - discharge[type,h]` (energy balance)
>      - `charge[type,h] <= residual_surplus[h] - sum_other_types(charge[other,h])` (can't charge more than available surplus across all types)
>      - `discharge[type,h] <= residual_gap[h] - sum_other_types(discharge[other,h])` (can't discharge more than needed)
>    - For tractability, solve in rolling windows: 24hr for battery-dominated mixes, 168hr (weekly) when LDES is present, 720hr (monthly) when H₂ is present.
> 4. **Integration**: Replace the sequential dispatch block in `reconstruct_hourly_dispatch()` (line 638+) with a call to `co_dispatch_storage_lp()`. Keep the same output format (per-resource dispatch profiles, matched/surplus arrays).
> 5. Add a config flag `STORAGE_DISPATCH_MODE = 'lp'  # or 'greedy'` in `pipeline_config.py` to allow switching between modes.
>
> **Key files to modify**:
> - `scripts/dispatch_utils.py` — add `co_dispatch_storage_lp()`, modify `reconstruct_hourly_dispatch()` (line 638)
> - `scripts/pipeline_config.py` — add `STORAGE_DISPATCH_MODE` config
>
> **Performance constraint**: `linprog` for a 4-type × 168-hour weekly window is ~500 decision variables + ~2000 constraints — solves in <50ms. With 52 weekly windows per year, that's ~2.6 seconds per ISO. Target: <5x runtime increase for storage dispatch portion (which is currently <1 second with Numba greedy).
>
> **Validation**: Compare LP co-dispatch total matched energy vs greedy sequential for the same inputs. LP should match 8–15% more demand (reduce residual gap by that amount). Run both modes on CAISO at 80% clean (high storage utilization) and verify the LP produces a more efficient charge/discharge schedule.
>
> **Read these files first**: `scripts/dispatch_utils.py` (lines 264–750 for existing storage dispatch), `scripts/pipeline_config.py` (storage parameters around lines 209–250), `docs/Third_Party_Expert_Review.md` (Section 3.2).

---

## Prompt 3: Exogenous Inter-Regional Flows

> **Task**: Add hourly net import/export profiles per ISO to correct the market simulator's resource adequacy and LMP calculations.
>
> **Context**: The market simulator models each ISO in complete isolation — no cross-border power flows. In reality, inter-regional interchange is substantial: MISO-PJM flows are 5–15 GW at any given hour, CAISO imports 5–8 GW from the Pacific NW, NEISO imports 2–4 GW from Hydro-Québec. Ignoring these flows overstates each ISO's self-sufficiency requirements and distorts resource adequacy calculations. NEISO's 16 GW fossil fleet looks oversized until you account for 4 GW of Quebec hydro imports.
>
> **What to build**: Exogenous hourly net import/export profiles per ISO, applied as a demand adjustment. This is NOT a full inter-regional trade model — just fixed import/export time series derived from historical data.
>
> **Data source**: EIA-930 Hourly Grid Monitor publishes hourly interchange data by balancing authority. Aggregate BA-level interchange to ISO level using the existing BA-to-ISO mapping in `fleet_model.py`. Use 2024 data as the base year. Net interchange = (imports - exports) per hour. Positive = net imports (reduces residual demand). Negative = net exports (increases residual demand).
>
> **Implementation approach**:
> 1. **Data preparation script** (`scripts/step0_fetch_interchange.py` or add to existing data pipeline):
>    - Download EIA-930 interchange data for 2024 (or load from `data/profiles/`)
>    - Aggregate by ISO using BA-to-ISO mapping
>    - Save as `data/profiles/eia_interchange_profiles.json` with format: `{ISO: [8760 hourly net_import_mw values]}`
>    - Normalize to demand units (net_import_mwh / annual_demand_mwh) for consistency with existing profile format
> 2. **Load interchange profiles** in `dispatch_utils.py`:
>    - Add `load_interchange_profile(iso)` function that returns 8760-hour normalized net import array
>    - Default: if no interchange data available, return zeros (current behavior)
> 3. **Apply to dispatch** in `reconstruct_hourly_dispatch()` (line 638):
>    - After computing clean generation and before computing residual demand:
>    - `residual_demand[h] = demand[h] - clean_supply[h] - net_imports[h]`
>    - Net imports reduce the fossil generation needed; net exports increase it
> 4. **Apply to LMP** in `compute_hourly_lmp_vectorized()` (line 1009):
>    - Imports act as additional supply in the merit-order stack (at the import price, or at $0 if treated as must-take)
>    - Simplest approach: reduce demand by net imports before dispatching the fossil stack
> 5. **Apply to resource adequacy** in `build_merit_order_stack()` (line 281):
>    - Reduce peak residual demand by the firm import contribution (e.g., NEISO can count 2 GW of HQ imports toward RA)
>    - This affects fossil fleet sizing: `ra_floor_mw = (peak_demand - clean_peak - firm_import) / gaf`
> 6. **Trajectory mode scaling**: In trajectory mode, scale interchange by demand growth (imports grow proportionally, or cap at transmission interface limits). Add `IMPORT_CAP_MW` per ISO for the maximum firm import capability.
> 7. **Setup page**: Add an "Inter-Regional Imports" toggle (On/Off or None/Historical) on the setup page. Default: On (uses historical profiles). Off: current copper-plate behavior.
>
> **Key constants to add** (`pipeline_config.py`):
> ```python
> FIRM_IMPORT_MW = {
>     'CAISO': 8000,   # Path 66 + PDCI from PNW
>     'ERCOT': 1200,   # DC ties to SPP/Mexico (limited)
>     'PJM': 5000,     # MISO/NYISO interchange
>     'NYISO': 4000,   # PJM/NEISO/HQ imports
>     'NEISO': 3500,   # HQ Phase I/II + NB Power + NYISO
>     'MISO': 4000,    # PJM/SPP interchange
>     'SPP': 3000,     # MISO/ERCOT interchange
> }
> ```
>
> **Key files to modify**:
> - `scripts/pipeline_config.py` — import cap constants
> - `scripts/dispatch_utils.py` — `load_interchange_profile()`, modify `reconstruct_hourly_dispatch()` (line 638)
> - `scripts/lmp_engine.py` — `build_merit_order_stack()` (line 281) for RA adjustment, `compute_hourly_lmp_vectorized()` (line 1009) for demand adjustment
> - `scripts/market_simulation.py` — thread interchange through `compute_lmp_at_threshold()` (line 360)
> - `frontend/setup.html` / `frontend/js/setup.js` — toggle for imports on/off
>
> **Validation**: Compare fossil fleet sizing with and without imports. NEISO should drop from ~16 GW required fossil to ~12 GW with 3.5 GW firm imports. CAISO should show reduced evening ramp severity with PNW imports. Average LMP should decrease by $1–4/MWh in import-heavy ISOs (NEISO, NYISO, CAISO).
>
> **Read these files first**: `scripts/dispatch_utils.py` (line 638 for dispatch reconstruction), `scripts/lmp_engine.py` (lines 281–415 for merit-order stack and RA sizing), `scripts/pipeline_config.py`, `docs/Third_Party_Expert_Review.md` (Section 3.4).

---

## Prompt 4: Demand Response Integration

> **Task**: Add price-elastic demand response to the market simulator's dispatch and LMP calculations.
>
> **Context**: The market simulator currently treats load as nearly perfectly inelastic — there's a logarithmic dampening factor above $200–300/MWh, but real markets have substantial demand-side flexibility: PJM has ~10 GW of registered DR (~6% of peak), ERCOT ~5 GW, MISO ~8 GW of Load Modifying Resources. Missing DR overstates scarcity pricing and understates the viability of high-clean-penetration scenarios.
>
> **What to build**: A simple price-elastic demand curtailment model. When LMP exceeds an ISO-specific trigger price, a portion of load sheds until the price drops below the trigger.
>
> **Implementation approach**:
> 1. **Add DR parameters** to `pipeline_config.py`:
>    ```python
>    DEMAND_RESPONSE = {
>        'CAISO':  {'max_dr_gw': 4.0,  'trigger_price': 150, 'participation': 0.70},
>        'ERCOT':  {'max_dr_gw': 5.0,  'trigger_price': 200, 'participation': 0.60},
>        'PJM':    {'max_dr_gw': 10.0, 'trigger_price': 100, 'participation': 0.75},
>        'NYISO':  {'max_dr_gw': 1.5,  'trigger_price': 150, 'participation': 0.70},
>        'NEISO':  {'max_dr_gw': 1.0,  'trigger_price': 150, 'participation': 0.65},
>        'MISO':   {'max_dr_gw': 8.0,  'trigger_price': 120, 'participation': 0.65},
>        'SPP':    {'max_dr_gw': 2.0,  'trigger_price': 150, 'participation': 0.60},
>    }
>    # max_dr_gw: Maximum demand response capacity (GW) - from FERC Form 714 / ISO DR registrations
>    # trigger_price: LMP threshold above which DR activates ($/MWh)
>    # participation: Fraction of registered DR that actually responds (historical performance)
>    ```
> 2. **Modify LMP computation** in `lmp_engine.py`, inside `compute_hourly_lmp_vectorized()` (line 1009):
>    - After computing initial hourly LMP from merit-order dispatch:
>    - For each hour where `lmp[h] > trigger_price`:
>      - Compute DR activation: `dr_mw = min(max_dr_gw * 1000 * participation, demand_mw[h] * 0.15)` (cap at 15% of hourly demand)
>      - Reduce demand by `dr_mw` and recompute LMP with the reduced demand
>      - The reduced demand shifts the marginal unit down the stack, lowering LMP
>      - If new LMP is still above trigger, iterate (converges in 1–2 iterations typically)
>    - Record `dr_curtailed_mwh[h]` for output
> 3. **Integrate with dispatch** in `dispatch_utils.py`:
>    - DR-adjusted demand should flow through to residual demand calculation
>    - After DR: `effective_demand[h] = demand[h] - dr_mw[h]`
>    - This reduces fossil dispatch requirements and changes the fossil fleet's capacity factor
> 4. **Track DR metrics** in output:
>    - Total DR curtailment (GWh/year)
>    - Peak DR activation (GW)
>    - Hours of DR activation
>    - Average price during DR events
> 5. **Setup page toggle**: Add "Demand Response" option — Off / Low / Medium / High
>    - Off: current behavior (no DR)
>    - Low: 50% of registered DR, higher trigger prices
>    - Medium: 70% participation (default)
>    - High: 90% participation, lower trigger prices
> 6. **Trajectory mode**: In trajectory mode, scale DR capacity with demand growth and add a DR adoption curve (DR participation increases as smart grid infrastructure expands).
>
> **Key files to modify**:
> - `scripts/pipeline_config.py` — DR parameters
> - `scripts/lmp_engine.py` — `compute_hourly_lmp_vectorized()` (line 1009) for DR-adjusted LMP
> - `scripts/dispatch_utils.py` — `reconstruct_hourly_dispatch()` (line 638) for DR-adjusted demand
> - `scripts/market_simulation.py` — thread DR through simulation, add DR metrics to output
> - `frontend/setup.html` / `frontend/js/setup.js` — DR toggle
> - `frontend/js/results.js` — display DR metrics
>
> **Validation**: At medium fuel prices in PJM, DR should reduce P90 LMP by $10–25/MWh and reduce average LMP by $1–3/MWh. Scarcity hours (>$200/MWh) should decrease by 30–50%. Total DR activation should be 50–200 hours/year (consistent with PJM DR event history).
>
> **Read these files first**: `scripts/lmp_engine.py` (lines 1009–1260 for hourly LMP computation), `scripts/pipeline_config.py`, `docs/Third_Party_Expert_Review.md` (Section 3.5).

---

## Prompt 5: Trajectory Mode Backtesting

> **Task**: Build a backtesting framework that runs the market simulator from 2020 starting conditions and compares predicted 2020–2024 outcomes against observed data.
>
> **Context**: The market simulator has forward-looking validation (LMP calibrated against 2024 SOM data in `docs/LMP_Validation_Results.md`) but no backward-looking validation. The most convincing test of the trajectory mode is to show it would have correctly predicted recent trends: coal retirements, LMP evolution, clean energy deployment, emission reductions. This is the single highest-impact validation exercise for building credibility with skeptical users.
>
> **What to build**: A backtesting script that:
> 1. Initializes the model at 2020 grid conditions (generation mix, fossil fleet, demand, fuel prices)
> 2. Runs trajectory mode from 2020 to 2024 with actual historical fuel prices, carbon prices, and policy settings
> 3. Compares predicted vs actual outcomes across 7 ISOs
>
> **Historical data needed** (2020–2024):
> - **Generation mix by ISO**: EIA-860M monthly generation by fuel type (solar, wind, nuclear, gas, coal, hydro) — publicly available
> - **Fuel prices**: EIA natural gas spot prices (Henry Hub), coal prices (Central Appalachian), oil prices — publicly available monthly
> - **Carbon prices**: RGGI auction clearing prices (quarterly), CA cap-and-trade prices — publicly available
> - **Demand**: EIA-930 annual demand by ISO — already in the tool's data directory
> - **LMP**: Annual average wholesale prices from ISO SOM reports — documented in `LMP_Validation_Results.md`
> - **Coal retirements**: EIA-860 retirement notices — publicly available
> - **Clean capacity additions**: EIA-860M monthly capacity additions by fuel type — publicly available
>
> **Implementation approach**:
> 1. Create `scripts/backtest_trajectory.py`:
>    - Define 2020 starting conditions per ISO: baseline_clean_pct, fossil_fleet_composition, demand_twh, fuel_prices
>    - Use actual 2020–2024 fuel price trajectories (not L/M/H presets)
>    - Run trajectory mode with annual steps (2020, 2021, 2022, 2023, 2024)
>    - Collect predicted: clean_pct, avg_lmp, total_co2, coal_capacity_gw, new_clean_gw_deployed per year per ISO
> 2. Create `data/backtest/historical_actuals.json` with observed values for the same metrics
> 3. Compute validation metrics:
>    - **Direction accuracy**: Did the model predict the correct direction of change for each metric? (e.g., coal declining, clean increasing)
>    - **Magnitude accuracy**: How close are predicted values to actuals? Express as % error and absolute error
>    - **Rank ordering**: Does the model correctly rank ISOs by clean deployment rate, LMP level, coal retirement speed?
>    - **Trend accuracy**: Does the model capture the 2020–2024 trend slope within ±25%?
> 4. Generate a validation report (`docs/Backtest_Validation_Report.md`) with tables and commentary
> 5. Key validation targets:
>    - Average LMP: ±$5/MWh per ISO per year
>    - Clean energy %: ±3 percentage points per ISO per year
>    - Coal retirement: correct rank ordering of ISOs (ERCOT/PJM should show most retirement)
>    - Direction of LMP change (2020→2024): should capture the 2021–2022 gas price spike and subsequent decline
>
> **Challenging aspects**:
> - 2020–2021 was unusual (COVID demand collapse, Winter Storm Uri). The model should be tested against this — if it can't handle 2021 ERCOT, that's a valid finding to document.
> - 2022 gas price spike ($6–9/MMBtu) is a good stress test for the LMP engine.
> - IRA passage (Aug 2022) changed clean energy economics mid-trajectory. The backtest should test with and without IRA provisions to see if the inflection point is captured.
>
> **Key files to create**:
> - `scripts/backtest_trajectory.py` — main backtesting script
> - `data/backtest/historical_actuals.json` — observed data
> - `docs/Backtest_Validation_Report.md` — results and analysis
>
> **Key files to read**: `scripts/market_simulation.py` (trajectory mode logic, `run_market_simulation()` function), `scripts/lmp_engine.py`, `docs/LMP_Validation_Results.md`, `docs/Third_Party_Expert_Review.md` (Section 7.5).

---

## Prompt 6: Confidence Visualization for Trajectory Results

> **Task**: Add visual confidence indicators to the market simulator's trajectory mode results that show where model outputs transition from "calibrated" to "extrapolated."
>
> **Context**: The market simulator's trajectory mode projects market outcomes from 2025 to 2060. The LMP engine is calibrated against 2024 SOM data, so 2025–2030 outputs are well-grounded. Beyond 2035, the demand-quantile pricing relationships, capacity market degradation curves, and Wright's Law learning curves are increasingly extrapolating outside their calibration domain. Users currently see no visual distinction between a high-confidence 2028 result and a speculative 2050 result.
>
> **What to build**: A confidence band system that communicates model reliability across the projection horizon.
>
> **Implementation approach**:
> 1. **Define confidence zones** (add to `pipeline_config.py` or `market_simulation.py`):
>    ```python
>    CONFIDENCE_ZONES = {
>        'high':     {'start': 2025, 'end': 2030, 'color': '#22C55E', 'label': 'Calibrated'},
>        'moderate': {'start': 2030, 'end': 2040, 'color': '#F59E0B', 'label': 'Moderate Extrapolation'},
>        'low':      {'start': 2040, 'end': 2060, 'color': '#EF4444', 'label': 'High Uncertainty'},
>    }
>    ```
> 2. **Add confidence metadata to trajectory results** in `market_simulation.py`:
>    - Each `YearResult` in the trajectory output should include a `confidence` field: `'high'`, `'moderate'`, or `'low'`
>    - Include in the API response (`backend/models.py` — add `confidence: str` to `YearResult`)
> 3. **Visualization — background shading on time-series charts** (`frontend/js/results.js`):
>    - On all trajectory charts (LMP, supply mix, emissions, clean %), add vertical background bands:
>      - Green (2025–2030): high confidence
>      - Yellow (2030–2040): moderate confidence
>      - Red (2040–2060): low confidence
>    - Use Chart.js annotation plugin or custom background drawing
>    - Add a legend entry explaining the confidence zones
> 4. **Visualization — confidence badge on KPI cards**:
>    - When displaying a single-year KPI (e.g., "2045 Clean %: 72%"), add a colored badge next to it indicating confidence level
>    - CSS classes: `.confidence-high` (green), `.confidence-moderate` (amber), `.confidence-low` (red)
> 5. **Tooltip text** explaining why confidence degrades:
>    - High: "Based on calibrated 2024 market data and near-term policy environment"
>    - Moderate: "Technology costs and market structure may diverge from calibration assumptions"
>    - Low: "Multiple compounding uncertainties — treat as scenario exploration, not forecast"
> 6. **Widening uncertainty bands on trajectory charts**:
>    - If sweep mode results are available, show P10/P90 bands that widen over time
>    - If single trajectory, add a synthetic ±X% band that grows with projection distance:
>      - 2025–2030: ±5% on LMP, ±3pp on clean%
>      - 2030–2040: ±15% on LMP, ±8pp on clean%
>      - 2040–2060: ±30% on LMP, ±15pp on clean%
>
> **Key files to modify**:
> - `scripts/pipeline_config.py` or `scripts/market_simulation.py` — confidence zone definitions
> - `backend/models.py` — add confidence field to YearResult
> - `backend/main.py` — include confidence in API response
> - `frontend/js/results.js` — chart background bands, confidence badges
> - `frontend/styles/results.css` — confidence badge styling
>
> **Read these files first**: `frontend/js/results.js` (chart rendering), `backend/models.py` (response schema), `scripts/market_simulation.py` (trajectory output construction), `docs/Third_Party_Expert_Review.md` (Section 3.3 and 7.6).

---

## Usage Notes

- Each prompt is designed to be self-contained — paste into a new Claude Code session with access to the `market-simulator/` directory
- Prompts are ordered by priority (matching the review's recommendations)
- Prompts 2, 3, and 4 are relatively independent and could be worked in parallel sessions
- Prompt 1 (zonal) is the most complex and has the highest impact — consider tackling it after 2/3/4 are done
- Prompt 5 (backtesting) can be done at any time but is most valuable after the other improvements are in place
- Prompt 6 (confidence viz) is lowest effort and can be done independently at any time
