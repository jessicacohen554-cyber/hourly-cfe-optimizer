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

## Prompt 7: VRE Cannibalization Feedback in Deployment

> **Task**: Wire time-matched energy revenue into the deployment model so solar and wind receive their actual profile-weighted LMP revenue instead of the system average.
>
> **Context**: This is the **#1 audit criticism** (AUDIT.md §2.1). The deployment model in `compute_market_deployment()` (line 1421 of `market_simulation.py`) uses `base_energy_rev = avg_lmp` (line 1463) for ALL resources — solar, wind, nuclear, CCS all get the same $/MWh energy revenue. In reality, solar revenue collapses with penetration (midday price depression) and wind gets different temporal value than baseload. The infrastructure to fix this **already exists**: `compute_energy_revenue_by_resource()` at line 1032 computes time-matched `LMP[h] × generation_profile[h]` weighted revenue per resource. It's just never called in the deployment path. There's also a sigmoid cannibalization function in `procurement_utils.py` (line 1166) used for procurement analysis but not deployment.
>
> **What to build**: Per-resource energy revenue in the deployment merit-order, with intra-deployment feedback that re-estimates revenue as each resource tranche is deployed.
>
> **Implementation approach**:
> 1. **Wire existing function into deployment path.** In `run_market_simulation()` (line ~2141), after computing `hourly_lmp` via `compute_lmp_at_threshold()`, call `compute_energy_revenue_by_resource(hourly_lmp, supply_profiles_iso, resource_pcts, demand_total_mwh)`. Pass the resulting `{resource: $/MWh}` dict into `compute_market_deployment()` as a new parameter `per_resource_energy_rev`.
> 2. **Replace static avg_lmp in deployment.** In `compute_market_deployment()`, replace line 1463 (`base_energy_rev = avg_lmp`) with a per-resource lookup inside the resource loop (line 1467+):
>    ```python
>    base_energy_rev = per_resource_energy_rev.get(res, avg_lmp)
>    ```
>    Firm resources (nuclear, CCS, geothermal) have flat profiles and will naturally get ~avg_lmp. Solar and wind will get their profile-weighted revenue, which is lower when VRE penetration is high.
> 3. **Add intra-deployment cannibalization feedback.** After deploying each resource tranche in the `for entry in sorted_economics` loop (line ~1547), apply a lightweight price depression estimate:
>    - Compute the incremental clean supply profile: `delta_supply[h] = deployed_twh * profile[h] / sum(profile)`
>    - For hours where the resource generates, estimate LMP depression from the merit-order stack slope: `lmp_delta[h] = -delta_supply_mw[h] * dLMP_dSupply` where `dLMP_dSupply` is the $/MWh-per-MW slope at each hour's operating point (derivable from the merit-order stack already computed)
>    - **Simpler alternative for v1**: Use the existing sigmoid from `procurement_utils.py:1166` applied per-resource based on that resource's penetration fraction: `depression = 0.55 * (1 / (1 + exp(-8 * (res_penetration - 0.6))))`. Solar at 30% penetration gets heavier cannibalization than wind at 15%.
>    - Update `per_resource_energy_rev` after each tranche so the next resource in the merit-order sees the updated prices
> 4. **Config flag**: Add `CANNIBALIZATION_ENABLED = True` to `pipeline_config.py` to allow toggling.
> 5. **Output enrichment**: Add `energy_rev_by_resource: Dict[str, float]` to the year result dict (and `backend/models.py` `YearResult` at line 291) so the frontend can display per-resource energy revenue vs avg_lmp.
> 6. **Frontend display**: On the results page, show a "Capture Rate" column in the resource deployment table: `capture_rate = resource_energy_rev / avg_lmp`. Solar at 0.78 means it earns 78% of the average price — immediately communicates cannibalization to the user.
>
> **Key files to modify**:
> - `scripts/market_simulation.py` — `compute_market_deployment()` (line 1421, specifically 1463 and 1518), `run_market_simulation()` (line ~2162), `compute_energy_revenue_by_resource()` (line 1032, no changes needed — already works)
> - `scripts/pipeline_config.py` — add `CANNIBALIZATION_ENABLED` flag
> - `backend/models.py` — add `energy_rev_by_resource` and `capture_rates` to `YearResult` (line 291)
> - `frontend/js/results.js` — display capture rates in deployment table
>
> **Performance constraint**: `compute_energy_revenue_by_resource()` is O(R × H) where R = number of resources (~7) and H = 8760. With numpy vectorization this is <1ms. The intra-deployment feedback adds one call per deployed tranche (typically 3–5 tranches per year). Total overhead: <10ms per year-ISO. Negligible.
>
> **Validation**:
> - **CAISO at Medium costs**: Solar energy revenue should be 15–25% below avg_lmp (capture rate 0.75–0.85) due to midday price depression from the duck curve. Wind should be 5–10% above avg_lmp (capture rate 1.05–1.10) due to evening generation when prices are highest.
> - **ERCOT**: Wind energy revenue should be 10–20% below avg_lmp (overnight generation in low-price hours). Solar closer to avg_lmp.
> - **Total deployment impact**: At >50% clean penetration, total deployment should decrease 5–15% vs current model because marginal solar tranches are no longer profitable once cannibalization is applied. This is the correct direction — the current model systematically overestimates clean deployment at high penetrations.
> - **Cross-check**: Compare capture rates against published CAISO DMM data (2024) and ERCOT IMM data. CAISO solar capture rate was ~0.70 in 2024; wind was ~1.05.
>
> **Read these files first**: `scripts/market_simulation.py` (lines 1032–1048 for existing revenue function, lines 1421–1600 for deployment model), `scripts/procurement_utils.py` (line 1166 for sigmoid cannibalization), `AUDIT.md` (Section 2.1).

---

## Prompt 8: ORDC Scarcity Pricing

> **Task**: Replace the demand-quantile statistical LMP overlays with an Operating Reserve Demand Curve (ORDC) pricing mechanism that structurally responds to generation mix changes.
>
> **Context**: The current LMP engine uses demand-quantile statistical overlays (lines 1167–1278 of `lmp_engine.py`) — curve-fit percentile-based adjustments calibrated against historical PJM price distributions. These overlays reproduce historical LMP *shapes* well, but they **cannot predict LMP response to structural generation mix changes**, which is the tool's entire purpose. Adding 10 GW of solar to ERCOT should reduce midday scarcity events (more operating reserves) but have no effect on evening scarcity — the current demand-quantile approach can't produce this structural response because scarcity pricing is tied to demand percentile, not operating reserve margin.
>
> ORDC is what ERCOT actually uses for real-time pricing. The concept: price = marginal cost + VOLL × LOLP(reserves), where LOLP increases as reserves fall below target. This makes price formation physically responsive to the generation mix rather than statistically anchored to a historical distribution shape.
>
> **What to build**: An ORDC pricing layer that replaces the scarcity tail and high-demand congestion adder while keeping the low-demand/VRE surplus effects (which model real phenomena ORDC doesn't address).
>
> **Implementation approach**:
> 1. **Add ORDC parameters** to `pipeline_config.py`:
>    ```python
>    ORDC_PARAMS = {
>        'ERCOT': {'voll': 5000, 'reserve_target_mw': 3000, 'lolp_k': 0.003},
>        'PJM':   {'voll': 3700, 'reserve_target_mw': 5500, 'lolp_k': 0.002},
>        'CAISO': {'voll': 2000, 'reserve_target_mw': 3000, 'lolp_k': 0.003},
>        'NYISO': {'voll': 2500, 'reserve_target_mw': 2000, 'lolp_k': 0.003},
>        'NEISO': {'voll': 2000, 'reserve_target_mw': 1500, 'lolp_k': 0.003},
>        'MISO':  {'voll': 3500, 'reserve_target_mw': 4000, 'lolp_k': 0.002},
>        'SPP':   {'voll': 2000, 'reserve_target_mw': 2500, 'lolp_k': 0.003},
>    }
>    # voll: Value of Lost Load ($/MWh) — ERCOT $5,000 (PUCT), PJM $3,700 (1/3 × penalty factor)
>    # reserve_target_mw: Target operating reserve level from NERC/ISO standards
>    # lolp_k: Steepness of LOLP sigmoid — controls how sharply price rises as reserves fall
>    SCARCITY_MODE = 'ordc'  # 'ordc' or 'demand_quantile'
>    ```
>    Sources: ERCOT PUCT ORDC parameters (Docket 52373), PJM RPM penalty factor, CAISO/NYISO/NEISO from FERC filings and reliability standards.
>
> 2. **Add ORDC computation to `PriceModel`** (line 541 of `lmp_engine.py`):
>    - Add method `compute_ordc_adder(self, reserves_mw)`:
>      ```python
>      def compute_ordc_adder(self, reserves_mw):
>          """ORDC price adder: VOLL × LOLP(reserves)."""
>          lolp = 1.0 / (1.0 + np.exp(self.lolp_k * (reserves_mw - self.reserve_target_mw)))
>          return self.voll * lolp
>      ```
>    - Add ORDC parameters to `PriceModel.__init__()` loaded from `ORDC_PARAMS`
>    - ISO-specific subclasses (e.g., `ERCOTPriceModel` at line ~670) inherit or override
>
> 3. **Compute hourly reserves** in `compute_hourly_lmp_vectorized()` (line 1014):
>    - After the merit-order dispatch loop, reserves are derivable from existing data:
>      ```python
>      total_available_mw = sum(unit['capacity_mw'] for unit in stack)
>      reserves_mw = total_available_mw - residual_mw  # per hour, already vectorized
>      ```
>    - Add clean resource ELCC contribution: `total_reserves = fossil_reserves + clean_elcc_mw` where `clean_elcc_mw` accounts for firm clean capacity (nuclear, hydro) that contributes to reserves
>
> 4. **Replace scarcity pricing blocks** in the demand-quantile layer:
>    - **Remove or gate** the high-demand congestion adder (lines 1181–1191) and scarcity tail (lines 1193–1204) when `SCARCITY_MODE == 'ordc'`
>    - **Replace with**: `hourly_lmp += price_model.compute_ordc_adder(reserves_mw)` applied to all hours (ORDC adder is ~$0 when reserves are ample, rises steeply only when reserves are tight)
>    - **Keep**: Low-demand negative pricing (lines 1224–1240), mid-low compression (lines 1242–1256), VRE surplus merit-order effect (lines 1258–1278) — these model real phenomena (must-run pricing, negative prices) that ORDC doesn't address
>
> 5. **Vectorize** the ORDC computation — the sigmoid is a numpy ufunc, so `lolp = 1 / (1 + np.exp(k * (reserves_arr - target)))` operates on the full 8760-hour array in one call. No loop needed.
>
> 6. **Config toggle**: `SCARCITY_MODE = 'ordc' | 'demand_quantile'` in `pipeline_config.py`. Default: `'ordc'`. The demand-quantile mode is preserved as fallback and for comparison.
>
> 7. **Setup page**: Add "Scarcity Pricing" toggle — "ORDC" (default) / "Statistical" (legacy). Brief tooltip explaining the difference.
>
> **Key files to modify**:
> - `scripts/lmp_engine.py` — `PriceModel` class (line 541), `compute_hourly_lmp_vectorized()` (line 1014, specifically lines 1181–1204 for replacement), ISO-specific subclasses
> - `scripts/pipeline_config.py` — `ORDC_PARAMS`, `SCARCITY_MODE`
> - `frontend/setup.html` / `frontend/js/setup.js` — scarcity mode toggle
>
> **Performance constraint**: The ORDC sigmoid is a single numpy vectorized operation on an 8760-element array — <0.1ms. Strictly faster than the current demand-quantile overlays (which involve multiple masked operations). No runtime impact.
>
> **Validation**:
> - **ERCOT behavioral test**: Run two scenarios — (A) current ERCOT mix, (B) add 10 GW solar. With ORDC: midday scarcity events should decrease (more reserves from solar), evening scarcity events should be unchanged. With demand-quantile: both periods change proportionally (wrong).
> - **ERCOT calibration**: ERCOT had ~100 hours with ORDC adder > $100/MWh in 2024. The model should produce a similar count at current fuel prices and generation mix.
> - **PJM calibration**: P99 LMP should be $800–1,500/MWh (consistent with PJM penalty factor events). P50 LMP should be largely unchanged (ORDC adder is ~$0 for most hours).
> - **Cross-ISO**: ISOs with tight reserve margins (ERCOT, PJM) should show higher ORDC impact than ISOs with excess capacity (SPP, MISO). This matches reality.
>
> **Read these files first**: `scripts/lmp_engine.py` (lines 541–600 for PriceModel, lines 1167–1278 for demand-quantile layer), `scripts/pipeline_config.py`, `AUDIT.md` (Section 2.3), `docs/Demand_Quantile_Pricing_Methodology.md`.

---

## Prompt 9: Synthetic Data Warning + Methodology Disclosure

> **Task**: Add prominent UI warnings when the tool runs on synthetic (fabricated) data, and create a methodology disclosure page that honestly positions the tool as a screening complement to production models.
>
> **Context**: When Step 2.2 parquets are absent, `_generate_synthetic_step3_data()` (line 1719 of `market_simulation.py`) fabricates resource mixes using hardcoded linear ramps (lines 1729–1737). These are illustrative at best and have no calibration to physics. The UI gives **zero warning** — users see the same results presentation whether data comes from a 21M-mix physics optimization or from hardcoded guesses. The audit (§2.6) flags this as a credibility risk.
>
> **What to build**: (A) Data source tracking through the full pipeline with frontend warning banners, (B) A methodology disclosure page that explains what the model does and doesn't do.
>
> **Implementation approach**:
>
> **Part A — Synthetic Data Warning:**
> 1. **Track data source in `load_step3_data()`** (line 1645 of `market_simulation.py`):
>    - Currently returns `all_data` dict. Change return to include source: `return all_data, 'parquet'` when loading from parquets, `return all_data, 'synthetic'` when falling back to `_generate_synthetic_step3_data()` (around line 1714 where the fallback triggers)
>    - Thread `data_source` through `run_market_simulation()` and into the year result dicts
> 2. **Add to API response** in `backend/models.py`:
>    - Add `data_source: str = 'parquet'` field to `SimulationResponse` (line 338) and optionally to `YearResult` (line 291)
>    - In `backend/main.py`, populate from simulation results
> 3. **Frontend warning banner** in `frontend/js/results.js`:
>    - At the top of results rendering, check `response.data_source`
>    - If `'synthetic'`: inject a persistent, non-dismissible warning banner using the existing `.insight-box.insight-warn` CSS pattern (from the shared design system):
>      ```html
>      <div class="insight-box insight-warn">
>        <strong>⚠ ILLUSTRATIVE ONLY</strong> — Running with synthetic resource mix profiles
>        (calibrated physics data not available). Results show directional patterns only.
>        For production-quality results, run the full optimization pipeline (Steps 1-2).
>      </div>
>      ```
>    - Banner should appear at the top of EVERY results tab/view, not just the first
> 4. **New `/api/data-status` endpoint** in `backend/main.py`:
>    - `GET /api/data-status` → returns `{iso: 'parquet' | 'synthetic'}` per ISO
>    - Useful for the setup page to show data availability before running a simulation
> 5. **Data tier badges**: Below each ISO heading on results, show a small badge: "Physics Data" (green) or "Synthetic: Illustrative Only" (orange/red). Use existing `.story-badge` / `.story-badge-red` CSS.
>
> **Part B — Methodology Disclosure Page:**
> 6. **Create `frontend/methodology.html`** following the existing `guide.html` template:
>    - **Section 1 — What This Tool Does**: Reduced-form market screening model. Evaluates generator profitability, clean energy deployment economics, and retirement pressure across 7 US ISOs under parametric sensitivity scenarios. Uses merit-order dispatch, multi-stream revenue decomposition, Wright's Law learning curves.
>    - **Section 2 — What This Tool Does NOT Do**: Not a production capacity expansion model. Does not co-optimize generation + storage + transmission. Does not perform unit commitment with physical constraints. Does not model nodal/zonal transmission (unless Prompt 1 is implemented). Not suitable for: investment decisions, regulatory filings, fleet retirement timing predictions.
>    - **Section 3 — When to Use This Tool**: Directional screening ("Does higher gas prices accelerate clean deployment?"). Identifying which ISOs/scenarios warrant detailed IPM/PLEXOS modeling. CCS retrofit breakeven analysis. Nuclear retirement risk screening. Stakeholder education. Relative scenario comparison.
>    - **Section 4 — When NOT to Use**: Absolute LMP forecasting. Optimal resource portfolio design. Policy impact quantification with specific emission reduction targets. Retirement timing for specific plants.
>    - **Section 5 — Comparison to Production Models**: Table comparing this tool vs GenX, ReEDS, PLEXOS, IPM, Aurora across dimensions (dispatch fidelity, network model, storage co-opt, VRE curtailment, reliability). Adapted from AUDIT.md §4.1.
>    - **Section 6 — Data Sources**: EIA-860/923, eGRID, EPA CAMPD, PJM/ERCOT/CAISO SOM reports. With citations.
>    - **Section 7 — Known Limitations**: Honest list from AUDIT.md §2 (VRE cannibalization approximation, statistical LMP overlays, heuristic retirement, no inter-regional flows).
>    - Use the standard page template: `.header` with shared-header.js, `.content-section`, nav bar via nav.js.
> 7. **Add to navigation**: Add "Methodology" link to the nav bar in `frontend/js/nav.js`, positioned between "Guide" and "IPP Report".
>
> **Key files to modify**:
> - `scripts/market_simulation.py` — `load_step3_data()` (line 1645), `_generate_synthetic_step3_data()` (line 1719), `run_market_simulation()` (line 1919)
> - `backend/models.py` — `SimulationResponse` (line 338), `YearResult` (line 291)
> - `backend/main.py` — `simulate()` endpoint, new `/api/data-status` endpoint
> - `frontend/js/results.js` — warning banner injection, data tier badges
> - `frontend/js/nav.js` — add methodology link
> - **New file**: `frontend/methodology.html`
>
> **Validation**: Start the backend without parquets in `data/step2.2-cost/`. Run a simulation. The warning banner must appear on every results view. Restart with parquets present — banner must not appear. The methodology page must render correctly and be accessible from the nav bar on all pages.
>
> **Read these files first**: `scripts/market_simulation.py` (lines 1645–1750 for data loading and synthetic fallback), `frontend/js/results.js`, `frontend/guide.html` (template for methodology page), `AUDIT.md` (Section 2.6 and 6.3).

---

## Prompt 10: IPM Trigger Indicators

> **Task**: Add automated indicators that flag when simulation results cross thresholds where the screening model's approximations break down, recommending production-model validation (IPM, PLEXOS, GenX).
>
> **Context**: The market simulator is positioned as a **pre-screening tool** — more sophisticated than spreadsheets but less rigorous than production models. Its highest-value function is telling users **where it's worth investing in a full IPM/PLEXOS run**. Currently it gives no signal about when its own results become unreliable. This prompt adds that signal, making the tool genuinely useful as a "triage" step in the modeling workflow.
>
> **What to build**: A set of trigger conditions computed per ISO per year that flag when specific modeling limitations become binding. Each trigger includes a severity level, a plain-English explanation, and a recommendation for which type of production analysis would address the limitation.
>
> **Trigger definitions**:
>
> | Trigger ID | Condition | Severity | Explanation |
> |-----------|-----------|----------|-------------|
> | `VRE_CANNIBALIZATION` | VRE (solar+wind) penetration > 40% of generation | Medium at 40–60%, High at >60% | "VRE penetration above 40% causes significant price cannibalization effects. A production dispatch model with hourly granularity and curtailment modeling would better quantify revenue erosion and optimal storage sizing." |
> | `TIGHT_RA_MARGIN` | Operating reserve margin < 10% (vs 15% target) | Medium at <10%, High at <5% | "Reserve margins are tight enough that unit commitment constraints (ramp rates, minimum generation, start-up costs) materially affect price formation and reliability. A UC-constrained dispatch model is recommended." |
> | `HIGH_CONGESTION` | Zonal LMP spread > $15/MWh (if zonal mode active) OR VRE deployment exceeds 2× historical queue completion in high-transmission ISOs | Medium at $15–25/MWh, High at >$25/MWh | "Transmission congestion is material. Zonal or nodal dispatch modeling would better capture locational price signals and their impact on resource siting decisions." |
> | `STORAGE_DOMINANCE` | Storage (battery + LDES + H₂) provides > 15% of total energy served | Medium at 15–25%, High at >25% | "Storage is a major contributor to supply. Co-optimized storage dispatch (jointly with generation and unit commitment) would materially change utilization patterns and economics." |
> | `RETIREMENT_CASCADE` | Economic retirement removes > 20% of fossil fleet capacity in a single period | Medium at 20–35%, High at >35% | "Large-scale fossil retirement is occurring. Binary plant-level retirement decisions, reliability-must-run contracts, and regulatory backstop interventions would significantly alter this trajectory. Plant-level modeling (EIA 860 fleet) is recommended." |
> | `NUCLEAR_AT_RISK` | Nuclear all-in revenue falls within $5/MWh of the retirement threshold ($30/MWh) | High (always — this is a cliff) | "Nuclear plant revenue is near the retirement cliff. Small changes in LMP assumptions could flip the retirement decision. Detailed plant-level economics with contract-specific data is recommended before acting on this result." |
>
> **Implementation approach**:
> 1. **Create `compute_ipm_triggers()` function** in `market_simulation.py`:
>    ```python
>    def compute_ipm_triggers(iso, year, year_result, gen_econ, state):
>        """Evaluate IPM trigger conditions for this year's results."""
>        triggers = []
>        clean_pct = year_result['clean_pct']
>        # ... check each condition, append IPMTrigger dicts
>        return triggers
>    ```
>    - Takes the year result dict, generator economics, and simulation state
>    - Returns list of `{'trigger_id': str, 'severity': str, 'explanation': str, 'metric_value': float, 'threshold': float, 'recommended_model': str}`
>
> 2. **Call from the year loop** in `run_market_simulation()` after line 2295 (after emission accounting, where all metrics are available):
>    ```python
>    ipm_triggers = compute_ipm_triggers(iso, year, year_result, gen_econ, state)
>    year_result['ipm_triggers'] = ipm_triggers
>    ```
>
> 3. **Add to API model** in `backend/models.py`:
>    - New `IPMTrigger` Pydantic model: `trigger_id: str`, `severity: str`, `explanation: str`, `metric_value: float`, `threshold: float`, `recommended_model: str`
>    - Add `ipm_triggers: List[IPMTrigger] = []` to `YearResult` (line 291)
>
> 4. **Frontend — trigger cards** in `frontend/js/results.js`:
>    - Below each year's KPI panel, render triggered indicators as colored cards
>    - **High severity**: Red card with exclamation icon and "Production Modeling Recommended" header
>    - **Medium severity**: Amber card with warning icon
>    - Card format: `[Trigger Name] — [Explanation]. Current: [metric_value] | Threshold: [threshold]`
>    - **Aggregate across years**: If the same trigger fires in consecutive years, consolidate into one card with the year range (e.g., "2035–2050")
>
> 5. **Summary badge** on ISO comparison view:
>    - Add a row to the ISO comparison table: "IPM Recommended: Yes/No"
>    - "Yes" (red) if any High-severity trigger fires in any year for that ISO
>    - "Maybe" (amber) if only Medium triggers
>    - "No" (green) if no triggers
>    - Tooltip lists which triggers fired and when
>
> 6. **Trigger suppression**: If the user has already acknowledged triggers (e.g., via a "Dismiss" button), don't re-show on re-render. Store dismissed state in sessionStorage.
>
> **Key files to modify**:
> - `scripts/market_simulation.py` — new `compute_ipm_triggers()` function, add to year loop (after line 2295)
> - `backend/models.py` — new `IPMTrigger` model, add `ipm_triggers` field to `YearResult` (line 291)
> - `backend/main.py` — ensure triggers flow through `_build_simulation_response()`
> - `frontend/js/results.js` — trigger card rendering, aggregate view badges
> - `frontend/styles/results.css` — trigger card styling (use existing `.insight-box` variants from `shared.css`)
>
> **Performance constraint**: Trigger computation is pure threshold checks — 6 comparisons per ISO per year. Negligible overhead (<0.01ms).
>
> **Validation**:
> - **CAISO with High demand growth to 2050**: Should trigger `VRE_CANNIBALIZATION` (solar penetration >40% by ~2035) and likely `STORAGE_DOMINANCE` (storage >15% at high clean%).
> - **PJM with High fuel prices**: Should trigger `RETIREMENT_CASCADE` if coal retires aggressively, and possibly `TIGHT_RA_MARGIN`.
> - **All ISOs at Medium defaults, 2025–2030**: Should produce few or no triggers (near-term results within model's calibration domain).
> - **All ISOs at Medium defaults, 2040–2050**: Should produce multiple triggers as extrapolation compounds.
> - **NEISO/NYISO at high clean%**: `NUCLEAR_AT_RISK` should fire when nuclear revenue approaches the retirement threshold — verify by checking that nuclear revenue is within $5/MWh of $30.
>
> **Read these files first**: `scripts/market_simulation.py` (lines 2280–2370 for the year loop where triggers would be computed), `backend/models.py` (line 291 for `YearResult`), `frontend/js/results.js` (results rendering), `AUDIT.md` (Sections 2.1–2.7 for which limitations to trigger on).

---

## Prompt 11: Tech-Differentiated Queue Caps

> **Task**: Replace the uniform GW/year interconnection queue cap with per-technology caps based on LBNL historical completion rate data.
>
> **Context**: The current `QUEUE_CAP_GW` (line 96 of `market_simulation.py`) applies a single GW/year cap uniformly across all clean energy technologies for each ISO. In reality, technologies have vastly different queue completion rates — solar projects complete at ~8 GW/yr nationally (shorter interconnection studies, smaller average project size), wind at ~5 GW/yr, while nuclear and CCS have <10% queue completion rates (~0.5 GW/yr). The undifferentiated cap creates a bias: when solar (fast) and nuclear (slow) compete for the same queue budget, solar deployment is artificially constrained while nuclear gets queue capacity it can't realistically use. Source: LBNL "Queued Up 2024" (Rand et al., 2024), https://emp.lbl.gov/queues.
>
> **What to build**: Per-technology queue caps by ISO, with the total approximately summing to the current uniform cap for backward compatibility.
>
> **Implementation approach**:
> 1. **Add `TECH_QUEUE_CAP_GW` dict** to `market_simulation.py` (near line 96, alongside existing `QUEUE_CAP_GW`):
>    ```python
>    TECH_QUEUE_CAP_GW = {
>        'Medium': {
>            'CAISO':  {'solar': 2.5, 'wind': 0.8, 'offshore_wind': 0.3, 'clean_firm': 0.2, 'ccs_ccgt': 0.4, 'geothermal': 0.3},
>            'ERCOT':  {'solar': 4.0, 'wind': 2.5, 'offshore_wind': 0.2, 'clean_firm': 0.2, 'ccs_ccgt': 0.5, 'geothermal': 0.0},
>            'PJM':    {'solar': 2.5, 'wind': 1.2, 'offshore_wind': 0.5, 'clean_firm': 0.3, 'ccs_ccgt': 0.5, 'geothermal': 0.0},
>            'NYISO':  {'solar': 1.0, 'wind': 0.6, 'offshore_wind': 0.5, 'clean_firm': 0.2, 'ccs_ccgt': 0.2, 'geothermal': 0.0},
>            'NEISO':  {'solar': 0.8, 'wind': 0.8, 'offshore_wind': 0.7, 'clean_firm': 0.2, 'ccs_ccgt': 0.3, 'geothermal': 0.0},
>            'MISO':   {'solar': 2.0, 'wind': 1.5, 'offshore_wind': 0.0, 'clean_firm': 0.2, 'ccs_ccgt': 0.5, 'geothermal': 0.0},
>            'SPP':    {'solar': 1.8, 'wind': 1.5, 'offshore_wind': 0.0, 'clean_firm': 0.1, 'ccs_ccgt': 0.3, 'geothermal': 0.0},
>        },
>        'Low': {  # ~50% of Medium (status quo permitting)
>            'CAISO':  {'solar': 1.3, 'wind': 0.4, 'offshore_wind': 0.15, 'clean_firm': 0.1, 'ccs_ccgt': 0.2, 'geothermal': 0.15},
>            'ERCOT':  {'solar': 2.0, 'wind': 1.3, 'offshore_wind': 0.1, 'clean_firm': 0.1, 'ccs_ccgt': 0.25, 'geothermal': 0.0},
>            'PJM':    {'solar': 1.3, 'wind': 0.6, 'offshore_wind': 0.25, 'clean_firm': 0.15, 'ccs_ccgt': 0.25, 'geothermal': 0.0},
>            'NYISO':  {'solar': 0.5, 'wind': 0.3, 'offshore_wind': 0.25, 'clean_firm': 0.1, 'ccs_ccgt': 0.1, 'geothermal': 0.0},
>            'NEISO':  {'solar': 0.4, 'wind': 0.4, 'offshore_wind': 0.35, 'clean_firm': 0.1, 'ccs_ccgt': 0.15, 'geothermal': 0.0},
>            'MISO':   {'solar': 1.0, 'wind': 0.8, 'offshore_wind': 0.0, 'clean_firm': 0.1, 'ccs_ccgt': 0.25, 'geothermal': 0.0},
>            'SPP':    {'solar': 0.9, 'wind': 0.8, 'offshore_wind': 0.0, 'clean_firm': 0.05, 'ccs_ccgt': 0.15, 'geothermal': 0.0},
>        },
>        'High': {  # ~133% of Medium (FERC Order 2023 reforms)
>            'CAISO':  {'solar': 3.3, 'wind': 1.1, 'offshore_wind': 0.4, 'clean_firm': 0.3, 'ccs_ccgt': 0.5, 'geothermal': 0.4},
>            'ERCOT':  {'solar': 5.3, 'wind': 3.3, 'offshore_wind': 0.3, 'clean_firm': 0.3, 'ccs_ccgt': 0.7, 'geothermal': 0.0},
>            'PJM':    {'solar': 3.3, 'wind': 1.6, 'offshore_wind': 0.7, 'clean_firm': 0.4, 'ccs_ccgt': 0.7, 'geothermal': 0.0},
>            'NYISO':  {'solar': 1.3, 'wind': 0.8, 'offshore_wind': 0.7, 'clean_firm': 0.3, 'ccs_ccgt': 0.3, 'geothermal': 0.0},
>            'NEISO':  {'solar': 1.1, 'wind': 1.1, 'offshore_wind': 0.9, 'clean_firm': 0.3, 'ccs_ccgt': 0.4, 'geothermal': 0.0},
>            'MISO':   {'solar': 2.7, 'wind': 2.0, 'offshore_wind': 0.0, 'clean_firm': 0.3, 'ccs_ccgt': 0.7, 'geothermal': 0.0},
>            'SPP':    {'solar': 2.4, 'wind': 2.0, 'offshore_wind': 0.0, 'clean_firm': 0.15, 'ccs_ccgt': 0.4, 'geothermal': 0.0},
>        },
>    }
>    TECH_DIFFERENTIATED_QUEUE = True  # Set False to use legacy uniform QUEUE_CAP_GW
>    ```
>    Per-tech caps approximately sum to the existing uniform `QUEUE_CAP_GW` per ISO (e.g., CAISO Medium: 2.5+0.8+0.3+0.2+0.4+0.3 = 4.5 GW vs current 4.5 GW).
>
> 2. **Modify `compute_market_deployment()`** (line 1421):
>    - Replace the single `queue_remaining_gw` parameter with a `tech_queue_budget` dict: `{resource: remaining_gw}`
>    - In the deployment loop (starting at line ~1547), constrain each resource by its own tech-specific budget:
>      ```python
>      tech_cap = tech_queue_budget.get(res, 0)
>      max_deploy_twh = min(max_twh, tech_cap * cf * 8.760)
>      ```
>    - After deploying, decrement only that resource's budget: `tech_queue_budget[res] -= deployed_gw`
>
> 3. **Modify `run_market_simulation()`** (lines 2084–2098):
>    - Replace the single `queue_budget_gw` accumulator with per-tech budgets:
>      ```python
>      if TECH_DIFFERENTIATED_QUEUE:
>          tech_budget = {res: cap * years_per_step
>                        for res, cap in TECH_QUEUE_CAP_GW[queue_level][iso].items()}
>      else:
>          # Legacy: distribute uniform cap equally (current behavior)
>          tech_budget = {res: queue_remaining_gw / len(DEPLOYABLE_RESOURCES)
>                        for res in DEPLOYABLE_RESOURCES}
>      ```
>    - Pass `tech_budget` to `compute_market_deployment()` instead of `queue_remaining_gw`
>
> 4. **Optional: Flex pool.** Some ISOs allow queue slots to be fungible across technologies. Add:
>    ```python
>    QUEUE_FLEX_FRACTION = 0.20  # 20% of total cap available as flex pool
>    ```
>    After a resource exhausts its dedicated budget, it can draw from the flex pool (shared across all techs). This prevents hard cutoffs when one tech is slightly over-subscribed.
>
> 5. **Backward compatibility**: When `TECH_DIFFERENTIATED_QUEUE = False`, fall back to existing `QUEUE_CAP_GW` behavior. Add a validation check that per-tech caps sum to approximately the uniform cap (within ±10%) to catch configuration errors.
>
> 6. **Setup page**: Add "Queue Model" toggle — "Uniform" / "Tech-Differentiated" (default). Brief tooltip: "Tech-differentiated caps reflect LBNL data showing solar and wind complete interconnection faster than nuclear or CCS."
>
> **Key files to modify**:
> - `scripts/market_simulation.py` — `TECH_QUEUE_CAP_GW` constants (near line 96), `compute_market_deployment()` (line 1421), `run_market_simulation()` (lines 2084–2098)
> - `scripts/pipeline_config.py` — `TECH_DIFFERENTIATED_QUEUE` flag, `QUEUE_FLEX_FRACTION`
> - `backend/models.py` — optionally add `queue_by_tech: Dict[str, float]` to `YearResult` for transparency
> - `frontend/setup.html` / `frontend/js/setup.js` — queue model toggle
>
> **Performance constraint**: Only changes how the queue budget is sliced — same number of iterations, same deployment loop structure. Zero runtime impact.
>
> **Validation**:
> - **Compare trajectories**: Uniform vs tech-differentiated at Medium costs, ERCOT, 2025–2050. With tech caps: solar deployment should increase 20–40% (larger dedicated budget) and nuclear/CCS deployment should decrease (smaller dedicated budget but reflecting actual completion rates).
> - **Sum check**: Per-tech caps for each ISO should sum to within ±10% of the uniform `QUEUE_CAP_GW` for that ISO.
> - **National totals**: Sum across all ISOs: solar should be ~8 GW/yr, wind ~5 GW/yr at Medium — matching LBNL 2024 national completion rates.
> - **Total clean% at 2050**: Should be slightly higher with tech-differentiated caps because fast-deploying technologies (solar, wind) are no longer constrained by slow technologies (nuclear, CCS) consuming shared queue budget.
>
> **Read these files first**: `scripts/market_simulation.py` (lines 90–109 for existing `QUEUE_CAP_GW`, lines 1421–1600 for deployment model, lines 2084–2098 for queue budget management), `AUDIT.md` (Section 2.3 on queue cap bias).

---

## Usage Notes

- Each prompt is designed to be self-contained — paste into a new Claude Code session with access to the `market-simulator/` directory
- **Prompt 2 (LP Storage Co-Dispatch) has been implemented** but produced limited differentiation vs greedy dispatch — extra compute with minimal value. This validates VRE cannibalization (Prompt 7) as the higher-impact fix.

### Recommended execution order:

| Order | Prompt | Effort | Impact | Dependencies |
|-------|--------|--------|--------|-------------|
| 1 | **Prompt 9** (Synthetic Warning) | ~2–3 hours | Quick credibility win | None |
| 2 | **Prompt 7** (VRE Cannibalization) | ~1 day | **Highest** — fixes #1 audit criticism | None |
| 3 | **Prompt 11** (Tech Queue Caps) | ~half day | Medium — removes deployment bias | None |
| 4 | **Prompt 8** (ORDC Scarcity) | ~1–2 days | High — makes LMP structurally responsive | None |
| 5 | **Prompt 10** (IPM Triggers) | ~1 day | High for positioning | Partial dep on 7 (VRE metric) |
| 6 | **Prompt 6** (Confidence Viz) | ~half day | Medium — UX transparency | None |
| 7 | **Prompt 3** (Inter-Regional Flows) | ~1 day | Medium — corrects RA in import-heavy ISOs | None |
| 8 | **Prompt 4** (Demand Response) | ~1 day | Medium — fixes scarcity overshoot | None |
| 9 | **Prompt 5** (Backtesting) | ~2 days | High for credibility — best after 7/8 | After 7, 8 |
| 10 | **Prompt 1** (Zonal Decomposition) | ~2–3 days | Highest structural — most complex | After 3, 4 |

### Groupings for parallel sessions:
- **Group A** (independent): Prompts 7, 9, 11 — can all be worked in parallel
- **Group B** (independent): Prompts 3, 4, 8 — can all be worked in parallel after Group A
- **Group C** (sequential): Prompt 10 after 7; Prompt 5 after 7+8; Prompt 1 after 3+4
