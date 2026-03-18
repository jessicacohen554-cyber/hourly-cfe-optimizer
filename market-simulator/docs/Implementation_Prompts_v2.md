# Implementation Prompts v2 — Synchronization, QA/QC, and Documentation

**Source**: Code audit of Integration gaps between Prompts 1-6 (foundational features) and Prompts 7-11 (audit-driven fixes)
**Purpose**: Copy-paste ready prompts for Claude Code sessions. Each prompt is self-contained with full context, specific file/line references, and validation criteria.

**Background**: Prompts 1-6 were implemented first. Prompts 7-11 were then designed with a recommended execution order that placed some 7-11 prompts before some 1-6 prompts. Prompts 7-11 were implemented in the recommended order, but 1-6 were not re-run. This created 7 synchronization conflicts ranging from critical to low severity.

---

## Execution Order

```
V2-1 (ORDC+Zonal) ──┐
V2-2 (IPM+Zonal)  ──┤── Group A: Can run in parallel (independent zonal integrations)
V2-4 (Flows+DR)   ──┘
         │
V2-3 (Cannibalization sync) ── Group B: After V2-1 (needs ORDC-in-zonal)
         │
V2-5 (Backtesting alignment) ── Group B: After V2-1, V2-3 (needs updated models)
         │
V2-6 (QA/QC Testing) ── Group C: After all sync prompts
V2-7 (UI/CSV QA) ── Group C: Parallel with V2-6
         │
V2-8 (Documentation) ── Group D: Last (reflects all changes)
```

---

## Category 1: Synchronization Prompts

### Prompt V2-1: Integrate ORDC Scarcity Pricing into Zonal LMP Path

> **Severity**: Critical
> **Problem**: ORDC scarcity pricing (Prompt 8) is only applied in the copper-plate LMP path. The zonal LMP path (Prompt 1) still uses demand-quantile pricing overlays exclusively. When zonal mode is active (e.g., PJM with 5 zones), reserve-margin-responsive scarcity pricing is completely missing, causing zonal LMPs to underestimate scarcity-driven price spikes.
>
> **Evidence**:
> - `lmp_engine.py:1210-1219`: ORDC adder computed and applied inside `compute_hourly_lmp_vectorized()` only when `SCARCITY_MODE == 'ordc'`
> - `lmp_engine.py:596-615`: `PriceModel` class has `compute_ordc_adder(reserves_mw)` method ready for use
> - `zonal_lmp.py:386-441`: `_apply_pricing_layers()` only applies demand-quantile adders (`dq_high_percentile`, `dq_scarcity_percentile`, `dq_low_percentile`). ORDC is never called here.
> - `pipeline_config.py:42`: `SCARCITY_MODE = 'ordc'` is set globally, but `zonal_lmp.py` ignores this setting
>
> **What to fix**:
> 1. Read `scripts/zonal_lmp.py`, focusing on `_apply_pricing_layers()` (line ~401) and `compute_zonal_lmp_hourly()` (line ~257).
> 2. Read `scripts/lmp_engine.py`, focusing on the ORDC block at lines 1210-1219 and `PriceModel.compute_ordc_adder()` at line 601.
> 3. Read `scripts/pipeline_config.py` for `SCARCITY_MODE` (line 42) and `ORDC_PARAMS` (line 50).
> 4. Modify `_apply_pricing_layers()` in `zonal_lmp.py` to check `SCARCITY_MODE`:
>    - If `'ordc'`: Compute per-zone reserves from the zonal dispatch solution (zone supply capacity minus zone dispatched generation). Call `price_model.compute_ordc_adder(zone_reserves_mw)` for each zone. Add the ORDC adder to the zone's LMP column. Skip the demand-quantile scarcity overlay (the `dq_scarcity_percentile` block).
>    - If `'demand_quantile'`: Keep existing behavior unchanged (backward compatible).
>    - Keep the `dq_high_percentile` (congestion) and `dq_low_percentile` (surplus) overlays in both modes — these represent congestion and surplus, not scarcity.
> 5. Pass `SCARCITY_MODE` and any needed fossil capacity data through `compute_zonal_lmp_hourly()` to `_apply_pricing_layers()`. The zonal dispatch solution already has per-zone generation; compute reserves as `zone_capacity - zone_dispatched`.
> 6. Ensure `PriceModel` is already passed into the zonal path (it is — check `compute_zonal_lmp_hourly()` signature).
>
> **Validation**:
> - Run ERCOT with zonal mode + ORDC: High-stress evening hours (reserves < 2000 MW) should show LMP spikes > $200/MWh in affected zones. Compare against copper-plate ORDC results — zonal should produce HIGHER spreads in constrained zones, not lower.
> - Run PJM with zonal mode: Western zone (cheap coal/gas) should have lower scarcity premium than EMAAC (load pocket with higher congestion + lower local reserves).
> - Regression check: Set `SCARCITY_MODE = 'demand_quantile'` and confirm zonal LMP output is identical to pre-change behavior.
>
> **Files to modify**: `scripts/zonal_lmp.py` (primary), `scripts/pipeline_config.py` (if any new constants needed)
> **Files to read first**: `scripts/zonal_lmp.py`, `scripts/lmp_engine.py` (lines 596-620, 1200-1220), `scripts/pipeline_config.py` (lines 42-70)
> **Dependencies**: None (can run in parallel with V2-2 and V2-4)

---

### Prompt V2-2: Wire Zonal Congestion Data into IPM Trigger Indicators

> **Severity**: Critical
> **Problem**: IPM trigger indicators (Prompt 10) include a `HIGH_CONGESTION` trigger, but it's computed from gross VRE deployment ratios — not from actual zonal congestion data. Meanwhile, the zonal LP solver (Prompt 1) computes inter-zonal flows and zonal price spreads, but this data is discarded immediately after LMP calculation. A region could deploy VRE in uncongested zones and still get flagged as "high congestion," while actual binding transmission constraints go undetected.
>
> **Evidence**:
> - `market_simulation.py:507`: `zonal_lmp_matrix, system_lmp, _, zonal_stats = compute_zonal_lmp_hourly(...)` — `zonal_stats` is computed but never stored or passed downstream.
> - `market_simulation.py:1549-1573`: `HIGH_CONGESTION` trigger logic uses `vre_gw / queue_cap` ratio as a proxy for congestion. No actual zonal price spread data is used.
> - `market_simulation.py:2748-2751`: `compute_ipm_triggers()` receives only `year_result` (aggregate metrics), not `zonal_stats`.
> - `zonal_lmp.py:273-284`: `compute_zonal_lmp_hourly()` returns `zonal_stats` dict containing per-zone LMP stats and inter-zonal flow data.
>
> **What to fix**:
> 1. Read `scripts/market_simulation.py` at line 507 (where `zonal_stats` is returned but discarded) and lines 1471-1620 (the full `compute_ipm_triggers()` function).
> 2. Read `scripts/zonal_lmp.py` to understand what `zonal_stats` contains (zone names, per-zone avg/P90 LMP, flow utilization).
> 3. Persist `zonal_stats` into `year_result` dict. At `market_simulation.py` around line 507, store: `zonal_congestion_data = zonal_stats` and later add it to `year_result` (around line 2701-2745).
> 4. Modify `compute_ipm_triggers()` to accept `zonal_stats` as an optional parameter.
> 5. Replace the `HIGH_CONGESTION` trigger logic (lines 1549-1573) with actual zonal data:
>    - **Medium trigger**: Max inter-zonal LMP spread (P50) > $15/MWh, OR any interface flow utilization > 70% for > 1000 hours/year
>    - **High trigger**: Max inter-zonal LMP spread (P50) > $25/MWh, OR any interface at 95%+ utilization for > 500 hours/year
>    - Include the specific zone pair with highest spread in the trigger `explanation` field
>    - Fallback: If `zonal_stats` is None (copper-plate mode), keep existing VRE-ratio proxy logic
> 6. Add `zonal_lmp_spread` and `congested_hours` to `year_result` for frontend display.
> 7. Update `frontend/js/results.js` to show zonal spread data in IPM trigger cards when available.
>
> **Validation**:
> - Run PJM (5 zones) at Medium fuel: Should see non-trivial zonal spreads ($8-15/MWh) but likely Medium (not High) severity.
> - Run ERCOT (4 zones) with high wind: West zone should show congestion when wind output exceeds West-North transfer limit. Should trigger HIGH_CONGESTION if spread > $25/MWh.
> - Run copper-plate mode: Should fall back to existing VRE-ratio trigger logic. Confirm identical behavior to pre-change.
>
> **Files to modify**: `scripts/market_simulation.py` (primary — persist zonal_stats + update trigger logic), `frontend/js/results.js` (display)
> **Files to read first**: `scripts/market_simulation.py` (lines 498-520, 1471-1620, 2700-2760), `scripts/zonal_lmp.py` (lines 257-390)
> **Dependencies**: None (can run in parallel with V2-1 and V2-4)

---

### Prompt V2-3: Make VRE Cannibalization ORDC-Aware and Zone-Aware

> **Severity**: High
> **Problem**: VRE cannibalization feedback (Prompt 7) applies a flat sigmoid depression to solar/wind energy revenue as VRE penetration rises. But it doesn't account for two things introduced by other prompts: (1) ORDC scarcity pricing (Prompt 8) creates a price floor in reserve-short hours that partially offsets cannibalization — even at 60% VRE, evening scarcity keeps average LMP higher than the sigmoid predicts. (2) Zonal LMP (Prompt 1) means cannibalization should vary by zone — CAISO SP15 solar curtailment is much worse than NP15.
>
> **Evidence**:
> - `market_simulation.py:1866-1877`: Cannibalization sigmoid: `depression = 0.55 * sigmoid(vre_penetration - 0.6)` applied to `per_resource_energy_rev.get(vre_res, avg_lmp)`. No ORDC awareness; no zonal differentiation.
> - `market_simulation.py:1876`: Falls back to system-average `avg_lmp` if per-resource revenue not available.
> - The sigmoid reduces VRE revenue linearly with penetration, but ORDC ensures scarcity hours still command high prices regardless of VRE penetration. The current model over-depresses revenue in markets with active scarcity pricing.
>
> **What to fix**:
> 1. Read `scripts/market_simulation.py` lines 1860-1920 (cannibalization + capture rate calculation) and lines 1210-1219 in `lmp_engine.py` (ORDC implementation).
> 2. **ORDC-aware cannibalization**: When `SCARCITY_MODE == 'ordc'`, add an ORDC floor to the cannibalization depression:
>    - Compute `scarcity_hours_fraction` = fraction of hours where ORDC adder > $50/MWh (from the LMP computation).
>    - Apply floor: `depression = max(depression, scarcity_floor)` where `scarcity_floor = scarcity_hours_fraction * 0.3`. This means if 10% of hours have significant scarcity pricing, the maximum depression is capped at 97% (vs current uncapped sigmoid).
>    - Pass `scarcity_hours_fraction` through to `compute_market_deployment()` — compute it during LMP calculation and store in a variable accessible to the deployment function.
> 3. **Zone-aware capture rates**: When zonal LMP is active, compute per-zone VRE penetration and use zone-level capture rates instead of system average:
>    - Solar capture rate should use the zone where solar is predominantly located (e.g., CAISO SP15, ERCOT West)
>    - Wind capture rate should use the wind-heavy zone (ERCOT West, PJM Western, MISO North)
>    - If zonal data is not available, fall back to system-average (existing behavior)
> 4. Add `ordc_scarcity_hours` to `year_result` dict for transparency.
>
> **Validation**:
> - CAISO at 60% VRE: With ORDC, capture rate should be ~0.75-0.80 (not the ~0.65 the current sigmoid would produce) because evening scarcity hours still pay $200+/MWh.
> - ERCOT at 50% wind: Capture rate should be ~0.95-1.00 (wind produces at night when demand is moderate, ORDC rarely fires).
> - Regression: With `SCARCITY_MODE = 'demand_quantile'`, output should be identical to pre-change.
>
> **Files to modify**: `scripts/market_simulation.py` (cannibalization logic at lines 1860-1920, deployment function)
> **Files to read first**: `scripts/market_simulation.py` (lines 1640-1930, 2490-2520), `scripts/lmp_engine.py` (lines 1200-1220)
> **Dependencies**: V2-1 should be implemented first (so zonal LMP has ORDC), but this prompt can proceed independently for the ORDC-awareness part.

---

### Prompt V2-4: Persist Zonal Flows for Export + Align DR Triggers with ORDC

> **Severity**: Low-Medium
> **Problem**: Two lower-severity desynchronizations: (1) Inter-regional flows (Prompt 3) are computed by the zonal LP solver but immediately discarded — they're never exported for validation, visualization, or downstream analysis. (2) Demand response trigger prices (Prompt 4) are hardcoded per ISO (e.g., PJM $100/MWh, ERCOT $200/MWh) and were calibrated before ORDC scarcity pricing (Prompt 8) was added. The triggers are independent of actual scarcity pricing behavior.
>
> **Evidence**:
> - `market_simulation.py:507`: `zonal_stats` returned from `compute_zonal_lmp_hourly()` but assigned to local variable only — never stored.
> - `pipeline_config.py:490-497`: `DEMAND_RESPONSE` dict has fixed `trigger_price` values per ISO, calibrated to pre-ORDC price levels.
> - `market_simulation.py:533`: DR activation uses `effective_trigger = dr_params['trigger_price'] * dr_lvl['trigger_mult']` — no reference to ORDC state or current reserve margin.
>
> **What to fix**:
> 1. **Persist zonal flows**:
>    - At `market_simulation.py:507`, store `zonal_stats` into a variable that gets added to `year_result`.
>    - Add to `year_result` dict (around line 2701-2745): `'zonal_stats': zonal_stats` (or a summary subset to keep JSON size reasonable — e.g., per-zone avg LMP, max LMP, flow utilization percentages).
>    - In `backend/models.py`, add `zonal_stats: Optional[dict] = None` to the response model.
>    - In `frontend/js/results.js`, optionally display a zonal flow summary table if data is present.
> 2. **DR-ORDC consistency**:
>    - In `pipeline_config.py`, add a `dr_ordc_link` flag (default `True`) to `DEMAND_RESPONSE` config.
>    - When `dr_ordc_link` is True and `SCARCITY_MODE == 'ordc'`: Set DR trigger dynamically as `max(fixed_trigger, VOLL * 0.05)` — DR activates when ORDC adder exceeds 5% of VOLL (i.e., when LOLP indicates real stress, not just high demand).
>    - This ensures DR responds to actual reserve stress rather than arbitrary price thresholds.
>    - When `SCARCITY_MODE == 'demand_quantile'` or `dr_ordc_link` is False: Use existing fixed trigger prices.
> 3. Add a `dr_activation_mode` field to results output showing which trigger method was used.
>
> **Validation**:
> - Run PJM with zonal mode: `year_result` should contain zonal flow data viewable in results JSON.
> - Run ERCOT with ORDC + DR: ERCOT VOLL=$5000, so dynamic trigger = $250 (vs fixed $200). DR should activate slightly less often but at genuinely stressed moments.
> - Run with `SCARCITY_MODE = 'demand_quantile'`: DR trigger should use fixed prices (no ORDC link).
>
> **Files to modify**: `scripts/market_simulation.py` (persist zonal_stats, update DR trigger), `scripts/pipeline_config.py` (DR config), `backend/models.py` (response schema)
> **Files to read first**: `scripts/market_simulation.py` (lines 498-560, 2700-2750), `scripts/pipeline_config.py` (lines 486-500), `scripts/zonal_lmp.py` (lines 273-290)
> **Dependencies**: None (can run in parallel with V2-1 and V2-2)

---

### Prompt V2-5: Align Backtesting with ORDC, Cannibalization, and Tech Queue Caps

> **Severity**: Medium
> **Problem**: The backtest trajectory module (Prompt 5) validates the model against 2020-2024 observed outcomes. But it was implemented before ORDC scarcity pricing (Prompt 8), VRE cannibalization (Prompt 7), and tech-differentiated queue caps (Prompt 11). The backtest may be using stale pricing/deployment models, producing validation metrics that don't reflect the current model's behavior. Passing validation on the old model doesn't validate the new model.
>
> **Evidence**:
> - `backtest_trajectory.py`: No references to `ORDC`, `ordc`, `cannibali`, or `tech_queue` found anywhere in the file. The backtest calls `run_market_simulation()` (which does incorporate these features), but the backtest's parameter setup, validation targets, and metric interpretation may not account for the behavioral changes.
> - If the backtest passes `conditions` dict without `tech_differentiated_queue: True`, deployment uses uniform queue caps (pre-Prompt 11 behavior).
> - If backtest doesn't pass `scarcity_mode` or the config picks up global `SCARCITY_MODE = 'ordc'`, backtest may inadvertently use ORDC on historical data calibrated for demand-quantile pricing.
>
> **What to fix**:
> 1. Read `scripts/backtest_trajectory.py` fully — understand how it calls `run_market_simulation()`, what `conditions` dict it passes, and how it interprets results.
> 2. **Ensure backtest uses current model features**:
>    - Verify that `conditions` dict includes `tech_differentiated_queue: True` (or whatever flag `market_simulation.py` checks at line 2409).
>    - Verify that `SCARCITY_MODE` is propagated correctly — decide whether backtest should use ORDC (more accurate physics) or demand-quantile (matched to historical calibration). **Recommendation**: Use ORDC for 2023-2024 (ERCOT had active ORDC pricing), demand-quantile for 2020-2022 (pre-ORDC era in most markets).
>    - Verify that VRE cannibalization is active in the backtest path — confirm `per_resource_energy_rev` is being computed and passed through.
> 3. **Update validation targets if needed**:
>    - With ORDC + cannibalization, predicted LMP distribution will differ from demand-quantile. Update tolerance bands if backtest metrics shift.
>    - With tech queue caps, deployment rates will differ. Solar may deploy faster, nuclear slower. Verify this matches 2020-2024 observed deployment rates.
> 4. **Add backtest mode toggle**: If backtest needs to switch between ORDC and demand-quantile by year, add a `backtest_scarcity_mode` config that maps year ranges to pricing models.
> 5. Update the backtest report generation (`generate_report()`) to note which model features were active during validation.
>
> **Validation**:
> - Run full backtest with ORDC + cannibalization + tech queue caps: LMP predictions should be within ±$5/MWh of actuals for 2023-2024. Clean energy share within ±3pp.
> - Compare backtest results with vs without ORDC: ERCOT should improve (ORDC is structurally appropriate); PJM difference should be minimal.
> - Check deployment rate predictions: Solar deployment should be higher with tech queue caps (unconstrained from uniform cap), matching LBNL observed ~8 GW/yr national.
>
> **Files to modify**: `scripts/backtest_trajectory.py` (primary), possibly `scripts/pipeline_config.py` (backtest-specific config)
> **Files to read first**: `scripts/backtest_trajectory.py` (full file), `scripts/market_simulation.py` (lines 2400-2420 for tech queue flag, lines 2490-2520 for deployment call)
> **Dependencies**: V2-1 and V2-3 should be completed first so the backtest validates the fully synchronized model.

---

## Category 2: Comprehensive QA/QC Testing

### Prompt V2-6: End-to-End Application Testing Regime

> **Task**: Conduct a comprehensive QA/QC audit of the market simulator's standalone application (run.bat / run.sh), testing all simulation modes, all ISOs, edge cases, user variable sensitivity, CSV overrides, and error handling.
>
> **Context**: The market simulator is a standalone FastAPI web application launched via `run.bat` (Windows) or `run.sh` (Mac/Linux). After implementing Prompts 1-11 and the v2 synchronization fixes, the entire application needs end-to-end validation to ensure nothing is broken and results respond correctly to user inputs.
>
> **Test Plan — Execute All of the Following**:
>
> **A. Startup & Environment**
> 1. Run `run.bat` (or `run.sh`) from a clean state — verify Python detection, dependency installation, synthetic data generation, and server startup on port 8000.
> 2. Verify all pages load without console errors: `guide.html`, `setup.html`, `fleet-config.html`, `results.html`, `emissions.html`, `ipp-report.html`, `methodology.html`.
> 3. Check navigation bar appears correctly on all pages with proper highlighting.
> 4. Verify shared-header.js SVG waveform renders on all page headers.
>
> **B. Single Simulation Runs — All 7 ISOs**
> 5. Run a default simulation for each ISO (CAISO, ERCOT, PJM, NYISO, NEISO, MISO, SPP) with all parameters at default values. Verify:
>    - Simulation completes without error
>    - Results page populates with charts (LMP trends, capacity mix, retirement timeline, deployment, revenue breakdown)
>    - Year-by-year results table has all years from start to end
>    - Narrative summary is generated and readable
>    - Results data CSV is written to `results/` directory
>    - Plant-level results CSV contains per-generator economics (if Tier 2 mode)
> 6. Verify IPM trigger indicators appear when conditions are met (e.g., CAISO high growth should eventually trigger VRE_CANNIBALIZATION).
> 7. Verify synthetic data warning banner appears when running without optimizer parquets.
> 8. Verify capture rates are displayed per-resource in results (VRE cannibalization output).
>
> **C. Parameter Sensitivity — Variables Respond Correctly**
> 9. **Fuel price sensitivity**: Run PJM with Low, Medium, High natural gas prices. Verify:
>    - Low gas → lower LMP, slower coal retirement, lower gas plant revenue
>    - High gas → higher LMP, faster coal retirement, higher gas plant revenue
>    - Differences should be material ($5-15/MWh LMP change, not $0.50)
> 10. **Demand growth sensitivity**: Run ERCOT with Low (0.5%/yr), Medium (1.5%/yr), High (3%/yr) demand growth. Verify:
>     - High growth → more deployment, higher LMP by 2040+, tighter reserves
>     - Low growth → less deployment, stable/declining LMP
> 11. **Carbon price sensitivity**: Run MISO with $0, $25, $75/ton carbon prices. Verify:
>     - Higher carbon → faster coal retirement, higher clean deployment
>     - $75/ton should make most coal uneconomic within 5-10 years
> 12. **Queue cap sensitivity**: Run SPP with Low, Medium, High interconnection queue caps. Verify:
>     - Higher cap → faster clean deployment → lower LMP sooner
>     - Tech-differentiated caps active — solar cap > wind cap > nuclear cap
> 13. **ORDC vs demand-quantile**: If toggle exists on setup.html, test both modes on ERCOT. Verify:
>     - ORDC produces sharper price spikes in low-reserve hours
>     - Demand-quantile produces smoother price distribution
>
> **D. Edge Cases & Error Handling**
> 14. **Extreme inputs**: Set demand growth to 10%/yr (unrealistic). Verify model doesn't crash; expect IPM triggers to fire for TIGHT_RA_MARGIN.
> 15. **Zero carbon price + low fuel**: Should produce minimal retirement. Verify fleet is stable.
> 16. **Maximum carbon price ($200/ton)**: Should retire all coal rapidly. Verify no negative revenue artifacts.
> 17. **Single-year vs multi-year**: Run a 1-year simulation. Verify results page handles correctly (no chart errors from missing years).
> 18. **Missing data gracefully**: Delete one profile file temporarily. Verify error message is clear, not a stack trace.
>
> **E. Sweep & Sensitivity Modes**
> 19. Run a parameter sweep (2-3 fuel price levels × 2 demand growth levels) on a single ISO. Verify:
>     - All scenario combinations complete
>     - Comparison table/chart shows meaningful differentiation
>     - No duplicate or missing scenarios
> 20. Run sensitivity mode comparing 3 scenarios. Verify side-by-side charts render correctly.
>
> **F. Results Export & Persistence**
> 21. Verify results CSV files contain all expected columns (year, LMP, clean_share, capacity_mw, retirements, deployments, etc.).
> 22. Verify narrative.txt is coherent and references correct ISO/scenario.
> 23. Verify input_parameters.csv accurately records the parameters used.
> 24. Run two simulations sequentially. Verify both results are preserved in `results/` with sequential run numbering.
>
> **G. Constellation Fleet Mode (if applicable)**
> 25. Load fleet-config.html. Verify all 204 plants appear with correct metadata.
> 26. Toggle a plant off. Run simulation. Verify that plant shows $0 revenue and is excluded from dispatch.
> 27. Toggle CCS retrofit on a gas plant. Verify CCS economics appear in results.
>
> **Deliverable**: A test results report documenting pass/fail for each test, with screenshots or console output for any failures. Fix any bugs found during testing before marking tests as passed.
>
> **Files to read first**: `backend/main.py` (API endpoints), `scripts/market_simulation.py` (simulation entry points), `frontend/js/setup.js` (parameter handling), `frontend/js/results.js` (results rendering), `app-startup/start.py` (startup sequence)
> **Dependencies**: All synchronization prompts (V2-1 through V2-5) should be completed first.

---

## Category 3: UI & CSV Template QA/QC

### Prompt V2-7: Setup Page Validation + Custom CSV Compatibility Audit

> **Task**: Audit and fix the setup.html parameter input system, CSV template format compliance, custom input file loading pipeline, and responsive design across viewports.
>
> **Context**: The market simulator's setup page (`frontend/setup.html`) is the primary user interface for configuring simulations. It connects to the backend via `frontend/js/setup.js`. Users can also upload custom CSV files (hourly LMP forecasts, fuel prices, capacity prices, REC prices) from `custom-user-inputs/`. After 11 prompts of feature additions, the setup page may have stale controls, missing toggles for new features, or broken CSV loading paths.
>
> **Audit Scope**:
>
> **A. Setup Page Parameter Completeness**
> 1. Read `frontend/setup.html` and `frontend/js/setup.js` fully.
> 2. Read `backend/models.py` for `SimulationRequest` schema — every field in the schema should have a corresponding UI control on setup.html.
> 3. Cross-reference: For each new feature from Prompts 7-11, verify a UI control exists:
>    - **Prompt 7 (VRE Cannibalization)**: Is there a toggle to enable/disable cannibalization feedback? If not, should there be one, or is it always-on?
>    - **Prompt 8 (ORDC)**: Is there a scarcity pricing mode selector (ORDC vs demand-quantile)? The Implementation_Prompts.md mentions adding this to setup.html.
>    - **Prompt 9 (Synthetic Warning)**: Does the data-status indicator show on the setup page (before simulation runs)?
>    - **Prompt 10 (IPM Triggers)**: Is there a way to set trigger sensitivity (Medium/High thresholds)?
>    - **Prompt 11 (Tech Queue Caps)**: Is there a toggle for tech-differentiated vs uniform queue caps? Can users adjust per-technology caps?
> 4. For each new control: verify it sends the correct parameter name to the API, and the API receives/processes it correctly.
>
> **B. Parameter Validation**
> 5. Test input validation on setup.html:
>    - Non-numeric input in numeric fields (demand growth, carbon price) → should show error, not submit
>    - Negative values where only positive allowed → should reject
>    - Out-of-range values (e.g., demand growth > 20%) → should warn but allow (advanced user)
>    - Empty required fields → should prevent submission with clear error message
> 6. Verify parameter ranges match what the backend expects — check `SimulationRequest` validators in `models.py`.
> 7. Verify ISO selector works for all 7 ISOs — each ISO should show/hide ISO-specific controls (e.g., CAISO geothermal toggle).
>
> **C. Custom CSV Template Compliance**
> 8. Read all template files in `custom-user-inputs/`:
>    - `template_lmp_hourly.csv` — verify 8760 rows, 7 ISO columns, $/MWh units
>    - `template_capacity_prices.csv` — verify 12 rows (months), 7 ISO columns, $/MW-day units
>    - `template_rec_prices.csv` — verify 12 rows, 7 ISO columns, $/MWh units
>    - `template_fuel_prices_gas.csv`, `template_fuel_prices_coal.csv`, `template_fuel_prices_oil.csv` — verify 12 rows, 7 ISO columns, $/MMBtu units
> 9. Verify template header row matches what the backend parser expects (exact column names).
> 10. Read the CSV parsing code in `backend/main.py` or `scripts/` to find where custom CSVs are loaded. Verify:
>     - Parser handles both comma and tab delimiters
>     - Parser handles missing values gracefully (NaN → default, not crash)
>     - Parser validates row count (8760 for hourly, 12 for monthly)
>     - Parser validates column names match expected ISOs
>     - Error messages are user-friendly (not Python tracebacks)
>
> **D. CSV Upload + Override Pipeline**
> 11. Test the full CSV override flow:
>     - Copy a template, modify values, save with correct name (no `template_` prefix)
>     - Enable custom CSV toggle on setup.html
>     - Run simulation — verify custom values are used (check results against custom input)
>     - Disable toggle — verify default synthetic values are restored
> 12. Test with malformed CSV:
>     - Wrong number of rows → should error clearly
>     - Missing ISO column → should error clearly
>     - Non-numeric values → should error clearly
>     - Extra columns → should ignore gracefully
>
> **E. Responsive Design Audit**
> 13. Test setup.html at 4 viewports: 320px (small mobile), 375px (iPhone), 768px (tablet), 1440px (desktop).
> 14. Verify at each viewport:
>     - All controls are visible and tappable (44px min touch targets)
>     - No horizontal overflow or text clipping
>     - Toggle groups wrap correctly (don't overflow off-screen)
>     - ISO selector pills are tappable on mobile
>     - Form submits correctly at all sizes
>     - Charts render at readable size in results
> 15. Check `frontend/styles/shared.css` and `frontend/styles/simulator.css` for responsive breakpoints. Verify they cover all 4 viewports.
>
> **F. Documentation Cross-Check**
> 16. Read `custom-user-inputs/readme.txt`. Verify instructions match actual template format and file naming expectations.
> 17. Read the CSV section of `frontend/guide.html`. Verify it matches the actual template format and upload process.
>
> **Deliverable**: Fix any issues found. Update CSV templates if column headers have drifted. Update readme.txt and guide.html if instructions are stale. Ensure all new Prompt 7-11 controls exist and function on setup.html.
>
> **Files to read first**: `frontend/setup.html`, `frontend/js/setup.js`, `backend/models.py`, `backend/main.py` (CSV loading), `custom-user-inputs/readme.txt`, all template CSVs
> **Dependencies**: Can run in parallel with V2-6.

---

## Category 4: Documentation Audit

### Prompt V2-8: Comprehensive Documentation Update — Manual, Guide, Methodology, Specification

> **Task**: Audit and update all user-facing and developer-facing documentation to reflect the complete state of the market simulator after all 11 implementation prompts and v2 synchronization fixes.
>
> **Context**: The market simulator has undergone significant feature additions across 11 prompts. Documentation was written at various stages and may not reflect the current model capabilities, especially features from Prompts 7-11 (VRE cannibalization, ORDC scarcity pricing, synthetic data warnings, IPM triggers, tech-differentiated queue caps) and the v2 synchronization fixes (ORDC-in-zonal, zonal congestion triggers, cannibalization-ORDC interaction, flow persistence, backtest alignment).
>
> **Scope — 4 Documents to Audit and Update**:
>
> **A. USER_MANUAL.md** (`market-simulator/USER_MANUAL.md`)
> 1. Read the full user manual.
> 2. Verify all sections reflect current functionality:
>    - **Installation/Setup**: Does it mention all dependencies? Is the startup process accurate?
>    - **Parameter descriptions**: Does each user-configurable parameter have a description? Are new parameters from Prompts 7-11 documented?
>       - Scarcity pricing mode (ORDC vs demand-quantile)
>       - VRE cannibalization toggle (if user-configurable)
>       - Tech-differentiated queue caps (if user-configurable)
>       - IPM trigger sensitivity thresholds
>    - **Results interpretation**: Does it explain new result fields?
>       - Capture rates per resource (from cannibalization)
>       - IPM trigger cards and what they mean
>       - Synthetic vs physics data indicators
>       - Zonal LMP data (if exposed)
>    - **Custom CSV documentation**: Is the CSV override process accurately described?
>    - **Troubleshooting**: Does it cover common issues (port conflicts, missing dependencies, data file errors)?
> 3. Add any missing sections. Update stale descriptions. Remove references to deprecated features.
>
> **B. guide.html** (`market-simulator/frontend/guide.html`)
> 1. Read the full guide page.
> 2. This is the user-facing in-app documentation. It must be accurate, complete, and match the current UI.
> 3. Verify all 7 pages in the flow are documented: Guide → Setup → Fleet Config → Results → CCS Emissions → IPP Report → Methodology.
> 4. For each page section, verify:
>    - Screenshots or descriptions match current UI (new controls from Prompts 7-11 should be shown)
>    - Parameter explanations are accurate
>    - Chart descriptions match actual charts rendered
> 5. Add documentation for new features:
>    - **VRE Cannibalization**: What are capture rates? Why does solar revenue differ from average LMP? What does the cannibalization feedback do?
>    - **ORDC Scarcity Pricing**: What is ORDC? How does it differ from demand-quantile? Why does it matter for results?
>    - **IPM Triggers**: What are the 6 trigger types? What does Medium vs High severity mean? What should users do when triggers fire?
>    - **Synthetic Data Warnings**: What does the warning banner mean? How to get physics-optimized data (optimizer parquets)?
>    - **Tech Queue Caps**: Why are per-technology caps different? What's the basis (LBNL data)?
>    - **Methodology page**: Brief overview and link to methodology.html
> 6. Ensure explanations assume minimal energy domain knowledge (business professional audience).
> 7. Use the existing style system — `shared.css` classes, proper heading hierarchy, responsive layout.
>
> **C. methodology.html** (`market-simulator/frontend/methodology.html`)
> 1. Read the full methodology page.
> 2. This page is the honest disclosure of what the tool does and doesn't do. It must be technically accurate.
> 3. Verify/update:
>    - **Model description**: Does it accurately describe the current dispatch model (merit-order + ORDC scarcity pricing + zonal decomposition)?
>    - **Limitations list**: Are limitations still accurate? Some may have been partially addressed by Prompts 1-11:
>       - "No transmission constraints" → now has simplified zonal model (Prompt 1)
>       - "No storage co-optimization" → LP co-dispatch exists (Prompt 2)
>       - "No demand response" → DR integrated (Prompt 4)
>       - "No VRE price effects" → cannibalization feedback (Prompt 7)
>       - "Statistical LMP only" → ORDC option now available (Prompt 8)
>    - **Appropriate use cases**: Still accurate? Update if the tool's capability envelope has expanded.
>    - **What this tool is NOT**: Still accurate and important to maintain.
>    - **Confidence zones**: Does it reference the backtest validation (Prompt 5) and confidence visualization (Prompt 6)?
> 4. Add sections for:
>    - IPM trigger framework (Prompt 10) — positioning the tool within a modeling workflow
>    - Data tier explanation (Prompt 9) — what synthetic vs physics data means for result reliability
>    - Synchronization notes — how ORDC, cannibalization, zonal, and DR interact
>
> **D. Model_Methodology_Specification.md** (`market-simulator/docs/Model_Methodology_Specification.md`)
> 1. Read the full specification document.
> 2. This is the developer/reviewer-facing technical spec. It must be precise and complete.
> 3. Verify all model equations and algorithms are documented:
>    - **ORDC formula**: `price = marginal_cost + VOLL × LOLP(reserves)` where `LOLP = 1/(1 + exp(k × (reserves - target)))`
>    - **Cannibalization sigmoid**: `depression = 0.55 × sigmoid(vre_penetration - 0.6)`, with ORDC floor adjustment
>    - **Tech queue caps**: Per-technology GW/yr caps with LBNL sourcing
>    - **Zonal LP**: Objective function, constraints, zone definitions, transfer limits
>    - **DR activation**: Trigger price, participation rate, max shed, iterative re-dispatch
>    - **IPM triggers**: All 6 trigger types with Medium/High thresholds
>    - **Backtesting**: Historical comparison methodology, validation metrics
> 4. Verify all parameter tables are current — especially ORDC_PARAMS, DEMAND_RESPONSE, TECH_QUEUE_CAP_GW, ZONE_CONFIG.
> 5. Add a "Feature Interaction Matrix" showing how Prompts 1-11 interact:
>    - Which features feed into which (e.g., zonal dispatch → ORDC reserves → scarcity pricing → DR activation)
>    - Which features are independent
>    - Which features have been synchronized (v2 fixes)
> 6. Update the revision history / changelog.
>
> **Writing standards** (from CLAUDE.md):
> - Direct, confident, analytical voice — not overly formal
> - Brevity over verbosity, but adequate detail for peer review
> - Lead with the insight, then supporting evidence
> - Active voice preferred
>
> **Deliverable**: Updated versions of all 4 documents. Each should be internally consistent and cross-referenced where appropriate. The methodology.html limitations list should honestly reflect what's been fixed vs what remains.
>
> **Files to read first**: `USER_MANUAL.md`, `frontend/guide.html`, `frontend/methodology.html`, `docs/Model_Methodology_Specification.md`, `scripts/pipeline_config.py` (for current parameter values), `AUDIT.md` (for original criticisms and what's been addressed)
> **Dependencies**: Should be run last, after all v2 synchronization prompts and QA/QC testing are complete, so documentation reflects the final state.

---

## Summary — All v2 Prompts

| Prompt | Category | Severity | Effort | What It Fixes |
|--------|----------|----------|--------|---------------|
| V2-1 | Sync | Critical | 1 day | ORDC missing from zonal LMP path |
| V2-2 | Sync | Critical | 1 day | IPM triggers blind to zonal congestion |
| V2-3 | Sync | High | Half day | Cannibalization ignores ORDC floor + zonal |
| V2-4 | Sync | Low-Med | Half day | Flows discarded + DR triggers stale |
| V2-5 | Sync | Medium | Half day | Backtest using pre-7-11 model behavior |
| V2-6 | QA/QC | — | 1-2 days | Full app end-to-end testing |
| V2-7 | UI/CSV | — | 1 day | Setup page + CSV template audit |
| V2-8 | Docs | — | 1-2 days | All documentation current and complete |

**Total estimated effort**: 5-7 sessions

**Parallel execution groups**:
- **Group A** (parallel): V2-1, V2-2, V2-4
- **Group B** (sequential after A): V2-3, V2-5
- **Group C** (parallel after B): V2-6, V2-7
- **Group D** (after C): V2-8
