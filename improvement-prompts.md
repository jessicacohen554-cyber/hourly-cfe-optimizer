# Sequenced Improvement Prompts for Step 6.1 & Market Simulator

Based on the forecast validation exercise. Each prompt is self-contained for a fresh session.

---

## Prompt 1: Add Unit Commitment / Min-Gen Constraints to LMP Engine
**Estimated scope: Medium (1 session)**
**Impact: HIGH — fixes known 25% off-peak bias**

```
Read CLAUDE.md and SPEC.md first. Then read scripts/lmp_engine.py (specifically compute_hourly_lmp_vectorized() lines 722-978) and scripts/calibrate_lmp_model.py.

TASK: Add unit commitment / minimum generation constraints to the LMP engine to fix the known off-peak pricing bias (PJM off-peak $34.75 vs actual $28.00 — a 25% overshoot documented in calibration).

CONTEXT: The current LMP engine uses a merit-order dispatch stack but has NO must-run / min-gen constraints. In reality, nuclear plants have ~100% must-run, coal steam units have minimum stable generation (~40% of nameplate), and gas CCGT has min-gen (~30% of nameplate). During low-demand hours, this must-run floor creates surplus that depresses prices — our model doesn't capture this.

IMPLEMENTATION:
1. In build_merit_order_stack() (lmp_engine.py lines 208-308), add a must_run_mw field to each stack unit:
   - Nuclear: 100% of capacity (fully must-run)
   - Coal steam: 40% of capacity (min stable generation)
   - Gas CCGT: 0% (can cycle off in most ISOs)
   - Gas CT: 0% (peaker, fully dispatchable)
2. In compute_hourly_lmp_vectorized(), when residual demand is below total must-run capacity:
   - Set the marginal price to the must-run unit's marginal cost MINUS a surplus depression factor
   - This pushes off-peak prices down toward the $28/MWh target
3. Add must-run parameters to pipeline_config.py as shared constants (MUST_RUN_PCT = {'coal_steam': 0.40, 'nuclear': 1.0, ...})
4. Re-run calibrate_lmp_model.py for all 7 ISOs to verify the off-peak fix
5. The compute_hourly_lmp_vectorized() return signature should NOT change (still returns hourly_lmp, hourly_marginal_unit)

CALIBRATION TARGETS (from calibrate_lmp_model.py):
- PJM: avg $34.70, off-peak $28.00, P10 $18.00
- Current model: avg $36.69 (GOOD), off-peak $34.75 (FAIL)
- Target: off-peak within 15% of $28.00 → $23.80-$32.20

DO NOT change any upstream pipeline steps or dashboard code. Only modify lmp_engine.py, pipeline_config.py, and calibrate_lmp_model.py. Commit and push when calibration passes.
```

---

## Prompt 2: Add Scarcity Pricing Policy Feedback Mechanism
**Estimated scope: Medium (1 session)**
**Impact: HIGH — fixes unrealistic $172/MWh ERCOT 2050 projections**

```
Read CLAUDE.md and SPEC.md first. Then read scripts/step6_1_smartargets.py (specifically run_market_simulation() lines 1010-1655) and scripts/lmp_engine.py (price models, lines 315-680).

TASK: Add a policy feedback mechanism that dampens sustained high LMP at high clean penetration. Currently, our reference sweep projects ERCOT P50 LMP reaching $172/MWh by 2050 — plausible under pure ORDC energy-only pricing, but unrealistic because regulators would intervene with capacity mechanisms or price reforms before averages sustained at those levels.

CONTEXT from forecast validation exercise:
- ERCOT 2040 P50 LMP: $54/MWh (vs EIA $32, Cambium $15)
- ERCOT 2050 P50 LMP: $172/MWh (vs EIA $35, Cambium $10)
- PJM 2050 P50 LMP: $111/MWh (vs EIA $43)
- The divergence is primarily from scarcity hours at high clean penetration pulling up averages

IMPLEMENTATION:
1. In step6_1_smartargets.py, add a "regulatory feedback" mechanism in the year-by-year market simulation loop:
   - Track rolling 3-year average LMP per ISO
   - If rolling avg exceeds a threshold (e.g., 2× the 2023 baseline LMP), trigger a capacity mechanism reform:
     - For energy-only ISOs (ERCOT, SPP): introduce a capacity payment that reduces scarcity hours
     - For capacity market ISOs (PJM, NYISO, etc.): increase capacity procurement target
   - The feedback should progressively cap effective average LMP at ~1.5-2× baseline
2. Make the feedback strength a parameter in pipeline_config.py (REGULATORY_FEEDBACK_THRESHOLD_MULTIPLIER = 2.0, REGULATORY_FEEDBACK_DAMPING = 0.5) so it can be toggled
3. The feedback should NOT affect the underlying dispatch physics — only the price signals that feed back into deployment economics
4. This is conceptually similar to how real ISOs respond: ERCOT reformed ORDC after 2021 freeze, PJM reformed RPM parameters after price spikes

Add to SPEC.md under a new "Regulatory Feedback" section. Commit and push when done.
```

---

## Prompt 3: Implement Announced Coal Retirements + Transmission Constraint Toggle
**Estimated scope: Medium-Large (1 session)**
**Impact: MEDIUM — improves near-term emissions accuracy, adds useful scenario dimension**

```
Read CLAUDE.md and SPEC.md first. Then read scripts/pipeline_config.py (fossil retirement constants, lines 461-472), scripts/dispatch_utils.py (compute_fossil_retirement() lines 743-850), and scripts/lmp_engine.py (build_merit_order_stack() lines 208-308).

TASK: Two related improvements from the forecast validation exercise.

PART A — ANNOUNCED COAL RETIREMENTS:
Replace threshold-based coal retirement (current: coal exits at 70% clean) with a hybrid model:
- Near-term (through 2035): Use EIA 860 announced retirement schedule. Key datapoint: EPA projects 36.5 GW of coal retiring in 2032 alone under the 111(d) rule.
- Long-term (2035+): Use threshold-based retirement as fallback
- The transition should be smooth (no cliff)

Implementation:
1. Add ANNOUNCED_COAL_RETIREMENTS to pipeline_config.py — a dict of {iso: {year: retired_gw}} based on EIA 860 data. Key entries:
   - PJM: ~25 GW retiring by 2032 (most RFC coal)
   - MISO: ~15 GW retiring by 2032
   - ERCOT: ~5 GW by 2030
   - SPP: ~8 GW by 2030
2. Modify compute_fossil_retirement() in dispatch_utils.py to check announced retirements first, then fall back to threshold-based
3. Modify build_merit_order_stack() in lmp_engine.py to size coal capacity based on remaining-after-retirement, not just threshold-based
4. This should improve our emissions trajectory alignment with EIA AEO (currently our PJM emissions are 356 Mt at 2035 vs EIA's 200 Mt)

PART B — TRANSMISSION CONSTRAINT TOGGLE:
Princeton REPEAT shows clean penetration drops from 75% → 61% at 2030 when transmission growth is limited to 1%/yr (vs required 2.3%/yr).

Implementation:
1. Add a TRANSMISSION_CONSTRAINT parameter to pipeline_config.py: {'unconstrained': 1.0, 'limited': 0.7, 'constrained': 0.5}
   - The factor reduces effective queue throughput (models transmission bottleneck limiting interconnection)
2. Apply the factor in step6_1_smartargets.py where queue_cap_gw is used
3. Default to 'unconstrained' for backward compatibility
4. This is a lightweight way to capture transmission constraints without full network modeling

Update SPEC.md with both changes. Commit and push when done.
```

---

## Prompt 4: Add IRA Policy Overlay to Reference Sweep + LMP Decomposition
**Estimated scope: Large (1 session)**
**Impact: HIGH — makes reference case directly comparable to published forecasts**

```
Read CLAUDE.md and SPEC.md first. Then read scripts/step6_1_smartargets.py (especially the sweep scenario logic, lines 287-437 and run_market_simulation() lines 1010-1655), scripts/pipeline_config.py, and scripts/lmp_engine.py (compute_hourly_lmp_vectorized() lines 722-978).

TASK: Two improvements from the forecast validation exercise.

PART A — IRA POLICY OVERLAY FOR REFERENCE SWEEP:
Our reference sweep is market-only (no policy drivers), causing large emissions gaps vs. every published benchmark. The "conditions" dimension (Facilitating/Challenging) only controls learning rates, queue caps, and DAC — NOT policy. We need an IRA overlay.

Implementation:
1. Add a new sweep dimension or modify the price_sensitivity dimension to include IRA effects:
   - PTC/ITC: Reduce effective LCOE for solar (-$26/MWh PTC equivalent), wind (-$26/MWh), battery (-30% ITC), nuclear (-$15/MWh 45U)
   - These are already partially in our LCOE tables but may not be consistently applied
   - Check: Does pipeline_config.py LCOE tables already include IRA incentives? If yes, document. If no, add.
2. Add RGGI/carbon pricing as a cost adder to fossil generation in the sweep:
   - Apply to NYISO, NEISO, PJM (RGGI states)
   - Use the existing CO2 price dimension (Low $3, Med $5.50, High $14 per ton) but make it ISO-specific
3. Add state RPS minimums as deployment floors in the simulation loop:
   - If clean_pct at a given year is below the RPS target, force deployment to meet it
   - Key RPS targets: CA 60% by 2030/100% by 2045, NY 70% by 2030, NJ 50% by 2030
4. These policy overlays should be TOGGLEABLE (on/off) so we can still run pure market-only for comparison

PART B — LMP DECOMPOSITION (Analytics only):
Add component tracking to compute_hourly_lmp_vectorized() in lmp_engine.py:
1. Track 3 parallel arrays during computation:
   - merit_order_base[h] = base marginal cost from stack dispatch
   - scarcity_component[h] = scarcity/ORDC adder
   - demand_quantile_component[h] = congestion/depression from demand-quantile layer
2. Return these as an optional dict: components={'merit_base': array, 'scarcity': array, 'dq_adder': array}
3. Add component averages to compute_lmp_stats() output
4. This is purely for analytics/validation — no physics changes

This enables direct comparison of our merit-order component against NREL Cambium's short-run marginal cost (SRMC), which is the apples-to-apples comparison. Currently we compare our all-in LMP (with scarcity+congestion) to their SRMC, which inflates apparent divergence.

Update SPEC.md. Commit and push when done.
```

---

## Prompt 5: NYISO Import Modeling + Validation Page Update
**Estimated scope: Small-Medium (1 session)**
**Impact: LOW-MEDIUM**

```
Read CLAUDE.md and SPEC.md first. Then read scripts/step6_1_smartargets.py and dashboard/forecast_validation.html.

TASK: Two items from the forecast validation exercise.

PART A — NYISO HYDRO-QUÉBEC IMPORTS (Low effort):
NYISO imports ~25 TWh/yr of clean energy from Hydro-Québec, which our model ignores (treating ISOs independently). This partly explains the NYISO clean penetration lag (our 40% vs benchmarks' 58-62% at 2030).

Implementation (simplest approach):
1. In pipeline_config.py, add EXTERNAL_CLEAN_IMPORTS_TWH = {'NYISO': 25, ...} (other ISOs: 0 for now)
2. In step6_1_smartargets.py, when computing clean_pct for NYISO, add the external import TWh to the clean generation numerator
3. This is a fixed constant, not a modeled flow — it's a ~15 pp boost to NYISO clean_pct
4. Document as a known simplification (real imports vary with market conditions)

PART B — UPDATE FORECAST VALIDATION NARRATIVE:
After Prompts 1-4 are implemented, the validation results will have changed. Re-run scripts/validate_market_forecasts.py and update dashboard/forecast_validation.html:
1. Update the narrative text with new comparison results
2. Update the stat counters (alignment %, deviations)
3. Add a "Model Improvements" section describing what changed since initial validation
4. Fix the demand growth reference in the narrative — we say "1.2%/yr Medium" but our model actually uses ISO-specific rates (ERCOT 3.5%, PJM 2.4%, etc. from pipeline_config.py:1052-1060). Update the text to reflect this.

Commit and push when done.
```

---

## Execution Order

| Order | Prompt | Dependency | Impact |
|-------|--------|-----------|--------|
| 1 | Unit commitment / min-gen | None | HIGH |
| 2 | Scarcity feedback | None (better after #1) | HIGH |
| 3 | Coal retirements + transmission | None | MEDIUM |
| 4 | IRA policy overlay + LMP decomp | Better after #1 and #2 | HIGH |
| 5 | NYISO imports + validation update | After #1-4 complete | LOW-MED |

Prompts 1-3 can run in parallel if you have multiple sessions. Prompt 4 benefits from #1 and #2 being done first. Prompt 5 is a cleanup pass after everything else.
