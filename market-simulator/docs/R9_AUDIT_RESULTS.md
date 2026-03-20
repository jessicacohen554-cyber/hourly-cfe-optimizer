# R9 QA/QC Code Audit Results — Section 2 Per-Recommendation Verification

**Date**: 2026-03-20
**Scope**: Read-only audit of all 8 recommendations (R1–R5, R7–R8, R10)
**Method**: Grep for each function, verify at stated line numbers, read and check each audit item

## Summary

| Rec | Checklist Items | Verified | Missing |
|-----|----------------|----------|---------|
| **R1** — Economics-Driven Storage Deployment | 8 | **8/8** | 0 |
| **R2** — Endogenous Wright's Law Learning Curves | 8 | **8/8** | 0 |
| **R3** — Synthetic Fallback → Explicit Warning | 6 | **6/6** | 0 |
| **R4** — VRE Basis Differential by Zone | 6 | **6/6** | 0 |
| **R5** — Confidence Intervals on Sweep Results | 7 | **7/7** | 0 |
| **R7** — Endogenous Capacity Market Prices | 8 | **8/8** | 0 |
| **R8** — Input Validation Hardening | 10 | **10/10** | 0 |
| **R10** — Curtailment Feedback on VRE Economics | 6 | **6/6** | 0 |
| **TOTAL** | **59** | **59/59** | **0** |

## Per-Recommendation Details

### R1: Economics-Driven Storage Deployment

All 6 key functions verified:
- `compute_storage_arbitrage_from_lmp` — market_simulation.py:1378 ✓
- `compute_storage_deployment` — market_simulation.py:2293 ✓
- `compute_capacity_degradation` — market_simulation.py:1337 ✓
- `co_dispatch_storage_lp` — dispatch_utils.py:724 ✓
- `_solve_storage_window_lp` — dispatch_utils.py:589 ✓
- `compute_storage_revenue_credit` — pipeline_config.py:773 ✓

Checklist:
- [x] `_STORAGE_TECH_PARAMS` defines all 4 techs (battery, battery8, ldes, h2) with duration, efficiency, cycles, window
- [x] `compute_storage_arbitrage_from_lmp()` uses rolling-window charge/discharge on hourly LMP
- [x] `compute_storage_deployment()` stacks: arbitrage + capacity + ancillary revenue
- [x] Revenue stacking uses `REVENUE_STACKING_FACTOR` (0.70) for arbitrage/ancillary hour overlap
- [x] Deploy rule: margin > 0 → deploy to `STORAGE_MAX` cap; else 0%
- [x] H2 gated to ≥95% clean thresholds (`H2_MIN_THRESHOLD = 95.0`)
- [x] Capacity price degraded by clean% via sigmoid before adding to storage revenue
- [x] `STORAGE_MAX` (V1) and `STORAGE_MAX_V2` (2050) caps defined in pipeline_config.py

### R2: Endogenous Wright's Law Learning Curves

All 4 key functions verified:
- `wright_cost` — market_simulation.py:405 ✓
- `get_effective_cumulative_gw` — market_simulation.py:427 ✓
- `get_resource_lcoe` — market_simulation.py:1684 ✓
- `compute_lcoe_snapshot` — market_simulation.py:1826 ✓

Checklist:
- [x] `ENDOGENOUS_LEARNING = True` at pipeline_config.py:1856
- [x] `WRIGHT_CUMULATIVE_GW_2025` dict has baselines for 10 techs (nuclear, ccs, ldes, h2, geothermal, battery, battery8, solar, wind, offshore_wind)
- [x] `WRIGHT_LEARNING_RATE` dict has Fast/Slow rates per tech
- [x] `wright_cost()` applies power-law: `foak × (cum_gw / ref_gw) ^ (-log₂(1 - lr))`, floored at `noak_floor`
- [x] `get_effective_cumulative_gw()` sums: 2025 baseline + model-deployed + background trajectory
- [x] Cumulative GW updated per deployment at line 2653: `cumulative_gw[tech] += deploy_gw`
- [x] When `_endogenous = False` (line 3342), cumulative_gw resets to 2025 baseline each year
- [x] `lcoe_trajectory` field exists in YearResult (models.py:479) and is populated at market_simulation.py:3872

### R3: Synthetic Fallback → Explicit Warning

All 3 key functions verified:
- `check_data_sources` — market_simulation.py:2850 ✓
- `_build_data_quality` — market_simulation.py:2911 ✓
- `_generate_synthetic_step3_data` — market_simulation.py:2947 ✓

Checklist:
- [x] `SYNTHETIC_DATA_MODE` at pipeline_config.py:40 defaults to `"warn"` via env var
- [x] Three-branch logic: `"error"` → RuntimeError, `"warn"` → logger.warning, `"silent"` → logger.info
- [x] `_generate_synthetic_step3_data()` emits `warnings.warn()` before generating data
- [x] `_build_data_quality()` returns structured metadata: `{synthetic_backed, missing_sources, mode}`
- [x] `data_quality` field exists on YearResult (models.py:479) and SimulationResponse (models.py:542)
- [x] Data quality populated in results at market_simulation.py:3850

### R4: VRE Basis Differential by Zone

All 2 key functions verified:
- `compute_energy_revenue_by_resource` — market_simulation.py:1493 ✓
- `compute_zone_revenue` — market_simulation.py:1609 ✓

Checklist:
- [x] `VRE_PRIMARY_ZONE` at pipeline_config.py:205 maps all 7 ISOs × {solar, wind, offshore_wind}
- [x] `compute_energy_revenue_by_resource()` accepts `zonal_lmp_matrix` and `zonal_zone_names` params
- [x] When zonal data available: VRE resources use zone-specific LMP
- [x] Fallback: system-average LMP when zonal data unavailable (copper-plate mode)
- [x] `compute_zone_revenue()` computes `basis_diffs[res] = zone_avg - system_avg`
- [x] `basis_differential` field on ZoneDetail (models.py:373), populated at market_simulation.py:2674

### R5: Confidence Intervals on Sweep Results

All 4 key functions verified:
- `_compute_weighted_percentiles` — market_simulation.py:3967 ✓
- `_percentile_dict` — market_simulation.py:4051 ✓
- `aggregate_sweep_percentiles` — market_simulation.py:4075 ✓
- `compute_sweep_uncertainty` — market_simulation.py:4206 ✓

Checklist:
- [x] `SCENARIO_WEIGHTS` at pipeline_config.py:2046 defines priors for 6 dimensions
- [x] Medium scenarios have highest weight (0.5–0.6), extremes lower (0.1–0.3)
- [x] `_compute_weighted_percentiles()` uses cumulative weight interpolation
- [x] `_percentile_dict()` returns both unweighted (p10–p90) and weighted (wp10–wp90) variants
- [x] `aggregate_sweep_percentiles()` groups by (ISO, year), computes bands for 15+ metrics
- [x] `UncertaintyBands` model (models.py:603) has: metric, p10–p90, mean, std, n, wp10–wp90
- [x] `SweepUncertainty` model (models.py:626) wraps: iso, year, List[UncertaintyBands]

### R7: Endogenous Capacity Market Prices

All 3 key functions verified:
- `compute_capacity_price` — pipeline_config.py:632 ✓
- `compute_capacity_degradation` — market_simulation.py:1337 ✓
- `compute_capacity_revenue` — market_simulation.py:1537 ✓

Checklist:
- [x] `CAPACITY_SCARCITY_PARAMS` at pipeline_config.py:625 defines: target 15% RM, max 3× multiplier, floor 1.0
- [x] `compute_capacity_price()` combines scarcity × clean degradation × base price
- [x] Scarcity multiplier: linear ramp from 1.0 (at target RM) to max (at 0% RM)
- [x] Clean degradation: sigmoid S-curve with per-ISO params
- [x] ERCOT base price = $0 → always returns $0
- [x] SPP base price = $0 → same behavior
- [x] `compute_capacity_revenue()` calls `compute_capacity_price()` unless overridden
- [x] Deployment loop at line 2538 uses endogenous capacity price

### R8: Input Validation Hardening

All 6 key functions/validators verified:
- `validate_iso` — models.py:202 ✓
- `validate_years_monotonic` — models.py:210 ✓
- `validate_cost_bounds` — models.py:263 ✓
- `validate_isos` (SweepRequest) — models.py:308 ✓
- `/api/validate-request` — main.py:1133 ✓
- `_compute_result_caveats` — main.py:832 ✓

Checklist:
- [x] `@field_validator('iso')` — uppercases and validates against VALID_ISOS
- [x] `@field_validator('years')` — rejects non-monotonic, out-of-range [2020, 2100]
- [x] `@field_validator('transmission_level')` — validates L/M/H/None
- [x] `@field_validator('ppa_level', 'gas_friction', 'queue_cap_level')` — validates L/M/H
- [x] `@field_validator('carbon_price')` — bounds 0–1000 $/ton
- [x] `@model_validator(mode='after')` validate_cost_bounds — LCOE ≤500, fuel 0–100, start < end
- [x] `@field_validator('isos')` on SweepRequest — validates each ISO in list
- [x] `/api/validate-request` endpoint — returns {valid, missing_data, scenario_count, caveats, data_tiers}
- [x] `result_caveats: List[str]` on SimulationResponse (models.py:544)
- [x] `_compute_result_caveats()` checks: synthetic data, zonal LMP, storage, offshore wind, geothermal, interchange, IPM

### R10: Curtailment Feedback on VRE Economics

All 2 key functions verified:
- `compute_lmp_at_threshold` — returns curtailment_rate as 10th tuple element at line 672 ✓
- `compute_market_deployment` — accepts `curtailment_rate=0.0` at line 2422 ✓

Checklist:
- [x] Curtailment rate computed at lines 617–634: `curtailed_mwh / total_vre_mwh`, clamped to [0, 1]
- [x] `curtailment_rate` returned as 10th tuple element from `compute_lmp_at_threshold()`
- [x] `compute_market_deployment()` accepts `curtailment_rate=0.0` parameter
- [x] R10 block at lines 2515–2528: effective_cf = cf × (1 - curtailment_rate), effective_lcoe = lcoe / (1 - curtailment_rate), inf guard
- [x] `effective_lcoe` (not raw lcoe) used in profit calculation downstream (line 2554)
- [x] `curtailment_rate` field on YearResult (models.py:486), populated at market_simulation.py:3859
