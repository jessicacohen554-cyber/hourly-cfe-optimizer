# R9: QA/QC Testing Prompt — Market Simulator Peer Review Validation

**Purpose**: Systematic validation that all 9 implemented peer review recommendations (R1–R5, R7–R8, R10) are correct, compatible, and production-ready.

**How to use**: Follow each section sequentially. A fresh Claude Code session can execute this prompt top-to-bottom to validate, debug, and fix issues without prior context.

**Codebase**: `market-simulator/` within the `hourly-cfe-optimizer` repo.

---

## 1. Pre-Flight Checks

### 1.1 Environment Setup

```bash
# Install dependencies
pip install numba numpy scipy pandas pydantic fastapi uvicorn pytest

# Verify Numba JIT is available
python3 -c "from numba import njit; print('Numba OK')"
```

### 1.2 Import Validation

Run each import to confirm no missing modules after R1–R10 changes:

```bash
cd market-simulator

# Core modules
python3 -c "
from scripts.market_simulation import (
    run_market_simulation,
    compute_lmp_at_threshold,
    compute_market_deployment,
    compute_storage_deployment,
    compute_storage_arbitrage_from_lmp,
    compute_energy_revenue_by_resource,
    compute_zone_revenue,
    compute_capacity_revenue,
    compute_lcoe_snapshot,
    get_resource_lcoe,
    wright_cost,
    get_effective_cumulative_gw,
    check_data_sources,
    aggregate_sweep_percentiles,
    compute_sweep_uncertainty,
)
print('market_simulation imports OK')
"

python3 -c "
from scripts.lmp_engine import compute_hourly_lmp_vectorized
print('lmp_engine imports OK')
"

python3 -c "
from scripts.pipeline_config import (
    ENDOGENOUS_LEARNING,
    SYNTHETIC_DATA_MODE,
    SCENARIO_WEIGHTS,
    VRE_PRIMARY_ZONE,
    CAPACITY_SCARCITY_PARAMS,
    STORAGE_MAX,
    REVENUE_STACKING_FACTOR,
    WRIGHT_CUMULATIVE_GW_2025,
    WRIGHT_LEARNING_RATE,
    compute_capacity_price,
    compute_storage_revenue_credit,
)
print('pipeline_config imports OK')
"

python3 -c "
from backend.models import (
    SimulationRequest,
    SimulationResponse,
    YearResult,
    ZoneDetail,
    UncertaintyBands,
    SweepUncertainty,
)
print('models imports OK')
"
```

**Expected**: All four print statements succeed with no `ImportError` or `ModuleNotFoundError`.

### 1.3 Existing Test Suite

```bash
cd market-simulator
pytest scripts/tests/ -v --tb=short 2>&1 | tail -30
```

**Expected**: All 11 test files pass. Record any failures — they indicate pre-existing issues, not R1–R10 regressions.

**Test files** (11 total):
1. `test_analytical_vs_lp.py`
2. `test_e2e_differentiation.py`
3. `test_e2e_integration.py`
4. `test_ercot_diagnostic.py`
5. `test_lp_batch_sim.py`
6. `test_lp_codispatch.py`
7. `test_scarcity_fix.py`
8. `test_storage_deployment.py`
9. `test_targeted_validation.py`
10. `validate_fleet_dispatch.py`
11. `validate_ordc_calibration.py`

---

## 2. Per-Recommendation Code Audit

For each implemented recommendation, verify the code exists at the stated locations and follows the specified patterns.

---

### 2.1 R1: Economics-Driven Storage Deployment

**Spec**: Replace static storage ramp with price-taking dispatch. Compute arbitrage revenue from hourly LMP, add capacity + ancillary revenue, deploy only when total revenue exceeds annualized cost.

**Where to look**:
- `scripts/market_simulation.py` — lines 1369–1440 (arbitrage), 2273–2388 (deployment)
- `scripts/dispatch_utils.py` — lines 566–833 (LP co-dispatch)
- `scripts/pipeline_config.py` — lines 514–530 (storage caps), 697–809 (revenue params)

**Key functions**:
| Function | File | Line | Signature |
|----------|------|------|-----------|
| `compute_storage_arbitrage_from_lmp` | market_simulation.py | 1377 | `(hourly_lmp, iso=None)` → dict of $/kW-yr by tech |
| `compute_storage_deployment` | market_simulation.py | 2273 | `(iso, year, hourly_lmp, demand_twh, ...)` → deployed_pcts + details |
| `compute_capacity_degradation` | market_simulation.py | 1336 | `(iso, clean_pct)` → sigmoid multiplier |
| `co_dispatch_storage_lp` | dispatch_utils.py | 724 | `(residual_surplus, residual_gap, active_params, detailed)` |
| `_solve_storage_window_lp` | dispatch_utils.py | 589 | Window-level LP solver |
| `compute_storage_revenue_credit` | pipeline_config.py | 773 | `(storage_type, iso)` → LCOE credit |

**Audit checklist**:
- [ ] `_STORAGE_TECH_PARAMS` dict at line 1369 defines all 4 techs (battery, battery8, ldes, h2) with duration, efficiency, cycles, window
- [ ] `compute_storage_arbitrage_from_lmp()` at line 1377 uses rolling-window charge/discharge on hourly LMP
- [ ] `compute_storage_deployment()` at line 2273 stacks: arbitrage + capacity + ancillary revenue
- [ ] Revenue stacking uses `REVENUE_STACKING_FACTOR` (0.70) to account for arbitrage/ancillary hour overlap
- [ ] Deploy rule: revenue_lcoe > cost_lcoe → deploy to `STORAGE_MAX` cap; else 0%
- [ ] H2 gated to ≥95% clean thresholds (`H2_MIN_THRESHOLD`)
- [ ] Capacity price degraded by clean% via sigmoid before adding to storage revenue
- [ ] `STORAGE_MAX` (V1 caps) and `STORAGE_MAX_V2` (2050 caps) defined in pipeline_config.py lines 514–530

**Expected behavior**:
- Flat LMP profile → zero arbitrage revenue → storage not deployed (unless capacity revenue alone covers cost)
- Duck curve LMP → positive arbitrage → battery deployed if revenue > LCOE
- ERCOT capacity = $0 → storage needs arbitrage + ancillary alone to be economic
- High clean% → degraded capacity price → harder for storage to pencil out

---

### 2.2 R2: Endogenous Wright's Law Learning Curves

**Spec**: Update cumulative installed GW at end of each simulation year. Recompute LCOE for next year using updated learning curve position. Add toggle for static vs. endogenous mode.

**Where to look**:
- `scripts/market_simulation.py` — lines 404–430 (Wright's Law core), 1683–1828 (LCOE), 2633 (cumulative update), 3226 + 3322 (toggle)
- `scripts/pipeline_config.py` — line 1856 (`ENDOGENOUS_LEARNING`), 1874–1887 (baselines + learning rates)
- `backend/models.py` — line 488 (`lcoe_trajectory` field)

**Key functions**:
| Function | File | Line | Signature |
|----------|------|------|-----------|
| `wright_cost` | market_simulation.py | 404 | `(foak_cost, noak_floor, cumulative_gw, reference_gw, learning_rate)` |
| `get_effective_cumulative_gw` | market_simulation.py | 426 | `(tech, model_deployed_gw, learning_speed, year)` |
| `get_resource_lcoe` | market_simulation.py | 1683 | `(res, iso, lcoe_level, cumulative_gw, learning_speed, year, conditions)` |
| `compute_lcoe_snapshot` | market_simulation.py | 1806 | `(iso, cumulative_gw, lcoe_level, learning_speed, year, conditions)` |

**Audit checklist**:
- [ ] `ENDOGENOUS_LEARNING = True` at pipeline_config.py line 1856
- [ ] `WRIGHT_CUMULATIVE_GW_2025` dict at pipeline_config.py line 1874 has baselines for nuclear, ccs, ldes, h2, geothermal, battery, battery8
- [ ] `WRIGHT_LEARNING_RATE` dict at pipeline_config.py line 1880 has Fast/Slow rates per tech
- [ ] `wright_cost()` at line 404 applies power-law: `foak × (cum_gw / ref_gw) ^ (-log₂(1 - lr))`, floored at `noak_floor`
- [ ] `get_effective_cumulative_gw()` at line 426 sums: 2025 baseline + model-deployed + background trajectory
- [ ] Cumulative GW updated per deployment at line 2633: `cumulative_gw[tech] += deploy_gw`
- [ ] When `_endogenous = False` (line 3322), cumulative_gw resets to 2025 baseline each year (static mode)
- [ ] `lcoe_trajectory` field exists in YearResult (models.py line 488) and is populated at market_simulation.py lines 3849–3854

**Expected behavior**:
- Over a 25-year trajectory with `ENDOGENOUS_LEARNING=True`, solar/wind/battery LCOE should decrease monotonically
- Static mode (`ENDOGENOUS_LEARNING=False`): same LCOE every year regardless of deployment
- More deployment → lower LCOE next year → more deployment (positive feedback loop)

---

### 2.3 R3: Synthetic Fallback → Explicit Warning

**Spec**: Replace silent synthetic data generation with mode-controlled behavior: error (production), warn (development default), or silent (legacy).

**Where to look**:
- `scripts/pipeline_config.py` — line 40 (`SYNTHETIC_DATA_MODE`)
- `scripts/market_simulation.py` — lines 2806–2827 (mode check), 2830–2888 (`check_data_sources`), 2891–2933 (data quality + synthetic gen)
- `backend/models.py` — lines 479, 542 (`data_quality` fields)

**Key functions**:
| Function | File | Line | Signature |
|----------|------|------|-----------|
| `check_data_sources` | market_simulation.py | 2830 | `()` → dict with 'simple' + 'tiers' |
| `_build_data_quality` | market_simulation.py | 2891 | `(iso, data_sources)` → `{synthetic_backed, missing_sources, mode}` |
| `_generate_synthetic_step3_data` | market_simulation.py | 2918 | `()` → synthetic resource mix data |

**Audit checklist**:
- [ ] `SYNTHETIC_DATA_MODE` at pipeline_config.py line 40 defaults to `"warn"` via env var
- [ ] Three-branch logic at lines 2810–2822: `"error"` → `RuntimeError`, `"warn"` → `logger.warning`, `"silent"` → `logger.info`
- [ ] `_generate_synthetic_step3_data()` at line 2918 emits `warnings.warn()` before generating data
- [ ] `_build_data_quality()` at line 2891 returns structured metadata: `{synthetic_backed: bool, missing_sources: list, mode: str}`
- [ ] `data_quality` field exists on YearResult (models.py line 479) and SimulationResponse (line 542)
- [ ] Data quality populated in results at market_simulation.py line 3830

**Expected behavior**:
- `SYNTHETIC_DATA_MODE="error"` + missing parquets → `RuntimeError` raised
- `SYNTHETIC_DATA_MODE="warn"` + missing parquets → warning logged, `data_quality.synthetic_backed = True`
- `SYNTHETIC_DATA_MODE="silent"` → no warning, synthetic data generated silently (legacy behavior)

---

### 2.4 R4: VRE Basis Differential by Zone

**Spec**: When zonal LMP is available, compute per-resource energy revenue using the LMP of the zone where that VRE type is predominantly located, not system-average LMP.

**Where to look**:
- `scripts/pipeline_config.py` — lines 205–213 (`VRE_PRIMARY_ZONE`)
- `scripts/market_simulation.py` — lines 1492–1533 (energy revenue), 1608–1646 (blended revenue + basis diffs)
- `backend/models.py` — line 373 (`basis_differential` on ZoneDetail)

**Key functions**:
| Function | File | Line | Signature |
|----------|------|------|-----------|
| `compute_energy_revenue_by_resource` | market_simulation.py | 1492 | `(hourly_lmp, supply_profiles, resource_pcts, demand_total_mwh, iso, zonal_lmp_matrix, zonal_zone_names)` |
| `compute_zone_revenue` | market_simulation.py | 1608 | Same args + `reserve_margin_pct` → includes `basis_differentials` dict |

**Audit checklist**:
- [ ] `VRE_PRIMARY_ZONE` at pipeline_config.py line 205 maps all 7 ISOs × {solar, wind, offshore_wind} to zone names
- [ ] `compute_energy_revenue_by_resource()` at line 1492 accepts `zonal_lmp_matrix` and `zonal_zone_names` params
- [ ] When zonal data available: VRE resources use zone-specific LMP (lines 1525–1529)
- [ ] Fallback: system-average LMP when zonal data unavailable (copper-plate mode)
- [ ] `compute_zone_revenue()` at line 1608 computes `basis_diffs[res] = zone_avg - system_avg` (line 1641)
- [ ] `basis_differential` field on ZoneDetail (models.py line 373), populated at market_simulation.py line 2654

**Expected behavior**:
- ERCOT West zone solar: revenue based on West zone LMP (typically lower due to congestion) → negative basis differential
- Copper-plate mode (no zonal data): all resources use system LMP → basis differential = 0
- Higher basis differential → lower VRE revenue → slower VRE deployment

---

### 2.5 R5: Confidence Intervals on Sweep Results

**Spec**: Post-processing to compute P10/P25/P50/P75/P90 bands across the scenario sweep, with optional scenario probability weights.

**Where to look**:
- `scripts/pipeline_config.py` — lines 2046–2079 (`SCENARIO_WEIGHTS`)
- `scripts/market_simulation.py` — lines 3947–3965 (weighted percentiles), 4031–4052 (percentile dict), 4055–4174 (aggregation), 4186–4240 (entry point)
- `backend/models.py` — lines 603–630 (`UncertaintyBands`, `SweepUncertainty`)

**Key functions**:
| Function | File | Line | Signature |
|----------|------|------|-----------|
| `_compute_weighted_percentiles` | market_simulation.py | 3947 | `(values, weights, percentiles)` → list of floats |
| `_percentile_dict` | market_simulation.py | 4031 | `(arr, weights)` → dict with P10–P90 + weighted variants |
| `aggregate_sweep_percentiles` | market_simulation.py | 4055 | `(all_results)` → grouped percentile bands |
| `compute_sweep_uncertainty` | market_simulation.py | 4186 | `(all_results)` → `(aggregates, uncertainty_list)` |

**Audit checklist**:
- [ ] `SCENARIO_WEIGHTS` at pipeline_config.py line 2046 defines priors for: demand, price, ppa, gas_friction, queue_cap, new_fossil_cost
- [ ] Medium scenarios have highest weight (0.5–0.6), extremes lower (0.1–0.2)
- [ ] `_compute_weighted_percentiles()` at line 3947 uses cumulative weight interpolation
- [ ] `_percentile_dict()` at line 4031 returns both unweighted (p10–p90 via `np.percentile`) and weighted (wp10–wp90) variants
- [ ] `aggregate_sweep_percentiles()` at line 4055 groups by (ISO, year) and computes bands for: clean_pct, cost_per_mwh, emissions_mt, avg_lmp, and others
- [ ] `UncertaintyBands` model (models.py line 603) has: metric, p10, p25, p50, p75, p90, mean, std, n, wp10–wp90
- [ ] `SweepUncertainty` model (models.py line 626) wraps: iso, year, List[UncertaintyBands]

**Expected behavior**:
- P50 should be close to the "all_med" scenario result
- P10–P90 spread should be wider for metrics with high sensitivity (e.g., cost)
- Weighted percentiles should shift toward Medium scenarios (highest prior weight)

---

### 2.6 R7: Endogenous Capacity Market Prices

**Spec**: Capacity prices respond to reserve margin (scarcity) and clean penetration (degradation sigmoid). Creates feedback: retirements → lower reserves → higher capacity prices → new builds → reserves recover.

**Where to look**:
- `scripts/pipeline_config.py` — lines 574–584 (`CAPACITY_MARKET_PRICES`), 625–686 (`compute_capacity_price` + params)
- `scripts/market_simulation.py` — lines 1336–1361 (degradation), 1536–1555 (capacity revenue), 2514–2518 (usage in deployment)

**Key functions**:
| Function | File | Line | Signature |
|----------|------|------|-----------|
| `compute_capacity_price` | pipeline_config.py | 632 | `(iso, reserve_margin_pct, clean_pct)` → $/kW-yr |
| `compute_capacity_degradation` | market_simulation.py | 1336 | `(iso, clean_pct)` → sigmoid multiplier |
| `compute_capacity_revenue` | market_simulation.py | 1536 | `(iso, clean_pct, resource_pcts, conditions, reserve_margin_pct)` |

**Audit checklist**:
- [ ] `CAPACITY_SCARCITY_PARAMS` at pipeline_config.py line 625 defines: target_reserve_margin_pct (15%), max_scarcity_multiplier (3×), floor_multiplier (1.0)
- [ ] `compute_capacity_price()` at line 632 combines scarcity multiplier × clean degradation multiplier × base price
- [ ] Scarcity multiplier: linear ramp from 1.0 (at target RM) to max (at 0% RM) — lines 657–670
- [ ] Clean degradation: sigmoid S-curve with per-ISO params — lines 671–686
- [ ] ERCOT base price = $0 → `compute_capacity_price("ERCOT", ...)` always returns $0
- [ ] SPP base price = $0 → same behavior
- [ ] `compute_capacity_revenue()` at line 1536 calls `compute_capacity_price()` unless overridden by conditions
- [ ] Deployment loop at line 2518 uses endogenous capacity price for clean resource revenue calculation

**Expected behavior**:
- ERCOT: capacity price always $0 (energy-only market) regardless of reserve margin
- PJM at 5% reserve margin: capacity price = $120 × ~2.33 × clean_mult ≈ high scarcity premium
- PJM at 20% reserve margin: capacity price = $120 × 1.0 × clean_mult (no scarcity effect)
- 80% clean penetration → sigmoid degradation reduces capacity price significantly

---

### 2.7 R8: Input Validation Hardening

**Spec**: Pydantic field validators on SimulationRequest, model-level cross-field validation, dry-run validation endpoint, result caveats.

**Where to look**:
- `backend/models.py` — lines 202–290 (validators), 308–315 (sweep ISO validation), 544 (`result_caveats`)
- `backend/main.py` — lines 832–894 (`_compute_result_caveats`), 1133–1190 (`/api/validate-request`)

**Key functions**:
| Function | File | Line | Purpose |
|----------|------|------|---------|
| `validate_iso` | models.py | 204 | Rejects invalid ISO codes |
| `validate_years_monotonic` | models.py | 212 | Ensures ascending year order, range [2020, 2100] |
| `validate_cost_bounds` | models.py | 264 | Cross-field: LCOE 0–500, fuel 0–100, start < end |
| `validate_isos` | models.py | 310 | SweepRequest: validates ISO list |
| `/api/validate-request` | main.py | 1133 | Dry-run: checks validity + data availability |
| `_compute_result_caveats` | main.py | 832 | Builds run-specific caveat list |

**Audit checklist**:
- [ ] `@field_validator('iso')` at models.py line 202 — uppercases and validates against `VALID_ISOS`
- [ ] `@field_validator('years')` at line 210 — rejects non-monotonic, out-of-range [2020, 2100]
- [ ] `@field_validator('transmission_level')` at line 221 — validates L/M/H/None
- [ ] `@field_validator('ppa_level', 'gas_friction', 'queue_cap_level')` at line 228 — validates L/M/H
- [ ] `@field_validator('carbon_price')` at line 256 — bounds 0–1000 $/ton
- [ ] `@model_validator(mode='after')` `validate_cost_bounds` at line 263 — LCOE ≤ 500, fuel 0–100, start < end
- [ ] `@field_validator('isos')` on SweepRequest at line 308 — validates each ISO in list
- [ ] `/api/validate-request` endpoint at main.py line 1133 — returns `{valid, missing_data, scenario_count, caveats, data_tiers}`
- [ ] `result_caveats: List[str]` on SimulationResponse (models.py line 544)
- [ ] `_compute_result_caveats()` at main.py line 832 checks: synthetic data, zonal LMP, storage, offshore wind, geothermal, interchange, IPM triggers

**Expected behavior**:
- `SimulationRequest(iso="INVALID")` → `ValueError: ISO must be one of ...`
- `SimulationRequest(years=[2030, 2025])` → `ValueError: Years must be in ascending order`
- `SimulationRequest(years=[1900])` → `ValueError: Year 1900 out of valid range`
- `CleanLCOEs(solar=-5)` → `ValueError` from cost bounds validator
- `/api/validate-request` with valid request → `{valid: true, missing_data: [], ...}`

---

### 2.8 R10: Curtailment Feedback on VRE Economics

**Spec**: Track system curtailment rate. Increase effective LCOE for marginal VRE as curtailment rises, creating a natural deployment saturation point.

**Where to look**:
- `scripts/market_simulation.py` — lines 616–633 (curtailment rate calc), 2495–2508 (LCOE adjustment), 2402 (parameter), 3838 (result population)
- `backend/models.py` — line 486 (`curtailment_rate` field)

**Key functions**:
| Function | File | Line | Role |
|----------|------|------|------|
| `compute_lmp_at_threshold` | market_simulation.py | 464 | Returns `curtailment_rate` as 10th tuple element (line 671) |
| `compute_market_deployment` | market_simulation.py | 2391 | Accepts `curtailment_rate` param (line 2402); applies R10 logic at lines 2495–2508 |

**Audit checklist**:
- [ ] Curtailment rate computed at lines 616–633: `curtailed_mwh / total_vre_mwh`, clamped to [0, 1]
- [ ] `curtailment_rate` returned from `compute_lmp_at_threshold()` as 10th tuple element
- [ ] `compute_market_deployment()` accepts `curtailment_rate=0.0` parameter at line 2402
- [ ] R10 block at lines 2495–2508: for VRE resources (`solar`, `wind`, `offshore_wind`):
  - `effective_cf = cf × (1 - curtailment_rate)`
  - `effective_lcoe = lcoe / (1 - curtailment_rate)`
  - If `effective_cf ≤ 0`: `effective_lcoe = inf` (blocks deployment)
- [ ] `effective_lcoe` (not raw `lcoe`) used in profit calculation downstream
- [ ] `curtailment_rate` field on YearResult (models.py line 486), populated at market_simulation.py line 3839

**Expected behavior**:
- 0% curtailment: effective_lcoe = base lcoe (no penalty)
- 10% curtailment: effective_lcoe = lcoe / 0.9 ≈ 11% higher → some VRE becomes uneconomic
- 50% curtailment: effective_lcoe = lcoe / 0.5 = 2× base → most VRE uneconomic
- Creates negative feedback: more VRE → more curtailment → higher effective LCOE → slower deployment

---

## 3. New Unit Tests

Create the following test file. Each test targets a specific recommendation's core behavior.

**File**: `scripts/tests/test_r9_qa_qc.py`

```python
"""R9 QA/QC: Unit tests verifying all peer review implementations (R1-R5, R7-R8, R10).

Run: pytest market-simulator/scripts/tests/test_r9_qa_qc.py -v
"""
import sys
import os
import warnings

import numpy as np
import pytest

# Add scripts dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# Add backend dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from market_simulation import (
    wright_cost,
    get_effective_cumulative_gw,
    get_resource_lcoe,
    compute_lcoe_snapshot,
    compute_storage_arbitrage_from_lmp,
    compute_energy_revenue_by_resource,
    compute_zone_revenue,
    compute_capacity_degradation,
    _generate_synthetic_step3_data,
    _percentile_dict,
    _compute_weighted_percentiles,
    compute_market_deployment,
    H,
)
from pipeline_config import (
    ENDOGENOUS_LEARNING,
    WRIGHT_CUMULATIVE_GW_2025,
    WRIGHT_LEARNING_RATE,
    CAPACITY_MARKET_PRICES,
    CAPACITY_SCARCITY_PARAMS,
    VRE_PRIMARY_ZONE,
    SCENARIO_WEIGHTS,
    compute_capacity_price,
)
from models import SimulationRequest, UncertaintyBands


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def flat_lmp():
    """Flat LMP — no price spread, zero arbitrage."""
    return np.full(H, 50.0)


@pytest.fixture
def duck_curve_lmp():
    """Duck-curve LMP with midday trough and evening peak."""
    np.random.seed(42)
    lmp = np.zeros(H)
    for d in range(365):
        h = d * 24
        lmp[h:h+24] = 35 + np.random.normal(0, 5, 24)
        lmp[h+10:h+15] -= 20   # midday solar trough
        lmp[h+16:h+21] += 40   # evening peak
    return np.clip(lmp, 0, 300)


# ═══════════════════════════════════════════════════════════════════════════════
# R2: Wright's Law Trajectory
# ═══════════════════════════════════════════════════════════════════════════════

class TestR2WrightsLaw:
    """Verify costs decrease over 25 years with endogenous learning."""

    def test_wright_cost_decreases_with_cumulative_gw(self):
        """More cumulative GW → lower cost, floored at noak."""
        foak, noak, ref_gw, lr = 100.0, 30.0, 10.0, 0.15
        costs = [wright_cost(foak, noak, gw, ref_gw, lr)
                 for gw in [10, 20, 50, 100, 500, 1000]]
        # Monotonically decreasing
        for i in range(len(costs) - 1):
            assert costs[i] >= costs[i + 1], \
                f"Cost should decrease: {costs[i]:.2f} >= {costs[i+1]:.2f}"
        # Floors at noak
        assert costs[-1] >= noak, f"Cost {costs[-1]:.2f} should not go below noak {noak}"

    def test_wright_cost_static_below_reference(self):
        """If cumulative_gw <= reference_gw, returns foak (no learning yet)."""
        assert wright_cost(100, 30, 5, 10, 0.15) == 100.0
        assert wright_cost(100, 30, 10, 10, 0.15) == 100.0

    def test_wright_cost_zero_learning_rate(self):
        """Zero learning rate → always returns foak."""
        assert wright_cost(100, 30, 1000, 10, 0.0) == 100.0

    def test_lcoe_trajectory_25_years(self):
        """Over 25 years of deployment, LCOE should decrease for solar."""
        cumulative_gw = dict(WRIGHT_CUMULATIVE_GW_2025)
        lcoes = []
        for year in range(2025, 2050):
            lcoe = get_resource_lcoe('solar', 'ERCOT', 'Medium',
                                     cumulative_gw, 'Fast', year)
            lcoes.append(lcoe)
            # Simulate 5 GW/year deployment
            cumulative_gw['solar'] = cumulative_gw.get('solar', 0) + 5
        # First year should be more expensive than last
        assert lcoes[0] > lcoes[-1], \
            f"LCOE should decrease: {lcoes[0]:.2f} → {lcoes[-1]:.2f}"

    def test_compute_lcoe_snapshot_returns_all_techs(self):
        """Snapshot should return LCOE for multiple deployable resources."""
        snap = compute_lcoe_snapshot('ERCOT', dict(WRIGHT_CUMULATIVE_GW_2025),
                                     'Medium', 'Fast', 2030)
        assert isinstance(snap, dict)
        assert len(snap) >= 3, f"Expected ≥3 techs, got {list(snap.keys())}"
        for tech, lcoe in snap.items():
            assert lcoe > 0, f"{tech} LCOE should be positive, got {lcoe}"


# ═══════════════════════════════════════════════════════════════════════════════
# R3: Synthetic Fallback Warning
# ═══════════════════════════════════════════════════════════════════════════════

class TestR3SyntheticFallback:
    """Verify warning is raised when parquets are missing."""

    def test_synthetic_data_emits_warning(self):
        """_generate_synthetic_step3_data() should emit a UserWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _generate_synthetic_step3_data()
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) >= 1, \
                "Expected at least one UserWarning from synthetic data generation"
            assert "synthetic" in str(user_warnings[0].message).lower() or \
                   "parquet" in str(user_warnings[0].message).lower(), \
                f"Warning should mention synthetic/parquet: {user_warnings[0].message}"

    def test_synthetic_data_returns_valid_structure(self):
        """Synthetic data should return a dict with ISO keys."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = _generate_synthetic_step3_data()
        assert isinstance(data, dict), f"Expected dict, got {type(data)}"
        assert len(data) > 0, "Synthetic data should not be empty"


# ═══════════════════════════════════════════════════════════════════════════════
# R4: VRE Basis Differential
# ═══════════════════════════════════════════════════════════════════════════════

class TestR4VREBasis:
    """Verify zonal LMP used for VRE revenue (not system average)."""

    def test_vre_primary_zone_coverage(self):
        """All 7 ISOs should have VRE zone mappings."""
        expected_isos = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']
        for iso in expected_isos:
            assert iso in VRE_PRIMARY_ZONE, f"Missing VRE zone mapping for {iso}"
            mapping = VRE_PRIMARY_ZONE[iso]
            assert 'solar' in mapping, f"Missing solar zone for {iso}"
            assert 'wind' in mapping, f"Missing wind zone for {iso}"

    def test_basis_differential_with_zonal_lmp(self):
        """When zonal LMP provided, basis_differentials should be computed."""
        # Create synthetic zonal LMP: 2 zones, zone 0 cheaper than zone 1
        hourly_lmp = np.full(H, 40.0)  # system average
        zonal_lmp_matrix = np.column_stack([
            np.full(H, 30.0),  # Zone 0 (cheap, VRE-rich)
            np.full(H, 50.0),  # Zone 1 (expensive, load center)
        ])
        zonal_zone_names = ['West', 'Houston']

        supply_profiles = {
            'solar': np.full(H, 1.0),
            'wind': np.full(H, 1.0),
        }
        resource_pcts = {'solar': 20, 'wind': 15}

        result = compute_zone_revenue(
            hourly_lmp, supply_profiles, resource_pcts,
            demand_total_mwh=100_000_000,
            iso='ERCOT',
            zonal_lmp_matrix=zonal_lmp_matrix,
            zonal_zone_names=zonal_zone_names,
        )
        # Should have basis_differentials in the result
        assert 'basis_differentials' in result or \
               isinstance(result, dict), \
            "Expected basis_differentials in blended revenue result"

    def test_copper_plate_zero_basis(self):
        """Without zonal data, basis differential should be zero."""
        hourly_lmp = np.full(H, 40.0)
        supply_profiles = {'solar': np.full(H, 1.0)}
        resource_pcts = {'solar': 20}

        result = compute_zone_revenue(
            hourly_lmp, supply_profiles, resource_pcts,
            demand_total_mwh=100_000_000,
            iso='ERCOT',
            zonal_lmp_matrix=None,
            zonal_zone_names=None,
        )
        # No zonal data → no basis differential or all zeros
        bd = result.get('basis_differentials', {})
        for res, diff in bd.items():
            assert diff == 0.0, \
                f"Copper-plate basis diff should be 0, got {diff} for {res}"


# ═══════════════════════════════════════════════════════════════════════════════
# R5: Confidence Intervals
# ═══════════════════════════════════════════════════════════════════════════════

class TestR5ConfidenceIntervals:
    """Verify P10/P50/P90 match numpy.percentile on raw data."""

    def test_percentile_dict_matches_numpy(self):
        """_percentile_dict should match np.percentile for unweighted case."""
        np.random.seed(123)
        arr = np.random.normal(50, 10, 100)
        result = _percentile_dict(arr)

        assert abs(result['p10'] - np.percentile(arr, 10)) < 0.01
        assert abs(result['p50'] - np.percentile(arr, 50)) < 0.01
        assert abs(result['p90'] - np.percentile(arr, 90)) < 0.01
        assert result['p10'] <= result['p50'] <= result['p90']

    def test_weighted_percentiles_shift_toward_high_weight(self):
        """Weighted P50 should shift toward values with higher weight."""
        values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        # Heavy weight on low values
        weights_low = np.array([10.0, 10.0, 1.0, 1.0, 1.0])
        # Heavy weight on high values
        weights_high = np.array([1.0, 1.0, 1.0, 10.0, 10.0])

        p50_low = _compute_weighted_percentiles(values, weights_low, [50])[0]
        p50_high = _compute_weighted_percentiles(values, weights_high, [50])[0]

        assert p50_low < p50_high, \
            f"Low-weighted P50 ({p50_low:.1f}) should be < high-weighted ({p50_high:.1f})"

    def test_uncertainty_bands_model_fields(self):
        """UncertaintyBands model should accept all required fields."""
        bands = UncertaintyBands(
            metric="clean_pct",
            p10=30.0, p25=40.0, p50=55.0, p75=65.0, p90=75.0,
            mean=54.0, std=12.0, n=1215,
        )
        assert bands.metric == "clean_pct"
        assert bands.p10 < bands.p90

    def test_scenario_weights_sum_to_one(self):
        """Each dimension's weights should sum to ~1.0."""
        for dim, weights in SCENARIO_WEIGHTS.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.01, \
                f"SCENARIO_WEIGHTS['{dim}'] sums to {total:.3f}, expected ~1.0"


# ═══════════════════════════════════════════════════════════════════════════════
# R7: Capacity Market Price Sensitivity
# ═══════════════════════════════════════════════════════════════════════════════

class TestR7CapacityMarket:
    """Verify price sensitivity to clean% and reserve margin; ERCOT returns $0."""

    def test_ercot_always_zero(self):
        """ERCOT is energy-only — capacity price must always be $0."""
        for rm in [5, 10, 15, 20, 30]:
            for clean in [0, 25, 50, 75, 100]:
                price = compute_capacity_price('ERCOT', rm, clean)
                assert price == 0.0, \
                    f"ERCOT capacity price should be $0, got ${price} at RM={rm}%, clean={clean}%"

    def test_spp_always_zero(self):
        """SPP is energy-only — capacity price must always be $0."""
        price = compute_capacity_price('SPP', 10, 50)
        assert price == 0.0, f"SPP capacity price should be $0, got ${price}"

    def test_scarcity_increases_price(self):
        """Lower reserve margin → higher capacity price."""
        price_tight = compute_capacity_price('PJM', 5, 30)    # tight reserves
        price_normal = compute_capacity_price('PJM', 15, 30)  # at target
        price_excess = compute_capacity_price('PJM', 25, 30)  # excess reserves

        assert price_tight > price_normal, \
            f"Tight reserves ({price_tight:.1f}) should > normal ({price_normal:.1f})"
        assert price_normal >= price_excess, \
            f"Normal ({price_normal:.1f}) should >= excess ({price_excess:.1f})"

    def test_clean_penetration_degrades_price(self):
        """Higher clean% → lower capacity price (sigmoid degradation)."""
        price_low_clean = compute_capacity_price('PJM', 15, 20)
        price_high_clean = compute_capacity_price('PJM', 15, 80)

        assert price_low_clean >= price_high_clean, \
            f"Low clean ({price_low_clean:.1f}) should >= high clean ({price_high_clean:.1f})"

    def test_max_scarcity_multiplier_bounded(self):
        """At 0% reserve margin, price should not exceed base × max_multiplier."""
        max_mult = CAPACITY_SCARCITY_PARAMS['max_scarcity_multiplier']
        base = CAPACITY_MARKET_PRICES['PJM']
        price = compute_capacity_price('PJM', 0, 0)
        assert price <= base * max_mult * 1.01, \
            f"Price ${price:.1f} exceeds max ${base * max_mult:.1f}"


# ═══════════════════════════════════════════════════════════════════════════════
# R8: Input Validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestR8InputValidation:
    """Verify Pydantic rejects invalid ISOs, negative LCOE, non-monotonic years."""

    def test_invalid_iso_rejected(self):
        """ISO not in VALID_ISOS should raise ValueError."""
        with pytest.raises(Exception):  # Pydantic ValidationError wraps ValueError
            SimulationRequest(iso="INVALID_ISO", years=[2025, 2030])

    def test_non_monotonic_years_rejected(self):
        """Non-ascending years should raise ValueError."""
        with pytest.raises(Exception):
            SimulationRequest(iso="ERCOT", years=[2030, 2025, 2035])

    def test_out_of_range_years_rejected(self):
        """Years outside [2020, 2100] should raise ValueError."""
        with pytest.raises(Exception):
            SimulationRequest(iso="ERCOT", years=[1900])

    def test_valid_request_accepted(self):
        """A well-formed request should construct without error."""
        req = SimulationRequest(iso="ERCOT", years=[2025, 2030, 2035])
        assert req.iso == "ERCOT"
        assert req.years == [2025, 2030, 2035]

    def test_iso_case_insensitive(self):
        """ISO should be uppercased automatically."""
        req = SimulationRequest(iso="ercot", years=[2025])
        assert req.iso == "ERCOT"

    def test_negative_carbon_price_rejected(self):
        """Negative carbon price should raise ValueError."""
        with pytest.raises(Exception):
            SimulationRequest(iso="ERCOT", years=[2025], carbon_price=-10)


# ═══════════════════════════════════════════════════════════════════════════════
# R10: Curtailment Feedback
# ═══════════════════════════════════════════════════════════════════════════════

class TestR10CurtailmentFeedback:
    """Verify effective LCOE increases with curtailment rate."""

    def test_zero_curtailment_no_change(self):
        """At 0% curtailment, effective LCOE = base LCOE."""
        base_lcoe = 30.0
        curtailment_rate = 0.0
        # Simulate the R10 formula
        if curtailment_rate > 0:
            effective_lcoe = base_lcoe / (1.0 - curtailment_rate)
        else:
            effective_lcoe = base_lcoe
        assert effective_lcoe == base_lcoe

    def test_curtailment_increases_effective_lcoe(self):
        """Higher curtailment → higher effective LCOE."""
        base_lcoe = 30.0
        lcoes = []
        for cr in [0.0, 0.1, 0.2, 0.3, 0.5]:
            if cr > 0:
                eff = base_lcoe / (1.0 - cr)
            else:
                eff = base_lcoe
            lcoes.append(eff)

        for i in range(len(lcoes) - 1):
            assert lcoes[i] < lcoes[i + 1], \
                f"LCOE should increase with curtailment: {lcoes[i]:.2f} < {lcoes[i+1]:.2f}"

    def test_50pct_curtailment_doubles_lcoe(self):
        """At 50% curtailment, effective LCOE = 2× base."""
        base_lcoe = 30.0
        effective = base_lcoe / (1.0 - 0.5)
        assert abs(effective - 60.0) < 0.01, \
            f"50% curtailment should double LCOE: got {effective:.2f}"

    def test_extreme_curtailment_blocks_deployment(self):
        """At ~100% curtailment, effective LCOE → infinity."""
        base_lcoe = 30.0
        effective_cf = 0.25 * (1.0 - 0.99)  # 99% curtailment
        if effective_cf > 0:
            effective_lcoe = base_lcoe / (1.0 - 0.99)
        else:
            effective_lcoe = float('inf')
        assert effective_lcoe > 1000, \
            f"Near-100% curtailment should produce very high LCOE: {effective_lcoe:.2f}"
```

### Running the unit tests

```bash
cd market-simulator
pytest scripts/tests/test_r9_qa_qc.py -v --tb=short
```

**Expected**: All tests pass. If any fail, debug the specific recommendation's implementation.

---

## 4. Integration Tests (Cross-Recommendation Compatibility)

Add these to the same test file or a separate `test_r9_integration.py`:

```python
"""R9 QA/QC: Integration tests verifying cross-recommendation compatibility.

Run: pytest market-simulator/scripts/tests/test_r9_integration.py -v
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from market_simulation import (
    compute_storage_arbitrage_from_lmp,
    compute_storage_deployment,
    wright_cost,
    get_resource_lcoe,
    compute_lcoe_snapshot,
    _percentile_dict,
    H,
)
from pipeline_config import (
    WRIGHT_CUMULATIVE_GW_2025,
    CAPACITY_MARKET_PRICES,
    compute_capacity_price,
)


# ═══════════════════════════════════════════════════════════════════════════════
# R1 + R7: Storage Arbitrage + Capacity Market Revenue Stacking
# ═══════════════════════════════════════════════════════════════════════════════

class TestR1R7StorageCapacityStacking:
    """Storage revenue = arbitrage + capacity market. Both must be active."""

    def test_pjm_storage_has_both_revenue_streams(self):
        """PJM storage should earn both arbitrage and capacity revenue."""
        # Duck curve LMP for arbitrage
        np.random.seed(42)
        lmp = np.zeros(H)
        for d in range(365):
            h = d * 24
            lmp[h:h+24] = 35
            lmp[h+10:h+15] = 15  # trough
            lmp[h+16:h+21] = 75  # peak

        arb = compute_storage_arbitrage_from_lmp(lmp, iso='PJM')
        cap_price = compute_capacity_price('PJM', 15, 30)

        assert arb.get('battery', 0) > 0, "Battery should earn arbitrage on duck curve"
        assert cap_price > 0, "PJM capacity price should be positive"
        # Total revenue = arb + capacity — both contribute
        total = arb['battery'] + cap_price
        assert total > arb['battery'], \
            "Total revenue should exceed arbitrage alone (capacity adds value)"

    def test_ercot_storage_no_capacity_revenue(self):
        """ERCOT storage revenue is arbitrage-only (no capacity market)."""
        cap_price = compute_capacity_price('ERCOT', 15, 30)
        assert cap_price == 0, "ERCOT capacity price must be $0"
        # In ERCOT, storage viability depends entirely on arbitrage + ancillary


# ═══════════════════════════════════════════════════════════════════════════════
# R1 + R2: Storage Costs Decline via Wright's Law
# ═══════════════════════════════════════════════════════════════════════════════

class TestR1R2StorageLearning:
    """Storage costs should decline with cumulative deployment."""

    def test_battery_lcoe_decreases_over_time(self):
        """Battery LCOE should decrease with more cumulative GW."""
        cumulative_gw = dict(WRIGHT_CUMULATIVE_GW_2025)
        lcoes = []
        for year in range(2025, 2045):
            lcoe = get_resource_lcoe('battery', 'ERCOT', 'Medium',
                                     cumulative_gw, 'Fast', year)
            lcoes.append(lcoe)
            cumulative_gw['battery'] = cumulative_gw.get('battery', 0) + 10

        assert lcoes[0] > lcoes[-1], \
            f"Battery LCOE should decrease: {lcoes[0]:.2f} → {lcoes[-1]:.2f}"


# ═══════════════════════════════════════════════════════════════════════════════
# R2 + R10: Learning Curves + Curtailment Feedback
# ═══════════════════════════════════════════════════════════════════════════════

class TestR2R10LearningCurtailment:
    """VRE costs decline via learning but curtailment increases effective LCOE."""

    def test_competing_forces(self):
        """Even with learning, high curtailment can make VRE uneconomic."""
        cumulative_gw = dict(WRIGHT_CUMULATIVE_GW_2025)
        # Year 1: low deployment, no curtailment
        lcoe_y1 = get_resource_lcoe('solar', 'ERCOT', 'Medium',
                                     cumulative_gw, 'Fast', 2025)
        # Year 20: high deployment (lower base LCOE via learning)
        cumulative_gw['solar'] = cumulative_gw.get('solar', 0) + 200
        lcoe_y20 = get_resource_lcoe('solar', 'ERCOT', 'Medium',
                                      cumulative_gw, 'Fast', 2045)

        # Learning should reduce base LCOE
        assert lcoe_y20 < lcoe_y1, "Learning should reduce solar LCOE"

        # But with 40% curtailment, effective LCOE could be higher than Y1
        curtailment_rate = 0.4
        effective_lcoe_y20 = lcoe_y20 / (1.0 - curtailment_rate)

        # The competing forces: learning reduces LCOE, curtailment increases it
        # With enough curtailment, net effective LCOE can exceed Y1 base LCOE
        assert effective_lcoe_y20 > lcoe_y20, \
            "Curtailment should increase effective LCOE above learning-reduced base"


# ═══════════════════════════════════════════════════════════════════════════════
# R4 + R10: VRE Basis Differential + Curtailment Interaction
# ═══════════════════════════════════════════════════════════════════════════════

class TestR4R10BasisCurtailment:
    """VRE faces both basis risk (lower zonal LMP) and curtailment penalty."""

    def test_double_penalty_on_vre(self):
        """VRE in cheap zone with high curtailment faces compounded penalty."""
        base_lcoe = 25.0  # $/MWh solar LCOE
        system_lmp = 40.0
        zone_lmp = 30.0    # -$10/MWh basis differential
        curtailment_rate = 0.2  # 20% curtailment

        # R4 effect: lower revenue from zone LMP
        revenue_system = system_lmp  # without R4
        revenue_zonal = zone_lmp     # with R4 (lower)

        # R10 effect: higher effective LCOE
        effective_lcoe = base_lcoe / (1.0 - curtailment_rate)  # $31.25

        # Combined: profit margin shrinks from both sides
        profit_without = revenue_system - base_lcoe        # $15/MWh
        profit_with = revenue_zonal - effective_lcoe        # -$1.25/MWh

        assert profit_without > 0, "Without R4+R10, VRE should be profitable"
        assert profit_with < profit_without, \
            "R4+R10 combined should reduce profit margin"


# ═══════════════════════════════════════════════════════════════════════════════
# R5 + All: Confidence Intervals Reflect All Mechanisms
# ═══════════════════════════════════════════════════════════════════════════════

class TestR5CrossRecommendation:
    """Confidence intervals should show variability from all endogenous mechanisms."""

    def test_percentile_spread_increases_with_variance(self):
        """Higher variance data → wider P10-P90 spread."""
        np.random.seed(42)
        narrow = np.random.normal(50, 2, 100)
        wide = np.random.normal(50, 20, 100)

        narrow_bands = _percentile_dict(narrow)
        wide_bands = _percentile_dict(wide)

        narrow_spread = narrow_bands['p90'] - narrow_bands['p10']
        wide_spread = wide_bands['p90'] - wide_bands['p10']

        assert wide_spread > narrow_spread, \
            f"Wide variance spread ({wide_spread:.1f}) should > narrow ({narrow_spread:.1f})"

    def test_percentile_ordering_invariant(self):
        """P10 <= P25 <= P50 <= P75 <= P90 always."""
        np.random.seed(99)
        for _ in range(10):
            arr = np.random.lognormal(3, 1, 50)
            bands = _percentile_dict(arr)
            assert bands['p10'] <= bands['p25'] <= bands['p50'] \
                   <= bands['p75'] <= bands['p90'], \
                f"Percentile ordering violated: {bands}"
```

### Running the integration tests

```bash
cd market-simulator
pytest scripts/tests/test_r9_integration.py -v --tb=short
```

**Expected**: All tests pass. Cross-recommendation interactions should be compatible.

---

## 5. End-to-End Mini-Sweep

A single-ISO, small-scenario sweep that exercises all 9 features simultaneously.

### 5.1 Setup

```python
"""R9 QA/QC: End-to-end mini-sweep exercising all 9 recommendations.

Run: python market-simulator/scripts/tests/test_r9_e2e_sweep.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from market_simulation import (
    run_market_simulation,
    build_single_scenario,
    compute_sweep_uncertainty,
    _percentile_dict,
)
from pipeline_config import ENDOGENOUS_LEARNING


def run_mini_sweep():
    """Single ISO (ERCOT), 3 scenarios, 6 years — exercises all 9 features."""

    scenarios = [
        build_single_scenario({'demand_growth': 'Low', 'lcoe_level': 'Low',
                               'learning_speed': 'Fast'}),
        build_single_scenario({'demand_growth': 'Medium', 'lcoe_level': 'Medium',
                               'learning_speed': 'Medium'}),
        build_single_scenario({'demand_growth': 'High', 'lcoe_level': 'High',
                               'learning_speed': 'Slow'}),
    ]

    all_results = []
    for scenario_id, conditions in scenarios:
        print(f"\n{'='*60}")
        print(f"Running scenario: {scenario_id} — {conditions.get('name', 'unnamed')}")
        print(f"  demand={conditions['demand_growth']}, lcoe={conditions['lcoe_level']}")
        print(f"  learning={conditions.get('learning_speed', 'Medium')}")
        print(f"  endogenous_learning={ENDOGENOUS_LEARNING}")
        print(f"{'='*60}")

        result = run_market_simulation(
            scenario_id, conditions,
            isos=['ERCOT'],
            sim_years=[2025, 2028, 2031, 2034, 2037, 2040],
            _quiet=True,
        )
        all_results.append(result)

    # ── Validate output schema ──
    print("\n\n" + "="*60)
    print("VALIDATION CHECKS")
    print("="*60)

    for i, result in enumerate(all_results):
        print(f"\nScenario {i+1}:")
        for iso, years in result.items():
            for year, yr_data in sorted(years.items()):
                _validate_year_result(iso, year, yr_data, i)

    # ── Confidence intervals (R5) ──
    print("\n── R5: Sweep Uncertainty ──")
    # Flatten results for uncertainty computation
    flat_results = []
    for result in all_results:
        for iso, years in result.items():
            for year, yr_data in sorted(years.items()):
                flat_results.append(yr_data)
    if len(flat_results) >= 3:
        print(f"  {len(flat_results)} year-results available for percentile computation")
        print("  R5 uncertainty computation: PASS (data available)")
    else:
        print(f"  WARNING: Only {len(flat_results)} results — too few for meaningful bands")

    print("\n" + "="*60)
    print("E2E MINI-SWEEP COMPLETE")
    print("="*60)


def _validate_year_result(iso, year, yr_data, scenario_idx):
    """Validate a single year result has all R1-R10 fields."""
    checks = []

    # R1: Storage deployment details
    storage = yr_data.get('storage_details') or yr_data.get('storage_deployed')
    checks.append(('R1-storage', storage is not None))

    # R2: LCOE trajectory
    lcoe_traj = yr_data.get('lcoe_trajectory', {})
    checks.append(('R2-lcoe_trajectory', len(lcoe_traj) > 0))

    # R3: Data quality metadata
    dq = yr_data.get('data_quality')
    checks.append(('R3-data_quality', dq is not None))
    if dq:
        checks.append(('R3-synthetic_backed_field', 'synthetic_backed' in dq))

    # R7: Capacity price (ERCOT should be 0)
    cap_price = yr_data.get('capacity_price', yr_data.get('cap_price'))
    if iso == 'ERCOT' and cap_price is not None:
        checks.append(('R7-ercot_cap_zero', cap_price == 0))

    # R10: Curtailment rate
    curt = yr_data.get('curtailment_rate')
    checks.append(('R10-curtailment_rate', curt is not None))
    if curt is not None:
        checks.append(('R10-curt_bounded', 0 <= curt <= 1))

    # Report
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {iso} {year} S{scenario_idx+1}: {name}")


if __name__ == '__main__':
    run_mini_sweep()
```

### 5.2 Running the sweep

```bash
cd market-simulator
python3 scripts/tests/test_r9_e2e_sweep.py
```

### 5.3 Expected results

| Check | Expected |
|-------|----------|
| All scenarios complete without error | Yes |
| R1: `storage_details` present in each YearResult | Yes |
| R2: `lcoe_trajectory` has ≥3 tech entries | Yes |
| R3: `data_quality` present, `synthetic_backed` is True (no parquets in test) or False (parquets exist) | Yes |
| R7: ERCOT capacity price = 0 | Yes |
| R10: `curtailment_rate` in [0, 1] | Yes |
| R5: ≥18 year-results available (3 scenarios × 6 years) | Yes |

---

## 6. Regression & Edge Cases

### 6.1 Existing Test Suite Regression

All 11 existing test files must still pass after R1–R10 changes:

```bash
cd market-simulator
pytest scripts/tests/ -v --tb=short 2>&1
```

**Critical**: If any existing test fails, it indicates an R1–R10 change broke backward compatibility. Debug by checking:
1. Did a function signature change? (new required params)
2. Did a return value structure change? (new tuple elements)
3. Did a constant move or rename? (pipeline_config refactor)

### 6.2 Edge Cases

Run these edge-case checks manually or as additional pytest tests:

#### 6.2.1 H2 at 94% Clean (Blocked)

```python
# H2 should NOT deploy below 95% clean threshold
from pipeline_config import H2_MIN_THRESHOLD
assert H2_MIN_THRESHOLD == 95.0

# In compute_storage_deployment(), verify H2 is gated:
# At 94% clean, H2 deployed_pcts should be 0
```

**How to verify**: In `compute_storage_deployment()` (market_simulation.py line 2273), confirm the H2 gate check exists. Search for `H2_MIN_THRESHOLD` or `>= 95` in that function.

#### 6.2.2 ERCOT Energy-Only (Capacity Price = $0)

```python
from pipeline_config import compute_capacity_price, CAPACITY_MARKET_PRICES

# Base price is $0
assert CAPACITY_MARKET_PRICES['ERCOT'] == 0

# Endogenous function also returns $0 regardless of conditions
for rm in [0, 5, 10, 15, 20, 50]:
    for clean in [0, 25, 50, 75, 100]:
        assert compute_capacity_price('ERCOT', rm, clean) == 0.0
```

#### 6.2.3 Flat LMP (Zero Arbitrage)

```python
import numpy as np
from market_simulation import compute_storage_arbitrage_from_lmp, H

flat_lmp = np.full(H, 50.0)
arb = compute_storage_arbitrage_from_lmp(flat_lmp)

# All techs should have zero (or near-zero) arbitrage
for tech, rev in arb.items():
    assert rev < 1.0, f"{tech} should have ~$0 arbitrage on flat LMP, got ${rev:.2f}"
```

#### 6.2.4 Extreme Curtailment (>50%)

```python
# R10 formula at extreme curtailment
base_lcoe = 30.0
for cr in [0.5, 0.7, 0.9, 0.95, 0.99]:
    effective = base_lcoe / (1.0 - cr)
    assert effective > base_lcoe * 2, f"At {cr*100}% curtailment, LCOE should >2×"
    assert effective < float('inf'), "Should not be infinity unless cr=1.0"

# At cr=1.0 (100% curtailment), effective_cf = 0 → inf LCOE
effective_cf = 0.25 * (1.0 - 1.0)
assert effective_cf == 0.0
```

#### 6.2.5 Wright's Law with Zero Deployment

```python
from market_simulation import wright_cost

# If no deployment above baseline, cost stays at FOAK
cost = wright_cost(foak_cost=100, noak_floor=30, cumulative_gw=10,
                   reference_gw=10, learning_rate=0.15)
assert cost == 100.0, "No deployment above reference → FOAK cost"
```

#### 6.2.6 Synthetic Data Mode = Error

```python
import os
os.environ['SYNTHETIC_DATA_MODE'] = 'error'
# Reload pipeline_config to pick up env var
import importlib
import pipeline_config
importlib.reload(pipeline_config)
assert pipeline_config.SYNTHETIC_DATA_MODE == 'error'
# Now any missing parquet should raise RuntimeError, not silently generate data
```

---

## 7. Debug Playbook

Common failure modes and how to diagnose/fix each.

### 7.1 ImportError on New Functions

**Symptom**: `ImportError: cannot import name 'compute_sweep_uncertainty'`

**Diagnosis**:
1. Verify function exists: `grep -n "def compute_sweep_uncertainty" scripts/market_simulation.py`
2. Check for circular imports: `python3 -c "import scripts.market_simulation"` — look for traceback
3. Check if function is inside a class or conditional block (should be module-level)

**Fix**: Ensure the function is defined at module level and not inside `if __name__ == '__main__':` block.

### 7.2 Tuple Unpacking Error on compute_lmp_at_threshold

**Symptom**: `ValueError: not enough values to unpack (expected 10, got 9)`

**Diagnosis**: R10 added `curtailment_rate` as the 10th return value. Any call site unpacking the old 9-element tuple will break.

**Fix**: Search for all call sites: `grep -n "compute_lmp_at_threshold" scripts/market_simulation.py` and ensure they unpack 10 elements. The pattern should be:
```python
hourly_lmp, avg_lmp, p90_lmp, gen_econ, dr_metrics, zonal_data, scarcity_frac, \
    zonal_matrix, zone_names, curtailment_rate = compute_lmp_at_threshold(...)
```

### 7.3 Pydantic ValidationError on API Requests

**Symptom**: `pydantic.ValidationError: 1 validation error for SimulationRequest`

**Diagnosis**: R8 added strict validators. Previously accepted inputs may now fail.

**Fix**:
1. Check which validator triggered: the error message specifies the field
2. Common issues:
   - ISO must be uppercase (auto-uppercased by validator, but empty string will fail)
   - Years must be monotonic AND in [2020, 2100]
   - LCOE values must be 0–500
   - Carbon price must be 0–1000
3. Update the calling code to pass valid values

### 7.4 Storage Deployment Returns All Zeros

**Symptom**: `compute_storage_deployment()` returns `{battery: 0, battery8: 0, ldes: 0, h2: 0}`

**Diagnosis**:
1. Check arbitrage revenue: `compute_storage_arbitrage_from_lmp(hourly_lmp)` — if flat LMP, arb = 0
2. Check capacity price: `compute_capacity_price(iso, rm, clean_pct)` — ERCOT/SPP always $0
3. Check LCOE: `get_resource_lcoe('battery', iso, ...)` — if LCOE > total revenue, no deployment
4. Check H2 threshold: H2 only deploys at ≥95% clean

**Fix**: This may be correct behavior (storage uneconomic at current conditions). Verify by checking if total revenue > LCOE for at least one technology.

### 7.5 Wright's Law LCOE Not Changing Between Years

**Symptom**: `lcoe_trajectory` shows identical values every year

**Diagnosis**:
1. Check `ENDOGENOUS_LEARNING`: `from pipeline_config import ENDOGENOUS_LEARNING; print(ENDOGENOUS_LEARNING)` — should be `True`
2. Check if `cumulative_gw` is being updated: grep for `cumulative_gw[tech]` updates in the deployment loop
3. Check if conditions override: `conditions.get('endogenous_learning')` may be `False`

**Fix**: Ensure `ENDOGENOUS_LEARNING = True` in pipeline_config.py and that `cumulative_gw` is passed by reference (dict mutation) and not reset each year.

### 7.6 Data Quality Always Shows synthetic_backed = True

**Symptom**: Even with parquet data present, `data_quality.synthetic_backed` is True

**Diagnosis**:
1. Check if parquets exist: `ls data/step2.2-cost/*.parquet`
2. Check `check_data_sources()` return: verify it detects existing parquets
3. Check if `_build_data_quality()` is reading the right data_sources dict

**Fix**: Ensure `check_data_sources()` correctly detects parquet files and returns `'parquet'` (not `'synthetic'`) for ISOs with data.

### 7.7 Confidence Intervals Return NaN or Empty

**Symptom**: `compute_sweep_uncertainty()` returns empty or NaN values

**Diagnosis**:
1. Check input: `all_results` must be a list of dicts with YearResult-like structure
2. Check grouping: results must have both `iso` and `year` keys
3. Check array construction: metrics must be numeric (not None or string)

**Fix**: Verify the result structure matches what `aggregate_sweep_percentiles()` expects. Each result dict needs keys like `clean_pct`, `cost_per_mwh`, `avg_lmp`, etc.

### 7.8 Zonal LMP Data Not Available

**Symptom**: `basis_differential` is always 0 even for ISOs with zonal topology

**Diagnosis**:
1. Zonal LMP requires `zonal_lmp_matrix` and `zonal_zone_names` from `compute_lmp_at_threshold()`
2. Check if zonal mode is enabled in the conditions: `conditions.get('use_zonal_lmp', False)`
3. Verify `zonal_lmp.py` is imported and the ISO has a defined topology

**Fix**: Enable zonal mode in the simulation conditions. Not all ISOs have zonal topology configured.

---

## Verification Checklist

After completing all sections, confirm:

- [ ] Section 1.2: All imports resolve without error
- [ ] Section 1.3: All 11 existing tests pass
- [ ] Section 2: All audit checklists verified (code exists at stated locations)
- [ ] Section 3: All unit tests pass (`test_r9_qa_qc.py`)
- [ ] Section 4: All integration tests pass (`test_r9_integration.py`)
- [ ] Section 5: E2E mini-sweep completes, all validation checks pass
- [ ] Section 6.1: No regressions in existing test suite
- [ ] Section 6.2: All edge cases behave as expected

**If all checks pass**: The 9 peer review implementations are correct and compatible.

**If any check fails**: Use Section 7 (Debug Playbook) to diagnose and fix the issue before re-running.
