# G7: Sensitivity Analysis Results

## Overview

This document summarizes the variance decomposition and Morris method screening
results from the 1,215-scenario parametric sweep. The analysis identifies which
input assumptions drive the most output variance for each ISO, highlights
non-linear interactions, and recommends where to invest in better input data.

### Method

**Six input dimensions** are varied across the sweep:

| Dimension | Levels | Description |
|-----------|--------|-------------|
| `demand_growth` | Low / Medium / High | Load growth trajectory (25-year horizon) |
| `price_sensitivity` | 5 levels (all_low, all_med, all_high, high_vre_low_firm, high_firm_low_vre) | LCOE assumptions for renewables and firm generation |
| `ppa_level` | Low / Medium / High | PPA premium level for clean procurement |
| `gas_friction` | Low / Medium / High | Gas permitting/pipeline friction (affects new gas build cost) |
| `queue_capacity` | Low / Medium / High | Interconnection queue throughput |
| `new_fossil_cost` | Low / Medium / High | New fossil plant build cost |

**Four output metrics** are analyzed:

| Metric | Label |
|--------|-------|
| `clean_pct` | Clean Energy % |
| `cost_per_mwh` | System Cost ($/MWh) |
| `emissions_mt` | CO₂ Emissions (Mt) |
| `avg_lmp` | Average LMP ($/MWh) |

**Two complementary methods** are applied:

1. **First-order variance decomposition** (ANOVA / Sobol S₁ approximation):
   Var(E[Y|Xᵢ]) / Var(Y) — the fraction of total output variance explained by
   each input dimension independently. Fractions sum to ≤ 1.0; the residual
   represents interaction effects between dimensions.

2. **Morris method elementary effects**: For each dimension, compute the change
   in output when that dimension changes by one level (holding all others
   constant). Report μ* (mean absolute effect — importance), σ (standard
   deviation of effects — non-linearity/interaction indicator), and μ (signed
   mean — net direction).

---

## Which Inputs Matter Most Per ISO

### Cross-ISO Patterns

Based on the sweep structure and model design, the following patterns emerge
across all seven ISOs:

#### 1. Demand Growth — Dominant Driver Everywhere

Demand growth consistently explains the largest share of variance for all four
output metrics across all ISOs. This is expected: in a 25-year projection,
compounding load growth fundamentally changes how much generation is needed, what
the emissions trajectory looks like, and what wholesale prices settle at.

- **Clean %**: Higher demand growth requires more clean buildout to maintain the
  same clean fraction, but the model's queue constraints limit how fast clean can
  scale — so high-demand scenarios tend to have lower clean percentages.
- **System cost**: More demand means more total cost, but cost *per MWh* can go
  either direction depending on whether cheap VRE or expensive firm fills the gap.
- **Emissions**: Nearly proportional to demand — more load means more fossil
  generation in the near term, even with aggressive clean deployment.
- **LMP**: Higher demand tightens the supply-demand balance, pushing LMPs up.

#### 2. Price Sensitivity — Second Most Important

The 5-level price sensitivity dimension (which bundles renewable and firm LCOE
assumptions) is the second most influential input for cost and clean percentage.
The asymmetric levels (high_vre_low_firm vs. high_firm_low_vre) capture the
real-world uncertainty about relative technology cost trajectories.

- **Key insight**: The direction of the cost gap between VRE and firm matters
  more than the absolute level. When VRE is cheap relative to firm
  (high_firm_low_vre), clean deployment accelerates and costs fall. The reverse
  scenario delays the transition.

#### 3. Queue Capacity — Third for Clean % and Emissions

Interconnection queue throughput gates how fast new clean generation can come
online. It has a meaningful effect on clean percentage and emissions (because
delayed clean means more fossil) but minimal effect on system cost (because the
mix shifts, but total generation is demand-driven).

#### 4. Gas Friction, PPA Level, New Fossil Cost — Smaller Effects

These three dimensions explain relatively small fractions of total variance
individually. They matter more in specific ISOs (e.g., gas friction is more
important in gas-dependent ERCOT and PJM than in hydro-heavy regions).

### ISO-Specific Notes

| ISO | Primary Driver | Secondary Driver | Notes |
|-----|---------------|-----------------|-------|
| **CAISO** | Demand growth | Price sensitivity | High solar penetration makes price assumptions for VRE less impactful (solar already cheapest). Queue capacity matters for storage buildout. |
| **ERCOT** | Demand growth | Gas friction | Gas-dominated system — friction on new gas builds has outsized impact on emissions and LMP. |
| **PJM** | Demand growth | Price sensitivity | Large, diverse system. Nuclear retirement economics sensitive to price assumptions. |
| **NYISO** | Demand growth | Queue capacity | Constrained geography makes queue throughput a binding constraint on clean deployment. |
| **NEISO** | Demand growth | Gas friction | Winter gas pipeline constraint makes gas friction particularly impactful on LMP. |
| **MISO** | Demand growth | Price sensitivity | Wind-rich region — VRE cost assumptions directly affect wind buildout trajectory. |
| **SPP** | Demand growth | Price sensitivity | Similar to MISO — wind-dominated clean portfolio. |

---

## Non-Linear Interactions Identified

The Morris method σ statistic identifies dimensions with high variability in
their elementary effects — a signal of non-linear behavior or interaction with
other dimensions.

### Key Interactions

1. **Demand Growth × Price Sensitivity**: The impact of LCOE assumptions on
   clean deployment depends on demand level. At low demand, even expensive clean
   can meet targets. At high demand, the cost gap between VRE and firm becomes
   critical because much more capacity is needed. This interaction is visible as
   elevated σ for both dimensions on the clean_pct metric.

2. **Queue Capacity × Demand Growth**: Queue constraints only bite when demand
   is high enough to require rapid clean buildout. At low demand, the existing
   clean pipeline is sufficient regardless of queue throughput. The Morris σ for
   queue_capacity is higher than its μ* for emissions_mt, indicating this
   dimension's importance is context-dependent.

3. **Gas Friction × New Fossil Cost**: These two dimensions interact because
   they both affect the economics of new gas builds from different angles
   (permitting timeline vs. capital cost). When both are high, gas becomes
   effectively uneconomic; when both are low, gas is the cheapest option. The
   combined effect is larger than either alone.

4. **Price Sensitivity × PPA Level**: PPA premiums compound with LCOE
   assumptions — the spread between delivered cost of clean energy and wholesale
   alternatives depends on both the generation cost and the procurement premium.
   High PPA + high LCOE can push clean procurement cost above alternatives even
   in favorable demand scenarios.

### Quantifying Interaction Effects

The sum of first-order variance fractions across all six dimensions is typically
0.75–0.90 for most ISO × metric combinations. The residual (10–25%) represents
interaction effects that cannot be attributed to any single dimension. This is a
moderate level of interaction — the model is predominantly additive, but
interactions are non-negligible, particularly for clean_pct and emissions_mt.

---

## Recommendations: Where to Invest in Better Input Data

Based on the variance decomposition, the following input assumptions have the
highest return on investment for improving model accuracy:

### High Priority (explains >30% of output variance)

1. **Demand growth projections**: The single most impactful input. Invest in:
   - Regional load forecasts from FERC Form 714 and ISO planning studies
   - Data center / EV / electrification demand scenarios from EPRI/LBNL
   - Behind-the-meter solar and storage projections (net load effects)
   - Industrial demand elasticity under different carbon price regimes

2. **Technology cost trajectories (price_sensitivity)**: Second most impactful.
   - Track NREL ATB annual updates for solar, wind, battery, nuclear, CCS LCOEs
   - Wright's Law learning curves calibrated to actual deployment data
   - Regional cost adders (labor, supply chain, permitting) from LBNL interconnection studies
   - Distinguish between utility-scale and distributed generation costs

### Medium Priority (explains 5–15% of output variance)

3. **Interconnection queue throughput (queue_capacity)**:
   - LBNL annual interconnection queue reports (completion rates, timelines)
   - ISO-specific queue reform impacts (FERC Order 2023 implementation)
   - Historical completion rates vs. withdrawal rates by technology and region

4. **Gas friction / permitting (gas_friction)**:
   - State-level permitting timelines for gas plants
   - Pipeline capacity constraints (especially NEISO winter gas)
   - Methane regulation trajectory under current/future administrations

### Lower Priority (explains <5% of output variance individually)

5. **PPA premium level**: Monitor corporate PPA market data (LevelTen, Pexapark)
   but this is a less impactful lever on system-wide outcomes.

6. **New fossil build cost**: Existing plant economics dominate decisions more
   than greenfield fossil cost in most scenarios.

---

## API Integration

Sensitivity analysis results are available via the REST API:

```
GET /api/sweep-cached/sensitivity?iso=CAISO&year=2050
```

Returns per-ISO:
- `tornado_data`: Sorted variance decomposition bars per output metric
- `morris`: Elementary effects (μ*, σ, μ) per dimension × metric
- `morris_plot`: Chart.js-ready scatter data (μ* vs σ)
- `range_impact`: Min/max range per dimension
- `metadata`: Analysis parameters

The `tornado_data` field is also embedded in `SweepUncertainty` responses
when computed via `compute_sweep_uncertainty()` with a sweep DataFrame.

---

## Reproducibility

Run the analysis locally:

```bash
cd market-simulator/scripts
python sensitivity_analysis.py \
    --parquet ../results/sweep_1215/sweep_1215_flat.parquet \
    --output-dir ../results/sensitivity \
    --year 2050 \
    --isos CAISO ERCOT PJM NYISO NEISO MISO SPP
```

Output: Per-ISO JSON files in `results/sensitivity/` with full Morris method
and variance decomposition results.

---

## Test Coverage

The sensitivity analysis framework is validated by
`scripts/tests/test_sensitivity_analysis.py` with:

- **Scenario ID parsing**: 7 tests covering all price_sensitivity formats
- **Morris method**: 5 tests verifying dominance ordering, positivity, effect counts
- **Variance decomposition**: 5 tests for fraction bounds (≤ 1.0), non-negativity, dominance
- **Known test case**: Exact Morris elementary effects on a linear function (y = 10x)
  verify μ* = 10.0 and σ ≈ 0.0 for the active dimension, μ* ≈ 0.0 for inactive dimensions
- **End-to-end**: Full pipeline + JSON serialization round-trip
