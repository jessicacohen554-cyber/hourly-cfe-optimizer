# Cross-Validation Results

Comparison of the profit-driven market simulator against published reference
trajectories from EIA AEO 2025, NREL ReEDS Standard Scenarios 2024, and EPA IPM v6.

**Generated from cached simulation results** (`data/cv_reference_results.json`).
No simulations were run to produce this document.

---

## Summary Table

ISO × Reference Model × Milestone Year → Divergence (percentage points)

| ISO | Comparison | Year | Model (%) | Reference (%) | Divergence (pp) | Status |
|-----|-----------|------|-----------|---------------|-----------------|--------|
| CAISO | CV_Reference_Zero_Carbon_vs_AEO_Reference | 2030 | 60.0 | 47 | +13.0 | expected |
| CAISO | CV_Reference_Zero_Carbon_vs_AEO_Reference | 2035 | 78.6 | 51 | +27.6 | investigate |
| CAISO | CV_Reference_Zero_Carbon_vs_AEO_Reference | 2040 | 90.0 | 54 | +36.0 | investigate |
| CAISO | CV_Reference_Zero_Carbon_vs_REEDS_Mid | 2030 | 60.0 | 52 | +8.0 | investigate |
| CAISO | CV_Reference_Zero_Carbon_vs_REEDS_Mid | 2035 | 78.6 | 60 | +18.6 | investigate |
| CAISO | CV_Reference_Zero_Carbon_vs_REEDS_Mid | 2040 | 90.0 | 67 | +23.0 | investigate |
| ERCOT | CV_Reference_Zero_Carbon_vs_AEO_Reference | 2030 | 59.2 | 47 | +12.2 | expected |
| ERCOT | CV_Reference_Zero_Carbon_vs_AEO_Reference | 2035 | 59.2 | 51 | +8.2 | expected |
| ERCOT | CV_Reference_Zero_Carbon_vs_AEO_Reference | 2040 | 65.7 | 54 | +11.7 | expected |
| ERCOT | CV_Reference_Zero_Carbon_vs_REEDS_Mid | 2030 | 59.2 | 52 | +7.2 | investigate |
| ERCOT | CV_Reference_Zero_Carbon_vs_REEDS_Mid | 2035 | 59.2 | 60 | -0.8 | expected |
| ERCOT | CV_Reference_Zero_Carbon_vs_REEDS_Mid | 2040 | 65.7 | 67 | -1.3 | expected |
| PJM | CV_Reference_Zero_Carbon_vs_AEO_Reference | 2030 | 46.6 | 47 | -0.4 | expected |
| PJM | CV_Reference_Zero_Carbon_vs_AEO_Reference | 2035 | 50.5 | 51 | -0.5 | expected |
| PJM | CV_Reference_Zero_Carbon_vs_AEO_Reference | 2040 | 53.9 | 54 | -0.1 | expected |
| PJM | CV_Reference_Zero_Carbon_vs_REEDS_Mid | 2030 | 46.6 | 52 | -5.4 | expected |
| PJM | CV_Reference_Zero_Carbon_vs_REEDS_Mid | 2035 | 50.5 | 60 | -9.5 | expected |
| PJM | CV_Reference_Zero_Carbon_vs_REEDS_Mid | 2040 | 53.9 | 67 | -13.1 | expected |

---

## Expected Divergences

These divergences are consistent with the structural differences between
our profit-driven market simulator and the reference models.

- **CAISO 2030** (CV_Reference_Zero_Carbon_vs_AEO_Reference): +13.0 pp — Model shows higher clean share than AEO — plausible if market economics favor renewables
- **ERCOT 2030** (CV_Reference_Zero_Carbon_vs_AEO_Reference): +12.2 pp — Model shows higher clean share than AEO — plausible if market economics favor renewables
- **ERCOT 2035** (CV_Reference_Zero_Carbon_vs_AEO_Reference): +8.2 pp — Within 10 pp of AEO Reference (current policy)
- **ERCOT 2040** (CV_Reference_Zero_Carbon_vs_AEO_Reference): +11.7 pp — Model shows higher clean share than AEO — plausible if market economics favor renewables
- **PJM 2030** (CV_Reference_Zero_Carbon_vs_AEO_Reference): -0.4 pp — Within 10 pp of AEO Reference (current policy)
- **PJM 2035** (CV_Reference_Zero_Carbon_vs_AEO_Reference): -0.5 pp — Within 10 pp of AEO Reference (current policy)
- **PJM 2040** (CV_Reference_Zero_Carbon_vs_AEO_Reference): -0.1 pp — Within 10 pp of AEO Reference (current policy)
- **ERCOT 2035** (CV_Reference_Zero_Carbon_vs_REEDS_Mid): -0.8 pp — Profit-driven model shows lower clean share than ReEDS cost-minimizing model with RPS mandates
- **ERCOT 2040** (CV_Reference_Zero_Carbon_vs_REEDS_Mid): -1.3 pp — Profit-driven model shows lower clean share than ReEDS cost-minimizing model with RPS mandates
- **PJM 2030** (CV_Reference_Zero_Carbon_vs_REEDS_Mid): -5.4 pp — Profit-driven model shows lower clean share than ReEDS cost-minimizing model with RPS mandates
- **PJM 2035** (CV_Reference_Zero_Carbon_vs_REEDS_Mid): -9.5 pp — Profit-driven model shows lower clean share than ReEDS cost-minimizing model with RPS mandates
- **PJM 2040** (CV_Reference_Zero_Carbon_vs_REEDS_Mid): -13.1 pp — Profit-driven model shows lower clean share than ReEDS cost-minimizing model with RPS mandates

### Why Profit-Driven ≠ Cost-Minimizing

Our model simulates **profit-maximizing generator behavior** in competitive
wholesale markets. Generators build or retire based on projected revenue vs.
cost, without a central planner enforcing least-cost outcomes. This differs
from reference models in key ways:

- **vs. ReEDS (NREL)**: ReEDS is a capacity-expansion model that minimizes
  system-wide cost subject to policy constraints (RPS mandates, clean energy
  standards). It assumes perfect foresight and coordinated deployment. Our
  model's clean share will typically be **lower** because profit-driven
  generators don't internalize policy mandates unless they create direct
  revenue signals (carbon prices, RECs, capacity payments).

- **vs. AEO (EIA)**: AEO Reference assumes current policy only, making it
  the closest analog to our zero-carbon-price scenario. Divergence should
  be modest (< 10 pp) — differences stem from our use of ISO-specific
  rather than national fuel mixes and our bottom-up generator dispatch.

- **vs. EPA IPM**: IPM models coal retirement under existing environmental
  regulations. Without explicit policy enforcement, our model may retire
  coal more slowly (profit-driven coal stays online if it covers variable
  costs). With carbon pricing ($51/ton EPA scenario), retirement accelerates.

---

## Unexplained Divergences

The following divergences warrant investigation — they suggest the model
may be producing results inconsistent with its structural assumptions.

- **CAISO 2035** (CV_Reference_Zero_Carbon_vs_AEO_Reference): +27.6 pp — Divergence of +27.6 pp exceeds 20 pp threshold; Model shows higher clean share than AEO — plausible if market economics favor renewables
- **CAISO 2040** (CV_Reference_Zero_Carbon_vs_AEO_Reference): +36.0 pp — Divergence of +36.0 pp exceeds 20 pp threshold; Model shows higher clean share than AEO — plausible if market economics favor renewables
- **CAISO 2030** (CV_Reference_Zero_Carbon_vs_REEDS_Mid): +8.0 pp — Model shows HIGHER clean share than ReEDS — unexpected without policy mandates
- **CAISO 2035** (CV_Reference_Zero_Carbon_vs_REEDS_Mid): +18.6 pp — Model shows HIGHER clean share than ReEDS — unexpected without policy mandates
- **CAISO 2040** (CV_Reference_Zero_Carbon_vs_REEDS_Mid): +23.0 pp — Divergence of +23.0 pp exceeds 20 pp threshold; Model shows HIGHER clean share than ReEDS — unexpected without policy mandates
- **ERCOT 2030** (CV_Reference_Zero_Carbon_vs_REEDS_Mid): +7.2 pp — Model shows HIGHER clean share than ReEDS — unexpected without policy mandates

### Investigation Notes

**CAISO** (max divergence: 36.0 pp):
  - Model shows higher clean share than ReEDS in some years. This is unexpected for a profit-driven model without policy mandates. Possible causes: aggressive renewable LCOE assumptions, high capacity factor assumptions, or carbon price scenario creating stronger incentives than ReEDS reference.
  - Divergence exceeds 20 pp threshold. Review input assumptions (demand growth, LCOE trajectory, fuel prices) for this ISO to verify they align with reference model assumptions.

**ERCOT** (max divergence: 7.2 pp):
  - Model shows higher clean share than ReEDS in some years. This is unexpected for a profit-driven model without policy mandates. Possible causes: aggressive renewable LCOE assumptions, high capacity factor assumptions, or carbon price scenario creating stronger incentives than ReEDS reference.

---

## Methodology Note

### What Divergence Tells Us

Divergence between our profit-driven model and reference trajectories is
**informative, not diagnostic**. These models answer different questions:

| Model | Question Answered |
|-------|------------------|
| Our simulator | What do profit-maximizing generators build in competitive markets? |
| NREL ReEDS | What's the least-cost pathway to meet clean energy targets? |
| EIA AEO | What happens under current policy with moderate assumptions? |
| EPA IPM | How do environmental regulations affect coal/gas fleet composition? |

A **large divergence** doesn't mean either model is wrong — it means the models'
structural assumptions produce different outcomes. Our model lacks policy mandates
(RPS, CES) that drive deployment in ReEDS; conversely, ReEDS doesn't model the
strategic timing and revenue-seeking behavior that shapes real investment decisions.

### What Divergence Does NOT Tell Us

- Divergence is NOT a measure of model accuracy (none of these models predict the future).
- Small divergence doesn't validate our model (could be right for wrong reasons).
- Large divergence doesn't invalidate our model (structural differences are expected).

### Flagging Rules

- **Expected**: Profit-driven model shows lower clean % than cost-minimizing (ReEDS),
  or retires coal slower than policy-driven (EPA IPM) — consistent with no mandates.
- **Investigate**: Model shows HIGHER clean % than ReEDS (shouldn't happen without
  mandates), or divergence exceeds 20 percentage points at any milestone.

---

## National-Level Comparisons

## Cross-Validation: EIA Annual Energy Outlook 2025, Table 8
**Metric**: clean_share_pct

| Year | Model | Reference | Abs Δ | Rel Δ (%) |
|------|-------|-----------|-------|-----------|
| 2030 | 55.27 | 47 | +8.3 | +17.6% |
| 2035 | 62.77 | 51 | +11.8 | +23.1% |
| 2040 | 69.87 | 54 | +15.9 | +29.4% |

**Mean |Δ|**: 11.97  **Max |Δ|**: 15.87

## Cross-Validation: NREL Standard Scenarios 2024, Mid-Case
**Metric**: clean_share_pct

| Year | Model | Reference | Abs Δ | Rel Δ (%) |
|------|-------|-----------|-------|-----------|
| 2030 | 55.27 | 52 | +3.3 | +6.3% |
| 2035 | 62.77 | 60 | +2.8 | +4.6% |
| 2040 | 69.87 | 67 | +2.9 | +4.3% |

**Mean |Δ|**: 2.97  **Max |Δ|**: 3.27

## Cross-Validation: EPA IPM v6 Reference Case (2023)
**Metric**: coal_share_pct

| Year | Model | Reference | Abs Δ | Rel Δ (%) |
|------|-------|-----------|-------|-----------|
| 2030 | — | 12 | Metric 'coal_share_pct' not found in model results for 2030 | — |
| 2035 | — | 8 | Metric 'coal_share_pct' not found in model results for 2035 | — |
| 2040 | — | 5 | Metric 'coal_share_pct' not found in model results for 2040 | — |

**Mean |Δ|**: N/A  **Max |Δ|**: N/A

### CV_Reference_High_Clean_vs_REEDS_Mid

Scenario 'CV_Reference_High_Clean' not found in simulation results
