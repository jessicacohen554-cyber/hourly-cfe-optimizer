# Contrast Audit Report

Generated: 2026-03-15T16:57:23.838Z

## Summary

| Metric | Count |
|--------|-------|
| Total elements | 8283 |
| Passing | 6824 |
| **Failing** | **1459** |
| Critical | 1288 |
| Major | 28 |
| Minor | 143 |

## dashboard/abatement_dashboard.html

**75 failures**

### [CRITICAL] html > body > section#densityDeepDive > div.density-header > h2.section-title

- **Text**: "Resource Investment Deep Dive"
- **Color**: #1E293B on #0F172A
- **Ratio**: 1.22:1 (need 3:1)
- **Font**: 24px, weight 700
- **Fix**: Change text to #5A6577 or adjust for 3:1 on bg #0F172A

### [CRITICAL] html > body > section#densityDeepDive > div.density-header > p#densitySubtitle

- **Text**: "Distribution of new-build capacity across cost scenarios in the optimal target r"
- **Color**: #475569 on #0F172A
- **Ratio**: 2.36:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #748296 or adjust for 4.5:1 on bg #0F172A

### [CRITICAL] html > body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > h2

- **Text**: "Methodology"
- **Color**: #FFFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 3:1)
- **Font**: 22px, weight 800
- **Fix**: Change text to #8C8C8C or adjust for 3:1 on bg #F8F9FC

### [CRITICAL] html > body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > h3

- **Text**: "Marginal Abatement Cost (MAC) Calculation"
- **Color**: #FFFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 3:1)
- **Font**: 17px, weight 700
- **Fix**: Change text to #8C8C8C or adjust for 3:1 on bg #F8F9FC

### [CRITICAL] html > body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > p

- **Text**: "The MAC curve answers:  We compute it from the cumulative cost and CO₂ abatement"
- **Color**: #FEFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727373 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > p > em

- **Text**: "how much does each additional ton of CO₂ abatement cost via clean energy procure"
- **Color**: #FEFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727373 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] html > body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > p

- **Text**: "At each threshold, the optimizer identifies the least-cost portfolio of clean re"
- **Color**: #FEFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727373 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > p > strong

- **Text**: "Step 1 — Cumulative supply curve."
- **Color**: #FFFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #F8F9FC

### [CRITICAL] body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > ul > li

- **Text**: "(metric tons) — total fossil emissions displaced by clean generation at that thr"
- **Color**: #FEFEFF on #F8F9FC
- **Ratio**: 1.05:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > ul > li > strong

- **Text**: "Cumulative CO₂ abated"
- **Color**: #FFFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > ul > li

- **Text**: "Cumulative new-build cost ($) — total annualized cost of all new clean resources"
- **Color**: #FEFEFF on #F8F9FC
- **Ratio**: 1.05:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > ul > li > strong

- **Text**: "Cumulative new-build cost"
- **Color**: #FFFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] html > body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > p

- **Text**: "The raw data points are inherently noisy because each threshold is independently"
- **Color**: #FEFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727373 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > p > strong

- **Text**: "Step 2 — PCHIP spline interpolation."
- **Color**: #FFFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #F8F9FC

### [CRITICAL] html > body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > p

- **Text**: "To extract the true underlying marginal cost signal, we fit a  spline to the cum"
- **Color**: #FEFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727373 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > p > strong

- **Text**: "Piecewise Cubic Hermite Interpolating Polynomial (PCHIP)"
- **Color**: #FFFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #F8F9FC

### [CRITICAL] body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > ul > li

- **Text**: "Preserves the monotonicity of the underlying data (cost and CO₂ both increase wi"
- **Color**: #FEFEFF on #F8F9FC
- **Ratio**: 1.05:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > ul > li

- **Text**: "Avoids overshoot and oscillation artifacts (Runge phenomenon) common with polyno"
- **Color**: #FEFEFF on #F8F9FC
- **Ratio**: 1.05:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > ul > li

- **Text**: "Produces smooth, locally-determined interpolation that respects the physical con"
- **Color**: #FEFEFF on #F8F9FC
- **Ratio**: 1.05:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] html > body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > p

- **Text**: "Step 3 — Marginal MAC via differentiation. The marginal abatement cost at each t"
- **Color**: #FEFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727373 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > p > strong

- **Text**: "Step 3 — Marginal MAC via differentiation."
- **Color**: #FFFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #F8F9FC

### [CRITICAL] html > body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > p

- **Text**: "MAC(t) = d(Cost) / d(CO₂)  evaluated at threshold t"
- **Color**: #FEFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 4.5:1)
- **Font**: 15px, weight 400
- **Fix**: Change text to #727373 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] html > body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > p

- **Text**: "This derivative captures the instantaneous rate of cost increase per unit of add"
- **Color**: #FEFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727373 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] html > body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > p

- **Text**: "Even after PCHIP smoothing, minor numerical artifacts can produce small non-mono"
- **Color**: #FEFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727373 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > p > strong

- **Text**: "Step 4 — Isotonic regression."
- **Color**: #FFFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #F8F9FC

### [CRITICAL] body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > p > strong

- **Text**: "isotonic regression"
- **Color**: #FFFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #F8F9FC

### [CRITICAL] html > body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > p

- **Text**: "The result is a smooth, monotonically non-decreasing MAC curve that starts low ("
- **Color**: #FEFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727373 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] html > body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > h3

- **Text**: "Sensitivity Bands (P10/P50/P90)"
- **Color**: #FFFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 3:1)
- **Font**: 17px, weight 700
- **Fix**: Change text to #8C8C8C or adjust for 3:1 on bg #F8F9FC

### [CRITICAL] html > body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > p

- **Text**: "Each region's MAC curve is computed under 5,832 cost sensitivity scenarios forme"
- **Color**: #FEFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727373 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > ul > li

- **Text**: "5 paired toggles (Low/Medium/High each): Renewable Generation, Firm Generation, "
- **Color**: #FEFEFF on #F8F9FC
- **Ratio**: 1.05:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > ul > li > strong

- **Text**: "5 paired toggles"
- **Color**: #FFFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > ul > li

- **Text**: "CCS cost (Low/Medium/High)"
- **Color**: #FEFEFF on #F8F9FC
- **Ratio**: 1.05:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > ul > li > strong

- **Text**: "CCS cost"
- **Color**: #FFFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > ul > li

- **Text**: "45Q tax credit (On/Off)"
- **Color**: #FEFEFF on #F8F9FC
- **Ratio**: 1.05:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > ul > li > strong

- **Text**: "45Q tax credit"
- **Color**: #FFFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] html > body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > p

- **Text**: "The P10 (optimistic), P50 (median), and P90 (pessimistic) bands shown on the cha"
- **Color**: #FEFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727373 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] html > body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > h3

- **Text**: "Crossover Analysis"
- **Color**: #FFFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 3:1)
- **Font**: 17px, weight 700
- **Fix**: Change text to #8C8C8C or adjust for 3:1 on bg #F8F9FC

### [CRITICAL] html > body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > p

- **Text**: "The optimal CFE target for each region is determined by the crossover point wher"
- **Color**: #FEFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727373 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] html > body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > p

- **Text**: "The analysis uses  (interquartile range) rather than P10/P90 to define the actio"
- **Color**: #FEFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727373 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > p > strong

- **Text**: "P25/P75 cost bounds"
- **Color**: #FFFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #F8F9FC

### [CRITICAL] body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > p > strong

- **Text**: "9 combinations"
- **Color**: #FFFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #F8F9FC

### [CRITICAL] html > body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > h3

- **Text**: "No-Regrets Resource Identification"
- **Color**: #FFFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 3:1)
- **Font**: 17px, weight 700
- **Fix**: Change text to #8C8C8C or adjust for 3:1 on bg #F8F9FC

### [CRITICAL] html > body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > p

- **Text**: "For each threshold within the crossover range, the optimizer evaluates all 5,832"
- **Color**: #FEFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727373 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > ul > li

- **Text**: "No-regrets (≥80% presence): appears in the optimal portfolio across 80%+ of all "
- **Color**: #FEFEFF on #F8F9FC
- **Ratio**: 1.05:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > ul > li > strong

- **Text**: "No-regrets"
- **Color**: #FFFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > ul > li

- **Text**: "Likely (50–80% presence): appears in a majority but not all cost futures."
- **Color**: #FEFEFF on #F8F9FC
- **Ratio**: 1.05:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > ul > li > strong

- **Text**: "Likely"
- **Color**: #FFFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > ul > li

- **Text**: "Scenario-dependent (<50% presence): only optimal under specific cost assumptions"
- **Color**: #FEFEFF on #F8F9FC
- **Ratio**: 1.05:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > ul > li > strong

- **Text**: "Scenario-dependent"
- **Color**: #FFFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] html > body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > p

- **Text**: "The floor (minimum allocation across all scenarios) represents the absolute mini"
- **Color**: #FEFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727373 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > p > em

- **Text**: "floor"
- **Color**: #FEFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727373 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > p > em

- **Text**: "average"
- **Color**: #FEFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727373 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] html > body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > h3

- **Text**: "Data Sources"
- **Color**: #FFFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 3:1)
- **Font**: 17px, weight 700
- **Fix**: Change text to #8C8C8C or adjust for 3:1 on bg #F8F9FC

### [CRITICAL] html > body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > p

- **Text**: "MAC curves derived from the 8,760 Problem hourly matching optimizer using EIA ho"
- **Color**: #FEFFFF on #F8F9FC
- **Ratio**: 1.05:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727373 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] body > section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > details > summary

- **Text**: "Full Sources & References"
- **Color**: #E2E8F0 on #F9FAFC
- **Ratio**: 1.18:1 (need 4.5:1)
- **Font**: 14px, weight 600
- **Fix**: Change text to #6A7078 or adjust for 4.5:1 on bg #F9FAFC

### [CRITICAL] section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > details > ol > li

- **Text**: "EPA eGRID 2022 — regional emission rates and CO₂ intensity by balancing authorit"
- **Color**: #FEFEFF on #F9FAFC
- **Ratio**: 1.04:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F9FAFC

### [CRITICAL] section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > details > ol > li

- **Text**: "EPA Social Cost of Carbon — Technical Support Document (2024), central estimate "
- **Color**: #FEFEFF on #F9FAFC
- **Ratio**: 1.04:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F9FAFC

### [CRITICAL] section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > details > ol > li

- **Text**: "Rennert, K. et al. (2022). “Comprehensive evidence implies a higher social cost "
- **Color**: #FEFEFF on #F9FAFC
- **Ratio**: 1.04:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F9FAFC

### [CRITICAL] div.methodology-inner.section-dark-inner > details > ol > li > em

- **Text**: "Nature"
- **Color**: #FEFEFF on #F9FAFC
- **Ratio**: 1.04:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F9FAFC

### [CRITICAL] section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > details > ol > li

- **Text**: "EU Emissions Trading System (EU ETS) — carbon price range $60–100/metric ton."
- **Color**: #FEFEFF on #F9FAFC
- **Ratio**: 1.04:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F9FAFC

### [CRITICAL] section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > details > ol > li

- **Text**: "NREL Annual Technology Baseline (ATB) 2024 — nuclear learning rates."
- **Color**: #FEFEFF on #F9FAFC
- **Ratio**: 1.04:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F9FAFC

### [CRITICAL] section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > details > ol > li

- **Text**: "Lazard LCOE+, Version 18 (June 2025) — unsubsidized technology costs."
- **Color**: #FEFEFF on #F9FAFC
- **Ratio**: 1.04:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F9FAFC

### [CRITICAL] section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > details > ol > li

- **Text**: "DOE (2024). “Pathways to Commercial Liftoff: Advanced Nuclear.”"
- **Color**: #FEFEFF on #F9FAFC
- **Ratio**: 1.04:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F9FAFC

### [CRITICAL] section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > details > ol > li

- **Text**: "DOE (2023). “Pathways to Commercial Liftoff: Direct Air Capture.”"
- **Color**: #FEFEFF on #F9FAFC
- **Ratio**: 1.04:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F9FAFC

### [CRITICAL] section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > details > ol > li

- **Text**: "Sievert, K. et al. (2024). “DAC cost learning.” Joule."
- **Color**: #FEFEFF on #F9FAFC
- **Ratio**: 1.04:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F9FAFC

### [CRITICAL] div.methodology-inner.section-dark-inner > details > ol > li > em

- **Text**: "Joule"
- **Color**: #FEFEFF on #F9FAFC
- **Ratio**: 1.04:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F9FAFC

### [CRITICAL] section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > details > ol > li

- **Text**: "IEA Net Zero by 2050 Roadmap (2023 update)."
- **Color**: #FEFEFF on #F9FAFC
- **Ratio**: 1.04:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F9FAFC

### [CRITICAL] section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > details > ol > li

- **Text**: "EIA Hourly Electric Grid Monitor — hourly generation by fuel type."
- **Color**: #FEFEFF on #F9FAFC
- **Ratio**: 1.04:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F9FAFC

### [CRITICAL] section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > details > ol > li

- **Text**: "EIA Annual Energy Outlook 2025 — demand growth and price projections."
- **Color**: #FEFEFF on #F9FAFC
- **Ratio**: 1.04:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F9FAFC

### [CRITICAL] section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > details > ol > li

- **Text**: "Fritsch, F.N. & Carlson, R.E. (1980). “Monotone Piecewise Cubic Interpolation.” "
- **Color**: #FEFEFF on #F9FAFC
- **Ratio**: 1.04:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F9FAFC

### [CRITICAL] div.methodology-inner.section-dark-inner > details > ol > li > em

- **Text**: "SIAM J. Numerical Analysis"
- **Color**: #FEFEFF on #F9FAFC
- **Ratio**: 1.04:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F9FAFC

### [CRITICAL] section.methodology-section.section-dark > div.methodology-inner.section-dark-inner > details > ol > li

- **Text**: "Barlow, R.E. et al. (1972). Statistical Inference Under Order Restrictions. Wile"
- **Color**: #FEFEFF on #F9FAFC
- **Ratio**: 1.04:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F9FAFC

### [CRITICAL] div.methodology-inner.section-dark-inner > details > ol > li > em

- **Text**: "Statistical Inference Under Order Restrictions"
- **Color**: #FEFEFF on #F9FAFC
- **Ratio**: 1.04:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F9FAFC

### [MAJOR] section.deepdive-section.section-light > div.deepdive-content > div.deepdive-narrative-col > div#deepdiveNarrative > span#ddNarrativeTag

- **Text**: "ERCOT"
- **Color**: #15803D on #11282E
- **Ratio**: 3.07:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #339E5B or adjust for 4.5:1 on bg #11282E

### [MINOR] body > section.hero-overview > div.hero-findings-row > div.hero-narrative-block > span.finding-highlight

- **Text**: "Grid wins to 69–98% (region-dependent) · SPP & MISO: deepest grid advantage · NE"
- **Color**: #15803D on #E7F5EF
- **Ratio**: 4.46:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #107B38 or adjust for 4.5:1 on bg #E7F5EF

## dashboard/about.html

**12 failures**

### [CRITICAL] body > div.content-wrap > div.theses-section > div.thesis-card.thesis-card-red > span.thesis-num

- **Text**: "II"
- **Color**: #EBECF1 on #F8F9FC
- **Ratio**: 1.12:1 (need 3:1)
- **Font**: 56px, weight 800
- **Fix**: Change text to #8C8D92 or adjust for 3:1 on bg #F8F9FC

### [CRITICAL] body > div.content-wrap > div.theses-section > div.thesis-card.thesis-card-blue > span.thesis-num

- **Text**: "III"
- **Color**: #EBECF1 on #F8F9FC
- **Ratio**: 1.12:1 (need 3:1)
- **Font**: 56px, weight 800
- **Fix**: Change text to #8C8D92 or adjust for 3:1 on bg #F8F9FC

### [CRITICAL] section#sec-layers > div.story-content > div.layer-stack > div.layer-card > span.layer-num

- **Text**: "01"
- **Color**: #F1F2F4 on #FFFFFF
- **Ratio**: 1.12:1 (need 3:1)
- **Font**: 40px, weight 800
- **Fix**: Change text to #929395 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] section#sec-layers > div.story-content > div.layer-stack > div.layer-card > span.layer-num

- **Text**: "02"
- **Color**: #F1F2F4 on #FFFFFF
- **Ratio**: 1.12:1 (need 3:1)
- **Font**: 40px, weight 800
- **Fix**: Change text to #929395 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] section#sec-layers > div.story-content > div.layer-stack > div.layer-card > span.layer-num

- **Text**: "03"
- **Color**: #F1F2F4 on #FFFFFF
- **Ratio**: 1.12:1 (need 3:1)
- **Font**: 40px, weight 800
- **Fix**: Change text to #929395 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] section#sec-layers > div.story-content > div.layer-stack > div.layer-card > span.layer-num

- **Text**: "04"
- **Color**: #F1F2F4 on #FFFFFF
- **Ratio**: 1.12:1 (need 3:1)
- **Font**: 40px, weight 800
- **Fix**: Change text to #929395 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] section#sec-layers > div.story-content > div.layer-stack > div.layer-card > span.layer-num

- **Text**: "05"
- **Color**: #F1F2F4 on #FFFFFF
- **Ratio**: 1.12:1 (need 3:1)
- **Font**: 40px, weight 800
- **Fix**: Change text to #929395 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] section#sec-layers > div.story-content > div.layer-stack > div.layer-card > span.layer-num

- **Text**: "06"
- **Color**: #F1F2F4 on #FFFFFF
- **Ratio**: 1.12:1 (need 3:1)
- **Font**: 40px, weight 800
- **Fix**: Change text to #929395 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] section.section-dark > div.section-dark-inner > div#sec-incentives > div.story-content > span.story-badge.story-badge-orange

- **Text**: "Incentive Design"
- **Color**: #B91C1C on #251E30
- **Ratio**: 2.48:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #F05353 or adjust for 4.5:1 on bg #251E30

### [CRITICAL] section.section-dark > div.section-dark-inner > div#sec-insights > div.story-content > span.story-badge.story-badge-teal

- **Text**: "Novel Insights"
- **Color**: #0F766E on #102D3C
- **Ratio**: 2.62:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #3CA39B or adjust for 4.5:1 on bg #102D3C

### [CRITICAL] div.content-wrap > div.explore-grid > div.explore-card.explore-card-accent-demand > div > a.explore-cta

- **Text**: "Reference Library"
- **Color**: #000000 on #1E293B
- **Ratio**: 1.44:1 (need 4.5:1)
- **Font**: 14px, weight 600
- **Fix**: Change text to #919191 or adjust for 4.5:1 on bg #1E293B

### [CRITICAL] body > div.content-wrap > div.explore-grid > div.explore-card.explore-card-accent-econ > a.explore-cta

- **Text**: "Interactive Cost Optimizer"
- **Color**: #000000 on #1A2744
- **Ratio**: 1.42:1 (need 4.5:1)
- **Font**: 14px, weight 600
- **Fix**: Change text to #919191 or adjust for 4.5:1 on bg #1A2744

## dashboard/archive/pipeline_map.html

**2 failures**

### [CRITICAL] html > body > section.pages-section.section-dark > div.pages-inner.section-dark-inner > h2.section-heading

- **Text**: "Dashboard Pages & Data Dependencies"
- **Color**: #000000 on #0B1120
- **Ratio**: 1.12:1 (need 3:1)
- **Font**: 26px, weight 800
- **Fix**: Change text to #646464 or adjust for 3:1 on bg #0B1120

### [CRITICAL] html > body > section.pages-section.section-dark > div.pages-inner.section-dark-inner > p.section-subheading

- **Text**: "Each page consumes one or more JS data files generated by the pipeline. Hover fo"
- **Color**: #000000 on #0B1120
- **Ratio**: 1.12:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #7D7D7D or adjust for 4.5:1 on bg #0B1120

## dashboard/archive/procurement_comparison.html

**182 failures**

### [CRITICAL] html > body > section.hero-opening > h1

- **Text**: "Every Strategy Looks Fine at Low Adoption"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 29px, weight 800
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > section.hero-opening > p.lead

- **Text**: "Today, voluntary corporate procurement covers a fraction of US commercial & indu"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 17px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.hero-opening > p.counter-context

- **Text**: "of C&I electricity covered by voluntary clean energy procurement"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > div#familyCards > div.family-cards > div#card-consequential > h3

- **Text**: "Cross-Regional Netting"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 16px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] body > div#familyCards > div.family-cards > div#card-consequential > p

- **Text**: "Buy the cheapest clean energy anywhere in the US. Net it against your emissions."
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > div#familyCards > div.family-cards > div#card-consequential > span.variant-count

- **Text**: "3 variants: 1A, 1B, 1C"
- **Color**: #ECF8FD on #ECF8FD
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #657176 or adjust for 4.5:1 on bg #ECF8FD

### [CRITICAL] body > div#familyCards > div.family-cards > div#card-hourly > h3

- **Text**: "Same-ISO Hourly Matching"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 16px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] body > div#familyCards > div.family-cards > div#card-hourly > p

- **Text**: "Match your load hour-by-hour within your own grid region. Most rigorous. Most ex"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > div#familyCards > div.family-cards > div#card-hourly > span.variant-count

- **Text**: "3 variants: 2A, 2B, 2C"
- **Color**: #ECF8FD on #ECF8FD
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #657176 or adjust for 4.5:1 on bg #ECF8FD

### [CRITICAL] body > div#familyCards > div.family-cards > div#card-annual > h3

- **Text**: "Annual Matching"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 16px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] body > div#familyCards > div.family-cards > div#card-annual > p

- **Text**: "Match annual consumption with clean energy certificates. The status quo for most"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > div#familyCards > div.family-cards > div#card-annual > span.variant-count

- **Text**: "4 variants: 3A, 3B, 3C, 3D"
- **Color**: #ECF8FD on #ECF8FD
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #657176 or adjust for 4.5:1 on bg #ECF8FD

### [CRITICAL] html > body > div#taxonomySection > div > p

- **Text**: "Within each family, design choices matter enormously. There are 10 distinct stra"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > div#taxonomySection > div > p > em

- **Text**: "within"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div#taxonomySection > div#strategyColumns > div#col-consequential > div.strat-col-header.strat-col-header-1 > h4

- **Text**: "Strategy 1Consequential Cross-Regional"
- **Color**: #F4F4FD on #F4F4FD
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #8B8B94 or adjust for 3:1 on bg #F4F4FD

### [CRITICAL] div#strategyColumns > div#col-consequential > div.strat-col-header.strat-col-header-1 > h4 > span.strat-subtitle

- **Text**: "Consequential Cross-Regional"
- **Color**: #F4F4FD on #F4F4FD
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #6D6D76 or adjust for 4.5:1 on bg #F4F4FD

### [CRITICAL] div#strategyColumns > div#col-consequential > div.strat-variant-list > div.strat-variant-item.selected > span.variant-badge.badge-data

- **Text**: "data"
- **Color**: #DFF3EC on #DFF3EC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 10px, weight 700
- **Fix**: Change text to #5D716A or adjust for 4.5:1 on bg #DFF3EC

### [CRITICAL] div#strategyColumns > div#col-consequential > div.strat-variant-list > div.strat-variant-item > span.variant-badge.badge-data

- **Text**: "data"
- **Color**: #E8F6ED on #E8F6ED
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 10px, weight 700
- **Fix**: Change text to #616F66 or adjust for 4.5:1 on bg #E8F6ED

### [CRITICAL] div#strategyColumns > div#col-consequential > div.strat-variant-list > div.strat-variant-item > span.variant-badge.badge-data

- **Text**: "data"
- **Color**: #E8F6ED on #E8F6ED
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 10px, weight 700
- **Fix**: Change text to #616F66 or adjust for 4.5:1 on bg #E8F6ED

### [CRITICAL] div#taxonomySection > div#strategyColumns > div#col-hourly > div.strat-col-header.strat-col-header-2 > h4

- **Text**: "Strategy 2Hourly Matching (Same-ISO)"
- **Color**: #F0F8FA on #F0F8FA
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #878F91 or adjust for 3:1 on bg #F0F8FA

### [CRITICAL] div#strategyColumns > div#col-hourly > div.strat-col-header.strat-col-header-2 > h4 > span.strat-subtitle

- **Text**: "Hourly Matching (Same-ISO)"
- **Color**: #F0F8FA on #F0F8FA
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #697173 or adjust for 4.5:1 on bg #F0F8FA

### [CRITICAL] div#strategyColumns > div#col-hourly > div.strat-variant-list > div.strat-variant-item.selected > span.variant-badge.badge-data

- **Text**: "data"
- **Color**: #DFF3EC on #DFF3EC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 10px, weight 700
- **Fix**: Change text to #5D716A or adjust for 4.5:1 on bg #DFF3EC

### [CRITICAL] div#strategyColumns > div#col-hourly > div.strat-variant-list > div.strat-variant-item > span.variant-badge.badge-pending

- **Text**: "pending"
- **Color**: #FEF5E7 on #FEF5E7
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 10px, weight 700
- **Fix**: Change text to #776E60 or adjust for 4.5:1 on bg #FEF5E7

### [CRITICAL] div#strategyColumns > div#col-hourly > div.strat-variant-list > div.strat-variant-item.selected > span.variant-badge.badge-data

- **Text**: "data"
- **Color**: #DFF3EC on #DFF3EC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 10px, weight 700
- **Fix**: Change text to #5D716A or adjust for 4.5:1 on bg #DFF3EC

### [CRITICAL] div#taxonomySection > div#strategyColumns > div#col-annual > div.strat-col-header.strat-col-header-3 > h4

- **Text**: "Strategy 3Annual Matching"
- **Color**: #FDF7F0 on #FDF7F0
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #948E87 or adjust for 3:1 on bg #FDF7F0

### [CRITICAL] div#strategyColumns > div#col-annual > div.strat-col-header.strat-col-header-3 > h4 > span.strat-subtitle

- **Text**: "Annual Matching"
- **Color**: #FDF7F0 on #FDF7F0
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #767069 or adjust for 4.5:1 on bg #FDF7F0

### [CRITICAL] div#strategyColumns > div#col-annual > div.strat-variant-list > div.strat-variant-item > span.variant-badge.badge-pending

- **Text**: "pending"
- **Color**: #FEF5E7 on #FEF5E7
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 10px, weight 700
- **Fix**: Change text to #776E60 or adjust for 4.5:1 on bg #FEF5E7

### [CRITICAL] div#strategyColumns > div#col-annual > div.strat-variant-list > div.strat-variant-item > span.variant-badge.badge-pending

- **Text**: "pending"
- **Color**: #FEF5E7 on #FEF5E7
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 10px, weight 700
- **Fix**: Change text to #776E60 or adjust for 4.5:1 on bg #FEF5E7

### [CRITICAL] div#strategyColumns > div#col-annual > div.strat-variant-list > div.strat-variant-item > span.variant-badge.badge-pending

- **Text**: "pending"
- **Color**: #FEF5E7 on #FEF5E7
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 10px, weight 700
- **Fix**: Change text to #776E60 or adjust for 4.5:1 on bg #FEF5E7

### [CRITICAL] div#strategyColumns > div#col-annual > div.strat-variant-list > div.strat-variant-item.selected > span.variant-badge.badge-data

- **Text**: "data"
- **Color**: #DFF3EC on #DFF3EC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 10px, weight 700
- **Fix**: Change text to #5D716A or adjust for 4.5:1 on bg #DFF3EC

### [CRITICAL] html > body > div#participationBar > div.participation-readout > span.pct-label

- **Text**: "% of C&I load"
- **Color**: #999999 on #FFFFFF
- **Ratio**: 2.85:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #767676 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > div.scroll-section > div.figure-block > div.figure-card > h3

- **Text**: "At Low Participation, All Strategies Cost Roughly the Same"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 17px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] body > div.scroll-section > div.figure-block > div.figure-card > p.chart-note

- **Text**: "Effective cost per MWh at 90% CFE threshold under each strategy variant.
       "
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section > div.figure-block > div.figure-card > div.callout-box > p

- **Text**: "The cheap strategies aren't actually cheap — they're deferring costs to the futu"
- **Color**: #FEF9F0 on #FEF9F0
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #777269 or adjust for 4.5:1 on bg #FEF9F0

### [CRITICAL] div.figure-block > div.figure-card > div.callout-box > p > strong

- **Text**: "The cheap strategies aren't actually cheap"
- **Color**: #FEF9F0 on #FEF9F0
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #777269 or adjust for 4.5:1 on bg #FEF9F0

### [CRITICAL] body > div.scroll-section > div.figure-block > div.figure-card > h3

- **Text**: "Where the Money Goes"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 17px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] body > div.scroll-section > div.figure-block > div.figure-card > p.chart-note

- **Text**: "Share of total clean energy investment allocated to each ISO under cross-regiona"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section > div.figure-block > div.figure-card > div.callout-box > p

- **Text**: "Cross-regional strategies concentrate capital in regions with the cheapest abate"
- **Color**: #FEF9F0 on #FEF9F0
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #777269 or adjust for 4.5:1 on bg #FEF9F0

### [CRITICAL] div.figure-block > div.figure-card > div.callout-box > p > strong

- **Text**: "Cross-regional strategies concentrate capital"
- **Color**: #FEF9F0 on #FEF9F0
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #777269 or adjust for 4.5:1 on bg #FEF9F0

### [CRITICAL] body > div.scroll-section > div.figure-block > div.figure-card > h3

- **Text**: "Every Zero-Marginal-Cost MWh Pushes Prices Down"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 17px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] body > div.scroll-section > div.figure-block > div.figure-card > p.chart-note

- **Text**: "Estimated wholesale LMP trajectory as clean energy penetration increases under e"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section > div.figure-block > div.figure-card > div.page-insight-box > p

- **Text**: "Its premium creates a revenue floor for existing generators rather than flooding"
- **Color**: #F3FBFE on #F3FBFE
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #6C7477 or adjust for 4.5:1 on bg #F3FBFE

### [CRITICAL] div.figure-block > div.figure-card > div.page-insight-box > p > strong

- **Text**: "Strategy 2C is the only variant with a built-in mechanism to prevent wholesale d"
- **Color**: #F3FBFE on #F3FBFE
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #6C7477 or adjust for 4.5:1 on bg #F3FBFE

### [CRITICAL] body > div.scroll-section > div.figure-block > div.figure-card > h3

- **Text**: "The Coal Wall"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 17px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] body > div.scroll-section > div.figure-block > div.figure-card > p.chart-note

- **Text**: "Marginal abatement cost for the next ton of CO₂ displaced under each strategy.
 "
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section > div.figure-block > div.figure-card > div.callout-box > p

- **Text**: "Once you've displaced all the coal,
                the next ton costs 3–5× more"
- **Color**: #FEF9F0 on #FEF9F0
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #777269 or adjust for 4.5:1 on bg #FEF9F0

### [CRITICAL] div.figure-block > div.figure-card > div.callout-box > p > strong

- **Text**: "The coal wall is a cliff, not a hill."
- **Color**: #FEF9F0 on #FEF9F0
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #777269 or adjust for 4.5:1 on bg #FEF9F0

### [CRITICAL] body > div.scroll-section > div.figure-block > div.figure-card > h3

- **Text**: "Different Strategies Build Different Grids"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 17px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] body > div.scroll-section > div.figure-block > div.figure-card > p.chart-note

- **Text**: "Resource mix composition at 90% CFE threshold for representative strategies.
   "
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section > div.figure-block > div.figure-card > div.page-insight-box > p

- **Text**: "Hourly matching is the only approach that forces investment in the resources you"
- **Color**: #F3FBFE on #F3FBFE
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #6C7477 or adjust for 4.5:1 on bg #F3FBFE

### [CRITICAL] div.figure-block > div.figure-card > div.page-insight-box > p > strong

- **Text**: "Hourly matching is the only approach that forces investment in the resources you"
- **Color**: #F3FBFE on #F3FBFE
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #6C7477 or adjust for 4.5:1 on bg #F3FBFE

### [CRITICAL] body > div#strandingSection > div.figure-block > div.figure-card > h3

- **Text**: "The Stranding Paradox: 2A vs 2B vs 2C"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 17px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] body > div#strandingSection > div.figure-block > div.figure-card > p.chart-note

- **Text**: "Six dimensions compared across the three hourly matching variants.
             "
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.table-scroll-wrapper > table#comparisonTable > thead > tr.row-visible > th

- **Text**: "Dimension"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.table-scroll-wrapper > table#comparisonTable > thead > tr.row-visible > th.col-2a

- **Text**: "2A — All New"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.table-scroll-wrapper > table#comparisonTable > thead > tr.row-visible > th.col-2b

- **Text**: "2B — Grid Baseline"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.table-scroll-wrapper > table#comparisonTable > thead > tr.row-visible > th.col-2c

- **Text**: "2C — Premium + New"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.table-scroll-wrapper > table#comparisonTable > tbody > tr > td

- **Text**: "Wholesale erosion"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.table-scroll-wrapper > table#comparisonTable > tbody > tr > td.cell-bad

- **Text**: "Accelerates locally"
- **Color**: #FDF2F2 on #FDF2F2
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 600
- **Fix**: Change text to #766B6B or adjust for 4.5:1 on bg #FDF2F2

### [CRITICAL] div.table-scroll-wrapper > table#comparisonTable > tbody > tr > td.cell-mixed

- **Text**: "Moderate"
- **Color**: #FEF9F0 on #FEF9F0
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 600
- **Fix**: Change text to #777269 or adjust for 4.5:1 on bg #FEF9F0

### [CRITICAL] div.table-scroll-wrapper > table#comparisonTable > tbody > tr > td.cell-good

- **Text**: "Revenue floor mitigates"
- **Color**: #F1F9F4 on #F1F9F4
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 600
- **Fix**: Change text to #6A726D or adjust for 4.5:1 on bg #F1F9F4

### [CRITICAL] div.table-scroll-wrapper > table#comparisonTable > tbody > tr > td

- **Text**: "Existing clean stranding"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.table-scroll-wrapper > table#comparisonTable > tbody > tr > td.cell-mixed

- **Text**: "Ignores — no help, no harm"
- **Color**: #FEF9F0 on #FEF9F0
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 600
- **Fix**: Change text to #777269 or adjust for 4.5:1 on bg #FEF9F0

### [CRITICAL] div.table-scroll-wrapper > table#comparisonTable > tbody > tr > td.cell-worst

- **Text**: "Worst — claims credit without paying"
- **Color**: #FBE5E5 on #FBE5E5
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #796363 or adjust for 4.5:1 on bg #FBE5E5

### [CRITICAL] div.table-scroll-wrapper > table#comparisonTable > tbody > tr > td.cell-good

- **Text**: "Addressed — premium keeps plants online"
- **Color**: #F1F9F4 on #F1F9F4
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 600
- **Fix**: Change text to #6A726D or adjust for 4.5:1 on bg #F1F9F4

### [CRITICAL] div.table-scroll-wrapper > table#comparisonTable > tbody > tr > td

- **Text**: "Learning curve signal"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.table-scroll-wrapper > table#comparisonTable > tbody > tr > td.cell-good

- **Text**: "Strong — maximum new deployment"
- **Color**: #F1F9F4 on #F1F9F4
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 600
- **Fix**: Change text to #6A726D or adjust for 4.5:1 on bg #F1F9F4

### [CRITICAL] div.table-scroll-wrapper > table#comparisonTable > tbody > tr > td.cell-mixed

- **Text**: "Diluted — less new build needed"
- **Color**: #FEF9F0 on #FEF9F0
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 600
- **Fix**: Change text to #777269 or adjust for 4.5:1 on bg #FEF9F0

### [CRITICAL] div.table-scroll-wrapper > table#comparisonTable > tbody > tr > td.cell-good

- **Text**: "Strong — new build layered on top"
- **Color**: #F1F9F4 on #F1F9F4
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 600
- **Fix**: Change text to #6A726D or adjust for 4.5:1 on bg #F1F9F4

### [CRITICAL] div.table-scroll-wrapper > table#comparisonTable > tbody > tr > td

- **Text**: "Cost trajectory"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.table-scroll-wrapper > table#comparisonTable > tbody > tr > td.cell-bad

- **Text**: "Highest near-term cost"
- **Color**: #FDF2F2 on #FDF2F2
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 600
- **Fix**: Change text to #766B6B or adjust for 4.5:1 on bg #FDF2F2

### [CRITICAL] div.table-scroll-wrapper > table#comparisonTable > tbody > tr > td.cell-good

- **Text**: "Lowest — free-rides on existing"
- **Color**: #F1F9F4 on #F1F9F4
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 600
- **Fix**: Change text to #6A726D or adjust for 4.5:1 on bg #F1F9F4

### [CRITICAL] div.table-scroll-wrapper > table#comparisonTable > tbody > tr > td.cell-mixed

- **Text**: "Moderate — premium + new build"
- **Color**: #FEF9F0 on #FEF9F0
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 600
- **Fix**: Change text to #777269 or adjust for 4.5:1 on bg #FEF9F0

### [CRITICAL] div.table-scroll-wrapper > table#comparisonTable > tbody > tr > td

- **Text**: "Firm investment signal"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.table-scroll-wrapper > table#comparisonTable > tbody > tr > td.cell-good

- **Text**: "Builds firm + storage from day one"
- **Color**: #F1F9F4 on #F1F9F4
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 600
- **Fix**: Change text to #6A726D or adjust for 4.5:1 on bg #F1F9F4

### [CRITICAL] div.table-scroll-wrapper > table#comparisonTable > tbody > tr > td.cell-mixed

- **Text**: "Reduces requirement for firm"
- **Color**: #FEF9F0 on #FEF9F0
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 600
- **Fix**: Change text to #777269 or adjust for 4.5:1 on bg #FEF9F0

### [CRITICAL] div.table-scroll-wrapper > table#comparisonTable > tbody > tr > td.cell-good

- **Text**: "Preserves existing + builds new firm"
- **Color**: #F1F9F4 on #F1F9F4
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 600
- **Fix**: Change text to #6A726D or adjust for 4.5:1 on bg #F1F9F4

### [CRITICAL] div.table-scroll-wrapper > table#comparisonTable > tbody > tr > td

- **Text**: "Additionality"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.table-scroll-wrapper > table#comparisonTable > tbody > tr > td.cell-good

- **Text**: "100% additional by definition"
- **Color**: #F1F9F4 on #F1F9F4
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 600
- **Fix**: Change text to #6A726D or adjust for 4.5:1 on bg #F1F9F4

### [CRITICAL] div.table-scroll-wrapper > table#comparisonTable > tbody > tr > td.cell-bad

- **Text**: "Reduced — existing clean offsets requirement"
- **Color**: #FDF2F2 on #FDF2F2
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 600
- **Fix**: Change text to #766B6B or adjust for 4.5:1 on bg #FDF2F2

### [CRITICAL] div.table-scroll-wrapper > table#comparisonTable > tbody > tr > td.cell-good

- **Text**: "Existing preserved + new added on top"
- **Color**: #F1F9F4 on #F1F9F4
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 600
- **Fix**: Change text to #6A726D or adjust for 4.5:1 on bg #F1F9F4

### [CRITICAL] div#strandingSection > div.figure-block > div.figure-card > div.callout-box > p

- **Text**: "—
                it actively takes credit for existing clean generation without"
- **Color**: #FEF9F0 on #FEF9F0
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #777269 or adjust for 4.5:1 on bg #FEF9F0

### [CRITICAL] div.figure-block > div.figure-card > div.callout-box > p > strong

- **Text**: "Strategy 2B is arguably the worst hourly variant"
- **Color**: #FEF9F0 on #FEF9F0
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #777269 or adjust for 4.5:1 on bg #FEF9F0

### [CRITICAL] div.section-dark-inner > div.scroll-section > div.figure-block > div.figure-card > h3

- **Text**: "Strategy 2C Has the Opposite Problem"
- **Color**: #F3F3F4 on #F3F3F4
- **Ratio**: 1:1 (need 3:1)
- **Font**: 17px, weight 700
- **Fix**: Change text to #8A8A8B or adjust for 3:1 on bg #F3F3F4

### [CRITICAL] div.section-dark-inner > div.scroll-section > div.figure-block > div.figure-card > p.chart-note

- **Text**: "Every other strategy degrades as adoption increases. Strategy 2C needs
         "
- **Color**: #F3F3F4 on #F3F3F4
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #6C6C6D or adjust for 4.5:1 on bg #F3F3F4

### [CRITICAL] div.scroll-section > div.figure-block > div.figure-card > p.chart-note > em

- **Text**: "enough"
- **Color**: #F3F3F4 on #F3F3F4
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #6C6C6D or adjust for 4.5:1 on bg #F3F3F4

### [CRITICAL] div.figure-block > div.figure-card > div.zone-diagram > div.zone.zone-maintenance > h4

- **Text**: "Maintenance Mode"
- **Color**: #F1E7E7 on #F1E7E7
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #8D8383 or adjust for 3:1 on bg #F1E7E7

### [CRITICAL] div.figure-block > div.figure-card > div.zone-diagram > div.zone.zone-maintenance > p

- **Text**: "Below critical mass. Premium spend dominates. Not enough new-build volume to tri"
- **Color**: #F1E7E7 on #F1E7E7
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #6F6565 or adjust for 4.5:1 on bg #F1E7E7

### [CRITICAL] div.figure-block > div.figure-card > div.zone-diagram > div.zone.zone-learning > h4

- **Text**: "Learning Activated"
- **Color**: #E6EEEA on #E6EEEA
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #7D8581 or adjust for 3:1 on bg #E6EEEA

### [CRITICAL] div.figure-block > div.figure-card > div.zone-diagram > div.zone.zone-learning > p

- **Text**: "Past critical mass. Aggregate new-build firm hits first Wright's Law doubling. N"
- **Color**: #E6EEEA on #E6EEEA
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #646C68 or adjust for 4.5:1 on bg #E6EEEA

### [CRITICAL] div.scroll-section > div.figure-block > div.figure-card > div.page-insight-box > p

- **Text**: "— because learning is global.
                A nuclear plant built in PJM drive"
- **Color**: #E7EFF3 on #E7EFF3
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #656D71 or adjust for 4.5:1 on bg #E7EFF3

### [CRITICAL] div.figure-block > div.figure-card > div.page-insight-box > p > strong

- **Text**: "The critical mass threshold is lower than you'd think"
- **Color**: #E7EFF3 on #E7EFF3
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #656D71 or adjust for 4.5:1 on bg #E7EFF3

### [CRITICAL] div.section-dark-inner > div.scroll-section > div.figure-block > div.figure-card > h3

- **Text**: "Different Regions Play Different Roles Under Strategy 2C"
- **Color**: #F3F3F4 on #F3F3F4
- **Ratio**: 1:1 (need 3:1)
- **Font**: 17px, weight 700
- **Fix**: Change text to #8A8A8B or adjust for 3:1 on bg #F3F3F4

### [CRITICAL] div.section-dark-inner > div.scroll-section > div.figure-block > div.figure-card > p.chart-note

- **Text**: "Nuclear-heavy ISOs are the premium-payers — their spend keeps existing clean ali"
- **Color**: #F3F3F4 on #F3F3F4
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #6C6C6D or adjust for 4.5:1 on bg #F3F3F4

### [CRITICAL] div.scroll-section > div.figure-block > div.figure-card > div.callout-box > p

- **Text**: "Regions with large existing clean fleets pay premiums to prevent stranding.
    "
- **Color**: #F3EEE6 on #F3EEE6
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #716C64 or adjust for 4.5:1 on bg #F3EEE6

### [CRITICAL] div.figure-block > div.figure-card > div.callout-box > p > strong

- **Text**: "Each region contributes differently to the same NOAK outcome."
- **Color**: #F3EEE6 on #F3EEE6
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #716C64 or adjust for 4.5:1 on bg #F3EEE6

### [CRITICAL] body > div.scroll-section > div.figure-block > div.figure-card > h3

- **Text**: "The FOAK→NOAK Timeline"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 17px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] body > div.scroll-section > div.figure-block > div.figure-card > p.chart-note

- **Text**: "Clean firm LCOE trajectory under two procurement regimes. SBTi milestones mark
 "
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section > div.figure-block > div.figure-card > div.callout-box > p

- **Text**: "Consequential strategies are still paying near-FOAK —
                for the sa"
- **Color**: #FEF9F0 on #FEF9F0
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #777269 or adjust for 4.5:1 on bg #FEF9F0

### [CRITICAL] div.figure-block > div.figure-card > div.callout-box > p > strong

- **Text**: "By 2040, hourly matching has driven clean firm costs to NOAK."
- **Color**: #FEF9F0 on #FEF9F0
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #777269 or adjust for 4.5:1 on bg #FEF9F0

### [CRITICAL] div.section-dark-inner > div#compoundingSection > div.figure-block > div.figure-card > h3

- **Text**: "Three Failures Compound on the SBTi Timeline"
- **Color**: #F3F3F4 on #F3F3F4
- **Ratio**: 1:1 (need 3:1)
- **Font**: 17px, weight 700
- **Fix**: Change text to #8A8A8B or adjust for 3:1 on bg #F3F3F4

### [CRITICAL] div.section-dark-inner > div#compoundingSection > div.figure-block > div.figure-card > p.chart-note

- **Text**: "Learning delay, stranded VRE overbuild, and gas lock-in create a compounding div"
- **Color**: #F3F3F4 on #F3F3F4
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #6C6C6D or adjust for 4.5:1 on bg #F3F3F4

### [CRITICAL] div.table-scroll-wrapper > table#timelineTable > thead > tr.row-visible > th

- **Text**: "Milestone"
- **Color**: #F3F3F4 on #F3F3F4
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #6C6C6D or adjust for 4.5:1 on bg #F3F3F4

### [CRITICAL] div.table-scroll-wrapper > table#timelineTable > thead > tr.row-visible > th.col-delayed

- **Text**: "Strategy 1/3Consequential / Annual"
- **Color**: #F3F3F4 on #F3F3F4
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #6C6C6D or adjust for 4.5:1 on bg #F3F3F4

### [CRITICAL] table#timelineTable > thead > tr.row-visible > th.col-delayed > small

- **Text**: "Consequential / Annual"
- **Color**: #F3F3F4 on #F3F3F4
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #6C6C6D or adjust for 4.5:1 on bg #F3F3F4

### [CRITICAL] div.table-scroll-wrapper > table#timelineTable > thead > tr.row-visible > th.col-hourly

- **Text**: "Strategy 2Hourly Matching"
- **Color**: #F3F3F4 on #F3F3F4
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #6C6C6D or adjust for 4.5:1 on bg #F3F3F4

### [CRITICAL] table#timelineTable > thead > tr.row-visible > th.col-hourly > small

- **Text**: "Hourly Matching"
- **Color**: #F3F3F4 on #F3F3F4
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #6C6C6D or adjust for 4.5:1 on bg #F3F3F4

### [CRITICAL] div.table-scroll-wrapper > table#timelineTable > tbody > tr > td

- **Text**: "2030SBTi 50%"
- **Color**: #F3F3F4 on #F3F3F4
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #8A8A8B or adjust for 3:1 on bg #F3F3F4

### [CRITICAL] table#timelineTable > tbody > tr > td > span.milestone-label

- **Text**: "SBTi 50%"
- **Color**: #F3F3F4 on #F3F3F4
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 400
- **Fix**: Change text to #6C6C6D or adjust for 4.5:1 on bg #F3F3F4

### [CRITICAL] div.table-scroll-wrapper > table#timelineTable > tbody > tr > td.td-green

- **Text**: "Cheap — lots of VRE, looks great on paper"
- **Color**: #E6EEEA on #E6EEEA
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #646C68 or adjust for 4.5:1 on bg #E6EEEA

### [CRITICAL] div.table-scroll-wrapper > table#timelineTable > tbody > tr > td.td-amber

- **Text**: "Slightly more expensive — investing in firm + storage"
- **Color**: #F3EEE6 on #F3EEE6
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #716C64 or adjust for 4.5:1 on bg #F3EEE6

### [CRITICAL] div.table-scroll-wrapper > table#timelineTable > tbody > tr > td

- **Text**: "2035SBTi ~70%"
- **Color**: #F3F3F4 on #F3F3F4
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #8A8A8B or adjust for 3:1 on bg #F3F3F4

### [CRITICAL] table#timelineTable > tbody > tr > td > span.milestone-label

- **Text**: "SBTi ~70%"
- **Color**: #F3F3F4 on #F3F3F4
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 400
- **Fix**: Change text to #6C6C6D or adjust for 4.5:1 on bg #F3F3F4

### [CRITICAL] div.table-scroll-wrapper > table#timelineTable > tbody > tr > td.td-green

- **Text**: "Still cheap — more VRE, gas fills gaps"
- **Color**: #E6EEEA on #E6EEEA
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #646C68 or adjust for 4.5:1 on bg #E6EEEA

### [CRITICAL] div.table-scroll-wrapper > table#timelineTable > tbody > tr > td.td-amber

- **Text**: "Firm clean hitting learning curve, storage displacing gas"
- **Color**: #F3EEE6 on #F3EEE6
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #716C64 or adjust for 4.5:1 on bg #F3EEE6

### [CRITICAL] div.table-scroll-wrapper > table#timelineTable > tbody > tr > td

- **Text**: "2040SBTi 90%"
- **Color**: #F3F3F4 on #F3F3F4
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #8A8A8B or adjust for 3:1 on bg #F3F3F4

### [CRITICAL] table#timelineTable > tbody > tr > td > span.milestone-label

- **Text**: "SBTi 90%"
- **Color**: #F3F3F4 on #F3F3F4
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 400
- **Fix**: Change text to #6C6C6D or adjust for 4.5:1 on bg #F3F3F4

### [CRITICAL] div.table-scroll-wrapper > table#timelineTable > tbody > tr > td.td-red

- **Text**: "WALL — VRE saturated, firm at FOAK, gas locked in"
- **Color**: #F1E3E3 on #F1E3E3
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 600
- **Fix**: Change text to #6F6161 or adjust for 4.5:1 on bg #F1E3E3

### [CRITICAL] div.table-scroll-wrapper > table#timelineTable > tbody > tr > td.td-green

- **Text**: "Firm at NOAK, storage mature, gas already retiring"
- **Color**: #E6EEEA on #E6EEEA
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #646C68 or adjust for 4.5:1 on bg #E6EEEA

### [CRITICAL] div.table-scroll-wrapper > table#timelineTable > tbody > tr > td

- **Text**: "2050Net-Zero 100%"
- **Color**: #F3F3F4 on #F3F3F4
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #8A8A8B or adjust for 3:1 on bg #F3F3F4

### [CRITICAL] table#timelineTable > tbody > tr > td > span.milestone-label

- **Text**: "Net-Zero 100%"
- **Color**: #F3F3F4 on #F3F3F4
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 400
- **Fix**: Change text to #6C6C6D or adjust for 4.5:1 on bg #F3F3F4

### [CRITICAL] div.table-scroll-wrapper > table#timelineTable > tbody > tr > td.td-red

- **Text**: "Scramble — paying FOAK for firm, retiring gas at huge cost, stranded VRE"
- **Color**: #F1E3E3 on #F1E3E3
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 600
- **Fix**: Change text to #6F6161 or adjust for 4.5:1 on bg #F1E3E3

### [CRITICAL] div.table-scroll-wrapper > table#timelineTable > tbody > tr > td.td-green

- **Text**: "Smooth glide — infrastructure already in place"
- **Color**: #E6EEEA on #E6EEEA
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #646C68 or adjust for 4.5:1 on bg #E6EEEA

### [CRITICAL] div.scroll-section > div.horse-race-container > div.race-grid > div.figure-card > h3

- **Text**: "Cost to Reach 90% CFE"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 17px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section > div.horse-race-container > div.race-grid > div.figure-card > p.chart-note

- **Text**: "What does each strategy cost to achieve the same level of hourly clean energy ma"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section > div.horse-race-container > div.race-grid > div.figure-card > h3

- **Text**: "What Does $60/MWh Get You?"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 17px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section > div.horse-race-container > div.race-grid > div.figure-card > p.chart-note

- **Text**: "Same budget, different outcomes. Which strategy achieves the most
              "
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > div.scroll-section > div.figure-block > div.figure-card > h3

- **Text**: "Required Participation by Strategy"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 17px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] body > div.scroll-section > div.figure-block > div.figure-card > p.chart-note

- **Text**: "What C&I participation rate would each strategy need to achieve a meaningful
   "
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section > div.figure-block > div.figure-card > div > label

- **Text**: "CO₂ Reduction Target"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.figure-block > div.figure-card > div > div > span#co2TargetValue

- **Text**: "30%"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 19px, weight 800
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] body > div.scroll-section > div.explorer-panel > div.figure-card > h3

- **Text**: "Explore the Full Parameter Space"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 17px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] body > div.scroll-section > div.explorer-panel > div.figure-card > p.chart-note

- **Text**: "Select strategies, adjust participation and targets, compare across ISOs."
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.figure-card > div.explorer-controls > div.controls-grid > div.control-group > label

- **Text**: "Strategies to Compare"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.explorer-controls > div.controls-grid > div.control-group > div#explorerStrategies > label.strat-check.active

- **Text**: "1A"
- **Color**: #ECF8FD on #ECF8FD
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #657176 or adjust for 4.5:1 on bg #ECF8FD

### [CRITICAL] div.explorer-controls > div.controls-grid > div.control-group > div#explorerStrategies > label.strat-check

- **Text**: "1B"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.explorer-controls > div.controls-grid > div.control-group > div#explorerStrategies > label.strat-check

- **Text**: "1C"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.explorer-controls > div.controls-grid > div.control-group > div#explorerStrategies > label.strat-check.active

- **Text**: "2A"
- **Color**: #ECF8FD on #ECF8FD
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #657176 or adjust for 4.5:1 on bg #ECF8FD

### [CRITICAL] div.explorer-controls > div.controls-grid > div.control-group > div#explorerStrategies > label.strat-check

- **Text**: "2B"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.explorer-controls > div.controls-grid > div.control-group > div#explorerStrategies > label.strat-check.active

- **Text**: "2C"
- **Color**: #ECF8FD on #ECF8FD
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #657176 or adjust for 4.5:1 on bg #ECF8FD

### [CRITICAL] div.explorer-controls > div.controls-grid > div.control-group > div#explorerStrategies > label.strat-check

- **Text**: "3A"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.explorer-controls > div.controls-grid > div.control-group > div#explorerStrategies > label.strat-check

- **Text**: "3B"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.explorer-controls > div.controls-grid > div.control-group > div#explorerStrategies > label.strat-check

- **Text**: "3C"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.explorer-controls > div.controls-grid > div.control-group > div#explorerStrategies > label.strat-check.active

- **Text**: "3D"
- **Color**: #ECF8FD on #ECF8FD
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #657176 or adjust for 4.5:1 on bg #ECF8FD

### [CRITICAL] div.figure-card > div.explorer-controls > div.controls-grid > div.control-group > label

- **Text**: "ISO Region"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.figure-card > div.explorer-controls > div.controls-grid > div.control-group > label

- **Text**: "CFE Threshold"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.explorer-controls > div.controls-grid > div.control-group > div > span#explorerThresholdVal

- **Text**: "90%"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 16px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div.figure-card > div.explorer-controls > div.controls-grid > div.control-group > label

- **Text**: "Learning Curves"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.explorer-controls > div.controls-grid > div.control-group > div > button#lcOn

- **Text**: "On"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 16px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.explorer-controls > div.controls-grid > div.control-group > div > button#lcOff

- **Text**: "Off"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 16px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.explorer-panel > div.figure-card > div.explorer-results > div.figure-card > h3

- **Text**: "Cost ($/MWh)"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.explorer-panel > div.figure-card > div.explorer-results > div.figure-card > h3

- **Text**: "Resource Mix"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.explorer-panel > div.figure-card > div.explorer-results > div.figure-card > h3

- **Text**: "CO₂ Abated (Mt/yr)"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > div#taxonomySection > div.fig-badge

- **Text**: "Figure 1 — Strategy Taxonomy"
- **Color**: #ECF8FD on #ECF8FD
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 700
- **Fix**: Change text to #657176 or adjust for 4.5:1 on bg #ECF8FD

### [CRITICAL] html > body > section#findingsHero > div.findings-inner > div.findings-label

- **Text**: "Key Findings"
- **Color**: #000000 on #162440
- **Ratio**: 1.36:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #8C8C8C or adjust for 4.5:1 on bg #162440

### [CRITICAL] html > body > div.act-divider.scroll-section > div.act-title

- **Text**: "What Breaks, and When"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 29px, weight 800
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > div.act-divider.scroll-section > div.act-subtitle

- **Text**: "Drag the participation slider from today's level to 80%.
        Watch every fai"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > div.scroll-section > div.figure-block > div.figure-card > div.fig-badge

- **Text**: "Figure 2 — Cost Divergence"
- **Color**: #ECF8FD on #ECF8FD
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 700
- **Fix**: Change text to #657176 or adjust for 4.5:1 on bg #ECF8FD

### [CRITICAL] body > div.scroll-section > div.figure-block > div.figure-card > div.fig-badge

- **Text**: "Figure 3 — Capital Allocation"
- **Color**: #ECF8FD on #ECF8FD
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 700
- **Fix**: Change text to #657176 or adjust for 4.5:1 on bg #ECF8FD

### [CRITICAL] body > div.scroll-section > div.figure-block > div.figure-card > div.fig-badge

- **Text**: "Figure 4 — Wholesale Erosion"
- **Color**: #ECF8FD on #ECF8FD
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 700
- **Fix**: Change text to #657176 or adjust for 4.5:1 on bg #ECF8FD

### [CRITICAL] body > div.scroll-section > div.figure-block > div.figure-card > div.fig-badge

- **Text**: "Figure 5 — MAC Escalation"
- **Color**: #ECF8FD on #ECF8FD
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 700
- **Fix**: Change text to #657176 or adjust for 4.5:1 on bg #ECF8FD

### [CRITICAL] body > div.scroll-section > div.figure-block > div.figure-card > div.fig-badge

- **Text**: "Figure 6 — Resource Mix"
- **Color**: #ECF8FD on #ECF8FD
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 700
- **Fix**: Change text to #657176 or adjust for 4.5:1 on bg #ECF8FD

### [CRITICAL] html > body > div.act-divider.scroll-section > div.act-title

- **Text**: "The Debate Within Hourly"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 29px, weight 800
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > div.act-divider.scroll-section > div.act-subtitle

- **Text**: "If hourly matching is the most robust family, the next question is: which versio"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > div#strandingSection > div.figure-block > div.figure-card > div.fig-badge

- **Text**: "Figure 7 — Strategy 2 Internal Comparison"
- **Color**: #ECF8FD on #ECF8FD
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 700
- **Fix**: Change text to #657176 or adjust for 4.5:1 on bg #ECF8FD

### [CRITICAL] div.section-dark-inner > div.scroll-section > div.figure-block > div.figure-card > div.fig-badge

- **Text**: "Figure 8 — Critical Mass Threshold"
- **Color**: #E0EDF3 on #E0EDF3
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 700
- **Fix**: Change text to #5E6B71 or adjust for 4.5:1 on bg #E0EDF3

### [CRITICAL] div.section-dark-inner > div.scroll-section > div.figure-block > div.figure-card > div.fig-badge

- **Text**: "Figure 9 — Regional Investment Roles"
- **Color**: #E0EDF3 on #E0EDF3
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 700
- **Fix**: Change text to #5E6B71 or adjust for 4.5:1 on bg #E0EDF3

### [CRITICAL] html > body > div.act-divider-dark.scroll-section > div.act-title

- **Text**: "The Timeline Trap"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 29px, weight 800
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > div.act-divider-dark.scroll-section > div.act-subtitle

- **Text**: "Corporate decarbonization is a 25-year ratchet. What you build at 50% determines"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > div.scroll-section > div.figure-block > div.figure-card > div.fig-badge

- **Text**: "Figure 10 — Learning Curve Divergence"
- **Color**: #ECF8FD on #ECF8FD
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 700
- **Fix**: Change text to #657176 or adjust for 4.5:1 on bg #ECF8FD

### [CRITICAL] div.section-dark-inner > div#compoundingSection > div.figure-block > div.figure-card > div.fig-badge

- **Text**: "Figure 11 — Compounding Timeline"
- **Color**: #E0EDF3 on #E0EDF3
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 700
- **Fix**: Change text to #5E6B71 or adjust for 4.5:1 on bg #E0EDF3

### [CRITICAL] html > body > div.act-divider.scroll-section > div.act-title

- **Text**: "The Horse Race"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 29px, weight 800
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > div.act-divider.scroll-section > div.act-subtitle

- **Text**: "Same outcome — which strategy gets there cheapest?
        Same budget — which s"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section > div.horse-race-container > div.race-grid > div.figure-card > div.fig-badge

- **Text**: "Figure 12 — Fixed Outcome"
- **Color**: #ECF8FD on #ECF8FD
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 700
- **Fix**: Change text to #657176 or adjust for 4.5:1 on bg #ECF8FD

### [CRITICAL] div.scroll-section > div.horse-race-container > div.race-grid > div.figure-card > div.fig-badge

- **Text**: "Figure 13 — Fixed Budget"
- **Color**: #ECF8FD on #ECF8FD
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 700
- **Fix**: Change text to #657176 or adjust for 4.5:1 on bg #ECF8FD

### [CRITICAL] html > body > div.act-divider.scroll-section > div.act-title

- **Text**: "The System View"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 29px, weight 800
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > div.act-divider.scroll-section > div.act-subtitle

- **Text**: "Flip the question: what participation level does each strategy need to hit a sys"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > div.scroll-section > div.figure-block > div.figure-card > div.fig-badge

- **Text**: "Figure 14 — System-Wide Requirements"
- **Color**: #ECF8FD on #ECF8FD
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 700
- **Fix**: Change text to #657176 or adjust for 4.5:1 on bg #ECF8FD

### [CRITICAL] body > div.scroll-section > div.explorer-panel > div.figure-card > div.fig-badge

- **Text**: "Figure 15 — Interactive Explorer"
- **Color**: #ECF8FD on #ECF8FD
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 700
- **Fix**: Change text to #657176 or adjust for 4.5:1 on bg #ECF8FD

### [MINOR] section#findingsHero > div.findings-inner > div.findings-grid > div.finding-card > p

- **Text**: "Cost spreads are negligible below 20% C&I participation. The divergence begins w"
- **Color**: #9198A5 on #24314B
- **Ratio**: 4.48:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #969DAA or adjust for 4.5:1 on bg #24314B

### [MINOR] section#findingsHero > div.findings-inner > div.findings-grid > div.finding-card > p

- **Text**: "Only hourly strategies force investment in firm clean generation and long-durati"
- **Color**: #9198A5 on #24314B
- **Ratio**: 4.48:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #969DAA or adjust for 4.5:1 on bg #24314B

### [MINOR] section#findingsHero > div.findings-inner > div.findings-grid > div.finding-card > p

- **Text**: "Consequential and annual strategies defer learning-curve investment. By 2040, th"
- **Color**: #9198A5 on #24314B
- **Ratio**: 4.48:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #969DAA or adjust for 4.5:1 on bg #24314B

## dashboard/clean_firm_case.html

**12 failures**

### [CRITICAL] section.section-dark > div.section-dark-inner > div.synthesis-inner > div.synthesis-card > h3

- **Text**: "The stranding risk is already here"
- **Color**: #333C4D on #333C4D
- **Ratio**: 1:1 (need 3:1)
- **Font**: 16px, weight 800
- **Fix**: Change text to #7E8798 or adjust for 3:1 on bg #333C4D

### [CRITICAL] section.section-dark > div.section-dark-inner > div.synthesis-inner > div.synthesis-card > p

- **Text**: "145 GW of gas backup capacity across 7 ISOs at 90% clean energy. This incumbent "
- **Color**: #333C4D on #333C4D
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #A1AABB or adjust for 4.5:1 on bg #333C4D

### [CRITICAL] section.section-dark > div.section-dark-inner > div.synthesis-inner > div.synthesis-card > span.stat-callout.stat-red

- **Text**: "145 GW gas backup · Diminishing utilization · Permanent cost"
- **Color**: #4F3D4C on #4F3D4C
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #BDABBA or adjust for 4.5:1 on bg #4F3D4C

### [CRITICAL] section.section-dark > div.section-dark-inner > div.synthesis-inner > div.synthesis-card > h3

- **Text**: "The FOAK premium is an investment, not a cost"
- **Color**: #333C4D on #333C4D
- **Ratio**: 1:1 (need 3:1)
- **Font**: 16px, weight 800
- **Fix**: Change text to #7E8798 or adjust for 3:1 on bg #333C4D

### [CRITICAL] section.section-dark > div.section-dark-inner > div.synthesis-inner > div.synthesis-card > p

- **Text**: "Lazard (2025) puts unsubsidized new nuclear at $141–220/MWh — consistent with Vo"
- **Color**: #333C4D on #333C4D
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #A1AABB or adjust for 4.5:1 on bg #333C4D

### [CRITICAL] section.section-dark > div.section-dark-inner > div.synthesis-inner > div.synthesis-card > span.stat-callout.stat-green

- **Text**: "~40% cost decline FOAK→NOAK · Fervo: 35% learning rate"
- **Color**: #305150 on #305150
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #A3C4C3 or adjust for 4.5:1 on bg #305150

### [CRITICAL] section.hero-overview.section-light > div.hero-inner > div.hero-chart-block > div.chart-panel > div.chart-title

- **Text**: "Gas Backup Capacity at 90% Clean (GW)"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] section.concept-section.section-light > div.concept-inner > div.concept-chart-col > div.chart-panel > div.chart-title

- **Text**: "Gas Backup Cost vs Nuclear LCOE at 90% Clean — All ISOs"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [MINOR] section.foak-hero.section-light > div.foak-inner > div.foak-text > div > span.stat-callout.stat-green

- **Text**: "Nuclear NOAK target: $68–75/MWh (vs. FOAK $135–170)"
- **Color**: #15803D on #E7F5EF
- **Ratio**: 4.46:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #107B38 or adjust for 4.5:1 on bg #E7F5EF

### [MINOR] section.foak-hero.section-light > div.foak-inner > div.foak-text > div > span.stat-callout.stat-red

- **Text**: "Gas true cost by 2040: $85–112/MWh (and rising)"
- **Color**: #DC2626 on #F7EBED
- **Ratio**: 4.14:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #D21C1C or adjust for 4.5:1 on bg #F7EBED

### [MINOR] div.concept-inner > div.concept-chart-col > div.chart-panel > div#gasTrapISO > button.iso-btn.active

- **Text**: "PJM"
- **Color**: #0284C7 on #E2F4FC
- **Ratio**: 3.63:1 (need 4.5:1)
- **Font**: 11px, weight 600
- **Fix**: Change text to #0070B3 or adjust for 4.5:1 on bg #E2F4FC

### [MINOR] div.concept-inner > div.concept-chart-col > div.chart-panel > div#displacementISO > button.iso-btn.active

- **Text**: "PJM"
- **Color**: #0284C7 on #E2F4FC
- **Ratio**: 3.63:1 (need 4.5:1)
- **Font**: 11px, weight 600
- **Fix**: Change text to #0070B3 or adjust for 4.5:1 on bg #E2F4FC

## dashboard/consequential_accounting.html

**340 failures**

### [CRITICAL] section.hero-overview.section-light > div.hero-inner > div.comparison-visual > div.compare-box.attr > p

- **Text**: "Current Scope 2 MBM✗ No temporal matching✗ No deliverability• Lowest cost & barr"
- **Color**: #F47C7C on #FFFFFF
- **Ratio**: 2.62:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #C74F4F or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] section.hero-overview.section-light > div.hero-inner > div.comparison-visual > div.compare-box.hourly > p

- **Text**: "Proposed Scope 2 Revision✓ Hourly granularity✓ Deliverability regions✓ Drives fi"
- **Color**: #56C0F0 on #FFFFFF
- **Ratio**: 2.06:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #107AAA or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] section.hero-overview.section-light > div.hero-inner > div.comparison-visual > div.compare-box.conseq > p

- **Text**: "Impact-Based Framework✓ Cheapest $/tCO₂✓ Cross-regional optimization✗ No tempora"
- **Color**: #64D68E on #FFFFFF
- **Ratio**: 1.81:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #14863E or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > div.section-header.fade-in > h2

- **Text**: "Three Methods, Three Philosophies"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 3:1)
- **Font**: 19px, weight 800
- **Fix**: Change text to #8F9093 or adjust for 3:1 on bg #F8F9FC

### [CRITICAL] html > body > div.section-header.fade-in > p

- **Text**: "The debate isn't just "attributional vs. consequential." It's a three-way contes"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] body > section.concept-section.section-light > div.concept-grid > div.concept-card > h3

- **Text**: "1. Annual Attribution Current"
- **Color**: #2C3344 on #2C3344
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #777E8F or adjust for 3:1 on bg #2C3344

### [CRITICAL] section.concept-section.section-light > div.concept-grid > div.concept-card > h3 > span.badge.attr

- **Text**: "Current"
- **Color**: #3B3444 on #3B3444
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #A49DAD or adjust for 4.5:1 on bg #3B3444

### [CRITICAL] body > section.concept-section.section-light > div.concept-grid > div.concept-card > p

- **Text**: "The status quo. Buy RECs equal to your annual consumption. No requirement for wh"
- **Color**: #2C3344 on #2C3344
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #959CAD or adjust for 4.5:1 on bg #2C3344

### [CRITICAL] section.concept-section.section-light > div.concept-grid > div.concept-card > ul > li

- **Text**: "GHG Protocol Scope 2 market-based method (2015)"
- **Color**: #2C3344 on #2C3344
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #959CAD or adjust for 4.5:1 on bg #2C3344

### [CRITICAL] section.concept-section.section-light > div.concept-grid > div.concept-card > ul > li

- **Text**: "Annual matching: total RECs ≥ total consumption"
- **Color**: #2C3344 on #2C3344
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #959CAD or adjust for 4.5:1 on bg #2C3344

### [CRITICAL] section.concept-section.section-light > div.concept-grid > div.concept-card > ul > li

- **Text**: "No temporal or locational granularity required"
- **Color**: #2C3344 on #2C3344
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #959CAD or adjust for 4.5:1 on bg #2C3344

### [CRITICAL] section.concept-section.section-light > div.concept-grid > div.concept-card > ul > li

- **Text**: "A New England company buys Texas wind RECs at 2 AM to cover a winter 6 PM gas pe"
- **Color**: #2C3344 on #2C3344
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #959CAD or adjust for 4.5:1 on bg #2C3344

### [CRITICAL] section.concept-section.section-light > div.concept-grid > div.concept-card > ul > li

- **Text**: "Near-zero consequential emissions impact (Xu et al. 2024, WattTime 2024)"
- **Color**: #2C3344 on #2C3344
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #959CAD or adjust for 4.5:1 on bg #2C3344

### [CRITICAL] div.concept-grid > div.concept-card > ul > li > strong

- **Text**: "Near-zero consequential emissions impact"
- **Color**: #2C3344 on #2C3344
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #959CAD or adjust for 4.5:1 on bg #2C3344

### [CRITICAL] body > section.concept-section.section-light > div.concept-grid > div.concept-card > h3

- **Text**: "2. Hourly Attribution Proposed"
- **Color**: #2C3344 on #2C3344
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #777E8F or adjust for 3:1 on bg #2C3344

### [CRITICAL] section.concept-section.section-light > div.concept-grid > div.concept-card > h3 > span.badge.hourly

- **Text**: "Proposed"
- **Color**: #293C51 on #293C51
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #97AABF or adjust for 4.5:1 on bg #293C51

### [CRITICAL] body > section.concept-section.section-light > div.concept-grid > div.concept-card > p

- **Text**: "The proposed Scope 2 revision. Match consumption with CFE hour-by-hour within de"
- **Color**: #2C3344 on #2C3344
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #959CAD or adjust for 4.5:1 on bg #2C3344

### [CRITICAL] section.concept-section.section-light > div.concept-grid > div.concept-card > ul > li

- **Text**: "Proposed GHG Protocol Scope 2 revision (expected ~2027)"
- **Color**: #2C3344 on #2C3344
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #959CAD or adjust for 4.5:1 on bg #2C3344

### [CRITICAL] section.concept-section.section-light > div.concept-grid > div.concept-card > ul > li

- **Text**: "Hourly matching: CFE supply = demand each hour"
- **Color**: #2C3344 on #2C3344
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #959CAD or adjust for 4.5:1 on bg #2C3344

### [CRITICAL] section.concept-section.section-light > div.concept-grid > div.concept-card > ul > li

- **Text**: "Deliverability constraints: same grid region / market boundary"
- **Color**: #2C3344 on #2C3344
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #959CAD or adjust for 4.5:1 on bg #2C3344

### [CRITICAL] section.concept-section.section-light > div.concept-grid > div.concept-card > ul > li

- **Text**: "Reveals need for firm clean power (nuclear, CCS, geothermal) + storage"
- **Color**: #2C3344 on #2C3344
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #959CAD or adjust for 4.5:1 on bg #2C3344

### [CRITICAL] section.concept-section.section-light > div.concept-grid > div.concept-card > ul > li

- **Text**: "Strong consequential impact as a byproduct of physics-aligned accounting"
- **Color**: #2C3344 on #2C3344
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #959CAD or adjust for 4.5:1 on bg #2C3344

### [CRITICAL] div.concept-grid > div.concept-card > ul > li > strong

- **Text**: "Strong consequential impact as a byproduct"
- **Color**: #2C3344 on #2C3344
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #959CAD or adjust for 4.5:1 on bg #2C3344

### [CRITICAL] body > section.concept-section.section-light > div.concept-grid > div.concept-card > h3

- **Text**: "3. Consequential / Impact Impact-Based"
- **Color**: #2C3344 on #2C3344
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #777E8F or adjust for 3:1 on bg #2C3344

### [CRITICAL] section.concept-section.section-light > div.concept-grid > div.concept-card > h3 > span.badge.conseq

- **Text**: "Impact-Based"
- **Color**: #2B3F46 on #2B3F46
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #99ADB4 or adjust for 4.5:1 on bg #2B3F46

### [CRITICAL] body > section.concept-section.section-light > div.concept-grid > div.concept-card > p

- **Text**: "Championed by WattTime and the Emissions First Partnership. Estimate marginal em"
- **Color**: #2C3344 on #2C3344
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #959CAD or adjust for 4.5:1 on bg #2C3344

### [CRITICAL] section.concept-section.section-light > div.concept-grid > div.concept-card > ul > li

- **Text**: "Separate GHG Protocol consultation track (parallel to Scope 2 revision)"
- **Color**: #2C3344 on #2C3344
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #959CAD or adjust for 4.5:1 on bg #2C3344

### [CRITICAL] section.concept-section.section-light > div.concept-grid > div.concept-card > ul > li

- **Text**: "No requirement to match load temporally or spatially"
- **Color**: #2C3344 on #2C3344
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #959CAD or adjust for 4.5:1 on bg #2C3344

### [CRITICAL] section.concept-section.section-light > div.concept-grid > div.concept-card > ul > li

- **Text**: "Metric: tCO₂ displaced per dollar, measured via marginal emissions rates"
- **Color**: #2C3344 on #2C3344
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #959CAD or adjust for 4.5:1 on bg #2C3344

### [CRITICAL] section.concept-section.section-light > div.concept-grid > div.concept-card > ul > li

- **Text**: "Philosophy: pursue the cheapest abatement wherever it exists on the grid"
- **Color**: #2C3344 on #2C3344
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #959CAD or adjust for 4.5:1 on bg #2C3344

### [CRITICAL] section.concept-section.section-light > div.concept-grid > div.concept-card > ul > li

- **Text**: "Optimizes for lowest-cost abatement; does not require temporal or spatial matchi"
- **Color**: #2C3344 on #2C3344
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #959CAD or adjust for 4.5:1 on bg #2C3344

### [CRITICAL] div.concept-grid > div.concept-card > ul > li > strong

- **Text**: "Optimizes for lowest-cost abatement"
- **Color**: #2C3344 on #2C3344
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #959CAD or adjust for 4.5:1 on bg #2C3344

### [CRITICAL] html > body > div.insight-callout.fade-in > div.insight-box > h3

- **Text**: "The Core Tradeoff"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 16px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > div.insight-callout.fade-in > div.insight-box > p

- **Text**: "Hourly attribution is  — it tracks what you consume and what you procure. Becaus"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > div.insight-callout.fade-in > div.insight-box > p > em

- **Text**: "attributional"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > div.section-header.fade-in > h2

- **Text**: "The Evolving Landscape"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 3:1)
- **Font**: 19px, weight 800
- **Fix**: Change text to #8F9093 or adjust for 3:1 on bg #F8F9FC

### [CRITICAL] html > body > div.section-header.fade-in > p

- **Text**: "The GHG Protocol is running two parallel consultations: one revising Scope 2 att"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] html > body > div.section-header.fade-in > h2

- **Text**: "Three Methods Compared"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 3:1)
- **Font**: 19px, weight 800
- **Fix**: Change text to #8F9093 or adjust for 3:1 on bg #F8F9FC

### [CRITICAL] html > body > div.section-header.fade-in > p

- **Text**: "How the three approaches differ across the dimensions that matter for real clima"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > thead > tr > th

- **Text**: "Dimension"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > thead > tr > th

- **Text**: "Annual Attribution Current"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] table.comparison-table > thead > tr > th > span.badge.attr

- **Text**: "Current"
- **Color**: #F7EBED on #F7EBED
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #75696B or adjust for 4.5:1 on bg #F7EBED

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > thead > tr > th

- **Text**: "Hourly Attribution Proposed"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] table.comparison-table > thead > tr > th > span.badge.hourly

- **Text**: "Proposed"
- **Color**: #E5F2FA on #E5F2FA
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #5E6B73 or adjust for 4.5:1 on bg #E5F2FA

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > thead > tr > th

- **Text**: "Consequential Impact-Based"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] table.comparison-table > thead > tr > th > span.badge.conseq

- **Text**: "Impact-Based"
- **Color**: #E7F5EF on #E7F5EF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #606E68 or adjust for 4.5:1 on bg #E7F5EF

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Core Question"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] table.comparison-table > tbody > tr > td > strong

- **Text**: "Core Question"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "What share of consumption can I allocate to clean sources?"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "What resources match my consumption every hour on my grid?"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "What procurement displaces the most CO₂ per dollar anywhere?"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Temporal"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] table.comparison-table > tbody > tr > td > strong

- **Text**: "Temporal"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Annual"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Hourly (8,760)"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Varies — not required"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Spatial"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] table.comparison-table > tbody > tr > td > strong

- **Text**: "Spatial"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "National / broad market"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Deliverability region"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Not required — anywhere on grid"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Additionality"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] table.comparison-table > tbody > tr > td > strong

- **Text**: "Additionality"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Not required"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Strongly incentivized (hard hours need new build)"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Required (must prove marginal impact)"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Metric"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] table.comparison-table > tbody > tr > td > strong

- **Text**: "Metric"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "% renewable (RECs ÷ MWh)"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "CFE% + resource mix + cost"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "tCO₂ displaced, $/tCO₂"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Firm Power Signal"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] table.comparison-table > tbody > tr > td > strong

- **Text**: "Firm Power Signal"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "None — solar/wind RECs suffice"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Strong — can't cover night/winter without it"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Weak — firms chase cheapest marginal displacement"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Grid Impact"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] table.comparison-table > tbody > tr > td > strong

- **Text**: "Grid Impact"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Near-zero (Xu et al. 2024)"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "High — drives local deployment that serves your grid"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Variable — depends on additionality verification"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Gaming Risk"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] table.comparison-table > tbody > tr > td > strong

- **Text**: "Gaming Risk"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "High (temporal/spatial arbitrage)"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Low (physics constrains claims)"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Moderate-High (counterfactual baselines are model-dependent)"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Typical Claim"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] table.comparison-table > tbody > tr > td > strong

- **Text**: "Typical Claim"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: ""100% renewable energy""
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: ""92% hourly CFE in PJM with firm clean power""
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: ""Displaced 45,000 tCO₂ at $38/ton via MISO wind""
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Standard"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] table.comparison-table > tbody > tr > td > strong

- **Text**: "Standard"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "GHG Protocol Scope 2 MBM (2015)"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "GHG Protocol Scope 2 revision (~2027)"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "GHG Protocol consequential track (parallel)"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] html > body > div.section-header.fade-in > h2

- **Text**: "Ten Strategies We Assess"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 3:1)
- **Font**: 19px, weight 800
- **Fix**: Change text to #8F9093 or adjust for 3:1 on bg #F8F9FC

### [CRITICAL] html > body > div.section-header.fade-in > p

- **Text**: "The same corporate dollar can displace dramatically different amounts of carbon "
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] html > body > div.section-header.fade-in > p > strong

- **Text**: "how"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] html > body > div.section-header.fade-in > p > strong

- **Text**: "where"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] section.content-section.fade-in > div#accordionConseq > div.strategy-accordion-header > div.strategy-accordion-text > h3

- **Text**: "Consequential Accounting (Strategies 1A, 1B & 1C)"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 17px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] section.content-section.fade-in > div#accordionConseq > div.strategy-accordion-header > div.strategy-accordion-text > p.strat-summary

- **Text**: "Cross-regional deployment queue — build clean energy wherever the marginal abate"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] section.content-section.fade-in > div#accordionConseq > div.strategy-accordion-body > div.strategy-accordion-content > h4

- **Text**: "How It Works"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] section.content-section.fade-in > div#accordionConseq > div.strategy-accordion-body > div.strategy-accordion-content > p

- **Text**: "Consequential procurement builds clean energy , regardless of the buyer's locati"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div#accordionConseq > div.strategy-accordion-body > div.strategy-accordion-content > p > strong

- **Text**: "wherever the marginal abatement cost is lowest"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div#accordionConseq > div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > h4

- **Text**: "Grid-Average Baseline"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div#accordionConseq > div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > p

- **Text**: "Credits emission reductions against the  — total emissions divided by total gene"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > p > strong

- **Text**: "full grid emission rate"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div#accordionConseq > div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > h4

- **Text**: "Fossil-Average Baseline"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div#accordionConseq > div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > p

- **Text**: "Credits reductions against . Uses actual marginal dispatch rates from 8,760-hour"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > p > strong

- **Text**: "only the fossil fleet's emission rate"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div#accordionConseq > div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > h4

- **Text**: "Marginal Emission Baseline"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div#accordionConseq > div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > p

- **Text**: "Credits reductions against  — the emission rate of the specific generator that w"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > p > strong

- **Text**: "marginal emission rates"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div#accordionConseq > div.strategy-accordion-body > div.strategy-accordion-content > div.insight-box > strong

- **Text**: "1A vs 1B vs 1C:"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div#accordionConseq > div.strategy-accordion-body > div.strategy-accordion-content > div.insight-box > em

- **Text**: "attributed"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] section.content-section.fade-in > div#accordionConseq > div.strategy-accordion-body > div.strategy-accordion-content > h4

- **Text**: "The Coal Wall"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] section.content-section.fade-in > div#accordionConseq > div.strategy-accordion-body > div.strategy-accordion-content > p

- **Text**: "Consequential deployment follows the fossil fuel merit order:  (highest emission"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div#accordionConseq > div.strategy-accordion-body > div.strategy-accordion-content > p > strong

- **Text**: "coal is displaced first"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] section.content-section.fade-in > div#accordionConseq > div.strategy-accordion-body > div.strategy-accordion-content > h4

- **Text**: "Considerations at Scale"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] section.content-section.fade-in > div#accordionConseq > div.strategy-accordion-body > div.strategy-accordion-content > p

- **Text**: "Five dynamics emerge when consequential accounting scales from individual buyer "
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.strategy-accordion-body > div.strategy-accordion-content > div.failure-mode-item > div.failure-mode-text > strong

- **Text**: "Marginal Signal Degradation."
- **Color**: #F9F9FA on #F9F9FA
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F9F9FA

### [CRITICAL] div.strategy-accordion-body > div.strategy-accordion-content > div.failure-mode-item > div.failure-mode-text > strong

- **Text**: "Saturation & Diminishing Returns."
- **Color**: #F9F9FA on #F9F9FA
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F9F9FA

### [CRITICAL] div.strategy-accordion-body > div.strategy-accordion-content > div.failure-mode-item > div.failure-mode-text > strong

- **Text**: "Firm Clean Underinvestment."
- **Color**: #F9F9FA on #F9F9FA
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F9F9FA

### [CRITICAL] div.strategy-accordion-body > div.strategy-accordion-content > div.failure-mode-item > div.failure-mode-text > strong

- **Text**: "Geographic Clustering."
- **Color**: #F9F9FA on #F9F9FA
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F9F9FA

### [CRITICAL] div.strategy-accordion-body > div.strategy-accordion-content > div.failure-mode-item > div.failure-mode-text > strong

- **Text**: "Collective Action Dynamics."
- **Color**: #F9F9FA on #F9F9FA
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F9F9FA

### [CRITICAL] div#accordionConseq > div.strategy-accordion-body > div.strategy-accordion-content > div.insight-box > strong

- **Text**: "The Role of Consequential Analysis:"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div#accordionConseq > div.strategy-accordion-body > div.strategy-accordion-content > div.insight-box > em

- **Text**: "complementary analytical lens"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] section.content-section.fade-in > div#accordionConseq > div.strategy-accordion-body > div.strategy-accordion-content > p

- **Text**: "Deep dive: Consequential Scaling Analysis → — detailed analysis of how consequen"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div#accordionConseq > div.strategy-accordion-body > div.strategy-accordion-content > p > strong

- **Text**: "Deep dive:"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div#accordionConseq > div.strategy-accordion-body > div.strategy-accordion-content > p > a

- **Text**: "Consequential Scaling Analysis →"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] section.content-section.fade-in > div#accordionHourly > div.strategy-accordion-header > div.strategy-accordion-text > h3

- **Text**: "Hourly Matching (Strategies 2A, 2B & 2C)"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 17px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] section.content-section.fade-in > div#accordionHourly > div.strategy-accordion-header > div.strategy-accordion-text > p.strat-summary

- **Text**: "Match clean generation to consumption every hour within the buyer's own ISO — th"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] section.content-section.fade-in > div#accordionHourly > div.strategy-accordion-body > div.strategy-accordion-content > h4

- **Text**: "How It Works"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] section.content-section.fade-in > div#accordionHourly > div.strategy-accordion-body > div.strategy-accordion-content > p

- **Text**: "Hourly matching requires clean energy generation to equal consumption  within th"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div#accordionHourly > div.strategy-accordion-body > div.strategy-accordion-content > p > strong

- **Text**: "every hour of every day"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div#accordionHourly > div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > h4

- **Text**: "Pure New-Build (100% Additionality)"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div#accordionHourly > div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > p

- **Text**: "Every MWh of clean energy must come from  contracted by the buyer. No credit for"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > p > strong

- **Text**: "new-build resources"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div#accordionHourly > div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > h4

- **Text**: "Grid Baseline Credit"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div#accordionHourly > div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > p

- **Text**: "Buyers receive credit for the existing clean energy already embedded in their gr"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > p > strong

- **Text**: "incremental clean energy above the grid baseline"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div#accordionHourly > div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > h4

- **Text**: "Pro-Rata Allocation with SSS (GHG Protocol Proposed)"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div#accordionHourly > div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > p

- **Text**: "The model aligned with the GHG Protocol's proposed Scope 2 revision. The grid's "
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > p > strong

- **Text**: "four distinct pools"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > p > em

- **Text**: "beyond"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > p > strong

- **Text**: "roll-off sensitivity"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] section.content-section.fade-in > div#accordionHourly > div.strategy-accordion-body > div.strategy-accordion-content > h4

- **Text**: "The Four-Pool Supply Model (Strategy 2C)"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div#accordionHourly > div.strategy-accordion-body > div.strategy-accordion-content > div.insight-box > strong

- **Text**: "What is Standard Supply Service (SSS)?"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div#accordionHourly > div.strategy-accordion-body > div.strategy-accordion-content > div.insight-box > em

- **Text**: "not"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div#accordionHourly > div.strategy-accordion-body > div.strategy-accordion-content > div.insight-box > em

- **Text**: "beyond"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.strategy-accordion-body > div.strategy-accordion-content > div.grid-2col > div.pool-card > h4

- **Text**: "Pool 1: SSS (Policy-Supported)"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div.strategy-accordion-content > div.grid-2col > div.pool-card > div.pool-desc > strong

- **Text**: "Not"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.strategy-accordion-body > div.strategy-accordion-content > div.grid-2col > div.pool-card > h4

- **Text**: "Pool 2: Corporate-Contracted (Locked)"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div.strategy-accordion-body > div.strategy-accordion-content > div.grid-2col > div.pool-card > h4

- **Text**: "Pool 3: Existing Merchant Clean"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div.strategy-accordion-content > div.grid-2col > div.pool-card > div.pool-desc > strong

- **Text**: "Massive in ERCOT"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.strategy-accordion-body > div.strategy-accordion-content > div.grid-2col > div.pool-card > h4

- **Text**: "Pool 4: New-Build (Investment Signal)"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div.strategy-accordion-content > div.grid-2col > div.pool-card > div.pool-desc > em

- **Text**: "incentivizes"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] section.content-section.fade-in > div#accordionHourly > div.strategy-accordion-body > div.strategy-accordion-content > h4

- **Text**: "SSS Varies Dramatically by Region"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div.strategy-accordion-body > div.strategy-accordion-content > div.grid-3col > div.insight-box.insight-success > strong

- **Text**: "ERCOT: Mostly Existing Supply"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div.strategy-accordion-body > div.strategy-accordion-content > div.grid-3col > div.insight-box.insight-warn > strong

- **Text**: "PJM: Large SSS, But Contracted Away"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div.strategy-accordion-body > div.strategy-accordion-content > div.grid-3col > div.insight-box.insight-danger > strong

- **Text**: "NYISO: Zero Headroom"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div#accordionHourly > div.strategy-accordion-body > div.strategy-accordion-content > div.insight-box > strong

- **Text**: "SSS Stability vs. Roll-Off:"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div#accordionHourly > div.strategy-accordion-body > div.strategy-accordion-content > div.insight-box > strong

- **Text**: "2025 snapshot"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div.strategy-accordion-body > div.strategy-accordion-content > div.insight-box > ul > li

- **Text**: "NJ ZEC — expires 2026 (−15 TWh from PJM SSS)"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.strategy-accordion-content > div.insight-box > ul > li > strong

- **Text**: "NJ ZEC"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.strategy-accordion-body > div.strategy-accordion-content > div.insight-box > ul > li

- **Text**: "IL CMC — expires 2028 (−50 TWh PJM, −15 TWh MISO)"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.strategy-accordion-content > div.insight-box > ul > li > strong

- **Text**: "IL CMC"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.strategy-accordion-body > div.strategy-accordion-content > div.insight-box > ul > li

- **Text**: "NY ZEC Tier 3 — expires 2030 (−42 TWh from NYISO SSS)"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.strategy-accordion-content > div.insight-box > ul > li > strong

- **Text**: "NY ZEC Tier 3"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div#accordionHourly > div.strategy-accordion-body > div.strategy-accordion-content > div.insight-box > strong

- **Text**: "roll-off sensitivity variant"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div#accordionHourly > div.strategy-accordion-body > div.strategy-accordion-content > div.insight-box > a

- **Text**: "Procurement Deployment"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div#accordionHourly > div.strategy-accordion-body > div.strategy-accordion-content > div.insight-box > strong

- **Text**: "The Hourly Constraint and Investment Signals:"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] section.content-section.fade-in > div#accordionHourly > div.strategy-accordion-body > div.strategy-accordion-content > h4

- **Text**: "Considerations & Tradeoffs"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] section.content-section.fade-in > div#accordionHourly > div.strategy-accordion-body > div.strategy-accordion-content > p

- **Text**: "Hourly matching is the most demanding standard. At high CFE targets (>95%), cost"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div#accordionHourly > div.strategy-accordion-body > div.strategy-accordion-content > p > strong

- **Text**: "cost accessibility"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] section.content-section.fade-in > div#accordionAnnual > div.strategy-accordion-header > div.strategy-accordion-text > h3

- **Text**: "Annual Matching (Strategies 3A, 3B, 3C & 3D)"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 17px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] section.content-section.fade-in > div#accordionAnnual > div.strategy-accordion-header > div.strategy-accordion-text > p.strat-summary

- **Text**: "Match total annual clean generation to annual consumption — simpler accounting, "
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] section.content-section.fade-in > div#accordionAnnual > div.strategy-accordion-body > div.strategy-accordion-content > h4

- **Text**: "How It Works"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] section.content-section.fade-in > div#accordionAnnual > div.strategy-accordion-body > div.strategy-accordion-content > p

- **Text**: "Annual matching requires total clean energy generation to equal or exceed total "
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div#accordionAnnual > div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > h4

- **Text**: "Same-ISO New Build (Additionality)"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div#accordionAnnual > div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > p

- **Text**: "Annual matching within the buyer's own ISO with an : only newly built clean ener"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > p > strong

- **Text**: "additionality requirement"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > p > strong

- **Text**: "limited signal for firm clean power or storage"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div#accordionAnnual > div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > h4

- **Text**: "Cross-Regional New Build (No Additionality)"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div#accordionAnnual > div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > p

- **Text**: "Annual matching from any US ISO with  — existing clean generation counts. Can pu"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > p > strong

- **Text**: "no additionality requirement"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div#accordionAnnual > div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > h4

- **Text**: "Same-ISO + No Additionality"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div#accordionAnnual > div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > p

- **Text**: "Annual matching constrained to the buyer's own ISO region, with  toward the clai"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > p > strong

- **Text**: "existing clean energy counting"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div#accordionAnnual > div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > h4

- **Text**: "Cross-Regional + No Additionality (Status Quo)"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div#accordionAnnual > div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > p

- **Text**: "The loosest standard and the  for most corporate procurement today. Buy unbundle"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > p > strong

- **Text**: "de facto status quo"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] section.content-section.fade-in > div#accordionAnnual > div.strategy-accordion-body > div.strategy-accordion-content > h4

- **Text**: "The Temporal Gap Problem"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] section.content-section.fade-in > div#accordionAnnual > div.strategy-accordion-body > div.strategy-accordion-content > p

- **Text**: "Annual matching's fundamental weakness is temporal mismatch. At 95% annual match"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div#accordionAnnual > div.strategy-accordion-body > div.strategy-accordion-content > ul > li

- **Text**: "Summer days (May–Aug): 130–160% surplus solar generation. Excess is curtailed or"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.strategy-accordion-body > div.strategy-accordion-content > ul > li > strong

- **Text**: "Summer days (May–Aug):"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div#accordionAnnual > div.strategy-accordion-body > div.strategy-accordion-content > ul > li

- **Text**: "Winter evenings (Nov–Feb, 5–9 PM): 15–30% clean coverage. Fossil gas fills the g"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.strategy-accordion-body > div.strategy-accordion-content > ul > li > strong

- **Text**: "Winter evenings (Nov–Feb, 5–9 PM):"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div#accordionAnnual > div.strategy-accordion-body > div.strategy-accordion-content > ul > li

- **Text**: "Low-wind weeks: Multi-day periods with <20% clean coverage regardless of season."
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.strategy-accordion-body > div.strategy-accordion-content > ul > li > strong

- **Text**: "Low-wind weeks:"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] section.content-section.fade-in > div#accordionAnnual > div.strategy-accordion-body > div.strategy-accordion-content > p

- **Text**: "The annual total balances — but the grid runs on fossil fuel for thousands of ho"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div#accordionAnnual > div.strategy-accordion-body > div.strategy-accordion-content > div.insight-box > strong

- **Text**: "The Physical Limit:"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > div.section-header.fade-in > h2

- **Text**: "Strategy Summary"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 3:1)
- **Font**: 19px, weight 800
- **Fix**: Change text to #8F9093 or adjust for 3:1 on bg #F8F9FC

### [CRITICAL] html > body > div.section-header.fade-in > p

- **Text**: "How each strategy is modeled and what it optimizes for."
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > thead > tr > th

- **Text**: "Strategy"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > thead > tr > th

- **Text**: "Family"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > thead > tr > th

- **Text**: "Spatial"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > thead > tr > th

- **Text**: "Temporal"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > thead > tr > th

- **Text**: "Additionality"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > thead > tr > th

- **Text**: "Firm Signal"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "1A"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] table.comparison-table > tbody > tr > td > strong

- **Text**: "1A"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Consequential"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Cross-regional"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "N/A"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Marginal impact"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Weak"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "1B"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] table.comparison-table > tbody > tr > td > strong

- **Text**: "1B"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Consequential"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Cross-regional"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "N/A"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Marginal impact (fossil baseline)"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Weak"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "1C"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] table.comparison-table > tbody > tr > td > strong

- **Text**: "1C"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Consequential"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Cross-regional"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "N/A"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Marginal impact (marginal baseline)"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Weak"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "2A"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] table.comparison-table > tbody > tr > td > strong

- **Text**: "2A"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Hourly"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Same ISO"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "8,760 hours"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "100% new-build"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Maximum"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "2B"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] table.comparison-table > tbody > tr > td > strong

- **Text**: "2B"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Hourly"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Same ISO"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "8,760 hours"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Incremental above baseline"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Strong"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "2C"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] table.comparison-table > tbody > tr > td > strong

- **Text**: "2C"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Hourly"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Same ISO"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "8,760 hours"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Above SSS (four pools)"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Strong"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "3A"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] table.comparison-table > tbody > tr > td > strong

- **Text**: "3A"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Annual"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Same ISO"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Annual"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "New-build required"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "None"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "3B"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] table.comparison-table > tbody > tr > td > strong

- **Text**: "3B"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Annual"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Cross-regional"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Annual"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "New-build required"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "None"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "3C"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] table.comparison-table > tbody > tr > td > strong

- **Text**: "3C"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Annual"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Same ISO"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Annual"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "None (existing counts)"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "None"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "3D"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] table.comparison-table > tbody > tr > td > strong

- **Text**: "3D"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Annual"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Cross-regional"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "Annual"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "None (unbundled RECs)"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div.comparison-table-wrap.fade-in > table.comparison-table > tbody > tr > td

- **Text**: "None"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] html > body > section.sources-section.fade-in > h3

- **Text**: "Sources & Further Reading"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] html > body > section.sources-section.fade-in > ol > li

- **Text**: "GHG Protocol, "Scope 2 Guidance" (2015; revision expected ~2027). ghgprotocol.or"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] body > section.sources-section.fade-in > ol > li > a

- **Text**: "ghgprotocol.org"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] html > body > section.sources-section.fade-in > ol > li

- **Text**: "GHG Protocol, "Scope 2 Proposed Updates & Consequential Accounting Consultation""
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] html > body > section.sources-section.fade-in > ol > li

- **Text**: "Clean Air Task Force, "Modernizing GHG Accounting Rules" (2024). catf.us"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] body > section.sources-section.fade-in > ol > li > a

- **Text**: "catf.us"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] html > body > section.sources-section.fade-in > ol > li

- **Text**: "EnergyTag, "Scope 2 for the Age of Deep Decarbonization" (2024). energytag.org"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] body > section.sources-section.fade-in > ol > li > a

- **Text**: "energytag.org"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] html > body > section.sources-section.fade-in > ol > li

- **Text**: "Xu et al., "System-level impacts of 24/7 carbon-free energy procurement" (2024)."
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] body > section.sources-section.fade-in > ol > li > a

- **Text**: "ScienceDirect"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] html > body > section.sources-section.fade-in > ol > li

- **Text**: "Bjørn et al., "Ensuring low-emission electricity purchasing requires broader sys"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] body > section.sources-section.fade-in > ol > li > a

- **Text**: "Nature Communications"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] html > body > section.section-dark.fade-in > div.section-dark-inner > div.findings-title

- **Text**: "Key Findings — Top Down"
- **Color**: #0F1A2E on #0F1A2E
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #788397 or adjust for 4.5:1 on bg #0F1A2E

### [CRITICAL] div.section-dark-inner > div.findings-grid > div.finding-row.top > div.finding-card.tier1 > div.finding-label

- **Text**: "The same corporate dollar produces dramatically different outcomes depending on "
- **Color**: #2C3547 on #2C3547
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #959EB0 or adjust for 4.5:1 on bg #2C3547

### [CRITICAL] div.section-dark-inner > div.findings-grid > div.finding-row.top > div.finding-card.tier1 > div.finding-desc

- **Text**: "Across 7 ISO/RTOs (~2.7 million GWh), the three accounting families — consequent"
- **Color**: #2C3547 on #2C3547
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #959EB0 or adjust for 4.5:1 on bg #2C3547

### [CRITICAL] div.section-dark-inner > div.findings-grid > div.finding-row.mid > div.finding-card.tier2 > div.finding-label

- **Text**: "The last-mile cost escalation reveals a real tradeoff"
- **Color**: #2C3547 on #2C3547
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #959EB0 or adjust for 4.5:1 on bg #2C3547

### [CRITICAL] div.section-dark-inner > div.findings-grid > div.finding-row.mid > div.finding-card.tier2 > div.finding-desc

- **Text**: "Hourly matching from 90% → ≥99.99% costs 2–5× more per MWh. These hard hours — n"
- **Color**: #2C3547 on #2C3547
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #959EB0 or adjust for 4.5:1 on bg #2C3547

### [CRITICAL] div.section-dark-inner > div.findings-grid > div.finding-row.mid > div.finding-card.tier2 > div.finding-label

- **Text**: "Hourly attribution produces both inventory accounting and real emissions displac"
- **Color**: #2C3547 on #2C3547
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #959EB0 or adjust for 4.5:1 on bg #2C3547

### [CRITICAL] div.section-dark-inner > div.findings-grid > div.finding-row.mid > div.finding-card.tier2 > div.finding-desc

- **Text**: "Because hourly matching requires procurement aligned to grid physics and deliver"
- **Color**: #2C3547 on #2C3547
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #959EB0 or adjust for 4.5:1 on bg #2C3547

### [CRITICAL] div.section-dark-inner > div.findings-grid > div.finding-row.base > div.finding-card.tier3 > div.finding-label

- **Text**: "Annual, hourly, and consequential are distinct frameworks"
- **Color**: #2C3547 on #2C3547
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #959EB0 or adjust for 4.5:1 on bg #2C3547

### [CRITICAL] div.section-dark-inner > div.findings-grid > div.finding-row.base > div.finding-card.tier3 > div.finding-desc

- **Text**: "Annual attribution (current Scope 2) and hourly attribution (proposed revision) "
- **Color**: #2C3547 on #2C3547
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #959EB0 or adjust for 4.5:1 on bg #2C3547

### [CRITICAL] div.section-dark-inner > div.findings-grid > div.finding-row.base > div.finding-card.tier3 > div.finding-label

- **Text**: "Cross-regional optimization raises collective action questions"
- **Color**: #2C3547 on #2C3547
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #959EB0 or adjust for 4.5:1 on bg #2C3547

### [CRITICAL] div.section-dark-inner > div.findings-grid > div.finding-row.base > div.finding-card.tier3 > div.finding-desc

- **Text**: "When buyers independently optimize for cheapest $/tCO₂ regardless of location, p"
- **Color**: #2C3547 on #2C3547
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #959EB0 or adjust for 4.5:1 on bg #2C3547

### [CRITICAL] div.section-dark-inner > div.findings-grid > div.finding-row.base > div.finding-card.tier3 > div.finding-label

- **Text**: "Marginal impact measurement depends on model assumptions"
- **Color**: #2C3547 on #2C3547
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #959EB0 or adjust for 4.5:1 on bg #2C3547

### [CRITICAL] div.section-dark-inner > div.findings-grid > div.finding-row.base > div.finding-card.tier3 > div.finding-desc

- **Text**: "Consequential accounting relies on estimating counterfactual outcomes — what wou"
- **Color**: #2C3547 on #2C3547
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #959EB0 or adjust for 4.5:1 on bg #2C3547

### [CRITICAL] body > section.timeline-section.section-light > div.timeline > div.timeline-item > div.timeline-title

- **Text**: "Google Commits to 24/7 CFE by 2030"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #8F9093 or adjust for 3:1 on bg #F8F9FC

### [CRITICAL] body > section.timeline-section.section-light > div.timeline > div.timeline-item > div.timeline-title

- **Text**: "UN 24/7 CFE Compact Launched"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #8F9093 or adjust for 3:1 on bg #F8F9FC

### [CRITICAL] body > section.timeline-section.section-light > div.timeline > div.timeline-item > div.timeline-title

- **Text**: "EnergyTag & Granular Certificates"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #8F9093 or adjust for 3:1 on bg #F8F9FC

### [CRITICAL] body > section.timeline-section.section-light > div.timeline > div.timeline-item > div.timeline-title

- **Text**: "GHG Protocol Dual-Track Consultation"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #8F9093 or adjust for 3:1 on bg #F8F9FC

### [CRITICAL] body > section.timeline-section.section-light > div.timeline > div.timeline-item > div.timeline-title

- **Text**: "New Standards Expected"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #8F9093 or adjust for 3:1 on bg #F8F9FC

### [CRITICAL] div#accordionConseq > div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > div.strat-tag

- **Text**: "Strategy 1A"
- **Color**: #E9F9EF on #E9F9EF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 700
- **Fix**: Change text to #627268 or adjust for 4.5:1 on bg #E9F9EF

### [CRITICAL] div#accordionConseq > div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > div.strat-tag

- **Text**: "Strategy 1B"
- **Color**: #DEF6E7 on #DEF6E7
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 700
- **Fix**: Change text to #576F60 or adjust for 4.5:1 on bg #DEF6E7

### [CRITICAL] div#accordionConseq > div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > div.strat-tag

- **Text**: "Strategy 1C"
- **Color**: #D3F3DF on #D3F3DF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 700
- **Fix**: Change text to #51715D or adjust for 4.5:1 on bg #D3F3DF

### [CRITICAL] section.content-section.fade-in > div#accordionConseq > div.strategy-accordion-body > div.strategy-accordion-content > div.insight-box

- **Text**: "All three strategies deploy identical resources in the same cross-regional seque"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div#accordionConseq > div.strategy-accordion-body > div.strategy-accordion-content > div.failure-mode-item > div.failure-mode-text

- **Text**: "At low volumes, marginal rates are meaningful. At 5–15% C&I participation, each "
- **Color**: #F9F9FA on #F9F9FA
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F9F9FA

### [CRITICAL] div#accordionConseq > div.strategy-accordion-body > div.strategy-accordion-content > div.failure-mode-item > div.failure-mode-text

- **Text**: "Every grid has a finite stock of high-emission generators available for displace"
- **Color**: #F9F9FA on #F9F9FA
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F9F9FA

### [CRITICAL] div#accordionConseq > div.strategy-accordion-body > div.strategy-accordion-content > div.failure-mode-item > div.failure-mode-text

- **Text**: "By directing marginal clean energy capital toward the cheapest abatement opportu"
- **Color**: #F9F9FA on #F9F9FA
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F9F9FA

### [CRITICAL] div#accordionConseq > div.strategy-accordion-body > div.strategy-accordion-content > div.failure-mode-item > div.failure-mode-text

- **Text**: "Because consequential optimizes for cheapest $/tCO₂ cross-regionally, investment"
- **Color**: #F9F9FA on #F9F9FA
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F9F9FA

### [CRITICAL] div#accordionConseq > div.strategy-accordion-body > div.strategy-accordion-content > div.failure-mode-item > div.failure-mode-text

- **Text**: "Each buyer optimizing for their own cheapest $/tCO₂ is individually rational. Bu"
- **Color**: #F9F9FA on #F9F9FA
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F9F9FA

### [CRITICAL] section.content-section.fade-in > div#accordionConseq > div.strategy-accordion-body > div.strategy-accordion-content > div.insight-box

- **Text**: "These dynamics suggest consequential accounting may be most effective as a  — me"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > section.content-section.fade-in > div#accordionHourly > div.strategy-accordion-header > div.strategy-accordion-text

- **Text**: "Hourly Matching (Strategies 2A, 2B & 2C)
            Match clean generation to c"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 16px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div#accordionHourly > div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > div.strat-tag

- **Text**: "Strategy 2A"
- **Color**: #E7F6FD on #E7F6FD
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 700
- **Fix**: Change text to #606F76 or adjust for 4.5:1 on bg #E7F6FD

### [CRITICAL] div#accordionHourly > div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > div.strat-tag

- **Text**: "Strategy 2B"
- **Color**: #E2F4FC on #E2F4FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 700
- **Fix**: Change text to #5B6D75 or adjust for 4.5:1 on bg #E2F4FC

### [CRITICAL] div#accordionHourly > div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > div.strat-tag

- **Text**: "Strategy 2C"
- **Color**: #DBF2FC on #DBF2FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 700
- **Fix**: Change text to #59707A or adjust for 4.5:1 on bg #DBF2FC

### [CRITICAL] section.content-section.fade-in > div#accordionHourly > div.strategy-accordion-body > div.strategy-accordion-content > div.insight-box

- **Text**: "SSS is  just "existing grid clean." It's specifically: publicly owned, rate-base"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.strategy-accordion-body > div.strategy-accordion-content > div.grid-2col > div.pool-card > div.pool-val

- **Text**: "227 TWh"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 22px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div.strategy-accordion-body > div.strategy-accordion-content > div.grid-2col > div.pool-card > div.pool-val

- **Text**: "36.8 TWh"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 22px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div.strategy-accordion-body > div.strategy-accordion-content > div.grid-2col > div.pool-card > div.pool-val

- **Text**: "Varies by ISO"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 22px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div.strategy-accordion-body > div.strategy-accordion-content > div.grid-2col > div.pool-card > div.pool-val

- **Text**: "The Gap"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 22px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div#accordionHourly > div.strategy-accordion-body > div.strategy-accordion-content > div.grid-3col > div.insight-box.insight-success

- **Text**: "No SSS (fully deregulated), but ~183 TWh of merchant solar+wind means hourly mat"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div#accordionHourly > div.strategy-accordion-body > div.strategy-accordion-content > div.grid-3col > div.insight-box.insight-warn

- **Text**: "95 TWh SSS (Exelon ZEC fleet) sounds large — but 18.8 TWh is locked to Amazon (S"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div#accordionHourly > div.strategy-accordion-body > div.strategy-accordion-content > div.grid-3col > div.insight-box.insight-danger

- **Text**: "Aggressive CLCPA targets consume all buildable capacity for RPS (SSS). Corporate"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] section.content-section.fade-in > div#accordionHourly > div.strategy-accordion-body > div.strategy-accordion-content > div.insight-box

- **Text**: "This analysis uses a  that treats all current nuclear support policies as stable"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] section.content-section.fade-in > div#accordionHourly > div.strategy-accordion-body > div.strategy-accordion-content > div.insight-box

- **Text**: "At 80–90% CFE targets, over 4,000 hours per year require firm clean or storage —"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div#accordionAnnual > div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > div.strat-tag

- **Text**: "Strategy 3A"
- **Color**: #FDECEC on #FDECEC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 700
- **Fix**: Change text to #766565 or adjust for 4.5:1 on bg #FDECEC

### [CRITICAL] div#accordionAnnual > div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > div.strat-tag

- **Text**: "Strategy 3B"
- **Color**: #FDE3E3 on #FDE3E3
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 700
- **Fix**: Change text to #7B6161 or adjust for 4.5:1 on bg #FDE3E3

### [CRITICAL] div#accordionAnnual > div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > div.strat-tag

- **Text**: "Strategy 3C"
- **Color**: #FCDADA on #FCDADA
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 700
- **Fix**: Change text to #7A5858 or adjust for 4.5:1 on bg #FCDADA

### [CRITICAL] div#accordionAnnual > div.strategy-accordion-body > div.strategy-accordion-content > div.strat-variant-card > div.strat-tag

- **Text**: "Strategy 3D"
- **Color**: #FBD0D0 on #FBD0D0
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 700
- **Fix**: Change text to #7E5353 or adjust for 4.5:1 on bg #FBD0D0

### [CRITICAL] section.content-section.fade-in > div#accordionAnnual > div.strategy-accordion-body > div.strategy-accordion-content > div.insight-box

- **Text**: "Annual matching faces a practical ceiling around ~95% — beyond this point, massi"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

## dashboard/consequential_vacuum.html

**182 failures**

### [CRITICAL] html > body > div.section-header.fade-in > h2

- **Text**: "The Marginal Signal Degrades With Every Buyer"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 19px, weight 800
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > div.section-header.fade-in > p

- **Text**: "Consequential accounting relies on marginal emissions rates — the emissions inte"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.content-section.section-light > p

- **Text**: "A single 100 MW wind PPA in MISO displaces the marginal gas or coal unit on each"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 16px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.content-section.section-light > p > strong

- **Text**: "At low volumes, marginal rates are meaningful."
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 16px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > section.content-section.section-light > p

- **Text**: "Each new PPA doesn't displace the same marginal unit. The first PPA displaces co"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 16px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.content-section.section-light > p > strong

- **Text**: "At 5–15% C&I participation, the signal begins to degrade."
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 16px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > section.content-section.section-light > p > em

- **Text**: "original"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 16px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.content-section.section-light > p

- **Text**: "It's a direct consequence of merit-order economics: each unit of renewable gener"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 16px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.content-section.section-light > p > strong

- **Text**: "This isn't theoretical."
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 16px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] body > section.content-section.section-light > div.mechanism-grid > div.mechanism-card > h4

- **Text**: "Double-Counting at the Margin"
- **Color**: #F9F9F9 on #F9F9F9
- **Ratio**: 1:1 (need 3:1)
- **Font**: 16px, weight 700
- **Fix**: Change text to #909090 or adjust for 3:1 on bg #F9F9F9

### [CRITICAL] body > section.content-section.section-light > div.mechanism-grid > div.mechanism-card > p

- **Text**: "If 50 buyers each claim they displaced coal at 2,200 lb/MWh, but the grid only h"
- **Color**: #F9F9F9 on #F9F9F9
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727272 or adjust for 4.5:1 on bg #F9F9F9

### [CRITICAL] body > section.content-section.section-light > div.mechanism-grid > div.mechanism-card > h4

- **Text**: "Temporal Instability"
- **Color**: #F9F9F9 on #F9F9F9
- **Ratio**: 1:1 (need 3:1)
- **Font**: 16px, weight 700
- **Fix**: Change text to #909090 or adjust for 3:1 on bg #F9F9F9

### [CRITICAL] body > section.content-section.section-light > div.mechanism-grid > div.mechanism-card > p

- **Text**: "Marginal rates vary hour-to-hour, season-to-season, and year-to-year. A buyer wh"
- **Color**: #F9F9F9 on #F9F9F9
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727272 or adjust for 4.5:1 on bg #F9F9F9

### [CRITICAL] body > section.content-section.section-light > div.mechanism-grid > div.mechanism-card > h4

- **Text**: "Model Dependence"
- **Color**: #F9F9F9 on #F9F9F9
- **Ratio**: 1:1 (need 3:1)
- **Font**: 16px, weight 700
- **Fix**: Change text to #909090 or adjust for 3:1 on bg #F9F9F9

### [CRITICAL] body > section.content-section.section-light > div.mechanism-grid > div.mechanism-card > p

- **Text**: "There is no single "correct" marginal emissions rate. Different dispatch models "
- **Color**: #F9F9F9 on #F9F9F9
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727272 or adjust for 4.5:1 on bg #F9F9F9

### [CRITICAL] html > body > div.insight-callout.fade-in > div.page-insight-box > h3

- **Text**: "The Measurement Paradox"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 17px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > div.insight-callout.fade-in > div.page-insight-box > p

- **Text**: "Consequential accounting claims superiority because it measures "actual impact.""
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 16px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > div.insight-callout.fade-in > div.page-insight-box > p > span.highlight

- **Text**: "becomes less accurate as more buyers use it"
- **Color**: #FDECCE on #FDECCE
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 16px, weight 600
- **Fix**: Change text to #7B6A4C or adjust for 4.5:1 on bg #FDECCE

### [CRITICAL] html > body > div.section-header.fade-in > h2

- **Text**: "Saturation: When Cheap Displacement Runs Out"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 19px, weight 800
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > div.section-header.fade-in > p

- **Text**: "Every grid has a finite stock of "cheap abatement" — the coal and inefficient ga"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > section.chart-section.section-light > div.chart-panel > div#saturationIsoSelector > button.iso-btn

- **Text**: "SPP"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > section.chart-section.section-light > div.chart-panel > div#saturationIsoSelector > button.iso-btn.active

- **Text**: "MISO"
- **Color**: #FEEEE3 on #FEEEE3
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #77675C or adjust for 4.5:1 on bg #FEEEE3

### [CRITICAL] body > section.chart-section.section-light > div.chart-panel > div#saturationIsoSelector > button.iso-btn

- **Text**: "PJM"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > section.chart-section.section-light > div.chart-panel > div#saturationIsoSelector > button.iso-btn

- **Text**: "ERCOT"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > section.chart-section.section-light > div.chart-panel > div#saturationIsoSelector > button.iso-btn

- **Text**: "NEISO"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > section.chart-section.section-light > div.chart-panel > div#saturationIsoSelector > button.iso-btn

- **Text**: "NYISO"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > section.chart-section.section-light > div.chart-panel > div#saturationIsoSelector > button.iso-btn

- **Text**: "CAISO"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.chart-section.section-light > div.chart-narrative-below > p

- **Text**: "The shaded band between the coal/oil reference line and the efficient gas baseli"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > section.chart-section.section-light > div.chart-narrative-below > p > strong

- **Text**: "The displacement curve is concave for every grid — steep at first, then flatteni"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > section.chart-section.section-light > div.chart-narrative-below > p

- **Text**: "SPP exhausts coal/oil at ~52% penetration, MISO at ~42%, PJM and ERCOT at ~24%. "
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > section.chart-section.section-light > div.chart-narrative-below > p > strong

- **Text**: "The transition thresholds are stark:"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > section.chart-section.section-light > div.chart-narrative-below > p

- **Text**: "because consequential capital from  competes for the same "attractive" coal disp"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > section.chart-section.section-light > div.chart-narrative-below > p > strong

- **Text**: "The saturation problem is worse than the curve suggests"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] body > section.chart-section.section-light > div.chart-narrative-below > p > em

- **Text**: "all seven ISOs"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.chart-section.section-light > div.chart-narrative-below > p

- **Text**: "consequential accounting produces the same $/tCO₂ as hourly matching — because t"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > section.chart-section.section-light > div.chart-narrative-below > p > strong

- **Text**: "After saturation:"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > div.section-header.fade-in > h2

- **Text**: "Fossil Lock-In: How Cheap Abatement Preserves Gas Infrastructure"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 19px, weight 800
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > div.section-header.fade-in > p

- **Text**: "The most insidious failure mode: by directing all marginal clean energy capital "
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.content-section.section-light > p

- **Text**: "Gas-dependent grids (NEISO, NYISO, CAISO) need firm clean power — nuclear, geoth"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 16px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.content-section.section-light > p > strong

- **Text**: "The mechanism is straightforward."
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 16px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > section.content-section.section-light > p

- **Text**: "Without demand signals for nuclear/CCS/LDES in these grids, no developer builds "
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 16px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.content-section.section-light > p > strong

- **Text**: "The result: gas plants in NEISO, NYISO, and CAISO face no competitive pressure f"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 16px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] body > section.content-section.section-light > div.mechanism-grid > div.mechanism-card > h4

- **Text**: "NEISO: The Pipeline Trap"
- **Color**: #F9F9F9 on #F9F9F9
- **Ratio**: 1:1 (need 3:1)
- **Font**: 16px, weight 700
- **Fix**: Change text to #909090 or adjust for 3:1 on bg #F9F9F9

### [CRITICAL] body > section.content-section.section-light > div.mechanism-grid > div.mechanism-card > p

- **Text**: "New England's grid is 70%+ gas, constrained by pipeline capacity from Appalachia"
- **Color**: #F9F9F9 on #F9F9F9
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727272 or adjust for 4.5:1 on bg #F9F9F9

### [CRITICAL] body > section.content-section.section-light > div.mechanism-grid > div.mechanism-card > h4

- **Text**: "CAISO: The Duck Curve Deepens"
- **Color**: #F9F9F9 on #F9F9F9
- **Ratio**: 1:1 (need 3:1)
- **Font**: 16px, weight 700
- **Fix**: Change text to #909090 or adjust for 3:1 on bg #F9F9F9

### [CRITICAL] body > section.content-section.section-light > div.mechanism-grid > div.mechanism-card > p

- **Text**: "California already has massive solar penetration. What it needs is dispatchable "
- **Color**: #F9F9F9 on #F9F9F9
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727272 or adjust for 4.5:1 on bg #F9F9F9

### [CRITICAL] body > section.content-section.section-light > div.mechanism-grid > div.mechanism-card > h4

- **Text**: "NYISO: Peaker Retirement Stalls"
- **Color**: #F9F9F9 on #F9F9F9
- **Ratio**: 1:1 (need 3:1)
- **Font**: 16px, weight 700
- **Fix**: Change text to #909090 or adjust for 3:1 on bg #F9F9F9

### [CRITICAL] body > section.content-section.section-light > div.mechanism-grid > div.mechanism-card > p

- **Text**: "New York City runs on gas peakers — old, inefficient, disproportionately located"
- **Color**: #F9F9F9 on #F9F9F9
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727272 or adjust for 4.5:1 on bg #F9F9F9

### [CRITICAL] html > body > div.insight-callout.fade-in > div.page-insight-box > h3

- **Text**: "The Lifespan Extension"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 17px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > div.insight-callout.fade-in > div.page-insight-box > p

- **Text**: "Under hourly matching, a NEISO corporate buyer must contract firm clean power to"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 16px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > div.insight-callout.fade-in > div.page-insight-box > p > span.highlight

- **Text**: "Every year of delayed firm clean investment in gas-dependent grids is another ye"
- **Color**: #FDECCE on #FDECCE
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 16px, weight 600
- **Fix**: Change text to #7B6A4C or adjust for 4.5:1 on bg #FDECCE

### [CRITICAL] html > body > div.section-header.fade-in > h2

- **Text**: "Geographic Clustering: Capital Concentration Across 7 ISOs"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 19px, weight 800
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > div.section-header.fade-in > p

- **Text**: "When corporate buyers across all major U.S. grids optimize for cheapest $/tCO₂, "
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > thead > tr > th

- **Text**: "ISO/RTO"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > thead > tr > th

- **Text**: "Annual Load (TWh)"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > thead > tr > th

- **Text**: "C&I Load @ 60%"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > thead > tr > th

- **Text**: "Coal %"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > thead > tr > th

- **Text**: "Gas %"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > thead > tr > th

- **Text**: "Fossil CO₂ (lb/MWh)"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > thead > tr > th

- **Text**: "Consequential Attractiveness"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "SPP"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] table.comparison-table > tbody > tr > td > strong

- **Text**: "SPP"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "260"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "156 TWh"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "52%"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "48%"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "1,340"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "Highest — cheap wind displaces coal"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] table.comparison-table > tbody > tr > td > strong

- **Text**: "Highest — cheap wind displaces coal"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "MISO"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] table.comparison-table > tbody > tr > td > strong

- **Text**: "MISO"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "625"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "375 TWh"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "42%"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "58%"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "1,250"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "Very High — large grid, coal-heavy"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] table.comparison-table > tbody > tr > td > strong

- **Text**: "Very High — large grid, coal-heavy"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "PJM"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] table.comparison-table > tbody > tr > td > strong

- **Text**: "PJM"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "843"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "506 TWh"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "24%"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "75%"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "1,187"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "Moderate — coal share falling"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "ERCOT"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] table.comparison-table > tbody > tr > td > strong

- **Text**: "ERCOT"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "488"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "293 TWh"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "23%"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "77%"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "1,180"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "Moderate — strong wind already"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "NEISO"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] table.comparison-table > tbody > tr > td > strong

- **Text**: "NEISO"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "115"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "69 TWh"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "<1%"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "99%"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "852"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "Lowest — expensive, gas-only"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] table.comparison-table > tbody > tr > td > strong

- **Text**: "Lowest — expensive, gas-only"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "NYISO"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] table.comparison-table > tbody > tr > td > strong

- **Text**: "NYISO"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "152"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "91 TWh"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "0%"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "99.6%"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "915"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "Very Low — no coal to displace"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] table.comparison-table > tbody > tr > td > strong

- **Text**: "Very Low — no coal to displace"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "CAISO"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] table.comparison-table > tbody > tr > td > strong

- **Text**: "CAISO"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "224"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "134 TWh"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "0%"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "99.3%"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "863"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.comparison-table-wrap > table.comparison-table > tbody > tr > td

- **Text**: "Low — gas-only, high solar"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] table.comparison-table > tbody > tr > td > strong

- **Text**: "Low — gas-only, high solar"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.chart-section.section-light > p

- **Text**: "Sources: EIA 860/861, eGRID 2022, ISO annual reports. C&I load estimated at 60% "
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.chart-section.section-light > div.chart-narrative-below > p

- **Text**: "The stacked bars show each ISO's fossil fuel mix (coal vs. gas), while the green"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > section.chart-section.section-light > div.chart-narrative-below > p > strong

- **Text**: "The chart makes the capital concentration visible."
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > section.chart-section.section-light > div.chart-narrative-below > p

- **Text**: "SPP + MISO have 531 TWh of C&I load (~33% of total) but attract 45–55% of conseq"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > section.chart-section.section-light > div.chart-narrative-below > p > strong

- **Text**: "The asymmetry is structural."
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > section.chart-section.section-light > div.chart-narrative-below > p

- **Text**: "Below 20%, capital allocation roughly matches each grid's share of C&I load — bu"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > section.chart-section.section-light > div.chart-narrative-below > p > strong

- **Text**: "The 20–30% C&I participation band is the tipping point."
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > section.chart-section.section-light > div.chart-narrative-below > p

- **Text**: "(~800+ TWh of consequential procurement), SPP's interconnection queue is saturat"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > section.chart-section.section-light > div.chart-narrative-below > p > strong

- **Text**: "At 50%+ participation"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > section.chart-section.section-light > div.chart-narrative-below > p

- **Text**: "if more than ~60% of corporate load uses consequential accounting, the remaining"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > section.chart-section.section-light > div.chart-narrative-below > p > strong

- **Text**: "The 60% threshold is existential:"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > div.section-header.fade-in > h2

- **Text**: "The Tragedy of the Carbon Commons"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 19px, weight 800
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > div.section-header.fade-in > p

- **Text**: "Each failure mode reinforces the others. Marginal signal degradation makes displ"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.section-dark.fade-in > div.section-dark-inner > p

- **Text**: "Each buyer optimizing for their own cheapest $/tCO₂ is locally rational. But the"
- **Color**: #0F1A2E on #0F1A2E
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 16px, weight 400
- **Fix**: Change text to #788397 or adjust for 4.5:1 on bg #0F1A2E

### [CRITICAL] body > section.section-dark.fade-in > div.section-dark-inner > p > strong

- **Text**: "The game-theoretic structure is a textbook tragedy of the commons."
- **Color**: #0F1A2E on #0F1A2E
- **Ratio**: 1:1 (need 3:1)
- **Font**: 16px, weight 700
- **Fix**: Change text to #5F6A7E or adjust for 3:1 on bg #0F1A2E

### [CRITICAL] html > body > section.section-dark.fade-in > div.section-dark-inner > p

- **Text**: "by construction. When procurement must match load hour-by-hour within a delivera"
- **Color**: #0F1A2E on #0F1A2E
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 16px, weight 400
- **Fix**: Change text to #788397 or adjust for 4.5:1 on bg #0F1A2E

### [CRITICAL] body > section.section-dark.fade-in > div.section-dark-inner > p > strong

- **Text**: "Hourly attribution avoids this trap"
- **Color**: #0F1A2E on #0F1A2E
- **Ratio**: 1:1 (need 3:1)
- **Font**: 16px, weight 700
- **Fix**: Change text to #5F6A7E or adjust for 3:1 on bg #0F1A2E

### [CRITICAL] html > body > section.section-dark.fade-in > div.section-dark-inner > p

- **Text**: "hourly matching produces  in the short run for any individual buyer — because it"
- **Color**: #0F1A2E on #0F1A2E
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 16px, weight 400
- **Fix**: Change text to #788397 or adjust for 4.5:1 on bg #0F1A2E

### [CRITICAL] body > section.section-dark.fade-in > div.section-dark-inner > p > strong

- **Text**: "The irony:"
- **Color**: #0F1A2E on #0F1A2E
- **Ratio**: 1:1 (need 3:1)
- **Font**: 16px, weight 700
- **Fix**: Change text to #5F6A7E or adjust for 3:1 on bg #0F1A2E

### [CRITICAL] body > section.section-dark.fade-in > div.section-dark-inner > p > em

- **Text**: "higher $/tCO₂"
- **Color**: #0F1A2E on #0F1A2E
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 16px, weight 400
- **Fix**: Change text to #788397 or adjust for 4.5:1 on bg #0F1A2E

### [CRITICAL] body > section.section-dark.fade-in > div.section-dark-inner > p > em

- **Text**: "lower total system emissions"
- **Color**: #0F1A2E on #0F1A2E
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 16px, weight 400
- **Fix**: Change text to #788397 or adjust for 4.5:1 on bg #0F1A2E

### [CRITICAL] html > body > section.section-dark.fade-in > div.section-dark-inner > blockquote

- **Text**: "Consequential accounting minimizes $/tCO₂ for each buyer. Hourly matching minimi"
- **Color**: #273143 on #273143
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 16px, weight 400
- **Fix**: Change text to #909AAC or adjust for 4.5:1 on bg #273143

### [CRITICAL] body > section.section-dark.fade-in > div.section-dark-inner > blockquote > strong

- **Text**: "Individual optimization vs. system outcome:"
- **Color**: #273143 on #273143
- **Ratio**: 1:1 (need 3:1)
- **Font**: 16px, weight 700
- **Fix**: Change text to #727C8E or adjust for 3:1 on bg #273143

### [CRITICAL] section.section-dark.fade-in > div.section-dark-inner > div.mechanism-grid > div.mechanism-card > h4

- **Text**: "Why This Isn't Just Theory"
- **Color**: #333C4D on #333C4D
- **Ratio**: 1:1 (need 3:1)
- **Font**: 16px, weight 700
- **Fix**: Change text to #7E8798 or adjust for 3:1 on bg #333C4D

### [CRITICAL] section.section-dark.fade-in > div.section-dark-inner > div.mechanism-grid > div.mechanism-card > p

- **Text**: "We're already seeing early signs: 70%+ of voluntary corporate renewable PPAs in "
- **Color**: #333C4D on #333C4D
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #A1AABB or adjust for 4.5:1 on bg #333C4D

### [CRITICAL] section.section-dark.fade-in > div.section-dark-inner > div.mechanism-grid > div.mechanism-card > h4

- **Text**: "What Hourly Matching Solves"
- **Color**: #333C4D on #333C4D
- **Ratio**: 1:1 (need 3:1)
- **Font**: 16px, weight 700
- **Fix**: Change text to #7E8798 or adjust for 3:1 on bg #333C4D

### [CRITICAL] section.section-dark.fade-in > div.section-dark-inner > div.mechanism-grid > div.mechanism-card > p

- **Text**: "By requiring temporal and spatial matching, hourly attribution creates guarantee"
- **Color**: #333C4D on #333C4D
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #A1AABB or adjust for 4.5:1 on bg #333C4D

### [CRITICAL] body > section.section-dark.fade-in > div.section-dark-inner > div.page-insight-box > h3

- **Text**: "The Bottom Line"
- **Color**: #2C3547 on #2C3547
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 17px, weight 600
- **Fix**: Change text to #959EB0 or adjust for 4.5:1 on bg #2C3547

### [CRITICAL] body > section.section-dark.fade-in > div.section-dark-inner > div.page-insight-box > p

- **Text**: "Consequential accounting works as a niche analytical tool — measuring displaceme"
- **Color**: #2C3547 on #2C3547
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 16px, weight 400
- **Fix**: Change text to #959EB0 or adjust for 4.5:1 on bg #2C3547

### [CRITICAL] section.section-dark.fade-in > div.section-dark-inner > div.page-insight-box > p > span.highlight

- **Text**: "primary procurement framework at scale"
- **Color**: #544A3B on #544A3B
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 16px, weight 600
- **Fix**: Change text to #C7BDAE or adjust for 4.5:1 on bg #544A3B

### [CRITICAL] body > section.section-dark.fade-in > div.section-dark-inner > div.page-insight-box > p

- **Text**: "Under SBTi-aligned pathways (~2–4% annual clean penetration growth), SPP exhaust"
- **Color**: #2C3547 on #2C3547
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 16px, weight 400
- **Fix**: Change text to #959EB0 or adjust for 4.5:1 on bg #2C3547

### [CRITICAL] section.section-dark.fade-in > div.section-dark-inner > div.page-insight-box > p > strong

- **Text**: "The coal/oil transition tells the timeline story."
- **Color**: #2C3547 on #2C3547
- **Ratio**: 1:1 (need 3:1)
- **Font**: 16px, weight 700
- **Fix**: Change text to #778092 or adjust for 3:1 on bg #2C3547

### [CRITICAL] section.section-dark.fade-in > div.section-dark-inner > div.page-insight-box > p > span.highlight

- **Text**: "By ~2035, the majority of U.S. load-weighted generation will have crossed this t"
- **Color**: #544A3B on #544A3B
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 16px, weight 600
- **Fix**: Change text to #C7BDAE or adjust for 4.5:1 on bg #544A3B

### [CRITICAL] body > section.section-dark.fade-in > div.section-dark-inner > div.page-insight-box > p

- **Text**: "The GHG Protocol's Scope 2 revision should recognize hourly attribution as the r"
- **Color**: #2C3547 on #2C3547
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 16px, weight 400
- **Fix**: Change text to #959EB0 or adjust for 4.5:1 on bg #2C3547

### [CRITICAL] html > body > section.sources-section.fade-in > h3

- **Text**: "Sources & Further Reading"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.sources-section.fade-in > ol > li

- **Text**: "GHG Protocol, "Scope 2 Guidance" (2015; revision expected ~2027). ghgprotocol.or"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > section.sources-section.fade-in > ol > li > a

- **Text**: "ghgprotocol.org"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.sources-section.fade-in > ol > li

- **Text**: "GHG Protocol, "Scope 2 Proposed Updates & Consequential Accounting Consultation""
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.sources-section.fade-in > ol > li

- **Text**: "Bjørn et al., "Ensuring low-emission electricity purchasing requires broader sys"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > section.sources-section.fade-in > ol > li > a

- **Text**: "Nature Communications"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.sources-section.fade-in > ol > li

- **Text**: "Clean Air Task Force, "Modernizing GHG Accounting Rules" (2024). catf.us"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > section.sources-section.fade-in > ol > li > a

- **Text**: "catf.us"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.sources-section.fade-in > ol > li

- **Text**: "EnergyTag, "Scope 2 for the Age of Deep Decarbonization" (2024). energytag.org"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > section.sources-section.fade-in > ol > li > a

- **Text**: "energytag.org"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.sources-section.fade-in > ol > li

- **Text**: "Xu et al., "System-level impacts of 24/7 carbon-free energy procurement" (2024)."
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > section.sources-section.fade-in > ol > li > a

- **Text**: "ScienceDirect"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.sources-section.fade-in > ol > li

- **Text**: "WattTime, "Hourly matching without additionality has little to no impact" (2024)"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > section.sources-section.fade-in > ol > li > a

- **Text**: "watttime.org"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.sources-section.fade-in > ol > li

- **Text**: "ISO-NE, "Winter Energy Security Improvements" and "Operational Fuel Security Ana"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.sources-section.fade-in > ol > li

- **Text**: "CAISO, "2024 Summer Loads and Resources Assessment.""
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.sources-section.fade-in > ol > li

- **Text**: "BNEF, "Corporate Energy Market Outlook" and "Corporate PPA Tracker" (2024)."
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.sources-section.fade-in > ol > li

- **Text**: "American Clean Power Association, "Clean Power Quarterly Market Report Q4 2024.""
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.sources-section.fade-in > ol > li

- **Text**: "GHGMI, Leggett & Gillenwater, "Limitations of Hourly Matching Claims for Scope 2"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > section.sources-section.fade-in > ol > li > a

- **Text**: "ghginstitute.org"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.sources-section.fade-in > ol > li

- **Text**: "RMI, "Carbon Accounting That Helps Companies Shift to Clean Energy Faster" (2024"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > section.sources-section.fade-in > ol > li > a

- **Text**: "rmi.org"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.sources-section.fade-in > ol > li

- **Text**: "UN Energy, "24/7 Carbon-Free Energy Compact" (2021). seforall.org"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > section.sources-section.fade-in > ol > li > a

- **Text**: "seforall.org"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.chart-section.section-light > div.chart-panel > div.chart-title

- **Text**: "Marginal Displacement Rate vs. Cumulative Clean Energy Penetration by ISO"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.chart-section.section-light > div.chart-panel > div.chart-title

- **Text**: "Capital Flow Under Consequential Accounting: Where the Money Goes"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.chart-section.section-light > div.chart-panel > div.chart-title

- **Text**: "Participation Threshold: When Perverse Incentives Dominate"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

## dashboard/design_preview.html

**24 failures**

### [CRITICAL] html > body.theme-observatory > div.theme-bar > button.theme-btn.active > span.theme-btn-fonts

- **Text**: "Plus Jakarta Sans + DM Sans"
- **Color**: #21719A on #0B263C
- **Ratio**: 2.89:1 (need 4.5:1)
- **Font**: 10px, weight 500
- **Fix**: Change text to #4494BD or adjust for 4.5:1 on bg #0B263C

### [CRITICAL] html > body.theme-observatory > section.hero-section > div.hero-inner > h2.hero-title.fade-in

- **Text**: "Every region has an optimal clean energy target"
- **Color**: #050810 on #050810
- **Ratio**: 1:1 (need 3:1)
- **Font**: 29px, weight 800
- **Fix**: Change text to #5A5D65 or adjust for 3:1 on bg #050810

### [CRITICAL] html > body.theme-observatory > section.hero-section > div.hero-inner > p.hero-subtitle.fade-in

- **Text**: "Grid decarbonization is cheaper than Direct Air Capture — up to a point.
       "
- **Color**: #050810 on #050810
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 17px, weight 400
- **Fix**: Change text to #787B83 or adjust for 4.5:1 on bg #050810

### [CRITICAL] div.hero-inner > div.stats-row > div.stat-card.fade-in > div.stat-value > span.stat-accent

- **Text**: "84–96%"
- **Color**: #171921 on #171921
- **Ratio**: 1:1 (need 3:1)
- **Font**: 38px, weight 700
- **Fix**: Change text to #676971 or adjust for 3:1 on bg #171921

### [CRITICAL] body.theme-observatory > section.content-section.section-light > div.narrative-chart-grid > div.narrative-block > span.finding-highlight

- **Text**: "Grid always cheaper than DAC · Wind + Solar dominate"
- **Color**: #0EA5E9 on #ECF8FD
- **Ratio**: 2.56:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #0073B7 or adjust for 4.5:1 on bg #ECF8FD

### [CRITICAL] section.hero-section > div.hero-inner > div.stats-row > div.stat-card.fade-in > div.stat-value

- **Text**: "84–96%"
- **Color**: #171921 on #171921
- **Ratio**: 1:1 (need 3:1)
- **Font**: 38px, weight 700
- **Fix**: Change text to #676971 or adjust for 3:1 on bg #171921

### [CRITICAL] section.hero-section > div.hero-inner > div.stats-row > div.stat-card.fade-in > div.stat-label

- **Text**: "Optimal crossover range under medium costs"
- **Color**: #171921 on #171921
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #80828A or adjust for 4.5:1 on bg #171921

### [CRITICAL] section.hero-section > div.hero-inner > div.stats-row > div.stat-card.fade-in > div.stat-value

- **Text**: "5,832"
- **Color**: #171921 on #171921
- **Ratio**: 1:1 (need 3:1)
- **Font**: 38px, weight 700
- **Fix**: Change text to #676971 or adjust for 3:1 on bg #171921

### [CRITICAL] section.hero-section > div.hero-inner > div.stats-row > div.stat-card.fade-in > div.stat-label

- **Text**: "Cost scenarios evaluated per region/threshold"
- **Color**: #171921 on #171921
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #80828A or adjust for 4.5:1 on bg #171921

### [CRITICAL] section.hero-section > div.hero-inner > div.stats-row > div.stat-card.fade-in > div.stat-value

- **Text**: "7 ISOs"
- **Color**: #171921 on #171921
- **Ratio**: 1:1 (need 3:1)
- **Font**: 38px, weight 700
- **Fix**: Change text to #676971 or adjust for 3:1 on bg #171921

### [CRITICAL] section.hero-section > div.hero-inner > div.stats-row > div.stat-card.fade-in > div.stat-label

- **Text**: "Regional power markets analyzed"
- **Color**: #171921 on #171921
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #80828A or adjust for 4.5:1 on bg #171921

### [CRITICAL] section.hero-section > div.hero-inner > div.stats-row > div.stat-card.fade-in > div.stat-value

- **Text**: "$47–412"
- **Color**: #171921 on #171921
- **Ratio**: 1:1 (need 3:1)
- **Font**: 38px, weight 700
- **Fix**: Change text to #676971 or adjust for 3:1 on bg #171921

### [CRITICAL] section.hero-section > div.hero-inner > div.stats-row > div.stat-card.fade-in > div.stat-label

- **Text**: "Marginal abatement cost range ($/tCO₂)"
- **Color**: #171921 on #171921
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #80828A or adjust for 4.5:1 on bg #171921

### [CRITICAL] section.dark-metrics-band.section-dark > div.metrics-inner > div.stats-row.metrics-stats > div.stat-card.fade-in > div.stat-value

- **Text**: "100%"
- **Color**: #171C2C on #171C2C
- **Ratio**: 1:1 (need 3:1)
- **Font**: 38px, weight 700
- **Fix**: Change text to #626777 or adjust for 3:1 on bg #171C2C

### [CRITICAL] section.dark-metrics-band.section-dark > div.metrics-inner > div.stats-row.metrics-stats > div.stat-card.fade-in > div.stat-label

- **Text**: "ERCOT: Grid always beats DAC"
- **Color**: #171C2C on #171C2C
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #808595 or adjust for 4.5:1 on bg #171C2C

### [CRITICAL] section.dark-metrics-band.section-dark > div.metrics-inner > div.stats-row.metrics-stats > div.stat-card.fade-in > div.stat-value

- **Text**: "18 GW"
- **Color**: #171C2C on #171C2C
- **Ratio**: 1:1 (need 3:1)
- **Font**: 38px, weight 700
- **Fix**: Change text to #626777 or adjust for 3:1 on bg #171C2C

### [CRITICAL] section.dark-metrics-band.section-dark > div.metrics-inner > div.stats-row.metrics-stats > div.stat-card.fade-in > div.stat-label

- **Text**: "No-regrets wind investment (ERCOT alone)"
- **Color**: #171C2C on #171C2C
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #808595 or adjust for 4.5:1 on bg #171C2C

### [CRITICAL] section.dark-metrics-band.section-dark > div.metrics-inner > div.stats-row.metrics-stats > div.stat-card.fade-in > div.stat-value

- **Text**: "15 pp"
- **Color**: #171C2C on #171C2C
- **Ratio**: 1:1 (need 3:1)
- **Font**: 38px, weight 700
- **Fix**: Change text to #626777 or adjust for 3:1 on bg #171C2C

### [CRITICAL] section.dark-metrics-band.section-dark > div.metrics-inner > div.stats-row.metrics-stats > div.stat-card.fade-in > div.stat-label

- **Text**: "Crossover uncertainty driven by firm gen costs"
- **Color**: #171C2C on #171C2C
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #808595 or adjust for 4.5:1 on bg #171C2C

### [CRITICAL] section.dark-metrics-band.section-dark > div.metrics-inner > div.stats-row.metrics-stats > div.stat-card.fade-in > div.stat-value

- **Text**: "$12–47B"
- **Color**: #171C2C on #171C2C
- **Ratio**: 1:1 (need 3:1)
- **Font**: 38px, weight 700
- **Fix**: Change text to #626777 or adjust for 3:1 on bg #171C2C

### [CRITICAL] section.dark-metrics-band.section-dark > div.metrics-inner > div.stats-row.metrics-stats > div.stat-card.fade-in > div.stat-label

- **Text**: "Cost of locking in today’s last-mile prices"
- **Color**: #171C2C on #171C2C
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #808595 or adjust for 4.5:1 on bg #171C2C

### [MAJOR] html > body.theme-observatory > div.theme-bar > button.theme-btn > span.theme-btn-fonts

- **Text**: "Fraunces + Inter"
- **Color**: #60636D on #0A0F1E
- **Ratio**: 3.18:1 (need 4.5:1)
- **Font**: 10px, weight 500
- **Fix**: Change text to #797C86 or adjust for 4.5:1 on bg #0A0F1E

### [MAJOR] html > body.theme-observatory > div.theme-bar > button.theme-btn > span.theme-btn-fonts

- **Text**: "Sora + Plus Jakarta Sans"
- **Color**: #60636D on #0A0F1E
- **Ratio**: 3.18:1 (need 4.5:1)
- **Font**: 10px, weight 500
- **Fix**: Change text to #797C86 or adjust for 4.5:1 on bg #0A0F1E

### [MAJOR] html > body.theme-observatory > div.theme-bar > button.theme-btn > span.theme-btn-fonts

- **Text**: "Bricolage Grotesque + Outfit"
- **Color**: #60636D on #0A0F1E
- **Ratio**: 3.18:1 (need 4.5:1)
- **Font**: 10px, weight 500
- **Fix**: Change text to #797C86 or adjust for 4.5:1 on bg #0A0F1E

## dashboard/gen_market_overview.html

**80 failures**

### [CRITICAL] div.card-grid-market > div.card > div > p > strong

- **Text**: "Advantages:"
- **Color**: #22C55E on #FFFFFF
- **Ratio**: 2.28:1 (need 3:1)
- **Font**: 16px, weight 700
- **Fix**: Change text to #09AC45 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div.card-grid-market > div.card > div > p > strong

- **Text**: "Advantages:"
- **Color**: #22C55E on #FFFFFF
- **Ratio**: 2.28:1 (need 3:1)
- **Font**: 16px, weight 700
- **Fix**: Change text to #09AC45 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] body > div.main-content > section.section.fade-in > div.section-header > h2

- **Text**: "The Top 15: At a Glance"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 3:1)
- **Font**: 24px, weight 700
- **Fix**: Change text to #8F9093 or adjust for 3:1 on bg #F8F9FC

### [CRITICAL] body > div.main-content > section.section.fade-in > div.section-header > p

- **Text**: "Ranked by annual electricity generation (TWh). Constellation Energy figures refl"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 17px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div > table#companyTable > thead > tr > th

- **Text**: "Company"
- **Color**: #1A2744 on #1A2744
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #8390AD or adjust for 4.5:1 on bg #1A2744

### [CRITICAL] div > table#companyTable > thead > tr > th

- **Text**: "Type"
- **Color**: #1A2744 on #1A2744
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #8390AD or adjust for 4.5:1 on bg #1A2744

### [CRITICAL] div > table#companyTable > thead > tr > th

- **Text**: "Gen (TWh)"
- **Color**: #1A2744 on #1A2744
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #8390AD or adjust for 4.5:1 on bg #1A2744

### [CRITICAL] div > table#companyTable > thead > tr > th

- **Text**: "CO₂ (Mt/yr)"
- **Color**: #1A2744 on #1A2744
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #8390AD or adjust for 4.5:1 on bg #1A2744

### [CRITICAL] div > table#companyTable > thead > tr > th

- **Text**: "Intensity (kg/MWh)"
- **Color**: #1A2744 on #1A2744
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #8390AD or adjust for 4.5:1 on bg #1A2744

### [CRITICAL] div > table#companyTable > thead > tr > th

- **Text**: "Generation Mix"
- **Color**: #1A2744 on #1A2744
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #8390AD or adjust for 4.5:1 on bg #1A2744

### [CRITICAL] html > body > div.main-content > section.section.fade-in > p.citation

- **Text**: "Notes: Generation and emissions estimates based on eGRID 2023 plant-level data a"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] section.story-section.fade-in > div.card-grid > div.card > p > strong

- **Text**: "Evergy, AEP, BH Energy, Vistra, PPL"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] section.story-section.fade-in > div.card-grid > div.card > p > strong

- **Text**: "Entergy, Dominion"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] section.story-section.fade-in > div.card-grid > div.card > p > strong

- **Text**: "Duke, Southern, Xcel, DTE, WEC, AES"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] section.story-section.fade-in > div.card-grid > div.card > p > strong

- **Text**: "Constellation, NextEra"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] body > div.main-content > section.section.fade-in > div.section-header > h2

- **Text**: "The Sector-Wide Transformation: 2012 to 2024"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 3:1)
- **Font**: 24px, weight 700
- **Fix**: Change text to #8F9093 or adjust for 3:1 on bg #F8F9FC

### [CRITICAL] body > div.main-content > section.section.fade-in > div.section-header > p

- **Text**: "The U.S. power sector has undergone a remarkable fuel-mix shift in just 12 years"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 17px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] div > table.company-table > thead > tr > th

- **Text**: "Fuel Source"
- **Color**: #1A2744 on #1A2744
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #8390AD or adjust for 4.5:1 on bg #1A2744

### [CRITICAL] div > table.company-table > thead > tr > th

- **Text**: "2012 Share"
- **Color**: #1A2744 on #1A2744
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #8390AD or adjust for 4.5:1 on bg #1A2744

### [CRITICAL] div > table.company-table > thead > tr > th

- **Text**: "2024 Share"
- **Color**: #1A2744 on #1A2744
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #8390AD or adjust for 4.5:1 on bg #1A2744

### [CRITICAL] div > table.company-table > thead > tr > th

- **Text**: "Change"
- **Color**: #1A2744 on #1A2744
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #8390AD or adjust for 4.5:1 on bg #1A2744

### [CRITICAL] div > table.company-table > tbody > tr > td

- **Text**: "Coal"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] table.company-table > tbody > tr > td > strong

- **Text**: "Coal"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div > table.company-table > tbody > tr > td

- **Text**: "37%"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div > table.company-table > tbody > tr > td

- **Text**: "~15%"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div > table.company-table > tbody > tr > td

- **Text**: "↓ 22 pp"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div > table.company-table > tbody > tr > td

- **Text**: "Natural Gas"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] table.company-table > tbody > tr > td > strong

- **Text**: "Natural Gas"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div > table.company-table > tbody > tr > td

- **Text**: "30%"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div > table.company-table > tbody > tr > td

- **Text**: "~43%"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div > table.company-table > tbody > tr > td

- **Text**: "↑ 13 pp"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div > table.company-table > tbody > tr > td

- **Text**: "Wind + Solar"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] table.company-table > tbody > tr > td > strong

- **Text**: "Wind + Solar"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div > table.company-table > tbody > tr > td

- **Text**: "<4%"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div > table.company-table > tbody > tr > td

- **Text**: "~16%"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div > table.company-table > tbody > tr > td

- **Text**: "↑ 12+ pp"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div > table.company-table > tbody > tr > td

- **Text**: "Nuclear"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] table.company-table > tbody > tr > td > strong

- **Text**: "Nuclear"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div > table.company-table > tbody > tr > td

- **Text**: "~19%"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div > table.company-table > tbody > tr > td

- **Text**: "~18%"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div > table.company-table > tbody > tr > td

- **Text**: "≈ stable"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > div.main-content > section.section.fade-in > div.callout.warning > p

- **Text**: "Despite 40% CO₂ reduction from 2007 peak, emissions have been essentially flat s"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.main-content > section.section.fade-in > div.callout.warning > p > strong

- **Text**: "The stall:"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div.main-content > section.section.fade-in > div.callout.warning > p > strong

- **Text**: "rose 3.8% in 2025"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] body > div.main-content > section.section.fade-in > div.section-header > h2

- **Text**: "Marginal Abatement Cost by Fleet Archetype"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 3:1)
- **Font**: 24px, weight 700
- **Fix**: Change text to #8F9093 or adjust for 3:1 on bg #F8F9FC

### [CRITICAL] body > div.main-content > section.section.fade-in > div.section-header > p

- **Text**: "The cost of each additional tonne of CO₂ avoided increases as generators pursue "
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 17px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] body > div.main-content > section.section.fade-in > div.chart-container.chart-panel > h3

- **Text**: "CO₂ Abatement Cost Curve: Cheap Early Wins, Expensive Last Mile"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 17px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] body > div.main-content > section.section.fade-in > div.chart-container.chart-panel > p.citation

- **Text**: "Based on Lazard LCOE 2024, NREL ATB 2024, and hourly CFE optimizer cost curves f"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > div.main-content > section.section.fade-in > div.section-header > h2

- **Text**: "Historical Emissions Trajectories: 2005–2024"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 3:1)
- **Font**: 24px, weight 700
- **Fix**: Change text to #8F9093 or adjust for 3:1 on bg #F8F9FC

### [CRITICAL] body > div.main-content > section.section.fade-in > div.section-header > p

- **Text**: "How have the top generators' CO₂ emission rates changed over the past two decade"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 17px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] body > div.main-content > section.section.fade-in > div.chart-container.chart-panel > h3

- **Text**: "Absolute CO₂ Emissions Trajectory: Top 15 Combined"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 17px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] body > div.main-content > section.section.fade-in > div.chart-container.chart-panel > p.citation

- **Text**: "Top 15 generators combined annual CO₂ emissions. Based on MJ Bradley/ERM data sh"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > div.main-content > section.section.fade-in > h2

- **Text**: "Explore the Full Analysis"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 3:1)
- **Font**: 25px, weight 700
- **Fix**: Change text to #8F9093 or adjust for 3:1 on bg #F8F9FC

### [CRITICAL] html > body > div.main-content > section.section.fade-in > p

- **Text**: "This market overview is the first of five pages examining IPP decarbonization pa"
- **Color**: #F8F9FC on #F8F9FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 16px, weight 400
- **Fix**: Change text to #717275 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] body > div.main-content > section.section.fade-in > div > a.pill

- **Text**: "Policy & Market Conditions"
- **Color**: #FDFDFE on #FDFDFE
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #717172 or adjust for 4.5:1 on bg #FDFDFE

### [CRITICAL] body > div.main-content > section.section.fade-in > div > a.pill

- **Text**: "Target Setting"
- **Color**: #FDFDFE on #FDFDFE
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #717172 or adjust for 4.5:1 on bg #FDFDFE

### [CRITICAL] body > div.main-content > section.section.fade-in > div > a.pill

- **Text**: "Regional Pathways"
- **Color**: #FDFDFE on #FDFDFE
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #717172 or adjust for 4.5:1 on bg #FDFDFE

### [CRITICAL] body > div.main-content > section.section.fade-in > div > a.pill

- **Text**: "IPP Deep Dives"
- **Color**: #FDFDFE on #FDFDFE
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #717172 or adjust for 4.5:1 on bg #FDFDFE

### [CRITICAL] body > div.main-content > section.section.fade-in > div > a.pill.active

- **Text**: "Cost Optimizer"
- **Color**: #1A2744 on #1A2744
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #8390AD or adjust for 4.5:1 on bg #1A2744

### [CRITICAL] body > div.main-content > div.stat-row.fade-in > div.stat-tile > div.stat-label

- **Text**: "Companies Analyzed"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 500
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > div.main-content > div.stat-row.fade-in > div.stat-tile > div.stat-sublabel

- **Text**: "Top generators by TWh"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > div.main-content > div.stat-row.fade-in > div.stat-tile > div.stat-label

- **Text**: "TWh Generated"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 500
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > div.main-content > div.stat-row.fade-in > div.stat-tile > div.stat-sublabel

- **Text**: "~40% of US total"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > div.main-content > div.stat-row.fade-in > div.stat-tile > div.stat-label

- **Text**: "Million Tonnes CO₂"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 500
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > div.main-content > div.stat-row.fade-in > div.stat-tile > div.stat-sublabel

- **Text**: "Annual Scope 1 emissions"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > div.main-content > div.stat-row.fade-in > div.stat-tile > div.stat-label

- **Text**: "Avg. kg CO₂/MWh"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 500
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > div.main-content > div.stat-row.fade-in > div.stat-tile > div.stat-sublabel

- **Text**: "Weighted fleet intensity"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.main-content > section.section.fade-in > div.stat-row > div.stat-tile > div.stat-value.success

- **Text**: "↓40%"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 26px, weight 800
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div.main-content > section.section.fade-in > div.stat-row > div.stat-tile > div.stat-label

- **Text**: "CO₂ from 2007 Peak"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 500
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.main-content > section.section.fade-in > div.stat-row > div.stat-tile > div.stat-sublabel

- **Text**: "Top 100 producers"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.main-content > section.section.fade-in > div.stat-row > div.stat-tile > div.stat-value.success

- **Text**: "↓96%"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 26px, weight 800
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div.main-content > section.section.fade-in > div.stat-row > div.stat-tile > div.stat-label

- **Text**: "SO₂ Since 1990"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 500
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.main-content > section.section.fade-in > div.stat-row > div.stat-tile > div.stat-sublabel

- **Text**: "A Clean Air Act success"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.main-content > section.section.fade-in > div.stat-row > div.stat-tile > div.stat-value.success

- **Text**: "↓90%"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 26px, weight 800
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div.main-content > section.section.fade-in > div.stat-row > div.stat-tile > div.stat-label

- **Text**: "NOₓ Since 1990"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 500
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.main-content > section.section.fade-in > div.stat-row > div.stat-tile > div.stat-sublabel

- **Text**: "Regulation-driven"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.main-content > section.section.fade-in > div.stat-row > div.stat-tile > div.stat-value.highlight

- **Text**: "44%"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 26px, weight 800
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div.main-content > section.section.fade-in > div.stat-row > div.stat-tile > div.stat-label

- **Text**: "Nuclear’s Share of Zero-Carbon"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 500
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.main-content > section.section.fade-in > div.stat-row > div.stat-tile > div.stat-sublabel

- **Text**: "Still the clean energy backbone"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [MINOR] div.main-content > section.story-section.fade-in > div.card-grid-market > div.card > h3

- **Text**: "Vertically Integrated Utilities"
- **Color**: #0284C7 on #FFFFFF
- **Ratio**: 4.1:1 (need 4.5:1)
- **Font**: 18px, weight 600
- **Fix**: Change text to #007ABD or adjust for 4.5:1 on bg #FFFFFF

## dashboard/gen_policy_conditions.html

**2 failures**

### [CRITICAL] section.story-section.fade-in > div.story-content > div.card-grid > div.stat-card > div.stat-value

- **Text**: "$25–35"
- **Color**: #22C55E on #F9FAFB
- **Ratio**: 2.18:1 (need 3:1)
- **Font**: 29px, weight 800
- **Fix**: Change text to #04A740 or adjust for 3:1 on bg #F9FAFB

### [CRITICAL] section.story-section.fade-in > div.story-content > div.card-grid > div.stat-card > div.stat-value

- **Text**: "$44–66"
- **Color**: #F59E0B on #F9FAFB
- **Ratio**: 2.06:1 (need 3:1)
- **Font**: 29px, weight 800
- **Fix**: Change text to #D27B00 or adjust for 3:1 on bg #F9FAFB

## dashboard/gen_regional_pathways.html

**6 failures**

### [CRITICAL] div#regionalContent > section#gapSection > div.grid-2up > div.chart-panel > h3

- **Text**: "Emission Trajectories: Market Fan vs. Net-Zero"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div#regionalContent > section#gapSection > div.grid-2up > div.chart-panel > h3#deltaTitle

- **Text**: "What Net-Zero Requires Beyond the Market"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div#regionalContent > section#buildoutSection > div#mixGridArea > div#mixPanelFac > h3#mixFacTitle

- **Text**: "Facilitating Conditions"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div#regionalContent > section#buildoutSection > div#mixGridArea > div#mixPanelChal > h3#mixChalTitle

- **Text**: "Challenging Conditions"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div#regionalContent > section#costSection > div.grid-2up > div.chart-panel > h3#carbonChartTitle

- **Text**: "Carbon Shadow Price & DAC Cost"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div#regionalContent > section#costSection > div.grid-2up > div.chart-panel > h3#costChartTitle

- **Text**: "Policy Subsidy Trajectory"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

## dashboard/gen_target_setting.html

**9 failures**

### [CRITICAL] section.content-section.fade-in > div.key-insights > div.key-insight-item > div > strong

- **Text**: "1 of 15 Validated"
- **Color**: #1A2744 on #1A2744
- **Ratio**: 1:1 (need 3:1)
- **Font**: 16px, weight 700
- **Fix**: Change text to #65728F or adjust for 3:1 on bg #1A2744

### [CRITICAL] section.content-section.fade-in > div.key-insights > div.key-insight-item > div > strong

- **Text**: "SBTi vs. SMARTargets"
- **Color**: #1A2744 on #1A2744
- **Ratio**: 1:1 (need 3:1)
- **Font**: 16px, weight 700
- **Fix**: Change text to #65728F or adjust for 3:1 on bg #1A2744

### [CRITICAL] section.content-section.fade-in > div.key-insights > div.key-insight-item > div > strong

- **Text**: "Scope 2 Dual Reporting"
- **Color**: #1A2744 on #1A2744
- **Ratio**: 1:1 (need 3:1)
- **Font**: 16px, weight 700
- **Fix**: Change text to #65728F or adjust for 3:1 on bg #1A2744

### [CRITICAL] section.content-section.fade-in > div.key-insights > div.key-insight-item > div > strong

- **Text**: "CDP Exodus"
- **Color**: #1A2744 on #1A2744
- **Ratio**: 1:1 (need 3:1)
- **Font**: 16px, weight 700
- **Fix**: Change text to #65728F or adjust for 3:1 on bg #1A2744

### [CRITICAL] div.main-content > section.story-section.fade-in > div.story-content > div.chart-panel > h3.section-subtitle

- **Text**: "Framework Trajectory Comparison: SBTi vs. SMARTargets AT"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 17px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] body > div.main-content > section.content-section.fade-in > div.key-insights > div.key-insight-item

- **Text**: "1 of 15 ValidatedOnly AES (Nov 2025) has SBTi-validated science-based targets am"
- **Color**: #1A2744 on #1A2744
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 16px, weight 400
- **Fix**: Change text to #8390AD or adjust for 4.5:1 on bg #1A2744

### [CRITICAL] body > div.main-content > section.content-section.fade-in > div.key-insights > div.key-insight-item

- **Text**: "SBTi vs. SMARTargetsSBTi requires ~85% intensity reduction by 2035 — SMARTargets"
- **Color**: #1A2744 on #1A2744
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 16px, weight 400
- **Fix**: Change text to #8390AD or adjust for 4.5:1 on bg #1A2744

### [CRITICAL] body > div.main-content > section.content-section.fade-in > div.key-insights > div.key-insight-item

- **Text**: "Scope 2 Dual ReportingGHG Protocol Scope 2 dual reporting creates both opportuni"
- **Color**: #1A2744 on #1A2744
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 16px, weight 400
- **Fix**: Change text to #8390AD or adjust for 4.5:1 on bg #1A2744

### [CRITICAL] body > div.main-content > section.content-section.fade-in > div.key-insights > div.key-insight-item

- **Text**: "CDP Exodus8 of 15 largest generators stopped CDP climate reporting in 2024 — a s"
- **Color**: #1A2744 on #1A2744
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 16px, weight 400
- **Fix**: Change text to #8390AD or adjust for 4.5:1 on bg #1A2744

## dashboard/grid_animation.html

**7 failures**

### [CRITICAL] div.grid-viz-wrapper > div.scenario-intro > div.scenario-cards-row > div.scenario-pick > span.badge

- **Text**: "Mar 8–9, 2025"
- **Color**: #06B6D4 on #E6F7FA
- **Ratio**: 2.21:1 (need 4.5:1)
- **Font**: 11px, weight 600
- **Fix**: Change text to #007A98 or adjust for 4.5:1 on bg #E6F7FA

### [MINOR] div.grid-viz-wrapper > div.scenario-intro > div.scenario-cards-row > div.scenario-pick.selected > span.badge.badge-danger

- **Text**: "Feb 10–20, 2021"
- **Color**: #DC2626 on #FDECEC
- **Ratio**: 4.22:1 (need 4.5:1)
- **Font**: 11px, weight 600
- **Fix**: Change text to #D21C1C or adjust for 4.5:1 on bg #FDECEC

### [MINOR] div.viz-main > div.map-wrap > div#mobileStatsOverlay > div.mobile-stat > span.ms-label

- **Text**: "Demand"
- **Color**: #6B7280 on #F4F4F4
- **Ratio**: 4.4:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F4F4F4

### [MINOR] div.viz-main > div.map-wrap > div#mobileStatsOverlay > div.mobile-stat > span.ms-label

- **Text**: "Gen"
- **Color**: #6B7280 on #F4F4F4
- **Ratio**: 4.4:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F4F4F4

### [MINOR] div.viz-main > div.map-wrap > div#mobileStatsOverlay > div.mobile-stat > span.ms-label

- **Text**: "Renew"
- **Color**: #6B7280 on #F4F4F4
- **Ratio**: 4.4:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F4F4F4

### [MINOR] div.viz-main > div.map-wrap > div#mobileStatsOverlay > div.mobile-stat > span.ms-label

- **Text**: "Fossil"
- **Color**: #6B7280 on #F4F4F4
- **Ratio**: 4.4:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F4F4F4

### [MINOR] div.viz-main > div.map-wrap > div#mobileStatsOverlay > div.mobile-stat > span.ms-label

- **Text**: "CO₂"
- **Color**: #6B7280 on #F4F4F4
- **Ratio**: 4.4:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F4F4F4

## dashboard/hourly_comparison.html

**3 failures**

### [CRITICAL] div.content > div#stratDefs > div.strategy-cards-3 > div#card2b > span.strat-tag.tag-2b

- **Text**: "Strategy 2B"
- **Color**: #D97706 on #FAF2E5
- **Ratio**: 2.86:1 (need 4.5:1)
- **Font**: 11px, weight 700
- **Fix**: Change text to #B14F00 or adjust for 4.5:1 on bg #FAF2E5

### [CRITICAL] div.content > div#stratDefs > div.strategy-cards-3 > div#card2c > span.strat-tag.tag-2c

- **Text**: "Strategy 2C"
- **Color**: #16A34A on #E4F6ED
- **Ratio**: 2.93:1 (need 4.5:1)
- **Font**: 11px, weight 700
- **Fix**: Change text to #008027 or adjust for 4.5:1 on bg #E4F6ED

### [MINOR] div.content > div#stratDefs > div.strategy-cards-3 > div#card2a > span.strat-tag.tag-2a

- **Text**: "Strategy 2A"
- **Color**: #DC2626 on #F9E9EB
- **Ratio**: 4.1:1 (need 4.5:1)
- **Font**: 11px, weight 700
- **Fix**: Change text to #D21C1C or adjust for 4.5:1 on bg #F9E9EB

## dashboard/ipp_aes.html

**4 failures**

### [CRITICAL] main > section#risk > div.grid-2col > div.card > h3

- **Text**: "Profit Trajectory (P10/P50/P90)"
- **Color**: #1A2744 on #2C3344
- **Ratio**: 1.17:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #929FBC or adjust for 4.5:1 on bg #2C3344

### [CRITICAL] main > section#risk > div.grid-2col > div.card > h3

- **Text**: "Operating vs. Stranded Capacity"
- **Color**: #1A2744 on #2C3344
- **Ratio**: 1.17:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #929FBC or adjust for 4.5:1 on bg #2C3344

### [CRITICAL] body > div#reportContent > main > section#risk > h3

- **Text**: "Breakeven Analysis by Dimension"
- **Color**: #1A2744 on #0F172A
- **Ratio**: 1.21:1 (need 4.5:1)
- **Font**: 16px, weight 600
- **Fix**: Change text to #74819E or adjust for 4.5:1 on bg #0F172A

### [CRITICAL] main > section#strategic > div.grid-2col > div.card > h3

- **Text**: "Robustness Scorecard"
- **Color**: #1A2744 on #2C3344
- **Ratio**: 1.17:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #929FBC or adjust for 4.5:1 on bg #2C3344

## dashboard/ipp_constellation.html

**1 failures**

### [CRITICAL] main > section#strategic > div.grid-2col > div.card > h3

- **Text**: "Robustness Scorecard"
- **Color**: #1A2744 on #2C3344
- **Ratio**: 1.17:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #929FBC or adjust for 4.5:1 on bg #2C3344

## dashboard/ipp_nextera.html

**4 failures**

### [CRITICAL] main > section#risk > div.grid-2col > div.card > h3

- **Text**: "Profit Trajectory (P10/P50/P90)"
- **Color**: #1A2744 on #2C3344
- **Ratio**: 1.17:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #929FBC or adjust for 4.5:1 on bg #2C3344

### [CRITICAL] main > section#risk > div.grid-2col > div.card > h3

- **Text**: "Operating vs. Stranded Capacity"
- **Color**: #1A2744 on #2C3344
- **Ratio**: 1.17:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #929FBC or adjust for 4.5:1 on bg #2C3344

### [CRITICAL] body > div#reportContent > main > section#risk > h3

- **Text**: "Breakeven Analysis by Dimension"
- **Color**: #1A2744 on #0F172A
- **Ratio**: 1.21:1 (need 4.5:1)
- **Font**: 16px, weight 600
- **Fix**: Change text to #74819E or adjust for 4.5:1 on bg #0F172A

### [CRITICAL] main > section#strategic > div.grid-2col > div.card > h3

- **Text**: "Robustness Scorecard"
- **Color**: #1A2744 on #2C3344
- **Ratio**: 1.17:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #929FBC or adjust for 4.5:1 on bg #2C3344

## dashboard/ipp_nrg.html

**4 failures**

### [CRITICAL] main > section#risk > div.grid-2col > div.card > h3

- **Text**: "Profit Trajectory (P10/P50/P90)"
- **Color**: #1A2744 on #2C3344
- **Ratio**: 1.17:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #929FBC or adjust for 4.5:1 on bg #2C3344

### [CRITICAL] main > section#risk > div.grid-2col > div.card > h3

- **Text**: "Operating vs. Stranded Capacity"
- **Color**: #1A2744 on #2C3344
- **Ratio**: 1.17:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #929FBC or adjust for 4.5:1 on bg #2C3344

### [CRITICAL] body > div#reportContent > main > section#risk > h3

- **Text**: "Breakeven Analysis by Dimension"
- **Color**: #1A2744 on #0F172A
- **Ratio**: 1.21:1 (need 4.5:1)
- **Font**: 16px, weight 600
- **Fix**: Change text to #74819E or adjust for 4.5:1 on bg #0F172A

### [CRITICAL] main > section#strategic > div.grid-2col > div.card > h3

- **Text**: "Robustness Scorecard"
- **Color**: #1A2744 on #2C3344
- **Ratio**: 1.17:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #929FBC or adjust for 4.5:1 on bg #2C3344

## dashboard/ipp_pseg.html

**4 failures**

### [CRITICAL] main > section#risk > div.grid-2col > div.card > h3

- **Text**: "Profit Trajectory (P10/P50/P90)"
- **Color**: #1A2744 on #2C3344
- **Ratio**: 1.17:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #929FBC or adjust for 4.5:1 on bg #2C3344

### [CRITICAL] main > section#risk > div.grid-2col > div.card > h3

- **Text**: "Operating vs. Stranded Capacity"
- **Color**: #1A2744 on #2C3344
- **Ratio**: 1.17:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #929FBC or adjust for 4.5:1 on bg #2C3344

### [CRITICAL] body > div#reportContent > main > section#risk > h3

- **Text**: "Breakeven Analysis by Dimension"
- **Color**: #1A2744 on #0F172A
- **Ratio**: 1.21:1 (need 4.5:1)
- **Font**: 16px, weight 600
- **Fix**: Change text to #74819E or adjust for 4.5:1 on bg #0F172A

### [CRITICAL] main > section#strategic > div.grid-2col > div.card > h3

- **Text**: "Robustness Scorecard"
- **Color**: #1A2744 on #2C3344
- **Ratio**: 1.17:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #929FBC or adjust for 4.5:1 on bg #2C3344

## dashboard/ipp_smartargets.html

**5 failures**

### [CRITICAL] html > body > div.global-controls > button#passiveBtn

- **Text**: "Passive"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > div#fanSection > p.section-desc

- **Text**: "P10/P50/P90 fan bands across all parametric scenarios. Purple = passive fleet, g"
- **Color**: #FEFEFF on #F8F9FC
- **Ratio**: 1.05:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] html > body > div#fanSection > div#fanNarrative > p#fanNarrativeText

- **Text**: "Select a company to see fleet-specific analysis."
- **Color**: #FEFEFF on #F9FAFC
- **Ratio**: 1.04:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F9FAFC

### [CRITICAL] html > body > div#breakevenSection > p.section-desc

- **Text**: "Which parametric dimensions produce profitable decarbonization vs stranded asset"
- **Color**: #FEFEFF on #F8F9FC
- **Ratio**: 1.05:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F8F9FC

### [CRITICAL] html > body > div#breakevenSection > div.narrative-box > p#breakevenNarrativeText

- **Text**: "Breakeven insights will appear here."
- **Color**: #FEFEFF on #F9FAFC
- **Ratio**: 1.04:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #727273 or adjust for 4.5:1 on bg #F9FAFC

## dashboard/ipp_talen.html

**4 failures**

### [CRITICAL] main > section#risk > div.grid-2col > div.card > h3

- **Text**: "Profit Trajectory (P10/P50/P90)"
- **Color**: #1A2744 on #2C3344
- **Ratio**: 1.17:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #929FBC or adjust for 4.5:1 on bg #2C3344

### [CRITICAL] main > section#risk > div.grid-2col > div.card > h3

- **Text**: "Operating vs. Stranded Capacity"
- **Color**: #1A2744 on #2C3344
- **Ratio**: 1.17:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #929FBC or adjust for 4.5:1 on bg #2C3344

### [CRITICAL] body > div#reportContent > main > section#risk > h3

- **Text**: "Breakeven Analysis by Dimension"
- **Color**: #1A2744 on #0F172A
- **Ratio**: 1.21:1 (need 4.5:1)
- **Font**: 16px, weight 600
- **Fix**: Change text to #74819E or adjust for 4.5:1 on bg #0F172A

### [CRITICAL] main > section#strategic > div.grid-2col > div.card > h3

- **Text**: "Robustness Scorecard"
- **Color**: #1A2744 on #2C3344
- **Ratio**: 1.17:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #929FBC or adjust for 4.5:1 on bg #2C3344

## dashboard/ipp_vistra.html

**4 failures**

### [CRITICAL] main > section#risk > div.grid-2col > div.card > h3

- **Text**: "Profit Trajectory (P10/P50/P90)"
- **Color**: #1A2744 on #2C3344
- **Ratio**: 1.17:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #929FBC or adjust for 4.5:1 on bg #2C3344

### [CRITICAL] main > section#risk > div.grid-2col > div.card > h3

- **Text**: "Operating vs. Stranded Capacity"
- **Color**: #1A2744 on #2C3344
- **Ratio**: 1.17:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #929FBC or adjust for 4.5:1 on bg #2C3344

### [CRITICAL] body > div#reportContent > main > section#risk > h3

- **Text**: "Breakeven Analysis by Dimension"
- **Color**: #1A2744 on #0F172A
- **Ratio**: 1.21:1 (need 4.5:1)
- **Font**: 16px, weight 600
- **Fix**: Change text to #74819E or adjust for 4.5:1 on bg #0F172A

### [CRITICAL] main > section#strategic > div.grid-2col > div.card > h3

- **Text**: "Robustness Scorecard"
- **Color**: #1A2744 on #2C3344
- **Ratio**: 1.17:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #929FBC or adjust for 4.5:1 on bg #2C3344

## dashboard/lmp_trends.html

**118 failures**

### [CRITICAL] div.paradox-callout > div.insight-box.insight-warn > div > p > a

- **Text**: "See the full nuclear economics analysis below →"
- **Color**: #38BDF8 on #FFFFFF
- **Ratio**: 2.14:1 (need 4.5:1)
- **Font**: 14px, weight 600
- **Fix**: Change text to #007CB7 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > div.region-pills-bar > div.region-pills-inner > button.region-pill.active

- **Text**: "PJM"
- **Color**: #FFFFFF on #0EA5E9
- **Ratio**: 2.77:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #323232 or adjust for 4.5:1 on bg #0EA5E9

### [CRITICAL] section.hero-section.section-light > div.hero-inner > div.hero-text > h1#heroTitle > span.accent

- **Text**: "halves wholesale prices"
- **Color**: #0EA5E9 on #F8F9FC
- **Ratio**: 2.63:1 (need 3:1)
- **Font**: 23px, weight 800
- **Fix**: Change text to #0096DA or adjust for 3:1 on bg #F8F9FC

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div#sect1Narrative > div.narrative-card > span.section-tag

- **Text**: "Price Drivers"
- **Color**: #ECF8FD on #ECF8FD
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 700
- **Fix**: Change text to #657176 or adjust for 4.5:1 on bg #ECF8FD

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div#sect1Narrative > div.narrative-card > h3

- **Text**: "Fuel prices explain 88–100% of wholesale cost variation — until they don't"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 800
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div#sect1Narrative > div.narrative-card > p

- **Text**: "From 50% to 95% clean energy, the single biggest determinant of wholesale prices"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section-inner > div#sect1Narrative > div.narrative-card > p > strong

- **Text**: "100%"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section-inner > div#sect1Narrative > div.narrative-card > p > strong

- **Text**: "88%"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div#sect1Narrative > div.narrative-card > p

- **Text**: "The gap between Low and High fuel scenarios in PJM is massive: at 80% clean, Low"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div#sect1Narrative > div.narrative-card > span.stat-badge.stat-badge-blue

- **Text**: "Low fuel: $17/MWh at 80% · High fuel: $37/MWh at 80%"
- **Color**: #ECF8FD on #ECF8FD
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #657176 or adjust for 4.5:1 on bg #ECF8FD

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div#sect1Narrative > div.narrative-card > h3

- **Text**: "At ≥99.99% clean, the game changes completely"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 800
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div#sect1Narrative > div.narrative-card > p

- **Text**: "When clean energy reaches ≥99.99%, fuel prices become irrelevant — there are no "
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section-inner > div#sect1Narrative > div.narrative-card > p > strong

- **Text**: "renewable costs"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section-inner > div#sect1Narrative > div.narrative-card > p > strong

- **Text**: "fuel level"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section-inner > div#sect1Narrative > div.narrative-card > p > strong

- **Text**: "transmission"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section-inner > div#sect1Narrative > div.narrative-card > p > strong

- **Text**: "45Q policy"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section-inner > div#sect1Narrative > div.narrative-card > p > strong

- **Text**: "CO₂ price"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div#sect1Narrative > div.narrative-card > p

- **Text**: "This is the structural break in the wholesale market. Everything that mattered b"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div#sect1Narrative > div.narrative-card > span.stat-badge.stat-badge-amber

- **Text**: "At ≥99.99%: No single factor explains >15% of price variance"
- **Color**: #FEF7EB on #FEF7EB
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #777064 or adjust for 4.5:1 on bg #FEF7EB

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div#sect2Narrative > div.narrative-card > span.section-tag

- **Text**: "Market Structure"
- **Color**: #EDFAF2 on #EDFAF2
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 700
- **Fix**: Change text to #66736B or adjust for 4.5:1 on bg #EDFAF2

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div#sect2Narrative > div.narrative-card > h3

- **Text**: "PJM zero-price hours explode from 265 to 7,980 — and the market flips inside out"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 800
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div#sect2Narrative > div.narrative-card > p

- **Text**: "At 50% clean, PJM sees about 265 zero-or-negative-price hours per year — roughly"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section-inner > div#sect2Narrative > div.narrative-card > p > strong

- **Text**: "4,268 hours"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section-inner > div#sect2Narrative > div.narrative-card > p > strong

- **Text**: "7,980 hours"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div#sect2Narrative > div.narrative-card > p

- **Text**: "This isn't a gradual shift. The curve accelerates sharply past 75%, as each incr"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div#sect2Narrative > div.narrative-card > span.stat-badge.stat-badge-green

- **Text**: "265 → 7,980 zero-price hours: a 30× increase"
- **Color**: #EDFAF2 on #EDFAF2
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #66736B or adjust for 4.5:1 on bg #EDFAF2

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div#sect2Narrative > div.narrative-card > h3

- **Text**: "Scarcity events decline but never disappear"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 800
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div#sect2Narrative > div.narrative-card > p

- **Text**: "Hours above $200/MWh drop from a median of 68 at 50% clean to 10 at ≥99.99% — bu"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div#sect2Narrative > div.narrative-card > p

- **Text**: "The P90 of scarcity hours (worst 10% of scenarios) remains above 40 even at ≥99."
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div#sect2Narrative > div.narrative-card > span.stat-badge.stat-badge-red

- **Text**: "Scarcity hours: 68 → 10 median, but P90 stays at 41"
- **Color**: #FEF4F4 on #FEF4F4
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #776D6D or adjust for 4.5:1 on bg #FEF4F4

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div#sect3Narrative > div.narrative-card > span.section-tag

- **Text**: "Price Structure"
- **Color**: #FEF7EB on #FEF7EB
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 700
- **Fix**: Change text to #777064 or adjust for 4.5:1 on bg #FEF7EB

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div#sect3Narrative > div.narrative-card > h3

- **Text**: "The peak premium persists — and that's the arbitrage signal"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 800
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div#sect3Narrative > div.narrative-card > p

- **Text**: "Off-peak prices crash faster than peak: at 95% clean, off-peak hits $14/MWh whil"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section-inner > div#sect3Narrative > div.narrative-card > p > em

- **Text**: "widens"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div#sect3Narrative > div.narrative-card > p

- **Text**: "For storage operators and demand-response providers, this is the signal: the eco"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section-inner > div#sect3Narrative > div.narrative-card > p > strong

- **Text**: "strengthens"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div#sect3Narrative > div.narrative-card > span.stat-badge.stat-badge-amber

- **Text**: "Peak–offpeak spread: $13–$16/MWh from 80% to 97.5%"
- **Color**: #FEF7EB on #FEF7EB
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #777064 or adjust for 4.5:1 on bg #FEF7EB

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div#sect3Narrative > div.narrative-card > h3

- **Text**: "Off-peak prices go negative first — creating the duck curve in price space"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 800
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div#sect3Narrative > div.narrative-card > p

- **Text**: "At ≥99.99% clean, off-peak median is  while peak is still . Even in a fully deca"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section-inner > div#sect3Narrative > div.narrative-card > p > strong

- **Text**: "−$12/MWh"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section-inner > div#sect3Narrative > div.narrative-card > p > strong

- **Text**: "−$2/MWh"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div#sect3Narrative > div.narrative-card > span.stat-badge.stat-badge-blue

- **Text**: "At ≥99.99%: peak −$2, off-peak −$12 — $10 spread remains"
- **Color**: #ECF8FD on #ECF8FD
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #657176 or adjust for 4.5:1 on bg #ECF8FD

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div.grid-2col > div.narrative-card > span.section-tag

- **Text**: "Revenue Streams"
- **Color**: #F3F3FE on #F3F3FE
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 700
- **Fix**: Change text to #6C6C77 or adjust for 4.5:1 on bg #F3F3FE

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div.grid-2col > div.narrative-card > h3

- **Text**: "Capacity payments cushion the price decline — but only where markets exist"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 800
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div.grid-2col > div.narrative-card > p

- **Text**: "ISOs with capacity markets (PJM, NYISO, CAISO, NEISO, MISO) provide generators w"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section-inner > div.grid-2col > div.narrative-card > p > strong#nucCapRevPJM

- **Text**: "$12/MWh"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div.grid-2col > div.narrative-card > p

- **Text**: "In energy-only markets (ERCOT, SPP), generators must survive on energy revenue a"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div.grid-2col > div.narrative-card > span.section-tag

- **Text**: "Nuclear Viability"
- **Color**: #FEF0F0 on #FEF0F0
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 700
- **Fix**: Change text to #776969 or adjust for 4.5:1 on bg #FEF0F0

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div.grid-2col > div.narrative-card > h3

- **Text**: "Nuclear retirement risk shifts with total market revenue"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 800
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div.grid-2col > div.narrative-card > p

- **Text**: "When capacity payments are included, nuclear viability extends further into the "
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section-inner > div.grid-2col > div.narrative-card > p > strong

- **Text**: "$38–44/MWh"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section-inner > div.grid-2col > div.narrative-card > p > strong#nucViableCapPct

- **Text**: "78%"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div.grid-2col > div.narrative-card > p

- **Text**: "The 45U PTC ($15/MWh) combined with capacity payments creates a stronger floor, "
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div.grid-2col > div.narrative-card > span.section-tag

- **Text**: "Apples-to-Apples"
- **Color**: #ECF8FD on #ECF8FD
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 700
- **Fix**: Change text to #657176 or adjust for 4.5:1 on bg #ECF8FD

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div.grid-2col > div.narrative-card > h3

- **Text**: "Same metric, different models"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 800
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div.grid-2col > div.narrative-card > p

- **Text**: "Both curves show pure power market revenue — energy payments + capacity payments"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section-inner > div.grid-2col > div.narrative-card > p > strong

- **Text**: "pure power market revenue"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div.grid-2col > div.narrative-card > p

- **Text**: "Step 5B models wholesale prices under a policy-driven decarbonization pathway (S"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div.grid-2col > div.narrative-card > span.section-tag

- **Text**: "Convergence Zone"
- **Color**: #EDFAF2 on #EDFAF2
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 700
- **Fix**: Change text to #66736B or adjust for 4.5:1 on bg #EDFAF2

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div.grid-2col > div.narrative-card > h3

- **Text**: "Models converge below ~65% clean"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 800
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div.grid-2col > div.narrative-card > p

- **Text**: "At lower clean energy penetrations (50–65%), both models produce similar wholesa"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div.grid-2col > div.narrative-card > p

- **Text**: "Above 75% clean, the models diverge: Step 5B's policy pathway forces deployment "
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section-inner > div.grid-2col > div.narrative-card > p > strong

- **Text**: "capacity market premium"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div.scroll-narrative-col > div.narrative-card > span.section-tag

- **Text**: "Cannibalization"
- **Color**: #FEF7EB on #FEF7EB
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 700
- **Fix**: Change text to #777064 or adjust for 4.5:1 on bg #FEF7EB

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div.scroll-narrative-col > div.narrative-card > h3

- **Text**: "Every megawatt of solar and wind pushes wholesale prices lower — undermining its"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 800
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div.scroll-narrative-col > div.narrative-card > p

- **Text**: "As clean energy grows from 50% to 95%, average wholesale LMPs decline 44–56% acr"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div.scroll-narrative-col > div.narrative-card > p

- **Text**: "This is the cannibalization paradox: renewables are victims of their own success"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div.scroll-narrative-col > div.narrative-card > span.stat-badge.stat-badge-amber

- **Text**: "200 → 8,000+ zero-price hours as clean energy scales"
- **Color**: #FEF7EB on #FEF7EB
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #777064 or adjust for 4.5:1 on bg #FEF7EB

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div.scroll-narrative-col > div.narrative-card > h3

- **Text**: "Fuel prices explain everything — until they don't"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 800
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div.scroll-narrative-col > div.narrative-card > p

- **Text**: "At 50% clean, fuel costs explain  of wholesale price variance. At 80%, still 88%"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section-inner > div.scroll-narrative-col > div.narrative-card > p > strong

- **Text**: "100%"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div.scroll-narrative-col > div.narrative-card > span.stat-badge.stat-badge-blue

- **Text**: "Fuel variance: 100% at 50% → 14% at ≥99.99%"
- **Color**: #ECF8FD on #ECF8FD
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #657176 or adjust for 4.5:1 on bg #ECF8FD

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div.scroll-narrative-col > div.narrative-card > span.section-tag

- **Text**: "Storage Economics"
- **Color**: #F6F2FE on #F6F2FE
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 700
- **Fix**: Change text to #6F6B77 or adjust for 4.5:1 on bg #F6F2FE

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div.scroll-narrative-col > div.narrative-card > h3

- **Text**: "Storage deployment grows with CFE targets — and the arbitrage signal persists"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 800
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div.scroll-narrative-col > div.narrative-card > p

- **Text**: "Battery dispatch grows from near-zero at 50% clean to 7–10% of annual demand at "
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div.scroll-narrative-col > div.narrative-card > p

- **Text**: "The peak-to-offpeak spread — the arbitrage signal for storage operators — actual"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section-inner > div.scroll-narrative-col > div.narrative-card > p > em

- **Text**: "widens"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div.scroll-narrative-col > div.narrative-card > span.stat-badge.stat-badge-blue

- **Text**: "Peak–offpeak spread: widens from 80% to 97.5% clean"
- **Color**: #ECF8FD on #ECF8FD
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #657176 or adjust for 4.5:1 on bg #ECF8FD

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div.scroll-narrative-col > div.narrative-card > h3

- **Text**: "LDES becomes economic when daily cycling can't bridge the gap"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 800
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div.scroll-narrative-col > div.narrative-card > p

- **Text**: "At 95%+ clean, the remaining unmatched hours cluster into multi-day events — win"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div.scroll-narrative-col > div.narrative-card > span.stat-badge.stat-badge-green

- **Text**: "LDES: 100hr iron-air, 50% RTE · H₂: 1000hr, 35% RTE"
- **Color**: #EDFAF2 on #EDFAF2
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #66736B or adjust for 4.5:1 on bg #EDFAF2

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div.scroll-narrative-col > div.narrative-card > span.section-tag

- **Text**: "Gas Fleet Risk"
- **Color**: #F3F4F5 on #F3F4F5
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 700
- **Fix**: Change text to #6C6D6E or adjust for 4.5:1 on bg #F3F4F5

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div.scroll-narrative-col > div.narrative-card > h3

- **Text**: "Gas plants face a utilization cliff — but capacity is still needed for reliabili"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 800
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div.scroll-narrative-col > div.narrative-card > p

- **Text**: "As clean energy scales, gas capacity utilization plummets — but the fleet can't "
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div.scroll-narrative-col > div.narrative-card > p

- **Text**: "CAISO and ERCOT face the earliest stranding signals, where abundant wind and sol"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div.scroll-narrative-col > div.narrative-card > span.stat-badge.stat-badge-red

- **Text**: "Gas utilization: <5% of hours at 95%+ clean, but capacity still required"
- **Color**: #FEF4F4 on #FEF4F4
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #776D6D or adjust for 4.5:1 on bg #FEF4F4

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div.scroll-narrative-col > div.narrative-card > h3

- **Text**: "The stranding math varies dramatically by region"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 800
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div.scroll-narrative-col > div.narrative-card > p

- **Text**: "Existing gas fleets range from 16 GW (NEISO) to 128 GW (PJM). The gap between "i"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.scroll-section.section-light > div.scroll-section-inner > div.scroll-narrative-col > div.narrative-card > span.stat-badge.stat-badge-amber

- **Text**: "CAISO: 47 GW installed · ERCOT: 80 GW · PJM: 128 GW"
- **Color**: #FEF7EB on #FEF7EB
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #777064 or adjust for 4.5:1 on bg #FEF7EB

### [CRITICAL] body > div#nuclear-crossover > div.scroll-section-inner > div.narrative-card > span.section-tag

- **Text**: "Revenue Crossover"
- **Color**: #FEF0F0 on #FEF0F0
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 700
- **Fix**: Change text to #776969 or adjust for 4.5:1 on bg #FEF0F0

### [CRITICAL] body > div#nuclear-crossover > div.scroll-section-inner > div.narrative-card > h2.section-title

- **Text**: "The Nuclear Revenue Crossover"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 24px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] body > div#nuclear-crossover > div.scroll-section-inner > div.narrative-card > p.section-subtitle

- **Text**: "As clean energy targets rise, wholesale energy prices and capacity market revenu"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div#nuclear-crossover > div.scroll-section-inner > div > div.narrative-card > h3.section-title

- **Text**: "Nuclear Viability Heatmap"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 800
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div#nuclear-crossover > div.scroll-section-inner > div > div.narrative-card > p.section-subtitle

- **Text**: "Total market revenue (energy + capacity) vs. $41/MWh nuclear viability threshold"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > div.scroll-section.section-light > div.scroll-section-inner > div.narrative-card > h2#ctrRevenueTitle

- **Text**: "PJM — Revenue Decomposition"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 24px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] body > div.scroll-section.section-light > div.scroll-section-inner > div.narrative-card > p.section-subtitle

- **Text**: "Energy (LMP) and capacity market revenue at each clean energy threshold, with th"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > div.scroll-section.section-light > div.scroll-section-inner > div#ctrRevenueInsight > p

- **Text**: "Select an ISO above to view revenue decomposition."
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > div.scroll-section.section-light > div.scroll-section-inner > div.narrative-card > h3#ctrCompTitle

- **Text**: "PJM — Cost to Replace Nuclear"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 800
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] body > div.scroll-section.section-light > div.scroll-section-inner > div.narrative-card > p.section-subtitle

- **Text**: "Track 1 (baseline with nuclear), Track 2 (all new-build), and Track 3 (replace n"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > div.scroll-section.section-light > div.scroll-section-inner > div#ctrCompInsight > p

- **Text**: "Loading track data..."
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] section.section-dark > div.section-dark-inner > div#synthesisGrid > div.synthesis-card > h3

- **Text**: "The uncertainty paradox: widest where it matters most"
- **Color**: #F3F3F4 on #F3F3F4
- **Ratio**: 1:1 (need 3:1)
- **Font**: 16px, weight 800
- **Fix**: Change text to #8A8A8B or adjust for 3:1 on bg #F3F3F4

### [CRITICAL] section.section-dark > div.section-dark-inner > div#synthesisGrid > div.synthesis-card > p

- **Text**: "The P10–P90 spread in wholesale prices peaks at 75–80% clean energy — exactly th"
- **Color**: #F3F3F4 on #F3F3F4
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #6C6C6D or adjust for 4.5:1 on bg #F3F3F4

### [CRITICAL] div.section-dark-inner > div#synthesisGrid > div.synthesis-card > p > strong

- **Text**: "$31/MWh"
- **Color**: #F3F3F4 on #F3F3F4
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #6C6C6D or adjust for 4.5:1 on bg #F3F3F4

### [CRITICAL] section.section-dark > div.section-dark-inner > div#synthesisGrid > div.synthesis-card > p

- **Text**: "The implication: forecasting wholesale prices at 75–85% clean requires getting f"
- **Color**: #F3F3F4 on #F3F3F4
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #6C6C6D or adjust for 4.5:1 on bg #F3F3F4

### [CRITICAL] section.section-dark > div.section-dark-inner > div#synthesisGrid > div.synthesis-card > span.stat-badge.stat-badge-amber

- **Text**: "$31/MWh spread at 80% · $0.4/MWh spread at ≥99.99%"
- **Color**: #F3E6D1 on #F3E6D1
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #71644F or adjust for 4.5:1 on bg #F3E6D1

### [CRITICAL] section.section-dark > div.section-dark-inner > div#synthesisGrid > div.synthesis-card > h3

- **Text**: "Three eras of the wholesale market"
- **Color**: #F3F3F4 on #F3F3F4
- **Ratio**: 1:1 (need 3:1)
- **Font**: 16px, weight 800
- **Fix**: Change text to #8A8A8B or adjust for 3:1 on bg #F3F3F4

### [CRITICAL] section.section-dark > div.section-dark-inner > div#synthesisGrid > div.synthesis-card > p

- **Text**: "— Fossil sets the price. Fuel costs dominate. Wholesale markets function as they"
- **Color**: #F3F3F4 on #F3F3F4
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #6C6C6D or adjust for 4.5:1 on bg #F3F3F4

### [CRITICAL] div.section-dark-inner > div#synthesisGrid > div.synthesis-card > p > strong

- **Text**: "Era 1: 50–75% (2030–~2034)"
- **Color**: #F3F3F4 on #F3F3F4
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #6C6C6D or adjust for 4.5:1 on bg #F3F3F4

### [CRITICAL] section.section-dark > div.section-dark-inner > div#synthesisGrid > div.synthesis-card > p

- **Text**: "Era 2: 75–95% (~2034–2045) — The transition zone. Price uncertainty is at its pe"
- **Color**: #F3F3F4 on #F3F3F4
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #6C6C6D or adjust for 4.5:1 on bg #F3F3F4

### [CRITICAL] div.section-dark-inner > div#synthesisGrid > div.synthesis-card > p > strong

- **Text**: "Era 2: 75–95% (~2034–2045)"
- **Color**: #F3F3F4 on #F3F3F4
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #6C6C6D or adjust for 4.5:1 on bg #F3F3F4

### [CRITICAL] section.section-dark > div.section-dark-inner > div#synthesisGrid > div.synthesis-card > p

- **Text**: "— The structural break. Fossil no longer sets marginal price. All hours converge"
- **Color**: #F3F3F4 on #F3F3F4
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #6C6C6D or adjust for 4.5:1 on bg #F3F3F4

### [CRITICAL] div.section-dark-inner > div#synthesisGrid > div.synthesis-card > p > strong

- **Text**: "Era 3: 95–≥99.99% (2045–2050)"
- **Color**: #F3F3F4 on #F3F3F4
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #6C6C6D or adjust for 4.5:1 on bg #F3F3F4

### [CRITICAL] section.section-dark > div.section-dark-inner > div#synthesisGrid > div.synthesis-card > span.stat-badge.stat-badge-green

- **Text**: "2030–2034: business as usual · 2034–2045: transition · 2045+: new paradigm"
- **Color**: #D3ECDD on #D3ECDD
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #516A5B or adjust for 4.5:1 on bg #D3ECDD

### [CRITICAL] html > body > div.scroll-section.section-light > div.scroll-section-inner > div#ctrRevenueInsight

- **Text**: "Select an ISO above to view revenue decomposition."
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > div.scroll-section.section-light > div.scroll-section-inner > div#ctrCompInsight

- **Text**: "Loading track data..."
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [MINOR] div.hero-inner > div.hero-text > div#heroStats > div.hero-stat > div.hero-stat-label

- **Text**: "Median wholesale price decline, 2030 (50%) → 2045 (95%)"
- **Color**: #7B7F8A on #FFFFFF
- **Ratio**: 3.99:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #717580 or adjust for 4.5:1 on bg #FFFFFF

### [MINOR] div.hero-inner > div.hero-text > div#heroStats > div.hero-stat > div.hero-stat-label

- **Text**: "Peak uncertainty band (P10–P90) at 75–80% clean"
- **Color**: #7B7F8A on #FFFFFF
- **Ratio**: 3.99:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #717580 or adjust for 4.5:1 on bg #FFFFFF

### [MINOR] div.hero-inner > div.hero-text > div#heroStats > div.hero-stat > div.hero-stat-label

- **Text**: "Year LMP falls below nuclear viability threshold (~70% clean)"
- **Color**: #7B7F8A on #FFFFFF
- **Ratio**: 3.99:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #717580 or adjust for 4.5:1 on bg #FFFFFF

## dashboard/market_convergence.html

**5 failures**

### [CRITICAL] body > div.content-section > div.verdict-grid > div.card > h3

- **Text**: "Step 5B — Policy-Driven Snapshot"
- **Color**: #0EA5E9 on #FFFFFF
- **Ratio**: 2.77:1 (need 4.5:1)
- **Font**: 17px, weight 600
- **Fix**: Change text to #0078BC or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > div.content-section > div.verdict-grid > div.card > h3

- **Text**: "Step 10 — Market-Driven Trajectory"
- **Color**: #22C55E on #FFFFFF
- **Ratio**: 2.28:1 (need 4.5:1)
- **Font**: 17px, weight 600
- **Fix**: Change text to #008922 or adjust for 4.5:1 on bg #FFFFFF

### [MAJOR] html > body > div.content-section > div#isoSelector > button.iso-btn.active

- **Text**: "PJM"
- **Color**: #0284C7 on #DCEFFA
- **Ratio**: 3.46:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #006BAE or adjust for 4.5:1 on bg #DCEFFA

### [MINOR] body > div.content-section > div.verdict-grid > div.card.converge-row > h3

- **Text**: "Where They Converge"
- **Color**: #DC2626 on #EBF6F3
- **Ratio**: 4.37:1 (need 4.5:1)
- **Font**: 17px, weight 600
- **Fix**: Change text to #D72121 or adjust for 4.5:1 on bg #EBF6F3

### [MINOR] body > div.content-section > div.verdict-grid > div.card.diverge-row > h3

- **Text**: "Where They Diverge"
- **Color**: #DC2626 on #F7EEF1
- **Ratio**: 4.25:1 (need 4.5:1)
- **Font**: 17px, weight 600
- **Fix**: Change text to #D21C1C or adjust for 4.5:1 on bg #F7EEF1

## dashboard/model_validation.html

**29 failures**

### [CRITICAL] body > div#executive-summary > div.grid-stats > div.stat-card > div.stat-value

- **Text**: "21"
- **Color**: #22C55E on #FFFFFF
- **Ratio**: 2.28:1 (need 3:1)
- **Font**: 26px, weight 700
- **Fix**: Change text to #09AC45 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] body > div#executive-summary > div.grid-stats > div.stat-card > div.stat-value

- **Text**: "5,832"
- **Color**: #F59E0B on #FFFFFF
- **Ratio**: 2.15:1 (need 3:1)
- **Font**: 26px, weight 700
- **Fix**: Change text to #D78000 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] body > div#executive-summary > div.grid-stats > div.stat-card > div.stat-value

- **Text**: "8,760"
- **Color**: #06B6D4 on #FFFFFF
- **Ratio**: 2.43:1 (need 3:1)
- **Font**: 26px, weight 700
- **Fix**: Change text to #00A2C0 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] body > div#hourly-match-score > div.card > div.formula-box > div.formula-label

- **Text**: "Total Clean Supply at Hour h"
- **Color**: #0EA5E9 on #F8FAFC
- **Ratio**: 2.65:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #0078BC or adjust for 4.5:1 on bg #F8FAFC

### [CRITICAL] body > div#hourly-match-score > div.card > div.formula-box > div.formula-label

- **Text**: "Hourly Match"
- **Color**: #0EA5E9 on #F8FAFC
- **Ratio**: 2.65:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #0078BC or adjust for 4.5:1 on bg #F8FAFC

### [CRITICAL] body > div#hourly-match-score > div.card > div.formula-box > div.formula-label

- **Text**: "Hourly Match Score (%)"
- **Color**: #0EA5E9 on #F8FAFC
- **Ratio**: 2.65:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #0078BC or adjust for 4.5:1 on bg #F8FAFC

### [CRITICAL] body > div#battery-dispatch > div.card > div.formula-box > div.formula-label

- **Text**: "Energy Accounting"
- **Color**: #0EA5E9 on #F8FAFC
- **Ratio**: 2.65:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #0078BC or adjust for 4.5:1 on bg #F8FAFC

### [CRITICAL] body > div#cost-function > div.card > div.formula-box > div.formula-label

- **Text**: "Total Incremental Cost ($/MWh of demand)"
- **Color**: #0EA5E9 on #F8FAFC
- **Ratio**: 2.65:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #0078BC or adjust for 4.5:1 on bg #F8FAFC

### [CRITICAL] body > div#cost-function > div.card > div.formula-box > div.formula-label

- **Text**: "Per-Resource Cost"
- **Color**: #0EA5E9 on #F8FAFC
- **Ratio**: 2.65:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #0078BC or adjust for 4.5:1 on bg #F8FAFC

### [CRITICAL] body > div#cost-function > div.card > div.formula-box > div.formula-label

- **Text**: "Storage Cost (annualized)"
- **Color**: #0EA5E9 on #F8FAFC
- **Ratio**: 2.65:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #0078BC or adjust for 4.5:1 on bg #F8FAFC

### [CRITICAL] body > div#cost-function > div.card > div.formula-box > div.formula-label

- **Text**: "Gas Backup Cost"
- **Color**: #0EA5E9 on #F8FAFC
- **Ratio**: 2.65:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #0078BC or adjust for 4.5:1 on bg #F8FAFC

### [CRITICAL] body > div#co2-model > div.card > div.formula-box > div.formula-label

- **Text**: "Displacement Calculation"
- **Color**: #0EA5E9 on #F8FAFC
- **Ratio**: 2.65:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #0078BC or adjust for 4.5:1 on bg #F8FAFC

### [CRITICAL] body > div#co2-model > div.card > div.formula-box > div.formula-label

- **Text**: "Weighted Displaced Emission Rate"
- **Color**: #0EA5E9 on #F8FAFC
- **Ratio**: 2.65:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #0078BC or adjust for 4.5:1 on bg #F8FAFC

### [CRITICAL] body > div#reproducibility > div.card > div.formula-box > div.formula-label

- **Text**: "Quick Start"
- **Color**: #22C55E on #F8FAFC
- **Ratio**: 2.18:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #00841D or adjust for 4.5:1 on bg #F8FAFC

### [MAJOR] div.card > table.data-table > tbody > tr > td

- **Text**: "✓ PASS"
- **Color**: #16A34A on #FFFFFF
- **Ratio**: 3.3:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #00852C or adjust for 4.5:1 on bg #FFFFFF

### [MAJOR] div.card > table.data-table > tbody > tr > td

- **Text**: "✓ PASS"
- **Color**: #16A34A on #FFFFFF
- **Ratio**: 3.3:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #00852C or adjust for 4.5:1 on bg #FFFFFF

### [MAJOR] div.card > table.data-table > tbody > tr > td

- **Text**: "✓ PASS"
- **Color**: #16A34A on #FFFFFF
- **Ratio**: 3.3:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #00852C or adjust for 4.5:1 on bg #FFFFFF

### [MAJOR] div.card > table.data-table > tbody > tr > td

- **Text**: "✓ PASS"
- **Color**: #16A34A on #FFFFFF
- **Ratio**: 3.3:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #00852C or adjust for 4.5:1 on bg #FFFFFF

### [MAJOR] div.card > table.data-table > tbody > tr > td

- **Text**: "✓ PASS"
- **Color**: #16A34A on #FFFFFF
- **Ratio**: 3.3:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #00852C or adjust for 4.5:1 on bg #FFFFFF

### [MAJOR] div.card > table.data-table > tbody > tr > td

- **Text**: "✓ PASS"
- **Color**: #16A34A on #FFFFFF
- **Ratio**: 3.3:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #00852C or adjust for 4.5:1 on bg #FFFFFF

### [MAJOR] div.card > table.data-table > tbody > tr > td

- **Text**: "✓ PASS"
- **Color**: #16A34A on #FFFFFF
- **Ratio**: 3.3:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #00852C or adjust for 4.5:1 on bg #FFFFFF

### [MAJOR] div.card > table.data-table > tbody > tr > td

- **Text**: "−9.5%"
- **Color**: #16A34A on #FFFFFF
- **Ratio**: 3.3:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #00852C or adjust for 4.5:1 on bg #FFFFFF

### [MAJOR] div.card > table.data-table > tbody > tr > td

- **Text**: "+9.2%"
- **Color**: #16A34A on #FFFFFF
- **Ratio**: 3.3:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #00852C or adjust for 4.5:1 on bg #FFFFFF

### [MAJOR] div.card > table.data-table > tbody > tr > td

- **Text**: "+3.7%"
- **Color**: #16A34A on #FFFFFF
- **Ratio**: 3.3:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #00852C or adjust for 4.5:1 on bg #FFFFFF

### [MAJOR] div.card > table.data-table > tbody > tr > td

- **Text**: "−9.0%"
- **Color**: #16A34A on #FFFFFF
- **Ratio**: 3.3:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #00852C or adjust for 4.5:1 on bg #FFFFFF

### [MAJOR] div.card > table.data-table > tbody > tr > td

- **Text**: "+7.6%"
- **Color**: #D97706 on #FFFFFF
- **Ratio**: 3.19:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #BB5900 or adjust for 4.5:1 on bg #FFFFFF

### [MAJOR] div.card > table.data-table > tbody > tr > td

- **Text**: "+3.9%"
- **Color**: #16A34A on #FFFFFF
- **Ratio**: 3.3:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #00852C or adjust for 4.5:1 on bg #FFFFFF

### [MAJOR] div.card > table.data-table > tbody > tr > td

- **Text**: "+8.5%"
- **Color**: #16A34A on #FFFFFF
- **Ratio**: 3.3:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #00852C or adjust for 4.5:1 on bg #FFFFFF

### [MAJOR] body > div#validation-results > div.card > p > span

- **Text**: "GOOD"
- **Color**: #16A34A on #FFFFFF
- **Ratio**: 3.3:1 (need 4.5:1)
- **Font**: 14px, weight 600
- **Fix**: Change text to #00852C or adjust for 4.5:1 on bg #FFFFFF

## dashboard/policy_context.html

**22 failures**

### [CRITICAL] div.content-wrap.section-light > section.section-dark > div.section-dark-inner > div.insight-box > strong

- **Text**: "What our optimizer reveals about this debate:"
- **Color**: #1A2744 on #2C3547
- **Ratio**: 1.21:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #74819E or adjust for 3:1 on bg #2C3547

### [CRITICAL] div.content-wrap.section-light > section.section-dark > div.section-dark-inner > div.insight-box > strong

- **Text**: "For any given grid region, with any given set of technology cost assumptions, wh"
- **Color**: #1A2744 on #2C3547
- **Ratio**: 1.21:1 (need 3:1)
- **Font**: 17px, weight 700
- **Fix**: Change text to #74819E or adjust for 3:1 on bg #2C3547

### [MINOR] body > div.content-wrap.section-light > div.sources-section > ul.source-list > li

- **Text**: "IPCC AR6 WGIII, Chapter 6: Energy Systems (2022)"
- **Color**: #6B7280 on #F3F4F8
- **Ratio**: 4.4:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F8

### [MINOR] body > div.content-wrap.section-light > div.sources-section > ul.source-list > li

- **Text**: "IEA Net Zero by 2050 Roadmap (2021, updated 2023)"
- **Color**: #6B7280 on #F3F4F8
- **Ratio**: 4.4:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F8

### [MINOR] body > div.content-wrap.section-light > div.sources-section > ul.source-list > li

- **Text**: "IEA World Energy Outlook 2025"
- **Color**: #6B7280 on #F3F4F8
- **Ratio**: 4.4:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F8

### [MINOR] body > div.content-wrap.section-light > div.sources-section > ul.source-list > li

- **Text**: "IEA Electricity 2025: Emissions"
- **Color**: #6B7280 on #F3F4F8
- **Ratio**: 4.4:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F8

### [MINOR] body > div.content-wrap.section-light > div.sources-section > ul.source-list > li

- **Text**: "Sepulveda, Jenkins, de Sisternes, Lester. "The Role of Firm Low-Carbon Electrici"
- **Color**: #6B7280 on #F3F4F8
- **Ratio**: 4.4:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F8

### [MINOR] div.content-wrap.section-light > div.sources-section > ul.source-list > li > em

- **Text**: "Joule"
- **Color**: #6B7280 on #F3F4F8
- **Ratio**: 4.4:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F8

### [MINOR] body > div.content-wrap.section-light > div.sources-section > ul.source-list > li

- **Text**: "Xu, Manocha, Patankar, Jenkins. "System-Level Impacts of 24/7 Carbon-Free Electr"
- **Color**: #6B7280 on #F3F4F8
- **Ratio**: 4.4:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F8

### [MINOR] body > div.content-wrap.section-light > div.sources-section > ul.source-list > li

- **Text**: "Riepin, Jenkins, Swezey, Brown. "24/7 Carbon-Free Electricity Matching Accelerat"
- **Color**: #6B7280 on #F3F4F8
- **Ratio**: 4.4:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F8

### [MINOR] div.content-wrap.section-light > div.sources-section > ul.source-list > li > em

- **Text**: "Joule"
- **Color**: #6B7280 on #F3F4F8
- **Ratio**: 4.4:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F8

### [MINOR] body > div.content-wrap.section-light > div.sources-section > ul.source-list > li

- **Text**: "NREL. "Examining Supply-Side Options to Achieve 100% Clean Electricity by 2035.""
- **Color**: #6B7280 on #F3F4F8
- **Ratio**: 4.4:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F8

### [MINOR] body > div.content-wrap.section-light > div.sources-section > ul.source-list > li

- **Text**: "NREL. "LA100: Los Angeles 100% Renewable Energy Study." March 2021."
- **Color**: #6B7280 on #F3F4F8
- **Ratio**: 4.4:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F8

### [MINOR] body > div.content-wrap.section-light > div.sources-section > ul.source-list > li

- **Text**: "GHG Protocol Scope 2 Public Consultation (Oct 2025 – Jan 2026)"
- **Color**: #6B7280 on #F3F4F8
- **Ratio**: 4.4:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F8

### [MINOR] body > div.content-wrap.section-light > div.sources-section > ul.source-list > li

- **Text**: "SBTi Corporate Net-Zero Standard V2.0, Second Consultation Draft (Nov 2025)"
- **Color**: #6B7280 on #F3F4F8
- **Ratio**: 4.4:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F8

### [MINOR] body > div.content-wrap.section-light > div.sources-section > ul.source-list > li

- **Text**: "Climate Group 24/7 Carbon-Free Coalition, Technical Criteria V1.0 (June 2025)"
- **Color**: #6B7280 on #F3F4F8
- **Ratio**: 4.4:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F8

### [MINOR] body > div.content-wrap.section-light > div.sources-section > ul.source-list > li

- **Text**: "RE100 FAQs and Technical Criteria (Jan 2025)"
- **Color**: #6B7280 on #F3F4F8
- **Ratio**: 4.4:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F8

### [MINOR] body > div.content-wrap.section-light > div.sources-section > ul.source-list > li

- **Text**: "CDP Scope 2 Accounting Guidance & 2025 Reporting Guide"
- **Color**: #6B7280 on #F3F4F8
- **Ratio**: 4.4:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F8

### [MINOR] body > div.content-wrap.section-light > div.sources-section > ul.source-list > li

- **Text**: "EnergyTag Granular Certificate Scheme Standard V2"
- **Color**: #6B7280 on #F3F4F8
- **Ratio**: 4.4:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F8

### [MINOR] body > div.content-wrap.section-light > div.sources-section > ul.source-list > li

- **Text**: "UN 24/7 Carbon-Free Energy Compact (Sept 2021)"
- **Color**: #6B7280 on #F3F4F8
- **Ratio**: 4.4:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F8

### [MINOR] body > div.content-wrap.section-light > div.sources-section > ul.source-list > li

- **Text**: "LBNL. "U.S. State Renewables Portfolio & Clean Energy Standards: 2024 Status Upd"
- **Color**: #6B7280 on #F3F4F8
- **Ratio**: 4.4:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F8

### [MINOR] body > div.content-wrap.section-light > div.sources-section > ul.source-list > li

- **Text**: "EPA. "24/7 Hourly Matching of Electricity." Green Power Markets."
- **Color**: #6B7280 on #F3F4F8
- **Ratio**: 4.4:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F8

## dashboard/procurement_research.html

**5 failures**

### [CRITICAL] div#strategies > div.strategy-grid > div.strategy-card > p > strong

- **Text**: "grid-average emission rate"
- **Color**: #FFFFFF on #FAFBFD
- **Ratio**: 1.03:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FAFBFD

### [CRITICAL] div#strategies > div.strategy-grid > div.strategy-card > p > strong

- **Text**: "fossil-only average emission rate"
- **Color**: #FFFFFF on #FAFBFD
- **Ratio**: 1.03:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FAFBFD

### [CRITICAL] div#strategies > div.strategy-grid > div.strategy-card > p > strong

- **Text**: "short-run marginal emission rate"
- **Color**: #FFFFFF on #FAFBFD
- **Ratio**: 1.03:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FAFBFD

### [CRITICAL] div#strategies > div.strategy-grid > div.strategy-card > p > strong

- **Text**: "Track 2 NB"
- **Color**: #FFFFFF on #FAFBFD
- **Ratio**: 1.03:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FAFBFD

### [CRITICAL] div#strategies > div.strategy-grid > div.strategy-card > p > strong

- **Text**: ""status quo""
- **Color**: #FFFFFF on #FAFBFD
- **Ratio**: 1.03:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FAFBFD

## dashboard/ref-ipp-vs-utility.html

**4 failures**

### [CRITICAL] html > body > article.article-wrapper.section-light > div.article-meta > span.article-tag

- **Text**: "Market Structure"
- **Color**: #0EA5E9 on #DBF2FC
- **Ratio**: 2.38:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #006EB2 or adjust for 4.5:1 on bg #DBF2FC

### [CRITICAL] html > body > article.article-wrapper.section-light > div.article-meta > span.article-tag

- **Text**: "Generation Ownership"
- **Color**: #0EA5E9 on #DBF2FC
- **Ratio**: 2.38:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #006EB2 or adjust for 4.5:1 on bg #DBF2FC

### [CRITICAL] html > body > article.article-wrapper.section-light > div.illustration-panel > div.illustration-title

- **Text**: "IPP vs. Regulated Utility Generation by ISO (TWh/yr)"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.illustration-panel > div.illustration-caption

- **Text**: "Approximate generation shares. “IPP/Merchant” includes competitive generators se"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

## dashboard/ref-lmp-ccs-nuclear.html

**24 failures**

### [CRITICAL] html > body > article.article-wrapper.section-light > div.article-meta > span.article-tag

- **Text**: "Market Design"
- **Color**: #0EA5E9 on #DBF2FC
- **Ratio**: 2.38:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #006EB2 or adjust for 4.5:1 on bg #DBF2FC

### [CRITICAL] html > body > article.article-wrapper.section-light > div.illustration-panel > p.illustration-caption

- **Text**: "Same demand level, different clearing price. Nuclear flexibility shifts the marg"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.content-section > h3

- **Text**: "The Merit Order Sets the Price"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.content-section > p

- **Text**: "Wholesale electricity prices in organized U.S. markets (LMPs) are determined by "
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > article.article-wrapper.section-light > div.content-section > p > strong

- **Text**: "marginal generator"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > article.article-wrapper.section-light > div.content-section > p > em

- **Text**: "which generator is marginal"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.content-section > h3

- **Text**: "CCS on the Margin: Prices Go Up"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.content-section > p

- **Text**: "Carbon capture equipment imposes a  on a combined-cycle gas plant. The same fuel"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > article.article-wrapper.section-light > div.content-section > p > strong

- **Text**: "25–30% energy penalty"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > article.article-wrapper.section-light > div.content-section > p > strong

- **Text**: "$10–25/MWh higher"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > article.article-wrapper.section-light > div.content-section > p > em

- **Text**: "lower"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.content-section > h3

- **Text**: "Dispatchable Nuclear: Prices Get Crushed"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.content-section > p

- **Text**: "Advanced nuclear designs (SMRs, load-following reactors) would bid into the mark"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > article.article-wrapper.section-light > div.content-section > p > strong

- **Text**: "$5–10/MWh marginal cost"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > article.article-wrapper.section-light > div.content-section > p > strong

- **Text**: "evening peaks, winter mornings, and overnight periods"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.content-section > h3

- **Text**: "Both Together: A Bimodal Price World"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.content-section > p

- **Text**: "If both resources dispatch flexibly, the supply curve reshapes into a steeper st"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > article.article-wrapper.section-light > div.content-section > p > strong

- **Text**: "bimodal"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.callout-box > h4

- **Text**: "Implication for CFE Procurement"
- **Color**: #ECF8FD on #ECF8FD
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #838F94 or adjust for 3:1 on bg #ECF8FD

### [CRITICAL] html > body > article.article-wrapper.section-light > div.callout-box > p

- **Text**: "Dispatchable nuclear is unambiguously beneficial for corporate CFE buyers: lower"
- **Color**: #ECF8FD on #ECF8FD
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #657176 or adjust for 4.5:1 on bg #ECF8FD

### [CRITICAL] body > article.article-wrapper.section-light > div.callout-box > p > em

- **Text**: "hardest-to-match hours"
- **Color**: #ECF8FD on #ECF8FD
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #657176 or adjust for 4.5:1 on bg #ECF8FD

### [CRITICAL] html > body > article.article-wrapper.section-light > div.illustration-panel > div.illustration-title

- **Text**: "Merit Order & LMP Formation: Current vs. With Dispatchable Clean Resources"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] article.article-wrapper.section-light > div.illustration-panel > div.illustration-grid > div.illustration-chart > div.illustration-chart-label

- **Text**: "Today's Merit Order"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] article.article-wrapper.section-light > div.illustration-panel > div.illustration-grid > div.illustration-chart > div.illustration-chart-label

- **Text**: "With Dispatchable Nuclear & CCS"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

## dashboard/ref-market-design-low-carbon.html

**90 failures**

### [CRITICAL] html > body > article.article-wrapper.section-light > div.article-meta > span.article-tag

- **Text**: "Market Design"
- **Color**: #0EA5E9 on #DBF2FC
- **Ratio**: 2.38:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #006EB2 or adjust for 4.5:1 on bg #DBF2FC

### [CRITICAL] html > body > article.article-wrapper.section-light > div.article-meta > span.article-tag

- **Text**: "Capital Recovery"
- **Color**: #0EA5E9 on #DBF2FC
- **Ratio**: 2.38:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #006EB2 or adjust for 4.5:1 on bg #DBF2FC

### [CRITICAL] article.article-wrapper.section-light > div.illustration-panel > div.paradox-grid > div.paradox-box > h4

- **Text**: "Today’s Market"
- **Color**: #F1FAFE on #F1FAFE
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #6A7377 or adjust for 4.5:1 on bg #F1FAFE

### [CRITICAL] article.article-wrapper.section-light > div.illustration-panel > div.paradox-grid > div.paradox-box > p

- **Text**: "Gas plants burn fuel at $35–50/MWh. This fuel cost sets the clearing price. Ever"
- **Color**: #F1FAFE on #F1FAFE
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #6A7377 or adjust for 4.5:1 on bg #F1FAFE

### [CRITICAL] article.article-wrapper.section-light > div.illustration-panel > div.paradox-grid > div.paradox-box > h4

- **Text**: "Tomorrow’s Market"
- **Color**: #F1FAFE on #F1FAFE
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #6A7377 or adjust for 4.5:1 on bg #F1FAFE

### [CRITICAL] article.article-wrapper.section-light > div.illustration-panel > div.paradox-grid > div.paradox-box > p

- **Text**: "Nuclear runs at $5–10/MWh marginal cost. Solar and wind at ~$0. No fuel to burn."
- **Color**: #F1FAFE on #F1FAFE
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #6A7377 or adjust for 4.5:1 on bg #F1FAFE

### [CRITICAL] body > article.article-wrapper.section-light > div.illustration-panel > div.paradox-result > p

- **Text**: "The cheaper clean energy gets, the harder it is to finance. Every new zero-margi"
- **Color**: #FEF0F0 on #FEF0F0
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #776969 or adjust for 4.5:1 on bg #FEF0F0

### [CRITICAL] article.article-wrapper.section-light > div.illustration-panel > div.paradox-result > p > strong

- **Text**: "The paradox:"
- **Color**: #FEF0F0 on #FEF0F0
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #958787 or adjust for 3:1 on bg #FEF0F0

### [CRITICAL] html > body > article.article-wrapper.section-light > div.illustration-panel > p.illustration-caption

- **Text**: "Current wholesale markets price at short-run marginal cost (SRMC). When SRMC is "
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.content-section > h3

- **Text**: "The Market Wasn’t Built for This"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.content-section > p

- **Text**: "Wholesale electricity markets — PJM, CAISO, ERCOT, ISO-NE, MISO — all price elec"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > article.article-wrapper.section-light > div.content-section > p > strong

- **Text**: "marginal generator"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.content-section > p

- **Text**: "But the clean energy transition inverts this logic. Wind, solar, and nuclear hav"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > article.article-wrapper.section-light > div.content-section > p > strong

- **Text**: "near-zero operating costs"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > article.article-wrapper.section-light > div.content-section > p > strong

- **Text**: "bimodal price distribution"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.content-section > h3

- **Text**: "The Nuclear Revenue Crisis Is Already Here"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.content-section > p

- **Text**: "This isn’t a hypothetical. Existing nuclear plants are already drowning in marke"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > article.article-wrapper.section-light > div.content-section > p > strong

- **Text**: "a zero-carbon, always-on, 90%+ capacity factor resource couldn’t survive on mark"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.content-section > p

- **Text**: "The market actively punishes the exact resource the grid needs most. Nuclear pro"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.content-section > h3

- **Text**: "What Should the “Clean Premium” Be?"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.content-section > p

- **Text**: "If the market doesn’t pay for clean attributes, what’s the right price? The answ"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.content-section > p

- **Text**: "A carbon price of $50–80/ton CO₂ would raise the cost of unabated gas by $20–35/"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > article.article-wrapper.section-light > div.content-section > p > strong

- **Text**: "Carbon-free generation."
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.content-section > p

- **Text**: "An intermittent MWh is not equivalent to a firm MWh. LCOE comparisons systematic"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > article.article-wrapper.section-light > div.content-section > p > strong

- **Text**: "Firmness and dispatchability."
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > article.article-wrapper.section-light > div.content-section > p > strong

- **Text**: "Levelized Value of Energy (LVOE)"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.content-section > p

- **Text**: "The hours that matter most — winter peaks, low-wind/low-solar periods, system st"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > article.article-wrapper.section-light > div.content-section > p > strong

- **Text**: "Reliability during scarcity."
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.content-section > h3

- **Text**: "Can You Put Capital Costs Into Marginal Prices?"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.content-section > p

- **Text**: "This is the radical question. Today’s markets clear at  — just fuel plus variabl"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > article.article-wrapper.section-light > div.content-section > p > strong

- **Text**: "short-run marginal cost (SRMC)"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > article.article-wrapper.section-light > div.content-section > p > strong

- **Text**: "long-run marginal cost (LRMC)"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.content-section > p

- **Text**: "The problem is that LRMC pricing fundamentally changes dispatch efficiency. SRMC"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.content-section > p

- **Text**: "The emerging consensus is a : keep SRMC dispatch for operational efficiency, but"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > article.article-wrapper.section-light > div.content-section > p > strong

- **Text**: "hybrid approach"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > article.article-wrapper.section-light > div.content-section > p > strong

- **Text**: "two-way Contracts for Difference (CfDs)"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.callout-box.warning > h4

- **Text**: "The Cannibalization Trap"
- **Color**: #FEF7EB on #FEF7EB
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #958E82 or adjust for 3:1 on bg #FEF7EB

### [CRITICAL] html > body > article.article-wrapper.section-light > div.callout-box.warning > p

- **Text**: "Every new wind or solar farm depresses wholesale prices during the hours it prod"
- **Color**: #FEF7EB on #FEF7EB
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #777064 or adjust for 4.5:1 on bg #FEF7EB

### [CRITICAL] body > article.article-wrapper.section-light > div.callout-box.warning > p > strong

- **Text**: "merit order effect"
- **Color**: #FEF7EB on #FEF7EB
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #958E82 or adjust for 3:1 on bg #FEF7EB

### [CRITICAL] html > body > article.article-wrapper.section-light > div.callout-box.warning > p

- **Text**: "Germany, Australia, UK, Ireland, Chile, and Spain have all seen surging hours of"
- **Color**: #FEF7EB on #FEF7EB
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #777064 or adjust for 4.5:1 on bg #FEF7EB

### [CRITICAL] html > body > article.article-wrapper.section-light > div.content-section > h3

- **Text**: "Four Schools of Thought on How to Fix It"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.content-section > p

- **Text**: "The literature on market redesign is deep and growing. Here are the four dominan"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > article.article-wrapper.section-light > div.school-grid > div.school-card > h4

- **Text**: "1. Fix the Existing Design"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > article.article-wrapper.section-light > div.school-grid > div.school-card > p

- **Text**: "Zero-marginal-cost generation doesn’t change the fundamentals. Implement the exi"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] article.article-wrapper.section-light > div.school-grid > div.school-card > div.school-verdict > strong

- **Text**: "Trade-off:"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > article.article-wrapper.section-light > div.school-grid > div.school-card > h4

- **Text**: "2. Hybrid Long-Term Contracting"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > article.article-wrapper.section-light > div.school-grid > div.school-card > p

- **Text**: "Keep short-term dispatch markets, but add organized long-term mechanisms: CfDs, "
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] article.article-wrapper.section-light > div.school-grid > div.school-card > div.school-verdict > strong

- **Text**: "Trade-off:"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > article.article-wrapper.section-light > div.school-grid > div.school-card > h4

- **Text**: "3. Structural Split Markets"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > article.article-wrapper.section-light > div.school-grid > div.school-card > p

- **Text**: "The cost structure of the industry has fundamentally changed — from variable-cos"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] article.article-wrapper.section-light > div.school-grid > div.school-card > div.school-verdict > strong

- **Text**: "Trade-off:"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > article.article-wrapper.section-light > div.school-grid > div.school-card > h4

- **Text**: "4. Forward Energy Markets"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > article.article-wrapper.section-light > div.school-grid > div.school-card > p

- **Text**: "Replace capacity markets with granular forward energy contracts — monthly and ho"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] article.article-wrapper.section-light > div.school-grid > div.school-card > div.school-verdict > strong

- **Text**: "Trade-off:"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.content-section > h3

- **Text**: "Money Has to Be Made: The Capital Deployment Problem"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.content-section > p

- **Text**: "Every proposed market design ultimately has to answer one question:  The clean e"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > article.article-wrapper.section-light > div.content-section > p > strong

- **Text**: "will investors deploy capital?"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.content-section > p

- **Text**: "The UK’s CfD program is the most mature evidence point. Since 2013, it has contr"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > article.article-wrapper.section-light > div.content-section > p > strong

- **Text**: "39+ GW"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.content-section > p

- **Text**: "For nuclear, the revenue problem is more acute. New nuclear projects take 10–15 "
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > article.article-wrapper.section-light > div.content-section > p > strong

- **Text**: "how do you guarantee a return on a $10 billion, 60-year asset in a market design"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.content-section > h3

- **Text**: "What’s Actually Happening on the Ground"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.content-section > p

- **Text**: "No coherent national market reform. Instead, a patchwork: state ZEC programs for"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > article.article-wrapper.section-light > div.content-section > p > strong

- **Text**: "United States."
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.content-section > p

- **Text**: "The most deliberate reform. May 2024 regulation retained marginal pricing for sp"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > article.article-wrapper.section-light > div.content-section > p > strong

- **Text**: "European Union."
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.content-section > p

- **Text**: "REMA (Review of Electricity Market Arrangements) concluded in July 2025: zonal p"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > article.article-wrapper.section-light > div.content-section > p > strong

- **Text**: "United Kingdom."
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.content-section > p

- **Text**: "Capacity Investment Scheme procuring 32 GW via CfD-like tenders. AEMO expects 90"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > article.article-wrapper.section-light > div.content-section > p > strong

- **Text**: "Australia."
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.callout-box > h4

- **Text**: "Real-World Evidence: Spain’s Partial Decoupling"
- **Color**: #ECF8FD on #ECF8FD
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #838F94 or adjust for 3:1 on bg #ECF8FD

### [CRITICAL] html > body > article.article-wrapper.section-light > div.callout-box > p

- **Text**: "Spain offers early evidence that the transition is already reshaping price forma"
- **Color**: #ECF8FD on #ECF8FD
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #657176 or adjust for 4.5:1 on bg #ECF8FD

### [CRITICAL] body > article.article-wrapper.section-light > div.callout-box > p > strong

- **Text**: "40% higher"
- **Color**: #ECF8FD on #ECF8FD
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #838F94 or adjust for 3:1 on bg #ECF8FD

### [CRITICAL] html > body > article.article-wrapper.section-light > div.content-section > h3

- **Text**: "A Contrarian View: Maybe Markets Survive Just Fine"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.content-section > p

- **Text**: "Not everyone agrees the sky is falling. Antweiler & Müsgens (2025, ) modeled all"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > article.article-wrapper.section-light > div.content-section > p > em

- **Text**: "Energy Economics"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > article.article-wrapper.section-light > div.content-section > p > strong

- **Text**: "energy-only markets remain viable"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.content-section > p

- **Text**: "De Vries & Verzijlbergh (2024, ) reached a similar conclusion: energy-limited re"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > article.article-wrapper.section-light > div.content-section > p > em

- **Text**: "Renewable and Sustainable Energy Reviews"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.content-section > p

- **Text**: "The catch: both analyses assume  and active demand response — neither of which e"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > article.article-wrapper.section-light > div.content-section > p > strong

- **Text**: "massive storage deployment"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > article.article-wrapper.section-light > div.callout-box > h4

- **Text**: "What This Means for 24/7 CFE Procurement"
- **Color**: #ECF8FD on #ECF8FD
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #838F94 or adjust for 3:1 on bg #ECF8FD

### [CRITICAL] html > body > article.article-wrapper.section-light > div.callout-box > p

- **Text**: "Our optimizer models the cost of hourly carbon-free energy matching from 50% to "
- **Color**: #ECF8FD on #ECF8FD
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #657176 or adjust for 4.5:1 on bg #ECF8FD

### [CRITICAL] body > article.article-wrapper.section-light > div.callout-box > p > strong

- **Text**: "the last 5–10% of hourly matching requires firm clean resources that can’t recov"
- **Color**: #ECF8FD on #ECF8FD
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #838F94 or adjust for 3:1 on bg #ECF8FD

### [CRITICAL] html > body > article.article-wrapper.section-light > div.callout-box > p

- **Text**: "If market reforms succeed in properly valuing firm clean attributes — through Cf"
- **Color**: #ECF8FD on #ECF8FD
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #657176 or adjust for 4.5:1 on bg #ECF8FD

### [CRITICAL] html > body > article.article-wrapper.section-light > div.content-section > h3

- **Text**: "Key References"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] div.ref-table-wrap > table.ref-table > thead > tr > th

- **Text**: "Source"
- **Color**: #0EA5E9 on #E7F6FD
- **Ratio**: 2.51:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #0073B7 or adjust for 4.5:1 on bg #E7F6FD

### [CRITICAL] div.ref-table-wrap > table.ref-table > thead > tr > th

- **Text**: "Year"
- **Color**: #0EA5E9 on #E7F6FD
- **Ratio**: 2.51:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #0073B7 or adjust for 4.5:1 on bg #E7F6FD

### [CRITICAL] div.ref-table-wrap > table.ref-table > thead > tr > th

- **Text**: "Contribution"
- **Color**: #0EA5E9 on #E7F6FD
- **Ratio**: 2.51:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #0073B7 or adjust for 4.5:1 on bg #E7F6FD

### [CRITICAL] html > body > article.article-wrapper.section-light > div.illustration-panel > div.illustration-title

- **Text**: "The Zero-Marginal-Cost Paradox"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

## dashboard/reference.html

**60 failures**

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > span.ref-card-tag

- **Text**: "Interactive Tool"
- **Color**: #DBF2FC on #DBF2FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #59707A or adjust for 4.5:1 on bg #DBF2FC

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > h3

- **Text**: "The Grid (Optimizer)"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > p

- **Text**: "Interactive cost optimizer with all sensitivity toggles. Explore resource mixes,"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > span.ref-card-tag

- **Text**: "Analysis"
- **Color**: #DBF2FC on #DBF2FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #59707A or adjust for 4.5:1 on bg #DBF2FC

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > h3

- **Text**: "Five Failure Modes"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > p

- **Text**: "How procurement strategies can fail at scale — from duck curve amplification to "
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > span.ref-card-tag

- **Text**: "Grid Physics"
- **Color**: #DBF2FC on #DBF2FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #59707A or adjust for 4.5:1 on bg #DBF2FC

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > h3

- **Text**: "Building Blocks"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > p

- **Text**: "Resource shapes, supply curves, and the hourly generation profiles that drive th"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > span.ref-card-tag

- **Text**: "Academic"
- **Color**: #DBF2FC on #DBF2FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #59707A or adjust for 4.5:1 on bg #DBF2FC

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > h3

- **Text**: "Literature Review"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > p

- **Text**: "Academic foundations: hourly accounting, cost benchmarks, firm power, storage ec"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > span.ref-card-tag

- **Text**: "Technical"
- **Color**: #DBF2FC on #DBF2FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #59707A or adjust for 4.5:1 on bg #DBF2FC

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > h3

- **Text**: "Methodology"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > p

- **Text**: "Detailed technical specification of the 10-step optimization pipeline, dispatch "
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > span.ref-card-tag

- **Text**: "Policy"
- **Color**: #DBF2FC on #DBF2FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #59707A or adjust for 4.5:1 on bg #DBF2FC

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > h3

- **Text**: "Policy Context"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > p

- **Text**: "How IRA tax credits, RPS mandates, Scope 2 revisions, SBTi pathways, and capacit"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > span.ref-card-tag

- **Text**: "Resource Economics"
- **Color**: #DBF2FC on #DBF2FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #59707A or adjust for 4.5:1 on bg #DBF2FC

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > h3

- **Text**: "Technology Tipping Points"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > p

- **Text**: "At what cost thresholds does the optimal clean energy mix undergo phase transiti"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > span.ref-card-tag

- **Text**: "Storage Economics"
- **Color**: #DBF2FC on #DBF2FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #59707A or adjust for 4.5:1 on bg #DBF2FC

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > h3

- **Text**: "The System Value of Energy Storage"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > p

- **Text**: "How much does each storage technology reduce total system cost? Not LCOE or LCOS"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > span.ref-card-tag

- **Text**: "Corporate Strategy"
- **Color**: #DBF2FC on #DBF2FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #59707A or adjust for 4.5:1 on bg #DBF2FC

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > h3

- **Text**: "Where Should You Build Your Data Center?"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > p

- **Text**: "Physics-based comparison of hourly clean energy costs across 7 US ISOs. For hype"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > span.ref-card-tag

- **Text**: "Carbon Markets"
- **Color**: #DBF2FC on #DBF2FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #59707A or adjust for 4.5:1 on bg #DBF2FC

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > h3

- **Text**: "What Should a Carbon Credit Actually Cost?"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > p

- **Text**: "Physics-grounded marginal abatement costs as fair-value carbon credit prices. Mo"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > span.ref-card-tag

- **Text**: "Procurement"
- **Color**: #DBF2FC on #DBF2FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #59707A or adjust for 4.5:1 on bg #DBF2FC

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > h3

- **Text**: "Scenario Comparison"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > p

- **Text**: "Consequential vs. hourly matching strategies compared across all 7 ISOs. The Lea"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > span.ref-card-tag

- **Text**: "Procurement"
- **Color**: #DBF2FC on #DBF2FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #59707A or adjust for 4.5:1 on bg #DBF2FC

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > h3

- **Text**: "Hourly vs. Annual Matching Comparison"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > p

- **Text**: "Side-by-side analysis of hourly CFE matching versus annual REC-based approaches."
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > span.ref-card-tag

- **Text**: "Grid Physics"
- **Color**: #DBF2FC on #DBF2FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #59707A or adjust for 4.5:1 on bg #DBF2FC

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > h3

- **Text**: "Grid Resilience Stress Testing"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > p

- **Text**: "How do optimized clean energy portfolios perform under worst-case weather? Synth"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > span.ref-card-tag

- **Text**: "Resource Economics"
- **Color**: #DBF2FC on #DBF2FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #59707A or adjust for 4.5:1 on bg #DBF2FC

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > h3

- **Text**: "The Hidden Value of Curtailed Energy"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > p

- **Text**: "High-renewable grids throw away massive amounts of clean energy. What if that su"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > span.ref-card-tag

- **Text**: "Market Design"
- **Color**: #DBF2FC on #DBF2FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #59707A or adjust for 4.5:1 on bg #DBF2FC

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > h3

- **Text**: "The Case for Cross-Regional Clean Energy Trading"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > p

- **Text**: "CAISO has midday solar surplus while PJM needs evening power. Cross-regional hou"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > span.ref-card-tag

- **Text**: "Market Design"
- **Color**: #DBF2FC on #DBF2FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #59707A or adjust for 4.5:1 on bg #DBF2FC

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > h3

- **Text**: "How Would Dispatchable CCS or Nuclear Change Wholesale Prices?"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > p

- **Text**: "If CCS-equipped gas plants or advanced nuclear could flex their output, how woul"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > span.ref-card-tag

- **Text**: "Market Design"
- **Color**: #DBF2FC on #DBF2FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #59707A or adjust for 4.5:1 on bg #DBF2FC

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > h3

- **Text**: "If Nobody Burns Fuel, How Do You Build a Market?"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > p

- **Text**: "Current wholesale markets price at marginal fuel cost. When the grid runs on nuc"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > span.ref-card-tag

- **Text**: "Resource Economics"
- **Color**: #DBF2FC on #DBF2FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #59707A or adjust for 4.5:1 on bg #DBF2FC

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > h3

- **Text**: "Energy Economics Metric Dictionary"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > p

- **Text**: "LCOE, $/kW capacity cost, LCOS, capacity factor, LMP, MAC, and more — what each "
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > span.ref-card-tag

- **Text**: "Market Structure"
- **Color**: #DBF2FC on #DBF2FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #59707A or adjust for 4.5:1 on bg #DBF2FC

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > h3

- **Text**: "Who Generates America’s Electricity? IPP vs. Utility Breakdown by ISO"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > p

- **Text**: "State-by-state, company-by-company breakdown of independent power producer vs. v"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > span.ref-card-tag

- **Text**: "Nuclear Risk"
- **Color**: #DBF2FC on #DBF2FC
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #59707A or adjust for 4.5:1 on bg #DBF2FC

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > h3

- **Text**: "Nuclear Retirement Risk Under the Clean Energy Transition"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 18px, weight 700
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > section.ref-grid.section-light > a.ref-card > p

- **Text**: "Three models bracket merchant nuclear risk: a probability fan from current marke"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

## dashboard/scenario_comparison.html

**36 failures**

### [CRITICAL] section.section-dark.panel-navy > div.section-dark-inner > div.scenario-cards > div.scenario-card > h3

- **Text**: "Cross-Regional Consequential Netting"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] section.section-dark.panel-navy > div.section-dark-inner > div.scenario-cards > div.scenario-card > p

- **Text**: "Strategy 1B deploys resources sequentially across ISOs, always buying the cheape"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] section.section-dark.panel-navy > div.section-dark-inner > div.scenario-cards > div.scenario-card > h3

- **Text**: "Grid-Wide Hourly CFE for Net Zero by 2050"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] section.section-dark.panel-navy > div.section-dark-inner > div.scenario-cards > div.scenario-card > p

- **Text**: "Strategy 2C deploys hourly matching across the entire grid to achieve net zero b"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] div.section-dark-inner > div.scenario-cards > div.scenario-card > p > a

- **Text**: "See Strategy 2C deep dive →"
- **Color**: #3EB7ED on #FFFFFF
- **Ratio**: 2.28:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #027BB1 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.content-section.story-section > div.card > span.story-badge

- **Text**: "75–90% • The Inflection Point"
- **Color**: #D97706 on #FEF5E7
- **Ratio**: 2.95:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #B65400 or adjust for 4.5:1 on bg #FEF5E7

### [CRITICAL] html > body > section.content-section.section-light > h2.section-title

- **Text**: "What Gets Built: Strategy 1B vs Strategy 2C"
- **Color**: #1A2744 on #0F172A
- **Ratio**: 1.21:1 (need 3:1)
- **Font**: 24px, weight 700
- **Fix**: Change text to #5B6885 or adjust for 3:1 on bg #0F172A

### [CRITICAL] html > body > section.content-section.section-light > p.section-subtitle

- **Text**: "Resource mix in TWh at each threshold.
Dashed line = CFE target — solid bars = u"
- **Color**: #1E293B on #0F172A
- **Ratio**: 1.22:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #788395 or adjust for 4.5:1 on bg #0F172A

### [CRITICAL] body > section.content-section.section-light > div.chart-pair > div.chart-box > h3#mixTitleA

- **Text**: "PJM — Resource Mix (TWh)"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > section.content-section.section-light > div.chart-pair > div.chart-box > h3#mixTitleB

- **Text**: "PJM — Resource Mix (TWh)"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.content-section.section-light > h2.section-title

- **Text**: "System Cost & Marginal Abatement Cost"
- **Color**: #1A2744 on #0F172A
- **Ratio**: 1.21:1 (need 3:1)
- **Font**: 24px, weight 700
- **Fix**: Change text to #5B6885 or adjust for 3:1 on bg #0F172A

### [CRITICAL] html > body > section.content-section.section-light > p.section-subtitle

- **Text**: "Effective $/MWh system cost (left) and stepwise $/tCO₂ (right) at each SBTi mile"
- **Color**: #1E293B on #0F172A
- **Ratio**: 1.22:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #788395 or adjust for 4.5:1 on bg #0F172A

### [CRITICAL] body > section.content-section.section-light > div.chart-pair > div.chart-box > h3#costChartTitle

- **Text**: "PJM — Effective System Cost ($/MWh)"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > section.content-section.section-light > div.chart-pair > div.chart-box > h3#macTimelineTitle

- **Text**: "PJM — Marginal $/tCO₂ by SBTi Year"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.content-section.section-light > h2.section-title

- **Text**: "Clean Firm & Gas Capacity"
- **Color**: #1A2744 on #0F172A
- **Ratio**: 1.21:1 (need 3:1)
- **Font**: 24px, weight 700
- **Fix**: Change text to #5B6885 or adjust for 3:1 on bg #0F172A

### [CRITICAL] html > body > section.content-section.section-light > p.section-subtitle

- **Text**: "Clean firm deployment (left) and gas backup capacity (right) at each milestone."
- **Color**: #1E293B on #0F172A
- **Ratio**: 1.22:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #788395 or adjust for 4.5:1 on bg #0F172A

### [CRITICAL] body > section.content-section.section-light > div.chart-pair > div.chart-box > h3#firmChartTitle

- **Text**: "PJM — Clean Firm + CCS (TWh)"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > section.content-section.section-light > div.chart-pair > div.chart-box > h3#gasChartTitle

- **Text**: "PJM — Gas Backup Capacity"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > section.content-section.section-light > h2.section-title

- **Text**: "Sequential Deployment Queue"
- **Color**: #1A2744 on #0F172A
- **Ratio**: 1.21:1 (need 3:1)
- **Font**: 24px, weight 700
- **Fix**: Change text to #5B6885 or adjust for 3:1 on bg #0F172A

### [CRITICAL] html > body > section.content-section.section-light > p.section-subtitle

- **Text**: "Cross-regional deployment order from Step 5D MAC queue, sorted by cheapest margi"
- **Color**: #1E293B on #0F172A
- **Ratio**: 1.22:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #788395 or adjust for 4.5:1 on bg #0F172A

### [CRITICAL] div > table > thead > tr > th

- **Text**: "Toggle"
- **Color**: #080E19 on #0F1A2E
- **Ratio**: 1.11:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #808691 or adjust for 4.5:1 on bg #0F1A2E

### [CRITICAL] div > table > thead > tr > th

- **Text**: "Strategy 1B"
- **Color**: #4F46E5 on #0F1A2E
- **Ratio**: 2.77:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #776EFF or adjust for 4.5:1 on bg #0F1A2E

### [MAJOR] body > section.hero > div.hero-chart-wrap > p.chart-note > span

- **Text**: "Strategy 2C"
- **Color**: #0891B2 on #ECEDEE
- **Ratio**: 3.13:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #007394 or adjust for 4.5:1 on bg #ECEDEE

### [MAJOR] div > table > tbody > tr > td

- **Text**: "Low (NOAK)"
- **Color**: #15803D on #0F1A2E
- **Ratio**: 3.47:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #299451 or adjust for 4.5:1 on bg #0F1A2E

### [MAJOR] div > table > tbody > tr > td

- **Text**: "Low (NOAK)"
- **Color**: #15803D on #0F1A2E
- **Ratio**: 3.47:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #299451 or adjust for 4.5:1 on bg #0F1A2E

### [MAJOR] div > table > tbody > tr > td

- **Text**: "Low (NOAK)"
- **Color**: #15803D on #0F1A2E
- **Ratio**: 3.47:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #299451 or adjust for 4.5:1 on bg #0F1A2E

### [MAJOR] div > table > tbody > tr > td

- **Text**: "Low (NOAK)"
- **Color**: #15803D on #0F1A2E
- **Ratio**: 3.47:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #299451 or adjust for 4.5:1 on bg #0F1A2E

### [MINOR] body > section.hero > div.hero-chart-wrap > p.chart-note > span

- **Text**: "Red"
- **Color**: #DC2626 on #ECEDEE
- **Ratio**: 4.11:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #D21C1C or adjust for 4.5:1 on bg #ECEDEE

### [MINOR] section.section-dark.panel-navy > div.section-dark-inner > div.scenario-cards > div.scenario-card > span.scenario-tag.tag-b

- **Text**: "Scenario B — Strategy 2C (Hourly Matching)"
- **Color**: #64748B on #E6F4F7
- **Ratio**: 4.23:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #5F6F86 or adjust for 4.5:1 on bg #E6F4F7

### [MINOR] html > body > section.content-section.story-section > div.card > span.story-badge.story-badge-red

- **Text**: "95–≥99.99% • The Last Mile"
- **Color**: #DC2626 on #FDE9E9
- **Ratio**: 4.13:1 (need 4.5:1)
- **Font**: 13px, weight 700
- **Fix**: Change text to #D21C1C or adjust for 4.5:1 on bg #FDE9E9

### [MINOR] body > section.content-section.section-light > div.chart-pair > div.chart-box > span.scenario-tag.tag-b

- **Text**: "Strategy 2C"
- **Color**: #64748B on #E6F4F7
- **Ratio**: 4.23:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #5F6F86 or adjust for 4.5:1 on bg #E6F4F7

### [MINOR] div > table > tbody > tr > td

- **Text**: "High (FOAK)"
- **Color**: #DC2626 on #0F1A2E
- **Ratio**: 3.6:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #F53F3F or adjust for 4.5:1 on bg #0F1A2E

### [MINOR] div > table > tbody > tr > td

- **Text**: "High (FOAK)"
- **Color**: #DC2626 on #0F1A2E
- **Ratio**: 3.6:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #F53F3F or adjust for 4.5:1 on bg #0F1A2E

### [MINOR] div > table > tbody > tr > td

- **Text**: "High (FOAK)"
- **Color**: #DC2626 on #0F1A2E
- **Ratio**: 3.6:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #F53F3F or adjust for 4.5:1 on bg #0F1A2E

### [MINOR] div > table > tbody > tr > td

- **Text**: "High (FOAK)"
- **Color**: #DC2626 on #0F1A2E
- **Ratio**: 3.6:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #F53F3F or adjust for 4.5:1 on bg #0F1A2E

### [MINOR] section.section-dark > div.section-dark-inner > div > p > a

- **Text**: "Optimizer Methodology"
- **Color**: #0284C7 on #0F1A2E
- **Ratio**: 4.25:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #0789CC or adjust for 4.5:1 on bg #0F1A2E

## dashboard/strategy_deep_dive.html

**4 failures**

### [CRITICAL] body > section.content-section.story-section > div.chart-grid-2 > div.card > p

- **Text**: "How the resource mix evolves as CO₂ ambition increases. Select a strategy to vie"
- **Color**: #94A3B8 on #FFFFFF
- **Ratio**: 2.56:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #67768B or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > section.content-section.story-section > div.chart-grid-2 > div.card > p

- **Text**: "Strategies that overbuild renewables early show rising curtailment at higher tar"
- **Color**: #94A3B8 on #FFFFFF
- **Ratio**: 2.56:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #67768B or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > section.content-section.story-section > div.chart-grid-2 > div.card > p

- **Text**: "Blended $/MWh as participation scales. Lower = faster learning curve realization"
- **Color**: #94A3B8 on #FFFFFF
- **Ratio**: 2.56:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #67768B or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > section.content-section.story-section > div.chart-grid-2 > div.card > p

- **Text**: "Participation level where blended costs reach net-negative (learning curves outp"
- **Color**: #94A3B8 on #FFFFFF
- **Ratio**: 2.56:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #67768B or adjust for 4.5:1 on bg #FFFFFF

## dashboard/ipp-transition-report.html

**29 failures**

### [MAJOR] table.company-score-table > tbody > tr.hero-row > td > span.score-pill.score-low

- **Text**: "14"
- **Color**: #DC2626 on #E9D9DF
- **Ratio**: 3.56:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #C30D0D or adjust for 4.5:1 on bg #E9D9DF

### [MINOR] table.company-score-table > tbody > tr.hero-row > td > span.score-pill.score-high

- **Text**: "86"
- **Color**: #4A7A2E on #D7E6DD
- **Ratio**: 3.96:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #407024 or adjust for 4.5:1 on bg #D7E6DD

### [MINOR] table.company-score-table > tbody > tr.hero-row > td > span.score-pill.score-high

- **Text**: "100"
- **Color**: #4A7A2E on #D7E6DD
- **Ratio**: 3.96:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #407024 or adjust for 4.5:1 on bg #D7E6DD

### [MINOR] table.company-score-table > tbody > tr.hero-row > td > span.score-pill.score-high

- **Text**: "100"
- **Color**: #4A7A2E on #D7E6DD
- **Ratio**: 3.96:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #407024 or adjust for 4.5:1 on bg #D7E6DD

### [MINOR] table.company-score-table > tbody > tr.hero-row > td > span.score-pill.score-high

- **Text**: "82"
- **Color**: #4A7A2E on #D7E6DD
- **Ratio**: 3.96:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #407024 or adjust for 4.5:1 on bg #D7E6DD

### [MINOR] table.company-score-table > tbody > tr > td > span.score-pill.score-low

- **Text**: "29"
- **Color**: #DC2626 on #F4E1E2
- **Ratio**: 3.83:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #C81212 or adjust for 4.5:1 on bg #F4E1E2

### [MINOR] table.company-score-table > tbody > tr > td > span.score-pill.score-high

- **Text**: "100"
- **Color**: #4A7A2E on #E2EDE0
- **Ratio**: 4.24:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #457529 or adjust for 4.5:1 on bg #E2EDE0

### [MINOR] table.company-score-table > tbody > tr > td > span.score-pill.score-high

- **Text**: "73"
- **Color**: #4A7A2E on #E2EDE0
- **Ratio**: 4.24:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #457529 or adjust for 4.5:1 on bg #E2EDE0

### [MINOR] table.company-score-table > tbody > tr > td > span.score-pill.score-mid

- **Text**: "57"
- **Color**: #B45309 on #F8ECDA
- **Ratio**: 4.29:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #AF4E04 or adjust for 4.5:1 on bg #F8ECDA

### [MINOR] table.company-score-table > tbody > tr > td > span.score-pill.score-high

- **Text**: "100"
- **Color**: #4A7A2E on #E2EDE0
- **Ratio**: 4.24:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #457529 or adjust for 4.5:1 on bg #E2EDE0

### [MINOR] table.company-score-table > tbody > tr > td > span.score-pill.score-mid

- **Text**: "65"
- **Color**: #B45309 on #F8ECDA
- **Ratio**: 4.29:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #AF4E04 or adjust for 4.5:1 on bg #F8ECDA

### [MINOR] table.company-score-table > tbody > tr > td > span.score-pill.score-mid

- **Text**: "67"
- **Color**: #B45309 on #F8ECDA
- **Ratio**: 4.29:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #AF4E04 or adjust for 4.5:1 on bg #F8ECDA

### [MINOR] table.company-score-table > tbody > tr > td > span.score-pill.score-low

- **Text**: "15"
- **Color**: #DC2626 on #F4E1E2
- **Ratio**: 3.83:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #C81212 or adjust for 4.5:1 on bg #F4E1E2

### [MINOR] table.company-score-table > tbody > tr > td > span.score-pill.score-high

- **Text**: "100"
- **Color**: #4A7A2E on #E2EDE0
- **Ratio**: 4.24:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #457529 or adjust for 4.5:1 on bg #E2EDE0

### [MINOR] table.company-score-table > tbody > tr > td > span.score-pill.score-high

- **Text**: "100"
- **Color**: #4A7A2E on #E2EDE0
- **Ratio**: 4.24:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #457529 or adjust for 4.5:1 on bg #E2EDE0

### [MINOR] table.company-score-table > tbody > tr > td > span.score-pill.score-high

- **Text**: "100"
- **Color**: #4A7A2E on #E2EDE0
- **Ratio**: 4.24:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #457529 or adjust for 4.5:1 on bg #E2EDE0

### [MINOR] table.company-score-table > tbody > tr > td > span.score-pill.score-high

- **Text**: "100"
- **Color**: #4A7A2E on #E2EDE0
- **Ratio**: 4.24:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #457529 or adjust for 4.5:1 on bg #E2EDE0

### [MINOR] table.company-score-table > tbody > tr > td > span.score-pill.score-mid

- **Text**: "42"
- **Color**: #B45309 on #F8ECDA
- **Ratio**: 4.29:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #AF4E04 or adjust for 4.5:1 on bg #F8ECDA

### [MINOR] table.company-score-table > tbody > tr > td > span.score-pill.score-low

- **Text**: "25"
- **Color**: #DC2626 on #F4E1E2
- **Ratio**: 3.83:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #C81212 or adjust for 4.5:1 on bg #F4E1E2

### [MINOR] table.company-score-table > tbody > tr > td > span.score-pill.score-low

- **Text**: "38"
- **Color**: #DC2626 on #F4E1E2
- **Ratio**: 3.83:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #C81212 or adjust for 4.5:1 on bg #F4E1E2

### [MINOR] table.company-score-table > tbody > tr > td > span.score-pill.score-mid

- **Text**: "44"
- **Color**: #B45309 on #F8ECDA
- **Ratio**: 4.29:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #AF4E04 or adjust for 4.5:1 on bg #F8ECDA

### [MINOR] table.company-score-table > tbody > tr > td > span.score-pill.score-high

- **Text**: "100"
- **Color**: #4A7A2E on #E2EDE0
- **Ratio**: 4.24:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #457529 or adjust for 4.5:1 on bg #E2EDE0

### [MINOR] table.company-score-table > tbody > tr > td > span.score-pill.score-low

- **Text**: "25"
- **Color**: #DC2626 on #F4E1E2
- **Ratio**: 3.83:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #C81212 or adjust for 4.5:1 on bg #F4E1E2

### [MINOR] table.company-score-table > tbody > tr > td > span.score-pill.score-high

- **Text**: "100"
- **Color**: #4A7A2E on #E2EDE0
- **Ratio**: 4.24:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #457529 or adjust for 4.5:1 on bg #E2EDE0

### [MINOR] table.company-score-table > tbody > tr > td > span.score-pill.score-high

- **Text**: "100"
- **Color**: #4A7A2E on #E2EDE0
- **Ratio**: 4.24:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #457529 or adjust for 4.5:1 on bg #E2EDE0

### [MINOR] table.company-score-table > tbody > tr > td > span.score-pill.score-mid

- **Text**: "50"
- **Color**: #B45309 on #F8ECDA
- **Ratio**: 4.29:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #AF4E04 or adjust for 4.5:1 on bg #F8ECDA

### [MINOR] table.company-score-table > tbody > tr > td > span.score-pill.score-high

- **Text**: "90"
- **Color**: #4A7A2E on #E2EDE0
- **Ratio**: 4.24:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #457529 or adjust for 4.5:1 on bg #E2EDE0

### [MINOR] table.company-score-table > tbody > tr > td > span.score-pill.score-mid

- **Text**: "58"
- **Color**: #B45309 on #F8ECDA
- **Ratio**: 4.29:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #AF4E04 or adjust for 4.5:1 on bg #F8ECDA

### [MINOR] table.company-score-table > tbody > tr > td > span.score-pill.score-high

- **Text**: "81"
- **Color**: #4A7A2E on #E2EDE0
- **Ratio**: 4.24:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #457529 or adjust for 4.5:1 on bg #E2EDE0

## dashboard/procurement_deployment.html

**2 failures**

### [MAJOR] div.agg-chart-card > div.grid-2col > div > div#crossoverIsoSelector > button.iso-btn.active

- **Text**: "CAISO"
- **Color**: #0284C7 on #E1F3FC
- **Ratio**: 3.59:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #0070B3 or adjust for 4.5:1 on bg #E1F3FC

### [MINOR] div.controls-bar > div.controls-inner > div.control-group > div.toggle-btn-group > button.toggle-btn

- **Text**: "Roll-Off"
- **Color**: #6B7280 on #F3F4F6
- **Ratio**: 4.39:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F6

## dashboard/ref-nuclear-retirement.html

**3 failures**

### [MAJOR] body > div.article-wrapper > div#heroPanel > div#heroInsight > div.insight-label

- **Text**: "Cross-Model Synthesis"
- **Color**: #06B6D4 on #48518C
- **Ratio**: 3.07:1 (need 4.5:1)
- **Font**: 11px, weight 600
- **Fix**: Change text to #2EDEFC or adjust for 4.5:1 on bg #48518C

### [MINOR] body > div.article-wrapper > div#accModelA > div.accordion-header > span.accordion-badge.model-a

- **Text**: "Step 10 Reference"
- **Color**: #0284C7 on #E2F4FC
- **Ratio**: 3.63:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #0070B3 or adjust for 4.5:1 on bg #E2F4FC

### [MINOR] body > div.article-wrapper > div#accModelB > div.accordion-header > span.accordion-badge.model-b

- **Text**: "Power Sector & Economy-Wide NZ"
- **Color**: #6366F1 on #ECEDFD
- **Ratio**: 3.84:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #5457E2 or adjust for 4.5:1 on bg #ECEDFD

## dashboard/cfe_strategy_assessment.html

**16 failures**

### [MINOR] section#scorecard > div > div > div#scorecardThresholdToggle > button

- **Text**: "90%"
- **Color**: #6B7280 on #F3F4F6
- **Ratio**: 4.39:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F6

### [MINOR] section#scorecard > div > div > div#scorecardThresholdToggle > button

- **Text**: "92.5%"
- **Color**: #6B7280 on #F3F4F6
- **Ratio**: 4.39:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F6

### [MINOR] section#scorecard > div > div > div#scorecardThresholdToggle > button

- **Text**: "97.5%"
- **Color**: #6B7280 on #F3F4F6
- **Ratio**: 4.39:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F6

### [MINOR] section#scorecard > div > div > div#scorecardThresholdToggle > button

- **Text**: "99%+"
- **Color**: #6B7280 on #F3F4F6
- **Ratio**: 4.39:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F6

### [MINOR] section#scorecard > div > div > div#scorecardParticipationToggle > button

- **Text**: "5%"
- **Color**: #6B7280 on #F3F4F6
- **Ratio**: 4.39:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F6

### [MINOR] section#scorecard > div > div > div#scorecardParticipationToggle > button

- **Text**: "10%"
- **Color**: #6B7280 on #F3F4F6
- **Ratio**: 4.39:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F6

### [MINOR] section#scorecard > div > div > div#scorecardParticipationToggle > button

- **Text**: "15%"
- **Color**: #6B7280 on #F3F4F6
- **Ratio**: 4.39:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F6

### [MINOR] section#scorecard > div > div > div#scorecardParticipationToggle > button

- **Text**: "20%"
- **Color**: #6B7280 on #F3F4F6
- **Ratio**: 4.39:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F6

### [MINOR] section#scorecard > div > div > div#scorecardParticipationToggle > button

- **Text**: "50%"
- **Color**: #6B7280 on #F3F4F6
- **Ratio**: 4.39:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F6

### [MINOR] section#scorecard > div > div > div#scorecardParticipationToggle > button

- **Text**: "75%"
- **Color**: #6B7280 on #F3F4F6
- **Ratio**: 4.39:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F6

### [MINOR] main > section#crossover > div > div#crossoverParticipationToggle > button

- **Text**: "5%"
- **Color**: #6B7280 on #F3F4F6
- **Ratio**: 4.39:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F6

### [MINOR] main > section#crossover > div > div#crossoverParticipationToggle > button

- **Text**: "10%"
- **Color**: #6B7280 on #F3F4F6
- **Ratio**: 4.39:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F6

### [MINOR] main > section#crossover > div > div#crossoverParticipationToggle > button

- **Text**: "15%"
- **Color**: #6B7280 on #F3F4F6
- **Ratio**: 4.39:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F6

### [MINOR] main > section#crossover > div > div#crossoverParticipationToggle > button

- **Text**: "20%"
- **Color**: #6B7280 on #F3F4F6
- **Ratio**: 4.39:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F6

### [MINOR] main > section#crossover > div > div#crossoverParticipationToggle > button

- **Text**: "50%"
- **Color**: #6B7280 on #F3F4F6
- **Ratio**: 4.39:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F6

### [MINOR] main > section#crossover > div > div#crossoverParticipationToggle > button

- **Text**: "75%"
- **Color**: #6B7280 on #F3F4F6
- **Ratio**: 4.39:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F6

## dashboard/dashboard.html

**22 failures**

### [MINOR] div#sensitivityPanel > div.sensitivity-toggles-grid > div.sensitivity-toggle-item > div.toggle-btn-group > button

- **Text**: "Low"
- **Color**: #6B7280 on #F3F4F6
- **Ratio**: 4.39:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F6

### [MINOR] div#sensitivityPanel > div.sensitivity-toggles-grid > div.sensitivity-toggle-item > div.toggle-btn-group > button

- **Text**: "High"
- **Color**: #6B7280 on #F3F4F6
- **Ratio**: 4.39:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F6

### [MINOR] div#sensitivityPanel > div.sensitivity-toggles-grid > div.sensitivity-toggle-item > div.toggle-btn-group > button

- **Text**: "Low"
- **Color**: #6B7280 on #F3F4F6
- **Ratio**: 4.39:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F6

### [MINOR] div#sensitivityPanel > div.sensitivity-toggles-grid > div.sensitivity-toggle-item > div.toggle-btn-group > button

- **Text**: "High"
- **Color**: #6B7280 on #F3F4F6
- **Ratio**: 4.39:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F6

### [MINOR] div#sensitivityPanel > div.sensitivity-toggles-grid > div.sensitivity-toggle-item > div.toggle-btn-group > button

- **Text**: "Low"
- **Color**: #6B7280 on #F3F4F6
- **Ratio**: 4.39:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F6

### [MINOR] div#sensitivityPanel > div.sensitivity-toggles-grid > div.sensitivity-toggle-item > div.toggle-btn-group > button

- **Text**: "High"
- **Color**: #6B7280 on #F3F4F6
- **Ratio**: 4.39:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F6

### [MINOR] div#sensitivityPanel > div.sensitivity-toggles-grid > div.sensitivity-toggle-item > div.toggle-btn-group > button

- **Text**: "Low"
- **Color**: #6B7280 on #F3F4F6
- **Ratio**: 4.39:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F6

### [MINOR] div#sensitivityPanel > div.sensitivity-toggles-grid > div.sensitivity-toggle-item > div.toggle-btn-group > button

- **Text**: "High"
- **Color**: #6B7280 on #F3F4F6
- **Ratio**: 4.39:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F6

### [MINOR] div#sensitivityPanel > div.sensitivity-toggles-grid > div.sensitivity-toggle-item > div.toggle-btn-group > button

- **Text**: "Low"
- **Color**: #6B7280 on #F3F4F6
- **Ratio**: 4.39:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F6

### [MINOR] div#sensitivityPanel > div.sensitivity-toggles-grid > div.sensitivity-toggle-item > div.toggle-btn-group > button

- **Text**: "High"
- **Color**: #6B7280 on #F3F4F6
- **Ratio**: 4.39:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F6

### [MINOR] div#sensitivityPanel > div.sensitivity-toggles-grid > div.sensitivity-toggle-item > div.toggle-btn-group > button

- **Text**: "None"
- **Color**: #6B7280 on #F3F4F6
- **Ratio**: 4.39:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F6

### [MINOR] div#sensitivityPanel > div.sensitivity-toggles-grid > div.sensitivity-toggle-item > div.toggle-btn-group > button

- **Text**: "Low"
- **Color**: #6B7280 on #F3F4F6
- **Ratio**: 4.39:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F6

### [MINOR] div#sensitivityPanel > div.sensitivity-toggles-grid > div.sensitivity-toggle-item > div.toggle-btn-group > button

- **Text**: "High"
- **Color**: #6B7280 on #F3F4F6
- **Ratio**: 4.39:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F6

### [MINOR] div#sensitivityPanel > div.sensitivity-toggles-grid > div.sensitivity-toggle-item > div.toggle-btn-group > button

- **Text**: "Low"
- **Color**: #6B7280 on #F3F4F6
- **Ratio**: 4.39:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F6

### [MINOR] div#sensitivityPanel > div.sensitivity-toggles-grid > div.sensitivity-toggle-item > div.toggle-btn-group > button

- **Text**: "High"
- **Color**: #6B7280 on #F3F4F6
- **Ratio**: 4.39:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F6

### [MINOR] div#sensitivityPanel > div.sensitivity-toggles-grid > div.sensitivity-toggle-item > div#toggle45Q > button

- **Text**: "Without 45Q"
- **Color**: #6B7280 on #F3F4F6
- **Ratio**: 4.39:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F6

### [MINOR] div#sensitivityPanel > div.sensitivity-toggles-grid > div#geoToggleItem > div.toggle-btn-group > button

- **Text**: "Low"
- **Color**: #6B7280 on #F3F4F6
- **Ratio**: 4.39:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F6

### [MINOR] div#sensitivityPanel > div.sensitivity-toggles-grid > div#geoToggleItem > div.toggle-btn-group > button

- **Text**: "High"
- **Color**: #6B7280 on #F3F4F6
- **Ratio**: 4.39:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F6

### [MINOR] div#sensitivityPanel > div.sensitivity-toggles-grid > div.sensitivity-toggle-item > div.toggle-btn-group > button

- **Text**: "Low"
- **Color**: #6B7280 on #F3F4F6
- **Ratio**: 4.39:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F6

### [MINOR] div#sensitivityPanel > div.sensitivity-toggles-grid > div.sensitivity-toggle-item > div.toggle-btn-group > button

- **Text**: "High"
- **Color**: #6B7280 on #F3F4F6
- **Ratio**: 4.39:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #666D7B or adjust for 4.5:1 on bg #F3F4F6

### [MINOR] body > div#dashboardContent > div#targetModePanel > div#targetModeToggle > button

- **Text**: "Long-term Target"
- **Color**: #6B7280 on #E5E7EB
- **Ratio**: 3.9:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #616876 or adjust for 4.5:1 on bg #E5E7EB

### [MINOR] html > body > div#dashboardContent > div#keyFinding > h4

- **Text**: "Key Finding"
- **Color**: #DC2626 on #F3EFEA
- **Ratio**: 4.22:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #D21C1C or adjust for 4.5:1 on bg #F3EFEA

## dashboard/market-simulation.html

**2 failures**

### [MINOR] div.demo-controls-inner > div.controls-grid > div.control-block > div#yearSelector > button.year-pill.active

- **Text**: "2025"
- **Color**: #F8F9FA on #007FA4
- **Ratio**: 4.36:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #FDFEFF or adjust for 4.5:1 on bg #007FA4

### [MINOR] html > body > nav > div.demo-badge

- **Text**: "EIA 860 + 923 Data"
- **Color**: #B45A1A on #FEEFE5
- **Ratio**: 4.23:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #AF5515 or adjust for 4.5:1 on bg #FEEFE5

## dashboard/ref-hourly-strategy-comparison.html

**7 failures**

### [MINOR] body > div.article-wrapper > div.strat-cards > div.strat-card.card-2a > span.strat-tag

- **Text**: "Strategy 2A"
- **Color**: #6366F1 on #ECEDFD
- **Ratio**: 3.84:1 (need 4.5:1)
- **Font**: 11px, weight 600
- **Fix**: Change text to #5457E2 or adjust for 4.5:1 on bg #ECEDFD

### [MINOR] body > div.article-wrapper > div.strat-cards > div.strat-card.card-2b > span.strat-tag

- **Text**: "Strategy 2B"
- **Color**: #0284C7 on #E2F4FC
- **Ratio**: 3.63:1 (need 4.5:1)
- **Font**: 11px, weight 600
- **Fix**: Change text to #0070B3 or adjust for 4.5:1 on bg #E2F4FC

### [MINOR] div.article-wrapper > table.cmp-table > tbody > tr > td.cell-mid

- **Text**: "Medium"
- **Color**: #0284C7 on #E7F6FD
- **Ratio**: 3.7:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #0070B3 or adjust for 4.5:1 on bg #E7F6FD

### [MINOR] div.article-wrapper > table.cmp-table > tbody > tr > td.cell-mid

- **Text**: "Lower"
- **Color**: #0284C7 on #E7F6FD
- **Ratio**: 3.7:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #0070B3 or adjust for 4.5:1 on bg #E7F6FD

### [MINOR] div.article-wrapper > table.cmp-table > tbody > tr > td.cell-mid

- **Text**: "Moderate (claims grid mix)"
- **Color**: #0284C7 on #E7F6FD
- **Ratio**: 3.7:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #0070B3 or adjust for 4.5:1 on bg #E7F6FD

### [MINOR] div.article-wrapper > table.cmp-table > tbody > tr > td.cell-mid

- **Text**: "Implicit"
- **Color**: #0284C7 on #E7F6FD
- **Ratio**: 3.7:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #0070B3 or adjust for 4.5:1 on bg #E7F6FD

### [MINOR] div.article-wrapper > table.cmp-table > tbody > tr > td.cell-mid

- **Text**: "Moderate"
- **Color**: #0284C7 on #E7F6FD
- **Ratio**: 3.7:1 (need 4.5:1)
- **Font**: 13px, weight 600
- **Fix**: Change text to #0070B3 or adjust for 4.5:1 on bg #E7F6FD

## dashboard/typography-mockup.html

**1 failures**

### [MINOR] body > div.mockup-grid > div.mockup-section > div.mockup-label > span.current-label

- **Text**: "Current"
- **Color**: #EF4444 on #322131
- **Ratio**: 3.98:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #FE5353 or adjust for 4.5:1 on bg #322131

## dashboard/vre-investment-thesis-deck.html

**14 failures**

### [MINOR] div > table.heatmap-table > tbody > tr > td.heat-4

- **Text**: "~95% today"
- **Color**: #2372B9 on #D7E6F2
- **Ratio**: 3.94:1 (need 4.5:1)
- **Font**: 16px, weight 600
- **Fix**: Change text to #1968AF or adjust for 4.5:1 on bg #D7E6F2

### [MINOR] div > table.heatmap-table > tbody > tr > td.heat-4

- **Text**: "High"
- **Color**: #2372B9 on #D7E6F2
- **Ratio**: 3.94:1 (need 4.5:1)
- **Font**: 16px, weight 600
- **Fix**: Change text to #1968AF or adjust for 4.5:1 on bg #D7E6F2

### [MINOR] div > table.heatmap-table > tbody > tr > td.heat-1

- **Text**: "170 GW queue"
- **Color**: #DC2626 on #FCE9E9
- **Ratio**: 4.14:1 (need 4.5:1)
- **Font**: 16px, weight 600
- **Fix**: Change text to #D21C1C or adjust for 4.5:1 on bg #FCE9E9

### [MINOR] div > table.heatmap-table > tbody > tr > td.heat-1

- **Text**: "No capacity mkt"
- **Color**: #DC2626 on #FCE9E9
- **Ratio**: 4.14:1 (need 4.5:1)
- **Font**: 16px, weight 600
- **Fix**: Change text to #D21C1C or adjust for 4.5:1 on bg #FCE9E9

### [MINOR] div > table.heatmap-table > tbody > tr > td.heat-4

- **Text**: "~98% today"
- **Color**: #2372B9 on #D7E6F2
- **Ratio**: 3.94:1 (need 4.5:1)
- **Font**: 16px, weight 600
- **Fix**: Change text to #1968AF or adjust for 4.5:1 on bg #D7E6F2

### [MINOR] div > table.heatmap-table > tbody > tr > td.heat-4

- **Text**: "Faster queue"
- **Color**: #2372B9 on #D7E6F2
- **Ratio**: 3.94:1 (need 4.5:1)
- **Font**: 16px, weight 600
- **Fix**: Change text to #1968AF or adjust for 4.5:1 on bg #D7E6F2

### [MINOR] div > table.heatmap-table > tbody > tr > td.heat-1

- **Text**: "<30%"
- **Color**: #DC2626 on #FCE9E9
- **Ratio**: 4.14:1 (need 4.5:1)
- **Font**: 16px, weight 600
- **Fix**: Change text to #D21C1C or adjust for 4.5:1 on bg #FCE9E9

### [MINOR] div > table.heatmap-table > tbody > tr > td.heat-4

- **Text**: "98% paired"
- **Color**: #2372B9 on #D7E6F2
- **Ratio**: 3.94:1 (need 4.5:1)
- **Font**: 16px, weight 600
- **Fix**: Change text to #1968AF or adjust for 4.5:1 on bg #D7E6F2

### [MINOR] div > table.heatmap-table > tbody > tr > td.heat-4

- **Text**: "Premium"
- **Color**: #2372B9 on #D7E6F2
- **Ratio**: 3.94:1 (need 4.5:1)
- **Font**: 16px, weight 600
- **Fix**: Change text to #1968AF or adjust for 4.5:1 on bg #D7E6F2

### [MINOR] div > table.heatmap-table > tbody > tr > td.heat-1

- **Text**: "Difficult"
- **Color**: #DC2626 on #FCE9E9
- **Ratio**: 4.14:1 (need 4.5:1)
- **Font**: 16px, weight 600
- **Fix**: Change text to #D21C1C or adjust for 4.5:1 on bg #FCE9E9

### [MINOR] div > table.heatmap-table > tbody > tr > td.heat-4

- **Text**: "Premium"
- **Color**: #2372B9 on #D7E6F2
- **Ratio**: 3.94:1 (need 4.5:1)
- **Font**: 16px, weight 600
- **Fix**: Change text to #1968AF or adjust for 4.5:1 on bg #D7E6F2

### [MINOR] div > table.heatmap-table > tbody > tr > td.heat-4

- **Text**: "FCA"
- **Color**: #2372B9 on #D7E6F2
- **Ratio**: 3.94:1 (need 4.5:1)
- **Font**: 16px, weight 600
- **Fix**: Change text to #1968AF or adjust for 4.5:1 on bg #D7E6F2

### [MINOR] div > table.heatmap-table > tbody > tr > td.heat-1

- **Text**: "Minimal"
- **Color**: #DC2626 on #FCE9E9
- **Ratio**: 4.14:1 (need 4.5:1)
- **Font**: 16px, weight 600
- **Fix**: Change text to #D21C1C or adjust for 4.5:1 on bg #FCE9E9

### [MINOR] div > table.heatmap-table > tbody > tr > td.heat-4

- **Text**: "~98%"
- **Color**: #2372B9 on #D7E6F2
- **Ratio**: 3.94:1 (need 4.5:1)
- **Font**: 16px, weight 600
- **Fix**: Change text to #1968AF or adjust for 4.5:1 on bg #D7E6F2

