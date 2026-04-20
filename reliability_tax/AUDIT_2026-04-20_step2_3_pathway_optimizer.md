# Audit: Step 2.3 Pathway Optimizer (Post-Fix)
**Date:** 2026-04-20
**Auditor:** pipeline-audit-agent
**Target:** `scripts/step_2_3_pathway_optimizer.py`, `scripts/pipeline_config.py`
**Prior audit:** `reliability_tax/AUDIT_2026-04-19_step2_3_pathway_optimizer.md`

## Header

**Design intent:** Proactive clean-firm procurement (Pathway 3, NOAK cost maturity by 2035) results in materially less cumulative new gas capacity built by 2050 compared to reactive VRE-first procurement (Pathway 1, NOAK 2045) across all major US ISO grids at CFE endpoints ≥ 80%, driven by Wright's Law learning curves that make nuclear/CCS/geothermal cost-competitive earlier and reduce worst-hour residual demand that would otherwise require new gas builds.

**Secondary claims:**
- Pathway 3 consistently procures more clean-firm (nuclear/CCS/geothermal) than Pathway 1 in the same year
- NPV of Pathway 3 is lower than or equal to Pathway 1 at endpoints ≥ 90% in ≥ 5 of 7 ISOs
- Gas sizing is physically consistent (decreasing or flat as CFE% rises — never inverted)

**Hypothesis:** P3_gas_mw < P1_gas_mw at ep80/90/99 across all 7 ISOs; material divergence (>10%) in ≥ 5/7 ISOs.

**Observed result that prompted audit:** Prior audit (2026-04-19) found P3 ≡ P1 (bit-for-bit identical) in ERCOT. Fix applied: `INCLUDE_GAS_COST_IN_ARGMIN = True` in `pipeline_config.py`. Current audit checks whether fix actually worked.

**Planned scope:**
- Phase 1: Read all listed files (step_2_3_pathway_optimizer.py, pipeline_config.py, prior audit, MANIFEST.json, 6 spot-check JSONs)
- Phase 2: Design-vs-code trace, with focus on INCLUDE_GAS_COST_IN_ARGMIN wiring
- Phase 3: Build hypothesis-vs-data table from MANIFEST + spot-check JSONs
- Phase 4: External benchmarks (nuclear NOAK, CCS, Wright's Law, discount rate)
- Phase 5: Classify verdict

## Findings

### Finding 1 — ERCOT P1 ≡ P3 gas at all endpoints: fix did not change ERCOT gas build [severity: HIGH] [phase: 3]

- **Where.** `analysis/reliability-tax/data/ERCOT/pathway1_ep{60,70,75,80,85,90,95,97p5,99,99p9}.json` vs `pathway3_ep*.json`; `MANIFEST.json` under `ERCOT__pathway{1,3}__*`.
- **What.** ERCOT P1 and P3 produce bit-for-bit identical `total_new_gas_built_mw` at every endpoint except ep85 (where P3 builds MORE gas than P1). Specifically: ep60=53,496, ep70=104,987, ep75=120,626, ep80=134,343, ep90=111,486, ep95=94,389, ep99=77,021 — identical for both pathways at all these endpoints. At ep85 P3=121,202 MW vs P1=97,769 MW — P3 is 24% worse. P3 NPV is lower than P1 NPV at ep90 ($480B vs $496B) and ep99 ($851B vs $855B) but this reflects cheaper intermediate clean-firm LCOE costs under NOAK-2035, not a gas-build reduction. The endpoint mix for ERCOT ep90 is {wind:90, solar_batt8:10} for both P1 and P3 — zero clean firm in the mix.
- **Why it matters.** The post-fix code with `INCLUDE_GAS_COST_IN_ARGMIN=True` did not change ERCOT's argmin. ERCOT's VRE (onshore wind at ~$52-61/MWh) is cheap enough that even after adding the gas-penalty from `score_ef_batch_with_gas`, the cheapest total-cost row remains a pure-VRE mix for P3 and P1 alike. The fix moved the argmin in VRE-constrained ISOs (PJM, CAISO) but not in VRE-rich ERCOT.
- **Needs follow-up in.** Phase 5 classification.

### Finding 2 — Gas monotonicity universally violated: ep60→ep80→ep90→ep99 is NOT monotone-decreasing in any ISO [severity: HIGH] [phase: 3]

- **Where.** `MANIFEST.json` all ISOs, all pathways; re-derived monotonicity check on ep60, ep80, ep90, ep99.
- **What.** In every ISO and every pathway combination tested, gas_mw(ep60) < gas_mw(ep80) > gas_mw(ep90) — gas peaks near ep80 and then declines. Example: ERCOT P1: ep60=53,496 → ep80=134,343 → ep90=111,486 → ep99=77,021. PJM P1: ep60=52,882 → ep80=106,089 → ep90=94,147 → ep99=96,998. None of the 14 ISO×pathway combinations tested is monotone-decreasing across the full audited range. The gas curve is a hump with peak near ep75-ep85, not monotone-decreasing.
- **Why it matters.** The audit brief specifies "Gas sizing is physically consistent (decreasing or flat as CFE% rises — never inverted)" as a sanity check. The data shows gas IS inverted between ep60 and ep80 in all ISOs. However, this matches the prior audit's Finding 4 (post-correction) — the hump from ep0→ep80 is physically defensible (more clean procurement is needed as the target tightens, requiring more gas backup before clean firm matures). The inversion between ep80→ep90 for some ISOs (e.g., PJM P1: ep90=94,147 < ep80=106,089, then ep99=96,998 > ep90=94,147) is more problematic. The PJM monotonicity violation is a real non-monotonicity on the descending side.
- **Needs follow-up in.** Phase 5 — determines if this is a code bug or an economically valid outcome of the mixed-resource EF.

### Finding 3 — P3 builds MORE gas than P1 in 3 ISO×endpoint combinations [severity: HIGH] [phase: 3]

- **Where.** `MANIFEST.json`: ERCOT ep85 (P3=121,202 MW vs P1=97,769 MW, +24%), SPP ep90 (P3=15,142 vs P1=11,381, +33%), MISO ep95 (P3=36,916 vs P1=29,746, +24%).
- **What.** After applying `INCLUDE_GAS_COST_IN_ARGMIN=True`, Pathway 3 selects a clean-firm-inclusive endpoint mix at these ISO×endpoint combinations. The clean firm provides hourly dispatch coverage (lowering gas residual in the argmin scoring), but requires complementary VRE to hit the CFE target — and the VRE+clean-firm combination's worst-hour residual is actually HIGHER than the pure-VRE P1 mix's residual at these specific iso×endpoint points. Mechanism: at ERCOT ep85, P3 selects {clean_firm:10%, wind:80%}; P1 selects {wind:80%, solar_batt8:10%}. The clean-firm-inclusive mix has higher worst-hour residual than the battery-backed VRE mix, so P3 sizes MORE gas, not less.
- **Why it matters.** The argmin with gas-cost endogenization is producing perverse outcomes: the gas cost penalty in the scoring function (computed from a simplistic `worst_hour_residual_per_row` approximation) differs from the actual gas sizing in `size_required_gas_mw`. The argmin scores one residual but the actual production path computes a different residual for the selected mix — so the argmin's gas-penalty expectation is violated in the realized output.
- **Needs follow-up in.** Phase 2 analysis of why `worst_hour_residual_per_row` in the argmin and `worst_hour_gas_sizing` in the sizing step may diverge.

### Finding 4 — MISO P3 NPV is 78% HIGHER than P1 at ep80 despite less gas built [severity: HIGH] [phase: 3]

- **Where.** `analysis/reliability-tax/data/MISO/pathway{1,3}_ep80.json`; MANIFEST under `MISO__pathway{1,3}__ep80`.
- **What.** MISO ep80: P3_npv=$1,030B vs P1_npv=$580B — P3 costs 78% more. P3 builds 34% less gas (40,250 MW vs 61,129 MW for P1). The gap arises because P3's argmin (with gas-penalty endogenization) selects a mix with more clean firm AND a large VRE complement. Annual gross_operating_usd for P3 is 1.91× P1's at 2040 and 2.95× at 2050. Terminal ledger shows P3 cumulates 4,437 TWh/yr of wind vs P1's 1,786 TWh/yr and 1,618 TWh/yr of solar vs P1's 363 TWh/yr — the gas-penalty-driven argmin selected a clean-firm-inclusive row whose complement VRE requirement is far more expensive than the pure-VRE P1 row's VRE requirement.
- **Why it matters.** The secondary hypothesis states "NPV of Pathway 3 is lower than or equal to Pathway 1 at endpoints ≥ 90% in ≥ 5 of 7 ISOs." At ep90, P3 NPV is lower in only 3 of 7 ISOs (ERCOT, PJM, CAISO). At ep80, P3 NPV is higher in 5 of 7 ISOs including an extreme outlier (MISO +78%). The fix worsened the NPV claim in MISO: the endogenized gas cost moved P3 to a more expensive mix that happens to use less gas but costs far more in VRE procurement.
- **Needs follow-up in.** Phase 2 trace, Phase 5 classification.

### Finding 5 — INCLUDE_GAS_COST_IN_ARGMIN pathway-differentiation is correct but argmin-vs-realized residual mismatch creates perverse outcomes [severity: HIGH] [phase: 2]

- **Where.** `scripts/step_2_3_pathway_optimizer.py:1790–1863` (`score_ef_batch_with_gas`); `:862–897` (`size_required_gas_mw`); `scripts/step2_2a_cost_optimization.py:547–585` (`worst_hour_residual_per_row`).
- **What.** The fix correctly wires INCLUDE_GAS_COST_IN_ARGMIN: `score_ef_batch_with_gas` calls `score_ef_batch` (pathway-aware, uses P3's NOAK-2035) then adds `new_gas_mw × (capex + fuel)` per EF row, where `new_gas_mw` is derived from `worst_hour_residual_per_row`. The argmin then picks the row with lowest (clean-energy-cost + gas-penalty). In `solve_pathway`, the actual gas sizing uses `size_required_gas_mw` → `worst_hour_gas_sizing`, which calls the same underlying `worst_hour_residual_per_row` via a different path but with the same dispatch cache. The residuals should theoretically match. However, the argmin computes `gap_mw = resid_norm × demand_mwh_grown` using the CURRENT year's grown demand, while `size_required_gas_mw` is called with BASE-year demand and a growth factor `gf`. Both should produce the same result (`demand_twh × 1e6 × gf = demand_mwh_grown`). The values should be consistent. The perverse MISO outcome (P3 selects a mix that has higher realized gas than the argmin predicted) is therefore not a calculation mismatch — it is a legitimate property of the EF: the row P3's argmin selected had lower `resid_norm` in the argmin's scoring, but `size_required_gas_mw` then uses `worst_hour_gas_sizing` with the full `mix_pct + storage_pct` dict and applies `gas_raw_2025` subtraction and RA margin. If the stored archetype in the dispatch cache for the chosen row differs from the one scored in the batch, results diverge.
- **Why it matters.** The fix is mechanically correct in its wiring (True flag reaches the scorer, scorer calls the right function). The perverse outcomes (P3 > P1 gas in 3 ISO×endpoint cases) are a consequence of the endogenized gas penalty distorting the argmin away from the NPV-minimizing row — the penalty correctly captures some gas cost but the penalty magnitude is not calibrated to match the realized NPV objective, so the argmin is no longer minimizing NPV; it is minimizing (clean-energy-cost + crude-gas-cost-proxy) which is a different objective.
- **Needs follow-up in.** Phase 4 (external sanity checks on parameter magnitudes), Phase 5.

### Finding 6 — Primary hypothesis partially confirmed: P3 < P1 gas by >10% in 4/7 ISOs at ep90, not 5/7+ [severity: HIGH] [phase: 3]

- **Where.** `MANIFEST.json` all ISOs, ep80/ep90/ep99.
- **What.** P3 < P1 gas by >10% at: ep80: CAISO(75%), MISO(34%), NYISO(19%), NEISO(100%) = 4/7. ep90: PJM(64%), CAISO(75%), MISO(18%), NEISO(91%) = 4/7. ep99: PJM(62%), CAISO(69%), NYISO(100%), NEISO(100%) = 4/7. P3 ≡ P1 in ERCOT at all endpoints. P3 builds MORE gas than P1 at: ERCOT ep85, SPP ep90, MISO ep95/ep99 (MISO ep99: P3=32,493 vs P1=29,746, P3 is 9% worse). The stated hypothesis requires "all major US ISOs" — ERCOT is the largest US ISO by load hours served and fails at every endpoint.
- **Why it matters.** The hypothesis (P3 < P1 in all 7 ISOs) is not confirmed. The fix improved results vs. the pre-fix state (where ERCOT was bit-for-bit identical but now has some differentiation at ep85) but the primary claim ("all ISOs") still fails. The NPV secondary claim fails even more strongly: P3 NPV < P1 NPV at ep90 in only 3/7 ISOs vs. the stated ≥5/7 requirement.
- **Needs follow-up in.** Phase 5 classification.

