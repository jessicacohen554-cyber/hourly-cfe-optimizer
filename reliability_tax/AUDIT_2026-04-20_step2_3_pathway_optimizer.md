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

