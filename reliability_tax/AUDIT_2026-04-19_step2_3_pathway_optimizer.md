# Audit — step 2.3 pathway optimizer + section 2/3 chart payloads

- **Date:** 2026-04-19
- **Auditor:** pipeline-audit-agent
- **Targets:**
  - `scripts/step_2_3_pathway_optimizer.py`
  - `reliability_tax/charts/gen_section2_gas_hump.py` (and `gen_section2_overbuild.py` as relevant)
  - `reliability_tax/charts/gen_section3_reliability_tax.py`
- **Scope restriction:** ERCOT + PJM only; endpoints 80% / 90% / 99% only.

## Design intent (as stated)

The reliability-tax page wants to prove that a grid pursuing aggressive clean-firm procurement early (Pathway 3 — proactive clean firm) ends up building materially less new gas than a grid that waits and reacts (Pathway 1 — reactive pivot). The mechanism is Wright's Law: early clean-firm orders drive cost-learning, NOAK arrives sooner and cheaper, and the system swaps gas for clean firm before the gas would have been built. The gap should appear in every ISO by 2050, and should widen as the clean-energy target tightens from 80% toward 99%.

## Hypothesis

Pathway 3 builds materially less new gas than Pathway 1 in every ISO and at every endpoint in the 80–99% range. The P3-minus-P1 gas gap is non-zero, directionally negative (P3 < P1), and widens as the endpoint rises.

## Observed result (prompting this audit)

Pathway 1 and Pathway 3 build essentially the same new-gas capacity in 6 of 7 ISOs (CAISO is the exception). Gas builds are also non-monotonic in the endpoint — they peak at 80%, then decline toward 99%.

## User's suspected sources (priors, to be independently verified)

1. SBTi target-year snapping may collapse P1 and P3 to identical clean-firm availability at medium endpoints.
2. NOAK-in-2035 as a hard timing constant may arrive too late to create a real P3 cost-learning advantage.
3. A bug in `_filter_pathway_3`, `solve_pathway`, `compute_clean_firm_tranches_for_year`, or `size_required_gas_mw` may prevent P3 from procuring clean-firm earlier than P1.
4. The 80% non-monotonicity may be a rounding / snapping / floor-to-ladder artifact at the endpoint edge.

## Planned scope

- Phase 1: read `CLAUDE.md`, `SPEC.md` current status, reliability-tax methodology, data_loader reference, target files in full, sibling `gen_*` examples.
- Phase 2: design-vs-code trace of step 2.3 + section 2/3 payload gens.
- Phase 3: open ERCOT + PJM cached outputs at ep80 / ep90 / ep99 for P1 and P3 and compute the P3 − P1 gas gap directly.
- Phase 4: external checks on NOAK 2035 timing, Wright's-Law learning rates, SBTi 1.5°C trajectory, FOAK / NOAK LCOE anchors.
- Phase 5: classify.

## Findings

### Finding 1 — P3 has no clean-firm floor; differentiation is NOAK-year only [severity: HIGH] [phase: 2]

- **Where.** `scripts/step_2_3_pathway_optimizer.py:1493–1513` (block comment + `_filter_pathway_3`) and `:1818–1831` (the pathway filter dispatch in `select_target_mix`).
- **What.** Pathway 3 is implemented as "no filter at all" — `_filter_pathway_3` returns the EF unchanged. A comment block explicitly states this was a design decision: "SPEC §24.8 — Pathway 3 is differentiated from P1/P2 *solely* by the exogenous NOAK year (`pc.NOAK_YEAR_BY_PATHWAY['3'] = 2035`). … A floor was prototyped and rejected because it hard-wired the outcome (P3 ≠ P1 by construction) and smothered the regional heterogeneity." The P2a/P2b/P3 branch of `select_target_mix` (`elif pathway in ('2a', '2b', '3'): pass`) shows the same thing — P2a, P2b, and P3 all see the identical unfiltered EF; only NOAK-year differs between them.
- **Why it matters.** The stated design intent says P3 should "build clean firm proactively" and "swap gas for clean firm before the gas would have been built." But the code does not force P3 to do that — it only lets P3 see a cheaper Wright's-Law curve via NOAK 2035 vs. 2045 for P1a/P2a. If the year's SBTi-snapped CFE target is low enough that a pure-VRE row still wins on cost even under the accelerated curve (ERCOT's VRE-richness is the explicit example the comment uses), P3 will pick the same VRE-only mix that P1 does. This is the most plausible mechanism for "P1 ≡ P3 in 6 of 7 ISOs" — P3 has no mechanism to *build clean firm* unless the cost minimizer chooses it, and at intermediate endpoints the minimizer frequently does not.
- **Needs follow-up in.** Phase 3 data check (confirm the endpoint mixes are identical for P1a vs. P3 in ERCOT/PJM at ep80/ep90).

### Finding 2 — Dead pivot logic; P2a/P2b/P3 distinguished only by NOAK year [severity: MED] [phase: 2]

- **Where.** `scripts/step_2_3_pathway_optimizer.py:2266–2271` (comment block in `solve_pathway`) and `:1818–1831` (pathway dispatch in `select_target_mix`).
- **What.** The methodology (`methodogy.md` Card E, Card C) describes pathway-2a/2b as "freeze-and-forward" with pivot triggers — P2a pivots at the 90% plateau, P2b pivots when marginal $/CFE% exceeds cheapest clean firm LCOE. But the code notes "P2a / P2b / P3 are now fully exogenous: pathway differentiation lives in `pc.NOAK_YEAR_BY_PATHWAY` (learning-curve speed), not in endogenous pivot triggers. PivotState is kept as a data type for JSON-schema compat but is never .trigger()'d — the should_pivot_2a / should_pivot_2b helpers are dead code." The methodology doc has drifted from the code.
- **Why it matters.** This is a "silent change of counterfactual." The methodology doc describes one experiment; the code runs a different one. Whether this matters for the hypothesis depends on whether it collapses P3 onto P1 — which it can, since P3 no longer has a "build clean firm proactively" forcing mechanism. This amplifies Finding 1.
- **Needs follow-up in.** Phase 3 data check, Phase 5 classification.

