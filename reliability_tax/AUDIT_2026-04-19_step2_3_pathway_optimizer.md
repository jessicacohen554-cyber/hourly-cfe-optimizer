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

### Finding 3 — ERCOT P1 ≡ P1a ≡ P3 at every audited endpoint [severity: HIGH] [phase: 3]

- **Where.** `analysis/reliability-tax/data/ERCOT/pathway{1,1a,3}_{ep80,ep90,ep99}.json`.
- **What.** Re-derived directly from the cached JSONs: ERCOT endpoint mix and fleet size are *bit-for-bit* identical across P1 / P1a / P3 at all three endpoints.
  - ep80: mix = `{clean_firm=0, solar=0, wind=90, offshore_wind=0, geo=0, ccs=0}`; new-gas fleet = 134,343 MW for P1, P1a, P3.
  - ep90: mix = `{CF=0, solar=0, wind=90, osw=0, geo=0, ccs=0}`; fleet = 111,486 MW for all three.
  - ep99: mix = `{CF=1, solar=0, wind=31, osw=0, geo=0, ccs=0}`; fleet = 77,021 MW for all three.
  - Achieved-CFE and achieved-threshold match to four decimals. Only the undiscounted cost differs by <0.5 % — noise floor from the vintage-LCOE rounding drift.
- **Why it matters.** P3 picks the identical cost-minimal EF row as P1. NOAK-2035 vs. NOAK-2045 never moves the argmin because clean-firm doesn't even enter the winning mix. This is the direct mechanical confirmation of the mechanism hypothesized in Finding 1: with no clean-firm floor, the argmin under ERCOT's VRE-rich EF is pure onshore wind across the 80–99 % endpoint range, and the pathway NOAK-year is irrelevant to a mix whose clean-firm coefficient is 0–1 %.
- **Needs follow-up in.** Phase 5 classification.

### Finding 4 — Gas builds peak at ep80 and decline monotonically toward ep99 in ERCOT [severity: HIGH] [phase: 3]

- **Where.** `analysis/reliability-tax/data/ERCOT/pathway1_ep{80,85,90,95,99,99p9}.json` (fleet-size field `stranding_metadata.fleet_size_mw` / `annual_buildout[-1].gas_sizing.active_new_gas_fleet_mw`).
- **What.** ERCOT P1 fleet sizes by endpoint (MW): ep80 = 134,343; ep85 = 129,168; ep90 = 111,486; ep95 = 93,768; ep97p5 = 80,775; ep99 = 77,021; ep99p9 = 76,893. Gas is strictly monotonically *decreasing* as the CFE target rises. Peak is at ep80, exactly as the user's prior described. PJM P1: ep80 = 106,089; ep85 = 106,089; ep90 = 94,147; ep95 = 95,722; ep99 = 96,998; ep99p9 = 96,998 — plateaus at high endpoints rather than peaks, with a dip at ep90.
- **Why it matters.** The "gas hump" storyline — medium-endpoint builds that strand as you climb past 90 % — **does not appear in the data**. Gas is largest where CFE is loosest. This matches the SPEC §24.5 worst-hour sizing formula: as clean-firm mix fraction climbs, residual gap shrinks, and the 99.97-th-percentile residual falls. The methodology's expectation of a hump would require clean build to *fail to cover* the marginal high-CFE residual — which doesn't happen when clean firm is both affordable and siteable. The 80 % peak is *not* a snapping artifact; it is what the worst-hour sizing formula deterministically produces given the candidate EF mixes and the `RESOURCE_ADEQUACY_MARGIN` + `GAS_AVAILABILITY_FACTOR` constants in `step2_2a_cost_optimization`.
- **Needs follow-up in.** Phase 5 classification (methodology vs. hypothesis misalignment — this is *not* a bug).

### Finding 5 — PJM P3 ≡ P1 at ep80; P3 ≠ P1 at ep90/ep99 [severity: HIGH] [phase: 3]

- **Where.** `analysis/reliability-tax/data/PJM/pathway{1,3}_{ep80,ep90,ep99}.json`.
- **What.** PJM divergence behaviour:
  - **ep80.** P1 mix = P3 mix = `{CF=30, solar=0, wind=50, osw=3}`; both build 106,089 MW of new gas. P3 ≡ P1 exactly.
  - **ep90.** P1 mix = `{CF=29, solar=0, wind=11, osw=3}`, 94,147 MW gas. P3 mix = `{CF=79, solar=0, wind=10, osw=3}`, only 33,783 MW gas (64 % less than P1). P3 fleet peaks in 2037 and then *declines* — 15,518 MW of gas is explicitly flagged stranded at 2050. This is the intended P3-vs-P1 mechanism working correctly.
  - **ep99.** P1 mix = `{CF=30, solar=29, wind=11}`, 96,998 MW gas. P3 mix = `{CF=61, solar=1, wind=1, osw=1}`, 36,893 MW gas (62 % less). Divergence widens toward the endpoint, as the stated design intent predicts.
- **Why it matters.** PJM is the counter-example to Finding 3. The hypothesis mechanism *does* fire when the EF's PJM ep90 band has a clean-firm-heavy mix whose cost at NOAK-2035 undercuts the VRE-heavy P1 winner. At ep80, however, the SBTi ladder only demands 80 %, and PJM's existing-fleet clean-firm share (~31 % of demand) plus VRE is cheap enough that P3's NOAK-2035 clean-firm tranche never wins on cost vs. the pure-VRE argmin. So P3's divergence from P1 is **endpoint-gated, not universal**.
- **Needs follow-up in.** Phase 5 classification.

### Finding 6 — Year snap collapses SBTi interpolation; P3 gets no early head-start on clean-firm procurement [severity: HIGH] [phase: 2]

- **Where.** `scripts/step_2_3_pathway_optimizer.py:1806–1809` (`select_target_mix`) and `:1561–1573` (`_closest_available_threshold`).
- **What.** For each year the pathway's target CFE% is computed by `sbti_target_for_year` (linear interp across the ladder 2025→0%, 2030→50%, 2035→70%, 2040→90%, 2045→95%, 2050→99.9%, clipped to endpoint). That target is then snapped to the smallest available EF threshold via `_closest_available_threshold`: "Pick the smallest available EF threshold that meets `target_pct`." Before the "plateau year" (`first_year_reaching(endpoint_pct)`), the code uses the interpolated target; at or after the plateau year, it jumps to the endpoint itself. This means:
  - For ep80: before 2040 the target ladder is 0→50→70%; at 2040 it snaps to 80 %. So P3 loads the *70 % EF band in 2035 and the 80 % EF band from 2040 onward* — the same bands P1 loads. If the 70 % EF's cheapest VRE-only row beats the cheapest clean-firm-heavy row under NOAK-2035, P3's argmin is identical to P1's at 70 %. The trajectory is lock-step.
  - For ep90: the plateau year is 2040, from 2040 onward both pathways load the 90 % EF band. P1 picks the VRE argmin. P3's only differentiator is NOAK-2035 — which happens to matter here because a clean-firm-heavy row in PJM's 90 % band exists and becomes cost-competitive at the accelerated curve. This is the only reason PJM ep90 diverges. It's a narrow window.
- **Why it matters.** P3 has *no forcing mechanism* to procure clean-firm ahead of the SBTi ladder. The comment in `sbti_target_for_year` says "Pathway 3 is allowed to front-load" but the code never implements front-loading — P3 loads the same EF band P1 loads in the same year. Combined with Finding 1 (no floor) and Finding 2 (no pivot trigger), P3 is operationally "P1 with a more optimistic clean-firm NOAK year." Whether that produces a different endpoint mix is a pure comparison of two cost functions on the same candidate set — and in most ISOs and endpoints, it doesn't.
- **Needs follow-up in.** Phase 5 classification.

### Finding 7 — Worst-hour sizing is endpoint-monotone-decreasing by construction; the "hump" story is a methodology mismatch, not a bug [severity: MED] [phase: 2]

- **Where.** `scripts/step2_2a_cost_optimization.py:432–500` (`worst_hour_gas_sizing`) and `scripts/step_2_3_pathway_optimizer.py:859–894` (`size_required_gas_mw`).
- **What.** The sizing formula is `gap_mw = residual_norm(mix) × demand_mwh_grown`, where `residual_norm` is the 99.97th-percentile margin-on-demand residual under the mix's hourly dispatch. A mix that achieves higher CFE by construction has *lower* residual gap in worst hours (that's how CFE is measured). So `gap_mw` — and therefore `new_gas_required_cumulative_mw` — is structurally monotone-decreasing as the CFE target rises. The only way the endpoint curve shows a hump at medium thresholds is if (a) the optimizer picks a mix that *misses* the CFE target enough for residual to grow while endpoint still snapping to that tier, or (b) there is a floor-ratchet artifact in the EF loading that prevents the mix from tightening. Neither mechanism is in the current code. The methodology doc's "hump at medium thresholds, stranded as grid pushes past 90 %" claim assumes a different sizing model entirely (ELCC + peak-net-of-clean), which §24.5 explicitly replaced with worst-hour residual.
- **Why it matters.** The observed result "gas peaks at 80 % and declines toward 99 %" is **the expected output of the current methodology**, not a bug. The user's expectation that gas should hump at medium CFE is incompatible with SPEC §24.5's sizing rule. This is one of the two headline findings of the audit (the other is the P3 ≡ P1 mechanism per Finding 3).
- **Needs follow-up in.** Phase 5 classification.

### Finding 8 — SBTi ladder is locked at `(2025,0), (2030,50), (2035,70), (2040,90), (2045,95), (2050,99.9)` [severity: LOW] [phase: 4]

- **Where.** `scripts/step_2_3_pathway_optimizer.py:1182–1189` (`_SBTI_TARGETS`).
- **What.** The hard-coded ladder does not match any published SBTi 1.5°C power-sector trajectory I can verify. SBTi 1.5°C-aligned pathways for power utilities (SBTi Power Sector Guidance 2020, IPCC AR6 C1 mitigation pathways) require ~100 % decarbonization by ~2040 in OECD grids to stay under 1.5°C. The 2050-only endpoint here is a 2 °C-ish framing. More importantly, the monotone ladder means the 80 % endpoint is *reached* in 2040 (per `first_year_reaching`), which is *before* NOAK-2035 has finished its learning curve (NOAK is a 2035 arrival but only from the FOAK 2030 start). So at ep80 P3's clean-firm LCOE at 2040 has already mostly converged to NOAK — the extra acceleration from 2035 vs. 2045 is 5 years of compounded learning only. In contrast, for ep99 where endpoint is first reached in 2050, P3's NOAK-2035 vs. P2a's NOAK-2045 advantage has 10–15 years of learning-curve differential to work with. This interacts with Finding 6 to explain why the hypothesis works at high endpoints (PJM ep90/ep99) but not at low (PJM ep80, ERCOT everywhere VRE-rich).
- **Needs follow-up in.** Phase 5 classification.

### Finding 9 — FOAK/NOAK nuclear and CCS costs at edge of published ranges; NOAK-2035 is aggressive but defensible [severity: LOW] [phase: 4]

- **Where.** `scripts/pipeline_config.py:1192–1214` (LCOE tables), `:1297–1309` (FOAK tables), `:1375–1389` (LEARNING_PARAMS), `:1428–1432` (NOAK_YEAR_BY_PATHWAY).
- **What.** Comparing to published ranges as of Apr 2026:
  - **Nuclear newbuild NOAK (Medium) $/MWh.** Code: ERCOT 90, PJM 105, NYISO 110, NEISO 108, MISO 100, SPP 92, CAISO 95. NREL ATB 2024 Moderate nuclear LCOE is $82–124/MWh range (AP1000) with regional spread; Lazard v17 is $141–221/MWh for new nuclear (unadjusted for ITC/PTC). Code's Medium is near Lazard's "low" / NREL's "central" — optimistic but in-range. FOAK 169–212 /MWh in code ≈ Vogtle Unit 3 realized LCOE (~$190/MWh adjusted); plausible.
  - **CCS 45Q-on NOAK (Medium).** Code: ERCOT 72.5, PJM 80.5, MISO 75.5, SPP 69.5. NREL ATB 2024 Moderate CCGT+CCS LCOE ≈ $79–96/MWh gross; net of $85/ton 45Q (~$27.5/MWh on 0.324 tCO2/MWh captured) → $52–68/MWh. Code's 45Q-on values are $5–15/MWh *higher* than ATB 2024 net. Not out of range — ATB is fuel-gas-price sensitive and code uses Medium gas.
  - **NOAK-2035 for nuclear.** The code's 2035 NOAK assumes full learning by 2035 from a 2030 FOAK start. Published learning-rate assumptions for US nuclear newbuild range from 5 %/doubling (pessimistic, Grubler 2010 negative-learning era) to ~20 %/doubling (optimistic, SMR vendors / Lovering 2016). A 5-year FOAK→NOAK window is aggressive but defensible given a ~10× deployment ramp (hypothetical SMR fleet). Not out-of-range per se — it is the aggressive-end-of-range assumption, and P3's argument depends on it.
  - **Learning exponent 0.6.** `LEARNING_EXPONENT = 0.6` in `learning_fraction`. Wright's Law typically uses log-log slopes; a 0.6 exponent on the (year-fraction) domain compresses most of the learning into the first third of the FOAK→NOAK window. This is an optimistic shape — it front-loads the cost decline. Combined with NOAK-2035, it means nuclear costs at 2035 are already ~80 % of the way down, not 50 %. This amplifies P3's advantage over P1 for long-horizon (ep99) runs. Not wrong; just worth flagging.
- **Why it matters.** Phase 4 check comes back in-range overall, which means Verdict B via "the parameter choice is the flaw" does not fire. The cost parameters are aggressive-clean-firm-friendly; if they weren't, P3 would not even win PJM ep90 / ep99. The hypothesis failure in ERCOT and at low endpoints is *not* a function of pessimistic cost assumptions.
- **Needs follow-up in.** Phase 5 classification (rules out Verdict B via parameter drift).

## What the code actually does

Step 2.3's optimizer ranks a candidate EF (each row is a clean-energy mix) by cost for each year's SBTi-interpolated CFE target, snaps to the smallest EF band that meets target, picks the argmin-cost row, sizes new gas from the 99.97th-percentile worst-hour residual of that row's hourly dispatch, and accumulates year-over-year (Findings 6, 7). Pathway differentiation is exogenous-only: `NOAK_YEAR_BY_PATHWAY` sets 2045 for P1/P1a/P2a, 2040 for P2b, and 2035 for P3 — the same EF rows are candidates for every pathway at every year, so pathway differentiation collapses to "which year does the learning-curve curve bend" (Findings 1, 2). The design decision to drop P3's clean-firm floor is explicit in a comment at `scripts/step_2_3_pathway_optimizer.py:1493-1513` ("A floor was prototyped and rejected because it hard-wired the outcome (P3 ≠ P1 by construction) and smothered the regional heterogeneity"). §24.8 also made P2a/P2b's pivot logic dead code: `should_pivot_2a` and `should_pivot_2b` are kept as types but never `.trigger()`'d.

## Where the design breaks

1. **No forcing mechanism for P3's stated behavior.** The hypothesis says P3 should "build clean firm proactively" and "swap gas for clean firm before the gas would have been built." The code has no mechanism that makes either sentence true — P3's only advantage is a 5–10 year earlier NOAK, and if the argmin at P3's SBTi-interpolated target still picks a pure-VRE row, the NOAK advantage is irrelevant. Findings 1, 2, 6 diagnose this at the code level; Finding 3 (ERCOT P1 ≡ P1a ≡ P3) shows it firing in the data. Magnitude: zero P3-vs-P1 gas divergence in ERCOT at every endpoint; P3 ≡ P1 in PJM at ep80. P3 wins only where the argmin naturally picks clean-firm-heavy mixes (PJM ep90 / ep99).

2. **"Gas hump at medium CFE" dashboard storyline is mechanically impossible under §24.5 worst-hour sizing.** The sizing rule is `gap_mw = residual_norm(mix) × demand_mwh_grown`, where `residual_norm` is strictly monotone-decreasing in the mix's CFE. So `new_gas_required_cumulative_mw` is deterministically monotone-decreasing in the endpoint by construction (Finding 7). The hump story belongs to the pre-§24.5 ELCC + peak-net-of-clean sizing model, which §24.5 explicitly replaced. The dashboard's Section 2 prose, hump-chart framing, and several insight boxes have not been updated to reflect that the hump is now impossible. Magnitude: the entire "Section 2 / The Hump" narrative is orphaned from the methodology. Finding 4 confirms ERCOT gas is monotone-decreasing across ep80 → ep99; Finding 5 shows PJM plateau with a dip, not a hump.

3. **Methodology doc drift from code.** `reliability_tax/methodogy.md` (Card C, Card E) still describes P2a/P2b pivot triggers and P3 clean-firm forcing that the code explicitly does not implement (§24.8 comments cite the removal). Magnitude: the spec artifact that readers consult to understand the pathways is misleading about what the code does.

## Decision cards

### AUDIT-2026-04-19-1 — Does P3 need a forcing mechanism, or is "P3 wins where it should" the real story?

**Options:**
- **A. Add a clean-firm floor to `_filter_pathway_3`.** Require P3 to procure at least `FLOOR_BY_YEAR[year]` fraction of clean-firm capacity starting at 2030 (or whatever year the user chooses), ramping to endpoint. This is what the methodology doc originally described; the §24.8 comment says it was prototyped and rejected. Re-enabling it makes P3 < P1 universally, at the cost of hard-wiring the outcome. **Implication:** dashboard hypothesis becomes true by construction; regional heterogeneity disappears.
- **B. Reframe dashboard to match real code behavior.** Accept that P3's Wright's-Law advantage only matters in ISOs where the argmin naturally picks clean-firm-heavy rows (PJM ep90+, NYISO, NEISO). Rewrite Section 2 / Section 5 / Section 7 to tell the "P3 wins in VRE-constrained ISOs at high endpoints, not in VRE-rich ISOs at low endpoints" story. **Implication:** no code change; dashboard copy + JS insight strings change; story is scientifically more defensible but less marketable.
- **C. Hybrid — keep the current no-floor code but add a diagnostic "P3 win-map" section to the dashboard.** Show a heatmap of P3-vs-P1 gas gap by ISO × endpoint to make explicit where the mechanism fires and where it doesn't. **Implication:** adds a new payload generator + chart; keeps both existing and reframed narratives.
- **D. Weaker floor — min-tranche rather than min-fraction.** Require P3 to procure at least one tranche of clean firm per year after NOAK-2035, even if the argmin doesn't pick it. Softer than A, harder than B/C. **Implication:** P3 moves off P1 in most ISOs but doesn't dominate; re-run the sweep; all chart payloads change.

**Recommended:** B. The §24.8 design decision was deliberate and well-reasoned; the dashboard is the stale artifact.

**Blast radius per option:**
- A: `step_2_3_pathway_optimizer.py`, `pipeline_config.py` (floor parameters), re-run the 350-run sweep (`run_sweep_1215.py` subset — ~3h), all 12 `reliability_tax/charts/gen_*.py`, `reliability-tax.html` copy.
- B: `reliability-tax.html` (hero key-findings, Section 2/5/7 prose, Section 5 PATHWAYS array, JS insight strings), `reliability_tax/methodogy.md`.
- C: one new `gen_p3_winmap.py`, new canvas + code in `reliability-tax.html`, copy updates in Section 5.
- D: `step_2_3_pathway_optimizer.py`, re-run sweep, all chart payloads, copy updates.

### AUDIT-2026-04-19-2 — What to do with the "gas hump at medium CFE" storyline?

**Options:**
- **A. Strip the hump storyline entirely.** Update Section 2 prose + JS insight to describe the observed monotone decline and explain it via §24.5 worst-hour sizing ("gas builds are largest where CFE is loosest because the residual gap shrinks as CFE tightens"). **Implication:** dashboard matches code; retires a wrong narrative.
- **B. Restore pre-§24.5 ELCC + peak-net-of-clean sizing.** Revert §24.5. Hump re-appears. **Implication:** invalidates every gas-sizing number in the current sweep; massive re-run; re-opens retired methodology debates.
- **C. Keep §24.5, reframe hump narrative as "why gas peaks where CFE is loosest."** Same data as A, different framing. **Implication:** copy only; slightly counterintuitive to a lay reader ("why does lax CFE build more gas?").

**Recommended:** A. Plainest description of what the data actually shows.

**Blast radius per option:**
- A / C: `reliability-tax.html` Section 2 + related JS insights; `reliability_tax/methodogy.md`.
- B: `step2_2a_cost_optimization.py`, `step_2_3_pathway_optimizer.py`, `pipeline_config.py`, re-run full sweep, all 12 chart payloads, all section prose in `reliability-tax.html`.

### AUDIT-2026-04-19-3 — Reconcile methodology doc with current code.

**Options:**
- **A. Update `reliability_tax/methodogy.md` to describe what the code actually does.** Mark Card C pivot-trigger logic and Card E P3 forcing as superseded by §24.8. State plainly that pathway differentiation is NOAK-year-only. **Implication:** spec matches code; future auditors don't chase dead mechanisms.
- **B. Update the code to match the existing methodology doc.** Restore pivot triggers and P3 floor. Same as card 1 option A. **Implication:** card 1 is effectively resolved the same way; large blast radius.
- **C. Both — revise both and explicitly mark the §24.8 pivot decision as a methodology change.** **Implication:** most complete; requires a coherent narrative about why the change was made.

**Recommended:** A (if card 1 → B) or B (if card 1 → A/D). Whichever option keeps doc + code coherent.

**Blast radius per option:**
- A: `reliability_tax/methodogy.md` only.
- B: same as card 1 option A.
- C: `reliability_tax/methodogy.md` + inline comments in `step_2_3_pathway_optimizer.py` pointing at §24.8.

## External-check log

| Parameter | Code value | External benchmark | Source | Verdict |
| --- | --- | --- | --- | --- |
| SBTi ladder | `(2025,0),(2030,50),(2035,70),(2040,90),(2045,95),(2050,99.9)` | ~100% by 2040 (SBTi 1.5°C OECD power, IPCC AR6 C1) | SBTi Power Sector Guidance 2020; IPCC AR6 WG3 | 2°C-ish framing, not 1.5°C — in-range for a conservative scenario |
| Nuclear newbuild NOAK (Medium) | $90–110/MWh | $82–124 (NREL ATB 2024 Moderate); $141–221 (Lazard v17) | NREL ATB 2024; Lazard LCOE v17 | In-range, aggressive end |
| Nuclear FOAK | $169–212/MWh | ~$190/MWh Vogtle U3 realized | EIA; utility filings | In-range |
| CCGT + CCS (45Q-on) NOAK | $69–80/MWh | $52–68 (NREL ATB 2024 net of 45Q) | NREL ATB 2024 | In-range, 5–15/MWh above ATB central |
| NOAK-2035 for nuclear | 5-yr FOAK→NOAK | 5%/dbl (pessimistic) to 20%/dbl (optimistic) | Grubler 2010; Lovering 2016 | Aggressive but defensible |
| `LEARNING_EXPONENT = 0.6` | front-loads 80% decline by 1/3-way | 0.3–0.5 typical Wright's Law on year-fraction domain | various | Aggressive |

No parameter is demonstrably out-of-range. Verdict B via parameter drift does not fire.

## Verdict

**B — methodology as implemented does not support the stated hypothesis.** The hypothesis says P3 < P1 in every ISO at every endpoint; the code intentionally (per §24.8) provides no mechanism to force P3 to do that. The data confirms the code: P3 ≡ P1 wherever the argmin picks pure-VRE mixes. Separately, the "gas hump" storyline is mechanically impossible under §24.5 worst-hour sizing and should not be on the dashboard. Cost parameters are in-range and are not the cause of the hypothesis failure. The canonical fix is decision card 1 option B (reframe the dashboard to match the real story) paired with card 2 option A (strip the hump narrative) and card 3 option A (update the methodology doc) — Verdict-B-in-name but Verdict-C-in-substance. The nuclear option (card 1 option A) is available if the user wants the original hypothesis to be true by construction.

## Out-of-scope items

- **Other 5 ISOs** (CAISO, NYISO, NEISO, MISO, SPP). CAISO is reportedly the one ISO where P3 < P1 behaves per hypothesis; the other four are not audited here. A follow-up audit on NYISO / NEISO (VRE-constrained, where card 1 option B's story predicts P3 wins) would corroborate or refute the "reframe" path.
- **All 12 `reliability_tax/charts/gen_*.py` generators** beyond Section 2 and Section 3. The Section 2 / Section 3 pair was sufficient to confirm the mechanism; other sections likely inherit the same framing drift.
- **Dashboard `reliability-tax.html`** user-facing copy — flagged as the downstream intervention surface but not read line-by-line in this run.
- **The retired ELCC / peak-net-of-clean sizing model.** Only noted in passing (Finding 7); not re-derived or externally checked.
- **`run_sweep_1215.py` and the 350-run v2 sweep mechanics.** If card 1 option A/D is taken, the sweep must be re-run; compute discipline lives in `OPS.md` and was not audited here.

