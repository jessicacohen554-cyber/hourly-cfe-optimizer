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
- **No further follow-up needed — Phase 5 classification follows.**

## What the code actually does (Phase 2 condensed trace)

`select_target_mix` (lines 1866–1951) runs each year 2025–2050. It computes the SBTi-interpolated CFE target for the year, snaps to the nearest available EF threshold, loads the EF parquet for that (iso, threshold), applies a pathway filter (P1/P1b block new nuclear via `_filter_pathway_1`; P2a/P2b/P3 see the full EF unchanged), then scores every candidate EF row.

With `INCLUDE_GAS_COST_IN_ARGMIN = True`, scoring is done by `score_ef_batch_with_gas` (lines 1790–1863). This calls `score_ef_batch` (pathway-aware: P3 gets NOAK-2035 for clean-firm LCOE; P1 gets NOAK default ~2040–2048) then adds a per-row gas penalty: `worst_hour_residual_per_row(iso, ef) × demand_mwh_grown / gaf × RA_margin × (capex_per_mw + fuel_per_mw)`. The argmin over this augmented score selects the winner.

In `solve_pathway` (lines 2295–2600+), the selected mix goes through `_derive_delta_vintages` (builds the annual ledger) and `size_required_gas_mw` (computes the realized worst-hour gas sizing for the year). The gas fleet is set to `max(new_gas_required_cumulative_mw)` over all years (peak-year snapshot, §24.6).

**Operation alignment vs. stated intent:**

| Operation | Line range | Verdict |
|---|---|---|
| `INCLUDE_GAS_COST_IN_ARGMIN` flag routing | 1932–1935 | Aligned |
| `score_ef_batch_with_gas` pathway-aware base cost | 1834 | Aligned — P3 gets cheaper clean-firm via NOAK-2035 |
| Gas penalty formula per row | 1841–1861 | Silent assumption — static `gas_raw_2025_mw` baseline does not account for gas-fleet retirement; penalty not discounted to NPV units |
| Argmin picks min(clean-energy-cost + gas-proxy) | 1940 | Misaligned — not the same as min(NPV); penalty miscalibrated; selects mixes worse for NPV in MISO ep80 (+78%) |
| P3 filter: full EF, no clean-firm floor | 1903–1910 | Aligned per §24.8 design decision |
| Gas fleet = peak-year snapshot, §24.6 | 2429–2434 | Silent assumption |

## Where the design breaks

**Intervention point 1 — Argmin objective mismatch** (Findings 3, 4, 5)

`score_ef_batch_with_gas` minimizes `(clean-energy-cost + gas-cost-proxy)` rather than NPV. The proxy uses a static `gas_raw_2025_mw` baseline (no retirement), a flat `capex + fuel` undiscounted rate, and current-year demand. In ISOs where clean-firm rows have lower residual than VRE rows, the penalty correctly steers toward clean firm. But when the selected clean-firm row requires a large VRE complement to meet CFE, the total cost explodes (MISO ep80 P3 NPV = $1,030B vs P1 = $580B). The penalty is also insufficient to flip ERCOT's argmin (P3 ≡ P1 at 9 of 10 endpoints). Location: `scripts/step_2_3_pathway_optimizer.py:1790–1863`, `scripts/pipeline_config.py:1435–1442`.

**Intervention point 2 — ERCOT gas identity persists** (Finding 1)

ERCOT's argmin does not change because ERCOT wind ($52–61/MWh + $9 TX) remains cheaper than nuclear NOAK-2035 ($90/MWh + TX) even after the gas penalty. ERCOT's clean-firm baseline (~4% of demand) means residual reduction from adding clean firm is small. P3 ≡ P1 at 9/10 endpoints; at ep85 P3 selects clean firm but builds 24% MORE gas than P1. Location: `scripts/step_2_3_pathway_optimizer.py:1903–1940`.

## Decision cards

### AUDIT-2026-04-20-1 — Recalibrate gas-penalty magnitude to fix perverse MISO/SPP outcomes

**Question:** `score_ef_batch_with_gas` uses an undiscounted, static-baseline gas proxy that over-steers MISO/SPP toward clean firm (worse NPV). Should the penalty be recalibrated?

**Options:**
- **A. Calibrate penalty to NPV-equivalent annualized cost.** Replace flat `capex_per_mw + fuel_per_mw` with an NPV-discounted annualized cost (7% real, 25-yr life, year-of-build timing). Keeps the mechanism, fixes magnitude. **Implication:** reduces MISO over-steering; preserves PJM/CAISO divergence.
- **B. Add a penalty cap (gas-penalty <= X% of base EF cost).** Pragmatic guard, tunable. **Implication:** reduces perverse outcomes; does not fix ERCOT.
- **C. Accept current behavior; reframe hypothesis as "P3 wins in VRE-constrained ISOs."** No code change. **Implication:** dashboard prose update only.

**Recommended:** A + C. Calibration is a ~15-line change; C needed for ERCOT/SPP framing regardless.
**Blast radius (A):** `score_ef_batch_with_gas` lines 1855–1861, re-run 350-run sweep, all chart payloads.

### AUDIT-2026-04-20-2 — Is ERCOT gas identity a bug or correct VRE-physics?

**Options:**
- **A. Accept ERCOT as VRE-wins exception; document it.** ERCOT wind beats nuclear at any reasonable clean-firm cost. Correct physics. **Implication:** dashboard explicitly calls out ERCOT. Recommended.
- **B. Increase gas penalty until ERCOT flips.** Arbitrary tuning; penalty loses physical meaning. Not recommended.
- **C. ERCOT-specific clean-firm floor for P3.** Overrides economics; weakens scientific credibility. Not recommended.

**Blast radius (A):** dashboard prose + SPEC.md only.

## External-check log (Phase 4)

| Parameter | Code value | External benchmark | Source | Verdict |
|---|---|---|---|---|
| Nuclear NOAK Medium ($/MWh) | $90–110 by ISO | NREL ATB 2024 Q2: ~$82–140; DOE Liftoff 2024 NOAK target: $66–$87; Lazard v17 2024: $141–$221 unsubsidized | NREL ATB 2024; DOE Advanced Nuclear Liftoff Sep 2024; Lazard LCOE+ June 2024 | In-range (NREL/DOE); below Lazard (Lazard excludes ITC/PTC) |
| Nuclear FOAK ($/MWh) | $169–212 by ISO | Vogtle 3/4 realized ~$190/MWh; DOE Liftoff FOAK range $150–$250 | EIA Vogtle filings; DOE 2024 | In-range |
| CCS+CCGT NOAK 45Q-on Medium ($/MWh) | $69–80 by ISO | NREL ATB 2024 CCGT+CCS Moderate gross ~$79–96; net of 45Q ($27.5) = $52–69 | NREL ATB 2024 fossil technologies | In-range; code $5–15/MWh above ATB net-of-45Q |
| NOAK-2035 for P3 | FOAK 2030 → NOAK 2035 | ATB 2024 Moderate puts nuclear NOAK ~2040; DOE Liftoff requires ~10 builds | NREL ATB 2024; DOE Liftoff 2024 | Edge of range — aggressive |
| Learning exponent 0.6 | `LEARNING_EXPONENT = 0.6` | 5–23%/doubling (Rubin et al. 2015); 0.6 on year-fraction front-loads ~80% by 1/3 of window | Rubin et al. (2015) IECM review | Aggressive shape; defensible |
| CCGT capex annualized ($/kW-yr) | $88–114 by ISO | Lazard v17 CCGT ~$76/MWh → ~$92/kW-yr; NREL ATB 2024 CCGT OCC $1,200/kW → ~$98/kW-yr | Lazard LCOE+ 2024; NREL ATB 2024 | In-range |
| WHOLESALE_PRICES in gas penalty | $25–42/MWh | EIA-930 2024 DA LMP weighted averages: ERCOT $27, PJM $34 | EIA-930; ISO market reports 2024 | In-range |
| RESOURCE_ADEQUACY_MARGIN | 15% | NERC standard 15–17%; PJM 14.7% installed reserve 2025 | NERC 2024 LTRA; PJM capacity market filings | In-range |
| Worst-hour percentile | 99.97th | LOLE 0.1 d/yr = 2.4 hrs/8760 = 99.97th pct. | NERC Resource Adequacy standard | In-range |

No parameter is out-of-range. Verdict B does not arise from parameter drift.

## Verdict

**B — the INCLUDE_GAS_COST_IN_ARGMIN fix partially worked but its objective function is misaligned with the stated design intent.** The fix is correctly wired and activated. It moved the argmin in VRE-constrained ISOs (PJM ep90+: 64% less gas; CAISO all endpoints: 69–75% less gas; NEISO: 91% less gas). However: (1) ERCOT remains P3 ≡ P1 at 9 of 10 endpoints because ERCOT wind economics dominate the gas penalty; (2) the penalty produces perverse outcomes in MISO and SPP — P3 builds MORE gas than P1 at 3 ISO×endpoint combinations (ERCOT ep85, SPP ep90, MISO ep95) and P3 NPV is 78% higher than P1 at MISO ep80. Root cause: the argmin minimizes `(EF-cost + gas-cost-proxy)` not NPV, and the proxy is miscalibrated (no discounting, static 2025 baseline, no fleet-retirement). Primary hypothesis holds in 4/7 ISOs at ep80/90/99. NPV secondary claim holds in 3/7 ISOs at ep90 (need >=5). Clean-firm share higher for P3 in 4–6/7 ISOs (mostly holds).

## Out-of-scope items

- **P2a / P2b pathways** — not audited; expected between P1 and P3 given NOAK 2040/2045.
- **P1a / P1b variants** — not audited; differences from P1 are in nuclear-cap treatment.
- **Chart payload generators** (`gen_section2_gas_hump.py`, `gen_section3_reliability_tax.py`, etc.) — downstream consumers; per-chart logic not re-read.
- **`reliability-tax.html` user-facing copy** — not read; needs updating per decision cards.
- **`run_sweep_1215.py` mechanics** — if card 1 option A is taken, 350-run sweep must be re-run per `OPS.md`.
