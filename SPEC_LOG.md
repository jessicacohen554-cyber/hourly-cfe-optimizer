# SPEC_LOG.md — Historical Decision Archive

> Rolling archive of retired `## Current Status` blocks from `SPEC.md`.
> Session-tactical notes (branch names, PIDs, completed-task checklists) preserved as-is.
> Canonical methodology/constants/cost-table decisions live in `SPEC.md` §N bodies — this file is reference, not source of truth.
> Order: reverse chronological (newest of the retired blocks first), matching the SPEC.md convention.

---

### Reliability Tax Dashboard Regeneration — LANDED (Apr 18, 2026)

**Branch:** `claude/regen-reliability-tax-v2-d0U5j` (3 commits; pushed; no PR opened per user default).

**What was accomplished this session:**
- **Phase A — chart payloads regenerated.** All 12 `reliability_tax/charts/gen_*.py` generators re-run against the banked 350-run v2 sweep. Key fixes: `data_loader.PATHWAYS` corrected to the real sweep set `["1", "1a", "2a", "2b", "3"]`; `gen_section3_reliability_tax.py` now reads `priced_vre_curtailment_usd` and `vre_storage_overbuild_capex_usd` directly from the solver (resolves the only "Action item (open)" flagged in the previous Current Status block); headline comparator swap from `"1a"` to `"1"` across 5 generators; `gen_act1_trilemma` reliability-tax definition updated to the gross Card M' framing.
- **Phase B — `dashboard/reliability-tax.html` walked section by section.** Pathway wiring switched from `[1a, 1b, 2a, 2b, 3]` → `[1, 1a, 2a, 2b, 3]`; Section 4 delta-insight JS rewritten to distinguish VRE-rich convergence (ERCOT, SPP: P1 ≡ P3) from VRE-constrained divergence (PJM, NYISO, NEISO: P3 saves 50–70%); Section 5 journey cards moved to ep90 (the §24.8 central-finding endpoint) with a new green-accent insight box listing the two §24.8 findings; Section 2 hump insight refreshed to §24.6 peak-year-snapshot semantics; Section 6 Sankey subtitle reframed for the single-consolidated-vintage schema; Section 7 cost-of-waiting cites the §24.8 NOAK-2035 window. ERCOT/SPP scarcity-pricing callout preserved verbatim.
- **Phase C — cross-page propagation.** `optimizer_methodology.html` Card K' / Card S updated, new §24.8 card appended, priced-curtailment formula rewritten to SPEC §24.7 form. `index.html` Act 4 scanned — no hardcoded reliability-tax numbers; methodology-neutral narrative compatible with v2, no edits required. `abatement_dashboard.html` has no reliability-tax references. `research_paper.html` does not exist in the repo.

**Headline numeric shifts (selected — see §24.9 for the full table):**
- `hook_reliability_tax.json` cross-ISO P1-vs-P3 cost delta: **17.7% → 29.6%** ($2.84T → $4.10T summed across feasible ISOs at ep95).
- `section3_reliability_tax.json` PJM ep90: **P1 $12.94/MWh vs P3 $2.47/MWh** (−81% on tax basis, −66.7% on undiscounted cost basis — matches §24.8 empirical table exactly).
- `section3_reliability_tax.json` ERCOT ep90: **P1 $11.36/MWh vs P3 $11.34/MWh** (Δ = $0.02/MWh — confirms the §24.8 VRE-rich convergence finding).
- `section5_stranding_sankey.json` total @ ep95: **$5,639B → $14.6B** (Card K' absolute framing correctly captures only CF<15%-for-2-years vintages; pre-regen value was the old Card K comparative VRE book-value delta).

**Schema invariants added to `reliability_tax/charts/data_loader.py` docstring.**
- Consumers MUST read `priced_vre_curtailment_usd` from the solver, not recompute.
- `stranding_metadata.peak_year` can be 2050 (no stranding) but never None.
- `terminal_ledger` top-level list now pre-seeds existing-fleet vintages at `cod_year = 2024, locked_lcoe = 0`; any future generator that iterates vintages must filter `cod_year >= 2025` for new-build attribution or explicitly bucket `resource == 'clean_firm_existing'` as existing fleet.

**Verification (all success gates passed).**
1. Every regenerated JSON parses (12 in `reliability_tax/charts/` + 7 mirrored in `dashboard/js/reliability-tax/`).
2. 5 canvases in `reliability-tax.html` (hump / abandonment / tax / sankey / wait) each wire to a `new Chart(` call; `getElementById` + IIFE scoping intact.
3. No orphan hardcoded numbers in HTML — all $/MWh, $B, GW values either (a) render from fetched payloads or (b) are methodology constants (NOAK years, CF<15% threshold, 99.97th percentile).
4. Mobile: chart-container-lg = 320px min-height (shared.css @768px); 44px min tap targets preserved.
5. `/security-review` clean on the diff (data/narrative refresh only; no new untrusted-input sinks, no deserialization, no crypto).

**Next open item (gated on user authorization):** optional 7-ISO × 3-endpoint L/H-cost sensitivity sweep. Current 350-run bank is Medium costs only. §24.8 finds the central story survives Medium; L/H corners would show sensitivity bands on the P1-vs-P3 delta in each ISO. Pre-Run Gate applies: ~6–9h wall clock; methodology already locked; no optimizer-code changes needed.

**Resume prompt for next session:** *"Pick up on the Reliability Tax workstream. Current state: Apr 18, 2026 dashboard regeneration is LANDED at §24.9. Next open item is the 7-ISO × 3-endpoint L/H-cost sensitivity sweep documented in §24.9 Scope Boundary — gated on user authorization. If user approves, follow the Pre-Run Gate in CLAUDE.md before launching."*

### Reliability Tax Sizing Bugs — Steps 1–3 Complete (Apr 17, 2026)

**Branch:** `claude/fix-step3-accounting-pkGI9` (rebased off `claude/fix-ercot-gas-capacity-i6cZD-v3Rza`, which merged Steps 1–3 v2).

**Context.** User audit of `analysis/reliability-tax/data/ERCOT/pathway1_ep90.json` surfaced a 450+ GW new-build gas fleet vs. 69 GW of actual ERCOT gas today and 197 GW 2050 peak demand. Also: Pathways 1 and 3 producing identical results across every endpoint.

**Planned 3-step fix (one commit per step):**
1. **Step 1 (LANDED — commit 6081324)** — Peak-year gas-fleet snapshot + drop cross-endpoint seeding + demand² scaling fix. See §24.6.
2. **Step 2 (LANDED — commit 9619b2c)** — VRE stranding priced as `Σ_years surplus_frac[r,y] × demand_mwh[y] × vintage_lcoe[r,y]`, with hybrid surplus priced at the underlying VRE's ledger key. See §24.7.
3. **Step 3 v1 (LANDED — commit acf5bd7, then superseded)** — Shipped with per-pathway NOAK override AND a clean-firm floor `clean_firm ≥ 0.30 × threshold_pct` on P3. User rejected the floor as hard-wiring the finding and smothering regional heterogeneity.
4. **Step 3 v2 (LANDED — commit 8f87a56)** — Floor removed. P2a/P2b/P3 all see the full EF from year 1. Pathway differentiation lives solely in `pc.NOAK_YEAR_BY_PATHWAY` (P3=2035, P2b=2040, P2a=2045). Cost-optimal mix surfaces organically under each pathway's accelerated Wright's-Law curve. See §24.8.
5. **Step 3 v2 accounting fix (LANDED — commit c576cec)** — Pre-seeded `VintageLedger` with zero-LCOE existing-fleet vintages (cod_year = 2024) per `GRID_MIX_SHARES`, plus clean-firm `target_twh` now uses BASE demand (not year-t grown demand) to match the EF-row encoding semantics. Result: when ERCOT P1 and P3 pick identical every-year mixes, costs match within 0.23 % (ep60) — down from the pre-fix +14.6 %. PJM P3 ep90 diverges into 79 % clean firm at −66.7 % vs P1 (sharper than the pre-fix −53.5 % because P1 no longer benefits from phantom existing-fleet scaling). See §24.8.

**Step 1 empirical results (ERCOT P1 Medium growth, medium costs):**
- ep60: 53 GW new gas — peak-year 2050, 0 stranded.
- ep90: 111 GW new — peak-year 2050, 0 stranded.
- ep99.9: 73 GW new — peak-year 2039, 10 GW stranded by 2050, $6.7 B stranded capex.

**Step 2 empirical (ERCOT P1 Medium growth, medium costs, POST accounting fix):**
- ep60: $0.00 B priced VRE curtailment (unchanged).
- ep99.9: $182.50 B cumulative priced curtailment (down from pre-fix $193.58 B — existing-fleet VRE vintages now at zero LCOE dilute the TWh-weighted vintage LCOE per §24.7's "existing fleet predating the ledger not charged a stranding tax" principle). Still non-zero, still captures real new-build VRE stranding.

**Step 3 v2 + accounting fix empirical (ERCOT + PJM Medium growth, medium costs, commit c576cec):**
```
ISO    ep    P1 $B    P3 $B    Δ         P3 mix
ERCOT  60    525.1    526.3    +0.23 %   cf=9  sol=14 wind=42  (≡ P1 every year)
ERCOT  90    1514.5   1460.1   −3.59 %   cf=0  sol=0  wind=90  (endpoint ≡ P1; intermediate years diverge)
ERCOT  99.9  3146.2   3137.0   −0.29 %   cf=10 sol=0  wind=41  (≈ P1; cf=9 vs 10 at endpoint)
PJM    90    7167.8   2390.3   −66.65 %  cf=79 sol=0  wind=10  (strong divergence, sharper than pre-fix −53.5 %)
```
- **ERCOT ep60 passes the <1 % gate**: every-year mix is identical and only a $1.2 B residual from a 1.9 TWh existing-vs-target-base rounding delta at year 1.
- **ERCOT ep90** endpoint mix is identical (0/0/90) but intermediate years 2032–2039 diverge: P3's NOAK-2035 override makes clean-firm-inclusive EF rows cheaper to serve the SBTi ramp, so P3 picks uprate + CCS at year 2032 while P1 stays pure-VRE. The −3.59 % delta is intended methodology (regional heterogeneity under accelerated NOAK), not an accounting artifact.
- **ERCOT ep99.9** mix differs slightly at endpoint (P1 cf=9, P3 cf=10) — cost delta is small and reflects the one-percentage-point EF-row swap.
- **PJM P3 ep90** sharpens from −53.5 % to −66.65 %: the previous accounting bug inflated P1's cost via phantom existing-fleet scaling; with that removed, the real VRE-curtailment-vs-clean-firm tradeoff comes through cleanly. P3 builds 79 % clean firm at $2.39 T vs P1's 29 % at $7.17 T.

**Resume instructions for next session (350-run sweep + dashboard regeneration):**
1. Read §24.6–§24.8 if context is lost.
2. Pre-seed / base-year accounting fix is LANDED (commit c576cec). Re-running the reproducer (`python3 scripts/step_2_3_pathway_optimizer.py --iso ERCOT --pathway 3 --endpoint 0.60 …`) now matches P1 within 0.23 %.
3. **350-run sweep is LIVE** as of 2026-04-17 ~23:43 UTC. Launched via `nohup setsid bash scripts/_sweep_all_isos_step3fix.sh > logs/rt_step3fix_sweep_wrapper.log 2>&1 &`. Streams ISO_START/ISO_DONE/ISO_FAIL/SWEEP_DONE events to `logs/rt_step3fix_sweep.log`. Check `ps -ef | grep run_pathway_sweep` before launching a second instance. Old pre-fix data archived to `analysis/reliability-tax/data-archive-pre-step3-fix-2026-04-17/`.
   - If sweep completes while session is idle, pick up at step 4.
   - If sweep died mid-run, inspect log, troubleshoot, rerun with `--force` if needed. MANIFEST-based resume should skip already-completed runs.
4. Bank results: `git add analysis/reliability-tax/data/ data/step3-dispatch/ analysis/reliability-tax/data-archive-pre-step3-fix-2026-04-17/` after the sweep writes MANIFEST + all 350 per-run JSONs and any expanded dispatch-cache archetypes, then commit + push.
5. Regenerate dashboard JS data via `reliability_tax/charts/*.py` readers. **Action item (open):** `gen_section3_reliability_tax.py::_compute_priced_curtailment_usd` currently computes curtailment locally from a mix-excess proxy rather than consuming the solver's new `components_usd.priced_vre_curtailment_usd` field (§24.7). Either (a) patch the chart to use the solver value directly, or (b) leave the proxy in place and verify the solver value flows through to a separate downstream artifact (e.g. via `data_loader.get_pathway_cost` → `annual_cost[].priced_vre_curtailment_usd_this_year`). Verify the final stacked bars show non-zero VRE-curtailment bars for VRE-heavy pathways (PJM P1 ep90 ≈ $275 B cumulative, ERCOT P1 ep99.9 ≈ $182 B).

### Reliability Tax v2 Dashboard Rewrite (Landed — Apr 17, 2026)

**What was accomplished this session (A1–A4 + B1 + B2 + C):**
- **A1–A4 — `dashboard/reliability-tax.html` rewritten against SPEC §24.4 v2 methodology (Cards M'/F'/K'/U/S).** 9 sections wired against the regenerated v2 JSON payloads for all 7 ISOs (CAISO, ERCOT, PJM, NYISO, NEISO, MISO, SPP) across 10 CFE endpoints (60/70/75/80/85/90/95/97.5/99/99.9): (0) hero title + lead + definition card, (1) duck-curve schematic + residual-gap explainer, (2) THE HUMP headline chart (new-gas MW vs achieved CFE), (3) abandonment framing, (4) reliability-tax decomposition (stacked components per the §24.4 locked formula), (5) four-journey pathway cards, (6) stranding Sankey, (7) cost of waiting, (8) 175-row summary table + closing QA sweep.
- **B1 — `optimizer_methodology.html`** updated to align with the v2 cards (capacity-market netting, developer-ROI, and comparative-to-P3 stranding framing removed; absolute gross-cost framing + CF<15% stranding test substituted).
- **B2 — `index.html` Act 4** reframed so the stranded-asset narrative matches the v2 methodology rather than the pre-rewrite comparative story. `abatement_dashboard.html` touched for the same cross-page language consistency.
- **ERCOT/SPP callout added** throughout: ratepayers in energy-only ISOs pay the reliability tax via scarcity-priced energy rather than capacity payments — no more "awkward exception" phrasing.
- **C — SPEC.md updated**: §24.4 Status subsection appended (this session); Current Status rewritten (this block).
- **Branch:** `claude/update-spec-reliability-tax-jd87x` — pushed to origin. Upstream v2 dashboard rewrite PRs (#3124–#3129) already merged to `master` before this session.

**Next steps:** None required for this workstream. If a `research_paper.html` is authored in the future, its reliability-tax sections should follow the v2 cards (see SPEC §24.4 Status note).

### Worst-Hour Gas Sizing Rerun (In Progress — Apr 16, 2026)

**Branch:** `claude/worst-hour-gas-sizing-JH0wH` (commits `02b04a9`, `29137f1`)

**Code + SPEC:** Complete. Worst-hour (99.97th percentile, margin-on-demand) gas sizing is locked in SPEC §24.5 and implemented in `scripts/step2_2a_cost_optimization.py` (scalar + batch paths), `scripts/step_2_3_pathway_optimizer.py`, and `scripts/step3a_build_dispatch_cache.py`. Validation (`scripts/tmp_validate_worst_hour_sizing.py`) passes for 10 mixes × 3 ISOs with `WH ≥ ELCC` strict invariant.

**Step-2.2A rerun:** Launched `2026-04-16 ~23:57 UTC`, **died silently around 23:58:30** (no OOM in dmesg, log ended cleanly mid-processing — likely session-cleanup SIGHUP'd the orphaned child). **Re-launched `2026-04-17 ~00:10 UTC` at PID 1656** via `nohup setsid python3 ... &` for full session detachment. All 7 ISOs, baseline track, `--workers 1`, logging to `logs/step2_2a_worst_hour_rerun.log`. Totals across ISOs: 47,953,153 raw EF mixes (CAISO 21.2M + ERCOT 8.7M + PJM 3.1M + NYISO 2.1M + NEISO 5.2M + MISO 6.2M + SPP 1.5M). Expected wall-clock 6–12 hours.

**What's left:**
- [ ] Step-2.2A baseline track completes for all 7 ISOs → `data/step2.2-cost/step_2_2a_CO_{ISO}.parquet`.
- [ ] Commit + push the new parquets alongside the expanded dispatch cache (`data/step3-dispatch/*`).
- [ ] Step-2.2B (`step2_2b_track_nb_ctr.py`) — newbuild + cost-to-replace tracks. Runs fast on Step-2.2A output; launch after baseline lands.
- [ ] Sanity audits on the new parquets: `gas_gas_backup_needed_mw` distributions, clean-firm-heavy vs VRE-heavy comparison, ERCOT baseline calibration.
- [ ] DOWNSTREAM (not this session per task brief): Step 3a cache rebuild, Step 4-7 analytics, reliability-tax 350-run sweep, dashboard JSONs.

**Resume instructions for next session:**
1. Check `ps -ef | grep step2_2a_cost_optimization` — if PID 1656 (or descendant) still running, do NOT launch a second instance. Tail `logs/step2_2a_worst_hour_rerun.log` to confirm progress. Relaunch if needed via `nohup setsid python3 scripts/step2_2a_cost_optimization.py --track baseline --workers 1 > logs/step2_2a_worst_hour_rerun.log 2>&1 &` (the nohup+setsid is load-bearing to survive session boundaries).
2. If completed, inspect `data/step2.2-cost/*.parquet` timestamps vs launch time. New parquets should have `mtime > 23:57 UTC`. Confirm all 7 ISOs present.
3. Run `python3 scripts/tmp_validate_worst_hour_sizing.py` — should still pass (no code changed).
4. If optimizer crashed: read last 500 lines of the log, identify the failure, and restart per `CLAUDE.md` Optimizer Run Discipline ("automatically troubleshoot, debug, and retry").
5. Once Step-2.2A is confirmed complete, launch Step-2.2B: `python3 scripts/step2_2b_track_nb_ctr.py`. That's the new-build + cost-to-replace track.
6. Commit results: `git add data/step2.2-cost/ data/step3-dispatch/ && git commit -m "Bank worst-hour step-2.2 results"`.

**Open questions:** None — all design decisions locked in SPEC §24.5.

### Phantom Clean Audit + Year-Scaled Floor Fix (Added — Apr 16, 2026)

**Branch:** `claude/fix-phantom-clean-generation-rPNpL`

**Audit question**: Did the EF filter or cost optimization phantom-credit existing clean generation that doesn't actually exist in demand-growth years?

**Audit findings (Step 2.1 + Step 2.2)**:
1. **Step 2.1 (`step2_1_efficient_frontier.py:500-558`)** — `resource_cap_filter` applies only solar cap (≤100%), total cap (≤350%), hydro cap (≤ existing), and hybrid family caps. **No clean-firm/solar/wind floor.** Clean.
2. **Step 2.2a pricing (`step2_2a_cost_optimization.py:247, 255, 285, 294, 642`)** — existing resources correctly scaled by `existing_scale = 1.0 / gf` in growth years, so credited existing MWh stays pinned to 2025 fleet absolute TWh. **No phantom MWh credit.**
3. **Step 2.2b (`step2_2b_track_nb_ctr.py:828-845`)** — same `existing_scale` pattern in the DG loop. Clean.

**Real issue identified**: The `apply_existing_clean_floor` filter (`step2_2a_cost_optimization.py:491-516`, called line 2100) uses **2025 base-year share as % of year-t demand** for every year. In PJM 2050 at 2.5% growth, this requires `clean_firm ≥ 32.1%` of ~1,725 TWh = 554 TWh, even though existing nuclear is only 270 TWh. The filter thus forces 284 TWh of new clean firm — artificially expensive overbuild — to hold a proportional 2025 share that the fixed fleet can't actually support.

**Regression from prior decision**: SPEC.md line 2003 (Feb 24, 2026) explicitly defined Track 1 ECF as *"Existing clean TWh remain constant across all demand growth scenarios (absolute TWh steady; share of total generation declines over time as demand grows)"*. SPEC.md line 1970–1972 (Feb 20, 2026) also documented a decision to **remove the Phase 0 floor filter** ("kept as dead code for reference") to support below-floor mix recovery.

**Regression timeline**:
- **Feb 20, 2026**: Phase 0 floor filter marked for removal; below-floor mixes recovered via temp script and merged into `pfs_post_ef.parquet`.
- **Feb 24, 2026**: Track 1 ECF formally defined with *declining share under demand growth*.
- **Apr 7, 2026 (commit `ab8de61`, PR #3059)**: `step2_2a_cost_optimization.py` created from scratch as part of a ~50,000-line pipeline consolidation merge. The new file reintroduced `apply_existing_clean_floor` using fixed base-year shares, contradicting both Feb 20 (no filter) and Feb 24 (declining share) decisions.
- **Apr 7 → Apr 16**: No commits touched `step2_2a_cost_optimization.py`. Every Track 1 baseline and DG-sweep parquet produced in this window carries the regression.
- **Root cause**: cross-session context loss — the Apr 7 author did not re-read SPEC.md's Feb 20/24 floor-filter decisions before rewriting the cost optimizer.

**Decision (Apr 16, 2026)**:
- **Step 2.1 EF**: **NO floor filter** (already true in code). EF must remain unfiltered so Track 2 (New Build greenfield) and Track 3 (Cost-to-Replace) can analyze replacement scenarios without being constrained by existing-clean floors.
- **Step 2.2a Track 1 (ECF baseline)**: Floor filter uses **year-scaled existing share**, proportional to demand growth. Each evaluation year's floor = `GRID_MIX_SHARES[iso][resource] × (1.0 / gf)` where `gf = (1 + growth_rate)^(year - 2025)` — i.e., `existing_TWh / year_t_demand × 100%`.
  - Base year (2025, gf=1.0): floor = 32.1% clean_firm (PJM). Unchanged — equals existing-fleet share.
  - 2040 at 2.5% growth (gf=1.491): floor = 21.5%.
  - 2050 at 2.5% growth (gf=1.854): floor = 17.3%.
  - Applies to `clean_firm`, `solar`, `wind`, `hydro`.
- **Tracks 2/3** (already pull from unfiltered `pfs[iso]` at line 2313+): no change, continue to see the full EF.

**Implementation**:
- Modify `apply_existing_clean_floor(arrays, iso, existing_scale=1.0)` to accept year-scaling param.
- Phase 1 base-year call uses `existing_scale=1.0` (unchanged behavior for 2025).
- Phase 2 DG sweep: re-filter archetypes per-year using scaled floor (or re-select archetypes per-year from a permissively-prefiltered pool so low-clean mixes valid in later years aren't excluded by Phase 1's strict base-year filter).
- `step2_2b_track_nb_ctr.py`: pricing already uses `existing_scale = 1/gf` correctly. No filter change needed unless a pre-filter is also present (to verify).

**Rerun scope**:
- ✅ **No Step 1 rerun** (PFS unchanged).
- ✅ **No Step 2.1 rerun** (EF unchanged; filter lives in 2.2a).
- 🔄 **Step 2.2a** — regenerate `data/step2.2-cost/step_2_2a_CO_*.parquet` (~30 min).
- 🔄 **Step 2.2b** — regenerate track parquets (~10 min).
- 🔄 **Steps 3–7** — cascade regeneration (dispatch cache, analysis, scenarios, SMARTargets, dashboard; ~2 hr total).

**Files modified**: `scripts/step2_2a_cost_optimization.py`, `scripts/step2_2b_track_nb_ctr.py` (if applicable).

---

### RPS Floor Fix: Renewable-Only Compliance for Non-CES ISOs (Added — Mar 15, 2026)

**Branch:** `claude/analyze-pjm-emissions-qHlEF`

**Problem**: PJM's P50 reference trajectory showed emissions *rising* from 267 Mt (2023) to 562 Mt (2035) — counterintuitive given existing RPS mandates in MD, NJ, IL, VA, DE, PA. Root cause: the RPS floor enforcement compared against **total clean%** (including 32.1% nuclear) instead of **RPS-eligible renewables only** (8.5% — solar 2.9%, wind 3.8%, hydro 1.8%). Since 40.6% > 22% floor, the RPS mandate never bound.

**Key insight**: Most PJM-state RPS mandates are **renewable-only** — they exclude existing nuclear. MD, NJ, IL, PA, VA, DE all have renewable portfolio standards (not clean energy standards). Nuclear Zero Emission Credits (ZECs) exist in some states (IL, NJ, NY) but are separate programs from RPS compliance.

**Fix (step6_1_smartargets.py)**:
1. **CES vs. RPS distinction for deployment**: CES ISOs (CAISO, NYISO, NEISO) count nuclear toward compliance floor; non-CES ISOs (PJM, MISO, SPP, ERCOT) use only `REC_ELIGIBLE` resources (solar, wind, offshore_wind, hydro, geothermal).
2. **`rps_eligible_pct` state tracking**: New field tracking renewables-only percentage through the simulation. Updated whenever market-driven or mandated deployment adds VRE capacity. Has its own TWh ratchet floor (absolute RPS-eligible TWh never decreases even as demand grows).
3. **ACP compliance fallback**: When queue constraints prevent physical renewable build, Alternative Compliance Payments ($45/MWh for PJM) cover the shortfall. ACP does NOT add clean MWh — fossil generation persists for that portion.
4. **ACP recycling (capped)**: ACP revenue funds future renewable development. Converted to bonus queue GW for next period at 65% fund efficiency and $1.2B/GW cost. **Capped at 20% of base queue budget** to prevent unrealistic acceleration.

**New output fields**: `rps_eligible_pct`, `rps_shortfall_pct`, `acp_cost_million`, `cumulative_acp_million`.

**Expected impact**:
- **PJM** (major): RPS-eligible baseline 8.5% vs. floors 22%→40%→55% — massive forced renewable build, queue-constrained + ACP. Emissions should decline meaningfully.
- **MISO** (moderate): RPS-eligible 18.2%, floor binding by 2030 (~7pp gap).
- **SPP** (none): RPS-eligible 41.8% (wind fleet exceeds all floors).
- **CAISO/NYISO/NEISO** (unchanged): CES ISOs — nuclear counts.

**Files modified**:
- `scripts/step6_1_smartargets.py` — RPS floor logic, ACP fallback, rps_eligible_pct tracking, result fields

---

### NYISO Hydro-Québec Imports + Forecast Validation Update (Added — Mar 15, 2026)

**Branch:** `claude/hydro-quebec-imports-validation-UP70G`

**Part A — External Clean Imports (NYISO Hydro-Québec)**

**Problem**: NYISO imports ~25 TWh/yr of clean hydropower from Hydro-Québec via HVDC ties. Our model treats ISOs as independent grids, ignoring these cross-border flows. This partly explains the NYISO clean penetration lag (our ~40% vs benchmarks' 58–62% at 2030).

**Solution**: Fixed-constant external clean import model:
- `pipeline_config.py`: `EXTERNAL_CLEAN_IMPORTS_TWH = {'NYISO': 25, ...}` (others 0 for now)
- `step6_1_smartargets.py`: Adds import TWh as percentage-point boost to clean_pct
  - Initial state: `ext_import_pct = ext_import_twh / REGIONAL_DEMAND_TWH[iso] * 100`
  - Per-year adjustment: recalculates ext_import_pct as demand grows (fixed TWh / growing demand = shrinking %)
  - 2023 baseline: adds ext_import_pct to eGRID baseline clean_pct
- NYISO: 25 TWh / 160 TWh = ~15.6 pp boost to clean penetration

**Known simplification**: Real HQ imports vary with market conditions, hydrology, and contract terms. 25 TWh is a representative annual average from NYISO Gold Book 2024 and HQ annual reports.

**Part B — Forecast Validation Narrative Update**

Updated `dashboard/forecast_validation.html`:
- Directional alignment stat: 86% → 60% (reflects actual computed value from validation data)
- Added NYISO HQ imports as 3rd key finding in executive summary
- Updated NYISO divergence section with dual root causes (HQ imports + state mandates)
- Fixed demand growth reference: was "0.5%/1.2%/2.0%" uniform → now shows actual ISO-specific rates (ERCOT 3.5%, PJM 2.4%, etc.)
- Updated Known Limitations table: coal retirement (now hybrid), policy modeling (now has IRA overlay), demand growth (now ISO-specific)
- Added "Model Improvements" section documenting 5 changes: HQ imports, IRA overlay, coal retirements, ISO-specific demand growth, LMP decomposition

---

### IRA Policy Overlay + LMP Decomposition (Added — Mar 15, 2026)

**Branch:** `claude/ira-policy-lmp-decomposition-55AOX`

**Part A — IRA Policy Overlay for Reference Sweep**

**Problem**: Reference sweep is market-only (no policy drivers), causing large emissions gaps vs. every published benchmark. The "conditions" dimension (Facilitating/Challenging) only controls learning rates, queue caps, and DAC — NOT policy.

**Solution**: New `ira_policy` sweep dimension ('on'/'off') that toggles three policy layers:

1. **IRA PTC/ITC** — Reduces effective LCOE when `ira_policy='on'`:
   - Solar: -$26/MWh (§45Y PTC)
   - Wind: -$26/MWh (§45Y PTC)
   - Offshore wind: -$26/MWh (§45Y PTC)
   - Battery/battery8: -30% ITC (§48E)
   - Nuclear new-build: Already has 45Y (-$26/MWh) — no double-count
   - CCS: Already has 45Q toggle — no change

2. **RGGI Carbon Pricing** — ISO-specific fossil cost adder via merit-order stack:
   - Applies to: NYISO, NEISO, PJM (RGGI member states)
   - Levels: Low $3/ton, Medium $5.50/ton, High $14/ton (mapped from fuel_level)
   - Non-RGGI ISOs: $0/ton (removes incorrect uniform CO2 pricing)
   - **Critical**: RGGI acts as a FLOOR, not additive to national carbon price. `effective = max(national, rggi)`. Same as CBAM — overlapping carbon markets don't stack.

3. **State RPS Deployment Floors** — Mandated minimum clean% by year:
   - CAISO: 44%→60%→75%→90%→100% (2025→2030→2035→2040→2045) — **CES** (nuclear counts)
   - NYISO: 35%→70%→80%→90%→100% — **CES** (nuclear counts via CLCPA Tier 3 ZECs)
   - PJM: 20%→35%→45%→55%→65% (NJ/PA/VA weighted) — **RPS** (renewables only, nuclear excluded)
   - NEISO: 35%→50%→65%→80%→100% (MA net-zero weighted) — **CES** (nuclear counts via CT CCEF)
   - ERCOT: None
   - MISO: 15%→20%→25%→30%→40% — **RPS** (renewables only)
   - SPP: 10%→15%→20%→25%→35% — **RPS** (renewables only)
   - **Critical**: CES ISOs (CAISO, NYISO, NEISO) compare floor against total clean%. Non-CES ISOs (PJM, MISO, SPP) compare against RPS-eligible% only (solar, wind, offshore_wind, hydro, geothermal). PJM has 32.1% nuclear but only 8.5% RPS-eligible → floor is deeply binding.

**Sweep impact**: Doubles scenarios from 270 → 540 (×2 policy modes). LMP cache partially shared (ira_policy='off' uses same cache as before; 'on' adds carbon_price_override to cache key for RGGI ISOs).

**Scenario ID format**: Appends `_P` (policy on) or `_M` (market only). Example: `S_F_M_all_med_M_M_P`.

**Files modified**:
- `scripts/pipeline_config.py` — Added `IRA_PTC_SOLAR`, `IRA_PTC_WIND`, `IRA_ITC_BATTERY_PCT`, `IRA_PTC_45U_NUCLEAR`, `RGGI_ISOS`, `RGGI_PRICE_PER_TON`, `STATE_RPS_FLOORS`, `get_rps_floor()`
- `scripts/lmp_engine.py` — Added `carbon_price_override` param to `compute_marginal_costs()` and `build_merit_order_stack()`
- `scripts/step6_1_smartargets.py` — Added `IRA_POLICY_MODES`, expanded `build_sweep_scenarios()`, wired `ira_policy` through `get_resource_lcoe()` → `compute_zone_cost()` → `run_market_simulation()`, added RPS floor enforcement, RGGI carbon pricing passthrough to LMP engine

**Backward compatibility**: `ira_policy='off'` (default) preserves existing market-only behavior for all callers.

---

**Part B — LMP Decomposition (Analytics Only)**

**Problem**: We compare our all-in LMP (with scarcity + congestion + must-run depression) against NREL Cambium's short-run marginal cost (SRMC), inflating apparent divergence.

**Solution**: Component tracking in `compute_hourly_lmp_vectorized()`:
- `merit_base[h]` = base marginal cost from stack dispatch (heat-rate-ramped, no scarcity)
- `scarcity[h]` = reserve scarcity adder + full scarcity pricing
- `dq_adder[h]` = residual (demand-quantile layers, must-run depression, clean surplus, NEISO winter gas)
- Identity: `merit_base[h] + scarcity[h] + dq_adder[h] == hourly_lmp[h]`

**Implementation**: `track_components=False` default param. When True, returns 3-tuple `(hourly_lmp, hourly_mu, components_dict)`. Existing 2-tuple callers unaffected.

**Component averages** added to `compute_lmp_stats()` when `components` provided: `merit_base_avg`, `scarcity_avg`, `dq_adder_avg`, `merit_base_p50`, `scarcity_hours_nonzero`.

**Use case**: Compare `merit_base_avg` against Cambium SRMC for apples-to-apples validation.

**Files modified**:
- `scripts/lmp_engine.py` — `compute_hourly_lmp_vectorized()` track_components param + component arrays, `compute_lmp_stats()` components param

**Decision**: LCOE tables (NREL ATB 2024) represent pre-incentive costs. IRA applied as explicit overlay, not baked into base tables. This enables on/off comparison.

---

### Announced Coal Retirements + Transmission Constraint Toggle (Added — Mar 15, 2026)

**Branch:** `claude/coal-retirements-transmission-kN6hR`

**Part A — Announced Coal Retirements (Hybrid Model)**

**Problem**: Threshold-based coal retirement (coal exits at 70% clean) doesn't match real-world retirement schedules. PJM emissions at 2035 show 356 Mt vs EIA's 200 Mt — the cliff model over-retains coal in early years.

**Solution**: Hybrid model — near-term (through 2035) uses EIA 860 announced retirement schedule; long-term (2035+) falls back to threshold-based retirement. Smooth transition via linear interpolation between milestone years.

**Data** (cumulative GW retired by year, EIA 860 Monthly Dec 2024 + EPA 111(d) projections):
```python
ANNOUNCED_COAL_RETIREMENTS = {
    'PJM':   {2025: 3.0, 2028: 10.0, 2030: 18.0, 2032: 25.0, 2035: 30.0},
    'MISO':  {2025: 2.0, 2028: 6.0,  2030: 10.0, 2032: 15.0, 2035: 18.0},
    'ERCOT': {2025: 1.5, 2028: 3.0,  2030: 5.0,  2032: 5.0,  2035: 5.0},
    'SPP':   {2025: 1.0, 2028: 3.0,  2030: 5.0,  2032: 8.0,  2035: 9.0},
    'CAISO': {},  'NYISO': {},  'NEISO': {},  # No/negligible coal
}
```

**Mechanism**:
1. `get_announced_coal_retired_gw(iso, year)` — linearly interpolates between milestone years
2. `compute_fossil_retirement()` — when `year` param provided and ≤2035, reduces `coal_cap` (TWh) by `retired_gw × 4.38` (50% CF) before merit-order displacement
3. `build_merit_order_stack()` — when `year` param provided, reduces coal share in capacity mix before threshold-based retirement

**Backward compatibility**: `year=None` (default) preserves existing threshold-only behavior for all callers not yet passing year.

**Files modified**:
- `scripts/pipeline_config.py` — Added `ANNOUNCED_COAL_RETIREMENTS` dict + `get_announced_coal_retired_gw()` helper
- `scripts/dispatch_utils.py` — Added `year` param to `compute_fossil_retirement()`, reduces coal_cap by announced retirements
- `scripts/lmp_engine.py` — Added `year` param to `build_merit_order_stack()`, reduces coal capacity shares by announced retirements

**Validation target**: PJM 2035 emissions should drop from ~356 Mt toward EIA AEO ~200 Mt.

---

**Part B — Transmission Constraint Toggle**

**Problem**: Clean energy deployment modeled with unconstrained interconnection queue throughput. Princeton REPEAT shows clean penetration drops 75%→61% at 2030 when transmission growth limited to 1%/yr (vs required 2.3%/yr).

**Solution**: Lightweight queue throughput factor applied to `QUEUE_CAP_GW` in SMARTargets simulation.

**Parameters**:
```python
TRANSMISSION_CONSTRAINT = {
    'unconstrained': 1.0,  # Full queue throughput (default)
    'limited': 0.7,        # Moderate constraints (~1.5%/yr TX growth)
    'constrained': 0.5,    # Severe constraints (~1%/yr TX growth, REPEAT low)
}
```

**Mechanism**: Applied via `conditions.get('transmission_constraint', 'unconstrained')` in `run_market_simulation()`. Multiplies `queue_budget_gw` before period scaling. Default `'unconstrained'` preserves backward compatibility.

**Files modified**:
- `scripts/pipeline_config.py` — Added `TRANSMISSION_CONSTRAINT` dict
- `scripts/step6_1_smartargets.py` — Imported constant, applied factor at both queue_cap_gw usage points

---

### Regulatory Feedback — LMP Dampening at High Clean Penetration (Added — Mar 15, 2026)

**Branch:** `claude/add-regulatory-feedback-n8MJ1`

**Problem**: SMARTargets forward sweep projects unrealistic LMP at high clean penetration. ERCOT P50 LMP reaches $172/MWh by 2050 (vs EIA $35, Cambium $10). PJM P50 hits $111/MWh by 2050 (vs EIA $43). The divergence comes from scarcity hours pulling up averages — plausible under static ORDC/energy-only pricing, but unrealistic because regulators would intervene with capacity mechanisms or price reforms well before averages sustained at those levels.

**Real-world precedent**: ERCOT reformed ORDC after 2021 freeze. PJM reformed RPM parameters after price spikes. MISO raised VOLL to $10K in 2025.

**Mechanism**:
1. Track rolling 3-period average LMP per ISO across simulation years
2. If rolling avg exceeds `REGULATORY_FEEDBACK_THRESHOLD_MULTIPLIER` × 2023 baseline LMP → trigger reform
3. Dampen scarcity hours (top 5% of hourly prices) by a factor proportional to the excess
4. Energy-only ISOs (ERCOT, SPP) get 30% stronger damping (capacity payment introduction is a bigger structural shift)
5. Hard cap: if avg LMP still exceeds `REGULATORY_FEEDBACK_CAP_MULTIPLIER` × baseline, scale all positive hours proportionally

**Parameters** (`pipeline_config.py`):
```python
REGULATORY_FEEDBACK_THRESHOLD_MULTIPLIER = 2.0  # Trigger when rolling avg > 2× baseline
REGULATORY_FEEDBACK_DAMPING = 0.5               # Damping strength (0=none, 1=full clamp)
REGULATORY_FEEDBACK_CAP_MULTIPLIER = 1.75       # Hard cap: effective avg ≤ 1.75× baseline
ENERGY_ONLY_ISOS = {'ERCOT', 'SPP'}
CAPACITY_MARKET_ISOS = {'PJM', 'NYISO', 'NEISO', 'MISO', 'CAISO'}
```

**What it does NOT change**: Dispatch physics, hourly generation profiles, merit-order stacks. Only price signals that feed back into deployment economics within the SMARTargets simulation loop.

**Files modified**:
- `scripts/pipeline_config.py` — Added `REGULATORY_FEEDBACK_*` constants + `ENERGY_ONLY_ISOS` / `CAPACITY_MARKET_ISOS`
- `scripts/step6_1_smartargets.py` — Added `apply_regulatory_feedback()` function; integrated into profit-driven and mandated deployment sections of `run_market_simulation()`; added `lmp_history` rolling tracker

**Validation targets**: ERCOT 2050 P50 LMP should come down from ~$172 to ~$50-60 range. PJM 2050 should drop from ~$111 to ~$60-75.

---

## Historical Status Log — Mar 14 → Feb 28, 2026

Compact session log. Every analytical decision (methodology, constants, cost
tables, dispatch rules, pipeline architecture) is already captured in the
canonical §N body below or in the pointed-to reference file. Session-tactical
notes (branch names, PIDs, "next steps" checklists, completed-task bullets)
have been dropped — git history is the record.

### Mar 14, 2026 — Unit Commitment + 5-Tier Heat-Rate Dispatch

- **Unit-commitment layer in LMP engine** — `compute_hourly_lmp_vectorized()`
  gained must-run pricing depression when residual demand drops below the
  must-run floor. `MUST_RUN_PCT = {nuclear: 1.0, coal_steam: 0.40, gas_ccgt:
  0.0, gas_ct: 0.0, oil_ct: 0.0}` in `pipeline_config.py`. Per-ISO
  `must_run_depression` parameter (PJM/MISO 0.35, SPP 0.30, ERCOT 0.25, other
  ISOs 0.15). Depression is multiplicative, applied before demand-quantile
  layers. PJM off-peak $34.75 → $28.77 (target $28 ±15%).
- **5-tier heat-rate dispatch** replaces the 3-bin model. Per fuel type
  (`gas_ccgt`, `coal_steam`, `gas_ct`) the stack is `very_low / low / medium /
  high / very_high` with canonical HR, CO₂, VOM, and capacity fractions in
  `EFFICIENCY_TIERS` (`scripts/lmp_engine.py`). `VINTAGE_TIER_THRESHOLDS`
  maps online year → tier (e.g., gas_ccgt 2010+ = very_low, pre-1993 =
  very_high). Retirement order is very_high → very_low within each tier;
  coal → oil → gas_CT → gas_CCGT across tiers. Per-tier sweep CFs load from
  parquet via `SWEEP_TIER_CF_COLS`; falls back to aggregate CF columns.

### Mar 9, 2026 — Nuclear Policy Toggle + Net-Zero Convergence + Step 12 Retirement

- **Nuclear Policy toggle (Stable / Roll-Off)** on procurement dashboard.
  SSS fixed fleet: PJM 95 TWh, NYISO 42 TWh, MISO 30 TWh. Rolloff schedule:
  NJ ZEC 2026 (−15 TWh PJM), IL CMC 2028 (−50 TWh PJM, −15 TWh MISO),
  NY ZEC Tier 3 2030 (−42 TWh NYISO). Rolled-off plants stay running under
  federal 45U PTC ($15/MWh) + capacity revenue — move from SSS pool (free)
  to merchant clean pool (priced at EAC market rate ~$5/MWh). Pool
  reallocation, NOT retirement.
- **Step 12 nuclear retirement & stranding** — 25 IPP nuclear plants across
  6 companies. CfD-style policy modeling (NOT additive). `total_rev =
  base_rev + max(0, applicable_floor - base_rev_per_mwh) × gen_mwh`. Floors:
  IL CEJA CMC $33.38 (expiry 2027, 11,782 MW), NY ZEC $17.48 (2029, 3,325
  MW), NJ ZEC $10 (expired 2025, 3,467 MW), 45U Federal PTC $15 (2032,
  fleet-wide). Added Seabrook (NEISO, 1,244 MW). Output:
  `scripts/step6_2b_nuclear_retirement.py`, `dashboard/nuclear_retirement.html`.
- **Step 10 SMARTargets net-zero convergence fix.** Facilitating scenarios
  (AT1/AT3/QT1/QT3) reach net-zero via DAC backstop (grid deploys until
  marginal MAC > DAC cost, then DAC offsets remainder). Challenging
  scenarios (AT2/AT4/QT2/QT4) forced to ~100% clean via $8/MWh queue
  overshoot adder, shadow price tracks implied policy cost. DAC trajectory
  (Low / Medium / High $/ton): 2030=$400/600/800, 2035=$250/400/600,
  2040=$180/300/450, 2045=$130/220/350, 2050=$100/150/250. New output
  fields: `gross_emissions_mt`, `dac_offset_mt`, `dac_cost_million`,
  `dac_cost_per_ton`, `carbon_shadow_price`.
- **Battery/LDES pipeline architecture fix.** `battery_dispatch_pct` was
  being read as dispatch when it actually encoded capacity (% of annual
  demand). Fix: Step 4 `build_annual_manifest()` is the single source of
  truth with separate `battery_capacity_pct` and `battery_dispatch_pct`
  columns. `enrich_parquets_with_dispatch_shares()` normalization fixed
  (`sum(profile) * 100`, not `/ H * 100`). Step 5AB merged CO₂ + LMP into
  one dispatch pass (`step5ab_fossil_dispatch.py`). Dispatch/capacity
  ratios: CAISO 162×, NYISO 111×, NEISO 118×, SPP 82×, ERCOT 77×, MISO
  64×, PJM 54×. Manifest schema at `data/step3-dispatch/{ISO}_annual_manifest.parquet`.
- **Storage unit standardization** — all storage grid levels now use
  `% of annual demand` (same as solar/wind/clean_firm). Old grids were
  8,760× too large. Recalibrated in 8 files (`step1_pfs_generator.py`,
  `step1d_fine_storage.py`, `step1d2_enhanced_storage.py`,
  `step3_cost_optimization.py`, `step3_track_nb_ctr.py`,
  `scenario_common.py`, `pipeline_config.py`, `dispatch_utils.py`
  comments). Cost verification: 0.01% bat4 CAISO Medium = $4.16/MWh
  matches physical calculation ($4.13/MWh). All pre-standardization
  step1/step1d parquets invalidated. See also §5.5.
- **Consequential queue consolidation + MAC bug fix.**
  `build_consequential_queue()` in `scenario_common.py` is the canonical
  queue builder; supports `per_iso_sequential` and `global_merit_order`.
  ISO-aware zone builder starts zones from the nearest threshold ≤ each
  ISO's existing clean floor: CAISO/ERCOT/PJM/SPP start at 40%,
  NYISO/NEISO/MISO start at 30%. Strategy 1 consumes the queue via
  `_load_queue()` in `step5_5_strategy1_consequential.py`. Canonical MAC
  formula also recorded below.

**Canonical MAC formula (locked Mar 4, 2026):**
```
MAC = clean_resource_LCOE × demand_MWh / CO₂_displaced_tons   ($/tCO₂)
```
where `clean_resource_LCOE = total_cost − gas_cost`. Clean resources = solar,
wind, offshore wind, nuclear, CCS-CCGT, geothermal, battery (4hr/8hr), LDES,
Green H₂. Hydro existing-only at $0 (sunk), contributes nothing to numerator.
Gas RA excluded (reliability, not abatement). No wholesale offset (Step 3
already prices existing at $0). Zones entirely below each ISO's existing
clean floor are excluded.

### Mar 10, 2026 — Codebase Audit Action Items (closed)

All C/H/M items from the Mar 10 audit are resolved or explicitly deferred.
- **C1 LDES double-counting** — fix applied in `dispatch_utils.py` +
  `step1_pfs_generator.py` (4 locations). Existing pre-fix caches carry
  a known 2–5pp CFE bias at ≥95% thresholds; regenerated caches are clean.
- **C2 Scope-2 emission baselines** — intentional design. Grid-avg
  baseline vs fossil-avg displacement maps to Scope-2 accounting tier
  (location / market / consequential). Strategy variants 1A/1B/1C
  labeled accordingly.
- **C3 MAC divergence** — deferred for future session (user decision
  pending).
- **C4 DAC cost trajectories** — citations added (Rubin 2015, Fuss 2018,
  IEA 2022). Optimistic floor raised $100 → $150/ton.
- **H3 ISO single-zone** — see §19.7.
- **H4 over-procurement ratios** — open; hardcoded values still in place.
- **H5 deployment queue double-allocation** — open.
- **H8 EPA SCC** — $51 → $190 (2024 EPA central, 3% SDR). $185 Rennert
  retained. EU ETS clarified as market price, not SCC.
- **M1 weather-year** — documented §19.8, future work §21.5.
- **M2 battery annualization** — capacity-based by design (0.1270 = CRF
  0.1019 + FOM 0.0251). Covered by §19.1.
- **M3 storage dispatch priority** — documented §19.9.
- **M4 DR/DSM** — documented §19.10, future work §21.6.
- **M5 learning rates** — citations added; solar 24%, wind 15%, battery
  18% verified vs Bolinger / BNEF.
- **M6 NEISO gas adder** — validated ($7.50/MMBtu winter adder matches
  Winter 2024/25 Algonquin Citygate actuals; sources: EIA, NGI, ISO-NE).
- **M7 LMP calibration** — blocked on `calibrate_lmp_model.py` import
  path update (module was merged into `step5ab_fossil_dispatch.py`).
- **M8 LCOE tables** — verified NREL ATB 2024; year header added to
  `pipeline_config.py`.
- **M9 geothermal cap citation** — USGS 2008 + CA Energy Commission
  2021 added to §19.6.
- **L1 code quality** — open (dispatch consolidation, type hints, test
  expansion).

### Mar 3, 2026 — Observatory Dashboard Refactor + Offshore Wind

- **Observatory design system** chosen: Space Grotesk (headings), DM Sans
  (body), Space Mono (data). Light/dark rhythm — alternating white data
  panels and dark navy narrative sections. Oscilloscope-style animated
  header. 16px-radius cards, mega-menu navigation (3-column Economics /
  Markets & Grid / Procurement & Strategy), 5-file CSS architecture
  (`shared.css` + `scrollytell.css` + `dashboard-ui.css` + `article.css` +
  `reference.css`, ~7,548 lines centralized). Design standards live in
  `DESIGN_SYSTEM.md` (canonical). Franklin Gothic removed site-wide;
  ~8,000 lines of inline CSS extracted. Archived `procurement_comparison.html`
  (superseded) and `pipeline_map.html` (covered by `pipeline.html`).
- **Offshore wind Steps 4–7 integration** — canonical spec at §21.4. Mar 3
  session drove the color/display-order update below and threaded
  offshore wind through dispatch, Step 5 cache, Step 6 scripts, Step 7
  shared-data, dashboards, and GitHub Actions.

**Canonical color update (Mar 3, 2026):** Nuclear `#1E3A5F` → `#6366F1`
indigo. CCS `#0D9488` → `#64748B` slate. Offshore wind added `#009688`
teal. Display order: Nuclear → Geothermal → Hydro → CCS → Offshore Wind →
Onshore Wind → Solar → Battery 4 → Battery 8 → LDES → H₂. Full palette in
`DESIGN_SYSTEM.md` and `dashboard/js/chart-colors.js`.

### Mar 2, 2026 — Step 1c/1d Streamlined PFS Architecture

- **"Score Once, Bin by Threshold" redesign.** A mix's hourly-match score
  is a fixed physics property — score every `(mix)` and `(mix, storage)`
  tuple exactly once, then assign to all thresholds where feasible/near-
  miss. 5-script pipeline: `step1_prior_windows.py` (search windows from
  prior EFs), `step1a_generate_mixes.py` (prior-informed bounds + 100
  scout mixes, 50 random + 50 corner), `step1b_score_mixes.py` (cached
  scoring kernel), `step1c_zone_search.py` (score-band zones with global
  dedup), `step1d_fine_storage.py` (two-pass: coarse global → fine
  targeted 0.05%). Score-band zones: A [0.45,0.78] = t50–t75,
  B [0.73,0.93] = t75–t90, C [0.88,1.00] = t90–t99.9. Accuracy targets:
  mix 1%, storage 0.05% on frontier mixes / 1% elsewhere. ~5× compute
  reduction vs per-threshold loop. Old 1c/1d preserved as `_legacy.py`.
  Workflow fixes: `--thresholds` flag, per-threshold git commits, graceful
  fallback when prior windows missing.

### Feb 28, 2026 — MAC / Scenario A / Procurement Compute Architecture

- **MAC formula — pure deployment LCOE / CO₂ displaced, NO wholesale
  offset.** Bug fixed: `add_mac_column()` and `compute_dg_mac()` in
  `step5_compute_mac_stats.py` were double-subtracting
  `existing_pct × wholesale` (Step 3 already zeroes existing resources).
  Sanity floors: SPP wind $37/MWh ÷ 1.021 tCO₂/MWh = $36/tCO₂ against
  coal, $94/tCO₂ against gas. Values below ~$27/tCO₂ are physically
  impossible. Canonical MAC definition at the Mar 4 block above.
- **Scenario A forward-stepping rewrite** — filter-first with PFS
  fallback. At each threshold: convert prior-step deployed TWh into
  per-resource floor %, filter EF mixes to those meeting ALL floors,
  price under scenario cost assumptions, pick cheapest `total_cost`. If
  floor eliminates all EF mixes → progressive PFS fallback (narrow
  window → wide window → floor-as-min-only → under-floor with
  cost-carry-forward). Per-resource floors, not aggregate. Implemented
  in `scripts/step6_scenario_comparison.py::_forward_step_optimization()`.
- **Consequential queue thresholds filtered to ≥ 50%.** Below 50% is
  pre-SBTi baseline. MAC formula in queue = newbuild LCOE / displaced
  emission rate (the narrow-buyer's metric — the analysis's whole point
  is that optimizing on this in isolation yields adverse system-level
  outcomes; comparative layers track the externalities).
- **Procurement strategy compute architecture** — resolved all 7 §15.14
  cards (now canonical in §15.14.1–7). Card 1=1A (one script per strategy
  family), Card 2=dual-toggle 45U + CTR NOAK premiums, Card 3=SSS 8760 +
  existing-minus-SSS at premium + new-build, Card 4=4B independent
  annual + hourly + scarcity, Card 5=5C full 8760-hr LMP all 7 ISOs,
  Card 6=6D SBTi default + manual override, Card 7=7B standalone
  `procurement-strategy-data.js`. PPA percentage-premium model (LCOE × (1
  + pct)): VRE 5/12/22%, Firm 12/22/38%, Uprate 10/20/35%. LMP extended
  to all 7 ISOs with MISO+SPP models in `step5_compute_lmp_prices.py` and
  2024 SOM calibration targets in `calibrate_lmp_model.py`.

---

---

## Current Status (Feb 26, 2026)

### Optimal CFE Targets + No-Regrets Investments (Feb 26, 2026)

**Completed:**
- Created `scripts/step5_compute_optimal_targets.py` — Step 6 post-processor computing optimal CFE target range per ISO
- Smooth marginal MAC via PCHIP spline derivatives on isotonic-corrected cost/CO₂ curves
- 3 grid cost tiers × 3 DAC scenarios = 9 crossover points → range per ISO
- L/M/H demand growth scenarios (scale-invariant for MAC %, but affects absolute resource quantities)
- No-regrets resource investment analysis: floor, consensus, and average resource investments within the crossover range
- Uses canonical CO₂ model from `dispatch_utils.compute_fossil_retirement()` (same as step5_compute_co2, step5_compute_mac_stats)
- Output: `data/step4-analysis/optimal_targets.json` + `dashboard/js/optimal-target-data.js`
- Wired into `step6_generate_shared_data.py` → OPTIMAL_TARGETS constant in shared-data.js
- Added to Step 6 GitHub Actions workflow (parallel batch with MAC, LMP, compressed day, etc.)
- Added scipy to workflow dependencies
- **Gas cost separation (Feb 26)**: MAC uses `effective_cost` (clean procurement only), NOT `total_system_cost` (which includes gas backup). Gas backup is a system reliability cost, not an abatement cost. See §7.4.1.
- **Step 3 cost function fix (Mar 7)**: Removed CCS residual pricing and gas backup from Step 3 optimization cost. Previously, `ccs_pct = 100% - sum(explicit resources)` was priced at CCS LCOE, forcing 100% procurement and making low-threshold mixes artificially expensive (e.g., ERCOT t=50% selected an 80.9% mix because 47% CCS residual at $72.5/MWh dominated the cost). Fix: CCS is ONLY priced within the clean_firm tranche system (Tranche 3). Gas backup computed post-hoc for output but excluded from optimization cost — it represents the consequence of not planning ahead, not the capital cost of decarbonization. Battery/storage revenue credits retained to drive correct battery selection.
- Added `CLEAN_COST_DATA` extraction to step7 (P10/P50/P90 of effective_cost across scenarios)
- Fixed step5_consequential_deployment_queue.py MAC to exclude gas backup cost

**Completed (Feb 27):**
- Revamped `abatement_dashboard.html` with optimal target tiles, ISO deep-dive, and no-regrets analysis
- Created `scripts/step6_extract_no_regrets.py` — extracts no-regrets resources from Step 3 DG parquets (5,832+ scenarios)
- Revised DAC cost trajectories upward across all 4 files (anchored to 2025 actual costs of $600-$1,500/tCO₂)
- **Wired smooth PCHIP MAC into dashboard** — `abatement_dashboard.html` now uses `OPTIMAL_TARGETS` smooth curves from `optimal-target-data.js` when available, falls back to `MAC_STEPWISE_FAN` stepwise data otherwise
- Dashboard data priority chain: `OPTIMAL_TARGETS` (smooth PCHIP) > `MAC_STEPWISE_FAN` (stepwise fallback)
- No-regrets data priority chain: `NO_REGRETS_DATA` (DG parquets, 5,832+ scenarios) > `OPTIMAL_TARGETS.no_regrets` (medium-cost) > client-side `RESOURCE_MIX_DATA` analysis (fallback)
- Crossover computation prefers pre-computed smooth crossover range from `OPTIMAL_TARGETS`, falls back to client-side stepwise computation
- Created placeholder `dashboard/js/optimal-target-data.js` — populated by running `step5_compute_optimal_targets.py`

**Next steps:**
- [ ] Run Step 6 workflow to generate `optimal_targets.json` + `optimal-target-data.js` (smooth MAC data)
- [ ] Generate DG parquets for remaining ISOs (ERCOT, PJM, NYISO, NEISO, MISO, SPP) — currently only CAISO
- [ ] Re-run `step6_extract_no_regrets.py` after all ISO DG data is available
- [ ] Wire no-regrets investment data into research paper narrative
- [ ] Add prominent gas capacity warning to consequential scenario dashboard

### Scenario Comparison Page Fixes (Feb 26, 2026)

**Completed:**
- Fixed FOAK→NOAK learning curve chart (inline data constants, was referencing missing shared-data.js vars)
- Moved target slider inline with ISO selector in sticky bar
- Redesigned metric tiles with heat-map styling (green/amber/red) and prominent conclusions
- Replaced infinite MAC bars with ⚠ symbol when no emissions displaced

### Pipeline Reorganization — Step 5/6/7 Split (Feb 26, 2026)

**Completed:**
- Reorganized post-processing pipeline by dispatch cache dependency:
  - **Step 5**: Dispatch cache build + cache-independent scripts (EAC scarcity, track export, track analysis)
  - **Step 6**: Dispatch-cache-dependent scripts (compressed day, CO2, MAC, LMP, consequential, scenarios)
  - **Step 7**: Generate shared data (dashboard aggregation)
- Refactored PP2 (consequential queue) to use dispatch cache for hourly emission accounting via `compute_co2_from_dispatch()`
- Refactored PP3 (scenario comparison) — both Scenario A and Scenario B now use dispatch-cache-based emission accounting
- Added `compute_co2_from_dispatch()` to `dispatch_utils.py` — shared hourly emission function using merit-order fuel displacement
- Integrated Track 2&3 workflow into Step 3 with track selector dropdown (Track 1 baseline / Track 2 NB / Track 3 CTR)
- Added `--track` flag to `step3_track_nb_ctr.py` (nb/ctr/both)
- Renamed all scripts from PP-numbered to step-numbered (see Pipeline Architecture below)
- Updated all internal references, docstrings, imports, workflow files

**Script renames:**
| Old name | New name |
|----------|----------|
| `step5_PP0_build_dispatch_cache.py` | `step4_build_dispatch_cache.py` |
| `step5_PP1_compressed_day.py` | `step5_compress_day_profiles.py` |
| `step5_PP2_consequential_queue.py` | `step5_consequential_deployment_queue.py` |
| `step6_scenario_comparison.py` | `step6_scenario_comparison.py` |
| `step5_compute_co2.py` | `step5_compute_co2.py` |
| `step5_compute_mac_stats.py` | `step5_compute_mac_stats.py` |
| `step5_compute_lmp_prices.py` | `step5_compute_lmp_prices.py` |
| `step5_PP7_compute_eac_scarcity.py` | `step5_compute_eac_scarcity.py` |
| `step5_PP8_export_track_results.py` | `step5_export_track_results.py` |
| `step5_PP9_analyze_tracks.py` | `step5_analyze_tracks.py` |
| `step6_generate_shared_data.py` | `step6_generate_shared_data.py` |

**Pipeline execution order:**
```
Step 4 → Step 5: dispatch cache build → Step 5: cache-independent (EAC, tracks) in parallel
                                       → Step 6: cache-dependent (CO2, MAC, LMP, compressed day, consequential, scenarios) in parallel
                                                → Step 7: generate shared data
```

### Pipeline Audit & Dispatch Consolidation (Feb 26, 2026)

**Completed:**
- Created `step4_build_dispatch_cache.py` — pre-computes 8,760-hour dispatch for all unique mixes across all ISOs. Versioned NPZ cache (v2) with per-resource matched/surplus + charge profiles.
- Extended `dispatch_utils.py` with `detailed=True` mode on `reconstruct_hourly_dispatch()`: adds per-resource merit-order breakdown (CF→CCS→hydro→wind→solar) and storage charge tracking.
- Added `DISPATCH_ORDER`, `CACHE_VERSION`, `_compute_per_resource_dispatch()`, `_battery_loop_detailed()`, `_ldes_loop_detailed()` to dispatch_utils.
- Fixed PP4 supply profile bug: was importing `get_supply_profiles_simple` (flat clean_firm, no DST correction) instead of canonical `get_supply_profiles`. CO₂ results on the dispatch path will now use correct nuclear seasonal derate + DST-corrected solar.
- Refactored `step5_compress_day_profiles.py`: removed ~170 lines of duplicate dispatch engine (battery/LDES loops, supply profiles, data loading). Now imports from dispatch_utils and reads from dispatch cache.
- Added cache-miss warning to `step5_compute_lmp_prices.py`.
- Removed dead code: `get_supply_profiles_simple` from dispatch_utils.
- Added version metadata support to `load/save_dispatch_cache()`.
- Updated all documentation: SPEC.md, CLAUDE.md, README.md, pipeline.html, optimizer_methodology.html.
- Added MISO and SPP ISO support across pipeline and dashboard.

**Pipeline execution order (legacy reference):**
```
Step 4 → dispatch cache → CO2/MAC/LMP/compressed day (read from cache, run in parallel)
```

### Step 1 PFS Rebuild — Two-Phase Adaptive Storage Sweep (Feb 21, 2026)

**Completed:**
- Energy-based battery caps (replaces power-based formula): `bat4_cap = max_daily_surplus`, `bat8_cap = max_2day_surplus`, `ldes_cap = max_7day_surplus`
- Two-phase adaptive storage sweep: Phase 1 coarse (0.25% steps) → analyze saturation → Phase 2 fine (0.05% steps) within saturation range
- Per-ISO/threshold output files: `data/step1-pfs/{ISO}_t{XX}_raw_pfs.parquet`
- ERCOT: 2,033,961 solutions (complete)
- CAISO: Running (Phase 2 in progress)

**In Progress:**
- CAISO two-phase run
- PJM, NYISO, NEISO queued

**Per-ISO Results:**
| ISO | Solutions | Phase 1 (coarse) | Phase 2 (fine) | bat4 sat | bat8 sat | Time |
|---|---|---|---|---|---|---|
| ERCOT | 2,033,961 | 226K | 1.8M | 0.75% | 1.50% | 11 min |
| CAISO | (running) | — | — | 1.00% | 2.00% | ~25 min |
| PJM | (queued) | — | — | — | — | — |
| NYISO | (queued) | — | — | — | — | — |
| NEISO | (queued) | — | — | — | — | — |

**Branch:** `claude/validate-battery-physics-bqZTr`

### LMP Module — v9 Calibration Complete (Feb 21, 2026)

**Completed:**
- `dispatch_utils.py` — shared dispatch module (constants, profiles, battery/LDES dispatch, fossil retirement, hourly dispatch cache)
- `step5_compute_lmp_prices.py` — core LMP engine with:
  - Merit-order fossil stack: PJM Manual 15 cost-based offer formula (HR × fuel + VOM + CO2 + 10% adder)
  - Heat rates calibrated to PJM SOM 2024 benchmarks (CCGT 7.0, CT 10.5, coal 10.0 MMBtu/MWh)
  - CO2 allowance costs (RGGI-weighted: L/M/H = $3/$5.50/$14 per ton)
  - 10% adder per PJM market rules ($2.00/MWh contribution per SOM 2024)
  - VOM split: variable maintenance + variable operations (SOM 2024: $3.18 + $1.43 = $4.61 fleet avg)
  - Load-dependent heat rate ramp (15% quadratic) for within-band price variation
  - Demand-quantile pricing: congestion, scarcity tail, off-peak compression
  - ISO-specific price formation: PJM (RPM), ERCOT (ORDC), CAISO (RA), NYISO (ICAP), NEISO (FCM + winter gas)
  - Archetype deduplication: (mix, fuel_level, threshold) → ~7,800 unique per ISO
  - Dispatch cache: append-mode NPZ per ISO, shared with step5_compute_co2.py
- `calibrate_lmp_model.py` — validation framework with embedded PJM IMM/EIA reference data
- `step5_compute_co2.py` — refactored to import from dispatch_utils.py (identical behavior)

**PJM v9 Calibration Results (2024 baseline, Medium fuel/CO2):**

| Metric | Synthetic | Target | Delta | Status |
|---|---|---|---|---|
| Avg LMP | $36.69 | $34.70 | +6% | GOOD |
| Peak avg | $38.82 | $42.00 | -8% | GOOD |
| P10 | $20.00 | $18.00 | +$2 | FAIR |
| P25 | $24.37 | $23.00 | +$1.37 | GOOD |
| P50 | $31.88 | $30.00 | +$1.88 | GOOD |
| P75 | $38.96 | $42.00 | -$3 | GOOD |
| P90 | $50.21 | $55.00 | -$5 | GOOD |
| Scarcity hours | 102 | 100 | +2 | GOOD |
| Negative hours | 246 | 200 | +46 | FAIR |
| Volatility | $31.03 | $25.00 | +$6 | FAIR |

- Calibration report: "No major adjustments needed"
- Known limitation: off-peak avg ($34.75 vs $28) — no unit commitment/min-gen constraints

**Marginal Costs at Medium (with 10% adder + CO2):**
- Gas CCGT: $33.04/MWh (PJM 2024 RT avg: $33.74 — matches within 2%)
- Coal: $36.55/MWh
- Gas CT: $49.25/MWh
- Oil CT: $131.81/MWh

**Track Sweep Status:**
- CAISO: Complete (NB + replace, 12 thresholds × 209,952 scenarios each)
- ERCOT: NB partial (10/20 thresholds), replace not started
- PJM, NYISO, NEISO: Not started
- Checkpoint: `data/track_checkpoint.json` (partial results)
- Parquet export: `dashboard/track_scenarios.parquet` (CAISO only)

**All-ISO Price Models (Added Feb 28):**
- All 7 ISOs now have calibrated price model classes (PJMPriceModel through SPPPriceModel)
- GAF (Gas Availability Factor) added for MISO (0.83) and SPP (0.84)
- `calibrate_lmp_model.py` updated with ISO_CALIBRATION_TARGETS for all 7 ISOs
- `--iso ALL` support in both `step5_compute_lmp_prices.py` and `calibrate_lmp_model.py`
- 2024 SOM calibration targets (avg LMP, percentiles, negative hours, scarcity):
  - SPP: $26/MWh (cheapest), ERCOT: $26, MISO: $31, PJM: $34.7, CAISO: $38, NEISO: $39.5, NYISO: $42

**Next Steps:**
- Fetch actual 2024 EIA hourly data for all ISOs and run calibration validation
- Tune demand-quantile parameters per ISO to match actual LMP distributions
- Run LMP model on full all-ISO ECF scenarios (all thresholds × fuel sensitivities)

### LMP Price Calculation & Existing vs New-Build Analysis (Feb 20, 2026)

**Goal**: Separate existing-generation vs new-build pricing to enable "with vs without existing" analysis. Shows cost of replacing nuclear/hydro, asset stranding risk, and true greenfield costs.

#### Decision 1: Hydro Treatment — Both Scenarios (1C)
- **With hydro**: Hydro included as existing wholesale-priced resource. Shows cost advantage of existing hydro and its value for hourly matching. Better view of cost to replace nuclear.
- **Without hydro**: Mixes with hydro=0 in the EF. Shows new-build requirement for hourly matching without hydro, procurement impact, and potential asset stranding if hydro is curtailed or unavailable.
- **Implementation**: Step 2 EF existing clean floor removed → hydro=0 mixes now available. Step 3 can filter to hydro=0 or hydro>0 mixes for comparison.

#### Decision 2: Nuclear Uprates — Both Scenarios
- **With uprates**: Tranche 1 (uprate) pricing preserved. Better for hourly matching since uprates are cheap dispatchable capacity.
- **Without uprates**: Tranche 1 disabled (uprate_cap=0). All new clean firm priced at tranche 2 (geothermal) / tranche 3 (new-build nuclear vs CCS). Provides better view of true new-build replacement cost.
- **Implementation**: `uprate_mode` flag in Step 3 cost function. When 'off', `uprate_cap=0` and all new CF flows directly to geothermal/new-build tranches.

#### Decision 3: Below-Floor Mix Recovery
- **Problem**: Step 2 Phase 0 (existing clean floor filter) removed mixes that under-allocated existing generation. This filtered out hydro=0 and low-clean-firm mixes needed for greenfield analysis.
- **Approach**: Temp recovery script reads PFS cache, inverts the floor filter, recovers below-floor mixes, runs Pareto procurement, and merges into existing `pfs_post_ef.parquet`.
- **Step 2 update**: Remove Phase 0 filter for future runs (kept as dead code for reference). No full PFS re-run needed — temp script recovers the delta.
- **Script**: `recover_below_floor.py` (one-time use, can be deleted after merge)

#### Decision 4: Existing Clean = $0, New-Build = LCOE (Updated Feb 25, 2026)
- **Approach**: Existing clean generation priced at **$0** (sunk fleet — already built and operating, no cost to buyer). New-build priced at LCOE + transmission adder. Previous wholesale pricing of existing resources was distorting the optimizer toward wind-heavy mixes that underutilized the existing fleet.
- **Rationale**: For Track 1 baseline analysis, existing clean resources are constants on the grid. Charging them at wholesale ($25-42/MWh) penalized mixes that used more existing, causing the optimizer to prefer over-procuring cheap new-build wind while leaving free existing nuclear/solar unused.
- **Effect**: Optimizer now strongly prefers mixes that maximize use of existing fleet (free) before adding new-build. ERCOT 50% at 2030 now correctly shows ~13% CF / 22% solar / 65% wind at 55% procurement (using ~42 TWh of existing nuclear) instead of 5% CF / 86% wind at 65% procurement.
- **Implementation**: `coeff_matrix[:, _COL_WHOLESALE] = 0` in coefficient model; `total_cost += 0` for existing in price_mix_batch.

#### Decision 6: LMP Module Architecture (Feb 21, 2026, updated v9)
- **Pipeline position**: Downstream of Step 4, reads ECF base case from `overprocure_scenarios.parquet`
- **Cost-based offer formula** (PJM Manual 15): `MC = (Heat Rate × Fuel Price + VOM + CO2 Rate × CO2 Price) × (1 + 10% Adder)`
- **Heat rates** (PJM SOM 2024 benchmarks): Coal 10.0, CCGT 7.0, CT 10.5, Oil 10.5 MMBtu/MWh
- **VOM** (SOM 2024 decomposition: $3.18 maintenance + $1.43 operations): Coal $5.50, CCGT $3.50, CT $5.00, Oil $6.00 $/MWh
- **CO2 allowance** (RGGI-weighted for PJM): Low $3/ton, Medium $5.50/ton, High $14/ton. SOM 2024: $1.94/MWh contribution.
- **10% adder**: PJM market rules allow 10% markup above cost-based offers. SOM 2024: $2.00/MWh contribution.
- **Stack walk**: `np.searchsorted` step function with load-dependent heat rate ramp (15% quadratic)
- **Demand-quantile pricing**: High-demand congestion adder (P75+), scarcity tail (P95.5+), mid-low compression (P10-P70), negative pricing (P0-P10)
- **ISO price formation**: PJM RPM ($2K cap), ERCOT ORDC (exponential knee, $500 cap), CAISO RA (-$60 floor), NYISO ICAP, NEISO FCM (+$13.13 winter gas). All ISOs use ORDC exponential knee model with per-ISO VOLL/knee/λ/cap parameters.
- **Installed capacity**: EIA 860 actuals (PJM 127.8 GW, ERCOT 80 GW, CAISO 47 GW, NYISO 28 GW, NEISO 16 GW)
- **Fuel prices**: Low/Medium/High sensitivity (coal $2.00-2.50, gas $2.00-6.00, oil $8.00-13.00 $/MMBtu)
- **Dispatch cache**: Shared with recompute_co2.py via dispatch_utils.py, append-mode NPZ per ISO
- **Calibration reference**: PJM IMM 2024 SOM: RT LW avg $33.74/MWh, total wholesale $55.54/MWh
- **LMP runs on ECF track only** (base case with existing clean floor). NB/CTR tracks are separate analysis, not priced through LMP.
- **Data sources**: PJM Manual 15 Rev. 47, Monitoring Analytics 2024 SOM, EIA Electric Power Annual Table 8.1, EPA eGRID 2022

#### Decision 5: Three Analysis Tracks (5C — Updated Feb 24, 2026)
Three distinct tracks with standardized naming. Each track represents a different set of assumptions about what existing clean energy is credited vs. what must be procured new.

**Track 1 — ECF (Existing Clean Floor)**: Baseline case
- Cost optimization built on **2025 absolute existing clean generation** — the "existing clean floor"
- Existing clean TWh remain **constant** across all demand growth scenarios (absolute TWh steady; share of total generation declines over time as demand grows)
- All existing resources priced at **$0** (sunk fleet — no cost to buyer)
- New-build procurement optimized on top of the existing floor to hit each target threshold
- Source: `overprocure_scenarios.parquet` (baseline results)
- LMP module runs on this track
- Files/caches use `ecf_` prefix

**Track 2 — NB (New-Build / Greenfield)**: What hourly matching incentivizes to build
- **Ignores ALL existing clean energy** — no existing solar, wind, nuclear, CCS, or hydro credited
- Hydro: **excluded** (hydro=0 mixes only, not considered at all)
- Uprates: **on** (uprate tranche active — cheapest new-build option, part of greenfield procurement)
- All resources must be procured as new-build (except uprates which are allowed)
- Purpose: What does hourly matching incentivize you to BUILD from scratch?
- Files/caches use `nb_` prefix

**Track 3 — CTR (Cost to Replace Clean Firm)**: What it costs to replace existing dispatchable clean
- **Does NOT include** existing clean firm (nuclear) or uprates — these are what's being "replaced"
- **Does NOT include** existing CCS (near-zero in most ISOs, but conceptually part of what's replaced)
- Uprates: **off** (uprate_cap=0, no uprate tranche)
- Hydro: **included** (existing floor, $0 — sunk fleet)
- Existing solar: **included** (existing floor, $0 — sunk fleet)
- Existing wind: **included** (existing floor, $0 — sunk fleet)
- Purpose: True cost of replacing existing dispatchable clean generation (nuclear/CCS), while keeping existing renewables and hydro as the floor
- Files/caches use `ctr_` prefix

**Naming convention**: All file names, cache keys, code comments, and output fields use ECF/NB/CTR abbreviations consistently. File rename deferred until NB/CTR sweep completes.

**Output**: Data files only. No research paper update yet — discuss findings with user first, then write.
**Architecture**: Step 3 runs 2 additional passes per (ISO, threshold):
  - Pass "NB": filter EF to hydro=0, zero all existing clean, uprate tranche on → pure greenfield results
  - Pass "CTR": keep hydro + solar + wind floors, zero clean firm + CCS existing, uprate tranche off → replacement cost results
  - Plus existing pass "ECF": full EF, all features on → current behavior (preserved)

#### Decision 5b: Track-to-Page Assignment (Feb 24, 2026)
Each dashboard page uses a specific track (or combination) for its data and visualizations:

**Track 1 (ECF) exclusively:**
| Page | File | Notes |
|---|---|---|
| Home | `index.html` | Scrollytelling narrative, key findings |
| Grid Simulation | `dashboard.html` | Interactive optimizer with all sensitivity toggles |
| CO₂ Abatement | `abatement_dashboard.html` | MAC curves, abatement ladders |
| Wholesale Price Assessment | `lmp_trends.html` | **PJM only** — LMP trends and price impact |
| Fossil Fuel Deep Dive | `fossil_fuel_deepdive.html` | Fossil retirement, fuel switching |

**Track 2 (NB) + Track 3 (CTR) compared against Track 1 (ECF):**
| Page | File | Notes |
|---|---|---|
| Nuclear Revenue Crossover | `lmp_trends.html#nuclear-crossover` | CTR vs ECF — replacement cost of dispatchable clean (merged from cost_to_replace.html) |
| New Build Analysis | `new_build_analysis.html` | NB vs ECF — procurement strategy, EAC scarcity scaling, LMP feedback loop (**PJM only**) |

#### Decision 5e: New Build Analysis Page Design (Feb 25, 2026)

**Page**: `new_build_analysis.html` — "The Build-or-Buy Decision"
**Scope**: PJM only. Scrollytell hybrid (Sections 1–2 interactive, Section 3 narrative scroll).

**Section 1: The Procurement Choice** (interactive)
- Target slider: range input snapping to 11 PJM thresholds (50–92.5%)
- SBTi year label displayed alongside slider value
- Side-by-side comparison: Track 1 (ECF, "With Existing Clean") vs Track 2 (NB, "New Build Only")
- Each side shows: effective cost ($/MWh), resource mix stacked bar, P10/P50/P90 sweep bands
- EAC premium sliders with dynamic defaults:
  - Nuclear: auto = max(0, $33/MWh operating cost + 10% margin − LMP(threshold))
  - Hydro: $3/MWh default
  - Existing solar/wind: $5/MWh default (REC market proxy)
- New-build resources priced at LCOE; existing resources at wholesale + EAC premium
- "Additionality Premium" callout: $/MWh delta between tracks

**Section 2: The Scaling Curve** (interactive)
- X-axis: % of C&I load participating in hourly CFE (0–100%)
- Y-axis: effective cost $/MWh
- Two curves: "With Existing" (Track 1) and "New Build Only" (Track 2) converging
- PJM supply stack: 280 TWh total clean → 95 SSS-fixed → 70 RPS → 25 existing PPAs → ~90 TWh available
- Vertical marker at ~90 TWh showing where existing merchant supply exhausts
- L/M/H cost sensitivity bands (P10/P50/P90)
- As participation scales, existing supply exhausts → Track A cost rises toward Track B

**Section 3: The Economic Feedback Loop** (scrollytell narrative)
- Step 1: LMP decline chart — PJM LMP from $36.86 (50%) to $6.80 (99%), with nuclear operating cost line ($33/MWh)
- Step 2: The widening gap IS the clean premium — above ~80% threshold, LMP < nuclear cost
- Step 3: PJM clean supply waterfall — 280 TWh total → decomposition
- Step 4: IL ZEC/CMC 2027 expiry callout — 94 TWh at risk of retirement
- Step 5: 45U PTC evidence — $15/MWh production tax credit as bridge, but expires; without policy backstop, LMP decline threatens existing nuclear viability
- Step 6: The paradox — new-build-only procurement accelerates LMP decline → existing nuclear uneconomic → retirements → more new build needed at higher cost

**Section 4: Bottom Line** (static summary)
- Cost delta summary at key thresholds
- Policy implication: procurement strategy must account for systemic feedback effects
- Clean premiums as market signal for existing nuclear viability

**Data sources**:
- `track_results.json` (Track 2 NB + Track 3 CTR vs baseline)
- `shared-data.js` (ECF baseline: EFFECTIVE_COST_DATA, RESOURCE_MIX_DATA)
- `lmp_summary.json` (PJM LMP trajectory by threshold)
- EAC scarcity parameters inline (from step5_compute_eac_scarcity constants)
- Nuclear operating cost: $33/MWh (fuel ~$5.5 + fixed O&M ~$25 + VOM ~$2.5)

#### Decision 5c: Visual Differentiation — Existing vs New-Build vs Curtailment (Feb 24, 2026)
All resource mix figures across the entire site must visually differentiate existing vs new-build generation:

- **Existing resources**: Full saturation fill (solid, opaque colors from the standard palette)
- **New-build resources**: Transparent fill with full-saturation outline (same hue, reduced opacity interior, solid border)
- **Curtailment**: Transparent fill with diagonal cross-hatching (curtailment is almost always from new-build)

This applies to ALL charts showing resource mixes (stacked bars, stacked areas, waterfall, etc.) across all pages and all ISOs.

#### Decision 5d: Resource Differentiation & Color Palette (Feb 24, 2026)
All resource mix figures must differentiate ALL individual resources (not aggregate categories). The following resources must each have a distinct, consistent color across the entire site and all ISOs:

| Resource | Color | Hex | Notes |
|---|---|---|---|
| Nuclear | Indigo | `#6366F1` | Existing dispatchable clean (updated Mar 3, 2026 from #1E3A5F) |
| CCS-CCGT | Slate | `#64748B` | Carbon capture gas (updated Mar 3, 2026 from #0D9488) |
| Geothermal | Ochre/Brown | `#B45309` | CAISO only |
| Offshore Wind | Teal | `#009688` | Atlantic ISOs + CAISO (added Mar 3, 2026) |
| Wind (Onshore) | Green | `#22C55E` | Onshore wind |
| Solar | Amber | `#F59E0B` | Utility-scale PV |
| Hydro | Cyan | `#0EA5E9` | Existing only, wholesale-priced |
| LDES (100hr) | Pink | `#EC4899` | Iron-air, 50% RTE |
| Battery (4hr) | Purple | `#8B5CF6` | Li-ion, 85% RTE |
| Battery (8hr) | Light Purple | `#A78BFA` | Li-ion extended duration |
| Green H₂ | Emerald | `#10B981` | 1000hr, 35% RTE, ≥95% only |
| Curtailment | Gray + cross-hatch | `#D1D5DB` | Diagonal hatching pattern |

**Display order** (updated Mar 3, 2026): Nuclear → Geothermal → Hydro → CCS → Offshore Wind → Onshore Wind → Solar → Battery 4 → Battery 8 → LDES → H2

**Bar chart styling**: ALL bar charts across the entire site must have **rounded corners** (`borderRadius` in Chart.js).

**Consistency rule**: These colors and styling rules apply to every page, every ISO, every chart type. No page-specific overrides unless explicitly approved.

### Columnar JSON Format for Feasible Mixes (Feb 19, 2026)

**Problem**: `feasible_mixes` in `overprocure_results.json` stored as array of dicts — each mix repeated key names (`resource_mix`, `clean_firm`, `solar`, etc.) across potentially 1.78M entries, inflating JSON from ~40 MB to ~312 MB.

**Decision**: Option 2 — Columnar format. Store as `{col_name: [values...]}` instead of `[{col_name: val}, ...]`.

**Format**:
```json
"feasible_mixes": {
  "clean_firm": [50, 60, ...],
  "solar": [25, 20, ...],
  "wind": [...],
  "ccs_ccgt": [...],
  "hydro": [...],
  "procurement_pct": [...],
  "hourly_match_score": [...],
  "battery_dispatch_pct": [...],
  "ldes_dispatch_pct": [...]
}
```

**Measured savings**: 81% reduction per threshold group (98K → 18K bytes for 510 mixes). Projected ~312 MB → ~40 MB at full 1.78M mix scale.

**Files changed**:
- `step3_cost_optimization.py` — writes columnar format
- `step5_compressed_day.py` — reads both columnar (new) and row (legacy) formats
- `step6_generate_shared_data.py` — reads both columnar (new) and row (legacy) formats
- Dashboard JS (`shared-data.js`) — already used compact arrays `[cf, sol, wnd, ccs, hyd, proc, match, bat, ldes]`; no change needed

**Backward compat**: Step 5 and generate_shared_data both auto-detect format (`isinstance(fmixes, dict)` vs `isinstance(fmixes, list)`).

### Compressed Day Chart: Curtailment Stacking Fix (Feb 19, 2026)

**Problem**: Curtailment was anchored to the demand line and stacked upward, creating a visual gap between the top of matched generation and the bottom of curtailment. On mobile, this made it look like curtailment was floating disconnected from the generation it belongs to.

**Fix**: Curtailment now stacks from `matchedTotal` (top of generation stack) upward, keeping it flush against the generation area. The demand line cuts through as a visual boundary — area above demand = true curtailment, area between matchedTotal and demand = unmatched gap.

### New Toggle Architecture Decisions (Feb 19, 2026)

Three new Step 3 cost model changes — no Step 1 physics re-run needed:

1. **CCS separated from Firm Gen toggle** — CCS gets its own L/M/H toggle (maturity-based: L=mature/low capex, H=immature/high capex) plus a binary 45Q On/Off switch. 6 CCS cost states total (3×2).
2. **Geothermal toggle (CAISO only)** — L/M/H based on published data (NREL ATB, USGS, Lazard). 5 GW cap (~39 TWh/yr) from USGS identified hydrothermal. After cap, remaining clean firm filled by cheapest of nuclear new-build vs CCS (toggle-dependent). Non-CAISO ISOs have zero geothermal resource — toggle hidden.
3. **Nuclear new-build Low target = $70/MWh** — nth-of-a-kind SMR deployment target. Regional variation: $68-75/MWh at Low.
4. **CAISO clean firm merit order**: Existing → uprates → geothermal (capped) → cheapest of nuclear/CCS (toggle-dependent)
5. **Non-CAISO clean firm merit order**: Existing → uprates → cheapest of nuclear/CCS (toggle-dependent)

**Sensitivity space expanded**: 324 → 5,832 (non-CAISO) / 17,496 (CAISO) combos. All Step 3 arithmetic — minutes, not hours.

### Demand Growth in Resource Mix Pricing (Feb 19, 2026)

**Decision**: Demand growth dynamically scales resource mix pricing. Existing generation stays flat in absolute TWh — as demand grows, existing's share of grown demand shrinks, requiring more new-build.

**Approach**: 1C (affects real pricing) + 2C (target year + growth rate as parameters) + 3C (client-side repricing handles it).

**Mechanics**:
- `grownDemandTwh = baseDemandTwh × (1 + annualRate)^(targetYear − 2025)`
- Existing share rescaled: `existingPctGrown = existingPct × (baseDemandTwh / grownDemandTwh)`
- More new-build fills the gap → higher costs at longer horizons / higher growth rates
- Growth rates per ISO from DEMAND_GROWTH_RATES (L/M/H): CAISO 1.4-2.5%, ERCOT 2.0-5.5%, PJM 1.5-3.6%, NYISO 1.3-4.4%, NEISO 0.9-2.9%
- Target year from dashboard selectors (interim: 2027-2035, longterm: 2036-2050)
- `priceMix()` accepts optional `targetYear` and `growthRate` parameters; defaults to 2025/0 (no growth = current behavior)
- Base year fixed at 2025 (snapshot year)

**Implication**: Same physical feasible mix can cost significantly more at a 2040 target than at 2028, because more of the mix must be new-build to replace the shrunken existing share. This is correct — the resource mix needs to adapt dynamically to absolute TWh.

### v4.0 Fresh Rebuild — Decisions Locked (Feb 19, 2026)

Complete optimizer rebuild with new architecture. All 9 design decisions + 5 efficiency optimizations locked below.

#### Design Decisions

| # | Decision | Choice | Detail |
|---|----------|--------|--------|
| 1 | Grid search strategy | **1C — Adaptive** | Start at 5% step, identify promising regions, refine to 1%. Replaces 3-phase 10%→5%→1%. |
| 2 | Solution output | **2B — Pareto frontier** | 3-5 points per mix along procurement/storage tradeoff (not single-point optimal). |
| 3 | Procurement bounds | **3C — Threshold-adaptive** | Narrow bounds at low thresholds (e.g., 100-110% at 50%), wider at high (100-150% at 99-≥99.9%). |
| 4 | min_dispatchable constraint | **4B — Drop it** | No dispatchable floor. Let physics prove/disprove — constraint was potentially biasing results. |
| 5 | Thresholds | **5F — 20 total** | 10/20/30/40 (coarse only) + 50/55/60/65/70/75/80/85/87.5/90/92.5/95/97.5/99/99.5/99.9 (16 active, full pipeline). Top threshold is ≥99.9% (effectively 100%) — true 100% hourly matching is physically unreachable. Thresholds 10–40 are coarse-grid only (no fine zone search, no step1d storage). |
| 6 | CCS-CCGT resource | **6D — Collapse into Clean Firm** | Merge CCS into Clean Firm allocation. Reduces resource space from 5D to 4D. CCS retains its own cost profile and dispatch characteristics within the merged allocation — the optimizer determines sub-allocation internally. |
| 7 | Storage parameters | **7A — Keep current** | Battery: 4hr Li-ion, 85% RT, daily cycle. LDES: 100hr iron-air, 50% RT, 7-day window. |
| 8 | Output format | **8C — Both** | JSON (backward compat) + Parquet (analytics). |
| 9 | Numba acceleration | **9C — Optional** | Try Numba JIT, fall back to NumPy if install fails. |

#### Efficiency Architecture (Fresh Rebuild)

| ID | Optimization | Description | Expected Speedup |
|----|-------------|-------------|-----------------|
| A | ISO parallelism | Run all 5 ISOs in parallel on 16 cores (3 cores/ISO) | 4-5× |
| B | Vectorized battery dispatch | Replace Python `for day in range(365)` with NumPy reshape + vectorized ops | 3-5× on storage scoring |
| C | Batch mix evaluation | Evaluate batches of mixes simultaneously via matrix ops: `(N,4) @ (4,8760)` | 5-10× on grid search |
| D | Numba JIT (try/fallback) | Compile storage scoring to machine code; fall back to B+C if install fails | 10-50× on storage (if available) |
| F | Shared memory cache | `multiprocessing.shared_memory` for parallel ISO workers to share data | Enables A |

**Scope**: Step 1 only (physics). No cost model — the optimizer generates the feasible solution space (all viable resource mixes per threshold×ISO). Cost sensitivities (5,832 paired-toggle scenarios) applied in Step 3 cost optimization. This reduces from 25,872 cost-coupled optimizations to 140 physics-only sweeps (20 thresholds × 7 ISOs), each finding the Pareto frontier of feasible mixes.

**Projected runtime**: ~1-3 min with Numba (installed successfully). Down from multi-hour current architecture.

**Cost model**: NOT in scope for this rebuild. Cost model will be updated separately with dynamic functionality. This optimizer produces the physics-only feasible solution space that the cost model will consume.

#### What needs building (fresh rebuild — Step 1 physics only)
- [ ] New optimizer with 4D resource space (Clean Firm absorbs CCS)
- [ ] ISO parallel execution with shared memory (A+F)
- [ ] Vectorized scoring functions (B+C)
- [ ] Numba JIT with fallback (D)
- [ ] Pareto frontier output (3-5 points per threshold×ISO)
- [ ] 13-threshold sweep with adaptive procurement bounds
- [ ] JSON + Parquet dual output of feasible solution space

### 8-Step Pipeline Architecture (Steps 0–7)

**Naming convention**: `.1/.2/.3` = sequential sub-steps (must run in order). `A/B/C` = parallel scripts (can run simultaneously).

The optimizer runs as an 8-step pipeline (Steps 0–7). Step 1 is expensive (hours). Steps 2–7 are cheap (seconds to minutes). Only re-run what changed.

**Data directories**: `data/step1-pfs/` (Step 1), `data/step2.1-ef/` (Step 2.1), `data/step2.2-cost/` (Step 2.2), `data/step3-dispatch/` (Step 3), `data/step4-analysis/` (Step 4), `data/step5-scenarios/` (Step 5), `data/step5-wrights/` (Step 5.2E), `data/step6-smartargets/` (Step 6).

#### Step 0: Data Acquisition

`step0_*.py` (9 scripts, annual cadence) — EIA hourly profiles (2021–2025), eGRID emissions, actual LMP data, offshore wind (NREL NOW-23), UTC fixes, MISO/SPP consolidation.

#### Step 1: Physics Feasible Space (sequential: 1.1 → 1.2 → 1.3 → 1.4 → 1.5)

- **Monolithic**: `step1_pfs_generator.py` — runs full PFS generation in one process.
- **Modular (CI/CD)**:

| Sub-step | Script(s) | What It Does |
|----------|-----------|-------------|
| **1.1** | `step1_1a_generate_mixes.py` → `step1_1b_score_mixes.py` | Generate 4D/5D resource combos, score against hourly demand → coarse cache. |
| **1.2** | `step1_2_zone_search.py` | Zone-specific fine search around promising PFS regions per threshold. |
| **1.3** | `step1_3_floor_aware_pfs.py` | Floor-aware PFS augmentation (50-80% range). |
| **1.4** | `step1_4_fine_grid_pfs.py` | Fine-grid PFS (40-75% range). |
| **1.5** | `step1_5_storage_refinement.py` | Storage dispatch refinement (battery/LDES/H2 grid search). |

Utilities: `step1_prior_windows.py` (search windows from prior EF results).
Output: `data/step1-pfs/`. **Only re-run if dispatch logic, generation profiles, or demand curves change.**

#### Step 2: Optimization (sequential: 2.1 → 2.2)

| Sub-step | Script(s) | What It Does | When to Re-run |
|----------|-----------|-------------|---------------|
| **2.1** | `step2_1_efficient_frontier.py` | **Efficient Frontier** — Non-dominated mix extraction. Output: `data/step2.1-ef/`. | PFS or filtering criteria change. |
| **2.2A** | `step2_2a_cost_optimization.py` | **Track 1 baseline** — 5,832 sensitivity combos (17,496 CAISO). Merit-order tranche pricing. ─┐ parallel | Cost assumptions change. |
| **2.2B** | `step2_2b_track_nb_ctr.py` | **Track 2 (NB) + Track 3 (CTR)** — Demand growth × FOAK→NOAK learning curves. ─┘ | Cost assumptions change. |

Output: `data/step2.2-cost/`.

#### Step 3: Caches (parallel: 3A ∥ 3B)

| Script | What It Does | Dependencies |
|--------|-------------|-------------|
| `step3a_build_dispatch_cache.py` | **Dispatch cache** — 8,760-hour dispatch for all unique mixes. NPZ v2. Output: `data/step3-dispatch/`. | Step 2.2 |
| `step3b_mac_queue.py` | **MAC queue** — Path-dependent consequential deployment queue. Output: `data/step3-dispatch/`. | Step 1 only (NOT Step 2) |

#### Step 4: Analysis (two tiers: 4.1 parallel → 4.2 parallel)

**Tier 4.1** (parallel, mixed deps on Steps 2/3):

| Script | What It Does | Dependencies |
|--------|-------------|-------------|
| `step4_1a_fossil_dispatch.py` | **CO₂ + LMP** — Fossil dispatch-stack, merit-order retirement, synthetic hourly LMP. | 3A |
| `step4_1b_compress_day_profiles.py` | **Day profiles** — 24-hour representative day from dispatch cache. | 3A |
| `step4_1c_compute_mac_stats.py` | **MAC stats** — 6 metrics, ANOVA decomposition. Crossover vs DAC/SCC/ETS. | Step 2.2 |
| `step4_1d_compute_optimal_targets.py` | **Optimal targets** — Marginal MAC × DAC crossover (PCHIP spline). 3×3 matrix. | Step 2.2 |
| `step4_1e_export_tracks.py` | **Track export** — NB + CTR parquets → `track_results.json`. | Step 2.2 |
| `step4_1f_extract_building_blocks.py` | **Building blocks** — Resource decomposition per ISO/threshold. | Step 2.2 |

**Tier 4.2** (parallel, needs specific 4.1 outputs):

| Script | What It Does | Dependencies |
|--------|-------------|-------------|
| `step4_2a_extract_resource_density.py` | **Resource density** — Density analysis across thresholds. | 4.1D |
| `step4_2b_analyze_storage.py` | **Storage analysis** — Battery/LDES utilization, dispatch patterns, capacity factors. | Step 2.2 + 4.1B |
| `step4_2c_analyze_tracks.py` | **Track analysis** — Cost envelopes (P10/P50/P90), resource mix differentials. | 4.1E |

#### Step 5: Scenarios & Procurement (sequential 5.1 → parallel 5.2)

| Script | What It Does | Dependencies |
|--------|-------------|-------------|
| `step5_1_scenario_hourly.py` | **Scenario B** — Hourly matching procurement strategy. | 4.1D (optimal targets) |
| `step5_2a_scenario_comparison.py` | **Scenario comparison** — Consequential vs. hourly. ─┐ | 3B + 5.1 |
| `step5_2b_strategy_consequential.py` | **Strategy 1 (A/B/C)** — Cross-regional consequential netting. │ | 3B |
| `step5_2c_strategy_hourly.py` | **Strategy 2 (A/B/C)** — Hourly matching same-ISO. │ parallel | 5.1 |
| `step5_2d_strategy_annual.py` | **Strategy 3 (A/B/C/D)** — Annual matching 2×2 matrix. │ | Step 2.2 |
| `step5_2e_wrights_law_curves.py` | **Wright's Law** — Learning curves & critical mass. ─┘ | Independent |

Shared utilities: `procurement_utils.py`.

#### Step 6: Policy Analysis (sequential 6.1 → parallel 6.2)

| Script | What It Does | Dependencies |
|--------|-------------|-------------|
| `step6_1_smartargets.py` | **Regional SMARTargets** — Market simulation per ISO/scenario. | Step 2.2 + 4.1A (LMP) |
| `step6_1b_dashboard_data.py` | **SMARTargets data export** — Parquets → dashboard JS. | 6.1 |
| `step6_2a_ipp_smartargets.py` | **IPP modeling** — Fleet economics under SMARTargets. ─┐ parallel | 6.1 |
| `step6_2b_nuclear_retirement.py` | **Nuclear retirement** — Stranding risk analysis. ─┘ | 6.1 + Step 2.2 |

#### Step 7: Dashboard Aggregation (parallel 7.1 → sequential 7.2)

| Script | What It Does | Dependencies |
|--------|-------------|-------------|
| `step7_1a_generate_shared_data.py` | **Main export** → `dashboard/js/shared-data.js`. SBTi mapping, DAC projections. ─┐ | All upstream |
| `step7_1b_extract_deployment_data.py` | **Deployment data** for dashboard. │ | Steps 3B, 5 |
| `step7_1c_generate_foak_noak.py` | **FOAK/NOAK** learning curve data. │ parallel | Step 2.2 |
| `step7_1d_generate_building_blocks.py` | **Building blocks** dashboard data. │ | Step 4 |
| `step7_1e_dispatch_deployment.py` | **Dispatch deployment** visualization data. │ | Step 3A |
| `step7_1f_extract_hourly_comparison.py` | **Hourly comparison** data. │ | Step 5 |
| `step7_1g_extract_use_case_data.py` | **Use case** analysis data. ─┘ | Steps 4-6 |
| `step7_2_extract_no_regrets.py` | **No-regrets analysis** — Optimal targets from crossover. | 7.1A |

#### Shared Utility Modules

| Script | What It Does |
|--------|-------------|
| `pipeline_config.py` | **Config.** Single source of truth for all shared constants (LCOE, fuel, CCS, storage, wholesale prices). |
| `dispatch_utils.py` | **Dispatch engine.** Dispatch reconstruction, supply profiles, fossil retirement, cache I/O (v2 NPZ). |
| `scenario_common.py` | **Scenario utilities.** Shared Scenario A/B logic: cost tables, demand growth, learning curves, EF/PFS loading. |
| `eia_data_io.py` | **EIA data I/O.** Standardized multi-year EIA generation/demand profile loading. |
| `calibrate_lmp_model.py` | **LMP calibration.** Validates synthetic LMP against actual ISO data (2024 SOM reports). |

#### Pipeline Execution Order

```
Step 0 (data acquisition, annual)
  → Step 1 (1.1 → 1.2 → 1.3 → 1.4 → 1.5: physics PFS)
    → Step 2 (2.1 EF → 2.2A+2.2B cost optimization)
      → Step 3A (dispatch cache) ∥ Step 3B (MAC queue, needs Step 1 only)
        → Step 4.1 (4.1A–4.1F: core analysis, all parallel)
          → Step 4.2 (4.2A–4.2C: derived analysis, all parallel)
            → Step 5.1 (scenario hourly) → Step 5.2A–E (scenarios + procurement, parallel)
            → Step 6.1 (SMARTargets) → Step 6.2A+6.2B (IPP + nuclear, parallel)
              → Step 7.1A–G (dashboard data, parallel) → Step 7.2 (no-regrets)
```

**Key principle**: Step 1 is expensive (hours). Step 2.1 ~40s. Steps 2.2–7 cheap (seconds to minutes). Changing cost assumptions → Step 2.2 + downstream only. Changing a single analysis → relevant step + Step 7.

#### GitHub Actions Workflows

All workflows: `workflow_dispatch`, ISO selectors. See `.github/workflows/README.md` for full docs.

| Step | Workflow | What It Does |
|------|----------|-------------|
| 0 | `step0-fetch-lmp-data.yml` | Fetch actual DA hourly LMP |
| 0 | `step0-fetch-offshore-wind.yml` | Fetch NREL NOW-23 offshore wind profiles |
| 1.1 | `step1-1-scored-database.yml` | Generate + score resource combos |
| 1.2–1.3 | `step1-2-3-zone-floor.yml` | Zone search + floor-aware PFS |
| 1.4–1.5 | `step1-4-5-fine-storage.yml` | Fine-grid PFS + storage refinement |
| 2.1 | `step2-1-efficient-frontier.yml` | EF extraction |
| 2.2 | `step2-2-cost-optimization.yml` | Cost optimization (3 tracks) |
| 3A | `step3a-dispatch-cache.yml` | Dispatch cache build |
| 3B | `step3b-mac-queue.yml` | MAC queue (parallel with 3A) |
| 4.1B | `step4-1b-day-profiles.yml` | Compressed day profiles |
| 4 | `step4-tracks.yml` | Track export + analysis (4.1E + 4.2C) |
| 4.2B | `step4-2b-storage-analysis.yml` | Storage dispatch analysis |
| 4 | `step4-derived-analytics.yml` | MAC stats + optimal targets + building blocks + resource density |
| 5 | `step5-scenarios.yml` | Scenarios (5.1 hourly → 5.2A comparison) |
| 5 | `step5-procurement.yml` | Procurement strategies (5.2B–D) |
| 5.2E | `step5-2e-wrights-law.yml` | Wright's Law learning curves |
| 6.1 | `step6-1-smartargets-reference.yml` | SMARTargets — Reference scenario |
| 6.1 | `step6-1-smartargets-power-nz.yml` | SMARTargets — Power Sector NZ |
| 6.1 | `step6-1-smartargets-economy-nz.yml` | SMARTargets — Economy-Wide NZ |
| 6.1 | `step6-1-smartargets-quick-transition.yml` | SMARTargets — Quick Transition |
| 7 | `step7-dashboard-data.yml` | Dashboard data export |

**Key acronyms**:
- **PFS** — Physics Feasible Space: all physically valid resource mixes (Step 1 output, `data/step1-pfs/`)
- **EF** — Efficient Frontier: non-dominated mixes optimal under some cost assumption (Step 2.1 output, `data/step2.1-ef/`)

**Data contract**: Step 2.2 must NOT change existing columns in shared-data.js or overprocure_results.json. Add new columns/fields only. This prevents recoding existing figures and dashboards.

### What was accomplished
- [x] Homepage (`index.html`) — 4 charts rendering with real data, region toggle pills, narrative sections
- [x] Carbon Abatement Dashboard (`abatement_dashboard.html`) — 3 charts (MAC, portfolio, ladder) fully rendering with hardcoded illustrative data + 4 stress-test toggles
- [x] Navigation site-wide: Home | Cost Optimizer | Analysis (CO₂ Abatement Analysis) | Research (Paper, Methodology, Policy, About)
- [x] "Back to Home" button on all non-home pages
- [x] Chart styling QA/QC on working charts (borderRadius 6, no grid lines, axis borders)
- [x] Merged methodology into research paper (Appendix B with 7 sub-sections)
- [x] Tagline: "Most climate solutions depend on" across all pages
- [x] **CO₂ methodology fixed**: Hourly fossil-fuel emission rates (eGRID per-fuel × EIA hourly mix) replacing flat rate
- [x] **Post-optimizer pipeline**: `step5_compute_co2.py`, `analyze_results.py`, `run_post_optimizer.sh`
- [x] **Multi-year data infra**: `fetch_eia_multiyear.py` (2021-2025 EIA API + DST + averaging)
- [x] **Phase 3 re-optimizer**: `optimize_phase3_only.py` (±5% neighborhood refinement)
- [x] **DST fix script**: `fix_dst_profiles.py` (UTC → local prevailing time conversion)
- [x] **Optimizer checkpointing**: Saves after each threshold + resume from checkpoint on restart
- [x] **Sequential ISO processing**: Runs one ISO at a time with incremental result saves (avoids OOM)

### Needs work (awaiting optimizer results)
- [ ] **Dashboard abatement section (`dashboard.html`)** — 5 paired toggles work, 4 core charts work. Abatement cost section has placeholder divs awaiting optimizer results data.

### Optimizer code — ready for run (all completed)
- [x] **Hydro caps** — 2025 actuals: CAISO 9.5%, ERCOT 0.1%, PJM 1.8%, NYISO 15.9%, NEISO 4.4%
- [x] **5-year profile averaging** — gen + demand shapes averaged 2021-2025 (leap year handled)
- [x] **DST-aware solar nighttime correction** — 6am-7pm local prevailing time, UTC offset adjusts during DST
- [x] **Nuclear seasonal derate** — monthly CF factors × nuclear share of clean_firm
- [x] **Nuclear uprate LCOE blending** — regional uprate share blends cheap uprates with new-build
- [x] **CCS 95% capture rate** — residual 0.0185 tCO2/MWh
- [x] **Capacity-constrained storage** — battery + LDES dispatch_pct = built capacity
- [x] **CO2 hourly dispatch attribution** — charge-side emission netting
- [x] **1-scenario checkpointing** — zero compute loss on interruption
- [x] **Wholesale fuel adjustments** — documented in §5.9 with per-ISO $/MWh table
- [x] **SPEC.md ↔ code audit** — 150+ values verified, 0 discrepancies

### Needs work (awaiting optimizer results)
- [ ] Dashboard abatement section (`dashboard.html`) — placeholder divs awaiting results
- [ ] Site content gap closure: incomplete pages need optimizer data
- [ ] Update narratives + research paper with new results

### MAC Methodology Decision (Feb 16, 2026) — Option B: Hybrid Stats + Path-Constrained Reference

**Problem**: Independent threshold optimization produces non-monotonic marginal abatement costs (MACs). Each threshold finds its own globally optimal portfolio; the "marginal cost" between thresholds compares two independently optimized systems, not incremental resource additions. This creates phantom MAC spikes ($887/ton NYISO 95→97.5%) and even negative abatement (NEISO 95→97.5%, ERCOT 97.5→99%) when the optimizer reshuffles the portfolio.

**Decision**: Option B — Hybrid approach combining statistical post-processing of existing 16,200 results with a small set of path-constrained optimization runs.

**Components**:
1. **Monotonic envelope** (stats): Convex hull of (cost, CO2) per ISO — filters rebalancing noise
2. **MAC uncertainty fan** (stats): P10/P50/P90 across 5,832 scenarios at each threshold
3. **Sensitivity decomposition** (stats): ANOVA on which toggles drive MAC variance
4. **Path-constrained reference MAC** (50 targeted runs): Force each threshold's mix to build on previous — monotonic by construction. One run per threshold × 5 ISOs at Medium costs.
5. **Visualization**: Central monotonic reference curve inside P10-P90 fan, with DAC/SCC horizontal bands

**Methodology statement**: "Path-dependent marginal costs (central line) with uncertainty characterized via factorial sensitivity analysis across 5,832 cost scenarios (shaded band)."

**Literature basis**: Systems MAC / MAC 2.0 (Evolved Energy/EDF 2021), scenario ensemble approach (Deane et al. 2020), conservation supply curve methodology (Meier & Rosenfeld 1982). Full lit review: `research/mac_methodology_lit_review.md`.

**Reference docs**:
- `research/mac_methodology_lit_review.md` — Full literature review with 17 key citations
- `research/optimizer_statistical_methodology.md` — Search space analysis, global optimum capture probability

### Marginal MAC Monotonicity Fix (Feb 16, 2026) — Two-Zone Approach

**Problem**: Stepwise marginal MAC (Δcost/ΔCO2 between consecutive thresholds) is wildly non-monotonic due to resource reshuffling. Current data oscillates by 2-10x between adjacent steps (e.g., CAISO P50: 214→116→475→138→290→305→347→340). Root cause: independent threshold optimization produces different optimal portfolios at each threshold — the "delta" between them measures portfolio switching cost, not incremental resource addition cost.

**Key insight**: Grid decarbonization holds to ~92.5% in all regions regardless of cost assumptions. Sub-90% marginal MAC granularity is noise from optimization artifacts, not economically meaningful signals.

**Decision**: Two-zone marginal MAC structure:

**Zone 1 — Grid Backbone (75% → 90%): Single aggregate marginal MAC**
- One value per (ISO, scenario): `MAC = (cost[90%] - cost[75%]) × demand / (CO2[90%] - CO2[75%])`
- Represents the cost per ton of grid backbone decarbonization
- No monotonicity issue (single value)

**Zone 2 — Last Mile (90% → ≥99.9%): Granular checkpoints with enforced monotonicity**
- 5 stepwise values: 90→92.5%, 92.5→95%, 95→97.5%, 97.5→99%, 99→99.5%, 99.5→≥99.9%
- Enforced non-decreasing: `step_mac[t] = max(raw_step_mac[t], step_mac[t-1])`
- Zone 1 aggregate MAC serves as floor for first Zone 2 step
- Convex hull interpolation for edge cases where ΔCO2 ≤ 0

**Result**: 6-value marginal MAC curve per (ISO, scenario):
```
[MAC_75→90, MAC_90→92.5, MAC_92.5→95, MAC_95→97.5, MAC_97.5→99, MAC_99→99.5, MAC_99.5→≥99.9]
```

**Fan chart fix**: Consistent scenario ranking (rank by total cost at 99%, select P10/P50/P90 scenarios, use their full curves) instead of independent per-step percentiles that mix different scenarios.

**Implementation plan**: See `PLAN_marginal_mac_fix.md` for detailed implementation steps and file-by-file changes.

### MAC Formula Change (Feb 25, 2026) — Full Portfolio LCOE, Not Incremental Above Wholesale

**Previous formula**: `MAC = (cost_incremental × demand) / CO₂_abated` where `cost_incremental = effective_cost - wholesale_price`. This measured the premium of clean energy over grid wholesale power per ton abated.

**New formula**: `MAC = (cost_effective_cost × demand) / CO₂_abated`. Uses the full portfolio LCOE (effective cost) rather than the incremental cost above wholesale.

**Rationale**:
- MAC should measure the standalone cost-effectiveness of the clean portfolio per ton of CO₂ displaced, not a premium relative to a wholesale baseline
- Removes wholesale price sensitivity from MAC (previously, higher wholesale prices made MAC look artificially low)
- Aligns with standard MAC curve methodology: cost of the abatement action itself
- CO₂ abatement continues to use merit-order fossil retirement (coal → oil → gas) from the dispatch model in `dispatch_utils.py`

**Impact**: MAC values will be higher across the board (roughly 2× for regions where wholesale ≈ effective_cost/2). DAC/SCC/ETS benchmark comparisons remain unchanged. Crossover thresholds will shift. Dashboard narrative text must be updated after PP5 re-run.

**Files changed**: `scripts/step5_compute_mac_stats.py` — all MAC computation functions (`add_mac_column`, `compute_stepwise_fan`, `compute_monotonic_envelope`, `compute_path_constrained_mac`, demand-growth MAC) now use `cost_effective_cost` instead of `cost_incremental`. No changes to Steps 1-4 or parquet outputs needed — only PP5 re-run required.

### FOAK→NOAK Learning Curve — Scenario-Differentiated, Year-Based (Feb 25, 2026)

**Decision**: Replaced single learning curve with scenario-differentiated, year-based curves. Both scenarios now have learning curves, but with different start dates and durations reflecting deployment pace differences.

**Rationale**: INL data shows SOAK (2nd unit) achieves 15% reduction, units 2-4 another 5%, before gradual decline toward NOAK. NEA shows -18 to -25% for unit 2, -25 to -40% by unit 4, -35 to -55% by unit 8. DOE Liftoff projects NOAK by early 2030s for advanced designs, scaled achievement by 2035, full low-price stabilization by 2040. Learning occurs sequentially and in compressed timelines — construction learning transfers through engineering teams, supply chains, and regulatory streamlining before each unit completes. Westinghouse plans 10 new AP1000s by 2030; commercial SMRs expected operational by 2030.

**Scenario B (Hourly Matching — aggressive deployment):**
- FOAK starts: 2029 (first commercial SMR deployments)
- Learning period: 2028-2038 (10 years)
- NOAK achieved: 2038
- 2038-2050: Stable at NOAK (Low) pricing
- Shape: Concave (exponent 0.6) — steep front-end matching INL/NEA unit-doubling data, asymptotic tail
- Pre-2029: Pure FOAK (no learning, no SMR deployment yet)

**Scenario A (Pure Consequential — delayed deployment):**
- FOAK starts: 2036 (5-year delay due to less investment, slower regulatory pathway)
- Learning period: 2036-2048 (12 years — stretched due to fewer units built per year)
- NOAK achieved: 2048
- 2048-2050: Stable at NOAK (Low) pricing
- Shape: Concave (exponent 0.6) — same steep front-end physics, stretched across 12 years
- Pre-2036: Pure FOAK

**Implementation**: `learning_fraction(threshold, scenario='B')` now uses year-based lookup via SBTI_YEAR_MAP. Each scenario defines `foak_start_year` and `noak_year`. Fraction is 0 (pure FOAK) before start, concave ramp during learning period, 1.0 (full NOAK) after NOAK year. Both `_adjust_costs_with_learning` (Scenario B) and `_adjust_costs_no_learning` → `_adjust_costs_delayed_learning` (Scenario A) use the same interpolation machinery with different timeline parameters.

**Nuclear LCOE trajectory (PJM, H=$160 → L=$72):**

Scenario B:
| Year | Threshold | Fraction | LCOE |
|------|-----------|----------|------|
| 2030 | 50% | 0.38 | $126 |
| 2031 | 55% | 0.49 | $117 |
| 2033 | 60% | 0.66 | $102 |
| 2035 | 70% | 0.81 | $89 |
| 2037 | 80% | 0.94 | $77 |
| 2038 | 85% | 1.00 | $72 |
| 2040 | 90% | 1.00 | $72 |
| 2045 | 95% | 1.00 | $72 |
| 2050 | 100% | 1.00 | $72 |

Scenario A:
| Year | Threshold | Fraction | LCOE |
|------|-----------|----------|------|
| 2030 | 50% | 0.00 | $160 |
| 2035 | 70% | 0.00 | $160 |
| 2036 | 75% | 0.00 | $160 |
| 2037 | 80% | 0.23 | $140 |
| 2038 | 85% | 0.34 | $130 |
| 2040 | 90% | 0.52 | $114 |
| 2045 | 95% | 0.84 | $86 |
| 2048 | 97.5% | 1.00 | $72 |
| 2050 | 100% | 1.00 | $72 |

**Supersedes**: Previous single learning curve (concave ramp, no learning below 70%, same curve for both scenarios). Scenario A previously had NO learning curve (flat FOAK forever).

**Scope**: PP3 scenario comparison only. Step 3 cost optimization is NOT modified — it remains the 2025 snapshot with static LCOE tables.

**Files changed**: `scripts/step6_scenario_comparison.py` — `learning_fraction()`, `_adjust_costs_no_learning()` → `_adjust_costs_delayed_learning()`. PP3 re-run required.

### Compressed Day Chart — Curtailment Double-Count Fix (Feb 25, 2026)

**Decision**: Fix the compressed day chart to:
1. Show "Total Generation" line (total clean energy output before curtailment) instead of "Demand" line
2. Net out storage charging from displayed surplus to eliminate double-count

**Problem**: Per-resource surplus arrays included energy absorbed by storage (battery/LDES charging). That same energy was also shown as negative charging bars below the x-axis — double-counting the same energy.

**Fix**:
- Compute net curtailment factor: `(grossSurplus - totalCharging) / grossSurplus` per hour
- Apply proportionally to each resource's surplus so only TRUE curtailment (not stored) is shown as hatched area
- Replace "Demand" line with "Total Generation" = sum of primary resource matched + gross surplus (total clean output before curtailment/storage)

**Files changed**: `dashboard/dashboard.html` — compressed day chart rendering. No PP1 changes needed.

### Annual Cost + 25-Year NPV Metrics (Feb 25, 2026)

**Decision**: Add `annual_cost_billion` ($B/yr) and `npv_25yr_billion` ($B, 25-year NPV) to PP3 trajectory entries alongside existing $/MWh metrics.

**Formulas**:
- `annual_cost_billion = effective_cost ($/MWh) × demand_twh / 1000`
- `npv_25yr_billion = annual_cost × annuity_factor(5%, 25)` where annuity factor ≈ 14.09
- Uses 5% real WACC (standard utility/infrastructure discount rate)

**Rationale**: $/MWh is useful for comparison but doesn't convey scale. Annual $B shows the absolute commitment at each threshold. 25-year NPV shows the total investment required, useful for investment framing and policy cost-benefit analysis.

**Files changed**: `scripts/step6_scenario_comparison.py` — trajectory entry construction. PP3 re-run required.

### Gas Capacity Costs — Already Integrated (Feb 25, 2026)

**Status**: Gas backup capacity costs (both existing FOM and new CCGT build) are already fully integrated into total system cost in PP3's `compute_mix_cost()`. New CCGT costs range from $88-114/kW-yr by ISO. No additional changes needed.

### Optimizer Statistical Properties (Feb 16, 2026)

**Search architecture**: 3-phase hierarchical grid search (10% → 5% → 1% resolution)

**Global optimum capture probability**: >99.9%
- Phase 1 coarse grid covers all 32 piecewise-linear regions of the cost function
- P(all regions sampled) ≈ 99.5% from grid alone, >99.9% with edge-case seeds
- Lipschitz gap bound (Nesterov 2003): <$0.01/MWh in mix dimensions at 1% resolution

**Maximum sub-optimality**: ~$2-4/MWh (~1-3% of typical $50-150/MWh total)
- Mix dimensions (1% steps): <$0.58/MWh
- Storage dimensions (2% steps): <$3.64/MWh — dominant error source
- Procurement (1% steps): <$1.00/MWh

**Why grid search, not LP**: Non-convex problem (hourly min() in matching score, nonlinear storage dispatch). Standard energy models (TIMES, ReEDS, GenX, PyPSA) use LP, but our hourly matching + greedy storage dispatch cannot be linearized without accuracy loss. Grid search is appropriate because evaluations are cheap (~0.1ms vectorized numpy) and dimensionality is low (5-7 DOF).

**Warm-start bias**: Non-Medium scenarios start from Medium optimum + ±17pp reach. 4 extreme archetypes get full exploration. Cross-pollination covers remaining risk.

### Pipeline when optimizer completes
1. Run `step5_compute_co2.py` → hourly CO₂ correction
2. Run `analyze_results.py` → monotonicity, literature alignment, VRE waste, DAC inputs
3. Update dashboards with real data, update narratives
4. Path-constrained MAC runs (50 targeted optimizations)
5. Statistical post-processing: envelope, fan, ANOVA
6. DAC-VRE analysis, resource mix analysis
7. Commit + push

### Pre-Run QA/QC Gate (Mandatory Before Every Optimizer Run)
**This gate exists because**: a previous run wasted 3+ hours of compute due to incorrect hydro caps that weren't caught before launch. Every optimizer run is expensive — never launch without verifying assumptions first.

Before launching `step1_pfs_generator.py`, the following must be verified:
1. All decisions from the current conversation implemented in optimizer code
2. All decisions captured in SPEC.md
3. No open questions that could change optimizer logic, cost tables, or methodology
4. Code passes syntax check (`python -c "import py_compile; py_compile.compile(...)"`)
5. **Full assumptions audit**: verify ALL key assumptions (hydro caps, cost tables, resource constraints, dispatch logic, procurement bounds, storage parameters) match SPEC.md and real-world data
6. **Dry-run test**: imports, constants, data loading, checkpoint save/load round-trip
7. **Checkpoint system verified**: save/load/resume works, interval set appropriately
8. Present user with summary of verified assumptions before starting
9. User explicitly approves the run

### Generator Analysis & Policy Page Decisions (Feb 15)
- [x] **Tone down Constellation-specific narrative** — generic archetypes (nuclear-led, coal-heavy, gas-dominant). Removed "unfairness" language. Applied across targets.html, index.html, fleet-analysis.html, policy.html.
- [x] **Add GHG Protocol Scope 2 revision context** — deep-dive on targets.html: 4 quality criteria (temporal, deliverability, incrementality, SSS), hourly premium economics, convergence with SBTi.
- [x] **Add EPRI SMARTargets context** — targets.html: AT/QT framework, Ceres criticism, investor credibility debate, "both/and" resolution.
- [x] **Add hourly RPS discussion** — targets.html: hourly RPS as policy frontier, convergence with GHG Protocol + SBTi, demand-side pull for clean firm.
- [x] **Policy page: RPS + corporate demand under hourly matching** — SSS baseline, corporate participation scenario table (10-50% × 5 ISOs), clean premium projections by ISO.
- [x] **EAC scarcity analysis REWRITE (Feb 15)** — corrected SSS framework + interactive dashboard
- [x] **SSS pro-rata derate** — corporate EAC demand is incremental above SSS baseline allocation, not gross
- [x] **Demand-proportional RPS** — clean supply = RPS target % × projected demand (not independent growth rate)
- [x] **Two-component SSS** — fixed-fleet (nuclear, hydro — constant TWh) + RPS (scales with demand)
- [x] **Diablo Canyon + NY nuclear as permanent fixed SSS** — state-supported indefinitely

### EAC Scarcity: Combined RPS + Voluntary Supply Stack Model (Feb 15, rev 2)
**Decision**: RPS mandates and voluntary corporate procurement compete for the same finite buildable clean capacity. Marginal cost is set by combined demand (RPS + voluntary) on the supply stack, not voluntary alone.

**Literature validation** (Xu et al. 2024, Joule / Princeton ZERO Lab): GenX capacity expansion model shows combined RPS + voluntary C&I demand on the same regional supply curve produces non-linear cost escalation as both compete for finite buildable capacity. Gillenwater (2008, Energy Policy): only when combined demand creates real scarcity does voluntary procurement drive new investment. Denholm et al. (NREL 2021): "last few percent" costs escalate exponentially.

**Previous approach (superseded, v1)**: RPS adder as economic gate — new clean only entered when LCOE < wholesale + RPS adder. This produced zero new build in all ISOs because the price signal never cleared any supply stack tier. Supply was frozen at 2025 levels.

**Current approach (v2) — two-track demand, unified supply stack**:

1. **RPS-mandated demand** (forced, regardless of economics):
   - `rps_new_need = max(0, rps_target × projected_demand - existing_total_clean)`
   - Regulators require this build — it happens whether or not LCOE < wholesale + adder
   - New RPS build splits into SSS vs merchant per `SSS_NEW_BUILD_FRACTION`

2. **Voluntary corporate demand** (additional, on top of RPS):
   - `corp_eac_demand = CI_share × participation × incremental_need × procurement_ratio`
   - Incremental need = `max(0, match_target% - sss_share_of_total%)`
   - SSS share grows over time as RPS mandates add clean capacity

3. **Combined demand on supply stack**:
   - `total_new_demand = rps_new_need + corp_eac_demand`
   - Walk up the supply stack (cheapest tier first) until total demand is met
   - RPS compliance absorbs cheap tiers first; corporates ride on top
   - **Marginal cost = LCOE of the tier where combined demand lands**
   - If combined demand exceeds total buildable capacity → scarcity pricing

4. **Scarcity classification**:
   - `demand_ratio = total_new_demand / total_buildable_capacity`
   - Bands: Abundant (<0.3), Adequate (<0.6), Tightening (<0.8), Scarce (<0.95), Critical (>0.95)

5. **Clean premium = marginal LCOE - wholesale price**
   - Reflects the REAL competition for clean resources — both mandated and voluntary

**Bug fixes in v2**:
- **SSS→non-SSS transfer**: When SSS policies expire (e.g., IL ZEC/CMC 2027), those TWh move to non-SSS merchant pool — not into the void. Plants don't disappear when subsidies end.
- **Annual resolution**: All years 2025–2050 (26 years, not just 6 milestone years)

**Supply stack per ISO** (static 2025 LCOEs, no decline curves — deliberate simplification):
- Resources ordered by LCOE from optimizer config
- Each tier has annual buildable TWh and cumulative max (from LBNL "Queued Up")
- No LCOE decline or wholesale escalation modeled — avoids overcomplication

**Procurement ratio** (theoretical, not optimizer-derived):
- 75%→0.80×, 90%→1.05×, ≥99.9%→1.45×
- Reflects temporal mismatch physics: higher match targets need more over-procurement

**What stays from v1**:
- SSS/non-SSS classification, two-component SSS (fixed + RPS), policy expirations
- C&I demand filter (62%), demand growth rates, participation scenarios
- Scarcity bands, supply stack LCOEs, committed hyperscaler pipeline
- Wholesale prices, RPS target trajectories

### EAC Scarcity: C&I Demand Filter (Feb 15)
**Decision**: Corporate EAC participation base = C&I (commercial + industrial) share of total demand, not total demand. Residential load does not participate in voluntary EAC procurement.

**C&I share**: ~62% of total demand (EIA 2024 national average: 38% residential, 36% commercial, 26% industrial). Applied as a flat multiplier across all ISOs and demand growth scenarios.

**RPS stays against total demand**: RPS mandates apply to total retail sales (including residential), so RPS/SSS calculations continue to use full demand. Only the voluntary corporate procurement base is filtered to C&I.

**Limitation (noted)**: C&I share held constant across demand growth scenarios. In practice, data center growth (classified as commercial by EIA) could shift C&I share higher over time, particularly in PJM and ERCOT. This simplification is acknowledged but not modeled.

### EAC Scarcity: Hyperscaler Committed Nuclear PPA Pipeline (Feb 15)
**Decision**: Model committed hyperscaler nuclear PPAs as a phased supply reduction rather than generic demand growth. Hyperscaler data center demand is disproportionately clean-energy-focused — these PPAs lock up specific clean generation that is no longer available for other corporate procurement.

**PJM committed pipeline**: ~4 GW nuclear PPAs committed by hyperscalers:
- Amazon-Talen: Susquehanna campus (~960 MW, operational)
- Microsoft-Constellation: TMI Unit 1 restart (~835 MW, targeting 2028)
- Other committed deals ramping through 2030

**Phasing** (cumulative GW online → TWh/yr at 90% CF):
- 2025: 1.0 GW → ~7.9 TWh (Susquehanna campus + early deals)
- 2027: 2.0 GW → ~15.8 TWh
- 2028: 3.0 GW → ~23.7 TWh (TMI restart)
- 2030: 4.0 GW → ~31.5 TWh (full pipeline)

**Implementation**: Subtracted from available non-SSS supply alongside existing corporate PPAs. Modeled as `COMMITTED_CLEAN_PIPELINE` with time-phased GW → TWh conversion. Applies only to PJM currently (can be extended to other ISOs as hyperscaler commitments are announced in those markets).

**Why supply reduction, not demand growth**: Generic demand growth is diluted by the C&I share (62%) and mixed across all electricity sources. Hyperscaler nuclear PPAs specifically target and lock up clean generation — modeling as supply reduction correctly captures that these MWh are spoken for by specific off-takers.

### Corrected SSS Framework (Feb 15)
**SSS = mandatory/non-bypassable procurement creating a financial relationship between customers and generation.** Determined by whether a policy acts upon the EAC:
- **RPS/CES mandates** — state renewable/clean energy standards that retire EACs on behalf of ratepayers
- **Public ownership** — municipal utilities, federal power agencies (NYPA, BPA, TVA, WAPA)
- **Vertically integrated / rate-base assets** — utility-owned generation in regulated territories (Dominion VA plants)
- **State nuclear programs** — ZEC, CMC, or CES programs that retire nuclear EACs (NY ZEC, IL ZEC/CMC, CT Millstone PPA)

**What is NOT SSS:**
- **45U Production Tax Credit** — does not act on EAC, designed to decrease at higher revenues, credit rolls off if clean premium increases
- **Merchant nuclear** — plants not in state programs are fair game for corporate procurement (LaSalle, Calvert Cliffs, Limerick, Peach Bottom)
- **Merchant renewables** — new-build wind/solar in ERCOT or other deregulated markets without RPS obligation
- **Corporate PPAs** — voluntary, not mandatory; reduce available supply but are not SSS

**SSS is temporal** — state programs expire:
- **IL ZEC/CMC**: expires mid-2027. Dresden, Braidwood, Byron, LaSalle, Clinton, Quad Cities (~94 TWh) shift from SSS to non-SSS/PPA
- **NJ ZEC**: expired June 2025. Salem + Hope Creek (~27 TWh) already non-SSS
- **NY ZEC**: extended through 2049. All 4 NYISO plants remain SSS
- **Diablo Canyon**: state extension through 2030, NRC renewal sought to 2045. Uncertain post-2030
- **CT Millstone PPA**: ~half of output under CT auction PPA. Remainder merchant

**Key implication**: Existing merchant nuclear is available for corporate procurement. Corporations CAN buy nighttime nuclear EACs in PJM. But data center PPAs (Amazon-Susquehanna, Meta-Vistra plants, Microsoft-Crane) are rapidly consuming this supply.

**National SSS estimates (2025):**
| ISO | Total Clean (TWh) | SSS (TWh) | Non-SSS (TWh) | SSS % |
|---|---|---|---|---|
| PJM | ~280 | ~150-180 | ~100-130 | ~57% |
| ERCOT | ~205 | ~20-25 | ~180-190 | ~12% |
| CAISO | ~172 | ~140-155 | ~17-32 | ~85% |
| NYISO | ~60 | ~49-55 | ~5-11 | ~85% |
| NEISO | ~50 | ~25-30 | ~15-20 | ~55% |

**Scarcity analysis parameters (expanded):**
- Corporate participation: 0-100% of ISO load
- Hourly match target: 75-100%
- Time horizons: 2025–2050 (annual, 26 years)
- Demand growth: Low/Med/High per ISO (from dashboard DEMAND_GROWTH_RATES)
- SSS supply evolves over time (policy expirations + new build from RPS mandates)
- Scarcity inflection = participation × match level where hourly demand > uncommitted non-SSS supply

**Interactive dashboard toggles:** Corporate participation (slider 0-100%), hourly match target (slider 75-100%), region selector (5 ISOs + national), demand growth (Low/Med/High), time horizon (2025-2050)

### Timezone / UTC Handling (Feb 15)
- **EIA hourly data**: Local time (NOT UTC), per EIA documentation. No offset needed during data loading.
- **Optimizer compressed_day**: New optimizer checkpoint outputs UTC-indexed arrays (h%24 from 0-8759 sequential UTC). Old pre-computed results were local time.
- **Dashboard fix applied**: CAISO 75%/80% rotated UTC→local (offset 8). All other ISOs were already local.
- **Future checkpoint merges**: Must apply UTC-8 rotation to CAISO compressed_day data from new optimizer run before merging into results JSON.
- **All other ISOs verified**: PJM, ERCOT, NYISO, NEISO show local-time profiles (solar 7-19, demand peaks 16-18). Issue was CAISO-specific from checkpoint merge.

### About Page (`about.html`) — Design Direction
- **Purpose**: Scrollytell explainer of the entire project scope and what it researches & explores
- **Narrative layers** (in order):
  1. System-level grid decarb economics (marginal dispatch, last-mile costs, hourly supply gaps)
  2. Power generation corporate targets and decarbonization efforts (fleet transition, nuclear, CCS, renewables)
  3. Voluntary corporate clean energy buyers (PPAs, 24/7 CFE, hourly matching, EAC demand)
  4. State and national policies (RPS, ITC/PTC, 45Q, capacity markets, mandates)
  5. Interconnected accounting & reporting frameworks (GHG Protocol Scope 2 revision, SBTi, EPRI SMARTargets)
  6. Global and national goals (Paris, IEA NZE, US targets, EU climate law)
- **Mind map visualization**: SVG-based infographic showing relationships between all six layers with:
  - Catalytic links (green dashed) — positive feedback loops accelerating decarbonization
  - Perverse incentive links (red dashed) — misaligned frameworks channeling dollars to paper compliance
  - Feedback loops (blue dashed) — systemic interdependencies
  - Animated node entrance + line drawing on scroll
- **Key themes**:
  - How frameworks can catalyze affordable/feasible decarbonization OR create perverse incentives (e.g., 45Q running gas at max CF, annual RECs hiding dirty hours, unbundled cross-region RECs)
  - Research gaps this project addresses: cost-as-variable co-optimization, regional variation, last-5% inflection zone, EAC scarcity quantification, policies evaluated against physical constraints
  - Novel insights produced: cost drives mix, inflection zone steeper than expected, region determines strategy, existing clean assets undervalued, 45Q perverse incentive, EAC scarcity already emerging
- **Standalone page** — no dependencies on other files, avoids merge conflicts with ongoing work on other branches

### LMP Price Calculation Module — Design Plan (Feb 20, 2026)

**Purpose**: Compute synthetic hourly LMP (Locational Marginal Prices) for each winning scenario by reconstructing 8760-hour dispatch and applying ISO-specific price formation models. Enables cost-of-energy analysis that accounts for how clean energy penetration reshapes wholesale electricity prices.

**Pipeline position**: Downstream of Step 4. Reads Step 3/4 outputs, writes to `data/step4-analysis/lmp/`. No changes to Steps 1–4.

```
Step 1 (PFS) → Step 2 (EF) → Step 3 (Cost) → Step 4 (Postprocess)
                                                      ↓
                                          step5_compute_lmp_prices.py
                                                      ↓
                              data/step4-analysis/lmp/{ISO}_lmp.parquet   (per-ISO output)
                              data/step4-analysis/lmp/{ISO}_archetypes.parquet
                              data/step4-analysis/lmp/lmp_summary.json  (dashboard-ready)
```

#### Shared Architecture: `dispatch_utils.py`

Single source of truth for dispatch reconstruction, fossil retirement, and profile loading. All Step 5/6 scripts import from this module. Step 1 Numba JIT dispatch functions consolidated here.

```
dispatch_utils.py (shared)
├── Constants: battery/LDES params, hydro caps, grid mix, coal/oil caps, base demand
├── get_supply_profiles(iso, gen_profiles)           ← nuclear derate + DST correction
├── reconstruct_hourly_dispatch(..., detailed=False)  ← battery + LDES dispatch
│   └── detailed=True adds: per-resource matched/surplus, charge profiles
├── _compute_per_resource_dispatch(...)               ← merit-order: CF→CCS→hydro→wind→solar
├── _battery_loop_detailed / _ldes_loop_detailed      ← Numba loops with charge tracking
├── compute_fossil_retirement(iso, clean_pct, ...)    ← remaining capacity at threshold
├── compute_co2_from_dispatch(iso, dispatch, ...)     ← hourly merit-order emission accounting
├── load_common_data()                                ← demand, gen profiles, emission rates, fossil mix
├── Dispatch cache: load/save per-ISO NPZ, versioned (CACHE_VERSION=2)
└── DISPATCH_ORDER, CACHE_VERSION constants

step4_build_dispatch_cache.py (runs first — Step 5)
├── extract_unique_mixes(iso, input_dir)              ← reads step4/step3 parquets
├── build_cache_for_iso(iso, mixes, ...)              ← detailed=True for all mixes
└── Output: data/step3-dispatch/{ISO}_dispatch_cache.npz (v2)

step5_compress_day_profiles.py (Step 6 — reads from dispatch cache)
├── dispatch_from_cache(iso, mix, ...)                ← cache lookup → result format
├── compress_to_24h(result)                           ← 8760 → 24 hour-of-day sums
└── No duplicate dispatch engine — imports from dispatch_utils

step5_compute_co2.py (Step 6 — dispatch-stack emission model)
├── compute_dispatch_stack_emission_rate()
├── fast_co2_from_match_score()                       ← ~1000x faster, no dispatch needed
├── compute_co2_hourly()                              ← fallback dispatch path
└── recompute_all_co2()

step5_consequential_deployment_queue.py (Step 6 — uses dispatch cache for emissions)
├── get_dispatch_co2()                                ← dispatch-cache-based emission lookup
├── compute_marginal_displaced_rate_dispatch()         ← zone-boundary CO₂
└── extract_medium_scenarios() + consequential queue logic

step6_scenario_comparison.py (Step 6 — dispatch cache for both scenarios)
├── _get_dispatch_co2_for_mix()                       ← scenario mix emission lookup
├── build_consequential_queue()                       ← dispatch-based emissions
└── find_optimal_mixes() / find_optimal_mixes_sequential()

step5_compute_lmp_prices.py (Step 6 — imports dispatch_utils)
├── PriceModel classes (ISO-specific)
├── build_merit_order_stack()
├── compute_lmp_stats()
└── calibration framework
```

**Dispatch cache pipeline position**: `step4_build_dispatch_cache.py` runs after Step 4, before all Step 6 scripts. Pre-computes dispatch for all unique mixes across 7 ISOs. Cache is versioned (v2) to invalidate stale v1 caches.

**CO₂ recompute bug fix (Feb 2026)**: Was importing `get_supply_profiles_simple` (flat clean_firm, no DST correction) instead of the canonical `get_supply_profiles`. Fixed to use the same nuclear-derated, DST-corrected profiles as Step 1.

**Compressed day compatibility note**: Local dispatch engine replaced with canonical dispatch_utils engine. Total energy dispatched is identical; hourly distribution differs slightly.

#### Design Decisions (Locked)

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | Merit-order stack walk | `np.searchsorted` step-function | More accurate than `np.interp` linear interpolation — discrete units don't interpolate. Same performance. |
| 2 | Archetype dedup key | `(mix_tuple, fuel_level, threshold)` | Threshold affects fossil stack (retirement changes available capacity). ~7,800 unique calcs per ISO. Still fast with Numba (<30s). |
| 3 | Dispatch functions | Shared module (`dispatch_utils.py`) | Single source of truth. Extracted from `step5_compute_co2.py` + Step 1 Numba JIT. |
| 4 | Surplus pricing | Calibrated empirical curve from start | Parameterized with calibration targets; "reasonable defaults" before actual LMP data. Phase 2 tunes parameters, no refactoring. |
| 5 | Calibration LMP source | Day-ahead LMP | Cleaner, better for structural model. RT sensitivity via `rt_sensitivity_factor` parameter baked in. |

#### Data Flow

**Inputs (all existing)**:
- `data/eia_generation_profiles.json` — hourly solar/wind/hydro/nuclear shapes (8760)
- `data/eia_demand_profiles.json` — hourly demand shape (8760)
- `dashboard/overprocure_scenarios.parquet` (or `overprocure_results.json`) — winning mixes
- `data/eia_fossil_mix.json` — coal/gas/oil shares for stack construction
- `data/egrid_emission_rates.json` — heat rates for marginal cost derivation
- Step 3/4 constants — wholesale prices, fuel adjustments, gas capacity (imported at runtime, NOT hardcoded)

**New inputs (Phase 2 — calibration)**:
- `data/step4-analysis/lmp/actual_lmp_PJM.json` — PJM Data Miner 2 API (Western Hub, 2021-2025)

**Outputs**:
- `data/step4-analysis/lmp/{ISO}_lmp.parquet` — summary stats per (threshold, scenario), ~2 MB/ISO
- `data/step4-analysis/lmp/{ISO}_archetypes.parquet` — 8760h profiles for unique archetypes, ~15-20 MB/ISO
- `data/step4-analysis/lmp/{ISO}_checkpoint.json` — resume state (transient, deleted on completion)
- `data/step4-analysis/lmp/lmp_summary.json` — dashboard-ready cross-ISO summary, <500 KB
- `dashboard/js/lmp-data.js` — client-side visualization data (Phase 4)

#### Hourly Dispatch Reconstruction

Reuses Step 1/Step 5 logic via shared `dispatch_utils.py`:
1. Build weighted supply curve: `Σ (mix_pct × profile)` for clean_firm, solar, wind, hydro
2. Apply procurement multiplier
3. Battery dispatch (4hr daily cycle, 85% RTE)
4. Battery8 dispatch (8hr daily cycle, 85% RTE)
5. LDES dispatch (100hr, 7-day window, 50% RTE)
6. Result: `residual_demand = demand - total_clean_supply` (8760 array; positive = fossil needed)

**Vectorization**: Base supply for N archetypes is a single matrix multiply `(N,4) @ (4,8760)`. Storage dispatch loops per-archetype (SOC state carries forward) — Numba JIT target.

#### Fossil Merit-Order Stack (Parameterized)

Heat rates (MMBtu/MWh) from EIA Electric Power Annual Table 8.1:
- Coal steam: 10.0, Gas CCGT: 6.4, Gas CT: 10.0, Oil CT: 10.5

Variable O&M ($/MWh) from EIA AEO / NREL ATB:
- Coal: $4.50, Gas CCGT: $2.00, Gas CT: $4.00, Oil CT: $5.00

Marginal cost = heat_rate × fuel_price + VOM. **Stack order determined by marginal cost** — fuel-switching aware:
- At Low gas ($2/MMBtu): gas CCGT MC = $14.80 < coal MC = $22.50 → gas dispatches first
- At High gas ($6/MMBtu): gas CCGT MC = $40.40 > coal MC = $26.50 → coal dispatches first

Fuel prices imported from Step 3/4 at runtime (L/M/H sensitivity). Fossil capacity from shared retirement model (threshold-dependent: coal retires first → oil → gas).

#### ISO-Specific Price Formation Models

Each ISO gets its own `PriceModel` class with calibratable parameters:

| ISO | Capacity Mechanism | Scarcity Model | Surplus Model | Key Parameters |
|---|---|---|---|---|
| **PJM** | RPM capacity market | Penalty factor → $2,000 cap | Moderate negative prices (coal/nuclear must-run) | `scarcity_cap=2000`, `floor=-30`, coal baseload min-gen |
| **ERCOT** | Energy-only | **ORDC** — exponential knee: $0 above 3 GW reserves, exp decay below, $500 cap | Aggressive negative prices (no must-run obligation) | `voll=5000`, `knee_mw=3000`, `lam=0.002`, `cap=500`, `floor=-50` |
| **CAISO** | Resource Adequacy | Soft cap + RDRR mechanism | Most negative prices (solar duck curve) | `scarcity_cap=2000`, `floor=-60`, solar curtailment premium |
| **NYISO** | ICAP | Penalty factor → $2,000 cap | Similar to PJM but tighter geography | `scarcity_cap=2000`, `floor=-20` |
| **NEISO** | FCM | Penalty factor → $2,000 cap | Winter gas constraint creates seasonal scarcity | `scarcity_cap=2000`, `winter_gas_adder`, pipeline constraint |

**ORDC scarcity pricing** (all ISOs, `SCARCITY_MODE='ordc'`): Exponential knee model — adder is exactly $0 when reserves ≥ knee threshold, rises as `min(cap, VOLL × exp(-λ × reserves))` below knee. Per-ISO parameters: ERCOT (VOLL=$5K, knee=3GW, λ=0.002, cap=$500), PJM (VOLL=$3.7K, knee=6GW, λ=0.0015, cap=$300), CAISO (VOLL=$2K, knee=4GW, λ=0.002, cap=$300), MISO (VOLL=$3.5K, knee=5GW, λ=0.0015, cap=$300). Calibrated for $2–8/MWh annual avg ORDC contribution, 30–100 scarcity hours.

**NEISO winter gas** (unique — already modeled in Step 4): Winter hours (Dec-Feb) get gas price spike due to pipeline constraints. Parameterized via existing `NEISO_CCS_GAS_ADDER = $13.13/MWh`.

**All surplus pricing uses calibrated empirical curves** — parameterized from the start with shape/floor/decay parameters that Phase 2 calibration tunes. Reasonable defaults used before actual LMP data.

**RT sensitivity**: `rt_sensitivity_factor` parameter scales volatility/spread to approximate real-time conditions from day-ahead calibration.

#### Output Schema

**Per-ISO stats: `data/step4-analysis/lmp/{ISO}_lmp.parquet`**

| Column | Type | Description |
|---|---|---|
| `threshold` | float32 | Clean energy % |
| `scenario` | string | 9-dim key |
| `archetype_key` | string | Dedup key |
| `avg_lmp` | float32 | Time-weighted average $/MWh |
| `peak_avg_lmp` | float32 | Peak hours (7am-11pm weekdays) |
| `offpeak_avg_lmp` | float32 | Off-peak hours |
| `zero_price_hours` | int16 | Hours at $0 or below |
| `negative_price_hours` | int16 | Hours below $0 |
| `scarcity_hours` | int16 | Hours above scarcity threshold |
| `lmp_p10/p25/p50/p75/p90` | float32 | Percentiles |
| `price_volatility` | float32 | Std dev |
| `duck_curve_depth_mw` | float32 | Max surplus MW |
| `net_peak_price` | float32 | Price at max net demand hour |
| `fossil_revenue_mwh` | float32 | Avg $/MWh earned by remaining fossil |

**Per-ISO archetype profiles: `data/step4-analysis/lmp/{ISO}_archetypes.parquet`**

| Column | Type |
|---|---|
| `archetype_key` | string |
| `threshold` | float32 |
| `fuel_level` | string |
| `hourly_lmp` | float32[8760] (list column) |
| `hourly_residual_mw` | float32[8760] (list column) |
| `hourly_marginal_unit` | uint8[8760] (list column) |

Size: ~15-20 MB/ISO compressed. Total all ISOs: ~75-100 MB.

#### Checkpointing & Session Resilience

- **Atomic checkpoint writes** (`os.replace` — POSIX atomic)
- **Append-mode parquet** (don't hold full ISO in memory)
- **Per-threshold flush** (max ~30s of lost work on crash)
- **Skip-if-exists** at ISO level (`--force` to override)
- **Per-ISO output files** (completing PJM doesn't risk ERCOT data)
- **Resume from checkpoint**: loads `{ISO}_checkpoint.json`, skips completed thresholds

#### Calibration Framework (Phase 2)

**Data source**: PJM Data Miner 2 API — Western Hub, DA LMP, 2021-2025. Free registration.

**Calibration targets (weighted)**:
- 40%: RMSE of hourly prices
- 15%: Error in annual mean price
- 15%: Error in zero/negative price hour count
- 15%: Error in P90 price (tail behavior)
- 15%: KS statistic of price duration curve shape

**Parameters calibrated**: floor_price, surplus_slope, stack price offsets, scarcity_shape.

**Validation**: Train 2021-2023, test 2024-2025. Cross-region sanity check: calibrate PJM, verify NYISO (similar market).

**Other ISO data sources** (Phase 3): ERCOT (`misportal.ercot.com`), CAISO (`oasis.caiso.com`), NYISO (`mis.nyiso.com`), NEISO (`isoexpress.iso-ne.com`). All free. `gridstatus` package wraps all APIs.

#### Implementation Phases

| Phase | Scope | Files | Size |
|---|---|---|---|
| **0** | Extract `dispatch_utils.py` from `step5_compute_co2.py` + compatibility test | `dispatch_utils.py`, `step5_compute_co2.py` | ~300 lines extracted |
| **1a** | Core engine — PJM only, Medium fuel, no calibration | `step5_compute_lmp_prices.py` | ~400 lines |
| **1b** | Full fuel sensitivity sweep (L/M/H) + all thresholds + checkpointing | Same | +~150 lines |
| **1c** | All 5 ISOs with market-specific models | Same | +~200 lines |
| **2a** | PJM LMP data fetch | `fetch_pjm_lmp.py` | ~150 lines |
| **2b** | Calibration + validation | `calibrate_lmp_model.py` | ~200 lines |
| **3** | All-ISO calibration data fetch | Per-ISO fetch scripts | ~150 lines |
| **4** | Dashboard integration | `step6_generate_shared_data.py` + JS | ~150 lines |

#### New Files

```
dispatch_utils.py              # Shared dispatch/retirement/profiles (~300 lines)
step5_compute_lmp_prices.py  # Main LMP script (~750 lines)
fetch_pjm_lmp.py               # Phase 2: LMP data fetcher (~150 lines)
calibrate_lmp_model.py         # Phase 2: Parameter calibration (~200 lines)
data/step4-analysis/lmp/                      # Output directory
  ├── {ISO}_lmp.parquet        # ~2 MB each
  ├── {ISO}_archetypes.parquet # ~15-20 MB each
  ├── {ISO}_checkpoint.json    # Transient
  ├── lmp_summary.json         # ~500 KB
  └── actual_lmp_*.json        # Phase 2: calibration data
```

### Open questions
- Path-dependent MAC visualization: may need alternative to MAC curve format
- ELCC: include in next run? Fixed or penetration-dependent?
- Multi-year re-run: Phase 1+3 hybrid recommended (~40% compute savings vs full)

---

