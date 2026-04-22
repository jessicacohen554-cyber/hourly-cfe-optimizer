# Advanced Sensitivity Model — Complete Specification

> Historical decisions and landed workstreams live in `SPEC_LOG.md`.

> **Authoritative reference for all design decisions.** If a future session needs context, read this file first.
> Last updated: 2026-04-22 (late — endpoint-year schedule applied to myopic solver).

## Current Status (Apr 22, 2026 — late)

### Step 2.3 — Endpoint-year schedule applied to myopic solver — LANDED on `claude/debug-step-2-3-script-SBNaV`

**What shipped.** Rebased onto `origin/master` (post-Phase-C-tasks-1–8, top commit `b3d7229`) and applied the SBTi-ladder schedule (`90→2040, 95→2045, 99→2049, 99.9→2050` from `pipeline_config.THRESHOLD_TARGET_YEARS`) to the myopic solver using the same freeze-forward convention that `solve_pathway_with_foresight` already uses.

**Design choice: freeze-forward vs. truncate.** An earlier draft on this branch truncated the myopic solver's per-year arrays to `[:n_years_run]` and renamed the `_2050` stranding fields. That turned out to be incompatible with Phase C's convention, which keeps full N_YEARS arrays, freezes the endpoint winner forward from `endpoint_yi` to 2050, and defers display truncation to a future Phase E post-processor (memo §6.5). Adopting Phase C's convention for the myopic path gives: (a) schema parity between myopic and foresight payloads (both 26-row); (b) no renames (2050 = endpoint state under freeze); (c) the Phase C `phase_c_regression_lambda_zero.py` regression passes bit-identity between myopic and foresight on the truncated prefix at all 4 ERCOT P3 endpoints without modification. Phase C's `test_foresight_scorer.py` (5 tests) also still passes.

**Code changes (12 lines of solver + 4 lines of config/headline).** `scripts/step_2_3_pathway_optimizer.py`:
1. `ENDPOINT_TO_THRESHOLD` shrunk from 10 entries to 4 (`{0.90, 0.95, 0.99, 0.999}`). Full-sweep combo count drops `7×5×10=350 → 7×5×4=140`.
2. `_EP_TAG_MAP` shrunk to the same 4 tags.
3. `_cfe_target_for_year` ramp now terminates at `_foresight_endpoint_year(endpoint_pct)` (reads from `pipeline_config.THRESHOLD_TARGET_YEARS`); interior buildup waypoints (2030/50, 2035/70, 2040/90, 2045/95) stay only when strictly before the deadline; post-endpoint years hold flat at `endpoint_pct`.
4. `solve_pathway` (myopic) computes `_myopic_endpoint_yi` once, bounds the main loop to `range(_myopic_endpoint_yi + 1)`, and after the loop freezes `winners[endpoint_yi+1:N_YEARS] = endpoint_w` plus propagates `winner_feasible` / resets `ratchet_violated`. Mirrors the `solve_pathway_with_foresight` block at lines 1792–1800 exactly.
5. `headline.endpoint_year: int` added to `_finalize_run`. Emits in both v2 (myopic) and v3 (foresight) payloads.
6. Argparse help string updated (`7×5×10 → 7×5×4`).

**Downstream updates (4-endpoint set):** `reliability_tax/charts/data_loader.py` (ENDPOINTS + _EP_LABEL), `reliability_tax/charts/gen_closing_summary_table.py` (CLOSING_ENDPOINTS + narrative), `scripts/run_pathway_sweep.py` (ALL_ENDPOINTS + help text), `scripts/_surrogate_ab_driver.py` (0.80 combos swapped for 0.90 — only canonical endpoints remain valid), `.github/workflows/step2-3-sweep-iso.yml` (input description). No field renames, no fallback `.get()` chains needed since the `_2050` field names still accurately describe the 2050 state under freeze-forward.

**Not updated:** `scripts/rebuild_reliability_tax_manifest.py` narrative string, `gen_section2_gas_hump.py` doc-comment — cosmetic only.

**Verification.** NEISO P3 × {ep90, ep95, ep99, ep99p9} all run end-to-end with `schema_version: 2`, `headline.endpoint_year` set to the expected SBTi year, full 26-row output with years past `endpoint_year` frozen at the endpoint CFE (ep90: years 2041–2050 all read CFE 90.80, matching year 2040). NEISO P3 ep95 in foresight mode also works end-to-end, emits `schema_version: "3.0.0"` + `solver_mode: foresight`. Phase C unit tests (5 in `test_foresight_scorer.py`) pass; `phase_c_regression_lambda_zero.py` confirms λ=0 foresight bit-identity to myopic on all 4 ERCOT P3 endpoints.

**Phase 5 still pending.** Full 140-combo sweep (7 ISOs × 5 pathways × 4 endpoints) + chart-payload regeneration for all 12 generators + `/fix-prose` across regenerated dashboards has NOT run this session. Stale cached payloads under `analysis/reliability-tax/data/*/pathway*_ep{60,70,75,80,85,97p5}*.json` and corresponding `step_2_1_EF_*_{60,70,75,80,85}_peakclean.parquet` sidecars are now orphans — the solver no longer accepts those endpoints. User decision deferred on whether to delete them before Phase 5 regen or let them sit.

**Resume prompt for next session.** *"Endpoint schedule + freeze-forward landed on `claude/debug-step-2-3-script-SBNaV`, rebased onto current master (post-Phase-C). Solver loop runs 2025→`_foresight_endpoint_year(endpoint_pct)` then freezes winner forward through 2050; schema v2 (myopic) and v3 (foresight) both emit 26-row payloads with `headline.endpoint_year`. Next steps: (a) user decide on deletion of orphaned `ep{60,70,75,80,85,97p5}` cached payloads + `_peakclean` sidecars; (b) run the 140-combo Phase 5 sweep via `scripts/step_2_3_pathway_optimizer.py --all`; (c) regenerate all 12 chart payloads; (d) run `/fix-prose` across any regenerated dashboard HTML. Deferred: Phase E display-truncation post-processor (memo §6.5) that would trim 2041–2050 rows from ep90 chart output — only a display fix; solver semantics are correct under freeze-forward."*

## Current Status (Apr 21, 2026 — late)

### Step 2.3 — Vintage-scaling bug fix — LANDED on `claude/fix-vintage-scaling-bug-dbwdI`

**What shipped (commits `b381824` code + `7f5f590` regenerated ERCOT JSONs).** Replaced the per-year share-delta ledger + demand-scaled gross_op with a sequential absolute-TWh build model. Five coupled changes in one commit to `scripts/step_2_3_pathway_optimizer.py` (+207/−144 lines):

1. **`solve_pathway` inline sunk-cost-aware scorer (Interpretation B).** The pre-built `operating_cost_matrix` is gone. Each year's argmin sees `score[m] = Σ_r max(0, target_twh_r[m] − floor_twh_r) × LCOE_r_y` over 10 non-CF resources + 4 CF tranches, plus the existing gas/storage cost buckets. Prior-built capacity is sunk — it prices to $0 in the scorer.
2. **Absolute-TWh floor ratchet.** Per-resource floors grow monotonically. Year-y argmin is restricted to mixes whose target TWh ≥ floor for every resource. Demand growth at constant share mechanically produces new vintages covering the gap (existing TWh stays frozen at its locked LCOE).
3. **4-tier fallback cascade.** Tier 1 = ratchet ∩ pmask ∩ cfe ∩ cf_feasible → Tier 2 = relax CFE ±5 pts → Tier 3 = drop cf_feasible, flag `feasibility.physical = False` → Tier 4 emergency = drop ratchet, flag `ratchet_violated`. Notes surface in `feasibility.notes`.
4. **Ledger + serializer rewrite.** `_build_ledger` and `serialize_run_result` both track absolute-TWh floors for non-CF and CF tranches. Baseline existing vintages stay frozen at `base_demand × grid_share / 100` with `locked_lcoe = 0`. New deltas are `target_twh_y − floor` (not `share_delta × demand_y`).
5. **`_finalize_run` drops `× (demand_vec[yi] / demand_vec[0])` on `gross_operating_usd`.** Under the new ledger semantics that multiplier double-scaled new-build vintages (already carrying `(1+g)^yi` in `twh_per_year`) and wrongly inflated the LCOE=0 baseline.

Removed: `operating_cost_matrix` (no callers under B).

**Before-fix witness (cached ERCOT P3@ep99).** 2050 `gross_operating_usd` = $282.6B decomposed as $29.7B baseline + $89.9B new-build, both inflated 2.36× by `demand_2050/demand_2025`. Under the fix the new-build portion prices at its absolute locked-LCOE value and baseline contributes $0.

**Delta report (ERCOT {P1, P2a, P3} × {ep90, ep99}, 6 combos, all `feasibility.physical=True`, no fallback notes):**

| Combo | undisc Δ vs 1a50ade | fleet Δ | rt Δ | cfe |
|---|---|---|---|---|
| P1@ep90 | +73.1% ($+755B) | −28% (−39GW) | +79.5% | 90.07→90.20 |
| P1@ep99 | +20.8% ($+332B) | +0.8% | +54.0% | 99.06→99.06 |
| P2a@ep90 | +61.9% ($+684B) | −22% | +91.4% | 90.02→90.20 |
| **P2a@ep99** | **−40.7% (−$1,325B)** | +50% (+40GW) | +195.8% | 99.00→99.06 |
| P3@ep90 | +52.2% ($+613B) | −22% | +104.1% | 90.02→90.20 |
| **P3@ep99** | **−50.5% (−$1,968B)** | +50% | +310.3% | 99.00→99.06 |

The two extreme pre-fix outliers ($3.3T P2a@ep99, $3.9T P3@ep99) collapse to $1.9T — consistent with removing the double-scale. ep90 combos move UP because the B scorer now picks genuinely different (and, over the 26-year integration window, more expensive) winners once demand-growth isn't implicitly free. Fleet swings cascade into `reliability_tax` components.

**P1/P2a/P3 winners converge at each endpoint under B.** P3's new-nuke unlock goes unused because incremental VRE+storage builds are cheaper at year-y LCOE than new-build nuclear in ERCOT. This is the intended methodology behavior — pathways differ only where the argmin genuinely finds different cost-minimal solutions, not because of per-year replacement-cost repricing artifacts. Worth auditing whether this holds outside ERCOT (VRE-rich); PJM / NYISO / NEISO may yield genuine P1≠P3 divergence once the full sweep runs.

**Threshold audit (hardcoded $T/$B captions).** Two caption sites reference pre-fix data and will need regeneration in Phase 5: `reliability_tax/charts/section1_narratives.json` (per-ISO "95%→99% adds $X.XXT" steps) and `dashboard/reliability-tax.html` + `reliability-tax-4.6.html` L577/L544 ("PJM 90% clean: $7.17T → $2.39T"). Both are cross-ISO / non-ERCOT; out of scope for this fix. Other $-values in the codebase (Google-Intersect, 45U PTC, Vogtle, AUM figures, etc.) are external facts unrelated to the vintage-scaling bug.

**Branch-state notes for the next session.**
- `scripts/_sweep_pathways_inproc.py` is broken on this branch — imports `ENDPOINTS`, `run_pathway`, `_batch_append_to_manifest` from `step_2_3_pathway_optimizer` (none exist) and `flush_expanded_cache`/`_get_worst_hour_state`/`prewarm_caches` from `step2_2a_cost_optimization` (module doesn't exist; was in the deleted `market-simulator/`). The driver was written against a planned refactor that didn't land. For this session's 6-combo rerun I called `step_2_3_pathway_optimizer._run_one()` directly in one Python process — same cold-start amortization without resurrecting the stale driver. Either delete the driver or rewrite it; leaving it as dead code for now.
- `market-simulator/` deletion is fine for the reliability-tax pipeline — `step_2_3_pathway_optimizer.py` is fully self-contained.

**Phase 5 gated behind user sign-off.** Full 350-combo sweep + chart-payload regeneration for all 7 ISOs is NOT started this session per the resume-prompt's explicit "BLOCKED" directive. Awaiting user review of the 6-combo ERCOT delta report before proceeding.

**Separate future workstream flagged (NOT this fix).** User mentioned a later-session item: implement a "work-backwards from the cheapest 95–99% mix" foresight-driven argmin for Pathway 3 only, so P3 represents strategic planning (vs. market-myopic P1). That's additive to the ratchet model — this session's fix is about correctness of the forward-simulation; P3-foresight is about adding a lookahead layer on top of it for one pathway.

**Resume prompt for next session.** *"Vintage-scaling fix landed (commits `b381824` + `7f5f590`) on `claude/fix-vintage-scaling-bug-dbwdI`. All 6 ERCOT combos regenerated and feasibility.physical=True with no fallback notes. Two extreme outliers (P2a/P3 @ ep99) collapsed from $3.3T/$3.9T to $1.9T; ep90 combos moved UP by 52–73% because the B scorer now picks genuinely different winners. P1/P2a/P3 converge at each endpoint in ERCOT — audit whether this holds in PJM/NYISO/NEISO. Next steps: (a) user review the delta report; (b) if approved, regenerate the 6 non-ERCOT ISOs' 50 combos each via the same `_run_one` direct-call pattern (the `_sweep_pathways_inproc.py` driver is broken on this branch — imports a stale API from a deleted `market-simulator/`); (c) regenerate 12 chart payloads in `reliability_tax/charts/` + the two flagged dashboard captions; (d) run `/fix-prose` across the regenerated dashboards. Deferred separately: the P3 foresight-driven argmin (lookahead layer on top of the ratchet model, P3 only)."*

## Current Status (Apr 21, 2026)

### Step 2.3 v3 — Clean-firm disaggregation (Phase 2.5+4) — ESCALATION PENDING USER DECISION

**Landed on branch `claude/phase-3-cost-matrix-fix-E12dS` (top commit `472392c` = Phase 3 cost-matrix fix; WIP ahead).**

**What shipped in Phase 3 (`472392c`).** Ratcheted new-gas capex via `np.maximum.accumulate(gas_required, axis=0)`, NEW_CCGT_FOM_KW_YR wired into cost_matrix + rt_components (was hard-coded 0), tranche-blended clean_firm LCOE from `decompose_clean_firm_tranches(...)['blended_lcoe']` replacing the scalar `_resource_lcoe_year('clean_firm', ...)` path, P1/P1a pathway mask widened to admit `cf ≤ baseline + UPRATE_CAP_TWH/demand`. ERCOT 6-combo gate vs pre-Phase-3 was clear (+0–23% cost moves) — pathway differentiation restored at ep90 (P1 cf=0 vs. P2a/P3 cf=10), but ep99 still collapsed to cf=0 across all three pathways.

**What's WIP and awaiting decision.** User pushed back on the blended-LCOE design: clean_firm carries a real merit-order cost stack (existing baseline $0 → uprate $25 → geo → min(new_nuke, CCS)) and blending that to a single $/MWh hides two distinct bugs — (a) the Phase-2 wrapper fed `target = cf_pct × demand` including the existing-nuclear baseline, charging it at uprate-LCOE when the ledger correctly has existing at $0; (b) the blend collapses the tranche signal the argmin needs. Full disaggregation implemented across `decompose_clean_firm_tranches` (returns per-tranche (Y, n) TWh + per-year effective $/MWh; no blend), `operating_cost_matrix` (sums `Σ tranche_twh × tranche_eff_lcoe × 1e6` directly; returns `(op_cost, feasible_yn)` tuple), `solve_pathway` (∞-masks infeasible cells, propagates `winner_feasible` to `_finalize_run`), `_build_ledger` + `serialize_run_result` (emit `clean_firm_existing/_uprate/_geo/_nuke_new/_ccs` vintages with tranche-specific locked_lcoe), and `feasibility.physical` in the output JSON.

**Gate outcome (ERCOT 6 combos, disaggregated vs Phase 3).** P1/ep90 and P1/ep99 unchanged (locked pathway; baseline already free, no new tranches available). P2a/P3 at ep90 shifted cf=10→20 with fleet −7.3%, cost −3.5/−4.6%, tax −3% — clean under 50% threshold. **P2a/P3 at ep99 broke from cf=0 to cf=79**: fleet −33%, reliability tax −48/−62%, **but total cost +104/+144%** because the argmin now builds 529 TWh/yr of new clean_firm priced at the merit-order bill ($55–70/MWh) instead of avoiding it. >50% triggers the plan's escalation gate. Diagnosis in session transcript: direction looks correct (this is the pathway collapse the Phase-1 probe flagged), magnitude is plausible given the build size, and reliability-tax drop makes physical sense (79% clean-firm grid needs much less gas backup). Pre-existing `_finalize_run` over-scale (`gross_op *= demand_vec[yi]/demand_vec[0]` double-scales new vintages that were already sized at cod-year demand) inflates absolute costs by ~20% in late years but affects both Phase 3 and disaggregated runs equally — not fixed this session, flagged for a later phase.

**Sample terminal_ledger for ERCOT P3/ep99 (disaggregated).** `clean_firm_existing 2025 42.0 TWh $0` → `clean_firm_uprate 2030 1.7 TWh $25` (cap hit exactly) → `clean_firm_ccs 2030–2035 cumulative 200 TWh $55.5–112` (cap hit in 2035, FOAK→NOAK year-adjusted) → `clean_firm_nuke_new 2035+ 283 TWh/yr growing to 310 TWh/yr by 2050 @ $70`. Merit order works — CCS is cheaper year-adjusted in ERCOT so fills to cap first; nuke picks up the leftover.

**Three options surfaced to user, awaiting decision.**
1. Commit the WIP as-is — gate threshold was a check against mechanical bugs, not methodology corrections; the ep99 move is exactly the pathway collapse fix the audit demanded.
2. Dive into a by-year cost breakdown on P3/ep99 first to confirm no second bug is inflating the $3.9T figure, then commit.
3. Revert and rethink — only if the magnitude reads as implausible.

**Resume prompt.** *"Resume Step 2.3 v3 — Clean-firm disaggregation escalation on branch `claude/phase-3-cost-matrix-fix-E12dS`. WIP commit contains `scripts/step_2_3_pathway_optimizer.py` rewritten so `decompose_clean_firm_tranches` returns per-tranche (Y, n) TWh + per-year eff LCOE (no blend), `operating_cost_matrix` sums tranche × LCOE directly and returns `(op_cost, feasible_yn)`, `solve_pathway` ∞-masks infeasible cells, `_build_ledger` + `serialize_run_result` emit `clean_firm_existing/_uprate/_geo/_nuke_new/_ccs` per-tranche vintages, `feasibility.physical` reflects winner feasibility. ERCOT 6-combo gate tripped on P2a/P3 ep99 (fleet −33%, cost +104/+144% vs. Phase 3; reliability tax −48/−62%) because the argmin broke out of cf=0 collapse into cf=79 and now prices the 529 TWh/yr new-clean_firm build at merit-order rates. 6 ERCOT JSONs in `analysis/reliability-tax/data/ERCOT/` reflect the new behavior. User has three options in the transcript (1) commit, (2) by-year cost breakdown then commit, (3) revert. Also flagged: pre-existing over-scale in `_finalize_run` line 1339 (`gross_op *= demand_vec[yi]/demand_vec[0]`) inflates new-vintage costs ~20% in late years — not this session's scope but worth a later phase. Do NOT proceed past ERCOT 6-combo until the user picks an option. Phase 5 (50-combo ERCOT smoke + regenerated cached outputs) remains pending behind the decision."*

> Older status blocks moved to `SPEC_LOG.md` (Apr 18, 2026 archive cut; Apr 19 rotations moved the oldest reliability-tax redesign block + the Apr 18 project-infra block to SPEC_LOG; Apr 20 rotation moved the Apr 18 Coding-Session sub-agent block to SPEC_LOG; Apr 22 late rotation moved the Apr 20 audit block to SPEC_LOG). See that file for the historical decision log.
---

## 1. Model Framework

- **2025 snapshot model** — all data, profiles, costs, grid mix shares reflect fixed 2025 actuals
- **No demand growth projections** — point-in-time scenario analysis only
- **Grid mix baseline** = actual 2025 regional shares, priced at wholesale, selectable as reference scenario (fixed, not adjustable by user)
- **Regions**: CAISO, ERCOT, PJM, NYISO, NEISO, MISO, SPP (7 ISOs)
- **Repo**: `jessicacohen554-cyber/hourly-cfe-optimizer`

---

## 2. Resources (6 total — v4.0 rebuild: CCS merged into Clean Firm)

| # | Resource | Profile Type | New-Build? | Cost Toggle? | Transmission Adder? |
|---|---|---|---|---|---|
| 1 | **Clean Firm** (nuclear/geothermal/CCS-CCGT) | Blended: seasonal-derated baseload (nuclear/geo) + flat baseload (CCS) | Yes | Low/Med/High (regional) | Yes (regional) |
| 2 | **Solar** | EIA 2025 hourly regional | Yes | Low/Med/High (regional) | Yes (regional) |
| 3 | **Wind** | EIA 2025 hourly regional | Yes | Low/Med/High (regional) | Yes (regional) |
| 4 | **Hydro** | EIA 2025 hourly regional | **No** — capped at existing | **No** — wholesale only | **No** — always $0 |
| 5 | **Battery** (4hr Li-ion) | Daily cycle dispatch | Yes | Low/Med/High (regional) | Yes (regional) |
| 6 | **LDES** (100hr iron-air) | Multi-day/seasonal dispatch | Yes | Low/Med/High (regional) | Yes (regional) |

### v4.0 Change: CCS-CCGT merged into Clean Firm (Decision 6D)
- **Rationale**: Reduces resource mix search space from 5D to 4D, dramatically cutting grid search combinatorics (~40-60% fewer combos). Both nuclear and CCS-CCGT are modeled as baseload (CCS runs flat due to 45Q incentives), making them functionally similar for dispatch purposes.
- **Implementation**: The optimizer allocates a single `clean_firm` percentage. Within that allocation, the sub-split between nuclear/geothermal and CCS-CCGT is determined by cost optimization — the cost model evaluates different sub-allocations and picks the cheapest blend. CCS retains its distinct cost profile (LCOE, 45Q offset, fuel linkage) and emission characteristics (95% capture, residual 0.0185 tCO2/MWh).
- **Dispatch profile**: Weighted blend of nuclear seasonal-derated profile and CCS flat profile, based on sub-allocation ratio.
- **Dashboard impact**: Results still report the nuclear/CCS sub-split for transparency.

### Key resource decisions:
- **Green H2 seasonal storage** (added Feb 2026):
  - **Parameters**: 35% round-trip efficiency (electrolysis 70% × storage 95% × turbine 55%), 1000hr duration (~42 days at full power), 30-day rolling dispatch window
  - **Physics**: Dispatches as Phase 4 after battery4 → battery8 → LDES on post-LDES residual surplus/gap. Same window-based charge/discharge as LDES but with longer window and lower RTE.
  - **Sweep levels**: Only evaluated at ≥95% thresholds (too expensive for lower). Levels: [0, 0.3, 1.0] % of demand (reduced March 2026 from 9 levels — H2 never won on cost, and the 9-level grid added a 9× multiplier to Step 1C combo counts causing high-threshold cells to stall).
  - **Cost**: LCOS-based, shares `ldes_lvl` sensitivity toggle. L=$185-230, M=$260-330, H=$365-460 $/MWh by ISO. Transmission adders: L=$2-3, M=$3-6, H=$5-10.
  - **Peak capacity credit**: 0.85 (dispatchable but slower ramp than gas/battery)
  - **Merit order rationale**: Battery → LDES → H2 is economically robust because (1) higher RTE storage should fill short gaps first to minimize surplus waste, (2) battery $/kW is lower than LDES for 4hr needs, (3) H2's only advantage is very cheap $/kWh (salt caverns) at multi-week timescales where LDES is prohibitively expensive.
- **CAISO geothermal as 5th physics dimension** (added Feb 2026):
  - CAISO uses 5D grid search: [clean_firm (nuclear/CCS only), solar, wind, hydro, geothermal] — each as independent % of demand (no sum constraint).
  - **Geothermal profile**: Flat year-round (1/8760 per hour). No seasonal derate — geothermal has no refueling outages.
  - **CAISO clean_firm profile**: Now purely nuclear with full seasonal derate (NUCLEAR_SHARE_OF_CLEAN_FIRM = 1.0 for CAISO). The 70/30 nuclear/geo blend is removed; geothermal physics are captured by the separate dimension.
  - **Geothermal cap**: (existing_geo_TWh + GEO_CAP_TWH) / CAISO_demand_TWh = (5.31 + 39.0) / 224.039 = 19.8% → capped at 20% in grid search.
  - **Non-CAISO ISOs**: Stay 4D. No geothermal resource.
  - **Rationale**: Geothermal has fundamentally different physics than nuclear/CCS (no seasonal derate, no outages). Lumping into clean_firm understated CAISO's winter/spring firm capacity.
- **Clean Firm nuclear derate**: Seasonal spring/fall derate applied to nuclear/CCS portion. Reflects staggered refueling outages for nuclear and scheduled maintenance for CCS-CCGT in shoulder months. Summer/winter: ~100% CF. Spring/fall: reduced CF based on observed EIA 2021-2025 patterns. CCS-CCGT aggregate fleet maintenance in shoulder months produces a similar derate pattern.
- **Hydro**: Existing only, capped at regional capacity, wholesale priced, no new-build tier, $0 transmission
- **CCS-CCGT** (within Clean Firm): 95% capture rate, residual ~0.0185 tCO2/MWh, 45Q ($85/ton = ~$27.5/MWh offset) baked into LCOE, fuel cost linked to gas price toggle. **Modeled as flat baseload (not dispatchable) by design** — while CCS-CCGT is physically dispatchable, the 45Q tax credit ($85/ton for geologic storage) incentivizes running at maximum capacity factor to maximize capture credits. This is an economics-driven decision, not a physical constraint.
- **LDES**: 100-hour iron-air, 50% round-trip efficiency, capacity-constrained dispatch with dynamic capacity sizing. LCOS reflects actual utilization of built capacity. (Decision 7A — kept current.)
- **Battery**: 4-hour Li-ion, 85% round-trip efficiency, capacity-constrained daily-cycle dispatch. LCOS reflects actual utilization — oversized capacity that sits idle drives cost up. (Decision 7A — kept current.)

---

## 3. Thresholds (20 total — v4.2: added 10/20/30/40 coarse low range + 99.5/99.9 last-mile; v4.3: dropped 99.99)

```
10, 20, 30, 40 [coarse only], 50, 55, 60, 65, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 99.5, 99.9
```

- **10%, 20%, 30%, 40%** (v4.2): Coarse-grid only — no fine zone search, no step1d storage refinement. Captures early adoption / RPS-range where mixes are easy to achieve and cost curves are flat.
- **50%, 55%, 60%, 65%, 70%**: Captures the easy-to-achieve baseline region where most mixes succeed. 5% granularity anchors the cost curve left side.
- 5% intervals from 75–85 (captures broad trend)
- 2.5% intervals from 87.5–97.5 (captures steep cost inflection zone)
- **99%, 99.5%, 99.9%** (v4.2 added 99.5/99.9): Last-mile granularity at the near-perfect end. ≥99.9% is the ceiling, labeled "effectively 100%" (8.76 unmatched hours/year). True 100% is physically unreachable.
- Key inflection behavior (CCS/LDES entering mix, storage costs spiking) captured at 90–97.5
- Dashboard interpolates smoothly between anchor points for abatement curves

---

## 4. Dashboard Controls (7 total — paired toggles)

### Preserved (2):
1. **Region/ISO select** (CAISO, ERCOT, PJM, NYISO, NEISO, MISO, SPP — 7 ISOs)
2. **Threshold select** (20 values: 10, 20, 30, 40 [coarse only], 50, 55, 60, 65, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 99.5, 99.9)

### Sensitivity toggles (7 toggles + 1 binary switch):

Cost sensitivities are organized into 7 graduated toggles (L/M/H) plus one binary policy switch (45Q). CCS and Geothermal are separated from Firm Gen to allow independent sensitivity analysis of these distinct technologies.

| # | Toggle | Options | Controls | Affects |
|---|---|---|---|---|
| 3 | **Renewable Generation Cost** | Low / Medium / High | Solar LCOE + Wind LCOE | Both solar and wind generation costs (regional) |
| 4 | **Firm Generation Cost** | Low / Medium / High | Clean Firm (nuclear) LCOE — uprate + new-build | Nuclear uprate and new-build costs (regional) |
| 5 | **Storage Cost** | Low / Medium / High | Battery LCOS + LDES LCOS | Both storage technology costs (regional) |
| 6 | **CCS Cost** | Low / Medium / High | CCS-CCGT underlying cost (capex, transport, storage) | CCS technology maturity — L=mature/low capex, H=immature/high capex |
| 7 | **45Q Credit** | On / Off | $29/MWh 45Q tax credit offset on CCS LCOE | Binary policy switch — On=full 45Q offset, Off=no offset |
| 8 | **Fossil Fuel Price** | Low / Medium / High | Gas + Coal + Oil prices | Wholesale electricity price + CCS fuel cost + emission rates |
| 9 | **Transmission Cost** | None / Low / Medium / High | All resource transmission adders | Transmission adders on all new-build resources (regional) |
| 10 | **Geothermal Cost** | Low / Medium / High | Geothermal LCOE (CAISO only) | **CAISO only** — no geothermal resource in other ISOs |

**Toggle separation rationale**:
- **CCS separated from Firm Gen**: CCS has a distinct cost structure (capture + transport + storage + fuel) and policy dependency (45Q) that makes it independently variable from nuclear. Pairing them hides the 45Q sensitivity.
- **Geothermal separated and CAISO-only**: Geothermal is a regionally constrained resource — only CAISO has meaningful hydrothermal potential (5 GW cap from USGS identified resources). Other ISOs have zero geothermal potential for power generation. Toggle is hidden/disabled for non-CAISO regions.
- **45Q as binary switch**: The 45Q credit is a policy decision (exists or doesn't), not a cost spectrum. Keeping it binary allows clean analysis of "what if 45Q expires/isn't renewed."

**L/M/H maturity mapping for CCS**:
- **Low**: Mature CCS deployment — nth-of-a-kind plants, established Class VI wells, optimized CO₂ transport networks, low capex
- **Medium**: Mid-range — some learning curve benefits, moderate infrastructure availability
- **High**: Immature/early deployment — first-of-a-kind plants, new well permitting, long transport distances, high capex

**Scenario count**:
- Non-CAISO: 3×3×3×3×2×3×4 = **5,832 cost scenarios** per region per threshold
- CAISO: 5,832 × 3 = **17,496 cost scenarios** per threshold (includes geothermal toggle)
- Total: 17,496 + 5,832×4 = **40,824 scenarios** per threshold set
- All Step 3 (arithmetic on cached physics) — runs in minutes, not hours

**Sensitivity key format**:
- Non-CAISO: `RFSC_QFF_TX` (e.g., `MMMM_1M_M` = all Medium, 45Q on)
- CAISO: `RFSC_QFF_TX_G` (e.g., `MMMM_1M_M_M` = all Medium, 45Q on, Medium geo)
- Q = `1` (45Q on) or `0` (45Q off)

**NOTE**: All graduated toggles use **Low / Medium / High** naming consistently (never "Base" or "Baseline").

**Optimizer approach**: Resource mix co-optimized with costs for EVERY scenario. Different cost assumptions produce different optimal resource mixes — this is the core scientific contribution. Physics cached from Step 1; Step 3 cross-evaluates all EF mixes under each sensitivity combo to find the cheapest valid mix.

---

## 5. Complete Cost Tables

### 5.1 Solar LCOE ($/MWh)

| Level | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|
| Low | $45 | $40 | $50 | $70 | $62 |
| Medium | $60 | $54 | $65 | $92 | $82 |
| High | $78 | $70 | $85 | $120 | $107 |

### 5.2 Wind LCOE ($/MWh)

| Level | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|
| Low | $55 | $30 | $47 | $61 | $55 |
| Medium | $73 | $40 | $62 | $81 | $73 |
| High | $95 | $52 | $81 | $105 | $95 |

### 5.3 Clean Firm LCOE ($/MWh) — Merit-Order Tranche Model (Step 3)

Clean firm cost uses a **merit-order supply curve** with two tranches, filled cheapest-first. The effective LCOE depends on how much clean firm a scenario requires — small amounts are cheap (all uprates), large amounts are expensive (hitting new-build tranche). This is a Step 3 cost calculation applied to the Step 2 efficient frontier.

#### Tranche 1: Nuclear Uprates (Cheapest, Capped)

**Uprate LCOE** (incremental cost of adding capacity to existing plants):
| Level | LCOE ($/MWh) | Basis |
|---|---|---|
| Low | $15 | MUR-dominated (measurement recapture, minimal capital) |
| Medium | $25 | Typical MUR + stretch blend |
| High | $40 | Stretch/small EPU with equipment replacement |

*Sources: INL LWRS Program, NRC uprate database, NEI fleet data, Thunder Said Energy capex analysis, IRA §45Y PTC*

**Uprate cap** — 8% of existing nuclear capacity (includes MUR + stretch + good EPU opportunities):

| Region | Existing Nuclear (GW) | Uprate Cap (GW) | Uprate Cap (TWh/yr @ 90% CF) |
|---|---|---|---|
| **CAISO** | 2.3 (Diablo Canyon) | 0.18 | 1.5 |
| **ERCOT** | 2.7 (South Texas Project) | 0.22 | 1.7 |
| **PJM** | 32.0 (largest US fleet) | 2.56 | 20.2 |
| **NYISO** | 3.4 (Nine Mile, FitzPatrick, Ginna) | 0.27 | 2.1 |
| **NEISO** | 3.5 (Millstone, Seabrook) | 0.28 | 2.2 |
| **Total** | **43.9** | **3.51** | **27.7** |

*8% chosen: NRC has approved ~8% fleet-wide historically (MUR + stretch + EPU). Good EPU opportunities remain across ~27 of 94 reactors, particularly BWR plants. DOE executive order targets ~3-5 GW; INL LWRS estimates 3-8% remaining. 8% reflects full remaining potential including EPU deployment at $15-40/MWh — the cheapest new dispatchable clean capacity available.*

#### Tranche 2: Geothermal (CAISO Only, Capped at 5 GW)

**CAISO only.** Geothermal fills before nuclear new-build, capped at 5 GW (~39 TWh/yr at 90% CF). Based on USGS identified hydrothermal resources (Salton Sea, Imperial Valley, The Geysers). Non-CAISO ISOs have zero geothermal potential for power generation (temperature gradients too low — see §5.4.3).

Geothermal LCOE controlled by **Geothermal Cost** toggle (CAISO only):

| Level | CAISO | Basis |
|---|---|---|
| Low | $63 | Mature hydrothermal flash (Lazard low-end, NREL ATB) |
| Medium | $88 | Blended hydrothermal flash + binary (NREL 2025 Market Report) |
| High | $110 | Binary plants + early EGS (NREL ATB conservative) |

*Sources: NREL ATB 2024, NREL 2025 US Geothermal Market Report, Lazard LCOE+ v18, USGS 2008 Assessment (FS 2008-3082), USGS 2025 Great Basin EGS Assessment.*

**Geothermal cap**: 5 GW = ~39 TWh/yr at 90% CF. Conservative bound using USGS identified hydrothermal only (excludes undiscovered and EGS). After geothermal cap is filled, remaining CAISO clean firm demand falls to Tranche 3 (nuclear new-build) or CCS, whichever is cheaper.

**Non-CAISO geothermal**: Zero. ERCOT has nascent EGS demos (Sage Geosystems) but no operating capacity. PJM/NYISO/NEISO have temperature gradients of 20-25°C/km — far below power generation thresholds. Toggle hidden/disabled for non-CAISO regions.

#### Tranche 3: Nuclear New-Build (Uncapped)

Nuclear new-build LCOE reflects advanced SMR/Gen IV technology. Controlled by **Firm Generation Cost** toggle. For CAISO, this tranche fills after geothermal cap is exhausted. For all other ISOs, this is the first new-build tranche after uprates.

| Level | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|
| Low | $70 | $68 | $72 | $75 | $73 |
| Medium | $95 | $90 | $105 | $110 | $108 |
| High | $140 | $135 | $160 | $170 | $165 |

*Low = nth-of-a-kind SMR deployment target ($70/MWh). Regional variation at Low is minimal (mature deployment compresses cost differences). Medium/High retain larger regional spreads reflecting siting, permitting, and labor differentials. ERCOT lowest (favorable siting/permitting). NYISO highest (siting constraints, labor costs).*

#### Merit-Order Cost Calculation (Step 3 Pipeline)

For each cached scenario's new clean firm demand (above existing grid share), the merit order fills cheapest-first. **CAISO has 4 tranches; other ISOs have 3.**

**Non-CAISO merit order:**
```
new_cf_twh = max(0, total_cf_pct - existing_cf_pct) / 100 × demand_twh
uprate_twh = min(new_cf_twh, uprate_cap_twh)
remaining = max(0, new_cf_twh - uprate_twh)
# Remaining filled by cheapest of: nuclear new-build vs CCS (toggle-dependent)
nuclear_price = NEWBUILD_LCOE[firm_level][iso] + tx_adder
ccs_price = CCS_LCOE[ccs_level][45q_state][iso] + tx_adder
# Each MWh goes to whichever is cheaper
```

**CAISO merit order (includes geothermal tranche):**
```
new_cf_twh = max(0, total_cf_pct - existing_cf_pct) / 100 × demand_twh
uprate_twh = min(new_cf_twh, uprate_cap_twh)
remaining_after_uprate = max(0, new_cf_twh - uprate_twh)
geo_twh = min(remaining_after_uprate, GEO_CAP_TWH)  # 39 TWh cap
remaining_after_geo = max(0, remaining_after_uprate - geo_twh)
# Remaining filled by cheapest of: nuclear new-build vs CCS (toggle-dependent)
```

At low clean firm demand → effective LCOE approaches uprate price ($25/MWh Medium).
At high clean firm demand → effective LCOE approaches new-build price ($88-110/MWh Medium).
The transition point (where uprate cap is exhausted) varies by region — PJM has the most uprate headroom.

**Replaces**: The previous fixed-blend model (§5.3 legacy: `uprate_share × uprate + (1-uprate_share) × new_build`) which applied the same effective LCOE regardless of quantity demanded. The tranche model makes clean firm cost quantity-dependent, which shifts optimal resource mixes at high thresholds.

#### Legacy Blended Values (Preserved for Reference)

Previous blended LCOE (still used in Step 1 physics optimization cache):
| Level | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|
| Low | $58 | $56 | $48 | $64 | $69 |
| Medium | $79 | $79 | $68 | $86 | $92 |
| High | $115 | $115 | $108 | $136 | $143 |

*These are what the Step 1 optimizer used. Step 3 reprices using the tranche model above.*

### 5.4 CCS-CCGT LCOE ($/MWh) — Separate Toggle with 45Q Switch

CCS cost is controlled by two independent toggles: **CCS Cost** (L/M/H maturity) and **45Q Credit** (On/Off).

#### 5.4.1 CCS LCOE with 45Q ON ($/MWh)

| Level | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|
| Low | $58 | $52 | $62 | $78 | $75 |
| Medium | $86 | $71 | $79 | $99 | $96 |
| High | $115 | $92 | $102 | $128 | $122 |

#### 5.4.2 CCS LCOE with 45Q OFF ($/MWh)

45Q OFF = add back $29/MWh offset. Same underlying capex/transport/storage assumptions.

| Level | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|
| Low | $87 | $81 | $91 | $107 | $104 |
| Medium | $115 | $100 | $108 | $128 | $125 |
| High | $144 | $121 | $131 | $157 | $151 |

*ERCOT lowest (Gulf Coast Class VI wells, abundant geology, cheap gas, shortest CO2 transport). NYISO highest (no suitable sequestration geology, longest transport, highest permitting burden).*

**L/M/H maturity mapping**:
- **Low**: Mature nth-of-a-kind CCS, established CO₂ infrastructure, low capex
- **Medium**: Mid-range deployment maturity
- **High**: Immature/early deployment, first-of-a-kind, high capex

**CCS-CCGT cost buildup**:
- Capture cost: ~$30-40/MWh (technology-dependent, relatively uniform)
- CO2 transport: $2-20/MWh (regional — distance to Class VI well)
- CO2 storage: $5-15/MWh (regional — geology, well costs)
- Fuel cost: Heat rate × gas price (responds to gas toggle)
- 45Q offset (when ON): -$29/MWh ($85/ton × 0.34 tCO2/MWh × 95% capture)
- Capture rate: 95%
- Residual emissions: ~0.0185 tCO2/MWh (= 0.37 × 0.05)

**45Q behavioral note**: With 45Q ON, CCS modeled as flat baseload (45Q incentivizes max CF to maximize capture credits). With 45Q OFF, CCS dispatch assumption unchanged in Step 3 (same cached physics), but the cost premium reflects the absence of the policy subsidy.

#### 5.4.3 Regional CCS Capacity Caps (TWh/yr)

CCS-CCGT allocation is capped per ISO based on geologic CO₂ storage availability, infrastructure, and regulatory feasibility — identical pattern to the geothermal cap (`GEO_CAP_TWH = 39.0`) for CAISO. The cap is enforced in Step 3 cost optimization (merit-order tranche logic) and will be propagated to Step 1 mix filtering in the next physics run.

```python
CCS_CAP_TWH = {
    'CAISO': 25.0,    # 11% of 224 TWh demand
    'ERCOT': 200.0,   # 41% of 488 TWh demand
    'PJM':   125.0,   # 15% of 843 TWh demand
    'NYISO': 0.0,     # Hard zero — no geologic storage
    'NEISO': 0.0,     # Hard zero — no geologic storage
    'MISO':  200.0,   # 30% of 660 TWh demand
    'SPP':   50.0,    # 17% of 296 TWh demand
}
```

**Regional justification:**

- **NYISO (0 TWh)**: No suitable onshore CO₂ storage geology. Newark Rift Basin assessed as "low potential" by USGS/NETL. Offshore Atlantic (Baltimore Canyon Trough) is decades from permitting. Zero Class VI well applications filed, state not pursuing primacy.
- **NEISO (0 TWh)**: Crystalline and metamorphic bedrock — zero identified CO₂ storage units in the USGS National Carbon Sequestration Database. No saline formations or depleted reservoirs anywhere in New England. Additionally constrained by winter gas pipeline bottleneck (Step 4 adder).
- **CAISO (25 TWh / 11%)**: Excellent geology (San Joaquin Basin 14–56 Gt, Sacramento Basin ~3 Gt) but SB 905 imposes strictest CCS regulatory framework in US. Zero operating CCS projects, zero CO₂ pipeline infrastructure in-state.
- **SPP (50 TWh / 17%)**: Good geology (Anadarko Basin, Arbuckle Group — 780 Mt P50 in KS) but Oklahoma induced seismicity from underground injection creates regulatory/social resistance. State pursuing but has not received Class VI primacy.
- **PJM (125 TWh / 15%)**: Stark east-west split. Western PJM (WV/OH/western PA) sits on Appalachian Basin (450–500 Gt theoretical); WV received Class VI primacy Jan 2025. Eastern PJM (DC/MD/VA/NJ/DE — majority of demand) has unsuitable Piedmont/Coastal Plain geology. No CO₂ transport infrastructure connecting east to west.
- **ERCOT (200 TWh / 41%)**: Best CCS region in US. Gulf Coast has 20+ Gt depleted offshore fields, 100s Gt offshore saline formations. TX received Class VI primacy Dec 2025 (64 apps from EPA). Denbury CO₂ pipeline network (900+ mi) is densest in US. Multiple storage hubs under development.
- **MISO (200 TWh / 30%)**: Mt. Simon Sandstone (12–172 Gt) is the most characterized formation in US with 2+ Mt successfully injected at ADM Decatur. ND has had primacy since 2018 with 3 active projects. Broadwing 400 MW CCS-CCGT (Google-backed, FID Q2 2026) would be first in US.

**Implementation (Step 3, March 2026)**: CCS cap enforced in Tranche 3 within `price_mix_batch`:
1. **Implicit CCS residual** (`ccs_pct = 100 - sum(cf, sol, wnd, hyd)`): **NOT priced** — tracked for output only. The residual represents unmatched demand served by the existing grid, not a real CCS build decision. Previously this was priced at CCS LCOE which made low-threshold mixes artificially expensive.
2. **Tranche 3 CCS** (clean_firm overflow after uprate + geothermal): CCS headroom = full `CCS_CAP_TWH[iso]` (not reduced by residual since residual isn't built). If CCS is cheaper than nuclear but headroom exhausted, overflow goes to nuclear new-build.
For NYISO/NEISO (cap=0), all tranche 3 CCS → nuclear automatically.

*Sources: USGS National Carbon Sequestration Database (NATCARB), NETL Carbon Storage Atlas V (2015), DOE CarbonSAFE program status (2024–2025), EPA Class VI well permit tracker, California SB 905 (2022), Princeton Net-Zero America (2021), Global CCS Institute Status Report (2024), IEEFA CCS deployment analysis (2024).*

### 5.5 Battery Costs — NREL Component Model + Wright's Law Decline

**Updated March 2026.** Battery costs re-anchored to NREL ATB 2024 component model with Wright's Law learning curves for future cost decline.

**CAPEX derivation** — NREL ATB 2024 separates battery costs into energy ($/kWh) and power ($/kW) components. Total installed cost per kWh = Energy + Power/Duration. This gives the correct 4hr→8hr ratio (~14% cheaper for 8hr, because power electronics spread over 2× the energy capacity).

| Level | Energy ($/kWh) | Power ($/kW) | 4hr Total | 8hr Total | 8hr/4hr |
|---|---|---|---|---|---|
| Low | $170 | $280 | $240/kWh | $205/kWh | 85.4% |
| Medium | $210 | $340 | $295/kWh | $253/kWh | 85.6% |
| High | $270 | $420 | $375/kWh | $323/kWh | 86.0% |

*Low = aggressive LFP procurement + competitive BOS. Medium = typical US utility project (~$295/kWh vs NREL $334 benchmark — reflecting 2025 market reality below NREL's conservative bottom-up model). High = tariff-exposed, constrained interconnection.*

**Financial parameters**: WACC=8%, 20yr life, FOM=2.5% of CAPEX($/kW) per NREL (includes augmentation). Annualized = CAPEX × (CRF + 0.025) / 8760 × 1000 × regional_mult. Regional multipliers: ERCOT=1.00 (cheapest), CAISO=1.11, NYISO=1.18 (highest).

**Annualized capacity costs** ($/MWh-cap, 2025 starting values):

| Level | Type | CAISO | ERCOT | PJM | NYISO | NEISO | MISO | SPP |
|---|---|---|---|---|---|---|---|---|
| Low | 4hr | 3.86 | 3.48 | 3.70 | 4.08 | 3.97 | 3.62 | 3.52 |
| Medium | 4hr | 4.75 | 4.27 | 4.55 | 5.02 | 4.87 | 4.46 | 4.33 |
| High | 4hr | 6.03 | 5.43 | 5.78 | 6.38 | 6.20 | 5.66 | 5.50 |
| Low | 8hr | 3.30 | 2.97 | 3.16 | 3.49 | 3.39 | 3.10 | 3.01 |
| Medium | 8hr | 4.06 | 3.66 | 3.89 | 4.30 | 4.17 | 3.81 | 3.70 |
| High | 8hr | 5.19 | 4.67 | 4.97 | 5.49 | 5.33 | 4.87 | 4.73 |

**LCOS cross-check** (4hr Medium ERCOT, 365 cycles, 85% RTE): **$121/MWh**. Consistent with Lazard 2024 ($115-220/MWh range).

**Wright's Law learning curves** — Battery costs decline from 2025 starting values toward terminal NOAK floor. This is the reverse direction from other technologies (which start at FOAK and decline to NOAK): batteries are already at manufacturing scale, so 2025 IS the starting point. Curves calibrated to NREL 2050 cost projections.

Terminal NOAK ($/kWh): Low=50%, Medium=56%, High=80% of 2025 starting cost.
- Low 4hr: $120/kWh by 2042 | Med 4hr: $165/kWh by 2048 | High 4hr: $300/kWh by 2050
- Low 8hr: $102/kWh by 2040 | Med 8hr: $141/kWh by 2046 | High 8hr: $258/kWh by 2050

Learning curve exponent: 0.6 (concave ramp — steeper initially, asymptotic approach). 8hr reaches NOAK ~2yr faster than 4hr because cell costs (which decline faster) are a larger share of 8hr total cost.

**Trajectory (4hr Medium ERCOT):**
| Year | Wright's fraction | CAPEX | Annualized | LCOS (365 cyc) |
|---|---|---|---|---|
| 2025 | 0.00 | $295/kWh | $4.27/MWh-cap | $121/MWh |
| 2030 | 0.40 | $243/kWh | $3.52/MWh-cap | $99/MWh |
| 2035 | 0.61 | $216/kWh | $3.13/MWh-cap | $88/MWh |
| 2040 | 0.77 | $194/kWh | $2.82/MWh-cap | $79/MWh |
| 2048+ | 1.00 | $165/kWh | $2.39/MWh-cap | $67/MWh |

*Sources: [NREL ATB 2024](https://atb.nrel.gov/electricity/2024/utility-scale_battery_storage), [NREL Cost Projections 2025 Update](https://docs.nrel.gov/docs/fy25osti/93281.pdf), [Ember Battery Storage Costs](https://ember-energy.org/latest-insights/how-cheap-is-battery-storage/), Wright's Law learning rate literature.*

### 5.6 LDES LCOS ($/MWh, 100hr iron-air) — Regionalized

| Level | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|
| Low | $135 | $116 | $128 | $150 | $143 |
| Medium | $180 | $155 | $170 | $200 | $190 |
| High | $234 | $202 | $221 | $260 | $247 |

*ERCOT lowest (Gulf Coast geology for compressed air variants, low labor). NYISO highest (expensive labor, constrained siting, limited geology).*

### 5.7 Transmission Adders ($/MWh, new-build only)

| Resource | Level | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|---|
| **Wind** | Low | $4 | $3 | $5 | $7 | $6 |
| | Medium | $8 | $6 | $10 | $14 | $12 |
| | High | $14 | $10 | $18 | $22 | $20 |
| **Solar** | Low | $1 | $1 | $2 | $3 | $3 |
| | Medium | $3 | $3 | $5 | $7 | $6 |
| | High | $6 | $5 | $9 | $12 | $10 |
| **Clean Firm** | Low | $1 | $1 | $1 | $2 | $2 |
| | Medium | $3 | $2 | $3 | $5 | $4 |
| | High | $6 | $4 | $6 | $9 | $7 |
| **CCS-CCGT** | Low | $1 | $1 | $1 | $2 | $2 |
| | Medium | $2 | $2 | $3 | $4 | $3 |
| | High | $4 | $3 | $5 | $7 | $6 |
| **Battery** | Low | $0 | $0 | $0 | $1 | $1 |
| | Medium | $1 | $1 | $1 | $2 | $2 |
| | High | $2 | $2 | $3 | $4 | $3 |
| **LDES** | Low | $1 | $1 | $1 | $2 | $2 |
| | Medium | $2 | $2 | $3 | $4 | $3 |
| | High | $4 | $3 | $5 | $7 | $6 |
| **Hydro** | All | $0 | $0 | $0 | $0 | $0 |

*ERCOT lowest (CREZ buildout, less congestion). NYISO highest (constrained corridors, siting opposition). Sources: LBNL "Queued Up", MISO/SPP interconnection data.*

### 5.8 Fuel Prices

| Fuel | Low | Medium | High |
|---|---|---|---|
| Natural Gas | $2.00/MMBtu | $3.50/MMBtu | $6.00/MMBtu |
| Coal | $1.80/MMBtu | $2.50/MMBtu | $4.00/MMBtu |
| Oil | $55/bbl | $75/bbl | $110/bbl |

### 5.9 Fuel Price → Wholesale + Emission Rate Impact

**Wholesale**: Shifts based on regional 2025 fossil fuel mix composition. Uses **hourly wholesale price profiles** from EIA 2025 data (not flat averages).

**Wholesale fuel price adjustments** ($/MWh adder to base wholesale, by fossil fuel toggle level):

| Region | Low | Medium | High | Rationale |
|--------|-----|--------|------|-----------|
| CAISO  | -5  |   0    | +10  | ~40% gas generation |
| ERCOT  | -7  |   0    | +12  | ~50% gas, most sensitive to fuel prices |
| PJM    | -6  |   0    | +11  | ~40% gas + coal mix |
| NYISO  | -4  |   0    |  +8  | ~35% gas, more nuclear insulates from fuel |
| NEISO  | -4  |   0    |  +8  | ~35% gas, more nuclear insulates from fuel |

**Emission rate — Regional fuel-switching elasticity**:

| Region | Coal Fleet Status | Switching Elasticity | Rationale |
|---|---|---|---|
| ERCOT | Largely retired (~10GW remaining) | **Low** | Limited coal to switch to; gas price barely shifts emission rate |
| PJM | Substantial remaining (~45GW) | **High** | Gas price ↑ drives meaningful coal resurgence, emission rate jumps |
| CAISO | Near zero | **Very low** | Almost no coal option |
| NYISO | Minimal | **Low** | Small effect |
| NEISO | Minimal (retiring) | **Low** | Small effect |

---

## 9. Existing Grid Mix (2025 Actuals)

### Grid Mix Shares (% of generation):
| Resource | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|
| Clean Firm | 7.9% | 8.6% | 32.1% | 18.4% | 23.8% |
| Solar | 22.3% | 13.8% | 2.9% | 0% | 1.4% |
| Wind | 8.8% | 23.6% | 3.8% | 4.7% | 3.9% |
| Hydro | 9.5% | 0.1% | 1.8% | 15.9% | 4.4% |

### Hydro Caps (2025 actual share of demand, from EIA):
| Region | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|
| Cap (%) | 9.5 | 0.1 | 1.8 | 15.9 | 4.4 |
| 5yr range (%) | 5.2–11.2 | 0.07–0.12 | 1.9–2.1 | 15.9–18.3 | 4.5–7.8 |

**Notes**: Using 2025 actuals (not 5-year average) to match our 2025 snapshot model. CAISO hydro varies enormously by water year (2025 was above average). NYISO imports significant hydro from Quebec/Ontario.

### Wholesale Market Prices (2025 hourly profiles from EIA, not flat averages):
- Average reference points: CAISO ~$30, ERCOT ~$27, PJM ~$34, NYISO ~$42, NEISO ~$41
- Actual hourly data used for storage arbitrage, deficit-hour costing, curtailment economics

---

## 24. Reliability Tax (Sub-Project)

A sub-project quantifying the cost gradient of pushing CFE matching from ~90% to 99.9% over 2025–2050 across the 7 ISOs, comparing four deployment pathways. **Authoritative documentation lives in `reliability_tax/README.md`** — this SPEC section captures only the locked invariants so they survive session handoff. Methodology, pathway implementation, stranding math, and results are TBD in Prompt 1B+ and will be documented here as decisions are made.

### 24.1 Locked Invariants (Prompt 1A)

These are load-bearing for every script, figure, and writeup in the sub-project. Any change requires an explicit user decision and an update both here and in `reliability_tax/README.md`.

1. **Endpoint targets**: CFE ≥ 90%, 95%, 97.5%, 99%, 99.9% by 2050.
2. **Planning horizon**: 2025–2050 (25 years; 26 year-indices including the 2025 baseline).
3. **Pathways**:
   - (1) VRE + batteries only
   - (2a) Behavioral pivot at the 90% CFE plateau
   - (2b) Economic pivot when marginal `$/CFE%` > clean firm LCOE
   - (3) Clean firm proactive from year 1
4. **Clean firm bucket**: Nuclear + CCGT+CCS + geothermal, subject to existing regional constraints (`CCS_CAP_TWH`, `GEOTHERMAL_ISOS = ['CAISO']`). **Offshore wind is NOT clean firm** — it is VRE, available to all pathways.
5. **Cost basis**: Real 2025 USD. No inflation adjustment.
6. **Cost reporting**: Undiscounted cumulative 2025–2050 (primary — actual ratepayer outlay) plus NPV@5%, 7%, 9% real (secondary sensitivity). **Objective = undiscounted cumulative cost.**
7. **ISO scope**: Fully ISO-parametric. Smoke-test on a single ISO, then run all 7.
8. **Stranding scope and thresholds**:
   - Stranded fossil = **new-build gas only** with capacity factor <20% in 2050. Existing fleet is out of scope.
   - VRE stranding = curtailment >30% in 2050.
   - Transmission stranding = underutilized new-build transmission (definition to be refined in 1B).
9. **Demand growth sensitivity**: Section 2 only, reusing the existing L/M/H values from `pipeline_config.DEMAND_GROWTH_RATES`. No new growth rates are invented.

### 24.2 Reusable infrastructure

The sub-project layers on top of the existing 8-step pipeline and reuses (without modifying) `scripts/pipeline_config.py`, `scripts/procurement_utils.py::build_25yr_trajectory`, `data/step2.1-ef/`, `data/step2.2-cost/`, `data/step3-dispatch/`, and `data/step5-wrights/`. See `reliability_tax/README.md` for the full map with file paths and per-asset usage notes.

### 24.3 Status

Prompt 1A (discovery + README) complete. Prompts 1B+ will lock methodology and begin implementation. No scripts, no results, no dashboard page exist yet for this sub-project.

### §24.9 — Argmin objective correction

Design for audit card AUDIT-2026-04-20-1 Option A. Acronyms: net present value (NPV), capital recovery factor (CRF), combined-cycle gas turbine (CCGT), nth-of-a-kind (NOAK), levelized cost of energy (LCOE), efficient frontier (EF), variable renewable energy (VRE), carbon-free energy (CFE).

**1. Unit audit.** `score_ef_batch` multiplies `pct × (demand_mwh/100) × $/MWh-LCOE` — one-year operating dollars. The gas leg in `score_ef_batch_with_gas` is also annual: `new_gas_mw × (NEW_CCGT_COST_KW_YR×1000 + CF×8760×price)`. Units match at the site. Realized objective in `solve_pathway` is the 26-year discounted sum via `multi_rate_npv` at 7% real (Card D, line 122). A year-T argmin sees only year-T dollars, missing that the chosen mix locks vintages through 2050. **Recommendation: convert both legs to NPV dollars.** Multiply EF cost by horizon-remaining NPV factor `NPVF(T) = Σ_{y=T..2050}(1+r)^-(y-2025)`; discount gas capex once at `(1+r)^-(T-2025)`; sum discounted fuel across `min(T+25, 2050)`.

**2. Gas-baseline correction.** Replace static `_gas_raw_2025_worst_hour(iso)` with a year-aware baseline:

```
retired_mw(y) = _twh_to_gw(
                  compute_fossil_retirement(iso, clean_pct(y), emission_rates,
                                            fossil_mix, gf(y), y).gas_displaced_twh,
                  _FOSSIL_CAPACITY_FACTORS['gas']) * 1000
existing_mw(y) = gas_raw_2025_mw - Σ_{y'<=y} retired_mw(y')
new_gas_mw(y) = max(0, gas_raw_mw(y) - existing_mw(y)) * (1 + RA_margin)
```

Candidate row's implied `clean_pct` avoids circularity; memoize per (iso, year).

**3. Discount handling.** `OBJECTIVE_DISCOUNT_RATE = 0.07` real (`step_2_3_pathway_optimizer.py:122`, Card D). CRF = `r(1+r)^n / ((1+r)^n−1)` with `n = NEW_GAS_ASSET_LIFE_YEARS = 25` → CRF ≈ 0.0858. CCGT capex $1,200/kW × CRF ≈ $103/kW-yr, consistent with `NEW_CCGT_COST_KW_YR`. Capex lands in year T at `(1+r)^-(T-2025)`; fuel runs through `min(T+25, 2050)`.

**4. ERCOT sensitivity.** Wind delivered ≈ $54 + $6 transmission = $60/MWh; NOAK-2035 nuclear ≈ $92/MWh. Swap margin ≈ $32/MWh. At the 90% endpoint with 5% worst-hour residual on ~80 GW peak, penalty ≈ 4 GW × NPV(CRF×$1,200/kW + fuel) ≈ $1.0B; wind-to-nuclear NPV swap ≈ $12B over 25 years. **NPV-corrected penalty does not flip ERCOT at the 90% endpoint.** At 97.5% and 99%, residual rises to 15–25% of peak; penalty reaches $3–5B; the swap flips at the 99% endpoint only.

**5. MISO 80%-endpoint diagnostic.** `analysis/reliability-tax/data/MISO/pathway{1,3}_ep80.json` landed in commit `c7868df` (2026-04-18), one day before fix commit `f060d38` (2026-04-19). All 350 manifest entries share that provenance. **The +78% NPV blow-up is pre-fix latent, not caused by `INCLUDE_GAS_COST_IN_ARGMIN=True`.** Re-audit must rerun the optimizer against the fix; open question until then.

**6. PyArrow migration — `step_2_3_pathway_optimizer.py` (2026-04-20).** `import pandas` removed; all parquet I/O now uses `pq.read_table` + per-column `.to_numpy()`. Migrated in two commits on branch `claude/finish-pyarrow-migration-MqhlO`: Phase 1 (EF loader + all EF consumers, commit `95dc598`); Phases 2+4 (deleted dead `score_ef_row_cost`, migrated `load_dg_costs` / `load_fossil_mix` / `load_annual_manifest` / `compute_vre_curtailment_at_endpoint`). Smoke test passes (`import` under `sys.modules['pandas']=None` → OK); `grep pandas|pd\.` → 0 hits. Diag phases 1+3 pass; phases 2/4/5 remain blocked by out-of-scope files (`eia_data_io.py`, `step6_1_smartargets.py`) that still require pandas — parity sweep deferred until those files are migrated or pandas is re-installed in the container.
