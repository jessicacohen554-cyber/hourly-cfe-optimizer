# Advanced Sensitivity Model — Complete Specification

> Historical decisions and landed workstreams live in `SPEC_LOG.md`.

> **Authoritative reference for all design decisions.** If a future session needs context, read this file first.
> Last updated: 2026-04-19 (session: hot-path cache-write regression fix).

## Current Status (Apr 19, 2026)

### Endogenize new-gas-build cost in argmin — ROOT-CAUSE IDENTIFIED / fix PENDING (Apr 19, 2026)

**Root cause of the "idling" probe runs (FOUND this session via code-only exploration).** `run_pathway(cfg)` → `solve_pathway` → `for year in YEARS:` (26 iterations, `scripts/step_2_3_pathway_optimizer.py:2344`) → `size_required_gas_mw` → `worst_hour_gas_sizing` (`scripts/step2_2a_cost_optimization.py:432`) → `_worst_hour_residual_norm` (`:277`) → `_expand_missing_archetypes` (`:357`) → `expand_cache_for_mixes(..., persist=True)` → `save_dispatch_cache(iso, cache, ...)` at `scripts/step3a_build_dispatch_cache.py:287–288` writes the **entire ~100 MB dispatch-cache parquet to disk every time a novel archetype appears in a given year**. That happens up to 26 times per `run_pathway` call. Fingerprint confirms: the ERCOT parquet grew 47 MB → 110 MB just during import side-effects this session. This is invisible to the user (no stdout inside `solve_pathway`) — runs look idle but are actually mid-parquet-write. Secondary cost (~50–200 ms per novel archetype) comes from `reconstruct_hourly_dispatch` reconstructing 8760-hr profiles in `step3a_build_dispatch_cache.py:273–284`, but the parquet write dominates.

**Why this bloats the v2 sweep 5–10× (and the probe 1.5–3×).** 350 runs × up-to-26 writes = up to 9,100 full-parquet serializations for the full sweep. At 0.1–0.3s per write on the 110 MB ERCOT cache, that's 15–45 minutes of pure disk I/O per ISO on top of the compute — regression introduced by the cache-expansion path, not by §24.9 endogenization.

**Caches that ARE hot across runs (verified).** `_EF_CACHE` (`step_2_3_pathway_optimizer.py:256`), `_DISK_DISPATCH_CACHE` (`:3218`), `_WH_STATE` in step2_2a (imported `:234`). None are invalidated inside `run_pathway`. The in-proc driver's claim that "caches stay resident across pathway/endpoint combos for one ISO" is correct — the regression is only inside the solve loop, not across runs.

**Fix plan locked this session (option 1, user preferred).** Thread a `persist=False` override through `_expand_missing_archetypes` and `expand_cache_for_mixes` during the `run_pathway` call; collect every novel archetype in-memory; invoke `save_dispatch_cache(iso, cache, ...)` exactly **once** at the end of `run_pathway`. Same on-disk result, ~1 write per run instead of up to 26. Also patches the v2-sweep regression automatically (once per run × 350 runs = 350 writes instead of up to 9,100). Alternative options considered: (2) pre-expand all 26 years' archetypes before entering the solve loop — more invasive, fine for a future refactor; (3) `persist=False` everywhere for probe-only — fastest but discards the cache growth the full sweep needs.

**What landed before the regression was found (from prior sessions).** SPEC §24.9 implementation (`f060d38`), WHOLESALE_PRICES fix (`fd89c22`), parity probe OFF-branch bit-for-bit on ERCOT P1 ep80 + PJM P3 ep90. ON-branch still unexercised (Lesson 8 rule unmet — see `LESSONS.md`). `scripts/_sweep_pathways_inproc.py` is the right driver shape vs. `run_pathway_sweep.py:227`'s subprocess-per-config loop.

**Pending.**
1. **Implement option 1 (batched persist).** Plan the exact diff:
   - `step3a_build_dispatch_cache.py`: confirm `save_dispatch_cache` is the single persist site (grep).
   - `step2_2a_cost_optimization.py::_expand_missing_archetypes(...)`: accept a `persist=True` kwarg, plumb to `expand_cache_for_mixes`.
   - `step2_2a_cost_optimization.py::expand_cache_for_mixes(...)`: already takes `persist=True` (verified in the exploration). Pass-through.
   - `step_2_3_pathway_optimizer.py::run_pathway` (or `solve_pathway`): wrap the year loop with `persist=False`; after the loop, call `save_dispatch_cache(iso, cache)` once if any archetypes were added.
   - Parity check: flag=OFF probe from §24.9 must still reproduce bit-for-bit after the fix.
   - Hot-path check: time ONE `ERCOT P3 @0.90` run and confirm it's in the 1–3s range, not 10–30s.
2. Re-run the 12-config sanity probe once the fix is verified.
3. Everything else in the §24.9 backlog (extract `stranding_metadata.fleet_size_mw`, P3-vs-P1 delta table, full 350-run v2 sweep, chart-payload regen, dashboard gate, methodogy.md §24.9 entry).
4. After probe + full sweep land: patch `run_pathway_sweep.py:227` to call the in-proc driver (remaining subprocess regression).

**Open questions.** None blocking. If anything, verify `save_dispatch_cache` is thread-safe enough that a single write at end-of-run is OK even if the process is killed mid-solve (probably fine — cold-start will re-expand the missing archetypes, just slower once).

**Resume prompt for next session:** *"SPEC §24.9 code landed (f060d38, fd89c22) but the probe hasn't run because there's a per-year parquet-rewrite regression in the solve loop that makes every run look idle. Root cause (found this session, code-read only): `scripts/step_2_3_pathway_optimizer.py::solve_pathway` `for year in YEARS:` at line 2344 → `size_required_gas_mw` → `worst_hour_gas_sizing` (step2_2a:432) → `_worst_hour_residual_norm` (:277) → `_expand_missing_archetypes` (:357) → `expand_cache_for_mixes(..., persist=True)` → `save_dispatch_cache` (step3a:287–288) writes the full ~100 MB parquet to disk up to 26× per `run_pathway` call. Fix is option 1 (batched persist): thread `persist=False` through `_expand_missing_archetypes` + `expand_cache_for_mixes` during the year loop, call `save_dispatch_cache(iso, cache)` exactly once at end of `run_pathway` if any archetypes were added. Same on-disk result, ~1 write per run. Same regression bloats the v2 350-run sweep, so the fix is doubly load-bearing. Before coding: (a) grep-confirm `save_dispatch_cache` is the only persist site; (b) grep every `persist=` / `expand_cache_for_mixes` call site to make sure threading is correct; (c) run the §24.9 flag=OFF parity probe (ERCOT P1 ep80 + PJM P3 ep90) after the patch — must reproduce bit-for-bit; (d) time ONE `ERCOT P3 @0.90` run post-fix to confirm it's 1–3s, not 10–30s. Only then relaunch the 12-config sanity probe (`pip install pandas && for iso in ERCOT PJM; do python3 scripts/_sweep_pathways_inproc.py --iso $iso --pathways 1,3 --endpoints 0.80,0.90,0.99 --output-root /tmp/reliability_probe_sanity; done`). ERCOT dispatch parquet is gitignored (`.gitignore:69`) and must NOT be committed — it regenerates to >100 MB on every run and will fail GitHub's push limit. Audit memo at `reliability_tax/AUDIT_2026-04-19_step2_3_pathway_optimizer.md` Locked-Decisions remains binding. Lesson 8 (flag-ON parity probe) still pending from §24.9."*

**Prior §24.9 background (full detail in `SPEC_LOG.md`).** SPEC §24.9 implementation (audit-driven fix from `reliability_tax/AUDIT_2026-04-19_step2_3_pathway_optimizer.md` Locked-Decisions):
- `scripts/pipeline_config.py` — new flag `INCLUDE_GAS_COST_IN_ARGMIN = True` (defaults True; flip to False to reproduce pre-change outputs).
- `scripts/step2_2a_cost_optimization.py` — new vectorized adapter `worst_hour_residual_per_row(iso, ef)` (~40 lines). Wraps existing batch `_worst_hour_residual_norm`; pulls EF DataFrame columns into the `arrays` dict shape that function already expects. Returns `(N,)` fractions of annual demand. No Python row loop.
- `scripts/step_2_3_pathway_optimizer.py` — new sibling scorer `score_ef_batch_with_gas(ef, iso, year, config)` (~70 lines). Per-row cost = `row_ef_cost + new_gas_mw × (capex_per_mw + fuel_per_mw)` where `new_gas_mw = max(0, gas_raw_mw − gas_raw_2025_mw) × (1 + RESOURCE_ADEQUACY_MARGIN)`. RA-margin multiplier applied per user's explicit directive. Uses the SAME constants `tax_components_cumulative` uses downstream: `NEW_CCGT_COST_KW_YR[iso]`, `WHOLESALE_PRICES[iso]['Medium']`, `NEW_GAS_REFERENCE_CF = 0.85`. `select_target_mix` call site gated on the flag.
### Hot-path cache-write regression — FIX LANDED (commit `4513cca`, Apr 19, 2026)

Fix for the §24.9 hot-path cache-write regression that bloated every pathway-run wall-clock 5–10× and would have bloated the full 350-run v2 sweep proportionally. Call chain: `solve_pathway` → 26-year loop → `size_required_gas_mw` → `worst_hour_gas_sizing` → `_worst_hour_residual_norm` → `_expand_missing_archetypes(...)` → `expand_cache_for_mixes(persist=True)` → full ~100 MB parquet rewrite of `data/step3-dispatch/<ISO>_dispatch_cache.parquet` on every archetype-batch expansion. Under the ELCC-era call shape, `_expand_missing_archetypes` ran ~once per ISO (archetype sets were small, pre-built by step3a); under §24.9's per-row EF-scorer (`worst_hour_residual_per_row`), it fires 26 years × O(EF-rows) times per pathway run.

Fix:
- `scripts/step2_2a_cost_optimization.py` — `_WH_STATE[iso]` gains a `'dirty'` bool. `_expand_missing_archetypes` now passes `persist=False` to `expand_cache_for_mixes` and flips the dirty flag only when the in-memory cache actually grew. New module function `flush_expanded_cache(iso)` writes the full parquet exactly once per call and clears the flag; no-op if not dirty.
- `scripts/step_2_3_pathway_optimizer.py` — `run_pathway` calls `flush_expanded_cache(config.iso)` once at the pathway-run boundary (after `write_run_json` / `append_to_manifest`). Covers both the P3-pre-solve branch (when target pathway != 3) and the target solve in a single flush.

Preserves SPEC §24.5's cross-session cache-inheritance semantic (parquets are still persisted — just once per pathway run instead of 26+ times).

**What's NOT done.** The sanity probe has still not been launched (4 attempts this session total, all aborted). The fix is in place and pushed, but wall-clock is unmeasured.

**Open decision — three paths for next session.** User flagged three candidates:
- **(A) Launch probe as-is.** Dirty-flag fix alone collapses per-pathway-run parquet writes to 1 (or 0 if cache saturates). Expected wall-clock ~1–1.5 min for 12 configs. Column-pruning / storage-swap become separate work items.
- **(B) Also add column-pruned read.** `_worst_hour_residual_norm` only reads `entry['total_clean']` — the other ~10 dispatch fields in the cache parquet are dead weight on load. A column-pruned `pq.read_table(path, columns=['archetype_key', 'total_clean'])` path in `load_dispatch_cache` (or a dedicated thin-load accessor) would cut cache load ~10×, meaningfully speeding subprocess-spawn paths. ~30 min extra before launching the probe.
- **(C) Storage-format swap.** Bigger surgery: DuckDB append-only, or pyarrow row-group append, to eliminate full-parquet rewrite entirely. Addresses write amplification more thoroughly than the dirty-flag fix and scales better if archetype counts grow. ~1 day + cache-version migration. Probably its own SPEC entry.

Recommendation is (A) → measure → decide on (B)/(C) based on probe output, but user hasn't ruled yet.

**Prior context still valid.** SPEC §24.9 implementation (f060d38), WHOLESALE_PRICES fix (fd89c22), flag=False parity bit-for-bit passing on `ERCOT/pathway1_ep80.json` and `PJM/pathway3_ep90.json`. In-proc driver `scripts/_sweep_pathways_inproc.py` is the correct shape for the probe — not `run_pathway_sweep.py`'s subprocess-per-config loop. Pandas install required on fresh container (`pip install pandas`). Dispatch parquets excluded from commit per `.gitignore:69` (ERCOT_dispatch_cache.parquet is legacy-tracked; untracking deferred until after probe lands).

**Pending (carry-over).**
1. User decision on A / B / C path above.
2. Launch 12-config sanity probe via in-proc driver (`for iso in ERCOT PJM; do python3 scripts/_sweep_pathways_inproc.py --iso $iso --pathways 1,3 --endpoints 0.80,0.90,0.99 --output-root /tmp/reliability_probe_sanity; done`).
3. Extract `stranding_metadata.fleet_size_mw` from 12 JSONs; present P3-vs-P1 gas-fleet delta table per ISO per endpoint.
4. Wait for user approval, then full 350-run v2 sweep per OPS.md.
5. Regenerate 12 chart payloads in `reliability_tax/charts/`.
6. Dashboard-gate check: P3 new-gas < P1 in every ISO at ep90 or ep99 minimum; gap widens ep80→ep99; hump visible; §4 tax in $4–18/MWh range.
7. Patch `scripts/run_pathway_sweep.py:227` from `subprocess.run` to an in-proc `run_pathway(RunConfig(...))` call (same subprocess-spawn amortization regression as the probe; bloats v2 sweep separately from the cache-write regression fixed this session).
8. Append §24.9 entry to `reliability_tax/methodogy.md` after sweep + chart-payload regen. Do NOT frame as reversal of §24.8.

**Key design decisions (unchanged from prior).** `NEW_GAS_REFERENCE_CF = 0.85`. `(1 + RESOURCE_ADEQUACY_MARGIN) = 1.15` multiplier on `new_gas_mw`. §24.8 no-floor for P3 stands. Hump narrative stays (§24.5 descending-side monotonicity is an ep80→ep99 claim; ascending ep0→ep80 carries the hump).

**Open questions.** None blocking once the probe lands. If ERCOT P3 ≥ P1 at every endpoint post-probe, penalty magnitude is too small — tune `NEW_GAS_REFERENCE_CF` down to 0.30–0.40 or raise RA multiplier.

**Pending.**
1. **Launch the 12-config sanity probe via the in-proc driver.** Chunked per (ISO, endpoint) with explicit `timeout` ceilings so one bad chunk doesn't blow the session (prior 4 attempts all timed out on monolithic invocations):
   ```bash
   pip install pandas
   for ep in 0.80 0.90 0.99; do
     for iso in ERCOT PJM; do
       timeout 180 python3 scripts/_sweep_pathways_inproc.py \
         --iso $iso --pathways 1,3 --endpoints $ep \
         --output-root /tmp/reliability_probe_sanity
     done
   done
   ```
   Logs per-run timing to stdout (`[ISO] [n/N] pathway@ep OK cfe=X.XX% in Ts`). ERCOT ep80 is the cold chunk (regenerates `ERCOT_dispatch_cache.parquet` to ~110 MB); subsequent chunks should be ~20–40 s each.
2. Extract `stranding_metadata.fleet_size_mw` from each of the 12 JSONs.
3. Build the P3-vs-P1 gas-fleet delta table per ISO per endpoint.
4. Present results, wait for user approval before full sweep.
5. Full 350-run v2 sweep re-run per `OPS.md` pre-run gate.
6. Regenerate 12 chart payloads in `reliability_tax/charts/`.
7. Dashboard-gate check: P3 new-gas < P1 in every ISO at ep90 or ep99 minimum; gap widens ep80→ep99; hump visible; §4 tax in $4–18/MWh range.
8. Patch `scripts/run_pathway_sweep.py:227` from `subprocess.run` to an in-proc `run_pathway(RunConfig(...))` call (separate subprocess-spawn regression from the cache-write regression fixed this session).
9. Append §24.9 entry to `reliability_tax/methodogy.md` documenting the change. Do NOT frame as reversal of §24.8 (§24.8 no-floor stands).

**Key design decisions locked this session.**
- Expected CF for new-gas fuel term in argmin = `NEW_GAS_REFERENCE_CF = 0.85` (user-selected via AskUserQuestion). Tunable knob — flag if probe penalty looks too large/small.
- `(1 + RESOURCE_ADEQUACY_MARGIN) = 1.15` multiplier on `new_gas_mw` is a deliberate planner-reserve conservatism ON TOP of §24.5's margin-on-demand residual. User directed this explicitly in-session.
- §24.8 no-floor for Pathway 3 stands. `_filter_pathway_3` unchanged. `should_pivot_2a` / `should_pivot_2b` stay dead — endogenization is expected to produce pivot-like behavior via economics.
- Hump narrative stays (§24.5 descending-side monotonicity is a claim about ep80→ep99 only; ascending side ep0→ep80 carries the hump).

**Open questions.** None blocking. If the 12-config probe shows ERCOT P3 ≥ P1 at every endpoint, the penalty magnitude is too small — candidates to tune: drop `NEW_GAS_REFERENCE_CF` to 0.30–0.40, or raise RA multiplier. Defer tuning until the probe lands.

### Pipeline-Audit Sub-Agent — HARDENED (Apr 19, 2026) / live re-run PENDING

**What landed (Apr 18).** `.claude/agents/pipeline-audit-agent.md` — sub-agent for systematic, third-party audits of pipeline code against a plain-language statement of design intent. Workflow: Phase 1 silent reads (`CLAUDE.md`, `SPEC.md` Current Status, every file in the code-reference table for the task, the target file in full, the local methodology doc, ≥3 cached output files); Phase 2 code-vs-intent trace marking each operation Aligned / Silent-assumption / Misaligned / Outside-scope; Phase 3 hypothesis-vs-data trace re-derived directly from cached JSONs (never trusts manifest summaries or README claims); Phase 4 mandatory external sanity checks against NREL ATB, Lazard, EIA AEO, BNEF, IEA, SBTi for every load-bearing parameter; Phase 5 verdict via fixed decision tree producing exactly one of three outcomes — A (sound design + sound results), B (fundamentally flawed methodology), C (sound design but hypothesis does not hold). Invoke with `subagent_type: "pipeline-audit-agent"`.

**Live test outcome (Apr 18 night).** Agent was launched against the full chain (`scripts/step_2_3_pathway_optimizer.py` + `reliability_tax/charts/gen_section{2,3}_*.py` + `dashboard/reliability-tax.html`) and ran past the tool-call budget mid-Phase-4 with no memo written, producing no usable artifact.

**Hardening (Apr 19, this session).** Rewrote Phase 0 around a **running findings log** instead of end-of-run memo dumps. New discipline: (a) open `<feature_dir>/AUDIT_<date>_<target>.md` immediately before Phase 1 with a `## Findings` section header; (b) append a Finding block (`Where` / `What` / `Why it matters` / `Needs follow-up in`) the moment any material observation surfaces — function that misaligns with intent, price-table cell out of range against its cited source, cached number that contradicts the hypothesis, timing constant that creates a mechanism inconsistent with intent, code path that silently changes the counterfactual, stale upstream input; (c) each Finding write is ≤3 tool calls (Read memo, Edit append, optional TodoWrite); (d) keep running after each Finding — writing is not a stopping decision; (e) no tool-call caps — audit runs full rigor across Phases 1-5; (f) PARTIAL path is now only for **hard blockers** (missing cached outputs, unreachable Phase-4 source, missing reference files), not for time-based escape. Verdict A/B/C sections append to the same findings log at the end — no rename step. Communication-style and Hard-rules sections reconciled with the new model.

**Pending for next session.** Re-launch the audit on the narrower-scope slice the user selected earlier: `scripts/step_2_3_pathway_optimizer.py` + `reliability_tax/charts/gen_section{2,3}_*.py` across ERCOT + PJM at endpoints 80% / 90% / 99%, no dashboard HTML in scope. Stated intent and hypothesis are the same as the aborted run: proactive clean-firm build (Pathway 3) plus Wright's Law cost decline should yield materially less new gas than Pathway 1 in every ISO by 2050; observed result is P1 and P3 build essentially the same gas in 6 of 7 ISOs with gas peaking at the 80% endpoint then declining. Suspected sources: SBTi target-year trajectory + NOAK-in-2035 timing, or a bug in `_filter_pathway_3` / `solve_pathway` / `compute_clean_firm_tranches_for_year` / `size_required_gas_mw`.

**AUDIT (2026-04-19):** Verdict B with user-locked fix — endogenize new-gas-build cost (capex + fuel) into `select_target_mix`'s argmin cost function. Current argmin (at `scripts/step_2_3_pathway_optimizer.py:1853`, `nanargmin(costs)`) is EF-row-cost-only; gas is sized downstream in the per-year loop via `size_required_gas_mw`, so the optimizer never "sees" that a pure-VRE row requires 134 GW of new gas in ERCOT. Folding expected new-gas-build cost into each candidate row's score should let P3's NOAK-2035 clean-firm rows naturally beat P1's NOAK-2045 pure-VRE rows in VRE-rich ISOs without hard-wiring a floor. §24.8's no-floor decision STAYS. Hump storyline STAYS (Finding 7 was incomplete — hump lives on the ep0→ep80 ascending side, not violated by the audited ep80→ep99 descending side). Add §24.9 entry documenting the endogenization. P2a/P2b pivot logic stays dead — endogenization is expected to produce pivot-like behavior via economics. See `reliability_tax/AUDIT_2026-04-19_step2_3_pathway_optimizer.md` Locked-decisions section for full direction. Next step: coding-session handoff. Findings log at `reliability_tax/AUDIT_2026-04-19_step2_3_pathway_optimizer.md` (114 lines) captured Phase 1-4 in full: Phase 2 code trace (3 HIGH-severity findings on P3 having no clean-firm forcing mechanism — `_filter_pathway_3` is a no-op, P2a/P2b/P3 differ only in NOAK year, SBTi year-snap lock-steps P1 and P3 onto the same EF band); Phase 3 data re-derivation from ERCOT + PJM cached JSONs confirming P1 ≡ P1a ≡ P3 bit-for-bit in ERCOT at every endpoint and P3 ≡ P1 in PJM at ep80 but diverging at ep90/ep99; Phase 4 external checks finding cost/learning parameters in-range (aggressive end of ATB/Lazard but defensible). Two compounding methodology issues surfaced: (A) no forcing mechanism makes P3 build clean firm proactively — it only sees cheaper clean firm via NOAK-2035 and declines to use it in VRE-rich ISOs; (B) worst-hour gas sizing per SPEC §24.5 is monotone-decreasing by construction, so the "gas hump" storyline on the dashboard contradicts what the code is actually designed to produce. These are the intervention points for a Verdict B memo, but the agent never emitted the formal decision-cards / handoff-prompt section. **Pending next session: either (i) re-launch the agent with a tiny continuation prompt to do Phase 5 classification + Phase 6 output on the existing findings log, or (ii) hand the findings directly to a `subagent_type: coding-session` as the intervention list.**

**Resume prompt for next session:** *"Pipeline-audit sub-agent is landed at `.claude/agents/pipeline-audit-agent.md` and was hardened Apr 19 with findings-log discipline (no tool-call caps, each finding writes in ≤3 tool calls the moment it surfaces, PARTIAL only fires on hard blockers). Re-launch the audit against `scripts/step_2_3_pathway_optimizer.py` + `reliability_tax/charts/gen_section{2,3}_*.py` restricted to ERCOT + PJM at endpoints 80% / 90% / 99%. Stated intent: proactive clean-firm build (Pathway 3) + Wright's Law cost decline should yield materially less new gas than Pathway 1 in every ISO by 2050. Hypothesis: P1 and P3 should diverge at the 80% endpoint, not converge. Observed result: P1 and P3 build essentially the same gas in 6 of 7 ISOs; gas peaks at 80% then declines. Invoke with `subagent_type: \"pipeline-audit-agent\"`. Expect either Verdict B with a findings log at `reliability_tax/AUDIT_<date>_step_2_3_pathway_optimizer.md` plus a coding-session handoff prompt, Verdict A with the findings log as audit trail, or Verdict C with a mechanism explanation."*

### Coding-Session Sub-Agent — LANDED (Apr 18, 2026, late evening)

**What landed.** `.claude/agents/coding-session.md` — sub-agent for analysis / code tasks in this project. Enforces the three-phase workflow from `CLAUDE.md`: (1) silent orient (read `CLAUDE.md`, `SPEC.md` current status, `LESSONS.md`, `OPS.md` if heavy compute, plus every file in the code-reference table for the task category, plus the reference page from the reference-page table); (2) structured plan with mandatory sections — restated task, insertion point in the 8-step pipeline (upstream producers / downstream consumers), reference anchors, data flow including existing-cache reuse, methodology decisions flagged for approval, performance plan (vectorized kernel signature, `numba @njit` decision, expected iteration count), reuse & drift (existing utilities to import, duplicates flagged as promotion candidates), validation — then wait for explicit OK; (3) implement with TodoWrite and run the promised validation. Hard rules baked in: never loop over data arrays >1k rows (vectorize first, canonical exemplar `fleet_dispatch.py`), use `numba @njit` only when numpy can't express the kernel, load caches once and slice, grep before writing helpers, never fork a utility to add a flag, honor step boundaries in `PIPELINE.md`. Load-bearing methodology choices (capacity metric, cost basis, time-binning, dispatch ordering, retirement rule, counterfactuals, aggregation level, weather year, fuel-price trajectory) require explicit approval before code is written. Invoke with `subagent_type: "coding-session"`.

**Resume prompt for next session:** *"Coding-session sub-agent landed in `.claude/agents/coding-session.md`. Invoke it for any new analysis / pipeline code via `subagent_type: \"coding-session\"`. The agent enforces a plan-then-approve-then-implement workflow, reads the code-reference table in `CLAUDE.md` to locate exemplars, and blocks on methodology approval before writing code. Next natural work: either (a) return to the reliability-tax project-infra workstream below (CLAUDE.draft migration, OPS.md creation, `/fix-prose` test-drive), or (b) take the coding-session agent for a real spin on a pipeline task."*

> Older status blocks moved to `SPEC_LOG.md` (Apr 18, 2026 archive cut; Apr 19 rotations moved the oldest reliability-tax redesign block + the Apr 18 project-infra block to SPEC_LOG). See that file for the historical decision log.
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
6. **Cost reporting**: Undiscounted cumulative 2025–2050 plus NPV@5%, 7%, 9% real. **Objective = NPV@7%.**
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
