# P3 Foresight — Design Memo

> Supersedes `reliability-tax_prompt-pack.md` as the methodology artifact for adding an endpoint-aware solver mode to Pathway 3.
>
> **Scope.** Design only. No code changes.
>
> **Out of scope.** P1 clean-firm forcing mechanism (separate session).

---

## 1. Critique of the prompt pack

See master for §1 (commit 508187a).

---

## 2. Restatement of the research question

What you need to build depends on where you start (an existing clean-tilted fleet with ~8–32% clean-firm share depending on ISO) and where you're going (deep grid decarbonization, ≥90% CFE by 2050). Pathway 1 represents the myopic counterfactual: year by year, pick the cheapest incremental mix that hits that year's CFE rung, without considering the 2050 endpoint. The penalty of that myopia shows up as the reliability tax — new gas buildout to firm variable generation, stranded gas capex when demand or policy shifts, curtailment, expensive last-mile firming at 95–99% CFE. Pathway 3 represents the endpoint-aware planner: the planner knows it's aiming for ≥90% CFE and builds clean firm proactively when doing so avoids gas buildout and avoids stranding, even when clean firm isn't the cheapest year-by-year increment. The comparison headline is the penalty of P1's missing foresight: how much more expensive P1 is than P3 in cumulative cost, reliability tax, gas fleet size, stranded capex, and time-to-90%.

**This memo's scope is narrower than the research question.** It is about adding an additive foresight layer to P3 so that P3's trajectory starts to diverge from the current market-myopic default. The current cached state where P1 ≈ P3 on ERCOT at ep90/ep99 is the expected baseline — neither pathway has foresight right now, and both argmin to the same mix at each year because the cost signal is the same for both. That convergence is correct given the current code. When the foresight layer lands, the argmin for P3 changes and P3 separates from P1. The comparison headline surfaces at that point, not before. P1's own representation of the myopic counterfactual is being addressed separately and is out of scope here.

---

## 3. Architecture decision

Four candidate architectures for where the foresight layer lives:

**(i) Same-script solver-mode flag on `solve_pathway`.** Add `RunConfig.solver_mode ∈ {'myopic', 'foresight'}`, default `'myopic'`. Inside `solve_pathway`, branch on the flag before the year loop. Pros: single source of truth, no duplication. Cons: ~150 lines added to a ~200-line solver, risk of silent drift between paths.

**(ii) Separate module `step_2_3_pathway_foresight.py`.** Foresight in a new file; shared primitives extracted into `step_2_3_shared.py`. Pros: cleanest separation. Cons: large refactor not obviously needed; doubles test surface.

**(iii) Post-hoc wrapper.** A `foresight_for_pathway(cfg)` function runs outside `solve_pathway` to produce a target trajectory, injected as an additional pathway-mask constraint. Pros: solve_pathway untouched. Cons: "target as constraint" handshake is fragile against the ratchet + fallback cascade; rules out soft-penalty algorithms.

**(iv) Runtime overlay — `solve_pathway_with_foresight(cfg)`.** A new entry point that calls shared primitives directly and owns a foresight-aware year loop end-to-end, reusing `_finalize_run`. `_run_one` dispatches on `cfg.solver_mode`. Pros: zero regression risk on myopic; algorithm can iterate freely; finalization shared. Cons: duplicates the year-loop skeleton (~60 lines).

### Recommendation

**Option (iv) — runtime overlay.**

1. **Zero regression risk on myopic.** The cached 350-run matrix is the null-state baseline and must not drift.
2. **The premise implies P3 *is* the planned pathway.** Myopic-P3 is a diagnostic; foresight-P3 is the canonical production pathway. `solver_mode` flag makes this explicit.
3. **The foresight algorithm will iterate.** §4 enumerates five candidates. Option (iv) lets iteration happen in one file without touching shared code.
4. **P2a/P2b need foresight variants too, eventually.** Same overlay wraps them with a one-line dispatch change.

### Tradeoff acknowledgment

Mitigation for the duplicated year-loop: factor the sunk-cost-scorer inner body into `_score_year_sunk_cost(...)` that both solvers call. Scoring logic stays single-source; control flow is independent.

---

## 4. Algorithm design

The foresight layer has to change the year-by-year argmin so that the decision at year `y` depends on a target endpoint `(endpoint_year, endpoint_pct)` rather than only on year-`y` cost. Five candidate algorithms, ordered from expensive-exact to cheap-heuristic.

### (a) Terminal-anchored rollouts (prompt pack's choice)

**Planner methodology.** Lookahead via forward simulation. At each year `y` and each candidate mix `m`, simulate the full trajectory `y+1 → endpoint_year` under the current rules (ratchet, cascade, sunk-cost scorer) assuming `m` is chosen this year. Score `m` by the cumulative discounted cost of the entire rollout, not just year-`y` cost. Pick the `m` that minimizes the path integral.

**Interaction with existing semantics.**
- *Absolute-TWh vintages.* Clean. Each rollout carries its own provisional ledger forward; the outer loop commits only the chosen `m`'s year-`y` tranches.
- *Sunk-cost B scorer.* Each rollout year uses the scorer; forward-year scorer calls re-price the (so-far provisional) capacity at $0 once committed inside that rollout.
- *Floor ratchet.* Rollouts must honor the ratchet forward; decisions that would violate the ratchet at any future year are infeasible.
- *Fallback cascade.* The 4-tier cascade fires inside rollouts wherever target-pct is infeasible at cost-binding physical mins.
- *Pathway NOAK window.* The year-by-year cost used in rollouts respects `NOAK_YEAR_BY_PATHWAY[pathway]`, so P3 rollouts see cheaper clean-firm new-build after 2035 than P2b rollouts see after 2040.

**P3 divergence from current-P3.** Interior trajectory changes dramatically: clean-firm commits move forward in time to pre-empt stranded gas that the rollout can see coming. Endpoint (2040/2045/2050) mix changes modestly — the ratchet already forces the physical minimum at endpoint — but the *cost composition* of getting there is very different.

**Degenerate case.** Set the rollout discount rate to infinity (or rollout horizon to 1 year) → only year-`y` cost matters → recovers myopic.

**Why not this one.** Rollout cost is roughly O(|candidates| × horizon × year_loop_cost). With ~100 candidates per year × ~25-year horizon × ~100 ms per scorer call = ~4 minutes per ISO-endpoint per outer year, times ~25 outer years times 56 runs → ~93 hours of wall-clock serial. Parallelizable, but the per-rollout cost also has to re-run the feasibility cascade, which is itself a fixed-point loop. Cost-per-iteration is the blocker, not cost-per-run.

### (b) Single-pass argmin with lookahead penalty (recommended)

**Planner methodology.** Augment the myopic scorer with a penalty that measures distance from a precomputed `endpoint_target_shares` vector. The year-`y` score for candidate mix `m` becomes:

```
score_y[m]  =  myopic_score_y[m]
             +  λ · w(y, endpoint_year) · ‖ shares[m] − endpoint_target_shares ‖₂²
```

where `shares[m]` is the resource-share vector implied by mix `m` projected to `endpoint_year`, `endpoint_target_shares` is a static reference mix chosen once per run (see §8 open question iii), and the time-weight

```
w(y, endpoint_year)  =  (endpoint_year − y) / (endpoint_year − start_year)
```

is 1 at `start_year` and 0 at `endpoint_year`. This biases early-year investment to move toward the endpoint mix — which is when the investment has the most leverage — and relaxes as the ratchet starts doing the enforcement work late in the horizon.

The prompt pack's original form used `/25` (fixed 2050 horizon). Under the 4-endpoint collapse — `{(2040,90), (2045,95), (2050,99), (2050,99.9)}` — the horizon is per-endpoint, so the divisor generalizes to `(endpoint_year − start_year)`.

**Interaction with existing semantics.**
- *Absolute-TWh vintages.* Fully compatible. Penalty is added to the scorer; ledger accumulation is unchanged.
- *Sunk-cost B scorer.* Additive. The penalty does not interact with sunk-cost pricing; committed vintages still price to $0 in the myopic term, and the penalty operates on share distance independent of sunk-cost state.
- *Floor ratchet.* Untouched. The ratchet enforces non-degrading cumulative clean share year-over-year. Penalty changes *which* feasible mix is picked from among ratchet-compatible candidates; infeasible mixes are still discarded.
- *Fallback cascade.* Untouched. Penalty is evaluated only on cascade-surviving candidates. When the cascade forces retreat to a physical-minimum mix, the penalty is computed but does not change the outcome (single candidate).
- *Pathway NOAK window.* Compounds cleanly. P3's `NOAK_YEAR=2035` already makes clean-firm cheaper on the myopic term after 2035; the penalty *additionally* steers clean-firm in earlier (pre-2035) years where the myopic term alone doesn't. The effect is P3 pulls clean firm *forward* past the point where NOAK pricing alone would.

**P3 divergence from current-P3.** Interior trajectory shifts clean-firm commits earlier than the NOAK-2035 crossover would on its own. Endpoint mix at (2040,90) / (2045,95) / (2050,99) / (2050,99.9) converges on `endpoint_target_shares` up to the ratchet's post-hoc enforcement. Current-P3 *is* myopic-P3 even though P3 has NOAK=2035 — the argmin is year-by-year myopic and sees only the current-year cost; NOAK just shifts *when* the myopic crossover happens. Foresight-P3 additionally steers by endpoint proximity and hits the endpoint via a different interior trajectory.

**Degenerate case.** λ = 0 → recovers myopic exactly. λ → ∞ → single-year snap-to-endpoint (pathological, triggers cascade). Moderate λ (swept in §8 open question ii) is the production setting.

**Why this one.** O(1) additional cost per candidate per year. No rollouts, no fixed-point iteration. Fully compatible with the existing `_score_year_sunk_cost` factoring. Runs on the full 56-run matrix in the same wall time as myopic.

### (c) Receding-horizon dynamic programming

**Planner methodology.** Backward induction from `endpoint_year`. State vector: cumulative committed absolute-TWh per resource class + year. Value function: minimum cumulative cost from state `s` at year `y` to reach `endpoint_pct` at `endpoint_year`. Policy: at each `(s, y)`, the decision that minimizes `cost_y + V(s', y+1)`.

**Interaction with existing semantics.**
- *Absolute-TWh vintages.* State space is continuous over resource TWh accumulations — requires discretization. Binning granularity trades policy quality vs. state-space size.
- *Sunk-cost B scorer.* DP cost function can encode sunk-cost pricing inside `cost_y`, but the state has to carry enough vintage information to know what's already committed. Substantially enlarges the state vector.
- *Floor ratchet.* DP transitions are restricted to ratchet-compatible successors. Tractable but prunes the action space.
- *Fallback cascade.* Disjunctive fallback rules don't fit cleanly into DP; each cascade tier becomes a separate transition class.
- *Pathway NOAK window.* Encoded in year-`y` cost function.

**P3 divergence from current-P3.** Provably optimal within the discretization — the benchmark against which heuristic (b) is compared.

**Degenerate case.** Horizon = 1 → recovers myopic.

**Why not this one.** State-space explosion with continuous TWh accumulations + disjunctive cascade. Engineering cost dominates value for a design memo; if (b) proves insufficient we revisit DP as the principled upgrade.

### (d) Terminal-constraint pre-commit / inverse-ratchet

**Planner methodology.** Backcast. Fix a required cumulative build schedule `target_twh_r(y)` for each resource class `r` over `y ∈ [start_year, endpoint_year]`, such that endpoint constraints are met and year-over-year deltas are monotone. Treat these as an *additional* set of floors injected above the existing ratchet floors.

**Interaction with existing semantics.**
- *Absolute-TWh vintages.* Compatible — additional floors act directly on the cumulative TWh ledger.
- *Sunk-cost B scorer.* Unchanged; scorer sees the injected floors as the new binding constraint.
- *Floor ratchet.* The injected floors replace the usual "non-degrading" ratchet floor with a stricter schedule; ratchet semantics still apply.
- *Fallback cascade.* Heavily stressed. Injected floors may be infeasible in early years at cost-binding physical mins → cascade fires often → results are cascade-driven, not foresight-driven.
- *Pathway NOAK window.* Interacts poorly — the backcast is computed against endpoint prices, so early-year NOAK-expensive commits are forced before NOAK pricing arrives.

**P3 divergence from current-P3.** Aggressive early clean-firm commits. But brittle: the backcast schedule is itself a search problem, and miscalibration triggers frequent cascade retreat.

**Degenerate case.** Pre-commit schedule identical to the existing ratchet → recovers myopic.

**Why not this one.** Backcast calibration is a research problem of its own, and the interaction with the fallback cascade is unpredictable. We would be replacing one heuristic with another while adding fragility.

### (e) MILP over trajectories

**Planner methodology.** Mixed-integer linear program over the full `(year × resource × tranche)` decision matrix, with endpoint constraints, ratchet constraints, and feasibility constraints expressed as linear inequalities. Solve to global optimum.

**Interaction with existing semantics.**
- *Absolute-TWh vintages.* The ledger's nonlinear feasibility checks (VRE contribution caps, ELCC curves, hybrid dispatch bounds) don't linearize without substantial approximation.
- *Sunk-cost B scorer.* Sunk-cost-$0 pricing for committed vintages is piecewise-linear and can be encoded with auxiliary binaries — expensive but feasible.
- *Floor ratchet.* Linear.
- *Fallback cascade.* Disjunctive (4-tier) — requires big-M or SOS1 encoding, both of which slow the solve.
- *Pathway NOAK window.* Piecewise-linear year-cost — encodable.

**P3 divergence from current-P3.** Provably optimal.

**Degenerate case.** Remove endpoint constraint → recovers myopic (year-decoupled LP).

**Why not this one.** Encoding the full feasibility cascade + ELCC nonlinearities + absolute-TWh semantics as MILP is an 8–12 week research project. Out of scope.

### Recommendation

**(b) Single-pass argmin with lookahead penalty.**

Reasoning:
1. Zero interaction risk with the sunk-cost scorer, ratchet, and cascade — the penalty is additive to an existing scalar score.
2. Wall-time identical to myopic — the 56-run matrix (4 endpoints × {P1, P3} × 7 ISOs) runs in the same budget.
3. Tunable via a single λ and a time-weight `w(y, endpoint_year)`; sensitivity is inspectable.
4. Degenerate at λ = 0 to exactly myopic-P3 — the null-state baseline is preserved as a free by-product of the same code path.
5. Compounds with P3's existing `NOAK_YEAR=2035` Wright's Law advantage rather than competing with it.

**Note on P3's current state.** P3 already has a foothold toward endpoint-awareness via `NOAK_YEAR_BY_PATHWAY['3']=2035` (vs. P2b=2040, P2a=2045) in `scripts/pipeline_config.py:1440`. That sets earlier Wright's Law maturation for clean-firm new-build. But the argmin at each year is still myopic — it picks the cheapest current-year mix, and NOAK only changes *which year* clean firm becomes cheapest. Foresight (b) adds a steering term on top, independent of the NOAK schedule, that biases early-year investment toward the endpoint mix regardless of current-year crossover timing.

---

## 5. Interaction with existing semantics

Code-level reconciliation for the recommended algorithm (b). Each subsection names the existing mechanism, what it does today, and where foresight plugs in.

### 5.1 Absolute-TWh vintage ledger — `_build_ledger` (L1294–1374)

**Today.** `_build_ledger` accumulates clean-firm tranches per-vintage-year in absolute TWh, not shares. Each year's solve adds tranches to the ledger; the ledger is the source of truth for "what has already been committed." Subsequent scorer calls read absolute committed TWh by (resource, vintage) and use it for sunk-cost pricing.

**Foresight hook.** None required in the ledger itself. Foresight operates on the *candidate* mix `m` *before* it's added to the ledger. `shares[m]` is computed by projecting the hypothetical-post-`m` ledger state to `endpoint_year`'s total TWh demand and dividing per-resource TWh into resource shares. The ledger's commit semantics are unchanged — only the argmin that decides *which* `m` gets committed is rewritten.

**Invariant preserved.** `_build_ledger` output is bit-identical when λ = 0.

### 5.2 Sunk-cost B scorer (L1122–1131)

**Today.** `score[m] = Σ_r max(0, target_twh_r[m] − floor_twh_r) × LCOE_r_y`. Only the *incremental* TWh above the ratchet floor is priced; already-committed vintages price to $0 (sunk-cost Interpretation B). `LCOE_r_y` comes from `year_adjusted_cost(r, y, pathway)` which honors the pathway-specific NOAK window.

**Foresight hook.** Additive. `score_y[m] = score_sunk_cost[m] + λ · w(y, endpoint_year) · ‖ shares[m] − endpoint_target_shares ‖₂²`. The sunk-cost term is untouched; the penalty is a second scalar added after the existing Σ.

**Code surface.** §3 already calls for factoring the L1122–1131 inner body into `_score_year_sunk_cost(mix, floors, twh_total, year, pathway) → score`. Foresight adds `_score_year_with_endpoint(mix, floors, twh_total, year, pathway, λ, target_shares) → score` that wraps it:

```
def _score_year_with_endpoint(mix, floors, twh_total, year, pathway, λ, target_shares, endpoint_year, start_year):
    sunk = _score_year_sunk_cost(mix, floors, twh_total, year, pathway)
    shares = _project_shares_to_endpoint(mix, endpoint_year)   # §5.6
    w = (endpoint_year - year) / (endpoint_year - start_year)
    return sunk + λ * w * np.sum((shares - target_shares) ** 2)
```

**Invariant preserved.** λ = 0 → identical numerical score.

### 5.3 Floor ratchet (L1082–1167)

**Today.** At each year `y`, a floor is computed per resource as `max(prior_year_committed_twh, minimum_physical_twh_for_target_pct)`. The ratchet enforces non-degrading cumulative commits year-over-year. Candidate mixes that fall below the floor for any resource are infeasible.

**Foresight hook.** None. The ratchet acts as a feasibility filter on candidates *before* the scorer is called. Foresight runs on the surviving candidate set. Any mix that the penalty would prefer but the ratchet rejects is discarded, same as under myopic.

**Invariant preserved.** Feasibility set at each year is identical to myopic. Foresight only re-ranks the feasible set.

**Watch item.** If the penalty pushes the solver into a mix whose year-`y+1` ratchet floor cannot be met with a feasible mix, the cascade (§5.4) activates. Sensitivity testing at high λ must confirm the cascade fires no more often than under myopic — otherwise we're trading foresight for cascade-driven retreat, which is not the design intent.

### 5.4 4-tier fallback cascade (L1133–1151)

**Today.** If the target-pct is infeasible at cost-binding physical mins, the cascade retreats in order: (i) relax clean-firm dispatch floor, (ii) allow gas firming, (iii) allow existing-gas CF expansion, (iv) allow target-pct relaxation down to the ratchet's minimum achievable.

**Foresight hook.** None in the cascade itself. Foresight re-ranks among tier-(0) feasible candidates; when the cascade fires, tier-(i) … tier-(iv) each produce a single forced mix (or a tiny set), and the penalty evaluates on that set without materially changing the outcome. The cascade retains its role as the feasibility-retreat mechanism.

**Invariant preserved.** Cascade activation triggers are unchanged; cascade output under λ = 0 matches myopic cascade output.

**Diagnostic.** Emit `cascade_activations_by_year` in the schema so we can confirm foresight does not inflate cascade firing rate versus myopic.

### 5.5 Pathway-specific NOAK windows — `pipeline_config.py:1440`

**Today.** `NOAK_YEAR_BY_PATHWAY = {'1': 2045, '2a': 2045, '2b': 2040, '3': 2035}`. `year_adjusted_cost(r, y, pathway)` applies Wright's Law learning curves referenced to `NOAK_YEAR_BY_PATHWAY[pathway]`, so P3 sees cheaper clean-firm new-build starting in 2035 while P2b waits until 2040.

**Foresight hook.** None. The sunk-cost term picks up NOAK pricing via `year_adjusted_cost`; the penalty term is cost-dimensionless (share distance, scaled by λ). Foresight and NOAK operate on orthogonal axes: NOAK changes the *cost* component, foresight changes the *steering* component.

**Compounding behavior.** Before 2035, only foresight steers P3 toward endpoint shares — the NOAK term offers no discount yet. From 2035 onward, the sunk-cost term alone already favors clean-firm (NOAK cheaper); the penalty's remaining work is to push past the point where NOAK pricing would stop before hitting `endpoint_target_shares`. This is why foresight-P3 will diverge from myopic-P3 more in the pre-2035 years than in the post-2035 years.

### 5.6 Tranche decomposition — `decompose_clean_firm_tranches` (L787)

**Today.** Given a candidate mix, `decompose_clean_firm_tranches` splits clean-firm TWh into the five tranche classes (`existing`, `uprate`, `geo`, `nuke_new`, `ccs`) based on pathway rules and year. The vintage ledger is keyed on these tranche classes.

**Foresight hook.** `_project_shares_to_endpoint(mix, endpoint_year)` — a new helper — extrapolates the candidate mix forward to `endpoint_year` total demand and computes per-tranche shares. For candidates that respect the ratchet at year `y`, the projection assumes no further commits (so the projected share for `m` is `(committed_before_y + delta_y[m]) / twh_total(endpoint_year)`). This is a cheap linear projection, not a forward solve; it answers "if we commit `m` today and freeze, what shares land at `endpoint_year`?"

**Why linear-freeze rather than continue-forward.** The penalty is a steering term, not a forecast. What we need is a gradient — "does committing `m` at year `y` move us toward or away from the endpoint?" The linear-freeze projection gives the right sign and monotonicity without requiring a nested solve. If forward-projection quality matters, algorithm (a) or (c) is the right upgrade.

### 5.7 `_pathway_mask` (L727) and `solve_pathway` (L991–1195)

**Today.** `_pathway_mask` filters the candidate mix set to those permissible for the pathway (e.g. P1 excludes nuclear new-build in early years). `solve_pathway` owns the year loop and calls the scorer + ratchet + cascade.

**Foresight hook.** `solve_pathway_with_foresight(cfg)` (new, per §3 option iv) duplicates the year-loop skeleton and swaps the scorer call for `_score_year_with_endpoint`. `_pathway_mask` is unchanged — foresight does not alter which mixes are pathway-permissible.

**Invariant preserved.** Under λ = 0, `solve_pathway_with_foresight` produces bit-identical output to `solve_pathway` for the same `cfg`. This is an explicit regression test (§7 exit gate).

---

## 6. Comparison framework

The framework compares three solver modes at each of the 4 endpoints × 7 ISOs = 28 `(endpoint, ISO)` cells:

| Mode | What it represents | Current status |
|---|---|---|
| **P1 (myopic)** | Year-by-year least-cost with no endpoint awareness. The BAU counterfactual. | Cached (current `pathway1_*.json`). Forcing mechanism for P1 being revised in a separate session. |
| **Myopic-P3** | Current P3: argmin year-by-year; NOAK=2035 but no endpoint steering. | Cached (current `pathway3_*.json`), equivalent to foresight (b) with λ = 0. |
| **Foresight-P3** | Recommended algorithm (b): penalized argmin that steers toward `endpoint_target_shares`. | New. Produced by `solve_pathway_with_foresight`. |

The within-P3 comparison (myopic-P3 vs foresight-P3) is the *direct product* of this memo. The P1-vs-foresight-P3 comparison is the *downstream headline* that surfaces once P1 is revised.

### 6.1 Per-year plots (one panel per `(endpoint, ISO)` cell)

For each cell, four stacked-panel time series over `[start_year, endpoint_year]`:

1. **Achieved CFE share** — line per mode. Shows trajectory to endpoint.
2. **Gas fleet MW (existing + new, total)** — line per mode. Shows whether foresight avoids gas buildout.
3. **System cost $/MWh (undiscounted, levelized to annual demand)** — line per mode. Shows year-by-year cost differences.
4. **Resource stack (TWh) — stacked area** — one panel per mode (three panels side by side). Shows composition differences directly.

Scrollytell section in the dashboard presents one cell at a time with mode toggles.

### 6.2 Tables — per `(endpoint, ISO)` cell

| Metric | P1 | Myopic-P3 | Foresight-P3 | Δ (myopic → foresight) |
|---|---|---|---|---|
| Cumulative undiscounted cost ($B) | … | … | … | … |
| Cumulative discounted cost ($B, r=0.07) | … | … | … | … |
| Reliability tax ($/MWh-yr, averaged) | … | … | … | … |
| Gas fleet peak MW | … | … | … | … |
| Stranded gas capex ($B) | … | … | … | … |
| Time-to-90% CFE (year) | … | … | … | … |
| Endpoint mix — existing clean (%) | … | … | … | … |
| Endpoint mix — new clean firm (%) | … | … | … | … |
| Endpoint mix — new VRE + storage (%) | … | … | … | … |
| Endpoint mix — gas firming (%) | … | … | … | … |
| Cascade activations (count) | … | … | … | … |

**Note on time-to-90%.** At endpoint `(2040, 90)` this metric equals `endpoint_year` by construction for any feasible run. At endpoints `(2045, 95)`, `(2050, 99)`, `(2050, 99.9)` it's informative — foresight-P3 should hit 90% earlier than myopic-P3 if the steering is working.

### 6.3 Headline metric

Primary headline (within-P3 foresight penalty):

```
foresight_penalty_vs_myopic  =  (cum_undisc_cost_myopic  −  cum_undisc_cost_foresight)
                               /  cum_undisc_cost_foresight
```

Expected sign: positive. A positive value means myopic-P3 is more expensive cumulatively than foresight-P3 — the foresight-planner wins on total cost by spending more early to avoid paying more late.

**Interpretation notes.**
- Magnitude is expected to be small on ERCOT at `(2040, 90)` — the endpoint is close and the ratchet does most of the enforcement work. Larger on `(2050, 99)` and `(2050, 99.9)` where foresight has more years to front-load cheap avoidance.
- Negative values mean the penalty parameterization is wrong (λ too high, or `endpoint_target_shares` poorly chosen). Sensitivity sweep (§7 phase D) calibrates λ to keep this positive across all 28 cells.
- Undiscounted is the headline; discounted is reported in tables as a sensitivity.

Secondary headlines (once P1 forcing is live and out-of-scope exits):

```
p1_penalty_vs_foresight  =  (cum_undisc_cost_p1  −  cum_undisc_cost_foresight_p3)
                            /  cum_undisc_cost_foresight_p3
```

```
p1_reliability_tax_premium  =  reliability_tax_p1  −  reliability_tax_foresight_p3
```

### 6.4 Schema changes

Backward-compatible additive on the v2 schema frozen 2026-04-20 (per `reliability_tax/PATHWAY_OUTPUT_SCHEMA.md`). No breaking changes.

**New fields in `config` block:**

```json
{
  "config": {
    "solver_mode": "foresight",              // NEW. One of {"myopic", "foresight"}. Default "myopic" if absent (back-compat).
    "foresight_lambda": 0.15,                // NEW. Null if solver_mode = "myopic".
    "endpoint_year": 2045,                   // NEW. Per-endpoint; replaces implicit 2050.
    "endpoint_pct": 95.0,                    // NEW. Per-endpoint.
    "endpoint_target_shares": { ... },       // NEW. The reference mix used as penalty target. Null if myopic.
    ...existing fields unchanged...
  }
}
```

**New fields in `headline` block:**

```json
{
  "headline": {
    "foresight_penalty_vs_myopic": 0.042,    // NEW. Null if no paired myopic run.
    "time_to_90pct": 2041,                   // NEW. Integer year; null if never hit.
    "cascade_activations": 2,                // NEW. Counter; also in diagnostics.
    ...existing fields unchanged...
  }
}
```

**Diagnostic sidecar (new, optional):** `pathway_{iso}_ep{pct}_{mode}_foresight_diagnostics.json` with per-year `share_distance_to_endpoint`, `penalty_component`, `sunk_cost_component`, `cascade_tier_activated`, `selected_mix_candidate_rank_among_feasible`. Emitted only when `solver_mode == "foresight"`.

### 6.5 Endpoint semantics and post-endpoint behavior

Under the 4-endpoint collapse, `endpoint_year` ∈ {2040, 2045, 2050}. The solver simulates `[start_year, endpoint_year]` inclusive and stops. Post-endpoint behavior (e.g. 2041–2050 for an ep90 run) is **not** simulated — the ledger, cost accumulation, and cascade counters all terminate at `endpoint_year`.

**Rationale.** The research question is "what does planning to endpoint `(ey, pct)` cost?" not "what happens after you arrive." Post-endpoint simulation adds cost to all runs equally without resolving the planning question.

**P1 comparison rule.** P1 runs its own `endpoint_year`, matching the P3 pair. Cumulative-cost comparisons at `(endpoint_year, endpoint_pct)` are apples-to-apples because both solvers cover the same years.

---

*§§7-9 pending in subsequent commits.*
