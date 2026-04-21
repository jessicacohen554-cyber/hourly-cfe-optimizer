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

*§§4-9 pending in subsequent commits. See next-session handoff prompt.*
