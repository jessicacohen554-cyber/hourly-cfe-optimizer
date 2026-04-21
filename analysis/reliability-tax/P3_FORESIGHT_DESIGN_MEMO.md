# P3 Foresight — Design Memo

> Supersedes `reliability-tax_prompt-pack.md` as the methodology artifact for adding an endpoint-aware solver mode to Pathway 3. The prompt pack (MS Copilot output) commits to an algorithm choice and a set of "surgical minimal-edit" constraints before the research question is settled. This memo steps back and reasons from the current code state.
>
> **Scope.** Design only. No code changes in the same commit. Phase B onward will land in follow-up branches.
>
> **Out of scope.** The P1 clean-firm forcing mechanism (being addressed in a separate session).

---

## 1. Critique of the prompt pack

The prompt pack is 275 lines across six subprompts (0 orientation, 1 design plan, 2 implementation, 3 testing, 4 performance, 5 changelog). It names the deliverable "Pathway Optimizer v2 — Terminal-Candidate Refactor (P1 vs P3)" and mandates: (a) P1 unchanged, (b) P3 terminal-anchored via top-K membership, (c) additive schema, (d) minimal diffs. Three categories of problems:

### 1.1 Stale or invented references

| Prompt-pack claim | Reality on this branch |
|---|---|
| "Reference optimizer: `step2_2a_cost_optimization.py`" (global context) | File **exists** at `scripts/step2_2a_cost_optimization.py` (4,444 lines), but it is a **per-(iso, threshold) 2025-snapshot cost optimizer** that cross-evaluates cached EF mixes against 5,832 cost scenarios. It is not a 26-year trajectory solver and has no "evaluate all mixes → select cheapest" single-year logic that transfers to terminal-candidate selection. The reference is misleading: step 2.2a answers "what's the cheapest mix today under this cost scenario," not "what endpoint mix should we anchor against in 2050." The prompt pack's Prompt 0 item 7 asks Claude to extract single-year logic from 2.2a for reuse — that logic exists but doesn't solve the problem the memo is trying to solve. |
| "Output schema (frozen): PATHWAY_OUTPUT_SCHEMA.md" | File **exists** at `reliability_tax/PATHWAY_OUTPUT_SCHEMA.md`, 38 lines, **v2 frozen 2026-04-20**. The schema has already evolved past what the prompt pack assumes: it now carries per-tranche clean-firm vintages (`clean_firm_existing/_uprate/_geo/_nuke_new/_ccs`), `feasibility.physical`, `stranding_metadata.methodology="peak_year_snapshot_v2"`, absolute-TWh-claim ledger semantics. Any "additive" change must be reckoned against v2, not v1. |
| "Sample outputs for regression: `pathway3_ep99.json.md`" | No `.json.md` files exist anywhere in the tree. The cached samples are plain JSONs at `analysis/reliability-tax/data/{ISO}/pathway{p}_ep{tag}.json`. The `.md` suffix appears to be invented. |
| "Per-year argmin occurs in `solve_pathway()`" (Prompt 0 item 1) | Correct in broad strokes, but the current argmin is a **sunk-cost-aware Interpretation B scorer** (scripts/step_2_3_pathway_optimizer.py:1122–1131): per-year score is `Σ_r max(0, target_twh_r[m] − floor_twh_r) × LCOE_r_y` over 10 non-CF resources plus 4 clean-firm tranches, plus gas capex/FOM/fuel and storage cost. **Prior-built capacity is sunk** — it prices to $0 in the scorer. Any foresight algorithm has to reconcile with this: a lookahead cost function computed at year-T cannot re-price already-built vintages at year-T LCOE, it must use the `locked_lcoe` from each vintage's COD year. |
| "Membership enforcement: endpoint winner ∈ candidate set" (Prompt 1 item 3) | Assumes every candidate mix is physically achievable at the endpoint under the current ratchet + pmask + cfe + cf_feasibility cascade. In practice, a top-K-cheapest-2050-mix pick may be inadmissible from 2025 onward because the mix implies a CF share drop the ratchet blocks, or an instant jump to 99% CFE that the SBTi ladder (`_cfe_target_for_year`, scripts/step_2_3_pathway_optimizer.py:759) forbids. The prompt pack's "force the endpoint winner" instruction has no handshake with the 4-tier fallback cascade (scripts/step_2_3_pathway_optimizer.py:1133–1151). |
| "P1 MUST remain unchanged" (non-negotiable 1) | P1's forcing mechanism is being addressed in a separate session — out of scope for this memo, but worth noting that freezing P1 as-is pre-commits to current P1 behavior regardless of whether it correctly represents the myopic counterfactual. |
| "Stage-1 sidecar behavior" (Prompt 0 item 6, Prompt 4 constraint 1) | Stage-1 in the current code is the `peakclean.parquet` sidecar (scripts/step_2_3_pathway_optimizer.py:460–595) that precomputes per-EF-row 99.97-percentile residual-gap fractions once per (iso, threshold). It is orthogonal to foresight — a foresight layer runs on top of Stage-2. The prompt pack treats Stage-1 as if it were the year loop. |

### 1.2 Under-specified design decisions presented as instructions

The prompt pack's Prompt 2 (implementation) embeds four load-bearing decisions without motivation:

1. **"Compute cost for all eligible mixes" at endpoint year — under what cost model?** Three incompatible options:
   - (a) Replacement cost at year-2050 LCOE (ignores sunk cost, treats existing vintages as re-priceable — violates the absolute-TWh vintage ledger).
   - (b) Cumulative path cost assuming a specific filler trajectory 2025→2050 (requires picking a path, which is the thing the algorithm is supposed to find).
   - (c) Terminal-year operating cost only (ignores the path, which is where the reliability tax lives).

   The B scorer requires a realized build path to even compute a per-year cost number — you can't score a 2050 mix in isolation.

2. **"Select top-K cheapest mixes"** — by what metric? Terminal-year operating cost ($/MWh for that year's delivered energy)? Lifetime-average LCOE (weighted by twh_per_year across vintages)? NPV of a straight-line glide path from 2025 baseline to that endpoint? Each gives a different K. The prompt pack picks one silently.

3. **"Run a forward solve with endpoint winner forced"** — what does "forced" mean?
   - Hard ratchet floor at endpoint-share values applied from 2025? Physically absurd — ERCOT 2050 mix has ~50% solar; you can't have 50% solar in 2025.
   - Soft penalty toward endpoint shares? Different algorithm, different tuning.
   - Terminal-year-only hard constraint with interior years free? What the prompt pack probably means but doesn't say. And this is exactly where the ratchet + pmask + cfe cascade bites — if the terminal mix is incompatible with the ratchet extrapolated backwards, the cascade falls through to Tier 4 (ratchet violated) and the "foresight" result becomes indistinguishable from myopic.

4. **"Cumulative cost to endpoint year"** — discounted or undiscounted? The existing SPEC invariant (§24 item 6, SPEC.md:543 and README.md:22) is `undiscounted cumulative` as primary with `NPV@5%/7%/9%` as secondary. The existing JSON already emits all four (`headline.undiscounted_cost_usd`, `headline.npv_at_5pct/7pct/9pct`). The prompt pack introduces `p3_relative_cost vs P1` without specifying which of the four it sits in.

### 1.3 Pre-committing to the algorithm

Non-negotiable 2 ("P3 MUST be terminal-anchored using a membership set") locks in one specific foresight algorithm (terminal-anchored rollouts with top-K) before the research question is stated. §4 below enumerates five foresight algorithms that all satisfy "planner sees the ≥90% endpoint and builds clean firm early when it avoids gas buildout or stranding." Terminal-candidate membership is one. It is not obviously the best — see §4.5.

The prompt pack's framing as "minimal surgical edits" compounds this: it presents the algorithm choice as a small change, which it isn't. Foresight is a net-new solver concept. Framing it as a diff against the current year-loop prematurely forecloses the architecture question (§3) and lets the algorithm question (§4) ride for free under the banner of "deterministic, minimal-change."

### 1.4 Misaligned test requirements

Prompt 3's tests would pass a lot of broken implementations. "Determinism tests: terminal candidate selection stable across runs" doesn't catch the stable-sort tie-breaking question (§7 of this memo). "Membership tests: P3 endpoint winner ∈ candidate set" doesn't catch the fallback cascade interaction. "Regression tests: Pathway 1 outputs unchanged except additive fields" is correct but doesn't demonstrate the headline — that foresight-P3 actually produces a different 2050 mix and interior trajectory than myopic-P3. The test plan is structured around whether the refactor compiles, not whether it answers the research question.
