# Claude Code — Session Instructions

> Draft replacement for `CLAUDE.md` — review this file, then move it into place.
> Source of truth for project decisions: **SPEC.md**.

## On every turn — non-negotiables

1. **Use full thinking depth** on every non-trivial edit. Don't downshift because a task "seems mechanical." The harness sets `MAX_THINKING_TOKENS=64000` — use it.
2. **Anchor every new page to a reference filename** ("build it like `abatement_dashboard.html`"). If no reference was named in the task brief, ask before starting.
3. **Plan before writing** any data-touching code or any user-facing prose >100 words. Output the plan, wait for OK.
4. **Vectorize before looping.** Never write a Python `for` loop over data arrays >1k rows. If you're about to, stop and write the vectorized kernel signature first, show it, wait for OK.
5. **Run `/fix-prose` before committing** any HTML/MD page you wrote or substantially edited. The `jargon-fixer` and `voice-fixer` agents catch self-referential jargon and AI-tell language you missed.
6. **End every session by writing one line** to `LESSONS.md` describing the most important fix-this-next-time learning.

## Source docs — what to read when

| File | Read when |
|---|---|
| `SPEC.md` | Session start, before picking up work — current status, locked-in spec, open questions |
| `LESSONS.md` | Session start — accumulated learnings from prior sessions |
| `OPS.md` | Before running optimizer / heavy compute / data ops |
| `DESIGN_SYSTEM.md` | Before any HTML/CSS work — color tokens, typography, chart conventions |
| `PIPELINE.md` | Reference for the 8-step pipeline only |
| `SPEC_LOG.md` | Historical decision archive — reference, don't restate |

## Reference-page table — anchor every new page

| Building a... | Use as reference |
|---|---|
| Scrollytell analysis page | `dashboard/abatement_dashboard.html` |
| Single-company report | `dashboard/ipp_vistra.html` |
| Multi-company comparison report | `dashboard/ipp-transition-report.html` |
| Market overview | `dashboard/gen_market_overview.html` |
| Interactive optimizer page | `dashboard/dashboard.html` |
| Methodology / spec page | `dashboard/optimizer_methodology.html` |
| Research-paper-style page | `dashboard/research_paper.html` |

## Voice — what gets shipped to readers

Direct, confident, analytical. Senior analyst briefing stakeholders, not PhD thesis. Active voice, contractions OK. Brevity over verbosity. Lead with the insight, then the evidence. Detail sufficient for peer review — brevity ≠ vagueness.

**Three voice rules, all enforced by the prose-fixer agents:**

- **No self-referential project shorthand in user copy.** That means no `SPEC §X.Y`, no `Card [A-Z]'`, no bare `§24.X`, no internal endpoint codes (`ep90`, `ep95`, `ep99p9`), no bare pathway codes (`P1`, `P1a`, `P2a`, `P2b`, `P3`), no `NOAK-2035`/`NOAK-2040`/`NOAK-2045` as codenames. Run `/fix-jargon`.
- **Industry acronyms must be defined on first use per page.** ELCC, NOAK, FOAK, LCOE, CCS, LDES, 45Q, 45U, ITC, PTC, ATB, LOLE, CFE, VRE, BESS, CCGT, IPP, ISO, AEO, NREL, LBNL, EIA, NERC. The `jargon-fixer` agent expands them on first occurrence and leaves subsequent uses alone.
- **No AI-tell language.** No "It's worth noting," "Importantly," "Moreover," "Furthermore," "leverages," "unlocks," "delves into," "underscores," "robust framework," "holistic approach," "paradigm." Run `/fix-voice`.

## Decision-making

- Multi-way decisions with real tradeoffs → `AskUserQuestion`. Each card: short question, 2–4 options, 1-sentence descriptions.
- Trivial decisions (formatting, naming consistent with existing patterns) → just do them.
- Decisions land in `SPEC.md` immediately after user confirmation, before any implementation.

## Communication style

- Don't narrate — just do. Skip "Let me read the file..." filler.
- TodoWrite on a frequent cadence — the todo list IS the status communication.
- Don't echo decisions back. Acknowledge briefly and act.
- Be concise. Bullets and numbered lists over prose walls.
- Only surface errors when self-recovery failed.

## Git

- Frequent incremental commits during active work.
- Squash into a clean commit before pushing.
- Descriptive paragraph commit messages — explain *what* and *why*.
- Push only when complete and QA'd.
- Never push to a branch other than the one assigned in the task brief.

## Change propagation

"Fix something" = fix it everywhere. Cross-cutting fixes touch every relevant page. Only scope to a single page if the user explicitly says so.

## Proactive session management

When context climbs past ~80% or the user signals wrap-up:
1. Stop starting new work.
2. Commit + push WIP to the current branch.
3. Update `## Current Status` at the top of `SPEC.md`.
4. Write the session lesson to `LESSONS.md`.
5. Output a resume prompt focused on task context.

The `Stop` hook in `.claude/settings.json` enforces this on exit.

## File boundaries

- Never modify raw data in `data/` — preserved from source.
- Never run the prose-fixer agents against `SPEC.md`, `CLAUDE.md`, `PIPELINE.md`, `DESIGN_SYSTEM.md`, `OPS.md`, or `.claude/` — those are internal docs and *supposed* to use project shorthand.

## Heavy compute

See `OPS.md` for the full pre-run gate, checkpoint discipline, and rerun policy. Short version: never start a Step 1 run while decisions are still being discussed; always pre-flight (syntax check, assumption verification, save/load round-trip) before launch.
