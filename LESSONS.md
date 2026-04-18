# LESSONS.md — Accumulated Learnings

> Fix-this-next-time learnings from prior sessions. Append one line per session at end.
> Read at session start. Append at session end. Don't rotate without user approval.

## Reliability Tax Redesign — Apr 18, 2026

1. **Section names must describe content, not metaphors.** "The Setup / The Hump / The Abandonment / The Tax / The Cost of Waiting" read as pretentious filler to the user. Descriptive titles using the numbered `section-header` + `section-number` pattern from `clean_firm_case.html` / `lmp_trends.html` — e.g. "Why a clean-only grid still builds gas", "How much new gas gets built", "How much of it gets stranded" — are what ships.
2. **Never cite `SPEC §` or `Card [A-Z]'` in user-facing copy.** Public readers see `SPEC §24.5`, `Card F'`, `§24.8 NOAK-2035` as internal shorthand and disengage. Methodology citations belong in `optimizer_methodology.html` and `SPEC.md`, not in hero text, chart captions, insight boxes, or footer notes.
3. **Pathway comparisons must show all pathways on one plot — toggles aren't comparisons.** The original §2 hump chart and §6 stranding chart used a pathway toggle that made users click through one-at-a-time views. Real comparison means overlay: all three pathways as colored lines in §2, all five pathways as sorted horizontal bars in §6. A toggle that hides the other options isn't a comparison, it's a slideshow.
4. **Industry acronyms get defined on first use per page, not banned.** ELCC, NOAK, FOAK, LCOE, CCS, LDES, 45Q, 45U, ITC, PTC, ATB, LOLE, CFE, VRE, BESS, CCGT, IPP, ISO, AEO, NREL, LBNL, EIA, NERC are legitimate terms for this audience — the fix is a parenthetical expansion on first occurrence, then the bare acronym thereafter. Stripping them entirely dumbs the copy down; leaving them undefined locks out readers.
5. **Prose-fixer agents (`jargon-fixer`, `voice-fixer`) replaced the originally proposed pre-commit jargon hook.** A regex-based git hook can't tell whether `ELCC` was already defined earlier on the page, or whether `LOLE` in a chart tooltip is a legit second-use or an undefined first-use. Both require reasoning about page-scoped context. Sub-agents invoked via `/fix-jargon`, `/fix-voice`, and `/fix-prose` handle that context-aware rewrite; a hook would false-positive on every clean page and miss the genuine problems.

## reliability-tax-4.7.html Bake-off Session 1 — Apr 18, 2026

6. **Banned-string acceptance greps must skip `<style>` and `<script>` blocks.** A naive `grep -oP '>[^<]+<'` treats CSS and JS as "user-facing text" because those blocks live between `>` and `<` too. Session 1 failed its first acceptance check on a `§` inside a CSS comment (`/* used by Session 2 §2 */`) — harmless but flagged. Strip `<style>…</style>` and `<script>…</script>` before greping, or scope the grep to text nodes only. Either works; the naive version gives false positives that waste a round-trip.
