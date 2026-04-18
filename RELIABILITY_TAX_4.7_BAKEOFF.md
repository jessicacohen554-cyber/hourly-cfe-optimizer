# Reliability Tax — v4.7 Bake-off Variant Plan

Standalone working doc for the 5-session build of `dashboard/reliability-tax-4.7.html`. This file is the single source of truth for the bake-off; each session prompt references it by path. Not part of SPEC.md — this work is scoped and disposable after the bake-off lands.

---

## Context

A new file `dashboard/reliability-tax-4.7.html` is being produced as one side of a bake-off. It is a full redesign that matches the narrative voice and chart conventions of `dashboard/gen_market_overview.html`, `dashboard/ipp-transition-report.html`, `dashboard/abatement_dashboard.html`, and `dashboard/ipp_vistra.html`.

The existing `dashboard/reliability-tax.html` must stay byte-for-byte untouched (acceptance gate #1). This work is orthogonal to any in-place redesign of the original.

Branch: `claude/redesign-reliability-tax-page-rW5Y2`. No PR.

---

## Shared Rules (every session must enforce)

- File: `dashboard/reliability-tax-4.7.html`. NEVER edit `dashboard/reliability-tax.html`.
- Variant tag: `<title>` is `"The Reliability Tax (v4.7) | The 8,760 Problem"`. Header shows a `v4.7` badge — `font-family: var(--font-data); font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: rgba(255,255,255,0.72)`.
- **Banned in user-facing strings** (HTML text, labels, legends, captions, tooltips): `SPEC`, `§`, `Card`, `NOAK`, `FOAK`, `LOLE`, `ELCC`, `CFE`, `P1`, `P1a`, `P2a`, `P2b`, `P3`, `ep90`, `ep95`, `ep99p9`, and `hump`/`sankey` as user-facing words. JS-internal variable or object keys like `data.per_pathway['1']` are fine — only strings rendered to users are checked.
- **Plain-English replacements** (locked):
  - `"1"` → "Wind, solar, and storage"
  - `"1a"` → "Onshore wind and solar only"
  - `"2a"` → "Reactive pivot at the 90% wall"
  - `"2b"` → "Reactive pivot when the math flips"
  - `"3"` → "Proactive clean firm"
  - `"ep95"` → "95% clean"
  - `"≥99.9%"` → "effectively 100%"
  - `"achieved CFE %"` → "clean energy reached (%)"
- Colors: ONLY `RESOURCE_COLORS` / `ISO_COLORS` / `SEMANTIC_COLORS` from `js/chart-colors.js`, CSS variables from `styles/shared.css`, or `withAlpha()` wrappers of those. ZERO hardcoded hex in any Chart.js dataset.
- Every `<canvas>`: `.chart-title` ABOVE, plain-English caption BELOW, axis titles with units, tick formatters (1200→"1.2k", %, $, GW).
- Chart defaults: `responsive: true`, `maintainAspectRatio: false`, legend `position: 'bottom'`, `usePointStyle: true`, `boxWidth: 10`, `font: { size: 12 }`. Plus Jakarta Sans 12/600 for axis + legend titles; DM Mono 11 for ticks.
- Peak markers / annotation-only datasets filtered OUT of legend.
- **No pathway toggles anywhere.** All pathway-comparison charts render all pathways on one plot (or as small-multiples per ISO).
- Load in `<head>`: `chart.js@4.4.7`, `chartjs-plugin-annotation@3.0.1`, plus existing `nav.js`, `chart-colors.js`, `shared-header.js`, `scroll-observer.js`, `shared-footer.js`. NO GSAP (not needed for this page).
- Preserve data wiring — `fetch()` paths unchanged:
  - `js/reliability-tax/section2_gas_hump.json` — `per_iso[iso][pw] = rows[]`; `peaks_per_pathway[iso][pw]`
  - `js/reliability-tax/section3_reliability_tax.json` — `per_iso[iso].per_pathway[pw][ep].components_usd_per_mwh` + `total_reliability_tax_usd_per_mwh`
  - `js/reliability-tax/section5_stranding_sankey.json` — `pathway_totals[pw]`; `per_iso_pathway.new_gas_stranding[iso][pw].total_stranded_capex_usd`
  - `js/reliability-tax/section6_cost_of_waiting.json` — `per_iso[iso].by_endpoint.ep95.commitment_curve[]`; `pivot_year_2a`, `pivot_year_2b`
  - `js/reliability-tax/closing_summary_table.json`
- Voice: short declarative sentences, lead with insight, active voice, no academic hedging. Match `gen_market_overview.html` and `abatement_dashboard.html`.
- Mobile: 44px min tap targets; no horizontal overflow at 320px.
- Commit branch: `claude/redesign-reliability-tax-page-rW5Y2`. DO NOT commit until Session 5. DO NOT open a PR.

---

## Chart shape choices (locked)

| Section | Title | Chart | Why |
|---|---|---|---|
| Hero | The bill nobody invoiced | 4 stat-tile counters | Matches `gen_market_overview.html` opener. |
| §1 | Why a clean-only grid still builds gas | Two schematic SVG duck-curve panels | Schematic illustration, not data — Chart.js rules N/A. |
| §2 | How much new gas gets built | 7-ISO small-multiples grid, each cell with 3 pathway lines on same axes | Rule: "Trajectory → line chart, every entity on same axes, never a toggle." |
| §3 | How much of that gas runs out of hours | Grouped stacked bar: 7 ISOs × 2 pathways, each bar stacked (running / written off) | Rule: "Two-series comparison → grouped bars, not a toggle." |
| §4 | The bill ratepayers pay | Grouped stacked bar: 7 ISOs × 2 pathways, 5 stacked cost components. Fixed at 95% clean (no threshold toggle). | Rule: "Cost decomposition → stacked bar; legend in plain English; total in tooltip footer." |
| §5 | Who pays the least, ranked | Horizontal bar, 5 pathways sorted descending by $/MWh, averaged across 7 ISOs at 90% clean | Rule: "Pathway ranking → horizontal bar, sorted descending." |
| §6 | Where the stranded capital lands | Horizontal stacked bar: 5 pathway rows stacked by ISO contribution in $B | Rule: "Flow / where-does-capital-go → horizontal stacked bar." Replaces original's fake-Sankey. |
| §7 | The cost of waiting | Line chart, all 7 ISOs on same axes. One vertical annotation at 2035. Uses `chartjs-plugin-annotation`. | Rule: "Trajectory across year → line chart on same axes, never a toggle." |
| §8 | Every run in the database | Filterable + sortable table | Data-discovery tool, not a chart. |

---

## Session boundaries

Each session replaces only its assigned `SESSION-N_SECTION-M` marker div. No overlap. Only Session 5 commits.

| Session | Scope |
|---|---|
| 1 | Head / CSS / hero / §1 duck curves / reading-progress + counter JS / 7 marker divs |
| 2 | §2 small-multiples (7 mini line charts) + §3 grouped stacked bar |
| 3 | §4 cost-decomposition stack + §5 pathway ranking horizontal bar |
| 4 | §6 ISO-stacked stranded capital + §7 cost-of-waiting line (7 ISOs, annotation at 2035) |
| 5 | §8 table + full QA sweep + commit + push |

Commit message (Session 5): `"Add reliability-tax-4.7.html — 4.7 variant for bake-off"`.

---

## Acceptance gates (run in Session 5 before commit)

1. `git diff dashboard/reliability-tax.html` returns empty.
2. `dashboard/reliability-tax-4.7.html` exists with visible `v4.7` badge in header.
3. Banned strings return zero hits in user-facing text (grep for each term between `>` and `<`).
4. Every canvas has a `.chart-title` above and a plain-English caption below.
5. No pathway toggles on any chart.
6. No hardcoded hex in Chart.js dataset configs (`grep -nE "(background|border)Color:\s*['\"]#"` returns nothing).
7. Every axis has a descriptive title with units.
8. Browser QA at 320px, 768px, desktop — no broken layouts, no mobile overflow.

---

## Data-shape notes (already validated)

- `peaks_per_pathway[iso][pw]` exists for all 5 pathways (`1`, `1a`, `2a`, `2b`, `3`) across all 7 ISOs. Keys: `cum_new_gas_mw`, `cum_new_gas_stranded_mw`, `achieved_cfe_pct`, `endpoint_label`.
- `per_iso_pathway.new_gas_stranding[iso][pw].total_stranded_capex_usd` exists (enables §6 ISO-stacked bar).
- `per_iso[iso].by_endpoint.ep95.commitment_curve` is an array of `{commit_year, rtax_usd_per_mwh}`. Years: 2026, 2030, 2035, 2040, 2045.
- `components_usd_per_mwh` has keys: `new_gas_capex_annualized_usd`, `new_gas_fom_usd`, `existing_gas_fom_carried_usd`, `priced_vre_curtailment_usd`, `vre_storage_overbuild_capex_usd`.

---

## Resume prompt (if dropped between sessions)

> Execute Session N of 5 for the `reliability-tax-4.7.html` bake-off variant. Read `RELIABILITY_TAX_4.7_BAKEOFF.md` in the repo root first — it contains the shared rules, chart-shape choices, session boundaries, and acceptance gates. Do NOT commit until Session 5. Do NOT edit `dashboard/reliability-tax.html`.
