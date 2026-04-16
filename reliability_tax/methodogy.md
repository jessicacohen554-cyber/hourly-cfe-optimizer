# METHODOLOGY — The Reliability Tax

Last updated: [today's date]
Status: Pre-implementation. All 17 decision cards approved.

## Project overview

Analysis of cost and buildout consequences across four decarbonization pathways that target five different hourly CFE endpoints by 2050. The central argument: when planning optimizes for "clean + affordable" without explicitly designing for reliability, the reliability axis is paid for implicitly through a redundant fossil system — the reliability tax.

## Shared invariants (locked)

| Parameter | Value |
|---|---|
| Endpoint targets | CFE ≥ 90%, 95%, 97.5%, 99%, 99.9% by 2050 |
| Planning horizon | 2025–2050 (25 years) |
| Cost basis | Real 2025 dollars, no inflation adjustment |
| Cost reporting | Undiscounted cumulative 2025–2050 + NPV at 5%, 7%, 9% real |
| Objective function | Minimize NPV@7% real subject to endpoint constraint |
| ISO scope | All 7 ISOs, fully parametric |
| Run count | 4 pathways × 5 endpoints × 7 ISOs = 140 runs + Section 2 demand sensitivity |
| Smoke test | ERCOT first |
| Framework | Deterministic cost minimization under constraint. NOT a market prediction. |
| Grid representation | Bubble-node model (no detailed transmission) |

## Pathways

| ID | Name | Clean firm availability |
|---|---|---|
| 1 | VRE + batteries only | Locked at 0 all years |
| 2a | Behavioral reactive pivot | Locked at 0 until Pathway 1 hits 90% CFE plateau for that ISO; unlocks at pivot year with FOAK Wright's Law starting point |
| 2b | Economic reactive pivot | Locked at 0 until marginal $/CFE% of next VRE+battery addition exceeds the cheapest available clean firm tech's LCOE for that year and ISO; unlocks at pivot year with FOAK Wright's Law starting point |
| 3 | Clean firm proactive | Available year 1 on the standard Wright's Law curve |

## Resource treatment

- **Clean firm bucket:** nuclear, CCGT+CCS, geothermal. Optimizer picks among these subject to existing regional constraints in the pipeline (nuclear siting, geothermal resource availability, CCGT+CCS feasibility).
- **Offshore wind:** NOT in clean firm bucket. Classified as VRE, available to all four pathways subject to regional constraints.
- **Onshore VRE:** solar and wind, available to all pathways.
- **Storage:** batteries, available to all pathways.
- **Existing fossil:** inherited from EIA 860, retires endogenously (see Card L).

## Approved decision cards

### Card A — FOAK→NOAK cost transition shape
Wright's Law based. Cost drops by the technology's learning rate with each doubling of cumulative global capacity. FOAK premium is the starting point at t=0 cumulative GW; the curve converges asymptotically to NOAK as cumulative deployment scales.

### Card B — FOAK premium magnitudes
Read existing values from Step 2.2 / pipeline config first. Use those values verbatim. If repo does not contain values for nuclear, CCGT+CCS, or geothermal, flag in implementation and escalate to user before picking defaults.

### Card C — Pathway 2b economic pivot comparator
Tech-specific LCOE at the margin. Pivot triggers when the best next VRE+battery addition's $/CFE% exceeds the cheapest available clean firm technology's marginal LCOE for that year and ISO. Regional constraints apply to which clean firm options are in the comparison set.

### Card D — Optimizer internal discount rate
Match objective function. Internal build-order ranking uses NPV@7% real. One discount rate across the entire pipeline for consistency.

### Card E — Pathway 2 pivot re-optimization
Freeze-and-forward. Pre-pivot decisions are locked. Only the post-pivot trajectory is re-optimized once clean firm unlocks. No retroactive cancellation of already-built assets. Early retirement of pre-pivot builds is not allowed in the base case.

### Card F — Stranding value calculation
Report both. Headline metric is book value of unrecovered capital at 2050 (used in Section 5 Sankey width and closing summary). Secondary metric is NPV of unrecovered capital through asset end-of-life, tracked in the stranding ledger for any reader who asks about rigor.

### Card G — Smoke test ISO
ERCOT. Highest VRE penetration of any US ISO, most dramatic overbuild and wall dynamics, existing site already has ERCOT-specific data infrastructure from prior work.

### Card H — Build-order tiebreaking
Cheaper LCOE wins. Consistent with the cost-minimization objective. Deterministic because LCOE is fully specified by the cost model.

### Card I — Transmission cost allocation in stranding ledger
EXCLUDED. Bubble-node model does not support detailed transmission attribution. Stranding ledger covers only new-build gas, VRE, and storage per the comparative definition in Card K revised.

### Card J — Infeasibility detection
Hybrid. Primary: physical floor — optimizer declares infeasible when no physically feasible addition can improve CFE score (regional constraints exhausted, no additional siteable capacity). Secondary diagnostic: economic floor — flag when marginal $/CFE% exceeds $10,000 per percentage point, even if physically feasible. Both reported in the manifest so readers can see physical vs. economic ceilings separately.

### Card K (revised) — Stranding definition
**Comparative-to-Pathway-3 stranding with VRE curtailment cross-check.** An asset is stranded if it represents capacity built by a given pathway beyond what Pathway 3 builds to reach the same CFE target in the same ISO. The stranding ledger is computed as:

`stranded_capacity[pathway, resource] = max(0, capacity[pathway, resource] - capacity[Pathway 3, resource])`

evaluated at 2050, per resource type, per ISO, multiplied by asset-level book value of unrecovered capital.

This comparative definition applies to: new-build gas, VRE (solar, onshore wind, offshore wind), and storage. It does NOT apply to clean firm — Pathway 3's clean firm IS the destination plan, so any clean firm built in other pathways (post-pivot in 2a/2b) is by definition not stranded.

**Cross-check:** for VRE only, also report per-resource-type curtailment rates at endpoint (2050) as an absolute diagnostic. If a VRE asset shows >50% curtailment AND is not flagged as comparatively stranded, that's an interesting signal worth investigating — it could indicate Pathway 3 is also overbuilding in ways the model didn't catch, or a deeper system dynamic worth surfacing.

**Framing language for the page:** "'Stranded' here means capital spent on capacity that a destination-optimized plan wouldn't have required to reach the same CFE target." This must appear somewhere in Section 5 so readers understand the counterfactual.

**Implication for clean firm in Pathway 2:** Clean firm built post-pivot in Pathway 2a/2b is NOT counted as stranded even if the post-pivot build exceeds Pathway 3's clean firm build. Pathway 2's stranding argument is about pre-pivot gas and VRE overbuild, not about the clean firm that eventually gets built.

### Card L (revised) — Existing gas fleet retirement
Endogenous economic retirement. Existing gas retires when it stops clearing capacity markets under each pathway's dispatch and capacity dynamics. Every pathway's output tracks existing-fleet retirement timelines explicitly in MANIFEST.json, so Section 4 and closing synthesis can show how much cost delta comes from new-build decisions vs. retirement dynamics.

### Card M — Capacity market revenue treatment
Netted against fixed costs. Reliability tax is net of capacity payment revenue — the true cost to customers after accounting for market payments. Applies consistently across all pathways.

### Card N — Learning curve cost year
Year of commercial operation. Wright's Law curves are indexed to cumulative installed capacity. Cost is locked at COD. Financing decisions use commitment-year cost as the basis for capital raise sizing but the locked cost on the books is COD-year cost.

### Card O — Endpoint CFE definition
Hourly CFE, same-ISO matching, consistent with the 8,760 Problem methodology used across the rest of the site.

### Card P — Pathway 2 FOAK premium scope
FOAK resets. Post-pivot clean firm starts at FOAK and rides the Wright's Law curve the same way Pathway 3 does — just starting later. The cost penalty from Pathway 2 comes from timing (later years, less time for learning to occur, stranded pre-pivot VRE and gas), not from a double FOAK penalty.

### Card Q — Section 6 "Cost of Waiting" commitment years
2026, 2030, 2035, 2040, 2045. Five points of resolution on the penalty curve. 2026 anchors "even committing now costs you something" vs. an implicit 2025 baseline.

### Card R — Pathway 1 resource scope / offshore wind classification
**Approved amendment (April 2026).** Original methodology classified offshore wind as VRE available to all pathways. Post-run audit found NYISO, NEISO, and PJM Pathway 1 runs included offshore wind and nuclear_newbuild — violating the "VRE + batteries only, no clean firm" definition.

**Resolution:**
- **Pathway 1 is split into P1a and P1b.**
  - **P1a (onshore VRE + storage):** solar, onshore wind, battery4, battery8, LDES, H2, existing hydro (wholesale-priced), existing nuclear uprates only. No offshore wind. No nuclear_newbuild. This is the true "no planning commitment" baseline.
  - **P1b (VRE + offshore wind + storage):** P1a resources plus offshore wind where regionally available (NYISO, NEISO, PJM, CAISO). Still no nuclear_newbuild. Offshore wind requires long permitting and supply-chain commitment but not the same dispatchability as clean firm — it is VRE-category but requires a distinct policy commitment.
- **Nuclear uprates (`uprate` resource):** Treated as existing-fleet operations (like hydro), available to all pathways including P1a. Not counted as new clean firm.
- **Nuclear_newbuild:** Locked to P2a/P2b/P3 only (post-pivot or proactive). Never allowed in P1a or P1b.
- **Offshore wind:** Locked to P1b and above (P2a, P2b, P3). Not in P1a.

**Re-run scope:** All 7 ISOs × 5 endpoints re-run for both P1a and P1b to enforce these constraints cleanly. All downstream payloads updated to use P1a as the primary "VRE stress" baseline.

**Downstream implications:**
- All references to "Pathway 1" in Section 1 (worst hours), Section 2 (overbuild), Section 3 (reliability tax) now use P1a.
- Section 2 and Section 3 can show P1a vs P1b comparison to illustrate the offshore wind effect for NYISO/NEISO/PJM.
- Closing summary table adds P1a and P1b as separate pathway rows (140 rows becomes 175 rows: 7 ISO × 5 pathway × 5 endpoint).

### Card S — Primary cost metric and new-build gas stranding
**Approved amendment (April 2026).** The optimizer objective function minimizes NPV@7%, but NPV is not the right lens for customer rate impacts — customers pay actual annual costs, not discounted present values.

**Resolution:**
- **Primary cost metric: undiscounted cumulative cost (real 2025$).** This is `sum(net_annual_cost_usd, 2025–2050)` with no time discounting. It directly translates to what customers pay in rates over the planning period.
- **Secondary cost metric: NPV@7%.** Retained for investment comparison purposes (captures time value of capital deployment), but not the headline figure.
- **New-build gas stranding included in absolute actual cost.** When a pathway builds new CCS-CCGT capacity and that capacity is retired before the end of its 25-year useful life (i.e., stranded at the 2050 endpoint), the remaining undepreciated book value is added to absolute actual cost. This captures the real economic cost borne by ratepayers who funded the capital.
  - Capital cost basis: $2,200/kW overnight (NREL ATB 2024 moderate, CCGT+CCS)
  - Useful life: 25 years
  - Stranded fraction: `max(0, (build_year + 25 − 2050) / 25)` — plants built in 2025 have 0 stranding (full 25-year life by 2050); plants built in 2032 have 28% stranding (7 of 25 years unused)
  - `new_gas_stranded_billion_usd = Σ (stranded_gw_by_vintage × $2,200/kW × 1e6 × stranded_fraction)`
  - `total_actual_cost_trillion = undiscounted_cost_trillion + new_gas_stranded_billion / 1000`
- **Existing-gas fleet costs are already in `undiscounted_cost_usd`** (via gas generation × fuel+O&M). The stranding correction applies only to newly-built CCS-CCGT because those plants have committed capital that won't be fully amortized within the planning horizon.

## Downstream implications for page sections

- **Section 4 (Four Journeys):** Annotation set must include existing-fleet retirement timelines as an overlay or secondary axis per Card L revised.
- **Section 5 (Stranding Ledger Sankey):** Fate buckets are "delivered value (matches Pathway 3 build)" vs. "excess over Pathway 3 (stranded)." Transmission excluded per Card I. Clean firm excluded from stranding per Card K revised. VRE curtailment cross-check reported as secondary callout, not as a primary bucket.
- **Section 5 copy:** Must include the stranding definition disclaimer: "'Stranded' here means capital spent on capacity that a destination-optimized plan wouldn't have required to reach the same CFE target."
- **Section 6 (Cost of Waiting):** Commitment year axis = 2026, 2030, 2035, 2040, 2045 per Card Q.
- **Closing synthesis table:** Add columns for physical feasibility ceiling and economic feasibility ceiling per Card J. Add column for existing-fleet retirement cumulative GW by 2050 per Card L revised.

## MANIFEST.json required fields

Per-run entry must include:
- ISO, pathway, endpoint
- Feasibility status (physical, economic)
- Achieved CFE
- Undiscounted cumulative cost
- NPV@5%, NPV@7%, NPV@9%
- Existing-fleet retirement timeline (year-by-year GW retired by resource)
- Comparative-stranding totals (GW and $ beyond Pathway 3 build)
- VRE curtailment rates per resource type at 2050

## Implementation escape hatch

If during any Prompt 2A–5B session Claude Code encounters a choice that should have been a decision card but wasn't, it stops and surfaces the question in chat rather than picking silently. Amendment cards get added to this document as "Card R, Card S, ..." with the same structure as the originals.

## Claude Code bug workaround (temporary)

As of April 13, 2026, Claude Code is experiencing a known bug where markdown file generation times out mid-stream. Until resolved:
- All narrative deliverables (methodology updates, sanity memos, numbers summaries) are reported in chat across multiple sequential messages
- Code files and JSON data files are written normally (bug does not affect these)
- User maintains METHODOLOGY.md locally and pastes current state into each new prompt
- No prompts use plan mode; all use code mode with explicit "do not write .md files" instruction

## Running log of implementation notes

[To be populated during Prompts 2A onward as implementation surfaces clarifications to approved decisions.]
