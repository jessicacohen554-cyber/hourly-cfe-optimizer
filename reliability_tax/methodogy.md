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
Tech-specific LCOE at the margin. P​​​​​​​​​​​​​​​​
