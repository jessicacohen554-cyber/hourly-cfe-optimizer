# Prompt 7: New Slide Candidate Evaluation

**Date:** 2026-03-27
**Current deck:** 17 slides + 3 gap sub-slides (4B/4C/4D) = 20 sections total
**Constraint:** Add no more than 1-2 slides to avoid bloat

---

## Candidate A: GHG Protocol Scope 2 Impact Slide

**Title:** The Regulatory Catalyst: GHG Protocol Scope 2 Overhaul
**Thesis:** Upcoming hourly matching + deliverability rules will disqualify 30-50% of current REC portfolios, creating a structural demand shock for localized VRE.

### Evaluation

**Already covered in the deck:**
- Slide 7 (Colocation Developers): "GHG Protocol exposure" callout with the 30-50% disqualification stat, 3-7x price multiplier, and 40-65% actual CFE scores. This is a full paragraph, not a passing mention.
- Slide 9 (24/7 CFE Premium): Entire slide built around annual-vs-hourly matching shift, with $35-50/MWh premium and 1.35x energy penalty KPIs.
- Slide 14 (Three Procurement Strategies): Opening paragraph explicitly cites GHG Protocol Scope 2 consultation moving toward hourly matching + deliverability (Citation #35).
- Slide 15 (Cross-Regional Procurement Flow): Visual comparison of consequential vs. hourly matching accounting.

**Gap analysis:** The GHG Protocol argument is already **woven throughout 4 different slides**. The deck treats it as a pervasive market driver rather than a standalone topic — which is the right approach. A dedicated slide would pull together facts already distributed across the deck, creating redundancy without adding new decision-relevant information.

**Would the audience decide differently without this slide?** No. The regulatory shift is already the explicit backbone of slides 9, 14, and 15. The audience understands the "before/after" transition from the existing content.

**Recommendation: SKIP**

The GHG Protocol narrative is stronger distributed across context-specific slides than concentrated on one. A standalone slide would be a summary of what the audience already absorbed. If anything, consider adding one more data point to the Slide 7 callout (e.g., specific timeline: "consultation open 2025, finalization expected 2027") — but that's a Prompt 5 (existing slide update) task, not a new slide.

---

## Candidate B: VRE Portfolio Value Within Firm Clean Context

**Title:** The Portfolio Effect: What VRE Adds to a Clean Firm Foundation
**Thesis:** Within an IPP portfolio that already has firm clean + storage, standalone VRE's incremental value is disproportionately high — it unlocks the daytime CFE hours that firm clean can't serve cost-effectively.

### Evaluation

**Already covered in the deck:**
- Slide 10 (Hybrid Economics): Shows standalone VRE ($35/MWh) vs. VRE+Storage ($94/MWh) — a 169% revenue uplift. Covers the storage pairing mechanics thoroughly.
- Slide 13 (The Investment Case): Summary metrics including the 639 TWh gap and +169% uplift.
- The research paper (vreresearch.md lines 210-215): Describes the "REC engine" concept — VRE generates daytime RECs, storage shifts energy, firm clean covers nights. This is the core strategic synthesis.

**What's MISSING from the deck:**
The deck shows VRE+Storage economics (Slide 10) and the investment case summary (Slide 13), but it **never shows the 24-hour portfolio composition** — i.e., here's what a 90% CFE day looks like with firm clean alone vs. firm clean + VRE + storage. The research paper describes this three-asset synthesis clearly, but the deck doesn't visualize it.

This is a genuine narrative gap. The audience can see that VRE+Storage is better than standalone VRE (Slide 10), and they know the investment case (Slide 13), but they can't see **how the three asset classes work together across a 24-hour period** to create the 24/7 CFE product. That's the "aha" moment for an IPP audience evaluating whether to add VRE to their existing portfolio.

**Would the audience decide differently without this slide?** Possibly. An executive with a firm clean fleet might still wonder: "I already generate clean power 24/7 with nuclear — why do I need VRE at all?" This slide answers that directly by showing the cost optimization of using cheap daytime VRE to free up expensive firm clean for nighttime premium hours.

**Recommendation: ADD**

### Slide Spec

**Slide Number:** 10B (insert between current Slide 10 "Hybrid Economics" and Slide 11 "Clean Firm Won't Arrive in Time")

**Title:** The Portfolio Effect: VRE Completes the 24/7 Product

**Subtitle:** A firm clean fleet covers nights. Storage covers transitions. VRE covers the bulk daytime hours at lowest cost — together, they synthesize a premium 24/7 CFE product no single technology can deliver alone.

**Layout:** Two-column

**Left Column — 24-Hour Dispatch Stack (Primary Visual)**
- Stacked area chart showing a representative 24-hour dispatch profile for a data center load at ~90% CFE
- X-axis: Hour of day (0-23)
- Y-axis: MW
- Flat horizontal line: Data center demand (~constant)
- Stacked areas (bottom to top):
  - Solar (gold, #F59E0B): Bell curve peaking midday
  - Wind (green, #22C55E): Variable, stronger overnight/morning
  - Battery discharge (cyan, #06B6D4): Evening ramp 17:00-23:00
  - Clean Firm / Nuclear (indigo, #6366F1): Flat baseload, covering remaining gap
  - Residual grid (gray, #D1D5DB): Small gap = the non-CFE hours
- Annotation arrows:
  - "VRE handles bulk daytime volume at $30-40/MWh" pointing to solar+wind midday
  - "Storage bridges the sunset gap" pointing to battery discharge
  - "Firm clean covers premium night hours" pointing to nuclear overnight
  - "Only 10% residual at 90% CFE" pointing to small gray gap

**Right Column — Value Decomposition**
- **KPI card 1:** "Without VRE" — Firm clean alone achieves X% CFE at $Y/MWh (nuclear running flat = expensive way to cover daytime when solar is available)
- **KPI card 2:** "With VRE" — Portfolio achieves same CFE at $Z/MWh (VRE displaces daytime firm clean, which shifts to premium night hours)
- **Table:** "Cost per incremental CFE point" showing marginal cost of each resource's contribution
  - Solar: cheapest per-point (daytime bulk)
  - Wind: moderate (overnight contribution)
  - Battery: high but essential (sunset bridge)
  - Firm clean: highest but irreplaceable (night/multiday)
- **Callout box:** "The REC Engine" — Under hourly matching, VRE generates localized daytime RECs. Storage enables energy arbitrage. Firm clean covers the premium nighttime attributes. The IPP packages all three into a single 24/7 CFE PPA commanding $35-50/MWh above market.

**Data Sources:**
- Dispatch profile: Main repo Step 3A dispatch cache (representative PJM mix at 90% threshold, Medium costs)
- Cost decomposition: Main repo Step 2.2 cost optimization outputs (marginal cost by resource)
- Already computed — no new scraping needed

**Implementation Notes:**
- The 24-hour dispatch chart is the hero visual — it's the single most intuitive way to show portfolio complementarity
- Use the same color scheme as the main repo dashboard (RESOURCE_COLORS from chart-colors.js)
- Chart.js stacked area, or SVG if more control needed
- This slide directly supports the "bridge gap" argument on Slide 11 by showing *how* the bridge works hour-by-hour

---

## Candidate C: Capture Rate Erosion Deep Dive

**Title:** Capture Rate Trajectories: The ISO-by-ISO Erosion Curve
**Thesis:** VRE capture rates are declining at different speeds across ISOs, with CAISO solar already below 30% — making regional siting the critical investment variable.

### Evaluation

**Already covered in the deck:**
- Slide 5 (LMP Cannibalization): Explicitly titled "The Standalone VRE Trap." Contains capture rate erosion chart by market and revenue stack comparison. Covers CAISO solar (<30%), ERCOT wind (54%), and PJM as the premium market.
- Slide 12 (Regional Opportunity Matrix): ISO-by-ISO breakdown with demand growth, cannibalization risk, capacity value, interconnection score, and storage arbitrage ratings. Includes the Tier 1/2/3 classification with ERCOT's $17/MWh capture rate called out.

**Gap analysis:** Capture rate erosion is the core message of Slide 5, and the regional differentiation is the core message of Slide 12. A deep dive slide would be a third pass at the same data — diminishing returns.

**Would the audience decide differently without this slide?** No. They already see capture rate erosion (Slide 5) and know which ISOs are attractive (Slide 12). A trajectory chart showing "it gets worse over time" adds anxiety but not actionable information beyond what's already conveyed.

**Recommendation: SKIP** (or at most, add ISO-specific trajectory lines to the existing Slide 5 capture rate chart as a Prompt 5 update — but that's incremental, not a new slide)

---

## Candidate D: Storage Pairing Economics

**Title:** Storage Economics: The $/MWh Uplift by ISO
**Thesis:** Adding storage to standalone VRE creates $40-60/MWh in value uplift, varying by ISO — quantifying what "storage willingness" actually costs and returns.

### Evaluation

**Already covered in the deck:**
- Slide 10 (Hybrid Economics): Entire slide dedicated to this. Shows $35 → $94/MWh uplift (+169%), 4hr vs. 8hr vs. 100hr LDES comparison table, ERCOT arbitrage case ($150/kW-yr), tolling agreement IRRs (10-12% vs. 6-8%).

**Gap analysis:** Slide 10 is exactly this candidate slide. The only thing Slide 10 doesn't show is ISO-by-ISO variation in the storage uplift — it uses a single representative example. Adding ISO-specific storage economics would be a nice detail but doesn't change the investment decision.

**Would the audience decide differently without this slide?** No. Slide 10 already makes the case convincingly. ISO-specific storage economics belong in an appendix or backup slide, not the main flow.

**Recommendation: MERGE INTO EXISTING** — If ISO-specific storage data is available from Step 3A, add a small "Storage value by ISO" row to the Regional Opportunity Matrix (Slide 12) rather than creating a new slide. This reinforces Slide 12's role as the comprehensive regional comparison.

---

## Candidate E: Wright's Law Learning Curves

**Title:** Wright's Law: Why VRE Gets Cheaper While SMRs Wait
**Thesis:** Solar (24%), Li-ion (18%), and wind (15%) learning rates guarantee continued cost decline, while SMR costs remain uncertain until 50+ units are deployed.

### Evaluation

**Already covered in the deck:**
- Slide 11 (Clean Firm Won't Arrive in Time): Contains a Wright's Law table with all five technologies (Solar, Li-ion, Wind, SMR, EGS), current LCOE, learning rates, 2035 projections, and "deployments to $50" column. Also includes the "50+ units for cost convergence" SMR reality check callout.

**Gap analysis:** This is literally Slide 11's right column. A standalone Wright's Law slide would be a duplicate.

**Would the audience decide differently without this slide?** No — they already see it on Slide 11.

**Recommendation: SKIP** — Slide 11 already handles this comprehensively. The data from step5_2e_wrights_law_curves.py is already incorporated.

---

## Summary

| Candidate | Recommendation | Rationale |
|-----------|---------------|-----------|
| A: GHG Protocol Scope 2 | **SKIP** | Already woven across 4 slides (7, 9, 14, 15). Stronger distributed than concentrated. |
| B: VRE Portfolio Value | **ADD as Slide 10B** | Genuine gap — deck never visualizes the 24-hour three-asset synthesis. The "aha" moment for an IPP audience. |
| C: Capture Rate Deep Dive | **SKIP** | Redundant with Slides 5 + 12. Third pass at same data. |
| D: Storage Pairing Economics | **MERGE into Slide 12** | Slide 10 covers this; ISO-specific data fits as a row in Regional Matrix. |
| E: Wright's Law Curves | **SKIP** | Literally Slide 11's right column already. |

**Net result:** +1 slide (Candidate B as Slide 10B). Deck goes from 17 → 18 slides, which is at the upper limit but justified because this slide fills the single largest analytical gap in the narrative arc.

**Revised flow after addition:**
1. Title
2. Where the Load Is Landing
3. The Clean Energy Gap
4/4B/4C/4D. Gap Analysis (2030/2035/2040/2050)
5. LMP Cannibalization
6. Energy Timeline
7. Colocation Developers
8. Hyperscaler Contract Dominance
9. The 24/7 CFE Premium
10. Hybrid Economics: Storage Changes Everything
**10B. The Portfolio Effect: VRE Completes the 24/7 Product** ← NEW
11. Clean Firm Won't Arrive in Time
12. Regional Opportunity Matrix
13. The Investment Case
14. Three Procurement Strategies
15. Cross-Regional Procurement Flow
16. Analytical Framework
17. Appendix
