# Citation Audit: VRE Research Integration

## Part A: Demand, Infrastructure & SMR (Citations 1–16)

**Audited:** March 27, 2026
**Source:** `data-center-cfe/research/vreresearch.md` (first ~40% — macroeconomic landscape, AI data center physics, demand forecasts, SMR NOAK timeline, clean energy gap quantification)
**Deck:** `data-center-cfe/output/vre-investment-thesis-deck.html` (16 slides)
**Filter:** VRE standalone investment thesis value. Deprioritize generic DC growth hype; prioritize VRE economics, procurement behavior, regulatory shifts, regional siting, thesis contradictions.

### Audit Table

| # | Citation | Key Data Point | Already in Deck? | Action | Priority |
|---|---------|---------------|-------------------|--------|----------|
| 1 | **McKinsey — "Scaling bigger, faster, cheaper data centers"** | Global DC capex to exceed $1.7T by 2030; AI server = 10.2 kW peak (15× increase in power density) | Partially — Slide 2 references demand supercycle but uses BNEF/Goldman/WoodMac as primary sources, not McKinsey capex figure. The $1.7T figure and power density stats are not in the deck. | **SKIP** — The $1.7T capex stat is about DC infrastructure spending, not VRE investment case. Power density (10.2 kW/server) is background context already implied by the demand narrative. Deck uses more recent/specific forecasts (BNEF 106 GW, Goldman 165%). | L |
| 2 | **Brattle Group — "Electricity Demand Growth and Forecasting"** | US total electricity demand projected to grow 50% by 2035; peak demand +30% (5× historical growth rate) | Partially — Slide 2 callout references "17–25% grid increase by 2030" (sourced to WRI/LBNL/EPRI/BCG). The Brattle 50% by 2035 figure is broader (all-sector, not DC-specific) and not explicitly in the deck. | **ADD** — The "5× historical growth rate" framing is compelling for Slide 2. Could strengthen the "largest demand shock since post-WWII" callout with the Brattle data point as a secondary source. | M |
| 3 | **XENDEE — "Meeting Energy Demand for Data Centers With DER and Future SMR"** | DER + SMR co-optimization for DC load; behind-the-meter microgrid configurations | Not in deck. | **SKIP** — DER/microgrid optimization is tangential to the standalone VRE pipeline thesis. The deck's focus is utility-scale VRE + storage within ISO markets, not behind-the-meter DER. | L |
| 4 | **Grid Strategies — "Power Demand Forecasts Revised Up"** | Utility demand forecasts systematically underestimate load growth; upward revisions across ISOs | Not directly cited, but Slide 2/3 implicitly captures this through multiple high-range forecasts (BNEF, Goldman, WoodMac). | **SKIP** — The thesis already uses the highest credible demand estimates. Adding "forecasts keep getting revised up" is rhetorical rather than analytical. The deck's multi-source approach already communicates this. | L |
| 5 | **LBNL — "2024 United States Data Center Energy Usage Report"** | US DC demand: 325–580 TWh by 2028 (6.7%–12.0% of US electricity) | **Yes** — Slide 2 source note explicitly cites LBNL range. Also referenced in vreresearch.md demand forecast table. | **SKIP** — Already fully incorporated in Slide 2 sources. | L |
| 6 | **DOE Office of Electricity — "Clean Energy Resources to Meet Data Center Demand"** | Federal assessment of clean energy supply adequacy for DC load growth; interconnection queue analysis (2,000+ GW proposed, 94% zero-carbon, only 14% solar/11% battery reach COD) | Partially — Slide 4 callout references 231 GW pipeline and interconnection constraints. The specific 14% solar / 11% battery completion rate stats from DOE are not in the deck but are in vreresearch.md Section 4 (clean energy gap). | **ADD** — The 14% solar / 11% battery completion rates are powerful data points for Slide 5 (LMP Cannibalization) or Slide 4 (Clean Energy Gap). They quantify *why* the queue doesn't translate to real supply — directly supporting the VRE scarcity thesis. Add to Slide 4 source notes or create a callout. | H |
| 7 | **IEA — "AI is set to drive surging electricity demand"** | Global DC demand to double by 2030 to ~945 TWh (more than Japan's total consumption) | **Yes** — Slide 2 bullet: "Global DC demand: 415 TWh (2024) → 945 TWh by 2030 — more than Japan's total consumption." Source note cites IEA 4E. | **SKIP** — Already fully incorporated. | L |
| 8 | **Brattle Group — "The Future of Clean Energy"** | Clean energy supply-demand dynamics; corporate procurement acceleration; market structure evolution | Not directly cited in deck. Brattle's broader clean energy framing overlaps with the deck's narrative but doesn't add a specific data point not already covered. | **SKIP** — General framing piece. The deck already builds this narrative from more specific sources (BNEF PPA data, DC Byte site data, company-specific gap analysis). No unique quantitative contribution. | L |
| 9 | **Future Markets Inc. — "Global Advanced Nuclear Technologies Market 2026-2045"** | SMR market size projections; global SMR capacity: 1–3 GW by 2040, scaling to 150 GW by 2045 under aggressive scenarios | Partially — Slide 10 ("Clean Firm Won't Arrive in Time") covers SMR timeline. The specific 1–3 GW by 2040 / 150 GW by 2045 range from Future Markets is in vreresearch.md but not explicitly in the deck (deck uses NuScale/X-energy/Kairos timelines). | **UPDATE** — The 1–3 GW by 2040 figure is *more conservative* than what Slide 10 may imply. If Slide 10's SMR timeline is optimistic, this data point provides a credible bear-case anchor. Add to Slide 10 source notes as a range bound. | M |
| 10 | **Nuclear Innovation Alliance — "Advanced Reactor Deployment Timelines"** | FOAK demonstrations targeting early 2030s (X-energy, TerraPower via DOE ARDP); NOAK commercialization not anticipated until late 2030s–2045 | Partially — Slide 10 covers FOAK/NOAK timeline. The NIA source corroborates the deck's existing narrative. | **SKIP** — Corroborative, not additive. The deck already presents this timeline accurately. Could add as a secondary source in Slide 10 notes for credibility, but low priority. | L |
| 11 | **NREL — "Nuclear | Electricity | 2024 | ATB"** | NREL ATB 2024 nuclear cost projections; provides standardized LCOE benchmarks for nuclear (large and SMR) | Partially — Slide 4C/4D source notes reference "NREL ATB 2024 cost curves" generically. The specific NREL nuclear LCOE figures aren't displayed in the deck. | **ADD** — If creating a cost comparison slide (e.g., LCOE waterfall: VRE vs. nuclear vs. gas), NREL ATB nuclear data would be essential. For existing slides, it's already referenced. Consider for Prompt 4 (new analytical opportunities). | M |
| 12 | **GOV.UK — "Small modular reactors" (cost reduction study)** | UK government analysis of SMR cost reduction pathways; 5–10% cost reduction per doubling of cumulative deployment | Not in deck. The 5–10% learning rate is referenced in vreresearch.md but the UK-specific study isn't cited in the deck. | **SKIP** — UK-specific study adds geographic diversification to the SMR cost narrative but doesn't change the core thesis. The learning rate concept is already embedded in the deck's SMR timeline framing. The deck is US-focused. | L |
| 13 | **Idaho National Laboratory — "Meta-Analysis of Advanced Nuclear Reactor Cost"** | INL meta-analysis of advanced reactor costs; confirms elevated FOAK costs and slow learning curves | Not directly cited. Deck references DOE Liftoff reports for nuclear cost context. | **ADD** — INL meta-analysis is a high-credibility US national lab source. If Slide 10 is updated with more granular cost data, this should be a primary reference. Particularly valuable for the "15-year gap" argument — INL's cost data supports the claim that nuclear won't be cost-competitive soon. | M |
| 14 | **GlobeNewswire — "Nuclear SMRs Market Report 2025-2045"** | Market research report on SMR commercial outlook; market size and deployment projections | Not in deck. | **SKIP** — Commercial market research report. Less credible than DOE/INL/NREL sources already used. Projections likely overlap with Future Markets (#9) and NIA (#10). No unique data point for the VRE thesis. | L |
| 15 | **McKinsey — "How data centers and the energy sector can sate AI's hunger"** | DC energy strategy recommendations; emphasizes need for diverse clean energy portfolios; McKinsey's view on the "energy trilemma" for AI | Not directly cited. Deck's narrative aligns with McKinsey's framing but uses primary data sources. | **SKIP** — Strategy consulting framing piece. The deck already presents a more specific, quantified version of this argument (the VRE + storage + clean firm portfolio thesis). No unique data point. | L |
| 16 | **PJM — "PJM's Role in the Energy Transition"** | PJM resource adequacy analysis; thermal retirement forecasts (up to 40 GW by 2030); peak demand projections; generation interconnection challenges ("D-minus" scorecard) | Partially — Slide 3 references "PJM alone projects 55 GW of large load by 2030." Slide 11 (Regional Opportunity Matrix) likely covers PJM. The 40 GW thermal retirement figure is in vreresearch.md but may not be in the deck. | **UPDATE** — The 40 GW thermal retirement figure is a critical data point for the clean energy gap narrative. Verify it's in Slide 4 or 11. If not, add to Slide 11's PJM row or Slide 4's gap analysis. The "D-minus" interconnection scorecard is already in the deck (Slide 11 PJM section). | H |

### Summary

| Action | Count | Citations |
|--------|-------|-----------|
| **SKIP** | 9 | #1, #3, #4, #5, #7, #8, #12, #14, #15 |
| **UPDATE** | 2 | #9, #16 |
| **ADD** | 4 | #2, #6, #11, #13 |
| **NEW SLIDE** | 0 | — |
| **SCRAPE** | 0 | — |

**9 SKIP, 2 UPDATE, 4 ADD, 0 NEW SLIDE, 0 SCRAPE**

### Key Findings

1. **Demand citations (1–8) are largely incorporated.** The deck uses more recent/specific sources (BNEF 1H 2026, Goldman, WoodMac, DC Byte) that supersede the vreresearch.md citations. Only Brattle's "5× historical growth rate" (#2) and DOE's interconnection completion rates (#6) add meaningful new data.

2. **SMR citations (9–14) corroborate but don't change the thesis.** The deck's SMR timeline narrative is sound. The main additions would be: (a) Future Markets' conservative 1–3 GW by 2040 figure as a bear-case anchor (#9), and (b) INL meta-analysis as a high-credibility cost source (#13).

3. **No citations warrant a new slide.** The demand and SMR themes are thoroughly covered by existing Slides 2, 3, 4, 10, and 11.

4. **Priority actions:**
   - **HIGH:** Add DOE interconnection completion rates (14% solar, 11% battery) to Slide 4 or 5 (#6). Add/verify PJM 40 GW thermal retirement in Slide 4 or 11 (#16).
   - **MEDIUM:** Strengthen Slide 2 with Brattle "5× growth rate" (#2). Add INL meta-analysis to Slide 10 sources (#13). Add NREL ATB nuclear cost data if building cost comparison content (#11). Update Slide 10 with Future Markets conservative SMR range (#9).

---

## Part B: Corporate Procurement & Hyperscaler (Citations 17–33)

**Audited:** March 27, 2026
**Source:** `data-center-cfe/research/vreresearch.md` (middle ~35% — interconnection queues, corporate procurement, hyperscaler deals, colocation providers, vertical integration)
**Deck:** `data-center-cfe/output/vre-investment-thesis-deck.html` (16 slides)
**Filter:** VRE standalone investment thesis value. Prioritize procurement behavior, hyperscaler deal data, interconnection bottlenecks, colocation market dynamics, and vertical integration risk signals.

### Audit Table

| # | Citation | Key Data Point | Already in Deck? | Action | Priority |
|---|---------|---------------|-------------------|--------|----------|
| 17 | **LBNL — "Queued Up: 2024 Edition"** | Interconnection completion rates: only 14% solar / 11% battery reach COD; average queue time ~5 years; 2,000+ GW proposed (94% zero-carbon) | Partially — Slide 11 source note references "LBNL — 'Queued Up' interconnection study (2025 ed.)" generically. Slide 16 (Appendix) lists it. However, the specific 14%/11% completion rate stats and 5-year queue time are NOT surfaced as data points in any slide body or callout. | **UPDATE** — The 14% solar / 11% battery completion rates are the single most powerful interconnection bottleneck data points for the VRE scarcity thesis. They belong as a callout on Slide 11 (Regional Opportunity Matrix) or Slide 4 (Clean Energy Gap). The 5-year queue time should appear alongside interconnection risk in Slide 12's risk table. | H |
| 18 | **Grid Strategies — "AEI 2024 Generator Interconnection Scorecard"** | ISO-level interconnection grades: ERCOT "B", CAISO "B-", PJM "D-minus", ISO-NE worst. Process efficiency comparison across ISOs. | Partially — Slide 11 implicitly captures this (PJM noted as capacity-constrained, ERCOT as fast-interconnection). The specific letter grades are in vreresearch.md ("D-minus" for PJM) but NOT displayed in the deck. | **UPDATE** — Add ISO scorecard grades to Slide 11's "Queue Position" column. Currently that column has generic text; replacing with letter grades (PJM: D-, ERCOT: B, CAISO: B-) is more impactful and citable. Would strengthen the regional differentiation argument. | M |
| 19 | **Wood Mackenzie — "US utilities to face significant challenge"** | US utilities facing unprecedented demand surge; utility large load commitments reach 160 GW. Power demand surges for first time in decades. | **Yes** — Slide 3 source note cites "Wood Mackenzie, 'US utility large load commitments reach 160 GW' (2025)." The data is incorporated into the DC pipeline narrative. | **SKIP** — Already incorporated. The WoodMac demand data is embedded in Slides 2–3. No new data point to extract. | L |
| 20 | **Belfer Center — "AI, Data Centers, and the U.S. Electric Grid"** | Harvard Belfer Center analysis of AI/DC grid impacts; frames the challenge as a "watershed moment" for US electric grid. Policy recommendations and reliability risks. | Not directly cited in deck body. The Belfer analysis provides academic framing but the deck uses more specific quantitative sources (BNEF, DC Byte, PJM/ERCOT data). | **SKIP** — Academic framing piece without unique quantitative data not already covered by primary ISO/market sources. The "watershed moment" language is rhetorical. Could add as a secondary reference in Slide 16 bibliography but no slide content changes needed. | L |
| 21 | **Brown Advisory — "The Data Center Balancing Act"** | Investment advisor perspective on sustainable AI growth; tensions between DC expansion and clean energy commitments. ESG investor framing. | Not in deck. | **SKIP** — Investment advisory perspective piece. The deck already presents a more sophisticated version of this argument through its gap analysis (Slide 4) and procurement timeline (Slide 6). No unique data point for the VRE thesis. | L |
| 22 | **Flexidao — "How to Practically Start a 24/7 CFE Journey"** | Practical guide for 24/7 CFE procurement; operational steps for hourly matching implementation. Companies claiming 100% renewable under annual matching achieve only 40–65% hourly CFE in reality. | Partially — Slide 6 callout references the 24/7 CFE concept. The specific 40–65% hourly gap stat from Flexidao is referenced in vreresearch.md (Section on GHG Protocol) but NOT in the deck. | **ADD** — The "40–65% actual hourly CFE" stat is a devastating data point that belongs on Slide 8 (GHG Protocol / EAC Arbitrage) or Slide 9 (24/7 CFE Strategy). It quantifies exactly why annual matching is insufficient and why storage-paired VRE is essential. Add as a callout or KPI card. | H |
| 23 | **MIT DSpace — "Demand-Driven Decarbonization: Impact of 24/7"** | MIT thesis on demand-driven decarbonization; quantifies impact of voluntary 24/7 low-carbon procurement on grid emissions. Academic validation of hourly matching effectiveness. | Not directly cited in deck. | **SKIP** — Academic thesis that validates the hourly matching concept already central to the deck's thesis. Corroborative but doesn't add a specific new data point. Could reference in Slide 16 bibliography for academic credibility but low priority. | L |
| 24 | **Bloomberg NEF — "Corporate Clean Energy Buying Fell in 2025"** | Global corporate clean energy buying fell 10% in 2025 to 55.9 GW — first decline after nearly a decade of growth. Market bifurcation: smaller buyers paralyzed by regulatory uncertainty and high prices. | Partially — Slide 2 KPI references "106 GW" BNEF figure (US DC demand by 2035). Slide 6 KPI shows "49%" Big 4 share. But the headline stat — **global procurement fell 10% to 55.9 GW** — is NOT in the deck despite being a critical market signal. | **SCRAPE + ADD** — The 10% decline is a powerful counterpoint that *strengthens* the VRE thesis: smaller buyers are dropping out, concentrating demand on hyperscalers who MUST buy from established IPPs. This market consolidation narrative belongs as a callout on Slide 6 or a new insight on Slide 4. Scrape the BNEF source for the exact figure and buyer-tier breakdown. | H |
| 25 | **ESG Today — "Amazon, Meta, Google, Microsoft Account for Half"** | Big 4 hyperscalers accounted for 49% of all global clean energy purchasing in 2025. Massive market concentration — four companies driving half of all deals. | **Yes** — Slide 2 KPI card: "49% — Global PPA volume from Big 4 hyperscalers." Slide 6 KPI card: "49% — Big 4 share of global clean energy buying (2025)." | **SKIP** — Already fully incorporated as a prominent KPI on two slides. The 49% figure is one of the deck's headline stats. | L |
| 26 | **Amazon Sustainability — "Carbon-free energy"** | Amazon's sustainability page: details on carbon-free energy commitments, 700+ projects globally, 40+ GW cumulative portfolio. | Partially — Slide 6 timeline entry: "Amazon — 700+ projects globally; 40+ GW cumulative portfolio." Slide 4 table includes Amazon's gap analysis. | **SKIP** — Already incorporated. Amazon's portfolio data is in both the timeline (Slide 6) and gap analysis (Slide 4). No new data to extract. | L |
| 27 | **Bloomberg NEF/BCSE — "2026 Sustainable Energy in America Factbook"** | Annual factbook on US sustainable energy market; comprehensive market data on clean energy deployment, investment trends, policy impacts. | Not directly cited in deck body, though BNEF data is used throughout (Slide 2 demand forecasts, Slide 5 capture rates, Slide 7 storage costs). | **SKIP** — The factbook is a compendium source. Specific data points from BNEF are already used throughout the deck from more targeted BNEF publications. No single data point from the factbook would add to what's already incorporated. | L |
| 28 | **Introl Blog — "Google's $4.75B Intersect Power Acquisition"** | Google acquired Intersect Power for $4.75B (10.8 GW solar + storage); signals shift from passive PPA buyer to direct infrastructure owner to bypass interconnection queues. | **Yes** — Slide 6 timeline: "Google / Intersect Power — $4.75B acquisition; shift to direct infrastructure ownership." Slide 6 callout: "Google's $4.75B Intersect Power acquisition signals a shift from passive PPA buyer to infrastructure owner — but the asset acquired was *solar + storage*, not nuclear. Even the pivot reinforces VRE." | **ADD (risk narrative)** — While the deal itself is in the deck, its implications as a **vertical integration risk** for IPPs are underplayed. If hyperscalers bypass IPPs by acquiring developers directly, the addressable market shrinks. This risk should be explicitly addressed in Slide 12 (Investment Case) risk table or Slide 15 (Risks). Currently the deck frames it positively ("even the pivot reinforces VRE") but should also flag: "If Big 4 vertically integrate at scale, Tier 2 colos become the primary VRE offtaker market." | M |
| 29 | **GlobeNewswire — "U.S. Data Center Construction Market 2026-2031"** | US DC construction market outlook; key investors include Apple, AWS, CyrusOne, DataBank, Digital Realty, Equinix, Google, Meta, Microsoft. CAGR >10% through decade. | Partially — Slide 3 covers DC pipeline data from DC Byte (2,073 sites, 231 GW). The GlobeNewswire market report adds the >10% CAGR figure used in vreresearch.md ("colocation market expected to expand at CAGR of over 10%"). | **UPDATE** — The >10% CAGR stat is already used on Slide 7 as "10%+ CAGR" KPI for US colocation market growth. Verify the source attribution is correct in Slide 7's source note. If not attributed, add GlobeNewswire as the source. | L |
| 30 | **Wood Mackenzie — "Largest data centers in the US: top 10 questions"** | Top 10 questions on US DCs answered; market sizing, site selection factors, power procurement challenges. | Not directly cited in deck body. WoodMac is cited for demand data (Slides 2–3) but this specific "top 10 questions" piece isn't referenced. | **SKIP** — General Q&A format piece. Specific WoodMac data already incorporated through more targeted publications. No unique quantitative data point for the VRE thesis. | L |
| 31 | **IEA 4E — "Data Centre Energy Use: Critical Review"** | Critical review of DC energy models and results; IEA methodology for estimating DC energy consumption. Provides the 945 TWh global demand figure. | **Yes** — Slide 2 bullet: "Global DC demand: 415 TWh (2024) → 945 TWh by 2030." Slide 5A source note: "IEA Energy & AI Base Case (1,200 TWh global)." | **SKIP** — Already fully incorporated. The IEA 4E data underpins the deck's global demand narrative. | L |
| 32 | **Clarity AI — "Data Center Emissions Are Rising"** | Analysis showing DC emissions are rising despite corporate clean energy commitments; firms may not be doing enough. Quantifies the gap between commitments and actual emissions reductions. | Not in deck. | **SCRAPE + ADD** — This is valuable for Slide 7 (Colocation Provider Gap) or Slide 4 (Clean Energy Gap). The "emissions rising despite commitments" framing directly supports the thesis that annual matching is failing and hourly matching is necessary. Scrape for specific emissions growth figures and company-level data. Could strengthen the colo provider narrative on Slide 7. | M |
| 33 | **Iron Mountain — "Data Center Sustainability"** | Iron Mountain's sustainability page; first colocation provider to commit to 24/7 CFE by 2040. Details on green data center operations. | **Yes** — Slide 7 colocation table includes Iron Mountain row: "100% annual RE currently; 24/7 CFE by 2040." The 24/7 CFE commitment is highlighted as differentiating from other colo providers. | **SKIP** — Already incorporated in Slide 7's colocation provider analysis. Iron Mountain's 24/7 CFE commitment is a key data point already in the deck. | L |

### Summary

| Action | Count | Citations |
|--------|-------|-----------|
| **SKIP** | 9 | #19, #20, #21, #23, #25, #26, #27, #30, #31 |
| **UPDATE** | 3 | #17, #18, #29 |
| **ADD** | 2 | #22, #28 |
| **SCRAPE + ADD** | 2 | #24, #32 |
| **NEW SLIDE** | 0 | — |
| **SCRAPE** | 0 | — |

**9 SKIP, 3 UPDATE, 2 ADD, 2 SCRAPE+ADD, 0 NEW SLIDE**

### Key Findings

1. **Interconnection queue data (#17–18) needs surfacing.** The LBNL "Queued Up" 14%/11% completion rates and Grid Strategies' ISO scorecard grades are referenced in source notes but not displayed as data points in the deck body. These are the strongest quantitative arguments for why the 2,000+ GW queue is illusory — and directly support the VRE scarcity thesis. **Priority: Add to Slide 11 (Regional Matrix) and/or Slide 4 (Clean Energy Gap).**

2. **BNEF 2025 procurement decline (#24) is a missing market signal.** The 10% drop in global corporate clean energy buying (55.9 GW) is NOT in the deck despite being highly relevant. The market bifurcation — smaller buyers dropping out while Big 4 consolidate to 49% share — strengthens the thesis that established IPPs with VRE pipelines become essential suppliers. **Priority: Scrape BNEF source for exact figures and add to Slide 6 or Slide 4 as a market consolidation callout.**

3. **Google Intersect Power (#28) vertical integration risk is underplayed.** The $4.75B acquisition is in the deck but framed only positively ("even the pivot reinforces VRE"). The risk that hyperscaler vertical integration shrinks the IPP addressable market needs explicit acknowledgment. **Priority: Add vertical integration risk to Slide 12 risk table with mitigation (Tier 2 colos as primary market).**

4. **Flexidao 40–65% hourly CFE stat (#22) is a high-value missing data point.** This quantifies exactly how much "100% renewable" claims under annual matching fall short on an hourly basis — the core driver for the GHG Protocol update and the VRE+Storage value proposition. **Priority: Add as KPI or callout on Slide 8 or 9.**

5. **Colocation provider data (#32–33) is largely covered.** Slide 7 already has comprehensive colo provider analysis. Clarity AI's "emissions rising" framing (#32) could strengthen the narrative but is medium priority. Iron Mountain (#33) is already fully incorporated.

6. **Priority actions:**
   - **HIGH:** Surface LBNL 14%/11% completion rates in slide body (#17). Scrape + add BNEF 2025 procurement decline data (#24). Add Flexidao 40–65% hourly CFE gap stat (#22).
   - **MEDIUM:** Add ISO scorecard grades to Slide 11 (#18). Flag vertical integration risk from Google/Intersect (#28). Scrape Clarity AI emissions data for Slide 7 (#32).
   - **LOW:** Verify GlobeNewswire CAGR source attribution on Slide 7 (#29). All others are SKIP.

---

## Part C: Regulatory, Regional & VRE Economics (Citations 34–47)

**Audited:** March 27, 2026
**Source:** `data-center-cfe/research/vreresearch.md` (final ~25% — GHG Protocol Scope 2, regional ISO analysis, VRE capture rates, storage/BESS, grid operations)
**Deck:** `data-center-cfe/output/vre-investment-thesis-deck.html` (16 slides)
**Filter:** VRE standalone investment thesis value. Prioritize GHG Protocol regulatory shifts, regional ISO differentiation, storage economics, and VRE capture rate data.

### Audit Table

| # | Citation | Key Data Point | Already in Deck? | Action | Priority |
|---|---------|---------------|-------------------|--------|----------|
| 34 | **Watershed — "Analyzing the GHG Protocol's proposed scope 2 changes"** | Analysis of proposed Scope 2 updates: hourly matching + deliverability requirements will disqualify 30–50% of existing unbundled REC portfolios. REC price increases of 3–7× in constrained regions following in-region eligibility shifts. | Partially — Slide 7 and 8 discuss GHG Protocol Scope 2 shifts, hourly matching, and the 40–65% actual CFE gap. The 30–50% REC disqualification figure and 3–7× price multiplier are in vreresearch.md (line 208) but NOT explicitly surfaced in the deck as data points. | **ADD** — The 30–50% REC disqualification rate and 3–7× price premium are the most concrete quantitative arguments for why the GHG Protocol update creates a VRE+Storage revenue supercycle. Add as a callout on Slide 8 (24/7 CFE Premium) or Slide 9 (Storage). These numbers directly justify the $35–50/MWh PPA premium already shown on Slide 8. | H |
| 35 | **GHG Protocol — "Upcoming Scope 2 Public Consultation: Hourly Matching"** | Official GHG Protocol consultation document confirming hourly matching and deliverability as proposed pillars. Timeline: public consultation late 2025, finalization expected 2027. | **Yes** — Slide 7 includes "GHG Protocol Scope 2 finalization" milestone in 2027. Slide 8 discusses annual-to-hourly shift. The timeline and pillars are incorporated. | **SKIP** — Already incorporated. The 2027 timeline and hourly+deliverability framework are in Slides 7–8. Could add as a source citation in Slide 16 bibliography if not already there, but no content changes needed. | L |
| 36 | **Climate Resource Solutions — "Missteps in Proposed Updates to GHG Protocol"** | Critical analysis arguing proposed GHG Protocol updates may have unintended consequences — could penalize early movers or create perverse incentives. Counterpoint to the bullish hourly-matching narrative. | Not in deck. No counterargument to the GHG Protocol update is presented. | **ADD (risk)** — This is a meaningful risk factor. If the GHG Protocol update is weakened, delayed beyond 2027, or modified to grandfather existing annual-matching contracts, the VRE+Storage premium thesis weakens. Add as a risk row in Slide 12 (Investment Case) risk table: "GHG Protocol delay/dilution" with mitigation "Storage arbitrage revenue is regulation-independent; only the REC premium is at risk." | M |
| 37 | **World Resources Institute — "24/7 Carbon-Free Energy Resources & Tools"** | WRI toolkit for implementing 24/7 CFE procurement. Provides methodology frameworks, measurement tools, and best practices for hourly matching. | Not directly cited. Slide 8 covers 24/7 CFE concepts using Google/Princeton and Eurelectric/EY as sources. | **SKIP** — Toolkit/methodology resource. The deck already presents 24/7 CFE concepts with sufficient rigor using primary research sources. WRI toolkit is an implementation guide, not an analytical data source. Could add to Slide 16 bibliography for completeness. | L |
| 38 | **Wood Mackenzie — "US seasonal power outlooks Summer 2025"** | Seasonal power price forecasts by ISO; summer 2025 outlook shows elevated prices in PJM/ERCOT due to demand growth outpacing supply additions. Seasonal price spreads relevant to storage arbitrage. | Not specifically cited. WoodMac is referenced generically in Slide 15–16 for demand data, but the seasonal power outlook and ISO-specific price forecasts are not in the deck. | **SCRAPE** — Seasonal price spread data by ISO would strengthen Slide 9 (Storage revenue stack) and Slide 11 (Regional Matrix). Summer/winter price differentials directly quantify storage arbitrage opportunity. Target: `data-inputs/woodmac-seasonal-prices.json` with fields: ISO, season, avg wholesale price, peak/off-peak spread, year-over-year change. | M |
| 39 | **Utility Dive — "ERCOT, CAISO offer best grid interconnection"** | ERCOT and CAISO have the fastest, most efficient interconnection processes among US ISOs. Contrasts with PJM's severe backlog ("D-minus" grade). | Partially — Slide 11 implicitly captures this: ERCOT rated "Strong Buy," CAISO rated "Buy (selective)," PJM noted for queue friction. The interconnection speed advantage is embedded in the regional ratings but not explicitly cited as a differentiator with specific process metrics. | **UPDATE** — Add interconnection timeline estimates to Slide 11's regional matrix. Currently the matrix has qualitative ratings; adding "Avg. queue time: ERCOT 1–2yr, CAISO 2–3yr, PJM 4–5yr" would quantify the speed advantage and strengthen the Tier 1/2/3 justification. | M |
| 40 | **GE Vernova — "A tale of two ISOs: ERCOT and CAISO"** | Comparative analysis of ERCOT vs. CAISO market structures, VRE penetration levels, and grid management approaches. ERCOT energy-only market vs. CAISO capacity market. Highlights divergent approaches to managing high VRE penetration. | Partially — Slide 11 covers ERCOT vs. CAISO with different strategic recommendations. Slide 4 mentions CAISO solar <30% capture and ERCOT wind at 54%. The market structure differences (energy-only vs. capacity) are implied but not explicitly compared. | **UPDATE** — The ERCOT/CAISO market structure comparison would strengthen Slide 11's rationale. Add a footnote or callout: "ERCOT's energy-only market creates extreme price volatility ($150+/kW-yr battery arbitrage) vs. CAISO's capacity market which caps upside but reduces risk." This explains *why* the strategic plays differ by region. | M |
| 41 | **Grid Strategies — "Strategic Industries Surging: Driving US Power Demand"** | Analysis of industrial load growth beyond data centers — manufacturing reshoring, EV charging, electrification of industrial heat. Broader demand context showing DCs are one of several simultaneous demand shocks. | Not directly cited. Slide 2–3 focus on DC-specific demand. The broader industrial demand context (manufacturing, EVs, heat) is not in the deck. | **SKIP** — The deck is intentionally focused on the DC/AI demand driver for the VRE thesis. Adding manufacturing/EV demand dilutes the narrative. The broader demand context is implicit — all load growth tightens supply and benefits VRE. No specific data point changes the investment case. | L |
| 42 | **Modo Energy — "US Research Roundup: BESS insights Q3 2025"** | BESS performance data: ERCOT 4hr battery arbitrage revenue $150+/kW-yr actual; degradation curves; utilization rates; revenue by market (energy arbitrage vs. ancillary services vs. capacity). | Partially — Slide 9 cites "$150/kW-yr" ERCOT battery arbitrage. Slide 16 source note: "Modo Energy — ERCOT BESS revenue analysis (Q3 2025)." The headline revenue figure is incorporated but granular breakdowns (degradation, utilization, revenue mix) are not. | **SCRAPE** — The detailed BESS revenue breakdown (energy vs. ancillary vs. capacity by ISO) would strengthen Slide 9's storage revenue stack and provide backup for the $150/kW-yr claim. Target: `data-inputs/modo-bess-revenue.json` with fields: ISO, duration (4hr/8hr), revenue_energy, revenue_ancillary, revenue_capacity, utilization_pct, degradation_annual_pct. | H |
| 43 | **Amperon — "How the Grid Changed in 2024"** | Grid transformation metrics: record VRE penetration hours, storage deployment acceleration, demand growth patterns by region. Provides 2024 retrospective on grid evolution. | Not in deck. No Amperon references found. | **SKIP** — Retrospective analysis. The deck uses forward-looking data (BNEF, WoodMac, PJM/ERCOT forecasts) rather than 2024 retrospectives. Specific data points (record VRE hours, storage deployment) are covered by ISO-specific sources already in the deck. | L |
| 44 | **California Energy Commission — "2024 Total System Electric Generation"** | CAISO grid reached 62% clean energy in 2024. Generation mix breakdown: solar, wind, hydro, nuclear, geothermal, gas. Quantifies the saturation level driving CAISO's declining VRE value. | Not in deck. The 62% figure is in vreresearch.md (line 226: "62% clean energy") but NOT in the deck. Slide 11 rates CAISO "Declining" opportunity without citing this specific saturation metric. | **ADD** — The 62% clean energy figure is the single best justification for CAISO's "Not Recommended for new VRE" rating. Add to Slide 11's CAISO row as a data point: "Already 62% clean — incremental VRE faces severe diminishing returns." This converts a qualitative rating into a quantified one. | H |
| 45 | **California ISO — "Managing the evolving grid"** | CAISO grid management challenges: duck curve deepening, curtailment increasing, negative pricing hours expanding. Operational perspective on managing high-VRE grid. | Partially — Slide 4 references CAISO solar <30% capture (a consequence of the duck curve). The duck curve and curtailment dynamics are implied but not explicitly cited. | **UPDATE** — Add CAISO curtailment data to Slide 11's CAISO row or Slide 4's cannibalization discussion. "CAISO curtailed X GWh of solar in 2024" would quantify the oversupply problem. However, without scraping the specific curtailment figure, this is a low-priority update — the <30% capture rate already communicates the problem. | L |
| 46 | **Eavor — "California needs clean firm power"** | Eavor (enhanced geothermal developer) argues California's path to 100% clean requires firm, dispatchable clean power — not more VRE. Supports the thesis that CAISO is VRE-saturated and needs firm resources. | Not in deck. Slide 10 mentions "Enhanced geothermal" generically but doesn't cite Eavor or apply the argument to CAISO specifically. | **SKIP** — Self-serving analysis from a geothermal developer. The deck already makes the CAISO case through market data (Slide 11: "Declining" opportunity). Adding a vendor's argument adds bias without analytical rigor. The geothermal point is already in Slide 10's clean firm discussion. | L |
| 47 | **LBNL — "Variable Renewable Energy Participation in Ancillary Services"** | LBNL analysis of VRE participation in ancillary services markets. Quantifies potential revenue uplift from VRE providing frequency regulation, voltage support, and spinning reserves. Emerging revenue stream that improves VRE standalone economics. | Partially — Slide 6 revenue table includes "Ancillary Services: $0 to +$6/MWh." Slide 9 shows "$6/MWh" ancillary in revenue stack. The revenue range is in the deck but not sourced to LBNL specifically. | **UPDATE** — Add LBNL as source citation for the ancillary services revenue line in Slide 6 and 9. The $0–6/MWh range should be attributed. If LBNL provides higher upside estimates for VRE ancillary participation, update the range. Low priority since the figure is already in the deck. | L |

### Summary

| Action | Count | Citations |
|--------|-------|-----------|
| **SKIP** | 5 | #35, #37, #41, #43, #46 |
| **UPDATE** | 4 | #39, #40, #45, #47 |
| **ADD** | 3 | #34, #36, #44 |
| **SCRAPE** | 2 | #38, #42 |
| **NEW SLIDE** | 0 | — |

**5 SKIP, 4 UPDATE, 3 ADD, 2 SCRAPE, 0 NEW SLIDE**

### Key Findings

1. **GHG Protocol citations (#34–36) are the regulatory backbone of the thesis.** The 30–50% REC disqualification rate and 3–7× price premium (#34) are the most impactful missing data points — they directly quantify why the Scope 2 update creates a VRE+Storage revenue supercycle. The counterargument (#36) belongs in the risk table. The official GHG Protocol timeline (#35) is already incorporated.

2. **CAISO "62% clean" (#44) is a critical missing justification.** Slide 11 rates CAISO as "Declining" opportunity but doesn't cite the 62% clean energy saturation figure that makes the case self-evident. This is a one-line add with high impact.

3. **Modo Energy BESS data (#42) needs scraping for Slide 9 depth.** The $150/kW-yr headline is in the deck, but the revenue breakdown (energy vs. ancillary vs. capacity) would make the storage investment case more granular and defensible.

4. **Regional ISO analysis (#38–40) could strengthen Slide 11** with quantified interconnection timelines and market structure comparisons, but the existing qualitative ratings are functional. Medium priority.

5. **No citations warrant a new slide.** The regulatory, regional, and storage themes are covered by existing Slides 4, 8, 9, 11, and 12.

6. **Priority actions:**
   - **HIGH:** Add Watershed 30–50% REC disqualification + 3–7× price premium to Slide 8 (#34). Add CEC 62% clean figure to Slide 11 CAISO row (#44). Scrape Modo BESS revenue breakdown for Slide 9 (#42).
   - **MEDIUM:** Add GHG Protocol delay risk to Slide 12 (#36). Scrape WoodMac seasonal prices for Slides 9/11 (#38). Add interconnection timelines to Slide 11 (#39). Add ERCOT/CAISO market structure comparison to Slide 11 (#40).
   - **LOW:** Update LBNL ancillary source attribution (#47). Add CAISO curtailment data (#45). All others SKIP.

---

## Consolidated Audit Summary

**Completed:** March 27, 2026
**Scope:** All 47 citations from `vreresearch.md` audited against `vre-investment-thesis-deck.html` (16 slides)

### Counts

| Action | Count | Citations |
|--------|-------|-----------|
| **SKIP** | 23 | #1, #3, #4, #5, #7, #8, #10, #12, #14, #15, #19, #20, #21, #23, #25, #26, #27, #30, #31, #35, #37, #41, #43 |
| **UPDATE** | 9 | #9, #16, #17, #18, #29, #39, #40, #45, #47 |
| **ADD** | 9 | #2, #6, #11, #13, #22, #28, #34, #36, #44 |
| **SCRAPE + ADD** | 2 | #24, #32 |
| **SCRAPE** | 2 | #38, #42 |
| **NEW SLIDE** | 0 | — |
| **Total** | **47** | — |

**Summary: 23 SKIP, 9 UPDATE, 9 ADD, 2 SCRAPE+ADD, 2 SCRAPE, 0 NEW SLIDE**

UPDATE targets: Slides 4, 6, 7, 9, 10, 11 (most updates), 12
ADD targets: Slides 2, 4, 5, 8, 9, 10, 11, 12
SCRAPE targets: 4 data files for Prompt 2

### Top 10 Highest-Priority Actions

| Rank | Cit. # | Action | Slide | What to Do | Why It Matters |
|------|--------|--------|-------|------------|----------------|
| 1 | #34 | ADD | Slide 8 | Add Watershed 30–50% REC disqualification rate and 3–7× price premium as callout | Quantifies the GHG Protocol revenue supercycle — the single biggest regulatory catalyst for VRE+Storage value. Without these numbers, the thesis relies on qualitative claims. |
| 2 | #17 | UPDATE | Slide 11 / Slide 4 | Surface LBNL 14% solar / 11% battery interconnection completion rates as a prominent data point | The strongest quantitative argument for why the 2,000+ GW queue is illusory. Already in source notes but buried — needs to be a headline stat. |
| 3 | #24 | SCRAPE+ADD | Slide 6 / Slide 4 | Scrape BNEF 2025 procurement decline data (55.9 GW, -10%) and add as market consolidation callout | The procurement market bifurcation — smaller buyers dropping out, Big 4 consolidating to 49% — proves IPP-scale VRE developers become essential suppliers. |
| 4 | #22 | ADD | Slide 8 / Slide 9 | Add Flexidao 40–65% actual hourly CFE stat as KPI card | Devastatingly quantifies why "100% renewable" under annual matching is a fiction. Already in Slide 7 text; needs promotion to a headline data point on Slide 8. |
| 5 | #44 | ADD | Slide 11 | Add CEC "62% clean energy" to CAISO row in Regional Matrix | Converts CAISO's qualitative "Declining" rating into a self-evident quantified case. One line, high impact. |
| 6 | #42 | SCRAPE | Slide 9 | Scrape Modo Energy BESS revenue breakdown (energy/ancillary/capacity by ISO) | Strengthens the $150/kW-yr ERCOT arbitrage claim with granular revenue components. Makes the storage investment case defensible under scrutiny. |
| 7 | #6 | ADD | Slide 4 / Slide 5 | Add DOE 14% solar / 11% battery completion rates as callout | Corroborates LBNL data (#17) from a federal source. Two independent sources confirming the same bottleneck = high credibility. |
| 8 | #16 | UPDATE | Slide 4 / Slide 11 | Add/verify PJM 40 GW thermal retirement figure | Critical for the clean energy gap narrative — quantifies the supply-side erosion that makes new VRE essential even in PJM. |
| 9 | #36 | ADD (risk) | Slide 12 | Add "GHG Protocol delay/dilution" as risk row with mitigation | Acknowledges the counterargument that the regulatory catalyst could be weakened. Demonstrates intellectual honesty and strengthens the thesis by addressing it head-on. |
| 10 | #28 | ADD (risk) | Slide 12 | Flag vertical integration risk from Google/Intersect $4.75B acquisition | If hyperscalers bypass IPPs by acquiring developers directly, the addressable market shrinks. Needs explicit acknowledgment with Tier 2 colo mitigation. |

### Data Scraping Queue (for Prompt 2)

| Priority | Cit. # | Source | Target Filename | Key Data Fields |
|----------|--------|--------|----------------|-----------------|
| 1 | #24 | Bloomberg NEF — "Corporate Clean Energy Buying Fell in 2025" | `data-inputs/bnef-corporate-procurement-2025.json` | total_gw, yoy_change_pct, big4_share_pct, big4_gw, tier2_gw, buyer_count_change, market_bifurcation_narrative |
| 2 | #42 | Modo Energy — "US Research Roundup: BESS insights Q3 2025" | `data-inputs/modo-bess-revenue.json` | iso, duration_hr, revenue_energy_kwyr, revenue_ancillary_kwyr, revenue_capacity_kwyr, total_revenue_kwyr, utilization_pct, degradation_annual_pct, sample_period |
| 3 | #32 | Clarity AI — "Data Center Emissions Are Rising" | `data-inputs/clarity-dc-emissions.json` | company, year, scope2_emissions_mtco2, yoy_change_pct, renewable_claim_pct, actual_cfe_pct, emissions_gap_narrative |
| 4 | #38 | Wood Mackenzie — "US seasonal power outlooks Summer 2025" | `data-inputs/woodmac-seasonal-prices.json` | iso, season, avg_wholesale_mwh, peak_offpeak_spread_mwh, yoy_price_change_pct, demand_forecast_gw, supply_margin_pct |
