# Power Generator Decarbonization Analysis — Complete Specification

> **Authoritative reference for all design decisions.** If a future session needs context, read this file first.
> Last updated: 2026-02-15.

## Current Status (Feb 15, 2026 — Session 2)

### What was accomplished — Session 2
- [x] **Enriched dashboard.html with sustainability/CDP data** — Each of the 15 companies now has: specific decarbonization target + year, SBTi validation status, CDP disclosure score (2024), interim targets, and 4 key decarbonization milestones. New "Sustainability Targets & CDP" section displays dynamically per company.
- [x] **Integrated CFE optimizer three-strategy framework into fleet-analysis.html** — Three strategy zones (Renewable-First 75-85%, Nuclear-Plus 85-95%, Regional Portfolio 95-100%) with specific $/MWh cost premiums, mapped to fleet archetypes. New "Hockey Stick" cost curve chart showing 5 ISO cost premiums at 10 thresholds.
- [x] **Enhanced targets.html** — Traffic-light table enriched with SBTi status, CDP scores, and specific company milestones for all 15. New "Climate Commitment Scorecard" section showing SBTi/CDP landscape (7 SBTi validated, 4 CDP A-list, 1 CDP F).
- [x] **Enriched index.html** — Updated with optimizer-derived cost data ($10-78/MWh range), SBTi/CDP summary stats, and sister project cross-link.
- [x] **Enhanced policy.html** — Added social cost of carbon comparison chart (RGGI $15, EPA $51, EU ETS $90, Rennert $185), updated LDES cost trajectory from TBD to -50% projected by 2030.
- [x] **Added 2 new key findings to fleet analysis** — "Storage alone can't solve the last mile" and "Regional variation is massive" based on optimizer insights.
- [x] QA verification: all 14 canvas elements across 6 pages have matching JS initialization, no missing tags, Chart.js CDN properly loaded.

### What was accomplished — Session 1
- [x] Project initialized — repo structure, CLAUDE.md with ported workflow preferences
- [x] SPEC.md created with comprehensive specification
- [x] Research launched in parallel (7 agents): MJ Bradley/ERM, top 15 generators, SBTi/IPCC/IEA targets, sustainability/CDP data, CFE optimizer insights, design cues, broader grid decarb context
- [x] eGRID 2023 data acquisition script (analysis/fetch_egrid.py) — EPA site blocked by proxy, script ready for manual download
- [x] 15 company profiles assembled from research (data/processed/company_profiles.json)
- [x] CSS stylesheet built matching hourly-cfe-optimizer visual identity
- [x] **Complete 6-page scrollytelling site built:**
  - [x] Executive Summary (index.html) — Bubble chart, company table, fleet paradox narrative, abatement costs, animated counters
  - [x] Company Dashboard (dashboard.html) — Interactive toggle for all 15 companies, per-company fuel mix, peer comparison, scenarios
  - [x] Fleet Analysis (fleet-analysis.html) — 4 fleet archetypes, MAC curves, Evergy vs Constellation paradox, regional analysis
  - [x] Targets & Feasibility (targets.html) — SBTi/IPCC/IEA/US frameworks, target gap chart, traffic-light table for all 15
  - [x] Policy & Conditions (policy.html) — IRA impact chart, carbon price thresholds, barriers, catalytic policy positions by archetype
  - [x] Methodology (methodology.html) — 22 inline citations, full bibliography, data methodology, limitations
- [x] All work siloed under `power-gen-decarbonization/` — zero modifications to root repo or optimizer

### Next steps (for future session)
1. **Download and process actual eGRID 2023 data** — Run analysis/fetch_egrid.py after manually downloading the EPA Excel file. Replace research-based estimates with authoritative plant-level aggregations
2. **Add comparison mode to dashboard** — Side-by-side or overlay charts for 2+ selected companies
3. **Add historical emissions trajectory charts** — Per-company emissions trends over time
4. **Refine scenario analysis** — Move from illustrative to data-driven scenarios using CFE optimizer cost curves and eGRID-based fleet data
5. **Create standalone deployment** — When outside the proxy sandbox, set up Netlify/Cloudflare Pages deployment for public site
6. **Mobile QA** — Full testing at 320px, 375px, 768px viewports
7. **Cross-link with sister project** — Replace placeholder `#` links with actual CFE optimizer site URL

### Open questions
- [ ] Include federal power generators (TVA, BPA) or just investor-owned + merchant?
- [ ] Confirm time horizon milestones: 2030/2035/2050?
- [ ] Should comparison mode allow selecting arbitrary 2-5 companies for side-by-side?
- [ ] How deep should individual company profiles go (current is ~1 screen each)?
- [ ] Add historical emissions trajectory charts per company?

---

## 1. Project Scope

### Objective
Analyze the top 15 US power generators by TWh and assess the feasibility, cost, and enabling conditions for their decarbonization based on:
- Actual fleet characteristics (fuel mix, emissions intensity, geography)
- Cost curves and resource optimization from the hourly CFE optimizer
- Alignment with major climate targets (SBTi v2, IPCC, IEA NZE, US govt)
- Corporate commitments vs. physical reality of their generation fleets

### Key Questions
1. **What do these fleets actually look like?** — Generation mix, emissions intensity, geographic exposure
2. **What pathways are physically and economically feasible?** — Based on CFE optimizer cost curves
3. **What enables ambition?** — Tax credits (IRA/45Y/45Q/48E), carbon price, RPS mandates, PPA markets
4. **What constrains ambition?** — Stranded assets, rate base recovery, load growth, reliability mandates
5. **Are climate targets realistic?** — SBTi v2 intensity pathways vs. actual fleet transformation costs

### Sister Project
This analysis builds on the hourly CFE optimizer (`jessicacohen554-cyber/hourly-cfe-optimizer`), which models the cost of achieving various levels of hourly clean energy matching across 5 US ISOs. The CFE optimizer's cost curves, resource mix optimization, and regional analysis directly inform the feasibility assessments here.

---

## 2. Top 15 US Power Generators (by TWh)

### Selection criteria
- Ranked by total annual net generation (TWh), most recent available year (2023/2024)
- **Constellation + Calpine treated as combined entity** (acquisition completed Jan 2026)
- Includes both investor-owned utilities and merchant/IPP generators
- Excludes federal power agencies (TVA, BPA) unless user directs otherwise

### Preliminary list (to be finalized with eGRID/EIA data)

| Rank | Company | Type | Est. Generation (TWh) | Key Fuel Mix | Key Markets |
|------|---------|------|----------------------|-------------|-------------|
| 1 | **Constellation Energy** (incl. Calpine) | Merchant/IPP | ~310 | Nuclear (~55%), Gas (~40%), Geo/Hydro/Wind | PJM, ERCOT, CAISO, NYISO, NEISO |
| 2 | **NextEra Energy** (FPL + NEER) | Hybrid (utility + merchant) | ~170 | Gas (~40%), Nuclear (~15%), Wind (~25%), Solar (~15%) | Florida, national (wind/solar) |
| 3 | **Duke Energy** | Vertically integrated | ~203 | Gas (~48%), Nuclear (~32%), Coal (~13%), Renewables (~5%) | Carolinas, Florida, Indiana, Ohio |
| 4 | **Southern Company** | Vertically integrated | ~180 | Gas (~50%), Nuclear (~20%), Coal (~20%), Renewables (~5%) | Georgia, Alabama, Mississippi |
| 5 | **Vistra Energy** | Merchant/IPP | ~140 | Gas (~50%), Coal (~25%), Nuclear (~15%), Solar (~5%) | ERCOT, PJM, NEISO |
| 6 | **AEP (American Electric Power)** | Vertically integrated | ~110 | Gas (~45%), Coal (~30%), Wind (~12%), Nuclear (~5%) | PJM, SPP, ERCOT |
| 7 | **Dominion Energy** | Vertically integrated | ~90 | Gas (~48%), Nuclear (~23%), Coal (~18%), Renewables (~8%) | Virginia, Carolinas |
| 8 | **Berkshire Hathaway Energy** | Vertically integrated | ~85 | Coal (~30%), Gas (~20%), Wind (~25%), Hydro (~10%), Solar (~10%) | Iowa, Utah, Nevada, Oregon |
| 9 | **Entergy** | Vertically integrated | ~80 | Gas (~55%), Nuclear (~30%), Coal (~5%) | Louisiana, Texas, Mississippi, Arkansas |
| 10 | **AES Corporation** | Hybrid (utility + merchant) | ~75 | Gas (~40%), Coal (~15%), Renewables (~30%) | Indiana, Ohio, national renewables |
| 11 | **Xcel Energy** | Vertically integrated | ~70 | Gas (~30%), Wind (~30%), Coal (~20%), Nuclear (~12%) | Minnesota, Colorado, Wisconsin |
| 12 | **Evergy** | Vertically integrated | ~50 | Coal (~40%), Gas (~20%), Wind (~25%), Nuclear (~10%) | Kansas, Missouri |
| 13 | **DTE Energy** | Vertically integrated | ~45 | Gas (~40%), Coal (~30%), Nuclear (~15%), Renewables (~10%) | Michigan |
| 14 | **WEC Energy Group** | Vertically integrated | ~40 | Gas (~45%), Coal (~25%), Nuclear (~15%), Renewables (~10%) | Wisconsin, Michigan, Illinois |
| 15 | **PPL Corporation / Talen Energy** | Hybrid | ~35 | Gas (~50%), Nuclear (~30%), Coal (~10%) | Pennsylvania, Kentucky |

**Notes:**
- Generation figures are preliminary estimates pending eGRID 2023 plant-level aggregation
- Fuel mix percentages are approximate from most recent company reports
- Constellation + Calpine combined: ~210 TWh (Constellation pre-merger) + ~101 TWh (Calpine) = ~310 TWh
- Rankings may shift once eGRID parent company aggregation is completed

---

## 3. Data Sources & Methodology

### Primary data
| Source | Data | Status |
|--------|------|--------|
| **EPA eGRID 2023** | Plant-level: generation, emissions, fuel type, operator, parent company | Acquisition script created — needs manual download (proxy blocks direct) |
| **EIA Form 923** | Monthly generation by plant and fuel | To be downloaded |
| **EIA Form 860** | Plant characteristics, ownership, capacity | To be downloaded |
| **S&P Global** | Parent company → subsidiary mapping, corporate structure | Manual research |
| **MJ Bradley / ERM** | Benchmarking Air Emissions report | Research in progress |
| **Company 10-K filings** | Generation, emissions, targets | Research in progress |
| **CDP disclosures** | Climate targets, Scope 1/2/3 emissions | Research in progress |
| **Sustainability reports** | Decarbonization strategies, progress | Research in progress |

### Parent company aggregation methodology
eGRID 2023 does not include parent company-level aggregation (discontinued after 2009). We aggregate plant-level data by mapping:
1. eGRID plant operator → EIA-860 ownership entity
2. Ownership entity → ultimate parent company (S&P / SEC filings / manual research)
3. Sum generation (MWh) and emissions (tons CO2) by parent
4. Calculate fleet-weighted average emissions intensity

### CFE optimizer integration
Key outputs from the hourly CFE optimizer that inform this analysis:
- **Cost curves** for achieving 75-100% hourly clean energy matching by region
- **Optimal resource mixes** (solar, wind, nuclear, CCS-CCGT, battery, LDES) at each threshold
- **Cost inflection points** — where marginal cost of additional CFE% spikes
- **Regional variation** — CAISO vs ERCOT vs PJM vs NYISO vs NEISO cost differences
- **Storage value** — battery vs LDES at high CFE thresholds

---

## 4. Climate Target Frameworks (Research-Verified, Session 2)

### SBTi Power Sector
- **Methodology**: Sectoral Decarbonization Approach (SDA) — company-specific intensity pathways converging to ~0 gCO₂/kWh
- **Key requirement**: ~85% intensity reduction by 2035 from base year
- **Net-zero deadline**: **2040** for power sector (a decade ahead of other sectors)
- **Offsets**: Carbon offsets **cannot** count toward near-term or long-term targets
- **CCS**: Facility-level CCS may count as emissions reduction; DAC classified as BVCM (does not count)
- **New standard (draft Sep 2025)**: Expands scope to entire value chain; shifts from "renewable" to "zero-carbon" (includes nuclear); strict phaseout timelines for unabated coal/oil/gas
- **Real-world benchmark**: Orsted validated at 96% intensity reduction per kWh by 2030 (to 6 gCO₂e/kWh)

### IPCC AR6 (1.5°C pathway)
- **1.5°C, no/limited overshoot**: CO₂ must fall ~48% by 2030, ~65% by 2035, ~80% by 2040, ~99% by 2050 (vs 2019)
- **Climate Action Tracker benchmarks**: 48–80 gCO₂/kWh by 2030; <5 by 2040; 0 by 2050
- **Coal without CCS**: 67–82% reduction by 2030
- **Low-carbon electricity share**: 93–97% by 2050
- **Developed countries**: Net-zero electricity by **2035**
- **CCS**: Included in models; BECCS enables net-negative emissions. AR6 flags uncertainty around CCS deployment at scale

### IEA Net Zero by 2050 (NZE)
- **By 2030**: Power sector CO₂ declines ~60% from 2019; renewables reach 60% of generation; intensity ~140–170 gCO₂/kWh
- **By 2035**: OECD achieves 100% clean power
- **By 2040**: **Power sector reaches net-zero** — the only sector to do so a full decade before economy-wide target
- **2025 update**: Acknowledges slower near-term progress; temperature exceeds 1.5°C ~2030, peaks ~1.65°C ~2050
- **CCS**: Explicitly included; 7.6 Gt CO₂/yr captured by 2050 economy-wide

### US Government
- **Biden-era 100% clean by 2035**: Effectively dead
- **EPA power plant rules**: Proposed for full repeal (June 2025). No binding federal GHG target currently exists
- **IRA remains law**: $369B in clean energy incentives; projected to drive 60–83% reduction below 2005 levels by 2040 through economics alone
- **State-level**: CA (100% by 2045), NY (70% by 2030), MI (100% by 2040), etc. continue independently

### Other Frameworks
- **CA100+**: World's largest investor engagement ($60T+ AUM). Net-zero by 2040 globally, 2035 in advanced economies. Explicitly: avoid offsets, minimize CCUS. No CA100+ power utility with coal capacity is Paris-aligned (Carbon Tracker)
- **RE100**: 450+ companies committed to 100% renewable electricity; nuclear does NOT count; starting 2025, only facilities built within 15 years fully credited
- **GFANZ**: 675+ financial institutions; financing managed phaseout of coal power; major US banks have left under political pressure
- **24/7 CFE**: Google/Microsoft initiative for hourly matching of consumption with clean generation (directly relevant to our sister project)
- **Powering Past Coal Alliance (PPCA)**: Government coalition; OECD coal phase-out by 2030, rest by 2040

---

## 5. Scenario Analysis Framework

### Scenarios for each generator
For each of the top 15, model:

**Scenario A: Business as Usual (BAU)**
- Current retirement schedules, approved IRPs
- Announced renewable additions
- No carbon price
- Existing tax credits only

**Scenario B: Policy-Enabled Acceleration**
- Full IRA tax credit utilization (45Y, 48E, 45Q)
- Carbon price at social cost ($51/ton EPA, or $185/ton Rennert)
- Accelerated coal retirement
- Regional wholesale market access for clean energy

**Scenario C: SBTi-Aligned Pathway**
- What would it take to meet SBTi v2 intensity targets?
- Resource additions needed (from CFE optimizer cost curves)
- Cost to ratepayers or shareholders
- Timeline feasibility

**Scenario D: IEA NZE / 1.5°C-Aligned**
- Near-zero by 2035 for generation fleet
- Maximum ambition scenario
- Identifies what's physically impossible vs. merely expensive

### Key metrics per scenario
- Total CO2 emissions (MT/year)
- Emissions intensity (lbs CO2/MWh)
- $/MWh system cost
- $/ton CO2 abatement cost
- Capital investment required ($B)
- Stranded asset exposure ($B)
- Jobs transition impact

### Enabling conditions analysis
For each scenario, identify:
- **Tax credits needed**: Which IRA provisions are critical?
- **Carbon price**: What $/ton CO2 makes the scenario economic?
- **Regulatory support**: RPS, CES, EPA rules
- **Market conditions**: Gas prices, renewable costs, storage costs
- **Grid infrastructure**: Transmission, interconnection queue
- **Social/political**: Rate impacts, community transition, permitting

---

## 6. Site Architecture

### Design identity
Shared visual ecosystem with hourly-cfe-optimizer:
- Same typography, color palette, spacing
- Scrollytelling format with infographic storytelling
- Chart.js for data visualization
- Sites may be cross-linked (this site nested as link from CFE optimizer and vice versa)

### Thematic Tone (CRITICAL — applies to all content)
**Affordability and grid decarbonization must be co-optimized.** The site's narrative should consistently frame decarbonization as both ambitious AND feasible AND affordable. This is not a "cost vs. climate" framing — it's a "how do we achieve both" framing. Every analysis, chart, and narrative section should reinforce that the most durable decarbonization pathways are ones that don't sacrifice affordability or reliability.

### Page structure (Decided)

**Page 1: Executive Summary (Landing Page)**
- High-level infographic storytelling across all 15 companies
- Key insights: comparing and contrasting fleet profiles, emissions, pathways
- Overarching narrative with visual storytelling (charts, callouts, key stats)
- The "so what" for policymakers, investors, and utility executives
- Central question framed: "Is ambitious decarbonization achievable for America's largest power generators — and at what cost?"

**Page 2: Company Dashboard (Interactive)**
- Toggle/selector to switch between all 15 companies
- Per-company deep dive: fleet profile, emissions, generation mix, targets, pathway analysis
- Comparison mode: side-by-side or overlay charts for 2+ companies
- Scenario toggles: BAU, Policy-Enabled, SBTi-Aligned, IEA NZE

**Page 3: Fleet Composition & Pathway Analysis**
- **Core analytical question**: How does fleet composition determine decarbonization pathway?
  - Coal-heavy generators: Larger absolute reduction potential, lower marginal cost per ton (retiring coal is cheap CO2 abatement), but massive stranded asset exposure
  - Efficient CCGT fleets: Lower starting intensity BUT harder marginal reductions (gas is already "cleaner" — going from gas to zero-carbon is expensive)
  - Nuclear-heavy fleets: Already low-carbon, but vulnerable to age-related retirements; uprates and life extensions are cheapest clean MWh available
- Regional variation: How geography (ISO, resource availability, policy environment) shapes strategy
- Insights from CFE optimizer cost curves applied to each fleet type

**Page 4: Targets & Feasibility Assessment**
- Each company's stated targets vs. SBTi v2, IPCC, IEA NZE requirements
- "Traffic light" assessment: Green (on track), Yellow (achievable with policy support), Red (physically/economically challenging)
- The paradox: coal-heavy generators may find MORE ambitious targets MORE achievable (because coal retirement is low-cost abatement) while efficient gas+nuclear fleets face diminishing returns

**Page 5: Enabling Conditions & Policy Positions**
- What policy positions would be catalytic to least-cost decarbonization?
- Tax credits: Which IRA provisions matter most for which fleet types?
- Carbon price: What $/ton makes ambitious targets economic?
- Regional policy: RPS, CES, state mandates
- Grid infrastructure: Transmission, interconnection, market design
- Framing: "If companies want to take policy positions that accelerate least-cost decarbonization, what should those be?"

**Page 6: Methodology & Sources**
- Analytical methodology description (data sources, processing, aggregation approach)
- Inline citations throughout (not just a bibliography — sources appear next to the claims they support)
- eGRID data processing methodology
- CFE optimizer integration approach
- Climate target framework descriptions
- Scenario analysis assumptions and limitations
- Full bibliography with links

### Hosting & Visibility (Decided)
- **Repo**: Private on GitHub (not searchable)
- **Site**: Public via Netlify or Cloudflare Pages (free tier, connects to private repo)
- Custom domain can be added later
- This keeps research/analysis private while the finished site is publicly accessible

### Navigation
Consistent with CFE optimizer site — top nav bar, responsive, mobile-compatible.
Cross-link to hourly CFE optimizer site.
Pages: Executive Summary | Company Dashboard | Fleet Analysis | Targets & Feasibility | Policy & Enabling Conditions | Methodology

### Key Analytical Insight to Develop
**The fleet composition paradox:**
- Coal-heavy generators (Evergy, AEP, BHE/PacifiCorp) have the HIGHEST current emissions but potentially the EASIEST path to large absolute reductions — because replacing coal with renewables+storage is now the cheapest new generation option.
- Efficient CCGT+nuclear fleets (Constellation, Entergy, NextEra) have LOWER current emissions but face HARDER marginal reductions — because going from efficient gas to zero-carbon requires expensive clean firm, CCS, or massive overbuild of renewables+storage.
- This paradox has major implications for target-setting: SBTi intensity targets may be easier for already-clean fleets but represent less absolute climate benefit; absolute reduction targets may be more meaningful but harder for efficient fleets.
- **The question**: Given real fleet physics, what does an "ambitious but affordable" decarbonization pathway actually look like for each archetype?

---

## 7. MJ Bradley / ERM Benchmarking Report

(Findings to be populated from research agent)

Key aspects:
- Annual ranking of ~100 US power producers by CO2, NOx, SO2, mercury
- Uses EPA data (CEMS, eGRID)
- Tracks intensity and absolute emissions trends
- Note: Constellation acquired Calpine in Jan 2026 — changes the rankings significantly

---

## 8. Key Insights from CFE Optimizer

### Three-Strategy Framework (from optimizer results)
1. **Renewable-First (75-85% CFE)**: Solar + wind + 4hr battery. Cost premium $10-25/MWh. Best for coal-heavy generators.
2. **Nuclear-Plus (85-95% CFE)**: Renewables + nuclear life-extension + CCS on select gas. Cost premium $25-50/MWh. Best for diversified fleets.
3. **Regional Portfolio (95-100% CFE)**: Full stack including LDES + DAC. Cost premium $50-78/MWh. The expensive last mile.

### Key cost findings
- Cost inflection zone: 92.5-100% CFE matching is where costs escalate sharply ("hockey stick")
- Going from 99% to 100% costs an additional $18-31/MWh for 1% more clean energy
- Storage accounts for <10% of total system cost even at 100% — clean firm is the bottleneck
- Regional variation: 2-3x cost difference between ERCOT (cheapest) and ISO-NE (most expensive)
- At 100% CFE, clean firm (nuclear, CCS-CCGT) dominates the resource mix

### SBTi & CDP Landscape (across top 15) — Corrected per research (Session 2)
- **SBTi 1.5°C validated**: AES only (validated Nov 2025) — 1 company
- **SBTi committed (pending validation)**: NextEra — 1 company
- **Aligned but not formally SBTi committed**: Xcel, AEP, WEC (CA100+ aligned) — 3 companies
- **No formal SBTi commitment**: Constellation, Duke, Southern, Vistra, Dominion, BHE, Entergy, Evergy, DTE, PPL — 10 companies
- **CDP A- or higher (active or historical)**: AES (A-, active), Xcel (A-, active), Southern (A-, 2021), Dominion (A-, 2019)
- **CDP B–C range (active)**: Vistra (C, 2024), Entergy (B-, 2023)
- **CDP: Not publicly reporting (2024)**: Constellation, NextEra, Duke, AEP, Evergy, DTE, WEC, PPL — 8 companies
- **CDP F (non-disclosure)**: BH Energy
- **Key finding — 2024 CDP Exodus**: 8 of 15 top generators stopped public CDP reporting in 2024, a major transparency regression
- **Key pattern**: Only 1 of 15 has SBTi-validated targets despite universal net-zero pledges — the gap between pledges and formal accountability is wide

---

## 9. Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-15 | Include Constellation + Calpine as combined entity | Acquisition completed Jan 2026 |
| 2026-02-15 | eGRID 2023 as primary emissions data source | Most recent comprehensive plant-level US data |
| 2026-02-15 | Shared visual identity with CFE optimizer | Sites part of same research ecosystem, may be cross-linked |
| 2026-02-15 | Scrollytelling format for site | Consistent with sister project, engaging for business audience |
| 2026-02-15 | 5-page site structure (Exec Summary, Dashboard, Fleet Analysis, Targets, Policy) | User direction |
| 2026-02-15 | "Affordability + grid decarb co-optimized" as thematic tone | User direction — NOT "cost vs climate" but "how to achieve both" |
| 2026-02-15 | Fleet composition paradox as core analytical thread | Coal-heavy = easier absolute reductions; efficient fleets = harder marginal gains |
| 2026-02-15 | Policy positions for catalytic least-cost decarb | What should companies advocate for given their fleet? |
| 2026-02-15 | Company dashboard with toggle between all 15 | Interactive comparison tool |
| 2026-02-15 | Sites cross-linked as same research ecosystem | New site may be nested as link from CFE optimizer |
| 2026-02-15 | Methodology page with inline sources & citations | Not just bibliography — sources appear next to claims |
| 2026-02-15 | Broader grid decarb research sweep | Don't rely solely on optimizer results — include NREL, Princeton, Rhodium, BNEF, RMI, etc. |
