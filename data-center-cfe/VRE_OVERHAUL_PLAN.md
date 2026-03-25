# VRE Portfolio Analysis Overhaul — Complete Plan & Session Prompts

## Overview

**Goal**: Transform the existing 14-slide "VRE + Storage Investment Thesis" deck into a **VRE-only investment case** focused on whether a standalone VRE development portfolio captures sufficient value from the data center clean energy gap — especially if you can develop storage yourself later.

**Core thesis shift**: "VRE+Storage hybrid is the product" → "VRE development captures the clean energy gap; storage optionality comes free with interconnection position"

**Deliverable**: Updated HTML deck + new analytical work (Python scripts to compute Meta+Amazon combined consequential queue depletion)

---

## Final Slide Order (15 slides)

| New # | Slide Title | Action | Old # |
|-------|------------|--------|-------|
| 1 | **VRE Development: Capturing the Clean Energy Gap** | MODIFY title | 1 |
| 2 | **Where the Load Is Landing** (by ISO + CoreWeave) | MODIFY | 3 |
| 3 | **The Clean Energy Gap** (2030 waterfall) | KEEP as-is | 4 |
| 4 | **The Gap Persists: 2035–2040 Even with SMRs** | MERGE 4B+4C | 4B, 4C |
| 5 | **The Clean Energy Gap: 2050 Outlook** | KEEP as-is | 4D |
| 6 | **Clean Firm Won't Arrive in Time** | KEEP as-is | 8 |
| 7 | **The Consequential Advantage: Coal Wall in MISO/SPP** | **NEW** | — |
| 8 | **VRE Development Timeline Advantage** | **NEW** | — |
| 9 | **Regional VRE Opportunity Matrix** | MODIFY (strip storage) | 9 |
| 10 | **The VRE Investment Case** | MODIFY (VRE-only IRR) | 10 |
| 11 | **Procurement Strategies Overview** | SPLIT 1/3 from old 11 | 11 |
| 12 | **Participation → VRE Pipeline Value** | SPLIT 2/3 from old 11 | 11 |
| 13 | **Regional Portfolio Impact by Strategy** | SPLIT 3/3 from old 11 | 11 |
| 14 | **VRE Pipeline Value by Scenario & ISO** | MODIFY (no jar SVG) | 12 |
| 15 | **Appendix: Methodology & Sources** | KEEP (renumber) | 14 |

**Deleted slides**: 2 (Demand Supercycle), 5 (LMP Cannibalization), 6 (Why Hyperscalers Will Pay More), 7 (Hybrid Economics), 13 (Appendix A: Optimizer)

---

## User Decisions (Captured)

- **Target IRR**: >8% levered for standalone VRE
- **Coal Wall slide**: Combine Meta (134 TWh 2030) + Amazon (164 TWh 2030) competing together for the consequential queue in MISO/SPP/PJM. Calculate fossil-average baseline in regions they plan to build, offset by moving down the consequential queue to displace coal. Show how much LESS TWh they need to buy.
- **CoreWeave**: Use public estimates (~8 GW total pipeline, ~28 sites, NVIDIA-backed GPU cloud)
- **Scope**: Both HTML changes AND new analytical work (Python scripts for consequential queue depletion)

---

## Data Sources Reference

| Data | File | Key Fields |
|------|------|------------|
| Hyperscaler demand 2030 | `data-center-cfe/data-inputs/us_hyperscaler_gap_analysis.csv` | Meta 134 TWh, Amazon 164 TWh, Big 4 596 TWh |
| DC load by ISO | `data-center-cfe/data-inputs/dc_load_by_iso.csv` | PJM ~125 GW, ERCOT ~40 GW, MISO ~24 GW, SPP ~44 GW, NYISO ~18 GW, CAISO ~13 GW, NEISO ~1.2 GW |
| VRE investment thesis | `data-center-cfe/data-inputs/vre_investment_thesis.csv` | Per-ISO VRE ratings, capture rates, PPA pricing |
| Accounting scenarios | `data-center-cfe/data-inputs/accounting_regime_scenarios.csv` | 4 scenarios with VRE/storage/firm demand splits |
| Extended gap analysis | `data-center-cfe/data-inputs/us_hyperscaler_gap_analysis_extended.csv` | 2030/2035/2040/2050 projections with L/M/H procurement |
| Strategy 1B results | `data/step5-scenarios/strategy1_consequential.json` | Per-ISO consequential cost/TWh at participation levels |
| Coal retirement model | `scripts/pipeline_config.py` lines 598-643 | COAL_PHASE_OUT_START=50%, END=85%, announced retirements by ISO/year |
| Consequential queue | `data/step3-dispatch/mac_queue/consequential_queue.json` | Queue entries with iso, threshold, MAC, co2_displaced, delta_resources |
| Fossil mix by ISO | `scripts/lmp_engine.py` lines 182-199 | MISO: 35% coal, SPP: 30% coal, PJM coal declining |
| Contracted clean TWh | `scripts/procurement_utils.py` lines 357-368 | PJM 18.8 TWh, MISO 18 TWh from nuclear PPAs |

---

## Session Prompts

Each prompt below is self-contained and can be copy/pasted into a new Claude session. They reference specific files, line numbers, and data. Run them in order — later prompts assume earlier ones are complete.

---

### Prompt 0: Setup — Delete Slides & Update Title

```
## Task: VRE Deck Overhaul — Deletions & Title Update

You are editing a 14-slide HTML investment thesis deck at:
`data-center-cfe/output/vre-investment-thesis-deck.html`

This is a single self-contained HTML file (all CSS inline in <style>, all content in <section class="slide"> blocks). The deck is being transformed from "VRE + Storage" to "VRE-only" investment thesis.

### Step 1: Delete these 5 slides entirely (remove the full <section class="slide">...</section> block for each):

1. **Slide 2 "The Demand Supercycle"** — starts at line ~536, `<h1>The Demand Supercycle</h1>`
2. **Slide 5 "LMP Cannibalization"** — starts at line ~996, `<h1>LMP Cannibalization: The Standalone VRE Trap</h1>`
3. **Slide 6 "Why Hyperscalers Will Pay More"** — starts at line ~1071, `<h1>The 24/7 CFE Premium: Why Hyperscalers Will Pay More</h1>`
4. **Slide 7 "Hybrid Economics"** — starts at line ~1134, `<h1>Hybrid Economics: Storage Changes Everything</h1>`
5. **Slide 13 "Appendix A"** — starts at line ~1851, `<h1>Appendix A: Constellation Hourly CFE Optimizer</h1>`

Also delete any HTML comment blocks above each slide (the <!-- ═══ SLIDE N: ... ═══ --> separators).

### Step 2: Update the title slide (slide 1):

Change:
- Title: `VRE + Storage:<br><strong>The Bridge to Clean Baseload</strong>` → `VRE Development:<br><strong>Capturing the Clean Energy Gap</strong>`
- Subtitle: Change to "Why a variable renewable energy development portfolio is the highest-conviction near-term investment in the data center power transition — and why you don't need storage to start"
- The `<title>` tag in `<head>`: Change to "VRE Development Investment Thesis"

### Step 3: Renumber all remaining slides

After deletions, renumber the <span class="slide-number">N</span> tags sequentially 1–10 (we'll add new slides later that bring it to 15).

Current mapping after deletions:
- Old 1 (title) → New 1
- Old 3 (Where Load Landing) → New 2
- Old 4 (Clean Energy Gap 2030) → New 3
- Old 4B (2035) → New 4 (will be merged with 4C in next prompt)
- Old 4C (2040) → stays temporarily as New 5
- Old 4D (2050) → New 6
- Old 8 (Clean Firm) → New 7
- Old 9 (Regional Matrix) → New 8
- Old 10 (Investment Case) → New 9
- Old 11 (Procurement) → New 10
- Old 12 (Cross-Regional) → New 11
- Old 14 (Appendix B) → New 12

Also update the HTML comment separators to match new numbering.

### Verification:
- Count exactly 12 <section class="slide"> blocks remain (title + 11 content slides)
- No references to "VRE + Storage" in the title slide
- All slide numbers sequential 1-12
- File still renders correctly (valid HTML)

Commit with message: "Strip storage focus: delete 5 slides, update title to VRE-only thesis"
```

---

### Prompt 1: Slide 2 — Where the Load Is Landing (ISO Breakdown + CoreWeave)

```
## Task: Modify Slide 2 — ISO-based Pipeline Breakdown + Add CoreWeave

Edit `data-center-cfe/output/vre-investment-thesis-deck.html`, the slide titled "Where the Load Is Landing" (currently slide number 2 after prior deletions).

### Change 1: Replace state-based left panel with ISO-based breakdown

The left panel currently shows "Top 10 States by Pipeline (GW)" with stacked bars for Texas, Virginia, Georgia, etc.

Replace with "Data Center Pipeline by ISO (GW)" using this data:

| ISO | Live | U/C | Committed | Early Stage | Total |
|-----|------|-----|-----------|-------------|-------|
| PJM | 8.9 | 8.6 | 29.5 | 78.0 | 125.0 |
| SPP | 1.0 | 1.0 | 2.0 | 40.0 | 44.0 |
| ERCOT | 3.4 | 4.1 | 6.5 | 26.0 | 40.0 |
| MISO | 3.5 | 3.3 | 6.0 | 11.5 | 24.3 |
| NYISO | 1.8 | 0.9 | 3.0 | 12.0 | 17.7 |
| CAISO | 3.0 | 3.0 | 2.0 | 5.0 | 13.0 |
| NEISO | 0.5 | 0.2 | 0.2 | 0.3 | 1.2 |

Use the same stacked bar visual format. Sort by total GW descending.

Update subtitle to reference ISOs instead of states.

### Change 2: Add CoreWeave to the hyperscaler table (right panel)

Add a row for CoreWeave after Google:
| CoreWeave | 28 | 1.2 | 0.8 | 2.0 | 4.0 | 8.0 |

CoreWeave: NVIDIA-backed GPU cloud, rapid growth, IPO 2025.

Update totals row. Adjust "150 additional operators" residual.

### Change 3: Update source note

Add: "CoreWeave: S-1 filing (Mar 2025), press releases, DC Byte tracking." and "ISO aggregations: Constellation analysis of DC Byte microdata mapped to ISO footprints."

Commit with message: "Slide 2: ISO-based pipeline breakdown, add CoreWeave to hyperscaler table"
```

---

### Prompt 2: Slide 4 — Merge 2035+2040 with SMR Timeline

```
## Task: Merge Slides 4B (2035) and 4C (2040) into "The Gap Persists: 2035–2040 Even with SMRs"

Edit `data-center-cfe/output/vre-investment-thesis-deck.html`.

After Prompt 0 deletions, there are two consecutive slides:
- "The Clean Energy Gap: 2035 Outlook"
- "The Clean Energy Gap: 2040 Outlook"

### Merge into a single slide:

**Title**: "The Gap Persists: 2035–2040 Even with SMRs"

**Layout**: Two-column (left-right grid)
- Left: 2035 waterfall (condensed from current 4B)
- Right: 2040 waterfall (condensed from current 4C)

**Key addition — SMR timeline bar below waterfalls**:

Use the existing .timeline CSS classes. Show:
- 2025: NuScale FOAK certified (only SMR with NRC design cert)
- 2030: First SMR site prep (optimistic)
- 2033-2035: First SMR commercial operation (GREEN = optimistic)
- 2038-2040: Meaningful SMR fleet 5-10 GW (ORANGE = realistic)
- Text: "Even at aggressive timelines, SMRs add <50 TWh by 2035 and <150 TWh by 2040"

**Callout box** at bottom:
"VRE fills the 2025–2035 bridge AND remains essential alongside SMRs post-2035. The 400+ TWh gap in 2035 cannot wait for nuclear timelines. Even by 2040 with an optimistic SMR fleet, 260–480 TWh of gap remains — VRE development today captures interconnection positions and offtake relationships that persist regardless of SMR deployment."

**Data** from `data-center-cfe/data-inputs/us_hyperscaler_gap_analysis_extended.csv`:
- 2035: 789 TWh need, gap 309–509 TWh (L/M/H procurement)
- 2040: 1042 TWh need, gap 262–662 TWh

Delete the standalone 2040 slide section after merging.

Renumber slides (one fewer slide now).

Commit with message: "Merge 2035+2040 gap slides with SMR timeline overlay"
```

---

### Prompt 3: NEW Slide 7 — The Consequential Advantage: Coal Wall (Analysis + HTML)

```
## Task: Create New Slide — "The Consequential Advantage: Coal Wall in MISO/SPP"

This prompt requires BOTH analytical computation AND HTML slide creation.

### Part A: Analytical Computation

Write a Python script `data-center-cfe/analysis/coal_wall_analysis.py` that computes:

**Inputs:**
- Meta 2030 demand: 134 TWh (from `data-center-cfe/data-inputs/us_hyperscaler_gap_analysis.csv`)
- Amazon 2030 demand: 164 TWh (same source)
- Combined: 298 TWh
- Both companies use consequential accounting (Strategy 1B, fossil-average baseline)
- Both compete for the same consequential queue in MISO, SPP, and PJM

**Data to load:**
1. `data/step3-dispatch/mac_queue/consequential_queue.json` — the ordered queue of clean energy steps by MAC ($/tCO2). Each entry has: iso, threshold_start, threshold_end, marginal_mac, co2_displaced_mt, delta_resources (TWh by resource type)
2. `scripts/pipeline_config.py` — coal retirement constants:
   - ANNOUNCED_COAL_RETIREMENTS: MISO {2025: 2 GW, 2028: 6 GW, 2030: 10 GW, 2035: 18 GW}, SPP {2025: 1 GW, 2028: 3 GW, 2030: 5 GW, 2035: 9 GW}
   - Fossil mix: MISO 35% coal, SPP 30% coal (from `scripts/lmp_engine.py` FOSSIL_MIX)
3. `scripts/procurement_utils.py` — CONTRACTED_CLEAN_TWH: PJM 18.8 TWh, MISO 18 TWh

**Computation:**
1. Calculate the fossil-average emission rate for each ISO where Meta/Amazon plan to build:
   - MISO: weighted average of coal (1.0 tCO2/MWh × 35%) + gas CCGT (0.4 × 40%) + gas CT (0.55 × 24%) + oil (0.65 × 1%) = ~0.642 tCO2/MWh
   - SPP: coal (1.0 × 30%) + gas CCGT (0.4 × 42%) + gas CT (0.55 × 27%) + oil (0.65 × 1%) = ~0.623 tCO2/MWh
   - PJM: use the lmp_engine.py FOSSIL_MIX for PJM

2. Calculate Meta+Amazon combined CO2 baseline: their demand × fossil-average rate in their respective ISOs. This is what they need to offset.

3. Walk the consequential queue (sorted by MAC ascending):
   - For each queue step, the displaced CO2 is the coal/fossil being retired
   - Coal displacement is MUCH higher CO2 per MWh than the fossil-average baseline (~1.0 vs ~0.6 tCO2/MWh)
   - This means each TWh of VRE that displaces coal offsets ~1.0 tCO2, but they only NEED to offset ~0.6 tCO2 per TWh of their demand
   - **Result: they need FEWER TWh of VRE procurement to achieve the same CO2 reduction**

4. Compute:
   - How many TWh of VRE do Meta+Amazon need under consequential (displacing coal first) vs. hourly matching (must match 1:1)?
   - The "savings" in TWh — how much LESS VRE they buy
   - At what participation level does the cheap coal displacement pool exhaust?
   - When does the pool "run out" (year) given announced coal retirements?

5. Output results as JSON: `data-center-cfe/data-inputs/coal_wall_analysis_results.json`

**Expected findings (approximate):**
- Under consequential: Meta+Amazon might need ~180-220 TWh of VRE (displacing high-emitting coal) instead of 298 TWh (1:1 hourly)
- Savings: ~80-120 TWh LESS procurement needed
- MISO cheap pool ($49/MWh) lasts through ~30% C&I participation
- Coal retirements shrink the pool: MISO loses ~18 GW coal by 2035, reducing displacement opportunity
- Window: the coal wall advantage is strongest 2025-2032, diminishes as coal retires

### Part B: HTML Slide Creation

Insert a new slide after slide 6 (Clean Firm Won't Arrive in Time) in the deck HTML.

**Title**: "The Consequential Advantage: Leveraging the Coal Wall"
**Slide number**: 7

**Layout**:
- Top: 3 KPI cards
  - Card 1 (green): "[X] TWh saved" / "Less VRE procurement needed vs. hourly matching"
  - Card 2 (blue): "$49/MWh" / "Consequential cost (flat to 30% participation)"
  - Card 3 (orange): "2032" / "Window closes as coal retires"

- Middle: Two-column
  - Left: "How the Coal Wall Works" explanation
    - When Meta/Amazon buy VRE that displaces COAL (1.0 tCO2/MWh), each TWh offsets more CO2 than their fossil-average baseline (~0.6 tCO2/MWh)
    - Result: fewer TWh needed to hit the same emissions reduction target
    - This is Strategy 1B (fossil-average baseline) from the Constellation Hourly CFE Optimizer
  - Right: Table showing the queue walk
    - Columns: Participation %, MISO Coal Displaced (TWh), SPP Coal Displaced (TWh), VRE Procured (TWh), CO2 Offset (Mt), Cost ($/MWh)
    - Rows: 5%, 10%, 20%, 30%, 40%, 50%
    - Highlight the row where the pool exhausts

- Bottom: Callout
  "Meta and Amazon's combined 298 TWh demand (2030) can be offset with ~[X] TWh of consequential VRE procurement — [Y]% less than hourly matching requires. But this advantage shrinks as coal retires: MISO loses 18 GW of coal by 2035. Early movers capture the deepest displacement value."

Use the deck's existing CSS classes (kpi-row, kpi-card, left-right, data-table, callout, source-note).

Fill in [X], [Y] values from the analysis script output.

Commit with message: "New slide 7: Consequential coal wall analysis — Meta+Amazon combined queue depletion"
```

---

### Prompt 4: NEW Slide 8 — VRE Development Timeline Advantage

```
## Task: Create New Slide — "VRE Development Timeline Advantage"

Insert a new slide after the Coal Wall slide (new slide 7) in `data-center-cfe/output/vre-investment-thesis-deck.html`.

**Title**: "VRE: The Only Technology That Can Deploy at Scale Now"
**Slide number**: 8

This slide makes the case that VRE development captures value even WITHOUT storage because:
1. VRE deploys in 2-3 years; nuclear/SMR takes 8-15 years
2. IRA PTC/ITC applies to standalone VRE (no storage required)
3. Corporate PPA market is deep and growing
4. Interconnection position is the scarce asset — storage can be added later

**Layout**: Two-column

**Left column — "Development Timeline Comparison"**:
A visual timeline comparison (use the existing .timeline CSS):

| Technology | Permit→COD | First GW available |
|-----------|-----------|-------------------|
| Solar (utility) | 2-3 years | Now (2025) |
| Onshore Wind | 2-4 years | Now (2025) |
| Battery (standalone) | 1-2 years | Now (2025) |
| Offshore Wind | 5-8 years | 2028-2030 |
| SMR Nuclear | 8-12 years | 2033-2038 |
| Large Nuclear | 10-15 years | 2035+ |
| Enhanced Geothermal | 5-8 years | 2030+ |

Show as horizontal bars with color coding: green (available now), blue (near-term), orange (long-term).

Below the timeline, add:
"VRE captures the interconnection queue position. Storage can be co-located later — the land, permits, and grid connection are the scarce assets, not the panels or turbines."

**Right column — "Standalone VRE Economics"**:

Table: "IRA Incentives — No Storage Required"
| Incentive | Value | Applies to standalone VRE? |
|-----------|-------|--------------------------|
| PTC (wind) | $26/MWh (10 yr) | Yes |
| ITC (solar) | 30% of capex | Yes |
| Domestic content bonus | +10% ITC / +$2.6/MWh PTC | Yes |
| Energy community bonus | +10% ITC / +$2.6/MWh PTC | Yes |
| Low-income bonus | +10-20% ITC | Yes (solar) |

Below table, add callout (green):
"Corporate PPA market: Big 4 hyperscalers executed 41.6 GW of clean energy deals through 2025. The buyer pool is deep and growing — $35-50/MWh premiums over wholesale for VRE PPAs. Storage pairing is valued but NOT required for offtake."

Source: hyperscaler_ppa_deals.csv shows 41.6 GW procured_gw_nameplate for Big_4_Total.

**Source note**: "Development timelines: LBNL Queued Up 2025, NRC licensing data, DOE Pathways to Commercial Liftoff (Nuclear, 2023). IRA incentives: Inflation Reduction Act §§45, 48 as amended; Treasury final rules (2024). PPA market: BNEF 1H 2026 Corporate Energy Market Outlook; LevelTen Q1 2026 PPA Price Index."

Commit with message: "New slide 8: VRE development timeline advantage — deploy now, add storage later"
```

---

### Prompt 5: Slide 9 — Regional VRE Opportunity Matrix (Strip Storage)

```
## Task: Modify the Regional Opportunity Matrix — VRE-Only Focus

Edit `data-center-cfe/output/vre-investment-thesis-deck.html`, the slide titled "Regional Opportunity Matrix".

### Changes to the heatmap table:

1. **Delete column**: "Storage Arb Value" — remove the entire column (header + all 7 ISO rows)

2. **Rename column**: "VRE+S Rating" → "VRE Rating"

3. **Update ratings** using VRE-only data from `data-center-cfe/data-inputs/vre_investment_thesis.csv` (use the non-Storage rows):
   - PJM: Strong Buy (Solar VRE-only = Strong Buy, Wind = Buy)
   - ERCOT: Buy (Solar standalone = "Buy with Storage" → downgrade to "Buy" since we're VRE-only; Wind = Hold)
   - MISO: Buy (Solar = Buy, Wind = Hold) — note: conditional on consequential accounting survival
   - CAISO: Hold (Solar standalone = "Sell standalone" per the CSV; adjusted to "Hold" acknowledging VRE-only limitations)
   - NYISO: Buy (Solar = Buy, premium market)
   - NEISO: Buy — small scale (Solar = Buy small scale)
   - SPP: Speculative Buy (Solar = Speculative Buy)

4. **Update "Strategic Play" column** — remove all storage references:
   - PJM: "Largest DC market. 30 GW growth by 2030. Capacity market pays $666/MW-day. PPA prices high — VRE captures corporate premium."
   - ERCOT: "67 GW DC pipeline. Cheapest VRE in US. Consequential + behind-meter demand. Solar capture declining but PPA premiums offset."
   - MISO: "Meta/Amazon primary consequential procurement market. Deep cheap wind pool ($49/MWh). Coal wall advantage for early movers."
   - CAISO: "Solar standalone challenging (<30% capture). Wind + geothermal more viable. Google HQ demand supports premium pricing."
   - NYISO: "Highest PPA prices in US. Zero cannibalization. Premium but constrained market."
   - NEISO: "Premium pricing, minimal cannibalization. Small scale only. Highest commercial rates."
   - SPP: "Cheapest wind in US. Consequential-only play. Limited local DC presence."

5. **Update tier callouts** at bottom:
   - Tier 1 (green): "PJM + ERCOT: Highest conviction. 165 GW combined DC pipeline. Corporate PPA demand guarantees offtake under any accounting regime."
   - Tier 2 (blue): "MISO + NYISO: Strong fundamentals. MISO = consequential buyer magnet (coal wall). NYISO = premium, zero cannibalization."
   - Tier 3 (orange): "CAISO/SPP/NEISO: Selective only. CAISO solar needs pairing (future storage add). SPP is speculative. NEISO is small but premium."

6. **Remove the "Solar Capture" column references to "Mitigated by storage"** — change to actual capture rate language:
   - PJM: "capture ~95% → ~70% by 2030"
   - ERCOT: "capture ~70% → ~45% by 2030"
   - MISO: "capture ~98% → ~70% by 2030"
   - etc.

Commit with message: "Slide 9: Strip storage from regional opportunity matrix, VRE-only ratings"
```

---
