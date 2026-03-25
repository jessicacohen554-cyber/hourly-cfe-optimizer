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
