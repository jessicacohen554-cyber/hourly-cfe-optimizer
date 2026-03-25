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
