# Data Center Clean Energy Portfolio Analysis

Self-contained package for the VRE Investment Thesis deck and supporting analysis. Estimating how much clean energy hyperscalers and major data center operators have under contract, by technology type, geography, and project status — and whether their contracted pipelines are sufficient to meet projected demand growth.

## Quick Start

```bash
cd data-center-cfe/
pip install -r requirements.txt

# Generate the 14-slide HTML deck from CSV data
python generate_deck.py          # → output/vre-investment-thesis-deck.html

# Generate standalone PNG figures
python generate_figures.py       # → output/figures/*.png

# (Optional) Convert reference article to PDF (requires weasyprint)
pip install weasyprint
python convert_to_pdf.py         # → output/ref-vre-investment-thesis.pdf
```

## Updating Data

Replace CSVs in `data-inputs/` with new data (same column schema), then re-run:
- `python generate_deck.py` — regenerates the HTML deck
- `python generate_figures.py` — regenerates all PNG figures

Key data files that drive the deck:
- `data-inputs/us_hyperscaler_gap_analysis.csv` — Slide 4 gap table + waterfall
- `data-inputs/capture_rates_by_iso.csv` — Slide 5 capture rate chart
- `data-inputs/cfe_premium.csv` — Slide 6 CFE premium chart
- `data-inputs/vre_investment_thesis.csv` — Slide 9 regional matrix
- `data-inputs/demand_forecasts.csv` — Slide 2 demand chart

## Directory Structure

```
data-center-cfe/
├── generate_deck.py           # CSV → HTML deck
├── generate_figures.py        # CSV → PNG figures
├── convert_to_pdf.py          # Reference article → PDF
├── requirements.txt           # Python dependencies
├── templates/
│   └── deck_template.html     # Jinja2 template for the deck
├── styles/                    # Shared CSS (for reference article)
├── js/                        # Shared JS (for reference article)
├── data-inputs/               # Source CSVs (replace these to update)
├── data/                      # Processed data
├── analysis/                  # Analysis scripts
├── research/                  # Reference article + docs
├── output/                    # Generated artifacts
│   ├── vre-investment-thesis-deck.html
│   └── figures/               # PNG figures
├── CEG-style.css              # Constellation branding reference
├── sources.md                 # Bibliography
└── README.md                  # This file
```

---

## Key Findings (as of March 2026)

### Portfolio Summary (De-duplicated)

| Company | Total Contracted (GW) | Operational (GW) | Under Construction (GW) | Planned/Framework (GW) | Nuclear (GW) |
|---------|----------------------|-------------------|-------------------------|------------------------|---------------|
| Amazon | ~40+ | ~20+ | ~10+ | ~10+ (5 GW SMR framework) | 1.9 existing + 5.0 SMR framework |
| Microsoft | ~40 | ~19 | ~10+ | ~11+ (10.5 GW Brookfield framework) | 0.8 existing + 0.05 fusion |
| Google | ~22+ | ~10+ | ~5+ | ~7+ (2.3 GW nuclear) | 2.3 SMR + 0.4 CCS |
| Meta | ~29 | ~17+ | ~5+ | ~7+ (6.6 GW nuclear) | 3.7 existing + 2.9 SMR |
| Switch | ~12 | minimal | — | 12 GW (non-binding Oklo) | 12.0 SMR (non-binding) |
| Oracle | ~1 | — | — | 1 GW (speculative) | 1.0 SMR (speculative) |
| Equinix | ~1.2 | ~1+ | <0.2 | <0.2 | — |
| CoreWeave | undisclosed | — | — | — | — |
| **Total** | **~145+ GW** | **~67+ GW** | **~30+ GW** | **~48+ GW** | **~27+ GW** |

### Critical Context

- **Big 4 = 49% of all global corporate clean energy deals in 2025** (BNEF)
- Global corporate PPA volume was 55.9 GW in 2025 — first decline in nearly a decade
- All four Big 4 companies claim 100% renewable electricity match (annual basis)
- Nuclear deals are accelerating: ~27 GW across all companies, mostly SMR (unproven at scale)
- SMR timeline risk is high: only NuScale has NRC design certification; most deployments target 2030+

### Demand Gap

Data center electricity demand is projected to reach 1,000-1,600 TWh globally by 2030 (up from ~536-860 TWh in 2025). At ~25% capacity factor for renewables, serving 1,000 TWh requires ~450 GW of nameplate capacity. The ~145 GW currently contracted across all companies leaves a significant gap — especially given that:

1. Much contracted capacity is framework/planned (not yet built)
2. SMR timelines are highly uncertain (most target 2030+, no commercial SMR operating yet)
3. Demand growth forecasts vary by 4-5x depending on AI adoption rates
4. Not all contracted capacity serves data centers specifically (some serves corporate offices/operations)

## Data Files

### `data/hyperscaler_ppa_deals.csv`
Deal-level dataset with 35+ rows covering named deals and aggregate portfolio entries for 8 companies. Each row includes:
- Company, counterparty, technology, capacity (MW)
- Status: `operational`, `under_construction`, `planned`, `framework`
- Geography (country, state, ISO market)
- Source citation with URL
- De-duplication notes (critical for avoiding double-counting)

### `data/demand_forecasts.csv`
Demand projections from 8 sources (Goldman Sachs, S&P/451, LBNL, IEA, McKinsey, Deloitte, BCG, Rystad) covering global and US-specific forecasts through 2035.

### `sources.md`
Full bibliography with 40+ sources, access notes (paywalled vs. public), and quality ratings.

### `analysis/gap_analysis.py`
Aggregation and gap analysis script. Loads the CSV, computes portfolio summaries by company/technology/status, and compares against demand forecasts.

## De-Duplication Methodology

Corporate clean energy announcements are notorious for double-counting. This dataset handles it by:

1. **Framework vs. named deals**: Framework agreements (e.g., Amazon's 5 GW X-energy target) are listed but flagged as `framework`. Named deals within them (e.g., Energy Northwest 960 MW) are listed separately with notes that they're INCLUDED in the framework total. Aggregation uses the larger number, not the sum.

2. **Existing plants vs. new PPAs**: When a company signs a PPA for an existing operational plant (e.g., Meta/Vistra Perry+Davis-Besse), the plant status is `operational` but the PPA start date may be future. The capacity exists on the grid today — the PPA is a financial arrangement, not new capacity.

3. **Aggregate residuals**: For portfolios with 100+ projects (Amazon 700+, Google 170+, Meta 128+), we list named deals individually and compute a residual aggregate row. The residual = company-reported total minus sum of named deals.

4. **Cross-company overlap on SMR vendors**: Oklo has deals with both Meta (1.2 GW) and Switch (12 GW). These are separate customer commitments, not the same capacity — but Oklo's ability to deliver is the constraint. Each is listed under its respective company.

## Known Gaps & Future Work

- [ ] Paywalled sources (BNEF, S&P, Bird & Bird) — would provide deal-level granularity for aggregate rows
- [ ] Digital Realty — sustainability report not yet scraped
- [ ] Apple — significant clean energy buyer, not yet included
- [ ] Geographic mapping to ISO regions (partially done for US deals)
- [ ] Capacity factor estimates to convert nameplate MW to expected GWh
- [ ] Company-specific electricity consumption data to compute true coverage ratios
- [ ] Forward demand projections per company (capex guidance, data center pipeline announcements)
