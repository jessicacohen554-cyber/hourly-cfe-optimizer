# CEG IPP Climate Transition Report Package

Standalone, portable package for generating Constellation Energy-branded IPP Climate Transition reports. Produces a 14-slide HTML deck, a scrollable HTML report, and 12 publication-quality PNG figures.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (First time only) Export data from the main repo
#    Run from the hourly-cfe-optimizer repo root:
python CEG-IPP-target/export_data.py

# 3. Generate figures (must run before deck/report)
cd CEG-IPP-target/
python generate_figures.py

# 4. Generate the slide deck (single portable HTML)
python generate_deck.py

# 5. Generate the standalone report
python generate_report.py
```

## Output Files

| File | Description |
|------|-------------|
| `output/figures/*.png` | 12 publication-quality PNG figures |
| `output/ceg-ipp-transition-deck.html` | 14-slide HTML deck (single file, all images embedded as base64) |
| `output/ceg-ipp-transition-report.html` | Scrollable HTML report (references `figures/` directory) |

## Updating Data

To refresh with new data:

1. Update the CSVs in `data/` (or re-run `export_data.py` from the main repo)
2. Re-run `python generate_figures.py`
3. Re-run `python generate_deck.py` and/or `python generate_report.py`

The CSVs are the single source of truth — all analysis reads from them.

## File Structure

```
CEG-IPP-target/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── data/                        # Input data (CSV)
│   ├── ipp_fleet_config.csv     # One row per plant
│   ├── ipp_fleet_companies.csv  # One row per company (7 rows)
│   ├── ipp_sweep_results.csv    # Scenario sweep results
│   ├── pipeline_constants.csv   # Key model constants
│   └── README.md                # Data dictionary
├── export_data.py               # One-time: convert JSON/parquet → CSVs
├── ipp_analysis.py              # Shared analysis (spider scores, aggregation)
├── generate_figures.py          # matplotlib → PNG
├── generate_deck.py             # Jinja2 → HTML slide deck
├── generate_report.py           # Jinja2 → HTML report
├── templates/
│   ├── deck_template.html       # 14in × 7.5in Jinja2 slide template
│   └── report_template.html     # Scrollable page Jinja2 template
├── styles/
│   └── CEG-style.css            # Constellation brand design system
└── output/                      # Generated outputs
    ├── figures/                  # PNG figures
    ├── ceg-ipp-transition-deck.html
    └── ceg-ipp-transition-report.html
```

## Key Feature: IPP Strategic Positioning Pentagon

The centerpiece visualization is a 5-axis radar chart comparing all 7 IPPs:

- **Nuclear Advantage** — nuclear capacity share (50% = score 100)
- **Geographic Diversity** — number of ISO regions (4+ = score 100)
- **Coal Exit Progress** — inverse coal share (0% coal = score 100)
- **Gas Flexibility** — CCGT vs peaker ratio (higher CCGT = better)
- **Clean Pipeline** — solar + wind + battery share (33%+ = score 100)

Constellation is highlighted with a thicker line and filled area.

## Dependencies

- Python 3.9+
- pandas, matplotlib, jinja2, numpy
- pyarrow (only needed for `export_data.py` parquet reading)
- No internet required for regeneration (fonts degrade gracefully)
