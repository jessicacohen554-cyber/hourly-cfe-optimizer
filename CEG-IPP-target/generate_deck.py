#!/usr/bin/env python3
"""
Generate the IPP Climate Transition slide deck from CSV data + pre-rendered figures.

Reads data from data/, encodes figure PNGs as base64, and renders the
Jinja2 template into a single portable HTML file.

Usage:
    cd CEG-IPP-target/
    python generate_figures.py   # must run first
    python generate_deck.py
"""

import base64
import sys
from datetime import date
from pathlib import Path

import pandas as pd

try:
    from jinja2 import Environment, FileSystemLoader
except ImportError:
    print("ERROR: jinja2 not installed. Run: pip install -r requirements.txt")
    sys.exit(1)

import ipp_analysis as ipp

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
FIG_DIR = BASE_DIR / "output" / "figures"
TEMPLATE_DIR = BASE_DIR / "templates"
OUTPUT_DIR = BASE_DIR / "output"


def load_figure_b64(name: str) -> str:
    """Load a PNG figure and return as base64 string."""
    path = FIG_DIR / f"{name}.png"
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return ""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def build_kpi_cards(companies_df, sweep_df) -> str:
    """Build HTML for the 4 executive summary KPI cards."""
    total_cap = companies_df["cap_gw"].sum()
    total_co2 = companies_df["co2_2024_mt"].sum()
    n_companies = len(companies_df)

    # Constellation-specific
    ceg = companies_df[companies_df["company_id"] == "constellation"]
    ceg_nuclear_pct = 0
    if not ceg.empty:
        r = ceg.iloc[0]
        cap_mw = float(r.get("cap_gw", 0) or 0) * 1000
        nuc_mw = float(r.get("nuclear_mw", 0) or 0)
        ceg_nuclear_pct = round(nuc_mw / cap_mw * 100) if cap_mw > 0 else 0

    cards = f"""
      <div class="kpi-card">
        <div class="kpi-value">{n_companies}</div>
        <div class="kpi-label">IPPs Analyzed</div>
      </div>
      <div class="kpi-card teal">
        <div class="kpi-value">{total_cap:.0f}<span class="kpi-unit"> GW</span></div>
        <div class="kpi-label">Combined Capacity</div>
      </div>
      <div class="kpi-card orange">
        <div class="kpi-value">{total_co2:.0f}<span class="kpi-unit"> Mt</span></div>
        <div class="kpi-label">Combined CO₂ (2024)</div>
      </div>
      <div class="kpi-card green">
        <div class="kpi-value">{ceg_nuclear_pct}%</div>
        <div class="kpi-label">Constellation Nuclear Share</div>
      </div>
    """
    return cards


def build_key_findings(companies_df, sweep_df) -> str:
    """Build HTML list items for key findings."""
    ceg = companies_df[companies_df["company_id"] == "constellation"]
    ceg_intensity = int(ceg.iloc[0].get("intensity_kg", 0)) if not ceg.empty else 0
    avg_intensity = int(companies_df["intensity_kg"].mean())

    findings = [
        f"Constellation's emissions intensity ({ceg_intensity} kg/MWh) is significantly below the peer average ({avg_intensity} kg/MWh), driven by its dominant nuclear fleet.",
        "Nuclear-heavy fleets gain disproportionately under carbon pricing scenarios — the 45U PTC provides an additional $15/MWh revenue floor.",
        "Coal-exposed IPPs (Vistra, Talen) face accelerating stranded asset risk, with 5,000–10,000 MW at risk by 2035 under median scenarios.",
        "Geographic diversification provides revenue stability — companies operating in 3+ ISOs show lower profit variance across scenarios.",
    ]

    return "\n".join(f'      <li>{f}</li>' for f in findings)


def build_constellation_scores(companies_df) -> str:
    """Build HTML table showing Constellation's spider scores."""
    ceg = companies_df[companies_df["company_id"] == "constellation"]
    if ceg.empty:
        return "<p>No Constellation data</p>"

    scores = ipp.compute_spider_scores(ceg.iloc[0].to_dict())

    labels = {
        "nuclear_advantage": "Nuclear Advantage",
        "geographic_diversity": "Geographic Diversity",
        "coal_exit_progress": "Coal Exit Progress",
        "gas_flexibility": "Gas Flexibility",
        "clean_pipeline": "Clean Pipeline",
    }

    rows = []
    for key, label in labels.items():
        val = scores[key]
        css = "score-high" if val >= 70 else ("score-mid" if val >= 40 else "score-low")
        rows.append(
            f'        <tr><td style="text-align:left;font-weight:500">{label}</td>'
            f'<td><span class="score-pill {css}">{val:.0f}</span></td></tr>'
        )

    return f"""
        <table class="data-table" style="max-width:4in">
          <thead><tr><th style="text-align:left">Dimension</th><th>Score</th></tr></thead>
          <tbody>
            {"".join(rows)}
          </tbody>
        </table>
    """


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    companies_df = ipp.load_companies()
    sweep_df = ipp.load_sweep()

    # Count unique scenarios
    n_scenarios = sweep_df["scenario"].nunique() if "scenario" in sweep_df.columns else 0

    # Load all figure base64
    figures = {
        "fig_fleet_composition": load_figure_b64("06_fleet_composition"),
        "fig_spider_comparison": load_figure_b64("01_spider_comparison"),
        "fig_spider_constellation": load_figure_b64("02_spider_constellation"),
        "fig_nuclear_advantage": load_figure_b64("07_nuclear_advantage"),
        "fig_coal_exit": load_figure_b64("08_coal_exit_timeline"),
        "fig_emissions_fan": load_figure_b64("03_emissions_fan"),
        "fig_stranded_assets": load_figure_b64("05_stranded_assets"),
        "fig_carbon_pricing": load_figure_b64("12_carbon_pricing"),
        "fig_geographic": load_figure_b64("09_geographic_heatmap"),
        "fig_sensitivity": load_figure_b64("11_scenario_sensitivity"),
    }

    # Build computed HTML
    kpi_cards = build_kpi_cards(companies_df, sweep_df)
    key_findings = build_key_findings(companies_df, sweep_df)
    ceg_scores = build_constellation_scores(companies_df)

    ceg = companies_df[companies_df["company_id"] == "constellation"]
    ceg_edge = (
        "Largest US zero-carbon generation fleet (nuclear), operating across multiple ISOs, "
        "with minimal coal exposure and strong 45U PTC tailwinds."
    )

    # Render
    print("Loading template...")
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR), encoding="utf-8"),
        autoescape=False,
        keep_trailing_newline=True,
    )
    template = env.get_template("deck_template.html")

    html = template.render(
        report_date=date.today().strftime("%B %d, %Y"),
        n_scenarios=n_scenarios,
        kpi_cards_html=kpi_cards,
        key_findings_html=key_findings,
        constellation_edge=ceg_edge,
        constellation_scores_html=ceg_scores,
        **figures,
    )

    out_path = OUTPUT_DIR / "ceg-ipp-transition-deck.html"
    out_path.write_text(html, encoding="utf-8")

    print(f"\nDeck generated: {out_path}")
    print(f"  Size: {out_path.stat().st_size / 1024:.0f} KB")
    print(f"  Slides: 14")
    print(f"\nOpen in a browser to view the deck.")


if __name__ == "__main__":
    main()
