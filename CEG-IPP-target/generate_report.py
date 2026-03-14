#!/usr/bin/env python3
"""
Generate the IPP Climate Transition standalone HTML report.

Reads CSVs from data/ and renders the Jinja2 report template.
Unlike the deck, figures are linked as relative <img> paths (not base64)
so the report + figures/ directory travel together.

Usage:
    cd CEG-IPP-target/
    python generate_figures.py   # must run first
    python generate_report.py
"""

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
TEMPLATE_DIR = BASE_DIR / "templates"
OUTPUT_DIR = BASE_DIR / "output"


def build_kpi_cards(companies_df) -> str:
    """Build KPI card HTML for the report (uses glass-card style)."""
    total_cap = companies_df["cap_gw"].sum()
    total_co2 = companies_df["co2_2024_mt"].sum()
    n_companies = len(companies_df)

    ceg = companies_df[companies_df["company_id"] == "constellation"]
    ceg_nuc_pct = 0
    if not ceg.empty:
        r = ceg.iloc[0]
        cap_mw = float(r.get("cap_gw", 0) or 0) * 1000
        nuc_mw = float(r.get("nuclear_mw", 0) or 0)
        ceg_nuc_pct = round(nuc_mw / cap_mw * 100) if cap_mw > 0 else 0

    return f"""
      <div class="kpi-card">
        <div class="kpi-value" style="font-size:2rem;color:#2372B9">{n_companies}</div>
        <div class="kpi-label">IPPs Analyzed</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-value" style="font-size:2rem;color:#007FA4">{total_cap:.0f} GW</div>
        <div class="kpi-label">Combined Capacity</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-value" style="font-size:2rem;color:#F47B27">{total_co2:.0f} Mt</div>
        <div class="kpi-label">Combined CO₂ (2024)</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-value" style="font-size:2rem;color:#6BA543">{ceg_nuc_pct}%</div>
        <div class="kpi-label">Constellation Nuclear</div>
      </div>
    """


def build_key_findings(companies_df) -> str:
    """Build bullet list HTML."""
    ceg = companies_df[companies_df["company_id"] == "constellation"]
    ceg_intensity = int(ceg.iloc[0].get("intensity_kg", 0)) if not ceg.empty else 0
    avg_intensity = int(companies_df["intensity_kg"].mean())

    findings = [
        f"Constellation's emissions intensity ({ceg_intensity} kg/MWh) is significantly below the peer average ({avg_intensity} kg/MWh), driven by its dominant nuclear fleet.",
        "Nuclear-heavy fleets gain disproportionately under carbon pricing scenarios — the 45U PTC provides an additional $15/MWh revenue floor.",
        "Coal-exposed IPPs (Vistra, Talen) face accelerating stranded asset risk under most transition scenarios.",
        "Geographic diversification across ISOs provides revenue stability and reduces exposure to single-market regulatory risk.",
    ]
    return "\n".join(f"      <li>{f}</li>" for f in findings)


def build_score_table(companies_df) -> str:
    """Build full comparison table of all companies' spider scores."""
    scores = ipp.compute_all_spider_scores(companies_df)
    name_map = {r["company_id"]: r["short_name"] for _, r in companies_df.iterrows()}

    dims = ["nuclear_advantage", "geographic_diversity", "coal_exit_progress",
            "gas_flexibility", "clean_pipeline"]
    dim_labels = ["Nuclear<br>Advantage", "Geographic<br>Diversity", "Coal Exit<br>Progress",
                  "Gas<br>Flexibility", "Clean<br>Pipeline"]

    header = "<tr><th style='text-align:left'>Company</th>"
    header += "".join(f"<th>{l}</th>" for l in dim_labels)
    header += "<th>Average</th></tr>"

    rows = []
    for cid in ipp.COMPANY_ORDER:
        if cid not in scores:
            continue
        s = scores[cid]
        name = name_map.get(cid, cid)
        avg = sum(s.values()) / len(s)
        hero = ' class="hero-row"' if cid == "constellation" else ""

        dot_color = ipp.COMPANY_COLORS.get(cid, "#7E8083")
        cells = f'<td style="text-align:left"><span class="company-color-dot" style="background:{dot_color}"></span>{name}</td>'
        for d in dims:
            val = s[d]
            css = "score-high" if val >= 70 else ("score-mid" if val >= 40 else "score-low")
            cells += f'<td><span class="score-pill {css}">{val:.0f}</span></td>'
        cells += f'<td><strong>{avg:.0f}</strong></td>'
        rows.append(f"<tr{hero}>{cells}</tr>")

    return f"""
    <table class="company-score-table">
      <thead>{header}</thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
    """


def build_constellation_narrative(companies_df) -> str:
    """Build narrative text for Constellation deep-dive."""
    ceg = companies_df[companies_df["company_id"] == "constellation"]
    if ceg.empty:
        return "<p>No data available.</p>"
    r = ceg.iloc[0]
    scores = ipp.compute_spider_scores(r.to_dict())

    parts = []
    parts.append(f"""
    <p>Constellation Energy operates a fleet of <strong>{r.get('cap_gw', 0):.0f} GW</strong>
    across <strong>{r.get('iso_count', 0)}</strong> ISO regions, generating
    <strong>{r.get('gen_twh', 0):.0f} TWh</strong> annually.
    Its nuclear fleet of <strong>{r.get('nuclear_mw', 0):,.0f} MW</strong> is the
    largest in the US and the cornerstone of its transition positioning.</p>
    """)

    # Strengths
    score_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top = score_items[0]
    bottom = score_items[-1]
    parts.append(f"""
    <div class="insight-callout green">
      <strong>Top strength: {top[0].replace('_', ' ').title()}</strong> (score: {top[1]:.0f}/100) —
      this reflects Constellation's dominant position in zero-carbon baseload generation.
    </div>
    """)

    if bottom[1] < 50:
        parts.append(f"""
        <div class="insight-callout orange">
          <strong>Area to watch: {bottom[0].replace('_', ' ').title()}</strong> (score: {bottom[1]:.0f}/100) —
          indicates room for improvement in this dimension relative to peers.
        </div>
        """)

    return "\n".join(parts)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    companies_df = ipp.load_companies()
    sweep_df = ipp.load_sweep()

    n_scenarios = sweep_df["scenario"].nunique() if "scenario" in sweep_df.columns else 0
    total_cap_gw = companies_df["cap_gw"].sum()

    # Build template variables
    kpi_cards = build_kpi_cards(companies_df)
    key_findings = build_key_findings(companies_df)
    ceg_edge = (
        "Largest US zero-carbon generation fleet (nuclear), operating across multiple ISOs, "
        "with minimal coal exposure and strong 45U PTC tailwinds."
    )
    score_table = build_score_table(companies_df)
    ceg_narrative = build_constellation_narrative(companies_df)

    # Render
    print("Loading template...")
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR), encoding="utf-8"),
        autoescape=False,
        keep_trailing_newline=True,
    )
    template = env.get_template("report_template.html")

    html = template.render(
        report_date=date.today().strftime("%B %d, %Y"),
        total_cap_gw=f"{total_cap_gw:.0f}",
        n_scenarios=n_scenarios,
        kpi_cards_html=kpi_cards,
        key_findings_html=key_findings,
        constellation_edge=ceg_edge,
        score_table_html=score_table,
        constellation_narrative_html=ceg_narrative,
    )

    out_path = OUTPUT_DIR / "ceg-ipp-transition-report.html"
    out_path.write_text(html, encoding="utf-8")

    print(f"\nReport generated: {out_path}")
    print(f"  Size: {out_path.stat().st_size / 1024:.0f} KB")
    print(f"\nOpen in a browser to view. Ensure output/figures/ directory is present alongside the HTML.")


if __name__ == "__main__":
    main()
