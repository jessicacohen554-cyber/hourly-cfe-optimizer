#!/usr/bin/env python3
"""
Generate the VRE Investment Thesis deck HTML from CSV data.

Reads data from data-inputs/, renders the Jinja2 template, and writes
the final deck to output/vre-investment-thesis-deck.html.

Usage:
    cd data-center-cfe/
    pip install -r requirements.txt
    python generate_deck.py
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data-inputs"
TEMPLATE_DIR = BASE_DIR / "templates"
OUTPUT_DIR = BASE_DIR / "output"

# ── CEG Colors (for chart datasets) ───────────────────────────────────
CEG_BLUE = "#2372B9"
CEG_ORANGE = "#F47B27"
CEG_GREEN = "#6BA543"
CEG_TEAL = "#007FA4"
CEG_YELLOW = "#FBB254"
CEG_GRAY = "#7E8083"

# ISO → chart color mapping for capture rate chart
ISO_CHART_COLORS = {
    "NYISO": "#6366F1",
    "PJM": CEG_BLUE,
    "MISO": CEG_ORANGE,
    "ERCOT": CEG_GREEN,
    "CAISO": "#DC2626",
}


def load_csv(name):
    """Load a CSV from data-inputs/."""
    path = DATA_DIR / name
    if not path.exists():
        print(f"WARNING: {path} not found, skipping.")
        return pd.DataFrame()
    return pd.read_csv(path)


# ══════════════════════════════════════════════════════════════════════
# CHART DATA BUILDERS
# ══════════════════════════════════════════════════════════════════════

def build_demand_chart_data():
    """
    Build Chart.js data for Slide 2: Demand Forecast bar chart.
    Source: demand_forecasts.csv (not directly usable as year-over-year series)
    and us_hyperscaler_gap_analysis.csv for the BNEF 106 GW figure.

    The demand chart shows BNEF, Goldman, and Wood Mac trajectories.
    Since the CSV has single-year snapshots per source, we interpolate
    the trajectories from known anchor points.
    """
    df = load_csv("demand_forecasts.csv")

    # BNEF trajectory: 38 GW (2024) → 106 GW (2035), interpolated
    # Goldman trajectory: 38 GW (2024) → ~100 GW (2030)
    # Wood Mac: 245 GW pipeline (2035 total, shown as outline bar)
    labels = ["2024", "2025", "2026", "2027", "2028", "2029", "2030", "2032", "2035"]

    # Extract anchor points from CSV where available
    bnef_2035 = 106  # BNEF 1H 2026 Outlook
    goldman_row = df[df["source"].str.contains("Goldman", case=False, na=False)]
    goldman_2025 = None
    if not goldman_row.empty:
        goldman_2025 = pd.to_numeric(goldman_row.iloc[0].get("year_2025", ""), errors="coerce")

    # Interpolated trajectories (from deck's original data)
    bnef_data = [38, 45, 54, 62, 72, 84, 96, None, bnef_2035]
    goldman_data = [38, 47, 58, 70, 82, 92, 100, None, None]
    woodmac_data = [None, None, None, None, None, None, None, None, 245]

    datasets = [
        {
            "label": "BNEF Forecast",
            "data": bnef_data,
            "backgroundColor": CEG_BLUE,
            "borderRadius": 4,
            "barPercentage": 0.7,
        },
        {
            "label": "Goldman Sachs",
            "data": goldman_data,
            "backgroundColor": CEG_ORANGE,
            "borderRadius": 4,
            "barPercentage": 0.7,
        },
        {
            "label": "Wood Mackenzie (Pipeline)",
            "data": woodmac_data,
            "backgroundColor": "rgba(107,165,67,0.4)",
            "borderColor": CEG_GREEN,
            "borderWidth": 2,
            "borderRadius": 4,
            "barPercentage": 0.7,
        },
    ]

    return json.dumps(labels), json.dumps(datasets)


def build_capture_chart_data():
    """
    Build Chart.js data for Slide 5: Capture Rate Erosion line chart.
    Source: capture_rates_by_iso.csv — solar rows for 5 ISOs.
    """
    df = load_csv("capture_rates_by_iso.csv")
    solar = df[df["technology"] == "Solar"].copy()

    labels = ["2020", "2022", "2024", "2026", "2030", "2035"]
    col_map = {
        "2020": "capture_rate_2020",
        "2022": "capture_rate_2022",
        "2024": "capture_rate_2024",
        "2026": "capture_rate_2026_est",
        "2030": "capture_rate_2030_est",
        "2035": "capture_rate_2035_est",
    }

    iso_order = ["NYISO", "PJM", "MISO", "ERCOT", "CAISO"]
    datasets = []

    for iso in iso_order:
        row = solar[solar["iso"] == iso]
        if row.empty:
            continue
        row = row.iloc[0]
        data = []
        for year_label in labels:
            col = col_map[year_label]
            val = pd.to_numeric(row.get(col, ""), errors="coerce")
            data.append(round(val * 100, 1) if pd.notna(val) else None)

        color = ISO_CHART_COLORS.get(iso, CEG_GRAY)
        ds = {
            "label": f"{iso} Solar",
            "data": data,
            "borderColor": color,
            "borderWidth": 2.5 if iso == "CAISO" else 2,
            "pointRadius": 3,
            "tension": 0.3,
            "fill": False,
        }
        datasets.append(ds)

    return json.dumps(labels), json.dumps(datasets)


def build_premium_chart_data():
    """
    Build Chart.js data for Slide 6: 24/7 CFE Premium bar chart.
    Source: cfe_premium.csv
    """
    df = load_csv("cfe_premium.csv")
    labels = [f"{int(t) if t == int(t) else t}%" for t in df["cfe_threshold_pct"]]
    data = df["premium_over_annual_mwh"].tolist()
    return json.dumps(labels), json.dumps(data)


# ══════════════════════════════════════════════════════════════════════
# TABLE DATA BUILDERS
# ══════════════════════════════════════════════════════════════════════

def build_gap_table_rows():
    """
    Build HTML table rows for Slide 4: Company-Level Gap table.
    Source: us_hyperscaler_gap_analysis.csv
    """
    df = load_csv("us_hyperscaler_gap_analysis.csv")
    rows = []

    companies = df[df["company"] != "Big_4_Total"]
    totals = df[df["company"] == "Big_4_Total"]

    for _, r in companies.iterrows():
        need = int(r["energy_need_twh_2030"])
        procured = int(r["procured_twh_prob_weighted"])
        gap = int(r["gap_twh_2030_vs_prob"])
        pct = int(round(r["procured_twh_prob_weighted"] / r["energy_need_twh_2030"] * 100))
        rows.append(
            f'          <tr><td>{r["company"]}</td><td>{need}</td><td>{procured}</td>'
            f'<td style="color:#F47B27;font-weight:700;">{gap}</td><td>{pct}%</td></tr>'
        )

    if not totals.empty:
        t = totals.iloc[0]
        need = int(t["energy_need_twh_2030"])
        procured = int(t["procured_twh_prob_weighted"])
        gap = int(t["gap_twh_2030_vs_prob"])
        pct = int(round(t["procured_twh_prob_weighted"] / t["energy_need_twh_2030"] * 100))
        rows.append(
            f'          <tr style="border-top:2px solid var(--ceg-blue);font-weight:700;">'
            f'<td><strong>Big 4 Total</strong></td><td><strong>{need}</strong></td>'
            f'<td><strong>{procured}</strong></td>'
            f'<td style="color:#F47B27;"><strong>{gap}</strong></td>'
            f'<td><strong>{pct}%</strong></td></tr>'
        )

    return "\n".join(rows)


def build_waterfall_svg():
    """
    Build SVG bars for the Slide 4 waterfall chart.
    Source: us_hyperscaler_gap_analysis.csv
    """
    df = load_csv("us_hyperscaler_gap_analysis.csv")
    totals = df[df["company"] == "Big_4_Total"]
    if totals.empty:
        return "<!-- No gap analysis data -->"

    t = totals.iloc[0]
    total_need = float(t["energy_need_twh_2030"])
    firm = float(t["procured_twh_firm"])
    prob_weighted = float(t["procured_twh_prob_weighted"])
    all_deals = float(t["procured_twh_all_deals"])

    probable = prob_weighted - firm
    announced = all_deals - prob_weighted
    gap = total_need - all_deals

    # SVG coordinate system: y range 40-240 (200px), maps to 0-max_val
    max_val = total_need * 1.05
    y_bottom = 240
    y_top = 40
    y_range = y_bottom - y_top

    def val_to_height(v):
        return max(4, v / max_val * y_range)

    def val_to_y(v_top_of_bar):
        return y_bottom - val_to_height(v_top_of_bar)

    bars = []

    # Bar 1: Total Demand
    h1 = val_to_height(total_need)
    y1 = y_bottom - h1
    bars.append(f'          <!-- Bar 1: Total Demand ({int(total_need)} TWh) -->')
    bars.append(f'          <rect x="80" y="{y1:.0f}" width="55" height="{h1:.0f}" rx="3" fill="#2372B9"/>')
    bars.append(f'          <text x="107" y="{y1 - 7:.0f}" text-anchor="middle" font-size="11" font-weight="700" fill="#2372B9">{int(total_need)}</text>')
    bars.append(f'          <text x="107" y="258" text-anchor="middle" font-size="8.5" fill="#1A232F">2030 Need</text>')

    # Bridge 2: Firm contracted
    remaining = total_need
    h2 = val_to_height(firm)
    y2 = y1
    bars.append(f'')
    bars.append(f'          <!-- Bridge: Firm contracted ({int(firm)} TWh) -->')
    bars.append(f'          <rect x="155" y="{y2:.0f}" width="55" height="{h2:.0f}" rx="3" fill="#6BA543"/>')
    bars.append(f'          <text x="182" y="{y2 - 7:.0f}" text-anchor="middle" font-size="10" font-weight="600" fill="#6BA543">-{int(firm)}</text>')
    bars.append(f'          <text x="182" y="258" text-anchor="middle" font-size="8.5" fill="#1A232F">Firm</text>')
    bars.append(f'          <text x="182" y="268" text-anchor="middle" font-size="7.5" fill="#7E8083">Contracted</text>')
    bars.append(f'          <line x1="135" y1="{y2:.0f}" x2="155" y2="{y2:.0f}" stroke="#E0E6EF" stroke-width="1" stroke-dasharray="3,2"/>')

    # Bridge 3: Probable deals
    remaining -= firm
    y3 = y2 + h2
    h3 = val_to_height(probable)
    bars.append(f'')
    bars.append(f'          <!-- Bridge: Probable deals ({int(probable)} TWh) -->')
    bars.append(f'          <rect x="230" y="{y3:.0f}" width="55" height="{h3:.0f}" rx="3" fill="#007FA4"/>')
    bars.append(f'          <text x="257" y="{y3 - 7:.0f}" text-anchor="middle" font-size="10" font-weight="600" fill="#007FA4">-{int(probable)}</text>')
    bars.append(f'          <text x="257" y="258" text-anchor="middle" font-size="8.5" fill="#1A232F">Probable</text>')
    bars.append(f'          <text x="257" y="268" text-anchor="middle" font-size="7.5" fill="#7E8083">Deals</text>')
    bars.append(f'          <line x1="210" y1="{y3:.0f}" x2="230" y2="{y3:.0f}" stroke="#E0E6EF" stroke-width="1" stroke-dasharray="3,2"/>')

    # Bridge 4: All other announced
    y4 = y3 + h3
    h4 = val_to_height(announced)
    bars.append(f'')
    bars.append(f'          <!-- Bridge: All announced ({int(announced)} TWh) -->')
    bars.append(f'          <rect x="305" y="{y4:.0f}" width="55" height="{h4:.0f}" rx="3" fill="#FBB254"/>')
    bars.append(f'          <text x="332" y="{y4 - 7:.0f}" text-anchor="middle" font-size="10" font-weight="600" fill="#B45309">-{int(announced)}</text>')
    bars.append(f'          <text x="332" y="258" text-anchor="middle" font-size="8.5" fill="#1A232F">All Other</text>')
    bars.append(f'          <text x="332" y="268" text-anchor="middle" font-size="7.5" fill="#7E8083">Announced</text>')
    bars.append(f'          <line x1="285" y1="{y4:.0f}" x2="305" y2="{y4:.0f}" stroke="#E0E6EF" stroke-width="1" stroke-dasharray="3,2"/>')

    # Bar 5: Remaining Gap
    y5 = y4 + h4
    h5 = val_to_height(gap)
    bars.append(f'')
    bars.append(f'          <!-- Bar: Remaining Gap ({int(gap)} TWh) -->')
    bars.append(f'          <rect x="380" y="{y5:.0f}" width="55" height="{h5:.0f}" rx="3" fill="#F47B27"/>')
    bars.append(f'          <text x="407" y="{y5 - 7:.0f}" text-anchor="middle" font-size="12" font-weight="700" fill="#F47B27">{int(gap)}</text>')
    bars.append(f'          <text x="407" y="258" text-anchor="middle" font-size="8.5" font-weight="700" fill="#F47B27">UNMET</text>')
    bars.append(f'          <text x="407" y="268" text-anchor="middle" font-size="8.5" font-weight="700" fill="#F47B27">GAP</text>')
    bars.append(f'          <line x1="360" y1="{y5:.0f}" x2="380" y2="{y5:.0f}" stroke="#E0E6EF" stroke-width="1" stroke-dasharray="3,2"/>')

    return "\n".join(bars)


def build_regional_matrix_rows():
    """
    Build HTML table rows for Slide 9: Regional Opportunity Matrix.
    Source: vre_investment_thesis.csv
    """
    df = load_csv("vre_investment_thesis.csv")

    # Use Solar+Storage row where available, else Solar
    iso_order = ["PJM", "ERCOT", "MISO", "CAISO", "NYISO", "NEISO", "SPP"]

    # Rating → CSS style mapping
    rating_styles = {
        "Strong Buy": 'style="background:var(--ceg-green);color:white;font-weight:700;"',
        "Buy": 'style="background:var(--ceg-blue);color:white;font-weight:700;"',
        "Buy with Storage": 'style="background:var(--ceg-blue);color:white;font-weight:700;"',
        "Buy (selective)": 'style="background:var(--ceg-yellow);color:#1A232F;font-weight:700;"',
        "Buy (small scale)": 'style="background:var(--ceg-teal);color:white;font-weight:700;"',
        "Hold": 'style="background:var(--ceg-yellow);color:#1A232F;font-weight:700;"',
        "Speculative Buy": 'style="background:var(--ceg-gray);color:white;font-weight:700;"',
        "Sell standalone / Buy with storage": 'style="background:var(--ceg-yellow);color:#1A232F;font-weight:700;"',
        "Avoid": 'style="background:#DC2626;color:white;font-weight:700;"',
    }

    rows = []
    for iso in iso_order:
        # Get the best row: prefer Solar+Storage, then Solar
        iso_rows = df[df["iso"] == iso]
        ss_row = iso_rows[iso_rows["vre_type"] == "Solar+Storage"]
        s_row = iso_rows[iso_rows["vre_type"] == "Solar"]

        if not ss_row.empty:
            r = ss_row.iloc[0]
        elif not s_row.empty:
            r = s_row.iloc[0]
        else:
            continue

        rating = str(r.get("investment_rating", "")).strip()
        rating_style = rating_styles.get(rating, 'style="font-weight:700;"')

        # Map cannibalization severity to heat class
        cannib = str(r.get("cannibalization_severity", ""))
        if "Advanced" in cannib or "Severe" in cannib:
            cap_heat = "heat-1"
        elif "Active" in cannib:
            cap_heat = "heat-2"
        elif "Early" in cannib:
            cap_heat = "heat-4"
        elif "Minimal" in cannib:
            cap_heat = "heat-5"
        else:
            cap_heat = "heat-3"

        # Capture rate text
        capture_text = cannib.split("(")[1].rstrip(")") if "(" in cannib else cannib

        # Storage value
        storage_val = str(r.get("storage_pairing_value", ""))
        if "Very High" in storage_val:
            storage_heat = "heat-5"
        elif "High" in storage_val:
            storage_heat = "heat-4"
        elif "Medium" in storage_val or "Moderate" in storage_val:
            storage_heat = "heat-3"
        elif "Low" in storage_val:
            storage_heat = "heat-2"
        else:
            storage_heat = "heat-3"

        # Key opportunity as strategic play (truncate)
        opp = str(r.get("key_opportunity", ""))
        # Take first sentence
        play = opp.split(". ")[0] + "." if ". " in opp else opp
        if len(play) > 120:
            play = play[:117] + "..."

        rows.append(f"""        <tr>
          <td><strong>{iso}</strong></td>
          <td class="{cap_heat}">{capture_text}</td>
          <td class="{storage_heat}">{storage_val}</td>
          <td {rating_style}>{rating}</td>
          <td style="font-size:8.5pt;text-align:left;">{play}</td>
        </tr>""")

    return "\n".join(rows)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load Jinja2 template — use custom delimiters to avoid conflicts with
    # existing CSS/JS {{ }} usage would be complex; instead we use the default
    # delimiters since our template variables are clearly outside CSS/JS.
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=False,
        # Keep template output as-is for non-template content
        keep_trailing_newline=True,
    )
    template = env.get_template("deck_template.html")

    # Build all data
    print("Loading data from CSVs...")
    demand_labels, demand_datasets = build_demand_chart_data()
    capture_labels, capture_datasets = build_capture_chart_data()
    premium_labels, premium_data = build_premium_chart_data()
    gap_rows = build_gap_table_rows()
    waterfall = build_waterfall_svg()
    matrix_rows = build_regional_matrix_rows()

    print("Rendering template...")
    html = template.render(
        # Chart data
        demand_chart_labels=demand_labels,
        demand_chart_datasets=demand_datasets,
        capture_chart_labels=capture_labels,
        capture_chart_datasets=capture_datasets,
        premium_chart_labels=premium_labels,
        premium_chart_data=premium_data,
        # Table data
        gap_table_rows=gap_rows,
        waterfall_svg_bars=waterfall,
        regional_matrix_rows=matrix_rows,
    )

    out_path = OUTPUT_DIR / "vre-investment-thesis-deck.html"
    out_path.write_text(html)

    print(f"\nDeck generated: {out_path}")
    print(f"  Size: {out_path.stat().st_size / 1024:.0f} KB")
    print(f"  Data sources: {len(list(DATA_DIR.glob('*.csv')))} CSVs from {DATA_DIR}/")
    print(f"\nOpen in browser to view the 14-slide presentation.")


if __name__ == "__main__":
    main()
