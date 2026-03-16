#!/usr/bin/env python3
"""
Generate standalone PNG figures from VRE Investment Thesis data.

Reads CSVs from data-inputs/ and produces publication-quality PNGs
in output/figures/.

Usage:
    cd data-center-cfe/
    pip install -r requirements.txt
    python generate_figures.py
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data-inputs"
FIG_DIR = BASE_DIR / "output" / "figures"

# ── CEG Color Palette ─────────────────────────────────────────────────
CEG_BLUE = "#2372B9"
CEG_BLUE_DARK = "#1B5E9E"
CEG_ORANGE = "#F47B27"
CEG_GREEN = "#6BA543"
CEG_TEAL = "#007FA4"
CEG_YELLOW = "#FBB254"
CEG_GRAY = "#7E8083"
CEG_RED = "#DC2626"
CEG_SLATE = "#7F8F97"
CEG_BG = "#F7FAFC"

ISO_COLORS = {
    "NYISO": "#6366F1",
    "PJM": CEG_BLUE,
    "MISO": CEG_ORANGE,
    "ERCOT": CEG_GREEN,
    "CAISO": CEG_RED,
    "NEISO": "#9C27B0",
    "SPP": "#14B8A6",
}

# ── Matplotlib Style ─────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Source Sans Pro", "Helvetica Neue", "Arial", "sans-serif"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.facecolor": "white",
    "axes.edgecolor": "#E0E6EF",
    "axes.grid": True,
    "grid.color": "#F0F3F7",
    "grid.linewidth": 0.8,
    "figure.facecolor": "white",
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "xtick.color": "#7E8083",
    "ytick.color": "#7E8083",
    "legend.frameon": False,
    "legend.fontsize": 9,
})


def load_csv(name):
    """Load a CSV from data-inputs/."""
    path = DATA_DIR / name
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping.")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8")
    except pd.errors.ParserError:
        # Some CSVs have unquoted commas in fields — try with python engine
        return pd.read_csv(path, encoding="utf-8", engine="python", on_bad_lines="warn")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 1: Demand Forecast
# ══════════════════════════════════════════════════════════════════════

def fig_demand_forecast():
    """Grouped bar chart: BNEF / Goldman / Wood Mac demand forecasts."""
    labels = ["2024", "2025", "2026", "2027", "2028", "2029", "2030", "2032", "2035"]
    bnef = [38, 45, 54, 62, 72, 84, 96, None, 106]
    goldman = [38, 47, 58, 70, 82, 92, 100, None, None]
    woodmac = [None, None, None, None, None, None, None, None, 245]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Plot each series
    bnef_vals = [v if v is not None else 0 for v in bnef]
    bnef_mask = [v is not None for v in bnef]
    goldman_vals = [v if v is not None else 0 for v in goldman]
    goldman_mask = [v is not None for v in goldman]
    woodmac_vals = [v if v is not None else 0 for v in woodmac]
    woodmac_mask = [v is not None for v in woodmac]

    bars1 = ax.bar(x[bnef_mask] - width, [bnef_vals[i] for i in range(len(bnef_vals)) if bnef_mask[i]],
                   width, label="BNEF Forecast", color=CEG_BLUE, zorder=3)
    bars2 = ax.bar(x[goldman_mask], [goldman_vals[i] for i in range(len(goldman_vals)) if goldman_mask[i]],
                   width, label="Goldman Sachs", color=CEG_ORANGE, zorder=3)
    bars3 = ax.bar(x[woodmac_mask] + width, [woodmac_vals[i] for i in range(len(woodmac_vals)) if woodmac_mask[i]],
                   width, label="Wood Mackenzie (Pipeline)", color=CEG_GREEN, alpha=0.5,
                   edgecolor=CEG_GREEN, linewidth=1.5, zorder=3)

    # Value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., h + 3,
                        f"{int(h)}", ha="center", va="bottom", fontsize=8,
                        fontweight="600", color="#1A232F")

    ax.set_xlabel("")
    ax.set_ylabel("GW", fontweight="600")
    ax.set_title("US Data Center Power Demand Forecasts", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 270)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#E0E6EF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


# ══════════════════════════════════════════════════════════════════════
# FIGURE 2: Capture Rate Erosion
# ══════════════════════════════════════════════════════════════════════

def fig_capture_rate_erosion():
    """Multi-line chart: Solar capture rate by ISO over time."""
    df = load_csv("capture_rates_by_iso.csv")
    solar = df[df["technology"] == "Solar"].copy()

    col_map = {
        "2020": "capture_rate_2020",
        "2022": "capture_rate_2022",
        "2024": "capture_rate_2024",
        "2026": "capture_rate_2026_est",
        "2030": "capture_rate_2030_est",
        "2035": "capture_rate_2035_est",
    }
    years = list(col_map.keys())
    x = np.arange(len(years))

    iso_order = ["NYISO", "PJM", "MISO", "ERCOT", "CAISO"]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for iso in iso_order:
        row = solar[solar["iso"] == iso]
        if row.empty:
            continue
        row = row.iloc[0]
        vals = []
        for yr in years:
            v = pd.to_numeric(row.get(col_map[yr], ""), errors="coerce")
            vals.append(v * 100 if pd.notna(v) else None)

        color = ISO_COLORS.get(iso, CEG_GRAY)
        lw = 2.5 if iso == "CAISO" else 2.0
        ax.plot(x, vals, marker="o", markersize=5, linewidth=lw, color=color,
                label=f"{iso} Solar", zorder=3)

    # Reference line at 100%
    ax.axhline(y=100, color="#E0E6EF", linewidth=1, linestyle="--", zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylabel("Capture Rate (%)", fontweight="600")
    ax.set_title("Solar Capture Rate Erosion by ISO Market", pad=12)
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#E0E6EF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


# ══════════════════════════════════════════════════════════════════════
# FIGURE 3: CFE Premium
# ══════════════════════════════════════════════════════════════════════

def fig_cfe_premium():
    """Color-coded bar chart: 24/7 CFE cost premium by matching level."""
    df = load_csv("cfe_premium.csv")
    if df.empty:
        return None

    thresholds = df["cfe_threshold_pct"].values
    premiums = df["premium_over_annual_mwh"].values
    labels = [f"{int(t) if t == int(t) else t}%" for t in thresholds]

    colors = []
    for v in premiums:
        if v <= 10:
            colors.append(CEG_GREEN)
        elif v <= 25:
            colors.append(CEG_BLUE)
        elif v <= 58:
            colors.append(CEG_ORANGE)
        else:
            colors.append(CEG_RED)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(labels)), premiums, color=colors, zorder=3, width=0.7)

    for bar, val in zip(bars, premiums):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1.5,
                f"${int(val)}", ha="center", va="bottom", fontsize=9,
                fontweight="600", color="#1A232F")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Hourly CFE Matching Level", fontweight="600")
    ax.set_ylabel("$/MWh Premium", fontweight="600")
    ax.set_title("24/7 CFE Cost Premium over Annual Matching", pad=12)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%d"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


# ══════════════════════════════════════════════════════════════════════
# FIGURE 4: Investment Matrix Heatmap
# ══════════════════════════════════════════════════════════════════════

def fig_investment_matrix():
    """Heatmap of ISO × VRE type investment ratings."""
    df = load_csv("vre_investment_thesis.csv")
    if df.empty:
        return None

    # Filter to main types
    types = ["Solar", "Solar+Storage", "Wind"]
    isos = ["PJM", "ERCOT", "MISO", "CAISO", "NYISO", "NEISO", "SPP"]

    rating_map = {
        "Strong Buy": 5, "Buy": 4, "Buy with Storage": 3.5,
        "Buy (selective)": 3, "Buy (small scale)": 2.5,
        "Speculative Buy": 2, "Hold": 1.5,
        "Sell standalone / Buy with storage": 1, "Avoid": 0,
    }
    rating_colors = {
        5: CEG_GREEN, 4: CEG_BLUE, 3.5: CEG_TEAL,
        3: CEG_YELLOW, 2.5: CEG_TEAL, 2: CEG_GRAY,
        1.5: CEG_YELLOW, 1: CEG_ORANGE, 0: CEG_RED,
    }

    matrix = np.full((len(isos), len(types)), np.nan)
    labels = [["" for _ in types] for _ in isos]

    for i, iso in enumerate(isos):
        for j, vtype in enumerate(types):
            row = df[(df["iso"] == iso) & (df["vre_type"] == vtype)]
            if not row.empty:
                rating = row.iloc[0]["investment_rating"].strip()
                matrix[i, j] = rating_map.get(rating, 0)
                labels[i][j] = rating

    fig, ax = plt.subplots(figsize=(8, 6))

    # Custom colormap: red → yellow → green
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "rating", [CEG_RED, CEG_ORANGE, CEG_YELLOW, CEG_TEAL, CEG_GREEN], N=256
    )

    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=5)

    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(types, fontsize=11, fontweight="600")
    ax.set_yticks(range(len(isos)))
    ax.set_yticklabels(isos, fontsize=11, fontweight="600")

    # Add text annotations
    for i in range(len(isos)):
        for j in range(len(types)):
            txt = labels[i][j]
            if txt:
                # Pick white or dark text based on score
                score = matrix[i, j]
                text_color = "white" if score >= 3.5 or score <= 1 else "#1A232F"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=8, fontweight="600", color=text_color)

    ax.set_title("VRE Investment Rating by ISO × Technology", pad=16,
                 fontsize=14, fontweight="bold")
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Grid
    for edge in ["top", "bottom", "left", "right"]:
        ax.spines[edge].set_visible(False)
    ax.set_xticks(np.arange(len(types) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(isos) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", size=0)

    return fig


# ══════════════════════════════════════════════════════════════════════
# FIGURE 5: Gap Analysis Waterfall
# ══════════════════════════════════════════════════════════════════════

def fig_gap_analysis_waterfall():
    """Waterfall chart: Big 4 hyperscaler clean energy gap."""
    df = load_csv("us_hyperscaler_gap_analysis.csv")
    totals = df[df["company"] == "Big_4_Total"]
    if totals.empty:
        return None

    t = totals.iloc[0]
    total_need = float(t["energy_need_twh_2030"])
    firm = float(t["procured_twh_firm"])
    prob_weighted = float(t["procured_twh_prob_weighted"])
    all_deals = float(t["procured_twh_all_deals"])

    probable = prob_weighted - firm
    announced = all_deals - prob_weighted
    gap = total_need - all_deals

    categories = ["2030 Need", "Firm\nContracted", "Probable\nDeals", "All Other\nAnnounced", "UNMET\nGAP"]
    values = [total_need, -firm, -probable, -announced, gap]
    colors_list = [CEG_BLUE, CEG_GREEN, CEG_TEAL, CEG_YELLOW, CEG_ORANGE]

    # Compute cumulative positions for waterfall
    cumulative = [0]
    running = 0
    bottoms = []
    for i, v in enumerate(values):
        if i == 0:
            bottoms.append(0)
            running = v
        elif i == len(values) - 1:
            bottoms.append(0)
        else:
            running += v
            bottoms.append(running)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bar_heights = [abs(v) for v in values]
    bars = ax.bar(categories, bar_heights, bottom=bottoms, color=colors_list,
                  width=0.6, zorder=3, edgecolor="white", linewidth=0.5)

    # Value labels
    for bar, val, bottom in zip(bars, values, bottoms):
        h = abs(val)
        y_pos = bottom + h + 8
        prefix = "" if val > 0 and categories[bars.index(bar)] != "UNMET\nGAP" else "-" if val < 0 else ""
        label = f"{prefix}{int(abs(val))} TWh"
        if val > 0 and "UNMET" in categories[bars.index(bar)]:
            label = f"{int(abs(val))} TWh"
        ax.text(bar.get_x() + bar.get_width() / 2., y_pos,
                label, ha="center", va="bottom", fontsize=10,
                fontweight="700", color=colors_list[bars.index(bar)])

    # Connector lines
    for i in range(len(values) - 2):
        top = bottoms[i] + bar_heights[i] if values[i] > 0 else bottoms[i]
        x1 = i + 0.3
        x2 = i + 0.7
        ax.plot([x1, x2], [bottoms[i + 1] + bar_heights[i + 1]] * 2,
                color="#E0E6EF", linewidth=1, linestyle="--", zorder=2)

    ax.set_ylabel("TWh", fontweight="600")
    ax.set_title("Big 4 Hyperscaler Clean Energy Gap (2030)", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, total_need * 1.15)

    return fig


# ══════════════════════════════════════════════════════════════════════
# FIGURE 6: Cost Projections
# ══════════════════════════════════════════════════════════════════════

def fig_cost_projections():
    """LCOE trajectory ranges by technology."""
    df = load_csv("cost_projections.csv")
    if df.empty:
        return None

    techs = ["Utility Solar", "Onshore Wind", "Battery 4hr (LCOS)",
             "Solar+4hr Storage", "Nuclear SMR (FOAK)", "Geothermal EGS"]
    years = [2025, 2030, 2035, 2040]
    tech_colors = {
        "Utility Solar": "#F59E0B",
        "Onshore Wind": "#22C55E",
        "Battery 4hr (LCOS)": "#06B6D4",
        "Solar+4hr Storage": CEG_ORANGE,
        "Nuclear SMR (FOAK)": "#6366F1",
        "Geothermal EGS": "#D97706",
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for tech in techs:
        row = df[df["technology"] == tech]
        if row.empty:
            continue
        row = row.iloc[0]

        lows = []
        highs = []
        mids = []
        for yr in years:
            lo = pd.to_numeric(row.get(f"lcoe_{yr}_low", ""), errors="coerce")
            hi = pd.to_numeric(row.get(f"lcoe_{yr}_high", ""), errors="coerce")
            if pd.notna(lo) and pd.notna(hi):
                lows.append(lo)
                highs.append(hi)
                mids.append((lo + hi) / 2)
            else:
                lows.append(None)
                highs.append(None)
                mids.append(None)

        color = tech_colors.get(tech, CEG_GRAY)

        # Filter valid points
        valid = [i for i in range(len(years)) if mids[i] is not None]
        if not valid:
            continue
        vx = [years[i] for i in valid]
        vy_mid = [mids[i] for i in valid]
        vy_lo = [lows[i] for i in valid]
        vy_hi = [highs[i] for i in valid]

        ax.plot(vx, vy_mid, marker="o", markersize=5, linewidth=2,
                color=color, label=tech, zorder=3)
        ax.fill_between(vx, vy_lo, vy_hi, alpha=0.15, color=color, zorder=2)

    ax.set_xlabel("Year", fontweight="600")
    ax.set_ylabel("LCOE ($/MWh)", fontweight="600")
    ax.set_title("Technology Cost Projections (LCOE Range)", pad=12)
    ax.set_xticks(years)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#E0E6EF",
              fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 280)

    return fig


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    figures = [
        ("demand_forecast.png", fig_demand_forecast, "US DC Power Demand Forecasts"),
        ("capture_rate_erosion.png", fig_capture_rate_erosion, "Solar Capture Rate Erosion"),
        ("cfe_premium.png", fig_cfe_premium, "24/7 CFE Cost Premium"),
        ("investment_matrix.png", fig_investment_matrix, "Investment Rating Matrix"),
        ("gap_analysis_waterfall.png", fig_gap_analysis_waterfall, "Clean Energy Gap Waterfall"),
        ("cost_projections.png", fig_cost_projections, "LCOE Cost Projections"),
    ]

    print(f"Generating {len(figures)} figures into {FIG_DIR}/\n")

    for filename, func, title in figures:
        print(f"  {filename}... ", end="", flush=True)
        fig = func()
        if fig is None:
            print("SKIPPED (missing data)")
            continue
        path = FIG_DIR / filename
        fig.savefig(path)
        plt.close(fig)
        size_kb = path.stat().st_size / 1024
        print(f"OK ({size_kb:.0f} KB)")

    print(f"\nDone. {len(figures)} figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
