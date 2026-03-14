#!/usr/bin/env python3
"""
Generate standalone PNG figures for the CEG IPP Climate Transition Report.

Reads CSVs from data/ and produces publication-quality PNGs in output/figures/.

Usage:
    cd CEG-IPP-target/
    pip install -r requirements.txt
    python generate_figures.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

import ipp_analysis as ipp

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
FIG_DIR = BASE_DIR / "output" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── CEG Color Palette ─────────────────────────────────────────────────
CEG_BLUE = "#2372B9"
CEG_BLUE_DARK = "#1B5E9E"
CEG_ORANGE = "#F47B27"
CEG_GREEN = "#6BA543"
CEG_TEAL = "#007FA4"
CEG_YELLOW = "#FBB254"
CEG_GRAY = "#7E8083"
CEG_RED = "#DC2626"
CEG_BG = "#F7FAFC"

FUEL_COLORS = {
    "nuclear": "#6366F1",
    "coal": "#374151",
    "gas_ccgt": "#7E8083",
    "gas_peaker": "#B0B5BA",
    "solar": "#F59E0B",
    "wind": "#22C55E",
    "hydro": "#0EA5E9",
    "battery": "#06B6D4",
    "geothermal": "#D97706",
    "oil": "#92400E",
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


def save_fig(fig, name):
    """Save figure to output/figures/ as PNG."""
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 1: Spider Map — All 7 IPPs
# ══════════════════════════════════════════════════════════════════════

def fig_spider_comparison():
    """All 7 IPPs on one radar chart with distinct colors."""
    companies_df = ipp.load_companies()
    scores = ipp.compute_all_spider_scores(companies_df)

    categories = ["Nuclear\nAdvantage", "Geographic\nDiversity",
                   "Coal Exit\nProgress", "Gas\nFlexibility", "Clean\nPipeline"]
    score_keys = ["nuclear_advantage", "geographic_diversity",
                   "coal_exit_progress", "gas_flexibility", "clean_pipeline"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Grid circles
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], fontsize=8, color="#B0B5BA")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight="600")

    # Draw grid
    ax.grid(True, color="#E0E6EF", linewidth=0.8)
    ax.spines["polar"].set_visible(False)

    # Plot each company
    name_map = {row["company_id"]: row["short_name"]
                for _, row in companies_df.iterrows()}

    for cid in ipp.COMPANY_ORDER:
        if cid not in scores:
            continue
        s = scores[cid]
        vals = [s[k] for k in score_keys]
        vals += vals[:1]
        color = ipp.COMPANY_COLORS[cid]
        label = name_map.get(cid, cid)

        if cid == "constellation":
            ax.plot(angles, vals, color=color, linewidth=3, label=label, zorder=10)
            ax.fill(angles, vals, color=color, alpha=0.15, zorder=5)
        else:
            ax.plot(angles, vals, color=color, linewidth=1.5, label=label,
                    alpha=0.8, zorder=4)

    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=10)
    ax.set_title("IPP Strategic Positioning Pentagon", fontsize=16, fontweight="bold",
                  pad=30, color=CEG_BLUE)

    save_fig(fig, "01_spider_comparison")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 2: Constellation Solo Spider
# ══════════════════════════════════════════════════════════════════════

def fig_spider_constellation():
    """Single-company radar chart for Constellation."""
    companies_df = ipp.load_companies()
    co_row = companies_df[companies_df["company_id"] == "constellation"]
    if co_row.empty:
        print("  WARNING: constellation not found, skipping solo spider.")
        return

    s = ipp.compute_spider_scores(co_row.iloc[0].to_dict())
    score_keys = ["nuclear_advantage", "geographic_diversity",
                   "coal_exit_progress", "gas_flexibility", "clean_pipeline"]
    categories = ["Nuclear\nAdvantage", "Geographic\nDiversity",
                   "Coal Exit\nProgress", "Gas\nFlexibility", "Clean\nPipeline"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    vals = [s[k] for k in score_keys]
    vals += vals[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], fontsize=8, color="#B0B5BA")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight="600")
    ax.grid(True, color="#E0E6EF", linewidth=0.8)
    ax.spines["polar"].set_visible(False)

    ax.plot(angles, vals, color=CEG_BLUE, linewidth=3, zorder=10)
    ax.fill(angles, vals, color=CEG_BLUE, alpha=0.2, zorder=5)

    # Add score labels at each point
    for i, (angle, val) in enumerate(zip(angles[:-1], vals[:-1])):
        ax.text(angle, val + 5, f"{val:.0f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=CEG_BLUE)

    ax.set_title("Constellation Energy — Strategic Profile", fontsize=16,
                  fontweight="bold", pad=30, color=CEG_BLUE)

    save_fig(fig, "02_spider_constellation")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 3: Fleet Emissions Trajectory (Fan Chart)
# ══════════════════════════════════════════════════════════════════════

def fig_emissions_fan():
    """Multi-company emissions fan chart (P10/P50/P90)."""
    sweep_df = ipp.load_sweep()
    companies_df = ipp.load_companies()
    name_map = {row["company_id"]: row["short_name"]
                for _, row in companies_df.iterrows()}

    fig, ax = plt.subplots(figsize=(12, 7))

    for cid in ipp.COMPANY_ORDER:
        bands = ipp.compute_fan_bands(sweep_df, cid)
        em = bands.get("fleet_emissions_mt", {})
        if not em:
            continue
        years = em["years"]
        p50 = em["p50"]
        p10 = em["p10"]
        p90 = em["p90"]
        color = ipp.COMPANY_COLORS[cid]
        label = name_map.get(cid, cid)

        if cid == "constellation":
            ax.fill_between(years, p10, p90, color=color, alpha=0.15, zorder=5)
            ax.plot(years, p50, color=color, linewidth=3, label=label, zorder=10)
        else:
            ax.plot(years, p50, color=color, linewidth=1.5, label=label, alpha=0.7)

    ax.set_xlabel("Year")
    ax.set_ylabel("Fleet Emissions (Mt CO₂)")
    ax.set_title("IPP Fleet Emissions Trajectory — P50 (Constellation shows P10–P90 band)",
                  fontsize=13, fontweight="bold", color=CEG_BLUE)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(left=min(years) if years else 2023)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

    save_fig(fig, "03_emissions_fan")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 4: Fleet Revenue Trajectory
# ══════════════════════════════════════════════════════════════════════

def fig_revenue_fan():
    """Multi-company revenue fan chart."""
    sweep_df = ipp.load_sweep()
    companies_df = ipp.load_companies()
    name_map = {row["company_id"]: row["short_name"]
                for _, row in companies_df.iterrows()}

    fig, ax = plt.subplots(figsize=(12, 7))

    for cid in ipp.COMPANY_ORDER:
        bands = ipp.compute_fan_bands(sweep_df, cid)
        rev = bands.get("fleet_revenue_m", {})
        if not rev:
            continue
        years = rev["years"]
        p50 = rev["p50"]
        color = ipp.COMPANY_COLORS[cid]
        label = name_map.get(cid, cid)

        if cid == "constellation":
            p10, p90 = rev["p10"], rev["p90"]
            ax.fill_between(years, p10, p90, color=color, alpha=0.15, zorder=5)
            ax.plot(years, p50, color=color, linewidth=3, label=label, zorder=10)
        else:
            ax.plot(years, p50, color=color, linewidth=1.5, label=label, alpha=0.7)

    ax.set_xlabel("Year")
    ax.set_ylabel("Fleet Revenue ($M)")
    ax.set_title("IPP Fleet Revenue Trajectory — P50", fontsize=13,
                  fontweight="bold", color=CEG_BLUE)
    ax.legend(loc="upper left", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    save_fig(fig, "04_revenue_fan")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 5: Stranded Asset Risk
# ══════════════════════════════════════════════════════════════════════

def fig_stranded_assets():
    """Stacked bar chart of stranded MW by company over time."""
    sweep_df = ipp.load_sweep()
    companies_df = ipp.load_companies()
    name_map = {row["company_id"]: row["short_name"]
                for _, row in companies_df.iterrows()}

    # Get P50 stranded MW per company per year
    years = sorted(sweep_df["year"].unique())
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(years))
    width = 0.11
    offsets = np.linspace(-3 * width, 3 * width, len(ipp.COMPANY_ORDER))

    for i, cid in enumerate(ipp.COMPANY_ORDER):
        bands = ipp.compute_fan_bands(sweep_df, cid)
        sm = bands.get("stranded_mw", {})
        if not sm:
            continue
        p50 = sm["p50"]
        color = ipp.COMPANY_COLORS[cid]
        label = name_map.get(cid, cid)
        ax.bar(x + offsets[i], p50, width, label=label, color=color, alpha=0.85)

    ax.set_xlabel("Year")
    ax.set_ylabel("Stranded Capacity (MW)")
    ax.set_title("Stranded Asset Risk by Company — P50 Across Scenarios",
                  fontsize=13, fontweight="bold", color=CEG_BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years])
    ax.legend(loc="upper left", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    save_fig(fig, "05_stranded_assets")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 6: Fleet Composition Comparison
# ══════════════════════════════════════════════════════════════════════

def fig_fleet_composition():
    """Horizontal stacked bar: capacity by fuel type per company."""
    fleet_df = ipp.load_fleet()
    companies_df = ipp.load_companies()
    name_map = {row["company_id"]: row["short_name"]
                for _, row in companies_df.iterrows()}

    fuel_matrix = ipp.build_company_fuel_matrix(fleet_df)
    fuel_order = ["nuclear", "gas_ccgt", "gas_peaker", "coal", "solar", "wind",
                  "hydro", "battery", "geothermal", "oil"]
    fuels_present = [f for f in fuel_order if f in fuel_matrix.columns]

    fig, ax = plt.subplots(figsize=(14, 7))

    y = np.arange(len(ipp.COMPANY_ORDER))
    left = np.zeros(len(ipp.COMPANY_ORDER))

    for fuel in fuels_present:
        vals = fuel_matrix[fuel].values if fuel in fuel_matrix.columns else np.zeros(len(y))
        color = FUEL_COLORS.get(fuel, CEG_GRAY)
        ax.barh(y, vals, left=left, label=fuel.replace("_", " ").title(),
                color=color, height=0.65)
        left += vals

    labels = [name_map.get(cid, cid) for cid in ipp.COMPANY_ORDER]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Capacity (MW)")
    ax.set_title("IPP Fleet Composition by Fuel Type",
                  fontsize=14, fontweight="bold", color=CEG_BLUE)
    ax.legend(loc="lower right", ncol=3, fontsize=9)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.invert_yaxis()

    save_fig(fig, "06_fleet_composition")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 7: Nuclear Fleet Advantage
# ══════════════════════════════════════════════════════════════════════

def fig_nuclear_advantage():
    """Bar chart: nuclear capacity % per company."""
    companies_df = ipp.load_companies()

    fig, ax = plt.subplots(figsize=(10, 6))

    names = []
    pcts = []
    colors = []
    for _, row in companies_df.iterrows():
        cid = row["company_id"]
        if cid not in ipp.COMPANY_ORDER:
            continue
        cap_mw = float(row.get("cap_gw", 0) or 0) * 1000
        nuc_mw = float(row.get("nuclear_mw", 0) or 0)
        pct = (nuc_mw / cap_mw * 100) if cap_mw > 0 else 0
        names.append(row.get("short_name", cid))
        pcts.append(pct)
        colors.append(ipp.COMPANY_COLORS.get(cid, CEG_GRAY))

    # Sort by nuclear pct descending
    order = np.argsort(pcts)[::-1]
    names = [names[i] for i in order]
    pcts = [pcts[i] for i in order]
    colors = [colors[i] for i in order]

    bars = ax.bar(names, pcts, color=colors, width=0.6, edgecolor="white")

    for bar, pct in zip(bars, pcts):
        if pct > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{pct:.0f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Nuclear Capacity (% of Total)")
    ax.set_title("Nuclear Fleet Advantage — Share of Total Capacity",
                  fontsize=14, fontweight="bold", color=CEG_BLUE)
    ax.set_ylim(0, max(pcts) * 1.15 if pcts else 100)

    save_fig(fig, "07_nuclear_advantage")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 8: Coal Exit Timeline
# ══════════════════════════════════════════════════════════════════════

def fig_coal_exit_timeline():
    """Timeline showing coal plant retirement dates by company."""
    fleet_df = ipp.load_fleet()
    coal = fleet_df[fleet_df["fuel"] == "coal"].copy()

    if coal.empty:
        print("  No coal plants found, skipping coal exit timeline.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # Group by company
    coal_sorted = coal.sort_values(["company_id", "retire_by"])
    companies_with_coal = coal_sorted["company_id"].unique()

    y_pos = 0
    yticks = []
    yticklabels = []

    for cid in ipp.COMPANY_ORDER:
        if cid not in companies_with_coal:
            continue
        co_coal = coal_sorted[coal_sorted["company_id"] == cid]
        color = ipp.COMPANY_COLORS.get(cid, CEG_GRAY)

        for _, plant in co_coal.iterrows():
            cap = plant["cap_mw"]
            retire = plant.get("retire_by")
            name = plant["plant_name"]
            short = name[:30] + "..." if len(str(name)) > 30 else str(name)

            bar_end = retire if retire and not np.isnan(retire) else 2055
            bar_start = 2023
            ax.barh(y_pos, bar_end - bar_start, left=bar_start,
                    height=0.7, color=color, alpha=0.8, edgecolor="white")

            # Capacity label
            ax.text(bar_end + 0.3, y_pos, f"{int(cap)} MW",
                    va="center", fontsize=8, color=CEG_GRAY)
            if retire and not np.isnan(retire):
                ax.text(bar_end, y_pos - 0.35, f"{int(retire)}",
                        va="top", ha="center", fontsize=7, color=color, fontweight="bold")

            yticks.append(y_pos)
            yticklabels.append(short)
            y_pos += 1

        y_pos += 0.5  # gap between companies

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=8)
    ax.set_xlabel("Year")
    ax.set_title("Coal Fleet Retirement Timeline",
                  fontsize=14, fontweight="bold", color=CEG_BLUE)
    ax.set_xlim(2022, 2058)
    ax.axvline(2030, color=CEG_RED, linestyle="--", alpha=0.5, label="2030")
    ax.legend(fontsize=9)
    ax.invert_yaxis()

    save_fig(fig, "08_coal_exit_timeline")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 9: Geographic Concentration Heatmap
# ══════════════════════════════════════════════════════════════════════

def fig_geographic_heatmap():
    """Heatmap: company × ISO capacity matrix."""
    fleet_df = ipp.load_fleet()
    companies_df = ipp.load_companies()
    name_map = {row["company_id"]: row["short_name"]
                for _, row in companies_df.iterrows()}

    matrix = ipp.build_company_iso_matrix(fleet_df)
    iso_order = ["PJM", "ERCOT", "MISO", "CAISO", "NYISO", "NEISO", "SPP"]
    isos_present = [iso for iso in iso_order if iso in matrix.columns]
    matrix = matrix[isos_present]

    fig, ax = plt.subplots(figsize=(12, 6))

    data = matrix.values
    im = ax.imshow(data, cmap="Blues", aspect="auto")

    ax.set_xticks(np.arange(len(isos_present)))
    ax.set_xticklabels(isos_present, fontsize=11)
    ax.set_yticks(np.arange(len(ipp.COMPANY_ORDER)))
    ax.set_yticklabels([name_map.get(c, c) for c in ipp.COMPANY_ORDER], fontsize=11)

    # Annotate cells
    for i in range(len(ipp.COMPANY_ORDER)):
        for j in range(len(isos_present)):
            val = data[i, j]
            if val > 0:
                text_color = "white" if val > data.max() * 0.6 else "#1A232F"
                ax.text(j, i, f"{int(val):,}", ha="center", va="center",
                        fontsize=9, color=text_color, fontweight="bold")

    ax.set_title("Geographic Capacity Distribution (MW)",
                  fontsize=14, fontweight="bold", color=CEG_BLUE)
    plt.colorbar(im, ax=ax, label="Capacity (MW)", shrink=0.8)

    save_fig(fig, "09_geographic_heatmap")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 10: Emissions Intensity Comparison
# ══════════════════════════════════════════════════════════════════════

def fig_emissions_intensity():
    """Bar chart: kg CO2/MWh per company."""
    companies_df = ipp.load_companies()

    fig, ax = plt.subplots(figsize=(10, 6))

    names = []
    intensities = []
    colors = []
    for cid in ipp.COMPANY_ORDER:
        row = companies_df[companies_df["company_id"] == cid]
        if row.empty:
            continue
        r = row.iloc[0]
        names.append(r.get("short_name", cid))
        intensities.append(float(r.get("intensity_kg", 0) or 0))
        colors.append(ipp.COMPANY_COLORS.get(cid, CEG_GRAY))

    # Sort by intensity descending
    order = np.argsort(intensities)[::-1]
    names = [names[i] for i in order]
    intensities = [intensities[i] for i in order]
    colors = [colors[i] for i in order]

    bars = ax.bar(names, intensities, color=colors, width=0.6, edgecolor="white")

    for bar, val in zip(bars, intensities):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{val:.0f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Emissions Intensity (kg CO₂/MWh)")
    ax.set_title("Fleet Emissions Intensity Comparison",
                  fontsize=14, fontweight="bold", color=CEG_BLUE)
    ax.axhline(y=410, color=CEG_RED, linestyle="--", alpha=0.5, label="US Grid Avg (~410)")
    ax.legend(fontsize=9)

    save_fig(fig, "10_emissions_intensity")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 11: Scenario Sensitivity Tornado
# ══════════════════════════════════════════════════════════════════════

def fig_scenario_sensitivity():
    """Tornado chart for Constellation's profit sensitivity to parameters."""
    sweep_df = ipp.load_sweep()

    # Find the latest year
    years = sorted(sweep_df["year"].unique())
    target_year = years[-1] if years else 2050

    sensitivity = ipp.compute_scenario_sensitivity(sweep_df, "constellation",
                                                     "fleet_profit_m", target_year)

    if not sensitivity:
        print("  No sensitivity data for Constellation, skipping tornado.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    params = list(sensitivity.keys())
    param_labels = {
        "demand_growth": "Demand Growth",
        "price_sens": "Price Sensitivity",
        "ppa_level": "PPA Level",
        "gas_friction": "Gas Transition Friction",
    }

    labels = [param_labels.get(p, p) for p in params]
    lows = [sensitivity[p]["low"] for p in params]
    highs = [sensitivity[p]["high"] for p in params]

    # Sort by total range
    ranges = [abs(h - l) for h, l in zip(highs, lows)]
    order = np.argsort(ranges)[::-1]
    labels = [labels[i] for i in order]
    lows = [lows[i] for i in order]
    highs = [highs[i] for i in order]

    y = np.arange(len(labels))
    ax.barh(y, [l for l in lows], color=CEG_ORANGE, height=0.5, label="Low scenario")
    ax.barh(y, [h for h in highs], color=CEG_BLUE, height=0.5, label="High scenario")
    ax.axvline(0, color="#374151", linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel(f"Δ Profit from Median ($M, {target_year})")
    ax.set_title(f"Constellation — Profit Sensitivity Analysis ({target_year})",
                  fontsize=14, fontweight="bold", color=CEG_BLUE)
    ax.legend(fontsize=10)
    ax.invert_yaxis()

    save_fig(fig, "11_scenario_sensitivity")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 12: Profit Under Carbon Pricing
# ══════════════════════════════════════════════════════════════════════

def fig_carbon_pricing():
    """Line chart: profit trajectories under different price sensitivity scenarios."""
    sweep_df = ipp.load_sweep()

    # Filter Constellation, look at price sensitivity dimension
    co_df = sweep_df[sweep_df["company"] == "constellation"]
    if co_df.empty:
        print("  No Constellation data, skipping carbon pricing chart.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    price_levels = sorted(co_df["price_sens"].unique())
    color_map = {
        0: CEG_GREEN,
        1: CEG_BLUE,
        2: CEG_ORANGE,
    }

    for i, level in enumerate(price_levels):
        subset = co_df[co_df["price_sens"] == level]
        years = sorted(subset["year"].unique())
        p50 = []
        for yr in years:
            vals = subset[subset["year"] == yr]["fleet_profit_m"]
            p50.append(float(vals.median()))

        color = color_map.get(i, CEG_GRAY)
        ax.plot(years, p50, color=color, linewidth=2.5, label=f"Price Scenario: {level}",
                marker="o", markersize=4)

    ax.set_xlabel("Year")
    ax.set_ylabel("Fleet Profit ($M)")
    ax.set_title("Constellation — Profit Under Different Price Scenarios",
                  fontsize=14, fontweight="bold", color=CEG_BLUE)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    save_fig(fig, "12_carbon_pricing")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("Generating figures...")
    print()

    fig_spider_comparison()
    fig_spider_constellation()
    fig_emissions_fan()
    fig_revenue_fan()
    fig_stranded_assets()
    fig_fleet_composition()
    fig_nuclear_advantage()
    fig_coal_exit_timeline()
    fig_geographic_heatmap()
    fig_emissions_intensity()
    fig_scenario_sensitivity()
    fig_carbon_pricing()

    print()
    print(f"Done. {len(list(FIG_DIR.glob('*.png')))} figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
