#!/usr/bin/env python3
"""
Shared analysis functions for CEG IPP Climate Transition Report.

Provides spider-map scoring, fleet aggregation, and sweep analysis
used by both generate_figures.py and the report/deck generators.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# ── Company Colors (CEG brand palette) ──────────────────────────────
COMPANY_COLORS = {
    "constellation": "#2372B9",  # CEG Blue (hero)
    "vistra":        "#F47B27",  # CEG Orange
    "nrg":           "#6BA543",  # CEG Green
    "nextera":       "#007FA4",  # CEG Teal
    "talen":         "#FBB254",  # CEG Yellow
    "pseg":          "#9C27B0",  # Purple
    "aes":           "#14B8A6",  # Teal-green
}

COMPANY_ORDER = ["constellation", "vistra", "nrg", "nextera", "talen", "pseg", "aes"]

SPIDER_LABELS = [
    "Nuclear\nAdvantage",
    "Geographic\nDiversity",
    "Coal Exit\nProgress",
    "Gas\nFlexibility",
    "Clean\nPipeline",
]

FUEL_CATEGORIES = {
    "clean": ["nuclear", "solar", "wind", "hydro", "geothermal", "battery"],
    "fossil": ["coal", "gas_ccgt", "gas_peaker", "oil"],
    "gas": ["gas_ccgt", "gas_peaker"],
    "new_clean": ["solar", "wind", "battery"],
}


# ══════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════

def load_fleet() -> pd.DataFrame:
    """Load per-plant fleet CSV."""
    return pd.read_csv(DATA_DIR / "ipp_fleet_config.csv")


def load_companies() -> pd.DataFrame:
    """Load per-company summary CSV."""
    return pd.read_csv(DATA_DIR / "ipp_fleet_companies.csv")


def load_sweep() -> pd.DataFrame:
    """Load sweep results CSV."""
    return pd.read_csv(DATA_DIR / "ipp_sweep_results.csv")


# ══════════════════════════════════════════════════════════════════════
# Spider Map Scoring
# ══════════════════════════════════════════════════════════════════════

def compute_spider_scores(company_row: dict) -> dict:
    """
    Compute 5 strategic positioning scores (0-100) for a company.

    Ported from dashboard/js/ipp-company-report.js lines 1255-1272.

    Args:
        company_row: dict with keys matching ipp_fleet_companies.csv columns

    Returns:
        dict with keys: nuclear_advantage, geographic_diversity,
        coal_exit_progress, gas_flexibility, clean_pipeline
    """
    total_gen = float(company_row.get("gen_twh", 0) or 0)

    # Nuclear advantage: 50% nuclear gen share = score of 100
    nuclear_mw = float(company_row.get("nuclear_mw", 0) or 0)
    # Approximate nuclear gen from capacity (90% CF)
    total_cap_mw = float(company_row.get("cap_gw", 0) or 0) * 1000
    if total_cap_mw > 0:
        nuclear_share = nuclear_mw / total_cap_mw
    else:
        nuclear_share = 0
    nuclear_score = min(100, nuclear_share * 200)

    # Geographic diversity: 4+ ISOs = score of 100
    iso_count = int(company_row.get("iso_count", 1) or 1)
    diversity_score = min(100, iso_count * 25)

    # Coal exit progress: 0% coal = 100, 50% coal = 0
    coal_mw = float(company_row.get("coal_mw", 0) or 0)
    if total_cap_mw > 0:
        coal_share = coal_mw / total_cap_mw
    else:
        coal_share = 0
    coal_exit_score = max(0, 100 - coal_share * 200)

    # Gas flexibility: CCGT / (CCGT + peaker) — higher CCGT ratio = more flexible
    ccgt_mw = float(company_row.get("gas_ccgt_mw", 0) or 0)
    peaker_mw = float(company_row.get("gas_peaker_mw", 0) or 0)
    if ccgt_mw + peaker_mw > 0:
        gas_flex_score = (ccgt_mw / (ccgt_mw + peaker_mw)) * 100
    else:
        gas_flex_score = 50  # neutral if no gas fleet

    # Clean pipeline: (solar+wind+battery) share × 300, capped at 100
    clean_new_mw = sum(
        float(company_row.get(f"{fuel}_mw", 0) or 0)
        for fuel in ["solar", "wind", "battery"]
    )
    if total_cap_mw > 0:
        clean_pipeline_score = min(100, (clean_new_mw / total_cap_mw) * 300)
    else:
        clean_pipeline_score = 0

    return {
        "nuclear_advantage": round(nuclear_score, 1),
        "geographic_diversity": round(diversity_score, 1),
        "coal_exit_progress": round(coal_exit_score, 1),
        "gas_flexibility": round(gas_flex_score, 1),
        "clean_pipeline": round(clean_pipeline_score, 1),
    }


def compute_all_spider_scores(companies_df: pd.DataFrame) -> dict:
    """
    Compute spider scores for all companies.

    Returns:
        dict mapping company_id → {score_name: score_value}
    """
    scores = {}
    for _, row in companies_df.iterrows():
        cid = row["company_id"]
        scores[cid] = compute_spider_scores(row.to_dict())
    return scores


# ══════════════════════════════════════════════════════════════════════
# Fleet Aggregation
# ══════════════════════════════════════════════════════════════════════

def aggregate_fleet_by_fuel(fleet_df: pd.DataFrame, company_id: str) -> pd.DataFrame:
    """Aggregate fleet by fuel type for a single company."""
    mask = fleet_df["company_id"] == company_id
    return (
        fleet_df[mask]
        .groupby("fuel")
        .agg({"cap_mw": "sum", "gen_twh": "sum", "co2_mt": "sum"})
        .reset_index()
    )


def aggregate_fleet_by_iso(fleet_df: pd.DataFrame, company_id: str) -> pd.DataFrame:
    """Aggregate fleet by ISO for a single company."""
    mask = fleet_df["company_id"] == company_id
    return (
        fleet_df[mask]
        .groupby("iso")
        .agg({"cap_mw": "sum", "gen_twh": "sum", "co2_mt": "sum"})
        .reset_index()
    )


def build_company_iso_matrix(fleet_df: pd.DataFrame) -> pd.DataFrame:
    """Build company × ISO capacity matrix (MW)."""
    return fleet_df.pivot_table(
        index="company_id", columns="iso", values="cap_mw",
        aggfunc="sum", fill_value=0,
    ).reindex(COMPANY_ORDER)


def build_company_fuel_matrix(fleet_df: pd.DataFrame) -> pd.DataFrame:
    """Build company × fuel capacity matrix (MW)."""
    return fleet_df.pivot_table(
        index="company_id", columns="fuel", values="cap_mw",
        aggfunc="sum", fill_value=0,
    ).reindex(COMPANY_ORDER)


# ══════════════════════════════════════════════════════════════════════
# Sweep Analysis
# ══════════════════════════════════════════════════════════════════════

def compute_fan_bands(sweep_df: pd.DataFrame, company_id: str) -> dict:
    """
    Compute P10/P50/P90 bands across all scenarios for a company.

    Returns:
        dict with keys for each metric (fleet_emissions_mt, fleet_profit_m,
        stranded_mw, etc.), each containing {years, p10, p50, p90}.
    """
    mask = sweep_df["company"] == company_id
    co_df = sweep_df[mask]

    if co_df.empty:
        return {}

    metrics = ["fleet_emissions_mt", "fleet_profit_m", "fleet_revenue_m",
               "stranded_mw", "operating_mw", "retired_mw"]
    years = sorted(co_df["year"].unique())

    bands = {}
    for metric in metrics:
        p10, p50, p90 = [], [], []
        for yr in years:
            vals = co_df[co_df["year"] == yr][metric].dropna()
            if len(vals) > 0:
                p10.append(round(float(np.percentile(vals, 10)), 2))
                p50.append(round(float(np.percentile(vals, 50)), 2))
                p90.append(round(float(np.percentile(vals, 90)), 2))
            else:
                p10.append(None)
                p50.append(None)
                p90.append(None)
        bands[metric] = {"years": years, "p10": p10, "p50": p50, "p90": p90}

    return bands


def compute_company_kpis(companies_df: pd.DataFrame, sweep_df: pd.DataFrame,
                         company_id: str) -> dict:
    """Compute headline KPIs for a company."""
    co_row = companies_df[companies_df["company_id"] == company_id]
    if co_row.empty:
        return {}
    co = co_row.iloc[0]

    bands = compute_fan_bands(sweep_df, company_id)
    emissions_band = bands.get("fleet_emissions_mt", {})
    profit_band = bands.get("fleet_profit_m", {})
    stranded_band = bands.get("stranded_mw", {})

    # Get last year values
    years = emissions_band.get("years", [])
    last_idx = len(years) - 1 if years else -1

    return {
        "company_id": company_id,
        "short_name": co.get("short_name", company_id),
        "cap_gw": co.get("cap_gw", 0),
        "gen_twh": co.get("gen_twh", 0),
        "co2_2024_mt": co.get("co2_2024_mt", 0),
        "intensity_kg": co.get("intensity_kg", 0),
        "nuclear_pct": round(
            float(co.get("nuclear_mw", 0)) / (float(co.get("cap_gw", 1)) * 1000) * 100, 1
        ) if co.get("cap_gw", 0) else 0,
        "emissions_p50_final": emissions_band.get("p50", [None])[last_idx] if last_idx >= 0 else None,
        "profit_p50_final": profit_band.get("p50", [None])[last_idx] if last_idx >= 0 else None,
        "stranded_p50_final": stranded_band.get("p50", [None])[last_idx] if last_idx >= 0 else None,
    }


def compute_scenario_sensitivity(sweep_df: pd.DataFrame, company_id: str,
                                 metric: str = "fleet_profit_m",
                                 year: int = 2050) -> dict:
    """
    Compute tornado-chart sensitivity: how much each parameter shifts the metric.

    Returns:
        dict mapping parameter_name → {low: delta, high: delta}
    """
    mask = (sweep_df["company"] == company_id) & (sweep_df["year"] == year)
    co_df = sweep_df[mask]

    if co_df.empty:
        return {}

    baseline = co_df[metric].median()
    params = ["demand_growth", "price_sens", "ppa_level", "gas_friction"]
    sensitivity = {}

    for param in params:
        if param not in co_df.columns:
            continue
        vals = co_df[param].unique()
        if len(vals) < 2:
            continue

        param_medians = co_df.groupby(param)[metric].median().sort_index()
        low_val = param_medians.iloc[0] - baseline
        high_val = param_medians.iloc[-1] - baseline
        sensitivity[param] = {"low": round(float(low_val), 1), "high": round(float(high_val), 1)}

    return sensitivity
