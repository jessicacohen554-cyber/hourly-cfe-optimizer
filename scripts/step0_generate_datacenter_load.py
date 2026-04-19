#!/usr/bin/env python3
"""
Generate a research-grade 8,760-hour data center load profile.

This script produces a synthetic hourly data center electricity demand profile
based on well-documented characteristics from peer-reviewed and government sources:

Sources:
  - LBNL 2024 "United States Data Center Energy Usage Report"
    https://eta-publications.lbl.gov/sites/default/files/2024-12/lbnl-2024-united-states-data-center-energy-usage-report.pdf
  - EPRI 2024 "Powering Intelligence: Analyzing AI and Data Center Energy Consumption"
  - DOE/LBNL prototype data center energy models (Shehabi et al., Energy & Buildings, 2020)
    DOI: 10.1016/j.enbuild.2020.110603
  - Google published PUE data (quarterly fleet-wide PUE 1.06-1.12, seasonal variation)
    https://datacenters.google/efficiency/
  - Uptime Institute 2023 Global Data Center Survey (industry PUE distributions)

Data center load profile characteristics (from literature):
  - IT load is nearly flat 24/7/365 (daily load factor 95-99%)
  - Total facility load = IT load x PUE
  - PUE varies seasonally due to cooling energy (higher in summer, lower in winter)
  - Modern hyperscale PUE: 1.08-1.20 (Google fleet: 1.10 avg)
  - Enterprise/colocation PUE: 1.30-1.60
  - Cooling = 25-40% of non-IT overhead
  - Minor diurnal variation from lighting, office areas, shift changes

The generated profile is suitable for:
  - Hourly CFE matching analysis
  - Corporate procurement strategy modeling
  - Grid impact studies
  - As a demand-side input to energy system optimization

Output: data/datacenter_load_profile.csv
  Columns: hour (0-8759), datetime, load_mw, it_load_mw, cooling_load_mw, other_load_mw, pue
"""

import numpy as np
import pandas as pd
import json
import os

# ─── Configuration ───────────────────────────────────────────────────────────

# Facility size (MW of IT load) — 100 MW is representative of a large hyperscale
# campus (e.g., Google, Microsoft, Meta). LBNL 2024 reports individual campuses
# from 50-500+ MW. 100 MW is a common modeling assumption in academic literature.
IT_CAPACITY_MW = 100.0

# Average IT utilization — fraction of nameplate IT capacity actually used.
# Hyperscale facilities run at 60-80% average utilization (LBNL 2024).
# Enterprise facilities are lower (30-50%). We use 70% for a modern hyperscale.
IT_UTILIZATION_MEAN = 0.70

# IT load variation parameters
# Daily variation: ±2-3% from diurnal patterns (batch jobs, traffic patterns)
# Random variation: ±1-2% from stochastic workload fluctuations
IT_DIURNAL_AMPLITUDE = 0.025  # ±2.5% daily swing
IT_RANDOM_STD = 0.01          # 1% random noise (σ)

# PUE parameters — based on Google fleet data and Uptime Institute surveys
# PUE = 1 + (cooling_overhead + electrical_overhead + lighting_overhead) / IT_load
# Seasonal variation driven primarily by cooling energy
PUE_BASE = 1.12              # Annual average PUE (modern hyperscale)
PUE_SEASONAL_AMPLITUDE = 0.04  # ±0.04 seasonal swing (summer peak ~1.16, winter low ~1.08)
PUE_DIURNAL_AMPLITUDE = 0.01   # ±0.01 daily swing (afternoon cooling peak)

# Breakdown of non-IT overhead (as fraction of IT load, summing to PUE - 1)
# These are typical for a modern hyperscale with evaporative + mechanical cooling
COOLING_FRACTION = 0.65       # Cooling is 65% of overhead
ELECTRICAL_FRACTION = 0.25    # UPS, PDU losses = 25% of overhead
OTHER_FRACTION = 0.10         # Lighting, office, misc = 10% of overhead

# Calendar year for datetime index
YEAR = 2024  # Leap year (8784 hours) — we use non-leap 8760 for standard profiles


def generate_datacenter_load_profile(
    it_capacity_mw: float = IT_CAPACITY_MW,
    it_utilization: float = IT_UTILIZATION_MEAN,
    pue_base: float = PUE_BASE,
    pue_seasonal_amp: float = PUE_SEASONAL_AMPLITUDE,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate an 8,760-hour data center load profile.

    Parameters
    ----------
    it_capacity_mw : float
        Nameplate IT capacity in MW.
    it_utilization : float
        Average IT utilization (0-1).
    pue_base : float
        Annual average PUE.
    pue_seasonal_amp : float
        Seasonal PUE variation amplitude.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        8,760-row DataFrame with columns:
        hour, datetime, load_mw, it_load_mw, cooling_load_mw, other_load_mw, pue
    """
    rng = np.random.default_rng(seed)
    n_hours = 8760

    # Hour-of-year index
    hours = np.arange(n_hours)

    # Create datetime index (2024, non-leap subset: Jan 1 – Dec 31, skip Feb 29)
    # Use 2023 as a non-leap year for clean 8760
    dt_index = pd.date_range("2023-01-01", periods=n_hours, freq="h")

    # ─── IT Load ─────────────────────────────────────────────────────────
    # Base IT load
    it_base = it_capacity_mw * it_utilization  # e.g., 70 MW

    # Diurnal pattern: slight increase during business hours (UTC-agnostic)
    # Peak at 14:00 local, trough at 04:00 local
    hour_of_day = hours % 24
    it_diurnal = IT_DIURNAL_AMPLITUDE * np.sin(2 * np.pi * (hour_of_day - 6) / 24)

    # Weekly pattern: ~1% lower on weekends (less interactive traffic)
    day_of_week = (hours // 24) % 7  # 0=Mon for 2023-01-01 (actually Sun)
    # 2023-01-01 is Sunday, so day_of_week 0 = Sunday
    is_weekend = (day_of_week == 0) | (day_of_week == 6)
    weekly_factor = np.where(is_weekend, -0.01, 0.0)

    # Random noise
    it_noise = rng.normal(0, IT_RANDOM_STD, n_hours)

    # Combine: IT load with all variation sources
    it_load = it_base * (1 + it_diurnal + weekly_factor + it_noise)

    # Clip to physical limits
    it_load = np.clip(it_load, it_capacity_mw * 0.5, it_capacity_mw * 0.95)

    # ─── PUE (seasonal + diurnal) ────────────────────────────────────────
    # Seasonal: peak in summer (hour ~4380 = July), trough in winter
    day_of_year = hours / 24.0
    pue_seasonal = pue_seasonal_amp * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    # Shift so peak is around day 200 (mid-July): sin peaks at 90+80=170...
    # Actually: sin(2π(d-80)/365) peaks at d=80+365/4=171.25 ≈ June 20
    # For peak in mid-July (day ~195): shift by ~105
    pue_seasonal = pue_seasonal_amp * np.sin(2 * np.pi * (day_of_year - 105) / 365)

    # Diurnal: afternoon cooling peak
    pue_diurnal = PUE_DIURNAL_AMPLITUDE * np.sin(2 * np.pi * (hour_of_day - 6) / 24)

    # Total PUE
    pue = pue_base + pue_seasonal + pue_diurnal

    # ─── Facility Load Breakdown ─────────────────────────────────────────
    overhead = it_load * (pue - 1)  # Total non-IT load
    cooling_load = overhead * COOLING_FRACTION
    electrical_loss = overhead * ELECTRICAL_FRACTION
    other_load = overhead * OTHER_FRACTION

    # Total facility load
    total_load = it_load * pue  # = it_load + overhead

    # ─── Build DataFrame ─────────────────────────────────────────────────
    df = pd.DataFrame({
        "hour": hours,
        "datetime": dt_index,
        "load_mw": np.round(total_load, 3),
        "it_load_mw": np.round(it_load, 3),
        "cooling_load_mw": np.round(cooling_load, 3),
        "other_load_mw": np.round(other_load + electrical_loss, 3),
        "pue": np.round(pue, 4),
    })

    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print key statistics about the generated profile."""
    print("=" * 60)
    print("DATA CENTER LOAD PROFILE SUMMARY")
    print("=" * 60)
    print(f"  Hours:              {len(df):,}")
    print(f"  IT Capacity:        {IT_CAPACITY_MW:.0f} MW")
    print(f"  Avg IT Utilization:  {IT_UTILIZATION_MEAN:.0%}")
    print()
    print("  Total Facility Load:")
    print(f"    Mean:             {df['load_mw'].mean():.1f} MW")
    print(f"    Min:              {df['load_mw'].min():.1f} MW")
    print(f"    Max:              {df['load_mw'].max():.1f} MW")
    print(f"    Std Dev:          {df['load_mw'].std():.2f} MW")
    print(f"    Load Factor:      {df['load_mw'].mean() / df['load_mw'].max():.1%}")
    print(f"    Annual Energy:    {df['load_mw'].sum() / 1000:.1f} GWh")
    print()
    print("  IT Load:")
    print(f"    Mean:             {df['it_load_mw'].mean():.1f} MW")
    print(f"    Min:              {df['it_load_mw'].min():.1f} MW")
    print(f"    Max:              {df['it_load_mw'].max():.1f} MW")
    print()
    print("  PUE:")
    print(f"    Mean:             {df['pue'].mean():.3f}")
    print(f"    Min (winter):     {df['pue'].min():.3f}")
    print(f"    Max (summer):     {df['pue'].max():.3f}")
    print()
    print("  Load Breakdown (annual avg):")
    print(f"    IT:               {df['it_load_mw'].mean():.1f} MW ({df['it_load_mw'].mean()/df['load_mw'].mean():.1%})")
    print(f"    Cooling:          {df['cooling_load_mw'].mean():.1f} MW ({df['cooling_load_mw'].mean()/df['load_mw'].mean():.1%})")
    print(f"    Other (UPS/misc): {df['other_load_mw'].mean():.1f} MW ({df['other_load_mw'].mean()/df['load_mw'].mean():.1%})")
    print("=" * 60)


def save_metadata(df: pd.DataFrame, output_dir: str) -> None:
    """Save profile metadata as JSON for downstream pipeline consumption."""
    metadata = {
        "description": "Synthetic data center 8760-hour load profile",
        "sources": [
            "LBNL 2024 United States Data Center Energy Usage Report",
            "EPRI 2024 Powering Intelligence Report",
            "DOE/LBNL Prototype Data Center Energy Models (Shehabi et al. 2020)",
            "Google published PUE data (2023-2025)",
            "Uptime Institute 2023 Global Data Center Survey",
        ],
        "parameters": {
            "it_capacity_mw": IT_CAPACITY_MW,
            "it_utilization_mean": IT_UTILIZATION_MEAN,
            "pue_base": PUE_BASE,
            "pue_seasonal_amplitude": PUE_SEASONAL_AMPLITUDE,
            "diurnal_amplitude": IT_DIURNAL_AMPLITUDE,
            "random_seed": 42,
        },
        "statistics": {
            "hours": len(df),
            "mean_load_mw": round(df["load_mw"].mean(), 2),
            "min_load_mw": round(df["load_mw"].min(), 2),
            "max_load_mw": round(df["load_mw"].max(), 2),
            "load_factor": round(df["load_mw"].mean() / df["load_mw"].max(), 4),
            "annual_energy_gwh": round(df["load_mw"].sum() / 1000, 2),
            "mean_pue": round(df["pue"].mean(), 4),
            "mean_it_load_mw": round(df["it_load_mw"].mean(), 2),
            "mean_cooling_load_mw": round(df["cooling_load_mw"].mean(), 2),
        },
        "column_descriptions": {
            "hour": "Hour index (0-8759)",
            "datetime": "ISO 8601 datetime (2023 calendar year, hourly)",
            "load_mw": "Total facility electrical load (MW)",
            "it_load_mw": "IT equipment load (MW) — servers, networking, storage",
            "cooling_load_mw": "Cooling system load (MW) — chillers, CRAHs, pumps",
            "other_load_mw": "Other facility load (MW) — UPS losses, PDUs, lighting, office",
            "pue": "Power Usage Effectiveness (total / IT)",
        },
    }

    path = os.path.join(output_dir, "datacenter_load_metadata.json")
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {path}")


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)

    # Generate profile
    df = generate_datacenter_load_profile()

    # Print summary
    print_summary(df)

    # Save CSV
    csv_path = os.path.join(output_dir, "datacenter_load_profile.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nProfile saved to {csv_path}")

    # Save metadata
    save_metadata(df, output_dir)

    print("\nDone.")
