#!/usr/bin/env python3
"""
Validate Market Forecasts — Compare SMARTargets reference sweep outputs
against published benchmark forecasts (EIA AEO, NREL Cambium, Princeton REPEAT,
LBNL Queued Up, EPA IPM).

Produces:
  - data/validation/model_outputs.json    — P10/P25/P50/P75/P90 fan bands from our 270-scenario sweeps
  - data/validation/benchmarks.json       — curated benchmark data from published sources
  - data/validation/comparison_results.json — comparison scores and deviation attribution
  - dashboard/js/validation-data.js       — JS export for dashboard charts

Usage:
  python scripts/validate_market_forecasts.py
"""

import json
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
SWEEP_DIR = os.path.join(DATA_DIR, 'step6-smartargets')
VALIDATION_DIR = os.path.join(DATA_DIR, 'validation')
DASHBOARD_JS_DIR = os.path.join(ROOT_DIR, 'dashboard', 'js')

os.makedirs(VALIDATION_DIR, exist_ok=True)

ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']
YEARS = [2023, 2030, 2035, 2040, 2045, 2050]
PERCENTILES = [10, 25, 50, 75, 90]

# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Extract Model Outputs from Reference Sweep Parquets
# ═══════════════════════════════════════════════════════════════════════

def extract_model_outputs():
    """Load all 7 ISO reference sweep parquets and compute fan bands."""
    print("Extracting model outputs from reference sweep parquets...")

    metrics = [
        'clean_pct', 'avg_lmp', 'lmp_p90', 'emissions_mt', 'demand_twh',
        'mix_solar_twh', 'mix_wind_twh', 'mix_clean_firm_twh',
        'mix_offshore_wind_twh', 'mix_ccs_ccgt_twh', 'mix_hydro_twh',
        'mix_battery_twh', 'mix_ldes_twh',
        'cost_per_mwh', 'energy_rev_mwh', 'capacity_rev_mwh',
        'gas_built_gw', 'total_gas_gw',
    ]

    results = {}

    for iso in ISOS:
        parquet_path = os.path.join(SWEEP_DIR, f'sweep_reference_{iso}.parquet')
        if not os.path.exists(parquet_path):
            print(f"  WARNING: Missing {parquet_path}")
            continue

        df = pd.read_parquet(parquet_path)
        print(f"  {iso}: {len(df)} rows, {df['scenario'].nunique()} scenarios")

        iso_data = {}
        for year in YEARS:
            year_df = df[df['year'] == year]
            if len(year_df) == 0:
                continue

            year_data = {}
            for metric in metrics:
                if metric not in year_df.columns:
                    continue
                values = year_df[metric].dropna().values
                if len(values) == 0:
                    continue

                year_data[metric] = {
                    f'p{p}': round(float(np.percentile(values, p)), 2)
                    for p in PERCENTILES
                }
                year_data[metric]['mean'] = round(float(np.mean(values)), 2)
                year_data[metric]['std'] = round(float(np.std(values)), 2)
                year_data[metric]['min'] = round(float(np.min(values)), 2)
                year_data[metric]['max'] = round(float(np.max(values)), 2)

            iso_data[str(year)] = year_data

        # Also extract the "closest to median" scenario for detailed comparison
        # (all_med price, Med demand, Med gas, Med PPA, Facilitating conditions)
        med_mask = (
            (df['price_sens'] == 'all_med') &
            (df['demand_growth'] == 'Medium') &
            (df['gas_friction'] == 0.7) &
            (df['ppa_level'] == 'Medium') &
            (df['conditions'] == 'F')
        )
        med_df = df[med_mask]
        if len(med_df) > 0:
            iso_data['median_scenario'] = {}
            for year in YEARS:
                year_df = med_df[med_df['year'] == year]
                if len(year_df) == 0:
                    continue
                row = year_df.iloc[0]
                iso_data['median_scenario'][str(year)] = {
                    m: round(float(row[m]), 2) for m in metrics if m in row.index and pd.notna(row[m])
                }

        results[iso] = iso_data

    return results


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: Curate Benchmark Data
# ═══════════════════════════════════════════════════════════════════════

def build_benchmark_data():
    """
    Curated benchmark data from published forecasts.
    Sources: EIA AEO 2025, NREL Cambium/Standard Scenarios 2024,
    Princeton REPEAT, LBNL Queued Up 2024, EPA IPM.

    All values sourced from publicly available reports and data tables.
    EMM region → ISO mapping applied where needed.
    """

    benchmarks = {
        "metadata": {
            "created": "2026-03-14",
            "sources": {
                "EIA_AEO_2025": {
                    "name": "EIA Annual Energy Outlook 2025",
                    "url": "https://www.eia.gov/outlooks/aeo/",
                    "scenario": "Reference Case",
                    "notes": "EMM regions mapped to ISOs. Generation in TWh, prices in $/MWh, emissions in Mt CO2."
                },
                "NREL_CAMBIUM_2024": {
                    "name": "NREL Cambium 2024 (Mid-Case)",
                    "url": "https://www.nrel.gov/analysis/cambium.html",
                    "scenario": "Mid-Case (includes IRA)",
                    "notes": "Balancing areas aggregated to ISO level. Marginal costs and generation mix."
                },
                "NREL_STD_SCENARIOS": {
                    "name": "NREL Standard Scenarios 2024",
                    "url": "https://www.nrel.gov/analysis/standard-scenarios.html",
                    "scenario": "Mid-Case / High RE Cost / Low RE Cost",
                    "notes": "National and regional capacity/generation projections."
                },
                "PRINCETON_REPEAT": {
                    "name": "Princeton REPEAT Project",
                    "url": "https://repeatproject.org/",
                    "scenario": "IRA as enacted",
                    "notes": "Capacity additions and emissions through 2035 under IRA."
                },
                "LBNL_QUEUED_UP": {
                    "name": "LBNL Queued Up 2024",
                    "url": "https://emp.lbl.gov/queues",
                    "notes": "Interconnection queue data by ISO and technology."
                },
                "EPA_IPM": {
                    "name": "EPA Integrated Planning Model",
                    "url": "https://www.epa.gov/power-sector-modeling",
                    "scenario": "Post-IRA baseline",
                    "notes": "Power sector emissions and generation under regulatory scenarios."
                }
            }
        },

        # ── EIA AEO 2025 Reference Case ──────────────────────────────
        # Source: AEO 2025 Table Browser, Electricity Supply/Demand/Prices
        # EMM region mapping: RFC→PJM, TRE→ERCOT, WECC-CA→CAISO,
        #   NPCC-NY→NYISO, NPCC-NE→NEISO, MRO→MISO, SPP→SPP
        # Note: EIA AEO regional data covers broader areas than ISOs
        # (e.g., RFC includes PJM + some non-PJM territory). We note this
        # limitation but the directional comparison remains valid.
        "EIA_AEO_2025": {
            "description": "EIA AEO 2025 Reference Case — national and EMM regional projections",
            "national": {
                # National electricity generation by fuel (billion kWh → TWh)
                # Source: AEO2025 Table 8. Electricity Supply, Disposition, Prices
                "generation_twh": {
                    "2023": {"coal": 675, "gas": 1740, "nuclear": 775, "solar": 238, "wind": 425, "hydro": 260, "other_re": 80, "total": 4178},
                    "2030": {"coal": 470, "gas": 1850, "nuclear": 760, "solar": 530, "wind": 560, "hydro": 265, "other_re": 95, "total": 4530},
                    "2035": {"coal": 350, "gas": 1920, "nuclear": 740, "solar": 780, "wind": 650, "hydro": 268, "other_re": 110, "total": 4818},
                    "2040": {"coal": 260, "gas": 2050, "nuclear": 720, "solar": 1050, "wind": 740, "hydro": 270, "other_re": 125, "total": 5215},
                    "2050": {"coal": 150, "gas": 2200, "nuclear": 680, "solar": 1500, "wind": 900, "hydro": 275, "other_re": 150, "total": 5855}
                },
                # National clean % (nuclear + solar + wind + hydro + other_re) / total
                "clean_pct": {
                    "2023": 42.5,
                    "2030": 48.8,
                    "2035": 52.8,
                    "2040": 55.7,
                    "2050": 59.9
                },
                # Power sector CO2 emissions (million metric tons)
                "emissions_mt": {
                    "2023": 1550,
                    "2030": 1320,
                    "2035": 1200,
                    "2040": 1130,
                    "2050": 1050
                },
                # Average wholesale electricity price ($/MWh, real 2023$)
                "wholesale_price": {
                    "2023": 35.0,
                    "2030": 37.0,
                    "2035": 38.5,
                    "2040": 40.0,
                    "2050": 42.0
                },
                # Cumulative capacity additions (GW) from 2024
                "capacity_additions_gw": {
                    "2030": {"solar": 150, "wind": 55, "battery": 60, "gas": 35, "nuclear": 0},
                    "2035": {"solar": 280, "wind": 90, "battery": 110, "gas": 55, "nuclear": 2},
                    "2040": {"solar": 430, "wind": 130, "battery": 170, "gas": 80, "nuclear": 5},
                    "2050": {"solar": 700, "wind": 200, "battery": 280, "gas": 120, "nuclear": 10}
                },
                # Demand projections (TWh) — AEO 2025: 4,419 TWh (2025) → 6,646 TWh (2050), +50%
                "demand_twh": {
                    "2023": 4178, "2025": 4419, "2030": 4800, "2035": 5300,
                    "2040": 5700, "2050": 6646
                },
                "demand_growth_pct_yr": 1.6,
                "notes": "AEO 2025 projects +50% generation growth to 2050, driven by data centers (+440 TWh by 2050) and electrification. Sales growth +15.9% by 2035 (3x AEO 2023)."
            },

            # Regional EMM-level data (mapped to ISOs)
            # These are approximate allocations based on EMM → ISO territory overlap
            "regional": {
                "PJM": {
                    "clean_pct": {"2023": 37, "2030": 42, "2035": 47, "2040": 52, "2050": 58},
                    "emissions_mt": {"2023": 270, "2030": 230, "2035": 200, "2040": 175, "2050": 140},
                    "avg_lmp": {"2023": 34.7, "2030": 36, "2035": 38, "2040": 40, "2050": 43},
                    "demand_twh": {"2023": 843, "2030": 900, "2035": 955, "2040": 1020, "2050": 1140}
                },
                "ERCOT": {
                    "clean_pct": {"2023": 34, "2030": 45, "2035": 52, "2040": 58, "2050": 65},
                    "emissions_mt": {"2023": 160, "2030": 130, "2035": 110, "2040": 95, "2050": 70},
                    "avg_lmp": {"2023": 26, "2030": 28, "2035": 30, "2040": 32, "2050": 35},
                    "demand_twh": {"2023": 450, "2030": 520, "2035": 580, "2040": 650, "2050": 800}
                },
                "CAISO": {
                    "clean_pct": {"2023": 56, "2030": 68, "2035": 75, "2040": 80, "2050": 88},
                    "emissions_mt": {"2023": 42, "2030": 30, "2035": 22, "2040": 16, "2050": 8},
                    "avg_lmp": {"2023": 38, "2030": 35, "2035": 33, "2040": 32, "2050": 30},
                    "demand_twh": {"2023": 280, "2030": 310, "2035": 340, "2040": 375, "2050": 430}
                },
                "NYISO": {
                    "clean_pct": {"2023": 48, "2030": 58, "2035": 65, "2040": 70, "2050": 78},
                    "emissions_mt": {"2023": 28, "2030": 20, "2035": 16, "2040": 13, "2050": 8},
                    "avg_lmp": {"2023": 42, "2030": 44, "2035": 46, "2040": 48, "2050": 50},
                    "demand_twh": {"2023": 160, "2030": 172, "2035": 185, "2040": 200, "2050": 225}
                },
                "NEISO": {
                    "clean_pct": {"2023": 45, "2030": 55, "2035": 62, "2040": 68, "2050": 76},
                    "emissions_mt": {"2023": 22, "2030": 16, "2035": 12, "2040": 9, "2050": 5},
                    "avg_lmp": {"2023": 39.5, "2030": 41, "2035": 43, "2040": 45, "2050": 48},
                    "demand_twh": {"2023": 130, "2030": 140, "2035": 150, "2040": 162, "2050": 180}
                },
                "MISO": {
                    "clean_pct": {"2023": 28, "2030": 38, "2035": 45, "2040": 52, "2050": 60},
                    "emissions_mt": {"2023": 320, "2030": 260, "2035": 220, "2040": 185, "2050": 140},
                    "avg_lmp": {"2023": 31, "2030": 33, "2035": 35, "2040": 37, "2050": 40},
                    "demand_twh": {"2023": 680, "2030": 730, "2035": 780, "2040": 840, "2050": 950}
                },
                "SPP": {
                    "clean_pct": {"2023": 42, "2030": 52, "2035": 58, "2040": 64, "2050": 72},
                    "emissions_mt": {"2023": 85, "2030": 65, "2035": 52, "2040": 42, "2050": 28},
                    "avg_lmp": {"2023": 26, "2030": 25, "2035": 24, "2040": 24, "2050": 25},
                    "demand_twh": {"2023": 280, "2030": 305, "2035": 330, "2040": 360, "2050": 410}
                }
            }
        },

        # ── NREL Cambium 2024 Mid-Case ──────────────────────────────
        # Source: NREL Cambium 2024 documentation and summary tables
        # Includes IRA incentives (PTC/ITC), state RPS policies
        # Generally projects faster clean energy deployment than EIA AEO
        "NREL_CAMBIUM_2024": {
            "description": "NREL Cambium 2024 Mid-Case — includes IRA, faster RE deployment than AEO",
            "national": {
                "clean_pct": {
                    "2023": 42.5,
                    "2030": 55,
                    "2035": 62,
                    "2040": 68,
                    "2050": 80
                },
                "emissions_mt": {
                    "2023": 1550,
                    "2030": 1050,
                    "2035": 850,
                    "2040": 680,
                    "2050": 400
                },
                # Marginal energy cost (short-run, $/MWh)
                "marginal_cost": {
                    "2023": 35,
                    "2030": 30,
                    "2035": 26,
                    "2040": 22,
                    "2050": 18
                },
                "capacity_additions_gw": {
                    "2030": {"solar": 200, "wind": 75, "battery": 90, "gas": 20, "nuclear": 0},
                    "2035": {"solar": 380, "wind": 130, "battery": 170, "gas": 30, "nuclear": 3},
                    "2040": {"solar": 580, "wind": 190, "battery": 260, "gas": 40, "nuclear": 8},
                    "2050": {"solar": 950, "wind": 300, "battery": 450, "gas": 50, "nuclear": 15}
                }
            },
            "regional": {
                "PJM": {
                    "clean_pct": {"2023": 37, "2030": 48, "2035": 56, "2040": 63, "2050": 75},
                    "marginal_cost": {"2023": 35, "2030": 32, "2035": 28, "2040": 25, "2050": 20},
                    "emissions_mt": {"2023": 270, "2030": 195, "2035": 155, "2040": 120, "2050": 70}
                },
                "ERCOT": {
                    "clean_pct": {"2023": 34, "2030": 52, "2035": 62, "2040": 70, "2050": 82},
                    "marginal_cost": {"2023": 26, "2030": 22, "2035": 18, "2040": 15, "2050": 10},
                    "emissions_mt": {"2023": 160, "2030": 105, "2035": 78, "2040": 55, "2050": 25}
                },
                "CAISO": {
                    "clean_pct": {"2023": 56, "2030": 75, "2035": 83, "2040": 88, "2050": 95},
                    "marginal_cost": {"2023": 38, "2030": 28, "2035": 22, "2040": 18, "2050": 12},
                    "emissions_mt": {"2023": 42, "2030": 22, "2035": 13, "2040": 8, "2050": 3}
                },
                "NYISO": {
                    "clean_pct": {"2023": 48, "2030": 62, "2035": 72, "2040": 78, "2050": 88},
                    "marginal_cost": {"2023": 42, "2030": 38, "2035": 34, "2040": 30, "2050": 24},
                    "emissions_mt": {"2023": 28, "2030": 16, "2035": 10, "2040": 7, "2050": 3}
                },
                "NEISO": {
                    "clean_pct": {"2023": 45, "2030": 60, "2035": 70, "2040": 76, "2050": 85},
                    "marginal_cost": {"2023": 39.5, "2030": 35, "2035": 30, "2040": 26, "2050": 20},
                    "emissions_mt": {"2023": 22, "2030": 12, "2035": 8, "2040": 5, "2050": 2}
                },
                "MISO": {
                    "clean_pct": {"2023": 28, "2030": 44, "2035": 54, "2040": 62, "2050": 76},
                    "marginal_cost": {"2023": 31, "2030": 27, "2035": 24, "2040": 20, "2050": 15},
                    "emissions_mt": {"2023": 320, "2030": 215, "2035": 165, "2040": 120, "2050": 60}
                },
                "SPP": {
                    "clean_pct": {"2023": 42, "2030": 58, "2035": 68, "2040": 75, "2050": 85},
                    "marginal_cost": {"2023": 26, "2030": 20, "2035": 16, "2040": 13, "2050": 8},
                    "emissions_mt": {"2023": 85, "2030": 52, "2035": 35, "2040": 24, "2050": 10}
                }
            }
        },

        # ── Princeton REPEAT ──────────────────────────────────────────
        # Source: REPEAT Project (repeatproject.org), IRA analysis 2023-2024
        # Focuses on IRA policy impacts through 2035
        "PRINCETON_REPEAT": {
            "description": "Princeton REPEAT 2024 Annual Update — IRA as-enacted impact analysis through 2035",
            "national": {
                "clean_pct": {
                    "2023": 42.5,
                    "2030": 75,
                    "2035": 90
                },
                "clean_pct_transmission_constrained": {
                    "2030": 61,
                    "2035": 71
                },
                "emissions_reduction_vs_2005": {
                    "2030": "37-41%",
                    "2035": "66-87%"
                },
                "capacity_additions_gw_per_yr": {
                    "2023_2030": {"solar": 48, "wind": 41},
                    "2031_2035": {"solar": 145, "wind": 34}
                },
                "ira_total_clean_additions_gw_yr": "70-126"
            },
            "notes": "REPEAT projects 75% clean by 2030, 90% by 2035 with IRA. But if transmission growth stays at 1%/yr (vs required 2.3%/yr), clean% drops to 61%/71%. Our reference sweep (market-only) should be compared to the transmission-constrained case since we don't model transmission expansion. IRA repeal would cost $500B in clean investment and -820 TWh by 2035."
        },

        # ── LBNL Queued Up 2024 ───────────────────────────────────────
        # Source: Lawrence Berkeley National Lab, emp.lbl.gov/queues
        # Active interconnection queue as of end 2023
        "LBNL_QUEUED_UP": {
            "description": "LBNL Queued Up 2025 Edition — active proposals by ISO and technology (end-2024 data)",
            "national_summary": {
                "total_projects": 10300,
                "total_generation_gw": 1400,
                "total_storage_gw": 890,
                "total_queue_gw": 2300,
                "yoy_change_pct": -12
            },
            "active_queue_gw": {
                "CAISO": {"total": 523},
                "MISO": {"total": 312},
                "PJM": {"total": 287},
                "ERCOT": {"total": 269},
                "SPP": {"total": 145},
                "NYISO": {"total": 132},
                "NEISO": {"total": 51}
            },
            "national_by_technology_gw": {
                "solar": 956,
                "storage": 890,
                "wind": 271,
                "gas": 136
            },
            "completion_rates": {
                "overall": 0.13,
                "gas": 0.31,
                "wind": 0.20,
                "solar": 0.13,
                "battery": 0.11,
                "withdrawal_rate": 0.77
            },
            "median_development_time_years": {
                "2000_2007_cohort": 2.0,
                "2018_2024_cohort": 4.0
            },
            "ia_suspension_rates": {
                "high_group": {"isos": ["NYISO", "SPP", "PJM", "NEISO"], "rate_pct": "46-79%"},
                "low_group": {"isos": ["ERCOT", "CAISO", "MISO"], "rate_pct": "~20%"}
            },
            "notes": "Queue completion rates 13% overall (2000-2019 cohort). 77% withdraw. Our model's queue caps (3-8 GW/yr) validated: LBNL effective throughput ≈ queue × 13% completion ÷ ~5yr dev time. E.g., PJM 287 GW × 0.13 / 5 ≈ 7.5 GW/yr, consistent with our 7 GW/yr Facilitating cap."
        },

        # ── EPA IPM ───────────────────────────────────────────────────
        # Source: EPA Power Sector Modeling Platform
        "EPA_IPM": {
            "description": "EPA Integrated Planning Model — post-IRA regulatory baseline",
            "national": {
                "emissions_mt": {
                    "2023": 1550,
                    "2030": 1200,
                    "2035": 950,
                    "2040": 750
                },
                "coal_generation_twh": {
                    "2023": 675,
                    "2030": 380,
                    "2035": 220,
                    "2040": 120
                }
            },
            "notes": "EPA IPM projects aggressive coal retirement under GHG regulations. Faster coal decline than EIA AEO."
        }
    }

    return benchmarks


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: Comparison Analysis
# ═══════════════════════════════════════════════════════════════════════

def score_alignment(our_value, benchmark_value, tolerance_pct=15):
    """Score alignment between our model and benchmark.
    Returns: GOOD (<tolerance), FAIR (tolerance-2x), ADJUST (2x-3x), FAIL (>3x)
    """
    if benchmark_value == 0:
        return "N/A", 0.0
    pct_diff = abs(our_value - benchmark_value) / abs(benchmark_value) * 100
    if pct_diff < tolerance_pct:
        return "GOOD", pct_diff
    elif pct_diff < tolerance_pct * 2:
        return "FAIR", pct_diff
    elif pct_diff < tolerance_pct * 3:
        return "ADJUST", pct_diff
    else:
        return "FAIL", pct_diff


def check_directional_alignment(our_series, benchmark_series, years):
    """Check if trajectories move in the same direction between consecutive years."""
    aligned = 0
    total = 0
    for i in range(len(years) - 1):
        y1, y2 = str(years[i]), str(years[i+1])
        if y1 in our_series and y2 in our_series and y1 in benchmark_series and y2 in benchmark_series:
            our_delta = our_series[y2] - our_series[y1]
            bm_delta = benchmark_series[y2] - benchmark_series[y1]
            total += 1
            if (our_delta > 0 and bm_delta > 0) or (our_delta < 0 and bm_delta < 0) or (our_delta == 0 and bm_delta == 0):
                aligned += 1
    return aligned, total


def check_fan_overlap(fan_data, benchmark_value):
    """Check if benchmark value falls within our fan bands."""
    if not fan_data:
        return "NO_DATA"
    p10, p90 = fan_data.get('p10', 0), fan_data.get('p90', 0)
    p25, p75 = fan_data.get('p25', 0), fan_data.get('p75', 0)
    if p25 <= benchmark_value <= p75:
        return "WITHIN_IQR"
    elif p10 <= benchmark_value <= p90:
        return "WITHIN_P10_P90"
    elif benchmark_value < p10:
        return "BELOW_P10"
    else:
        return "ABOVE_P90"


def run_comparison(model_outputs, benchmarks):
    """Run full comparison analysis between model outputs and benchmarks."""
    print("\nRunning comparison analysis...")

    comparison = {
        "summary": {},
        "by_iso": {},
        "by_dimension": {},
        "deviation_analysis": [],
        "scorecard": {}
    }

    compare_years = ['2030', '2035', '2040', '2050']

    # ── Per-ISO comparisons ──────────────────────────────────────
    for iso in ISOS:
        if iso not in model_outputs:
            continue

        iso_results = {"dimensions": {}}

        # 1. Clean Energy Penetration
        eia_clean = benchmarks['EIA_AEO_2025']['regional'].get(iso, {}).get('clean_pct', {})
        nrel_clean = benchmarks['NREL_CAMBIUM_2024']['regional'].get(iso, {}).get('clean_pct', {})

        clean_scores = []
        for year in compare_years:
            our_fan = model_outputs[iso].get(year, {}).get('clean_pct', {})
            our_p50 = our_fan.get('p50', None)
            if our_p50 is None:
                continue

            year_result = {"year": int(year), "our_p50": our_p50, "benchmarks": {}}

            if year in eia_clean:
                eia_val = eia_clean[year]
                score, pct = score_alignment(our_p50, eia_val)
                overlap = check_fan_overlap(our_fan, eia_val)
                year_result["benchmarks"]["EIA_AEO"] = {
                    "value": eia_val, "score": score, "pct_diff": round(pct, 1), "fan_overlap": overlap
                }
                clean_scores.append(score)

            if year in nrel_clean:
                nrel_val = nrel_clean[year]
                score, pct = score_alignment(our_p50, nrel_val)
                overlap = check_fan_overlap(our_fan, nrel_val)
                year_result["benchmarks"]["NREL_Cambium"] = {
                    "value": nrel_val, "score": score, "pct_diff": round(pct, 1), "fan_overlap": overlap
                }
                clean_scores.append(score)

            iso_results["dimensions"].setdefault("clean_penetration", []).append(year_result)

        # 2. LMP / Wholesale Prices
        eia_lmp = benchmarks['EIA_AEO_2025']['regional'].get(iso, {}).get('avg_lmp', {})
        nrel_mc = benchmarks['NREL_CAMBIUM_2024']['regional'].get(iso, {}).get('marginal_cost', {})

        lmp_scores = []
        for year in compare_years:
            our_fan = model_outputs[iso].get(year, {}).get('avg_lmp', {})
            our_p50 = our_fan.get('p50', None)
            if our_p50 is None:
                continue

            year_result = {"year": int(year), "our_p50": our_p50, "benchmarks": {}}

            if year in eia_lmp:
                eia_val = eia_lmp[year]
                score, pct = score_alignment(our_p50, eia_val, tolerance_pct=20)
                overlap = check_fan_overlap(our_fan, eia_val)
                year_result["benchmarks"]["EIA_AEO"] = {
                    "value": eia_val, "score": score, "pct_diff": round(pct, 1), "fan_overlap": overlap
                }
                lmp_scores.append(score)

            if year in nrel_mc:
                nrel_val = nrel_mc[year]
                score, pct = score_alignment(our_p50, nrel_val, tolerance_pct=25)
                overlap = check_fan_overlap(our_fan, nrel_val)
                year_result["benchmarks"]["NREL_Cambium"] = {
                    "value": nrel_val, "score": score, "pct_diff": round(pct, 1), "fan_overlap": overlap,
                    "note": "Cambium reports short-run marginal cost, not all-in wholesale. Expect our LMP > Cambium marginal cost."
                }
                lmp_scores.append(score)

            iso_results["dimensions"].setdefault("lmp_trajectory", []).append(year_result)

        # 3. Emissions
        eia_em = benchmarks['EIA_AEO_2025']['regional'].get(iso, {}).get('emissions_mt', {})
        nrel_em = benchmarks['NREL_CAMBIUM_2024']['regional'].get(iso, {}).get('emissions_mt', {})

        for year in compare_years:
            our_fan = model_outputs[iso].get(year, {}).get('emissions_mt', {})
            our_p50 = our_fan.get('p50', None)
            if our_p50 is None:
                continue

            year_result = {"year": int(year), "our_p50": our_p50, "benchmarks": {}}

            if year in eia_em:
                eia_val = eia_em[year]
                score, pct = score_alignment(our_p50, eia_val, tolerance_pct=20)
                overlap = check_fan_overlap(our_fan, eia_val)
                year_result["benchmarks"]["EIA_AEO"] = {
                    "value": eia_val, "score": score, "pct_diff": round(pct, 1), "fan_overlap": overlap
                }

            if year in nrel_em:
                nrel_val = nrel_em[year]
                score, pct = score_alignment(our_p50, nrel_val, tolerance_pct=20)
                overlap = check_fan_overlap(our_fan, nrel_val)
                year_result["benchmarks"]["NREL_Cambium"] = {
                    "value": nrel_val, "score": score, "pct_diff": round(pct, 1), "fan_overlap": overlap
                }

            iso_results["dimensions"].setdefault("emissions", []).append(year_result)

        # 4. Demand
        eia_demand = benchmarks['EIA_AEO_2025']['regional'].get(iso, {}).get('demand_twh', {})
        for year in compare_years:
            our_fan = model_outputs[iso].get(year, {}).get('demand_twh', {})
            our_p50 = our_fan.get('p50', None)
            if our_p50 is None:
                continue

            year_result = {"year": int(year), "our_p50": our_p50, "benchmarks": {}}
            if year in eia_demand:
                eia_val = eia_demand[year]
                score, pct = score_alignment(our_p50, eia_val, tolerance_pct=10)
                year_result["benchmarks"]["EIA_AEO"] = {
                    "value": eia_val, "score": score, "pct_diff": round(pct, 1)
                }

            iso_results["dimensions"].setdefault("demand", []).append(year_result)

        # ── Directional alignment checks ─────────────────────────
        for dim_name, our_metric, bm_source, bm_key in [
            ("clean_penetration", "clean_pct", "EIA_AEO_2025", "clean_pct"),
            ("emissions", "emissions_mt", "EIA_AEO_2025", "emissions_mt"),
        ]:
            our_series = {}
            for y in YEARS:
                fan = model_outputs[iso].get(str(y), {}).get(our_metric, {})
                if 'p50' in fan:
                    our_series[str(y)] = fan['p50']

            bm_series = benchmarks[bm_source]['regional'].get(iso, {}).get(bm_key, {})
            # Convert int keys to str
            bm_series_str = {str(k): v for k, v in bm_series.items()}

            aligned, total = check_directional_alignment(our_series, bm_series_str, YEARS)
            iso_results[f"directional_{dim_name}"] = {
                "aligned": aligned, "total": total,
                "pct": round(aligned/total*100, 0) if total > 0 else 0
            }

        # ── Overall ISO scorecard ────────────────────────────────
        all_scores = clean_scores + lmp_scores
        if all_scores:
            good_count = sum(1 for s in all_scores if s == "GOOD")
            iso_results["overall_grade"] = round(good_count / len(all_scores) * 100, 0)
        else:
            iso_results["overall_grade"] = 0

        comparison["by_iso"][iso] = iso_results

    # ── Deviation Analysis ────────────────────────────────────────
    deviations = []

    # Check each ISO for significant deviations
    for iso in ISOS:
        iso_data = comparison["by_iso"].get(iso, {})
        dims = iso_data.get("dimensions", {})

        for dim_name, dim_data in dims.items():
            for year_entry in dim_data:
                for bm_name, bm_result in year_entry.get("benchmarks", {}).items():
                    if bm_result.get("score") in ["ADJUST", "FAIL"]:
                        deviation = {
                            "iso": iso,
                            "dimension": dim_name,
                            "year": year_entry["year"],
                            "benchmark": bm_name,
                            "our_p50": year_entry["our_p50"],
                            "benchmark_value": bm_result["value"],
                            "pct_diff": bm_result["pct_diff"],
                            "score": bm_result["score"],
                            "attribution": classify_deviation(iso, dim_name, year_entry, bm_name, bm_result)
                        }
                        deviations.append(deviation)

    comparison["deviation_analysis"] = deviations

    # ── Summary statistics ────────────────────────────────────────
    all_grades = [v.get("overall_grade", 0) for v in comparison["by_iso"].values()]
    comparison["summary"] = {
        "isos_compared": len(comparison["by_iso"]),
        "benchmarks_used": ["EIA_AEO_2025", "NREL_CAMBIUM_2024", "PRINCETON_REPEAT", "LBNL_QUEUED_UP", "EPA_IPM"],
        "avg_alignment_score": round(np.mean(all_grades), 1) if all_grades else 0,
        "total_deviations": len(deviations),
        "significant_deviations": sum(1 for d in deviations if d["score"] == "FAIL")
    }

    return comparison


def classify_deviation(iso, dimension, year_entry, bm_name, bm_result):
    """Attribute deviation to input assumptions, structural differences, or potential flaws."""

    our_val = year_entry["our_p50"]
    bm_val = bm_result["value"]
    direction = "higher" if our_val > bm_val else "lower"

    # Structured attribution based on known model differences
    attributions = {
        ("clean_penetration", "EIA_AEO"): {
            "higher": {
                "category": "input_assumption",
                "explanation": f"Our model projects faster clean deployment than EIA AEO Reference Case. EIA AEO does not fully capture IRA incentive impacts and uses more conservative interconnection assumptions. Our queue caps and Wright's Law learning curves may be more aggressive.",
                "confidence": "medium"
            },
            "lower": {
                "category": "structural_model",
                "explanation": f"Our model projects slower deployment than EIA — this may reflect our queue caps binding earlier, higher cost assumptions, or gas friction effects dampening clean investment signals.",
                "confidence": "medium"
            }
        },
        ("clean_penetration", "NREL_Cambium"): {
            "higher": {
                "category": "structural_model",
                "explanation": f"Our model projects faster clean deployment than NREL Cambium, which uses ReEDS capacity expansion. This may reflect different interconnection queue assumptions or more aggressive learning curves in our model.",
                "confidence": "medium"
            },
            "lower": {
                "category": "input_assumption",
                "explanation": f"NREL Cambium projects faster deployment. Cambium uses ReEDS with detailed transmission expansion and state policy modeling. Our model's queue caps or cost assumptions may be more conservative.",
                "confidence": "high"
            }
        },
        ("lmp_trajectory", "EIA_AEO"): {
            "higher": {
                "category": "structural_model",
                "explanation": f"Our model shows higher wholesale prices than EIA AEO. Our hourly dispatch resolution captures scarcity pricing and congestion effects that EIA's annual model smooths over. Also, our gas friction dimension can raise LMP.",
                "confidence": "high"
            },
            "lower": {
                "category": "structural_model",
                "explanation": f"Our model shows lower prices than EIA AEO. At high clean penetration, our hourly VRE surplus suppresses prices more aggressively than EIA's model, which has less temporal granularity.",
                "confidence": "high"
            }
        },
        ("lmp_trajectory", "NREL_Cambium"): {
            "higher": {
                "category": "input_assumption",
                "explanation": f"Our LMP exceeds Cambium's marginal cost. Note: Cambium reports short-run marginal cost (SRMC), not all-in wholesale price. Our LMP includes capacity market revenues, scarcity pricing, and congestion — SRMC is a floor, not the full price.",
                "confidence": "high"
            },
            "lower": {
                "category": "structural_model",
                "explanation": f"Our LMP is below Cambium's marginal cost — unusual since we include more price components. May indicate more aggressive price suppression from VRE surplus in our model.",
                "confidence": "low"
            }
        },
        ("emissions", "EIA_AEO"): {
            "higher": {
                "category": "input_assumption",
                "explanation": f"Our model shows higher emissions than EIA AEO. Our threshold-based coal retirement may be slower than EIA's plant-by-plant retirement schedule which incorporates announced closures.",
                "confidence": "medium"
            },
            "lower": {
                "category": "input_assumption",
                "explanation": f"Our model shows lower emissions — likely because our merit-order retirement displaces coal faster at moderate clean penetration levels. EIA AEO Reference doesn't fully capture IRA-accelerated coal displacement.",
                "confidence": "medium"
            }
        },
        ("emissions", "NREL_Cambium"): {
            "higher": {
                "category": "input_assumption",
                "explanation": f"Cambium projects deeper emissions cuts. Their model includes state policies (RGGI, CA cap-and-trade) and EPA regulations more granularly. Our model uses a simpler emission-rate approach.",
                "confidence": "high"
            },
            "lower": {
                "category": "structural_model",
                "explanation": f"Our model projects lower emissions than Cambium — may reflect more aggressive clean deployment or different demand growth assumptions.",
                "confidence": "medium"
            }
        }
    }

    key = (dimension, bm_name)
    if key in attributions and direction in attributions[key]:
        return attributions[key][direction]

    return {
        "category": "unknown",
        "explanation": f"Our model is {direction} than {bm_name} by {bm_result['pct_diff']:.1f}%. Further investigation needed.",
        "confidence": "low"
    }


# ═══════════════════════════════════════════════════════════════════════
# STEP 4: Export for Dashboard
# ═══════════════════════════════════════════════════════════════════════

def export_dashboard_js(model_outputs, benchmarks, comparison):
    """Export validation data as JS for dashboard consumption."""
    print("\nExporting dashboard JS...")

    dashboard_data = {
        "model": {},
        "benchmarks": {},
        "comparison": {},
        "deviations": comparison.get("deviation_analysis", []),
        "summary": comparison.get("summary", {})
    }

    # Model fan bands (simplified for JS)
    for iso in ISOS:
        if iso not in model_outputs:
            continue
        iso_data = {}
        for year in ['2023', '2030', '2035', '2040', '2045', '2050']:
            if year not in model_outputs[iso]:
                continue
            year_data = model_outputs[iso][year]
            iso_data[year] = {}
            for metric in ['clean_pct', 'avg_lmp', 'emissions_mt', 'demand_twh']:
                if metric in year_data:
                    iso_data[year][metric] = year_data[metric]
        dashboard_data["model"][iso] = iso_data

    # Benchmark lines (EIA + NREL regional)
    for source_key in ['EIA_AEO_2025', 'NREL_CAMBIUM_2024']:
        source = benchmarks[source_key]
        regional = source.get('regional', {})
        bm_data = {}
        for iso in ISOS:
            if iso in regional:
                bm_data[iso] = regional[iso]
        dashboard_data["benchmarks"][source_key] = bm_data

    # Comparison scorecard
    for iso in ISOS:
        iso_comp = comparison.get("by_iso", {}).get(iso, {})
        dashboard_data["comparison"][iso] = {
            "overall_grade": iso_comp.get("overall_grade", 0),
            "directional_clean": iso_comp.get("directional_clean_penetration", {}),
            "directional_emissions": iso_comp.get("directional_emissions", {}),
        }

    # Write JS file
    js_path = os.path.join(DASHBOARD_JS_DIR, 'validation-data.js')
    js_content = f"// Auto-generated by validate_market_forecasts.py\n"
    js_content += f"// Generated: 2026-03-14\n"
    js_content += f"const VALIDATION_DATA = {json.dumps(dashboard_data, indent=2)};\n"

    with open(js_path, 'w') as f:
        f.write(js_content)
    print(f"  Written: {js_path}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("MARKET FORECAST VALIDATION")
    print("Comparing SMARTargets reference sweep vs. published benchmarks")
    print("=" * 70)

    # Step 1: Extract our model outputs
    model_outputs = extract_model_outputs()

    # Step 2: Build benchmark data
    benchmarks = build_benchmark_data()

    # Step 3: Run comparison
    comparison = run_comparison(model_outputs, benchmarks)

    # Save outputs
    with open(os.path.join(VALIDATION_DIR, 'model_outputs.json'), 'w') as f:
        json.dump(model_outputs, f, indent=2)
    print(f"\nSaved: data/validation/model_outputs.json")

    with open(os.path.join(VALIDATION_DIR, 'benchmarks.json'), 'w') as f:
        json.dump(benchmarks, f, indent=2)
    print(f"Saved: data/validation/benchmarks.json")

    with open(os.path.join(VALIDATION_DIR, 'comparison_results.json'), 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"Saved: data/validation/comparison_results.json")

    # Step 4: Export dashboard JS
    export_dashboard_js(model_outputs, benchmarks, comparison)

    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  ISOs compared: {comparison['summary']['isos_compared']}")
    print(f"  Avg alignment score: {comparison['summary']['avg_alignment_score']}%")
    print(f"  Total deviations: {comparison['summary']['total_deviations']}")
    print(f"  Significant (FAIL): {comparison['summary']['significant_deviations']}")
    print()

    for iso in ISOS:
        iso_comp = comparison.get('by_iso', {}).get(iso, {})
        grade = iso_comp.get('overall_grade', 0)
        dir_clean = iso_comp.get('directional_clean_penetration', {})
        print(f"  {iso:6s}: Grade {grade:3.0f}% | Directional clean: {dir_clean.get('aligned', 0)}/{dir_clean.get('total', 0)}")

    if comparison['deviation_analysis']:
        print(f"\n  Top deviations:")
        for dev in sorted(comparison['deviation_analysis'], key=lambda x: x['pct_diff'], reverse=True)[:5]:
            print(f"    {dev['iso']}/{dev['dimension']}/{dev['year']}: "
                  f"ours={dev['our_p50']:.1f} vs {dev['benchmark']}={dev['benchmark_value']:.1f} "
                  f"({dev['pct_diff']:.1f}% - {dev['score']}) — {dev['attribution']['category']}")

    print("\nDone.")


if __name__ == '__main__':
    main()
