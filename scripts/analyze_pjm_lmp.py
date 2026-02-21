#!/usr/bin/env python3
"""
Comprehensive PJM LMP Analysis Script
Reads PJM_lmp.parquet, computes statistical summaries, sensitivity analysis,
ANOVA variance decomposition, and outputs pjm_lmp_trends.json for dashboard use.
"""

import json
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

INPUT_PATH = "/home/user/hourly-cfe-optimizer/data/lmp/PJM_lmp.parquet"
OUTPUT_PATH = "/home/user/hourly-cfe-optimizer/data/lmp/pjm_lmp_trends.json"

THRESHOLDS = [50, 60, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 100]

TOGGLE_NAMES = {
    "renewable_gen": "Renewable Generation Cost",
    "firm_gen": "Firm Generation Cost",
    "storage": "Storage Cost",
    "fossil_fuel": "Fossil Fuel Price",
    "ccs": "CCS Toggle",
    "transmission": "Transmission Cost",
    "ccs_cost": "CCS Cost Level",
    "q45": "45Q Tax Credit",
}

LEVEL_MAP = {"L": "Low", "M": "Medium", "H": "High", "N": "None"}
Q45_MAP = {"0": "Off", "1": "On"}

LMP_METRICS = [
    "avg_lmp", "peak_avg_lmp", "offpeak_avg_lmp",
    "price_volatility", "zero_price_hours",
    "negative_price_hours", "scarcity_hours",
]


def parse_scenario(scenario):
    parts = scenario.split("_")
    paired = parts[0]
    return {
        "renewable_gen": LEVEL_MAP[paired[0]],
        "firm_gen": LEVEL_MAP[paired[1]],
        "storage": LEVEL_MAP[paired[2]],
        "fossil_fuel": LEVEL_MAP[paired[3]],
        "ccs": LEVEL_MAP[parts[1]],
        "transmission": LEVEL_MAP[parts[2]],
        "ccs_cost": LEVEL_MAP[parts[3][0]],
        "q45": Q45_MAP[parts[3][1]],
    }


def percentile_stats(series):
    return {
        "mean": round(float(series.mean()), 2),
        "std": round(float(series.std()), 2),
        "min": round(float(series.min()), 2),
        "p10": round(float(series.quantile(0.10)), 2),
        "p25": round(float(series.quantile(0.25)), 2),
        "p50": round(float(series.quantile(0.50)), 2),
        "p75": round(float(series.quantile(0.75)), 2),
        "p90": round(float(series.quantile(0.90)), 2),
        "max": round(float(series.max()), 2),
        "p10_p90_spread": round(float(series.quantile(0.90) - series.quantile(0.10)), 2),
    }


def safe_float(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        if np.isnan(x) or np.isinf(x):
            return None
        return round(float(x), 4)
    return x


def main():
    print("=" * 70)
    print("PJM LMP Comprehensive Analysis")
    print("=" * 70)

    df = pd.read_parquet(INPUT_PATH)
    print(f"\nLoaded {len(df):,} rows, {df['scenario'].nunique()} unique scenarios")
    print(f"Thresholds: {sorted(df['threshold'].unique())}")
    print(f"Fuel levels: {sorted(df['fuel_level'].unique())}")

    # Parse toggles into columns
    print("\nParsing scenario toggles...")
    toggle_cols = list(TOGGLE_NAMES.keys())
    parsed = df["scenario"].apply(parse_scenario)
    for col in toggle_cols:
        df[col] = parsed.apply(lambda x, c=col: x[c])

    # === 1. Threshold-level statistics ===
    print("\n--- 1. Computing threshold-level statistics ---")
    threshold_stats = {}

    for t in THRESHOLDS:
        t_key = str(t)
        tdf = df[df["threshold"] == t]
        stats = {"n_rows": len(tdf)}

        metrics_stats = {}
        for metric in LMP_METRICS:
            if metric in tdf.columns:
                metrics_stats[metric] = percentile_stats(tdf[metric])
        stats["metrics"] = metrics_stats

        fuel_stats = {}
        for fl in sorted(tdf["fuel_level"].unique()):
            fl_df = tdf[tdf["fuel_level"] == fl]
            fuel_stats[fl] = {
                "n_rows": len(fl_df),
                "avg_lmp": percentile_stats(fl_df["avg_lmp"]),
            }
        stats["by_fuel_level"] = fuel_stats

        fuel_means = tdf.groupby("fuel_level")["avg_lmp"].mean()
        stats["lowest_lmp_fuel"] = fuel_means.idxmin()
        stats["highest_lmp_fuel"] = fuel_means.idxmax()
        stats["fuel_lmp_means"] = {k: round(v, 2) for k, v in fuel_means.items()}

        threshold_stats[t_key] = stats

    print("\n  Threshold | Median LMP | P10-P90 Spread | Lowest Fuel")
    print("  " + "-" * 55)
    for t in THRESHOLDS:
        s = threshold_stats[str(t)]
        m = s["metrics"]["avg_lmp"]
        print(f"  {t:>6.1f}%   | ${m['p50']:>7.2f}   | ${m['p10_p90_spread']:>7.2f}       | {s['lowest_lmp_fuel']}")

    # === 2. Sensitivity analysis by toggle ===
    print("\n--- 2. Sensitivity analysis by toggle ---")
    sensitivity = {}

    for toggle in toggle_cols:
        toggle_data = {}
        levels = sorted(df[toggle].unique())
        for t in THRESHOLDS:
            t_key = str(t)
            tdf = df[df["threshold"] == t]
            level_means = {}
            for level in levels:
                ldf = tdf[tdf[toggle] == level]
                level_means[level] = round(float(ldf["avg_lmp"].mean()), 2)
            vals = list(level_means.values())
            spread = round(max(vals) - min(vals), 2)
            toggle_data[t_key] = {"level_means": level_means, "spread": spread}
        sensitivity[toggle] = {
            "name": TOGGLE_NAMES[toggle],
            "levels": levels,
            "by_threshold": toggle_data,
        }

    print("\n  Toggle impact on avg_lmp (spread = max_level - min_level):")
    key_thresholds = [50, 80, 90, 95, 100]
    header = "  Toggle              | " + " | ".join(f"{t}%" for t in key_thresholds)
    print(header)
    print("  " + "-" * len(header))
    for toggle in toggle_cols:
        name = TOGGLE_NAMES[toggle][:20].ljust(20)
        spreads = [f"${sensitivity[toggle]['by_threshold'][str(t)]['spread']:>5.2f}" for t in key_thresholds]
        print(f"  {name} | " + " | ".join(spreads))

    # === 3. ANOVA-style variance decomposition ===
    print("\n--- 3. ANOVA variance decomposition ---")
    anova_results = {}

    for t in THRESHOLDS:
        t_key = str(t)
        tdf = df[df["threshold"] == t].copy()
        total_var = tdf["avg_lmp"].var()

        toggle_var_explained = {}
        for toggle in toggle_cols:
            group_means = tdf.groupby(toggle)["avg_lmp"].transform("mean")
            ss_between = ((group_means - tdf["avg_lmp"].mean()) ** 2).sum()
            ss_total = ((tdf["avg_lmp"] - tdf["avg_lmp"].mean()) ** 2).sum()
            eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0
            toggle_var_explained[toggle] = round(eta_sq * 100, 2)

        sorted_toggles = sorted(toggle_var_explained.items(), key=lambda x: -x[1])
        anova_results[t_key] = {
            "total_variance": round(float(total_var), 4),
            "variance_explained_pct": {k: v for k, v in sorted_toggles},
            "top_driver": sorted_toggles[0][0] if sorted_toggles else None,
            "top_driver_pct": sorted_toggles[0][1] if sorted_toggles else 0,
        }

    print("\n  Threshold | Top Driver          | Var Explained")
    print("  " + "-" * 50)
    for t in THRESHOLDS:
        a = anova_results[str(t)]
        driver_name = TOGGLE_NAMES.get(a["top_driver"], a["top_driver"])[:20]
        print(f"  {t:>6.1f}%   | {driver_name:<20s} | {a['top_driver_pct']:>5.1f}%")

    # === 4. Steepest decline scenarios ===
    print("\n--- 4. Steepest decline scenarios (50% -> 100%) ---")

    pivot = df.pivot_table(
        index=["scenario", "fuel_level"],
        columns="threshold",
        values="avg_lmp",
        aggfunc="first",
    )

    steepest_list = []
    if 50.0 in pivot.columns and 100.0 in pivot.columns:
        pivot["decline"] = pivot[50.0] - pivot[100.0]
        pivot["decline_pct"] = (pivot["decline"] / pivot[50.0]) * 100

        steepest = pivot.nlargest(10, "decline")
        for (scenario, fuel), row in steepest.iterrows():
            steepest_list.append({
                "scenario": scenario,
                "fuel_level": fuel,
                "lmp_at_50": round(float(row[50.0]), 2),
                "lmp_at_100": round(float(row[100.0]), 2),
                "decline": round(float(row["decline"]), 2),
                "decline_pct": round(float(row["decline_pct"]), 1),
                "toggles": parse_scenario(scenario),
            })

        print(f"\n  Top 5 steepest LMP decline (50% -> 100%):")
        for i, s in enumerate(steepest_list[:5]):
            print(f"  {i+1}. {s['scenario']} ({s['fuel_level']}): ${s['lmp_at_50']:.2f} -> ${s['lmp_at_100']:.2f} (decline: ${s['decline']:.2f}, {s['decline_pct']:.1f}%)")

    # === 5. Most stable scenarios ===
    print("\n--- 5. Most stable scenarios (smallest LMP spread across thresholds) ---")

    threshold_cols_float = [c for c in pivot.columns if isinstance(c, float) and c <= 100]
    pivot["lmp_std"] = pivot[threshold_cols_float].std(axis=1)
    pivot["lmp_range"] = pivot[threshold_cols_float].max(axis=1) - pivot[threshold_cols_float].min(axis=1)

    most_stable = pivot.nsmallest(10, "lmp_range")
    stable_list = []
    for (scenario, fuel), row in most_stable.iterrows():
        stable_list.append({
            "scenario": scenario,
            "fuel_level": fuel,
            "lmp_std": round(float(row["lmp_std"]), 2),
            "lmp_range": round(float(row["lmp_range"]), 2),
            "lmp_at_50": round(float(row[50.0]), 2) if 50.0 in row.index else None,
            "lmp_at_100": round(float(row[100.0]), 2) if 100.0 in row.index else None,
            "toggles": parse_scenario(scenario),
        })

    for i, s in enumerate(stable_list[:5]):
        print(f"  {i+1}. {s['scenario']} ({s['fuel_level']}): range=${s['lmp_range']:.2f}, std=${s['lmp_std']:.2f}")

    # === 6. Metric correlations ===
    print("\n--- 6. Cross-metric correlations ---")
    corr_metrics = [
        "avg_lmp", "peak_avg_lmp", "offpeak_avg_lmp", "price_volatility",
        "zero_price_hours", "negative_price_hours", "scarcity_hours",
        "duck_curve_depth_mw", "net_peak_price", "fossil_revenue_mwh",
    ]
    available_metrics = [m for m in corr_metrics if m in df.columns]
    corr_matrix = df[available_metrics].corr()

    correlations = {}
    for m1 in available_metrics:
        correlations[m1] = {}
        for m2 in available_metrics:
            correlations[m1][m2] = round(float(corr_matrix.loc[m1, m2]), 4)

    key_pairs = [
        ("avg_lmp", "price_volatility"),
        ("avg_lmp", "zero_price_hours"),
        ("avg_lmp", "negative_price_hours"),
        ("avg_lmp", "scarcity_hours"),
        ("avg_lmp", "duck_curve_depth_mw"),
        ("avg_lmp", "net_peak_price"),
        ("zero_price_hours", "price_volatility"),
        ("peak_avg_lmp", "offpeak_avg_lmp"),
    ]
    key_correlations = []
    for m1, m2 in key_pairs:
        if m1 in available_metrics and m2 in available_metrics:
            r = float(corr_matrix.loc[m1, m2])
            key_correlations.append({
                "metric_1": m1,
                "metric_2": m2,
                "correlation": round(r, 4),
                "strength": "strong" if abs(r) > 0.7 else "moderate" if abs(r) > 0.4 else "weak",
                "direction": "positive" if r > 0 else "negative",
            })

    print("\n  Key Correlations:")
    for kc in key_correlations:
        print(f"  {kc['metric_1']:>25s} vs {kc['metric_2']:<25s}: r={kc['correlation']:>7.4f} ({kc['strength']}, {kc['direction']})")

    # === 7. Threshold transition analysis ===
    print("\n--- 7. Threshold transition analysis ---")
    transition_stats = {}
    for i in range(len(THRESHOLDS) - 1):
        t1, t2 = THRESHOLDS[i], THRESHOLDS[i + 1]
        df1 = df[df["threshold"] == t1]
        df2 = df[df["threshold"] == t2]

        merged = df1[["scenario", "fuel_level", "avg_lmp"]].merge(
            df2[["scenario", "fuel_level", "avg_lmp"]],
            on=["scenario", "fuel_level"],
            suffixes=("_low", "_high"),
        )

        if len(merged) > 0:
            delta = merged["avg_lmp_high"] - merged["avg_lmp_low"]
            pct_delta = (delta / merged["avg_lmp_low"]) * 100
            transition_stats[f"{t1}->{t2}"] = {
                "mean_delta": round(float(delta.mean()), 2),
                "median_delta": round(float(delta.median()), 2),
                "mean_pct_delta": round(float(pct_delta.mean()), 2),
                "std_delta": round(float(delta.std()), 2),
                "pct_scenarios_declining": round(float((delta < 0).mean() * 100), 1),
                "threshold_from": t1,
                "threshold_to": t2,
            }

    print("\n  Transition      | Mean Delta | % Declining | Steepest Decline Zone?")
    print("  " + "-" * 65)
    max_decline = min(transition_stats.values(), key=lambda x: x["mean_delta"])
    for key, ts in transition_stats.items():
        is_steepest = " <-- STEEPEST" if ts == max_decline else ""
        print(f"  {key:>12s}    | ${ts['mean_delta']:>7.2f}  | {ts['pct_scenarios_declining']:>5.1f}%      |{is_steepest}")

    # === 8. Hourly LMP percentile bands ===
    print("\n--- 8. Hourly LMP percentile bands (from precomputed p10-p90) ---")
    hourly_bands = {}
    for t in THRESHOLDS:
        tdf = df[df["threshold"] == t]
        hourly_bands[str(t)] = {
            "lmp_p10": {"mean": round(float(tdf["lmp_p10"].mean()), 2), "min": round(float(tdf["lmp_p10"].min()), 2), "max": round(float(tdf["lmp_p10"].max()), 2)},
            "lmp_p25": {"mean": round(float(tdf["lmp_p25"].mean()), 2), "min": round(float(tdf["lmp_p25"].min()), 2), "max": round(float(tdf["lmp_p25"].max()), 2)},
            "lmp_p50": {"mean": round(float(tdf["lmp_p50"].mean()), 2), "min": round(float(tdf["lmp_p50"].min()), 2), "max": round(float(tdf["lmp_p50"].max()), 2)},
            "lmp_p75": {"mean": round(float(tdf["lmp_p75"].mean()), 2), "min": round(float(tdf["lmp_p75"].min()), 2), "max": round(float(tdf["lmp_p75"].max()), 2)},
            "lmp_p90": {"mean": round(float(tdf["lmp_p90"].mean()), 2), "min": round(float(tdf["lmp_p90"].min()), 2), "max": round(float(tdf["lmp_p90"].max()), 2)},
        }

    # === 9. Archetype analysis ===
    print("\n--- 9. Dominant resource archetypes per threshold ---")
    archetype_summary = {}
    for t in THRESHOLDS:
        tdf = df[df["threshold"] == t]
        arch_counts = tdf["archetype_key"].value_counts()
        top_5 = arch_counts.head(5)
        arch_list = []
        for arch_key, count in top_5.items():
            arch_lmp = tdf[tdf["archetype_key"] == arch_key]["avg_lmp"].mean()
            arch_list.append({
                "archetype_key": arch_key,
                "count": int(count),
                "pct_of_scenarios": round(count / len(tdf) * 100, 1),
                "avg_lmp": round(float(arch_lmp), 2),
            })
        archetype_summary[str(t)] = arch_list

    for t in [50, 90, 100]:
        print(f"\n  Threshold {t}%:")
        for a in archetype_summary[str(t)][:3]:
            print(f"    {a['archetype_key']}: {a['pct_of_scenarios']}% of scenarios, avg_lmp=${a['avg_lmp']:.2f}")

    # === 10. Interaction effects ===
    print("\n--- 10. Toggle interaction effects ---")
    interaction_results = {}
    t90 = df[df["threshold"] == 90.0].copy()
    grand_mean = t90["avg_lmp"].mean()
    ss_total_90 = ((t90["avg_lmp"] - grand_mean) ** 2).sum()

    for i, t1 in enumerate(toggle_cols):
        for t2 in toggle_cols[i + 1:]:
            main1 = t90.groupby(t1)["avg_lmp"].transform("mean")
            main2 = t90.groupby(t2)["avg_lmp"].transform("mean")
            predicted_additive = main1 + main2 - grand_mean
            actual = t90.groupby([t1, t2])["avg_lmp"].transform("mean")
            interaction = actual - predicted_additive
            ss_interaction = (interaction ** 2).sum()
            interaction_pct = round(float(ss_interaction / ss_total_90 * 100), 2) if ss_total_90 > 0 else 0
            interaction_results[f"{t1}:{t2}"] = {
                "toggle_1": t1,
                "toggle_2": t2,
                "interaction_var_pct": interaction_pct,
            }

    sorted_interactions = sorted(interaction_results.values(), key=lambda x: -x["interaction_var_pct"])
    print("\n  Top 5 toggle interactions (at 90% threshold):")
    for inter in sorted_interactions[:5]:
        print(f"    {TOGGLE_NAMES[inter['toggle_1']]:>25s} x {TOGGLE_NAMES[inter['toggle_2']]:<25s}: {inter['interaction_var_pct']:.2f}% of variance")

    # === Assemble final JSON ===
    print("\n--- Assembling JSON output ---")

    output = {
        "metadata": {
            "iso": "PJM",
            "n_scenarios": int(df["scenario"].nunique()),
            "n_thresholds": len(THRESHOLDS),
            "thresholds": THRESHOLDS,
            "n_total_rows": len(df),
            "fuel_levels": sorted(df["fuel_level"].unique().tolist()),
            "toggle_names": TOGGLE_NAMES,
            "scenario_format": {
                "description": "ABCD_E_F_GH_I where A=Renewable Gen, B=Firm Gen, C=Storage, D=Fossil Fuel, E=CCS, F=Transmission, G=CCS Cost, H=45Q, I=Geothermal(X=N/A)",
                "positions": {
                    "0_char0": "renewable_gen (L/M/H)",
                    "0_char1": "firm_gen (L/M/H)",
                    "0_char2": "storage (L/M/H)",
                    "0_char3": "fossil_fuel (L/M/H)",
                    "1": "ccs (L/M/H)",
                    "2": "transmission (N/L/M/H)",
                    "3_char0": "ccs_cost (L/M/H)",
                    "3_char1": "q45 (0=Off/1=On)",
                    "4": "geothermal (X=N/A for PJM)",
                },
            },
            "generated_at": pd.Timestamp.now().isoformat(),
        },
        "threshold_stats": threshold_stats,
        "sensitivity": sensitivity,
        "anova_variance_decomposition": anova_results,
        "steepest_decline_scenarios": steepest_list,
        "most_stable_scenarios": stable_list,
        "correlations": {
            "full_matrix": correlations,
            "key_pairs": key_correlations,
        },
        "threshold_transitions": transition_stats,
        "hourly_lmp_bands": hourly_bands,
        "archetype_summary": archetype_summary,
        "interaction_effects": {
            "threshold_analyzed": 90.0,
            "top_interactions": sorted_interactions[:10],
            "all_interactions": {k: v for k, v in interaction_results.items()},
        },
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=safe_float)

    print(f"\nJSON output written to: {OUTPUT_PATH}")
    file_size = len(json.dumps(output, indent=2, default=safe_float))
    print(f"Output size: {file_size:,} characters ({file_size/1024:.1f} KB)")

    # === Key Findings Summary ===
    print("\n" + "=" * 70)
    print("KEY FINDINGS SUMMARY")
    print("=" * 70)

    lmp_50 = threshold_stats["50"]["metrics"]["avg_lmp"]["p50"]
    lmp_100 = threshold_stats["100"]["metrics"]["avg_lmp"]["p50"]
    decline = lmp_50 - lmp_100
    decline_pct = (decline / lmp_50) * 100
    print(f"\n1. MEDIAN LMP TRAJECTORY: ${lmp_50:.2f} (50%) -> ${lmp_100:.2f} (100%)")
    print(f"   Total decline: ${decline:.2f}/MWh ({decline_pct:.1f}%)")

    spread_50 = threshold_stats["50"]["metrics"]["avg_lmp"]["p10_p90_spread"]
    spread_100 = threshold_stats["100"]["metrics"]["avg_lmp"]["p10_p90_spread"]
    print(f"\n2. UNCERTAINTY (P10-P90 spread):")
    print(f"   At 50%: ${spread_50:.2f}/MWh")
    print(f"   At 100%: ${spread_100:.2f}/MWh")
    print(f"   Uncertainty {'widens' if spread_100 > spread_50 else 'narrows'} at higher thresholds")

    anova_90 = anova_results["90"]
    print(f"\n3. TOP LMP DRIVER (at 90% threshold):")
    print(f"   {TOGGLE_NAMES[anova_90['top_driver']]}: {anova_90['top_driver_pct']:.1f}% of variance explained")
    print("   Full variance decomposition at 90%:")
    for toggle, pct in anova_90["variance_explained_pct"].items():
        print(f"     {TOGGLE_NAMES[toggle]:<25s}: {pct:>5.1f}%")

    if steepest_list:
        top = steepest_list[0]
        print(f"\n4. STEEPEST LMP DECLINE:")
        print(f"   {top['scenario']} ({top['fuel_level']}): ${top['lmp_at_50']:.2f} -> ${top['lmp_at_100']:.2f} (-${top['decline']:.2f}, -{top['decline_pct']:.1f}%)")

    if stable_list:
        stab = stable_list[0]
        print(f"\n5. MOST STABLE LMP:")
        print(f"   {stab['scenario']} ({stab['fuel_level']}): range=${stab['lmp_range']:.2f}, std=${stab['lmp_std']:.2f}")

    print(f"\n6. KEY CORRELATIONS:")
    for kc in key_correlations[:4]:
        print(f"   {kc['metric_1']} vs {kc['metric_2']}: r={kc['correlation']:.4f} ({kc['strength']})")

    steepest_transition = min(transition_stats.values(), key=lambda x: x["mean_delta"])
    print(f"\n7. STEEPEST THRESHOLD TRANSITION:")
    print(f"   {steepest_transition['threshold_from']}% -> {steepest_transition['threshold_to']}%: mean change = ${steepest_transition['mean_delta']:.2f}/MWh")

    print("\n" + "=" * 70)
    print("Analysis complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
