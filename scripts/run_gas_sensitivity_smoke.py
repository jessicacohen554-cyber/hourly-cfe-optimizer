#!/usr/bin/env python3
"""
Gas-sensitivity smoke test harness for Step 2.3.

12 configs: 2 ISOs (PJM, ERCOT) × 2 pathways (A, B) × 3 gas treatments.

Gas treatments:
  shadow+endo      STRANDING_SHADOW_WEIGHT=0.5  cost_mode=2
  noShadow+endo    STRANDING_SHADOW_WEIGHT=0.0  cost_mode=2
  noShadow+noEndo  STRANDING_SHADOW_WEIGHT=0.0  cost_mode=1

For each (ISO, pathway): load profiles once, run stage1_select once to get shared
seeds, then for each gas config patch the module global and run solve_pathway on
the top-3 seeds. Best beam = most thresholds achieved; tiebreak on lowest
cumulative cost at the 99.9% threshold.

Output dir: data/step2.3-adaptive/smoke_gas_sensitivity/
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import step2_3_reliability_tax_adaptive as m23
from step2_3_reliability_tax_adaptive import (
    RunConfig,
    RA_MARGIN,
    build_profile_matrix,
    _load_demand_norm,
    stage1_select,
    solve_pathway,
    warmup_jit,
)

OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "data" / "step2.3-adaptive" / "smoke_gas_sensitivity"
)

ISOS = ["PJM", "ERCOT"]
PATHWAYS = ["A", "B"]

GAS_CONFIGS = [
    {"label": "shadow+endo",     "shadow": 0.5, "cost_mode": 2},
    {"label": "noShadow+endo",   "shadow": 0.0, "cost_mode": 2},
    {"label": "noShadow+noEndo", "shadow": 0.0, "cost_mode": 1},
]

N_BEAMS = 3
STAGE2_SAMPLES = 5000
TOP_N_SEEDS = 3


# ---------------------------------------------------------------------------
# Beam selection
# ---------------------------------------------------------------------------

def _cost_at_999(result: dict) -> float:
    """Cumulative cost ($) at the 99.9% threshold; inf if not reached."""
    for snap in reversed(result["threshold_snapshots"]):
        if snap["threshold_pct"] >= 99.9:
            return snap["cumulative_cost_usd"]
    if result["threshold_snapshots"]:
        return result["threshold_snapshots"][-1]["cumulative_cost_usd"]
    return float("inf")


def pick_best_beam(results: list[dict]) -> dict:
    """Most thresholds achieved; tiebreak = lowest cumulative cost at 99.9%."""
    return min(
        results,
        key=lambda r: (-r["n_thresholds_achieved"], _cost_at_999(r)),
    )


# ---------------------------------------------------------------------------
# Trajectory builder
# ---------------------------------------------------------------------------

def build_trajectory(snapshots: list[dict]) -> list[dict]:
    """Build one row per year 2025-2050, carrying forward snapshot values."""
    # Keep the highest-threshold snapshot per year (last crossing in that year).
    year_map: dict[int, dict] = {}
    for snap in snapshots:
        yr = snap["year_achieved"]
        if yr not in year_map or snap["threshold_pct"] > year_map[yr]["threshold_pct"]:
            year_map[yr] = snap

    prev = {"cfe_pct": 0.0, "gas_mw": 0.0, "inc": 0.0, "cum": 0.0}
    rows = []
    for year in range(2025, 2051):
        if year in year_map:
            s = year_map[year]
            prev = {
                "cfe_pct": round(s["achieved_cfe_pct"], 2),
                "gas_mw":  round(s["gas_need_mw"], 1),
                "inc":     round(s["incremental_cost_usd"] / 1e9, 3),
                "cum":     round(s["cumulative_cost_usd"] / 1e9, 3),
            }
        rows.append({
            "year":               year,
            "cfe_pct":            prev["cfe_pct"],
            "gas_mw":             prev["gas_mw"],
            "incremental_cost_B": prev["inc"],
            "cumulative_cost_B":  prev["cum"],
        })
    return rows


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------

def snap_year(snapshots: list[dict], threshold: float) -> int | str:
    """Year first snapshot with threshold_pct >= threshold was achieved."""
    for s in snapshots:
        if s["threshold_pct"] >= threshold:
            return s["year_achieved"]
    return "N/A"


def snap_cost(snapshots: list[dict], threshold: float) -> str:
    """Cumulative cost ($B) at first snapshot with threshold_pct >= threshold."""
    for s in snapshots:
        if s["threshold_pct"] >= threshold:
            return f"{s['cumulative_cost_usd'] / 1e9:.2f}"
    return "N/A"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    warmup_jit()

    all_summaries: list[dict] = []
    t_start = time.time()

    for iso in ISOS:
        for pathway in PATHWAYS:
            print(f"\n{'='*65}")
            print(f"[harness] {iso} Pathway {pathway} — loading profiles")

            P  = build_profile_matrix(iso)
            dn = _load_demand_norm(iso)
            dm = dn * (1.0 + RA_MARGIN)
            P32  = P.astype(np.float32)
            dm32 = dm.astype(np.float32)
            dn32 = dn.astype(np.float32)

            # Mirror the storage cap logic from main() in the target module.
            m23._STORAGE_CAP_PCT = float(dn.max() * m23.H * 100.0 * 1.15)

            # Use default shadow weight for seed selection.
            m23.STRANDING_SHADOW_WEIGHT = 0.5
            seed_cfg = RunConfig(
                iso=iso,
                pathway=pathway,
                scenario_name="base",
                demand_growth="Medium",
                n_beams=N_BEAMS,
                stage2_samples=STAGE2_SAMPLES,
                cost_mode=2,
            )
            seeds = stage1_select(iso, seed_cfg, P32, dm32, dn32)
            if not seeds:
                print(f"[harness] ERROR: No seeds for {iso} P{pathway}")
                continue

            top_seeds = seeds[:TOP_N_SEEDS]
            print(f"[harness] Seeds selected: {[s['archetype'] for s in top_seeds]}")

            for gc in GAS_CONFIGS:
                label     = gc["label"]
                shadow    = gc["shadow"]
                cost_mode = gc["cost_mode"]

                print(f"\n[harness] {iso} P{pathway} | {label}  "
                      f"shadow={shadow}  mode={cost_mode}")

                # Patch the module-level global before each set of runs.
                m23.STRANDING_SHADOW_WEIGHT = shadow

                cfg = RunConfig(
                    iso=iso,
                    pathway=pathway,
                    scenario_name="base",
                    demand_growth="Medium",
                    n_beams=N_BEAMS,
                    stage2_samples=STAGE2_SAMPLES,
                    cost_mode=cost_mode,
                )

                beam_results: list[dict] = []
                for bi, seed in enumerate(top_seeds):
                    print(f"  beam {bi + 1}/{len(top_seeds)}: {seed['archetype']}")
                    t0 = time.time()
                    res = solve_pathway(seed, cfg, P32, dm32, dn32, beam_idx=bi)
                    dt = time.time() - t0
                    print(f"  → {res['n_thresholds_achieved']}/20 thresholds  "
                          f"peak_gas={res['peak_gas_mw']:,.0f} MW  {dt:.0f}s")
                    beam_results.append(res)

                best = pick_best_beam(beam_results)

                out_name = f"{iso}__P{pathway}__{label}__best_beam.json"
                (OUTPUT_DIR / out_name).write_text(json.dumps(best, indent=2))
                print(f"  wrote {out_name}")

                snaps = best["threshold_snapshots"]
                snaps_90plus = [s for s in snaps if s["threshold_pct"] >= 90]

                all_summaries.append({
                    "test_tag":                  f"{iso}_P{pathway}_{label}",
                    "iso":                       iso,
                    "pathway":                   pathway,
                    "gas_label":                 label,
                    "peak_gas_mw":               best["peak_gas_mw"],
                    "peak_gas_year":             best["peak_gas_year"],
                    "beam_archetype":            best["beam_archetype"],
                    "n_thresholds_achieved":     best["n_thresholds_achieved"],
                    "trajectory":                build_trajectory(snaps),
                    "threshold_snapshots_90plus": snaps_90plus,
                })

    summary_path = OUTPUT_DIR / "smoke_gas_sensitivity_summary.json"
    summary_path.write_text(json.dumps(all_summaries, indent=2))
    print(f"\nWrote smoke_gas_sensitivity_summary.json ({len(all_summaries)} configs)")
    print(f"Total elapsed: {time.time() - t_start:.0f}s")

    # ---- summary table ----
    W = 130
    print("\n" + "=" * W)
    print(
        f"{'ISO':<6} {'PW':<3} {'Gas Config':<18} {'Peak Gas MW':>12} "
        f"{'Peak Yr':>8} {'Yr@90%':>7} {'Yr@95%':>7} {'Yr@99.9%':>9} "
        f"{'Cost@90%$B':>11} {'Cost@99.9%$B':>13} {'Thresh':>7}"
    )
    print("-" * W)

    incomplete: list[str] = []
    for s in all_summaries:
        snaps = s["threshold_snapshots_90plus"]
        n = s["n_thresholds_achieved"]
        flag = "" if n == 20 else "  *** INCOMPLETE"
        if n != 20:
            incomplete.append(s["test_tag"])
        print(
            f"{s['iso']:<6} {s['pathway']:<3} {s['gas_label']:<18} "
            f"{s['peak_gas_mw']:>12,.0f} {s['peak_gas_year']:>8} "
            f"{str(snap_year(snaps, 90)):>7} {str(snap_year(snaps, 95)):>7} "
            f"{str(snap_year(snaps, 99.9)):>9} "
            f"{snap_cost(snaps, 90):>11} {snap_cost(snaps, 99.9):>13} "
            f"{n:>5}/20{flag}"
        )

    print("=" * W)
    if not incomplete:
        print("\nAll 12 configs achieved 20/20 thresholds.")
    else:
        print(f"\nWARNING: {len(incomplete)} config(s) did NOT achieve 20/20 thresholds:")
        for tag in incomplete:
            print(f"  {tag}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
