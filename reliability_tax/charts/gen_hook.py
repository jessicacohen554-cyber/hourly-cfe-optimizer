"""
gen_hook.py — Generate hook_reliability_tax.json

Data lineage:
  - analysis/reliability-tax/data/MANIFEST.json
  - analysis/reliability-tax/data/<ISO>/pathway{N}_ep{E}.json
  via reliability_tax/charts/data_loader.py

Payload structure:
  gas_cf_gauge:
    per_iso:       per-ISO residual gas reliance at 2050 under Pathway 1 @ 95% CFE
                   = (1 - achieved_cfe_pct/100) at endpoint year for feasible ISOs
                   = None (infeasible) for ISOs that can't reach 95% VRE-only
    average_feasible:  mean residual gas reliance across feasible ISOs (excl infeasible)
    feasibility_map:   per-ISO feasibility flag
  cost_headline:
    pathway1_total_usd:  sum of Pathway 1 undiscounted costs @ 95% across all feasible ISOs
    pathway3_total_usd:  same for Pathway 3 (the destination plan)
    delta_usd:           P1 - P3 (excess cost attributable to VRE-only)
    per_iso:             per-ISO P1 vs P3 undiscounted cost @ 95%
  meta:
    endpoint: 0.95
    note: "Pathway 1 = VRE + batteries + existing nuclear (no new clean firm).
           Infeasible ISOs hit physical ceiling before 95% target."
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "reliability_tax" / "charts"))
from data_loader import get_run, is_feasible, ISOS  # noqa: E402

OUT_PATH = Path(__file__).parent / "hook_reliability_tax.json"
ENDPOINT = 0.95


def main() -> None:
    per_iso_gauge: dict[str, float | None] = {}
    per_iso_cost: dict[str, dict] = {}
    feasibility_map: dict[str, bool] = {}

    for iso in ISOS:
        # Pathway 1 @ 95%
        p1 = get_run(iso, "1", ENDPOINT)
        # Pathway 3 @ 95% (destination plan)
        p3 = get_run(iso, "3", ENDPOINT)

        p1_phys = is_feasible(p1, "physical")
        feasibility_map[iso] = p1_phys

        # Gas reliance gauge: fraction of 2050 demand NOT matched by CFE
        # For feasible runs this is (1 - achieved_cfe_pct/100).
        # For infeasible runs the achieved_cfe is the ceiling hit, still meaningful.
        achieved = p1.get("achieved_cfe_pct", 0.0)
        residual_gas_pct = round(100.0 - achieved, 2)

        # Mark infeasible ISOs as None so the chart layer can annotate differently
        per_iso_gauge[iso] = residual_gas_pct if p1_phys else None

        p1_cost = p1.get("undiscounted_cost_usd", 0.0)
        p3_cost = p3.get("undiscounted_cost_usd", 0.0)
        per_iso_cost[iso] = {
            "pathway1_undiscounted_usd": p1_cost,
            "pathway3_undiscounted_usd": p3_cost,
            "delta_usd": p1_cost - p3_cost,
            "pathway1_feasible": p1_phys,
            "pathway1_achieved_cfe_pct": achieved,
            "pathway3_achieved_cfe_pct": p3.get("achieved_cfe_pct", 0.0),
        }

    # Cross-ISO aggregates — feasible ISOs only (those that reached the target)
    feasible_isos = [iso for iso in ISOS if feasibility_map[iso]]
    gauge_values = [per_iso_gauge[iso] for iso in feasible_isos]
    avg_gauge = round(sum(gauge_values) / len(gauge_values), 2) if gauge_values else None

    p1_total = sum(per_iso_cost[iso]["pathway1_undiscounted_usd"] for iso in feasible_isos)
    p3_total = sum(per_iso_cost[iso]["pathway3_undiscounted_usd"] for iso in feasible_isos)

    payload = {
        "gas_cf_gauge": {
            "description": (
                "Residual non-CFE reliance at 2050 under Pathway 1 (VRE + existing nuclear, "
                "no new clean firm). Expressed as % of 2050 demand not matched by clean sources. "
                "null = ISO hit physical ceiling before the 95% target."
            ),
            "per_iso": per_iso_gauge,
            "average_feasible_isos": avg_gauge,
            "feasibility_map": feasibility_map,
            "feasible_isos": feasible_isos,
            "infeasible_isos": [iso for iso in ISOS if not feasibility_map[iso]],
        },
        "cost_headline": {
            "description": (
                "Undiscounted cumulative 2025–2050 system cost under Pathway 1 vs Pathway 3 "
                "at 95% CFE endpoint, summed across feasible ISOs."
            ),
            "pathway1_total_usd": p1_total,
            "pathway3_total_usd": p3_total,
            "delta_usd": p1_total - p3_total,
            "delta_pct": round((p1_total - p3_total) / p3_total * 100, 1) if p3_total else None,
            "per_iso": per_iso_cost,
        },
        "meta": {
            "endpoint": ENDPOINT,
            "endpoint_pct": 95.0,
            "isos_total": len(ISOS),
            "isos_feasible": len(feasible_isos),
            "note": (
                "Pathway 1 = VRE + batteries + existing nuclear (no new clean firm). "
                "Infeasible ISOs hit a physical ceiling before the 95% CFE target."
            ),
        },
    }

    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT_PATH}")

    # Quick sanity print
    print(f"\n--- hook_reliability_tax quick summary ---")
    print(f"Feasible ISOs @ P1/95%: {feasible_isos}")
    print(f"Infeasible ISOs: {payload['gas_cf_gauge']['infeasible_isos']}")
    print(f"Avg residual gas reliance (feasible): {avg_gauge}%")
    print(f"P1 total cost (feasible ISOs): ${p1_total/1e12:.3f}T")
    print(f"P3 total cost (feasible ISOs): ${p3_total/1e12:.3f}T")
    print(f"Delta (P1-P3): ${(p1_total-p3_total)/1e12:.3f}T  ({payload['cost_headline']['delta_pct']}%)")
    print(f"\nPer-ISO residual gas %:")
    for iso in ISOS:
        g = per_iso_gauge[iso]
        c = per_iso_cost[iso]
        g_str = f"{g:.1f}%" if g is not None else "INFEASIBLE"
        print(f"  {iso}: {g_str}  P1=${c['pathway1_undiscounted_usd']/1e12:.3f}T  P3=${c['pathway3_undiscounted_usd']/1e12:.3f}T  delta=${c['delta_usd']/1e12:.3f}T")


if __name__ == "__main__":
    main()
