"""
gen_four_journeys.py — Generate section4_four_journeys.json

Data lineage:
  - analysis/reliability-tax/data/MANIFEST.json
  - analysis/reliability-tax/data/ERCOT/pathway{N}_ep95.json
  via data_loader

Payload (ERCOT @ 95% endpoint):
  annual_buildout: dict[pathway -> list[{year, resources: [{name, twh_per_year, lcoe}], total_twh}]]
  cumulative_cost: dict[pathway -> list[{year, cumulative_usd, annual_usd, achieved_cfe_pct}]]
  pivot_markers:   dict[pathway -> {pivot_year, pivot_reason} | null]
  retirements:     dict[pathway -> list[{year, clean_pct, gas_retired_gw, coal_retired_gw}]]
  headline_costs:  dict[pathway -> {undiscounted_usd, npv_7pct_usd, achieved_cfe_pct}]
  meta: {iso, endpoint, note}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "reliability_tax" / "charts"))
from data_loader import (  # noqa: E402
    get_run, get_pathway_buildout, get_pathway_cost,
    get_pathway_retirements, get_pivot_info, PATHWAYS,
)

OUT_PATH = Path(__file__).parent / "section4_four_journeys.json"
ISO = "ERCOT"
ENDPOINT = 0.95


def _buildout_for_pathway(pathway: str) -> list[dict]:
    """Aggregate annual_buildout by year and resource category."""
    rows = get_pathway_buildout(ISO, pathway, ENDPOINT)
    result = []
    for row in rows:
        year = row["year"]
        resources = []
        for v in row.get("new_vintages", []):
            resources.append({
                "resource": v["resource"],
                "twh_per_year": round(v["twh_per_year"], 2),
                "locked_lcoe_usd_per_mwh": v.get("locked_lcoe_usd_per_mwh"),
                "tx_adder_usd_per_mwh": v.get("tx_adder_usd_per_mwh", 0.0),
            })
        total_twh = round(sum(r["twh_per_year"] for r in resources), 2)
        result.append({"year": year, "resources": resources, "total_twh_per_year": total_twh})
    return result


def _cumulative_cost_for_pathway(pathway: str) -> list[dict]:
    """Build cumulative cost series from annual cost table."""
    rows = get_pathway_cost(ISO, pathway, ENDPOINT)
    result = []
    cumulative = 0.0
    for row in rows:
        cumulative += row["net_annual_cost_usd"]
        result.append({
            "year": row["year"],
            "annual_usd": row["net_annual_cost_usd"],
            "cumulative_usd": cumulative,
            "achieved_cfe_pct": row["achieved_cfe_pct"],
            "demand_twh": row["demand_twh"],
        })
    return result


def _retirement_series_for_pathway(pathway: str) -> list[dict]:
    """Slim retirement timeline (just the columns needed for section 4 overlay)."""
    rows = get_pathway_retirements(ISO, pathway, ENDPOINT)
    return [
        {
            "year": r["year"],
            "clean_pct": r["clean_pct"],
            "gas_retired_gw": r["gas_retired_gw"],
            "coal_retired_gw": r["coal_retired_gw"],
            "gas_retired_twh": r["gas_retired_twh"],
        }
        for r in rows
    ]


def main() -> None:
    annual_buildout: dict[str, list] = {}
    cumulative_cost: dict[str, list] = {}
    pivot_markers: dict[str, dict | None] = {}
    retirements: dict[str, list] = {}
    headline_costs: dict[str, dict] = {}

    for pathway in PATHWAYS:
        run = get_run(ISO, pathway, ENDPOINT)
        phys = run["feasibility"].get("physical", False)

        annual_buildout[pathway] = _buildout_for_pathway(pathway) if phys else []
        cumulative_cost[pathway] = _cumulative_cost_for_pathway(pathway) if phys else []
        retirements[pathway] = _retirement_series_for_pathway(pathway) if phys else []
        pivot_markers[pathway] = get_pivot_info(run) if phys else None

        headline = run.get("headline", {})
        headline_costs[pathway] = {
            "undiscounted_usd": run.get("undiscounted_cost_usd", 0.0),
            "npv_at_7pct_usd": run.get("npv_at_7pct", 0.0),
            "achieved_cfe_pct": run.get("achieved_cfe_pct", 0.0),
            "physically_feasible": phys,
        }

    payload = {
        "annual_buildout": annual_buildout,
        "cumulative_cost": cumulative_cost,
        "pivot_markers": pivot_markers,
        "retirements": retirements,
        "headline_costs": headline_costs,
        "meta": {
            "iso": ISO,
            "endpoint": ENDPOINT,
            "endpoint_pct": 95.0,
            "note": (
                "ERCOT annual buildout 2025–2050 per pathway at 95% CFE endpoint. "
                "Pivot markers show year Pathway 2a/2b unlocks clean firm. "
                "Retirement overlay = cumulative GW of existing fleet economically displaced."
            ),
        },
    }

    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT_PATH}")

    # Sanity print
    print("\n--- section4_four_journeys quick summary (ERCOT @ 95%) ---")
    for pw in PATHWAYS:
        hc = headline_costs[pw]
        pivot = pivot_markers[pw]
        pivot_str = (
            f"pivot={pivot['pivot_year']} ({pivot['pivot_reason']})"
            if pivot and pivot.get("pivoted")
            else "no pivot"
        )
        rt = retirements[pw]
        gas_2050 = rt[-1]["gas_retired_gw"] if rt else 0
        cum = cumulative_cost[pw]
        cum_2050 = cum[-1]["cumulative_usd"] if cum else 0
        print(
            f"  P{pw}: ${hc['undiscounted_usd']/1e12:.3f}T undiscounted  "
            f"cfe={hc['achieved_cfe_pct']:.1f}%  {pivot_str}  "
            f"gas_retired_2050={gas_2050:.1f}GW  cum_check=${cum_2050/1e12:.3f}T"
        )


if __name__ == "__main__":
    main()
