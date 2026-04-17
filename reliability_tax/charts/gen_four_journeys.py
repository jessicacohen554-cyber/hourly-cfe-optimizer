"""
gen_four_journeys.py — Generate section4_four_journeys.json

v2 (Apr 2026). Ratepayer framing: every cost series surfaced here is
gross-of-any-capacity-market-netting per SPEC Card M'. No net_fom or
capacity_rev_netted values are exposed — those framings misleadingly
implied that cap-market revenue offsets the reliability tax to the
ratepayer, which it does not (the payer changes but the dollar flow is
the same).

Five pathways at a single headline ISO (ERCOT) and endpoint (95% CFE).

Payload:
  annual_buildout: dict[pathway -> list[{year, resources:[{name, twh_per_year, lcoe}], total_twh}]]
  cumulative_cost: dict[pathway -> list[{year, cumulative_usd, annual_usd, achieved_cfe_pct, demand_twh}]]
    — annual_usd is gross ratepayer cost (gen + T&D + fuel + gas FOM + new-gas capex annualization).
       No capacity-market netting.
  pivot_markers:   dict[pathway -> {pivot_year, pivot_reason} | null]
  retirements:     dict[pathway -> list[{year, clean_pct, gas_retired_gw, coal_retired_gw, gas_retired_twh}]]
  headline_costs:  dict[pathway -> {undiscounted_usd, npv_7pct_usd, achieved_cfe_pct, physically_feasible}]
  reliability_tax: dict[pathway -> {total_usd, per_mwh, components}]
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
DASH_OUT = REPO_ROOT / "dashboard" / "js" / "reliability-tax" / "section4_four_journeys.json"
ISO = "ERCOT"
ENDPOINT = 0.95


def _gross_ratepayer_annual_cost(row: dict) -> float:
    """
    Gross ratepayer $ for one year (SPEC Card M' — no capacity market netting).

    Components the ratepayer actually pays:
      gross_operating_usd    — gen LCOE + transmission + variable
      new_gas_annualized_capex_fom_usd  — annualized new-build gas capex + FOM
      existing_gas_fom_carried_usd      — existing-fleet FOM kept online for reliability
      gas_fuel_usd           — new + existing gas fuel burn

    Note: gross_operating_usd in the solver output already excludes gas fuel
    and gas FOM (those are broken out separately above). Summing the above
    four terms gives gross ratepayer cost.
    """
    return (
        float(row.get("gross_operating_usd", 0.0))
        + float(row.get("new_gas_annualized_capex_fom_usd", 0.0))
        + float(row.get("existing_gas_fom_carried_usd", 0.0))
        + float(row.get("gas_fuel_usd", 0.0))
    )


def _buildout_for_pathway(pathway: str) -> list[dict]:
    rows = get_pathway_buildout(ISO, pathway, ENDPOINT)
    result = []
    for row in rows:
        year = row["year"]
        resources = []
        for v in row.get("new_vintages", []):
            resources.append({
                "resource": v["resource"],
                "twh_per_year": round(v.get("twh_per_year", 0.0), 2),
                "locked_lcoe_usd_per_mwh": v.get("locked_lcoe_usd_per_mwh"),
                "tx_adder_usd_per_mwh": v.get("tx_adder_usd_per_mwh", 0.0),
            })
        total_twh = round(sum(r["twh_per_year"] for r in resources), 2)
        # Capture the worst-hour sized gas fleet state for this year
        gas = row.get("gas_sizing", {}) or {}
        result.append({
            "year": year,
            "resources": resources,
            "total_twh_per_year": total_twh,
            "active_new_gas_fleet_mw": round(float(gas.get("active_new_gas_fleet_mw", 0.0)), 0),
            "residual_gap_p99p97_mw": round(float(gas.get("residual_gap_p99p97_mw", 0.0)), 0),
        })
    return result


def _cumulative_cost_for_pathway(pathway: str) -> list[dict]:
    rows = get_pathway_cost(ISO, pathway, ENDPOINT)
    result = []
    cumulative = 0.0
    for row in rows:
        annual = _gross_ratepayer_annual_cost(row)
        cumulative += annual
        result.append({
            "year": row["year"],
            "annual_usd": round(annual, 0),
            "cumulative_usd": round(cumulative, 0),
            "achieved_cfe_pct": row.get("achieved_cfe_pct", 0.0),
            "demand_twh": row.get("demand_twh", 0.0),
            "new_gas_annualized_capex_fom_usd": round(
                float(row.get("new_gas_annualized_capex_fom_usd", 0.0)), 0),
            "existing_gas_fom_carried_usd": round(
                float(row.get("existing_gas_fom_carried_usd", 0.0)), 0),
            "gas_fuel_usd": round(float(row.get("gas_fuel_usd", 0.0)), 0),
            "gas_fleet_cf": row.get("gas_fleet_cf"),
        })
    return result


def _retirement_series_for_pathway(pathway: str) -> list[dict]:
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


def _reliability_tax_for_pathway(run: dict) -> dict:
    rt = run.get("reliability_tax", {}) or {}
    if not rt:
        return {"total_usd": 0.0, "usd_per_mwh": 0.0, "components_usd": {}}
    return {
        "total_usd": rt.get("total_usd", 0.0),
        "usd_per_mwh": rt.get("usd_per_mwh", 0.0),
        "total_demand_mwh": rt.get("total_demand_mwh_2025_2050", 0.0),
        "components_usd": rt.get("components_usd", {}),
    }


def main() -> None:
    annual_buildout: dict[str, list] = {}
    cumulative_cost: dict[str, list] = {}
    pivot_markers: dict[str, dict | None] = {}
    retirements: dict[str, list] = {}
    headline_costs: dict[str, dict] = {}
    reliability_tax_by_pathway: dict[str, dict] = {}

    for pathway in PATHWAYS:
        run = get_run(ISO, pathway, ENDPOINT)
        phys = run["feasibility"].get("physical", False)

        annual_buildout[pathway] = _buildout_for_pathway(pathway) if phys else []
        cumulative_cost[pathway] = _cumulative_cost_for_pathway(pathway) if phys else []
        retirements[pathway] = _retirement_series_for_pathway(pathway) if phys else []
        pivot_markers[pathway] = get_pivot_info(run) if phys else None
        reliability_tax_by_pathway[pathway] = _reliability_tax_for_pathway(run) if phys else {
            "total_usd": 0.0, "usd_per_mwh": 0.0, "components_usd": {},
        }

        headline_costs[pathway] = {
            "undiscounted_gross_usd": run.get("undiscounted_cost_usd", 0.0),
            "npv_at_7pct_gross_usd": run.get("npv_at_7pct", 0.0),
            "achieved_cfe_pct": run.get("achieved_cfe_pct", 0.0),
            "physically_feasible": phys,
        }

    payload = {
        "annual_buildout": annual_buildout,
        "cumulative_cost": cumulative_cost,
        "pivot_markers": pivot_markers,
        "retirements": retirements,
        "headline_costs": headline_costs,
        "reliability_tax_by_pathway": reliability_tax_by_pathway,
        "meta": {
            "payload_id": "section4_four_journeys",
            "schema_version": 2,
            "iso": ISO,
            "endpoint": ENDPOINT,
            "endpoint_pct": ENDPOINT * 100,
            "cost_framing": (
                "Gross ratepayer cost per SPEC Card M'. No capacity-market netting. "
                "Annual cost = gross_operating + new_gas_annualized_capex_fom + "
                "existing_gas_fom_carried + gas_fuel. 'net_annual_cost_usd' and "
                "'capacity_rev_netted_usd' fields from the solver are intentionally "
                "dropped here — the capacity rev changes who pays, not whether a "
                "dollar leaves the ratepayer system."
            ),
            "gas_sizing_basis": (
                "active_new_gas_fleet_mw is worst-hour sized (SPEC §24.5 "
                "99.97th percentile of residual gap with RA margin on demand), "
                "not ELCC."
            ),
            "note": (
                "ERCOT 2025–2050 annual buildout per pathway at the 95% CFE "
                "endpoint. Pivot markers show year Pathway 2a/2b unlocks clean "
                "firm. Retirement overlay = cumulative GW of existing fleet "
                "economically displaced."
            ),
        },
    }

    OUT_PATH.write_text(json.dumps(payload, indent=2))
    DASH_OUT.parent.mkdir(parents=True, exist_ok=True)
    DASH_OUT.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT_PATH}")
    print(f"Wrote {DASH_OUT}")

    # Sanity print
    print(f"\n--- section4_four_journeys quick summary ({ISO} @ {ENDPOINT*100:.0f}%) ---")
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
        rtax = reliability_tax_by_pathway[pw]
        print(
            f"  P{pw}: ${hc['undiscounted_gross_usd']/1e12:.2f}T undisc.  "
            f"cfe={hc['achieved_cfe_pct']:.1f}%  {pivot_str}  "
            f"rtax=${rtax['usd_per_mwh']:.1f}/MWh  "
            f"gas_retired_2050={gas_2050:.1f}GW  cum_check=${cum_2050/1e12:.2f}T"
        )


if __name__ == "__main__":
    main()
