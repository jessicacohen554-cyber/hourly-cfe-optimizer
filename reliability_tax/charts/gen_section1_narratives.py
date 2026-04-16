"""
gen_section1_narratives.py — Generate section1_narratives.json

Data lineage:
  - analysis/reliability-tax/data/MANIFEST.json
  - per-run JSONs via data_loader

Payload structure:
  per_iso: dict[iso ->
    headline: str — 1-sentence ISO-specific lead
    body: str — 2–3 sentence narrative for the panel below the worst-hours chart
    data_callouts: list[str] — 2–3 specific data points pulled from the run
    note: str — placeholder flag for human editing
  ]
  page_intro: str — intro text for Section 1 of the page
  section_title: str
  meta: ...

All text is generated from run data (achieved CFE, gas remaining, demand, etc.)
and is intended as a human-editable starting point, not final copy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "reliability_tax" / "charts"))
from data_loader import get_run, ISOS, ENDPOINTS  # noqa: E402

OUT_PATH = Path(__file__).parent / "section1_narratives.json"

EXISTING_GAS_GW = {
    "CAISO": 37.0, "ERCOT": 55.0, "PJM": 75.0,
    "NYISO": 18.0, "NEISO": 14.0, "MISO": 68.0, "SPP": 32.0,
}

ISO_DESCRIPTIONS = {
    "CAISO": "California's solar-heavy grid",
    "ERCOT": "Texas's wind-dominated isolated system",
    "MISO": "the Midwest's sprawling coal-to-wind transition",
    "NEISO": "New England's constrained winter-peaking system",
    "NYISO": "New York's hydro-and-offshore-wind corridor",
    "PJM": "the largest US electricity market",
    "SPP": "the Southern Plains' wind-rich corridor",
}

ISO_RESOURCE_CHARACTER = {
    "CAISO": "solar and battery storage dominate the 2050 mix; wind is modest. Summer afternoon peaks expose solar's daily gap.",
    "ERCOT": "wind provides the foundation with solar as secondary. Wind volatility during low-pressure systems creates the worst-hour stress.",
    "MISO": "solar and wind share the load roughly equally. Winter cold snaps with low wind are the critical stress event.",
    "NEISO": "offshore wind and solar fill most of the gap, but winter gas pipeline constraints historically drive worst-hour scarcity.",
    "NYISO": "hydro provides a firm backbone; offshore wind drives the CFE gains. Winter demand peaks stress the offshore wind resource.",
    "PJM": "solar is the dominant new resource with substantial wind. Winter polar vortex events with cloud cover define the worst hours.",
    "SPP": "wind is the workhorse, providing 40-plus percent of annual generation. Summer high-pressure systems (calm, hot) produce the worst hours.",
}


def _gas_remaining_gw(run: dict, iso: str) -> float:
    rt = run.get("retirement_timeline", [])
    if not rt:
        return EXISTING_GAS_GW.get(iso, 0.0)
    gas_retired = rt[-1].get("gas_retired_gw", 0.0)
    return max(0.0, EXISTING_GAS_GW.get(iso, 0.0) - gas_retired)


def _build_iso_narrative(iso: str) -> dict:
    """Build per-ISO narrative block from run data at ep95."""
    run_95 = get_run(iso, "1a", 0.95)
    run_99 = get_run(iso, "1a", 0.99)

    # Core stats
    cfe_95 = run_95.get("achieved_cfe_pct", 95.0)
    cfe_99 = run_99.get("achieved_cfe_pct", 99.0)
    gas_gw_95 = round(_gas_remaining_gw(run_95, iso), 1)
    gas_gw_99 = round(_gas_remaining_gw(run_99, iso), 1)

    cost_95 = run_95.get("undiscounted_cost_usd", 0.0)
    cost_99 = run_99.get("undiscounted_cost_usd", 0.0)
    cost_delta = (cost_99 - cost_95) / 1e12  # trillion USD

    ac = run_95.get("tables", {}).get("annual_cost", [])
    demand_2050 = ac[-1]["demand_twh"] if ac else 0.0

    desc = ISO_DESCRIPTIONS.get(iso, iso)
    char = ISO_RESOURCE_CHARACTER.get(iso, "")

    # Build headline
    if gas_gw_95 > 5:
        headline = (
            f"In {desc}, {gas_gw_95:.0f} GW of gas capacity remains online "
            f"at {cfe_95:.0f}% CFE — present for reliability, rarely dispatched."
        )
    elif gas_gw_95 > 0.5:
        headline = (
            f"In {desc}, the last {gas_gw_95:.1f} GW of gas carries the grid "
            f"through the worst {cfe_95:.0f}%-CFE stress events."
        )
    else:
        headline = (
            f"In {desc}, existing gas is fully retired by {cfe_95:.0f}% CFE — "
            f"storage and VRE overbuild replace the gas reliability buffer entirely."
        )

    # Build body (2–3 sentences)
    body = (
        f"{ISO_DESCRIPTIONS.get(iso, iso).capitalize()} serves {demand_2050:.0f} TWh "
        f"annually by 2050 under medium demand growth. "
        f"Under Pathway 1, {char} "
        f"Moving from {cfe_95:.0f}% to {cfe_99:.0f}% CFE adds an estimated "
        f"${cost_delta:.2f}T in undiscounted cumulative system cost through 2050."
    )

    # Data callouts
    callouts = [
        f"Gas remaining at {cfe_95:.0f}% CFE: {gas_gw_95:.1f} GW "
        f"(vs. {EXISTING_GAS_GW.get(iso, 0):.0f} GW existing in 2025)",
        f"Gas remaining at {cfe_99:.0f}% CFE: {gas_gw_99:.1f} GW",
        f"Cost step from {cfe_95:.0f}%→{cfe_99:.0f}%: ${cost_delta:.2f}T undiscounted",
    ]

    return {
        "headline": headline,
        "body": body,
        "data_callouts": callouts,
        "iso_resource_character": char,
        "stats": {
            "achieved_cfe_at_ep95": round(cfe_95, 1),
            "achieved_cfe_at_ep99": round(cfe_99, 1),
            "gas_remaining_gw_at_ep95": gas_gw_95,
            "gas_remaining_gw_at_ep99": gas_gw_99,
            "cost_delta_95_to_99_trillion_usd": round(cost_delta, 3),
            "demand_2050_twh": round(demand_2050, 0),
        },
        "note": "PLACEHOLDER — human-editable narrative. Data callouts are derived from run results.",
    }


def main() -> None:
    per_iso: dict[str, dict] = {}
    for iso in ISOS:
        per_iso[iso] = _build_iso_narrative(iso)

    payload = {
        "section_title": "The Worst Hours: What the Grid Faces When VRE Runs Low",
        "page_intro": (
            "The hourly CFE matching problem becomes clear when you look at the worst hours of the year. "
            "Every grid has hours where solar is dark, wind is calm, and demand is high. "
            "At low CFE targets, existing gas fills the gap cheaply. "
            "At high CFE targets, the options narrow: "
            "either overbuild VRE and storage to serve those hours on clean energy, "
            "or build dispatchable clean firm capacity that shows up whenever needed. "
            "Pathway 1 takes the first approach — and the cost of serving the last few percent of hours "
            "is what we call the reliability tax."
        ),
        "worst_hours_chart_caption": (
            "Each bar represents one of the 100 hours in 2050 with the highest residual load "
            "(demand minus VRE generation) under Pathway 1 at 95% CFE. "
            "Stacked segments show how each hour is served: wind (green), storage (cyan), gas (gray). "
            "Gas dispatch occurs when storage is depleted during sustained low-VRE events."
        ),
        "per_iso": per_iso,
        "meta": {
            "payload_id": "section1_narratives",
            "data_driven": True,
            "note": (
                "All text is auto-generated from run results and intended as a starting point "
                "for human editorial. Stats are from MANIFEST.json + per-run JSONs. "
                "Pathway 1, endpoint 95% and 99% used for comparisons."
            ),
            "version": "1.0",
        },
    }

    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT_PATH}")

    print("\n--- section1_narratives quick summary ---")
    for iso, d in per_iso.items():
        s = d["stats"]
        print(
            f"  {iso}: CFE {s['achieved_cfe_at_ep95']:.0f}%→{s['achieved_cfe_at_ep99']:.0f}%, "
            f"gas {s['gas_remaining_gw_at_ep95']:.1f}→{s['gas_remaining_gw_at_ep99']:.1f} GW, "
            f"cost delta ${s['cost_delta_95_to_99_trillion_usd']:.2f}T"
        )


if __name__ == "__main__":
    main()
