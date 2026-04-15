"""
gen_act1_trilemma.py — Generate act1_trilemma.json

Data lineage:
  - No data values. Metadata only.
  - Describes the energy trilemma framing and pathway-to-corner mapping
    used by the Act 1 trilemma visualization.

Payload structure:
  corners: list of 3 corner labels with descriptions
  pathways: dict[pathway_id -> {name, description, corner_weights, outcome}]
    corner_weights: {"affordability": float, "reliability": float, "decarbonization": float}
    All weights are qualitative/categorical (null/low/medium/high) — no data values.
  framing_copy:
    headline: str
    body: str
    source_note: str
  meta: standard metadata block
"""
from __future__ import annotations

import json
from pathlib import Path

OUT_PATH = Path(__file__).parent / "act1_trilemma.json"


def main() -> None:
    payload = {
        "corners": [
            {
                "id": "affordability",
                "label": "Affordable",
                "description": (
                    "Minimizing total system cost — capital, O&M, fuel — for electricity customers "
                    "over the planning horizon. Includes cost of new-build capacity, dispatch costs, "
                    "and stranded asset write-offs."
                ),
            },
            {
                "id": "reliability",
                "label": "Reliable",
                "description": (
                    "Maintaining sufficient dispatchable capacity to serve load in every hour, "
                    "including the 100 worst demand/low-VRE coincidence hours per year. "
                    "Measured by hourly CFE matching rate and residual gas capacity factor."
                ),
            },
            {
                "id": "decarbonization",
                "label": "Clean",
                "description": (
                    "Achieving a high hourly CFE match — the fraction of consumption covered "
                    "by zero-carbon generation in the same hour, same ISO. Endpoints range "
                    "from 90% to 99.9% by 2050."
                ),
            },
        ],
        "pathways": {
            "1": {
                "id": "1",
                "name": "VRE + Batteries Only",
                "description": (
                    "Solar, wind, and battery storage with no new clean firm capacity. "
                    "Optimizes for affordability and decarbonization up to the VRE+storage ceiling; "
                    "reliability is maintained by legacy gas fleet operating at declining utilization."
                ),
                "corner_weights": {
                    "affordability": "high",
                    "reliability": "low",
                    "decarbonization": "medium",
                },
                "tension": (
                    "At high CFE thresholds (≥95%), VRE overbuild drives marginal costs "
                    "above the clean firm crossover while the residual gas fleet becomes "
                    "uneconomic but still operationally necessary — the reliability tax."
                ),
                "outcome_label": "Overbuild + stranded gas",
            },
            "2a": {
                "id": "2a",
                "name": "Behavioral Reactive Pivot",
                "description": (
                    "VRE-first until Pathway 1 hits its 90% CFE plateau for the ISO; "
                    "clean firm unlocks at that pivot year, starting at FOAK cost on the "
                    "Wright's Law curve. Pre-pivot assets are locked (Card E)."
                ),
                "corner_weights": {
                    "affordability": "medium",
                    "reliability": "medium",
                    "decarbonization": "medium",
                },
                "tension": (
                    "Pivot arrives after years of VRE-only overbuild. Pre-pivot solar, wind, "
                    "and battery assets are only partially stranded relative to Pathway 3. "
                    "FOAK penalty applies from pivot year forward, not from 2025."
                ),
                "outcome_label": "Late pivot, partial stranding",
            },
            "2b": {
                "id": "2b",
                "name": "Economic Reactive Pivot",
                "description": (
                    "VRE-first until the marginal $/CFE% of the next VRE+battery addition "
                    "exceeds the cheapest available clean firm LCOE for that year and ISO. "
                    "Pivot is triggered by real-time cost math, not an exogenous target."
                ),
                "corner_weights": {
                    "affordability": "high",
                    "reliability": "medium",
                    "decarbonization": "medium",
                },
                "tension": (
                    "Economic pivot fires earlier than behavioral in high-resource ISOs "
                    "(ERCOT, SPP) and later in constrained ISOs (NEISO). Stranding profile "
                    "is ISO-specific and path-dependent."
                ),
                "outcome_label": "Cost-triggered pivot, ISO-varied timing",
            },
            "3": {
                "id": "3",
                "name": "Clean Firm Proactive",
                "description": (
                    "Clean firm (nuclear / CCS-CCGT / geothermal) available from year 1 "
                    "on the standard Wright's Law curve starting from FOAK. Optimizer "
                    "co-selects VRE and clean firm to minimize NPV@7% at each CFE endpoint."
                ),
                "corner_weights": {
                    "affordability": "medium",
                    "reliability": "high",
                    "decarbonization": "high",
                },
                "tension": (
                    "Higher near-term costs due to FOAK clean firm premium in early years; "
                    "lower long-run costs due to avoided overbuild and gas stranding. "
                    "Requires institutional confidence that clean firm will be needed."
                ),
                "outcome_label": "Proactive buildout, lowest total cost",
            },
        },
        "framing_copy": {
            "headline": "Every grid faces the same trilemma — the pathways differ in which corner they sacrifice first.",
            "body": (
                "The energy trilemma is the constraint that grids face when they optimize without "
                "explicit reliability design: the system can be clean, affordable, or reliable — "
                "but pursuing any two without the third creates hidden costs. "
                "The reliability tax is what you pay when reliability is treated as a residual "
                "rather than a design objective."
            ),
            "reliability_tax_definition": (
                "The reliability tax = the fixed costs of the gas fleet that must remain "
                "online to fill hours that VRE and storage cannot economically serve, "
                "net of capacity market revenues, expressed as $/MWh of total consumption."
            ),
            "source_note": "Pathway and cost data: see MANIFEST.json and per-run JSONs in analysis/reliability-tax/data/.",
        },
        "axis_hint": {
            "type": "ternary",
            "note": (
                "Render as ternary/triangle plot. Each pathway is a point whose "
                "corner_weights determine position. Weights are ordinal (low/medium/high); "
                "map to [0.1, 0.5, 0.9] for rendering. No quantitative axis scale."
            ),
        },
        "meta": {
            "payload_id": "act1_trilemma",
            "data_values": False,
            "note": (
                "Metadata only — no quantitative values. Corner weights are qualitative "
                "ordinal labels for visualization positioning only."
            ),
            "version": "1.0",
        },
    }

    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT_PATH}")
    print("\n--- act1_trilemma summary ---")
    print(f"  Corners: {[c['id'] for c in payload['corners']]}")
    print(f"  Pathways: {list(payload['pathways'].keys())}")
    print(f"  Data values: {payload['meta']['data_values']}")


if __name__ == "__main__":
    main()
