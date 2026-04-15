"""
gen_sankey.py — Generate section5_stranding_sankey.json

Data lineage:
  - analysis/reliability-tax/data/MANIFEST.json
  - per-run stranding_ledger tables (Card K revised)
  via data_loader.get_pathway_stranding()

Methodology (Card K revised):
  stranded_capacity[pathway, resource] = max(0, capacity[pathway] - capacity[Pathway 3])
  evaluated at 2050, per resource, per ISO.
  Book value of unrecovered capital = stranded_book_value_usd in the ledger.

  Clean firm excluded from stranding (Pathway 3's clean firm IS the destination).
  Transmission excluded (Card I, bubble-node model).

Sankey structure:
  nodes:
    sources: pathways 1, 2a, 2b (4 source nodes)
    fates:
      - "delivered_value"  (matches Pathway 3 build) — width = P3 cost
      - "stranded"         (excess over Pathway 3)   — width = stranded book value

  flows: list[{source_pathway, fate, resource, book_value_usd, npv_eol_usd}]

  vre_curtailment_crosscheck: per-pathway, per-ISO curtailment rates at endpoint

Endpoint: 95% CFE, all 7 ISOs aggregated.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "reliability_tax" / "charts"))
from data_loader import (  # noqa: E402
    get_run, get_pathway_stranding, is_feasible, ISOS, PATHWAYS,
)

OUT_PATH = Path(__file__).parent / "section5_stranding_sankey.json"
ENDPOINT = 0.95
STRANDING_PATHWAYS = ["1", "2a", "2b"]  # Pathway 3 is the reference, not a source


def main() -> None:
    # Per-pathway aggregate stranding totals across all feasible ISOs
    pathway_totals: dict[str, dict] = {
        pw: {
            "total_stranded_book_value_usd": 0.0,
            "delivered_value_usd": 0.0,   # = pathway total cost - stranded
            "total_pathway_cost_usd": 0.0,
            "resource_breakdown": {},     # resource -> stranded_book_value_usd
            "iso_count": 0,
            "feasible_isos": [],
        }
        for pw in STRANDING_PATHWAYS
    }

    # Per-flow detail for Sankey
    flows: list[dict] = []

    # VRE curtailment cross-check (per pathway, per ISO)
    curtailment_crosscheck: dict[str, dict] = {pw: {} for pw in STRANDING_PATHWAYS}

    for iso in ISOS:
        for pathway in STRANDING_PATHWAYS:
            run = get_run(iso, pathway, ENDPOINT)
            if not is_feasible(run, "physical"):
                continue

            stranding = get_pathway_stranding(iso, pathway, ENDPOINT)
            path_cost = run.get("undiscounted_cost_usd", 0.0)

            total_stranded = stranding["total_stranded_usd"]
            delivered = max(0.0, path_cost - total_stranded)

            pt = pathway_totals[pathway]
            pt["total_stranded_book_value_usd"] += total_stranded
            pt["delivered_value_usd"] += delivered
            pt["total_pathway_cost_usd"] += path_cost
            pt["iso_count"] += 1
            pt["feasible_isos"].append(iso)

            # Resource breakdown
            for row in stranding["ledger"]:
                res = row["resource"]
                bv = row.get("stranded_book_value_usd", 0.0)
                pt["resource_breakdown"][res] = pt["resource_breakdown"].get(res, 0.0) + bv

            # Individual flows for Sankey
            # Delivered value flow
            if delivered > 0:
                flows.append({
                    "source_pathway": pathway,
                    "fate": "delivered_value",
                    "iso": iso,
                    "resource": "all",
                    "book_value_usd": delivered,
                    "note": "Matches Pathway 3 destination build",
                })

            # Stranded flows by resource
            for row in stranding["ledger"]:
                if row.get("stranded_book_value_usd", 0) > 0:
                    flows.append({
                        "source_pathway": pathway,
                        "fate": "stranded",
                        "iso": iso,
                        "resource": row["resource"],
                        "stranded_twh": row["stranded_twh"],
                        "book_value_usd": row["stranded_book_value_usd"],
                        "note": "Exceeds Pathway 3 build — would not have been needed",
                    })

            # VRE curtailment cross-check
            curtailment = stranding["vre_curtailment"]
            if any(v > 0 for v in curtailment.values()):
                curtailment_crosscheck[pathway][iso] = {
                    "curtailment_rates": curtailment,
                    "flag": any(v > 0.5 for v in curtailment.values()),
                    "note": (
                        "Flag: VRE asset >50% curtailment AND not comparatively stranded "
                        "warrants investigation per Card K cross-check."
                        if any(v > 0.5 for v in curtailment.values())
                        else "Curtailment below 50% threshold"
                    ),
                }

    # Sankey node definitions
    nodes = {
        "sources": [
            {"id": f"pathway_{pw}", "label": f"Pathway {pw}",
             "total_cost_usd": pathway_totals[pw]["total_pathway_cost_usd"]}
            for pw in STRANDING_PATHWAYS
        ],
        "fates": [
            {
                "id": "delivered_value",
                "label": "Delivered value (matches Pathway 3 build)",
                "total_usd": sum(pathway_totals[pw]["delivered_value_usd"] for pw in STRANDING_PATHWAYS),
            },
            {
                "id": "stranded",
                "label": "Stranded (excess over Pathway 3)",
                "total_usd": sum(pathway_totals[pw]["total_stranded_book_value_usd"] for pw in STRANDING_PATHWAYS),
                "description": (
                    "'Stranded' here means capital spent on capacity that a "
                    "destination-optimized plan wouldn't have required to reach "
                    "the same CFE target."
                ),
            },
        ],
    }

    # Aggregate resource breakdown across all pathways (for stranded bucket)
    all_resource_stranding: dict[str, float] = {}
    for pw_data in pathway_totals.values():
        for res, bv in pw_data["resource_breakdown"].items():
            all_resource_stranding[res] = all_resource_stranding.get(res, 0.0) + bv

    payload = {
        "nodes": nodes,
        "flows": flows,
        "pathway_totals": {
            pw: {
                **pathway_totals[pw],
                "stranded_share_pct": round(
                    pathway_totals[pw]["total_stranded_book_value_usd"]
                    / pathway_totals[pw]["total_pathway_cost_usd"] * 100, 1
                ) if pathway_totals[pw]["total_pathway_cost_usd"] > 0 else 0.0,
            }
            for pw in STRANDING_PATHWAYS
        },
        "resource_stranding_aggregate": all_resource_stranding,
        "vre_curtailment_crosscheck": curtailment_crosscheck,
        "framing_copy": (
            "'Stranded' here means capital spent on capacity that a "
            "destination-optimized plan wouldn't have required to reach "
            "the same CFE target."
        ),
        "meta": {
            "endpoint": ENDPOINT,
            "endpoint_pct": 95.0,
            "reference_pathway": "3",
            "stranding_definition": "Card K revised — comparative-to-Pathway-3",
            "excluded": ["clean_firm", "transmission"],
            "note": (
                "Flow widths proportional to book value of unrecovered capital at 2050. "
                "Clean firm excluded from stranding per Card K. Transmission excluded per Card I."
            ),
        },
    }

    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT_PATH}")

    # Sanity print
    print("\n--- section5_stranding_sankey quick summary ---")
    for pw in STRANDING_PATHWAYS:
        pt = pathway_totals[pw]
        pct = pt["total_stranded_book_value_usd"] / pt["total_pathway_cost_usd"] * 100 if pt["total_pathway_cost_usd"] > 0 else 0
        print(
            f"  P{pw}: total cost ${pt['total_pathway_cost_usd']/1e12:.3f}T  "
            f"stranded ${pt['total_stranded_book_value_usd']/1e12:.3f}T ({pct:.1f}%)  "
            f"delivered ${pt['delivered_value_usd']/1e12:.3f}T  "
            f"({len(pt['feasible_isos'])} ISOs)"
        )
    print("\n  Resource stranding aggregate:")
    for res, bv in sorted(all_resource_stranding.items(), key=lambda x: -x[1]):
        print(f"    {res}: ${bv/1e12:.3f}T")
    total_stranded = sum(pt["total_stranded_book_value_usd"] for pt in pathway_totals.values())
    print(f"\n  TOTAL stranded across all pathways: ${total_stranded/1e12:.3f}T")


if __name__ == "__main__":
    main()
