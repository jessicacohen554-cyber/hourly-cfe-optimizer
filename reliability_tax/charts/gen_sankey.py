"""
gen_sankey.py — Generate section5_stranding_sankey.json

v2 (Apr 2026). Card K' framing: ABSOLUTE stranded capex per-vintage, not the
old Card K comparative-to-P3 framing.

Primary data source: tables.new_gas_fleet[].stranded_capex_usd for vintages
with stranded_flag=True. A vintage is "stranded" (SPEC Card F') when its
capacity factor falls below 15% for two consecutive years before book-life end.
When that trigger fires, recovered_revenue_per_kw is locked and the
unrecovered capital is charged against the ratepayer as stranded capex.

Secondary data block: stranding_ledger (VRE + storage comparative stranding)
is preserved as a supplementary read on "how much extra the pathway built
vs a hypothetical destination-first plan." This is retained in the payload
but is NOT the primary Sankey flow in v2.

Sankey structure (v2):
  Sources: one node per (pathway, vintage_year) with absolute stranded capex
  Sink:    "stranded_new_gas" (single absorbing node)
  Flow widths: stranded_capex_usd per vintage

Endpoint: 95% CFE, aggregated across all 7 ISOs. Per-ISO detail is also
included in the payload so the chart layer can stratify.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "reliability_tax" / "charts"))
from data_loader import (  # noqa: E402
    DL_BASE, ISOS, PATHWAYS,
)

OUT_PATH = Path(__file__).parent / "section5_stranding_sankey.json"
DASH_OUT = REPO_ROOT / "dashboard" / "js" / "reliability-tax" / "section5_stranding_sankey.json"
ENDPOINT = 0.95
# Card K' covers all 5 pathways — gas can be built and stranded in any
# pathway that relies on new-build gas. P3 (clean firm build-out) is the
# low-gas reference but can still show some stranding in edge ISOs.
SANKEY_PATHWAYS = PATHWAYS  # ["1", "1a", "2a", "2b", "3"] per §24.1 v2


def _collect_new_gas_stranding(iso: str, pathway: str) -> dict:
    """Pull per-vintage stranded new-gas data for one run."""
    try:
        run = DL_BASE.get_run(iso, pathway, ENDPOINT)
    except (KeyError, FileNotFoundError):
        return {"feasible": False, "vintages": []}
    if not run["feasibility"].get("physical", False):
        return {"feasible": False, "vintages": []}

    ngf = run.get("tables", {}).get("new_gas_fleet", []) or []
    stranded_vintages = [
        {
            "year_built": v["year_built"],
            "initial_cap_mw": float(v.get("initial_cap_mw", 0.0)),
            "stranded_year": v.get("stranded_year"),
            "stranded_capex_usd": float(v.get("stranded_capex_usd", 0.0)),
            "recovered_revenue_per_kw": float(v.get("recovered_revenue_per_kw", 0.0)),
        }
        for v in ngf if v.get("stranded_flag")
    ]

    total_stranded_mw = sum(v["initial_cap_mw"] for v in stranded_vintages)
    total_stranded_capex = sum(v["stranded_capex_usd"] for v in stranded_vintages)

    # Also capture all-vintage totals (including non-stranded) for context
    all_vintages_mw = sum(float(v.get("initial_cap_mw", 0.0)) for v in ngf)

    return {
        "feasible": True,
        "vintages": stranded_vintages,
        "total_stranded_mw": total_stranded_mw,
        "total_stranded_capex_usd": total_stranded_capex,
        "all_new_gas_vintages_mw": all_vintages_mw,
        "stranded_share_of_new_gas_pct": (
            round(total_stranded_mw / all_vintages_mw * 100, 1)
            if all_vintages_mw > 0 else 0.0
        ),
    }


def _collect_vre_storage_stranding(iso: str, pathway: str) -> dict:
    """Pull comparative VRE + storage stranding from stranding_ledger."""
    stranding = DL_BASE.get_pathway_stranding(iso, pathway, ENDPOINT)
    ledger = stranding["ledger"]
    # Each ledger row: {resource, pathway_twh, pathway3_twh, stranded_twh, stranded_book_value_usd}
    resource_breakdown: dict[str, float] = {}
    for row in ledger:
        bv = float(row.get("stranded_book_value_usd", 0.0))
        if bv > 0:
            resource_breakdown[row["resource"]] = resource_breakdown.get(row["resource"], 0.0) + bv
    return {
        "by_resource_usd": resource_breakdown,
        "total_usd": sum(resource_breakdown.values()),
    }


def main() -> None:
    # Primary Sankey data — absolute new-gas stranding per (pathway, vintage)
    # Aggregated across ISOs by (pathway, vintage_year).
    vintage_flows: dict[tuple[str, int], dict] = defaultdict(
        lambda: {"capex_usd": 0.0, "mw": 0.0, "isos": []}
    )

    per_iso_pathway_new_gas: dict[str, dict[str, dict]] = {}
    per_iso_pathway_vre_storage: dict[str, dict[str, dict]] = {}

    pathway_totals: dict[str, dict] = {
        pw: {
            "new_gas_stranded_capex_usd": 0.0,
            "new_gas_stranded_mw": 0.0,
            "new_gas_all_vintages_mw": 0.0,
            "vre_storage_comparative_stranded_usd": 0.0,
            "feasible_isos": [],
        }
        for pw in SANKEY_PATHWAYS
    }

    iso_totals: dict[str, dict] = {}

    for iso in ISOS:
        per_iso_pathway_new_gas[iso] = {}
        per_iso_pathway_vre_storage[iso] = {}
        iso_totals[iso] = {
            "new_gas_stranded_capex_usd": 0.0,
            "new_gas_stranded_mw": 0.0,
        }

        for pathway in SANKEY_PATHWAYS:
            ng = _collect_new_gas_stranding(iso, pathway)
            vs = _collect_vre_storage_stranding(iso, pathway) if ng["feasible"] else {
                "by_resource_usd": {},
                "total_usd": 0.0,
            }

            per_iso_pathway_new_gas[iso][pathway] = ng
            per_iso_pathway_vre_storage[iso][pathway] = vs

            if not ng["feasible"]:
                continue

            pt = pathway_totals[pathway]
            pt["new_gas_stranded_capex_usd"] += ng["total_stranded_capex_usd"]
            pt["new_gas_stranded_mw"] += ng["total_stranded_mw"]
            pt["new_gas_all_vintages_mw"] += ng["all_new_gas_vintages_mw"]
            pt["vre_storage_comparative_stranded_usd"] += vs["total_usd"]
            pt["feasible_isos"].append(iso)

            iso_totals[iso]["new_gas_stranded_capex_usd"] += ng["total_stranded_capex_usd"]
            iso_totals[iso]["new_gas_stranded_mw"] += ng["total_stranded_mw"]

            # Aggregate per-vintage flows
            for v in ng["vintages"]:
                key = (pathway, int(v["year_built"]))
                vintage_flows[key]["capex_usd"] += v["stranded_capex_usd"]
                vintage_flows[key]["mw"] += v["initial_cap_mw"]
                vintage_flows[key]["isos"].append(iso)

    # Build Sankey nodes + flows
    source_nodes = []
    for (pw, yr), agg in sorted(vintage_flows.items()):
        source_nodes.append({
            "id": f"P{pw}_vintage_{yr}",
            "pathway": pw,
            "vintage_year": yr,
            "stranded_capex_usd": round(agg["capex_usd"], 0),
            "stranded_mw": round(agg["mw"], 0),
            "iso_count": len(set(agg["isos"])),
            "isos": sorted(set(agg["isos"])),
        })

    flows = [
        {
            "source": node["id"],
            "target": "stranded_new_gas",
            "value_usd": node["stranded_capex_usd"],
            "mw": node["stranded_mw"],
            "pathway": node["pathway"],
            "vintage_year": node["vintage_year"],
        }
        for node in source_nodes if node["stranded_capex_usd"] > 0
    ]

    total_stranded_capex = sum(f["value_usd"] for f in flows)
    total_stranded_mw = sum(f["mw"] for f in flows)

    sink_nodes = [{
        "id": "stranded_new_gas",
        "label": "Stranded new-build gas capital",
        "total_usd": total_stranded_capex,
        "total_mw": total_stranded_mw,
        "definition": (
            "Unrecovered capital on new-build gas vintages that hit the "
            "Card F' trigger (CF<15% for 2 consecutive years before book-life). "
            "When the trigger fires the vintage locks recovered_revenue_per_kw "
            "and the remainder of the $/kW book value is charged as stranded."
        ),
    }]

    # Aggregate shares
    for pw in SANKEY_PATHWAYS:
        pt = pathway_totals[pw]
        pt["stranded_share_of_new_gas_pct"] = (
            round(pt["new_gas_stranded_mw"] / pt["new_gas_all_vintages_mw"] * 100, 1)
            if pt["new_gas_all_vintages_mw"] > 0 else 0.0
        )

    payload = {
        "nodes": {"sources": source_nodes, "sinks": sink_nodes},
        "flows": flows,
        "pathway_totals": pathway_totals,
        "iso_totals": iso_totals,
        "per_iso_pathway": {
            "new_gas_stranding": per_iso_pathway_new_gas,
            "vre_storage_comparative_stranding": per_iso_pathway_vre_storage,
        },
        "headline": {
            "total_stranded_new_gas_capex_usd": round(total_stranded_capex, 0),
            "total_stranded_new_gas_mw": round(total_stranded_mw, 0),
            "n_vintages": len(source_nodes),
            "endpoint_cfe_pct": ENDPOINT * 100,
        },
        "framing_copy": (
            "Every dollar to the right of this diagram is capital that ratepayers "
            "put into new-build gas which the plant will never earn back. The "
            "trigger is Card F': two consecutive years of capacity factor below "
            "15%, which locks recovery at the revenue earned to that date."
        ),
        "meta": {
            "payload_id": "section5_stranding_sankey",
            "schema_version": 2,
            "endpoint": ENDPOINT,
            "endpoint_pct": ENDPOINT * 100,
            "stranding_definition": (
                "Absolute stranded capex per SPEC Card K'/F'. Not comparative to P3."
            ),
            "primary_source": "tables.new_gas_fleet[] where stranded_flag=True",
            "secondary_source": "tables.stranding_ledger (retained for VRE/storage context only)",
            "sizing_basis": (
                "New-gas capacity on each vintage is worst-hour sized (SPEC §24.5 "
                "99.97th percentile residual gap), not ELCC. Stranded capex then "
                "follows from the locked revenue recovery at Card F' trigger."
            ),
            "note": (
                "Flow widths proportional to absolute stranded capex ($USD). "
                "Dropped old Card K comparative framing in favor of Card K' absolute. "
                "Seed vintages in the SPEC §24.8 terminal_ledger (cod_year=2024, "
                "locked_lcoe=0) represent pre-existing fleet and are NOT consumed "
                "by this Sankey — only new-gas vintages in tables.new_gas_fleet with "
                "stranded_flag=True contribute. year_built is by construction ≥ 2025."
            ),
        },
    }

    OUT_PATH.write_text(json.dumps(payload, indent=2))
    DASH_OUT.parent.mkdir(parents=True, exist_ok=True)
    DASH_OUT.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT_PATH}")
    print(f"Wrote {DASH_OUT}")

    # Sanity print
    print("\n--- section5_stranding_sankey (Card K') quick summary ---")
    print(f"Total stranded new-gas capex: ${total_stranded_capex/1e12:.2f}T across "
          f"{total_stranded_mw/1e3:.0f} GW over {len(source_nodes)} vintage nodes")
    print()
    print("Per pathway:")
    for pw in SANKEY_PATHWAYS:
        pt = pathway_totals[pw]
        print(f"  P{pw}: ${pt['new_gas_stranded_capex_usd']/1e12:.2f}T "
              f"stranded ({pt['new_gas_stranded_mw']/1e3:.0f} GW, "
              f"{pt['stranded_share_of_new_gas_pct']:.0f}% of new gas) "
              f"over {len(pt['feasible_isos'])} ISOs")
    print()
    print("Per ISO:")
    for iso in ISOS:
        it = iso_totals[iso]
        print(f"  {iso}: ${it['new_gas_stranded_capex_usd']/1e9:.1f}B "
              f"({it['new_gas_stranded_mw']/1e3:.1f} GW)")


if __name__ == "__main__":
    main()
