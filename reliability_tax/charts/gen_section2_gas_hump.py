"""
gen_section2_gas_hump.py — Generate section2_gas_hump.json

v2 (Apr 2026). HEADLINE chart for the reliability-tax page.

For each ISO × pathway, plots cumulative new-build gas MW at the 2050 endpoint
as a function of achieved CFE %. The central finding under SPEC §24.4+§24.5:
P1a new-gas builds peak at a middle CFE threshold (roughly 75-85%) and then
flatten or decline as the grid pushes past 90% — the builds that came online
to serve medium-CFE residual demand are now stranded.

Schema:
  per_iso[ISO][pathway] = [
    {endpoint, endpoint_label, achieved_cfe_pct, cum_new_gas_mw, cum_new_gas_stranded_mw},
    ...  # one entry per endpoint in the canonical 10-endpoint sweep
  ]

Gas sizing (SPEC §24.5 worst-hour): active_new_gas_fleet_mw is pulled from
the terminal row of tables.annual_buildout[].gas_sizing. Stranded fleet MW is
the sum of initial_cap_mw for vintages with stranded_flag=True in the
tables.new_gas_fleet array.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "reliability_tax" / "charts"))
from data_loader import (  # noqa: E402
    DL_BASE, ISOS, PATHWAYS, ENDPOINTS, _EP_LABEL,
)

OUT_PATH = Path(__file__).parent / "section2_gas_hump.json"
DASH_OUT = REPO_ROOT / "dashboard" / "js" / "reliability-tax" / "section2_gas_hump.json"


def _hump_row(iso: str, pathway: str, ep: float) -> dict | None:
    try:
        run = DL_BASE.get_run(iso, pathway, ep)
    except (KeyError, FileNotFoundError):
        return None
    if not run["feasibility"].get("physical", False):
        return {
            "endpoint": ep,
            "endpoint_label": _EP_LABEL[ep],
            "achieved_cfe_pct": None,
            "cum_new_gas_mw": None,
            "cum_new_gas_stranded_mw": None,
            "available": False,
        }
    bo = run.get("tables", {}).get("annual_buildout", []) or []
    if not bo:
        return None
    terminal = bo[-1]
    cum_mw = float(terminal["gas_sizing"].get("active_new_gas_fleet_mw", 0.0))
    ngf = run.get("tables", {}).get("new_gas_fleet", []) or []
    stranded_mw = sum(
        float(v.get("initial_cap_mw", 0.0))
        for v in ngf if v.get("stranded_flag")
    )
    return {
        "endpoint": ep,
        "endpoint_label": _EP_LABEL[ep],
        "achieved_cfe_pct": round(run.get("achieved_cfe_pct", 0.0), 2),
        "cum_new_gas_mw": round(cum_mw, 0),
        "cum_new_gas_stranded_mw": round(stranded_mw, 0),
        "available": True,
    }


def main() -> None:
    per_iso: dict[str, dict] = {}
    peaks: dict[str, dict[str, dict]] = {}

    for iso in ISOS:
        per_pathway: dict[str, list] = {}
        iso_peaks: dict[str, dict] = {}
        for pathway in PATHWAYS:
            rows = []
            for ep in ENDPOINTS:
                r = _hump_row(iso, pathway, ep)
                if r is not None:
                    rows.append(r)
            per_pathway[pathway] = rows
            # Identify peak new-gas MW endpoint for this pathway
            feasible = [r for r in rows if r.get("available") and r.get("cum_new_gas_mw") is not None]
            if feasible:
                peak = max(feasible, key=lambda r: r["cum_new_gas_mw"])
                iso_peaks[pathway] = {
                    "endpoint_label": peak["endpoint_label"],
                    "achieved_cfe_pct": peak["achieved_cfe_pct"],
                    "cum_new_gas_mw": peak["cum_new_gas_mw"],
                    "cum_new_gas_stranded_mw": peak["cum_new_gas_stranded_mw"],
                }
        per_iso[iso] = per_pathway
        peaks[iso] = iso_peaks

    payload = {
        "per_iso": per_iso,
        "peaks_per_pathway": peaks,
        "meta": {
            "payload_id": "section2_gas_hump",
            "schema_version": 2,
            "note": (
                "Headline chart: cumulative new-build gas MW at 2050 vs achieved CFE %, "
                "per ISO per pathway. Under worst-hour sizing (SPEC §24.5), P1a shows a "
                "hump at middle thresholds — gas built to cover residual peaks becomes "
                "stranded as the grid climbs past 90%."
            ),
            "sizing_method": (
                "active_new_gas_fleet_mw from tables.annual_buildout[-1].gas_sizing. "
                "Sized to 99.97th percentile residual gap per SPEC §24.5. Not ELCC."
            ),
            "stranded_method": (
                "Σ initial_cap_mw for vintages in tables.new_gas_fleet where stranded_flag=True. "
                "Stranded = CF < 15% for 2 consecutive years per SPEC Card F'."
            ),
        },
    }

    OUT_PATH.write_text(json.dumps(payload, indent=2))
    DASH_OUT.parent.mkdir(parents=True, exist_ok=True)
    DASH_OUT.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT_PATH}")
    print(f"Wrote {DASH_OUT}")

    # Summary
    print("\nP1a peak new-gas MW per ISO:")
    for iso in ISOS:
        pk = peaks.get(iso, {}).get("1a")
        if pk:
            print(f"  {iso}: peak at {pk['endpoint_label']} "
                  f"({pk['achieved_cfe_pct']:.1f}% CFE) = {pk['cum_new_gas_mw']:.0f} MW, "
                  f"stranded {pk['cum_new_gas_stranded_mw']:.0f} MW")


if __name__ == "__main__":
    main()
