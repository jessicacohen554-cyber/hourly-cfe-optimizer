"""
gen_hockey_stick.py — Generate section1_hockey_stick.json

Data lineage:
  - analysis/reliability-tax/data/MANIFEST.json
  - per-run JSONs via data_loader.marginal_cost_per_cfe_pct()

Payload structure:
  per_iso: dict[iso -> list[segment]]
    Each segment:
      from_cfe_pct, to_cfe_pct
      from_endpoint, to_endpoint
      marginal_usd_per_cfe_pct  (null if physically infeasible or delta_cfe=0)
      from_feasible_physical, to_feasible_physical

  annotations:
    plateau_90:   dict[iso -> {x_cfe_pct, y_marginal_usd, note}]
                  — annotation position at the 90% plateau kink
    econ_crossover: dict[iso -> {x_cfe_pct, y_marginal_usd, note}]
                  — first segment where marginal P1 cost > median P3 annual LCOE

  axis_ranges:
    x: [min_cfe_pct, max_cfe_pct]
    y_log: [min_marginal_usd, max_marginal_usd]   (for log-scale axis)
    y_linear: same but for linear scale (clip extreme outliers)

  meta:
    pathway: "1"
    note: "Per-ISO marginal cost per CFE percentage point. Pathway 1 = VRE + existing nuclear."
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "reliability_tax" / "charts"))
from data_loader import get_run, marginal_cost_per_cfe_pct, ISOS, ENDPOINTS  # noqa: E402

OUT_PATH = Path(__file__).parent / "section1_hockey_stick.json"


def _plateau_annotation(segments: list[dict]) -> dict | None:
    """Find the segment where the curve flattens/bends at the 90% region."""
    for seg in segments:
        if seg["from_cfe_pct"] >= 88 and seg["to_cfe_pct"] <= 93:
            return {
                "x_cfe_pct": seg["from_cfe_pct"],
                "y_marginal_usd": seg["marginal_usd_per_cfe_pct"],
                "note": "90% plateau — VRE-only buildout plateaus here",
            }
    return None


def _econ_crossover_annotation(
    segments: list[dict],
    cheapest_p3_annual_lcoe_usd_per_mwh: float,
) -> dict | None:
    """
    First segment where marginal cost per CFE% exceeds a rough crossover signal.
    We proxy the crossover as: segment marginal > $50B/CFE% (rough threshold
    above which clean firm becomes clearly cheaper on a system-cost basis).
    Exact Card C crossover is pathway-optimizer-internal; this is a visual annotation.
    """
    CROSSOVER_THRESHOLD_USD = 50e9  # $50B per CFE%
    for seg in segments:
        m = seg.get("marginal_usd_per_cfe_pct")
        if m is not None and m > CROSSOVER_THRESHOLD_USD:
            return {
                "x_cfe_pct": seg["from_cfe_pct"],
                "y_marginal_usd": m,
                "note": f"Economic crossover — marginal $/CFE% > $50B/pt",
            }
    return None


def main() -> None:
    per_iso: dict[str, list[dict]] = {}
    plateau_annotations: dict[str, dict | None] = {}
    crossover_annotations: dict[str, dict | None] = {}

    all_marginals: list[float] = []

    for iso in ISOS:
        segments = marginal_cost_per_cfe_pct(iso, "1")
        per_iso[iso] = segments

        # Collect finite marginals for axis range
        for seg in segments:
            m = seg.get("marginal_usd_per_cfe_pct")
            if m is not None and m > 0:
                all_marginals.append(m)

        # Annotation: 90% plateau
        plateau_annotations[iso] = _plateau_annotation(segments)

        # Annotation: economic crossover
        crossover_annotations[iso] = _econ_crossover_annotation(segments, 0.0)

    # Axis ranges
    pos_marginals = sorted(m for m in all_marginals if m > 0)
    if pos_marginals:
        # Linear range clips top 5% outliers
        clip_idx = max(0, int(len(pos_marginals) * 0.95))
        y_max_linear = pos_marginals[min(clip_idx, len(pos_marginals) - 1)]
        y_min_linear = pos_marginals[0]
        y_min_log = pos_marginals[0]
        y_max_log = pos_marginals[-1]
    else:
        y_min_linear = y_max_linear = y_min_log = y_max_log = 0.0

    # CFE range from data
    all_cfe = []
    for segs in per_iso.values():
        for seg in segs:
            all_cfe.extend([seg["from_cfe_pct"], seg["to_cfe_pct"]])
    x_range = [min(all_cfe, default=90.0), max(all_cfe, default=99.9)]

    payload = {
        "per_iso": per_iso,
        "annotations": {
            "plateau_90": plateau_annotations,
            "econ_crossover": crossover_annotations,
        },
        "axis_ranges": {
            "x": x_range,
            "y_linear": [y_min_linear, y_max_linear],
            "y_log": [y_min_log, y_max_log],
        },
        "meta": {
            "pathway": "1",
            "units": "USD per CFE percentage point",
            "note": (
                "Per-ISO marginal cost per CFE percentage point under Pathway 1 "
                "(VRE + batteries + existing nuclear, no new clean firm). "
                "Null segments = physically infeasible or zero delta-CFE."
            ),
        },
    }

    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT_PATH}")

    # Sanity print
    print("\n--- section1_hockey_stick quick summary ---")
    for iso in ISOS:
        segs = per_iso[iso]
        print(f"\n  {iso}:")
        for seg in segs:
            m = seg.get("marginal_usd_per_cfe_pct")
            m_str = f"${m/1e9:.1f}B/pt" if m is not None else "null"
            phys = "OK" if seg["to_feasible_physical"] else "INFEAS"
            print(f"    {seg['from_cfe_pct']:.1f}→{seg['to_cfe_pct']:.1f}%: {m_str} [{phys}]")
        if plateau_annotations[iso]:
            print(f"    PLATEAU annotation @ {plateau_annotations[iso]['x_cfe_pct']:.1f}%")
        if crossover_annotations[iso]:
            print(f"    CROSSOVER annotation @ {crossover_annotations[iso]['x_cfe_pct']:.1f}%")


if __name__ == "__main__":
    main()
