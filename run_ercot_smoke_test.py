#!/usr/bin/env python3
"""ERCOT smoke-test orchestrator — Prompt 2B.

Runs 4 pathways × 5 endpoints = 20 runs for ERCOT.
Order per endpoint: P3 first (reference), then P1, P2a, P2b.

Outputs:
  analysis/reliability-tax/data/ERCOT/pathway{P}_ep{E}.json  (per run)
  analysis/reliability-tax/data/MANIFEST.json                 (incremental)

Logs a one-line summary after each completed run.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

import step_2_3_pathway_optimizer as opt

ISO = "ERCOT"
# Pathway order per endpoint: 3 first (reference), then 1, 2a, 2b.
PATHWAY_ORDER = ("3", "1", "2a", "2b")
ENDPOINTS = (0.90, 0.95, 0.975, 0.99, 0.999)

OUTPUT_ROOT = Path(__file__).resolve().parent / "analysis" / "reliability-tax" / "data"


def fmt_ep(ep: float) -> str:
    return f"{ep * 100:g}%"


def run_one(pathway: str, endpoint: float) -> dict:
    config = opt.RunConfig(
        iso=ISO,
        pathway=pathway,
        endpoint=endpoint,
        output_root=OUTPUT_ROOT,
    )
    t0 = time.time()
    try:
        summary = opt.run_pathway(config)
        elapsed = time.time() - t0
        summary["elapsed_s"] = round(elapsed, 1)
        return summary
    except Exception as exc:
        elapsed = time.time() - t0
        return {
            "run_key": f"{ISO}__pathway{pathway}__ep{fmt_ep(endpoint)}",
            "status": "error",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_s": round(elapsed, 1),
        }


def main() -> None:
    total = len(PATHWAY_ORDER) * len(ENDPOINTS)
    completed = 0
    results = []

    print(f"ERCOT smoke test: {total} runs ({len(ENDPOINTS)} endpoints × {len(PATHWAY_ORDER)} pathways)")
    print(f"Output root: {OUTPUT_ROOT}")
    print("=" * 70)

    for endpoint in ENDPOINTS:
        for pathway in PATHWAY_ORDER:
            completed += 1
            label = f"[{completed:02d}/{total}] ERCOT P{pathway} @ {fmt_ep(endpoint)}"
            print(f"\n{label} — starting...")
            sys.stdout.flush()

            summary = run_one(pathway, endpoint)
            results.append(summary)

            status = summary.get("status", "?")
            cfe = summary.get("achieved_cfe_pct", "N/A")
            npv7 = summary.get("npv_at_7pct", summary.get("npv_at_7pct", "N/A"))
            elapsed = summary.get("elapsed_s", "?")
            err = summary.get("error", "")

            if status == "error":
                print(f"  FAILED ({elapsed}s): {err[:120]}")
            else:
                print(
                    f"  {status.upper()} | CFE={cfe:.3f}% | NPV@7%=${npv7:,.0f} | "
                    f"elapsed={elapsed}s"
                )
            sys.stdout.flush()

    print("\n" + "=" * 70)
    print("All runs complete. Summary JSON:")
    print(json.dumps(results, indent=2, default=str))

    # Save summary to a convenience file (not the MANIFEST — that's managed by the optimizer).
    summary_path = OUTPUT_ROOT / "ercot_smoke_test_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
