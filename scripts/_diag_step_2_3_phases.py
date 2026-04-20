#!/usr/bin/env python3
"""
Per-phase startup probe for step 2.3 with 60s timeouts and 5s heartbeats.

Usage:
    python3 scripts/_diag_step_2_3_phases.py --phase <1-5>

Exits 0 on success, 1 on exception, 2 on timeout.
"""
import argparse
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

# Insert scripts/ dir so bare imports (pipeline_config, step_2_3_*, etc.) resolve.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

TIMEOUT_S = 60
HEARTBEAT_S = 5


def _heartbeat(phase: int, stop_event: threading.Event) -> None:
    start = time.monotonic()
    while not stop_event.wait(HEARTBEAT_S):
        elapsed = time.monotonic() - start
        print(f"[heartbeat] phase {phase} elapsed {elapsed:.1f}s", flush=True)


def _run_with_timeout(phase: int, fn) -> None:
    stop_event = threading.Event()
    hb = threading.Thread(target=_heartbeat, args=(phase, stop_event), daemon=True)

    def _timeout_handler(signum, frame):
        stop_event.set()
        print(f"phase {phase} TIMEOUT after {TIMEOUT_S}s", flush=True)
        sys.exit(2)

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(TIMEOUT_S)
    hb.start()
    try:
        fn()
    except Exception:
        stop_event.set()
        signal.alarm(0)
        traceback.print_exc()
        sys.exit(1)
    signal.alarm(0)
    stop_event.set()


# ---------------------------------------------------------------------------
# Phase functions
# ---------------------------------------------------------------------------

def phase1():
    # Adapted: no scripts/__init__.py, so import without 'scripts.' prefix.
    import step_2_3_pathway_optimizer  # noqa: F401
    import step2_2a_cost_optimization  # noqa: F401
    ts = time.strftime("%H:%M:%S")
    print(f"{ts} phase 1 OK", flush=True)


def phase2():
    from step2_2a_cost_optimization import _ensure_common_data
    _ensure_common_data("ERCOT")
    print("phase 2 OK", flush=True)


def phase3():
    from step_2_3_pathway_optimizer import load_ef_mixes
    # Adapted: threshold is percent-form (80.0 == 80% CFE), not fractional 0.80.
    ef = load_ef_mixes("ERCOT", 80.0)
    print(f"phase 3 OK ({len(ef)} rows)", flush=True)


def phase4():
    from step_2_3_pathway_optimizer import select_target_mix, RunConfig, PivotState
    # output_root must be a Path; endpoint=0.80 (fractional, validated against ENDPOINTS).
    cfg = RunConfig(iso="ERCOT", pathway="1", endpoint=0.80,
                    output_root=Path("/tmp/diag"))
    # Adapted: endpoint_pct arg is percent-form (80.0), not fractional.
    target = select_target_mix("ERCOT", "1", 2030, 80.0, cfg, PivotState())
    print(f"phase 4 OK (cfe={target.get('cfe_pct', '?')})", flush=True)


def phase5():
    from step_2_3_pathway_optimizer import run_pathway, RunConfig
    cfg = RunConfig(iso="ERCOT", pathway="1", endpoint=0.80,
                    output_root=Path("/tmp/diag"))
    # Adapted: return key is 'undiscounted_cost_usd', not 'total_cost_usd'.
    result = run_pathway(cfg, skip_manifest=True)
    print(f"phase 5 OK (cost={result.get('undiscounted_cost_usd', '?')})", flush=True)


# ---------------------------------------------------------------------------

PHASES = {1: phase1, 2: phase2, 3: phase3, 4: phase4, 5: phase5}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 2.3 per-phase startup diagnostics with timeout + heartbeat"
    )
    parser.add_argument(
        "--phase", type=int, required=True, choices=range(1, 6),
        metavar="N", help="Phase to run (1-5)",
    )
    args = parser.parse_args()
    _run_with_timeout(args.phase, PHASES[args.phase])


if __name__ == "__main__":
    main()
