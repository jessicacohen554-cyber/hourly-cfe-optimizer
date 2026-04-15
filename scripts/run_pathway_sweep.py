#!/usr/bin/env python3
"""run_pathway_sweep.py — Sequenced orchestrator for reliability-tax pathway runs.

Called by .github/workflows/reliability_tax_full_run.yml. Wraps
step_2_3_pathway_optimizer.py with:

  - Pathway 3 first sequencing per (ISO, endpoint) — comparative stranding
    (Card K) requires Pathway 3's buildout as a reference for 1/2a/2b.
  - MANIFEST-based graceful resumption: already-completed runs are skipped
    unless --force is passed.
  - Incremental MANIFEST.json updates after each run (the optimizer writes
    these; this script re-reads the MANIFEST to check completion status).
  - Progress logging to stdout in a format useful for GitHub Actions logs.
  - --dry-run mode that lists the planned run sequence without executing.

Usage
-----
Single ISO, all defaults (called by workflow per-ISO matrix job):

    python scripts/run_pathway_sweep.py --iso CAISO

Partial rerun — only two endpoints, only Pathway 2b:

    python scripts/run_pathway_sweep.py \\
        --iso PJM \\
        --endpoints 0.99,0.999 \\
        --pathways 2b

Demand sensitivity pass:

    python scripts/run_pathway_sweep.py \\
        --iso MISO \\
        --growth Low \\
        --output-root analysis/reliability-tax/data-demand-sensitivity

Dry run to preview what would execute:

    python scripts/run_pathway_sweep.py \\
        --iso NYISO \\
        --dry-run

Force re-run of everything (ignores MANIFEST):

    python scripts/run_pathway_sweep.py \\
        --iso ERCOT \\
        --force

Notes
-----
- Demand-sensitivity runs use a separate --output-root so they do not collide
  with medium-growth entries in the canonical MANIFEST. The optimizer's run_key
  does not include the demand growth level.
- The optimizer writes MANIFEST.json atomically after each run. This script
  re-reads it before each run to decide whether to skip.
- Pathway 3 is always moved to the front of the run list regardless of the
  order supplied via --pathways. If Pathway 3 is not in the requested pathways
  list, it is NOT prepended — only the explicitly requested pathways run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Constants — must match step_2_3_pathway_optimizer.py
# ---------------------------------------------------------------------------

ALL_ISOS = ('CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP')
ALL_PATHWAYS = ('1', '2a', '2b', '3')
ALL_ENDPOINTS = (0.90, 0.95, 0.975, 0.99, 0.999)
DEMAND_GROWTH_LEVELS = ('Low', 'Medium', 'High')

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OPTIMIZER_SCRIPT = _REPO_ROOT / 'scripts' / 'step_2_3_pathway_optimizer.py'
_DEFAULT_OUTPUT_ROOT = _REPO_ROOT / 'analysis' / 'reliability-tax' / 'data'
_MANIFEST_FILENAME = 'MANIFEST.json'


# ---------------------------------------------------------------------------
# Run plan
# ---------------------------------------------------------------------------


class PlannedRun(NamedTuple):
    iso: str
    pathway: str
    endpoint: float
    growth: str
    output_root: Path
    seed_run_path: Path | None = None  # Prior-endpoint JSON for floor-ratchet seed

    @property
    def run_key(self) -> str:
        """Matches the key format used by step_2_3_pathway_optimizer._run_key."""
        ep_pct = self.endpoint * 100.0
        ep_tag = f"{ep_pct:g}".replace('.', 'p')
        return f"{self.iso}__pathway{self.pathway}__ep{ep_tag}"

    @property
    def output_path(self) -> Path:
        """Expected JSON output path for this run (mirrors optimizer RunConfig.output_path)."""
        ep_pct = self.endpoint * 100.0
        ep_tag = f"{ep_pct:g}".replace('.', 'p')
        return self.output_root / self.iso / f"pathway{self.pathway}_ep{ep_tag}.json"

    @property
    def label(self) -> str:
        seed_str = f" seed={self.seed_run_path.name}" if self.seed_run_path else ""
        return f"{self.iso} pathway={self.pathway} endpoint={self.endpoint:.3g} growth={self.growth}{seed_str}"


def _parse_endpoints(raw: str) -> list[float]:
    """Parse comma-separated endpoint string into floats; validate against allowed set."""
    parts = [p.strip() for p in raw.split(',') if p.strip()]
    result = []
    for p in parts:
        val = float(p)
        if val > 1.0:
            val = val / 100.0
        # Find nearest approved endpoint.
        closest = min(ALL_ENDPOINTS, key=lambda e: abs(e - val))
        if abs(closest - val) > 0.001:
            raise ValueError(
                f"Endpoint {p!r} does not match any approved endpoint {ALL_ENDPOINTS}. "
                f"Nearest: {closest}"
            )
        result.append(closest)
    if not result:
        raise ValueError("--endpoints produced an empty list")
    return list(dict.fromkeys(result))  # deduplicate, preserve order


def _parse_pathways(raw: str) -> list[str]:
    """Parse comma-separated pathway string; validate; ensure Pathway 3 is first if present."""
    parts = [p.strip() for p in raw.split(',') if p.strip()]
    for p in parts:
        if p not in ALL_PATHWAYS:
            raise ValueError(f"Unknown pathway {p!r}; must be one of {ALL_PATHWAYS}")
    # Deduplicate, preserve order, then move '3' to front if present.
    seen: dict[str, None] = {}
    for p in parts:
        seen[p] = None
    ordered = list(seen.keys())
    if '3' in ordered:
        ordered.remove('3')
        ordered.insert(0, '3')
    return ordered


def build_run_plan(
    iso: str,
    endpoints: list[float],
    pathways: list[str],
    growth: str,
    output_root: Path,
) -> list[PlannedRun]:
    """Build the ordered run list for one ISO.

    Loop order: pathway (outer) → endpoint (inner, ascending). This enables
    the cross-endpoint floor ratchet: each endpoint run can be seeded from
    the previous endpoint run of the same pathway. Pathway 3 is always moved
    to the front by _parse_pathways(), so all P3 endpoints complete before P1
    begins. When P1@ep runs, the P3@ep JSON is already on disk and the
    optimizer loads it instead of re-solving (preserving the seeded P3).

    Seed paths are NOT set here — sweep() injects them dynamically as each
    run completes so it can track the actual output path written to disk.
    """
    runs: list[PlannedRun] = []
    for pathway in pathways:
        for endpoint in endpoints:
            runs.append(PlannedRun(
                iso=iso,
                pathway=pathway,
                endpoint=endpoint,
                growth=growth,
                output_root=output_root,
            ))
    return runs


# ---------------------------------------------------------------------------
# MANIFEST helpers
# ---------------------------------------------------------------------------


def load_manifest(output_root: Path) -> dict:
    manifest_path = output_root / _MANIFEST_FILENAME
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[sweep] WARNING: Could not read MANIFEST ({exc}); treating as empty.")
    return {'runs': {}}


def is_completed(manifest: dict, run: PlannedRun) -> bool:
    return run.run_key in manifest.get('runs', {})


# ---------------------------------------------------------------------------
# Run executor
# ---------------------------------------------------------------------------


def run_one(run: PlannedRun, dry_run: bool = False) -> bool:
    """Execute a single optimizer run. Returns True on success."""
    cmd = [
        sys.executable,
        str(_OPTIMIZER_SCRIPT),
        '--iso', run.iso,
        '--pathway', run.pathway,
        '--endpoint', str(run.endpoint),
        '--growth', run.growth,
        '--output-root', str(run.output_root),
    ]
    # Floor-ratchet: pass prior endpoint's JSON as seed if available.
    if run.seed_run_path is not None and run.seed_run_path.exists():
        cmd.extend(['--seed-run', str(run.seed_run_path)])
    if dry_run:
        print(f"  [dry-run] Would execute: {' '.join(cmd)}")
        return True

    print(f"  Executing: {' '.join(cmd)}", flush=True)
    t0 = time.monotonic()
    result = subprocess.run(cmd, check=False)
    elapsed = time.monotonic() - t0
    if result.returncode == 0:
        print(f"  Completed in {elapsed:.1f}s (exit 0)", flush=True)
        return True
    else:
        print(
            f"  FAILED after {elapsed:.1f}s (exit {result.returncode})",
            flush=True,
        )
        return False


# ---------------------------------------------------------------------------
# Main sweep logic
# ---------------------------------------------------------------------------


def sweep(
    iso: str,
    endpoints: list[float],
    pathways: list[str],
    growth: str,
    output_root: Path,
    dry_run: bool,
    force: bool,
) -> None:
    """Run all (pathway, endpoint) combos for the given ISO, skipping completed runs.

    Run order: pathway (outer) → endpoint (inner, ascending). The terminal
    ledger from each completed run is passed as --seed-run to the next
    endpoint in the same pathway (cross-endpoint floor ratchet).
    """
    output_root.mkdir(parents=True, exist_ok=True)

    plan = build_run_plan(iso, endpoints, pathways, growth, output_root)

    total = len(plan)
    print(f"\n{'='*60}")
    print(f"ISO: {iso}  growth: {growth}  output-root: {output_root}")
    print(f"Planned runs: {total}  (pathways={pathways}  endpoints={endpoints})")
    print(f"dry-run: {dry_run}  force: {force}")
    print(f"{'='*60}\n")

    skipped = 0
    succeeded = 0
    failed = 0

    # Track the most recent successful output path per pathway for seed chaining.
    last_output: dict[str, Path | None] = {pw: None for pw in pathways}

    for i, run in enumerate(plan, start=1):
        # Inject seed from previous endpoint of the same pathway.
        seed_path = last_output.get(run.pathway)
        seeded_run = run._replace(seed_run_path=seed_path)

        print(f"\n[{i}/{total}] {seeded_run.label}")

        if not force and not dry_run:
            # Reload MANIFEST before each run to catch in-process writes.
            manifest = load_manifest(output_root)
            if is_completed(manifest, run):
                print(f"  SKIP — already in MANIFEST (key: {run.run_key}). Pass --force to re-run.")
                skipped += 1
                # Still update seed tracker so subsequent endpoint runs can chain.
                if seeded_run.output_path.exists():
                    last_output[run.pathway] = seeded_run.output_path
                continue

        ok = run_one(seeded_run, dry_run=dry_run)
        if dry_run:
            continue
        if ok:
            succeeded += 1
            # Record output path so next endpoint of same pathway can seed from it.
            last_output[run.pathway] = seeded_run.output_path
        else:
            failed += 1
            # Don't update seed tracker on failure — next endpoint starts fresh.
            # Continue rather than abort — partial results are still valuable.
            print(f"  Continuing to next run despite failure.")

    # Final summary
    print(f"\n{'='*60}")
    if dry_run:
        print(f"DRY-RUN COMPLETE — {total} runs would execute for {iso} ({growth})")
    else:
        print(f"SWEEP COMPLETE — {iso} ({growth})")
        print(f"  Succeeded: {succeeded} | Skipped (already done): {skipped} | Failed: {failed}")
        if failed > 0:
            print(f"  WARNING: {failed} run(s) failed. Check logs above.")
            # Exit non-zero so the GitHub Actions step is marked as failed.
            sys.exit(1)
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog='run_pathway_sweep',
        description=(
            'Orchestrate step_2_3_pathway_optimizer.py runs for one ISO with '
            'Pathway-3-first sequencing and MANIFEST-based resumption.'
        ),
    )
    ap.add_argument(
        '--iso',
        required=True,
        choices=ALL_ISOS,
        help='Target ISO (one per invocation; workflow fans out per-ISO).',
    )
    ap.add_argument(
        '--endpoints',
        default=','.join(str(e) for e in ALL_ENDPOINTS),
        help=(
            'Comma-separated CFE endpoints to run. '
            'Default: all five (0.90,0.95,0.975,0.99,0.999).'
        ),
    )
    ap.add_argument(
        '--pathways',
        default=','.join(ALL_PATHWAYS),
        help=(
            'Comma-separated pathway IDs. Pathway 3 is always moved to the '
            'front regardless of supplied order. Default: 3,1,2a,2b.'
        ),
    )
    ap.add_argument(
        '--growth',
        default='Medium',
        choices=DEMAND_GROWTH_LEVELS,
        help='Demand growth level (default: Medium).',
    )
    ap.add_argument(
        '--output-root',
        default=str(_DEFAULT_OUTPUT_ROOT),
        help=(
            'Root directory for per-run JSON outputs and MANIFEST.json. '
            'Use a different path for demand-sensitivity runs to avoid '
            'MANIFEST key collisions with medium-growth runs.'
        ),
    )
    ap.add_argument(
        '--dry-run',
        action='store_true',
        help='Print the planned run sequence without executing anything.',
    )
    ap.add_argument(
        '--force',
        action='store_true',
        help='Skip MANIFEST check and re-run everything.',
    )
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)

    if not _OPTIMIZER_SCRIPT.exists():
        print(
            f"ERROR: Optimizer script not found at {_OPTIMIZER_SCRIPT}.\n"
            "Run from the repo root or ensure scripts/ is on the path.",
            file=sys.stderr,
        )
        sys.exit(1)

    endpoints = _parse_endpoints(args.endpoints)
    pathways = _parse_pathways(args.pathways)
    output_root = Path(args.output_root)

    sweep(
        iso=args.iso,
        endpoints=endpoints,
        pathways=pathways,
        growth=args.growth,
        output_root=output_root,
        dry_run=args.dry_run,
        force=args.force,
    )


if __name__ == '__main__':
    main()
