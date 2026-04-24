"""Rebuild data/step2.3-pathway/MANIFEST.json from per-run JSONs.

Walks every ``pathway*_ep*.json`` under the data tree, reconstructs the
manifest entry schema used by ``_manifest_entry`` in
``scripts/step_2_3_pathway_optimizer.py``, and writes MANIFEST.json in place.

Needed because the concurrent 7-ISO sweep races on MANIFEST.json.tmp
(``os.replace`` across processes); the per-run JSONs themselves are fine.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / 'data' / 'step2.3-pathway'


def _run_key(cfg: dict) -> str:
    ep_pct = float(cfg['endpoint_pct'])
    ep_tag = f"{ep_pct:g}".replace('.', 'p')
    return f"{cfg['iso']}__pathway{cfg['pathway']}__ep{ep_tag}"


def _manifest_entry_from_file(path: Path) -> dict:
    with open(path) as fh:
        d = json.load(fh)
    cfg = d['config']
    stranding_totals = {'gw_by_resource': {}, 'usd_by_resource': {}}
    for row in d.get('tables', {}).get('stranding_ledger', []) or []:
        res = row.get('resource')
        if res is None:
            continue
        stranding_totals['gw_by_resource'][res] = float(row.get('stranded_twh', 0.0))
        stranding_totals['usd_by_resource'][res] = float(row.get('stranded_book_value_usd', 0.0))
    headline = d.get('headline', {})
    strand_meta = d.get('stranding_metadata', {})
    return {
        'run_key': _run_key(cfg),
        'iso': cfg['iso'],
        'pathway': cfg['pathway'],
        'endpoint': cfg['endpoint'],
        'feasibility': d.get('feasibility', {}),
        'achieved_cfe_pct': headline.get('achieved_cfe_pct', 0.0),
        'undiscounted_cost_usd': headline.get('undiscounted_cost_usd', 0.0),
        'npv_at_5pct': headline.get('npv_at_5pct', 0.0),
        'npv_at_7pct': headline.get('npv_at_7pct', 0.0),
        'npv_at_9pct': headline.get('npv_at_9pct', 0.0),
        'retirement_timeline': d.get('retirement_timeline', []),
        'comparative_stranding': stranding_totals,
        'vre_curtailment_at_endpoint': d.get('vre_curtailment_at_endpoint', {}),
        'reliability_tax': d.get('reliability_tax', {}),
        'new_gas_summary': {
            'total_new_gas_built_mw': strand_meta.get('total_new_gas_built_mw', 0.0),
            'total_stranded_capex_usd': strand_meta.get('total_stranded_capex_usd', 0.0),
            'stranded_vintage_count': strand_meta.get('stranded_vintage_count', 0),
        },
        'output_path': str(path),
    }


def main() -> None:
    files = sorted(DATA_ROOT.glob('*/pathway*_ep*.json'))
    runs: dict[str, dict] = {}
    for p in files:
        entry = _manifest_entry_from_file(p)
        runs[entry['run_key']] = entry
    manifest = {
        'schema_version': 2,
        'description': (
            'Reliability Tax pathway optimizer manifest (v2: Card U new-build '
            'gas + Card F\' CF-stranding + Card S 10-endpoint sweep — '
            '7 ISOs \u00d7 5 pathways \u00d7 10 endpoints = 350 runs).'
        ),
        'runs': runs,
    }
    out = DATA_ROOT / 'MANIFEST.json'
    tmp = out.with_suffix('.json.tmp')
    with open(tmp, 'w') as fh:
        json.dump(manifest, fh, indent=2, default=str)
    tmp.replace(out)
    print(f"Rebuilt MANIFEST.json with {len(runs)} runs from {len(files)} files.")


if __name__ == '__main__':
    main()
