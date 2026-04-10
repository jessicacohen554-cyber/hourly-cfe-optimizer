"""
FastAPI backend for profile-modeling desktop tool.

Endpoints:
  GET  /                   → serves frontend/index.html
  GET  /api/config         → available ISOs, resources, profiles, presets
  POST /api/optimize       → run optimization for up to 3 portfolios
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import List, Optional

# Path setup
BACKEND_DIR = Path(__file__).resolve().parent
PKG_DIR = BACKEND_DIR.parent
FRONTEND_DIR = PKG_DIR / "frontend"
SCRIPTS_DIR = PKG_DIR.parent / "scripts"

sys.path.insert(0, str(PKG_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import (
    ISOS, THRESHOLDS, COST_LEVELS, WHOLESALE_PRICES,
    GENERATION_RESOURCES, STORAGE_RESOURCES, H,
)
from load_profiles import LOAD_PROFILES
from supply import load_supply_shapes
from dispatch import warmup as warmup_numba
from optimizer import optimize_single, resolve_resources, PRESETS
from backend.repricing import reprice_mix

# ── App setup ──────────────────────────────────────────────────────────────

app = FastAPI(title="Clean Energy Profile Modeler")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

# Serve frontend static files
app.mount("/styles", StaticFiles(directory=str(FRONTEND_DIR / "styles")), name="styles")
if (FRONTEND_DIR / "js").exists():
    app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")

# ── Cache ──────────────────────────────────────────────────────────────────

_supply_cache = {}
_numba_warmed = False


def _ensure_numba():
    global _numba_warmed
    if not _numba_warmed:
        warmup_numba()
        _numba_warmed = True


def _get_supply(iso):
    if iso not in _supply_cache:
        _supply_cache[iso] = load_supply_shapes(iso)
    return _supply_cache[iso]


# ── Models ─────────────────────────────────────────────────────────────────

class PortfolioRequest(BaseModel):
    name: str
    resources: List[str]

class OptimizeRequest(BaseModel):
    iso: str = 'ERCOT'
    profile: str = 'flat'
    portfolios: List[PortfolioRequest]
    targets: Optional[List[float]] = None
    maxiter: int = 200
    popsize: int = 25


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/")
def serve_frontend():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/api/config")
def get_config():
    """Return available configuration options."""
    all_resources = list(GENERATION_RESOURCES) + list(STORAGE_RESOURCES)
    resource_labels = {
        'wind': 'Wind (Onshore)', 'solar': 'Solar PV',
        'offshore_wind': 'Offshore Wind', 'clean_firm': 'Nuclear (New-Build)',
        'ccs_ccgt': 'CCS-CCGT', 'hydro': 'Hydro (Existing)',
        'geothermal': 'Geothermal', 'battery4': 'Battery (4hr Li-ion)',
        'battery8': 'Battery (8hr Li-ion)', 'ldes': 'LDES (100hr Iron-Air)',
        'h2': 'Green H₂ (1000hr)',
    }
    resource_categories = {
        'generation': list(GENERATION_RESOURCES),
        'storage': list(STORAGE_RESOURCES),
    }

    return {
        'isos': ISOS,
        'profiles': list(LOAD_PROFILES.keys()) + ['eia_demand'],
        'resources': all_resources,
        'resource_labels': resource_labels,
        'resource_categories': resource_categories,
        'presets': {k: v for k, v in PRESETS.items()},
        'thresholds': THRESHOLDS,
        'wholesale_prices': WHOLESALE_PRICES,
    }


@app.post("/api/optimize")
def run_optimization(req: OptimizeRequest):
    """Run cost optimization for up to 3 portfolios across all thresholds.

    For each (portfolio, threshold):
      1. Optimize resource mix at Medium costs (DE)
      2. Reprice the fixed mix under all toggle combos → P10/P50/P90

    Returns results for charting.
    """
    _ensure_numba()

    iso = req.iso.upper()
    targets = req.targets or THRESHOLDS
    supply_shapes = _get_supply(iso)

    from load_profiles import get_load_profile
    demand = get_load_profile(req.profile, iso=iso)

    wholesale = WHOLESALE_PRICES.get(iso, 30)

    results = {}
    total_start = time.time()

    for portfolio in req.portfolios[:3]:  # max 3
        pname = portfolio.name
        resources = portfolio.resources[:6]  # max 6

        portfolio_results = []

        for i, target in enumerate(sorted(targets)):
            t0 = time.time()

            # Optimize at Medium costs
            res = optimize_single(
                iso, 'Medium', target, resources, supply_shapes, demand,
                maxiter=req.maxiter, popsize=req.popsize, seed=42 + i,
            )

            if not res or res['score'] < target - 0.005:
                # Infeasible — try to report best achievable
                portfolio_results.append({
                    'target': target,
                    'feasible': False,
                    'score': res['score'] if res else 0,
                    'medium_cost': res['total_cost'] if res else 0,
                })
                continue

            # Extract allocations for repricing
            gen_res = res['gen_resources']
            stor_res = res['stor_resources']
            gen_allocs = [res['resources'][r]['alloc_pct'] / 100.0
                          for r in gen_res]
            stor_caps = [res['resources'][s]['cap_pct'] / 100.0
                         for s in stor_res]

            # Reprice under all toggle combos
            envelope = reprice_mix(gen_res, gen_allocs, stor_res, stor_caps, iso)

            # Build resource breakdown
            breakdown = {}
            for r in gen_res:
                info = res['resources'][r]
                breakdown[r] = {
                    'alloc_pct': round(info['alloc_pct'], 2),
                    'cost_medium': round(info['cost_per_mwh'], 2),
                }
            for s in stor_res:
                info = res['resources'][s]
                if info['cap_pct'] > 0.001:
                    breakdown[s] = {
                        'cap_pct': round(info['cap_pct'], 4),
                        'cap_mwh': round(info.get('cap_mwh', 0), 2),
                        'cost_medium': round(info['cost_per_mwh'], 2),
                    }

            dt = time.time() - t0

            portfolio_results.append({
                'target': target,
                'feasible': True,
                'score': round(res['score'], 4),
                'medium_cost': round(res['total_cost'], 2),
                'total_gen_pct': round(res['total_gen_pct'], 1),
                'envelope': envelope,
                'breakdown': breakdown,
                'time_s': round(dt, 1),
            })

        results[pname] = {
            'resources': resources,
            'data': portfolio_results,
        }

    total_time = time.time() - total_start

    return {
        'iso': iso,
        'profile': req.profile,
        'wholesale': wholesale,
        'total_time_s': round(total_time, 1),
        'portfolios': results,
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8050)
