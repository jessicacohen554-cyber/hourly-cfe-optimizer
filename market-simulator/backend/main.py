"""
FastAPI backend for the Market Simulator.

Exposes REST endpoints for single simulations, parametric sweeps,
sensitivity analysis, and ISO metadata.  Serves the frontend static
files from ../frontend/.
"""

from __future__ import annotations

import asyncio
import csv
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

# ─────────────────────────────────────────────────────────────────────────────
# Path setup — import the simulation engine from ../scripts/
# Supports env-var overrides for PyInstaller frozen builds (desktop_app.py).
# ─────────────────────────────────────────────────────────────────────────────

_bundle_dir = os.environ.get('MARKET_SIM_BUNDLE_DIR')
_data_dir = os.environ.get('MARKET_SIM_DATA_DIR')

if _bundle_dir:
    # Frozen / desktop mode — paths set by desktop_app.py
    MARKET_SIM_ROOT = Path(_bundle_dir)
    SCRIPTS_DIR = MARKET_SIM_ROOT / "scripts"
    FRONTEND_DIR = MARKET_SIM_ROOT / "frontend"
else:
    # Dev mode — resolve relative to this file
    BACKEND_DIR = Path(__file__).resolve().parent
    MARKET_SIM_ROOT = BACKEND_DIR.parent
    SCRIPTS_DIR = MARKET_SIM_ROOT / "scripts"
    FRONTEND_DIR = MARKET_SIM_ROOT / "frontend"

sys.path.insert(0, str(SCRIPTS_DIR))

from market_simulation import (
    run_market_simulation,
    run_full_sweep,
    run_correlated_scenarios,
    build_single_scenario,
    build_market_scenarios,
    save_results,
    load_common_data,
    load_egrid_baselines,
    check_data_sources,
    EGRID_2023_CLEAN_PCT,
    EGRID_2023_LMP,
    GAS_FRICTION_LEVELS,
    QUEUE_CAP_GW,
    QUEUE_LEARNING_MAP,
    DEMAND_GROWTH_LEVELS,
    PRICE_SENSITIVITIES,
    PPA_LEVELS,
)
from lmp_engine import (
    HEAT_RATES,
    VOM,
    CO2_RATES,
    FUEL_PRICES,
    _get_fuel_prices,
    INSTALLED_FOSSIL_MW,
    FOSSIL_CAPACITY_SHARES,
    NOX_RATES,
    SOX_RATES,
    NOX_PRICES,
    SOX_PRICES,
)
from pipeline_config import (
    ISOS,
    REGIONAL_DEMAND_TWH,
    GRID_MIX_SHARES,
    CAPACITY_MARKET_PRICES,
    WHOLESALE_PRICES,
    CONFIDENCE_ZONES,
    CORRELATED_SCENARIOS,
    get_confidence_zone,
    adjust_confidence_for_triggers,
)

from .models import (
    SimulationRequest,
    SimulationResponse,
    CustomOverrides,
    SweepRequest,
    SweepJob,
    SweepStatus,
    SensitivityRequest,
    SensitivityResponse,
    SensitivityResult,
    GeneratorEconomics,
    NuclearRevenue,
    ZoneDetail,
    YearResult,
    IPMTrigger,
    FuelBinRow,
    HourlyProfile,
    SupplyStackEntry,
    ISOSummary,
    ISODefaults,
    CorrelatedScenarioRequest,
    CorrelatedScenarioResult,
    CorrelatedScenarioResponse,
    TornadoBar,
    TornadoMetric,
)


# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Hourly CFE Market Simulator",
    description="Profit-driven market simulation for US ISO electricity markets.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Static files & frontend serving
# ─────────────────────────────────────────────────────────────────────────────

if FRONTEND_DIR.exists():
    # Mount CSS/JS sub-directories
    styles_dir = FRONTEND_DIR / "styles"
    js_dir = FRONTEND_DIR / "js"
    if styles_dir.exists():
        app.mount("/styles", StaticFiles(directory=str(styles_dir)), name="styles")
    if js_dir.exists():
        app.mount("/js", StaticFiles(directory=str(js_dir)), name="js")
    # Mount the entire frontend as a fallback for any other static assets
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# Mount brand assets
brand_assets_dir = FRONTEND_DIR / "brand-assets"
if brand_assets_dir.exists():
    app.mount("/brand-assets", StaticFiles(directory=str(brand_assets_dir)), name="brand-assets")

# Ensure results directory exists
# In frozen mode, results go to app_data/results/ (writable)
if _data_dir:
    RESULTS_DIR = Path(_data_dir) / "results"
    CUSTOM_INPUTS_DIR = Path(_data_dir) / "custom-user-inputs"
else:
    RESULTS_DIR = MARKET_SIM_ROOT / "results"
    CUSTOM_INPUTS_DIR = MARKET_SIM_ROOT / "custom-user-inputs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ISOS = ["CAISO", "ERCOT", "PJM", "NYISO", "NEISO", "MISO", "SPP"]


def load_custom_timeseries_csv(filepath: Path) -> dict:
    """Load a custom input CSV (fuel, capacity, or REC) into a year-indexed lookup.

    Supports two formats:
      - Legacy (12 rows): month × 7 ISOs → returns {None: {month: {iso: val}}}
        (None key = static, applies to all years)
      - Time-series (year × month × zone × 7 ISOs) →
        returns {year: {month: {zone: {iso: val}}}}

    When zone column is absent or all blank, zone key is 'system'.
    """
    import pandas as pd

    try:
        df = pd.read_csv(filepath, sep=',')
        if len(df.columns) <= 2:
            df = pd.read_csv(filepath, sep='\t')
    except Exception:
        df = pd.read_csv(filepath, sep='\t')

    iso_cols = [c for c in df.columns if c in set(ISOS)]
    has_year = 'year' in df.columns
    has_zone = 'zone' in df.columns

    result = {}

    if has_year:
        df['year'] = df['year'].astype(int)
        if has_zone:
            df['_zone'] = df['zone'].fillna('').replace('', 'system')
        else:
            df['_zone'] = 'system'

        for _, row in df.iterrows():
            yr = int(row['year'])
            mo = int(row['month'])
            zn = row['_zone']
            if yr not in result:
                result[yr] = {}
            if mo not in result[yr]:
                result[yr][mo] = {}
            result[yr][mo][zn] = {iso: float(row[iso]) for iso in iso_cols}
    else:
        # Legacy: no year column → key is None (applies to all years)
        for _, row in df.iterrows():
            mo = int(row['month'])
            result[mo] = {'system': {iso: float(row[iso]) for iso in iso_cols}}
        result = {None: result}  # None = static/all years

    return result


def resolve_fuel_prices_for_year(timeseries_lookup: dict, year: int,
                                 iso: str, month: int = None,
                                 zone: str = 'system') -> float:
    """Look up a price from a timeseries lookup dict for a specific year/ISO.

    Falls back to nearest available year if exact year not found.
    If month is None, returns average across months.
    """
    # Static (legacy) format — None key
    if None in timeseries_lookup:
        months_data = timeseries_lookup[None]
        if month and month in months_data:
            return months_data[month].get(zone, months_data[month].get('system', {})).get(iso, 0)
        # Average across months
        vals = []
        for mo_data in months_data.values():
            zd = mo_data.get(zone, mo_data.get('system', {}))
            if iso in zd:
                vals.append(zd[iso])
        return sum(vals) / len(vals) if vals else 0

    # Time-series — find exact or nearest year
    available_years = sorted(timeseries_lookup.keys())
    if year in timeseries_lookup:
        target_year = year
    elif available_years:
        # Nearest year
        target_year = min(available_years, key=lambda y: abs(y - year))
    else:
        return 0

    months_data = timeseries_lookup[target_year]
    if month and month in months_data:
        zd = months_data[month].get(zone, months_data[month].get('system', {}))
        return zd.get(iso, 0)
    # Average across months
    vals = []
    for mo_data in months_data.values():
        zd = mo_data.get(zone, mo_data.get('system', {}))
        if iso in zd:
            vals.append(zd[iso])
    return sum(vals) / len(vals) if vals else 0


def load_custom_fuel_overrides() -> dict | None:
    """Load all 3 fuel CSVs into a structured timeseries dict.

    Returns dict with keys 'gas', 'coal', 'oil' each containing a
    timeseries lookup, or None if no files found.
    """
    fuel_files = {
        'gas': CUSTOM_INPUTS_DIR / "fuel_prices_gas.csv",
        'coal': CUSTOM_INPUTS_DIR / "fuel_prices_coal.csv",
        'oil': CUSTOM_INPUTS_DIR / "fuel_prices_oil.csv",
    }
    result = {}
    for fuel_type, fpath in fuel_files.items():
        if fpath.exists():
            result[fuel_type] = load_custom_timeseries_csv(fpath)
    return result if result else None


def load_custom_capacity_overrides() -> dict | None:
    """Load capacity prices CSV into a timeseries lookup."""
    fpath = CUSTOM_INPUTS_DIR / "capacity_prices.csv"
    if fpath.exists():
        return load_custom_timeseries_csv(fpath)
    return None


def load_custom_rec_overrides() -> dict | None:
    """Load REC prices CSV into a timeseries lookup."""
    fpath = CUSTOM_INPUTS_DIR / "rec_prices.csv"
    if fpath.exists():
        return load_custom_timeseries_csv(fpath)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# In-memory stores
# ─────────────────────────────────────────────────────────────────────────────

# Sweep job tracking (job_id → SweepJob)
_sweep_jobs: Dict[str, SweepJob] = {}

# Lazy-loaded shared data cache (populated on first simulation call)
_preloaded_data: Dict | None = None


def _get_preloaded_data() -> Dict:
    """Load common data once and cache for subsequent calls."""
    global _preloaded_data
    if _preloaded_data is None:
        demand_data, gen_profiles, emission_rates, fossil_mix = load_common_data()
        egrid_baselines = load_egrid_baselines()
        # Load interchange profiles (empty dict if unavailable → copper-plate fallback)
        try:
            from eia_data_io import load_interchange_profiles
            interchange_data = load_interchange_profiles()
        except Exception:
            interchange_data = {}
        _preloaded_data = {
            "demand_data": demand_data,
            "gen_profiles": gen_profiles,
            "emission_rates": emission_rates,
            "fossil_mix": fossil_mix,
            "egrid_baselines": egrid_baselines,
            "interchange_data": interchange_data,
            "data_sources": check_data_sources(),
        }
    return _preloaded_data


# ─────────────────────────────────────────────────────────────────────────────
# HTML page routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_guide_page():
    """Serve the guide / landing page."""
    guide_path = FRONTEND_DIR / "guide.html"
    if not guide_path.exists():
        # Fallback to setup if guide doesn't exist yet
        return FileResponse(str(FRONTEND_DIR / "setup.html"), media_type="text/html")
    return FileResponse(str(guide_path), media_type="text/html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Serve favicon if it exists, otherwise return 204 No Content."""
    favicon_path = FRONTEND_DIR / "brand-assets" / "favicon.ico"
    if favicon_path.exists():
        return FileResponse(str(favicon_path))
    return Response(status_code=204)


@app.get("/setup", response_class=HTMLResponse)
async def serve_setup_page():
    """Serve the setup / input form page."""
    setup_path = FRONTEND_DIR / "setup.html"
    if not setup_path.exists():
        raise HTTPException(status_code=404, detail="setup.html not found")
    return FileResponse(str(setup_path), media_type="text/html")


@app.get("/setup-template.csv")
async def serve_setup_template():
    """Serve the CSV configuration template for bulk setup."""
    csv_path = FRONTEND_DIR / "setup-template.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="setup-template.csv not found")
    return FileResponse(str(csv_path), media_type="text/csv",
                        filename="setup-template.csv")


@app.get("/results", response_class=HTMLResponse)
async def serve_results_page():
    """Serve the results page."""
    results_path = FRONTEND_DIR / "results.html"
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="results.html not found")
    return FileResponse(str(results_path), media_type="text/html")


@app.get("/fleet-config", response_class=HTMLResponse)
async def serve_fleet_config_page():
    """Serve the fleet configuration page."""
    fleet_path = FRONTEND_DIR / "fleet-config.html"
    if not fleet_path.exists():
        raise HTTPException(status_code=404, detail="fleet-config.html not found")
    return FileResponse(str(fleet_path), media_type="text/html")


@app.get("/fleet-scenarios", response_class=HTMLResponse)
async def serve_fleet_scenarios_page():
    """Serve the Fleet Climate Scenarios page."""
    page_path = FRONTEND_DIR / "fleet-scenarios.html"
    if not page_path.exists():
        raise HTTPException(status_code=404, detail="fleet-scenarios.html not found")
    return FileResponse(str(page_path), media_type="text/html")


@app.get("/methodology", response_class=HTMLResponse)
async def serve_methodology_page():
    """Serve the methodology disclosure page."""
    meth_path = FRONTEND_DIR / "methodology.html"
    if not meth_path.exists():
        raise HTTPException(status_code=404, detail="methodology.html not found")
    return FileResponse(str(meth_path), media_type="text/html")


# ─────────────────────────────────────────────────────────────────────────────
# Fleet configuration endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/fleet-config")
async def get_fleet_config():
    """Return Constellation/Calpine fleet data for the fleet configuration UI."""
    fleet_path = MARKET_SIM_ROOT / "data" / "constellation_fleet.json"
    if not fleet_path.exists():
        raise HTTPException(status_code=404, detail="constellation_fleet.json not found")
    import json
    with open(fleet_path) as f:
        return json.load(f)


@app.get("/api/sweep-dispatch-data")
async def get_sweep_dispatch_data():
    """Return pre-exported sweep dispatch data for client-side fleet recalculation."""
    path = FRONTEND_DIR / "data" / "sweep_dispatch_data.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="sweep_dispatch_data.json not found. Run scripts/export_sweep_dispatch_data.py first.")
    return FileResponse(str(path), media_type="application/json")


@app.get("/api/fleet-scenarios-config")
async def get_fleet_scenarios_config():
    """Return fleet scenario configuration (base fleet + scenarios + targets)."""
    path = MARKET_SIM_ROOT / "fleet_scenarios" / "constellation_scenarios.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="constellation_scenarios.json not found")
    return FileResponse(str(path), media_type="application/json")


@app.get("/api/fleet-scenario-results")
async def get_fleet_scenario_results():
    """Return pre-computed fleet scenario results.

    Prefers the Rosetta-derived sample data (full fleet with nuclear/renewables)
    over sweep-based results which only cover fossil dispatch.
    """
    # Primary: Rosetta-derived full fleet results (build_fleet_scenario_data.py output)
    sample_path = FRONTEND_DIR / "data" / "fleet_scenario_results_sample.json"
    if sample_path.exists():
        return FileResponse(str(sample_path), media_type="application/json")
    # Fallback: sweep-based results (fossil only)
    path = MARKET_SIM_ROOT / "results" / "fleet_scenario_results.json"
    if path.exists():
        return FileResponse(str(path), media_type="application/json")
    raise HTTPException(status_code=404, detail="No fleet scenario results found")


# ─────────────────────────────────────────────────────────────────────────────
# File download endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/download/{run_id}/{filename}")
async def download_result_file(run_id: str, filename: str):
    """Download a result file from a specific run directory."""
    # Sanitize filename to prevent directory traversal
    import re
    if not re.match(r'^[\w\-\.]+$', filename):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not re.match(r'^[\w\-]+$', run_id):
        raise HTTPException(status_code=400, detail="Invalid run_id")

    file_path = RESULTS_DIR / run_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    return FileResponse(str(file_path), filename=filename)


# ─────────────────────────────────────────────────────────────────────────────
# ISO metadata endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/isos", response_model=List[ISOSummary])
async def list_isos():
    """List available ISOs with summary data."""
    summaries = []
    for iso in ISOS:
        summaries.append(ISOSummary(
            iso=iso,
            demand_twh=REGIONAL_DEMAND_TWH[iso],
            grid_mix=GRID_MIX_SHARES.get(iso, {}),
            capacity_market_price=CAPACITY_MARKET_PRICES.get(iso, 0),
            installed_fossil_mw=INSTALLED_FOSSIL_MW.get(iso, 0),
            fossil_capacity_shares=FOSSIL_CAPACITY_SHARES.get(iso, {}),
        ))
    return summaries


@app.get("/api/defaults/{iso}", response_model=ISODefaults)
async def get_iso_defaults(iso: str):
    """Return default parameters for a specific ISO."""
    iso = iso.upper()
    if iso not in ISOS:
        raise HTTPException(
            status_code=404,
            detail=f"ISO '{iso}' not found. Available: {', '.join(ISOS)}",
        )
    return ISODefaults(
        iso=iso,
        demand_twh=REGIONAL_DEMAND_TWH[iso],
        grid_mix=GRID_MIX_SHARES.get(iso, {}),
        capacity_market_price=CAPACITY_MARKET_PRICES.get(iso, 0),
        installed_fossil_mw=INSTALLED_FOSSIL_MW.get(iso, 0),
        fossil_capacity_shares=FOSSIL_CAPACITY_SHARES.get(iso, {}),
        fuel_prices=FUEL_PRICES,
        heat_rates=HEAT_RATES,
        vom=VOM,
        co2_rates=CO2_RATES,
        nox_rates=NOX_RATES,
        sox_rates=SOX_RATES,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Simulation endpoint
# ─────────────────────────────────────────────────────────────────────────────

def _map_request_to_conditions(req: SimulationRequest) -> dict:
    """Convert a SimulationRequest into the conditions dict expected by
    run_market_simulation."""

    # Map gas_friction string level to numeric
    gas_friction_val = GAS_FRICTION_LEVELS.get(req.gas_friction, 0.7)

    # Learning speed: use explicit user setting if provided, else derive from queue cap
    queue_cap_level = getattr(req, 'queue_cap_level', None) or 'Medium'
    learning_speed = getattr(req, 'learning_speed', None)
    if not learning_speed or learning_speed not in ('Slow', 'Medium', 'Fast'):
        learning_speed = QUEUE_LEARNING_MAP.get(queue_cap_level, 'Medium')

    # Determine fuel level from fuel_prices (use closest named level)
    # If custom prices provided, find best match; otherwise use "Medium"
    fuel_level = "Medium"
    if req.fuel_prices:
        gas = req.fuel_prices.gas
        if gas <= 2.5:
            fuel_level = "Low"
        elif gas >= 5.0:
            fuel_level = "High"

    # Map transmission level
    tx_level = req.transmission_level if req.transmission_level != "None" else "None"

    # LCOE level: infer from clean_lcoes (approximate)
    lcoe_level = "Medium"
    if req.clean_lcoes:
        avg_lcoe = (req.clean_lcoes.solar + req.clean_lcoes.wind) / 2
        if avg_lcoe < 55:
            lcoe_level = "Low"
        elif avg_lcoe > 75:
            lcoe_level = "High"

    # Build custom fuel prices dict if user provided specific values
    custom_fuel_prices = None
    if req.fuel_prices:
        custom_fuel_prices = {
            'coal': req.fuel_prices.coal,
            'gas': req.fuel_prices.gas,
            'oil': req.fuel_prices.oil,
        }

    # Queue cap override (GW/yr) from user
    queue_cap_override = getattr(req, 'queue_cap_override_gw', None)

    # Validate and cap demand_growth: if numeric, cap at 7.5%
    demand_growth = req.demand_growth
    if isinstance(demand_growth, (int, float)):
        demand_growth = min(float(demand_growth), 7.5)

    return {
        "name": f"API: {req.iso} | Q={queue_cap_level} | {demand_growth} demand | carbon=${req.carbon_price}",
        "demand_growth": demand_growth,
        "lcoe_level": lcoe_level,
        "learning_speed": learning_speed,
        "queue_cap_level": queue_cap_level,
        "queue_cap_override_gw": queue_cap_override,
        "gas_friction": gas_friction_val,
        "carbon_price": req.carbon_price,
        "fuel_level": fuel_level,
        "tx_level": tx_level,
        "ppa_level": req.ppa_level,
        # NOx/SOx emission pricing and limits
        "nox_price": req.emission_prices.nox if req.emission_prices else 0.0,
        "sox_price": req.emission_prices.sox if req.emission_prices else 0.0,
        "nox_limit": req.emission_limits.nox_limit if req.emission_limits else None,
        "sox_limit": req.emission_limits.sox_limit if req.emission_limits else None,
        # Custom fuel prices (override fuel_level presets)
        "custom_fuel_prices": custom_fuel_prices,
        "custom_co2_price": req.carbon_price,
        # Custom heat rates and VOM (override lmp_engine defaults)
        "custom_heat_rates": {
            'coal_steam': req.heat_rates.coal_steam,
            'gas_ccgt': req.heat_rates.gas_ccgt,
            'gas_ct': req.heat_rates.gas_ct,
            'oil_ct': req.heat_rates.oil_ct,
            'new_gas_ccgt': req.heat_rates.new_gas_ccgt,
            'new_gas_ct': req.heat_rates.new_gas_ct,
            'new_coal': req.heat_rates.new_coal,
        } if req.heat_rates else None,
        "custom_vom": {
            'coal_steam': req.vom.coal_steam,
            'gas_ccgt': req.vom.gas_ccgt,
            'gas_ct': req.vom.gas_ct,
            'oil_ct': req.vom.oil_ct,
        } if req.vom else None,
        # Wholesale price override
        "wholesale_price_override": req.wholesale_price_override,
        # Incentives (PTC/ITC/REC)
        "ptc_wind": req.incentives.ptc_wind if req.incentives else 26.0,
        "ptc_solar": req.incentives.ptc_solar if req.incentives else 26.0,
        "ptc_nuclear_new": req.incentives.ptc_nuclear_new if req.incentives else 26.0,
        "ptc_45u_max": req.incentives.ptc_45u_max if req.incentives else 15.0,
        "ptc_45u_floor": req.incentives.ptc_45u_floor if req.incentives else 40.0,
        "ptc_45u_floor_escalation": req.incentives.ptc_45u_floor_escalation if req.incentives else 0.0,
        "ptc_45u_sunset_year": req.incentives.ptc_45u_sunset_year if req.incentives else 2032,
        "itc_pct": req.incentives.itc_pct if req.incentives else 30.0,
        "rec_price_override": req.incentives.rec_price if req.incentives else None,
        # 45Q toggle and CCS credit override
        "q45": req.q45,
        "ccs_credit_override": req.ccs_credit_override,
        # Capacity market price override
        "capacity_market_price": req.capacity_market_price,
        # Storage costs ($/kW-yr → $/MWh LCOS conversion)
        "custom_storage_lcoe": {
            'battery': req.storage_costs.battery / 1.241,
            'battery8': req.storage_costs.battery8 / 2.040,
            'ldes': req.storage_costs.ldes / 0.500,
        } if req.storage_costs else None,
        # Custom LCOE overrides
        "custom_lcoes": {
            'solar': req.clean_lcoes.solar,
            'wind': req.clean_lcoes.wind,
            'offshore_wind': req.clean_lcoes.offshore_wind,
            'nuclear': req.clean_lcoes.nuclear,
            'ccs_ccgt': req.clean_lcoes.ccs_ccgt,
            'geothermal': req.clean_lcoes.geothermal if req.clean_lcoes else None,
        } if req.clean_lcoes else None,
        # Inter-regional interchange toggle
        "interchange_enabled": req.interchange_enabled,
        # Demand response level
        "dr_level": req.dr_level,
        # Per-resource transmission overrides ($/MWh) — None = use master L/M/H
        "tx_overrides": req.tx_overrides,
        # Fossil new-build LCOEs ($/MWh) — None = use defaults
        "custom_fossil_lcoes": {
            'gas_ccgt': req.fossil_lcoes.get('gas_ccgt'),
            'gas_ct': req.fossil_lcoes.get('gas_ct'),
            'coal': req.fossil_lcoes.get('coal'),
        } if req.fossil_lcoes else None,
        # New-build fossil parameters
        "new_fossil_cost_level": getattr(req, 'new_fossil_cost_level', 'Medium'),
        "new_fossil_enabled": getattr(req, 'new_fossil_enabled', True),
        "new_fossil_capex_override": getattr(req, 'new_fossil_capex_override', None) or {},
        "new_fossil_min_cf_override": getattr(req, 'new_fossil_min_cf_override', None) or {},
        # Learning curves toggle — False = skip Wright's Law cost decline
        "learning_curves_enabled": getattr(req, 'learning_curves', True),
        # Tech-differentiated queue caps (per-technology interconnection limits)
        "tech_differentiated_queue": getattr(req, 'tech_differentiated_queue', True),
        # Scarcity pricing mode: 'ordc' or 'demand_quantile'
        "scarcity_mode": getattr(req, 'scarcity_mode', 'ordc'),
    }


def _extract_generator_economics(gen_econ_raw: dict) -> List[GeneratorEconomics]:
    """Convert raw generator economics dict into typed list."""
    results = []
    if not gen_econ_raw or not isinstance(gen_econ_raw, dict):
        return results

    for unit_type, data in gen_econ_raw.items():
        if not isinstance(data, dict):
            continue
        mc = data.get("marginal_cost", 0)
        avg_rev = data.get("avg_revenue_mwh", data.get("avg_lmp", 0))
        profit = avg_rev - mc
        cf = data.get("capacity_factor", 0)

        if profit > 5:
            status = "profitable"
        elif profit > -5:
            status = "marginal"
        else:
            status = "retiring"

        results.append(GeneratorEconomics(
            unit_type=unit_type,
            capacity_mw=data.get("capacity_mw", 0),
            marginal_cost=round(mc, 2),
            dispatch_hours=int(data.get("dispatch_hours", cf * 8760)),
            capacity_factor=round(cf, 4),
            avg_revenue_mwh=round(avg_rev, 2),
            profit_mwh=round(profit, 2),
            status=status,
        ))
    return results


def _extract_nuclear_revenue(nuc_raw: dict) -> NuclearRevenue:
    """Convert raw nuclear revenue dict."""
    if not nuc_raw or not isinstance(nuc_raw, dict):
        return NuclearRevenue()
    return NuclearRevenue(
        energy_rev_mwh=nuc_raw.get("energy_rev_mwh", 0),
        capacity_rev_mwh=nuc_raw.get("capacity_rev_mwh", 0),
        ptc_mwh=nuc_raw.get("ptc_mwh", 0),
        total_mwh=nuc_raw.get("total_mwh", 0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Chart data computation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_threshold_sweep(year_results, zone_details, existing_clean, base_lmp):
    """Build threshold → avg_lmp map from zone details across all years.

    JS expects: {threshold_str: {avg_lmp: float}}
    """
    sweep = {}
    # Start with baseline
    sweep[str(round(existing_clean, 1))] = {"avg_lmp": round(base_lmp, 1)}

    # Collect zone details from ALL year results (trajectory has multiple years)
    for yr in year_results:
        for zd in yr.get("zone_details", []):
            t = zd.get("threshold", 0)
            lmp = zd.get("avg_lmp", 0)
            if t > 0 and lmp > 0:
                sweep[str(round(t, 1))] = {"avg_lmp": round(lmp, 1)}

    # Also use zone_details from the response-level list
    for zd in zone_details:
        t = zd.threshold if hasattr(zd, 'threshold') else zd.get("threshold", 0)
        lmp = zd.avg_lmp if hasattr(zd, 'avg_lmp') else zd.get("avg_lmp", 0)
        if t > 0 and lmp > 0:
            sweep[str(round(t, 1))] = {"avg_lmp": round(lmp, 1)}

    return sweep if len(sweep) > 1 else None


def _compute_what_gets_built(final_year):
    """Extract cumulative GW deployed by resource type.

    JS expects: {resource_name: gw_float}
    """
    cum_gw = final_year.get("cumulative_gw", {})
    if not cum_gw:
        # Fallback: derive from resource_mix_twh using typical capacity factors
        rmix = final_year.get("resource_mix_twh", {})
        cf_map = {
            'solar': 0.25, 'wind': 0.35, 'offshore_wind': 0.45,
            'clean_firm': 0.90, 'nuclear': 0.92, 'ccs_ccgt': 0.85,
            'hydro': 0.40, 'geothermal': 0.90,
        }
        cum_gw = {}
        for res, twh in rmix.items():
            cf = cf_map.get(res, 0.30)
            gw = twh / (cf * 8.760) if cf > 0 else 0
            if gw > 0.01:
                cum_gw[res] = round(gw, 1)

    # Filter out zero/negative values
    built = {k: round(v, 1) for k, v in cum_gw.items() if v > 0.01}
    return built if built else None


def _compute_cost_ladder(year_results):
    """Build cost ladder: cumulative GW vs $/MWh cost and revenue.

    JS expects: [{cumulative_gw, cost_mwh, revenue_mwh}, ...]
    """
    # Collect all zone details with cost/revenue data
    zones = []
    for yr in year_results:
        for zd in yr.get("zone_details", []):
            cost = zd.get("cost", 0)
            rev = zd.get("revenue", 0)
            new_gw = zd.get("new_gw", 0)
            t = zd.get("threshold", 0)
            if new_gw > 0 and (cost > 0 or rev > 0):
                zones.append({
                    "threshold": t,
                    "cost": cost,
                    "revenue": rev,
                    "new_gw": new_gw,
                })

    if not zones:
        return None

    # Sort by threshold (ascending clean %) and compute cumulative GW
    zones.sort(key=lambda z: z["threshold"])
    ladder = []
    cum_gw = 0
    for z in zones:
        cum_gw += z["new_gw"]
        ladder.append({
            "cumulative_gw": round(cum_gw, 1),
            "cost_mwh": round(z["cost"], 1),
            "revenue_mwh": round(z["revenue"], 1),
        })

    return ladder if ladder else None


def _compute_gas_fleet_shift(iso, final_year):
    """Compute gas fleet capacity factors by efficiency tier at different carbon prices.

    JS expects: [{carbon_price, efficient_cf, avg_cf, old_cf}, ...]
    """
    # Get baseline generator economics to establish capacity factors
    gen_econ = final_year.get("generator_economics", {})

    # Efficient = gas_ccgt (HR ~7.0), Average = fleet avg (~8.5), Old = gas_ct (HR ~10.5)
    efficient_hr = HEAT_RATES.get('gas_ccgt', 7.0)
    avg_hr = (HEAT_RATES.get('gas_ccgt', 7.0) + HEAT_RATES.get('gas_ct', 10.5)) / 2
    old_hr = HEAT_RATES.get('gas_ct', 10.5)

    final_yr = final_year.get('year', 2050)
    base_gas_price = _get_fuel_prices(final_yr, 'Medium').get('gas', 3.5)
    base_lmp = final_year.get("avg_lmp", WHOLESALE_PRICES.get(iso, 30))

    carbon_prices = [0, 10, 25, 50, 75, 100, 150, 200]
    shift_data = []

    for cp in carbon_prices:
        # Marginal cost at each efficiency tier
        mc_efficient = efficient_hr * base_gas_price + VOM.get('gas_ccgt', 3.5) + CO2_RATES.get('gas_ccgt', 0.37) * cp
        mc_avg = avg_hr * base_gas_price + 4.0 + 0.46 * cp
        mc_old = old_hr * base_gas_price + VOM.get('gas_ct', 5.0) + CO2_RATES.get('gas_ct', 0.55) * cp

        # LMP increases with carbon price (pass-through from marginal generator)
        # Marginal generator is typically mid-efficiency gas
        lmp_at_cp = base_lmp + 0.46 * cp  # ~0.46 tCO2/MWh avg fossil

        # CF approximation: higher profit margin → higher dispatch
        # CF = clamp(0.05, (lmp - mc) / lmp * base_cf_factor, 0.95)
        def cf_from_margin(mc, lmp):
            if lmp <= 0:
                return 0.05
            margin_ratio = max(0, (lmp - mc) / lmp)
            return min(0.90, max(0.02, margin_ratio * 0.85))

        shift_data.append({
            "carbon_price": cp,
            "efficient_cf": round(cf_from_margin(mc_efficient, lmp_at_cp), 3),
            "avg_cf": round(cf_from_margin(mc_avg, lmp_at_cp), 3),
            "old_cf": round(cf_from_margin(mc_old, lmp_at_cp), 3),
        })

    return shift_data


def _compute_sensitivity_matrix(iso, base_clean_pct, base_lmp):
    """Compute clean % sensitivity to gas price × carbon price.

    JS expects: {values: number[][], gas_prices: number[], carbon_prices: number[]}
    """
    gas_prices = [2.0, 3.0, 3.5, 4.5, 6.0]
    carbon_prices = [0, 25, 50, 75, 100]

    # Base parameters
    base_gas = 3.5
    base_carbon = 0

    # Sensitivity: higher gas → higher LMP → more clean competitive
    # Higher carbon → higher fossil cost → more clean competitive
    values = []
    for cp in carbon_prices:
        row = []
        for gp in gas_prices:
            # Each $1/MMBtu gas increase raises avg fossil MC by ~$7-8/MWh (HR ~7-8)
            gas_delta = (gp - base_gas) * 7.5  # $/MWh LMP increase
            # Each $10/ton carbon raises MC by ~$3.7-5.5/MWh (avg ~$4.5/MWh)
            carbon_delta = (cp - base_carbon) * 0.45  # $/MWh

            # LMP increase makes clean more competitive → more clean deployed
            # Approximate: each $10/MWh LMP increase → +3-5% clean
            lmp_increase = gas_delta + carbon_delta
            pct_increase = lmp_increase * 0.35  # ~3.5% per $10/MWh

            clean_pct = min(99.9, max(10, base_clean_pct + pct_increase))
            row.append(round(clean_pct, 1))
        values.append(row)

    return {
        "values": values,
        "gas_prices": gas_prices,
        "carbon_prices": carbon_prices,
    }


def _compute_ccs_analysis(iso):
    """Compute CCS retrofit vs new gas cost curves at different carbon prices.

    JS expects: {carbon_prices: [], existing_ccgt_cost: [], ccs_retrofit_cost: [], new_gas_cost: []}
    """
    carbon_prices = [0, 10, 25, 50, 75, 100, 150, 200]

    gas_price = _get_fuel_prices(2050, 'Medium').get('gas', 3.5)  # End-of-horizon price for CCS viability

    # Existing CCGT: HR=7.0, VOM=$3.50, CO2=0.37 t/MWh (no capture)
    ccgt_hr = HEAT_RATES.get('gas_ccgt', 7.0)
    ccgt_vom = VOM.get('gas_ccgt', 3.5)
    ccgt_co2 = CO2_RATES.get('gas_ccgt', 0.37)

    # CCS retrofit: HR=8.4 (20% energy penalty), VOM=$8, CO2=0.037 (90% capture)
    # Plus ~$30/MWh capex amortization for retrofit
    ccs_hr = ccgt_hr * 1.20  # 20% efficiency penalty
    ccs_vom = 8.0
    ccs_co2 = ccgt_co2 * 0.10  # 90% capture
    ccs_capex_mwh = 30.0  # Amortized retrofit cost
    ccs_45q = 27.5  # 45Q credit per ton captured (§45Q at $85/ton × capture rate)

    # New gas CCGT: HR=6.4 (newer fleet), VOM=$3.0, CO2=0.34
    new_hr = 6.4
    new_vom = 3.0
    new_co2 = 0.34
    new_capex_mwh = 12.0  # Amortized new-build cost

    existing_costs = []
    ccs_costs = []
    new_gas_costs = []

    for cp in carbon_prices:
        # Existing CCGT total cost
        existing = ccgt_hr * gas_price + ccgt_vom + ccgt_co2 * cp
        existing_costs.append(round(existing, 1))

        # CCS retrofit total cost (with 45Q credit offset)
        tons_captured = (ccgt_co2 - ccs_co2)  # tons captured per MWh
        ccs_credit = min(ccs_45q, tons_captured * 85)  # $85/ton × captured
        ccs_total = ccs_hr * gas_price + ccs_vom + ccs_co2 * cp + ccs_capex_mwh - ccs_credit
        ccs_costs.append(round(ccs_total, 1))

        # New gas CCGT
        new_total = new_hr * gas_price + new_vom + new_co2 * cp + new_capex_mwh
        new_gas_costs.append(round(new_total, 1))

    return {
        "carbon_prices": carbon_prices,
        "existing_ccgt_cost": existing_costs,
        "ccs_retrofit_cost": ccs_costs,
        "new_gas_cost": new_gas_costs,
    }


def _get_data_tiers(iso: str) -> dict:
    """Get multi-tier data quality info for a single ISO."""
    ds = check_data_sources()
    return ds.get('tiers', {}).get(iso, {})


def _aggregate_data_quality(year_results: list) -> Optional[dict]:
    """Aggregate data_quality across all typed YearResult objects."""
    any_synthetic = any(
        (getattr(yr, 'data_quality', None) or {}).get('synthetic_backed', False)
        for yr in year_results
    )
    all_missing = set()
    for yr in year_results:
        all_missing.update(
            (getattr(yr, 'data_quality', None) or {}).get('missing_sources', [])
        )
    if not any_synthetic and not all_missing:
        return None
    return {'synthetic_backed': any_synthetic, 'missing_sources': sorted(all_missing)}


def _compute_result_caveats(iso: str, final_year: dict) -> List[str]:
    """Build a list of caveats relevant to this specific simulation run."""
    from pipeline_config import ZONE_CONFIG, OFFSHORE_ISOS, GEOTHERMAL_ISOS

    caveats: List[str] = []

    # Data source caveat
    data_source = final_year.get('data_source', 'unknown')
    if data_source == 'synthetic':
        caveats.append(
            f"Results for {iso} use synthetic generation profiles — "
            "treat as illustrative, not calibrated."
        )

    # Zonal LMP availability
    if iso not in ZONE_CONFIG:
        caveats.append(
            f"Zonal LMP decomposition not available for {iso} — "
            "only system-level LMP reported."
        )

    # Storage dispatch model limitation
    caveats.append(
        "Storage deployment uses greedy daily-cycle dispatch — "
        "see peer review C1 for LP co-optimization caveat."
    )

    # Offshore wind applicability
    if iso not in OFFSHORE_ISOS:
        resource_mix = final_year.get('resource_mix_twh', {})
        if resource_mix.get('offshore_wind', 0) > 0:
            caveats.append(
                f"Offshore wind is not a validated resource for {iso}."
            )

    # Geothermal applicability
    if iso not in GEOTHERMAL_ISOS:
        resource_mix = final_year.get('resource_mix_twh', {})
        if resource_mix.get('geothermal', 0) > 0:
            caveats.append(
                f"Geothermal is only modeled for CAISO — "
                f"results for {iso} exclude geothermal potential."
            )

    # Interchange caveat
    tiers = _get_data_tiers(iso)
    if tiers.get('interchange') == 'none':
        caveats.append(
            f"Inter-regional interchange data not available for {iso} — "
            "using copper-plate isolation assumption."
        )

    # Confidence zone warnings from IPM triggers
    ipm_triggers = final_year.get('ipm_triggers', [])
    high_triggers = [t for t in ipm_triggers
                     if isinstance(t, dict) and t.get('severity') == 'high']
    if high_triggers:
        caveats.append(
            f"{len(high_triggers)} high-severity IPM trigger(s) detected — "
            "consider validating with a production dispatch model."
        )

    return caveats


def _build_simulation_response(iso: str, year_results: list) -> SimulationResponse:
    """Build a SimulationResponse from raw simulation year_results for one ISO."""
    if not year_results:
        existing_clean = sum(GRID_MIX_SHARES.get(iso, {}).values())
        return SimulationResponse(
            iso=iso,
            existing_clean_pct=existing_clean,
            market_outcome_clean_pct=existing_clean,
            avg_lmp=WHOLESALE_PRICES.get(iso, 30.0),
            demand_twh=REGIONAL_DEMAND_TWH.get(iso, 0),
        )

    # Use the last year's result as the summary
    final = year_results[-1]
    existing_clean = sum(GRID_MIX_SHARES.get(iso, {}).values())

    # Parse generator economics
    gen_econ = _extract_generator_economics(final.get("generator_economics", {}))
    nuc_rev = _extract_nuclear_revenue(final.get("nuclear_revenue", {}))

    # Zone details
    zone_details = []
    for zd in final.get("zone_details", []):
        zone_details.append(ZoneDetail(**zd))

    # CCS breakeven
    ccs_be = final.get("ccs_breakeven", {})
    ccs_be_price = ccs_be.get("carbon_price_breakeven", 0) if isinstance(ccs_be, dict) else 0

    # Typed year results
    typed_years = []
    for yr in year_results:
        # Compute confidence: year-based zone, then adjust for IPM trigger severity
        year_zone = get_confidence_zone(yr.get("year", 0))
        adj_zone, was_adjusted = adjust_confidence_for_triggers(
            year_zone, yr.get("ipm_triggers", [])
        )
        conf_zone = adj_zone
        conf_label = CONFIDENCE_ZONES[conf_zone]['label']
        if was_adjusted:
            conf_label += ' (adjusted)'

        typed_years.append(YearResult(
            iso=yr.get("iso", iso),
            scenario=yr.get("scenario", ""),
            year=yr.get("year", 0),
            clean_pct=yr.get("clean_pct", 0),
            demand_twh=yr.get("demand_twh", 0),
            emissions_mt=yr.get("emissions_mt", 0),
            emission_rate_tco2_mwh=yr.get("emission_rate_tco2_mwh", 0),
            cost_per_mwh=yr.get("cost_per_mwh", 0),
            revenue_per_mwh=yr.get("revenue_per_mwh", 0),
            energy_rev_mwh=yr.get("energy_rev_mwh", 0),
            capacity_rev_mwh=yr.get("capacity_rev_mwh", 0),
            rec_rev_mwh=yr.get("rec_rev_mwh", 0),
            avg_lmp=yr.get("avg_lmp", 0),
            lmp_p90=yr.get("lmp_p90", 0),
            gas_built_gw=yr.get("gas_built_gw", 0),
            fossil_built_gw=yr.get("fossil_built_gw", 0),
            total_gas_gw=yr.get("total_gas_gw", 0),
            total_new_fossil_mw=yr.get("total_new_fossil_mw", 0),
            new_fossil_builds_mw=yr.get("new_fossil_builds_mw", {}),
            market_stop=yr.get("market_stop", False),
            confidence=conf_zone,
            confidence_label=conf_label,
            confidence_adjusted=was_adjusted,
            resource_mix_twh=yr.get("resource_mix_twh", {}),
            cumulative_gw=yr.get("cumulative_gw", {}),
            zones_deployed=yr.get("zones_deployed", []),
            zone_details=[
                ZoneDetail(**zd) if isinstance(zd, dict) else zd
                for zd in yr.get("zone_details", [])
            ],
            generator_economics=yr.get("generator_economics", {}),
            emissions_by_fuel=yr.get("emissions_by_fuel", {}),
            nuclear_revenue=yr.get("nuclear_revenue", {}),
            nuclear_retired=yr.get("nuclear_retired", False),
            ccs_breakeven=yr.get("ccs_breakeven", {}),
            economic_retirements_mw=yr.get("economic_retirements_mw", {}),
            total_economic_retirement_mw=yr.get("total_economic_retirement_mw", 0),
            ipm_triggers=[
                IPMTrigger(**t) if isinstance(t, dict) else t
                for t in yr.get("ipm_triggers", [])
            ],
            data_source=yr.get("data_source", "unknown"),
            data_quality=yr.get("data_quality"),
        ))

    # Build emissions_by_fuel_by_year aggregate for trajectory chart
    emissions_by_fuel_by_year = None
    if len(year_results) > 1:
        all_fuel_types = set()
        for yr in year_results:
            all_fuel_types.update(yr.get("emissions_by_fuel", {}).keys())
        emissions_by_fuel_by_year = {
            "years": [yr.get("year", 0) for yr in year_results],
        }
        for fuel in sorted(all_fuel_types):
            emissions_by_fuel_by_year[fuel] = [
                yr.get("emissions_by_fuel", {}).get(fuel, 0) for yr in year_results
            ]

    # Build supply stack summary from resource_mix_twh
    supply_stack = []
    rmix = final.get("resource_mix_twh", {})
    for res, twh in sorted(rmix.items(), key=lambda x: -x[1]):
        if twh > 0:
            supply_stack.append(SupplyStackEntry(
                resource=res,
                capacity_gw=0,  # Not available from current data
                generation_twh=round(twh, 1),
            ))

    # Build fuel bin table from generator_economics
    # Use adjusted economics (with economic retirements applied) when available
    fuel_bins = []
    gen_raw = final.get("adjusted_generator_economics", final.get("generator_economics", {}))
    if isinstance(gen_raw, dict):
        for unit_type, data in gen_raw.items():
            if not isinstance(data, dict):
                continue
            cap_mw = data.get("capacity_mw", 0)
            cf = data.get("capacity_factor", data.get("cf", 0))
            gen_twh = cap_mw * cf * 8760 / 1e6  # MW × CF × hours → TWh
            mc = data.get("marginal_cost", data.get("var_cost_mwh", 0))
            avg_rev = data.get("avg_revenue_mwh", data.get("avg_rev_mwh", 0))
            margin = avg_rev - mc if avg_rev and mc else 0

            if margin > 5:
                status = "operating"
            elif margin > -5:
                status = "marginal"
            else:
                status = "retiring"

            fuel_bins.append(FuelBinRow(
                fuel_type=unit_type,
                heat_rate_bin="—",
                capacity_gw=round(cap_mw / 1000, 2),
                capacity_factor=round(cf, 4),
                generation_twh=round(gen_twh, 1),
                marginal_cost=round(mc, 1),
                avg_revenue=round(avg_rev, 1),
                status=status,
            ))

    # Build LMP time series from year_results (trajectory) or hourly data
    lmp_ts = None
    cap_rev_ts = None
    if len(year_results) > 1:
        lmp_ts = HourlyProfile(
            hours=[float(yr.get("year", 0)) for yr in year_results],
            values=[float(yr.get("avg_lmp", 0)) for yr in year_results],
            label="Avg LMP by Year",
        )
        cap_rev_ts = HourlyProfile(
            hours=[float(yr.get("year", 0)) for yr in year_results],
            values=[float(yr.get("capacity_rev_mwh", 0)) for yr in year_results],
            label="Capacity Revenue by Year",
        )

    # ── Compute additional chart data ──

    # 1. Threshold sweep: LMP at each threshold from zone_details
    threshold_sweep = _compute_threshold_sweep(
        year_results, zone_details, existing_clean,
        final.get("avg_lmp", WHOLESALE_PRICES.get(iso, 30)),
    )

    # 2. What gets built: cumulative GW by resource
    what_gets_built = _compute_what_gets_built(final)

    # 3. Cost ladder: cost vs cumulative GW from zone details
    cost_ladder = _compute_cost_ladder(year_results)

    # 4. Gas fleet shift: CF by efficiency tier at different carbon prices
    gas_fleet_shift = _compute_gas_fleet_shift(iso, final)

    # 5. Sensitivity matrix: clean % at gas × carbon price grid
    sensitivity_matrix = _compute_sensitivity_matrix(
        iso, final.get("clean_pct", existing_clean),
        final.get("avg_lmp", 30),
    )

    # 6. CCS analysis: cost curves at different carbon prices
    ccs_analysis = _compute_ccs_analysis(iso)

    # Extract sim_years from year_results
    sim_years_list = sorted(set(yr.get("year", 0) for yr in year_results))

    # Build result caveats based on data availability and model limitations
    caveats = _compute_result_caveats(iso, final)

    return SimulationResponse(
        iso=iso,
        existing_clean_pct=round(existing_clean, 1),
        market_outcome_clean_pct=final.get("clean_pct", existing_clean),
        avg_lmp=final.get("avg_lmp", 0),
        generator_economics=gen_econ,
        nuclear_revenue=nuc_rev,
        ccs_breakeven_carbon_price=round(ccs_be_price, 2),
        emissions_mt=final.get("emissions_mt", 0),
        demand_twh=final.get("demand_twh", 0),
        resource_mix_twh=final.get("resource_mix_twh", {}),
        sim_years=sim_years_list,
        year_results=typed_years,
        zones_deployed=zone_details,
        lmp_time_series=lmp_ts,
        capacity_rev_time_series=cap_rev_ts,
        supply_stack_summary=supply_stack,
        fuel_bin_table=fuel_bins,
        emissions_by_fuel_by_year=emissions_by_fuel_by_year,
        economic_retirements_mw=final.get("economic_retirements_mw"),
        total_economic_retirement_mw=final.get("total_economic_retirement_mw", 0),
        threshold_sweep=threshold_sweep,
        what_gets_built=what_gets_built,
        cost_ladder=cost_ladder,
        gas_fleet_shift=gas_fleet_shift,
        sensitivity_matrix=sensitivity_matrix,
        ccs_analysis=ccs_analysis,
        # Demand response metrics from final year
        dr_curtailed_gwh=final.get("dr_curtailed_gwh", 0),
        dr_peak_gw=final.get("dr_peak_gw", 0),
        dr_hours=final.get("dr_hours", 0),
        dr_avg_price=final.get("dr_avg_price", 0),
        # Confidence zone metadata for trajectory visualization
        confidence_zones=[
            {'zone': k, **v} for k, v in CONFIDENCE_ZONES.items()
        ],
        data_source=final.get('data_source', 'unknown'),
        data_tiers=_get_data_tiers(iso),
        data_quality=_aggregate_data_quality(typed_years),
        result_caveats=caveats,
    )


@app.post("/api/validate-request")
async def validate_request(req: SimulationRequest):
    """Dry-run validation: check request validity and data availability
    without running a simulation.

    Returns:
        valid: bool — whether all required data is present
        missing_data: list — parquet/profile files not found for this ISO
        scenario_count: int — number of simulation years that would be computed
        caveats: list — known limitations for this ISO/configuration
    """
    iso = req.iso.upper()

    # Map request to conditions (validates parameter mapping)
    conditions = _map_request_to_conditions(req)

    # Check data availability
    ds = check_data_sources()
    simple = ds.get('simple', {})
    tiers = ds.get('tiers', {}).get(iso, {})

    missing: List[str] = []
    if simple.get(iso) == 'synthetic':
        missing.append(f"step2.2-cost parquet for {iso} (using synthetic fallback)")
    elif simple.get(iso) == 'ef_parquet':
        # EF data available but no cost optimization — note but don't flag as missing
        pass
    if tiers.get('interchange') == 'none':
        missing.append(f"EIA interchange profiles for {iso}")
    if tiers.get('fleet_data') in ('none', 'synthetic'):
        missing.append(f"EIA-860 plant-level fleet data for {iso}")

    # Compute scenario count (number of year-steps)
    from market_simulation import build_sim_years
    if req.start_year and req.end_year:
        sim_years = build_sim_years(
            start=req.start_year, end=req.end_year, step=req.year_step or 1,
        )
    else:
        sim_years = req.years
    scenario_count = len(sim_years)

    # Pre-compute caveats
    from pipeline_config import ZONE_CONFIG, OFFSHORE_ISOS
    caveats: List[str] = []
    if simple.get(iso) == 'synthetic':
        caveats.append(f"Results for {iso} will use synthetic profiles (illustrative only).")
    if iso not in ZONE_CONFIG:
        caveats.append(f"Zonal LMP not available for {iso}.")
    caveats.append(
        "Storage deployment uses greedy daily-cycle dispatch — see peer review C1."
    )

    return {
        "valid": len(missing) == 0,
        "missing_data": missing,
        "scenario_count": scenario_count,
        "sim_years": sim_years,
        "caveats": caveats,
        "data_tiers": tiers,
    }


@app.post("/api/simulate", response_model=SimulationResponse)
async def simulate(req: SimulationRequest):
    """Run a single market simulation and return results."""
    iso = req.iso.upper()
    if iso not in ISOS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ISO '{iso}'. Available: {', '.join(ISOS)}",
        )

    try:
        conditions = _map_request_to_conditions(req)

        # Load custom CSV overrides if enabled (time-series aware)
        co = req.custom_overrides
        if co and co.fuel:
            fuel_ts = load_custom_fuel_overrides()
            if fuel_ts:
                conditions['custom_fuel_timeseries'] = fuel_ts
        if co and co.capacity:
            cap_ts = load_custom_capacity_overrides()
            if cap_ts:
                conditions['custom_capacity_timeseries'] = cap_ts
        if co and co.rec:
            rec_ts = load_custom_rec_overrides()
            if rec_ts:
                conditions['custom_rec_timeseries'] = rec_ts

        # Apply scarcity mode from user request to global config
        import pipeline_config
        scarcity_mode = conditions.get('scarcity_mode', 'ordc')
        original_scarcity_mode = pipeline_config.SCARCITY_MODE
        if scarcity_mode in ('ordc', 'demand_quantile'):
            pipeline_config.SCARCITY_MODE = scarcity_mode

        # Determine nuclear retirement threshold
        nrt = req.nuclear_retirement_threshold if req.nuclear_retirement_threshold > 0 else None

        preloaded = _get_preloaded_data()
        lmp_cache: dict = {}

        # Build sim_years from request parameters
        from market_simulation import build_sim_years
        if req.mode == "snapshot":
            # Snapshot: single year only (fast, 5-20s)
            custom_sim_years = [req.start_year or 2025]
        elif req.start_year and req.end_year:
            custom_sim_years = build_sim_years(
                start=req.start_year,
                end=req.end_year,
                step=req.year_step or 1,
            )
        else:
            custom_sim_years = None  # use legacy SIM_YEARS

        results = run_market_simulation(
            scenario_id="API_SINGLE",
            conditions=conditions,
            isos=[iso],
            nuclear_retirement_threshold=nrt,
            snapshot_mode=False,
            sim_years=custom_sim_years,
            _preloaded=preloaded,
            _lmp_cache=lmp_cache,
            _quiet=True,
            _data_sources=preloaded.get('data_sources'),
        )

        iso_results = results.get(iso, [])
        response = _build_simulation_response(iso, iso_results)

        # Compute plant-level economics for ALL simulation years
        plant_level_data = []  # List of dicts, each tagged with 'year'
        plant_summary = None
        try:
            import numpy as np
            from market_simulation import compute_plant_level_economics
            from lmp_engine import build_plant_level_merit_order

            fuel_level = conditions.get("fuel_level", "Medium")
            carbon_price_val = conditions.get("carbon_price", 0)

            # Fuel prices — resolved per-year using EIA AEO projections
            has_fuel_timeseries = 'custom_fuel_timeseries' in conditions
            has_custom_static = bool(conditions.get("custom_fuel_prices"))

            # Iterate over each year result from the simulation
            years_to_process = iso_results if iso_results else []
            last_zonal_stats = None
            for yr_data in years_to_process:
                yr_year = yr_data.get("year", 0)
                yr_clean_pct = yr_data.get("clean_pct", 50.0) / 100.0
                yr_demand_twh = yr_data.get("demand_twh", REGIONAL_DEMAND_TWH.get(iso, 300))
                yr_avg_lmp = yr_data.get("avg_lmp", WHOLESALE_PRICES.get(iso, 30.0))

                # Resolve fuel prices for this year
                if has_fuel_timeseries:
                    from market_simulation import _resolve_fuel_prices_for_year
                    fuel_prices_dict = _resolve_fuel_prices_for_year(
                        conditions, yr_year, iso
                    ) or _get_fuel_prices(yr_year, fuel_level)
                elif has_custom_static:
                    fuel_prices_dict = conditions["custom_fuel_prices"]
                else:
                    # Year-varying EIA AEO 2025 projections
                    fuel_prices_dict = _get_fuel_prices(yr_year, fuel_level)

                try:
                    plant_stack, total_cap = build_plant_level_merit_order(
                        iso, yr_clean_pct,
                        fuel_level=fuel_level,
                        carbon_price=carbon_price_val,
                        custom_fuel_prices=fuel_prices_dict,
                    )

                    hourly_lmp = np.full(8760, yr_avg_lmp)
                    avg_demand_mw = yr_demand_twh * 1e6 / 8760
                    demand_mw_profile = np.full(8760, avg_demand_mw)
                    fossil_frac = max(0, 1.0 - yr_clean_pct)
                    residual_demand = np.full(8760, fossil_frac)
                    dispatch = {'residual_demand': residual_demand}

                    # Zonal LMP (optional — don't fail year if unavailable)
                    zonal_lmp_result = None
                    plant_zone_list = None
                    zone_name_list = None
                    try:
                        from pipeline_config import ZONE_CONFIG
                        if iso in ZONE_CONFIG:
                            from lmp_engine import compute_hourly_lmp_zonal, get_price_model
                            from fleet_model import FleetModel

                            zone_config = ZONE_CONFIG[iso]
                            zone_name_list = zone_config['zones']
                            price_model = get_price_model(iso, fuel_level)

                            fm = FleetModel(iso=iso)
                            fm.build_fleet()
                            if fm.fleet is not None and not fm.fleet.empty:
                                fm.assign_zones()
                                zone_stacks = fm.build_zonal_merit_order_stacks(
                                    fuel_level=fuel_level, co2_price=carbon_price_val,
                                    custom_fuel_prices=fuel_prices_dict)
                                zonal_lmp_matrix, system_lmp_arr, _, yr_zonal_stats = \
                                    compute_hourly_lmp_zonal(
                                        dispatch, demand_mw_profile, zone_stacks,
                                        zone_config, price_model, iso=iso,
                                        vre_penetration=yr_clean_pct,
                                    )
                                zonal_lmp_result = zonal_lmp_matrix
                                hourly_lmp = system_lmp_arr
                                last_zonal_stats = yr_zonal_stats

                                plant_zone_list = []
                                from pipeline_config import get_zone_for_plant
                                for p in plant_stack:
                                    pzone = get_zone_for_plant(
                                        iso,
                                        ba_code=p.get('ba'),
                                        lat=p.get('latitude'),
                                        lon=p.get('longitude'),
                                    )
                                    plant_zone_list.append(pzone)
                    except Exception:
                        pass  # Zonal LMP optional per-year

                    yr_plants = compute_plant_level_economics(
                        plant_stack, hourly_lmp, dispatch,
                        demand_mw_profile, fuel_prices_dict, carbon_price_val,
                        zonal_lmp=zonal_lmp_result,
                        plant_zones=plant_zone_list,
                        zone_names=zone_name_list,
                    )
                    # Tag each plant row with the simulation year
                    for p in yr_plants:
                        p['year'] = yr_year
                    plant_level_data.extend(yr_plants)
                except Exception as yr_err:
                    print(f"Note: Plant-level economics for year {yr_year} failed: {yr_err}")
                    continue

            # Attach zonal LMP stats from the last year to response
            if last_zonal_stats:
                from models import ZonalLMPStats
                response.zonal_lmp_stats = [
                    ZonalLMPStats(
                        zone_name=s['zone_name'],
                        avg_lmp=round(s['avg_lmp'], 2),
                        peak_lmp=round(s['peak_lmp'], 2),
                        offpeak_lmp=round(s['offpeak_lmp'], 2),
                        p10_lmp=round(s['p10_lmp'], 2),
                        p90_lmp=round(s['p90_lmp'], 2),
                        price_spread_vs_system=round(s['price_spread_vs_system'], 2),
                    )
                    for key, s in last_zonal_stats.items()
                    if key != '_congestion'
                ]

            # Compute summary from the final year's plant data
            final_year = years_to_process[-1].get("year", 0) if years_to_process else 0
            final_plants = [p for p in plant_level_data if p.get("year") == final_year]
            operating = sum(1 for p in final_plants if p.get("status") == "operating")
            at_risk = sum(1 for p in final_plants if p.get("status") == "at_risk")
            stranded = sum(1 for p in final_plants if p.get("status") == "stranded")
            plant_summary = {
                "operating": operating,
                "at_risk": at_risk,
                "stranded": stranded,
                "total": len(final_plants),
            }
            response.plant_level_summary = plant_summary
        except Exception as plant_err:
            import traceback as tb
            # Plant-level economics is optional — don't fail the simulation
            print(f"Note: Plant-level economics not available: {plant_err}")
            tb.print_exc()

        # Save results to indexed run directory
        try:
            run_id = _next_run_id(iso)
            response_data = response.model_dump()
            params_data = req.model_dump()
            # Attach plant-level data for CSV saving (not included in JSON response)
            response_data["_plant_level_data"] = plant_level_data
            narrative = _save_run_artifacts(run_id, iso, response_data, params_data)
            response.run_id = run_id
            response.narrative = narrative
        except Exception as save_err:
            # Don't fail the simulation if saving fails
            print(f"Warning: Failed to save run artifacts: {save_err}")

        return response

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")
    finally:
        # Restore global scarcity mode
        try:
            pipeline_config.SCARCITY_MODE = original_scarcity_mode
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Sweep endpoints (async background task)
# ─────────────────────────────────────────────────────────────────────────────

def _run_sweep_sync(job_id: str, req: SweepRequest):
    """Run the full sweep synchronously (called from background task)."""
    job = _sweep_jobs[job_id]
    job.status = SweepStatus.running

    try:
        t0 = time.time()
        isos = [iso.upper() for iso in req.isos] if req.isos else list(ISOS)
        for iso in isos:
            if iso not in ISOS:
                raise ValueError(f"Invalid ISO: {iso}")

        # Build scenario list (optionally filtered)
        all_scenarios = build_market_scenarios()

        # Filter by queue cap levels if specified
        if req.queue_cap_levels:
            q_set = set(req.queue_cap_levels)
            all_scenarios = [
                (sid, c) for sid, c in all_scenarios
                if c.get('queue_cap_level', 'Medium') in q_set
            ]

        job.total_scenarios = len(all_scenarios)

        preloaded = _get_preloaded_data()
        lmp_cache: dict = {}
        all_results = {}

        # Build sim_years for sweep
        from market_simulation import build_sim_years
        sweep_sim_years = build_sim_years(
            start=req.start_year,
            end=req.end_year,
            step=req.year_step or 5,
        )

        for i, (scenario_id, conditions) in enumerate(all_scenarios):
            results = run_market_simulation(
                scenario_id=scenario_id,
                conditions=conditions,
                isos=isos,
                nuclear_retirement_threshold=req.nuclear_retirement_threshold,
                snapshot_mode=False,
                sim_years=sweep_sim_years,
                _preloaded=preloaded,
                _lmp_cache=lmp_cache,
                _quiet=True,
                _data_sources=preloaded.get('data_sources'),
            )
            all_results[scenario_id] = results
            job.completed_scenarios = i + 1
            job.progress = round((i + 1) / job.total_scenarios * 100, 1)
            job.elapsed_seconds = round(time.time() - t0, 1)

        # Save to disk
        output_dir = str(MARKET_SIM_ROOT / "data" / "results")
        save_results(all_results, output_dir=output_dir)

        # Store summary in job (full results can be large — store file path)
        job.results = {
            "scenario_count": len(all_results),
            "iso_count": len(isos),
            "isos": isos,
            "output_path": output_dir,
            "scenarios": list(all_results.keys()),
        }
        job.status = SweepStatus.completed
        job.elapsed_seconds = round(time.time() - t0, 1)

    except Exception as e:
        traceback.print_exc()
        job.status = SweepStatus.failed
        job.error = str(e)


@app.post("/api/sweep")
async def start_sweep(req: SweepRequest, background_tasks: BackgroundTasks):
    """Launch a full parametric sweep as a background task. Returns job ID."""
    job_id = str(uuid.uuid4())[:12]

    job = SweepJob(
        job_id=job_id,
        status=SweepStatus.pending,
        total_scenarios=1215,  # Updated when actual count is known
    )
    _sweep_jobs[job_id] = job

    background_tasks.add_task(_run_sweep_sync, job_id, req)

    return {"job_id": job_id, "status": "pending", "message": "Sweep started."}


@app.get("/api/sweep/{job_id}", response_model=SweepJob)
async def get_sweep_status(job_id: str):
    """Poll sweep job status and results."""
    if job_id not in _sweep_jobs:
        raise HTTPException(status_code=404, detail=f"Sweep job '{job_id}' not found.")
    return _sweep_jobs[job_id]


# ─────────────────────────────────────────────────────────────────────────────
# Cached sweep endpoints — read pre-computed 1215-scenario results
# ─────────────────────────────────────────────────────────────────────────────

SWEEP_CACHE_DIR = MARKET_SIM_ROOT / "results" / "sweep_1215"


def _sanitize_for_json(obj):
    """Recursively replace inf/NaN floats with None so json.dumps() succeeds."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
        return None
    return obj


@app.get("/api/sweep-cached/status")
async def sweep_cached_status():
    """Check whether pre-computed sweep results exist and return metadata."""
    agg_path = SWEEP_CACHE_DIR / "sweep_1215_aggregates.json"
    parquet_path = SWEEP_CACHE_DIR / "sweep_1215_flat.parquet"

    available = parquet_path.exists() or agg_path.exists()

    meta = {
        "available": available,
        "aggregates_exists": agg_path.exists(),
        "parquet_exists": parquet_path.exists(),
        "sweep_dimensions": {
            "demand_growth": list(DEMAND_GROWTH_LEVELS),
            "price_sensitivity": list(PRICE_SENSITIVITIES.keys()),
            "ppa_level": list(PPA_LEVELS),
            "gas_friction": list(GAS_FRICTION_LEVELS.keys()),
            "queue_cap": ["Low", "Medium", "High"],
            "new_fossil_cost_level": ["Low", "Medium", "High"],
        },
        "total_scenarios": 1215,
        "years": [2023, 2030, 2035, 2040, 2045, 2050],
        "isos": list(ISOS),
    }

    if parquet_path.exists():
        meta["parquet_size_mb"] = round(parquet_path.stat().st_size / 1e6, 1)
    if agg_path.exists():
        meta["aggregates_size_mb"] = round(agg_path.stat().st_size / 1e6, 1)

    return meta


@app.get("/api/sweep-cached/aggregates")
async def sweep_cached_aggregates(iso: str = None):
    """Return P10/P50/P90 aggregates from cached sweep. Optionally filter by ISO."""
    agg_path = SWEEP_CACHE_DIR / "sweep_1215_aggregates.json"
    if not agg_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Cached sweep aggregates not found. Run the 1,215-scenario sweep first "
                   "(GitHub Actions workflow 'Market Simulator: 1215-Scenario Sweep').",
        )

    with open(agg_path) as f:
        aggregates = json.load(f)

    # Sanitize any stringified inf/NaN values that slipped through default=str
    aggregates = _sanitize_for_json(aggregates)

    if iso:
        iso = iso.upper()
        if iso not in aggregates:
            raise HTTPException(status_code=404, detail=f"ISO '{iso}' not found in cached results.")
        return {iso: aggregates[iso]}

    return aggregates


@app.get("/api/sweep-cached/results")
async def sweep_cached_results(iso: str = None, scenario: str = None):
    """Return cached sweep results from parquet. Filter by ISO and/or scenario ID.

    Results are served from the flat parquet file (~10 MB) instead of JSON (~300 MB).
    """
    parquet_path = SWEEP_CACHE_DIR / "sweep_1215_flat.parquet"
    if not parquet_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Cached sweep results not found. Run the 1,215-scenario sweep first.",
        )

    df = pd.read_parquet(parquet_path)

    # Filter by scenario
    if scenario:
        df = df[df['scenario_id'] == scenario]
        if df.empty:
            raise HTTPException(status_code=404, detail=f"Scenario '{scenario}' not found.")

    # Filter by ISO
    if iso:
        iso = iso.upper()
        df = df[df['iso'] == iso]
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No results for ISO '{iso}'.")

    # Convert to records, then sanitize inf/NaN that survive pandas → dict conversion
    records = _sanitize_for_json(df.to_dict(orient='records'))

    return {
        "row_count": len(records),
        "results": records,
    }


@app.get("/api/sweep-cached/sensitivity")
async def sweep_cached_sensitivity(iso: str = None, year: int = 2050):
    """Return G7 sensitivity analysis (tornado data + Morris method) from cached sweep.

    Computes variance decomposition and Morris elementary effects on-the-fly from
    the cached parquet. Results include tornado_data suitable for embedding in
    SweepUncertainty responses.
    """
    parquet_path = SWEEP_CACHE_DIR / "sweep_1215_flat.parquet"
    if not parquet_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Cached sweep results not found. Run the 1,215-scenario sweep first.",
        )

    # Import sensitivity analysis module
    try:
        from sensitivity_analysis import (
            run_sensitivity_analysis,
            METRIC_LABELS,
        )
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="sensitivity_analysis module not found in scripts/.",
        )

    target_isos = [iso.upper()] if iso else None
    results = run_sensitivity_analysis(
        parquet_path=str(parquet_path),
        year=year,
        isos=target_isos,
    )

    if not results:
        raise HTTPException(status_code=404, detail="No sensitivity results for the given parameters.")

    # Build structured tornado_data per ISO
    response = {}
    for iso_key, iso_data in results.items():
        tornado_metrics = []
        var_decomp = iso_data.get('variance_decomposition', {})
        tornado_raw = iso_data.get('tornado', {})

        for metric, bars in tornado_raw.items():
            total_var = var_decomp.get(metric, {}).get('_total_variance', 0.0)
            tornado_metrics.append(
                TornadoMetric(
                    metric=metric,
                    metric_label=METRIC_LABELS.get(metric, metric),
                    bars=[TornadoBar(**b) for b in bars],
                    total_variance=total_var,
                ).model_dump()
            )

        response[iso_key] = {
            'tornado_data': tornado_metrics,
            'morris': iso_data.get('morris', {}),
            'morris_plot': iso_data.get('morris_plot', {}),
            'range_impact': iso_data.get('range_impact', {}),
            'metadata': iso_data.get('metadata', {}),
        }

    return _sanitize_for_json(response)


# ─────────────────────────────────────────────────────────────────────────────
# Sensitivity endpoint
# ─────────────────────────────────────────────────────────────────────────────

# Map of user-friendly param names → how to apply them to a SimulationRequest
_SENSITIVITY_PARAM_MAP = {
    "gas_price": lambda req, v: setattr(req.fuel_prices, "gas", v),
    "coal_price": lambda req, v: setattr(req.fuel_prices, "coal", v),
    "oil_price": lambda req, v: setattr(req.fuel_prices, "oil", v),
    "carbon_price": lambda req, v: setattr(req, "carbon_price", v),
    "solar_lcoe": lambda req, v: setattr(req.clean_lcoes, "solar", v),
    "wind_lcoe": lambda req, v: setattr(req.clean_lcoes, "wind", v),
    "offshore_wind_lcoe": lambda req, v: setattr(req.clean_lcoes, "offshore_wind", v),
    "nuclear_lcoe": lambda req, v: setattr(req.clean_lcoes, "nuclear", v),
    "ccs_lcoe": lambda req, v: setattr(req.clean_lcoes, "ccs_ccgt", v),
    "nuclear_retirement_threshold": lambda req, v: setattr(req, "nuclear_retirement_threshold", v),
    "capacity_market_price": lambda req, v: setattr(req, "capacity_market_price", v),
}


@app.post("/api/sensitivity", response_model=SensitivityResponse)
async def run_sensitivity(req: SensitivityRequest):
    """Vary a single parameter across a range and return results for each value."""
    if req.vary_param not in _SENSITIVITY_PARAM_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown parameter '{req.vary_param}'. "
                   f"Available: {', '.join(_SENSITIVITY_PARAM_MAP.keys())}",
        )

    iso = req.base_params.iso.upper()
    if iso not in ISOS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ISO '{iso}'. Available: {', '.join(ISOS)}",
        )

    apply_fn = _SENSITIVITY_PARAM_MAP[req.vary_param]
    preloaded = _get_preloaded_data()
    lmp_cache: dict = {}
    results: List[SensitivityResult] = []

    try:
        for val in req.values:
            # Deep copy the base params
            params = req.base_params.model_copy(deep=True)
            apply_fn(params, val)

            conditions = _map_request_to_conditions(params)
            nrt = params.nuclear_retirement_threshold if params.nuclear_retirement_threshold > 0 else None

            sim_results = run_market_simulation(
                scenario_id=f"SENS_{req.vary_param}_{val}",
                conditions=conditions,
                isos=[iso],
                nuclear_retirement_threshold=nrt,
                snapshot_mode=False,
                _preloaded=preloaded,
                _lmp_cache=lmp_cache,
                _quiet=True,
                _data_sources=preloaded.get('data_sources'),
            )

            iso_results = sim_results.get(iso, [])
            if iso_results:
                final = iso_results[-1]
                results.append(SensitivityResult(
                    param_value=val,
                    iso=iso,
                    clean_pct=final.get("clean_pct", 0),
                    avg_lmp=final.get("avg_lmp", 0),
                    emissions_mt=final.get("emissions_mt", 0),
                    cost_per_mwh=final.get("cost_per_mwh", 0),
                    revenue_per_mwh=final.get("revenue_per_mwh", 0),
                ))

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Sensitivity analysis failed: {str(e)}")

    return SensitivityResponse(
        vary_param=req.vary_param,
        base_iso=iso,
        results=results,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Correlated scenario analysis (IEA-aligned bundles)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/correlated-scenarios", response_model=CorrelatedScenarioResponse)
async def run_correlated(req: CorrelatedScenarioRequest):
    """Run IEA-aligned correlated scenario bundles for a single ISO.

    Unlike the independent Cartesian sweep, these scenarios represent
    internally-consistent macro futures where parameters move together
    as they would in reality (e.g., high gas prices + fast renewable
    cost decline in NZE pathway).
    """
    iso = req.iso.upper()
    if iso not in ISOS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ISO '{iso}'. Available: {', '.join(ISOS)}",
        )

    scenario_names = req.scenarios
    if scenario_names:
        invalid = [s for s in scenario_names if s not in CORRELATED_SCENARIOS]
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown scenario(s): {invalid}. "
                       f"Valid: {list(CORRELATED_SCENARIOS.keys())}",
            )

    try:
        raw = run_correlated_scenarios(
            iso=iso,
            scenario_names=scenario_names,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Correlated scenario analysis failed: {str(e)}",
        )

    provenance = raw.pop('_provenance', None)

    scenario_results = []
    for name, data in raw.items():
        iso_results = data.get('results', {}).get(iso, [])
        scenario_results.append(CorrelatedScenarioResult(
            scenario_name=name,
            scenario_id=data['scenario_id'],
            description=data['description'],
            parameters=data['parameters'],
            year_results={iso: iso_results},
        ))

    return CorrelatedScenarioResponse(
        iso=iso,
        scenarios=scenario_results,
        provenance=provenance,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Results directory management
# ─────────────────────────────────────────────────────────────────────────────

def _next_run_id(iso: str = "") -> str:
    """Determine the next run ID, including ISO name."""
    existing = [d.name for d in RESULTS_DIR.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not existing:
        num = 1
    else:
        nums = []
        for name in existing:
            match = re.match(r"run_(\d+)", name)
            if match:
                nums.append(int(match.group(1)))
        num = max(nums) + 1 if nums else 1
    iso_suffix = f"_{iso}" if iso else ""
    return f"run_{num:03d}{iso_suffix}"


def _generate_narrative(iso: str, response_data: dict, params: dict) -> str:
    """Generate a plain-English narrative interpretation of simulation results."""
    lines = []
    lines.append(f"MARKET SIMULATION RESULTS — {iso}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)
    lines.append("")

    # Market outcome
    clean_pct = response_data.get("market_outcome_clean_pct", 0)
    existing_clean = response_data.get("existing_clean_pct", 0)
    avg_lmp = response_data.get("avg_lmp", 0)
    emissions = response_data.get("emissions_mt", 0)
    demand = response_data.get("demand_twh", 0)

    lines.append("MARKET OUTCOME:")
    lines.append(f"Under the specified market conditions, {iso} achieves a {clean_pct:.1f}% clean energy mix,")
    delta = clean_pct - existing_clean
    direction = "up from" if delta > 0 else "down from" if delta < 0 else "unchanged from"
    lines.append(f"{direction} the existing {existing_clean:.1f}% baseline ({delta:+.1f} percentage points).")
    lines.append("")

    # LMP
    lines.append(f"ENERGY PRICES:")
    lines.append(f"Average LMP settles at ${avg_lmp:.1f}/MWh.")
    if avg_lmp < 30:
        lines.append("This is a low-price environment — marginal generators face revenue pressure.")
    elif avg_lmp > 60:
        lines.append("This is a high-price environment — fossil generators earn strong margins.")
    else:
        lines.append("This is a moderate price environment.")
    lines.append("")

    # Emissions
    if emissions > 0:
        lines.append(f"EMISSIONS:")
        lines.append(f"Annual CO2 emissions: {emissions:.1f} MT from {demand:.0f} TWh of demand.")
        intensity = emissions / demand * 1000 if demand > 0 else 0
        lines.append(f"Grid emission intensity: {intensity:.0f} kg CO2/MWh.")
        lines.append("")

    # Generator economics
    gen_econ = response_data.get("generator_economics", [])
    if gen_econ:
        retiring = [g for g in gen_econ if isinstance(g, dict) and g.get("status") == "retiring"]
        profitable = [g for g in gen_econ if isinstance(g, dict) and g.get("status") == "profitable"]
        lines.append("GENERATOR FLEET:")
        if retiring:
            ret_names = [g.get("unit_type", "unknown") for g in retiring]
            lines.append(f"Units facing retirement pressure: {', '.join(ret_names)}.")
        if profitable:
            prof_names = [g.get("unit_type", "unknown") for g in profitable]
            lines.append(f"Profitable units: {', '.join(prof_names)}.")
        lines.append("")

    # Nuclear
    nuc_rev = response_data.get("nuclear_revenue", {})
    if isinstance(nuc_rev, dict) and nuc_rev.get("total_mwh", 0) > 0:
        total_nuc = nuc_rev["total_mwh"]
        nrt = params.get("nuclear_retirement_threshold", 30)
        lines.append(f"NUCLEAR:")
        lines.append(f"Nuclear total revenue stack: ${total_nuc:.1f}/MWh (threshold: ${nrt}/MWh).")
        if total_nuc > nrt:
            lines.append("Nuclear fleet is economically viable under these conditions.")
        else:
            lines.append("WARNING: Nuclear revenue falls below retirement threshold — retirement risk.")
        lines.append("")

    # Key inputs summary
    lines.append("KEY INPUT ASSUMPTIONS:")
    fp = params.get("fuel_prices", {})
    lines.append(f"  Gas: ${fp.get('gas', 3.5)}/MMBtu | Coal: ${fp.get('coal', 2.25)}/MMBtu | Oil: ${fp.get('oil', 10.5)}/MMBtu")
    lines.append(f"  Carbon price: ${params.get('carbon_price', 5.5)}/ton CO2")
    lines.append(f"  Demand growth: {params.get('demand_growth', 'Medium')}")
    lines.append(f"  Transmission: {params.get('transmission_level', 'Medium')}")
    lines.append("")

    lines.append("NOTE: This is a screening-level analysis for directional guidance.")
    lines.append("For production-grade results, validate with IPM or GenX capacity models.")

    return "\n".join(lines)


def _save_run_artifacts(run_id: str, iso: str, response_data: dict, params: dict):
    """Save all result artifacts to the run directory."""
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(exist_ok=True)
    charts_dir = run_dir / "charts"
    charts_dir.mkdir(exist_ok=True)

    # 1. Inputs text file
    with open(run_dir / "inputs.txt", "w") as f:
        f.write(f"SIMULATION INPUT PARAMETERS — {run_id}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(json.dumps(params, indent=2, default=str))

    # 2. Narrative text file
    narrative = _generate_narrative(iso, response_data, params)
    with open(run_dir / "narrative.txt", "w") as f:
        f.write(narrative)

    # 3. Results data CSV
    _save_results_csv(run_dir / "results_data.csv", response_data)

    # 4. Input parameters CSV
    _save_params_csv(run_dir / "input_parameters.csv", params)

    # 5. Copy custom input files if used
    overrides = params.get("custom_overrides", {})
    if isinstance(overrides, dict):
        for key, enabled in overrides.items():
            if not enabled:
                continue
            file_map = {
                "fuel": ["fuel_prices_gas.csv", "fuel_prices_coal.csv", "fuel_prices_oil.csv"],
                "lmp": ["lmp_hourly.csv"],
                "capacity": ["capacity_prices.csv"],
                "rec": ["rec_prices.csv"],
            }
            for fname in file_map.get(key, []):
                src = CUSTOM_INPUTS_DIR / fname
                if src.exists():
                    shutil.copy2(src, run_dir / f"custom_{fname}")

    # 6. Plant-level detailed results CSV
    plant_data = response_data.get("_plant_level_data", [])
    if plant_data:
        _save_plant_level_csv(run_dir / "detailed_plant_results.csv", plant_data)

    return narrative


def _save_plant_level_csv(filepath: Path, plant_data: list):
    """Save detailed per-plant economics as CSV.

    Each row is tagged with a 'year' column so trajectory simulations
    output all years (not just the final year).
    """
    if not plant_data:
        return

    columns = [
        'year',
        'entity', 'plant_name', 'plant_id', 'generator_id', 'state', 'county',
        'latitude', 'longitude', 'capacity_mw', 'heat_rate_mmbtu_mwh',
        'fuel_type', 'prime_mover', 'online_year', 'age_years',
        'capacity_factor', 'mwh_generated', 'fuel_consumed_mmbtu',
        'co2_tons', 'nox_lbs', 'sox_lbs',
        'revenue_per_mwh', 'vom_per_mwh', 'fuel_cost_per_mwh', 'profit_per_mwh',
        'total_revenue_million', 'total_cost_million', 'total_profit_million',
        'status'
    ]

    with open(filepath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        for row in plant_data:
            w.writerow(row)


def _save_constellation_dispatch_csv(filepath: Path, dispatch_results: list):
    """Save constellation dispatch results as CSV for external analysis.

    Each row is one plant × one year with dispatch economics.
    """
    if not dispatch_results:
        return

    columns = [
        'year', 'orispl', 'plant_name', 'iso', 'capacity_mw', 'equity_pct',
        'capacity_factor', 'generation_mwh', 'co2_tons', 'co2_mmt',
        'ccs_residual_mmt', 'ccs_delta_mmt', 'revenue_mwh', 'fuel_cost_mwh',
        'profit_mwh', 'status',
    ]

    with open(filepath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        for row in dispatch_results:
            w.writerow(row)


def _save_results_csv(filepath: Path, response_data: dict):
    """Save comprehensive result data as CSV with multiple sections."""
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)

        # ── Section 1: Summary Metrics ──
        w.writerow(["=== Summary Metrics ==="])
        w.writerow(["metric", "value", "unit"])
        w.writerow(["iso", response_data.get("iso", ""), ""])
        w.writerow(["existing_clean_pct", response_data.get("existing_clean_pct", 0), "%"])
        w.writerow(["market_outcome_clean_pct", response_data.get("market_outcome_clean_pct", 0), "%"])
        w.writerow(["avg_lmp", response_data.get("avg_lmp", 0), "$/MWh"])
        w.writerow(["emissions_mt", response_data.get("emissions_mt", 0), "MT CO2"])
        w.writerow(["demand_twh", response_data.get("demand_twh", 0), "TWh"])
        w.writerow(["nuclear_revenue_mwh", response_data.get("nuclear_revenue_mwh", 0), "$/MWh"])
        ccs_be = response_data.get("ccs_breakeven_carbon_price", 0)
        w.writerow(["ccs_breakeven_carbon_price", ccs_be, "$/ton"])
        w.writerow([])

        # ── Section 2: Resource Mix ──
        rmix = response_data.get("resource_mix_twh", {})
        if rmix:
            w.writerow(["=== Resource Mix (TWh) ==="])
            w.writerow(["resource", "generation_twh", "pct_of_total"])
            total_twh = sum(rmix.values()) or 1
            for r, v in sorted(rmix.items(), key=lambda x: -x[1]):
                w.writerow([r, round(v, 1), round(v / total_twh * 100, 1)])
            w.writerow([])

        # ── Section 3: Generator Economics ──
        gen_econ = response_data.get("generator_economics", [])
        if gen_econ:
            w.writerow(["=== Generator Economics ==="])
            w.writerow(["unit_type", "capacity_mw", "marginal_cost_mwh",
                         "dispatch_hours", "capacity_factor",
                         "avg_revenue_mwh", "vom_mwh", "fuel_cost_mwh",
                         "profit_mwh", "status"])
            for g in gen_econ:
                if isinstance(g, dict):
                    w.writerow([
                        g.get("unit_type"), g.get("capacity_mw"),
                        g.get("marginal_cost"), g.get("dispatch_hours"),
                        g.get("capacity_factor"), g.get("avg_revenue_mwh"),
                        g.get("vom_mwh", ""), g.get("fuel_cost_mwh", ""),
                        g.get("profit_mwh"), g.get("status"),
                    ])
            w.writerow([])

        # ── Section 4: Supply Stack ──
        supply = response_data.get("supply_stack_summary", [])
        if supply:
            w.writerow(["=== Supply Stack ==="])
            w.writerow(["resource", "generation_twh"])
            for s in supply:
                if isinstance(s, dict):
                    w.writerow([s.get("resource"), s.get("generation_twh")])
            w.writerow([])

        # ── Section 5: Year-by-Year Trajectory ──
        years = response_data.get("year_results", [])
        if years:
            w.writerow(["=== Year-by-Year Trajectory ==="])
            w.writerow([
                "year", "clean_pct", "demand_twh", "emissions_mt",
                "emission_rate_tco2_mwh", "avg_lmp", "lmp_p90",
                "cost_per_mwh", "revenue_per_mwh",
                "energy_rev_mwh", "capacity_rev_mwh", "rec_rev_mwh",
                "gas_built_gw", "market_stop",
                "zones_deployed", "nuclear_retired",
                "rps_mandated_pct", "rps_eligible_pct", "rps_shortfall_pct",
                "acp_cost_million",
            ])
            for yr in years:
                if isinstance(yr, dict):
                    w.writerow([
                        yr.get("year"), yr.get("clean_pct"),
                        yr.get("demand_twh"), yr.get("emissions_mt"),
                        yr.get("emission_rate_tco2_mwh"), yr.get("avg_lmp"),
                        yr.get("lmp_p90"), yr.get("cost_per_mwh"),
                        yr.get("revenue_per_mwh"), yr.get("energy_rev_mwh"),
                        yr.get("capacity_rev_mwh"), yr.get("rec_rev_mwh"),
                        yr.get("gas_built_gw"), yr.get("market_stop"),
                        "|".join(str(z) for z in yr.get("zones_deployed", [])),
                        yr.get("nuclear_retired"),
                        yr.get("rps_mandated_pct"), yr.get("rps_eligible_pct"),
                        yr.get("rps_shortfall_pct"), yr.get("acp_cost_million"),
                    ])
            w.writerow([])

        # ── Section 5b: Year-by-Year Resource Mix ──
        if years:
            # Collect all resource types across all years
            all_resources = set()
            for yr in years:
                if isinstance(yr, dict):
                    all_resources.update(yr.get("resource_mix_twh", {}).keys())
            all_resources = sorted(all_resources)
            if all_resources:
                w.writerow(["=== Year-by-Year Resource Mix (TWh) ==="])
                w.writerow(["year"] + all_resources)
                for yr in years:
                    if isinstance(yr, dict):
                        rmix = yr.get("resource_mix_twh", {})
                        w.writerow([yr.get("year")] + [round(rmix.get(r, 0), 2) for r in all_resources])
                w.writerow([])

        # ── Section 5c: Year-by-Year Generator Economics ──
        if years:
            w.writerow(["=== Year-by-Year Generator Economics ==="])
            w.writerow(["year", "unit_type", "capacity_mw", "marginal_cost_mwh",
                         "dispatch_hours", "capacity_factor",
                         "avg_revenue_mwh", "vom_mwh", "fuel_cost_mwh",
                         "profit_mwh", "status"])
            for yr in years:
                if not isinstance(yr, dict):
                    continue
                yr_year = yr.get("year", "")
                # Use adjusted economics when available
                gen_raw = yr.get("adjusted_generator_economics", yr.get("generator_economics", {}))
                if isinstance(gen_raw, dict):
                    for unit_type, data in gen_raw.items():
                        if not isinstance(data, dict):
                            continue
                        w.writerow([
                            yr_year, unit_type,
                            data.get("capacity_mw"), data.get("marginal_cost"),
                            data.get("dispatch_hours"), data.get("capacity_factor"),
                            data.get("avg_revenue_mwh"), data.get("vom_mwh", ""),
                            data.get("fuel_cost_mwh", ""), data.get("profit_mwh"),
                            data.get("status"),
                        ])
                elif isinstance(gen_raw, list):
                    for g in gen_raw:
                        if not isinstance(g, dict):
                            continue
                        w.writerow([
                            yr_year, g.get("unit_type"),
                            g.get("capacity_mw"), g.get("marginal_cost"),
                            g.get("dispatch_hours"), g.get("capacity_factor"),
                            g.get("avg_revenue_mwh"), g.get("vom_mwh", ""),
                            g.get("fuel_cost_mwh", ""), g.get("profit_mwh"),
                            g.get("status"),
                        ])
            w.writerow([])

        # ── Section 6: Zone Economics Detail ──
        zone_details = response_data.get("zone_details", [])
        if zone_details:
            w.writerow(["=== Zone Economics Detail ==="])
            w.writerow(["threshold", "revenue_mwh", "cost_mwh", "profit_mwh",
                         "new_gw", "avg_lmp",
                         "energy_rev_mwh", "capacity_rev_mwh", "rec_rev_mwh"])
            for zd in zone_details:
                if isinstance(zd, dict):
                    w.writerow([
                        zd.get("threshold"), zd.get("revenue"),
                        zd.get("cost"), zd.get("profit"),
                        zd.get("new_gw"), zd.get("avg_lmp"),
                        zd.get("energy_rev_mwh"), zd.get("capacity_rev_mwh"),
                        zd.get("rec_rev_mwh"),
                    ])
            w.writerow([])

        # ── Section 7: Nuclear Revenue Breakdown ──
        nuc_rev = response_data.get("nuclear_revenue", {})
        if nuc_rev:
            w.writerow(["=== Nuclear Revenue Breakdown ==="])
            w.writerow(["component", "value_mwh"])
            for k, v in nuc_rev.items():
                w.writerow([k, v])
            w.writerow([])

        # ── Section 8: Plant-Level Summary ──
        plant_summary = response_data.get("plant_level_summary")
        if plant_summary:
            w.writerow(["=== Plant-Level Summary ==="])
            w.writerow(["status", "count"])
            for status in ["operating", "at_risk", "stranded", "total"]:
                w.writerow([status, plant_summary.get(status, 0)])


def _save_params_csv(filepath: Path, params: dict):
    """Save input parameters as CSV."""
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "value"])
        _flatten_dict(params, "", w)


def _flatten_dict(d: dict, prefix: str, writer):
    """Recursively flatten a dict into CSV rows."""
    for key, val in d.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(val, dict):
            _flatten_dict(val, full_key, writer)
        elif isinstance(val, list):
            writer.writerow([full_key, json.dumps(val)])
        else:
            writer.writerow([full_key, val])


# ─────────────────────────────────────────────────────────────────────────────
# Custom input file endpoints
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_FILE_MAP = {
    "fuel": {
        "files": ["fuel_prices_gas.csv", "fuel_prices_coal.csv", "fuel_prices_oil.csv"],
        "expected_rows": 12,       # legacy; time-series = n_years * 12 * n_zones
        "time_series_ok": True,    # accepts year×month×zone format
    },
    "lmp": {
        "files": ["lmp_hourly.csv"],
        "expected_rows": 8760,
        "time_series_ok": False,   # hourly file has its own format
    },
    "capacity": {
        "files": ["capacity_prices.csv"],
        "expected_rows": 12,
        "time_series_ok": True,
    },
    "rec": {
        "files": ["rec_prices.csv"],
        "expected_rows": 12,
        "time_series_ok": True,
    },
}

EXPECTED_ISO_COLS = {"CAISO", "ERCOT", "PJM", "NYISO", "NEISO", "MISO", "SPP"}


def _validate_custom_file(filepath: Path, expected_rows: int,
                          time_series_ok: bool = False) -> dict:
    """Validate a custom input CSV file.

    Supports two formats for monthly files (fuel, capacity, REC):
      - Legacy: 12 rows (month × 7 ISOs) — static prices for all years
      - Time-series: year × month × zone (optional) × 7 ISOs — annual projections

    The ``time_series_ok`` flag enables the expanded validation path.
    """
    if not filepath.exists():
        return {"found": False, "valid": False, "error": "File not found"}

    try:
        import pandas as pd

        # Try comma first, then tab delimiter
        try:
            df = pd.read_csv(filepath, sep=',')
            if len(df.columns) <= 2:
                df = pd.read_csv(filepath, sep='\t')
        except Exception:
            df = pd.read_csv(filepath, sep='\t')

        # Check ISO columns
        header_set = set(df.columns)
        missing_isos = EXPECTED_ISO_COLS - header_set
        if missing_isos:
            return {"found": True, "valid": False,
                    "error": f"Missing columns: {', '.join(sorted(missing_isos))}"}

        # Determine format: legacy (no 'year' col) vs time-series ('year' col present)
        has_year = 'year' in df.columns
        has_zone = 'zone' in df.columns
        is_time_series = has_year and time_series_ok

        if is_time_series:
            # --- Time-series validation ---
            # year must be integer-like
            if not pd.to_numeric(df['year'], errors='coerce').notna().all():
                return {"found": True, "valid": False,
                        "error": "Column 'year' contains non-numeric values"}
            df['year'] = df['year'].astype(int)

            # month must be 1-12
            if 'month' not in df.columns:
                return {"found": True, "valid": False,
                        "error": "Time-series format requires 'month' column"}
            months = df['month'].unique()
            if not set(months).issubset(set(range(1, 13))):
                return {"found": True, "valid": False,
                        "error": f"Month values must be 1-12, found: {sorted(months)}"}

            # Each year must have exactly 12 months (per zone if zones exist)
            if has_zone:
                # Fill blank zones with 'system' for grouping
                df['_zone'] = df['zone'].fillna('').replace('', 'system')
                zones = sorted(df['_zone'].unique())
                for yr in df['year'].unique():
                    for z in zones:
                        subset = df[(df['year'] == yr) & (df['_zone'] == z)]
                        if len(subset) != 12:
                            return {"found": True, "valid": False,
                                    "error": f"Year {yr}, zone '{z}': expected 12 months, "
                                             f"found {len(subset)}"}
                n_years = df['year'].nunique()
                n_zones = len(zones)
                expected_total = n_years * 12 * n_zones
            else:
                for yr in df['year'].unique():
                    subset = df[df['year'] == yr]
                    if len(subset) != 12:
                        return {"found": True, "valid": False,
                                "error": f"Year {yr}: expected 12 months, found {len(subset)}"}
                expected_total = df['year'].nunique() * 12

            if len(df) != expected_total:
                return {"found": True, "valid": False,
                        "error": f"Row count {len(df)} doesn't match "
                                 f"years×months×zones = {expected_total}"}

            fmt = "time-series"
            n_years_found = df['year'].nunique()
            year_range = f"{df['year'].min()}-{df['year'].max()}"
        else:
            # --- Legacy validation ---
            if len(df) != expected_rows:
                return {"found": True, "valid": False,
                        "error": f"Expected {expected_rows} rows, found {len(df)}"}
            fmt = "legacy"
            n_years_found = 1
            year_range = "static"

        # Check for NaN/blank values in ISO columns
        iso_cols = [c for c in df.columns if c in EXPECTED_ISO_COLS]
        if df[iso_cols].isnull().any().any():
            return {"found": True, "valid": False, "error": "Contains NaN/blank values"}

        # Check all ISO columns are numeric
        for col in iso_cols:
            if not pd.to_numeric(df[col], errors='coerce').notna().all():
                return {"found": True, "valid": False,
                        "error": f"Column '{col}' contains non-numeric values"}

        result = {"found": True, "valid": True, "rows": len(df), "format": fmt}
        if is_time_series:
            result["years"] = n_years_found
            result["year_range"] = year_range
            if has_zone:
                result["zones"] = zones
        return result

    except Exception as e:
        return {"found": True, "valid": False, "error": str(e)}


@app.get("/api/custom-input-status")
async def custom_input_status():
    """Check status of custom input files in the custom-user-inputs folder."""
    result = {}
    for category, config in CUSTOM_FILE_MAP.items():
        # Validate ALL files in the category (fuel has 3 files)
        files = config["files"]
        all_valid = True
        any_found = False
        file_results = []
        for fname in files:
            filepath = CUSTOM_INPUTS_DIR / fname
            file_result = _validate_custom_file(
                filepath, config["expected_rows"],
                time_series_ok=config.get("time_series_ok", False))
            file_results.append((fname, file_result))
            if file_result.get("found"):
                any_found = True
            if not file_result.get("valid"):
                all_valid = False

        if len(files) == 1:
            result[category] = file_results[0][1]
        else:
            # Multi-file category (fuel): aggregate results
            if not any_found:
                result[category] = {"found": False, "valid": False, "error": "No files found"}
            elif all_valid:
                agg = {"found": True, "valid": True,
                       "files_checked": len(files)}
                # Report time-series metadata from first valid file
                first_valid = file_results[0][1]
                agg["rows"] = first_valid.get("rows", config["expected_rows"])
                agg["format"] = first_valid.get("format", "legacy")
                if first_valid.get("year_range"):
                    agg["year_range"] = first_valid["year_range"]
                result[category] = agg
            else:
                # Report first invalid file
                for fname, fr in file_results:
                    if fr.get("found") and not fr.get("valid"):
                        result[category] = {"found": True, "valid": False,
                                            "error": f"{fname}: {fr['error']}"}
                        break
                else:
                    # Some files missing
                    missing = [fn for fn, fr in file_results if not fr.get("found")]
                    result[category] = {"found": True, "valid": False,
                                        "error": f"Missing files: {', '.join(missing)}"}
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Run management endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/runs")
async def list_runs():
    """List all saved simulation runs."""
    runs = []
    if RESULTS_DIR.exists():
        for d in sorted(RESULTS_DIR.iterdir()):
            if d.is_dir() and d.name.startswith("run_"):
                info = {"run_id": d.name}
                inputs_file = d / "inputs.txt"
                if inputs_file.exists():
                    info["created"] = datetime.fromtimestamp(inputs_file.stat().st_mtime).isoformat()
                runs.append(info)
    return {"runs": runs}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    """Get data from a specific run."""
    run_dir = RESULTS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    result = {"run_id": run_id}
    for fname in ["inputs.txt", "narrative.txt"]:
        fpath = run_dir / fname
        if fpath.exists():
            result[fname.replace(".txt", "")] = fpath.read_text()
    return result


import base64


@app.post("/api/runs/{run_id}/save-chart")
async def save_chart(run_id: str, request: Request):
    """Save a base64-encoded chart PNG from client-side html2canvas."""
    run_dir = RESULTS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    charts_dir = run_dir / "charts"
    charts_dir.mkdir(exist_ok=True)

    body = await request.json()
    chart_name = body.get("name", "chart")
    image_data = body.get("image", "")

    # Strip data URL prefix if present
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(image_data)
        filepath = charts_dir / f"{chart_name}.png"
        filepath.write_bytes(img_bytes)
        return {"status": "ok", "path": str(filepath)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to save chart: {str(e)}")


@app.get("/api/runs/{run_id}/plant-csv")
async def download_plant_csv(run_id: str):
    """Download the detailed plant-level results CSV for a given run."""
    run_dir = RESULTS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    csv_path = run_dir / "detailed_plant_results.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Plant-level CSV not available for this run")

    return FileResponse(
        path=str(csv_path),
        media_type="text/csv",
        filename=f"{run_id}_plant_results.csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Constellation CCS dispatch endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/constellation-dispatch")
async def constellation_dispatch(request: Request):
    """Run Constellation CCGT fleet dispatch using simulation year results.

    Expects JSON body with:
      - year_results: list of {year, clean_pct, avg_lmp, iso, ...}
      - fleet_overrides: optional dict {plant_id: status}
      - run_id: optional run ID for saving CSV output
    """
    body = await request.json()
    year_results = body.get("year_results", [])
    fleet_overrides = body.get("fleet_overrides", None)
    run_id = body.get("run_id", None)

    if not year_results:
        raise HTTPException(status_code=400, detail="No year_results provided")

    try:
        from constellation_dispatch_integrated import run_dispatch_from_sim_results

        # Convert year_results to plain dicts if they're Pydantic models
        yr_dicts = []
        for yr in year_results:
            if isinstance(yr, dict):
                yr_dicts.append(yr)
            else:
                yr_dicts.append(yr.dict() if hasattr(yr, 'dict') else dict(yr))

        result = run_dispatch_from_sim_results(yr_dicts, fleet_overrides=fleet_overrides)

        # Save CSV if run_id provided
        if run_id and result.get("csv_rows"):
            run_dir = RESULTS_DIR / run_id
            if run_dir.exists():
                csv_path = run_dir / "constellation_dispatch.csv"
                _save_constellation_dispatch_csv(csv_path, result["csv_rows"])

        return result

    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Constellation dispatch module not available: {e}")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Dispatch failed: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/data-status")
async def data_status():
    """Return multi-tier data availability per ISO.

    Response shape:
      { simple: {ISO: 'parquet'|'synthetic'},
        tiers:  {ISO: {resource_mix, zonal_config, interchange, fleet_data, dr_params}} }
    """
    return check_data_sources()


@app.get("/api/health")
async def health_check():
    """Basic health check."""
    return {
        "status": "ok",
        "isos": ISOS,
        "frontend_available": FRONTEND_DIR.exists(),
        "scripts_available": SCRIPTS_DIR.exists(),
    }
