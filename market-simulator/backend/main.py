"""
FastAPI backend for the Market Simulator.

Exposes REST endpoints for single simulations, parametric sweeps,
sensitivity analysis, and ISO metadata.  Serves the frontend static
files from ../frontend/.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# ─────────────────────────────────────────────────────────────────────────────
# Path setup — import the simulation engine from ../scripts/
# ─────────────────────────────────────────────────────────────────────────────

BACKEND_DIR = Path(__file__).resolve().parent
MARKET_SIM_ROOT = BACKEND_DIR.parent
SCRIPTS_DIR = MARKET_SIM_ROOT / "scripts"
FRONTEND_DIR = MARKET_SIM_ROOT / "frontend"

sys.path.insert(0, str(SCRIPTS_DIR))

from market_simulation import (
    run_market_simulation,
    run_full_sweep,
    build_single_scenario,
    build_market_scenarios,
    save_results,
    load_common_data,
    load_step3_data,
    load_egrid_baselines,
    EGRID_2023_CLEAN_PCT,
    EGRID_2023_LMP,
    GAS_FRICTION_LEVELS,
    CONDITIONS_BUNDLE,
    DEMAND_GROWTH_LEVELS,
    PRICE_SENSITIVITIES,
    PPA_LEVELS,
)
from lmp_engine import (
    HEAT_RATES,
    VOM,
    CO2_RATES,
    FUEL_PRICES,
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
)

from models import (
    SimulationRequest,
    SimulationResponse,
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
    FuelBinRow,
    HourlyProfile,
    SupplyStackEntry,
    ISOSummary,
    ISODefaults,
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
        step3_data = load_step3_data()
        egrid_baselines = load_egrid_baselines()
        _preloaded_data = {
            "demand_data": demand_data,
            "gen_profiles": gen_profiles,
            "emission_rates": emission_rates,
            "fossil_mix": fossil_mix,
            "step3_data": step3_data,
            "egrid_baselines": egrid_baselines,
        }
    return _preloaded_data


# ─────────────────────────────────────────────────────────────────────────────
# HTML page routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_setup_page():
    """Serve the setup / landing page."""
    setup_path = FRONTEND_DIR / "setup.html"
    if not setup_path.exists():
        raise HTTPException(status_code=404, detail="setup.html not found")
    return FileResponse(str(setup_path), media_type="text/html")


@app.get("/results", response_class=HTMLResponse)
async def serve_results_page():
    """Serve the results page."""
    results_path = FRONTEND_DIR / "results.html"
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="results.html not found")
    return FileResponse(str(results_path), media_type="text/html")


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

    # Map condition string to learning speed + queue type
    bundle = CONDITIONS_BUNDLE.get(req.condition, CONDITIONS_BUNDLE["Facilitating"])

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

    return {
        "name": f"API: {req.iso} | {req.condition} | {req.demand_growth} demand | carbon=${req.carbon_price}",
        "demand_growth": req.demand_growth,
        "lcoe_level": lcoe_level,
        "learning_speed": bundle["learning_speed"],
        "queue_type": bundle["queue_type"],
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
        # Custom LCOE overrides
        "custom_lcoes": {
            'solar': req.clean_lcoes.solar,
            'wind': req.clean_lcoes.wind,
            'offshore_wind': req.clean_lcoes.offshore_wind,
            'nuclear': req.clean_lcoes.nuclear,
            'ccs_ccgt': req.clean_lcoes.ccs_ccgt,
        } if req.clean_lcoes else None,
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
        zone_details.append(ZoneDetail(
            threshold=zd.get("threshold", 0),
            revenue=round(zd.get("revenue", 0), 2),
            cost=round(zd.get("cost", 0), 2),
            profit=round(zd.get("profit", 0), 2),
            new_gw=round(zd.get("new_gw", 0), 2),
            avg_lmp=round(zd.get("avg_lmp", 0), 1),
            energy_rev_mwh=zd.get("energy_rev_mwh", 0),
            capacity_rev_mwh=zd.get("capacity_rev_mwh", 0),
            rec_rev_mwh=zd.get("rec_rev_mwh", 0),
        ))

    # CCS breakeven
    ccs_be = final.get("ccs_breakeven", {})
    ccs_be_price = ccs_be.get("carbon_price_breakeven", 0) if isinstance(ccs_be, dict) else 0

    # Typed year results
    typed_years = []
    for yr in year_results:
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
            total_gas_gw=yr.get("total_gas_gw", 0),
            market_stop=yr.get("market_stop", False),
            resource_mix_twh=yr.get("resource_mix_twh", {}),
            cumulative_gw=yr.get("cumulative_gw", {}),
            zones_deployed=yr.get("zones_deployed", []),
            zone_details=[
                ZoneDetail(**zd) if isinstance(zd, dict) else zd
                for zd in yr.get("zone_details", [])
            ],
            generator_economics=yr.get("generator_economics", {}),
            nuclear_revenue=yr.get("nuclear_revenue", {}),
            nuclear_retired=yr.get("nuclear_retired", False),
            ccs_breakeven=yr.get("ccs_breakeven", {}),
        ))

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
    fuel_bins = []
    gen_raw = final.get("generator_economics", {})
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
        year_results=typed_years,
        zones_deployed=zone_details,
        lmp_time_series=lmp_ts,
        capacity_rev_time_series=cap_rev_ts,
        supply_stack_summary=supply_stack,
        fuel_bin_table=fuel_bins,
    )


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
        snapshot_mode = req.mode == "snapshot"

        # Determine nuclear retirement threshold
        nrt = req.nuclear_retirement_threshold if req.nuclear_retirement_threshold > 0 else None

        preloaded = _get_preloaded_data()
        lmp_cache: dict = {}

        results = run_market_simulation(
            scenario_id="API_SINGLE",
            conditions=conditions,
            isos=[iso],
            nuclear_retirement_threshold=nrt,
            snapshot_mode=snapshot_mode,
            _preloaded=preloaded,
            _lmp_cache=lmp_cache,
            _quiet=True,
        )

        iso_results = results.get(iso, [])
        return _build_simulation_response(iso, iso_results)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")


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

        # Filter if user specified subsets
        if req.conditions:
            cond_set = set(req.conditions)
            all_scenarios = [
                (sid, c) for sid, c in all_scenarios
                if any(k in sid for k in ("_F_" if "Facilitating" in cond_set else "",
                                           "_C_" if "Challenging" in cond_set else ""))
            ]

        job.total_scenarios = len(all_scenarios)

        preloaded = _get_preloaded_data()
        lmp_cache: dict = {}
        all_results = {}

        for i, (scenario_id, conditions) in enumerate(all_scenarios):
            results = run_market_simulation(
                scenario_id=scenario_id,
                conditions=conditions,
                isos=isos,
                nuclear_retirement_threshold=req.nuclear_retirement_threshold,
                snapshot_mode=req.snapshot_mode,
                _preloaded=preloaded,
                _lmp_cache=lmp_cache,
                _quiet=True,
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
        total_scenarios=270,  # Updated when actual count is known
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
            snapshot_mode = params.mode == "snapshot"
            nrt = params.nuclear_retirement_threshold if params.nuclear_retirement_threshold > 0 else None

            sim_results = run_market_simulation(
                scenario_id=f"SENS_{req.vary_param}_{val}",
                conditions=conditions,
                isos=[iso],
                nuclear_retirement_threshold=nrt,
                snapshot_mode=snapshot_mode,
                _preloaded=preloaded,
                _lmp_cache=lmp_cache,
                _quiet=True,
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
# Health check
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    """Basic health check."""
    return {
        "status": "ok",
        "isos": ISOS,
        "frontend_available": FRONTEND_DIR.exists(),
        "scripts_available": SCRIPTS_DIR.exists(),
    }
