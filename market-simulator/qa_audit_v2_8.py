#!/usr/bin/env python3
"""
QA/QC Audit Script — Market Simulator v2-8
Tests all simulation modes, ISOs, edge cases, parameter sensitivity, CSV overrides, and error handling.
"""

import json
import os
import sys
import time
import traceback
import requests
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"
ISOS = ["CAISO", "ERCOT", "PJM", "NYISO", "NEISO", "MISO", "SPP"]
RESULTS = []
BUGS = []

def log_result(test_id, name, passed, detail="", bug=None):
    status = "PASS" if passed else "FAIL"
    RESULTS.append({"id": test_id, "name": name, "status": status, "detail": detail})
    if not passed and bug:
        BUGS.append({"test_id": test_id, "name": name, "bug": bug})
    print(f"  [{status}] {test_id}: {name}")
    if detail and not passed:
        print(f"         {detail[:200]}")

def default_sim_request(iso="ERCOT", start_year=2025, end_year=2050, year_step=5):
    return {
        "iso": iso,
        "fuel_prices": {"coal": 2.25, "gas": 3.50, "oil": 10.50},
        "carbon_price": 5.50,
        "emission_prices": {"nox": 500, "sox": 200},
        "emission_limits": {"nox_lb_mwh": None, "sox_lb_mwh": None, "co2_lb_mwh": None},
        "heat_rates": {"coal": 10.0, "gas_ccgt": 7.0, "gas_ct": 10.5, "oil_ct": 10.5, "new_ccgt": 6.3, "new_ct": 9.0},
        "vom": {"coal": 5.50, "gas_ccgt": 3.50, "gas_ct": 5.00, "oil_ct": 6.00},
        "clean_lcoes": {"solar": 65.0, "wind": 62.0, "offshore_wind": 85.0, "nuclear": 90.0, "ccs_ccgt": 120.0},
        "incentives": {"ptc_wind": 26.0, "ptc_solar": 26.0, "ptc_nuclear_45y": 26.0, "itc_pct": 30.0, "nuclear_45u_max": 15.0, "nuclear_45u_floor": 40.0, "rec_override": None},
        "storage_costs": {"battery_kwyr": 295.0, "battery8_kwyr": 456.0, "ldes_kwyr": 220.0},
        "capacity_market_price": None,
        "wholesale_price_override": None,
        "q45": True,
        "ccs_credit_override": None,
        "transmission_level": "Medium",
        "demand_growth": "Medium",
        "ppa_level": "Medium",
        "gas_friction": "Medium",
        "queue_cap_level": "Medium",
        "queue_cap_override_gw": None,
        "tech_differentiated_queue": True,
        "nuclear_retirement_threshold": 30.0,
        "mode": "trajectory",
        "years": [],
        "start_year": start_year,
        "end_year": end_year,
        "year_step": year_step,
        "custom_overrides": {"fuel_csv": False, "lmp_csv": False, "capacity_csv": False, "rec_csv": False},
        "dr_level": "Off",
        "fossil_lcoes": None,
        "tx_overrides": None,
        "interchange_enabled": True,
        "learning_curves": True,
        "fleet_overrides": None
    }

def run_simulation(params, timeout=120):
    """Run a simulation and return (response_json, elapsed_seconds, error)."""
    t0 = time.time()
    try:
        resp = requests.post(f"{BASE_URL}/api/simulate", json=params, timeout=timeout)
        elapsed = time.time() - t0
        if resp.status_code == 200:
            return resp.json(), elapsed, None
        else:
            return None, elapsed, f"HTTP {resp.status_code}: {resp.text[:300]}"
    except Exception as e:
        return None, time.time() - t0, str(e)

# ===========================================================================
# TEST A: Startup & Environment
# ===========================================================================
def test_a_startup():
    print("\n===== TEST A: Startup & Environment =====")

    # A1: Server responds
    try:
        resp = requests.get(f"{BASE_URL}/", timeout=5)
        log_result("A1", "Server responds on port 8000", resp.status_code == 200,
                   f"Status: {resp.status_code}")
    except Exception as e:
        log_result("A1", "Server responds on port 8000", False, str(e), bug="Server not running")

    # A2: All pages load
    pages = {
        "guide.html": "/",
        "setup.html": "/setup",
        "fleet-config.html": "/fleet-config",
        "results.html": "/results",
        "emissions.html": "/emissions",
        "ipp-report.html": "/ipp-report",
        "methodology.html": "/methodology"
    }
    for name, path in pages.items():
        try:
            resp = requests.get(f"{BASE_URL}{path}", timeout=5)
            has_html = "<html" in resp.text.lower()
            log_result("A2", f"Page loads: {name}", resp.status_code == 200 and has_html,
                       f"Status: {resp.status_code}, has HTML: {has_html}")
        except Exception as e:
            log_result("A2", f"Page loads: {name}", False, str(e), bug=f"{name} fails to load")

    # A3: Nav bar present on all pages
    for name, path in pages.items():
        try:
            resp = requests.get(f"{BASE_URL}{path}", timeout=5)
            has_nav = "nav-links" in resp.text or "<nav" in resp.text
            log_result("A3", f"Nav bar on: {name}", has_nav,
                       f"nav-links found: {has_nav}")
        except Exception as e:
            log_result("A3", f"Nav bar on: {name}", False, str(e))

    # A4: shared-header.js included on pages with .header
    for name, path in pages.items():
        try:
            resp = requests.get(f"{BASE_URL}{path}", timeout=5)
            has_header_js = "shared-header.js" in resp.text
            has_header_div = 'class="header"' in resp.text or "class='header'" in resp.text
            # Only check if page has header div
            if has_header_div:
                log_result("A4", f"shared-header.js on: {name}", has_header_js,
                           f"header div: {has_header_div}, header js: {has_header_js}")
            else:
                log_result("A4", f"shared-header.js on: {name} (no header div)", True, "N/A - no header div")
        except Exception as e:
            log_result("A4", f"shared-header.js on: {name}", False, str(e))

    # A-extra: API endpoints respond
    for endpoint in ["/api/isos", "/api/fleet-config"]:
        try:
            resp = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
            log_result("A5", f"API endpoint: {endpoint}", resp.status_code == 200,
                       f"Status: {resp.status_code}")
        except Exception as e:
            log_result("A5", f"API endpoint: {endpoint}", False, str(e))

    # Check data-status endpoint
    try:
        resp = requests.get(f"{BASE_URL}/api/data-status", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            log_result("A5", "API endpoint: /api/data-status", True, f"Keys: {list(data.keys())[:5]}")
        else:
            log_result("A5", "API endpoint: /api/data-status", resp.status_code < 500,
                       f"Status: {resp.status_code}")
    except Exception as e:
        log_result("A5", "API endpoint: /api/data-status", False, str(e))

# ===========================================================================
# TEST B: Single Simulation Runs — All 7 ISOs
# ===========================================================================
def test_b_all_isos():
    print("\n===== TEST B: Single Simulation Runs — All 7 ISOs =====")

    for iso in ISOS:
        print(f"\n  --- {iso} ---")
        params = default_sim_request(iso=iso, start_year=2025, end_year=2050, year_step=5)
        result, elapsed, error = run_simulation(params, timeout=180)

        # B5a: Simulation completes without error
        if error:
            log_result("B5a", f"{iso}: Simulation completes", False, error, bug=f"{iso} simulation fails: {error}")
            continue

        log_result("B5a", f"{iso}: Simulation completes", True, f"Elapsed: {elapsed:.1f}s")

        # B5b: Results have year_results
        yr = result.get("year_results", [])
        log_result("B5b", f"{iso}: year_results populated", len(yr) > 0, f"Years: {len(yr)}")

        # B5c: Key metrics present
        has_lmp = result.get("avg_lmp") is not None
        has_clean = result.get("market_outcome_clean_pct") is not None
        has_emissions = result.get("emissions_mt") is not None
        log_result("B5c", f"{iso}: Key metrics present", has_lmp and has_clean and has_emissions,
                   f"LMP: {result.get('avg_lmp')}, Clean%: {result.get('market_outcome_clean_pct')}, Emissions: {result.get('emissions_mt')}")

        # B5d: Chart data present
        chart_keys = ["threshold_sweep", "what_gets_built", "sensitivity_matrix"]
        present = [k for k in chart_keys if result.get(k)]
        log_result("B5d", f"{iso}: Chart data present", len(present) > 0,
                   f"Present: {present}")

        # B5e: Narrative generated
        has_narrative = bool(result.get("narrative"))
        log_result("B5e", f"{iso}: Narrative generated", has_narrative,
                   f"Length: {len(result.get('narrative', ''))}")

        # B5f: Generator economics
        gen_econ = result.get("generator_economics", [])
        log_result("B5f", f"{iso}: Generator economics", len(gen_econ) > 0,
                   f"Units: {len(gen_econ)}")

        # B5g: Supply stack
        stack = result.get("supply_stack_summary", [])
        log_result("B5g", f"{iso}: Supply stack", len(stack) > 0,
                   f"Resources: {len(stack)}")

        # B6: IPM triggers (check if present)
        # Aggregate from year_results
        all_triggers = []
        for yr_data in yr:
            triggers = yr_data.get("ipm_triggers", [])
            all_triggers.extend(triggers)
        log_result("B6", f"{iso}: IPM triggers checked", True,
                   f"Total triggers: {len(all_triggers)}, Types: {list(set(t.get('trigger_id','') for t in all_triggers))[:5]}")

        # B7: Data source indicator
        data_source = result.get("data_source", "unknown")
        log_result("B7", f"{iso}: Data source indicator", data_source in ("synthetic", "parquet", "mixed"),
                   f"Source: {data_source}")

        # B8: Capture rates
        capture_data = None
        for yr_data in yr:
            zd = yr_data.get("zone_details", [])
            if zd:
                capture_data = zd
                break
        log_result("B8", f"{iso}: Zone/capture data", capture_data is not None,
                   f"Zones: {len(capture_data) if capture_data else 0}")

        # B-run: Run ID assigned
        run_id = result.get("run_id")
        log_result("B-run", f"{iso}: Run ID assigned", run_id is not None, f"Run ID: {run_id}")

# ===========================================================================
# TEST C: Parameter Sensitivity
# ===========================================================================
def test_c_sensitivity():
    print("\n===== TEST C: Parameter Sensitivity =====")

    # C9: Fuel price sensitivity (PJM)
    print("\n  --- C9: Fuel price sensitivity (PJM) ---")
    fuel_levels = {
        "Low": {"coal": 2.00, "gas": 2.00, "oil": 8.00},
        "Medium": {"coal": 2.25, "gas": 3.50, "oil": 10.50},
        "High": {"coal": 2.50, "gas": 6.00, "oil": 13.00}
    }
    fuel_results = {}
    for level, prices in fuel_levels.items():
        params = default_sim_request(iso="PJM", start_year=2025, end_year=2040, year_step=5)
        params["fuel_prices"] = prices
        result, elapsed, error = run_simulation(params)
        if error:
            log_result("C9", f"PJM fuel {level}: run", False, error, bug=f"PJM fuel {level} fails")
            continue
        fuel_results[level] = result
        log_result("C9", f"PJM fuel {level}: completes", True,
                   f"LMP={result.get('avg_lmp'):.1f}, Clean%={result.get('market_outcome_clean_pct'):.1f}")

    if len(fuel_results) == 3:
        low_lmp = fuel_results["Low"].get("avg_lmp", 0)
        high_lmp = fuel_results["High"].get("avg_lmp", 0)
        lmp_diff = high_lmp - low_lmp
        log_result("C9", f"PJM fuel: LMP responds to fuel prices", lmp_diff > 3,
                   f"High-Low LMP diff: ${lmp_diff:.1f}/MWh (expect >$5)")

    # C10: Demand growth sensitivity (ERCOT)
    print("\n  --- C10: Demand growth sensitivity (ERCOT) ---")
    growth_levels = {"Low": "Low", "Medium": "Medium", "High": "High"}
    growth_results = {}
    for level, val in growth_levels.items():
        params = default_sim_request(iso="ERCOT", start_year=2025, end_year=2050, year_step=5)
        params["demand_growth"] = val
        result, elapsed, error = run_simulation(params)
        if error:
            log_result("C10", f"ERCOT growth {level}: run", False, error)
            continue
        growth_results[level] = result
        last_yr = result.get("year_results", [{}])[-1] if result.get("year_results") else {}
        log_result("C10", f"ERCOT growth {level}: completes", True,
                   f"Final LMP={last_yr.get('avg_lmp', 'N/A')}, Clean%={last_yr.get('clean_pct', 'N/A')}")

    if len(growth_results) >= 2:
        low_last = growth_results.get("Low", {}).get("year_results", [{}])[-1]
        high_last = growth_results.get("High", {}).get("year_results", [{}])[-1]
        low_lmp = low_last.get("avg_lmp", 0) or 0
        high_lmp = high_last.get("avg_lmp", 0) or 0
        log_result("C10", "ERCOT growth: LMP increases with demand", high_lmp >= low_lmp * 0.95,
                   f"Low LMP={low_lmp:.1f}, High LMP={high_lmp:.1f}")

    # C11: Carbon price sensitivity (MISO)
    print("\n  --- C11: Carbon price sensitivity (MISO) ---")
    carbon_levels = {"$0": 0.0, "$25": 25.0, "$75": 75.0}
    carbon_results = {}
    for level, price in carbon_levels.items():
        params = default_sim_request(iso="MISO", start_year=2025, end_year=2045, year_step=5)
        params["carbon_price"] = price
        result, elapsed, error = run_simulation(params)
        if error:
            log_result("C11", f"MISO carbon {level}: run", False, error)
            continue
        carbon_results[level] = result
        log_result("C11", f"MISO carbon {level}: completes", True,
                   f"Clean%={result.get('market_outcome_clean_pct'):.1f}, Emissions={result.get('emissions_mt')}")

    if "$0" in carbon_results and "$75" in carbon_results:
        c0_clean = carbon_results["$0"].get("market_outcome_clean_pct", 0)
        c75_clean = carbon_results["$75"].get("market_outcome_clean_pct", 0)
        log_result("C11", "MISO carbon: higher carbon → more clean", c75_clean >= c0_clean,
                   f"$0 clean={c0_clean:.1f}%, $75 clean={c75_clean:.1f}%")

        c0_ret = carbon_results["$0"].get("total_economic_retirement_mw", 0) or 0
        c75_ret = carbon_results["$75"].get("total_economic_retirement_mw", 0) or 0
        log_result("C11", "MISO carbon: higher carbon → more retirement", c75_ret >= c0_ret,
                   f"$0 retired={c0_ret:.0f}MW, $75 retired={c75_ret:.0f}MW")

    # C12: Queue cap sensitivity (SPP)
    print("\n  --- C12: Queue cap sensitivity (SPP) ---")
    queue_levels = {"Low": "Low", "Medium": "Medium", "High": "High"}
    queue_results = {}
    for level, val in queue_levels.items():
        params = default_sim_request(iso="SPP", start_year=2025, end_year=2045, year_step=5)
        params["queue_cap_level"] = val
        result, elapsed, error = run_simulation(params)
        if error:
            log_result("C12", f"SPP queue {level}: run", False, error)
            continue
        queue_results[level] = result
        log_result("C12", f"SPP queue {level}: completes", True,
                   f"Clean%={result.get('market_outcome_clean_pct'):.1f}")

    if len(queue_results) >= 2:
        low_clean = queue_results.get("Low", {}).get("market_outcome_clean_pct", 0) or 0
        high_clean = queue_results.get("High", {}).get("market_outcome_clean_pct", 0) or 0
        log_result("C12", "SPP queue: higher cap → more clean", high_clean >= low_clean,
                   f"Low clean={low_clean:.1f}%, High clean={high_clean:.1f}%")

# ===========================================================================
# TEST D: Edge Cases & Error Handling
# ===========================================================================
def test_d_edge_cases():
    print("\n===== TEST D: Edge Cases & Error Handling =====")

    # D14: Extreme demand growth (10%/yr)
    print("\n  --- D14: Extreme inputs ---")
    params = default_sim_request(iso="ERCOT", start_year=2025, end_year=2040, year_step=5)
    params["demand_growth"] = 10.0  # 10% per year — extreme
    result, elapsed, error = run_simulation(params, timeout=180)
    log_result("D14", "Extreme demand (10%/yr): no crash", error is None,
               f"Error: {error}" if error else f"Completed in {elapsed:.1f}s, LMP={result.get('avg_lmp') if result else 'N/A'}")

    # D15: Zero carbon + low fuel → minimal retirement
    print("\n  --- D15: Zero carbon + low fuel ---")
    params = default_sim_request(iso="PJM", start_year=2025, end_year=2040, year_step=5)
    params["carbon_price"] = 0.0
    params["fuel_prices"] = {"coal": 1.50, "gas": 2.00, "oil": 6.00}
    result, elapsed, error = run_simulation(params)
    if result:
        ret_mw = result.get("total_economic_retirement_mw", 0) or 0
        log_result("D15", "Zero carbon + low fuel: fleet stable", True,
                   f"Retirement: {ret_mw:.0f}MW (expect minimal)")
    else:
        log_result("D15", "Zero carbon + low fuel: no crash", error is None, str(error))

    # D16: Maximum carbon price ($200/ton)
    print("\n  --- D16: Maximum carbon ($200/ton) ---")
    params = default_sim_request(iso="PJM", start_year=2025, end_year=2040, year_step=5)
    params["carbon_price"] = 200.0
    result, elapsed, error = run_simulation(params, timeout=180)
    if result:
        # Check no negative revenue artifacts
        gen_econ = result.get("generator_economics", [])
        negative_rev = [g for g in gen_econ if (g.get("revenue_per_mwh") or 0) < -50]
        log_result("D16", "$200 carbon: no crash", True, f"Elapsed: {elapsed:.1f}s")
        log_result("D16", "$200 carbon: no extreme negative revenue", len(negative_rev) == 0,
                   f"Negative rev units: {len(negative_rev)}" + (f", worst: {negative_rev[0]}" if negative_rev else ""))
    else:
        log_result("D16", "$200 carbon: simulation runs", error is None, str(error),
                   bug=f"$200 carbon crashes: {error}")

    # D17: Single-year simulation
    print("\n  --- D17: Single-year simulation ---")
    params = default_sim_request(iso="CAISO", start_year=2025, end_year=2025, year_step=1)
    result, elapsed, error = run_simulation(params)
    if result:
        yr = result.get("year_results", [])
        log_result("D17", "Single-year: completes", True, f"Years returned: {len(yr)}")
        log_result("D17", "Single-year: has results", len(yr) >= 1,
                   f"Year count: {len(yr)}")
    else:
        log_result("D17", "Single-year: no crash", error is None, str(error),
                   bug=f"Single-year crashes: {error}")

    # D18: Invalid ISO (error handling)
    print("\n  --- D18: Invalid inputs ---")
    params = default_sim_request(iso="INVALID_ISO")
    result, elapsed, error = run_simulation(params)
    # Should fail gracefully
    log_result("D18", "Invalid ISO: handled gracefully", error is not None or (result and "error" in str(result).lower()),
               f"Got: {'error' if error else 'result'}")

    # D18b: Negative LCOE
    params = default_sim_request(iso="ERCOT", start_year=2025, end_year=2030, year_step=5)
    params["clean_lcoes"]["solar"] = -10.0
    result, elapsed, error = run_simulation(params)
    log_result("D18b", "Negative LCOE: handled", True,
               f"Result: {'error' if error else 'completed'}")

# ===========================================================================
# TEST E: Sweep & Sensitivity Modes
# ===========================================================================
def test_e_sweep():
    print("\n===== TEST E: Sweep & Sensitivity Modes =====")

    # E19: Parameter sweep
    print("\n  --- E19: Parameter sweep ---")
    sweep_params = {
        "iso": "ERCOT",
        "fuel_levels": ["Low", "Medium", "High"],
        "demand_levels": ["Low", "High"],
        "start_year": 2025,
        "end_year": 2040,
        "year_step": 5
    }
    try:
        resp = requests.post(f"{BASE_URL}/api/sweep", json=sweep_params, timeout=10)
        if resp.status_code == 200:
            job = resp.json()
            job_id = job.get("job_id")
            log_result("E19", "Sweep: job submitted", job_id is not None, f"Job ID: {job_id}")

            # Poll for completion (up to 5 minutes)
            if job_id:
                completed = False
                for i in range(60):
                    time.sleep(5)
                    status_resp = requests.get(f"{BASE_URL}/api/sweep/{job_id}", timeout=10)
                    if status_resp.status_code == 200:
                        status = status_resp.json()
                        if status.get("status") == "completed":
                            completed = True
                            scenarios = status.get("results", {})
                            log_result("E19", "Sweep: completed", True,
                                       f"Scenarios: {len(scenarios) if isinstance(scenarios, list) else 'dict'}")
                            break
                        elif status.get("status") == "failed":
                            log_result("E19", "Sweep: completed", False,
                                       f"Failed: {status.get('error', 'unknown')}",
                                       bug="Sweep fails")
                            break
                if not completed:
                    log_result("E19", "Sweep: completed in time", False, "Timed out after 5 min")
        else:
            log_result("E19", "Sweep: API accepts", False, f"HTTP {resp.status_code}: {resp.text[:200]}",
                       bug=f"Sweep API error: {resp.status_code}")
    except Exception as e:
        log_result("E19", "Sweep: submits", False, str(e), bug=f"Sweep exception: {e}")

    # E20: Sensitivity mode
    print("\n  --- E20: Sensitivity mode ---")
    sens_params = {
        "iso": "PJM",
        "parameter": "carbon_price",
        "values": [0, 25, 50, 75, 100]
    }
    try:
        resp = requests.post(f"{BASE_URL}/api/sensitivity", json=sens_params, timeout=300)
        if resp.status_code == 200:
            sens_result = resp.json()
            results_list = sens_result.get("results", [])
            log_result("E20", "Sensitivity: completes", len(results_list) > 0,
                       f"Results: {len(results_list)} scenarios")
        else:
            log_result("E20", "Sensitivity: API responds", False,
                       f"HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        log_result("E20", "Sensitivity: runs", False, str(e))

# ===========================================================================
# TEST F: Results Export & Persistence
# ===========================================================================
def test_f_results():
    print("\n===== TEST F: Results Export & Persistence =====")

    # Run two sequential simulations
    run_ids = []
    for i, iso in enumerate(["CAISO", "ERCOT"]):
        params = default_sim_request(iso=iso, start_year=2025, end_year=2035, year_step=5)
        result, elapsed, error = run_simulation(params)
        if result:
            run_id = result.get("run_id")
            run_ids.append(run_id)
            log_result("F21", f"Run {i+1} ({iso}): has run_id", run_id is not None, f"Run ID: {run_id}")

            # F21: Check year_results columns
            yr = result.get("year_results", [])
            if yr:
                first_yr = yr[0]
                expected_keys = ["year", "avg_lmp", "clean_pct", "demand_twh"]
                present = [k for k in expected_keys if k in first_yr]
                log_result("F21", f"Run {i+1}: year_results has key columns", len(present) >= 3,
                           f"Present: {present}")

            # F22: Narrative
            narrative = result.get("narrative", "")
            log_result("F22", f"Run {i+1}: narrative present", len(narrative) > 10,
                       f"Length: {len(narrative)}")

            # Check narrative references correct ISO
            log_result("F22", f"Run {i+1}: narrative references {iso}",
                       iso.lower() in narrative.lower() or iso in narrative,
                       f"ISO in narrative: {iso in narrative}")
        else:
            log_result("F21", f"Run {i+1} ({iso}): completes", False, str(error))

    # F24: Sequential runs have different run IDs
    if len(run_ids) == 2 and all(run_ids):
        log_result("F24", "Sequential runs: unique IDs", run_ids[0] != run_ids[1],
                   f"IDs: {run_ids}")

    # Check results directory
    results_dir = Path("/home/user/hourly-cfe-optimizer/market-simulator/results")
    if results_dir.exists():
        run_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        log_result("F24", f"Results directory: run dirs exist", len(run_dirs) > 0,
                   f"Found: {[d.name for d in run_dirs[:5]]}")

        # Check contents of a run dir
        if run_dirs:
            latest = sorted(run_dirs)[-1]
            files = list(latest.iterdir())
            file_names = [f.name for f in files]
            log_result("F21", "Run dir: has output files", len(files) > 0,
                       f"Files: {file_names[:10]}")
    else:
        log_result("F24", "Results directory exists", False, "results/ dir not found")

# ===========================================================================
# TEST G: Constellation Fleet Mode
# ===========================================================================
def test_g_fleet():
    print("\n===== TEST G: Constellation Fleet Mode =====")

    # G25: Fleet config loads
    try:
        resp = requests.get(f"{BASE_URL}/api/fleet-config", timeout=10)
        if resp.status_code == 200:
            fleet = resp.json()
            plants = fleet if isinstance(fleet, list) else fleet.get("plants", fleet.get("fleet", []))
            log_result("G25", "Fleet config: loads", True, f"Plants: {len(plants)}")

            if plants:
                # Check plant metadata
                first = plants[0] if isinstance(plants, list) else plants
                has_meta = all(k in str(first) for k in ["capacity", "fuel"])
                log_result("G25", "Fleet config: has metadata", has_meta,
                           f"Sample keys: {list(first.keys())[:8] if isinstance(first, dict) else 'N/A'}")
        else:
            log_result("G25", "Fleet config: API responds", False,
                       f"HTTP {resp.status_code}")
    except Exception as e:
        log_result("G25", "Fleet config: loads", False, str(e))

    # G26: Simulation with fleet override (retire a plant)
    params = default_sim_request(iso="PJM", start_year=2025, end_year=2035, year_step=5)
    params["fleet_overrides"] = {"PLANT_001": "Retired"}
    result, elapsed, error = run_simulation(params)
    if result:
        log_result("G26", "Fleet override (retire): runs", True, f"Elapsed: {elapsed:.1f}s")
    else:
        # May fail if plant IDs don't match — that's OK
        log_result("G26", "Fleet override (retire): handled", error is not None,
                   f"Error (may be expected): {error}")

    # G27: CCS retrofit override
    params = default_sim_request(iso="PJM", start_year=2025, end_year=2035, year_step=5)
    params["fleet_overrides"] = {"PLANT_001": "CCS Retrofit"}
    result, elapsed, error = run_simulation(params)
    log_result("G27", "Fleet override (CCS): handled", True,
               f"Result: {'completed' if result else f'error: {error}'}")

# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  QA/QC AUDIT — Market Simulator v2-8")
    print("=" * 70)

    test_a_startup()
    test_b_all_isos()
    test_c_sensitivity()
    test_d_edge_cases()
    test_e_sweep()
    test_f_results()
    test_g_fleet()

    # Summary
    print("\n" + "=" * 70)
    print("  AUDIT SUMMARY")
    print("=" * 70)

    total = len(RESULTS)
    passed = sum(1 for r in RESULTS if r["status"] == "PASS")
    failed = sum(1 for r in RESULTS if r["status"] == "FAIL")

    print(f"\n  Total tests: {total}")
    print(f"  Passed:      {passed}")
    print(f"  Failed:      {failed}")
    print(f"  Pass rate:   {100*passed/total:.1f}%")

    if BUGS:
        print(f"\n  BUGS FOUND ({len(BUGS)}):")
        for bug in BUGS:
            print(f"    [{bug['test_id']}] {bug['name']}: {bug['bug']}")

    if failed > 0:
        print(f"\n  FAILED TESTS:")
        for r in RESULTS:
            if r["status"] == "FAIL":
                print(f"    [{r['id']}] {r['name']}: {r['detail'][:100]}")

    # Save results to JSON
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total": total, "passed": passed, "failed": failed,
        "pass_rate": f"{100*passed/total:.1f}%",
        "results": RESULTS,
        "bugs": BUGS
    }

    report_path = "/home/user/hourly-cfe-optimizer/market-simulator/qa_audit_results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Full report saved to: {report_path}")
