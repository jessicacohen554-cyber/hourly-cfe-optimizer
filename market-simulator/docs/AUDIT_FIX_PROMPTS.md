# Market Simulator Audit: Draft Prompts for Fixes

## Context

The market-simulator (`market-simulator/`) is a standalone desktop tool intended to run via `run.bat` (Windows), `run.sh` (Mac/Linux), `app-startup/start.py`, or as a packaged PyInstaller app (`desktop_app.py`). An audit found three categories of issues:

1. **Launcher script inconsistencies** — `run.sh` and `start.py` are missing steps and have wrong paths vs. `run.bat`
2. **Hybrid resource gaps** — backend supports solar+battery/wind+battery hybrids, but frontend, synthetic fallback, and sweep don't surface them
3. **Packaging documentation gaps** — `PACKAGING_PLAN.md` and `market_simulator.spec` omit hybrid profile NPZ files

Each section below is a **self-contained prompt** that can be given to an AI coding agent to execute that fix.

---

## Findings (Reference)

### A. Launcher Parity Issues
| Issue | run.bat | run.sh | start.py |
|-------|---------|--------|----------|
| Requirements path | `app-startup\requirements.txt` (correct) | `requirements.txt` (WRONG) | `app-startup/requirements.txt` (correct) |
| Data check file | `data\profiles\eia_demand_profiles.json` | `data/demand_profiles_2025.json` (WRONG) | `data/profiles/eia_demand_profiles.json` (correct) |
| Plant heat rates | Yes (line 43) | Missing | Yes |
| Interchange profiles | Yes (line 51) | Missing | Yes |
| 1,215-scenario sweep | Yes (line 59) | Missing | Missing |
| Fleet scenario data | Yes (line 80) | Missing | Missing |
| Constellation scenarios | Yes (line 96) | Missing | Missing |
| ISO sweep extraction | Yes (line 110) | Missing | Missing |
| Error handling | `2>nul` suppresses all errors | `set -e` fails fast | `capture_output=True` |
| JS lib download | No | Yes (plotly, html2canvas) | No |

### B. Hybrid Resource Gaps
- `market_simulation.py` lines 3947-3964 and 4020-4038: full hybrid support when reading EF/cost parquets
- `_generate_synthetic_step3_data()` (line 4188): generates only 6 base resources, **zero hybrids**
- `frontend/results.html` and `frontend/setup.html`: **zero** references to hybrid resources
- `run_sweep_1215.py`: **zero** hybrid deployment in the parametric sweep
- `data/hybrid_profiles/`: all 7 ISO NPZ files exist and are loaded by `dispatch_utils.py`
- `pipeline_config.py`: full LCOE, DC:AC ratios, capacity factors, queue caps for all 4 hybrid types

### C. Packaging Gaps
- `market_simulator.spec` line 28: bundles `data/profiles` but NOT `data/hybrid_profiles`
- `PACKAGING_PLAN.md` section 2: directory layout omits `hybrid_profiles/`
- No guidance on bundling optional EF parquets or data tier documentation

---

## Prompt 1: Fix run.sh to Match run.bat

> **Objective:** Update `market-simulator/run.sh` to have full startup parity with `run.bat`, fixing wrong paths and adding all missing data generation steps.
>
> **File to modify:** `market-simulator/run.sh`
>
> **Current state of run.sh (problems):**
> - Line 36: `pip install -r requirements.txt` — should be `app-startup/requirements.txt`
> - Line 40: checks `data/demand_profiles_2025.json` — should be `data/profiles/eia_demand_profiles.json`
> - Line 6: `set -e` causes the script to abort on any non-zero exit — optional steps like sweep generation should warn, not crash
> - Missing 5 startup steps that run.bat has (heat rates, interchange, sweep, fleet data, constellation, ISO sweep)
>
> **Required changes:**
>
> 1. **Fix requirements path** (line 36): Change to `app-startup/requirements.txt`
>
> 2. **Fix data check path** (line 40): Change to `data/profiles/eia_demand_profiles.json`
>
> 3. **Remove `set -e`** (line 6): Replace with manual error checks per step. Optional steps should warn and continue.
>
> 4. **Add PYTHONPATH export** early (after cd): `export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/backend:$SCRIPT_DIR/scripts"`
>
> 5. **Add plant heat rate generation** (after synthetic profiles):
>    ```bash
>    if [ ! -f "data/plant_heat_rates.json" ]; then
>        echo "Generating plant-specific heat rates..."
>        $PYTHON scripts/generate_plant_heat_rates.py || echo "  WARNING: Heat rate generation failed"
>    fi
>    echo "✓ Plant heat rates ready"
>    ```
>
> 6. **Add interchange profile generation** (after heat rates):
>    ```bash
>    if [ ! -f "data/profiles/eia_interchange_profiles.json" ]; then
>        echo "Generating inter-regional interchange profiles..."
>        $PYTHON scripts/generate_synthetic_profiles.py || echo "  WARNING: Interchange generation failed"
>    fi
>    echo "✓ Interchange profiles ready"
>    ```
>
> 7. **Add 1,215-scenario parametric sweep** (after interchange):
>    ```bash
>    if [ ! -f "results/sweep_1215/sweep_1215_flat.parquet" ]; then
>        echo ""
>        echo "================================================================"
>        echo "  Sweep cache not found. Running 1,215-scenario parametric sweep."
>        echo "  This is a one-time operation and may take a while..."
>        echo "  (3 demand x 5 price x 3 PPA x 3 gas x 3 queue x 3 fossil cost)"
>        echo "  x 7 ISOs x 6 years = 51,030 simulations"
>        echo "================================================================"
>        echo ""
>        $PYTHON scripts/run_sweep_1215.py --output-dir results/sweep_1215 || \
>            echo "  WARNING: Sweep generation failed. Sweep-dependent features will be unavailable."
>    fi
>    echo "✓ Sweep cache ready"
>    ```
>
> 8. **Add fleet scenario data build** (after sweep):
>    ```bash
>    if [ ! -f "frontend/data/fleet_scenario_results_sample.json" ]; then
>        echo "Building fleet scenario data..."
>        $PYTHON scripts/build_fleet_scenario_data.py || echo "  WARNING: Fleet scenario build failed"
>    fi
>    echo "✓ Fleet scenario data ready"
>    ```
>
> 9. **Add constellation scenarios** (after fleet):
>    ```bash
>    if [ ! -f "frontend/data/constellation_scenarios.json" ]; then
>        echo "Generating constellation scenarios..."
>        $PYTHON scripts/generate_constellation_scenarios.py || echo "  WARNING: Constellation scenarios failed"
>    fi
>    echo "✓ Constellation scenarios ready"
>    ```
>
> 10. **Add ISO sweep data extraction** (after constellation):
>     ```bash
>     if [ ! -f "frontend/data/sweep_dispatch_data.json" ]; then
>         echo "Extracting ISO sweep data for dashboard..."
>         $PYTHON scripts/extract_iso_sweep_data.py || echo "  WARNING: ISO sweep extraction failed"
>     fi
>     echo "✓ ISO sweep data ready"
>     ```
>
> 11. **Keep the JS library download steps** (plotly, html2canvas) — these are good for offline use and run.bat should arguably add them too.
>
> **Verification:** Run `bash run.sh` from the `market-simulator/` directory. All steps should execute in order. If any optional step fails, the script should print a warning and continue to server startup.

---

## Prompt 2: Fix start.py to Match run.bat

> **Objective:** Update `market-simulator/app-startup/start.py` to include the full startup sequence matching `run.bat`. Currently it handles profiles, heat rates, and interchange, but is missing the sweep, fleet data, constellation, and ISO sweep steps.
>
> **File to modify:** `market-simulator/app-startup/start.py`
>
> **Current state** (see `app-startup/start.py`, 102 lines):
> - `ensure_data()` handles 3 checks: demand profiles, plant heat rates, interchange profiles
> - Missing: sweep generation, fleet scenario data, constellation scenarios, ISO sweep extraction
> - Error handling uses `capture_output=True` which silently swallows errors
>
> **Required changes:**
>
> 1. **Add sweep generation** to `ensure_data()` (after interchange profiles):
>    ```python
>    # 1,215-scenario parametric sweep (one-time, may take 10-30 min)
>    sweep_file = ROOT / "results" / "sweep_1215" / "sweep_1215_flat.parquet"
>    if not sweep_file.exists():
>        print("\n" + "=" * 60)
>        print("  Sweep cache not found. Running 1,215-scenario parametric sweep.")
>        print("  This is a one-time operation and may take a while...")
>        print("  (3 demand x 5 price x 3 PPA x 3 gas x 3 queue x 3 fossil cost)")
>        print("  x 7 ISOs x 6 years = 51,030 simulations")
>        print("=" * 60 + "\n")
>        result = subprocess.run(
>            [sys.executable, str(ROOT / "scripts" / "run_sweep_1215.py"),
>             "--output-dir", str(ROOT / "results" / "sweep_1215")],
>            env=env,
>        )
>        if result.returncode != 0:
>            print("  WARNING: Sweep generation failed. Sweep features unavailable.")
>    print("✓ Sweep cache ready")
>    ```
>
> 2. **Add fleet scenario data build**:
>    ```python
>    fleet_file = ROOT / "frontend" / "data" / "fleet_scenario_results_sample.json"
>    if not fleet_file.exists():
>        print("Building fleet scenario data...")
>        result = subprocess.run(
>            [sys.executable, str(ROOT / "scripts" / "build_fleet_scenario_data.py")],
>            env=env, capture_output=True,
>        )
>        if result.returncode != 0:
>            print("  WARNING: Fleet scenario data build failed.")
>    print("✓ Fleet scenario data ready")
>    ```
>
> 3. **Add constellation scenarios**:
>    ```python
>    constellation_file = ROOT / "frontend" / "data" / "constellation_scenarios.json"
>    if not constellation_file.exists():
>        print("Generating constellation scenarios...")
>        result = subprocess.run(
>            [sys.executable, str(ROOT / "scripts" / "generate_constellation_scenarios.py")],
>            env=env, capture_output=True,
>        )
>        if result.returncode != 0:
>            print("  WARNING: Constellation scenario generation failed.")
>    print("✓ Constellation scenarios ready")
>    ```
>
> 4. **Add ISO sweep data extraction**:
>    ```python
>    sweep_dispatch = ROOT / "frontend" / "data" / "sweep_dispatch_data.json"
>    if not sweep_dispatch.exists():
>        print("Extracting ISO sweep data for dashboard...")
>        result = subprocess.run(
>            [sys.executable, str(ROOT / "scripts" / "extract_iso_sweep_data.py")],
>            env=env, capture_output=True,
>        )
>        if result.returncode != 0:
>            print("  WARNING: ISO sweep data extraction failed.")
>    print("✓ ISO sweep data ready")
>    ```
>
> 5. **Fix error reporting** for existing 3 steps: Add return code checking after each `subprocess.run` call. Print warnings on failure instead of silently continuing.
>
> **Verification:** Run `python app-startup/start.py` from `market-simulator/`. All 7 startup steps should execute with progress messages. Server starts even if optional steps fail.

---

## Prompt 3: Fix run.bat Minor Issues

> **Objective:** Fix error handling and path issues in `market-simulator/run.bat`.
>
> **File to modify:** `market-simulator/run.bat`
>
> **Issues and fixes:**
>
> 1. **Line 31 — pip install has no error check:**
>    Current: `%PYTHON% -m pip install -r app-startup\requirements.txt --quiet 2>nul`
>    Problem: If pip fails (no network, corrupted env), the script silently continues and crashes later on import.
>    Fix: Add error check:
>    ```bat
>    %PYTHON% -m pip install -r app-startup\requirements.txt --quiet 2>nul
>    if errorlevel 1 (
>        echo WARNING: Dependency installation may have failed. Some features may not work.
>    )
>    ```
>
> 2. **Lines 38, 54 — `2>nul` suppresses all stderr on data generation scripts:**
>    Current: `%PYTHON% scripts\generate_synthetic_profiles.py 2>nul`
>    Problem: Real errors (missing modules, file permission issues) are silently swallowed. If generation fails, the script says "[OK]" anyway.
>    Fix: Remove `2>nul` and add error check:
>    ```bat
>    %PYTHON% scripts\generate_synthetic_profiles.py
>    if errorlevel 1 (
>        echo WARNING: Synthetic profile generation failed.
>    )
>    ```
>    Apply same pattern to line 54 (interchange profiles).
>
> 3. **Line 54 — Clarify interchange reuse:**
>    Current: Runs `generate_synthetic_profiles.py` again to generate interchange profiles.
>    Fix: Add comment explaining the reuse:
>    ```bat
>    REM generate_synthetic_profiles.py also produces interchange profiles
>    %PYTHON% scripts\generate_synthetic_profiles.py
>    ```
>
> 4. **Lines 37, 68 — PYTHONPATH trailing backslash:**
>    Current: `set PYTHONPATH=%~dp0;%~dp0backend;%~dp0scripts`
>    `%~dp0` expands to a path WITH trailing backslash (e.g., `C:\Users\foo\market-simulator\`).
>    This means PYTHONPATH includes `C:\Users\foo\market-simulator\;C:\Users\foo\market-simulator\backend;...`
>    The trailing backslash is usually harmless on Windows but could cause issues with some Python path resolution.
>    Fix: Strip trailing backslash or verify behavior. Low priority — note in comments if not fixed.
>
> **Verification:** On Windows, delete `data/profiles/eia_demand_profiles.json` and run `run.bat`. Verify that generation succeeds with visible output (not suppressed). Verify that if pip install fails (e.g., with `--dry-run` flag), a warning is printed.

---

## Prompt 4: Add Hybrid Resources to Synthetic Data Fallback

> **Objective:** Update `_generate_synthetic_step3_data()` in `market_simulation.py` to include hybrid solar+battery and wind+battery resource allocations. Without this, standalone deployments without EF parquets show zero hybrid resources.
>
> **File to modify:** `market-simulator/scripts/market_simulation.py`
> **Function:** `_generate_synthetic_step3_data()` starting at line 4188
>
> **Current state:**
> The function generates synthetic resource mixes for standalone operation. It produces `resource_pcts` with only 6 base types: `clean_firm`, `solar`, `wind`, `offshore_wind`, `ccs_ccgt`, `hydro`. No hybrid resources. The result dict also lacks the `'include_hybrids'` flag.
>
> **Context — hybrid types and constraints** (from `pipeline_config.py`):
> - 4 hybrid types: `solar_batt4`, `solar_batt8`, `wind_batt4`, `wind_batt8`
> - `HYBRID_MAX_PER_TYPE = 40` (max 40% of demand per individual hybrid type)
> - Solar family cap: solar + solar_batt4 + solar_batt8 combined ≤ `SOLAR_FAMILY_CAP[iso]` (120% typical)
> - Wind family cap: wind + wind_batt4 + wind_batt8 combined ≤ `WIND_FAMILY_CAP[iso]` (80-210% depending on ISO)
> - Queue caps for hybrids already defined in `market_simulation.py` lines 290-314 (GW/year by ISO)
>
> **Required changes:**
>
> 1. **Import hybrid constants** at top of function:
>    ```python
>    from pipeline_config import HYBRID_TYPES, HYBRID_MAX_PER_TYPE
>    ```
>
> 2. **Add hybrid ramps** after the base resource allocation (after line 4257, after `resource_pcts` dict is built):
>    ```python
>    # Hybrid solar+battery and wind+battery — fraction of parent resource
>    # Hybrids ramp slower than standalone: new tech, requires co-siting
>    solar_total = resource_pcts['solar']
>    wind_total = resource_pcts['wind']
>
>    sb4 = min(HYBRID_MAX_PER_TYPE, round(solar_total * 0.25 * progress, 2))
>    sb8 = min(HYBRID_MAX_PER_TYPE, round(solar_total * 0.12 * max(0, progress - 0.3) / 0.7, 2))
>    wb4 = min(HYBRID_MAX_PER_TYPE, round(wind_total * 0.18 * progress, 2))
>    wb8 = min(HYBRID_MAX_PER_TYPE, round(wind_total * 0.08 * max(0, progress - 0.3) / 0.7, 2))
>
>    resource_pcts['solar_batt4'] = sb4
>    resource_pcts['solar_batt8'] = sb8
>    resource_pcts['wind_batt4'] = wb4
>    resource_pcts['wind_batt8'] = wb8
>    ```
>
> 3. **Set `include_hybrids` flag** in the result dict (line 4265-4276):
>    Add `'include_hybrids': True,` to the `iso_data[t]` dict.
>
> 4. **Rationale for ramp fractions:**
>    - `solar_batt4`: 25% of solar at full progress — 4hr battery is mature, common co-location
>    - `solar_batt8`: 12% of solar, delayed onset (progress > 0.3) — 8hr is newer, higher cost
>    - `wind_batt4`: 18% of wind — less common than solar+batt but growing
>    - `wind_batt8`: 8% of wind, delayed onset — niche application
>    - All capped at `HYBRID_MAX_PER_TYPE` (40%) per pipeline_config constraints
>
> **Verification:**
> 1. Temporarily rename/remove any `data/step2.1-ef/` and `data/step2.2-cost/` directories
> 2. Start the server: `python -m uvicorn backend.main:app`
> 3. Call `GET /api/simulate` with default params for CAISO at 90% threshold
> 4. Verify the response includes `solar_batt4`, `solar_batt8`, `wind_batt4`, `wind_batt8` in `resource_pcts` with non-zero values
> 5. Verify `include_hybrids` is `true` in the response

---

## Prompt 5: Surface Hybrid Resources in Frontend Results

> **Objective:** Update the results page JavaScript to display hybrid solar+battery and wind+battery resources in charts and tables when present in the API response.
>
> **Files to modify:**
> - `market-simulator/frontend/js/results.js`
>
> **Current state:**
> - `chart-colors.js` already defines hybrid colors: `solarBatt4` (#E6890B), `solarBatt8` (#CC7A0A), `windBatt4` (#1AA34E), `windBatt8` (#158F42) — plus transparent and background variants
> - `results.js` line 33-43: `CLEAN_COLORS` dict has only 9 base resource colors — no hybrids
> - `results.js` line 46-53: `ALL_RESOURCE_COLORS` spreads `CLEAN_COLORS` + fossil colors — no hybrids
> - Supply stack chart (line 791+): iterates over `resources` from API response — will show hybrids IF they're in the data AND have a color mapping
> - "What Gets Built" donut (line 1185+): same pattern — iterates API keys, maps to `CLEAN_COLORS`
> - Without color mappings, hybrid resources render as gray (#9CA3AF fallback)
>
> **Required changes:**
>
> 1. **Add hybrid colors to `CLEAN_COLORS`** (line 33-43):
>    ```javascript
>    const CLEAN_COLORS = {
>        solar: '#FBB254',
>        wind: '#6BA543',
>        offshore_wind: '#007FA4',
>        nuclear: '#2372B9',
>        ccs_ccgt: '#7F8F97',
>        hydro: '#007FA4',
>        battery: '#CADB2E',
>        ldes: '#F47B27',
>        geothermal: '#9B6B3A',
>        // Hybrid solar+battery and wind+battery
>        solar_batt4: '#E6890B',
>        solar_batt8: '#CC7A0A',
>        wind_batt4: '#1AA34E',
>        wind_batt8: '#158F42',
>    };
>    ```
>
> 2. **Add human-readable labels** for hybrid resources wherever `r.replace(/_/g, ' ')` is used for display. The raw names `solar batt4` aren't user-friendly. Add a label map:
>    ```javascript
>    const RESOURCE_LABELS = {
>        solar_batt4: 'Solar + 4hr Battery',
>        solar_batt8: 'Solar + 8hr Battery',
>        wind_batt4: 'Wind + 4hr Battery',
>        wind_batt8: 'Wind + 8hr Battery',
>        clean_firm: 'Clean Firm',
>        ccs_ccgt: 'CCS Gas',
>        offshore_wind: 'Offshore Wind',
>    };
>    function resourceLabel(key) {
>        return RESOURCE_LABELS[key] || key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
>    }
>    ```
>    Then replace `r.replace(/_/g, ' ')` calls (lines ~879, 1195, and elsewhere) with `resourceLabel(r)`.
>
> 3. **Supply stack chart** (line 791+): No structural changes needed — it already iterates API keys. The color mapping fix (step 1) makes hybrids render correctly. Verify that the stacked bar chart doesn't overflow with 10+ resource types on small screens.
>
> 4. **"What Gets Built" donut** (line 1185+): Same — color mapping fix enables it. Verify the donut legend doesn't overflow.
>
> 5. **Resource Economics table** (line 317-326): If this table shows per-resource capture rates, add hybrid rows. Check the API response structure for `resource_economics` or similar keys.
>
> **Verification:**
> 1. Start server with synthetic data (Prompt 4 must be implemented first)
> 2. Run a simulation for CAISO at 90% threshold
> 3. Verify supply stack chart shows hybrid slices in correct colors (amber/green variants)
> 4. Verify "What Gets Built" donut includes hybrid slices with readable labels
> 5. Verify no gray (#9CA3AF) fallback colors appear for recognized resource types
> 6. Check mobile viewport (375px) — charts should not overflow

---

## Prompt 6: Include Hybrid Resources in Sweep Runner

> **Objective:** Add hybrid solar+battery and wind+battery resource deployment to the 1,215-scenario parametric sweep so that sweep-dependent charts and analysis show hybrid resources.
>
> **Files to modify:**
> - `market-simulator/scripts/run_sweep_1215.py`
> - `market-simulator/scripts/market_simulation.py` (deployment logic used by sweep)
>
> **Context:**
> - The sweep runs `market_simulation.run_full_sweep()` which calls `apply_economic_deployment()` for each scenario
> - `apply_economic_deployment()` has queue caps for all 4 hybrid types (lines 290-314 of `market_simulation.py`)
> - But the function currently only deploys base resources (solar, wind, clean_firm, etc.)
> - Hybrid deployment requires: (a) checking if hybrid profiles are available, (b) allocating capacity from the parent resource's queue to hybrids based on economic merit
>
> **This is the largest change in the audit.** It touches the core simulation loop. Approach:
>
> 1. **In `apply_economic_deployment()`**: After deploying standalone solar and wind, check if hybrid profiles are loaded. If so, split a fraction of remaining queue capacity into hybrid deployment:
>    - `solar_batt4` and `solar_batt8` draw from the solar queue
>    - `wind_batt4` and `wind_batt8` draw from the wind queue
>    - Economic decision: hybrid if `hybrid_lcoe < standalone_lcoe + battery_value_of_shifting` (simplified: deploy hybrids when standalone renewable curtailment exceeds a threshold)
>
> 2. **In `run_sweep_1215.py`**: Ensure the sweep output parquet schema includes hybrid columns. If the sweep aggregation functions (groupby, mean, etc.) don't expect hybrids, they'll silently drop them.
>
> 3. **Hybrid deployment fraction heuristic** (if full economic dispatch is too complex for the sweep):
>    ```python
>    # Simplified: at high clean %, more curtailment → more hybrid value
>    hybrid_fraction = min(0.4, max(0.0, (clean_pct - 60) / 100))
>    solar_batt4_gw = solar_deployed_gw * hybrid_fraction * 0.5
>    solar_batt8_gw = solar_deployed_gw * hybrid_fraction * 0.3
>    wind_batt4_gw = wind_deployed_gw * hybrid_fraction * 0.4
>    wind_batt8_gw = wind_deployed_gw * hybrid_fraction * 0.2
>    ```
>
> 4. **Risk note:** This prompt changes simulation results. All sweep caches will need to be regenerated after implementation. Test on a single ISO (e.g., CAISO) before running the full 7-ISO sweep.
>
> **Verification:**
> 1. Run single-ISO sweep: `python scripts/run_sweep_1215.py --output-dir results/sweep_test --isos CAISO`
> 2. Load the output parquet and verify hybrid columns exist with non-zero values
> 3. Compare total clean % with and without hybrids — should be higher with hybrids (hybrids shift energy to match demand better)
> 4. Verify sweep aggregation (mean, P10, P90) includes hybrid columns

---

## Prompt 7: Update PyInstaller Spec to Bundle Hybrid Profiles

> **Objective:** Add hybrid profile NPZ files to the PyInstaller bundle so the packaged desktop app can dispatch hybrid resources.
>
> **File to modify:** `market-simulator/market_simulator.spec`
>
> **Current state (line 28):**
> ```python
> ('data/profiles', 'market-simulator/data/profiles'),
> ```
> The spec bundles `data/profiles/` (synthetic demand/gen/fossil/interchange JSON files) but does NOT bundle `data/hybrid_profiles/` (7 NPZ files, one per ISO, required for hybrid dispatch).
>
> **Required change — add one line** after line 28:
> ```python
> ('data/hybrid_profiles', 'market-simulator/data/hybrid_profiles'),
> ```
>
> **Full datas section after fix:**
> ```python
> datas=[
>     # Frontend (HTML, CSS, JS, images)
>     ('frontend', 'market-simulator/frontend'),
>     # Backend (FastAPI app)
>     ('backend', 'market-simulator/backend'),
>     # Scripts (simulation engine)
>     ('scripts', 'market-simulator/scripts'),
>     # Bundled default data (NOT EIA/CAMPD — just synthetic + fleet)
>     ('data/profiles', 'market-simulator/data/profiles'),
>     ('data/hybrid_profiles', 'market-simulator/data/hybrid_profiles'),  # ← NEW
>     ('data/CEG_fleet_rosetta.csv', 'market-simulator/data'),
>     ('data/constellation_fleet.json', 'market-simulator/data'),
>     ('data/cv_reference_results.json', 'market-simulator/data'),
>     ('data/cv_comparison_report.json', 'market-simulator/data'),
>     ('data/DATA_README.md', 'market-simulator/data'),
>     # Custom user input templates
>     ('custom-user-inputs', 'market-simulator/custom-user-inputs'),
>     # Pre-computed results
>     ('results/sweep_1215', 'market-simulator/results/sweep_1215'),
>     ('results/fleet_results.json', 'market-simulator/results'),
>     ('results/fleet_scenario_results.json', 'market-simulator/results'),
> ] + pyarrow_datas,
> ```
>
> **Why this matters:** Without bundling `hybrid_profiles/`, the packaged `.exe`/`.app` cannot load hybrid dispatch shapes. `dispatch_utils._load_hybrid_profiles()` will print "WARNING: No hybrid profiles at ..." and return empty dicts. Hybrid resources will show 0% contribution even when EF parquets or synthetic data include them.
>
> **Verification:**
> 1. Run `pyinstaller market_simulator.spec`
> 2. Check `dist/MarketSimulator/_internal/market-simulator/data/hybrid_profiles/` exists
> 3. Verify all 7 NPZ files are present: `CAISO_hybrid_profiles.npz`, `ERCOT_hybrid_profiles.npz`, etc.

---

## Prompt 8: Update PACKAGING_PLAN.md

> **Objective:** Update packaging documentation to reflect hybrid resources, full startup sequence, and data tier model.
>
> **File to modify:** `market-simulator/PACKAGING_PLAN.md`
>
> **Changes:**
>
> 1. **Section 2 (Directory Layout):** Add `hybrid_profiles/` and optional EF data to the `_internal/market-simulator/data/` tree. Current layout shows:
>    ```
>    │       └── data/                ← Bundled defaults only:
>    │           ├── profiles/        ←   synthetic demand/gen/fossil/interchange
>    │           ├── CEG_fleet_rosetta.csv
>    │           ├── constellation_fleet.json
>    │           ├── cv_reference_results.json
>    │           └── cv_comparison_report.json
>    ```
>    Updated layout:
>    ```
>    │       └── data/                ← Bundled defaults only:
>    │           ├── profiles/        ←   synthetic demand/gen/fossil/interchange
>    │           ├── hybrid_profiles/ ←   7 ISO NPZ files (solar+batt/wind+batt 8760 dispatch shapes)
>    │           ├── CEG_fleet_rosetta.csv
>    │           ├── constellation_fleet.json
>    │           ├── cv_reference_results.json
>    │           └── cv_comparison_report.json
>    ```
>    Also add to `app_data/data/`:
>    ```
>    │   ├── step2.1-ef/             ← OPTIONAL: efficient frontier parquets (enhances resource mix accuracy)
>    │   ├── step2.2-cost/           ← OPTIONAL: cost-optimized parquets (full cost scenario analysis)
>    ```
>
> 2. **Section 4 (PyInstaller Spec):** Add the hybrid_profiles data line to match the actual spec (Prompt 7).
>
> 3. **Add new section: "Data Tiers"** (after section 3). Document the 4 data tiers that control result quality:
>
>    ```markdown
>    ## 3b. Data Tiers
>
>    The market simulator operates at different quality levels depending on
>    available data. Each tier adds accuracy:
>
>    | Tier | Data Required | What It Adds |
>    |------|--------------|--------------|
>    | 0 — Synthetic | None (ships with app) | Approximate resource mixes, synthetic LMP, illustrative results |
>    | 1 — EF Physics | `step2.1-ef/` parquets | Physics-optimal resource mixes with hybrid resources |
>    | 2 — Cost-Optimized | `step2.2-cost/` parquets | Full cost-optimized results across 5,832 scenarios |
>    | 3 — Plant-Level | `eia-860/`, `eia-923/`, `epa-campd/` | Plant-specific heat rates, emissions, fleet dispatch |
>
>    Tier 0 works out of the box. Users drop parquet files into
>    `app_data/data/step2.1-ef/` or `app_data/data/step2.2-cost/` to upgrade.
>    The app auto-detects which tier is available per ISO.
>    ```
>
> 4. **Section on first-run startup sequence:** Document the full 7-step startup that `run.bat`/`run.sh`/`start.py` perform:
>    1. Install dependencies
>    2. Generate synthetic profiles (if missing)
>    3. Generate plant heat rates (if missing)
>    4. Generate interchange profiles (if missing)
>    5. Run 1,215-scenario parametric sweep (if missing) — one-time, 10-30 min
>    6. Build fleet scenario data (if missing)
>    7. Generate constellation scenarios + extract ISO sweep data (if missing)
>
>    Note that the desktop app (`desktop_app.py`) handles steps 2-4 in `first_run_setup()`. Steps 5-7 should also be added to `desktop_app.py` or triggered by the backend on first request.
>
> **Verification:** Read the updated PACKAGING_PLAN.md and verify:
> 1. Directory layout includes `hybrid_profiles/`
> 2. Data tiers table is accurate
> 3. Startup sequence matches run.bat
> 4. PyInstaller spec section references hybrid_profiles

---

## Priority Order

1. **Prompts 1-3** (launcher fixes) — critical for standalone usability on Mac/Linux
2. **Prompt 4** (synthetic hybrid data) — makes hybrids appear in default mode
3. **Prompt 7** (PyInstaller spec) — one-line fix, high impact for packaged app
4. **Prompt 8** (PACKAGING_PLAN.md) — documentation alignment
5. **Prompt 5** (frontend) — surfaces hybrid data to users in charts
6. **Prompt 6** (sweep) — largest change, lowest urgency, changes simulation results
