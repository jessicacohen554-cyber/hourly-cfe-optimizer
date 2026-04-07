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
