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
