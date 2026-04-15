@echo off
REM ─────────────────────────────────────────────────────────────────────────
REM Market Simulator Launcher (Windows)
REM Double-click this file to start the simulator
REM ─────────────────────────────────────────────────────────────────────────
title Market Simulator
cd /d "%~dp0"

REM ── Find Python ────────────────────────────────────────────────────────
set PYTHON=
where python3 >nul 2>&1 && set PYTHON=python3
if "%PYTHON%"=="" (where python >nul 2>&1 && set PYTHON=python)
if "%PYTHON%"=="" (
    echo ERROR: Python 3.9+ is required but not found.
    echo Install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

%PYTHON% -c "import sys; exit(0 if sys.version_info >= (3,9) else 1)" 2>nul
if errorlevel 1 (
    echo ERROR: Python 3.9+ is required. Current version is too old.
    pause
    exit /b 1
)

echo [OK] Found Python

REM ── Install dependencies ───────────────────────────────────────────────
echo Installing dependencies...
%PYTHON% -m pip install -r app-startup\requirements.txt --quiet 2>nul
if errorlevel 1 (
    echo WARNING: Dependency installation may have failed. Some features may not work.
) else (
    echo [OK] Dependencies installed
)

REM ── Generate synthetic data if needed ──────────────────────────────────
if not exist "data\profiles\eia_demand_profiles.json" (
    echo Generating synthetic data profiles...
    REM Note: %~dp0 includes trailing backslash; harmless on Windows but noted for awareness
    set PYTHONPATH=%~dp0;%~dp0backend;%~dp0scripts
    %PYTHON% scripts\generate_synthetic_profiles.py
    if errorlevel 1 (
        echo WARNING: Synthetic profile generation failed.
    )
)
echo [OK] Data profiles ready

REM ── Generate plant heat rates if needed ─────────────────────────────
if not exist "data\plant_heat_rates.json" (
    echo Generating plant-specific heat rates...
    set PYTHONPATH=%~dp0;%~dp0backend;%~dp0scripts
    %PYTHON% scripts\generate_plant_heat_rates.py
    if errorlevel 1 (
        echo WARNING: Plant heat rate generation failed.
    )
)
echo [OK] Plant heat rates ready

REM ── Generate interchange profiles if needed ────────────────────────
if not exist "data\profiles\eia_interchange_profiles.json" (
    echo Generating inter-regional interchange profiles...
    REM generate_synthetic_profiles.py also produces interchange profiles
    set PYTHONPATH=%~dp0;%~dp0backend;%~dp0scripts
    %PYTHON% scripts\generate_synthetic_profiles.py
    if errorlevel 1 (
        echo WARNING: Interchange profile generation failed.
    )
)
echo [OK] Interchange profiles ready

REM ── Run 1,215-scenario parametric sweep if cached results missing ─────
if not exist "results\sweep_1215\sweep_1215_flat.parquet" (
    echo.
    echo ================================================================
    echo   Sweep cache not found. Running 1,215-scenario parametric sweep.
    echo   This is a one-time operation and may take a while...
    echo   ^(3 demand x 5 price x 3 PPA x 3 gas x 3 queue x 3 fossil cost^)
    echo   x 7 ISOs x 6 years = 51,030 simulations
    echo ================================================================
    echo.
    set PYTHONPATH=%~dp0;%~dp0backend;%~dp0scripts
    %PYTHON% scripts\run_sweep_1215.py --output-dir results\sweep_1215
    if errorlevel 1 (
        echo WARNING: Sweep generation failed. The UI will still load but
        echo sweep-dependent features will be unavailable.
    ) else (
        echo [OK] Sweep results generated
    )
)
echo [OK] Sweep cache ready

REM ── Build fleet scenario data if missing ─────────────────────────────────
if not exist "frontend\data\fleet_scenario_results_sample.json" (
    echo.
    echo Building fleet scenario data from Rosetta + CAMPD...
    set PYTHONPATH=%~dp0;%~dp0backend;%~dp0scripts
    %PYTHON% scripts\build_fleet_scenario_data.py
    if errorlevel 1 (
        echo WARNING: Fleet scenario data build failed. Fleet scenarios page
        echo will be unavailable.
    ) else (
        echo [OK] Fleet scenario data generated
    )
) else (
    echo [OK] Fleet scenario data ready
    REM ── Sanity-check data: 2050 reduction must be at least 35%% of 2023 ─────
    %PYTHON% -c "import json,sys; d=json.load(open('frontend\\data\\fleet_scenario_results_sample.json')); e=d['scenarios']['baseline']['envelope']; b=e['2023']['p50']; e50=e['2050']['p50']; redux=(b-e50)/b*100; sys.exit(0 if redux >= 35 else 1)" 2>nul
    if errorlevel 1 (
        echo   WARNING: Fleet scenario data failed sanity check ^(2050 reduction ^< 35%%^). Rebuilding...
        set PYTHONPATH=%~dp0;%~dp0backend;%~dp0scripts
        %PYTHON% scripts\build_fleet_scenario_data.py
        if errorlevel 1 (
            echo WARNING: Rebuild also failed. Check sweep data integrity.
        ) else (
            echo [OK] Fleet scenario data rebuilt successfully
        )
    )
)

REM ── Generate constellation scenarios if missing ──────────────────────────
if not exist "frontend\data\constellation_scenarios.json" (
    echo Generating constellation scenarios...
    set PYTHONPATH=%~dp0;%~dp0backend;%~dp0scripts
    %PYTHON% scripts\generate_constellation_scenarios.py
    if errorlevel 1 (
        echo WARNING: Constellation scenarios generation failed.
    ) else (
        echo [OK] Constellation scenarios generated
    )
) else (
    echo [OK] Constellation scenarios ready
)

REM ── Extract ISO sweep data for dashboard if missing ──────────────────────
if not exist "frontend\data\sweep_dispatch_data.json" (
    echo Extracting ISO sweep data for dashboard...
    set PYTHONPATH=%~dp0;%~dp0backend;%~dp0scripts
    %PYTHON% scripts\extract_iso_sweep_data.py
    if errorlevel 1 (
        echo WARNING: ISO sweep data extraction failed.
    ) else (
        echo [OK] ISO sweep data extracted
    )
) else (
    echo [OK] ISO sweep data ready
)

REM ── Open browser after short delay ─────────────────────────────────────
start "" /b cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8000"

REM ── Start server ───────────────────────────────────────────────────────
echo.
echo ================================================================
echo   Market Simulator running at http://127.0.0.1:8000
echo   Close this window to stop the server
echo ================================================================
echo.

set PYTHONPATH=%~dp0;%~dp0backend;%~dp0scripts
%PYTHON% -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

pause
