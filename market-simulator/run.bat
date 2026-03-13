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
%PYTHON% -m pip install -r requirements.txt --quiet 2>nul
echo [OK] Dependencies installed

REM ── Generate synthetic data if needed ──────────────────────────────────
if not exist "data\demand_profiles_2025.json" (
    echo Generating synthetic data profiles...
    set PYTHONPATH=%~dp0
    %PYTHON% scripts\generate_synthetic_profiles.py 2>nul
)
echo [OK] Data profiles ready

REM ── Open browser after short delay ─────────────────────────────────────
start "" /b cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8000"

REM ── Start server ───────────────────────────────────────────────────────
echo.
echo ================================================================
echo   Market Simulator running at http://127.0.0.1:8000
echo   Close this window to stop the server
echo ================================================================
echo.

set PYTHONPATH=%~dp0
%PYTHON% -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

pause
