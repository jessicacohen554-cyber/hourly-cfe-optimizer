@echo off
REM ─────────────────────────────────────────────────────────────────────────
REM Clean Energy Profile Modeler — Windows Launcher
REM Double-click this file to start the tool
REM ─────────────────────────────────────────────────────────────────────────
title Clean Energy Profile Modeler
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
    echo ERROR: Python 3.9+ is required.
    pause
    exit /b 1
)
echo [OK] Found Python

REM ── Install dependencies ───────────────────────────────────────────────
echo Installing dependencies...
%PYTHON% -m pip install -r requirements.txt --quiet 2>nul
if errorlevel 1 (
    echo WARNING: Some dependencies may have failed to install.
) else (
    echo [OK] Dependencies installed
)

REM ── Open browser after short delay ─────────────────────────────────────
start "" /b cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:8050"

REM ── Start server ───────────────────────────────────────────────────────
echo.
echo ================================================================
echo   Clean Energy Profile Modeler
echo   Running at http://127.0.0.1:8050
echo   Close this window to stop the server
echo ================================================================
echo.

set PYTHONPATH=%~dp0;%~dp0..\\scripts
%PYTHON% -m uvicorn backend.server:app --host 127.0.0.1 --port 8050

pause
