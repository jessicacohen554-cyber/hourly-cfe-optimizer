#!/usr/bin/env python3
"""
Cross-platform Market Simulator launcher.
Run: python start.py
"""
import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

# app-startup/ is one level below the project root
ROOT = Path(__file__).resolve().parent.parent
PORT = 8000
URL = f"http://127.0.0.1:{PORT}"


def check_python_version():
    if sys.version_info < (3, 9):
        print(f"ERROR: Python 3.9+ required (found {sys.version})")
        sys.exit(1)
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")


def install_deps():
    print("Installing dependencies...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", str(ROOT / "app-startup" / "requirements.txt"), "--quiet"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("✓ Dependencies installed")


def ensure_data():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "backend"), str(ROOT / "scripts")])

    # 1. Demand + generation profiles (also generates interchange profiles)
    data_file = ROOT / "data" / "profiles" / "eia_demand_profiles.json"
    if not data_file.exists():
        print("Generating synthetic data profiles...")
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "generate_synthetic_profiles.py")],
            env=env,
        )
        if result.returncode != 0:
            print("  WARNING: Data profile generation failed.")
    print("✓ Data profiles ready")

    # 2. Plant heat rates (optional — enhances fleet-level dispatch)
    heat_file = ROOT / "data" / "plant_heat_rates.json"
    if not heat_file.exists():
        print("Generating plant heat rates...")
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "generate_plant_heat_rates.py")],
            env=env,
        )
        if result.returncode != 0:
            print("  WARNING: Plant heat rate generation failed.")
    print("✓ Plant heat rates ready")

    # 3. Interchange profiles (inter-regional import/export flows)
    interchange_file = ROOT / "data" / "profiles" / "eia_interchange_profiles.json"
    if not interchange_file.exists():
        print("Generating interchange profiles...")
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "generate_synthetic_profiles.py")],
            env=env,
        )
        if result.returncode != 0:
            print("  WARNING: Interchange profile generation failed.")
    print("✓ Interchange profiles ready")

    # 4. 1,215-scenario parametric sweep (one-time, may take 10-30 min)
    sweep_file = ROOT / "results" / "sweep_1215" / "sweep_1215_flat.parquet"
    if not sweep_file.exists():
        print("\n" + "=" * 60)
        print("  Sweep cache not found. Running 1,215-scenario parametric sweep.")
        print("  This is a one-time operation and may take a while...")
        print("  (3 demand x 5 price x 3 PPA x 3 gas x 3 queue x 3 fossil cost)")
        print("  x 7 ISOs x 6 years = 51,030 simulations")
        print("=" * 60 + "\n")
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_sweep_1215.py"),
             "--output-dir", str(ROOT / "results" / "sweep_1215")],
            env=env,
        )
        if result.returncode != 0:
            print("  WARNING: Sweep generation failed. Sweep features unavailable.")
    print("✓ Sweep cache ready")

    # 5. Fleet scenario data
    fleet_file = ROOT / "frontend" / "data" / "fleet_scenario_results_sample.json"
    if not fleet_file.exists():
        print("Building fleet scenario data...")
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_fleet_scenario_data.py")],
            env=env,
        )
        if result.returncode != 0:
            print("  WARNING: Fleet scenario data build failed.")
    print("✓ Fleet scenario data ready")

    # 6. Constellation scenarios
    constellation_file = ROOT / "frontend" / "data" / "constellation_scenarios.json"
    if not constellation_file.exists():
        print("Generating constellation scenarios...")
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "generate_constellation_scenarios.py")],
            env=env,
        )
        if result.returncode != 0:
            print("  WARNING: Constellation scenario generation failed.")
    print("✓ Constellation scenarios ready")

    # 7. ISO sweep data extraction for dashboard
    sweep_dispatch = ROOT / "frontend" / "data" / "sweep_dispatch_data.json"
    if not sweep_dispatch.exists():
        print("Extracting ISO sweep data for dashboard...")
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "extract_iso_sweep_data.py")],
            env=env,
        )
        if result.returncode != 0:
            print("  WARNING: ISO sweep data extraction failed.")
    print("✓ ISO sweep data ready")


def open_browser():
    time.sleep(2)
    webbrowser.open(URL)


def main():
    os.chdir(ROOT)
    check_python_version()
    install_deps()
    ensure_data()

    # Set PYTHONPATH for module imports (root + backend + scripts)
    os.environ["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "backend"), str(ROOT / "scripts")])

    print()
    print("═" * 60)
    print(f"  Market Simulator running at {URL}")
    print("  Press Ctrl+C to stop")
    print("═" * 60)
    print()

    # Open browser in background
    threading.Thread(target=open_browser, daemon=True).start()

    # Start uvicorn
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=PORT)


if __name__ == "__main__":
    main()
