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

    # Demand + generation profiles (also generates interchange profiles)
    data_file = ROOT / "data" / "profiles" / "eia_demand_profiles.json"
    if not data_file.exists():
        print("Generating synthetic data profiles...")
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "generate_synthetic_profiles.py")],
            env=env, capture_output=True,
        )
    print("✓ Data profiles ready")

    # Plant heat rates (optional — enhances fleet-level dispatch)
    heat_file = ROOT / "data" / "plant_heat_rates.json"
    if not heat_file.exists():
        print("Generating plant heat rates...")
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "generate_plant_heat_rates.py")],
            env=env, capture_output=True,
        )
    print("✓ Plant heat rates ready")

    # Interchange profiles (inter-regional import/export flows)
    interchange_file = ROOT / "data" / "profiles" / "eia_interchange_profiles.json"
    if not interchange_file.exists():
        print("Generating interchange profiles...")
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "generate_synthetic_profiles.py")],
            env=env, capture_output=True,
        )
    print("✓ Interchange profiles ready")


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
