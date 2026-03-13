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

ROOT = Path(__file__).resolve().parent
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
        [sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt"), "--quiet"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("✓ Dependencies installed")


def ensure_data():
    data_file = ROOT / "data" / "demand_profiles_2025.json"
    if not data_file.exists():
        print("Generating synthetic data profiles...")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT)
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "generate_synthetic_profiles.py")],
            env=env, capture_output=True,
        )
    print("✓ Data profiles ready")


def open_browser():
    time.sleep(2)
    webbrowser.open(URL)


def main():
    os.chdir(ROOT)
    check_python_version()
    install_deps()
    ensure_data()

    # Set PYTHONPATH for module imports
    os.environ["PYTHONPATH"] = str(ROOT)

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
