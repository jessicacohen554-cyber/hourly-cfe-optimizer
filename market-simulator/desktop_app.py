"""
Desktop application entry point for Market Simulator.

Launches a native pywebview window backed by the FastAPI/uvicorn server
running in a daemon thread. Works in both development mode (run directly)
and frozen mode (PyInstaller bundle).
"""

import os
import shutil
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Path resolution helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_frozen():
    """True when running inside a PyInstaller bundle."""
    return getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')


def get_bundle_dir():
    """Read-only bundled assets (frontend, scripts, default data)."""
    if is_frozen():
        return Path(sys._MEIPASS) / "market-simulator"
    return Path(__file__).resolve().parent


def get_app_data_dir():
    """Writable user data directory (next to exe when frozen, same as bundle in dev)."""
    if is_frozen():
        return Path(sys.executable).parent / "app_data"
    return Path(__file__).resolve().parent


# ─────────────────────────────────────────────────────────────────────────────
# First-run setup
# ─────────────────────────────────────────────────────────────────────────────

def first_run_setup():
    """Create writable app_data/ tree and copy templates from bundle on first run."""
    app_data = get_app_data_dir()
    bundle = get_bundle_dir()

    if not is_frozen():
        # In dev mode, directories already exist in-place
        return

    # Create directory structure
    dirs_to_create = [
        app_data / "data" / "eia-860",
        app_data / "data" / "eia-923",
        app_data / "data" / "epa-campd",
        app_data / "data" / "profiles",
        app_data / "custom-user-inputs",
        app_data / "results",
    ]
    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)

    # Copy bundled profiles to writable location if not already present
    bundled_profiles = bundle / "data" / "profiles"
    target_profiles = app_data / "data" / "profiles"
    if bundled_profiles.exists():
        for src_file in bundled_profiles.iterdir():
            dest = target_profiles / src_file.name
            if not dest.exists():
                shutil.copy2(src_file, dest)

    # Copy bundled fleet/reference data
    for filename in ["CEG_fleet_rosetta.csv", "constellation_fleet.json",
                     "cv_reference_results.json", "cv_comparison_report.json"]:
        src = bundle / "data" / filename
        dest = app_data / "data" / filename
        if src.exists() and not dest.exists():
            shutil.copy2(src, dest)

    # Copy custom-user-inputs templates
    bundled_templates = bundle / "custom-user-inputs"
    target_templates = app_data / "custom-user-inputs"
    if bundled_templates.exists():
        for src_file in bundled_templates.iterdir():
            dest = target_templates / src_file.name
            if not dest.exists():
                shutil.copy2(src_file, dest)


# ─────────────────────────────────────────────────────────────────────────────
# Heat rate staleness detection
# ─────────────────────────────────────────────────────────────────────────────

def _newest_mtime(directory):
    """Return the newest file mtime under a directory, or 0 if empty/missing."""
    directory = Path(directory)
    if not directory.is_dir():
        return 0
    newest = 0
    for f in directory.rglob("*"):
        if f.is_file():
            newest = max(newest, f.stat().st_mtime)
    return newest


def check_heat_rate_staleness():
    """Regenerate plant_heat_rates.json if source data is newer than the cache."""
    app_data = get_app_data_dir()
    data_dir = app_data / "data" if is_frozen() else app_data / "data"

    heat_rates_file = data_dir / "plant_heat_rates.json"
    eia923_dir = data_dir / "eia-923"
    campd_dir = data_dir / "epa-campd"

    # If neither source directory has actual files, nothing to do.
    # (first_run_setup creates empty dirs — don't trigger regeneration on those)
    def _has_files(d):
        return d.is_dir() and any(d.iterdir())

    if not _has_files(eia923_dir) and not _has_files(campd_dir):
        return

    # If heat rates file doesn't exist, regenerate
    if not heat_rates_file.exists():
        _regenerate_heat_rates(str(data_dir), str(heat_rates_file))
        return

    # Compare mtimes: if any source file is newer than cache, regenerate
    cache_mtime = heat_rates_file.stat().st_mtime
    source_mtime = max(_newest_mtime(eia923_dir), _newest_mtime(campd_dir))

    if source_mtime > cache_mtime:
        print("Heat rate source data is newer than cache — regenerating...")
        _regenerate_heat_rates(str(data_dir), str(heat_rates_file))


def _regenerate_heat_rates(data_dir, output_file):
    """Run heat rate generation with configurable paths."""
    if is_frozen():
        # In frozen mode, sys.executable is the bundle binary — can't use it
        # to run Python scripts. The heat rates data should be pre-bundled.
        # Fall back to importing the module directly if available.
        try:
            bundle = get_bundle_dir()
            sys.path.insert(0, str(bundle / "scripts"))
            from generate_plant_heat_rates import main as gen_main
            gen_main(data_dir=data_dir, output=output_file)
        except Exception:
            pass  # Non-critical — bundled data should suffice
        return

    import subprocess
    bundle = get_bundle_dir()
    script = bundle / "scripts" / "generate_plant_heat_rates.py"
    if not script.exists():
        # Dev mode — script is right next to us
        script = Path(__file__).resolve().parent / "scripts" / "generate_plant_heat_rates.py"
    if script.exists():
        subprocess.run(
            [sys.executable, str(script), "--data-dir", data_dir, "--output", output_file],
            check=False,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Free port selection
# ─────────────────────────────────────────────────────────────────────────────

def find_free_port():
    """Find an available port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


# ─────────────────────────────────────────────────────────────────────────────
# Server management
# ─────────────────────────────────────────────────────────────────────────────

def start_server(port):
    """Start uvicorn in the current thread (called from daemon thread)."""
    import traceback
    log_path = Path(get_app_data_dir()) / "server_debug.log"
    try:
        import uvicorn

        # Import the app object directly instead of using a string reference.
        # String-based import ("backend.main:app") fails in frozen mode because
        # PyInstaller's import system doesn't resolve package paths the same way.
        from backend.main import app

        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        # Disable signal handlers — we're in a daemon thread, not the main thread.
        server.install_signal_handlers = lambda: None
        server.run()
    except Exception:
        tb = traceback.format_exc()
        with open(log_path, "w") as f:
            f.write(f"port={port}\n")
            f.write(f"BUNDLE_DIR={os.environ.get('MARKET_SIM_BUNDLE_DIR')}\n")
            f.write(f"DATA_DIR={os.environ.get('MARKET_SIM_DATA_DIR')}\n")
            f.write(f"\nFATAL ERROR:\n{tb}\n")


def wait_for_server(port, timeout=30):
    """Poll /api/isos until the server is ready, up to timeout seconds."""
    url = f"http://127.0.0.1:{port}/api/isos"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(0.3)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def _get_log_path():
    """Return path to startup log file."""
    return get_app_data_dir() / "startup.log"


def main():
    import traceback

    # Set up Numba cache directory before any imports
    if is_frozen():
        app_data = get_app_data_dir()
        numba_cache = app_data / ".numba_cache"
        numba_cache.mkdir(parents=True, exist_ok=True)
        os.environ['NUMBA_CACHE_DIR'] = str(numba_cache)

    # Run first-run setup (creates directories, copies templates)
    first_run_setup()

    # Write startup log for debugging (survives even if GUI crashes)
    log = _get_log_path()
    with open(log, "w") as f:
        f.write(f"frozen={is_frozen()}\n")

    # Check heat rate staleness and auto-regenerate if needed
    check_heat_rate_staleness()

    # Find a free port
    port = find_free_port()

    # Set environment variables BEFORE importing the backend
    os.environ['MARKET_SIM_PORT'] = str(port)
    os.environ['MARKET_SIM_BUNDLE_DIR'] = str(get_bundle_dir())
    os.environ['MARKET_SIM_DATA_DIR'] = str(get_app_data_dir())

    # Start server in daemon thread
    server_thread = threading.Thread(target=start_server, args=(port,), daemon=True)
    server_thread.start()

    # Wait for server readiness
    if not wait_for_server(port):
        msg = f"Server failed to start on port {port} within 30 seconds."
        server_log = get_app_data_dir() / "server_debug.log"
        if server_log.exists():
            msg += f"\nServer log:\n{server_log.read_text()}"
        with open(log, "a") as f:
            f.write(f"ERROR: {msg}\n")
        print(f"ERROR: {msg}", file=sys.stderr)
        sys.exit(1)

    with open(log, "a") as f:
        f.write(f"server ready on port {port}\n")

    # Launch native window
    import webview

    window = webview.create_window(
        "Market Simulator",
        f"http://127.0.0.1:{port}",
        width=1400,
        height=900,
        min_size=(1024, 700),
    )
    webview.start()  # Blocks until window closes

    # Kill server thread and exit cleanly
    os._exit(0)


if __name__ == "__main__":
    main()
