# PACKAGING_PLAN.md — Desktop App Packaging (PyInstaller + pywebview)

## Overview

Package the market-simulator as a standalone desktop application. Users double-click a single `.exe` (Windows) or `.app` (macOS) — no Python, no terminal, no browser required. The app opens in a native window via pywebview, backed by the existing FastAPI/uvicorn server running locally.

**Target platforms**: Windows + macOS (separate builds — PyInstaller doesn't cross-compile).

---

## 1. Entry Point: `desktop_app.py`

New file at `market-simulator/desktop_app.py`. Responsibilities:

1. **Resolve paths** — detect frozen (PyInstaller) vs dev mode
2. **First-run setup** — create writable `app_data/` directory structure, copy templates
3. **Generate synthetic data** if missing (profiles, heat rates)
4. **Start uvicorn** in a daemon thread on a dynamically-selected free port
5. **Poll for readiness** — hit `/api/isos` every 0.3s, up to 30s
6. **Open pywebview** native window pointed at `http://127.0.0.1:{PORT}`
7. **Shutdown** — when window closes, `os._exit(0)` to kill server + process

### Path resolution helpers

```python
import sys
from pathlib import Path

def is_frozen():
    return getattr(sys, 'frozen', False)

def get_bundle_dir():
    """Read-only bundled assets (frontend, scripts, default data)."""
    if is_frozen():
        return Path(sys._MEIPASS) / "market-simulator"
    return Path(__file__).resolve().parent

def get_app_data_dir():
    """Writable user data directory next to the executable."""
    if is_frozen():
        return Path(sys.executable).parent / "app_data"
    return Path(__file__).resolve().parent
```

### Free port selection

```python
import socket

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]
```

### Server thread

```python
import threading
import uvicorn

def start_server(port):
    # Set env vars BEFORE importing backend
    os.environ['MARKET_SIM_PORT'] = str(port)
    os.environ['MARKET_SIM_BUNDLE_DIR'] = str(get_bundle_dir())
    os.environ['MARKET_SIM_DATA_DIR'] = str(get_app_data_dir())

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )

server_thread = threading.Thread(target=start_server, args=(port,), daemon=True)
server_thread.start()
```

### Readiness polling + pywebview launch

```python
import urllib.request
import webview

def wait_for_server(port, timeout=30):
    import time
    url = f"http://127.0.0.1:{port}/api/isos"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(0.3)
    return False

# Launch
wait_for_server(port)
window = webview.create_window(
    "Market Simulator",
    f"http://127.0.0.1:{port}",
    width=1400, height=900, min_size=(1024, 700),
)
webview.start()  # Blocks until window closes
os._exit(0)      # Kill server thread + process
```

---

## 2. Directory Layout (Installed)

```
MarketSimulator/
├── MarketSimulator.exe          (Windows) / MarketSimulator.app (macOS)
├── _internal/                   ← PyInstaller bundle (READ-ONLY)
│   └── market-simulator/
│       ├── frontend/            ← HTML, CSS, JS, brand-assets
│       ├── scripts/             ← Python simulation engine modules
│       ├── backend/             ← FastAPI app + models
│       └── data/                ← Bundled defaults only:
│           ├── profiles/        ←   synthetic demand/gen/fossil/interchange
│           ├── CEG_fleet_rosetta.csv
│           ├── constellation_fleet.json
│           ├── cv_reference_results.json
│           └── cv_comparison_report.json
│
└── app_data/                    ← User-writable (created on first run)
    ├── data/
    │   ├── eia-860/             ← USER DROPS IN: state subdirs with JSON
    │   ├── eia-923/             ← USER DROPS IN: state/month JSON files
    │   ├── epa-campd/           ← USER DROPS IN: state/month JSON files
    │   ├── profiles/            ← Copied from bundle on first run
    │   ├── plant_heat_rates.json  ← Auto-cached; regenerated when source data changes
    │   ├── constellation_fleet.json
    │   └── CEG_fleet_rosetta.csv
    ├── custom-user-inputs/       ← Templates copied from bundle + user overrides
    │   ├── readme.txt
    │   ├── template_lmp_hourly.csv
    │   ├── template_capacity_prices.csv
    │   ├── template_fuel_prices_*.csv
    │   ├── template_rec_prices.csv
    │   └── (user drops: lmp_hourly.csv, etc.)
    └── results/
        ├── sweep_1215/          ← Persists across runs once generated
        │   ├── sweep_1215_flat.parquet
        │   └── sweep_1215_aggregates.json
        ├── fleet_results.json
        ├── fleet_scenario_results.json
        └── sensitivity/
```

### Key principles

- **EIA 860/923 and EPA CAMPD data is NOT bundled** in the executable. Users place their local copies in `app_data/data/` subdirectories. The app ships with synthetic defaults.
- **`custom-user-inputs/`** templates are copied from the bundle to `app_data/` on first run so users can edit them.
- **`results/`** is always in `app_data/` so simulation outputs persist across app restarts.
- **`sweep_1215/`** results persist once generated (first-run generation takes ~10-30 min).

---

## 3. Data Path Resolution

### Two-tier lookup

All data-loading functions resolve paths in this order:

1. **`app_data/data/{path}`** — user-provided data (highest priority)
2. **`_internal/market-simulator/data/{path}`** — bundled defaults (fallback)

Implementation: Set `MARKET_SIM_DATA_DIR` env var in `desktop_app.py`. Modify scripts to check this directory first:

```python
import os
from pathlib import Path

def resolve_data_path(relative_path):
    """Check user data dir first, fall back to bundled data."""
    user_dir = os.environ.get('MARKET_SIM_DATA_DIR')
    if user_dir:
        user_path = Path(user_dir) / "data" / relative_path
        if user_path.exists():
            return user_path

    bundle_dir = os.environ.get('MARKET_SIM_BUNDLE_DIR')
    if bundle_dir:
        bundle_path = Path(bundle_dir) / "data" / relative_path
        if bundle_path.exists():
            return bundle_path

    # Dev mode fallback
    return Path(__file__).resolve().parent.parent / "data" / relative_path
```

### Files to modify

| File | What changes |
|------|-------------|
| `backend/main.py` | Read `MARKET_SIM_BUNDLE_DIR` / `MARKET_SIM_DATA_DIR` env vars; patch `MARKET_SIM_ROOT`, `FRONTEND_DIR`, `RESULTS_DIR`, `CUSTOM_INPUTS_DIR` |
| `scripts/market_simulation.py` | `load_common_data()` uses `resolve_data_path()` |
| `scripts/dispatch_utils.py` | Data loading functions use `resolve_data_path()` |
| `scripts/lmp_engine.py` | If it loads files, use `resolve_data_path()` |
| `scripts/generate_plant_heat_rates.py` | Accept `--data-dir` / `--output` CLI args |
| `scripts/generate_synthetic_profiles.py` | Accept `--data-dir` / `--output` CLI args |

### Heat rate caching

- On startup, `desktop_app.py` checks `app_data/data/plant_heat_rates.json`
- If missing, or if `eia-923/` or `epa-campd/` directories have files newer than the cache, auto-regenerate
- Uses existing `generate_plant_heat_rates.py` with configurable input/output paths

### Sweep persistence

- `results/sweep_1215/` lives in `app_data/results/sweep_1215/`
- On startup, if the sweep parquet doesn't exist, the app can optionally generate it (same as `run.bat` does today)
- Once generated, it persists indefinitely

---

## 4. PyInstaller Spec: `market_simulator.spec`

```python
# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_submodules

# Collect all pyarrow data files and binaries
pyarrow_datas, pyarrow_binaries, pyarrow_hiddenimports = collect_all('pyarrow')

# Collect numba submodules
numba_hiddenimports = collect_submodules('numba')

a = Analysis(
    ['desktop_app.py'],
    pathex=['scripts', 'backend'],
    datas=[
        # Frontend (HTML, CSS, JS, images)
        ('frontend', 'market-simulator/frontend'),
        # Backend (FastAPI app)
        ('backend', 'market-simulator/backend'),
        # Scripts (simulation engine)
        ('scripts', 'market-simulator/scripts'),
        # Bundled default data (NOT EIA/CAMPD — just synthetic + fleet)
        ('data/profiles', 'market-simulator/data/profiles'),
        ('data/CEG_fleet_rosetta.csv', 'market-simulator/data'),
        ('data/constellation_fleet.json', 'market-simulator/data'),
        ('data/cv_reference_results.json', 'market-simulator/data'),
        ('data/cv_comparison_report.json', 'market-simulator/data'),
        ('data/DATA_README.md', 'market-simulator/data'),
        # Custom user input templates
        ('custom-user-inputs', 'market-simulator/custom-user-inputs'),
        # Pre-computed results (if they exist at build time)
        ('results/sweep_1215', 'market-simulator/results/sweep_1215'),
        ('results/fleet_results.json', 'market-simulator/results'),
        ('results/fleet_scenario_results.json', 'market-simulator/results'),
    ] + pyarrow_datas,
    binaries=pyarrow_binaries,
    hiddenimports=[
        # --- uvicorn internals ---
        'uvicorn', 'uvicorn.logging',
        'uvicorn.loops', 'uvicorn.loops.auto', 'uvicorn.loops.asyncio',
        'uvicorn.protocols', 'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.http.h11_impl',
        'uvicorn.protocols.http.httptools_impl',
        'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.protocols.websockets.wsproto_impl',
        'uvicorn.lifespan', 'uvicorn.lifespan.on', 'uvicorn.lifespan.off',
        # --- httptools / websockets ---
        'httptools', 'httptools.parser', 'httptools.parser.parser',
        'websockets', 'wsproto',
        # --- fastapi / starlette ---
        'fastapi', 'fastapi.routing', 'fastapi.middleware',
        'starlette', 'starlette.routing',
        'starlette.middleware', 'starlette.middleware.cors',
        'starlette.responses', 'starlette.staticfiles',
        'starlette.templating', 'starlette.formparsers',
        'anyio', 'anyio._backends', 'anyio._backends._asyncio',
        'sniffio',
        # --- pydantic v2 ---
        'pydantic', 'pydantic_core', 'pydantic.deprecated',
        'annotated_types', 'typing_extensions',
        # --- data science ---
        'numpy', 'pandas', 'scipy', 'scipy.optimize',
        'openpyxl', 'jinja2',
        # --- numba + llvmlite ---
        'llvmlite', 'llvmlite.binding',
        # --- pywebview ---
        'webview',
        # --- project modules ---
        'backend', 'backend.main', 'backend.models',
        'market_simulation', 'lmp_engine', 'pipeline_config',
        'dispatch_utils', 'fleet_model', 'fleet_dispatch',
        'scenario_common', 'procurement_utils', 'eia_data_io',
        'sensitivity_analysis',
        'generate_synthetic_profiles', 'generate_plant_heat_rates',
    ] + pyarrow_hiddenimports + numba_hiddenimports,
    excludes=['tkinter', 'matplotlib', 'IPython', 'notebook', 'pytest'],
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    name='MarketSimulator',
    console=False,               # No terminal window
    icon='frontend/brand-assets/favicon.ico',
    exclude_binaries=True,       # Required for COLLECT (folder mode)
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name='MarketSimulator',
)
```

### Estimated bundle size: ~350-500MB

| Component | Size |
|-----------|------|
| Numba + llvmlite | ~150MB |
| NumPy + SciPy + Pandas + PyArrow | ~200MB |
| Python runtime | ~30MB |
| App code + bundled data | ~30MB |

---

## 5. Known Gotchas & Mitigations

### 5.1 Numba JIT in frozen apps

**Problem**: Numba compiles code at runtime via LLVM. Cache directory and source paths change in frozen builds.

**Mitigation**: Set `NUMBA_CACHE_DIR` to `app_data/.numba_cache/` in `desktop_app.py` before any imports. First run is slower (JIT compilation); subsequent runs use the cache.

**Fallback**: If Numba causes persistent issues, set `NUMBA_DISABLE_JIT=1`. The codebase has numpy fallbacks — 10-50x slower but functional.

### 5.2 PyInstaller + uvicorn dynamic imports

**Problem**: uvicorn discovers its event loop and protocol implementations via dynamic imports at runtime.

**Mitigation**: Comprehensive `hiddenimports` list in the spec file (see above). Use `collect_submodules('uvicorn')` if the manual list proves insufficient.

### 5.3 PyInstaller + pyarrow native binaries

**Problem**: pyarrow includes C++ libraries that PyInstaller sometimes misses, causing `ImportError` at runtime.

**Mitigation**: Use `collect_all('pyarrow')` to gather all data files, binaries, and hidden imports.

### 5.4 Relative imports (`from .models import ...`)

**Problem**: `backend/main.py` uses `from .models import ...` which requires `backend` to be a recognized package.

**Mitigation**: Ensure `backend/__init__.py` exists. Add `'backend'` to `pathex` in the spec.

### 5.5 `__file__` in frozen mode

**Problem**: `Path(__file__).resolve().parent` in `main.py` points inside the PyInstaller temp directory, not the exe's directory.

**Mitigation**: `desktop_app.py` sets `MARKET_SIM_BUNDLE_DIR` and `MARKET_SIM_DATA_DIR` environment variables before importing the backend. `main.py` reads these instead of relying on `__file__`.

### 5.6 Port conflicts

**Problem**: Port 8000 may already be in use by another application.

**Mitigation**: `desktop_app.py` binds to port 0 to discover a free port, then passes it to uvicorn and constructs the pywebview URL with it.

### 5.7 Windows Defender / antivirus false positives

**Problem**: PyInstaller-built executables are frequently flagged by Windows Defender.

**Mitigation**: Code-sign the `.exe` for external distribution. For internal use, document how to add an antivirus exclusion.

### 5.8 macOS Gatekeeper

**Problem**: Unsigned `.app` bundles are blocked by macOS Gatekeeper.

**Mitigation**: Sign with `codesign` and optionally notarize via `xcrun notarytool`. For internal use, users can right-click → Open to bypass.

### 5.9 pywebview platform backends

**Problem**: pywebview uses different rendering backends per OS — Edge/WebView2 on Windows, WebKit on macOS.

**Mitigation**: On Windows, pywebview requires Edge WebView2 Runtime (pre-installed on Windows 10+). For older systems, bundle the WebView2 bootstrapper or fall back to MSHTML.

---

## 6. Build Instructions (Per-Session Prompts)

### Session 1: Entry point + path resolution

```
Tasks:
1. Create market-simulator/desktop_app.py with:
   - is_frozen(), get_bundle_dir(), get_app_data_dir() path helpers
   - first_run_setup() — creates app_data/ tree, copies templates from bundle
   - start_server(port) — daemon thread running uvicorn
   - wait_for_server(port) — polls /api/isos, 30s timeout
   - pywebview window launch (1400x900, min 1024x700)
   - os._exit(0) on window close

2. Modify market-simulator/backend/main.py:
   - At top of module, read MARKET_SIM_BUNDLE_DIR and MARKET_SIM_DATA_DIR env vars
   - If set, override MARKET_SIM_ROOT, FRONTEND_DIR, SCRIPTS_DIR, RESULTS_DIR, CUSTOM_INPUTS_DIR
   - Ensure RESULTS_DIR points to app_data/results/ when frozen

3. Add to app-startup/requirements.txt:
   - pywebview>=5.0
   - pyinstaller>=6.0

4. Test in dev mode:
   cd market-simulator && python desktop_app.py
   → Should open native window showing the guide page
```

### Session 2: Configurable data paths

```
Tasks:
1. Create a shared resolve_data_path() utility (new file or add to existing utils)
   - Checks MARKET_SIM_DATA_DIR/data/ first
   - Falls back to MARKET_SIM_BUNDLE_DIR/data/
   - Falls back to local data/ (dev mode)

2. Modify these files to use resolve_data_path():
   - scripts/market_simulation.py — load_common_data(), load_egrid_baselines()
   - scripts/dispatch_utils.py — all data loading
   - scripts/lmp_engine.py — if it reads data files
   - scripts/generate_plant_heat_rates.py — add --data-dir, --output args
   - scripts/generate_synthetic_profiles.py — add --data-dir, --output args

3. Add heat rate staleness detection in desktop_app.py:
   - Compare mtime of eia-923/ and epa-campd/ dirs vs plant_heat_rates.json
   - Auto-regenerate if source data is newer

4. Test:
   - Set MARKET_SIM_DATA_DIR to a temp directory
   - Place test EIA-860 data in it
   - Verify the app loads from the override directory
```

### Session 3: PyInstaller build

```
Tasks:
1. Create market-simulator/market_simulator.spec (see section 4 above)
2. pip install pyinstaller pywebview
3. Build:
   cd market-simulator
   pyinstaller market_simulator.spec
4. Test dist/MarketSimulator/MarketSimulator.exe:
   - Double-click → native window opens (no terminal, no browser)
   - app_data/ directory created automatically
   - Run a single simulation → results returned
   - Navigate all 6 pages
   - Check custom-user-inputs/ templates copied
5. Fix missing hidden imports (iterate on spec until clean)
6. Test sweep: delete app_data/results/sweep_1215/ → verify auto-generation
```

### Session 4: Polish + cross-platform

```
Tasks:
1. Add app icon:
   - Convert brand-assets logo to .ico (Windows) and .icns (macOS)
   - Reference in spec file

2. Loading splash:
   - Show a simple pywebview window with "Starting Market Simulator..." while server boots
   - Navigate to actual app once /api/isos responds

3. macOS build:
   - Run same spec on macOS (or adjust for .app bundle)
   - Test Gatekeeper behavior
   - Document signing if needed

4. End-to-end testing on clean machines:
   - Windows: fresh VM with no Python → double-click exe → full workflow
   - macOS: fresh user account → open .app → full workflow
   - Data drop-in: EIA → heat rates regenerate
   - Custom CSV: detected on setup page
   - Results + sweep persist across restarts

5. Update USER_MANUAL.md with desktop app section
```

---

## 7. Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `market-simulator/desktop_app.py` | **CREATE** | Entry point: pywebview + uvicorn thread + path resolution |
| `market-simulator/market_simulator.spec` | **CREATE** | PyInstaller build configuration |
| `market-simulator/backend/main.py` | **MODIFY** | Add env-var-based path resolution for frozen mode |
| `market-simulator/scripts/market_simulation.py` | **MODIFY** | Use `resolve_data_path()` in `load_common_data()` |
| `market-simulator/scripts/dispatch_utils.py` | **MODIFY** | Use `resolve_data_path()` in data loading |
| `market-simulator/scripts/generate_plant_heat_rates.py` | **MODIFY** | Add `--data-dir` / `--output` CLI args |
| `market-simulator/scripts/generate_synthetic_profiles.py` | **MODIFY** | Add `--data-dir` / `--output` CLI args |
| `market-simulator/app-startup/requirements.txt` | **MODIFY** | Add `pywebview>=5.0`, `pyinstaller>=6.0` |

---

## 8. Verification Checklist

- [ ] `python desktop_app.py` launches native window in dev mode
- [ ] PyInstaller build completes without errors
- [ ] Built exe launches on clean Windows (no Python installed)
- [ ] Built .app launches on clean macOS
- [ ] Native window shows guide.html on startup
- [ ] All 6 pages navigate correctly
- [ ] Single simulation runs and returns results
- [ ] Parametric sweep works (cached or generated on first run)
- [ ] Fleet config page loads 146 plants
- [ ] `app_data/` directory auto-created on first run
- [ ] `custom-user-inputs/` templates copied from bundle
- [ ] Custom CSV drop-in detected on setup page toggle
- [ ] EIA 860/923/CAMPD data drop-in loads correctly
- [ ] Heat rates auto-regenerate when source data changes
- [ ] Results persist across app restarts
- [ ] Sweep results persist across app restarts
- [ ] No console/terminal window visible
- [ ] App shuts down cleanly when window is closed
- [ ] Port conflict handled (finds free port)
- [ ] Numba JIT cache persists across restarts (faster second launch)
- [ ] No antivirus false positives (or documented workaround)
