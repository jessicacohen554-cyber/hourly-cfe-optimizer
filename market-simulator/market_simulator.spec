# -*- mode: python ; coding: utf-8 -*-
import sys
from PyInstaller.utils.hooks import collect_all, collect_submodules

# Platform-specific icon
if sys.platform == 'darwin':
    APP_ICON = 'frontend/brand-assets/app.icns'
else:
    APP_ICON = 'frontend/brand-assets/favicon.ico'

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
        # Pre-computed results
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
        'zonal_lmp', 'build_fleet_scenario_data',
        'generate_constellation_scenarios',
        'run_cv_simulations', 'run_sweep_405',
        'validate_plant_retirement',
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
    icon=APP_ICON,
    exclude_binaries=True,       # Required for COLLECT (folder mode)
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name='MarketSimulator',
)

# macOS .app bundle
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='MarketSimulator.app',
        icon=APP_ICON,
        bundle_identifier='com.constellation.marketsimulator',
        info_plist={
            'CFBundleShortVersionString': '1.0.0',
            'CFBundleName': 'Market Simulator',
            'NSHighResolutionCapable': True,
            'LSMinimumSystemVersion': '10.15',
        },
    )
