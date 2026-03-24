"""
Canonical path resolution for both development and PyInstaller-frozen modes.

All modules should import from here instead of computing their own paths.
Supports three-tier lookup for data files:
  1. MARKET_SIM_DATA_DIR/data/  (user-writable, highest priority)
  2. MARKET_SIM_BUNDLE_DIR/data/ (read-only bundled defaults)
  3. Local data/ relative to project root (dev mode fallback)
"""

import os
import sys
from pathlib import Path


def _is_frozen():
    return getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')


if _is_frozen():
    BUNDLE_DIR = Path(sys._MEIPASS) / "market-simulator"
    APP_DIR = Path(sys.executable).parent
    PARENT_PROJECT_DIR = None  # No parent project in frozen mode
else:
    BUNDLE_DIR = Path(__file__).resolve().parent
    APP_DIR = BUNDLE_DIR
    # Dev mode: parent project may have pipeline output data (step2.1-ef, step2.2-cost, etc.)
    _candidate = BUNDLE_DIR.parent / "data"
    PARENT_PROJECT_DIR = _candidate if _candidate.is_dir() else None

# Read-only (bundled) paths
BACKEND_DIR = BUNDLE_DIR / "backend"
SCRIPTS_DIR = BUNDLE_DIR / "scripts"
FRONTEND_DIR = BUNDLE_DIR / "frontend"

# Module root alias (backward compat for scripts that use MODULE_ROOT)
MODULE_ROOT = BUNDLE_DIR


def resolve_data_path(relative_path):
    """Resolve a data file path using three-tier lookup.

    Args:
        relative_path: Path relative to the data/ directory, e.g.
            "profiles/eia_demand_profiles.json" or "constellation_fleet.json"

    Returns:
        Path object to the resolved file. If the file doesn't exist in any
        tier, returns the dev-mode path (allows callers to raise their own
        FileNotFoundError with context).
    """
    relative_path = str(relative_path)

    # Tier 1: User-writable data dir (MARKET_SIM_DATA_DIR/data/)
    user_dir = os.environ.get('MARKET_SIM_DATA_DIR')
    if user_dir:
        user_path = Path(user_dir) / "data" / relative_path
        if user_path.exists():
            return user_path

    # Tier 2: Bundled data dir (MARKET_SIM_BUNDLE_DIR/data/)
    bundle_dir = os.environ.get('MARKET_SIM_BUNDLE_DIR')
    if bundle_dir:
        bundle_path = Path(bundle_dir) / "data" / relative_path
        if bundle_path.exists():
            return bundle_path

    # Tier 3: Dev mode fallback (project root / data/)
    dev_path = BUNDLE_DIR / "data" / relative_path
    if dev_path.exists():
        return dev_path

    # Tier 4: Parent project data (dev mode only — pipeline outputs like step2.1-ef/)
    if PARENT_PROJECT_DIR:
        parent_path = PARENT_PROJECT_DIR / relative_path
        if parent_path.exists():
            return parent_path

    return dev_path  # Return dev path for callers to raise their own error


def resolve_data_dir(relative_dir=""):
    """Resolve a data directory path using three-tier lookup.

    Similar to resolve_data_path but checks for directory existence.
    Returns the first existing directory, or the dev-mode path as fallback.
    """
    relative_dir = str(relative_dir)

    user_dir = os.environ.get('MARKET_SIM_DATA_DIR')
    if user_dir:
        user_path = Path(user_dir) / "data" / relative_dir
        if user_path.is_dir():
            return user_path

    bundle_dir = os.environ.get('MARKET_SIM_BUNDLE_DIR')
    if bundle_dir:
        bundle_path = Path(bundle_dir) / "data" / relative_dir
        if bundle_path.is_dir():
            return bundle_path

    dev_path = BUNDLE_DIR / "data" / relative_dir
    if dev_path.is_dir():
        return dev_path

    # Tier 4: Parent project data (dev mode only)
    if PARENT_PROJECT_DIR:
        parent_path = PARENT_PROJECT_DIR / relative_dir
        if parent_path.is_dir():
            return parent_path

    return dev_path


def get_data_search_dirs(subdirs=None):
    """Get ordered list of directories to search for data files.

    Args:
        subdirs: Optional list of subdirectory names within data/ to include
            (e.g., ["profiles", "eia-930"]). If None, returns just the data roots.

    Returns:
        List of Path objects to search, ordered by priority (user > bundle > dev).
    """
    roots = []

    user_dir = os.environ.get('MARKET_SIM_DATA_DIR')
    if user_dir:
        roots.append(Path(user_dir) / "data")

    bundle_dir = os.environ.get('MARKET_SIM_BUNDLE_DIR')
    if bundle_dir:
        bundle_path = Path(bundle_dir) / "data"
        if bundle_path not in roots:
            roots.append(bundle_path)

    dev_path = BUNDLE_DIR / "data"
    if dev_path not in roots:
        roots.append(dev_path)

    # Tier 4: Parent project data (dev mode only)
    if PARENT_PROJECT_DIR and PARENT_PROJECT_DIR not in roots:
        roots.append(PARENT_PROJECT_DIR)

    if subdirs is None:
        return [r for r in roots if r.exists()]

    result = []
    for root in roots:
        for sub in subdirs:
            d = root / sub
            if d.exists():
                result.append(d)
    return result
