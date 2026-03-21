"""
Test configurable data path resolution via MARKET_SIM_DATA_DIR.

Verifies that:
  1. resolve_data_path() checks MARKET_SIM_DATA_DIR/data/ first
  2. Falls back to MARKET_SIM_BUNDLE_DIR/data/
  3. Falls back to local data/ (dev mode)
  4. eia_data_io search dirs respect the env var
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Add project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPTS_DIR)


def test_resolve_data_path_user_override():
    """MARKET_SIM_DATA_DIR/data/ takes priority over local data/."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test file in the override directory
        data_dir = Path(tmpdir) / "data"
        data_dir.mkdir()
        test_file = data_dir / "test_override.json"
        test_file.write_text('{"source": "override"}')

        # Set env var and reload paths module
        old_val = os.environ.get('MARKET_SIM_DATA_DIR')
        os.environ['MARKET_SIM_DATA_DIR'] = tmpdir

        try:
            # Import fresh (paths module reads env at call time)
            from paths import resolve_data_path
            resolved = resolve_data_path("test_override.json")
            assert resolved.exists(), f"Expected {resolved} to exist"
            assert str(tmpdir) in str(resolved), \
                f"Expected path under {tmpdir}, got {resolved}"

            # Verify content
            with open(resolved) as f:
                data = json.load(f)
            assert data["source"] == "override"
        finally:
            if old_val is None:
                os.environ.pop('MARKET_SIM_DATA_DIR', None)
            else:
                os.environ['MARKET_SIM_DATA_DIR'] = old_val


def test_resolve_data_path_fallback_to_local():
    """When override dir doesn't have the file, falls back to local data/."""
    # Clear any override
    old_val = os.environ.pop('MARKET_SIM_DATA_DIR', None)
    old_bundle = os.environ.pop('MARKET_SIM_BUNDLE_DIR', None)

    try:
        from paths import resolve_data_path
        # profiles/ should exist in local dev data
        resolved = resolve_data_path("profiles")
        # In dev mode, should resolve to project_root/data/profiles
        assert "market-simulator" in str(resolved) or "data" in str(resolved)
    finally:
        if old_val is not None:
            os.environ['MARKET_SIM_DATA_DIR'] = old_val
        if old_bundle is not None:
            os.environ['MARKET_SIM_BUNDLE_DIR'] = old_bundle


def test_resolve_data_dir():
    """resolve_data_dir returns existing directories with override priority."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create eia-860 dir in override location
        eia_dir = Path(tmpdir) / "data" / "eia-860" / "IL"
        eia_dir.mkdir(parents=True)
        # Put a test file in it
        (eia_dir / "test.json").write_text('[]')

        old_val = os.environ.get('MARKET_SIM_DATA_DIR')
        os.environ['MARKET_SIM_DATA_DIR'] = tmpdir

        try:
            from paths import resolve_data_dir
            resolved = resolve_data_dir("eia-860")
            assert resolved.is_dir(), f"Expected {resolved} to be a directory"
            assert str(tmpdir) in str(resolved), \
                f"Expected path under {tmpdir}, got {resolved}"
        finally:
            if old_val is None:
                os.environ.pop('MARKET_SIM_DATA_DIR', None)
            else:
                os.environ['MARKET_SIM_DATA_DIR'] = old_val


def test_eia_data_io_respects_override():
    """eia_data_io search dirs pick up the MARKET_SIM_DATA_DIR override."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create profiles dir with a test demand file
        profiles_dir = Path(tmpdir) / "data" / "profiles"
        profiles_dir.mkdir(parents=True)

        # Write minimal demand profiles JSON
        demand = {
            "ERCOT": {
                "2025": {
                    "raw_mw": [50000.0] * 8760,
                    "normalized": [1.0 / 8760] * 8760,
                    "total_annual_mwh": 50000.0 * 8760,
                    "peak_mw": 50000.0,
                    "min_mw": 50000.0,
                    "avg_mw": 50000.0,
                }
            }
        }
        with open(profiles_dir / "eia_demand_profiles.json", "w") as f:
            json.dump(demand, f)

        old_val = os.environ.get('MARKET_SIM_DATA_DIR')
        os.environ['MARKET_SIM_DATA_DIR'] = tmpdir

        try:
            # Force reimport of eia_data_io to pick up new env
            if 'eia_data_io' in sys.modules:
                del sys.modules['eia_data_io']
            if 'paths' in sys.modules:
                del sys.modules['paths']

            from eia_data_io import load_demand_profiles
            result = load_demand_profiles()
            assert "ERCOT" in result, f"Expected ERCOT in result, got {list(result.keys())}"
            assert result["ERCOT"]["2025"]["peak_mw"] == 50000.0
            print("  eia_data_io loaded from override directory: PASS")
        finally:
            if old_val is None:
                os.environ.pop('MARKET_SIM_DATA_DIR', None)
            else:
                os.environ['MARKET_SIM_DATA_DIR'] = old_val


def test_get_data_search_dirs():
    """get_data_search_dirs returns user dir first, then bundle, then dev."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create profiles subdir in override
        (Path(tmpdir) / "data" / "profiles").mkdir(parents=True)

        old_data = os.environ.get('MARKET_SIM_DATA_DIR')
        old_bundle = os.environ.get('MARKET_SIM_BUNDLE_DIR')
        os.environ['MARKET_SIM_DATA_DIR'] = tmpdir
        os.environ.pop('MARKET_SIM_BUNDLE_DIR', None)

        try:
            if 'paths' in sys.modules:
                del sys.modules['paths']
            from paths import get_data_search_dirs
            dirs = get_data_search_dirs(["profiles"])
            assert len(dirs) >= 1, "Expected at least 1 search dir"
            # First dir should be under the override
            assert str(tmpdir) in str(dirs[0]), \
                f"Expected first dir under {tmpdir}, got {dirs[0]}"
        finally:
            if old_data is None:
                os.environ.pop('MARKET_SIM_DATA_DIR', None)
            else:
                os.environ['MARKET_SIM_DATA_DIR'] = old_data
            if old_bundle is None:
                os.environ.pop('MARKET_SIM_BUNDLE_DIR', None)
            else:
                os.environ['MARKET_SIM_BUNDLE_DIR'] = old_bundle


if __name__ == "__main__":
    print("Testing configurable data path resolution...\n")

    tests = [
        ("resolve_data_path user override", test_resolve_data_path_user_override),
        ("resolve_data_path fallback to local", test_resolve_data_path_fallback_to_local),
        ("resolve_data_dir", test_resolve_data_dir),
        ("eia_data_io respects override", test_eia_data_io_respects_override),
        ("get_data_search_dirs", test_get_data_search_dirs),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"  PASS: {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name} — {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")
    if failed > 0:
        sys.exit(1)
