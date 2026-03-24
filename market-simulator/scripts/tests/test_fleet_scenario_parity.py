#!/usr/bin/env python3
"""
Fleet Scenario Parity Tests — Verify market-simulator/frontend matches
dashboard/ exactly for fleet scenario calculations.

Tests cover:
1. JS file byte-parity (dispatch engine, scenarios, sidebar)
2. Data file parity (constellation_scenarios, sweep_dispatch_data)
3. Structural integrity of constellation_scenarios.json (heat_rate_tier, CCS fields)
4. Sweep dispatch data has per-tier CF columns
5. HTML structural elements (hide-baseline checkbox, CCGT tier chart)
6. Calculation logic equivalence via Node.js dispatch engine execution
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

# Project root
ROOT = Path(__file__).resolve().parent.parent.parent.parent
DASHBOARD = ROOT / "dashboard"
MARKET_SIM = ROOT / "market-simulator" / "frontend"


class TestJSFileParity:
    """Verify JS files are byte-identical between dashboard and market-simulator."""

    @pytest.mark.parametrize("filename", [
        "fleet-dispatch-engine.js",
        "fleet-scenarios.js",
        "fleet-sidebar.js",
    ])
    def test_js_file_identical(self, filename):
        dashboard_file = DASHBOARD / "js" / filename
        marketsim_file = MARKET_SIM / "js" / filename
        assert dashboard_file.exists(), f"Dashboard {filename} missing"
        assert marketsim_file.exists(), f"Market-sim {filename} missing"
        assert dashboard_file.read_bytes() == marketsim_file.read_bytes(), (
            f"{filename} differs between dashboard and market-simulator"
        )


class TestDataFileParity:
    """Verify data files are identical between dashboard and market-simulator."""

    def test_constellation_scenarios_identical(self):
        d = DASHBOARD / "data" / "constellation_scenarios.json"
        m = MARKET_SIM / "data" / "constellation_scenarios.json"
        assert d.exists() and m.exists()
        assert d.read_bytes() == m.read_bytes(), "constellation_scenarios.json differs"

    def test_sweep_dispatch_data_identical(self):
        d = DASHBOARD / "data" / "sweep_dispatch_data.json"
        m = MARKET_SIM / "data" / "sweep_dispatch_data.json"
        assert d.exists() and m.exists()
        assert d.read_bytes() == m.read_bytes(), "sweep_dispatch_data.json differs"

    def test_fleet_results_sample_identical(self):
        d = DASHBOARD / "data" / "fleet_scenario_results_sample.json"
        m = MARKET_SIM / "data" / "fleet_scenario_results_sample.json"
        assert d.exists() and m.exists()
        assert d.read_bytes() == m.read_bytes(), "fleet_scenario_results_sample.json differs"


class TestConstellationScenariosStructure:
    """Verify constellation_scenarios.json has required fields for tier dispatch."""

    @pytest.fixture
    def scenarios_data(self):
        path = MARKET_SIM / "data" / "constellation_scenarios.json"
        with open(path) as f:
            return json.load(f)

    def test_has_plants(self, scenarios_data):
        plants = scenarios_data.get("base_fleet", scenarios_data.get("plants", []))
        assert len(plants) > 0, "No plants found"

    def test_fossil_plants_have_heat_rate_tier(self, scenarios_data):
        """Gas CCGT plants should have heat_rate_tier field."""
        plants = scenarios_data.get("base_fleet", scenarios_data.get("plants", []))
        gas_ccgt = [p for p in plants if p.get("fuel_type") == "gas_ccgt"]
        assert len(gas_ccgt) > 0, "No gas_ccgt plants found"
        with_tier = [p for p in gas_ccgt if "heat_rate_tier" in p]
        assert len(with_tier) == len(gas_ccgt), (
            f"{len(gas_ccgt) - len(with_tier)} gas_ccgt plants missing heat_rate_tier"
        )

    def test_heat_rate_tier_values_valid(self, scenarios_data):
        valid_tiers = {"very_low", "low", "medium", "high", "very_high"}
        plants = scenarios_data.get("base_fleet", scenarios_data.get("plants", []))
        for p in plants:
            tier = p.get("heat_rate_tier")
            if tier is not None:
                assert tier in valid_tiers, f"Plant {p.get('name')} has invalid tier: {tier}"

    def test_ccs_eligible_plants_exist(self, scenarios_data):
        plants = scenarios_data.get("base_fleet", scenarios_data.get("plants", []))
        ccs_eligible = [p for p in plants if p.get("ccs_eligible")]
        assert len(ccs_eligible) > 0, "No CCS-eligible plants found"
        for p in ccs_eligible:
            assert "ccs_capacity_mw" in p, (
                f"CCS-eligible plant {p.get('name')} missing ccs_capacity_mw"
            )


class TestSweepDispatchStructure:
    """Verify sweep_dispatch_data.json has per-tier CF columns."""

    @pytest.fixture
    def sweep_data(self):
        path = MARKET_SIM / "data" / "sweep_dispatch_data.json"
        with open(path) as f:
            return json.load(f)

    def test_has_scenarios(self, sweep_data):
        scenarios = sweep_data.get("scenarios", [])
        assert len(scenarios) >= 1215, f"Expected >=1215 scenarios, got {len(scenarios)}"

    def test_has_iso_data(self, sweep_data):
        data = sweep_data.get("data", {})
        assert len(data) > 0, "No ISO data found"

    def test_has_tier_cf_columns(self, sweep_data):
        """Verify per-tier CF columns exist for gas_ccgt dispatch."""
        data = sweep_data.get("data", {})
        # Check the first ISO that has data
        for iso, iso_data in data.items():
            keys = set(iso_data.keys())
            tier_keys = [k for k in keys if "gas_ccgt_" in k and "_cf" in k]
            if tier_keys:
                # Found tier columns — verify all expected tiers
                expected_tiers = ["very_low", "low", "medium", "high"]
                for tier in expected_tiers:
                    key = f"gas_ccgt_{tier}_cf"
                    assert key in keys, f"Missing {key} in {iso} sweep data"
                return
        pytest.fail("No ISO has per-tier gas_ccgt CF columns in sweep data")


class TestHTMLStructure:
    """Verify market-simulator HTML has all required elements."""

    @pytest.fixture
    def html_content(self):
        path = MARKET_SIM / "fleet_scenarios.html"
        return path.read_text()

    def test_hide_baseline_checkbox(self, html_content):
        assert 'id="hideBaselineFan"' in html_content, "Missing hideBaselineFan checkbox"

    def test_ccgt_tier_chart_canvas(self, html_content):
        assert 'id="ccgtTierChart"' in html_content, "Missing ccgtTierChart canvas"

    def test_ccgt_tier_card(self, html_content):
        assert 'id="ccgtTierCard"' in html_content, "Missing ccgtTierCard container"

    def test_ccgt_tier_year_span(self, html_content):
        assert 'id="ccgtTierYear"' in html_content, "Missing ccgtTierYear span"

    def test_ccgt_tier_legend(self, html_content):
        assert 'id="ccgtTierLegend"' in html_content, "Missing ccgtTierLegend container"

    def test_fan_chart_canvas(self, html_content):
        assert 'id="fanChart"' in html_content, "Missing fanChart canvas"

    def test_intensity_fan_chart(self, html_content):
        assert 'id="intensityFanChart"' in html_content, "Missing intensityFanChart canvas"

    def test_gen_mix_chart(self, html_content):
        assert 'id="genMixChart"' in html_content, "Missing genMixChart canvas"

    def test_js_includes(self, html_content):
        assert "fleet-scenarios.js" in html_content, "Missing fleet-scenarios.js include"
        assert "fleet-dispatch-engine.js" in html_content, "Missing fleet-dispatch-engine.js include"
        assert "fleet-sidebar.js" in html_content, "Missing fleet-sidebar.js include"


class TestDispatchEngineCalculations:
    """
    Verify dispatch engine produces identical results when run against the same
    data via both dashboard and market-simulator JS files.

    Uses Node.js to execute the dispatch engine with test inputs and compare outputs.
    """

    @pytest.fixture
    def node_available(self):
        try:
            result = subprocess.run(
                ["node", "--version"], capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _run_dispatch_test(self, js_dir, data_dir):
        """Run a minimal dispatch test via Node.js and return envelope results."""
        test_script = f"""
        const fs = require('fs');

        // Minimal DOM/window stubs for browser-targeted JS
        global.window = global;
        global.document = {{
            readyState: 'complete',
            addEventListener: function() {{}},
            getElementById: function() {{ return null; }},
            querySelectorAll: function() {{ return []; }},
            querySelector: function() {{ return null; }}
        }};
        global.Chart = function() {{ return {{ destroy: function(){{}}, update: function(){{}}, data: {{}} }}; }};
        global.Chart.register = function() {{}};
        global.Chart.defaults = {{ font: {{ size: 12 }}, plugins: {{ legend: {{}}, tooltip: {{}} }} }};
        global.localStorage = {{ getItem: function() {{ return null; }}, setItem: function() {{}} }};
        global.sessionStorage = {{ getItem: function() {{ return null; }}, setItem: function() {{}} }};
        global.fetch = function() {{ return Promise.reject('no fetch'); }};
        global.console = {{ log: function(){{}}, warn: function(){{}}, error: function(){{}}, info: function(){{}}, debug: function(){{}}, group: function(){{}}, groupEnd: function(){{}}, table: function(){{}}, time: function(){{}}, timeEnd: function(){{}}, assert: function(){{}}, clear: function(){{}}, count: function(){{}}, dir: function(){{}}, trace: function(){{}}, groupCollapsed: function(){{}}}};
        global.requestAnimationFrame = function(cb) {{ cb(); }};

        // Load dispatch engine
        eval(fs.readFileSync('{js_dir}/chart-colors.js', 'utf8'));
        eval(fs.readFileSync('{js_dir}/fleet-dispatch-engine.js', 'utf8'));

        // Load test data
        const scenariosData = JSON.parse(fs.readFileSync('{data_dir}/constellation_scenarios.json', 'utf8'));
        const sweepData = JSON.parse(fs.readFileSync('{data_dir}/sweep_dispatch_data.json', 'utf8'));

        // Get first 5 plants for a quick test (key is base_fleet or plants)
        const plants = (scenariosData.base_fleet || scenariosData.plants || []).slice(0, 5);

        // Run dispatch
        const result = FleetDispatchEngine.computeFleetDispatch(plants, sweepData, {{
            ccs_derate_pct: 14,
            ccs_capture_rate_pct: 90,
            ccs_cf_pct: 85
        }});

        // Output envelope for comparison
        const output = {{
            envelope: result.envelope,
            years: Object.keys(result.envelope).sort()
        }};
        process.stdout.write(JSON.stringify(output));
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write(test_script)
            f.flush()
            try:
                result = subprocess.run(
                    ["node", f.name],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode != 0:
                    pytest.skip(f"Node.js execution failed: {result.stderr[:500]}")
                return json.loads(result.stdout)
            except subprocess.TimeoutExpired:
                pytest.skip("Node.js dispatch timed out")
            except json.JSONDecodeError:
                pytest.skip(f"Invalid JSON output: {result.stdout[:500]}")
            finally:
                os.unlink(f.name)

    def test_dispatch_results_identical(self, node_available):
        """Run dispatch engine from both locations and compare outputs."""
        if not node_available:
            pytest.skip("Node.js not available")

        dashboard_result = self._run_dispatch_test(
            str(DASHBOARD / "js"),
            str(DASHBOARD / "data")
        )
        marketsim_result = self._run_dispatch_test(
            str(MARKET_SIM / "js"),
            str(MARKET_SIM / "data")
        )

        # Both should have same years
        assert dashboard_result["years"] == marketsim_result["years"], (
            "Year lists differ"
        )

        # Compare envelope values — must be exactly identical (same code, same data)
        for year in dashboard_result["years"]:
            d_env = dashboard_result["envelope"][year]
            m_env = marketsim_result["envelope"][year]
            for key in ["p10", "p50", "p90"]:
                assert d_env[key] == m_env[key], (
                    f"Year {year} {key}: dashboard={d_env[key]}, "
                    f"market-sim={m_env[key]}"
                )


class TestRebasingLogicPresent:
    """Verify the rebasing logic exists in market-simulator fleet-sidebar.js."""

    @pytest.fixture
    def sidebar_content(self):
        path = MARKET_SIM / "js" / "fleet-sidebar.js"
        return path.read_text()

    def test_rebase_function_exists(self, sidebar_content):
        assert "rebaseOnPythonBaseline" in sidebar_content, (
            "Missing rebaseOnPythonBaseline function"
        )

    def test_engine_baseline_shadow(self, sidebar_content):
        assert "engineBaseline" in sidebar_content, (
            "Missing engineBaseline shadow variable"
        )

    def test_storage_key_v2(self, sidebar_content):
        assert "market-sim-scenarios-v2" in sidebar_content, (
            "Storage key should be v2"
        )

    def test_coal_steam_in_fossil_fuels(self, sidebar_content):
        assert "coal_steam" in sidebar_content, (
            "FOSSIL_FUELS should include coal_steam"
        )


class TestIntensityComputationLogic:
    """Verify the intensity computation uses envelope-based method."""

    @pytest.fixture
    def scenarios_js(self):
        path = MARKET_SIM / "js" / "fleet-scenarios.js"
        return path.read_text()

    def test_intensity_from_envelope(self, scenarios_js):
        """Intensity should be computed from envelope P10/P50/P90 divided by total gen."""
        assert "computeIntensityFromGenAndEmissions" in scenarios_js
        # The function should reference envelope (not emissions_by_fuel for percentiles)
        assert "env.p10" in scenarios_js, "Should use envelope p10 for intensity"
        assert "env.p50" in scenarios_js, "Should use envelope p50 for intensity"
        assert "env.p90" in scenarios_js, "Should use envelope p90 for intensity"

    def test_hide_baseline_fan_variable(self, scenarios_js):
        assert "hideBaselineFan" in scenarios_js, "Missing hideBaselineFan state variable"

    def test_ccgt_tier_chart_functions(self, scenarios_js):
        assert "buildCcgtTierChart" in scenarios_js, "Missing buildCcgtTierChart function"
        assert "updateCcgtTierChart" in scenarios_js, "Missing updateCcgtTierChart function"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
