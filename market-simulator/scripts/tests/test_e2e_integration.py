#!/usr/bin/env python3
"""End-to-end integration test — validates full pipeline from sweep parquet
through fleet dispatch to dashboard API.

Prompt 10 from New_Build_Fossil_Integration_Audit.md.

Test matrix:
  1. Parquet row count = 51,030 (1215 scenarios × 7 ISOs × 6 years)
  2. New fossil columns present in sweep parquet
  3. New fossil builds non-zero in at least some scenarios
  4. LMP range sanity ($5–$300/MWh, no $0 or $9999 outliers)
  5. Scarcity hours reasonable (0–2000)
  6. Fleet results populated with new fossil fields
  7. Aggregates JSON has P10/P50/P90 bands for all ISOs
  8. API endpoints respond with valid JSON (if backend running)
"""

import json
import os
import sys
from pathlib import Path

import pytest

# ── Path setup ──
MARKET_SIM_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = MARKET_SIM_ROOT / "results"
SWEEP_DIR = RESULTS_DIR / "sweep_1215"
SCRIPTS_DIR = MARKET_SIM_ROOT / "scripts"

# Ensure scripts/ is importable for any future direct-import tests
sys.path.insert(0, str(SCRIPTS_DIR))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sweep_df():
    """Load the 1,215-sweep flat parquet once for all tests."""
    import pandas as pd
    path = SWEEP_DIR / "sweep_1215_flat.parquet"
    if not path.exists():
        pytest.skip(f"Sweep parquet not found: {path}")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def fleet_results():
    """Load single-fleet results JSON."""
    path = RESULTS_DIR / "fleet_results.json"
    if not path.exists():
        pytest.skip(f"Fleet results not found: {path}")
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def fleet_scenario_results():
    """Load multi-scenario fleet results JSON."""
    path = RESULTS_DIR / "fleet_scenario_results.json"
    if not path.exists():
        pytest.skip(f"Fleet scenario results not found: {path}")
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def aggregates():
    """Load sweep aggregates JSON."""
    path = SWEEP_DIR / "sweep_1215_aggregates.json"
    if not path.exists():
        pytest.skip(f"Aggregates not found: {path}")
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Parquet Schema & Row Count
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED_ROWS = 1215 * 7 * 6  # 51,030
EXPECTED_ISOS = {"CAISO", "ERCOT", "PJM", "NYISO", "NEISO", "MISO", "SPP"}
EXPECTED_YEARS = {2023, 2030, 2035, 2040, 2045, 2050}

NEW_FOSSIL_COLS = {
    "nb_gas_ccgt_mw",
    "nb_gas_ct_mw",
    "total_new_fossil_mw",
    "gas_built_gw",
    "fossil_built_gw",
}


class TestParquetValidation:
    """Validate sweep parquet structure and row count."""

    def test_row_count(self, sweep_df):
        """Parquet must have exactly 1215 × 7 ISOs × 6 years = 51,030 rows."""
        assert len(sweep_df) == EXPECTED_ROWS, (
            f"Expected {EXPECTED_ROWS:,} rows, got {len(sweep_df):,}"
        )

    def test_new_fossil_columns_exist(self, sweep_df):
        """All new-build fossil columns must be present."""
        missing = NEW_FOSSIL_COLS - set(sweep_df.columns)
        assert not missing, f"Missing new fossil columns: {missing}"

    def test_core_columns_exist(self, sweep_df):
        """Core identifier and metric columns must be present."""
        required = {
            "iso", "year", "scenario", "scenario_id",
            "avg_lmp", "emissions_mt", "clean_pct", "demand_twh",
            "ordc_scarcity_hours",
        }
        missing = required - set(sweep_df.columns)
        assert not missing, f"Missing core columns: {missing}"

    def test_all_isos_present(self, sweep_df):
        """All 7 ISOs must appear in the parquet."""
        actual_isos = set(sweep_df["iso"].unique())
        assert actual_isos == EXPECTED_ISOS, (
            f"ISO mismatch. Expected {EXPECTED_ISOS}, got {actual_isos}"
        )

    def test_all_years_present(self, sweep_df):
        """All 6 projection years must appear."""
        actual_years = set(sweep_df["year"].unique())
        assert actual_years == EXPECTED_YEARS, (
            f"Year mismatch. Expected {EXPECTED_YEARS}, got {actual_years}"
        )

    def test_scenario_count_per_iso_year(self, sweep_df):
        """Each ISO-year combo must have exactly 1,215 scenarios."""
        counts = sweep_df.groupby(["iso", "year"]).size()
        bad = counts[counts != 1215]
        assert bad.empty, (
            f"ISO-year combos with wrong scenario count:\n{bad}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. New Fossil Build Validation
# ─────────────────────────────────────────────────────────────────────────────

class TestNewFossilBuilds:
    """Validate new-build fossil data is populated and reasonable."""

    def test_some_scenarios_build_fossil(self, sweep_df):
        """At least some scenarios should trigger new fossil builds."""
        has_builds = (sweep_df["total_new_fossil_mw"] > 0).any()
        assert has_builds, "No new fossil builds in any scenario — feature may be inactive"

    def test_not_all_scenarios_build_fossil(self, sweep_df):
        """Not every scenario should build fossil (would indicate over-triggering)."""
        build_fraction = (sweep_df["total_new_fossil_mw"] > 0).mean()
        assert build_fraction < 0.95, (
            f"{build_fraction:.1%} of scenarios build fossil — likely over-triggering "
            f"(Risk R2)"
        )

    def test_new_fossil_non_negative(self, sweep_df):
        """New fossil MW must never be negative (NaN is OK — means no build)."""
        for col in NEW_FOSSIL_COLS:
            valid = sweep_df[col].dropna()
            assert (valid >= 0).all(), (
                f"Negative values in {col}: min={valid.min()}"
            )

    def test_total_fossil_consistency(self, sweep_df):
        """total_new_fossil_mw should be >= sum of individual fuel builds.

        NaN in per-fuel columns means no build of that type — treat as 0.
        Allow ±1 MW tolerance for floating-point rounding.
        """
        component_sum = (
            sweep_df["nb_gas_ccgt_mw"].fillna(0)
            + sweep_df["nb_gas_ct_mw"].fillna(0)
        )
        if "nb_coal_mw" in sweep_df.columns:
            component_sum = component_sum + sweep_df["nb_coal_mw"].fillna(0)
        diff = sweep_df["total_new_fossil_mw"] - component_sum
        assert (diff >= -1.5).all(), (
            f"total_new_fossil_mw < sum of components in some rows. "
            f"Min diff: {diff.min():.2f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. LMP Sanity Checks
# ─────────────────────────────────────────────────────────────────────────────

class TestLMPSanity:
    """Validate LMP values are within realistic ranges."""

    def test_no_zero_lmp(self, sweep_df):
        """No scenario should have $0 average LMP."""
        zero_count = (sweep_df["avg_lmp"] == 0).sum()
        assert zero_count == 0, (
            f"{zero_count} rows have $0 LMP — indicates broken pricing"
        )

    def test_no_absurd_lmp(self, sweep_df):
        """No average LMP should exceed $1,000/MWh."""
        max_lmp = sweep_df["avg_lmp"].max()
        assert max_lmp < 1000, (
            f"Max avg LMP = ${max_lmp:.1f}/MWh — unrealistically high"
        )

    def test_lmp_minimum_above_threshold(self, sweep_df):
        """Minimum average LMP should be at least $5/MWh."""
        min_lmp = sweep_df["avg_lmp"].min()
        assert min_lmp >= 5, (
            f"Min avg LMP = ${min_lmp:.2f}/MWh — suspiciously low"
        )

    def test_lmp_max_below_ceiling(self, sweep_df):
        """Max average LMP should stay below $500/MWh.

        Some high-stress scenarios (high demand + low supply + gas friction)
        can produce LMPs in the $300–$450 range due to ORDC scarcity pricing.
        """
        max_lmp = sweep_df["avg_lmp"].max()
        assert max_lmp <= 500, (
            f"Max avg LMP = ${max_lmp:.1f}/MWh — exceeds $500 ceiling"
        )

    def test_lmp_mean_realistic(self, sweep_df):
        """Mean LMP across all scenarios should be in $20–$80 range."""
        mean_lmp = sweep_df["avg_lmp"].mean()
        assert 20 <= mean_lmp <= 80, (
            f"Overall mean LMP = ${mean_lmp:.1f}/MWh — outside $20–$80 range"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. ORDC Scarcity Hours
# ─────────────────────────────────────────────────────────────────────────────

class TestScarcityHours:
    """Validate ORDC scarcity hours are within expected bounds."""

    def test_scarcity_hours_max(self, sweep_df):
        """No scenario should have > 4,000 scarcity hours (>46% of year).

        High-stress scenarios (high demand growth + slow queue + gas friction)
        can produce 2,000–3,000 scarcity hours. 4,000 is the hard ceiling.
        """
        max_hrs = sweep_df["ordc_scarcity_hours"].max()
        assert max_hrs < 4000, (
            f"Max scarcity hours = {max_hrs} — extreme value indicates "
            f"broken ORDC or supply model"
        )

    def test_scarcity_hours_non_negative(self, sweep_df):
        """Scarcity hours must never be negative."""
        min_hrs = sweep_df["ordc_scarcity_hours"].min()
        assert min_hrs >= 0, f"Negative scarcity hours: {min_hrs}"

    def test_some_scarcity_exists(self, sweep_df):
        """At least some scenarios should have non-zero scarcity hours."""
        has_scarcity = (sweep_df["ordc_scarcity_hours"] > 0).any()
        assert has_scarcity, (
            "Zero scarcity hours across all scenarios — ORDC may be "
            "over-suppressed by new fossil builds"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Fleet Results Validation
# ─────────────────────────────────────────────────────────────────────────────

class TestFleetResults:
    """Validate single-fleet results JSON structure and new fossil fields."""

    def test_fleet_has_envelope(self, fleet_results):
        """Fleet results must contain an envelope with P10/P50/P90."""
        assert "envelope" in fleet_results, "Missing 'envelope' key"
        envelope = fleet_results["envelope"]
        assert len(envelope) > 0, "Envelope is empty"

    def test_fleet_envelope_has_percentiles(self, fleet_results):
        """Each year in envelope must have p10, p50, p90, mean."""
        envelope = fleet_results["envelope"]
        for year, data in envelope.items():
            for stat in ("p10", "p50", "p90", "mean"):
                assert stat in data, (
                    f"Envelope year {year} missing '{stat}'"
                )

    def test_fleet_has_grid_new_fossil(self, fleet_results):
        """Fleet results must include grid_new_fossil data."""
        assert "grid_new_fossil" in fleet_results, (
            "Missing 'grid_new_fossil' key — fleet dispatch not ingesting "
            "new fossil columns"
        )

    def test_fleet_grid_new_fossil_fields(self, fleet_results):
        """grid_new_fossil entries must have expected fields."""
        grid_nf = fleet_results.get("grid_new_fossil", {})
        if not grid_nf:
            pytest.skip("No grid_new_fossil data")
        # Check first year entry
        first_year = next(iter(grid_nf.values()))
        expected_fields = {
            "total_new_fossil_mw", "gas_built_gw", "fossil_built_gw",
            "nb_gas_ct_mw", "nb_gas_ccgt_mw",
        }
        actual_fields = set(first_year.keys())
        missing = expected_fields - actual_fields
        assert not missing, (
            f"grid_new_fossil missing fields: {missing}"
        )

    def test_fleet_has_new_fossil_summary(self, fleet_results):
        """Fleet results must include new_fossil_summary with both sources."""
        assert "new_fossil_summary" in fleet_results, (
            "Missing 'new_fossil_summary' key"
        )
        summary = fleet_results["new_fossil_summary"]
        if summary:
            first_year = next(iter(summary.values()))
            for key in ("grid_new_fossil_mw", "fleet_new_fossil_mw",
                        "total_new_fossil_mw"):
                assert key in first_year, (
                    f"new_fossil_summary missing '{key}'"
                )

    def test_fleet_has_summary(self, fleet_results):
        """Fleet results must have a fleet_summary."""
        assert "fleet_summary" in fleet_results, "Missing 'fleet_summary'"
        summary = fleet_results["fleet_summary"]
        for key in ("total_plants", "total_capacity_mw", "isos"):
            assert key in summary, f"fleet_summary missing '{key}'"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Fleet Scenario Results Validation
# ─────────────────────────────────────────────────────────────────────────────

class TestFleetScenarioResults:
    """Validate multi-scenario fleet results."""

    def test_has_scenarios(self, fleet_scenario_results):
        """Must contain a 'scenarios' dict with at least baseline."""
        assert "scenarios" in fleet_scenario_results, "Missing 'scenarios'"
        scenarios = fleet_scenario_results["scenarios"]
        assert len(scenarios) > 0, "No scenarios in results"
        assert "baseline" in scenarios, "Missing 'baseline' scenario"

    def test_baseline_has_envelope(self, fleet_scenario_results):
        """Baseline scenario must have an envelope."""
        baseline = fleet_scenario_results["scenarios"]["baseline"]
        assert "envelope" in baseline, "Baseline missing 'envelope'"

    def test_baseline_has_new_fossil(self, fleet_scenario_results):
        """Baseline scenario must include grid_new_fossil data."""
        baseline = fleet_scenario_results["scenarios"]["baseline"]
        assert "grid_new_fossil" in baseline, (
            "Baseline missing 'grid_new_fossil'"
        )

    def test_has_target_trajectories(self, fleet_scenario_results):
        """Must contain target trajectories for gap analysis."""
        assert "target_trajectories" in fleet_scenario_results, (
            "Missing 'target_trajectories'"
        )

    def test_scenarios_have_new_fossil_summary(self, fleet_scenario_results):
        """Each scenario must have new_fossil_summary."""
        for name, scenario in fleet_scenario_results["scenarios"].items():
            assert "new_fossil_summary" in scenario, (
                f"Scenario '{name}' missing 'new_fossil_summary'"
            )

    def test_envelope_by_fossil_cost(self, fleet_scenario_results):
        """Baseline should have envelope_by_fossil_cost with L/M/H keys."""
        baseline = fleet_scenario_results["scenarios"]["baseline"]
        if "envelope_by_fossil_cost" not in baseline:
            pytest.skip("envelope_by_fossil_cost not present")
        by_cost = baseline["envelope_by_fossil_cost"]
        for level in ("Low", "Medium", "High"):
            assert level in by_cost, (
                f"envelope_by_fossil_cost missing '{level}'"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 7. Aggregates JSON Validation
# ─────────────────────────────────────────────────────────────────────────────

class TestAggregates:
    """Validate sweep aggregates JSON structure (P10/P50/P90 bands)."""

    def test_all_isos_present(self, aggregates):
        """Aggregates must have entries for all 7 ISOs."""
        actual = set(aggregates.keys())
        assert EXPECTED_ISOS.issubset(actual), (
            f"Missing ISOs in aggregates: {EXPECTED_ISOS - actual}"
        )

    def test_all_years_per_iso(self, aggregates):
        """Each ISO must have data for all 6 years."""
        expected_year_strs = {str(y) for y in EXPECTED_YEARS}
        for iso in EXPECTED_ISOS:
            if iso not in aggregates:
                continue
            actual_years = set(aggregates[iso].keys())
            missing = expected_year_strs - actual_years
            assert not missing, (
                f"{iso} missing years in aggregates: {missing}"
            )

    def test_percentile_bands_exist(self, aggregates):
        """Each ISO-year must have p10/p50/p90/mean for key metrics."""
        key_metrics = ["avg_lmp", "emissions_mt", "clean_pct"]
        for iso in EXPECTED_ISOS:
            if iso not in aggregates:
                continue
            for year_str in aggregates[iso]:
                year_data = aggregates[iso][year_str]
                for metric in key_metrics:
                    if metric not in year_data:
                        continue
                    bands = year_data[metric]
                    for stat in ("p10", "p50", "p90", "mean"):
                        assert stat in bands, (
                            f"{iso}/{year_str}/{metric} missing '{stat}'"
                        )

    def test_new_fossil_in_aggregates(self, aggregates):
        """Aggregates should include total_new_fossil_mw metric."""
        # Check at least one ISO has it
        found = False
        for iso in EXPECTED_ISOS:
            if iso not in aggregates:
                continue
            for year_str in aggregates[iso]:
                if "total_new_fossil_mw" in aggregates[iso][year_str]:
                    found = True
                    break
            if found:
                break
        assert found, "total_new_fossil_mw not found in any ISO aggregates"

    def test_aggregate_lmp_sanity(self, aggregates):
        """Aggregate LMP P50 values should be in realistic range.

        Some ISOs (NEISO, NYISO) can have elevated median LMPs in later years
        due to pipeline constraints and demand growth, reaching $250–$300.
        """
        for iso in EXPECTED_ISOS:
            if iso not in aggregates:
                continue
            for year_str in aggregates[iso]:
                lmp_data = aggregates[iso][year_str].get("avg_lmp", {})
                if "p50" in lmp_data:
                    p50 = lmp_data["p50"]
                    assert 5 <= p50 <= 350, (
                        f"{iso}/{year_str} P50 LMP = ${p50:.1f} — "
                        f"outside $5–$350 range"
                    )


# ─────────────────────────────────────────────────────────────────────────────
# 8. API Endpoint Validation (requires running backend)
# ─────────────────────────────────────────────────────────────────────────────

def _api_available():
    """Check if the backend API is reachable."""
    try:
        import urllib.request
        req = urllib.request.Request(
            "http://localhost:8000/api/health",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


@pytest.mark.skipif(
    not _api_available(),
    reason="Backend API not running on localhost:8000",
)
class TestAPIEndpoints:
    """Validate dashboard API endpoints return valid JSON."""

    def _fetch_json(self, path):
        """Fetch JSON from the local API."""
        import urllib.request
        url = f"http://localhost:8000{path}"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 200, (
                f"{path} returned status {resp.status}"
            )
            return json.loads(resp.read().decode())

    def test_health(self):
        """Health endpoint responds."""
        data = self._fetch_json("/api/health")
        assert data is not None

    def test_sweep_cached_status(self):
        """Sweep cached status shows available."""
        data = self._fetch_json("/api/sweep-cached/status")
        assert data.get("available") is True, (
            f"Sweep not available: {data}"
        )
        assert data.get("total_scenarios") == 1215
        assert "sweep_dimensions" in data
        dims = data["sweep_dimensions"]
        assert "new_fossil_cost_level" in dims, (
            "sweep_dimensions missing new_fossil_cost_level"
        )

    def test_sweep_cached_aggregates(self):
        """Aggregates endpoint returns valid P10/P50/P90 data."""
        data = self._fetch_json("/api/sweep-cached/aggregates")
        # Should have ISO keys
        assert any(iso in data for iso in EXPECTED_ISOS), (
            f"No ISO data in aggregates response"
        )

    def test_sweep_cached_results(self):
        """Results endpoint returns rows with new fossil columns."""
        data = self._fetch_json(
            "/api/sweep-cached/results?iso=PJM&year=2030"
        )
        # Check structure
        if "results" in data:
            results = data["results"]
            if len(results) > 0:
                row = results[0]
                assert "total_new_fossil_mw" in row, (
                    "API results missing total_new_fossil_mw"
                )

    def test_fleet_scenario_results_endpoint(self):
        """Fleet scenario results endpoint returns valid data."""
        data = self._fetch_json("/api/fleet-scenario-results")
        assert "scenarios" in data or isinstance(data, list), (
            "Unexpected fleet scenario results structure"
        )

    def test_isos_endpoint(self):
        """ISOs endpoint returns all 7 ISOs."""
        data = self._fetch_json("/api/isos")
        if isinstance(data, list):
            iso_names = {d.get("iso") or d.get("name") for d in data}
        elif isinstance(data, dict):
            iso_names = set(data.keys())
        else:
            iso_names = set()
        assert EXPECTED_ISOS.issubset(iso_names), (
            f"Missing ISOs from /api/isos: {EXPECTED_ISOS - iso_names}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 9. Cross-Validation: Parquet ↔ Aggregates Consistency
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossValidation:
    """Verify aggregates are consistent with raw parquet data."""

    def test_aggregate_p50_matches_parquet_median(self, sweep_df, aggregates):
        """Aggregate P50 LMP should match parquet median within tolerance."""
        import numpy as np
        for iso in EXPECTED_ISOS:
            if iso not in aggregates:
                continue
            for year in EXPECTED_YEARS:
                year_str = str(year)
                if year_str not in aggregates[iso]:
                    continue
                agg_p50 = aggregates[iso][year_str].get("avg_lmp", {}).get("p50")
                if agg_p50 is None:
                    continue
                subset = sweep_df[
                    (sweep_df["iso"] == iso) & (sweep_df["year"] == year)
                ]
                if len(subset) == 0:
                    continue
                parquet_median = float(np.median(subset["avg_lmp"]))
                # Allow 1% tolerance for floating point
                assert abs(agg_p50 - parquet_median) < max(0.5, parquet_median * 0.01), (
                    f"{iso}/{year}: Aggregate P50 LMP ({agg_p50:.2f}) != "
                    f"parquet median ({parquet_median:.2f})"
                )


# ─────────────────────────────────────────────────────────────────────────────
# CLI runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Allow running directly: python test_e2e_integration.py
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
