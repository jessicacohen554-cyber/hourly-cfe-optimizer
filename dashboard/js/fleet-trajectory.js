// ============================================================================
// FLEET TRAJECTORY DEEP-DIVE — Baseline 1,215-scenario analysis
// ============================================================================
(function () {
    'use strict';

    // ── Constants ──
    var YEARS_6PT = [2023, 2030, 2035, 2040, 2045, 2050];
    var PROJECTION_YEARS = [2030, 2035, 2040, 2045, 2050];
    var FOSSIL_FUELS = { gas_ccgt: 1, gas_ct: 1, gas_oil_ct: 1, oil_ct: 1 };
    var FUEL_LABELS = {
        gas_ccgt: 'Gas CCGT', gas_ct: 'Gas CT',
        gas_oil_ct: 'Gas/Oil', oil_ct: 'Oil CT'
    };
    var TIER_KEYS = ['very_low', 'low', 'medium', 'high', 'very_high'];
    var TIER_LABELS = {
        very_low: 'Very Low HR (<6.5)',
        low: 'Low HR (6.5–7.5)',
        medium: 'Medium HR (7.5–8.5)',
        high: 'High HR (8.5–10)',
        very_high: 'Very High HR (≥10)'
    };
    var TIER_COLORS = {
        very_low: '#22C55E', low: '#0EA5E9', medium: '#6366F1',
        high: '#F59E0B', very_high: '#EF4444'
    };
    var BAND_COLORS = {
        p0_p10:   'rgba(56, 189, 248, 0.20)',
        p10_p25:  'rgba(34, 197, 94, 0.20)',
        p25_p50:  'rgba(163, 230, 53, 0.20)',
        p50_p75:  'rgba(234, 179, 8, 0.20)',
        p75_p90:  'rgba(249, 115, 22, 0.20)',
        p90_p100: 'rgba(239, 68, 68, 0.20)'
    };
    var DIMENSION_LABELS = {
        conditions: 'Market Conditions',
        demand_growth: 'Demand Growth',
        price_sens: 'Price Sensitivity',
        ppa_level: 'PPA Level',
        gas_friction: 'Gas Friction'
    };
    var DIMENSION_SCENARIO_LABELS = {
        conditions: { F: 'Favorable', C: 'Challenging' },
        demand_growth: { L: 'Low', M: 'Medium', H: 'High' },
        price_sens: {
            all_low: 'All Low', all_med: 'All Medium', all_high: 'All High',
            high_vre_low_firm: 'High VRE / Low Firm', high_firm_low_vre: 'High Firm / Low VRE'
        },
        ppa_level: { L: 'Low', M: 'Medium', H: 'High' },
        gas_friction: { L: 'Low', M: 'Medium', H: 'High' }
    };
    // Scenario ID dimension positions: MKT_{demand}_{price_sens}_{ppa}_{gas}_{queue}_{nfc}
    var SWEEP_DIMENSIONS = ['demand', 'price_sens', 'ppa', 'gas_friction', 'queue', 'new_fossil_cost'];

    // ── State ──
    var selectedISO = 'ALL';
    var fleetData = null;       // fleet_scenario_results_sample.json
    var fleetConfig = null;     // constellation_scenarios.json (has ISO + heat rate per plant)
    var sweepData = null;       // sweep_dispatch_data.json
    var smartargets = null;     // IPP_SMARTARGETS_DATA.companies.constellation
    var activeBins = {};
    TIER_KEYS.forEach(function (k) { activeBins[k] = true; });

    var emissionsFanChart = null;
    var cfFanChart = null;
    var dimensionCharts = [];
    var cfDimensionCharts = [];
    var plantTableData = [];
    var sortCol = 'capacity_mw';
    var sortAsc = false;

    // ── Helpers ──
    function percentile(arr, p) {
        if (!arr.length) return 0;
        var sorted = arr.slice().sort(function (a, b) { return a - b; });
        var idx = (p / 100) * (sorted.length - 1);
        var lo = Math.floor(idx), hi = Math.ceil(idx);
        if (lo === hi) return sorted[lo];
        return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
    }

    function hexToRgba(hex, alpha) {
        var r = parseInt(hex.slice(1, 3), 16);
        var g = parseInt(hex.slice(3, 5), 16);
        var b = parseInt(hex.slice(5, 7), 16);
        return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
    }

    function cfFromGen(genTwh, capMw) {
        if (!capMw || capMw <= 0) return 0;
        return genTwh / (capMw * 8.76 / 1000);
    }

    // ── Init ──
    init();

    function init() {
        setupISOSelector();
        loadAllData();
    }

    function setupISOSelector() {
        var container = document.getElementById('isoSelector');
        if (!container) return;
        container.addEventListener('click', function (e) {
            var pill = e.target.closest('.iso-pill');
            if (!pill) return;
            container.querySelectorAll('.iso-pill').forEach(function (p) { p.classList.remove('active'); });
            pill.classList.add('active');
            selectedISO = pill.dataset.iso;
            onISOChanged();
        });
    }

    function loadAllData() {
        var p1 = fetch('data/fleet_scenario_results_sample.json').then(function (r) { return r.json(); });
        var p2 = fetch('data/constellation_scenarios.json').then(function (r) { return r.json(); });
        var p3 = fetch('data/sweep_dispatch_data.json').then(function (r) { return r.json(); });

        // Load smartargets from global
        if (typeof IPP_SMARTARGETS_DATA !== 'undefined' && IPP_SMARTARGETS_DATA.companies) {
            smartargets = IPP_SMARTARGETS_DATA.companies.constellation || null;
        }

        Promise.all([p1, p2, p3]).then(function (results) {
            fleetData = results[0];
            fleetConfig = results[1];
            sweepData = results[2];
            onAllDataLoaded();
        }).catch(function (err) {
            console.error('Failed to load data:', err);
            document.querySelectorAll('.loading-overlay').forEach(function (el) {
                el.textContent = 'Failed to load data. Ensure data files are present.';
            });
        });
    }

    function onAllDataLoaded() {
        buildPlantTableData();
        buildEmissionsFanChart();
        buildDimensionCharts();
        buildPlantTable();
        buildBinToggles();
        buildCfFanChart();
        buildCfDimensionCharts();
        updateNarrativeInsights();
    }

    function onISOChanged() {
        updateEmissionsFanChart();
        updateDimensionCharts();
        buildPlantTable();
        updateCfFanChart();
        updateCfDimensionCharts();
        updateNarrativeInsights();

        var suffix = selectedISO === 'ALL' ? 'All Fleet' : selectedISO;
        var t1 = document.getElementById('emissionsFanTitle');
        if (t1) t1.textContent = 'Fleet Emissions — Percentile Bands — ' + suffix;
        var t2 = document.getElementById('plantTableTitle');
        if (t2) t2.textContent = 'Fossil Fleet — ' + suffix;
        var t3 = document.getElementById('cfFanTitle');
        if (t3) t3.textContent = 'Capacity Factor by Heat-Rate Bin — ' + suffix;
    }

    // ── Build ISO-to-ORISPL lookup from constellation_scenarios.json ──
    var plantISOMap = {};   // orispl → iso
    var plantHRMap = {};    // orispl → heat_rate_mmbtu_mwh
    function buildPlantTableData() {
        if (!fleetConfig || !fleetConfig.base_fleet) return;
        fleetConfig.base_fleet.forEach(function (p) {
            plantISOMap[p.orispl] = p.iso || 'NA';
            if (p.heat_rate_mmbtu_mwh) plantHRMap[p.orispl] = p.heat_rate_mmbtu_mwh;
        });
    }

    // ====================================================================
    // PLACEHOLDER: Sections 1-5 will be added in subsequent chunks
    // ====================================================================

    // Section 1: buildEmissionsFanChart / updateEmissionsFanChart
    // Section 2: buildDimensionCharts / updateDimensionCharts
    // Section 3: buildPlantTable (sortable table)
    // Section 4: buildCfFanChart / updateCfFanChart + buildCfDimensionCharts / updateCfDimensionCharts
    // Section 5: updateNarrativeInsights

})();
