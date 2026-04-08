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
        var ALL_ISOS = ['CAISO', 'ERCOT', 'MISO', 'NEISO', 'NYISO', 'PJM', 'SPP'];

        // Load smartargets from global
        if (typeof IPP_SMARTARGETS_DATA !== 'undefined' && IPP_SMARTARGETS_DATA.companies) {
            smartargets = IPP_SMARTARGETS_DATA.companies.constellation || null;
        }

        // 1. Constellation config (single file, no per-ISO variant)
        var pConfig = fetch('data/constellation_scenarios.json')
            .then(function (r) { return r.json(); });

        // 2. Fleet scenario results — try per-ISO files first, fall back to combined
        var pFleet = (function () {
            var perIsoFetches = ALL_ISOS.map(function (iso) {
                return fetch('data/fleet_scenario_results_' + iso + '.json')
                    .then(function (r) { return r.ok ? r.json() : null; })
                    .catch(function () { return null; });
            });
            return Promise.all(perIsoFetches).then(function (results) {
                var loaded = results.filter(function (r) { return r !== null; });
                if (loaded.length > 0) {
                    var merged = JSON.parse(JSON.stringify(loaded[0]));
                    for (var i = 1; i < loaded.length; i++) {
                        var isoData = loaded[i];
                        Object.keys(isoData.scenarios || {}).forEach(function (skey) {
                            var src = isoData.scenarios[skey];
                            var dst = merged.scenarios[skey];
                            if (!dst) { merged.scenarios[skey] = src; return; }
                            Object.keys(src.plant_detail || {}).forEach(function (yr) {
                                if (!dst.plant_detail[yr]) dst.plant_detail[yr] = [];
                                dst.plant_detail[yr] = dst.plant_detail[yr].concat(src.plant_detail[yr] || []);
                            });
                            Object.keys(src.plant_percentiles || {}).forEach(function (yr) {
                                if (!dst.plant_percentiles[yr]) dst.plant_percentiles[yr] = [];
                                dst.plant_percentiles[yr] = dst.plant_percentiles[yr].concat(src.plant_percentiles[yr] || []);
                            });
                        });
                    }
                    return merged;
                }
                return fetch('data/fleet_scenario_results_sample.json')
                    .then(function (r) { if (r.ok) return r.json(); throw new Error('No fleet data'); });
            });
        })();

        // 3. Sweep dispatch data — try per-ISO files first, fall back to combined
        var pSweep = (function () {
            var perIsoFetches = ALL_ISOS.map(function (iso) {
                return fetch('data/sweep_dispatch_data_' + iso + '.json')
                    .then(function (r) { return r.ok ? r.json() : null; })
                    .catch(function () { return null; });
            });
            return Promise.all(perIsoFetches).then(function (results) {
                var loaded = results.filter(function (r) { return r !== null; });
                if (loaded.length > 0) {
                    var merged = Object.assign({}, loaded[0]);
                    merged.data = {};
                    merged.isos = [];
                    loaded.forEach(function (isoData) {
                        Object.keys(isoData.data || {}).forEach(function (k) {
                            merged.data[k] = isoData.data[k];
                            if (merged.isos.indexOf(k) === -1) merged.isos.push(k);
                        });
                    });
                    merged.isos.sort();
                    return merged;
                }
                return fetch('data/sweep_dispatch_data.json')
                    .then(function (r) { if (r.ok) return r.json(); throw new Error('No sweep data'); });
            });
        })();

        Promise.all([pFleet, pConfig, pSweep]).then(function (results) {
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
    // SECTION 1: ENHANCED EMISSIONS FAN CHART — 6 percentile bands
    // ====================================================================
    // Uses IPP_SMARTARGETS_DATA for the 6-point trajectory (min/p10/p50/p90/max)
    // and interpolates p25/p75 between p10↔p50 and p50↔p90.
    // For ISO-specific view, we compute from sweep_dispatch_data by aggregating
    // fleet-level emissions per scenario, but fall back to the ipp data for "ALL".

    function getEmissionsPercentiles() {
        // For "ALL" fleet, use ipp-smartargets-data which has pre-computed percentiles
        if (selectedISO === 'ALL' && smartargets && smartargets.fan_bands) {
            var fb = smartargets.fan_bands.reference || smartargets.fan_bands.all;
            if (!fb || !fb.emissions) return null;
            var e = fb.emissions;
            // e.p10, e.p50, e.p90, e.min, e.max are each 6-element arrays for YEARS_6PT
            var result = {};
            YEARS_6PT.forEach(function (yr, i) {
                var p0  = e.min ? e.min[i] : e.p10[i];
                var p10 = e.p10[i];
                var p50 = e.p50[i];
                var p90 = e.p90[i];
                var p100 = e.max ? e.max[i] : e.p90[i];
                var p25 = (p10 + p50) / 2;
                var p75 = (p50 + p90) / 2;
                result[yr] = { p0: p0, p10: p10, p25: p25, p50: p50, p75: p75, p90: p90, p100: p100 };
            });
            return result;
        }

        // For ISO-specific, use the fleet_scenario_results envelope (p10/p50/p90 per year)
        // and approximate p0=min, p100=max, p25/p75 interpolated
        if (fleetData && fleetData.scenarios && fleetData.scenarios.baseline) {
            var env = fleetData.scenarios.baseline.envelope;
            if (!env) return null;
            var result2 = {};
            var years = Object.keys(env).map(Number).sort(function (a, b) { return a - b; });
            years.forEach(function (yr) {
                var e2 = env[yr];
                if (!e2) return;
                var p10 = e2.p10, p50 = e2.p50, p90 = e2.p90;
                // Approximate min/max from spread
                var spread = p90 - p10;
                var p0 = Math.max(0, p10 - spread * 0.15);
                var p100 = p90 + spread * 0.15;
                result2[yr] = {
                    p0: p0, p10: p10, p25: (p10 + p50) / 2,
                    p50: p50, p75: (p50 + p90) / 2, p90: p90, p100: p100
                };
            });
            return result2;
        }
        return null;
    }

    function buildEmissionsFanChart() {
        var ctx = document.getElementById('emissionsFanChart');
        if (!ctx) return;
        emissionsFanChart = new Chart(ctx.getContext('2d'), {
            type: 'line',
            data: { labels: [], datasets: [] },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 400 },
                interaction: { mode: 'index', intersect: false },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { font: { size: 12 }, maxRotation: 0 }
                    },
                    y: {
                        title: { display: true, text: 'Emissions (Mt CO₂)', font: { size: 13, weight: '600' } },
                        grid: { color: '#E0E6EF' },
                        beginAtZero: true,
                        ticks: { font: { size: 11 } }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            title: function (items) { return items[0] ? items[0].label : ''; },
                            label: function (ctx) {
                                var ds = ctx.dataset;
                                if (ds._bandLabel) return ds._bandLabel + ': ' + ctx.raw.toFixed(1) + ' Mt';
                                if (ds._isP50) return 'P50 Median: ' + ctx.raw.toFixed(1) + ' Mt';
                                return null;
                            }
                        }
                    }
                }
            }
        });
        updateEmissionsFanChart();
    }

    function updateEmissionsFanChart() {
        if (!emissionsFanChart) return;
        var pctls = getEmissionsPercentiles();
        if (!pctls) {
            emissionsFanChart.data.labels = [];
            emissionsFanChart.data.datasets = [];
            emissionsFanChart.update();
            return;
        }

        var years = Object.keys(pctls).map(Number).sort(function (a, b) { return a - b; });
        var labels = years.map(String);

        // Extract arrays for each percentile level
        var arrP0  = years.map(function (y) { return pctls[y].p0; });
        var arrP10 = years.map(function (y) { return pctls[y].p10; });
        var arrP25 = years.map(function (y) { return pctls[y].p25; });
        var arrP50 = years.map(function (y) { return pctls[y].p50; });
        var arrP75 = years.map(function (y) { return pctls[y].p75; });
        var arrP90 = years.map(function (y) { return pctls[y].p90; });
        var arrP100 = years.map(function (y) { return pctls[y].p100; });

        // Build datasets: 6 filled bands + P50 solid line
        // Band approach: boundary lines (invisible) with fill between them
        var datasets = [];

        // Helper to add a band between two boundary arrays
        function addBand(upper, lower, color, label, dsIndex) {
            // Lower boundary (invisible)
            datasets.push({
                label: label + ' lower',
                data: lower,
                borderWidth: 0, pointRadius: 0, fill: false,
                backgroundColor: 'transparent', borderColor: 'transparent',
                tension: 0.3, _bandLabel: null
            });
            // Upper boundary with fill to previous dataset
            datasets.push({
                label: label,
                data: upper,
                borderWidth: 0, pointRadius: 0,
                fill: { target: dsIndex, above: color, below: color },
                backgroundColor: color, borderColor: 'transparent',
                tension: 0.3, _bandLabel: label
            });
        }

        // Bottom to top: p0→p10, p10→p25, p25→p50, p50→p75, p75→p90, p90→p100
        addBand(arrP10, arrP0, BAND_COLORS.p0_p10, 'P0–P10 (Highly Optimistic)', 0);
        addBand(arrP25, arrP10, BAND_COLORS.p10_p25, 'P10–P25 (Optimistic)', 1);
        addBand(arrP50, arrP25, BAND_COLORS.p25_p50, 'P25–P50', 3);
        addBand(arrP75, arrP50, BAND_COLORS.p50_p75, 'P50–P75', 5);
        addBand(arrP90, arrP75, BAND_COLORS.p75_p90, 'P75–P90', 7);
        addBand(arrP100, arrP90, BAND_COLORS.p90_p100, 'P90–P100 (Pessimistic)', 9);

        // P50 median line on top
        datasets.push({
            label: 'P50 Median',
            data: arrP50,
            borderColor: '#111111',
            borderWidth: 3,
            pointRadius: years.map(function (y) { return YEARS_6PT.indexOf(y) >= 0 ? 4 : 0; }),
            pointBackgroundColor: '#111',
            fill: false,
            tension: 0.3,
            _isP50: true, _bandLabel: null
        });

        emissionsFanChart.data.labels = labels;
        emissionsFanChart.data.datasets = datasets;
        emissionsFanChart.update();
    }
    // ====================================================================
    // SECTION 2: DIMENSION SENSITIVITY — Small-multiple line charts
    // ====================================================================
    // Uses dimension_fans from IPP_SMARTARGETS_DATA.companies.constellation
    // Each dimension has N scenario variants, each with a P50 emissions trajectory

    var DIM_LINE_PALETTE = ['#2372B9', '#F47B27', '#6BA543', '#DC2626', '#6366F1', '#0EA5E9'];

    function buildDimensionCharts() {
        var grid = document.getElementById('dimensionChartsGrid');
        if (!grid) return;

        if (!smartargets || !smartargets.dimension_fans) {
            grid.innerHTML = '<div class="loading-overlay">No dimension sensitivity data available.</div>';
            return;
        }

        var dims = smartargets.dimension_fans;
        var dimKeys = Object.keys(dims);
        grid.innerHTML = '';
        dimensionCharts = [];

        dimKeys.forEach(function (dimKey) {
            var dimData = dims[dimKey];
            var scenKeys = Object.keys(dimData);
            var card = document.createElement('div');
            card.style.cssText = 'min-height:220px;';
            card.innerHTML = '<div style="font-size:0.82rem;font-weight:700;color:var(--text-bright);margin-bottom:6px;">' +
                (DIMENSION_LABELS[dimKey] || dimKey) + '</div>' +
                '<canvas></canvas>';
            grid.appendChild(card);

            var ctx = card.querySelector('canvas').getContext('2d');
            var datasets = [];
            scenKeys.forEach(function (scKey, i) {
                var scData = dimData[scKey];
                var emissions = scData.emissions ? scData.emissions.p50 : null;
                if (!emissions) return;
                var scLabel = (DIMENSION_SCENARIO_LABELS[dimKey] && DIMENSION_SCENARIO_LABELS[dimKey][scKey])
                    ? DIMENSION_SCENARIO_LABELS[dimKey][scKey] : scKey;
                datasets.push({
                    label: scLabel,
                    data: emissions,
                    borderColor: DIM_LINE_PALETTE[i % DIM_LINE_PALETTE.length],
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    pointRadius: YEARS_6PT.map(function () { return 3; }),
                    pointBackgroundColor: DIM_LINE_PALETTE[i % DIM_LINE_PALETTE.length],
                    tension: 0.3,
                    fill: false
                });
            });

            var chart = new Chart(ctx, {
                type: 'line',
                data: { labels: YEARS_6PT.map(String), datasets: datasets },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: { duration: 300 },
                    interaction: { mode: 'index', intersect: false },
                    scales: {
                        x: { grid: { display: false }, ticks: { font: { size: 10 }, maxRotation: 0 } },
                        y: {
                            title: { display: true, text: 'Mt CO₂', font: { size: 10 } },
                            grid: { color: '#E0E6EF' },
                            beginAtZero: true,
                            ticks: { font: { size: 9 } }
                        }
                    },
                    plugins: {
                        legend: {
                            display: true, position: 'bottom',
                            labels: { font: { size: 10 }, boxWidth: 14, padding: 8 }
                        },
                        tooltip: {
                            callbacks: {
                                label: function (ctx) {
                                    return ctx.dataset.label + ': ' + ctx.raw.toFixed(1) + ' Mt';
                                }
                            }
                        }
                    }
                }
            });
            dimensionCharts.push(chart);
        });
    }

    function updateDimensionCharts() {
        // Dimension charts use IPP_SMARTARGETS_DATA which is fleet-wide (not ISO-specific)
        // For now, they don't change with ISO toggle — they always show fleet-level sensitivity
        // (The dimension_fans are pre-computed at the company level, not per-ISO)
    }
    // ====================================================================
    // SECTION 3: FOSSIL PLANT TABLE — Sortable with projections
    // ====================================================================

    function buildPlantTable() {
        var tbody = document.getElementById('plantTableBody');
        if (!tbody || !fleetData) return;

        var baseline = fleetData.scenarios.baseline;
        if (!baseline || !baseline.plant_detail) {
            tbody.innerHTML = '<tr><td colspan="15" style="text-align:center;padding:24px;">No plant data available.</td></tr>';
            return;
        }

        // Build rows from plant_detail + plant_percentiles
        var pp = baseline.plant_percentiles || {};
        var plants2025 = baseline.plant_detail['2025'] || baseline.plant_detail['2023'] || [];

        var rows = [];
        plants2025.forEach(function (p) {
            if (!FOSSIL_FUELS[p.fuel_type]) return;
            var iso = plantISOMap[p.orispl] || '—';
            if (selectedISO !== 'ALL' && iso !== selectedISO) return;

            var hr = plantHRMap[p.orispl] || null;
            var cf2025 = cfFromGen(p.gen_twh, p.capacity_mw);
            var emis2025 = p.emissions_mt ? p.emissions_mt * 1000 : 0; // Mt → kt

            // P50 CF projections from plant_percentiles
            var pPerc = pp[String(p.orispl)];
            var cfProj = {};
            PROJECTION_YEARS.forEach(function (yr) {
                if (pPerc && pPerc.gen_gwh && pPerc.gen_gwh[String(yr)]) {
                    var genGwh = pPerc.gen_gwh[String(yr)].p50 || 0;
                    cfProj[yr] = genGwh / (p.capacity_mw * 8760 / 1000);
                } else {
                    // Fall back to plant_detail at that year
                    var yrPlants = baseline.plant_detail[String(yr)];
                    if (yrPlants) {
                        var found = yrPlants.find(function (pp2) { return pp2.orispl === p.orispl; });
                        if (found) {
                            cfProj[yr] = cfFromGen(found.gen_twh, found.capacity_mw);
                        } else {
                            cfProj[yr] = 0; // retired
                        }
                    } else {
                        cfProj[yr] = null;
                    }
                }
            });

            // Detect retirement: find first year where plant is missing or status = retired
            var retireYear = null;
            var retireReason = '';
            var allYears = Object.keys(baseline.plant_detail).map(Number).sort(function (a, b) { return a - b; });
            for (var yi = 0; yi < allYears.length; yi++) {
                var yr = allYears[yi];
                if (yr <= 2025) continue;
                var yrPlants = baseline.plant_detail[String(yr)];
                if (!yrPlants) continue;
                var found = yrPlants.find(function (pp2) { return pp2.orispl === p.orispl; });
                if (found && found.status === 'retired') {
                    retireYear = yr;
                    retireReason = 'Economic';
                    break;
                }
                if (!found) {
                    retireYear = yr;
                    retireReason = 'Economic';
                    break;
                }
            }

            rows.push({
                name: p.name,
                orispl: p.orispl,
                iso: iso,
                year_built: p.year_built || 0,
                capacity_mw: p.capacity_mw || 0,
                fuel_type: p.fuel_type,
                heat_rate: hr,
                heat_rate_tier: p.heat_rate_tier || '—',
                cf_2025: cf2025,
                emis_2025: emis2025,
                cf_2030: cfProj[2030],
                cf_2035: cfProj[2035],
                cf_2040: cfProj[2040],
                cf_2045: cfProj[2045],
                cf_2050: cfProj[2050],
                retireYear: retireYear,
                retirement: retireYear ? retireYear + ' (' + retireReason + ')' : 'Operating'
            });
        });

        // Sort
        rows.sort(function (a, b) {
            var va = a[sortCol], vb = b[sortCol];
            if (va == null) va = sortAsc ? Infinity : -Infinity;
            if (vb == null) vb = sortAsc ? Infinity : -Infinity;
            if (typeof va === 'string') {
                var cmp = va.localeCompare(vb);
                return sortAsc ? cmp : -cmp;
            }
            return sortAsc ? va - vb : vb - va;
        });

        plantTableData = rows;

        // Render
        function fmtCf(v) {
            if (v == null) return '—';
            if (v <= 0.001) return '<span class="status-retired">0%</span>';
            return (v * 100).toFixed(1) + '%';
        }
        function fmtNum(v, d) {
            if (v == null) return '—';
            return v.toFixed(d !== undefined ? d : 0);
        }

        var html = '';
        rows.forEach(function (r) {
            var retClass = r.retireYear ? 'status-retired' : 'status-operating';
            html += '<tr>' +
                '<td>' + r.name + '</td>' +
                '<td>' + r.iso + '</td>' +
                '<td class="num-cell">' + (r.year_built || '—') + '</td>' +
                '<td class="num-cell">' + fmtNum(r.capacity_mw, 0) + '</td>' +
                '<td>' + (FUEL_LABELS[r.fuel_type] || r.fuel_type) + '</td>' +
                '<td class="num-cell">' + (r.heat_rate ? r.heat_rate.toFixed(2) : '—') + '</td>' +
                '<td>' + (TIER_LABELS[r.heat_rate_tier] || r.heat_rate_tier || '—') + '</td>' +
                '<td class="cf-cell">' + fmtCf(r.cf_2025) + '</td>' +
                '<td class="num-cell">' + fmtNum(r.emis_2025, 1) + '</td>' +
                '<td class="cf-cell">' + fmtCf(r.cf_2030) + '</td>' +
                '<td class="cf-cell">' + fmtCf(r.cf_2035) + '</td>' +
                '<td class="cf-cell">' + fmtCf(r.cf_2040) + '</td>' +
                '<td class="cf-cell">' + fmtCf(r.cf_2045) + '</td>' +
                '<td class="cf-cell">' + fmtCf(r.cf_2050) + '</td>' +
                '<td class="' + retClass + '">' + r.retirement + '</td>' +
                '</tr>';
        });

        if (!html) html = '<tr><td colspan="15" style="text-align:center;color:var(--text-dim);padding:24px;">No fossil plants in selected region.</td></tr>';
        tbody.innerHTML = html;

        // Wire up sort headers (once)
        if (!plantTableData._sortWired) {
            var table = document.getElementById('plantTable');
            if (table) {
                table.querySelector('thead').addEventListener('click', function (e) {
                    var th = e.target.closest('th');
                    if (!th || !th.dataset.col) return;
                    if (sortCol === th.dataset.col) {
                        sortAsc = !sortAsc;
                    } else {
                        sortCol = th.dataset.col;
                        sortAsc = true;
                    }
                    // Update sort arrow visuals
                    table.querySelectorAll('th').forEach(function (t) { t.classList.remove('sorted'); });
                    th.classList.add('sorted');
                    th.querySelector('.sort-arrow').innerHTML = sortAsc ? '&#9650;' : '&#9660;';
                    buildPlantTable();
                });
            }
            plantTableData._sortWired = true;
        }
    }
    // ====================================================================
    // SECTION 4: CF FAN BY HEAT-RATE BIN — Percentile bands from sweep
    // ====================================================================

    function buildBinToggles() {
        var container = document.getElementById('binToggles');
        if (!container) return;
        var html = '';
        TIER_KEYS.forEach(function (tier) {
            html += '<label class="bin-toggle">' +
                '<input type="checkbox" data-tier="' + tier + '" checked>' +
                '<span class="bin-swatch" style="background:' + TIER_COLORS[tier] + ';"></span>' +
                (TIER_LABELS[tier] || tier) + '</label>';
        });
        container.innerHTML = html;
        container.addEventListener('change', function (e) {
            var cb = e.target;
            if (!cb.dataset.tier) return;
            activeBins[cb.dataset.tier] = cb.checked;
            updateCfFanChart();
        });
    }

    function getCfPercentilesForTier(tier, iso) {
        if (!sweepData || !sweepData.data) return null;
        var cfKey = 'gas_ccgt_' + tier + '_cf';
        // Aggregate across ISOs or use specific ISO
        var isos = iso === 'ALL' ? sweepData.isos : [iso];
        var years = sweepData.years;
        var nScenarios = sweepData.n_scenarios;
        var nYears = years.length;

        // For each year, collect CF values across all scenarios (weighted avg across ISOs if ALL)
        var result = {};
        years.forEach(function (yr, yi) {
            var vals = [];
            for (var si = 0; si < nScenarios; si++) {
                var sum = 0, count = 0;
                isos.forEach(function (isoKey) {
                    var isoData = sweepData.data[isoKey];
                    if (!isoData || !isoData[cfKey]) return;
                    var cfArr = isoData[cfKey][si];
                    if (cfArr && cfArr[yi] != null) {
                        sum += cfArr[yi];
                        count++;
                    }
                });
                if (count > 0) vals.push(sum / count);
            }
            if (vals.length > 0) {
                result[yr] = {
                    p0: percentile(vals, 0),
                    p10: percentile(vals, 10),
                    p25: percentile(vals, 25),
                    p50: percentile(vals, 50),
                    p75: percentile(vals, 75),
                    p90: percentile(vals, 90),
                    p100: percentile(vals, 100)
                };
            }
        });
        return result;
    }

    function buildCfFanChart() {
        var ctx = document.getElementById('cfFanChart');
        if (!ctx) return;
        cfFanChart = new Chart(ctx.getContext('2d'), {
            type: 'line',
            data: { labels: [], datasets: [] },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 400 },
                interaction: { mode: 'index', intersect: false },
                scales: {
                    x: { grid: { display: false }, ticks: { font: { size: 12 }, maxRotation: 0 } },
                    y: {
                        title: { display: true, text: 'Capacity Factor', font: { size: 13, weight: '600' } },
                        grid: { color: '#E0E6EF' },
                        min: 0, max: 1,
                        ticks: {
                            font: { size: 11 },
                            callback: function (v) { return (v * 100).toFixed(0) + '%'; }
                        }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function (ctx) {
                                var ds = ctx.dataset;
                                if (ds._tierLabel && ds._isP50) return ds._tierLabel + ' P50: ' + (ctx.raw * 100).toFixed(1) + '%';
                                if (ds._tierLabel && ds._bandLabel) return ds._tierLabel + ' ' + ds._bandLabel + ': ' + (ctx.raw * 100).toFixed(1) + '%';
                                return null;
                            }
                        }
                    }
                }
            }
        });
        updateCfFanChart();
    }

    function updateCfFanChart() {
        if (!cfFanChart || !sweepData) return;

        var years = sweepData.years;
        var labels = years.map(String);
        var datasets = [];
        var legendItems = [];

        TIER_KEYS.forEach(function (tier) {
            if (!activeBins[tier]) return;
            var pctls = getCfPercentilesForTier(tier, selectedISO);
            if (!pctls) return;

            var color = TIER_COLORS[tier];
            var arrP10 = years.map(function (y) { return pctls[y] ? pctls[y].p10 : null; });
            var arrP50 = years.map(function (y) { return pctls[y] ? pctls[y].p50 : null; });
            var arrP90 = years.map(function (y) { return pctls[y] ? pctls[y].p90 : null; });

            var baseIdx = datasets.length;

            // P10 boundary
            datasets.push({
                data: arrP10, borderWidth: 0, pointRadius: 0, fill: false,
                backgroundColor: 'transparent', borderColor: 'transparent', tension: 0.3,
                _tierLabel: null, _bandLabel: null
            });
            // P90 boundary with fill to P10
            datasets.push({
                data: arrP90, borderWidth: 0, pointRadius: 0,
                fill: { target: baseIdx, above: hexToRgba(color, 0.15), below: hexToRgba(color, 0.15) },
                backgroundColor: hexToRgba(color, 0.15), borderColor: 'transparent', tension: 0.3,
                _tierLabel: TIER_LABELS[tier], _bandLabel: 'P10–P90'
            });
            // P50 line
            datasets.push({
                label: TIER_LABELS[tier] + ' (P50)',
                data: arrP50,
                borderColor: color, borderWidth: 2.5,
                pointRadius: years.map(function (y) { return y % 5 === 0 ? 3 : 0; }),
                pointBackgroundColor: color,
                fill: false, tension: 0.3,
                _tierLabel: TIER_LABELS[tier], _isP50: true, _bandLabel: null
            });

            legendItems.push({ label: TIER_LABELS[tier], color: color });
        });

        cfFanChart.data.labels = labels;
        cfFanChart.data.datasets = datasets;
        cfFanChart.update();

        // Legend
        var legendEl = document.getElementById('cfFanLegend');
        if (legendEl) {
            legendEl.innerHTML = legendItems.map(function (item) {
                return '<span style="display:inline-flex;align-items:center;gap:5px;margin-right:16px;">' +
                    '<span style="width:24px;height:3px;background:' + item.color + ';display:inline-block;border-radius:1px;"></span>' +
                    '<span style="width:20px;height:10px;background:' + hexToRgba(item.color, 0.15) + ';display:inline-block;border-radius:2px;border:1px solid ' + hexToRgba(item.color, 0.3) + ';"></span>' +
                    item.label + '</span>';
            }).join('');
        }
    }

    // ── CF Dimension Breakdown per Bin ──
    function buildCfDimensionCharts() {
        updateCfDimensionCharts();
    }

    function updateCfDimensionCharts() {
        var grid = document.getElementById('cfDimensionChartsGrid');
        if (!grid || !sweepData) {
            if (grid) grid.innerHTML = '<div class="loading-overlay">Waiting for sweep data...</div>';
            return;
        }

        // Parse scenario IDs to extract dimensions
        var scenarios = sweepData.scenarios;
        var parsedDims = scenarios.map(function (sid) {
            // MKT_{demand}_{price_sens}_{ppa}_{gas}_{queue}_{nfc}
            var parts = sid.split('_');
            // demand is parts[1], price_sens can be multi-word (all_high, high_vre_low_firm)
            // Use a more robust parser
            var m = sid.match(/^MKT_([HML])_(.+?)_([HML])_([HML])_([HML])_([HML])$/);
            if (!m) return null;
            return {
                demand: m[1], price_sens: m[2],
                ppa: m[3], gas_friction: m[4],
                queue: m[5], new_fossil_cost: m[6]
            };
        });

        // Show dimension breakdown for the first active bin
        var activeTier = TIER_KEYS.find(function (t) { return activeBins[t]; });
        if (!activeTier) {
            grid.innerHTML = '<div style="color:var(--text-dim);padding:24px;text-align:center;">Enable at least one heat-rate bin to see CF sensitivity.</div>';
            return;
        }

        var cfKey = 'gas_ccgt_' + activeTier + '_cf';
        var isos = selectedISO === 'ALL' ? sweepData.isos : [selectedISO];
        var years = sweepData.years;
        var yearIdx2030 = years.indexOf(2030);
        var yearIdx2040 = years.indexOf(2040);
        var yearIdx2050 = years.indexOf(2050);

        // For each analysis dimension, group scenarios and compute median CF at key years
        var dimAnalysis = ['demand', 'gas_friction', 'ppa', 'new_fossil_cost', 'queue'];
        var dimLabelsMap = {
            demand: 'Demand Growth', gas_friction: 'Gas Friction',
            ppa: 'PPA Level', new_fossil_cost: 'New Fossil Cost', queue: 'Queue Capacity'
        };
        var dimValLabels = {
            L: 'Low', M: 'Medium', H: 'High'
        };

        // Destroy old charts
        cfDimensionCharts.forEach(function (c) { try { c.destroy(); } catch (e) {} });
        cfDimensionCharts = [];
        grid.innerHTML = '';

        dimAnalysis.forEach(function (dimKey) {
            // Group scenario indices by dimension value
            var groups = {};
            parsedDims.forEach(function (pd, si) {
                if (!pd) return;
                var val = pd[dimKey];
                if (!val) return;
                if (!groups[val]) groups[val] = [];
                groups[val].push(si);
            });

            var card = document.createElement('div');
            card.style.cssText = 'min-height:200px;';
            card.innerHTML = '<div style="font-size:0.82rem;font-weight:700;color:var(--text-bright);margin-bottom:6px;">' +
                (dimLabelsMap[dimKey] || dimKey) + ' → ' + (TIER_LABELS[activeTier] || activeTier) + ' CF</div>' +
                '<canvas></canvas>';
            grid.appendChild(card);

            var ctx = card.querySelector('canvas').getContext('2d');
            var datasets = [];
            var groupKeys = Object.keys(groups).sort();

            groupKeys.forEach(function (gKey, gi) {
                var indices = groups[gKey];
                // Compute median CF across years for this group
                var p50Line = years.map(function (yr, yi) {
                    var vals = [];
                    indices.forEach(function (si) {
                        var sum = 0, cnt = 0;
                        isos.forEach(function (isoKey) {
                            var isoData = sweepData.data[isoKey];
                            if (!isoData || !isoData[cfKey]) return;
                            var v = isoData[cfKey][si] ? isoData[cfKey][si][yi] : null;
                            if (v != null) { sum += v; cnt++; }
                        });
                        if (cnt > 0) vals.push(sum / cnt);
                    });
                    return vals.length > 0 ? percentile(vals, 50) : null;
                });

                datasets.push({
                    label: dimValLabels[gKey] || gKey,
                    data: p50Line,
                    borderColor: DIM_LINE_PALETTE[gi % DIM_LINE_PALETTE.length],
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.3,
                    fill: false
                });
            });

            var chart = new Chart(ctx, {
                type: 'line',
                data: { labels: years.map(String), datasets: datasets },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: { duration: 200 },
                    scales: {
                        x: { grid: { display: false }, ticks: { font: { size: 9 }, maxRotation: 0, callback: function (v) { var y = Number(this.getLabelForValue(v)); return y % 5 === 0 ? y : ''; } } },
                        y: {
                            title: { display: true, text: 'CF', font: { size: 9 } },
                            grid: { color: '#E0E6EF' },
                            min: 0,
                            ticks: { font: { size: 9 }, callback: function (v) { return (v * 100).toFixed(0) + '%'; } }
                        }
                    },
                    plugins: {
                        legend: { display: true, position: 'bottom', labels: { font: { size: 9 }, boxWidth: 12, padding: 6 } },
                        tooltip: {
                            callbacks: {
                                label: function (ctx) { return ctx.dataset.label + ': ' + (ctx.raw * 100).toFixed(1) + '%'; }
                            }
                        }
                    }
                }
            });
            cfDimensionCharts.push(chart);
        });
    }
    // ====================================================================
    // SECTION 5: NARRATIVE INSIGHTS — Data-driven callout cards
    // ====================================================================

    function updateNarrativeInsights() {
        if (!plantTableData || !plantTableData.length) return;

        var rows = plantTableData; // already filtered by ISO in buildPlantTable

        // ── Insight 1: Efficiency determines dispatch priority ──
        var tierGroups = {};
        rows.forEach(function (r) {
            var tier = r.heat_rate_tier || 'unknown';
            if (!tierGroups[tier]) tierGroups[tier] = { cfs2025: [], cfs2050: [] };
            if (r.cf_2025 != null) tierGroups[tier].cfs2025.push(r.cf_2025);
            if (r.cf_2050 != null) tierGroups[tier].cfs2050.push(r.cf_2050);
        });

        function median(arr) {
            if (!arr.length) return 0;
            var s = arr.slice().sort(function (a, b) { return a - b; });
            var mid = Math.floor(s.length / 2);
            return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
        }

        var bestTier = tierGroups['very_low'] || tierGroups['low'] || { cfs2025: [], cfs2050: [] };
        var worstTier = tierGroups['very_high'] || tierGroups['high'] || { cfs2025: [], cfs2050: [] };
        var bestCF = median(bestTier.cfs2050) * 100;
        var worstCF = median(worstTier.cfs2050) * 100;
        var bestLabel = tierGroups['very_low'] ? 'Very Low HR' : 'Low HR';
        var worstLabel = tierGroups['very_high'] ? 'Very High HR' : 'High HR';

        var el1 = document.querySelectorAll('#insightsSection .insight-box')[0];
        if (el1) {
            var p1 = el1.querySelector('p');
            if (p1) {
                if (bestCF > 0 || worstCF > 0) {
                    p1.textContent = 'By 2050, the most efficient CCGTs (' + bestLabel + ') maintain a median CF of ' +
                        bestCF.toFixed(0) + '%, while legacy units (' + worstLabel + ') drop to ' + worstCF.toFixed(0) +
                        '% — a ' + Math.abs(bestCF - worstCF).toFixed(0) + ' percentage-point gap. ' +
                        'The spread widens after 2030 as renewable capacity displaces marginal gas generation, ' +
                        'making heat-rate efficiency the primary determinant of which plants survive.';
                }
            }
        }

        // ── Insight 2: Gas friction & demand growth sensitivity ──
        // Use dimension_fans data if available to compute spreads
        var el2 = document.querySelectorAll('#insightsSection .insight-box')[1];
        if (el2 && smartargets && smartargets.dimension_fans) {
            var p2 = el2.querySelector('p');
            if (p2) {
                var dimSpreads = {};
                var dims = Object.keys(smartargets.dimension_fans);
                dims.forEach(function (dim) {
                    var scenarios = smartargets.dimension_fans[dim];
                    var keys = Object.keys(scenarios);
                    if (!keys.length) return;
                    // Look at 2050 endpoint (last value) spread across scenario variants
                    var vals2050 = [];
                    keys.forEach(function (k) {
                        var traj = scenarios[k];
                        if (traj && traj.p50 && traj.p50.length) {
                            vals2050.push(traj.p50[traj.p50.length - 1]);
                        }
                    });
                    if (vals2050.length >= 2) {
                        var mn = Math.min.apply(null, vals2050);
                        var mx = Math.max.apply(null, vals2050);
                        dimSpreads[dim] = mx - mn;
                    }
                });

                // Rank dimensions by spread
                var ranked = Object.keys(dimSpreads).sort(function (a, b) {
                    return dimSpreads[b] - dimSpreads[a];
                });

                if (ranked.length >= 2) {
                    var top1 = DIMENSION_LABELS[ranked[0]] || ranked[0];
                    var top2 = DIMENSION_LABELS[ranked[1]] || ranked[1];
                    var spread1 = dimSpreads[ranked[0]].toFixed(1);
                    var spread2 = dimSpreads[ranked[1]].toFixed(1);
                    p2.textContent = top1 + ' drives the most emissions uncertainty (' + spread1 +
                        ' Mt spread at 2050), followed by ' + top2 + ' (' + spread2 +
                        ' Mt). Under the worst-case combination, fleet emissions persist well above ' +
                        'SBTi trajectories through 2050. Under the best case, economic retirements ' +
                        'accelerate and emissions decline faster than required.';
                }
            }
        }

        // ── Insight 3: Location shapes outcomes ──
        var el3 = document.querySelectorAll('#insightsSection .insight-box')[2];
        if (el3) {
            var p3 = el3.querySelector('p');
            if (p3) {
                // Compute avg CF decline by ISO
                var isoDeclines = {};
                rows.forEach(function (r) {
                    if (r.cf_2025 == null || r.cf_2050 == null) return;
                    var iso = r.iso;
                    if (!isoDeclines[iso]) isoDeclines[iso] = [];
                    isoDeclines[iso].push(r.cf_2025 - r.cf_2050);
                });

                var isoAvg = {};
                Object.keys(isoDeclines).forEach(function (iso) {
                    var arr = isoDeclines[iso];
                    isoAvg[iso] = arr.reduce(function (s, v) { return s + v; }, 0) / arr.length;
                });

                var isoRanked = Object.keys(isoAvg).sort(function (a, b) {
                    return isoAvg[b] - isoAvg[a];
                });

                if (isoRanked.length >= 2) {
                    var steepest = isoRanked[0];
                    var mildest = isoRanked[isoRanked.length - 1];
                    p3.textContent = steepest + ' plants face the steepest CF declines (avg ' +
                        (isoAvg[steepest] * 100).toFixed(0) + ' pp drop by 2050) due to aggressive ' +
                        'renewable buildout, while ' + mildest + ' plants see the smallest decline (' +
                        (isoAvg[mildest] * 100).toFixed(0) + ' pp). Capacity market revenue and ' +
                        'transmission constraints create divergent outcomes even for plants with ' +
                        'identical heat rates across different ISOs.';
                }
            }
        }

        // ── Insight 4: Retirement risk concentrates in the tail ──
        var el4 = document.getElementById('retirementInsight');
        if (el4) {
            var p4 = document.getElementById('retirementInsightText');
            if (p4) {
                var totalPlants = rows.length;
                var retired2035 = 0, retired2040 = 0, retired2050 = 0;
                rows.forEach(function (r) {
                    if (r.retireYear && r.retireYear <= 2035) retired2035++;
                    if (r.retireYear && r.retireYear <= 2040) retired2040++;
                    if (r.retireYear && r.retireYear <= 2050) retired2050++;
                });

                if (totalPlants > 0) {
                    var pct2035 = (retired2035 / totalPlants * 100).toFixed(0);
                    var pct2040 = (retired2040 / totalPlants * 100).toFixed(0);
                    var pct2050 = (retired2050 / totalPlants * 100).toFixed(0);

                    if (retired2050 > 0) {
                        p4.textContent = 'Under median (P50) conditions, ' + pct2035 + '% of the ' +
                            (selectedISO === 'ALL' ? 'fleet' : selectedISO) + '\'s ' + totalPlants +
                            ' fossil plants face economic retirement by 2035, rising to ' + pct2040 +
                            '% by 2040 and ' + pct2050 + '% by 2050. The transition is not gradual — ' +
                            'retirements cluster when renewable additions cross regional dispatch tipping ' +
                            'points, making the 2028–2035 window the critical transition period.';
                    } else {
                        p4.textContent = 'Under median (P50) conditions, all ' + totalPlants +
                            ' fossil plants in ' + (selectedISO === 'ALL' ? 'the fleet' : selectedISO) +
                            ' remain operational through 2050. However, under optimistic (P10) scenarios, ' +
                            'economic retirements could reduce the fossil fleet significantly as renewable ' +
                            'additions cross regional dispatch tipping points.';
                    }
                }
            }
        }
    }

})();
