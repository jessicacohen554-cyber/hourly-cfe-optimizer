// ============================================================================
// FLEET SCENARIOS — Interactive fleet scenario comparison dashboard
// ============================================================================
// Loads fleet_scenario_results.json and renders:
//   1. Emissions Fan Chart (P10/P50/P90 bands per scenario + target overlays)
//   2. Plant-level Waterfall (delta vs baseline at selected year)
//   3. Generation Mix (stacked bar by fuel per scenario at selected year)
//   4. Emissions by Fuel (stacked area over time for selected scenario)
// ============================================================================

(function () {
    'use strict';

    // ── Scenario color map (from design spec, uses CEG brand palette) ──
    var SCENARIO_COLORS = {
        baseline:           '#7E8083',
        ccs_only:           '#2372B9',
        ccs_plus_new_gas:   '#F47B27',
        retire_coal_ccs_gas:'#6BA543'
    };

    var SCENARIO_LABELS = {
        baseline:           'Baseline',
        ccs_only:           'CCS-Only',
        ccs_plus_new_gas:   'CCS + New Gas',
        retire_coal_ccs_gas:'Retire + CCS'
    };

    var TARGET_STYLES = {
        sbti_15:     { color: '#DC2626', dash: [8, 4], label: 'SBTi 1.5°C' },
        at_power_nz: { color: '#6366F1', dash: [],     label: 'AT Power NZ' },
        custom:      { color: '#6B7280', dash: [4, 4], label: 'Custom' }
    };

    var FUEL_KEYS   = ['coal_steam', 'gas_ccgt', 'gas_ct', 'oil_ct', 'nuclear', 'ccs_ccgt'];
    var FUEL_LABELS = { coal_steam: 'Coal', gas_ccgt: 'Gas CCGT', gas_ct: 'Gas CT', oil_ct: 'Oil', nuclear: 'Nuclear', ccs_ccgt: 'CCS-CCGT' };
    var FUEL_COLORS = {
        coal_steam: RESOURCE_COLORS.fossilCoal,
        gas_ccgt:   RESOURCE_COLORS.fossilGas,
        gas_ct:     RESOURCE_COLORS.fossilGasCT || '#007FA4',
        oil_ct:     RESOURCE_COLORS.fossilOil,
        nuclear:    RESOURCE_COLORS.nuclear,
        ccs_ccgt:   RESOURCE_COLORS.ccs
    };
    var EMISSION_FUEL_KEYS = ['coal_steam', 'gas_ccgt', 'gas_ct', 'oil_ct'];

    // ── State ──
    var DATA = null;
    var scenarioVisibility = {};
    var selectedTarget = 'sbti_15';
    var selectedYear = 2030;
    var fanChart = null;
    var waterfallChart = null;
    var genMixChart = null;
    var fuelEmissionsChart = null;

    // ── Helpers ──
    function getYears() {
        if (!DATA) return [];
        var first = Object.keys(DATA.scenarios)[0];
        return Object.keys(DATA.scenarios[first].envelope).map(Number).sort(function (a, b) { return a - b; });
    }

    function hexToRgba(hex, alpha) {
        var r = parseInt(hex.slice(1, 3), 16);
        var g = parseInt(hex.slice(3, 5), 16);
        var b = parseInt(hex.slice(5, 7), 16);
        return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
    }

    function nearestYear(year, years) {
        var best = years[0];
        for (var i = 1; i < years.length; i++) {
            if (Math.abs(years[i] - year) < Math.abs(best - year)) best = years[i];
        }
        return best;
    }

    function getVisibleScenarios() {
        var out = [];
        for (var key in scenarioVisibility) {
            if (scenarioVisibility[key]) out.push(key);
        }
        return out;
    }

    function getFirstVisibleNonBaseline() {
        var vis = getVisibleScenarios();
        for (var i = 0; i < vis.length; i++) {
            if (vis[i] !== 'baseline') return vis[i];
        }
        return vis[0] || 'ccs_only';
    }

    // ── Init ──
    function init() {
        loadData();
    }

    function loadData() {
        // Try production path first, then sample data
        fetch('/api/fleet-scenario-results')
            .then(function (r) { if (r.ok) return r.json(); throw new Error('API not available'); })
            .then(onDataLoaded)
            .catch(function () {
                fetch('data/fleet_scenario_results_sample.json')
                    .then(function (r) { if (r.ok) return r.json(); throw new Error('No data'); })
                    .then(onDataLoaded)
                    .catch(function () {
                        fetch('/data/fleet_scenario_results_sample.json')
                            .then(function (r) { if (r.ok) return r.json(); throw new Error('No data'); })
                            .then(onDataLoaded)
                            .catch(showError);
                    });
            });
    }

    function showError() {
        var section = document.querySelector('.content-section');
        if (section) {
            section.innerHTML = '<div class="error-banner">Failed to load fleet scenario results. Run the fleet scenario pipeline first, or place <code>fleet_scenario_results_sample.json</code> in <code>frontend/data/</code>.</div>';
        }
    }

    function onDataLoaded(data) {
        DATA = data;

        // Initialize visibility — all on
        for (var key in DATA.scenarios) {
            scenarioVisibility[key] = true;
        }

        buildControls();
        buildFanChart();
        buildWaterfallChart();
        buildGenMixChart();
        buildFuelEmissionsChart();
        updateFanLegend();
    }

    // ====================================================================
    // CONTROLS
    // ====================================================================
    function buildControls() {
        // Scenario checkboxes
        var checksEl = document.getElementById('scenarioChecks');
        checksEl.innerHTML = '';
        for (var key in DATA.scenarios) {
            var color = DATA.scenarios[key].color || SCENARIO_COLORS[key] || '#888';
            var label = SCENARIO_LABELS[key] || key;
            var lbl = document.createElement('label');
            lbl.className = 'scenario-check';
            lbl.innerHTML =
                '<input type="checkbox" data-scenario="' + key + '" checked>' +
                '<span class="scenario-swatch" style="background:' + color + '"></span>' +
                label;
            checksEl.appendChild(lbl);
        }
        checksEl.addEventListener('change', function (e) {
            var cb = e.target;
            if (cb.dataset.scenario) {
                scenarioVisibility[cb.dataset.scenario] = cb.checked;
                updateAllCharts();
            }
        });

        // Target radios
        var radiosEl = document.getElementById('targetRadios');
        radiosEl.innerHTML = '';
        var targetKeys = DATA.targets ? Object.keys(DATA.targets) : [];
        targetKeys.forEach(function (tKey, i) {
            var t = DATA.targets[tKey];
            var style = TARGET_STYLES[tKey] || TARGET_STYLES.custom;
            var lbl = document.createElement('label');
            lbl.className = 'target-radio';
            lbl.innerHTML =
                '<input type="radio" name="targetOverlay" value="' + tKey + '"' + (i === 0 ? ' checked' : '') + '>' +
                t.label;
            radiosEl.appendChild(lbl);
        });
        // "None" option
        var noneLbl = document.createElement('label');
        noneLbl.className = 'target-radio';
        noneLbl.innerHTML = '<input type="radio" name="targetOverlay" value="none"> None';
        radiosEl.appendChild(noneLbl);

        radiosEl.addEventListener('change', function (e) {
            selectedTarget = e.target.value;
            updateFanChart();
            updateFanLegend();
        });

        // Year slider
        var slider = document.getElementById('yearSlider');
        var sliderVal = document.getElementById('yearSliderValue');
        var years = getYears();
        if (years.length) {
            slider.min = years[0];
            slider.max = years[years.length - 1];
            slider.value = nearestYear(2030, years);
            selectedYear = Number(slider.value);
            sliderVal.textContent = slider.value;
        }
        slider.addEventListener('input', function () {
            selectedYear = Number(slider.value);
            sliderVal.textContent = slider.value;
            updateWaterfallChart();
            updateGenMixChart();
            updateFanChartAnnotation();
        });

        // Waterfall scenario selector
        var wfSelect = document.getElementById('waterfallScenarioSelect');
        wfSelect.innerHTML = '';
        for (var sKey in DATA.scenarios) {
            if (sKey === 'baseline') continue;
            var opt = document.createElement('option');
            opt.value = sKey;
            opt.textContent = SCENARIO_LABELS[sKey] || sKey;
            wfSelect.appendChild(opt);
        }
        wfSelect.addEventListener('change', function () { updateWaterfallChart(); });

        // Fuel emissions scenario selector
        var feSelect = document.getElementById('fuelScenarioSelect');
        feSelect.innerHTML = '';
        for (var fKey in DATA.scenarios) {
            var fOpt = document.createElement('option');
            fOpt.value = fKey;
            fOpt.textContent = SCENARIO_LABELS[fKey] || fKey;
            feSelect.appendChild(fOpt);
        }
        feSelect.addEventListener('change', function () { updateFuelEmissionsChart(); });
    }

    function updateAllCharts() {
        updateFanChart();
        updateFanLegend();
        updateWaterfallChart();
        updateGenMixChart();
        updateFuelEmissionsChart();
    }

    // ====================================================================
    // CHART 1: EMISSIONS FAN CHART
    // ====================================================================
    function buildFanChart() {
        var ctx = document.getElementById('fanChart').getContext('2d');
        fanChart = new Chart(ctx, {
            type: 'line',
            data: { labels: [], datasets: [] },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 800 },
                interaction: { mode: 'index', intersect: false },
                scales: {
                    x: {
                        title: { display: true, text: 'Year', font: { size: 13 } },
                        grid: { display: false },
                        ticks: {
                            callback: function (val, idx) {
                                var year = this.getLabelForValue(val);
                                var show = [2023, 2025, 2030, 2035, 2040, 2045, 2050];
                                return show.indexOf(Number(year)) >= 0 ? year : '';
                            },
                            maxRotation: 0, font: { size: 12 }
                        }
                    },
                    y: {
                        title: { display: true, text: 'Fleet Emissions (Mt CO₂)', font: { size: 13 } },
                        grid: { color: '#E0E6EF' },
                        beginAtZero: true,
                        ticks: { font: { size: 12 } }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: { enabled: false, external: fanTooltipHandler },
                    annotation: { annotations: {} }
                },
                onClick: function (evt) {
                    var points = fanChart.getElementsAtEventForMode(evt, 'index', { intersect: false }, false);
                    if (points.length) {
                        var year = Number(fanChart.data.labels[points[0].index]);
                        document.getElementById('yearSlider').value = year;
                        selectedYear = year;
                        document.getElementById('yearSliderValue').textContent = year;
                        updateWaterfallChart();
                        updateGenMixChart();
                        updateFanChartAnnotation();
                    }
                }
            }
        });
        updateFanChart();
    }

    function updateFanChart() {
        if (!fanChart || !DATA) return;
        var years = getYears();
        var labels = years.map(String);
        var datasets = [];

        // Scenario bands + P50 lines
        for (var key in DATA.scenarios) {
            if (!scenarioVisibility[key]) continue;
            var sc = DATA.scenarios[key];
            var color = sc.color || SCENARIO_COLORS[key] || '#888';
            var p10 = [], p50 = [], p90 = [];
            years.forEach(function (y) {
                var e = sc.envelope[String(y)];
                p10.push(e ? e.p10 : null);
                p50.push(e ? e.p50 : null);
                p90.push(e ? e.p90 : null);
            });

            // P90 upper bound (drawn first, fill down to P10)
            datasets.push({
                label: (SCENARIO_LABELS[key] || key) + ' P90',
                data: p90,
                borderColor: hexToRgba(color, 0.3),
                backgroundColor: hexToRgba(color, 0.0),
                borderWidth: 1,
                pointRadius: 0,
                fill: false,
                tension: 0.3,
                _scenarioKey: key,
                _band: 'p90'
            });

            // P10 lower bound (fill up to P90)
            datasets.push({
                label: (SCENARIO_LABELS[key] || key) + ' P10',
                data: p10,
                borderColor: hexToRgba(color, 0.3),
                backgroundColor: hexToRgba(color, 0.20),
                borderWidth: 1,
                pointRadius: 0,
                fill: '-1',
                tension: 0.3,
                _scenarioKey: key,
                _band: 'p10'
            });

            // P50 line
            datasets.push({
                label: (SCENARIO_LABELS[key] || key) + ' P50',
                data: p50,
                borderColor: color,
                backgroundColor: 'transparent',
                borderWidth: 2.5,
                pointRadius: 3,
                pointBackgroundColor: color,
                pointHoverRadius: 5,
                fill: false,
                tension: 0.3,
                _scenarioKey: key,
                _band: 'p50'
            });
        }

        // Target overlay
        if (selectedTarget && selectedTarget !== 'none' && DATA.targets && DATA.targets[selectedTarget]) {
            var tgt = DATA.targets[selectedTarget];
            var style = TARGET_STYLES[selectedTarget] || TARGET_STYLES.custom;
            var tgtData = years.map(function (y) {
                return tgt.trajectory[String(y)] != null ? tgt.trajectory[String(y)] : null;
            });
            datasets.push({
                label: tgt.label,
                data: tgtData,
                borderColor: style.color,
                backgroundColor: 'transparent',
                borderWidth: 2,
                borderDash: style.dash,
                pointRadius: 0,
                fill: false,
                tension: 0.3,
                _target: true
            });
        }

        fanChart.data.labels = labels;
        fanChart.data.datasets = datasets;

        // Year reference line annotation
        updateFanChartAnnotation();

        fanChart.update('active');
    }

    function updateFanChartAnnotation() {
        if (!fanChart) return;
        var years = getYears();
        var labels = fanChart.data.labels;
        var idx = labels.indexOf(String(selectedYear));
        if (idx < 0) idx = labels.indexOf(String(nearestYear(selectedYear, years)));

        fanChart.options.plugins.annotation.annotations = {
            yearLine: {
                type: 'line',
                xMin: idx, xMax: idx,
                borderColor: 'rgba(26, 35, 47, 0.3)',
                borderWidth: 1.5,
                borderDash: [4, 4],
                label: {
                    display: true,
                    content: String(selectedYear),
                    position: 'start',
                    backgroundColor: 'rgba(26, 35, 47, 0.7)',
                    color: '#fff',
                    font: { size: 11 }
                }
            }
        };
        fanChart.update('none');
    }

    // ── Custom tooltip for fan chart ──
    function fanTooltipHandler(context) {
        var tooltipEl = document.getElementById('fanTooltip');
        var tooltipModel = context.tooltip;

        if (tooltipModel.opacity === 0) {
            tooltipEl.classList.remove('visible');
            return;
        }

        if (tooltipModel.dataPoints && tooltipModel.dataPoints.length) {
            var idx = tooltipModel.dataPoints[0].dataIndex;
            var year = fanChart.data.labels[idx];
            var html = '<div class="fan-tooltip-title">' + year + '</div>';

            var visible = getVisibleScenarios();
            visible.forEach(function (sKey) {
                var sc = DATA.scenarios[sKey];
                var env = sc.envelope[year];
                if (!env) return;
                var color = sc.color || SCENARIO_COLORS[sKey] || '#888';
                html += '<div class="fan-tooltip-row">' +
                    '<span class="fan-tooltip-swatch" style="background:' + color + '"></span>' +
                    '<span class="fan-tooltip-label">' + (SCENARIO_LABELS[sKey] || sKey) + '</span>' +
                    '<span class="fan-tooltip-vals">P10: ' + env.p10.toFixed(1) +
                    '  P50: ' + env.p50.toFixed(1) +
                    '  P90: ' + env.p90.toFixed(1) + '</span></div>';
            });

            // Gap to target
            if (selectedTarget && selectedTarget !== 'none' && DATA.gap_analysis) {
                html += '<div class="fan-tooltip-gap">';
                var tgtLabel = DATA.targets[selectedTarget] ? DATA.targets[selectedTarget].label : selectedTarget;
                html += '<strong>Gap to ' + tgtLabel + ':</strong><br>';
                visible.forEach(function (sKey) {
                    var ga = DATA.gap_analysis[sKey];
                    if (!ga || !ga[selectedTarget]) return;
                    var gap = ga[selectedTarget].gap_mt[year];
                    if (gap == null) return;
                    var cls = gap > 0 ? 'gap-positive' : 'gap-negative';
                    var sign = gap > 0 ? '+' : '';
                    var check = gap <= 0 ? ' ✓' : '';
                    html += '<span class="' + cls + '">' + (SCENARIO_LABELS[sKey] || sKey) + ' ' + sign + gap.toFixed(1) + ' Mt' + check + '</span>  ';
                });
                html += '</div>';
            }

            tooltipEl.innerHTML = html;
        }

        // Position
        var canvas = context.chart.canvas;
        var rect = canvas.getBoundingClientRect();
        var left = tooltipModel.caretX;
        var top = tooltipModel.caretY;

        // Flip if too close to right edge
        if (left + 320 > rect.width) {
            left = left - 340;
        }
        if (left < 10) left = 10;
        if (top < 10) top = 10;

        tooltipEl.style.left = left + 'px';
        tooltipEl.style.top = top + 'px';
        tooltipEl.classList.add('visible');
    }

    // ── Fan legend ──
    function updateFanLegend() {
        var el = document.getElementById('fanLegend');
        if (!el || !DATA) return;
        var items = [];
        for (var key in DATA.scenarios) {
            if (!scenarioVisibility[key]) continue;
            var color = DATA.scenarios[key].color || SCENARIO_COLORS[key];
            items.push({ label: SCENARIO_LABELS[key] || key, color: color, type: 'band' });
        }
        if (selectedTarget && selectedTarget !== 'none' && DATA.targets && DATA.targets[selectedTarget]) {
            var style = TARGET_STYLES[selectedTarget] || TARGET_STYLES.custom;
            items.push({
                label: DATA.targets[selectedTarget].label,
                color: style.color,
                type: style.dash.length ? 'dashed' : 'line'
            });
        }
        if (typeof buildLegend === 'function') {
            buildLegend(el, items);
        }
    }

    // ====================================================================
    // CHART 2: PLANT-LEVEL WATERFALL
    // ====================================================================
    function buildWaterfallChart() {
        var ctx = document.getElementById('waterfallChart').getContext('2d');
        waterfallChart = new Chart(ctx, {
            type: 'bar',
            data: { labels: [], datasets: [] },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                animation: { duration: 300 },
                scales: {
                    x: {
                        title: { display: true, text: 'Emissions Delta (Mt CO₂)', font: { size: 12 } },
                        grid: { color: '#E0E6EF' },
                        ticks: { font: { size: 11 } }
                    },
                    y: {
                        ticks: { font: { size: 11 } },
                        grid: { display: false }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function (ctx) {
                                var v = ctx.raw;
                                return (v > 0 ? '+' : '') + v.toFixed(2) + ' Mt CO₂';
                            }
                        }
                    }
                }
            }
        });
        updateWaterfallChart();
    }

    function updateWaterfallChart() {
        if (!waterfallChart || !DATA) return;

        var scenarioKey = document.getElementById('waterfallScenarioSelect').value || getFirstVisibleNonBaseline();
        var yr = String(nearestYear(selectedYear, getYears()));

        document.getElementById('waterfallTitle').textContent = 'Emissions Delta vs. Baseline — ' + yr;

        var basePlants = DATA.scenarios.baseline && DATA.scenarios.baseline.plant_detail ? DATA.scenarios.baseline.plant_detail[yr] : null;
        var scPlants = DATA.scenarios[scenarioKey] && DATA.scenarios[scenarioKey].plant_detail ? DATA.scenarios[scenarioKey].plant_detail[yr] : null;

        if (!basePlants || !scPlants) {
            waterfallChart.data.labels = ['No data for ' + yr];
            waterfallChart.data.datasets = [{ data: [0], backgroundColor: '#ddd' }];
            waterfallChart.update('active');
            return;
        }

        // Build lookup by orispl
        var baseMap = {};
        basePlants.forEach(function (p) { baseMap[p.orispl] = p; });

        var deltas = [];
        // Plants in scenario
        scPlants.forEach(function (p) {
            var baseP = baseMap[p.orispl];
            var baseE = baseP ? baseP.emissions_mt : 0;
            var delta = baseE - p.emissions_mt; // positive = reduction
            if (Math.abs(delta) > 0.001) {
                deltas.push({ name: p.name, delta: delta });
            }
        });
        // Plants in baseline but not in scenario (retired without listing)
        basePlants.forEach(function (p) {
            var found = scPlants.some(function (sp) { return sp.orispl === p.orispl; });
            if (!found && p.emissions_mt > 0.001) {
                deltas.push({ name: p.name + ' (retired)', delta: p.emissions_mt });
            }
        });

        // Sort by absolute delta descending
        deltas.sort(function (a, b) { return Math.abs(b.delta) - Math.abs(a.delta); });

        // Limit to 15
        var other = 0;
        if (deltas.length > 15) {
            for (var i = 15; i < deltas.length; i++) other += deltas[i].delta;
            deltas = deltas.slice(0, 15);
            if (Math.abs(other) > 0.001) deltas.push({ name: 'Other', delta: other });
        }

        var labels = deltas.map(function (d) { return d.name; });
        var values = deltas.map(function (d) { return d.delta; });
        var colors = values.map(function (v) { return v >= 0 ? '#6BA543' : '#DC2626'; });

        waterfallChart.data.labels = labels;
        waterfallChart.data.datasets = [{
            data: values,
            backgroundColor: colors,
            borderColor: colors,
            borderWidth: 1,
            borderRadius: 3,
            barPercentage: 0.7
        }];
        waterfallChart.update('active');
    }

    // ====================================================================
    // CHART 3: GENERATION MIX (Stacked Bar)
    // ====================================================================
    function buildGenMixChart() {
        var ctx = document.getElementById('genMixChart').getContext('2d');
        genMixChart = new Chart(ctx, {
            type: 'bar',
            data: { labels: [], datasets: [] },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 300 },
                scales: {
                    x: {
                        stacked: true,
                        grid: { display: false },
                        ticks: { font: { size: 11 } }
                    },
                    y: {
                        stacked: true,
                        title: { display: true, text: 'Generation (TWh)', font: { size: 12 } },
                        grid: { color: '#E0E6EF' },
                        beginAtZero: true,
                        ticks: { font: { size: 11 } }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function (ctx) {
                                var total = 0;
                                ctx.chart.data.datasets.forEach(function (ds) {
                                    total += ds.data[ctx.dataIndex] || 0;
                                });
                                var pct = total > 0 ? ((ctx.raw / total) * 100).toFixed(1) : '0.0';
                                return ctx.dataset.label + ': ' + ctx.raw.toFixed(1) + ' TWh (' + pct + '%)';
                            }
                        }
                    }
                }
            }
        });
        updateGenMixChart();
    }

    function updateGenMixChart() {
        if (!genMixChart || !DATA) return;

        var yr = String(nearestYear(selectedYear, getYears()));
        document.getElementById('genMixTitle').textContent = 'Fleet Generation by Fuel — ' + yr;

        var visible = getVisibleScenarios();
        var labels = visible.map(function (k) { return SCENARIO_LABELS[k] || k; });

        var datasets = FUEL_KEYS.map(function (fuel) {
            var data = visible.map(function (sKey) {
                var gen = DATA.scenarios[sKey].generation_by_fuel;
                if (!gen || !gen[yr]) return 0;
                return gen[yr][fuel] || 0;
            });
            return {
                label: FUEL_LABELS[fuel] || fuel,
                data: data,
                backgroundColor: FUEL_COLORS[fuel],
                borderColor: '#fff',
                borderWidth: 1,
                barPercentage: 0.7
            };
        });

        genMixChart.data.labels = labels;
        genMixChart.data.datasets = datasets;
        genMixChart.update('active');

        // Legend
        var legendEl = document.getElementById('genMixLegend');
        if (typeof buildLegend === 'function') {
            buildLegend(legendEl, FUEL_KEYS.map(function (f) {
                return { label: FUEL_LABELS[f], color: FUEL_COLORS[f], type: 'band' };
            }));
        }
    }

    // ====================================================================
    // CHART 4: EMISSIONS BY FUEL (Stacked Area)
    // ====================================================================
    function buildFuelEmissionsChart() {
        var ctx = document.getElementById('fuelEmissionsChart').getContext('2d');
        fuelEmissionsChart = new Chart(ctx, {
            type: 'line',
            data: { labels: [], datasets: [] },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 300 },
                interaction: { mode: 'index', intersect: false },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { font: { size: 12 }, maxRotation: 0 }
                    },
                    y: {
                        stacked: true,
                        title: { display: true, text: 'Emissions (Mt CO₂)', font: { size: 12 } },
                        grid: { color: '#E0E6EF' },
                        beginAtZero: true,
                        ticks: { font: { size: 11 } }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function (ctx) {
                                var total = 0;
                                ctx.chart.data.datasets.forEach(function (ds) {
                                    total += ds.data[ctx.dataIndex] || 0;
                                });
                                var pct = total > 0 ? ((ctx.raw / total) * 100).toFixed(1) : '0.0';
                                return ctx.dataset.label + ': ' + ctx.raw.toFixed(1) + ' Mt (' + pct + '%)';
                            }
                        }
                    }
                }
            }
        });
        updateFuelEmissionsChart();
    }

    function updateFuelEmissionsChart() {
        if (!fuelEmissionsChart || !DATA) return;

        var feSelect = document.getElementById('fuelScenarioSelect');
        var scenarioKey = feSelect.value || 'baseline';
        var sc = DATA.scenarios[scenarioKey];

        document.getElementById('fuelEmissionsTitle').textContent =
            'Emissions by Fuel Type — ' + (SCENARIO_LABELS[scenarioKey] || scenarioKey);

        var years = getYears();
        var labels = years.map(String);

        var datasets = EMISSION_FUEL_KEYS.map(function (fuel) {
            var data = years.map(function (y) {
                var emf = sc.emissions_by_fuel[String(y)];
                return emf ? (emf[fuel] || 0) : 0;
            });
            return {
                label: FUEL_LABELS[fuel] || fuel,
                data: data,
                borderColor: FUEL_COLORS[fuel],
                backgroundColor: hexToRgba(FUEL_COLORS[fuel], 0.55),
                borderWidth: 1.5,
                pointRadius: 2,
                pointHoverRadius: 4,
                fill: true,
                tension: 0.3
            };
        });

        fuelEmissionsChart.data.labels = labels;
        fuelEmissionsChart.data.datasets = datasets;
        fuelEmissionsChart.update('active');

        // Legend
        var legendEl = document.getElementById('fuelEmissionsLegend');
        if (typeof buildLegend === 'function') {
            buildLegend(legendEl, EMISSION_FUEL_KEYS.map(function (f) {
                return { label: FUEL_LABELS[f], color: FUEL_COLORS[f], type: 'band' };
            }));
        }
    }

    // ── Public API for sidebar integration ──
    window.FLEET_SCENARIOS_API = {
        addScenario: function (key, label, data) {
            if (!DATA) return;
            DATA.scenarios[key] = data;
            SCENARIO_LABELS[key] = label;
            SCENARIO_COLORS[key] = data.color || '#8B5CF6';
            scenarioVisibility[key] = true;
            buildControls();
            updateAllCharts();
        },
        removeScenario: function (key) {
            if (!DATA) return;
            delete DATA.scenarios[key];
            delete SCENARIO_LABELS[key];
            delete SCENARIO_COLORS[key];
            delete scenarioVisibility[key];
            buildControls();
            updateAllCharts();
        },
        updateAllCharts: updateAllCharts,
        getData: function () { return DATA; },
        getYears: getYears
    };

    // ── Boot ──
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
