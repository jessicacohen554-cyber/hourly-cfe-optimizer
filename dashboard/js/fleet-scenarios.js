// ============================================================================
// FLEET SCENARIOS — Baseline + Custom overlay dashboard
// ============================================================================
// Shows baseline fleet emissions fan (P10/P50/P90) from precomputed data.
// When the user edits the fleet via the sidebar and recalculates, a custom
// scenario overlay appears on the same chart. Both fans share the same
// starting point and diverge when the user's changes (CCS/retirement) kick in.
// ============================================================================

(function () {
    'use strict';

    var TARGET_STYLES = {
        sbti_15:     { color: '#DC2626', dash: [8, 4], label: 'SBTi 1.5°C' },
        at_power_nz: { color: '#6366F1', dash: [],     label: 'AT Power NZ' },
        custom:      { color: '#6B7280', dash: [4, 4], label: 'Custom' }
    };

    var FUEL_KEYS   = ['nuclear', 'geothermal', 'wind', 'solar', 'hydro', 'gas_ccgt', 'gas_ct', 'gas_oil_ct', 'oil_ct', 'ccs_ccgt'];
    var FUEL_LABELS = { nuclear: 'Nuclear', geothermal: 'Geothermal', wind: 'Wind', solar: 'Solar', hydro: 'Hydro', gas_ccgt: 'Gas CCGT', gas_ct: 'Gas CT', gas_oil_ct: 'Gas/Oil', oil_ct: 'Oil', ccs_ccgt: 'CCS-CCGT' };
    var FUEL_COLORS = {
        nuclear:    RESOURCE_COLORS.nuclear,
        geothermal: RESOURCE_COLORS.geothermal || '#D97706',
        wind:       RESOURCE_COLORS.wind || '#22C55E',
        solar:      RESOURCE_COLORS.solar || '#F59E0B',
        hydro:      RESOURCE_COLORS.hydro || '#0EA5E9',
        gas_ccgt:   RESOURCE_COLORS.fossilGas,
        gas_ct:     RESOURCE_COLORS.fossilGasCT || '#007FA4',
        gas_oil_ct: '#8B7355',
        oil_ct:     RESOURCE_COLORS.fossilOil,
        ccs_ccgt:   RESOURCE_COLORS.ccs
    };
    var EMISSION_FUEL_KEYS = ['gas_ccgt', 'gas_ct', 'gas_oil_ct', 'oil_ct', 'ccs_ccgt'];

    // ── State ──
    var DATA = null;
    var customScenario = null;  // Set when user recalculates from sidebar
    var savedScenarioOverlays = []; // Array of saved scenarios with .isVisible, .color, .results
    var selectedTargets = ['sbti_15'];
    var selectedYear = 2030;
    var selectedFossilCost = 'All';
    var fanChart = null;
    var waterfallChart = null;
    var genMixChart = null;
    var fuelEmissionsChart = null;

    // ── Helpers ──
    function getYears() {
        if (!DATA || !DATA.scenarios || !DATA.scenarios.baseline) return [];
        return Object.keys(DATA.scenarios.baseline.envelope).map(Number).sort(function (a, b) { return a - b; });
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

    function getEnvelope(sc) {
        if (selectedFossilCost !== 'All' && sc.envelope_by_fossil_cost &&
            sc.envelope_by_fossil_cost[selectedFossilCost]) {
            return sc.envelope_by_fossil_cost[selectedFossilCost];
        }
        return sc.envelope;
    }

    // ── Init ──
    function init() {
        loadData();
    }

    function loadData() {
        fetch('data/fleet_scenario_results_sample.json')
            .then(function (r) { if (r.ok) return r.json(); throw new Error('No data'); })
            .then(onDataLoaded)
            .catch(showError);
    }

    function showError() {
        var section = document.querySelector('.content-section');
        if (section) {
            section.innerHTML = '<div class="error-banner">Failed to load fleet scenario results. Run the fleet scenario pipeline first, or place <code>fleet_scenario_results_sample.json</code> in <code>frontend/data/</code>.</div>';
        }
    }

    function onDataLoaded(data) {
        DATA = data;
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
        // Target checkboxes
        var radiosEl = document.getElementById('targetRadios');
        radiosEl.innerHTML = '';
        var targetKeys = DATA.targets ? Object.keys(DATA.targets) : [];
        targetKeys.forEach(function (tKey, i) {
            var t = DATA.targets[tKey];
            var lbl = document.createElement('label');
            lbl.className = 'target-radio';
            lbl.innerHTML =
                '<input type="checkbox" data-target="' + tKey + '"' + (i === 0 ? ' checked' : '') + '> ' +
                t.label;
            radiosEl.appendChild(lbl);
        });
        var noneLbl = document.createElement('label');
        noneLbl.className = 'target-radio';
        noneLbl.innerHTML = '<input type="checkbox" data-target="none"> None';
        radiosEl.appendChild(noneLbl);

        radiosEl.addEventListener('change', function (e) {
            var cb = e.target;
            if (!cb.dataset || !cb.dataset.target) return;
            var tKey = cb.dataset.target;
            if (tKey === 'none') {
                if (cb.checked) {
                    selectedTargets = [];
                    radiosEl.querySelectorAll('input[type="checkbox"]').forEach(function (box) {
                        if (box.dataset.target !== 'none') box.checked = false;
                    });
                }
            } else {
                var noneBox = radiosEl.querySelector('input[data-target="none"]');
                if (noneBox) noneBox.checked = false;
                if (cb.checked) {
                    if (selectedTargets.indexOf(tKey) === -1) selectedTargets.push(tKey);
                } else {
                    selectedTargets = selectedTargets.filter(function (t) { return t !== tKey; });
                }
            }
            updateFanChart();
            updateFanLegend();
        });

        // Fossil cost toggle
        var fossilToggle = document.getElementById('fossilCostToggle');
        if (fossilToggle) {
            fossilToggle.addEventListener('click', function (e) {
                var btn = e.target.closest('.toggle-btn');
                if (!btn) return;
                fossilToggle.querySelectorAll('.toggle-btn').forEach(function (b) {
                    b.classList.remove('active');
                });
                btn.classList.add('active');
                selectedFossilCost = btn.dataset.value;
                updateFanChart();
                updateFanLegend();
            });
        }

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
    }

    function updateAllCharts() {
        // Dismiss tooltip when chart data changes (e.g. scenario toggled)
        var tooltipEl = document.getElementById('fanTooltip');
        if (tooltipEl) tooltipEl.classList.remove('visible');

        updateFanChart();
        updateFanLegend();
        updateWaterfallChart();
        updateGenMixChart();
        updateFuelEmissionsChart();
    }

    // ====================================================================
    // CHART 1: EMISSIONS FAN CHART — Baseline + Custom overlay
    // ====================================================================
    var BASELINE_COLOR = '#7E8083';
    var CUSTOM_COLOR = '#2372B9';

    // Empty-state inline plugin for fan chart
    var fanEmptyStatePlugin = {
        id: 'fanEmptyState',
        afterDraw: function (chart) {
            if (chart.data.datasets.length === 0) {
                var ctx = chart.ctx;
                var w = chart.width, h = chart.height;
                ctx.save();
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillStyle = '#9CA3AF';
                ctx.font = '14px sans-serif';
                ctx.fillText('No scenarios visible. Toggle a scenario on in the sidebar.', w / 2, h / 2);
                ctx.restore();
            }
        }
    };

    function buildFanChart() {
        var ctx = document.getElementById('fanChart').getContext('2d');
        fanChart = new Chart(ctx, {
            type: 'line',
            data: { labels: [], datasets: [] },
            plugins: [fanEmptyStatePlugin],
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
                            callback: function (val) {
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
                    legend: {
                        display: true,
                        position: 'top',
                        align: 'end',
                        labels: {
                            filter: function (item, chartData) {
                                var ds = chartData.datasets[item.datasetIndex];
                                return ds && ds._showInLegend;
                            },
                            sort: function (a, b, chartData) {
                                var dsA = chartData.datasets[a.datasetIndex];
                                var dsB = chartData.datasets[b.datasetIndex];
                                // Baseline always first
                                if (dsA._isBaseline && !dsB._isBaseline) return -1;
                                if (!dsA._isBaseline && dsB._isBaseline) return 1;
                                // Targets after scenarios
                                if (dsA._target && !dsB._target) return 1;
                                if (!dsA._target && dsB._target) return -1;
                                // Then by creation order (dataset index)
                                return a.datasetIndex - b.datasetIndex;
                            },
                            generateLabels: function (chart) {
                                var defaultLabels = Chart.defaults.plugins.legend.labels.generateLabels(chart);
                                return defaultLabels.filter(function (item) {
                                    var ds = chart.data.datasets[item.datasetIndex];
                                    return ds && ds._showInLegend;
                                }).map(function (item) {
                                    var ds = chart.data.datasets[item.datasetIndex];
                                    if (ds._isBaseline) {
                                        item.text = ds._scenarioName + ' (Baseline)';
                                    } else if (ds._scenarioName) {
                                        item.text = ds._scenarioName;
                                    }
                                    item.pointStyle = 'line';
                                    item.lineWidth = ds.borderWidth || 2.5;
                                    item.strokeStyle = ds.borderColor;
                                    item.lineDash = ds.borderDash || [];
                                    return item;
                                });
                            },
                            usePointStyle: true,
                            pointStyle: 'line',
                            padding: 16,
                            font: { size: 12 }
                        }
                    },
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

        // Dismiss tooltip on mouse leave
        var chartCanvas = document.getElementById('fanChart');
        if (chartCanvas) {
            chartCanvas.addEventListener('mouseleave', function () {
                var tooltipEl = document.getElementById('fanTooltip');
                if (tooltipEl) tooltipEl.classList.remove('visible');
            });
        }

        updateFanChart();
    }

    // ── Determine color mode ──
    function getColorMode() {
        // Collect visible scenarios: baseline is always present, plus savedScenarioOverlays
        var visibleCount = savedScenarioOverlays.length;
        if (customScenario) visibleCount++;
        // Single mode: 0 overlays (baseline only), or exactly 1 overlay
        // where the overlay count ≤1 (baseline + 1 custom = single)
        var hasBaselineOverlay = savedScenarioOverlays.some(function (s) { return s.isBaseline; });
        if (visibleCount <= 1 || (visibleCount === 2 && hasBaselineOverlay)) {
            return 'single';
        }
        return 'multi';
    }

    // ── Build 5-band fan datasets for a single scenario ──
    // mode: 'single' = green/red semantic colors, 'multi' = scenario's own color
    // p50Style: { color, width, dash } for the p50 line specifically
    function addFanDatasets(datasets, env, years, color, label, mode, p50Style, meta) {
        var p10 = [], p25 = [], p50 = [], p75 = [], p90 = [];
        years.forEach(function (y) {
            var e = env[String(y)];
            if (e) {
                p10.push(e.p10);
                p25.push(e.p25 != null ? e.p25 : (e.p10 + e.p50) / 2);
                p50.push(e.p50);
                p75.push(e.p75 != null ? e.p75 : (e.p50 + e.p90) / 2);
                p90.push(e.p90);
            } else {
                p10.push(null); p25.push(null); p50.push(null);
                p75.push(null); p90.push(null);
            }
        });

        var startIdx = datasets.length;

        if (mode === 'single') {
            // ── Single-scenario mode: green (optimistic) below p50, red (pessimistic) above ──
            var greenLight = 'rgba(34, 197, 94, 0.08)';
            var greenMed   = 'rgba(34, 197, 94, 0.15)';
            var greenLine  = 'rgb(34, 197, 94)';
            var redLight   = 'rgba(239, 68, 68, 0.08)';
            var redMed     = 'rgba(239, 68, 68, 0.15)';
            var redLine    = 'rgb(239, 68, 68)';

            // 1. p10 (invisible bottom boundary)
            datasets.push({
                label: label + ' p10', data: p10,
                borderColor: 'transparent', backgroundColor: 'transparent',
                borderWidth: 0, pointRadius: 0, fill: false,
                tension: 0.3, showLegend: false, _scenarioKey: label, _band: 'p10'
            });
            // 2. p10→p25 fill (light green)
            datasets.push({
                label: label + ' p10-p25', data: p25,
                borderColor: 'transparent', backgroundColor: greenLight,
                borderWidth: 0, pointRadius: 0,
                fill: { target: startIdx, above: greenLight, below: greenLight },
                tension: 0.3, _scenarioKey: label, _band: 'fill'
            });
            // 3. p25 dashed line (green)
            datasets.push({
                label: label + ' P25', data: p25,
                borderColor: greenLine, backgroundColor: 'transparent',
                borderWidth: 1.5, borderDash: [6, 4], pointRadius: 0,
                fill: false, tension: 0.3, _scenarioKey: label, _band: 'p25'
            });
            // 4. p25→p50 fill (medium green)
            datasets.push({
                label: label + ' p25-p50', data: p50,
                borderColor: 'transparent', backgroundColor: greenMed,
                borderWidth: 0, pointRadius: 0,
                fill: { target: startIdx + 2, above: greenMed, below: greenMed },
                tension: 0.3, _scenarioKey: label, _band: 'fill'
            });
            // 5. p50 solid line
            datasets.push({
                label: p50Style.legendLabel || label,
                data: p50,
                borderColor: p50Style.color,
                backgroundColor: 'transparent',
                borderWidth: p50Style.width || 2.5,
                borderDash: p50Style.dash || [],
                pointRadius: p50Style.pointRadius != null ? p50Style.pointRadius : 3,
                pointBackgroundColor: p50Style.color,
                pointHoverRadius: 5,
                fill: false, tension: 0.3,
                _scenarioKey: label, _band: 'p50',
                _showInLegend: true,
                _isBaseline: meta && meta.isBaseline,
                _scenarioName: (meta && meta.scenarioName) || label
            });
            // 6. p50→p75 fill (medium red)
            datasets.push({
                label: label + ' p50-p75', data: p75,
                borderColor: 'transparent', backgroundColor: redMed,
                borderWidth: 0, pointRadius: 0,
                fill: { target: startIdx + 4, above: redMed, below: redMed },
                tension: 0.3, _scenarioKey: label, _band: 'fill'
            });
            // 7. p75 dashed line (red)
            datasets.push({
                label: label + ' P75', data: p75,
                borderColor: redLine, backgroundColor: 'transparent',
                borderWidth: 1.5, borderDash: [6, 4], pointRadius: 0,
                fill: false, tension: 0.3, _scenarioKey: label, _band: 'p75'
            });
            // 8. p75→p90 fill (light red)
            datasets.push({
                label: label + ' p75-p90', data: p90,
                borderColor: 'transparent', backgroundColor: redLight,
                borderWidth: 0, pointRadius: 0,
                fill: { target: startIdx + 6, above: redLight, below: redLight },
                tension: 0.3, _scenarioKey: label, _band: 'fill'
            });
            // 9. p90 (invisible top boundary)
            datasets.push({
                label: label + ' p90', data: p90,
                borderColor: 'transparent', backgroundColor: 'transparent',
                borderWidth: 0, pointRadius: 0, fill: false,
                tension: 0.3, _scenarioKey: label, _band: 'p90'
            });
        } else {
            // ── Multi-scenario mode: color-matched shading using scenario's assigned color ──
            var veryLight = hexToRgba(color, 0.06);
            var light     = hexToRgba(color, 0.12);

            // 1. p10 (invisible bottom boundary)
            datasets.push({
                label: label + ' p10', data: p10,
                borderColor: 'transparent', backgroundColor: 'transparent',
                borderWidth: 0, pointRadius: 0, fill: false,
                tension: 0.3, _scenarioKey: label, _band: 'p10'
            });
            // 2. p10→p25 fill
            datasets.push({
                label: label + ' p10-p25', data: p25,
                borderColor: 'transparent', backgroundColor: veryLight,
                borderWidth: 0, pointRadius: 0,
                fill: { target: startIdx, above: veryLight, below: veryLight },
                tension: 0.3, _scenarioKey: label, _band: 'fill'
            });
            // 3. p25 dashed line
            datasets.push({
                label: label + ' P25', data: p25,
                borderColor: color, backgroundColor: 'transparent',
                borderWidth: 1, borderDash: [5, 3], pointRadius: 0,
                fill: false, tension: 0.3, _scenarioKey: label, _band: 'p25'
            });
            // 4. p25→p50 fill
            datasets.push({
                label: label + ' p25-p50', data: p50,
                borderColor: 'transparent', backgroundColor: light,
                borderWidth: 0, pointRadius: 0,
                fill: { target: startIdx + 2, above: light, below: light },
                tension: 0.3, _scenarioKey: label, _band: 'fill'
            });
            // 5. p50 solid line
            datasets.push({
                label: label,
                data: p50,
                borderColor: color,
                backgroundColor: 'transparent',
                borderWidth: 2.5,
                borderDash: [],
                pointRadius: 3,
                pointBackgroundColor: color,
                pointHoverRadius: 5,
                fill: false, tension: 0.3,
                _scenarioKey: label, _band: 'p50',
                _showInLegend: true,
                _isBaseline: meta && meta.isBaseline,
                _scenarioName: (meta && meta.scenarioName) || label
            });
            // 6. p50→p75 fill
            datasets.push({
                label: label + ' p50-p75', data: p75,
                borderColor: 'transparent', backgroundColor: light,
                borderWidth: 0, pointRadius: 0,
                fill: { target: startIdx + 4, above: light, below: light },
                tension: 0.3, _scenarioKey: label, _band: 'fill'
            });
            // 7. p75 dashed line
            datasets.push({
                label: label + ' P75', data: p75,
                borderColor: color, backgroundColor: 'transparent',
                borderWidth: 1, borderDash: [5, 3], pointRadius: 0,
                fill: false, tension: 0.3, _scenarioKey: label, _band: 'p75'
            });
            // 8. p75→p90 fill
            datasets.push({
                label: label + ' p75-p90', data: p90,
                borderColor: 'transparent', backgroundColor: veryLight,
                borderWidth: 0, pointRadius: 0,
                fill: { target: startIdx + 6, above: veryLight, below: veryLight },
                tension: 0.3, _scenarioKey: label, _band: 'fill'
            });
            // 9. p90 (invisible)
            datasets.push({
                label: label + ' p90', data: p90,
                borderColor: 'transparent', backgroundColor: 'transparent',
                borderWidth: 0, pointRadius: 0, fill: false,
                tension: 0.3, _scenarioKey: label, _band: 'p90'
            });
        }
    }

    function updateFanChart() {
        if (!fanChart || !DATA) return;
        var years = getYears();
        var labels = years.map(String);
        var datasets = [];
        var baseline = DATA.scenarios.baseline;
        var mode = getColorMode();

        if (mode === 'single') {
            // ── Single-scenario mode ──
            // Determine if it's baseline-only, or baseline + one custom/saved
            var customOrSaved = customScenario ? customScenario : null;
            if (!customOrSaved && savedScenarioOverlays.length === 1) {
                customOrSaved = savedScenarioOverlays[0];
            }

            if (customOrSaved && baseline) {
                // Baseline + 1 custom: baseline gets gray p50, custom gets green/red fan
                var baseEnv = getEnvelope(baseline);
                // Only show baseline p50 as a gray reference line
                var baseP50 = [];
                years.forEach(function (y) {
                    var e = baseEnv[String(y)];
                    baseP50.push(e ? e.p50 : null);
                });
                datasets.push({
                    label: 'Baseline',
                    data: baseP50,
                    borderColor: '#6B7280',
                    backgroundColor: 'transparent',
                    borderWidth: 2, borderDash: [],
                    pointRadius: 0, pointHoverRadius: 3,
                    fill: false, tension: 0.3,
                    _scenarioKey: 'Baseline', _band: 'p50', _showInLegend: true,
                    _isBaseline: true, _scenarioName: 'Baseline'
                });

                // Custom scenario gets the full green/red fan
                var custEnv = customOrSaved.results ? customOrSaved.results.envelope : customOrSaved.envelope;
                var custColor = customOrSaved.color || CUSTOM_COLOR;
                var custName = customOrSaved.name || 'Custom';
                addFanDatasets(datasets, custEnv, years, custColor, custName, 'single', {
                    color: custColor, width: 2.5, dash: [], pointRadius: 3,
                    legendLabel: custName
                }, { isBaseline: false, scenarioName: custName });
            } else if (baseline) {
                // Baseline only: show full green/red fan with baseline's own color for p50
                var baseEnv = getEnvelope(baseline);
                addFanDatasets(datasets, baseEnv, years, BASELINE_COLOR, 'Baseline', 'single', {
                    color: BASELINE_COLOR, width: 2.5, dash: [], pointRadius: 3,
                    legendLabel: 'Baseline'
                }, { isBaseline: true, scenarioName: 'Baseline' });
            }
        } else {
            // ── Multi-scenario mode ──
            // Baseline always gets its fan
            if (baseline) {
                var baseEnv = getEnvelope(baseline);
                addFanDatasets(datasets, baseEnv, years, BASELINE_COLOR, 'Baseline', 'multi', null,
                    { isBaseline: true, scenarioName: 'Baseline' });
            }

            // Custom overlay
            if (customScenario) {
                addFanDatasets(datasets, customScenario.envelope, years, CUSTOM_COLOR, 'Custom', 'multi', null,
                    { isBaseline: false, scenarioName: 'Custom' });
            }

            // Saved scenario overlays — sorted: baseline first, then by creation date
            var sortedOverlays = savedScenarioOverlays.slice().sort(function (a, b) {
                if (a.isBaseline && !b.isBaseline) return -1;
                if (!a.isBaseline && b.isBaseline) return 1;
                var aTime = a.createdAt ? new Date(a.createdAt).getTime() : 0;
                var bTime = b.createdAt ? new Date(b.createdAt).getTime() : 0;
                return aTime - bTime;
            });
            sortedOverlays.forEach(function (s) {
                if (s.results && s.results.envelope) {
                    addFanDatasets(datasets, s.results.envelope, years, s.color, s.name, 'multi', null,
                        { isBaseline: !!s.isBaseline, scenarioName: s.name || 'Saved' });
                }
            });
        }

        // Target overlays (same in both modes)
        selectedTargets.forEach(function (selTgt) {
            if (!selTgt || selTgt === 'none' || !DATA.targets || !DATA.targets[selTgt]) return;
            var tgt = DATA.targets[selTgt];
            var style = TARGET_STYLES[selTgt] || TARGET_STYLES.custom;
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
                _target: true,
                _showInLegend: true
            });
        });

        fanChart.data.labels = labels;
        fanChart.data.datasets = datasets;
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

    // ── Tooltip helper: format a scenario's p50 row with % reduction ──
    function tooltipP50Row(env, year, color, label, baseline2023, isBaseline) {
        var e = env[year];
        if (!e) return '';
        var pctStr = '';
        if (baseline2023 && baseline2023 > 0) {
            var pct = ((baseline2023 - e.p50) / baseline2023 * 100).toFixed(1);
            pctStr = '<span style="opacity:0.7;margin-left:6px;">(' +
                (pct >= 0 ? '−' : '+') + Math.abs(pct) + '% vs 2023)</span>';
        }
        var baselineSuffix = isBaseline ? '<span style="opacity:0.5;font-weight:400;"> (Baseline)</span>' : '';
        return '<div class="fan-tooltip-row">' +
            '<span class="fan-tooltip-swatch" style="background:' + color + '"></span>' +
            '<span class="fan-tooltip-label">' + label + baselineSuffix + '</span>' +
            '<span class="fan-tooltip-vals">' + e.p50.toFixed(1) + ' Mt' + pctStr + '</span></div>';
    }

    // ── Tooltip helper: format full envelope detail row (expanded view) ──
    function tooltipEnvelopeRow(env, year, color, label, baseline2023) {
        var e = env[year];
        if (!e) return '';
        var p25 = e.p25 != null ? e.p25 : (e.p10 + e.p50) / 2;
        var p75 = e.p75 != null ? e.p75 : (e.p50 + e.p90) / 2;
        return '<div style="font-size:0.75rem;color:var(--text-dim);padding-left:20px;margin-top:-2px;margin-bottom:4px;">' +
            'P10: ' + e.p10.toFixed(1) + '  P25: ' + p25.toFixed(1) +
            '  <strong>P50: ' + e.p50.toFixed(1) + '</strong>' +
            '  P75: ' + p75.toFixed(1) + '  P90: ' + e.p90.toFixed(1) + '</div>';
    }

    // ── Custom tooltip ──
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

            var baseEnv = DATA.scenarios.baseline ? getEnvelope(DATA.scenarios.baseline) : null;
            var baseYear = baseEnv ? baseEnv[year] : null;
            var baseline2023 = baseEnv && baseEnv['2023'] ? baseEnv['2023'].p50 : null;

            // Baseline row — always first
            if (baseYear && baseEnv) {
                html += tooltipP50Row(baseEnv, year, BASELINE_COLOR, 'Baseline', baseline2023, true);
            }

            // Custom (unsaved) scenario row
            if (customScenario && customScenario.envelope) {
                html += tooltipP50Row(customScenario.envelope, year, CUSTOM_COLOR, 'Custom', baseline2023, false);
                if (baseYear && customScenario.envelope[year]) {
                    var delta = customScenario.envelope[year].p50 - baseYear.p50;
                    var cls = delta < 0 ? 'gap-negative' : 'gap-positive';
                    var sign = delta > 0 ? '+' : '';
                    html += '<div class="fan-tooltip-gap"><span class="' + cls + '">' +
                        sign + delta.toFixed(1) + ' Mt vs Baseline</span></div>';
                }
            }

            // Saved scenario rows — sorted: baseline first, then by creation date
            var sortedSaved = savedScenarioOverlays.slice().sort(function (a, b) {
                if (a.isBaseline && !b.isBaseline) return -1;
                if (!a.isBaseline && b.isBaseline) return 1;
                var aTime = a.createdAt ? new Date(a.createdAt).getTime() : 0;
                var bTime = b.createdAt ? new Date(b.createdAt).getTime() : 0;
                return aTime - bTime;
            });
            sortedSaved.forEach(function (s) {
                if (!s.results || !s.results.envelope) return;
                html += tooltipP50Row(s.results.envelope, year, s.color, s.name || 'Saved', baseline2023, !!s.isBaseline);
                if (baseYear && s.results.envelope[year]) {
                    var sDelta = s.results.envelope[year].p50 - baseYear.p50;
                    var sCls = sDelta < 0 ? 'gap-negative' : 'gap-positive';
                    var sSign = sDelta > 0 ? '+' : '';
                    html += '<div class="fan-tooltip-gap"><span class="' + sCls + '">' +
                        sSign + sDelta.toFixed(1) + ' Mt vs Baseline</span></div>';
                }
            });

            // Gap to targets
            selectedTargets.forEach(function (selTgt) {
                if (!selTgt || selTgt === 'none' || !DATA.targets || !DATA.targets[selTgt]) return;
                var tgt = DATA.targets[selTgt];
                var tgtVal = tgt.trajectory[year];
                if (tgtVal == null) return;
                var refP50 = customScenario && customScenario.envelope[year] ? customScenario.envelope[year].p50 : (baseYear ? baseYear.p50 : null);
                if (refP50 == null) return;
                var gap = refP50 - tgtVal;
                var cls = gap > 0 ? 'gap-positive' : 'gap-negative';
                var sign = gap > 0 ? '+' : '';
                var check = gap <= 0 ? ' ✓' : '';
                html += '<div class="fan-tooltip-gap"><strong>Gap to ' + tgt.label + ':</strong> ' +
                    '<span class="' + cls + '">' + sign + gap.toFixed(1) + ' Mt' + check + '</span></div>';
            });

            tooltipEl.innerHTML = html;
        }

        var canvas = context.chart.canvas;
        var rect = canvas.getBoundingClientRect();
        var left = tooltipModel.caretX;
        var top = tooltipModel.caretY;
        if (left + 320 > rect.width) left = left - 340;
        if (left < 10) left = 10;
        if (top < 10) top = 10;
        tooltipEl.style.left = left + 'px';
        tooltipEl.style.top = top + 'px';
        tooltipEl.classList.add('visible');
    }

    // ── Fan legend (below chart — shows band key in single mode) ──
    function updateFanLegend() {
        var el = document.getElementById('fanLegend');
        if (!el || !DATA) return;
        var mode = getColorMode();
        var items = [];

        if (mode === 'single') {
            // Show green/red band explanation
            items.push({ label: 'P25 (optimistic)', color: 'rgb(34, 197, 94)', type: 'dashed' });
            items.push({ label: 'P10–P50 range', color: 'rgba(34, 197, 94, 0.15)', type: 'band' });
            items.push({ label: 'P75 (pessimistic)', color: 'rgb(239, 68, 68)', type: 'dashed' });
            items.push({ label: 'P50–P90 range', color: 'rgba(239, 68, 68, 0.15)', type: 'band' });
        }
        // In multi mode, the Chart.js built-in legend handles p50 lines

        // Targets always shown
        selectedTargets.forEach(function (selTgt) {
            if (selTgt && selTgt !== 'none' && DATA.targets && DATA.targets[selTgt]) {
                var style = TARGET_STYLES[selTgt] || TARGET_STYLES.custom;
                items.push({
                    label: DATA.targets[selTgt].label,
                    color: style.color,
                    type: style.dash.length ? 'dashed' : 'line'
                });
            }
        });

        var html = items.map(function(item) {
            var swatch;
            if (item.type === 'dashed') {
                swatch = '<span style="width:24px;height:0;border-top:2px dashed ' + item.color + ';display:inline-block;"></span>';
            } else {
                swatch = '<span style="width:24px;height:10px;border-radius:3px;background:' + item.color + ';display:inline-block;"></span>';
            }
            return '<span style="display:inline-flex;align-items:center;gap:4px;margin-right:16px;">' +
                swatch + item.label + '</span>';
        }).join('');

        // In single mode, add a green/red shading annotation note below the legend items
        if (mode === 'single') {
            html += '<div style="margin-top:6px;font-size:0.78rem;color:var(--text-dim);font-style:italic;">' +
                '<span style="color:rgb(34,197,94);font-weight:600;">Green</span> = optimistic (lower emissions), ' +
                '<span style="color:rgb(239,68,68);font-weight:600;">Red</span> = pessimistic (higher emissions)</div>';
        }

        el.innerHTML = html;
    }

    // ====================================================================
    // CHART 2: PLANT-LEVEL WATERFALL (Custom vs Baseline)
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

        var yr = String(nearestYear(selectedYear, getYears()));
        document.getElementById('waterfallTitle').textContent = 'Emissions Delta vs. Baseline — ' + yr;

        if (!customScenario) {
            waterfallChart.data.labels = ['Edit fleet to see changes'];
            waterfallChart.data.datasets = [{ data: [0], backgroundColor: '#ddd' }];
            waterfallChart.update('active');
            return;
        }

        var basePlants = DATA.scenarios.baseline && DATA.scenarios.baseline.plant_detail ? DATA.scenarios.baseline.plant_detail[yr] : null;
        var scPlants = customScenario.plant_detail ? customScenario.plant_detail[yr] : null;

        if (!basePlants || !scPlants) {
            waterfallChart.data.labels = ['No data for ' + yr];
            waterfallChart.data.datasets = [{ data: [0], backgroundColor: '#ddd' }];
            waterfallChart.update('active');
            return;
        }

        var baseMap = {};
        basePlants.forEach(function (p) { baseMap[p.orispl] = p; });

        var deltas = [];
        scPlants.forEach(function (p) {
            var baseP = baseMap[p.orispl];
            var baseE = baseP ? baseP.emissions_mt : 0;
            var delta = baseE - p.emissions_mt;
            if (Math.abs(delta) > 0.001) {
                deltas.push({ name: p.name, delta: delta });
            }
        });
        basePlants.forEach(function (p) {
            var found = scPlants.some(function (sp) { return sp.orispl === p.orispl; });
            if (!found && p.emissions_mt > 0.001) {
                deltas.push({ name: p.name + ' (retired)', delta: p.emissions_mt });
            }
        });

        deltas.sort(function (a, b) { return Math.abs(b.delta) - Math.abs(a.delta); });

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
    // CHART 3: GENERATION MIX (Stacked Bar) — Baseline vs Custom
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
                    x: { stacked: true, grid: { display: false }, ticks: { font: { size: 11 } } },
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

        var scenarios = ['baseline'];
        var labels = ['Baseline'];
        if (customScenario) {
            scenarios.push('custom');
            labels.push('Custom');
        }

        var datasets = FUEL_KEYS.map(function (fuel) {
            var data = scenarios.map(function (sKey) {
                var sc = sKey === 'custom' ? customScenario : DATA.scenarios[sKey];
                if (!sc) return 0;
                var gen = sc.generation_by_fuel;
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

        var legendEl = document.getElementById('genMixLegend');
        legendEl.innerHTML = FUEL_KEYS.map(function (f) {
            return '<span style="display:inline-flex;align-items:center;gap:4px;margin-right:16px;">' +
                '<span style="width:24px;height:10px;border-radius:3px;background:' + FUEL_COLORS[f] + ';display:inline-block;"></span>' +
                (FUEL_LABELS[f] || f) + '</span>';
        }).join('');
    }

    // ====================================================================
    // CHART 4: EMISSIONS BY FUEL (Stacked Area) — Custom or Baseline
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
                    x: { grid: { display: false }, ticks: { font: { size: 12 }, maxRotation: 0 } },
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

        // Use custom scenario if available, otherwise baseline
        var sc = customScenario || DATA.scenarios.baseline;
        var scenarioLabel = customScenario ? 'Custom' : 'Baseline';

        document.getElementById('fuelEmissionsTitle').textContent =
            'Emissions by Fuel Type — ' + scenarioLabel;

        var years = getYears();
        var labels = years.map(String);

        var datasets = EMISSION_FUEL_KEYS.map(function (fuel) {
            var data = years.map(function (y) {
                var emf = sc.emissions_by_fuel ? sc.emissions_by_fuel[String(y)] : null;
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

        var legendEl = document.getElementById('fuelEmissionsLegend');
        legendEl.innerHTML = EMISSION_FUEL_KEYS.map(function (f) {
            return '<span style="display:inline-flex;align-items:center;gap:4px;margin-right:16px;">' +
                '<span style="width:24px;height:10px;border-radius:3px;background:' + FUEL_COLORS[f] + ';display:inline-block;"></span>' +
                (FUEL_LABELS[f] || f) + '</span>';
        }).join('');
    }

    // ── Public API for sidebar integration ──
    window.FLEET_SCENARIOS_API = {
        setCustomScenario: function (label, data) {
            customScenario = data;
            updateAllCharts();
        },
        clearCustomScenario: function () {
            customScenario = null;
            updateAllCharts();
        },
        // Legacy compatibility — addScenario now sets as custom overlay
        addScenario: function (key, label, data) {
            customScenario = data;
            updateAllCharts();
        },
        removeScenario: function (key) {
            customScenario = null;
            updateAllCharts();
        },
        // Saved scenarios — array of {name, color, results, isBaseline}
        setSavedScenarios: function (scenarios) {
            savedScenarioOverlays = scenarios || [];
            updateAllCharts();
        },
        getSavedScenarioOverlays: function () { return savedScenarioOverlays; },
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
