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
        sbti_15:     { color: '#ad1dca', dash: [8, 4], label: 'SBTi 1.5°C' },
        at_power_nz: { color: '#6366F1', dash: [8, 4],     label: 'AT Power NZ' },
        custom:      { color: '#000205', dash: [4, 4], label: 'Custom' }
    };

    var FUEL_KEYS   = ['nuclear', 'geothermal', 'hydro', 'wind', 'solar', 'battery', 'ldes', 'ccs_ccgt', 'gas_ccgt', 'gas_ct', 'gas_oil_ct', 'oil_ct'];
    var FUEL_LABELS = { nuclear: 'Nuclear', geothermal: 'Geothermal', wind: 'Wind', solar: 'Solar', hydro: 'Hydro', battery: 'Battery', ldes: 'LDES', gas_ccgt: 'Gas CCGT', gas_ct: 'Gas CT', gas_oil_ct: 'Gas/Oil', oil_ct: 'Oil', ccs_ccgt: 'CCS-CCGT' };
    var FUEL_COLORS = {
        nuclear:    '#2372B9',
        geothermal: '#F47B27',
        wind:       '#6BA543',
        solar:      '#FBB254',
        hydro:      '#007FA4',
        battery:    '#651dda',
        ldes:       '#651dda9a',
        gas_ccgt:   '#7F8F97',
        gas_ct:     '#7E8083',
        gas_oil_ct: '#8B7355',
        oil_ct:     '#6b5942',
        ccs_ccgt:   '#CADB2E',
    };
    var EMISSION_FUEL_KEYS = ['gas_ccgt', 'gas_ct', 'gas_oil_ct', 'oil_ct', 'ccs_ccgt'];


    // ── State ──
    var DATA = null;
    var customScenario = null;  // Legacy — no longer auto-displayed; kept for API compat
    var savedScenarioOverlays = []; // Array of saved scenarios with .isVisible, .color, .results
    var selectedTargets = ['sbti_15'];
    var selectedYear = 2030;
    var selectedFossilCost = 'All';
    var selectedLowerScenarioId = null; // null = baseline; string = saved scenario ID
    var fanChart = null;
    var waterfall2023Chart = null;
    var genMixChart = null;
    var newCleanChart = null;
    var intensityFanChart = null;

    var topEmittersChart = null;
    var ccgtTierChart = null;

    // ── Clean fuel keys for % low-carbon calculation ──
    var CLEAN_FUEL_SET = { nuclear: 1, geothermal: 1, hydro: 1, wind: 1, solar: 1, battery: 1, ldes: 1, ccs_ccgt: 1 };

    // ── Plant color palette for top emitters chart ──
    var PLANT_PALETTE = ['#E53935','#1E88E5','#43A047','#FB8C00','#8E24AA','#00ACC1','#F4511E','#3949AB','#7CB342','#FFB300','#5E35B1','#00897B','#6D4C41','#C62828','#0D47A1'];
    function plantColor(name) {
        var h = 0;
        for (var i = 0; i < name.length; i++) h = ((h << 5) - h + name.charCodeAt(i)) | 0;
        return PLANT_PALETTE[Math.abs(h) % PLANT_PALETTE.length];
    }

    // ── Helpers ──
    function getYears() {
        if (!DATA || !DATA.scenarios || !DATA.scenarios.baseline) return [];
        return Object.keys(DATA.scenarios.baseline.envelope).map(Number).sort(function (a, b) { return a - b; });
    }

    /** Pad labels array to 2052 for x-axis breathing room (data ends at 2050). */
    function padLabelsTo2052(labels) {
        var last = Number(labels[labels.length - 1]) || 2050;
        for (var yr = last + 1; yr <= 2052; yr++) labels.push(String(yr));
        return labels;
    }

    /** Return an array of pointRadius values: `radius` at 5-year intervals, 0 elsewhere. */
    function fiveYearRadii(years, radius) {
        return years.map(function (y) { return y % 5 === 0 ? radius : 0; });
    }

    /** Tooltip filter: only show tooltips at 5-year intervals. */
    function fiveYearTooltipFilter(item) {
        var label = item.label || '';
        return Number(label) % 5 === 0;
    }

    /** Linearly interpolate trajectory data so every year has a value (no gaps). */
    function interpolateTrajectory(trajectory, years) {
        // Collect known (year, value) pairs sorted by year
        var known = [];
        Object.keys(trajectory).forEach(function (k) {
            if (trajectory[k] != null) known.push({ y: Number(k), v: trajectory[k] });
        });
        known.sort(function (a, b) { return a.y - b.y; });
        if (known.length === 0) return years.map(function () { return null; });

        return years.map(function (yr) {
            // Exact match
            for (var i = 0; i < known.length; i++) {
                if (known[i].y === yr) return known[i].v;
            }
            // Before first known point or after last
            if (yr <= known[0].y) return known[0].v;
            if (yr >= known[known.length - 1].y) return known[known.length - 1].v;
            // Find bracketing points and interpolate
            for (var j = 0; j < known.length - 1; j++) {
                if (yr > known[j].y && yr < known[j + 1].y) {
                    var t = (yr - known[j].y) / (known[j + 1].y - known[j].y);
                    return known[j].v + t * (known[j + 1].v - known[j].v);
                }
            }
            return null;
        });
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

    // Derive intensity_envelope from generation_by_fuel + emissions_by_fuel if not present
    function deriveIntensityEnvelope(sc) {
        if (sc.intensity_envelope) return;
        var gen = sc.generation_by_fuel;
        var emis = sc.emissions_by_fuel;
        if (!gen || !emis) return;
        var intEnv = {};
        Object.keys(gen).forEach(function (yr) {
            var totalGenTwh = 0;
            var totalEmisMt = 0;
            var gYear = gen[yr] || {};
            var eYear = emis[yr] || {};
            Object.keys(gYear).forEach(function (f) { if (f[0] !== '_') totalGenTwh += gYear[f] || 0; });
            Object.keys(eYear).forEach(function (f) { if (f[0] !== '_') totalEmisMt += eYear[f] || 0; });
            // TWh → MWh: ×1e6. Mt → kg: ×1e9. intensity = (Mt × 1e9) / (TWh × 1e6) = Mt/TWh × 1e3
            var intensityKg = totalGenTwh > 0 ? (totalEmisMt / totalGenTwh) * 1e3 : 0;
            intEnv[yr] = {
                p10: Math.round(intensityKg * 100) / 100,
                p50: Math.round(intensityKg * 100) / 100,
                p90: Math.round(intensityKg * 100) / 100,
                mean: Math.round(intensityKg * 100) / 100
            };
        });
        sc.intensity_envelope = intEnv;
    }

    function onDataLoaded(data) {
        DATA = data;
        // Derive intensity envelope for precomputed scenarios that lack it
        if (DATA.scenarios) {
            Object.keys(DATA.scenarios).forEach(function (k) {
                deriveIntensityEnvelope(DATA.scenarios[k]);
            });
        }
        buildControls();
        buildFanChart();
        buildWaterfall2023Chart();
        buildGenMixChart();
        buildCcgtTierChart();
        buildNewCleanChart();
        buildIntensityFanChart();

        buildTopEmittersChart();
        updateFanLegend();
        buildScenarioSelector();
        buildPlantDrilldownCharts();
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
            updateWaterfall2023Chart();
            updateGenMixChart();
            updateCcgtTierChart();
            updateNewCleanChart();
            updateTopEmittersChart();
            updateFanChartAnnotation();
        });
    }

    function updateAllCharts() {
        // Dismiss tooltip when chart data changes (e.g. scenario toggled)
        var tooltipEl = document.getElementById('fanTooltip');
        if (tooltipEl) tooltipEl.classList.remove('visible');

        // Wrap each chart update in try/catch so one failure doesn't cascade
        var updates = [
            ['fanChart', updateFanChart],
            ['fanLegend', updateFanLegend],
            ['scenarioSelector', updateScenarioSelector],
            ['waterfall2023Chart', updateWaterfall2023Chart],
            ['genMixChart', updateGenMixChart],
            ['ccgtTierChart', updateCcgtTierChart],
            ['newCleanChart', updateNewCleanChart],
            ['intensityFanChart', updateIntensityFanChart],

            ['topEmittersChart', updateTopEmittersChart]
        ];
        updates.forEach(function (pair) {
            try { pair[1](); }
            catch (err) { console.error('updateAllCharts — ' + pair[0] + ' failed:', err); }
        });
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
                        updateWaterfall2023Chart();
                        updateGenMixChart();
                        updateNewCleanChart();
                        updateTopEmittersChart();
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
        var visibleCount = savedScenarioOverlays.length;
        var hasBaselineOverlay = savedScenarioOverlays.some(function (s) { return s.isBaseline; });
        if (visibleCount <= 1 || (visibleCount === 2 && hasBaselineOverlay)) {
            return 'single';
        }
        return 'multi';
    }

    // ── Build 5-band fan datasets for a single scenario ──
    // Always uses scenario's own color for the p10-p90 bands (transparent shading).
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

        // Always use scenario-matched color shading (gray for baseline, blue for custom, etc.)
        var veryLight = hexToRgba(color, 0.08);
        var light     = hexToRgba(color, 0.15);
        var lineColor = hexToRgba(color, 0.35);

        // Use p50Style overrides if provided, otherwise defaults from color
        var p50Color = (p50Style && p50Style.color) ? p50Style.color : color;
        var p50Width = (p50Style && p50Style.width) ? p50Style.width : 2.5;
        var p50Dash  = (p50Style && p50Style.dash) ? p50Style.dash : [];
        var p50Radius = (p50Style && p50Style.pointRadius != null) ? p50Style.pointRadius : 3;
        var p50RadiusArr = fiveYearRadii(years, p50Radius);
        var p50HoverArr  = fiveYearRadii(years, 5);
        var p50Label = (p50Style && p50Style.legendLabel) ? p50Style.legendLabel : label;

        // 1. p10 (invisible bottom boundary)
        datasets.push({
            label: label + ' p10', data: p10,
            borderColor: 'transparent', backgroundColor: 'transparent',
            borderWidth: 0, pointRadius: 0, fill: false,
            tension: 0.3, _scenarioKey: label, _band: 'p10'
        });
        // 2. p10→p50 fill (light shading)
        datasets.push({
            label: label + ' p10-p50', data: p50,
            borderColor: 'transparent', backgroundColor: veryLight,
            borderWidth: 0, pointRadius: 0,
            fill: { target: startIdx, above: veryLight, below: veryLight },
            tension: 0.3, _scenarioKey: label, _band: 'fill'
        });
        // 3. p50 solid line
        datasets.push({
            label: p50Label,
            data: p50,
            borderColor: p50Color,
            backgroundColor: 'transparent',
            borderWidth: p50Width,
            borderDash: p50Dash,
            pointRadius: p50RadiusArr,
            pointBackgroundColor: p50Color,
            pointHoverRadius: p50HoverArr,
            fill: false, tension: 0.3,
            _scenarioKey: label, _band: 'p50',
            _showInLegend: true,
            _isBaseline: meta && meta.isBaseline,
            _scenarioName: (meta && meta.scenarioName) || label
        });
        // 4. p50→p90 fill (light shading)
        datasets.push({
            label: label + ' p50-p90', data: p90,
            borderColor: 'transparent', backgroundColor: veryLight,
            borderWidth: 0, pointRadius: 0,
            fill: { target: startIdx + 2, above: veryLight, below: veryLight },
            tension: 0.3, _scenarioKey: label, _band: 'fill'
        });
        // 5. p90 (invisible top boundary)
        datasets.push({
            label: label + ' p90', data: p90,
            borderColor: 'transparent', backgroundColor: 'transparent',
            borderWidth: 0, pointRadius: 0, fill: false,
            tension: 0.3, _scenarioKey: label, _band: 'p90'
        });
    }

    // ── Find the earliest intervention year from the fleet sidebar modifications ──
    function getFirstInterventionYear() {
        if (typeof FleetSidebar !== 'undefined' && FleetSidebar.getFleetPlants) {
            var plants = FleetSidebar.getFleetPlants();
            var earliest = Infinity;
            if (plants) {
                plants.forEach(function (p) {
                    if (p._action && p._year_online && p._year_online < earliest) {
                        earliest = p._year_online;
                    }
                });
            }
            return earliest === Infinity ? null : earliest;
        }
        return null;
    }

    // ── Merge custom envelope with baseline for pre-intervention years ──
    // Also injects the intervention year as an interpolated baseline data point
    // so the chart shows a clean transition at the correct year (not before).
    function alignEnvelopeWithBaseline(custEnv, baseEnv, firstYear) {
        if (!firstYear || !baseEnv) return custEnv;
        var merged = {};
        var keys = Object.keys(custEnv);
        keys.forEach(function (yStr) {
            var y = parseInt(yStr);
            if (y < firstYear) {
                // Use baseline values for years before any intervention
                merged[yStr] = baseEnv[yStr] || custEnv[yStr];
            } else {
                merged[yStr] = custEnv[yStr];
            }
        });

        // If firstYear doesn't exist as a data point, inject it with interpolated
        // baseline values so the chart shows a clean kink at the intervention year
        var firstYearStr = String(firstYear);
        if (!custEnv[firstYearStr]) {
            var sortedYears = Object.keys(baseEnv).map(Number).sort(function (a, b) { return a - b; });
            var lo = null, hi = null;
            for (var i = 0; i < sortedYears.length; i++) {
                if (sortedYears[i] <= firstYear) lo = sortedYears[i];
                if (sortedYears[i] >= firstYear && hi === null) hi = sortedYears[i];
            }
            if (lo !== null && hi !== null && lo !== hi) {
                var frac = (firstYear - lo) / (hi - lo);
                var loEnv = baseEnv[String(lo)];
                var hiEnv = baseEnv[String(hi)];
                if (loEnv && hiEnv) {
                    merged[firstYearStr] = {
                        p10: loEnv.p10 + (hiEnv.p10 - loEnv.p10) * frac,
                        p50: loEnv.p50 + (hiEnv.p50 - loEnv.p50) * frac,
                        p90: loEnv.p90 + (hiEnv.p90 - loEnv.p90) * frac,
                        mean: loEnv.mean + (hiEnv.mean - loEnv.mean) * frac
                    };
                }
            } else if (lo !== null && lo === firstYear && baseEnv[firstYearStr]) {
                merged[firstYearStr] = baseEnv[firstYearStr];
            }
        }

        return merged;
    }

    // Helper: linearly interpolate an envelope value between two surrounding years
    function interpolateEnvAtYear(env, year) {
        var keys = Object.keys(env).map(Number).sort(function (a, b) { return a - b; });
        var lo = null, hi = null;
        for (var i = 0; i < keys.length; i++) {
            if (keys[i] <= year) lo = keys[i];
            if (keys[i] >= year && hi === null) hi = keys[i];
        }
        if (lo === year && env[String(lo)]) return env[String(lo)];
        if (lo === null || hi === null || lo === hi) return null;
        var loE = env[String(lo)], hiE = env[String(hi)];
        if (!loE || !hiE) return null;
        var f = (year - lo) / (hi - lo);
        return {
            p10: loE.p10 + (hiE.p10 - loE.p10) * f,
            p50: loE.p50 + (hiE.p50 - loE.p50) * f,
            p90: loE.p90 + (hiE.p90 - loE.p90) * f,
            mean: loE.mean + (hiE.mean - loE.mean) * f
        };
    }

    function updateFanChart() {
        if (!fanChart || !DATA) return;
        var years = getYears();
        var datasets = [];
        var baseline = DATA.scenarios.baseline;
        var mode = getColorMode();

        // Check if we need to inject the intervention year into the timeline
        var firstYear = getFirstInterventionYear();
        var hasOverlays = savedScenarioOverlays.length > 0;
        if (firstYear && hasOverlays && years.indexOf(firstYear) === -1) {
            years = years.slice();
            years.push(firstYear);
            years.sort(function (a, b) { return a - b; });
        }
        var labels = years.map(String);

        // ── Always show baseline with gray p10-p90 band ──
        // If we injected an extra year, interpolate baseline envelope at that year
        if (baseline) {
            var baseEnv = getEnvelope(baseline);
            if (firstYear && !baseEnv[String(firstYear)]) {
                var interpBase = interpolateEnvAtYear(baseEnv, firstYear);
                if (interpBase) {
                    baseEnv = Object.assign({}, baseEnv);
                    baseEnv[String(firstYear)] = interpBase;
                }
            }
            addFanDatasets(datasets, baseEnv, years, BASELINE_COLOR, 'Baseline', mode, {
                color: BASELINE_COLOR, width: 2.5, dash: [], pointRadius: 3,
                legendLabel: 'Baseline (Baseline)'
            }, { isBaseline: true, scenarioName: 'Baseline' });
        }

        // ── Saved scenario overlays only (no unsaved custom scenario on charts) ──
        var sortedOverlays = savedScenarioOverlays.slice().sort(function (a, b) {
            if (a.isBaseline && !b.isBaseline) return -1;
            if (!a.isBaseline && b.isBaseline) return 1;
            var aTime = a.createdAt ? new Date(a.createdAt).getTime() : 0;
            var bTime = b.createdAt ? new Date(b.createdAt).getTime() : 0;
            return aTime - bTime;
        });

        if (mode === 'single' && sortedOverlays.length === 1) {
            var s = sortedOverlays[0];
            if (s.results && s.results.envelope) {
                var sEnv = s.results.envelope;
                var baseEnvForAlign = baseline ? getEnvelope(baseline) : null;
                var alignedEnv = alignEnvelopeWithBaseline(sEnv, baseEnvForAlign, firstYear);
                addFanDatasets(datasets, alignedEnv, years, s.color, s.name, mode, {
                    color: s.color, width: 2.5, dash: [], pointRadius: 3,
                    legendLabel: s.name
                }, { isBaseline: !!s.isBaseline, scenarioName: s.name });
            }
        } else {
            sortedOverlays.forEach(function (s) {
                if (s.results && s.results.envelope) {
                    addFanDatasets(datasets, s.results.envelope, years, s.color, s.name, mode, null,
                        { isBaseline: !!s.isBaseline, scenarioName: s.name || 'Saved' });
                }
            });
        }

        // Target overlays (same in both modes)
        selectedTargets.forEach(function (selTgt) {
            if (!selTgt || selTgt === 'none' || !DATA.targets || !DATA.targets[selTgt]) return;
            var tgt = DATA.targets[selTgt];
            var style = TARGET_STYLES[selTgt] || TARGET_STYLES.custom;
            var tgtData = interpolateTrajectory(tgt.trajectory, years);
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

        fanChart.data.labels = padLabelsTo2052(labels);
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

            // Only show tooltip at 5-year intervals
            if (Number(year) % 5 !== 0) {
                tooltipEl.classList.remove('visible');
                return;
            }
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

    // ── Custom tooltip for Intensity Fan Chart ──
    function intensityFanTooltipHandler(context) {
        var tooltipEl = document.getElementById('intensityFanTooltip');
        if (!tooltipEl) return;
        var tooltipModel = context.tooltip;

        if (tooltipModel.opacity === 0) {
            tooltipEl.classList.remove('visible');
            return;
        }

        if (tooltipModel.dataPoints && tooltipModel.dataPoints.length) {
            var idx = tooltipModel.dataPoints[0].dataIndex;
            var year = intensityFanChart.data.labels[idx];

            // Only show tooltip at 5-year intervals
            if (Number(year) % 5 !== 0) {
                tooltipEl.classList.remove('visible');
                return;
            }

            var html = '<div class="fan-tooltip-title">' + year + '</div>';

            var baseline = DATA.scenarios.baseline;
            var baseIntEnv = baseline && baseline.intensity_envelope ? baseline.intensity_envelope : null;
            var baseYear = baseIntEnv ? baseIntEnv[year] : null;
            var baseline2023 = baseIntEnv && baseIntEnv['2023'] ? baseIntEnv['2023'].p50 : null;

            // Baseline row
            if (baseYear) {
                html += intensityTooltipRow(baseYear.p50, BASELINE_COLOR, 'Baseline', baseline2023);
            }

            // Custom scenario row
            if (customScenario && customScenario.intensity_envelope && customScenario.intensity_envelope[year]) {
                var custVal = customScenario.intensity_envelope[year].p50;
                html += intensityTooltipRow(custVal, CUSTOM_COLOR, 'Custom', baseline2023);
                if (baseYear) {
                    var delta = custVal - baseYear.p50;
                    var cls = delta < 0 ? 'gap-negative' : 'gap-positive';
                    var sign = delta > 0 ? '+' : '';
                    html += '<div class="fan-tooltip-gap"><span class="' + cls + '">' +
                        sign + delta.toFixed(1) + ' kgCO₂/MWh vs Baseline</span></div>';
                }
            }

            // Saved scenario rows
            savedScenarioOverlays.forEach(function (s) {
                if (!s.results || !s.results.intensity_envelope || !s.results.intensity_envelope[year]) return;
                var sVal = s.results.intensity_envelope[year].p50;
                html += intensityTooltipRow(sVal, s.color, s.name || 'Saved', baseline2023);
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

    function intensityTooltipRow(value, color, label, baseline2023) {
        var pctStr = '';
        if (baseline2023 && baseline2023 > 0) {
            var pct = ((baseline2023 - value) / baseline2023 * 100).toFixed(1);
            pctStr = '<span style="opacity:0.7;margin-left:6px;">(' +
                (pct >= 0 ? '−' : '+') + Math.abs(parseFloat(pct)).toFixed(1) + '% vs 2023)</span>';
        }
        return '<div class="fan-tooltip-row">' +
            '<span class="fan-tooltip-swatch" style="background:' + color + '"></span>' +
            '<span class="fan-tooltip-label">' + label + '</span>' +
            '<span class="fan-tooltip-vals">' + value.toFixed(1) + ' kgCO₂/MWh' + pctStr + '</span></div>';
    }

    // ── Fan legend (below chart — shows band key in single mode) ──
    function updateFanLegend() {
        var el = document.getElementById('fanLegend');
        if (!el || !DATA) return;
        var mode = getColorMode();
        var items = [];

        // Show scenario bands explanation
        items.push({ label: 'Baseline (P50)', color: BASELINE_COLOR, type: 'line' });
        items.push({ label: 'Baseline P10–P90', color: hexToRgba(BASELINE_COLOR, 0.15), type: 'band' });

        // Saved scenario legend items
        savedScenarioOverlays.forEach(function (s) {
            if (s.results && s.results.envelope) {
                items.push({ label: s.name + ' (P50)', color: s.color, type: 'line' });
                items.push({ label: s.name + ' P10–P90', color: hexToRgba(s.color, 0.15), type: 'band' });
            }
        });

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
            } else if (item.type === 'line') {
                swatch = '<span style="width:24px;height:0;border-top:2.5px solid ' + item.color + ';display:inline-block;"></span>';
            } else {
                swatch = '<span style="width:24px;height:10px;border-radius:3px;background:' + item.color + ';display:inline-block;"></span>';
            }
            return '<span style="display:inline-flex;align-items:center;gap:4px;margin-right:16px;">' +
                swatch + item.label + '</span>';
        }).join('');

        // Shading explanation
        html += '<div style="margin-top:6px;font-size:0.78rem;color:var(--text-dim);font-style:italic;">' +
            'Shaded bands show P10–P90 confidence intervals across fossil fuel price scenarios</div>';

        el.innerHTML = html;
    }

    // (Waterfall vs market trajectory chart — removed per Task 4)

    // ====================================================================
    // CHART 2b: PLANT-LEVEL WATERFALL vs 2023 BASELINE
    // ====================================================================
    function buildWaterfall2023Chart() {
        var ctx = document.getElementById('waterfall2023Chart').getContext('2d');
        waterfall2023Chart = new Chart(ctx, {
            type: 'bar',
            data: { labels: [], datasets: [] },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                animation: { duration: 300 },
                scales: {
                    x: {
                        title: { display: true, text: 'Emissions Delta vs 2023 (Mt CO₂)', font: { size: 12 } },
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
                                return (v > 0 ? '+' : '') + v.toFixed(2) + ' Mt CO₂ vs 2023';
                            }
                        }
                    }
                }
            }
        });
        updateWaterfall2023Chart();
    }

    function updateWaterfall2023Chart() {
        if (!waterfall2023Chart || !DATA) return;

        var yr = String(nearestYear(selectedYear, getYears()));
        document.getElementById('waterfall2023Title').textContent =
            'Emissions Delta vs 2023 Baseline — ' + yr;

        var scenarioToCompare = getSelectedLowerScenario();
        var scPlants = scenarioToCompare.plant_detail ? scenarioToCompare.plant_detail[yr] : null;

        // 2023 actual emissions baseline — fixed reference point
        var base2023Plants = DATA.scenarios.baseline.plant_detail ?
            DATA.scenarios.baseline.plant_detail['2023'] : null;

        if (!base2023Plants || !scPlants) {
            waterfall2023Chart.data.labels = ['No 2023 data'];
            waterfall2023Chart.data.datasets = [{ data: [0], backgroundColor: '#ddd' }];
            waterfall2023Chart.update('active');
            return;
        }

        // Build delta: baseline[2023] emissions - scenario[yr] emissions per plant
        var baseMap = {};
        base2023Plants.forEach(function (p) { baseMap[p.orispl] = p; });

        var deltas = [];
        scPlants.forEach(function (p) {
            var baseP = baseMap[p.orispl];
            var baseE = baseP ? baseP.emissions_mt : 0;
            var delta = baseE - p.emissions_mt;  // positive = reduction from 2023
            if (Math.abs(delta) > 0.001) {
                deltas.push({ name: p.name, delta: delta });
            }
        });
        // Also check for plants in 2023 that are missing from scenario (retired)
        base2023Plants.forEach(function (p) {
            var found = scPlants.some(function (sp) { return sp.orispl === p.orispl; });
            if (!found && p.emissions_mt > 0.001) {
                deltas.push({ name: p.name + ' (retired)', delta: p.emissions_mt });
            }
        });

        deltas.sort(function (a, b) { return Math.abs(b.delta) - Math.abs(a.delta); });

        // Truncate to top 15 + Other
        var other = 0;
        if (deltas.length > 15) {
            for (var i = 15; i < deltas.length; i++) other += deltas[i].delta;
            deltas = deltas.slice(0, 15);
            if (Math.abs(other) > 0.001) deltas.push({ name: 'Other', delta: other });
        }

        var labels = deltas.map(function (d) { return d.name; });
        var values = deltas.map(function (d) { return d.delta; });
        var colors = values.map(function (v) { return v >= 0 ? '#6BA543' : '#DC2626'; });

        waterfall2023Chart.data.labels = labels;
        waterfall2023Chart.data.datasets = [{
            data: values,
            backgroundColor: colors.map(function (c) { return hexToRgba(c, 0.45); }),
            borderColor: colors,
            borderWidth: 1.5,
            borderRadius: 4,
            barPercentage: 0.7
        }];
        waterfall2023Chart.update('active');
    }

    // ====================================================================
    // SCENARIO SELECTOR — Toggle between saved scenarios for lower charts
    // ====================================================================
    function getSelectedLowerScenario() {
        if (!selectedLowerScenarioId) return DATA ? DATA.scenarios.baseline : null;
        var found = savedScenarioOverlays.find(function (s) { return s.id === selectedLowerScenarioId; });
        if (found && found.results) return found.results;
        return DATA ? DATA.scenarios.baseline : null;
    }

    function getSelectedLowerScenarioLabel() {
        if (!selectedLowerScenarioId) return 'Baseline';
        var found = savedScenarioOverlays.find(function (s) { return s.id === selectedLowerScenarioId; });
        return found ? found.name : 'Baseline';
    }

    function buildScenarioSelector() {
        updateScenarioSelector();
        var bar = document.getElementById('scenarioToggleBar');
        if (bar) {
            bar.addEventListener('click', function (e) {
                var btn = e.target.closest('.toggle-btn');
                if (!btn) return;
                bar.querySelectorAll('.toggle-btn').forEach(function (b) { b.classList.remove('active'); });
                btn.classList.add('active');
                selectedLowerScenarioId = btn.dataset.scenarioId || null;
                updateGenMixChart();
                updateWaterfall2023Chart();
                updateNewCleanChart();

                updateTopEmittersChart();
            });
        }
    }

    function updateScenarioSelector() {
        var bar = document.getElementById('scenarioToggleBar');
        if (!bar) return;
        var html = '<button class="toggle-btn' + (!selectedLowerScenarioId ? ' active' : '') + '" data-scenario-id="">Baseline</button>';
        savedScenarioOverlays.forEach(function (s) {
            var isActive = selectedLowerScenarioId === s.id;
            html += '<button class="toggle-btn' + (isActive ? ' active' : '') + '" data-scenario-id="' + s.id + '">' +
                '<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:' + s.color + ';margin-right:6px;vertical-align:middle;"></span>' +
                (s.name || 'Scenario') + '</button>';
        });
        bar.innerHTML = html;

        // If selected scenario was deleted, fall back to baseline
        if (selectedLowerScenarioId && !savedScenarioOverlays.find(function (s) { return s.id === selectedLowerScenarioId; })) {
            selectedLowerScenarioId = null;
        }

        // Show/hide the selector card
        var card = document.getElementById('scenarioSelectorCard');
        if (card) card.style.display = savedScenarioOverlays.length > 0 ? '' : 'none';
    }

    // ====================================================================
    // CHART 3: GENERATION MIX (Stacked Area Timeseries)
    // ====================================================================
    // Custom plugin to draw 5-year labels (total TWh + % low carbon) above bars
    var genMixLabelPlugin = {
        id: 'genMixLabels',
        afterDraw: function (chart) {
            var meta0 = chart.getDatasetMeta(0);
            if (!meta0 || !meta0.data || !meta0.data.length) return;
            var ctx = chart.ctx;
            ctx.save();
            ctx.textAlign = 'center';
            ctx.font = '600 10px sans-serif';

            var labelYears = [2025, 2030, 2035, 2040, 2045, 2050];
            chart.data.labels.forEach(function (label, idx) {
                var yr = Number(label);
                if (labelYears.indexOf(yr) === -1) return;

                // Sum total and clean for this bar index
                var total = 0, clean = 0;
                chart.data.datasets.forEach(function (ds) {
                    var v = ds.data[idx] || 0;
                    total += v;
                    if (ds._fuelKey && CLEAN_FUEL_SET[ds._fuelKey]) clean += v;
                });
                if (total <= 0) return;

                var pctClean = (clean / total * 100).toFixed(0);

                // Find the top of the stacked bar
                var topY = Infinity;
                for (var di = 0; di < chart.data.datasets.length; di++) {
                    var barMeta = chart.getDatasetMeta(di);
                    if (barMeta.hidden || !barMeta.data[idx]) continue;
                    var barEl = barMeta.data[idx];
                    if (barEl.y < topY) topY = barEl.y;
                }
                if (!isFinite(topY)) return;

                var x = meta0.data[idx].x;
                ctx.fillStyle = '#374151';
                ctx.fillText(total.toFixed(0) + ' TWh', x, topY - 16);
                ctx.fillStyle = '#059669';
                ctx.fillText(pctClean + '% clean', x, topY - 4);
            });
            ctx.restore();
        }
    };

    function buildGenMixChart() {
        var ctx = document.getElementById('genMixChart').getContext('2d');
        genMixChart = new Chart(ctx, {
            type: 'bar',
            data: { labels: [], datasets: [] },
            plugins: [genMixLabelPlugin],
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 300 },
                interaction: { mode: 'index', intersect: false },
                layout: { padding: { top: 30 } },
                scales: {
                    x: {
                        stacked: true,
                        grid: { display: false },
                        ticks: { font: { size: 11 }, maxRotation: 0, callback: function (val) { var y = Number(this.getLabelForValue(val)); return y > 2050 ? '' : y; } }
                    },
                    y: {
                        stacked: true,
                        title: { display: true, text: 'Generation (TWh)', font: { size: 12 } },
                        grid: { color: '#E0E6EF' },
                        beginAtZero: true,
                        min: 0,
                        ticks: { font: { size: 11 } }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        filter: fiveYearTooltipFilter,
                        callbacks: {
                            title: function (tooltipItems) {
                                if (!tooltipItems.length) return '';
                                var idx = tooltipItems[0].dataIndex;
                                var total = 0, clean = 0;
                                tooltipItems[0].chart.data.datasets.forEach(function (ds) {
                                    var v = ds.data[idx] || 0;
                                    total += v;
                                    if (ds._fuelKey && CLEAN_FUEL_SET[ds._fuelKey]) clean += v;
                                });
                                var pct = total > 0 ? (clean / total * 100).toFixed(1) : '0.0';
                                return tooltipItems[0].label + ' — ' + total.toFixed(1) + ' TWh | ' + pct + '% Low Carbon';
                            },
                            label: function (ctx) {
                                if (!ctx.raw) return null;
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

        var sc = getSelectedLowerScenario();
        var scenarioLabel = getSelectedLowerScenarioLabel();
        document.getElementById('genMixTitle').textContent = 'Fleet Generation by Fuel — ' + scenarioLabel;

        var years = getYears();
        var labels = years.map(String);
        var gen = sc.generation_by_fuel;

        // Build stacked bar datasets with semi-transparent fill + saturated outline
        var datasets = [];
        FUEL_KEYS.forEach(function (fuel) {
            var data = years.map(function (y) {
                if (!gen || !gen[String(y)]) return 0;
                var val = gen[String(y)][fuel] || 0;
                return Math.max(val, 0);
            });
            var hasData = data.some(function (v) { return v > 0; });
            if (!hasData) return; // Skip empty fuels
            datasets.push({
                label: FUEL_LABELS[fuel] || fuel,
                _fuelKey: fuel,
                data: data,
                backgroundColor: hexToRgba(FUEL_COLORS[fuel], 0.55),
                borderColor: FUEL_COLORS[fuel],
                borderWidth: 1.5,
                borderSkipped: true
            });
        });

        // Only the topmost dataset gets rounded corners
        if (datasets.length > 0) {
            var last = datasets[datasets.length - 1];
            last.borderRadius = { topLeft: 4, topRight: 4, bottomLeft: 0, bottomRight: 0 };
            last.borderSkipped = 'start';
        }

        var paddedLabels = padLabelsTo2052(labels);
        // Pad data arrays to match padded labels (fill trailing years with 0)
        var padCount = paddedLabels.length - years.length;
        if (padCount > 0) {
            datasets.forEach(function (ds) {
                for (var p = 0; p < padCount; p++) ds.data.push(0);
            });
        }
        genMixChart.data.labels = paddedLabels;
        genMixChart.data.datasets = datasets;
        genMixChart.update('active');

        // Build legend showing only fuels with data
        var legendEl = document.getElementById('genMixLegend');
        legendEl.innerHTML = datasets.map(function (ds) {
            return '<span style="display:inline-flex;align-items:center;gap:4px;margin-right:16px;">' +
                '<span style="width:24px;height:10px;border-radius:3px;background:' + ds.borderColor + ';display:inline-block;"></span>' +
                ds.label + '</span>';
        }).join('');
    }

    // ====================================================================
    // CHART 3B: CCGT TIER CAPACITY FACTORS (Grouped Bar) — Selected Year
    // ====================================================================
    var TIER_KEYS = ['very_low', 'low', 'medium', 'high', 'very_high'];
    var TIER_LABELS = {
        very_low: 'Very Low HR (6.2)',
        low: 'Low HR (6.8)',
        medium: 'Medium HR (7.5)',
        high: 'High HR (8.1)',
        very_high: 'Very High HR (9.0)'
    };
    var TIER_COLORS = {
        very_low: '#22C55E',
        low: '#0EA5E9',
        medium: '#6366F1',
        high: '#F59E0B',
        very_high: '#EF4444'
    };

    function buildCcgtTierChart() {
        var ctx = document.getElementById('ccgtTierChart');
        if (!ctx) return;
        ccgtTierChart = new Chart(ctx.getContext('2d'), {
            type: 'bar',
            data: { labels: [], datasets: [] },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 300 },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { font: { size: 11 } }
                    },
                    y: {
                        title: { display: true, text: 'Capacity Factor', font: { size: 12 } },
                        grid: { color: '#E0E6EF' },
                        beginAtZero: true,
                        max: 1.0,
                        ticks: {
                            font: { size: 11 },
                            callback: function (v) { return Math.round(v * 100) + '%'; }
                        }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function (ctx) {
                                return ctx.dataset.label + ': ' + (ctx.raw * 100).toFixed(1) + '%';
                            }
                        }
                    }
                }
            }
        });
        updateCcgtTierChart();
    }

    function updateCcgtTierChart() {
        if (!ccgtTierChart || !DATA) return;

        var yearStr = String(selectedYear);
        var yearEl = document.getElementById('ccgtTierYear');
        if (yearEl) yearEl.textContent = yearStr;

        // Aggregate CCGT plants by tier from plant_detail for the selected scenario/year
        var sc = getSelectedLowerScenario();
        var plants = (sc.plant_detail && sc.plant_detail[yearStr]) || [];

        // Group CCGT plants by tier — compute implied CF from gen/capacity
        var tierData = {};
        TIER_KEYS.forEach(function (t) {
            tierData[t] = { gen_twh: 0, cap_mw: 0, count: 0, emis_mt: 0 };
        });

        plants.forEach(function (p) {
            if (p.fuel_type !== 'gas_ccgt') return;
            if (p.status === 'retired') return;
            var tier = p.heat_rate_tier || 'medium';
            if (!tierData[tier]) tier = 'medium';
            tierData[tier].gen_twh += p.gen_twh || 0;
            tierData[tier].cap_mw += p.capacity_mw || 0;
            tierData[tier].count += 1;
            tierData[tier].emis_mt += p.emissions_mt || 0;
        });

        // Compute CF = gen_twh * 1e6 / (cap_mw * 8760) for each tier
        var labels = [];
        var cfValues = [];
        var colors = [];
        var borderColors = [];

        TIER_KEYS.forEach(function (tier) {
            var d = tierData[tier];
            if (d.cap_mw <= 0) return; // Skip tiers with no capacity
            var cf = (d.gen_twh * 1e6) / (d.cap_mw * 8760);
            labels.push(TIER_LABELS[tier] + '\n' + d.count + ' plants, ' + Math.round(d.cap_mw).toLocaleString() + ' MW');
            cfValues.push(Math.min(cf, 1.0));
            colors.push(hexToRgba(TIER_COLORS[tier], 0.6));
            borderColors.push(TIER_COLORS[tier]);
        });

        ccgtTierChart.data.labels = labels;
        ccgtTierChart.data.datasets = [{
            label: 'Capacity Factor',
            data: cfValues,
            backgroundColor: colors,
            borderColor: borderColors,
            borderWidth: 2,
            borderRadius: 6
        }];
        ccgtTierChart.update('active');

        // Build legend
        var legendEl = document.getElementById('ccgtTierLegend');
        if (legendEl) {
            legendEl.innerHTML = TIER_KEYS.map(function (tier) {
                var d = tierData[tier];
                if (d.cap_mw <= 0) return '';
                var cf = (d.gen_twh * 1e6) / (d.cap_mw * 8760);
                return '<span style="display:inline-flex;align-items:center;gap:4px;margin-right:16px;">' +
                    '<span style="width:24px;height:10px;border-radius:3px;background:' + TIER_COLORS[tier] + ';display:inline-block;"></span>' +
                    TIER_LABELS[tier] + ': ' + (cf * 100).toFixed(1) + '% CF, ' +
                    d.emis_mt.toFixed(1) + ' Mt CO₂' +
                    '</span>';
            }).filter(Boolean).join('');
        }
    }

    // ====================================================================
    // CHART 4: NEW CLEAN GENERATION SNAPSHOT (Bar) — Selected Year
    // ====================================================================
    var CLEAN_FUEL_KEYS = ['nuclear', 'geothermal', 'wind', 'solar', 'hydro', 'battery', 'ldes', 'ccs_ccgt'];
    var CLEAN_FUEL_LABELS = {
        nuclear: 'Nuclear', geothermal: 'Geothermal', wind: 'Wind', solar: 'Solar',
        hydro: 'Hydro', battery: 'Battery', ldes: 'LDES', ccs_ccgt: 'CCS-CCGT'
    };

    function buildNewCleanChart() {
        var ctx = document.getElementById('newCleanChart');
        if (!ctx) return;
        newCleanChart = new Chart(ctx.getContext('2d'), {
            type: 'bar',
            data: { labels: [], datasets: [] },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                animation: { duration: 300 },
                scales: {
                    x: {
                        title: { display: true, text: 'Generation (TWh)', font: { size: 12 } },
                        grid: { color: '#E0E6EF' },
                        beginAtZero: true,
                        ticks: { font: { size: 11 } }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { font: { size: 12, weight: '500' } }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function (ctx) {
                                return ctx.raw.toFixed(2) + ' TWh';
                            }
                        }
                    }
                }
            }
        });
        updateNewCleanChart();
    }

    function updateNewCleanChart() {
        if (!newCleanChart || !DATA) return;

        var yr = String(nearestYear(selectedYear, getYears()));
        document.getElementById('newCleanTitle').textContent = 'New Clean Generation — ' + yr;

        var baseGen = {};
        var custGen = {};
        var baseline = DATA.scenarios.baseline;
        var selectedSc = getSelectedLowerScenario();
        if (baseline && baseline.generation_by_fuel && baseline.generation_by_fuel[yr]) {
            baseGen = baseline.generation_by_fuel[yr];
        }
        if (selectedSc && selectedSc.generation_by_fuel && selectedSc.generation_by_fuel[yr]) {
            custGen = selectedSc.generation_by_fuel[yr];
        }

        // New clean gen: use explicit _new_clean_gen from recalculate if available,
        // otherwise fall back to delta approach
        var labels = [];
        var values = [];
        var colors = [];
        var newClean = custGen._new_clean_gen;
        if (newClean) {
            // Explicit tracking: crane (full gen), uprate (increment), CCS (full), new builds (full)
            CLEAN_FUEL_KEYS.forEach(function (fuel) {
                var val = newClean[fuel] || 0;
                if (val > 0.001) {
                    labels.push(CLEAN_FUEL_LABELS[fuel] || fuel);
                    values.push(Math.round(val * 100) / 100);
                    colors.push(FUEL_COLORS[fuel] || '#9CA3AF');
                }
            });
        } else {
            // Fallback: delta above baseline for clean fuels, plus all CCS
            CLEAN_FUEL_KEYS.forEach(function (fuel) {
                var custVal = custGen[fuel] || 0;
                var baseVal = baseGen[fuel] || 0;
                var delta = (fuel === 'ccs_ccgt') ? custVal : (custVal - baseVal);
                if (delta > 0.001) {
                    labels.push(CLEAN_FUEL_LABELS[fuel] || fuel);
                    values.push(Math.round(delta * 100) / 100);
                    colors.push(FUEL_COLORS[fuel] || '#9CA3AF');
                }
            });
        }

        // If viewing baseline (no scenario selected), show empty state
        if (!selectedLowerScenarioId) {
            labels = [];
            values = [];
            colors = [];
        }

        newCleanChart.data.labels = labels;
        newCleanChart.data.datasets = [{
            data: values,
            backgroundColor: colors.map(function (c) { return hexToRgba(c, 0.45); }),
            borderColor: colors,
            borderWidth: 1.5,
            borderRadius: 4,
            barPercentage: 0.7
        }];
        newCleanChart.update('active');

        // Legend
        var legendEl = document.getElementById('newCleanLegend');
        if (legendEl) {
            if (labels.length === 0) {
                legendEl.innerHTML = '<span style="color:#9CA3AF;font-size:0.85rem;">Select a saved scenario to see new clean generation vs baseline.</span>';
            } else {
                legendEl.innerHTML = labels.map(function (lbl, i) {
                    return '<span style="display:inline-flex;align-items:center;gap:4px;margin-right:16px;">' +
                        '<span style="width:24px;height:10px;border-radius:3px;background:' + colors[i] + ';display:inline-block;"></span>' +
                        lbl + ': ' + values[i].toFixed(1) + ' TWh</span>';
                }).join('');
            }
        }
    }

    // ====================================================================
    // CHART 5: EMISSIONS INTENSITY FAN (kgCO2/MWh) — Timeseries
    // ====================================================================
    function buildIntensityFanChart() {
        var ctx = document.getElementById('intensityFanChart');
        if (!ctx) return;
        intensityFanChart = new Chart(ctx.getContext('2d'), {
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
                            callback: function (val) {
                                var year = this.getLabelForValue(val);
                                var show = [2023, 2025, 2030, 2035, 2040, 2045, 2050];
                                return show.indexOf(Number(year)) >= 0 ? year : '';
                            },
                            maxRotation: 0, font: { size: 12 }
                        }
                    },
                    y: {
                        title: { display: true, text: 'Emissions Intensity (kgCO₂/MWh)', font: { size: 13 } },
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
                            usePointStyle: true,
                            pointStyle: 'line',
                            padding: 16,
                            font: { size: 12 }
                        }
                    },
                    tooltip: {
                        enabled: false,
                        external: intensityFanTooltipHandler
                    }
                }
            }
        });
        updateIntensityFanChart();
    }

    function updateIntensityFanChart() {
        if (!intensityFanChart || !DATA) return;

        var years = getYears();
        var datasets = [];
        var baseline = DATA.scenarios.baseline;

        // Inject intervention year for chart resolution (same as fan chart)
        var firstYear = getFirstInterventionYear();
        var hasOverlays = savedScenarioOverlays.length > 0;
        if (firstYear && hasOverlays && years.indexOf(firstYear) === -1) {
            years = years.slice();
            years.push(firstYear);
            years.sort(function (a, b) { return a - b; });
        }
        var labels = years.map(String);

        // Baseline intensity fan (interpolate at intervention year if needed)
        if (baseline && baseline.intensity_envelope) {
            var baseIntEnv = baseline.intensity_envelope;
            if (firstYear && !baseIntEnv[String(firstYear)]) {
                var interpBase = interpolateEnvAtYear(baseIntEnv, firstYear);
                if (interpBase) {
                    baseIntEnv = Object.assign({}, baseIntEnv);
                    baseIntEnv[String(firstYear)] = interpBase;
                }
            }
            addIntensityFanDatasets(datasets, baseIntEnv, years, BASELINE_COLOR, 'Baseline', true);
        }

        // Saved scenario intensity overlays
        savedScenarioOverlays.forEach(function (s) {
            if (s.results && s.results.intensity_envelope) {
                var alignedIntEnv = alignEnvelopeWithBaseline(
                    s.results.intensity_envelope,
                    baseline ? baseline.intensity_envelope : null,
                    firstYear
                );
                addIntensityFanDatasets(datasets, alignedIntEnv, years, s.color, s.name, false);
            }
        });

        intensityFanChart.data.labels = padLabelsTo2052(labels);
        intensityFanChart.data.datasets = datasets;
        intensityFanChart.update('active');
    }

    function addIntensityFanDatasets(datasets, env, years, color, label, isBaseline) {
        if (!env) return;
        var p10 = [], p50 = [], p90 = [];
        years.forEach(function (y) {
            var e = env[String(y)];
            if (e) { p10.push(e.p10); p50.push(e.p50); p90.push(e.p90); }
            else { p10.push(null); p50.push(null); p90.push(null); }
        });

        var startIdx = datasets.length;
        var bandColor = hexToRgba(color, 0.12);

        // p10 boundary
        datasets.push({
            label: label + ' p10', data: p10,
            borderColor: 'transparent', backgroundColor: 'transparent',
            borderWidth: 0, pointRadius: 0, fill: false, tension: 0.3
        });
        // p10→p50 fill
        datasets.push({
            label: label + ' band-low', data: p50,
            borderColor: 'transparent', backgroundColor: bandColor,
            borderWidth: 0, pointRadius: 0,
            fill: { target: startIdx, above: bandColor, below: bandColor },
            tension: 0.3
        });
        // p50 line
        var r = isBaseline ? 3 : 3;
        datasets.push({
            label: label,
            data: p50,
            borderColor: color,
            backgroundColor: 'transparent',
            borderWidth: 2.5,
            borderDash: isBaseline ? [] : [],
            pointRadius: fiveYearRadii(years, r),
            pointBackgroundColor: color,
            pointHoverRadius: fiveYearRadii(years, 5),
            fill: false, tension: 0.3,
            _showInLegend: true
        });
        // p50→p90 fill
        datasets.push({
            label: label + ' band-high', data: p90,
            borderColor: 'transparent', backgroundColor: bandColor,
            borderWidth: 0, pointRadius: 0,
            fill: { target: startIdx + 2, above: bandColor, below: bandColor },
            tension: 0.3
        });
    }


    // ====================================================================
    // CHART 7: TOP EMITTING PLANTS — Stacked area time series
    // ====================================================================
    var FOSSIL_FUEL_SET = { gas_ccgt: 1, gas_ct: 1, gas_oil_ct: 1, oil_ct: 1 };
    var TOP_EMITTER_COUNT = 12;

    function buildTopEmittersChart() {
        var ctx = document.getElementById('topEmittersChart');
        if (!ctx) return;
        topEmittersChart = new Chart(ctx.getContext('2d'), {
            type: 'line',
            data: { labels: [], datasets: [] },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 300 },
                interaction: { mode: 'index', intersect: false },
                scales: {
                    x: { grid: { display: false }, ticks: { font: { size: 12 }, maxRotation: 0, callback: function (val) { var y = Number(this.getLabelForValue(val)); return y > 2050 ? '' : y; } } },
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
                        filter: fiveYearTooltipFilter,
                        callbacks: {
                            title: function (tooltipItems) {
                                if (!tooltipItems.length) return '';
                                var idx = tooltipItems[0].dataIndex;
                                var total = 0;
                                tooltipItems[0].chart.data.datasets.forEach(function (ds) {
                                    total += ds.data[idx] || 0;
                                });
                                return tooltipItems[0].label + ' — Total: ' + total.toFixed(2) + ' Mt CO₂';
                            },
                            label: function (ctx) {
                                if (!ctx.raw || ctx.raw < 0.001) return null;
                                var total = 0;
                                ctx.chart.data.datasets.forEach(function (ds) {
                                    total += ds.data[ctx.dataIndex] || 0;
                                });
                                var pct = total > 0 ? ((ctx.raw / total) * 100).toFixed(1) : '0.0';
                                return ctx.dataset.label + ': ' + ctx.raw.toFixed(3) + ' Mt (' + pct + '%)';
                            }
                        }
                    }
                }
            }
        });
        updateTopEmittersChart();
    }

    function updateTopEmittersChart() {
        if (!topEmittersChart || !DATA) return;
        var sc = getSelectedLowerScenario();
        var scenarioLabel = getSelectedLowerScenarioLabel();
        var titleEl = document.getElementById('topEmittersTitle');
        if (titleEl) titleEl.textContent = 'Top Emitting Plants — ' + scenarioLabel;

        var years = getYears();
        var labels = years.map(String);

        if (!sc.plant_detail) {
            topEmittersChart.data.labels = labels;
            topEmittersChart.data.datasets = [];
            topEmittersChart.update('active');
            return;
        }

        // Identify top emitters: rank by total emissions across all years
        var plantTotalEmis = {};
        var plantNames = {};
        years.forEach(function (yr) {
            var plants = sc.plant_detail[String(yr)];
            if (!plants) return;
            plants.forEach(function (p) {
                if (!FOSSIL_FUEL_SET[p.fuel_type] || !p.emissions_mt || p.emissions_mt <= 0) return;
                var key = p.orispl || p.name;
                plantTotalEmis[key] = (plantTotalEmis[key] || 0) + p.emissions_mt;
                if (!plantNames[key]) plantNames[key] = p.name;
            });
        });

        // Sort and take top N
        var ranked = Object.keys(plantTotalEmis).sort(function (a, b) {
            return plantTotalEmis[b] - plantTotalEmis[a];
        });
        var topKeys = ranked.slice(0, TOP_EMITTER_COUNT);
        var topSet = {};
        topKeys.forEach(function (k) { topSet[k] = true; });

        // Build per-plant time series + "Other" bucket
        var seriesData = {}; // key → [val per year]
        topKeys.forEach(function (k) { seriesData[k] = []; });
        seriesData['_other'] = [];

        years.forEach(function (yr) {
            var plants = sc.plant_detail[String(yr)] || [];
            var byKey = {};
            plants.forEach(function (p) {
                if (!FOSSIL_FUEL_SET[p.fuel_type] || !p.emissions_mt || p.emissions_mt <= 0) return;
                var key = p.orispl || p.name;
                byKey[key] = (byKey[key] || 0) + p.emissions_mt;
            });

            var otherVal = 0;
            Object.keys(byKey).forEach(function (k) {
                if (!topSet[k]) otherVal += byKey[k];
            });

            topKeys.forEach(function (k) {
                seriesData[k].push(byKey[k] || 0);
            });
            seriesData['_other'].push(otherVal);
        });

        // Build datasets — top plants first (bottom of stack), "Other" on top
        var datasets = [];
        topKeys.forEach(function (k) {
            datasets.push({
                label: plantNames[k] || k,
                data: seriesData[k],
                borderColor: plantColor(plantNames[k] || k),
                backgroundColor: hexToRgba(plantColor(plantNames[k] || k), 0.45),
                borderWidth: 1.5,
                pointRadius: 0,
                pointHoverRadius: 0,
                fill: 'origin',
                tension: 0.3
            });
        });
        if (seriesData['_other'].some(function (v) { return v > 0.001; })) {
            datasets.push({
                label: 'Other Plants',
                data: seriesData['_other'],
                borderColor: '#9CA3AF',
                backgroundColor: hexToRgba('#9CA3AF', 0.35),
                borderWidth: 1,
                pointRadius: 0,
                fill: 'origin',
                tension: 0.3
            });
        }

        topEmittersChart.data.labels = padLabelsTo2052(labels);
        topEmittersChart.data.datasets = datasets;
        topEmittersChart.update('active');

        // Legend
        var legendEl = document.getElementById('topEmittersLegend');
        if (legendEl) {
            var items = datasets.map(function (ds) {
                return '<span style="display:inline-flex;align-items:center;gap:4px;margin-right:12px;font-size:0.8rem;">' +
                    '<span style="width:20px;height:8px;border-radius:3px;background:' + ds.borderColor + ';display:inline-block;"></span>' +
                    ds.label + '</span>';
            });
            legendEl.innerHTML = items.join('');
        }
    }

    // ── Public API for sidebar integration ──
    window.FLEET_SCENARIOS_API = {
        setCustomScenario: function (label, data) {
            // No longer auto-displays on charts — user must save scenario first
            customScenario = data;
        },
        clearCustomScenario: function () {
            customScenario = null;
            updateAllCharts();
        },
        // Replace precomputed baseline with engine-computed baseline.
        // Called by fleet-sidebar.js on load so baseline and custom scenarios
        // use the exact same dispatch engine — eliminating phantom deltas.
        setComputedBaseline: function (baselineData) {
            if (!DATA) return;
            // Replace the precomputed baseline with engine-computed baseline
            DATA.scenarios.baseline = baselineData;
            // Only derive intensity envelope if engine didn't provide one
            // (engine computes proper P10/P50/P90 spread across all scenarios)
            if (!DATA.scenarios.baseline.intensity_envelope) {
                deriveIntensityEnvelope(DATA.scenarios.baseline);
            }
            // Rebuild all charts with the new baseline
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

    // ====================================================================
    // PLANT DRILL-DOWN CHARTS
    // ====================================================================
    var plantGenChart = null;
    var plantEmisChart = null;

    function buildPlantDrilldownCharts() {
        if (!DATA) return;
        populatePlantSelector();

        var genCtx = document.getElementById('plantGenChart');
        var emisCtx = document.getElementById('plantEmisChart');
        if (!genCtx || !emisCtx) return;

        var chartOpts = {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 600 },
            interaction: { mode: 'index', intersect: false },
            scales: {
                x: {
                    title: { display: true, text: 'Year', font: { size: 12 } },
                    grid: { display: false },
                    ticks: { maxRotation: 0, font: { size: 11 } }
                },
                y: {
                    grid: { color: '#E0E6EF' },
                    beginAtZero: true,
                    ticks: { font: { size: 11 } }
                }
            },
            plugins: {
                legend: {
                    display: true, position: 'top', align: 'end',
                    labels: {
                        filter: function (item, chartData) {
                            var ds = chartData.datasets[item.datasetIndex];
                            return ds && ds._showInLegend;
                        },
                        usePointStyle: true, pointStyle: 'line', padding: 12,
                        font: { size: 11 }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function (ctx) {
                            if (ctx.dataset._band) return null;
                            var val = ctx.parsed.y;
                            return ctx.dataset.label + ': ' + val.toLocaleString(undefined, { maximumFractionDigits: 0 });
                        }
                    },
                    filter: function (item) {
                        if (Number(item.label) % 5 !== 0) return false;
                        return item.dataset._band === 'p50' || !item.dataset._band;
                    }
                }
            }
        };

        plantGenChart = new Chart(genCtx.getContext('2d'), {
            type: 'line',
            data: { labels: [], datasets: [] },
            options: Object.assign({}, chartOpts, {
                scales: Object.assign({}, chartOpts.scales, {
                    y: Object.assign({}, chartOpts.scales.y, {
                        title: { display: true, text: 'GWh', font: { size: 12 } }
                    })
                })
            })
        });
        plantEmisChart = new Chart(emisCtx.getContext('2d'), {
            type: 'line',
            data: { labels: [], datasets: [] },
            options: Object.assign({}, chartOpts, {
                scales: Object.assign({}, chartOpts.scales, {
                    y: Object.assign({}, chartOpts.scales.y, {
                        title: { display: true, text: 'Metric tons CO₂', font: { size: 12 } }
                    })
                })
            })
        });

        // Wire events
        var plantSel = document.getElementById('plantSelector');
        var intSel = document.getElementById('interventionSelector');
        var intYear = document.getElementById('interventionYear');
        if (plantSel) plantSel.addEventListener('change', function () {
            updateInterventionOptions();
            updatePlantDrilldownCharts();
        });
        if (intSel) intSel.addEventListener('change', updatePlantDrilldownCharts);
        if (intYear) intYear.addEventListener('change', updatePlantDrilldownCharts);
    }

    function populatePlantSelector() {
        var sel = document.getElementById('plantSelector');
        if (!sel || !DATA || !DATA.scenarios || !DATA.scenarios.baseline) return;

        var pp = DATA.scenarios.baseline.plant_percentiles;
        if (!pp) return;

        // Group by fuel type
        var groups = {};
        Object.keys(pp).forEach(function (oid) {
            var p = pp[oid];
            var ft = p.fuel_type;
            var label = ft === 'nuclear' ? 'Nuclear' :
                        ft === 'geothermal' ? 'Geothermal' :
                        ft === 'gas_ccgt' ? 'Gas CCGT' :
                        ft === 'gas_ct' ? 'Gas CT' :
                        ft === 'oil_ct' ? 'Oil CT' :
                        ft === 'gas_oil_ct' ? 'Gas/Oil CT' : ft;
            if (!groups[label]) groups[label] = [];
            groups[label].push({ oid: oid, name: p.name, cap: p.capacity_mw });
        });

        sel.innerHTML = '<option value="">— Select a plant —</option>';
        var order = ['Nuclear', 'Gas CCGT', 'Gas CT', 'Gas/Oil CT', 'Oil CT', 'Geothermal'];
        order.forEach(function (grp) {
            if (!groups[grp]) return;
            var og = document.createElement('optgroup');
            og.label = grp;
            groups[grp].sort(function (a, b) { return b.cap - a.cap; });
            groups[grp].forEach(function (p) {
                var opt = document.createElement('option');
                opt.value = p.oid;
                opt.textContent = p.name + ' (' + Math.round(p.cap) + ' MW)';
                og.appendChild(opt);
            });
            sel.appendChild(og);
        });
    }

    function updateInterventionOptions() {
        var plantSel = document.getElementById('plantSelector');
        var intSel = document.getElementById('interventionSelector');
        if (!plantSel || !intSel || !DATA) return;

        var oid = plantSel.value;
        var pp = DATA.scenarios.baseline.plant_percentiles || {};
        var plant = pp[oid];
        if (!plant) return;

        var isNuclear = plant.fuel_type === 'nuclear';
        var isCCGT = plant.fuel_type === 'gas_ccgt';

        intSel.innerHTML = '';
        var options = [
            { val: 'default_market', label: 'Default Market' },
            { val: 'operating_override', label: 'Operating (forced)' },
            { val: 'retire', label: 'Retire' }
        ];
        if (isCCGT) {
            options.push({ val: 'ccs_retrofit', label: 'CCS Retrofit' });
        }
        if (isNuclear) {
            options.push({ val: 'uprate', label: 'Uprate' });
        }
        options.forEach(function (o) {
            var opt = document.createElement('option');
            opt.value = o.val;
            opt.textContent = o.label;
            intSel.appendChild(opt);
        });
    }

    function updatePlantDrilldownCharts() {
        if (!plantGenChart || !plantEmisChart || !DATA) return;

        var plantSel = document.getElementById('plantSelector');
        var intSel = document.getElementById('interventionSelector');
        var intYearEl = document.getElementById('interventionYear');
        if (!plantSel) return;

        var oid = plantSel.value;
        if (!oid) {
            plantGenChart.data = { labels: [], datasets: [] };
            plantEmisChart.data = { labels: [], datasets: [] };
            plantGenChart.update();
            plantEmisChart.update();
            return;
        }

        var pp = DATA.scenarios.baseline.plant_percentiles || {};
        var plant = pp[oid];
        if (!plant) return;

        var years = getYears();
        if (!years.length) {
            years = [2023, 2024, 2025, 2030, 2035, 2040, 2045, 2050];
        }
        var labels = years.map(String);

        // ── Build baseline envelope ──
        var baseGenEnv = {};
        var baseEmisEnv = {};
        years.forEach(function (y) {
            var ys = String(y);
            var gd = plant.gen_gwh[ys];
            var ed = plant.emissions_t[ys];
            baseGenEnv[ys] = gd ? { p10: gd.p10, p50: gd.p50, p90: gd.p90 } : { p10: 0, p50: 0, p90: 0 };
            baseEmisEnv[ys] = ed ? { p10: ed.p10, p50: ed.p50, p90: ed.p90 } : { p10: 0, p50: 0, p90: 0 };
        });

        // ── Build intervention envelope ──
        var intervention = intSel ? intSel.value : 'default_market';
        var intYear = intYearEl ? parseInt(intYearEl.value, 10) : 2030;
        var intGenEnv = null;
        var intEmisEnv = null;

        if (intervention !== 'default_market') {
            intGenEnv = {};
            intEmisEnv = {};
            years.forEach(function (y) {
                var ys = String(y);
                var bg = baseGenEnv[ys];
                var be = baseEmisEnv[ys];

                if (intervention === 'retire' && y >= intYear) {
                    intGenEnv[ys] = { p10: 0, p50: 0, p90: 0 };
                    intEmisEnv[ys] = { p10: 0, p50: 0, p90: 0 };
                } else if (intervention === 'ccs_retrofit' && y >= intYear) {
                    // Read CCS parameters from sidebar (or defaults)
                    var ccsP = (typeof FleetSidebar !== 'undefined' && FleetSidebar.getCcsParams)
                        ? FleetSidebar.getCcsParams()
                        : { derate_pct: 14, capture_rate_pct: 90, cf_pct: 85 };
                    var derate = ccsP.derate_pct / 100.0;
                    var captureRate = ccsP.capture_rate_pct / 100.0;
                    var cfFrac = ccsP.cf_pct / 100.0;

                    // Generation: capacity × sidebar CF × (1 - derate) × 8760 / 1000 for GWh
                    var capMW = plant.capacity_mw || 0;
                    var ccsGenGwh = Math.round(capMW * cfFrac * (1 - derate) * 8760 / 1000);
                    intGenEnv[ys] = { p10: ccsGenGwh, p50: ccsGenGwh, p90: ccsGenGwh };

                    // Emissions: instant full capture at retrofit year (no ramp)
                    intEmisEnv[ys] = {
                        p10: Math.round(be.p10 * (1 - captureRate)),
                        p50: Math.round(be.p50 * (1 - captureRate)),
                        p90: Math.round(be.p90 * (1 - captureRate))
                    };
                } else if (intervention === 'uprate' && y >= intYear) {
                    // Uprate: +10% generation, 0 emissions (nuclear)
                    var uprateFactor = 1.10;
                    intGenEnv[ys] = {
                        p10: Math.round(bg.p10 * uprateFactor),
                        p50: Math.round(bg.p50 * uprateFactor),
                        p90: Math.round(bg.p90 * uprateFactor)
                    };
                    intEmisEnv[ys] = { p10: 0, p50: 0, p90: 0 };
                } else {
                    // Before intervention year or operating_override: same as baseline
                    intGenEnv[ys] = { p10: bg.p10, p50: bg.p50, p90: bg.p90 };
                    intEmisEnv[ys] = { p10: be.p10, p50: be.p50, p90: be.p90 };
                }
            });
        }

        // ── Build datasets ──
        var genDs = [];
        var emisDs = [];

        // Baseline fan (gray)
        addFanDatasets(genDs, baseGenEnv, years, BASELINE_COLOR, 'Baseline', 'single', {
            color: BASELINE_COLOR, width: 2.5, dash: [], pointRadius: 3,
            legendLabel: 'Baseline (Market)'
        }, { isBaseline: true, scenarioName: 'Baseline' });

        addFanDatasets(emisDs, baseEmisEnv, years, BASELINE_COLOR, 'Baseline', 'single', {
            color: BASELINE_COLOR, width: 2.5, dash: [], pointRadius: 3,
            legendLabel: 'Baseline (Market)'
        }, { isBaseline: true, scenarioName: 'Baseline' });

        // Intervention fan (blue)
        if (intGenEnv) {
            var intColor = '#2372B9';
            var intLabel = intervention === 'retire' ? 'Retire (' + intYear + ')' :
                           intervention === 'ccs_retrofit' ? 'CCS Retrofit (' + intYear + ')' :
                           intervention === 'uprate' ? 'Uprate (' + intYear + ')' :
                           intervention === 'operating_override' ? 'Operating (forced)' :
                           'Intervention';
            addFanDatasets(genDs, intGenEnv, years, intColor, intLabel, 'single', {
                color: intColor, width: 2.5, dash: [], pointRadius: 3,
                legendLabel: intLabel
            }, { isBaseline: false, scenarioName: intLabel });

            addFanDatasets(emisDs, intEmisEnv, years, intColor, intLabel, 'single', {
                color: intColor, width: 2.5, dash: [], pointRadius: 3,
                legendLabel: intLabel
            }, { isBaseline: false, scenarioName: intLabel });
        }

        plantGenChart.data.labels = labels;
        plantGenChart.data.datasets = genDs;
        plantGenChart.update('active');

        plantEmisChart.data.labels = labels;
        plantEmisChart.data.datasets = emisDs;
        plantEmisChart.update('active');
    }

    // ── Boot ──
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
