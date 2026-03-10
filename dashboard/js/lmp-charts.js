// ============================================================
// LMP Trends — Chart Builders, Content Updater & Region Switching
// Depends on: lmp-data.js (LMP_ALL, REGION_META, CAL, ISO_NAMES)
// ============================================================

// ===== CHART DEFAULTS =====
Chart.defaults.font.family = "'Plus Jakarta Sans', 'Helvetica Neue', sans-serif";
Chart.defaults.color = 'rgba(15,23,42,0.5)';

var BLUE = '#0EA5E9', GREEN = '#22C55E', AMBER = '#F59E0B', RED = '#EF4444', PURPLE = '#9C27B0';

var SBTI = [
    { year: 2030, threshold: 50, label: '2030' },
    { year: 2035, threshold: 70, label: '2035' },
    { year: 2040, threshold: 90, label: '2040' },
    { year: 2045, threshold: 95, label: '2045' },
    { year: 2050, threshold: 99.99, label: '2050' }
];

function sbtiAnnotations() {
    var a = {};
    SBTI.forEach(function(ms) {
        a['sbti_' + ms.year] = {
            type: 'line', xMin: ms.threshold, xMax: ms.threshold,
            borderColor: 'rgba(0,0,0,0.08)', borderWidth: 1, borderDash: [3, 3],
            label: { display: true, content: ms.label, position: 'start',
                color: 'rgba(15,23,42,0.35)', font: { size: 9, weight: '600' },
                backgroundColor: 'rgba(255,255,255,0.92)', padding: 3, yAdjust: -4 }
        };
    });
    return a;
}

function pts(xArr, yArr) {
    return xArr.map(function(x, i) { return { x: x, y: yArr[i] }; });
}

function baseScales(yLabel, yMin, yMax) {
    return {
        x: {
            type: 'linear', min: 48, max: 100.5,
            title: { display: true, text: 'Clean Energy Threshold (%)', color: 'rgba(15,23,42,0.45)', font: { size: 11 } },
            grid: { color: 'rgba(0,0,0,0.025)' },
            ticks: { callback: function(v) { return v === 99.99 ? '\u226599.99%' : v + '%'; }, color: 'rgba(15,23,42,0.4)', font: { size: 10 } },
            afterBuildTicks: function(axis) { axis.ticks = [50,60,70,80,90,95,99.99].map(function(v) { return { value: v }; }); }
        },
        y: {
            title: { display: true, text: yLabel, color: 'rgba(15,23,42,0.45)', font: { size: 11 } },
            grid: { color: 'rgba(0,0,0,0.025)' },
            ticks: { color: 'rgba(15,23,42,0.4)', font: { size: 10 } },
            min: yMin, max: yMax
        }
    };
}

function baseTooltip() {
    return {
        backgroundColor: 'rgba(255,255,255,0.97)', titleColor: '#1E293B', bodyColor: 'rgba(15,23,42,0.7)',
        borderColor: 'rgba(0,0,0,0.1)', borderWidth: 1, padding: 12, cornerRadius: 8,
        titleFont: { weight: '700', size: 12 }, bodyFont: { size: 11 },
        filter: function(item) { return item.dataset._type !== 'band'; },
        callbacks: {
            title: function(items) { return items[0] ? items[0].raw.x + '% Clean Energy' : ''; },
            label: function(item) {
                if (item.dataset._type === 'band') return null;
                var u = item.dataset._unit || '$/MWh';
                return item.dataset.label + ': ' + (u === 'hrs' ? Math.round(item.raw.y) + ' hours' : '$' + item.raw.y.toFixed(1) + '/MWh');
            }
        }
    };
}

// ===== CHART INSTANCES (destroyed & rebuilt on region switch) =====
var chartInstances = {};

function destroyCharts() {
    Object.keys(chartInstances).forEach(function(k) { if (chartInstances[k]) chartInstances[k].destroy(); });
    chartInstances = {};
}

// ===== ISO ACCENT COLOR HELPER =====
function isoAccent(iso) {
    return (typeof ISO_COLORS !== 'undefined' && ISO_COLORS[iso]) || BLUE;
}

// Convert hex to rgba string
function hexToRgba(hex, alpha) {
    var r = parseInt(hex.slice(1,3), 16);
    var g = parseInt(hex.slice(3,5), 16);
    var b = parseInt(hex.slice(5,7), 16);
    return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
}

// ===== 1. HERO ENVELOPE CHART =====
function buildHeroChart(D, accent) {
    var c = accent || BLUE;
    var ctx = document.getElementById('heroEnvelopeChart').getContext('2d');
    var T = D.thresholds, E = D.envelope;
    chartInstances.hero = new Chart(ctx, {
        type: 'line',
        data: { datasets: [
            { label: 'P90', data: pts(T, E.p90), borderColor: 'transparent', backgroundColor: hexToRgba(c, 0.09), borderWidth: 0, pointRadius: 0, fill: '+3', tension: 0.35, order: 12, _type: 'band' },
            { label: 'P75', data: pts(T, E.p75), borderColor: 'transparent', backgroundColor: hexToRgba(c, 0.13), borderWidth: 0, pointRadius: 0, fill: '+1', tension: 0.35, order: 11, _type: 'band' },
            { label: 'Median (P50)', data: pts(T, E.p50), borderColor: c, backgroundColor: hexToRgba(c, 0.12), borderWidth: 2.5, pointRadius: 3, pointBackgroundColor: c, fill: false, tension: 0.35, order: 5 },
            { label: 'P25', data: pts(T, E.p25), borderColor: 'transparent', backgroundColor: 'transparent', borderWidth: 0, pointRadius: 0, fill: false, tension: 0.35, order: 11, _type: 'band' },
            { label: 'P10', data: pts(T, E.p10), borderColor: hexToRgba(c, 0.31), borderDash: [4, 3], backgroundColor: 'transparent', borderWidth: 1, pointRadius: 0, fill: false, tension: 0.35, order: 10, _type: 'band' }
        ] },
        options: {
            responsive: true, maintainAspectRatio: false,
            animation: { duration: 1200, easing: 'easeOutQuart', delay: function(ctx) { return ctx.type === 'data' ? ctx.dataIndex * 40 + ctx.datasetIndex * 80 : 0; } },
            scales: baseScales('Average Wholesale Price ($/MWh)', -12, 55),
            plugins: { legend: { display: false }, tooltip: baseTooltip(),
                annotation: { annotations: Object.assign({ zeroLine: { type: 'line', yMin: 0, yMax: 0, borderColor: 'rgba(0,0,0,0.12)', borderWidth: 1, borderDash: [6, 4] } }, sbtiAnnotations()) } }
        }
    });

    // Update hero chart legend swatches to match ISO color
    var legendEl = document.querySelector('.hero-chart-block .chart-legend');
    if (legendEl) {
        legendEl.innerHTML =
            '<span class="chart-legend-item"><span class="chart-legend-swatch swatch-line" style="background:' + c + ';"></span>Median (P50)</span>' +
            '<span class="chart-legend-divider"></span>' +
            '<span class="chart-legend-item"><span class="chart-legend-swatch swatch-band" style="background:' + hexToRgba(c, 0.35) + ';"></span>P25\u2013P75</span>' +
            '<span class="chart-legend-item"><span class="chart-legend-swatch swatch-band" style="background:' + hexToRgba(c, 0.18) + ';"></span>P10\u2013P90</span>';
    }
}

// ===== 2. FUEL ENVELOPE CHART =====
function buildFuelChart(D) {
    var ctx = document.getElementById('fuelEnvelopeChart').getContext('2d');
    var T = D.thresholds, F = D.fuel, B = D.fuelBands;
    var fuelColors = { Low: GREEN, Medium: AMBER, High: RED };
    var ds = [];
    ['Low', 'Medium', 'High'].forEach(function(fl) {
        var c = fuelColors[fl];
        ds.push({ label: fl + ' P90', data: pts(T, B[fl].p90), borderColor: 'transparent', backgroundColor: c + '15', borderWidth: 0, pointRadius: 0, fill: '+1', tension: 0.35, order: 12, _type: 'band' });
        ds.push({ label: fl + ' fuel', data: pts(T, F[fl].p50), borderColor: c, backgroundColor: c + '18', borderWidth: 2.5, pointRadius: 2, pointBackgroundColor: c, fill: false, tension: 0.35, order: 5 });
        ds.push({ label: fl + ' P10', data: pts(T, B[fl].p10), borderColor: 'transparent', backgroundColor: 'transparent', borderWidth: 0, pointRadius: 0, fill: false, tension: 0.35, order: 12, _type: 'band' });
    });
    chartInstances.fuel = new Chart(ctx, {
        type: 'line', data: { datasets: ds },
        options: {
            responsive: true, maintainAspectRatio: false,
            animation: { duration: 1000, easing: 'easeOutQuart' },
            scales: baseScales('Average Wholesale Price ($/MWh)', -12, 55),
            plugins: { legend: { display: false }, tooltip: baseTooltip(),
                annotation: { annotations: Object.assign({ zeroLine: { type: 'line', yMin: 0, yMax: 0, borderColor: 'rgba(0,0,0,0.12)', borderWidth: 1, borderDash: [6, 4] } }, sbtiAnnotations()) } }
        }
    });
}

// ===== 3. MARKET REGIME CHART =====
function buildRegimeChart(D) {
    var ctx = document.getElementById('regimeChart').getContext('2d');
    var T = D.thresholds, R = D.regime;
    chartInstances.regime = new Chart(ctx, {
        type: 'line',
        data: { datasets: [
            { label: 'Zero P90', data: pts(T, R.zeroP90), borderColor: 'transparent', backgroundColor: BLUE + '1A', borderWidth: 0, pointRadius: 0, fill: '+1', tension: 0.35, order: 12, _type: 'band', _unit: 'hrs' },
            { label: 'Zero-price hours', data: pts(T, R.zeroP50), borderColor: BLUE, backgroundColor: BLUE + '18', borderWidth: 2.5, pointRadius: 3, pointBackgroundColor: BLUE, fill: false, tension: 0.35, order: 5, _unit: 'hrs' },
            { label: 'Scarcity P90', data: pts(T, R.scarcityP90), borderColor: 'transparent', backgroundColor: RED + '18', borderWidth: 0, pointRadius: 0, fill: '+1', tension: 0.35, order: 12, _type: 'band', _unit: 'hrs' },
            { label: 'Scarcity hours (>$200)', data: pts(T, R.scarcityP50), borderColor: RED, backgroundColor: RED + '18', borderWidth: 2.5, pointRadius: 3, pointBackgroundColor: RED, fill: false, tension: 0.35, order: 5, _unit: 'hrs' }
        ] },
        options: {
            responsive: true, maintainAspectRatio: false,
            animation: { duration: 1000, easing: 'easeOutQuart' },
            scales: {
                x: baseScales('', 0, 0).x,
                y: { title: { display: true, text: 'Hours per Year', color: 'rgba(15,23,42,0.45)', font: { size: 11 } }, grid: { color: 'rgba(0,0,0,0.025)' },
                    ticks: { color: 'rgba(15,23,42,0.4)', font: { size: 10 }, callback: function(v) { return v >= 1000 ? (v/1000).toFixed(1) + 'k' : v; } }, min: 0, max: 8500 }
            },
            plugins: { legend: { display: false },
                tooltip: { backgroundColor: 'rgba(255,255,255,0.98)', titleColor: '#1E293B', bodyColor: 'rgba(15,23,42,0.75)', borderColor: 'rgba(74,144,217,0.18)', borderWidth: 1, padding: 12, cornerRadius: 8,
                    filter: function(item) { return item.dataset._type !== 'band'; },
                    callbacks: { title: function(items) { return items[0] ? items[0].raw.x + '% Clean Energy' : ''; }, label: function(item) { if (item.dataset._type === 'band') return null; return item.dataset.label + ': ' + Math.round(item.raw.y) + ' hours/year'; } } },
                annotation: { annotations: Object.assign({ halfYear: { type: 'line', yMin: 4380, yMax: 4380, borderColor: 'rgba(0,0,0,0.1)', borderWidth: 1, borderDash: [6, 4], label: { display: true, content: '50% of year', position: 'end', color: 'rgba(15,23,42,0.35)', font: { size: 10 }, backgroundColor: 'transparent' } } }, sbtiAnnotations()) } }
        }
    });
}

// ===== 4. PEAK / OFF-PEAK CHART =====
function buildPeakChart(D) {
    var ctx = document.getElementById('peakSpreadChart').getContext('2d');
    var T = D.thresholds, P = D.peak;
    chartInstances.peak = new Chart(ctx, {
        type: 'line',
        data: { datasets: [
            { label: 'Peak P50', data: pts(T, P.peakP50), borderColor: 'transparent', backgroundColor: 'rgba(0,0,0,0.025)', borderWidth: 0, pointRadius: 0, fill: '+1', tension: 0.35, order: 12, _type: 'band' },
            { label: 'Offpeak P50 (fill target)', data: pts(T, P.offpeakP50), borderColor: 'transparent', backgroundColor: 'transparent', borderWidth: 0, pointRadius: 0, fill: false, tension: 0.35, order: 12, _type: 'band' },
            { label: 'Peak (7am-10pm weekdays)', data: pts(T, P.peakP50), borderColor: AMBER, backgroundColor: AMBER + '20', borderWidth: 2.5, pointRadius: 3, pointBackgroundColor: AMBER, fill: false, tension: 0.35, order: 5 },
            { label: 'Off-peak', data: pts(T, P.offpeakP50), borderColor: PURPLE, backgroundColor: PURPLE + '20', borderWidth: 2.5, pointRadius: 3, pointBackgroundColor: PURPLE, fill: false, tension: 0.35, order: 5 }
        ] },
        options: {
            responsive: true, maintainAspectRatio: false,
            animation: { duration: 1000, easing: 'easeOutQuart' },
            scales: baseScales('Wholesale Price ($/MWh)', -16, 55),
            plugins: { legend: { display: false }, tooltip: baseTooltip(),
                annotation: { annotations: Object.assign({ zeroLine: { type: 'line', yMin: 0, yMax: 0, borderColor: 'rgba(0,0,0,0.12)', borderWidth: 1, borderDash: [6, 4] } }, sbtiAnnotations()) } }
        }
    });
}

// ===== 6. CANNIBALIZATION CHART =====
function buildCannibalizationChart(D, accent) {
    var ctx = document.getElementById('cannibalizationChart');
    if (!ctx) return;
    var T = D.thresholds, E = D.envelope, R = D.regime;
    var c = accent || BLUE;
    // Zero-price hours as % of year for area fill
    var zeroPct = R.zeroP50.map(function(h) { return Math.round(h / 87.6) / 10; }); // % of 8760
    chartInstances.cannibalization = new Chart(ctx, {
        type: 'line',
        data: { datasets: [
            // Shaded area: zero/negative price hours (% of year) — right axis
            { label: 'Zero/negative price hours', data: pts(T, R.zeroP50), borderColor: AMBER + '80',
              backgroundColor: AMBER + '25', borderWidth: 1.5, pointRadius: 0, fill: 'origin',
              tension: 0.35, order: 8, yAxisID: 'y1', _unit: 'hrs' },
            // Main line: wholesale LMP P50 — left axis
            { label: 'Avg wholesale LMP (P50)', data: pts(T, E.p50), borderColor: c,
              backgroundColor: 'transparent', borderWidth: 2.5, pointRadius: 3,
              pointBackgroundColor: c, fill: false, tension: 0.35, order: 3, yAxisID: 'y' }
        ] },
        options: {
            responsive: true, maintainAspectRatio: false,
            animation: { duration: 1000, easing: 'easeOutQuart' },
            scales: {
                x: {
                    type: 'linear', min: 48, max: 100.5,
                    title: { display: true, text: 'Clean Energy Threshold (%)', color: 'rgba(15,23,42,0.45)', font: { size: 11 } },
                    grid: { color: 'rgba(0,0,0,0.025)' },
                    ticks: { callback: function(v) { return v === 99.99 ? '\u226599.99%' : v + '%'; }, color: 'rgba(15,23,42,0.4)', font: { size: 10 } },
                    afterBuildTicks: function(axis) { axis.ticks = [50,60,70,80,90,95,99.99].map(function(v) { return { value: v }; }); }
                },
                y: {
                    position: 'left',
                    title: { display: true, text: 'Average Wholesale Price ($/MWh)', color: 'rgba(15,23,42,0.45)', font: { size: 11 } },
                    grid: { color: 'rgba(0,0,0,0.025)' },
                    ticks: { color: 'rgba(15,23,42,0.4)', font: { size: 10 } },
                    min: -15, max: 55
                },
                y1: {
                    position: 'right',
                    title: { display: true, text: 'Zero/Negative Price Hours', color: AMBER + 'BB', font: { size: 11 } },
                    grid: { drawOnChartArea: false },
                    ticks: { color: AMBER + 'BB', font: { size: 10 },
                        callback: function(v) { return v >= 1000 ? (v/1000).toFixed(1) + 'k' : v; } },
                    min: 0, max: 8760
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(255,255,255,0.97)', titleColor: '#1E293B', bodyColor: 'rgba(15,23,42,0.7)',
                    borderColor: 'rgba(0,0,0,0.1)', borderWidth: 1, padding: 12, cornerRadius: 8,
                    titleFont: { weight: '700', size: 12 }, bodyFont: { size: 11 },
                    callbacks: {
                        title: function(items) { return items[0] ? items[0].raw.x + '% Clean Energy' : ''; },
                        label: function(item) {
                            if (item.dataset.yAxisID === 'y1') return item.dataset.label + ': ' + Math.round(item.raw.y) + ' hours (' + (item.raw.y / 87.6).toFixed(0) + '% of year)';
                            return item.dataset.label + ': $' + item.raw.y.toFixed(1) + '/MWh';
                        }
                    }
                },
                annotation: { annotations: Object.assign({ zeroLine: { type: 'line', yMin: 0, yMax: 0, borderColor: 'rgba(0,0,0,0.12)', borderWidth: 1, borderDash: [6, 4] } }, sbtiAnnotations()) }
            }
        }
    });
}

// ===== 7. STORAGE ARBITRAGE CHART =====
function buildStorageArbitrageChart(D, iso) {
    var ctx = document.getElementById('storageArbitrageChart');
    if (!ctx) return;
    var T = D.thresholds, P = D.peak;

    // Storage dispatch data from RESOURCE_MIX_DATA (21 thresholds)
    // Map to LMP thresholds (15 points: 50,55,...,99,99.99)
    var lmpThresholds = T; // [50,55,60,...,99,99.99]
    var fullThresholds = typeof THRESHOLDS !== 'undefined' ? THRESHOLDS : null;
    var mix = typeof RESOURCE_MIX_DATA !== 'undefined' ? RESOURCE_MIX_DATA[iso] : null;

    // Map storage dispatch from 21-threshold to 15-threshold space
    function mapToLmpThresholds(arr21) {
        if (!arr21 || !fullThresholds) return lmpThresholds.map(function() { return 0; });
        return lmpThresholds.map(function(t) {
            var tVal = t >= 99.99 ? 99.99 : t;
            var idx = -1;
            for (var i = 0; i < fullThresholds.length; i++) {
                if (Math.abs(fullThresholds[i] - tVal) < 0.01) { idx = i; break; }
            }
            return idx >= 0 ? arr21[idx] : 0;
        });
    }

    var bat4 = mix ? mapToLmpThresholds(mix.battery) : lmpThresholds.map(function() { return 0; });
    var bat8 = mix ? mapToLmpThresholds(mix.battery8) : lmpThresholds.map(function() { return 0; });
    var ldes = mix ? mapToLmpThresholds(mix.ldes) : lmpThresholds.map(function() { return 0; });

    var CYAN = typeof RESOURCE_COLORS !== 'undefined' ? RESOURCE_COLORS.battery : '#06B6D4';
    var TEAL = typeof RESOURCE_COLORS !== 'undefined' ? RESOURCE_COLORS.battery8 : '#0891B2';
    var PINK = typeof RESOURCE_COLORS !== 'undefined' ? RESOURCE_COLORS.ldes : '#E91E63';

    chartInstances.storageArbitrage = new Chart(ctx, {
        type: 'line',
        data: { datasets: [
            // Shaded area: peak-offpeak spread — right axis
            { label: 'Peak\u2013offpeak spread', data: pts(T, P.spreadP50), borderColor: AMBER + '80',
              backgroundColor: AMBER + '20', borderWidth: 1.5, pointRadius: 0, fill: 'origin',
              tension: 0.35, order: 8, yAxisID: 'y1', _type: 'band' },
            // Storage dispatch lines — left axis
            { label: 'Battery 4hr dispatch %', data: pts(T, bat4), borderColor: CYAN,
              backgroundColor: 'transparent', borderWidth: 2.5, pointRadius: 3,
              pointBackgroundColor: CYAN, fill: false, tension: 0.35, order: 3, yAxisID: 'y' },
            { label: 'Battery 8hr dispatch %', data: pts(T, bat8), borderColor: TEAL,
              backgroundColor: 'transparent', borderWidth: 2.5, pointRadius: 3,
              pointBackgroundColor: TEAL, fill: false, tension: 0.35, order: 3, yAxisID: 'y' },
            { label: 'LDES dispatch %', data: pts(T, ldes), borderColor: PINK,
              backgroundColor: 'transparent', borderWidth: 2.5, pointRadius: 3,
              pointBackgroundColor: PINK, fill: false, tension: 0.35, order: 3, yAxisID: 'y' }
        ] },
        options: {
            responsive: true, maintainAspectRatio: false,
            animation: { duration: 1000, easing: 'easeOutQuart' },
            scales: {
                x: {
                    type: 'linear', min: 48, max: 100.5,
                    title: { display: true, text: 'Clean Energy Threshold (%)', color: 'rgba(15,23,42,0.45)', font: { size: 11 } },
                    grid: { color: 'rgba(0,0,0,0.025)' },
                    ticks: { callback: function(v) { return v === 99.99 ? '\u226599.99%' : v + '%'; }, color: 'rgba(15,23,42,0.4)', font: { size: 10 } },
                    afterBuildTicks: function(axis) { axis.ticks = [50,60,70,80,90,95,99.99].map(function(v) { return { value: v }; }); }
                },
                y: {
                    position: 'left',
                    title: { display: true, text: 'Storage Dispatch (% of demand)', color: 'rgba(15,23,42,0.45)', font: { size: 11 } },
                    grid: { color: 'rgba(0,0,0,0.025)' },
                    ticks: { color: 'rgba(15,23,42,0.4)', font: { size: 10 },
                        callback: function(v) { return v + '%'; } },
                    min: 0
                },
                y1: {
                    position: 'right',
                    title: { display: true, text: 'Peak\u2013Offpeak Spread ($/MWh)', color: AMBER + 'BB', font: { size: 11 } },
                    grid: { drawOnChartArea: false },
                    ticks: { color: AMBER + 'BB', font: { size: 10 },
                        callback: function(v) { return '$' + v; } },
                    min: 0
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(255,255,255,0.97)', titleColor: '#1E293B', bodyColor: 'rgba(15,23,42,0.7)',
                    borderColor: 'rgba(0,0,0,0.1)', borderWidth: 1, padding: 12, cornerRadius: 8,
                    titleFont: { weight: '700', size: 12 }, bodyFont: { size: 11 },
                    filter: function(item) { return item.dataset._type !== 'band'; },
                    callbacks: {
                        title: function(items) { return items[0] ? items[0].raw.x + '% Clean Energy' : ''; },
                        label: function(item) {
                            if (item.dataset._type === 'band') return null;
                            if (item.dataset.yAxisID === 'y1') return item.dataset.label + ': $' + item.raw.y.toFixed(1) + '/MWh';
                            return item.dataset.label + ': ' + item.raw.y.toFixed(1) + '%';
                        }
                    }
                },
                annotation: { annotations: sbtiAnnotations() }
            }
        }
    });
}

// ===== 8. GAS STRANDING CHART =====
function buildGasStrandingChart(iso) {
    var ctx = document.getElementById('gasStrandingChart');
    if (!ctx) return;
    var gasData = typeof GAS_BACKUP_DATA !== 'undefined' ? GAS_BACKUP_DATA[iso] : null;
    if (!gasData) return;

    // Use full 21 thresholds from shared-data.js but filter to 50%+
    var fullT = typeof THRESHOLDS !== 'undefined' ? THRESHOLDS : [];
    var startIdx = 4; // index of 50.0 in THRESHOLDS [10,20,30,40,50,...]
    var T = fullT.slice(startIdx);
    var gasNeeded = gasData.gas_backup_mw.slice(startIdx).map(function(v) { return v / 1000; }); // GW
    var baselineGas = gasData.gas_backup_mw[startIdx] / 1000; // GW at 50%

    // Stranded = baseline - needed (clamp to 0)
    var stranded = gasNeeded.map(function(v) { return Math.max(0, baselineGas - v); });

    var GAS_COLOR = typeof RESOURCE_COLORS !== 'undefined' ? RESOURCE_COLORS.fossilGas : '#6B7280';

    chartInstances.gasStranding = new Chart(ctx, {
        type: 'line',
        data: { datasets: [
            // Installed gas fleet baseline — horizontal reference
            { label: '2025 installed gas fleet', data: T.map(function(t) { return { x: t >= 99.99 ? 99.99 : t, y: baselineGas }; }),
              borderColor: GAS_COLOR + '50', backgroundColor: 'transparent',
              borderWidth: 1.5, borderDash: [6, 4], pointRadius: 0, fill: false,
              tension: 0, order: 5 },
            // Stranded capacity area (fill between baseline and needed)
            { label: 'Stranded capacity', data: T.map(function(t, i) { return { x: t >= 99.99 ? 99.99 : t, y: baselineGas }; }),
              borderColor: 'transparent', backgroundColor: 'rgba(239,68,68,0.12)',
              borderWidth: 0, pointRadius: 0, fill: '+1', tension: 0.35, order: 10, _type: 'band' },
            // Gas capacity required line
            { label: 'Gas capacity required (GW)', data: T.map(function(t, i) { return { x: t >= 99.99 ? 99.99 : t, y: gasNeeded[i] }; }),
              borderColor: GAS_COLOR, backgroundColor: hexToRgba(GAS_COLOR, 0.08),
              borderWidth: 2.5, pointRadius: 3, pointBackgroundColor: GAS_COLOR,
              fill: 'origin', tension: 0.35, order: 3 }
        ] },
        options: {
            responsive: true, maintainAspectRatio: false,
            animation: { duration: 1000, easing: 'easeOutQuart' },
            scales: {
                x: {
                    type: 'linear', min: 48, max: 100.5,
                    title: { display: true, text: 'Clean Energy Threshold (%)', color: 'rgba(15,23,42,0.45)', font: { size: 11 } },
                    grid: { color: 'rgba(0,0,0,0.025)' },
                    ticks: { callback: function(v) { return v === 99.99 ? '\u226599.99%' : v + '%'; }, color: 'rgba(15,23,42,0.4)', font: { size: 10 } },
                    afterBuildTicks: function(axis) { axis.ticks = [50,60,70,80,90,95,99.99].map(function(v) { return { value: v }; }); }
                },
                y: {
                    title: { display: true, text: 'Gas Capacity (GW)', color: 'rgba(15,23,42,0.45)', font: { size: 11 } },
                    grid: { color: 'rgba(0,0,0,0.025)' },
                    ticks: { color: 'rgba(15,23,42,0.4)', font: { size: 10 },
                        callback: function(v) { return v + ' GW'; } },
                    min: 0
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(255,255,255,0.97)', titleColor: '#1E293B', bodyColor: 'rgba(15,23,42,0.7)',
                    borderColor: 'rgba(0,0,0,0.1)', borderWidth: 1, padding: 12, cornerRadius: 8,
                    titleFont: { weight: '700', size: 12 }, bodyFont: { size: 11 },
                    filter: function(item) { return item.dataset._type !== 'band'; },
                    callbacks: {
                        title: function(items) { return items[0] ? (items[0].raw.x >= 99.99 ? '\u226599.99' : items[0].raw.x) + '% Clean Energy' : ''; },
                        label: function(item) {
                            if (item.dataset._type === 'band') return null;
                            return item.dataset.label + ': ' + item.raw.y.toFixed(1) + ' GW';
                        }
                    }
                },
                annotation: { annotations: sbtiAnnotations() }
            }
        }
    });
}

// ===== BUILD ALL LMP CHARTS (nuclear crossover charts are in lmp_trends.html inline script) =====
function buildAllCharts(D, accent) {
    destroyCharts();
    buildHeroChart(D, accent);
    buildFuelChart(D);
    buildRegimeChart(D);
    buildPeakChart(D);
    if (typeof LMP_CAPACITY !== 'undefined') {
        buildCapRevChart(currentISO, accent);
        buildNucRevChart(currentISO);
    }
    if (typeof REFERENCE_CASE !== 'undefined') {
        buildRefCaseChart(currentISO, accent);
    }
    buildCannibalizationChart(D, accent);
    buildStorageArbitrageChart(D, currentISO);
    buildGasStrandingChart(currentISO);
}

// ===== CAPACITY MARKET REVENUE CHART =====
function buildCapRevChart(iso, accent) {
    var cap = LMP_CAPACITY[iso];
    if (!cap) return;
    var ctx = document.getElementById('capRevChart');
    if (!ctx) return;

    var T = cap.thresholds;
    var lmpP50 = cap.avg_lmp_p50;
    var sys = cap.capacity_rev_system_mwh;
    var total = cap.total_market_rev_mwh;

    var datasets = [
        {
            label: 'Energy Only (LMP)',
            data: pts(T, lmpP50),
            borderColor: accent || BLUE,
            backgroundColor: 'transparent',
            borderWidth: 2,
            borderDash: [5, 3],
            pointRadius: 2,
            order: 2
        },
        {
            label: 'Total Market Revenue (Energy + Capacity)',
            data: pts(T, total.p50),
            borderColor: accent || BLUE,
            backgroundColor: hexToRgba(accent || BLUE, 0.08),
            borderWidth: 2.5,
            pointRadius: 3,
            fill: true,
            order: 1
        }
    ];

    // P10/P90 band for total market revenue
    if (total.p10 && total.p90) {
        datasets.push({
            label: 'Total Rev P10',
            data: pts(T, total.p10),
            borderColor: 'transparent',
            backgroundColor: 'transparent',
            pointRadius: 0,
            order: 3
        });
        datasets.push({
            label: 'Total Rev P90',
            data: pts(T, total.p90),
            borderColor: 'transparent',
            backgroundColor: hexToRgba(accent || BLUE, 0.05),
            fill: '-1',
            pointRadius: 0,
            order: 3
        });
    }

    chartInstances.capRev = new Chart(ctx, {
        type: 'line',
        data: { datasets: datasets },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { display: true, position: 'top', labels: { usePointStyle: true, font: { size: 11 } } },
                annotation: { annotations: sbtiAnnotations() },
                tooltip: {
                    callbacks: {
                        label: function(c) { return c.dataset.label + ': $' + c.parsed.y.toFixed(1) + '/MWh'; }
                    }
                }
            },
            scales: baseScales('Wholesale Revenue ($/MWh)', undefined, undefined)
        }
    });

    // Update narrative
    var el = document.getElementById('nucCapRevPJM');
    if (el && cap.capacity_rev_clean_firm_mwh) {
        el.textContent = '$' + cap.capacity_rev_clean_firm_mwh.p50[0].toFixed(0) + '/MWh';
    }

    // Update subtitle
    var sub = document.getElementById('capRevSubtitle');
    if (sub && sys) {
        var capAt50 = sys.p50[0].toFixed(1);
        sub.textContent = iso + ': Capacity market adds $' + capAt50 + '/MWh at 50% clean to energy-only LMP';
    }
}

// ===== NUCLEAR TOTAL MARKET REVENUE CHART =====
function buildNucRevChart(iso) {
    var cap = LMP_CAPACITY[iso];
    if (!cap || !cap.nuclear_total_rev_by_fuel) return;
    var ctx = document.getElementById('nucRevChart');
    if (!ctx) return;

    var T = cap.thresholds;
    var nuc = cap.nuclear_total_rev_by_fuel;

    var datasets = [
        {
            label: 'Low Fuel Prices',
            data: pts(T, nuc.Low),
            borderColor: '#22C55E', backgroundColor: 'transparent',
            borderWidth: 1.5, borderDash: [4, 2], pointRadius: 2
        },
        {
            label: 'Medium Fuel Prices',
            data: pts(T, nuc.Medium),
            borderColor: BLUE, backgroundColor: 'transparent',
            borderWidth: 2.5, pointRadius: 3
        },
        {
            label: 'High Fuel Prices',
            data: pts(T, nuc.High),
            borderColor: '#EF4444', backgroundColor: 'transparent',
            borderWidth: 1.5, borderDash: [4, 2], pointRadius: 2
        }
    ];

    // Nuclear cost thresholds as annotations
    var nucAnnotations = Object.assign({}, sbtiAnnotations(), {
        opCostLow: {
            type: 'line', yMin: 25, yMax: 25,
            borderColor: 'rgba(34,197,94,0.3)', borderWidth: 1, borderDash: [6, 3],
            label: { display: true, content: 'Cash OpEx ($25)', position: 'start',
                color: 'rgba(34,197,94,0.6)', font: { size: 9 }, backgroundColor: 'rgba(255,255,255,0.9)', padding: 2 }
        },
        allIn: {
            type: 'line', yMin: 38, yMax: 38,
            borderColor: 'rgba(245,158,11,0.4)', borderWidth: 1.5, borderDash: [6, 3],
            label: { display: true, content: 'All-In Sustaining ($38)', position: 'start',
                color: 'rgba(245,158,11,0.7)', font: { size: 9 }, backgroundColor: 'rgba(255,255,255,0.9)', padding: 2 }
        },
        viable: {
            type: 'line', yMin: 44, yMax: 44,
            borderColor: 'rgba(239,68,68,0.4)', borderWidth: 1.5, borderDash: [6, 3],
            label: { display: true, content: 'Viability Threshold ($44)', position: 'start',
                color: 'rgba(239,68,68,0.7)', font: { size: 9 }, backgroundColor: 'rgba(255,255,255,0.9)', padding: 2 }
        },
        ptc: {
            type: 'box', yMin: 0, yMax: 15,
            backgroundColor: 'rgba(99,102,241,0.04)', borderWidth: 0,
            label: { display: true, content: '45U PTC floor ($15/MWh)', position: 'start',
                color: 'rgba(99,102,241,0.4)', font: { size: 9 } }
        }
    });

    chartInstances.nucRev = new Chart(ctx, {
        type: 'line',
        data: { datasets: datasets },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { display: true, position: 'top', labels: { usePointStyle: true, font: { size: 11 } } },
                annotation: { annotations: nucAnnotations },
                tooltip: {
                    callbacks: {
                        label: function(c) { return c.dataset.label + ': $' + c.parsed.y.toFixed(1) + '/MWh (energy + capacity)'; }
                    }
                }
            },
            scales: baseScales('Nuclear Total Market Revenue ($/MWh)', 0, 65)
        }
    });

    // Update viability threshold text
    var el = document.getElementById('nucViableCapPct');
    if (el && nuc.Medium) {
        var t38 = T.find(function(t, i) { return nuc.Medium[i] < 38; });
        if (t38) el.textContent = t38 + '%';
    }
}

// ===== REFERENCE CASE vs STEP 5B: PURE POWER MARKET COMPARISON =====
function buildRefCaseChart(iso, accent) {
    var ref = REFERENCE_CASE[iso];
    var cap = (typeof LMP_CAPACITY !== 'undefined') ? LMP_CAPACITY[iso] : null;
    var lmpAll = (typeof LMP_ALL !== 'undefined') ? LMP_ALL[iso] : null;
    if (!lmpAll) return;
    var ctx = document.getElementById('refCaseChart');
    if (!ctx) return;

    var T = lmpAll.thresholds; // clean% thresholds (50, 55, 60, ..., 99.99)

    var datasets = [];

    // Step 5B: Energy-only wholesale LMP (P10/P50/P90 envelope)
    datasets.push({
        label: 'Step 5B Energy LMP (P50)',
        data: pts(T, lmpAll.envelope.p50),
        borderColor: accent || BLUE,
        backgroundColor: 'transparent',
        borderWidth: 2.5, pointRadius: 3, order: 1
    });
    // P10/P90 band
    datasets.push({
        label: 'Step 5B P10',
        data: pts(T, lmpAll.envelope.p10),
        borderColor: 'transparent', backgroundColor: 'transparent',
        pointRadius: 0, order: 3, _type: 'band'
    });
    datasets.push({
        label: 'Step 5B P90',
        data: pts(T, lmpAll.envelope.p90),
        borderColor: 'transparent',
        backgroundColor: hexToRgba(accent || BLUE, 0.08),
        fill: '-1', pointRadius: 0, order: 3, _type: 'band'
    });

    // Step 5B: Energy + Capacity (total market revenue, no RECs)
    if (cap && cap.total_market_rev_mwh) {
        datasets.push({
            label: 'Step 5B Power Market (Energy + Capacity)',
            data: pts(cap.thresholds, cap.total_market_rev_mwh.p50),
            borderColor: '#6366F1',
            backgroundColor: 'transparent',
            borderWidth: 2, borderDash: [6, 3],
            pointRadius: 3, order: 2
        });
    }

    // Step 10: Per-family power market curves with P10/P50/P90 CI bands
    var FAMILY_STYLES = {
        'reference':  { color: '#EF4444', label: 'Reference (Market)',   pointStyle: 'triangle' },
        'power_nz':   { color: '#F59E0B', label: 'AT Power Net-Zero',   pointStyle: 'rect' },
        'economy_nz': { color: '#8B5CF6', label: 'AT Economy Net-Zero',  pointStyle: 'rectRot' }
    };

    if (ref && ref.families) {
        var familyKeys = Object.keys(ref.families);
        familyKeys.forEach(function(key) {
            var family = ref.families[key];
            var style = FAMILY_STYLES[key] || { color: '#6B7280', label: key, pointStyle: 'circle' };
            var pmap = family.clean_price_map_2050 || [];

            var p50pts = [], p10pts = [], p90pts = [];
            pmap.forEach(function(pt) {
                if (pt.n >= 3 && pt.clean_pct >= 50 && pt.clean_pct <= 100) {
                    var val50 = (pt.power_market_p50 !== undefined && pt.power_market_p50 !== 0)
                        ? pt.power_market_p50 : pt.avg_lmp_p50;
                    p50pts.push({ x: pt.clean_pct, y: val50 });

                    if (pt.power_market_p10 !== undefined) {
                        p10pts.push({ x: pt.clean_pct, y: pt.power_market_p10 });
                    }
                    if (pt.power_market_p90 !== undefined) {
                        p90pts.push({ x: pt.clean_pct, y: pt.power_market_p90 });
                    }
                }
            });

            if (p50pts.length === 0) return;

            // P50 median line
            datasets.push({
                label: 'Step 10 ' + style.label + ' (P50)',
                data: p50pts,
                borderColor: style.color,
                backgroundColor: 'transparent',
                borderWidth: 2,
                pointRadius: 4, pointStyle: style.pointStyle,
                showLine: true, tension: 0.3, order: 1
            });

            // P10/P90 CI band
            if (p10pts.length > 0 && p90pts.length > 0) {
                datasets.push({
                    label: 'Step 10 ' + style.label + ' P10',
                    data: p10pts,
                    borderColor: 'transparent', backgroundColor: 'transparent',
                    pointRadius: 0, showLine: true, tension: 0.3,
                    order: 3, _type: 'band'
                });
                datasets.push({
                    label: 'Step 10 ' + style.label + ' P90',
                    data: p90pts,
                    borderColor: 'transparent',
                    backgroundColor: hexToRgba(style.color, 0.10),
                    fill: '-1', pointRadius: 0, showLine: true, tension: 0.3,
                    order: 3, _type: 'band'
                });
            }
        });
    } else if (ref && ref.clean_price_map_2050) {
        // Fallback: legacy single-curve mode (no families data)
        var s10pts = [];
        ref.clean_price_map_2050.forEach(function(pt) {
            var val = (pt.power_market_p50 !== undefined && pt.power_market_p50 !== 0)
                ? pt.power_market_p50 : pt.avg_lmp_p50;
            if (pt.n >= 3 && pt.clean_pct >= 50 && pt.clean_pct <= 100) {
                s10pts.push({ x: pt.clean_pct, y: val });
            }
        });
        if (s10pts.length > 0) {
            datasets.push({
                label: 'Step 10 Power Market @ 2050 (No RECs)',
                data: s10pts,
                borderColor: '#EF4444',
                backgroundColor: hexToRgba('#EF4444', 0.15),
                borderWidth: 2,
                pointRadius: 5, pointStyle: 'triangle',
                showLine: true, tension: 0.3, order: 1
            });
        }
    }

    chartInstances.refCase = new Chart(ctx, {
        type: 'line',
        data: { datasets: datasets },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { display: true, position: 'top', labels: { usePointStyle: true, font: { size: 11 },
                    filter: function(item) {
                        // Hide all P10 and P90 band entries from legend
                        return item.text.indexOf('P10') === -1 && item.text.indexOf('P90') === -1;
                    }
                } },
                annotation: { annotations: sbtiAnnotations() },
                tooltip: {
                    filter: function(item) { return item.dataset._type !== 'band'; },
                    callbacks: {
                        title: function(items) { return items[0] ? items[0].raw.x + '% Clean Energy' : ''; },
                        label: function(c) {
                            if (c.dataset._type === 'band') return null;
                            return c.dataset.label + ': $' + c.parsed.y.toFixed(1) + '/MWh';
                        }
                    }
                }
            },
            scales: baseScales('Power Market Revenue ($/MWh)', undefined, undefined)
        }
    });

    // Update subtitle
    var sub = document.getElementById('refCaseSubtitle');
    if (sub) {
        sub.textContent = iso + ': Step 5B (policy-driven) vs Step 10 (market-driven) — pure power markets, no REC revenue';
    }
}

// ===== CONTENT UPDATER =====
function updateContent(iso) {
    var m = REGION_META[iso];
    if (!m) return;

    // Hero section
    var el;
    el = document.getElementById('heroKicker'); if (el) el.textContent = m.kicker;
    el = document.getElementById('heroTitle'); if (el) el.innerHTML = m.heroTitle;
    el = document.getElementById('heroDesc1'); if (el) el.textContent = m.heroDesc1;
    el = document.getElementById('heroDesc2'); if (el) el.textContent = m.heroDesc2;

    // Hero stats
    el = document.getElementById('heroStats');
    if (el && m.heroStats) {
        el.innerHTML = m.heroStats.map(function(s) {
            return '<div class="hero-stat"><div class="hero-stat-value ' + s.color + '">' + s.value + '</div><div class="hero-stat-label">' + s.label + '</div></div>';
        }).join('');
    }

    // Scroll sections
    el = document.getElementById('sect1Narrative'); if (el) el.innerHTML = m.sect1HTML;
    el = document.getElementById('sect2Narrative'); if (el) el.innerHTML = m.sect2HTML;
    el = document.getElementById('sect3Narrative'); if (el) el.innerHTML = m.sect3HTML;

    // Nuclear section elements (on lmp_trends.html — safe to call, elements may not exist)
    el = document.getElementById('nuclearCalloutContent');
    if (el) el.innerHTML = '<h4>' + m.nuclearCalloutH4 + '</h4><p>' + m.nuclearCalloutP + '</p>';
    el = document.getElementById('nuclearSectionHeader');
    if (el) el.innerHTML = '<span class="section-tag" style="background:rgba(239,68,68,0.06);color:#f87171;">' + m.nuclearHeaderTag + '</span>' +
        '<h2>' + m.nuclearHeaderTitle + '</h2><p>' + m.nuclearHeaderBody + '</p>';
    el = document.getElementById('nuclearChartTitle1'); if (el) el.textContent = m.nuclearChartTitle1;
    el = document.getElementById('nuclearChartTitle2'); if (el) el.textContent = m.nuclearChartTitle2;
    el = document.getElementById('nuclearNarrative'); if (el) el.innerHTML = m.nuclearNarrativeHTML;

    // Synthesis
    el = document.getElementById('synthesisGrid'); if (el) el.innerHTML = m.synthesisHTML;

    // Footer
    el = document.getElementById('footerNote'); if (el) el.textContent = m.footerNote;

    // Re-observe scroll animations
    observeScrollElements();
}

// ===== SCROLL OBSERVER =====
var fadeObs = new IntersectionObserver(function(entries) {
    entries.forEach(function(e) {
        if (e.isIntersecting) e.target.classList.add('visible');
    });
}, { threshold: 0.15, rootMargin: '0px 0px -50px 0px' });

function observeScrollElements() {
    document.querySelectorAll('.narrative-card, .synthesis-card').forEach(function(el) {
        el.classList.remove('visible');
        fadeObs.observe(el);
    });
    // Check if any are already visible in viewport
    setTimeout(function() {
        document.querySelectorAll('.narrative-card, .synthesis-card').forEach(function(el) {
            var r = el.getBoundingClientRect();
            if (r.top < window.innerHeight && r.bottom > 0) el.classList.add('visible');
        });
    }, 100);
}

// ===== ISO ACCENT THEMING =====
function applyISOAccent(iso) {
    var c = isoAccent(iso);

    // Update kicker color
    var kicker = document.getElementById('heroKicker');
    if (kicker) kicker.style.color = c;

    // Update accent spans in hero title
    document.querySelectorAll('.hero-text .accent').forEach(function(el) {
        el.style.color = c;
    });

    // Update header subtitle accent
    var subtitle = document.getElementById('headerSubtitle');
    if (subtitle) subtitle.style.color = c;
}

// ===== REGION SWITCHING =====
var currentISO = 'PJM';

function switchRegion(iso) {
    if (iso === currentISO) return;
    currentISO = iso;
    var c = isoAccent(iso);
    buildAllCharts(LMP_ALL[iso], c);
    updateContent(iso);
    applyISOAccent(iso);
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ===== PILLS HANDLER (LMP page only) =====
if (document.getElementById('heroEnvelopeChart')) {
    document.querySelectorAll('.region-pill').forEach(function(pill) {
        pill.addEventListener('click', function() {
            var active = document.querySelector('.region-pill.active');
            if (active) active.classList.remove('active');
            this.classList.add('active');
            switchRegion(this.dataset.iso);
        });
    });

    // ===== INIT (LMP page only) =====
    var initAccent = isoAccent('PJM');
    buildAllCharts(LMP_ALL.PJM, initAccent);
    applyISOAccent('PJM');
    observeScrollElements();
}
