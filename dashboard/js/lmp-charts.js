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

// ===== 5. NUCLEAR ECONOMICS CHART =====
function buildNuclearEconChart(D) {
    var ctx = document.getElementById('nuclearEconChart').getContext('2d');
    var T = D.thresholds, N = D.nuclear;
    var viU = T.map(function() { return N.viableHigh; }), viL = T.map(function() { return N.viableLow; });
    var aiU = T.map(function() { return N.allInHigh; }), aiL = T.map(function() { return N.allInLow; });
    var coU = T.map(function() { return N.opCostHigh; }), coL = T.map(function() { return N.opCostLow; });
    chartInstances.nucEcon = new Chart(ctx, {
        type: 'line',
        data: { datasets: [
            { label: 'Viable ceiling', data: pts(T, viU), borderColor: 'transparent', backgroundColor: 'rgba(245,158,11,0.08)', borderWidth: 0, pointRadius: 0, fill: '+1', tension: 0, order: 14, _type: 'band' },
            { label: 'Viable floor', data: pts(T, viL), borderColor: 'rgba(245,158,11,0.35)', backgroundColor: 'transparent', borderWidth: 1, borderDash: [6,3], pointRadius: 0, fill: false, tension: 0, order: 14, _type: 'band' },
            { label: 'All-in ceiling', data: pts(T, aiU), borderColor: 'transparent', backgroundColor: 'rgba(239,68,68,0.08)', borderWidth: 0, pointRadius: 0, fill: '+1', tension: 0, order: 13, _type: 'band' },
            { label: 'All-in floor', data: pts(T, aiL), borderColor: 'rgba(239,68,68,0.3)', backgroundColor: 'transparent', borderWidth: 1, borderDash: [4,3], pointRadius: 0, fill: false, tension: 0, order: 13, _type: 'band' },
            { label: 'Cash ops ceiling', data: pts(T, coU), borderColor: 'transparent', backgroundColor: 'rgba(239,68,68,0.14)', borderWidth: 0, pointRadius: 0, fill: '+1', tension: 0, order: 12, _type: 'band' },
            { label: 'Cash ops floor', data: pts(T, coL), borderColor: RED + '60', backgroundColor: 'transparent', borderWidth: 1, borderDash: [4,3], pointRadius: 0, fill: false, tension: 0, order: 12, _type: 'band' },
            { label: 'High fuel LMP', data: pts(T, N.fuelHigh), borderColor: RED, backgroundColor: RED + '18', borderWidth: 2.5, pointRadius: 3, pointBackgroundColor: RED, fill: false, tension: 0.35, order: 3 },
            { label: 'Medium fuel LMP', data: pts(T, N.fuelMedium), borderColor: AMBER, backgroundColor: AMBER + '18', borderWidth: 2.5, pointRadius: 3, pointBackgroundColor: AMBER, fill: false, tension: 0.35, order: 2 },
            { label: 'Low fuel LMP', data: pts(T, N.fuelLow), borderColor: GREEN, backgroundColor: GREEN + '18', borderWidth: 2.5, pointRadius: 3, pointBackgroundColor: GREEN, fill: false, tension: 0.35, order: 1 }
        ] },
        options: {
            responsive: true, maintainAspectRatio: false,
            animation: { duration: 1200, easing: 'easeOutQuart' },
            scales: baseScales('Average Wholesale Price ($/MWh)', -12, 55),
            plugins: { legend: { display: false }, tooltip: baseTooltip(),
                annotation: { annotations: Object.assign({
                    zeroLine: { type: 'line', yMin: 0, yMax: 0, borderColor: 'rgba(0,0,0,0.1)', borderWidth: 1, borderDash: [6, 4] },
                    cashLabel: { type: 'label', xValue: 100, yValue: 27.5, content: ['Cash ops', '$25\u201330'], color: 'rgba(239,68,68,0.65)', font: { size: 9, weight: '600' }, position: 'start' },
                    allInLabel: { type: 'label', xValue: 100, yValue: 37.5, content: ['All-in', '$35\u201340'], color: 'rgba(239,68,68,0.55)', font: { size: 9, weight: '600' }, position: 'start' },
                    viableLabel: { type: 'label', xValue: 100, yValue: 43, content: ['+ 10% margin', '$38\u201344'], color: 'rgba(245,158,11,0.6)', font: { size: 9, weight: '600' }, position: 'start' },
                    ptcFloor: { type: 'line', yMin: 40, yMax: 40, borderColor: 'rgba(96,165,250,0.45)', borderWidth: 2, borderDash: [8,4],
                        label: { display: true, content: '45U effective floor $40 (thru 2032)', position: 'start', color: 'rgba(96,165,250,0.7)', font: { size: 9, weight: '600' }, backgroundColor: 'rgba(255,255,255,0.9)', padding: 3 } }
                }, sbtiAnnotations()) } }
        }
    });
}

// ===== 6. NUCLEAR ZERO-HOURS CHART =====
function buildNuclearZeroChart(D) {
    var ctx = document.getElementById('nuclearZeroHoursChart').getContext('2d');
    var T = D.thresholds, N = D.nuclear;
    chartInstances.nucZero = new Chart(ctx, {
        type: 'line',
        data: { datasets: [
            { label: 'Off-peak avg LMP', data: pts(T, N.offpeakP50), borderColor: PURPLE, backgroundColor: PURPLE + '25', borderWidth: 2.5, pointRadius: 3, pointBackgroundColor: PURPLE, fill: false, tension: 0.35, yAxisID: 'y', order: 2 },
            { label: 'Zero/negative price hours', data: pts(T, N.zeroP50), borderColor: BLUE, backgroundColor: BLUE + '14', borderWidth: 2.5, pointRadius: 3, pointBackgroundColor: BLUE, fill: true, tension: 0.35, yAxisID: 'y1', order: 3, _unit: 'hrs' }
        ] },
        options: {
            responsive: true, maintainAspectRatio: false,
            animation: { duration: 1000, easing: 'easeOutQuart' },
            scales: {
                x: baseScales('', 0, 0).x,
                y: { position: 'left', title: { display: true, text: 'Off-Peak LMP ($/MWh)', color: PURPLE + '99', font: { size: 11 } }, grid: { color: 'rgba(0,0,0,0.025)' }, ticks: { color: PURPLE + '99', font: { size: 10 } }, min: -15, max: 42 },
                y1: { position: 'right', title: { display: true, text: 'Zero/Negative Price Hours', color: BLUE + '99', font: { size: 11 } }, grid: { drawOnChartArea: false }, ticks: { color: BLUE + '99', font: { size: 10 }, callback: function(v) { return v >= 1000 ? (v/1000).toFixed(0) + 'k' : v; } }, min: 0, max: 8500 }
            },
            plugins: { legend: { display: false },
                tooltip: { backgroundColor: 'rgba(255,255,255,0.98)', titleColor: '#1E293B', bodyColor: 'rgba(15,23,42,0.75)', borderColor: 'rgba(74,144,217,0.18)', borderWidth: 1, padding: 12, cornerRadius: 8,
                    callbacks: { title: function(items) { return items[0] ? items[0].raw.x + '% Clean Energy' : ''; },
                        label: function(item) { return item.dataset._unit === 'hrs' ? item.dataset.label + ': ' + Math.round(item.raw.y) + ' hours' : item.dataset.label + ': $' + item.raw.y.toFixed(1) + '/MWh'; } } },
                annotation: { annotations: Object.assign({
                    nucFloor: { type: 'line', yMin: 25, yMax: 25, yScaleID: 'y', borderColor: 'rgba(239,68,68,0.3)', borderWidth: 1, borderDash: [4,3], label: { display: true, content: 'Cash ops floor $25', position: 'end', color: 'rgba(239,68,68,0.5)', font: { size: 9 }, backgroundColor: 'transparent' } },
                    viableFloor: { type: 'line', yMin: 38, yMax: 38, yScaleID: 'y', borderColor: 'rgba(245,158,11,0.4)', borderWidth: 1.5, borderDash: [6,3], label: { display: true, content: 'Viable floor $38', position: 'start', color: 'rgba(245,158,11,0.6)', font: { size: 9 }, backgroundColor: 'transparent' } },
                    halfYear: { type: 'line', yMin: 4380, yMax: 4380, yScaleID: 'y1', borderColor: 'rgba(0,0,0,0.08)', borderWidth: 1, borderDash: [4,4], label: { display: true, content: '50% of year', position: 'start', color: 'rgba(15,23,42,0.35)', font: { size: 9 }, backgroundColor: 'transparent' } }
                }, sbtiAnnotations()) } }
        }
    });
}

// ===== BUILD ALL CHARTS =====
function buildAllCharts(D, accent) {
    destroyCharts();
    buildHeroChart(D, accent);
    buildFuelChart(D);
    buildRegimeChart(D);
    buildPeakChart(D);
    buildNuclearEconChart(D);
    buildNuclearZeroChart(D);
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

    // Nuclear callout
    el = document.getElementById('nuclearCalloutContent');
    if (el) el.innerHTML = '<h4>' + m.nuclearCalloutH4 + '</h4><p>' + m.nuclearCalloutP + '</p>';

    // Scroll sections
    el = document.getElementById('sect1Narrative'); if (el) el.innerHTML = m.sect1HTML;
    el = document.getElementById('sect2Narrative'); if (el) el.innerHTML = m.sect2HTML;
    el = document.getElementById('sect3Narrative'); if (el) el.innerHTML = m.sect3HTML;

    // Nuclear section header
    el = document.getElementById('nuclearSectionHeader');
    if (el) el.innerHTML = '<span class="section-tag" style="background:rgba(239,68,68,0.06);color:#f87171;">' + m.nuclearHeaderTag + '</span>' +
        '<h2>' + m.nuclearHeaderTitle + '</h2><p>' + m.nuclearHeaderBody + '</p>';

    // Nuclear chart titles
    el = document.getElementById('nuclearChartTitle1'); if (el) el.textContent = m.nuclearChartTitle1;
    el = document.getElementById('nuclearChartTitle2'); if (el) el.textContent = m.nuclearChartTitle2;

    // Nuclear narrative
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
    document.querySelectorAll('.narrative-card, .synthesis-card, .nuclear-paradox').forEach(function(el) {
        el.classList.remove('visible');
        fadeObs.observe(el);
    });
    // Check if any are already visible in viewport
    setTimeout(function() {
        document.querySelectorAll('.narrative-card, .synthesis-card, .nuclear-paradox').forEach(function(el) {
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

// ===== PILLS HANDLER =====
document.querySelectorAll('.region-pill').forEach(function(pill) {
    pill.addEventListener('click', function() {
        document.querySelector('.region-pill.active').classList.remove('active');
        this.classList.add('active');
        switchRegion(this.dataset.iso);
    });
});

// ===== INIT =====
var initAccent = isoAccent('PJM');
buildAllCharts(LMP_ALL.PJM, initAccent);
applyISOAccent('PJM');
observeScrollElements();
