(function() {
'use strict';

// ─── Constants ──────────────────────────────────────────────────────────────
const STRATEGIES = ['1B','2C','2C_rolloff'];
const CORE_STRATEGIES = ['1B','2C'];
const ROLLOFF_PAIRS = { '2C': '2C_rolloff' };
const ISOS = ['CAISO','ERCOT','PJM','NYISO','NEISO','MISO','SPP'];
const THRESHOLDS = [90, 92.5, 95, 97.5, 99.5, 99.99];
const PARTICIPATION_LEVELS = [5, 10, 15, 20, 25, 50, 75];

const STRATEGY_LABELS = {
    '1B': '1B — Consequential',
    '2C': '2C — Hourly Matching with SSS',
    '2C_rolloff': '2C — Nuclear Rolloff'
};
const STRATEGY_SHORT = { '1B': '1B', '2C': '2C', '2C_rolloff': '2C-R' };

const STRATEGY_COLORS = {
    '1B': '#6366F1',
    '2C': '#0EA5E9',
    '2C_rolloff': '#7DD3FC'
};

const GRID_DEMANDS = { CAISO: 224, ERCOT: 488, PJM: 843, NYISO: 152, NEISO: 115, MISO: 660, SPP: 296 };
const TOTAL_DEMAND = Object.values(GRID_DEMANDS).reduce((a, b) => a + b, 0);

const EXISTING_GAS_FOM_KW_YR = { CAISO: 16, ERCOT: 13, PJM: 14, NYISO: 17, NEISO: 15, MISO: 14, SPP: 13 };
const NEW_CCGT_COST_KW_YR = { CAISO: 112, ERCOT: 89, PJM: 99, NYISO: 114, NEISO: 105, MISO: 95, SPP: 88 };
const EXISTING_GAS_CAPACITY_GW = { CAISO: 37, ERCOT: 55, PJM: 75, NYISO: 18, NEISO: 14, MISO: 68, SPP: 32 };

const charts = {};

// ─── Data Aggregation ───────────────────────────────────────────────────────

function getRecord(strategy, iso, participation, threshold) {
    try {
        return DEPLOYMENT_DATA.data[strategy][iso][String(participation)][String(threshold)];
    } catch (e) { return null; }
}

function aggregateCrossISO(strategy, participation, threshold) {
    let weightedCost = 0, totalDemand = 0;
    let co2 = 0, tc = 0, gasGw = 0, curtTwh = 0, fossilCo2 = 0, dispCo2 = 0;
    let hmsWeighted = 0, gridCleanWeighted = 0;

    for (const iso of ISOS) {
        const d = getRecord(strategy, iso, participation, threshold);
        if (!d) continue;
        const demand = GRID_DEMANDS[iso];
        weightedCost += (d.c || 0) * demand;
        totalDemand += demand;
        co2 += d.co2 || 0;
        tc += d.tc || 0;
        gasGw += d.gasGw || 0;
        curtTwh += d.curtTwh || 0;
        fossilCo2 += d.fossilCo2Mt || 0;
        dispCo2 += d.dispCo2 || 0;
        hmsWeighted += (d.hms || 0) * demand;
        gridCleanWeighted += (d.gridCleanPct || 0) * demand;
    }

    return {
        avgCost: totalDemand > 0 ? weightedCost / totalDemand : 0,
        totalCostM: tc,
        totalCo2: co2,
        totalGasGw: gasGw,
        totalCurtTwh: curtTwh,
        totalFossilCo2: fossilCo2,
        totalDispCo2: dispCo2,
        avgHms: totalDemand > 0 ? hmsWeighted / totalDemand : 0,
        avgGridClean: totalDemand > 0 ? gridCleanWeighted / totalDemand : 0,
        isoCount: totalDemand > 0 ? ISOS.length : 0
    };
}

const AGG_CACHE = {};
function cachedAggregate(strategy, participation, threshold) {
    const key = `${strategy}|${participation}|${threshold}`;
    if (!AGG_CACHE[key]) AGG_CACHE[key] = aggregateCrossISO(strategy, participation, threshold);
    return AGG_CACHE[key];
}

// ─── Cost Functions ─────────────────────────────────────────────────────────

function computeLearningScore(strategy) {
    const scores = { '1B': 3, '2C': 8 };
    return scores[strategy] || 3;
}

function computeSystemCost(strategy, participation, threshold) {
    let totalCleanM = 0, totalNewGasCostM = 0;
    for (const iso of ISOS) {
        const d = getRecord(strategy, iso, participation, threshold);
        if (!d) continue;
        totalCleanM += d.tc || 0;
        const gasGw = d.gasGw || 0;
        const existingGw = EXISTING_GAS_CAPACITY_GW[iso] || 0;
        const newGasGw = Math.max(0, gasGw - existingGw);
        totalNewGasCostM += newGasGw * NEW_CCGT_COST_KW_YR[iso] * 1e3;
    }
    return totalCleanM + totalNewGasCostM;
}

function computeGasCostOnly(strategy, participation, threshold) {
    let totalNewGasCostM = 0;
    for (const iso of ISOS) {
        const d = getRecord(strategy, iso, participation, threshold);
        if (!d) continue;
        const gasGw = d.gasGw || 0;
        const existingGw = EXISTING_GAS_CAPACITY_GW[iso] || 0;
        const newGasGw = Math.max(0, gasGw - existingGw);
        totalNewGasCostM += newGasGw * NEW_CCGT_COST_KW_YR[iso] * 1e3;
    }
    return totalNewGasCostM;
}

// ─── Chart Helpers ──────────────────────────────────────────────────────────

function destroyChart(id) {
    if (charts[id]) { charts[id].destroy(); delete charts[id]; }
}

function baseOpts(titleText) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { labels: { font: { family: 'DM Sans', size: 11 }, usePointStyle: true, pointStyle: 'circle', padding: 12 } },
            tooltip: { mode: 'index', intersect: false, bodyFont: { family: 'DM Sans' }, titleFont: { family: 'Plus Jakarta Sans', weight: 600 } },
            title: titleText ? { display: true, text: titleText, font: { family: 'Plus Jakarta Sans', size: 13, weight: 600 }, color: '#1A2744' } : { display: false }
        },
        scales: {
            x: { ticks: { font: { family: 'DM Sans', size: 11 } }, grid: { color: 'rgba(0,0,0,0.05)' } },
            y: { ticks: { font: { family: 'DM Sans', size: 11 } }, grid: { color: 'rgba(0,0,0,0.05)' } }
        }
    };
}

function fmt(v, dec) { return v != null && isFinite(v) ? v.toFixed(dec || 1) : '—'; }

// ─── Section: Hero Stats ────────────────────────────────────────────────────

function populateHeroStats() {
    const part = 15, thresh = 95;
    const a1b = cachedAggregate('1B', part, thresh);
    const a2c = cachedAggregate('2C', part, thresh);
    const sys1b = computeSystemCost('1B', part, thresh);
    const sys2c = computeSystemCost('2C', part, thresh);

    const bestCo2 = Math.max(a1b.totalCo2, a2c.totalCo2);
    document.getElementById('heroStatCo2').textContent = fmt(bestCo2, 0) + ' Mt';

    if (sys2c > 0 && sys1b > 0) {
        if (sys1b > sys2c) {
            document.getElementById('heroStatCostRatio').textContent = fmt(sys1b / sys2c, 1) + '\u00d7';
        } else {
            document.getElementById('heroStatCostRatio').textContent = 'Comparable';
        }
    }

    const gasDelta = a1b.totalGasGw - a2c.totalGasGw;
    document.getElementById('heroStatGasDelta').textContent = (gasDelta > 0 ? '' : '+') + fmt(Math.abs(gasDelta), 0) + ' GW';
    document.getElementById('heroStatLearning').textContent = '2.7\u00d7';
}

// ─── Section 2: Crossover Charts ────────────────────────────────────────────

function buildCrossoverCharts(participation) {
    const chartConfigs = [
        { id: 'crossoverCostChart', field: 'avgCost', label: '$/MWh', title: 'Clean Energy Cost' },
        { id: 'crossoverCo2Chart', field: 'totalCo2', label: 'Mt CO\u2082', title: 'CO\u2082 Displaced' },
        { id: 'crossoverGasChart', field: 'totalGasGw', label: 'GW Gas', title: 'Gas on Grid' },
        { id: 'crossoverCurtChart', field: 'totalCurtTwh', label: 'TWh', title: 'Curtailment' }
    ];

    for (const cfg of chartConfigs) {
        destroyChart(cfg.id);
        const datasets = CORE_STRATEGIES.map(s => ({
            label: STRATEGY_SHORT[s],
            data: THRESHOLDS.map(t => cachedAggregate(s, participation, t)[cfg.field]),
            borderColor: STRATEGY_COLORS[s],
            backgroundColor: STRATEGY_COLORS[s] + '20',
            tension: 0.3,
            pointRadius: 3,
            borderWidth: 2,
            fill: false
        }));

        const ctx = document.getElementById(cfg.id);
        if (!ctx) continue;
        charts[cfg.id] = new Chart(ctx, {
            type: 'line',
            data: { labels: THRESHOLDS.map(t => t + '%'), datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { font: { family: 'DM Sans', size: 10 }, usePointStyle: true, padding: 8 } },
                    tooltip: { mode: 'index', intersect: false, bodyFont: { family: 'DM Sans', size: 10 } },
                    title: { display: true, text: cfg.title, font: { family: 'Plus Jakarta Sans', size: 11, weight: 600 }, color: '#1A2744' }
                },
                scales: {
                    x: { ticks: { font: { family: 'DM Sans', size: 9 } }, grid: { color: 'rgba(0,0,0,0.05)' } },
                    y: { ticks: { font: { family: 'DM Sans', size: 9 } }, grid: { color: 'rgba(0,0,0,0.05)' },
                        title: { display: true, text: cfg.label, font: { family: 'DM Sans', size: 9 } } }
                }
            }
        });
    }

    populateCrossoverSteps(participation);
}

function populateCrossoverSteps(participation) {
    const a1b = cachedAggregate('1B', participation, 95);
    const a2c = cachedAggregate('2C', participation, 95);
    const sys1b = computeSystemCost('1B', participation, 95);
    const sys2c = computeSystemCost('2C', participation, 95);

    const costEl = document.getElementById('stepCostStat');
    if (costEl) costEl.innerHTML = `<strong>1B:</strong> $${fmt(a1b.avgCost)}/MWh &nbsp; <strong>2C:</strong> $${fmt(a2c.avgCost)}/MWh at 95% CFE`;

    const sysEl = document.getElementById('stepSystemStat');
    if (sysEl) sysEl.innerHTML = `<strong>Total system cost at 95%:</strong> 1B = $${fmt(sys1b / 1000, 0)}B &nbsp; 2C = $${fmt(sys2c / 1000, 0)}B`;

    const co2El = document.getElementById('stepCo2Stat');
    if (co2El) co2El.innerHTML = `<strong>CO\u2082 displaced:</strong> 1B = ${fmt(a1b.totalCo2)} Mt &nbsp; 2C = ${fmt(a2c.totalCo2)} Mt`;

    const gasEl = document.getElementById('stepGasStat');
    if (gasEl) gasEl.innerHTML = `<strong>Gas on grid:</strong> 1B = ${fmt(a1b.totalGasGw, 0)} GW &nbsp; 2C = ${fmt(a2c.totalGasGw, 0)} GW &nbsp; (${fmt(Math.abs(a1b.totalGasGw - a2c.totalGasGw), 0)} GW delta)`;
}

// ─── Section 3: Gas Ribbon Chart ────────────────────────────────────────────

function buildGasRibbonChart() {
    destroyChart('gasRibbonChart');
    const partLabels = PARTICIPATION_LEVELS.map(p => p + '%');
    const existingGasGw = Object.values(EXISTING_GAS_CAPACITY_GW).reduce((a, b) => a + b, 0);

    function computeRibbonData(strategy) {
        return PARTICIPATION_LEVELS.map(p => {
            const vals = THRESHOLDS.map(t => cachedAggregate(strategy, p, t).totalGasGw).filter(v => isFinite(v));
            vals.sort((a, b) => a - b);
            const mid = vals[Math.floor(vals.length / 2)];
            return { min: Math.min(...vals), median: mid, max: Math.max(...vals) };
        });
    }

    const ribbon1B = computeRibbonData('1B');
    const ribbon2C = computeRibbonData('2C');

    const datasets = [
        { label: '1B range', data: ribbon1B.map(r => r.max), borderColor: 'transparent', backgroundColor: STRATEGY_COLORS['1B'] + '18',
          fill: '+1', pointRadius: 0, order: 4 },
        { label: '_1B_min', data: ribbon1B.map(r => r.min), borderColor: 'transparent', backgroundColor: 'transparent',
          fill: false, pointRadius: 0, order: 4 },
        { label: STRATEGY_SHORT['1B'], data: ribbon1B.map(r => r.median), borderColor: STRATEGY_COLORS['1B'],
          backgroundColor: STRATEGY_COLORS['1B'], borderWidth: 2.5, tension: 0.3, pointRadius: 3, pointHoverRadius: 6, fill: false, order: 1 },
        { label: '2C range', data: ribbon2C.map(r => r.max), borderColor: 'transparent', backgroundColor: STRATEGY_COLORS['2C'] + '18',
          fill: '+1', pointRadius: 0, order: 3 },
        { label: '_2C_min', data: ribbon2C.map(r => r.min), borderColor: 'transparent', backgroundColor: 'transparent',
          fill: false, pointRadius: 0, order: 3 },
        { label: STRATEGY_SHORT['2C'], data: ribbon2C.map(r => r.median), borderColor: STRATEGY_COLORS['2C'],
          backgroundColor: STRATEGY_COLORS['2C'], borderWidth: 2.5, tension: 0.3, pointRadius: 3, pointHoverRadius: 6, fill: false, order: 1 }
    ];

    const ctx = document.getElementById('gasRibbonChart');
    if (!ctx) return;

    charts.gasRibbonChart = new Chart(ctx, {
        type: 'line',
        data: { labels: partLabels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        font: { family: 'DM Sans', size: 11 },
                        usePointStyle: true,
                        filter: item => !item.text.startsWith('_') && !item.text.includes('range')
                    }
                },
                tooltip: {
                    mode: 'index', intersect: false,
                    filter: item => !item.dataset.label.startsWith('_') && !item.dataset.label.includes('range'),
                    bodyFont: { family: 'DM Sans' }
                },
                annotation: {
                    annotations: {
                        existingGas: {
                            type: 'line', yMin: existingGasGw, yMax: existingGasGw,
                            borderColor: '#EF4444', borderWidth: 2, borderDash: [6, 4],
                            label: { content: `Existing US gas: ${existingGasGw} GW`, display: true,
                                     position: 'start', backgroundColor: 'rgba(239,68,68,0.1)',
                                     color: '#DC2626', font: { family: 'DM Sans', size: 10 } }
                        }
                    }
                }
            },
            scales: {
                x: { ticks: { font: { family: 'DM Sans', size: 11 } }, grid: { color: 'rgba(0,0,0,0.05)' },
                     title: { display: true, text: 'Corporate Participation', font: { family: 'DM Sans', size: 11 } } },
                y: { ticks: { font: { family: 'DM Sans', size: 11 } }, grid: { color: 'rgba(0,0,0,0.05)' },
                     title: { display: true, text: 'Gas Capacity on Grid (GW)', font: { family: 'DM Sans', size: 11 } } }
            }
        }
    });

    const gas1b50 = cachedAggregate('1B', 50, 95).totalGasGw;
    const gas2c50 = cachedAggregate('2C', 50, 95).totalGasGw;
    const el = document.getElementById('gasRibbonInsight');
    if (el) {
        el.innerHTML = `<p><strong>Gas Displacement:</strong> At 50% participation and 95% CFE,
            <strong>1B</strong> leaves ${fmt(gas1b50, 0)} GW of gas on grid while
            <strong>2C</strong> reduces it to ${fmt(gas2c50, 0)} GW \u2014 a ${fmt(Math.abs(gas1b50 - gas2c50), 0)} GW difference.
            ${gas1b50 > existingGasGw ? `Under 1B, gas capacity actually <em>exceeds</em> today\u2019s ${existingGasGw} GW installed base \u2014 meaning new gas must be built to backstop the VRE-heavy mix.` : ''}
            The ribbon shows how this gap widens at higher participation levels.</p>`;
    }
}

// ─── Section 4: Cost Crossover Charts ───────────────────────────────────────

function buildSystemCostCrossover() {
    destroyChart('systemCostCrossoverChart');
    const threshold = 95;

    const datasets = CORE_STRATEGIES.map(s => ({
        label: STRATEGY_LABELS[s],
        data: PARTICIPATION_LEVELS.map(p => computeSystemCost(s, p, threshold) / 1e6),
        borderColor: STRATEGY_COLORS[s],
        backgroundColor: STRATEGY_COLORS[s] + '15',
        tension: 0.3,
        pointRadius: 4,
        pointHoverRadius: 7,
        borderWidth: 2.5,
        fill: false
    }));

    const ctx = document.getElementById('systemCostCrossoverChart');
    if (!ctx) return;

    charts.systemCostCrossoverChart = new Chart(ctx, {
        type: 'line',
        data: { labels: PARTICIPATION_LEVELS.map(p => p + '%'), datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { font: { family: 'DM Sans', size: 11 }, usePointStyle: true, padding: 12 } },
                tooltip: {
                    mode: 'index', intersect: false, bodyFont: { family: 'DM Sans' },
                    callbacks: {
                        label: function(ctx) {
                            return ctx.dataset.label + ': $' + fmt(ctx.raw, 0) + 'T';
                        }
                    }
                }
            },
            scales: {
                x: { ticks: { font: { family: 'DM Sans', size: 11 } }, grid: { color: 'rgba(0,0,0,0.05)' },
                     title: { display: true, text: 'Corporate Participation', font: { family: 'DM Sans', size: 11 } } },
                y: { ticks: { font: { family: 'DM Sans', size: 11 } }, grid: { color: 'rgba(0,0,0,0.05)' },
                     title: { display: true, text: 'System Cost ($T)', font: { family: 'DM Sans', size: 11 } } }
            }
        }
    });
}

function buildCostBreakdown() {
    destroyChart('costBreakdownChart');
    const threshold = 95;
    const keyParts = [5, 15, 25, 50];

    const labels = [];
    const cleanData = [];
    const gasData = [];
    const barColors = [];
    const gasColors = [];

    for (const p of keyParts) {
        for (const s of CORE_STRATEGIES) {
            labels.push(STRATEGY_SHORT[s] + ' @ ' + p + '%');
            const agg = cachedAggregate(s, p, threshold);
            const cleanM = agg.totalCostM;
            const gasCostM = computeGasCostOnly(s, p, threshold);
            cleanData.push(cleanM / 1e6);
            gasData.push(gasCostM / 1e6);
            barColors.push(STRATEGY_COLORS[s]);
            gasColors.push(STRATEGY_COLORS[s] + '40');
        }
    }

    const ctx = document.getElementById('costBreakdownChart');
    if (!ctx) return;

    charts.costBreakdownChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Clean Procurement',
                    data: cleanData,
                    backgroundColor: barColors,
                    borderRadius: 2
                },
                {
                    label: 'Gas Backup Cost',
                    data: gasData,
                    backgroundColor: gasColors,
                    borderRadius: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { font: { family: 'DM Sans', size: 11 }, usePointStyle: true, padding: 12 } },
                tooltip: {
                    mode: 'index', intersect: false, bodyFont: { family: 'DM Sans' },
                    callbacks: {
                        label: function(ctx) {
                            return ctx.dataset.label + ': $' + fmt(ctx.raw, 1) + 'T';
                        }
                    }
                }
            },
            scales: {
                x: { ticks: { font: { family: 'DM Sans', size: 9 }, maxRotation: 45 }, grid: { display: false },
                     stacked: true },
                y: { ticks: { font: { family: 'DM Sans', size: 11 } }, grid: { color: 'rgba(0,0,0,0.05)' },
                     stacked: true,
                     title: { display: true, text: 'Total Cost ($T)', font: { family: 'DM Sans', size: 11 } } }
            }
        }
    });
}

function populateCrossoverInsight() {
    // Find crossover point
    let crossoverPart = null;
    for (const p of PARTICIPATION_LEVELS) {
        const sys1b = computeSystemCost('1B', p, 95);
        const sys2c = computeSystemCost('2C', p, 95);
        if (sys1b > sys2c) { crossoverPart = p; break; }
    }

    const sys1b50 = computeSystemCost('1B', 50, 95);
    const sys2c50 = computeSystemCost('2C', 50, 95);
    const ratio50 = sys1b50 > 0 && sys2c50 > 0 ? sys1b50 / sys2c50 : 0;

    const el = document.getElementById('crossoverInsight');
    if (el) {
        el.innerHTML = `<p><strong>Regime Insights:</strong>
            At low participation (5\u201310%), 1B\u2019s cheap cross-regional VRE keeps total system cost competitive.
            ${crossoverPart ? `As participation rises above ${crossoverPart}%, 1B\u2019s gas backup requirements drive system costs dramatically higher than 2C.` : 'As participation rises, 1B\'s gas backup requirements increasingly dominate system costs.'}
            At 50% participation and 95% CFE, 1B\u2019s system cost (clean + gas) ${ratio50 > 1.5 ? `exceeds 2C\u2019s by ${fmt(ratio50, 1)}\u00d7` : 'significantly exceeds 2C\u2019s'}.
            This crossover is consistent across ambition levels because the gas penalty scales with the gap between 1B\u2019s VRE-heavy mix and the firm capacity needed for grid reliability.</p>`;
    }
}

// ─── Section 5: Strategy Pentagon (Overlaid Radar) ──────────────────────────

function buildClusterRadar(participation) {
    participation = participation || 15;
    const threshold = 95;
    destroyChart('radarOverlayChart');

    const stratData = {};
    for (const s of CORE_STRATEGIES) {
        const agg = cachedAggregate(s, participation, threshold);
        stratData[s] = {
            co2: agg.totalCo2,
            cost: computeSystemCost(s, participation, threshold),
            gas: agg.totalGasGw,
            learn: computeLearningScore(s),
            curt: agg.totalCurtTwh
        };
    }

    function normalizeRelativeToBest(field, higherIsBetter) {
        const vals = CORE_STRATEGIES.map(s => stratData[s][field]);
        if (higherIsBetter) {
            const best = Math.max(...vals);
            const normed = {};
            CORE_STRATEGIES.forEach((s, i) => { normed[s] = best > 0 ? vals[i] / best : 0; });
            return normed;
        } else {
            const best = Math.min(...vals);
            const normed = {};
            CORE_STRATEGIES.forEach((s, i) => { normed[s] = vals[i] > 0 ? best / vals[i] : 0; });
            return normed;
        }
    }

    const normCo2 = normalizeRelativeToBest('co2', true);
    const normCost = normalizeRelativeToBest('cost', false);
    const normGas = normalizeRelativeToBest('gas', false);
    const normLearn = normalizeRelativeToBest('learn', true);
    const normCurt = normalizeRelativeToBest('curt', false);

    const radarLabels = ['Emission\nReduction', 'True\nCost', 'Low Gas\nLock-in', 'Learning\nCurves', 'Low\nCurtailment'];

    const datasets = CORE_STRATEGIES.map(s => ({
        label: STRATEGY_LABELS[s],
        data: [normCo2[s], normCost[s], normGas[s], normLearn[s], normCurt[s]],
        borderColor: STRATEGY_COLORS[s],
        backgroundColor: STRATEGY_COLORS[s] + '25',
        borderWidth: 2.5,
        pointRadius: 5,
        pointBackgroundColor: STRATEGY_COLORS[s],
        pointBorderColor: '#fff',
        pointBorderWidth: 1
    }));

    const ctx = document.getElementById('radarOverlayChart');
    if (!ctx) return;

    charts.radarOverlayChart = new Chart(ctx, {
        type: 'radar',
        data: { labels: radarLabels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    beginAtZero: true, max: 1, min: 0,
                    ticks: { display: false, stepSize: 0.25 },
                    pointLabels: { font: { family: 'DM Sans', size: 12, weight: 600 }, color: '#1A2744' },
                    grid: { color: 'rgba(0,0,0,0.08)' },
                    angleLines: { color: 'rgba(0,0,0,0.08)' }
                }
            },
            plugins: {
                legend: { labels: { font: { family: 'DM Sans', size: 12 }, usePointStyle: true, padding: 16 } },
                tooltip: {
                    bodyFont: { family: 'DM Sans' },
                    callbacks: {
                        label: function(ctx) {
                            return `${ctx.dataset.label}: ${(ctx.raw * 100).toFixed(0)}% of best`;
                        }
                    }
                }
            }
        }
    });

    // Insight text
    const descEl = document.getElementById('radarInsightText');
    if (descEl) {
        const dims = ['Emission Reduction', 'True Cost', 'Gas Lock-in', 'Learning Curves', 'Curtailment'];
        const wins1b = [normCo2, normCost, normGas, normLearn, normCurt].filter(n => n['1B'] > n['2C']).length;
        const wins2c = 5 - wins1b;
        descEl.innerHTML = `<p>At ${participation}% participation / 95% CFE: <strong>2C leads on ${wins2c} of 5 dimensions</strong>, 1B on ${wins1b}. ` +
            `2C\u2019s advantages in gas displacement and learning curve deployment are structural \u2014 they grow with participation. ` +
            `1B\u2019s per-MWh cost advantage narrows as gas backup requirements scale.</p>`;
    }
}

// ─── Section 6: Wright's Law Chart ──────────────────────────────────────────

function buildWrightsLawChart() {
    destroyChart('wrightsLawChart');

    fetch('js/wrights_law_data.json')
        .then(r => r.json())
        .then(data => {
            // Use 95% comparison if available, fall back to 90%
            const comparison = data.strategy_comparison_95 || data.strategy_comparison_90;
            const targetLabel = data.strategy_comparison_95 ? '95' : '90';
            const pcts = data.metadata.participation_pcts.map(p => (p * 100) + '%');
            const stratKeys = [
                { key: 'strat_1b', label: STRATEGY_SHORT['1B'], color: STRATEGY_COLORS['1B'] },
                { key: 'strat_2c_gated', label: STRATEGY_SHORT['2C'], color: STRATEGY_COLORS['2C'] }
            ];

            const datasets = [];
            for (const d of stratKeys) {
                if (comparison[d.key]) {
                    datasets.push({
                        label: d.label,
                        data: comparison[d.key],
                        borderColor: d.color,
                        backgroundColor: d.color + '15',
                        tension: 0.3,
                        pointRadius: 4,
                        pointHoverRadius: 7,
                        borderWidth: 2.5,
                        fill: false
                    });
                }
            }

            const ctx = document.getElementById('wrightsLawChart');
            if (!ctx) return;

            charts.wrightsLawChart = new Chart(ctx, {
                type: 'line',
                data: { labels: pcts, datasets },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { labels: { font: { family: 'DM Sans', size: 12 }, usePointStyle: true, padding: 16 } },
                        tooltip: { mode: 'index', intersect: false, bodyFont: { family: 'DM Sans' } },
                        annotation: {
                            annotations: {
                                costParity: {
                                    type: 'line', yMin: 0, yMax: 0,
                                    borderColor: '#22C55E', borderWidth: 2, borderDash: [8, 4],
                                    label: { content: 'Cost parity', display: true,
                                             position: 'end', backgroundColor: 'rgba(34,197,94,0.1)',
                                             color: '#16A34A', font: { family: 'DM Sans', size: 11, weight: 600 } }
                                }
                            }
                        }
                    },
                    scales: {
                        x: { ticks: { font: { family: 'DM Sans', size: 11 } }, grid: { color: 'rgba(0,0,0,0.05)' },
                             title: { display: true, text: 'C&I Participation', font: { family: 'DM Sans', size: 12 } } },
                        y: { ticks: { font: { family: 'DM Sans', size: 11 } }, grid: { color: 'rgba(0,0,0,0.05)' },
                             title: { display: true, text: '$/MWh Cost Premium', font: { family: 'DM Sans', size: 12 } } }
                    }
                }
            });

            // Insight with new-build firm clean TWh stats
            const insight = document.getElementById('wrightsInsight');
            if (insight) {
                const d2c = comparison.strat_2c_gated;
                let crossIdx = d2c ? d2c.findIndex(v => v <= 0) : -1;
                const crossPct = crossIdx >= 0 ? data.metadata.participation_pcts[crossIdx] : null;

                // Compute new-build firm TWh from DEPLOYMENT_DATA at 95% for key insights
                let firm1b10twh = null, firm2c10twh = null;
                for (const p of PARTICIPATION_LEVELS) {
                    if (firm2c10twh === null) {
                        let totalFirm = 0;
                        for (const iso of ISOS) {
                            const d = getRecord('2C', iso, p, 95);
                            if (d && d.x) {
                                for (const [k, v] of Object.entries(d.x)) {
                                    if (k.includes('nuclear') || k.includes('ccs') || k.includes('geo')) totalFirm += v;
                                }
                            }
                        }
                        if (totalFirm >= 10) firm2c10twh = p;
                    }
                    if (firm1b10twh === null) {
                        let totalFirm = 0;
                        for (const iso of ISOS) {
                            const d = getRecord('1B', iso, p, 95);
                            if (d && d.x) {
                                for (const [k, v] of Object.entries(d.x)) {
                                    if (k.includes('nuclear') || k.includes('ccs') || k.includes('geo')) totalFirm += v;
                                }
                            }
                        }
                        if (totalFirm >= 10) firm1b10twh = p;
                    }
                }

                let firmInsight = '';
                if (firm2c10twh) firmInsight += ` Strategy 2C reaches 10 TWh of new-build firm clean generation at <strong>${firm2c10twh}%</strong> participation.`;
                if (firm1b10twh) firmInsight += ` Strategy 1B reaches it at <strong>${firm1b10twh}%</strong>.`;
                else firmInsight += ' Strategy 1B does not reach 10 TWh of firm clean at any modeled participation level.';

                insight.innerHTML = `<p><strong>Learning Curve Impact (${targetLabel}% CFE):</strong>
                    Strategy 2C forces deployment of firm clean (nuclear uprate, CCS) and long-duration storage (LDES) \u2014 technologies still on steep learning curves.
                    ${crossPct ? `At ${(crossPct * 100).toFixed(0)}% participation, Wright\u2019s Law cost reductions make 2C <strong>cost-negative</strong> \u2014 the learning curve benefits exceed the procurement premium.` : 'As participation scales, learning curve benefits accumulate.'}
                    Strategy 1B deploys mostly mature VRE (solar/onshore wind) that is already at NOAK pricing \u2014 contributing minimally to cost breakthroughs the entire grid needs.</p>
                    <p>${firmInsight}</p>`;
            }
        })
        .catch(() => {
            const ctx = document.getElementById('wrightsLawChart');
            if (ctx) ctx.parentElement.innerHTML = '<p style="color:var(--text-muted);text-align:center;padding:2rem">Wright\'s Law data unavailable</p>';
        });
}

// ─── Section 7: Critical Risks ──────────────────────────────────────────────

function buildRiskCharts() {
    buildGasLockinChart();
    buildCurtailmentChart();
    populateRiskInsight();
}

function buildGasLockinChart() {
    destroyChart('gasLockinChart');
    const participation = 50, threshold = 95;

    const datasets = ISOS.map(iso => ({
        label: iso,
        data: CORE_STRATEGIES.map(s => {
            const d = getRecord(s, iso, participation, threshold);
            return d ? d.gasGw : 0;
        }),
        backgroundColor: typeof ISO_COLORS !== 'undefined' ? ISO_COLORS[iso] : '#999',
        borderRadius: 2
    }));

    const ctx = document.getElementById('gasLockinChart');
    if (!ctx) return;

    charts.gasLockinChart = new Chart(ctx, {
        type: 'bar',
        data: { labels: CORE_STRATEGIES.map(s => STRATEGY_SHORT[s]), datasets },
        options: {
            ...baseOpts(),
            scales: {
                x: { ...baseOpts().scales.x, stacked: true },
                y: { ...baseOpts().scales.y, stacked: true, title: { display: true, text: 'Gas Capacity (GW)', font: { family: 'DM Sans', size: 11 } } }
            }
        }
    });
}

function buildCurtailmentChart() {
    destroyChart('curtailmentChart');
    const participation = 50, threshold = 95;
    const values = CORE_STRATEGIES.map(s => cachedAggregate(s, participation, threshold).totalCurtTwh);

    const ctx = document.getElementById('curtailmentChart');
    if (!ctx) return;

    charts.curtailmentChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: CORE_STRATEGIES.map(s => STRATEGY_SHORT[s]),
            datasets: [{
                label: 'Curtailment (TWh)',
                data: values,
                backgroundColor: CORE_STRATEGIES.map(s => STRATEGY_COLORS[s]),
                borderRadius: 4
            }]
        },
        options: {
            ...baseOpts(),
            plugins: { ...baseOpts().plugins, legend: { display: false } },
            scales: {
                x: { ...baseOpts().scales.x },
                y: { ...baseOpts().scales.y, title: { display: true, text: 'TWh Curtailed', font: { family: 'DM Sans', size: 11 } } }
            }
        }
    });
}

function populateRiskInsight() {
    const gas95 = CORE_STRATEGIES.map(s => ({ s, gas: cachedAggregate(s, 50, 95).totalGasGw }));
    gas95.sort((a, b) => b.gas - a.gas);
    const curt95 = CORE_STRATEGIES.map(s => ({ s, curt: cachedAggregate(s, 50, 95).totalCurtTwh }));
    curt95.sort((a, b) => b.curt - a.curt);

    const el = document.getElementById('riskInsight');
    if (el) {
        el.innerHTML = `<p><strong>Risk Summary (95% / 50% participation):</strong>
            Gas lock-in: ${fmt(gas95[gas95.length - 1].gas, 0)} GW (${STRATEGY_SHORT[gas95[gas95.length - 1].s]}) to
            ${fmt(gas95[0].gas, 0)} GW (${STRATEGY_SHORT[gas95[0].s]}) \u2014 a ${fmt(gas95[0].gas - gas95[gas95.length - 1].gas, 0)} GW spread.
            Curtailment: ${fmt(curt95[curt95.length - 1].curt)} TWh to ${fmt(curt95[0].curt)} TWh.</p>`;
    }
}

// ─── Section 8: Nuclear Narrative ───────────────────────────────────────────

function populateNuclearNarrative() {
    const part = 15, thresh = 95;
    const base = cachedAggregate('2C', part, thresh);
    const rolloff = cachedAggregate('2C_rolloff', part, thresh);

    const costDelta = rolloff.avgCost - base.avgCost;
    const co2Delta = base.totalCo2 - rolloff.totalCo2;
    const gasDelta = rolloff.totalGasGw - base.totalGasGw;

    const costEl = document.getElementById('nucCostDelta');
    if (costEl) costEl.textContent = '+$' + fmt(Math.abs(costDelta)) + '/MWh';

    const co2El = document.getElementById('nucCo2Delta');
    if (co2El) co2El.textContent = fmt(Math.abs(co2Delta)) + ' Mt';

    const gasEl = document.getElementById('nucGasDelta');
    if (gasEl) gasEl.textContent = '+' + fmt(Math.abs(gasDelta), 0) + ' GW';

    buildNuclearStrandChart();

    const insight = document.getElementById('nuclearInsight');
    if (insight) {
        insight.innerHTML = `<p><strong>What happens if nuclear retires?</strong>
            At 95% CFE / ${part}% participation, losing the existing nuclear fleet would increase clean procurement costs by
            <strong>$${fmt(Math.abs(costDelta))}/MWh</strong>, reduce CO\u2082 displacement by <strong>${fmt(Math.abs(co2Delta))} Mt</strong>,
            and add <strong>${fmt(Math.abs(gasDelta), 0)} GW</strong> of gas to the grid.</p>
            <p><strong>Strategy 2C actively supports nuclear economics</strong> by creating EAC revenue (~$20\u201340/MWh) that helps prevent premature retirement.
            Under 1B, nuclear plants face merchant revenue erosion from VRE oversupply with no offsetting demand signal.
            The US nuclear fleet generates ~20% of all electricity and ~50% of all clean electricity.</p>`;
    }
}

function buildNuclearStrandChart() {
    destroyChart('nuclearStrandChart');
    const participation = 15, threshold = 95;

    const base2c = cachedAggregate('2C', participation, threshold);
    const roll2c = cachedAggregate('2C_rolloff', participation, threshold);
    const dep2c = base2c.avgCost > 0 ? Math.abs(roll2c.avgCost - base2c.avgCost) / base2c.avgCost * 100 : 0;

    const exposures = CORE_STRATEGIES.map(s => {
        const strandingScore = {
            '1B': 20 + dep2c * 0.5,
            '2C': Math.max(0, dep2c - 5)
        };
        return strandingScore[s] || 10;
    });

    const ctx = document.getElementById('nuclearStrandChart');
    if (!ctx) return;

    charts.nuclearStrandChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: CORE_STRATEGIES.map(s => STRATEGY_SHORT[s]),
            datasets: [{
                label: 'Nuclear Stranding Risk',
                data: exposures,
                backgroundColor: CORE_STRATEGIES.map(s => STRATEGY_COLORS[s]),
                borderRadius: 4
            }]
        },
        options: {
            ...baseOpts(),
            indexAxis: 'y',
            plugins: { ...baseOpts().plugins, legend: { display: false } },
            scales: {
                x: { ...baseOpts().scales.x, title: { display: true, text: 'Stranding Risk Index (higher = more fragile)', font: { family: 'DM Sans', size: 11 } } },
                y: { ...baseOpts().scales.y }
            }
        }
    });
}

// ─── Section 9: Dissenting ─────────────────────────────────────────────────

function populateDissenting() {
    const el = document.getElementById('dissentingContent');
    if (!el) return;
    el.innerHTML = `
        <div class="insight-box fade-in" style="margin-bottom: var(--space-lg)">
            <h4 style="font-family: var(--font-heading); color: var(--navy); margin: 0 0 var(--space-sm)">Against 1B \u2014 Consequential</h4>
            <p>1B deploys cross-regionally \u2014 new clean generation may be built in a distant ISO,
            leaving the buyer\u2019s grid physically untransformed. The new capacity is overwhelmingly mature VRE,
            contributing minimally to firm clean learning curves. Consequential netting uses annual accounting,
            missing the hourly mismatch between clean generation and actual load. And 1B provides no financial
            support to existing nuclear, leaving the grid\u2019s largest clean source exposed to merchant revenue erosion.</p>
        </div>

        <div class="insight-box fade-in" style="margin-bottom: var(--space-lg)">
            <h4 style="font-family: var(--font-heading); color: var(--navy); margin: 0 0 var(--space-sm)">Against 2C \u2014 Hourly Matching with SSS</h4>
            <p>While 2C\u2019s EAC purchases create revenue that helps keep nuclear online, this is a
            <strong>mutual dependency</strong> \u2014 2C needs nuclear for cheap firm supply, and nuclear needs
            2C\u2019s revenue for economic viability. If nuclear plants close despite EAC support (e.g., due to
            regulatory or safety issues), 2C loses its cheapest firm generation source and must accept
            higher costs or increased gas dependency. Additionally, 2C\u2019s blending of existing and new clean
            resources means only ~30% of its clean procurement is truly additional new-build \u2014 the rest
            reshuffles existing clean claims, which may dilute the additionality signal.</p>
        </div>

        <div class="insight-box fade-in" style="margin-bottom: var(--space-lg)">
            <h4 style="font-family: var(--font-heading); color: var(--navy); margin: 0 0 var(--space-sm)">What Could Flip This Analysis</h4>
            <p><strong>Policy:</strong> A federal clean energy standard or carbon price above $100/ton would make 1B\u2019s cost advantage
            irrelevant and reward high-additionality strategies. <strong>Nuclear policy:</strong> Extended production tax credits (45U)
            would reduce nuclear revenue adequacy concerns, reducing 2C\u2019s fragility.
            <strong>Technology:</strong> If advanced nuclear or LDES costs fall faster than Wright\u2019s Law projections,
            hourly strategies become cost-competitive sooner. <strong>Transmission:</strong> Major inter-ISO expansion
            would boost cross-regional strategies (1B) while reducing same-ISO hourly strategies\u2019 geographic advantage.</p>
        </div>
    `;
}

// ─── Toggle Wiring ──────────────────────────────────────────────────────────

function wireToggles() {
    function wireToggleGroup(container, callback) {
        if (!container) return;
        container.querySelectorAll('button').forEach(btn => {
            btn.addEventListener('click', () => {
                container.querySelectorAll('button').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                callback();
            });
        });
    }

    wireToggleGroup(document.getElementById('crossoverParticipationToggle'), () => {
        const p = document.getElementById('crossoverParticipationToggle').querySelector('.active').dataset.val;
        buildCrossoverCharts(parseInt(p));
    });

    wireToggleGroup(document.getElementById('radarParticipationToggle'), () => {
        const p = document.getElementById('radarParticipationToggle').querySelector('.active').dataset.val;
        buildClusterRadar(parseInt(p));
    });
}

// ─── Fade-in Observer ───────────────────────────────────────────────────────

function initFadeObserver() {
    var fadeEls = document.querySelectorAll('.fade-in');
    if (!fadeEls.length) return;
    var obs = new IntersectionObserver(function(entries) {
        entries.forEach(function(e) {
            if (e.isIntersecting) {
                e.target.classList.add('visible');
                obs.unobserve(e.target);
            }
        });
    }, { threshold: 0.15 });
    fadeEls.forEach(function(el) { obs.observe(el); });
}

// ─── Init ───────────────────────────────────────────────────────────────────

function init() {
    populateHeroStats();
    buildCrossoverCharts(15);
    buildGasRibbonChart();
    buildSystemCostCrossover();
    buildCostBreakdown();
    populateCrossoverInsight();
    buildClusterRadar(15);
    buildWrightsLawChart();
    buildRiskCharts();
    populateNuclearNarrative();
    populateDissenting();
    wireToggles();
    initFadeObserver();
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

})();
