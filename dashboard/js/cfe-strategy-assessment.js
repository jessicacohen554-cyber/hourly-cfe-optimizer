(function() {
'use strict';

// ─── Constants ──────────────────────────────────────────────────────────────
const STRATEGIES = ['1B','2C','2C_rolloff'];
const CORE_STRATEGIES = ['1B','2C'];
const ROLLOFF_PAIRS = { '2C': '2C_rolloff' };
const ISOS = ['CAISO','ERCOT','PJM','NYISO','NEISO','MISO','SPP'];
const THRESHOLDS = [90, 92.5, 95, 97.5, 99.5, 99.99];
const PARTICIPATION_LEVELS = [5, 10, 15, 20, 25, 50, 75];

// Display labels — internal data keys stay '1B'/'2C' to match DEPLOYMENT_DATA
const STRATEGY_LABELS = {
    '1B': '1A — Consequential',
    '2C': '2C — Hourly Matching with SSS',
    '2C_rolloff': '2C — Nuclear Rolloff'
};
const STRATEGY_SHORT = { '1B': '1A', '2C': '2C', '2C_rolloff': '2C-R' };

const STRATEGY_COLORS = {
    '1B': '#6366F1',
    '2C': '#0EA5E9',
    '2C_rolloff': '#7DD3FC'
};

// TWh demand per ISO (approximate 2023 load)
const GRID_DEMANDS = { CAISO: 224, ERCOT: 488, PJM: 843, NYISO: 152, NEISO: 115, MISO: 660, SPP: 296 };
const TOTAL_DEMAND = Object.values(GRID_DEMANDS).reduce((a, b) => a + b, 0);

// Cluster definitions
const CLUSTERS = [
    { name: 'Consequential Netting', strategies: ['1B'], color: '#6366F1',
      desc: 'Cross-regional new-build deployment via queue-ordered procurement. Deploys mostly mature VRE (solar/wind). New firm clean only at extreme thresholds. Minimal contribution to learning curves for immature technologies. Provides zero financial support to existing nuclear.' },
    { name: 'Hourly Matching with SSS', strategies: ['2C'], color: '#0EA5E9',
      desc: 'Same-ISO hourly matching blending existing clean (SSS, merchant EAC, nuclear uprate) with new-build. Forces firm clean + storage deployment. Creates EAC revenue that keeps existing nuclear economically viable. Proposed framework for GHG Protocol Scope 2 revision.' }
];

const charts = {};

// Gas costs from pipeline_config.py
const EXISTING_GAS_FOM_KW_YR = { CAISO: 16, ERCOT: 13, PJM: 14, NYISO: 17, NEISO: 15, MISO: 14, SPP: 13 };
const NEW_CCGT_COST_KW_YR = { CAISO: 112, ERCOT: 89, PJM: 99, NYISO: 114, NEISO: 105, MISO: 95, SPP: 88 };
const EXISTING_GAS_CAPACITY_GW = { CAISO: 37, ERCOT: 55, PJM: 75, NYISO: 18, NEISO: 14, MISO: 68, SPP: 32 };

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
    let count = 0;

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
        count++;
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
        isoCount: count
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

// ─── Section 1: Hero Stats ──────────────────────────────────────────────────

function populateHeroStats() {
    const part = 15, thresh = 95;
    const a1b = cachedAggregate('1B', part, thresh);
    const a2c = cachedAggregate('2C', part, thresh);
    const sys1b = computeSystemCost('1B', part, thresh);
    const sys2c = computeSystemCost('2C', part, thresh);

    // Best CO2
    const bestCo2 = Math.max(a1b.totalCo2, a2c.totalCo2);
    document.getElementById('heroStatCo2').textContent = fmt(bestCo2, 0) + ' Mt';

    // System cost ratio
    if (sys2c > 0 && sys1b > 0) {
        if (sys1b > sys2c) {
            const ratio = sys1b / sys2c;
            document.getElementById('heroStatCostRatio').textContent = fmt(ratio, 1) + '×';
        } else {
            document.getElementById('heroStatCostRatio').textContent = 'Comparable';
        }
    }

    // Gas delta
    const gasDelta = a1b.totalGasGw - a2c.totalGasGw;
    document.getElementById('heroStatGasDelta').textContent = (gasDelta > 0 ? '' : '+') + fmt(Math.abs(gasDelta), 0) + ' GW';

    // Learning advantage
    document.getElementById('heroStatLearning').textContent = '2.7×';
}

// ─── Section 3: Crossover Charts ────────────────────────────────────────────

function buildCrossoverCharts(participation) {
    const chartConfigs = [
        { id: 'crossoverCostChart', field: 'avgCost', label: '$/MWh', title: 'Clean Energy Cost' },
        { id: 'crossoverCo2Chart', field: 'totalCo2', label: 'Mt CO₂', title: 'CO₂ Displaced' },
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
    if (costEl) costEl.innerHTML = `<strong>1A:</strong> $${fmt(a1b.avgCost)}/MWh &nbsp; <strong>2C:</strong> $${fmt(a2c.avgCost)}/MWh at 95% CFE`;

    const sysEl = document.getElementById('stepSystemStat');
    if (sysEl) sysEl.innerHTML = `<strong>Total system cost at 95%:</strong> 1A = $${fmt(sys1b / 1000, 0)}B &nbsp; 2C = $${fmt(sys2c / 1000, 0)}B`;

    const co2El = document.getElementById('stepCo2Stat');
    if (co2El) co2El.innerHTML = `<strong>CO₂ displaced:</strong> 1A = ${fmt(a1b.totalCo2)} Mt &nbsp; 2C = ${fmt(a2c.totalCo2)} Mt`;

    const gasEl = document.getElementById('stepGasStat');
    if (gasEl) gasEl.innerHTML = `<strong>Gas on grid:</strong> 1A = ${fmt(a1b.totalGasGw, 0)} GW &nbsp; 2C = ${fmt(a2c.totalGasGw, 0)} GW &nbsp; (${fmt(Math.abs(a1b.totalGasGw - a2c.totalGasGw), 0)} GW delta)`;
}

// ─── Section 4: Gas Ribbon Chart ────────────────────────────────────────────

function buildGasRibbonChart() {
    destroyChart('gasRibbonChart');
    const partLabels = PARTICIPATION_LEVELS.map(p => p + '%');
    const existingGasGw = Object.values(EXISTING_GAS_CAPACITY_GW).reduce((a, b) => a + b, 0);

    // For each strategy, compute min/median/max gas across thresholds at each participation
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
        // 1A ribbon band
        { label: '1A range', data: ribbon1B.map(r => r.max), borderColor: 'transparent', backgroundColor: STRATEGY_COLORS['1B'] + '18',
          fill: '+1', pointRadius: 0, order: 4 },
        { label: '_1A_min', data: ribbon1B.map(r => r.min), borderColor: 'transparent', backgroundColor: 'transparent',
          fill: false, pointRadius: 0, order: 4 },
        // 1A median line
        { label: STRATEGY_SHORT['1B'], data: ribbon1B.map(r => r.median), borderColor: STRATEGY_COLORS['1B'],
          backgroundColor: STRATEGY_COLORS['1B'], borderWidth: 2.5, tension: 0.3, pointRadius: 3, pointHoverRadius: 6, fill: false, order: 1 },
        // 2C ribbon band
        { label: '2C range', data: ribbon2C.map(r => r.max), borderColor: 'transparent', backgroundColor: STRATEGY_COLORS['2C'] + '18',
          fill: '+1', pointRadius: 0, order: 3 },
        { label: '_2C_min', data: ribbon2C.map(r => r.min), borderColor: 'transparent', backgroundColor: 'transparent',
          fill: false, pointRadius: 0, order: 3 },
        // 2C median line
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

    // Insight
    const gas1b50 = cachedAggregate('1B', 50, 95).totalGasGw;
    const gas2c50 = cachedAggregate('2C', 50, 95).totalGasGw;
    const el = document.getElementById('gasRibbonInsight');
    if (el) {
        el.innerHTML = `<p><strong>Gas Displacement:</strong> At 50% participation and 95% CFE,
            <strong>1A</strong> leaves ${fmt(gas1b50, 0)} GW of gas on grid while
            <strong>2C</strong> reduces it to ${fmt(gas2c50, 0)} GW — a ${fmt(Math.abs(gas1b50 - gas2c50), 0)} GW difference.
            ${gas1b50 > existingGasGw ? `Under 1A, gas capacity actually <em>exceeds</em> today's ${existingGasGw} GW installed base — meaning new gas must be built to backstop the VRE-heavy mix.` : ''}
            The ribbon shows how this gap widens at higher participation levels, where 1A's cheap-VRE-plus-gas model becomes increasingly expensive relative to 2C's firm-clean approach.</p>`;
    }
}

// ─── Section 5: Strategy Pentagon (Radar) Charts ────────────────────────────

function buildClusterRadar() {
    const participation = 15, threshold = 95;

    // Compute raw values per strategy
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

    // Normalize: best strategy = 1.0 (outer edge), other = proportion of best
    // "Higher is better" metrics: value / max(values)
    // "Lower is better" metrics: min(values) / value
    function normalizeRelativeToBest(field, higherIsBetter) {
        const vals = CORE_STRATEGIES.map(s => stratData[s][field]);
        if (higherIsBetter) {
            const best = Math.max(...vals);
            const normed = {};
            CORE_STRATEGIES.forEach((s, i) => {
                normed[s] = best > 0 ? vals[i] / best : 0;
            });
            return normed;
        } else {
            const best = Math.min(...vals);
            const normed = {};
            CORE_STRATEGIES.forEach((s, i) => {
                normed[s] = vals[i] > 0 ? best / vals[i] : 0;
            });
            return normed;
        }
    }

    const normCo2 = normalizeRelativeToBest('co2', true);     // higher = better
    const normCost = normalizeRelativeToBest('cost', false);    // lower = better
    const normGas = normalizeRelativeToBest('gas', false);      // lower = better
    const normLearn = normalizeRelativeToBest('learn', true);   // higher = better
    const normCurt = normalizeRelativeToBest('curt', false);    // lower = better

    const radarLabels = ['Emission\nReduction', 'True\nCost', 'Low Gas\nLock-in', 'Learning\nCurves', 'Low\nCurtailment'];

    for (const cluster of CLUSTERS) {
        const s = cluster.strategies[0];
        const chartId = 'radarChart' + s.replace('_', '');
        destroyChart(chartId);

        const data = [normCo2[s], normCost[s], normGas[s], normLearn[s], normCurt[s]];

        const ctx = document.getElementById(chartId);
        if (!ctx) continue;

        charts[chartId] = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: radarLabels,
                datasets: [{
                    label: STRATEGY_LABELS[s],
                    data: data,
                    borderColor: cluster.color,
                    backgroundColor: cluster.color + '30',
                    borderWidth: 2.5,
                    pointRadius: 5,
                    pointBackgroundColor: cluster.color,
                    pointBorderColor: '#fff',
                    pointBorderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        beginAtZero: true, max: 1, min: 0,
                        ticks: { display: false, stepSize: 0.25 },
                        pointLabels: { font: { family: 'DM Sans', size: 11, weight: 600 }, color: '#1A2744' },
                        grid: { color: 'rgba(0,0,0,0.08)' },
                        angleLines: { color: 'rgba(0,0,0,0.08)' }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        bodyFont: { family: 'DM Sans' },
                        callbacks: {
                            label: function(ctx) {
                                return `${ctx.label}: ${(ctx.raw * 100).toFixed(0)}% of best`;
                            }
                        }
                    }
                }
            }
        });

        // Description
        const descEl = document.getElementById('clusterDesc' + s.replace('_', ''));
        if (descEl) {
            descEl.innerHTML = `<p style="font-size: 0.82rem; color: var(--text-secondary); margin: 0; line-height: 1.6">${cluster.desc}</p>`;
        }
    }
}

// ─── Section 6: Wright's Law Chart ──────────────────────────────────────────

function buildWrightsLawChart() {
    destroyChart('wrightsLawChart');

    fetch('js/wrights_law_data.json')
        .then(r => r.json())
        .then(data => {
            const pcts = data.metadata.participation_pcts.map(p => (p * 100) + '%');
            const stratKeys = [
                { key: 'strat_1b', label: STRATEGY_SHORT['1B'], color: STRATEGY_COLORS['1B'] },
                { key: 'strat_2c_gated', label: STRATEGY_SHORT['2C'], color: STRATEGY_COLORS['2C'] }
            ];

            const datasets = stratKeys.map(d => ({
                label: d.label,
                data: data.strategy_comparison_90[d.key],
                borderColor: d.color,
                backgroundColor: d.color + '15',
                tension: 0.3,
                pointRadius: 4,
                pointHoverRadius: 7,
                borderWidth: 2.5,
                fill: false
            }));

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

            // Find crossover points
            const insight = document.getElementById('wrightsInsight');
            if (insight) {
                const d2c = data.strategy_comparison_90.strat_2c_gated;
                let crossIdx = d2c ? d2c.findIndex(v => v <= 0) : -1;
                const crossPct = crossIdx >= 0 ? data.metadata.participation_pcts[crossIdx] : null;

                insight.innerHTML = `<p><strong>Learning Curve Impact:</strong>
                    Strategy 2C forces deployment of firm clean (nuclear uprate, CCS) and long-duration storage (LDES) — technologies still on steep learning curves.
                    ${crossPct ? `At ${(crossPct * 100).toFixed(0)}% participation, Wright's Law cost reductions make 2C <strong>cost-negative</strong> — the learning curve benefits exceed the procurement premium.` : 'As participation scales, learning curve benefits accumulate.'}
                    Strategy 1A deploys mostly mature VRE (solar/onshore wind) that is already at NOAK pricing — contributing minimally to the cost breakthroughs the entire grid needs for deep decarbonization.</p>`;
            }
        })
        .catch(() => {
            const ctx = document.getElementById('wrightsLawChart');
            if (ctx) ctx.parentElement.innerHTML = '<p style="color:var(--text-muted);text-align:center;padding:2rem">Wright\'s Law data unavailable</p>';
        });
}

// ─── Section 7: Nuclear Narrative ───────────────────────────────────────────

function populateNuclearNarrative() {
    const part = 15, thresh = 95;
    const base = cachedAggregate('2C', part, thresh);
    const rolloff = cachedAggregate('2C_rolloff', part, thresh);

    const costDelta = rolloff.avgCost - base.avgCost;
    const co2Delta = base.totalCo2 - rolloff.totalCo2; // positive = CO2 displaced lost
    const gasDelta = rolloff.totalGasGw - base.totalGasGw;

    const costEl = document.getElementById('nucCostDelta');
    if (costEl) costEl.textContent = '+$' + fmt(Math.abs(costDelta)) + '/MWh';

    const co2El = document.getElementById('nucCo2Delta');
    if (co2El) co2El.textContent = fmt(Math.abs(co2Delta)) + ' Mt';

    const gasEl = document.getElementById('nucGasDelta');
    if (gasEl) gasEl.textContent = '+' + fmt(Math.abs(gasDelta), 0) + ' GW';

    const insight = document.getElementById('nuclearInsight');
    if (insight) {
        insight.innerHTML = `<p><strong>What happens if nuclear retires?</strong>
            At 95% CFE / ${part}% participation, losing the existing nuclear fleet would increase clean procurement costs by
            <strong>$${fmt(Math.abs(costDelta))}/MWh</strong>, reduce CO₂ displacement by <strong>${fmt(Math.abs(co2Delta))} Mt</strong>,
            and add <strong>${fmt(Math.abs(gasDelta), 0)} GW</strong> of gas to the grid.</p>
            <p><strong>Strategy 2C actively prevents this outcome</strong> by creating EAC revenue (~$20–40/MWh) that supports nuclear plant economics and prevents premature retirement.
            <strong>Strategy 1A provides zero financial support to existing nuclear</strong> — under 1A, nuclear plants face the same merchant revenue erosion from VRE oversupply with no offsetting demand signal, accelerating the risk of premature closure.
            The US nuclear fleet generates ~20% of all electricity and ~50% of all clean electricity. Losing it would be catastrophic for grid decarbonization.</p>`;
    }
}

// ─── Section 8: Critical Risks ──────────────────────────────────────────────

function buildRiskCharts() {
    buildGasLockinChart();
    buildCurtailmentChart();
    buildNuclearStrandChart();
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

    charts.gasLockinChart = new Chart(document.getElementById('gasLockinChart'), {
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

    charts.curtailmentChart = new Chart(document.getElementById('curtailmentChart'), {
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

function buildNuclearStrandChart() {
    destroyChart('nuclearStrandChart');
    const participation = 15, threshold = 95;

    const base2c = cachedAggregate('2C', participation, threshold);
    const roll2c = cachedAggregate('2C_rolloff', participation, threshold);
    const dep2c = base2c.avgCost > 0 ? Math.abs(roll2c.avgCost - base2c.avgCost) / base2c.avgCost * 100 : 0;

    const exposures = CORE_STRATEGIES.map(s => {
        // 1B: zero nuclear support → high stranding risk
        // 2C: EAC revenue supports nuclear → low stranding risk
        const strandingScore = {
            '1B': 20 + dep2c * 0.5,  // high: no support + would face same consequences if nuclear retires
            '2C': Math.max(0, dep2c - 5)  // lower: actively supports but has dependency
        };
        return strandingScore[s] || 10;
    });

    charts.nuclearStrandChart = new Chart(document.getElementById('nuclearStrandChart'), {
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

function populateRiskInsight() {
    const gas95 = CORE_STRATEGIES.map(s => ({ s, gas: cachedAggregate(s, 50, 95).totalGasGw }));
    gas95.sort((a, b) => b.gas - a.gas);
    const curt95 = CORE_STRATEGIES.map(s => ({ s, curt: cachedAggregate(s, 50, 95).totalCurtTwh }));
    curt95.sort((a, b) => b.curt - a.curt);

    document.getElementById('riskInsight').innerHTML = `<p><strong>Risk Summary (95% / 50% participation):</strong>
        Gas lock-in: ${fmt(gas95[gas95.length - 1].gas, 0)} GW (${STRATEGY_SHORT[gas95[gas95.length - 1].s]}) to
        ${fmt(gas95[0].gas, 0)} GW (${STRATEGY_SHORT[gas95[0].s]}) — a ${fmt(gas95[0].gas - gas95[gas95.length - 1].gas, 0)} GW spread.
        Curtailment: ${fmt(curt95[curt95.length - 1].curt)} TWh to ${fmt(curt95[0].curt)} TWh.
        <strong>Nuclear preservation is the critical climate variable.</strong> 2C actively supports nuclear via EAC revenue;
        1A provides zero support — leaving the grid's largest clean generation source exposed to merchant revenue erosion and premature retirement.</p>`;
}

// ─── Section 9: Regime Map ──────────────────────────────────────────────────

function buildRegimeMap() {
    const ambitionTiers = [
        { label: 'High Ambition (90–95%)', thresholds: [90, 92.5, 95] },
        { label: 'Last Mile (97.5–99.99%)', thresholds: [97.5, 99.5, 99.99] }
    ];
    const participations = [5, 10, 15, 25, 50];

    const tbody = document.getElementById('regimeMapBody');
    tbody.innerHTML = '';

    for (const tier of ambitionTiers) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td style="font-weight:600; white-space:nowrap">${tier.label}</td>`;

        for (const p of participations) {
            // Average system cost across thresholds in tier — lower wins
            const avgCosts = {};
            for (const s of CORE_STRATEGIES) {
                avgCosts[s] = tier.thresholds.reduce((sum, t) => sum + computeSystemCost(s, p, t), 0) / tier.thresholds.length;
            }

            let bestStrat = CORE_STRATEGIES[0];
            for (const s of CORE_STRATEGIES) {
                if (avgCosts[s] < avgCosts[bestStrat]) bestStrat = s;
            }

            const td = document.createElement('td');
            td.style.backgroundColor = STRATEGY_COLORS[bestStrat] + '25';
            td.style.fontWeight = '700';
            td.style.textAlign = 'center';
            td.style.color = STRATEGY_COLORS[bestStrat];
            td.textContent = STRATEGY_SHORT[bestStrat];
            td.title = `${STRATEGY_LABELS[bestStrat]} — system cost: $${fmt(avgCosts[bestStrat] / 1000, 0)}B`;
            tr.appendChild(td);
        }
        tbody.appendChild(tr);
    }

    document.getElementById('regimeInsight').innerHTML = `<p><strong>Regime Insights:</strong>
        At low participation (5–10%), 1A's cheap cross-regional VRE keeps total system cost competitive.
        As participation rises above 15–25%, 1A's gas backup requirements drive system costs dramatically higher than 2C.
        At 50% participation and 95% CFE, 1A's system cost (clean + gas) can exceed 2C's by 3× or more.
        This crossover is consistent across ambition levels because the gas penalty scales with the gap between 1A's VRE-heavy mix and the firm capacity needed for grid reliability.</p>`;
}

// ─── Section 10: Dissenting ─────────────────────────────────────────────────

function populateDissenting() {
    const el = document.getElementById('dissentingContent');
    el.innerHTML = `
        <div class="insight-box scroll-reveal" style="margin-bottom: var(--space-lg)">
            <h4 style="font-family: var(--font-heading); color: var(--navy); margin: 0 0 var(--space-sm)">Against 1A — Consequential</h4>
            <p>1A's new-build requirement means it provides <strong>zero financial support to existing nuclear</strong>.
            Under 1A, nuclear faces the same merchant revenue erosion from VRE oversupply with no offsetting demand signal —
            accelerating premature retirement of the largest source of clean firm generation on the grid.
            Beyond nuclear, 1A deploys cross-regionally — new clean generation may be built in a distant ISO,
            leaving the buyer's grid physically untransformed. The new capacity is overwhelmingly mature VRE,
            contributing minimally to firm clean learning curves. And consequential netting uses annual accounting,
            missing the hourly mismatch between clean generation and actual load.</p>
        </div>

        <div class="insight-box scroll-reveal" style="margin-bottom: var(--space-lg)">
            <h4 style="font-family: var(--font-heading); color: var(--navy); margin: 0 0 var(--space-sm)">Against 2C — Hourly Matching with SSS</h4>
            <p>While 2C's EAC purchases create revenue that helps keep nuclear online, this is a
            <strong>mutual dependency</strong> — 2C needs nuclear for cheap firm supply, and nuclear needs
            2C's revenue for economic viability. If nuclear plants close despite EAC support (e.g., due to
            regulatory or safety issues), 2C loses its cheapest firm generation source and must accept
            higher costs or increased gas dependency. Additionally, 2C's blending of existing and new clean
            resources means only ~30% of its clean procurement is truly additional new-build — the rest
            reshuffles existing clean claims, which may dilute the additionality signal.</p>
        </div>

        <div class="insight-box scroll-reveal" style="margin-bottom: var(--space-lg)">
            <h4 style="font-family: var(--font-heading); color: var(--navy); margin: 0 0 var(--space-sm)">What Could Flip This Analysis</h4>
            <p><strong>Policy:</strong> A federal clean energy standard or carbon price above $100/ton would make 1A's cost advantage
            irrelevant and reward high-additionality strategies. <strong>Nuclear policy:</strong> Extended production tax credits (45U)
            would reduce nuclear revenue adequacy concerns, reducing 2C's fragility.
            <strong>Technology:</strong> If advanced nuclear or LDES costs fall faster than Wright's Law projections,
            hourly strategies become cost-competitive sooner. <strong>Transmission:</strong> Major inter-ISO expansion
            would boost cross-regional strategies (1A) while reducing same-ISO hourly strategies' geographic advantage.</p>
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
}

// ─── Init ───────────────────────────────────────────────────────────────────

function init() {
    populateHeroStats();
    buildCrossoverCharts(15);
    buildGasRibbonChart();
    buildClusterRadar();
    buildWrightsLawChart();
    populateNuclearNarrative();
    buildRiskCharts();
    buildRegimeMap();
    populateDissenting();
    wireToggles();
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

})();
