(function() {
'use strict';

// ─── Constants ──────────────────────────────────────────────────────────────
const STRATEGIES = ['1B','2A','2C','2C_rolloff'];
const CORE_STRATEGIES = ['1B','2A','2C'];
const ROLLOFF_PAIRS = { '2C': '2C_rolloff' };
const ISOS = ['CAISO','ERCOT','PJM','NYISO','NEISO','MISO','SPP'];
const THRESHOLDS = [80, 85, 90, 92.5, 95, 97.5, 99.5, 99.99];
const PARTICIPATION_LEVELS = [5, 10, 15, 20, 25, 50, 75];
const KEY_THRESHOLDS = [85, 90, 95, 99.5];

const STRATEGY_LABELS = {
    '1B': '1B — Consequential (Fossil-Avg)',
    '2A': '2A — Hourly (New-Build Only)',
    '2C': '2C — Hourly (All Clean)',
    '2C_rolloff': '2C — Nuclear Rolloff'
};

const STRATEGY_SHORT = {
    '1B': '1B', '2A': '2A', '2C': '2C',
    '2C_rolloff': '2C-R'
};

const STRATEGY_COLORS = {
    '1B': '#6366F1',
    '2A': '#F59E0B', '2C': '#0EA5E9', '2C_rolloff': '#7DD3FC'
};

// TWh demand per ISO (approximate 2023 load)
const GRID_DEMANDS = { CAISO: 224, ERCOT: 488, PJM: 843, NYISO: 152, NEISO: 115, MISO: 660, SPP: 296 };
const TOTAL_DEMAND = Object.values(GRID_DEMANDS).reduce((a, b) => a + b, 0);

// Cluster definitions
const CLUSTERS = [
    { name: 'Consequential Netting', strategies: ['1B'], color: '#6366F1', desc: 'Cross-regional new-build deployment via queue-ordered procurement using fossil-avg emission baseline. Cheapest per-MWh, but deploys mostly mature VRE (solar/wind). New firm clean (nuclear, CCS, LDES) only appears at extreme thresholds (95%+) — minimal contribution to learning curves for immature technologies.' },
    { name: 'Hourly New-Build', strategies: ['2A'], color: '#F59E0B', desc: 'Same-ISO hourly matching using 100% new-build resources. The hourly constraint forces early deployment of new firm clean (10–38% of mix) and storage (battery + LDES, 10–33%) — the primary driver of learning curves for next-generation clean technologies.' },
    { name: 'Hourly Hybrid', strategies: ['2C'], color: '#0EA5E9', desc: 'Same-ISO hourly matching blending existing clean (SSS, merchant EAC, nuclear uprate) with new-build. Cost-effective but depends on nuclear staying online for cheapest firm supply.' }
];

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

// Pre-compute aggregate cache for performance
const AGG_CACHE = {};
function cachedAggregate(strategy, participation, threshold) {
    const key = `${strategy}|${participation}|${threshold}`;
    if (!AGG_CACHE[key]) AGG_CACHE[key] = aggregateCrossISO(strategy, participation, threshold);
    return AGG_CACHE[key];
}

// ─── Scoring Functions ──────────────────────────────────────────────────────

function rankStrategies(strategies, values, lowerIsBetter) {
    const sorted = strategies.map((s, i) => ({ s, v: values[i] }))
        .sort((a, b) => lowerIsBetter ? a.v - b.v : b.v - a.v);
    const ranks = {};
    sorted.forEach((item, i) => { ranks[item.s] = i + 1; });
    return ranks;
}

function rankToGrade(rank, total) {
    const pct = rank / total;
    if (pct <= 0.25) return 'A';
    if (pct <= 0.5) return 'B';
    if (pct <= 0.75) return 'C';
    return 'D';
}

function gradeClass(grade) {
    if (grade === 'A') return 'story-badge-green';
    if (grade === 'D') return 'story-badge-red';
    return 'story-badge';
}

function computeLearningScore(strategy) {
    // Learning curves are driven by new FIRM CLEAN deployment (nuclear, CCS, LDES, offshore wind)
    // NOT by new-build VRE (solar/onshore wind are mature, already at NOAK pricing)
    // Hourly matching forces firm clean + storage deployment from 50% CFE onward
    // Consequential netting deploys mostly VRE; firm clean only at extreme thresholds (95%+)
    const scores = {
        '1B': 3, // mostly mature VRE, minimal firm clean until 95%+, Scenario A delayed learning
        '2A': 9, // hourly constraint forces 10-38% firm clean + 10-33% storage, Scenario B fast learning
        '2C': 8  // same hourly template but mixed sourcing dilutes new firm clean signal
    };
    return scores[strategy] || 3;
}

function computeNuclearDependency(strategy, participation, threshold) {
    const base = cachedAggregate(strategy, participation, threshold);
    const rolloffKey = ROLLOFF_PAIRS[strategy];
    if (!rolloffKey) return 0;
    const rolloff = cachedAggregate(rolloffKey, participation, threshold);
    if (!base.avgCost || !rolloff.avgCost) return 0;
    return Math.abs(rolloff.avgCost - base.avgCost) / Math.max(base.avgCost, 1) * 100;
}

// Gas backup capacity costs from pipeline_config.py (step 2.2)
// Existing gas: fixed O&M only ($/kW-yr). New gas: full annualized CCGT ($/kW-yr).
const EXISTING_GAS_FOM_KW_YR = { CAISO: 16, ERCOT: 13, PJM: 14, NYISO: 17, NEISO: 15, MISO: 14, SPP: 13 };
const NEW_CCGT_COST_KW_YR = { CAISO: 112, ERCOT: 89, PJM: 99, NYISO: 114, NEISO: 105, MISO: 95, SPP: 88 };
const EXISTING_GAS_CAPACITY_GW = { CAISO: 37, ERCOT: 55, PJM: 75, NYISO: 18, NEISO: 14, MISO: 68, SPP: 32 };

function computeSystemCost(strategy, participation, threshold) {
    // Total system cost = clean procurement ($M) + NEW-BUILD gas capacity cost ($M)
    // Existing gas is sunk cost — only new gas above existing capacity counts
    // Uses step 2.2 annualized CCGT cost ($/kW-yr) from pipeline_config.py
    let totalCleanM = 0, totalNewGasCostM = 0;

    for (const iso of ISOS) {
        const d = getRecord(strategy, iso, participation, threshold);
        if (!d) continue;
        totalCleanM += d.tc || 0;

        const gasGw = d.gasGw || 0;
        const existingGw = EXISTING_GAS_CAPACITY_GW[iso] || 0;
        const newGasGw = Math.max(0, gasGw - existingGw);

        // Convert: GW × $/kW-yr × 1e6 kW/GW = $M/yr
        totalNewGasCostM += newGasGw * NEW_CCGT_COST_KW_YR[iso] * 1e3;
    }

    return totalCleanM + totalNewGasCostM; // $M total
}

function buildScorecardData(threshold, participation) {
    const results = [];
    const n = CORE_STRATEGIES.length;

    const aggs = {};
    for (const s of CORE_STRATEGIES) {
        aggs[s] = cachedAggregate(s, participation, threshold);
    }

    // Compute raw values for each criterion
    const co2Vals = CORE_STRATEGIES.map(s => aggs[s].totalCo2);
    const costVals = CORE_STRATEGIES.map(s => aggs[s].avgCost);
    const gasVals = CORE_STRATEGIES.map(s => aggs[s].totalGasGw);
    const sysCostVals = CORE_STRATEGIES.map(s => computeSystemCost(s, participation, threshold));
    const learnVals = CORE_STRATEGIES.map(s => computeLearningScore(s));
    const curtVals = CORE_STRATEGIES.map(s => aggs[s].totalCurtTwh);

    // Rank: co2 higher is better, cost lower better, gas lower better, sysCost lower better, learning higher better, curt lower better
    const co2Ranks = rankStrategies(CORE_STRATEGIES, co2Vals, false);
    const costRanks = rankStrategies(CORE_STRATEGIES, costVals, true);
    const gasRanks = rankStrategies(CORE_STRATEGIES, gasVals, true);
    const sysCostRanks = rankStrategies(CORE_STRATEGIES, sysCostVals, true);  // lower total system cost = better
    const learnRanks = rankStrategies(CORE_STRATEGIES, learnVals, false);
    const curtRanks = rankStrategies(CORE_STRATEGIES, curtVals, true);

    for (let i = 0; i < n; i++) {
        const s = CORE_STRATEGIES[i];
        const co2Grade = rankToGrade(co2Ranks[s], n);
        const costGrade = rankToGrade(costRanks[s], n);
        const gasGrade = rankToGrade(gasRanks[s], n);
        const sysCostGrade = rankToGrade(sysCostRanks[s], n);
        const learnGrade = rankToGrade(learnRanks[s], n);
        const curtGrade = rankToGrade(curtRanks[s], n);

        // Composite: equally weighted ranks across 6 criteria
        const composite = 100 - ((co2Ranks[s] + costRanks[s] + gasRanks[s] + sysCostRanks[s] + learnRanks[s] + curtRanks[s]) / (6 * n)) * 100;

        results.push({
            strategy: s,
            co2: { val: co2Vals[i], grade: co2Grade, rank: co2Ranks[s] },
            cost: { val: costVals[i], grade: costGrade, rank: costRanks[s] },
            gas: { val: gasVals[i], grade: gasGrade, rank: gasRanks[s] },
            sysCost: { val: sysCostVals[i], grade: sysCostGrade, rank: sysCostRanks[s] },
            learn: { val: learnVals[i], grade: learnGrade, rank: learnRanks[s] },
            curt: { val: curtVals[i], grade: curtGrade, rank: curtRanks[s] },
            composite: Math.round(composite)
        });
    }

    results.sort((a, b) => b.composite - a.composite);
    return results;
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

// ─── Section 01: Executive Summary ──────────────────────────────────────────

function populateExecutiveSummary() {
    const scorecard = buildScorecardData(95, 25);
    const best = scorecard[0];

    // Find strategy with lowest cost at 95%/25%
    let lowestCost = Infinity, lowestCostStrat = '';
    let highestCo2 = -Infinity, highestCo2Strat = '';
    for (const s of CORE_STRATEGIES) {
        const agg = cachedAggregate(s, 25, 95);
        if (agg.avgCost < lowestCost) { lowestCost = agg.avgCost; lowestCostStrat = s; }
        if (agg.totalCo2 > highestCo2) { highestCo2 = agg.totalCo2; highestCo2Strat = s; }
    }

    // Gas range
    const gasVals = CORE_STRATEGIES.map(s => cachedAggregate(s, 25, 95).totalGasGw);
    const gasMin = Math.min(...gasVals), gasMax = Math.max(...gasVals);

    // Nuclear dependency — 2C
    const nucDep = computeNuclearDependency('2C', 25, 95);

    document.getElementById('statBestCost').textContent = STRATEGY_SHORT[lowestCostStrat];
    document.getElementById('statBestCo2').textContent = STRATEGY_SHORT[highestCo2Strat];
    document.getElementById('statGasRisk').textContent = `${fmt(gasMin, 0)}–${fmt(gasMax, 0)} GW`;
    document.getElementById('statLearning').textContent = '2A';
    document.getElementById('statNuclearDep').textContent = `${fmt(nucDep, 0)}% cost shift`;
    document.getElementById('statRecommended').textContent = STRATEGY_SHORT[best.strategy];

    // Executive finding narrative
    const finding = document.getElementById('executiveFinding');
    finding.innerHTML = `<p><strong>Key Finding:</strong> No single strategy dominates across all six criteria.
        Strategy <strong>${STRATEGY_SHORT[best.strategy]}</strong> (${STRATEGY_LABELS[best.strategy]}) achieves the best composite score
        at 95% CFE / 25% participation — the threshold where firm clean deployment becomes decisive.
        Consequential netting (1B) deploys new-build capacity cross-regionally at the lowest per-MWh cost
        ($${fmt(lowestCost)}/MWh), but primarily builds mature VRE (solar/wind) and avoids firm clean until forced.
        Strategy 2A (hourly new-build only) costs
        $${fmt(cachedAggregate('2A', 25, 95).avgCost)}/MWh but the hourly matching constraint forces early deployment of
        new firm clean (10–38% of mix) and storage (10–33%) — directly accelerating learning curves for immature technologies.
        At 95%, the gap between strategies that deploy firm clean and those that don't becomes critical:
        strategies without firm clean must either accept significantly more gas on the grid or pay exponentially more
        for VRE overbuild plus curtailment.
        Strategy 2C (hourly hybrid) combines existing clean (SSS, EACs, nuclear uprate) with new-build,
        but total system cost (including new-build gas backup) reveals hidden costs that per-MWh pricing obscures.</p>`;
}

function buildSummaryBarChart() {
    destroyChart('summaryBarChart');
    const scorecard = buildScorecardData(90, 25);
    const labels = scorecard.map(r => STRATEGY_LABELS[r.strategy]);
    const values = scorecard.map(r => r.composite);
    const colors = scorecard.map(r => STRATEGY_COLORS[r.strategy]);

    const ctx = document.getElementById('summaryBarChart');
    charts.summaryBarChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Composite Score',
                data: values,
                backgroundColor: colors,
                borderRadius: 4,
                barThickness: 24
            }]
        },
        options: {
            ...baseOpts(),
            indexAxis: 'y',
            plugins: {
                ...baseOpts().plugins,
                legend: { display: false }
            },
            scales: {
                x: { ...baseOpts().scales.x, title: { display: true, text: 'Composite Score (higher = better)', font: { family: 'DM Sans', size: 11 } }, min: 0, max: 100 },
                y: { ...baseOpts().scales.y, ticks: { font: { family: 'DM Sans', size: 10 } } }
            }
        }
    });
}

// ─── Section 02: Scorecard Table ────────────────────────────────────────────

function renderScorecardTable(threshold, participation) {
    const data = buildScorecardData(threshold, participation);
    const tbody = document.getElementById('scorecardBody');
    tbody.innerHTML = '';

    for (const row of data) {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td style="font-weight:600; white-space:nowrap; color:${STRATEGY_COLORS[row.strategy]}">${STRATEGY_LABELS[row.strategy]}</td>
            <td><span class="${gradeClass(row.co2.grade)}">${row.co2.grade}</span> <span style="font-size:0.8em;color:var(--text-muted)">${fmt(row.co2.val)} Mt</span></td>
            <td><span class="${gradeClass(row.cost.grade)}">${row.cost.grade}</span> <span style="font-size:0.8em;color:var(--text-muted)">$${fmt(row.cost.val)}/MWh</span></td>
            <td><span class="${gradeClass(row.gas.grade)}">${row.gas.grade}</span> <span style="font-size:0.8em;color:var(--text-muted)">${fmt(row.gas.val, 0)} GW</span></td>
            <td><span class="${gradeClass(row.sysCost.grade)}">${row.sysCost.grade}</span> <span style="font-size:0.8em;color:var(--text-muted)">$${fmt(row.sysCost.val / 1000, 0)}B</span></td>
            <td><span class="${gradeClass(row.learn.grade)}">${row.learn.grade}</span></td>
            <td><span class="${gradeClass(row.curt.grade)}">${row.curt.grade}</span> <span style="font-size:0.8em;color:var(--text-muted)">${fmt(row.curt.val)} TWh</span></td>
            <td style="font-weight:700; color:var(--navy)">${row.composite}</td>
        `;
        tbody.appendChild(tr);
    }
}

// ─── Section 03: Crossover Charts ───────────────────────────────────────────

function buildCrossoverCharts(participation) {
    const chartConfigs = [
        { id: 'crossoverCostChart', field: 'avgCost', label: '$/MWh', lowerBetter: true },
        { id: 'crossoverCo2Chart', field: 'totalCo2', label: 'Mt CO₂', lowerBetter: false },
        { id: 'crossoverGasChart', field: 'totalGasGw', label: 'GW Gas', lowerBetter: true },
        { id: 'crossoverCurtChart', field: 'totalCurtTwh', label: 'TWh Curtailed', lowerBetter: true }
    ];

    for (const cfg of chartConfigs) {
        destroyChart(cfg.id);
        const datasets = CORE_STRATEGIES.map(s => ({
            label: STRATEGY_SHORT[s],
            data: THRESHOLDS.map(t => {
                const agg = cachedAggregate(s, participation, t);
                return agg[cfg.field];
            }),
            borderColor: STRATEGY_COLORS[s],
            backgroundColor: STRATEGY_COLORS[s] + '20',
            tension: 0.3,
            pointRadius: 3,
            borderWidth: 2,
            fill: false
        }));

        const ctx = document.getElementById(cfg.id);
        charts[cfg.id] = new Chart(ctx, {
            type: 'line',
            data: { labels: THRESHOLDS.map(t => t + '%'), datasets },
            options: {
                ...baseOpts(),
                scales: {
                    x: { ...baseOpts().scales.x, title: { display: true, text: 'CFE Threshold', font: { family: 'DM Sans', size: 11 } } },
                    y: { ...baseOpts().scales.y, title: { display: true, text: cfg.label, font: { family: 'DM Sans', size: 11 } } }
                }
            }
        });
    }

    // Crossover insight
    populateCrossoverInsight(participation);
}

function populateCrossoverInsight(participation) {
    // Find where 1B loses cost advantage to 2C
    let crossoverThreshold = null;
    for (const t of THRESHOLDS) {
        const a1b = cachedAggregate('1B', participation, t);
        const a2c = cachedAggregate('2C', participation, t);
        if (a2c.totalCo2 > a1b.totalCo2 * 1.5) {
            crossoverThreshold = t;
            break;
        }
    }

    const el = document.getElementById('crossoverInsight');
    const costAt90 = {};
    for (const s of CORE_STRATEGIES) costAt90[s] = cachedAggregate(s, participation, 90).avgCost;
    const cheapest = CORE_STRATEGIES.reduce((a, b) => costAt90[a] < costAt90[b] ? a : b);
    const priciest = CORE_STRATEGIES.reduce((a, b) => costAt90[a] > costAt90[b] ? a : b);

    el.innerHTML = `<p><strong>Key Crossover:</strong> At ${participation}% participation and 90% CFE, the cost spread between
        the cheapest strategy (${STRATEGY_SHORT[cheapest]} at $${fmt(costAt90[cheapest])}/MWh) and
        most expensive (${STRATEGY_SHORT[priciest]} at $${fmt(costAt90[priciest])}/MWh) is
        $${fmt(costAt90[priciest] - costAt90[cheapest])}/MWh.
        ${crossoverThreshold ? `CO₂ displacement diverges significantly above ${crossoverThreshold}% — consequential netting strategies begin decoupling from grid-level effects while hourly strategies drive measurable fossil displacement.` : 'CO₂ displacement patterns remain relatively stable across thresholds at this participation level.'}
        Gas capacity on grid diverges most sharply above 85% CFE, where strategies requiring firm backup either build new gas (harmful) or deploy clean firm (beneficial but expensive).</p>`;
}

// ─── Section 04: Strategy Clusters ──────────────────────────────────────────

function buildClusterRadar() {
    // Compute raw values per strategy at 95%/25%
    const stratData = {};
    for (const s of CORE_STRATEGIES) {
        const agg = cachedAggregate(s, 25, 95);
        stratData[s] = {
            co2: agg.totalCo2,
            cost: agg.avgCost,
            gas: agg.totalGasGw,
            sysCost: computeSystemCost(s, 25, 95),
            learn: computeLearningScore(s),
            curt: agg.totalCurtTwh
        };
    }

    // Normalize across all 3 strategies (0-1, higher = better)
    function normalize(field, invert) {
        const vals = CORE_STRATEGIES.map(s => stratData[s][field]);
        const min = Math.min(...vals), max = Math.max(...vals);
        const range = max - min || 1;
        const normed = {};
        CORE_STRATEGIES.forEach((s, i) => {
            normed[s] = invert ? 1 - (vals[i] - min) / range : (vals[i] - min) / range;
        });
        return normed;
    }

    const normCo2 = normalize('co2', false);    // higher = better
    const normCost = normalize('cost', true);    // lower = better
    const normGas = normalize('gas', true);      // lower = better
    const normSysCost = normalize('sysCost', true); // lower = better
    const normLearn = normalize('learn', false); // higher = better
    const normCurt = normalize('curt', true);    // lower = better

    const radarLabels = ['Emission\nReduction', 'Cost\nEfficiency', 'Low Gas\nLock-in', 'System\nCost', 'Learning\nCurves', 'Low\nCurtailment'];

    // Build individual radar chart per strategy
    for (const cluster of CLUSTERS) {
        const s = cluster.strategies[0];
        const chartId = 'radarChart' + s.replace('_', '');
        destroyChart(chartId);

        const data = [normCo2[s], normCost[s], normGas[s], normSysCost[s], normLearn[s], normCurt[s]];

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
                    borderWidth: 2,
                    pointRadius: 4,
                    pointBackgroundColor: cluster.color
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        beginAtZero: true, max: 1,
                        ticks: { display: false },
                        pointLabels: { font: { family: 'DM Sans', size: 10 } },
                        grid: { color: 'rgba(0,0,0,0.08)' }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: { bodyFont: { family: 'DM Sans' } }
                }
            }
        });

        // Description below chart
        const descEl = document.getElementById('clusterDesc' + s.replace('_', ''));
        if (descEl) {
            descEl.innerHTML = `<p style="font-size: 0.82rem; color: var(--text-secondary); margin: 0">${cluster.desc}</p>`;
        }
    }
}

// ─── Section 05: Nuclear Fragility ──────────────────────────────────────────

function buildFragilityCharts() {
    const thresholds = [70, 80, 90, 95, 99.5];
    const participation = 25;
    const labels = thresholds.map(t => t + '%');

    const configs = [
        { id: 'fragilityCostChart', field: 'avgCost', label: '$/MWh' },
        { id: 'fragilityCo2Chart', field: 'totalCo2', label: 'Mt CO₂' },
        { id: 'fragilityGasChart', field: 'totalGasGw', label: 'GW Gas' },
        { id: 'fragilityCurtChart', field: 'totalCurtTwh', label: 'TWh Curtailed' }
    ];

    for (const cfg of configs) {
        destroyChart(cfg.id);
        const base = thresholds.map(t => cachedAggregate('2C', participation, t)[cfg.field]);
        const rolloff = thresholds.map(t => cachedAggregate('2C_rolloff', participation, t)[cfg.field]);

        charts[cfg.id] = new Chart(document.getElementById(cfg.id), {
            type: 'bar',
            data: {
                labels,
                datasets: [
                    { label: '2C (Nuclear Online)', data: base, backgroundColor: '#0EA5E9', borderRadius: 4, barPercentage: 0.8, categoryPercentage: 0.7 },
                    { label: '2C Rolloff (Nuclear Retires)', data: rolloff, backgroundColor: '#7DD3FC', borderRadius: 4, barPercentage: 0.8, categoryPercentage: 0.7 }
                ]
            },
            options: {
                ...baseOpts(),
                scales: {
                    x: { ...baseOpts().scales.x },
                    y: { ...baseOpts().scales.y, title: { display: true, text: cfg.label, font: { family: 'DM Sans', size: 11 } } }
                }
            }
        });
    }

    // Fragility insight
    const base90 = cachedAggregate('2C', participation, 90);
    const roll90 = cachedAggregate('2C_rolloff', participation, 90);
    const costDelta = roll90.avgCost - base90.avgCost;
    const co2Delta = roll90.totalCo2 - base90.totalCo2;
    const gasDelta = roll90.totalGasGw - base90.totalGasGw;

    document.getElementById('fragilityInsight').innerHTML = `<p><strong>Nuclear Fragility Result:</strong>
        At 90% CFE / ${participation}% participation, nuclear retirement shifts costs by
        ${costDelta >= 0 ? '+' : ''}$${fmt(costDelta)}/MWh, CO₂ by ${co2Delta >= 0 ? '+' : ''}${fmt(co2Delta)} Mt,
        and gas capacity by ${gasDelta >= 0 ? '+' : ''}${fmt(gasDelta, 0)} GW.
        ${Math.abs(costDelta) > 5 ? 'This is a <strong>material fragility</strong> — Strategy 2C\'s cost advantage depends significantly on nuclear staying online. If merchant revenue erodes and plants close, the strategy\'s economics deteriorate.' : 'The impact is <strong>moderate</strong> — 2C retains most of its advantage even under nuclear retirement, though at a cost premium.'}
        ${gasDelta > 5 ? ' Notably, nuclear retirement results in ' + fmt(gasDelta, 0) + ' GW more gas on the grid — converting a clean energy asset loss into a fossil fuel lock-in.' : ''}</p>`;
}

// ─── Section 06: Critical Risks ─────────────────────────────────────────────

function buildRiskCharts() {
    buildGasLockinChart();
    buildCurtailmentChart();
    buildLearningCurveChart();
    buildNuclearStrandChart();
    populateRiskInsight();
}

function buildGasLockinChart() {
    destroyChart('gasLockinChart');
    const participation = 50, threshold = 90;

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
    const participation = 50, threshold = 90;
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

function buildLearningCurveChart() {
    destroyChart('learningCurveChart');

    // Load Wright's Law data via fetch
    fetch('js/wrights_law_data.json')
        .then(r => r.json())
        .then(data => {
            const pcts = data.metadata.participation_pcts.map(p => (p * 100) + '%');
            const datasets = [
                { key: 'strat_1b', label: '1B', color: STRATEGY_COLORS['1B'] },
                { key: 'strat_2a', label: '2A', color: STRATEGY_COLORS['2A'] },
                { key: 'strat_2c_gated', label: '2C', color: STRATEGY_COLORS['2C'] }
            ].map(d => ({
                label: d.label,
                data: data.strategy_comparison_90[d.key],
                borderColor: d.color,
                backgroundColor: d.color + '20',
                tension: 0.3,
                pointRadius: 3,
                borderWidth: 2,
                fill: false
            }));

            charts.learningCurveChart = new Chart(document.getElementById('learningCurveChart'), {
                type: 'line',
                data: { labels: pcts, datasets },
                options: {
                    ...baseOpts(),
                    scales: {
                        x: { ...baseOpts().scales.x, title: { display: true, text: 'C&I Participation', font: { family: 'DM Sans', size: 11 } } },
                        y: { ...baseOpts().scales.y, title: { display: true, text: '$/MWh (Blended)', font: { family: 'DM Sans', size: 11 } } }
                    }
                }
            });
        })
        .catch(() => {
            // Fallback if JSON not available — use static learning scores
            const ctx = document.getElementById('learningCurveChart');
            ctx.parentElement.innerHTML = '<p style="color:var(--text-muted);text-align:center;padding:2rem">Wright\'s Law data unavailable</p>';
        });
}

function buildNuclearStrandChart() {
    destroyChart('nuclearStrandChart');
    const participation = 25, threshold = 90;

    const exposures = CORE_STRATEGIES.map(s => {
        const dep = computeNuclearDependency(s, participation, threshold);
        // Nuclear stranding risk = does the strategy create revenue for existing nuclear?
        // 2C actively funds nuclear via EAC purchases — lowest stranding risk
        // 2A is 100% new-build, zero nuclear dependency — no stranding exposure either way
        // 1A/1B deploy new-build cross-regionally, no EAC revenue pathway for existing nuclear
        const strandingAdj = {
            '1B': 15,  // no EAC revenue pathway — nuclear unsupported, highest stranding risk
            '2A': 0,   // 100% new-build, zero nuclear dependency — no stranding exposure
            '2C': 0    // directly funds existing nuclear via EAC purchases — lowest risk
        };
        return dep + (strandingAdj[s] || 5);
    });

    charts.nuclearStrandChart = new Chart(document.getElementById('nuclearStrandChart'), {
        type: 'bar',
        data: {
            labels: CORE_STRATEGIES.map(s => STRATEGY_SHORT[s]),
            datasets: [{
                label: 'Nuclear Exposure Index',
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
                x: { ...baseOpts().scales.x, title: { display: true, text: 'Exposure Index (higher = more fragile)', font: { family: 'DM Sans', size: 11 } } },
                y: { ...baseOpts().scales.y }
            }
        }
    });
}

function populateRiskInsight() {
    const gas90 = CORE_STRATEGIES.map(s => ({ s, gas: cachedAggregate(s, 50, 90).totalGasGw }));
    gas90.sort((a, b) => b.gas - a.gas);
    const curt90 = CORE_STRATEGIES.map(s => ({ s, curt: cachedAggregate(s, 50, 90).totalCurtTwh }));
    curt90.sort((a, b) => b.curt - a.curt);

    document.getElementById('riskInsight').innerHTML = `<p><strong>Risk Summary (90% / 50% participation):</strong>
        Gas lock-in varies from ${fmt(gas90[gas90.length - 1].gas, 0)} GW (${STRATEGY_SHORT[gas90[gas90.length - 1].s]}) to
        ${fmt(gas90[0].gas, 0)} GW (${STRATEGY_SHORT[gas90[0].s]}) — a ${fmt(gas90[0].gas - gas90[gas90.length - 1].gas, 0)} GW spread.
        Curtailment ranges from ${fmt(curt90[curt90.length - 1].curt)} TWh to ${fmt(curt90[0].curt)} TWh.
        Strategy 2C (hourly hybrid) carries nuclear stranding risk because it relies on existing nuclear for
        cheap firm supply — if nuclear plants close, 2C loses its cost advantage (see Section 05).
        Strategy 2A (hourly new-build only) has zero nuclear dependency since it deploys 100% new capacity.
        Consequential netting (1B) deploys 100% new-build cross-regionally but provides no EAC revenue
        to support existing nuclear — it contributes to stranding risk indirectly by not creating a revenue pathway.</p>`;
}

// ─── Section 07: Regime Map ─────────────────────────────────────────────────

function buildRegimeMap() {
    const ambitionTiers = [
        { label: 'Moderate (80–85%)', thresholds: [80, 85] },
        { label: 'Ambitious (90–95%)', thresholds: [90, 92.5, 95] },
        { label: 'Last Mile (97.5–99.99%)', thresholds: [97.5, 99.5, 99.99] }
    ];
    const participations = PARTICIPATION_LEVELS;

    const tbody = document.getElementById('regimeMapBody');
    tbody.innerHTML = '';

    for (const tier of ambitionTiers) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td style="font-weight:600; white-space:nowrap">${tier.label}</td>`;

        for (const p of participations) {
            // Average composite score across thresholds in tier
            const stratScores = {};
            for (const s of CORE_STRATEGIES) stratScores[s] = 0;

            for (const t of tier.thresholds) {
                const scorecard = buildScorecardData(t, p);
                for (const row of scorecard) {
                    stratScores[row.strategy] += row.composite;
                }
            }

            // Normalize
            for (const s of CORE_STRATEGIES) stratScores[s] /= tier.thresholds.length;

            // Find winner
            let bestStrat = CORE_STRATEGIES[0], bestScore = -Infinity;
            for (const s of CORE_STRATEGIES) {
                if (stratScores[s] > bestScore) { bestScore = stratScores[s]; bestStrat = s; }
            }

            const td = document.createElement('td');
            td.style.backgroundColor = STRATEGY_COLORS[bestStrat] + '25';
            td.style.fontWeight = '700';
            td.style.textAlign = 'center';
            td.style.color = STRATEGY_COLORS[bestStrat];
            td.textContent = STRATEGY_SHORT[bestStrat];
            td.title = STRATEGY_LABELS[bestStrat] + ' (score: ' + Math.round(bestScore) + ')';
            tr.appendChild(td);
        }
        tbody.appendChild(tr);
    }

    // Regime insight
    document.getElementById('regimeInsight').innerHTML = `<p><strong>Regime Insights:</strong>
        At 80–85% CFE, consequential netting (1B) dominates on cost because the threshold is achievable with
        mostly VRE deployed cross-regionally. At 90%+, the value of grid-transformative strategies increases —
        hourly strategies (2A, 2C) force deployment of new firm clean (nuclear, CCS) and storage (LDES) that
        consequential netting avoids until extreme thresholds.
        At 95%+, strategies must deploy significant firm clean and storage, making the learning curve and
        scalability criteria decisive. 2A's costs decrease at higher participation (learning curve benefits),
        while 2C's costs increase sharply (existing clean supply exhaustion).
        Participation level matters most at high ambition: early adopters (5–10%) cannot move the learning
        curve alone, but at 25%+ participation, collective procurement volumes begin to reach critical mass
        for technology cost breakthroughs.</p>`;
}

// ─── Section 08: Dissenting Considerations ──────────────────────────────────

function populateDissenting() {
    const el = document.getElementById('dissentingContent');
    el.innerHTML = `
        <div class="insight-box" style="margin-bottom: var(--space-lg)">
            <h4 style="font-family: var(--font-heading); color: var(--navy); margin: 0 0 var(--space-sm)">Against Consequential Netting (1B)</h4>
            <p>While 1B deploys 100% new-build capacity, it does so <strong>cross-regionally</strong> via
            queue-ordered procurement — the new clean generation may be built in a distant ISO, not in
            the buyer's grid. This means the buyer's local grid sees no physical transformation. The emission
            reduction is real on a system-wide basis, but the buyer's ISO may retain the same fossil fleet.
            At scale, this concentrates new clean capacity in the cheapest-to-build regions while leaving
            harder-to-decarbonize grids untouched. Additionally, consequential netting uses annual accounting,
            missing the hourly mismatch between clean generation and actual load. The new capacity deployed is
            overwhelmingly mature VRE — minimal contribution to firm clean learning curves.</p>
        </div>

        <div class="insight-box" style="margin-bottom: var(--space-lg)">
            <h4 style="font-family: var(--font-heading); color: var(--navy); margin: 0 0 var(--space-sm)">Against Hourly New-Build Only (2A)</h4>
            <p>Strategy 2A deploys 100% new-build capacity in the buyer's ISO with hourly matching — maximum
            additionality and local grid transformation. However, this comes at the highest cost among the three
            strategies because every MWh must come from newly built resources matched to each hour. The premium
            reflects the full cost of new clean firm, storage, and variable renewables without the cost relief
            of existing clean supply. At lower participation levels, 2A's high per-buyer cost may limit adoption,
            slowing the very learning curves it aims to accelerate. Its resource mix, templated from the efficient
            frontier, may also over-build in ISOs where existing clean supply could be leveraged more cost-effectively.</p>
        </div>

        <div class="insight-box" style="margin-bottom: var(--space-lg)">
            <h4 style="font-family: var(--font-heading); color: var(--navy); margin: 0 0 var(--space-sm)">Against Hourly Hybrid (2C)</h4>
            <p>Strategy 2C's cost advantage depends heavily on existing nuclear remaining online.
            The nuclear fragility test (Section 05) quantifies this risk. If merchant revenue
            erosion from VRE oversupply closes nuclear plants, 2C loses its cheapest firm generation
            source and must either accept higher costs or increased gas dependency. Additionally,
            2C's blending of existing and new clean resources may dilute the additionality signal
            that drives learning curve acceleration for advanced technologies.</p>
        </div>

        <div class="insight-box" style="margin-bottom: var(--space-lg)">
            <h4 style="font-family: var(--font-heading); color: var(--navy); margin: 0 0 var(--space-sm)">What Could Flip This Analysis</h4>
            <p><strong>Policy changes:</strong> A federal clean energy standard or carbon price above
            $100/ton CO₂ would make consequential netting's cost advantage irrelevant and reward
            high-additionality strategies. <strong>Nuclear policy:</strong> If production tax credits
            (45U) are extended or expanded, nuclear revenue adequacy concerns diminish, reducing
            Strategy 2C's fragility. <strong>Technology breakthroughs:</strong> If advanced nuclear or
            LDES costs fall faster than Wright's Law projections, hourly strategies become more
            cost-competitive sooner. <strong>Grid topology:</strong> If inter-ISO
            transmission capacity expands significantly, cross-regional strategies (1B) gain
            effectiveness while same-ISO hourly strategies lose their geographic advantage.</p>
        </div>
    `;
}

// ─── Toggle Wiring ──────────────────────────────────────────────────────────

function wireToggles() {
    // Scorecard toggles
    const thresholdToggle = document.getElementById('scorecardThresholdToggle');
    const participationToggle = document.getElementById('scorecardParticipationToggle');

    function getScorecardParams() {
        const t = thresholdToggle.querySelector('.active').dataset.val;
        const p = participationToggle.querySelector('.active').dataset.val;
        return { threshold: parseFloat(t), participation: parseInt(p) };
    }

    function wireToggleGroup(container, callback) {
        container.querySelectorAll('button').forEach(btn => {
            btn.addEventListener('click', () => {
                container.querySelectorAll('button').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                callback();
            });
        });
    }

    wireToggleGroup(thresholdToggle, () => {
        const { threshold, participation } = getScorecardParams();
        renderScorecardTable(threshold, participation);
    });

    wireToggleGroup(participationToggle, () => {
        const { threshold, participation } = getScorecardParams();
        renderScorecardTable(threshold, participation);
    });

    // Crossover participation toggle
    wireToggleGroup(document.getElementById('crossoverParticipationToggle'), () => {
        const p = document.getElementById('crossoverParticipationToggle').querySelector('.active').dataset.val;
        buildCrossoverCharts(parseInt(p));
    });
}

// ─── TOC Active State ───────────────────────────────────────────────────────

function initTocHighlight() {
    const sections = document.querySelectorAll('.report-section');
    const tocLinks = document.querySelectorAll('.report-toc-list a');

    const observer = new IntersectionObserver((entries) => {
        for (const entry of entries) {
            if (entry.isIntersecting) {
                const id = entry.target.id;
                tocLinks.forEach(link => {
                    link.classList.toggle('active', link.getAttribute('href') === '#' + id);
                });
            }
        }
    }, { rootMargin: '-80px 0px -60% 0px', threshold: 0.1 });

    sections.forEach(s => observer.observe(s));
}

// ─── Init ───────────────────────────────────────────────────────────────────

function init() {
    populateExecutiveSummary();
    buildSummaryBarChart();
    renderScorecardTable(95, 25);
    buildCrossoverCharts(25);
    buildClusterRadar();
    buildFragilityCharts();
    buildRiskCharts();
    buildRegimeMap();
    populateDissenting();
    wireToggles();
    initTocHighlight();
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

})();
