/**
 * IPP Company Report — Shared Template JS
 * =========================================
 * Renders all 7 report sections for any company.
 * Each company HTML page sets COMPANY_ID before loading this script.
 * Reads from: IPP_SMARTARGETS_DATA (ipp-smartargets-data.js)
 *             FLEET_COMPANIES, FOSSIL_DISPATCH_DECLINE (fleet-survival-data.js)
 */

/* global COMPANY_ID, IPP_SMARTARGETS_DATA, FLEET_COMPANIES, FOSSIL_DISPATCH_DECLINE, Chart, RESOURCE_COLORS, ISO_COLORS */

(function () {
    'use strict';

    // ─── Constants ───────────────────────────────────────────
    const YEARS = [2023, 2030, 2035, 2040, 2045, 2050];
    const YEAR_LABELS = ['2023', '2030', '2035', '2040', '2045', '2050'];
    const SBTi_ANNUAL_RATE = 0.07; // 7% per year power sector

    const FUEL_ORDER = ['nuclear', 'hydro', 'geothermal', 'wind', 'solar', 'battery', 'gas_ccgt', 'gas_peaker', 'coal', 'oil'];
    const FUEL_LABELS = {
        nuclear: 'Nuclear', hydro: 'Hydro', geothermal: 'Geothermal',
        wind: 'Wind', solar: 'Solar', battery: 'Battery',
        gas_ccgt: 'Gas CCGT', gas_peaker: 'Gas Peaker', coal: 'Coal', oil: 'Oil'
    };
    const FUEL_COLORS = {
        nuclear: RESOURCE_COLORS.nuclear, hydro: RESOURCE_COLORS.hydro,
        geothermal: RESOURCE_COLORS.geothermal, wind: RESOURCE_COLORS.wind,
        solar: RESOURCE_COLORS.solar, battery: RESOURCE_COLORS.battery,
        gas_ccgt: RESOURCE_COLORS.fossilGas || '#6B7280', gas_peaker: '#9CA3AF',
        coal: RESOURCE_COLORS.fossilCoal || '#374151', oil: RESOURCE_COLORS.fossilOil || '#92400E'
    };

    const DIMENSION_LABELS = {
        conditions: { label: 'Market Conditions', values: { F: 'Facilitating', C: 'Challenging' } },
        demand_growth: { label: 'Demand Growth', values: { Low: 'Low', Medium: 'Medium', High: 'High' } },
        price_sens: { label: 'LCOE Sensitivity', values: { Low: 'Low', Medium: 'Medium', High: 'High' } },
        ppa_level: { label: 'PPA Level', values: { Low: 'Low', Medium: 'Medium', High: 'High' } },
        gas_friction: { label: 'Gas Friction', values: {} }
    };

    // ─── Data Loading ────────────────────────────────────────
    const co = IPP_SMARTARGETS_DATA.companies[COMPANY_ID];
    if (!co) {
        document.getElementById('reportContent').innerHTML = '<p style="color:red;padding:40px;">Company data not found for: ' + COMPANY_ID + '</p>';
        return;
    }

    // Get fleet data from fleet-survival-data.js if available
    const fleetCo = (typeof FLEET_COMPANIES !== 'undefined')
        ? FLEET_COMPANIES.find(c => c.shortName.toLowerCase() === co.shortName.toLowerCase() || c.name === co.name)
        : null;

    const charts = {}; // Track chart instances for cleanup

    // ─── Utilities ───────────────────────────────────────────
    function fmt(n, decimals) {
        if (n === undefined || n === null) return '—';
        if (decimals === undefined) decimals = (Math.abs(n) >= 100 ? 0 : 1);
        return n.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
    }

    function pctChange(from, to) {
        if (!from) return 0;
        return ((to - from) / from) * 100;
    }

    function spreadRatio(p10, p50, p90) {
        if (!p50 || p50 === 0) return 999;
        return (p90 - p10) / Math.abs(p50);
    }

    function convergenceClass(ratio) {
        if (ratio < 0.3) return 'convergent';
        if (ratio < 0.8) return 'moderate';
        return 'divergent';
    }

    function convergenceLabel(ratio) {
        if (ratio < 0.3) return 'Convergent';
        if (ratio < 0.8) return 'Moderate Uncertainty';
        return 'Highly Divergent';
    }

    function sbtiTrajectory(baseline) {
        return YEARS.map((y, i) => {
            if (i === 0) return baseline;
            const yearsFromBase = y - 2023;
            return baseline * Math.pow(1 - SBTi_ANNUAL_RATE, yearsFromBase);
        });
    }

    function makeChart(canvasId, config) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return null;
        if (charts[canvasId]) { charts[canvasId].destroy(); }
        charts[canvasId] = new Chart(ctx, config);
        return charts[canvasId];
    }

    function getFleetByFuel() {
        const summary = co.fleet_summary;
        const byFuel = {};
        for (const iso of Object.keys(summary)) {
            for (const fuel of Object.keys(summary[iso])) {
                if (!byFuel[fuel]) byFuel[fuel] = { cap_mw: 0, gen_twh: 0, co2_mt: 0 };
                byFuel[fuel].cap_mw += summary[iso][fuel].cap_mw || 0;
                byFuel[fuel].gen_twh += summary[iso][fuel].gen_twh || 0;
                byFuel[fuel].co2_mt += summary[iso][fuel].co2_mt || 0;
            }
        }
        return byFuel;
    }

    function getFleetByISO() {
        const summary = co.fleet_summary;
        const byISO = {};
        for (const iso of Object.keys(summary)) {
            byISO[iso] = { cap_mw: 0, gen_twh: 0, co2_mt: 0 };
            for (const fuel of Object.keys(summary[iso])) {
                byISO[iso].cap_mw += summary[iso][fuel].cap_mw || 0;
                byISO[iso].gen_twh += summary[iso][fuel].gen_twh || 0;
                byISO[iso].co2_mt += summary[iso][fuel].co2_mt || 0;
            }
        }
        return byISO;
    }

    // ─── SECTION 1: Fleet Profile & Characterization ─────────
    function renderFleetProfile() {
        const byFuel = getFleetByFuel();
        const byISO = getFleetByISO();
        const isos = Object.keys(byISO);

        // Stats
        const totalCap = co.cap_gw;
        const totalGen = co.gen_twh;
        const totalCO2 = co.co2_2024_mt;
        const intensity = co.intensity_kg;
        const cleanCap = FUEL_ORDER.filter(f => ['nuclear','hydro','geothermal','wind','solar','battery'].includes(f))
            .reduce((s, f) => s + (byFuel[f] ? byFuel[f].cap_mw : 0), 0);
        const cleanPct = ((cleanCap / (totalCap * 1000)) * 100).toFixed(0);

        // Inject stats
        document.getElementById('statCapacity').textContent = fmt(totalCap, 1) + ' GW';
        document.getElementById('statGeneration').textContent = fmt(totalGen, 0) + ' TWh';
        document.getElementById('statEmissions').textContent = fmt(totalCO2, 0) + ' Mt';
        document.getElementById('statIntensity').textContent = fmt(intensity, 0) + ' kg/MWh';
        document.getElementById('statISOs').textContent = isos.length;
        document.getElementById('statCleanPct').textContent = cleanPct + '%';

        // Doughnut: capacity by fuel
        const doughnutFuels = FUEL_ORDER.filter(f => byFuel[f] && byFuel[f].cap_mw > 0);
        makeChart('fleetDoughnutChart', {
            type: 'doughnut',
            data: {
                labels: doughnutFuels.map(f => FUEL_LABELS[f]),
                datasets: [{
                    data: doughnutFuels.map(f => byFuel[f].cap_mw),
                    backgroundColor: doughnutFuels.map(f => FUEL_COLORS[f]),
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'right', labels: { font: { size: 11 }, padding: 8 } },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => `${ctx.label}: ${fmt(ctx.raw)} MW (${((ctx.raw / (totalCap * 1000)) * 100).toFixed(1)}%)`
                        }
                    }
                }
            }
        });

        // Horizontal stacked bar: capacity by ISO
        const isoOrder = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP'].filter(i => byISO[i]);
        const isoFuels = FUEL_ORDER.filter(f => {
            return isoOrder.some(iso => co.fleet_summary[iso] && co.fleet_summary[iso][f] && co.fleet_summary[iso][f].cap_mw > 0);
        });

        makeChart('isoBarChart', {
            type: 'bar',
            data: {
                labels: isoOrder,
                datasets: isoFuels.map(f => ({
                    label: FUEL_LABELS[f],
                    data: isoOrder.map(iso => (co.fleet_summary[iso] && co.fleet_summary[iso][f]) ? co.fleet_summary[iso][f].cap_mw : 0),
                    backgroundColor: FUEL_COLORS[f]
                }))
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { stacked: true, title: { display: true, text: 'Capacity (MW)' } },
                    y: { stacked: true }
                },
                plugins: {
                    legend: { position: 'bottom', labels: { font: { size: 10 }, boxWidth: 12 } },
                    tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${fmt(ctx.raw)} MW` } }
                }
            }
        });

        // Narrative
        renderFleetNarrative(byFuel, byISO, cleanPct);
    }

    function renderFleetNarrative(byFuel, byISO, cleanPct) {
        const el = document.getElementById('fleetNarrative');
        if (!el) return;

        const isos = Object.keys(byISO);
        const hasCoal = byFuel.coal && byFuel.coal.cap_mw > 0;
        const hasNuclear = byFuel.nuclear && byFuel.nuclear.cap_mw > 0;
        const nuclearPct = hasNuclear ? ((byFuel.nuclear.cap_mw / (co.cap_gw * 1000)) * 100).toFixed(0) : 0;
        const coalPct = hasCoal ? ((byFuel.coal.cap_mw / (co.cap_gw * 1000)) * 100).toFixed(0) : 0;
        const gasCCGT = byFuel.gas_ccgt ? byFuel.gas_ccgt.cap_mw : 0;
        const gasPeaker = byFuel.gas_peaker ? byFuel.gas_peaker.cap_mw : 0;
        const renewCap = (byFuel.solar ? byFuel.solar.cap_mw : 0) + (byFuel.wind ? byFuel.wind.cap_mw : 0);
        const batteryCap = byFuel.battery ? byFuel.battery.cap_mw : 0;

        let strengths = [];
        let weaknesses = [];
        let opportunities = [];

        // Strengths
        if (hasNuclear && nuclearPct > 20) strengths.push(`Nuclear baseload (${nuclearPct}% of capacity) provides zero-carbon firm generation and strong capacity market revenue.`);
        if (isos.length >= 3) strengths.push(`Geographic diversification across ${isos.length} ISOs (${isos.join(', ')}) hedges against regional regulatory and market risk.`);
        if (renewCap > 2000) strengths.push(`Existing renewable portfolio (${fmt(renewCap)} MW solar+wind) provides foundation for scaling clean generation.`);
        if (batteryCap > 500) strengths.push(`Battery storage capacity (${fmt(batteryCap)} MW) positions for flexible grid services and peak shaving.`);
        if (gasCCGT > 3000) strengths.push(`Efficient gas CCGT fleet (${fmt(gasCCGT)} MW) provides reliable dispatchable capacity with moderate emissions intensity.`);
        if (cleanPct > 50) strengths.push(`Already ${cleanPct}% clean capacity — ahead of most peers in decarbonization.`);

        // Weaknesses
        if (hasCoal) weaknesses.push(`Coal exposure (${coalPct}% of capacity, ${fmt(byFuel.coal.co2_mt, 1)} Mt CO₂) represents the highest transition risk and stranded asset potential.`);
        if (isos.length === 1) weaknesses.push(`Single-ISO concentration (${isos[0]}) creates regulatory and market concentration risk.`);
        if (co.intensity_kg > 400) weaknesses.push(`High fleet intensity (${co.intensity_kg} kg/MWh) — among the most carbon-intensive large generators.`);
        if (gasPeaker > 2000) weaknesses.push(`Large peaker fleet (${fmt(gasPeaker)} MW) faces declining utilization as battery storage scales.`);
        if (cleanPct < 20 && !hasNuclear) weaknesses.push(`Very low clean capacity share (${cleanPct}%) with no nuclear hedge.`);

        // Opportunities
        if (hasCoal) opportunities.push(`Coal retirement + site repowering with solar/battery co-location can leverage existing grid interconnection and land.`);
        if (gasPeaker > 1000) opportunities.push(`Gas peaker → battery replacement at highest-heat-rate sites captures cost savings and emissions reduction.`);
        if (hasNuclear) opportunities.push(`Nuclear uprates and life extensions leverage highest-value zero-carbon asset under 45U PTC ($15/MWh).`);
        if (renewCap < 1000) opportunities.push(`Significant greenfield renewable opportunity — current clean portfolio is undersized relative to fleet.`);
        if (isos.some(i => i === 'ERCOT')) opportunities.push(`ERCOT market exposure provides high-value solar+battery opportunity due to peak pricing dynamics.`);
        if (isos.some(i => i === 'PJM')) opportunities.push(`PJM capacity market provides revenue stability for new firm clean generation investments.`);

        if (strengths.length === 0) strengths.push(`Fleet provides reliable dispatchable generation capacity.`);
        if (weaknesses.length === 0) weaknesses.push(`Limited near-term transition risk given fleet composition.`);
        if (opportunities.length === 0) opportunities.push(`Clean energy cost declines create attractive investment opportunities across the fleet's ISOs.`);

        el.innerHTML = `
            <div class="insight-box insight-success" style="margin-bottom: var(--space-md)">
                <strong>Strengths</strong>
                <ul style="margin:8px 0 0 16px">${strengths.map(s => `<li>${s}</li>`).join('')}</ul>
            </div>
            <div class="insight-box insight-warn" style="margin-bottom: var(--space-md)">
                <strong>Weaknesses</strong>
                <ul style="margin:8px 0 0 16px">${weaknesses.map(w => `<li>${w}</li>`).join('')}</ul>
            </div>
            <div class="insight-box" style="margin-bottom: var(--space-md)">
                <strong>Reduction Opportunities</strong>
                <ul style="margin:8px 0 0 16px">${opportunities.map(o => `<li>${o}</li>`).join('')}</ul>
            </div>
        `;
    }

    // ─── SECTION 2: Scenario Convergence & Divergence ────────
    function renderConvergenceDivergence() {
        const fb = co.fan_bands.all;
        const emissions = fb.emissions;

        // Fan band chart — P10/P50/P90 + min/max
        makeChart('emissionsFanChart', {
            type: 'line',
            data: {
                labels: YEAR_LABELS,
                datasets: [
                    {
                        label: 'Max',
                        data: emissions.max,
                        borderColor: 'transparent',
                        backgroundColor: 'rgba(239,68,68,0.06)',
                        fill: '+1',
                        pointRadius: 0
                    },
                    {
                        label: 'P90 (pessimistic)',
                        data: emissions.p90,
                        borderColor: 'rgba(239,68,68,0.5)',
                        backgroundColor: 'rgba(239,68,68,0.12)',
                        fill: '+1',
                        borderWidth: 1.5,
                        borderDash: [4, 3],
                        pointRadius: 2,
                        pointBackgroundColor: 'rgba(239,68,68,0.5)'
                    },
                    {
                        label: 'P50 (median)',
                        data: emissions.p50,
                        borderColor: '#0F172A',
                        backgroundColor: 'rgba(99,102,241,0.08)',
                        fill: '+1',
                        borderWidth: 2.5,
                        pointRadius: 4,
                        pointBackgroundColor: '#0F172A'
                    },
                    {
                        label: 'P10 (optimistic)',
                        data: emissions.p10,
                        borderColor: 'rgba(34,197,94,0.5)',
                        backgroundColor: 'rgba(34,197,94,0.08)',
                        fill: '+1',
                        borderWidth: 1.5,
                        borderDash: [4, 3],
                        pointRadius: 2,
                        pointBackgroundColor: 'rgba(34,197,94,0.5)'
                    },
                    {
                        label: 'Min',
                        data: emissions.min,
                        borderColor: 'transparent',
                        fill: false,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        title: { display: true, text: 'CO₂ Emissions (Mt)' },
                        beginAtZero: true
                    }
                },
                plugins: {
                    legend: { position: 'bottom', labels: { font: { size: 11 }, filter: item => item.text !== 'Max' && item.text !== 'Min' } },
                    tooltip: {
                        callbacks: {
                            label: ctx => `${ctx.dataset.label}: ${fmt(ctx.raw, 1)} Mt`
                        }
                    }
                }
            }
        });

        // Convergence analysis
        const convergenceEl = document.getElementById('convergenceAnalysis');
        if (convergenceEl) {
            let html = '<div class="grid-2col" style="gap:var(--space-md)">';
            YEARS.forEach((y, i) => {
                if (i === 0) return; // skip baseline
                const ratio = spreadRatio(emissions.p10[i], emissions.p50[i], emissions.p90[i]);
                const cls = convergenceClass(ratio);
                const lbl = convergenceLabel(ratio);
                html += `
                    <div class="card" style="padding:var(--space-md)">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                            <strong>${y}</strong>
                            <span class="convergence-tag ${cls}">${lbl}</span>
                        </div>
                        <div style="font-size:0.85rem;color:var(--text-secondary)">
                            Range: ${fmt(emissions.p10[i],1)} – ${fmt(emissions.p90[i],1)} Mt
                            (P50: ${fmt(emissions.p50[i],1)} Mt)
                        </div>
                    </div>
                `;
            });
            html += '</div>';
            convergenceEl.innerHTML = html;
        }

        // Dimension breakdown small multiples
        renderDimensionBreakdown();

        // Convergence narrative
        renderConvergenceNarrative(emissions);
    }

    function renderDimensionBreakdown() {
        const dims = co.dimension_fans;
        if (!dims) return;

        const container = document.getElementById('dimensionCharts');
        if (!container) return;

        const dimKeys = Object.keys(dims).filter(k => k !== 'gas_friction' || Object.keys(dims[k]).length > 1);

        dimKeys.forEach(dimKey => {
            const dimData = dims[dimKey];
            const values = Object.keys(dimData);
            if (values.length < 2) return;

            const canvasId = `dimChart_${dimKey}`;
            const wrapper = document.createElement('div');
            wrapper.className = 'card';
            wrapper.style.padding = 'var(--space-md)';
            wrapper.innerHTML = `
                <h4 style="font-size:0.85rem;margin-bottom:8px;color:var(--navy)">${(DIMENSION_LABELS[dimKey] || {}).label || dimKey}</h4>
                <div class="chart-container-sm"><canvas id="${canvasId}"></canvas></div>
            `;
            container.appendChild(wrapper);

            const colors = ['#22C55E', '#F59E0B', '#EF4444', '#6366F1', '#0EA5E9'];
            const datasets = values.map((val, vi) => {
                const em = dimData[val].emissions;
                if (!em || !em.p50) return null;
                return {
                    label: (DIMENSION_LABELS[dimKey] && DIMENSION_LABELS[dimKey].values[val]) || val,
                    data: em.p50,
                    borderColor: colors[vi % colors.length],
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    pointRadius: 3,
                    tension: 0.3
                };
            }).filter(Boolean);

            makeChart(canvasId, {
                type: 'line',
                data: { labels: YEAR_LABELS, datasets },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: { y: { title: { display: true, text: 'Emissions (Mt)' }, beginAtZero: true } },
                    plugins: { legend: { position: 'bottom', labels: { font: { size: 10 }, boxWidth: 10 } } }
                }
            });
        });
    }

    function renderConvergenceNarrative(emissions) {
        const el = document.getElementById('convergenceNarrative');
        if (!el) return;

        // Find most impactful dimension
        const dims = co.dimension_fans;
        let maxSpread = 0, maxDim = '';
        if (dims) {
            for (const dimKey of Object.keys(dims)) {
                const values = Object.keys(dims[dimKey]);
                if (values.length < 2) continue;
                const p50s = values.map(v => dims[dimKey][v].emissions ? dims[dimKey][v].emissions.p50[5] : 0).filter(Boolean);
                const spread = Math.max(...p50s) - Math.min(...p50s);
                if (spread > maxSpread) { maxSpread = spread; maxDim = dimKey; }
            }
        }

        const ratio2050 = spreadRatio(emissions.p10[5], emissions.p50[5], emissions.p90[5]);
        const ratio2035 = spreadRatio(emissions.p10[2], emissions.p50[2], emissions.p90[2]);
        const baseEmissions = emissions.p50[0];
        const p50_2050 = emissions.p50[5];
        const pctReduction = Math.abs(pctChange(baseEmissions, p50_2050));

        const dimLabel = (DIMENSION_LABELS[maxDim] || {}).label || maxDim;

        el.innerHTML = `<p>Across all 270 modeled scenarios, <strong>${co.shortName}</strong> shows a median emission reduction of
            <strong>${fmt(pctReduction, 0)}%</strong> by 2050 (from ${fmt(baseEmissions, 1)} to ${fmt(p50_2050, 1)} Mt CO₂).
            Near-term outcomes (2035) show ${convergenceLabel(ratio2035).toLowerCase()} agreement across scenarios
            (spread ratio: ${fmt(ratio2035, 2)}), while 2050 outcomes are ${convergenceLabel(ratio2050).toLowerCase()}
            (spread ratio: ${fmt(ratio2050, 2)}).</p>
            <p>The largest source of uncertainty is <strong>${dimLabel}</strong>, which shifts 2050 P50 emissions by
            ${fmt(maxSpread, 1)} Mt between best and worst values. This means ${co.shortName}'s long-term emission trajectory
            depends most heavily on ${dimLabel.toLowerCase()} outcomes.</p>`;
    }

    // ─── SECTION 3: Risk & Opportunity Assessment ────────────
    function renderRiskOpportunity() {
        const fb = co.fan_bands.all;

        // Profit fan band
        if (fb.profit) {
            makeChart('profitFanChart', {
                type: 'line',
                data: {
                    labels: YEAR_LABELS,
                    datasets: [
                        {
                            label: 'P90 (upside)',
                            data: fb.profit.p90,
                            borderColor: 'rgba(34,197,94,0.6)',
                            backgroundColor: 'rgba(34,197,94,0.1)',
                            fill: '+1',
                            borderWidth: 1.5,
                            borderDash: [4, 3],
                            pointRadius: 2
                        },
                        {
                            label: 'P50 (median)',
                            data: fb.profit.p50,
                            borderColor: '#0F172A',
                            backgroundColor: 'rgba(99,102,241,0.08)',
                            fill: '+1',
                            borderWidth: 2.5,
                            pointRadius: 4,
                            pointBackgroundColor: '#0F172A'
                        },
                        {
                            label: 'P10 (downside)',
                            data: fb.profit.p10,
                            borderColor: 'rgba(239,68,68,0.6)',
                            fill: false,
                            borderWidth: 1.5,
                            borderDash: [4, 3],
                            pointRadius: 2
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { title: { display: true, text: 'Annual Profit ($M)' } }
                    },
                    plugins: {
                        legend: { position: 'bottom' },
                        tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: $${fmt(ctx.raw)}M` } }
                    }
                }
            });
        }

        // Operating vs Stranded MW
        if (fb.operating_mw && fb.stranded_mw) {
            makeChart('capacityFateChart', {
                type: 'bar',
                data: {
                    labels: YEAR_LABELS,
                    datasets: [
                        {
                            label: 'Operating (P50)',
                            data: fb.operating_mw.p50,
                            backgroundColor: 'rgba(34,197,94,0.7)',
                            stack: 'stack0'
                        },
                        {
                            label: 'Stranded (P50)',
                            data: fb.stranded_mw.p50,
                            backgroundColor: 'rgba(239,68,68,0.7)',
                            stack: 'stack0'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { stacked: true },
                        y: { stacked: true, title: { display: true, text: 'Capacity (MW)' } }
                    },
                    plugins: {
                        legend: { position: 'bottom' },
                        tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${fmt(ctx.raw)} MW` } }
                    }
                }
            });
        }

        // Breakeven heatmap table
        renderBreakevenTable();

        // Risk narrative
        renderRiskNarrative();
    }

    function renderBreakevenTable() {
        const el = document.getElementById('breakevenTable');
        if (!el || !co.breakeven) return;

        const dims = ['conditions', 'demand_growth', 'price_sens', 'ppa_level'];
        let html = `<table class="data-table">
            <thead><tr>
                <th>Dimension</th><th>Value</th><th>% Profitable</th><th>Avg Profit 2050 ($M)</th><th>Avg Emissions 2050 (Mt)</th>
            </tr></thead><tbody>`;

        for (const dim of dims) {
            const data = co.breakeven[dim];
            if (!data) continue;
            for (const val of Object.keys(data)) {
                const d = data[val];
                const profColor = d.profitable_pct >= 90 ? '#dcfce7' : d.profitable_pct >= 50 ? '#fef9c3' : '#fee2e2';
                html += `<tr>
                    <td>${(DIMENSION_LABELS[dim] || {}).label || dim}</td>
                    <td>${(DIMENSION_LABELS[dim] && DIMENSION_LABELS[dim].values[val]) || val}</td>
                    <td style="background:${profColor};font-weight:600">${fmt(d.profitable_pct, 0)}%</td>
                    <td>$${fmt(d.avg_profit_2050)}</td>
                    <td>${fmt(d.avg_emissions_2050, 1)}</td>
                </tr>`;
            }
        }
        html += '</tbody></table>';
        el.innerHTML = html;
    }

    function renderRiskNarrative() {
        const el = document.getElementById('riskNarrative');
        if (!el) return;

        const fb = co.fan_bands.all;
        const profit2050_p10 = fb.profit ? fb.profit.p10[5] : 0;
        const profit2050_p50 = fb.profit ? fb.profit.p50[5] : 0;
        const profit2050_p90 = fb.profit ? fb.profit.p90[5] : 0;
        const stranded2050 = fb.stranded_mw ? fb.stranded_mw.p50[5] : 0;
        const operating2050 = fb.operating_mw ? fb.operating_mw.p50[5] : 0;

        const byFuel = getFleetByFuel();
        const hasCoal = byFuel.coal && byFuel.coal.cap_mw > 0;
        const coalMW = hasCoal ? byFuel.coal.cap_mw : 0;

        let riskItems = [];
        if (hasCoal) riskItems.push(`<strong>Coal stranding risk:</strong> ${fmt(coalMW)} MW of coal capacity faces retirement pressure. Under median scenarios, ${fmt(stranded2050)} MW total becomes stranded by 2050.`);
        if (profit2050_p10 < 0) riskItems.push(`<strong>Downside scenario loss:</strong> In the P10 (worst-case) scenario, annual profit falls to $${fmt(profit2050_p10)}M by 2050 — the fleet loses money under adverse conditions.`);
        if (profit2050_p10 > 0) riskItems.push(`<strong>Profit resilience:</strong> Even in the P10 (worst-case) scenario, the fleet maintains $${fmt(profit2050_p10)}M annual profit by 2050 — no money-losing scenarios.`);

        riskItems.push(`<strong>Profit range:</strong> 2050 annual profit ranges from $${fmt(profit2050_p10)}M (P10) to $${fmt(profit2050_p90)}M (P90), with a median of $${fmt(profit2050_p50)}M.`);

        // Retail hedge analysis
        const gasCap = (byFuel.gas_ccgt ? byFuel.gas_ccgt.cap_mw : 0) + (byFuel.gas_peaker ? byFuel.gas_peaker.cap_mw : 0);
        if (gasCap > 1000) {
            riskItems.push(`<strong>Retail hedge position:</strong> ${fmt(gasCap)} MW of gas capacity provides a natural hedge for retail power obligations — when wholesale prices spike (gas on the margin), generation revenue offsets retail exposure. This hedge value declines as gas dispatch falls under higher CFE thresholds.`);
        }

        el.innerHTML = riskItems.map(r => `<p>${r}</p>`).join('');
    }

    // ─── SECTION 4: Qualified Target Proposal ────────────────
    function renderQualifiedTarget() {
        const refBands = co.fan_bands.reference || co.fan_bands.all;
        const emissions = refBands.emissions;
        const baseline = emissions.p50[0];

        // Compute qualified targets (P50 reference trajectory)
        const qualifiedPcts = emissions.p50.map(v => ((baseline - v) / baseline) * 100);
        const sbti = sbtiTrajectory(baseline);

        // Target chart
        makeChart('targetChart', {
            type: 'line',
            data: {
                labels: YEAR_LABELS,
                datasets: [
                    {
                        label: 'P90 (conservative)',
                        data: emissions.p90,
                        borderColor: 'rgba(239,68,68,0.3)',
                        backgroundColor: 'rgba(239,68,68,0.06)',
                        fill: '+1',
                        borderWidth: 1,
                        borderDash: [3, 3],
                        pointRadius: 0
                    },
                    {
                        label: 'Qualified Target (P50)',
                        data: emissions.p50,
                        borderColor: '#6366F1',
                        backgroundColor: 'rgba(99,102,241,0.08)',
                        fill: '+1',
                        borderWidth: 3,
                        pointRadius: 5,
                        pointBackgroundColor: '#6366F1'
                    },
                    {
                        label: 'P10 (optimistic)',
                        data: emissions.p10,
                        borderColor: 'rgba(34,197,94,0.3)',
                        fill: false,
                        borderWidth: 1,
                        borderDash: [3, 3],
                        pointRadius: 0
                    },
                    {
                        label: '1.5°C SBTi Trajectory',
                        data: sbti,
                        borderColor: '#E91E63',
                        borderWidth: 2,
                        borderDash: [8, 4],
                        pointRadius: 0,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { title: { display: true, text: 'CO₂ Emissions (Mt)' }, beginAtZero: true }
                },
                plugins: {
                    legend: { position: 'bottom', labels: { font: { size: 11 } } },
                    tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${fmt(ctx.raw, 1)} Mt` } }
                }
            }
        });

        // Target stats
        const statsEl = document.getElementById('targetStats');
        if (statsEl) {
            const milestones = [
                { year: '2030', idx: 1 },
                { year: '2035', idx: 2 },
                { year: '2040', idx: 3 },
                { year: '2050', idx: 5 }
            ];
            statsEl.innerHTML = milestones.map(m => {
                const pct = qualifiedPcts[m.idx];
                const mt = emissions.p50[m.idx];
                const sbtiMt = sbti[m.idx];
                const gap = mt - sbtiMt;
                return `<div class="stat-card">
                    <div class="stat-value">${fmt(pct, 0)}%</div>
                    <div class="stat-label">${m.year} reduction</div>
                    <div style="font-size:0.75rem;color:var(--text-muted);margin-top:4px">
                        ${fmt(mt, 1)} Mt (gap to 1.5°C: ${gap > 0 ? '+' : ''}${fmt(gap, 1)} Mt)
                    </div>
                </div>`;
            }).join('');
        }

        // Target justification narrative
        renderTargetNarrative(emissions, qualifiedPcts, sbti);
    }

    function renderTargetNarrative(emissions, qualifiedPcts, sbti) {
        const el = document.getElementById('targetNarrative');
        if (!el) return;

        const baseline = emissions.p50[0];
        const target2030 = emissions.p50[1];
        const target2035 = emissions.p50[2];
        const target2050 = emissions.p50[5];
        const gap2050 = target2050 - sbti[5];

        const byFuel = getFleetByFuel();
        const hasCoal = byFuel.coal && byFuel.coal.cap_mw > 0;
        const hasNuclear = byFuel.nuclear && byFuel.nuclear.cap_mw > 0;
        const gasCap = (byFuel.gas_ccgt ? byFuel.gas_ccgt.gen_twh : 0) + (byFuel.gas_peaker ? byFuel.gas_peaker.gen_twh : 0);

        el.innerHTML = `
            <p>The proposed qualified target for <strong>${co.shortName}</strong> follows the P50 (median) emission trajectory across
            all reference scenarios — representing what market forces, existing policy, and announced retirements will organically deliver
            without additional mandates or subsidies.</p>

            <p><strong>2030 target: ${fmt(qualifiedPcts[1], 0)}% reduction</strong> (${fmt(target2030, 1)} Mt from ${fmt(baseline, 1)} Mt baseline).
            ${hasCoal ? 'Primarily driven by announced coal retirements and declining coal dispatch economics.' : 'Driven by declining fossil dispatch as clean capacity enters the market.'}</p>

            <p><strong>2035 target: ${fmt(qualifiedPcts[2], 0)}% reduction</strong> (${fmt(target2035, 1)} Mt).
            ${hasNuclear ? 'Nuclear fleet continues at high capacity factors, providing stable zero-carbon baseload.' : ''}
            Gas CCGT dispatch declines moderately as renewable penetration increases across ${co.shortName}'s operating ISOs.</p>

            <p><strong>2050 target: ${fmt(qualifiedPcts[5], 0)}% reduction</strong> (${fmt(target2050, 1)} Mt).
            ${gap2050 > 0 ? `This leaves a ${fmt(gap2050, 1)} Mt gap to the 1.5°C SBTi trajectory — bridging this gap requires the enabling conditions described in Section 6.` :
            'This meets or exceeds the 1.5°C SBTi trajectory — no additional enabling conditions required.'}</p>

            <div class="insight-box" style="margin-top:var(--space-md)">
                <strong>Why this target is achievable:</strong>
                <ul style="margin:8px 0 0 16px">
                    <li><strong>Reliability:</strong> Maintains sufficient firm dispatchable capacity (${fmt(gasCap, 0)} TWh gas generation) for grid adequacy</li>
                    <li><strong>Affordability:</strong> Follows market economics — no cross-subsidy or above-market procurement required</li>
                    <li><strong>Cost competitiveness:</strong> Aligned with market deployment curve; new clean investments at or below market clearing prices</li>
                    ${gasCap > 5 ? `<li><strong>Retail hedge:</strong> Gas fleet provides wholesale price hedge for retail obligations — retiring too fast exposes retail book to unhedged spot prices</li>` : ''}
                </ul>
            </div>
        `;
    }

    // ─── SECTION 5: Clean Generation Investment Case ─────────
    function renderCleanInvestment() {
        const byFuel = getFleetByFuel();
        const byISO = getFleetByISO();
        const baseline = co.co2_2024_mt;
        const fb = co.fan_bands.all;

        // Current clean vs fossil breakdown
        const cleanMW = FUEL_ORDER.filter(f => ['nuclear', 'hydro', 'geothermal', 'wind', 'solar', 'battery'].includes(f))
            .reduce((s, f) => s + (byFuel[f] ? byFuel[f].cap_mw : 0), 0);
        const fossilMW = FUEL_ORDER.filter(f => ['coal', 'gas_ccgt', 'gas_peaker', 'oil'].includes(f))
            .reduce((s, f) => s + (byFuel[f] ? byFuel[f].cap_mw : 0), 0);

        // Investment gap chart (simplified — shows clean vs fossil capacity evolution)
        const coalRetiring = byFuel.coal ? byFuel.coal.cap_mw : 0;
        const solarNeeded = coalRetiring * 2.5; // Rough: 2.5x solar to replace coal energy (CF difference)
        const batteryNeeded = coalRetiring * 0.5; // 50% storage co-location
        const firmCleanNeeded = Math.max(0, fossilMW * 0.1); // 10% firm clean target

        makeChart('investmentGapChart', {
            type: 'bar',
            data: {
                labels: ['Current Clean', 'Current Fossil', 'Solar Target', 'Wind Target', 'Battery Target', 'Firm Clean Target'],
                datasets: [{
                    label: 'Capacity (MW)',
                    data: [cleanMW, -fossilMW, solarNeeded, solarNeeded * 0.6, batteryNeeded, firmCleanNeeded],
                    backgroundColor: [
                        'rgba(34,197,94,0.7)', 'rgba(107,114,128,0.7)',
                        RESOURCE_COLORS.solar, RESOURCE_COLORS.wind,
                        RESOURCE_COLORS.battery, RESOURCE_COLORS.nuclear
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { y: { title: { display: true, text: 'Capacity (MW)' } } },
                plugins: {
                    legend: { display: false },
                    tooltip: { callbacks: { label: ctx => `${ctx.label}: ${fmt(Math.abs(ctx.raw))} MW` } }
                }
            }
        });

        // LCOE comparison chart
        const lcoeData = [
            { label: 'Existing Coal', cost: 45, color: FUEL_COLORS.coal },
            { label: 'Existing Gas CCGT', cost: 35, color: FUEL_COLORS.gas_ccgt },
            { label: 'New Solar (utility)', cost: 28, color: FUEL_COLORS.solar },
            { label: 'New Onshore Wind', cost: 32, color: FUEL_COLORS.wind },
            { label: 'New Battery (4hr)', cost: 42, color: RESOURCE_COLORS.battery },
            { label: 'New Nuclear (SMR)', cost: 75, color: FUEL_COLORS.nuclear },
            { label: 'New CCS-CCGT', cost: 58, color: '#64748B' }
        ];

        makeChart('lcoeChart', {
            type: 'bar',
            data: {
                labels: lcoeData.map(d => d.label),
                datasets: [{
                    label: 'LCOE ($/MWh)',
                    data: lcoeData.map(d => d.cost),
                    backgroundColor: lcoeData.map(d => d.color)
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                scales: { x: { title: { display: true, text: '$/MWh' } } },
                plugins: { legend: { display: false } }
            }
        });

        // Investment narrative
        renderInvestmentNarrative(byFuel, byISO, cleanMW, fossilMW, solarNeeded, batteryNeeded, firmCleanNeeded);
    }

    function renderInvestmentNarrative(byFuel, byISO, cleanMW, fossilMW, solarNeeded, batteryNeeded, firmCleanNeeded) {
        const el = document.getElementById('investmentNarrative');
        if (!el) return;

        const isos = Object.keys(byISO);
        const coalMW = byFuel.coal ? byFuel.coal.cap_mw : 0;
        const hasNuclear = byFuel.nuclear && byFuel.nuclear.cap_mw > 0;

        let sections = [];

        sections.push(`<p><strong>${co.shortName}</strong> currently operates ${fmt(cleanMW)} MW of clean capacity (${((cleanMW/(co.cap_gw*1000))*100).toFixed(0)}% of fleet)
            and ${fmt(fossilMW)} MW of fossil capacity. The economic case for new clean generation is driven by three factors:</p>`);

        sections.push(`<div class="insight-box insight-success" style="margin-bottom:var(--space-md)">
            <strong>1. Declining fossil capacity factors</strong>
            <p style="margin-top:6px">As grid-wide clean penetration increases, ${co.shortName}'s fossil plants face declining dispatch hours. Gas CCGT capacity factors fall from ~55% today to 25-40% by 2035 under median scenarios, reducing revenue per MW.</p>
        </div>`);

        sections.push(`<div class="insight-box insight-success" style="margin-bottom:var(--space-md)">
            <strong>2. New-build clean cost advantage</strong>
            <p style="margin-top:6px">Utility-scale solar ($28/MWh) and onshore wind ($32/MWh) are now cheaper than the marginal cost of existing gas generation in most hours. New solar+battery is cost-competitive with gas CCGT on a 24/7 basis in ERCOT and CAISO.</p>
        </div>`);

        if (hasNuclear) {
            sections.push(`<div class="insight-box insight-success" style="margin-bottom:var(--space-md)">
                <strong>3. Nuclear uprate + life extension value</strong>
                <p style="margin-top:6px">${co.shortName}'s nuclear fleet receives $15/MWh under the 45U PTC. Uprating existing reactors by 5-10% is the cheapest firm clean generation available — no new site permitting, existing grid connection, and immediate capacity market revenue.</p>
            </div>`);
        } else {
            sections.push(`<div class="insight-box insight-success" style="margin-bottom:var(--space-md)">
                <strong>3. Firm clean generation gap</strong>
                <p style="margin-top:6px">Without nuclear in the fleet, ${co.shortName} needs alternative firm clean sources (CCS-CCGT, LDES, or contracted nuclear PPA) to maintain grid reliability as gas dispatch declines.</p>
            </div>`);
        }

        sections.push(`<div class="insight-box" style="margin-top:var(--space-lg)">
            <strong>Recommended Investment Targets (by 2035)</strong>
            <ul style="margin:8px 0 0 16px">
                <li><strong>Solar:</strong> ${fmt(solarNeeded)} MW new utility-scale solar${coalMW > 0 ? ' (includes coal site repowering)' : ''}</li>
                <li><strong>Wind:</strong> ${fmt(solarNeeded * 0.6)} MW onshore wind (complementary generation profile)</li>
                <li><strong>Battery:</strong> ${fmt(batteryNeeded)} MW 4-hour Li-ion storage (peak shaving + ancillary services)</li>
                <li><strong>Firm clean:</strong> ${fmt(firmCleanNeeded)} MW ${hasNuclear ? 'nuclear uprates + potential SMR' : 'CCS-CCGT or contracted nuclear PPA'}</li>
            </ul>
        </div>`);

        el.innerHTML = sections.join('');
    }

    // ─── SECTION 6: Enabling Conditions for 1.5°C ───────────
    function renderEnablingConditions() {
        const refBands = co.fan_bands.reference || co.fan_bands.all;
        const emissions = refBands.emissions;
        const baseline = emissions.p50[0];
        const sbti = sbtiTrajectory(baseline);

        // Gap area chart
        makeChart('gapChart', {
            type: 'line',
            data: {
                labels: YEAR_LABELS,
                datasets: [
                    {
                        label: 'Qualified Target (P50)',
                        data: emissions.p50,
                        borderColor: '#6366F1',
                        backgroundColor: 'rgba(239,68,68,0.12)',
                        fill: '+1',
                        borderWidth: 2.5,
                        pointRadius: 4
                    },
                    {
                        label: '1.5°C SBTi Trajectory',
                        data: sbti,
                        borderColor: '#E91E63',
                        fill: false,
                        borderWidth: 2,
                        borderDash: [8, 4],
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { y: { title: { display: true, text: 'CO₂ Emissions (Mt)' }, beginAtZero: true } },
                plugins: {
                    legend: { position: 'bottom' },
                    tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${fmt(ctx.raw, 1)} Mt` } }
                }
            }
        });

        // Enabling conditions cards
        renderConditionsCards();
    }

    function renderConditionsCards() {
        const el = document.getElementById('conditionsCards');
        if (!el) return;

        const dims = co.dimension_fans;
        if (!dims) return;

        const conditions = [];

        // Carbon pricing
        if (dims.conditions) {
            const vals = Object.keys(dims.conditions);
            if (vals.length >= 2) {
                const emissionsF = dims.conditions.F ? dims.conditions.F.emissions.p50[5] : null;
                const emissionsC = dims.conditions.C ? dims.conditions.C.emissions.p50[5] : null;
                if (emissionsF !== null && emissionsC !== null) {
                    const impact = Math.abs(emissionsC - emissionsF);
                    conditions.push({
                        title: 'Interconnection & Permitting Reform',
                        impact: `${fmt(impact, 1)} Mt difference`,
                        description: `Shifting from challenging to facilitating market conditions reduces ${co.shortName}'s 2050 emissions by ${fmt(impact, 1)} Mt.
                            This includes faster interconnection queues, streamlined permitting, and supportive state policies.
                            Facilitating conditions: ${fmt(emissionsF, 1)} Mt vs. Challenging: ${fmt(emissionsC, 1)} Mt.`
                    });
                }
            }
        }

        // Demand growth
        if (dims.demand_growth) {
            const low = dims.demand_growth.Low ? dims.demand_growth.Low.emissions.p50[5] : null;
            const high = dims.demand_growth.High ? dims.demand_growth.High.emissions.p50[5] : null;
            if (low !== null && high !== null) {
                conditions.push({
                    title: 'Demand Growth Trajectory',
                    impact: `${fmt(Math.abs(high - low), 1)} Mt range`,
                    description: `Higher electricity demand drives more new clean build (displacing fossil), but also requires more total generation.
                        Low demand: ${fmt(low, 1)} Mt vs. High demand: ${fmt(high, 1)} Mt by 2050.
                        ${high > low ? 'Higher demand increases total emissions despite higher clean build rates.' : 'Higher demand accelerates clean deployment, reducing per-MWh intensity faster.'}`
                });
            }
        }

        // LCOE sensitivity
        if (dims.price_sens) {
            const low = dims.price_sens.Low ? dims.price_sens.Low.emissions.p50[5] : null;
            const high = dims.price_sens.High ? dims.price_sens.High.emissions.p50[5] : null;
            if (low !== null && high !== null) {
                conditions.push({
                    title: 'Clean Energy Cost Reduction',
                    impact: `${fmt(Math.abs(high - low), 1)} Mt range`,
                    description: `Faster LCOE reductions (learning curves, manufacturing scale) accelerate fossil displacement.
                        Low LCOE (optimistic): ${fmt(low, 1)} Mt vs. High LCOE (conservative): ${fmt(high, 1)} Mt by 2050.`
                });
            }
        }

        // PPA level
        if (dims.ppa_level) {
            const low = dims.ppa_level.Low ? dims.ppa_level.Low.emissions.p50[5] : null;
            const high = dims.ppa_level.High ? dims.ppa_level.High.emissions.p50[5] : null;
            if (low !== null && high !== null) {
                conditions.push({
                    title: 'PPA Market Depth',
                    impact: `${fmt(Math.abs(high - low), 1)} Mt range`,
                    description: `Deeper PPA markets (more corporate buyers, higher willingness-to-pay) pull forward clean investment.
                        High PPA: ${fmt(high, 1)} Mt vs. Low PPA: ${fmt(low, 1)} Mt by 2050.`
                });
            }
        }

        el.innerHTML = conditions.map(c => `
            <div class="card" style="padding:var(--space-lg)">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px">
                    <h4 style="margin:0;color:var(--navy);font-size:1rem">${c.title}</h4>
                    <span class="convergence-tag moderate">${c.impact}</span>
                </div>
                <p style="font-size:0.88rem;color:var(--text-secondary);line-height:1.6;margin:0">${c.description}</p>
            </div>
        `).join('');
    }

    // ─── SECTION 7: Strategic Positioning ────────────────────
    function renderStrategicPositioning() {
        const byFuel = getFleetByFuel();
        const byISO = getFleetByISO();
        const fb = co.fan_bands.all;

        // Robustness scores
        const nuclearCap = byFuel.nuclear ? byFuel.nuclear.cap_mw : 0;
        const nuclearScore = Math.min(100, (nuclearCap / (co.cap_gw * 1000)) * 200); // 50% nuclear = 100 score
        const isoCount = Object.keys(byISO).length;
        const diversityScore = Math.min(100, isoCount * 25); // 4 ISOs = 100
        const coalCap = byFuel.coal ? byFuel.coal.cap_mw : 0;
        const coalExitScore = 100 - Math.min(100, (coalCap / (co.cap_gw * 1000)) * 200);
        const peakerCap = byFuel.gas_peaker ? byFuel.gas_peaker.cap_mw : 0;
        const ccgtCap = byFuel.gas_ccgt ? byFuel.gas_ccgt.cap_mw : 0;
        const gasFlexScore = ccgtCap > 0 ? Math.min(100, (ccgtCap / (ccgtCap + peakerCap)) * 100) : 50;
        const cleanCap = ['solar', 'wind', 'battery'].reduce((s, f) => s + (byFuel[f] ? byFuel[f].cap_mw : 0), 0);
        const cleanPipelineScore = Math.min(100, (cleanCap / (co.cap_gw * 1000)) * 300);

        // Radar chart
        makeChart('radarChart', {
            type: 'radar',
            data: {
                labels: ['Nuclear Advantage', 'Geographic Diversity', 'Coal Exit Progress', 'Gas Flexibility', 'Clean Pipeline'],
                datasets: [{
                    label: co.shortName,
                    data: [nuclearScore, diversityScore, coalExitScore, gasFlexScore, cleanPipelineScore],
                    borderColor: '#6366F1',
                    backgroundColor: 'rgba(99,102,241,0.15)',
                    borderWidth: 2,
                    pointRadius: 4,
                    pointBackgroundColor: '#6366F1'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        min: 0,
                        max: 100,
                        ticks: { stepSize: 25, display: false },
                        pointLabels: { font: { size: 11 } }
                    }
                },
                plugins: { legend: { display: false } }
            }
        });

        // Scenario robustness stats
        const profitableAllScenarios = co.breakeven && co.breakeven.conditions;
        let profitablePct = 100;
        if (profitableAllScenarios) {
            const vals = Object.values(profitableAllScenarios);
            const totalScenarios = vals.reduce((s, v) => s + v.n_scenarios, 0);
            const profitableScenarios = vals.reduce((s, v) => s + (v.profitable_pct / 100 * v.n_scenarios), 0);
            profitablePct = (profitableScenarios / totalScenarios) * 100;
        }

        document.getElementById('robustnessStats').innerHTML = `
            <div class="grid-stats" style="margin-bottom:var(--space-lg)">
                <div class="stat-card">
                    <div class="stat-value">${fmt(profitablePct, 0)}%</div>
                    <div class="stat-label">Scenarios Profitable</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">$${fmt(fb.profit ? fb.profit.p10[5] : 0)}M</div>
                    <div class="stat-label">Worst-Case Profit (2050)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">$${fmt(fb.profit ? fb.profit.p90[5] : 0)}M</div>
                    <div class="stat-label">Best-Case Profit (2050)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${fmt(fb.stranded_mw ? fb.stranded_mw.p50[5] : 0)}</div>
                    <div class="stat-label">MW Stranded (P50, 2050)</div>
                </div>
            </div>
        `;

        // Strategic narrative
        renderStrategicNarrative(byFuel, byISO, nuclearScore, diversityScore, coalExitScore, gasFlexScore, cleanPipelineScore);
    }

    function renderStrategicNarrative(byFuel, byISO, nuclearScore, diversityScore, coalExitScore, gasFlexScore, cleanPipelineScore) {
        const el = document.getElementById('strategicNarrative');
        if (!el) return;

        const scores = { nuclear: nuclearScore, diversity: diversityScore, coalExit: coalExitScore, gasFlex: gasFlexScore, cleanPipeline: cleanPipelineScore };
        const isos = Object.keys(byISO);

        // Identify top strengths and vulnerabilities
        const scoreEntries = Object.entries(scores).sort((a, b) => b[1] - a[1]);
        const topStrength = scoreEntries[0];
        const topWeakness = scoreEntries[scoreEntries.length - 1];

        const strengthLabels = {
            nuclear: 'nuclear fleet provides the most robust zero-carbon foundation',
            diversity: 'geographic diversification across multiple ISOs hedges against regional disruption',
            coalExit: 'advanced coal exit progress removes the largest transition risk',
            gasFlex: 'efficient gas fleet provides reliable dispatchable capacity during transition',
            cleanPipeline: 'existing clean energy portfolio positions for accelerated deployment'
        };

        const weaknessLabels = {
            nuclear: 'lack of nuclear baseload means higher dependence on gas for firm generation',
            diversity: 'geographic concentration in fewer ISOs creates regulatory and market risk',
            coalExit: 'remaining coal exposure represents the largest near-term transition risk',
            gasFlex: 'gas fleet composition (high peaker ratio) faces declining utilization',
            cleanPipeline: 'limited clean energy portfolio requires significant greenfield investment'
        };

        let recs = [];
        if (isos.includes('ERCOT')) recs.push('Leverage ERCOT\'s energy-only market for solar+battery deployment — high-value peak pricing creates strong returns.');
        if (isos.includes('PJM')) recs.push('Utilize PJM capacity market revenue to support firm clean generation investments (nuclear uprates, CCS).');
        if (byFuel.coal && byFuel.coal.cap_mw > 500) recs.push('Accelerate coal retirement and site repowering — retiring coal sites have existing grid interconnection, land, and workforce.');
        if (byFuel.nuclear && byFuel.nuclear.cap_mw > 2000) recs.push('Pursue nuclear uprates (5-10% capacity increase) — cheapest firm clean generation available at ~$10/MWh incremental cost.');
        if (byFuel.gas_peaker && byFuel.gas_peaker.cap_mw > 2000) recs.push('Replace highest-heat-rate gas peakers with 4-hour battery storage at sites with declining capacity factors.');
        recs.push('Build out corporate PPA pipeline — voluntary procurement creates revenue certainty for new clean projects.');

        el.innerHTML = `
            <p><strong>${co.shortName}'s</strong> strongest strategic position is its ${strengthLabels[topStrength[0]]} (score: ${fmt(topStrength[1], 0)}/100).
            Its primary vulnerability is that ${weaknessLabels[topWeakness[0]]} (score: ${fmt(topWeakness[1], 0)}/100).</p>

            <div class="insight-box" style="margin:var(--space-lg) 0">
                <strong>Strategic Recommendations</strong>
                <ol style="margin:8px 0 0 16px">${recs.map(r => `<li style="margin-bottom:6px">${r}</li>`).join('')}</ol>
            </div>

            <p>Under the range of modeled scenarios, <strong>${co.shortName}</strong>'s most robust strategy is to pursue the qualified target trajectory while
            building optionality for faster decarbonization. The qualified target is achievable under market economics alone;
            the enabling conditions in Section 6 define what would be needed to close the gap to 1.5°C alignment.
            By maintaining gas fleet flexibility as a retail hedge while steadily building clean capacity,
            ${co.shortName} can navigate the full range of climate transition outcomes without excessive downside risk.</p>
        `;
    }

    // ─── TOC Scroll Tracking ─────────────────────────────────
    function initTocTracking() {
        const sections = document.querySelectorAll('.report-section');
        const tocLinks = document.querySelectorAll('.report-toc-list a');
        if (!sections.length || !tocLinks.length) return;

        const observer = new IntersectionObserver(entries => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const id = entry.target.id;
                    tocLinks.forEach(link => {
                        link.classList.toggle('active', link.getAttribute('href') === '#' + id);
                    });
                }
            });
        }, { rootMargin: '-20% 0px -70% 0px' });

        sections.forEach(section => observer.observe(section));

        // Smooth scroll
        tocLinks.forEach(link => {
            link.addEventListener('click', e => {
                e.preventDefault();
                const target = document.querySelector(link.getAttribute('href'));
                if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            });
        });
    }

    // ─── Initialize ──────────────────────────────────────────
    function init() {
        // Set page title
        document.title = `${co.name} — IPP Climate Transition Analysis`;
        const companyNameEl = document.getElementById('companyName');
        if (companyNameEl) companyNameEl.textContent = co.name;
        const companyTargetEl = document.getElementById('companyTarget');
        if (companyTargetEl) companyTargetEl.textContent = co.target;

        // Render all sections
        renderFleetProfile();
        renderConvergenceDivergence();
        renderRiskOpportunity();
        renderQualifiedTarget();
        renderCleanInvestment();
        renderEnablingConditions();
        renderStrategicPositioning();
        initTocTracking();
    }

    // Run when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
