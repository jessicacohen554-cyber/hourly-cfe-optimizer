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

    // SBTi Power Sector v2: Net zero by 2040 (linear decline 2023→2040)
    const SBTI_POWER_V2_REMAINING = { 2023: 1.00, 2030: 0.412, 2035: 0.118, 2040: 0.0, 2045: 0.0, 2050: 0.0 };

    // EPRI Aspirational Target (AT): −57/−82/−88/−94/−100% from 2023
    const AT_TRAJECTORY_REMAINING = { 2023: 1.00, 2030: 0.43, 2035: 0.18, 2040: 0.12, 2045: 0.06, 2050: 0.00 };

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

    function powerSectorV2Trajectory(baseline) {
        return YEARS.map(y => baseline * (SBTI_POWER_V2_REMAINING[y] || 0));
    }

    function atTrajectory(baseline) {
        return YEARS.map(y => baseline * (AT_TRAJECTORY_REMAINING[y] || 0));
    }

    function pctFromBaseline(value, baseline) {
        if (!baseline || baseline === 0) return '';
        const pct = ((value - baseline) / baseline * 100);
        return ` (${pct >= 0 ? '+' : ''}${pct.toFixed(1)}% vs 2023)`;
    }

    // Convert a hex or named color to rgba with given alpha
    function toRGBA(color, alpha) {
        if (!color || typeof color !== 'string') return color;
        // Already rgba — adjust alpha
        if (color.startsWith('rgba(')) {
            return color.replace(/,\s*[\d.]+\)$/, ', ' + alpha + ')');
        }
        if (color.startsWith('rgb(')) {
            return color.replace('rgb(', 'rgba(').replace(')', ', ' + alpha + ')');
        }
        // Hex to rgba
        if (color.startsWith('#')) {
            var hex = color.slice(1);
            if (hex.length === 3) hex = hex[0]+hex[0]+hex[1]+hex[1]+hex[2]+hex[2];
            var r = parseInt(hex.slice(0,2), 16);
            var g = parseInt(hex.slice(2,4), 16);
            var b = parseInt(hex.slice(4,6), 16);
            return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
        }
        return color;
    }

    // Apply glass-morphism chart styling: semi-transparent fills + saturated borders
    function applyChartStyle(config) {
        if (!config || !config.data || !config.data.datasets) return config;
        var chartType = config.type;
        config.data.datasets.forEach(function(ds) {
            // Skip datasets that explicitly opt out
            if (ds._skipStyle) return;
            if (chartType === 'doughnut' || chartType === 'pie') {
                // Doughnut/pie: semi-transparent fills with saturated borders
                if (Array.isArray(ds.backgroundColor)) {
                    ds.borderColor = ds.backgroundColor.map(function(c) { return c; });
                    ds.backgroundColor = ds.backgroundColor.map(function(c) { return toRGBA(c, 0.3); });
                }
                ds.borderWidth = ds.borderWidth || 2;
            } else if (chartType === 'radar') {
                // Radar: keep border saturated, make fill semi-transparent
                if (ds.backgroundColor && !Array.isArray(ds.backgroundColor)) {
                    ds.backgroundColor = toRGBA(ds.borderColor || ds.backgroundColor, 0.2);
                }
                ds.borderWidth = ds.borderWidth || 2;
            } else if (chartType === 'bar') {
                // Bars: semi-transparent fill + saturated border
                if (ds.backgroundColor && !Array.isArray(ds.backgroundColor)) {
                    if (!ds.borderColor || ds.borderColor === '#fff') {
                        ds.borderColor = ds.backgroundColor.startsWith('rgba') ?
                            ds.backgroundColor.replace(/,\s*[\d.]+\)$/, ', 1)') : ds.backgroundColor;
                    }
                    ds.backgroundColor = toRGBA(ds.borderColor, 0.25);
                } else if (Array.isArray(ds.backgroundColor)) {
                    ds.borderColor = ds.backgroundColor.map(function(c) {
                        return c.startsWith && c.startsWith('rgba') ? c.replace(/,\s*[\d.]+\)$/, ', 1)') : c;
                    });
                    ds.backgroundColor = ds.backgroundColor.map(function(c) { return toRGBA(c, 0.25); });
                }
                ds.borderWidth = ds.borderWidth || 1.5;
            } else if (chartType === 'line') {
                // Lines: saturated border, semi-transparent fill if fill is true
                if (ds.borderColor) {
                    var saturated = ds.borderColor.startsWith && ds.borderColor.startsWith('rgba') ?
                        ds.borderColor.replace(/,\s*[\d.]+\)$/, ', 1)') : ds.borderColor;
                    ds.borderColor = saturated;
                }
                if (ds.fill !== false && ds.fill !== undefined) {
                    ds.backgroundColor = toRGBA(ds.borderColor || '#6366F1', 0.15);
                }
                ds.borderWidth = ds.borderWidth || 2;
            }
        });
        return config;
    }

    function makeChart(canvasId, config) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return null;
        if (charts[canvasId]) { charts[canvasId].destroy(); }
        applyChartStyle(config);
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

        // Stats — generation-based (TWh tells the real story, not nameplate MW)
        const totalGen = co.gen_twh;
        const totalCO2 = co.co2_2024_mt;
        const intensity = co.intensity_kg;
        const cleanGen = FUEL_ORDER.filter(f => ['nuclear','hydro','geothermal','wind','solar','battery'].includes(f))
            .reduce((s, f) => s + (byFuel[f] ? byFuel[f].gen_twh : 0), 0);
        const cleanPct = totalGen > 0 ? ((cleanGen / totalGen) * 100).toFixed(0) : '0';

        // Inject stats
        document.getElementById('statCapacity').textContent = fmt(totalGen, 0) + ' TWh';
        document.getElementById('statGeneration').textContent = fmt(totalGen, 0) + ' TWh';
        document.getElementById('statEmissions').textContent = fmt(totalCO2, 0) + ' Mt';
        document.getElementById('statIntensity').textContent = fmt(intensity, 0) + ' kg/MWh';
        document.getElementById('statISOs').textContent = isos.length;
        document.getElementById('statCleanPct').textContent = cleanPct + '%';

        // Doughnut: generation by fuel (TWh)
        const doughnutFuels = FUEL_ORDER.filter(f => byFuel[f] && byFuel[f].gen_twh > 0);
        makeChart('fleetDoughnutChart', {
            type: 'doughnut',
            data: {
                labels: doughnutFuels.map(f => FUEL_LABELS[f]),
                datasets: [{
                    data: doughnutFuels.map(f => byFuel[f].gen_twh),
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
                            label: (ctx) => `${ctx.label}: ${fmt(ctx.raw, 1)} TWh (${(totalGen > 0 ? (ctx.raw / totalGen * 100) : 0).toFixed(1)}%)`
                        }
                    }
                }
            }
        });

        // Horizontal stacked bar: generation by ISO (TWh)
        const isoOrder = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP'].filter(i => byISO[i]);
        const isoFuels = FUEL_ORDER.filter(f => {
            return isoOrder.some(iso => co.fleet_summary[iso] && co.fleet_summary[iso][f] && (co.fleet_summary[iso][f].gen_twh || 0) > 0);
        });

        makeChart('isoBarChart', {
            type: 'bar',
            data: {
                labels: isoOrder,
                datasets: isoFuels.map(f => ({
                    label: FUEL_LABELS[f],
                    data: isoOrder.map(iso => (co.fleet_summary[iso] && co.fleet_summary[iso][f]) ? (co.fleet_summary[iso][f].gen_twh || 0) : 0),
                    backgroundColor: FUEL_COLORS[f]
                }))
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { stacked: true, title: { display: true, text: 'Generation (TWh)' } },
                    y: { stacked: true }
                },
                plugins: {
                    legend: { position: 'bottom', labels: { font: { size: 10 }, boxWidth: 12 } },
                    tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${fmt(ctx.raw, 1)} TWh` } }
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
        const totalGen = co.gen_twh;
        const hasCoal = byFuel.coal && byFuel.coal.gen_twh > 0;
        const hasNuclear = byFuel.nuclear && byFuel.nuclear.gen_twh > 0;
        const nuclearGen = hasNuclear ? byFuel.nuclear.gen_twh : 0;
        const nuclearPct = totalGen > 0 ? ((nuclearGen / totalGen) * 100).toFixed(0) : 0;
        const coalGen = hasCoal ? byFuel.coal.gen_twh : 0;
        const coalPct = totalGen > 0 ? ((coalGen / totalGen) * 100).toFixed(0) : 0;
        const gasCCGTGen = byFuel.gas_ccgt ? byFuel.gas_ccgt.gen_twh : 0;
        const gasPeakerGen = byFuel.gas_peaker ? byFuel.gas_peaker.gen_twh : 0;
        const renewGen = (byFuel.solar ? byFuel.solar.gen_twh : 0) + (byFuel.wind ? byFuel.wind.gen_twh : 0);
        const batteryGen = byFuel.battery ? byFuel.battery.gen_twh : 0;

        let strengths = [];
        let weaknesses = [];
        let opportunities = [];

        // Strengths
        if (hasNuclear && nuclearPct > 20) strengths.push(`Nuclear generation (${nuclearPct}% of output, ${fmt(nuclearGen, 1)} TWh) provides zero-carbon firm baseload and strong capacity market revenue.`);
        if (isos.length >= 3) strengths.push(`Geographic diversification across ${isos.length} ISOs (${isos.join(', ')}) hedges against regional regulatory and market risk.`);
        if (renewGen > 5) strengths.push(`Existing renewable generation (${fmt(renewGen, 1)} TWh solar+wind) provides foundation for scaling clean output.`);
        if (batteryGen > 0.5) strengths.push(`Battery storage dispatch (${fmt(batteryGen, 1)} TWh) positions for flexible grid services and peak shaving.`);
        if (gasCCGTGen > 10) strengths.push(`Efficient gas CCGT fleet (${fmt(gasCCGTGen, 1)} TWh) provides reliable dispatchable generation with moderate emissions intensity.`);
        if (cleanPct > 50) strengths.push(`Already ${cleanPct}% clean by generation, ahead of most peers in decarbonization.`);

        // Weaknesses
        if (hasCoal) weaknesses.push(`Coal generation (${coalPct}% of output, ${fmt(coalGen, 1)} TWh, ${fmt(byFuel.coal.co2_mt, 1)} Mt CO₂) represents the highest transition risk and stranded asset potential.`);
        if (isos.length === 1) weaknesses.push(`Single-ISO concentration (${isos[0]}) creates regulatory and market concentration risk.`);
        if (co.intensity_kg > 400) weaknesses.push(`High fleet intensity (${co.intensity_kg} kg/MWh), among the most carbon-intensive large generators.`);
        if (gasPeakerGen > 3) weaknesses.push(`Gas peaker generation (${fmt(gasPeakerGen, 1)} TWh) faces declining utilization as battery storage scales.`);
        if (cleanPct < 20 && !hasNuclear) weaknesses.push(`Very low clean generation share (${cleanPct}%) with no nuclear hedge.`);

        // Opportunities
        if (hasCoal) opportunities.push(`Coal retirement + site repowering with solar/battery co-location can leverage existing grid interconnection and land.`);
        if (gasPeakerGen > 1) opportunities.push(`Gas peaker → battery replacement at highest-heat-rate sites captures cost savings and emissions reduction.`);
        if (hasNuclear) opportunities.push(`Nuclear uprates and life extensions leverage highest-value zero-carbon asset under 45U PTC ($15/MWh).`);
        if (renewGen < 3) opportunities.push(`Significant greenfield renewable opportunity: current clean generation is undersized relative to fleet.`);
        if (isos.some(i => i === 'ERCOT')) opportunities.push(`ERCOT market exposure provides high-value solar+battery opportunity due to peak pricing dynamics.`);
        if (isos.some(i => i === 'PJM')) opportunities.push(`PJM capacity market provides revenue stability for new firm clean generation investments.`);

        if (strengths.length === 0) strengths.push(`Fleet provides reliable dispatchable generation.`);
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
        const fb = co.fan_bands.reference;
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
                            label: ctx => {
                                const bl = co.co2_2024_mt || co.co2_2023_mt;
                                return `${ctx.dataset.label}: ${fmt(ctx.raw, 1)} Mt${pctFromBaseline(ctx.raw, bl)}`;
                            }
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
        const fb = co.fan_bands.reference;

        // Profit fan band — % growth in fleet profit margin vs 2023 baseline
        if (fb.profit) {
            const profitBaseline = fb.profit.p50[0] || 1; // 2023 baseline
            const toPctGrowth = arr => arr.map(v => ((v - profitBaseline) / Math.abs(profitBaseline)) * 100);

            makeChart('profitFanChart', {
                type: 'line',
                data: {
                    labels: YEAR_LABELS,
                    datasets: [
                        {
                            label: 'P90 (upside)',
                            data: toPctGrowth(fb.profit.p90),
                            borderColor: 'rgba(34,197,94,0.6)',
                            backgroundColor: 'rgba(34,197,94,0.1)',
                            fill: '+1',
                            borderWidth: 1.5,
                            borderDash: [4, 3],
                            pointRadius: 2
                        },
                        {
                            label: 'P50 (median)',
                            data: toPctGrowth(fb.profit.p50),
                            borderColor: '#0F172A',
                            backgroundColor: 'rgba(99,102,241,0.08)',
                            fill: '+1',
                            borderWidth: 2.5,
                            pointRadius: 4,
                            pointBackgroundColor: '#0F172A'
                        },
                        {
                            label: 'P10 (downside)',
                            data: toPctGrowth(fb.profit.p10),
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
                        y: {
                            title: { display: true, text: '% Change in Fleet Profit Margin vs 2023' },
                            ticks: { callback: v => v + '%' }
                        }
                    },
                    plugins: {
                        legend: { position: 'bottom' },
                        tooltip: {
                            callbacks: {
                                label: ctx => {
                                    const pct = ctx.raw;
                                    const absVal = profitBaseline * (1 + pct / 100);
                                    return `${ctx.dataset.label}: ${pct >= 0 ? '+' : ''}${fmt(pct, 1)}% ($${fmt(absVal)}M)`;
                                }
                            }
                        }
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

        const fb = co.fan_bands.reference;
        const profitBase = fb.profit ? fb.profit.p50[0] : 1;
        const profit2050_p10 = fb.profit ? fb.profit.p10[5] : 0;
        const profit2050_p50 = fb.profit ? fb.profit.p50[5] : 0;
        const profit2050_p90 = fb.profit ? fb.profit.p90[5] : 0;
        const pctGrowth_p10 = ((profit2050_p10 - profitBase) / Math.abs(profitBase) * 100);
        const pctGrowth_p50 = ((profit2050_p50 - profitBase) / Math.abs(profitBase) * 100);
        const pctGrowth_p90 = ((profit2050_p90 - profitBase) / Math.abs(profitBase) * 100);
        const stranded2050 = fb.stranded_mw ? fb.stranded_mw.p50[5] : 0;
        const operating2050 = fb.operating_mw ? fb.operating_mw.p50[5] : 0;

        const byFuel = getFleetByFuel();
        const hasCoal = byFuel.coal && byFuel.coal.gen_twh > 0;
        const coalGen = hasCoal ? byFuel.coal.gen_twh : 0;

        let riskItems = [];
        if (hasCoal) riskItems.push(`<strong>Coal stranding risk:</strong> ${fmt(coalGen, 1)} TWh of coal generation (${fmt(byFuel.coal.cap_mw)} MW nameplate) faces retirement pressure. Under median scenarios, ${fmt(stranded2050)} MW total becomes stranded by 2050.`);
        if (profit2050_p10 < 0) riskItems.push(`<strong>Downside scenario loss:</strong> In the P10 (worst-case) scenario, annual profit falls to $${fmt(profit2050_p10)}M by 2050. The fleet loses money under adverse conditions.`);
        if (profit2050_p10 > 0) riskItems.push(`<strong>Profit resilience:</strong> Even in the P10 (worst-case) scenario, the fleet maintains $${fmt(profit2050_p10)}M annual profit by 2050, with no money-losing scenarios.`);

        riskItems.push(`<strong>Profit margin trajectory:</strong> 2050 fleet profit margin ranges from ${pctGrowth_p10 >= 0 ? '+' : ''}${fmt(pctGrowth_p10, 0)}% (P10) to ${pctGrowth_p90 >= 0 ? '+' : ''}${fmt(pctGrowth_p90, 0)}% (P90) vs 2023 baseline, with a median of ${pctGrowth_p50 >= 0 ? '+' : ''}${fmt(pctGrowth_p50, 0)}% ($${fmt(profit2050_p50)}M).`);

        // Retail hedge analysis
        const gasGen = (byFuel.gas_ccgt ? byFuel.gas_ccgt.gen_twh : 0) + (byFuel.gas_peaker ? byFuel.gas_peaker.gen_twh : 0);
        if (gasGen > 5) {
            riskItems.push(`<strong>Retail hedge position:</strong> ${fmt(gasGen, 1)} TWh of gas generation provides a natural hedge for retail power obligations. When wholesale prices spike (gas on the margin), generation revenue offsets retail exposure. This hedge value declines as gas dispatch falls under higher CFE thresholds.`);
        }

        el.innerHTML = riskItems.map(r => `<p>${r}</p>`).join('');
    }

    // ─── SECTION 4: Qualified Target Proposal ────────────────
    function renderQualifiedTarget() {
        const refBands = co.fan_bands.reference;
        if (!refBands || !refBands.emissions) { console.warn('No reference fan bands for QT'); return; }
        const emissions = refBands.emissions;
        const baseline = emissions.p50[0];

        // Compute qualified targets (P50 reference trajectory)
        const qualifiedPcts = emissions.p50.map(v => ((baseline - v) / baseline) * 100);
        const psV2 = powerSectorV2Trajectory(baseline);
        const at = atTrajectory(baseline);

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
                        label: 'SBTi Power Sector v2 (NZ 2040)',
                        data: psV2,
                        borderColor: '#E91E63',
                        borderWidth: 2,
                        borderDash: [8, 4],
                        pointRadius: 0,
                        fill: false
                    },
                    {
                        label: 'SMARTargets AT (−57/−82/−100%)',
                        data: at,
                        borderColor: '#9C27B0',
                        borderWidth: 2,
                        borderDash: [4, 6],
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
                    tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${fmt(ctx.raw, 1)} Mt${pctFromBaseline(ctx.raw, baseline)}` } }
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
                const psV2Mt = psV2[m.idx];
                const atMt = at[m.idx];
                const gapPS = mt - psV2Mt;
                const gapAT = mt - atMt;
                return `<div class="stat-card">
                    <div class="stat-value">${fmt(pct, 0)}%</div>
                    <div class="stat-label">${m.year} reduction</div>
                    <div style="font-size:0.75rem;color:var(--text-muted);margin-top:4px">
                        ${fmt(mt, 1)} Mt<br>
                        <span style="color:${gapPS > 0.5 ? '#dc2626' : '#15803d'}">Gap to NZ 2040: ${gapPS > 0 ? '+' : ''}${fmt(gapPS, 1)} Mt</span><br>
                        <span style="color:${gapAT > 0.5 ? '#dc2626' : '#15803d'}">Gap to AT: ${gapAT > 0 ? '+' : ''}${fmt(gapAT, 1)} Mt</span>
                    </div>
                </div>`;
            }).join('');
        }

        // Target justification narrative
        renderTargetNarrative(emissions, qualifiedPcts);
    }

    function renderTargetNarrative(emissions, qualifiedPcts) {
        const el = document.getElementById('targetNarrative');
        if (!el) return;

        const baseline = emissions.p50[0];
        const target2030 = emissions.p50[1];
        const target2035 = emissions.p50[2];
        const target2040 = emissions.p50[3];
        const target2050 = emissions.p50[5];
        const psV2_local = powerSectorV2Trajectory(baseline);
        const at_local = atTrajectory(baseline);
        const gap2040_ps = target2040 - psV2_local[3]; // gap to NZ 2040
        const gap2050_at = target2050 - at_local[5]; // gap to AT 2050

        const byFuel = getFleetByFuel();
        const hasCoal = byFuel.coal && byFuel.coal.gen_twh > 0;
        const hasNuclear = byFuel.nuclear && byFuel.nuclear.gen_twh > 0;
        const gasGen = (byFuel.gas_ccgt ? byFuel.gas_ccgt.gen_twh : 0) + (byFuel.gas_peaker ? byFuel.gas_peaker.gen_twh : 0);

        el.innerHTML = `
            <p>The proposed qualified target for <strong>${co.shortName}</strong> follows the P50 (median) emission trajectory across
            all reference scenarios, representing what market forces, existing policy, and announced retirements will organically deliver
            without additional mandates or subsidies.</p>

            <p><strong>2030 target: ${fmt(qualifiedPcts[1], 0)}% reduction</strong> (${fmt(target2030, 1)} Mt from ${fmt(baseline, 1)} Mt baseline).
            ${hasCoal ? 'Primarily driven by announced coal retirements and declining coal dispatch economics.' : 'Driven by declining fossil dispatch as clean generation enters the market.'}</p>

            <p><strong>2035 target: ${fmt(qualifiedPcts[2], 0)}% reduction</strong> (${fmt(target2035, 1)} Mt).
            ${hasNuclear ? 'Nuclear fleet continues at high capacity factors, providing stable zero-carbon baseload.' : ''}
            Gas CCGT dispatch declines moderately as renewable generation increases across ${co.shortName}'s operating ISOs.</p>

            <p><strong>2040 target: ${fmt(qualifiedPcts[3], 0)}% reduction</strong> (${fmt(target2040, 1)} Mt).
            ${gap2040_ps > 0.5 ? `This leaves a ${fmt(gap2040_ps, 1)} Mt gap to the SBTi Power Sector v2 net-zero-by-2040 trajectory.` :
            'This meets or exceeds the SBTi Power Sector v2 net-zero-by-2040 trajectory.'}</p>

            <p><strong>2050 target: ${fmt(qualifiedPcts[5], 0)}% reduction</strong> (${fmt(target2050, 1)} Mt).
            ${gap2050_at > 0.5 ? `This leaves a ${fmt(gap2050_at, 1)} Mt gap to the SMARTargets AT trajectory. Bridging this gap requires the enabling conditions described in Section 7.` :
            'This meets or exceeds the SMARTargets AT trajectory; no additional enabling conditions required.'}</p>

            <div class="insight-box" style="margin-top:var(--space-md)">
                <strong>Why this target is achievable:</strong>
                <ul style="margin:8px 0 0 16px">
                    <li><strong>Reliability:</strong> Maintains sufficient firm dispatchable generation (${fmt(gasGen, 0)} TWh gas output) for grid adequacy</li>
                    <li><strong>Affordability:</strong> Follows market economics with no cross-subsidy or above-market procurement required</li>
                    <li><strong>Cost competitiveness:</strong> Aligned with market deployment curve; new clean investments at or below market clearing prices</li>
                    ${gasGen > 5 ? `<li><strong>Retail hedge:</strong> Gas fleet provides wholesale price hedge for retail obligations. Retiring too fast exposes retail book to unhedged spot prices</li>` : ''}
                </ul>
            </div>
        `;
    }

    // ─── SECTION 5: Clean Generation Investment Case ─────────
    function renderCleanInvestment() {
        const byFuel = getFleetByFuel();
        const byISO = getFleetByISO();
        const baseline = co.co2_2024_mt;
        const fb = co.fan_bands.reference;

        // Current clean vs fossil breakdown (by generation TWh)
        const cleanTWh = FUEL_ORDER.filter(f => ['nuclear', 'hydro', 'geothermal', 'wind', 'solar', 'battery'].includes(f))
            .reduce((s, f) => s + (byFuel[f] ? byFuel[f].gen_twh : 0), 0);
        const fossilTWh = FUEL_ORDER.filter(f => ['coal', 'gas_ccgt', 'gas_peaker', 'oil'].includes(f))
            .reduce((s, f) => s + (byFuel[f] ? byFuel[f].gen_twh : 0), 0);

        // Investment gap chart — based on total fossil displacement target
        // Target: displace enough fossil to achieve ~50% emission reduction by 2035
        const coalGen = byFuel.coal ? byFuel.coal.gen_twh : 0;
        const gasGen = (byFuel.gas_ccgt ? byFuel.gas_ccgt.gen_twh : 0) + (byFuel.gas_peaker ? byFuel.gas_peaker.gen_twh : 0);
        const targetDisplaceTWh = fossilTWh * 0.5; // displace half of fossil by 2035
        const solarNeeded = Math.max(coalGen * 1.2, targetDisplaceTWh * 0.5);
        const windNeeded = targetDisplaceTWh * 0.3;
        const batteryNeeded = Math.max(coalGen * 0.3, targetDisplaceTWh * 0.15);
        const firmCleanNeeded = Math.max(0, fossilTWh * 0.1);

        makeChart('investmentGapChart', {
            type: 'bar',
            data: {
                labels: ['Current Clean', 'Current Fossil', 'Solar Target', 'Wind Target', 'Battery Target', 'Firm Clean Target'],
                datasets: [{
                    label: 'Generation (TWh)',
                    data: [cleanTWh, -fossilTWh, solarNeeded, windNeeded, batteryNeeded, firmCleanNeeded],
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
                scales: { y: { title: { display: true, text: 'Generation (TWh)' } } },
                plugins: {
                    legend: { display: false },
                    tooltip: { callbacks: { label: ctx => `${ctx.label}: ${fmt(Math.abs(ctx.raw), 1)} TWh` } }
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
        renderInvestmentNarrative(byFuel, byISO, cleanTWh, fossilTWh, solarNeeded, windNeeded, batteryNeeded, firmCleanNeeded);
    }

    function renderInvestmentNarrative(byFuel, byISO, cleanTWh, fossilTWh, solarNeeded, windNeeded, batteryNeeded, firmCleanNeeded) {
        const el = document.getElementById('investmentNarrative');
        if (!el) return;

        const isos = Object.keys(byISO);
        const totalGen = co.gen_twh;
        const cleanPct = totalGen > 0 ? ((cleanTWh / totalGen) * 100).toFixed(0) : 0;
        const coalGen = byFuel.coal ? byFuel.coal.gen_twh : 0;
        const hasNuclear = byFuel.nuclear && byFuel.nuclear.gen_twh > 0;

        let sections = [];

        sections.push(`<p><strong>${co.shortName}</strong> currently generates ${fmt(cleanTWh, 1)} TWh of clean energy (${cleanPct}% of output)
            and ${fmt(fossilTWh, 1)} TWh from fossil sources. The economic case for new clean generation is driven by three factors:</p>`);

        sections.push(`<div class="insight-box insight-success" style="margin-bottom:var(--space-md)">
            <strong>1. Declining fossil dispatch</strong>
            <p style="margin-top:6px">As grid-wide clean penetration increases, ${co.shortName}'s fossil plants face declining dispatch hours. Gas CCGT generation falls from current levels to 50-70% by 2035 under median scenarios, reducing revenue.</p>
        </div>`);

        sections.push(`<div class="insight-box insight-success" style="margin-bottom:var(--space-md)">
            <strong>2. New-build clean cost advantage</strong>
            <p style="margin-top:6px">Utility-scale solar ($28/MWh) and onshore wind ($32/MWh) are now cheaper than the marginal cost of existing gas generation in most hours. New solar+battery is cost-competitive with gas CCGT on a 24/7 basis in ERCOT and CAISO.</p>
        </div>`);

        if (hasNuclear) {
            sections.push(`<div class="insight-box insight-success" style="margin-bottom:var(--space-md)">
                <strong>3. Nuclear uprate + life extension value</strong>
                <p style="margin-top:6px">${co.shortName}'s nuclear fleet (${fmt(byFuel.nuclear.gen_twh, 1)} TWh) receives $15/MWh under the 45U PTC. Uprating existing reactors by 5-10% is the cheapest firm clean generation available: no new site permitting, existing grid connection, and immediate output increase.</p>
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
                <li><strong>Solar:</strong> ${fmt(solarNeeded, 1)} TWh new utility-scale solar generation${coalGen > 0 ? ' (includes coal site repowering)' : ''}</li>
                <li><strong>Wind:</strong> ${fmt(windNeeded, 1)} TWh onshore wind (complementary generation profile)</li>
                <li><strong>Battery:</strong> ${fmt(batteryNeeded, 1)} TWh storage dispatch (peak shaving + ancillary services)</li>
                <li><strong>Firm clean:</strong> ${fmt(firmCleanNeeded, 1)} TWh ${hasNuclear ? 'nuclear uprates + potential SMR' : 'CCS-CCGT or contracted nuclear PPA'}</li>
            </ul>
        </div>`);

        el.innerHTML = sections.join('');
    }

    // ─── SECTION 6: Enabling Conditions ──────────────────────

    // Custom Chart.js plugin to draw gap labels and % reduction annotations
    const gapLabelPlugin = {
        id: 'gapLabels',
        afterDraw(chart) {
            const meta = chart.options.plugins.gapLabels;
            if (!meta || !meta.gaps || meta.visible === false) return;
            const ctx = chart.ctx;
            const xScale = chart.scales.x;
            const yScale = chart.scales.y;

            meta.gaps.forEach(function (g) {
                const xPx = xScale.getPixelForValue(g.xIndex);
                const yTopPx = yScale.getPixelForValue(g.yTop);
                const yBotPx = yScale.getPixelForValue(g.yBot);
                const midY = (yTopPx + yBotPx) / 2;

                // Draw vertical bracket line
                ctx.save();
                ctx.strokeStyle = 'rgba(239,68,68,0.6)';
                ctx.lineWidth = 1.5;
                ctx.setLineDash([4, 3]);
                ctx.beginPath();
                ctx.moveTo(xPx + 8, yTopPx);
                ctx.lineTo(xPx + 8, yBotPx);
                ctx.stroke();
                // bracket caps
                ctx.setLineDash([]);
                ctx.beginPath();
                ctx.moveTo(xPx + 4, yTopPx);
                ctx.lineTo(xPx + 12, yTopPx);
                ctx.moveTo(xPx + 4, yBotPx);
                ctx.lineTo(xPx + 12, yBotPx);
                ctx.stroke();
                ctx.restore();

                // Gap label (MMT CO₂)
                ctx.save();
                ctx.font = 'bold 11px "DM Sans", sans-serif';
                ctx.fillStyle = '#DC2626';
                ctx.textAlign = 'left';
                ctx.textBaseline = 'middle';
                ctx.fillText(g.label, xPx + 16, midY);
                ctx.restore();
            });

            // Draw % reduction labels on data points
            if (meta.pctLabels) {
                meta.pctLabels.forEach(function (p) {
                    const xPx = xScale.getPixelForValue(p.xIndex);
                    const yPx = yScale.getPixelForValue(p.y);
                    ctx.save();
                    ctx.font = 'bold 13px "DM Sans", sans-serif';
                    ctx.fillStyle = p.color || '#374151';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = p.above ? 'bottom' : 'top';
                    const offset = p.above ? -8 : 8;
                    ctx.fillText(p.label, xPx, yPx + offset);
                    ctx.restore();
                });
            }
        }
    };

    function pctLabel(value, baseline) {
        if (!baseline || baseline === 0) return '0%';
        const pct = ((value - baseline) / baseline * 100);
        return `${pct >= 0 ? '+' : ''}${Math.round(pct)}%`;
    }

    function buildGapChart(canvasId, p50, targetData, targetLabel, targetColor, baseline) {
        // Compute gaps and % labels at key milestones (indices: 0=2023, 1=2030, 2=2035, 3=2040, 4=2045, 5=2050)
        const gapAnnotations = [];
        const pctLabels = [];
        const milestones = [1, 2, 3, 4, 5]; // 2030, 2035, 2040, 2045, 2050

        milestones.forEach(function (i) {
            const gap = p50[i] - targetData[i];
            if (gap > 0.01) {
                gapAnnotations.push({
                    xIndex: i,
                    yTop: p50[i],
                    yBot: targetData[i],
                    label: fmt(gap, 1) + ' Mt'
                });
            }
            // QT % reduction label
            pctLabels.push({
                xIndex: i, y: p50[i],
                label: pctLabel(p50[i], baseline),
                color: '#6366F1', above: true
            });
            // Target % reduction label
            pctLabels.push({
                xIndex: i, y: targetData[i],
                label: pctLabel(targetData[i], baseline),
                color: targetColor, above: false
            });
        });

        makeChart(canvasId, {
            type: 'line',
            data: {
                labels: YEAR_LABELS,
                datasets: [
                    {
                        label: 'Qualified Target (P50)',
                        data: p50,
                        borderColor: '#6366F1',
                        backgroundColor: 'rgba(239,68,68,0.08)',
                        fill: '+1',
                        borderWidth: 2.5,
                        pointRadius: 4,
                        pointBackgroundColor: '#6366F1'
                    },
                    {
                        label: targetLabel,
                        data: targetData,
                        borderColor: targetColor,
                        backgroundColor: targetColor.replace(')', ',0.08)').replace('rgb', 'rgba'),
                        fill: false,
                        borderWidth: 2.5,
                        borderDash: [8, 4],
                        pointRadius: 4,
                        pointBackgroundColor: targetColor
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                layout: { padding: { right: 60, top: 20 } },
                scales: {
                    y: { title: { display: true, text: 'CO₂ Emissions (Mt)' }, beginAtZero: true },
                    x: { grid: { display: false } }
                },
                plugins: {
                    legend: { position: 'bottom', labels: { usePointStyle: true, padding: 16 } },
                    tooltip: { callbacks: { label: function (ctx) { return ctx.dataset.label + ': ' + fmt(ctx.raw, 2) + ' Mt' + pctFromBaseline(ctx.raw, baseline); } } },
                    gapLabels: { gaps: gapAnnotations, pctLabels: pctLabels }
                }
            },
            plugins: [gapLabelPlugin]
        });
    }

    function renderEnablingConditions() {
        const refBands = co.fan_bands.reference;
        if (!refBands || !refBands.emissions) return;
        const emissions = refBands.emissions;
        const baseline = emissions.p50[0];
        const psV2_local = powerSectorV2Trajectory(baseline);
        const at_local = atTrajectory(baseline);

        // 6.1 — SMARTargets AT gap chart
        buildGapChart('gapChartAT', emissions.p50, at_local,
            'SMARTargets AT (−57/−82/−88/−100%)', '#9C27B0', baseline);

        // 6.2 — SBTi Power Sector v2 gap chart
        buildGapChart('gapChartSBTi', emissions.p50, psV2_local,
            'SBTi Power Sector v2 (NZ 2040)', '#E91E63', baseline);

        // Enabling conditions cards — split into AT and SBTi
        renderConditionsCards(baseline, psV2_local, at_local, emissions);

        // Wire label toggle buttons
        document.querySelectorAll('.gap-label-toggle').forEach(function (btn) {
            btn.addEventListener('click', function () {
                var chartId = btn.getAttribute('data-chart');
                var chart = charts[chartId];
                if (!chart) return;
                var meta = chart.options.plugins.gapLabels;
                var show = meta.visible === false;
                meta.visible = show;
                btn.classList.toggle('active', show);
                chart.update('none');
            });
        });
    }

    function renderConditionsCards(baseline, psV2_local, at_local, emissions) {
        const elAT = document.getElementById('conditionsCardsAT');
        const elSBTi = document.getElementById('conditionsCardsSBTi');
        if (!elAT && !elSBTi) return;

        const dims = co.dimension_fans;

        // Helper to build condition card HTML
        function condCard(title, impact, desc, variant) {
            const tag = variant === 'at' ? 'convergence-tag convergent' : 'convergence-tag moderate';
            return `<div class="card" style="padding:var(--space-lg)">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px">
                    <h4 style="margin:0;color:var(--navy);font-size:1rem">${title}</h4>
                    <span class="${tag}">${impact}</span>
                </div>
                <p style="font-size:0.88rem;color:var(--text-secondary);line-height:1.6;margin:0">${desc}</p>
            </div>`;
        }

        // Compute gap to AT at each milestone
        const p50 = emissions.p50;
        const gap2030_at = p50[1] - at_local[1];
        const gap2035_at = p50[2] - at_local[2];
        const gap2040_at = p50[3] - at_local[3];
        const gap2040_ps = p50[3] - psV2_local[3];

        // ── 6.1: Enabling Conditions for SMARTargets AT ──
        if (elAT) {
            let html = '';
            html += `<p style="font-size:0.88rem;color:var(--text-secondary);margin-bottom:var(--space-md);grid-column:1/-1">
                AT trajectory: −57% by 2030, −82% by 2035, −88% by 2040, −100% by 2050.
                ${gap2030_at > 0.5 ? `${co.shortName} has a ${fmt(gap2030_at, 1)} Mt gap at 2030.` : `${co.shortName} is on track for the 2030 milestone.`}
            </p>`;

            let atCards = [];
            if (gap2030_at > 0.5) {
                atCards.push(condCard('Near-Term Coal/Gas Retirement Acceleration', `${fmt(gap2030_at, 1)} Mt gap by 2030`,
                    `Closing the ${fmt(gap2030_at, 1)} Mt gap to the AT trajectory by 2030 (−57%) requires accelerated fossil retirement beyond current market-driven pace. This means either mandated coal exit, carbon pricing that makes high-heat-rate gas uneconomic, or dramatically faster clean interconnection.`, 'at'));
            }
            if (gap2035_at > 0.5) {
                atCards.push(condCard('Mid-Term Clean Deployment at Scale', `${fmt(gap2035_at, 1)} Mt gap by 2035`,
                    `The −82% target by 2035 requires aggressive clean deployment: utility-scale solar+storage, wind, and potentially nuclear uprates or SMR deployment. ${co.shortName} would need to roughly triple clean build rates vs. market trajectory.`, 'at'));
            }
            if (gap2040_at > 0.1) {
                atCards.push(condCard('Firm Clean & CCS for Residual Emissions', `${fmt(gap2040_at, 1)} Mt gap by 2040`,
                    `Reaching −88% by 2040 means near-total elimination of gas CCGT dispatch. Requires firm clean alternatives (CCS-CCGT, advanced nuclear, LDES) to replace gas for grid reliability.`, 'at'));
            }
            if (dims) {
                if (dims.conditions) {
                    const emF = dims.conditions.F ? dims.conditions.F.emissions.p50[3] : null;
                    const emC = dims.conditions.C ? dims.conditions.C.emissions.p50[3] : null;
                    if (emF !== null && emC !== null) {
                        atCards.push(condCard('Interconnection & Permitting Reform', `${fmt(Math.abs(emC - emF), 1)} Mt by 2040`,
                            `Facilitating conditions (faster queues, streamlined permitting): ${fmt(emF, 1)} Mt vs. Challenging: ${fmt(emC, 1)} Mt by 2040. The AT trajectory requires facilitating conditions at minimum.`, 'at'));
                    }
                }
                if (dims.price_sens) {
                    const low = dims.price_sens.Low ? dims.price_sens.Low.emissions.p50[3] : null;
                    const high = dims.price_sens.High ? dims.price_sens.High.emissions.p50[3] : null;
                    if (low !== null && high !== null) {
                        atCards.push(condCard('Aggressive LCOE Reduction', `${fmt(Math.abs(high - low), 1)} Mt range`,
                            `Low LCOE (Wright's Law outperformance): ${fmt(low, 1)} Mt vs. High LCOE: ${fmt(high, 1)} Mt by 2040. AT compliance requires LCOE trajectories at or below the low case.`, 'at'));
                    }
                }
                if (dims.ppa_level) {
                    const lowP = dims.ppa_level.Low ? dims.ppa_level.Low.emissions.p50[3] : null;
                    const highP = dims.ppa_level.High ? dims.ppa_level.High.emissions.p50[3] : null;
                    if (lowP !== null && highP !== null) {
                        atCards.push(condCard('High PPA Market Depth', `${fmt(Math.abs(highP - lowP), 1)} Mt range at 2040`,
                            `SMARTargets requires faster decarbonization than SBTi, which means clean investment must be pulled forward by market demand, not just policy. High corporate PPA commitment: ${fmt(highP, 1)} Mt vs. Low PPA: ${fmt(lowP, 1)} Mt by 2040. Deep PPA markets create bankable revenue certainty for new clean projects, accelerating deployment to meet the AT pace.`, 'at'));
                    }
                }
            }
            html += atCards.join('');
            elAT.innerHTML = html;
        }

        // ── 6.2: Enabling Conditions for SBTi Power Sector v2 ──
        if (elSBTi) {
            let html = '';
            html += `<p style="font-size:0.88rem;color:var(--text-secondary);margin-bottom:var(--space-md);grid-column:1/-1">
                SBTi Power Sector v2 requires net-zero emissions by 2040.
                ${gap2040_ps > 0.5 ? `${co.shortName} has a ${fmt(gap2040_ps, 1)} Mt gap at 2040 under market conditions.` : `${co.shortName}'s market trajectory aligns with net-zero by 2040.`}
                Achieving this <em>competitively</em>, without destroying shareholder value, requires specific enabling conditions.
            </p>`;

            let nzCards = [];
            if (gap2040_ps > 0.5) {
                nzCards.push(condCard('Carbon Pricing Signal', `Critical for NZ 2040`,
                    `Net-zero by 2040 requires eliminating all remaining gas generation revenue within 14 years. A carbon price of $50-100/tCO₂ would internalize the externality and make gas CCGT uneconomic relative to clean firm alternatives. Without it, retiring profitable gas assets destroys ~$${fmt(gap2040_ps * 150, 0)}M in annual revenue.`, 'nz'));
            }
            nzCards.push(condCard('Clean Firm Technology at Scale', 'Required by 2035',
                `Net-zero by 2040 requires firm clean generation to replace gas CCGT for grid reliability. This means commercially available advanced nuclear (SMR), CCS at scale, or LDES (iron-air) by 2035, five years before the deadline, to allow buildout.`, 'nz'));

            if (dims && dims.ppa_level) {
                const lowP = dims.ppa_level.Low ? dims.ppa_level.Low.emissions.p50[3] : null;
                const highP = dims.ppa_level.High ? dims.ppa_level.High.emissions.p50[3] : null;
                if (lowP !== null && highP !== null) {
                    nzCards.push(condCard('Deep Corporate PPA Market', `${fmt(Math.abs(highP - lowP), 1)} Mt range at 2040`,
                        `High PPA commitment (corporate buyers willing to pay premium for 24/7 clean): ${fmt(highP, 1)} Mt vs. Low PPA: ${fmt(lowP, 1)} Mt. NZ 2040 requires high PPA market depth to finance new clean investment at acceptable returns.`, 'nz'));
                }
            }

            if (dims && dims.demand_growth) {
                const low = dims.demand_growth.Low ? dims.demand_growth.Low.emissions.p50[3] : null;
                const high = dims.demand_growth.High ? dims.demand_growth.High.emissions.p50[3] : null;
                if (low !== null && high !== null) {
                    nzCards.push(condCard('Demand Growth Management', `${fmt(Math.abs(high - low), 1)} Mt range`,
                        `Under high demand growth (AI/data centers), total generation increases, requiring even more clean build to hit zero. Low demand: ${fmt(low, 1)} Mt vs. High demand: ${fmt(high, 1)} Mt by 2040.`, 'nz'));
                }
            }

            nzCards.push(condCard('Stranded Asset Compensation', 'Shareholder protection',
                `Accelerating gas retirement to meet NZ 2040 strands assets with remaining economic life. Competitive transition requires either regulated asset recovery mechanisms, capacity market reform to value clean firm, or carbon credit revenue to offset lost gas dispatch revenue.`, 'nz'));

            html += nzCards.join('');
            elSBTi.innerHTML = html;
        }
    }

    // ─── SECTION 7: Strategic Positioning ────────────────────
    function renderStrategicPositioning() {
        const byFuel = getFleetByFuel();
        const byISO = getFleetByISO();
        const fb = co.fan_bands.reference;

        // Robustness scores — based on generation share (TWh), not nameplate capacity
        const totalGen = co.gen_twh;
        const nuclearGen = byFuel.nuclear ? byFuel.nuclear.gen_twh : 0;
        const nuclearScore = totalGen > 0 ? Math.min(100, (nuclearGen / totalGen) * 200) : 0; // 50% nuclear gen = 100 score
        const isoCount = Object.keys(byISO).length;
        const diversityScore = Math.min(100, isoCount * 25); // 4 ISOs = 100
        const coalGen = byFuel.coal ? byFuel.coal.gen_twh : 0;
        const coalExitScore = totalGen > 0 ? (100 - Math.min(100, (coalGen / totalGen) * 200)) : 100;
        const peakerGen = byFuel.gas_peaker ? byFuel.gas_peaker.gen_twh : 0;
        const ccgtGen = byFuel.gas_ccgt ? byFuel.gas_ccgt.gen_twh : 0;
        const gasFlexScore = ccgtGen > 0 ? Math.min(100, (ccgtGen / (ccgtGen + peakerGen)) * 100) : 50;
        const cleanNewGen = ['solar', 'wind', 'battery'].reduce((s, f) => s + (byFuel[f] ? byFuel[f].gen_twh : 0), 0);
        const cleanPipelineScore = totalGen > 0 ? Math.min(100, (cleanNewGen / totalGen) * 300) : 0;

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
        if (isos.includes('ERCOT')) recs.push('Leverage ERCOT\'s energy-only market for solar+battery deployment. High-value peak pricing creates strong returns.');
        if (isos.includes('PJM')) recs.push('Utilize PJM capacity market revenue to support firm clean generation investments (nuclear uprates, CCS).');
        if (byFuel.coal && byFuel.coal.gen_twh > 1) recs.push('Accelerate coal retirement and site repowering. Retiring coal sites have existing grid interconnection, land, and workforce.');
        if (byFuel.nuclear && byFuel.nuclear.gen_twh > 10) recs.push('Pursue nuclear uprates (5-10% output increase) at ~$10/MWh incremental cost, the cheapest firm clean generation available.');
        if (byFuel.gas_peaker && byFuel.gas_peaker.gen_twh > 3) recs.push('Replace highest-heat-rate gas peakers with 4-hour battery storage at sites with declining dispatch.');
        recs.push('Build out corporate PPA pipeline. Voluntary procurement creates revenue certainty for new clean projects.');

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

    // ─── Section 8: Gas Development Analysis ───────────────
    function renderGasDevelopment() {
        const fb = co.fan_bands?.all;
        if (!fb) return;

        // Determine gas fleet characteristics
        const fleet = co.fleet_summary || {};
        const byFuel = {};
        for (const iso of Object.keys(fleet)) {
            for (const fuel of Object.keys(fleet[iso])) {
                if (!byFuel[fuel]) byFuel[fuel] = { cap_mw: 0, gen_twh: 0, co2_mt: 0, isos: [] };
                byFuel[fuel].cap_mw += fleet[iso][fuel].cap_mw || 0;
                byFuel[fuel].gen_twh += fleet[iso][fuel].gen_twh || 0;
                byFuel[fuel].co2_mt += fleet[iso][fuel].co2_mt || 0;
                if (!byFuel[fuel].isos.includes(iso)) byFuel[fuel].isos.push(iso);
            }
        }

        const gasGen = (byFuel.gas_ccgt?.gen_twh || 0) + (byFuel.gas_peaker?.gen_twh || 0);
        const gasCap = (byFuel.gas_ccgt?.cap_mw || 0) + (byFuel.gas_peaker?.cap_mw || 0);
        const gasCO2 = (byFuel.gas_ccgt?.co2_mt || 0) + (byFuel.gas_peaker?.co2_mt || 0);
        const totalGen = co.gen_twh || 1;
        const gasPct = (gasGen / totalGen * 100).toFixed(0);

        // Identify ISOs where company has gas and where new gas is most likely
        const gasISOs = {};
        for (const iso of Object.keys(fleet)) {
            const isoGas = (fleet[iso].gas_ccgt?.cap_mw || 0) + (fleet[iso].gas_peaker?.cap_mw || 0);
            if (isoGas > 0) gasISOs[iso] = isoGas;
        }

        // Gas development drivers by ISO
        const isoDrivers = {
            ERCOT: { likelihood: 'High', reason: 'Energy-only market, data center demand, reliability premium after Winter Storm Uri', newMW: Math.round(gasCap * 0.15) },
            PJM: { likelihood: 'Medium', reason: 'Capacity market revenue, coal retirement replacements, grid reliability mandates', newMW: Math.round(gasCap * 0.10) },
            CAISO: { likelihood: 'Low', reason: 'Aggressive clean mandates (SB 100), but gas needed for evening ramp reliability', newMW: Math.round(gasCap * 0.05) },
            NYISO: { likelihood: 'Low', reason: 'CLCPA constraints, peaker rule phase-out, limited new gas permitting', newMW: 0 },
            NEISO: { likelihood: 'Medium', reason: 'Winter reliability concerns, pipeline constraints, gas peaker replacement cycle', newMW: Math.round(gasCap * 0.08) },
            MISO: { likelihood: 'Medium', reason: 'Coal retirement gap, industrial load growth, capacity shortfall warnings', newMW: Math.round(gasCap * 0.10) },
            SPP: { likelihood: 'Low', reason: 'Wind-rich region, limited gas fleet presence', newMW: 0 }
        };

        // Gas impact chart: emissions trajectory with and without new gas
        const gasImpactCtx = document.getElementById('gasImpactChart');
        if (gasImpactCtx) {
            const p50Emissions = fb.emissions?.p50 || [co.co2_2024_mt, 30, 25, 20, 15, 10];
            const totalNewGasMW = Object.values(gasISOs).reduce((s, mw) => {
                const iso = Object.keys(gasISOs).find(k => gasISOs[k] === mw);
                return s + (isoDrivers[iso]?.newMW || 0);
            }, 0);
            // New gas adds ~0.4 tCO2/MWh * CF ~50% * 8760h * newMW
            const newGasAnnualCO2 = totalNewGasMW * 0.4 * 0.5 * 8.76 / 1000; // Mt

            const withGas = p50Emissions.map((v, i) => {
                if (i === 0) return v;
                // New gas comes online by 2030, peaks mid-period, then dispatched down
                const rampFactor = i <= 1 ? 1.0 : i <= 2 ? 0.9 : i <= 3 ? 0.7 : i <= 4 ? 0.5 : 0.3;
                return v + newGasAnnualCO2 * rampFactor;
            });

            charts.gasImpact = new Chart(gasImpactCtx, applyChartStyle({
                type: 'line',
                data: {
                    labels: YEAR_LABELS,
                    datasets: [
                        {
                            label: 'Baseline Trajectory (P50)',
                            data: p50Emissions,
                            borderColor: '#6366F1',
                            backgroundColor: 'rgba(99,102,241,0.08)',
                            fill: true,
                            tension: 0.3,
                            pointRadius: 5,
                            borderWidth: 3
                        },
                        {
                            label: 'With New Gas Development',
                            data: withGas,
                            borderColor: '#EF4444',
                            backgroundColor: 'rgba(239,68,68,0.08)',
                            fill: true,
                            tension: 0.3,
                            pointRadius: 5,
                            borderWidth: 3,
                            borderDash: [8, 4]
                        }
                    ]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'bottom', labels: { usePointStyle: true, padding: 12 } },
                        tooltip: { callbacks: { label: function (ctx) {
                            var baseline = p50Emissions[0];
                            var pct = baseline ? ((ctx.raw - baseline) / baseline * 100) : 0;
                            var sign = pct >= 0 ? '+' : '';
                            return ctx.dataset.label + ': ' + fmt(ctx.raw) + ' Mt CO\u2082 (' + sign + pct.toFixed(1) + '% vs 2023)';
                        } } }
                    },
                    scales: {
                        x: { title: { display: true, text: 'Year' }, grid: { display: false } },
                        y: { title: { display: true, text: 'Annual CO₂ (Mt)' }, min: 0, grid: { color: 'rgba(0,0,0,0.05)' } }
                    }
                }
            }));
        }

        // Gas location chart: potential new gas by ISO
        const gasLocCtx = document.getElementById('gasLocationChart');
        if (gasLocCtx) {
            const relevantISOs = Object.keys(gasISOs);
            if (relevantISOs.length === 0) relevantISOs.push('None');

            charts.gasLocation = new Chart(gasLocCtx, applyChartStyle({
                type: 'bar',
                data: {
                    labels: relevantISOs,
                    datasets: [
                        {
                            label: 'Existing Gas (MW)',
                            data: relevantISOs.map(iso => gasISOs[iso] || 0),
                            backgroundColor: 'rgba(107,114,128,0.6)',
                            borderRadius: 4
                        },
                        {
                            label: 'Potential New Gas (MW)',
                            data: relevantISOs.map(iso => isoDrivers[iso]?.newMW || 0),
                            backgroundColor: 'rgba(239,68,68,0.6)',
                            borderRadius: 4
                        }
                    ]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { position: 'bottom', labels: { usePointStyle: true, padding: 12 } } },
                    scales: {
                        x: { stacked: true, grid: { display: false } },
                        y: { stacked: true, title: { display: true, text: 'Capacity (MW)' }, grid: { color: 'rgba(0,0,0,0.05)' } }
                    }
                }
            }));
        }

        // Gas trajectory delta chart
        const gasDeltaCtx = document.getElementById('gasTrajectoryDelta');
        if (gasDeltaCtx) {
            const p50 = fb.emissions?.p50 || [];
            const p90 = fb.emissions?.p90 || [];
            charts.gasDelta = new Chart(gasDeltaCtx, applyChartStyle({
                type: 'bar',
                data: {
                    labels: YEAR_LABELS,
                    datasets: [{
                        label: 'Additional CO₂ from New Gas (Mt)',
                        data: YEAR_LABELS.map((_, i) => {
                            const totalNewGas = Object.keys(gasISOs).reduce((s, iso) => s + (isoDrivers[iso]?.newMW || 0), 0);
                            const annualCO2 = totalNewGas * 0.4 * 0.5 * 8.76 / 1000;
                            if (i === 0) return 0;
                            const ramp = i <= 1 ? 1.0 : i <= 2 ? 0.9 : i <= 3 ? 0.7 : i <= 4 ? 0.5 : 0.3;
                            return +(annualCO2 * ramp).toFixed(1);
                        }),
                        backgroundColor: 'rgba(239,68,68,0.5)',
                        borderColor: '#EF4444',
                        borderWidth: 1,
                        borderRadius: 4
                    }]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { display: false } },
                        y: { title: { display: true, text: 'Additional Mt CO₂/yr' }, min: 0, grid: { color: 'rgba(0,0,0,0.05)' } }
                    }
                }
            }));
        }

        // Gas narrative
        const gasNarEl = document.getElementById('gasNarrative');
        if (gasNarEl) {
            const highLikelihood = Object.entries(gasISOs)
                .filter(([iso]) => isoDrivers[iso]?.likelihood === 'High')
                .map(([iso]) => iso);
            const medLikelihood = Object.entries(gasISOs)
                .filter(([iso]) => isoDrivers[iso]?.likelihood === 'Medium')
                .map(([iso]) => iso);

            let drivers = '';
            if (highLikelihood.length > 0) {
                drivers += `<p><strong>High likelihood regions (${highLikelihood.join(', ')}):</strong> ${highLikelihood.map(iso => isoDrivers[iso].reason).join('. ')}.</p>`;
            }
            if (medLikelihood.length > 0) {
                drivers += `<p><strong>Moderate likelihood regions (${medLikelihood.join(', ')}):</strong> ${medLikelihood.map(iso => isoDrivers[iso].reason).join('. ')}.</p>`;
            }

            gasNarEl.innerHTML = `
                <p><strong>${co.shortName}</strong> currently operates <strong>${fmt(gasCap, 0)} MW</strong> of gas generation
                (${gasPct}% of fleet output), producing <strong>${fmt(gasCO2)} Mt CO₂</strong> annually across
                ${Object.keys(gasISOs).length} ISOs.</p>

                <div class="insight-box" style="margin: var(--space-lg) 0">
                    <strong>Key Question:</strong> Would ${co.shortName} develop new gas-fired generation, and if so, where?
                    The answer depends on three converging factors: (1) data center / industrial load growth creating demand for firm capacity,
                    (2) wholesale market price signals and capacity market revenue adequacy, and
                    (3) the pace at which clean firm alternatives (CCS, SMRs, LDES) become commercially available.
                </div>

                ${drivers}

                <div class="insight-box insight-warn" style="margin: var(--space-lg) 0">
                    <strong>Trajectory Impact:</strong> New gas development would slow decarbonization by extending the fossil generation tail.
                    Each 1 GW of new CCGT operating at 50% capacity factor adds approximately ${fmt(0.4 * 0.5 * 8.76, 1)} Mt CO₂/year
                    to the fleet's emissions, potentially widening the gap to 1.5°C alignment by ${fmt(0.4 * 0.5 * 8.76 / (co.co2_2024_mt || 55) * 100, 0)}%
                    of current total emissions. However, if new gas displaces higher-emitting coal or older peakers in neighboring regions,
                    the net system-wide impact may be partially offset through consequential accounting.
                </div>

                <p>The strategic tension: ${co.shortName} faces pressure to grow revenue through load-following capacity for data centers
                and electrification. New gas generation offers quick-to-market, bankable returns. But each new gas plant extends the fossil
                tail and creates stranded asset risk under accelerating decarbonization scenarios. The optimal strategy likely involves
                developing gas with CCS-readiness, signing offtake agreements that include carbon pricing escalators, and positioning
                new plants for eventual conversion to hydrogen or CCS retrofit.</p>
            `;
        }
    }

    // ─── Initialize ──────────────────────────────────────────
    function init() {
        // Set page title
        document.title = `${co.name} | IPP Climate Transition Analysis`;
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
        renderGasDevelopment();
        initTocTracking();
    }

    // Run when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
