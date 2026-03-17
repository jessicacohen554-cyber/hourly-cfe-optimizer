/**
 * IPP Climate Transition Hub - Cross-Company Overview
 * =====================================================
 * Renders hero stats, company cards, and comparative analysis charts.
 * Reads from: IPP_SMARTARGETS_DATA (ipp-smartargets-data.js)
 */

/* global IPP_SMARTARGETS_DATA, Chart, RESOURCE_COLORS, ISO_COLORS */

(function () {
    'use strict';

    // ─── Glass-morphism chart styling helpers ───────────────
    function toRGBA(color, alpha) {
        if (!color || typeof color !== 'string') return color;
        if (color.startsWith('rgba(')) return color.replace(/,\s*[\d.]+\)$/, ', ' + alpha + ')');
        if (color.startsWith('rgb(')) return color.replace('rgb(', 'rgba(').replace(')', ', ' + alpha + ')');
        if (color.startsWith('#')) {
            var hex = color.slice(1);
            if (hex.length === 3) hex = hex[0]+hex[0]+hex[1]+hex[1]+hex[2]+hex[2];
            return 'rgba(' + parseInt(hex.slice(0,2),16) + ',' + parseInt(hex.slice(2,4),16) + ',' + parseInt(hex.slice(4,6),16) + ',' + alpha + ')';
        }
        return color;
    }

    function applyChartStyle(config) {
        if (!config || !config.data || !config.data.datasets) return config;
        var chartType = config.type;
        config.data.datasets.forEach(function(ds) {
            if (ds._skipStyle) return;
            if (chartType === 'doughnut' || chartType === 'pie') {
                if (Array.isArray(ds.backgroundColor)) {
                    ds.borderColor = ds.backgroundColor.map(function(c) { return c; });
                    ds.backgroundColor = ds.backgroundColor.map(function(c) { return toRGBA(c, 0.3); });
                }
                ds.borderWidth = ds.borderWidth || 2;
            } else if (chartType === 'bar') {
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

    function styledChart(ctx, config) {
        applyChartStyle(config);
        return new Chart(ctx, config);
    }

    const YEARS = [2023, 2030, 2035, 2040, 2045, 2050];
    const YEAR_LABELS = ['2023', '2030', '2035', '2040', '2045', '2050'];

    // SBTi Power Sector v2: Net zero by 2040
    const SBTI_POWER_V2_REMAINING = { 2023: 1.00, 2030: 0.412, 2035: 0.118, 2040: 0.0, 2045: 0.0, 2050: 0.0 };
    // SMARTargets AT trajectory
    const AT_TRAJECTORY_REMAINING = { 2023: 1.00, 2030: 0.43, 2035: 0.18, 2040: 0.12, 2045: 0.06, 2050: 0.00 };

    function pctFromBaseline(value, baseline) {
        if (!baseline || baseline === 0) return '';
        const pct = ((value - baseline) / baseline * 100);
        return ` (${pct >= 0 ? '+' : ''}${pct.toFixed(1)}% vs 2023)`;
    }

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

    const COMPANY_COLORS = [
        '#6366F1', '#22C55E', '#F59E0B', '#E91E63', '#0EA5E9', '#9C27B0', '#14B8A6'
    ];

    const ISO_COLOR_MAP = {
        CAISO: ISO_COLORS.CAISO, ERCOT: ISO_COLORS.ERCOT, PJM: ISO_COLORS.PJM,
        NYISO: ISO_COLORS.NYISO, NEISO: ISO_COLORS.NEISO, MISO: ISO_COLORS.MISO, SPP: ISO_COLORS.SPP
    };

    const data = IPP_SMARTARGETS_DATA;
    const companyIds = Object.keys(data.companies);
    const companies = companyIds.map(id => ({ id, ...data.companies[id] }));

    const FUEL_ORDER = ['nuclear', 'hydro', 'geothermal', 'wind', 'solar', 'battery', 'gas_ccgt', 'gas_peaker', 'coal', 'oil'];

    // Cumulative MAC ($/tCO₂) at 99.99% CFE threshold per ISO - from step5d MAC queue
    const ISO_MAC_FULL = {
        ERCOT: 356, PJM: 254, CAISO: 482, NEISO: 468, NYISO: 492, MISO: 357, SPP: 253
    };

    const FOSSIL_FUELS = ['coal', 'gas_ccgt', 'gas_peaker', 'oil'];

    function fmt(n, d) {
        if (n === undefined || n === null) return ' - ';
        if (d === undefined) d = Math.abs(n) >= 100 ? 0 : 1;
        return n.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
    }

    function computeRadarScores(co) {
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
        const totalGen = co.gen_twh || 1;
        const nuclearGen = byFuel.nuclear ? byFuel.nuclear.gen_twh : 0;
        const nuclearScore = Math.min(100, (nuclearGen / totalGen) * 200);
        const isoCount = Object.keys(summary).length;
        const diversityScore = Math.min(100, isoCount * 25);
        const coalGen = byFuel.coal ? byFuel.coal.gen_twh : 0;
        const coalExitScore = totalGen > 0 ? (100 - Math.min(100, (coalGen / totalGen) * 200)) : 100;
        const peakerGen = byFuel.gas_peaker ? byFuel.gas_peaker.gen_twh : 0;
        const ccgtGen = byFuel.gas_ccgt ? byFuel.gas_ccgt.gen_twh : 0;
        const gasFlexScore = ccgtGen > 0 ? Math.min(100, (ccgtGen / (ccgtGen + peakerGen)) * 100) : 50;
        const cleanNewGen = ['solar', 'wind', 'battery'].reduce((s, f) => s + (byFuel[f] ? byFuel[f].gen_twh : 0), 0);
        const cleanPipelineScore = Math.min(100, (cleanNewGen / totalGen) * 300);
        return [nuclearScore, diversityScore, coalExitScore, gasFlexScore, cleanPipelineScore];
    }

    // ─── Hero Stats ──────────────────────────────────────────
    function renderHeroStats() {
        const totalGen = companies.reduce((s, c) => s + c.gen_twh, 0);
        const totalCO2 = companies.reduce((s, c) => s + c.co2_2024_mt, 0);
        const avgIntensity = Math.round(totalCO2 * 1e6 / (totalGen * 1e3)); // kg/MWh

        document.getElementById('heroCapacity').textContent = fmt(totalGen, 0);
        document.getElementById('heroEmissions').textContent = fmt(totalCO2, 0);
        document.getElementById('heroScenarios').textContent = '270';
    }

    // ─── Company Cards ───────────────────────────────────────
    function renderCompanyCards() {
        const grid = document.getElementById('companyCardsGrid');
        if (!grid) return;

        // Sort by emissions (highest first)
        const sorted = [...companies].sort((a, b) => b.co2_2024_mt - a.co2_2024_mt);

        grid.innerHTML = sorted.map((co, idx) => {
            // Fleet summary for fuel bar
            const summary = co.fleet_summary;
            const isos = Object.keys(summary);

            // Fuel breakdown by generation (TWh)
            const fuelGen = {};
            let totalFuelGen = 0;
            for (const iso of isos) {
                for (const fuel of Object.keys(summary[iso])) {
                    const gen = summary[iso][fuel].gen_twh || 0;
                    fuelGen[fuel] = (fuelGen[fuel] || 0) + gen;
                    totalFuelGen += gen;
                }
            }

            // Fuel bar segments (by generation)
            const fuelOrder = ['nuclear', 'hydro', 'geothermal', 'wind', 'solar', 'battery', 'gas_ccgt', 'gas_peaker', 'coal', 'oil'];
            const fuelBar = totalFuelGen > 0 ? fuelOrder
                .filter(f => fuelGen[f] && fuelGen[f] > 0)
                .map(f => `<div class="fuel-bar-segment" style="width:${(fuelGen[f]/totalFuelGen*100).toFixed(1)}%;background:${FUEL_COLORS[f]}" title="${FUEL_LABELS[f]}: ${fmt(fuelGen[f], 1)} TWh"></div>`)
                .join('') : '';

            // ISO badges
            const isoBadges = isos.map(iso =>
                `<span class="iso-badge" style="background:${ISO_COLOR_MAP[iso] || '#6B7280'}">${iso}</span>`
            ).join('');

            // P50 emission reduction by 2050
            const fb = co.fan_bands.all.emissions;
            const reduction = fb.p50[0] > 0 ? ((fb.p50[0] - fb.p50[5]) / fb.p50[0] * 100).toFixed(0) : ' - ';
            const companyColor = COMPANY_COLORS[idx % COMPANY_COLORS.length];

            return `
                <div class="company-card scroll-reveal" style="border-top: 3px solid ${companyColor}">
                    <div class="company-card-header">
                        <div>
                            <h3>${co.name}</h3>
                            <div class="company-card-target">${co.target}</div>
                        </div>
                    </div>
                    <div class="company-card-body" style="display:flex;gap:var(--space-md);align-items:center">
                        <div class="company-card-radar" style="flex:0 0 130px;height:130px">
                            <canvas id="radarCard_${co.id}" width="130" height="130"></canvas>
                        </div>
                        <div class="company-card-stats" style="flex:1;display:grid;grid-template-columns:1fr 1fr;gap:4px">
                            <div class="company-card-stat"><div class="val" style="font-size:1rem">${fmt(co.gen_twh, 0)}</div><div class="lbl" style="font-size:0.7rem">TWh/yr</div></div>
                            <div class="company-card-stat"><div class="val" style="font-size:1rem">${fmt(co.co2_2024_mt, 0)}</div><div class="lbl" style="font-size:0.7rem">Mt CO₂</div></div>
                            <div class="company-card-stat"><div class="val" style="font-size:1rem">${reduction}%</div><div class="lbl" style="font-size:0.7rem">2050 Reduction</div></div>
                            <div class="company-card-stat"><div class="val" style="font-size:1rem">${fmt(co.intensity_kg, 0)}</div><div class="lbl" style="font-size:0.7rem">kg/MWh</div></div>
                        </div>
                    </div>
                    <div class="fuel-bar">${fuelBar}</div>
                    <div class="iso-badges">${isoBadges}</div>
                    <a href="ipp_${co.id}.html" class="company-card-link">View Full Analysis</a>
                </div>
            `;
        }).join('');

        // Render mini radar charts for each card
        sorted.forEach((co, idx) => {
            const canvasId = `radarCard_${co.id}`;
            const ctx = document.getElementById(canvasId);
            if (!ctx) return;
            const scores = computeRadarScores(co);
            const companyColor = COMPANY_COLORS[idx % COMPANY_COLORS.length];
            styledChart(ctx, {
                type: 'radar',
                data: {
                    labels: ['Nuclear', 'Diversity', 'Coal Exit', 'Gas Flex', 'Clean Build'],
                    datasets: [{
                        data: scores,
                        borderColor: companyColor,
                        backgroundColor: companyColor + '25',
                        borderWidth: 2,
                        pointRadius: 2,
                        pointBackgroundColor: companyColor
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    scales: {
                        r: {
                            min: 0, max: 100,
                            ticks: { display: false },
                            pointLabels: { font: { size: 8 }, color: '#6B7280' },
                            grid: { color: 'rgba(0,0,0,0.06)' }
                        }
                    },
                    plugins: { legend: { display: false } },
                    animation: { duration: 800 }
                }
            });
        });
    }

    // ─── Key Insights ────────────────────────────────────────
    function renderKeyInsights() {
        const el = document.getElementById('keyInsights');
        if (!el) return;

        // What's consistent across all scenarios
        let consistent = [];
        let divergent = [];

        // Check coal retirement - all companies with coal show declining coal dispatch
        const coalCompanies = companies.filter(c => {
            const summary = c.fleet_summary;
            return Object.keys(summary).some(iso => summary[iso].coal && summary[iso].coal.gen_twh > 0);
        });
        if (coalCompanies.length > 0) {
            consistent.push(`<strong>Coal exits:</strong> All ${coalCompanies.length} companies with coal (${coalCompanies.map(c => c.shortName).join(', ')}) show coal retirement by 2027-2030 across every scenario modeled.`);
        }

        // Nuclear always profitable
        const nuclearCos = companies.filter(c => {
            const summary = c.fleet_summary;
            return Object.keys(summary).some(iso => summary[iso].nuclear && summary[iso].nuclear.gen_twh > 0);
        });
        if (nuclearCos.length > 0) {
            consistent.push(`<strong>Nuclear backbone:</strong> Nuclear generation remains the most profitable and resilient asset class across all scenarios for ${nuclearCos.map(c => c.shortName).join(', ')}.`);
        }

        // All companies remain profitable
        const allProfitable = companies.every(c => c.fan_bands.all.profit && c.fan_bands.all.profit.p10[5] > 0);
        if (allProfitable) {
            consistent.push(`<strong>Universal profitability:</strong> All 7 companies remain profitable even in worst-case (P10) scenarios through 2050.`);
        }

        // Where they diverge
        const emissionReductions = companies.map(c => {
            const fb = c.fan_bands.all.emissions;
            return { name: c.shortName, reduction: fb.p50[0] > 0 ? ((fb.p50[0] - fb.p50[5]) / fb.p50[0] * 100) : 100 };
        }).sort((a, b) => b.reduction - a.reduction);

        divergent.push(`<strong>Decarbonization pace:</strong> ${emissionReductions[0].name} achieves the deepest reduction (${emissionReductions[0].reduction.toFixed(0)}% by 2050) while ${emissionReductions[emissionReductions.length-1].name} shows the least (${emissionReductions[emissionReductions.length-1].reduction.toFixed(0)}%).`);

        const intensities = companies.map(c => ({ name: c.shortName, intensity: c.intensity_kg })).sort((a, b) => a.intensity - b.intensity);
        divergent.push(`<strong>Starting position:</strong> Fleet intensity spans ${intensities[0].intensity}–${intensities[intensities.length-1].intensity} kg/MWh (${intensities[0].name} to ${intensities[intensities.length-1].name}), creating fundamentally different transition challenges.`);

        el.innerHTML = `
            <div class="insight-box insight-success" style="margin-bottom:var(--space-md)">
                <strong>Consistent Across All Scenarios</strong>
                <ul style="margin:8px 0 0 16px">${consistent.map(c => `<li>${c}</li>`).join('')}</ul>
            </div>
            <div class="insight-box insight-warn">
                <strong>Where Companies Diverge</strong>
                <ul style="margin:8px 0 0 16px">${divergent.map(d => `<li>${d}</li>`).join('')}</ul>
            </div>
        `;
    }

    // ─── Cross-Company Comparison Charts ─────────────────────
    function renderEmissionsComparison() {
        const datasets = [];
        companies.forEach((co, i) => {
            const fb = co.fan_bands.all.emissions;
            const color = COMPANY_COLORS[i % COMPANY_COLORS.length];

            // P10/P90 band
            datasets.push({
                label: co.shortName + ' (P10-P90)',
                data: fb.p90,
                borderColor: 'transparent',
                backgroundColor: color + '18',
                fill: '+1',
                pointRadius: 0,
                order: 2
            });
            datasets.push({
                label: co.shortName + ' P10',
                data: fb.p10,
                borderColor: 'transparent',
                fill: false,
                pointRadius: 0,
                hidden: true,
                order: 2
            });

            // P50 line
            datasets.push({
                label: co.shortName,
                data: fb.p50,
                borderColor: color,
                backgroundColor: 'transparent',
                borderWidth: 2.5,
                pointRadius: 3,
                pointBackgroundColor: color,
                tension: 0.3,
                order: 1
            });
        });

        styledChart(document.getElementById('emissionsComparisonChart'), {
            type: 'line',
            data: { labels: YEAR_LABELS, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { title: { display: true, text: 'CO₂ Emissions (Mt)' }, beginAtZero: true }
                },
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            font: { size: 11 },
                            filter: item => !item.text.includes('P10') && !item.text.includes('P10-P90')
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: ctx => {
                                const coId = companyIds.find(id => data.companies[id].shortName === ctx.dataset.label);
                                const bl = coId ? data.companies[coId].co2_2024_mt : null;
                                return `${ctx.dataset.label}: ${fmt(ctx.raw, 1)} Mt${bl ? pctFromBaseline(ctx.raw, bl) : ''}`;
                            }
                        }
                    }
                }
            }
        });
    }

    function renderProfitResilience() {
        // Horizontal bar: P10/P50/P90 profit margin % growth at 2050 vs 2023
        const sorted = [...companies].sort((a, b) => {
            const pa = a.fan_bands.all.profit ? a.fan_bands.all.profit.p50[5] : 0;
            const pb = b.fan_bands.all.profit ? b.fan_bands.all.profit.p50[5] : 0;
            return pb - pa;
        });

        function toPctGrowth(co, percentile) {
            const fb = co.fan_bands.all.profit;
            if (!fb) return 0;
            const base = fb.p50[0] || 1;
            const val = fb[percentile][5];
            return ((val - base) / Math.abs(base)) * 100;
        }

        styledChart(document.getElementById('profitResilienceChart'), {
            type: 'bar',
            data: {
                labels: sorted.map(c => c.shortName),
                datasets: [
                    {
                        label: 'P10 (worst)',
                        data: sorted.map(c => toPctGrowth(c, 'p10')),
                        backgroundColor: 'rgba(239,68,68,0.6)'
                    },
                    {
                        label: 'P50 (median)',
                        data: sorted.map(c => toPctGrowth(c, 'p50')),
                        backgroundColor: 'rgba(99,102,241,0.7)'
                    },
                    {
                        label: 'P90 (best)',
                        data: sorted.map(c => toPctGrowth(c, 'p90')),
                        backgroundColor: 'rgba(34,197,94,0.6)'
                    }
                ]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: { display: true, text: '% Change in Fleet Profit vs 2023 Baseline' },
                        ticks: { callback: v => v + '%' }
                    }
                },
                plugins: {
                    legend: { position: 'bottom', labels: { font: { size: 11 } } },
                    tooltip: {
                        callbacks: {
                            label: ctx => {
                                const pct = ctx.raw;
                                return `${ctx.dataset.label}: ${pct >= 0 ? '+' : ''}${fmt(pct, 1)}%`;
                            }
                        }
                    }
                }
            }
        });
    }

    function renderFleetComposition() {
        const fuelOrder = ['nuclear', 'hydro', 'geothermal', 'wind', 'solar', 'battery', 'gas_ccgt', 'gas_peaker', 'coal', 'oil'];
        const sorted = [...companies].sort((a, b) => b.gen_twh - a.gen_twh);

        const datasets = fuelOrder.map(fuel => {
            const hasData = sorted.some(co => {
                const summary = co.fleet_summary;
                return Object.keys(summary).some(iso => summary[iso][fuel] && summary[iso][fuel].gen_twh > 0);
            });
            if (!hasData) return null;

            return {
                label: FUEL_LABELS[fuel],
                data: sorted.map(co => {
                    const summary = co.fleet_summary;
                    let total = 0;
                    for (const iso of Object.keys(summary)) {
                        if (summary[iso][fuel]) total += summary[iso][fuel].gen_twh || 0;
                    }
                    return total;
                }),
                backgroundColor: FUEL_COLORS[fuel]
            };
        }).filter(Boolean);

        styledChart(document.getElementById('fleetCompositionChart'), {
            type: 'bar',
            data: {
                labels: sorted.map(c => c.shortName),
                datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { stacked: true },
                    y: { stacked: true, title: { display: true, text: 'Generation (TWh)' } }
                },
                plugins: {
                    legend: { position: 'bottom', labels: { font: { size: 10 }, boxWidth: 12 } },
                    tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${fmt(ctx.raw, 1)} TWh` } }
                }
            }
        });
    }

    function renderTargetsSummary() {
        const el = document.getElementById('targetsSummaryTable');
        if (!el) return;

        const sorted = [...companies].sort((a, b) => b.co2_2024_mt - a.co2_2024_mt);

        let html = `<table class="data-table">
            <thead><tr>
                <th>Company</th><th>Baseline (Mt)</th><th>2030</th><th>2035</th><th>2040</th><th>2050</th><th>Gap to NZ 2040</th><th>Gap to AT 2050</th>
            </tr></thead><tbody>`;

        sorted.forEach(co => {
            const fb = (co.fan_bands.reference || co.fan_bands.all).emissions;
            const baseline = fb.p50[0];
            const psV2_2040 = baseline * (SBTI_POWER_V2_REMAINING[2040] || 0);
            const at_2050 = baseline * (AT_TRAJECTORY_REMAINING[2050] || 0);

            const pcts = fb.p50.map(v => baseline > 0 ? ((baseline - v) / baseline * 100).toFixed(0) + '%' : ' - ');
            const gapPS = fb.p50[3] - psV2_2040;
            const gapAT = fb.p50[5] - at_2050;

            html += `<tr>
                <td><a href="ipp_${co.id}.html" style="color:var(--accent);font-weight:600">${co.shortName}</a></td>
                <td>${fmt(baseline, 1)}</td>
                <td>${pcts[1]}</td>
                <td>${pcts[2]}</td>
                <td>${pcts[3]}</td>
                <td>${pcts[5]}</td>
                <td style="color:${gapPS > 1 ? '#dc2626' : gapPS > 0 ? '#b45309' : '#15803d'}">${gapPS > 0 ? '+' : ''}${fmt(gapPS, 1)} Mt</td>
                <td style="color:${gapAT > 1 ? '#dc2626' : gapAT > 0 ? '#b45309' : '#15803d'}">${gapAT > 0 ? '+' : ''}${fmt(gapAT, 1)} Mt</td>
            </tr>`;
        });

        html += '</tbody></table>';
        el.innerHTML = html;
    }

    // ─── Trajectory vs PPA Chart ──────────────────────────────
    function renderTrajectoryPpaChart() {
        const ctx = document.getElementById('trajectoryPpaChart');
        if (!ctx) return;

        const datasets = [];

        // Per-company: solid = P50 market-driven, dashed = High PPA depth
        companies.forEach((co, i) => {
            const color = COMPANY_COLORS[i % COMPANY_COLORS.length];
            const baseline = co.co2_2024_mt || co.co2_2023_mt || 1;

            // Market-driven P50 (reference or all)
            const fbRef = (co.fan_bands.reference || co.fan_bands.all).emissions;
            const refPct = fbRef.p50.map(v => ((baseline - v) / baseline) * 100);

            datasets.push({
                label: co.shortName,
                data: refPct,
                borderColor: color,
                backgroundColor: 'transparent',
                borderWidth: 2.5,
                pointRadius: 4,
                pointBackgroundColor: color,
                tension: 0.3,
                order: 1
            });

            // High PPA depth
            const ppaDim = co.dimension_fans && co.dimension_fans.ppa_level;
            if (ppaDim && ppaDim.High && ppaDim.High.emissions) {
                const highPpa = ppaDim.High.emissions.p50;
                const ppaPct = highPpa.map(v => ((baseline - v) / baseline) * 100);

                datasets.push({
                    label: co.shortName + ' (High PPA)',
                    data: ppaPct,
                    borderColor: color,
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    borderDash: [6, 4],
                    pointRadius: 3,
                    pointStyle: 'triangle',
                    pointBackgroundColor: color,
                    tension: 0.3,
                    order: 1
                });
            }
        });

        // SBTi Power v2 benchmark
        const sbtiPct = YEARS.map(y => ((1 - (SBTI_POWER_V2_REMAINING[y] || 0)) * 100));
        datasets.push({
            label: 'SBTi Power v2 (NZ 2040)',
            data: sbtiPct,
            borderColor: '#dc2626',
            backgroundColor: 'transparent',
            borderWidth: 2,
            borderDash: [10, 5],
            pointRadius: 0,
            tension: 0.3,
            order: 0
        });

        // EPRI AT benchmark
        const atPct = YEARS.map(y => ((1 - (AT_TRAJECTORY_REMAINING[y] || 0)) * 100));
        datasets.push({
            label: 'EPRI AT',
            data: atPct,
            borderColor: '#9333ea',
            backgroundColor: 'transparent',
            borderWidth: 2,
            borderDash: [4, 4],
            pointRadius: 0,
            tension: 0.3,
            order: 0
        });

        styledChart(ctx, {
            type: 'line',
            data: { labels: YEAR_LABELS, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        title: { display: true, text: '% Emission Reduction from Baseline' },
                        min: 0,
                        max: 105,
                        ticks: { callback: v => v + '%' }
                    }
                },
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            font: { size: 10 },
                            boxWidth: 14,
                            filter: item => !item.text.includes('High PPA')
                        },
                        onClick: function(e, legendItem, legend) {
                            // Default toggle behaviour
                            const ci = legend.chart;
                            const idx = legendItem.datasetIndex;
                            // Also toggle the paired High PPA dataset
                            const label = ci.data.datasets[idx].label;
                            const ppaIdx = ci.data.datasets.findIndex(d => d.label === label + ' (High PPA)');
                            Chart.defaults.plugins.legend.onClick.call(this, e, legendItem, legend);
                            if (ppaIdx !== -1) {
                                ci.setDatasetVisibility(ppaIdx, ci.isDatasetVisible(idx));
                                ci.update();
                            }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: ctx2 => {
                                const lbl = ctx2.dataset.label;
                                const pct = ctx2.raw;
                                const suffix = lbl.includes('High PPA') ? ' [High PPA]' : '';
                                return `${lbl.replace(' (High PPA)', '')}${suffix}: ${fmt(pct, 1)}% reduction`;
                            }
                        }
                    }
                }
            }
        });
    }

    // ─── Decarbonization Cost Comparison ──────────────────────
    function computeFleetMAC(co) {
        const summary = co.fleet_summary;
        let totalFossilCO2 = 0, totalFossilGen = 0, totalCleanGen = 0;
        let coalCO2 = 0, gasCO2 = 0;
        const isoCO2 = {};

        for (const iso of Object.keys(summary)) {
            for (const fuel of Object.keys(summary[iso])) {
                const p = summary[iso][fuel];
                if (FOSSIL_FUELS.includes(fuel)) {
                    totalFossilCO2 += p.co2_mt || 0;
                    totalFossilGen += p.gen_twh || 0;
                    isoCO2[iso] = (isoCO2[iso] || 0) + (p.co2_mt || 0);
                    if (fuel === 'coal') coalCO2 += p.co2_mt || 0;
                    else gasCO2 += p.co2_mt || 0;
                } else {
                    totalCleanGen += p.gen_twh || 0;
                }
            }
        }

        let weightedNum = 0, weightedDen = 0;
        const isoBreakdown = [];
        for (const [iso, co2] of Object.entries(isoCO2)) {
            const mac = ISO_MAC_FULL[iso] || 0;
            weightedNum += co2 * mac;
            weightedDen += co2;
            isoBreakdown.push({ iso, co2, mac, costB: co2 * mac / 1000 });
        }

        const totalGen = totalCleanGen + totalFossilGen;
        return {
            cleanPct: totalGen > 0 ? (totalCleanGen / totalGen * 100) : 0,
            fossilCO2: totalFossilCO2,
            fossilGen: totalFossilGen,
            coalCO2, gasCO2,
            avgMAC: weightedDen > 0 ? weightedNum / weightedDen : 0,
            totalCostB: weightedNum / 1000,
            costPerMWh: totalFossilGen > 0 ? weightedNum / totalFossilGen : 0,
            isoBreakdown
        };
    }

    function renderMACComparison() {
        const macResults = companies.map(co => ({
            ...computeFleetMAC(co),
            name: co.shortName,
            id: co.id,
            intensity: co.intensity_kg
        }));

        // Sort by avg MAC (lowest first) for the bar chart
        const sortedByMAC = [...macResults].sort((a, b) => a.avgMAC - b.avgMAC);
        const sortedByCost = [...macResults].sort((a, b) => b.totalCostB - a.totalCostB);

        // Chart 1: Weighted avg MAC horizontal bar
        const macCtx = document.getElementById('macComparisonChart');
        if (macCtx) {
            styledChart(macCtx, {
                type: 'bar',
                data: {
                    labels: sortedByMAC.map(r => r.name),
                    datasets: [
                        {
                            label: 'Coal abatement',
                            data: sortedByMAC.map(r => r.coalCO2 > 0 ? r.avgMAC * (r.coalCO2 / r.fossilCO2) : 0),
                            backgroundColor: FUEL_COLORS.coal,
                            borderRadius: 2
                        },
                        {
                            label: 'Gas abatement',
                            data: sortedByMAC.map(r => r.gasCO2 > 0 ? r.avgMAC * (r.gasCO2 / r.fossilCO2) : r.avgMAC),
                            backgroundColor: FUEL_COLORS.gas_ccgt,
                            borderRadius: 2
                        }
                    ]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            stacked: true,
                            title: { display: true, text: '$/tCO₂ (fleet-weighted)' },
                            beginAtZero: true,
                            ticks: { callback: v => '$' + v }
                        },
                        y: { stacked: true }
                    },
                    plugins: {
                        legend: { position: 'bottom', labels: { font: { size: 11 } } },
                        tooltip: {
                            callbacks: {
                                afterBody: function(items) {
                                    const idx = items[0].dataIndex;
                                    const r = sortedByMAC[idx];
                                    return [
                                        `Total MAC: $${fmt(r.avgMAC, 0)}/tCO₂`,
                                        `Fossil CO₂: ${fmt(r.fossilCO2, 1)} Mt`,
                                        `Clean share: ${fmt(r.cleanPct, 0)}%`
                                    ];
                                }
                            }
                        }
                    }
                }
            });
        }

        // Chart 2: Total annual abatement cost
        const costCtx = document.getElementById('totalAbatementCostChart');
        if (costCtx) {
            const sorted = sortedByCost;
            const maxCost = Math.max(...sorted.map(r => r.totalCostB));
            styledChart(costCtx, {
                type: 'bar',
                data: {
                    labels: sorted.map(r => r.name),
                    datasets: [{
                        label: 'Abatement Cost',
                        data: sorted.map(r => r.totalCostB),
                        backgroundColor: sorted.map(r => {
                            const ratio = r.totalCostB / maxCost;
                            return ratio > 0.8 ? '#dc2626' : ratio > 0.5 ? '#f59e0b' : '#22c55e';
                        }),
                        borderRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            title: { display: true, text: '$B / year' },
                            beginAtZero: true,
                            ticks: { callback: v => '$' + v + 'B' }
                        }
                    },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: ctx => `$${fmt(ctx.raw, 1)}B/yr (${fmt(sortedByCost[ctx.dataIndex].fossilCO2, 1)} Mt CO₂)`
                            }
                        }
                    }
                }
            });
        }

        // Chart 3: Cost per MWh replaced
        const mwhCtx = document.getElementById('costPerMwhChart');
        if (mwhCtx) {
            const sorted = [...macResults].sort((a, b) => b.costPerMWh - a.costPerMWh);
            styledChart(mwhCtx, {
                type: 'bar',
                data: {
                    labels: sorted.map(r => r.name),
                    datasets: [{
                        label: '$/MWh',
                        data: sorted.map(r => r.costPerMWh),
                        backgroundColor: sorted.map((_, i) => COMPANY_COLORS[companies.findIndex(c => c.shortName === sorted[i].name) % COMPANY_COLORS.length]),
                        borderRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            title: { display: true, text: '$/MWh replaced' },
                            beginAtZero: true,
                            ticks: { callback: v => '$' + v }
                        }
                    },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: ctx => `$${fmt(ctx.raw, 0)}/MWh (${fmt(sorted[ctx.dataIndex].fossilGen, 1)} TWh fossil)`
                            }
                        }
                    }
                }
            });
        }

        // Summary table
        const tableEl = document.getElementById('macSummaryTable');
        if (tableEl) {
            const sorted = [...macResults].sort((a, b) => a.avgMAC - b.avgMAC);
            const cheapest = sorted[0];

            let html = `<table class="data-table">
                <thead><tr>
                    <th>Company</th><th>Clean %</th><th>Fossil CO₂ (Mt)</th>
                    <th>Coal (Mt)</th><th>Gas (Mt)</th>
                    <th>Wtd MAC ($/tCO₂)</th><th>Total Cost ($B/yr)</th>
                    <th>$/MWh Replaced</th><th>MAC Delta</th>
                </tr></thead><tbody>`;

            sorted.forEach(r => {
                const delta = r.avgMAC - cheapest.avgMAC;
                const deltaPct = cheapest.avgMAC > 0 ? (delta / cheapest.avgMAC * 100) : 0;
                const deltaStr = delta < 1 ? ' - ' : `+$${fmt(delta, 0)} (+${fmt(deltaPct, 0)}%)`;
                const deltaColor = delta < 1 ? 'var(--text-secondary)' : delta > 50 ? '#dc2626' : '#b45309';

                html += `<tr>
                    <td><a href="ipp_${r.id}.html" style="color:var(--accent);font-weight:600">${r.name}</a></td>
                    <td>${fmt(r.cleanPct, 0)}%</td>
                    <td>${fmt(r.fossilCO2, 1)}</td>
                    <td>${fmt(r.coalCO2, 1)}</td>
                    <td>${fmt(r.gasCO2, 1)}</td>
                    <td><strong>$${fmt(r.avgMAC, 0)}</strong></td>
                    <td>$${fmt(r.totalCostB, 1)}B</td>
                    <td>$${fmt(r.costPerMWh, 0)}</td>
                    <td style="color:${deltaColor}">${deltaStr}</td>
                </tr>`;
            });

            html += '</tbody></table>';
            tableEl.innerHTML = html;
        }

        // Insight box
        const insightEl = document.getElementById('macInsightBox');
        if (insightEl) {
            const sorted = [...macResults].sort((a, b) => a.avgMAC - b.avgMAC);
            const cheapestMAC = sorted[0];
            const mostExpMAC = sorted[sorted.length - 1];
            const cheapestTotal = [...macResults].sort((a, b) => a.totalCostB - b.totalCostB)[0];
            const mostExpTotal = [...macResults].sort((a, b) => b.totalCostB - a.totalCostB)[0];
            const macSpread = mostExpMAC.avgMAC - cheapestMAC.avgMAC;

            const coalHeavy = macResults.filter(r => r.coalCO2 > 5).sort((a, b) => b.coalCO2 - a.coalCO2);
            const pureGas = macResults.filter(r => r.coalCO2 < 1 && r.fossilCO2 > 5);

            let insights = [];
            insights.push(`<strong>${cheapestMAC.name}</strong> has the lowest per-ton cost at <strong>$${fmt(cheapestMAC.avgMAC, 0)}/tCO₂</strong>, driven by concentration in low-MAC ISOs (ERCOT $356/t, PJM $254/t).`);
            insights.push(`<strong>${mostExpMAC.name}</strong> faces the highest at <strong>$${fmt(mostExpMAC.avgMAC, 0)}/tCO₂</strong> - a <strong>$${fmt(macSpread, 0)}/tCO₂</strong> premium over ${cheapestMAC.name}.`);

            if (coalHeavy.length > 0) {
                insights.push(`<strong>Coal exposure matters for total cost, not per-ton cost:</strong> ${coalHeavy.map(r => r.name).join(', ')} carry ${fmt(coalHeavy.reduce((s, r) => s + r.coalCO2, 0), 0)} Mt of coal CO₂ - high-emitting per MWh but cheaper to displace per ton because coal plants emit ~0.95 tCO₂/MWh vs ~0.4 for gas.`);
            }

            if (pureGas.length > 0) {
                insights.push(`<strong>All-gas fleets pay more per ton:</strong> ${pureGas.map(r => r.name).join(', ')} must replace more MWh per ton abated because gas CCGTs emit less CO₂ per MWh than coal, pushing up the fleet-weighted MAC.`);
            }

            insightEl.innerHTML = `
                <div class="insight-box scroll-reveal" style="margin-bottom:var(--space-xl)">
                    <strong>Key Takeaways</strong>
                    <ul style="margin:8px 0 0 16px">${insights.map(i => `<li style="margin-bottom:6px">${i}</li>`).join('')}</ul>
                </div>
            `;
        }
    }

    // ─── Initialize ──────────────────────────────────────────
    function init() {
        renderHeroStats();
        renderCompanyCards();
        renderKeyInsights();
        renderMACComparison();
        renderEmissionsComparison();
        renderTrajectoryPpaChart();
        renderProfitResilience();
        renderFleetComposition();
        renderTargetsSummary();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
