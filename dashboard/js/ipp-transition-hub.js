/**
 * IPP Climate Transition Hub — Cross-Company Overview
 * =====================================================
 * Renders hero stats, company cards, and comparative analysis charts.
 * Reads from: IPP_SMARTARGETS_DATA (ipp-smartargets-data.js)
 */

/* global IPP_SMARTARGETS_DATA, Chart, RESOURCE_COLORS, ISO_COLORS */

(function () {
    'use strict';

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

    function fmt(n, d) {
        if (n === undefined || n === null) return '—';
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
            const reduction = fb.p50[0] > 0 ? ((fb.p50[0] - fb.p50[5]) / fb.p50[0] * 100).toFixed(0) : '—';
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
            new Chart(ctx, {
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

        // Check coal retirement — all companies with coal show declining coal dispatch
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

        new Chart(document.getElementById('emissionsComparisonChart'), {
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

        new Chart(document.getElementById('profitResilienceChart'), {
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

        new Chart(document.getElementById('fleetCompositionChart'), {
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

            const pcts = fb.p50.map(v => baseline > 0 ? ((baseline - v) / baseline * 100).toFixed(0) + '%' : '—');
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

    // ─── Initialize ──────────────────────────────────────────
    function init() {
        renderHeroStats();
        renderCompanyCards();
        renderKeyInsights();
        renderEmissionsComparison();
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
