/**
 * IPP Fleet Report — ipp-report.js
 *
 * Renders Constellation/Calpine fleet intelligence from CCS dispatch results.
 * Reads cached dispatch data from sessionStorage (set by emissions.js).
 */

(function () {
    'use strict';

    let dispatchData = null;

    function init() {
        const cached = sessionStorage.getItem('constellationDispatch');
        if (cached) {
            try {
                dispatchData = JSON.parse(cached);
                renderReport();
            } catch (e) {
                setStatus('error', 'Failed to load dispatch data', e.message);
            }
        }
    }

    function setStatus(state, text, detail) {
        const dot = document.getElementById('reportStatusDot');
        const txt = document.getElementById('reportStatusText');
        const det = document.getElementById('reportStatusDetail');
        if (dot) {
            dot.style.background = state === 'ready' ? '#22C55E' : state === 'error' ? '#DC2626' : '#F59E0B';
        }
        if (txt) txt.textContent = text;
        if (det) det.textContent = detail || '';
    }

    function renderReport() {
        if (!dispatchData) return;

        setStatus('ready', 'Fleet report loaded',
            `${dispatchData.plants?.length || 0} plants, baseline ${(dispatchData.baseline_mmt || 0).toFixed(1)} MMt CO2`);

        // Show all sections
        ['sec1Header', 'fleetKpiGrid', 'isoCapCard', 'emissionsCard', 'timelineCard', 'rankingCard']
            .forEach(id => {
                const el = document.getElementById(id);
                if (el) el.style.display = '';
            });

        // Hide status bar
        const statusBar = document.getElementById('reportStatus');
        if (statusBar) statusBar.style.display = 'none';

        renderKPIs();
        renderISOCapChart();
        renderEmissionsChart();
        renderTimeline();
        renderRankingTable();
    }

    // ── KPIs ──
    function renderKPIs() {
        const plants = dispatchData.plants || [];
        const totalCap = plants.reduce((s, p) => s + (p.capacity_mw || 0), 0);
        const baseline = dispatchData.baseline_mmt || 0;
        const ccsPotential = plants.reduce((s, p) => {
            return s + (p.baseline_co2_mmt || 0) - (p.ccs_co2_mmt || 0);
        }, 0);

        document.getElementById('rptTotalCap').textContent = Math.round(totalCap).toLocaleString();
        document.getElementById('rptTotalPlants').textContent = plants.length;
        document.getElementById('rptBaselineEmissions').textContent = baseline.toFixed(1);
        document.getElementById('rptCCSPotential').textContent = ccsPotential.toFixed(2);
    }

    // ── ISO Capacity Pie/Bar ──
    function renderISOCapChart() {
        const plants = dispatchData.plants || [];
        const isoMap = {};
        for (const p of plants) {
            const iso = p.iso || 'Unknown';
            isoMap[iso] = (isoMap[iso] || 0) + (p.capacity_mw || 0);
        }

        const isos = Object.keys(isoMap).sort((a, b) => isoMap[b] - isoMap[a]);
        const values = isos.map(iso => isoMap[iso]);

        const colors = isos.map(iso => {
            const map = { CAISO: '#F59E0B', ERCOT: '#22C55E', PJM: '#0EA5E9', NYISO: '#E91E63', NEISO: '#9C27B0', MISO: '#F97316', SPP: '#14B8A6' };
            return map[iso] || '#6B7280';
        });

        Plotly.newPlot('chartISOCap', [{
            labels: isos,
            values: values,
            type: 'pie',
            marker: { colors: colors },
            textinfo: 'label+percent',
            textposition: 'outside',
            hovertemplate: '%{label}<br>%{value:,.0f} MW<br>%{percent}<extra></extra>',
        }], {
            margin: { t: 20, r: 20, b: 20, l: 20 },
            showlegend: false,
        }, { responsive: true });
    }

    // ── Emissions Trajectory ──
    function renderEmissionsChart() {
        const years = (dispatchData.year_labels || []).map(Number);
        const fb = dispatchData.fan_bands || {};
        const plants = dispatchData.plants || [];

        // Compute fleet CCS residual total (flat across years since CCS parameters are static)
        const totalCCSResidual = plants.reduce((s, p) => s + (p.ccs_co2_mmt || 0), 0);

        const traces = [];

        // Dispatch emissions
        if (fb.p50) {
            traces.push({
                x: years, y: fb.p50,
                mode: 'lines+markers',
                line: { color: '#2372B9', width: 3 },
                name: 'Dispatch Emissions',
            });
        }

        // CCS residual line
        traces.push({
            x: years, y: years.map(() => totalCCSResidual),
            mode: 'lines',
            line: { color: '#22C55E', width: 2, dash: 'dot' },
            name: 'CCS Residual (all plants)',
        });

        // Baseline
        traces.push({
            x: years, y: years.map(() => dispatchData.baseline_mmt || 0),
            mode: 'lines',
            line: { color: '#DC2626', width: 1, dash: 'dash' },
            name: 'Baseline',
        });

        // AT/SBTi trajectories
        const traj = dispatchData.trajectories || {};
        if (traj.at_power_nz?.values) {
            traces.push({
                x: years, y: traj.at_power_nz.values,
                mode: 'lines',
                line: { color: '#DC2626', width: 1.5, dash: 'dashdot' },
                name: 'AT Power NZ',
            });
        }
        if (traj.sbti_15c?.values) {
            traces.push({
                x: years, y: traj.sbti_15c.values,
                mode: 'lines',
                line: { color: '#F59E0B', width: 1.5, dash: 'dashdot' },
                name: 'SBTi 1.5C',
            });
        }

        Plotly.newPlot('chartFleetEmissions', traces, {
            xaxis: { title: 'Year', dtick: 5 },
            yaxis: { title: 'CO2 (MMt)', rangemode: 'tozero' },
            margin: { t: 20, r: 30, b: 50, l: 60 },
            legend: { orientation: 'h', y: -0.2 },
            hovermode: 'x unified',
        }, { responsive: true });
    }

    // ── Transition Timeline ──
    function renderTimeline() {
        const container = document.getElementById('transitionTimeline');
        container.innerHTML = '';
        const years = (dispatchData.year_labels || []).map(Number);
        const fb = dispatchData.fan_bands || {};
        const baseline = dispatchData.baseline_mmt || 0;

        if (!fb.p50 || !years.length) return;

        // Find key milestones
        const milestones = [];

        // First year where emissions drop below 75% of baseline
        for (let i = 0; i < years.length; i++) {
            if (fb.p50[i] < baseline * 0.75) {
                milestones.push({ year: years[i], type: 'transition', desc: `Fleet emissions drop below 75% of baseline (${fb.p50[i].toFixed(2)} MMt)` });
                break;
            }
        }

        // First year where emissions drop below 50%
        for (let i = 0; i < years.length; i++) {
            if (fb.p50[i] < baseline * 0.50) {
                milestones.push({ year: years[i], type: 'ccs', desc: `Fleet emissions at 50% of baseline (${fb.p50[i].toFixed(2)} MMt) — CCS retrofit window` });
                break;
            }
        }

        // First year where emissions drop below 25%
        for (let i = 0; i < years.length; i++) {
            if (fb.p50[i] < baseline * 0.25) {
                milestones.push({ year: years[i], type: 'retired', desc: `Fleet emissions below 25% of baseline (${fb.p50[i].toFixed(2)} MMt)` });
                break;
            }
        }

        // Final year
        const lastEmission = fb.p50[fb.p50.length - 1];
        const pctReduction = ((1 - lastEmission / baseline) * 100).toFixed(0);
        milestones.push({
            year: years[years.length - 1],
            type: lastEmission < baseline * 0.1 ? 'ccs' : 'transition',
            desc: `End state: ${lastEmission.toFixed(2)} MMt (${pctReduction}% reduction from baseline)`,
        });

        for (const m of milestones) {
            const div = document.createElement('div');
            div.className = `timeline-item ${m.type}`;
            div.innerHTML = `
                <div class="timeline-year">${m.year}</div>
                <div class="timeline-desc">${m.desc}</div>
            `;
            container.appendChild(div);
        }
    }

    // ── Ranking Table ──
    function renderRankingTable() {
        const tbody = document.getElementById('rankingTableBody');
        tbody.innerHTML = '';
        const plants = dispatchData.plants || [];

        const ranked = [...plants].map(p => ({
            ...p,
            delta: (p.baseline_co2_mmt || 0) - (p.ccs_co2_mmt || 0),
        })).sort((a, b) => b.delta - a.delta);

        const maxDelta = ranked[0]?.delta || 1;

        ranked.forEach((p, idx) => {
            const score = Math.round((p.delta / maxDelta) * 100);
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td style="text-align: center; font-weight: 700;">${idx + 1}</td>
                <td>${p.name || p.orispl}</td>
                <td>${p.iso || ''}</td>
                <td>${(p.capacity_mw || 0).toLocaleString()}</td>
                <td>${(p.baseline_co2_mmt || 0).toFixed(4)}</td>
                <td>${(p.ccs_co2_mmt || 0).toFixed(4)}</td>
                <td style="color: #166534; font-weight: 600;">${p.delta.toFixed(4)}</td>
                <td>
                    <div style="display:flex;align-items:center;gap:0.5rem;">
                        <div style="flex:1;height:6px;background:#e5e7eb;border-radius:3px;overflow:hidden;">
                            <div style="width:${score}%;height:100%;background:#22C55E;border-radius:3px;"></div>
                        </div>
                        <span style="font-size:0.8rem;font-weight:600;">${score}</span>
                    </div>
                </td>
            `;
            tbody.appendChild(tr);
        });
    }

    document.addEventListener('DOMContentLoaded', init);
})();
