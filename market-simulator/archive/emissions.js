/**
 * CCS Emissions Dashboard — emissions.js
 *
 * Renders CCS analysis results from the constellation dispatch endpoint.
 * Uses Plotly for fan charts and bar charts.
 */

(function () {
    'use strict';

    // ── State ──
    let dispatchData = null;
    let lastSimResults = null;
    let selectedPlantIdx = 0;

    // ── DOM refs ──
    const btnRun = document.getElementById('btnRunDispatch');
    const btnCSV = document.getElementById('btnDownloadDispatchCSV');
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');
    const statusDetail = document.getElementById('statusDetail');

    // ── Init ──
    function init() {
        // Check if simulation results exist in sessionStorage
        const stored = sessionStorage.getItem('simulationResult');
        if (stored) {
            try {
                lastSimResults = JSON.parse(stored);
                enableDispatch();
            } catch (e) {
                console.warn('Could not parse stored sim results:', e);
            }
        }

        btnRun.addEventListener('click', runDispatch);
        btnCSV.addEventListener('click', downloadCSV);

        // Check if we already have dispatch results cached
        const cachedDispatch = sessionStorage.getItem('constellationDispatch');
        if (cachedDispatch) {
            try {
                dispatchData = JSON.parse(cachedDispatch);
                renderAll();
                setStatus('ready', 'CCS dispatch results loaded',
                    `${dispatchData.plants?.length || 0} plants, ${dispatchData.year_labels?.length || 0} years`);
            } catch (e) { /* ignore */ }
        }
    }

    function enableDispatch() {
        btnRun.disabled = false;
        setStatus('ready', 'Simulation results available',
            `ISO: ${lastSimResults?.iso || '?'} — Click "Run CCS Dispatch" to analyze fleet`);
    }

    function setStatus(state, text, detail) {
        statusDot.className = 'status-dot ' + state;
        statusText.textContent = text;
        statusDetail.textContent = detail || '';
    }

    // ── Run dispatch ──
    async function runDispatch() {
        if (!lastSimResults) {
            setStatus('error', 'No simulation results', 'Run a simulation first from the Setup page.');
            return;
        }

        btnRun.disabled = true;
        setStatus('pending', 'Running CCS dispatch...', 'Dispatching Constellation CCGT fleet through merit order');

        try {
            // Extract year_results from the simulation response
            const yearResults = lastSimResults.year_results || [];
            if (!yearResults.length) {
                setStatus('error', 'No year results', 'Simulation must be in trajectory mode for CCS analysis.');
                btnRun.disabled = false;
                return;
            }

            // Get fleet overrides from localStorage
            let fleetOverrides = null;
            const saved = localStorage.getItem('fleet_overrides');
            if (saved) {
                try { fleetOverrides = JSON.parse(saved); } catch (e) { /* ignore */ }
            }

            const resp = await fetch('/api/constellation-dispatch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    year_results: yearResults,
                    fleet_overrides: fleetOverrides,
                    run_id: lastSimResults.run_id || null,
                }),
            });

            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}));
                throw new Error(err.detail || `HTTP ${resp.status}`);
            }

            dispatchData = await resp.json();
            sessionStorage.setItem('constellationDispatch', JSON.stringify(dispatchData));

            renderAll();
            setStatus('ready', 'CCS dispatch complete',
                `${dispatchData.plants?.length || 0} plants dispatched across ${dispatchData.year_labels?.length || 0} years`);
            btnCSV.style.display = '';

        } catch (err) {
            setStatus('error', 'Dispatch failed', err.message);
            console.error('CCS dispatch error:', err);
        } finally {
            btnRun.disabled = false;
        }
    }

    // ── Download CSV ──
    function downloadCSV() {
        if (!dispatchData?.csv_rows?.length) return;
        const runId = lastSimResults?.run_id;
        if (runId) {
            window.open(`/api/download/${runId}/constellation_dispatch.csv`, '_blank');
        } else {
            // Generate client-side CSV
            const rows = dispatchData.csv_rows;
            const cols = Object.keys(rows[0]);
            let csv = cols.join(',') + '\n';
            for (const row of rows) {
                csv += cols.map(c => row[c] ?? '').join(',') + '\n';
            }
            const blob = new Blob([csv], { type: 'text/csv' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'constellation_dispatch.csv';
            a.click();
            URL.revokeObjectURL(url);
        }
    }

    // ── Render all charts ──
    function renderAll() {
        if (!dispatchData) return;

        // Show sections
        document.getElementById('ccsKpiGrid').style.display = '';
        document.getElementById('fanChartCard').style.display = '';
        document.getElementById('deltaChartCard').style.display = '';
        document.getElementById('plantDetailCard').style.display = '';
        document.getElementById('fleetTableCard').style.display = '';

        renderKPIs();
        renderFanChart();
        renderDeltaChart();
        renderPlantSelector();
        renderPlantDetail(0);
        renderFleetTable();
    }

    // ── KPIs ──
    function renderKPIs() {
        const d = dispatchData;
        document.getElementById('kpiBaseline').textContent = (d.baseline_mmt || 0).toFixed(1);

        // Sum last year P50 dispatch emissions
        const p50 = d.fan_bands?.p50 || [];
        const lastDispatch = p50[p50.length - 1] || 0;
        document.getElementById('kpiDispatch').textContent = lastDispatch.toFixed(2);

        // CCS delta = sum of all plant CCS deltas
        const plants = d.plants || [];
        const totalCCSDelta = plants.reduce((sum, p) => {
            return sum + (p.baseline_co2_mmt || 0) - (p.ccs_co2_mmt || 0);
        }, 0);
        document.getElementById('kpiCCSDelta').textContent = totalCCSDelta.toFixed(2);

        document.getElementById('kpiPlants').textContent = plants.length;
    }

    // ── Fan Chart ──
    function renderFanChart() {
        const d = dispatchData;
        const years = (d.year_labels || []).map(Number);
        const fb = d.fan_bands || {};
        const traj = d.trajectories || {};

        const traces = [];

        // P10-P90 band
        if (fb.p10 && fb.p90) {
            traces.push({
                x: years.concat([...years].reverse()),
                y: fb.p90.concat([...fb.p10].reverse()),
                fill: 'toself',
                fillcolor: 'rgba(35,114,185,0.15)',
                line: { color: 'transparent' },
                name: 'P10-P90 Range',
                showlegend: false,
                hoverinfo: 'skip',
            });
        }

        // P50 line
        if (fb.p50) {
            traces.push({
                x: years, y: fb.p50,
                mode: 'lines+markers',
                line: { color: '#2372B9', width: 3 },
                marker: { size: 6 },
                name: 'Fleet Emissions (P50)',
            });
        }

        // AT trajectory
        if (traj.at_power_nz?.values) {
            traces.push({
                x: years, y: traj.at_power_nz.values,
                mode: 'lines',
                line: { color: '#DC2626', width: 2, dash: 'dash' },
                name: 'AT Power NZ',
            });
        }

        // SBTi trajectory
        if (traj.sbti_15c?.values) {
            traces.push({
                x: years, y: traj.sbti_15c.values,
                mode: 'lines',
                line: { color: '#F59E0B', width: 2, dash: 'dash' },
                name: 'SBTi 1.5C',
            });
        }

        const layout = {
            xaxis: { title: 'Year', dtick: 5 },
            yaxis: { title: 'Fleet CO2 (MMt)', rangemode: 'tozero' },
            margin: { t: 20, r: 30, b: 50, l: 60 },
            legend: { orientation: 'h', y: -0.2 },
            hovermode: 'x unified',
        };

        Plotly.newPlot('chartFleetFan', traces, layout, { responsive: true });
    }

    // ── CCS Delta Bar Chart ──
    function renderDeltaChart() {
        const plants = dispatchData.plants || [];
        const sorted = [...plants].sort((a, b) =>
            (b.baseline_co2_mmt - b.ccs_co2_mmt) - (a.baseline_co2_mmt - a.ccs_co2_mmt)
        );

        const names = sorted.map(p => p.name || `Plant ${p.orispl}`);
        const deltas = sorted.map(p => Math.max(0, (p.baseline_co2_mmt || 0) - (p.ccs_co2_mmt || 0)));

        const traces = [{
            x: deltas, y: names,
            type: 'bar',
            orientation: 'h',
            marker: { color: '#22C55E' },
            text: deltas.map(d => d.toFixed(3)),
            textposition: 'outside',
            hovertemplate: '%{y}<br>CCS Delta: %{x:.3f} MMt CO2<extra></extra>',
        }];

        const layout = {
            xaxis: { title: 'CCS Reduction Potential (MMt CO2/yr)' },
            yaxis: { automargin: true },
            margin: { t: 10, r: 30, b: 50, l: 200 },
            height: Math.max(400, plants.length * 28),
        };

        Plotly.newPlot('chartCCSDelta', traces, layout, { responsive: true });
    }

    // ── Plant Selector ──
    function renderPlantSelector() {
        const container = document.getElementById('plantSelector');
        container.innerHTML = '';
        const plants = dispatchData.plants || [];

        plants.forEach((p, idx) => {
            const chip = document.createElement('div');
            chip.className = 'plant-chip' + (idx === selectedPlantIdx ? ' selected' : '');
            chip.textContent = p.name || `Plant ${p.orispl}`;
            chip.addEventListener('click', () => {
                selectedPlantIdx = idx;
                container.querySelectorAll('.plant-chip').forEach(c => c.classList.remove('selected'));
                chip.classList.add('selected');
                renderPlantDetail(idx);
            });
            container.appendChild(chip);
        });
    }

    // ── Plant Detail Chart ──
    function renderPlantDetail(idx) {
        const plants = dispatchData.plants || [];
        if (idx >= plants.length) return;

        const plant = plants[idx];
        const years = (dispatchData.year_labels || []).map(Number);
        const p50 = plant.sweep_p50 || [];

        const traces = [];

        // Dispatch emissions
        traces.push({
            x: years, y: p50,
            mode: 'lines+markers',
            line: { color: '#2372B9', width: 2 },
            name: 'Dispatch Emissions',
            fill: 'tozeroy',
            fillcolor: 'rgba(35,114,185,0.1)',
        });

        // CCS residual (flat line)
        if (plant.ccs_co2_mmt != null) {
            traces.push({
                x: years, y: years.map(() => plant.ccs_co2_mmt),
                mode: 'lines',
                line: { color: '#22C55E', width: 2, dash: 'dot' },
                name: 'CCS Residual',
            });
        }

        // Baseline (flat line)
        if (plant.baseline_co2_mmt) {
            traces.push({
                x: years, y: years.map(() => plant.baseline_co2_mmt),
                mode: 'lines',
                line: { color: '#DC2626', width: 1, dash: 'dash' },
                name: 'Baseline',
            });
        }

        const layout = {
            title: { text: `${plant.name || 'Plant ' + plant.orispl} (${plant.iso}, ${plant.capacity_mw} MW)`, font: { size: 14 } },
            xaxis: { title: 'Year', dtick: 5 },
            yaxis: { title: 'CO2 (MMt)', rangemode: 'tozero' },
            margin: { t: 40, r: 30, b: 50, l: 60 },
            legend: { orientation: 'h', y: -0.2 },
            hovermode: 'x unified',
        };

        Plotly.newPlot('chartPlantDetail', traces, layout, { responsive: true });
    }

    // ── Fleet Table ──
    function renderFleetTable() {
        const tbody = document.getElementById('fleetTableBody');
        tbody.innerHTML = '';
        const plants = dispatchData.plants || [];

        for (const p of plants) {
            const delta = (p.baseline_co2_mmt || 0) - (p.ccs_co2_mmt || 0);
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${p.name || p.orispl}</td>
                <td>${p.iso || ''}</td>
                <td>${(p.capacity_mw || 0).toLocaleString()}</td>
                <td>${(p.baseline_co2_mmt || 0).toFixed(4)}</td>
                <td>${(p.ccs_co2_mmt || 0).toFixed(4)}</td>
                <td style="color: ${delta > 0 ? '#166534' : '#991b1b'}; font-weight: 600;">${delta.toFixed(4)}</td>
            `;
            tbody.appendChild(tr);
        }
    }

    // ── Boot ──
    document.addEventListener('DOMContentLoaded', init);
})();
